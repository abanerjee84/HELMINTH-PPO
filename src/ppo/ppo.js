import { OBSERVATION_COUNT } from '../config.js';
import { clamp } from '../sim/random.js';
import { Adam } from './optim.js';
import { Policy, gaussianEntropy } from './network.js';

const HALF_LOG_TWO_PI = 0.9189385332046727;

/**
 * Proximal Policy Optimization over a shared actor-critic policy.
 *
 * The update is split into minibatches so the caller can spend a small,
 * bounded slice of frame time per batch instead of blocking the render loop.
 */
export class PPO {
  constructor(config, rng) {
    this.config = config;
    this.settings = config.ppo;
    this.rng = rng;
    this.policy = new Policy({
      inputSize: OBSERVATION_COUNT,
      hiddenSizes: this.settings.hiddenSizes,
      actionSize: 2,
      rng,
      config,
    });
    this.optimizer = new Adam(this.policy.tensors, {
      learningRate: this.settings.learningRate,
    });
    this.actorGrad = new Float32Array(2);
    this.criticGrad = new Float32Array(1);
    this.samples = null;
    this.order = new Int32Array(0);
    this.epoch = 0;
    this.minibatchCursor = 0;
    this.iterations = 0;
    this.accumulator = null;
    this.rewardScale = 1;
    this.returnStd = 1;
    this.scaleReady = false;
    this.stats = {
      policyLoss: 0,
      valueLoss: 0,
      entropy: 0,
      approxKl: 0,
      clipFraction: 0,
      gradientNorm: 0,
      meanReward: 0,
      samples: 0,
      batches: 0,
      learningRate: this.settings.learningRate,
      parameters: this.policy.parameterCount,
      iterations: 0,
    };
  }

  act(observation, rng = this.rng) {
    return this.policy.act(observation, rng);
  }

  value(observation) {
    return this.policy.value(observation);
  }

  /**
   * Keeps the critic's regression targets on a unit scale. The arena pays
   * 180 for a nutrient bloom, so raw returns span several orders of
   * magnitude and would otherwise swamp the policy gradient.
   */
  observeReturns(stats) {
    if (!stats || stats.count === 0) return;
    const rawStd = stats.std / Math.max(this.rewardScale, 1e-6);
    const target = 1 / Math.max(rawStd, 1e-3);
    if (!this.scaleReady) {
      this.rewardScale = target;
      this.scaleReady = true;
    } else {
      this.rewardScale = 0.8 * this.rewardScale + 0.2 * target;
    }
    this.returnStd = rawStd;
  }

  get parameterChecksum() {
    let total = 0;
    for (const tensor of this.policy.tensors) {
      for (let index = 0; index < tensor.params.length; index += 1) {
        total += Math.abs(tensor.params[index]);
      }
    }
    return total;
  }

  beginUpdate(samples) {
    this.samples = samples;
    this.order = new Int32Array(samples.count);
    for (let index = 0; index < samples.count; index += 1) {
      this.order[index] = index;
    }
    this.epoch = 0;
    this.minibatchCursor = 0;
    this.shuffle();
    let rewardSum = 0;
    let advantageSum = 0;
    for (let index = 0; index < samples.count; index += 1) {
      rewardSum += samples.rewards[index];
      advantageSum += samples.advantages[index];
    }
    this.accumulator = {
      policyLoss: 0,
      valueLoss: 0,
      entropy: 0,
      kl: 0,
      clipFraction: 0,
      gradientNorm: 0,
      batches: 0,
      samples: samples.count,
      meanReward: samples.count ? rewardSum / samples.count : 0,
      meanAdvantage: samples.count ? advantageSum / samples.count : 0,
    };
  }

  hasPendingWork() {
    return this.samples !== null && this.epoch < this.settings.epochs;
  }

  shuffle() {
    const order = this.order;
    for (let index = order.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(this.rng() * (index + 1));
      const held = order[index];
      order[index] = order[swap];
      order[swap] = held;
    }
  }

  updateMinibatch() {
    if (!this.hasPendingWork()) return false;
    if (this.minibatchCursor >= this.order.length) {
      this.epoch += 1;
      this.minibatchCursor = 0;
      if (this.epoch >= this.settings.epochs) {
        this.finishUpdate();
        return false;
      }
      this.shuffle();
    }
    const end = Math.min(
      this.order.length,
      this.minibatchCursor + this.settings.minibatchSize,
    );
    this.processBatch(this.order.subarray(this.minibatchCursor, end));
    this.minibatchCursor = end;
    return true;
  }

  processBatch(indices, { apply = true } = {}) {
    const samples = this.samples;
    if (!samples || indices.length === 0) return;
    const { obsSize, actionSize } = samples;
    const count = indices.length;
    const settings = this.settings;

    let advantageMean = 0;
    for (const index of indices) advantageMean += samples.advantages[index];
    advantageMean /= count;
    let advantageVariance = 0;
    for (const index of indices) {
      const delta = samples.advantages[index] - advantageMean;
      advantageVariance += delta * delta;
    }
    const advantageStd = Math.sqrt(advantageVariance / count) + 1e-8;

    this.optimizer.zeroGrad();

    const sigma = this.policy.sigma;
    const logStd = this.policy.logStd;
    let policyLoss = 0;
    let valueLoss = 0;
    let entropy = 0;
    let kl = 0;
    let clipped = 0;

    for (const index of indices) {
      const obsOffset = index * obsSize;
      const actionOffset = index * actionSize;

      const mean = this.policy.actor.forward(samples.obs, obsOffset);
      const mu0 = mean[0];
      const mu1 = mean[1];
      const u0 = samples.actions[actionOffset];
      const u1 = samples.actions[actionOffset + 1];
      const sigma0 = sigma[0];
      const sigma1 = sigma[1];
      const z0 = (u0 - mu0) / sigma0;
      const z1 = (u1 - mu1) / sigma1;
      const logProbNew =
        -0.5 * (z0 * z0 + z1 * z1) -
        this.policy.logSigma0 -
        this.policy.logSigma1 -
        2 * HALF_LOG_TWO_PI;
      const logProbOld = samples.logProb[index];
      const ratio = Math.exp(logProbNew - logProbOld);
      const advantage = (samples.advantages[index] - advantageMean) / advantageStd;
      const boundedRatio = clamp(
        ratio,
        1 - settings.clipRatio,
        1 + settings.clipRatio,
      );
      const unclippedSurrogate = ratio * advantage;
      const clippedSurrogate = boundedRatio * advantage;
      policyLoss += -Math.min(unclippedSurrogate, clippedSurrogate);

      const dLossDRatio =
        unclippedSurrogate <= clippedSurrogate
          ? -advantage
          : ratio < 1 - settings.clipRatio || ratio > 1 + settings.clipRatio
            ? 0
            : -boundedRatio * advantage;
      const dLogProb = dLossDRatio * ratio;

      this.actorGrad[0] = dLogProb * (z0 / sigma0);
      this.actorGrad[1] = dLogProb * (z1 / sigma1);
      this.policy.actor.backward(this.actorGrad);

      this.policy.logStdGrad[0] +=
        dLogProb * (z0 * z0 - 1) - settings.entropyCoef;
      this.policy.logStdGrad[1] +=
        dLogProb * (z1 * z1 - 1) - settings.entropyCoef;
      const value = this.policy.critic.forward(samples.obs, obsOffset)[0];
      const target = samples.returns[index];
      const oldValue = samples.values[index];
      const boundedValue =
        oldValue + clamp(value - oldValue, -settings.valueClip, settings.valueClip);
      const squared = (value - target) * (value - target);
      const boundedSquared = (boundedValue - target) * (boundedValue - target);
      valueLoss += 0.5 * Math.max(squared, boundedSquared);
      let dValue = 0;
      if (squared >= boundedSquared) {
        dValue = value - target;
      } else if (boundedValue === value) {
        dValue = value - target;
      }
      this.criticGrad[0] = dValue * settings.valueCoef;
      this.policy.critic.backward(this.criticGrad);

      entropy += gaussianEntropy(logStd[0]) + gaussianEntropy(logStd[1]);
      kl += logProbOld - logProbNew;
      if (Math.abs(ratio - 1) > settings.clipRatio) clipped += 1;
    }

    this.optimizer.scaleGradients(1 / count);
    const norm = this.optimizer.gradientNorm();
    this.lastBatchLoss = {
      count,
      policyLoss: policyLoss / count,
      valueLoss: valueLoss / count,
      entropy: entropy / count,
      total:
        policyLoss / count +
        settings.valueCoef * (valueLoss / count) -
        settings.entropyCoef * (entropy / count),
    };
    if (!apply) return;
    this.optimizer.clipGradients(settings.gradClip);
    this.optimizer.step();
    this.policy.refreshSigma();

    const accumulator = this.accumulator;
    accumulator.policyLoss += policyLoss / count;
    accumulator.valueLoss += valueLoss / count;
    accumulator.entropy += entropy / count;
    accumulator.kl += kl / count;
    accumulator.clipFraction += clipped / count;
    accumulator.gradientNorm += norm;
    accumulator.batches += 1;
  }

  finishUpdate() {
    const accumulator = this.accumulator;
    const batches = Math.max(1, accumulator?.batches ?? 0);
    this.iterations += 1;
    this.stats = {
      policyLoss: (accumulator?.policyLoss ?? 0) / batches,
      valueLoss: (accumulator?.valueLoss ?? 0) / batches,
      entropy: (accumulator?.entropy ?? 0) / batches,
      approxKl: (accumulator?.kl ?? 0) / batches,
      clipFraction: (accumulator?.clipFraction ?? 0) / batches,
      gradientNorm: (accumulator?.gradientNorm ?? 0) / batches,
      meanReward: accumulator?.meanReward ?? 0,
      meanAdvantage: accumulator?.meanAdvantage ?? 0,
      rewardScale: this.rewardScale,
      returnStd: this.returnStd,
      samples: accumulator?.samples ?? 0,
      batches: accumulator?.batches ?? 0,
      learningRate: this.settings.learningRate,
      parameters: this.policy.parameterCount,
      iterations: this.iterations,
    };
    this.samples = null;
    this.accumulator = null;
  }
}
