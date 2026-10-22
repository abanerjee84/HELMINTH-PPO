import { OBSERVATION_COUNT, SENSOR_COUNT } from '../config.js';
import { gaussian } from '../sim/random.js';

const FOOD_LABELS = ['F', 'FL', 'L', 'BL', 'B', 'BR', 'R', 'FR'];
const HAZARD_LABELS = ['H·F', 'H·L', 'H·B', 'H·R'];
const WALL_LABELS = ['W·F', 'W·L', 'W·B', 'W·R'];
const WORM_LABELS = ['W·F', 'W·L', 'W·B', 'W·R'];
const CHOMPER_LABELS = ['C·F', 'C·L', 'C·B', 'C·R'];
const SELF_LABELS = ['S·P', 'S·B'];

const HALF_LOG_TWO_PI = 0.9189385332046727;
const HALF_LOG_TWO_PI_E = 1.4189385332046727;

function ringPosition(angle, radius) {
  return {
    x: Math.cos(angle) * radius,
    y: Math.sin(angle) * radius,
  };
}

function createInputLayout() {
  const inputs = [];

  for (let index = 0; index < 8; index += 1) {
    const angle = Math.PI / 2 + index * (Math.PI / 4);
    inputs.push({
      kind: 'food',
      index,
      label: FOOD_LABELS[index],
      ...ringPosition(angle, 0.9),
    });
  }

  for (let index = 0; index < 4; index += 1) {
    const angle = Math.PI / 2 + index * (Math.PI / 2);
    inputs.push({
      kind: 'hazard',
      index,
      label: HAZARD_LABELS[index],
      ...ringPosition(angle, 0.56),
    });
  }

  for (let index = 0; index < 4; index += 1) {
    const angle = Math.PI / 2 + index * (Math.PI / 2);
    inputs.push({
      kind: 'wall',
      index,
      label: WALL_LABELS[index],
      ...ringPosition(angle, 0.68),
    });
  }

  for (let index = 0; index < 4; index += 1) {
    const angle = Math.PI / 2 + index * (Math.PI / 2);
    inputs.push({
      kind: 'worm',
      index,
      label: WORM_LABELS[index],
      ...ringPosition(angle, 0.4),
    });
  }

  for (let index = 0; index < 4; index += 1) {
    const angle = Math.PI / 2 + index * (Math.PI / 2);
    inputs.push({
      kind: 'chomper',
      index,
      label: CHOMPER_LABELS[index],
      ...ringPosition(angle, 0.72),
    });
  }

  inputs.push({ kind: 'self', index: 0, label: SELF_LABELS[0], x: -0.74, y: 0.44 });
  inputs.push({ kind: 'self', index: 1, label: SELF_LABELS[1], x: 0.74, y: 0.44 });
  inputs.push({ kind: 'energy', index: 0, label: 'E', x: 0, y: 0.21 });
  inputs.push({ kind: 'bias', index: 0, label: 'B', x: 0, y: -0.17 });
  inputs.push({ kind: 'drift', index: 0, label: 'D·X', x: -0.3, y: -0.46 });
  inputs.push({ kind: 'drift', index: 1, label: 'D·Y', x: 0.3, y: -0.46 });
  return inputs;
}

export const INPUT_LAYOUT = Object.freeze(createInputLayout());

export const OUTPUT_LAYOUT = Object.freeze([
  { key: 'turn', label: 'TURN', x: -0.24, y: -1.1 },
  { key: 'thrust', label: 'THRUST', x: 0.24, y: -1.1 },
]);

/** Differential entropy of a Gaussian policy expressed through its log-std. */
export function gaussianEntropy(logStd) {
  return logStd + HALF_LOG_TWO_PI_E;
}

/** A single dense layer with fused weight/bias storage and matching gradients. */
export class Dense {
  constructor(inSize, outSize, rng, gain = 1) {
    this.inSize = inSize;
    this.outSize = outSize;
    this.weightCount = outSize * inSize;
    this.params = new Float32Array(this.weightCount + outSize);
    this.grads = new Float32Array(this.params.length);
    const limit = gain * Math.sqrt(6 / (inSize + outSize));
    for (let index = 0; index < this.weightCount; index += 1) {
      this.params[index] = (rng() * 2 - 1) * limit;
    }
  }

  get tensor() {
    return { params: this.params, grads: this.grads };
  }
}

/**
 * A tanh feed-forward stack. The last layer is linear so it can emit an
 * unbounded action mean and an unbounded state value.
 */
export class MLP {
  constructor(sizes, rng, outputGain = 0.1) {
    this.sizes = sizes.slice();
    this.layers = [];
    for (let index = 0; index < sizes.length - 1; index += 1) {
      const gain = index === sizes.length - 2 ? outputGain : 1;
      this.layers.push(new Dense(sizes[index], sizes[index + 1], rng, gain));
    }
    this.activations = sizes.map((size) => new Float32Array(size));
    this.deltas = this.layers.map((layer) => new Float32Array(layer.inSize));
    this.dInput = new Float32Array(sizes[0]);
    this.parameterCount = this.layers.reduce(
      (total, layer) => total + layer.params.length,
      0,
    );
  }

  get tensors() {
    return this.layers.map((layer) => layer.tensor);
  }

  /** Runs the stack, caching activations for a later backward pass. */
  forward(input, offset = 0) {
    const first = this.activations[0];
    for (let index = 0; index < first.length; index += 1) {
      first[index] = input[offset + index];
    }
    for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex += 1) {
      const layer = this.layers[layerIndex];
      const source = this.activations[layerIndex];
      const target = this.activations[layerIndex + 1];
      const params = layer.params;
      const isOutput = layerIndex === this.layers.length - 1;
      for (let out = 0; out < layer.outSize; out += 1) {
        const base = out * layer.inSize;
        let sum = params[layer.weightCount + out];
        for (let i = 0; i < layer.inSize; i += 1) {
          sum += params[base + i] * source[i];
        }
        target[out] = isOutput ? sum : Math.tanh(sum);
      }
    }
    return this.activations[this.layers.length];
  }

  /** Accumulates parameter gradients for dLoss/dOutput and returns dLoss/dInput. */
  backward(dOutput) {
    const last = this.layers.length - 1;
    let gradient = dOutput;
    for (let layerIndex = last; layerIndex >= 0; layerIndex -= 1) {
      const layer = this.layers[layerIndex];
      const source = this.activations[layerIndex];
      const produced = this.activations[layerIndex + 1];
      const params = layer.params;
      const grads = layer.grads;
      const delta = this.deltas[layerIndex];
      delta.fill(0);
      const isOutput = layerIndex === last;
      for (let out = 0; out < layer.outSize; out += 1) {
        let g = gradient[out];
        if (!isOutput) g *= 1 - produced[out] * produced[out];
        if (g === 0) continue;
        const base = out * layer.inSize;
        grads[layer.weightCount + out] += g;
        for (let i = 0; i < layer.inSize; i += 1) {
          grads[base + i] += g * source[i];
          delta[i] += params[base + i] * g;
        }
      }
      gradient = delta;
    }
    this.dInput.set(gradient);
    return this.dInput;
  }
}

/**
 * Actor-critic policy: a Gaussian actor over a two-dimensional motor command
 * and a scalar critic that estimates the discounted return of the arena.
 */
export class Policy {
  constructor({ inputSize = OBSERVATION_COUNT, hiddenSizes = [24, 24], actionSize = 2, rng, config }) {
    this.config = config;
    this.inputSize = inputSize;
    this.actionSize = actionSize;
    this.hiddenSizes = hiddenSizes.slice();
    this.actor = new MLP([inputSize, ...hiddenSizes, actionSize], rng, 0.08);
    this.critic = new MLP([inputSize, ...hiddenSizes, 1], rng, 0.6);
    this.logStd = new Float32Array(actionSize);
    this.logStdGrad = new Float32Array(actionSize);
    this.sigma = new Float32Array(actionSize);
    this.minLogStd = config.ppo.minLogStd;
    this.maxLogStd = config.ppo.maxLogStd;
    this.turnGain = config.ppo.turnGain;
    this.thrustGain = config.ppo.thrustGain;
    for (let index = 0; index < actionSize; index += 1) {
      this.logStd[index] = config.ppo.initialLogStd;
    }
    this.refreshSigma();
  }

  get tensors() {
    return [
      ...this.actor.tensors,
      ...this.critic.tensors,
      { params: this.logStd, grads: this.logStdGrad },
    ];
  }

  get parameterCount() {
    return (
      this.actor.parameterCount +
      this.critic.parameterCount +
      this.logStd.length
    );
  }

  get hiddenCount() {
    return this.hiddenSizes.reduce((total, size) => total + size, 0);
  }

  refreshSigma() {
    this.logSigma0 = 0;
    this.logSigma1 = 0;
    for (let index = 0; index < this.logStd.length; index += 1) {
      this.logStd[index] = Math.min(
        this.maxLogStd,
        Math.max(this.minLogStd, this.logStd[index]),
      );
      this.sigma[index] = Math.exp(this.logStd[index]);
      const logSigma = Math.log(this.sigma[index]);
      if (index === 0) this.logSigma0 = logSigma;
      else this.logSigma1 = logSigma;
    }
  }

  value(observation) {
    return this.critic.forward(observation)[0];
  }

  /** Samples a motor command and returns everything the rollout needs. */
  act(observation, rng) {
    const mean = this.actor.forward(observation);
    const value = this.critic.forward(observation)[0];
    const mu0 = mean[0];
    const mu1 = mean[1];
    const noise0 = gaussian(rng);
    const noise1 = gaussian(rng);
    const sigma0 = this.sigma[0];
    const sigma1 = this.sigma[1];
    const u0 = mu0 + sigma0 * noise0;
    const u1 = mu1 + sigma1 * noise1;
    const z0 = (u0 - mu0) / sigma0;
    const z1 = (u1 - mu1) / sigma1;
    const logProb =
      -0.5 * (z0 * z0 + z1 * z1) -
      this.logSigma0 -
      this.logSigma1 -
      2 * HALF_LOG_TWO_PI;
    const turn = Math.tanh(u0 * this.turnGain);
    const thrust = 1 / (1 + Math.exp(-u1 * this.thrustGain));
    return {
      turn,
      thrust,
      u0,
      u1,
      logProb,
      value,
      mean: [mu0, mu1],
      hidden: this.hiddenActivations(),
    };
  }

  hiddenActivations() {
    const values = [];
    for (let layer = 1; layer < this.actor.activations.length - 1; layer += 1) {
      const activations = this.actor.activations[layer];
      for (let index = 0; index < activations.length; index += 1) {
        values.push(activations[index]);
      }
    }
    return values;
  }
}

/**
 * Describes the live actor network as a graph so the telemetry panel can draw
 * the policy the way the arena draws it: sensors in, hidden units, motors out.
 */
export function describeNetwork(actor) {
  const layers = actor.layers;
  const hiddenSizes = layers.slice(0, -1).map((layer) => layer.outSize);
  const hiddenLayout = [];
  const hiddenLayers = [];
  let counter = 0;
  hiddenSizes.forEach((size, layer) => {
    const nodes = [];
    for (let row = 0; row < size; row += 1) {
      const node = {
        index: counter,
        layer,
        row,
        label: `L${layer + 1}H${row + 1}`,
        candidate: counter,
      };
      nodes.push(node);
      hiddenLayout.push(node);
      counter += 1;
    }
    hiddenLayers.push(nodes);
  });

  const connections = [];
  let sourceCount = INPUT_LAYOUT.length;
  let sourceKind = 'input';
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
    const layer = layers[layerIndex];
    const targetKind = layerIndex === layers.length - 1 ? 'output' : 'hidden';
    const targetCount = layer.outSize;
    for (let target = 0; target < targetCount; target += 1) {
      for (let source = 0; source < sourceCount; source += 1) {
        connections.push({
          fromKind: sourceKind,
          from: source,
          toKind: targetKind,
          to: target,
          hop: layerIndex,
          weight: 0,
        });
      }
    }
    sourceCount = targetCount;
    sourceKind = 'hidden';
  }

  const refresh = () => {
    let cursor = 0;
    let previousCount = INPUT_LAYOUT.length;
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex += 1) {
      const layer = layers[layerIndex];
      for (let target = 0; target < layer.outSize; target += 1) {
        for (let source = 0; source < previousCount; source += 1) {
          connections[cursor].weight = layer.params[target * layer.inSize + source];
          cursor += 1;
        }
      }
      previousCount = layer.outSize;
    }
  };
  refresh();

  const hiddenCount = hiddenLayout.length;
  const stats = {
    hidden: hiddenCount,
    nodes: hiddenCount + 2,
    outputs: 2,
    enabled: connections.length,
    signature: `${INPUT_LAYOUT.length} · ${hiddenSizes.length}L · ${hiddenSizes[0] ?? 0}H · 2`,
    hiddenSizes: hiddenSizes.slice(),
  };

  return {
    inputLayout: INPUT_LAYOUT,
    hiddenLayout,
    hiddenLayers,
    hiddenDepth: hiddenSizes.length,
    hiddenPerLayer: hiddenSizes[0] ?? 0,
    hiddenCandidates: hiddenLayout,
    outputLayout: OUTPUT_LAYOUT,
    connections,
    stats,
    refresh,
  };
}
