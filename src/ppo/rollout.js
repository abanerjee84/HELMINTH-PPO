/**
 * Fixed-capacity on-policy storage for a cohort of parallel agents.
 *
 * Every agent owns one column of the buffer. A column stops accepting samples
 * the moment its worm dies, which keeps the generalised advantage estimate
 * aligned with the trajectory that actually produced the reward.
 */
export class RolloutBuffer {
  constructor({ steps, agents, obsSize, actionSize }) {
    this.steps = steps;
    this.agents = agents;
    this.obsSize = obsSize;
    this.actionSize = actionSize;
    const cells = steps * agents;
    this.obs = new Float32Array(cells * obsSize);
    this.actions = new Float32Array(cells * actionSize);
    this.logProb = new Float32Array(cells);
    this.values = new Float32Array(cells);
    this.rewards = new Float32Array(cells);
    this.terminals = new Float32Array(cells);
    this.masks = new Float32Array(cells);
    this.advantages = new Float32Array(cells);
    this.returns = new Float32Array(cells);
    this.cursor = 0;
  }

  reset() {
    this.cursor = 0;
    this.masks.fill(0);
    this.terminals.fill(0);
  }

  get filled() {
    return this.cursor;
  }

  write(step, agentIndex, sample) {
    const cell = step * this.agents + agentIndex;
    const obsBase = cell * this.obsSize;
    for (let index = 0; index < this.obsSize; index += 1) {
      this.obs[obsBase + index] = sample.obs[index];
    }
    const actionBase = cell * this.actionSize;
    this.actions[actionBase] = sample.u0;
    this.actions[actionBase + 1] = sample.u1;
    this.logProb[cell] = sample.logProb;
    this.values[cell] = sample.value;
    this.rewards[cell] = sample.reward;
    this.terminals[cell] = sample.terminal ? 1 : 0;
    this.masks[cell] = 1;
  }

  /**
   * Generalised advantage estimate over the filled prefix of the buffer.
   * `lastValues` bootstraps the value of the state that follows the final
   * stored transition; a terminated agent bootstraps with zero instead.
   * `rewardScale` keeps the critic's targets on a unit scale.
   */
  computeGae(gamma, lambda, lastValues, rewardScale = 1) {
    const steps = this.cursor;
    this.advantages.fill(0);
    for (let step = steps - 1; step >= 0; step -= 1) {
      for (let agent = 0; agent < this.agents; agent += 1) {
        const cell = step * this.agents + agent;
        if (this.masks[cell] === 0) continue;
        const terminal = this.terminals[cell] === 1;
        let nextValue = 0;
        let nextAdvantage = 0;
        let nonTerminal = 0;
        if (!terminal) {
          nonTerminal = 1;
          if (step === steps - 1) {
            nextValue = lastValues[agent] ?? 0;
          } else {
            nextValue = this.values[cell + this.agents];
            nextAdvantage = this.advantages[cell + this.agents];
          }
        }
        const delta =
          this.rewards[cell] * rewardScale +
          gamma * nextValue * nonTerminal -
          this.values[cell];
        this.advantages[cell] =
          delta + gamma * lambda * nonTerminal * nextAdvantage;
      }
    }
    for (let cell = 0; cell < this.advantages.length; cell += 1) {
      this.returns[cell] = this.advantages[cell] + this.values[cell];
    }
  }

  /** Spread of the discounted returns, used to keep the critic on unit scale. */
  returnStats() {
    const steps = this.cursor;
    let count = 0;
    let sum = 0;
    let sumSquares = 0;
    for (let cell = 0; cell < steps * this.agents; cell += 1) {
      if (this.masks[cell] !== 1) continue;
      const value = this.returns[cell];
      count += 1;
      sum += value;
      sumSquares += value * value;
    }
    if (count === 0) return { count: 0, mean: 0, std: 1 };
    const mean = sum / count;
    const variance = Math.max(1e-6, sumSquares / count - mean * mean);
    return { count, mean, std: Math.sqrt(variance) };
  }

  /** Flattens the live columns into a contiguous training set. */
  compact() {
    const steps = this.cursor;
    const cells = steps * this.agents;
    let count = 0;
    for (let cell = 0; cell < cells; cell += 1) {
      if (this.masks[cell] === 1) count += 1;
    }

    const obs = new Float32Array(count * this.obsSize);
    const actions = new Float32Array(count * this.actionSize);
    const logProb = new Float32Array(count);
    const values = new Float32Array(count);
    const advantages = new Float32Array(count);
    const returns = new Float32Array(count);
    const rewards = new Float32Array(count);

    let cursor = 0;
    for (let cell = 0; cell < cells; cell += 1) {
      if (this.masks[cell] !== 1) continue;
      obs.set(this.obs.subarray(cell * this.obsSize, (cell + 1) * this.obsSize), cursor * this.obsSize);
      actions.set(
        this.actions.subarray(cell * this.actionSize, (cell + 1) * this.actionSize),
        cursor * this.actionSize,
      );
      logProb[cursor] = this.logProb[cell];
      values[cursor] = this.values[cell];
      advantages[cursor] = this.advantages[cell];
      returns[cursor] = this.returns[cell];
      rewards[cursor] = this.rewards[cell];
      cursor += 1;
    }

    return {
      count,
      obsSize: this.obsSize,
      actionSize: this.actionSize,
      obs,
      actions,
      logProb,
      values,
      advantages,
      returns,
      rewards,
    };
  }
}
