import { CONFIG, OBSERVATION_COUNT, WORM_PALETTE, WORLD } from '../config.js';
import { resolveWormCollisions } from '../sim/collisions.js';
import { Ecology } from '../sim/ecology.js';
import { ChomperPack } from '../sim/predators.js';
import { mulberry32 } from '../sim/random.js';
import { Worm } from '../sim/worm.js';
import { describeNetwork } from './network.js';
import { PPO } from './ppo.js';
import { RolloutBuffer } from './rollout.js';

const now = () =>
  globalThis.performance?.now ? globalThis.performance.now() : Date.now();

/**
 * Drives the arena and the learner together.
 *
 * A cohort of parallel worms shares one policy. Every simulation step feeds a
 * transition into the rollout buffer; when the buffer fills (or the cohort
 * dies out) the buffer is turned into advantages, the policy is updated, and
 * the next cohort starts from the improved weights.
 */
export class Trainer {
  constructor(config = CONFIG) {
    this.config = config;
    this.reset();
  }

  reset() {
    this.seedRng = mulberry32(this.config.seed);
    this.policyRng = mulberry32(this.config.seed ^ 0x9e3779b9);
    this.ppo = new PPO(this.config, this.policyRng);
    this.network = describeNetwork(this.ppo.policy.actor);
    this.phase = 'ready';
    this.generation = 0;
    this.iteration = 0;
    this.elapsed = 0;
    this.totalElapsed = 0;
    this.events = [];
    this.history = [];
    this.ranking = [];
    this.champion = null;
    this.lastSummary = null;
    this.bestEver = 0;
    this.cohortEnded = false;
    this.timeAlive = 0;
    this.buffer = new RolloutBuffer({
      steps: this.config.ppo.rolloutSteps,
      agents: this.config.populationSize,
      obsSize: OBSERVATION_COUNT,
      actionSize: 2,
    });
    this.startCohort();
  }

  startCohort() {
    this.generation += 1;
    this.elapsed = 0;
    this.events = [];
    this.cohortEnded = false;
    this.buffer.reset();

    const seed = this.config.seed;
    const spawnRng = mulberry32((seed ^ (this.generation * 2654435761)) >>> 0);
    this.ecology = new Ecology(seed, this.config);
    this.predators = new ChomperPack(seed + 131, this.config, this.ecology);
    this.ecology.relocateSpawnPointsAwayFrom(this.predators.chompers);

    this.agents = Array.from(
      { length: this.config.populationSize },
      (_, index) => {
        const spawn = this.ecology.spawnPoints[index];
        const heading =
          Math.atan2(WORLD.height * 0.5 - spawn.y, WORLD.width * 0.5 - spawn.x) +
          (spawnRng() - 0.5) * 0.9;
        const worm = new Worm({
          id: `${this.generation}-${index}`,
          index,
          color: WORM_PALETTE[index % WORM_PALETTE.length],
          x: spawn.x,
          y: spawn.y,
          heading,
          hiddenCount: this.ppo.policy.hiddenCount,
          config: this.config,
        });
        // A per-worm drift vector keeps a shared policy from marching the whole
        // cohort into the same corner — and into each other.
        const drift = spawnRng() * Math.PI * 2;
        worm.identity = [Math.cos(drift), Math.sin(drift)];
        return {
          id: `${this.generation}-${index}`,
          index,
          color: worm.color,
          network: this.network,
          worm,
          // Each worm draws its own action noise so a shared policy does not
          // make the whole cohort behave like a single body.
          rng: mulberry32((seed ^ (0x9e3779b9 * (index + 1))) >>> 0),
        };
      },
    );
    this.ranking = [...this.agents];
    this.timeAlive = this.agents.length;
  }

  start() {
    if (this.phase === 'ready' || this.phase === 'paused') {
      this.phase = 'running';
    }
  }

  pause() {
    if (this.phase === 'running') this.phase = 'paused';
  }

  step(dt) {
    if (this.phase !== 'running' && this.phase !== 'optimizing') return;
    const collecting = this.phase === 'running';

    this.ecology.update(dt);
    this.elapsed += dt;
    this.totalElapsed += dt;

    const step = collecting ? this.buffer.cursor : -1;
    for (const agent of this.agents) {
      const worm = agent.worm;
      if (!worm.alive) continue;

      const observation = worm.sense(
        this.ecology,
        this.predators.chompers,
        this.agents,
      );
      const action = this.ppo.act(observation, agent.rng);
      worm.applyAction(action);

      const result = worm.step(dt, this.ecology);
      if (result?.food) {
        this.events.push({
          type: 'eat',
          x: result.food.x,
          y: result.food.y,
          color: worm.color,
        });
      }
      if (result?.died) {
        this.events.push({
          type: 'death',
          x: worm.x,
          y: worm.y,
          color: worm.color,
          cause: result.deathCause,
        });
      }
      if (result?.hazard && Math.floor(this.elapsed * 60) % 14 === 0) {
        this.events.push({
          type: 'hazard',
          x: worm.x,
          y: worm.y,
          color: '#ff5f8f',
        });
      }

      if (!collecting) continue;

      this.buffer.write(step, agent.index, {
        obs: observation,
        u0: action.u0,
        u1: action.u1,
        logProb: action.logProb,
        value: action.value,
        reward: worm.stepReward,
        terminal: Boolean(result?.died),
      });
    }
    const collisionDeaths = resolveWormCollisions(this.agents);
    for (const worm of collisionDeaths) {
      this.events.push({
        type: 'death',
        x: worm.x,
        y: worm.y,
        color: worm.color,
        cause: worm.deathCause,
      });
    }

    const bites = this.predators.update(dt, this.agents, this.elapsed);
    for (const bite of bites) {
      this.events.push({
        type: 'chomp',
        x: bite.x,
        y: bite.y,
        color: bite.color,
      });
      if (bite.died) {
        this.events.push({
          type: 'death',
          x: bite.x,
          y: bite.y,
          color: bite.color,
          cause: 'predation',
        });
      }
    }

    for (const agent of this.agents) {
      const worm = agent.worm;
      if (worm.alive || worm.terminalSettled) continue;
      const adjustment = worm.finalizeTerminal();
      if (!collecting) continue;
      const cell = step * this.agents + agent.index;
      if (this.buffer.masks[cell] === 1) {
        this.buffer.terminals[cell] = 1;
        this.buffer.rewards[cell] += adjustment;
      }
    }

    this.ranking = [...this.agents].sort(
      (a, b) => b.worm.fitness - a.worm.fitness || a.index - b.index,
    );
    this.timeAlive = this.agents.reduce(
      (total, agent) => total + (agent.worm.alive ? 1 : 0),
      0,
    );

    const cohortDone = this.cohortFinished();

    if (!collecting) {
      if (cohortDone) this.cohortEnded = true;
      return;
    }

    this.buffer.cursor = step + 1;

    if (this.buffer.cursor >= this.config.ppo.rolloutSteps || cohortDone) {
      this.cohortEnded = cohortDone;
      this.beginUpdate();
    }
  }

  /**
   * A cohort is a wave: it ends when the last worm in it is dead, and the
   * next wave spawns into the same arena with the improved policy. A single
   * survivor keeps the cohort alive.
   */
  cohortFinished() {
    return this.timeAlive === 0;
  }

  /** Turns the collected rollout into advantages and queues a policy update. */
  beginUpdate() {
    const lastValues = new Float32Array(this.agents.length);
    for (const agent of this.agents) {
      if (!agent.worm.alive) continue;
      lastValues[agent.index] = this.ppo.value(agent.worm.observation);
    }
    this.buffer.computeGae(
      this.config.ppo.gamma,
      this.config.ppo.gaeLambda,
      lastValues,
      this.ppo.rewardScale,
    );
    this.ppo.observeReturns(this.buffer.returnStats());
    this.ppo.beginUpdate(this.buffer.compact());
    this.phase = 'optimizing';
  }

  /** Spends a bounded slice of frame time on the queued policy update. */
  continueUpdate(budgetMs = this.config.ppo.updateBudgetMs) {
    if (this.phase !== 'optimizing') return false;
    const started = now();
    while (this.ppo.hasPendingWork()) {
      this.ppo.updateMinibatch();
      if (now() - started >= budgetMs) break;
    }
    if (!this.ppo.hasPendingWork()) this.completeUpdate();
    return true;
  }

  completeUpdate() {
    this.iteration += 1;
    this.network.refresh();
    this.buffer.reset();
    if (this.cohortEnded) {
      this.completeCohort();
    } else {
      this.phase = 'running';
    }
  }

  completeCohort() {
    this.phase = 'intermission';
    this.ranking = [...this.agents].sort(
      (a, b) => b.worm.fitness - a.worm.fitness || a.index - b.index,
    );
    this.champion = this.ranking[0];
    this.bestEver = Math.max(this.bestEver, this.champion.worm.fitness);

    const average =
      this.ranking.reduce((total, agent) => total + agent.worm.fitness, 0) /
      this.ranking.length;
    const record = {
      generation: this.generation,
      best: this.champion.worm.fitness,
      average,
      foodEaten: this.champion.worm.foodEaten,
      timeAlive: this.champion.worm.age,
      color: this.champion.color,
      iteration: this.iteration,
      policyLoss: this.ppo.stats.policyLoss,
      valueLoss: this.ppo.stats.valueLoss,
      entropy: this.ppo.stats.entropy,
      approxKl: this.ppo.stats.approxKl,
      meanReward: this.ppo.stats.meanReward,
    };
    this.history.push(record);
    this.lastSummary = record;
    this.events.push({
      type: 'generation',
      x: WORLD.width * 0.5,
      y: WORLD.height * 0.5,
      color: this.champion.color,
    });
  }

  nextCohort() {
    if (this.phase !== 'intermission') return;
    this.startCohort();
    this.phase = 'running';
  }

  get leader() {
    if (this.phase === 'intermission' && this.champion) return this.champion;
    if (this.phase === 'ready') return this.ranking[0] ?? this.agents[0];
    let leader = null;
    for (const agent of this.agents) {
      if (!agent.worm.alive) continue;
      if (!leader || agent.worm.fitness > leader.worm.fitness) leader = agent;
    }
    return leader ?? this.ranking[0] ?? this.agents[0];
  }

  get aliveCount() {
    return this.agents.reduce(
      (total, agent) => total + (agent.worm.alive ? 1 : 0),
      0,
    );
  }

  drainEvents() {
    const events = this.events;
    this.events = [];
    return events;
  }
}
