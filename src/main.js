import '@fontsource-variable/sora';
import '@fontsource-variable/jetbrains-mono';
import './style.css';

import { CONFIG, STEP } from './config.js';
import { Trainer } from './ppo/trainer.js';
import { ArenaRenderer } from './render/arena.js';
import { Dashboard } from './ui/dashboard.js';

const host = document.getElementById('arenaHost');
const dashboard = new Dashboard();
const renderer = new ArenaRenderer(host);

let trainer = new Trainer(CONFIG);
let running = false;
let speed = 4;
let accumulator = 0;
let intermissionTimer = 0;
let lastFrame = performance.now();
let fps = 60;

const state = () => ({ running, speed, fps });

function toggleRunning() {
  if (running) {
    running = false;
    trainer.pause();
  } else {
    running = true;
    if (trainer.phase === 'ready' || trainer.phase === 'paused') {
      trainer.start();
    }
  }
  dashboard.update(trainer, state(), performance.now());
}

function resetRun() {
  trainer = new Trainer(CONFIG);
  running = false;
  accumulator = 0;
  intermissionTimer = 0;
  dashboard.resetState();
  renderer.populationSource = null;
  renderer.ecologySource = null;
  renderer.predatorSource = null;
  dashboard.update(trainer, state(), performance.now());
}

function setSpeed(value) {
  speed = value;
  dashboard.setSpeed(value);
}

dashboard.bindSpeed(setSpeed);
dashboard.bindActions({ onStart: toggleRunning, onReset: resetRun });
dashboard.setSpeed(speed);

function simulate(realDelta) {
  if (!running) {
    accumulator = 0;
    return;
  }

  if (trainer.phase === 'intermission') {
    if (intermissionTimer <= 0) intermissionTimer = 1.8;
    intermissionTimer -= realDelta;
    if (intermissionTimer <= 0) {
      trainer.nextCohort();
      intermissionTimer = 0;
      accumulator = 0;
    }
    return;
  }

  if (trainer.phase === 'paused') {
    return;
  }

  accumulator += realDelta * speed;
  const maxSteps = 40;
  let steps = 0;
  while (accumulator >= STEP && steps < maxSteps) {
    trainer.step(STEP);
    accumulator -= STEP;
    steps += 1;
    if (trainer.phase === 'intermission') {
      intermissionTimer = 1.8;
      accumulator = 0;
      break;
    }
  }
  if (steps >= maxSteps) accumulator = 0;
}

function animate(now) {
  requestAnimationFrame(animate);
  const rawDelta = Math.min(0.1, Math.max(0, (now - lastFrame) / 1000));
  lastFrame = now;
  if (rawDelta > 0) fps = fps * 0.92 + (1 / rawDelta) * 0.08;

  simulate(rawDelta);
  trainer.continueUpdate();

  const events = trainer.drainEvents();
  if (events.length) renderer.ingest(events);
  renderer.update(rawDelta, trainer);
  dashboard.update(trainer, state(), now);
}

window.__HELMINTH__ = {
  getTrainer: () => trainer,
  getEvolution: () => trainer,
  getState: () => ({
    ...state(),
    phase: trainer.phase,
    generation: trainer.generation,
    iteration: trainer.iteration,
    elapsed: trainer.elapsed,
    simulatedSeconds: trainer.totalElapsed,
    alive: trainer.aliveCount,
    chompers: trainer.predators?.chompers.length ?? 0,
    history: trainer.history,
    ppo: trainer.ppo.stats,
    parameterChecksum: trainer.ppo.parameterChecksum,
  }),
  getRenderer: () => renderer,
  setSpeed,
};

dashboard.update(trainer, state(), performance.now());
renderer.update(0.016, trainer);
requestAnimationFrame(animate);
window.__HELMINTH_READY__ = true;
