const numberFormat = new Intl.NumberFormat('en-US');

const HOP_COLORS = ['105, 239, 209', '150, 178, 255', '255, 176, 122', '200, 255, 109'];
const LIT_PER_HOP = 11;
const PULSE_PER_HOP = 3;

function formatInteger(value) {
  return numberFormat.format(Math.round(value || 0));
}

function formatTimer(seconds) {
  const elapsed = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(elapsed / 60);
  const remainder = elapsed % 60;
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function formatFraction(value) {
  return (value ?? 0).toFixed(2);
}

function formatSigned(value) {
  const normalized = Math.abs(value) < 0.005 ? 0 : value;
  return `${normalized >= 0 ? '+' : '-'}${Math.abs(normalized).toFixed(2)}`;
}

function withAlpha(hex, alpha) {
  const value = Number.parseInt(hex.slice(1), 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function canvasSize(canvas, context) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

function inputColor(kind) {
  if (kind === 'food') return '#b9ff68';
  if (kind === 'hazard') return '#ff6e9b';
  if (kind === 'wall') return '#d9e4ee';
  if (kind === 'worm') return '#6bc8ff';
  if (kind === 'chomper') return '#ff9b71';
  if (kind === 'self') return '#c9a0ff';
  if (kind === 'energy') return '#ffc96b';
  if (kind === 'kinematic') return '#5ee7f0';
  if (kind === 'drift') return '#7f96a8';
  return '#85a99a';
}

export class Dashboard {
  constructor() {
    this.app = document.getElementById('app');
    this.startButton = document.getElementById('startBtn');
    this.startLabel = document.getElementById('startLabel');
    this.startHint = document.getElementById('startHint');
    this.resetButton = document.getElementById('resetBtn');
    this.runStatus = document.getElementById('runStatus');
    this.statusMeta = document.getElementById('statusMeta');
    this.fpsValue = document.getElementById('fpsValue');
    this.phaseLabel = document.getElementById('phaseLabel');
    this.phaseDetail = document.getElementById('phaseDetail');
    this.genNumber = document.getElementById('genNumber');
    this.genTime = document.getElementById('genTime');
    this.genProgress = document.getElementById('genProgress');
    this.aliveCount = document.getElementById('aliveCount');
    this.foodCount = document.getElementById('foodCount');
    this.avgFit = document.getElementById('avgFit');
    this.bestFit = document.getElementById('bestFit');
    this.leaderCard = document.getElementById('leaderCard');
    this.leaderName = document.getElementById('leaderName');
    this.leaderPolicy = document.getElementById('leaderPolicy');
    this.leaderScore = document.getElementById('leaderScore');
    this.leaderFood = document.getElementById('leaderFood');
    this.leaderTime = document.getElementById('leaderTime');
    this.leaderForward = document.getElementById('leaderForward');
    this.leaderStraight = document.getElementById('leaderStraight');
    this.leaderEnergy = document.getElementById('leaderEnergy');
    this.leaderEnergyBar = document.getElementById('leaderEnergyBar');
    this.standings = document.getElementById('standings');
    this.idlePrompt = document.getElementById('idlePrompt');
    this.toast = document.getElementById('generationToast');
    this.toastTitle = document.getElementById('toastTitle');
    this.toastDetail = document.getElementById('toastDetail');
    this.policySignature = document.getElementById('policySignature');
    this.brainDominant = document.getElementById('brainDominant');
    this.brainDominantValue = document.getElementById('brainDominantValue');
    this.brainHiddenMean = document.getElementById('brainHiddenMean');
    this.brainActiveLinks = document.getElementById('brainActiveLinks');
    this.brainTotalLinks = document.getElementById('brainTotalLinks');
    this.brainArchitecture = document.getElementById('brainArchitecture');
    this.brainLinkCount = document.getElementById('brainLinkCount');
    this.hiddenUnits = document.getElementById('hiddenUnits');
    this.weightCount = document.getElementById('weightCount');
    this.learningRate = document.getElementById('learningRate');
    this.ppoIteration = document.getElementById('ppoIteration');
    this.ppoEntropy = document.getElementById('ppoEntropy');
    this.ppoLoss = document.getElementById('ppoLoss');
    this.bestEver = document.getElementById('bestEver');
    this.lastChampion = document.getElementById('lastChampion');
    this.cohortCount = document.getElementById('cohortCount');
    this.turnValue = document.getElementById('turnValue');
    this.thrustValue = document.getElementById('thrustValue');
    this.turnFill = document.getElementById('turnFill');
    this.thrustFill = document.getElementById('thrustFill');
    this.aliveTotal = document.getElementById('aliveTotal');
    this.idlePopulation = document.getElementById('idlePopulation');

    this.brainCanvas = document.getElementById('brainCanvas');
    this.brainContext = this.brainCanvas.getContext('2d');
    this.fitnessCanvas = document.getElementById('fitnessCanvas');
    this.fitnessContext = this.fitnessCanvas.getContext('2d');

    this.textCache = new Map();
    this.lastDomUpdate = -Infinity;
    this.lastBrainDraw = -Infinity;
    this.lastFitnessSignature = '__initial__';
    this.lastStandings = '';
    this.lastToastGeneration = 0;
    this.brainTrace = { key: null, turn: [], thrust: [] };
    this.running = false;
  }

  resetState() {
    this.lastDomUpdate = -Infinity;
    this.lastBrainDraw = -Infinity;
    this.lastFitnessSignature = '__reset__';
    this.lastStandings = '';
    this.lastToastGeneration = 0;
    this.brainTrace = { key: null, turn: [], thrust: [] };
    this.textCache.clear();
    this.toast.classList.remove('is-visible');
  }

  setText(element, value) {
    if (!element) return;
    if (this.textCache.get(element) === value) return;
    this.textCache.set(element, value);
    element.textContent = value;
  }

  setSpeed(speed) {
    document.querySelectorAll('[data-speed]').forEach((button) => {
      const active = Number(button.dataset.speed) === Number(speed);
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }

  bindSpeed(handler) {
    document.querySelectorAll('[data-speed]').forEach((button) => {
      button.addEventListener('click', () => handler(Number(button.dataset.speed)));
    });
  }

  bindActions({ onStart, onReset }) {
    this.startButton.addEventListener('click', onStart);
    this.resetButton.addEventListener('click', onReset);
  }

  update(trainer, state, now = performance.now()) {
    this.running = state.running;
    const interval = 90;
    if (now - this.lastDomUpdate < interval) {
      this.updateBrain(trainer, now);
      return;
    }
    this.lastDomUpdate = now;

    const { running, speed, fps } = state;
    const phase = trainer.phase;
    const leader = trainer.leader;
    const intermission = phase === 'intermission';
    const paused = !running && phase !== 'ready';
    const ready = phase === 'ready';

    this.app.classList.toggle('is-running', running);
    this.app.classList.toggle('is-paused', paused || (running && intermission));
    this.setSpeed(speed);
    this.setText(this.fpsValue, String(Math.round(fps || 0)));

    if (ready) {
      this.setText(this.runStatus, 'STANDBY');
      this.setText(this.statusMeta, 'Awaiting the first rollout');
      this.setText(this.startLabel, 'Start training');
      this.setText(this.startHint, 'One policy · ten parallel worms');
      this.startButton.setAttribute('aria-label', 'Start training');
      this.setText(this.phaseLabel, 'PRE-FLIGHT');
      this.setText(this.phaseDetail, `${trainer.agents.length} actors bound to one policy`);
      this.setText(
        this.idlePopulation,
        `${trainer.agents.length} sensory worms. Three tiny chompers.`,
      );
    } else if (intermission) {
      this.setText(this.runStatus, running ? 'COHORT SCORED' : 'ROLLOUT ON HOLD');
      this.setText(
        this.statusMeta,
        running ? 'Updating the policy · next cohort queued' : 'Resume to update the policy',
      );
      this.setText(this.startLabel, running ? 'Pause training' : 'Resume training');
      this.setText(
        this.startHint,
        running ? 'Policy updated · policy gradient applied' : 'Next cohort waiting',
      );
      this.startButton.setAttribute('aria-label', running ? 'Pause training' : 'Resume training');
      this.setText(this.phaseLabel, 'COHORT SCORED');
      this.setText(this.phaseDetail, `Champion fitness ${formatInteger(leader?.worm.fitness)}`);
    } else if (paused) {
      this.setText(this.runStatus, 'OBSERVATION HOLD');
      this.setText(this.statusMeta, 'Simulation paused by operator');
      this.setText(this.startLabel, 'Resume training');
      this.setText(this.startHint, 'Continue cohort evaluation');
      this.startButton.setAttribute('aria-label', 'Resume training');
      this.setText(this.phaseLabel, 'HOLD');
      this.setText(this.phaseDetail, `${trainer.aliveCount} organisms suspended`);
    } else if (trainer.phase === 'optimizing') {
      this.setText(this.runStatus, 'POLICY UPDATE');
      this.setText(
        this.statusMeta,
        `Clipped surrogate · iteration ${String(trainer.iteration + 1).padStart(3, '0')}`,
      );
      this.setText(this.startLabel, 'Pause training');
      this.setText(this.startHint, 'Optimising actor and critic');
      this.startButton.setAttribute('aria-label', 'Pause training');
      this.setText(this.phaseLabel, 'OPTIMISING');
      this.setText(this.phaseDetail, `${trainer.ppo.stats.samples ?? 0} samples on policy`);
    } else {
      this.setText(this.runStatus, 'ROLLOUT LIVE');
      this.setText(
        this.statusMeta,
        `Cohort ${String(trainer.generation).padStart(2, '0')} · ${speed}× realtime`,
      );
      this.setText(this.startLabel, 'Pause training');
      this.setText(this.startHint, `Collecting rollout · ${trainer.agents.length} parallel worms`);
      this.startButton.setAttribute('aria-label', 'Pause training');
      this.setText(this.phaseLabel, 'EXPLORING');
      this.setText(this.phaseDetail, `${trainer.aliveCount} organisms active`);
    }

    this.idlePrompt.classList.toggle('is-hidden', !ready);

    const generation = String(trainer.generation).padStart(2, '0');
    this.setText(this.genNumber, generation);
    this.setText(this.genTime, formatTimer(trainer.elapsed));
    const survival = intermission
      ? 0
      : (trainer.aliveCount / Math.max(1, trainer.agents.length)) * 100;
    this.genProgress.style.width = `${survival.toFixed(2)}%`;

    const alive = trainer.agents.filter((agent) => agent.worm.alive).length;
    const average =
      trainer.agents.reduce((total, agent) => total + agent.worm.fitness, 0) /
      Math.max(1, trainer.agents.length);
    const currentBest = trainer.ranking[0]?.worm.fitness ?? 0;
    const population = trainer.agents.length;
    this.setText(this.aliveCount, String(alive));
    this.setText(this.aliveTotal, `/${population}`);
    this.setText(this.foodCount, String(trainer.ecology.activeFoodCount()));
    this.setText(this.avgFit, formatInteger(intermission ? trainer.lastSummary?.average : average));
    this.setText(
      this.bestFit,
      formatInteger(intermission ? trainer.lastSummary?.best : currentBest),
    );

    if (leader) {
      const worm = leader.worm;
      const network = leader.network;
      this.leaderCard.style.setProperty('--agent', leader.color);
      this.setText(this.leaderName, `WORM ${String(leader.index).padStart(2, '0')}`);
      this.setText(
        this.leaderPolicy,
        `PPO ${network.stats.signature.replaceAll(' · ', '·')} · C${String(
          trainer.generation,
        ).padStart(2, '0')}`,
      );
      this.setText(this.leaderScore, formatInteger(worm.fitness));
      this.setText(this.leaderFood, String(worm.foodEaten));
      this.setText(this.leaderForward, formatInteger(worm.forwardProgress));
      this.setText(
        this.leaderStraight,
        `${Math.round(
          (worm.straightTravel / Math.max(1, worm.distanceTravelled)) * 100,
        )}%`,
      );
      this.setText(this.leaderTime, `${worm.age.toFixed(0)}s`);
      const energyPercent = Math.round((worm.energy / worm.maxEnergy) * 100);
      this.setText(this.leaderEnergy, `${energyPercent}%`);
      this.leaderEnergyBar.style.width = `${energyPercent}%`;

      this.setText(this.policySignature, `PPO ${network.stats.signature.replaceAll(' · ', ' · ')}`);
      this.setText(this.brainLinkCount, `${network.connections.length} WEIGHTS`);
      this.setText(this.hiddenUnits, String(network.stats.hidden));
      this.setText(this.weightCount, String(network.stats.enabled));
      this.setText(
        this.learningRate,
        `${(trainer.ppo.stats.learningRate ?? 0).toExponential(1).replace('e-', 'e−')}`,
      );
      this.setText(this.ppoIteration, String(trainer.iteration).padStart(3, '0'));
      this.setText(this.ppoEntropy, formatFraction(trainer.ppo.stats.entropy ?? 0));
      this.setText(this.ppoLoss, formatFraction(trainer.ppo.stats.policyLoss ?? 0));

      this.setText(this.turnValue, formatFraction(worm.outputs.turn));
      this.setText(this.thrustValue, formatFraction(worm.outputs.thrust));
      const turn = Math.max(-1, Math.min(1, worm.outputs.turn));
      const turnWidth = Math.abs(turn) * 50;
      const turnLeft = turn >= 0 ? 50 : 50 - turnWidth;
      this.turnFill.style.left = `${turnLeft}%`;
      this.turnFill.style.width = `${turnWidth}%`;
      this.thrustFill.style.width = `${Math.round(worm.outputs.thrust * 100)}%`;
    }

    this.updateStandings(trainer);

    const lastRecord = trainer.history[trainer.history.length - 1];
    this.setText(
      this.bestEver,
      formatInteger(Math.max(trainer.bestEver, currentBest)),
    );
    this.setText(this.lastChampion, lastRecord ? formatInteger(lastRecord.best) : '—');
    this.setText(this.cohortCount, String(trainer.history.length).padStart(2, '0'));

    this.handleToast(trainer, running);
    this.updateBrain(trainer, now);
    this.updateFitness(trainer, now);
  }

  updateStandings(trainer) {
    const ranking = trainer.ranking.slice(1, 7);
    const key = ranking
      .map((agent) => `${agent.id}:${Math.round(agent.worm.fitness)}`)
      .join('|');
    if (key === this.lastStandings) return;
    this.lastStandings = key;

    this.standings.innerHTML = ranking
      .map(
        (agent, index) => `
          <div class="standing-row" style="--row-agent:${agent.color}">
            <span class="standing-rank">${String(index + 2).padStart(2, '0')}</span>
            <span class="standing-swatch"></span>
            <span class="standing-name">W-${String(agent.index).padStart(2, '0')}<small>${agent.worm.foodEaten} NUT</small></span>
            <span class="standing-food">${agent.worm.age.toFixed(0)}s</span>
            <span class="standing-score">${formatInteger(agent.worm.fitness)}</span>
          </div>
        `,
      )
      .join('');
  }

  handleToast(trainer, running) {
    if (trainer.phase !== 'intermission') {
      this.toast.classList.remove('is-visible');
      return;
    }
    if (this.lastToastGeneration !== trainer.generation) {
      this.lastToastGeneration = trainer.generation;
      const summary = trainer.lastSummary;
      this.setText(
        this.toastTitle,
        `Cohort ${String(trainer.generation).padStart(2, '0')} scored`,
      );
      this.setText(
        this.toastDetail,
        `Champion ${formatInteger(summary?.best)} fit · ${summary?.foodEaten ?? 0} nutrients · ${Math.round(
          summary?.timeAlive ?? 0,
        )}s alive${running ? ' · policy updated' : ' · on hold'}`,
      );
    }
    this.toast.classList.add('is-visible');
  }

  updateBrain(trainer, now) {
    if (now - this.lastBrainDraw < 74) return;
    this.lastBrainDraw = now;
    const leader = trainer.leader;
    if (!leader) return;
    const traceKey = `${trainer.generation}:${leader.id}`;
    this.drawBrain(leader, now / 1000, traceKey);
  }

  drawBrain(agent, time, traceKey) {
    const ctx = this.brainContext;
    const { width, height } = canvasSize(this.brainCanvas, ctx);
    ctx.clearRect(0, 0, width, height);

    const worm = agent.worm;
    const network = agent.network;
    const layerTop = 9;
    const networkTop = 20;
    const traceTop = height - 31;
    const networkBottom = traceTop - 8;
    const inputX = width * 0.15;
    const outputX = width * 0.87;
    const groupGap = Math.max(2, height * 0.012);
    const inputLayout = network.inputLayout;
    const inputGapAfter = new Set();
    for (let index = 1; index < inputLayout.length; index += 1) {
      const previousKind = inputLayout[index - 1].kind;
      const kind = inputLayout[index].kind;
      if (kind !== previousKind && !(previousKind === 'energy' && kind === 'bias')) {
        inputGapAfter.add(index - 1);
      }
    }
    const rowHeight =
      (networkBottom - networkTop - groupGap * inputGapAfter.size) /
      inputLayout.length;
    const inputLabelFont = `600 ${Math.max(
      5.8,
      Math.min(8.4, rowHeight * 0.92),
    ).toFixed(1)}px "JetBrains Mono Variable", monospace`;
    const inputY = [];
    let cursor = networkTop + rowHeight * 0.5;
    for (let index = 0; index < inputLayout.length; index += 1) {
      inputY.push(cursor);
      cursor += rowHeight;
      if (inputGapAfter.has(index)) cursor += groupGap;
    }
    const hiddenCount = network.hiddenLayout.length;
    const hiddenDepth = network.hiddenDepth;
    const span = outputX - inputX;
    const hiddenLeft = inputX + span * 0.36;
    const hiddenRight = inputX + span * 0.66;
    const hiddenColumnX =
      hiddenDepth === 1
        ? [(hiddenLeft + hiddenRight) * 0.5]
        : Array.from(
            { length: hiddenDepth },
            (_, layer) =>
              hiddenLeft + (layer / (hiddenDepth - 1)) * (hiddenRight - hiddenLeft),
          );
    const hiddenTop = networkTop + rowHeight * 0.5;
    const hiddenBottom = networkBottom - rowHeight * 0.5;
    const hiddenRowY = (layer, row) => {
      const count = network.hiddenLayers[layer]?.length ?? 1;
      if (count === 1) return (hiddenTop + hiddenBottom) * 0.5;
      return hiddenTop + (row / (count - 1)) * (hiddenBottom - hiddenTop);
    };
    const outputY = [
      networkTop + (networkBottom - networkTop) * 0.36,
      networkTop + (networkBottom - networkTop) * 0.66,
    ];

    const inputs = inputLayout.map((node, index) => {
      const value = worm.observation?.[index] ?? worm.sensorValues[index] ?? 0;
      return {
        ...node,
        index,
        point: { x: inputX, y: inputY[index] },
        value,
        color: inputColor(node.kind),
        displayLabel: node.kind === 'bias' ? 'BIAS' : node.label,
      };
    });
    const hidden = network.hiddenLayout.map((node, index) => {
      const value = worm.activations.hidden[index] ?? 0;
      return {
        ...node,
        index,
        point: { x: hiddenColumnX[node.layer], y: hiddenRowY(node.layer, node.row) },
        value,
        color: value >= 0 ? '#69efd1' : '#ff7fa6',
      };
    });
    const outputs = network.outputLayout.map((node, index) => ({
      ...node,
      index,
      point: { x: outputX, y: outputY[index] },
      value: worm.activations.outputs[index] ?? 0,
      displayValue: index === 0 ? worm.outputs.turn : worm.outputs.thrust,
      color: index === 0 ? '#69efd1' : '#c8ff6d',
    }));

    const connections = network.connections.map((connection, order) => {
      const source =
        connection.fromKind === 'input'
          ? inputs[connection.from]
          : hidden[connection.from];
      const target =
        connection.toKind === 'hidden'
          ? hidden[connection.to]
          : outputs[connection.to];
      return {
        ...connection,
        source,
        target,
        order,
        signal: source && target ? Math.abs(source.value * connection.weight) : 0,
      };
    }).filter((connection) => connection.source && connection.target);

    // Group the mesh by layer transition so every hop gets its own budget of
    // highlighted links — otherwise the first hop swallows the whole panel.
    const hops = [];
    for (const connection of connections) {
      const index = connection.hop ?? 0;
      if (!hops[index]) {
        hops[index] = {
          color: HOP_COLORS[index % HOP_COLORS.length],
          connections: [],
          maxSignal: 0.001,
        };
      }
      const hop = hops[index];
      hop.connections.push(connection);
      if (connection.signal > hop.maxSignal) hop.maxSignal = connection.signal;
    }

    const activeConnections = connections.filter((connection) => connection.signal > 0.015);
    const meanHidden = hidden.length
      ? hidden.reduce((total, node) => total + Math.abs(node.value), 0) / hidden.length
      : 0;

    ctx.save();
    ctx.strokeStyle = 'rgba(151, 255, 201, 0.06)';
    ctx.lineWidth = 1;
    for (const index of inputGapAfter) {
      const separatorY = inputY[index] + rowHeight * 0.5 + groupGap * 0.5;
      ctx.beginPath();
      ctx.moveTo(4, separatorY);
      ctx.lineTo(inputX - 6, separatorY);
      ctx.stroke();
    }
    ctx.restore();

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineWidth = 0.4;
    for (const hop of hops) {
      ctx.strokeStyle = `rgba(${hop.color}, 0.026)`;
      ctx.beginPath();
      for (const connection of hop.connections) {
        ctx.moveTo(connection.source.point.x, connection.source.point.y);
        ctx.lineTo(connection.target.point.x, connection.target.point.y);
      }
      ctx.stroke();
    }

    for (const hop of hops) {
      const lit = hop.connections
        .filter((connection) => connection.signal > 0.015)
        .sort((a, b) => a.signal - b.signal)
        .slice(-LIT_PER_HOP);
      for (const connection of lit) {
        const normalized = Math.min(1, connection.signal / hop.maxSignal);
        const alpha = 0.05 + normalized ** 0.7 * 0.3;
        ctx.strokeStyle = `rgba(${hop.color}, ${alpha})`;
        ctx.lineWidth = 0.5 + normalized * 1.15;
        ctx.setLineDash(connection.weight < 0 ? [1.6, 1.8] : []);
        ctx.beginPath();
        ctx.moveTo(connection.source.point.x, connection.source.point.y);
        ctx.lineTo(connection.target.point.x, connection.target.point.y);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    for (const hop of hops) {
      const pulsing = hop.connections
        .filter((connection) => connection.signal > 0.015)
        .sort((a, b) => a.signal - b.signal)
        .slice(-PULSE_PER_HOP);
      for (const connection of pulsing) {
        const normalized = Math.min(1, connection.signal / hop.maxSignal);
        const phase =
          (time * (0.22 + normalized * 0.52) + connection.order * 0.137) % 1;
        const px =
          connection.source.point.x +
          (connection.target.point.x - connection.source.point.x) * phase;
        const py =
          connection.source.point.y +
          (connection.target.point.y - connection.source.point.y) * phase;
        const alpha = Math.sin(phase * Math.PI) * (0.14 + normalized * 0.66);
        ctx.fillStyle = `rgba(${hop.color}, ${alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, 0.9 + normalized * 1.25, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();

    this.drawLabel(
      ctx,
      `SENSE ${inputLayout.length}`,
      4,
      layerTop,
      'rgba(137, 177, 157, 0.72)',
      'left',
      '600 7.6px "JetBrains Mono Variable", monospace',
    );
    const candidateCount = network.hiddenCandidates.length;
    this.drawLabel(
      ctx,
      candidateCount > 0
        ? `HIDDEN ${hiddenDepth}L · ${hiddenCount} UNITS`
        : 'HIDDEN 0L',
      (hiddenColumnX[0] + hiddenColumnX[hiddenColumnX.length - 1]) * 0.5,
      layerTop,
      'rgba(137, 177, 157, 0.72)',
      'center',
      '600 7.6px "JetBrains Mono Variable", monospace',
    );
    for (let layer = 0; layer < hiddenDepth; layer += 1) {
      const count = network.hiddenLayers[layer]?.length ?? 0;
      this.drawLabel(
        ctx,
        `L${layer + 1}·${count}`,
        hiddenColumnX[layer],
        networkTop - 3,
        count > 0 ? 'rgba(105, 239, 209, 0.58)' : 'rgba(105, 239, 209, 0.24)',
        'center',
        '600 6.5px "JetBrains Mono Variable", monospace',
      );
    }
    this.drawLabel(
      ctx,
      'MOTOR 2',
      outputX,
      layerTop,
      'rgba(137, 177, 157, 0.72)',
      'center',
      '600 7.6px "JetBrains Mono Variable", monospace',
    );

    for (const input of inputs) {
      const value = Math.max(0, Math.min(1, input.value));
      this.drawLabel(
        ctx,
        input.displayLabel,
        4,
        input.point.y,
        withAlpha(input.color, 0.56 + value * 0.38),
        'left',
        inputLabelFont,
      );
      this.drawNode(
        ctx,
        input.point.x,
        input.point.y,
        2 + value * 2.5,
        input.color,
        0.3 + value * 0.62,
      );
    }

    for (const node of hidden) {
      const intensity = Math.min(1, Math.abs(node.value));
      this.drawNode(
        ctx,
        node.point.x,
        node.point.y,
        2.3 + intensity * 2.1,
        node.color,
        0.36 + intensity * 0.52,
      );
      const bar = intensity * 11;
      ctx.save();
      ctx.strokeStyle = withAlpha(node.color, 0.28 + intensity * 0.42);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.moveTo(node.point.x + 4.2, node.point.y);
      ctx.lineTo(node.point.x + 4.2 + bar, node.point.y);
      ctx.stroke();
      ctx.restore();
    }

    for (let layer = 0; layer < hiddenDepth; layer += 1) {
      const units = network.hiddenLayers[layer] ?? [];
      let peak = null;
      for (const unit of units) {
        const value = worm.activations.hidden[unit.index] ?? 0;
        if (!peak || Math.abs(value) > Math.abs(peak.value)) peak = { unit, value };
      }
      if (!peak) continue;
      const point = hidden[peak.unit.index]?.point;
      if (!point) continue;
      this.drawLabel(
        ctx,
        formatSigned(peak.value),
        point.x - 6,
        point.y,
        withAlpha(peak.value >= 0 ? '#69efd1' : '#ff7fa6', 0.92),
        'right',
        '600 7.4px "JetBrains Mono Variable", monospace',
      );
    }

    for (const output of outputs) {
      const magnitude = Math.min(1, Math.abs(Math.tanh(output.value)));
      const radius = 4.4 + magnitude * 2.2;
      this.drawNode(ctx, output.point.x, output.point.y, radius, output.color, 0.75);
      ctx.save();
      ctx.strokeStyle = withAlpha(output.color, 0.26 + magnitude * 0.24);
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.arc(output.point.x, output.point.y, radius + 2.4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      this.drawLabel(
        ctx,
        output.label,
        output.point.x,
        output.point.y - 10,
        withAlpha(output.color, 0.72),
        'center',
        '600 7.8px "JetBrains Mono Variable", monospace',
      );
      this.drawLabel(
        ctx,
        output.index === 0 ? formatSigned(output.displayValue) : formatFraction(output.displayValue),
        output.point.x,
        output.point.y + 11,
        output.color,
        'center',
        '600 9.6px "JetBrains Mono Variable", monospace',
      );
    }

    if (this.brainTrace.key !== traceKey) {
      this.brainTrace = { key: traceKey, turn: [], thrust: [] };
    }
    this.brainTrace.turn.push(worm.outputs.turn);
    this.brainTrace.thrust.push(worm.outputs.thrust);
    if (this.brainTrace.turn.length > 72) this.brainTrace.turn.shift();
    if (this.brainTrace.thrust.length > 72) this.brainTrace.thrust.shift();

    const traceLeft = 38;
    const traceWidth = width - traceLeft - 4;
    const turnLaneY = traceTop + 8;
    const thrustLaneY = traceTop + 20;
    ctx.save();
    ctx.strokeStyle = 'rgba(151, 255, 201, 0.09)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(4, traceTop - 3);
    ctx.lineTo(width - 4, traceTop - 3);
    ctx.stroke();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = 'rgba(151, 255, 201, 0.12)';
    ctx.beginPath();
    ctx.moveTo(traceLeft, turnLaneY);
    ctx.lineTo(width - 4, turnLaneY);
    ctx.moveTo(traceLeft, thrustLaneY + 3);
    ctx.lineTo(width - 4, thrustLaneY + 3);
    ctx.stroke();
    ctx.setLineDash([]);
    this.drawLabel(
      ctx,
      'TURN',
      4,
      turnLaneY,
      'rgba(105, 239, 209, 0.78)',
      'left',
      '600 7.4px "JetBrains Mono Variable", monospace',
    );
    this.drawLabel(
      ctx,
      'THRUST',
      4,
      thrustLaneY,
      'rgba(200, 255, 109, 0.76)',
      'left',
      '600 7.4px "JetBrains Mono Variable", monospace',
    );

    const drawTrace = (values, yFor, color) => {
      if (!values.length) return;
      ctx.beginPath();
      values.forEach((value, index) => {
        const x =
          traceLeft + (index / Math.max(1, values.length - 1)) * traceWidth;
        const y = yFor(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.shadowColor = color;
      ctx.shadowBlur = 4;
      ctx.stroke();
      ctx.shadowBlur = 0;
      const lastX =
        traceLeft +
        ((values.length - 1) / Math.max(1, values.length - 1)) * traceWidth;
      const lastY = yFor(values[values.length - 1]);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 1.5, 0, Math.PI * 2);
      ctx.fill();
    };
    drawTrace(
      this.brainTrace.turn,
      (value) => turnLaneY - Math.max(-1, Math.min(1, value)) * 3.3,
      'rgba(105, 239, 209, 0.78)',
    );
    drawTrace(
      this.brainTrace.thrust,
      (value) => thrustLaneY + 3 - Math.max(0, Math.min(1, value)) * 5.6,
      'rgba(200, 255, 109, 0.74)',
    );
    ctx.restore();

    const dominant = inputs
      .filter((input) =>
        ['food', 'hazard', 'worm', 'chomper'].includes(input.kind),
      )
      .reduce((best, input) => (input.value > best.value ? input : best));
    this.setText(this.brainDominant, dominant.label);
    this.setText(this.brainDominantValue, formatFraction(dominant.value));
    this.brainDominant.style.color = dominant.value > 0.01 ? dominant.color : '';
    this.setText(this.brainHiddenMean, formatFraction(meanHidden));
    this.setText(this.brainActiveLinks, String(activeConnections.length));
    this.setText(this.brainTotalLinks, `/${connections.length}`);
    this.setText(
      this.brainArchitecture,
      hiddenDepth > 0
        ? `${inputLayout.length} → ${hiddenDepth}L · ${hiddenCount}H → 2`
        : `${inputLayout.length} → 0H → 2`,
    );
    this.brainArchitecture.title = `${hiddenDepth} hidden layers of ${network.stats.hiddenSizes.join(' and ')} units`;
    this.brainCanvas.setAttribute(
      'aria-label',
      `Live actor-critic network. ${hiddenDepth} hidden layers with ${hiddenCount} units. Strongest sensor ${dominant.label} at ${formatFraction(
        dominant.value,
      )}. Mean hidden activation ${formatFraction(
        meanHidden,
      )}. ${activeConnections.length} of ${connections.length} weights carrying signal.`,
    );
  }

  drawNode(ctx, x, y, radius, color, alpha) {
    const glow = ctx.createRadialGradient(x, y, 0, x, y, radius * 4.4);
    glow.addColorStop(0, color);
    glow.addColorStop(0.18, `${color}aa`);
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.globalAlpha = alpha * 0.28;
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, radius * 4.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(1, 10, 7, 0.7)';
    ctx.lineWidth = 0.7;
    ctx.stroke();
  }

  drawLabel(
    ctx,
    text,
    x,
    y,
    color,
    align = 'left',
    font = '500 6.2px "JetBrains Mono Variable", monospace',
  ) {
    ctx.save();
    ctx.font = font;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(2, 9, 6, 0.82)';
    ctx.lineWidth = 1.7;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  updateFitness(trainer, now) {
    const signature = trainer.history
      .map((entry) => `${entry.generation}:${entry.best.toFixed(1)}`)
      .join('|');
    if (signature === this.lastFitnessSignature) return;
    this.lastFitnessSignature = signature;
    this.drawFitness(trainer.history);
  }

  drawFitness(history) {
    const ctx = this.fitnessContext;
    const { width, height } = canvasSize(this.fitnessCanvas, ctx);
    ctx.clearRect(0, 0, width, height);

    const padding = { left: 24, right: 22, top: 10, bottom: 18 };
    const plotWidth = Math.max(10, width - padding.left - padding.right);
    const plotHeight = Math.max(10, height - padding.top - padding.bottom);
    const x = (index, length) =>
      padding.left +
      (length <= 1 ? plotWidth * 0.5 : (index / (length - 1)) * plotWidth);

    ctx.save();
    ctx.strokeStyle = 'rgba(151, 255, 201, 0.07)';
    ctx.lineWidth = 1;
    ctx.font = '500 6.4px "JetBrains Mono Variable", monospace';
    ctx.fillStyle = 'rgba(92, 125, 109, 0.7)';
    for (let index = 0; index <= 3; index += 1) {
      const y = padding.top + (plotHeight / 3) * index;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }
    ctx.restore();

    if (!history.length) {
      ctx.save();
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = 'rgba(105, 239, 209, 0.22)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top + plotHeight * 0.6);
      ctx.bezierCurveTo(
        width * 0.32,
        padding.top + plotHeight * 0.68,
        width * 0.68,
        padding.top + plotHeight * 0.42,
        width - padding.right,
        padding.top + plotHeight * 0.5,
      );
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(112, 143, 127, 0.72)';
      ctx.font = '500 6.5px "JetBrains Mono Variable", monospace';
      ctx.fillText('AWAITING COHORT 01', padding.left + 4, padding.top + 10);
      ctx.restore();
      return;
    }

    const bestMax = Math.max(100, ...history.map((entry) => entry.best));
    const meanMax = Math.max(100, ...history.map((entry) => entry.average));
    const niceBestMax = Math.ceil((bestMax * 1.12) / 100) * 100;
    const niceMeanMax = Math.ceil((meanMax * 1.12) / 50) * 50;
    const yBest = (value) =>
      padding.top + plotHeight - Math.min(1, value / niceBestMax) * plotHeight;
    const yMean = (value) =>
      padding.top + plotHeight - Math.min(1, value / niceMeanMax) * plotHeight;

    ctx.save();
    ctx.font = '500 6.4px "JetBrains Mono Variable", monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(200, 255, 109, 0.62)';
    ctx.fillText(formatInteger(niceBestMax), padding.left - 5, padding.top + 3);
    ctx.fillText(formatInteger(niceBestMax / 2), padding.left - 5, padding.top + plotHeight / 2 + 3);
    ctx.fillText('0', padding.left - 5, padding.top + plotHeight + 2);
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(105, 239, 209, 0.62)';
    ctx.fillText(
      formatInteger(niceMeanMax),
      width - padding.right + 5,
      padding.top + 3,
    );
    ctx.fillText(
      formatInteger(niceMeanMax / 2),
      width - padding.right + 5,
      padding.top + plotHeight / 2 + 3,
    );
    ctx.fillText('0', width - padding.right + 5, padding.top + plotHeight + 2);
    const tickCount = Math.min(4, history.length);
    for (let index = 0; index < tickCount; index += 1) {
      const historyIndex =
        tickCount === 1 ? 0 : Math.round((index / (tickCount - 1)) * (history.length - 1));
      const entry = history[historyIndex];
      const tickX = x(historyIndex, history.length);
      ctx.fillStyle = 'rgba(92, 125, 109, 0.62)';
      ctx.fillText(`C${String(entry.generation).padStart(2, '0')}`, tickX - 4, height - 4);
    }
    ctx.restore();

    const bestPoints = history.map((entry, index) => ({
      x: x(index, history.length),
      y: yBest(entry.best),
    }));
    const averagePoints = history.map((entry, index) => ({
      x: x(index, history.length),
      y: yMean(entry.average),
    }));

    const area = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotHeight);
    area.addColorStop(0, 'rgba(200, 255, 109, 0.14)');
    area.addColorStop(1, 'rgba(200, 255, 109, 0)');
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(bestPoints[0].x, padding.top + plotHeight);
    for (const point of bestPoints) ctx.lineTo(point.x, point.y);
    ctx.lineTo(bestPoints[bestPoints.length - 1].x, padding.top + plotHeight);
    ctx.closePath();
    ctx.fillStyle = area;
    ctx.fill();

    ctx.beginPath();
    averagePoints.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = 'rgba(105, 239, 209, 0.64)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    bestPoints.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = '#c8ff6d';
    ctx.lineWidth = 1.3;
    ctx.shadowColor = 'rgba(200, 255, 109, 0.65)';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.shadowBlur = 0;

    const last = bestPoints[bestPoints.length - 1];
    ctx.fillStyle = '#c8ff6d';
    ctx.beginPath();
    ctx.arc(last.x, last.y, 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(200, 255, 109, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}
