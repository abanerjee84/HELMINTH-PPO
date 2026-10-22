<div align="center">

# HELMINTH

**A PPO ecology for worms that forage, sense rivals, and flee chompers.**

A browser-native 2D artificial-life experiment: ten sensory worms share one Proximal
Policy Optimization actor-critic, learn to find nutrient blooms in a toxic arena, and
get hunted by three small chompers. The policy is written from scratch on typed arrays —
no PyTorch, no TensorFlow, no ML dependency at all.

[![assessment](https://img.shields.io/badge/assessment-21%2F21-passing-b9ff6a?style=flat-square)](assess/visual-assessment.mjs)
[![license](https://img.shields.io/badge/license-MIT-c8ff6d?style=flat-square)](LICENSE)
[![stack](https://img.shields.io/badge/stack-vite%20%2B%20three.js%20%2B%20vanilla%20JS-69efd1?style=flat-square)](package.json)

<p align="center">
  <img src="docs/screenshots/02-rollout-live.png" alt="HELMINTH running: ten worms foraging in a dark arena while three chompers patrol" width="100%">
</p>

</div>

---

## Table of contents

- [Highlights](#highlights)
- [Screenshots](#screenshots)
- [Quick start](#quick-start)
- [How a cohort works](#how-a-cohort-works)
- [The policy](#the-policy)
- [The PPO update](#the-ppo-update)
- [Does it actually learn?](#does-it-actually-learn)
- [Two bugs worth reading about](#two-bugs-worth-reading-about)
- [Can the worms see the wall?](#can-the-worms-see-the-wall)
- [Multi-agent notes](#multi-agent-notes)
- [The arena](#the-arena)
- [The sensorium](#the-sensorium)
- [Reward and fitness](#reward-and-fitness)
- [The chompers](#the-chompers)
- [Project layout](#project-layout)
- [Verification](#verification)
- [Notes and constraints](#notes-and-constraints)

---

## Highlights

- **PPO from scratch** — Gaussian actor, separate value critic, clipped surrogate
  objective, clipped value loss, entropy bonus, global gradient-norm clipping, and
  Adam, all implemented on `Float32Array`s with hand-written forward and backward
  passes. 2 765 parameters, ~110 ms for a full update.
- **Parallel on-policy rollouts** — all ten worms are sampled in lockstep, so one
  policy update consumes a full cohort of trajectories. A cohort ends only when the
  last worm in it is dead.
- **28-channel sensorium** — 8 food rays, 4 hazard rays, 4 wall rays, 4 rival-worm
  rays, 4 chomper rays, 2 self-body rays, energy, and bias, plus a 2-channel per-worm
  drift vector.
- **Body-aware worms** — 15-segment bodies, self-collision, worm-to-worm collision,
  wall response, and lethal energy loss.
- **Chomper patrol** — three predators that wander, chase, lunge, bite, digest, and
  retreat.
- **Non-blocking optimization** — the policy update is spread across frames on a 6 ms
  budget, so the arena never freezes while the weights are being rewritten.
- **Live neurotelemetry** — the actor graph with per-hop link colouring, hidden-layer
  activations, motor outputs, entropy, policy loss, fitness trace, and live ranking.
- **Evidence-based UI checks** — a Playwright assessment verifies the ecology, the
  24-sensor contract, the PPO update loop, responsive layout, and the absence of page
  errors: **21/21**.

---

## Screenshots

| | |
| --- | --- |
| ![Standby](docs/screenshots/01-standby.png) | ![Cohort scored](docs/screenshots/04-cohort-scored.png) |
| **Standby** — the arena waits with ten worms, 120 blooms, six toxic pockets, and three chompers. | **Cohort scored** — the toast reports the champion's score right after the policy update lands. |
| ![Policy panel](docs/screenshots/05-policy-panel.png) | ![Fitness trace](docs/screenshots/06-fitness-trace.png) |
| **Policy activity** — 26 inputs, two hidden layers of 24 units, two motors. Each layer transition gets its own colour and its own link budget. | **Fitness trace** — cohort best (solid) against cohort mean (dashed). The saw-tooth is cohort noise around a rising trend. |

| | |
| --- | --- |
| ![Arena](docs/screenshots/03-arena.png) | ![Leaderboard](docs/screenshots/07-leaderboard.png) |
| **The arena** — nutrient blooms, toxic pockets, and chompers sharing one 1360×1240 field. | **Leaderboard** — live ranking by cohort-cumulative fitness, with nutrients, distance, straightness, and lifespan. |

![Trained](docs/screenshots/08-trained.png)

**After eight cohorts** the policy has learned to hold a heading instead of curling into
its own tail, and the cohort score has climbed from ~2 600 to ~3 500.

---

## Quick start

### Requirements

- Node.js `^20.19.0` or `>=22.12.0`
- A WebGL-capable browser (Chrome, Edge, Firefox, Safari)

### Run locally

```bash
npm install
npm run dev -- --port 4173 --strictPort
```

Open <http://127.0.0.1:4173/>, press **Start training**, and watch the policy learn.

On Windows, `START.bat` installs dependencies if needed, starts the server on port
4173, and opens the browser.

### Production build

```bash
npm run build        # bundle to dist/
npm run preview      # serve the bundle
```

### Run the visual assessment

```bash
npx playwright install chromium
npm run assess
```

The assessment builds the production bundle, serves it from a temporary static server
on an ephemeral port, drives it with headless Chromium, and writes screenshots plus
`report.json` to `assess/output/` (gitignored). The curated images in this README
live in `docs/screenshots/`.

### Controls

| Control | Effect |
| --- | --- |
| **Start / Pause training** | Runs or freezes the simulation. The policy update keeps working while paused only if it was already in flight. |
| **1× / 4× / 8×** | Simulation rate. The policy update is unaffected — the arena just advances faster between updates. |
| **Reset** | Rebuilds the trainer from the configured seed: a fresh policy, a fresh arena, an empty fitness trace. |

---

## How a cohort works

```mermaid
flowchart LR
    A["Sense<br/>28 sensors + drift"] --> B["Act<br/>actor samples a motor command"]
    B --> C["Move<br/>steering, thrust, body, energy"]
    C --> D["Interact<br/>food, toxins, chompers, bodies"]
    D --> E["Reward<br/>delta of the fitness terms"]
    E --> F["Store<br/>rollout buffer"]
    F --> G["GAE<br/>advantages + returns"]
    G --> H["Update<br/>4 epochs x 256-sample minibatches"]
    H --> I["Next cohort<br/>same arena, better policy"]
```

1. **Sense.** Each living worm samples the 28-channel sensorium and appends its own
   drift vector.
2. **Act.** The shared actor emits a Gaussian mean; the policy samples a
   two-dimensional motor command (`turn`, `thrust`).
3. **Move.** Steering, thrust, speed, body joints, wall contact, and energy drain
   update at a fixed 60 Hz simulation step.
4. **Interact.** Food is consumed, toxic blooms damage worms, worm bodies collide,
   chompers bite, and a wall hit slides the worm along the boundary instead of
   pinning it.
5. **Reward.** The step reward is the exact delta of the fitness terms below, so the
   credit a worm receives is the credit the scorecard shows.
6. **Store.** Every transition lands in the rollout buffer. A worm that dies ends its
   own trajectory (bootstrapping with zero).
7. **Learn.** When the rollout fills (256 steps), GAE turns the buffer into advantages
   and four epochs of clipped minibatches rewrite the actor and the critic.

**A cohort ends when the last worm in it is dead.** There is no timer: a single
survivor keeps the cohort alive, so a cohort score always means "this wave was
wiped out", never "we stopped watching while they were still swimming". The next
cohort spawns into the same arena with the improved policy, and the arena never
freezes during an update — the optimization is spread across frames on a millisecond
budget while the worms keep swimming.

The consequence is visible in the data: an untrained policy loses its wave in about
30 seconds, while a trained one holds on for two minutes.

---

## The policy

| | |
| --- | --- |
| Actor | `30 → 24 → 24 → 2`, tanh hidden layers, linear output |
| Critic | `30 → 24 → 24 → 1`, a **separate** network so a bad value estimate can never poison the policy gradient |
| Parameters | 1 394 actor + 1 369 critic + 2 log-std = **2 765** |
| Action distribution | Diagonal Gaussian with a state-independent log-std |
| Motor mapping | `turn = tanh(1.8·u₀)`, `thrust = σ(2.2·u₁)` |
| Observation | 28 sensors (all in `[0, 1]`) + a 2-channel drift vector |

**The 26th and 27th channels matter.** A single policy driving ten worms makes the
cohort behave like one body: identical headings, mutual collisions, and a lot of
self-inflicted curling. A per-worm random unit vector — re-rolled every cohort — is
appended to the observation, which gives each worm a distinct trajectory from the
first frame. The policy can learn to use it.

**Why the action mean gradient is a score function.** The motor command is a *fixed
sample* `u` of the Gaussian, so

```
∂ log π(u) / ∂ μ = (u − μ) / σ²
```

This is not a pathwise gradient. Getting its sign wrong makes the policy walk away
from good actions; see [Two bugs worth reading about](#two-bugs-worth-reading-about).

---

## The PPO update

**Clipped surrogate** (`ε = 0.2`):

```
L_policy = −E[ min( r·A ,  clip(r, 1−ε, 1+ε)·A ) ],   r = π_θ(u) / π_θold(u)
```

**Clipped value loss** (`v` is the critic, `v̂` the clipped prediction):

```
L_value = ½ · max( (v − R)² , (v̂ − R)² )
```

**Entropy bonus** on the Gaussian policy, so the action distribution does not collapse
onto a single motor command.

**GAE** (`γ = 0.99`, `λ = 0.95`):

```
δ_t     = r_t + γ·V(s_{t+1}) − V(s_t)
A_t     = δ_t + γ·λ·(1 − terminal)·A_{t+1}
R_t     = A_t + V(s_t)
```

| Hyper-parameter | Value | Why |
| --- | --- | --- |
| Discount γ | 0.99 | ≈ 1.7 s of lookahead |
| GAE λ | 0.95 | Low-variance advantage estimates |
| Clip ratio ε | 0.2 | The trust region that makes PPO stable |
| Value clip | 0.2 | The same trust region on the critic |
| Value coefficient | 0.5 | The critic never dominates the shared gradient norm |
| Entropy coefficient | 0.01 | Keeps the action distribution from collapsing |
| Learning rate | 3 × 10⁻⁴ | Adam, with global gradient clipping at 0.5 |
| Epochs × minibatch | 4 × 256 | Four passes over each 2 560-sample rollout |
| Rollout length | 256 steps | ≈ 4.3 s of arena time |
| Initial log-σ | −0.7 | σ ≈ 0.50 at the start of a run |

**Returns are rescaled.** The arena pays 180 for a nutrient bloom, so raw discounted
returns span several orders of magnitude. A running estimate of the return spread
divides the rewards before the critic sees them, otherwise the value loss dominates
the shared gradient norm and the policy gradient is clipped away. The measured value
loss drops from ~700 to ~0.3 with the rescaling in place.

**The update is incremental.** `processBatch` is called a few minibatches at a time
under a 6 ms per-frame budget, so a 40-minibatch update spreads over ~20 frames
instead of blocking the render loop for 110 ms.

---

## Does it actually learn?

Measured over a single 400 000-step run (55 cohorts, one fixed arena):

| Metric | First 6 cohorts | Middle 6 | Last 6 | Change |
| --- | ---: | ---: | ---: | ---: |
| Mean cohort fitness | 1 350 | 1 221 | 4 338 | **+221%** |
| Mean cohort length (seconds alive) | 73 s | 61 s | 172 s | **+136%** |
| Best single cohort | — | — | 38 671 | — |
| Longest single cohort | — | — | 302 s | — |

The second row is the one that matters most: a cohort is only scored once it has
been *wiped out*, so the score measures how long the policy kept its worms alive
against a hostile arena. The last cohorts survive more than twice as long as the
first ones, and the best single cohort scored 38 671 across five minutes of survival.

The same build, before the policy gradient was corrected, *degraded* instead of
improving — the policy learned to stop foraging and die within seconds. That failure
is what the gradient check below was written for.

Reproduce it yourself (exact numbers drift a little with the RNG stream, the trend does not):

```bash
node -e "
import('./src/ppo/trainer.js').then(async ({Trainer}) => {
  const {CONFIG} = await import('./src/config.js');
  const t = new Trainer(CONFIG); t.start();
  const mean = (rows) => rows.reduce((s, r) => s + r.average, 0) / rows.length;
  for (let i = 0; i < 400000; i++) {
    t.step(1/60);
    if (t.phase === 'intermission') t.nextCohort();
    if (i % 600 === 0) t.continueUpdate(100);
  }
  const h = t.history;
  console.log('first6', mean(h.slice(0,6)).toFixed(0), 'last6', mean(h.slice(-6)).toFixed(0));
});
"
```

---

## Two bugs worth reading about

A policy-gradient implementation is easy to get subtly wrong, and both of these bugs
looked like "the agent is just bad at this task". Both were found by comparing the
analytic gradient against a finite-difference estimate of the same loss.

**1. The action-mean gradient had the wrong sign.** The update used `−(u − μ)/σ²`
instead of `+(u − μ)/σ²`, so every positive-advantage action pushed the mean *away*
from the sample that earned the reward. The policy reliably walked into hazards and
stopped foraging; by cohort 80 the arena was empty and every worm died within a
second. The fitness trace declined monotonically.

**2. The clipped branch kept a phantom gradient.** When the importance ratio leaves
`[1−ε, 1+ε]`, the clamp is flat, so `∂ clip(r)/∂r = 0` and the surrogate contributes
nothing. The implementation instead used `−clip(r)·A`, which is a non-zero push from
exactly the samples that should have been ignored.

After the fixes, the check agrees with finite differences to **2.7 × 10⁻⁶** (the
float32 storage precision is the floor):

```
  actor.layer0[0] numeric=-2.6400e-4 analytic=-2.6413e-4 err=1.32e-7
  actor.layer1[0] numeric= 6.2355e-3 analytic= 6.2355e-3 err=5.00e-8
  actor.layer2[0] numeric= 4.1557e-2 analytic= 4.1557e-2 err=2.71e-8
  critic.layer2[0] numeric= 7.5885e-2 analytic= 7.5886e-2 err=1.29e-6
  logStd[0]      numeric= 3.6948e-2 analytic= 3.6948e-2 err=3.00e-7
  worst relative error 2.749e-6 — GRADIENT CHECK PASSED
```

---

## Multi-agent notes

**A shared policy with ten identical agents is a degenerate multi-agent problem.** The
first build inherited the HyperNEAT layout — ten worms, each with its own genome — and
replaced the genome with one shared policy. Every worm then faced the arena with the
same heading policy, so they converged on the arena centre, folded into their own
tails, and collided. Roughly 77 % of all deaths were body collisions.

Two changes fixed it:

1. **Per-worm action noise.** Each worm draws its motor command from its own RNG
   stream, so identical weights do not mean identical trajectories.
2. **A per-worm drift vector** in the observation, which breaks the symmetry
   *systematically* rather than stochastically.

A third change — replacing fallen worms mid-cohort — was tried and then removed: it
kept the arena populated, but it made extinction impossible, and with it the meaning
of a cohort score. A cohort is now a wave: it is scored when the last worm in it is
dead, and a single survivor keeps it running.

---

## The arena

A fixed 1360×1240 field, seeded identically every run, so a cohort's fitness is
comparable to the one before it:

- **120 nutrient blooms** with a 3.8 s respawn, spread across the field;
- **6 toxic pockets** (radius 58–104) that drain 27 energy per second and are placed
  clear of the spawns;
- **3 chompers** that patrol, chase, and bite;
- **10 worms** that start at evenly spaced positions — at least 150 units apart, clear
  of toxic pockets, and kept well away from chomper patrol routes.

Worms start with 58 energy, drain 1.15/s plus a speed term, and cap at 100. The
boundary is 26 units from the edge of the field; a worm that reaches it slides along
it and drifts back into the arena rather than stopping dead.

---

## The sensorium

| Channel | Inputs | Range | What it measures |
| --- | ---: | --- | --- |
| Food | 8 | 330 | Directional alignment and closeness to active nutrient blooms |
| Hazard | 4 | 260 | Toxic-pocket proximity, with a stronger signal inside the bloom |
| Wall | 4 | 210 | Distance to the arena boundary along the front, left, back and right rays |
| Rival worm | 4 | 300 | Direction and closeness to the nearest point on another worm's body |
| Chomper | 4 | 300 | Direction and closeness to a chomper, boosted by chase excitement and active chomping |
| Self body | 2 | — | How tightly the head is folded against its own trailing segments, and which side that body lies on |
| Energy | 1 | — | Current energy normalized to maximum |
| Bias | 1 | — | Constant `1` |
| Drift | 2 | — | A per-worm random unit vector, re-rolled each cohort |

Food uses eight rays around the heading. Hazard, wall, rival, and chomper channels
each use four rays. Rival sensing samples the nearest *body segment* rather than only
the other worm's head, so body contact is represented too. The two self channels let a
worm feel its own body: one reports how far the head has folded back onto its
trailing segments, the other reports which side that body lies on, so a coil can be
unwound before it becomes a lethal self-collision.

---

## Reward and fitness

```text
fitness = max(0,
    foodEaten      × 180
  + age            × 6
  + straightTravel × 0.018
  + forwardProgress× 0.045
  + energy         × 0.35
  - hazardTime     × 3.5
  - radiansTurned  × 2.4
)
```

The PPO step reward is the **delta** of that expression, plus a 25-point death penalty.

`forwardProgress` is the net displacement from the worm's spawn point, not total path
length, so looping in place does not earn the forward reward. `radiansTurned` is the
absolute accumulated steering rotation, so continuous circling carries a cost.
`straightTravel` is path length weighted by directional consistency: each step only
earns travel credit in proportion to how straight the worm has been moving, so a
straight sprint is worth far more per unit than a tight circle. Together these three
terms make circling unprofitable — a worm that loops in place earns neither travel
nor forward credit and pays for every radian it turns.

---

## Can the worms see the wall?

They could not, and the arena was quietly killing them for it.

The first build had no wall channel at all. A boundary was felt only *after* contact:
the head was clamped, the speed halved, and the heading snapped toward the arena
centre. Nothing in the observation said "the edge is over there", so the policy
learned to hug walls and then die there.

The tell was in the death census:

```
collision:        168 (45.4%)   rival worm bodies
self-collision:   117 (31.6%)   ← 115 of these within 40 units of a wall
predation:         47 (12.7%)
toxicosis:         29 (7.8%)
starvation:         9 (2.4%)
direct 'wall' deaths: 0
```

The wall never killed anything by itself. What happened is worse: a worm grazed the
boundary, the head was pinned against it, the trailing body kept advancing into the
pinned head, and the worm folded onto itself. **98 % of wall-adjacent self-collisions
happened within one second of a wall hit** — the wall was the trigger, not the
location.

Two changes fixed it:

1. **A wall channel.** Four heading-relative rays (front, left, back, right) report
   the distance to the boundary within 210 units, so the policy can see the edge
   before it touches it. This is worth roughly 10 % of the final cohort score once
   the trap below is gone (paired across four seeds, it wins 3/4 on both score and
   survival time).
2. **A slide instead of a pin.** A wall hit now redirects the head along the boundary
   tangent, with a gentle drift back into the arena, instead of stopping it dead. The
   body stays in a line behind the head instead of piling into it.

The second change is the one that matters:

```
before:  self-collision deaths 117   total deaths 370
after:   self-collision deaths   3   total deaths 229
```

Self-collision deaths fell by 97 %, and cohorts that used to end after 30 seconds now
run for minutes — the longest measured cohort survived **302 seconds** before the
chompers finished it.

---

## The chompers

The three chompers are deliberately simple and readable:

- cruise toward a wandering target when no worm is nearby;
- lock onto the nearest living worm within 270 units;
- accelerate into a lunge inside roughly 42 % of their sensing range;
- bite for 18 energy on contact, then digest and cool down;
- retreat briefly after a successful bite.

Their bodies, googly eyes, teeth, and `CHOMP!` bubbles are rendered in the Three.js
arena. Chompers have their own sensor channel rather than being folded into generic
hazard data.

---

## Project layout

```text
src/
  config.js            Simulation, fitness, predator, sensor, and PPO constants
  main.js              App loop, controls, debug hooks
  ppo/
    network.js         Dense/MLP primitives, Gaussian actor-critic, graph description
    rollout.js         Parallel rollout buffer, GAE, return statistics
    optim.js           Adam with global gradient-norm clipping
    ppo.js             Clipped surrogate, clipped value loss, entropy, minibatch updates
    trainer.js         Cohort lifecycle, rollout collection, update scheduling
  sim/
    worm.js            Sensors, policy actions, movement, energy, bites, step reward
    predators.js       Chomper pack, chase/lunge/chomp state machine
    ecology.js         Food, toxic blooms, hazard and spawn placement
    collisions.js      Worm body collisions and death resolution
    random.js          Seeded RNG and math helpers
  render/
    arena.js           Three.js arena, worm visuals, chomper visuals, effects
  ui/
    dashboard.js       Live telemetry, policy graph, fitness trace
index.html             App shell and copy
START.bat              Windows one-click launcher
assess/                Playwright visual and behavioral assessment
docs/screenshots/      Curated README images
```

---

## Verification

`npm run assess` currently passes **21/21** checks, including:

- WebGL arena geometry and visible motion;
- three active chompers;
- 28 sensors with food, hazard, wall, worm, chomper, and self-body channels;
- a PPO update running on the collected rollout, with finite telemetry;
- policy weights actually changing under the optimizer;
- cohort scoring and advancement to the next cohort;
- cohort termination on extinction — never on a timer with worms still alive;
- pause/resume behavior;
- rendered fitness trace;
- 1440×900 compact layout with no clipped panels or page overflow;
- no uncaught page errors.

The policy gradient is additionally verified against finite differences, and the
learner is verified end-to-end by the measured cohort curve above.

---

## Notes and constraints

- The arena is a fixed, seeded field, so a run is reproducible: the same seed
  produces the same blooms, pockets, chomper routes, and starting positions.
- Mortality is real and lethal — worm-to-worm and self-collisions both kill — and an
  untrained policy curls into its own tail quickly. Watching that mortality fall as
  the policy improves is the point.
- There is no backend and no persistence. Reset restarts the run from the configured
  seed.
- The simulation is tuned for desktop; the right-side telemetry panel is hidden below
  1040 px.
- The assessment screenshots are generated artifacts. The checked-in
  `docs/screenshots/` copies are the stable README images.

---

<div align="center">

**HELMINTH** — written as a study in replacing an evolutionary search loop with a
policy gradient, and in how easily the latter can be wrong.

</div>
