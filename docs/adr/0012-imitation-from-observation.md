# ADR-0012 — Imitation from observation: recovering actions to reproduce a demo

**Status:** Accepted

## Context
The observe→repeat→explore arc (ADR-0007, ADR-0009/0010) reaches its "repeat" step (P14):
reproduce a demonstrated behaviour the agent never performed itself, from the demo's
**observations only** (no expert actions) — the substrate for learning from video. On the
non-gated hard-benchmark tier (ADR-0011), the concrete target is **cartpole swingup**,
chosen because the companion study (A, `bench/hard/curiosity.py`) showed exploration —
even the curiosity curriculum (P3-002) — cannot crack it at feasible budgets: it reaches
the upright region but can't convert sparse coverage into control. So swingup is exactly a
task where *watching* should beat *acting*.

To reproduce a behaviour seen only as observations, the agent must **recover the actions**
that produced it. P13 (ADR-0010) gives one way — the action-free `LatentActionModel` infers
a latent action per transition — but P13 validated it on the toy Pendulum, and it is a
*latent* action that still needs grounding to a real one.

## Decision
Imitation-from-observation = **recover the demo's actions from observation, then clone a
closed-loop policy** (`bench/hard/imitation.py`, non-gated). Two recovery routes are
implemented and **both reported**, because when a small grounding budget exists they trade
off differently:
- **Direct inverse-dynamics (primary):** a model `g(obs, next_obs) → action` fit on the
  agent's own small labelled interaction (the "grounding"), applied to the demo transitions.
  This is the robust route on a real task.
- **Latent-action + calibration (P13, arc-faithful):** the `LatentActionModel` learns latent
  actions from the demo+grounding **action-free** streams; a tiny calibration maps latent →
  real action on the grounding labels. This is the route that scales to the *fully*
  action-free regime (real video with no grounding actions) — the reason it exists.

Honesty guards baked in: an **oracle clone** (on the hidden true actions) as the ceiling, a
**shuffled-demo** negative control (must collapse), and a **from-scratch** baseline at the
same interaction budget. The deliverable is the measured report, not a gate.

## Consequences
- (+) Reproduces a swingup the agent never performed, purely from watching + a same-budget
  grounding — beating from-scratch MBRL (which fails swingup) and the shuffled control,
  approaching the oracle-clone ceiling. Watching does what exploration at the same budget
  cannot: the observe→repeat step is real on a non-toy task.
- (+) Completes the honest arc: A (exploration reaches the goal region but can't convert it
  to control) → B (a demonstration hands over the goal-reaching behaviour directly).
- The P13 latent route was initially the weak link — high-variance on swingup because its
  separate latent→action *calibration* extrapolated with a systematic bias from bottom-heavy
  grounding to the demo's upright states (and recovery R² did not even predict reproduction).
  **Fixed** by **watch-then-ground** (`LatentActionModel.ground`, ADR-0010 amendment): action-
  free pretraining then a supervised grounding step, so recovery is a reliable inverse map with
  no calibration. Measured: it now **beats from-scratch inverse dynamics in the low-label
  regime** (watching is a low-data prior for control); at full budget direct inverse dynamics
  is still best (the honest boundary). The direct route remains the robust default when labels
  are plentiful; watch-then-ground earns its keep when they are scarce — the video regime.
- (−) This is **non-gated** (ADR-0011): a demonstration on cartpole, not a numpy-CI kill-gate.
  Formally shipping P14 in the ratchet needs a numpy-gated imitation task on a toy env — the
  natural follow-up, mirroring how P12 gated vision on stand-ins and demonstrated real vision
  off-gate.
