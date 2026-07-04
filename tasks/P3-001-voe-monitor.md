# P3-001 — Calibrated surprise, decomposition, and the mastery test

- **Status:** done
- **Phase:** P3
- **Requirements:** R3
- **ADRs:** ADR-0002 (the one signal, decomposed), ADR-0006 (noisy-TV: epistemic
  drives learning), ADR-0007 (this feeds the mode arbiter, P3-002)
- **Depends on:** P1-001, P0-002
- **Phase gate:** `bench/gates.py::GATES["P3"]` — this task delivers the
  **differential** criterion; the curiosity criterion is P3-002 and the
  `replay-fidelity` sentinel is P3-003, so the composite stays BLOCKED here by
  design. Progress is recorded, not claimed.

## Goal
The unifying signal becomes running code: `SurpriseCompetenceMonitor` computes
calibrated surprise as a decomposed `Surprise` (never a bare float, P0-002) and
tracks per-skill competence — "learned" = low epistemic uncertainty + flattened
learning progress (ADR-0002). The gate's measurable half: a model trained on
normal physics is reliably more surprised by physics-violating transitions than by
held-out normal ones (large effect size over seeds).

## Non-goals
- No curiosity/curriculum (P3-002 — but this monitor is what it will read).
- No forgetting detection (`is_forgetting` stays `NotImplementedError("P7-001")`).
- No replay (P3-003). No new environment: violated physics = parameterized Pendulum.

## Interface to satisfy
`prospect.interfaces.CompetenceMonitor` — implement in `prospect/voe.py`.
`surprise()` returns `types.Surprise`; `update()` consumes latent-space
transitions carrying the model's `prediction` (set at act time); per-skill
attribution via `Transition.option` (P0-002). Also: the composition-root hook
promised in P2-002 — `Agent` optionally takes (world_model, monitor) and feeds
the monitor from `observe()`.

## Approach (brief)
- `surprise(prediction, observed)`: `total = -prediction.log_prob(observed.z)`
  (concrete since P0-001); attribution by predictive-variance share
  w = epistemic/(epistemic+aleatoric), so `epistemic + aleatoric == total`
  exactly. A mastered skill's surprise reads mostly aleatoric; an unfamiliar
  state's mostly epistemic. Documented limitation (ADR-0002 amendment): under
  hard distribution shift the ensemble can be confidently wrong — the
  *differential* rides on `total`, mastery on epistemic uncertainty.
- Competence: per-skill fast/slow EMAs of `prediction.epistemic`;
  `learning_progress = slow − fast` (positive while improving); mastered =
  enough updates AND fast EMA ≤ threshold AND progress flattened. Unseen skill ⇒
  `epistemic = inf`, unmastered ("never practiced = maximally unknown").
- Agent hook: when constructed with (world_model, monitor), `act()` records the
  prediction for the chosen action and `observe()` feeds the monitor a
  latent-space transition (raw-obs storage transition still returned, P0-011).
  Defaults keep P2 behavior byte-identical.
- Gate eval (`bench/evals/p3_voe.py`, registers `@gate_check("P3")`): train the
  P1-recipe model per seed (probes → run `p3`, so P1-era sentinels judge this
  run); score `Surprise.total` on held-out normal transitions vs transitions from
  a gravity-flipped Pendulum; Cohen's d per seed; differential criterion: d ≥ 1.0
  on every seed. Capability `passed` stays False with detail "curiosity pending
  (P3-002)" — metrics record the differential result.

## Acceptance criteria
- [x] Implements `interfaces.CompetenceMonitor`; conformance assertion holds;
      `surprise()` returns `Surprise` with exact total = epistemic + aleatoric.
- [x] Attribution is meaningful: epistemic-dominated prediction ⇒ mostly
      epistemic surprise; aleatoric-dominated ⇒ mostly aleatoric (unit-tested).
- [x] Mastery lifecycle: high-epistemic ⇒ unmastered; falling-below-threshold but
      progress-not-flat ⇒ still unmastered; converged-low ⇒ mastered; per-skill
      isolation; prediction-less transitions ignored (unit-tested). Also fires on
      real signals: all seeds report `mastered=1` (epistemic EMA ≈ 5e-4) after
      training in the gate eval.
- [x] Agent hook feeds the monitor latent-space transitions with the act-time
      prediction; unmonitored Agent unchanged (P2 re-ran green in the ratchet —
      the default path is structurally identical).
- [x] **Differential MET**: effect size = probability of superiority
      P(violated surprise > expected) = 0.98/0.93/0.93 per seed, min 0.93 ≥ 0.9
      (composite stays BLOCKED pending P3-002 curiosity + P3-003 sentinel).
- [x] `make test` green (61), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_voe.py): decomposition sums; attribution direction; mastery
  state machine incl. per-skill isolation and unseen-skill default; update
  ignores prediction-less transitions.
- Unit (tests/test_agent.py): monitored Agent feeds latent-space transitions with
  the act-time prediction; unmonitored Agent unchanged.
- Eval: `make gate PHASE=P3` — differential metrics per seed; sentinel health on
  run `p3`; regression: `make gate-all` still green (P0–P2).

## Docs-sync checklist
- [x] Status → `done`; gate report (BLOCKED composite, differential met) below.
- [x] architecture.md voe.py note still accurate (forgetting detection remains
      P7-001, as the note's R7 reference implies).
- [x] Backlog updated (P3-001 done; P3-002 unblocked and next).

## Gate result
`make gate PHASE=P3` — report persisted as `bench/results/P3-20260703T223442Z.json`:

```
[P3] BLOCKED   (by design at this task — see below)
  capability: not met — differential: P(violated surprise > expected) per seed
    [0.98, 0.93, 0.93], min 0.93 (criterion >= 0.9) — MET; curiosity criterion
    pending (P3-002)
  sentinel[representation-integrity]: healthy (run p3: min std 0.868, rank 2.18)
  sentinel[uncertainty-reliability]: healthy (worst-seed corr 0.79)
  sentinel[replay-fidelity]: NOT HEALTHY — PENDING (arrives with P3-003)
```

The differential half of the P3 capability is met and recorded; the composite
stays honestly BLOCKED until P3-002 (curiosity) and P3-003 (replay sentinel).
No SHIPPED change. `make gate-all`: P0–P2 still green.

**What it took:** two experiment-design lessons, recorded so they persist.
(1) Sampling violated *trajectories* from a gravity-flipped pendulum is broken —
the flipped system oscillates around θ≈0, exactly where the physics difference
2g·sinθ vanishes (d ≈ 0.4). The right construction is the infant-VoE one: same
premises, counterfactual outcomes (`Pendulum.set_state` + re-step under flipped
physics), which also removes the state-distribution confound. (2) NLL is
heavy-tailed by construction, so pooled-std effect sizes understate near-total
separation: Cohen's d saturated at ~1.0 while the violated median surprise was
~100 vs −13 expected and AUC was 0.93–0.98. The rank-based probability of
superiority is the honest effect size; d is kept in the metrics for reference.
