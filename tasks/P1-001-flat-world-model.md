# P1-001 — Flat latent world model + calibrated uncertainty

- **Status:** done
- **Phase:** P1
- **Requirements:** R1, R4
- **ADRs:** ADR-0001, ADR-0002, ADR-0006
- **Depends on:** P0-001 (`Prediction` distribution contract), P0-003 (`Learner`
  training seam), P0-004 (`Environment` protocol for the toy task), P0-005 (run-metrics
  artifact the sentinels read), P0-006 (gate/sentinel check registration)
- **Phase gate:** `bench/gates.py::GATES["P1"]`

## Goal
A `WorldModel` that, given a latent state and an action, predicts a **distribution**
over the next latent state and reward, with uncertainty split into epistemic and
aleatoric. This is the substrate everything else hangs off — build it first.

## Non-goals
- No planning yet (that is P2). No skills, hierarchy, codec, or retrieval.
- No pixel/observation reconstruction — predict in latent space (ADR-0001).
- No multi-modality — a single toy modality (low-dim state) is enough for the gate.

## Interface to satisfy
`prospect.interfaces.WorldModel` **and** `prospect.interfaces.Learner` — implement in
`prospect/world_model.py` (replace the `FlatWorldModel` skeleton). `predict()` returns
`types.Prediction` with real `mean`, per-dimension `var`, `epistemic`, `aleatoric`,
`reward`. `log_prob` is concrete on `Prediction` (diagonal Gaussian, P0-001) —
subclass only to vectorize for a tensor backend, keeping the same definition.
`update(batch)` trains from transitions and returns the metrics dict (losses +
integrity stats) the harness logs for the sentinels (P0-003, P0-005).

## Approach (brief)
- Encode to a latent, learn a probabilistic transition head (e.g. Gaussian) for
  aleatoric uncertainty; use a small **ensemble** so disagreement gives epistemic
  uncertainty. Train from a replay of transitions on a toy control task (harness).
- Guard against collapse from the start (ADR-0006), since the gate now includes the
  integrity sentinels: EMA target encoder + variance–covariance regularization +
  inverse-dynamics and reward auxiliary heads; independent ensemble inits with
  decorrelated data order.
- Log per-step sentinel metrics (latent std / effective rank, ensemble-disagreement
  vs held-out error) to the run log (`bench.runlog.RunLog`, P0-005) from
  `update()`'s returned dict + held-out probes; the P1 sentinel `check()`s read the
  run back.
- Keep it minimal — smallest model that clears the gate (sentinels included). No config
  knobs beyond what the gate needs.

## Acceptance criteria
- [x] Implements `interfaces.WorldModel` and `interfaces.Learner`; `predict` returns
      a proper `Prediction`; `update` returns the training-metrics dict.
- [x] `surprise = -log_prob(observed)` is finite and calibrated on held-out data
      (probes log `heldout_nll` and `calibration_ratio` — standardized residuals).
- [x] **Gate P1:** latent 1-step prediction beats a persistence/linear baseline
      (median held-out MSE 0.0121 vs 0.2788 / 0.2421 — the linear baseline maps raw
      (obs, action) to the target latent, not model-assisted), AND on the stochastic
      variant epistemic falls with more data (ratio 0.29 across 128→8192 samples)
      while aleatoric persists (ratio 1.36, inside [0.5, 2.0]).
- [x] **Sentinel `representation-integrity`:** min per-dim std 0.868 (floor 0.3),
      min effective rank 2.18 (floor 2.0 = the task's intrinsic dimension; the
      pendulum has 2 DOF) across 39 held-out probes after a 300-step formation
      warm-up — no collapse.
- [x] **Sentinel `uncertainty-reliability`:** worst-seed disagreement-vs-error rank
      correlation 0.79 (min 0.3) on a mixed in-dist + OOD probe set; high-error
      decile carries 21x the median disagreement.
- [x] `make test` green (43), `make lint` clean, `make typecheck` clean.

## Test plan
- Unit: shapes/dtypes of `Prediction`; `log_prob` sane on a known Gaussian.
- Eval (harness, in `bench/`): the toy control task implements `bench.Environment`
  (P0-004, seeded resets); baseline comparison + the epistemic/aleatoric
  separation experiment. Register the eval in `bench/evals/` via `@gate_check("P1")`
  and the two P1 sentinel bodies via `@sentinel_check(...)` (P0-006); record the
  seed list in `GateResult.seeds`.

## Docs-sync checklist
- [x] This Status → `done`; P1 `GateReport` (capability + sentinels) pasted below.
- [x] The `representation-integrity` and `uncertainty-reliability` sentinel `check()`s
      are implemented in `bench/evals/p1_world_model.py` (no longer PENDING) and healthy.
- [x] Requirement rows R1/R4 still accurate (verified — world_model.py/codec-free
      latent path unchanged in the table).
- [x] ADR-0002 consequence holds in code: `predict()` returns a full `Prediction`
      (mean + per-dim total `var` + epistemic/aleatoric split + working `log_prob`).
- [x] architecture.md component note for world_model.py still accurate.
- [x] `P1` appended to `bench/SHIPPED` in this commit; the ratchet re-runs the gate
      in CI (deterministic seeds — the re-run reproduces identical metrics).

## Gate result
`make gate PHASE=P1` — report persisted as `bench/results/P1-20260703T160422Z.json`:

```
[P1] PASS
  capability: ok — median held-out latent MSE 0.0121 vs persistence 0.2788 /
    linear 0.2421; epistemic ratio 0.29 (must be < 0.7), aleatoric ratio 1.36
    (must stay in (0.5, 2.0))
  sentinel[representation-integrity]: healthy — across 39 held-out probes
    (warm-up 300 steps): min per-dim std 0.868 (floor 0.3), min effective rank
    2.18 (floor 2.0)
  sentinel[uncertainty-reliability]: healthy — worst seed at end of training:
    disagreement-vs-error rank corr 0.79 (min 0.3), high-error-decile
    disagreement 21.01x median (min 1.0) on a mixed in-dist + OOD probe set
```

Seeds [0, 1, 2]; full eval runs in ~40s and is deterministic, so the ratchet
re-runs it in full (artifact-based re-runs deferred until a gate outgrows this,
per the P0-007 policy).

**What it took (recorded so the lessons persist):** the first run was BLOCKED by
exactly the collapse ADR-0006 predicts — effective rank 1.2 with a good-looking
loss; the sentinel caught it. Fixes, in order of what actually mattered:
(1) **input standardization** — the raw pendulum observation's variance is
dominated by ω, so the input itself has effective rank ≈ 1.17 and the encoder
faithfully reproduced that degeneracy; (2) **VICReg-proportionate weights**
(variance hinge 25, covariance 1 — cranking covariance to 15 destroyed prediction
and the uncertainty structure); (3) **global gradient-norm clipping** — the
remaining rank dips were abrupt coordinated encoder moves, not steady learning;
(4) an honest **linear baseline** (raw obs → target latent, not one fed the
model's own trained encoder); (5) sentinel thresholds instantiated with reasons:
rank floor 2.0 = the task's intrinsic dimension, 300-step warm-up excludes
representation *formation* (the sentinel guards against collapse *after* it).
