# P1-001 — Flat latent world model + calibrated uncertainty

- **Status:** blocked (P0 contract hardening — see Depends on)
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
`prospect.interfaces.WorldModel` — implement in `prospect/world_model.py`
(replace the `FlatWorldModel` skeleton). `predict()` returns `types.Prediction`
with real `mean`, `epistemic`, `aleatoric`, `reward`, and a working `log_prob`.

## Approach (brief)
- Encode to a latent, learn a probabilistic transition head (e.g. Gaussian) for
  aleatoric uncertainty; use a small **ensemble** so disagreement gives epistemic
  uncertainty. Train from a replay of transitions on a toy control task (harness).
- Guard against collapse from the start (ADR-0006), since the gate now includes the
  integrity sentinels: EMA target encoder + variance–covariance regularization +
  inverse-dynamics and reward auxiliary heads; independent ensemble inits with
  decorrelated data order.
- Keep it minimal — smallest model that clears the gate (sentinels included). No config
  knobs beyond what the gate needs.

## Acceptance criteria
- [ ] Implements `interfaces.WorldModel`; `predict` returns a proper `Prediction`.
- [ ] `surprise = -log_prob(observed)` is finite and calibrated on held-out data.
- [ ] **Gate P1:** latent 1-step prediction beats a persistence/linear baseline,
      AND on a stochastic variant epistemic uncertainty **falls with more data**
      while aleatoric **persists** (the two are separable).
- [ ] **Sentinel `representation-integrity`:** latent per-dimension std and effective
      rank stay above their floors on held-out data throughout training (no constant /
      low-rank collapse).
- [ ] **Sentinel `uncertainty-reliability`:** ensemble disagreement is rank-correlated
      with held-out error (so the epistemic estimate is trustworthy, not collapsed).
- [ ] `make test` green, `make lint` clean.

## Test plan
- Unit: shapes/dtypes of `Prediction`; `log_prob` sane on a known Gaussian.
- Eval (harness, in `bench/`): baseline comparison + the epistemic/aleatoric
  separation experiment; wire the result into `GATES["P1"].check`.

## Docs-sync checklist
- [ ] This Status → `done`; paste the P1 `GateReport` (capability + sentinels) below.
- [ ] The `representation-integrity` and `uncertainty-reliability` sentinel `check()`s
      are implemented (no longer PENDING) and healthy.
- [ ] Requirement rows R1/R4 still accurate.
- [ ] ADR-0002 consequence ("returns Prediction, never a float") holds in code.
- [ ] architecture.md component note for world_model.py still accurate.

## Gate result
_not run yet_
