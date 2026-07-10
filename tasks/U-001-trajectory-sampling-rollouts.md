# U-001 — Trajectory-sampling (TS∞) imagination rollouts + uncertainty-gated truncation

- **Status:** ready
- **Phase:** U (upgrade track; re-gates against P1/P2/P5)
- **Requirements:** R1, R2, R3 (everything that reads horizon uncertainty)
- **ADRs:** ADR-0001, ADR-0006 (bounded-horizon rollouts), ADR-0002 (the split)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P2"]` and `["P5"]` (must not regress);
  new probe folded into the P1/P2 uncertainty-reliability check
- **Source:** `docs/sota-review-2026-07.md` U-001 · [PETS](https://arxiv.org/abs/1805.12114)
  · [MACURA](https://arxiv.org/abs/2405.19014) · [Infoprop](https://arxiv.org/abs/2501.16918)

## Goal
Replace mean-latent imagination with per-member trajectory sampling (TS∞): roll one
trajectory per ensemble member so horizon uncertainty is *propagated*, not reported
around a state no member believes in. Epistemic-at-horizon becomes the spread across
member trajectories; aleatoric is the mean of per-member accumulated variance. Add
optional truncation where accumulated epistemic crosses a calibrated bound.

## Non-goals
- Not sampling aleatoric noise *into* the trajectory (Infoprop's finding: injecting
  epistemic corrupts the propagated state — track it alongside, terminate on it).
- Not risk-measure planning (that is U-114-class / CVaR — out of scope).
- No change to `var`/`log_prob` semantics of a single-step `Prediction`.

## Interface to satisfy
`world_model.FlatWorldModel.imagine` (world_model.py:237) and the planner rollout in
`planning.FlatPlanner._imagined_returns` (planning.py:77-89) gain a TS∞ path: K member
trajectories propagated in parallel, mean over members = planned state, variance over
members = epistemic-at-step. `WorldModel` protocol unchanged (imagine still returns
`list[Prediction]`; the per-step epistemic now reflects propagated spread). Truncation
is an opt-in planner param (`epistemic_horizon_bound: float | None = None`).

## Approach (brief)
- Vectorize the existing `predict_batch` over `(K members × C candidates)`; each member
  keeps its own rollout state. This is 5× the current single-mean forward — negligible
  at this scale.
- Per-step: `mean = member_states.mean(0)`, `epistemic = member_states.var(0).mean()`,
  `aleatoric = per_member_var.mean(0)`. Accumulate epistemic; if a bound is set,
  stop scoring a candidate past the step its accumulated epistemic exceeds the bound
  (MACURA/Infoprop truncation — the same disagreement signal, one more consumer).
- The exploit-mode epistemic penalty (planning.py:86) now reads the *propagated*
  epistemic — strictly more honest than the current per-step-around-the-mean value.

## Acceptance criteria
- [ ] `imagine` and the planner rollout propagate K member trajectories; unit test
      shows horizon-k epistemic under TS∞ ≥ the old mean-rollout value on an OOD rollout
      (propagation no longer hides compounding uncertainty).
- [ ] Optional `epistemic_horizon_bound` truncates scoring; off by default (`None`).
- [ ] **P2 and P5 gates still PASS** every seed; `make gate-all` green (no regression).
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_world_model.py, tests/test_planner.py): TS∞ epistemic grows with
  horizon on an OOD start where mean-rollout stays flat; truncation stops at the bound;
  `bound=None` reproduces full-horizon behaviour.
- Eval: `make gate PHASE=P2`, `make gate PHASE=P5`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; gate result recorded below.
- [ ] architecture.md: "open-loop rollout on the ensemble mean" → TS∞ note; ADR-0006
      rollout paragraph amended (uncertainty is propagated, not just reported).
- [ ] Requirement traceability unchanged (R1/R2/R3 modules same).
- [ ] `docs/sota-review-2026-07.md`: mark U-001 shipped.

## Gate result
<paste the GateResult once run>
