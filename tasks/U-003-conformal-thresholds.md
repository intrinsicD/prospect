# U-003 — Adaptive conformal calibration of VoE thresholds (termination + retrieval gate)

- **Status:** ready
- **Phase:** U (upgrade track; re-gates against P5/P8/P9)
- **Requirements:** R2, R3, R8
- **ADRs:** ADR-0002 (the one signal), ADR-0004 (retrieval gating)
- **Depends on:** none
- **Phase gate:** `bench/gates.py::GATES["P5"]`, `["P8"]`, `["P9"]` (must not regress)
- **Source:** `docs/sota-review-2026-07.md` U-003 · [ACI](https://arxiv.org/abs/2106.00170)
  · [decaying-step ACI](https://arxiv.org/pdf/2402.01139) · [conformal failure detection](https://arxiv.org/pdf/2503.08558)

## Goal
Replace fixed NLL thresholds — which have no false-alarm semantics and silently stop
firing as the model improves — with adaptive conformal inference (ACI): maintain a
target trigger rate α on the surprise stream via online update
`θ_{t+1} = θ_t + η(1{s_t > θ_t} − α)`, distribution-free, no exchangeability assumption.

## Non-goals
- **Do NOT touch the forgetting detector's latched mastered-error floor**
  (`voe.py:133-143`): an adaptive threshold would adapt away exactly the sustained
  error rise it must catch. ACI is for the *termination* and *retrieval* gates only.
- Not recalibrating the whole predictive distribution (would perturb the load-bearing
  epistemic/aleatoric split — cite Kuleshov-style recalibration as the rejected route).
- Not the PID conformal variant (unnecessary complexity at this scale).

## Interface to satisfy
A small `voe`-side helper `AdaptiveThreshold(alpha: float, eta: float)` with
`update(score) -> None` and `.value -> float`. `HierarchicalManager.should_terminate`
(planning.py:283-289) reads a threshold that is ACI-tracked on the nominal one-step
surprise stream; `memory.UncertaintyMemoryRouter` (memory.py:213-219) reads an
ACI-tracked epistemic gate. **Separate trackers, separate α per consumer.** No Protocol
change (thresholds stay floats to the consumers).

## Approach (brief)
- Two-line online quantile tracker on the running score stream; `η` decays
  (`η_t = η_0/√t`) so the threshold converges to the true quantile once the
  distribution settles ([ICML 2024](https://arxiv.org/pdf/2402.01139)).
- The harness (`bench/`) owns α (target interrupt / retrieval rate); the core exposes
  the tracker, the harness calibrates it — same split as the current threshold policy.
- **Pair with U-011:** feed the tracker the *epistemic-normalized* surprise (not raw
  total NLL) for termination, so the interrupt fires on "model wrong", not "world
  noisy". If U-011 lands first, this consumes its normalized signal directly.

## Acceptance criteria
- [ ] `AdaptiveThreshold` holds an empirical α-trigger-rate on a synthetic score stream
      (unit-tested: realized rate → α as the stream lengthens; converges under a shift).
- [ ] Termination and retrieval gates read ACI-tracked thresholds; forgetting floor
      untouched (assert the mastered-error latch is still frozen).
- [ ] **P5, P8, P9 gates PASS**; `make gate-all` green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_voe.py, tests/test_knowledge.py): realized trigger rate vs α;
  convergence to the quantile after a distribution shift; forgetting detector unchanged.
- Eval: `make gate PHASE=P5`, `PHASE=P8`, `PHASE=P9`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; gate results recorded below.
- [ ] ADR-0002: add a consequence — thresholds are conformally calibrated (controlled
      false-alarm rate), forgetting floor deliberately excluded.
- [ ] architecture.md: note the calibrated trigger-rate for termination/retrieval.
- [ ] `docs/sota-review-2026-07.md`: mark U-003 shipped.

## Gate result
<paste the GateResult once run>
