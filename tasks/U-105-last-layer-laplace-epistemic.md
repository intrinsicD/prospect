# U-105 — Last-layer Laplace / neural-linear epistemic

- **Status:** deferred
- **Phase:** U (deferred upgrade; trigger-gated)
- **Requirements:** R1, R3, R4
- **ADRs:** ADR-0002 (the split), ADR-0006
- **Depends on:** U-007 (exhaust the cheaper distance-aware fixes first)
- **Phase gate:** `bench/gates.py::GATES["P9"]` (uncertainty reliability)
- **Source:** `docs/sota-review-2026-07.md` U-105 · [laplax](https://arxiv.org/abs/2507.17013)
  · [CLAPS](https://arxiv.org/abs/2512.01384)

## Trigger (promote to `ready` when…)
**Measured evidence that 5-member disagreement is too coarse** — the epistemic-vs-error
rank correlation fails its floor on some environment *despite* U-007's latent Mahalanobis
density and U-001's propagation, or the ensemble is shown to collapse (members agree where
wrong) in a way the distance-aware fixes don't repair. The **upgrade-triggers** workflow
step checks: if a P9-class uncertainty-reliability check regresses after U-007/U-001 are
in, promote. Until then the 5-member ensemble is 2026 best practice at this scale (review
Q2) — do not replace a working mechanism.

## Goal
When triggered: add a closed-form last-layer Laplace (neural-linear) epistemic estimate
on the tiny dynamics heads — a `d×d` posterior covariance per head giving smooth,
distance-aware epistemic that grows with feature-space novelty, complementing or replacing
ensemble disagreement.

## Non-goals
- Not epinets (review SKIP: value is replacing huge ensembles; 5 tiny MLPs aren't the
  bottleneck) and not evidential deep learning (documented trap, ADR-0002).
- Not before U-007 — the cheaper distance-aware density comes first.

## Interface to satisfy (when promoted)
`world_model.FlatWorldModel`: a last-layer Gaussian posterior (Laplace approx) over each
member's output layer; epistemic = predictive variance from that posterior, folded into
the epistemic scalar. `WorldModel` protocol unchanged.

## Approach (brief, when promoted)
- Neural-linear: treat the last layer's weights as Gaussian, closed-form posterior from
  the feature covariance (numpy-feasible at `d≈tens`); adds one prior-precision knob.
- Compare against the ensemble on the failing env's rank-correlation before committing;
  keep whichever wins (measured, ADR-0008 discipline).

## Acceptance criteria (when promoted)
- [ ] Last-layer Laplace epistemic implemented; epistemic-vs-error rank corr on the
      failing env ≥ floor (the reason for promotion).
- [ ] In-distribution gates preserved; `make gate-all` green; tests/lint/typecheck clean.

## Test plan (when promoted)
- Unit: posterior variance grows with feature-space distance on a synthetic head.
- Eval: `make gate PHASE=P9` + `make gate-all`.

## Docs-sync checklist
- [ ] On promotion: Status → ready; follow lifecycle.
- [ ] ADR-0002: record the Laplace option and the measurement that triggered it.
- [ ] `docs/sota-review-2026-07.md`: note U-105 outcome.

## Gate result
<deferred — no gate until promoted>
