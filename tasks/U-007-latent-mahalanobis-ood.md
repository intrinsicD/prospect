# U-007 — Latent-space Mahalanobis density as a second OOD signal (DDU-style)

- **Status:** ready
- **Phase:** U (upgrade track; re-gates against P9)
- **Requirements:** R1, R3, R4, R7, R8 (everything that reads the epistemic signal)
- **ADRs:** ADR-0002 (epistemic is distance-aware — extend from pre-encoder to
  feature-space), ADR-0006
- **Depends on:** none (extends P9-005)
- **Phase gate:** `bench/gates.py::GATES["P9"]` — the uncertainty-generalizes criterion
  (must hold or improve the epistemic-vs-error rank correlation)
- **Source:** `docs/sota-review-2026-07.md` U-007 · [SNGP](https://arxiv.org/abs/2006.10108)
  · [DDU](https://arxiv.org/abs/2102.11582)

## Goal
The pre-encoder OOD score (P9-005, `LatentState.ood`) is a legitimate homebrew of the
documented "feature collapse" fix, but it lives *before* the encoder and cannot see OOD
that is only visible in feature combinations. Add a DDU-style Gaussian/Mahalanobis
density fitted on the *latent* features as a second distance signal — the literature's
feature-space complement to the input-space score.

## Non-goals
- Not a full SNGP/GP/spectral-normalization rebuild (spectral norm of the numpy MLP is
  the deferred heavier option) — the minimal feature-space density only.
- Not replacing the P9-005 input-space score — the two compose (input-space catches
  raw distance, latent-space catches feature-combination OOD).
- No change to `var`/`log_prob` (the split's likelihood stays the ensemble's).

## Interface to satisfy
`world_model.FlatWorldModel`: maintain a running latent mean+covariance (EMA, updated in
`update`); `encode`/`predict` compute a Mahalanobis distance of the latent to that
Gaussian and fold it into the epistemic scalar alongside the existing `ood` term
(world_model.py:216-235). `LatentState` may carry the extra score or it is combined into
the existing scaling. No new Protocol.

## Approach (brief)
- Running `μ, Σ` over online latents (8-dim — a trivial `d×d` EMA in numpy); Mahalanobis
  `√((z-μ)ᵀ Σ⁻¹ (z-μ))`, ~0 in-distribution, rising in unseen feature regions.
- Combine with the pre-encoder `ood`: `epistemic_scalar ·= 1 + w₁·ood + w₂·mahalanobis`
  (extends world_model.py:226-228). Both self-calibrate to ~0 in-distribution so the
  P9-005 gate preservation (in-distribution epistemic unchanged) still holds.
- Cite SNGP/DDU in ADR-0002 as the principled basis for the whole distance-aware family
  (the review noted the current ADR states the mechanism without its literature anchor).

## Acceptance criteria
- [ ] Latent Mahalanobis density computed and folded into epistemic; in-distribution
      score ≈ 0 (self-calibrated gates preserved), rises on feature-combination OOD in a
      unit test where the input-space `ood` alone stays low.
- [ ] **P9 uncertainty-generalizes criterion PASS with epistemic-vs-error rank corr ≥
      current** (0.80 on PointMass, P9-005); `make gate-all` green.
- [ ] `make test` green, `make lint` clean, `make typecheck` clean.

## Test plan
- Unit (tests/test_world_model.py): a crafted latent far in feature space but near in
  input space raises Mahalanobis while `ood` stays ~0; in-distribution both ≈ 0.
- Eval: `make gate PHASE=P9`, `make gate-all`.

## Docs-sync checklist
- [ ] Status → done; rank-corr before/after recorded below.
- [ ] ADR-0002: amend the distance-aware clause — pre-encoder score + latent Mahalanobis;
      cite SNGP/DDU as the principled basis.
- [ ] `docs/sota-review-2026-07.md`: mark U-007 shipped.

## Gate result
<paste the GateResult once run>
