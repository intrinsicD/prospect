# MM-008 synthetic-control no-go

**Date:** 2026-07-15  
**Protocol tested:** `d6b024619ca7d3057dcbf487a7b3822f7b99ccb31661a1e1ea97926068141a80`  
**Stage:** pre-formal synthetic calibration only  
**Real-video outcomes exposed:** none  
**Decision:** no-go; do not implement the formal lifecycle or score the 453 real transitions

## Outcome

The frozen MM-008 v1 controls failed for structural reasons. The failure is useful:
it prevents an optimizer-limited or non-identifiable assay from being sealed and then
misread as evidence about real video.

Exact-argmin replay and literal normalized-to-raw-to-normalized target construction
confirmed all of the following:

- the affine and combined coordinate searches did not recover known parameters;
- the one-coordinate convergence probe could report convergence at a point requiring
  a two-coordinate move;
- the low-rank analytic texture let independently generated targets share enough basis
  structure for false cross-fitted alignment;
- single mechanisms legitimately removed large fractions of error in the generated
  combined condition, contradicting the `>=0.90` single-arm isolation requirement;
- the literal inverse-normalization round trip made the nominal stationary target
  differ from the source at floating-point precision, contradicting exact-zero
  persistence.

No threshold, seed, expected predicate, or real decision gate was relaxed after these
observations.

## Primitive failures

The six-row frozen controls produced these diagnostic ranges:

- translation scenario, appearance-only xfit ratio: `0.874--0.892`, below the frozen
  negative margin `0.90`;
- affine scenario, quadrant-translation xfit ratio: `0.773--0.861`, including one
  false complete-support row under the v1 expectation map;
- affine and combined searches locked at a competing parameter vector such as
  `(0,0,2,-2,0,0)` instead of `(0,0,2,0,0,-2)`, with maximum parameter error `2`;
- combined scenario, global/quadrant/appearance single-arm ratios ranged
  `0.57--0.85`, affine-only ranged `0.28--0.55`, and combined endpoint errors were
  `4` full / `2` cross-fitted;
- independent-target combined ratios ranged `0.381--0.651` and falsely supported
  `6/6` rows;
- stationary persistence was tiny but nonzero after the frozen inverse-normalization
  round trip.

The exact-selection correction did not repair these patterns. They are not threshold
rounding artifacts.

## Diagnosis

### 1. The optimizer probe was too weak

Scalar coordinate descent and a scalar-coordinate probe cannot detect a basin that
requires moving two coupled affine coefficients together. The observed competing
solution specifically requires a joint move in the flow vector associated with one
input coordinate. Calling this converged would make a real negative result
optimizer-limited.

### 2. The independent negative was not high-rank

Every analytic texture used the same seven Fourier modes. Coefficients differed, but
the target remained in the same small shared basis, allowing target-aware affine and
appearance fitting on one checkerboard half to generalize spuriously to the other.
This tests shared-basis alignment, not independent content.

There is also a comparator defect independent of the Fourier basis. For independent
equal-variance source and target, unrestricted gain/bias OLS can select gain near zero
and predict the fitted target mean. Its expected residual is then about one target
variance, versus about two variances for source persistence, so an xfit/persistence
ratio near `0.5` is legitimate even without correspondence. The per-row/channel random
constant in v1 additionally creates genuinely learnable global appearance. Therefore
an independent appearance null cannot require gain/bias to retain 90% of persistence;
it needs a cross-fitted constant/shrinkage comparator plus pairing and complete-support
controls.

### 3. “Combined-only” is not a generic factorial invariant

With squared error and two mechanisms that each remove a component of the persistence
residual, either single mechanism can legitimately improve substantially even when the
joint operator is uniquely capable of exact recovery. Requiring both singles to retain
at least 90% of persistence while the combined arm removes at least 50% is not a sound
generic control for additive geometry and appearance. A combined condition should
validate exact joint recovery and prespecified incremental benefit over both singles;
it should not require the singles to be useless.

In the first-order orthogonal case `persistence_error = G + A`, affine-only leaves `A`
and appearance-only leaves `G`, so the two single-arm MSE ratios sum approximately one.
The v1 rule requires both ratios to be at least `0.90`, hence a sum of at least `1.80`.
That can occur only through strong cancellation/suppressor behavior, which would be an
ill-conditioned control rather than evidence of clean operator isolation.

### 4. Stationarity must share raw bytes

An exact stationary control should set raw target equal to raw source and pass both
through the same current-only normalizer. Algebraic inversion and reapplication is not
bit-exact floating-point identity.

## Frozen requirements for a separate v2 protocol

MM-008 v1 remains a no-go record. A new protocol version must be frozen and audited
before another calibration. It should:

1. replace scalar affine updates with natural two-vector blocks:
   `(ty,tx)`, `(ayy,axy)`, and `(ayx,axx)`, exhaustively evaluating all 25 values per
   block during each sweep and during a read-only block probe;
2. use full-rank independently drawn native-pixel texture (or another demonstrably
   high-rank construction) for implementation isolation, with exact seeds and array
   construction frozen;
3. make raw stationary target byte-identical to raw source;
4. preserve the positive endpoint and 50%-of-persistence margins for operator-realizable
   arms;
5. reserve negative `>=0.90` margins for genuinely incapable/nonnested arms and the
   high-rank geometric arms; evaluate independent appearance/combined against a
   cross-fitted per-channel constant or shrinkage baseline rather than persistence;
6. for the combined scenario, require exact combined parameter recovery and frozen
   conditional dominance over both single cells, for example
   `1.25*combined_mse <= single_mse`, without requiring either single cell to fail the
   ordinary 20% real-data gate;
7. retain all real-video support, pairing, fold, boundary, full/xfit, and claim-boundary
   gates unchanged.

Before a formal lifecycle, v2 must also retain appearance macrocell identities,
objective histories, boundary count numerators/denominators, endpoint SSE/counts, and
hashes in evidence, and must fail closed on missing or non-finite applicable endpoints.

The v2 synthetic controls must pass untouched before lifecycle implementation or any
real-video target-aware scoring begins.
