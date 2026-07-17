# MM-006 causal-warp ceiling protocol

**Date:** 2026-07-15  
**Status:** frozen before MM-006 real-video warp outcomes  
**Parent:** sealed MM-005
(`half_second_tested_spatial_local_linear_objective_failure_supported`)  
**Scope:** outcome-informed, single-step mechanism diagnostic on the same eight videos

## 1. Question

MM-005 held rows, folds, normalization, local features, estimator, controls, and
support gates fixed while changing the target from one second to half a second. Both
tested local-linear residual predictors still supported 0/8 in TAESD and source
pixels, paired half-horizon advantage supported 0/8, and central pixel activity was
present on 7/8 videos. Horizon shortening alone is therefore not the supported fix.

MM-006 separates three remaining questions without training a neural network:

1. **Transport realizability:** can the half-second target be represented by a small
   spatial warp of the current grid?
2. **Causal observability:** can that warp be inferred from `previous,current` without
   seeing the target?
3. **Representation transfer:** do the answers agree between source pixels and TAESD?

The target-informed warp is an explicitly leaking diagnostic ceiling. It can establish
that a tested transport family is expressive enough, but it can never count as a
deployable prediction or capability result.

## 2. Claim boundary

MM-006 is outcome-informed and reuses the eight already visible Perception Test sample
videos. Videos are the support units; rows and cells are repeated measurements. A
6/8 result is descriptive on this panel, not a population estimate, and two videos in
each fold share a train-only normalizer.

The assay predicts one teacher-forced half-second step on central cells. It does not
test teacher-free rollout, decoded video quality, action conditioning, independent
videos, end-to-end Prospect learning, dense optical flow, or general nonlinear
dynamics. A passing causal warp only licenses an independent-panel and rollout gate.

MM-001 through MM-005 remain sealed and are never reclassified.

## 3. Immutable parent receipts

The live MM-005 package must fast-verify before preparation, before formal execution,
and during every verification. MM-006 copies exactly seven selected receipts beneath
`inputs/MM-005`:

1. `artifact-manifest.json`
2. `input-manifest.json`
3. `MM-005-evidence.json`
4. `MM-005-results.json`
5. `inputs/MM-004/MM-004-pixel-grids.npz`
6. `inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-features.npz`
7. `inputs/MM-004/inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz`

The MM-005 artifact-manifest SHA-256 is pinned to
`c0e8fc7772799631b1b9e57167d4b8d70b71dc14f1fbd8d21847a9695d9c3e66`.
Every selected byte, mode, manifest record, parent evidence/result relationship,
synthetic-control status, and parent classification must replay exactly. Strict NPZ
loaders reject extra keys, pickle/object arrays, wrong dtype/shape, non-finite values,
out-of-range pixels, paths, symlinks, or mode drift.

Preparation reconstructs MM-005's exact 453 rows and verifies both domains, counts,
timestamps, bit-exact target alignment, and identity SHA-256
`d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b`.
It may compute hashes and train-current channel statistics. It may not call an MM-006
warp search, synthetic generator, scorer, or decision function before the formal
marker; replaying the already sealed MM-005 decision remains mandatory authentication.

## 4. Exact panel and normalization

Only the half-second target is evaluated:

```text
previous = current[i-1] at t-0.5 s
current  = current[i]   at t
target   = current[i+1] at t+0.5 s
```

The exact MM-005 source positions `i=1..N-3`, 453 identities, per-video counts
`60/61/56/62/62/45/63/44`, and four six-train/two-test whole-video folds are reused.
For each domain and fold, one float64 channel mean and standard deviation is computed
from training-video **current** grids only, with scale floor `1e-6`. The same
normalizer is used by every real arm and control in the fold. Candidate selection and
all warp-arm/control MSEs operate in that normalized space. The separately reported
pixel-activity check remains in raw pixel space to replay MM-005's descriptive guard.

The causal estimator receives only normalized `previous,current`. The target is passed
later to a separate scorer. Mutating the target must leave causal flow and prediction
fingerprints bit-exact. The oracle receives `current,target` and is always marked
`uses_target=true`, `diagnostic_only=true`.

## 5. Frozen warp convention and search

Displacement `d=(dy,dx)` moves content down and right:

```text
warp(X,d)[y,x] = X[y-dy, x-dx]
```

Sampling is deterministic float64 bilinear interpolation. Real observations are never
wrapped, clamped, reflected, or padded. Every primary candidate has valid source
support because fitting and scoring use the fixed central `6x6` cells (`y,x=1..6`)
and

```text
dy,dx in {-1.0,-0.5,0.0,0.5,1.0}.
```

Candidates are ordered by squared magnitude, then `dy`, then `dx`; exact objective
ties select the first candidate. Matching computes channel-mean squared residual at
each selected cell, drops the largest floor(`n/4`) spatial residuals, and averages the
remainder. Prediction metrics are ordinary, untrimmed SSE over identical cells,
channels, and rows.

Two prespecified spatial families are evaluated:

- `global_translation`: one displacement per row;
- `quadrant_flow` (primary): four displacements for the `3x3` quadrants of the central
  `6x6`. Each quadrant's objective adds
  `0.05 * (identity_loss + 1e-12) * ||d-d_global||^2`, anchoring local matches to the
  corresponding global estimate. The four quadrant predictions tile the central
  region exactly; there is no target-derived mask or post-outcome arm selection.

For each family, the real arms are:

- `causal`: estimate `previous -> current` on all central cells, then apply that
  displacement to `current`; this target-isolated predictor is the primary causal arm;
- `oracle_full`: estimate `current -> target` and score the same cells; diagnostic
  overfit measurement only;
- `oracle_xfit`: checkerboard cross-fit the oracle. A displacement selected on even
  cells predicts only odd cells and vice versa, so every scored target cell is absent
  from its own flow selection.

The controls are:

- persistence (`prediction=current`);
- horizon-correct constant velocity (`current + current-previous`);
- history shuffle: fixed-point-free half-cycle permutation of `previous` within each
  video before causal estimation;
- reverse sign: apply `-d_causal` to `current`;
- target shuffle: estimate the cross-fitted oracle against a fixed-point-free
  within-video target permutation, but evaluate its prediction against the true target;
- source reconstruction: separately checkerboard-cross-fit `previous -> current`, so
  every reconstructed current cell was excluded from its own displacement selection;
  compare its held-out MSE both with unwarped `previous,current` and with an identically
  cross-fitted, within-video shuffled-`previous` reconstruction. This diagnostic never
  replaces the full-cell target-isolated causal predictor.

## 6. Primitive evidence

For each domain, family, and video, aggregate SSE numerators and element counts across
all rows/cells/channels before division. Save:

- persistence, constant-velocity, causal, history-shuffle, reverse-sign,
  full-oracle, cross-fitted-oracle, and target-shuffled-oracle SSE/MSE;
- causal and oracle ratios to persistence;
- cross-fitted ordered-source, shuffled-history-source, and identity-source MSE;
- causal capture of oracle-reducible error;
- causal/oracle flow cosine and endpoint error;
- forward/reverse cycle error;
- mean flow magnitude, boundary-candidate fraction, and objective confidence gap;
- exact source, target, causal/source-cross-fit flow, prediction, normalizer, and
  control fingerprints;
- finite flags, candidate counts, row counts, and common element counts.

Zero persistence is saved explicitly and is ineligible for support. No row, cell, or
video is dropped after outcomes.

## 7. Frozen video predicates

Let `p,o,f,s,c,h,r,v,a,q,b` be, respectively, persistence, cross-fitted oracle,
full oracle, target-shuffled oracle, causal, history-shuffled causal, reverse-sign,
constant velocity, cross-fitted causal source reconstruction, cross-fitted shuffled-
history source reconstruction, and identity source reconstruction MSE for one video.

Cross-fitted oracle performance support is `p>0` and `1.25*o<=p`. Oracle pairing
support additionally requires `1.10*o<=s`; only their conjunction is the complete
oracle support predicate. This split prevents a failed pairing control from being
misreported as a failed transport ceiling.

Complete oracle support requires all of:

```text
p > 0
1.25 * o <= p                 # at least 20% better than persistence
1.10 * o <= s                 # target pairing beats shuffled-target selection
```

Causal-fix support requires oracle support and all of:

```text
1.25 * c <= p                 # at least 20% better than persistence
1.10 * c <= h                 # ordered history beats shuffled history
1.10 * c <= r                 # correct direction beats reverse sign
1.10 * c <= v                 # warp beats value-space constant velocity
1.25 * a <= b                 # ordered past beats source identity
1.10 * a <= q                 # ordered past beats matched shuffled-history warp
2.0 * c <= p + o              # captures at least half oracle-reducible gain
```

Full-oracle-only support is `1.25*f <= p` while cross-fitted oracle performance
support is false. Source-flow support requires `b>0`, `1.25*a<=b`, and `1.10*a<=q`;
zero identity-source error is ineligible, just like zero persistence.
A null target-shuffle hit is `1.25*s<=p`. Causal performance support records the
target-performance and oracle-capture inequalities separately from the complete
history, direction, velocity, pairing, and source controls; a performance/control
disagreement is inconclusive, never a strong mechanism label.

A causal or oracle family-level pass requires at least 6/8 supporting videos, at least
one supporting video in every fold's two-video test pair, and any improvement over
persistence on at least 7/8. The full oracle uses the same three requirements with
`f` in place of `o`. Source-flow pass uses at least 6/8 source-supporting videos, every
fold, `a < b` on at least 7/8, and `a < q` on at least 7/8. Counts 3-5 are borderline;
a raw count at least six that misses fold coverage or either 7/8 improvement
requirement is also inconclusive.
Counts at most 2 are strong failure. A per-video boundary warning is present when at
least 25% of selected displacement coordinates have absolute value 1.0; the tested
range is declared inconclusive only when at least 3/8 primary pixel videos warn.

All inequalities are inclusive and evaluated by multiplication, not rounded ratios.

## 8. Synthetic and metamorphic controls

Before real interpretation, three PCG64 seeds `660060,660061,660062`, both channel
counts (3 and 4), exact MM-005 identities/folds, and analytic low-frequency textures
plus seeded independent-Gaussian nulls exercise:

1. global constant translation, including cardinal, diagonal, integer, and half-cell
   displacement: causal and both oracles must recover the signed motion and strongly
   beat persistence; history shuffle and reverse sign must fail;
2. reversal/acceleration (`future flow = -past flow`): oracle must pass and causal must
   fail, validating the oracle-only branch;
3. independent Gaussian appearance innovation, generated from the frozen scenario
   seed and independent of the analytic source texture: both cross-fitted warp
   ceilings must fail;
4. independent Gaussian `previous,current,target`: cross-fitted ordered-source
   reconstruction must not beat the matched shuffled-history warp null by the frozen
   10% control margin (`1.10*a<=q` must be false), irrespective of the identity-source
   comparison;
5. stationary `previous=current=target`: persistence zero remains ineligible and
   cannot pass via `0<=0`;
6. periodic ambiguity: deterministic ties and boundary/confidence diagnostics must be
   stable and cannot produce a clean causal pass.

Tests additionally freeze signed displacement, bilinear interpolation, central-mask
accounting, candidate order/ties, channel permutation invariance, target mutation
independence of end-to-end causal outputs, flow/prediction reconstruction,
source-null isolation, and every decision branch.

Synthetic controls may tune implementation defects before the marker; no threshold,
candidate radius, regularizer, mask, arm, or branch may change after real outcomes.

## 9. Decision order

The summary executes the following frozen ladder:

1. Any parent, alignment, schema, numerical, determinism, leakage, or synthetic positive
   failure -> invalid MM-006 package/control classification.
2. Real target-shuffle null support at least 6/8 -> invalid real control; 3-5/8 ->
   inconclusive real control.
3. Evaluate pixel `quadrant_flow` cross-fitted oracle.
   - If it strongly fails while the complete full-oracle family gate passes ->
     `target_fitted_oracle_overfit_supported`.
   - If it fails with a boundary warning -> `tested_transport_range_inconclusive`.
   - Otherwise -> `tested_pixel_warp_ceiling_failure_supported`.
4. If the pixel oracle passes, evaluate pixel causal flow.
   - If it strongly fails while the complete source-flow family gate passes ->
     `two_frame_motion_extrapolation_failure_supported`.
   - If source-flow support fails -> `low_resolution_correspondence_failure_supported`.
5. If pixel causal flow passes, evaluate TAESD oracle and causal flow.
   - Pixel causal pass plus TAESD oracle failure ->
     `taesd_transport_equivariance_failure_supported`.
   - TAESD oracle pass plus strong TAESD causal failure and a complete TAESD
     source-flow pass -> `taesd_two_frame_motion_extrapolation_failure_supported`.
   - The same causal failure with strong TAESD source-flow failure ->
     `taesd_causal_correspondence_failure_supported`; borderline source flow is
     inconclusive.
   - Both causal families pass -> `single_step_causal_warp_fix_supported`.
6. Any 3-5/8 primary count, any raw count at least six that misses its complete pass
   gate, any performance/control disagreement, or any domain where global translation
   passes an oracle or causal family gate while the corresponding quadrant primary
   fails, yields `MM006_diagnostic_inconclusive` before a strong mechanism label.

Global-translation results attach mechanism labels but cannot replace the frozen
quadrant primary. A clean local pass with global failure supports spatially varying
motion; a global pass is a simpler sufficient mechanism.

## 10. Lifecycle and verification

The new package is exactly four Python files plus two focused test files and this
protocol. Its source set is the exact 54-file MM-005 set plus these seven files (61
total). Preparation creates exactly nine files (protocol, manifest, seven receipts),
with no marker or outcomes. Formal execution writes the read-only `0444` marker via
exclusive creation before the first scientific call, executes once, writes evidence,
result, report, and finally the artifact manifest. The completed tree contains exactly
14 files; generated files are `0644`, inherited modes are preserved, and symlinks or
extras fail closed.

Fast verification replays membership, modes, manifests, source/config/dependency
bindings, live and copied parent receipts, exact panel provenance, evidence schemas,
primitive predicates, decision, and report without rerunning warp search. Semantic
verification regenerates every synthetic and real flow/prediction from copied arrays
and compares nested values at `rtol=atol=1e-12`.

An interrupted package is terminal evidence and is never resumed. Any post-marker
defect receives a new experiment identifier rather than repairing MM-006 in place.
