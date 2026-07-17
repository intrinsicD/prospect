# MM-005 matched half-horizon replay protocol

**Date:** 2026-07-15  
**Status:** frozen before MM-005 real-video outcomes  
**Parent:** sealed MM-004 (`tested_local_objective_or_horizon_failure_supported`)  
**Scope:** outcome-informed, matched-row horizon diagnostic on the same eight videos

## 1. Question

MM-004 established that source pixels change on all eight videos, but neither TAESD nor
pixel grids were predicted at the frozen one-second target by any tested local-linear
spatial/history arm. MM-002 had also found no half-second rescue for a flattened
single-frame linear probe. The unresolved narrow hypothesis is whether the spatial
history model in MM-004 was asked to predict too far ahead.

MM-005 asks, on exactly the same causal source rows:

1. Does changing only the target from `t+1.0 s` to `t+0.5 s` rescue a shared local
   spatial predictor?
2. Is any apparent rescue present in TAESD, only in identically sampled source pixels,
   or in neither domain?
3. Does the paired normalized advantage improve at the shorter horizon, rather than
   merely reflecting the smaller raw target displacement?

No encoder, decoder, media extraction, neural inference, new dataset, or
`FlatWorldModel` training is permitted. MM-005 replays authenticated MM-004 arrays with
a new pure-NumPy matched-horizon estimator.

## 2. Claim boundary

This is an outcome-informed diagnostic on eight already visible Perception Test sample
videos. It may identify a proximate sensitivity to the tested half-second horizon for
the frozen local-linear family. It cannot establish population prevalence, nonlinear
predictability, end-to-end Prospect capability, or that two learned half-steps compose
into a good one-second prediction.

In particular, the local predictors emit only the central valid `6 x 6` region. MM-005
does **not** roll that prediction forward. A clean half-second result licenses a later
teacher-free two-step rollout assay with a separately frozen boundary rule; it is not
itself evidence that rollout works.

No MM-001 through MM-004 result is reclassified.

## 3. Immutable parent receipts

The live MM-004 package must fast-verify before preparation, before formal execution,
and during verification. Its exact completed 15-file tree is recorded in the MM-005
input manifest. The live artifact-manifest hash is pinned to:

`eb17a9b324658de95546325f1bd7c8c4cd27c9c5914988efd65eb01bd94b794c`

MM-005 copies exactly seven MM-004 receipts under `inputs/MM-004`, preserving their
MM-004-relative paths and modes:

1. `artifact-manifest.json`
2. `input-manifest.json`
3. `MM-004-evidence.json`
4. `MM-004-results.json`
5. `MM-004-pixel-grids.npz`
6. `inputs/MM-003/inputs/MM-001/MM-001-features.npz`
7. `inputs/MM-003/inputs/MM-001/MM-001-component-audit.npz`

The copied artifact manifest authenticates the other six receipts; its own hash is the
hard trust anchor. Preparation also requires the copied MM-004 decision to be exactly
`tested_local_objective_or_horizon_failure_supported`, both parent synthetic controls
to pass, and the parent evidence/result summary relationship to replay exactly.

Strict loaders reject extra NPZ keys, pickle/object arrays, wrong dtype or shape,
non-finite values, out-of-range pixels, path additions, symlinks, mode drift, or hash
drift. The TAESD arrays are exactly `float32 [477,4,8,8]`; the pixel arrays are exactly
`float32 [477,3,8,8]`.

## 4. Exact common causal panel

Within each video, let the authenticated raw current-grid positions be `0..N-1` in
strict half-second timestamp order. Retain only source positions
`i = 1..N-3` inclusive, implemented as `range(1, N-2)`:

```text
previous     = current[i-1]   at t-0.5 s
current      = current[i]     at t
target_0.5s  = current[i+1]   at t+0.5 s
target_1.0s  = current[i+2]   at t+1.0 s
```

This produces exactly 453 common rows:

| Video | Rows |
|---|---:|
| `video_10993` | 60 |
| `video_1580` | 61 |
| `video_2564` | 56 |
| `video_3501` | 62 |
| `video_6860` | 62 |
| `video_8241` | 45 |
| `video_874` | 63 |
| `video_9253` | 44 |

Before the formal marker, both TAESD and pixel panels must satisfy bit-exactly on all
453 rows:

```text
target_0.5s == saved_MM004_target[i-1] == current[i+1]
target_1.0s == saved_MM004_target[i]   == current[i+2]
```

The identities, previous grids, current grids, folds, and train/test membership are
identical at both horizons. The canonical identity SHA-256 is
`d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b`.
Any cadence, count, identity, channel, alignment, or saved-target parity failure is
`invalid_MM005_parent_alignment` and must stop before a formal marker exists.

The four whole-video folds remain MM-001's six-train/two-test folds. Their training-row
counts are exactly `332`, `335`, `346`, and `346`. Videos are the only support units;
rows and spatial patches are repeated measurements.

## 5. Frozen real estimator

Only the two spatial arms that directly test MM-004's remaining horizon hypothesis are
replayed:

| Arm | Input per output cell | TAESD design | Role |
|---|---|---:|---|
| `current_3x3` | current `3x3` patch | 36 | spatial current-only comparator |
| `current_diff_3x3` | current and `current-previous` `3x3` patches | 72 | primary spatial-history arm |

The earlier pointwise arms are not reopened: they add no spatial capacity and cannot
discriminate the remaining hypothesis. The primary arm is fixed before outcomes;
results are never selected by ranking post-outcome MSEs.

Both arms predict the future residual `target_h-current` in the central valid `6x6`
region. Fits use float64 ridge, penalty `1e-3`, and an unpenalized intercept. For each
domain and fold, one mean and standard deviation per channel is fit on training
**current** grids, with scale floor `1e-6`. That exact normalizer is shared across both
horizons, arms, and controls. Previous, current, targets, differences, and residuals
use the same channel scale.

For a given domain, fold, arm, and control, paired horizon records must have identical
source identities, excluded videos, normalizer fingerprint, design matrix, and patch
count. Only the target/RHS may differ. Held-out values may enter no fitted statistic.

The real workload is fixed at 80 fits:

```text
2 domains x 2 horizons x 4 folds x
  (2 fits for current_3x3 + 3 fits for current_diff_3x3)
```

and exactly 64 held-out metric rows (`2 x 2 x 2 x 8`).

## 6. Baselines, controls, and primitive metrics

For every held-out video, horizon, domain, and arm save normalized MSE for:

- persistence: `prediction = current`;
- horizon-correct constant velocity:
  `prediction = current + (horizon / 0.5) * (current-previous)`;
- the ordered fit;
- a target-shuffled fit using a deterministic fixed-point-free within-video
  half-cycle permutation of training targets; and
- for `current_diff_3x3`, a history-shuffled fit using the same kind of permutation
  on training previous grids only.

The permutation for a domain/fold/control is reused across horizons. Test inputs and
targets are never shuffled. Target shuffle changes only the training target; history
shuffle changes only the training previous grid.

Each fit saves its horizon, feature dimension, train/excluded videos, row and patch
counts, source/input/target/design fingerprints, shared normalizer fingerprint, weight
shape and fingerprint, finite flag, and normalized linear-system residual. Metrics also
save raw and normalized persistence MSE, ordered/persistence ratio, shuffle advantages,
past/future delta energies and cosine, and the linked weight fingerprints.

No raw MSE is compared between horizons. The reported paired causal statistic is

```text
q_h = ordered_mse_h / persistence_mse_h
```

when `persistence_mse_h > 0`; otherwise the saved ratio is JSON `null`. Paired advantage
is false/ineligible unless both horizon-specific persistence MSEs are strictly positive.
For eligible rows, the gate is evaluated without division using

```text
1.10 * ordered_mse_0.5 * persistence_mse_1.0
    <= ordered_mse_1.0 * persistence_mse_0.5
```

It requires 6/8 videos. Thus a zero-persistence case is visible and deterministic,
cannot acquire an implicit `0/0` or infinity convention, and cannot pass as a trivial
`0 <= 0` horizon advantage.

## 7. New dual-horizon synthetic control

MM-004's sealed synthetic result is necessary parent evidence but does not validate the
new matched-horizon engine. MM-005 therefore runs a fresh positive/negative control
after the formal marker, using PCG64 seeds `550050`, `550051`, and `550052`, the exact
453 identities, folds, and `[4,8,8]` shape.

Independently smoothed fields `C` and `D` define `previous=C-D`, `current=C`, and:

```text
target_0.5 = C + 0.25 * (shift_right(C)-C) + 0.75 * D
target_1.0 = C + 0.50 * (shift_right(C)-C) + 1.50 * D
```

The rules have distinct known kernels and are exactly representable by
`current_diff_3x3`. Synthetic-only ordered `current_diff_1x1` and real-arm
`current_3x3` fits are structural ablations; target and history shuffle are temporal
controls. Each fold's synthetic train-current normalizer is shared across both horizons
and every synthetic arm/control. The frozen synthetic workload is 120 fits and 144
metric rows:

```text
3 seeds x 2 horizons x 4 folds x
  (main ordered/target-shuffle/history-shuffle + two ordered ablations)
```

Across all `3 seeds x 2 horizons x 8 videos = 48` main conditions require:

```text
main_mse <= 0.10 * persistence_mse
main_mse <= 0.50 * current_3x3_mse
main_mse <= 0.50 * current_diff_1x1_mse
main_mse <= 0.50 * target_shuffle_mse
main_mse <= 0.50 * history_shuffle_mse
```

Every fit must be finite with normalized linear-system residual at most `1e-10`.
Every ordered main fit must recover its named physical kernel with relative Frobenius
error at most `0.05` after undoing normalization, and must match its own horizon more
closely than the other horizon's kernel under strict, unnormalized Frobenius distance.
Failure of the main/numerical/kernel conditions is
`invalid_MM005_synthetic_positive_control`; failure of an ablation, shuffle, or
horizon-selector condition is `invalid_MM005_synthetic_negative_control`.

This panel validates target selection and fitting. It is not a physically consistent
teacher-free rollout trajectory.

## 8. Frozen real support gates

An ordered `current_3x3` row supports a video only when:

```text
ordered_mse * 1.20 <= persistence_mse
ordered_mse * 1.10 <= target_shuffle_mse
```

The primary `current_diff_3x3` row must additionally satisfy:

```text
ordered_mse * 1.10 <= history_shuffle_mse
```

An arm/horizon/domain passes at 6/8 videos, strongly fails at at most 2/8, and is
borderline at 3-5/8. Inclusive equality passes.

A target-shuffled fit itself supports a video when
`target_shuffle_mse * 1.20 <= persistence_mse`. If any frozen
domain/horizon/arm target-null reaches 6/8, the assay is
`invalid_MM005_real_negative_control`; 3-5/8 is
`inconclusive_MM005_real_negative_control`. This shortcut check precedes real
interpretation.

History-shuffle support does not globally invalidate the assay because its unchanged
current patch may predict. It blocks history attribution unless support is at most 2/8.

## 9. Source activity

Pixel activity is recomputed separately for each horizon on the exact common rows:

```text
P_h = mean((target_h-current)^2)
S_h = mean((within_video_half_cycle(target_h)-current)^2)
R_h = P_h / max(S_h, 1e-15)
active iff P_h >= 1e-4 and 0.10 <= R_h <= 1/1.2
```

Activity supports at 6/8, strongly fails at at most 2/8, and is heterogeneous at
3-5/8. Only half-second activity selects the short-horizon failure branches.

## 10. Frozen decision order

Predicates replay from primitive evidence in this exact order:

1. Parent, identity, cadence, or exact alignment failure:
   `invalid_MM005_parent_alignment`; stop before marker.
2. Synthetic numerical/main/kernel failure:
   `invalid_MM005_synthetic_positive_control`.
3. Synthetic ablation/shuffle/horizon-selector failure:
   `invalid_MM005_synthetic_negative_control`.
4. Any real target-shuffle null at 6/8:
   `invalid_MM005_real_negative_control`; any at 3-5/8:
   `inconclusive_MM005_real_negative_control`.
5. Any TAESD one-second arm passing 6/8:
   `matched_one_second_taesd_signal_supported`. This is common-row/endpoint
   sensitivity, not evidence for a horizon failure. A one-second TAESD arm at 3-5/8
   is `inconclusive_matched_one_second_taesd_signal`.
6. Otherwise, a same TAESD arm with half-second support at least 6/8, one-second
   support at most 2/8, and paired half-horizon advantage at least 6/8:
   `half_second_horizon_mismatch_supported`.
7. Any other TAESD half-second arm pass:
   `inconclusive_half_second_taesd_signal`.
   If no arm passes but any TAESD half-second arm is borderline at 3-5/8:
   `inconclusive_half_second_taesd_borderline`.
8. If TAESD does not pass, any pixel one-second arm pass is
   `matched_one_second_pixel_signal_supported`; 3-5/8 is
   `inconclusive_matched_one_second_pixel_signal`.
9. Otherwise, a same pixel arm with half-second support at least 6/8, one-second
   support at most 2/8, and paired advantage at least 6/8 is
   `half_second_taesd_representation_failure_supported`.
10. Any other pixel half-second arm pass is
    `inconclusive_half_second_pixel_signal`.
    If no arm passes but any pixel half-second arm is borderline at 3-5/8:
    `inconclusive_half_second_pixel_borderline`.
11. Only if every TAESD and pixel half-second arm strongly fails at at most 2/8 and
    half-second pixel activity is at least 6/8:
    `half_second_tested_spatial_local_linear_objective_failure_supported`.
12. With every half-second arm strongly failing and pixel activity at most 2/8:
    `insufficient_half_second_source_change`; activity at 3-5/8:
    `inconclusive_half_second_video_heterogeneity`.
13. Any uncovered valid combination: `MM005_diagnostic_inconclusive`.

The two arms are checked in fixed order, primary `current_diff_3x3` then comparator
`current_3x3`; all arm counts remain visible. A horizon-mismatch label names the arm
that satisfied all three same-arm conditions. No arm is chosen by lowest observed MSE.

Recommendations are frozen:

- clean TAESD half-horizon mismatch: implement a half-second target, then require a
  separately frozen teacher-free two-step rollout test before treating it as a fix;
- clean pixel-only mismatch: replace or temporally fine-tune TAESD before changing the
  dynamics model;
- matched one-second signal: isolate endpoint/common-row sensitivity before changing
  horizon;
- active pixels but no half-second local-linear signal: shortening alone is falsified
  for this family; next test a nonlinear causal warp/flow objective;
- inactive/heterogeneous source: curate or enlarge the independent video panel; or
- invalid/inconclusive controls: repair or replicate the assay before mechanism claims.

## 11. Evidence and sealed lifecycle

The canonical output is
`bench/multimodal_horizon_diagnostics/results/MM-005`. Preparation creates exactly nine
files: copied protocol, input manifest, and seven parent receipts. It derives the
matched panels in memory and writes no duplicate grid artifact. Preparation performs
only provenance, schema, alignment, and parent-parity checks; it performs no real or
synthetic fit and exposes no MM-005 outcome.

Formal execution revalidates all prepared inputs, then exclusively creates a mode
`0444` `formal-start.json` before the first fit. It writes evidence, results, and report
as mode `0644` exclusive artifacts, and writes the artifact manifest last. The
completed tree has exactly 14 files; 13 are artifact-bound before the manifest.
File and directory fsync are mandatory. Interruption after the marker permanently
consumes MM-005; retry, overwrite, or deletion is forbidden.

The marker binds protocol, input manifest, exact prepared membership, all seven receipt
hashes, alignment record, source set, environment, and complete frozen config. The
input manifest records the exact live 15-file MM-004 snapshot and the seven receipt
records.

Fast verification checks canonical location, exact membership/directories/modes,
symlinks, hashes, parent and source preservation, strict schemas, alignment parity,
primitive evidence, and exact result/report reconstruction. Semantic verification
first fast-verifies, then regenerates all 200 fits and compares identities,
fingerprints, and numeric values at `rtol=atol=1e-12`. It uses copied arrays only and
performs no media access, model inference, or network operation.

## 12. Source seal and pre-marker acceptance

MM-005 adds exactly seven source files to MM-004's sealed 47-file source set:

- `bench/multimodal_horizon_diagnostics/__init__.py`
- `bench/multimodal_horizon_diagnostics/__main__.py`
- `bench/multimodal_horizon_diagnostics/experiment.py`
- `bench/multimodal_horizon_diagnostics/method.py`
- this protocol
- `tests/test_mm005_experiment.py`
- `tests/test_mm005_method.py`

The new package contains exactly four Python files. No file is added to a sealed parent
package.

Before canonical preparation and execution:

- focused method and lifecycle/tamper tests pass;
- all decision branches and exact threshold boundaries are tested;
- exact 453-row alignment and paired source/design identities are tested;
- held-out mutation cannot change a normalizer or fit;
- shuffle invariants, patch order/dimensions, ridge parity, scale-floor behavior, and
  horizon-scaled velocity are tested;
- exact 80 real-fit/64 real-metric and 120 synthetic-fit/144 synthetic-metric
  memberships are tested;
- Ruff, formatting, strict Mypy, and whitespace checks pass;
- two independent pre-marker reviews return GO; and
- MM-001 through MM-004 remain verifiable and unchanged.

No real MM-005 predictor outcome may be inspected before all conditions pass and the
canonical formal marker exists.
