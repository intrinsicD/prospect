# MM-004 spatial/history signal-isolation protocol

**Date:** 2026-07-15  
**Status:** frozen before MM-004 real-video outcomes  
**Parent:** sealed MM-003 (`no_linear_full_taesd_signal_at_frozen_margin`)  
**Scope:** outcome-informed causal diagnostic on the same eight videos

## 1. Question

MM-003 showed that score scaling, the fixed nonorthogonal basis, the fixed 32-D
subspace, PCA-to-32, and even direct flattened 256-D linear prediction do not clear the
frozen temporal-prediction margin. It did not test whether flattening a single frame
omits two useful inductive biases: spatial weight sharing and explicit recent history.

MM-004 asks:

1. Can the assay recover a known history-dependent latent-grid process and reject
   broken temporal pairings?
2. Does a previous-to-current difference rescue TAESD prediction?
3. Does a shared local `3 x 3` map add predictive value beyond a pointwise map?
4. Does constant-velocity extrapolation expose coherent increments?
5. If TAESD fails, do identically sampled source pixels retain predictable motion, or
   are the eight clips themselves inadequate for this local-history assay?

No new `FlatWorldModel` is trained. A closed-form probe must first establish that a
representation exposes useful dynamics before another end-to-end trajectory is
scientifically justified.

## 2. Claim boundary

This is an outcome-informed diagnostic on eight already visible Perception Test sample
videos. It may identify a proximate sensitivity of these frozen TAESD grids under the
specified local-linear probes. It cannot establish population prevalence, TAESD's
general suitability, nonlinear predictability, full-dataset performance, or end-to-end
Prospect capability. Pixel results are measurement-only and never become model input.

No MM-001, MM-002, or MM-003 result is reclassified.

## 3. Immutable parent and inputs

The live MM-003 package must fast-verify before preparation. Its exact 19-file completed
tree is recorded in the MM-004 input manifest. The artifact-manifest hash is pinned to:

`ddb0449c29411578ced95b2b221a54e1733480d6c1e6d7b9c2203f4a011bf6f6`

MM-004 copies exactly seven parent receipts, preserving MM-003-relative paths:

1. `artifact-manifest.json`
2. `input-manifest.json`
3. `MM-003-evidence.json`
4. `MM-003-results.json`
5. `inputs/MM-001/MM-001-features.npz`
6. `inputs/MM-001/MM-001-component-audit.npz`
7. `inputs/MM-001/input-manifest.json`

The feature table supplies the exact 477 video identities and timestamps. The component
audit supplies `taesd_latents` and `target_taesd_latents`, both exactly
`float32 [477,4,8,8]`. Strict loaders reject extra keys, wrong dtype/shape, pickle,
object arrays, non-finite values, path additions, symlinks, mode drift, or hash drift.

The nested MM-001 manifest pins the eight cached MP4 hashes and exact 2-fps,
letterboxed `64 x 64` decoding contract. During preparation only, MM-004 verifies every
media hash, uses MM-001's source-bound decoder, area-pools each authenticated RGB frame
to `float32 [3,8,8]`, and writes `MM-004-pixel-grids.npz`. Current and one-second target
pixel grids share the TAESD arrays' 477 identities. The prepared pixel file is
hash-bound by the MM-004 input manifest; formal execution and semantic verification
need no live media or decoder.

Before the formal marker, MM-004 reconstructs the C-order flattened raw table and
reproduces all 16 saved MM-003 `raw256_native` probe rows (eight `absolute_target` and
eight `residual_delta`) at `rtol=atol=1e-12`. Failure is terminal
`invalid_MM004_parent_parity`; it is not a scientific outcome.

## 4. Real history panels

Within each video, rows are stable-sorted by timestamp. MM-004 retains source positions
`1:`:

- position `i-1` is the previous frame at `t-0.5 s`;
- position `i` is the current frame at `t`;
- the saved target at `i` is exactly `t+1.0 s`.

Every MM-001 target is already authenticated one second after its current input, so no
terminal rows are removed. This yields exactly 469 rows, with per-video counts equal to
the MM-001 window counts minus one. Both TAESD and pixel branches use these identities.

The four whole-video outer folds remain MM-001's six-train/two-test folds. No cell,
patch, or row from a held-out video may enter any fitted statistic. Videos are the only
support units; spatial patches and timestamps are repeated measurements.

## 5. Predictor ladder

All learned arms predict the future residual `target - current` in a normalized grid,
use float64 ridge with penalty `1e-3`, and leave the intercept unpenalized.

| Arm | Input per output cell | TAESD design | Purpose |
|---|---|---:|---|
| `current_1x1` | current center | 4 | minimal current-only comparator |
| `current_diff_1x1` | current and `current-previous` centers | 8 | history without neighbors |
| `current_3x3` | current `3x3` patch | 36 | spatial neighborhood without history |
| `current_diff_3x3` | current and difference `3x3` patches | 72 | spatial neighborhood plus history |

All arms predict only the central valid `6x6` region, avoiding padding and the strongest
letterbox-boundary artifacts. A train-current mean and standard deviation per channel,
shape `[channels,1,1]` and floor `1e-6`, standardizes previous, current, target,
differences, and residuals. Patches are pooled only after train-only normalization. The
same scale is used for all arms, so zero residual is exactly persistence. The main
TAESD fit has 292 weights and at least 12,384 training patch rows per fold.

Every fit saves feature dimension, train/excluded videos, row and patch counts, input
identity and matrix hashes, weight fingerprint, numerical residual, and finite flags.

## 6. Baselines and temporal controls

For every held-out video and arm save:

- persistence: `prediction = current`;
- constant velocity: `prediction = current + 2 * (current - previous)`;
- the ordered arm prediction;
- a target-shuffled fit using a deterministic within-video half-cycle derangement of
  training targets while keeping previous/current inputs fixed; and
- for history arms, a history-shuffled fit that deranges training previous grids while
  keeping current/target pairs fixed.

Null fits use identical rows, dimensions, ridge penalty, and held-out inputs. Test
histories and targets are never shuffled. Derangements remain within video and have no
fixed points.

Saved held-out metrics are normalized MSEs for persistence, constant velocity, ordered,
target-shuffle, and applicable history-shuffle predictors; ordered/persistence and
shuffle-advantage ratios; past/future delta cosine and energies; plus raw pixel
persistence and half-cycle-target persistence MSEs for source activity.

## 7. Deterministic synthetic control

The positive control uses `numpy.random.Generator(PCG64(seed))`, never global RNG or a
Python hash. It creates three panels with seeds `440040`, `440041`, and `440042`, each
using the exact 469 identities, eight groups, folds, and `[4,8,8]` shape.

Independently sampled spatially smoothed float64 fields `C` and `D` define each triplet:

```text
previous = C - D
current  = C
target   = C + 0.5 * (shift_right(C) - C) + 1.5 * D
```

`shift_right` is a one-cell neighbor visible in a valid `3x3` patch. The rule is exactly
representable by `current_diff_3x3`; `current_3x3` lacks `D`, and
`current_diff_1x1` lacks the neighboring current cell. Generator configuration, array
fingerprints, identities, recovered kernels, and primitive control rows are saved.

The positive/negative control gate passes only if all 24 held-out video/panel
conditions satisfy:

```text
main_mse <= 0.10 * persistence_mse
main_mse <= 0.50 * current_3x3_mse
main_mse <= 0.50 * current_diff_1x1_mse
main_mse <= 0.50 * target_shuffle_mse
main_mse <= 0.50 * history_shuffle_mse
```

Every solution must be finite with normalized linear-system residual at most `1e-10`,
and every fold/panel main fit must recover the true kernel with relative Frobenius error
at most `0.05` after undoing normalization. Positive/numerical failure emits
`invalid_MM004_synthetic_positive_control`; ablation or shuffle-separation failure
emits `invalid_MM004_synthetic_negative_control`. Both precede real interpretation.

## 8. Frozen real support and contrasts

An ordered no-history arm supports a video only when:

```text
ordered_mse * 1.20 <= persistence_mse
ordered_mse * 1.10 <= target_shuffle_mse
```

A history arm must additionally satisfy:

```text
ordered_mse * 1.10 <= history_shuffle_mse
```

An arm passes at 6/8 videos. Constant velocity supports a video when its MSE times
`1.20` is at most persistence MSE and passes at 6/8.

A target-shuffled control itself clearing the 20%-over-persistence rule on 6/8 videos
is `real_negative_control_failed`; 3-5/8 is `real_negative_control_inconclusive`.
This is the global shortcut check and precedes all real interpretation.

A history-shuffled main arm may legitimately beat persistence through its unchanged
current `3x3` block. It therefore never globally invalidates the assay. Support on 3/8
or more only blocks history/temporal attribution and emits a descriptive
`history_control_predictive_from_current` label; spatial/current-only branches remain
interpretable. History attribution requires the history-shuffled control to support at
most 2/8 in addition to the ordered-versus-history-shuffle margin.

Mechanism attribution additionally requires the named 10% paired improvement on 6/8:

1. history contribution: `current_diff_3x3` versus `current_3x3`;
2. spatial-neighborhood contribution: `current_diff_3x3` versus
   `current_diff_1x1`; and
3. structured advantage: `current_diff_3x3` versus constant velocity.

No mechanism is attributed from an unpaired arm pass.

## 9. Source-pixel measurement branch

The same ladder, persistence, constant velocity, shuffles, folds, support gates, and
central `6x6` evaluation run on prepared `[3,8,8]` pixel grids. Pixels diagnose source
signal only; they never enter Prospect or a TAESD predictor.

For each video, define raw source activity:

```text
P = mean((target - current)^2)
S = mean((half_cycle_deranged_target - current)^2)
active iff P >= 1e-4 and 0.10 <= P/S <= 1/1.2
```

Activity supports the sample at 6/8 videos, strongly fails at at most 2/8, and is
heterogeneous at 3-5/8. Target-informed translation oracles are excluded from MM-004:
they cannot support prediction or select a branch, and omitting them avoids an unused
post-outcome degree of freedom.

## 10. Frozen decision order

Predicates replay from saved primitive rows in this exact order:

1. Parent/index parity failure: `invalid_MM004_parent_parity`; stop before marker.
2. Synthetic main-fit/kernel/numerical failure:
   `invalid_MM004_synthetic_positive_control`.
3. Synthetic ablation/shuffle separation failure:
   `invalid_MM004_synthetic_negative_control`.
4. A real target-shuffle null reaching 6/8:
   `invalid_MM004_real_negative_control`; 3-5/8:
   `inconclusive_MM004_real_negative_control`.
5. Any TAESD arm passing 6/8: `taesd_local_linear_signal_supported`, with only
   history, spatial-neighborhood, and structured-advantage labels whose paired 6/8
   contrasts pass. A history label additionally requires history-shuffle support at
   most 2/8; otherwise add `history_control_predictive_from_current` and retain only
   non-history attributions.
6. All TAESD arms fail and any pixel arm passes 6/8:
   `taesd_representation_failure_supported`.
7. All TAESD and pixel arms fail but source activity supports 6/8:
   `tested_local_objective_or_horizon_failure_supported`.
8. All TAESD and pixel arms fail and source activity supports at most 2/8:
   `data_dynamics_insufficient_for_local_history_assay`.
9. All TAESD and pixel arms fail and source activity supports 3-5/8:
   `inconclusive_video_heterogeneity`.
10. Any uncovered valid combination: `MM004_diagnostic_inconclusive`.

Every decision saves all arm, null-control, contrast, and activity counts.
Recommendations are frozen:

- TAESD signal: implement only supported history/spatial factors in a tiny adapter,
  then test it end to end;
- pixel pass / TAESD fail: replace or temporally fine-tune the visual frontend;
- source active / both predictors fail: change the local objective or horizon before
  changing data;
- source inactive: curate deliberately dynamic clips before changing model size; or
- heterogeneous/inconclusive: enlarge the independently sampled video panel first.

## 11. Evidence and verification

The canonical output is
`bench/multimodal_spatial_diagnostics/results/MM-004`. Preparation creates exactly ten
files: copied protocol, input manifest, seven parent receipts, and prepared pixel grids.
The run creates `formal-start.json`, evidence, results, and report by exclusive creation,
then writes the artifact manifest last. The completed tree has exactly 15 files.

The marker is mode `0444`; all other generated files are `0644`. File and directory
fsync are mandatory. Interruption after the marker permanently consumes MM-004. No
retry, deletion, or overwrite is permitted.

The input manifest hash-binds the complete live MM-003 snapshot, seven receipts, exact
47-file source set (40 parent plus seven MM-004 additions), Python/NumPy/Prospect
environment, authenticated media identities, prepared pixel hash/extraction contract,
and every shape, seed, fold, arm, shuffle, metric, threshold, and decision rule.

Fast verification checks canonical location, membership, modes, symlinks, hashes,
parent/source preservation, strict schemas, 16-row parent parity, synthetic and pixel
provenance, and exact reconstruction of results and report from primitive evidence.
Semantic verification first fast-verifies, then regenerates all synthetic and real fits
from copied inputs with exact identities/fingerprints and `rtol=atol=1e-12` numeric
agreement. It performs no media or neural inference.

## 12. Pre-marker acceptance

Before preparation and formal execution:

- focused method and lifecycle tests pass;
- every decision branch and threshold boundary has a unit test;
- train/test leakage and within-video derangement tests pass;
- raw-grid C-order reconstruction and all 16 parent rows reproduce;
- authenticated pixel extraction is deterministic;
- Ruff, strict Mypy, and whitespace checks pass;
- independent science and lifecycle/security reviews return GO; and
- MM-001, MM-002, and MM-003 remain verifiable and unchanged.

No real MM-004 predictor outcome may be inspected before these conditions and canonical
preparation are complete.
