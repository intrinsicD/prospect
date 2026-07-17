# MM-003 — TAESD projection/scale isolation protocol

Status: **frozen before formal execution**  
Date: 2026-07-15  
Experiment ID: `MM-003`

## 1. Question and scope

MM-001 found no supported one-second visual world-model result on the sealed
eight-video Perception Test sample. MM-002 found no rescue from horizon, world
budget, codec order, codec sharing, or codec budget, while the nominal learned
representation failed frozen integrity sentinels and its fixed 32-D ridge probe
failed against persistence.

MM-003 asks whether MM-001's fixed dense 256-to-32 TAESD projection hid usable
temporal information, or whether scale/basis conditioning prevented the
unchanged Prospect world model from using retained information.

This is an outcome-informed diagnostic on the same eight videos. It can identify
a proximate mechanism and compatible fix on this sample. It is not independent
confirmation, does not reclassify MM-001, and makes no population claim. No
media decoding, model download, neural frontend, or codec training is allowed.

## 2. Immutable parents and preservation

Canonical parents:

- `bench/multimodal_preflight/results/MM-001`
- `bench/multimodal_diagnostics/results/MM-002`

Before preparation, immediately before the formal marker, and after outcomes,
both parents must pass their fast verifiers in the frozen environment
(`PYTHONPATH=src`, NumPy `2.1.3`). Their complete package snapshots and bound
source maps must remain unchanged.

MM-003 copies exactly these inputs after validating them against the respective
recursive artifact manifest.

MM-001:

- `artifact-manifest.json`
- `input-manifest.json`
- `MM-001-features.npz`
- `MM-001-component-audit.npz`
- `MM-001-projections.npz`
- `MM-001-results.json`

MM-002:

- `artifact-manifest.json`
- `input-manifest.json`
- `MM-002-evidence.json`
- `MM-002-results.json`

Pinned SHA-256 digests:

- MM-001 artifact manifest:
  `a394104a6e9bcdb6c18b206d090e4afb9a540b9e3a2a2875985980e23ecaf52c`
- MM-001 features:
  `3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755`
- MM-001 component audit:
  `476da8f2192c6bd57ecab6f861e975fc0827977fa8081462423fa4644e0c89e4`
- MM-001 projections:
  `b131039b540735b0942f9608ab0ebda5a3ccc2018ec9126fdcbaa3b44f9aaaea`
- MM-001 result:
  `16504f4bfb36e5252aea9aa6604bc88d64233e256d184bf0e3b2889f5fd76fb7`
- MM-002 artifact manifest:
  `3e119c35f4a6731df88e68bd16fc7b4e8d44c37776ad72ef660b60681628c139`
- MM-002 evidence:
  `093da00fbc8ef8a68cc8922f463d9febd92fb02f3d9db2b5f5bdb8660f0dbaaa`
- MM-002 result:
  `5bf8cb1e37847cced02e304dac07b1b816cb5453d6342b6e08e353354d2953fb`

The receipt also records the full live tree of each parent. MM-002's embedded
MM-001 receipt must agree with the live MM-001 pins. After the formal marker,
analysis reads only copied inputs.

MM-003 source binding is MM-002's sealed source set plus exactly this protocol,
`bench/multimodal_transform_diagnostics/*.py`, and
`tests/test_mm003_{method,experiment}.py`. MM-003 must not modify either parent,
`Makefile`, `pyproject.toml`, or `src/prospect/**`.

## 3. Raw panel and mandatory parity

The component audit supplies float32 `taesd_latents` and
`target_taesd_latents`, each `[477,4,8,8]`. Flatten them in NumPy C order to
float64 `[477,256]`. Identities remain the exact MM-001 `(video_id,timestamp)`
order. The sealed vision projection is float64 `[256,32]`.

Interpretation requires all of the following:

1. raw current/target multiplied by the sealed matrix reproduce all saved
   MM-001 vision/target features with `rtol=atol=1e-6`;
2. inherited `r32_native` 1,500-update rows reproduce all 24 MM-002
   `full_1s_1500` rows, metrics, and fingerprints at `1e-12`;
3. inherited `r32_native` integrity at 300/600/1,500 reproduces all 144 MM-002
   `full_1s` rows exactly;
4. the native fixed-penalty absolute ridge reproduces MM-002's eight
   `matched_1s` raw-probe rows at `1e-12`.

Failure is `invalid_MM003_parent_parity`, not a scientific result.

Use the four original whole-video folds (six train, two test) and seeds
`0,1,2`. World models use all 477 rows. Linear probes use MM-002's matched
461-source panel: remove the last two sources per video and retain each saved
one-second target.

## 4. Frozen representation arms

Let `u` be one raw 256-D row and `R` the sealed MM-001 matrix.

| Arm | Transformation | Isolated factor |
|---|---|---|
| `r32_native` | `u @ R` | exact baseline |
| `r32_postz` | train-current z-score of `u @ R` | scale only, same information/subspace |
| `raw256_native` | `u` | remove compression |
| `raw256_postz` | train-current coordinate z-score of `u` | full-information scale |
| `r32_qr_postz` | z-scored scores under an orthonormal basis of `col(R)` | basis conditioning, same subspace |
| `pca32_postz` | z-scored top-32 train-current PCA scores | alternative 32-D subspace |

The post-z and QR controls are load-bearing: without them PCA confounds scale,
nonorthogonal basis conditioning, and subspace information.

Every fitted statistic uses only the six outer-training videos' **current** raw
rows. Targets and held-out videos never fit a transform. Apply the frozen
transform unchanged to all current/target rows. Population standard deviations
use `scale=max(std(ddof=0),1e-6)`. No clipping, per-video normalization, or
test-time recentering is allowed.

QR uses reduced QR of `R`, with each basis column signed so its
largest-absolute loading is positive. Require
`max_abs(Q@Q.T - R@pinv(R)) <= 1e-10`.

For PCA, center outer-training current raw rows, run deterministic full SVD,
take the first 32 right-singular vectors, apply the same sign rule, and z-score
training scores. Emit `pca_rank_below_32` when `s[31]/s[0] < 1e-8`. Emit
`pca_boundary_degenerate` when `(s[31]-s[32])/s[31] < 1e-6`; the PCA result then
cannot support a stable subspace-cause label.

Persist each fold/arm's train/excluded IDs, ordered identity hash, raw fit-matrix
hash, parameter arrays/hashes, output dimension, fingerprint, PCA spectrum and
variance retention where applicable, and QR projector error. Mutating held-out
values must not change the transform.

## 5. Scale-neutral linear probes

Run all six arms with two preregistered predictor forms:

- `absolute_target`: predict the transformed next row;
- `residual_delta`: predict transformed `(next-current)` and add it to current.

The residual form is persistence-aware: zero predicted delta is exactly
persistence.

Within each fold/arm, independently standardize training predictors and ordered
training targets (or deltas) using training rows only. Apply those statistics
unchanged to held-out rows. Ridge penalty is `1e-3`; the intercept is
unpenalized. The equal-compute shuffle ridge uses the same penalty with training
targets deranged by the frozen within-video half-cycle mapping. Ordered target
statistics are also used for shuffle scoring.

Errors are evaluated in ordered training-target-standardized coordinates.
A video supports a probe when:

```text
ridge_mse * 1.2 <= persistence_mse
ridge_mse * 1.1 <= shuffled_ridge_mse
```

An arm/predictor passes at `6/8` videos. Report `ridge/persistence` and
`shuffle/ridge` ratios.

The neutral probe must make `r32_native/r32_postz` and
`raw256_native/raw256_postz` metrics agree at `rtol=1e-8,atol=1e-10`.
Disagreement is invalid numerical/implementation evidence, not scale effect.

For parent parity only, also recompute MM-002's unstandardized native 32-D
absolute ridge/persistence/shuffle metrics with penalty `1e-3`.

## 6. Smallest end-to-end world test

Test the architecture-compatible 32-D arms at MM-001's original budget:

- `r32_native`, inherited exactly from MM-002 and never retrained;
- `r32_postz`;
- `r32_qr_postz`;
- `pca32_postz`.

New arms use unchanged MM-001 hyperparameters, folds, seeds, batch-index stream,
temporal derangement, and primary/shuffle pairing. The sole change is the
fold-fitted input representation. Train uninterrupted to 1,500 updates and copy
checkpoints at 300, 600, and 1,500 without consuming RNG.

This is update-matched, not FLOP-matched. Raw-256 world training is deferred:
the model-free full-256 arms first determine whether retaining all dimensions is
warranted. This keeps MM-003 the smallest end-to-end cause test.

For every held-out video/seed save world, persistence, diagnostic latent-ridge,
shuffle-world, and shuffle-persistence MSE. Raw MSE is not compared across arms.
After seed medians, a video supports an arm when:

```text
world_mse * 1.2 <= persistence_mse
world_mse < diagnostic_latent_ridge_mse
(world_mse/persistence_mse) * 1.1
    <= shuffle_world_mse/shuffle_persistence_mse
```

An arm passes at `6/8` videos. A candidate materially improves over comparator
`c` when its world/persistence ratio is at least 10% lower on `6/8` paired
videos. A cause label requires this paired improvement, not only a gate crossing.

## 7. Representation integrity

At 300, 600, and 1,500 updates, pool both held-out videos per fold/seed and
probe primary/shuffle models with online/EMA-target encoders. Every row requires:

```text
minimum per-dimension latent std >= 0.3
effective rank >= 2.0
all predictive quantities finite
all predictive variances > 0
```

An arm is healthy only if every row passes. An unhealthy crossing is
`apparent_rescue_via_representation_collapse` and cannot be a fix.

## 8. Frozen interpretation and fixes

Nonexclusive probe labels:

- `persistence_aware_parameterization_rescue_<arm>`: residual passes while
  absolute fails;
- `fixed_random_subspace_linear_signal_loss_supported`: stable PCA-32 passes,
  every `r32_native`/`r32_postz`/`r32_qr_postz` predictor fails, and a full-256
  probe passes;
- `tested_32d_compression_linear_signal_loss_supported`: full-256 passes while
  every tested fixed-subspace predictor and PCA-32 fail;
- `no_linear_full_taesd_signal_at_frozen_margin`: neither predictor passes for
  the full-256 scale pair.

Define `rescue(a,c)` at 1,500 updates as: `a` is healthy, passes the world gate,
and materially improves over `c`.

Primary cause labels:

- `mm001_coordinate_scale_cause_supported`:
  `rescue(r32_postz,r32_native)` while native fails or is unhealthy;
- `fixed_subspace_basis_conditioning_cause_supported`:
  `rescue(r32_qr_postz,r32_postz)` while post-z Rademacher fails;
- `fixed_random_subspace_information_loss_supported`:
  `rescue(pca32_postz,r32_qr_postz)`, stable PCA/full-256 probes pass, and every
  predictor in the exact fixed subspace fails.

Secondary labels:

- `pca_subspace_world_sensitivity_only`: PCA rescues the world contrast without
  model-free differential signal;
- `linear_temporal_signal_present_world_path_not_supported`: a tested 32-D arm
  passes a probe but no corresponding new arm is healthy and world-passing;
- `full_information_signal_present_no_compatible_32d_fix`: full-256 passes but
  no tested 32-D arm is an admissible end-to-end fix;
- `tested_projection_scale_factors_not_supported`: no admissible contrast.

Multiple primary labels yield `multiple_projection_scale_mechanisms_supported`.
One yields that label as classification. With none, choose the most specific
bounded secondary label above, ending in `tested_projection_scale_factors_not_supported`.

Mechanical recommendation:

- scale: freeze train-current mean/std after the existing projection;
- basis: replace `R` by its canonical orthonormal basis and freeze score stats;
- PCA information: fit/freeze train-current PCA32 and score stats;
- full-information only: test a wider or supervised temporal projection next;
- no full linear signal: stop projection tuning and test a spatial/difference-
  aware visual objective or more dynamic data.

## 9. Evidence lifecycle and verification

Canonical output is
`bench/multimodal_transform_diagnostics/results/MM-003`.
Prepared membership is exactly the protocol copy, input manifest, and ten
selected parent files. It contains no marker or outcomes.

The one-shot formal marker uses `O_CREAT|O_EXCL`, mode `0444`, and file/directory
fsync. It binds the manifest/protocol/source/config digests, both parent root
manifest hashes, and prepared-membership digest. Outcomes are exclusive-created
and fsynced:

- `formal-start.json`
- `MM-003-transforms.npz`
- `MM-003-transform-records.json`
- `MM-003-evidence.json`
- `MM-003-results.json`
- `MM-003-report.md`
- recursive `artifact-manifest.json` last.

Interruption after the marker permanently consumes MM-003; no resume/overwrite
is allowed. Fast verification enforces exact files/directories, no symlinks,
source/parent preservation, receipts, schemas, identities, reconstructed
transforms, fingerprints, parity, branches, report, modes, and recursive
manifest. Semantic verification uses copied inputs only, reconstructs every
transform, retrains all new trajectories, and compares arrays/fingerprints
exactly and floats at `1e-12`. It never invokes media or neural frontends.

## 10. Claim boundary

An admissible label is a proximate mechanism on eight outcome-visible videos
under one fixed projection and one NumPy world model. One projection does not
characterize random projections generally. PCA is unsupervised. A failed linear
probe excludes only linear one-second signal at the frozen margin. Flattening
retains TAESD values but discards explicit spatial structure, so a null does not
clear convolutional, nonlinear, motion-specific, or larger-data alternatives.
