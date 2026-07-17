# MM-002 frozen feature-only failure-isolation protocol

**Status:** frozen before formal execution  
**Date frozen:** 2026-07-15  
**Scope:** outcome-informed diagnostic of MM-001; non-gated  
**Experiment ID:** `MM-002`

## Question and claim boundary

MM-001 established a valid small-system negative result: its one-second visual dynamics
and vision-codec migration gates each supported 0/8 Perception Test sample videos. MM-002
asks the smallest follow-up questions that require no new media or neural inference:

1. Is the visual-dynamics failure sensitive to the tested prediction horizon or to the
   fixed 1,500-update world-model budget?
2. Is the vision-codec migration failure sensitive to shared-trunk modality interference,
   within-cycle modality order, or the fixed 600-cycle distillation budget?

This experiment is explicitly informed by MM-001's visible outcome. It is a diagnostic,
not an independent replication, benchmark estimate, or rescue attempt. Crossing a frozen
MM-001 gate under one changed factor identifies sensitivity on this sample; it does not
establish a general causal mechanism. No threshold is relaxed and MM-001 is never
reclassified.

## Immutable parent and isolation boundary

The parent is the exact verified package at
`bench/multimodal_preflight/results/MM-001/`, whose fixed identities include:

- artifact-manifest SHA-256
  `a394104a6e9bcdb6c18b206d090e4afb9a540b9e3a2a2875985980e23ecaf52c`;
- result SHA-256
  `16504f4bfb36e5252aea9aa6604bc88d64233e256d184bf0e3b2889f5fd76fb7`;
- feature-package SHA-256
  `3fdf0c988cf0bdb428432b67c71fc7a18404080b6e12bfe8b6226d2276330755`;
- classification `real_visual_temporal_prediction_not_supported`.

Preparation must first run MM-001's fast verifier, require `verified_results`, reject
source or package drift, then copy all 14 parent files into MM-002's prepared input
directory. The copied set, byte sizes, permissions, and SHA-256 hashes are checked
against both the live parent and the parent's manifest. Formal MM-002 reads only the
copied feature/result evidence. It never writes into MM-001.

MM-002 implementation and tests live outside every MM-001 source glob. It does not
modify `src/prospect`, `Makefile`, `pyproject.toml`, the MM-001 protocol,
`bench/multimodal_preflight/`, or `tests/test_multimodal_preflight*.py`. MM-001 must
fast-verify before preparation, immediately before the MM-002 marker, and after MM-002.

## Units, folds, and seeds

The experimental unit remains one of MM-001's eight videos. Reuse its four mechanically
fixed whole-video folds and optimizer seeds `0, 1, 2`. Seeds and timestamps are repeated
measurements. Every endpoint is first reduced within a video/seed, then by the median
over three seeds. The unchanged support floor is at least `6/8` videos.

The full-parent baseline uses all 477 rows. The matched horizon ladder uses the same
source rows for every horizon by dropping exactly the final two rows of each video:

```text
{video_10993: 61, video_1580: 62, video_2564: 57, video_3501: 63,
 video_6860: 63, video_8241: 46, video_874: 64, video_9253: 45}
```

This totals 461 rows. Within a video's ordered half-second grid, matched source row `i`
uses:

- 0.5-second target: `vision[i + 1]`;
- 1.0-second target: `target_vision[i]`;
- 2.0-second target: `target_vision[i + 2]`.

The last definition is valid because `target_vision[j]` is the frozen TAESD feature at
one second after row `j`; row `i + 2` is one second after row `i`. Source identities,
folds, counts, and feature-package bytes are invariant across matched variants.

## World-model ladder

Each trajectory trains both the ordinary model and its equal-compute within-video
half-cycle temporal-shuffle control with the unchanged MM-001 `FlatWorldModel`. The
matched one-second trajectory is continued without resetting its model, optimizer,
sampling generator, EMA target, or input statistics; checkpoints are evaluated after
exactly 1,500, 3,000, and 6,000 completed updates:

| ID | Rows | Horizon | Updates | Purpose |
|---|---:|---:|---:|---|
| `full_1s_1500` | 477 | 1.0 s | 1,500 | exact MM-001 parity control |
| `matched_0p5s_1500` | 461 | 0.5 s | 1,500 | shorter-horizon sensitivity |
| `matched_1s_1500` | 461 | 1.0 s | 1,500 | matched diagnostic baseline |
| `matched_1s_3000` | 461 | 1.0 s | 3,000 | two-times update-budget checkpoint |
| `matched_2s_1500` | 461 | 2.0 s | 1,500 | longer-horizon sensitivity |
| `matched_1s_6000` | 461 | 1.0 s | 6,000 | update-budget sensitivity |

All other parameters remain exactly MM-001: 32-dimensional input, 8-dimensional latent,
hidden width 64, ensemble 5, learning rate `3e-3`, EMA `0.995`, variance/covariance
weights 25/1, disabled reward/inverse losses, batch 64 with replacement from
`default_rng(seed + 1)`, null action, and identical temporal derangement.

For each held-out video save primary and shuffle model fingerprints plus:

```text
world_mse
persistence_mse
ridge_mse
shuffle_model_mse
shuffle_model_persistence_mse
```

A video supports a variant only under MM-001's unchanged conjunction:

```text
world_mse * 1.2 <= persistence_mse
world_mse < ridge_mse
(world_mse / persistence_mse) * 1.1
    <= shuffle_model_mse / shuffle_model_persistence_mse
```

`full_1s_1500` must reproduce every corresponding MM-001 fold/seed/video metric and
primary-model fingerprint within `rtol=1e-12, atol=1e-12`; mismatch invalidates MM-002.

### Raw-feature temporal-signal probe

For every fold and each matched horizon, fit a deterministic 32-D ridge directly in
the frozen feature space. This probe has no optimizer seeds. Let `X = [vision_t, 1]`,
`Y = target_horizon`, and use penalty `1e-3 * I` with the intercept unpenalized. Save
held-out-video MSE for raw persistence, the ordered ridge, and a control ridge fitted
to the same `X` against `Y[temporal_derangement(train)]`. A video supports linear raw
predictability only when:

```text
raw_ridge_mse * 1.2 <= raw_persistence_mse
raw_ridge_mse * 1.1 <= raw_shuffle_ridge_mse
```

The panel threshold remains 6/8. A raw pass with a learned-world failure means that
the projected 32-D features contain usable linear temporal information that this
learned latent/dynamics path did not exploit. A raw failure means only no linear
evidence at this frozen margin; it cannot exclude nonlinear signal or information
discarded by the fixed 256-to-32 projection.

### Representation-integrity requirement

After exactly 300, 600, and 1,500 completed updates on every trajectory, and also
3,000 and 6,000 updates on the continued one-second trajectory, probe both held-out
videos pooled within each fold/seed. Probe both primary and shuffle models and both
their online and EMA-target encoders. For each `n x 8` latent matrix `Z`, record the
population per-dimension minimum standard deviation and
`sum(eigenvalues)^2 / sum(eigenvalues^2)` for
`cov(Z.T) + 1e-8 I`. Also require all held-out prediction means, variances,
epistemic values, and aleatoric values to be finite, with every variance positive.

A trajectory checkpoint is healthy only if every applicable probe in all 12
fold/seed runs has minimum standard deviation at least `0.3`, effective rank at least
`2.0`, and finite predictions. The original visual gate remains unchanged, but a new
factor is an admissible diagnostic rescue only when its trajectory is healthy.
An unhealthy nominal full baseline emits `baseline_representation_integrity_failure`
and prevents horizon/budget attribution. A frozen-gate crossing on an unhealthy
candidate emits `apparent_rescue_via_representation_collapse`, not a rescue. A later
checkpoint that becomes unhealthy emits `extended_training_representation_instability`.

### World diagnosis

World diagnosis is mechanical after parent parity and integrity checks:

1. If the full baseline is unhealthy, emit its integrity failure and make no causal
   horizon/budget attribution.
2. If healthy `matched_1s_1500` reaches 6/8 while the reproduced 477-row parent remains
   0/8, emit `endpoint_truncation_sensitive`; do not interpret horizon or budget.
3. Otherwise, a healthy 0.5-second crossing emits `short_horizon_rescue`, a healthy
   2.0-second crossing emits `long_horizon_rescue`, and both emit
   `broad_horizon_rescue`.
4. Healthy crossings at both 3,000 and 6,000 emit `stable_world_budget_rescue`; only
   6,000 emits `late_four_x_world_budget_rescue`; 3,000 pass followed by 6,000 fail
   emits `world_overtraining_or_nonmonotonicity`; neither emits
   `not_rescued_through_four_x_budget`.
5. Record raw-feature conclusions separately. Multiple admissible labels remain
   explicitly ambiguous rather than being collapsed to one cause.

## Vision-codec ladder

Reuse each `full_1s_1500` primary world model and its six-video training fold. The
incumbent target is always `model.encode(vision_t).z`; held-out evaluation always uses
the same model's one-second target latent. Decoder-head updates are omitted because they
cannot change adapters or the shared trunk and are outside the migration endpoint.

Every codec retains MM-001 dimensions and optimization: 32-dimensional modality inputs,
8-dimensional latent, token width 32, hidden width 64, learning rate `3e-3`, seed
`seed + 1`, one full-fold initialization update per included modality, batch 128 with
replacement from `default_rng(seed + 303)`, and the stated number of cycles.

| ID | Included modalities | Per-cycle order | Cycles | Purpose |
|---|---|---|---:|---|
| `shared_vat_600` | vision/audio/text | V,A,T | 600 | exact MM-001 encode parity |
| `shared_atv_600` | vision/audio/text | A,T,V | 600 | nominal order screen |
| `vision_only_600` | vision | V | 600 | cross-modal shared-training screen |
| `shared_vat_2400_after_v` | vision/audio/text | V,A,T, pre-final-A/T snapshot | 2,400 | terminal recency screen |
| `shared_vat_2400` | vision/audio/text | V,A,T | 2,400 | shared-codec budget screen |
| `vision_only_2400` | vision | V | 2,400 | isolated-adapter budget screen |

Only three codec trajectories are trained per fold/seed: shared VAT continued to 2,400
cycles, shared ATV stopped at 600, and vision-only continued to 2,400. Checkpoints are
deep-copied without consuming randomness or changing subsequent updates. The modality
dictionary and per-modality network initialization remain VISION, AUDIO, TEXT ordered
even for the `shared_atv_600` update schedule, so only update order changes. The
`shared_vat_2400_after_v` checkpoint is taken after the final V update but before that
cycle's final A/T updates and tests terminal recency only; it is not a full high-budget
ATV trajectory.

The initial full-fold statistics calls are optimizer updates. Shared codecs therefore
receive 601 or 2,401 updates per modality (1,803 or 7,203 shared-trunk updates), while
vision-only codecs receive 601 or 2,401 total updates. Shared and isolated variants are
not compute-matched; they test an operational cross-modal shared-training effect, not
gradient interference in the strict causal sense.
For each held-out video save codec fingerprints, vision latent-alignment MSE, vision
one-step prediction MSE, incumbent world MSE, and their ratio. A video supports migration
only when the unchanged MM-001 rule holds:

```text
vision_mse <= 1.5 * incumbent_world_mse
```

`shared_vat_600` must reproduce MM-001's fold/seed/video `vision_mse`,
`vision_latent_mse`, incumbent MSE, and primary-model fingerprint within
`rtol=1e-12, atol=1e-12`; mismatch invalidates MM-002.

Codec diagnosis is mechanical. With baseline support below 6/8, report:

- `nominal_codec_update_order_rescue` when shared ATV-600 crosses;
- `nominal_cross_modal_shared_training_penalty` when vision-only-600 crosses while
  both shared-600 schedules fail;
- `codec_update_budget_rescue` when shared and vision-only 2,400 both cross while
  nominal variants fail;
- `codec_sharing_by_budget_interaction` when only vision-only-2,400 crosses;
- `shared_positive_transfer_or_isolated_instability` when only shared-VAT-2,400 crosses;
- `isolated_codec_overtraining_or_instability` when vision-only-600 crosses but its
  2,400-cycle checkpoint fails;
- `terminal_audio_text_update_recency_sensitivity` when the pre-A/T snapshot crosses
  but the post-T shared checkpoint fails; and
- `codec_not_rescued_by_order_isolation_or_four_x_cycles` when no candidate crosses.

Multiple supported labels remain ambiguous and are never collapsed to one cause. A
failed pre-A/T snapshot does not rule out cumulative schedule effects.

## Integrity, execution, and verification

Preparation creates only the copied parent, byte-identical protocol copy, and input
manifest at the single canonical output path. The manifest binds the exact parent tree,
source tree, dependency versions, feature schemas, folds, seeds, variants,
hyperparameters, thresholds, row-construction rules, and expected artifact sets.

Formal execution revalidates the prepared state and MM-001, then atomically writes a
read-only `formal-start.json` before any new fit. After that marker, all variants run
without outcome-dependent stopping. Any execution, schema, parity, finiteness,
fingerprint, source, parent, packaging, or verifier mismatch yields
`invalid_MM002_package`; repair requires a new experiment ID.

The completed package contains the copied parent, protocol and marker, input manifest,
all fold/seed/video world rows, all raw-feature and integrity probe rows, all codec rows,
canonical summary and report, and a final recursive artifact manifest. Exact membership rejects extras, missing files, and
symlinks. Fast verification rechecks every hash/schema/parity record and recomputes all
video medians, support counts, factor labels, and report. Semantic verification reruns
every world and codec fit in memory and reproduces metrics and fingerprints within
`rtol=1e-12, atol=1e-12`.

MM-002 does not alter or make claims about T5, SNAC, simultaneous fusion, raw media
generation, action recovery, imitation, planning, control, the full Perception Test
distribution, or production capability. It changes no shipped Prospect task or gate.
The 6/8 rule is an engineering robustness threshold rather than strong statistical
evidence; two-second direct prediction is not a two-step rollout; and additional world
updates jointly change weights, Adam state, EMA targets, and observation statistics.
