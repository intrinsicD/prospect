# MM-001 frozen small real-multimodal preflight protocol

**Status:** frozen before formal execution  
**Date frozen:** 2026-07-15  
**Scope:** non-gated, small-system Perception Test preflight  
**Experiment ID:** `MM-001`

Pre-formal amendments (before any start marker or held-out outcome): the synthetic API
smoke showed the measured fastest path was 64-pixel CUDA TAESD/SNAC plus CPU T5, and
that T5 is a native span-denoising seq2seq model rather than an identity autoencoder.
The permitted `video_9253` wiring smoke then exposed a missing deterministic-cuBLAS
setting and SNAC's one-time FP16 batch-shape initialization. Independent code/protocol
review found an approximate 10% inequality, incomplete input authentication, a final-frame
clamp risk, and insufficient compact verifier evidence. The exact runtime, endpoint,
projection, evidence, and text-component clauses below incorporate those contract and
integrity findings. No scientific threshold or non-development-video outcome informed them.

## Question and scope

MM-001 asks the smallest question the current Prospect implementation can answer on
real aligned media:

> Can frozen image, audio, and annotation-derived text representations separately
> enter one vision-anchored Prospect latent and preserve useful one-second-ahead visual
> prediction on whole held-out videos?

The experiment tests **modality-by-modality substitution**, not simultaneous multimodal
fusion. The current `UniversalCodec` routes one fixed-width numeric modality at a time;
it has no learned fusion, masking, or missing-modality interface. The world model predicts
the next **visual latent**, not pixels, waveforms, text, rewards, or actions.

MM-001 is an engineering and mechanism preflight over eight sample videos. It is not an
independent confirmation, a benchmark result, a prevalence estimate, a claim about the
full Perception Test distribution, or evidence of production capability.

## Development smoke and formal units

- `video_9253` is the only development smoke input. Before formal start it may be used
  repeatedly to exercise download-independent API wiring, media decoding, tensor shapes,
  model loading, determinism checks, and artifact rendering.
- No `video_9253` smoke metric, reconstruction, feature, threshold choice, model state,
  fitted statistic, or branch decision enters formal evidence. All thresholds and branches
  in this document are frozen without consulting its smoke outcome.
- Formal execution re-encodes and reevaluates `video_9253` from clean state together with
  the other seven videos. Its formal metric is retained under the same rule as every other
  video. Because one input was used for instrumentation development, even the formal
  eight-video result is explicitly a small-system preflight, not untouched confirmation.
- Let `ids` be the exactly eight video IDs present in both the official sample media and
  annotation package, sorted lexicographically. Preparation must assert `len(ids) == 8`
  and persist the exact ordered list before formal start.
- Four folds are fixed mechanically. Fold `k in {0,1,2,3}` holds out
  `ids[2*k:2*k+2]` as two whole test videos and trains on the other six. No frame, audio
  window, text row, feature statistic, projection choice, or distillation target from a
  held-out video may enter that fold's training.
- Seeds `0`, `1`, and `2` are optimizer/model restarts inside each fold. A video is the
  experimental unit; timestamps and seeds are repeated measurements, not independent
  replicates.
- Hyperparameters are fixed below. There is no validation split, early stopping, fold
  selection, seed selection, or outcome-dependent retry.

For every endpoint, first aggregate timestamps within each test video, then take the
median over the three seed-specific values for that video. A formal gate requires its
complete per-video condition on at least `6/8` videos. Pooled timestamp counts, fold
averages, and seed-wise values are descriptive only and cannot override the `6/8` rule.

## Frozen external inputs and components

### Dataset

Use the official Google DeepMind Perception Test sample:

- `sample_videos.zip`: the eight MP4 files; their embedded audio is authoritative;
- `sample_annotations.zip`: the official annotations;
- the optional duplicate WAV archive is not used.

Preparation records source URLs, archive byte sizes, SHA-256 digests, extracted-file
digests, exact video/annotation membership, durations, stream metadata, and the decoder
tool/version. Perception Test materials are CC BY 4.0 and its repository software is
Apache 2.0: <https://github.com/google-deepmind/perception_test>.

### Frozen pretrained components

The execution path is frozen from the measured fastest smoke on this workstation:
TAESD and SNAC run on the RTX 3050 in CUDA float16; T5 runs on CPU in float32 because
its measured eight-token latency was no better on this GPU. All run in evaluation and
inference mode with gradients disabled. CUDA deterministic algorithms and deterministic
cuDNN are enabled, and each stochastic SNAC decode resets the recorded global torch seed.
SNAC consumes one all-zero encoder warm-up for each encountered batch size before data
inference, then requires two encodes to return exactly equal token IDs and two same-seed
decodes to agree within `rtol=1e-6, atol=1e-6`; warm-up outputs are discarded.
No component is fine-tuned. Model files, configurations, tokenizer files, library
versions, immutable revisions, local SHA-256 digests, device names, driver/runtime
versions, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are bound in the formal input manifest.
TAESD processes one video's window rows at a time, SNAC uses batches of 8, and T5 uses
batches of 16 with at most 96 non-special input tokens and 32 newly generated tokens.
Failure of this exact fast path is a
preparation failure; MM-001 does not silently fall back to a different device or dtype.

- **Image:** `madebyollin/taesd` at revision
  `614f76814bbe30edbe2e627ace1c2234c81a2c0e`. TAESD is a small MIT-licensed image
  autoencoder with `2,445,063` parameters whose encoder produces four-channel spatial latents at one-eighth input
  resolution. It expects RGB values scaled to `[-1, 1]` and uses latent scaling factor `1.0`.
  Primary source: <https://github.com/madebyollin/taesd>.
- **Audio:** `hubertsiuzdak/snac_24khz` at revision
  `d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec`. It is the MIT-licensed, mono, 24 kHz,
  `19,842,914`-parameter, three-level SNAC codec with a 4,096-entry codebook. Its official recommended use is
  speech; Perception Test also contains general sounds, so its adequacy here is earned
  only through the frozen component control below. Primary source:
  <https://github.com/hubertsiuzdak/snac>.
- **Text:** `google/t5-efficient-tiny` at revision
  `3441d7e8bf3f89841f366d39452b95200416e4a9`. It is an Apache-2.0, English,
  pretrained-only `15,570,688`-parameter encoder-decoder checkpoint with 256-dimensional hidden states and a
  span-corruption pretraining objective. Its frozen tokenizer IDs are pad `0`, EOS `1`,
  `<extra_id_0>` `32099`, and `<extra_id_1>` `32098`. Primary model card:
  <https://huggingface.co/google/t5-efficient-tiny>.
  The model vocabulary size is exactly `32128`; every retained non-padding token ID must
  satisfy `0 <= id < 32128`, while teacher-forcing padding alone uses `-100`.

Failure to obtain exactly these revisions is a preparation failure, not permission to
substitute a newer model.

## Frozen temporal examples

For a video of duration `D`, formal current times are

```text
t_k = 1.0 + 0.5*k seconds, for every integer k with t_k + 1.0 + 0.5 <= D.
```

Thus steps are 0.5 seconds and the prediction horizon is exactly 1.0 second. The final
half-second reserve ensures the 2-fps decoder has a real frame centered at the target
timestamp instead of clamping to its last frame. Every example is causal at its input time.
Authenticated annotation metadata therefore fixes counts before inference at
`{video_10993: 63, video_1580: 64, video_2564: 59, video_3501: 65,
video_6860: 65, video_8241: 48, video_874: 66, video_9253: 47}`, totaling `477`.

### Vision

- Decode once through ffmpeg's frozen two-frames-per-second filter and select the
  deterministic half-second-grid frame at `t` and `t + 1.0`; record the selected indices.
  A missing index is an integrity failure and is never clamped.
- Preserve aspect ratio, bicubic-resize into a `64 x 64` canvas, center-letterbox with
  black, convert to channel-first RGB in `[0, 1]`, then map to `[-1, 1]` before the
  TAESD encoder. Map decoder samples back with `(sample + 1) / 2` before pixel MSE.
- The raw frame representation is the flattened `4 x 8 x 8` encoder latent. TAESD's
  decoded reconstruction is retained only for the component checks.

### Audio

- Decode the MP4's embedded audio, mix to mono without loudness normalization, and
  deterministically resample to 24,000 Hz float32.
- The input at `t` is exactly the 24,000 samples in `[t - 1.0, t)`. Formal times start at
  one second, so no left padding is needed. A short decoded stream is an integrity failure
  and is never zero-padded.
- SNAC returns three variable-rate token streams. Token IDs and their level/position
  coordinates are retained losslessly; the decoded waveform is retained only for the
  component check.

### Text

At time `t`, select only temporal action and sound segments satisfying
`start <= t < end`. Clip-level QA answers, predictive questions, counterfactual answers,
future labels, and labels beginning after `t` are never inputs.

Normalize each active label by trimming and collapsing whitespace, preserve its text,
remove duplicates, and sort by Unicode code point. The canonical English input is

```text
action: <labels joined by " | ", or "none">; sound: <labels joined by " | ", or "none">.
```

The uncorrupted canonical string is tokenized by the frozen T5 tokenizer. Its fixed-width
raw representation is the attention-mask-weighted mean of the final encoder hidden states.

## Frozen 32-dimensional projections

Every modality reaches Prospect as a 32-dimensional numeric vector. Projections are
fixed, untrained, and identical across folds and seeds; no PCA, vocabulary fitting, or
formal-data-dependent feature selection is allowed.

- VISION seed: `12001`; flatten the TAESD latent and multiply by a dense Rademacher
  matrix with entries `{-1,+1}/sqrt(256)`.
- TEXT seed: `12003`; multiply the 256-dimensional T5 mean-pooled state by a dense
  Rademacher matrix with entries `{-1,+1}/sqrt(256)`.
- AUDIO seed: `12002`; concatenate the fixed one-second SNAC streams (12, 24, and 48
  token IDs), map IDs from `[0,4095]` to `[-1,1]`, and multiply the resulting
  84-dimensional vector by a dense Rademacher matrix with entries
  `{-1,+1}/sqrt(84)`.

Dense matrices are generated as `2 * default_rng(seed).integers(0, 2, dtype=int8) - 1`
in row-major order and stored with their SHA-256 digests. Projected values are float64
in the experiment harness. There is no per-row normalization. Training-only
standardization remains owned by
`FlatWorldModel` and `UniversalCodec`.

## Frozen component checks

Component checks establish that each frozen frontend carries sample identity before its
features are interpreted through Prospect. They are relative checks, not quality or
state-of-the-art claims.

### TAESD frame/image check

For every sampled formal frame, compare TAESD round-trip pixel MSE with the MSE of the
frame's per-channel spatial-mean image. A video supports the frame/image component when
its median TAESD MSE is strictly lower than its median mean-image MSE.

### TAESD framewise-video check

Within each video, circularly shift its decoded TAESD reconstructions by
`floor(n_frames / 2)` while leaving source frames fixed. A video supports the framewise
path when matched source/reconstruction MSE is strictly lower than shifted-pair MSE.
This checks frame identity and order only; TAESD is applied framewise and no temporal
coherence or video-generation capability is claimed.

VISION is component-eligible only when both TAESD checks support at least `6/8` videos.

### SNAC-24k audio check

Compare every source window with its own SNAC reconstruction and with a reconstruction
from the next sorted video at nearest normalized progress, wrapping across all eight
videos. A video supports SNAC when its median matched waveform MSE is strictly lower
than its median different-video MSE. A half-cycle roll of each level's token positions is
retained as a descriptive within-window control. AUDIO is component-eligible at `6/8`
videos.

### T5-efficient-tiny masked-span decoder check

For every canonical text with `n` non-special tokens, mask one contiguous span of length
`max(1, ceil(0.15*n))`. Select its start by interpreting the first eight bytes of
SHA-256 of `"MM-001|video_id|timestamp"` as an unsigned little-endian integer modulo
the number of valid starts. `timestamp` is formatted with exactly one decimal place
(`1.0`, `1.5`, ...). Replace the span with `<extra_id_0>` and use the native T5
target `<extra_id_0> masked_tokens <extra_id_1>`. The negative target is drawn from the
next sorted video at nearest normalized progress and uses that example's masked tokens.

Compute mean teacher-forced token NLL for the correct and negative targets and also save
greedy decoding with a fixed maximum of 32 new tokens. A video supports the text decoder
when its median correct-target NLL is strictly lower than its median negative-target NLL
and every greedy result is finite, terminates within the limit, and preserves a parseable
sentinel sequence. Exact span match is descriptive. TEXT is
component-eligible at `6/8` videos.

A generated sequence with zero non-pad tokens is a valid bounded but unparseable decoder
outcome. It fails that video's text component condition; it is not a package-integrity
error. Lengths outside `0..32`, non-finite inference, or token/schema corruption remain
integrity errors.

Repeated component inference must reproduce token IDs exactly and floating outputs within
`rtol=1e-6, atol=1e-6` on the frozen mixed CUDA/CPU environment. A schema, finiteness, ordering, or
determinism failure is an integrity failure; a valid relative component failure is a
scientific result and merely makes that modality ineligible.

## Vision-anchored Prospect model

The incumbent dynamics stream is VISION only. For each fold and seed instantiate

```python
FlatWorldModel(
    obs_dim=32,
    action_dim=1,
    latent_dim=8,
    hidden=64,
    ensemble=5,
    lr=3e-3,
    ema_tau=0.995,
    seed=seed,
    w_reward=0.0,
    w_inverse=0.0,
    w_var=25.0,
    w_cov=1.0,
)
```

Training transitions contain raw projected vectors in `LatentState.z`:

```text
state      = v_t
action     = [0.0]
next_state = v_(t+1.0)
reward     = 0.0
```

Run exactly 1,500 updates, sampling batch size 64 with replacement from the six training
videos using `default_rng(seed + 1)`. The null action makes this an autonomous predictive
perception model. Disabling reward and inverse losses prevents zero reward/action from
being misrepresented as learned controllability.

At test time:

```text
z_t      = model.encode(v_t)
target   = model.encode_target(v_(t+1.0))
forecast = model.predict(z_t, Action([0.0])).mean
```

The primary error is mean squared error in this fold/seed model's own EMA target-latent
space. Raw MSE magnitudes are never compared across independently trained models without
normalization.

## Frozen visual baselines and temporal negative control

- **Persistence:** copy `model.encode_target(v_t).z` as the next-latent prediction.
- **Linear ridge:** fit, on training videos only, a ridge map from `[v_t, 1]` to the
  primary model's `model.encode_target(v_(t+1.0)).z`, with penalty `1e-3` and no penalty
  on the intercept. Evaluate in the same target latent as the primary model.
- **Temporal shuffle model:** train an equal-compute `FlatWorldModel` with the same seed
  and hyperparameters after circularly shifting successors within each training video by
  `floor(n_pairs / 2)`. Evaluate it on ordered held-out transitions in its own target
  latent. Compute both its world-model and persistence MSE.

A video supports real visual temporal prediction only when all three conditions hold
after the seed median:

```text
world_mse * 1.2 <= persistence_mse
world_mse < ridge_mse
(world_mse / persistence_mse) * 1.1
    <= shuffle_model_mse / shuffle_model_persistence_mse
```

The normalized third comparison is load-bearing: raw MSEs from independently learned
latent scales are not commensurate. The visual temporal gate passes at `6/8` videos.

## Frozen codec distillation and substitution tests

For each already-trained primary world model instantiate

```python
UniversalCodec(
    {
        Modality.VISION: 32,
        Modality.AUDIO: 32,
        Modality.TEXT: 32,
    },
    latent_dim=8,
    token_dim=32,
    hidden=64,
    lr=3e-3,
    seed=seed + 1,
)
```

The frozen distillation target at every training time is
`z_inc_t = model.encode(v_t).z`. Make one full-training-array call per modality first,
in VISION, AUDIO, TEXT order, to freeze that modality's standardization from the complete
training fold. Then run exactly 600 round-robin cycles; each cycle performs one batch-128
`distill_encode` update for each modality, sampled with replacement using
`default_rng(seed + 303)`. The world model remains frozen throughout.

On the aligned codec only, the same full-array initialization and each round-robin
cycle also perform one `fit_decode` update per modality from the frozen incumbent
latent to that modality's 32-dimensional feature. Decoder heads do not update the
shared trunk. Held-out feature reconstruction MSE is descriptive evidence that the
public `decode` seam is exercised; it is not raw pixel, waveform, or text generation.
Because `UniversalCodec.decode` currently returns standardized feature units, the
harness records the explicit training-statistic inverse transform as a known API gap.

For AUDIO and TEXT, train an equal-compute negative codec from the same initialization.
Its feature rows remain unchanged, but targets come from the next sorted training video
at nearest normalized progress. For source rank `r` among `n` ordered rows, choose rank
`round((r / max(n - 1, 1)) * (m - 1))` in the next video's `m` ordered rows using Python's
ties-to-even `round`. This different-video derangement preserves real modality
values while breaking cross-modal correspondence. A training-mean incumbent latent is the
content-free downstream control.

For each modality `m`, held-out downstream error is

```text
z_m       = codec.encode(Observation(m, feature_m_t))
forecast  = model.predict(z_m, Action([0.0])).mean
target    = model.encode_target(v_(t+1.0)).z
sub_mse   = mse(forecast, target)
latent_mse = mse(z_m, model.encode(v_t).z)
```

### VISION migration gate

A video supports the VISION codec migration when

```text
vision_sub_mse <= 1.5 * incumbent_world_mse
```

The `1.5x` tolerance is inherited from P6/P12. The gate passes at `6/8` videos.

### AUDIO substitution gate

A video supports AUDIO only when its aligned codec satisfies all conditions:

```text
aligned_audio_latent_mse * 1.1 <= deranged_audio_latent_mse
aligned_audio_sub_mse * 1.1 <= deranged_audio_sub_mse
aligned_audio_sub_mse * 1.1 <= mean_latent_sub_mse
```

The gate passes at `6/8` videos. Ratio to the VISION incumbent is reported but not gated:
audio need not contain every visual detail.

### TEXT substitution gate

TEXT uses the identical three conditions, replacing AUDIO with TEXT, and passes at
`6/8` videos. It tests predictive information in gold, current-time annotation text;
it is not a raw-language-understanding or deployable-text-sensor test.

No latent averaging, concatenation, attention across modalities, modality dropout, or
test-time selection of the best modality is permitted. Such operations would add an
unfrozen fusion mechanism.

## Frozen decision branches

Apply branches in this order after all folds and seeds complete:

1. Any input, revision, hash, fold, timestamp, causal-alignment, projection,
   determinism, finiteness, raw/derived recomputation, or artifact mismatch raises an
   `InvalidMM001Package` exception with classification `invalid_MM001_package` and yields
   no scientific conclusion.
2. If VISION is not component-eligible, emit `vision_component_not_supported`.
3. If the visual temporal gate fails, emit `real_visual_temporal_prediction_not_supported`.
4. If the VISION migration gate fails, emit `vision_codec_migration_not_supported`.
5. Otherwise evaluate component eligibility plus substitution for AUDIO and TEXT:
   - both pass: `three_seam_predictive_substitution_supported`;
   - AUDIO only: `vision_audio_predictive_substitution_only`;
   - TEXT only: `vision_text_predictive_substitution_only`;
   - neither: `real_visual_prediction_only`.

A frontend component failure is not an invalid package. It makes that seam ineligible and
routes to the appropriate valid partial/null branch. Thresholds, support counts, masks,
folds, or branches are never changed after outcomes.

## Existing-system scope controls

Existing P9, P13, and P14 evaluations remain separate controls:

- P9 continues to test the existing integration, generalization, ablation, invariant, and
  uncertainty behavior on its authored fixtures.
- P13 remains the action-free latent-action test on Pendulum. MM-001 does not run
  `LatentActionModel`, decode Perception Test labels as true controls, or claim hidden
  motor-action recovery.
- P14 remains imitation in an executable same-dynamics environment with the agent's own
  labelled grounding transitions. MM-001 has neither executable controls nor a rollout
  environment and therefore cannot test imitation or policy return.

Run focused MM-001 tests plus the existing P9, P13, and P14 gates, `make test`, `make lint`,
`make typecheck`, and `make gate-all` as regression controls. Their pass/fail status never
enters an MM-001 scientific branch. A regression failure blocks shipping the implementation
package but is not evidence for or against multimodal substitution.

## Claim boundary

Only the strongest branch permits this statement:

> On the official eight-video Perception Test sample under four whole-video folds, a
> frozen-feature, vision-anchored Prospect model predicted held-out next-visual latents;
> separately supplied aligned SNAC audio and annotation-derived T5 text carried predictive
> information through `UniversalCodec` beyond different-video and content-free controls.

Even that branch does **not** support claims of simultaneous multimodal fusion, learned
cross-attention, missing-modality robustness, raw pixel/audio generation, free-form text
generation from Prospect latents, semantic language understanding, calibrated cross-modal
OOD surprise, true action recovery, causal intervention, imitation, planning, control,
full-dataset generalization, or state-of-the-art performance. TAESD/SNAC/T5 component
checks establish only that their frozen representations are usable on this sample.

Partial branches permit only the capabilities explicitly named by their branch. A valid
null result is retained and reported without replacing a component, changing the horizon,
adding data, tuning a threshold, or selecting favorable videos.

## Integrity, stopping, and reproduction package

Before formal computation, write and fsync an atomic formal-start marker binding this
protocol, source tree, exact dataset files, model revisions/files, dependency lock,
decoder tool, eight IDs and folds, projections, hyperparameters, schemas, and expected
artifact list. After that marker:

- run all four folds and all three seeds without outcome-dependent stopping;
- never retry or omit a model/video because of its metric;
- preserve any valid partial, failure, or null branch;
- treat an execution, packaging, verifier, or provenance defect as terminal for `MM-001`;
  any repair requires a new experiment identifier.

The reproduction package must contain a byte-identical copy of the frozen protocol;
formal-start record; input, source, dependency, model, dataset, and exact timestamp
manifests; all three projection matrices and their SHA-256 digests; compact raw frontend
outputs (current/target TAESD latents, SNAC code IDs, T5 mean-pooled encoder states,
and T5 masked targets/generated IDs); the exact temporal-shuffle and cross-video training
index arrays for every fold, with dtype, shape, and SHA-256 bindings; window-level
component metrics; per-video and per-seed metrics; result JSON; concise report; and artifact
manifest. To keep this smallest preflight compact, decoded pixel and waveform tensors are
not duplicated in the package: their exact window metrics are stored and the tensors remain
regenerable from authenticated media and model snapshots. The fast verifier must reconstruct
every fold, control pairing, per-video aggregate, support count, projection, and branch from
stored evidence. The semantic verifier must regenerate features, retrain all primary/shuffle
models and codecs, and reproduce metrics within the frozen mixed-runtime tolerances.

MM-001 does not modify production tasks, ADRs, shipped gates, or prior result packages.
