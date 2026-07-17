# MM-007 physically matched resolution protocol

**Date:** 2026-07-15  
**Status:** frozen before MM-007 real-video multiresolution outcomes  
**Parent:** sealed MM-006 (`tested_pixel_warp_ceiling_failure_supported`)  
**Scope:** outcome-informed, target-aware single-step mechanism diagnostic on the same eight videos

## 1. Question

MM-006 found a sharp full-fit/held-out inversion. Its 8x8 pixel quadrant oracle improved
over persistence in-sample on 7/8 videos but met the frozen 20% effect gate on only 1/8;
the checkerboard-cross-fitted oracle improved on 1/8 and supported 0/8. Each 3x3
quadrant selected among 25 shifts from only four or five fitting cells, with three or
four cells left after trimming. The sealed result therefore establishes failure of that
tested 8x8 assay, not failure of pixel transport in general.

MM-007 changes exactly one mechanism: spatial sampling/support. It asks whether the
same physical translation family, candidate set, field of view, fitting units, controls,
normalization rule, and video identities recover when evaluated at 16x16, 32x32, or the
decoded 64x64 source resolution.

No causal flow, deformation, photometric residual, learned model, or visibility mask is
added. Those are later experiments only if the physically matched resolution ladder
fails cleanly.

## 2. Claim boundary

The panel is the same outcome-informed eight-video Perception Test sample and the same
453 matched identities already exposed in MM-005/MM-006. Videos are support units;
pixels, macrocells, and timestamps are repeated measurements. The target-aware oracle
is deliberately leaking and diagnostic only.

MM-007 evaluates one teacher-forced 0.5-second RGB step on the central physical 48x48
region. A pass can support a high-resolution correspondence frontend as the next causal
experiment. It cannot establish causal prediction, rollout, decoded-video quality,
independent-video generalization, population prevalence, or end-to-end Prospect
capability. A failure applies only to the frozen four-quadrant translation family and
physical +/-8-pixel range; it does not rule out dense flow, affine/deformable motion,
appearance change, disocclusion, or content generation.

MM-001 through MM-006 remain sealed and are never reclassified.

## 3. Immutable receipts and preparation-only extraction

Preparation copies exactly six pinned parent/provenance receipts:

1. `inputs/MM-006/artifact-manifest.json`
2. `inputs/MM-006/input-manifest.json`
3. `inputs/MM-006/MM-006-evidence.json`
4. `inputs/MM-006/MM-006-results.json`
5. `inputs/MM-006/inputs/MM-005/inputs/MM-004/MM-004-pixel-grids.npz`
6. `inputs/MM-004/input-manifest.json`

The live MM-006 and MM-004 packages must fast-verify before preparation, immediately
before formal execution, and during verification. Every copied byte, mode, manifest
relationship, parent classification, and hard-pinned SHA-256 must match.

The copied MM-004 manifest supplies the authenticated cache path, eight MP4 hashes and
sizes, decoded 2-fps frame counts, and exact FFmpeg executable identity. During
preparation only, MM-007 rehashes those files and executable, decodes the same
letterboxed 64x64 RGB frames, selects the 477 current-frame identities from the copied
MM-004 grid, and writes one deterministic input:

```text
MM-007-frames-64x64.npz
  video_ids   <U11   [477]
  timestamps  <f8    [477]
  frames_uint8 u1    [477,64,64,3]
```

The decoder's float32 values must round-trip through uint8 bit-exactly. Converting the
stored frames back to float32 in `[0,1]`, area-pooling 64->8 with float64 block means,
casting to float32, and transposing to NCHW must reproduce all 477 copied MM-004
`pixel_current` arrays bit-for-bit. The copied `pixel_target` must likewise equal the
appropriately indexed decoded frames pooled to 8x8. Preparation reconstructs the exact
453 matched identities and requires identity SHA-256
`d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b`.

The authenticated preparation bytes are additionally frozen before formal outcomes.
The canonical array SHA-256 values are:

```text
video_ids    06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50
timestamps   128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55
frames_uint8 46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9
```

With sorted keys and the sealed NumPy dependency, the canonical compressed NPZ has
SHA-256 `fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f`.
Preparation, formal pre-marker validation, and every verifier must check these hard
pins independently of the mutable input-manifest record. Thus an R8-preserving
within-block rearrangement cannot become a formal MM-007 input by merely recomputing
the manifest.

Preparation may validate identities, hashes, pooling, and current-only normalizer
inputs. It may not call a displacement search, real scorer, synthetic generator, or
MM-007 decision function.

## 4. Exact panel and pooling intervention

Within every video, reuse MM-005's positions and counts:

```text
previous = frame at t-0.5 s
current  = frame at t
target   = frame at t+0.5 s
counts   = 60/61/56/62/62/45/63/44
```

For each `R in {8,16,32,64}`, derive RGB grids directly from the same stored 64x64
frames. `R=64` is the float32 RGB identity; lower resolutions use non-overlapping
float64 block means followed by float32 storage. No resolution is derived from another
pooled resolution.

All arms score the same central native-pixel region `[8:56,8:56]`:

| Resolution | Central grid slice | Scored cells | Grid cells per 8px macrocell |
|---:|---:|---:|---:|
| 8 | `[1:7,1:7]` | 6x6 | 1x1 |
| 16 | `[2:14,2:14]` | 12x12 | 2x2 |
| 32 | `[4:28,4:28]` | 24x24 | 4x4 |
| 64 | `[8:56,8:56]` | 48x48 | 8x8 |

For real-video rows, one train-current-only RGB channel mean and scale is fit per
whole-video fold from the R8 grids, exactly preserving MM-006 normalization, with scale
floor `1e-6`. That same fold normalizer is reused unchanged at R8/R16/R32/R64 and by
every real arm and wrong-target control. For synthetic controls, each scenario
independently fits one current-only R8 normalizer and reuses it unchanged across all
resolutions and families; no real-video statistics enter synthetic generation or
scoring. Targets never enter normalization. This prevents resolution-dependent channel
reweighting from becoming a second intervention.

## 5. Frozen physical matcher

Displacement moves content down/right:

```text
warp(X,d)[y,x] = X[y-dy,x-dx]
```

The native-pixel candidate set is identical at every resolution:

```text
dy_native, dx_native in {-8,-4,0,4,8}
```

Grid displacements are native displacement times `R/64`, yielding
`{-1,-0.5,0,0.5,1}` at R8, `{-2,-1,0,1,2}` at R16,
`{-4,-2,0,2,4}` at R32, and `{-8,-4,0,4,8}` at R64. Candidates are ordered by
squared native magnitude, then `dy`, then `dx`; exact ties select the first. Sampling
is deterministic float64 bilinear interpolation without padding, wrapping, reflection,
or clamping. The central crop guarantees valid support.

The central region is partitioned into a physical 6x6 lattice of 8x8-native-pixel
macrocells and four 3x3-macrocell quadrants. Candidate loss is first averaged over RGB
and all grid pixels within each macrocell. The largest `floor(n/4)` macrocell losses are
dropped, then the retained macrocell losses are averaged. Reported prediction metrics
are ordinary untrimmed SSE/MSE over every scored RGB grid cell.

Every quadrant flow is regularized toward a global translation fitted from the same
allowed target macrocells:

```text
0.05 * (identity_loss + 1e-12)
     * ||d_local_native - d_global_native||^2 / 8^2
```

This is algebraically identical to MM-006 at R8 and does not strengthen with
resolution.

## 6. Physical macrocell cross-fitting and arms

Macrocell parity is `(macro_y + macro_x) % 2`. For output parity `p`, global and all
four local flows are selected using only opposite-parity target macrocells; predictions
and scores are then produced only on parity-`p` macrocells. The two held-out halves are
combined. Mutating any held-out target macrocell must leave fitted flows and predictions
bit-exact; only its final score may change. A global anchor fitted with all target cells
may never regularize a cross-fitted local flow.

The following arms are saved at every resolution:

- `persistence`: current grid, no fit;
- `oracle_full`: target-aware quadrant flow fitted and scored on all macrocells;
- `oracle_xfit` (primary): physical-macrocell-cross-fitted quadrant flow;
- `global_xfit`: the cross-fitted global anchor applied on held-out macrocells;
- `near_target_xfit`: fit against an adjacent wrong target, score against the true target;
- `far_target_xfit`: fit against MM-006's half-cycle wrong target, score against the true target.

The near mapping is fixed-point-free within video: swap consecutive target rows; for an
odd count, use a three-cycle for the final three. Thus every wrong target is 0.5 or 1.0
seconds from the true target without a wraparound jump. The far mapping is MM-006's
within-video half-cycle roll. Both mappings are frozen before outcomes.

## 7. Parent parity and controls

R8 must replay the sealed MM-006 pixel `quadrant_flow` primitives for persistence,
full oracle, cross-fitted oracle, far-target oracle, support predicates, and boundary
fractions at every video. A discrepancy is an invalid experiment, not a new result.

Synthetic controls are generated without real-video access and must establish:

1. an exact operator-realizable translation passes full and cross-fitted matching at
   every resolution with the frozen signed displacement convention;
2. a stationary panel has zero persistence error and cannot create support;
3. an independent appearance target cannot create cross-fitted support;
4. an 8-pixel-periodic alias panel is unresolved after 8x8 pooling but recovers at
   R16/R32/R64, validating information erasure and signed-flow recovery at the four
   resolutions. Because its R8 persistence error is exactly zero, it does not validate
   the real-data ratio, fold, or onset ladder; those branches are covered separately by
   focused decision-rule tests;
5. held-out-target mutation cannot change a cross-fitted fit or prediction.

Every synthetic scenario, seed, resolution, expected predicate, and expected endpoint
is recorded. Any positive- or negative-control failure invalidates MM-007.

## 8. Frozen gates and decision ladder

Let `p_R`, `o_R`, `f_R`, `g_R`, `n_R`, and `s_R` denote persistence, primary xfit,
full-fit, global-xfit, near-wrong-target, and far-wrong-target MSE for one video.

Primary per-video support requires all of:

```text
p_R > 0
1.25 * o_R <= p_R
1.10 * o_R <= n_R
1.10 * o_R <= s_R
```

Full-oracle per-video support is `p_R > 0 and 1.25*f_R <= p_R`. A resolution family
passes only with primary support on at least 6/8 videos, `o_R < p_R` on at least 7/8,
and at least one supporting video in every whole-video fold. The full oracle must also
pass its corresponding 6/8, 7/8, and fold-coverage gates. A wrong-target arm is a strong
null hit when it independently clears the 20% persistence effect.

For a higher resolution to count as a resolution response, in addition to passing it
must improve the per-video ratio `rho_R=o_R/p_R` over R8 on at least 6/8 videos with
every fold represented:

```text
1.10 * rho_R <= rho_8
```

and it must have any ratio improvement on at least 7/8. This prevents calling a small
threshold crossing a mechanistic recovery.

Boundary occupancy is the fraction of selected native flow components with absolute
value 8. At least three videos with occupancy >=0.25 make the physical range
inconclusive. At least six strong near/far null hits invalidate the real assay; three to
five are inconclusive. A global pass with quadrant failure, an R8 parent-parity failure,
or a full/xfit contradiction affecting at least three videos is inconclusive.

After valid synthetic, parent, null, boundary, and consistency controls, select exactly
one branch:

1. `resolution_recovery_at_16_supported`: R16, R32, and R64 all pass and meet the
   R8-response gate.
2. `resolution_recovery_at_32_supported`: R16 fails cleanly while R32 and R64 both pass
   and meet the R8-response gate.
3. `MM007_resolution_response_inconclusive`: R64 alone passes (no finer replication), a pass
   is followed by a finer-resolution failure, or a prespecified borderline/control
   condition occurs.
4. `physically_matched_resolution_failure_supported`: R16/R32/R64 all fail cleanly,
   higher-resolution primary and full support counts remain below three, and controls
   are valid. This unlocks a separately frozen deformation experiment.
Any other valid but non-claimable pattern also emits
`MM007_resolution_response_inconclusive`; there is no second generic inconclusive token.

No best-of-three resolution is selected post hoc. A supported recovery nominates the
earliest replicated resolution as a correspondence frontend and requires a new causal
source-to-current extrapolation experiment before adoption. A clean failure nominates
deformation, then photometric residual, then visibility as separate interventions.

## 9. Sealed lifecycle

The completed package contains exactly 14 files: nine prepared files (protocol copy,
input manifest, frame NPZ, and six copied receipts), four outcomes (formal marker,
evidence, result, report), and the artifact manifest written last.

`prepare` uses exclusive creation and finishes before any scientific call. `run`
validates exact prepared membership, parents, source/config/protocol hashes, NPZ schema,
477/453 identities, pooling parity, and environment; then it creates and fsyncs a
read-only `0444` `formal-start.json` immediately before the first scientific call.
After that marker it reads only copied inputs, runs exactly once, writes outcomes with
exclusive creation, and writes the artifact manifest last. Interruption after the
marker is terminal and cannot be resumed under MM-007.

Fast verification rejects extra/missing files, symlinks, mode drift, path overlap,
receipt/source/dependency/config drift, NPZ key/dtype/shape/order/byte drift, identity
or pooling drift, evidence membership drift, non-finite values, predicate/denominator
errors, summary/report mismatch, and artifact-manifest mismatch. It regenerates the
complete scientific result from the copied frame NPZ. Semantic verification additionally
rehashes and re-decodes authenticated media in memory, requires bit-exact equality with
the copied uint8 frames, and then repeats full regeneration.

The formal `run` may be invoked exactly once. Verification never rewrites outcomes.

## 10. Interpretation

This is an outcome-informed mechanism split, not a benchmark score. A replicated
high-resolution recovery satisfying the frozen comparisons to R8 supports the hypothesis that 8x8 pooling/support
caused the MM-006 held-out correspondence failure. A clean non-recovery rules against
that specific remedy and justifies testing richer geometry next. Neither branch by
itself makes the whole model work; it identifies the smallest next mechanism that can
be tested causally and then integrated behind an independent-panel rollout gate.
