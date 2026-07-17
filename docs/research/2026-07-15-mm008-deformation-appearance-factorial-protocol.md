# MM-008 deformation-versus-appearance factorial protocol

**Date:** 2026-07-15  
**Status:** frozen before any MM-008 real-video outcome  
**Parent:** sealed MM-007 (`physically_matched_resolution_failure_supported`)  
**Scope:** outcome-informed, target-aware single-step mechanism diagnostic on the same eight videos

## 1. Question

MM-007 held the physical crop, fitting units, candidate translations, normalization,
and video identities fixed while increasing RGB resolution from 8x8 through 64x64.
At R16, R32, and R64, the primary four-quadrant cross-fitted translation oracle
improved over persistence on 0/8 videos and met the frozen support gate on 0/8. R32
and R64 reduced the R8 cross-fit penalty on 6/8 videos, but every higher-resolution
primary pooled ratio remained above persistence. Resolution therefore attenuated the
damage without recovering the frozen translation family.

MM-008 changes the transformation family at R64 and asks the smallest next causal
mechanism question: is the missing target-aware ceiling explained by a smooth spatial
family, global channel-wise appearance change, or only their joint use?

The 2x2 diagnostic cells are:

1. `00`: identity/persistence (no geometry and no appearance fit);
2. `10`: a smooth six-parameter affine-flow family;
3. `01`: per-channel gain-and-bias appearance without spatial motion; and
4. `11`: the affine-flow family jointly scored with gain and bias.

The exact R64 MM-007 global-translation and quadrant-translation families are external
parent sentinels, not factorial cells. MM-008 does not estimate a statistical
interaction contrast; a combined-only result supports joint recovery, not an
interaction claim.

No visibility mask, dense flow, learned correspondence, residual generator, causal
source-to-current extrapolator, or rollout is introduced. Those remain later
interventions if all three newly fitted families fail cleanly.

## 2. Claim boundary

The panel is the same outcome-informed eight-video Perception Test sample and the same
453 half-second transitions exposed in MM-005 through MM-007. Videos, not pixels or
timestamps, are the eight support units. Every fitted arm uses the true future target
and is an intentionally leaking diagnostic ceiling.

A supported arm can identify the next mechanism to implement causally. It cannot
establish target-isolated correspondence, independent-video generalization,
population prevalence, causal prediction, rollout quality, or end-to-end Prospect
capability. A clean failure applies only to the frozen low-dimensional affine and
global per-channel appearance families. It does not rule out dense or piecewise flow,
occlusion, disocclusion, local illumination, nonlinear appearance, or new-content
generation.

MM-001 through MM-007 remain sealed and are never reclassified.

## 3. Inputs, panel, and immutable parent replay

MM-008 consumes the sealed MM-007 package and copies its authenticated protocol,
input manifest, artifact manifest, evidence, results, formal marker, and
`MM-007-frames-64x64.npz`. Every copied byte, mode, artifact record, package hash,
frame-array pin, source/config/dependency binding, and parent classification must match
the live fast-verified MM-007 package before the MM-008 formal marker is written.

Preparation copies exactly these eight parent files under `inputs/MM-007/`:

```text
artifact-manifest.json     db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb  0644
input-manifest.json        1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f  0644
formal-start.json          ea5c7bda870d71ead3172c1fc6e504d6a6b02d2ba785e9fd2fc75a91c667eee3  0444
MM-007-evidence.json       13dfa89e541e6122263ea9814d42fb328da303dcc74556cdaaa5d5860d99abaf  0644
MM-007-results.json        3c92729e1e5c18c14461e36602bdb86acd31750d9f5a85f535cd33a43fb9c47b  0644
MM-007-report.md           b18760128941ab2eff893b8c0afc469b92f71077d489e060d56519407990b8a2  0644
MM-007-protocol.md         24bbac1855cc2b51d2a65012b9c63037637c53555b86bbad7c66a6249108a73c  0644
MM-007-frames-64x64.npz    fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f  0644
```

The live parent must pass verification during preparation, immediately before marker
creation, and in both MM-008 verifiers. Copied receipts and live bytes must still agree
at each boundary.

The exact hard-pinned frame identities remain:

```text
video_ids    06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50
timestamps   128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55
frames_uint8 46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9
frame NPZ    fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f
```

Within each video, reuse MM-007's 453 current/target identities, counts, four
whole-video folds, near-target derangement, and far half-cycle derangement. Score only
the central native `[8:56,8:56]` region of the R64 RGB frame. Refit the exact MM-007
fold normalizer from training-video R8 current frames only and apply the same channel
mean and scale to every R64 arm. Targets never enter normalization.

Before interpreting a new arm, both R64 translation sentinels must replay every sealed
MM-007 global-translation and quadrant-flow primitive and predicate exactly. A mismatch
invalidates MM-008. The affine family's incremental comparator is the nested global
translation sentinel; the quadrant sentinel remains the parent's primary external
reference.

## 4. Physical macrocells, loss, and cross-fitting

The central 48x48 region is partitioned into the same physical 6x6 lattice of 8x8
macrocells. Macrocell parity is `(macro_y + macro_x) % 2`. Parameter fitting uses
ordinary RGB pixel residuals but aggregates them in two stages: mean squared error
within each macrocell, removal of the largest `floor(n/4)` macrocell losses, then the
mean over retained macrocells. Reported prediction SSE/MSE remains untrimmed over all
central RGB pixels.

Each target-aware family has a full-fit diagnostic and a checkerboard-cross-fitted
primary arm. For output parity `p`, every geometry and appearance parameter is fit
using only target macrocells of parity `1-p`; the fitted transform predicts only
parity-`p`. The two held-out predictions are combined before scoring. Mutating a
held-out target macrocell must leave all fitted parameters and predictions bit-exact;
only its final score may change.

Parameters are independent per transition. There is one six-parameter/appearance
vector for a full fit and two independent vectors for its parity-specific cross-fit;
nothing is shared across timestamps, videos, true/wrong targets, or folds. Only the
fold's current-only channel normalizer is shared as already frozen.

Full-fit success without cross-fit success is target overfitting, not mechanism
support.

## 5. Frozen affine-flow family

Let target coordinate `(y,x)` lie in `[8,55]^2` and define centered native
coordinates

```text
u = (y - 31.5) / 23.5
v = (x - 31.5) / 23.5
```

The backward native flow is

```text
dy(u,v) = ty + ayy*u + ayx*v
dx(u,v) = tx + axy*u + axx*v
warp(X,d)[y,x] = X[y-dy(u,v), x-dx(u,v)]
```

Sampling is deterministic float64 bilinear interpolation without padding, wrapping,
reflection, or clipping. Any candidate for which any central-site flow component has
absolute value greater than 8 is inadmissible, preserving the authenticated border.

Fit starts with all four gradient coefficients at zero and searches the exact MM-007
translation candidates `ty,tx in {-8,-4,0,4,8}`, ordered by squared magnitude and then
`ty,tx`. It then performs exactly two coordinate-descent sweeps in order
`ty, tx, ayy, ayx, axy, axx`. Translation coordinates use the absolute ordered
sequence

```text
0, -4, 4, -8, 8
```

and gradient coordinates use

```text
0, -2, 2, -4, 4
```

while all other coefficients are held fixed. The current value is therefore always a
candidate. Revisiting translation once at the start of each sweep prevents the initial
global search from limiting the affine result. Exact objective ties select the first
listed candidate. There is no hidden
optimizer, stochastic initialization, early stopping, or outcome-dependent range
change.

The affine-only candidate objective is the frozen macro-trimmed fit loss. This family
nests global translation but not MM-007's discontinuous quadrant flow.

## 6. Frozen appearance family

For each RGB channel independently, appearance maps a sampled normalized source value
`z` to

```text
z_hat = gain_c * z + bias_c
```

The deterministic fitter has two passes:

1. fit ordinary least squares on every allowed fitting pixel;
2. compute the resulting RGB-averaged loss per allowed macrocell, retain the lowest
   `n-floor(n/4)` macrocells with stable macrocell-index tie-breaking, and refit on all
   pixels of those retained macrocells.

The slope denominator has floor `1e-6`. The final parameters are clipped only as a
numerical guard to `gain in [-2,4]` and `bias in [-4,4]` normalized units. No target
statistics cross the checkerboard split.

For one channel on one retained pixel set, the exact solve is

```text
mx   = mean(x)
my   = mean(y)
var      = mean((x-mx)^2)
cov      = mean((x-mx)*(y-my))
gain_raw = cov / max(var, 1e-6)
bias_raw = my - gain_raw*mx
gain     = clip(gain_raw, -2, 4)
bias     = clip(bias_raw, -4, 4)
```

Both passes use float64 means. The first-pass macrocell residual is averaged jointly
over RGB, so all three channels share one retained macrocell set; channel-specific
trimming is forbidden. Stable `(loss, macrocell_id)` ordering retains exactly 14/18
macrocells for a checkerboard fit and 27/36 for a full fit. After the second solve, the
candidate objective is the mean final RGB loss on that same retained set. It may not
select or trim a third macrocell set.

The appearance-only arm fixes all flow parameters to zero. For the combined arm,
the complete two-pass appearance solve is rerun on the allowed fitting cells for every
affine candidate before that candidate's retained-set objective is
evaluated. After the final affine
parameters are selected, appearance is refit once by the same two-pass rule and the
joint transform is applied to held-out cells. Thus a global brightness or color change
cannot win the geometry search merely by leaving an avoidable photometric residual.

The transform order is always spatial sampling first, then gain and bias.

## 7. Saved real arms and wrong-target controls

For every video, save untrimmed SSE, common element count, MSE, parameter arrays and
hashes, prediction hashes, retained-macrocell arrays and hashes, objective histories,
convergence-probe improvements, and recomputable predicates for:

- `persistence`;
- `global_translation_full/xfit` and `quadrant_translation_full/xfit` (exact MM-007
  external sentinels);
- `affine_full` and `affine_xfit`;
- `appearance_full` and `appearance_xfit`;
- `combined_full` and `combined_xfit`;
- near-target and far-target cross-fitted versions of affine, appearance, and combined.

Wrong-target transforms are fit against the frozen deranged target but scored against
the true target. No best arm, candidate set, parameter bound, or optimizer iteration is
selected after looking at real outcomes.

Evidence also stores every fold normalizer record; full/cross-fit parameter endpoint
SSE and element counts; separate site-flow, gradient, gain, and bias boundary counts
and denominators; exact synthetic expected parameters; and primitive performance,
pairing, complete-support, full-support, full-only, null-hit, fold, and relative-gate
booleans. Hash-only parameter evidence is forbidden.

## 8. Synthetic controls

Synthetic scenarios use independent PCG64 seeds and never read real-video values:

```text
translation  800800
affine       800801
appearance   800802
combined     800803
stationary   800804
independent  800805
```

Each scenario contains exactly six independently textured rows and uses the same R64
geometry, macrocells, cross-fitting, and current-only R8 normalizer rule. For native
pixel centers `y,x in {0,...,63}`, define `qy=(y+0.5)/64`, `qx=(x+0.5)/64` and the
ordered frequency list

```text
F = ((1,0),(0,1),(1,1),(2,1),(1,2),(3,2),(2,3))
```

For each seed, PCG64 draws, in order, float64 arrays `a ~ N(0,1)[6,3,7]`,
`b ~ N(0,1)[6,3,7]`, and `c ~ N(0,1)[6,3]`. The current texture for row `r`, channel
`ch` is

```text
0.35 * sum_k(
    a[r,ch,k] * sin(2*pi*(fy[k]*qy + fx[k]*qx))
  + b[r,ch,k] * cos(2*pi*(fy[k]*qy + fx[k]*qx))
) + 0.15*c[r,ch]
```

Fit the scenario's sole normalizer from the area-pooled R8 current texture, apply it
to R64, construct the central target in normalized space, invert that same normalizer
to form the target input, and then reapply it for scoring. The independent scenario
draws a second complete `(a,b,c)` triple from the same generator after the current
triple and uses it as target; it never refits on target.

Parameter order is `(ty,tx,ayy,ayx,axy,axx)`. The exact ground truths are:

```text
translation  theta=( 4,-4,0,0, 0, 0), gain=(1,1,1),          bias=(0,0,0)
affine       theta=( 0, 0,2,0, 0,-2), gain=(1,1,1),          bias=(0,0,0)
appearance   theta=( 0, 0,0,0, 0, 0), gain=(1.25,.75,1.5),   bias=(.35,-.25,.15)
combined     theta=(-4, 4,0,2,-2, 0), gain=(1.2,.8,1.4),     bias=(.3,-.2,.1)
stationary   theta=( 0, 0,0,0, 0, 0), gain=(1,1,1),          bias=(0,0,0)
```

These are interior flow and appearance values. A required positive arm must recover
every applicable parameter with maximum absolute error at most `1e-10`, in addition
to passing its full, cross-fit, and wrong-target-separation predicates.

Required controls are:

1. `translation`: translation, affine, and combined recover the signed transform and
   pass full and cross-fit performance; appearance does not pass;
2. `affine`: affine and combined pass and recover the known affine transform;
   translation and appearance do not pass;
3. `appearance`: appearance and combined pass and recover gain/bias; translation and
   affine do not pass;
4. `combined`: only the combined arm clears complete support and recovers both known
   parameter families;
5. `stationary`: persistence error is exactly zero and no arm creates support;
6. `independent`: an independently generated target with the same marginal texture
   construction cannot produce cross-fitted performance or complete support in any
   family;
7. held-out target mutation cannot change cross-fitted parameters or predictions;
8. deterministic replay produces bit-identical parameters, predictions, and summary.

The exact expected positive sets are `{global_translation, quadrant_translation,
affine, combined}` for `translation`, `{affine, combined}` for `affine`,
`{appearance, combined}` for `appearance`, and `{combined}` for `combined`. All other
families in those scenarios are excluded. Every positive arm must satisfy both
`2.0*full_mse <= persistence_mse` and `2.0*xfit_mse <= persistence_mse`, complete
support, and its parameter endpoint. Every excluded arm and every arm in `independent`
must satisfy `xfit_mse >= 0.90*persistence_mse` and fail complete support. Stationary
retains the exact-zero predicates above. These margins are frozen before synthetic
execution; they may not be weakened after observing a control. Any failed positive,
negative, endpoint, isolation, or invariance control invalidates MM-008.

## 9. Frozen support gates

For one video and mechanism `m`, let `p`, `o_m`, `n_m`, and `s_m` be persistence,
true-target cross-fit, near-wrong-target cross-fit, and far-wrong-target MSE. Per-video
complete support requires all of:

```text
p > 0
1.25 * o_m <= p
1.10 * o_m <= n_m
1.10 * o_m <= s_m
```

An arm passes only with complete support on at least 6/8 videos, any improvement
`o_m < p` on at least 7/8, and at least one completely supporting video in every
whole-video fold. Its full-fit arm must independently meet the corresponding 6/8,
7/8, and fold-coverage gate. A full/xfit inversion on at least three videos makes that
mechanism non-supporting and is reported explicitly.

Affine must also improve over its nested replayed R64 global-translation sentinel.
With `rho_A=o_affine/p` and `rho_G=o_global/p`, this incremental gate requires
`1.10*rho_A <= rho_G` on at least 6/8 videos with every fold represented and any
`rho_A < rho_G` on at least 7/8. Appearance has no nested translation comparator; its
identity comparator is already persistence. For the joint branch, combined must meet
the analogous gate separately against affine and appearance.

A wrong-target arm is a strong null hit if it independently clears the 20% persistence
effect. Counts are computed separately for each mechanism and separately for the near
and far mapping; they are never pooled across mappings or mechanisms. Six or more hits
in any one count invalidate the real assay; three to five make it
inconclusive. For one arm and video, `full_only` means the full fit clears the 20%
persistence gate while the cross-fit does not; a full/xfit inversion affects a family
when this occurs on at least three videos. A family fails cleanly only when primary
complete support, primary performance support, and full support are each below 3/8,
wrong-target null hits are below 3/8, no boundary warning is present, and synthetic
controls pass.

Boundary occupancy includes affine flow components at +/-8, gradient coefficients at
+/-4, and gain/bias values at either clip. A video warns when at least 25% of the
applicable saved full and cross-fit parameters occupy a boundary; warnings on at least
three videos make that mechanism range-inconclusive. Site-flow, four-gradient,
three-gain, and three-bias fractions are computed separately, with `rtol=0` and
`atol=1e-12`; the video statistic is their maximum. Large site-flow denominators may
not dilute a saturated scalar coefficient family.

After the two frozen coordinate-descent sweeps, evaluate one read-only convergence
probe: for each of the six coefficients in the same order, evaluate its five alternatives while
holding the final other coefficients fixed, without updating any coefficient. Record
the best strict objective improvement exceeding
`max(1e-12, 1e-10*final_objective)` for that fit context. The probe never changes the
reported prediction.

Probe aggregation is frozen per mechanism and video. Compute four fractions separately:
the fraction of true-target full fits (`rows` contexts), true-target cross-fits
(`2*rows`), near-target cross-fits (`2*rows`), and far-target cross-fits (`2*rows`)
with a remaining improvement above tolerance. A video warns when the maximum of those
four fractions is at least 0.25; warnings on at least three videos trigger the
optimizer-inconclusive preemption. Every combined-arm probe candidate reruns the exact
two-pass appearance solve on that context's frozen fitting cells. Appearance-only has
no iterative optimizer and therefore no convergence probe.

## 10. Frozen decision ladder

After exact parent replay and valid synthetic, null, boundary, fold, and consistency
controls, evaluate cross-fitted family passes:

1. `smooth_affine_family_supported`: affine and combined pass, affine meets its
   relative gate against global translation, and appearance fails cleanly;
2. `global_appearance_supported`: appearance and combined pass while affine fails
   cleanly;
3. `joint_affine_appearance_recovery_supported`: combined passes and meets its
   relative gate against both affine and appearance while those single families both
   fail cleanly;
4. `mixed_mechanism_nonidentifiable`: affine, appearance, and combined all pass and
   affine meets its incremental global-translation gate;
5. `tested_affine_appearance_ceiling_failure_supported`: affine, appearance, and
   combined all fail cleanly with fewer than three full-fit supporting videos per
   family;
6. `MM008_mechanism_factorial_inconclusive`: any valid pattern not covered above,
   including a single-family pass contradicted by combined failure, a full/xfit
   inversion affecting at least three videos, a boundary condition, or three to five
   null hits.

Also preempt to `MM008_mechanism_factorial_inconclusive` when a family has 3--5
performance, complete, or full supports; reaches six supports without the 7/8
improvement or fold gate; passes performance but fails true-versus-wrong-target
pairing; or has an unresolved optimizer probe. These patterns may not fall through to
a clean failure claim.

The translation sentinels cannot become supported new mechanisms: they must replay the
sealed MM-007 results. A smooth-affine result nominates a causal source-to-current
affine extrapolator; an appearance result nominates a causal channel-transform
extrapolator; a joint result nominates both in a separately frozen causal test. Clean failure
advances to a visibility/occlusion-versus-new-content diagnostic. No target-aware
parameter is eligible for the production model.

## 11. Lifecycle and verification requirements

Implementation must expose separate preparation, formal run, fast verification, and
semantic verification phases. Preparation may copy and validate sealed inputs and
current-only normalization prerequisites, but may not invoke any target-aware fitter,
synthetic generator, real scorer, or decision function. The formal marker is created
read-only immediately before the first scientific call and binds the protocol,
prepared inputs, exact source files, frozen config, interpreter, NumPy, and dependency
receipts.

Marker creation uses exclusive write, file `fsync`, chmod `0444`, directory `fsync`,
and a read-back check before exactly one top-level scientific executor is invoked.
After that marker boundary, science may read only the copied package inputs and
in-memory values derived from them; live parent or media reads are forbidden. Every
exclusive outcome write is file-fsynced and followed by directory fsync before the
artifact manifest is exclusively written and fsynced last.

The canonical formal run occurs exactly once. Fast verification must fail closed on
extra files, symlinks, wrong modes, mutable marker, hash drift, malformed/non-finite
JSON, receipt drift, arithmetic drift, report/result divergence, or source/config
changes. Semantic verification must additionally reconstruct the authenticated table,
normalizers, all synthetic and real primitives, summary, decision, and report from
the copied sealed inputs. Both verifiers are read-only.

The only canonical output is
`bench/multimodal_mechanism_diagnostics/results/MM-008`. A prepared package has exactly
10 files: `MM-008-protocol.md`, `input-manifest.json`, and the eight pinned parent
copies. A completed package has exactly 15 files: those ten, read-only
`formal-start.json`, `MM-008-evidence.json`, `MM-008-results.json`, `MM-008-report.md`,
and `artifact-manifest.json` written last. The only directories are the output root,
`inputs/`, and `inputs/MM-007/`. All files are regular and `0644` except the marker at
`0444`; extra files, directories, or symlinks fail closed. Preparation and
run use exclusive creation, temporary outputs are forbidden in the canonical tree,
and an interrupted run that has created its marker is terminal: the same canonical
path may be verified as interrupted but never resumed or rerun.

Real outcomes remain provisional until an independent arithmetic audit, package audit,
and scientific claim-boundary audit all pass.
