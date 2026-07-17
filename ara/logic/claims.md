# Claims

## C01: The BridgeControl transition stack is a material failure source
- **Statement**: On the frozen BC-001 balanced cell with matched planner compute,
  replacing learned recursive ensemble-mean rollouts with exact target-refresh
  transitions improves mean return in 8/8 formal seeds, closes 79.35% of the paired
  oracle-return gap, and raises success from 6.25% to 84.375%. This identifies the
  transition-mean/recursive-refresh stack, not representation capacity alone.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The sealed semantic verifier fails, or a protocol-matched
  rerun does not satisfy the frozen 7/8 direction, 20% gap-closure, and fixed-bank
  regret criteria.
- **Proof**: [`bench/oracle_ladder_v2/results/OL-002/OL-002-results.json`,
  `bench/oracle_ladder_v2/results/OL-002/OL-002-report.md`,
  `ara/evidence/oracle-ladder-ol002-2026-07-14.md`]
- **Dependencies**: []
- **Tags**: BridgeControl, transition stack, recursive rollout, simulator oracle
- **From staging**: O11

## C02: Learned reward is a second material BridgeControl bottleneck
- **Statement**: On exact online-refresh BridgeControl paths, replacing learned reward
  with exact reward improves mean return in 8/8 formal seeds, closes 100% of the
  remaining paired oracle-return gap, and raises success from 84.375% to 100%. This
  identifies learned reward composed with online encoding, not reward-head weights
  alone.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The sealed semantic verifier fails, or a protocol-matched
  rerun does not satisfy the frozen 7/8 direction, 20% gap-closure, and fixed-bank
  regret criteria.
- **Proof**: [`bench/oracle_ladder_v2/results/OL-002/OL-002-results.json`,
  `bench/oracle_ladder_v2/results/OL-002/OL-002-report.md`,
  `ara/evidence/oracle-ladder-ol002-2026-07-14.md`]
- **Dependencies**: []
- **Tags**: BridgeControl, reward stack, online encoding, simulator oracle
- **From staging**: O12

## C03: Exact-reference injection does not rescue frozen BridgeControl MPC
<!-- CONFLICT: see staging/observations.yaml:O45 and trace/exploration_tree.yaml:N72 -->
- **Statement**: Under the frozen PI-003 protocol, injecting current-state exact-
  simulator reference elites into the unchanged learned candidate budget improves mean
  return from -2.660520 to -1.368870 but yields only 5/8 positive seeds, 16.28%
  oracle-gap closure, and 9.38% success. It misses the predeclared 7/8, 50%, and 80%
  rescue thresholds, while the action-permuted control also reaches 9.38% success;
  simple native proposal scarcity is therefore not the primary failure source on this
  authored fixture.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The PI-003 semantic verifier fails, or a protocol-matched
  preregistered rerun meets all frozen rescue thresholds and separates privileged from
  action-permuted injection.
- **Proof**: [`bench/proposal_injection_v3/results/PI-003/PI-003-results.json`,
  `bench/proposal_injection_v3/results/PI-003/PI-003-report.md`,
  `ara/evidence/proposal-injection-pi003-2026-07-14.md`]
- **Dependencies**: []
- **Tags**: BridgeControl, iCEM, proposal injection, proposal scarcity, negative result
- **From staging**: O17

## C04: The fixed-bank injection trigger does not transfer statewise
- **Statement**: Exact references are selected on 32/32 frozen initial candidate banks,
  but across 448 PI-003 online MPC calls only 9.598% place an injected reference in the
  learned first-round top elite, 7.366% make one first-round best, and 0% retain one as
  final best. The fixed-start trigger therefore fails the predeclared 50% statewise-
  transfer and final-retention conditions on this fixture.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The sealed semantic verifier fails, or a protocol-matched
  preregistered rerun places injected references in the first-round top elite and final
  best at or above the frozen 50% statewise floors.
- **Proof**: [`bench/proposal_injection/results/PI-001-trigger.json`,
  `bench/proposal_injection_v3/results/PI-003/PI-003-results.json::decision.commitment_audit`,
  `ara/evidence/proposal-injection-pi003-2026-07-14.md#action-commitment-audit`]
- **Dependencies**: [C03]
- **Tags**: BridgeControl, fixed bank, statewise transfer, receding-horizon MPC
- **From staging**: O18

## C05: CL-001 establishes neither proposed mechanism as fixture-wide robust
- **Statement**: Under the frozen CL-001 joint per-start and model-seed rules,
  within-call exploitation supports 6/12 untouched confirmatory seeds and visited-
  state scorer shift supports 3/12, both below the descriptive 10/12 floor. The
  accepted classification is `neither_mechanism_supported`; this is a failure to
  establish robust mechanisms, not evidence that either mechanism is absent.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The CL-001 semantic verifier fails, or a protocol-
  matched preregistered replication meets the frozen 10/12 rule for either endpoint.
- **Proof**: [`bench/candidate_landscape/results/CL-001/CL-001-results.json`,
  `bench/candidate_landscape/results/CL-001/CL-001-report.md`,
  `ara/evidence/candidate-landscape-cl001-2026-07-15.md`]
- **Dependencies**: [C04]
- **Tags**: BridgeControl, iCEM, candidate landscape, negative result, heterogeneity
- **From staging**: O24

## C06: The inherited harmful candidate signature recurs across fresh generators
- **Statement**: Under SS-001's frozen weaker recurrence rule, the inherited step-0
  signature occurs on 35/48 fresh calls and on at least 2/4 starts for 11/12 untouched
  generator seeds, passing the descriptive 10/12 recurrence gate. This establishes
  restart recurrence of the signature on the authored fixture, not its causal source;
  SS-001's failed calibration leaves mechanism and rescue branches non-interpretable.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The SS-001 semantic verifier fails, or a protocol-matched
  preregistered fresh-generator replication fails the frozen >=2/4-start and 10/12-seed
  recurrence rule.
- **Proof**: [`bench/scorer_swap/results/SS-001/SS-001-results.json`,
  `bench/scorer_swap/results/SS-001/SS-001-report.md`,
  `ara/evidence/scorer-swap-ss001-2026-07-15.md`]
- **Dependencies**: [C05]
- **Tags**: BridgeControl, iCEM, scorer swap, restart recurrence, calibration failure
- **From staging**: O25

## C07: Selected exact-good X preferences do not transport robustly in VP-001
- **Statement**: Under VP-001's frozen fixed-X sensitivity rule, SS-001's 25
  exact-good, old-panel-selected X identities support only 3/11 source generators and
  11/25 calls on untouched validator seeds 44-55, below the required 9/11 seed floor.
  This establishes failed same-data restart transport for those selected identities;
  it does not establish general inability to rank exact-better candidates.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The VP-001 semantic verifier fails, or deterministic
  replay of the sealed score tensors does not reproduce the 3/11 source-seed and
  11/25 call counts under the frozen 9/12 and 75%-within-seed rules.
- **Proof**: [`bench/validator_panel/results/VP-001/VP-001-results.json`,
  `bench/validator_panel/results/VP-001/VP-001-report.md`,
  `ara/evidence/validator-panel-vp001-2026-07-15.md`]
- **Dependencies**: [C06]
- **Tags**: BridgeControl, validator panel, selected candidate, transport failure,
  sensitivity control
- **From staging**: O30

## C08: MM-001 does not establish real visual temporal prediction
- **Statement**: Under the frozen MM-001 eight-video Perception Test sample protocol,
  TAESD's frame/image and framewise-video checks support 8/8 videos and SNAC supports
  6/8, but T5 supports 0/8 and the real-visual temporal gate supports 0/8. Every video
  beats ridge and the normalized shuffle control, yet the best world/persistence MSE
  ratio is 0.928 against the required <=0.833 margin. The accepted classification is
  `real_visual_temporal_prediction_not_supported`; this fails to establish temporal
  prediction or modality substitution on the preflight and does not assert general
  impossibility.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The MM-001 fast or semantic verifier fails, independent
  predicate replay does not reproduce the stored support counts and branch, or a
  protocol-matched preregistered replication reaches at least 6/8 support for the
  frozen real-visual temporal gate.
- **Proof**: [`bench/multimodal_preflight/results/MM-001/MM-001-results.json`,
  `bench/multimodal_preflight/results/MM-001/MM-001-report.md`,
  `ara/evidence/mm001-small-real-multimodal-preflight-2026-07-15.md`]
- **Dependencies**: []
- **Tags**: Perception Test, multimodal preflight, frozen codecs, temporal prediction,
  negative result
- **From staging**: O38

## C09: MM-002 does not identify a tested cause of MM-001's visual failure
- **Statement**: On the exact MM-001 eight-video projected-feature package, all six
  MM-002 world horizon/budget endpoints support 0/8 videos, the direct raw 32-D linear
  probe supports 0/8 at 0.5, 1.0, and 2.0 seconds, and no codec order/isolation/budget
  endpoint reaches 6/8. The nominal world baseline fails the frozen representation-
  integrity precondition on 21/144 probe rows, although all predictions remain finite
  with positive variance. The accepted classification is
  `world_diagnostic_inconclusive_representation_integrity`: MM-002 establishes no
  tested factor rescue and licenses no general claim that temporal signal or the tested
  factors are absent.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The MM-002 fast or semantic verifier fails, exact parent
  parity no longer holds, predicate replay changes the frozen support/integrity counts,
  or a protocol-matched replication establishes a healthy 6/8 rescue for a tested arm.
- **Proof**: [`bench/multimodal_diagnostics/results/MM-002/MM-002-results.json`,
  `bench/multimodal_diagnostics/results/MM-002/MM-002-report.md`,
  `ara/evidence/mm002-feature-only-failure-isolation-2026-07-15.md`]
- **Dependencies**: [C08]
- **Tags**: Perception Test, representation integrity, persistence baseline, codec
  migration, failure isolation, negative result
- **From staging**: O39

## C10: MM-003 does not support the tested projection or scale factors as the fix
- **Statement**: On MM-001's exact eight-video TAESD feature package, all twelve
  MM-003 linear conditions support 0/8 videos, including absolute-target and
  persistence-aware residual prediction from the complete raw 256-D latent. The
  inherited, post-z, orthonormal-subspace, and PCA-to-32 world arms also support 0/8
  and remain unhealthy; their paired material-improvement counts are 0/8, 1/8, and
  1/8. The accepted classification is
  `no_linear_full_taesd_signal_at_frozen_margin`: score scale, fixed-basis
  conditioning, 32-D random-subspace selection, and unsupervised PCA-to-32 are not
  supported as proximate fixes on this sample. This does not exclude nonlinear,
  spatial, motion-aware, history-aware, or more dynamic-data solutions.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The MM-003 fast or semantic verifier fails, exact
  raw/projection alignment or MM-002 parent parity no longer holds, predicate replay
  changes the frozen 0/8 probe/world counts, or a protocol-matched replication
  establishes a healthy 6/8 rescue for one of the tested transform factors.
- **Proof**: [`bench/multimodal_transform_diagnostics/results/MM-003/MM-003-results.json`,
  `bench/multimodal_transform_diagnostics/results/MM-003/MM-003-report.md`,
  `ara/evidence/mm003-taesd-projection-scale-isolation-2026-07-15.md`]
- **Dependencies**: [C08, C09]
- **Tags**: Perception Test, TAESD, projection, feature scaling, PCA, persistence,
  representation integrity, negative result
- **From staging**: O40

## C11: MM-004 does not support the tested one-second local predictors
- **Statement**: On MM-001's exact eight-video TAESD and authenticated source-pixel
  grids, MM-004's deterministic synthetic positive/negative controls pass all 24
  conditions and source activity supports 8/8 videos, but all four local-linear arms
  in each representation and horizon-scaled constant velocity support 0/8. The
  accepted classification is `tested_local_objective_or_horizon_failure_supported`:
  the frozen one-second local-linear objective is inadequate at its specified margins
  on this outcome-visible sample. This does not establish objective or horizon as a
  unique cause or exclude shorter, global, nonlinear, flow-based, longer-history, or
  action-conditioned solutions.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The MM-004 fast or semantic verifier fails, exact parent
  parity or synthetic recovery no longer holds, predicate replay changes the frozen
  8/8 activity and 0/8 arm/baseline counts, or a protocol-matched replication reaches
  at least 6/8 support for a tested one-second arm.
- **Proof**: [`bench/multimodal_spatial_diagnostics/results/MM-004/MM-004-results.json`,
  `bench/multimodal_spatial_diagnostics/results/MM-004/MM-004-report.md`,
  `ara/evidence/mm004-spatial-history-signal-isolation-2026-07-15.md`]
- **Dependencies**: [C08, C09, C10]
- **Tags**: Perception Test, TAESD, source pixels, spatial prediction, temporal
  history, prediction horizon, negative result
- **From staging**: O42

## C12: MM-005 does not support a half-second rescue for the tested local predictors
- **Statement**: On the exact 453-row matched panel reused from eight outcome-visible
  Perception Test sample videos, MM-005's dual-horizon synthetic positive and negative
  controls pass, every target-shuffle null supports 0/8, and central half-second source
  activity supports 7/8. Nevertheless, both frozen `current_3x3` and
  `current_diff_3x3` local-linear residual predictors support 0/8 in both TAESD and
  source pixels at 0.5 and 1.0 seconds, while paired half-horizon advantage supports
  0/8 in every arm/domain. The accepted classification is
  `half_second_tested_spatial_local_linear_objective_failure_supported`: halving the
  horizon does not rescue this tested family on this panel. It does not establish
  population-level unpredictability, failure of nonlinear/global prediction,
  teacher-free rollout failure, or that causal warp/flow is already a working fix.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The MM-005 fast or semantic verifier fails, exact parent
  or 453-row target alignment parity no longer holds, synthetic controls or primitive
  predicate replay change, or a protocol-matched replication reaches at least 6/8
  support and 6/8 paired normalized advantage for a tested half-second arm.
- **Proof**: [`bench/multimodal_horizon_diagnostics/results/MM-005/MM-005-results.json`,
  `bench/multimodal_horizon_diagnostics/results/MM-005/MM-005-report.md`,
  `ara/evidence/mm005-matched-half-horizon-replay-2026-07-15.md`]
- **Dependencies**: [C08, C09, C10, C11]
- **Tags**: Perception Test, TAESD, source pixels, matched horizon, spatial prediction,
  temporal history, local linear model, negative result
- **From staging**: O46

## C13: MM-006 does not support the tested discrete translation ceiling
- **Statement**: On MM-005's exact 453 half-second transitions from eight reused,
  outcome-visible Perception Test videos, MM-006's synthetic controls pass and central
  pixel activity supports 7/8, but every target-aware, causal, and source global or
  four-quadrant translation arm supports 0/8 in both source pixels and TAESD. The
  accepted classification `tested_pixel_warp_ceiling_failure_supported` rejects this
  frozen 8x8 discrete translation assay as the immediate fix. It does not rule out
  higher-resolution, continuous, affine, dense, photometric, visibility, generative,
  causal, or rollout mechanisms broadly.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: MM-006 fast or semantic verification fails, synthetic or
  parent parity changes, primitive replay changes the 0/8 support counts, or a
  protocol-matched replication reaches at least 6/8 support for a tested arm.
- **Proof**: [`bench/multimodal_warp_diagnostics/results/MM-006/MM-006-results.json`,
  `bench/multimodal_warp_diagnostics/results/MM-006/MM-006-report.md`,
  `ara/evidence/mm006-causal-warp-ceiling-2026-07-15.md`]
- **Dependencies**: [C08, C09, C10, C11, C12]
- **Tags**: Perception Test, pixel warp, translation, causal flow, oracle ceiling,
  cross-fitting, negative result
- **From staging**: O48

## C14: MM-007 does not support resolution alone as the frozen translation fix
- **Statement**: On the same eight videos and 453 transitions, MM-007 holds the native
  crop, physical macrocells, candidate translations, R8 normalization, folds, and
  gates fixed while increasing RGB resolution to R16/R32/R64. Every higher-resolution
  primary quadrant arm improves over persistence on 0/8 and supports 0/8; full-fit
  support is also 0/8. R32/R64 reduce the R8 penalty on 6/8 but retain pooled xfit
  ratios above one, so the relative pass is attenuation rather than recovery. The
  classification `physically_matched_resolution_failure_supported` rejects resolution
  alone for this discrete global/four-quadrant family, not pixel transport broadly.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: MM-007 fast or full semantic verification fails,
  authenticated-frame or R8 parent parity changes, control/primitive replay changes
  the absolute support counts, or a protocol-matched higher resolution reaches the
  frozen 6/8 absolute and relative recovery gates.
- **Proof**: [`bench/multimodal_resolution_diagnostics/results/MM-007/MM-007-results.json`,
  `bench/multimodal_resolution_diagnostics/results/MM-007/MM-007-report.md`,
  `ara/evidence/mm007-physically-matched-resolution-2026-07-15.md`]
- **Dependencies**: [C13]
- **Tags**: Perception Test, physical resolution, translation, cross-fitting, oracle
  ceiling, negative result
- **From staging**: O49

## C15: MM-011's frozen coarse grid fails generated translation directionality
- **Statement**: On MM-011's frozen six-cell synthetic-target-only panel, the exact
  v2.2 coarse-grid candidate passes activity, historical identification, forward
  persistence improvement, and exact replay for all six cells; A1-A3 also pass
  directionality. Translation cells T1-T3 fail only directionality: their forward
  error ratios are 0.57414, 0.62426, and 0.56189, while their previous-as-future
  reversal ratios are 0.59482, 0.69081, and 0.75176 against a required value greater
  than 0.8. The preregistered decision is `ABANDON_FINITE_GRID_BEFORE_REAL_DATA`.
  This rejects this exact grid on this generated panel, not all finite grids,
  continuous registration, motion prediction, real-video performance, or Prospect
  capability.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The immutable audit or semantic receipt fails its bound
  hashes, clean regeneration ceases to be bit-exact, independent arithmetic or input
  reconstruction changes any declared predicate, or a protocol-matched reproduction
  makes every T1-T3 reversal ratio strictly greater than 0.8 while preserving all
  other gates.
- **Proof**: [`docs/research/2026-07-16-mm011-offgrid-sensitivity-audit.json`,
  `docs/research/2026-07-16-mm011-offgrid-sensitivity-semantic-verification.json`,
  `docs/research/2026-07-16-mm011-offgrid-sensitivity-result-audit.md`,
  `ara/evidence/mm011-offgrid-sensitivity-2026-07-16.md`]
- **Dependencies**: [C14]
- **Tags**: synthetic target, off-grid motion, coarse affine grid, directionality,
  reversal control, causal prediction, pre-real negative result
- **From staging**: O59
