# Heuristics

## H01: Test colored noise through the scored candidate pool
- **Rationale**: Compare beta-2 and beta-0 proposals under the same seed at a
  nontrivial horizon, and inspect the batch received by `plan()`, so the test covers
  planner-loop wiring as well as the FFT helper.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: medium
- **Code ref**: [`tests/test_planner.py::test_colored_noise_is_temporally_correlated_without_losing_offsets`]
- **From staging**: O01

## H02: Preserve finite DC power in short colored sequences
- **Rationale**: Assign the DC bin the lowest resolved frequency's scale and normalize
  by the inverse-filter kernel norm. This preserves unit Gaussian marginals without
  forcing each short proposal to have zero temporal mean.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/prospect/planning.py::FlatPlanner._sample_colored_noise`]
- **From staging**: O04

## H03: Test a matched assay before adding a coverage mechanism
- **Rationale**: Use a non-gated, matched fixture with the unchanged world model and
  planner before changing production collection or coverage semantics. This makes an
  invalid assay or negative result useful without contaminating shipped claims. The
  verbally accepted five-seed draft was strengthened to eight formal blocks by N13
  before execution.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high
- **Code ref**: [`bench/bridge_control/`, `docs/research/2026-07-13-bridge-control-protocol.md`]
- **From staging**: O07

## H04: Gate a causal factorial with exact and learned positive controls
- **Rationale**: Structural contrasts are uninterpretable when the frozen learner and
  planner cannot solve the designated balanced arm. Run exact and balanced learned
  controls first, allow at most one factor-separation redesign, and stop before other
  arms when either control misses its frozen threshold.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/bridge_control/experiment.py::run`, `tests/test_bridge_control.py`]
- **From staging**: O09

## H05: Verify experiment semantics in the artifact's canonical domain
- **Rationale**: Sort human-report records explicitly and normalize regenerated and
  saved semantic fields through exact finite sorted-key JSON before equality. This
  preserves scientific content while preventing mapping insertion order and tuple/list
  implementation details from causing false terminal mismatches; non-finite or
  content-changing values must still fail.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/proposal_injection_v3/experiment.py::_canonical_json_value`,
  `tests/test_proposal_injection_v3.py::test_canonical_json_value_normalizes_containers_but_not_content`]
- **From staging**: O20

## H06: Audit adaptive candidate pools before changing planner compute
- **Rationale**: When a useful fixed-bank proposal trigger disappears during iCEM,
  retain every live pool and score identical sequences learned and exact before adding
  candidates or a mitigation. Separate later refinement from the round-0 best, use
  joint repeated-measure support, and hold cold-bank companions byte-identical across
  visited states. CL-001 strengthened the staged 7/8 draft to 12 untouched seeds with
  a descriptive 10/12 floor and exposed heterogeneity that aggregate medians would
  have hidden.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/candidate_landscape/`,
  `docs/research/2026-07-15-candidate-landscape-cl001-protocol.md`,
  `tests/test_candidate_landscape_experiment.py`,
  `tests/test_candidate_landscape_planner.py`]
- **From staging**: O23

## H07: Triangulate visual failures before scaling model or data
- **Rationale**: Preserve native spatial grids, expose recent differences to a tiny
  structured predictor, require recovery of a known synthetic process, and cross-check
  the same causal rows with target/history shuffles and authenticated source pixels.
  This combination made MM-004 informative even though every real arm failed: it
  separated assay validity and source activity from isolated TAESD failure and from
  adequate one-second local-linear prediction before any larger adapter or dataset
  investment.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/multimodal_spatial_diagnostics/`,
  `docs/research/2026-07-15-mm004-spatial-history-signal-isolation-protocol.md`,
  `ara/evidence/mm004-spatial-history-signal-isolation-2026-07-15.md`]
- **From staging**: O41

## H08: Match causal rows and normalize to persistence before changing horizon
- **Rationale**: When horizon is the remaining explanation for a visual prediction
  failure, compare targets on the exact same source rows, folds, train-only
  normalizer, features, controls, and estimator. Judge each horizon relative to its
  own persistence error and require paired within-video improvement. MM-005 used this
  design on 453 rows and showed that the apparently easier half-second target did not
  rescue either tested arm, avoiding a raw-MSE or endpoint-membership confound before
  moving to a nonlinear model class.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/multimodal_horizon_diagnostics/`,
  `docs/research/2026-07-15-mm005-matched-half-horizon-replay-protocol.md`,
  `ara/evidence/mm005-matched-half-horizon-replay-2026-07-15.md`]
- **From staging**: O44

## H09: Use a target-leaking transport ceiling before adopting a causal warp
- **Rationale**: Compare persistence, a teacher-free previous-to-current transform
  applied forward, and a clearly target-leaking current-to-future ceiling on identical
  causal rows. Retain moving-texture positives and shuffled-history negatives. MM-006
  used this ordering to show that the tested discrete global/quadrant translation
  family failed even with future-target access, preventing a causal warp implementation
  from being mistaken for a validated fix.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/multimodal_warp_diagnostics/`,
  `docs/research/2026-07-15-mm006-causal-warp-ceiling-protocol.md`,
  `ara/evidence/mm006-causal-warp-ceiling-2026-07-15.md`]
- **From staging**: O47

## H10: Select expensive control subsets only after exhaustive parity
- **Rationale**: If an expensive metamorphic control retains a declared subset, score
  only that subset through a closed internal API, but first compare every retained
  result against the legacy superset at nested bit level. MM-008 retained 108 of 336
  transpose contexts with 108/108 parity and removed exactly 640,452 candidate
  reductions. Exact work removal is the causal efficiency fact; single before/after
  wall-clock observations remain diagnostics.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/multimodal_mechanism_diagnostics/transpose_v22.py`,
  `bench/multimodal_mechanism_diagnostics/controls_v22.py`,
  `tests/test_mm008_v22_transpose_selective.py`,
  `ara/evidence/mm008-v22-development-validation-2026-07-16.md`]
- **From staging**: O53

## H11: Test a source-only causal operator before retraining the multimodal model
- **Rationale**: Fit deformation and appearance strictly from previous-to-current
  observations, freeze every prediction before any target access, and score application
  to the current frame against the unseen future. Require synthetic recovery,
  history/reversal failures, mutation-based future isolation, and a persistence-relative
  real-video gate before changing the end-to-end training path. MM-009 commits this
  smallest mechanism test while keeping its outcome exploratory and target-custodied.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/multimodal_causal_diagnostics/`,
  `docs/research/2026-07-16-mm009-causal-deformation-appearance-prediction-protocol.md`,
  `tests/test_mm009_*.py`]
- **From staging**: O57

## H12: Require temporal directionality before promoting image registration
- **Rationale**: A transform can lower current-to-future persistence error by
  interpolating or smoothing without estimating the arrow of motion. On generated
  off-grid cases, repeat the fitted previous-to-current change from current toward
  future, then substitute the previous frame as the target and require the same
  forecast to lose the declared improvement margin. MM-011's T1-T3 cells looked
  strong forward yet also approached the previous frame, so this control rejected the
  coarse grid before any scarce real target was opened.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`bench/multimodal_causal_assay/offgrid_sensitivity.py`,
  `tests/test_mm011_offgrid_sensitivity.py`,
  `docs/research/2026-07-16-mm011-lcv-backed-causal-deformation-appearance-prediction-protocol.md`,
  `ara/evidence/mm011-offgrid-sensitivity-2026-07-16.md`]
- **From staging**: O60
