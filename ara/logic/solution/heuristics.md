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
