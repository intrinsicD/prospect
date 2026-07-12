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
