# Architecture

## A01: One continuous causal evidence chain

- **Statement**: The first maturity milestone is one bounded run in which the
  same shared-parameter world model collects identified experience, consumes
  exactly those transition identities, improves executed held-out behavior,
  encounters shared-weight interference, retains the gain, and restores it in a
  fresh process.
- **Rationale**: Separately successful collection, prediction, control, and
  checkpoint demos cannot attribute a behavioral gain to one persistent update.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/experiment.py`,
  `bench/world_model_lifecycle/analysis.py`,
  `bench/world_model_lifecycle/parity.py`]
- **Evidence**: [N03, `docs/wm001-v130-formal-results.md`]
- **From staging**: O01

## A02: Owned, transactional model updates

- **Statement**: Attributable adaptive behavior requires a versioned,
  checkpointable model owner and a prepare-validate-commit learner boundary that
  binds consumed experience, predecessor bytes, candidate bytes, committed
  bytes, and the downstream model version.
- **Rationale**: A receipt alone cannot establish causality if a learner can
  mutate predictive parameters before the runtime validates and commits the
  update.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`src/prospect/runtime/learning.py`,
  `src/prospect/runtime/agent.py`,
  `bench/world_model_lifecycle/learning.py`]
- **Evidence**: [N03, `docs/architecture.md`]
- **From staging**: O02
