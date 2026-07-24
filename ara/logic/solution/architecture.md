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

## A03: Separate known-model acquisition from learned uncertainty

- **Statement**: Decision-relevant active acquisition is qualified in WM-002 as
  a known-model integration problem: Q0 proves exact finite semantics without
  interaction, and only a separately authorized Q1 may test the continuous
  execute-to-update-to-act-to-restore path. Learning calibrated predictive
  uncertainty is isolated in U-001 before transfer or scale.
- **Rationale**: Oracle agreement can prove value-of-information arithmetic and
  runtime wiring, but it cannot show that uncertainty was learned from
  experience. Combining acquisition, uncertainty learning, transfer, replay,
  horizon, and representation would make a result causally uninterpretable.
- **Provenance**: user-revised
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/active_acquisition/qualification.py`,
  `bench/active_acquisition/protocol.json`,
  `docs/u001-learned-uncertainty-plan.md`]
- **Evidence**: [N34, N35, N36, `docs/wm002-q0-results.md`]
- **From staging**: O22


## A04: Attempt-scoped authenticated worker bootstrap

- **Statement**: Every WM-002 Q1 producer and restore child must consume one
  attempt-scoped operational capability that binds its exact role, run, parent
  and child PID, master, arm, and normalized path set before reading durable
  entry or private salt inputs. The parent commits only the fresh secret digest
  in marker v2 and requires a domain-separated authenticated consumption
  acknowledgement over the same inherited AF_UNIX stream endpoint.
- **Rationale**: Path-bearing argv or environment state cannot authenticate a
  worker, and successful pipe enqueue cannot prove that a child decoded its
  authority. Site-disabled startup, a single inherited socket, PID binding,
  durable marker comparison, and a positive bounded acknowledgement close that
  bootstrap gap without adding scientific state or result evidence. The
  initially proposed one-way pipe transport was replaced after N41 falsified
  its receiver-progress assumption.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/active_acquisition/attempt.py`,
  `bench/active_acquisition/worker_capability.py`,
  `bench/active_acquisition/q1.py`,
  `bench/active_acquisition/restore_worker.py`]
- **Evidence**: [N39, N41, N42]
- **From staging**: O23
