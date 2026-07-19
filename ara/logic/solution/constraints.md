# Constraints

## R01: Offline data cannot establish the first closed-loop claim

- **Statement**: The retained BridgeControl and Perception Test datasets can
  support adapters, leakage controls, and later perception experiments, but
  cannot alone establish that one agent acted, observed, learned, and improved
  its executed behavior online.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`datasets/README.md`, N03]
- **From staging**: O04

## R02: Coverage endpoints require bound arithmetic semantics

- **Statement**: An inclusive probabilistic-coverage interval is not
  implementation-independent unless the protocol specifies the arithmetic and
  classification semantics at its discontinuous endpoints.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N02, `docs/wm001-v130-formal-results.md`]
- **From staging**: O05

## R03: Post-hoc audit correction cannot create confirmation

- **Statement**: Correcting or rerunning an auditor after formal outcomes may
  provide diagnostic evidence but cannot upgrade the bound attempt. New
  confirmation requires a prospectively fixed auditor, fresh protocol and seed
  domain, clean binding, and new immutable attempt.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N04, N05, `docs/wm001-v130-formal-results.md`]
- **From staging**: O08

## R04: Rejection custody must not weaken acceptance

- **Statement**: Pending and accepted adjudications require a coherent, clean,
  complete, passing audit. A coherent failed or incomplete audit may enter only
  an explicit rejected package with a separate fatal semantic finding.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/adjudication.py`,
  `tests/test_world_model_adjudication.py`]
- **Evidence**: [N04, N05]
- **From staging**: O09

## R05: Seal dependency visibility under exact adjudication execution

- **Statement**: A formal binding must establish dependency visibility under
  the exact adjudication execution mode, not only record distribution
  identities. Before outcomes, a no-outcome conformance fixture must produce
  byte-identical canonical audit reports under direct execution and the actual
  descriptor-bound isolated execution path.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/adjudication.py`,
  `bench/world_model_lifecycle/artifact_audit.py`]
- **Evidence**:
  [N09, N10, `docs/wm001-v140-formal-results.md`,
  `artifacts/wm001-audits/20260719-v140-adjudication-replay-diagnostic-1.json`]
- **From staging**: O11

## R06: Rejection custody must admit conformance failures

- **Statement**: A rejected adjudication path must be able to publish a
  source- and binding-identified failing audit when the fatal finding is
  runtime or coverage conformance itself. Acceptance still requires complete
  passing conformance; only the rejected evidence envelope may carry the
  failure.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/adjudication.py`,
  `tests/test_world_model_adjudication.py`]
- **Evidence**:
  [N09, N10, `docs/wm001-v140-formal-results.md`,
  `artifacts/wm001-audits/20260719-v140-formal-semantic-review-rejected.json`]
- **From staging**: O12
