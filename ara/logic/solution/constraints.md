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

## R07: Stream large terminal members under full custody

- **Statement**: Manifest-bound terminal verification of a potentially large
  producer member must stream its exact byte count and SHA-256 through a
  no-follow descriptor while preserving regular-file, exact-link-count, stable
  before/after inode, mutation, and pathname-namespace checks.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/binding.py`,
  `tests/test_world_model_binding.py`,
  `tests/test_world_model_runtime_custody.py`]
- **Evidence**:
  [N11, N12, N13,
  `bench/world_model_lifecycle/results/operator-v1.7/closures/development-closure-v1.7.0`]
- **From staging**: O13

## R08: Stream bulk evidence at every independent consumer

- **Statement**: Every consumer that reconstructs a live producer namespace
  must stream each bulk file exactly once under canonical no-follow path,
  regular-file/link, pre/post descriptor identity, post-read path-to-descriptor,
  exact size/SHA-256, typed sorted manifest, namespace-equality, per-file, and
  aggregate limits. Passing one streamed closure or archive reader does not
  establish that a separate launcher or auditor scales.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/launch_bootstrap.py`,
  `bench/world_model_lifecycle/artifact_audit.py`,
  pending: fresh successor implementation and production-scale integration test]
- **Evidence**: [N19, N20, N22,
  `docs/wm001-v1150-formal-invocation-failure.md`]
- **From staging**: O14

## R09: Authenticate result-free pre-root rehearsal custody

- **Statement**: A result-free accepted-binding rehearsal used to establish
  return-code and exactly-once behavior must acquire a deterministic single-use
  claim and publish an authenticated accepted or failed terminal package. The
  rehearsal must neither grant formal authority nor create the binding-keyed
  formal root.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code refs**:
  [`bench/world_model_lifecycle/launch_bootstrap.py`,
  `bench/world_model_lifecycle/operator.py`,
  pending: fresh successor rehearsal claim, terminal, and adversarial tests]
- **Evidence**: [N23, N25, N26,
  `docs/wm001-v1160-accepted-binding-rehearsal-failure.md`]
- **From staging**: O19
