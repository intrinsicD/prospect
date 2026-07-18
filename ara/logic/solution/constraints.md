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
