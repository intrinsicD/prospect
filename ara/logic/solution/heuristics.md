# Heuristics

## H01: Audit coverage in target-count space

- **Rationale**: Coverage is a discrete count divided by a known target count.
  Comparing aggregate binary fractions can reject an intended one-target
  allowance by an ulp. Recover and compare integer counts, reject off-grid
  stored fractions, and require any permitted disagreement to be prospectively
  bound to explicit endpoint arithmetic and a numerical guard.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**:
  [`bench/world_model_lifecycle/artifact_audit.py`;
  pending: exact endpoint guard and arithmetic must be sealed in the next
  protocol]
- **Evidence**: [N02, N05]
- **From staging**: O06

## H02: Rehearse exact accepted-binding custody before formal claim

- **Rationale**: Small synthetic launch fixtures and neighboring streaming
  verifiers can pass while the standard-library formal consumer fails on real
  file roles. Preserve the existing result-qualification projection as a
  terminal-bound binding sidecar, rejoin it to one streamed raw-result digest,
  then invoke the existing result-free outer preformal-runtime path with the
  accepted binding and production-scale roles before creating the formal root.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**:
  [`bench/world_model_lifecycle/launch_bootstrap.py`,
  `bench/world_model_lifecycle/binding.py`,
  pending: fresh successor preclaim gate and integration fixture]
- **Evidence**: [N20, N21, N22,
  `docs/wm001-v1150-formal-invocation-failure.md`]
- **From staging**: O16
