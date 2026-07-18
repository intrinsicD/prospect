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
