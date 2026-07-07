# Requirements & traceability

Eight requirements, stable IDs. Every requirement maps to a core module, a locked
decision (ADR), and a benchmark kill-gate. An agent working on R_n should be able to
find its seam, its rationale, and its acceptance test from this table.

| ID | Requirement | Core module | ADR | Gate |
|----|-------------|-------------|-----|------|
| R1 | Predict consequences of actions and plan action sequences | world_model.py, planning.py | 0001 | P1, P2 |
| R2 | Hierarchical planning | planning.py | 0003 | P5 |
| R3 | Use violation of expectation to test whether an action is learned | voe.py | 0002 | P3 |
| R4 | Identify the right patterns in the input | world_model.py (latent), codec.py | 0001 | P1 |
| R5 | Use the right learned patterns correctly | skills.py | 0002, 0003 | P4 |
| R6 | Process any kind of input, produce any kind of output | codec.py | 0001, 0009 | P6, P12 |
| R7 | Improve over time | memory.py, voe.py, observation.py | 0002, 0005, 0010 | P7, P13 |
| R8 | Use different knowledge bases (internal and external) for any use case | memory.py, knowledge.py | 0004 | P8, P10, P11 |

## Notes
- R4 is not a separate module: it is a *property* pressured into the latent by
  predicting in latent space. Its gate is the same as R1's (calibration/prediction).
- R7 is mostly a **measurement discipline** (retention, plasticity) layered on the
  learning loop, plus generative replay introduced early in P3 — not a big new module.
- R2 was designed in detail before it became a first-class requirement; the two-level
  jumpy-model planner is part of the spec, not an add-on.
- Collapse prevention (ADR-0006) is not a separate requirement — it protects the shared
  latent and the calibrated uncertainty signal that R1, R3, R4 and R7 all read. It is
  enforced by integrity **sentinels** in `bench/gates.py`, which gate every phase.
- **R1 has non-gated supplementary evidence** beyond P1/P2: the optional harder-benchmark
  tier (BH-001, ADR-0011) re-runs the P2 claim on real MuJoCo (DeepMind Control Suite) via
  the `bench.Environment` seam. It is *evidence*, not a gate — deliberately outside the
  numpy-only ratchet (`make bench-hard`, `[bench-hard]` extra). Report: `bench/hard/results/`.
