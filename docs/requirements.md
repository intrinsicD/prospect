# Requirements & traceability

Eight requirements, stable IDs. Every requirement maps to a core module, a locked
decision (ADR), and a benchmark kill-gate. An agent working on R_n should be able to
find its seam, its rationale, and its acceptance test from this table.

| ID | Requirement | Active E-series seam | Governing ADR | Evidence |
|----|-------------|----------------------|---------------|----------|
| R1 | Predict consequences of actions and plan action sequences | `domain.Prediction`, `PredictiveModel`, `CandidateAssessment`, runtime | 0001, 0014 | E1 semantics; E3 prediction; E4 behavior |
| R2 | Hierarchical planning | future model/policy adapter | 0003, 0014 | not yet earned in E-series |
| R3 | Test what was learned against expectation | `Prediction` → `ProperScore`; `BeliefUpdate` → typed `EpistemicEffect` | 0014 | E1, E3 |
| R4 | Identify outcome-relevant input structure | versioned `Belief`/`Distribution`; future representation adapter | 0001, 0006, 0014 | E3 plus representation sentinels, pending |
| R5 | Use learned patterns correctly | `CandidateAssessment`, `DecisionRecord`, external held-out evaluator | 0003, 0014 | E4 |
| R6 | Process required input/output modalities | future backend codec adapter with representation identity | 0001, 0009, 0014 | modality-specific E-series gate pending |
| R7 | Improve over time | runtime, canonical experience/transition ledger, `UpdateReceipt`, checkpoint manifest | 0005, 0014 | E2 collect, E3 learn, E4 improve, E5 retain |
| R8 | Use internal/external knowledge safely | retrieval/tool/query as explicit assessed actions with evidence provenance | 0004, 0014 | exact negative controls in E1; live source gate pending |

## Notes
- R4 is not established merely because a latent predicts a training set. It needs
  held-out prediction, calibration, intervention/shift controls, and
  representation-integrity evidence under a named model and representation version.
- R7 is a causal evidence program, not a training-loss metric. Collect, learn,
  improve, checkpoint equivalence, retention, and plasticity are separate results.
- R2 was designed in detail before it became a first-class requirement; the two-level
  jumpy-model planner is part of the spec, not an add-on.
- Collapse prevention (ADR-0006) remains necessary, but an uncertainty sentinel is
  not allowed to substitute for a held-out learning or behavior result.
- P0–P14 and supplementary historical runs remain in Git history and research
  narratives under their original contracts. Their active code/tests were removed
  at cutover; none pre-passes E0–E5.
