# Roadmap (benchmark-gated)

Each phase ships **only** when its kill-gate in `bench/gates.py` passes. Do not build
ahead of the current gate. Gates are the project's fitness function; keep their pass
criteria precise even while the eval body is a TODO.

| Phase | Goal | Deliverable | Kill-gate (summary) | Advances |
|-------|------|-------------|---------------------|----------|
| P0 | Scaffold | this repo | imports clean, smoke tests green | — |
| P1 | Flat world model + calibrated uncertainty | `FlatWorldModel` | latent 1-step beats a persistence/linear baseline; epistemic falls with data while aleatoric persists on a stochastic variant | R1, R4 |
| P2 | Planning beats reaction | `FlatPlanner` (MPC/CEM) | imagined MPC beats a model-free baseline at **equal env-step budget** | R1 |
| P3 | VoE + curriculum + replay | `SurpriseCompetenceMonitor`, `ReplayBuffer` | expected-vs-violated surprise differential reliable over seeds; curiosity beats random exploration | R3 (R7 groundwork) |
| P4 | Skills + router | `SkillRouter` | simulate-to-select picks the right skill above baseline; misapplication flagged by a surprise spike | R5 |
| P5 | Hierarchical planning | `JumpyOptionModel`, `HierarchicalManager` | two-level planning beats flat on a long-horizon task at **equal compute** | R2 |
| P6 | Any-to-any codec | `UniversalCodec` | swapping the single-modality codec for the universal one preserves core-loop performance within tolerance | R6 |
| P7 | Continual improvement | consolidation + metrics | on a task sequence: retention above threshold (no catastrophic forgetting) AND plasticity retained (late tasks learn as fast as early) | R7 |
| P8 | Knowledge bases | `UncertaintyMemoryRouter`, `KnowledgeSource`s | uncertainty-gated retrieval beats no-retrieval on the use-case benchmark AND stays robust to a poisoned/low-trust source | R8 |

## Sequencing notes
- **Build the predictive core and its uncertainty estimate first** (P1). Every other
  requirement is a consumer of that core or a wrapper around it.
- **Introduce the episodic buffer + generative replay early** (P3): it is both the
  memory substrate and the anti-forgetting mechanism, so it earns its place twice.
- **The universal codec comes last** (P6): any-to-any is an interface change to a
  working system, not a prerequisite.
- Treat **P7 as a discipline**, not a module: name the improvement metric, watch for
  forgetting via VoE, watch for plasticity loss.
- **Every phase gate also enforces its applicable collapse sentinels** (ADR-0006): a
  phase passes only if its capability criterion holds AND representation-integrity &
  uncertainty-reliability (active from P1), replay-fidelity (from P3) and
  option-diversity (from P5) are healthy. Collapse hides in a good loss, so integrity
  is measured, not assumed.
