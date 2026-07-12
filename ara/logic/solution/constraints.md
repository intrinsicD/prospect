# Constraints

## K01: Retained elites do not expand the candidate budget
- **Constraint**: Kept and receding-horizon-shifted elites replace fresh proposals;
  they never increase the configured candidate population.
- **Rationale**: P5's ablation accounting requires equal world-model evaluation
  budgets across planner variants.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/prospect/planning.py::FlatPlanner.plan`, `tests/test_planner.py::test_icem_keeps_elites_with_constant_candidate_budget_and_shifts_them`]
- **From staging**: O02

## K02: U-002 ratchets per-seed P2 margins
- **Constraint**: For each seed, compute the P2 margin as planner return minus the
  larger of baseline and random return, then hold at least the U-001 floors
  `[17.263924584, 0.708487315, 2.304867583]`.
- **Rationale**: An ordinary P2 pass only requires a positive margin and would not
  detect a regression relative to the immediately preceding planner.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`tasks/U-002-icem-planner.md`, `bench/results/P2-20260710T131536Z.json`]
- **From staging**: O03
