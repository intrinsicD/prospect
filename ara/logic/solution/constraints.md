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

## K03: Joint oracle rungs do not identify a single failure component
- **Constraint**: BC-001's exact-transition/learned-reward diagnostic also replaces
  learned latent rollouts with raw exact state and sets epistemic to zero. A rescue can
  exclude the reward head as the sole blocker, but cannot uniquely identify transition,
  representation, or uncertainty handling as the cause.
- **Rationale**: Component attribution requires rungs that change one interface at a
  time; endpoint success under a bundled intervention supplies only a joint bound.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`bench/bridge_control/results/BC-001/BC-001-results.json`, `bench/bridge_control/results/BC-001/BC-001-report.md`]
- **From staging**: O10

## K04: OL-001 and OL-002 count as one experiment
- **Constraint**: OL-002 is an administrative full rerun frozen after OL-001 numeric
  outcomes were available. Matching outcomes cannot be treated as an independent
  replication or counted twice.
- **Rationale**: OL-002 changes only the experiment/schema identifiers and canonical
  CSV newline rendering while inheriting and machine-checking every scientific field;
  outcome visibility removes independence.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-14-ol001-verifier-failure.md`,
  `docs/research/2026-07-14-oracle-prefix-ladder-ol002-protocol.md`,
  `bench/oracle_ladder_v2/results/OL-002/artifact-manifest.json`]
- **From staging**: O14

## K05: OL-002 establishes no minimum oracle-prefix recovery depth
- **Constraint**: Do not report a minimum recovery depth from OL-002 even though
  `k=8` and `k=12` each pass the aggregate recovery rule.
- **Rationale**: Seven of eight seed returns reverse from `k=8` to `k=12`, triggering
  the frozen no-knee rule; success is also nonmonotonic across the executed prefix
  curve.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`bench/oracle_ladder_v2/results/OL-002/OL-002-results.json::decision.prefix`,
  `bench/oracle_ladder_v2/results/OL-002/OL-002-report.md`,
  `ara/evidence/oracle-ladder-ol002-2026-07-14.md`]
- **From staging**: O15
