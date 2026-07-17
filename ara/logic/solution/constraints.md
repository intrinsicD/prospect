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

## K06: PI-001, PI-002, and PI-003 count as one experiment
- **Constraint**: PI-001 and PI-002 are preserved terminal verifier failures, and
  PI-003 is an outcome-visible administrative full rerun. Their matching scientific
  outcomes cannot be treated as independent replications or counted more than once.
- **Rationale**: PI-002 changes only canonical report ordering; PI-003 changes only
  canonical semantic comparison. Each fresh namespace retrains and reevaluates all
  seeds and hash-binds its predecessor, but neither repair restores independence after
  prior outcomes were visible.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-14-pi001-verifier-failure.md`,
  `docs/research/2026-07-14-pi002-semantic-verifier-failure.md`,
  `docs/research/2026-07-14-proposal-injection-pi003-protocol.md`,
  `bench/proposal_injection_v3/results/PI-003/PI-003-results.json::protocol.method_delta`]
- **From staging**: O21

## K07: CL-001's seed threshold is descriptive, not inferential
- **Constraint**: Treat each frozen 10/12 endpoint as a descriptive robustness rule;
  do not report a binomial p-value or environment-level generality from CL-001.
- **Rationale**: The 12 model restarts share one authored dataset, four starts are
  repeated measures within a seed, and each endpoint is a conjunction of directional,
  rank, provenance, and threshold conditions rather than an exchangeable fair-sign
  statistic.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-15-candidate-landscape-cl001-protocol.md`,
  `bench/candidate_landscape/results/CL-001/CL-001-results.json::protocol.decision`,
  `bench/candidate_landscape/results/CL-001/CL-001-report.md`]
- **From staging**: O27

## K08: SS-001's mechanism and rescue branches are non-interpretable
- **Constraint**: Preserve `auditor_direction_control_failed` as SS-001's terminal
  classification. Do not promote its 24/35 call-level shared-transfer count, 25/35
  exact-rescue count, or corresponding 4/12 seed summaries into mechanism support.
- **Rationale**: All 11 exact-improving controls passed pairwise auditor direction, but
  the global rank selector materially degraded C on three controls. That left only 3/6
  calibration seeds at a frozen 5/6 gate, so pairwise calibration did not validate
  adaptive winner selection over the 186-candidate union.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-15-scorer-swap-ss001-protocol.md`,
  `bench/scorer_swap/results/SS-001/SS-001-results.json::decision`,
  `bench/scorer_swap/results/SS-001/SS-001-report.md`,
  `ara/evidence/scorer-swap-ss001-2026-07-15.md`]
- **From staging**: O28

## K09: VP-001's target panel is non-interpretable after sensitivity failure
- **Constraint**: Preserve `validator_fixed_X_sensitivity_control_failed` as VP-001's
  terminal classification. Do not promote its one held-out rejection, one inconclusive
  row, and one held-out transfer to `heterogeneous_target_failure`, finite-panel
  winner's-curse support, or shared-blind-spot support.
- **Rationale**: Direction passed 6/6 source seeds and target-local C-over-B0 passed
  3/3, but fixed-X sensitivity supported only 3/11 source generators against the
  frozen 9/11 requirement. The protocol places this control before every target
  branch and saves `target_panel.interpretable=false`.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-15-validator-panel-vp001-protocol.md`,
  `bench/validator_panel/results/VP-001/VP-001-results.json::decision`,
  `bench/validator_panel/results/VP-001/VP-001-report.md`,
  `ara/evidence/validator-panel-vp001-2026-07-15.md`]
- **From staging**: O31

## K10: Frozen modality codecs require explicit Prospect adapters
- **Constraint**: Do not pass a pretrained codec's native spatial tensors,
  hierarchical audio indices, or token sequences directly into Prospect's shared
  `LatentState`. Bind a deterministic pooling/projection representation first, then
  use `UniversalCodec` to distill each modality into the incumbent world-model latent.
- **Rationale**: The pretrained components have incompatible native structures and do
  not implement Prospect's one-vector codec or diagonal-Gaussian prediction contracts.
  MM-001 therefore retains raw frontend outputs for audit, uses fixed Rademacher
  projections for its 32-dimensional inputs, and interprets SNAC IDs only as projected
  features rather than Gaussian latent states.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-15-mm001-small-real-multimodal-preflight-protocol.md`,
  `bench/multimodal_preflight/backends.py`,
  `bench/multimodal_preflight/core.py`,
  `bench/multimodal_preflight/results/MM-001/MM-001-projections.npz`]
- **From staging**: O36

## K11: MM-004's activity branch is not a unique causal attribution
- **Constraint**: Preserve `tested_local_objective_or_horizon_failure_supported` as a
  residual disjunction. Do not report source activity as predictability, select horizon
  over objective/model class as proven, or treat the priority recommendation to change
  objective/horizon before data as a general causal result.
- **Rationale**: Pixel activity supports 8/8 while every frozen arm supports 0/8, but
  both main arms improve over persistence on all videos and the assay tests only one
  horizon, two local receptive fields, one history interval, and a linear ridge class.
  Negative delta cosine and catastrophic constant velocity prioritize a matched shorter
  horizon test; they do not exclude nonlinearity, global motion, partial observability,
  aliasing, stochastic futures, or concurrent TAESD limitations.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-15-mm004-spatial-history-signal-isolation-protocol.md`,
  `bench/multimodal_spatial_diagnostics/results/MM-004/MM-004-results.json`,
  `ara/evidence/mm004-spatial-history-signal-isolation-2026-07-15.md`]
- **From staging**: O43

## K12: Appearance nulls require a comparator stronger than persistence
- **Constraint**: Do not validate an unrestricted per-channel gain/bias target-aware
  fitter on independent targets by requiring its MSE to remain near source persistence.
  Compare it with a cross-fitted per-channel constant or shrinkage baseline and retain
  wrong-target pairing and complete-support gates.
- **Rationale**: For independent equal-variance source and target, gain near zero plus
  the fitted target mean has expected residual near one target variance, while source
  persistence has roughly two. MM-008 v1 consequently produced false complete support
  even without correspondence; the shared low-rank texture made the defect worse but
  was not its sole cause.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-15-mm008-synthetic-control-no-go.md`,
  `bench/multimodal_mechanism_diagnostics/method.py`,
  `tests/test_mm008_method.py`]
- **From staging**: O51

## K13: Cold-process equality is a condition-scoped reproducibility diagnostic
- **Constraint**: Report an equal complete-evidence root only for the exact source,
  machine, interpreter/library environment, config token, exposed seed, and process
  conditions executed. Do not call two administrative fresh runs independent
  replication, sealed evidence, config authentication, cross-machine portability,
  unseen-seed generalization, or capability support.
- **Rationale**: MM-008 collections under `PYTHONHASHSEED=0` and `4294967295` produced
  the same alias-insensitive bit-value root while holding every other condition fixed.
  The experiment closes hidden hash-order dependence for those two settings but varies
  neither implementation, machine, dataset, nor scientific seed.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`bench/multimodal_mechanism_diagnostics/cold_digest_v22.py`,
  `tests/test_mm008_v22_cold_digest.py`,
  `ara/evidence/mm008-v22-development-validation-2026-07-16.md`]
- **From staging**: O54

## K14: Development schema infrastructure cannot self-authorize formal execution
- **Constraint**: Keep MM-008 formal config/output emission unavailable until an
  authoritative artifact fixes exhaustive output roles, record/context grammar,
  branch identities, and native runtime policy. A syntactically valid config hash,
  development schema DSL, or ephemeral in-memory evidence graph is insufficient.
- **Rationale**: The adversarial development schema model is closed and validated, but
  independent review found that the protocol does not uniquely determine the missing
  role and runtime bindings. The contract therefore exposes `formal_authority=false`
  and fails closed instead of inventing a schema.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`bench/multimodal_mechanism_diagnostics/contract_v22.py`,
  `bench/multimodal_mechanism_diagnostics/schema_model_v22.py`,
  `tests/test_mm008_v22_contract.py`, `tests/test_mm008_v22_schema_model.py`,
  `ara/evidence/mm008-v22-development-validation-2026-07-16.md`]
- **From staging**: O55

## K15: Hard lifecycle walls require containment beyond process ancestry
- **Constraint**: Do not claim a complete experiment wall by killing only the root
  process group or by snapshotting `/proc` descendants when nested workers may create
  sessions or reparent. Place the whole formal lifecycle in one OS-owned cgroup, bind
  runtime expiry and interruption cleanup to that cgroup, and reject inner-body calls
  outside the authorized lifecycle role.
- **Rationale**: MM-009's PID-tree prototype returned success while a nested `setsid`
  child survived its root and wrote a delayed sentinel; a double-forked timeout orphan
  escaped independently. A transient cgroup-v2 user-systemd service contained normal
  root exit, timeout/reparenting, and SIGINT cleanup in permanent regressions.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`bench/multimodal_causal_diagnostics/experiment.py`,
  `tests/test_mm009_experiment.py`,
  `docs/research/2026-07-16-mm009-causal-deformation-appearance-prediction-protocol.md`,
  N96, N97, N98]
- **From staging**: O58

## K16: Durable cleanup and executed controller bytes are canonical evidence
- **Constraint**: Do not treat a formal or semantic package as self-authenticating if
  cleanup exists only in ephemeral supervisor output, cannot be cross-linked to the
  actual child runtime unit, or the executed controller can import live source or
  bytecode outside its reviewed closure. Finalize through a separate durable record
  that binds the child unit, cleanup state, provisional result, and copied bytecode-free
  controller bytes.
- **Rationale**: Independent MM-011 code review found that an otherwise plausible
  LCV-backed harness could lose formal/semantic cleanup at the CLI boundary, inspect a
  different unit than the child that performed the work, and import broad live roots
  before validating source identity. Those defects make a future result ambiguous
  even when its scientific payload is deterministic; MM-011 therefore never received
  real-run authorization.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Evidence**: [`docs/research/2026-07-16-mm011-offgrid-sensitivity-result-audit.md`,
  `tasks/MM-011-lcv-backed-causal-deformation-appearance-prediction.md`,
  `bench/multimodal_causal_assay/supervisor.py`, N102]
- **From staging**: O61
