# Architecture

## A01: Harness-side frozen tiny multimodal preflight stack
- **Design**: Keep heavyweight modality backbones outside the task-unspecific Prospect
  core. MM-001 runs immutable TAESD vision and framewise video, SNAC-24k audio, and
  T5-Efficient-Tiny text sequentially in the benchmark harness, projects retained
  frontend representations to fixed 32-dimensional vectors, and trains only the
  existing small `FlatWorldModel` and `UniversalCodec` seams.
- **Selected path**: TAESD, SNAC, and T5 were selected for the smallest formal path;
  the earlier TAEHV, DAC, and FLAN-T5 alternatives were not executed or evaluated.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Evidence**: [`docs/research/2026-07-15-mm001-small-real-multimodal-preflight-protocol.md`,
  `bench/multimodal_preflight/`,
  `bench/multimodal_preflight/results/MM-001/input-manifest.json`]
- **From staging**: O35

## A02: Linked structured epistemic-transition architecture
- **Design**: Make an epistemic transition the central auditable aggregate, implemented
  as links among immutable records rather than one overloaded tuple: pre-decision
  belief and information-set identity; an action-conditional prediction frozen before
  execution; the decision alternatives and decomposed expected value; intended and
  executed actions; real observations, feedback, provenance, time, and lineage; the
  post-evidence belief update; any persistent learned-configuration update; and a
  separate external evaluation.
- **Derived projections**: Compute prediction error or surprise, expected and realized
  information gain, durable calibrated knowledge gain, external goal utility, cost,
  risk, and constraint violations as distinct typed assessments. No universal
  backward-compatible epistemic scalar may stand in for all of them.
- **Decision rule**: Select admissible actions by expected external goal value plus
  expected decision-relevant information value minus resource cost and risk. A purely
  information-seeking agent is the special case in which the external goal term is
  absent; uncertainty reduction is not the definition of general reward.
- **Learning rule**: Store real experience independently of model-version-bound beliefs
  and predictions. Learning consumes declared experience views and emits a versioned
  update receipt. Imagined transitions and retrieved content keep separate lineage and
  cannot count as newly observed external experience.
- **Evaluation rule**: Collection, learning, behavioral improvement, and retention are
  separate claims. A frozen external evaluator measures them on held-out targets and
  resource-matched controls without mutating the agent.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Dependencies**: [G01, G02, G03, G04, N119]
- **Evidence**: [N118, N119, pending]
- **From staging**: O82
