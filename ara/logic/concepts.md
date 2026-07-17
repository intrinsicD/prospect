# Concepts

## G01: Agent state, latent state, belief, history, and memory are distinct
- **Definition**: Agent state is the complete agent-side state relevant to future
  behavior. A latent state is an internal representational component of that state.
  A belief is the agent's structured epistemic stance over possible world values or
  propositions and may be parameterized by a latent state. World history is the
  actual world-state trajectory; observation or interaction history is the evidence
  made available to the agent; memory is the persistent, possibly selective
  transformation of accessible history that the agent retains.
- **Special case**: A perfect append-only observation memory makes stored memory
  identical to observation history, but the general ontology must not assume this.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Dependencies**: [N115, N116]
- **From staging**: O75

## G02: The foundational agent vocabulary is layered and operation-relative
- **Definition**: Keep ontic and dynamical terms, epistemic terms, representational
  terms, and persistence or adaptation terms in separate layers. Observation is an
  information event crossing the agent boundary, not necessarily an encoder input;
  context is the operation-relative information supplied to a particular process;
  latent state is internal representation rather than the whole agent; belief is an
  epistemic stance rather than hidden ground truth; and memory is retained
  information rather than history itself.
- **Qualification rule**: State terms identify their owner and temporal phase;
  history terms identify which sequence they contain; context identifies its
  consumer; and belief, uncertainty, and prediction identify their target,
  conditioning information, and horizon.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Dependencies**: [N114, N115, N116]
- **From staging**: O77

## G03: Operational closure requires normative, adaptive, and lifecycle primitives
- **Definition**: A descriptive ontology of world, observation, agent state, belief,
  representation, history, and memory does not yet define an adaptive agent. An
  operational specification also identifies transition and time semantics; goals,
  preferences, costs, risks, and constraints; decisions and execution; persistent
  learned configuration; experience-dependent update and credit; lifecycle, identity,
  reset, and resource boundaries; and an external evaluator.
- **Separation rule**: These primitives define roles and proof obligations. A planner,
  neural network, replay algorithm, probabilistic formalism, or training backend is
  one possible implementation and is not part of the foundational definition.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Dependencies**: [N114, N117, N119]
- **From staging**: O78

## G04: Uncertainty, error, surprise, information gain, and knowledge are distinct
- **Definition**: Belief is an information-conditioned stance over named alternatives.
  Uncertainty is an ex-ante functional of unresolved predictive or decision risk.
  Expectation is a summary of belief; discrepancy with a realized outcome is ex-post
  prediction error; surprise is a proper score of that outcome under the full
  prediction. Information gain is evidence-caused belief change or expected reducible
  risk reduction. Knowledge is retained, scoped, evidence-grounded, externally
  calibrated epistemic competence, not merely a narrow internal distribution.
- **Qualification rule**: Every belief, prediction, uncertainty, information-gain, or
  knowledge assessment identifies its target, conditioning information, horizon,
  estimator or score, units, and relevant model or representation version.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Dependencies**: [N118, N119]
- **From staging**: O81
