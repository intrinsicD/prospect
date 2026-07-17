# Architecture

Prospect is an adaptive-agent research runtime whose central claim must be
demonstrable as a causal chain:

> the agent collected this experience, changed persistent state because of that
> experience, behaved better on held-out cases, and retained the gain after restart
> and interference.

The predictive world model remains the main mechanism for anticipating
consequences. It is no longer treated as a source of one scalar that can stand in
for every epistemic concept. Prediction, uncertainty, surprise, information gain,
decision value, learning, knowledge, and retention have different definitions and
different evidence.

ADR-0014 is the governing decision. The superseded P-series implementation and
tests were removed from the active tree; Git history preserves their research
record without retaining compatibility constraints.

## System state

- **World state** is the joint environment state and agent state.
- **History** is the actual sequence of world states. The agent generally cannot
  access it directly.
- **Observation** is evidence made available to the agent through an observation
  channel, with time and provenance.
- **Information set** is the evidence causally available at a declared cutoff.
- **Belief** is a versioned probability or uncertainty representation about a named
  target under one information set.
- **Latent state** is backend-specific internal state used to update beliefs or make
  predictions. It is neither automatically a world state nor canonical memory.
- **Memory** stores observations and experience; replay is a sampling index over
  that evidence, not the evidence itself.
- **Agent snapshot** is an immutable view of current belief, versions, resources,
  pending intentions, and latest learning receipt.

This distinction matters: an observation is real evidence; a latent is an internal
representation; an imagined outcome is model output with separate lineage.

## The linked lifecycle

The authoritative runtime is:

```text
snapshot -> assess alternatives -> decide -> execute -> observe
         -> store experience -> revise belief -> score prediction
         -> record epistemic transition -> explicitly learn -> evaluate
```

The main immutable records are:

| Record | What it establishes |
|---|---|
| `Evidence` / `Observation` | What became available, when, from whom, with what lineage |
| `Belief` | The agent's stance about one target under a specific information set and model version |
| `Prediction` | A pre-outcome distribution, target, horizon, model/representation/calibration versions, and qualified uncertainty estimates |
| `CandidateAssessment` | Goal utility, information value, acquisition/action cost, risk, soft penalty, and hard admissibility for one action |
| `DecisionRecord` | All assessed alternatives and the selected intended action |
| `ExecutedAction` | What the environment actually accepted or performed |
| `ExperienceEvent` | One canonical real step with run/task/episode/step identity, termination/truncation, discount, decision, execution, observation, and outcome |
| `BeliefUpdate` | The prior, exact experience, and posterior |
| `EpistemicTransition` | The action-time prediction, realized proper score, posterior change, and typed epistemic effects |
| `UpdateReceipt` | Which transitions a learner consumed and which persistent versions changed |
| `EvaluationRecord` | External held-out metrics, resource budget, snapshot identity, and whether updates were allowed |

Records are linked by stable identity and checked for time, agent, action, target,
evidence, and version consistency. There is deliberately no universal
`.epistemic` float.

## Uncertainty and evidence

The architecture keeps four commonly collapsed quantities separate:

1. **Predictive uncertainty** is an ex-ante property of a distribution.
2. **Surprise** is an ex-post proper score of a realized outcome under the immutable
   action-time distribution.
3. **Information gain** is a prior-to-posterior change about a named target.
4. **Value of information** is the expected improvement in downstream goal utility
   enabled by evidence, less acquisition and opportunity costs.

High observation entropy can be irrelevant noise. Lower physical-state entropy can
come from destroying the state rather than learning it. A posterior can become
more concentrated while becoming wrong. For those reasons, internal entropy
reduction is telemetry, not automatically knowledge, reward, or improvement.

A useful action maximizes the explicit decision objective subject to hard
constraints. Information value is one term; external task value, risk, and cost
remain independent terms.

## Learning and knowledge

Assimilation and learning are different operations:

- **Assimilation** updates the current belief with newly available evidence without
  changing the predictive model version.
- **Learning** changes persistent model, representation, policy, calibration,
  configuration, or knowledge state and returns an `UpdateReceipt`.

If learning changes a model or representation version, it must also provide a
linked resulting belief under the new version; a snapshot may not pair new model
weights with an old-model belief.

Prospect calls a result **knowledge** only within a declared scope when:

- a calibrated belief or policy is supported by identified evidence;
- it predicts or acts well on disjoint held-out cases;
- the claim survives the specified perturbation or interference; and
- the responsible persistent state survives checkpoint/restart.

Low internal uncertainty alone is not knowledge.

## Runtime and storage

`prospect.runtime.EpistemicAgent` is the active composition root. A
`DecisionRecord` is passed explicitly into observation handling; there is no hidden
“last prediction.” The runtime stores one canonical `ExperienceEvent` for each
environment step and derives learner-specific views afterward.

`InMemoryExperienceStore` and `EpistemicLedger` are append-only custody stores.
`TensorDictExperienceReplay` is an optional TorchRL sampling index and cannot
replace them. Imagined outcomes never enter the real-experience namespace.

Persistence uses a manifest over component-owned byte states. The coordinator
binds versions, media types, sizes, and hashes and performs atomic local
replacement. The caller must declare the resume boundary and supply every
stateful component; the generic coordinator cannot prove component completeness.
See `docs/runtime-substrate.md`. Exact mid-episode restoration is not claimed.

## Decision policy

The first transparent policy selects the highest-total admissible
`CandidateAssessment`:

```text
total =
    expected goal utility
  + expected decision-relevant information value
  - information acquisition cost
  - action/resource cost
  - expected risk
  - soft constraint penalty
```

Hard constraint violations are not made selectable by a sufficiently large
utility. Retrieval, external queries, tools, and experiments enter through this
same candidate-action interface; they may not invisibly overwrite a prediction.

## Independent evidence ladder

The E-series reports each row separately:

| Gate | Claim | Required comparison |
|---|---|---|
| E0 | Trace integrity | one real step equals one uniquely linked experience; no future or imagined evidence |
| E1 | Epistemic semantics | exact Bayes/EIG/EVSI oracle rejects irrelevant noise, destructive certainty, label artifacts, and future leakage |
| E2 | Collect | relevant evidence is acquired under a matched budget, with random/raw-entropy controls |
| E3 | Learn | declared training experience improves held-out proper score/calibration over frozen and shuffled/irrelevant controls |
| E4 | Improve | the updated frozen policy improves held-out external utility at equal evaluation budget |
| E5 | Retain | save/reload is equivalent and the gain remains after a prespecified delay/interference task |

Passing E3 does not imply E4. Reload parity alone does not imply E5. The complete
lifecycle statement is permitted only when every row passes an independently
audited protocol.

The current exact reference report does not pass this ladder. Its numeric
predicates execute, but E2/E3 are reference-only and E4/E5 are blocked because
the rows use different agents, the “learning” is task-local belief assimilation,
behavior is evaluated analytically, and the checkpoint omits canonical custody.

## Cutover boundary

The old flat modules, tests, evaluator registry, and regression ratchet are not
part of the active repository. A historical mechanism may return only through an
explicit new adapter that supplies every required identity, distribution,
calibration, version, and persistence fact. An adapter must fail rather than
invent missing provenance or pretend a scalar forecast is a qualified
distribution.

## Open engineering boundaries

- A neural/control backend must replace the exact finite reference without changing
  the lifecycle semantics.
- The learner protocol needs a transactional prepare/validate/commit boundary;
  the current `update` call cannot guarantee that backend state and the canonical
  receipt/state ledger advance atomically.
- Value-of-information estimates require their own calibration and adversarial
  controls.
- The current in-memory lifecycle journal exposes partial completion but cannot
  automatically continue or survive restart. Durable idempotent recovery is still
  required; atomic rollback is not a valid answer once real experience is stored.
- Runtime multi-operation sequences are not globally serialized; concurrent
  callers can race between preflight, append, and state application.
- Exact mid-episode resume requires environment, recurrent-belief, pending-action,
  side-effect, and RNG reconciliation.
- Continual learning must demonstrate both retention and plasticity across
  overlapping tasks; task-keyed independent tables are only a semantic reference.
- External benchmark results and strong published baselines are required before
  making a capability or novelty claim.
