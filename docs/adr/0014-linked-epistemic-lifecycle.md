# ADR-0014 — Linked epistemic transitions and independently evidenced lifecycle

**Status:** Accepted — core contracts implemented; capability evidence blocked
(`E0-001`)

**Supersedes:** ADR-0002, ADR-0007<br>
**Amends:** ADR-0001, ADR-0004, ADR-0006, ADR-0008

## Context
Prospect's first architecture made violation of expectation (VoE) one scalar
backbone for prediction failure, mastery, curiosity, option termination,
forgetting, and retrieval. That produced useful component experiments, but it also
collapsed quantities with different meanings:

- predictive uncertainty is a property of a forecast before an outcome;
- surprise is a proper score after the outcome;
- information gain is the change from prior to posterior belief;
- value of information is the goal-dependent utility of obtaining that
  information; and
- learning and retention are changes across model versions, not properties of one
  transition.

The P0–P14 gates are evidence about the legacy-v1 components and their composition.
They do not, by themselves, demonstrate that one running agent collected
experience, learned from that exact experience, improved its behavior because of
the learning, and retained the improvement. The current records also lack the
identity and version links needed to establish that causal chain independently.

Prospect should reuse mature collectors, replay stores, models, planners, and
checkpoint mechanisms where they fit. Its architectural contribution is the
explicit epistemic lifecycle that connects them and makes each scientific claim
auditable.

## Decision

### 1. Replace the single-signal contract with linked, typed records

The E-series architecture uses the following distinct records:

| Record | Meaning |
|---|---|
| `Experience` | Immutable raw environment fact: observation, action, reward, next observation, termination/truncation, episode/step/task/goal identity, and the decision that caused it. |
| `BeliefState` | A model-versioned posterior sufficient statistic after assimilating observations; it is not the replay storage format. |
| `Prediction` | The action-time predictive distribution, explicit epistemic/aleatoric uncertainty measure, horizon, model version, and calibration identity. |
| `Decision` | The selected action linked to its goal, prediction, task utility, information value, costs/risks, and model/policy versions. |
| `EpistemicTransition` | A link from the raw `Experience` to the before/after beliefs, exact action-time `Prediction`, realized proper score, and realized information gain. |

These records form a traceable graph through stable identifiers. The prediction is
captured when the action is chosen; it must not be recomputed after learning and
presented as the earlier expectation. Raw observations remain canonical so a new
representation can re-encode them. Latents, beliefs, calibration objects, and
knowledge indexed by them carry representation/model versions.

Imagined outcomes use a separate type and lineage. They never masquerade as raw
`Experience`.

### 2. Keep uncertainty, surprise, information gain, and information value separate

`Prediction` declares the forecast distribution and the measure and units of its
uncertainty. A scorer computes a named proper score from the realized outcome.
Prospect does not apportion a negative log-likelihood into "epistemic surprise" and
"aleatoric surprise" merely in proportion to two variance summaries.

Expected information gain measures anticipated posterior change. Goal-conditioned
value of information measures the expected improvement in decision utility after
obtaining information, less acquisition and opportunity costs. High epistemic
uncertainty alone therefore means neither "seek" nor "avoid."

Every candidate decision is evaluated through an explicit utility decomposition:
task value, information value, risk, resource cost, and total. A regulator may
change these terms or budgets, but it records the result in `Decision`; it must not
silently flip the sign of a mutable planner coefficient. Explore/exploit labels may
remain telemetry, not the governing semantic contract.

### 3. Treat collect, learn, improve, and retain as independent claims

The lifecycle evaluator reports four results, each with its own controls and
evidence:

| Claim | Required evidence |
|---|---|
| **Collect** | The runtime persists exactly one complete raw `Experience` for each environment step, with intact episode/step/decision links and declared provenance. |
| **Learn** | Updating on a declared training set improves a preregistered held-out proper prediction score and/or calibration metric against a frozen/no-update control. Training loss alone is insufficient. |
| **Improve** | The post-update decision policy improves matched, held-out behavioral utility over the frozen pre-update policy at equal evaluation budget. Better prediction alone is insufficient. |
| **Retain** | The demonstrated behavioral gain survives a checkpoint/restart equivalence test and remains above the pre-learning baseline after a prespecified delay or interference block. Immediate post-training performance alone is insufficient. |

Splits, seeds, budgets, model versions, calibration versions, and checkpoint
identities are recorded. Calibration fitting data is disjoint from the evidence
used to claim learning, improvement, or retention. A failure in one row is reported
as that row's result; another row cannot stand in for it. Prospect may make the full
lifecycle claim only when all four pass their preregistered criteria.

### 4. Make knowledge and calibration evidence-bearing

A knowledge claim links its scope, answer, provenance, supporting experience or
evaluation records, model/representation version, and calibration identity.
Calibration state records its method, fitted split, sample count, scope, and model
version; a model update invalidates it unless the calibration policy explicitly
updates and versions it.

Retrieval and tool use remain candidate actions, but uncertainty only nominates
them for evaluation. Their expected information value, trust, latency, monetary or
compute cost, and downstream utility are recorded in `Decision`. An automatic
prediction wrapper that invisibly overwrites a model forecast is a legacy-v1
mechanism, not the E-series target.

### 5. Persist the learning state, not only model weights

One checkpoint coordinator writes a manifest covering model and target-model
parameters, optimizers, replay state and sampler state, normalization, calibration,
regulator and knowledge state, counters, configuration/source identities, and all
relevant random-number-generator states. A checkpoint declares either exact
mid-episode or episode-boundary resume semantics. Episode-boundary restoration is
the initial requirement; exact resume cannot be claimed without environment and
recurrent-belief state.

Existing substrate-native state and replay serialization should be adapted rather
than reimplemented.

### 6. Establish a non-retroactive evidence boundary

P0–P14 and their recorded gate results are historical **legacy-v1 evidence** under
the contracts and environments that produced them. They do not satisfy or
partially pre-pass the E-series lifecycle claims. New implementation and formal
evidence begin at `E0-001`.

The 2026-07-17 cutover removed the legacy implementation, tests, evaluator
registry, and ratchet from the active tree after dependency isolation was
confirmed. Git history and authored research narratives preserve the record;
compatibility code is not retained merely to reproduce obsolete contracts.

## Consequences
- (+) The exact experience, expectation, update, decision, and checkpoint involved
  in an improvement can be traced across versions.
- (+) Prospect can distinguish "the model learned" from "the agent behaved better"
  and "the improvement persisted."
- (+) Existing learning and planning systems can be swapped behind stable adapters;
  Prospect-owned work is concentrated in epistemic regulation and evidence
  linkage. This is an ownership boundary, not a novelty claim.
- (+) Surprise and uncertainty can use suitable distribution families and proper
  scores instead of a hard-coded diagonal-Gaussian value object.
- (−) More identities, version checks, held-out splits, and checkpoint state are
  required. The lifecycle experiment costs more than a training-loss check.
- (−) Goal-conditioned information value is an estimator and can itself be wrong;
  it requires calibration, ablation, and negative controls.
- (−) Legacy-v1 gates cannot be cited as proof of the new lifecycle claim, even
  where a component name is unchanged.
- No new capability is claimed by accepting or partially implementing this ADR.
  The exact reference diagnostics support semantics and narrow plumbing only;
  linked model learning, executed improvement, and retention evidence remain
  pending in `E0-001`.

## Supersession details
- ADR-0002's requirements to predict distributions, distinguish reducible from
  irreducible uncertainty, and use proper scoring remain useful mechanisms.
  ADR-0014 supersedes the claim that one VoE scalar is the semantic controller for
  all jobs and supersedes variance-share "surprise decomposition."
- ADR-0007's binary seek/avoid sign arbitration is superseded. The E-series uses
  explicit goal-conditioned information value, risk, and cost in each decision.
