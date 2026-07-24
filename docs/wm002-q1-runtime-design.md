# WM-002 Q1 authoritative-runtime design

Status: **Q0 is independently accepted for protocol completeness only and was
rebound on 2026-07-24 after two bound sources changed. The Q1 runtime and
independent auditor are implemented. The two blockers that reopened result-free
qualification — causal noninterference and complete-orchestration coverage —
are now closed with negative controls and an end-to-end rehearsal. An
adversarial [prospective review](wm002-q1-prospective-review.md) fixed two
further defects and found nothing blocking, but it is self-review: the reviewer
wrote the code, and the entry gate cannot verify reviewer independence. A second
non-author review, then a regenerated review artifact and entry qualification
against the final bytes, remain required. Q1 execution authorization remains
false, and no Q1 outcome or capability evidence exists.**

This document describes the implemented authoritative runtime for the
five-action hidden-actuator qualification fixture. Successor protocol
`0.3.0-q1` fixes the selection, cost, seed-privacy, restore, watchdog,
single-attempt, publication, schema, and audit semantics. The sole
claim-ineligible Q1 run is permitted only after `execution_authorized` is set on
the final protocol bytes, a result-free independent review binds the exact
selected-source manifest, and the canonical entry report passes before any
private draw.

## Purpose and evidence boundary

The runtime must test one narrow causal chain:

```text
one immutable prior
  -> five assessed acquisition candidates
  -> one selected and executed action
  -> one canonical real experience and epistemic transition
  -> one transactional persistent posterior update
  -> one pre-terminal checkpoint
  -> one terminal assessment and executed outcome
  -> one fresh-process restoration and identical terminal replay
```

Passing this chain would qualify the known-model value-of-information
integration. It would not show learned uncertainty, general active learning,
transfer, continual-learning scale, multimodal competence, or formal WM-002
evidence.

Each episode starts from an independent symmetric prior. Posterior state must
not carry between episodes.

## Existing components to reuse

Q1 should use Prospect's existing authoritative path rather than create a
second collector:

- `prospect.runtime.EpistemicAgent.interact`, `observe`, and `learn`;
- `prospect.runtime.AgentState`;
- `prospect.storage.InMemoryExperienceStore`;
- `prospect.storage.EpistemicLedger`;
- `prospect.runtime.ModelState`, `PreparedLearningUpdate`,
  `VersionedModelOwner`, and the existing three-custodian transactional commit;
- `prospect.decision.CounterIdentitySource`;
- `prospect.storage.CheckpointCoordinator` and its component digest checks; and
- the explicit allowlisted domain-graph codec in
  `prospect.storage.domain_graph`.

The experiment-neutral domain-graph codec was extracted into generic storage;
WM-001 retains a compatibility re-export. Q1 does not duplicate it and does not
use pickle, arbitrary class lookup, or generic dataclass deserialization.

`MaxValuePolicy` is suitable for the Prospect acquisition arm and the terminal
decision only if the implementation supplies the protocol's explicit tie
order. A small arm-aware policy is still needed for the controls, because raw
entropy, EIG-only, shuffled information, and uniform random intentionally use
different selection scores.

The existing `CategoricalScorer` should not be used unchanged. It always links
the score to outcome evidence, while an acquisition prediction is resolved by
the observed acquisition symbol. Q1 needs a phase-aware scorer that links the
acquisition score to observation evidence and the terminal-success score to
outcome evidence.

## New bench-only components

The minimum new implementation belongs under
`bench/active_acquisition/`:

- `runtime_lane.py`
  - `HiddenActuatorCandidateAssessor`
  - `CandidateDiagnosticRow`
  - `ArmDecisionPolicy`
  - `TerminalCandidateAssessor`
  - `HiddenActuatorAcquisitionEnvironment`
  - `HiddenActuatorTerminalEnvironment`
  - `NoOpObservationAssimilator`
  - `AcquisitionTerminalScorer`
  - `NoOpAssimilationEffect`
  - `ExactPosteriorTransactionalLearner`
  - the per-episode runtime composition function
- `seeding.py`
  - the balanced hidden-regime schedule
  - semantic keyed potential outcomes
  - uniform-policy keys
- `checkpoint.py`
  - the exact Q1 component contract and safe codecs
- `q1.py`
  - the streaming producer, batched restore launcher, and result writer
- `restore_worker.py`
  - the fresh-interpreter checkpoint reader and terminal replay
- `attempt.py`
  - the exact run identity and fixed experiment-global durable one-shot
    tombstone
- `q1_qualification.py`
  - the result-free entry gate, implementation manifest, resource probe, and
    prospective-review binding
- `q1_audit.py` and `q1_audit_privacy.py`
  - the independent streaming semantic auditor, exact prospective-review
    revalidation and entry binding, and global private-prefix scanner
- `schemas/q1-*.schema.json`
  - strict entry, review, trace, checkpoint, aggregate, and audit contracts

The existing `problem.py`, `policies.py`, `oracle.py`, `qualification.py`, and
`run.py` own Q0. Q1 must consume a separately audited canonical Q0 report digest;
it must not reimplement or silently reinterpret that result inside the runtime
lane.
The independent auditor must live in a separate module and must not import
producer gate booleans or producer aggregates.

The selected-source manifest is built with descriptor-relative, no-follow
opens, bounded reads, and before/after descriptor and path-identity checks.
This binds the source bytes observed during manifest construction. It does not
bind code already loaded into an interpreter and does not defend against a
malicious coordinated same-account writer capable of manipulating paths or
metadata.

## Candidate identities and order

Q1 has exactly five acquisition candidates:

| Ordinal | Canonical semantic ID | Existing problem value |
|---:|---|---|
| 0 | `skip` | `SKIP` |
| 1 | `weak` | `WEAK_POSITIVE` |
| 2 | `strong` | `STRONG_POSITIVE` |
| 3 | `overpowered` | `OVERPOWERED_POSITIVE` |
| 4 | `nuisance` | `NUISANCE_SCAN` |

Negative signed pulses remain Q0 sign-label/permutation probes. They are not
additional Q1 candidates.

The domain `Action` should carry a stable, phase-qualified ID such as
`acquisition:00:skip` and the semantic ID in its parameters. The selection
implementation must use the declared ordinal directly. It must not rely on
the incidental lexical ordering of existing enum values or generated record
identities.

The terminal candidates are ordered:

1. `+1`
2. `-1`

An exact tie at the symmetric prior therefore selects `+1`. This rule must be
used by the live path, restored path, Q0 oracle, producer analyzer, and
independent auditor.

## Truthful assessment and control selection

`HiddenActuatorCandidateAssessor` receives one frozen `AgentSnapshot`, verifies
that its model version and posterior agree with the live
`VersionedModelOwner`, and emits all five linked `CandidateAssessment`
records.

For the real external-return decomposition:

```text
Utility.expected_value
    = prior-optimal terminal value
    + expected immediate acquisition payoff

InformationValue.expected_reduction
    = expected decision value of the observation

InformationValue.expected_cost
    = information-acquisition cost

CandidateAssessment.expected_action_cost
    = physical action cost

CandidateAssessment.total_value
    = expected net episode return
```

The two costs are distinct and are each subtracted once:

| Action | Physical action cost | Information-acquisition cost |
|---|---:|---:|
| `skip` | 0.00 | 0.00 |
| `weak` | 0.53 | 0.00 |
| `strong` | 0.58 | 0.00 |
| `overpowered` | 0.95 | 0.00 |
| `nuisance` | 0.00 | 0.01 |

The control objectives must not be forced into
`CandidateAssessment.total_value`. Raw entropy is measured in nats, EIG is
measured in nats, and uniform random has no scalar value decomposition.
Relabelling any of them as external utility would violate the core record
semantics.

Instead, the assessor also creates an immutable bench-only row linked by
`assessment_id`:

```text
CandidateDiagnosticRow
  assessment_id
  semantic_action
  expected_episode_value
  expected_immediate_payoff
  expected_terminal_value
  expected_decision_value
  expected_information_gain_nats
  raw_observation_entropy_nats
  physical_action_cost
  information_acquisition_cost
  arm_selection_score
  selection_unit
```

`ArmDecisionPolicy` records and digests all five rows, selects using the
declared arm score, and constructs the normal linked `DecisionRecord`. The
core `DecisionRecord` invariant does not require the selected assessment's
external-return total to be maximal, so a control can remain semantically
honest.

The selection functions are:

- Prospect: expected net episode return;
- exact oracle: independent `Fraction` expected net episode return;
- goal-only: prior terminal value plus expected immediate payoff minus both
  costs, with EVSI set to zero;
- raw entropy: acquisition-observation entropy only;
- EIG-only: information gain about the hidden regime only;
- shuffled information: true utility and costs plus the prospectively
  permuted EVSI score; and
- uniform random: the keyed uniform-policy draw over the five ordinals.

The shuffled arm changes selection scores only. Its environment and posterior
learner use the true likelihood of the action that was actually executed.
Changing the shuffled learner's likelihood would introduce a second causal
intervention and invalidate the intended control.

## Agent-visible environment records

The trusted producer harness privately owns the episode's hidden regime and
keyed potential-outcome draws. It computes only the selected action's visible
symbol before constructing `HiddenActuatorAcquisitionEnvironment`. That
single-use gateway receives the visible symbol, accepts only an
`IntendedAction`, and returns:

- an `ExecutedAction`;
- an observation payload containing only phase, executed semantic action, and
  observed symbol; and
- an outcome payload containing immediate task payoff, physical action cost,
  information-acquisition cost, and their reconstructed net reward.

An illustrative allowed acquisition payload is:

```json
{
  "phase": "acquisition",
  "semantic_action": "strong",
  "observed_symbol": 1
}
```

An illustrative allowed acquisition outcome is:

```json
{
  "task_payoff": 1.0,
  "physical_action_cost": 0.58,
  "information_acquisition_cost": 0.0,
  "net_reward": 0.42
}
```

The trusted producer or restore harness retains the same episode regime and
terminal-action-keyed draw. It supplies only the selected terminal decision's
success bit to `HiddenActuatorTerminalEnvironment`, which returns a neutral
terminal observation and an outcome containing only the executed terminal
action, success bit, and terminal reward.

Neither environment may expose the hidden regime, its schedule position, its
derivation key, any unchosen potential outcome, or any future outcome.

## No-op assimilation before persistent learning

`EpistemicAgent.observe` necessarily calls a `BeliefUpdater` before
`EpistemicAgent.learn`. If that updater performs the Bayesian regime update,
terminal behavior could change before the declared persistent model update.
K3 would then fail to identify the transactional update as the cause.

`NoOpObservationAssimilator` must therefore:

- append the executed observation to the prior `InformationSet`;
- create new information-set, memory, posterior-belief, and update identities;
- retain the prior categorical regime probabilities exactly;
- retain the prior model and representation versions exactly; and
- never read the terminal result, hidden regime, unchosen outcomes, or another
  arm's state.

This is not an ignored observation. It is the canonical registration step
that makes the real transition available to the learner. It deliberately does
not perform persistent inference.

`AcquisitionTerminalScorer` scores the immutable acquisition prediction
against the realized observation symbol using a categorical log score. For
the terminal phase, it scores the predicted success distribution against the
realized success outcome.

`NoOpAssimilationEffect` records a zero internal entropy change with an
explicit assimilation-only measure. The persistent posterior entropy change
is recorded in the learning receipt, because it occurs during learning rather
than during the no-op assimilation.

## Exact transactional posterior learner

The model owner starts each episode from canonical exact bytes equivalent to:

```json
{
  "evidence_count": 0,
  "last_experience_id": null,
  "last_transition_id": null,
  "likelihood_version": "wm002-hidden-actuator-true-v1",
  "posterior_direct": {
    "denominator": 2,
    "numerator": 1
  },
  "schema": "prospect.wm002.posterior-model.v1"
}
```

The actual encoding must use sorted-key, finite, canonical UTF-8 JSON without
an executable deserializer. Probabilities are stored as reduced numerator and
denominator integers. A float pair may be derived for the domain `Belief`, but
the authoritative model bytes remain rational.

`ExactPosteriorTransactionalLearner.prepare` must:

1. require exactly one transition;
2. require that it is the canonical transition from the same episode, arm,
   and acquisition step;
3. verify that its `ExperienceEvent`, `DecisionRecord`, `ExecutedAction`,
   observation, and predecessor model version are mutually linked;
4. decode only the immutable predecessor model bytes;
5. read only the executed semantic action and observed symbol from the
   transition;
6. update with that executed action's true likelihood;
7. increment the evidence count and store the consumed experience and
   transition identities;
8. emit different canonical candidate bytes and a digest-derived model
   version;
9. rebase the transition's no-op posterior into a new `resulting_belief`
   containing the exact Bayesian posterior and candidate model version; and
10. return an `UpdateReceipt` naming exactly the supplied transition.

The candidate payload is:

```json
{
  "evidence_count": 1,
  "last_experience_id": "<canonical acquisition experience ID>",
  "last_transition_id": "<canonical acquisition transition ID>",
  "likelihood_version": "wm002-hidden-actuator-true-v1",
  "posterior_direct": {
    "denominator": "<reduced positive denominator>",
    "numerator": "<reduced nonnegative numerator>"
  },
  "schema": "prospect.wm002.posterior-model.v1"
}
```

The version should be `wm002-model-sha256:<payload-sha256>` and the
configuration version should bind the same digest. Policy and representation
versions do not change.

The `VersionedModelOwner` validator must reject a noncanonical payload, wrong
schema, unreduced fraction, posterior outside `[0,1]`, nonmonotonic evidence
count, or undeclared likelihood version. The existing runtime then validates
the source digest, reserves the model swap, and commits ledger, agent state,
and model owner as one rollback-safe transaction.

Evidence count and ancestry make the candidate bytes differ even when `skip`
or `nuisance` leaves the posterior probability unchanged. For the Prospect
strong pulse, both the bytes and posterior probability must change.

Receipt metrics should include:

- posterior probability before and after;
- entropy before and after;
- entropy reduction;
- consumed transition count; and
- evidence count before and after.

String digests and transition identities remain explicit receipt/trace fields,
not floating-point metrics.

## Terminal decision and observation

After the transactional commit, `TerminalCandidateAssessor` must verify:

- the snapshot model version equals the `VersionedModelOwner` version;
- the snapshot posterior equals the rational posterior decoded from model
  bytes; and
- the snapshot's latest receipt is the acquisition receipt.

It emits two terminal candidate assessments using the posterior-implied
success probabilities. Both have zero information value and zero acquisition
cost. The terminal policy selects maximum expected success with the explicit
`+1` tie rule.

The terminal action is executed and observed through
`EpistemicAgent.interact`, yielding a second canonical experience and
transition. `EpistemicAgent.learn` is not called for this transition.
Therefore each episode has:

- two decisions;
- two executions;
- two experiences;
- two epistemic transitions;
- one acquisition learning invocation; and
- zero terminal learning invocations.

The episode return is reconstructed only from primitive executed fields:

```text
acquisition task payoff
  - physical action cost
  - information-acquisition cost
  + terminal success reward
```

No stored aggregate may substitute for that reconstruction.

## Pre-terminal checkpoint

The checkpoint boundary is immediately after the acquisition receipt commits
and before any terminal assessment. Each logical episode checkpoint contains
all and only:

1. `posterior_model`
   - exact `VersionedModelOwner` payload, version, and predecessor digest;
2. `domain_custody`
   - an allowlisted domain graph rooted at the pre-terminal `AgentSnapshot`,
     acquisition `ExperienceEvent`, `EpistemicTransition`, and
     `UpdateReceipt`;
3. `identity_counter`
   - `CounterIdentitySource.checkpoint_bytes()`;
4. `episode_accumulator`
   - executed acquisition payoff and the two charged cost components; and
5. `qualification_binding`
   - protocol version/digest, implementation binding, and accepted Q0 report
     digest.

The hidden regime, any seed or secret salt, unchosen potential outcomes, and
terminal draw are forbidden checkpoint components.

On load, a fresh runtime graph must be constructed in this order:

1. verify the aggregate payload, every component digest, the manifest, and the
   external qualification binding;
2. decode and validate the model, accumulator, and allowlisted domain roots
   before constructing mutable custodians;
3. restore the identity counter and construct `VersionedModelOwner` from the
   verified model bytes;
4. append the decoded acquisition experience to a fresh canonical experience
   store;
5. relink and append the canonical transition and acquisition receipt to a
   fresh ledger;
6. construct `AgentState` from the relinked receipt belief and restored
   snapshot; and
7. construct fresh assessor, policy, scorer, effect assessor, and learner
   instances.

The restored terminal execution receives the same private keyed terminal
potential outcome from the harness. That draw is not restored into or exposed
through the agent.

Live and restored paths must compare:

- component and aggregate digests;
- model bytes, digest, model version, and configuration version;
- posterior fraction;
- acquisition experience, transition, and receipt ancestry;
- identity-counter state before terminal assessment;
- all terminal candidate rows and their canonical digest;
- selected terminal action;
- executed terminal success; and
- reconstructed episode return.

Identity-counter restoration should make live and restored record identities
equal as well as their semantics.

## Hidden-state boundary

The private harness may hold:

- the realized hidden regime;
- the private balanced schedule or its secret key;
- all action-conditional potential-outcome draws;
- all unchosen outcomes;
- terminal draws for both terminal decisions; and
- independent-oracle working state.

The agent may hold:

- the declared symmetric prior;
- public action likelihood tables and costs;
- the five candidate descriptions;
- the one executed action;
- the one observed acquisition symbol;
- its inferred posterior distribution;
- executed payoff and cost components;
- the executed terminal success; and
- canonical lineage, version, and digest metadata.

The implementation must test the actual serialized shapes, not only inspect
dataclass field names. Before Q1 it needs:

- a recursive allowlist for every observation, outcome, action parameter,
  model payload, domain-graph root, checkpoint component, and public result
  row;
- a test that substitutes recognizable sentinels for the private regime,
  schedule key, and unchosen draws and verifies that no sentinel reaches an
  agent-visible serialization;
- a test that the learner produces the same candidate from the same visible
  transition when the inaccessible counterfactual table is changed; and
- a test that acquisition decisions and updates are unchanged when future
  terminal draws are changed.

Protocol `0.3.0-q1` uses the private-salt option. The raw salt is an exact-0600
regular file supplied only to the trusted private execution lane (parent,
producer workers, and restore workers) and separately to the independent
auditor. Public artifacts contain its SHA-256 commitment, never the salt or
private HMAC outputs. This is local procedural custody, not proof against the
machine or same-account owner.

## Paired seed and potential-outcome schedule

Each master contains exactly 512 episodes from each hidden regime. A private
key determines only their order. Stochastic outcomes are derived from semantic
keys rather than one order-consumed RNG:

```text
(protocol version, private salt, namespace, master, episode,
 semantic action or terminal decision, draw role)
```

The four private HMAC namespaces remain distinct for:

- hidden-regime order;
- pulse observations;
- nuisance observations;
- terminal success.

Uniform-policy selection is instead a public SHA-256 rule over protocol,
master, and episode. Identity counters start from public constant zero in each
fresh episode namespace. There is no private restore-order namespace.

The terminal draw is keyed by terminal decision, not by the arm that selected
it. The same master/episode/terminal-decision tuple therefore yields the same
potential outcome for all arms.

Producer and auditor record commitments and semantic-key digests, not the
private regime or secret derivation material, in public rows. Private audit
material must be a separately permissioned sidecar.

## Making the full Q1 budget tractable

Four masters and 1,024 episodes produce 4,096 episodes per arm. Across seven
arms, the full budget is 28,672 episodes, 57,344 environment steps, 57,344
transitions, and 28,672 transactional updates. The
finite Bayesian arithmetic and tiny model payload are cheap. Per-episode
process startup and forced durable file synchronization are the dominant
avoidable costs.

The recommended execution plan is:

- run one producer worker per master, with deterministic arm and episode
  ordering inside the worker;
- write compact canonical JSONL trace rows and bounded checkpoint frames as
  episodes finish instead of retaining every object graph in memory;
- precompute static numeric diagnostic matrices while still creating fresh
  linked domain records for every decision;
- use exact rational arithmetic for the model update and Q0/audit, without
  repeatedly invoking the full independent enumeration inside every producer
  assessment;
- merge worker outputs only by declared semantic keys;
- start one new restore interpreter per `(master, arm)`, for 28 fresh
  processes rather than 28,672; and
- inside each restore worker, construct a new component graph for each of its
  1,024 independently framed checkpoints and release it immediately after the
  terminal replay.

The implementation runs four producer processes and 28 lane restore processes,
with at most four restores live concurrently. Producer, per-restore-child, and
restore-stage watchdogs are 3,600, 900, and 7,200 seconds, with a 10-second
terminate-before-kill grace. Every parent-side error must terminate, kill if
needed, and reap all started children before a failed marker is finalized; an
unproven child exit leaves the marker permanently `started`.

The execution root and external attempt registry are exact-0700 directories;
the salt, attempt marker, and private sidecar are exact-0600 files. Every
proposed run identity competes for the same fixed
`wm002-q1.attempt.json` tombstone; any existing started, failed, or completed
marker rejects another attempt. Before the sole run, entry qualification
exercises Linux
`renameat2(RENAME_NOREPLACE)` on the actual execution filesystem, including
successful publication, collision refusal, and cleanup. Worker trees are
removed only after all six canonical artifacts are regular files, then the
validated directory is fsynced and published atomically without replacement.
This makes Q1 intentionally Linux-specific at this qualification boundary.

To avoid tens of thousands of fsynced ZIP files, extend the generic checkpoint
codec with deterministic in-memory `dump_bytes`/`load_bytes` operations and
frame those complete logical bundles in a bounded sidecar. The existing
`save`/`load` path can remain the durable-file wrapper. Every episode still has
its own complete manifest, component set, aggregate digest, and independent
restore operation.

The protocol should prospectively clarify that “fresh process” means:

> terminal restoration occurs in an interpreter process whose PID differs
> from the producer, and every checkpoint initializes a fresh component graph;
> one fresh interpreter may verify a bounded batch.

Literal one-new-OS-process-per-episode adds no relevant semantic coverage and
should be relaxed before outcomes are observed.

## Q0 binding and Q1 entry condition

Q0 must produce a canonical, independently checkable report containing:

- the protocol version and digest;
- the full reachable rational semantic matrix;
- the matching floating Prospect matrix;
- all selector identities;
- both explicit tie rules;
- sign-label permutation checks;
- exact cost decomposition checks;
- Q0 field-schema and emitted-report leakage checks, with actual serialized
  runtime-value checks reserved for Q1 entry; and
- a terminal `passed` result.

The Q1 entry point must require the Q0 report path and expected SHA-256. It
must fail before constructing an environment or deriving a Q1 draw if:

- the report is missing or noncanonical;
- its digest differs;
- any Q0 check failed;
- its embedded Q0 protocol and implementation digests differ from the accepted
  Q0 identities;
- the implementation binding differs; or
- a formal authority or formal seed set is present.

The accepted Q0 digest is embedded in every Q1 checkpoint and result partition.
Q0 and Q1 remain claim-ineligible regardless of outcome.

## Implementation and blocker closure

Most pre-implementation blockers have concrete result-free implementation and
test coverage. Adversarial traceability review has nevertheless reopened
qualification, which is not green until every newly found blocker closes. The
inventory below records implemented mechanisms and known unresolved gaps; it
is not a readiness disposition:

1. **Shuffled-control confound — closed.** Only selection diagnostics are
   shuffled; learning uses the true likelihood of the executed action.
2. **Cost ambiguity — closed.** Physical and information-acquisition costs are
   separate fields and each is charged exactly once.
3. **Candidate and tie ambiguity — closed.** Five acquisition ordinals and the
   terminal `+1` tie rule are machine-bound and independently recomputed.
4. **Hidden schedule — closed locally.** Four private HMAC namespaces use a
   committed exact-0600 salt; public uniform selection is salt-independent.
5. **Serialized leakage contract — closed with negative controls.** Strict
   recursive schemas, actual-sample scans, checkpoint private-key scans, and the
   auditor's global private-prefix scan cover emitted shapes and known private
   byte prefixes. The causal probe now varies the hidden sign and every private
   HMAC field through the exercised runtime while holding the visible executed
   transition fixed, and requires byte-identical acquisition probes, checkpoint
   payloads, checkpoint indices, and public trace projections. Five mutation
   controls establish that the probe can fail: leaking the hidden sign into the
   public acquisition or checkpoint section, making the checkpoint payload
   depend on it, deriving both arms from unvaried private material, replaying
   the reference material where the variant belongs, and flattening the
   terminal sensitivity control are each detected.
6. **Q0 authority — closed.** The accepted canonical Q0 report, protocol, and
   implementation digests are hard entry and audit bindings.
7. **Restore interpretation — closed.** Each `(master, arm)` uses a distinct
   restore PID and every episode constructs a fresh component graph.
8. **Schemas and independent auditor — implemented.** Entry, review, six
   producer artifacts, and audit output are strict; producer summaries are
   non-authoritative. The auditor descriptor-safely reopens the prospective
   review, validates its canonical schema, exact protocol and implementation
   digests, method, assurance boundary, scope, selected-source count,
   zero-interaction/private-draw counters, false claim/formal flags, passing
   disposition, and finding types, then binds its exact digest to entry.
9. **Protocol and target identity — implemented with a local trust limit.**
   Protocol `0.3.0-q1`, the `q1v3` namespaces, one composite target, exact
   source origins, and the sorted implementation manifest are bound.
   Descriptor-stable reads bind observed source bytes but not already-loaded
   bytecode and do not resist a malicious coordinated same-account writer.
10. **Lifecycle and resource safety — closed by the orchestration rehearsal.**
    The fixed experiment-global `wm002-q1.attempt.json` tombstone is claimed
    before a private schedule exists and cannot be bypassed with another run
    identity; watchdog cleanup, exact permissions, conservative disk sizing,
    actual-filesystem no-replace publication, durable completion, and failure
    preservation have focused tests. The complete four-producer,
    28-restore-lane, merge, validation, publication, and completed-marker
    sequence now runs end to end in the rehearsal mode described below, with
    real authenticated children, a genuine entry qualification, and no
    monkeypatched runtime.

## Rehearsal mode

The entry gate requires `execution_authorized: true`, so before this change the
whole-orchestration path could only be approximated with stubs. Rehearsal mode
removes that gap without touching the authorization boundary:

- `execution_authorized` selects the mode. `true` permits only the sole
  full-budget production attempt; `false` permits only a rehearsal. Callers must
  state the mode they intend, so authorization can never silently downgrade a
  production run to a rehearsal or promote a rehearsal to the real attempt, and
  the two modes are never simultaneously reachable.
- Children never trust the parent for the mode. Each derives it from the
  canonical protocol bytes it reads itself, and the entry report it must match
  already binds those bytes by digest.
- A rehearsal lane holds two episodes instead of 1,024. `bench/active_acquisition/rehearsal.py`
  regenerates the accepted Q0 report, mints a throwaway salt, writes a
  self-declared rehearsal-only prospective review, runs a genuine entry
  qualification, and executes the identical orchestration.

A rehearsal is mechanics coverage and never evidence. Its hidden-regime
schedule is unbalanced at two episodes, its arm means are statistically
meaningless, its aggregate carries a separate rehearsal schema whose budget
constants are exact, and its parent process loads a module outside the
selected-source closure. Running the independent auditor over a rehearsal fails
Q1-K0 on `execution_authorized`, Q1-K1, Q1-K3, Q1-K4, and Q1-K5; that rejection
is itself a test.

The rehearsal does execute real environment interactions on the fixture under
its own run identity and salt. It is not a Q1 draw: the Q1 attempt is a
different run identity under different protocol bytes, and the rehearsal budget
is 1/512 of one arm's, far too small to inform any threshold or interval.

The last completed result-free checkpoint before subsequent integration
changes recorded:

- 339 combined active-acquisition tests passing;
- Ruff clean;
- strict Mypy clean across 62 configured source files, with the Q1-scoped
  invocation covering 34 files;
- ten protocol/schema JSON documents valid;
- a 15-interaction synthetic rehearsal completing under 30 seconds with all
  six artifact schemas and fresh-process restore covered; and
- a prior adversarial code-level review whose readiness disposition has been
  superseded by the reopened qualification.

These are historical result-free test facts only. They are not readiness, Q1
outcomes, or evidence that the agent improves. Before the execution sequence
begins, every newly found blocker must close and the full result-free
qualification and independent prospective review must be rerun against the
final bytes. Only then is the remaining sequence permitted:

1. enable `execution_authorized` on the final successor-protocol bytes while
   retaining `claim_eligible: false`, `formal_authorized: false`, and no formal
   seeds;
2. obtain a canonical, result-free independent prospective review against the
   exact final implementation manifest;
3. generate one fresh exact-0600 salt, exact-0700 execution root and registry,
   then require the canonical entry report to pass with zero Q1 interactions
   and private draws;
4. claim and execute the sole full-budget Q1 attempt; and
5. run the separate independent auditor before interpreting any result.

Any source, schema, protocol, review, salt, Q0, directory-binding, or entry
change after step 2 invalidates the review and entry. Q1 remains permanently
claim-ineligible. Single-attempt integrity, salt custody, source immutability,
and process identity are local procedural assurances; external attestation is
future work.
