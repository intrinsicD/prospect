# Architecture

Prospect is an adaptive-agent runtime whose central claim must be
demonstrable as a causal chain:

> the agent collected this experience, changed persistent state because of that
> experience, behaved better on held-out cases, and retained the gain after restart
> and interference.

The predictive world model remains the main mechanism for anticipating
consequences. It is no longer treated as a source of one scalar that can stand in
for every epistemic concept. Prediction, uncertainty, surprise, information gain,
decision value, learning, knowledge, and retention have different definitions and
different evidence.

This document is the canonical system definition. Code, tests, and experiments
must implement or explicitly challenge it.

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

Persistent model learning uses explicit ownership. A transactional learner
prepares candidate bytes from an immutable model snapshot. The runtime validates
experience ancestry, receipt versions, and predecessor/candidate digests, then
commits the update ledger, agent learning state, and owned model inside one lock
order. If an in-process exception occurs during commit, all three participants
are restored before the exception escapes. This provides in-process
failure-atomicity; it is not durable crash recovery.

Persistence uses a manifest over component-owned byte states. The coordinator
binds versions, media types, sizes, and hashes and performs atomic local
replacement. The caller must declare the resume boundary and supply every
stateful component; the generic coordinator cannot prove component completeness.
WM-001 makes that declaration concrete at an episode boundary with 15 required
model, optimizer, ledger, experience, replay, runtime, scaling, and RNG
components, then verifies behavior in a fresh process. Exact mid-episode
restoration is not claimed: it additionally requires the environment, recurrent
belief, pending action, external side effects, and RNG state at that boundary.

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

## Evidence ladder

Each experiment reports each row separately:

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

Exact finite references may validate individual semantics but do not establish
the complete ladder unless the same agent, learner, experience ancestry,
executed held-out behavior, interference path, and restored state form one
continuous causal chain.

## First end-to-end evidence program

WM-001 is the first confirmatory implementation of this architecture. It is
deliberately narrow: a five-member probabilistic ensemble learns two
observed-context Pendulum actuator regimes, supplies a fixed-budget CEM
controller, and uses balanced replay for continual-learning retention.

Its causal controls separate several explanations that a simple before/after
score would conflate:

- a frozen cold model tests evaluation and collection repetition;
- a joint-target permutation keeps target marginals and optimizer work while
  breaking the input/outcome link;
- an independently evolving, action-independent phase oscillator supplies a
  prespecified nuisance-process control from the exact cold compound state with
  a matched transition count and optimizer index schedule; a disjoint
  own-process split must first verify that the control actually learned;
- a naive B-only update demonstrates actual interference;
- random control and true-dynamics CEM bound executed behavior.

The required evidence path is end to end: immutable real records, exact receipt
ancestry, held-out prediction, paired real-environment return, interference,
retention, a 15-component checkpoint, and fresh-process behavior. Both the live
and restored K7 evaluation traces are persisted and independently reopened.
Formal evidence uses fresh unseen seeds and an implementation binding over a
clean commit. The producer cannot accept its own claim; an external artifact
audit and a separate adversarial semantic review must both pass before an
accepted adjudication package can exist.

For protocol 1.4.0, the auditor selects formal arithmetic only from the bound
runtime, verifies the result and its live accelerator/CUDA and dependency bytes
against that runtime, and requires exact prediction-target bytes and coverage
counts. Adjudication reruns an exclusive descriptor-bound copy of the captured
pre-bound auditor bytes, closing the source-path swap window before it compares
the canonical report byte for byte.

The first protocol-1.1.1 formal attempt is preserved as a rejected attempt. It
provided bounded K0–K6 pilot evidence, but its original live K7 evaluation was
not retained and an independent learned-source control did not yet exist.
Protocol 1.2.0 repaired those two evidence defects, but its pre-formal
development review found that the oscillator arm had no held-out manipulation
check. No v1.2.0 formal seed was opened. Protocol 1.3.0 adds that check, narrows
the conclusion to this specified learned nuisance process, uses a new seed
domain and transparently derived masters, and does not repair or relabel prior
artifacts.

The eight-seed v1.3.0 formal producer result passed K0–K7, with large held-out
predictive and executed-return effects, demonstrated naive interference,
replay-based retention, and exact fresh-process parity. It did not receive
formal acceptance. The pre-bound independent auditor returned two false
negatives: a duplicated wrong seed constant and a single coverage classification
at an underspecified floating-point endpoint. Because the protocol requires a
clean pass from the auditor bound before outcomes, the attempt is explicitly
rejected even though neither finding changes a gate. The
[formal results review](wm001-v130-formal-results.md) records the exact effects,
claim dispositions, and next confirmation requirements.

Protocol 1.4.0 performs that fresh confirmation without changing the formal
agent, data budgets, controls, thresholds, or gate order. Coverage is now
defined by a fixed scalar-binary64 operation sequence over the exact persisted
float32 prediction tensors. Each row stores an authoritative covered count and
target count; K3 applies `[0.70, 0.99]` using integer cross-products, and the
independent auditor must reproduce every count exactly. The binding preserves
endpoint-neighbor conformance, the disclosed v1.3 boundary coordinate, the
auditor source digest, and the test-report digest. Fresh master seeds are
formula-derived and collision-checked. Development is evidence/custody-only,
and the first formal reset begins the sole permitted v1.4.0 attempt. See the
[v1.4 confirmation plan](wm001-v140-confirmation-plan.md).

The v1.4 formal producer completed all eight fresh seeds and passed K0–K7. A
direct run of the corrected bound auditor also passed 6,393,031 checks with zero
failures or gaps, and one content-addressed semantic review incorporating three
adversarial referee passes accepted the narrow fixture claim. Formal acceptance
still failed at the final custody boundary: the adjudicator's `python -I`
execution hid the user-site locations from which bound distributions had been
resolved, so its private audit replay did not match the passing report. It
returned 289 dependent failures and one replay gap; no accepted package was
published. The harness also refused to package that conformance-failing replay
as rejected. Protocol 1.4 is therefore retired without an accepted
demonstration. The
[v1.4 formal results review](wm001-v140-formal-results.md) records the exact
effects and failure.

Protocol 1.5.0 implemented the replacement evidence boundary but its sole
outcome-producing development qualification exposed one remaining custody
gap: Gymnasium's lazy import added two process variables after the environment
had been sealed. The producer was not outer-finalized, no result or development
closure was published, and formal execution never became eligible.

Protocol 1.6.0 kept the same agent, scientific blocks, budgets, controls,
metrics, thresholds, and killing order. It fixed
`PYGAME_HIDE_SUPPORT_PROMPT=hide` and `SDL_AUDIODRIVER=dsp` from process start,
crossed that lazy boundary in a result-free sealed rehearsal, and made
development qualification consumption explicitly single-use. A non-editable
isolated wheel and complete transitive package inventory were shared by
prebinding, producer, auditor, and adjudicator.
Its prospective runtime seals, producer attempts, development audit/closure
attempts, formal audit, and adjudication all used an outer launcher that held
one repository-global cooperative sealed-runtime lock and committed the
terminal only
after child exit and descriptor rechecks by a deterministic same-inode
hardlink. Formal execution could consume only the accepted, outer-finalized
canonical binding attempt; copied or direct bindings were rejected. The
formal-audit and adjudication claims were version-scoped and single-use, and a
no-report audit failure reached a terminal rejected package with no replay.
Those were application-level custody properties under an explicit trusted
same-principal/kernel/filesystem boundary, not external tamper resistance. See
the [v1.6 confirmation plan](wm001-v160-confirmation-plan.md).

The sole v1.6 development producer completed and was outer-finalized, but its
sole independent audit failed before emitting a report. The captured auditor
ran from a private temporary directory and tried to open an undeclared sibling
`producer_bootstrap.py`; only the protocol and raw-result schema had been
captured. The audit attempt was outer-finalized as failure evidence, so v1.6
has no development closure, binding, or formal authority. Its K3–K6 values
remain unopened and unused. The exact disposition is recorded in the
[v1.6 development-audit failure review](wm001-v160-development-audit-failure.md).

Protocol 1.7.0 preserved the same scientific system and repaired only this
auditor input boundary. Full outcome audits now require
`producer_bootstrap.py`, `protocol.json`, and
`schemas/raw-result.schema.json` as exact captured support files. Development
reopens the captured bootstrap identity directly; formal audit additionally
requires equality with the bound source snapshot. No ambient `HERE`, working
directory, or importable-install fallback is permitted. Both restart-runtime
branches and negative omission/mutation cases are exercised before the
single-use development path exists. The expected bootstrap digest is bound
independently of the captured file, and the canonical branch report plus all
three path and three descriptor execution identities are retained and
cross-checked from preformal rehearsal through formal verification. Version
1.7 also drops nonexistent startup search entries, rejects extant ambient
import roots, and restricts every retained child search directory to the bound
standard-library or explicit package-root inventory. Its sole development
producer completed and its independent audit passed. Its sole closure then
failed because the terminal live recheck tried to materialize the
320,556,697-byte authenticated result through a helper capped at 64 MiB. The
failure is recorded in the
[v1.7 closure review](wm001-v170-development-closure-failure.md).

Protocol 1.8.0 keeps the v1.7 support and runtime repairs, replaces that
whole-file read with bounded descriptor-streamed hashing plus before/after
namespace identity, requires byte-canonical qualification archives and strict
JSON scalar types, adds bounded digest-bound failure diagnostics, and requires
the exact `confirmation-v1.8.0` formal child across every producer and
independent reader. It uses fresh seeds, environments, schemas, result
namespaces, seal, review, and binding.
See the
[v1.8 confirmation plan](wm001-v180-confirmation-plan.md).

Its sole development producer, independent audit, and closure completed, but a
fresh-process verification of the closure rejected the archived result
qualification. The matrix-contract digest had serialized two frozensets without
sorting them, so interpreter hash randomization changed the digest across
processes. No K3–K6 value was opened and no preformal report, binding, formal
launch, formal audit, semantic review, or adjudication was authorized. The
terminal disposition is recorded in the
[v1.8 closure-verification failure review](wm001-v180-development-closure-verification-failure.md).

Protocol 1.9.0 preserves the same scientific system and repairs only this
portability boundary. It canonicalizes every matrix-contract row, binds the
golden digest, and requires the accepted closure and operator attempt to be
reopened in an independent fresh sealed-runtime process before preformal
evidence may be created. It uses fresh seeds, environments, schemas, result
namespaces, seal, review, and binding. See the
[v1.9 confirmation plan](wm001-v190-confirmation-plan.md).

Its sole development producer, independent audit, closure transaction, and
post-finalization sealed-runtime reopen completed. The prospective preformal
gate then failed because a runner test observed the now-existing canonical
development closure outside its temporary fixture. The pre-evidence suite had
passed only while that path was absent. Every recorded runtime, source, Git,
input, and package identity remained stable, and no K3–K6 value was opened.
The terminal disposition and required fresh-version repair are recorded in the
[v1.9 preformal-test failure review](wm001-v190-preformal-test-failure.md).

Protocol 1.10.0 preserves the same scientific system and repairs the complete
preformal-to-formal-input boundary exposed by that failure and its forensic
review. Runner lifecycle paths are module-level and derived from redirected
roots, so a test fixture cannot silently retain the live closure path. The
canonical preformal artifact is an initially absent nested bundle at
`results/development/v1.10.0/preformal/`. A deterministic hidden sibling claims
the attempt; all ten subprocesses finish before the 20 stdout/stderr logs and
report are staged, fsynced, and atomically published as one no-replace
directory. Its generation envelope reports the exact failing command and exit
status separately from source, runtime, Git, input drift, or semantic runtime
output failure, returns nonzero on failure, and cannot claim `passed: true`
when the report has `all_pass: false`.

The public runtime surface no longer offers the obsolete
`development-evidence` mode. Command 9 is exactly
`runtime-accepted-closure-evidence` and carries seven independently declared
inputs: the canonical closure, the accepted closure-attempt terminal and its
same-inode outer completion, the runtime seal, both bootstraps, and the
prospective review. Both the producer-side verifier and the independent formal
auditor parse its canonical JSON semantically, bind the relevant digests, and
require the two closure-attempt links to identify the same inode. A direct
cross-contract test prevents auditor-authored fixtures from concealing command
or input drift.

Finally, binding creation invokes the exact independent formal-input consumer
before the binding attempt can be accepted. That consumer reconstructs the
preformal report, command 9 evidence, runtime conformance, accepted closure,
development qualification, source, dependencies, and audit-execution identity.
Its canonical `formal-input-preflight.json` receipt is retained in the accepted
binding attempt, recomputed during attempt verification, required and
cross-checked by the standard-library outer launcher, and carried into the
formal artifact. The v1.10 plan and exact operator sequence are in the
[v1.10 confirmation plan](wm001-v1100-confirmation-plan.md) and
[v1.10 operator runbook](wm001-v1100-operator-runbook.md).

Its sole development producer, independent audit, closure transaction, and
post-finalization sealed-runtime reopen completed. The prospective preformal
gate then failed after all ten commands had run because the QA-side semantic
check invoked the full runtime-inventory-bound closure verifier. That verifier
correctly rejected QA's additional pytest, Ruff, mypy, pip, and support
packages. The deterministic hidden claim contains the 20 command logs, while
the canonical report bundle, binding, and formal launch are absent. No
performance or K value was opened. The terminal disposition and required
fresh-version repair are recorded in the
[v1.10 preformal-test failure review](wm001-v1100-preformal-test-failure.md).

Protocol 1.11.0 preserves the same scientific system and repairs the full
cross-environment composition class before another outcome is permitted. Live
closure creation/reopen, binding creation, and formal launch still recompute
the exact package closure, package-root and standard-library inventories,
ownership, executable, source, flags, and process environment under sealed
runtime custody. Downstream QA roles instead authenticate canonical recorded
objects and their cross-links; they do not replay a live-runtime predicate
whose precondition cannot hold in the deliberately larger QA environment.
`verify_live_binding` remains the explicit ambient guard and is tested to fail
under QA-only inventory.

Command 9's QA consumer now parses the exact closure schema, safe ordered
archive-member projection, fixed roles, custody digests, accepted-attempt
terminal and same-inode completion, and runtime receipt without importing the
live closure verifier. Ordinary semantic exceptions become named failed checks
and a truthful nonzero report-generation envelope; interrupts and other
`BaseException` subclasses still propagate.

Command 10 now records the complete canonical runtime package/root/
standard-library/ownership inventory and complete fresh-child identity receipt,
not only opaque hashes. The preformal producer, formal-binding producer,
central verifier, independent auditor, and standard-library launcher all
require a zero exit, passing status, empty stderr, exact typed objects,
recomputed hashes, and equality with the binding's dependency custody. The
independent formal-input consumer cross-links recorded Python versions rather
than comparing them with its ambient QA interpreter, and requires exact
true/true/false development-closure status.

Protocol 1.11 used fresh paths, environments, seeds, schemas, seal, and
prospective review. Its sealed result-free command-10 rehearsal completed the
intended inventory, fresh-child, and branch-exact semantics, but PyTorch 2.9
emitted a benign deprecation `UserWarning` on stderr when legacy TF32
getters/setters were accessed. Because command 10 requires exactly zero stderr
bytes and sealed executable bytes cannot be repaired in place, v1.11 is
terminal without a development producer, collected experience, training, or
metric. Its disposition is recorded in the
[v1.11 result-free rehearsal failure](wm001-v1110-result-free-rehearsal-failure.md).

Protocol 1.12 moved runtime configuration, independent CUDA replay, and
runtime identity to PyTorch 2.9's explicit string-valued `fp32_precision`
hierarchy and passed the exact-zero-stderr rehearsal. It completed
development, accepted independent audit, closure, and all ten preformal
commands. Its sole binding transaction then exposed a representation
contradiction: `source.test_log_files` reused a digest definition requiring
`bytes >= 1`, while the ten successful stderr streams were required to have
`bytes == 0`. The failed binding attempt was outer-finalized and no formal
launch occurred. Its disposition is recorded in the
[v1.12 binding-schema failure](wm001-v1120-binding-schema-failure.md).

Protocol 1.13 preserved the same scientific system and all v1.12 runtime
custody, introduced formal-binding v10, validated the actual 20-row stream
projection, and validated the complete assembled binding against its root
schema. It passed command 10, development, accepted audit, closure, and all ten
preformal commands. Its strict binding consumer then passed the preserved
report copy to the canonical live-bundle verifier. That role requires the
original development/preformal path and an exclusive report-plus-logs
directory, so it correctly rejected the mixed binding staging directory. The
sole binding attempt failed before acceptance and was outer-finalized; no
formal launch occurred. Its disposition is recorded in the
[v1.13 binding-verifier failure](wm001-v1130-binding-verifier-failure.md).

Protocol 1.14 repaired the canonical-versus-preserved report role and passed
its prospective gates, development, independent audit, closure, fresh-runtime
reopen, and preformal report. Its sole binding transaction then exposed a
separate independent archive-reader defect: after all 86 declared members had
verified, a second `TarFile` iterator replayed a cached member and the valid
archive was falsely rejected as having a physical extra member. The terminal
failed binding was outer-finalized; no formal launch or formal outcome exists.

Protocol 1.15 repaired the archive-reader boundary and passed its prospective
gates, development, accepted audit, closure, fresh-runtime reopen, preformal
report, accepted binding, and the operator-recorded final stop/go sequence. Its
sole formal invocation then returned `1` before producer custody. The
operator-diagnostic traceback
identifies a separate standard-library custody defect: the launcher reused its
64 MiB bounded control-file reader for every development-producer file and
therefore rejected the authenticated 320,977,868-byte `result.json`. The
fsynced binding-keyed root was consumed, but no formal
marker, confirmation directory, producer, outcome, audit, review, or
adjudication exists. The frozen v1.15 plan and operator sequence remain in the
[v1.15 confirmation plan](wm001-v1150-confirmation-plan.md) and
[v1.15 operator runbook](wm001-v1150-operator-runbook.md); the terminal
disposition and fresh-version repair requirements are in the
[v1.15 formal-invocation failure](wm001-v1150-formal-invocation-failure.md).

Protocol 1.16 repaired the bulk-streaming and result-qualification custody
boundaries and passed every authenticated gate through accepted binding; the
operator-recorded final stop/go also passed. Its operator-observed exact
production-path accepted-binding rehearsal then rejected the real
audit terminal before child dispatch: the terminal and reproduction receipt
correctly name a content-addressed runtime sidecar, while the standard-library
consumer required a stale fixed execution filename. Because the rehearsal
preceded formal-root creation, no formal authority or outcome exists. Protocol
1.16 is retired, and its bounded engineering evidence and fresh-version repair
requirements are recorded in the
[v1.16 accepted-binding rehearsal failure](wm001-v1160-accepted-binding-rehearsal-failure.md).
Protocol 1.17 repaired the real 15-member audit-package boundary and reached an
authentic accepted, outer-finalized result-free rehearsal. Its separate
post-rehearsal verifier then rejected the valid matrix identity by reading a
field forbidden by the production formal-binding v10 schema. No formal
authority or outcome exists. The terminal evidence, counterfactual, and fixture
gap are recorded in the
[v1.17 independent rehearsal-verifier failure](wm001-v1170-independent-rehearsal-verifier-failure.md).
Protocol 1.18 repaired that boundary: the production launcher and separate
verifier accepted the exact same binding-keyed child bytes, and the sole sealed
formal producer completed. Its only independent formal-audit execution then
reached the bound 600-second timeout before emitting a report. The authenticated
failure has empty streams, no return code, no replay, no independent-audit
digest, and no reviewed gates. The one-shot adjudication rejected the version
and forbids same-version replay. This is an audit-capacity failure, not evidence
for or against any scientific endpoint. The exact evidence and successor
requirements are recorded in the
[v1.18 formal-audit timeout](wm001-v1180-formal-audit-timeout.md).
Protocol 1.19 repaired the missing scale/liveness contract with closed audit
roles, two captured development executions, and independently recomputed
capacity evidence. The complete development audit passed twice. Its sole
closure transaction then exposed a separate custody-composition defect: the
valid producer terminal and outer completion are one two-link inode, but the
closure's generic canonical-JSON reader silently required one link. The
authenticated failure occurred before canonical closure publication, so no
binding, formal authority, or scientific outcome exists. The evidence and
fresh-version correction are recorded in the
[v1.19 development-closure custody failure](wm001-v1190-development-closure-custody-failure.md).

## Open engineering boundaries

- WM-001 supplies the first probabilistic neural world-model and fixed-budget
  control backend without changing lifecycle semantics, but its collect → learn
  → improve → retain claim remains unestablished until one formal artifact
  passes independent audit, semantic review, descriptor-bound reproduction, and
  external adjudication packaging.
- Protocols 1.10 through 1.19 are terminally retired. Version 1.19 verified its
  development audit-capacity repair but failed before closure because one
  consumer imposed the wrong hard-link contract. The complete lifecycle claim
  remains unestablished until a fresh version's one formal artifact passes
  every audit and adjudication gate.
- The custody layer is deliberately not hardened against the repository or
  environment owner, noncooperating same-account writers, privileged actors, a
  compromised kernel, or transient mutate-and-restore attacks. External
  attestation, read-only media, or an independently operated transparency log
  would be needed for a stronger trust model.
- Value-of-information estimates require their own calibration and adversarial
  controls.
- The current in-memory lifecycle journal exposes partial completion but cannot
  automatically continue or survive restart. Durable idempotent recovery is still
  required; the transactional learning path handles in-process failures but not
  abrupt process death between durable writes.
- The learning commit is serialized across state, ledger, and model ownership,
  but other runtime multi-operation sequences are not globally serialized.
  Concurrent callers can still race across interaction-stage boundaries.
- Exact mid-episode resume requires environment, recurrent-belief, pending-action,
  side-effect, and RNG reconciliation.
- Continual learning must demonstrate both retention and plasticity across
  broader overlapping tasks; WM-001's two observed-context actuator regimes are
  only the first bounded test.
- External benchmark results and strong published baselines are required before
  making a capability or novelty claim.
