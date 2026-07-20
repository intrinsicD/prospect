# WM-001: World-model lifecycle

WM-001 is Prospect's first end-to-end causal learning experiment. It asks one
deliberately narrow question:

> Can one agent collect identified experience, change one persistent world model
> because of that experience, improve held-out prediction and executed behavior,
> learn a conflicting contextual task, retain the original gain, and reproduce
> the retained state after a fresh-process restart?

The scientific contract is [`protocol.json`](protocol.json), and its raw-byte
seal is [`SEALED_PROTOCOL.sha256`](SEALED_PROTOCOL.sha256). The schema contracts
are
[`schemas/raw-result.schema.json`](schemas/raw-result.schema.json) and
[`schemas/formal-binding.schema.json`](schemas/formal-binding.schema.json).
Numerical thresholds are hypotheses fixed before formal outcomes, not evidence
that Prospect passes them.

## Current status

Protocol 1.10.0 is the active prospective confirmation. It preserves the exact
v1.4 model, learning algorithm, optimizer, planner, controller, budgets,
controls, metrics, thresholds, and K0–K7 order. Its scientific-block digest is
`fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`.
Only the execution and evidence boundary changes. Protocol 1.5 is retired
after its sole outcome-producing development qualification found that
Gymnasium's lazy import added two unsealed process variables. That producer
has no result, outer completion, development closure, binding, or formal
marker. Protocol 1.6 fixed those variables from process start and completed
its sole development producer, but its sole canonical audit failed because
the captured auditor support omitted `producer_bootstrap.py`. It has no
accepted audit, closure, binding, or formal launch. Version 1.7 made that
bootstrap explicit bound support and its development audit passed, but its
sole closure tried to read a 320,556,697 byte authenticated result through a
64 MiB payload limit. Version 1.8 streams that terminal digest under descriptor
and namespace custody, requires a byte-canonical USTAR qualification archive
with strict JSON scalar types, retains only bounded authenticated failure
diagnostics, and enforces the exact `confirmation-v1.8.0` formal child in every
producer and independent reader. Its development producer, independent audit,
and closure completed, but the first public fresh-process closure verification
rejected the archived result qualification because its matrix-contract digest
had serialized two frozensets without sorting them. Version 1.8 is therefore
retired before preformal or formal authorization. Version 1.9 sorts every
matrix-contract row, binds a golden digest, and requires an independent fresh
sealed-runtime closure verification before preformal evidence can exist. Its
sole producer, independent audit, closure transaction, retained reopen receipt,
and post-finalization sealed-runtime reopen completed. The fixed preformal gate
then failed because one runner test redirected its producer root but still read
the real canonical closure path. The same test had passed while that path was
absent. The failed report retired v1.9 before binding or formal launch; no
K3–K6 value was opened or used.

Version 1.10 repairs only that evidence boundary and two independently found
late blockers. Runner lifecycle paths are root-derived and fixture-isolated.
Preformal evidence is created under the initially absent, version-owned
`results/development/v1.10.0/preformal/` bundle. A deterministic hidden claim
makes interruption nonretryable and is made durable by fsyncing its parent
before the first command; all ten commands finish before its 20 logs and report
are staged, fsynced, and atomically published as one directory. Operator
attempt namespaces are likewise created component by component with each new
directory entry durably committed before the hidden attempt claim. Report
generation returns a truthful nonzero failure envelope that distinguishes
exact command failures, identity drift, and semantic runtime-output failures.
The obsolete public `development-evidence` runtime mode is removed. Producer
and independent-auditor contracts now agree directly on command 9,
`runtime-accepted-closure-evidence`, its seven inputs, semantic digest checks,
and the two-link same-inode closure-attempt custody. Finally, the exact
independent formal-input consumer must pass during binding creation and again
during binding-attempt verification; the outer launcher also requires its
terminal-bound identity and independently reopens command 9 and command 10 to
cross-check both semantic receipt digests before importing the formal producer.
Its canonical
`formal-input-preflight.json` receipt is retained in the accepted attempt and
later formal artifact.

The v1.10 harness uses a non-editable isolated wheel, complete dependency and
package-root inventories, a deterministic module-search path restricted to
explicit package roots and inventoried standard-library directories, one
captured descriptor runner, a repository-global cooperative outer-launch lock,
deterministic same-inode terminal completion, canonical development
audit/closure/binding attempts, and single-use formal audit and adjudication
claims. Formal execution accepts only
`results/operator-v1.10/bindings/formal-binding-v1.10.0/formal-binding.json`
after that attempt is accepted and outer-finalized. A copy or directly created
binding is not valid authorization.

The trust boundary is explicit:
`prospect.wm001.trust-model.v1` has `tamper_resistant: false`. The kernel,
filesystem, base interpreter/standard library, account, and all writers to the
repository, environment, and results roots are trusted. The cooperative lock,
hashes, descriptors, inventories, and no-replace publication detect accidental
or persistent drift; they do not resist an owner, noncooperating same-account
writer, privileged actor, compromised kernel, or transient
mutate-and-restore attack.

Before formal launch, exactly one fresh full-budget two-seed development
qualification must complete, followed by its canonical independent audit,
closure attempt, exact ten-row/20-log preformal report, independent prospective
harness review, independent formal-input preflight, and canonical binding
attempt. Development performance remains
descriptive and permanently claim-ineligible. Creation of the canonical
qualification root consumes the sole v1.10 development attempt; a failure
retires the version rather than authorizing a sibling retry. The complete lifecycle claim
remains unestablished until the one new eight-seed formal producer is
outer-finalized, the official independent audit and semantic review accept
K0–K7, and the terminal adjudication package is accepted and independently
verified.

Protocols 1.9.0, 1.8.0, 1.7.0, 1.6.0, 1.5.0, and 1.4.0 are immutable and retired.
Protocol 1.9.0 completed development, audit, closure, and sealed reopen, but its
post-closure preformal test failure made every development artifact
claim-ineligible and authorized no binding or formal launch. Its terminal
disposition is preserved in the
[v1.9 preformal failure review](../../docs/wm001-v190-preformal-test-failure.md).
Protocol 1.8.0's development producer, audit, and closure completed, but its
closure was not portable across fresh interpreter hash seeds; no preformal
report, binding, or formal launch was authorized. Protocol 1.7.0's development
producer and audit completed, but its sole closure exposed
the size-limited recheck and published only failure evidence. Protocol 1.6.0's
development outcome remains opaque and claim-ineligible after its captured
auditor failed. Protocol 1.5.0 stopped during development custody and never
reached formal eligibility. Protocol 1.4.0's eight-seed producer and direct
corrected audit passed, but isolated adjudication could not reproduce the
report because the bound closure depended on user-site visibility, and the
rejection path then failed to preserve an official rejected package. Protocol
1.3.0 is also rejected because its pre-bound auditor had a duplicated seed
constant and underspecified endpoint arithmetic. Their exact evidence and
dispositions remain in the
[v1.4 results review](../../docs/wm001-v140-formal-results.md) and
[v1.3 results review](../../docs/wm001-v130-formal-results.md); v1.10 does not
repair or relabel either attempt. The active design is the
[v1.10 confirmation plan](../../docs/wm001-v1100-confirmation-plan.md).

## What the experiment must establish

The claim is a continuous causal chain, not a collection of unrelated scores:

```text
real interaction
  -> canonical experience custody
  -> exact consumed-transition ancestry
  -> transactional shared-model update
  -> held-out predictive improvement
  -> better executed fixed-budget control
  -> conflicting-task plasticity
  -> replay-based retention
  -> component-complete fresh-process parity
```

The evidence ladder is intentionally strict:

| Stage | Killing gate | What can be said if it passes |
|---|---|---|
| Bind | K0 | Protocol, implementation, dependencies, runtime, seeds, and budgets are the predeclared ones. |
| Collect | K1 | Real experience has unique lineage and training is isolated from validation, behavior, and imagined transitions. |
| Learn | K2–K3 | A failure-atomic update changed the declared shared model from eligible experience; the oscillator arm first learns its own held-out process; and task-A experience beats frozen, corrupted, and that verified learned-source control on held-out predictive NLL. |
| Improve | K4 | The updated model improves executed task-A return beyond cold, frozen, and the verified oscillator-trained control under the same frozen MPC and paired evaluation budget. |
| Interfere | K5 | The same weights learn task B, and the matched naive B-only path demonstrates that A/B interference is real. |
| Retain | K6 | Balanced replay preserves the prespecified fraction of the A gain without an unacceptable loss of B plasticity. |
| Persist | K7 | All 15 declared stateful components restore in another process with exact identity, prediction, action, and return parity. |

Passing a later numeric comparison cannot rescue an earlier failed gate. In
particular, a parameter digest change is not learning, prediction improvement is
not behavioral improvement, reload parity is not retention, and retention is not
tested unless task B improves while the naive learner measurably forgets task A.

## Fixed mechanism and controls

Pendulum-v1 keeps representation learning out of this first test. Task A uses
normal torque. Task B reverses torque and exposes one observed context scalar.
Both tasks use the same five-member probabilistic MLP ensemble; task-specific
models, adapters, heads, and checkpoints are forbidden. The ensemble's
deterministic mean prediction feeds TorchRL 0.13.3's `CEMPlanner` through a
`ModelBasedEnvBase` adapter. Predictive uncertainty is evaluated, but it is not
an intrinsic planning reward in WM-001.

Each formal replicate follows the sealed sequence:

1. initialize the cold shared model and fork an exact isolated control state;
2. collect eight complete task-A episodes;
3. collect eight independent-oscillator episodes in isolated custody;
4. prepare, validate, and commit the 2,000-step task-A update;
5. train the matched oscillator arm, verify its disjoint own-process predictive
   gain, and run the frozen and joint-target-permutation controls from their
   cold forks;
6. reload immutable cold, frozen, corrupted, irrelevant, and learned checkpoints
   and evaluate them on identical held-out task-A budgets with learning and
   replay writes disabled;
7. collect eight complete task-B episodes through the same agent;
8. fork the post-A state into balanced A/B replay and naive B-only updates;
9. require the naive path to learn B and measurably forget A;
10. require replay to learn B while retaining A;
11. checkpoint the complete declared state at an episode boundary; and
12. restore in a fresh process and reproduce identities, predictions, actions,
    and paired returns.

Prediction validation uses eight held-out episodes per task and replicate.
Executed behavior uses 32 paired reset seeds per task, condition, and replicate.
The inferential unit is the replicate seed, not a transition or episode.

The required controls are:

- **frozen cold model:** rules out collection or repeated evaluation alone;
- **corrupted joint target:** preserves target marginals and optimizer budget
  while breaking the input/outcome relationship;
- **independent learned-source evidence:** uses real, action-independent
  oscillator experience with matched cold ancestry, transition count, optimizer
  steps, sampled-index schedule, validation rows, planner budget, and behavior
  resets; a separate held-out oscillator split first verifies that its update
  learned. It is one prespecified nuisance process, not a universal
  causal-relevance control;
- **naive sequential learner:** demonstrates interference and provides the
  no-retention baseline;
- **executed random policy:** supplies a paired lower bound; and
- **true-dynamics MPC:** supplies a separately namespaced ceiling under the same
  planner and evaluation budget.

The exact seed derivation, task semantics, budgets, metrics, Student-t
intervals, thresholds, and K0–K7 killing order live in `protocol.json`.

## Transactions, checkpoints, and evidence custody

Learner preparation operates on immutable model bytes outside the short commit
critical section. Commit revalidates canonical transition ancestry, then advances
the update ledger, agent learning state, and owned model under a common lock
order. Any in-process `BaseException` during that composed commit restores all
three participants. This is failure-atomic in process; abrupt process death still
requires durable write-ahead recovery and is not claimed.

Every evaluated condition is reloaded from an immutable content-addressed model
snapshot. Optimizer consumption is represented by content-addressed bootstrap
manifests, while the canonical experience store remains the authority. Replay is
an index over that store, never an alternative evidence namespace.

K7 checkpoints exactly these component IDs:

1. `world_model`
2. `optimizer`
3. `model_version_ledger`
4. `experience_store`
5. `replay_index`
6. `replay_sampling_history`
7. `update_receipts`
8. `agent_runtime`
9. `scaling_configuration`
10. `python_rng`
11. `numpy_rng`
12. `torch_cpu_rng`
13. `torch_accelerator_rng`
14. `collection_rng`
15. `planner_rng`

A producer attempt gets a new exclusive directory before the first environment
reset. It writes logs, raw result rows, model and prediction sidecars, optimizer
manifests, checkpoint archives, restart-process evidence, and progress records.
On success or failure it finalizes `producer-manifest.json`, which binds the
complete producer file set by byte length and SHA-256. Attempts are never
resumed, overwritten, or repaired in place.

## Two lanes and two seals

The v1.10 development rehearsal uses only seeds `1647437737` and `1156509260`
and the complete formal budgets. It is useful only for schema, deterministic
execution, exact arithmetic, audit coverage, restart, and custody validation.
Its K3–K6 performance values are descriptive, cannot decide whether formal may
launch, are never claim-eligible, and cannot be relabeled. After the exclusive
development closure is published, every further v1.10 development run is
forbidden.

The one v1.10 formal execution uses:

```text
3363134750, 2153178322, 2277484641, 572614265,
3119775486, 3121614244, 3646941950, 827253974
```

The launch-time prebinding replay may reset isolated QA-only Pendulum fixtures;
those resets collect no formal experience, train no model, and do not consume
the attempt. The first outcome-producing formal replicate/task reset after
`results/formal/formal-launch-v1.10.0.json` begins the sole attempt. No resume,
retry, corrected-audit upgrade, early stopping, extra training, exclusion, or
analysis change is allowed.

There are two pre-outcome bindings:

1. The scientific seal fixes the protocol and result/binding schemas.
2. The one canonical formal implementation-binding attempt fixes a clean Git
   commit and tree, all
   executed source and test digests, dependency closure, runtime, deterministic
   settings, environment conformance, exact coverage conformance, auditor/test
   digests, checkpoint implementation, and the content-addressed
   restart-runtime branch report plus complete repeated path/descriptor
   execution receipt already sealed by the preformal rehearsal. The binding
   attempt also preserves `formal-input-preflight.json`, produced by the exact
   independent formal consumer and recomputed when the attempt is verified.

Changing scientific semantics requires a new protocol version. Changing bound
source, dependencies, or runtime before the first outcome-producing formal
replicate/task reset requires a new implementation binding. After that reset,
any such change requires a new protocol version; previous failed attempts
remain evidence.

## Active v1.10 runbook

Use the
[WM-001 v1.10 operator runbook](../../docs/wm001-v1100-operator-runbook.md).
It defines the typed runtime seals, canonical operator paths, outer-completion
checks, exact development/preformal/binding order, and the one formal,
official-audit, and adjudication sequence.

## Retired v1.4 command sketch — do not execute

The commands below document the retired v1.4 workflow and are retained only to
explain its evidence history. They do not satisfy v1.10 custody and must not be
used to launch an experiment.

### 1. Install and verify the pre-outcome contract

```bash
python -m venv .venv
source .venv/bin/activate
make install-runtime

python -m bench.world_model_lifecycle.verify protocol
CUBLAS_WORKSPACE_CONFIG=:4096:8 make check-runtime
```

Do not continue to formal binding unless both commands pass. `check-runtime`
runs lint, the scoped static type check, epistemic tests, WM-001 tests, and
epistemic diagnostics.

### 2. Run and audit the development lane

Run the single required complete two-seed rehearsal:

```bash
DEV_ARTIFACT="bench/world_model_lifecycle/results/development/$(date -u +%Y%m%dT%H%M%SZ)-$$"
python -m bench.world_model_lifecycle.run development \
  --device cuda \
  --output "$DEV_ARTIFACT"
python -m bench.world_model_lifecycle.verify result "$DEV_ARTIFACT/result.json"
python -m bench.world_model_lifecycle.artifact_audit "$DEV_ARTIFACT" \
  --producer-bootstrap bench/world_model_lifecycle/producer_bootstrap.py \
  --output "artifacts/wm001-audits/$(basename "$DEV_ARTIFACT").json"
```

The audit output must be outside the immutable producer directory. The
rehearsal must have complete schema/matrix evidence, deterministic execution,
seed parity, exact coverage agreement, restart parity, custody integrity, and
zero audit failures or gaps. Its performance gates do not screen formal launch
and its numbers cannot support the WM-001 claim.

### 3. Commit the exact candidate and create a formal binding

Resolve every development engineering or audit finding first. Do not use K3–K6
performance values to modify the candidate. Commit the exact candidate source,
then confirm that the tracked and untracked worktree is clean:

```bash
git status --short --untracked-files=all
```

Create a fresh ignored evidence directory, preserve the final check output, and
bind that clean commit. The binding function runs 1,024 Pendulum conformance
cases, 512 full 200-step paired-action oscillator cases, and the exact coverage
endpoint/regression corpus; it refuses a source, auditor, test-report,
dependency-lock, or runtime mismatch.

```bash
BINDING_DIR="artifacts/wm001-binding-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$BINDING_DIR"
TEST_REPORT="$BINDING_DIR/prebinding-test-report.txt"
set -o pipefail
CUBLAS_WORKSPACE_CONFIG=:4096:8 make check-runtime 2>&1 | tee "$TEST_REPORT"

BINDING="$BINDING_DIR/formal-binding.json"
BINDING="$BINDING" TEST_REPORT="$TEST_REPORT" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 python - <<'PY'
import os
from pathlib import Path

from bench.world_model_lifecycle.binding import create_formal_binding

create_formal_binding(
    output_path=Path(os.environ["BINDING"]),
    test_report_path=Path(os.environ["TEST_REPORT"]),
    conformance_cases=1024,
    device="cuda",
)
PY

python -m bench.world_model_lifecycle.verify binding "$BINDING"
```

`create_formal_binding` writes content-addressed copies of the test, Pendulum,
oscillator, and coverage conformance reports beside `formal-binding.json`. It
refuses to replace any existing binding evidence.

### 4. Launch exactly one formal attempt

```bash
BINDING_SHA="$(sha256sum "$BINDING" | cut -d' ' -f1)"
FORMAL_ATTEMPT="bench/world_model_lifecycle/results/formal/$BINDING_SHA/$(date -u +%Y%m%dT%H%M%SZ)-$$"

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m bench.world_model_lifecycle.run formal \
  --device cuda \
  --binding "$BINDING" \
  --output "$FORMAL_ATTEMPT"
```

The launcher rechecks the live source, Git tree, dependency closure, runtime,
deterministic settings, environment wrapper, and copied binding before the first
reset. Immediately before the first possible reset, it atomically creates the
single protocol-wide `results/formal/formal-launch.json` and copies those exact
bytes into the attempt. It also copies the protocol, seal, schemas, lock, test
report, conformance report, binding, and every byte named by the complete bound
implementation manifest into the attempt. Any existing protocol-wide launch
marker blocks every same-version binding.

Do not issue this command a second time for protocol 1.4.0, regardless of the
first attempt's outcome.

### 5. Verify custody, recompute, and judge

```bash
ARTIFACT="$FORMAL_ATTEMPT" python - <<'PY'
import os
from pathlib import Path

from bench.world_model_lifecycle.artifact import verify_producer_manifest

manifest = verify_producer_manifest(Path(os.environ["ARTIFACT"]))
if manifest["status"] != "completed":
    raise SystemExit(f"producer attempt status: {manifest['status']}")
print(f"producer custody valid: {manifest['file_count']} files")
PY

python -m bench.world_model_lifecycle.verify result \
  "$FORMAL_ATTEMPT/result.json" \
  --binding "$FORMAL_ATTEMPT/formal-binding.json"

mkdir -p artifacts/wm001-audits
AUDIT_REPORT="artifacts/wm001-audits/formal-$BINDING_SHA-$(date -u +%Y%m%dT%H%M%SZ).json"
python -m bench.world_model_lifecycle.artifact_audit "$FORMAL_ATTEMPT" \
  --producer-bootstrap \
  "$FORMAL_ATTEMPT/source/bench/world_model_lifecycle/producer_bootstrap.py" \
  --output "$AUDIT_REPORT"
sha256sum "$AUDIT_REPORT"

ADJUDICATION_PACKAGE="artifacts/wm001-adjudications/formal-$BINDING_SHA-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$ADJUDICATION_PACKAGE")"
python -m bench.world_model_lifecycle.adjudication \
  --producer "$FORMAL_ATTEMPT" \
  --audit "$AUDIT_REPORT" \
  --output "$ADJUDICATION_PACKAGE" \
  --disposition pending
```

The producer root is finalized before independent audit and is never modified
afterward. Before creating an external adjudication package, adjudication
captures and verifies the current pre-bound auditor bytes, executes those bytes
through an inherited descriptor to an exclusive private copy, and requires its
fresh canonical output to be byte-identical to the supplied report. The formal
auditor derives its execution device from the binding and requires the result,
live runtime, accelerator/CUDA identity, and installed dependency bytes to
match. Adjudication then copies and binds the reproduced report to the
producer-manifest, result, auditor-source, and formal-binding digests. An
`accepted` or `rejected` disposition additionally requires a canonical
content-addressed semantic review. `pending` and `accepted` packages require a
complete, clean, passing audit. An explicit `rejected` package may instead
preserve an identity- and custody-valid failed or incomplete audit, but its
semantic review must record at least one fatal finding explaining why the claim
is rejected.

The envelope verifier checks schemas, hashes, identities, seed derivation, split
custody, update ancestry, budgets, gate order, and binding consistency. The
artifact auditor reopens untrusted sidecars and checkpoints, recomputes dynamics,
model predictions, metrics, controller RNG evidence, aggregate statistics,
thresholds, checkpoint completeness, and restart parity. It does not trust the
producer's stored aggregate or gate rows.

Only the external audit report may return `passed: true`. Producer gate rows
always keep `claim_supported: false`, including in the formal lane. The claim may
advance only when all of the following are true:

- the producer manifest is complete with status `completed`;
- protocol, binding, and result verification pass;
- the artifact audit reports `integrity_passed: true`,
  `complete_for_claim: true`, `passed: true`, zero failed checks, and zero
  coverage gaps;
- independently recomputed K0–K7 all pass in order; and
- an adversarial semantic review finds no critical lineage, leakage, mechanism,
  statistical, or scope error.

If any condition fails, report the last valid rung and the failure evidence. Do
not describe the complete collect → learn → improve → retain claim as
demonstrated.

## Interpretation boundary

Even a complete WM-001 pass would establish one causal lifecycle on two
observed-context variants of Pendulum-v1 under one bound software and hardware
environment. It would not establish general intelligence, multimodal learning,
long-horizon exploration, cross-hardware bitwise reproducibility, architectural
novelty, or superiority to published state of the art. Those require later
experiments and external benchmarks.
