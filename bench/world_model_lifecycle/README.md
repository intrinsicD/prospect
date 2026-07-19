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

Protocol 1.4.0 completed its sole eight-seed formal attempt. The immutable
producer evidence passed K0–K7, exact pooled coverage was
`49,949/51,200`, and a direct run of the corrected pre-bound auditor passed
6,393,031 checks with zero failures or gaps. One content-addressed semantic
review incorporating three adversarial referee passes accepted the narrow
two-regime Pendulum claim.

The attempt is nevertheless not formally accepted. Mandatory adjudication
executes the captured auditor through a private descriptor under `python -I`.
The bound runtime used transitive distributions from the Python user site;
isolated execution removed that search root and could not reproduce the passing
audit byte-for-byte. The accepted package was refused. A preserved diagnostic
replay reports one bound-input-package failure, 288 dependent prediction/CEM
failures, and one coverage gap. The harness then also refused to package that
failing audit as rejected. Protocol 1.4 is retired, and no accepted or official
rejected adjudication package exists. See the
[v1.4 formal results review](../../docs/wm001-v140-formal-results.md).

Protocol 1.3.0 remains an earlier rejected eight-seed attempt. Its producer
evidence passed K0–K7, but its pre-bound independent auditor contained a
duplicated seed constant and underspecified endpoint arithmetic. See the
[v1.3 formal results review](../../docs/wm001-v130-formal-results.md).

Version 1.4 changed no formal model, learning, controller, budget, control,
threshold, or killing gate. It replaced underspecified coverage arithmetic with
an exact count contract over persisted float32 tensors, corrected schedule
parity, and bound a coverage conformance corpus. Its prospective design and
execution policy remain in the
[v1.4 confirmation plan](../../docs/wm001-v140-confirmation-plan.md).

A development run remains diagnostic. A formal producer result is only
claim-eligible; it is not self-certifying. The lifecycle claim remains
unestablished until a finalized formal artifact passes the independent artifact
audit, a separate semantic results review accepts every killing gate K0 through
K7, and adjudication reproduces and packages those exact bytes.

Protocol 1.3.0 supersedes two non-accepted predecessors. The first v1.1.1
formal artifact supports bounded K0–K6 pilot evidence, but adversarial review
rejected the complete claim because the original live K7 trace was not retained
and no learned independent-source control existed. Protocol 1.2.0 repaired
those defects, but pre-formal review then found that it never verified that the
new oscillator control had learned its own process. No v1.2.0 formal seed was
opened; its two development replicates remain diagnostic only. Neither
predecessor is repaired or relabeled.

Version 1.3.0 preserves separate content-addressed live and restored K7 traces
and uses an isolated independent phase-oscillator learner that forks the exact
cold compound state, uses the same data count and optimizer index schedule as
task A, and is evaluated on the same held-out task-A prediction and control
budgets. A new disjoint oscillator validation split must first prove that this
control learned its source process. The protocol uses a fresh derivation domain
and transparently derived master seeds. The manipulation threshold reuses the
existing predictive minimum-effect floor and was fixed before any v1.3.0
development or formal outcome.

The v1.4 producer and independent auditor recompute the same prospectively
specified scalar-binary64 mixture PIT from persisted float32 evidence and
require exact covered-target counts with no tolerance. K3 applies the unchanged
inclusive `[0.70, 0.99]` bounds by integer cross-products. These repairs cannot
upgrade the immutable v1.3.0 attempt, whose failed audit and auditor bytes
remain preserved in its rejected package.

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

The final development rehearsal uses only seeds `2439054559` and `3246851043`
and the complete formal budgets. It is useful only for schema, deterministic
execution, exact arithmetic, audit coverage, restart, and custody validation.
Its K3–K6 performance values are descriptive, cannot decide whether formal may
launch, are never claim-eligible, and cannot be relabeled.

Formal execution used the eight sealed master seeds and exact declared budgets.
The first formal environment reset started v1.4.0's sole attempt. No resume,
retry, corrected-audit upgrade, early stopping, extra training, exclusion, or
analysis change is allowed. The adjudication reproduction failure retired
v1.4.0.

There are two pre-outcome bindings:

1. The scientific seal fixes the protocol and result/binding schemas.
2. A formal implementation binding fixes a clean Git commit and tree, all
   executed source and test digests, dependency closure, runtime, deterministic
   settings, environment conformance, exact coverage conformance, auditor/test
   digests, and checkpoint implementation.

Changing scientific semantics requires a new protocol version. Changing bound
source, dependencies, or runtime before the first formal reset requires a new
implementation binding. After that reset, any such change requires a new
protocol version; previous failed attempts remain evidence.

## Executable runbook

Run these commands from the repository root. CUDA is the intended formal device;
replace `cuda` with `cpu` only if the binding and subsequent run both use CPU.

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
