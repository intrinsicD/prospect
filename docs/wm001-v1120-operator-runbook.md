# WM-001 v1.12.0 operator runbook

Status: prospective. Seal this runbook with the exact v1.12 protocol,
implementation, schemas, lock, tests, and independent review before any v1.12
producer path is created.

Protocols 1.10.0 and 1.11.0 are terminally retired. Version 1.10 reached its
preformal gate before a cross-environment QA/runtime composition error
consumed that version's hidden claim. Version 1.11 passed its sealed static
gates, created and outer-finalized only its prospective runtime seal, and
completed command 10's result-free bootstrap/inventory semantics. That
successful PyTorch 2.9 child emitted a benign TF32 deprecation `UserWarning`
on stderr through legacy precision getters/setters. Command 10 requires
exactly zero stderr bytes, and changing sealed source or wheel bytes is
forbidden, so v1.11 is terminal. It created no development producer, collected
no experience, trained no model, and produced no development or formal metric.
Do not inspect or use any retained K or performance value. Nothing in this
runbook repairs, resumes, removes, or reuses v1.10 or v1.11 evidence,
environments, wheels, seals, locks, or outer completions.
The immutable v1.11 disposition is documented in the
[result-free rehearsal failure](wm001-v1110-result-free-rehearsal-failure.md).

## Fixed paths and clean launcher environment

```bash
set -euo pipefail

REPO="$(pwd -P)"
test "$REPO" = "$(git rev-parse --show-toplevel)"

QA_PY="/home/alex/.venvs/prospect-wm001-v112-reviewed/bin/python"
RUNTIME_PY="/home/alex/.venvs/prospect-wm001-v112-reviewed-runtime/bin/python"
LAUNCH="$REPO/bench/world_model_lifecycle/launch_bootstrap.py"
BOOTSTRAP="$REPO/bench/world_model_lifecycle/producer_bootstrap.py"

DEV_ROOT="$REPO/bench/world_model_lifecycle/results/development"
DEV="$DEV_ROOT/qualification-v1.12.0"
RUNTIME_SEAL="$DEV_ROOT/runtime-seal-v1.12.0.json"
DEV_CLOSURE="$DEV_ROOT/development-closure-v1.12.0.json"
PREFORMAL_ROOT="$DEV_ROOT/v1.12.0/preformal"
PREFORMAL="$PREFORMAL_ROOT/preformal-test-report-v1.12.0.json"

OPERATOR_ROOT="$REPO/bench/world_model_lifecycle/results/operator-v1.12"
DEV_AUDIT="$OPERATOR_ROOT/audits/development-audit-v1.12.0"
CLOSURE_ATTEMPT="$OPERATOR_ROOT/closures/development-closure-v1.12.0"
BINDING_ATTEMPT="$OPERATOR_ROOT/bindings/formal-binding-v1.12.0"
BINDING="$BINDING_ATTEMPT/formal-binding.json"
FORMAL_INPUT_PREFLIGHT="$BINDING_ATTEMPT/formal-input-preflight.json"

OUTER_ROOT="$REPO/bench/world_model_lifecycle/results/outer-completions/v1.12"
RUNTIME_LOCK="$REPO/bench/world_model_lifecycle/results/.wm001-v1.12-runtime.lock"
FORMAL_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-launch-v1.12.0.json"
FORMAL_AUDIT="$OPERATOR_ROOT/audits/formal-audit-v1.12.0"
FORMAL_AUDIT_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-audit-v1.12.0.json"
ADJUDICATION_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-adjudication-v1.12.0.json"
ADJUDICATION_ROOT="$REPO/bench/world_model_lifecycle/results/adjudication-v1.12"
ADJUDICATION_PACKAGE="$ADJUDICATION_ROOT/formal-adjudication-v1.12.0"

REVIEW="$REPO/docs/wm001-v1120-prospective-harness-review.json"
SEMANTIC_REVIEW="$REPO/artifacts/wm001-reviews/formal-v1.12.0.json"

SAFE_ENV=(
  env -i
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  LAZY_LEGACY_OP=False
  LC_ALL=C.UTF-8
  PATH=/usr/bin:/bin
  PYGAME_HIDE_SUPPORT_PROMPT=hide
  SDL_AUDIODRIVER=dsp
  TZ=UTC
)

require_absent() {
  if [ -e "$1" ] || [ -L "$1" ]; then
    echo "required absent path already exists lexically: $1" >&2
    return 2
  fi
}
```

The development masters are exactly `2530568307, 3822916726`. The formal
masters are exactly `402304386, 1582362517, 3717100311, 3870324956,
2551652339, 986753049, 4074588580, 1996653376`. The verifier must verify 110
unique prior masters and 14,960 unique prior streams through v1.11 before any
evidence run.

The prebinding runtime identity must use
`prospect.wm001.prebinding-conformance-request.v2` and
`prospect.wm001.prebinding-conformance.v2`. Its exact string-valued precision
fields cover the global, CUDA-matmul, cuDNN-backend, cuDNN-convolution, and
cuDNN-RNN `fp32_precision` settings; no legacy boolean alias is accepted.

## Build the two environments once

Create both environments at previously absent paths. Build one reviewed wheel
and install that same wheel non-editably. Never reinstall into the runtime
environment after its inventory is rendered.

```bash
BASE_PY="/home/alex/miniconda3/bin/python"
QA_ENV="${QA_PY%/bin/python}"
RUNTIME_ENV="${RUNTIME_PY%/bin/python}"
WHEELHOUSE="$(mktemp -d /tmp/prospect-wm001-v112-wheelhouse.XXXXXX)"

require_absent "$QA_ENV"
require_absent "$RUNTIME_ENV"
test -z "$(git ls-files --others --exclude-standard -- src/prospect bench)"

mapfile -t RUNTIME_PINS < <(
  sed -nE \
    '/^(python|prospect)==/d; s/^(.*==[^ ]+) --distribution-sha256=.*/\1/p' \
    "$REPO/requirements-wm001.lock"
)
test "${#RUNTIME_PINS[@]}" -eq 43

"$BASE_PY" -m pip --isolated wheel --no-deps \
  --wheel-dir "$WHEELHOUSE" "$REPO"
mapfile -t WHEELS < <(
  find "$WHEELHOUSE" -maxdepth 1 -type f -name 'prospect-*.whl' -print
)
test "${#WHEELS[@]}" -eq 1
WHEEL="${WHEELS[0]}"

"$BASE_PY" -m venv "$QA_ENV"
"$BASE_PY" -m pip --isolated --python "$QA_PY" install \
  --no-compile --only-binary=:all: \
  "${WHEEL}[runtime,dev]" "${RUNTIME_PINS[@]}"

"$BASE_PY" -m venv --without-pip "$RUNTIME_ENV"
"$BASE_PY" -m pip --isolated --python "$RUNTIME_PY" install \
  --no-compile --only-binary=:all: \
  "${WHEEL}[runtime]" "${RUNTIME_PINS[@]}"
```

Render the lock from the untouched runtime with bytecode disabled. The
generated WM-001 header must name v1.12.0. The Prospect row must equal the
reviewed wheel's package version and installed-distribution digest. Verify the
lock immediately, then do not install anything else into the runtime.

```bash
LOCK_TMP="$(mktemp "$REPO/.requirements-wm001.lock.XXXXXX")"
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B - "$LOCK_TMP" <<'PY'
from pathlib import Path
import sys

from bench.world_model_lifecycle.binding import installed_package_rows

rows = installed_package_rows()
lines = [
    "# WM-001 protocol 1.12.0 isolated live execution closure.",
    "# Versions preserve the v1.4 scientific runtime; each v1.12 digest covers",
    "# the executable or every stable installed distribution file.",
]
lines.extend(
    f"{row['name']}=={row['version']} "
    f"--distribution-sha256={row['distribution_sha256']}"
    for row in rows
)
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
mv "$LOCK_TMP" "$REPO/requirements-wm001.lock"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B -c \
  'from bench.world_model_lifecycle.binding import installed_package_rows,verify_lockfile_rows; verify_lockfile_rows(installed_package_rows())'
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal --help >/dev/null
```

Stage every intended lifecycle source and test before generating the exact
prospective-review implementation manifest. Author and stage
`wm001-v1120-prospective-harness-review.json`, then commit the whole candidate.
Do not rebuild, reinstall, or change a bound byte after that commit.

## Pre-outcome static gates

All version-scoped paths below must be absent:

```bash
require_absent "$OPERATOR_ROOT"
require_absent "$ADJUDICATION_ROOT"
require_absent "$DEV"
require_absent "$RUNTIME_SEAL"
require_absent "$DEV_AUDIT"
require_absent "$DEV_CLOSURE"
require_absent "$CLOSURE_ATTEMPT"
require_absent "$PREFORMAL_ROOT"
require_absent "$BINDING_ATTEMPT"
require_absent "$FORMAL_INPUT_PREFLIGHT"
require_absent "$FORMAL_MARKER"
require_absent "$FORMAL_AUDIT"
require_absent "$FORMAL_AUDIT_MARKER"
require_absent "$ADJUDICATION_MARKER"
require_absent "$ADJUDICATION_PACKAGE"
require_absent "$SEMANTIC_REVIEW"
require_absent "$RUNTIME_LOCK"
require_absent "$OUTER_ROOT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal \
  verify-prospective-review --review "$REVIEW"
"$QA_PY" -I -B -m pytest -q
"$QA_PY" -I -B -m ruff check src/prospect bench tests
"$QA_PY" -I -B -m mypy
"$QA_PY" -I -B -m mypy --follow-imports=skip \
  bench/world_model_lifecycle/audit_runner.py \
  bench/world_model_lifecycle/artifact.py \
  bench/world_model_lifecycle/artifact_audit.py \
  bench/world_model_lifecycle/adjudication.py \
  bench/world_model_lifecycle/binding.py \
  bench/world_model_lifecycle/experiment.py \
  bench/world_model_lifecycle/launch_bootstrap.py \
  bench/world_model_lifecycle/operator.py \
  bench/world_model_lifecycle/preformal.py \
  bench/world_model_lifecycle/producer_bootstrap.py \
  bench/world_model_lifecycle/restore_eval.py \
  bench/world_model_lifecycle/run.py

"$QA_PY" -I -B - <<'PY'
from pathlib import Path

root = Path("bench/world_model_lifecycle")
forbidden = (
    "allow_" + "tf32",
    "get_float32_" + "matmul_precision",
    "set_float32_" + "matmul_precision",
)
violations = [
    f"{path}:{token}"
    for path in sorted(root.glob("*.py"))
    for token in forbidden
    if token in path.read_text(encoding="utf-8")
]
if violations:
    raise SystemExit("legacy TF32 API remains: " + ", ".join(violations))
PY

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B -W error - <<'PY'
import torch

torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "ieee"
torch.backends.cudnn.rnn.fp32_precision = "ieee"
leaves = (
    torch.backends.cuda.matmul.fp32_precision,
    torch.backends.cudnn.conv.fp32_precision,
    torch.backends.cudnn.rnn.fp32_precision,
)
if leaves != ("ieee", "ieee", "ieee"):
    raise SystemExit("PyTorch fp32_precision leaves are not exact IEEE")
parents = (
    torch.backends.fp32_precision,
    torch.backends.cudnn.fp32_precision,
)
if not all(isinstance(value, str) for value in parents):
    raise SystemExit("PyTorch fp32_precision parent hierarchy is incomplete")
PY

SOURCE_CHECK='from bench.world_model_lifecycle.binding import verify_installed_source_snapshot; verify_installed_source_snapshot()'
"${SAFE_ENV[@]}" "$QA_PY" -I -B -c "$SOURCE_CHECK"
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B -c "$SOURCE_CHECK"
PROSPECT_ROW='import importlib.metadata; from bench.world_model_lifecycle.binding import distribution_sha256,package_roots; distribution=importlib.metadata.distribution("prospect"); print(distribution.version,distribution_sha256("prospect",roots=package_roots()))'
QA_PROSPECT="$("${SAFE_ENV[@]}" "$QA_PY" -I -B -c "$PROSPECT_ROW")"
RUNTIME_PROSPECT="$("${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B -c "$PROSPECT_ROW")"
test "$QA_PROSPECT" = "$RUNTIME_PROSPECT"
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B -c \
  'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1; print(torch.cuda.get_device_name(0))'
```

Prove the intentionally unequal environments and the exact
recorded-versus-live boundary before any runtime seal or development path is
created:

```bash
"$QA_PY" -I -B -c \
  'from importlib.metadata import version; assert version("pytest"); assert version("ruff"); assert version("mypy")'
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B - <<'PY'
from importlib.metadata import PackageNotFoundError, version

assert version("prospect")
for name in ("pytest", "ruff", "mypy"):
    try:
        version(name)
    except PackageNotFoundError:
        continue
    raise AssertionError(f"runtime unexpectedly contains QA-only {name}")
PY

"$QA_PY" -I -B -m pytest -q \
  tests/test_world_model_preformal.py::test_qa_closure_parser_never_reenters_runtime_inventory_verifier \
  tests/test_world_model_preformal.py::test_recorded_report_verifier_uses_explicit_qa_not_caller \
  tests/test_world_model_binding.py::test_recorded_closure_and_coverage_verifiers_ignore_qa_ambient_identity \
  tests/test_world_model_binding.py::test_live_binding_rejects_qa_runtime_package_inventory \
  tests/test_world_model_launch_bootstrap.py::test_recorded_accepted_closure_evidence_is_complete_and_cross_linked \
  tests/test_world_model_launch_bootstrap.py::test_recorded_runtime_conformance_is_cross_linked_to_binding \
  tests/test_world_model_prebinding_audit.py::test_v2_runtime_version_uses_recorded_runtime_not_ambient_qa \
  tests/test_world_model_prebinding_audit.py::test_v2_machine_test_receipt_rejects_command10_nonempty_stderr \
  tests/test_world_model_prebinding_audit.py::test_v2_machine_test_receipt_rejects_command10_identity_mutation
```

The runtime check must pass only because all three QA-only distributions are
absent. The focused tests then prove that recorded QA consumers never call the
live verifier and that the live verifier still rejects an unequal inventory.

Require the protocol verifier to reproduce:

- scientific-block SHA-256
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- 10 current masters, 1,360 current streams, 110 prior masters, and 14,960
  prior streams with zero collision;
- deterministic matrix-contract SHA-256
  `09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`;
- the unchanged four scientific source hashes; and
- the exact sorted full outcome-audit support list:
  `producer_bootstrap.py`, `protocol.json`,
  `schemas/raw-result.schema.json`.

## Create the prospective seal

```bash
mkdir -p "$DEV_ROOT"
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --create-runtime-seal "$RUNTIME_SEAL"
```

The seal and its deterministic marker below `OUTER_ROOT` must be the same inode
with link count two. Any refusal before seal creation may be corrected only
without changing a reviewed byte; otherwise bump the protocol version.

## Result-free, branch-exact rehearsal

Run the exact sealed bootstrap-inventory rehearsal with separate byte captures.
This is a mandatory stop/go gate, not an informal console check:

```bash
COMMAND10_CAPTURE="$(mktemp -d /tmp/prospect-wm001-v112-command10.XXXXXX)"
COMMAND10_STDOUT="$COMMAND10_CAPTURE/stdout.json"
COMMAND10_STDERR="$COMMAND10_CAPTURE/stderr.log"

set +e
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- preformal-runtime bootstrap-inventory-conformance \
  --device cuda \
  >"$COMMAND10_STDOUT" \
  2>"$COMMAND10_STDERR"
COMMAND10_RC=$?
set -e

if [ "$COMMAND10_RC" -ne 0 ] || [ -s "$COMMAND10_STDERR" ]; then
  echo "result-free command 10 failed or wrote stderr; v1.12 is retired" >&2
  wc -c -- "$COMMAND10_STDOUT" "$COMMAND10_STDERR" >&2
  echo "captures retained at $COMMAND10_CAPTURE" >&2
  exit 2
fi

"$QA_PY" -I -B - "$COMMAND10_STDOUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = path.read_bytes()
value = json.loads(payload)
canonical = (
    json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
required = {
    "inventory",
    "inventory_sha256",
    "fresh_runtime_identity_conformance",
    "fresh_runtime_identity_conformance_sha256",
    "conformance_sha256",
}
if payload != canonical:
    raise SystemExit("command-10 stdout is not canonical JSON plus LF")
if (
    value.get("schema") != "prospect.wm001.preformal-runtime-check.v1"
    or value.get("mode") != "bootstrap-inventory-conformance"
    or value.get("device") != "cuda"
    or value.get("passed") is not True
    or not required.issubset(value)
):
    raise SystemExit("command-10 stdout is not the passing CUDA receipt")
PY

test "$(wc -c <"$COMMAND10_STDERR")" -eq 0
test ! -e "$DEV" && test ! -L "$DEV"
rm -rf -- "$COMMAND10_CAPTURE"
unset COMMAND10_CAPTURE COMMAND10_STDOUT COMMAND10_STDERR COMMAND10_RC
```

This command must remain result-free and prove all of the following before
`DEV` exists:

- Gymnasium imports, creates, and closes `Pendulum-v1` without reset or step;
- the exact seven fixed environment values and bootstrap custody survive;
- a separately exec'd `-I -S -B` child reopens the inherited bootstrap and
  runtime-seal descriptors, answers a fresh challenge, and reproduces the
  protocol-bound matrix-contract golden without recursively entering the
  outer launcher or its lock;
- nonexistent absolute startup search entries are dropped and every retained
  standard-library search directory is canonical and inventoried;
- an extant startup search root outside the standard-library tree is rejected;
- private-path and descriptor auditor modes each repeat at least three times
  with byte-identical canonical reports;
- a synthetic development restart-runtime check executes with `source=None`;
- a synthetic formal restart-runtime check executes with a bound source
  manifest and snapshot;
- both use the exact captured support set
  `producer_bootstrap.py`, `protocol.json`,
  `schemas/raw-result.schema.json`;
- development support equals the sealed installed bootstrap source;
- formal support equals the bound source snapshot and implementation row; and
- missing, extra, mutated, or branch-substituted bootstrap support is rejected.

The subprocess must return zero, its stdout must be canonical passing CUDA
receipt JSON, and its separately captured stderr must contain exactly zero
bytes. A benign warning is still a contract failure. Never merge streams,
filter or suppress warnings, or accept the semantic object while discarding
stderr. Any failure after the v1.12 seal retires v1.12; do not repair its
source, wheel, environment, seal, or capture in place.

The runner invocation must also bind the expected bootstrap SHA-256 from the
sealed implementation source. This result-free command emits digest identities
for its reconstructed branch report, execution receipt, and fresh-interpreter
probe. Preformal command 10 later retains that output, while binding retains
the complete branch report and all three-path plus three-descriptor receipt
rows. The binding is valid only if its rebuilt audit-execution block matches
the retained preformal evidence exactly.

Only the initial Gymnasium smoke subcheck is forbidden to reset or step.
The auditor conformance subchecks intentionally perform isolated QA-only
resets and steps, and tests may create temporary synthetic fixtures. Neither
path may collect experiment experience, train a model, write or read a real
experiment result, inspect real K3–K6 evidence, or create either canonical
v1.12 producer root. Verify the exact real-subprocess branch tests explicitly
if they are not already a separate command row:

```bash
"$QA_PY" -I -B -m pytest -q \
  tests/test_world_model_artifact_audit.py \
  tests/test_world_model_audit_runner.py \
  tests/test_world_model_operator.py \
  tests/test_world_model_prebinding_audit.py
```

A shell-side invocation correction is permitted only when no sealed rehearsal
process was launched. Once the command-10 launcher starts, any refusal,
nonzero return, or nonempty stderr consumes and retires v1.12; do not repeat
the rehearsal. Any source, environment, dependency, protocol, schema, seed,
seal, or support change requires a new version and fresh paths.

## Sole development qualification

The command below has no seed override. Exclusive creation of `DEV` consumes
the v1.12 development qualification even if any later phase fails.

```bash
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- development \
  --device cuda \
  --output "$DEV"
```

Require the producer terminal and its deterministic outer completion to be the
same inode. Never resume, overwrite, move into, rename, or create a sibling of
`DEV`.

Run the only development audit:

```bash
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- --audit-entry development \
  --producer "$DEV" \
  --output "$DEV_AUDIT"
```

Before closure, strictly verify that its runtime and reproduction manifests
contain the exact captured `producer_bootstrap.py` support identity and that
the accepted report reproduces byte-for-byte. A rejected, failed, unfinalized,
or missing ordinary audit retires v1.12; never rerun it.

Close development once:

```bash
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- --closure-entry \
  --producer "$DEV" \
  --audit-attempt "$DEV_AUDIT" \
  --output "$CLOSURE_ATTEMPT"
```

The accepted, outer-finalized closure attempt must create `DEV_CLOSURE`
exclusively. After `DEV` was created, any producer, audit, closure, preformal,
binding, or stop/go failure retires v1.12. Absence of a downstream path never
restores authorization.

Before publishing the canonical marker, the closure writer must complete an
unretained fresh-interpreter reopen of the prospective marker. After marker
publication, the still-open closure transaction must run a second fresh sealed
interpreter and retain its challenge-bound report as
`fresh-runtime-reopen.json` before publishing and outer-finalizing the accepted
attempt. Now perform the post-finalization sealed reopen; this validates the
marker, retained second receipt, accepted attempt, and same-inode outer
completion under the bound runtime:

```bash
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- preformal-runtime accepted-closure-evidence \
  --development-closure "$DEV_CLOSURE" \
  --closure-attempt "$CLOSURE_ATTEMPT"
```

Do not substitute a direct `$QA_PY` import here. Closure verification is
intentionally bound to the live producer executable, flags, environment,
package root, dependency inventory, and captured seal descriptors.

## Preformal report and binding

Recheck that the canonical version-owned bundle and deterministic hidden claim
are absent. `generate-report` itself exclusively creates
`$DEV_ROOT/v1.12.0/.preformal.staging`, then fsyncs its parent before its first
command; either that hidden claim or the final bundle consumes the one-shot
attempt. Operator namespaces and their deterministic hidden claims are also
created durably, with each new directory entry committed through its parent.
Do not create either preformal path during setup, QA, rehearsal, production,
audit, or closure.

```bash
require_absent "$PREFORMAL_ROOT"
require_absent "$DEV_ROOT/v1.12.0/.preformal.staging"

"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal generate-report \
  --output "$PREFORMAL" \
  --runtime-executable "$RUNTIME_PY" \
  --runtime-seal "$RUNTIME_SEAL" \
  --development-closure "$DEV_CLOSURE" \
  --closure-attempt "$CLOSURE_ATTEMPT" \
  --prospective-review "$REVIEW" \
  --device cuda

"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal verify-report \
  --report "$PREFORMAL"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B -c \
  'import sys; from pathlib import Path; from bench.world_model_lifecycle.binding import verify_machine_test_report; report=verify_machine_test_report(Path(sys.argv[1])); assert report["all_pass"] is True' \
  "$PREFORMAL"
```

The report must contain exactly ten ordered command rows—eight QA and two
sealed-runtime rows—with exactly 20 separate stdout/stderr logs. The generator
runs all ten commands while `PREFORMAL_ROOT` remains absent, stages and fsyncs
all 21 files under the hidden claim, and atomically publishes the complete
directory with no replacement. It prints a
`prospect.wm001.preformal-test-report-generation.v2` envelope and exits nonzero
with `passed: false` for an exact command failure, identity check failure, or
semantic accepted-closure/runtime-conformance failure. The report and logs
remain terminal evidence in that case. A write fault or interruption may leave
only the hidden claim and no envelope; that is still terminal. Never remove
either path or retry v1.12. A successful generation is not sufficient by
itself: the separate `verify-report` command above must also pass.

Command 9 must be exactly `runtime-accepted-closure-evidence`. Its canonical
JSON stdout and empty stderr must bind the canonical closure and accepted
closure attempt semantically, not merely return zero. Its seven captured input
identities are `development_closure`, `closure_attempt_terminal`,
`closure_outer_completion`, `runtime_seal`, `launch_bootstrap`,
`producer_bootstrap`, and `prospective_review`; the two closure-attempt paths
must be two links to the same inode. No public `development-evidence` runtime
mode is permitted. The captured outcome-audit runtime must list the three exact
support files and bind the branch-exact result-free conformance evidence.
Command 10 must also have zero exit, `passed: true`, and empty stderr. Its
canonical stdout must carry the complete runtime package/root/
standard-library/ownership inventory and complete fresh-child identity receipt,
with both advertised digests exactly recomputable. QA verification may parse
and cross-link those recorded objects but may not replay live runtime inventory
against the QA interpreter. The read-only runtime-side verifier above must also
accept the report before the binding one-shot claim exists; it reopens the
report's explicit QA executable and closure instead of substituting the
caller's intentionally smaller runtime environment.

Create the only binding attempt:

```bash
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- --binding-entry \
  --output "$BINDING_ATTEMPT" \
  --test-report "$PREFORMAL" \
  --development-closure "$DEV_CLOSURE" \
  --closure-attempt "$CLOSURE_ATTEMPT" \
  --device cuda

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify binding "$BINDING"
```

The binding is valid only inside that accepted, outer-finalized attempt.
Binding creation must run the exact independent formal-input consumer and
publish `$BINDING_ATTEMPT/formal-input-preflight.json`. Attempt verification
reruns that consumer and requires the retained receipt to match exactly. The
standard-library outer launcher requires that terminal-bound receipt before it
imports the formal producer and cross-checks its binding, report, closure, and
bound auditor identities. It reopens the canonical stdout for command 9 and
command 10, validates command 10's empty stderr and full inventory/fresh-child
objects, requires that inventory to equal the binding dependencies, and
requires each semantic-object digest to equal the corresponding receipt digest.
Commands 1–8 are authorized by the accepted binding and its terminal-bound
preflight receipt; the standard-library launcher does not duplicate their full
QA report verifier. Its additional direct report parsing is deliberately
limited to the two sealed-runtime authorization commands.
The receipt must have schema
`prospect.wm001.formal-input-preflight.v1`, protocol
version `1.12.0`, `passed: true`, and digests for the binding, preformal report,
development closure, accepted-closure evidence, runtime conformance, and
auditor source. Any missing, false, stale, or mismatched receipt retires v1.12.
Confirm that its formal outcome runtime captures
`producer_bootstrap.py` from the bound source snapshot, not from a live
repository or whichever installed package is importable.

## Final stop/go gate

```bash
require_absent "$FORMAL_MARKER"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m bench.world_model_lifecycle.verify result \
  "$DEV/result.json"
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal verify-report \
  --report "$PREFORMAL"
"$QA_PY" -I -B -m bench.world_model_lifecycle.verify binding "$BINDING"

"$QA_PY" -I -B - "$BINDING" "$FORMAL_INPUT_PREFLIGHT" <<'PY'
import json
import sys
from pathlib import Path

from bench.world_model_lifecycle.artifact_audit import (
    preflight_formal_input_package,
)

binding = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
expected = preflight_formal_input_package(binding)
expected_bytes = (
    json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
).encode("utf-8")
if receipt_path.read_bytes() != expected_bytes:
    raise SystemExit("formal-input preflight receipt differs from exact rerun")
if expected["passed"] is not True:
    raise SystemExit("formal-input preflight did not pass")
print(expected["schema"], expected["binding_sha256"])
PY

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- preformal-runtime accepted-closure-evidence \
  --development-closure "$DEV_CLOSURE" \
  --closure-attempt "$CLOSURE_ATTEMPT"

"$QA_PY" -I -B -c \
  'import sys; from pathlib import Path; from bench.world_model_lifecycle.artifact import verify_producer_manifest; from bench.world_model_lifecycle.operator import verify_operator_attempt,verify_outer_completion; producer=Path(sys.argv[1]); verify_producer_manifest(producer); verify_outer_completion(producer/"producer-manifest.json"); verify_operator_attempt(Path(sys.argv[2])); verify_outer_completion(Path(sys.argv[3])); verify_operator_attempt(Path(sys.argv[4]))' \
  "$DEV" "$DEV_AUDIT" "$RUNTIME_SEAL" "$BINDING_ATTEMPT"
```

Stop unless every input remains byte-identical to the binding, every required
outer completion is same-inode, the formal marker and canonical formal path
are absent, the retained formal-input preflight receipt equals a fresh run of
the independent consumer, and the accepted binding attempt verifier also
accepts that receipt. The sole formal invocation below must perform its
descriptor-mode conformance replay
and require equality with the bound manifest and report before publishing the
formal marker.

## Sole formal producer

```bash
BINDING_SHA="$("$QA_PY" -I -B -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$BINDING")"
FORMAL_BINDING_ROOT="$REPO/bench/world_model_lifecycle/results/formal/$BINDING_SHA"
FORMAL="$FORMAL_BINDING_ROOT/confirmation-v1.12.0"
require_absent "$FORMAL_BINDING_ROOT"
require_absent "$FORMAL"
require_absent "$FORMAL_MARKER"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- formal \
  --device cuda \
  --binding "$BINDING" \
  --output "$FORMAL"

cmp -- "$FORMAL_INPUT_PREFLIGHT" "$FORMAL/formal-input-preflight.json"
```

If both `FORMAL_BINDING_ROOT` and `FORMAL_MARKER` remain absent after a
pre-creation invocation refusal, only an invocation correction that changes no
bound byte may be considered. Any child below `FORMAL_BINDING_ROOT`, including
a noncanonical sibling left by an earlier launcher, contaminates and consumes
the v1.12 attempt. If either canonical path exists, the attempt is likewise
consumed. Never resume or rerun it. A nonzero return after marker publication
is terminal evidence. A completed formal producer must preserve the exact
binding-attempt preflight receipt at its root; the independent formal auditor
reconstructs the live authorization lineage and rejects a missing or
byte-different copy.

## One formal audit and one adjudication

```bash
require_absent "$FORMAL_AUDIT"
require_absent "$FORMAL_AUDIT_MARKER"

set +e
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- --audit-entry formal \
  --producer "$FORMAL" \
  --output "$FORMAL_AUDIT"
FORMAL_AUDIT_RC=$?
set -e
case "$FORMAL_AUDIT_RC" in
  0|1|2) ;;
  *) exit "$FORMAL_AUDIT_RC" ;;
esac
if [ ! -d "$FORMAL_AUDIT" ]; then
  echo "formal audit failed before publishing an authenticated attempt; v1.12 is retired" >&2
  exit 2
fi
"$QA_PY" -I -B -c \
  'import os,sys; from pathlib import Path; from bench.world_model_lifecycle.operator import inspect_unfinalized_operator_attempt,outer_completion_marker,verify_operator_attempt; attempt=Path(sys.argv[1]); terminal=attempt/"operator-attempt.json"; (verify_operator_attempt if os.path.lexists(outer_completion_marker(terminal)) else inspect_unfinalized_operator_attempt)(attempt)' \
  "$FORMAL_AUDIT"
```

Only an authenticated canonical finalized or explicitly unfinalized attempt
establishes that the sole v1.12 formal-audit claim was consumed, even if no
ordinary report was emitted. Do not rerun it. A refusal that does not publish
an authentic attempt is terminal at this post-formal stage rather than
silently treated as evidence. Inspect the authenticated attempt and create one
independent semantic-review object from the supported read-only inspector:

```bash
mkdir -p "$(dirname "$SEMANTIC_REVIEW")"
"$QA_PY" -I -B -c \
  'import json,sys; from pathlib import Path; from bench.world_model_lifecycle.adjudication import inspect_adjudication_evidence; print(json.dumps(inspect_adjudication_evidence(Path(sys.argv[1])),sort_keys=True,separators=(",",":")))' \
  "$FORMAL_AUDIT"
```

The inspector output is a construction aid, not a semantic review or an
automatic verdict. Do not copy the whole inspector object:
`required_verdict` and `execution_failure_record` are helper-only fields and
are forbidden in semantic-review v2. Independently inspect the official
evidence, then write one canonical UTF-8, sorted compact JSON object plus LF at
`SEMANTIC_REVIEW` with exactly these 17 fields:

```text
schema
experiment_id
protocol_version
assurance
evidence_kind
artifact_root
result_sha256
audit_attempt_path
audit_attempt_manifest_sha256
formal_audit_claim_sha256
independent_audit_sha256
execution_failure_sha256
reviewer
reviewed_gates
verdict
fatal_findings
conclusion
```

Copy the first 12 identity fields exactly from the inspector. Copy
`reviewed_gates` exactly from it. Author `reviewer`, `verdict`,
`fatal_findings`, and `conclusion` from the independent judgment. An accepted
report requires verdict `accepted` and no fatal findings. A rejected report or
execution failure requires verdict `rejected` and at least one fatal finding;
an execution failure also requires an empty `reviewed_gates`.

Set the requested disposition to that independently authored verdict, then
strictly preflight the exact review without creating an adjudication claim:

```bash
: "${DISPOSITION:?set only after independent semantic review}"
case "$DISPOSITION" in
  accepted|rejected) ;;
  *) echo "invalid disposition" >&2; exit 2 ;;
esac

"$QA_PY" -I -B -c \
  'import sys; from pathlib import Path; from bench.world_model_lifecycle.adjudication import verify_semantic_review_for_adjudication; review=verify_semantic_review_for_adjudication(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]); print(review["verdict"])' \
  "$FORMAL_AUDIT" "$SEMANTIC_REVIEW" "$DISPOSITION"
```

Only after that read-only verifier succeeds may the one-shot adjudicator run:

```bash
set +e
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- --adjudication-entry \
  --audit-attempt "$FORMAL_AUDIT" \
  --semantic-review "$SEMANTIC_REVIEW" \
  --disposition "$DISPOSITION"
ADJUDICATION_RC=$?
set -e
case "$ADJUDICATION_RC" in
  0|1) ;;
  2) echo "Adjudication refused; inspect state without retry" >&2; exit 2 ;;
  *) exit "$ADJUDICATION_RC" ;;
esac
```

An ordinary audit report receives exactly one byte-equality replay. An
authenticated no-report failure receives zero replay and can only be rejected.
The adjudication claim is single-use. Do not run the following recovery block
after an ordinary terminal adjudication. It is a conditional alternative only
when the ordinary command was interrupted after claim-marker or hidden-staging
publication and before an outer-finalized package appeared. Recovery never
invokes the runner:

```bash
set +e
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- --adjudication-entry --recover
RECOVERY_RC=$?
set -e
case "$RECOVERY_RC" in
  0|1) ;;
  *) exit "$RECOVERY_RC" ;;
esac
```

Strictly verify the exact outer-finalized package:

```bash
"$QA_PY" -I -B -c \
  'import sys; from pathlib import Path; from bench.world_model_lifecycle.adjudication import verify_adjudication_package; manifest=verify_adjudication_package(Path(sys.argv[1])); print(manifest["disposition"])' \
  "$ADJUDICATION_PACKAGE"
```

Only a verified package whose printed disposition is `accepted` may support the
bounded WM-001 claim. Every other terminal state retires v1.12 without
same-version repair or retry.
