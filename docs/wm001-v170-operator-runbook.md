# WM-001 v1.7.0 operator runbook

Status: prospective. Seal this runbook with the exact v1.7 protocol,
implementation, schemas, lock, tests, and independent review before any v1.7
producer path is created.

Protocol 1.6.0 is retired. Its completed development producer is not reusable:
the sole canonical development audit ended as outer-finalized failure evidence
because `producer_bootstrap.py` was not present in the captured auditor support
manifest. There is no accepted v1.6 audit, closure, binding, or formal launch.
Do not inspect, summarize, compare, or use any v1.6 K3–K6 value.

## Fixed paths and clean launcher environment

```bash
set -euo pipefail

REPO="$(pwd -P)"
test "$REPO" = "$(git rev-parse --show-toplevel)"

QA_PY="/home/alex/.venvs/prospect-wm001-v17/bin/python"
RUNTIME_PY="/home/alex/.venvs/prospect-wm001-v17-runtime/bin/python"
LAUNCH="$REPO/bench/world_model_lifecycle/launch_bootstrap.py"
BOOTSTRAP="$REPO/bench/world_model_lifecycle/producer_bootstrap.py"

DEV_ROOT="$REPO/bench/world_model_lifecycle/results/development"
DEV="$DEV_ROOT/qualification-v1.7.0"
RUNTIME_SEAL="$DEV_ROOT/runtime-seal-v1.7.0.json"
DEV_CLOSURE="$DEV_ROOT/development-closure-v1.7.0.json"
PREFORMAL="$DEV_ROOT/preformal-test-report-v1.7.0.json"

OPERATOR_ROOT="$REPO/bench/world_model_lifecycle/results/operator-v1.7"
DEV_AUDIT="$OPERATOR_ROOT/audits/development-audit-v1.7.0"
CLOSURE_ATTEMPT="$OPERATOR_ROOT/closures/development-closure-v1.7.0"
BINDING_ATTEMPT="$OPERATOR_ROOT/bindings/formal-binding-v1.7.0"
BINDING="$BINDING_ATTEMPT/formal-binding.json"

OUTER_ROOT="$REPO/bench/world_model_lifecycle/results/outer-completions/v1.7"
FORMAL_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-launch-v1.7.0.json"
FORMAL_AUDIT_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-audit-v1.7.0.json"
ADJUDICATION_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-adjudication-v1.7.0.json"

REVIEW="$REPO/docs/wm001-v170-prospective-harness-review.json"
SEMANTIC_REVIEW="$REPO/artifacts/wm001-reviews/formal-v1.7.0.json"

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
```

The development masters are exactly `3920043614, 3703229797`. The formal
masters are exactly `2080036362, 865871218, 3636713390, 2195564811,
2000167339, 329754669, 4064290468, 1911057116`. The verifier must report 60
unique prior masters and 8,160 unique prior streams before any evidence run.

## Build the two environments once

Create both environments at previously absent paths. Build one reviewed wheel
and install that same wheel non-editably. Never reinstall into the runtime
environment after its inventory is rendered.

```bash
BASE_PY="/home/alex/miniconda3/bin/python"
QA_ENV="${QA_PY%/bin/python}"
RUNTIME_ENV="${RUNTIME_PY%/bin/python}"
WHEELHOUSE="$(mktemp -d /tmp/prospect-wm001-v17-wheelhouse.XXXXXX)"

test ! -e "$QA_ENV"
test ! -e "$RUNTIME_ENV"
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
generated header and Prospect distribution identity must name v1.7.0. Verify
the lock immediately, then do not install anything else into the runtime.

```bash
LOCK_TMP="$(mktemp "$REPO/.requirements-wm001.lock.XXXXXX")"
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -B - "$LOCK_TMP" <<'PY'
from pathlib import Path
import sys

from bench.world_model_lifecycle.binding import installed_package_rows

rows = installed_package_rows()
lines = [
    "# WM-001 protocol 1.7.0 isolated live execution closure.",
    "# Versions preserve the v1.4 scientific runtime; each v1.7 digest covers",
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
`wm001-v170-prospective-harness-review.json`, then commit the whole candidate.
Do not rebuild, reinstall, or change a bound byte after that commit.

## Pre-outcome static gates

All version-scoped paths below must be absent:

```bash
test ! -e "$DEV"
test ! -e "$DEV_AUDIT"
test ! -e "$DEV_CLOSURE"
test ! -e "$CLOSURE_ATTEMPT"
test ! -e "$PREFORMAL"
test ! -e "$BINDING_ATTEMPT"
test ! -e "$FORMAL_MARKER"
test ! -e "$FORMAL_AUDIT_MARKER"
test ! -e "$ADJUDICATION_MARKER"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m pytest -q
"$QA_PY" -I -B -m ruff check src/prospect bench tests
"$QA_PY" -I -B -m mypy
```

Require the protocol verifier to reproduce:

- scientific-block SHA-256
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- 10 current masters, 1,360 current streams, 60 prior masters, and 8,160
  prior streams with zero collision;
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

Run the sealed bootstrap-inventory rehearsal:

```bash
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- preformal-runtime bootstrap-inventory-conformance \
  --device cuda
```

This command must remain result-free and prove all of the following before
`DEV` exists:

- Gymnasium imports, creates, and closes `Pendulum-v1` without reset or step;
- the exact seven fixed environment values and bootstrap custody survive;
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

The runner invocation must also bind the expected bootstrap SHA-256 from the
sealed implementation source. Preserve the canonical branch report and every
row of the three-path plus three-descriptor execution receipt. The later
formal binding is valid only if its rebuilt audit-execution block is
byte-identical in canonical identity to this command's sealed rehearsal.

The rehearsal and its tests may not reset or step a task, collect experience,
train, read an earlier result, inspect K3–K6, or create a producer root. Verify
the exact real-subprocess branch tests explicitly if they are not already a
separate command row:

```bash
"$QA_PY" -I -B -m pytest -q \
  tests/test_world_model_artifact_audit.py \
  tests/test_world_model_audit_runner.py \
  tests/test_world_model_operator.py \
  tests/test_world_model_prebinding_audit.py
```

Only a byte-identical invocation correction may repeat a refused rehearsal.
Any source, environment, dependency, protocol, schema, seed, seal, or support
change requires a new version and fresh paths.

## Sole development qualification

The command below has no seed override. Exclusive creation of `DEV` consumes
the v1.7 development qualification even if any later phase fails.

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
or missing ordinary audit retires v1.7; never rerun it.

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
binding, or stop/go failure retires v1.7. Absence of a downstream path never
restores authorization.

## Preformal report and binding

```bash
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal generate-report \
  --output "$PREFORMAL" \
  --runtime-executable "$RUNTIME_PY" \
  --runtime-seal "$RUNTIME_SEAL" \
  --development-closure "$DEV_CLOSURE" \
  --prospective-review "$REVIEW" \
  --device cuda

"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal verify-report \
  --report "$PREFORMAL"
```

The report must contain exactly ten ordered command rows—eight QA and two
sealed-runtime rows—with exactly 20 separate stdout/stderr logs. Its captured
outcome-audit runtime must list the three exact support files and must bind the
branch-exact result-free conformance evidence.

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
Confirm that its formal outcome runtime captures
`producer_bootstrap.py` from the bound source snapshot, not from a live
repository or whichever installed package is importable.

## Final stop/go gate

```bash
test ! -e "$FORMAL_MARKER"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m bench.world_model_lifecycle.verify result \
  "$DEV/result.json"
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal verify-report \
  --report "$PREFORMAL"
"$QA_PY" -I -B -m bench.world_model_lifecycle.verify binding "$BINDING"

"$QA_PY" -I -B -c \
  'import sys; from pathlib import Path; from bench.world_model_lifecycle.artifact import verify_producer_manifest; from bench.world_model_lifecycle.binding import verify_development_closure; from bench.world_model_lifecycle.operator import verify_operator_attempt,verify_outer_completion; producer=Path(sys.argv[1]); verify_producer_manifest(producer); verify_outer_completion(producer/"producer-manifest.json"); [verify_operator_attempt(Path(value)) for value in sys.argv[2:5]]; verify_development_closure(Path(sys.argv[5])); verify_outer_completion(Path(sys.argv[6]))' \
  "$DEV" "$DEV_AUDIT" "$CLOSURE_ATTEMPT" "$BINDING_ATTEMPT" \
  "$DEV_CLOSURE" "$RUNTIME_SEAL"
```

Stop unless every input remains byte-identical to the binding, every required
outer completion is same-inode, the formal marker and canonical formal path
are absent, and the formal descriptor-mode conformance replay equals its bound
manifest and report.

## Sole formal producer

```bash
BINDING_SHA="$("$QA_PY" -I -B -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$BINDING")"
FORMAL="$REPO/bench/world_model_lifecycle/results/formal/$BINDING_SHA/confirmation-v1.7.0"
test ! -e "$FORMAL"
test ! -e "$FORMAL_MARKER"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- formal \
  --device cuda \
  --binding "$BINDING" \
  --output "$FORMAL"
```

If both `FORMAL` and `FORMAL_MARKER` remain absent after a pre-creation
invocation refusal, only an invocation correction that changes no bound byte
may be considered. If either path exists, the formal attempt is consumed.
Never resume or rerun it. A nonzero return after marker publication is
terminal evidence.

## One formal audit and one adjudication

```bash
FORMAL_AUDIT="$OPERATOR_ROOT/audits/formal-audit-v1.7.0"
test ! -e "$FORMAL_AUDIT"
test ! -e "$FORMAL_AUDIT_MARKER"

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
```

This consumes the sole v1.7 formal-audit claim even if no ordinary report is
emitted. Do not rerun it. Inspect the canonical attempt and create one
independent semantic-review object from the supported read-only inspector:

```bash
"$QA_PY" -I -B -c \
  'import json,sys; from pathlib import Path; from bench.world_model_lifecycle.adjudication import inspect_adjudication_evidence; print(json.dumps(inspect_adjudication_evidence(Path(sys.argv[1])),sort_keys=True,separators=(",",":")))' \
  "$FORMAL_AUDIT"

: "${DISPOSITION:?set only after independent semantic review}"
case "$DISPOSITION" in
  accepted|rejected) ;;
  *) echo "invalid disposition" >&2; exit 2 ;;
esac

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
The adjudication claim is single-use. Use sealed recovery only for a documented
claim-marker/hidden-staging interruption; recovery never invokes the runner:

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

The final package path is:

```text
bench/world_model_lifecycle/results/adjudication-v1.7/formal-adjudication-v1.7.0
```

Strictly verify that exact outer-finalized package. Only a verified accepted
package may support the bounded WM-001 claim. Every other terminal state
retires v1.7 without same-version repair or retry.
