# WM-001 v1.5.0 operator runbook

This is the executable ordering contract for the one v1.5 confirmation. It
does not authorize a retry. Run every command from the canonical repository
root, with no other process writing the repository, environments, or WM-001
results roots.

The repository-wide formal marker is:

```text
bench/world_model_lifecycle/results/formal/formal-launch-v1.5.0.json
```

If it exists, do not run a formal command again.

The outcome-free failed development preflight at
`results/development/qualification-v1.5.0` and its finalized
`runtime-seal-v1.5.0.json` are retained and must not be changed. The commands
below use fresh `*-attempt-2` paths after binding TorchRL's
`LAZY_LEGACY_OP=False` default.

## Fixed paths and safe launcher

The QA and runtime environments must be separate, dedicated, non-editable
virtual environments. The runtime environment must contain only the exact
marker-selected dependency closure of the Prospect wheel and must not contain
an editable install or inherit system/user site packages.

```bash
set -euo pipefail

REPO="$(pwd -P)"
QA_PY="/home/alex/.venvs/prospect-wm001-v15/bin/python"
RUNTIME_PY="/home/alex/.venvs/prospect-wm001-v15-runtime/bin/python"
LAUNCH="$REPO/bench/world_model_lifecycle/launch_bootstrap.py"
BOOTSTRAP="$REPO/bench/world_model_lifecycle/producer_bootstrap.py"
RUNTIME_SEAL="$REPO/bench/world_model_lifecycle/results/development/runtime-seal-v1.5.0-attempt-2.json"
DEV_ROOT="$REPO/bench/world_model_lifecycle/results/development"
OPERATOR_ROOT="$REPO/bench/world_model_lifecycle/results/operator-v1.5"
OUTER_ROOT="$REPO/bench/world_model_lifecycle/results/outer-completions/v1.5"

SAFE_ENV=(
  env -i
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  LAZY_LEGACY_OP=False
  LC_ALL=C.UTF-8
  PATH=/usr/bin:/bin
  TZ=UTC
)
```

## Build the two final environments once

Preserve the v1.4 package versions, but regenerate every distribution digest
under the v1.5 distribution-hash semantics. Build one final wheel and install
that same wheel into two previously absent environments. Never reinstall into
the runtime environment: even `pip --no-compile` may create bytecode while
inspecting an already installed target.

```bash
BASE_PY="/home/alex/miniconda3/bin/python"
QA_ENV="${QA_PY%/bin/python}"
RUNTIME_ENV="${RUNTIME_PY%/bin/python}"
WHEELHOUSE="$(mktemp -d /tmp/prospect-wm001-v15-wheelhouse.XXXXXX)"

test ! -e "$QA_ENV"
test ! -e "$RUNTIME_ENV"

mapfile -t RUNTIME_PINS < <(
  sed -nE \
    '/^(python|prospect)==/d; s/^(.*==[^ ]+) --distribution-sha256=.*/\1/p' \
    "$REPO/requirements-wm001.lock"
)
test "${#RUNTIME_PINS[@]}" -eq 43

"$BASE_PY" -m pip wheel --no-deps --wheel-dir "$WHEELHOUSE" "$REPO"
mapfile -t WHEELS < <(find "$WHEELHOUSE" -maxdepth 1 -type f -name 'prospect-*.whl' -print)
test "${#WHEELS[@]}" -eq 1
WHEEL="${WHEELS[0]}"

"$BASE_PY" -m venv "$QA_ENV"
"$BASE_PY" -m pip --python "$QA_PY" install \
  --no-compile --only-binary=:all: \
  "${WHEEL}[runtime,dev]" "${RUNTIME_PINS[@]}"

"$BASE_PY" -m venv --without-pip "$RUNTIME_ENV"
"$BASE_PY" -m pip --python "$RUNTIME_PY" install \
  --no-compile --only-binary=:all: \
  "${WHEEL}[runtime]" "${RUNTIME_PINS[@]}"
```

Render the v1.5 lock from that untouched runtime. This construction step runs
with bytecode disabled; evidence execution later uses the sealed
`-I -S -B` bootstrap.

```bash
LOCK_TMP="$(mktemp "$REPO/.requirements-wm001.lock.XXXXXX")"
"$RUNTIME_PY" -B - "$LOCK_TMP" <<'PY'
from pathlib import Path
import sys

from bench.world_model_lifecycle.binding import installed_package_rows

rows = installed_package_rows()
lines = [
    "# WM-001 protocol 1.5.0 isolated live execution closure.",
    "# Versions preserve the v1.4 scientific runtime; each v1.5 digest covers",
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

"$RUNTIME_PY" -B -c \
  'from bench.world_model_lifecycle.binding import installed_package_rows,verify_lockfile_rows; verify_lockfile_rows(installed_package_rows())'
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal --help >/dev/null
```

Before authoring the prospective review, stage every intended new lifecycle
module and test so `git ls-files` includes it. The review must then be authored
against the exact staged implementation manifest, staged itself, and committed
with the rest of the candidate. Do not rebuild or reinstall after that commit;
the executable bytes were already fixed by the one wheel.

The outer launcher derives the repository and its one nonblocking runtime lock
from `BOOTSTRAP`, not from caller-controlled working-directory text. It runs
the child under `-I -S -B`, verifies the typed runtime seal, and commits a
publisher's terminal only after a physical child exit of zero and complete
receipt/input rechecks.

## Pre-outcome repository and environment checks

Before producing evidence:

```bash
test ! -e "$REPO/bench/world_model_lifecycle/results/formal/formal-launch-v1.5.0.json"
git status --short --untracked-files=all
"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m pytest -q
"$QA_PY" -I -B -m ruff check src/prospect bench tests
"$QA_PY" -I -B -m mypy
```

The worktree must be clean. The QA environment must contain the final
non-editable Prospect wheel. Its installed
`bench/world_model_lifecycle/preformal.py` must be byte-identical to the live
reviewed source; the preformal verifier checks this under isolated `-I`
execution. The protocol verifier must reproduce the fixed scientific-block
digest and source-kernel digests. The exact runtime distribution rows must
equal `requirements-wm001.lock`; the runtime must have no `pip` distribution,
bytecode, bytecode-cache directory, editable distribution, unowned file, or
system-site inheritance.

Create the prospective runtime seal exactly once:

```bash
mkdir -p "$DEV_ROOT"
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --create-runtime-seal "$RUNTIME_SEAL"
```

The command succeeds only if the repository is clean. The seal and its
deterministic file in `OUTER_ROOT` must be the same inode with link count two.

## One development qualification

Choose one new direct child of the development root and run the fixed two
development seeds without overrides:

```bash
DEV="$DEV_ROOT/qualification-v1.5.0-attempt-2"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- development \
  --device cuda \
  --output "$DEV"
```

The development producer terminal is
`$DEV/producer-manifest.json`. It is public evidence only after the outer
launcher creates its deterministic same-inode completion marker.

Run the one canonical development audit attempt:

```bash
DEV_AUDIT="$OPERATOR_ROOT/audits/development-audit-v1.5.0"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- --audit-entry development \
  --producer "$DEV" \
  --output "$DEV_AUDIT"
```

Close development with the accepted audit attempt:

```bash
DEV_CLOSURE="$DEV_ROOT/development-closure-v1.5.0.json"
CLOSURE_ATTEMPT="$OPERATOR_ROOT/closures/development-closure-v1.5.0"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- --closure-entry \
  --producer "$DEV" \
  --audit-attempt "$DEV_AUDIT" \
  --output "$CLOSURE_ATTEMPT"
```

After `DEV_CLOSURE` exists, no v1.5 development run or audit is permitted.

## Exact preformal report and canonical binding

The independent prospective review is:

```text
docs/wm001-v150-prospective-harness-review.json
```

It must accept the exact implementation manifest with no unresolved blocker.
Generate the preformal report in the QA environment:

```bash
PREFORMAL="$DEV_ROOT/preformal-test-report-v1.5.0.json"

"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal generate-report \
  --output "$PREFORMAL" \
  --runtime-executable "$RUNTIME_PY" \
  --runtime-seal "$RUNTIME_SEAL" \
  --development-closure "$DEV_CLOSURE" \
  --prospective-review "$REPO/docs/wm001-v150-prospective-harness-review.json" \
  --device cuda

"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal verify-report \
  --report "$PREFORMAL"
```

The report must contain exactly ten ordered command rows: eight QA rows and two
sealed-runtime rows, with exactly 20 separate stdout/stderr logs. Its QA
closure and runtime custody/inventory snapshots must agree before and after.

Create the only binding attempt:

```bash
BINDING_ATTEMPT="$OPERATOR_ROOT/bindings/formal-binding-v1.5.0"
BINDING="$BINDING_ATTEMPT/formal-binding.json"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$RUNTIME_SEAL" \
  -- --binding-entry \
  --output "$BINDING_ATTEMPT" \
  --test-report "$PREFORMAL" \
  --development-closure "$DEV_CLOSURE" \
  --closure-attempt "$CLOSURE_ATTEMPT" \
  --device cuda

"$QA_PY" -m bench.world_model_lifecycle.verify binding "$BINDING"
```

The formal binding remains singly linked inside the accepted attempt. Formal
authorization comes from the exact binding bytes plus the sibling
`operator-attempt.json` and its same-inode outer completion. A binding copy is
invalid.

## Final stop/go check

Before formal launch, repeat all read-only repository checks and verify:

- the formal marker is absent;
- the worktree is clean and still equals the bound commit/tree;
- the prospective runtime seal, development producer, development audit,
  closure attempt, preformal report, and binding attempt all pass strict public
  verification;
- every required outer-completion marker is present and same-inode;
- no unreviewed source, lock, protocol, schema, documentation, or environment
  change occurred after the prospective review and binding.

Any failure stops the confirmation. Fixing it requires a new protocol version
if it changes a closed qualification, bound source, dependency, environment,
protocol, schema, or prospective review. A command refused before publishing
its single-use claim or formal marker has not consumed that claim, but the
operator must prove the marker is absent and must not change bound state before
correcting a purely invocational error.

## Exactly one formal producer

Use the sole canonical direct child of
`results/formal/<formal-binding-sha256>/`. Never invoke this command after the
version-wide marker exists:

```bash
BINDING_SHA="$("$QA_PY" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$BINDING")"
FORMAL="$REPO/bench/world_model_lifecycle/results/formal/$BINDING_SHA/confirmation-v1.5.0"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- formal \
  --device cuda \
  --binding "$BINDING" \
  --output "$FORMAL"
```

Immediately before the first possible environment reset, the producer creates
`formal-launch.json` and hard-links it as the version-wide formal marker. On
return, the outer launcher commits `producer-manifest.json`. A nonzero logical
return after marker publication is terminal evidence, not permission to retry.
A refusal while the marker is still absent is not a formal attempt; inspect and
correct only the invocation without changing any bound byte.

## One official audit and terminal adjudication

The formal audit path and claim are fixed:

```bash
FORMAL_AUDIT="$OPERATOR_ROOT/audits/formal-audit-v1.5.0"

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

This consumes the v1.5 formal-audit claim even if no ordinary report can be
emitted. If its version-wide claim marker exists, do not rerun it. Inspect the
canonical attempt even when the logical return is nonzero.

Create the version-2 independent semantic review against that exact audit
attempt. The supported read-only inspector supplies the exact identity fields
and whether a report can be accepted or must be rejected:

```bash
"$QA_PY" -I -B -c \
  'import json,sys; from pathlib import Path; from bench.world_model_lifecycle.adjudication import inspect_adjudication_evidence; print(json.dumps(inspect_adjudication_evidence(Path(sys.argv[1])),sort_keys=True,separators=(",",":")))' \
  "$FORMAL_AUDIT"
```

Independently review every returned gate and the copied official report. Write
one canonical semantic-review v2 object at the fixed path; never infer an
accepted disposition merely from the producer's summary. Then adjudicate:

```bash
SEMANTIC_REVIEW="$REPO/artifacts/wm001-reviews/formal-v1.5.0.json"
DISPOSITION="accepted"  # use rejected if the audit/review requires rejection

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
  2) echo "Adjudication refused; inspect claim/package state before any action" >&2 ;;
  *) exit "$ADJUDICATION_RC" ;;
esac
```

An ordinary audit report receives exactly one byte-equality replay. A
no-report or authenticated outer-completion failure receives zero replays and
can only be rejected. The package uses schema
`prospect.wm001.adjudication-package.v7`; its version-wide claim uses
`prospect.wm001.formal-adjudication-claim.v2`.

After claim publication, an ordinary in-process fault automatically produces a
strictly verified rejected `adjudication_recovery_failure` without another
replay. An abrupt marker-only/hidden-staging interruption, or an exact renamed
package awaiting outer finalization, is recovered only through the sealed
single-use command below. Recovery never invokes the runner. Do not use it
unless the adjudication claim exists and the canonical package is not already
outer-finalized.

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

The final package is fixed at:

```text
bench/world_model_lifecycle/results/adjudication-v1.5/formal-adjudication-v1.5.0
```

Only an outer-finalized, strictly verified accepted package can support the
bounded WM-001 claim. Verify that exact package independently:

```bash
ADJUDICATION_PACKAGE="$REPO/bench/world_model_lifecycle/results/adjudication-v1.5/formal-adjudication-v1.5.0"
"$QA_PY" -I -B -c \
  'import json,sys; from pathlib import Path; from bench.world_model_lifecycle.adjudication import verify_adjudication_package; print(json.dumps(verify_adjudication_package(Path(sys.argv[1])),sort_keys=True,separators=(",",":")))' \
  "$ADJUDICATION_PACKAGE"
```
