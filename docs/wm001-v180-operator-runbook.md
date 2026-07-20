# WM-001 v1.8.0 operator runbook

Status: prospective. Seal this runbook with the exact v1.8 protocol,
implementation, schemas, lock, tests, and independent review before any v1.8
producer path is created.

Protocol 1.7.0 is retired. Its completed development producer and accepted
development audit are not reusable: the sole canonical closure ended as
outer-finalized failure evidence because its final live recheck passed the
320,556,697 byte result through a 64 MiB payload limit. There is no accepted
v1.7 closure, binding, or formal launch. Do not inspect, summarize, compare,
or use any v1.7 K3–K6 value.

## Fixed paths and clean launcher environment

```bash
set -euo pipefail

REPO="$(pwd -P)"
test "$REPO" = "$(git rev-parse --show-toplevel)"

QA_PY="/home/alex/.venvs/prospect-wm001-v18/bin/python"
RUNTIME_PY="/home/alex/.venvs/prospect-wm001-v18-runtime/bin/python"
LAUNCH="$REPO/bench/world_model_lifecycle/launch_bootstrap.py"
BOOTSTRAP="$REPO/bench/world_model_lifecycle/producer_bootstrap.py"

DEV_ROOT="$REPO/bench/world_model_lifecycle/results/development"
DEV="$DEV_ROOT/qualification-v1.8.0"
RUNTIME_SEAL="$DEV_ROOT/runtime-seal-v1.8.0.json"
DEV_CLOSURE="$DEV_ROOT/development-closure-v1.8.0.json"
PREFORMAL="$DEV_ROOT/preformal-test-report-v1.8.0.json"

OPERATOR_ROOT="$REPO/bench/world_model_lifecycle/results/operator-v1.8"
DEV_AUDIT="$OPERATOR_ROOT/audits/development-audit-v1.8.0"
CLOSURE_ATTEMPT="$OPERATOR_ROOT/closures/development-closure-v1.8.0"
BINDING_ATTEMPT="$OPERATOR_ROOT/bindings/formal-binding-v1.8.0"
BINDING="$BINDING_ATTEMPT/formal-binding.json"

OUTER_ROOT="$REPO/bench/world_model_lifecycle/results/outer-completions/v1.8"
RUNTIME_LOCK="$REPO/bench/world_model_lifecycle/results/.wm001-v1.8-runtime.lock"
FORMAL_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-launch-v1.8.0.json"
FORMAL_AUDIT="$OPERATOR_ROOT/audits/formal-audit-v1.8.0"
FORMAL_AUDIT_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-audit-v1.8.0.json"
ADJUDICATION_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-adjudication-v1.8.0.json"
ADJUDICATION_ROOT="$REPO/bench/world_model_lifecycle/results/adjudication-v1.8"
ADJUDICATION_PACKAGE="$ADJUDICATION_ROOT/formal-adjudication-v1.8.0"

REVIEW="$REPO/docs/wm001-v180-prospective-harness-review.json"
SEMANTIC_REVIEW="$REPO/artifacts/wm001-reviews/formal-v1.8.0.json"

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

The development masters are exactly `1196068124, 758859051`. The formal
masters are exactly `3362668913, 1230840469, 428983069, 1629522391,
1347202040, 1247885121, 3968594484, 3609284286`. The verifier must report 70
unique prior masters and 9,520 unique prior streams before any evidence run.

## Build the two environments once

Create both environments at previously absent paths. Build one reviewed wheel
and install that same wheel non-editably. Never reinstall into the runtime
environment after its inventory is rendered.

```bash
BASE_PY="/home/alex/miniconda3/bin/python"
QA_ENV="${QA_PY%/bin/python}"
RUNTIME_ENV="${RUNTIME_PY%/bin/python}"
WHEELHOUSE="$(mktemp -d /tmp/prospect-wm001-v18-wheelhouse.XXXXXX)"

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
generated WM-001 header must name v1.8.0. The Prospect row must equal the
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
    "# WM-001 protocol 1.8.0 isolated live execution closure.",
    "# Versions preserve the v1.4 scientific runtime; each v1.8 digest covers",
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
`wm001-v180-prospective-harness-review.json`, then commit the whole candidate.
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
require_absent "$PREFORMAL"
require_absent "$BINDING_ATTEMPT"
require_absent "$FORMAL_MARKER"
require_absent "$FORMAL_AUDIT"
require_absent "$FORMAL_AUDIT_MARKER"
require_absent "$ADJUDICATION_MARKER"
require_absent "$ADJUDICATION_PACKAGE"
require_absent "$SEMANTIC_REVIEW"
require_absent "$RUNTIME_LOCK"
require_absent "$OUTER_ROOT"
test -z "$(find "$DEV_ROOT" -maxdepth 1 \
  -name 'preformal-v1.8.0-command-*' -print -quit)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal \
  verify-prospective-review --review "$REVIEW"
"$QA_PY" -I -B -m pytest -q
"$QA_PY" -I -B -m ruff check src/prospect bench tests
"$QA_PY" -I -B -m mypy
```

Require the protocol verifier to reproduce:

- scientific-block SHA-256
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- 10 current masters, 1,360 current streams, 70 prior masters, and 9,520
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
the v1.8 development qualification even if any later phase fails.

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
or missing ordinary audit retires v1.8; never rerun it.

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
binding, or stop/go failure retires v1.8. Absence of a downstream path never
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
require_absent "$FORMAL_MARKER"
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
FORMAL_BINDING_ROOT="$REPO/bench/world_model_lifecycle/results/formal/$BINDING_SHA"
FORMAL="$FORMAL_BINDING_ROOT/confirmation-v1.8.0"
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
```

If both `FORMAL_BINDING_ROOT` and `FORMAL_MARKER` remain absent after a
pre-creation invocation refusal, only an invocation correction that changes no
bound byte may be considered. Any child below `FORMAL_BINDING_ROOT`, including
a noncanonical sibling left by an earlier launcher, contaminates and consumes
the v1.8 attempt. If either canonical path exists, the attempt is likewise
consumed. Never resume or rerun it. A nonzero return after marker publication
is terminal evidence.

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
```

This consumes the sole v1.8 formal-audit claim even if no ordinary report is
emitted. Do not rerun it. Inspect the canonical attempt and create one
independent semantic-review object from the supported read-only inspector:

```bash
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
bounded WM-001 claim. Every other terminal state retires v1.8 without
same-version repair or retry.
