# WM-001 v1.9.0 operator runbook

Status: prospective. Seal this runbook with the exact v1.9 protocol,
implementation, schemas, lock, tests, and independent review before any v1.9
producer path is created.

Protocol 1.8.0 is retired. Its completed producer, accepted audit, canonical
archive, marker, and accepted closure attempt are not reusable: a fresh
interpreter rejected their matrix-contract digest because two frozensets were
serialized without sorting. There is no v1.8 preformal report, binding, formal
launch, formal audit, semantic review, or adjudication. Do not inspect,
summarize, compare, or use any v1.8 K3–K6 value.

## Fixed paths and clean launcher environment

```bash
set -euo pipefail

REPO="$(pwd -P)"
test "$REPO" = "$(git rev-parse --show-toplevel)"

QA_PY="/home/alex/.venvs/prospect-wm001-v19-reviewed/bin/python"
RUNTIME_PY="/home/alex/.venvs/prospect-wm001-v19-reviewed-runtime/bin/python"
LAUNCH="$REPO/bench/world_model_lifecycle/launch_bootstrap.py"
BOOTSTRAP="$REPO/bench/world_model_lifecycle/producer_bootstrap.py"

DEV_ROOT="$REPO/bench/world_model_lifecycle/results/development"
DEV="$DEV_ROOT/qualification-v1.9.0"
RUNTIME_SEAL="$DEV_ROOT/runtime-seal-v1.9.0.json"
DEV_CLOSURE="$DEV_ROOT/development-closure-v1.9.0.json"
PREFORMAL="$DEV_ROOT/preformal-test-report-v1.9.0.json"

OPERATOR_ROOT="$REPO/bench/world_model_lifecycle/results/operator-v1.9"
DEV_AUDIT="$OPERATOR_ROOT/audits/development-audit-v1.9.0"
CLOSURE_ATTEMPT="$OPERATOR_ROOT/closures/development-closure-v1.9.0"
BINDING_ATTEMPT="$OPERATOR_ROOT/bindings/formal-binding-v1.9.0"
BINDING="$BINDING_ATTEMPT/formal-binding.json"

OUTER_ROOT="$REPO/bench/world_model_lifecycle/results/outer-completions/v1.9"
RUNTIME_LOCK="$REPO/bench/world_model_lifecycle/results/.wm001-v1.9-runtime.lock"
FORMAL_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-launch-v1.9.0.json"
FORMAL_AUDIT="$OPERATOR_ROOT/audits/formal-audit-v1.9.0"
FORMAL_AUDIT_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-audit-v1.9.0.json"
ADJUDICATION_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-adjudication-v1.9.0.json"
ADJUDICATION_ROOT="$REPO/bench/world_model_lifecycle/results/adjudication-v1.9"
ADJUDICATION_PACKAGE="$ADJUDICATION_ROOT/formal-adjudication-v1.9.0"

REVIEW="$REPO/docs/wm001-v190-prospective-harness-review.json"
SEMANTIC_REVIEW="$REPO/artifacts/wm001-reviews/formal-v1.9.0.json"

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

The development masters are exactly `86535224, 2906056242`. The formal
masters are exactly `1369779618, 2721934008, 2798280967, 926105433,
4118470289, 919763803, 2112633694, 2832104894`. The verifier must report 80
unique prior masters and 10,880 unique prior streams before any evidence run.

## Build the two environments once

Create both environments at previously absent paths. Build one reviewed wheel
and install that same wheel non-editably. Never reinstall into the runtime
environment after its inventory is rendered.

```bash
BASE_PY="/home/alex/miniconda3/bin/python"
QA_ENV="${QA_PY%/bin/python}"
RUNTIME_ENV="${RUNTIME_PY%/bin/python}"
WHEELHOUSE="$(mktemp -d /tmp/prospect-wm001-v19-wheelhouse.XXXXXX)"

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
generated WM-001 header must name v1.9.0. The Prospect row must equal the
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
    "# WM-001 protocol 1.9.0 isolated live execution closure.",
    "# Versions preserve the v1.4 scientific runtime; each v1.9 digest covers",
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
`wm001-v190-prospective-harness-review.json`, then commit the whole candidate.
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
  -name 'preformal-v1.9.0-command-*' -print -quit)"
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

Require the protocol verifier to reproduce:

- scientific-block SHA-256
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- 10 current masters, 1,360 current streams, 80 prior masters, and 10,880
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
v1.9 producer root. Verify the exact real-subprocess branch tests explicitly
if they are not already a separate command row:

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
the v1.9 development qualification even if any later phase fails.

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
or missing ordinary audit retires v1.9; never rerun it.

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
binding, or stop/go failure retires v1.9. Absence of a downstream path never
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

```bash
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
are absent, and the binding authorizes the exact formal preflight. The sole
formal invocation below must perform its descriptor-mode conformance replay
and require equality with the bound manifest and report before publishing the
formal marker.

## Sole formal producer

```bash
BINDING_SHA="$("$QA_PY" -I -B -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$BINDING")"
FORMAL_BINDING_ROOT="$REPO/bench/world_model_lifecycle/results/formal/$BINDING_SHA"
FORMAL="$FORMAL_BINDING_ROOT/confirmation-v1.9.0"
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
the v1.9 attempt. If either canonical path exists, the attempt is likewise
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
if [ ! -d "$FORMAL_AUDIT" ]; then
  echo "formal audit failed before publishing an authenticated attempt; v1.9 is retired" >&2
  exit 2
fi
"$QA_PY" -I -B -c \
  'import os,sys; from pathlib import Path; from bench.world_model_lifecycle.operator import inspect_unfinalized_operator_attempt,outer_completion_marker,verify_operator_attempt; attempt=Path(sys.argv[1]); terminal=attempt/"operator-attempt.json"; (verify_operator_attempt if os.path.lexists(outer_completion_marker(terminal)) else inspect_unfinalized_operator_attempt)(attempt)' \
  "$FORMAL_AUDIT"
```

Only an authenticated canonical finalized or explicitly unfinalized attempt
establishes that the sole v1.9 formal-audit claim was consumed, even if no
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
bounded WM-001 claim. Every other terminal state retires v1.9 without
same-version repair or retry.
