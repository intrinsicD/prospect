# WM-001 v1.16.0 operator runbook

Status: prospective. Seal this runbook with the exact v1.16 protocol,
implementation, schemas, lock, tests, and independent review before any v1.16
producer path is created.

Protocols 1.10.0 through 1.15.0 are terminally retired. Version 1.15 passed
every prospective, development, audit, closure, preformal, binding, preflight,
and final stop/go gate. After the operator created and fsynced the empty
binding-keyed formal root, its sole formal invocation failed before producer
bootstrap dispatch, formal-marker publication, confirmation-directory
creation, or any formal outcome.

The standard-library outer launcher had applied its 64 MiB control-object
reader to the live bulk development producer. Five authenticated producer
files exceeded that limit, including the 320,977,868-byte raw result. Do not
inspect or use any retained K or performance value. Nothing in this runbook
repairs, resumes, removes, or reuses v1.10 through v1.15 evidence, environments,
wheels, seals, locks, bindings, outer completions, diagnostics, or the v1.15
empty formal root. The exact disposition is documented in the
[v1.15 formal-invocation failure](wm001-v1150-formal-invocation-failure.md).

The active protocol must directly supersede v1.15.0 and pin its exact sealed
protocol SHA-256 as
`8db5560044bbedfb491be12a26bd8b39c43fd6d6a314ce86d6afdc71f50486bb`.
No earlier lineage digest is an acceptable substitute.

## Fixed paths and clean launcher environment

```bash
set -euo pipefail

REPO="$(pwd -P)"
test "$REPO" = "$(git rev-parse --show-toplevel)"

QA_PY="/home/alex/.venvs/prospect-wm001-v116-reviewed/bin/python"
RUNTIME_PY="/home/alex/.venvs/prospect-wm001-v116-reviewed-runtime/bin/python"
LAUNCH="$REPO/bench/world_model_lifecycle/launch_bootstrap.py"
BOOTSTRAP="$REPO/bench/world_model_lifecycle/producer_bootstrap.py"

DEV_ROOT="$REPO/bench/world_model_lifecycle/results/development"
DEV="$DEV_ROOT/qualification-v1.16.0"
DEV_DIAGNOSTICS="$DEV_ROOT/diagnostics-v1.16.0"
RUNTIME_SEAL="$DEV_ROOT/runtime-seal-v1.16.0.json"
DEV_CLOSURE="$DEV_ROOT/development-closure-v1.16.0.json"
PREFORMAL_ROOT="$DEV_ROOT/v1.16.0/preformal"
PREFORMAL_CLAIM="$DEV_ROOT/v1.16.0/.preformal.staging"
PREFORMAL="$PREFORMAL_ROOT/preformal-test-report-v1.16.0.json"

OPERATOR_ROOT="$REPO/bench/world_model_lifecycle/results/operator-v1.16"
DEV_AUDIT="$OPERATOR_ROOT/audits/development-audit-v1.16.0"
CLOSURE_ATTEMPT="$OPERATOR_ROOT/closures/development-closure-v1.16.0"
BINDING_ATTEMPT="$OPERATOR_ROOT/bindings/formal-binding-v1.16.0"
BINDING="$BINDING_ATTEMPT/formal-binding.json"
FORMAL_INPUT_PREFLIGHT="$BINDING_ATTEMPT/formal-input-preflight.json"

OUTER_ROOT="$REPO/bench/world_model_lifecycle/results/outer-completions/v1.16"
RUNTIME_LOCK="$REPO/bench/world_model_lifecycle/results/.wm001-v1.16-runtime.lock"
FORMAL_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-launch-v1.16.0.json"
FORMAL_AUDIT="$OPERATOR_ROOT/audits/formal-audit-v1.16.0"
FORMAL_AUDIT_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-audit-v1.16.0.json"
ADJUDICATION_MARKER="$REPO/bench/world_model_lifecycle/results/formal/formal-adjudication-v1.16.0.json"
ADJUDICATION_ROOT="$REPO/bench/world_model_lifecycle/results/adjudication-v1.16"
ADJUDICATION_PACKAGE="$ADJUDICATION_ROOT/formal-adjudication-v1.16.0"

REVIEW="$REPO/docs/wm001-v1160-prospective-harness-review.json"
SEMANTIC_REVIEW="$REPO/artifacts/wm001-reviews/formal-v1.16.0.json"

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

require_v116_lifecycle_absent() {
  local path
  for path in \
    "$OPERATOR_ROOT" \
    "$ADJUDICATION_ROOT" \
    "$DEV" \
    "$DEV_DIAGNOSTICS" \
    "$RUNTIME_SEAL" \
    "$DEV_AUDIT" \
    "$DEV_CLOSURE" \
    "$CLOSURE_ATTEMPT" \
    "$PREFORMAL_ROOT" \
    "$PREFORMAL_CLAIM" \
    "$BINDING_ATTEMPT" \
    "$FORMAL_INPUT_PREFLIGHT" \
    "$FORMAL_MARKER" \
    "$FORMAL_AUDIT" \
    "$FORMAL_AUDIT_MARKER" \
    "$ADJUDICATION_MARKER" \
    "$ADJUDICATION_PACKAGE" \
    "$SEMANTIC_REVIEW" \
    "$RUNTIME_LOCK" \
    "$OUTER_ROOT"
  do
    require_absent "$path"
  done
  if find "$REPO/bench/world_model_lifecycle/results/formal" \
      -mindepth 1 -name '*v1.16.0*' -print -quit 2>/dev/null |
      grep -q .
  then
    echo "unexpected v1.16 formal lifecycle path exists" >&2
    return 2
  fi
}
```

The development masters are exactly `3922749719, 1847570536`. The formal
masters are exactly `721000968, 1733386057, 1129257495, 1461304433,
345413014, 76587833, 404195464, 3550251066`. The verifier must verify 150
unique prior masters and 20,400 unique prior streams through v1.15 before any
evidence run.

The prebinding runtime identity must use
`prospect.wm001.prebinding-conformance-request.v2` and
`prospect.wm001.prebinding-conformance.v2`. Its exact string-valued precision
fields cover the global, CUDA-matmul, cuDNN-backend, cuDNN-convolution, and
cuDNN-RNN `fp32_precision` settings; no legacy boolean alias is accepted.

## Regenerate and verify the protocol seal

Regenerate the three-row seal mechanically after the protocol and both schemas
are final, and before building the reviewed wheel. The wheel packages this
file, so any later protocol, schema, or seal change invalidates both
environments and requires starting this preparation again at new absent paths.

```bash
SEAL="$REPO/bench/world_model_lifecycle/SEALED_PROTOCOL.sha256"
SEAL_TMP="$(mktemp "$REPO/bench/world_model_lifecycle/.SEALED_PROTOCOL.XXXXXX")"
(
  cd "$REPO/bench/world_model_lifecycle"
  sha256sum \
    protocol.json \
    schemas/formal-binding.schema.json \
    schemas/raw-result.schema.json
) >"$SEAL_TMP"
mv "$SEAL_TMP" "$SEAL"
test "$(wc -l <"$SEAL")" -eq 3
(
  cd "$REPO/bench/world_model_lifecycle"
  sha256sum --check SEALED_PROTOCOL.sha256
)
```

## Build the two environments once

Create both environments at previously absent paths. Build one reviewed wheel
and install that same wheel non-editably. Never reinstall into the runtime
environment after its inventory is rendered.

```bash
BASE_PY="/home/alex/miniconda3/bin/python"
QA_ENV="${QA_PY%/bin/python}"
RUNTIME_ENV="${RUNTIME_PY%/bin/python}"
WHEELHOUSE="$(mktemp -d /tmp/prospect-wm001-v116-wheelhouse.XXXXXX)"

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
generated WM-001 header must name v1.16.0. The Prospect row must equal the
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
    "# WM-001 protocol 1.16.0 isolated live execution closure.",
    "# Versions preserve the v1.4 scientific runtime; each v1.16 digest covers",
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
prospective-review implementation manifest. An independent referee first
authors a canonical draft with the exact top-level review fields, its
disposition, blockers, and findings, using empty `implementation_files` and an
empty `implementation_manifest_sha256`. Bind that reviewed prose to the staged
source mechanically:

```bash
REVIEW_DRAFT="$REPO/docs/wm001-v1160-prospective-harness-review.draft.json"
test -f "$REVIEW_DRAFT"
"$QA_PY" -I -B - "$REVIEW_DRAFT" "$REVIEW" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

from bench.world_model_lifecycle.preformal import (
    _canonical_json_bytes,
    _implementation_files,
)

draft = json.loads(Path(sys.argv[1]).read_bytes())
rows = _implementation_files()
draft["implementation_files"] = rows
draft["implementation_manifest_sha256"] = hashlib.sha256(
    _canonical_json_bytes(rows)
).hexdigest()
Path(sys.argv[2]).write_bytes(_canonical_json_bytes(draft) + b"\n")
PY
rm "$REVIEW_DRAFT"
git add "$REVIEW"
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal \
  verify-prospective-review --review "$REVIEW"
```

Commit the whole candidate. Do not rebuild, reinstall, or change a bound byte
after that commit.

## Pre-outcome static gates

All version-scoped paths below must be absent. This helper is deliberately run
before and after the complete pre-outcome test and static-gate set because
these paths are gitignored and a test-created lifecycle claim would not dirty
the worktree:

```bash
require_v116_lifecycle_absent
test -z "$(git status --porcelain=v1 --untracked-files=all)"

"$QA_PY" -I -B -m bench.world_model_lifecycle.verify protocol
"$QA_PY" -I -B -m bench.world_model_lifecycle.preformal \
  verify-prospective-review --review "$REVIEW"
"$QA_PY" -I -B -m pytest -q
"$QA_PY" -I -B -m pytest -q \
  tests/test_world_model_preformal.py \
  tests/test_world_model_binding.py \
  tests/test_world_model_operator.py \
  -k 'preserved_preformal'
"$QA_PY" -I -B -m pytest -q \
  tests/test_world_model_prebinding_audit.py::test_independent_development_archive_accepts_one_continuous_member_pass \
  tests/test_world_model_prebinding_audit.py::test_independent_development_archive_rejects_a_real_unbound_member \
  tests/test_world_model_prebinding_audit.py::test_development_qualification_is_linked_field_for_field \
  tests/test_world_model_prebinding_audit.py::test_formal_input_preflight_runs_both_substantive_validators \
  tests/test_world_model_shared_archive_boundary.py::test_one_writer_archive_crosses_both_real_readers_and_rejects_stale_member_digest \
  tests/test_world_model_binding.py::test_development_qualification_archive_rejects_invalid_evidence_namespace \
  tests/test_world_model_binding.py::test_central_development_audit_requires_exact_full_report \
  tests/test_world_model_artifact_audit.py::test_local_producer_manifest_requires_real_ordered_utc_timestamps \
  tests/test_world_model_artifact_audit.py::test_independent_preformal_log_parser_requires_empty_stderr \
  tests/test_world_model_launch_bootstrap.py::test_outer_rejects_malformed_producer_time_or_numeric_count \
  tests/test_world_model_launch_bootstrap.py::test_outer_rejects_numeric_preflight_binding_byte_alias \
  tests/test_world_model_launch_bootstrap.py::test_outer_streams_valid_result_at_and_above_control_limit \
  tests/test_world_model_launch_bootstrap.py::test_outer_streams_production_scale_five_role_producer_once_per_file \
  tests/test_world_model_launch_bootstrap.py::test_real_outer_launcher_rehearses_accepted_binding_with_large_producer \
  tests/test_world_model_launch_bootstrap.py::test_outer_streaming_manifest_rejects_malformed_identity_metadata \
  tests/test_world_model_launch_bootstrap.py::test_outer_streaming_manifest_enforces_exact_producer_limits \
  tests/test_world_model_launch_bootstrap.py::test_producer_tree_snapshot_bounds_entries_before_sorting \
  tests/test_world_model_launch_bootstrap.py::test_outer_streaming_rejects_short_read \
  tests/test_world_model_launch_bootstrap.py::test_outer_streaming_rejects_in_read_path_mutation \
  tests/test_world_model_launch_bootstrap.py::test_outer_streaming_rejects_symlink_and_hardlink \
  tests/test_world_model_launch_bootstrap.py::test_outer_rejects_invalid_terminal_bound_qualification_custody \
  tests/test_world_model_launch_bootstrap.py::test_terminal_bound_qualification_rejects_semantic_mutations \
  tests/test_world_model_artifact_authorization.py::test_formal_binding_authorization_requires_exact_result_qualification \
  tests/test_world_model_prebinding_audit.py::test_exact_formal_input_preflight_rejects_result_qualification_mismatch \
  tests/test_world_model_preformal.py::test_nonempty_command_1_stderr_prevents_preformal_authorization \
  tests/test_world_model_preformal.py::test_preserved_preformal_bound_projection_rejects_any_command_stderr \
  tests/test_world_model_artifact_authorization.py::test_formal_binding_authorization_reconstructs_exact_ordered_inputs \
  tests/test_world_model_artifact_authorization.py::test_development_closure_authorization_reconstructs_producer_and_audit
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
import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

schema = json.loads(
    Path("bench/world_model_lifecycle/schemas/formal-binding.schema.json")
    .read_text(encoding="utf-8")
)
Draft202012Validator.check_schema(schema)
source = schema["properties"]["source"]["properties"]
if (
    schema["$id"]
    != "https://prospect.local/schemas/wm-001-formal-binding-v10.json"
    or schema["properties"]["schema"]["const"]
    != "prospect.world-model-lifecycle.formal-binding.v10"
    or source["implementation_files"]["items"]["$ref"]
    != "#/$defs/fileDigest"
    or source["test_log_files"]["items"]["$ref"]
    != "#/$defs/streamFileDigest"
    or schema["$defs"]["fileDigest"]["properties"]["bytes"]["minimum"] != 1
    or schema["$defs"]["streamFileDigest"]["properties"]["bytes"]["minimum"] != 0
):
    raise SystemExit("formal-binding v10 stream/source schema split is not exact")

empty_sha256 = hashlib.sha256(b"").hexdigest()
nonempty_sha256 = hashlib.sha256(b"x").hexdigest()
rows = [
    {
        "path": f"command-{index // 2 + 1:02d}.{'stdout' if index % 2 == 0 else 'stderr'}",
        "bytes": 1 if index % 2 == 0 else 0,
        "sha256": nonempty_sha256 if index % 2 == 0 else empty_sha256,
    }
    for index in range(20)
]
projection = {
    "$schema": schema["$schema"],
    "$defs": schema["$defs"],
    **source["test_log_files"],
}
validator = Draft202012Validator(projection)
if list(validator.iter_errors(rows)):
    raise SystemExit("real-shaped 20-row stdout/empty-stderr manifest is rejected")
negative = copy.deepcopy(rows)
negative[1]["bytes"] = -1
if not list(validator.iter_errors(negative)):
    raise SystemExit("negative stream length was accepted")
implementation = Draft202012Validator(
    {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        **schema["$defs"]["fileDigest"],
    }
)
if not list(
    implementation.iter_errors(
        {"path": "empty.py", "bytes": 0, "sha256": empty_sha256}
    )
):
    raise SystemExit("empty implementation source was accepted")
PY

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

The targeted preformal-report-role tests are a mandatory producer-to-consumer
gate, not name-only unit coverage. They must generate one valid canonical preformal
report and its 20 real stream files, preserve those bytes in a mixed binding
staging package, and run the same strict binding consumer used by the operator.
They must prove all of the following:

- the canonical verifier accepts only the original v1.16 development bundle
  and rejects the preserved copy;
- the preserved verifier accepts the exact report and referenced logs while
  legitimate declared binding sidecars share that directory;
- the preserved verifier rejects a missing or changed referenced stream and
  any additional report or `preformal-v1.16.0-command-*` member; and
- the complete created binding equals the strict consumer's returned object
  before an accepted attempt can be published.

The targeted development-archive tests are a second independent mandatory
gate. One archive generated by the production writer must traverse both the
actual central and independent archive readers, with no mocked reader seam, and
both must retain identical bytes. A shared physical payload mutation with only
the outer digest/path rebound must fail at both readers. A real canonical
multi-member USTAR must also be accepted by one continuous independent
iterator pass, while a physically present undeclared member is rejected. The
production writer must reject every key outside the exact one-level
`evidence/*` namespace, including a key that could collide with `producer/*`.
A real producer manifest must reject invalid or reversed UTC timestamps. The
complete qualification and formal-input tests must bind both bootstraps, all
nine evidence roles, archived runtime/invocation semantics, and the live
producer/audit inputs field-for-field. A real producer, real outer-finalized
audit, and real closure authorization must reject substitutions in every
archived/live role. Mutation coverage must also reject missing, reordered,
duplicated, linked, noncanonical, changed, truncated, padded, or trailing
archive members; incorrect producer membership; JSON
boolean/integer/float aliases; archive-size overflow; and stable-file or
single-link custody violations. Restoring the old `next(iter(stream))`
exhaustion pattern must make the positive regression fail.

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

- direct supersession of v1.15.0 protocol SHA-256
  `8db5560044bbedfb491be12a26bd8b39c43fd6d6a314ce86d6afdc71f50486bb`;
- scientific-block SHA-256
  `fa44fd93a672db3905d45a0e99c568985e7e2e5d02d32043c830db413005a5c3`;
- 10 current masters, 1,360 current streams, 150 prior masters, and 20,400
  prior streams with zero collision;
- deterministic matrix-contract SHA-256
  `09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84`;
- the unchanged four scientific source hashes; and
- unchanged raw-result v9 and formal-binding v10 representation identities,
  with only the v1.16 protocol and formal-seed constants updated; and
- the exact sorted full outcome-audit support list:
  `producer_bootstrap.py`, `protocol.json`,
  `schemas/raw-result.schema.json`.

## Create the prospective seal

```bash
require_v116_lifecycle_absent
test -z "$(git status --porcelain=v1 --untracked-files=all)"
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
COMMAND10_CAPTURE="$(mktemp -d /tmp/prospect-wm001-v116-command10.XXXXXX)"
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
  echo "result-free command 10 failed or wrote stderr; v1.16 is retired" >&2
  wc -c -- "$COMMAND10_STDOUT" "$COMMAND10_STDERR" >&2
  echo "captures retained at $COMMAND10_CAPTURE" >&2
  exit 2
fi

"$QA_PY" -I -B - "$COMMAND10_STDOUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from bench.world_model_lifecycle.preformal import (
    _canonical_json_bytes,
    _validate_recorded_fresh_identity_conformance,
    _validate_recorded_result_free_inventory,
)

path = Path(sys.argv[1])
payload = path.read_bytes()
value = json.loads(payload)
if not isinstance(value, dict):
    raise SystemExit("command-10 stdout is not one JSON object")
inventory = _validate_recorded_result_free_inventory(value.get("inventory"))
fresh_identity = _validate_recorded_fresh_identity_conformance(
    value.get("fresh_runtime_identity_conformance")
)
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
    "schema",
    "mode",
    "device",
    "passed",
    "inventory",
    "inventory_sha256",
    "conformance_sha256",
    "fresh_runtime_identity_conformance",
    "fresh_runtime_identity_conformance_sha256",
    "restart_runtime_conformance_report_sha256",
    "restart_runtime_execution_receipt_sha256",
    "restart_runtime_support_files",
    "restart_runtime_repeat_count",
    "restart_runtime_path_descriptor_equal",
    "repeat_count",
    "path_descriptor_equal",
}
digest_fields = (
    "inventory_sha256",
    "conformance_sha256",
    "fresh_runtime_identity_conformance_sha256",
    "restart_runtime_conformance_report_sha256",
    "restart_runtime_execution_receipt_sha256",
)


def is_sha256(candidate: object) -> bool:
    return (
        isinstance(candidate, str)
        and len(candidate) == 64
        and all(character in "0123456789abcdef" for character in candidate)
    )


if payload != canonical:
    raise SystemExit("command-10 stdout is not canonical JSON plus LF")
if (
    set(value) != required
    or value.get("schema") != "prospect.wm001.preformal-runtime-check.v1"
    or value.get("mode") != "bootstrap-inventory-conformance"
    or value.get("device") != "cuda"
    or value.get("passed") is not True
    or not all(is_sha256(value.get(field)) for field in digest_fields)
    or value.get("inventory_sha256")
    != hashlib.sha256(_canonical_json_bytes(inventory)).hexdigest()
    or value.get("fresh_runtime_identity_conformance_sha256")
    != hashlib.sha256(_canonical_json_bytes(fresh_identity)).hexdigest()
    or value.get("restart_runtime_support_files")
    != [
        "producer_bootstrap.py",
        "protocol.json",
        "schemas/raw-result.schema.json",
    ]
    or value.get("restart_runtime_repeat_count") != 3
    or value.get("restart_runtime_path_descriptor_equal") is not True
    or value.get("repeat_count") != 3
    or value.get("path_descriptor_equal") is not True
):
    raise SystemExit("command-10 stdout is not the passing CUDA receipt")
PY

test "$(wc -c <"$COMMAND10_STDERR")" -eq 0
test ! -e "$DEV" && test ! -L "$DEV"
echo "result-free command-10 captures retained at $COMMAND10_CAPTURE" >&2
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
stderr. Once the command-10 launcher starts, any nonzero return or nonempty
stderr retires v1.16; do not repair its source, wheel, environment, seal, or
capture in place.

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
v1.16 producer root. Verify the exact real-subprocess branch tests explicitly
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
nonzero return, or nonempty stderr consumes and retires v1.16; do not repeat
the rehearsal. Any source, environment, dependency, protocol, schema, seed,
seal, or support change requires a new version and fresh paths.

## Sole development qualification

The command below has no seed override. Exclusive creation of `DEV` consumes
the v1.16 development qualification even if any later phase fails.

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
or missing ordinary audit retires v1.16; never rerun it.

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
binding, or stop/go failure retires v1.16. Absence of a downstream path never
restores authorization.

The closure must bind one byte-canonical USTAR archive whose physical sequence
equals its ordered rows. Its exact `producer/*` members must equal the archived
producer manifest, and its one-level `evidence/*` members must be exactly the
result qualification, audit, reproduction, runtime seal, launch bootstrap,
producer bootstrap, runtime manifest, invocation manifest, and audit stderr.
The archive writer must have rejected namespace collisions before combining
the two member maps. Both central closure verification and the later
independent formal-input consumer must use strict JSON scalar identity, real
ordered UTC producer timestamps, the 64 MiB per-retained-member and 256 MiB
aggregate-retained limits, and stable descriptor custody. Do not manually open
the archived result, K values, performance summary, tensor, or trace; only the
precommitted verifiers and formal evidence procedures may process them.

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
`$DEV_ROOT/v1.16.0/.preformal.staging`, then fsyncs its parent before its first
command; either that hidden claim or the final bundle consumes the one-shot
attempt. Operator namespaces and their deterministic hidden claims are also
created durably, with each new directory entry committed through its parent.
Do not create either preformal path during setup, QA, rehearsal, production,
audit, or closure.

```bash
require_absent "$PREFORMAL_ROOT"
require_absent "$DEV_ROOT/v1.16.0/.preformal.staging"

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
  'import sys; from pathlib import Path; from bench.world_model_lifecycle.binding import verify_canonical_machine_test_report; report=verify_canonical_machine_test_report(Path(sys.argv[1])); assert report["all_pass"] is True' \
  "$PREFORMAL"
```

The report must contain exactly ten ordered command rows—eight QA and two
sealed-runtime rows—with exactly 20 separate stdout/stderr logs. Every command
must return zero and every stderr log must be the canonical empty stream; one
stderr byte in any of the ten rows fails and retires the version. The generator
runs all ten commands while `PREFORMAL_ROOT` remains absent, stages and fsyncs
all 21 files under the hidden claim, and atomically publishes the complete
directory with no replacement. It prints a
`prospect.wm001.preformal-test-report-generation.v2` envelope and exits nonzero
with `passed: false` for an exact command failure, identity check failure, or
semantic accepted-closure/runtime-conformance failure. The report and logs
remain terminal evidence in that case. A write fault or interruption may leave
only the hidden claim and no envelope; that is still terminal. Never remove
either path or retry v1.16. A successful generation is not sufficient by
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

Before the operator creates the deterministic binding claim, the canonical
report role must validate the report at `$PREFORMAL` in its exclusive
development bundle. It must then validate the actual 20-row
`preformal_log_rows()` projection against the root formal-binding v10 schema
and its referenced definitions. Ten stdout rows are nonempty and all ten
stderr rows are exact empty-stream identities. This preclaim check must also
prove that implementation files still use the nonempty `fileDigest`
definition.

After the complete binding object and sidecars exist only in the operator's
private staging package, the strict consumer must select the distinct
preserved-report role. That role independently enforces the same fixed
command, input, QA/runtime, semantic, and stream-byte contract but expects the
report as a safe binding sibling and permits only non-preformal package
sidecars in the shared directory. It must reject any unreferenced
preformal-named member. The producer must also validate the full binding object
against the root schema before any accepted attempt is published. Calling the
canonical bundle verifier on the preserved sibling, or weakening the canonical
verifier to accept both roles, is terminal.

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
attempt must also contain the fixed single-link
`development-result-qualification.json` copied byte-for-byte from the
qualification archive. Its digest must equal the existing
`development_qualification.result_qualification_sha256` binding identity; it
must be listed in the terminal manifest. The outer launcher rejoins its
`raw_result_sha256` to the once-streamed live result, while the independent
formal auditor rejoins the live attempt sidecar, formal-root copy, archived
bytes, and binding digest.
The
consumer must parse and root membership in the archived producer manifest,
retain one archive iterator through exhaustion, require the exact nine-member
evidence namespace, verify both archived bootstraps and the complete archived
runtime/invocation semantics, and join every live producer, audit,
reproduction, runtime, invocation, stderr, seal, and bootstrap byte string to
its archived role. No mocked archive helper, closure-only digest projection, or
self-reported manifest can satisfy this gate. The
standard-library outer launcher requires that terminal-bound receipt before it
imports the formal producer and cross-checks its binding, report, closure, and
bound auditor identities. It reopens the canonical stdout for command 9 and
command 10, validates command 10's empty stderr and full inventory/fresh-child
objects, requires that inventory to equal the binding dependencies, and
requires each semantic-object digest to equal the corresponding receipt digest.
Commands 1–8 are authorized by the accepted binding and its terminal-bound
preflight receipt, including their exact empty-stderr identities; the
standard-library launcher does not duplicate their full QA report verifier. Its
additional direct report parsing is deliberately limited to the two
sealed-runtime authorization commands.
The receipt must have schema
`prospect.wm001.formal-input-preflight.v1`, protocol
version `1.16.0`, `passed: true`, and digests for the binding, preformal report,
development closure, accepted-closure evidence, runtime conformance, and
auditor source. Any missing, false, stale, or mismatched receipt retires v1.16.
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

## Accepted-binding outer rehearsal

This result-free gate runs exactly once after the accepted binding and final
stop/go checks, and before the binding-keyed formal root exists. Compute its
paths without creating them, capture stdout and stderr separately, and require
every formal authority path to remain absent:

```bash
BINDING_SHA="$("$QA_PY" -I -B -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$BINDING")"
FORMAL_BINDING_ROOT="$REPO/bench/world_model_lifecycle/results/formal/$BINDING_SHA"
FORMAL="$FORMAL_BINDING_ROOT/confirmation-v1.16.0"
PREFORMAL_BINDING_CAPTURE="$(mktemp -d /tmp/prospect-wm001-v116-binding-rehearsal.XXXXXX)"

require_absent "$FORMAL_BINDING_ROOT"
require_absent "$FORMAL"
require_absent "$FORMAL_MARKER"
require_absent "$FORMAL_AUDIT"
require_absent "$FORMAL_AUDIT_MARKER"
require_absent "$SEMANTIC_REVIEW"
require_absent "$ADJUDICATION_MARKER"
require_absent "$ADJUDICATION_PACKAGE"

set +e
"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B "$LAUNCH" \
  --bootstrap "$BOOTSTRAP" \
  --runtime-seal "$BINDING" \
  -- preformal-runtime bootstrap-inventory-conformance \
  --device cuda \
  >"$PREFORMAL_BINDING_CAPTURE/stdout" \
  2>"$PREFORMAL_BINDING_CAPTURE/stderr"
PREFORMAL_BINDING_RC=$?
set -e

test "$PREFORMAL_BINDING_RC" -eq 0
test ! -s "$PREFORMAL_BINDING_CAPTURE/stderr"

"$QA_PY" -I -B - \
  "$BINDING" "$PREFORMAL_BINDING_CAPTURE/stdout" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from bench.world_model_lifecycle.preformal import (
    _canonical_json_bytes,
    _validate_recorded_fresh_identity_conformance,
    _validate_recorded_result_free_inventory,
)

binding = json.loads(Path(sys.argv[1]).read_bytes())
payload = Path(sys.argv[2]).read_bytes()
value = json.loads(payload)
if payload != _canonical_json_bytes(value) + b"\n":
    raise SystemExit("accepted-binding rehearsal stdout is not canonical")
expected_fields = {
    "schema", "mode", "device", "passed", "inventory",
    "inventory_sha256", "conformance_sha256",
    "fresh_runtime_identity_conformance",
    "fresh_runtime_identity_conformance_sha256",
    "restart_runtime_conformance_report_sha256",
    "restart_runtime_execution_receipt_sha256",
    "restart_runtime_support_files", "restart_runtime_repeat_count",
    "restart_runtime_path_descriptor_equal", "repeat_count",
    "path_descriptor_equal",
}
inventory = _validate_recorded_result_free_inventory(value.get("inventory"))
fresh = _validate_recorded_fresh_identity_conformance(
    value.get("fresh_runtime_identity_conformance")
)
dependencies = binding["dependencies"]
expected_inventory = {
    "packages": dependencies["packages"],
    "package_roots": dependencies["package_roots"],
    "standard_library": dependencies["standard_library"],
    "package_ownership": dependencies["package_ownership"],
}
execution = binding["audit_execution"]
if (
    set(value) != expected_fields
    or value.get("schema")
    != "prospect.wm001.preformal-runtime-check.v1"
    or value.get("mode") != "bootstrap-inventory-conformance"
    or value.get("device") != "cuda"
    or value.get("passed") is not True
    or inventory != expected_inventory
    or value.get("inventory_sha256")
    != hashlib.sha256(_canonical_json_bytes(inventory)).hexdigest()
    or value.get("conformance_sha256")
    != hashlib.sha256(_canonical_json_bytes(execution)).hexdigest()
    or value.get("fresh_runtime_identity_conformance_sha256")
    != hashlib.sha256(_canonical_json_bytes(fresh)).hexdigest()
    or value.get("restart_runtime_conformance_report_sha256")
    != execution["restart_runtime_conformance_report_sha256"]
    or value.get("restart_runtime_execution_receipt_sha256")
    != execution["restart_runtime_execution_receipt_sha256"]
    or value.get("restart_runtime_support_files")
    != execution["restart_runtime_support_files"]
    or value.get("restart_runtime_support_files")
    != [
        "producer_bootstrap.py",
        "protocol.json",
        "schemas/raw-result.schema.json",
    ]
    or value.get("restart_runtime_repeat_count")
    != execution["restart_runtime_repeat_count"]
    or value.get("restart_runtime_repeat_count") != 3
    or value.get("restart_runtime_path_descriptor_equal")
    != execution["restart_runtime_path_descriptor_equal"]
    or value.get("restart_runtime_path_descriptor_equal") is not True
    or value.get("repeat_count") != execution["repeat_count"]
    or value.get("repeat_count") != 3
    or value.get("path_descriptor_equal")
    != execution["path_descriptor_equal"]
    or value.get("path_descriptor_equal") is not True
):
    raise SystemExit("accepted-binding rehearsal differs from its binding")
print(hashlib.sha256(payload).hexdigest())
PY

require_absent "$FORMAL_BINDING_ROOT"
require_absent "$FORMAL"
require_absent "$FORMAL_MARKER"
require_absent "$FORMAL_AUDIT"
require_absent "$FORMAL_AUDIT_MARKER"
require_absent "$SEMANTIC_REVIEW"
require_absent "$ADJUDICATION_MARKER"
require_absent "$ADJUDICATION_PACKAGE"
```

Any nonzero status, stderr byte, noncanonical or misbound field, or created
formal path retires v1.16. Do not create or remove a formal root to retry it.

## Sole formal producer

```bash
require_absent "$FORMAL_BINDING_ROOT"
require_absent "$FORMAL"
require_absent "$FORMAL_MARKER"

"${SAFE_ENV[@]}" "$RUNTIME_PY" -I -S -B - "$FORMAL_BINDING_ROOT" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
parent = root.parent
if (
    not root.is_absolute()
    or parent.resolve(strict=True) != parent
    or root.exists()
    or root.is_symlink()
):
    raise SystemExit("formal binding root is not one absent canonical child")
os.mkdir(root)
descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
test -d "$FORMAL_BINDING_ROOT" && test ! -L "$FORMAL_BINDING_ROOT"
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
cmp -- \
  "$BINDING_ATTEMPT/development-result-qualification.json" \
  "$FORMAL/development-result-qualification.json"
```

The exclusive, fsynced creation of `FORMAL_BINDING_ROOT` is the final
operator-side preparation and supplies the canonical existing parent required
by `ProducerAttempt`; from that point onward, any invocation refusal or failure
is terminal. Any child below `FORMAL_BINDING_ROOT`, including a noncanonical
sibling left by an earlier launcher, contaminates and consumes the v1.16
attempt. Never remove the prepared root, resume, or rerun. A nonzero return
after marker publication is terminal evidence. A completed formal producer
must preserve the exact binding-attempt preflight receipt at its root; the
independent formal auditor reconstructs the live authorization lineage and
rejects a missing or
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
  echo "formal audit failed before publishing an authenticated attempt; v1.16 is retired" >&2
  exit 2
fi
"$QA_PY" -I -B -c \
  'import os,sys; from pathlib import Path; from bench.world_model_lifecycle.operator import inspect_unfinalized_operator_attempt,outer_completion_marker,verify_operator_attempt; attempt=Path(sys.argv[1]); terminal=attempt/"operator-attempt.json"; (verify_operator_attempt if os.path.lexists(outer_completion_marker(terminal)) else inspect_unfinalized_operator_attempt)(attempt)' \
  "$FORMAL_AUDIT"
```

Only an authenticated canonical finalized or explicitly unfinalized attempt
establishes that the sole v1.16 formal-audit claim was consumed, even if no
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
bounded WM-001 claim. Every other terminal state retires v1.16 without
same-version repair or retry.
