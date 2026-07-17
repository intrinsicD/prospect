"""One-shot sealed lifecycle for the MM-009 causal prediction experiment.

Preparation is deliberately outcome blind: it may census and opaque-copy the pinned
MM-007 package, but it cannot deserialize its evidence or frame archive.  Real bytes
become scientific inputs only after the immutable formal marker and formal synthetic
gate.  Every source-only prediction is then produced in a fresh Landlock/seccomp
process, chained by the supervisor, and frozen before a score-only child can open a
detached target.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np

from bench.multimodal_resolution_diagnostics import experiment as mm007_experiment
from bench.multimodal_resolution_diagnostics import method as mm007_method

from . import decision, launcher, preparation, records, runtime, score_worker, scoring, synthetic_controls, worker

SCHEMA_VERSION: Final = "mm009-formal-v1"
EXPERIMENT_ID: Final = "MM-009"
REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT: Final = Path("bench/multimodal_causal_diagnostics/results/MM-009")
EXPECTED_OUTPUT: Final = REPO_ROOT / DEFAULT_OUTPUT
PARENT_ROOT: Final = REPO_ROOT / "bench/multimodal_resolution_diagnostics/results/MM-007"
PROTOCOL_DOC: Final = REPO_ROOT / "docs/research/2026-07-16-mm009-causal-deformation-appearance-prediction-protocol.md"
PRE_REAL_AUDIT_DOC: Final = REPO_ROOT / "docs/research/2026-07-16-mm009-pre-real-audit.json"

PROTOCOL_FILE: Final = Path("MM-009-protocol.md")
AUDIT_FILE: Final = Path("pre-real-audit.json")
CONFIG_FILE: Final = Path("config.json")
WORKER_CONFIG_FILE: Final = Path("worker-config.json")
SUPERSESSION_FILE: Final = Path("MM-008-supersession.json")
INPUT_MANIFEST_FILE: Final = Path("input-manifest.json")
FREEZE_FILE: Final = Path("freeze-record.json")
START_FILE: Final = Path("formal-start.json")
CONTROL_FILE: Final = Path("formal-synthetic-controls.json")
DETACHMENT_FILE: Final = Path("detachment-index.json")
POST_ISOLATION_FILE: Final = Path("post-detachment-isolation.json")
PREDICTION_ATTEMPT_FILE: Final = Path("prediction-attempt.json")
PREDICTION_FREEZE_FILE: Final = Path("prediction-freeze.json")
PRE_SCORE_BUDGET_FILE: Final = Path("pre-score-budget.json")
FUTURE_ISOLATION_FILE: Final = Path("future-isolation.json")
SCORE_ATTEMPT_FILE: Final = Path("score-attempt.json")
EVIDENCE_FILE: Final = Path("MM-009-evidence.json")
RESULT_FILE: Final = Path("MM-009-results.json")
REPORT_FILE: Final = Path("MM-009-report.md")
ARTIFACT_MANIFEST_FILE: Final = Path("artifact-manifest.json")

PARENT_COPY_ROOT: Final = Path("inputs/MM-007")
RUNTIME_ROOT: Final = Path("runtime/source-only")
DEPENDENCY_ROOT: Final = Path("runtime/numpy-only")
STDLIB_ROOT: Final = Path("runtime/python-stdlib")
CUSTODY_RUNTIME_ROOT: Final = Path("runtime/custody-only")
PRE_PROBE_ROOT: Final = Path("isolation/pre-marker")
POST_PROBE_ROOT: Final = Path("isolation/post-detachment")
SOURCE_ROOT: Final = Path("rows/source")
TARGET_ROOT: Final = Path("rows/target")
PREDICTION_ROOT: Final = Path("predictions")
SCORE_ROOT: Final = Path("scores")
WORK_ROOT: Final = Path(".formal-work")

V22_SOURCE_PINS: Final[Mapping[str, str]] = {
    "bench/multimodal_mechanism_diagnostics/fitting_v22.py": (
        "e3ecdc27077d8a35d818359f3d6ada92cc91c1cff2c66e9a27af4d1bef33cb0b"
    ),
    "bench/multimodal_mechanism_diagnostics/geometry_v22.py": (
        "041f161a09b73a43343a5f33ba99b6074d94a3fbb03cd5bd04045d2be7a03044"
    ),
    "bench/multimodal_mechanism_diagnostics/global_v22.py": (
        "1ab51dcf9ddf7a3731372541f6dcc34b60012b4e6d0ef1a1010925bdf652f893"
    ),
    "bench/multimodal_mechanism_diagnostics/nongrid_v22.py": (
        "cf504ecfdb55ced94456189feaaaeef2c5070c7ee1da66c74fc71f8ec4d6415a"
    ),
}
V22_SYNTHETIC_SOURCE_PINS: Final[Mapping[str, str]] = {
    "bench/multimodal_mechanism_diagnostics/calibration_v22.py": (
        "225283b2adc002ba154611bf65ad4657cb4a1c7dd6ce1560fe3328c6a43e97d4"
    ),
    "bench/multimodal_mechanism_diagnostics/synthetic_v22.py": (
        "f119a6040e5e57efeadf37cb7eb6fe6e64f7d166909e0dbf09f56c6ca5ae0a82"
    ),
}
MM007_VERIFIER_SOURCES: Final = (
    Path("bench/multimodal_resolution_diagnostics/experiment.py"),
    Path("bench/multimodal_resolution_diagnostics/method.py"),
)
MM008_PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
MM008_RESULT_ROOT: Final = REPO_ROOT / "bench/multimodal_mechanism_diagnostics/results/MM-008"
MM008_FORBIDDEN_FORMAL_DOCS: Final = (
    REPO_ROOT / "docs/research/2026-07-16-mm008-v2.2-preseal-review.json",
    REPO_ROOT / "docs/research/2026-07-16-mm008-v2.2-freeze-record.json",
    REPO_ROOT / "docs/research/2026-07-16-mm008-v2.2-reviewer-handoff.json",
    REPO_ROOT / "docs/research/2026-07-16-mm008-v2.2-formal-start.json",
    REPO_ROOT / "docs/research/2026-07-16-mm008-v2.2-fast-verification.json",
    REPO_ROOT / "docs/research/2026-07-16-mm008-v2.2-semantic-verification.json",
)

MAX_WORKERS: Final = 8
WORKER_TIMEOUT_SECONDS: Final = 900
TOTAL_WALL_SECONDS: Final = 4 * 60 * 60
MAX_ARTIFACT_BYTES: Final = 2_000_000_000
MAX_JSON_BYTES: Final = 256_000_000
MAX_TEXT_BYTES: Final = 16_000_000
MAX_BUDGET_FILE_BYTES: Final = 1_000_000
MAX_FUTURE_FILE_BYTES: Final = 1_000_000
MAX_SCORE_ATTEMPT_BYTES: Final = 1_000_000
MAX_SCORE_FILE_BYTES: Final = 1_000_000
MAX_EVIDENCE_FILE_BYTES: Final = 16_000_000
MAX_RESULT_FILE_BYTES: Final = 1_000_000
MAX_REPORT_FILE_BYTES: Final = 1_000_000
MAX_ARTIFACT_MANIFEST_FILE_BYTES: Final = 16_000_000

SYSTEMD_RUN: Final = Path("/usr/bin/systemd-run")
SYSTEMCTL: Final = Path("/usr/bin/systemctl")
ENV_EXECUTABLE: Final = Path("/usr/bin/env")
SYSTEMD_UNIT_PREFIX: Final = "mm009-custody"


class InvalidMM009Package(ValueError):
    """Fail-closed package/lifecycle classification."""

    classification = "invalid_MM009"


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise InvalidMM009Package(f"formal total-wall deadline exceeded: {stage}")


def _deadline_timeout(deadline: float, ceiling_seconds: float, stage: str) -> float:
    _check_deadline(deadline, stage)
    return min(ceiling_seconds, max(0.001, deadline - time.monotonic()))


def _run_cancellable_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    cancel_event: threading.Event | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one isolated child while retaining prompt process-group cancellation."""

    with tempfile.TemporaryFile(mode="w+b") as stdout_stream, tempfile.TemporaryFile(mode="w+b") as stderr_stream:
        process = subprocess.Popen(  # noqa: S603 - closed, supervisor-constructed argv
            command,
            cwd=cwd,
            env=(launcher.frozen_environment() if environment is None else dict(environment)),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        child_deadline = time.monotonic() + timeout_seconds
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise InvalidMM009Package("isolated child cancelled after sibling failure")
                remaining = child_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    process.wait(timeout=min(0.25, remaining))
                except subprocess.TimeoutExpired:
                    continue
            if _process_group_exists(process.pid):
                raise InvalidMM009Package("isolated child left a live descendant process")
            stdout = _stream_tail(stdout_stream)
            stderr = _stream_tail(stderr_stream)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        except BaseException:
            _kill_process_tree(process)
            raise


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None or _process_group_exists(process.pid):
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired as error:
                raise InvalidMM009Package("isolated child process group could not be reaped") from error


def _proc_snapshot() -> dict[int, tuple[int, int]]:
    """Return Linux PID -> (PPID, starttime) without trusting process names."""

    snapshot: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            payload = (entry / "stat").read_text(encoding="ascii")
            suffix = payload[payload.rfind(")") + 2 :].split()
            if suffix[0] == "Z":
                continue
            snapshot[int(entry.name)] = (int(suffix[1]), int(suffix[19]))
        except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
            continue
    return snapshot


def _descendant_identities(root_pid: int) -> dict[int, int]:
    snapshot = _proc_snapshot()
    descendants: dict[int, int] = {}
    frontier = {root_pid}
    while frontier:
        children = {
            pid for pid, (parent, _starttime) in snapshot.items() if parent in frontier and pid not in descendants
        }
        for pid in children:
            descendants[pid] = snapshot[pid][1]
        frontier = children
    return descendants


def _same_process(pid: int, starttime: int) -> bool:
    return _proc_snapshot().get(pid, (-1, -1))[1] == starttime


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Freeze and kill a child plus descendants, including nested new sessions."""

    try:
        os.kill(process.pid, signal.SIGSTOP)
    except ProcessLookupError:
        _kill_process_group(process)
        return

    descendants: dict[int, int] = {}
    stable_scans = 0
    while stable_scans < 2:
        observed = _descendant_identities(process.pid)
        new = {pid: starttime for pid, starttime in observed.items() if pid not in descendants}
        for pid in new:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                continue
        if new:
            descendants.update(new)
            stable_scans = 0
        else:
            stable_scans += 1

    for pid, starttime in reversed(tuple(descendants.items())):
        if _same_process(pid, starttime):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    _kill_process_group(process)
    survivors = {pid for pid, starttime in descendants.items() if _same_process(pid, starttime)}
    deadline = time.monotonic() + 1.0
    while survivors and time.monotonic() < deadline:
        time.sleep(0.01)
        survivors = {pid for pid, starttime in descendants.items() if _same_process(pid, starttime)}
    if survivors:
        raise InvalidMM009Package("isolated descendant process tree could not be reaped")


def _systemd_client_environment() -> dict[str, str]:
    runtime_directory = Path(f"/run/user/{os.getuid()}")
    bus = runtime_directory / "bus"
    if runtime_directory.is_symlink() or not runtime_directory.is_dir() or bus.is_symlink() or not bus.exists():
        raise InvalidMM009Package("user systemd bus boundary is unavailable")
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(runtime_directory),
    }


def _systemd_supervisor_manifest() -> dict[str, records.JsonValue]:
    executables: dict[str, records.JsonValue] = {}
    for path in (SYSTEMD_RUN, SYSTEMCTL, ENV_EXECUTABLE):
        if path.is_symlink() or not path.is_file():
            raise InvalidMM009Package("systemd supervisor executable boundary differs")
        executables[str(path)] = records.file_record(path)
    controllers = Path("/sys/fs/cgroup/cgroup.controllers")
    if controllers.is_symlink() or not controllers.is_file():
        raise InvalidMM009Package("cgroup v2 boundary is unavailable")
    _systemd_client_environment()
    scientific_environment: dict[str, records.JsonValue] = {
        name: value for name, value in launcher.frozen_environment().items()
    }
    return {
        "cgroup_version": 2,
        "executables": executables,
        "kill_mode": "control-group",
        "kill_signal": "SIGKILL",
        "runtime_max_seconds": TOTAL_WALL_SECONDS,
        "schema_version": "mm009-systemd-cgroup-supervisor-v1",
        "scientific_environment": scientific_environment,
        "send_sigkill": True,
        "unit_prefix": SYSTEMD_UNIT_PREFIX,
    }


def _assert_current_cgroup_role(role: str) -> None:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise InvalidMM009Package("current cgroup identity is unavailable") from error
    prefix = f"{SYSTEMD_UNIT_PREFIX}-{role}-"
    for line in lines:
        path = line.split(":", 2)[-1]
        unit = Path(path).name
        if not unit.startswith(prefix) or not unit.endswith(".service"):
            continue
        identity = unit[len(prefix) : -len(".service")].split("-")
        if len(identity) == 3 and all(value.isdecimal() for value in identity):
            return
    raise InvalidMM009Package(f"{role} lifecycle is outside its frozen cgroup")


def _systemctl_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return _run_cancellable_process(
        (str(SYSTEMCTL), "--user", *arguments),
        cwd=REPO_ROOT,
        timeout_seconds=5.0,
        environment=_systemd_client_environment(),
    )


def _cleanup_systemd_unit(unit: str) -> None:
    for arguments in (
        ("kill", "--kill-whom=all", "--signal=SIGKILL", unit),
        ("stop", unit),
        ("reset-failed", unit),
    ):
        try:
            _systemctl_command(*arguments)
        except (InvalidMM009Package, subprocess.TimeoutExpired):
            if arguments[0] != "reset-failed":
                continue
    state = _systemctl_command("is-active", unit)
    state_name = state.stdout.strip()
    if state.returncode == 0 or state_name not in {"failed", "inactive", "unknown"}:
        raise InvalidMM009Package(f"systemd custody unit remained active: {unit}")


def _run_cgroup_supervised_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    role: str,
) -> subprocess.CompletedProcess[str]:
    """Run a lifecycle root in a transient cgroup that contains nested sessions."""

    if (
        not role
        or len(role) > 32
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in role)
    ):
        raise InvalidMM009Package("systemd custody role grammar differs")
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise InvalidMM009Package("systemd custody timeout differs")
    if not command or not Path(command[0]).is_absolute():
        raise InvalidMM009Package("systemd custody command boundary differs")
    try:
        command_executable = Path(command[0]).resolve(strict=True)
    except OSError as error:
        raise InvalidMM009Package("systemd custody command is unavailable") from error
    if not command_executable.is_file():
        raise InvalidMM009Package("systemd custody command boundary differs")
    normalized_command = (str(command_executable), *command[1:])
    if cwd.is_symlink() or not cwd.is_absolute() or not cwd.is_dir():
        raise InvalidMM009Package("systemd custody working directory differs")
    unit = f"{SYSTEMD_UNIT_PREFIX}-{role}-{os.getpid()}-{threading.get_native_id()}-{time.monotonic_ns()}.service"
    guard = (
        "import os,sys;"
        "unit=sys.argv[1];"
        "paths=[line.split(':',2)[-1] for line in "
        "open('/proc/self/cgroup',encoding='ascii').read().splitlines()];"
        "assert any(path.endswith('/'+unit) for path in paths);"
        "os.execv(sys.argv[2],sys.argv[2:])"
    )
    frozen_environment = launcher.frozen_environment()
    service_command = (
        str(ENV_EXECUTABLE),
        "-i",
        *(f"{name}={value}" for name, value in sorted(frozen_environment.items())),
        sys.executable,
        "-I",
        "-B",
        "-c",
        guard,
        unit,
        *normalized_command,
    )
    systemd_command = (
        str(SYSTEMD_RUN),
        "--user",
        "--wait",
        "--collect",
        "--quiet",
        "--pipe",
        "--expand-environment=no",
        f"--unit={unit}",
        "--service-type=exec",
        f"--property=RuntimeMaxSec={timeout_seconds:.6f}s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGKILL",
        "--property=SendSIGKILL=yes",
        "--property=Restart=no",
        f"--property=WorkingDirectory={cwd}",
        *service_command,
    )
    started = time.monotonic()
    try:
        result = _run_cancellable_process(
            systemd_command,
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds + 2.0,
            environment=_systemd_client_environment(),
        )
        if result.returncode != 0 and time.monotonic() - started >= timeout_seconds:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        return subprocess.CompletedProcess(
            tuple(command),
            result.returncode,
            result.stdout,
            result.stderr,
        )
    finally:
        _cleanup_systemd_unit(unit)


def _systemd_containment_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="mm009-cgroup-preflight-", dir="/tmp") as temporary:
        sentinel = Path(temporary).resolve() / "escaped.txt"
        script = (
            "import subprocess,sys;"
            "subprocess.Popen((sys.executable,'-I','-S','-B','-c',"
            "\"import pathlib,sys,time;time.sleep(0.4);pathlib.Path(sys.argv[1]).write_text('escaped')\","
            "sys.argv[1]),start_new_session=True)"
        )
        result = _run_cgroup_supervised_process(
            (sys.executable, "-I", "-S", "-B", "-c", script, str(sentinel)),
            cwd=REPO_ROOT,
            timeout_seconds=2.0,
            role="preflight",
        )
        if result.returncode != 0 or result.stdout or result.stderr:
            raise InvalidMM009Package("systemd cgroup containment preflight failed")
        time.sleep(0.5)
        if sentinel.exists() or sentinel.is_symlink():
            raise InvalidMM009Package("systemd cgroup containment preflight escaped")


def _stream_tail(stream: object, maximum_bytes: int = 4000) -> str:
    if not hasattr(stream, "fileno") or not hasattr(stream, "seek") or not hasattr(stream, "read"):
        raise InvalidMM009Package("isolated child capture stream differs")
    descriptor = cast(int, stream.fileno())
    size = os.fstat(descriptor).st_size
    stream.seek(max(0, size - maximum_bytes))
    payload = stream.read(maximum_bytes)
    if not isinstance(payload, bytes):
        raise InvalidMM009Package("isolated child capture stream is not binary")
    return payload.decode("utf-8", errors="replace")


def _strict_json(path: Path) -> object:
    def reject_constant(value: str) -> None:
        raise InvalidMM009Package(f"nonfinite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise InvalidMM009Package(f"duplicate JSON key is forbidden: {key}")
            value[key] = item
        return value

    try:
        payload = records.read_regular_bytes(path, maximum_bytes=MAX_JSON_BYTES).decode("ascii")
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, records.RecordValidationError) as error:
        raise InvalidMM009Package(f"strict JSON decode failed: {path}") from error


def _write_bounded_immutable_json(
    path: Path,
    value: records.JsonValue,
    *,
    maximum_bytes: int,
    label: str,
) -> None:
    payload = records.json_file_bytes(value)
    if len(payload) > maximum_bytes:
        raise InvalidMM009Package(f"{label} exceeds its frozen byte bound")
    records.write_immutable_bytes_exclusive(path, payload)


def _write_bounded_immutable_bytes(
    path: Path,
    payload: bytes,
    *,
    maximum_bytes: int,
    label: str,
) -> None:
    if len(payload) > maximum_bytes:
        raise InvalidMM009Package(f"{label} exceeds its frozen byte bound")
    records.write_immutable_bytes_exclusive(path, payload)


def _as_json(value: object) -> records.JsonValue:
    """Convert frozen dataclasses/tuples into the closed canonical JSON grammar."""

    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise InvalidMM009Package("cannot serialize a nonfinite result")
        return cast(records.JsonValue, value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _as_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, tuple | list):
        return [_as_json(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise InvalidMM009Package("JSON mapping contains a non-string key")
        return {cast(str, key): _as_json(item) for key, item in value.items()}
    raise InvalidMM009Package(f"unsupported result serialization type: {type(value).__name__}")


def _assert_output(output: Path) -> Path:
    candidate = output if output.is_absolute() else REPO_ROOT / output
    if candidate.resolve() != EXPECTED_OUTPUT.resolve():
        raise InvalidMM009Package(f"MM-009 output must be {DEFAULT_OUTPUT}")
    cursor = candidate
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise InvalidMM009Package(f"output path contains a symlink: {cursor}")
        cursor = cursor.parent
    if candidate.exists() and not candidate.is_dir():
        raise InvalidMM009Package("output path is not a directory")
    return candidate


def _scientific_source_paths() -> tuple[Path, ...]:
    package = REPO_ROOT / "bench/multimodal_causal_diagnostics"
    own = tuple(sorted((path.relative_to(REPO_ROOT) for path in package.glob("*.py")), key=str))
    required = {
        Path("bench/multimodal_causal_diagnostics/__init__.py"),
        Path("bench/multimodal_causal_diagnostics/__main__.py"),
        Path("bench/multimodal_causal_diagnostics/experiment.py"),
        Path("bench/multimodal_causal_diagnostics/predictor.py"),
        Path("bench/multimodal_causal_diagnostics/preparation.py"),
        Path("bench/multimodal_causal_diagnostics/records.py"),
        Path("bench/multimodal_causal_diagnostics/scoring.py"),
        Path("bench/multimodal_causal_diagnostics/decision.py"),
        Path("bench/multimodal_causal_diagnostics/synthetic_controls.py"),
        Path("bench/multimodal_causal_diagnostics/worker.py"),
        Path("bench/multimodal_causal_diagnostics/runtime.py"),
        Path("bench/multimodal_causal_diagnostics/launcher.py"),
        Path("bench/multimodal_causal_diagnostics/probe.py"),
        Path("bench/multimodal_causal_diagnostics/sandbox.py"),
    }
    if not required.issubset(set(own)):
        raise InvalidMM009Package("MM-009 scientific package membership is incomplete")
    dependencies = tuple(Path(name) for name in sorted({*V22_SOURCE_PINS, *V22_SYNTHETIC_SOURCE_PINS}))
    return tuple(
        sorted(
            {
                *own,
                *dependencies,
                *MM007_VERIFIER_SOURCES,
                PROTOCOL_DOC.relative_to(REPO_ROOT),
            },
            key=str,
        )
    )


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _scientific_source_paths():
        path = REPO_ROOT / relative
        output[str(relative)] = records.file_sha256(path)
    for source_name, expected in V22_SOURCE_PINS.items():
        if output.get(source_name) != expected:
            raise InvalidMM009Package(f"pinned v2.2 source differs: {source_name}")
    for source_name, expected in V22_SYNTHETIC_SOURCE_PINS.items():
        if output.get(source_name) != expected:
            raise InvalidMM009Package(f"pinned v2.2 synthetic source differs: {source_name}")
    return output


def _protocol_sha256() -> str:
    return cast(str, records.file_sha256(PROTOCOL_DOC))


def _runtime_source_manifest_sha256() -> dict[str, str]:
    manifests = {
        "custody_runtime": runtime.custody_runtime_source_manifest(),
        "numpy_dependency": runtime.numpy_dependency_source_manifest(),
        "python_stdlib": runtime.stdlib_source_manifest(),
    }
    return {
        name: records.require_sha256(manifest.get("manifest_sha256"), f"{name} source-manifest SHA-256")
        for name, manifest in manifests.items()
    }


def _scientific_config(protocol_sha256: str) -> dict[str, records.JsonValue]:
    source_hashes = _source_hashes()
    runtime_sources = _runtime_source_manifest_sha256()
    return {
        "decision": {
            "activity_mse_min": decision.ACTIVITY_MSE_MIN,
            "control_factor": decision.CONTROL_FACTOR,
            "null_inconclusive_min": decision.NULL_INCONCLUSIVE_MIN,
            "null_invalid_count": decision.NULL_INVALID_COUNT,
            "primary_factor": decision.PRIMARY_FACTOR,
            "range_warning_videos": decision.RANGE_WARNING_VIDEOS,
            "required_directional": decision.REQUIRED_DIRECTIONAL,
            "required_support": decision.REQUIRED_SUPPORT,
        },
        "experiment_id": EXPERIMENT_ID,
        "matched_identity_sha256": preparation.MATCHED_IDENTITY_SHA256,
        "matched_rows": preparation.MATCHED_ROWS,
        "mm008_protocol_sha256": MM008_PROTOCOL_SHA256,
        "host_runtime_trust": _as_json(runtime.host_runtime_trust_manifest()),
        "parent_pins": {
            name: {"mode": pin.mode, "sha256": pin.sha256} for name, pin in sorted(records.PARENT_PINS.items())
        },
        "parent_excluded_entries": [
            {
                "authority": "excluded_non_authoritative_lineage_only",
                "copied": False,
                "name": name,
                "required_type": "non_symlink_directory",
                "traversed": False,
            }
            for name in records.PARENT_EXCLUDED_NONAUTHORITATIVE_DIRECTORIES
        ],
        "prediction_roles": list(worker.PREDICTION_ROLES),
        "protocol_sha256": protocol_sha256,
        "systemd_cgroup_supervisor": _systemd_supervisor_manifest(),
        "resources": {
            "artifact_bytes_max": MAX_ARTIFACT_BYTES,
            "artifact_manifest_file_bytes_max": MAX_ARTIFACT_MANIFEST_FILE_BYTES,
            "budget_file_bytes_max": MAX_BUDGET_FILE_BYTES,
            "evidence_file_bytes_max": MAX_EVIDENCE_FILE_BYTES,
            "future_file_bytes_max": MAX_FUTURE_FILE_BYTES,
            "report_file_bytes_max": MAX_REPORT_FILE_BYTES,
            "result_file_bytes_max": MAX_RESULT_FILE_BYTES,
            "score_attempt_bytes_max": MAX_SCORE_ATTEMPT_BYTES,
            "score_file_bytes_max": MAX_SCORE_FILE_BYTES,
            "total_wall_seconds": TOTAL_WALL_SECONDS,
            "worker_count": MAX_WORKERS,
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        },
        "runtime_source_manifest_sha256": {name: digest for name, digest in runtime_sources.items()},
        "schema_version": "mm009-scientific-config-v1",
        "source_hashes": {name: digest for name, digest in source_hashes.items()},
        "synthetic_seeds": list(synthetic_controls.ALL_SEEDS),
        "v22_source_pins": {name: digest for name, digest in V22_SOURCE_PINS.items()},
        "v22_synthetic_source_pins": {name: digest for name, digest in V22_SYNTHETIC_SOURCE_PINS.items()},
        "video_specs": [[video, fold, rows] for video, fold, rows in decision.VIDEO_SPECS],
    }


def _config_bundle() -> tuple[dict[str, records.JsonValue], str, dict[str, records.JsonValue]]:
    protocol = _protocol_sha256()
    if not (
        protocol
        == records.PROTOCOL_SHA256
        == preparation.PROTOCOL_SHA256
        == scoring.PROTOCOL_SHA256
        == decision.PROTOCOL_SHA256
    ):
        raise InvalidMM009Package("MM-009 modules bind different protocol bytes")
    scientific = _scientific_config(protocol)
    config_sha256 = records.canonical_json_sha256(scientific, protocol_sha256=protocol)
    complete: dict[str, records.JsonValue] = {
        "config_sha256": config_sha256,
        "scientific_config": scientific,
        "schema_version": "mm009-config-record-v1",
    }
    worker_config: dict[str, records.JsonValue] = {
        "config_sha256": config_sha256,
        "prediction_roles": list(worker.PREDICTION_ROLES),
        "protocol_sha256": protocol,
        "schema_version": "mm009-worker-config-v1",
    }
    return complete, config_sha256, worker_config


def audit_template() -> dict[str, records.JsonValue]:
    config, config_sha256, _ = _config_bundle()
    scientific = cast(dict[str, records.JsonValue], config["scientific_config"])
    source_hashes = cast(dict[str, records.JsonValue], scientific["source_hashes"])
    runtime_sources = cast(dict[str, records.JsonValue], scientific["runtime_source_manifest_sha256"])
    protocol = cast(str, scientific["protocol_sha256"])
    return {
        "claim_boundary": "pre-real design, implementation, controls, and custody only; no MM-009 real outcome",
        "config_sha256": config_sha256,
        "decision": "GO",
        "experiment_id": EXPERIMENT_ID,
        "findings_closed": [],
        "protocol_sha256": protocol,
        "reviewed_source_hashes": source_hashes,
        "reviewed_source_hashes_sha256": records.canonical_json_sha256(source_hashes, protocol_sha256=protocol),
        "reviewed_runtime_source_manifest_sha256": runtime_sources,
        "reviewer": "prospect-results-audit",
        "schema_version": "mm009-pre-real-audit-v1",
    }


def _validate_audit(path: Path) -> dict[str, records.JsonValue]:
    value = _strict_json(path)
    if type(value) is not dict:
        raise InvalidMM009Package("pre-real audit must be a JSON object")
    expected = audit_template()
    required = set(expected)
    if set(value) != required:
        raise InvalidMM009Package("pre-real audit schema differs")
    checked = cast(dict[str, records.JsonValue], value)
    for key in required - {"findings_closed"}:
        if checked[key] != expected[key]:
            raise InvalidMM009Package(f"pre-real audit binding differs: {key}")
    findings = checked["findings_closed"]
    if type(findings) is not list or any(type(item) is not str or not item for item in findings):
        raise InvalidMM009Package("pre-real audit closed findings must be a string list")
    if checked["decision"] != "GO":
        raise InvalidMM009Package("pre-real audit did not authorize freeze")
    return checked


def _supersession_record(protocol_sha256: str) -> dict[str, records.JsonValue]:
    if MM008_RESULT_ROOT.exists() or MM008_RESULT_ROOT.is_symlink():
        raise InvalidMM009Package("canonical MM-008 result root exists; supersession is false")
    present = [str(path.relative_to(REPO_ROOT)) for path in MM008_FORBIDDEN_FORMAL_DOCS if path.exists()]
    if present:
        raise InvalidMM009Package(f"MM-008 formal records exist: {present}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "forbidden_formal_records_absent": True,
        "mm008_result_root_absent": True,
        "mm008_v22_protocol_sha256": MM008_PROTOCOL_SHA256,
        "protocol_sha256": protocol_sha256,
        "reserved_or_challenge_use_authorized": False,
        "schema_version": "mm009-mm008-supersession-v1",
        "status": "untouched_MM008_real_target_route_superseded_before_formal_use",
    }


def _file_map(root: Path, paths: Sequence[Path]) -> dict[str, records.JsonValue]:
    return {str(path): records.file_record(root / path) for path in sorted(paths, key=str)}


def _validate_opaque_parent_copy(
    copy_root: Path,
    *,
    live_root: Path = PARENT_ROOT,
) -> dict[str, dict[str, records.JsonValue]]:
    """Validate the exact immutable copy without requiring the live file modes.

    The sealed MM-007 package contains a mixture of 0644 and 0444 files.  MM-009
    deliberately copies every opaque byte string into its own custody as 0444, so a
    copied record must not be compared wholesale with the live record.  Membership,
    byte count, and digest remain identical; only the copied mode is normalized.
    """

    try:
        root_stat = copy_root.lstat()
    except FileNotFoundError as error:
        raise InvalidMM009Package("opaque parent copy root is missing") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise InvalidMM009Package("opaque parent copy root must be a real directory")
    expected = set(records.PARENT_PINS)
    with os.scandir(copy_root) as entries:
        actual = {entry.name for entry in entries}
    if actual != expected:
        raise InvalidMM009Package(
            f"opaque parent copy membership differs: expected={sorted(expected)}, actual={sorted(actual)}"
        )

    live = records.validate_pinned_parent_tree(
        live_root,
        records.PARENT_PINS,
        excluded_nonauthoritative_directories=records.PARENT_EXCLUDED_NONAUTHORITATIVE_DIRECTORIES,
    )
    copied: dict[str, dict[str, records.JsonValue]] = {}
    for name in sorted(expected):
        pin = records.PARENT_PINS[name]
        copied_record = records.file_record(copy_root / name)
        live_record = live[name]
        if (
            copied_record["mode"] != 0o444
            or copied_record["sha256"] != pin.sha256
            or copied_record["sha256"] != live_record["sha256"]
            or copied_record["bytes"] != live_record["bytes"]
        ):
            raise InvalidMM009Package(f"opaque immutable parent copy differs: {name}")
        copied[name] = copied_record
    return copied


def _pre_manifest(output: Path) -> dict[str, records.JsonValue]:
    config, config_sha256, worker_config = _config_bundle()
    scientific_config = cast(dict[str, records.JsonValue], config["scientific_config"])
    expected_runtime_sources = cast(
        dict[str, records.JsonValue],
        scientific_config["runtime_source_manifest_sha256"],
    )
    protocol = _protocol_sha256()
    audit = _validate_audit(PRE_REAL_AUDIT_DOC)
    if records.read_regular_bytes(output / PROTOCOL_FILE, maximum_bytes=MAX_TEXT_BYTES) != records.read_regular_bytes(
        PROTOCOL_DOC,
        maximum_bytes=MAX_TEXT_BYTES,
    ):
        raise InvalidMM009Package("copied protocol differs from reviewed bytes")
    if _strict_json(output / AUDIT_FILE) != audit:
        raise InvalidMM009Package("copied pre-real audit differs")
    if _strict_json(output / CONFIG_FILE) != config or _strict_json(output / WORKER_CONFIG_FILE) != worker_config:
        raise InvalidMM009Package("copied configuration differs")
    parent = _validate_opaque_parent_copy(output / PARENT_COPY_ROOT)
    runtime_record = runtime.runtime_manifest(output / RUNTIME_ROOT)
    closure_records: dict[str, dict[str, records.JsonValue]] = {
        "custody_runtime": runtime.custody_runtime_manifest(output / CUSTODY_RUNTIME_ROOT),
        "numpy_dependency": runtime.numpy_dependency_manifest(output / DEPENDENCY_ROOT),
        "python_stdlib": runtime.stdlib_manifest(output / STDLIB_ROOT),
    }
    for name, closure_record in closure_records.items():
        if closure_record.get("source_manifest_sha256") != expected_runtime_sources.get(name):
            raise InvalidMM009Package(f"prepared runtime closure differs from pre-audit source binding: {name}")
    host_runtime_trust = runtime.host_runtime_trust_manifest()
    if host_runtime_trust != scientific_config["host_runtime_trust"]:
        raise InvalidMM009Package("host runtime trust boundary differs from the audited configuration")
    pre_probe = _strict_json(output / PRE_PROBE_ROOT / "isolation-probe.json")
    live_python_root_count = len(runtime.live_python_roots())
    if (
        type(pre_probe) is not dict
        or pre_probe.get("schema_version") != "mm009-isolation-probe-v3"
        or pre_probe.get("requested_denied_path_count") != 3
        or pre_probe.get("live_python_roots_denied") != live_python_root_count
        or pre_probe.get("denied_path_count") != 3 + live_python_root_count
        or pre_probe.get("network_families_denied") != 3
        or pre_probe.get("network_socket_variants_denied") != 6
        or pre_probe.get("process_vm_readv_denied") is not True
    ):
        raise InvalidMM009Package("pre-marker isolation evidence differs")
    fixed_paths = (
        PROTOCOL_FILE,
        AUDIT_FILE,
        CONFIG_FILE,
        WORKER_CONFIG_FILE,
        SUPERSESSION_FILE,
        Path("isolation/pre-marker/allowed.txt"),
        Path("isolation/pre-marker/sentinel.txt"),
        Path("isolation/pre-marker/isolation-probe.json"),
        *(PARENT_COPY_ROOT / name for name in sorted(records.PARENT_PINS)),
    )
    if _strict_json(output / SUPERSESSION_FILE) != _supersession_record(protocol):
        raise InvalidMM009Package("MM-008 supersession record differs")
    return {
        "audit": audit,
        "config_sha256": config_sha256,
        "experiment_id": EXPERIMENT_ID,
        "fixed_artifacts": _file_map(output, fixed_paths),
        "host_runtime_trust": _as_json(host_runtime_trust),
        "opaque_parent": {name: item for name, item in parent.items()},
        "protocol_sha256": protocol,
        "runtime": runtime_record,
        "runtime_closures": {name: _as_json(closure_record) for name, closure_record in closure_records.items()},
        "runtime_source_manifest_sha256": {name: digest for name, digest in expected_runtime_sources.items()},
        "schema_version": "mm009-input-manifest-v1",
        "source_hashes": {name: digest for name, digest in _source_hashes().items()},
    }


def prepare(
    output: Path = DEFAULT_OUTPUT,
    *,
    audit_path: Path = PRE_REAL_AUDIT_DOC,
) -> dict[str, records.JsonValue]:
    """Opaque-copy and authority-probe the pre-real package without decoding science."""

    destination = _assert_output(output)
    if destination.exists():
        raise FileExistsError("MM-009 output must be wholly absent before preparation")
    if audit_path.resolve() != PRE_REAL_AUDIT_DOC.resolve():
        raise InvalidMM009Package("MM-009 audit must use the canonical reviewed path")
    _validate_audit(audit_path)
    records.validate_pinned_parent_tree(
        PARENT_ROOT,
        excluded_nonauthoritative_directories=records.PARENT_EXCLUDED_NONAUTHORITATIVE_DIRECTORIES,
    )
    config, _, worker_config = _config_bundle()
    scientific_config = cast(dict[str, records.JsonValue], config["scientific_config"])
    runtime_sources = cast(
        dict[str, records.JsonValue],
        scientific_config["runtime_source_manifest_sha256"],
    )
    protocol = _protocol_sha256()

    records.ensure_directory(destination)
    records.copy_opaque_immutable_exclusive(PROTOCOL_DOC, destination / PROTOCOL_FILE)
    records.copy_opaque_immutable_exclusive(audit_path, destination / AUDIT_FILE)
    records.write_immutable_json_exclusive(destination / CONFIG_FILE, config)
    records.write_immutable_json_exclusive(destination / WORKER_CONFIG_FILE, worker_config)
    records.write_immutable_json_exclusive(destination / SUPERSESSION_FILE, _supersession_record(protocol))
    for name in sorted(records.PARENT_PINS):
        records.copy_opaque_immutable_exclusive(PARENT_ROOT / name, destination / PARENT_COPY_ROOT / name)
    runtime.build_shadow_runtime(destination / RUNTIME_ROOT)
    runtime.build_numpy_dependency_closure(
        (destination / DEPENDENCY_ROOT).resolve(),
        expected_source_manifest_sha256=cast(str, runtime_sources["numpy_dependency"]),
    )
    runtime.build_stdlib_closure(
        (destination / STDLIB_ROOT).resolve(),
        expected_source_manifest_sha256=cast(str, runtime_sources["python_stdlib"]),
    )
    runtime.build_custody_runtime(
        (destination / CUSTODY_RUNTIME_ROOT).resolve(),
        expected_source_manifest_sha256=cast(str, runtime_sources["custody_runtime"]),
    )

    pre_probe = destination / PRE_PROBE_ROOT
    records.ensure_directory(pre_probe)
    records.write_immutable_bytes_exclusive(pre_probe / "allowed.txt", b"MM009-pre-marker-allowed\n")
    records.write_immutable_bytes_exclusive(pre_probe / "sentinel.txt", b"MM009-denied-sentinel\n")
    probe_output = pre_probe / "output"
    records.ensure_directory(probe_output)
    runtime.run_isolation_probe(
        (destination / RUNTIME_ROOT).resolve(),
        (pre_probe / "allowed.txt").resolve(),
        (destination / WORKER_CONFIG_FILE).resolve(),
        probe_output.resolve(),
        (
            (REPO_ROOT / "README.md").resolve(),
            (PARENT_ROOT / "MM-007-frames-64x64.npz").resolve(),
            (pre_probe / "sentinel.txt").resolve(),
        ),
        dependency_root=(destination / DEPENDENCY_ROOT).resolve(),
        stdlib_root=(destination / STDLIB_ROOT).resolve(),
    )
    probe_file = probe_output / "isolation-probe.json"
    records.copy_opaque_immutable_exclusive(probe_file, pre_probe / "isolation-probe.json")
    shutil.rmtree(probe_output)

    manifest = _pre_manifest(destination)
    records.write_immutable_json_exclusive(destination / INPUT_MANIFEST_FILE, manifest)
    return {
        "config_sha256": cast(str, manifest["config_sha256"]),
        "outcomes": "pre-real-prepared-only",
        "protocol_sha256": protocol,
        "status": "prepared",
    }


def _validate_prepared(output: Path) -> dict[str, records.JsonValue]:
    destination = _assert_output(output)
    saved = _strict_json(destination / INPUT_MANIFEST_FILE)
    expected = _pre_manifest(destination)
    if saved != expected:
        raise InvalidMM009Package("input manifest no longer recomputes")
    return expected


def _freeze_record(output: Path, manifest: Mapping[str, records.JsonValue]) -> dict[str, records.JsonValue]:
    closures = cast(dict[str, records.JsonValue], manifest["runtime_closures"])
    protocol_sha256 = cast(str, manifest["protocol_sha256"])
    closure_manifest_sha256: dict[str, records.JsonValue] = {
        name: records.canonical_json_sha256(
            cast(dict[str, records.JsonValue], closure),
            protocol_sha256=protocol_sha256,
        )
        for name, closure in closures.items()
    }
    return {
        "audit_file": records.file_record(output / AUDIT_FILE),
        "config_sha256": cast(str, manifest["config_sha256"]),
        "experiment_id": EXPERIMENT_ID,
        "input_manifest": records.file_record(output / INPUT_MANIFEST_FILE),
        "protocol_sha256": protocol_sha256,
        "runtime_closure_manifest_sha256": closure_manifest_sha256,
        "runtime_source_manifest_sha256": cast(
            dict[str, records.JsonValue], manifest["runtime_source_manifest_sha256"]
        ),
        "schema_version": "mm009-freeze-record-v1",
        "source_hashes": cast(dict[str, records.JsonValue], manifest["source_hashes"]),
        "status": "design_frozen_pre_real",
    }


def freeze(output: Path = DEFAULT_OUTPUT) -> dict[str, records.JsonValue]:
    destination = _assert_output(output)
    manifest = _validate_prepared(destination)
    record = _freeze_record(destination, manifest)
    records.write_immutable_json_exclusive(destination / FREEZE_FILE, record)
    return record


def _validate_frozen(output: Path) -> tuple[dict[str, records.JsonValue], dict[str, records.JsonValue]]:
    manifest = _validate_prepared(output)
    expected = _freeze_record(output, manifest)
    if _strict_json(output / FREEZE_FILE) != expected:
        raise InvalidMM009Package("freeze record differs from the prepared closure")
    return manifest, expected


def _formal_start(
    output: Path,
    manifest: Mapping[str, records.JsonValue],
    freeze_record: Mapping[str, records.JsonValue],
) -> dict[str, records.JsonValue]:
    return {
        "config_sha256": cast(str, manifest["config_sha256"]),
        "experiment_id": EXPERIMENT_ID,
        "freeze_record": records.file_record(output / FREEZE_FILE),
        "input_manifest": records.file_record(output / INPUT_MANIFEST_FILE),
        "protocol_sha256": cast(str, manifest["protocol_sha256"]),
        "schema_version": "mm009-formal-start-v1",
        "source_hashes": cast(dict[str, records.JsonValue], freeze_record["source_hashes"]),
        "status": "formal_execution_started_terminal_no_resume",
    }


def _validate_parent_normalizers(
    evidence: Mapping[str, object], normalizers: Sequence[preparation.FoldNormalizer]
) -> None:
    rows = evidence.get("normalizer_rows")
    if type(rows) is not list:
        raise InvalidMM009Package("MM-007 evidence lacks normalizer rows")
    for normalizer in normalizers:
        matches = [
            row
            for row in rows
            if type(row) is dict and row.get("resolution") == 8 and row.get("fold") == normalizer.fold_index
        ]
        if len(matches) != 1:
            raise InvalidMM009Package("MM-007 R8 normalizer membership differs")
        row = cast(dict[str, object], matches[0])
        if (
            row.get("train_video_ids") != list(normalizer.train_ids)
            or row.get("test_video_ids") != list(normalizer.test_ids)
            or row.get("train_rows") != normalizer.train_rows
            or row.get("uses_target") is not False
            or row.get("fingerprint") != normalizer.fingerprint
            or row.get("mean") != normalizer.mean.tolist()
            or row.get("scale") != normalizer.scale.tolist()
        ):
            raise InvalidMM009Package("MM-009 normalizer does not replay sealed MM-007 R8 evidence")


def _post_marker_inputs(
    output: Path,
) -> tuple[
    Mapping[str, np.ndarray],
    tuple[preparation.RowIndex, ...],
    tuple[preparation.FoldNormalizer, ...],
    np.ndarray,
    dict[str, records.JsonValue],
]:
    verification = mm007_experiment.verify(mm007_experiment.DEFAULT_OUTPUT)
    if verification.get("outcomes") != "verified_results":
        raise InvalidMM009Package("sealed MM-007 fast verification failed after marker")
    parent_copy = output / PARENT_COPY_ROOT
    arrays = records.load_npz_file(
        parent_copy / "MM-007-frames-64x64.npz",
        preparation.PARENT_FRAME_SCHEMA,
        label="copied MM-007 frame archive",
    )
    preparation.validate_parent_frame_arrays(arrays, require_pins=True)
    rows = preparation.build_row_index(arrays["video_ids"], arrays["timestamps"])
    normalizers = preparation.fit_fold_normalizers(arrays["frames_uint8"], rows)
    evidence_value = _strict_json(parent_copy / "MM-007-evidence.json")
    if type(evidence_value) is not dict:
        raise InvalidMM009Package("copied MM-007 evidence must be an object")
    evidence = mm007_method.validate_evidence(cast(dict[str, object], evidence_value))
    _validate_parent_normalizers(evidence, normalizers)
    derangement = preparation.half_cycle_derangement(rows)
    parent_record: dict[str, records.JsonValue] = {
        "array_sha256": {
            name: records.legacy_mm007_array_sha256(arrays[name]) for name in sorted(preparation.PARENT_FRAME_SCHEMA)
        },
        "fast_verification": _as_json(verification),
        "matched_identity_sha256": preparation.row_identity_sha256(rows),
        "normalizer_fingerprints": [item.fingerprint for item in normalizers],
        "schema_version": "mm009-parent-alignment-v1",
    }
    return arrays, rows, normalizers, derangement, parent_record


def _detach_rows(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    rows: tuple[preparation.RowIndex, ...],
    normalizers: tuple[preparation.FoldNormalizer, ...],
    derangement: np.ndarray,
    *,
    deadline: float,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    source_records: list[records.JsonValue] = []
    target_records: list[records.JsonValue] = []
    for row in rows:
        _check_deadline(deadline, f"row detachment {row.ordinal}")
        source_directory = output / SOURCE_ROOT / f"{row.ordinal:06d}"
        target_directory = output / TARGET_ROOT / f"{row.ordinal:06d}"
        records.ensure_directory(source_directory)
        records.ensure_directory(target_directory)
        source_arrays = preparation.construct_source_row(
            arrays["frames_uint8"], row.ordinal, rows, normalizers, derangement
        )
        target_arrays = preparation.construct_target_row(
            arrays["frames_uint8"], row.ordinal, rows, normalizers, derangement
        )
        preparation.validate_detached_pair(source_arrays, target_arrays)
        source_path = source_directory / "source.npz"
        target_path = target_directory / "target.npz"
        preparation.write_source_row_npz(source_path, source_arrays)
        preparation.write_target_row_npz(target_path, target_arrays)
        source_records.append(
            {
                "file": records.file_record(source_path),
                "manifest": preparation.source_row_manifest(source_arrays, protocol_sha256=protocol_sha256),
                "ordinal": row.ordinal,
            }
        )
        target_records.append(
            {
                "file": records.file_record(target_path),
                "manifest": preparation.target_row_manifest(target_arrays, protocol_sha256=protocol_sha256),
                "ordinal": row.ordinal,
            }
        )
    value: dict[str, records.JsonValue] = {
        "derangement": derangement.tolist(),
        "derangement_sha256": records.scientific_array_sha256(
            "half-cycle-derangement", derangement, protocol_sha256=protocol_sha256
        ),
        "row_count": len(rows),
        "row_identity_sha256": preparation.row_identity_sha256(rows),
        "schema_version": "mm009-detachment-index-v1",
        "sources": source_records,
        "targets": target_records,
    }
    _check_deadline(deadline, "row detachment completion")
    return value


def _post_detachment_probe(
    output: Path,
    rows: Sequence[preparation.RowIndex],
    *,
    deadline: float,
) -> dict[str, records.JsonValue]:
    probe_root = output / POST_PROBE_ROOT
    records.ensure_directory(probe_root)
    denied = (
        (output / SOURCE_ROOT / "000001/source.npz").resolve(),
        *((output / TARGET_ROOT / f"{row.ordinal:06d}").resolve() for row in rows),
    )
    value = runtime.run_isolation_probe(
        (output / RUNTIME_ROOT).resolve(),
        (output / SOURCE_ROOT / "000000/source.npz").resolve(),
        (output / WORKER_CONFIG_FILE).resolve(),
        probe_root.resolve(),
        denied,
        dependency_root=(output / DEPENDENCY_ROOT).resolve(),
        stdlib_root=(output / STDLIB_ROOT).resolve(),
        timeout_seconds=max(
            1,
            math.ceil(_deadline_timeout(deadline, 60.0, "post-detachment isolation probe")),
        ),
    )
    _check_deadline(deadline, "post-detachment isolation probe completion")
    denied_manifest: records.JsonValue = [
        {
            "path_role": "sibling_source" if index == 0 else "target_row_directory",
            "relative": str(path.relative_to(output)),
        }
        for index, path in enumerate(denied)
    ]
    return {
        "denied_manifest": denied_manifest,
        "denied_manifest_sha256": records.canonical_json_sha256(denied_manifest, protocol_sha256=_protocol_sha256()),
        "probe": _as_json(value),
        "probe_file": records.file_record(probe_root / "isolation-probe.json"),
        "schema_version": "mm009-post-detachment-isolation-v1",
    }


def _load_worker_config(path: Path) -> dict[str, object]:
    value = _strict_json(path)
    if type(value) is not dict:
        raise InvalidMM009Package("worker configuration must be an object")
    return cast(dict[str, object], value)


def _run_one_child(
    output: Path,
    ordinal: int,
    work_root: Path,
    deadline: float,
    process_registry: runtime.WorkerProcessRegistry,
) -> Path:
    remaining = _deadline_timeout(deadline, WORKER_TIMEOUT_SECONDS, f"prediction child {ordinal} launch")
    child = work_root / f"{ordinal:06d}"
    records.ensure_directory(child, mode=0o700)
    runtime.run_worker_process(
        (output / RUNTIME_ROOT).resolve(),
        (output / SOURCE_ROOT / f"{ordinal:06d}/source.npz").resolve(),
        (output / WORKER_CONFIG_FILE).resolve(),
        child.resolve(),
        dependency_root=(output / DEPENDENCY_ROOT).resolve(),
        stdlib_root=(output / STDLIB_ROOT).resolve(),
        timeout_seconds=max(1, math.ceil(remaining)),
        process_registry=process_registry,
    )
    _check_deadline(deadline, f"prediction child {ordinal} completion")
    source_path = output / SOURCE_ROOT / f"{ordinal:06d}/source.npz"
    source = preparation.load_source_row_npz(source_path)
    worker.validate_worker_output(
        child,
        _load_worker_config(output / WORKER_CONFIG_FILE),
        source,
        source_path=source_path,
    )
    return child


def _supervisor_commit(
    output: Path,
    row: preparation.RowIndex,
    destination: Path,
    predecessor_sha256: str,
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    source_path = output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz"
    return {
        "config_sha256": config_sha256,
        "fold_index": row.fold_index,
        "predecessor_sha256": records.require_sha256(predecessor_sha256, "predecessor SHA-256"),
        "protocol_sha256": protocol_sha256,
        "row_ordinal": row.ordinal,
        "schema_version": "mm009-supervisor-prediction-commit-v1",
        "source_file": records.file_record(source_path),
        "video_id": row.video_id,
        "video_row": row.video_row,
        "worker_evidence_file": records.file_record(destination / "worker-evidence.json"),
        "worker_prediction_file": records.file_record(destination / "predictions.npy"),
    }


def _prediction_attempt_record(
    output: Path,
    rows: Sequence[preparation.RowIndex],
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    return {
        "config_sha256": config_sha256,
        "formal_controls": records.file_record(output / CONTROL_FILE),
        "formal_start": records.file_record(output / START_FILE),
        "post_detachment_isolation": records.file_record(output / POST_ISOLATION_FILE),
        "protocol_sha256": protocol_sha256,
        "row_count": len(rows),
        "schema_version": "mm009-prediction-attempt-v1",
        "status": "started_terminal_no_resume_no_retry",
    }


def _predict_all(
    output: Path,
    rows: tuple[preparation.RowIndex, ...],
    *,
    config_sha256: str,
    deadline: float,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    _check_deadline(deadline, "prediction attempt")
    attempt = _prediction_attempt_record(
        output,
        rows,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    records.write_immutable_json_exclusive(output / PREDICTION_ATTEMPT_FILE, attempt)
    work_root = output / WORK_ROOT / "children"
    records.ensure_directory(work_root, mode=0o700)
    completed: dict[int, Path] = {}
    process_registry = runtime.WorkerProcessRegistry()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures: dict[concurrent.futures.Future[Path], int] = {}
    try:
        for row in rows:
            future = executor.submit(
                _run_one_child,
                output,
                row.ordinal,
                work_root,
                deadline,
                process_registry,
            )
            futures[future] = row.ordinal
        for future in concurrent.futures.as_completed(
            futures,
            timeout=_deadline_timeout(deadline, TOTAL_WALL_SECONDS, "prediction child collection"),
        ):
            ordinal = futures[future]
            completed[ordinal] = future.result()
            _check_deadline(deadline, f"prediction child {ordinal} collected")
    except TimeoutError as error:
        process_registry.cancel_all()
        for future in futures:
            future.cancel()
        raise InvalidMM009Package("formal total-wall deadline exceeded: prediction children") from error
    except BaseException:
        process_registry.cancel_all()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if process_registry.active_pids or process_registry.cancelled:
        process_registry.cancel_all()
        raise InvalidMM009Package("prediction process registry did not close cleanly")
    if set(completed) != set(range(len(rows))):
        raise InvalidMM009Package("prediction child census is incomplete")

    predecessor = records.file_sha256(output / PREDICTION_ATTEMPT_FILE)
    chain: list[records.JsonValue] = []
    worker_config = _load_worker_config(output / WORKER_CONFIG_FILE)
    for row in rows:
        _check_deadline(deadline, f"prediction custody row {row.ordinal}")
        child = completed[row.ordinal]
        source_path = output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz"
        source = preparation.load_source_row_npz(source_path)
        worker.validate_worker_output(
            child,
            worker_config,
            source,
            source_path=source_path,
        )
        destination = output / PREDICTION_ROOT / f"{row.ordinal:06d}"
        records.ensure_directory(destination)
        records.copy_opaque_immutable_exclusive(child / worker.PREDICTION_FILE, destination / "predictions.npy")
        records.copy_opaque_immutable_exclusive(child / worker.COMMIT_FILE, destination / "worker-evidence.json")
        _validate_canonical_worker_custody(
            output,
            row,
            destination,
            worker_config,
            supervisor_commit_required=False,
        )
        commit = _supervisor_commit(
            output,
            row,
            destination,
            predecessor,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        records.write_immutable_json_exclusive(destination / "commit.json", commit)
        predecessor = records.file_sha256(destination / "commit.json")
        chain.append(
            {
                "commit_sha256": predecessor,
                "ordinal": row.ordinal,
                "prediction_sha256": records.file_sha256(destination / "predictions.npy"),
                "worker_evidence_sha256": records.file_sha256(destination / "worker-evidence.json"),
            }
        )
    shutil.rmtree(output / WORK_ROOT)
    value: dict[str, records.JsonValue] = {
        "chain": chain,
        "config_sha256": config_sha256,
        "genesis_sha256": records.file_sha256(output / PREDICTION_ATTEMPT_FILE),
        "last_commit_sha256": predecessor,
        "protocol_sha256": protocol_sha256,
        "row_count": len(rows),
        "schema_version": "mm009-prediction-freeze-v1",
        "status": "all_predictions_frozen_before_target_scoring",
    }
    records.write_immutable_json_exclusive(output / PREDICTION_FREEZE_FILE, value)
    _check_deadline(deadline, "prediction freeze")
    return value


def _validate_canonical_worker_custody(
    output: Path,
    row: preparation.RowIndex,
    directory: Path,
    config: Mapping[str, object],
    *,
    supervisor_commit_required: bool,
) -> None:
    """Replay the complete child validator over one supervisor-custodied bundle."""

    if directory.is_symlink() or not directory.is_dir():
        raise InvalidMM009Package(f"canonical prediction row is not a real directory: {row.ordinal}")
    expected_members = {"predictions.npy", "worker-evidence.json"}
    if supervisor_commit_required:
        expected_members.add("commit.json")
    if {path.name for path in directory.iterdir()} != expected_members:
        raise InvalidMM009Package(f"canonical prediction row membership differs: {row.ordinal}")
    prediction_path = directory / "predictions.npy"
    evidence_path = directory / "worker-evidence.json"
    custody_paths = [prediction_path, evidence_path]
    if supervisor_commit_required:
        custody_paths.append(directory / "commit.json")
    if any(records.file_record(path)["mode"] != 0o444 for path in custody_paths):
        raise InvalidMM009Package(f"canonical prediction custody is not immutable: {row.ordinal}")

    source_path = output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz"
    if records.file_record(source_path)["mode"] != 0o444:
        raise InvalidMM009Package(f"detached source custody is not immutable: {row.ordinal}")
    source = preparation.load_source_row_npz(source_path)
    with tempfile.TemporaryDirectory(prefix=f"mm009-worker-replay-{row.ordinal:06d}-") as temporary_name:
        temporary = Path(temporary_name)
        records.copy_opaque_immutable_exclusive(prediction_path, temporary / worker.PREDICTION_FILE)
        records.copy_opaque_immutable_exclusive(evidence_path, temporary / worker.COMMIT_FILE)
        try:
            worker.validate_worker_output(temporary, config, source, source_path=source_path)
        except (records.RecordValidationError, worker.WorkerError) as error:
            raise InvalidMM009Package(f"canonical worker evidence differs at row {row.ordinal}") from error


def _validate_prediction_chain(
    output: Path,
    rows: tuple[preparation.RowIndex, ...],
) -> dict[str, records.JsonValue]:
    value = _strict_json(output / PREDICTION_FREEZE_FILE)
    if type(value) is not dict:
        raise InvalidMM009Package("prediction freeze must be an object")
    freeze = cast(dict[str, records.JsonValue], value)
    config = _load_worker_config(output / WORKER_CONFIG_FILE)
    config_sha256 = cast(str, config["config_sha256"])
    protocol = cast(str, config["protocol_sha256"])
    expected_attempt = _prediction_attempt_record(
        output,
        rows,
        config_sha256=config_sha256,
        protocol_sha256=protocol,
    )
    if _strict_json(output / PREDICTION_ATTEMPT_FILE) != expected_attempt:
        raise InvalidMM009Package("prediction attempt differs from its ordered prerequisites")
    if (
        set(freeze)
        != {
            "chain",
            "config_sha256",
            "genesis_sha256",
            "last_commit_sha256",
            "protocol_sha256",
            "row_count",
            "schema_version",
            "status",
        }
        or freeze.get("schema_version") != "mm009-prediction-freeze-v1"
        or freeze.get("row_count") != len(rows)
        or freeze.get("genesis_sha256") != records.file_sha256(output / PREDICTION_ATTEMPT_FILE)
        or freeze.get("config_sha256") != config_sha256
        or freeze.get("protocol_sha256") != protocol
        or freeze.get("status") != "all_predictions_frozen_before_target_scoring"
    ):
        raise InvalidMM009Package("prediction freeze header differs")
    predecessor = cast(str, freeze["genesis_sha256"])
    observed: list[records.JsonValue] = []
    for row in rows:
        directory = output / PREDICTION_ROOT / f"{row.ordinal:06d}"
        prediction_path = directory / "predictions.npy"
        evidence_path = directory / "worker-evidence.json"
        commit_path = directory / "commit.json"
        predictions = worker.load_prediction_array(prediction_path)
        evidence_value = _strict_json(evidence_path)
        if type(evidence_value) is not dict:
            raise InvalidMM009Package("worker evidence must be an object")
        evidence = cast(dict[str, object], evidence_value)
        hashes = evidence.get("prediction_hashes")
        if type(hashes) is not dict:
            raise InvalidMM009Package("worker evidence lacks prediction hashes")
        for index, role in enumerate(worker.PREDICTION_ROLES):
            item = hashes.get(role)
            if type(item) is not dict or item.get("mm009_sha256") != records.scientific_array_sha256(
                f"prediction:{role}", predictions[index], protocol_sha256=protocol
            ):
                raise InvalidMM009Package(f"prediction role hash differs: {row.ordinal}/{role}")
        _validate_canonical_worker_custody(
            output,
            row,
            directory,
            config,
            supervisor_commit_required=True,
        )
        expected = _supervisor_commit(
            output,
            row,
            directory,
            predecessor,
            config_sha256=config_sha256,
            protocol_sha256=protocol,
        )
        if _strict_json(commit_path) != expected:
            raise InvalidMM009Package(f"supervisor commit differs at row {row.ordinal}")
        predecessor = records.file_sha256(commit_path)
        observed.append(
            {
                "commit_sha256": predecessor,
                "ordinal": row.ordinal,
                "prediction_sha256": records.file_sha256(prediction_path),
                "worker_evidence_sha256": records.file_sha256(evidence_path),
            }
        )
    if freeze.get("chain") != observed or freeze.get("last_commit_sha256") != predecessor:
        raise InvalidMM009Package("prediction freeze chain differs")
    return freeze


def _run_future_isolation(
    output: Path,
    *,
    evidence_path: Path | None = None,
    deadline: float | None = None,
) -> dict[str, records.JsonValue]:
    destination = output / FUTURE_ISOLATION_FILE if evidence_path is None else evidence_path
    private_output = Path(tempfile.mkdtemp(prefix="mm009-future-child-", dir="/tmp")).resolve()
    child_evidence = private_output / FUTURE_ISOLATION_FILE
    timeout = 300.0 if deadline is None else _deadline_timeout(deadline, 300.0, "future-isolation launch")
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str((output / CUSTODY_RUNTIME_ROOT / runtime.FUTURE_ISOLATION_LAUNCHER_RELATIVE).resolve()),
        str((output / CUSTODY_RUNTIME_ROOT).resolve()),
        str((output / DEPENDENCY_ROOT).resolve()),
        str((output / STDLIB_ROOT).resolve()),
        str((output / PREDICTION_FREEZE_FILE).resolve()),
        str((output / TARGET_ROOT).resolve()),
        str((output / PREDICTION_ROOT).resolve()),
        str(child_evidence),
    )
    try:
        try:
            result = _run_cancellable_process(
                command,
                cwd=private_output,
                timeout_seconds=timeout,
            )
        except subprocess.TimeoutExpired as error:
            if deadline is not None and time.monotonic() >= deadline:
                raise InvalidMM009Package("formal total-wall deadline exceeded: future isolation") from error
            raise InvalidMM009Package("fitting-free future-isolation gate timed out") from error
        if result.returncode != 0:
            raise InvalidMM009Package(f"fitting-free future-isolation gate failed: {result.stderr[-4000:]}")
        if result.stdout or result.stderr:
            raise InvalidMM009Package("future-isolation gate emitted an unexpected stream")
        if deadline is not None:
            _check_deadline(deadline, "future-isolation child completion")
        _validate_future_isolation(output, evidence_path=child_evidence)
        records.write_immutable_bytes_exclusive(
            destination,
            records.read_regular_bytes(
                child_evidence,
                maximum_bytes=MAX_FUTURE_FILE_BYTES,
            ),
        )
        if deadline is not None:
            _check_deadline(deadline, "future-isolation supervisor commit")
        return _validate_future_isolation(output, evidence_path=destination)
    finally:
        shutil.rmtree(private_output, ignore_errors=True)


def _validate_future_isolation(
    output: Path,
    *,
    evidence_path: Path | None = None,
) -> dict[str, records.JsonValue]:
    # Imported lazily because this supervisor has already loaded the fitting stack;
    # the formal future gate itself imports the same module in a fresh custody-only
    # process where its import guard remains authoritative.
    from . import future_isolation

    destination = output / FUTURE_ISOLATION_FILE if evidence_path is None else evidence_path
    value = _strict_json(destination)
    prediction_freeze_path = output / PREDICTION_FREEZE_FILE
    protocol_sha256 = _protocol_sha256()
    try:
        record = future_isolation.validate_evidence(
            value,
            prediction_freeze_record=records.file_record(prediction_freeze_path),
            protocol_sha256=protocol_sha256,
        )
        destination_record = records.file_record(destination)
        if (
            destination_record["mode"] != 0o444
            or cast(int, destination_record["bytes"]) > MAX_FUTURE_FILE_BYTES
            or records.read_regular_bytes(destination, maximum_bytes=MAX_FUTURE_FILE_BYTES)
            != records.json_file_bytes(record)
        ):
            raise InvalidMM009Package("future-isolation evidence is not immutable canonical JSON")

        freeze, freeze_protocol = future_isolation._validate_freeze(prediction_freeze_path)  # noqa: SLF001
        if freeze_protocol != protocol_sha256:
            raise InvalidMM009Package("future-isolation freeze protocol binding differs")
        source_snapshot = future_isolation._source_side_snapshot(  # noqa: SLF001
            prediction_freeze_path,
            output / PREDICTION_ROOT,
            freeze,
        )
        target_snapshot = future_isolation._target_snapshot(output / TARGET_ROOT)  # noqa: SLF001
        recomputed_counts, finite_manifest_sha256 = future_isolation._mutation_sweep(  # noqa: SLF001
            output / TARGET_ROOT,
            protocol_sha256=protocol_sha256,
        )
        source_manifest_sha256 = records.canonical_json_sha256(
            source_snapshot,
            protocol_sha256=protocol_sha256,
        )
        target_manifest_sha256 = records.canonical_json_sha256(
            target_snapshot,
            protocol_sha256=protocol_sha256,
        )
    except future_isolation.FutureIsolationError as error:
        raise InvalidMM009Package(f"future-isolation evidence differs: {error}") from error
    except records.RecordValidationError as error:
        raise InvalidMM009Package("future-isolation evidence record differs") from error

    source_manifest = cast(dict[str, records.JsonValue], record["source_side_manifest"])
    target_manifest = cast(dict[str, records.JsonValue], record["target_input_manifest"])
    if (
        record["mutation_counts"] != recomputed_counts
        or record["finite_mutation_manifest_sha256"] != finite_manifest_sha256
        or record["source_side_aggregate_manifest_sha256"] != source_manifest_sha256
        or source_manifest["sha256"] != source_manifest_sha256
        or source_manifest["file_count"] != source_snapshot["file_count"]
        or target_manifest["sha256"] != target_manifest_sha256
        or target_manifest["file_count"] != target_snapshot["file_count"]
    ):
        raise InvalidMM009Package("future-isolation recomputed evidence digest differs")
    return cast(dict[str, records.JsonValue], record)


def _run_one_score(
    output: Path,
    row: preparation.RowIndex,
    deadline: float,
    cancel_event: threading.Event,
) -> Path:
    timeout = _deadline_timeout(deadline, 120.0, f"score child {row.ordinal} launch")
    score_directory = output / SCORE_ROOT / f"{row.ordinal:06d}"
    records.ensure_directory(score_directory, mode=0o700)
    prediction_directory = output / PREDICTION_ROOT / f"{row.ordinal:06d}"
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str((output / CUSTODY_RUNTIME_ROOT / runtime.SCORE_LAUNCHER_RELATIVE).resolve()),
        str((output / CUSTODY_RUNTIME_ROOT).resolve()),
        str((output / DEPENDENCY_ROOT).resolve()),
        str((output / STDLIB_ROOT).resolve()),
        str((output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz").resolve()),
        str((output / TARGET_ROOT / f"{row.ordinal:06d}/target.npz").resolve()),
        str((prediction_directory / "predictions.npy").resolve()),
        str((prediction_directory / "commit.json").resolve()),
        str((output / WORKER_CONFIG_FILE).resolve()),
        str(score_directory.resolve()),
    )
    try:
        result = _run_cancellable_process(
            command,
            cwd=score_directory,
            timeout_seconds=timeout,
            cancel_event=cancel_event,
        )
    except subprocess.TimeoutExpired as error:
        if time.monotonic() >= deadline:
            raise InvalidMM009Package(f"formal total-wall deadline exceeded: score row {row.ordinal}") from error
        raise InvalidMM009Package(f"score-only child timed out for row {row.ordinal}") from error
    if result.returncode != 0:
        raise InvalidMM009Package(f"score-only child failed for row {row.ordinal}: {result.stderr[-4000:]}")
    if result.stdout or result.stderr:
        raise InvalidMM009Package("score-only child emitted an unexpected stream")
    _check_deadline(deadline, f"score child {row.ordinal} completion")
    path = score_directory / "score.json"
    if (
        set(score_directory.iterdir()) != {path}
        or stat.S_IMODE(score_directory.lstat().st_mode) != 0o700
        or records.file_record(path)["mode"] != 0o444
        or cast(int, records.file_record(path)["bytes"]) > MAX_SCORE_FILE_BYTES
    ):
        raise InvalidMM009Package("score-only child output census or mode differs")
    return path


def _error_from(value: object, label: str) -> scoring.ErrorPrimitive:
    if type(value) is not dict or set(value) != {"count", "sse"}:
        raise InvalidMM009Package(f"score primitive schema differs: {label}")
    item = cast(dict[str, object], value)
    return scoring.ErrorPrimitive(sse=cast(float, item["sse"]), count=cast(int, item["count"]))


def _row_score_from_record(record: Mapping[str, object], family: scoring.Family) -> scoring.RowScores:
    families = record.get("families")
    if type(families) is not dict or type(families.get(family)) is not dict:
        raise InvalidMM009Package("score record family membership differs")
    metrics = cast(dict[str, object], families[family])
    expected_metrics = ("i", "a", "q", "p", "c", "h", "r", "z", "d", "pd", "u", "b", "bd")
    if set(metrics) != set(expected_metrics):
        raise InvalidMM009Package("score record metric membership differs")
    kwargs = {name: _error_from(metrics[name], f"{family}/{name}") for name in expected_metrics}
    return scoring.RowScores(
        video_id=cast(str, record["video_id"]),
        fold=cast(int, record["fold_index"]),
        row_index=cast(int, record["video_row"]),
        family=family,
        **kwargs,
    )


def _score_records(output: Path, rows: tuple[preparation.RowIndex, ...]) -> list[dict[str, object]]:
    output_records: list[dict[str, object]] = []
    config = _load_worker_config(output / WORKER_CONFIG_FILE)
    for row in rows:
        path = output / SCORE_ROOT / f"{row.ordinal:06d}/score.json"
        _validate_bounded_canonical_file(
            path,
            maximum_bytes=MAX_SCORE_FILE_BYTES,
            label=f"row score {row.ordinal}",
        )
        value = _strict_json(path)
        if type(value) is not dict:
            raise InvalidMM009Package("row score must be an object")
        record = cast(dict[str, object], value)
        expected_keys = {
            "config_sha256",
            "families",
            "fold_index",
            "prediction_file_sha256",
            "protocol_sha256",
            "row_ordinal",
            "schema_version",
            "source_file_sha256",
            "supervisor_commit_sha256",
            "target_file_sha256",
            "video_id",
            "video_row",
        }
        prediction_dir = output / PREDICTION_ROOT / f"{row.ordinal:06d}"
        if (
            set(record) != expected_keys
            or record.get("schema_version") != "mm009-row-score-v1"
            or record.get("config_sha256") != config.get("config_sha256")
            or record.get("protocol_sha256") != config.get("protocol_sha256")
            or (record.get("row_ordinal"), record.get("video_id"), record.get("fold_index"), record.get("video_row"))
            != (row.ordinal, row.video_id, row.fold_index, row.video_row)
            or record.get("prediction_file_sha256") != records.file_sha256(prediction_dir / "predictions.npy")
            or record.get("supervisor_commit_sha256") != records.file_sha256(prediction_dir / "commit.json")
            or record.get("source_file_sha256")
            != records.file_sha256(output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz")
            or record.get("target_file_sha256")
            != records.file_sha256(output / TARGET_ROOT / f"{row.ordinal:06d}/target.npz")
        ):
            raise InvalidMM009Package(f"row score custody differs at ordinal {row.ordinal}")
        for family in scoring.FAMILIES:
            _row_score_from_record(record, family)
        output_records.append(record)
    return output_records


def _decision_from_scores(
    output: Path,
    rows: tuple[preparation.RowIndex, ...],
    score_records: Sequence[Mapping[str, object]],
) -> tuple[decision.DecisionEvidence, decision.DecisionSummary]:
    grouped: dict[tuple[scoring.Family, str], list[scoring.RowScores]] = defaultdict(list)
    bounded: dict[tuple[scoring.Family, str], int] = defaultdict(int)
    for row, record in zip(rows, score_records, strict=True):
        worker_evidence = _strict_json(output / PREDICTION_ROOT / f"{row.ordinal:06d}/worker-evidence.json")
        if type(worker_evidence) is not dict or type(worker_evidence.get("bounded")) is not dict:
            raise InvalidMM009Package("worker bounded evidence differs")
        bound_record = cast(dict[str, object], worker_evidence["bounded"])
        if set(bound_record) != set(scoring.FAMILIES) or any(
            type(value) is not bool for value in bound_record.values()
        ):
            raise InvalidMM009Package("worker bounded family schema differs")
        for family in scoring.FAMILIES:
            grouped[(family, row.video_id)].append(_row_score_from_record(record, family))
            bounded[(family, row.video_id)] += int(cast(bool, bound_record[family]))

    family_evidence: list[decision.FamilyEvidence] = []
    for family in scoring.FAMILIES:
        videos: list[decision.FamilyVideoEvidence] = []
        for video_id, _, expected_rows in decision.VIDEO_SPECS:
            row_scores = tuple(grouped[(family, video_id)])
            if len(row_scores) != expected_rows:
                raise InvalidMM009Package("score aggregation video census differs")
            aggregate = scoring.aggregate_video(row_scores)
            videos.append(
                decision.FamilyVideoEvidence(
                    scores=aggregate,
                    bounded_rows=bounded[(family, video_id)],
                    completeness=decision.CompletenessPrimitives(
                        rows_complete=True,
                        certificates_valid=True,
                        apply_replays_valid=True,
                        hashes_valid=True,
                    ),
                )
            )
        family_evidence.append(decision.FamilyEvidence(family=family, videos=tuple(videos)))
    evidence = decision.DecisionEvidence(
        prerequisites=decision.PrerequisitePrimitives(
            parent_alignment_valid=True,
            synthetic_controls_valid=True,
            future_isolation_valid=True,
            source_binding_valid=True,
            package_integrity_valid=True,
        ),
        families=tuple(family_evidence),
    )
    return evidence, decision.derive_decision(evidence)


def _score_attempt_record(
    output: Path,
    rows: Sequence[preparation.RowIndex],
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    return {
        "config_sha256": config_sha256,
        "future_isolation": records.file_record(output / FUTURE_ISOLATION_FILE),
        "pre_score_budget": records.file_record(output / PRE_SCORE_BUDGET_FILE),
        "prediction_freeze": records.file_record(output / PREDICTION_FREEZE_FILE),
        "protocol_sha256": protocol_sha256,
        "row_count": len(rows),
        "schema_version": "mm009-score-attempt-v1",
        "status": "target_scoring_started_after_complete_prediction_freeze",
    }


def _run_scores(
    output: Path,
    rows: tuple[preparation.RowIndex, ...],
    *,
    config_sha256: str,
    deadline: float,
    protocol_sha256: str,
) -> tuple[list[dict[str, object]], decision.DecisionEvidence, decision.DecisionSummary]:
    _validate_pre_score_budget(output, protocol_sha256=protocol_sha256)
    _validate_prediction_chain(output, rows)
    attempt = _score_attempt_record(
        output,
        rows,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    _write_bounded_immutable_json(
        output / SCORE_ATTEMPT_FILE,
        attempt,
        maximum_bytes=MAX_SCORE_ATTEMPT_BYTES,
        label="score attempt",
    )
    cancel_event = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures: dict[concurrent.futures.Future[Path], int] = {}
    try:
        for row in rows:
            future = executor.submit(_run_one_score, output, row, deadline, cancel_event)
            futures[future] = row.ordinal
        for future in concurrent.futures.as_completed(
            futures,
            timeout=_deadline_timeout(deadline, TOTAL_WALL_SECONDS, "score child collection"),
        ):
            future.result()
            _check_deadline(deadline, f"score child {futures[future]} collected")
    except TimeoutError as error:
        cancel_event.set()
        for future in futures:
            future.cancel()
        raise InvalidMM009Package("formal total-wall deadline exceeded: score children") from error
    except BaseException:
        cancel_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    score_records = _score_records(output, rows)
    _check_deadline(deadline, "score aggregation")
    evidence, summary = _decision_from_scores(output, rows, score_records)
    return score_records, evidence, summary


def _evidence_record(
    output: Path,
    parent_alignment: Mapping[str, records.JsonValue],
    score_records: Sequence[Mapping[str, object]],
    decision_evidence: decision.DecisionEvidence,
    summary: decision.DecisionSummary,
) -> dict[str, records.JsonValue]:
    return {
        "decision_evidence": _as_json(decision_evidence),
        "decision_summary": _as_json(summary),
        "detachment_index": records.file_record(output / DETACHMENT_FILE),
        "formal_synthetic_controls": records.file_record(output / CONTROL_FILE),
        "future_isolation": records.file_record(output / FUTURE_ISOLATION_FILE),
        "parent_alignment": {name: item for name, item in parent_alignment.items()},
        "pre_score_budget": records.file_record(output / PRE_SCORE_BUDGET_FILE),
        "post_detachment_isolation": records.file_record(output / POST_ISOLATION_FILE),
        "prediction_freeze": records.file_record(output / PREDICTION_FREEZE_FILE),
        "row_scores": [
            {
                "file": records.file_record(output / SCORE_ROOT / f"{row.ordinal:06d}/score.json"),
                "ordinal": row.ordinal,
            }
            for row in preparation.canonical_row_index()
        ],
        "schema_version": "mm009-evidence-v1",
        "score_record_count": len(score_records),
    }


def _result_record(
    output: Path,
    summary: decision.DecisionSummary,
) -> dict[str, records.JsonValue]:
    branch = _branch_for_decision(go=summary.go, decision_label=summary.decision)
    return {
        "branch": branch,
        "decision": summary.decision,
        "epistemic_role": "outcome-visible eight-video exploratory causal mechanism diagnostic",
        "evidence_file": records.file_record(output / EVIDENCE_FILE),
        "experiment_id": EXPERIMENT_ID,
        "formal_start": records.file_record(output / START_FILE),
        "go": summary.go,
        "passing_families": list(summary.passing_families),
        "schema_version": "mm009-result-v1",
        "status": "completed_pending_independent_result_audit",
    }


def _branch_for_decision(*, go: bool, decision_label: str) -> str:
    if go:
        return "preregister_new_TAESD_MM001_successor_then_audit_before_execution"
    if decision_label == "tested_causal_operator_failure_supported":
        return "preregister_MM010_source_only_analog_coverage_then_audit_before_execution"
    return "neither_branch_diagnose_and_preregister_new_assay"


def _report_text(result: Mapping[str, records.JsonValue], summary: decision.DecisionSummary) -> str:
    lines = [
        "# MM-009 causal deformation/appearance prediction result",
        "",
        f"Decision: `{result['decision']}`.",
        f"GO: `{str(result['go']).lower()}`.",
        f"Frozen branch: `{result['branch']}`.",
        "",
        "This is an exploratory result on eight outcome-visible videos. It is not an",
        "independent confirmation, population claim, trained latent-dynamics result, or",
        "end-to-end Prospect capability claim.",
        "",
        "Family aggregates:",
        "",
    ]
    for family in summary.families:
        lines.append(
            f"- `{family.family}`: history {family.historical_support_count}/8, future "
            f"{family.future_support_count}/8, joint {family.joint_support_count}/8, "
            f"directional {family.directional_improvement_count}/8, pass={str(family.passes).lower()}, "
            f"diagnosis={family.failure_diagnosis or 'none'}."
        )
    lines.extend(
        (
            "",
            "No successor is executed by this result package. The frozen branch requires a",
            "new task/protocol and an independent result audit first.",
            "",
        )
    )
    return "\n".join(lines)


def _pre_score_snapshot(output: Path) -> dict[str, records.JsonValue]:
    later_files = {
        PRE_SCORE_BUDGET_FILE,
        FUTURE_ISOLATION_FILE,
        SCORE_ATTEMPT_FILE,
        EVIDENCE_FILE,
        RESULT_FILE,
        REPORT_FILE,
        ARTIFACT_MANIFEST_FILE,
    }
    artifacts: dict[str, records.JsonValue] = {}
    for path in sorted(output.rglob("*"), key=lambda item: str(item.relative_to(output))):
        if path.is_symlink():
            raise InvalidMM009Package("pre-score package contains a symlink")
        relative = path.relative_to(output)
        if path.is_dir():
            continue
        if not path.is_file():
            raise InvalidMM009Package("pre-score package contains a special entry")
        if relative in later_files or relative.parts[0] == SCORE_ROOT.parts[0]:
            continue
        artifacts[str(relative)] = records.file_record(path)
    return artifacts


def _require_pre_score_output_exclusivity(output: Path) -> None:
    forbidden = (
        PRE_SCORE_BUDGET_FILE,
        FUTURE_ISOLATION_FILE,
        SCORE_ATTEMPT_FILE,
        EVIDENCE_FILE,
        RESULT_FILE,
        REPORT_FILE,
        ARTIFACT_MANIFEST_FILE,
        SCORE_ROOT,
    )
    if any((output / relative).exists() or (output / relative).is_symlink() for relative in forbidden):
        raise InvalidMM009Package("pre-score outputs must be wholly absent before budget projection")


def _pre_score_budget_record(
    output: Path,
    *,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    artifacts = _pre_score_snapshot(output)
    current_bytes = sum(cast(int, cast(dict[str, records.JsonValue], item)["bytes"]) for item in artifacts.values())
    remaining: dict[str, records.JsonValue] = {
        "artifact_manifest": MAX_ARTIFACT_MANIFEST_FILE_BYTES,
        "budget_record": MAX_BUDGET_FILE_BYTES,
        "evidence": MAX_EVIDENCE_FILE_BYTES,
        "future_isolation": MAX_FUTURE_FILE_BYTES,
        "report": MAX_REPORT_FILE_BYTES,
        "result": MAX_RESULT_FILE_BYTES,
        "row_scores": preparation.MATCHED_ROWS * MAX_SCORE_FILE_BYTES,
        "score_attempt": MAX_SCORE_ATTEMPT_BYTES,
    }
    projected = current_bytes + sum(cast(int, value) for value in remaining.values())
    return {
        "artifact_ceiling_bytes": MAX_ARTIFACT_BYTES,
        "current_artifact_bytes": current_bytes,
        "current_artifact_count": len(artifacts),
        "current_manifest_sha256": records.canonical_json_sha256(
            artifacts,
            protocol_sha256=protocol_sha256,
        ),
        "projected_max_artifact_bytes": projected,
        "protocol_sha256": protocol_sha256,
        "remaining_upper_bounds": remaining,
        "schema_version": "mm009-pre-score-budget-v1",
        "status": ("within_frozen_ceiling" if projected <= MAX_ARTIFACT_BYTES else "projected_ceiling_exceeded"),
    }


def _validate_pre_score_budget(
    output: Path,
    *,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    path = output / PRE_SCORE_BUDGET_FILE
    value = _strict_json(path)
    expected = _pre_score_budget_record(output, protocol_sha256=protocol_sha256)
    file_record = records.file_record(path)
    if (
        value != expected
        or expected["status"] != "within_frozen_ceiling"
        or file_record["mode"] != 0o444
        or cast(int, file_record["bytes"]) > MAX_BUDGET_FILE_BYTES
    ):
        raise InvalidMM009Package("pre-score artifact budget differs or is exceeded")
    return expected


def _expected_formal_tree(output: Path) -> tuple[set[Path], set[Path]]:
    files = {
        PROTOCOL_FILE,
        AUDIT_FILE,
        CONFIG_FILE,
        WORKER_CONFIG_FILE,
        SUPERSESSION_FILE,
        INPUT_MANIFEST_FILE,
        FREEZE_FILE,
        START_FILE,
        CONTROL_FILE,
        DETACHMENT_FILE,
        POST_ISOLATION_FILE,
        PREDICTION_ATTEMPT_FILE,
        PREDICTION_FREEZE_FILE,
        PRE_SCORE_BUDGET_FILE,
        FUTURE_ISOLATION_FILE,
        SCORE_ATTEMPT_FILE,
        EVIDENCE_FILE,
        RESULT_FILE,
        REPORT_FILE,
        Path("isolation/pre-marker/allowed.txt"),
        Path("isolation/pre-marker/sentinel.txt"),
        Path("isolation/pre-marker/isolation-probe.json"),
        Path("isolation/post-detachment/isolation-probe.json"),
        *(PARENT_COPY_ROOT / name for name in records.PARENT_PINS),
    }
    directories: set[Path] = set()

    def add_manifest(prefix: Path, manifest: Mapping[str, records.JsonValue]) -> None:
        artifacts = manifest.get("artifacts")
        if type(artifacts) is not dict:
            raise InvalidMM009Package(f"runtime artifact membership is missing: {prefix}")
        files.update(prefix / name for name in artifacts)
        manifest_directories = manifest.get("directories")
        if manifest_directories is not None:
            if type(manifest_directories) is not list or any(type(name) is not str for name in manifest_directories):
                raise InvalidMM009Package(f"runtime directory membership differs: {prefix}")
            directories.update(prefix / cast(str, name) for name in manifest_directories)

    add_manifest(RUNTIME_ROOT, runtime.runtime_manifest(output / RUNTIME_ROOT))
    add_manifest(DEPENDENCY_ROOT, runtime.numpy_dependency_manifest(output / DEPENDENCY_ROOT))
    add_manifest(STDLIB_ROOT, runtime.stdlib_manifest(output / STDLIB_ROOT))
    add_manifest(CUSTODY_RUNTIME_ROOT, runtime.custody_runtime_manifest(output / CUSTODY_RUNTIME_ROOT))
    for ordinal in range(preparation.MATCHED_ROWS):
        row = f"{ordinal:06d}"
        files.update(
            {
                SOURCE_ROOT / row / "source.npz",
                TARGET_ROOT / row / "target.npz",
                PREDICTION_ROOT / row / "predictions.npy",
                PREDICTION_ROOT / row / "worker-evidence.json",
                PREDICTION_ROOT / row / "commit.json",
                SCORE_ROOT / row / "score.json",
            }
        )
    for file_path in files:
        parent = file_path.parent
        while parent != Path("."):
            directories.add(parent)
            parent = parent.parent
    return files, directories


def _artifact_manifest(output: Path, *, enforce_formal_tree: bool = False) -> dict[str, records.JsonValue]:
    artifacts: dict[str, records.JsonValue] = {}
    total = 0
    directories: list[str] = []
    output_stat = output.lstat()
    if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
        raise InvalidMM009Package("artifact root must be a real directory")
    directory_modes: dict[str, records.JsonValue] = {".": stat.S_IMODE(output_stat.st_mode)}
    for path in sorted(output.rglob("*"), key=lambda item: str(item.relative_to(output))):
        if path.is_symlink():
            raise InvalidMM009Package("artifact package contains a symlink")
        if path.is_dir():
            relative_directory = str(path.relative_to(output))
            directories.append(relative_directory)
            directory_modes[relative_directory] = stat.S_IMODE(path.lstat().st_mode)
        elif path.is_file() and path.relative_to(output) != ARTIFACT_MANIFEST_FILE:
            relative = str(path.relative_to(output))
            record = records.file_record(path)
            artifacts[relative] = record
            total += cast(int, record["bytes"])
        elif not path.is_file():
            raise InvalidMM009Package(f"artifact package contains a non-regular entry: {path}")
    if enforce_formal_tree:
        expected_files, expected_directories = _expected_formal_tree(output)
        if set(map(Path, artifacts)) != expected_files or set(map(Path, directories)) != expected_directories:
            raise InvalidMM009Package("artifact package membership differs from the closed formal tree")
    if total > MAX_ARTIFACT_BYTES:
        raise InvalidMM009Package("artifact package exceeds the frozen byte ceiling")
    return {
        "artifact_bytes": total,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "directories": list(directories),
        "directory_modes": directory_modes,
        "experiment_id": EXPERIMENT_ID,
        "schema_version": "mm009-artifact-manifest-v1",
    }


def _run_formal_in_process(output: Path = DEFAULT_OUTPUT) -> dict[str, records.JsonValue]:
    """Execute the formal body inside the hard-wall supervisor child."""

    _assert_current_cgroup_role("formal")
    deadline = time.monotonic() + TOTAL_WALL_SECONDS
    destination = _assert_output(output)
    _check_deadline(deadline, "output validation")
    manifest, freeze_record = _validate_frozen(destination)
    _check_deadline(deadline, "frozen-package validation")
    if (destination / START_FILE).exists():
        raise FileExistsError("MM-009 formal execution cannot resume or retry")
    protocol = cast(str, manifest["protocol_sha256"])
    config_sha256 = cast(str, manifest["config_sha256"])
    start = _formal_start(destination, manifest, freeze_record)
    records.write_immutable_json_exclusive(destination / START_FILE, start)
    _check_deadline(deadline, "formal marker write")

    # This is intentionally the first scientific operation after the marker.
    controls = synthetic_controls.run_controls(
        config_sha256=config_sha256,
        protocol_sha256=protocol,
    )
    records.write_immutable_json_exclusive(destination / CONTROL_FILE, controls)
    durable_controls = _validate_controls(
        destination,
        config_sha256=config_sha256,
        protocol_sha256=protocol,
    )
    if records.json_file_bytes(durable_controls) != records.json_file_bytes(controls):
        raise InvalidMM009Package(
            "durable formal synthetic evidence differs from the trusted in-memory result"
        )
    _check_deadline(deadline, "formal synthetic controls")

    arrays, rows, normalizers, derangement, parent_alignment = _post_marker_inputs(destination)
    _check_deadline(deadline, "post-marker parent alignment")
    detachment = _detach_rows(
        destination,
        arrays,
        rows,
        normalizers,
        derangement,
        deadline=deadline,
        protocol_sha256=protocol,
    )
    records.write_immutable_json_exclusive(destination / DETACHMENT_FILE, detachment)
    _check_deadline(deadline, "detachment freeze")
    isolation = _post_detachment_probe(destination, rows, deadline=deadline)
    records.write_immutable_json_exclusive(destination / POST_ISOLATION_FILE, isolation)
    _check_deadline(deadline, "post-detachment isolation")
    _predict_all(
        destination,
        rows,
        config_sha256=config_sha256,
        deadline=deadline,
        protocol_sha256=protocol,
    )
    _check_deadline(deadline, "prediction stage")
    _require_pre_score_output_exclusivity(destination)
    budget = _pre_score_budget_record(destination, protocol_sha256=protocol)
    _write_bounded_immutable_json(
        destination / PRE_SCORE_BUDGET_FILE,
        budget,
        maximum_bytes=MAX_BUDGET_FILE_BYTES,
        label="pre-score budget record",
    )
    if budget["status"] != "within_frozen_ceiling":
        raise InvalidMM009Package("projected artifact ceiling exceeded before target scoring")
    _validate_pre_score_budget(destination, protocol_sha256=protocol)
    _check_deadline(deadline, "pre-score artifact budget")
    _run_future_isolation(destination, deadline=deadline)
    score_records, decision_evidence, summary = _run_scores(
        destination,
        rows,
        config_sha256=config_sha256,
        deadline=deadline,
        protocol_sha256=protocol,
    )
    _check_deadline(deadline, "scoring stage")
    evidence = _evidence_record(destination, parent_alignment, score_records, decision_evidence, summary)
    _write_bounded_immutable_json(
        destination / EVIDENCE_FILE,
        evidence,
        maximum_bytes=MAX_EVIDENCE_FILE_BYTES,
        label="formal evidence",
    )
    _check_deadline(deadline, "evidence write")
    result = _result_record(destination, summary)
    _write_bounded_immutable_json(
        destination / RESULT_FILE,
        result,
        maximum_bytes=MAX_RESULT_FILE_BYTES,
        label="formal result",
    )
    _write_bounded_immutable_bytes(
        destination / REPORT_FILE,
        _report_text(result, summary).encode("ascii"),
        maximum_bytes=MAX_REPORT_FILE_BYTES,
        label="formal report",
    )
    _check_deadline(deadline, "result and report writes")
    _write_bounded_immutable_json(
        destination / ARTIFACT_MANIFEST_FILE,
        _artifact_manifest(destination, enforce_formal_tree=True),
        maximum_bytes=MAX_ARTIFACT_MANIFEST_FILE_BYTES,
        label="artifact manifest",
    )
    _check_deadline(deadline, "artifact manifest write")
    verified = verify(destination)
    _check_deadline(deadline, "final fast verification")
    return verified


def _write_formal_receipt(output: Path, receipt: Path) -> None:
    records.write_immutable_json_exclusive(receipt, _run_formal_in_process(output))


def _formal_child_command(output: Path, receipt: Path) -> tuple[str, ...]:
    bootstrap = (
        "import sys;from pathlib import Path;"
        f"sys.path[:0]=[{str(REPO_ROOT)!r},{str(REPO_ROOT / 'src')!r}];"
        "from bench.multimodal_causal_diagnostics import experiment as module;"
        "module._write_formal_receipt(Path(sys.argv[1]),Path(sys.argv[2]))"
    )
    return (
        sys.executable,
        "-I",
        "-B",
        "-c",
        bootstrap,
        str(output),
        str(receipt),
    )


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, records.JsonValue]:
    """Execute once under an OS-enforced descendant-tree total-wall supervisor."""

    destination = _assert_output(output)
    _systemd_containment_preflight()
    with tempfile.TemporaryDirectory(prefix="mm009-formal-receipt-", dir="/tmp") as temporary:
        receipt = Path(temporary).resolve() / "formal-receipt.json"
        try:
            completed = _run_cgroup_supervised_process(
                _formal_child_command(destination, receipt),
                cwd=REPO_ROOT,
                timeout_seconds=float(TOTAL_WALL_SECONDS),
                role="formal",
            )
        except subprocess.TimeoutExpired as error:
            raise InvalidMM009Package("formal hard total-wall deadline exceeded") from error
        if completed.returncode != 0:
            raise InvalidMM009Package(f"formal supervisor child failed: {completed.stderr[-4000:]}")
        if completed.stdout or completed.stderr:
            raise InvalidMM009Package("formal supervisor child emitted an unexpected stream")
        value = _strict_json(receipt)
        if type(value) is not dict:
            raise InvalidMM009Package("formal supervisor result receipt differs")
        verified = cast(dict[str, records.JsonValue], value)
        if records.file_record(receipt)["mode"] != 0o444 or records.read_regular_bytes(
            receipt, maximum_bytes=MAX_JSON_BYTES
        ) != records.json_file_bytes(verified):
            raise InvalidMM009Package("formal supervisor receipt is not immutable canonical JSON")
        return verified


def _validate_controls(
    output: Path,
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    path = output / CONTROL_FILE
    value = _strict_json(path)
    try:
        controls = synthetic_controls.validate_control_evidence(
            value,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        if records.file_record(path)["mode"] != 0o444 or records.read_regular_bytes(
            path, maximum_bytes=MAX_JSON_BYTES
        ) != records.json_file_bytes(controls):
            raise InvalidMM009Package("formal synthetic evidence is not immutable canonical JSON")
    except synthetic_controls.SyntheticControlError as error:
        raise InvalidMM009Package(f"formal synthetic evidence semantics differ: {error}") from error
    except records.RecordValidationError as error:
        raise InvalidMM009Package("formal synthetic evidence record differs") from error
    return controls


def _validate_detachment(
    output: Path,
    arrays: Mapping[str, np.ndarray],
    rows: tuple[preparation.RowIndex, ...],
    normalizers: tuple[preparation.FoldNormalizer, ...],
    derangement: np.ndarray,
    *,
    protocol_sha256: str,
) -> dict[str, object]:
    value = _strict_json(output / DETACHMENT_FILE)
    if type(value) is not dict:
        raise InvalidMM009Package("detachment index must be an object")
    saved = cast(dict[str, object], value)
    sources = saved.get("sources")
    targets = saved.get("targets")
    if (
        saved.get("schema_version") != "mm009-detachment-index-v1"
        or saved.get("row_count") != len(rows)
        or saved.get("row_identity_sha256") != preparation.row_identity_sha256(rows)
        or saved.get("derangement") != derangement.tolist()
        or saved.get("derangement_sha256")
        != records.scientific_array_sha256("half-cycle-derangement", derangement, protocol_sha256=protocol_sha256)
        or type(sources) is not list
        or type(targets) is not list
        or len(sources) != len(rows)
        or len(targets) != len(rows)
    ):
        raise InvalidMM009Package("detachment index header differs")
    for row, source_item, target_item in zip(rows, sources, targets, strict=True):
        if type(source_item) is not dict or type(target_item) is not dict:
            raise InvalidMM009Package("detachment row index schema differs")
        source_path = output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz"
        target_path = output / TARGET_ROOT / f"{row.ordinal:06d}/target.npz"
        source = preparation.load_source_row_npz(source_path)
        target = preparation.load_target_row_npz(target_path)
        preparation.validate_detached_pair(source, target)
        expected_source = preparation.construct_source_row(
            arrays["frames_uint8"], row.ordinal, rows, normalizers, derangement
        )
        expected_target = preparation.construct_target_row(
            arrays["frames_uint8"], row.ordinal, rows, normalizers, derangement
        )
        if any(not np.array_equal(source[name], expected_source[name]) for name in source):
            raise InvalidMM009Package(f"detached source differs from parent row {row.ordinal}")
        if any(not np.array_equal(target[name], expected_target[name]) for name in target):
            raise InvalidMM009Package(f"detached target differs from parent row {row.ordinal}")
        expected_source_item = {
            "file": records.file_record(source_path),
            "manifest": preparation.source_row_manifest(source, protocol_sha256=protocol_sha256),
            "ordinal": row.ordinal,
        }
        expected_target_item = {
            "file": records.file_record(target_path),
            "manifest": preparation.target_row_manifest(target, protocol_sha256=protocol_sha256),
            "ordinal": row.ordinal,
        }
        if source_item != expected_source_item or target_item != expected_target_item:
            raise InvalidMM009Package(f"detachment row manifest differs at {row.ordinal}")
    return saved


def _validate_post_isolation(output: Path, rows: tuple[preparation.RowIndex, ...]) -> dict[str, object]:
    value = _strict_json(output / POST_ISOLATION_FILE)
    if type(value) is not dict:
        raise InvalidMM009Package("post-detachment isolation record must be an object")
    saved = cast(dict[str, object], value)
    expected_manifest: records.JsonValue = [
        {"path_role": "sibling_source", "relative": "rows/source/000001/source.npz"}
    ]
    assert isinstance(expected_manifest, list)
    expected_manifest.extend(
        {
            "path_role": "target_row_directory",
            "relative": str(TARGET_ROOT / f"{row.ordinal:06d}"),
        }
        for row in rows
    )
    probe = saved.get("probe")
    requested_denied_count = len(rows) + 1
    live_python_root_count = len(runtime.live_python_roots())
    if (
        set(saved) != {"denied_manifest", "denied_manifest_sha256", "probe", "probe_file", "schema_version"}
        or saved.get("schema_version") != "mm009-post-detachment-isolation-v1"
        or saved.get("denied_manifest") != expected_manifest
        or saved.get("denied_manifest_sha256")
        != records.canonical_json_sha256(expected_manifest, protocol_sha256=_protocol_sha256())
        or type(probe) is not dict
        or probe.get("schema_version") != "mm009-isolation-probe-v3"
        or probe.get("requested_denied_path_count") != requested_denied_count
        or probe.get("live_python_roots_denied") != live_python_root_count
        or probe.get("denied_path_count") != requested_denied_count + live_python_root_count
        or probe.get("network_families_denied") != 3
        or probe.get("network_socket_variants_denied") != 6
        or probe.get("process_vm_readv_denied") is not True
        or saved.get("probe_file") != records.file_record(output / POST_PROBE_ROOT / "isolation-probe.json")
    ):
        raise InvalidMM009Package("post-detachment isolation evidence differs")
    return saved


def _recompute_score_records(
    output: Path,
    rows: tuple[preparation.RowIndex, ...],
) -> list[dict[str, object]]:
    saved = _score_records(output, rows)
    for row, record in zip(rows, saved, strict=True):
        source = preparation.load_source_row_npz(output / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz")
        target = preparation.load_target_row_npz(output / TARGET_ROOT / f"{row.ordinal:06d}/target.npz")
        predictions = worker.load_prediction_array(output / PREDICTION_ROOT / f"{row.ordinal:06d}/predictions.npy")
        regenerated = score_worker._score_families(  # noqa: SLF001 - independent pure verifier seam
            predictions,
            source,
            target,
            video_id=row.video_id,
            fold_index=row.fold_index,
            video_row=row.video_row,
        )
        if record.get("families") != regenerated:
            raise InvalidMM009Package(f"row score arithmetic differs at ordinal {row.ordinal}")
    return saved


def _validate_completed_science(
    output: Path,
    parent_alignment: Mapping[str, records.JsonValue],
    rows: tuple[preparation.RowIndex, ...],
) -> tuple[decision.DecisionEvidence, decision.DecisionSummary, dict[str, records.JsonValue]]:
    score_records = _recompute_score_records(output, rows)
    evidence_typed, summary = _decision_from_scores(output, rows, score_records)
    expected_evidence = _evidence_record(output, parent_alignment, score_records, evidence_typed, summary)
    if _strict_json(output / EVIDENCE_FILE) != expected_evidence:
        raise InvalidMM009Package("MM-009 evidence does not regenerate")
    expected_result = _result_record(output, summary)
    if _strict_json(output / RESULT_FILE) != expected_result:
        raise InvalidMM009Package("MM-009 result does not regenerate")
    try:
        report = records.read_regular_bytes(output / REPORT_FILE, maximum_bytes=MAX_TEXT_BYTES).decode("ascii")
    except (OSError, UnicodeError, records.RecordValidationError) as error:
        raise InvalidMM009Package("MM-009 report is not ASCII") from error
    if report != _report_text(expected_result, summary):
        raise InvalidMM009Package("MM-009 report does not regenerate")
    return evidence_typed, summary, expected_result


def _validate_bounded_canonical_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> None:
    record = records.file_record(path)
    if record["mode"] != 0o444 or cast(int, record["bytes"]) > maximum_bytes:
        raise InvalidMM009Package(f"{label} file bound or mode differs")
    value = _strict_json(path)
    try:
        canonical = records.json_file_bytes(cast(records.JsonValue, value))
    except records.RecordValidationError as error:
        raise InvalidMM009Package(f"{label} JSON grammar differs") from error
    if records.read_regular_bytes(path, maximum_bytes=maximum_bytes) != canonical:
        raise InvalidMM009Package(f"{label} is not canonical JSON")


def _validate_formal_file_bounds(output: Path) -> None:
    bounded_json = (
        (PRE_SCORE_BUDGET_FILE, MAX_BUDGET_FILE_BYTES, "pre-score budget"),
        (FUTURE_ISOLATION_FILE, MAX_FUTURE_FILE_BYTES, "future isolation"),
        (SCORE_ATTEMPT_FILE, MAX_SCORE_ATTEMPT_BYTES, "score attempt"),
        (EVIDENCE_FILE, MAX_EVIDENCE_FILE_BYTES, "formal evidence"),
        (RESULT_FILE, MAX_RESULT_FILE_BYTES, "formal result"),
        (
            ARTIFACT_MANIFEST_FILE,
            MAX_ARTIFACT_MANIFEST_FILE_BYTES,
            "artifact manifest",
        ),
    )
    for relative, maximum_bytes, label in bounded_json:
        _validate_bounded_canonical_file(
            output / relative,
            maximum_bytes=maximum_bytes,
            label=label,
        )
    report_record = records.file_record(output / REPORT_FILE)
    if report_record["mode"] != 0o444 or cast(int, report_record["bytes"]) > MAX_REPORT_FILE_BYTES:
        raise InvalidMM009Package("formal report file bound or mode differs")
    for ordinal in range(preparation.MATCHED_ROWS):
        _validate_bounded_canonical_file(
            output / SCORE_ROOT / f"{ordinal:06d}/score.json",
            maximum_bytes=MAX_SCORE_FILE_BYTES,
            label=f"row score {ordinal}",
        )


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, records.JsonValue]:
    """Fast, refit-free verification of custody, scores, decision, and exact tree."""

    destination = _assert_output(output)
    manifest, freeze_record = _validate_frozen(destination)
    _validate_formal_file_bounds(destination)
    expected_start = _formal_start(destination, manifest, freeze_record)
    if _strict_json(destination / START_FILE) != expected_start:
        raise InvalidMM009Package("formal start differs from the frozen closure")
    protocol = cast(str, manifest["protocol_sha256"])
    config_sha256 = cast(str, manifest["config_sha256"])
    _validate_controls(destination, config_sha256=config_sha256, protocol_sha256=protocol)
    arrays, rows, normalizers, derangement, parent_alignment = _post_marker_inputs(destination)
    _validate_detachment(
        destination,
        arrays,
        rows,
        normalizers,
        derangement,
        protocol_sha256=protocol,
    )
    _validate_post_isolation(destination, rows)
    _validate_prediction_chain(destination, rows)
    _validate_pre_score_budget(destination, protocol_sha256=protocol)
    _validate_future_isolation(destination)
    score_attempt = _strict_json(destination / SCORE_ATTEMPT_FILE)
    expected_score_attempt = _score_attempt_record(
        destination,
        rows,
        config_sha256=config_sha256,
        protocol_sha256=protocol,
    )
    if score_attempt != expected_score_attempt:
        raise InvalidMM009Package("score attempt did not bind the complete prediction freeze")
    _, summary, result = _validate_completed_science(destination, parent_alignment, rows)
    saved_manifest = _strict_json(destination / ARTIFACT_MANIFEST_FILE)
    manifest_file_record = records.file_record(destination / ARTIFACT_MANIFEST_FILE)
    saved_artifact_bytes = saved_manifest.get("artifact_bytes") if type(saved_manifest) is dict else None
    if (
        manifest_file_record["mode"] != 0o444
        or type(saved_artifact_bytes) is not int
        or saved_artifact_bytes + cast(int, manifest_file_record["bytes"]) > MAX_ARTIFACT_BYTES
        or saved_manifest != _artifact_manifest(destination, enforce_formal_tree=True)
    ):
        raise InvalidMM009Package("artifact manifest or exact tree differs")
    return {
        "artifact_count": cast(int, cast(dict[str, object], saved_manifest)["artifact_count"]),
        "classification": summary.decision,
        "go": summary.go,
        "outcomes": "verified_results",
        "result_sha256": records.file_sha256(destination / RESULT_FILE),
        "status": "verified",
        "branch": cast(str, result["branch"]),
    }


def _verify_semantic_in_process(
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, records.JsonValue]:
    """Run the semantic body inside the hard-wall supervisor child."""

    _assert_current_cgroup_role("semantic")
    started = time.monotonic()
    deadline = started + TOTAL_WALL_SECONDS
    destination = _assert_output(output)
    _check_deadline(deadline, "semantic output validation")
    fast = verify(destination)
    _check_deadline(deadline, "semantic fast verification")
    worker_config = _load_worker_config(destination / WORKER_CONFIG_FILE)
    config_sha256 = cast(str, worker_config["config_sha256"])
    protocol_sha256 = cast(str, worker_config["protocol_sha256"])
    try:
        replayed_controls = synthetic_controls.run_controls(
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        synthetic_controls.validate_control_evidence(
            replayed_controls,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
    except synthetic_controls.SyntheticControlError as error:
        raise InvalidMM009Package("semantic synthetic-control replay failed") from error
    if records.json_file_bytes(replayed_controls) != records.read_regular_bytes(
        destination / CONTROL_FILE,
        maximum_bytes=MAX_JSON_BYTES,
    ):
        raise InvalidMM009Package("semantic synthetic-control replay differs")
    _check_deadline(deadline, "semantic synthetic-control replay")
    rows = preparation.canonical_row_index()
    temporary = Path(tempfile.mkdtemp(prefix="mm009-semantic-", dir="/tmp"))
    process_registry = runtime.WorkerProcessRegistry()
    try:
        replayed_future = temporary / "future-isolation.json"
        _run_future_isolation(destination, evidence_path=replayed_future, deadline=deadline)
        if records.read_regular_bytes(
            replayed_future,
            maximum_bytes=MAX_JSON_BYTES,
        ) != records.read_regular_bytes(
            destination / FUTURE_ISOLATION_FILE,
            maximum_bytes=MAX_JSON_BYTES,
        ):
            raise InvalidMM009Package("semantic future-isolation replay differs")

        def replay(row: preparation.RowIndex) -> None:
            remaining = _deadline_timeout(
                deadline,
                WORKER_TIMEOUT_SECONDS,
                f"semantic prediction row {row.ordinal} launch",
            )
            row_output = temporary / f"{row.ordinal:06d}"
            records.ensure_directory(row_output, mode=0o700)
            runtime.run_worker_process(
                (destination / RUNTIME_ROOT).resolve(),
                (destination / SOURCE_ROOT / f"{row.ordinal:06d}/source.npz").resolve(),
                (destination / WORKER_CONFIG_FILE).resolve(),
                row_output.resolve(),
                dependency_root=(destination / DEPENDENCY_ROOT).resolve(),
                stdlib_root=(destination / STDLIB_ROOT).resolve(),
                timeout_seconds=max(1, math.ceil(remaining)),
                process_registry=process_registry,
            )
            _check_deadline(deadline, f"semantic prediction row {row.ordinal} completion")
            if records.file_sha256(row_output / worker.PREDICTION_FILE) != records.file_sha256(
                destination / PREDICTION_ROOT / f"{row.ordinal:06d}/predictions.npy"
            ):
                raise InvalidMM009Package(f"semantic prediction replay differs at {row.ordinal}")
            if records.file_sha256(row_output / worker.COMMIT_FILE) != records.file_sha256(
                destination / PREDICTION_ROOT / f"{row.ordinal:06d}/worker-evidence.json"
            ):
                raise InvalidMM009Package(f"semantic worker evidence replay differs at {row.ordinal}")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        futures: list[concurrent.futures.Future[None]] = []
        try:
            for row in rows:
                futures.append(executor.submit(replay, row))
            for future in concurrent.futures.as_completed(
                futures,
                timeout=_deadline_timeout(deadline, TOTAL_WALL_SECONDS, "semantic prediction collection"),
            ):
                future.result()
                _check_deadline(deadline, "semantic prediction collection")
        except TimeoutError as error:
            process_registry.cancel_all()
            for future in futures:
                future.cancel()
            raise InvalidMM009Package("semantic total-wall deadline exceeded") from error
        except BaseException:
            process_registry.cancel_all()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if process_registry.active_pids or process_registry.cancelled:
            process_registry.cancel_all()
            raise InvalidMM009Package("semantic prediction process registry did not close cleanly")
    finally:
        process_registry.cancel_all()
        shutil.rmtree(temporary, ignore_errors=True)
    completed = time.monotonic()
    if completed >= deadline:
        raise InvalidMM009Package("formal total-wall deadline exceeded: semantic verification completion")
    elapsed_seconds = completed - started
    return {
        **fast,
        "outcomes": "verified_semantic_results",
        "semantic_controls_replayed": True,
        "semantic_future_isolation_replayed": True,
        "semantic_prediction_rows_replayed": len(rows),
        "semantic_total_wall_elapsed_seconds": elapsed_seconds,
        "semantic_total_wall_limit_seconds": TOTAL_WALL_SECONDS,
        "semantic_replay": (
            "fresh synthetic controls, fitting-free future-isolation, and Landlock/seccomp refit of every "
            "source row matched bit-exact"
        ),
    }


def _write_semantic_receipt(output: Path, receipt: Path) -> None:
    records.write_immutable_json_exclusive(
        receipt,
        _verify_semantic_in_process(output),
    )


def _semantic_child_command(output: Path, receipt: Path) -> tuple[str, ...]:
    bootstrap = (
        "import sys;from pathlib import Path;"
        f"sys.path[:0]=[{str(REPO_ROOT)!r},{str(REPO_ROOT / 'src')!r}];"
        "from bench.multimodal_causal_diagnostics import experiment as module;"
        "module._write_semantic_receipt(Path(sys.argv[1]),Path(sys.argv[2]))"
    )
    return (
        sys.executable,
        "-I",
        "-B",
        "-c",
        bootstrap,
        str(output),
        str(receipt),
    )


def verify_semantic(output: Path = DEFAULT_OUTPUT) -> dict[str, records.JsonValue]:
    """Semantically verify under an OS-enforced descendant-tree hard wall."""

    destination = _assert_output(output)
    _systemd_containment_preflight()
    with tempfile.TemporaryDirectory(prefix="mm009-semantic-receipt-", dir="/tmp") as temporary:
        receipt = Path(temporary).resolve() / "semantic-receipt.json"
        try:
            completed = _run_cgroup_supervised_process(
                _semantic_child_command(destination, receipt),
                cwd=REPO_ROOT,
                timeout_seconds=float(TOTAL_WALL_SECONDS),
                role="semantic",
            )
        except subprocess.TimeoutExpired as error:
            raise InvalidMM009Package("semantic hard total-wall deadline exceeded") from error
        if completed.returncode != 0:
            raise InvalidMM009Package(f"semantic supervisor child failed: {completed.stderr[-4000:]}")
        if completed.stdout or completed.stderr:
            raise InvalidMM009Package("semantic supervisor child emitted an unexpected stream")
        value = _strict_json(receipt)
        if type(value) is not dict:
            raise InvalidMM009Package("semantic supervisor receipt differs")
        result = cast(dict[str, records.JsonValue], value)
        if records.file_record(receipt)["mode"] != 0o444 or records.read_regular_bytes(
            receipt, maximum_bytes=MAX_JSON_BYTES
        ) != records.json_file_bytes(result):
            raise InvalidMM009Package("semantic supervisor receipt is not immutable canonical JSON")
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("audit-template", "prepare", "freeze", "run", "verify", "verify-semantic"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.command == "audit-template":
        value: records.JsonValue = audit_template()
    elif arguments.command == "prepare":
        value = prepare(arguments.output)
    elif arguments.command == "freeze":
        value = freeze(arguments.output)
    elif arguments.command == "run":
        value = run(arguments.output)
    elif arguments.command == "verify-semantic":
        value = verify_semantic(arguments.output)
    else:
        value = verify(arguments.output)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "ARTIFACT_MANIFEST_FILE",
    "AUDIT_FILE",
    "CONFIG_FILE",
    "CONTROL_FILE",
    "DEFAULT_OUTPUT",
    "DETACHMENT_FILE",
    "EVIDENCE_FILE",
    "EXPERIMENT_ID",
    "FREEZE_FILE",
    "INPUT_MANIFEST_FILE",
    "InvalidMM009Package",
    "PREDICTION_FREEZE_FILE",
    "PROTOCOL_FILE",
    "REPORT_FILE",
    "RESULT_FILE",
    "SCHEMA_VERSION",
    "START_FILE",
    "WORKER_CONFIG_FILE",
    "audit_template",
    "freeze",
    "main",
    "prepare",
    "run",
    "verify",
    "verify_semantic",
]
