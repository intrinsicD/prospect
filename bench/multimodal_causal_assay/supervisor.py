"""Host-bound cgroup-v2 supervisor for the MM-011 lexical runtime.

The service command deliberately has two interpreter identities.  A pinned,
terminal CPython binary runs a stdlib-only containment guard.  The guard then
executes that same terminal binary with the literal virtual-environment path as
``argv[0]``.  This preserves the lexical ``sys.executable`` without trusting a
mutable symlink as the executable object.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = "mm011-systemd-cgroup-supervisor-v1"
REPO_ROOT: Final = Path("/home/alex/Documents/prospect")
LEXICAL_PYTHON: Final = REPO_ROOT / ".venv/bin/python"
REAL_PYTHON: Final = Path("/home/alex/miniconda3/bin/python3.12")
SYSTEMD_RUN: Final = Path("/usr/bin/systemd-run")
SYSTEMCTL: Final = Path("/usr/bin/systemctl")
ENV_EXECUTABLE: Final = Path("/usr/bin/env")
CGROUP_CONTROLLERS: Final = Path("/sys/fs/cgroup/cgroup.controllers")
UNIT_PREFIX: Final = "mm011-custody"
RUNTIME_MAX_SECONDS: Final = 14_400.0
_CLEANUP_RESERVE_SECONDS: Final = 30.0
_CAPTURE_LIMIT_BYTES: Final = 8 * 1024 * 1024
_UNIT_ENV: Final = "MM011_CGROUP_UNIT"
_ROLE_ENV: Final = "MM011_CGROUP_ROLE"
_DANGEROUS_LOADER_ENV: Final = ("LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD")

# These records are host pins, not portable discovery defaults.
_EXECUTABLE_PINS: Final = {
    SYSTEMD_RUN: {
        "bytes": 68_392,
        "mode": 0o755,
        "sha256": "49f0bf95eb8a781b93853bf9fc981b4929dd0009f55a3e6db95534c0a2d11716",
    },
    SYSTEMCTL: {
        "bytes": 1_501_304,
        "mode": 0o755,
        "sha256": "7ba82b5ba146759c710e1b80fadaa3fdbc0f9b85c8fb2c8c3196b7b1a0037ef8",
    },
    ENV_EXECUTABLE: {
        "bytes": 48_072,
        "mode": 0o755,
        "sha256": "0aefff8f912fb75716c5d4de3b6acde93edbe8fa280fc8ee895c1226d3e373ef",
    },
    REAL_PYTHON: {
        "bytes": 30_753_048,
        "mode": 0o775,
        "sha256": "d9b73f600a860cefe799d7b4d21bad64719231227c2cf326ca17ed94624841af",
    },
}

_ROLE_PATTERN: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?\Z")
_ENVIRONMENT_NAME_PATTERN: Final = re.compile(r"[A-Z][A-Z0-9_]*\Z")

# No project import is available here: the guard runs with -I -S -B.
_GUARD_SOURCE: Final = (
    "import os,sys;"
    "unit,role,real,lexical=sys.argv[1:5];"
    "lines=open('/proc/self/cgroup',encoding='ascii').read().splitlines();"
    "assert os.environ['MM011_CGROUP_UNIT']==unit;"
    "assert os.environ['MM011_CGROUP_ROLE']==role;"
    "assert all(os.environ.get(name,'')=='' for name in "
    "('LD_AUDIT','LD_LIBRARY_PATH','LD_PRELOAD'));"
    "assert any(line.split(':',2)[:2]==['0',''] and "
    "line.split(':',2)[2].rstrip('/').endswith('/'+unit) for line in lines);"
    "os.execv(real,[lexical,*sys.argv[5:]])"
)


class SupervisorError(RuntimeError):
    """The host boundary or cgroup lifecycle did not fail closed."""


@dataclass(frozen=True, slots=True)
class SupervisedCompletedProcess:
    """Child result plus the independently observed post-child cleanup receipt."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cleanup_receipt: dict[str, object]


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_record(path: Path) -> dict[str, object]:
    """Hash one real executable while binding its path and descriptor identity."""

    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise SupervisorError(f"supervisor executable is missing: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SupervisorError(f"supervisor executable is not one real regular file: {path}")
    if before.st_mode & 0o111 == 0:
        raise SupervisorError(f"supervisor executable has no execute bit: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise SupervisorError(f"supervisor executable changed before open: {path}")
        remaining = opened.st_size
        while remaining:
            payload = os.read(descriptor, min(1024 * 1024, remaining))
            if not payload:
                raise SupervisorError(f"supervisor executable ended early: {path}")
            digest.update(payload)
            remaining -= len(payload)
        if os.read(descriptor, 1):
            raise SupervisorError(f"supervisor executable grew while read: {path}")
        after_descriptor = os.fstat(descriptor)
        try:
            after_path = path.lstat()
        except FileNotFoundError as error:
            raise SupervisorError(f"supervisor executable disappeared: {path}") from error
        if _identity(after_descriptor) != _identity(opened) or _identity(after_path) != _identity(opened):
            raise SupervisorError(f"supervisor executable mutated while read: {path}")
    finally:
        os.close(descriptor)
    return {
        "bytes": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
        "sha256": digest.hexdigest(),
    }


def _validate_executables() -> dict[str, dict[str, object]]:
    records = {str(path): _file_record(path) for path in _EXECUTABLE_PINS}
    expected = {str(path): record for path, record in _EXECUTABLE_PINS.items()}
    if records != expected:
        raise SupervisorError("host supervisor executable pins differ")
    return records


def _validate_cgroup_v2() -> None:
    try:
        metadata = CGROUP_CONTROLLERS.lstat()
        payload = CGROUP_CONTROLLERS.read_text(encoding="ascii")
    except (FileNotFoundError, OSError, UnicodeError) as error:
        raise SupervisorError("cgroup-v2 controllers boundary is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or not payload.strip():
        raise SupervisorError("cgroup-v2 controllers boundary differs")
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise SupervisorError("current cgroup identity is unavailable") from error
    if not any(line.startswith("0::") for line in lines):
        raise SupervisorError("unified cgroup-v2 identity is unavailable")


def _systemd_client_environment() -> dict[str, str]:
    runtime = Path(f"/run/user/{os.getuid()}")
    bus = runtime / "bus"
    try:
        runtime_metadata = runtime.lstat()
        bus_metadata = bus.lstat()
    except FileNotFoundError as error:
        raise SupervisorError("user systemd bus boundary is unavailable") from error
    if (
        stat.S_ISLNK(runtime_metadata.st_mode)
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or stat.S_ISLNK(bus_metadata.st_mode)
        or not stat.S_ISSOCK(bus_metadata.st_mode)
        or runtime_metadata.st_uid != os.getuid()
        or bus_metadata.st_uid != os.getuid()
    ):
        raise SupervisorError("user systemd bus boundary differs")
    return {
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(runtime),
    }


def manifest() -> dict[str, object]:
    """Validate and describe the exact host supervisor boundary."""

    executables = _validate_executables()
    _validate_cgroup_v2()
    _systemd_client_environment()
    return {
        "cgroup_version": 2,
        "capture_limit_bytes": _CAPTURE_LIMIT_BYTES,
        "cleanup_receipt_schema": "mm011-cgroup-cleanup-v1",
        "dynamic_operational_environment": [_ROLE_ENV, _UNIT_ENV],
        "executables": executables,
        "kill_mode": "control-group",
        "kill_signal": "SIGKILL",
        "loader_environment_overrides": {name: "" for name in _DANGEROUS_LOADER_ENV},
        "runtime_max_seconds": 14_400,
        "schema_version": SCHEMA_VERSION,
        "send_sigkill": True,
        "unit_prefix": UNIT_PREFIX,
    }


def _validate_role(role: str) -> None:
    if _ROLE_PATTERN.fullmatch(role) is None:
        raise SupervisorError("MM-011 cgroup role grammar differs")


def _unit_name(role: str) -> str:
    _validate_role(role)
    return f"{UNIT_PREFIX}-{role}-{os.getpid()}-{threading.get_native_id()}-{time.monotonic_ns()}.service"


def _validate_unit_name(unit: str, role: str) -> None:
    _validate_role(role)
    prefix = f"{UNIT_PREFIX}-{role}-"
    if not unit.startswith(prefix) or not unit.endswith(".service"):
        raise SupervisorError("MM-011 cgroup unit identity differs")
    identity = unit[len(prefix) : -len(".service")].split("-")
    if len(identity) != 3 or not all(value.isdecimal() and value for value in identity):
        raise SupervisorError("MM-011 cgroup unit identity grammar differs")


def assert_current_cgroup(role: str) -> str:
    """Require the exact supervisor-provided unit in the current v2 cgroup."""

    unit = os.environ.get(_UNIT_ENV, "")
    observed_role = os.environ.get(_ROLE_ENV, "")
    if observed_role != role:
        raise SupervisorError("MM-011 cgroup role environment differs")
    _validate_unit_name(unit, role)
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise SupervisorError("current MM-011 cgroup identity is unavailable") from error
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[:2] == ["0", ""] and Path(fields[2].rstrip("/")).name == unit:
            return unit
    raise SupervisorError(f"{role} lifecycle is outside its frozen MM-011 cgroup")


def _validate_environment(environment: Mapping[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, value in environment.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or _ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None
            or name in {_UNIT_ENV, _ROLE_ENV, *_DANGEROUS_LOADER_ENV}
            or "\x00" in value
        ):
            raise SupervisorError("MM-011 frozen environment grammar differs")
        output[name] = value
    if not output:
        raise SupervisorError("MM-011 frozen environment is empty")
    return output


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if not command or any(not isinstance(item, str) or not item or "\x00" in item for item in command):
        raise SupervisorError("MM-011 command grammar differs")
    required_prefix = (str(LEXICAL_PYTHON), "-I", "-S", "-B")
    if len(command) < len(required_prefix) or tuple(command[:4]) != required_prefix:
        raise SupervisorError("MM-011 command must retain the pinned lexical argv[0] and -I -S -B prefix")
    try:
        resolved = Path(command[0]).resolve(strict=True)
    except OSError as error:
        raise SupervisorError("MM-011 lexical interpreter is unavailable") from error
    if resolved != REAL_PYTHON:
        raise SupervisorError("MM-011 lexical interpreter terminal path differs")
    if _file_record(resolved) != _EXECUTABLE_PINS[REAL_PYTHON]:
        raise SupervisorError("MM-011 terminal interpreter hash differs")
    return tuple(command)


def _validate_cwd(cwd: Path) -> Path:
    if not isinstance(cwd, Path) or not cwd.is_absolute():
        raise SupervisorError("MM-011 working directory must be absolute")
    try:
        metadata = cwd.lstat()
        resolved = cwd.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise SupervisorError("MM-011 working directory is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or resolved != cwd:
        raise SupervisorError("MM-011 working directory must be one exact real directory")
    return cwd


def _remaining(deadline: float, stage: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0.0:
        raise subprocess.TimeoutExpired(stage, 0.0)
    return value


def _process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_process_group(process: subprocess.Popen[bytes], *, deadline: float) -> None:
    if process.poll() is None or _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    remaining = max(0.001, deadline - time.monotonic())
    try:
        process.wait(timeout=min(1.0, remaining))
    except subprocess.TimeoutExpired as error:
        raise SupervisorError("MM-011 local supervisor process group could not be reaped") from error
    if _process_group_exists(process.pid):
        raise SupervisorError("MM-011 local supervisor left a live process group")


def _read_capture(stream: object, label: str) -> str:
    if not hasattr(stream, "fileno") or not hasattr(stream, "seek") or not hasattr(stream, "read"):
        raise SupervisorError(f"MM-011 {label} capture stream differs")
    size = os.fstat(stream.fileno()).st_size
    if size > _CAPTURE_LIMIT_BYTES:
        raise SupervisorError(f"MM-011 {label} exceeded its capture bound")
    stream.seek(0)
    payload = stream.read()
    if not isinstance(payload, bytes):
        raise SupervisorError(f"MM-011 {label} capture was not binary")
    return payload.decode("utf-8", errors="replace")


def _enforce_capture_bound(stdout_stream: object, stderr_stream: object) -> None:
    if not hasattr(stdout_stream, "fileno") or not hasattr(stderr_stream, "fileno"):
        raise SupervisorError("MM-011 capture streams differ")
    stdout_bytes = os.fstat(stdout_stream.fileno()).st_size
    stderr_bytes = os.fstat(stderr_stream.fileno()).st_size
    if stdout_bytes > _CAPTURE_LIMIT_BYTES or stderr_bytes > _CAPTURE_LIMIT_BYTES:
        raise SupervisorError("MM-011 child exceeded its capture bound")
    if stdout_bytes + stderr_bytes > _CAPTURE_LIMIT_BYTES:
        raise SupervisorError("MM-011 child exceeded its aggregate capture bound")


def _run_cancellable_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    deadline: float,
    reap_deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one local client in a killable process group under one deadline."""

    with tempfile.TemporaryFile(mode="w+b") as stdout_stream, tempfile.TemporaryFile(mode="w+b") as stderr_stream:
        process = subprocess.Popen(  # noqa: S603 - argv is closed and supervisor-constructed
            tuple(command),
            cwd=cwd,
            env=dict(environment),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                _enforce_capture_bound(stdout_stream, stderr_stream)
                if cancel_event is not None and cancel_event.is_set():
                    raise SupervisorError("MM-011 supervisor was cancelled")
                remaining = _remaining(deadline, str(command[0]))
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    continue
            _enforce_capture_bound(stdout_stream, stderr_stream)
            if _process_group_exists(process.pid):
                raise SupervisorError("MM-011 local supervisor left a live descendant")
            return subprocess.CompletedProcess(
                tuple(command),
                process.returncode,
                _read_capture(stdout_stream, "stdout"),
                _read_capture(stderr_stream, "stderr"),
            )
        except BaseException:
            _kill_process_group(process, deadline=deadline if reap_deadline is None else reap_deadline)
            raise


def _systemctl(
    arguments: Sequence[str],
    *,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    return _run_cancellable_process(
        (str(SYSTEMCTL), "--user", *arguments),
        cwd=REPO_ROOT,
        environment=_systemd_client_environment(),
        deadline=deadline,
    )


def _unit_cgroup_paths(unit: str) -> list[str]:
    matches: list[str] = []

    def fail_walk(error: OSError) -> None:
        raise error

    try:
        for base, directories, _files in os.walk(
            "/sys/fs/cgroup",
            topdown=True,
            onerror=fail_walk,
            followlinks=False,
        ):
            directories.sort()
            if unit in directories:
                matches.append(str(Path(base) / unit))
    except OSError as error:
        raise SupervisorError("could not census the cgroup-v2 tree after cleanup") from error
    return matches


def _cleanup_unit(unit: str, *, deadline: float) -> dict[str, object]:
    """Kill the complete service cgroup and prove that it is no longer active."""

    actions: list[dict[str, object]] = []
    for arguments in (
        ("kill", "--kill-whom=all", "--signal=SIGKILL", unit),
        ("stop", unit),
        ("reset-failed", unit),
    ):
        try:
            completed = _systemctl(arguments, deadline=deadline)
            actions.append(
                {
                    "action": arguments[0],
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "stdout": completed.stdout,
                }
            )
        except BaseException as error:  # keep attempting the security cleanup
            actions.append({"action": arguments[0], "error": type(error).__name__})
    try:
        state = _systemctl(("is-active", unit), deadline=deadline)
    except BaseException as error:
        raise SupervisorError(f"could not verify inactive MM-011 cgroup unit: {unit}") from error
    state_name = state.stdout.strip()
    if state.returncode == 0 or state_name not in {"failed", "inactive", "unknown"}:
        raise SupervisorError(f"MM-011 cgroup unit remained active: {unit} ({state_name})")
    cgroup_paths = _unit_cgroup_paths(unit)
    if cgroup_paths:
        raise SupervisorError(f"MM-011 cgroup path remained after cleanup: {unit}")
    return {
        "actions": actions,
        "cgroup_paths": cgroup_paths,
        "is_active_returncode": state.returncode,
        "schema_version": "mm011-cgroup-cleanup-v1",
        "state": state_name,
        "status": "inactive_and_cgroup_absent",
        "unit": unit,
    }


def validate_cleanup_receipt(value: object, *, role: str) -> dict[str, object]:
    """Strictly validate one post-service inactive/absent cleanup receipt."""

    _validate_role(role)
    if not isinstance(value, dict) or set(value) != {
        "actions",
        "cgroup_paths",
        "is_active_returncode",
        "schema_version",
        "state",
        "status",
        "unit",
    }:
        raise SupervisorError("MM-011 cleanup receipt schema differs")
    unit = value.get("unit")
    if not isinstance(unit, str):
        raise SupervisorError("MM-011 cleanup receipt unit differs")
    _validate_unit_name(unit, role)
    if (
        value.get("schema_version") != "mm011-cgroup-cleanup-v1"
        or value.get("status") != "inactive_and_cgroup_absent"
        or value.get("cgroup_paths") != []
        or value.get("state") not in {"failed", "inactive", "unknown"}
        or type(value.get("is_active_returncode")) is not int
        or value.get("is_active_returncode") == 0
    ):
        raise SupervisorError("MM-011 cleanup receipt postcondition differs")
    actions = value.get("actions")
    if not isinstance(actions, list) or len(actions) != 3:
        raise SupervisorError("MM-011 cleanup receipt action count differs")
    for expected_action, action in zip(("kill", "stop", "reset-failed"), actions, strict=True):
        if not isinstance(action, dict) or action.get("action") != expected_action:
            raise SupervisorError("MM-011 cleanup receipt action order differs")
        if set(action) == {"action", "error"}:
            if not isinstance(action["error"], str) or not action["error"]:
                raise SupervisorError("MM-011 cleanup error receipt differs")
        elif set(action) == {"action", "returncode", "stderr", "stdout"}:
            if (
                type(action["returncode"]) is not int
                or not isinstance(action["stderr"], str)
                or not isinstance(action["stdout"], str)
                or len(action["stderr"].encode("utf-8")) > _CAPTURE_LIMIT_BYTES
                or len(action["stdout"].encode("utf-8")) > _CAPTURE_LIMIT_BYTES
            ):
                raise SupervisorError("MM-011 cleanup command receipt differs")
        else:
            raise SupervisorError("MM-011 cleanup action schema differs")
    return value


def _service_command(
    command: tuple[str, ...],
    *,
    cwd: Path,
    role: str,
    unit: str,
    environment: Mapping[str, str],
) -> tuple[str, ...]:
    service_environment = dict(environment)
    service_environment[_ROLE_ENV] = role
    service_environment[_UNIT_ENV] = unit
    guarded = (
        str(ENV_EXECUTABLE),
        "-i",
        *(f"{name}={value}" for name, value in sorted(service_environment.items())),
        str(REAL_PYTHON),
        "-I",
        "-S",
        "-B",
        "-c",
        _GUARD_SOURCE,
        unit,
        role,
        str(REAL_PYTHON),
        command[0],
        *command[1:],
    )
    return (
        str(SYSTEMD_RUN),
        "--user",
        "--wait",
        "--collect",
        "--quiet",
        "--pipe",
        "--expand-environment=no",
        *(f"--setenv={name}=" for name in _DANGEROUS_LOADER_ENV),
        f"--unit={unit}",
        "--service-type=exec",
        "--property=RuntimeMaxSec=14400s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGKILL",
        "--property=SendSIGKILL=yes",
        "--property=Restart=no",
        f"--property=WorkingDirectory={cwd}",
        *guarded,
    )


def run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float = RUNTIME_MAX_SECONDS,
    role: str = "formal",
    cancel_event: threading.Event | None = None,
) -> SupervisedCompletedProcess:
    """Run the pinned lexical interpreter inside one transient user cgroup.

    ``timeout_seconds`` is the lifecycle-root wall ceiling and may only tighten the
    frozen 14,400-second systemd ceiling.  Cleanup is a separate, bounded
    infrastructure finalization tail: it cannot make a scientific result actionable,
    but it must not silently shorten the promised scientific wall interval either.
    """

    if (
        not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= RUNTIME_MAX_SECONDS
    ):
        raise SupervisorError("MM-011 total-wall timeout boundary differs")
    argv = _validate_command(command)
    directory = _validate_cwd(cwd)
    frozen_environment = _validate_environment(environment)
    _validate_role(role)
    manifest()
    unit = _unit_name(role)
    service = _service_command(
        argv,
        cwd=directory,
        role=role,
        unit=unit,
        environment=frozen_environment,
    )
    started = time.monotonic()
    service_deadline = started + float(timeout_seconds)
    cleanup_deadline = service_deadline + _CLEANUP_RESERVE_SECONDS
    try:
        completed = _run_cancellable_process(
            service,
            cwd=REPO_ROOT,
            environment=_systemd_client_environment(),
            deadline=service_deadline,
            reap_deadline=cleanup_deadline,
            cancel_event=cancel_event,
        )
    except BaseException as error:
        receipt = _cleanup_unit(unit, deadline=cleanup_deadline)
        try:
            error.__dict__["mm011_cleanup_receipt"] = receipt
        except (AttributeError, TypeError):
            pass
        # Detect replacement of any executable after the service lifecycle.
        _validate_executables()
        raise
    receipt = _cleanup_unit(unit, deadline=cleanup_deadline)
    # Detect replacement of any executable after the service lifecycle.
    _validate_executables()
    return SupervisedCompletedProcess(argv, completed.returncode, completed.stdout, completed.stderr, receipt)


def preflight(environment: Mapping[str, str]) -> dict[str, object]:
    """Exercise lexical argv[0] and cgroup identity without repository artifacts."""

    source = (
        "import json,os,sys;"
        "unit=os.environ['MM011_CGROUP_UNIT'];"
        "lines=open('/proc/self/cgroup',encoding='ascii').read().splitlines();"
        "assert any(line.split(':',2)[:2]==['0',''] and "
        "line.split(':',2)[2].rstrip('/').endswith('/'+unit) for line in lines);"
        "print(json.dumps({'executable':sys.executable,'proc_exe':os.readlink('/proc/self/exe'),"
        "'unit':unit},sort_keys=True,separators=(',',':')))"
    )
    completed = run(
        (str(LEXICAL_PYTHON), "-I", "-S", "-B", "-c", source),
        cwd=REPO_ROOT,
        environment=environment,
        role="preflight",
    )
    if completed.returncode != 0 or completed.stderr:
        raise SupervisorError(f"MM-011 cgroup preflight failed: {completed.stderr[-1000:]}")
    import json

    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SupervisorError("MM-011 cgroup preflight emitted invalid JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("executable") != str(LEXICAL_PYTHON)
        or value.get("proc_exe") != str(REAL_PYTHON)
        or not isinstance(value.get("unit"), str)
    ):
        raise SupervisorError("MM-011 cgroup preflight runtime identity differs")
    _validate_unit_name(value["unit"], "preflight")
    return value


__all__ = [
    "LEXICAL_PYTHON",
    "REAL_PYTHON",
    "RUNTIME_MAX_SECONDS",
    "SCHEMA_VERSION",
    "SupervisorError",
    "SupervisedCompletedProcess",
    "assert_current_cgroup",
    "manifest",
    "preflight",
    "run",
    "validate_cleanup_receipt",
]
