"""Fail-closed WM-002 Q1 producer and result-free development rehearsal.

The public entry point in this module runs only the complete frozen Q1 budget.
There is deliberately no partial, retry, early-stop, or resume mode.  A private
``Q1ExecutionAuthority`` can be created only after the canonical entry report,
Q0 report, exact selected-source implementation, successor protocol, and salt
commitment all agree.  The private seed schedule and Q1 environments are
constructed only after that validation succeeds.

The independent auditor is intentionally separate and must not import the
producer aggregate helpers in this module.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
import errno
import hashlib
import hmac
import json
import math
import os
import selectors
import shutil
import socket
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import cache, lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Final, Literal, cast

from bench.active_acquisition.attempt import (
    Q1RunIdentity,
    attempt_marker_path,
    claim_attempt,
    derive_run_identity,
    finalize_attempt,
    load_attempt_marker,
)
from bench.active_acquisition.checkpoint import (
    EpisodeAccumulator,
    QualificationBinding,
    dump_q1_checkpoint,
)
from bench.active_acquisition.contracts import (
    ARM_ORDER,
    Q0_REPORT_SHA256,
    Q1_PROTOCOL_PATH,
    Q1_PROTOCOL_VERSION,
    assert_public_value_safe,
    canonical_json_bytes,
    canonical_sha256,
    load_canonical_json,
    schema_documents,
    sha256_bytes,
)
from bench.active_acquisition.problem import TerminalAction
from bench.active_acquisition.runtime_lane import (
    ArmMode,
    CandidateDiagnosticRow,
    EpisodeRuntimeComposition,
    HiddenActuatorAcquisitionEnvironment,
    HiddenActuatorTerminalEnvironment,
    acquisition_public_probe_bytes,
    assert_acquisition_public_noninterference,
    compose_episode_runtime,
    decode_posterior_model,
)
from bench.active_acquisition.seeding import (
    EPISODES_PER_MASTER,
    MASTER_COUNT,
    PrivateEpisodeSeedMaterial,
    PrivateQ1SeedSchedule,
    Q1ExecutionMode,
    derive_public_uniform_selection,
    episodes_per_master,
    nuisance_observation_from_digest,
    private_material_paths,
    public_identity_counter_initialization,
    pulse_observation_from_digest,
    terminal_success_from_digest,
)
from bench.active_acquisition.worker_capability import (
    WORKER_CAPABILITY_ACK_BYTES,
    Q1WorkerCapability,
    consume_worker_capability_fd,
    encode_worker_capability,
    make_worker_capability,
    worker_capability_acknowledgement,
    worker_capability_commitment,
)
from prospect.domain import TimePoint, UpdateStatus
from prospect.runtime import (
    InteractionContext,
    InteractionResult,
    PreparedModelSwap,
    VersionedModelOwner,
)

RAW_TRACE_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-raw-trace.v1"
CHECKPOINT_FRAME_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-checkpoint-frame.v1"
RESTORED_TRACE_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-restored-trace.v1"
AGGREGATE_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-aggregate.v1"
REHEARSAL_AGGREGATE_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-rehearsal-aggregate.v1"
AUDIT_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-independent-audit.v2"

EPISODES_TOTAL: Final = MASTER_COUNT * len(ARM_ORDER) * EPISODES_PER_MASTER
ENVIRONMENT_STEPS_TOTAL: Final = 2 * EPISODES_TOTAL
MAX_RESTORE_CONCURRENCY: Final = 4
PRODUCER_STAGE_TIMEOUT_SECONDS: Final = 3600.0
RESTORE_CHILD_TIMEOUT_SECONDS: Final = 900.0
RESTORE_STAGE_TIMEOUT_SECONDS: Final = 7200.0
PROCESS_TERMINATE_GRACE_SECONDS: Final = 10.0
WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS: Final = 10.0
WORKER_STDOUT_CAPTURE_MAX_BYTES: Final = 0
WORKER_STDERR_CAPTURE_MAX_BYTES: Final = 64 * 1024
_WORKER_CAPTURE_TAIL_BYTES: Final = 2000
_WORKER_CAPTURE_FINISH_TIMEOUT_SECONDS: Final = 2.0
_FRAME_LENGTH: Final = struct.Struct(">Q")
_MAX_JSONL_ROW_BYTES: Final = 32 * 1024 * 1024
_MAX_CHECKPOINT_FRAME_BYTES: Final = 32 * 1024 * 1024
_DEVELOPMENT_PROBE_TIMEOUT_SECONDS: Final = 30.0
_NONINTERFERENCE_SYNTHETIC_INTERACTIONS: Final = 4
_RESOURCE_PADDING_BYTES: Final = 4096
_RESOURCE_MAX_PID: Final = (1 << 31) - 1
_RESOURCE_MAX_FRAME_OFFSET: Final = (1 << 63) - 1
_EXPECTED_ACTION_BY_ARM: Final = {
    ArmMode.PROSPECT: "strong",
    ArmMode.ORACLE: "strong",
    ArmMode.GOAL_ONLY: "skip",
    ArmMode.RAW_ENTROPY: "nuisance",
    ArmMode.EIG_ONLY: "overpowered",
    ArmMode.SHUFFLED_INFORMATION: "weak",
}
_CONTROL_ARMS: Final = (
    "goal_only",
    "raw_observation_entropy",
    "eig_only",
    "shuffled_information",
    "uniform_random",
)
_T_CRITICAL_DF3: Final = 3.182446305284263
_SHA256_CHARS: Final = frozenset("0123456789abcdef")
_CANONICAL_ARTIFACT_NAMES: Final = (
    "raw-trace.jsonl",
    "private-audit.jsonl",
    "checkpoint-index.jsonl",
    "checkpoint-frames.bin",
    "restored-trace.jsonl",
    "aggregate.json",
)


class Q1ExecutionError(RuntimeError):
    """The frozen Q1 execution contract was not satisfied."""


class Q1ChildQuiescenceError(Q1ExecutionError):
    """A child could not be proven exited, so the attempt must remain started."""


@dataclass(frozen=True, slots=True)
class Q1RuntimeDirectoryIdentity:
    """Exact entry-bound identity of one private Q1 runtime directory."""

    canonical_path: str
    st_dev: int
    st_ino: int
    st_uid: int
    st_gid: int
    file_type: str = "directory"
    mode: str = "0700"

    def as_dict(self) -> dict[str, object]:
        return {
            "canonical_path": self.canonical_path,
            "file_type": self.file_type,
            "mode": self.mode,
            "st_dev": self.st_dev,
            "st_gid": self.st_gid,
            "st_ino": self.st_ino,
            "st_uid": self.st_uid,
        }


@dataclass(frozen=True, slots=True)
class _FilesystemIdentity:
    """Stable local identity for a runtime-owned filesystem object."""

    device: int
    inode: int
    owner: int
    group: int


@dataclass(frozen=True, slots=True, repr=False)
class Q1ExecutionAuthority:
    """Validated private capability required by every Q1 producer/restorer."""

    protocol_sha256: str
    implementation_sha256: str
    q0_report_sha256: str
    entry_qualification_sha256: str
    salt_commitment_sha256: str
    run_identity: Q1RunIdentity
    execution_root_identity: Q1RuntimeDirectoryIdentity
    attempt_registry_identity: Q1RuntimeDirectoryIdentity
    execution_mode: Q1ExecutionMode
    secret_salt: bytes = field(repr=False)

    @property
    def episodes_per_master(self) -> int:
        return episodes_per_master(self.execution_mode)

    @property
    def episodes_total(self) -> int:
        return MASTER_COUNT * len(ARM_ORDER) * self.episodes_per_master

    def __repr__(self) -> str:
        return (
            "Q1ExecutionAuthority("
            f"execution_mode={self.execution_mode.value!r}, "
            f"protocol_sha256={self.protocol_sha256!r}, "
            f"implementation_sha256={self.implementation_sha256!r}, "
            f"q0_report_sha256={self.q0_report_sha256!r}, "
            f"entry_qualification_sha256={self.entry_qualification_sha256!r}, "
            f"salt_commitment_sha256={self.salt_commitment_sha256!r}, "
            f"run_id={self.run_identity.run_id!r}, "
            f"attempt_id={self.run_identity.attempt_id!r}, "
            "secret_salt=<redacted>)"
        )

    @property
    def execution_root(self) -> Path:
        return Path(self.execution_root_identity.canonical_path)

    @property
    def attempt_registry_directory(self) -> Path:
        return Path(self.attempt_registry_identity.canonical_path)

    @property
    def checkpoint_binding(self) -> QualificationBinding:
        return QualificationBinding(
            protocol_version=Q1_PROTOCOL_VERSION,
            protocol_sha256=self.protocol_sha256,
            implementation_sha256=self.implementation_sha256,
            q0_report_sha256=self.q0_report_sha256,
            entry_qualification_sha256=self.entry_qualification_sha256,
            salt_commitment_sha256=self.salt_commitment_sha256,
            run_id=self.run_identity.run_id,
            attempt_id=self.run_identity.attempt_id,
        )


@dataclass(frozen=True, slots=True)
class CheckpointFrameLocation:
    """Location of one length-prefixed checkpoint payload."""

    frame_offset: int
    frame_length: int


@dataclass(frozen=True, slots=True)
class EpisodeArtifacts:
    """One live episode's public/private rows and pre-terminal checkpoint."""

    raw_trace: Mapping[str, object]
    private_audit: Mapping[str, object]
    checkpoint_index: Mapping[str, object]
    checkpoint_payload: bytes
    acquisition_public_probe: bytes


@dataclass(frozen=True, slots=True)
class DevelopmentQualificationProbe:
    """Result-free evidence returned to the entry-qualification harness."""

    synthetic_interactions: int
    artifact_samples: Mapping[str, object]
    resource_samples: Mapping[str, int | bool]
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Q1RunOutputs:
    """Paths produced only after a complete no-retry Q1 execution."""

    output_directory: Path
    raw_trace: Path
    private_audit: Path
    checkpoint_index: Path
    checkpoint_frames: Path
    restored_trace: Path
    producer_aggregate: Path


@dataclass(frozen=True, slots=True)
class AuthenticatedWorkerLaunch:
    """Immutable secret-free specification for one exact Q1 child launch."""

    role: Literal["producer", "restore"]
    run_identity: Q1RunIdentity
    parent_pid: int
    master: int
    arm: str | None
    paths: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        probe_child_pid = self.parent_pid + 1
        self.capability(probe_child_pid)

    @property
    def base_command(self) -> tuple[str, ...]:
        if self.role == "producer":
            return (sys.executable, "-S", "-m", "bench.active_acquisition.q1", "_producer-master")
        return (sys.executable, "-S", "-m", "bench.active_acquisition.restore_worker", "q1")

    @property
    def identity(self) -> str:
        suffix = self.arm if self.arm is not None else "producer"
        return f"{self.master}:{suffix}"

    def capability(self, child_pid: int) -> Q1WorkerCapability:
        """Bind this launch to the exact child PID returned by ``Popen``."""

        return make_worker_capability(
            role=self.role,
            run_identity=self.run_identity,
            parent_pid=self.parent_pid,
            child_pid=child_pid,
            master=self.master,
            arm=self.arm,
            paths=dict(self.paths),
        )


class _BoundedWorkerCapture:
    """Continuously drain one child stream into a bounded in-memory tail."""

    def __init__(self, *, label: str, max_bytes: int) -> None:
        if not label or max_bytes < 0:
            raise ValueError("worker capture contract is invalid")
        self.label = label
        self.max_bytes = max_bytes
        self._read_descriptor, self._write_descriptor = os.pipe2(os.O_CLOEXEC)
        os.set_blocking(self._read_descriptor, False)
        self._tail = bytearray()
        self._observed_bytes = 0
        self._overflow = False
        self._monitor_error: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def write_descriptor(self) -> int:
        if self._write_descriptor < 0:
            raise Q1ExecutionError("worker capture writer is already closed")
        return self._write_descriptor

    @property
    def closed(self) -> bool:
        return self._read_descriptor < 0 and self._write_descriptor < 0

    def start(self, process: subprocess.Popen[bytes]) -> None:
        if self._thread is not None or self._write_descriptor < 0:
            raise Q1ExecutionError("worker capture monitor cannot start twice")
        self._process = process
        os.close(self._write_descriptor)
        self._write_descriptor = -1
        self._thread = threading.Thread(
            target=self._drain,
            name=f"wm002-q1-{self.label}-capture",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> tuple[int, bytes, bool]:
        if self._thread is None:
            raise Q1ExecutionError("worker capture monitor was not started")
        if not self._done.wait(_WORKER_CAPTURE_FINISH_TIMEOUT_SECONDS):
            self._stop.set()
            if not self._done.wait(_WORKER_CAPTURE_FINISH_TIMEOUT_SECONDS):
                raise Q1ExecutionError(f"{self.label} capture monitor did not quiesce")
        if self._monitor_error is not None:
            raise Q1ExecutionError(f"{self.label} capture monitor failed: {self._monitor_error}")
        with self._lock:
            return self._observed_bytes, bytes(self._tail), self._overflow

    def close(self) -> None:
        if self._write_descriptor >= 0:
            try:
                os.close(self._write_descriptor)
            except OSError:
                pass
            self._write_descriptor = -1
        self._stop.set()
        if self._thread is not None:
            self._done.wait(_WORKER_CAPTURE_FINISH_TIMEOUT_SECONDS)
        if self._read_descriptor >= 0:
            try:
                os.close(self._read_descriptor)
            except OSError:
                pass
            self._read_descriptor = -1

    def _drain(self) -> None:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self._read_descriptor, selectors.EVENT_READ)
                while not self._stop.is_set():
                    try:
                        chunk = os.read(self._read_descriptor, 64 * 1024)
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        return
                    if chunk:
                        if self._record(chunk):
                            self._terminate_overflowing_process()
                            return
                        continue
                    selector.select(timeout=0.1)
        except OSError as error:
            if not self._stop.is_set():
                self._monitor_error = f"{type(error).__name__}: {error}"
        except BaseException as error:
            self._monitor_error = f"{type(error).__name__}: {error}"
        finally:
            if self._read_descriptor >= 0:
                try:
                    os.close(self._read_descriptor)
                except OSError:
                    pass
                self._read_descriptor = -1
            self._done.set()

    def _record(self, chunk: bytes) -> bool:
        with self._lock:
            self._observed_bytes = min(
                self.max_bytes + 1,
                self._observed_bytes + len(chunk),
            )
            self._tail.extend(chunk)
            if len(self._tail) > _WORKER_CAPTURE_TAIL_BYTES:
                del self._tail[:-_WORKER_CAPTURE_TAIL_BYTES]
            if self._observed_bytes > self.max_bytes:
                self._overflow = True
            return self._overflow

    def _terminate_overflowing_process(self) -> None:
        process = self._process
        if process is None:
            self._monitor_error = "capture overflow occurred without a live process"
            return
        try:
            if process.poll() is None:
                process.terminate()
        except BaseException as error:
            self._monitor_error = f"overflow termination failed: {type(error).__name__}: {error}"


@dataclass(slots=True)
class _AuthenticatedWorkerProcess:
    """Live child plus continuously drained bounded diagnostic captures."""

    launch: AuthenticatedWorkerLaunch
    process: subprocess.Popen[bytes]
    stdout_capture: _BoundedWorkerCapture
    stderr_capture: _BoundedWorkerCapture
    launched_at: float

    def close_captures(self) -> None:
        self.stdout_capture.close()
        self.stderr_capture.close()


def _read_private_salt(path: Path) -> bytes:
    """Read one private regular salt through a single no-follow descriptor."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Q1ExecutionError("secret salt cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise Q1ExecutionError("secret salt must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise Q1ExecutionError("secret salt permissions must be exactly 0600")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) < 32:
        raise Q1ExecutionError("secret salt must contain at least 32 bytes")
    return payload


def _validate_selected_module_origins() -> None:
    """Fail closed unless loaded and module entrypoints are selected repo files."""

    if sys.flags.optimize != 0:
        raise Q1ExecutionError("Q1 requires an unoptimized Python interpreter")
    if sys.flags.no_site != 1:
        raise Q1ExecutionError("Q1 requires Python site initialization to be disabled with -S")

    from bench.active_acquisition import q1_qualification

    repository_root = Path(__file__).resolve().parents[2]
    selected_paths = set(q1_qualification.Q1_IMPLEMENTATION_PATHS)
    main_module = sys.modules.get("__main__")
    main_spec = getattr(main_module, "__spec__", None)
    main_name = getattr(main_spec, "name", None)

    def require_origin(
        module_name: str,
        relative_path: str,
        module: object | None,
    ) -> None:
        if relative_path not in selected_paths:
            raise Q1ExecutionError(f"required runtime module is outside the selected-source closure: {module_name}")
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if not isinstance(origin, str):
            raise Q1ExecutionError(f"required runtime module has no file origin: {module_name}")
        expected = (repository_root / relative_path).resolve()
        if Path(origin).resolve() != expected:
            raise Q1ExecutionError(f"required runtime module origin mismatch: {module_name}")

    executable_paths = {
        "bench.active_acquisition.q1": "bench/active_acquisition/q1.py",
        "bench.active_acquisition.restore_worker": ("bench/active_acquisition/restore_worker.py"),
    }
    if isinstance(main_name, str) and main_name in executable_paths:
        require_origin(
            main_name,
            executable_paths[main_name],
            main_module,
        )
    for module_name, relative_path in executable_paths.items():
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            require_origin(module_name, relative_path, loaded)

    required = {
        "bench": "bench/__init__.py",
        "bench.active_acquisition": "bench/active_acquisition/__init__.py",
        "bench.active_acquisition.attempt": "bench/active_acquisition/attempt.py",
        "bench.active_acquisition.checkpoint": "bench/active_acquisition/checkpoint.py",
        "bench.active_acquisition.contracts": "bench/active_acquisition/contracts.py",
        "bench.active_acquisition.problem": "bench/active_acquisition/problem.py",
        "bench.active_acquisition.q1": "bench/active_acquisition/q1.py",
        "bench.active_acquisition.q1_qualification": ("bench/active_acquisition/q1_qualification.py"),
        "bench.active_acquisition.runtime_lane": ("bench/active_acquisition/runtime_lane.py"),
        "bench.active_acquisition.seeding": "bench/active_acquisition/seeding.py",
        "bench.active_acquisition.worker_capability": "bench/active_acquisition/worker_capability.py",
        "prospect": "src/prospect/__init__.py",
        "prospect.decision": "src/prospect/decision/__init__.py",
        "prospect.domain": "src/prospect/domain/__init__.py",
        "prospect.epistemics": "src/prospect/epistemics/__init__.py",
        "prospect.runtime": "src/prospect/runtime/__init__.py",
        "prospect.storage": "src/prospect/storage/__init__.py",
    }
    for module_name, relative_path in required.items():
        module = sys.modules.get(module_name)
        if module is None and main_name == module_name:
            module = main_module
        require_origin(module_name, relative_path, module)

    for module_name, relative_directory in (
        ("bench", "bench"),
        ("bench.active_acquisition", "bench/active_acquisition"),
        ("prospect", "src/prospect"),
        ("prospect.decision", "src/prospect/decision"),
        ("prospect.domain", "src/prospect/domain"),
        ("prospect.epistemics", "src/prospect/epistemics"),
        ("prospect.runtime", "src/prospect/runtime"),
        ("prospect.storage", "src/prospect/storage"),
    ):
        module = sys.modules[module_name]
        package_path = getattr(module, "__path__", None)
        resolved_paths = tuple(Path(value).resolve() for value in package_path or ())
        if resolved_paths != ((repository_root / relative_directory).resolve(),):
            raise Q1ExecutionError(f"required runtime package search path mismatch: {module_name}")


def _read_stable_protocol_payload(protocol_path: Path) -> bytes:
    """Read protocol bytes twice through one regular no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(protocol_path, flags)
    except OSError as error:
        raise Q1ExecutionError("successor protocol cannot be opened safely") from error

    def read_all() -> bytes:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def signature(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Q1ExecutionError("successor protocol must be a regular non-symlink file")
        payload = read_all()
        after = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        confirmation = read_all()
        confirmed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        signature(before) != signature(after)
        or signature(after) != signature(confirmed)
        or payload != confirmation
        or len(payload) != confirmed.st_size
    ):
        raise Q1ExecutionError("successor protocol changed during its stable descriptor read")
    return payload


def _validate_q1_protocol_execution_boundary(
    protocol_path: Path,
    *,
    execution_mode: Q1ExecutionMode,
) -> str:
    """Bind one stable protocol payload and require the mode's exact boundary.

    ``execution_authorized`` selects the mode: ``true`` permits only the sole
    full-budget production attempt and ``false`` permits only a miniature
    rehearsal. The caller must state which mode it intends, so authorization
    can never silently downgrade a production run into a rehearsal or promote a
    rehearsal into the authorized attempt.
    """

    if protocol_path.resolve() != Q1_PROTOCOL_PATH.resolve():
        raise Q1ExecutionError("Q1 execution requires the canonical successor protocol path")
    protocol_payload = _read_stable_protocol_payload(protocol_path)
    try:
        decoded = json.loads(protocol_payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Q1ExecutionError("successor protocol is not valid JSON") from error
    protocol = _mapping(decoded, "successor protocol")
    if protocol.get("schema") != "prospect.wm002.active-acquisition.q1-protocol.v1":
        raise Q1ExecutionError("successor protocol schema mismatch")
    experiment = _mapping(protocol.get("experiment"), "protocol experiment")
    expected_boundary = {
        "protocol_version": Q1_PROTOCOL_VERSION,
        "claim_eligible": False,
        "formal_authorized": False,
        "execution_authorized": execution_mode is Q1ExecutionMode.PRODUCTION,
    }
    for name, expected in expected_boundary.items():
        actual = experiment.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise Q1ExecutionError(f"successor protocol execution boundary mismatch: {name}")
    return sha256_bytes(protocol_payload)


def _protocol_execution_mode(protocol_path: Path) -> tuple[str, Q1ExecutionMode]:
    """Derive the single mode the canonical protocol bytes currently permit."""

    for mode in (Q1ExecutionMode.PRODUCTION, Q1ExecutionMode.REHEARSAL):
        try:
            return _validate_q1_protocol_execution_boundary(protocol_path, execution_mode=mode), mode
        except Q1ExecutionError:
            continue
    raise Q1ExecutionError("successor protocol satisfies no Q1 execution boundary")


def _parse_q1_runtime_directory_identity(
    value: object,
    *,
    label: str,
) -> Q1RuntimeDirectoryIdentity:
    row = _mapping(value, label)
    expected_fields = {
        "canonical_path",
        "file_type",
        "mode",
        "st_dev",
        "st_gid",
        "st_ino",
        "st_uid",
    }
    if set(row) != expected_fields:
        raise Q1ExecutionError(f"{label} fields differ from the exact directory identity contract")
    canonical_path = row["canonical_path"]
    if not isinstance(canonical_path, str) or not Path(canonical_path).is_absolute():
        raise Q1ExecutionError(f"{label} canonical_path must be an absolute string")
    if row["file_type"] != "directory" or type(row["file_type"]) is not str:
        raise Q1ExecutionError(f"{label} file_type must be directory")
    if row["mode"] != "0700" or type(row["mode"]) is not str:
        raise Q1ExecutionError(f"{label} mode must be 0700")
    integer_fields: dict[str, int] = {}
    for field_name in ("st_dev", "st_gid", "st_ino", "st_uid"):
        field_value = row[field_name]
        if type(field_value) is not int or field_value < 0:
            raise Q1ExecutionError(f"{label} {field_name} must be a nonnegative exact integer")
        integer_fields[field_name] = field_value
    return Q1RuntimeDirectoryIdentity(
        canonical_path=canonical_path,
        st_dev=integer_fields["st_dev"],
        st_ino=integer_fields["st_ino"],
        st_uid=integer_fields["st_uid"],
        st_gid=integer_fields["st_gid"],
    )


def _capture_q1_runtime_directory_identity(
    path: Path,
    *,
    label: str,
) -> Q1RuntimeDirectoryIdentity:
    """Recompute all seven bound fields through a stable private descriptor."""

    descriptor, identity = _open_private_directory_descriptor(path, label=label)
    try:
        try:
            canonical_before = path.resolve(strict=True)
            _require_directory_descriptor_identity(
                descriptor,
                path=path,
                expected_identity=identity,
                label=label,
            )
            _require_directory_descriptor_identity(
                descriptor,
                path=canonical_before,
                expected_identity=identity,
                label=label,
            )
            canonical_after = path.resolve(strict=True)
            metadata = os.fstat(descriptor)
        except (OSError, RuntimeError) as error:
            raise Q1ExecutionError(f"{label} canonical identity cannot be verified") from error
        if canonical_after != canonical_before:
            raise Q1ExecutionError(f"{label} canonical path changed during identity capture")
        _require_private_directory_metadata(metadata, label=label)
        return Q1RuntimeDirectoryIdentity(
            canonical_path=str(canonical_before),
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_uid=metadata.st_uid,
            st_gid=metadata.st_gid,
        )
    finally:
        os.close(descriptor)


def _entry_bound_q1_runtime_directories(
    report: Mapping[str, object],
    *,
    label: str,
) -> tuple[Q1RuntimeDirectoryIdentity, Q1RuntimeDirectoryIdentity]:
    resources = _mapping(report.get("resource_preflight"), f"{label} resource preflight")
    execution_identity = _parse_q1_runtime_directory_identity(
        resources.get("execution_root"),
        label=f"{label} execution root identity",
    )
    registry_identity = _parse_q1_runtime_directory_identity(
        resources.get("attempt_registry_directory"),
        label=f"{label} attempt registry identity",
    )
    if (execution_identity.st_dev, execution_identity.st_ino) == (
        registry_identity.st_dev,
        registry_identity.st_ino,
    ) or execution_identity.canonical_path == registry_identity.canonical_path:
        raise Q1ExecutionError(f"{label} runtime directories must have distinct identities")
    execution_path = Path(execution_identity.canonical_path)
    registry_path = Path(registry_identity.canonical_path)
    if execution_path in registry_path.parents or registry_path in execution_path.parents:
        raise Q1ExecutionError(f"{label} runtime directories must not be nested")
    return execution_identity, registry_identity


def _require_q1_runtime_directory_identity(
    path: Path,
    expected: Q1RuntimeDirectoryIdentity,
    *,
    label: str,
) -> None:
    observed = _capture_q1_runtime_directory_identity(path, label=label)
    if observed != expected:
        raise Q1ExecutionError(f"{label} differs from its entry-bound directory identity")


def _revalidate_q1_runtime_directories(
    authority: Q1ExecutionAuthority,
) -> tuple[Path, Path]:
    execution_root = authority.execution_root
    attempt_registry = authority.attempt_registry_directory
    _require_q1_runtime_directory_identity(
        execution_root,
        authority.execution_root_identity,
        label="Q1 execution root",
    )
    _require_q1_runtime_directory_identity(
        attempt_registry,
        authority.attempt_registry_identity,
        label="Q1 attempt registry",
    )
    return execution_root, attempt_registry


def validate_q1_execution_authority(
    *,
    entry_report_path: Path,
    q0_report_path: Path,
    secret_salt_path: Path,
    protocol_path: Path = Q1_PROTOCOL_PATH,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    execution_mode: Q1ExecutionMode = Q1ExecutionMode.PRODUCTION,
) -> Q1ExecutionAuthority:
    """Validate every execution binding before constructing private schedule state."""

    _validate_selected_module_origins()
    protocol_sha256 = _validate_q1_protocol_execution_boundary(
        protocol_path,
        execution_mode=execution_mode,
    )

    secret_salt = _read_private_salt(secret_salt_path)
    salt_commitment = hashlib.sha256(secret_salt).hexdigest()

    from bench.active_acquisition.q1_qualification import run_entry_qualification

    recomputed = run_entry_qualification(
        q0_report_path=q0_report_path,
        secret_salt_path=secret_salt_path,
        prospective_review_path=prospective_review_path,
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        protocol_path=protocol_path,
        execution_mode=execution_mode,
    )
    recomputed_value = recomputed.as_dict()
    expected_recomputed_identity = {
        "protocol_version": Q1_PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "q0_report_sha256": Q0_REPORT_SHA256,
        "salt_commitment_sha256": salt_commitment,
    }
    for field_name, expected in expected_recomputed_identity.items():
        if recomputed_value.get(field_name) != expected:
            raise Q1ExecutionError(
                f"fresh Q1 entry qualification identity differs from the parent binding: {field_name}"
            )
    recomputed_report = _mapping(recomputed_value, "fresh Q1 entry qualification")
    recomputed_execution_identity, recomputed_registry_identity = (
        _entry_bound_q1_runtime_directories(
            recomputed_report,
            label="fresh Q1 entry qualification",
        )
    )
    _require_q1_runtime_directory_identity(
        execution_root,
        recomputed_execution_identity,
        label="fresh Q1 execution root",
    )
    _require_q1_runtime_directory_identity(
        attempt_registry_directory,
        recomputed_registry_identity,
        label="fresh Q1 attempt registry",
    )
    final_protocol_sha256 = _validate_q1_protocol_execution_boundary(
        protocol_path,
        execution_mode=execution_mode,
    )
    if final_protocol_sha256 != protocol_sha256:
        raise Q1ExecutionError("successor protocol changed across fresh entry qualification")
    final_salt_commitment = hashlib.sha256(_read_private_salt(secret_salt_path)).hexdigest()
    if final_salt_commitment != salt_commitment:
        raise Q1ExecutionError("secret salt changed across fresh entry qualification")
    recomputed_payload = canonical_json_bytes(
        recomputed_value,
        newline=True,
    )
    entry_payload = entry_report_path.read_bytes()
    if entry_payload != recomputed_payload:
        raise Q1ExecutionError("supplied Q1 entry report differs from fresh exact qualification recomputation")
    try:
        entry_decoded = json.loads(entry_payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Q1ExecutionError("supplied Q1 entry report is not JSON") from error
    entry_report = _mapping(entry_decoded, "supplied Q1 entry report")
    entry_execution_identity, entry_registry_identity = _entry_bound_q1_runtime_directories(
        entry_report,
        label="supplied Q1 entry report",
    )
    if (
        entry_execution_identity != recomputed_execution_identity
        or entry_registry_identity != recomputed_registry_identity
    ):
        raise Q1ExecutionError("supplied Q1 entry directory identities differ from fresh recomputation")
    _require_q1_runtime_directory_identity(
        Path(entry_execution_identity.canonical_path),
        entry_execution_identity,
        label="entry-bound Q1 execution root",
    )
    _require_q1_runtime_directory_identity(
        Path(entry_registry_identity.canonical_path),
        entry_registry_identity,
        label="entry-bound Q1 attempt registry",
    )
    if not recomputed.passed:
        raise Q1ExecutionError("fresh Q1 entry qualification did not pass")

    entry_qualification_sha256 = sha256_bytes(entry_payload)
    run_identity = derive_run_identity(
        protocol_version=Q1_PROTOCOL_VERSION,
        protocol_sha256=protocol_sha256,
        implementation_sha256=recomputed.implementation_sha256,
        q0_report_sha256=Q0_REPORT_SHA256,
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=salt_commitment,
    )
    return Q1ExecutionAuthority(
        protocol_sha256=protocol_sha256,
        implementation_sha256=recomputed.implementation_sha256,
        q0_report_sha256=Q0_REPORT_SHA256,
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=salt_commitment,
        run_identity=run_identity,
        execution_root_identity=entry_execution_identity,
        attempt_registry_identity=entry_registry_identity,
        execution_mode=execution_mode,
        secret_salt=secret_salt,
    )


def append_checkpoint_frame(stream: BinaryIO, payload: bytes) -> CheckpointFrameLocation:
    """Append one ``uint64 length || payload`` frame and return its header offset."""

    if not isinstance(payload, bytes) or not payload:
        raise Q1ExecutionError("checkpoint frame payload must be nonempty bytes")
    if len(payload) > _MAX_CHECKPOINT_FRAME_BYTES:
        raise Q1ExecutionError(f"checkpoint frame exceeds the {_MAX_CHECKPOINT_FRAME_BYTES}-byte bound")
    offset = stream.tell()
    stream.write(_FRAME_LENGTH.pack(len(payload)))
    stream.write(payload)
    return CheckpointFrameLocation(frame_offset=offset, frame_length=len(payload))


def validate_q1_child_execution_authority(
    *,
    entry_report_path: Path,
    q0_report_path: Path,
    secret_salt_path: Path,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    protocol_path: Path = Q1_PROTOCOL_PATH,
) -> Q1ExecutionAuthority:
    """Verify immutable parent-qualified inputs without rerunning synthetic probes.

    The child never trusts its parent for the execution mode: it derives the
    mode from the canonical protocol bytes it reads itself, and the entry
    report it must match already binds those bytes by digest.
    """

    _validate_selected_module_origins()
    protocol_sha256, execution_mode = _protocol_execution_mode(protocol_path)
    if sha256_bytes(q0_report_path.read_bytes()) != Q0_REPORT_SHA256:
        raise Q1ExecutionError("Q1 child accepted Q0 report digest mismatch")

    entry_payload = entry_report_path.read_bytes()
    try:
        decoded = json.loads(entry_payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Q1ExecutionError("Q1 child entry report is not JSON") from error
    if entry_payload != canonical_json_bytes(decoded, newline=True):
        raise Q1ExecutionError("Q1 child entry report is not canonical JSON")
    entry_report = _mapping(decoded, "Q1 child entry report")

    from bench.active_acquisition.q1_qualification import validate_entry_report

    try:
        validate_entry_report(entry_report)
    except Exception as error:
        raise Q1ExecutionError("Q1 child entry report failed strict selected-source validation") from error
    if entry_report.get("passed") is not True:
        raise Q1ExecutionError("Q1 child entry report did not pass")
    if entry_report.get("protocol_sha256") != protocol_sha256:
        raise Q1ExecutionError("Q1 child protocol digest differs from its entry report")
    prospective_review_sha256 = entry_report.get("prospective_review_sha256")
    if (
        not isinstance(prospective_review_sha256, str)
        or sha256_bytes(prospective_review_path.read_bytes()) != prospective_review_sha256
    ):
        raise Q1ExecutionError("Q1 child prospective review digest mismatch")

    entry_execution_identity, entry_registry_identity = _entry_bound_q1_runtime_directories(
        entry_report,
        label="Q1 child entry report",
    )
    _require_q1_runtime_directory_identity(
        execution_root,
        entry_execution_identity,
        label="Q1 child execution root",
    )
    _require_q1_runtime_directory_identity(
        attempt_registry_directory,
        entry_registry_identity,
        label="Q1 child attempt registry",
    )

    secret_salt = _read_private_salt(secret_salt_path)
    salt_commitment = hashlib.sha256(secret_salt).hexdigest()
    if entry_report.get("salt_commitment_sha256") != salt_commitment:
        raise Q1ExecutionError("Q1 child salt commitment differs from its entry report")
    implementation_sha256 = _require_sha256(
        cast(str, entry_report.get("implementation_sha256")),
        "implementation_sha256",
    )
    entry_qualification_sha256 = sha256_bytes(entry_payload)
    run_identity = derive_run_identity(
        protocol_version=Q1_PROTOCOL_VERSION,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=Q0_REPORT_SHA256,
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=salt_commitment,
    )
    _require_q1_runtime_directory_identity(
        Path(entry_execution_identity.canonical_path),
        entry_execution_identity,
        label="entry-bound Q1 child execution root",
    )
    _require_q1_runtime_directory_identity(
        Path(entry_registry_identity.canonical_path),
        entry_registry_identity,
        label="entry-bound Q1 child attempt registry",
    )
    return Q1ExecutionAuthority(
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=Q0_REPORT_SHA256,
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=salt_commitment,
        run_identity=run_identity,
        execution_root_identity=entry_execution_identity,
        attempt_registry_identity=entry_registry_identity,
        execution_mode=execution_mode,
        secret_salt=secret_salt,
    )


def read_checkpoint_frame(
    path: Path,
    *,
    frame_offset: int,
    frame_length: int,
) -> bytes:
    """Read one externally indexed frame and reject prefix/truncation mismatch."""

    with _private_binary_reader(path) as stream:
        return _read_checkpoint_frame_stream(
            stream,
            frame_offset=frame_offset,
            frame_length=frame_length,
        )


def _read_checkpoint_frame_stream(
    stream: BinaryIO,
    *,
    frame_offset: int,
    frame_length: int,
) -> bytes:
    """Read an indexed frame from an already-open stream."""

    if frame_offset < 0 or frame_length <= 0:
        raise Q1ExecutionError("checkpoint frame coordinates are invalid")
    if frame_length > _MAX_CHECKPOINT_FRAME_BYTES:
        raise Q1ExecutionError(f"checkpoint frame exceeds the {_MAX_CHECKPOINT_FRAME_BYTES}-byte bound")
    stream.seek(frame_offset)
    header = stream.read(_FRAME_LENGTH.size)
    if len(header) != _FRAME_LENGTH.size:
        raise Q1ExecutionError("checkpoint frame header is truncated")
    declared = _FRAME_LENGTH.unpack(header)[0]
    if declared != frame_length:
        raise Q1ExecutionError("checkpoint frame length prefix differs from its index")
    payload = stream.read(frame_length)
    if len(payload) != frame_length:
        raise Q1ExecutionError("checkpoint frame payload is truncated")
    return payload


def run_q1(
    *,
    entry_report_path: Path,
    q0_report_path: Path,
    secret_salt_path: Path,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    output_directory: Path,
) -> Q1RunOutputs:
    """Claim and run the sole full-budget Q1 attempt without a retry path."""

    return _run_q1_attempt(
        entry_report_path=entry_report_path,
        q0_report_path=q0_report_path,
        secret_salt_path=secret_salt_path,
        prospective_review_path=prospective_review_path,
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        output_directory=output_directory,
        execution_mode=Q1ExecutionMode.PRODUCTION,
    )


def run_q1_orchestration_rehearsal(
    *,
    entry_report_path: Path,
    q0_report_path: Path,
    secret_salt_path: Path,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    output_directory: Path,
) -> Q1RunOutputs:
    """Run the identical orchestration at the miniature rehearsal budget.

    This is result-free mechanics coverage for the four-producer,
    28-restore-lane, merge, validation, publication, and completed-marker path.
    It requires ``execution_authorized: false``, so it is unreachable once the
    protocol authorizes Q1, and its artifacts carry the rehearsal aggregate
    schema and a two-episode lane budget that every frozen Q1 contract rejects.
    """

    return _run_q1_attempt(
        entry_report_path=entry_report_path,
        q0_report_path=q0_report_path,
        secret_salt_path=secret_salt_path,
        prospective_review_path=prospective_review_path,
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        output_directory=output_directory,
        execution_mode=Q1ExecutionMode.REHEARSAL,
    )


def _run_q1_attempt(
    *,
    entry_report_path: Path,
    q0_report_path: Path,
    secret_salt_path: Path,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    output_directory: Path,
    execution_mode: Q1ExecutionMode,
) -> Q1RunOutputs:
    """Claim one attempt registry and execute exactly one orchestration."""

    _validate_q1_execution_root(
        execution_root=execution_root,
        output_directory=output_directory,
    )
    authority = validate_q1_execution_authority(
        entry_report_path=entry_report_path,
        q0_report_path=q0_report_path,
        secret_salt_path=secret_salt_path,
        prospective_review_path=prospective_review_path,
        execution_root=execution_root,
        attempt_registry_directory=attempt_registry_directory,
        execution_mode=execution_mode,
    )
    execution_root = authority.execution_root
    attempt_registry_directory = authority.attempt_registry_directory
    output_directory = execution_root / output_directory.name
    _validate_q1_execution_root(
        execution_root=execution_root,
        output_directory=output_directory,
    )
    if _path_lexists(output_directory):
        raise Q1ExecutionError("Q1 output directory must not already exist")
    incomplete = output_directory.with_name(output_directory.name + ".incomplete")
    if _path_lexists(incomplete):
        raise Q1ExecutionError("an incomplete Q1 execution directory already exists")
    worker_capability_secret = os.urandom(32)
    worker_capability_sha256 = worker_capability_commitment(worker_capability_secret)
    execution_root, attempt_registry_directory = _revalidate_q1_runtime_directories(authority)
    marker_path = claim_attempt(
        attempt_registry_directory,
        authority.run_identity,
        worker_capability_sha256=worker_capability_sha256,
    )
    try:
        outputs = _execute_claimed_q1(
            authority=authority,
            entry_report_path=entry_report_path,
            q0_report_path=q0_report_path,
            secret_salt_path=secret_salt_path,
            prospective_review_path=prospective_review_path,
            execution_root=execution_root,
            attempt_registry_directory=attempt_registry_directory,
            output_directory=output_directory,
            worker_capability_secret=worker_capability_secret,
        )
        final_paths = (
            outputs.raw_trace,
            outputs.private_audit,
            outputs.checkpoint_index,
            outputs.checkpoint_frames,
            outputs.restored_trace,
            outputs.producer_aggregate,
        )
        _require_exact_private_artifact_set(
            final_paths,
            directory=outputs.output_directory,
        )
        artifacts = {
            "aggregate": outputs.producer_aggregate,
            "checkpoint_frames": outputs.checkpoint_frames,
            "checkpoint_index": outputs.checkpoint_index,
            "private_audit": outputs.private_audit,
            "raw_trace": outputs.raw_trace,
            "restored_trace": outputs.restored_trace,
        }
        _revalidate_q1_runtime_directories(authority)
        finalize_attempt(
            marker_path,
            authority.run_identity,
            status="completed",
            expected_worker_capability_sha256=worker_capability_sha256,
            artifacts=artifacts,
        )
    except BaseException as error:
        _finalize_failed_q1_attempt(
            error=error,
            marker_path=marker_path,
            authority=authority,
            output_directory=output_directory,
            incomplete_directory=incomplete,
            expected_worker_capability_sha256=worker_capability_sha256,
        )
        raise
    return outputs


def _execute_claimed_q1(
    *,
    authority: Q1ExecutionAuthority,
    entry_report_path: Path,
    q0_report_path: Path,
    secret_salt_path: Path,
    prospective_review_path: Path,
    execution_root: Path,
    attempt_registry_directory: Path,
    output_directory: Path,
    worker_capability_secret: bytes,
) -> Q1RunOutputs:
    """Execute the full budget after the parent has durably claimed the attempt."""

    bound_execution_root, bound_registry = _revalidate_q1_runtime_directories(authority)
    if execution_root.resolve(strict=True) != bound_execution_root:
        raise Q1ExecutionError("claimed Q1 execution root differs from its authority")
    if attempt_registry_directory.resolve(strict=True) != bound_registry:
        raise Q1ExecutionError("claimed Q1 attempt registry differs from its authority")
    execution_root = bound_execution_root
    attempt_registry_directory = bound_registry
    _validate_q1_execution_root(
        execution_root=execution_root,
        output_directory=output_directory,
    )
    output_directory = execution_root / output_directory.name
    incomplete = output_directory.with_name(output_directory.name + ".incomplete")
    _mkdir_private_exact(incomplete, label="incomplete Q1 directory")
    common_paths: dict[str, str | None] = {
        "attempt_registry_directory": str(attempt_registry_directory.resolve()),
        "entry_report_path": str(entry_report_path.resolve()),
        "execution_root": str(execution_root.resolve()),
        "incomplete_directory": str(incomplete.resolve()),
        "prospective_review_path": str(prospective_review_path.resolve()),
        "q0_report_path": str(q0_report_path.resolve()),
        "secret_salt_path": str(secret_salt_path.resolve()),
    }
    parent_pid = os.getpid()
    producer_processes: list[tuple[int, _AuthenticatedWorkerProcess]] = []
    try:
        for master in range(MASTER_COUNT):
            master_dir = incomplete / f"master-{master}"
            _mkdir_private_exact(master_dir, label="producer master directory")
            paths = {
                **common_paths,
                "frame_path": None,
                "index_path": None,
                "master_directory": str(master_dir.resolve()),
                "output_path": str(master_dir.resolve()),
                "raw_trace_path": None,
            }
            launch = AuthenticatedWorkerLaunch(
                role="producer",
                run_identity=authority.run_identity,
                parent_pid=parent_pid,
                master=master,
                arm=None,
                paths=tuple(sorted(paths.items())),
            )
            process = _spawn_authenticated_worker(launch, secret=worker_capability_secret)
            producer_processes.append((master, process))
    except BaseException as error:
        try:
            _quiesce_after_failure(
                tuple((identity, child.process) for identity, child in producer_processes),
                label="producer master startup",
                cause=error,
            )
        finally:
            _close_authenticated_processes(child for _identity, child in producer_processes)
        raise
    _wait_exact_processes(producer_processes, "producer master")

    restorer_launches: list[tuple[str, AuthenticatedWorkerLaunch]] = []
    for master in range(MASTER_COUNT):
        master_dir = incomplete / f"master-{master}"
        for arm in ARM_ORDER:
            paths = {
                **common_paths,
                "frame_path": str((master_dir / f"{arm}.frames.bin").resolve()),
                "index_path": str((master_dir / f"{arm}.index.jsonl").resolve()),
                "master_directory": str(master_dir.resolve()),
                "output_path": str((master_dir / f"{arm}.restored.jsonl").resolve()),
                "raw_trace_path": str((master_dir / "raw.jsonl").resolve()),
            }
            launch = AuthenticatedWorkerLaunch(
                role="restore",
                run_identity=authority.run_identity,
                parent_pid=parent_pid,
                master=master,
                arm=arm,
                paths=tuple(sorted(paths.items())),
            )
            restorer_launches.append((launch.identity, launch))
    _run_bounded_commands(
        restorer_launches,
        secret=worker_capability_secret,
        label="restore lane",
        max_concurrency=MAX_RESTORE_CONCURRENCY,
    )

    raw_path = incomplete / "raw-trace.jsonl"
    private_path = incomplete / "private-audit.jsonl"
    index_path = incomplete / "checkpoint-index.jsonl"
    frames_path = incomplete / "checkpoint-frames.bin"
    restored_path = incomplete / "restored-trace.jsonl"
    aggregate_path = incomplete / "aggregate.json"
    _merge_worker_outputs(
        root=incomplete,
        raw_path=raw_path,
        private_path=private_path,
        index_path=index_path,
        frames_path=frames_path,
        restored_path=restored_path,
    )
    aggregate = _stream_validate_and_aggregate(
        raw_path=raw_path,
        private_path=private_path,
        index_path=index_path,
        frames_path=frames_path,
        restored_path=restored_path,
        execution_mode=authority.execution_mode,
    )
    _validate_aggregate(aggregate, execution_mode=authority.execution_mode)
    with _private_binary_writer(aggregate_path) as aggregate_stream:
        aggregate_stream.write(canonical_json_bytes(aggregate, newline=True))
    canonical_paths = (
        raw_path,
        private_path,
        index_path,
        frames_path,
        restored_path,
        aggregate_path,
    )
    _remove_worker_trees_and_require_exact_artifacts(root=incomplete, canonical_paths=canonical_paths)
    publication_identities = _fsync_publication_set(
        canonical_paths,
        directory=incomplete,
    )
    _revalidate_q1_runtime_directories(authority)
    _publish_directory_noreplace(incomplete, output_directory)
    published_paths = tuple(output_directory / path.name for path in canonical_paths)
    _require_exact_private_artifact_set(
        published_paths,
        directory=output_directory,
        expected_identities=publication_identities,
    )
    _fsync_directory(output_directory.parent)
    return Q1RunOutputs(
        output_directory=output_directory,
        raw_trace=output_directory / raw_path.name,
        private_audit=output_directory / private_path.name,
        checkpoint_index=output_directory / index_path.name,
        checkpoint_frames=output_directory / frames_path.name,
        restored_trace=output_directory / restored_path.name,
        producer_aggregate=output_directory / aggregate_path.name,
    )


def run_development_qualification_probe(
    *,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    salt_commitment_sha256: str,
) -> DevelopmentQualificationProbe:
    """Exercise synthetic authoritative paths without a Q1 schedule or Q1 draw."""

    started_at = time.perf_counter()
    probe_deadline = started_at + _DEVELOPMENT_PROBE_TIMEOUT_SECONDS
    violations: list[str] = []
    live: list[EpisodeArtifacts] = []
    entry_qualification_sha256 = hashlib.sha256(b"wm002-explicit-synthetic-development-entry").hexdigest()
    run_identity = derive_run_identity(
        protocol_version=Q1_PROTOCOL_VERSION,
        protocol_sha256=_require_sha256(protocol_sha256, "protocol_sha256"),
        implementation_sha256=_require_sha256(implementation_sha256, "implementation_sha256"),
        q0_report_sha256=_require_sha256(q0_report_sha256, "q0_report_sha256"),
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=_require_sha256(salt_commitment_sha256, "salt_commitment_sha256"),
    )
    binding = QualificationBinding(
        protocol_version=Q1_PROTOCOL_VERSION,
        protocol_sha256=_require_sha256(protocol_sha256, "protocol_sha256"),
        implementation_sha256=_require_sha256(
            implementation_sha256,
            "implementation_sha256",
        ),
        q0_report_sha256=_require_sha256(q0_report_sha256, "q0_report_sha256"),
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=_require_sha256(
            salt_commitment_sha256,
            "salt_commitment_sha256",
        ),
        run_id=run_identity.run_id,
        attempt_id=run_identity.attempt_id,
    )
    for ordinal, arm in enumerate(ArmMode):
        try:
            artifact = _run_synthetic_episode(
                arm=arm,
                synthetic_ordinal=ordinal,
                binding=binding,
                protocol_sha256=protocol_sha256,
                implementation_sha256=implementation_sha256,
                q0_report_sha256=q0_report_sha256,
                salt_commitment_sha256=salt_commitment_sha256,
            )
            live.append(artifact)
            _validate_artifact_fast("raw_trace", artifact.raw_trace)
            _validate_artifact_fast("private_audit", artifact.private_audit)
            expected_action = (
                ("skip", "weak", "strong", "overpowered", "nuisance")[ordinal % 5]
                if arm is ArmMode.UNIFORM_RANDOM
                else _EXPECTED_ACTION_BY_ARM[arm]
            )
            acquired = _mapping(artifact.raw_trace["acquisition"], "synthetic acquisition")
            if acquired.get("selected_action") != expected_action:
                violations.append(f"{arm.value}:unexpected selected action")
        except Exception as error:
            violations.append(f"{arm.value}:{type(error).__name__}:{error}")
    if not live:
        return DevelopmentQualificationProbe(
            synthetic_interactions=0,
            artifact_samples=MappingProxyType({}),
            resource_samples=MappingProxyType({}),
            violations=tuple(sorted(set(violations))),
        )

    first = live[0]
    checkpoint_sample = dict(first.checkpoint_index)
    checkpoint_sample["frame_offset"] = 0
    checkpoint_sample["frame_length"] = len(first.checkpoint_payload)
    restored_sample: Mapping[str, object] | None = None
    try:
        remaining = probe_deadline - time.perf_counter()
        if remaining <= 0.0:
            raise Q1ExecutionError("result-free development probe exhausted its fresh-process timeout")
        restored_sample = _fresh_process_synthetic_restore(
            artifact=first,
            binding=binding,
            timeout_seconds=remaining,
        )
        _validate_artifact_fast("restored_trace", restored_sample)
        if restored_sample["restorer_pid"] == restored_sample["producer_pid"]:
            violations.append("synthetic restore PID did not differ from producer PID")
    except Exception as error:
        violations.append(f"fresh_process_restore:{type(error).__name__}:{error}")

    aggregate_sample = _aggregate_schema_sample(
        binding=binding,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    audit_sample = _audit_schema_sample(
        binding=binding,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    samples: dict[str, object] = {
        "raw_trace": first.raw_trace,
        "private_audit": first.private_audit,
        "checkpoint_frame": checkpoint_sample,
        "restored_trace": restored_sample or _restored_schema_sample(first.raw_trace),
        "aggregate": aggregate_sample,
        "audit_output": audit_sample,
    }
    for name, sample in samples.items():
        try:
            _validate_artifact_fast(name, sample)
        except Exception as error:
            violations.append(f"sample:{name}:{type(error).__name__}:{error}")
    try:
        _exercise_synthetic_noninterference(
            artifact=first,
            binding=binding,
            protocol_sha256=protocol_sha256,
            implementation_sha256=implementation_sha256,
            q0_report_sha256=q0_report_sha256,
            salt_commitment_sha256=salt_commitment_sha256,
            public_samples={name: value for name, value in samples.items() if name != "private_audit"},
        )
    except Exception as error:
        violations.append(f"noninterference:{type(error).__name__}:{error}")
    restored_for_resources = restored_sample or _restored_schema_sample(first.raw_trace)
    resource_samples = _development_resource_samples(
        live=live,
        restored_trace=restored_for_resources,
        duration_under_30_seconds=time.perf_counter() < probe_deadline,
    )
    return DevelopmentQualificationProbe(
        synthetic_interactions=(
            2 * len(live) + (1 if restored_sample is not None else 0) + _NONINTERFERENCE_SYNTHETIC_INTERACTIONS
        ),
        artifact_samples=MappingProxyType(samples),
        resource_samples=resource_samples,
        violations=tuple(sorted(set(violations))),
    )


def _run_q1_episode(
    *,
    authority: Q1ExecutionAuthority,
    schedule: PrivateQ1SeedSchedule,
    master: int,
    arm: ArmMode,
    episode: int,
    producer_pid: int,
) -> EpisodeArtifacts:
    """Run one live Q1 episode; callers can obtain inputs only after entry validation."""

    if schedule.salt_commitment_sha256 != authority.salt_commitment_sha256:
        raise Q1ExecutionError("private schedule differs from the validated salt commitment")
    theta = schedule.theta(master, episode)
    uniform = derive_public_uniform_selection(master, episode) if arm is ArmMode.UNIFORM_RANDOM else None
    expected_action = uniform.action_id if uniform is not None else _EXPECTED_ACTION_BY_ARM[arm]
    return _run_live_episode(
        master=master,
        episode=episode,
        arm=arm,
        producer_pid=producer_pid,
        identity_next_counter=public_identity_counter_initialization(master, episode, arm.value),
        acquisition_observed_symbol=schedule.acquisition_observation(
            master,
            episode,
            expected_action,
        ),
        terminal_success=lambda decision: schedule.terminal_success(
            master,
            episode,
            decision,
            theta,
        ),
        uniform_ordinal=(uniform.index if uniform is not None else None),
        uniform_selector_digest=(uniform.semantic_key_sha256 if uniform is not None else None),
        semantic_key_sha256=schedule.theta_reference(
            master,
            episode,
        ).semantic_key_sha256,
        binding=authority.checkpoint_binding,
        protocol_sha256=authority.protocol_sha256,
        implementation_sha256=authority.implementation_sha256,
        q0_report_sha256=authority.q0_report_sha256,
        salt_commitment_sha256=authority.salt_commitment_sha256,
        entry_qualification_sha256=authority.entry_qualification_sha256,
        private_audit=schedule.private_audit_material(
            master,
            episode,
            arm.value,
        ),
        expected_action=expected_action,
        public_privacy_schedule=schedule,
    )


def _run_synthetic_episode(
    *,
    arm: ArmMode,
    synthetic_ordinal: int,
    binding: QualificationBinding,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    salt_commitment_sha256: str,
) -> EpisodeArtifacts:
    """Run explicit development values that cannot overlap the Q1 schedule."""

    uniform_ordinal = synthetic_ordinal % 5 if arm is ArmMode.UNIFORM_RANDOM else None
    uniform_digest = (
        hashlib.sha256(f"wm002-development-uniform:{synthetic_ordinal}".encode("ascii")).hexdigest()
        if arm is ArmMode.UNIFORM_RANDOM
        else None
    )
    expected_action = (
        ("skip", "weak", "strong", "overpowered", "nuisance")[cast(int, uniform_ordinal)]
        if arm is ArmMode.UNIFORM_RANDOM
        else _EXPECTED_ACTION_BY_ARM[arm]
    )
    synthetic_master = MASTER_COUNT - 1
    synthetic_episode = EPISODES_PER_MASTER - 1
    private = _synthetic_private_material(
        arm=arm,
        master=synthetic_master,
        episode=synthetic_episode,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    acquisition_potential, terminal_potential = _synthetic_private_potential_outcomes(private)
    synthetic_observed_symbol = acquisition_potential[expected_action]
    return _run_live_episode(
        master=synthetic_master,
        episode=synthetic_episode,
        arm=arm,
        producer_pid=os.getpid(),
        identity_next_counter=0,
        acquisition_observed_symbol=synthetic_observed_symbol,
        terminal_success=lambda decision: terminal_potential[int(decision)],
        uniform_ordinal=uniform_ordinal,
        uniform_selector_digest=uniform_digest,
        semantic_key_sha256=hashlib.sha256(f"wm002-development-semantic:{arm.value}".encode("ascii")).hexdigest(),
        binding=binding,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
        entry_qualification_sha256=binding.entry_qualification_sha256,
        private_audit=private,
        expected_action=expected_action,
        public_privacy_schedule=None,
        exercise_transaction_rollback=synthetic_ordinal == 0,
    )


def _run_live_episode(
    *,
    master: int,
    episode: int,
    arm: ArmMode,
    producer_pid: int,
    identity_next_counter: int,
    acquisition_observed_symbol: int,
    terminal_success: Callable[[TerminalAction], bool],
    uniform_ordinal: int | None,
    uniform_selector_digest: str | None,
    semantic_key_sha256: str,
    binding: QualificationBinding,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    salt_commitment_sha256: str,
    entry_qualification_sha256: str,
    private_audit: PrivateEpisodeSeedMaterial,
    expected_action: str,
    public_privacy_schedule: PrivateQ1SeedSchedule | None,
    exercise_transaction_rollback: bool = False,
    checkpoint_observer: Callable[[bytes], None] | None = None,
) -> EpisodeArtifacts:
    episode_id = f"{binding.run_id}:m{master}:a{arm.value}:e{episode}"
    identity_namespace = episode_id
    runtime = compose_episode_runtime(
        episode_id=episode_id,
        arm=arm,
        identity_namespace=identity_namespace,
        identity_next_counter=identity_next_counter,
        uniform_ordinal=uniform_ordinal,
        uniform_selector_digest=uniform_selector_digest,
    )
    predecessor = runtime.model_owner.snapshot_state()
    acquisition_environment = HiddenActuatorAcquisitionEnvironment(
        agent_id=runtime.agent_id,
        observed_symbol=acquisition_observed_symbol,
        identities=runtime.identities,
    )
    acquisition = runtime.acquisition_agent.interact(
        acquisition_environment,
        runtime.acquisition_goal,
        context=InteractionContext(
            run_id=binding.run_id,
            task_id=runtime.acquisition_goal.task_id,
            episode_id=episode_id,
            step_index=0,
        ),
        decide_at=TimePoint(1),
    )
    selected_action = _selected_acquisition_action(acquisition)
    if selected_action != expected_action:
        raise Q1ExecutionError("live arm selected an action other than its frozen selector")
    candidate_rows = _acquisition_candidate_rows(runtime.acquisition_policy.last_diagnostic_rows)
    learn_at = TimePoint(acquisition.transition.created_at.tick + 1)
    if exercise_transaction_rollback:
        _exercise_synthetic_transaction_rollback(runtime, acquisition, at=learn_at)
    receipt = runtime.acquisition_agent.learn(
        (acquisition.transition,),
        at=learn_at,
    )
    if receipt.status is not UpdateStatus.APPLIED:
        raise Q1ExecutionError("the sole acquisition update did not apply")
    candidate_model = runtime.model_owner.snapshot_state()
    acquisition_public_probe = acquisition_public_probe_bytes(
        candidate_rows=runtime.acquisition_policy.last_diagnostic_rows,
        decision=acquisition.decision,
        transition=acquisition.transition,
        receipt=receipt,
        model_state=candidate_model,
        identity_next_counter=runtime.identities.next_counter,
        experience_count=len(runtime.store),
        transition_count=runtime.ledger.transition_count,
        update_count=runtime.ledger.update_count,
    )
    before = decode_posterior_model(predecessor.payload)
    after = decode_posterior_model(candidate_model.payload)
    acquisition_outcome = _mapping(
        acquisition.experience.outcome.evidence.payload,
        "acquisition outcome",
    )
    acquisition_observation = _mapping(
        acquisition.experience.observation.evidence.payload,
        "acquisition observation",
    )
    accumulator = EpisodeAccumulator(
        task_payoff=_number(acquisition_outcome["task_payoff"], "task_payoff"),
        physical_action_cost=_number(
            acquisition_outcome["physical_action_cost"],
            "physical_action_cost",
        ),
        information_acquisition_cost=_number(
            acquisition_outcome["information_acquisition_cost"],
            "information_acquisition_cost",
        ),
    )
    preterminal = runtime.acquisition_agent.snapshot(receipt.completed_at)
    checkpoint = dump_q1_checkpoint(
        checkpoint_id=f"{episode_id}:checkpoint:preterminal",
        agent_id=runtime.agent_id,
        created_at=receipt.completed_at,
        model_owner=runtime.model_owner,
        predecessor_model_sha256=predecessor.digest,
        snapshot=preterminal,
        experience=acquisition.experience,
        transition=acquisition.transition,
        receipt=receipt,
        identity_source=runtime.identities,
        accumulator=accumulator,
        binding=binding,
    )
    if checkpoint_observer is not None:
        checkpoint_observer(checkpoint.payload)
    selected_terminal = TerminalAction.DIRECT if after.posterior_direct >= Fraction(1, 2) else TerminalAction.REVERSED
    terminal_environment = HiddenActuatorTerminalEnvironment(
        agent_id=runtime.agent_id,
        terminal_success=terminal_success(selected_terminal),
        identities=runtime.identities,
    )
    terminal = runtime.terminal_agent.interact(
        terminal_environment,
        runtime.terminal_goal,
        context=InteractionContext(
            run_id=binding.run_id,
            task_id=runtime.terminal_goal.task_id,
            episode_id=episode_id,
            step_index=1,
        ),
        decide_at=TimePoint(receipt.completed_at.tick + 1),
    )
    terminal_outcome = _mapping(
        terminal.experience.outcome.evidence.payload,
        "terminal outcome",
    )
    actual_terminal = terminal_outcome.get("executed_terminal_action")
    if actual_terminal != int(selected_terminal):
        raise Q1ExecutionError("terminal execution differs from the posterior decision")
    terminal_reward = _number(terminal_outcome["terminal_reward"], "terminal_reward")
    episode_return = (
        accumulator.task_payoff
        - accumulator.physical_action_cost
        - accumulator.information_acquisition_cost
        + terminal_reward
    )
    terminal_candidate_rows = _terminal_candidate_rows(terminal)
    terminal_rows_sha256 = canonical_sha256(terminal_candidate_rows)
    components = {row.component_id: row.sha256 for row in checkpoint.report.component_digests}
    raw_trace: dict[str, object] = {
        "schema": RAW_TRACE_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "q0_report_sha256": q0_report_sha256,
        "salt_commitment_sha256": salt_commitment_sha256,
        "entry_qualification_sha256": entry_qualification_sha256,
        "master": master,
        "arm": arm.value,
        "episode": episode,
        "producer_pid": producer_pid,
        "semantic_key_sha256": semantic_key_sha256,
        "candidate_rows": candidate_rows,
        "candidate_rows_sha256": canonical_sha256(candidate_rows),
        "acquisition": {
            "selected_action": selected_action,
            "uniform_selector_sha256": uniform_selector_digest,
            "observed_symbol": acquisition_observation["observed_symbol"],
            "task_payoff": accumulator.task_payoff,
            "physical_action_cost": accumulator.physical_action_cost,
            "information_acquisition_cost": accumulator.information_acquisition_cost,
            "net_reward": _number(acquisition_outcome["net_reward"], "net_reward"),
            "decision_id": acquisition.decision.decision_id,
            "intention_id": acquisition.decision.intended_action.intention_id,
            "execution_id": _execution_id(acquisition),
            "observation_id": acquisition.experience.observation.observation_id,
            "outcome_id": acquisition.experience.outcome.outcome_id,
            "experience_id": acquisition.experience.experience_id,
            "transition_id": acquisition.transition.transition_id,
            "receipt_id": receipt.receipt_id,
            "model_before_sha256": predecessor.digest,
            "model_after_sha256": candidate_model.digest,
            "model_before_version": predecessor.version,
            "model_after_version": candidate_model.version,
            "posterior_before": _fraction_string(before.posterior_direct),
            "posterior_after": _fraction_string(after.posterior_direct),
            "evidence_count_before": before.evidence_count,
            "evidence_count_after": after.evidence_count,
        },
        "checkpoint": {
            "sha256": checkpoint.report.aggregate_sha256,
            "size_bytes": checkpoint.report.size_bytes,
            "component_sha256": components,
        },
        "terminal": {
            "candidate_rows": terminal_candidate_rows,
            "candidate_rows_sha256": terminal_rows_sha256,
            "selected_action": int(selected_terminal),
            "success": terminal_outcome["success"],
            "terminal_reward": terminal_reward,
            "episode_return": episode_return,
            "decision_id": terminal.decision.decision_id,
            "intention_id": terminal.decision.intended_action.intention_id,
            "execution_id": _execution_id(terminal),
            "observation_id": terminal.experience.observation.observation_id,
            "outcome_id": terminal.experience.outcome.outcome_id,
            "experience_id": terminal.experience.experience_id,
            "transition_id": terminal.transition.transition_id,
        },
        "counts": {
            "decisions": 2,
            "executions": 2,
            "experiences": len(runtime.store),
            "transitions": runtime.ledger.transition_count,
            "acquisition_updates": runtime.ledger.update_count,
            "terminal_updates": 0,
        },
    }
    checkpoint_index = {
        "schema": CHECKPOINT_FRAME_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "entry_qualification_sha256": entry_qualification_sha256,
        "master": master,
        "arm": arm.value,
        "episode": episode,
        "frame_offset": 0,
        "frame_length": len(checkpoint.payload),
        "frame_header_bytes": _FRAME_LENGTH.size,
        "checkpoint_sha256": checkpoint.report.aggregate_sha256,
        "component_sha256": components,
    }
    private_audit_row = {
        **private_audit.as_private_dict(),
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
    }
    _validate_artifact_fast("raw_trace", raw_trace)
    _validate_artifact_fast("private_audit", private_audit_row)
    if public_privacy_schedule is not None:
        public_privacy_schedule.assert_public_artifact(
            raw_trace,
            private_hmac_digests=private_audit.private_hmac_digests(),
        )
    else:
        assert_public_value_safe(raw_trace)
    return EpisodeArtifacts(
        raw_trace=raw_trace,
        private_audit=private_audit_row,
        checkpoint_index=checkpoint_index,
        checkpoint_payload=checkpoint.payload,
        acquisition_public_probe=acquisition_public_probe,
    )


def _exercise_synthetic_transaction_rollback(
    runtime: EpisodeRuntimeComposition,
    acquisition: InteractionResult,
    *,
    at: TimePoint,
) -> None:
    """Prove post-commit failure compensation on one result-free transition."""

    before_model = runtime.model_owner.snapshot_state()
    before_state = runtime.state.snapshot(runtime.agent_id, at)
    before_store = tuple(runtime.store.history(runtime.agent_id, at))
    before_transitions = tuple(runtime.ledger.transitions(runtime.agent_id, at))
    before_updates = tuple(runtime.ledger.updates(runtime.agent_id, at))
    before_counts = (
        len(runtime.store),
        runtime.ledger.transition_count,
        runtime.ledger.update_count,
    )
    original_commit = VersionedModelOwner.commit_swap
    commit_applied = False

    def commit_then_raise(
        owner: VersionedModelOwner,
        swap: PreparedModelSwap,
    ) -> None:
        nonlocal commit_applied
        original_commit(owner, swap)
        commit_applied = True
        raise Q1ExecutionError("synthetic post-commit failure")

    setattr(VersionedModelOwner, "commit_swap", commit_then_raise)  # noqa: B010
    try:
        try:
            runtime.acquisition_agent.learn((acquisition.transition,), at=at)
        except Q1ExecutionError as error:
            if str(error) != "synthetic post-commit failure":
                raise
        else:
            raise Q1ExecutionError("synthetic post-commit failure did not propagate")
    finally:
        setattr(VersionedModelOwner, "commit_swap", original_commit)  # noqa: B010

    after_counts = (
        len(runtime.store),
        runtime.ledger.transition_count,
        runtime.ledger.update_count,
    )
    if not commit_applied:
        raise Q1ExecutionError("synthetic model commit was not reached")
    if runtime.model_owner.snapshot_state() != before_model:
        raise Q1ExecutionError("synthetic rollback changed the model snapshot")
    if runtime.state.snapshot(runtime.agent_id, at) != before_state:
        raise Q1ExecutionError("synthetic rollback changed the agent-state snapshot")
    if tuple(runtime.store.history(runtime.agent_id, at)) != before_store or after_counts != before_counts:
        raise Q1ExecutionError("synthetic rollback changed the experience store or ledger counts")
    if (
        tuple(runtime.ledger.transitions(runtime.agent_id, at)) != before_transitions
        or tuple(runtime.ledger.updates(runtime.agent_id, at)) != before_updates
    ):
        raise Q1ExecutionError("synthetic rollback changed ledger records")


def _require_local_worker_capability(
    capability: Q1WorkerCapability,
    *,
    role: Literal["producer", "restore"],
    worker_capability_sha256: str,
) -> None:
    if capability.role != role:
        raise Q1ExecutionError("worker capability role differs from the child entrypoint")
    if capability.parent_pid != os.getppid() or capability.child_pid != os.getpid():
        raise Q1ExecutionError("worker capability PID binding differs from the live process")
    _validate_selected_module_origins()
    if _require_sha256(worker_capability_sha256, "worker_capability_sha256") == "0" * 64:
        raise Q1ExecutionError("worker capability commitment must be nonzero")


def _run_master_worker(
    *,
    capability: Q1WorkerCapability,
    worker_capability_sha256: str,
) -> None:
    _require_local_worker_capability(
        capability,
        role="producer",
        worker_capability_sha256=worker_capability_sha256,
    )
    paths = dict(capability.paths)
    master = capability.master
    output_directory = Path(cast(str, paths["output_path"]))
    output_identity = _require_private_directory(
        output_directory,
        label="producer master output directory",
    )
    if any(output_directory.iterdir()):
        raise Q1ExecutionError("producer master output directory must be empty")
    _require_private_directory(
        output_directory,
        expected_identity=output_identity,
        label="producer master output directory",
    )
    _require_started_child_attempt(
        attempt_registry_directory=Path(cast(str, paths["attempt_registry_directory"])),
        expected_run_identity=capability.run_identity,
        expected_worker_capability_sha256=worker_capability_sha256,
    )
    authority = validate_q1_child_execution_authority(
        entry_report_path=Path(cast(str, paths["entry_report_path"])),
        q0_report_path=Path(cast(str, paths["q0_report_path"])),
        prospective_review_path=Path(cast(str, paths["prospective_review_path"])),
        secret_salt_path=Path(cast(str, paths["secret_salt_path"])),
        execution_root=Path(cast(str, paths["execution_root"])),
        attempt_registry_directory=Path(cast(str, paths["attempt_registry_directory"])),
    )
    if authority.run_identity != capability.run_identity:
        raise Q1ExecutionError("child run/attempt identity differs from fresh derivation")
    schedule = PrivateQ1SeedSchedule(authority.secret_salt)
    raw_path = output_directory / "raw.jsonl"
    private_path = output_directory / "private.jsonl"
    with _private_binary_writer(raw_path) as raw_stream, _private_binary_writer(private_path) as private_stream:
        for arm_name in ARM_ORDER:
            arm = ArmMode(arm_name)
            frame_path = output_directory / f"{arm_name}.frames.bin"
            index_path = output_directory / f"{arm_name}.index.jsonl"
            with _private_binary_writer(frame_path) as frame_stream, _private_binary_writer(index_path) as index_stream:
                for episode in range(authority.episodes_per_master):
                    artifact = _run_q1_episode(
                        authority=authority,
                        schedule=schedule,
                        master=master,
                        arm=arm,
                        episode=episode,
                        producer_pid=os.getpid(),
                    )
                    location = append_checkpoint_frame(
                        frame_stream,
                        artifact.checkpoint_payload,
                    )
                    index = dict(artifact.checkpoint_index)
                    index["frame_offset"] = location.frame_offset
                    index["frame_length"] = location.frame_length
                    _validate_artifact_fast("checkpoint_frame", index)
                    raw_stream.write(canonical_json_bytes(artifact.raw_trace, newline=True))
                    private_stream.write(canonical_json_bytes(artifact.private_audit, newline=True))
                    index_stream.write(canonical_json_bytes(index, newline=True))


def _merge_worker_outputs(
    *,
    root: Path,
    raw_path: Path,
    private_path: Path,
    index_path: Path,
    frames_path: Path,
    restored_path: Path,
) -> None:
    with (
        _private_binary_writer(raw_path) as raw_output,
        _private_binary_writer(private_path) as private_output,
        _private_binary_writer(index_path) as index_output,
        _private_binary_writer(frames_path) as frame_output,
        _private_binary_writer(restored_path) as restored_output,
    ):
        for master in range(MASTER_COUNT):
            master_dir = root / f"master-{master}"
            _copy_bytes(master_dir / "raw.jsonl", raw_output)
            _copy_bytes(master_dir / "private.jsonl", private_output)
            for arm in ARM_ORDER:
                pair_frames = master_dir / f"{arm}.frames.bin"
                with _private_binary_reader(pair_frames) as pair_stream:
                    for index in _read_jsonl(master_dir / f"{arm}.index.jsonl"):
                        _validate_artifact_fast("checkpoint_frame", index)
                        payload = _read_checkpoint_frame_stream(
                            pair_stream,
                            frame_offset=cast(int, index["frame_offset"]),
                            frame_length=cast(int, index["frame_length"]),
                        )
                        location = append_checkpoint_frame(frame_output, payload)
                        merged = dict(index)
                        merged["frame_offset"] = location.frame_offset
                        merged["frame_length"] = location.frame_length
                        index_output.write(canonical_json_bytes(merged, newline=True))
                    if pair_stream.tell() != os.fstat(pair_stream.fileno()).st_size:
                        raise Q1ExecutionError("worker checkpoint sidecar contains trailing bytes")
                _copy_bytes(master_dir / f"{arm}.restored.jsonl", restored_output)


def _stream_validate_and_aggregate(
    *,
    raw_path: Path,
    private_path: Path,
    index_path: Path,
    frames_path: Path,
    restored_path: Path,
    execution_mode: Q1ExecutionMode,
) -> dict[str, object]:
    """Validate consolidated artifacts in lockstep while retaining bounded state."""

    lane_episodes = episodes_per_master(execution_mode)
    episodes_total = MASTER_COUNT * len(ARM_ORDER) * lane_episodes
    totals = {(master, arm): 0.0 for master in range(MASTER_COUNT) for arm in ARM_ORDER}
    counts = {(master, arm): 0 for master in range(MASTER_COUNT) for arm in ARM_ORDER}
    binding_fields = (
        "run_id",
        "attempt_id",
        "protocol_sha256",
        "implementation_sha256",
        "q0_report_sha256",
        "entry_qualification_sha256",
        "salt_commitment_sha256",
    )
    aggregate_binding: dict[str, object] | None = None
    row_count = 0
    next_frame_offset = 0
    rows = zip(
        _read_jsonl(raw_path),
        _read_jsonl(private_path),
        _read_jsonl(index_path),
        _read_jsonl(restored_path),
        strict=True,
    )
    try:
        with _private_binary_reader(frames_path) as frame_stream:
            for ordinal, (raw, private, index, restored) in enumerate(rows):
                if ordinal >= episodes_total:
                    raise Q1ExecutionError("consolidated artifacts exceed the frozen episode budget")
                master = ordinal // (len(ARM_ORDER) * lane_episodes)
                arm = ARM_ORDER[(ordinal // lane_episodes) % len(ARM_ORDER)]
                episode = ordinal % lane_episodes
                expected = (master, arm, episode)
                if (
                    (raw.get("master"), raw.get("arm"), raw.get("episode")) != expected
                    or (private.get("master"), private.get("arm_id"), private.get("episode")) != expected
                    or (index.get("master"), index.get("arm"), index.get("episode")) != expected
                    or (restored.get("master"), restored.get("arm"), restored.get("episode")) != expected
                ):
                    raise Q1ExecutionError("consolidated artifact semantic order mismatch")
                _validate_artifact_fast("raw_trace", raw)
                _validate_artifact_fast("private_audit", private)
                _validate_artifact_fast("checkpoint_frame", index)
                _validate_artifact_fast("restored_trace", restored)
                _validate_consolidated_binding(
                    raw=raw,
                    private=private,
                    index=index,
                    restored=restored,
                )
                if index.get("frame_offset") != next_frame_offset:
                    raise Q1ExecutionError("consolidated frame offsets are not contiguous")
                frame_length = index.get("frame_length")
                if type(frame_length) is not int or frame_length <= 0:
                    raise Q1ExecutionError("consolidated checkpoint frame length is invalid")
                payload = _read_checkpoint_frame_stream(
                    frame_stream,
                    frame_offset=next_frame_offset,
                    frame_length=frame_length,
                )
                if hashlib.sha256(payload).hexdigest() != index.get("checkpoint_sha256"):
                    raise Q1ExecutionError("consolidated checkpoint frame digest mismatch")
                next_frame_offset += _FRAME_LENGTH.size + frame_length
                key = (master, arm)
                terminal = _mapping(raw["terminal"], "terminal trace")
                episode_return = _number(terminal["episode_return"], "episode_return")
                totals[key] = math.fsum((totals[key], episode_return))
                counts[key] += 1
                if aggregate_binding is None:
                    aggregate_binding = {name: raw[name] for name in binding_fields}
                elif any(raw.get(name) != aggregate_binding[name] for name in binding_fields):
                    raise Q1ExecutionError("raw trace qualification binding changed within the run")
                row_count += 1
            if frame_stream.tell() != os.fstat(frame_stream.fileno()).st_size:
                raise Q1ExecutionError("consolidated checkpoint frames contain trailing bytes")
    except ValueError as error:
        raise Q1ExecutionError("consolidated artifact row counts differ") from error
    if row_count != episodes_total or aggregate_binding is None:
        raise Q1ExecutionError("consolidated artifacts differ from the exact episode budget")
    return _producer_aggregate(
        totals=totals,
        counts=counts,
        binding=aggregate_binding,
        execution_mode=execution_mode,
    )


def _validate_consolidated_binding(
    *,
    raw: Mapping[str, object],
    private: Mapping[str, object],
    index: Mapping[str, object],
    restored: Mapping[str, object],
) -> None:
    """Require exact run/attempt, qualification, checkpoint, and replay ancestry."""

    for field_name in (
        "protocol_version",
        "run_id",
        "attempt_id",
        "protocol_sha256",
        "implementation_sha256",
        "q0_report_sha256",
        "entry_qualification_sha256",
        "salt_commitment_sha256",
    ):
        if restored.get(field_name) != raw.get(field_name):
            raise Q1ExecutionError(f"restored trace binding mismatch: {field_name}")
    expected_private = {
        "run_id": raw.get("run_id"),
        "attempt_id": raw.get("attempt_id"),
        "salt_commitment_sha256": raw.get("salt_commitment_sha256"),
    }
    for field_name, expected in expected_private.items():
        if private.get(field_name) != expected:
            raise Q1ExecutionError(f"private audit binding mismatch: {field_name}")
    expected_index = {
        "protocol_version": raw.get("protocol_version"),
        "run_id": raw.get("run_id"),
        "attempt_id": raw.get("attempt_id"),
        "entry_qualification_sha256": raw.get("entry_qualification_sha256"),
    }
    for field_name, expected in expected_index.items():
        if index.get(field_name) != expected:
            raise Q1ExecutionError(f"checkpoint index binding mismatch: {field_name}")

    checkpoint = _mapping(raw["checkpoint"], "raw checkpoint")
    if (
        index.get("checkpoint_sha256") != checkpoint.get("sha256")
        or restored.get("checkpoint_sha256") != checkpoint.get("sha256")
        or index.get("component_sha256") != checkpoint.get("component_sha256")
        or restored.get("component_sha256") != checkpoint.get("component_sha256")
    ):
        raise Q1ExecutionError("checkpoint identity differs across consolidated artifacts")
    if restored.get("producer_pid") != raw.get("producer_pid"):
        raise Q1ExecutionError("restored trace producer PID differs from raw trace")

    acquisition = _mapping(raw["acquisition"], "raw acquisition")
    for restored_field, raw_field in (
        ("model_sha256", "model_after_sha256"),
        ("model_version", "model_after_version"),
        ("posterior_direct", "posterior_after"),
    ):
        if restored.get(restored_field) != acquisition.get(raw_field):
            raise Q1ExecutionError(f"restored model parity mismatch: {restored_field}")
    for restored_field, raw_field in (
        ("acquisition_experience_id", "experience_id"),
        ("acquisition_transition_id", "transition_id"),
        ("acquisition_receipt_id", "receipt_id"),
    ):
        if restored.get(restored_field) != acquisition.get(raw_field):
            raise Q1ExecutionError(f"restored acquisition ancestry mismatch: {restored_field}")
    terminal = _mapping(raw["terminal"], "raw terminal")
    for restored_field, raw_field in (
        ("terminal_candidate_rows", "candidate_rows"),
        ("terminal_candidate_rows_sha256", "candidate_rows_sha256"),
        ("selected_terminal_action", "selected_action"),
        ("terminal_success", "success"),
        ("episode_return", "episode_return"),
        ("terminal_decision_id", "decision_id"),
        ("terminal_intention_id", "intention_id"),
        ("terminal_execution_id", "execution_id"),
        ("terminal_observation_id", "observation_id"),
        ("terminal_outcome_id", "outcome_id"),
        ("terminal_experience_id", "experience_id"),
        ("terminal_transition_id", "transition_id"),
    ):
        if restored.get(restored_field) != terminal.get(raw_field):
            raise Q1ExecutionError(f"restored terminal replay mismatch: {restored_field}")


def _producer_aggregate(
    *,
    totals: Mapping[tuple[int, str], float],
    counts: Mapping[tuple[int, str], int],
    binding: Mapping[str, object],
    execution_mode: Q1ExecutionMode,
) -> dict[str, object]:
    """Compute a convenience summary; it is explicitly non-authoritative."""

    lane_episodes = episodes_per_master(execution_mode)
    episodes_total = MASTER_COUNT * len(ARM_ORDER) * lane_episodes
    arm_means: list[dict[str, object]] = []
    means: dict[tuple[int, str], float] = {}
    for master in range(MASTER_COUNT):
        for arm in ARM_ORDER:
            if counts[(master, arm)] != lane_episodes:
                raise Q1ExecutionError("producer aggregate saw an incomplete master/arm lane")
            mean = totals[(master, arm)] / lane_episodes
            means[(master, arm)] = mean
            arm_means.append(
                {
                    "master": master,
                    "arm": arm,
                    "episode_count": lane_episodes,
                    "mean_return": mean,
                }
            )
    comparisons: list[dict[str, object]] = []
    for control in _CONTROL_ARMS:
        differences = [
            means[(master, ArmMode.PROSPECT.value)] - means[(master, control)] for master in range(MASTER_COUNT)
        ]
        mean = math.fsum(differences) / MASTER_COUNT
        variance = math.fsum((value - mean) ** 2 for value in differences) / (MASTER_COUNT - 1)
        half_width = _T_CRITICAL_DF3 * math.sqrt(variance) / math.sqrt(MASTER_COUNT)
        comparisons.append(
            {
                "control_arm": control,
                "master_differences": differences,
                "mean_difference": mean,
                "ci95_lower": mean - half_width,
                "ci95_upper": mean + half_width,
            }
        )
    return {
        "schema": (
            AGGREGATE_SCHEMA if execution_mode is Q1ExecutionMode.PRODUCTION else REHEARSAL_AGGREGATE_SCHEMA
        ),
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding["run_id"],
        "attempt_id": binding["attempt_id"],
        "protocol_sha256": binding["protocol_sha256"],
        "implementation_sha256": binding["implementation_sha256"],
        "q0_report_sha256": binding["q0_report_sha256"],
        "entry_qualification_sha256": binding["entry_qualification_sha256"],
        "salt_commitment_sha256": binding["salt_commitment_sha256"],
        "claim_eligible": False,
        "formal_authorized": False,
        "producer_analysis_authoritative": False,
        "counts": {
            "masters": MASTER_COUNT,
            "arms": len(ARM_ORDER),
            "episodes": episodes_total,
            "environment_steps": 2 * episodes_total,
            "transitions": 2 * episodes_total,
            "acquisition_updates": episodes_total,
            "terminal_updates": 0,
            "checkpoints": episodes_total,
            "restores": episodes_total,
        },
        "arm_means": arm_means,
        "comparisons": comparisons,
    }


def _rehearsal_aggregate_schema_document() -> dict[str, object]:
    """Build the rehearsal aggregate schema from the frozen production schema.

    The rehearsal budget is a different exact contract, not a relaxation: every
    production constant is replaced by its rehearsal counterpart and the schema
    identifier changes, so a rehearsal aggregate can never satisfy the frozen
    Q1 aggregate schema the independent auditor requires.
    """

    lane_episodes = episodes_per_master(Q1ExecutionMode.REHEARSAL)
    episodes_total = MASTER_COUNT * len(ARM_ORDER) * lane_episodes
    document = copy.deepcopy(dict(schema_documents()["aggregate"]))
    properties = cast(dict[str, Any], document["properties"])
    properties["schema"] = {"const": REHEARSAL_AGGREGATE_SCHEMA}
    counts = cast(dict[str, Any], properties["counts"])["properties"]
    for name, value in (
        ("episodes", episodes_total),
        ("environment_steps", 2 * episodes_total),
        ("transitions", 2 * episodes_total),
        ("acquisition_updates", episodes_total),
        ("checkpoints", episodes_total),
        ("restores", episodes_total),
    ):
        counts[name] = {"const": value}
    arm_means = cast(dict[str, Any], properties["arm_means"])
    cast(dict[str, Any], arm_means["items"])["properties"]["episode_count"] = {"const": lane_episodes}
    return document


@cache
def _rehearsal_aggregate_validator() -> Any:
    from jsonschema import Draft202012Validator

    document = _rehearsal_aggregate_schema_document()
    Draft202012Validator.check_schema(document)
    return Draft202012Validator(document)


def _validate_aggregate(value: Mapping[str, object], *, execution_mode: Q1ExecutionMode) -> None:
    """Validate one aggregate against the exact schema its mode requires."""

    if execution_mode is Q1ExecutionMode.PRODUCTION:
        _validate_artifact_fast("aggregate", value)
        return
    errors = sorted(_rehearsal_aggregate_validator().iter_errors(value), key=lambda row: list(row.path))
    if errors:
        location = "/".join(str(part) for part in errors[0].absolute_path) or "<root>"
        raise Q1ExecutionError(f"rehearsal aggregate violates its schema at {location}")


def _remove_worker_trees_and_require_exact_artifacts(
    *,
    root: Path,
    canonical_paths: Sequence[Path],
) -> None:
    """Remove worker trees only after binding the exact private artifacts."""

    if tuple(path.name for path in canonical_paths) != _CANONICAL_ARTIFACT_NAMES:
        raise Q1ExecutionError("publication paths differ from the exact six artifacts")
    if any(path.parent != root for path in canonical_paths):
        raise Q1ExecutionError("publication artifact lies outside the incomplete directory")
    root_identity = _require_private_directory(
        root,
        label="incomplete Q1 directory",
    )
    canonical_names = set(_CANONICAL_ARTIFACT_NAMES)
    worker_names = {f"master-{master}" for master in range(MASTER_COUNT)}
    present = {path.name for path in root.iterdir()}
    if present != canonical_names | worker_names:
        raise Q1ExecutionError("incomplete directory contains an undeclared artifact")
    artifact_identities = {path.name: _require_private_regular_path(path) for path in canonical_paths}
    worker_identities = {
        worker_name: _require_private_directory(
            root / worker_name,
            label="internal worker directory",
        )
        for worker_name in sorted(worker_names)
    }
    for worker_name, worker_identity in worker_identities.items():
        _require_private_directory(
            root / worker_name,
            expected_identity=worker_identity,
            label="internal worker directory",
        )
    for worker_name in sorted(worker_names):
        shutil.rmtree(root / worker_name)
    _require_private_directory(
        root,
        expected_identity=root_identity,
        label="incomplete Q1 directory",
    )
    _require_exact_private_artifact_set(
        canonical_paths,
        directory=root,
        expected_identities=artifact_identities,
    )


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one bound private directory without replacement."""

    if os.name != "posix":
        raise Q1ExecutionError("atomic no-replace directory publication is unavailable")
    if source.name in ("", ".", "..") or destination.name in ("", ".", ".."):
        raise Q1ExecutionError("atomic publication path is invalid")
    source_descriptor, source_identity = _open_private_directory_descriptor(
        source,
        label="publication source",
    )
    source_parent_descriptor, source_parent_identity = _open_private_directory_descriptor(
        source.parent,
        label="publication source parent",
    )
    destination_parent_descriptor, destination_parent_identity = _open_private_directory_descriptor(
        destination.parent,
        label="publication destination parent",
    )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = cast(Any, getattr(libc, "renameat2", None))
        if renameat2 is None:
            raise Q1ExecutionError("atomic no-replace directory publication is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            source_parent_descriptor,
            os.fsencode(source.name),
            destination_parent_descriptor,
            os.fsencode(destination.name),
            1,
        )
        if result == 0:
            _require_directory_descriptor_identity(
                source_descriptor,
                path=destination,
                expected_identity=source_identity,
                label="published Q1 directory",
            )
            try:
                os.stat(
                    source.name,
                    dir_fd=source_parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise Q1ExecutionError("publication source still exists after atomic move")
            _require_directory_descriptor_identity(
                source_parent_descriptor,
                path=source.parent,
                expected_identity=source_parent_identity,
                label="publication source parent",
            )
            _require_directory_descriptor_identity(
                destination_parent_descriptor,
                path=destination.parent,
                expected_identity=destination_parent_identity,
                label="publication destination parent",
            )
            return
        error_number = ctypes.get_errno()
        _require_directory_descriptor_identity(
            source_descriptor,
            path=source,
            expected_identity=source_identity,
            label="publication source",
        )
        if error_number == errno.EEXIST:
            raise Q1ExecutionError("Q1 output destination appeared before atomic publication")
        operating_error = OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
        raise Q1ExecutionError("atomic no-replace directory publication failed") from operating_error
    finally:
        os.close(source_descriptor)
        os.close(source_parent_descriptor)
        os.close(destination_parent_descriptor)


def probe_atomic_noreplace_publication(execution_root: Path) -> None:
    """Exercise no-replace directory publication on the execution filesystem."""

    execution_root_identity = _require_private_directory(
        execution_root,
        label="atomic publication probe execution root",
    )
    probe_root = Path(
        tempfile.mkdtemp(
            prefix=".wm002-q1-publication-probe-",
            dir=execution_root,
        )
    )
    _force_private_directory_exact(probe_root, label="atomic publication probe root")
    try:
        source = probe_root / "source"
        destination = probe_root / "destination"
        _mkdir_private_exact(source, label="atomic publication probe source")
        with _private_binary_writer(source / "source-sentinel") as stream:
            stream.write(b"source")
        _publish_directory_noreplace(source, destination)
        if source.exists() or not (destination / "source-sentinel").is_file():
            raise Q1ExecutionError("atomic publication probe did not move the source exactly")

        collision_source = probe_root / "collision-source"
        _mkdir_private_exact(collision_source, label="atomic publication collision source")
        with _private_binary_writer(collision_source / "collision-sentinel") as stream:
            stream.write(b"collision")
        try:
            _publish_directory_noreplace(collision_source, destination)
        except Q1ExecutionError as error:
            if "appeared before atomic publication" not in str(error):
                raise
        else:
            raise Q1ExecutionError("atomic no-replace publication accepted an existing destination")
        if not (collision_source / "collision-sentinel").is_file() or not (destination / "source-sentinel").is_file():
            raise Q1ExecutionError("atomic no-replace collision changed source or destination")
    finally:
        shutil.rmtree(probe_root)
        _require_private_directory(
            execution_root,
            expected_identity=execution_root_identity,
            label="atomic publication probe execution root",
        )


def _finalize_failed_q1_attempt(
    *,
    error: BaseException,
    marker_path: Path,
    authority: Q1ExecutionAuthority,
    output_directory: Path,
    incomplete_directory: Path,
    expected_worker_capability_sha256: str,
) -> None:
    """Preserve incomplete state and finalize a still-started claimed attempt."""

    if isinstance(error, Q1ChildQuiescenceError):
        error.add_note(
            "child quiescence is unproven; attempt marker remains started and incomplete output is left untouched"
        )
        return
    notes: list[str] = []
    try:
        _revalidate_q1_runtime_directories(authority)
    except BaseException as identity_error:
        error.add_note(
            "runtime directory identity cannot be revalidated; attempt marker remains started "
            f"and output is left untouched: {type(identity_error).__name__}: {identity_error}"
        )
        return
    identity = authority.run_identity
    expected_marker_path = attempt_marker_path(
        authority.attempt_registry_directory,
        identity,
    )
    if marker_path != expected_marker_path:
        error.add_note("claimed marker path differs from its entry-bound registry; attempt remains started")
        return
    try:
        marker = load_attempt_marker(
            marker_path,
            expected_identity=identity,
            expected_worker_capability_sha256=expected_worker_capability_sha256,
        )
    except BaseException as marker_error:
        notes.append(f"failed to reload claimed attempt marker: {type(marker_error).__name__}: {marker_error}")
    else:
        if marker.status == "started":
            try:
                if _path_lexists(output_directory):
                    if _path_lexists(incomplete_directory):
                        raise Q1ExecutionError("cannot preserve final output because incomplete path also exists")
                    _publish_directory_noreplace(
                        output_directory,
                        incomplete_directory,
                    )
                    _fsync_directory(output_directory.parent)
                elif not _path_lexists(incomplete_directory):
                    _mkdir_private_exact(
                        incomplete_directory,
                        label="failed incomplete Q1 directory",
                    )
                    _fsync_directory(incomplete_directory.parent)
            except BaseException as preserve_error:
                notes.append(f"failed to preserve incomplete output: {type(preserve_error).__name__}: {preserve_error}")
            try:
                finalize_attempt(
                    marker_path,
                    identity,
                    status="failed",
                    expected_worker_capability_sha256=expected_worker_capability_sha256,
                )
            except BaseException as finalize_error:
                notes.append(f"failed to finalize attempt marker: {type(finalize_error).__name__}: {finalize_error}")
        elif marker.status != "failed":
            notes.append(f"claimed attempt marker unexpectedly has status {marker.status!r}")
    for note in notes:
        error.add_note(note)


def _fsync_publication_set(
    paths: Sequence[Path],
    *,
    directory: Path,
    expected_identities: Mapping[str, _FilesystemIdentity] | None = None,
) -> dict[str, _FilesystemIdentity]:
    """Validate and durably flush the exact private publication set."""

    if tuple(path.name for path in paths) != _CANONICAL_ARTIFACT_NAMES:
        raise Q1ExecutionError("fsync publication set differs from the exact six artifacts")
    identities = _require_exact_private_artifact_set(
        paths,
        directory=directory,
        expected_identities=expected_identities,
    )
    for path in paths:
        with _private_binary_reader(path) as stream:
            os.fsync(stream.fileno())
    identities = _require_exact_private_artifact_set(
        paths,
        directory=directory,
        expected_identities=identities,
    )
    _fsync_directory(directory)
    return _require_exact_private_artifact_set(
        paths,
        directory=directory,
        expected_identities=identities,
    )


def _fsync_directory(path: Path) -> None:
    descriptor, identity = _open_private_directory_descriptor(
        path,
        label="fsync directory",
    )
    try:
        os.fsync(descriptor)
        _require_directory_descriptor_identity(
            descriptor,
            path=path,
            expected_identity=identity,
            label="fsync directory",
        )
    finally:
        os.close(descriptor)


def _require_started_child_attempt(
    *,
    attempt_registry_directory: Path,
    expected_run_identity: Q1RunIdentity,
    expected_worker_capability_sha256: str,
) -> None:
    identity = expected_run_identity
    marker_path = attempt_marker_path(attempt_registry_directory, identity)
    try:
        marker = load_attempt_marker(
            marker_path,
            expected_identity=identity,
            expected_worker_capability_sha256=expected_worker_capability_sha256,
        )
    except Exception as error:
        raise Q1ExecutionError("child cannot load the exact durable attempt marker") from error
    if marker.status != "started":
        raise Q1ExecutionError("child requires the exact attempt marker in started state")


def _validate_q1_execution_root(*, execution_root: Path, output_directory: Path) -> None:
    identity = _require_private_directory(
        execution_root,
        label="execution root",
    )
    if output_directory.parent.resolve() != execution_root.resolve():
        raise Q1ExecutionError("Q1 output directory must be a direct child of the execution root")
    _require_private_directory(
        execution_root,
        expected_identity=identity,
        label="execution root",
    )


def _fresh_process_synthetic_restore(
    *,
    artifact: EpisodeArtifacts,
    binding: QualificationBinding,
    timeout_seconds: float = _DEVELOPMENT_PROBE_TIMEOUT_SECONDS,
) -> Mapping[str, object]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise Q1ExecutionError("synthetic restore timeout must be finite and positive")
    with tempfile.TemporaryDirectory(prefix="wm002-development-restore-") as directory:
        root = Path(directory)
        _force_private_directory_exact(root, label="development restore directory")
        frame_path = root / "frame.bin"
        with _private_binary_writer(frame_path) as stream:
            location = append_checkpoint_frame(stream, artifact.checkpoint_payload)
        raw_path = root / "raw.jsonl"
        with _private_binary_writer(raw_path) as stream:
            stream.write(canonical_json_bytes(artifact.raw_trace, newline=True))
        spec = {
            "schema": "prospect.wm002.q1-development-restore-input.v1",
            "frame_path": str(frame_path),
            "frame_offset": location.frame_offset,
            "frame_length": location.frame_length,
            "checkpoint_index": {
                **artifact.checkpoint_index,
                "frame_offset": location.frame_offset,
                "frame_length": location.frame_length,
            },
            "raw_trace": artifact.raw_trace,
            "binding": binding.as_dict(),
            "synthetic_terminal_success": _mapping(artifact.raw_trace["terminal"], "terminal")["success"],
        }
        spec_path = root / "spec.json"
        with _private_binary_writer(spec_path) as stream:
            stream.write(canonical_json_bytes(spec, newline=True))
        output_path = root / "restored.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "bench.active_acquisition.restore_worker",
                "development",
                "--spec-path",
                str(spec_path),
                "--output-path",
                str(output_path),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=_q1_child_environment(),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise Q1ExecutionError(f"synthetic restore worker failed: {detail}")
        return _mapping(
            load_canonical_json(output_path),
            "synthetic restored trace",
        )


def _acquisition_candidate_rows(
    rows: Sequence[CandidateDiagnosticRow],
) -> list[dict[str, object]]:
    return [{"ordinal": ordinal, **asdict(row)} for ordinal, row in enumerate(rows)]


def _terminal_candidate_rows(
    interaction: InteractionResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal, assessment in enumerate(interaction.decision.alternatives):
        action = _mapping(assessment.action.parameters, "terminal action parameters").get("terminal_action")
        rows.append(
            {
                "ordinal": ordinal,
                "assessment_id": assessment.assessment_id,
                "terminal_action": action,
                "prediction_id": assessment.prediction.prediction_id,
                "expected_success": assessment.total_value,
                "unit": assessment.unit,
            }
        )
    return rows


def _terminal_candidate_digest(interaction: InteractionResult) -> str:
    return canonical_sha256(_terminal_candidate_rows(interaction))


def _selected_acquisition_action(interaction: InteractionResult) -> str:
    value = _mapping(
        interaction.decision.selected_assessment.action.parameters,
        "acquisition action parameters",
    ).get("semantic_action")
    if not isinstance(value, str):
        raise Q1ExecutionError("selected acquisition action has no semantic identity")
    return value


def _execution_id(interaction: InteractionResult) -> str:
    execution = interaction.experience.execution
    if execution is None:
        raise Q1ExecutionError("interaction experience has no execution")
    return execution.execution_id


def _exercise_synthetic_noninterference(
    *,
    artifact: EpisodeArtifacts,
    binding: QualificationBinding,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    salt_commitment_sha256: str,
    public_samples: Mapping[str, object],
) -> None:
    """Exercise causal acquisition isolation with explicit development values."""

    raw = _mapping(artifact.raw_trace, "synthetic noninterference baseline")
    acquisition = _mapping(raw["acquisition"], "synthetic noninterference acquisition")
    arm = ArmMode(str(raw["arm"]))
    if arm is ArmMode.UNIFORM_RANDOM:
        raise Q1ExecutionError("synthetic noninterference baseline must use a deterministic arm")
    master = cast(int, raw["master"])
    episode = cast(int, raw["episode"])
    reference_private = _synthetic_private_material(
        arm=arm,
        master=master,
        episode=episode,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    varied_private = _synthetic_private_material(
        arm=arm,
        master=master,
        episode=episode,
        salt_commitment_sha256=salt_commitment_sha256,
        private_variant="noninterference-variant",
        theta=-1,
    )
    reference_hmacs = reference_private.private_hmac_digests()
    varied_hmacs = varied_private.private_hmac_digests()
    if reference_private.theta == varied_private.theta or any(
        reference == varied for reference, varied in zip(reference_hmacs, varied_hmacs, strict=True)
    ):
        raise Q1ExecutionError("synthetic noninterference did not vary every private theta/HMAC field")

    def run_variant(
        private: PrivateEpisodeSeedMaterial,
        *,
        terminal_override: bool | None,
    ) -> EpisodeArtifacts:
        order: list[str] = []
        acquisition_potential, terminal_potential = _synthetic_private_potential_outcomes(private)
        selected_action = cast(str, acquisition["selected_action"])
        observed_symbol = acquisition_potential[selected_action]
        if observed_symbol != acquisition["observed_symbol"]:
            raise Q1ExecutionError("synthetic private potential tables changed the held-visible acquisition input")

        def observe_checkpoint(payload: bytes) -> None:
            if not payload:
                raise Q1ExecutionError("synthetic checkpoint observer received an empty payload")
            order.append("checkpoint")

        def terminal_callback(_decision: TerminalAction) -> bool:
            order.append("terminal_callback")
            if order != ["checkpoint", "terminal_callback"]:
                raise Q1ExecutionError("terminal callback ran before checkpoint production")
            return terminal_potential[int(_decision)] if terminal_override is None else terminal_override

        candidate = _run_live_episode(
            master=master,
            episode=episode,
            arm=arm,
            producer_pid=cast(int, raw["producer_pid"]),
            identity_next_counter=0,
            acquisition_observed_symbol=observed_symbol,
            terminal_success=terminal_callback,
            uniform_ordinal=None,
            uniform_selector_digest=None,
            semantic_key_sha256=cast(str, raw["semantic_key_sha256"]),
            binding=binding,
            protocol_sha256=protocol_sha256,
            implementation_sha256=implementation_sha256,
            q0_report_sha256=q0_report_sha256,
            salt_commitment_sha256=salt_commitment_sha256,
            entry_qualification_sha256=binding.entry_qualification_sha256,
            private_audit=private,
            expected_action=selected_action,
            public_privacy_schedule=None,
            exercise_transaction_rollback=True,
            checkpoint_observer=observe_checkpoint,
        )
        if order != ["checkpoint", "terminal_callback"]:
            raise Q1ExecutionError("synthetic checkpoint/terminal callback order was incomplete")
        return candidate

    private_variant = run_variant(varied_private, terminal_override=None)
    terminal_variant = run_variant(reference_private, terminal_override=True)
    _require_preterminal_bytes_equal(artifact, private_variant, label="private-material variation")
    _require_preterminal_bytes_equal(artifact, terminal_variant, label="terminal sensitivity control")

    reference_sidecar = canonical_json_bytes(artifact.private_audit, newline=True)
    if reference_sidecar == canonical_json_bytes(private_variant.private_audit, newline=True):
        raise Q1ExecutionError("private-material variation did not change the private sidecar")
    if reference_sidecar != canonical_json_bytes(terminal_variant.private_audit, newline=True):
        raise Q1ExecutionError("terminal sensitivity control changed fixed private material")

    reference_terminal = _mapping(raw["terminal"], "synthetic reference terminal")
    sensitive_terminal = _mapping(terminal_variant.raw_trace["terminal"], "synthetic sensitive terminal")
    if reference_terminal["success"] is not False or sensitive_terminal["success"] is not True:
        raise Q1ExecutionError("terminal sensitivity control did not vary executed success")
    for metric_field in ("terminal_reward", "episode_return"):
        if reference_terminal[metric_field] == sensitive_terminal[metric_field]:
            raise Q1ExecutionError(f"terminal sensitivity control did not vary {metric_field}")

    private_hmac_corpus = (*reference_hmacs, *varied_hmacs)
    public_violations = private_material_paths(
        (
            public_samples,
            artifact.raw_trace,
            artifact.checkpoint_index,
            private_variant.raw_trace,
            private_variant.checkpoint_index,
            terminal_variant.raw_trace,
            terminal_variant.checkpoint_index,
        ),
        private_bytes=private_hmac_corpus,
    )
    if public_violations:
        raise Q1ExecutionError(
            "synthetic public projections contain private HMAC material at " + ", ".join(public_violations)
        )
    _assert_checkpoint_hmac_corpora_absent(
        (
            artifact.checkpoint_payload,
            private_variant.checkpoint_payload,
            terminal_variant.checkpoint_payload,
        ),
        private_hmac_corpus,
    )


def _require_preterminal_bytes_equal(
    reference: EpisodeArtifacts,
    candidate: EpisodeArtifacts,
    *,
    label: str,
) -> None:
    assert_acquisition_public_noninterference(reference.acquisition_public_probe, candidate.acquisition_public_probe)
    if reference.checkpoint_payload != candidate.checkpoint_payload:
        raise Q1ExecutionError(f"{label} changed the preterminal checkpoint payload")
    if canonical_json_bytes(reference.checkpoint_index, newline=True) != canonical_json_bytes(
        candidate.checkpoint_index,
        newline=True,
    ):
        raise Q1ExecutionError(f"{label} changed the preterminal checkpoint index")
    if _acquisition_public_trace_projection(reference) != _acquisition_public_trace_projection(candidate):
        raise Q1ExecutionError(f"{label} changed the acquisition/public trace projection")


def _acquisition_public_trace_projection(artifact: EpisodeArtifacts) -> bytes:
    raw = artifact.raw_trace
    fields = (
        "schema",
        "protocol_version",
        "run_id",
        "attempt_id",
        "protocol_sha256",
        "implementation_sha256",
        "q0_report_sha256",
        "salt_commitment_sha256",
        "entry_qualification_sha256",
        "master",
        "arm",
        "episode",
        "producer_pid",
        "semantic_key_sha256",
        "candidate_rows",
        "candidate_rows_sha256",
        "acquisition",
        "checkpoint",
    )
    if any(field not in raw for field in fields):
        raise Q1ExecutionError("synthetic raw trace lacks the exact acquisition/public projection")
    return canonical_json_bytes({field: raw[field] for field in fields}, newline=True)


def _assert_checkpoint_hmac_corpora_absent(
    payloads: Sequence[bytes],
    private_hmac_corpus: Sequence[bytes],
) -> None:
    patterns: list[bytes] = [b'"theta"', b'"hmac_sha256"']
    for digest in private_hmac_corpus:
        for prefix_length in range(min(16, len(digest)), len(digest) + 1):
            prefix = digest[:prefix_length]
            patterns.extend(
                (
                    prefix,
                    prefix.hex().encode("ascii"),
                    prefix.hex().upper().encode("ascii"),
                    base64.b64encode(prefix),
                    base64.b64encode(prefix).rstrip(b"="),
                    base64.urlsafe_b64encode(prefix),
                    base64.urlsafe_b64encode(prefix).rstrip(b"="),
                )
            )
    for payload in payloads:
        if any(pattern and pattern in payload for pattern in patterns):
            raise Q1ExecutionError("synthetic checkpoint payload contains private HMAC material or field names")


def _development_resource_samples(
    *,
    live: Sequence[EpisodeArtifacts],
    restored_trace: Mapping[str, object],
    duration_under_30_seconds: bool,
) -> Mapping[str, int | bool]:
    """Return deterministic conservative maxima independent of live PID width."""

    def json_size(value: object) -> int:
        return len(canonical_json_bytes(value, newline=True)) + _RESOURCE_PADDING_BYTES

    raw_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for artifact in live:
        raw = dict(artifact.raw_trace)
        raw["producer_pid"] = _RESOURCE_MAX_PID
        raw_rows.append(raw)
        index = dict(artifact.checkpoint_index)
        index["frame_offset"] = _RESOURCE_MAX_FRAME_OFFSET
        index_rows.append(index)
    restored = dict(restored_trace)
    restored["producer_pid"] = _RESOURCE_MAX_PID
    restored["restorer_pid"] = _RESOURCE_MAX_PID
    samples: dict[str, int | bool] = {
        "raw_trace_max_bytes": max(json_size(row) for row in raw_rows),
        "private_audit_max_bytes": max(json_size(artifact.private_audit) for artifact in live),
        "checkpoint_index_max_bytes": max(json_size(row) for row in index_rows),
        "checkpoint_frame_max_bytes": max(_FRAME_LENGTH.size + len(artifact.checkpoint_payload) for artifact in live)
        + _RESOURCE_PADDING_BYTES,
        "restored_trace_max_bytes": json_size(restored),
        "probe_duration_under_30_seconds": duration_under_30_seconds,
    }
    return MappingProxyType(samples)


def _synthetic_private_material(
    *,
    arm: ArmMode,
    master: int,
    episode: int,
    salt_commitment_sha256: str,
    private_variant: str = "reference",
    theta: int = 1,
) -> PrivateEpisodeSeedMaterial:
    def digest(label: str, *, high: bool) -> str:
        payload = f"wm002-explicit-development:{arm.value}:{private_variant}:{label}"
        seed = int(hashlib.sha256(payload.encode("ascii")).hexdigest(), 16)
        low = seed % (1 << 240)
        value = (1 << 256) - 1 - low if high else low
        return f"{value:064x}"

    if not private_variant or theta not in {-1, 1}:
        raise Q1ExecutionError("synthetic private material variant is invalid")
    varied = private_variant != "reference"
    theta_order_value = int(digest("theta-order", high=False), 16)
    theta_order_value = (theta_order_value & ~1) | int(theta == -1)

    return PrivateEpisodeSeedMaterial(
        arm_id=arm.value,
        episode=episode,
        master=master,
        nuisance_hmac_sha256=digest("nuisance", high=varied),
        pulse_hmac_sha256=tuple(
            (action, digest(f"pulse:{action}", high=varied)) for action in ("weak", "strong", "overpowered")
        ),
        salt_commitment_sha256=salt_commitment_sha256,
        terminal_hmac_sha256=(
            ("+1", digest("terminal:+1", high=True)),
            ("-1", digest("terminal:-1", high=True)),
        ),
        theta=theta,
        theta_order_hmac_sha256=f"{theta_order_value:064x}",
    )


def _synthetic_private_potential_outcomes(
    material: PrivateEpisodeSeedMaterial,
) -> tuple[dict[str, int], dict[int, bool]]:
    """Map every explicit private HMAC/theta field into development potentials."""

    theta_from_order_hmac = -1 if int(material.theta_order_hmac_sha256, 16) & 1 else 1
    if theta_from_order_hmac != material.theta:
        raise Q1ExecutionError("synthetic theta differs from its development order-HMAC mapping")
    acquisition = {
        "skip": 0,
        "nuisance": nuisance_observation_from_digest(bytes.fromhex(material.nuisance_hmac_sha256)),
        **{
            action: pulse_observation_from_digest(bytes.fromhex(digest), action, material.theta)
            for action, digest in material.pulse_hmac_sha256
        },
    }
    terminal = {
        int(action): terminal_success_from_digest(bytes.fromhex(digest), int(action), material.theta)
        for action, digest in material.terminal_hmac_sha256
    }
    return acquisition, terminal


def _aggregate_schema_sample(
    *,
    binding: QualificationBinding,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    salt_commitment_sha256: str,
) -> dict[str, object]:
    arm_means = [
        {
            "master": master,
            "arm": arm,
            "episode_count": EPISODES_PER_MASTER,
            "mean_return": 0.0,
        }
        for master in range(MASTER_COUNT)
        for arm in ARM_ORDER
    ]
    comparisons = [
        {
            "control_arm": arm,
            "master_differences": [0.0] * MASTER_COUNT,
            "mean_difference": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
        }
        for arm in _CONTROL_ARMS
    ]
    return {
        "schema": AGGREGATE_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "q0_report_sha256": q0_report_sha256,
        "entry_qualification_sha256": hashlib.sha256(b"wm002-explicit-synthetic-development-entry").hexdigest(),
        "salt_commitment_sha256": salt_commitment_sha256,
        "claim_eligible": False,
        "formal_authorized": False,
        "producer_analysis_authoritative": False,
        "counts": {
            "masters": MASTER_COUNT,
            "arms": len(ARM_ORDER),
            "episodes": EPISODES_TOTAL,
            "environment_steps": ENVIRONMENT_STEPS_TOTAL,
            "transitions": ENVIRONMENT_STEPS_TOTAL,
            "acquisition_updates": EPISODES_TOTAL,
            "terminal_updates": 0,
            "checkpoints": EPISODES_TOTAL,
            "restores": EPISODES_TOTAL,
        },
        "arm_means": arm_means,
        "comparisons": comparisons,
    }


def _audit_schema_sample(
    *,
    binding: QualificationBinding,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    salt_commitment_sha256: str,
) -> dict[str, object]:
    placeholder = hashlib.sha256(b"wm002-development-artifact-placeholder").hexdigest()
    return {
        "schema": AUDIT_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "q0_report_sha256": q0_report_sha256,
        "entry_qualification_sha256": hashlib.sha256(b"wm002-explicit-synthetic-development-entry").hexdigest(),
        "salt_commitment_sha256": salt_commitment_sha256,
        "raw_trace_sha256": placeholder,
        "private_audit_sha256": placeholder,
        "checkpoint_index_sha256": placeholder,
        "checkpoint_frames_sha256": placeholder,
        "restored_trace_sha256": placeholder,
        "producer_aggregate_sha256": placeholder,
        "prospective_review_sha256": placeholder,
        "attempt_marker_sha256": placeholder,
        "worker_capability_sha256": placeholder,
        "claim_eligible": False,
        "formal_authorized": False,
        "independent_recomputation": True,
        "counts": {
            "acquisition_updates": 0,
            "arms": 0,
            "checkpoint_frames": 0,
            "environment_steps": 0,
            "episodes": 0,
            "masters": 0,
            "private_rows": 0,
            "raw_rows": 0,
            "restored_rows": 0,
            "terminal_updates": 0,
            "transitions": 0,
        },
        "comparisons": [
            {
                "control_arm": arm,
                "master_differences": [0.0] * MASTER_COUNT,
                "mean_difference": 0.0,
                "ci95_lower": 0.0,
                "ci95_upper": 0.0,
                "passed": False,
            }
            for arm in _CONTROL_ARMS
        ],
        "gates": [
            {"gate": f"Q1-K{index}", "passed": False, "violations": ["development sample"]} for index in range(6)
        ],
        "accepted_prefix": [],
        "passed": False,
        "verdict": "Schema-only synthetic development sample; no Q1 outcome.",
        "scope_limitations": ["This in-memory value is not a Q1 result."],
    }


def _restored_schema_sample(raw_trace: Mapping[str, object]) -> dict[str, object]:
    acquisition = _mapping(raw_trace["acquisition"], "acquisition")
    checkpoint = _mapping(raw_trace["checkpoint"], "checkpoint")
    terminal = _mapping(raw_trace["terminal"], "terminal")
    return {
        "schema": RESTORED_TRACE_SCHEMA,
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": raw_trace["run_id"],
        "attempt_id": raw_trace["attempt_id"],
        "protocol_sha256": raw_trace["protocol_sha256"],
        "implementation_sha256": raw_trace["implementation_sha256"],
        "q0_report_sha256": raw_trace["q0_report_sha256"],
        "entry_qualification_sha256": raw_trace["entry_qualification_sha256"],
        "salt_commitment_sha256": raw_trace["salt_commitment_sha256"],
        "master": raw_trace["master"],
        "arm": raw_trace["arm"],
        "episode": raw_trace["episode"],
        "producer_pid": raw_trace["producer_pid"],
        "restorer_pid": max(1, cast(int, raw_trace["producer_pid"]) + 1),
        "checkpoint_sha256": checkpoint["sha256"],
        "component_sha256": checkpoint["component_sha256"],
        "model_sha256": acquisition["model_after_sha256"],
        "model_version": acquisition["model_after_version"],
        "configuration_version": f"wm002-config-sha256:{acquisition['model_after_sha256']}",
        "posterior_direct": acquisition["posterior_after"],
        "acquisition_experience_id": acquisition["experience_id"],
        "acquisition_transition_id": acquisition["transition_id"],
        "acquisition_receipt_id": acquisition["receipt_id"],
        "identity_counter_sha256": _mapping(
            checkpoint["component_sha256"],
            "checkpoint components",
        )["identity_counter"],
        "terminal_candidate_rows": terminal["candidate_rows"],
        "terminal_candidate_rows_sha256": terminal["candidate_rows_sha256"],
        "selected_terminal_action": terminal["selected_action"],
        "terminal_success": terminal["success"],
        "episode_return": terminal["episode_return"],
        "terminal_decision_id": terminal["decision_id"],
        "terminal_intention_id": terminal["intention_id"],
        "terminal_execution_id": terminal["execution_id"],
        "terminal_observation_id": terminal["observation_id"],
        "terminal_outcome_id": terminal["outcome_id"],
        "terminal_experience_id": terminal["experience_id"],
        "terminal_transition_id": terminal["transition_id"],
    }


def _q1_child_environment() -> dict[str, str]:
    """Return a site-disabled repository-first import environment for Q1 children."""

    repository_root = Path(__file__).resolve().parents[2]
    import_roots = [repository_root, repository_root / "src"]
    for name in ("purelib", "platlib"):
        raw_path = sysconfig.get_path(name)
        if not raw_path:
            raise Q1ExecutionError(f"Python {name} dependency path is unavailable")
        path = Path(raw_path).resolve()
        if path.is_symlink() or not path.is_dir():
            raise Q1ExecutionError(f"Python {name} dependency path is not a regular directory")
        if path not in import_roots:
            import_roots.append(path)
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "TZ": "UTC",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in import_roots),
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
    }
    return environment


def _write_all_descriptor(
    descriptor: int,
    payload: bytes,
    *,
    process: subprocess.Popen[bytes],
    deadline: float | None = None,
) -> None:
    """Deliver one capability within a frozen deadline without blocking."""

    if deadline is None:
        deadline = time.monotonic() + WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS
    os.set_blocking(descriptor, False)
    view = memoryview(payload)
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_WRITE)
        while view:
            if process.poll() is not None:
                raise Q1ExecutionError("worker exited before capability delivery completed")
            try:
                written = os.write(descriptor, view)
            except BlockingIOError:
                written = 0
            if written > 0:
                view = view[written:]
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Q1ExecutionError(
                    f"worker capability delivery exceeded the frozen "
                    f"{WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS:g}-second timeout"
                )
            events = selector.select(timeout=remaining)
            if not events and time.monotonic() >= deadline:
                raise Q1ExecutionError(
                    f"worker capability delivery exceeded the frozen "
                    f"{WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS:g}-second timeout"
                )


def _read_capability_acknowledgement(
    descriptor: int,
    *,
    expected: bytes,
    process: subprocess.Popen[bytes],
    deadline: float,
) -> None:
    """Require one exact authenticated ACK and peer close before the deadline."""

    if len(expected) != WORKER_CAPABILITY_ACK_BYTES:
        raise Q1ExecutionError("worker capability acknowledgement length is invalid")
    os.set_blocking(descriptor, False)
    received = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            try:
                chunk = os.read(
                    descriptor,
                    WORKER_CAPABILITY_ACK_BYTES + 1 - len(received),
                )
            except BlockingIOError:
                chunk = None
            if chunk == b"":
                break
            if chunk:
                received.extend(chunk)
                if len(received) > WORKER_CAPABILITY_ACK_BYTES:
                    raise Q1ExecutionError("worker capability acknowledgement contains trailing bytes")
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Q1ExecutionError(
                    f"worker capability acknowledgement exceeded the frozen "
                    f"{WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS:g}-second timeout"
                )
            events = selector.select(timeout=remaining)
            if not events and time.monotonic() >= deadline:
                raise Q1ExecutionError(
                    f"worker capability acknowledgement exceeded the frozen "
                    f"{WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS:g}-second timeout"
                )
    if not hmac.compare_digest(bytes(received), expected):
        raise Q1ExecutionError("worker capability acknowledgement differs")


def _spawn_authenticated_worker(
    launch: AuthenticatedWorkerLaunch,
    *,
    secret: bytes,
) -> _AuthenticatedWorkerProcess:
    """Spawn one child and complete its PID-bound authenticated exchange."""

    if launch.parent_pid != os.getpid():
        raise Q1ExecutionError("worker launch parent PID differs from the spawning process")
    worker_capability_commitment(secret)
    parent_socket: socket.socket | None = None
    child_socket: socket.socket | None = None
    stdout_capture: _BoundedWorkerCapture | None = None
    stderr_capture: _BoundedWorkerCapture | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        socket_type = socket.SOCK_STREAM | getattr(socket, "SOCK_CLOEXEC", 0)
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket_type)
        stdout_capture = _BoundedWorkerCapture(
            label="stdout",
            max_bytes=WORKER_STDOUT_CAPTURE_MAX_BYTES,
        )
        stderr_capture = _BoundedWorkerCapture(
            label="stderr",
            max_bytes=WORKER_STDERR_CAPTURE_MAX_BYTES,
        )
        capability_descriptor = child_socket.fileno()
        command = (*launch.base_command, "--capability-fd", str(capability_descriptor))
        process = subprocess.Popen(
            list(command),
            cwd=Path(__file__).resolve().parents[2],
            env=_q1_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stdout_capture.write_descriptor,
            stderr=stderr_capture.write_descriptor,
            pass_fds=(capability_descriptor,),
        )
        launched_at = time.monotonic()
        child_socket.close()
        child_socket = None
        stdout_capture.start(process)
        stderr_capture.start(process)
        wire = encode_worker_capability(secret, launch.capability(process.pid))
        expected_acknowledgement = worker_capability_acknowledgement(wire)
        deadline = launched_at + WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS
        _write_all_descriptor(
            parent_socket.fileno(),
            wire,
            process=process,
            deadline=deadline,
        )
        parent_socket.shutdown(socket.SHUT_WR)
        _read_capability_acknowledgement(
            parent_socket.fileno(),
            expected=expected_acknowledgement,
            process=process,
            deadline=deadline,
        )
        parent_socket.close()
        parent_socket = None
        return _AuthenticatedWorkerProcess(
            launch=launch,
            process=process,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
            launched_at=launched_at,
        )
    except BaseException as error:
        if process is not None:
            try:
                _quiesce_after_failure(
                    ((launch.identity, process),),
                    label="authenticated worker startup",
                    cause=error,
                )
            finally:
                if stdout_capture is not None:
                    stdout_capture.close()
                if stderr_capture is not None:
                    stderr_capture.close()
        else:
            if stdout_capture is not None:
                stdout_capture.close()
            if stderr_capture is not None:
                stderr_capture.close()
        raise
    finally:
        if parent_socket is not None:
            parent_socket.close()
        if child_socket is not None:
            child_socket.close()


def _sanitize_stderr_tail(payload: bytes) -> str:
    decoded = payload.decode("utf-8", errors="replace")
    return "".join(character if " " <= character <= "~" else " " for character in decoded).strip()


def _close_authenticated_processes(processes: Iterable[_AuthenticatedWorkerProcess]) -> None:
    for process in processes:
        try:
            process.close_captures()
        except OSError:
            pass


def _wait_authenticated_worker(
    child: _AuthenticatedWorkerProcess,
    *,
    identity: object,
    label: str,
    timeout: float,
) -> None:
    returncode = child.process.wait(timeout=timeout)
    try:
        stdout_size, _stdout_tail, stdout_overflow = child.stdout_capture.snapshot()
        _stderr_size, stderr_tail, stderr_overflow = child.stderr_capture.snapshot()
    finally:
        child.close_captures()
    detail = _sanitize_stderr_tail(stderr_tail)
    if stdout_overflow:
        raise Q1ExecutionError(
            f"{label} failure: {identity}:stdout exceeded the frozen {WORKER_STDOUT_CAPTURE_MAX_BYTES}-byte bound"
        )
    if stderr_overflow:
        raise Q1ExecutionError(
            f"{label} failure: {identity}:stderr exceeded the frozen "
            f"{WORKER_STDERR_CAPTURE_MAX_BYTES}-byte bound:{detail}"
        )
    if returncode:
        raise Q1ExecutionError(f"{label} failure: {identity}:exit={returncode}:{detail}")
    if stdout_size:
        raise Q1ExecutionError(f"{label} failure: {identity}:unexpected stdout")


def _process_is_running(
    process: subprocess.Popen[bytes],
    *,
    identity: object,
    diagnostics: list[str],
) -> bool:
    """Return whether a child is live, recording any inability to inspect it."""

    try:
        return process.poll() is None
    except BaseException as error:
        diagnostics.append(f"{identity}:poll:{type(error).__name__}:{error}")
        return True


def _quiesce_processes(
    processes: Sequence[tuple[object, subprocess.Popen[bytes]]],
    *,
    label: str,
) -> None:
    """Terminate, kill if needed, and reap every still-live child."""

    diagnostics: list[str] = []
    for identity, process in processes:
        if not _process_is_running(
            process,
            identity=identity,
            diagnostics=diagnostics,
        ):
            continue
        try:
            process.terminate()
        except BaseException as error:
            diagnostics.append(f"{identity}:terminate:{type(error).__name__}:{error}")

    terminate_deadline = time.monotonic() + PROCESS_TERMINATE_GRACE_SECONDS
    for identity, process in processes:
        if not _process_is_running(
            process,
            identity=identity,
            diagnostics=diagnostics,
        ):
            continue
        remaining = terminate_deadline - time.monotonic()
        if remaining <= 0:
            continue
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            diagnostics.append(f"{identity}:terminate-wait:{type(error).__name__}:{error}")

    for identity, process in processes:
        if not _process_is_running(
            process,
            identity=identity,
            diagnostics=diagnostics,
        ):
            continue
        try:
            process.kill()
        except BaseException as error:
            diagnostics.append(f"{identity}:kill:{type(error).__name__}:{error}")

    kill_deadline = time.monotonic() + PROCESS_TERMINATE_GRACE_SECONDS
    for identity, process in processes:
        if not _process_is_running(
            process,
            identity=identity,
            diagnostics=diagnostics,
        ):
            continue
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            continue
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            diagnostics.append(f"{identity}:kill-wait:{type(error).__name__}:{error}")

    unproven: list[str] = []
    for identity, process in processes:
        if _process_is_running(
            process,
            identity=identity,
            diagnostics=diagnostics,
        ):
            unproven.append(str(identity))
    if unproven:
        detail = " | ".join(diagnostics[-20:])
        suffix = f"; diagnostics: {detail}" if detail else ""
        raise Q1ChildQuiescenceError(f"{label} child quiescence is unproven for {', '.join(unproven)}{suffix}")


def _quiesce_after_failure(
    processes: Sequence[tuple[object, subprocess.Popen[bytes]]],
    *,
    label: str,
    cause: BaseException,
) -> None:
    """Quiesce children or replace the cause with an explicit unproven state."""

    try:
        _quiesce_processes(processes, label=label)
    except Q1ChildQuiescenceError as quiescence_error:
        quiescence_error.add_note(f"parent-side cause: {type(cause).__name__}: {cause}")
        raise quiescence_error from cause
    except BaseException as cleanup_error:
        replacement_error = Q1ChildQuiescenceError(f"{label} cleanup failed before child quiescence could be proven")
        replacement_error.add_note(f"parent-side cause: {type(cause).__name__}: {cause}")
        replacement_error.add_note(f"cleanup cause: {type(cleanup_error).__name__}: {cleanup_error}")
        raise replacement_error from cleanup_error


def _wait_exact_processes(
    processes: Sequence[tuple[object, _AuthenticatedWorkerProcess]],
    label: str,
) -> None:
    """Wait for a fixed child set within the frozen producer-stage deadline."""

    deadline = time.monotonic() + PRODUCER_STAGE_TIMEOUT_SECONDS
    raw_processes = tuple((identity, child.process) for identity, child in processes)
    try:
        for identity, child in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    cmd=f"{label}:{identity}",
                    timeout=PRODUCER_STAGE_TIMEOUT_SECONDS,
                )
            _wait_authenticated_worker(
                child,
                identity=identity,
                label=label,
                timeout=remaining,
            )
    except subprocess.TimeoutExpired as timeout_error:
        error = Q1ExecutionError(f"{label} exceeded the frozen {PRODUCER_STAGE_TIMEOUT_SECONDS:g}-second stage timeout")
        _quiesce_after_failure(raw_processes, label=label, cause=error)
        raise error from timeout_error
    except BaseException as error:
        _quiesce_after_failure(raw_processes, label=label, cause=error)
        raise
    finally:
        _close_authenticated_processes(child for _identity, child in processes)


def _run_bounded_commands(
    launches: Sequence[tuple[str, AuthenticatedWorkerLaunch]],
    *,
    secret: bytes,
    label: str,
    max_concurrency: int,
) -> None:
    """Run each authenticated child once under per-child and stage watchdogs."""

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    worker_capability_commitment(secret)
    pending = iter(launches)
    active: list[tuple[str, _AuthenticatedWorkerProcess, float]] = []
    stage_deadline = time.monotonic() + RESTORE_STAGE_TIMEOUT_SECONDS

    def start_next() -> bool:
        try:
            identity, launch = next(pending)
        except StopIteration:
            return False
        child = _spawn_authenticated_worker(launch, secret=secret)
        child_deadline = child.launched_at + RESTORE_CHILD_TIMEOUT_SECONDS
        active.append((identity, child, child_deadline))
        return True

    def active_processes() -> tuple[tuple[str, subprocess.Popen[bytes]], ...]:
        return tuple((identity, child.process) for identity, child, _deadline in active)

    try:
        while len(active) < max_concurrency and start_next():
            pass
        while active:
            identity, child, child_deadline = active[0]
            now = time.monotonic()
            stage_remaining = stage_deadline - now
            child_remaining = child_deadline - now
            if stage_remaining <= 0:
                raise subprocess.TimeoutExpired(
                    cmd=f"{label}:{identity}",
                    timeout=RESTORE_STAGE_TIMEOUT_SECONDS,
                )
            if child_remaining <= 0:
                raise subprocess.TimeoutExpired(
                    cmd=f"{label}:{identity}",
                    timeout=RESTORE_CHILD_TIMEOUT_SECONDS,
                )
            _wait_authenticated_worker(
                child,
                identity=identity,
                label=label,
                timeout=min(child_remaining, stage_remaining),
            )
            active.pop(0)
            start_next()
    except subprocess.TimeoutExpired as timeout_error:
        error = Q1ExecutionError(
            f"{label} exceeded a frozen child or stage timeout "
            f"(child={RESTORE_CHILD_TIMEOUT_SECONDS:g}s, "
            f"stage={RESTORE_STAGE_TIMEOUT_SECONDS:g}s)"
        )
        _quiesce_after_failure(active_processes(), label=label, cause=error)
        raise error from timeout_error
    except BaseException as error:
        _quiesce_after_failure(active_processes(), label=label, cause=error)
        raise
    finally:
        _close_authenticated_processes(child for _identity, child, _deadline in active)


def _filesystem_identity(metadata: os.stat_result) -> _FilesystemIdentity:
    return _FilesystemIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        owner=metadata.st_uid,
        group=metadata.st_gid,
    )


def _runtime_file_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_private_directory_metadata(
    metadata: os.stat_result,
    *,
    label: str,
) -> _FilesystemIdentity:
    if not stat.S_ISDIR(metadata.st_mode):
        raise Q1ExecutionError(f"{label} must be a non-symlink directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise Q1ExecutionError(f"{label} permissions must be exactly 0700")
    if metadata.st_uid != os.geteuid():
        raise Q1ExecutionError(f"{label} must be owned by the effective user")
    return _filesystem_identity(metadata)


def _require_private_regular_metadata(
    metadata: os.stat_result,
    *,
    label: str,
) -> _FilesystemIdentity:
    if not stat.S_ISREG(metadata.st_mode):
        raise Q1ExecutionError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise Q1ExecutionError(f"{label} permissions must be exactly 0600")
    if metadata.st_nlink != 1:
        raise Q1ExecutionError(f"{label} must have exactly one hard link")
    if metadata.st_uid != os.geteuid():
        raise Q1ExecutionError(f"{label} must be owned by the effective user")
    return _filesystem_identity(metadata)


def _open_private_directory_descriptor(
    path: Path,
    *,
    label: str,
) -> tuple[int, _FilesystemIdentity]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Q1ExecutionError(f"{label} cannot be opened safely") from error
    try:
        identity = _require_private_directory_metadata(
            os.fstat(descriptor),
            label=label,
        )
        path_metadata = os.stat(path, follow_symlinks=False)
        path_identity = _require_private_directory_metadata(
            path_metadata,
            label=label,
        )
        if path_identity != identity:
            raise Q1ExecutionError(f"{label} path differs from its open descriptor")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


def _require_directory_descriptor_identity(
    descriptor: int,
    *,
    path: Path,
    expected_identity: _FilesystemIdentity,
    label: str,
) -> None:
    descriptor_identity = _require_private_directory_metadata(
        os.fstat(descriptor),
        label=label,
    )
    try:
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise Q1ExecutionError(f"{label} path cannot be revalidated") from error
    path_identity = _require_private_directory_metadata(path_metadata, label=label)
    if descriptor_identity != expected_identity or path_identity != expected_identity:
        raise Q1ExecutionError(f"{label} identity changed")


def _require_private_directory(
    path: Path,
    *,
    expected_identity: _FilesystemIdentity | None = None,
    label: str = "runtime directory",
) -> _FilesystemIdentity:
    descriptor, identity = _open_private_directory_descriptor(path, label=label)
    try:
        if expected_identity is not None and identity != expected_identity:
            raise Q1ExecutionError(f"{label} identity differs from its creation binding")
        _require_directory_descriptor_identity(
            descriptor,
            path=path,
            expected_identity=identity,
            label=label,
        )
        return identity
    finally:
        os.close(descriptor)


def _mkdir_private_exact(
    path: Path,
    *,
    label: str = "runtime directory",
) -> _FilesystemIdentity:
    if path.name in ("", ".", ".."):
        raise Q1ExecutionError(f"{label} path is invalid")
    parent_descriptor, parent_identity = _open_private_directory_descriptor(
        path.parent,
        label=f"{label} parent",
    )
    descriptor = -1
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent_descriptor)
        created_metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(created_metadata.st_mode) or created_metadata.st_uid != os.geteuid():
            raise Q1ExecutionError(f"{label} creation produced an invalid directory")
        os.chmod(
            path.name,
            0o700,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        os.fchmod(descriptor, 0o700)
        identity = _require_private_directory_metadata(
            os.fstat(descriptor),
            label=label,
        )
        path_metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _require_private_directory_metadata(path_metadata, label=label) != identity:
            raise Q1ExecutionError(f"{label} path differs from its created descriptor")
        _require_directory_descriptor_identity(
            parent_descriptor,
            path=path.parent,
            expected_identity=parent_identity,
            label=f"{label} parent",
        )
        return identity
    except OSError as error:
        raise Q1ExecutionError(f"{label} cannot be created safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _force_private_directory_exact(
    path: Path,
    *,
    label: str,
) -> _FilesystemIdentity:
    if path.name in ("", ".", ".."):
        raise Q1ExecutionError(f"{label} path is invalid")
    parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(path.parent, parent_flags)
    except OSError as error:
        raise Q1ExecutionError(f"{label} parent cannot be opened safely") from error
    try:
        parent_metadata = os.fstat(parent_descriptor)
        parent_path_metadata = os.stat(path.parent, follow_symlinks=False)
        if not stat.S_ISDIR(parent_metadata.st_mode) or not stat.S_ISDIR(parent_path_metadata.st_mode):
            raise Q1ExecutionError(f"{label} parent must be a non-symlink directory")
        parent_identity = _filesystem_identity(parent_metadata)
        if _filesystem_identity(parent_path_metadata) != parent_identity:
            raise Q1ExecutionError(f"{label} parent path differs from its descriptor")
        metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise Q1ExecutionError(f"{label} must be an owned non-symlink directory")
        os.chmod(
            path.name,
            0o700,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        parent_after = os.fstat(parent_descriptor)
        parent_path_after = os.stat(path.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_after.st_mode)
            or not stat.S_ISDIR(parent_path_after.st_mode)
            or _filesystem_identity(parent_after) != parent_identity
            or _filesystem_identity(parent_path_after) != parent_identity
        ):
            raise Q1ExecutionError(f"{label} parent identity changed")
    finally:
        os.close(parent_descriptor)
    descriptor, _identity = _open_private_directory_descriptor(path, label=label)
    try:
        os.fchmod(descriptor, 0o700)
        identity = _require_private_directory_metadata(os.fstat(descriptor), label=label)
        _require_directory_descriptor_identity(
            descriptor,
            path=path,
            expected_identity=identity,
            label=label,
        )
        return identity
    finally:
        os.close(descriptor)


def _open_private_regular_descriptor(
    path: Path,
    *,
    flags: int,
    label: str,
    create: bool,
) -> tuple[int, int, _FilesystemIdentity, _FilesystemIdentity]:
    if path.name in ("", ".", ".."):
        raise Q1ExecutionError(f"{label} path is invalid")
    parent_descriptor, parent_identity = _open_private_directory_descriptor(
        path.parent,
        label=f"{label} parent",
    )
    descriptor = -1
    try:
        open_flags = flags | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        if create:
            open_flags |= os.O_CREAT | os.O_EXCL
            descriptor = os.open(
                path.name,
                open_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            os.fchmod(descriptor, 0o600)
        else:
            descriptor = os.open(path.name, open_flags, dir_fd=parent_descriptor)
        identity = _require_private_regular_metadata(os.fstat(descriptor), label=label)
        path_metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _require_private_regular_metadata(path_metadata, label=label) != identity:
            raise Q1ExecutionError(f"{label} path differs from its open descriptor")
        return descriptor, parent_descriptor, identity, parent_identity
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise Q1ExecutionError(f"{label} cannot be opened safely") from error
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise


def _revalidate_private_regular_descriptor(
    descriptor: int,
    *,
    path: Path,
    parent_descriptor: int,
    expected_identity: _FilesystemIdentity,
    expected_parent_identity: _FilesystemIdentity,
    label: str,
    expected_signature: tuple[int, ...] | None = None,
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    identity = _require_private_regular_metadata(metadata, label=label)
    try:
        path_metadata = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise Q1ExecutionError(f"{label} path cannot be revalidated") from error
    path_identity = _require_private_regular_metadata(path_metadata, label=label)
    if identity != expected_identity or path_identity != expected_identity:
        raise Q1ExecutionError(f"{label} identity changed")
    if expected_signature is not None and _runtime_file_signature(metadata) != expected_signature:
        raise Q1ExecutionError(f"{label} changed during its descriptor read")
    _require_directory_descriptor_identity(
        parent_descriptor,
        path=path.parent,
        expected_identity=expected_parent_identity,
        label=f"{label} parent",
    )
    return metadata


@contextmanager
def _private_binary_reader(path: Path) -> Iterator[BinaryIO]:
    label = f"runtime file {path.name!r}"
    descriptor, parent_descriptor, identity, parent_identity = _open_private_regular_descriptor(
        path,
        flags=os.O_RDONLY,
        label=label,
        create=False,
    )
    initial_signature = _runtime_file_signature(os.fstat(descriptor))
    stream: BinaryIO | None = None
    try:
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        try:
            yield stream
        finally:
            if not stream.closed:
                _revalidate_private_regular_descriptor(
                    stream.fileno(),
                    path=path,
                    parent_descriptor=parent_descriptor,
                    expected_identity=identity,
                    expected_parent_identity=parent_identity,
                    expected_signature=initial_signature,
                    label=label,
                )
    finally:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


@contextmanager
def _private_binary_writer(path: Path) -> Iterator[BinaryIO]:
    label = f"runtime file {path.name!r}"
    descriptor, parent_descriptor, identity, parent_identity = _open_private_regular_descriptor(
        path,
        flags=os.O_WRONLY,
        label=label,
        create=True,
    )
    stream: BinaryIO | None = None
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        try:
            yield stream
        finally:
            if not stream.closed:
                stream.flush()
                _revalidate_private_regular_descriptor(
                    stream.fileno(),
                    path=path,
                    parent_descriptor=parent_descriptor,
                    expected_identity=identity,
                    expected_parent_identity=parent_identity,
                    label=label,
                )
    finally:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def _require_private_regular_path(
    path: Path,
    *,
    expected_identity: _FilesystemIdentity | None = None,
) -> _FilesystemIdentity:
    label = f"runtime file {path.name!r}"
    descriptor, parent_descriptor, identity, parent_identity = _open_private_regular_descriptor(
        path,
        flags=os.O_RDONLY,
        label=label,
        create=False,
    )
    try:
        if expected_identity is not None and identity != expected_identity:
            raise Q1ExecutionError(f"{label} identity differs from its publication binding")
        _revalidate_private_regular_descriptor(
            descriptor,
            path=path,
            parent_descriptor=parent_descriptor,
            expected_identity=identity,
            expected_parent_identity=parent_identity,
            expected_signature=_runtime_file_signature(os.fstat(descriptor)),
            label=label,
        )
        return identity
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _require_exact_private_artifact_set(
    paths: Sequence[Path],
    *,
    directory: Path,
    expected_identities: Mapping[str, _FilesystemIdentity] | None = None,
) -> dict[str, _FilesystemIdentity]:
    if tuple(path.name for path in paths) != _CANONICAL_ARTIFACT_NAMES:
        raise Q1ExecutionError("publication paths differ from the exact six artifacts")
    if any(path.parent != directory for path in paths):
        raise Q1ExecutionError("publication artifact lies outside its bound directory")
    directory_identity = _require_private_directory(
        directory,
        label="publication directory",
    )
    if {path.name for path in directory.iterdir()} != set(_CANONICAL_ARTIFACT_NAMES):
        raise Q1ExecutionError("publication directory is not the exact six-artifact set")
    identities: dict[str, _FilesystemIdentity] = {}
    for path in paths:
        expected = None if expected_identities is None else expected_identities.get(path.name)
        if expected_identities is not None and expected is None:
            raise Q1ExecutionError("publication identity binding is incomplete")
        identities[path.name] = _require_private_regular_path(
            path,
            expected_identity=expected,
        )
    _require_private_directory(
        directory,
        expected_identity=directory_identity,
        label="publication directory",
    )
    return identities


def _read_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with _private_binary_reader(path) as stream:
        line_number = 0
        while True:
            line = stream.readline(_MAX_JSONL_ROW_BYTES + 1)
            if not line:
                return
            line_number += 1
            if len(line) > _MAX_JSONL_ROW_BYTES:
                raise Q1ExecutionError(f"{path}:{line_number} exceeds the {_MAX_JSONL_ROW_BYTES}-byte row bound")
            if not line.endswith(b"\n"):
                raise Q1ExecutionError(f"{path}:{line_number} lacks a newline")
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise Q1ExecutionError(f"{path}:{line_number} is not JSON") from error
            if line != canonical_json_bytes(value, newline=True):
                raise Q1ExecutionError(f"{path}:{line_number} is not canonical JSON")
            yield _mapping(value, f"{path}:{line_number}")


@lru_cache(maxsize=len(ARM_ORDER))
def _artifact_validator(name: str) -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise Q1ExecutionError("Q1 artifact validation requires jsonschema") from error
    schemas = schema_documents()
    if name not in schemas:
        raise Q1ExecutionError(f"unknown Q1 artifact schema {name!r}")
    return Draft202012Validator(schemas[name])


def _validate_artifact_fast(name: str, value: object) -> None:
    validator = _artifact_validator(name)
    errors = sorted(validator.iter_errors(value), key=lambda row: list(row.path))
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path)
        raise Q1ExecutionError(f"{name} artifact violates schema at {location or '<root>'}: {first.message}")


def _copy_bytes(path: Path, output: BinaryIO) -> None:
    with _private_binary_reader(path) as source:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Q1ExecutionError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Q1ExecutionError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise Q1ExecutionError(f"{label} must be finite")
    return numeric


def _fraction_string(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SHA256_CHARS for character in value):
        raise Q1ExecutionError(f"{label} must be lowercase SHA-256 hexadecimal")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--entry-report", type=Path, required=True)
    run.add_argument("--q0-report", type=Path, required=True)
    run.add_argument("--prospective-review", type=Path, required=True)
    run.add_argument("--secret-salt", type=Path, required=True)
    run.add_argument("--execution-root", type=Path, required=True)
    run.add_argument("--attempt-registry", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    worker = subparsers.add_parser("_producer-master")
    worker.add_argument("--capability-fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_q1(
            entry_report_path=args.entry_report,
            q0_report_path=args.q0_report,
            prospective_review_path=args.prospective_review,
            secret_salt_path=args.secret_salt,
            execution_root=args.execution_root,
            attempt_registry_directory=args.attempt_registry,
            output_directory=args.output_directory,
        )
        return 0
    decoded = consume_worker_capability_fd(
        args.capability_fd,
        expected_parent_pid=os.getppid(),
        expected_child_pid=os.getpid(),
    )
    _run_master_worker(
        capability=decoded.capability,
        worker_capability_sha256=decoded.worker_capability_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AuthenticatedWorkerLaunch",
    "CheckpointFrameLocation",
    "DevelopmentQualificationProbe",
    "EpisodeArtifacts",
    "Q1ChildQuiescenceError",
    "Q1ExecutionAuthority",
    "Q1ExecutionError",
    "Q1RunOutputs",
    "WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS",
    "WORKER_STDERR_CAPTURE_MAX_BYTES",
    "WORKER_STDOUT_CAPTURE_MAX_BYTES",
    "append_checkpoint_frame",
    "probe_atomic_noreplace_publication",
    "read_checkpoint_frame",
    "run_development_qualification_probe",
    "run_q1",
    "validate_q1_child_execution_authority",
    "validate_q1_execution_authority",
)
