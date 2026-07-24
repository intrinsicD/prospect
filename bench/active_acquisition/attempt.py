"""Durable experiment-global single-attempt registry for WM-002 Q1.

Every run binding claims the same fixed tombstone while retaining its complete
identity inside that marker.  It is a local study-integrity control, not
external attestation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from bench.active_acquisition.contracts import canonical_json_bytes, canonical_sha256

ATTEMPT_MARKER_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-attempt-marker.v2"
ATTEMPT_MARKER_FILENAME: Final = "wm002-q1.attempt.json"
RUN_IDENTITY_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-run-identity.v1"
_EXPECTED_ARTIFACT_NAMES: Final = (
    "aggregate",
    "checkpoint_frames",
    "checkpoint_index",
    "private_audit",
    "raw_trace",
    "restored_trace",
)
_EXPECTED_COUNTS: Final = {
    "acquisition_updates": 28_672,
    "checkpoints": 28_672,
    "environment_steps": 57_344,
    "episodes": 28_672,
    "restores": 28_672,
    "terminal_updates": 0,
    "transitions": 57_344,
}
_SHA256_CHARS: Final = frozenset("0123456789abcdef")


class AttemptRegistryError(RuntimeError):
    """The local single-attempt contract was violated."""


@dataclass(frozen=True, slots=True)
class Q1RunIdentity:
    """Canonical result-bearing run and sole-attempt identity."""

    protocol_version: str
    protocol_sha256: str
    implementation_sha256: str
    q0_report_sha256: str
    entry_qualification_sha256: str
    salt_commitment_sha256: str
    run_sha256: str
    run_id: str
    attempt_id: str
    schema: str = RUN_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if not self.protocol_version.strip():
            raise AttemptRegistryError("protocol_version must be nonempty")
        for name in (
            "protocol_sha256",
            "implementation_sha256",
            "q0_report_sha256",
            "entry_qualification_sha256",
            "salt_commitment_sha256",
            "run_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.run_id != f"wm002-q1-{self.run_sha256}":
            raise AttemptRegistryError("run_id does not bind run_sha256")
        if self.attempt_id != f"{self.run_id}-attempt-0001":
            raise AttemptRegistryError("attempt_id is not the sole frozen attempt")

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "entry_qualification_sha256": self.entry_qualification_sha256,
            "implementation_sha256": self.implementation_sha256,
            "protocol_sha256": self.protocol_sha256,
            "protocol_version": self.protocol_version,
            "q0_report_sha256": self.q0_report_sha256,
            "run_id": self.run_id,
            "run_sha256": self.run_sha256,
            "salt_commitment_sha256": self.salt_commitment_sha256,
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class Q1AttemptMarker:
    """One canonical durable marker state."""

    identity: Q1RunIdentity
    status: Literal["started", "completed", "failed"]
    worker_capability_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...] = ()
    schema: str = ATTEMPT_MARKER_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in ("started", "completed", "failed"):
            raise AttemptRegistryError("attempt marker status is invalid")
        _require_sha256(self.worker_capability_sha256, "worker_capability_sha256")
        names = tuple(name for name, _ in self.artifact_sha256)
        if self.status == "completed":
            if names != _EXPECTED_ARTIFACT_NAMES:
                raise AttemptRegistryError("completed marker lacks the exact artifact digest set")
        elif self.artifact_sha256:
            raise AttemptRegistryError("non-completed marker cannot bind result artifacts")
        for name, digest in self.artifact_sha256:
            if name not in _EXPECTED_ARTIFACT_NAMES:
                raise AttemptRegistryError(f"unknown attempt artifact {name!r}")
            _require_sha256(digest, f"artifact_sha256.{name}")

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": dict(self.artifact_sha256) if self.artifact_sha256 else None,
            "attempt_id": self.identity.attempt_id,
            "entry_qualification_sha256": self.identity.entry_qualification_sha256,
            "expected_counts": dict(_EXPECTED_COUNTS),
            "implementation_sha256": self.identity.implementation_sha256,
            "protocol_sha256": self.identity.protocol_sha256,
            "protocol_version": self.identity.protocol_version,
            "q0_report_sha256": self.identity.q0_report_sha256,
            "run_id": self.identity.run_id,
            "run_sha256": self.identity.run_sha256,
            "salt_commitment_sha256": self.identity.salt_commitment_sha256,
            "schema": self.schema,
            "status": self.status,
            "worker_capability_sha256": self.worker_capability_sha256,
        }


def derive_run_identity(
    *,
    protocol_version: str,
    protocol_sha256: str,
    implementation_sha256: str,
    q0_report_sha256: str,
    entry_qualification_sha256: str,
    salt_commitment_sha256: str,
) -> Q1RunIdentity:
    """Derive the sole Q1 attempt from all immutable execution bindings."""

    binding = {
        "entry_qualification_sha256": _require_sha256(
            entry_qualification_sha256,
            "entry_qualification_sha256",
        ),
        "implementation_sha256": _require_sha256(implementation_sha256, "implementation_sha256"),
        "protocol_sha256": _require_sha256(protocol_sha256, "protocol_sha256"),
        "protocol_version": _require_nonempty(protocol_version, "protocol_version"),
        "q0_report_sha256": _require_sha256(q0_report_sha256, "q0_report_sha256"),
        "salt_commitment_sha256": _require_sha256(
            salt_commitment_sha256,
            "salt_commitment_sha256",
        ),
        "schema": RUN_IDENTITY_SCHEMA,
    }
    run_sha256 = canonical_sha256(binding)
    run_id = f"wm002-q1-{run_sha256}"
    return Q1RunIdentity(
        protocol_version=protocol_version,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        entry_qualification_sha256=entry_qualification_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
        run_sha256=run_sha256,
        run_id=run_id,
        attempt_id=f"{run_id}-attempt-0001",
    )


def attempt_marker_path(registry_directory: Path, identity: Q1RunIdentity) -> Path:
    """Return the fixed experiment-global marker path for every run binding."""

    del identity
    descriptor = _open_registry_directory(registry_directory)
    os.close(descriptor)
    return registry_directory / ATTEMPT_MARKER_FILENAME


def claim_attempt(
    registry_directory: Path,
    identity: Q1RunIdentity,
    *,
    worker_capability_sha256: str,
) -> Path:
    """Durably claim the one experiment-global Q1 tombstone with ``O_EXCL``."""

    path = registry_directory / ATTEMPT_MARKER_FILENAME
    payload = canonical_json_bytes(
        Q1AttemptMarker(identity, "started", worker_capability_sha256).as_dict(),
        newline=True,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with _locked_registry_directory(registry_directory, exclusive=True) as directory_descriptor:
        descriptor: int | None = None
        created = False
        try:
            try:
                descriptor = os.open(
                    ATTEMPT_MARKER_FILENAME,
                    flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError as error:
                raise AttemptRegistryError("the experiment-global Q1 attempt has already been claimed") from error
            created = True
            os.fchmod(descriptor, 0o600)
            _require_private_regular_descriptor(descriptor, label="attempt marker")
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            _require_private_regular_descriptor(descriptor, label="attempt marker")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if created:
                os.fsync(directory_descriptor)
    return path


def finalize_attempt(
    path: Path,
    identity: Q1RunIdentity,
    *,
    status: Literal["completed", "failed"],
    expected_worker_capability_sha256: str,
    artifacts: Mapping[str, Path] | None = None,
) -> Q1AttemptMarker:
    """Serialize and atomically finalize the fixed marker exactly once."""

    parent = _marker_parent(path)
    with _locked_registry_directory(parent, exclusive=True) as directory_descriptor:
        current, current_signature = _load_attempt_marker_from_directory(
            directory_descriptor,
            expected_identity=identity,
            expected_worker_capability_sha256=expected_worker_capability_sha256,
        )
        if current.status != "started":
            raise AttemptRegistryError("only a started marker can be finalized")
        if status == "completed":
            if artifacts is None or tuple(sorted(artifacts)) != _EXPECTED_ARTIFACT_NAMES:
                raise AttemptRegistryError("completed attempt requires the exact artifact path set")
            artifact_rows = tuple((name, _sha256_file(artifacts[name])) for name in _EXPECTED_ARTIFACT_NAMES)
        else:
            if artifacts is not None:
                raise AttemptRegistryError("failed attempt cannot bind artifacts")
            artifact_rows = ()
        marker = Q1AttemptMarker(
            identity=identity,
            status=status,
            worker_capability_sha256=current.worker_capability_sha256,
            artifact_sha256=artifact_rows,
        )
        _replace_marker_locked(
            directory_descriptor,
            canonical_json_bytes(marker.as_dict(), newline=True),
            expected_signature=current_signature,
        )
        persisted, _ = _load_attempt_marker_from_directory(
            directory_descriptor,
            expected_identity=identity,
            expected_worker_capability_sha256=expected_worker_capability_sha256,
        )
        if persisted != marker:
            raise AttemptRegistryError("finalized attempt marker differs from the durable replacement")
        return marker


def load_attempt_marker(
    path: Path,
    *,
    expected_identity: Q1RunIdentity,
    expected_worker_capability_sha256: str,
) -> Q1AttemptMarker:
    """Strictly load the fixed marker through a locked registry descriptor."""

    parent = _marker_parent(path)
    with _locked_registry_directory(parent, exclusive=False) as directory_descriptor:
        marker, _ = _load_attempt_marker_from_directory(
            directory_descriptor,
            expected_identity=expected_identity,
            expected_worker_capability_sha256=expected_worker_capability_sha256,
        )
        return marker


def _load_attempt_marker_from_directory(
    directory_descriptor: int,
    *,
    expected_identity: Q1RunIdentity,
    expected_worker_capability_sha256: str,
) -> tuple[Q1AttemptMarker, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            ATTEMPT_MARKER_FILENAME,
            flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise AttemptRegistryError("cannot open the fixed Q1 attempt marker safely") from error
    try:
        before = _require_private_regular_descriptor(descriptor, label="attempt marker")
        raw = _read_bounded_descriptor(descriptor, limit=1024 * 1024)
        after = _require_private_regular_descriptor(descriptor, label="attempt marker")
    finally:
        os.close(descriptor)
    before_signature = _metadata_signature(before)
    if before_signature != _metadata_signature(after) or len(raw) != after.st_size:
        raise AttemptRegistryError("attempt marker changed during its descriptor read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttemptRegistryError("attempt marker is not valid JSON") from error
    if canonical_json_bytes(value, newline=True) != raw or not isinstance(value, dict):
        raise AttemptRegistryError("attempt marker is not canonical JSON")
    expected_keys = set(Q1AttemptMarker(expected_identity, "started", expected_worker_capability_sha256).as_dict())
    if set(value) != expected_keys:
        raise AttemptRegistryError("attempt marker fields differ from the strict contract")
    for name, expected in expected_identity.as_dict().items():
        if name == "schema":
            continue
        if value.get(name) != expected:
            raise AttemptRegistryError(f"attempt marker identity mismatch: {name}")
    _require_sha256(expected_worker_capability_sha256, "expected_worker_capability_sha256")
    if value.get("worker_capability_sha256") != expected_worker_capability_sha256:
        raise AttemptRegistryError("attempt marker worker capability commitment mismatch")
    if value.get("schema") != ATTEMPT_MARKER_SCHEMA or value.get("expected_counts") != _EXPECTED_COUNTS:
        raise AttemptRegistryError("attempt marker schema or budget differs")
    status = value.get("status")
    if status not in ("started", "completed", "failed"):
        raise AttemptRegistryError("attempt marker status is invalid")
    raw_artifacts = value.get("artifact_sha256")
    if raw_artifacts is None:
        artifact_rows: tuple[tuple[str, str], ...] = ()
    elif isinstance(raw_artifacts, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in raw_artifacts.items()
    ):
        artifact_rows = tuple(sorted(raw_artifacts.items()))
    else:
        raise AttemptRegistryError("attempt marker artifact digests are invalid")
    return (
        Q1AttemptMarker(
            expected_identity,
            status,
            expected_worker_capability_sha256,
            artifact_rows,
        ),
        before_signature,
    )


def _replace_marker_locked(
    directory_descriptor: int,
    payload: bytes,
    *,
    expected_signature: tuple[int, ...],
) -> None:
    temporary_name = f".{ATTEMPT_MARKER_FILENAME}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_descriptor)
    created = True
    try:
        os.fchmod(descriptor, 0o600)
        _require_private_regular_descriptor(descriptor, label="temporary attempt marker")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _require_private_regular_descriptor(descriptor, label="temporary attempt marker")
        os.close(descriptor)
        descriptor = -1
        current = os.stat(
            ATTEMPT_MARKER_FILENAME,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        _require_private_regular_metadata(current, label="attempt marker")
        if _metadata_signature(current) != expected_signature:
            raise AttemptRegistryError("attempt marker changed before serialized finalization")
        os.replace(
            temporary_name,
            ATTEMPT_MARKER_FILENAME,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        created = False
        os.fsync(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


def _marker_parent(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name != ATTEMPT_MARKER_FILENAME:
        raise AttemptRegistryError(f"attempt marker path must use the fixed name {ATTEMPT_MARKER_FILENAME!r}")
    return candidate.parent


def _open_registry_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AttemptRegistryError("attempt registry must be an existing non-symlink directory") from error
    try:
        _require_registry_descriptor(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _locked_registry_directory(path: Path, *, exclusive: bool) -> Iterator[int]:
    descriptor = _open_registry_directory(path)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        _require_registry_descriptor(descriptor)
        yield descriptor
        _require_registry_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _require_registry_descriptor(descriptor: int) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise AttemptRegistryError("attempt registry must be a directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise AttemptRegistryError("attempt registry permissions must be exactly 0700")
    return metadata


def _require_private_regular_descriptor(descriptor: int, *, label: str) -> os.stat_result:
    metadata = os.fstat(descriptor)
    _require_private_regular_metadata(metadata, label=label)
    return metadata


def _require_private_regular_metadata(metadata: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise AttemptRegistryError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise AttemptRegistryError(f"{label} permissions must be exactly 0600")
    if metadata.st_nlink != 1:
        raise AttemptRegistryError(f"{label} must have exactly one hard link")


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AttemptRegistryError(f"cannot open result artifact {path.name!r} safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttemptRegistryError(f"result artifact {path.name!r} must be a regular file")
        if before.st_nlink != 1:
            raise AttemptRegistryError(f"result artifact {path.name!r} must have exactly one hard link")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if _metadata_signature(before) != _metadata_signature(after) or total != after.st_size:
            raise AttemptRegistryError(f"result artifact {path.name!r} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_bounded_descriptor(descriptor: int, *, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(1024 * 1024, limit + 1 - total)):
        total += len(chunk)
        if total > limit:
            raise AttemptRegistryError(f"attempt marker exceeds the bounded {limit}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise AttemptRegistryError("attempt marker write made no progress")
        offset += written


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SHA256_CHARS for character in value):
        raise AttemptRegistryError(f"{label} must be lowercase SHA-256")
    return value


def _require_nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttemptRegistryError(f"{label} must be nonempty")
    return value


__all__ = (
    "ATTEMPT_MARKER_FILENAME",
    "ATTEMPT_MARKER_SCHEMA",
    "RUN_IDENTITY_SCHEMA",
    "AttemptRegistryError",
    "Q1AttemptMarker",
    "Q1RunIdentity",
    "attempt_marker_path",
    "claim_attempt",
    "derive_run_identity",
    "finalize_attempt",
    "load_attempt_marker",
)
