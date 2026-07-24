"""Bounded parent-to-worker operational capabilities for WM-002 Q1.

The raw 32-byte operational secret exists only inside the inherited socket wire.
Decoded values expose its SHA-256 commitment, never the secret itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import stat
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from bench.active_acquisition.attempt import RUN_IDENTITY_SCHEMA, Q1RunIdentity
from bench.active_acquisition.contracts import (
    ARM_ORDER,
    Q1_PROTOCOL_VERSION,
    canonical_json_bytes,
)

WORKER_CAPABILITY_SCHEMA: Final = "prospect.wm002.active-acquisition.q1-worker-capability.v1"
WORKER_CAPABILITY_SCHEMA_PATH: Final = Path(__file__).with_name("schemas") / "q1-worker-capability.schema.json"
WORKER_CAPABILITY_SECRET_BYTES: Final = 32
MAX_WORKER_CAPABILITY_PAYLOAD_BYTES: Final = 64 * 1024
WORKER_CAPABILITY_ACK_BYTES: Final = hashlib.sha256().digest_size

_WIRE_LENGTH: Final = struct.Struct(">I")
_AUTHENTICATOR_BYTES: Final = hashlib.sha256().digest_size
_AUTHENTICATION_DOMAIN: Final = b"prospect.wm002.active-acquisition.q1-worker-capability.v1\x00"
_ACKNOWLEDGEMENT_DOMAIN: Final = b"prospect.wm002.active-acquisition.q1-worker-capability-ack.v1\x00"
_MASTER_COUNT: Final = 4
_PATH_NAMES: Final = (
    "attempt_registry_directory",
    "entry_report_path",
    "execution_root",
    "frame_path",
    "incomplete_directory",
    "index_path",
    "master_directory",
    "output_path",
    "prospective_review_path",
    "q0_report_path",
    "raw_trace_path",
    "secret_salt_path",
)


class WorkerCapabilityError(RuntimeError):
    """An operational worker capability was malformed or unauthenticated."""


@dataclass(frozen=True, slots=True)
class Q1WorkerCapability:
    """One exact producer or restore worker launch binding."""

    role: Literal["producer", "restore"]
    run_identity: Q1RunIdentity
    parent_pid: int
    child_pid: int
    master: int
    arm: str | None
    paths: tuple[tuple[str, str | None], ...]
    schema: str = WORKER_CAPABILITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != WORKER_CAPABILITY_SCHEMA:
            raise WorkerCapabilityError("worker capability schema differs")
        if self.role not in ("producer", "restore"):
            raise WorkerCapabilityError("worker capability role differs")
        if not isinstance(self.run_identity, Q1RunIdentity):
            raise WorkerCapabilityError("worker capability run identity is invalid")
        for name, value in (("parent_pid", self.parent_pid), ("child_pid", self.child_pid)):
            if type(value) is not int or value <= 0:
                raise WorkerCapabilityError(f"worker capability {name} is invalid")
        if self.parent_pid == self.child_pid:
            raise WorkerCapabilityError("worker capability parent and child PIDs must differ")
        if type(self.master) is not int or self.master not in range(_MASTER_COUNT):
            raise WorkerCapabilityError("worker capability master is outside the frozen set")
        if tuple(name for name, _value in self.paths) != _PATH_NAMES:
            raise WorkerCapabilityError("worker capability path fields differ from the exact set")
        values = dict(self.paths)
        for name in _PATH_NAMES:
            path_value = values[name]
            if path_value is not None:
                _require_absolute_normal_path(path_value, name)
        _validate_role_paths(
            role=self.role,
            master=self.master,
            arm=self.arm,
            paths=values,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the strict secret-free canonical payload value."""

        return {
            "arm": self.arm,
            "child_pid": self.child_pid,
            "master": self.master,
            "parent_pid": self.parent_pid,
            "paths": dict(self.paths),
            "role": self.role,
            "run_identity": self.run_identity.as_dict(),
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class DecodedWorkerCapability:
    """Authenticated secret-free result returned by a wire consumer."""

    capability: Q1WorkerCapability
    worker_capability_sha256: str


def make_worker_capability(
    *,
    role: Literal["producer", "restore"],
    run_identity: Q1RunIdentity,
    parent_pid: int,
    child_pid: int,
    master: int,
    arm: str | None,
    paths: Mapping[str, str | None],
) -> Q1WorkerCapability:
    """Construct one capability while canonicalizing the path-field order."""

    if set(paths) != set(_PATH_NAMES):
        raise WorkerCapabilityError("worker capability path fields differ from the exact set")
    return Q1WorkerCapability(
        role=role,
        run_identity=run_identity,
        parent_pid=parent_pid,
        child_pid=child_pid,
        master=master,
        arm=arm,
        paths=tuple((name, paths[name]) for name in _PATH_NAMES),
    )


def worker_capability_commitment(secret: bytes) -> str:
    """Return the marker commitment for one exact 32-byte operational secret."""

    _require_secret(secret)
    return hashlib.sha256(secret).hexdigest()


def encode_worker_capability(secret: bytes, capability: Q1WorkerCapability) -> bytes:
    """Encode one bounded authenticated socket wire."""

    _require_secret(secret)
    if not isinstance(capability, Q1WorkerCapability):
        raise WorkerCapabilityError("worker capability value is invalid")
    payload = canonical_json_bytes(capability.as_dict())
    if not payload or len(payload) > MAX_WORKER_CAPABILITY_PAYLOAD_BYTES:
        raise WorkerCapabilityError("worker capability payload exceeds its bounded size")
    authenticator = hmac.digest(
        secret,
        _AUTHENTICATION_DOMAIN + payload,
        "sha256",
    )
    return _WIRE_LENGTH.pack(len(payload)) + secret + payload + authenticator


def worker_capability_acknowledgement(wire: bytes) -> bytes:
    """Derive the fixed authenticated acknowledgement for one exact wire."""

    if type(wire) is not bytes or len(wire) < _minimum_wire_bytes():
        raise WorkerCapabilityError("worker capability wire is truncated")
    payload_length = _WIRE_LENGTH.unpack(wire[: _WIRE_LENGTH.size])[0]
    expected_length = _WIRE_LENGTH.size + WORKER_CAPABILITY_SECRET_BYTES + payload_length + _AUTHENTICATOR_BYTES
    if payload_length <= 0 or payload_length > MAX_WORKER_CAPABILITY_PAYLOAD_BYTES:
        raise WorkerCapabilityError("worker capability payload length is invalid")
    if len(wire) != expected_length:
        raise WorkerCapabilityError("worker capability wire length differs")
    secret_start = _WIRE_LENGTH.size
    payload_start = secret_start + WORKER_CAPABILITY_SECRET_BYTES
    authenticator_start = payload_start + payload_length
    secret = wire[secret_start:payload_start]
    payload = wire[payload_start:authenticator_start]
    authenticator = wire[authenticator_start:]
    return hmac.digest(
        secret,
        _ACKNOWLEDGEMENT_DOMAIN + hashlib.sha256(payload).digest() + authenticator,
        "sha256",
    )


def decode_worker_capability_wire(
    wire: bytes,
    *,
    expected_parent_pid: int | None = None,
    expected_child_pid: int | None = None,
) -> DecodedWorkerCapability:
    """Authenticate and strictly decode one complete in-memory socket wire."""

    if type(wire) is not bytes or len(wire) < _minimum_wire_bytes():
        raise WorkerCapabilityError("worker capability wire is truncated")
    payload_length = _WIRE_LENGTH.unpack(wire[: _WIRE_LENGTH.size])[0]
    if payload_length <= 0 or payload_length > MAX_WORKER_CAPABILITY_PAYLOAD_BYTES:
        raise WorkerCapabilityError("worker capability payload length is invalid")
    expected_length = _WIRE_LENGTH.size + WORKER_CAPABILITY_SECRET_BYTES + payload_length + _AUTHENTICATOR_BYTES
    if len(wire) != expected_length:
        raise WorkerCapabilityError("worker capability wire length differs")
    secret_start = _WIRE_LENGTH.size
    payload_start = secret_start + WORKER_CAPABILITY_SECRET_BYTES
    authenticator_start = payload_start + payload_length
    secret = wire[secret_start:payload_start]
    payload = wire[payload_start:authenticator_start]
    authenticator = wire[authenticator_start:]
    expected_authenticator = hmac.digest(
        secret,
        _AUTHENTICATION_DOMAIN + payload,
        "sha256",
    )
    if not hmac.compare_digest(authenticator, expected_authenticator):
        raise WorkerCapabilityError("worker capability authentication failed")
    value = _decode_canonical_payload(payload)
    capability = _decode_capability(value)
    _require_expected_pid(capability.parent_pid, expected_parent_pid, "parent")
    _require_expected_pid(capability.child_pid, expected_child_pid, "child")
    return DecodedWorkerCapability(
        capability=capability,
        worker_capability_sha256=hashlib.sha256(secret).hexdigest(),
    )


def consume_worker_capability_fd(
    descriptor: int,
    *,
    expected_parent_pid: int | None = None,
    expected_child_pid: int | None = None,
) -> DecodedWorkerCapability:
    """Consume exactly one wire through an inherited FD and always close it."""

    if type(descriptor) is not int or descriptor < 0:
        raise WorkerCapabilityError("worker capability descriptor is invalid")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISSOCK(metadata.st_mode):
            raise WorkerCapabilityError("worker capability descriptor must be a socket")
        probe = socket.socket(fileno=os.dup(descriptor))
        try:
            if (
                probe.family != socket.AF_UNIX
                or probe.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
            ):
                raise WorkerCapabilityError("worker capability descriptor must be an AF_UNIX stream socket")
        finally:
            probe.close()
        header = _read_exact(descriptor, _WIRE_LENGTH.size)
        payload_length = _WIRE_LENGTH.unpack(header)[0]
        if payload_length <= 0 or payload_length > MAX_WORKER_CAPABILITY_PAYLOAD_BYTES:
            raise WorkerCapabilityError("worker capability payload length is invalid")
        remainder = _read_exact(
            descriptor,
            WORKER_CAPABILITY_SECRET_BYTES + payload_length + _AUTHENTICATOR_BYTES,
        )
        if os.read(descriptor, 1):
            raise WorkerCapabilityError("worker capability socket contains trailing bytes")
        wire = header + remainder
        decoded = decode_worker_capability_wire(
            wire,
            expected_parent_pid=expected_parent_pid,
            expected_child_pid=expected_child_pid,
        )
        _write_all(descriptor, worker_capability_acknowledgement(wire))
        return decoded
    except WorkerCapabilityError:
        raise
    except OSError as error:
        raise WorkerCapabilityError("worker capability socket exchange failed") from error
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _decode_canonical_payload(payload: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: _reject_nonfinite(token),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise WorkerCapabilityError("worker capability payload is not JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise WorkerCapabilityError("worker capability payload is not canonical JSON")
    return value


def _decode_capability(value: Mapping[str, object]) -> Q1WorkerCapability:
    expected = {
        "arm",
        "child_pid",
        "master",
        "parent_pid",
        "paths",
        "role",
        "run_identity",
        "schema",
    }
    if set(value) != expected:
        raise WorkerCapabilityError("worker capability fields differ from the exact set")
    role = value["role"]
    if role not in ("producer", "restore"):
        raise WorkerCapabilityError("worker capability role differs")
    run_identity = _decode_run_identity(value["run_identity"])
    paths_value = value["paths"]
    if not isinstance(paths_value, dict) or set(paths_value) != set(_PATH_NAMES):
        raise WorkerCapabilityError("worker capability path fields differ from the exact set")
    paths: dict[str, str | None] = {}
    for name in _PATH_NAMES:
        item = paths_value[name]
        if item is not None and not isinstance(item, str):
            raise WorkerCapabilityError("worker capability path value is invalid")
        paths[name] = item
    parent_pid = value["parent_pid"]
    child_pid = value["child_pid"]
    master = value["master"]
    arm = value["arm"]
    if type(parent_pid) is not int or type(child_pid) is not int or type(master) is not int:
        raise WorkerCapabilityError("worker capability integer field is invalid")
    if arm is not None and not isinstance(arm, str):
        raise WorkerCapabilityError("worker capability arm is invalid")
    return make_worker_capability(
        role=role,
        run_identity=run_identity,
        parent_pid=parent_pid,
        child_pid=child_pid,
        master=master,
        arm=arm,
        paths=paths,
    )


def _decode_run_identity(value: object) -> Q1RunIdentity:
    if not isinstance(value, dict):
        raise WorkerCapabilityError("worker capability run identity is invalid")
    expected = {
        "attempt_id",
        "entry_qualification_sha256",
        "implementation_sha256",
        "protocol_sha256",
        "protocol_version",
        "q0_report_sha256",
        "run_id",
        "run_sha256",
        "salt_commitment_sha256",
        "schema",
    }
    if set(value) != expected or not all(isinstance(value[name], str) for name in expected):
        raise WorkerCapabilityError("worker capability run identity fields differ")
    if value["schema"] != RUN_IDENTITY_SCHEMA or value["protocol_version"] != Q1_PROTOCOL_VERSION:
        raise WorkerCapabilityError("worker capability run identity constants differ")
    try:
        return Q1RunIdentity(
            protocol_version=value["protocol_version"],
            protocol_sha256=value["protocol_sha256"],
            implementation_sha256=value["implementation_sha256"],
            q0_report_sha256=value["q0_report_sha256"],
            entry_qualification_sha256=value["entry_qualification_sha256"],
            salt_commitment_sha256=value["salt_commitment_sha256"],
            run_sha256=value["run_sha256"],
            run_id=value["run_id"],
            attempt_id=value["attempt_id"],
            schema=value["schema"],
        )
    except Exception as error:
        raise WorkerCapabilityError("worker capability run identity is invalid") from error


def _validate_role_paths(
    *,
    role: str,
    master: int,
    arm: str | None,
    paths: Mapping[str, str | None],
) -> None:
    required_common = (
        "attempt_registry_directory",
        "entry_report_path",
        "execution_root",
        "incomplete_directory",
        "master_directory",
        "output_path",
        "prospective_review_path",
        "q0_report_path",
        "secret_salt_path",
    )
    if any(paths[name] is None for name in required_common):
        raise WorkerCapabilityError("worker capability required path is absent")
    execution_root = Path(_path_string(paths, "execution_root"))
    incomplete = Path(_path_string(paths, "incomplete_directory"))
    master_directory = Path(_path_string(paths, "master_directory"))
    output = Path(_path_string(paths, "output_path"))
    if incomplete.parent != execution_root or not incomplete.name.endswith(".incomplete"):
        raise WorkerCapabilityError("worker capability incomplete directory relationship differs")
    if master_directory != incomplete / f"master-{master}":
        raise WorkerCapabilityError("worker capability master directory relationship differs")
    if role == "producer":
        if arm is not None or any(paths[name] is not None for name in ("frame_path", "index_path", "raw_trace_path")):
            raise WorkerCapabilityError("producer capability contains restore-only values")
        if output != master_directory:
            raise WorkerCapabilityError("producer capability output directory relationship differs")
        return
    if arm not in ARM_ORDER:
        raise WorkerCapabilityError("restore capability arm is outside the frozen set")
    frame = Path(_path_string(paths, "frame_path"))
    index = Path(_path_string(paths, "index_path"))
    raw_trace = Path(_path_string(paths, "raw_trace_path"))
    if frame != master_directory / f"{arm}.frames.bin":
        raise WorkerCapabilityError("restore capability frame path relationship differs")
    if index != master_directory / f"{arm}.index.jsonl":
        raise WorkerCapabilityError("restore capability index path relationship differs")
    if raw_trace != master_directory / "raw.jsonl":
        raise WorkerCapabilityError("restore capability raw path relationship differs")
    if output != master_directory / f"{arm}.restored.jsonl":
        raise WorkerCapabilityError("restore capability output path relationship differs")


def _path_string(paths: Mapping[str, str | None], name: str) -> str:
    value = paths[name]
    if not isinstance(value, str):
        raise WorkerCapabilityError("worker capability required path is absent")
    return value


def _require_absolute_normal_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkerCapabilityError(f"worker capability {label} is invalid")
    if not Path(value).is_absolute() or os.path.normpath(value) != value:
        raise WorkerCapabilityError(f"worker capability {label} is not an absolute normalized path")
    return value


def _require_secret(secret: object) -> bytes:
    if type(secret) is not bytes or len(secret) != WORKER_CAPABILITY_SECRET_BYTES:
        raise WorkerCapabilityError("worker capability secret must be exactly 32 bytes")
    return secret


def _require_expected_pid(actual: int, expected: int | None, label: str) -> None:
    if expected is None:
        return
    if type(expected) is not int or expected <= 0 or actual != expected:
        raise WorkerCapabilityError(f"worker capability {label} PID differs")


def _read_exact(descriptor: int, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise WorkerCapabilityError("worker capability socket is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise WorkerCapabilityError("worker capability acknowledgement write failed")
        view = view[written:]


def _minimum_wire_bytes() -> int:
    return _WIRE_LENGTH.size + WORKER_CAPABILITY_SECRET_BYTES + 1 + _AUTHENTICATOR_BYTES


def _reject_nonfinite(token: str) -> object:
    raise WorkerCapabilityError(f"non-finite worker capability token {token!r} is forbidden")


__all__ = (
    "MAX_WORKER_CAPABILITY_PAYLOAD_BYTES",
    "WORKER_CAPABILITY_ACK_BYTES",
    "WORKER_CAPABILITY_SCHEMA",
    "WORKER_CAPABILITY_SCHEMA_PATH",
    "WORKER_CAPABILITY_SECRET_BYTES",
    "DecodedWorkerCapability",
    "Q1WorkerCapability",
    "WorkerCapabilityError",
    "consume_worker_capability_fd",
    "decode_worker_capability_wire",
    "encode_worker_capability",
    "make_worker_capability",
    "worker_capability_acknowledgement",
    "worker_capability_commitment",
)
