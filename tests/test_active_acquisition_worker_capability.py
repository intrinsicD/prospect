from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import bench.active_acquisition.worker_capability as capability_module
from bench.active_acquisition.attempt import Q1RunIdentity, derive_run_identity
from bench.active_acquisition.contracts import ARM_ORDER, canonical_json_bytes
from bench.active_acquisition.worker_capability import (
    MAX_WORKER_CAPABILITY_PAYLOAD_BYTES,
    WORKER_CAPABILITY_SCHEMA_PATH,
    Q1WorkerCapability,
    WorkerCapabilityError,
    consume_worker_capability_fd,
    decode_worker_capability_wire,
    encode_worker_capability,
    make_worker_capability,
    worker_capability_acknowledgement,
    worker_capability_commitment,
)

_SECRET = bytes(range(32))


def _identity() -> Q1RunIdentity:
    return derive_run_identity(
        protocol_version="0.3.0-q1",
        protocol_sha256="1" * 64,
        implementation_sha256="2" * 64,
        q0_report_sha256="3" * 64,
        entry_qualification_sha256="4" * 64,
        salt_commitment_sha256="5" * 64,
    )


def _paths(*, role: str, master: int = 1, arm: str = ARM_ORDER[0]) -> dict[str, str | None]:
    execution = Path("/execution")
    incomplete = execution / "result.incomplete"
    master_directory = incomplete / f"master-{master}"
    restore = role == "restore"
    return {
        "attempt_registry_directory": "/attempts",
        "entry_report_path": "/inputs/entry.json",
        "execution_root": str(execution),
        "frame_path": str(master_directory / f"{arm}.frames.bin") if restore else None,
        "incomplete_directory": str(incomplete),
        "index_path": str(master_directory / f"{arm}.index.jsonl") if restore else None,
        "master_directory": str(master_directory),
        "output_path": str(master_directory / f"{arm}.restored.jsonl") if restore else str(master_directory),
        "prospective_review_path": "/inputs/review.json",
        "q0_report_path": "/inputs/q0.json",
        "raw_trace_path": str(master_directory / "raw.jsonl") if restore else None,
        "secret_salt_path": "/private/salt.bin",
    }


def _capability(*, role: str = "producer") -> Q1WorkerCapability:
    arm = ARM_ORDER[0] if role == "restore" else None
    return make_worker_capability(
        role=role,  # type: ignore[arg-type]
        run_identity=_identity(),
        parent_pid=101,
        child_pid=202,
        master=1,
        arm=arm,
        paths=_paths(role=role),
    )


@pytest.mark.parametrize("role", ["producer", "restore"])
def test_capability_wire_round_trip_binds_role_run_pid_and_paths(role: str) -> None:
    source = _capability(role=role)
    wire = encode_worker_capability(_SECRET, source)

    decoded = decode_worker_capability_wire(
        wire,
        expected_parent_pid=101,
        expected_child_pid=202,
    )

    assert decoded.capability == source
    assert decoded.worker_capability_sha256 == hashlib.sha256(_SECRET).hexdigest()
    assert decoded.worker_capability_sha256 == worker_capability_commitment(_SECRET)
    assert _SECRET not in canonical_json_bytes(source.as_dict())


def test_socket_consumer_closes_descriptor_acknowledges_and_rejects_trailing_bytes() -> None:
    wire = encode_worker_capability(_SECRET, _capability())
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    descriptor = child.detach()
    parent.sendall(wire)
    parent.shutdown(socket.SHUT_WR)

    decoded = consume_worker_capability_fd(
        descriptor,
        expected_parent_pid=101,
        expected_child_pid=202,
    )
    assert decoded.capability.role == "producer"
    assert parent.recv(1024) == worker_capability_acknowledgement(wire)
    assert parent.recv(1) == b""
    parent.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)

    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    descriptor = child.detach()
    parent.sendall(wire + b"extra")
    parent.shutdown(socket.SHUT_WR)
    with pytest.raises(WorkerCapabilityError, match="trailing bytes"):
        consume_worker_capability_fd(descriptor)
    parent.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_consumer_rejects_replayable_regular_file_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "capability-wire.bin"
    path.write_bytes(encode_worker_capability(_SECRET, _capability()))
    descriptor = os.open(path, os.O_RDONLY)

    with pytest.raises(WorkerCapabilityError, match="descriptor must be a socket"):
        consume_worker_capability_fd(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_wire_rejects_bad_secret_length_authenticator_pid_truncation_and_trailing() -> None:
    source = _capability()
    with pytest.raises(WorkerCapabilityError, match="exactly 32 bytes"):
        encode_worker_capability(b"short", source)

    wire = encode_worker_capability(_SECRET, source)
    tampered = wire[:-1] + bytes((wire[-1] ^ 1,))
    with pytest.raises(WorkerCapabilityError, match="authentication failed"):
        decode_worker_capability_wire(tampered)
    with pytest.raises(WorkerCapabilityError, match="parent PID differs"):
        decode_worker_capability_wire(wire, expected_parent_pid=999)
    with pytest.raises(WorkerCapabilityError, match="child PID differs"):
        decode_worker_capability_wire(wire, expected_child_pid=999)
    with pytest.raises(WorkerCapabilityError, match="wire length differs"):
        decode_worker_capability_wire(wire[:-1])
    with pytest.raises(WorkerCapabilityError, match="wire length differs"):
        decode_worker_capability_wire(wire + b"extra")


def test_authenticated_noncanonical_payload_and_oversized_length_fail_closed() -> None:
    source = _capability()
    noncanonical = json.dumps(source.as_dict(), indent=2).encode("ascii")
    authenticator = hmac.digest(
        _SECRET,
        capability_module._AUTHENTICATION_DOMAIN + noncanonical,
        "sha256",
    )
    wire = capability_module._WIRE_LENGTH.pack(len(noncanonical)) + _SECRET + noncanonical + authenticator
    with pytest.raises(WorkerCapabilityError, match="not canonical JSON"):
        decode_worker_capability_wire(wire)

    oversized = capability_module._WIRE_LENGTH.pack(MAX_WORKER_CAPABILITY_PAYLOAD_BYTES + 1)
    with pytest.raises(WorkerCapabilityError, match="payload length is invalid"):
        decode_worker_capability_wire(oversized + _SECRET + b"x" + b"0" * capability_module._AUTHENTICATOR_BYTES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "prospect.wm002.active-acquisition.q1-run-identity.v999"),
        ("protocol_version", "0.3.0-q1-mutated"),
    ],
)
def test_authenticated_run_identity_constant_mutations_fail_closed(
    field: str,
    value: str,
) -> None:
    payload_value = _capability().as_dict()
    run_identity = payload_value["run_identity"]
    assert isinstance(run_identity, dict)
    run_identity[field] = value
    payload = canonical_json_bytes(payload_value)
    authenticator = hmac.digest(
        _SECRET,
        capability_module._AUTHENTICATION_DOMAIN + payload,
        "sha256",
    )
    wire = capability_module._WIRE_LENGTH.pack(len(payload)) + _SECRET + payload + authenticator

    with pytest.raises(WorkerCapabilityError, match="run identity constants differ"):
        decode_worker_capability_wire(wire)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("relative", "absolute normalized"),
        ("master_directory", "master directory relationship"),
        ("producer_restore_path", "restore-only values"),
        ("restore_frame", "frame path relationship"),
        ("restore_output", "output path relationship"),
    ],
)
def test_role_path_relationship_mutations_are_rejected(mutation: str, message: str) -> None:
    role = "restore" if mutation.startswith("restore_") else "producer"
    paths = _paths(role=role)
    if mutation == "relative":
        paths["entry_report_path"] = "relative/entry.json"
    elif mutation == "master_directory":
        paths["master_directory"] = "/execution/result.incomplete/master-3"
    elif mutation == "producer_restore_path":
        paths["frame_path"] = "/execution/result.incomplete/master-1/frame.bin"
    elif mutation == "restore_frame":
        paths["frame_path"] = "/execution/result.incomplete/master-1/wrong.frames.bin"
    else:
        paths["output_path"] = "/execution/result.incomplete/master-1/wrong.restored.jsonl"

    with pytest.raises(WorkerCapabilityError, match=message):
        make_worker_capability(
            role=role,  # type: ignore[arg-type]
            run_identity=_identity(),
            parent_pid=101,
            child_pid=202,
            master=1,
            arm=ARM_ORDER[0] if role == "restore" else None,
            paths=paths,
        )


def test_capability_schema_is_strict_and_matches_codec_payloads() -> None:
    schema = json.loads(WORKER_CAPABILITY_SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for role in ("producer", "restore"):
        validator.validate(_capability(role=role).as_dict())

    mutated = _capability().as_dict()
    mutated["secret"] = _SECRET.hex()
    errors = tuple(validator.iter_errors(mutated))
    assert errors


def test_secret_never_appears_in_decoded_repr_or_errors() -> None:
    source = _capability()
    wire = encode_worker_capability(_SECRET, source)
    decoded = decode_worker_capability_wire(wire)
    secret_hex = _SECRET.hex()
    assert secret_hex not in repr(decoded)
    assert repr(_SECRET) not in repr(decoded)

    tampered = wire[:-1] + bytes((wire[-1] ^ 1,))
    with pytest.raises(WorkerCapabilityError) as captured:
        decode_worker_capability_wire(tampered)
    assert secret_hex not in str(captured.value)
    assert repr(_SECRET) not in str(captured.value)
