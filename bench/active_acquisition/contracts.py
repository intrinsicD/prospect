"""Frozen Q1 protocol, schema, binding, and canonical-serialization helpers.

This module owns mechanics shared by the producer and independent auditor.  It
does not compute outcomes, aggregate metrics, or gate verdicts.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

Q1_PROTOCOL_VERSION: Final = "0.3.0-q1"
Q1_PROTOCOL_PATH: Final = Path(__file__).with_name("q1_protocol.json")
Q1_SCHEMA_DIRECTORY: Final = Path(__file__).with_name("schemas")
Q1_SCHEMA_PATHS: Final = {
    "aggregate": Q1_SCHEMA_DIRECTORY / "q1-aggregate.schema.json",
    "audit_output": Q1_SCHEMA_DIRECTORY / "q1-audit-output.schema.json",
    "checkpoint_frame": Q1_SCHEMA_DIRECTORY / "q1-checkpoint-frame.schema.json",
    "private_audit": Q1_SCHEMA_DIRECTORY / "q1-private-audit.schema.json",
    "raw_trace": Q1_SCHEMA_DIRECTORY / "q1-raw-trace.schema.json",
    "restored_trace": Q1_SCHEMA_DIRECTORY / "q1-restored-trace.schema.json",
}
Q0_REPORT_SHA256: Final = "779e8d8128312da2239107058137faac54751df620efb31291c0af98c2b8f243"
Q0_PROTOCOL_SHA256: Final = "90b73ad4815380f113f91d0542bf7b91fd7e5196b5afd7f8c46b7fde9ec070cb"
Q0_IMPLEMENTATION_SHA256: Final = "c9e6689a0ce66e5b79f733c057b839a155500908ba21a5adbf64637cb090c324"

ARM_ORDER: Final = (
    "prospect_expected_return",
    "independent_fraction_oracle",
    "goal_only",
    "raw_observation_entropy",
    "eig_only",
    "shuffled_information",
    "uniform_random",
)
ACTION_ORDER: Final = ("skip", "weak", "strong", "overpowered", "nuisance")
TERMINAL_ORDER: Final = (1, -1)
CHECKPOINT_COMPONENTS: Final = (
    "domain_custody",
    "episode_accumulator",
    "identity_counter",
    "posterior_model",
    "qualification_binding",
)

_PUBLIC_FORBIDDEN_KEY_FRAGMENTS: Final = (
    "counterfactual",
    "hidden_regime",
    "hmac",
    "private_salt",
    "schedule_position",
    "theta",
)
_MAX_IMPLEMENTATION_MEMBER_BYTES: Final = 16 * 1024 * 1024


class ContractError(ValueError):
    """A Q1 protocol, artifact, binding, or privacy contract was violated."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One exact selected-source implementation member."""

    relative_path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    """Encode finite, sorted, compact UTF-8 JSON."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    """Digest the canonical JSON encoding of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return lowercase SHA-256 for immutable bytes."""

    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key is forbidden")
        value[key] = item
    return value


def load_json(path: Path) -> object:
    """Decode one finite JSON document."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_pairs,
            parse_constant=lambda token: _reject_nonfinite(token),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(f"cannot decode JSON document {path}") from error


def load_canonical_json(path: Path, *, trailing_newline: bool = True) -> object:
    """Decode and require the exact compact canonical representation."""

    raw = path.read_bytes()
    value = load_json(path)
    expected = canonical_json_bytes(value, newline=trailing_newline)
    if raw != expected:
        raise ContractError(f"{path} is not canonical JSON")
    return value


def protocol_document(path: Path = Q1_PROTOCOL_PATH) -> Mapping[str, object]:
    """Load the exact successor protocol and verify its identity fields."""

    value = load_json(path)
    if not isinstance(value, dict):
        raise ContractError("Q1 protocol must be a JSON object")
    experiment = value.get("experiment")
    if not isinstance(experiment, dict):
        raise ContractError("Q1 protocol has no experiment object")
    if experiment.get("protocol_version") != Q1_PROTOCOL_VERSION:
        raise ContractError("Q1 protocol version mismatch")
    if value.get("schema") != "prospect.wm002.active-acquisition.q1-protocol.v1":
        raise ContractError("Q1 protocol schema mismatch")
    return value


@cache
def _schema_documents_cached() -> tuple[tuple[str, Mapping[str, object]], ...]:
    """Load and metaschema-check frozen schemas once per interpreter."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise ContractError("Q1 qualification requires the runtime jsonschema extra") from error

    documents: dict[str, Mapping[str, object]] = {}
    for name, path in sorted(Q1_SCHEMA_PATHS.items()):
        value = load_json(path)
        if not isinstance(value, dict):
            raise ContractError(f"{name} schema is not an object")
        Draft202012Validator.check_schema(value)
        documents[name] = value
    return tuple(documents.items())


def schema_documents() -> dict[str, Mapping[str, object]]:
    """Return the cached, metaschema-checked artifact-schema inventory."""

    return dict(_schema_documents_cached())


@cache
def _compiled_validator(name: str) -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise ContractError("Q1 artifact validation requires the runtime jsonschema extra") from error
    try:
        schema = dict(_schema_documents_cached())[name]
    except KeyError as error:
        raise ContractError(f"unknown Q1 artifact schema {name!r}") from error
    return Draft202012Validator(schema)


def validate_artifact(name: str, value: object) -> None:
    """Validate one artifact value against its cached strict JSON Schema."""

    errors = sorted(_compiled_validator(name).iter_errors(value), key=lambda row: list(row.path))
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise ContractError(f"{name} artifact violates schema at {location}: {first.message}")


def implementation_manifest(
    relative_paths: Iterable[str],
    *,
    repository_root: Path | None = None,
) -> tuple[tuple[ManifestEntry, ...], str]:
    """Bind a descriptor-stable sorted source closure without self-reference.

    This binds the selected bytes observed during this read.  It does not bind
    already-loaded bytecode or claim protection from a malicious same-account
    owner capable of coordinated path or metadata manipulation.
    """

    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    paths = tuple(sorted(set(relative_paths)))
    if not paths:
        raise ContractError("Q1 implementation manifest cannot be empty")
    rows: list[ManifestEntry] = []
    root_descriptor, root_signature = _open_manifest_root(root)
    try:
        for relative_path in paths:
            parts = _safe_manifest_parts(relative_path)
            payload = _read_manifest_member(root_descriptor, relative_path, parts)
            rows.append(
                ManifestEntry(
                    relative_path=relative_path,
                    sha256=sha256_bytes(payload),
                    size_bytes=len(payload),
                )
            )
        _require_root_descriptor_stable(root, root_descriptor, root_signature)
    finally:
        os.close(root_descriptor)
    canonical_rows = [row.as_dict() for row in rows]
    return tuple(rows), canonical_sha256(canonical_rows)


def _safe_manifest_parts(relative_path: str) -> tuple[str, ...]:
    candidate = Path(relative_path)
    parts = candidate.parts
    if not relative_path or candidate.is_absolute() or not parts or ".." in parts:
        raise ContractError(f"unsafe implementation path {relative_path!r}")
    return parts


def _required_open_flags() -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not nofollow or not cloexec:
        raise ContractError("descriptor-stable implementation manifests require O_NOFOLLOW and O_CLOEXEC")
    return nofollow, cloexec


def _open_manifest_root(root: Path) -> tuple[int, tuple[int, ...]]:
    nofollow, cloexec = _required_open_flags()
    flags = os.O_RDONLY | os.O_DIRECTORY | nofollow | cloexec
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise ContractError("implementation repository root must be a non-symlink directory") from error
    try:
        metadata = _require_manifest_kind(os.fstat(descriptor), directory=True, label="implementation repository root")
        path_metadata = _require_manifest_kind(
            os.stat(root, follow_symlinks=False),
            directory=True,
            label="implementation repository root",
        )
        signature = _manifest_metadata_signature(metadata)
        if _manifest_metadata_signature(path_metadata) != signature:
            raise ContractError("implementation repository root path differs from its opened descriptor")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, signature


def _read_manifest_member(
    root_descriptor: int,
    relative_path: str,
    parts: tuple[str, ...],
) -> bytes:
    nofollow, cloexec = _required_open_flags()
    opened_descriptors: list[int] = []
    components: list[tuple[int, str, int, bool, tuple[int, ...]]] = []
    parent_descriptor = root_descriptor
    try:
        for index, component in enumerate(parts):
            directory = index < len(parts) - 1
            flags = os.O_RDONLY | nofollow | cloexec
            if directory:
                flags |= os.O_DIRECTORY
            else:
                flags |= os.O_NONBLOCK
            descriptor = os.open(component, flags, dir_fd=parent_descriptor)
            opened_descriptors.append(descriptor)
            label = f"implementation member {relative_path!r}"
            metadata = _require_manifest_kind(os.fstat(descriptor), directory=directory, label=label)
            path_metadata = _require_manifest_kind(
                os.stat(component, dir_fd=parent_descriptor, follow_symlinks=False),
                directory=directory,
                label=label,
            )
            signature = _manifest_metadata_signature(metadata)
            if _manifest_metadata_signature(path_metadata) != signature:
                raise ContractError(f"{label} path differs from its opened descriptor")
            components.append((parent_descriptor, component, descriptor, directory, signature))
            parent_descriptor = descriptor

        final_descriptor = opened_descriptors[-1]
        final_signature = components[-1][4]
        final_size = os.fstat(final_descriptor).st_size
        if final_size > _MAX_IMPLEMENTATION_MEMBER_BYTES:
            raise ContractError(
                f"implementation member {relative_path!r} exceeds the "
                f"{_MAX_IMPLEMENTATION_MEMBER_BYTES}-byte read limit"
            )
        payload = _read_bounded_descriptor(
            final_descriptor,
            limit=_MAX_IMPLEMENTATION_MEMBER_BYTES,
            label=f"implementation member {relative_path!r}",
        )
        final_after = _require_manifest_kind(
            os.fstat(final_descriptor),
            directory=False,
            label=f"implementation member {relative_path!r}",
        )
        if _manifest_metadata_signature(final_after) != final_signature or len(payload) != final_after.st_size:
            raise ContractError(f"implementation member {relative_path!r} changed during its descriptor read")
        _require_manifest_components_stable(relative_path, components)
        return payload
    except ContractError:
        raise
    except OSError as error:
        raise ContractError(f"cannot safely read implementation member {relative_path!r}") from error
    finally:
        for descriptor in reversed(opened_descriptors):
            os.close(descriptor)


def _require_manifest_components_stable(
    relative_path: str,
    components: Iterable[tuple[int, str, int, bool, tuple[int, ...]]],
) -> None:
    label = f"implementation member {relative_path!r}"
    for parent_descriptor, component, descriptor, directory, before_signature in components:
        descriptor_metadata = _require_manifest_kind(os.fstat(descriptor), directory=directory, label=label)
        if _manifest_metadata_signature(descriptor_metadata) != before_signature:
            raise ContractError(f"{label} descriptor signature changed during manifest read")
        path_metadata = _require_manifest_kind(
            os.stat(component, dir_fd=parent_descriptor, follow_symlinks=False),
            directory=directory,
            label=label,
        )
        if _manifest_metadata_signature(path_metadata) != _manifest_metadata_signature(descriptor_metadata):
            raise ContractError(f"{label} path-to-descriptor signature changed during manifest read")


def _require_root_descriptor_stable(
    root: Path,
    descriptor: int,
    before_signature: tuple[int, ...],
) -> None:
    descriptor_metadata = _require_manifest_kind(
        os.fstat(descriptor),
        directory=True,
        label="implementation repository root",
    )
    if _manifest_metadata_signature(descriptor_metadata) != before_signature:
        raise ContractError("implementation repository root descriptor changed during manifest read")
    try:
        path_metadata = _require_manifest_kind(
            os.stat(root, follow_symlinks=False),
            directory=True,
            label="implementation repository root",
        )
    except OSError as error:
        raise ContractError("cannot revalidate implementation repository root") from error
    if _manifest_metadata_signature(path_metadata) != _manifest_metadata_signature(descriptor_metadata):
        raise ContractError("implementation repository root path-to-descriptor signature changed during manifest read")


def _require_manifest_kind(
    metadata: os.stat_result,
    *,
    directory: bool,
    label: str,
) -> os.stat_result:
    if directory:
        if not stat.S_ISDIR(metadata.st_mode):
            raise ContractError(f"{label} has a symlinked or non-directory path component")
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"{label} must be a regular non-symlink file")
        if metadata.st_nlink != 1:
            raise ContractError(f"{label} must have exactly one hard link")
    return metadata


def _read_bounded_descriptor(descriptor: int, *, limit: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(1024 * 1024, limit + 1 - total)):
        total += len(chunk)
        if total > limit:
            raise ContractError(f"{label} exceeds the {limit}-byte read limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _manifest_metadata_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def assert_public_value_safe(
    value: object,
    *,
    private_sentinels: Iterable[object] = (),
) -> None:
    """Reject private-looking keys and recognizable private values recursively."""

    _scan_public_keys(value, path=())
    encoded = canonical_json_bytes(value)
    for sentinel in private_sentinels:
        if sentinel is None:
            continue
        candidates: tuple[bytes, ...]
        if isinstance(sentinel, bytes):
            candidates = (sentinel, sentinel.hex().encode("ascii"))
        else:
            candidates = (str(sentinel).encode("utf-8"),)
        for candidate in candidates:
            if candidate and candidate in encoded:
                raise ContractError("a private sentinel occurs in a public serialization")


def assert_no_sentinel_bytes(
    payloads: Iterable[bytes],
    *,
    private_sentinels: Iterable[object],
) -> None:
    """Scan opaque serialized payloads for exact private sentinel material."""

    candidates: list[bytes] = []
    for sentinel in private_sentinels:
        if sentinel is None:
            continue
        if isinstance(sentinel, bytes):
            candidates.extend((sentinel, sentinel.hex().encode("ascii")))
        else:
            candidates.append(str(sentinel).encode("utf-8"))
    for payload in payloads:
        for candidate in candidates:
            if candidate and candidate in payload:
                raise ContractError("a private sentinel occurs in serialized component bytes")


def _scan_public_keys(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ContractError(f"public mapping at {'/'.join(path) or '<root>'} has a non-string key")
            lowered = key.lower()
            if any(fragment in lowered for fragment in _PUBLIC_FORBIDDEN_KEY_FRAGMENTS):
                raise ContractError(f"private field name {key!r} occurs in public serialization")
            _scan_public_keys(nested, path=(*path, key))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _scan_public_keys(nested, path=(*path, str(index)))


def _reject_nonfinite(token: str) -> object:
    raise ContractError(f"non-finite JSON token {token!r} is forbidden")


__all__ = (
    "ACTION_ORDER",
    "ARM_ORDER",
    "CHECKPOINT_COMPONENTS",
    "ContractError",
    "ManifestEntry",
    "Q0_IMPLEMENTATION_SHA256",
    "Q0_PROTOCOL_SHA256",
    "Q0_REPORT_SHA256",
    "Q1_PROTOCOL_PATH",
    "Q1_PROTOCOL_VERSION",
    "Q1_SCHEMA_PATHS",
    "TERMINAL_ORDER",
    "assert_no_sentinel_bytes",
    "assert_public_value_safe",
    "canonical_json_bytes",
    "canonical_sha256",
    "implementation_manifest",
    "load_canonical_json",
    "load_json",
    "protocol_document",
    "schema_documents",
    "sha256_bytes",
    "validate_artifact",
)
