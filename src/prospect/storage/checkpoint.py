"""Atomic, integrity-checked checkpoint bundles of opaque component bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from prospect.domain import TimePoint

_FORMAT = "prospect-checkpoint"
_SCHEMA_VERSION = 1
_MANIFEST_PATH = "manifest.json"
_COMPONENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CheckpointIntegrityError(RuntimeError):
    """Checkpoint bytes fail structural or cryptographic integrity checks."""


class CheckpointFormatError(CheckpointIntegrityError):
    """Checkpoint manifest or archive structure is unsupported or malformed."""


def _require_identifier(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a nonempty identifier")


def _require_component_name(name: str) -> None:
    if _COMPONENT_NAME.fullmatch(name) is None:
        raise ValueError(
            "component name must start with an alphanumeric and contain only alphanumerics, dot, underscore, or hyphen"
        )


def _normalize_string_map(label: str, values: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if values is None:
        return ()
    normalized: list[tuple[str, str]] = []
    for key, value in values.items():
        _require_identifier(f"{label} key", key)
        if not isinstance(value, str):
            raise TypeError(f"{label} values must be strings")
        normalized.append((key, value))
    normalized.sort()
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{label} contains duplicate keys")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class CheckpointComponent:
    """Opaque state bytes supplied by one versioned runtime component."""

    name: str
    version: str
    payload: bytes
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        _require_component_name(self.name)
        _require_identifier("component version", self.version)
        _require_identifier("component media_type", self.media_type)
        if not isinstance(self.payload, bytes):
            raise TypeError("checkpoint component payload must be bytes")


@dataclass(frozen=True, slots=True)
class CheckpointComponentManifest:
    """Integrity and compatibility metadata for one component blob."""

    name: str
    version: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _require_component_name(self.name)
        _require_identifier("component version", self.version)
        _require_identifier("component path", self.path)
        _require_identifier("component media_type", self.media_type)
        if self.path != f"components/{self.name}.bin":
            raise CheckpointFormatError(f"component {self.name!r} has noncanonical path {self.path!r}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise CheckpointFormatError("component sha256 must be lowercase hexadecimal")
        if self.size_bytes < 0:
            raise CheckpointFormatError("component size_bytes must be nonnegative")


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Canonical description of a complete checkpoint bundle."""

    checkpoint_id: str
    agent_id: str
    created_at: TimePoint
    versions: tuple[tuple[str, str], ...]
    metadata: tuple[tuple[str, str], ...]
    components: tuple[CheckpointComponentManifest, ...]
    format: str = _FORMAT
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier("checkpoint_id", self.checkpoint_id)
        _require_identifier("agent_id", self.agent_id)
        if self.format != _FORMAT:
            raise CheckpointFormatError(f"unsupported checkpoint format {self.format!r}")
        if self.schema_version != _SCHEMA_VERSION:
            raise CheckpointFormatError(f"unsupported checkpoint schema version {self.schema_version}")
        names = tuple(component.name for component in self.components)
        if not names:
            raise CheckpointFormatError("checkpoint requires at least one component")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise CheckpointFormatError("checkpoint components must have unique, sorted names")
        for label, values in (
            ("versions", self.versions),
            ("metadata", self.metadata),
        ):
            keys = tuple(key for key, _ in values)
            if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
                raise CheckpointFormatError(f"checkpoint {label} must have unique, sorted keys")
            for key, value in values:
                _require_identifier(f"{label} key", key)
                if not isinstance(value, str):
                    raise CheckpointFormatError(f"checkpoint {label} values must be strings")
                if label == "versions":
                    _require_identifier("versions value", value)

    @property
    def version_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.versions))

    @property
    def metadata_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.metadata))


Restorer = Callable[[bytes, CheckpointComponentManifest], None]


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A fully verified checkpoint ready to initialize fresh components."""

    manifest: CheckpointManifest
    payloads: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.payloads)
        expected = tuple(component.name for component in self.manifest.components)
        if names != expected:
            raise CheckpointIntegrityError("loaded payload names do not match the checkpoint manifest")

    def payload(self, component_name: str) -> bytes:
        _require_component_name(component_name)
        for name, payload in self.payloads:
            if name == component_name:
                return payload
        raise KeyError(f"checkpoint has no component {component_name!r}")

    def restore(
        self,
        restorers: Mapping[str, Restorer],
        *,
        require_all: bool = True,
    ) -> None:
        """Restore fresh components after preflighting the complete callback set.

        All bundle bytes have already passed integrity checks.  Callback coverage is
        checked before the first callback runs, preventing a missing component from
        causing a partially initialized restart.
        """

        expected = {component.name for component in self.manifest.components}
        supplied = set(restorers)
        unknown = supplied - expected
        if unknown:
            raise KeyError(f"restorers supplied for unknown components: {sorted(unknown)}")
        missing = expected - supplied
        if require_all and missing:
            raise KeyError(f"missing restorers for checkpoint components: {sorted(missing)}")

        entries = {component.name: component for component in self.manifest.components}
        for name, payload in self.payloads:
            restorer = restorers.get(name)
            if restorer is not None:
                restorer(payload, entries[name])


class CheckpointCoordinator:
    """Write and load deterministic single-file checkpoint bundles atomically."""

    def __init__(
        self,
        *,
        max_component_bytes: int = 1 << 30,
        max_total_bytes: int = 4 << 30,
    ) -> None:
        if max_component_bytes < 1 or max_total_bytes < 1:
            raise ValueError("checkpoint byte limits must be positive")
        if max_component_bytes > max_total_bytes:
            raise ValueError("component byte limit cannot exceed total byte limit")
        self._max_component_bytes = max_component_bytes
        self._max_total_bytes = max_total_bytes

    def save(
        self,
        path: str | Path,
        *,
        checkpoint_id: str,
        agent_id: str,
        created_at: TimePoint,
        components: Mapping[str, CheckpointComponent],
        versions: Mapping[str, str],
        metadata: Mapping[str, str] | None = None,
    ) -> CheckpointManifest:
        """Atomically replace ``path`` with a complete, deterministic checkpoint."""

        destination = Path(path)
        _require_identifier("checkpoint_id", checkpoint_id)
        _require_identifier("agent_id", agent_id)
        if not components:
            raise ValueError("checkpoint requires at least one component")

        ordered_components: list[CheckpointComponent] = []
        total_bytes = 0
        for key, component in sorted(components.items()):
            _require_component_name(key)
            if key != component.name:
                raise ValueError(f"component mapping key {key!r} does not match name {component.name!r}")
            size = len(component.payload)
            if size > self._max_component_bytes:
                raise ValueError(f"component {component.name!r} exceeds checkpoint byte limit")
            total_bytes += size
            ordered_components.append(component)
        if total_bytes > self._max_total_bytes:
            raise ValueError("checkpoint exceeds total byte limit")

        entries = tuple(
            CheckpointComponentManifest(
                name=component.name,
                version=component.version,
                path=f"components/{component.name}.bin",
                sha256=hashlib.sha256(component.payload).hexdigest(),
                size_bytes=len(component.payload),
                media_type=component.media_type,
            )
            for component in ordered_components
        )
        manifest = CheckpointManifest(
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            created_at=created_at,
            versions=_normalize_string_map("versions", versions),
            metadata=_normalize_string_map("metadata", metadata),
            components=entries,
        )
        manifest_bytes = _encode_manifest(manifest)

        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                _write_entry(archive, _MANIFEST_PATH, manifest_bytes)
                for component, entry in zip(ordered_components, entries, strict=True):
                    _write_entry(archive, entry.path, component.payload)
            with temporary.open("rb") as checkpoint_file:
                os.fsync(checkpoint_file.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return manifest

    def load(
        self,
        path: str | Path,
        *,
        expected_agent_id: str | None = None,
    ) -> LoadedCheckpoint:
        """Read all component bytes and reject malformed or corrupted bundles."""

        source = Path(path)
        try:
            with zipfile.ZipFile(source, mode="r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise CheckpointFormatError("checkpoint archive contains duplicate member paths")
                if _MANIFEST_PATH not in names:
                    raise CheckpointFormatError("checkpoint archive has no manifest")
                manifest_info = archive.getinfo(_MANIFEST_PATH)
                if manifest_info.file_size > 1 << 20:
                    raise CheckpointFormatError("checkpoint manifest is unreasonably large")
                raw_manifest = archive.read(manifest_info)
                manifest = _decode_manifest(raw_manifest)
                if raw_manifest != _encode_manifest(manifest):
                    raise CheckpointFormatError("checkpoint manifest is not canonical JSON")
                if expected_agent_id is not None and manifest.agent_id != expected_agent_id:
                    raise CheckpointIntegrityError(
                        f"checkpoint belongs to agent {manifest.agent_id!r}, not {expected_agent_id!r}"
                    )

                expected_paths = {
                    _MANIFEST_PATH,
                    *(component.path for component in manifest.components),
                }
                if set(names) != expected_paths:
                    raise CheckpointFormatError("checkpoint archive members do not match its manifest")
                total_bytes = 0
                payloads: list[tuple[str, bytes]] = []
                for component in manifest.components:
                    info = archive.getinfo(component.path)
                    if info.file_size != component.size_bytes:
                        raise CheckpointIntegrityError(f"component {component.name!r} size disagrees with manifest")
                    if info.file_size > self._max_component_bytes:
                        raise CheckpointIntegrityError(f"component {component.name!r} exceeds checkpoint byte limit")
                    total_bytes += info.file_size
                    if total_bytes > self._max_total_bytes:
                        raise CheckpointIntegrityError("checkpoint exceeds total byte limit")
                    payload = archive.read(info)
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest != component.sha256:
                        raise CheckpointIntegrityError(f"component {component.name!r} failed sha256 verification")
                    payloads.append((component.name, payload))
        except zipfile.BadZipFile as error:
            raise CheckpointFormatError("checkpoint is not a valid ZIP bundle") from error
        return LoadedCheckpoint(manifest=manifest, payloads=tuple(payloads))

    def restore(
        self,
        path: str | Path,
        restorers: Mapping[str, Restorer],
        *,
        expected_agent_id: str | None = None,
        require_all: bool = True,
    ) -> CheckpointManifest:
        """Verify a bundle, restore fresh components, and return its manifest."""

        loaded = self.load(path, expected_agent_id=expected_agent_id)
        loaded.restore(restorers, require_all=require_all)
        return loaded.manifest


def _manifest_dict(manifest: CheckpointManifest) -> dict[str, object]:
    return {
        "agent_id": manifest.agent_id,
        "checkpoint_id": manifest.checkpoint_id,
        "components": [
            {
                "media_type": component.media_type,
                "name": component.name,
                "path": component.path,
                "sha256": component.sha256,
                "size_bytes": component.size_bytes,
                "version": component.version,
            }
            for component in manifest.components
        ],
        "created_at": {
            "clock_id": manifest.created_at.clock_id,
            "tick": manifest.created_at.tick,
        },
        "format": manifest.format,
        "metadata": dict(manifest.metadata),
        "schema_version": manifest.schema_version,
        "versions": dict(manifest.versions),
    }


def _encode_manifest(manifest: CheckpointManifest) -> bytes:
    try:
        encoded = json.dumps(
            _manifest_dict(manifest),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointFormatError("checkpoint manifest is not JSON encodable") from error
    return encoded.encode("utf-8")


def _decode_manifest(payload: bytes) -> CheckpointManifest:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointFormatError("checkpoint manifest is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise CheckpointFormatError("checkpoint manifest root must be an object")
    expected_keys = {
        "agent_id",
        "checkpoint_id",
        "components",
        "created_at",
        "format",
        "metadata",
        "schema_version",
        "versions",
    }
    if set(decoded) != expected_keys:
        raise CheckpointFormatError("checkpoint manifest has unexpected fields")
    components_value = decoded["components"]
    time_value = decoded["created_at"]
    versions_value = decoded["versions"]
    metadata_value = decoded["metadata"]
    if not isinstance(components_value, list):
        raise CheckpointFormatError("checkpoint components must be an array")
    if not isinstance(time_value, dict) or set(time_value) != {"clock_id", "tick"}:
        raise CheckpointFormatError("checkpoint created_at is malformed")
    if not isinstance(versions_value, dict) or not isinstance(metadata_value, dict):
        raise CheckpointFormatError("checkpoint versions and metadata must be objects")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in versions_value.items()):
        raise CheckpointFormatError("checkpoint versions must map strings to strings")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata_value.items()):
        raise CheckpointFormatError("checkpoint metadata must map strings to strings")
    if not isinstance(time_value["tick"], int) or isinstance(time_value["tick"], bool):
        raise CheckpointFormatError("checkpoint time tick must be an integer")
    if not isinstance(time_value["clock_id"], str):
        raise CheckpointFormatError("checkpoint clock_id must be a string")

    components: list[CheckpointComponentManifest] = []
    component_keys = {"media_type", "name", "path", "sha256", "size_bytes", "version"}
    for item in components_value:
        if not isinstance(item, dict) or set(item) != component_keys:
            raise CheckpointFormatError("checkpoint component entry is malformed")
        if not all(isinstance(item[key], str) for key in ("media_type", "name", "path", "sha256", "version")):
            raise CheckpointFormatError("checkpoint component string field is malformed")
        if not isinstance(item["size_bytes"], int) or isinstance(item["size_bytes"], bool):
            raise CheckpointFormatError("checkpoint component size must be an integer")
        try:
            components.append(
                CheckpointComponentManifest(
                    name=item["name"],
                    version=item["version"],
                    path=item["path"],
                    sha256=item["sha256"],
                    size_bytes=item["size_bytes"],
                    media_type=item["media_type"],
                )
            )
        except (TypeError, ValueError) as error:
            raise CheckpointFormatError("checkpoint component violates its schema") from error

    string_fields = ("agent_id", "checkpoint_id", "format")
    if not all(isinstance(decoded[field], str) for field in string_fields):
        raise CheckpointFormatError("checkpoint identifier field is malformed")
    if not isinstance(decoded["schema_version"], int) or isinstance(decoded["schema_version"], bool):
        raise CheckpointFormatError("checkpoint schema version must be an integer")
    try:
        return CheckpointManifest(
            checkpoint_id=decoded["checkpoint_id"],
            agent_id=decoded["agent_id"],
            created_at=TimePoint(
                tick=time_value["tick"],
                clock_id=time_value["clock_id"],
            ),
            versions=tuple(sorted(versions_value.items())),
            metadata=tuple(sorted(metadata_value.items())),
            components=tuple(components),
            format=decoded["format"],
            schema_version=decoded["schema_version"],
        )
    except (TypeError, ValueError) as error:
        raise CheckpointFormatError("checkpoint manifest violates its schema") from error


def _write_entry(archive: zipfile.ZipFile, path: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    archive.writestr(info, payload)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "CheckpointComponent",
    "CheckpointComponentManifest",
    "CheckpointCoordinator",
    "CheckpointFormatError",
    "CheckpointIntegrityError",
    "CheckpointManifest",
    "LoadedCheckpoint",
)
