"""Component-complete WM-001 checkpoints and safe RNG state codecs.

The project-wide :class:`prospect.storage.CheckpointCoordinator` owns atomic
archive construction and byte-level integrity.  This module narrows that generic
primitive to the exact checkpoint contract sealed by WM-001:

* all and only the fifteen declared stateful components are present;
* every component records a logical version, media type, digest, byte length,
  and predecessor digest;
* one aggregate digest binds the complete logical manifest; and
* restore callback coverage is checked before the first callback is invoked.

Non-RNG payloads stay opaque.  RNG helpers use canonical JSON and raw byte
encodings; no pickle or executable deserialization is used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prospect.domain import TimePoint
from prospect.storage import (
    CheckpointComponent,
    CheckpointCoordinator,
    CheckpointIntegrityError,
    CheckpointManifest,
)

CHECKPOINT_SCHEMA = "prospect.world-model-lifecycle.checkpoint.v1"
RNG_SCHEMA_VERSION = 1

CANONICAL_COMPONENT_IDS = (
    "world_model",
    "optimizer",
    "model_version_ledger",
    "experience_store",
    "replay_index",
    "replay_sampling_history",
    "update_receipts",
    "agent_runtime",
    "scaling_configuration",
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_accelerator_rng",
    "collection_rng",
    "planner_rng",
)

RNG_COMPONENT_IDS = (
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_accelerator_rng",
    "collection_rng",
    "planner_rng",
)

OPAQUE_COMPONENT_IDS = tuple(
    component_id for component_id in CANONICAL_COMPONENT_IDS if component_id not in RNG_COMPONENT_IDS
)

_BOUNDARY = "episode_complete"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_SCHEMA = "wm001.schema"
_METADATA_BOUNDARY = "wm001.boundary"
_METADATA_AGGREGATE = "wm001.aggregate_manifest_sha256"
_METADATA_ANCESTRY_PREFIX = "wm001.predecessor."
_NO_PREDECESSOR = "none"
_RNG_MEDIA_TYPE = "application/vnd.prospect.rng-state+json"


class WMCheckpointError(CheckpointIntegrityError):
    """A WM-001 checkpoint violates the sealed component contract."""


class RNGStateError(WMCheckpointError):
    """An RNG snapshot is malformed or incompatible with this interpreter."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON value using the canonical WM-001 hashing rules."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical-JSON encodable") from error
    return text.encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _decode_canonical_json(payload: bytes, *, label: str) -> object:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RNGStateError(f"{label} is not valid UTF-8 JSON") from error
    try:
        canonical = canonical_json_bytes(value)
    except ValueError as error:
        raise RNGStateError(f"{label} is not canonical-JSON encodable") from error
    if canonical != payload:
        raise RNGStateError(f"{label} is not canonical JSON")
    return value


def _require_exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RNGStateError(f"{label} has an invalid field set")
    if not all(isinstance(key, str) for key in value):
        raise RNGStateError(f"{label} object keys must be strings")
    return value


def _require_schema(value: dict[str, Any], schema: str, *, label: str) -> None:
    if value.get("schema") != schema or value.get("schema_version") != RNG_SCHEMA_VERSION:
        raise RNGStateError(f"{label} has an unsupported schema")


def _require_sha256(value: str | None, *, label: str) -> None:
    if value is not None and _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest or None")


@dataclass(frozen=True, slots=True)
class ComponentPayload:
    """Opaque bytes and custody metadata supplied by the WM-001 harness."""

    component_id: str
    logical_version: str
    payload: bytes
    media_type: str = "application/octet-stream"
    predecessor_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.component_id not in CANONICAL_COMPONENT_IDS:
            raise ValueError(f"unknown WM-001 checkpoint component {self.component_id!r}")
        if not self.logical_version or not self.logical_version.strip():
            raise ValueError("component logical_version must be nonempty")
        if not isinstance(self.payload, bytes):
            raise TypeError("component payload must be bytes")
        if not self.media_type or not self.media_type.strip():
            raise ValueError("component media_type must be nonempty")
        _require_sha256(
            self.predecessor_sha256,
            label=f"{self.component_id} predecessor_sha256",
        )


@dataclass(frozen=True, slots=True)
class ComponentDigest:
    """Raw-result-compatible digest record for one checkpoint component."""

    checkpoint_id: str
    component_id: str
    logical_version: str
    media_type: str
    bytes: int
    sha256: str
    predecessor_sha256: str | None

    def __post_init__(self) -> None:
        if self.component_id not in CANONICAL_COMPONENT_IDS:
            raise WMCheckpointError(f"unknown manifest component {self.component_id!r}")
        if self.bytes < 0:
            raise WMCheckpointError("component byte length must be nonnegative")
        if _SHA256.fullmatch(self.sha256) is None:
            raise WMCheckpointError("component sha256 must be lowercase hexadecimal")
        try:
            _require_sha256(
                self.predecessor_sha256,
                label=f"{self.component_id} predecessor_sha256",
            )
        except ValueError as error:
            raise WMCheckpointError(str(error)) from error

    def as_dict(self) -> dict[str, object]:
        """Return the exact record shape declared by ``raw-result.schema.json``."""

        return {
            "bytes": self.bytes,
            "checkpoint_id": self.checkpoint_id,
            "component_id": self.component_id,
            "logical_version": self.logical_version,
            "media_type": self.media_type,
            "predecessor_sha256": self.predecessor_sha256,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class WMCheckpointReport:
    """Logical manifest report bound by ``manifest_sha256``."""

    checkpoint_id: str
    agent_id: str
    created_at: TimePoint
    components: tuple[ComponentDigest, ...]
    manifest_sha256: str
    archive_manifest: CheckpointManifest

    def __post_init__(self) -> None:
        component_ids = tuple(component.component_id for component in self.components)
        if component_ids != CANONICAL_COMPONENT_IDS:
            raise WMCheckpointError("checkpoint report components must match the canonical order exactly")
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise WMCheckpointError("aggregate manifest sha256 is malformed")
        if self.manifest_sha256 != canonical_json_sha256(self.manifest_body()):
            raise WMCheckpointError("aggregate manifest sha256 does not match the logical manifest")

    def manifest_body(self) -> dict[str, object]:
        """Return the canonical logical manifest body whose digest is reported."""

        return _manifest_body(
            checkpoint_id=self.checkpoint_id,
            agent_id=self.agent_id,
            created_at=self.created_at,
            components=self.components,
        )

    def component_rows(self) -> tuple[dict[str, object], ...]:
        """Return raw-result-compatible component rows in protocol order."""

        return tuple(component.as_dict() for component in self.components)


ComponentRestorer = Callable[[bytes, ComponentDigest], None]


@dataclass(frozen=True, slots=True)
class LoadedWMCheckpoint:
    """A fully verified WM-001 checkpoint consumable by a fresh interpreter."""

    report: WMCheckpointReport
    payloads: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if tuple(component_id for component_id, _ in self.payloads) != CANONICAL_COMPONENT_IDS:
            raise WMCheckpointError("loaded payloads do not match the canonical component order")

    def payload(self, component_id: str) -> bytes:
        """Return verified opaque bytes for one canonical component."""

        if component_id not in CANONICAL_COMPONENT_IDS:
            raise KeyError(f"unknown WM-001 component {component_id!r}")
        return dict(self.payloads)[component_id]

    def restore(self, restorers: Mapping[str, ComponentRestorer]) -> None:
        """Invoke an exact callback set after complete coverage preflight.

        The callback mapping must contain all and only the fifteen canonical IDs.
        Missing and extra IDs are rejected before any callback can mutate state.
        """

        _require_exact_component_set(restorers, label="restorers")
        records = {component.component_id: component for component in self.report.components}
        for component_id, payload in self.payloads:
            restorers[component_id](payload, records[component_id])


def _require_exact_component_set(values: Mapping[str, object], *, label: str) -> None:
    supplied = set(values)
    expected = set(CANONICAL_COMPONENT_IDS)
    missing = expected - supplied
    extra = supplied - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise WMCheckpointError(
            f"{label} must contain all and only the canonical WM-001 components ({', '.join(details)})"
        )


def _manifest_body(
    *,
    checkpoint_id: str,
    agent_id: str,
    created_at: TimePoint,
    components: tuple[ComponentDigest, ...],
) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "boundary": _BOUNDARY,
        "checkpoint_id": checkpoint_id,
        "components": [component.as_dict() for component in components],
        "created_at": {
            "clock_id": created_at.clock_id,
            "tick": created_at.tick,
        },
        "schema": CHECKPOINT_SCHEMA,
    }


def _component_digests(
    *,
    checkpoint_id: str,
    components: Mapping[str, ComponentPayload],
) -> tuple[ComponentDigest, ...]:
    return tuple(
        ComponentDigest(
            checkpoint_id=checkpoint_id,
            component_id=component_id,
            logical_version=components[component_id].logical_version,
            media_type=components[component_id].media_type,
            bytes=len(components[component_id].payload),
            sha256=hashlib.sha256(components[component_id].payload).hexdigest(),
            predecessor_sha256=components[component_id].predecessor_sha256,
        )
        for component_id in CANONICAL_COMPONENT_IDS
    )


def _checkpoint_metadata(
    components: tuple[ComponentDigest, ...],
    aggregate_digest: str,
    metadata: Mapping[str, str] | None,
) -> dict[str, str]:
    result = dict(metadata or {})
    reserved = {
        _METADATA_SCHEMA,
        _METADATA_BOUNDARY,
        _METADATA_AGGREGATE,
        *(f"{_METADATA_ANCESTRY_PREFIX}{component_id}" for component_id in CANONICAL_COMPONENT_IDS),
    }
    collisions = reserved & set(result)
    if collisions:
        raise ValueError(f"checkpoint metadata uses reserved keys: {sorted(collisions)}")
    result[_METADATA_SCHEMA] = CHECKPOINT_SCHEMA
    result[_METADATA_BOUNDARY] = _BOUNDARY
    result[_METADATA_AGGREGATE] = aggregate_digest
    for component in components:
        result[f"{_METADATA_ANCESTRY_PREFIX}{component.component_id}"] = (
            component.predecessor_sha256 if component.predecessor_sha256 is not None else _NO_PREDECESSOR
        )
    return result


def save_checkpoint(
    path: str | Path,
    *,
    checkpoint_id: str,
    agent_id: str,
    created_at: TimePoint,
    components: Mapping[str, ComponentPayload],
    versions: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
    coordinator: CheckpointCoordinator | None = None,
) -> WMCheckpointReport:
    """Atomically save an exact, component-complete WM-001 checkpoint."""

    _require_exact_component_set(components, label="components")
    for component_id, component in components.items():
        if component.component_id != component_id:
            raise ValueError(
                f"component mapping key {component_id!r} does not match payload ID {component.component_id!r}"
            )

    records = _component_digests(
        checkpoint_id=checkpoint_id,
        components=components,
    )
    body = _manifest_body(
        checkpoint_id=checkpoint_id,
        agent_id=agent_id,
        created_at=created_at,
        components=records,
    )
    aggregate_digest = canonical_json_sha256(body)
    checkpoint_metadata = _checkpoint_metadata(records, aggregate_digest, metadata)
    archive_components = {
        component_id: CheckpointComponent(
            name=component_id,
            version=component.logical_version,
            payload=component.payload,
            media_type=component.media_type,
        )
        for component_id, component in components.items()
    }
    supplied_versions = dict(versions or {})
    if "wm001_checkpoint" in supplied_versions:
        raise ValueError("versions uses reserved key 'wm001_checkpoint'")
    archive_coordinator = coordinator or CheckpointCoordinator()
    archive_manifest = archive_coordinator.save(
        path,
        checkpoint_id=checkpoint_id,
        agent_id=agent_id,
        created_at=created_at,
        components=archive_components,
        versions={**supplied_versions, "wm001_checkpoint": CHECKPOINT_SCHEMA},
        metadata=checkpoint_metadata,
    )
    return WMCheckpointReport(
        checkpoint_id=checkpoint_id,
        agent_id=agent_id,
        created_at=created_at,
        components=records,
        manifest_sha256=aggregate_digest,
        archive_manifest=archive_manifest,
    )


def _predecessors_from_metadata(
    metadata: Mapping[str, str],
) -> dict[str, str | None]:
    if metadata.get(_METADATA_SCHEMA) != CHECKPOINT_SCHEMA:
        raise WMCheckpointError("checkpoint has a missing or unsupported WM-001 schema")
    if metadata.get(_METADATA_BOUNDARY) != _BOUNDARY:
        raise WMCheckpointError("checkpoint was not captured at the sealed episode boundary")
    result: dict[str, str | None] = {}
    for component_id in CANONICAL_COMPONENT_IDS:
        key = f"{_METADATA_ANCESTRY_PREFIX}{component_id}"
        if key not in metadata:
            raise WMCheckpointError(f"checkpoint is missing ancestry for component {component_id!r}")
        value = metadata[key]
        if value == _NO_PREDECESSOR:
            result[component_id] = None
        elif _SHA256.fullmatch(value) is not None:
            result[component_id] = value
        else:
            raise WMCheckpointError(f"checkpoint ancestry for component {component_id!r} is malformed")
    aggregate = metadata.get(_METADATA_AGGREGATE)
    if aggregate is None or _SHA256.fullmatch(aggregate) is None:
        raise WMCheckpointError("checkpoint has no valid aggregate manifest digest")
    return result


def load_checkpoint(
    path: str | Path,
    *,
    expected_agent_id: str | None = None,
    coordinator: CheckpointCoordinator | None = None,
) -> LoadedWMCheckpoint:
    """Load and completely verify a WM-001 bundle without restoring state."""

    archive_coordinator = coordinator or CheckpointCoordinator()
    loaded = archive_coordinator.load(path, expected_agent_id=expected_agent_id)

    archive_entries = {component.name: component for component in loaded.manifest.components}
    _require_exact_component_set(archive_entries, label="archive components")
    loaded_payloads = dict(loaded.payloads)
    _require_exact_component_set(loaded_payloads, label="archive payloads")
    if loaded.manifest.version_map.get("wm001_checkpoint") != CHECKPOINT_SCHEMA:
        raise WMCheckpointError("checkpoint version ledger does not bind the WM-001 schema")

    predecessors = _predecessors_from_metadata(loaded.manifest.metadata_map)
    records = tuple(
        ComponentDigest(
            checkpoint_id=loaded.manifest.checkpoint_id,
            component_id=component_id,
            logical_version=archive_entries[component_id].version,
            media_type=archive_entries[component_id].media_type,
            bytes=archive_entries[component_id].size_bytes,
            sha256=archive_entries[component_id].sha256,
            predecessor_sha256=predecessors[component_id],
        )
        for component_id in CANONICAL_COMPONENT_IDS
    )
    body = _manifest_body(
        checkpoint_id=loaded.manifest.checkpoint_id,
        agent_id=loaded.manifest.agent_id,
        created_at=loaded.manifest.created_at,
        components=records,
    )
    aggregate_digest = canonical_json_sha256(body)
    if aggregate_digest != loaded.manifest.metadata_map[_METADATA_AGGREGATE]:
        raise WMCheckpointError("aggregate manifest sha256 verification failed")
    report = WMCheckpointReport(
        checkpoint_id=loaded.manifest.checkpoint_id,
        agent_id=loaded.manifest.agent_id,
        created_at=loaded.manifest.created_at,
        components=records,
        manifest_sha256=aggregate_digest,
        archive_manifest=loaded.manifest,
    )
    payloads = tuple((component_id, loaded_payloads[component_id]) for component_id in CANONICAL_COMPONENT_IDS)
    return LoadedWMCheckpoint(report=report, payloads=payloads)


def snapshot_python_rng(rng: random.Random | None = None) -> bytes:
    """Serialize Python's global RNG or one explicit ``random.Random`` safely."""

    state = rng.getstate() if rng is not None else random.getstate()
    version, internal_state, gaussian_cache = state
    value = {
        "gaussian_cache": gaussian_cache,
        "internal_state": list(internal_state),
        "python_version": platform.python_version(),
        "random_state_version": version,
        "schema": "prospect.rng.python",
        "schema_version": RNG_SCHEMA_VERSION,
    }
    return canonical_json_bytes(value)


def _decode_python_rng(payload: bytes) -> tuple[int, tuple[int, ...], float | None]:
    value = _require_exact_keys(
        _decode_canonical_json(payload, label="Python RNG state"),
        {
            "gaussian_cache",
            "internal_state",
            "python_version",
            "random_state_version",
            "schema",
            "schema_version",
        },
        label="Python RNG state",
    )
    _require_schema(value, "prospect.rng.python", label="Python RNG state")
    if value["python_version"] != platform.python_version():
        raise RNGStateError("Python RNG state was produced by a different Python version")
    version = value["random_state_version"]
    internal = value["internal_state"]
    gaussian_cache = value["gaussian_cache"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise RNGStateError("Python RNG state version must be an integer")
    if (
        not isinstance(internal, list)
        or not internal
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in internal)
    ):
        raise RNGStateError("Python RNG internal state must be a nonempty integer array")
    if gaussian_cache is not None and not isinstance(gaussian_cache, (int, float)):
        raise RNGStateError("Python RNG Gaussian cache must be numeric or null")
    state = (
        version,
        tuple(internal),
        None if gaussian_cache is None else float(gaussian_cache),
    )
    probe = random.Random()
    try:
        probe.setstate(state)
    except (TypeError, ValueError) as error:
        raise RNGStateError("Python RNG state is invalid") from error
    return state


def restore_python_rng(payload: bytes, rng: random.Random | None = None) -> None:
    """Restore a safe Python RNG snapshot after full structural validation."""

    state = _decode_python_rng(payload)
    if rng is None:
        random.setstate(state)
    else:
        rng.setstate(state)


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise RNGStateError("NumPy is required to consume this RNG state") from error
    return numpy


def snapshot_numpy_rng() -> bytes:
    """Serialize NumPy's legacy process-global RNG without pickle."""

    numpy = _import_numpy()
    algorithm, keys, position, has_gaussian, cached_gaussian = numpy.random.get_state(legacy=True)
    value = {
        "algorithm": algorithm,
        "cached_gaussian": float(cached_gaussian),
        "has_gaussian": int(has_gaussian),
        "keys": [int(item) for item in keys.tolist()],
        "numpy_version": numpy.__version__,
        "position": int(position),
        "schema": "prospect.rng.numpy-legacy",
        "schema_version": RNG_SCHEMA_VERSION,
    }
    return canonical_json_bytes(value)


def _decode_numpy_rng(payload: bytes) -> tuple[str, Any, int, int, float]:
    numpy = _import_numpy()
    value = _require_exact_keys(
        _decode_canonical_json(payload, label="NumPy RNG state"),
        {
            "algorithm",
            "cached_gaussian",
            "has_gaussian",
            "keys",
            "numpy_version",
            "position",
            "schema",
            "schema_version",
        },
        label="NumPy RNG state",
    )
    _require_schema(value, "prospect.rng.numpy-legacy", label="NumPy RNG state")
    if value["numpy_version"] != numpy.__version__:
        raise RNGStateError("NumPy RNG state was produced by a different NumPy version")
    if value["algorithm"] != "MT19937":
        raise RNGStateError("unsupported NumPy process-global RNG algorithm")
    keys = value["keys"]
    if (
        not isinstance(keys, list)
        or len(keys) != 624
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in keys)
    ):
        raise RNGStateError("NumPy RNG key state must contain 624 integers")
    if any(item < 0 or item > 0xFFFFFFFF for item in keys):
        raise RNGStateError("NumPy RNG key state contains a non-uint32 value")
    position = value["position"]
    has_gaussian = value["has_gaussian"]
    cached_gaussian = value["cached_gaussian"]
    if not isinstance(position, int) or isinstance(position, bool):
        raise RNGStateError("NumPy RNG position must be an integer")
    if has_gaussian not in (0, 1):
        raise RNGStateError("NumPy RNG has_gaussian must be zero or one")
    if not isinstance(cached_gaussian, (int, float)):
        raise RNGStateError("NumPy RNG cached_gaussian must be numeric")
    state = (
        "MT19937",
        numpy.asarray(keys, dtype=numpy.uint32),
        position,
        has_gaussian,
        float(cached_gaussian),
    )
    probe = numpy.random.RandomState()
    try:
        probe.set_state(state)
    except (TypeError, ValueError) as error:
        raise RNGStateError("NumPy RNG state is invalid") from error
    return state


def restore_numpy_rng(payload: bytes) -> None:
    """Restore NumPy's process-global RNG after compatibility validation."""

    numpy = _import_numpy()
    numpy.random.set_state(_decode_numpy_rng(payload))


def _numpy_json_value(value: Any, numpy: Any) -> object:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise RNGStateError("NumPy Generator state contains a non-string key")
        return {key: _numpy_json_value(item, numpy) for key, item in value.items()}
    if isinstance(value, tuple):
        return {
            "__tuple__": [_numpy_json_value(item, numpy) for item in value],
        }
    if isinstance(value, numpy.ndarray):
        contiguous = numpy.ascontiguousarray(value)
        return {
            "__ndarray__": {
                "data_base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            }
        }
    if isinstance(value, numpy.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RNGStateError(f"unsupported NumPy Generator state value {type(value).__name__}")


def _numpy_state_value(value: object, numpy: Any) -> Any:
    if isinstance(value, list):
        return [_numpy_state_value(item, numpy) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            items = value["__tuple__"]
            if not isinstance(items, list):
                raise RNGStateError("NumPy Generator tuple marker is malformed")
            return tuple(_numpy_state_value(item, numpy) for item in items)
        if set(value) == {"__ndarray__"}:
            array = _require_exact_keys(
                value["__ndarray__"],
                {"data_base64", "dtype", "shape"},
                label="NumPy Generator array",
            )
            if (
                not isinstance(array["data_base64"], str)
                or not isinstance(array["dtype"], str)
                or not isinstance(array["shape"], list)
                or not all(
                    isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in array["shape"]
                )
            ):
                raise RNGStateError("NumPy Generator array marker is malformed")
            try:
                raw = base64.b64decode(array["data_base64"], validate=True)
                dtype = numpy.dtype(array["dtype"])
                decoded = numpy.frombuffer(raw, dtype=dtype).copy()
                return decoded.reshape(tuple(array["shape"]))
            except (TypeError, ValueError) as error:
                raise RNGStateError("NumPy Generator array cannot be decoded") from error
        return {key: _numpy_state_value(item, numpy) for key, item in value.items()}
    return value


def snapshot_numpy_generator(rng: Any) -> bytes:
    """Serialize one explicit ``numpy.random.Generator`` safely."""

    numpy = _import_numpy()
    if not isinstance(rng, numpy.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    value = {
        "bit_generator": type(rng.bit_generator).__name__,
        "numpy_version": numpy.__version__,
        "schema": "prospect.rng.numpy-generator",
        "schema_version": RNG_SCHEMA_VERSION,
        "state": _numpy_json_value(rng.bit_generator.state, numpy),
    }
    return canonical_json_bytes(value)


def _decode_numpy_generator(payload: bytes, rng: Any) -> dict[str, Any]:
    numpy = _import_numpy()
    if not isinstance(rng, numpy.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    value = _require_exact_keys(
        _decode_canonical_json(payload, label="NumPy Generator state"),
        {
            "bit_generator",
            "numpy_version",
            "schema",
            "schema_version",
            "state",
        },
        label="NumPy Generator state",
    )
    _require_schema(
        value,
        "prospect.rng.numpy-generator",
        label="NumPy Generator state",
    )
    if value["numpy_version"] != numpy.__version__:
        raise RNGStateError("NumPy Generator state was produced by a different NumPy version")
    if value["bit_generator"] != type(rng.bit_generator).__name__:
        raise RNGStateError("NumPy Generator bit-generator implementation differs")
    state = _numpy_state_value(value["state"], numpy)
    if not isinstance(state, dict):
        raise RNGStateError("NumPy Generator decoded state must be an object")
    bit_generator_type = type(rng.bit_generator)
    try:
        probe = numpy.random.Generator(bit_generator_type())
        probe.bit_generator.state = state
    except (TypeError, ValueError) as error:
        raise RNGStateError("NumPy Generator state is invalid") from error
    return state


def restore_numpy_generator(payload: bytes, rng: Any) -> None:
    """Restore one explicit ``numpy.random.Generator`` safely."""

    rng.bit_generator.state = _decode_numpy_generator(payload, rng)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RNGStateError("Torch is required to consume this RNG state") from error
    return torch


def _torch_state_bytes(state: Any) -> bytes:
    return state.detach().cpu().contiguous().numpy().tobytes()


def _torch_state_tensor(raw: bytes, torch: Any) -> Any:
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def snapshot_torch_cpu_rng() -> bytes:
    """Serialize Torch's process-global CPU RNG as safe canonical JSON."""

    torch = _import_torch()
    raw = _torch_state_bytes(torch.get_rng_state())
    value = {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "schema": "prospect.rng.torch-cpu",
        "schema_version": RNG_SCHEMA_VERSION,
        "torch_version": torch.__version__,
    }
    return canonical_json_bytes(value)


def _decode_torch_cpu_rng(payload: bytes) -> Any:
    torch = _import_torch()
    value = _require_exact_keys(
        _decode_canonical_json(payload, label="Torch CPU RNG state"),
        {"data_base64", "schema", "schema_version", "torch_version"},
        label="Torch CPU RNG state",
    )
    _require_schema(value, "prospect.rng.torch-cpu", label="Torch CPU RNG state")
    if value["torch_version"] != torch.__version__:
        raise RNGStateError("Torch CPU RNG state was produced by a different Torch version")
    if not isinstance(value["data_base64"], str):
        raise RNGStateError("Torch CPU RNG data must be base64 text")
    try:
        raw = base64.b64decode(value["data_base64"], validate=True)
    except ValueError as error:
        raise RNGStateError("Torch CPU RNG data is not valid base64") from error
    current_size = int(torch.get_rng_state().numel())
    if len(raw) != current_size:
        raise RNGStateError("Torch CPU RNG byte length is incompatible")
    return _torch_state_tensor(raw, torch)


def restore_torch_cpu_rng(payload: bytes) -> None:
    """Restore Torch's process-global CPU RNG after compatibility validation."""

    torch = _import_torch()
    torch.set_rng_state(_decode_torch_cpu_rng(payload))


def snapshot_torch_accelerator_rng() -> bytes:
    """Serialize every Torch CUDA RNG, or an explicit no-accelerator state."""

    torch = _import_torch()
    available = bool(torch.cuda.is_available())
    states = torch.cuda.get_rng_state_all() if available else []
    value = {
        "available": available,
        "backend": "cuda",
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_names": [str(torch.cuda.get_device_name(index)) for index in range(len(states))],
        "schema": "prospect.rng.torch-accelerator",
        "schema_version": RNG_SCHEMA_VERSION,
        "states_base64": [base64.b64encode(_torch_state_bytes(state)).decode("ascii") for state in states],
        "torch_version": torch.__version__,
    }
    return canonical_json_bytes(value)


def _decode_torch_accelerator_rng(payload: bytes) -> tuple[Any, ...]:
    torch = _import_torch()
    value = _require_exact_keys(
        _decode_canonical_json(payload, label="Torch accelerator RNG state"),
        {
            "available",
            "backend",
            "device_count",
            "device_names",
            "schema",
            "schema_version",
            "states_base64",
            "torch_version",
        },
        label="Torch accelerator RNG state",
    )
    _require_schema(
        value,
        "prospect.rng.torch-accelerator",
        label="Torch accelerator RNG state",
    )
    if value["torch_version"] != torch.__version__:
        raise RNGStateError("Torch accelerator RNG state was produced by a different Torch version")
    if value["backend"] != "cuda" or not isinstance(value["available"], bool):
        raise RNGStateError("Torch accelerator RNG backend metadata is malformed")
    available = bool(torch.cuda.is_available())
    if value["available"] != available:
        raise RNGStateError("Torch CUDA availability differs from the checkpoint")
    device_count = value["device_count"]
    names = value["device_names"]
    encoded_states = value["states_base64"]
    if (
        not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or not isinstance(encoded_states, list)
        or not all(isinstance(state, str) for state in encoded_states)
    ):
        raise RNGStateError("Torch accelerator RNG device metadata is malformed")
    expected_count = int(torch.cuda.device_count()) if available else 0
    if device_count != expected_count or len(names) != expected_count:
        raise RNGStateError("Torch CUDA device count differs from the checkpoint")
    current_names = [str(torch.cuda.get_device_name(index)) for index in range(expected_count)]
    if names != current_names:
        raise RNGStateError("Torch CUDA device identities differ from the checkpoint")
    current_states = torch.cuda.get_rng_state_all() if available else []
    if len(encoded_states) != len(current_states):
        raise RNGStateError("Torch CUDA RNG state count differs from the checkpoint")
    decoded: list[Any] = []
    for encoded, current in zip(encoded_states, current_states, strict=True):
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise RNGStateError("Torch CUDA RNG data is not valid base64") from error
        if len(raw) != int(current.numel()):
            raise RNGStateError("Torch CUDA RNG byte length is incompatible")
        decoded.append(_torch_state_tensor(raw, torch))
    return tuple(decoded)


def restore_torch_accelerator_rng(payload: bytes) -> None:
    """Restore all Torch CUDA RNGs after hardware compatibility validation."""

    torch = _import_torch()
    states = _decode_torch_accelerator_rng(payload)
    if states:
        torch.cuda.set_rng_state_all(list(states))


def rng_component_payloads(
    *,
    collection_rng: Any,
    planner_rng: Any,
    logical_version: str = "rng-v1",
    predecessors: Mapping[str, str | None] | None = None,
) -> dict[str, ComponentPayload]:
    """Capture the six canonical RNG components for a checkpoint."""

    if collection_rng is planner_rng:
        raise ValueError("collection_rng and planner_rng must be distinct generators")
    predecessor_map = dict(predecessors or {})
    unknown = set(predecessor_map) - set(RNG_COMPONENT_IDS)
    if unknown:
        raise ValueError(f"unknown RNG predecessor component IDs: {sorted(unknown)}")
    payloads = {
        "python_rng": snapshot_python_rng(),
        "numpy_rng": snapshot_numpy_rng(),
        "torch_cpu_rng": snapshot_torch_cpu_rng(),
        "torch_accelerator_rng": snapshot_torch_accelerator_rng(),
        "collection_rng": snapshot_numpy_generator(collection_rng),
        "planner_rng": snapshot_numpy_generator(planner_rng),
    }
    return {
        component_id: ComponentPayload(
            component_id=component_id,
            logical_version=logical_version,
            payload=payload,
            media_type=_RNG_MEDIA_TYPE,
            predecessor_sha256=predecessor_map.get(component_id),
        )
        for component_id, payload in payloads.items()
    }


def rng_restorers(
    loaded: LoadedWMCheckpoint,
    *,
    collection_rng: Any,
    planner_rng: Any,
) -> dict[str, ComponentRestorer]:
    """Preflight all six RNG states, then return no-fail-shape restore callbacks.

    Calling this function performs all decoding and compatibility checks without
    changing RNG state.  The returned callbacks apply the already validated states.
    """

    if collection_rng is planner_rng:
        raise ValueError("collection_rng and planner_rng must be distinct generators")
    torch = _import_torch()
    python_state = _decode_python_rng(loaded.payload("python_rng"))
    numpy_state = _decode_numpy_rng(loaded.payload("numpy_rng"))
    torch_cpu_state = _decode_torch_cpu_rng(loaded.payload("torch_cpu_rng"))
    accelerator_states = _decode_torch_accelerator_rng(loaded.payload("torch_accelerator_rng"))
    collection_state = _decode_numpy_generator(
        loaded.payload("collection_rng"),
        collection_rng,
    )
    planner_state = _decode_numpy_generator(
        loaded.payload("planner_rng"),
        planner_rng,
    )
    numpy = _import_numpy()

    def restore_python(_payload: bytes, _record: ComponentDigest) -> None:
        random.setstate(python_state)

    def restore_numpy(_payload: bytes, _record: ComponentDigest) -> None:
        numpy.random.set_state(numpy_state)

    def restore_torch_cpu(_payload: bytes, _record: ComponentDigest) -> None:
        torch.set_rng_state(torch_cpu_state)

    def restore_accelerator(_payload: bytes, _record: ComponentDigest) -> None:
        if accelerator_states:
            torch.cuda.set_rng_state_all(list(accelerator_states))

    def restore_collection(_payload: bytes, _record: ComponentDigest) -> None:
        collection_rng.bit_generator.state = collection_state

    def restore_planner(_payload: bytes, _record: ComponentDigest) -> None:
        planner_rng.bit_generator.state = planner_state

    return {
        "python_rng": restore_python,
        "numpy_rng": restore_numpy,
        "torch_cpu_rng": restore_torch_cpu,
        "torch_accelerator_rng": restore_accelerator,
        "collection_rng": restore_collection,
        "planner_rng": restore_planner,
    }


def restore_runtime(
    loaded: LoadedWMCheckpoint,
    *,
    opaque_restorers: Mapping[str, ComponentRestorer],
    collection_rng: Any,
    planner_rng: Any,
) -> None:
    """Restore all WM-001 state into fresh objects with complete preflight.

    Opaque callback coverage and all RNG payload compatibility are checked before
    the first callback runs.  The opaque restorers should initialize fresh
    components; they are deliberately not rollback wrappers for live objects.
    """

    if set(opaque_restorers) != set(OPAQUE_COMPONENT_IDS):
        missing = set(OPAQUE_COMPONENT_IDS) - set(opaque_restorers)
        extra = set(opaque_restorers) - set(OPAQUE_COMPONENT_IDS)
        raise WMCheckpointError(
            "opaque_restorers must contain exactly the non-RNG component IDs "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    prepared_rng_restorers = rng_restorers(
        loaded,
        collection_rng=collection_rng,
        planner_rng=planner_rng,
    )
    loaded.restore({**dict(opaque_restorers), **prepared_rng_restorers})


def _manifest_component_schema(component_id: str) -> dict[str, object]:
    return {
        "additionalProperties": False,
        "properties": {
            "bytes": {"minimum": 0, "type": "integer"},
            "checkpoint_id": {"minLength": 1, "type": "string"},
            "component_id": {"const": component_id},
            "logical_version": {"minLength": 1, "type": "string"},
            "media_type": {"minLength": 1, "type": "string"},
            "predecessor_sha256": {
                "oneOf": [
                    {"pattern": "^[0-9a-f]{64}$", "type": "string"},
                    {"type": "null"},
                ]
            },
            "sha256": {
                "pattern": "^[0-9a-f]{64}$",
                "type": "string",
            },
        },
        "required": [
            "bytes",
            "checkpoint_id",
            "component_id",
            "logical_version",
            "media_type",
            "predecessor_sha256",
            "sha256",
        ],
        "type": "object",
    }


MANIFEST_SCHEMA_DOCUMENT: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "prospect.world-model-lifecycle.checkpoint-manifest.v1",
    "additionalProperties": False,
    "properties": {
        "agent_id": {"minLength": 1, "type": "string"},
        "boundary": {"const": _BOUNDARY},
        "checkpoint_id": {"minLength": 1, "type": "string"},
        "components": {
            "items": False,
            "maxItems": len(CANONICAL_COMPONENT_IDS),
            "minItems": len(CANONICAL_COMPONENT_IDS),
            "prefixItems": [_manifest_component_schema(component_id) for component_id in CANONICAL_COMPONENT_IDS],
            "type": "array",
        },
        "created_at": {
            "additionalProperties": False,
            "properties": {
                "clock_id": {"minLength": 1, "type": "string"},
                "tick": {"minimum": 0, "type": "integer"},
            },
            "required": ["clock_id", "tick"],
            "type": "object",
        },
        "schema": {"const": CHECKPOINT_SCHEMA},
    },
    "required": [
        "agent_id",
        "boundary",
        "checkpoint_id",
        "components",
        "created_at",
        "schema",
    ],
    "type": "object",
}


def manifest_schema_bytes() -> bytes:
    """Return the canonical schema bytes bound by the formal manifest."""

    return canonical_json_bytes(MANIFEST_SCHEMA_DOCUMENT)


def manifest_schema_sha256() -> str:
    """Return the formal-binding digest for this manifest schema."""

    return hashlib.sha256(manifest_schema_bytes()).hexdigest()


__all__ = (
    "CANONICAL_COMPONENT_IDS",
    "CHECKPOINT_SCHEMA",
    "MANIFEST_SCHEMA_DOCUMENT",
    "OPAQUE_COMPONENT_IDS",
    "RNG_COMPONENT_IDS",
    "ComponentDigest",
    "ComponentPayload",
    "LoadedWMCheckpoint",
    "RNGStateError",
    "WMCheckpointError",
    "WMCheckpointReport",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_checkpoint",
    "manifest_schema_bytes",
    "manifest_schema_sha256",
    "restore_numpy_generator",
    "restore_numpy_rng",
    "restore_python_rng",
    "restore_runtime",
    "restore_torch_accelerator_rng",
    "restore_torch_cpu_rng",
    "rng_component_payloads",
    "rng_restorers",
    "save_checkpoint",
    "snapshot_numpy_generator",
    "snapshot_numpy_rng",
    "snapshot_python_rng",
    "snapshot_torch_accelerator_rng",
    "snapshot_torch_cpu_rng",
)
