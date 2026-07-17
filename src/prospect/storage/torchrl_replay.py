"""Optional TensorDict/TorchRL replay index behind a backend-neutral codec."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import zipfile
from collections.abc import Sequence
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from prospect.domain import ExperienceEvent

from .memory import DuplicateRecordError


class TorchRLUnavailableError(ImportError):
    """The optional TorchRL replay dependency is unavailable."""


@runtime_checkable
class ExperienceTensorCodec(Protocol):
    """Translate domain experience to and from one scalar-batch TensorDict."""

    version: str

    def encode(self, event: ExperienceEvent) -> object: ...

    def decode(self, encoded: object) -> ExperienceEvent: ...


def torchrl_available() -> bool:
    """Return whether both optional replay packages can be imported."""

    return importlib.util.find_spec("torchrl") is not None and importlib.util.find_spec("tensordict") is not None


class TensorDictExperienceReplay:
    """Capacity-bounded sampling index; never a canonical experience ledger.

    TorchRL and TensorDict are imported only when this class is instantiated.
    The caller owns the codec, so no tensors or backend-specific payload rules
    leak into :mod:`prospect.domain`.
    """

    def __init__(
        self,
        *,
        capacity: int,
        codec: ExperienceTensorCodec,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        if not codec.version or not codec.version.strip():
            raise ValueError("experience tensor codec requires a nonempty version")
        try:
            import torch
            from tensordict import TensorDictBase  # type: ignore[import-untyped]
            from torchrl.data import (  # type: ignore[import-untyped]
                LazyTensorStorage,
                TensorDictReplayBuffer,
            )
        except ImportError as error:
            raise TorchRLUnavailableError(
                "TensorDictExperienceReplay requires the optional runtime dependencies 'torchrl' and 'tensordict'"
            ) from error

        self._tensor_type: type[Any] = TensorDictBase
        self._storage_type: type[Any] = LazyTensorStorage
        self._replay_type: type[Any] = TensorDictReplayBuffer
        self._torch = torch
        self._capacity = capacity
        self._buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=capacity))
        self._buffer.set_rng(torch.Generator().manual_seed(seed))
        self._codec = codec
        self._seen_ids: set[str] = set()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def add(self, event: ExperienceEvent) -> None:
        """Index one unique canonical event for later sampling."""

        with self._lock:
            if event.experience_id in self._seen_ids:
                raise DuplicateRecordError(f"experience id {event.experience_id!r} is already indexed")
            encoded = self._codec.encode(event)
            if not isinstance(encoded, self._tensor_type):
                raise TypeError("experience tensor codec must return a TensorDictBase")
            if encoded.batch_dims != 0:
                raise ValueError("experience tensor codec must return a scalar-batch TensorDict")
            self._buffer.add(encoded)
            self._seen_ids.add(event.experience_id)

    def sample(self, count: int) -> Sequence[ExperienceEvent]:
        """Sample decoded events with replacement using TorchRL's sampler."""

        if count < 1:
            raise ValueError("replay sample count must be positive")
        with self._lock:
            if len(self._buffer) == 0:
                raise ValueError("cannot sample an empty replay index")
            encoded_batch = self._buffer.sample(batch_size=count)
            return tuple(self._codec.decode(encoded) for encoded in encoded_batch.unbind(0))

    def dump_checkpoint_bytes(self) -> bytes:
        """Serialize replay storage, writer, sampler RNG, and lifetime seen IDs.

        TorchRL's directory-based ``dumps`` format is wrapped in an in-memory ZIP
        with a Prospect compatibility manifest.  No pickle is introduced by this
        adapter.  The canonical experience store remains a separate checkpoint
        component and must be restored before a codec that resolves event IDs.
        """

        with self._lock, tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            backend = temporary / "backend"
            self._buffer.dumps(backend)
            metadata = {
                "capacity": self._capacity,
                "codec_version": self._codec.version,
                "format": "prospect-torchrl-replay",
                "schema_version": 1,
                "seen_experience_ids": sorted(self._seen_ids),
                "torchrl_version": package_version("torchrl"),
            }
            metadata_bytes = json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            output = io.BytesIO()
            with zipfile.ZipFile(
                output,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                _write_replay_member(archive, "prospect-replay.json", metadata_bytes)
                for file_path in sorted(path for path in backend.rglob("*") if path.is_file()):
                    relative = file_path.relative_to(backend).as_posix()
                    _write_replay_member(
                        archive,
                        f"backend/{relative}",
                        file_path.read_bytes(),
                    )
            return output.getvalue()

    def load_checkpoint_bytes(self, payload: bytes) -> None:
        """Atomically replace this replay index from verified checkpoint bytes."""

        if not isinstance(payload, bytes):
            raise TypeError("replay checkpoint payload must be bytes")
        try:
            archive_context = zipfile.ZipFile(io.BytesIO(payload), mode="r")
        except zipfile.BadZipFile as error:
            raise ValueError("replay checkpoint is not a valid ZIP bundle") from error

        with self._lock, archive_context as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("replay checkpoint contains duplicate member paths")
            if "prospect-replay.json" not in names:
                raise ValueError("replay checkpoint has no Prospect manifest")
            for info in infos:
                _validate_replay_member(info)
            manifest_info = archive.getinfo("prospect-replay.json")
            if manifest_info.file_size > 16 << 20:
                raise ValueError("replay checkpoint manifest is unreasonably large")
            raw_metadata = archive.read(manifest_info)
            metadata = _decode_replay_metadata(raw_metadata)
            if metadata["capacity"] != self._capacity:
                raise ValueError("replay checkpoint capacity does not match the target replay")
            if metadata["codec_version"] != self._codec.version:
                raise ValueError("replay checkpoint codec version does not match the target codec")
            if metadata["torchrl_version"] != package_version("torchrl"):
                raise ValueError("replay checkpoint TorchRL version does not match the runtime")
            seen_ids = metadata["seen_experience_ids"]
            backend_infos = [info for info in infos if info.filename.startswith("backend/")]
            if not backend_infos:
                raise ValueError("replay checkpoint has no TorchRL backend state")
            total_bytes = sum(info.file_size for info in backend_infos)
            if total_bytes > 4 << 30:
                raise ValueError("replay checkpoint backend state is too large")

            with tempfile.TemporaryDirectory() as temporary_name:
                backend = Path(temporary_name) / "backend"
                for info in backend_infos:
                    relative = PurePosixPath(info.filename).relative_to("backend")
                    destination = backend.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(info))

                candidate = self._replay_type(storage=self._storage_type(max_size=self._capacity))
                candidate.set_rng(self._torch.Generator().manual_seed(0))
                candidate.loads(backend)
                if len(candidate) > self._capacity:
                    raise ValueError("restored replay exceeds configured capacity")
                restored = self._replay_type(storage=self._storage_type(max_size=self._capacity))
                restored.set_rng(self._torch.Generator().manual_seed(0))
                restored.load_state_dict(candidate.state_dict())

            self._buffer = restored
            self._seen_ids = set(seen_ids)


def _write_replay_member(archive: zipfile.ZipFile, path: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(filename=path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    archive.writestr(info, payload)


def _validate_replay_member(info: zipfile.ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe replay checkpoint member path {info.filename!r}")
    if info.is_dir():
        raise ValueError("replay checkpoint must contain files, not directory entries")
    if info.filename != "prospect-replay.json" and (len(path.parts) < 2 or path.parts[0] != "backend"):
        raise ValueError(f"unexpected replay checkpoint member {info.filename!r}")


def _decode_replay_metadata(payload: bytes) -> dict[str, Any]:
    try:
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("replay checkpoint manifest is not valid JSON") from error
    expected_keys = {
        "capacity",
        "codec_version",
        "format",
        "schema_version",
        "seen_experience_ids",
        "torchrl_version",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ValueError("replay checkpoint manifest has unexpected fields")
    canonical = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical != payload:
        raise ValueError("replay checkpoint manifest is not canonical JSON")
    if metadata["format"] != "prospect-torchrl-replay":
        raise ValueError("unsupported replay checkpoint format")
    if metadata["schema_version"] != 1:
        raise ValueError("unsupported replay checkpoint schema version")
    if not isinstance(metadata["capacity"], int) or isinstance(metadata["capacity"], bool):
        raise ValueError("replay checkpoint capacity must be an integer")
    for field in ("codec_version", "torchrl_version"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"replay checkpoint {field} must be a nonempty string")
    seen_ids = metadata["seen_experience_ids"]
    if (
        not isinstance(seen_ids, list)
        or not all(isinstance(value, str) and value.strip() for value in seen_ids)
        or seen_ids != sorted(seen_ids)
        or len(seen_ids) != len(set(seen_ids))
    ):
        raise ValueError("replay checkpoint seen IDs must be unique, sorted nonempty strings")
    return metadata


__all__ = (
    "ExperienceTensorCodec",
    "TensorDictExperienceReplay",
    "TorchRLUnavailableError",
    "torchrl_available",
)
