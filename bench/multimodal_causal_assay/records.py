"""Canonical, target-agnostic record primitives for MM-011.

This module owns serialization, hashing, sealed-parent pins, strict NPZ schema
checking, and exclusive immutable writes.  It deliberately knows nothing about
prediction fitting, scoring, decisions, launcher mechanics, or the MM-007 scientific
contents or launcher backend.  In particular, validating a parent tree hashes
its files as opaque byte strings and never opens an NPZ archive.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import struct
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeAlias, cast

import numpy as np

SCHEMA_VERSION: Final = "mm011-records-v1"
PROTOCOL_PATH: Final = Path(
    "docs/research/2026-07-16-mm011-lcv-backed-causal-deformation-appearance-prediction-protocol.md"
)
# Frozen after the independent implementation/custody audit and before any real
# MM-007 scientific input was opened by MM-011.
PROTOCOL_SHA256: Final[str] = "ca39f7cea6a2a5b041956b419bf3530dd54eb8403096963a044d7fcf1e2121cc"

MM011_ARRAY_TAG: Final = b"MM011-scientific-array-v1\0"
MM011_JSON_TAG: Final = b"MM011-canonical-json-v1\0"

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class RecordValidationError(ValueError):
    """Raised when canonical data or a sealed record fails closed."""


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecordValidationError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


@dataclass(frozen=True, slots=True)
class ParentPin:
    sha256: str
    mode: int

    def __post_init__(self) -> None:
        require_sha256(self.sha256, "parent SHA-256")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise RecordValidationError("parent mode must be an exact permission integer")


PARENT_PINS: Final[Mapping[str, ParentPin]] = MappingProxyType(
    {
        "artifact-manifest.json": ParentPin("db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb", 0o444),
        "input-manifest.json": ParentPin("1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f", 0o444),
        "formal-start.json": ParentPin("ea5c7bda870d71ead3172c1fc6e504d6a6b02d2ba785e9fd2fc75a91c667eee3", 0o444),
        "MM-007-evidence.json": ParentPin("13dfa89e541e6122263ea9814d42fb328da303dcc74556cdaaa5d5860d99abaf", 0o444),
        "MM-007-results.json": ParentPin("3c92729e1e5c18c14461e36602bdb86acd31750d9f5a85f535cd33a43fb9c47b", 0o444),
        "MM-007-report.md": ParentPin("b18760128941ab2eff893b8c0afc469b92f71077d489e060d56519407990b8a2", 0o444),
        "MM-007-protocol.md": ParentPin("24bbac1855cc2b51d2a65012b9c63037637c53555b86bbad7c66a6249108a73c", 0o444),
        "MM-007-frames-64x64.npz": ParentPin("fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f", 0o444),
    }
)

# The copied LCV tree retains MM-007's lineage inputs.  MM-011 validates only this
# directory's top-level identity when checking the eight direct consumer files; the
# complete enclosing LCV tree is independently authenticated by ``lcv_parent``.
PARENT_EXCLUDED_NONAUTHORITATIVE_DIRECTORIES: Final[tuple[str, ...]] = ("inputs",)

PARENT_FRAME_ARRAY_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "video_ids": "06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50",
        "timestamps": "128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55",
        "frames_uint8": "46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9",
    }
)


@dataclass(frozen=True, slots=True)
class ArraySpec:
    dtype: np.dtype[Any]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = np.dtype(self.dtype)
        if normalized.hasobject:
            raise RecordValidationError("an NPZ schema cannot authorize object arrays")
        if any(type(size) is not int or size < 0 for size in self.shape):
            raise RecordValidationError("array schema dimensions must be nonnegative integers")
        object.__setattr__(self, "dtype", normalized)


def _validate_json(value: object, path: str, ancestors: set[int]) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RecordValidationError(f"{path} contains a nonfinite JSON number")
        return
    if type(value) is list:
        identity = id(value)
        if identity in ancestors:
            raise RecordValidationError(f"{path} contains a JSON cycle")
        ancestors.add(identity)
        try:
            for index, item in enumerate(cast(list[object], value)):
                _validate_json(item, f"{path}[{index}]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if type(value) is dict:
        identity = id(value)
        if identity in ancestors:
            raise RecordValidationError(f"{path} contains a JSON cycle")
        ancestors.add(identity)
        try:
            for key, item in cast(dict[object, object], value).items():
                if type(key) is not str:
                    raise RecordValidationError(f"{path} contains a non-string JSON key")
                _validate_json(item, f"{path}.{key}", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise RecordValidationError(f"{path} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Return the unique ASCII JSON encoding used by MM-011 (without LF)."""

    _validate_json(value, "$", set())
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RecordValidationError("value is not canonical MM-011 JSON") from error
    return payload


def json_file_bytes(value: JsonValue) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def canonical_json_sha256(value: JsonValue, *, protocol_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(MM011_JSON_TAG)
    digest.update(bytes.fromhex(require_sha256(protocol_sha256, "frozen protocol SHA-256")))
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def _canonical_array(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RecordValidationError("scientific array must be a NumPy ndarray")
    if value.dtype.hasobject or value.dtype.fields is not None or value.dtype.subdtype is not None:
        raise RecordValidationError("scientific array dtype is outside the closed MM-011 grammar")
    if value.ndim > 255 or any(size > 2**63 - 1 for size in value.shape):
        raise RecordValidationError("scientific array rank or extent is too large")
    kind = value.dtype.kind
    if kind not in "bui fUS".replace(" ", ""):
        raise RecordValidationError("scientific array dtype is outside the closed MM-011 grammar")
    if kind == "f" and not np.all(np.isfinite(value)):
        raise RecordValidationError("scientific float array contains a nonfinite value")
    dtype = value.dtype
    if dtype.byteorder not in ("|", "<"):
        dtype = dtype.newbyteorder("<")
    return cast(np.ndarray, np.ascontiguousarray(value, dtype=dtype))


def scientific_array_sha256(role: str, value: np.ndarray, *, protocol_sha256: str) -> str:
    """Hash an ndarray with protocol, role, dtype, rank, shape, and exact bytes."""

    if not isinstance(role, str) or not role or len(role) > 255:
        raise RecordValidationError("array role must be a nonempty string of at most 255 characters")
    try:
        role_bytes = role.encode("ascii")
    except UnicodeEncodeError as error:
        raise RecordValidationError("array role must be ASCII") from error
    array = _canonical_array(value)
    dtype_bytes = array.dtype.str.encode("ascii")
    if len(dtype_bytes) > 255:
        raise RecordValidationError("array dtype descriptor is too long")
    digest = hashlib.sha256()
    digest.update(MM011_ARRAY_TAG)
    digest.update(bytes.fromhex(require_sha256(protocol_sha256, "frozen protocol SHA-256")))
    digest.update(struct.pack("<B", len(role_bytes)))
    digest.update(role_bytes)
    digest.update(struct.pack("<B", len(dtype_bytes)))
    digest.update(dtype_bytes)
    digest.update(struct.pack("<B", array.ndim))
    for size in array.shape:
        digest.update(struct.pack("<Q", size))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def legacy_mm007_array_sha256(value: np.ndarray) -> str:
    """Replay the sealed MM-007 frame-array digest grammar exactly."""

    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise RecordValidationError("MM-007 array digest requires a non-object ndarray")
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def immutable_array(value: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes."""

    array = _canonical_array(value)
    return cast(np.ndarray, np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape))


def _regular_stat(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RecordValidationError(f"required regular file is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecordValidationError(f"path is not a regular non-symlink file: {path}")
    if metadata.st_nlink != 1:
        raise RecordValidationError(f"hard-linked files are forbidden: {path}")
    return metadata


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return every stable identity field enforced across one opened-file read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _digest_opened_file(path: str | Path) -> tuple[str, os.stat_result]:
    """Hash one unique regular file while binding the pathname to the opened inode.

    A separate ``lstat`` followed by ``open`` is otherwise vulnerable to a pathname
    replacement between the two operations.  ``O_NOFOLLOW`` closes the symlink case;
    the device/inode/size/mode comparisons below close replacement and in-flight
    mutation cases for the local formal-artifact threat model.
    """

    candidate = Path(path)
    before = _regular_stat(candidate)
    before_identity = _stable_file_identity(before)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("MM-011 requires O_NOFOLLOW")
    descriptor = os.open(candidate, flags | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        identity = _stable_file_identity(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity != before_identity:
            raise RecordValidationError(f"opened path is not a unique regular file: {candidate}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_read = os.fstat(descriptor)
        try:
            after_path = candidate.lstat()
        except FileNotFoundError as error:
            raise RecordValidationError(f"file path disappeared while hashing: {candidate}") from error
        if _stable_file_identity(after_read) != identity or _stable_file_identity(after_path) != identity:
            raise RecordValidationError(f"file identity changed while hashing: {candidate}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), opened


def read_regular_bytes(path: str | Path, *, maximum_bytes: int | None = None) -> bytes:
    """Read exact bytes from a pathname-bound, stable, unique regular file."""

    if maximum_bytes is not None and (type(maximum_bytes) is not int or maximum_bytes < 0):
        raise RecordValidationError("maximum byte count must be a nonnegative integer")
    candidate = Path(path)
    before = _regular_stat(candidate)
    before_identity = _stable_file_identity(before)
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise RecordValidationError(f"regular file exceeds its byte ceiling: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("MM-011 requires O_NOFOLLOW")
    descriptor = os.open(candidate, flags | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        identity = _stable_file_identity(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity != before_identity:
            raise RecordValidationError(f"opened path is not a unique regular file: {candidate}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RecordValidationError(f"regular file ended before its bound size: {candidate}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RecordValidationError(f"regular file grew while being read: {candidate}")
        after_read = os.fstat(descriptor)
        try:
            after_path = candidate.lstat()
        except FileNotFoundError as error:
            raise RecordValidationError(f"file path disappeared while being read: {candidate}") from error
        if _stable_file_identity(after_read) != identity or _stable_file_identity(after_path) != identity:
            raise RecordValidationError(f"file identity changed while being read: {candidate}")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def file_sha256(path: str | Path) -> str:
    return _digest_opened_file(path)[0]


def file_record(path: str | Path) -> dict[str, JsonValue]:
    digest, metadata = _digest_opened_file(path)
    return {
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "sha256": digest,
    }


def validate_pinned_parent_tree(
    root: str | Path,
    pins: Mapping[str, ParentPin] = PARENT_PINS,
    *,
    excluded_nonauthoritative_directories: tuple[str, ...] = (),
) -> dict[str, dict[str, JsonValue]]:
    """Opaque-hash pinned files while never traversing excluded lineage directories."""

    directory = Path(root)
    try:
        root_stat = directory.lstat()
    except FileNotFoundError as error:
        raise RecordValidationError("pinned parent root is missing") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RecordValidationError("pinned parent root must be a non-symlink directory")
    exclusions = set(excluded_nonauthoritative_directories)
    if (
        len(exclusions) != len(excluded_nonauthoritative_directories)
        or exclusions.intersection(pins)
        or any(not name or name in {".", ".."} or Path(name).name != name for name in exclusions)
    ):
        raise RecordValidationError("pinned parent excluded-directory declaration is invalid")
    expected_files = set(pins)
    expected = expected_files | exclusions
    try:
        with os.scandir(directory) as scanned:
            entries = {entry.name: entry.stat(follow_symlinks=False) for entry in scanned}
    except OSError as error:
        raise RecordValidationError("pinned parent top-level census failed") from error
    actual = set(entries)
    if actual != expected:
        raise RecordValidationError(
            f"pinned parent membership differs: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name in sorted(exclusions):
        mode = entries[name].st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RecordValidationError(
                f"excluded non-authoritative parent entry must be a non-symlink directory: {name}"
            )
    for name in sorted(expected_files):
        mode = entries[name].st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RecordValidationError(f"pinned parent file must be a non-symlink regular file: {name}")
    records: dict[str, dict[str, JsonValue]] = {}
    for name in sorted(expected_files):
        pin = pins[name]
        path = directory / name
        record = file_record(path)
        if record["mode"] != pin.mode or record["sha256"] != pin.sha256:
            raise RecordValidationError(f"pinned parent file differs: {name}")
        records[name] = record
    try:
        after_root_stat = directory.lstat()
        with os.scandir(directory) as rescanned:
            after_entries = {entry.name: entry.stat(follow_symlinks=False) for entry in rescanned}
    except OSError as error:
        raise RecordValidationError("pinned parent post-hash top-level census failed") from error
    after_actual = set(after_entries)
    if after_actual != expected:
        raise RecordValidationError(
            "pinned parent membership changed during validation: "
            f"expected={sorted(expected)}, actual={sorted(after_actual)}"
        )
    for name in sorted(expected_files):
        if _stable_file_identity(after_entries[name]) != _stable_file_identity(entries[name]):
            raise RecordValidationError(f"pinned parent file identity changed during validation: {name}")
    for name in sorted(exclusions):
        before = entries[name]
        after = after_entries[name]
        before_identity = (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode))
        after_identity = (after.st_dev, after.st_ino, stat.S_IFMT(after.st_mode))
        if after_identity != before_identity:
            raise RecordValidationError(
                f"excluded non-authoritative parent directory identity changed during validation: {name}"
            )
    if _stable_file_identity(after_root_stat) != _stable_file_identity(root_stat):
        raise RecordValidationError("pinned parent root identity changed during validation")
    return records


def validate_arrays(
    arrays: Mapping[str, np.ndarray], schema: Mapping[str, ArraySpec], *, label: str
) -> Mapping[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or set(arrays) != set(schema):
        raise RecordValidationError(f"{label} NPZ membership differs")
    output: dict[str, np.ndarray] = {}
    for name in sorted(schema):
        value = arrays[name]
        spec = schema[name]
        if not isinstance(value, np.ndarray):
            raise RecordValidationError(f"{label}:{name} is not an ndarray")
        if value.dtype != spec.dtype or value.shape != spec.shape or not value.flags.c_contiguous:
            raise RecordValidationError(
                f"{label}:{name} differs from dtype={spec.dtype.str}, shape={spec.shape}, C-order schema"
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise RecordValidationError(f"{label}:{name} contains a nonfinite value")
        output[name] = immutable_array(value)
    return MappingProxyType(output)


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, value, version=(1, 0), allow_pickle=False)
    return buffer.getvalue()


def canonical_npz_bytes(arrays: Mapping[str, np.ndarray], schema: Mapping[str, ArraySpec], *, label: str) -> bytes:
    """Create a deterministic ZIP of canonical NPY members in sorted order."""

    checked = validate_arrays(arrays, schema, label=label)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(checked):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(checked[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _load_npz_stream(
    stream: io.BytesIO | Path, schema: Mapping[str, ArraySpec], *, label: str
) -> Mapping[str, np.ndarray]:
    with zipfile.ZipFile(stream, "r") as zipped:
        infos = zipped.infolist()
        expected_names = [f"{name}.npy" for name in sorted(schema)]
        actual_names = [info.filename for info in infos]
        if actual_names != expected_names or len(set(actual_names)) != len(actual_names):
            raise RecordValidationError(f"{label} ZIP member order or membership differs")
        for info, name in zip(infos, sorted(schema), strict=True):
            expected_bytes = int(np.prod(schema[name].shape, dtype=np.int64)) * schema[name].dtype.itemsize
            if info.file_size < expected_bytes or info.file_size > expected_bytes + 1024:
                raise RecordValidationError(f"{label}:{name} NPY member has an invalid byte extent")
    if isinstance(stream, io.BytesIO):
        stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        if archive.files != sorted(schema):
            raise RecordValidationError(f"{label} NPZ logical membership differs")
        arrays = {name: np.asarray(archive[name]) for name in sorted(schema)}
    return validate_arrays(arrays, schema, label=label)


def load_npz_bytes(payload: bytes, schema: Mapping[str, ArraySpec], *, label: str) -> Mapping[str, np.ndarray]:
    if not isinstance(payload, bytes):
        raise RecordValidationError("NPZ payload must be exact bytes")
    try:
        return _load_npz_stream(io.BytesIO(payload), schema, label=label)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, RecordValidationError):
            raise
        raise RecordValidationError(f"{label} NPZ cannot be decoded safely") from error


def load_npz_file(path: str | Path, schema: Mapping[str, ArraySpec], *, label: str) -> Mapping[str, np.ndarray]:
    candidate = Path(path)
    try:
        payload = read_regular_bytes(candidate, maximum_bytes=2_000_000_000)
        return _load_npz_stream(io.BytesIO(payload), schema, label=label)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, RecordValidationError):
            raise
        raise RecordValidationError(f"{label} NPZ cannot be decoded safely") from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory(path: str | Path, mode: int = 0o755) -> Path:
    directory = Path(path)
    if directory == directory.parent:
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RecordValidationError("filesystem root is not a real directory")
        return directory
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        ensure_directory(directory.parent, mode)
        os.mkdir(directory, mode)
        _fsync_directory(directory.parent)
        return directory
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RecordValidationError(f"directory component is not a real directory: {directory}")
    return directory


def write_immutable_bytes_exclusive(path: str | Path, payload: bytes) -> dict[str, JsonValue]:
    """Create one 0444 regular file exclusively, fsyncing file and directory."""

    if not isinstance(payload, bytes):
        raise RecordValidationError("immutable payload must be exact bytes")
    destination = Path(path)
    ensure_directory(destination.parent)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("MM-011 requires O_NOFOLLOW")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("exclusive immutable write made no progress")
            written += count
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)
    return file_record(destination)


def write_immutable_json_exclusive(path: str | Path, value: JsonValue) -> dict[str, JsonValue]:
    return write_immutable_bytes_exclusive(path, json_file_bytes(value))


def write_immutable_npz_exclusive(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    schema: Mapping[str, ArraySpec],
    *,
    label: str,
) -> dict[str, JsonValue]:
    return write_immutable_bytes_exclusive(path, canonical_npz_bytes(arrays, schema, label=label))


def copy_opaque_immutable_exclusive(source: str | Path, destination: str | Path) -> dict[str, JsonValue]:
    """Copy a regular file as opaque bytes without parsing or deserializing it."""

    candidate = Path(source)
    payload = read_regular_bytes(candidate, maximum_bytes=2_000_000_000)
    return write_immutable_bytes_exclusive(destination, payload)


__all__ = [
    "ArraySpec",
    "JsonValue",
    "PARENT_FRAME_ARRAY_SHA256",
    "PARENT_PINS",
    "PROTOCOL_PATH",
    "PROTOCOL_SHA256",
    "ParentPin",
    "RecordValidationError",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_npz_bytes",
    "copy_opaque_immutable_exclusive",
    "ensure_directory",
    "file_record",
    "file_sha256",
    "immutable_array",
    "json_file_bytes",
    "legacy_mm007_array_sha256",
    "load_npz_bytes",
    "load_npz_file",
    "read_regular_bytes",
    "require_sha256",
    "scientific_array_sha256",
    "validate_arrays",
    "validate_pinned_parent_tree",
    "write_immutable_bytes_exclusive",
    "write_immutable_json_exclusive",
    "write_immutable_npz_exclusive",
]
