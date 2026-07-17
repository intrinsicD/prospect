"""Fail-closed filesystem and coverage controls for MM-008 v2.1.

This module is deliberately backend-neutral.  It does not import NumPy or any
MM-008 scientific module, instantiate a random generator, derive a challenge
seed, or read real data.  Its only responsibilities are provenance capture,
one-shot filesystem boundaries, and exact execution-coverage schemas.

Formal admission is intentionally disabled.  The v2.1 exact-13 allowlist is not
the actual executed Python closure because eager package initializers execute
unarchived code.  Its disclosed nonce receipt is also ordered incorrectly for a
fresh seal: v2.2 must freeze source/config/runtime first and then obtain a receipt
that binds that freeze hash.  The v2.1 constants below exist only to exercise
fail-closed fake fixtures and must never authorize a run.
"""

from __future__ import annotations

import ast
import base64
import binascii
import json
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Final, Literal, cast

PROTOCOL_SHA256: Final = "6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f"
RETIRED_PROTOCOL_SHA256: Final = (
    "14a9f6a0f72c118a2107a938dc162b806a53bfc583cb15d3ac308a71fb264b32"
)
FRESH_RECEIPT_SHA256: Final = (
    "82abb8f0f108cc94d7de9da68a5bda7a98e011e5844f3e7dd96d7e816fcf1cd9"
)
RETIRED_RECEIPT_SHA256: Final = (
    "ad893d1bb8d9ae0729f40e37937076a075a76e3fc2115f383e40d5c7dbab574f"
)

SOURCE_ARCHIVE_SCHEMA: Final = "mm008-v2.1-source-archive-v1"
COVERAGE_SCHEMA: Final = "mm008-v2.1-coverage-v1"
COVERAGE_POLICY_STATUS: Final = "provisional_unsealed_pending_v2.2_science"
FREEZE_SCHEMA: Final = "mm008-v2.1-freeze-record-v1"
FORMAL_START_SCHEMA: Final = "mm008-v2.1-formal-start-v1"
TERMINAL_SCHEMA: Final = "mm008-v2.1-terminal-receipt-v1"
NONCE_SCHEMA: Final = "mm008-v2-challenge-nonce-v1"
NONCE_REVIEWER: Final = "/root/mm005_security_review:Hubble"
LIFECYCLE_ADMISSION_STATUS: Final = (
    "blocked_pending_v2.2_executed_closure_and_freeze_bound_nonce"
)

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SOURCE_ARCHIVE_PATH: Final = (
    "docs/research/2026-07-16-mm008-v2-source-archive.json"
)
FORMAL_START_PATH: Final = (
    "bench/multimodal_mechanism_diagnostics/results/MM-008/formal-start.json"
)
TERMINAL_RECEIPT_PATH: Final = (
    "bench/multimodal_mechanism_diagnostics/results/MM-008/terminal-receipt.json"
)


def _required_os_flag(name: str) -> int:
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise RuntimeError(f"MM-008 requires the operating-system flag {name}")
    return value


O_CLOEXEC: Final = _required_os_flag("O_CLOEXEC")
O_DIRECTORY: Final = _required_os_flag("O_DIRECTORY")
O_NOFOLLOW: Final = _required_os_flag("O_NOFOLLOW")

PROTOCOL_PATH: Final = (
    "docs/research/2026-07-16-mm008-v2-robust-deformation-appearance-protocol.md"
)
FRESH_RECEIPT_PATH: Final = (
    "docs/research/2026-07-16-mm008-v2-challenge-nonce-2.json"
)
RETIRED_RECEIPT_PATH: Final = (
    "docs/research/2026-07-16-mm008-v2-challenge-nonce.json"
)

SCIENCE_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "bench/multimodal_horizon_diagnostics/__init__.py",
    "bench/multimodal_horizon_diagnostics/method.py",
    "bench/multimodal_mechanism_diagnostics/calibration_v2.py",
    "bench/multimodal_mechanism_diagnostics/control_v2.py",
    "bench/multimodal_mechanism_diagnostics/method.py",
    "bench/multimodal_mechanism_diagnostics/method_v2.py",
    "bench/multimodal_mechanism_diagnostics/synthetic_v2.py",
    "bench/multimodal_preflight/__init__.py",
    "bench/multimodal_preflight/dataset.py",
    "bench/multimodal_resolution_diagnostics/__init__.py",
    "bench/multimodal_resolution_diagnostics/method.py",
    "bench/multimodal_warp_diagnostics/__init__.py",
    "bench/multimodal_warp_diagnostics/method.py",
)
MM008_TEST_PATHS: Final[tuple[str, ...]] = (
    "tests/test_mm008_method.py",
    "tests/test_mm008_v2_calibration_support.py",
    "tests/test_mm008_v2_control.py",
    "tests/test_mm008_v2_optimizer.py",
    "tests/test_mm008_v2_synthetic_dev.py",
)
SUPPORT_PATHS: Final[tuple[str, ...]] = (
    PROTOCOL_PATH,
    FRESH_RECEIPT_PATH,
    RETIRED_RECEIPT_PATH,
    "pyproject.toml",
)
ARCHIVE_MEMBER_PATHS: Final[tuple[str, ...]] = tuple(
    sorted((*SCIENCE_SOURCE_PATHS, *MM008_TEST_PATHS, *SUPPORT_PATHS), key=str.encode)
)
MECHANISM_ROOT_PATHS: Final[tuple[str, ...]] = tuple(
    path for path in SCIENCE_SOURCE_PATHS if path.startswith("bench/multimodal_mechanism_diagnostics/")
)

if len(SCIENCE_SOURCE_PATHS) != 13 or len(MM008_TEST_PATHS) != 5:
    raise RuntimeError("MM-008 source/test closure cardinality drifted")
if len(ARCHIVE_MEMBER_PATHS) != 22 or len(set(ARCHIVE_MEMBER_PATHS)) != 22:
    raise RuntimeError("MM-008 archive membership is not an exact 22-file set")


class ControlV2Error(ValueError):
    """Stable fail-closed classification for a lifecycle control defect."""


class TerminalRunError(ControlV2Error):
    """Raised when a one-shot marker makes a run identifier terminal."""


def _reject_json_constant(value: str) -> None:
    raise ControlV2Error(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ControlV2Error(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _validate_finite_json(value: object) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ControlV2Error("non-finite JSON number is forbidden")
        return
    if isinstance(value, list):
        for item in value:
            _validate_finite_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ControlV2Error("JSON object keys must be strings")
            _validate_finite_json(item)
        return
    raise ControlV2Error("value is outside the finite JSON data model")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize finite JSON as sorted ASCII with exactly one trailing LF."""

    _validate_finite_json(value)
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ControlV2Error("value cannot be encoded as canonical ASCII JSON") from error
    return payload + b"\n"


def parse_canonical_json_bytes(payload: bytes) -> object:
    """Parse only canonical ASCII JSON with one LF and no duplicate keys."""

    if not isinstance(payload, bytes):
        raise ControlV2Error("canonical JSON input must be immutable bytes")
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ControlV2Error("canonical JSON must have exactly one trailing LF")
    if b"\r" in payload or b"\x00" in payload:
        raise ControlV2Error("canonical JSON contains a forbidden control byte")
    try:
        text = payload.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlV2Error("canonical JSON is not valid ASCII JSON") from error
    _validate_finite_json(value)
    if canonical_json_bytes(value) != payload:
        raise ControlV2Error("JSON bytes are not in the exact canonical form")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise ControlV2Error(f"{name} has missing or extra keys")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ControlV2Error(f"{name} must be a string")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlV2Error(f"{name} must be an integer, not a boolean")
    return value


def _require_sha256(value: object, name: str) -> str:
    digest = _require_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ControlV2Error(f"{name} must be a lowercase SHA-256 hex digest")
    return digest


def _validate_relative_path(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise ControlV2Error("relative path must be a string")
    if not relative or "\\" in relative or "\x00" in relative:
        raise ControlV2Error("relative path is empty or contains a forbidden character")
    path = PurePosixPath(relative)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ControlV2Error("path is not a normalized repository-relative POSIX path")
    if str(path) != relative:
        raise ControlV2Error("relative path has noncanonical spelling")
    return parts


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_root(repo_root: Path) -> int:
    lexical = repo_root.absolute()
    try:
        before = os.lstat(lexical)
    except OSError as error:
        raise ControlV2Error("repository root cannot be lstat'ed") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ControlV2Error("repository root must be a non-symlink directory")
    flags = (
        os.O_RDONLY
        | O_DIRECTORY
        | O_CLOEXEC
        | O_NOFOLLOW
    )
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise ControlV2Error("repository root cannot be opened securely") from error
    after = os.fstat(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        os.close(descriptor)
        raise ControlV2Error("repository root changed while it was opened")
    return descriptor


def _open_anchored_parent(root_descriptor: int, relative: str) -> tuple[int, str]:
    parts = _validate_relative_path(relative)
    current = os.dup(root_descriptor)
    directory_flags = (
        os.O_RDONLY
        | O_DIRECTORY
        | O_CLOEXEC
        | O_NOFOLLOW
    )
    try:
        for part in parts[:-1]:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ControlV2Error("anchored path ancestor is not a regular directory")
            child = os.open(part, directory_flags, dir_fd=current)
            after = os.fstat(child)
            if _stat_identity(before) != _stat_identity(after):
                os.close(child)
                raise ControlV2Error("anchored path ancestor changed while opened")
            os.close(current)
            current = child
        return current, parts[-1]
    except (OSError, ControlV2Error) as error:
        os.close(current)
        if isinstance(error, ControlV2Error):
            raise
        raise ControlV2Error("anchored path traversal failed") from error


@dataclass(frozen=True, slots=True)
class AnchoredFile:
    path: str
    mode: int
    size: int
    sha256: str
    payload: bytes


def _read_anchored_regular(repo_root: Path, relative: str) -> AnchoredFile:
    root_descriptor = _open_root(repo_root)
    try:
        parent, leaf = _open_anchored_parent(root_descriptor, relative)
    finally:
        os.close(root_descriptor)
    descriptor = -1
    try:
        before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ControlV2Error("anchored input must be a single-link regular file")
        flags = os.O_RDONLY | O_CLOEXEC | O_NOFOLLOW
        descriptor = os.open(leaf, flags, dir_fd=parent)
        opened = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(opened):
            raise ControlV2Error("anchored input changed while opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_read = os.fstat(descriptor)
        after_path = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if (
            _stat_identity(opened) != _stat_identity(after_read)
            or _stat_identity(opened) != _stat_identity(after_path)
        ):
            raise ControlV2Error("anchored input drifted during read")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise ControlV2Error("anchored input read length differs from stat size")
        return AnchoredFile(
            path=relative,
            mode=stat.S_IMODE(opened.st_mode),
            size=opened.st_size,
            sha256=sha256(payload).hexdigest(),
            payload=payload,
        )
    except (OSError, ControlV2Error) as error:
        if isinstance(error, ControlV2Error):
            raise
        raise ControlV2Error("anchored regular-file read failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _read_absolute_regular(path: Path) -> tuple[bytes, os.stat_result]:
    """Read one absolute non-symlink regular file with identity stability."""

    if not path.is_absolute():
        raise ControlV2Error("absolute runtime file path is not absolute")
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ControlV2Error("absolute runtime input must be a non-symlink regular file")
        descriptor = os.open(path, os.O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    except OSError as error:
        raise ControlV2Error("absolute runtime file cannot be securely opened") from error
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(opened):
            raise ControlV2Error("absolute runtime input changed while opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(after):
            raise ControlV2Error("absolute runtime input drifted while read")
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise ControlV2Error("absolute runtime input read length differs")
        return payload, opened
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise ControlV2Error("exclusive output write made no progress")
        offset += written


def _write_exclusive_readonly(repo_root: Path, relative: str, payload: bytes) -> AnchoredFile:
    root_descriptor = _open_root(repo_root)
    try:
        parent, leaf = _open_anchored_parent(root_descriptor, relative)
    finally:
        os.close(root_descriptor)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | O_CLOEXEC
        | O_NOFOLLOW
    )
    descriptor = -1
    created = False
    try:
        descriptor = os.open(leaf, flags, 0o600, dir_fd=parent)
        created = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except FileExistsError as error:
        raise TerminalRunError(f"exclusive output already exists: {relative}") from error
    except (OSError, ControlV2Error) as error:
        if isinstance(error, ControlV2Error):
            raise
        raise ControlV2Error("exclusive read-only output write failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            os.fsync(parent)
        os.close(parent)
    observed = _read_anchored_regular(repo_root, relative)
    if observed.mode != 0o444 or observed.payload != payload:
        raise ControlV2Error("exclusive output readback differs or is not mode 0444")
    return observed


def _anchored_path_exists(repo_root: Path, relative: str) -> bool:
    """Treat any existing leaf, including an unsafe/partial one, as present."""

    root_descriptor = _open_root(repo_root)
    try:
        parent, leaf = _open_anchored_parent(root_descriptor, relative)
    finally:
        os.close(root_descriptor)
    try:
        try:
            os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ControlV2Error("anchored path presence cannot be inspected") from error
        return True
    finally:
        os.close(parent)


@dataclass(frozen=True, slots=True)
class SourceArchiveEntry:
    path: str
    mode: int
    size: int
    sha256: str
    bytes_b64: str

    @classmethod
    def from_file(cls, value: AnchoredFile) -> SourceArchiveEntry:
        return cls(
            path=value.path,
            mode=value.mode,
            size=value.size,
            sha256=value.sha256,
            bytes_b64=base64.b64encode(value.payload).decode("ascii"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "bytes_b64": self.bytes_b64,
            "mode": self.mode,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class SourceArchive:
    archive_path: str
    archive_sha256: str
    entries: tuple[SourceArchiveEntry, ...]

    @property
    def payload(self) -> dict[str, object]:
        return {
            "files": [entry.as_dict() for entry in self.entries],
            "schema_version": SOURCE_ARCHIVE_SCHEMA,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload)

    def by_path(self) -> dict[str, SourceArchiveEntry]:
        return {entry.path: entry for entry in self.entries}


def _existing_repo_path(repo_root: Path, relative: str) -> str | None:
    try:
        os.lstat(repo_root / relative)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ControlV2Error(f"local import candidate cannot be inspected: {relative}") from error
    _read_anchored_regular(repo_root, relative)
    return relative


def _local_import_paths(repo_root: Path, relative: str, payload: bytes) -> tuple[str, ...]:
    try:
        tree = ast.parse(payload.decode("utf-8"), filename=relative)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ControlV2Error(f"scientific source cannot be parsed: {relative}") from error
    discovered: set[str] = set()

    def add_module(module_name: str) -> None:
        if not module_name.startswith("bench."):
            return
        module_parts = module_name.split(".")
        for length in range(1, len(module_parts)):
            initializer = f"{'/'.join(module_parts[:length])}/__init__.py"
            if _existing_repo_path(repo_root, initializer) is not None:
                discovered.add(initializer)
        stem = module_name.replace(".", "/")
        module_path = f"{stem}.py"
        package_path = f"{stem}/__init__.py"
        for candidate in (module_path, package_path):
            if _existing_repo_path(repo_root, candidate) is not None:
                discovered.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in {"importlib", "runpy"}:
                    raise ControlV2Error(
                        f"dynamic import helper module is forbidden: {relative}"
                    )
                add_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise ControlV2Error(
                    f"relative import is forbidden in frozen scientific source: {relative}"
                )
            if not node.module:
                raise ControlV2Error(f"empty import source is forbidden: {relative}")
            if node.module.split(".", 1)[0] in {"importlib", "runpy"}:
                raise ControlV2Error(
                    f"dynamic import helper module is forbidden: {relative}"
                )
            add_module(node.module)
            base = node.module.replace(".", "/")
            package_init = f"{base}/__init__.py"
            if _existing_repo_path(repo_root, package_init) is not None:
                discovered.add(package_init)
            for alias in node.names:
                if alias.name != "*":
                    add_module(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__",
                "compile",
                "eval",
                "exec",
            }:
                raise ControlV2Error(
                    f"dynamic code/import call is forbidden in scientific source: {relative}"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            ):
                raise ControlV2Error(
                    f"dynamic importlib call is forbidden in scientific source: {relative}"
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "exec_module",
                "find_module",
                "find_spec",
                "load_module",
                "run_module",
                "run_path",
            }:
                raise ControlV2Error(
                    f"dynamic loader call is forbidden in scientific source: {relative}"
                )
    return tuple(sorted(discovered, key=str.encode))


def recompute_science_import_closure(repo_root: Path) -> tuple[str, ...]:
    """Return the AST-derived local closure of the five direct mechanism roots."""

    pending = list(MECHANISM_ROOT_PATHS)
    visited: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        source = _read_anchored_regular(repo_root, relative)
        visited.add(relative)
        for imported in _local_import_paths(repo_root, relative, source.payload):
            if imported not in visited:
                pending.append(imported)
    return tuple(sorted(visited, key=str.encode))


def validate_science_import_closure(repo_root: Path) -> tuple[str, ...]:
    observed = recompute_science_import_closure(repo_root)
    if observed != SCIENCE_SOURCE_PATHS:
        raise ControlV2Error("AST-derived local scientific import closure differs")
    return observed


def _archive_entry_from_value(value: object) -> SourceArchiveEntry:
    if not isinstance(value, Mapping):
        raise ControlV2Error("source archive entry must be an object")
    _require_exact_keys(
        value,
        frozenset({"bytes_b64", "mode", "path", "sha256", "size"}),
        "source archive entry",
    )
    path = _require_string(value["path"], "archive path")
    _validate_relative_path(path)
    mode = _require_int(value["mode"], "archive mode")
    size = _require_int(value["size"], "archive size")
    if not 0 <= mode <= 0o7777 or size < 0:
        raise ControlV2Error("archive mode or size is outside its valid range")
    digest = _require_sha256(value["sha256"], "archive SHA-256")
    encoded = _require_string(value["bytes_b64"], "archive base64 bytes")
    if any(character.isspace() for character in encoded):
        raise ControlV2Error("archive base64 contains whitespace")
    try:
        decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as error:
        raise ControlV2Error("archive base64 bytes are malformed") from error
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise ControlV2Error("archive base64 is not in canonical padded form")
    if len(decoded) != size or sha256(decoded).hexdigest() != digest:
        raise ControlV2Error("archive decoded bytes differ from size or SHA-256")
    return SourceArchiveEntry(path, mode, size, digest, encoded)


def _validate_source_archive(repo_root: Path, archive_path: str) -> SourceArchive:
    """Validate archive syntax, membership, closure, and every live source byte."""

    archive_file = _read_anchored_regular(repo_root, archive_path)
    if archive_file.mode != 0o444:
        raise ControlV2Error("source archive must be read-only mode 0444")
    value = parse_canonical_json_bytes(archive_file.payload)
    if not isinstance(value, Mapping):
        raise ControlV2Error("source archive must be a JSON object")
    _require_exact_keys(value, frozenset({"files", "schema_version"}), "source archive")
    if value["schema_version"] != SOURCE_ARCHIVE_SCHEMA:
        raise ControlV2Error("source archive schema version differs")
    raw_files = value["files"]
    if not isinstance(raw_files, list):
        raise ControlV2Error("source archive files must be an ordered list")
    entries = tuple(_archive_entry_from_value(item) for item in raw_files)
    paths = tuple(entry.path for entry in entries)
    if paths != ARCHIVE_MEMBER_PATHS or len(paths) != len(set(paths)):
        raise ControlV2Error("source archive membership, order, or uniqueness differs")
    validate_science_import_closure(repo_root)
    for entry in entries:
        live = _read_anchored_regular(repo_root, entry.path)
        if (
            live.mode != entry.mode
            or live.size != entry.size
            or live.sha256 != entry.sha256
            or base64.b64encode(live.payload).decode("ascii") != entry.bytes_b64
        ):
            raise ControlV2Error(f"live source bytes drifted after archive: {entry.path}")
    return SourceArchive(archive_path, archive_file.sha256, entries)


def _build_source_archive(repo_root: Path, archive_path: str) -> SourceArchive:
    """Exclusively create and read back the exact self-describing source archive."""

    _validate_relative_path(archive_path)
    if archive_path in ARCHIVE_MEMBER_PATHS:
        raise ControlV2Error("source archive cannot recursively contain itself")
    validate_science_import_closure(repo_root)
    entries = tuple(
        SourceArchiveEntry.from_file(_read_anchored_regular(repo_root, relative))
        for relative in ARCHIVE_MEMBER_PATHS
    )
    payload = canonical_json_bytes(
        {
            "files": [entry.as_dict() for entry in entries],
            "schema_version": SOURCE_ARCHIVE_SCHEMA,
        }
    )
    _write_exclusive_readonly(repo_root, archive_path, payload)
    return _validate_source_archive(repo_root, archive_path)


def build_source_archive() -> SourceArchive:
    """Refuse formal admission until v2.2 fixes closure and nonce ordering."""

    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def validate_source_archive() -> SourceArchive:
    """Refuse formal validation until v2.2 fixes closure and nonce ordering."""

    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


PARENT_ROOT: Final = "bench/multimodal_resolution_diagnostics/results/MM-007"
_PARENT_SPECS: Final[tuple[tuple[str, str, int], ...]] = (
    (
        "artifact-manifest.json",
        "db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb",
        0o644,
    ),
    (
        "input-manifest.json",
        "1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f",
        0o644,
    ),
    (
        "formal-start.json",
        "ea5c7bda870d71ead3172c1fc6e504d6a6b02d2ba785e9fd2fc75a91c667eee3",
        0o444,
    ),
    (
        "MM-007-evidence.json",
        "13dfa89e541e6122263ea9814d42fb328da303dcc74556cdaaa5d5860d99abaf",
        0o644,
    ),
    (
        "MM-007-results.json",
        "3c92729e1e5c18c14461e36602bdb86acd31750d9f5a85f535cd33a43fb9c47b",
        0o644,
    ),
    (
        "MM-007-report.md",
        "b18760128941ab2eff893b8c0afc469b92f71077d489e060d56519407990b8a2",
        0o644,
    ),
    (
        "MM-007-protocol.md",
        "24bbac1855cc2b51d2a65012b9c63037637c53555b86bbad7c66a6249108a73c",
        0o644,
    ),
    (
        "MM-007-frames-64x64.npz",
        "fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f",
        0o644,
    ),
)


@dataclass(frozen=True, slots=True)
class ParentPin:
    path: str
    sha256: str
    mode: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        _require_sha256(self.sha256, "parent pin SHA-256")
        if self.mode not in {0o444, 0o644}:
            raise ControlV2Error("parent pin mode must be 0444 or 0644")

    def as_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "path": self.path, "sha256": self.sha256}


PARENT_PINS: Final[tuple[ParentPin, ...]] = tuple(
    ParentPin(f"{PARENT_ROOT}/{name}", digest, mode)
    for name, digest, mode in _PARENT_SPECS
)


@dataclass(frozen=True, slots=True)
class FileRecord:
    path: str
    mode: int
    size: int
    sha256: str

    @classmethod
    def from_file(cls, value: AnchoredFile) -> FileRecord:
        return cls(value.path, value.mode, value.size, value.sha256)

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if not 0 <= self.mode <= 0o7777 or self.size < 0:
            raise ControlV2Error("file record mode or size is outside its valid range")
        _require_sha256(self.sha256, "file record SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class NonceReceiptBinding:
    path: str
    status: Literal["fresh_authorized", "retired_unused"]
    file_sha256: str
    receipt: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        _require_sha256(self.file_sha256, "nonce receipt file SHA-256")
        expected_keys = (
            "created_at_utc",
            "nonce_hex",
            "protocol_sha256",
            "reviewer_id",
            "schema_version",
        )
        if tuple(key for key, _ in self.receipt) != expected_keys:
            raise ControlV2Error("nonce receipt binding key order differs")

    def as_dict(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "path": self.path,
            "receipt": dict(self.receipt),
            "status": self.status,
        }

    @property
    def receipt_dict(self) -> dict[str, str]:
        return dict(self.receipt)


_NONCE_KEYS: Final = frozenset(
    {"created_at_utc", "nonce_hex", "protocol_sha256", "reviewer_id", "schema_version"}
)


def _parse_nonce_receipt(payload: bytes, expected_protocol_sha256: str) -> tuple[tuple[str, str], ...]:
    value = parse_canonical_json_bytes(payload)
    if not isinstance(value, Mapping):
        raise ControlV2Error("nonce receipt must be an object")
    _require_exact_keys(value, _NONCE_KEYS, "nonce receipt")
    fields = {key: _require_string(value[key], f"nonce receipt {key}") for key in _NONCE_KEYS}
    _require_sha256(fields["nonce_hex"], "nonce hex")
    _require_sha256(fields["protocol_sha256"], "nonce protocol SHA-256")
    if fields["protocol_sha256"] != expected_protocol_sha256:
        raise ControlV2Error("nonce receipt is bound to the wrong protocol")
    if fields["reviewer_id"] != NONCE_REVIEWER or fields["schema_version"] != NONCE_SCHEMA:
        raise ControlV2Error("nonce receipt reviewer or schema differs")
    try:
        parsed_time = datetime.strptime(fields["created_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ControlV2Error("nonce timestamp is not whole-second Gregorian UTC") from error
    if parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ") != fields["created_at_utc"]:
        raise ControlV2Error("nonce timestamp spelling is noncanonical")
    return tuple((key, fields[key]) for key in sorted(_NONCE_KEYS))


@dataclass(frozen=True, slots=True)
class ScienceMaterials:
    """Raw frozen configuration materials whose hashes are recomputed, never trusted."""

    method_config_json: bytes
    candidate_order_bytes: bytes
    synthetic_config_json: bytes

    def __post_init__(self) -> None:
        for name, payload in (
            ("method config", self.method_config_json),
            ("candidate order", self.candidate_order_bytes),
            ("synthetic config", self.synthetic_config_json),
        ):
            if not isinstance(payload, bytes) or not payload:
                raise ControlV2Error(f"{name} material must be nonempty immutable bytes")
        _parse_canonical_json_without_lf(self.method_config_json, "method config")
        _parse_canonical_json_without_lf(self.synthetic_config_json, "synthetic config")


def _parse_canonical_json_without_lf(payload: bytes, name: str) -> object:
    if payload.endswith(b"\n"):
        raise ControlV2Error(f"{name} must omit a trailing LF in its hashed representation")
    return parse_canonical_json_bytes(payload + b"\n")


@dataclass(frozen=True, slots=True)
class ScienceHashes:
    method_config_sha256: str
    candidate_order_sha256: str
    candidate_order_size: int
    synthetic_config_sha256: str

    @classmethod
    def from_materials(cls, materials: ScienceMaterials) -> ScienceHashes:
        return cls(
            method_config_sha256=sha256(
                b"MM008-v2-config\0" + materials.method_config_json
            ).hexdigest(),
            candidate_order_sha256=sha256(materials.candidate_order_bytes).hexdigest(),
            candidate_order_size=len(materials.candidate_order_bytes),
            synthetic_config_sha256=sha256(
                b"MM008-v2.1-synthetic-config\0" + materials.synthetic_config_json
            ).hexdigest(),
        )

    def __post_init__(self) -> None:
        _require_sha256(self.method_config_sha256, "method config SHA-256")
        _require_sha256(self.candidate_order_sha256, "candidate order SHA-256")
        _require_sha256(self.synthetic_config_sha256, "synthetic config SHA-256")
        if self.candidate_order_size <= 0:
            raise ControlV2Error("candidate order size must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_order_sha256": self.candidate_order_sha256,
            "candidate_order_size": self.candidate_order_size,
            "method_config_sha256": self.method_config_sha256,
            "synthetic_config_sha256": self.synthetic_config_sha256,
        }


SYS_FLAG_NAMES: Final[tuple[str, ...]] = (
    "bytes_warning",
    "debug",
    "dev_mode",
    "dont_write_bytecode",
    "hash_randomization",
    "ignore_environment",
    "inspect",
    "int_max_str_digits",
    "interactive",
    "isolated",
    "no_site",
    "no_user_site",
    "optimize",
    "quiet",
    "safe_path",
    "utf8_mode",
    "verbose",
    "warn_default_encoding",
)
SANITIZED_ENV_NAMES: Final[tuple[str, ...]] = (
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "VECLIB_MAXIMUM_THREADS",
)
NUMERIC_ENV_NAMES: Final = frozenset(SANITIZED_ENV_NAMES[2:])


def _science_module_names() -> tuple[str, ...]:
    names: list[str] = []
    for relative in SCIENCE_SOURCE_PATHS:
        name = relative[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        names.append(name)
    return tuple(sorted((*names, "numpy"), key=str.encode))


RUNTIME_MODULE_NAMES: Final = _science_module_names()


@dataclass(frozen=True, slots=True)
class ModuleOrigin:
    name: str
    origin: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.name.isascii()
            or self.name.startswith(".")
            or self.name.endswith(".")
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
                for character in self.name
            )
        ):
            raise ControlV2Error("runtime module name is malformed")
        if not Path(self.origin).is_absolute() or "\x00" in self.origin:
            raise ControlV2Error("runtime module origin must be an absolute path")
        _require_sha256(self.sha256, "runtime module origin SHA-256")

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "origin": self.origin, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    python_executable: str
    python_prefix: str
    python_real_executable: str
    python_binary_sha256: str
    implementation: str
    python_version: str
    python_build: tuple[str, str]
    numpy_version: str
    numpy_origin: str
    numpy_origin_sha256: str
    numpy_tree_sha256: str
    numpy_build_sha256: str
    pip_freeze: tuple[str, ...]
    module_origins: tuple[ModuleOrigin, ...]
    sys_flags: tuple[tuple[str, int], ...]
    cwd: str
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment": {key: value for key, value in self.environment},
            "implementation": self.implementation,
            "module_origins": [origin.as_dict() for origin in self.module_origins],
            "numpy": {
                "build_sha256": self.numpy_build_sha256,
                "origin": self.numpy_origin,
                "origin_sha256": self.numpy_origin_sha256,
                "tree_sha256": self.numpy_tree_sha256,
                "version": self.numpy_version,
            },
            "pip_freeze": list(self.pip_freeze),
            "python_binary_sha256": self.python_binary_sha256,
            "python_build": list(self.python_build),
            "python_executable": self.python_executable,
            "python_prefix": self.python_prefix,
            "python_real_executable": self.python_real_executable,
            "python_version": self.python_version,
            "sys_flags": {key: value for key, value in self.sys_flags},
        }


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    counters: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, int]:
        return dict(self.counters)


NO_USE_COUNTER_NAMES: Final[tuple[str, ...]] = (
    "challenge_seed_derivations",
    "pcg64_instantiations",
    "real_data_reads",
    "scientific_fitter_calls",
    "synthetic_generator_calls",
)

RuntimeProbe = Callable[[], RuntimeSnapshot]
UsageProbe = Callable[[], UsageSnapshot]


@dataclass(frozen=True, slots=True)
class _RuntimePolicy:
    python_executable: str
    python_prefix: str
    cwd: str
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _FreezePolicy:
    repo_root: Path
    archive_path: str
    formal_start_path: str
    terminal_receipt_path: str
    protocol_sha256: str
    fresh_receipt_sha256: str
    retired_protocol_sha256: str
    retired_receipt_sha256: str
    parent_pins: tuple[ParentPin, ...]
    runtime: _RuntimePolicy


def _default_runtime_policy(repo_root: Path) -> _RuntimePolicy:
    executable = str((repo_root / ".venv/bin/python").absolute())
    environment = (
        ("LANG", "C.UTF-8"),
        ("LC_ALL", "C.UTF-8"),
        ("MKL_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("PYTHONHASHSEED", "0"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
    )
    return _RuntimePolicy(
        python_executable=executable,
        python_prefix=str((repo_root / ".venv").absolute()),
        cwd=str(repo_root.absolute()),
        argv=(
            executable,
            "-m",
            "bench.multimodal_mechanism_diagnostics.control_v2",
            "formal-run",
        ),
        environment=environment,
    )


DEFAULT_FREEZE_POLICY: Final = _FreezePolicy(
    repo_root=REPO_ROOT,
    archive_path=SOURCE_ARCHIVE_PATH,
    formal_start_path=FORMAL_START_PATH,
    terminal_receipt_path=TERMINAL_RECEIPT_PATH,
    protocol_sha256=PROTOCOL_SHA256,
    fresh_receipt_sha256=FRESH_RECEIPT_SHA256,
    retired_protocol_sha256=RETIRED_PROTOCOL_SHA256,
    retired_receipt_sha256=RETIRED_RECEIPT_SHA256,
    parent_pins=PARENT_PINS,
    runtime=_default_runtime_policy(REPO_ROOT),
)


def _validate_runtime_snapshot(snapshot: RuntimeSnapshot, policy: _RuntimePolicy) -> None:
    if not isinstance(snapshot, RuntimeSnapshot):
        raise ControlV2Error("runtime probe did not return a RuntimeSnapshot")
    expected_fixed = (
        (snapshot.python_executable, policy.python_executable, "python executable"),
        (snapshot.python_prefix, policy.python_prefix, "python prefix"),
        (snapshot.cwd, policy.cwd, "working directory"),
        (snapshot.implementation, "CPython", "Python implementation"),
    )
    for observed, expected, name in expected_fixed:
        if observed != expected:
            raise ControlV2Error(f"runtime {name} differs from the frozen policy")
    for path_name, path_value in (
        ("python executable", snapshot.python_executable),
        ("python prefix", snapshot.python_prefix),
        ("real Python executable", snapshot.python_real_executable),
        ("NumPy origin", snapshot.numpy_origin),
    ):
        if not Path(path_value).is_absolute():
            raise ControlV2Error(f"runtime {path_name} must be absolute")
    if Path(snapshot.python_real_executable).is_symlink():
        raise ControlV2Error("resolved Python executable may not remain a symlink")
    for digest_name, digest in (
        ("Python binary", snapshot.python_binary_sha256),
        ("NumPy origin", snapshot.numpy_origin_sha256),
        ("NumPy tree", snapshot.numpy_tree_sha256),
        ("NumPy build", snapshot.numpy_build_sha256),
    ):
        _require_sha256(digest, f"{digest_name} SHA-256")
    if not snapshot.python_version or not snapshot.numpy_version or len(snapshot.python_build) != 2:
        raise ControlV2Error("runtime version/build strings are incomplete")
    if (
        not snapshot.pip_freeze
        or snapshot.pip_freeze != tuple(sorted(snapshot.pip_freeze, key=str.encode))
        or len(snapshot.pip_freeze) != len(set(snapshot.pip_freeze))
        or any(not line or not line.isascii() or "\n" in line or "\r" in line for line in snapshot.pip_freeze)
    ):
        raise ControlV2Error("pip-freeze receipt must be nonempty, unique, sorted ASCII lines")
    if tuple(origin.name for origin in snapshot.module_origins) != RUNTIME_MODULE_NAMES:
        raise ControlV2Error("runtime module-origin membership or order differs")
    if tuple(name for name, _ in snapshot.sys_flags) != SYS_FLAG_NAMES:
        raise ControlV2Error("sys.flags membership or order differs")
    if any(isinstance(value, bool) or not isinstance(value, int) for _, value in snapshot.sys_flags):
        raise ControlV2Error("sys.flags values must be normalized integers")
    if snapshot.argv != policy.argv:
        raise ControlV2Error("runtime argv differs from the one formal command schema")
    if snapshot.environment != policy.environment:
        raise ControlV2Error("sanitized environment differs from the exact allowlist")
    if tuple(key for key, _ in snapshot.environment) != SANITIZED_ENV_NAMES:
        raise ControlV2Error("sanitized environment membership or order differs")
    for key, value in snapshot.environment:
        if key in NUMERIC_ENV_NAMES:
            if not value.isascii() or not value.isdecimal():
                raise ControlV2Error(f"sanitized numeric environment value is invalid: {key}")
            number = int(value)
            if key == "PYTHONHASHSEED":
                if not 0 <= number <= 4_294_967_295:
                    raise ControlV2Error("PYTHONHASHSEED is outside CPython's range")
            elif number <= 0:
                raise ControlV2Error(f"thread-count environment value must be positive: {key}")
        elif value not in {"C", "C.UTF-8"}:
            raise ControlV2Error("locale environment must be C or C.UTF-8")


def _validate_runtime_file_bindings(
    snapshot: RuntimeSnapshot,
    repo_root: Path,
    archived: Mapping[str, SourceArchiveEntry | FileRecord],
) -> None:
    executable = Path(snapshot.python_executable)
    real_executable = Path(snapshot.python_real_executable)
    try:
        if executable.resolve(strict=True) != real_executable:
            raise ControlV2Error("lexical .venv Python does not resolve to recorded real binary")
    except OSError as error:
        raise ControlV2Error("lexical .venv Python cannot be resolved") from error
    python_bytes, _ = _read_absolute_regular(real_executable)
    if sha256(python_bytes).hexdigest() != snapshot.python_binary_sha256:
        raise ControlV2Error("real Python binary SHA-256 differs")
    numpy_path = Path(snapshot.numpy_origin)
    try:
        numpy_path.relative_to(Path(snapshot.python_prefix))
    except ValueError as error:
        raise ControlV2Error("NumPy origin is outside the exact virtual environment") from error
    numpy_bytes, _ = _read_absolute_regular(numpy_path)
    if sha256(numpy_bytes).hexdigest() != snapshot.numpy_origin_sha256:
        raise ControlV2Error("NumPy origin SHA-256 differs")
    source_by_module: dict[str, SourceArchiveEntry | FileRecord] = {}
    for relative in SCIENCE_SOURCE_PATHS:
        name = relative[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        source_by_module[name] = archived[relative]
    for origin in snapshot.module_origins:
        if origin.name == "numpy":
            if origin.origin != snapshot.numpy_origin or origin.sha256 != snapshot.numpy_origin_sha256:
                raise ControlV2Error("NumPy module-origin record differs")
            continue
        entry = source_by_module[origin.name]
        expected_origin = str((repo_root / entry.path).absolute())
        if origin.origin != expected_origin or origin.sha256 != entry.sha256:
            raise ControlV2Error(f"scientific module origin differs: {origin.name}")


def _validate_usage_snapshot(snapshot: UsageSnapshot) -> None:
    if not isinstance(snapshot, UsageSnapshot):
        raise ControlV2Error("usage probe did not return a UsageSnapshot")
    if tuple(name for name, _ in snapshot.counters) != NO_USE_COUNTER_NAMES:
        raise ControlV2Error("no-use counter membership or order differs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for _, value in snapshot.counters):
        raise ControlV2Error("every pre-marker no-use counter must be integer zero")


@dataclass(frozen=True, slots=True)
class FreezeRecord:
    archive: FileRecord
    live_files: tuple[FileRecord, ...]
    live_fileset_sha256: str
    protocol: FileRecord
    fresh_receipt: NonceReceiptBinding
    retired_receipt: NonceReceiptBinding
    science: ScienceHashes
    parent_files: tuple[FileRecord, ...]
    runtime: RuntimeSnapshot
    no_use: UsageSnapshot

    def __post_init__(self) -> None:
        _require_sha256(self.live_fileset_sha256, "live fileset SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "archive": self.archive.as_dict(),
            "fresh_receipt": self.fresh_receipt.as_dict(),
            "live_files": [record.as_dict() for record in self.live_files],
            "live_fileset_sha256": self.live_fileset_sha256,
            "no_use": self.no_use.as_dict(),
            "parent_files": [record.as_dict() for record in self.parent_files],
            "protocol": self.protocol.as_dict(),
            "retired_receipt": self.retired_receipt.as_dict(),
            "runtime": self.runtime.as_dict(),
            "schema_version": FREEZE_SCHEMA,
            "science": self.science.as_dict(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def _binding_from_archive(
    archive: SourceArchive,
    *,
    path: str,
    status: Literal["fresh_authorized", "retired_unused"],
    expected_protocol_sha256: str,
    expected_file_sha256: str,
) -> NonceReceiptBinding:
    try:
        entry = archive.by_path()[path]
    except KeyError as error:
        raise ControlV2Error("nonce receipt is missing from the source archive") from error
    decoded = base64.b64decode(entry.bytes_b64.encode("ascii"), validate=True)
    if entry.sha256 != expected_file_sha256:
        raise ControlV2Error("nonce receipt archive hash differs from the frozen receipt")
    receipt = _parse_nonce_receipt(decoded, expected_protocol_sha256)
    return NonceReceiptBinding(path, status, entry.sha256, receipt)


def _file_records_sha256(records: Sequence[FileRecord]) -> str:
    return sha256(canonical_json_bytes([record.as_dict() for record in records])).hexdigest()


def _recompute_freeze_record(
    policy: _FreezePolicy,
    materials: ScienceMaterials,
    runtime_probe: RuntimeProbe,
    usage_probe: UsageProbe,
) -> FreezeRecord:
    archive = _validate_source_archive(policy.repo_root, policy.archive_path)
    archive_file = _read_anchored_regular(policy.repo_root, policy.archive_path)
    archive_record = FileRecord.from_file(archive_file)
    protocol_entry = archive.by_path()[PROTOCOL_PATH]
    if protocol_entry.sha256 != policy.protocol_sha256:
        raise ControlV2Error("archived protocol SHA-256 differs from the frozen protocol")
    protocol_record = FileRecord(
        protocol_entry.path, protocol_entry.mode, protocol_entry.size, protocol_entry.sha256
    )
    fresh = _binding_from_archive(
        archive,
        path=FRESH_RECEIPT_PATH,
        status="fresh_authorized",
        expected_protocol_sha256=policy.protocol_sha256,
        expected_file_sha256=policy.fresh_receipt_sha256,
    )
    retired = _binding_from_archive(
        archive,
        path=RETIRED_RECEIPT_PATH,
        status="retired_unused",
        expected_protocol_sha256=policy.retired_protocol_sha256,
        expected_file_sha256=policy.retired_receipt_sha256,
    )
    if fresh.receipt_dict["nonce_hex"] == retired.receipt_dict["nonce_hex"]:
        raise ControlV2Error("fresh and retired nonce receipts unexpectedly share entropy")
    parent_files: list[FileRecord] = []
    for pin in policy.parent_pins:
        live = _read_anchored_regular(policy.repo_root, pin.path)
        if live.sha256 != pin.sha256 or live.mode != pin.mode:
            raise ControlV2Error(f"live MM-007 parent differs from its pin: {pin.path}")
        parent_files.append(FileRecord.from_file(live))
    runtime = runtime_probe()
    _validate_runtime_snapshot(runtime, policy.runtime)
    _validate_runtime_file_bindings(runtime, policy.repo_root, archive.by_path())
    no_use = usage_probe()
    _validate_usage_snapshot(no_use)
    science = ScienceHashes.from_materials(materials)
    archive_members = tuple(
        FileRecord(entry.path, entry.mode, entry.size, entry.sha256) for entry in archive.entries
    )
    live_files = tuple(
        sorted((*archive_members, archive_record, *parent_files), key=lambda item: item.path.encode())
    )
    if len(live_files) != len(set(record.path for record in live_files)):
        raise ControlV2Error("freeze live-file set contains duplicate paths")
    return FreezeRecord(
        archive=archive_record,
        live_files=live_files,
        live_fileset_sha256=_file_records_sha256(live_files),
        protocol=protocol_record,
        fresh_receipt=fresh,
        retired_receipt=retired,
        science=science,
        parent_files=tuple(parent_files),
        runtime=runtime,
        no_use=no_use,
    )


_CAPABILITY_TOKEN: Final = object()


class _SealedCapability:
    """Process-private authority produced only after a complete freeze recomputation."""

    __slots__ = ("_materials", "_policy", "_record", "_runtime_probe", "_token", "_usage_probe")

    def __init__(
        self,
        policy: _FreezePolicy,
        materials: ScienceMaterials,
        runtime_probe: RuntimeProbe,
        usage_probe: UsageProbe,
        record: FreezeRecord,
        *,
        token: object,
    ) -> None:
        if token is not _CAPABILITY_TOKEN:
            raise ControlV2Error("sealed capability construction is private")
        self._policy = policy
        self._materials = materials
        self._runtime_probe = runtime_probe
        self._usage_probe = usage_probe
        self._record = record
        self._token = token

    @property
    def freeze_record(self) -> FreezeRecord:
        return self._record


def _build_freeze_capability(
    policy: _FreezePolicy,
    materials: ScienceMaterials,
    runtime_probe: RuntimeProbe,
    usage_probe: UsageProbe,
) -> _SealedCapability:
    """Test/integration seam; production callers use the fixed policy wrapper."""

    record = _recompute_freeze_record(policy, materials, runtime_probe, usage_probe)
    return _SealedCapability(
        policy,
        materials,
        runtime_probe,
        usage_probe,
        record,
        token=_CAPABILITY_TOKEN,
    )


def build_freeze_capability(
    materials: ScienceMaterials,
    runtime_probe: RuntimeProbe,
    usage_probe: UsageProbe,
) -> _SealedCapability:
    """Refuse formal admission until v2.2 fixes closure and nonce ordering."""

    del materials, runtime_probe, usage_probe
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def _require_capability(value: object) -> _SealedCapability:
    if not isinstance(value, _SealedCapability) or value._token is not _CAPABILITY_TOKEN:
        raise ControlV2Error("operation requires a process-private sealed capability")
    return value


def _revalidate_freeze_capability_test_only(capability: object) -> FreezeRecord:
    """Recompute every fake-test pre-marker field and reject snapshot drift."""

    sealed = _require_capability(capability)
    observed = _recompute_freeze_record(
        sealed._policy,
        sealed._materials,
        sealed._runtime_probe,
        sealed._usage_probe,
    )
    if observed != sealed._record or observed.canonical_bytes != sealed._record.canonical_bytes:
        raise ControlV2Error("freeze capability no longer matches complete recomputation")
    return observed


def revalidate_freeze_capability(capability: object) -> FreezeRecord:
    """Hard-fail production validation until v2.2 has a disk verifier."""

    del capability
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def _validate_archive_after_marker(sealed: _SealedCapability) -> None:
    """Rehash archived science without reading either external nonce receipt."""

    archive_file = _read_anchored_regular(sealed._policy.repo_root, sealed._policy.archive_path)
    if (
        archive_file.mode != sealed._record.archive.mode
        or archive_file.size != sealed._record.archive.size
        or archive_file.sha256 != sealed._record.archive.sha256
    ):
        raise ControlV2Error("source archive drifted after the formal marker")
    archived = sealed._record.live_files
    forbidden = {FRESH_RECEIPT_PATH, RETIRED_RECEIPT_PATH}
    parent_paths = {pin.path for pin in sealed._policy.parent_pins}
    for record in archived:
        if record.path in forbidden or record.path in parent_paths or record.path == sealed._policy.archive_path:
            continue
        live = _read_anchored_regular(sealed._policy.repo_root, record.path)
        if (
            live.mode != record.mode
            or live.size != record.size
            or live.sha256 != record.sha256
        ):
            raise ControlV2Error(f"live scientific source drifted post-marker: {record.path}")
    runtime = sealed._runtime_probe()
    _validate_runtime_snapshot(runtime, sealed._policy.runtime)
    archived_records = {
        record.path: record
        for record in sealed._record.live_files
        if record.path in ARCHIVE_MEMBER_PATHS
    }
    _validate_runtime_file_bindings(
        runtime, sealed._policy.repo_root, archived_records
    )
    if runtime != sealed._record.runtime:
        raise ControlV2Error("runtime drifted after the formal marker")
    if ScienceHashes.from_materials(sealed._materials) != sealed._record.science:
        raise ControlV2Error("scientific config material drifted after the formal marker")


def _formal_start_payload(sealed: _SealedCapability) -> dict[str, object]:
    return {
        "challenge_receipt": sealed._record.fresh_receipt.receipt_dict,
        "challenge_receipt_file_sha256": sealed._record.fresh_receipt.file_sha256,
        "freeze_record": sealed._record.as_dict(),
        "freeze_record_sha256": sealed._record.sha256,
        "schema_version": FORMAL_START_SCHEMA,
    }


def _write_formal_start_test_only(capability: object) -> FileRecord:
    """Exercise marker mechanics only against a private fake-test capability."""

    sealed = _require_capability(capability)
    if _anchored_path_exists(
        sealed._policy.repo_root, sealed._policy.formal_start_path
    ) or _anchored_path_exists(
        sealed._policy.repo_root, sealed._policy.terminal_receipt_path
    ):
        raise TerminalRunError("formal marker already exists; run identifier is terminal")
    _revalidate_freeze_capability_test_only(sealed)
    payload = canonical_json_bytes(_formal_start_payload(sealed))
    written = _write_exclusive_readonly(
        sealed._policy.repo_root, sealed._policy.formal_start_path, payload
    )
    try:
        _validate_archive_after_marker(sealed)
        _validate_formal_start(sealed, check_live=False)
    except ControlV2Error as error:
        raise TerminalRunError("formal marker exists but post-write revalidation failed") from error
    return FileRecord.from_file(written)


def write_formal_start(capability: object) -> FileRecord:
    """Hard-fail production mutation until the v2.2 lifecycle is executable."""

    del capability
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def _validate_formal_start(capability: object, *, check_live: bool) -> dict[str, object]:
    sealed = _require_capability(capability)
    marker = _read_anchored_regular(sealed._policy.repo_root, sealed._policy.formal_start_path)
    if marker.mode != 0o444:
        raise ControlV2Error("formal marker is not read-only mode 0444")
    value = parse_canonical_json_bytes(marker.payload)
    if not isinstance(value, Mapping) or dict(value) != _formal_start_payload(sealed):
        raise ControlV2Error("formal marker differs from its recomputed capability payload")
    if check_live:
        _validate_archive_after_marker(sealed)
    return cast(dict[str, object], dict(value))


def _validate_formal_start_test_only(capability: object) -> dict[str, object]:
    """Validate fake marker bytes and the permitted post-marker test closure."""

    return _validate_formal_start(capability, check_live=True)


def validate_formal_start(capability: object) -> dict[str, object]:
    """Hard-fail production validation until v2.2 has a disk verifier."""

    del capability
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def _challenge_receipt_from_formal_start_test_only(
    capability: object,
) -> dict[str, str]:
    """Exercise marker-only receipt access against a fake-test capability."""

    marker = _validate_formal_start_test_only(capability)
    raw = marker["challenge_receipt"]
    sealed = _require_capability(capability)
    expected = sealed._record.fresh_receipt.receipt_dict
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise ControlV2Error("embedded challenge receipt differs from the sealed receipt")
    return dict(expected)


def challenge_receipt_from_formal_start(capability: object) -> dict[str, str]:
    """Hard-fail production receipt access until the v2.2 disk verifier exists."""

    del capability
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


@dataclass(frozen=True, slots=True)
class TerminalReceipt:
    status: Literal["completed", "failed"]
    payload_sha256: str
    failure_code: str | None
    diagnostic_b64: str | None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed"}:
            raise ControlV2Error("terminal receipt status differs")
        _require_sha256(self.payload_sha256, "terminal payload SHA-256")
        if self.status == "completed":
            if self.failure_code is not None or self.diagnostic_b64 is not None:
                raise ControlV2Error(
                    "completed terminal receipt cannot have failure diagnostics"
                )
        elif (
            self.failure_code is None
            or not self.failure_code
            or not self.failure_code.isascii()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in self.failure_code)
        ):
            raise ControlV2Error("failed terminal receipt requires a safe ASCII failure code")
        else:
            if not isinstance(self.diagnostic_b64, str) or any(
                character.isspace() for character in self.diagnostic_b64
            ):
                raise ControlV2Error("failed terminal receipt requires canonical diagnostics")
            try:
                diagnostic = base64.b64decode(
                    self.diagnostic_b64.encode("ascii"), validate=True
                )
            except (UnicodeEncodeError, binascii.Error) as error:
                raise ControlV2Error("failed terminal diagnostic base64 is malformed") from error
            if (
                base64.b64encode(diagnostic).decode("ascii") != self.diagnostic_b64
                or sha256(diagnostic).hexdigest() != self.payload_sha256
            ):
                raise ControlV2Error("failed terminal diagnostic bytes or hash differ")

    def as_dict(self, formal_start_sha256: str, freeze_record_sha256: str) -> dict[str, object]:
        return {
            "diagnostic_b64": self.diagnostic_b64,
            "failure_code": self.failure_code,
            "formal_start_sha256": formal_start_sha256,
            "freeze_record_sha256": freeze_record_sha256,
            "payload_sha256": self.payload_sha256,
            "schema_version": TERMINAL_SCHEMA,
            "status": self.status,
        }


def _marker_file(sealed: _SealedCapability) -> AnchoredFile:
    _validate_formal_start(sealed, check_live=False)
    return _read_anchored_regular(sealed._policy.repo_root, sealed._policy.formal_start_path)


def _write_terminal_receipt_test_only(
    capability: object, receipt: TerminalReceipt
) -> FileRecord:
    """Exercise one-shot result mechanics only against a fake-test capability."""

    sealed = _require_capability(capability)
    if _anchored_path_exists(sealed._policy.repo_root, sealed._policy.terminal_receipt_path):
        raise TerminalRunError("terminal receipt already exists")
    marker = _marker_file(sealed)
    if receipt.status == "completed":
        _validate_archive_after_marker(sealed)
    payload = canonical_json_bytes(receipt.as_dict(marker.sha256, sealed._record.sha256))
    written = _write_exclusive_readonly(
        sealed._policy.repo_root, sealed._policy.terminal_receipt_path, payload
    )
    if receipt.status == "completed":
        try:
            _validate_archive_after_marker(sealed)
        except ControlV2Error as error:
            raise TerminalRunError("terminal result exists but post-write revalidation failed") from error
    _validate_terminal_receipt(sealed, check_live=receipt.status == "completed")
    return FileRecord.from_file(written)


def write_terminal_receipt(capability: object, receipt: TerminalReceipt) -> FileRecord:
    """Hard-fail production mutation until the v2.2 lifecycle is executable."""

    del capability, receipt
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def _write_failure_receipt_test_only(
    capability: object, *, failure_code: str, diagnostic_payload: bytes
) -> FileRecord:
    """Persist a terminal failure without treating current live state as valid."""

    if not isinstance(diagnostic_payload, bytes):
        raise ControlV2Error("failure diagnostic payload must be immutable bytes")
    return _write_terminal_receipt_test_only(
        capability,
        TerminalReceipt(
            "failed",
            sha256(diagnostic_payload).hexdigest(),
            failure_code,
            base64.b64encode(diagnostic_payload).decode("ascii"),
        ),
    )


def write_failure_receipt(
    capability: object, *, failure_code: str, diagnostic_payload: bytes
) -> FileRecord:
    """Hard-fail production mutation until the v2.2 lifecycle is executable."""

    del capability, failure_code, diagnostic_payload
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


def _validate_terminal_receipt(
    capability: object, *, check_live: bool
) -> dict[str, object]:
    sealed = _require_capability(capability)
    marker = _marker_file(sealed)
    terminal = _read_anchored_regular(
        sealed._policy.repo_root, sealed._policy.terminal_receipt_path
    )
    if terminal.mode != 0o444:
        raise ControlV2Error("terminal receipt is not mode 0444")
    value = parse_canonical_json_bytes(terminal.payload)
    if not isinstance(value, Mapping):
        raise ControlV2Error("terminal receipt must be an object")
    _require_exact_keys(
        value,
        frozenset(
            {
                "failure_code",
                "diagnostic_b64",
                "formal_start_sha256",
                "freeze_record_sha256",
                "payload_sha256",
                "schema_version",
                "status",
            }
        ),
        "terminal receipt",
    )
    status = value["status"]
    if status not in {"completed", "failed"}:
        raise ControlV2Error("terminal status differs")
    failure = value["failure_code"]
    if failure is not None and not isinstance(failure, str):
        raise ControlV2Error("terminal failure code is malformed")
    reconstructed = TerminalReceipt(
        cast(Literal["completed", "failed"], status),
        _require_sha256(value["payload_sha256"], "terminal payload SHA-256"),
        cast(str | None, failure),
        cast(str | None, value["diagnostic_b64"]),
    ).as_dict(marker.sha256, sealed._record.sha256)
    if (
        dict(value) != reconstructed
        or value["schema_version"] != TERMINAL_SCHEMA
        or value["formal_start_sha256"] != marker.sha256
        or value["freeze_record_sha256"] != sealed._record.sha256
    ):
        raise ControlV2Error("terminal receipt binding differs")
    if check_live:
        _validate_archive_after_marker(sealed)
    return cast(dict[str, object], dict(value))


def _validate_terminal_receipt_test_only(capability: object) -> dict[str, object]:
    """Validate a fake terminal receipt and rehash its test closure."""

    return _validate_terminal_receipt(capability, check_live=True)


def validate_terminal_receipt(capability: object) -> dict[str, object]:
    """Hard-fail production validation until the v2.2 disk verifier exists."""

    del capability
    raise ControlV2Error(LIFECYCLE_ADMISSION_STATUS)


Scenario = Literal[
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "coupled_boundary",
    "constant_target",
]
Arm = Literal[
    "global_translation", "quadrant_translation", "affine", "appearance", "combined"
]
Mode = Literal["full", "p0", "p1"]

SCENARIO_ORDER: Final[tuple[Scenario, ...]] = (
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "coupled_boundary",
    "constant_target",
)
ARM_ORDER: Final[tuple[Arm, ...]] = (
    "global_translation",
    "quadrant_translation",
    "affine",
    "appearance",
    "combined",
)
MODE_ORDER: Final[tuple[Mode, ...]] = ("full", "p0", "p1")
ARM_BANK_PAIRS: Final[tuple[tuple[Scenario, Arm], ...]] = tuple(
    (scenario, arm)
    for scenario in SCENARIO_ORDER
    for arm in (
        ("affine", "combined")
        if scenario == "coupled_boundary"
        else ("appearance", "combined")
        if scenario == "constant_target"
        else ARM_ORDER
    )
)
ORACLE_PAIRS: Final[tuple[tuple[Scenario, Arm], ...]] = (
    ("translation", "affine"),
    ("affine", "affine"),
    ("combined", "combined"),
    ("coupled_boundary", "affine"),
)
MUTATION_PAIRS: Final[tuple[tuple[Scenario, Arm], ...]] = (
    ("affine", "affine"),
    ("combined", "combined"),
)
TRANSPOSE_PAIRS: Final[tuple[tuple[Scenario, Arm], ...]] = (
    ("translation", "affine"),
    ("translation", "combined"),
    ("affine", "affine"),
    ("affine", "combined"),
    ("appearance", "combined"),
    ("combined", "combined"),
)
NAMED_CONTROL_ORDER: Final[tuple[str, ...]] = (
    "replay.bit_exact",
    "q.label_swap_invariant",
    "q.pairing_all_labels",
    "q.hit_any_label_once",
    "q.dual_start_certificate",
    "q.selected_panel_target_blind",
    "sampler.accept_negative_eight",
    "sampler.accept_positive_eight",
    "sampler.reject_negative_nextafter",
    "sampler.reject_positive_nextafter",
    "boundary.interior_zero_clip",
    "boundary.interior_zero_site_flow",
    "boundary.interior_zero_gradient",
    "boundary.coupled_zero_clip",
    "boundary.coupled_site_flow",
    "boundary.coupled_gradient",
)


@dataclass(frozen=True, slots=True, order=True)
class CoverageKey:
    """One exact backend-neutral execution key."""

    category: str
    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.category or not self.category.isascii():
            raise ControlV2Error("coverage category must be nonempty ASCII")
        if not self.parts or any(not part or not part.isascii() for part in self.parts):
            raise ControlV2Error("coverage key parts must be nonempty ASCII strings")

    def as_dict(self) -> dict[str, object]:
        return {"category": self.category, "parts": list(self.parts)}


SCENARIO_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("scenario", (scenario,)) for scenario in SCENARIO_ORDER
)
ARM_BANK_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("arm_bank", (scenario, arm)) for scenario, arm in ARM_BANK_PAIRS
)
ROW_ARM_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("row_arm", (scenario, arm, str(row)))
    for scenario, arm in ARM_BANK_PAIRS
    for row in range(6)
)
ORACLE_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("oracle", (scenario, arm, mode))
    for scenario, arm in ORACLE_PAIRS
    for mode in MODE_ORDER
)
MUTATION_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("mutation", (scenario, arm, f"output_p{parity}"))
    for scenario, arm in MUTATION_PAIRS
    for parity in (0, 1)
)
TRANSPOSE_PAIR_ROW_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("transpose_pair_row", (scenario, arm, str(row)))
    for scenario, arm in TRANSPOSE_PAIRS
    for row in range(6)
)
TRANSPOSE_MODE_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("transpose_mode", (*pair_row.parts, mode))
    for pair_row in TRANSPOSE_PAIR_ROW_KEYS
    for mode in MODE_ORDER
)
NAMED_CONTROL_KEYS: Final[tuple[CoverageKey, ...]] = tuple(
    CoverageKey("named_control", (name,)) for name in NAMED_CONTROL_ORDER
)
EXPECTED_COVERAGE_KEYS: Final[tuple[CoverageKey, ...]] = (
    *SCENARIO_KEYS,
    *ARM_BANK_KEYS,
    *ROW_ARM_KEYS,
    *ORACLE_KEYS,
    *MUTATION_KEYS,
    *TRANSPOSE_PAIR_ROW_KEYS,
    *TRANSPOSE_MODE_KEYS,
    *NAMED_CONTROL_KEYS,
)

if (
    len(SCENARIO_KEYS),
    len(ARM_BANK_KEYS),
    len(ROW_ARM_KEYS),
    len(ORACLE_KEYS),
    len(MUTATION_KEYS),
    len(TRANSPOSE_PAIR_ROW_KEYS),
    len(TRANSPOSE_MODE_KEYS),
) != (8, 34, 204, 12, 4, 36, 108):
    raise RuntimeError("MM-008 exact coverage cardinalities drifted")
if len(EXPECTED_COVERAGE_KEYS) != len(set(EXPECTED_COVERAGE_KEYS)):
    raise RuntimeError("MM-008 exact coverage keys are not unique")


def _coverage_payload(keys: Sequence[CoverageKey]) -> list[dict[str, object]]:
    return [key.as_dict() for key in keys]


def _coverage_hash(keys: Sequence[CoverageKey]) -> str:
    return sha256(canonical_json_bytes(_coverage_payload(keys))).hexdigest()


COVERAGE_CATEGORY_HASHES: Final[dict[str, str]] = {
    "scenarios": _coverage_hash(SCENARIO_KEYS),
    "arm_banks": _coverage_hash(ARM_BANK_KEYS),
    "row_arms": _coverage_hash(ROW_ARM_KEYS),
    "oracles": _coverage_hash(ORACLE_KEYS),
    "mutations": _coverage_hash(MUTATION_KEYS),
    "transpose_pair_rows": _coverage_hash(TRANSPOSE_PAIR_ROW_KEYS),
    "transpose_modes": _coverage_hash(TRANSPOSE_MODE_KEYS),
    "named_controls": _coverage_hash(NAMED_CONTROL_KEYS),
}
COVERAGE_SCHEMA_SHA256: Final = _coverage_hash(EXPECTED_COVERAGE_KEYS)


def validate_coverage_keys(observed: Sequence[CoverageKey]) -> tuple[CoverageKey, ...]:
    """Reject removal, addition, duplication, or reordering of any coverage key."""

    values = tuple(observed)
    if values != EXPECTED_COVERAGE_KEYS or len(values) != len(set(values)):
        raise ControlV2Error("execution coverage keys differ in membership, uniqueness, or order")
    return values


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """One externally computed primitive/result pair; no PASS is inferred here."""

    key: CoverageKey
    primitive_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.primitive_sha256, "coverage primitive SHA-256")
        _require_sha256(self.result_sha256, "coverage result SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key.as_dict(),
            "primitive_sha256": self.primitive_sha256,
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True)
class CoverageReceipt:
    """Structural coverage schema, explicitly not a semantic PASS receipt.

    The v2.2 science backend must independently reconstruct every result before
    this structure can be admitted to a formal marker.  This module intentionally
    exposes no function that upgrades arbitrary bytes into a trusted receipt.
    """

    evidence_sha256: str
    results: tuple[CoverageResult, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.evidence_sha256, "coverage evidence SHA-256")
        validate_coverage_keys(tuple(result.key for result in self.results))

    def as_dict(self) -> dict[str, object]:
        return {
            "category_sha256": dict(COVERAGE_CATEGORY_HASHES),
            "coverage_schema_sha256": COVERAGE_SCHEMA_SHA256,
            "evidence_sha256": self.evidence_sha256,
            "policy_status": COVERAGE_POLICY_STATUS,
            "results": [result.as_dict() for result in self.results],
            "schema_version": COVERAGE_SCHEMA,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()


def _coverage_key_from_value(value: object) -> CoverageKey:
    if not isinstance(value, Mapping):
        raise ControlV2Error("coverage key must be an object")
    _require_exact_keys(value, frozenset({"category", "parts"}), "coverage key")
    category = _require_string(value["category"], "coverage category")
    raw_parts = value["parts"]
    if not isinstance(raw_parts, list) or any(not isinstance(item, str) for item in raw_parts):
        raise ControlV2Error("coverage key parts must be a string list")
    return CoverageKey(category, tuple(cast(list[str], raw_parts)))


def _coverage_result_from_value(value: object) -> CoverageResult:
    if not isinstance(value, Mapping):
        raise ControlV2Error("coverage result must be an object")
    _require_exact_keys(
        value,
        frozenset({"key", "primitive_sha256", "result_sha256"}),
        "coverage result",
    )
    return CoverageResult(
        key=_coverage_key_from_value(value["key"]),
        primitive_sha256=_require_sha256(
            value["primitive_sha256"], "coverage primitive SHA-256"
        ),
        result_sha256=_require_sha256(value["result_sha256"], "coverage result SHA-256"),
    )


def parse_coverage_receipt_structure(payload: bytes) -> CoverageReceipt:
    """Parse structure only; this never certifies the scientific result hashes."""

    value = parse_canonical_json_bytes(payload)
    if not isinstance(value, Mapping):
        raise ControlV2Error("coverage receipt must be an object")
    _require_exact_keys(
        value,
        frozenset(
            {
                "category_sha256",
                "coverage_schema_sha256",
                "evidence_sha256",
                "policy_status",
                "results",
                "schema_version",
            }
        ),
        "coverage receipt",
    )
    if value["schema_version"] != COVERAGE_SCHEMA:
        raise ControlV2Error("coverage receipt schema differs")
    if value["coverage_schema_sha256"] != COVERAGE_SCHEMA_SHA256:
        raise ControlV2Error("coverage schema hash differs")
    if value["policy_status"] != COVERAGE_POLICY_STATUS:
        raise ControlV2Error("coverage policy status differs")
    category_hashes = value["category_sha256"]
    if not isinstance(category_hashes, Mapping) or dict(category_hashes) != COVERAGE_CATEGORY_HASHES:
        raise ControlV2Error("coverage category hashes differ")
    raw_results = value["results"]
    if not isinstance(raw_results, list):
        raise ControlV2Error("coverage receipt results must be a list")
    results = tuple(_coverage_result_from_value(item) for item in raw_results)
    receipt = CoverageReceipt(
        evidence_sha256=_require_sha256(value["evidence_sha256"], "coverage evidence SHA-256"),
        results=results,
    )
    if receipt.canonical_bytes != payload:
        raise ControlV2Error("coverage receipt canonical reconstruction differs")
    return receipt
