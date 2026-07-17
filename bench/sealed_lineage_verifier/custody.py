"""Descriptor-anchored custody primitives for sealed lineage packages.

The verifier treats a package as an exact tree of pinned regular-file bytes.
All traversal below the caller-provided root is relative to already-open
directory descriptors.  A successful snapshot therefore contains only bytes
read from pathname-bound, ``O_NOFOLLOW`` file descriptors; a later copy never
reopens its source.

This module is intentionally experiment-agnostic.  ``ExpectedFile`` records,
``{"sha256", "bytes", "mode"}`` records, and ``(sha256, bytes, mode)`` tuples
are accepted so callers can bind existing pin tables without duplicating the
custody implementation.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, TypeAlias

O_CLOEXEC: Final = getattr(os, "O_CLOEXEC", 0)
O_DIRECTORY: Final = getattr(os, "O_DIRECTORY", 0)
O_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
READ_ONLY_FILE_MODE: Final = 0o444
READ_ONLY_DIRECTORY_MODE: Final = 0o555
_READ_CHUNK: Final = 1024 * 1024


class CustodyError(ValueError):
    """A tree is not the exact, stable, descriptor-bound package expected."""


@dataclass(frozen=True, slots=True)
class ExpectedFile:
    """Content and source-mode pin for one regular file."""

    sha256: str
    bytes: int
    mode: int


RecordMapping: TypeAlias = Mapping[str, object]
RecordTuple: TypeAlias = tuple[str, int, int]
RecordLike: TypeAlias = ExpectedFile | RecordMapping | RecordTuple
DirectoryModes: TypeAlias = Mapping[Path, int]


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """One authenticated payload returned from an already-open source fd."""

    path: Path
    payload: bytes
    source: ExpectedFile


@dataclass(frozen=True, slots=True)
class TreeSnapshot:
    """Immutable-value snapshot of an exact source tree."""

    root: Path
    files: tuple[SnapshotFile, ...]
    directories: tuple[Path, ...]

    @property
    def payloads(self) -> Mapping[Path, bytes]:
        return MappingProxyType({item.path: item.payload for item in self.files})

    @property
    def records(self) -> Mapping[Path, ExpectedFile]:
        return MappingProxyType({item.path: item.source for item in self.files})


_Identity: TypeAlias = tuple[int, int, int, int, int, int, int]
_NamespaceIdentity: TypeAlias = tuple[int, int, int]


@dataclass(slots=True)
class _OpenDirectory:
    path: Path
    descriptor: int
    identity: _Identity
    expected_names: frozenset[str]
    expected_directories: frozenset[str]
    parent_descriptor: int | None
    leaf: str | None


@dataclass(slots=True)
class _OpenFile:
    path: Path
    descriptor: int
    identity: _Identity
    parent_descriptor: int
    leaf: str
    snapshot: SnapshotFile


@dataclass(slots=True)
class _OpenAncestor:
    path: Path
    descriptor: int
    identity: _NamespaceIdentity
    parent_descriptor: int | None
    leaf: str | None


@dataclass(slots=True)
class _OpenedTree:
    root: Path
    root_parent_descriptor: int
    root_leaf: str
    root_lexical_identity: _Identity
    ancestors: list[_OpenAncestor]
    directories: list[_OpenDirectory]
    files: list[_OpenFile]

    def close(self) -> None:
        for opened_file in reversed(self.files):
            _close_quietly(opened_file.descriptor)
        for opened_directory in reversed(self.directories):
            _close_quietly(opened_directory.descriptor)
        for ancestor in reversed(self.ancestors):
            _close_quietly(ancestor.descriptor)


def _close_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _identity(metadata: os.stat_result) -> _Identity:
    """Fields which bind inode, type/mode, length, links, and mutations."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _namespace_identity(metadata: os.stat_result) -> _NamespaceIdentity:
    """Fields that prove a retained pathname edge still names one directory.

    Ancestors outside the authenticated tree can legitimately gain or lose
    unrelated children while a snapshot is in progress.  Their size, link
    count, and timestamps therefore are not stable custody facts.  Device,
    inode, and file type are sufficient to detect a namespace substitution
    while the retained descriptor prevents traversal through a replacement.
    """

    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _validate_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CustodyError("expected sha256 must be 64 lowercase hexadecimal characters")
    return value


def _validate_integer(value: object, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise CustodyError(f"expected {label} is invalid")
    return value


def _normalize_record(value: RecordLike) -> ExpectedFile:
    if isinstance(value, ExpectedFile):
        record = value
    elif isinstance(value, tuple):
        if len(value) != 3:
            raise CustodyError("expected tuple record must be (sha256, bytes, mode)")
        record = ExpectedFile(value[0], value[1], value[2])
    elif isinstance(value, Mapping):
        if set(value) == {"sha256", "bytes", "mode"}:
            size = value["bytes"]
        elif set(value) == {"sha256", "size", "mode"}:
            size = value["size"]
        else:
            raise CustodyError("expected mapping record schema differs")
        record = ExpectedFile(value["sha256"], size, value["mode"])  # type: ignore[arg-type]
    else:
        raise CustodyError("expected file record has an unsupported type")
    return ExpectedFile(
        sha256=_validate_sha256(record.sha256),
        bytes=_validate_integer(record.bytes, "byte count"),
        mode=_validate_integer(record.mode, "mode", maximum=0o7777),
    )


def _normalize_relative_path(value: Path) -> Path:
    if not isinstance(value, Path):
        raise CustodyError("expected tree keys must be pathlib.Path values")
    raw = value.as_posix()
    pure = PurePosixPath(raw)
    if (
        raw in {"", "."}
        or "\\" in raw
        or "\x00" in raw
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != raw
    ):
        raise CustodyError(f"expected path is not normalized and relative: {raw!r}")
    return Path(*pure.parts)


def normalize_expectations(expected: Mapping[Path, RecordLike]) -> dict[Path, ExpectedFile]:
    """Validate and canonicalize a generic ``{Path: record}`` pin table."""

    if not isinstance(expected, Mapping) or not expected:
        raise CustodyError("expected tree must be a nonempty mapping")
    output: dict[Path, ExpectedFile] = {}
    for candidate, value in expected.items():
        relative = _normalize_relative_path(candidate)
        if relative in output:
            raise CustodyError(f"duplicate normalized expected path: {relative}")
        output[relative] = _normalize_record(value)
    file_paths = set(output)
    for relative in file_paths:
        parent = relative.parent
        while parent != Path("."):
            if parent in file_paths:
                raise CustodyError(f"expected file is also a directory ancestor: {parent}")
            parent = parent.parent
    return dict(sorted(output.items(), key=lambda item: item[0].as_posix()))


def expected_directories(expected: Mapping[Path, RecordLike]) -> tuple[Path, ...]:
    """Return the exact non-root directory closure implied by a pin table."""

    normalized = normalize_expectations(expected)
    directories: set[Path] = set()
    for relative in normalized:
        parent = relative.parent
        while parent != Path("."):
            directories.add(parent)
            parent = parent.parent
    return tuple(sorted(directories, key=lambda path: path.as_posix()))


def sealed_expectations(expected: Mapping[Path, RecordLike]) -> dict[Path, ExpectedFile]:
    """Translate authenticated source records to the copied-file mode contract."""

    return {
        path: ExpectedFile(record.sha256, record.bytes, READ_ONLY_FILE_MODE)
        for path, record in normalize_expectations(expected).items()
    }


def _absolute_lexical(path: Path) -> Path:
    if not isinstance(path, Path):
        raise CustodyError("tree root must be a pathlib.Path")
    if "\x00" in os.fspath(path) or any(part == ".." for part in path.parts):
        raise CustodyError("tree root contains forbidden traversal")
    candidate = path if path.is_absolute() else Path.cwd() / path
    if candidate == Path(candidate.anchor):
        raise CustodyError("filesystem root cannot be used as a custody tree")
    return candidate


def _assert_ancestor_chain(ancestors: list[_OpenAncestor]) -> None:
    """Prove every retained namespace edge still names its opened directory."""

    for ancestor in ancestors:
        if _namespace_identity(os.fstat(ancestor.descriptor)) != ancestor.identity:
            raise CustodyError(f"tree ancestor descriptor changed: {ancestor.path}")
        try:
            if ancestor.parent_descriptor is None:
                path_metadata = os.lstat(ancestor.path)
            else:
                assert ancestor.leaf is not None
                path_metadata = os.stat(
                    ancestor.leaf,
                    dir_fd=ancestor.parent_descriptor,
                    follow_symlinks=False,
                )
        except OSError as error:
            raise CustodyError(f"tree ancestor disappeared: {ancestor.path}") from error
        if _namespace_identity(path_metadata) != ancestor.identity:
            raise CustodyError(f"tree ancestor changed while retained: {ancestor.path}")


def _close_ancestors(ancestors: list[_OpenAncestor]) -> None:
    for ancestor in reversed(ancestors):
        _close_quietly(ancestor.descriptor)


def _open_anchored_parent(path: Path) -> tuple[Path, int, str, list[_OpenAncestor]]:
    """Open every existing parent component without following a symlink."""

    lexical = _absolute_lexical(path)
    parts = lexical.parts
    current = os.open(parts[0], os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
    root_identity = _namespace_identity(os.fstat(current))
    ancestors = [_OpenAncestor(Path(parts[0]), current, root_identity, None, None)]
    try:
        for index, part in enumerate(parts[1:-1], start=1):
            try:
                before = os.stat(part, dir_fd=current, follow_symlinks=False)
            except OSError as error:
                raise CustodyError(f"tree ancestor cannot be inspected: {part}") from error
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise CustodyError(f"tree ancestor is not a real directory: {part}")
            try:
                child = os.open(part, os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW, dir_fd=current)
            except OSError as error:
                raise CustodyError(f"tree ancestor cannot be opened securely: {part}") from error
            opened = os.fstat(child)
            if _namespace_identity(before) != _namespace_identity(opened):
                os.close(child)
                raise CustodyError(f"tree ancestor changed while opened: {part}")
            ancestors.append(
                _OpenAncestor(
                    Path(*parts[: index + 1]),
                    child,
                    _namespace_identity(opened),
                    current,
                    part,
                )
            )
            current = child
        _assert_ancestor_chain(ancestors)
        return lexical, current, parts[-1], ancestors
    except BaseException:
        _close_ancestors(ancestors)
        raise


def _open_existing_root(root: Path) -> tuple[Path, int, str, _Identity, int, list[_OpenAncestor]]:
    lexical, parent, leaf, ancestors = _open_anchored_parent(root)
    try:
        lexical_before = os.lstat(lexical)
        before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        _close_ancestors(ancestors)
        raise CustodyError(f"tree root is missing or inaccessible: {lexical}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        _close_ancestors(ancestors)
        raise CustodyError(f"tree root is not a real directory: {lexical}")
    if _identity(lexical_before) != _identity(before):
        _close_ancestors(ancestors)
        raise CustodyError("tree root lexical and anchored identities differ")
    try:
        descriptor = os.open(leaf, os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW, dir_fd=parent)
    except OSError as error:
        _close_ancestors(ancestors)
        raise CustodyError(f"tree root cannot be opened securely: {lexical}") from error
    opened = os.fstat(descriptor)
    if _identity(before) != _identity(opened):
        os.close(descriptor)
        _close_ancestors(ancestors)
        raise CustodyError("tree root changed while opened")
    _assert_ancestor_chain(ancestors)
    return lexical, parent, leaf, _identity(opened), descriptor, ancestors


def _child_contracts(
    normalized: Mapping[Path, ExpectedFile], directories: tuple[Path, ...]
) -> tuple[dict[Path, frozenset[str]], dict[Path, frozenset[str]]]:
    all_directories = {Path("."), *directories}
    names: dict[Path, set[str]] = {path: set() for path in all_directories}
    directory_names: dict[Path, set[str]] = {path: set() for path in all_directories}
    for directory in directories:
        names[directory.parent].add(directory.name)
        directory_names[directory.parent].add(directory.name)
    for relative in normalized:
        names[relative.parent].add(relative.name)
    return (
        {path: frozenset(values) for path, values in names.items()},
        {path: frozenset(values) for path, values in directory_names.items()},
    )


def _scan_directory(directory: _OpenDirectory) -> None:
    scan_descriptor = -1
    try:
        # A fresh open file description keeps scandir's directory offset from
        # affecting subsequent pre/post censuses on the retained custody fd.
        scan_descriptor = os.open(
            ".",
            os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
            dir_fd=directory.descriptor,
        )
        if _identity(os.fstat(scan_descriptor)) != _identity(os.fstat(directory.descriptor)):
            raise CustodyError(f"directory identity changed before census: {directory.path}")
        with os.scandir(scan_descriptor) as entries:
            names = {entry.name for entry in entries}
    except CustodyError:
        raise
    except OSError as error:
        raise CustodyError(f"directory census failed: {directory.path}") from error
    finally:
        if scan_descriptor >= 0:
            _close_quietly(scan_descriptor)
    if names != directory.expected_names:
        raise CustodyError(
            f"tree membership differs at {directory.path}: "
            f"expected={sorted(directory.expected_names)}, actual={sorted(names)}"
        )
    for name in sorted(names):
        try:
            metadata = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        except OSError as error:
            raise CustodyError(f"tree entry disappeared during census: {directory.path / name}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise CustodyError(f"tree contains a symlink: {directory.path / name}")
        expected_directory = name in directory.expected_directories
        if expected_directory and not stat.S_ISDIR(metadata.st_mode):
            raise CustodyError(f"expected directory is not a real directory: {directory.path / name}")
        if not expected_directory and not stat.S_ISREG(metadata.st_mode):
            raise CustodyError(f"expected file is not a regular file: {directory.path / name}")
        if not expected_directory and metadata.st_nlink != 1:
            raise CustodyError(f"tree contains a hard-linked file: {directory.path / name}")


def _read_open_file(
    parent: _OpenDirectory,
    relative: Path,
    expected: ExpectedFile,
    *,
    check_mode: bool,
) -> _OpenFile:
    leaf = relative.name
    try:
        before = os.stat(leaf, dir_fd=parent.descriptor, follow_symlinks=False)
    except OSError as error:
        raise CustodyError(f"expected file is missing: {relative}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CustodyError(f"expected file is not regular and non-symlink: {relative}")
    if before.st_nlink != 1:
        raise CustodyError(f"expected file is hard-linked: {relative}")
    if before.st_size != expected.bytes:
        raise CustodyError(f"expected file size differs: {relative}")
    if check_mode and stat.S_IMODE(before.st_mode) != expected.mode:
        raise CustodyError(f"expected file mode differs: {relative}")
    try:
        descriptor = os.open(leaf, os.O_RDONLY | O_CLOEXEC | O_NOFOLLOW, dir_fd=parent.descriptor)
    except OSError as error:
        raise CustodyError(f"expected file cannot be opened securely: {relative}") from error
    try:
        opened = os.fstat(descriptor)
        opened_identity = _identity(opened)
        if _identity(before) != opened_identity or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise CustodyError(f"expected file changed while opened: {relative}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK, remaining))
            if not chunk:
                raise CustodyError(f"expected file ended before its bound size: {relative}")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CustodyError(f"expected file grew while read: {relative}")
        after_descriptor = os.fstat(descriptor)
        after_path = os.stat(leaf, dir_fd=parent.descriptor, follow_symlinks=False)
        if _identity(after_descriptor) != opened_identity or _identity(after_path) != opened_identity:
            raise CustodyError(f"expected file mutated or was replaced while read: {relative}")
        if digest.hexdigest() != expected.sha256:
            raise CustodyError(f"expected file hash differs: {relative}")
        payload = b"".join(chunks)
        if len(payload) != expected.bytes:
            raise CustodyError(f"expected file read length differs: {relative}")
        snapshot = SnapshotFile(relative, payload, expected)
        return _OpenFile(relative, descriptor, opened_identity, parent.descriptor, leaf, snapshot)
    except BaseException:
        os.close(descriptor)
        raise


def _normalize_directory_modes(
    directories: tuple[Path, ...],
    *,
    directory_mode: int | None,
    directory_modes: DirectoryModes | None,
) -> dict[Path, int] | None:
    if directory_mode is not None and directory_modes is not None:
        raise CustodyError("provide either one directory mode or an exact mode map")
    closure = (Path("."), *directories)
    if directory_mode is not None:
        mode = _validate_integer(directory_mode, "directory mode", maximum=0o7777)
        return {path: mode for path in closure}
    if directory_modes is None:
        return None
    if not isinstance(directory_modes, Mapping):
        raise CustodyError("directory modes must be an exact mapping")
    normalized: dict[Path, int] = {}
    for path, mode_value in directory_modes.items():
        if path == Path("."):
            relative = path
        else:
            relative = _normalize_relative_path(path)
        normalized[relative] = _validate_integer(mode_value, "directory mode", maximum=0o7777)
    if set(normalized) != set(closure):
        raise CustodyError("directory mode map differs from the exact directory closure")
    return normalized


def _acquire_tree(
    root: Path,
    expected: Mapping[Path, RecordLike],
    *,
    check_file_modes: bool,
    directory_mode: int | None,
    directory_modes: DirectoryModes | None = None,
) -> _OpenedTree:
    normalized = normalize_expectations(expected)
    directories = expected_directories(normalized)
    expected_names, expected_directory_names = _child_contracts(normalized, directories)
    normalized_directory_modes = _normalize_directory_modes(
        directories,
        directory_mode=directory_mode,
        directory_modes=directory_modes,
    )
    lexical, parent_descriptor, root_leaf, root_identity, root_descriptor, ancestors = _open_existing_root(root)
    opened = _OpenedTree(
        root=lexical,
        root_parent_descriptor=parent_descriptor,
        root_leaf=root_leaf,
        root_lexical_identity=root_identity,
        ancestors=ancestors,
        directories=[
            _OpenDirectory(
                Path("."),
                root_descriptor,
                root_identity,
                expected_names[Path(".")],
                expected_directory_names[Path(".")],
                None,
                None,
            )
        ],
        files=[],
    )
    by_path: dict[Path, _OpenDirectory] = {Path("."): opened.directories[0]}
    try:
        for relative in (Path("."), *directories):
            if relative == Path("."):
                directory = by_path[relative]
            else:
                parent = by_path[relative.parent]
                try:
                    before = os.stat(relative.name, dir_fd=parent.descriptor, follow_symlinks=False)
                except OSError as error:
                    raise CustodyError(f"expected directory is missing: {relative}") from error
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise CustodyError(f"expected directory is not real: {relative}")
                try:
                    descriptor = os.open(
                        relative.name,
                        os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
                        dir_fd=parent.descriptor,
                    )
                except OSError as error:
                    raise CustodyError(f"expected directory cannot be opened securely: {relative}") from error
                identity = _identity(os.fstat(descriptor))
                if identity != _identity(before):
                    os.close(descriptor)
                    raise CustodyError(f"expected directory changed while opened: {relative}")
                directory = _OpenDirectory(
                    relative,
                    descriptor,
                    identity,
                    expected_names[relative],
                    expected_directory_names[relative],
                    parent.descriptor,
                    relative.name,
                )
                opened.directories.append(directory)
                by_path[relative] = directory
            if (
                normalized_directory_modes is not None
                and stat.S_IMODE(os.fstat(directory.descriptor).st_mode) != normalized_directory_modes[relative]
            ):
                raise CustodyError(f"directory mode differs: {relative}")
            _scan_directory(directory)
        for relative, record in normalized.items():
            opened.files.append(
                _read_open_file(by_path[relative.parent], relative, record, check_mode=check_file_modes)
            )
        _assert_tree_unchanged(opened)
        return opened
    except BaseException:
        opened.close()
        raise


def _assert_tree_unchanged(opened: _OpenedTree) -> None:
    """Post-read membership and inode/ctime/mtime checks with all fds live."""

    for directory in opened.directories:
        _scan_directory(directory)
    for item in opened.files:
        descriptor_metadata = os.fstat(item.descriptor)
        try:
            path_metadata = os.stat(item.leaf, dir_fd=item.parent_descriptor, follow_symlinks=False)
        except OSError as error:
            raise CustodyError(f"file disappeared after read: {item.path}") from error
        if _identity(descriptor_metadata) != item.identity or _identity(path_metadata) != item.identity:
            raise CustodyError(f"file changed after read: {item.path}")
    for directory in reversed(opened.directories[1:]):
        assert directory.parent_descriptor is not None and directory.leaf is not None
        descriptor_metadata = os.fstat(directory.descriptor)
        try:
            path_metadata = os.stat(
                directory.leaf,
                dir_fd=directory.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise CustodyError(f"directory disappeared after census: {directory.path}") from error
        if _identity(descriptor_metadata) != directory.identity or _identity(path_metadata) != directory.identity:
            raise CustodyError(f"directory changed during census: {directory.path}")
    root = opened.directories[0]
    try:
        anchored = os.stat(opened.root_leaf, dir_fd=opened.root_parent_descriptor, follow_symlinks=False)
        lexical = os.lstat(opened.root)
    except OSError as error:
        raise CustodyError("tree root disappeared during census") from error
    if (
        _identity(os.fstat(root.descriptor)) != root.identity
        or _identity(anchored) != root.identity
        or _identity(lexical) != opened.root_lexical_identity
    ):
        raise CustodyError("tree root changed during census")
    _assert_ancestor_chain(opened.ancestors)


def snapshot_exact_tree(
    root: Path,
    expected: Mapping[Path, RecordLike],
    *,
    directory_mode: int | None = None,
    directory_modes: DirectoryModes | None = None,
) -> TreeSnapshot:
    """Return authenticated bytes only if the descriptor-relative tree is exact."""

    opened = _acquire_tree(
        root,
        expected,
        check_file_modes=True,
        directory_mode=directory_mode,
        directory_modes=directory_modes,
    )
    try:
        # A final check immediately precedes copying payload references out of the
        # live-fd acquisition.  The returned bytes can no longer alias the source.
        _assert_tree_unchanged(opened)
        return TreeSnapshot(
            root=opened.root,
            files=tuple(item.snapshot for item in opened.files),
            directories=tuple(directory.path for directory in opened.directories[1:]),
        )
    finally:
        opened.close()


def _snapshot_contract(snapshot: TreeSnapshot) -> tuple[dict[Path, ExpectedFile], tuple[Path, ...]]:
    if not isinstance(snapshot, TreeSnapshot) or not snapshot.files:
        raise CustodyError("copy source is not a nonempty TreeSnapshot")
    records: dict[Path, ExpectedFile] = {}
    for item in snapshot.files:
        if not isinstance(item, SnapshotFile):
            raise CustodyError("snapshot contains an invalid file entry")
        relative = _normalize_relative_path(item.path)
        record = _normalize_record(item.source)
        if relative in records:
            raise CustodyError(f"snapshot repeats a file: {relative}")
        if len(item.payload) != record.bytes or hashlib.sha256(item.payload).hexdigest() != record.sha256:
            raise CustodyError(f"snapshot payload no longer matches its record: {relative}")
        records[relative] = record
    normalized = normalize_expectations(records)
    directories = expected_directories(normalized)
    if tuple(snapshot.directories) != directories:
        raise CustodyError("snapshot directory closure differs from its files")
    return normalized, directories


def _fsync_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise CustodyError("durability fsync failed") from error


def write_snapshot_exclusive(destination: Path, snapshot: TreeSnapshot) -> None:
    """Create one sealed copy from snapshot bytes, never reopening the source."""

    records, directories = _snapshot_contract(snapshot)
    payloads = {item.path: item.payload for item in snapshot.files}
    lexical, parent_descriptor, leaf, ancestors = _open_anchored_parent(destination)
    root_descriptor = -1
    directory_descriptors: dict[Path, int] = {}
    file_descriptors: list[int] = []
    try:
        try:
            os.mkdir(leaf, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise CustodyError(f"copy destination already exists: {lexical}") from error
        except OSError as error:
            raise CustodyError(f"copy destination cannot be created: {lexical}") from error
        _fsync_descriptor(parent_descriptor)
        # Creating the destination legitimately changes its direct parent's
        # size/timestamps.  Namespace custody depends only on the retained
        # directory object and its still-bound pathname edge.
        refreshed_parent = os.fstat(parent_descriptor)
        previous_parent = ancestors[-1].identity
        if _namespace_identity(refreshed_parent) != previous_parent:
            raise CustodyError("copy destination parent identity changed during mkdir")
        _assert_ancestor_chain(ancestors)
        root_descriptor = os.open(
            leaf,
            os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        directory_descriptors[Path(".")] = root_descriptor
        for relative in directories:
            parent = directory_descriptors[relative.parent]
            try:
                os.mkdir(relative.name, 0o700, dir_fd=parent)
                descriptor = os.open(
                    relative.name,
                    os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
                    dir_fd=parent,
                )
            except OSError as error:
                raise CustodyError(f"copy directory cannot be created securely: {relative}") from error
            directory_descriptors[relative] = descriptor
        for relative, record in records.items():
            parent = directory_descriptors[relative.parent]
            try:
                descriptor = os.open(
                    relative.name,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | O_CLOEXEC | O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
            except OSError as error:
                raise CustodyError(f"copy file cannot be created exclusively: {relative}") from error
            file_descriptors.append(descriptor)
            payload = payloads[relative]
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise CustodyError(f"copy write made no progress: {relative}")
                offset += written
            os.fchmod(descriptor, READ_ONLY_FILE_MODE)
            _fsync_descriptor(descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != record.bytes
                or stat.S_IMODE(metadata.st_mode) != READ_ONLY_FILE_MODE
            ):
                raise CustodyError(f"copy file post-write metadata differs: {relative}")
        for relative in sorted(directories, key=lambda path: (len(path.parts), path.as_posix()), reverse=True):
            descriptor = directory_descriptors[relative]
            _fsync_descriptor(descriptor)
            os.fchmod(descriptor, READ_ONLY_DIRECTORY_MODE)
            _fsync_descriptor(descriptor)
        _fsync_descriptor(root_descriptor)
        os.fchmod(root_descriptor, READ_ONLY_DIRECTORY_MODE)
        _fsync_descriptor(root_descriptor)
        _fsync_descriptor(parent_descriptor)
    except BaseException:
        # A partially created destination is intentionally left in place.  Silent
        # cleanup could erase the only evidence of an interrupted exclusive copy.
        raise
    finally:
        for descriptor in reversed(file_descriptors):
            _close_quietly(descriptor)
        for relative, descriptor in sorted(
            directory_descriptors.items(),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if relative != Path("."):
                _close_quietly(descriptor)
        if root_descriptor >= 0:
            _close_quietly(root_descriptor)
        _close_ancestors(ancestors)
    verify_sealed_tree(lexical, records)


def copy_exact_tree(
    source: Path,
    destination: Path,
    expected: Mapping[Path, RecordLike],
) -> TreeSnapshot:
    """Authenticate an exact source, then create and verify a sealed copy."""

    snapshot = snapshot_exact_tree(source, expected)
    write_snapshot_exclusive(destination, snapshot)
    return snapshot


def seal_copied_tree(root: Path, expected: Mapping[Path, RecordLike]) -> None:
    """Authenticate an exact existing copy and seal files 0444/directories 0555."""

    opened = _acquire_tree(root, expected, check_file_modes=False, directory_mode=None)
    try:
        _assert_tree_unchanged(opened)
        for item in opened.files:
            os.fchmod(item.descriptor, READ_ONLY_FILE_MODE)
            _fsync_descriptor(item.descriptor)
        for directory in reversed(opened.directories):
            os.fchmod(directory.descriptor, READ_ONLY_DIRECTORY_MODE)
            _fsync_descriptor(directory.descriptor)
        _fsync_descriptor(opened.root_parent_descriptor)
    finally:
        opened.close()
    verify_sealed_tree(root, expected)


def verify_sealed_tree(root: Path, expected: Mapping[Path, RecordLike]) -> TreeSnapshot:
    """Verify exact membership, copied modes, hashes, and stable identities."""

    return snapshot_exact_tree(
        root,
        sealed_expectations(expected),
        directory_mode=READ_ONLY_DIRECTORY_MODE,
    )


def _remove_directory_contents(descriptor: int, relative: Path) -> None:
    try:
        os.fchmod(descriptor, 0o700)
        scan_descriptor = os.open(
            ".",
            os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            with os.scandir(scan_descriptor) as entries:
                names = sorted(entry.name for entry in entries)
        finally:
            _close_quietly(scan_descriptor)
        for name in names:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_path = relative / name
            if stat.S_ISLNK(metadata.st_mode):
                raise CustodyError(f"created tree cleanup encountered a symlink: {child_path}")
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    if _identity(os.fstat(child)) != _identity(metadata):
                        raise CustodyError(f"created directory changed before cleanup: {child_path}")
                    _remove_directory_contents(child, child_path)
                finally:
                    _close_quietly(child)
                os.rmdir(name, dir_fd=descriptor)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                child = os.open(name, os.O_RDONLY | O_CLOEXEC | O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if _identity(os.fstat(child)) != _identity(metadata):
                        raise CustodyError(f"created file changed before cleanup: {child_path}")
                    os.fchmod(child, 0o600)
                finally:
                    _close_quietly(child)
                os.unlink(name, dir_fd=descriptor)
            else:
                raise CustodyError(f"created tree cleanup encountered a special or linked file: {child_path}")
        _fsync_descriptor(descriptor)
    except CustodyError:
        raise
    except OSError as error:
        raise CustodyError(f"created tree could not be removed securely: {relative}") from error


def remove_created_tree(root: Path) -> None:
    """Descriptor-remove one private tree, rejecting links and special entries."""

    lexical, parent, leaf, ancestors = _open_anchored_parent(root)
    descriptor = -1
    try:
        before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise CustodyError(f"created cleanup root is not a real directory: {lexical}")
        descriptor = os.open(
            leaf,
            os.O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW,
            dir_fd=parent,
        )
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise CustodyError("created cleanup root changed while opened")
        _remove_directory_contents(descriptor, Path("."))
        _close_quietly(descriptor)
        descriptor = -1
        os.rmdir(leaf, dir_fd=parent)
        _fsync_descriptor(parent)
        refreshed_parent = os.fstat(parent)
        previous_parent = ancestors[-1].identity
        if _namespace_identity(refreshed_parent) != previous_parent:
            raise CustodyError("created cleanup parent identity changed")
        _assert_ancestor_chain(ancestors)
    except OSError as error:
        raise CustodyError(f"created cleanup root could not be removed: {lexical}") from error
    finally:
        if descriptor >= 0:
            _close_quietly(descriptor)
        _close_ancestors(ancestors)


__all__ = [
    "CustodyError",
    "ExpectedFile",
    "READ_ONLY_DIRECTORY_MODE",
    "READ_ONLY_FILE_MODE",
    "RecordLike",
    "SnapshotFile",
    "TreeSnapshot",
    "copy_exact_tree",
    "expected_directories",
    "normalize_expectations",
    "remove_created_tree",
    "seal_copied_tree",
    "sealed_expectations",
    "snapshot_exact_tree",
    "verify_sealed_tree",
    "write_snapshot_exclusive",
]
