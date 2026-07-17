#!/usr/bin/env python3
"""Download, verify, and safely materialize an external research artifact."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


class ArtifactError(RuntimeError):
    """Raised when an artifact or pointer fails closed."""


@dataclass(frozen=True)
class TreeSummary:
    digest: str
    file_count: int
    directory_count: int
    payload_bytes: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_paths(root: Path) -> list[tuple[bytes, str, Path]]:
    def fail_closed(error: OSError) -> None:
        raise ArtifactError(f"cannot traverse artifact tree: {error}") from error

    paths: list[tuple[bytes, str, Path]] = [(b".", ".", root)]
    for current, directories, files in os.walk(root, followlinks=False, onerror=fail_closed):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            relative = path.relative_to(root)
            display = f"./{relative.as_posix()}"
            paths.append((os.fsencode(display), display, path))
    return sorted(paths, key=lambda item: item[0])


def summarize_tree(root: Path) -> TreeSummary:
    """Hash path, type, mode, size, and content using prospect-tree-digest-v1."""
    if not root.is_dir():
        raise ArtifactError(f"artifact root is not a directory: {root}")

    tree_digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    payload_bytes = 0

    for _, display, path in _tree_paths(root):
        metadata = path.lstat()
        mode = f"{stat.S_IMODE(metadata.st_mode):o}"
        if stat.S_ISLNK(metadata.st_mode):
            fields = ("L", mode, "-", os.readlink(path), display)
        elif stat.S_ISREG(metadata.st_mode):
            size = metadata.st_size
            fields = ("F", mode, str(size), _sha256_file(path), display)
            file_count += 1
            payload_bytes += size
        elif stat.S_ISDIR(metadata.st_mode):
            fields = ("D", mode, "-", "-", display)
            directory_count += 1
        else:
            raise ArtifactError(f"unsupported filesystem entry in artifact: {path}")

        for field in fields:
            tree_digest.update(os.fsencode(field))
            tree_digest.update(b"\0")

    return TreeSummary(
        digest=tree_digest.hexdigest(),
        file_count=file_count,
        directory_count=directory_count,
        payload_bytes=payload_bytes,
    )


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"pointer field {name!r} must be an object")
    return value


def _require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"pointer field {key!r} must be a non-empty string")
    return value


def _require_integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ArtifactError(f"pointer field {key!r} must be a non-negative integer")
    return value


def _load_pointer(path: Path) -> dict[str, Any]:
    try:
        pointer = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot read artifact pointer {path}: {error}") from error
    pointer = _require_mapping(pointer, "root")
    if pointer.get("schema_version") != "prospect-external-artifact-pointer-v1":
        raise ArtifactError("unsupported external artifact pointer schema")
    return pointer


def _find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ArtifactError(f"cannot locate repository root from {start}")


def _copy_download(url: str, destination: Path, expected_bytes: int) -> None:
    if urlparse(url).scheme != "https":
        raise ArtifactError("artifact release URL must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "prospect-artifact-materializer/1"})
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    header_bytes = int(content_length)
                except ValueError as error:
                    raise ArtifactError(f"invalid download length header: {content_length!r}") from error
                if header_bytes != expected_bytes:
                    raise ArtifactError(
                        f"download length header mismatch: expected {expected_bytes}, observed {content_length}"
                    )
            observed_bytes = 0
            while block := response.read(min(1024 * 1024, expected_bytes + 1 - observed_bytes)):
                observed_bytes += len(block)
                if observed_bytes > expected_bytes:
                    raise ArtifactError(f"download exceeds pinned length of {expected_bytes} bytes")
                output.write(block)
            if observed_bytes != expected_bytes:
                raise ArtifactError(
                    f"download length mismatch: expected {expected_bytes}, observed {observed_bytes}"
                )
    except OSError as error:
        raise ArtifactError(f"artifact download failed: {error}") from error


def _validated_members(archive: tarfile.TarFile, expected_root: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    seen: set[PurePosixPath] = set()
    for member in members:
        member_path = PurePosixPath(member.name)
        if (
            not member.name
            or member_path.is_absolute()
            or ".." in member_path.parts
            or not member_path.parts
            or member_path.parts[0] != expected_root
        ):
            raise ArtifactError(f"unsafe or unexpected archive path: {member.name!r}")
        if member_path in seen:
            raise ArtifactError(f"duplicate archive path: {member.name!r}")
        if not (member.isdir() or member.isfile()):
            raise ArtifactError(f"unsupported archive member type: {member.name!r}")
        if member.mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ArtifactError(f"privileged archive mode is not allowed: {member.name!r}")
        seen.add(member_path)
    if PurePosixPath(expected_root) not in seen:
        raise ArtifactError(f"archive does not contain its declared root: {expected_root!r}")
    return members


def _canonical_destination(pointer: dict[str, Any], repository_root: Path) -> Path:
    declared = Path(_require_string(pointer, "canonical_materialization_path"))
    if declared.is_absolute() or ".." in declared.parts or declared == Path("."):
        raise ArtifactError("canonical materialization path must be a repository-relative child")
    root = repository_root.resolve()
    current = root
    for part in declared.parts:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ArtifactError(f"canonical materialization path crosses a symbolic link: {current}")
    destination = (root / declared).resolve()
    try:
        relative = destination.relative_to(root)
    except ValueError as error:
        raise ArtifactError("canonical materialization path escapes the repository") from error
    if not relative.parts or relative.parts[0] == ".git":
        raise ArtifactError("canonical materialization path targets repository metadata")
    return destination


def _archive_root(archive_spec: dict[str, Any]) -> str:
    expected_root = _require_string(archive_spec, "root")
    root_path = PurePosixPath(expected_root)
    if root_path.is_absolute() or len(root_path.parts) != 1 or expected_root in {".", ".."}:
        raise ArtifactError("archive root must be one safe top-level name")
    return expected_root


def _release_asset(release: dict[str, Any]) -> str:
    asset = _require_string(release, "asset")
    if PurePosixPath(asset).name != asset or asset in {".", ".."}:
        raise ArtifactError("release asset must be one safe filename")
    return asset


def _assert_summary(summary: TreeSummary, archive_spec: dict[str, Any]) -> None:
    tree_digest = _require_mapping(archive_spec.get("tree_digest"), "archive.tree_digest")
    if tree_digest.get("algorithm") != "sha256":
        raise ArtifactError("unsupported artifact tree-digest algorithm")
    if tree_digest.get("schema") != "prospect-tree-digest-v1":
        raise ArtifactError("unsupported artifact tree-digest schema")
    expected = TreeSummary(
        digest=_require_string(tree_digest, "value"),
        file_count=_require_integer(archive_spec, "file_count"),
        directory_count=_require_integer(archive_spec, "directory_count"),
        payload_bytes=_require_integer(archive_spec, "payload_bytes"),
    )
    if summary != expected:
        raise ArtifactError(f"artifact tree verification failed: expected {expected}, observed {summary}")


def _projection_matches(projection: Path, complete: Path) -> bool:
    """Return true when every projected path is a same-type, same-byte archive member."""
    for _, _, projected_path in _tree_paths(projection):
        relative = projected_path.relative_to(projection)
        complete_path = complete / relative
        projected_metadata = projected_path.lstat()
        try:
            complete_metadata = complete_path.lstat()
        except FileNotFoundError:
            return False
        if stat.S_IFMT(projected_metadata.st_mode) != stat.S_IFMT(complete_metadata.st_mode):
            return False
        if stat.S_ISREG(projected_metadata.st_mode) and _sha256_file(projected_path) != _sha256_file(complete_path):
            return False
        if stat.S_ISLNK(projected_metadata.st_mode) and os.readlink(projected_path) != os.readlink(complete_path):
            return False
    return True


def _rmtree_writable(path: Path) -> None:
    for current, _, _ in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        os.chmod(current_path, stat.S_IMODE(current_path.stat().st_mode) | stat.S_IRWXU)
    shutil.rmtree(path)


def _install_verified_tree(extracted: Path, destination: Path) -> None:
    extracted_mode = stat.S_IMODE(extracted.stat().st_mode)
    os.chmod(extracted, extracted_mode | stat.S_IWUSR)
    try:
        extracted.rename(destination)
        os.chmod(destination, extracted_mode)
    except BaseException:
        if os.path.lexists(destination):
            os.chmod(destination, extracted_mode | stat.S_IWUSR)
            destination.rename(extracted)
            os.chmod(extracted, extracted_mode)
        raise


def materialize(pointer_path: Path, archive_path: Path, destination: Path) -> TreeSummary:
    pointer = _load_pointer(pointer_path)
    archive_spec = _require_mapping(pointer.get("archive"), "archive")
    expected_archive_bytes = _require_integer(archive_spec, "bytes")
    expected_archive_sha = _require_string(archive_spec, "sha256")
    expected_root = _archive_root(archive_spec)

    observed_bytes = archive_path.stat().st_size
    if observed_bytes != expected_archive_bytes:
        raise ArtifactError(
            f"archive length mismatch: expected {expected_archive_bytes}, observed {observed_bytes}"
        )
    observed_sha = _sha256_file(archive_path)
    if observed_sha != expected_archive_sha:
        raise ArtifactError(f"archive SHA-256 mismatch: expected {expected_archive_sha}, observed {observed_sha}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if destination.is_symlink():
            raise ArtifactError(f"existing destination is a symbolic link: {destination}")
        if not destination.is_dir():
            raise ArtifactError(f"existing destination is not a directory: {destination}")

    with tempfile.TemporaryDirectory(prefix=f".{expected_root}-materialize-", dir=destination.parent) as temporary:
        temporary_path = Path(temporary)
        try:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                members = _validated_members(archive, expected_root)
                if "filter" in inspect.signature(archive.extractall).parameters:
                    archive.extractall(temporary_path, members=members, filter="fully_trusted")
                else:  # pragma: no cover - compatibility with Python 3.11.0-3.11.3
                    archive.extractall(temporary_path, members=members)
        except (OSError, tarfile.TarError) as error:
            raise ArtifactError(f"cannot extract verified archive: {error}") from error

        extracted = temporary_path / expected_root
        summary = summarize_tree(extracted)
        _assert_summary(summary, archive_spec)

        if os.path.lexists(destination):
            backup = destination.parent / f"{temporary_path.name}-repository-projection"
            if os.path.lexists(backup):
                raise ArtifactError(f"temporary projection backup already exists: {backup}")
            destination.rename(backup)
            installed = False
            try:
                current_summary = summarize_tree(backup)
                if current_summary == summary:
                    backup.rename(destination)
                    return summary
                if not _projection_matches(backup, extracted):
                    raise ArtifactError(
                        f"existing destination is not a byte-compatible repository projection: {destination}"
                    )
                _install_verified_tree(extracted, destination)
                installed = True
            except BaseException:
                if not installed and not os.path.lexists(destination) and os.path.lexists(backup):
                    backup.rename(destination)
                raise
            _rmtree_writable(backup)
        else:
            _install_verified_tree(extracted, destination)

    return summary


def _parse_args(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pointer", type=Path, help="path to a prospect external-artifact pointer")
    parser.add_argument("--archive", type=Path, help="use this local archive instead of downloading the release asset")
    parser.add_argument("--destination", type=Path, help="override the pointer's repository-relative destination")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    options = _parse_args(arguments)
    pointer_path = options.pointer.resolve()
    try:
        pointer = _load_pointer(pointer_path)
        repository_root = _find_repository_root(pointer_path.parent)
        destination = (
            Path(os.path.abspath(options.destination))
            if options.destination is not None
            else _canonical_destination(pointer, repository_root)
        )
        if options.archive is not None:
            summary = materialize(pointer_path, options.archive.resolve(), destination)
        else:
            release = _require_mapping(pointer.get("release"), "release")
            archive_spec = _require_mapping(pointer.get("archive"), "archive")
            with tempfile.TemporaryDirectory(prefix="prospect-artifact-download-") as temporary:
                archive_path = Path(temporary) / _release_asset(release)
                _copy_download(
                    _require_string(release, "url"),
                    archive_path,
                    _require_integer(archive_spec, "bytes"),
                )
                summary = materialize(pointer_path, archive_path, destination)
    except (ArtifactError, OSError) as error:
        print(f"materialization failed: {error}", file=sys.stderr)
        return 2

    print(f"materialized: {destination}")
    print(f"tree sha256: {summary.digest}")
    print(
        f"files: {summary.file_count}; directories: {summary.directory_count}; "
        f"payload bytes: {summary.payload_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
