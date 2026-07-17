from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path

import pytest

from tools.materialize_artifact import (
    ArtifactError,
    _canonical_destination,
    materialize,
    summarize_tree,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    source = tmp_path / "source" / "MM-TEST"
    runtime = source / "runtime"
    runtime.mkdir(parents=True)
    (source / "record.json").write_text('{"status": "terminal"}\n', encoding="utf-8")
    (runtime / "dependency.bin").write_bytes(b"\x00numpy\xff")
    os.chmod(source / "record.json", 0o444)
    os.chmod(runtime / "dependency.bin", 0o444)
    os.chmod(runtime, 0o555)
    os.chmod(source, 0o555)

    archive = tmp_path / "MM-TEST.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="MM-TEST")

    summary = summarize_tree(source)
    pointer = repository / "MM-TEST.artifact-pointer.json"
    pointer.write_text(
        json.dumps(
            {
                "archive": {
                    "bytes": archive.stat().st_size,
                    "directory_count": summary.directory_count,
                    "file_count": summary.file_count,
                    "payload_bytes": summary.payload_bytes,
                    "root": "MM-TEST",
                    "sha256": _sha256(archive),
                    "tree_digest": {
                        "algorithm": "sha256",
                        "schema": "prospect-tree-digest-v1",
                        "value": summary.digest,
                    },
                },
                "canonical_materialization_path": "results/MM-TEST",
                "release": {
                    "asset": archive.name,
                    "url": "https://example.invalid/MM-TEST.tar.gz",
                },
                "schema_version": "prospect-external-artifact-pointer-v1",
            }
        ),
        encoding="utf-8",
    )

    projection = repository / "results" / "MM-TEST"
    projection.mkdir(parents=True)
    (projection / "record.json").write_text('{"status": "terminal"}\n', encoding="utf-8")
    return pointer, archive, projection


def test_materialize_replaces_only_a_byte_compatible_projection(tmp_path: Path) -> None:
    pointer, archive, projection = _fixture(tmp_path)

    summary = materialize(pointer, archive, projection)

    assert summary == summarize_tree(projection)
    assert (projection / "runtime" / "dependency.bin").read_bytes() == b"\x00numpy\xff"
    assert stat.S_IMODE(projection.stat().st_mode) == 0o555
    assert stat.S_IMODE((projection / "record.json").stat().st_mode) == 0o444


def test_materialize_rejects_a_modified_projection(tmp_path: Path) -> None:
    pointer, archive, projection = _fixture(tmp_path)
    (projection / "record.json").write_text('{"status": "changed"}\n', encoding="utf-8")

    with pytest.raises(ArtifactError, match="byte-compatible"):
        materialize(pointer, archive, projection)
    assert (projection / "record.json").read_text(encoding="utf-8") == '{"status": "changed"}\n'


def test_materialize_fails_closed_on_an_unreadable_projection_subtree(tmp_path: Path) -> None:
    pointer, archive, projection = _fixture(tmp_path)
    runtime = projection / "runtime"
    runtime.mkdir()
    dependency = runtime / "dependency.bin"
    dependency.write_bytes(b"changed")
    os.chmod(runtime, 0)
    try:
        with pytest.raises(ArtifactError, match="cannot traverse"):
            materialize(pointer, archive, projection)
    finally:
        os.chmod(runtime, 0o755)


def test_pointer_destination_must_stay_beneath_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    with pytest.raises(ArtifactError, match="repository-relative"):
        _canonical_destination({"canonical_materialization_path": "../outside"}, repository)


def test_materialize_rejects_a_broken_destination_symlink(tmp_path: Path) -> None:
    pointer, archive, _ = _fixture(tmp_path)
    destination = tmp_path / "broken"
    destination.symlink_to(tmp_path / "missing")

    with pytest.raises(ArtifactError, match="symbolic link"):
        materialize(pointer, archive, destination)
    assert destination.is_symlink()
