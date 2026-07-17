from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from bench.sealed_lineage_verifier import custody


def _record(path: Path) -> custody.ExpectedFile:
    payload = path.read_bytes()
    return custody.ExpectedFile(
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        stat.S_IMODE(path.stat().st_mode),
    )


def _fixture_tree(root: Path) -> dict[Path, custody.ExpectedFile]:
    (root / "nested/deep").mkdir(parents=True)
    payloads = {
        Path("root.bin"): b"root-payload-0000",
        Path("nested/item.txt"): b"nested-payload-1111",
        Path("nested/deep/data.bin"): b"deep-payload-2222",
    }
    modes = {
        Path("root.bin"): 0o644,
        Path("nested/item.txt"): 0o444,
        Path("nested/deep/data.bin"): 0o600,
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.write_bytes(payload)
        path.chmod(modes[relative])
    return {relative: _record(root / relative) for relative in sorted(payloads, key=str)}


def test_exact_snapshot_accepts_generic_pin_shapes_and_returns_bound_payloads(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    generic: dict[Path, custody.RecordLike] = {
        Path("root.bin"): (
            expected[Path("root.bin")].sha256,
            expected[Path("root.bin")].bytes,
            expected[Path("root.bin")].mode,
        ),
        Path("nested/item.txt"): {
            "sha256": expected[Path("nested/item.txt")].sha256,
            "bytes": expected[Path("nested/item.txt")].bytes,
            "mode": expected[Path("nested/item.txt")].mode,
        },
        Path("nested/deep/data.bin"): expected[Path("nested/deep/data.bin")],
    }

    snapshot = custody.snapshot_exact_tree(root, generic)

    assert snapshot.directories == (Path("nested"), Path("nested/deep"))
    assert snapshot.payloads == {
        Path("nested/deep/data.bin"): b"deep-payload-2222",
        Path("nested/item.txt"): b"nested-payload-1111",
        Path("root.bin"): b"root-payload-0000",
    }
    assert snapshot.records == expected


def test_copy_is_exclusive_durable_and_sealed_at_every_depth(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _fixture_tree(source)
    destination = tmp_path / "copy"

    custody.copy_exact_tree(source, destination, expected)
    replay = custody.verify_sealed_tree(destination, expected)

    assert replay.payloads == custody.snapshot_exact_tree(source, expected).payloads
    for directory in (destination, destination / "nested", destination / "nested/deep"):
        assert stat.S_IMODE(directory.stat().st_mode) == custody.READ_ONLY_DIRECTORY_MODE
    for relative in expected:
        assert stat.S_IMODE((destination / relative).stat().st_mode) == custody.READ_ONLY_FILE_MODE
    with pytest.raises(custody.CustodyError, match="already exists"):
        custody.copy_exact_tree(source, destination, expected)


def test_copy_uses_snapshot_bytes_and_never_reopens_mutated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    expected = _fixture_tree(source)
    destination = tmp_path / "copy"
    target = source / "root.bin"
    original_payload = target.read_bytes()
    original_write = custody.write_snapshot_exclusive

    def mutate_source_then_write(path: Path, snapshot: custody.TreeSnapshot) -> None:
        target.write_bytes(b"source-mutated-after-snapshot")
        original_write(path, snapshot)

    monkeypatch.setattr(custody, "write_snapshot_exclusive", mutate_source_then_write)
    custody.copy_exact_tree(source, destination, expected)

    assert (destination / "root.bin").read_bytes() == original_payload
    assert target.read_bytes() != original_payload


def test_existing_tree_can_be_authenticated_then_sealed(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    expected = _fixture_tree(root)
    for relative in expected:
        (root / relative).chmod(0o600)
    for directory in (root / "nested/deep", root / "nested", root):
        directory.chmod(0o700)

    custody.seal_copied_tree(root, expected)

    custody.verify_sealed_tree(root, expected)
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert all(stat.S_IMODE((root / relative).stat().st_mode) == 0o444 for relative in expected)


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(custody.CustodyError, match="root is not a real directory"):
        custody.snapshot_exact_tree(alias, expected)


def test_symlink_in_a_root_ancestor_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    root = actual / "source"
    expected = _fixture_tree(root)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(custody.CustodyError, match="ancestor is not a real directory"):
        custody.snapshot_exact_tree(alias / "source", expected)


def test_retained_ancestor_chain_rejects_rename_and_symlink_back_to_same_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = tmp_path / "outer"
    root = outer / "source"
    expected = _fixture_tree(root)
    retired = tmp_path / "retired-outer"
    real_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode_argument: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and path == "source" and flags & custody.O_DIRECTORY and dir_fd is not None:
            outer.rename(retired)
            outer.symlink_to(retired, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode_argument, dir_fd=dir_fd)

    monkeypatch.setattr(custody.os, "open", swapping_open)
    with pytest.raises(custody.CustodyError, match="ancestor .*changed"):
        custody.snapshot_exact_tree(root, expected)
    assert swapped


def test_retained_ancestor_chain_allows_unrelated_directory_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    real_open = os.open
    churned = False

    def churning_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode_argument: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal churned
        if not churned and path == "source" and flags & custody.O_DIRECTORY and dir_fd is not None:
            (tmp_path / "unrelated-sibling").mkdir()
            churned = True
        return real_open(path, flags, mode_argument, dir_fd=dir_fd)

    monkeypatch.setattr(custody.os, "open", churning_open)
    snapshot = custody.snapshot_exact_tree(root, expected)

    assert churned
    assert snapshot.records == expected


def test_intermediate_tree_symlink_is_rejected_without_traversal(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    moved = tmp_path / "moved-nested"
    (root / "nested").rename(moved)
    (root / "nested").symlink_to(moved, target_is_directory=True)

    with pytest.raises(custody.CustodyError, match="symlink"):
        custody.snapshot_exact_tree(root, expected)


def test_final_file_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    target = tmp_path / "outside.bin"
    target.write_bytes((root / "root.bin").read_bytes())
    (root / "root.bin").unlink()
    (root / "root.bin").symlink_to(target)

    with pytest.raises(custody.CustodyError, match="symlink"):
        custody.snapshot_exact_tree(root, expected)


def test_hard_link_is_rejected_even_when_bytes_and_mode_match(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    path = root / "root.bin"
    outside = tmp_path / "outside.bin"
    path.rename(outside)
    os.link(outside, path)

    with pytest.raises(custody.CustodyError, match="hard-linked"):
        custody.snapshot_exact_tree(root, expected)


def test_special_file_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    path = root / "root.bin"
    path.unlink()
    os.mkfifo(path, expected[Path("root.bin")].mode)

    with pytest.raises(custody.CustodyError, match="not a regular file"):
        custody.snapshot_exact_tree(root, expected)


Mutation = Callable[[Path, dict[Path, custody.ExpectedFile]], None]


def _add_extra(root: Path, expected: dict[Path, custody.ExpectedFile]) -> None:
    del expected
    (root / "extra.bin").write_bytes(b"extra")


def _remove_file(root: Path, expected: dict[Path, custody.ExpectedFile]) -> None:
    del expected
    (root / "root.bin").unlink()


def _change_mode(root: Path, expected: dict[Path, custody.ExpectedFile]) -> None:
    del expected
    (root / "root.bin").chmod(0o600)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_add_extra, "membership differs"),
        (_remove_file, "membership differs"),
        (_change_mode, "mode differs"),
    ],
)
def test_extra_missing_and_wrong_mode_are_rejected(
    tmp_path: Path,
    mutation: Mutation,
    message: str,
) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    mutation(root, expected)

    with pytest.raises(custody.CustodyError, match=message):
        custody.snapshot_exact_tree(root, expected)


def test_wrong_size_and_hash_pins_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    relative = Path("root.bin")
    record = expected[relative]
    wrong_size = dict(expected)
    wrong_size[relative] = custody.ExpectedFile(record.sha256, record.bytes + 1, record.mode)
    with pytest.raises(custody.CustodyError, match="size differs"):
        custody.snapshot_exact_tree(root, wrong_size)

    wrong_hash = dict(expected)
    wrong_hash[relative] = custody.ExpectedFile("00" * 32, record.bytes, record.mode)
    with pytest.raises(custody.CustodyError, match="hash differs"):
        custody.snapshot_exact_tree(root, wrong_hash)


def test_same_size_path_replacement_between_stat_and_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    target = root / "root.bin"
    retired = tmp_path / "retired.bin"
    payload = target.read_bytes()
    mode = stat.S_IMODE(target.stat().st_mode)
    real_open = os.open
    replaced = False

    def replacing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode_argument: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == "root.bin" and dir_fd is not None and not flags & os.O_CREAT:
            target.rename(retired)
            target.write_bytes(payload)
            target.chmod(mode)
            replaced = True
        return real_open(path, flags, mode_argument, dir_fd=dir_fd)

    monkeypatch.setattr(custody.os, "open", replacing_open)
    with pytest.raises(custody.CustodyError, match="changed while opened"):
        custody.snapshot_exact_tree(root, expected)
    assert replaced


def test_in_place_mutation_during_descriptor_read_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    target = root / "root.bin"
    target_inode = target.stat().st_ino
    real_read = os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, count)
        if not mutated and chunk and os.fstat(descriptor).st_ino == target_inode:
            with target.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"X")
                stream.flush()
                os.fsync(stream.fileno())
            mutated = True
        return chunk

    monkeypatch.setattr(custody.os, "read", mutating_read)
    with pytest.raises(custody.CustodyError, match="mutated|hash differs"):
        custody.snapshot_exact_tree(root, expected)
    assert mutated


def test_directory_swap_between_stat_and_open_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    original = root / "nested"
    retired = tmp_path / "retired-nested"
    real_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode_argument: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and path == "nested" and flags & custody.O_DIRECTORY and dir_fd is not None:
            original.rename(retired)
            shutil.copytree(retired, original)
            swapped = True
        return real_open(path, flags, mode_argument, dir_fd=dir_fd)

    monkeypatch.setattr(custody.os, "open", swapping_open)
    with pytest.raises(custody.CustodyError, match="directory changed while opened"):
        custody.snapshot_exact_tree(root, expected)
    assert swapped


def test_invalid_relative_paths_and_file_directory_conflicts_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    record = next(iter(expected.values()))

    for invalid in (Path("../escape"), Path("/absolute"), Path(".")):
        with pytest.raises(custody.CustodyError, match="normalized and relative"):
            custody.snapshot_exact_tree(root, {invalid: record})
    with pytest.raises(custody.CustodyError, match="also a directory ancestor"):
        custody.snapshot_exact_tree(root, {Path("nested"): record, Path("nested/item.txt"): record})


def test_sealed_verifier_rejects_file_and_directory_mode_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _fixture_tree(source)
    destination = tmp_path / "copy"
    custody.copy_exact_tree(source, destination, expected)

    destination.chmod(0o755)
    with pytest.raises(custody.CustodyError, match="directory mode differs"):
        custody.verify_sealed_tree(destination, expected)
    destination.chmod(0o555)
    (destination / "root.bin").chmod(0o644)
    with pytest.raises(custody.CustodyError, match="mode differs"):
        custody.verify_sealed_tree(destination, expected)


def test_exact_heterogeneous_directory_mode_map_is_enforced(tmp_path: Path) -> None:
    root = tmp_path / "source"
    expected = _fixture_tree(root)
    root.chmod(0o755)
    (root / "nested").chmod(0o555)
    (root / "nested/deep").chmod(0o700)
    modes = {Path("."): 0o755, Path("nested"): 0o555, Path("nested/deep"): 0o700}

    custody.snapshot_exact_tree(root, expected, directory_modes=modes)
    wrong = {**modes, Path("nested/deep"): 0o755}
    with pytest.raises(custody.CustodyError, match="directory mode differs"):
        custody.snapshot_exact_tree(root, expected, directory_modes=wrong)
    with pytest.raises(custody.CustodyError, match="exact directory closure"):
        custody.snapshot_exact_tree(root, expected, directory_modes={Path("."): 0o755})


def test_descriptor_cleanup_removes_sealed_created_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _fixture_tree(source)
    staging = tmp_path / ".staging"
    custody.copy_exact_tree(source, staging, expected)

    custody.remove_created_tree(staging)

    assert not staging.exists()
