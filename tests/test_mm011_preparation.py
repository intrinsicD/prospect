from __future__ import annotations

import hashlib
import math
import os
import stat
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from bench.multimodal_causal_assay import preparation, records

TEST_PROTOCOL_SHA256 = "ab" * 32


@pytest.fixture(scope="module")
def raw_identities() -> tuple[np.ndarray, np.ndarray]:
    return preparation.expected_raw_identities()


@pytest.fixture(scope="module")
def rows(raw_identities: tuple[np.ndarray, np.ndarray]) -> tuple[preparation.RowIndex, ...]:
    return preparation.build_row_index(*raw_identities)


@pytest.fixture(scope="module")
def synthetic_frames() -> np.ndarray:
    # Generated, deterministic, broadband-enough values; no repository data is read.
    linear = np.arange(
        preparation.RAW_ROWS * preparation.NATIVE_SIZE * preparation.NATIVE_SIZE * preparation.CHANNELS,
        dtype=np.uint32,
    )
    return np.ascontiguousarray(
        (linear.reshape(preparation.RAW_ROWS, 64, 64, 3) * 17 + 29) % 251,
        dtype=np.uint8,
    )


@pytest.fixture(scope="module")
def normalizers(
    synthetic_frames: np.ndarray, rows: tuple[preparation.RowIndex, ...]
) -> tuple[preparation.FoldNormalizer, ...]:
    return preparation.fit_fold_normalizers(synthetic_frames, rows)


def test_protocol_binding_is_frozen_and_explicit() -> None:
    assert records.PROTOCOL_SHA256 == (
        "ca39f7cea6a2a5b041956b419bf3530dd54eb8403096963a044d7fcf1e2121cc"
    )
    assert preparation.PROTOCOL_SHA256 == records.PROTOCOL_SHA256
    with pytest.raises(records.RecordValidationError, match="frozen protocol"):
        records.scientific_array_sha256("row", np.zeros(1, dtype="<f8"), protocol_sha256="unfrozen")


def test_canonical_json_and_array_hashes_are_domain_bound_and_mutation_sensitive() -> None:
    value: records.JsonValue = {"ascii": "yes", "nested": [1, 2.5, None], "unicode": "\u03b1"}
    assert records.canonical_json_bytes(value) == (b'{"ascii":"yes","nested":[1,2.5,null],"unicode":"\\u03b1"}')
    first = records.canonical_json_sha256(value, protocol_sha256=TEST_PROTOCOL_SHA256)
    second = records.canonical_json_sha256(value, protocol_sha256="cd" * 32)
    assert first != second

    array = np.arange(12, dtype="<f8").reshape(3, 4)
    replay = records.scientific_array_sha256("source:current", array, protocol_sha256=TEST_PROTOCOL_SHA256)
    assert replay == records.scientific_array_sha256(
        "source:current", array.copy(), protocol_sha256=TEST_PROTOCOL_SHA256
    )
    mutated = array.copy()
    mutated[0, 0] += 1.0
    assert replay != records.scientific_array_sha256("source:current", mutated, protocol_sha256=TEST_PROTOCOL_SHA256)
    assert replay != records.scientific_array_sha256("target:current", array, protocol_sha256=TEST_PROTOCOL_SHA256)
    with pytest.raises(records.RecordValidationError, match="nonfinite"):
        records.scientific_array_sha256("bad", np.asarray([np.nan], dtype="<f8"), protocol_sha256=TEST_PROTOCOL_SHA256)
    with pytest.raises(records.RecordValidationError, match="nonfinite"):
        records.canonical_json_bytes({"bad": math.inf})  # type: ignore[arg-type]


def test_parent_pins_are_exact_and_opaque_tree_validation_is_generic(tmp_path: Path) -> None:
    assert set(records.PARENT_PINS) == {
        "artifact-manifest.json",
        "input-manifest.json",
        "formal-start.json",
        "MM-007-evidence.json",
        "MM-007-results.json",
        "MM-007-report.md",
        "MM-007-protocol.md",
        "MM-007-frames-64x64.npz",
    }
    assert records.PARENT_PINS["formal-start.json"].mode == 0o444
    assert records.PARENT_PINS["MM-007-frames-64x64.npz"].sha256 == (
        "fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f"
    )
    assert records.PARENT_EXCLUDED_NONAUTHORITATIVE_DIRECTORIES == ("inputs",)

    root = tmp_path / "mock-parent"
    root.mkdir()
    payload = b"opaque synthetic fixture"
    records.write_immutable_bytes_exclusive(root / "one.bin", payload)
    pin = records.ParentPin(hashlib.sha256(payload).hexdigest(), 0o444)
    validated = records.validate_pinned_parent_tree(root, {"one.bin": pin})
    assert validated["one.bin"]["sha256"] == pin.sha256

    records.write_immutable_bytes_exclusive(root / "extra.bin", b"extra")
    with pytest.raises(records.RecordValidationError, match="membership"):
        records.validate_pinned_parent_tree(root, {"one.bin": pin})


def test_parent_lineage_directory_is_required_but_never_traversed_or_returned(tmp_path: Path) -> None:
    root = tmp_path / "mock-parent"
    root.mkdir()
    payload = b"opaque synthetic fixture"
    records.write_immutable_bytes_exclusive(root / "one.bin", payload)
    pin = records.ParentPin(hashlib.sha256(payload).hexdigest(), 0o444)
    with pytest.raises(records.RecordValidationError, match="membership"):
        records.validate_pinned_parent_tree(
            root,
            {"one.bin": pin},
            excluded_nonauthoritative_directories=("inputs",),
        )
    lineage = root / "inputs"
    lineage.mkdir()
    # A dangling symlink below the exclusion proves validation does not recurse into
    # or try to hash its contents.
    (lineage / "must-not-be-opened").symlink_to(tmp_path / "absent")

    validated = records.validate_pinned_parent_tree(
        root,
        {"one.bin": pin},
        excluded_nonauthoritative_directories=("inputs",),
    )
    assert set(validated) == {"one.bin"}

    (root / "unexpected").mkdir()
    with pytest.raises(records.RecordValidationError, match="membership"):
        records.validate_pinned_parent_tree(
            root,
            {"one.bin": pin},
            excluded_nonauthoritative_directories=("inputs",),
        )


def test_parent_lineage_exclusion_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "mock-parent"
    root.mkdir()
    payload = b"opaque synthetic fixture"
    records.write_immutable_bytes_exclusive(root / "one.bin", payload)
    pin = records.ParentPin(hashlib.sha256(payload).hexdigest(), 0o444)
    lineage_target = tmp_path / "lineage-target"
    lineage_target.mkdir()
    (root / "inputs").symlink_to(lineage_target, target_is_directory=True)

    with pytest.raises(records.RecordValidationError, match="non-symlink directory"):
        records.validate_pinned_parent_tree(
            root,
            {"one.bin": pin},
            excluded_nonauthoritative_directories=("inputs",),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_entry", "membership changed during validation"),
        ("replace_exclusion", "directory identity changed during validation"),
    ],
)
def test_parent_post_hash_census_rejects_deterministic_top_level_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / "mock-parent"
    root.mkdir()
    payload = b"opaque synthetic fixture"
    records.write_immutable_bytes_exclusive(root / "one.bin", payload)
    pin = records.ParentPin(hashlib.sha256(payload).hexdigest(), 0o444)
    (root / "inputs").mkdir()
    original_file_record = records.file_record
    mutated = False

    def file_record_with_top_level_mutation(path: str | Path) -> dict[str, records.JsonValue]:
        nonlocal mutated
        record = original_file_record(path)
        if not mutated:
            mutated = True
            if mutation == "extra_entry":
                (root / "unexpected.bin").write_bytes(b"injected after opaque hash")
            else:
                (root / "inputs").rename(root / "retired-inputs")
                (root / "inputs").mkdir()
                (root / "retired-inputs").rmdir()
        return record

    monkeypatch.setattr(records, "file_record", file_record_with_top_level_mutation)
    with pytest.raises(records.RecordValidationError, match=message):
        records.validate_pinned_parent_tree(
            root,
            {"one.bin": pin},
            excluded_nonauthoritative_directories=("inputs",),
        )


def test_raw_identity_validation_and_row_index_are_exact(
    raw_identities: tuple[np.ndarray, np.ndarray], rows: tuple[preparation.RowIndex, ...]
) -> None:
    ids, times = raw_identities
    preparation.validate_raw_identities(ids, times)
    assert len(rows) == preparation.MATCHED_ROWS == 453
    assert Counter(row.video_id for row in rows) == Counter(preparation.MATCHED_COUNTS)
    assert preparation.row_identity_sha256(rows) == preparation.MATCHED_IDENTITY_SHA256
    assert rows[0].previous_timestamp == 1.0
    assert rows[0].current_timestamp == 1.5
    assert rows[0].future_timestamp == 2.0
    assert rows[-1].video_id == preparation.VIDEO_IDS[-1]
    assert rows[-1].video_row == preparation.MATCHED_COUNTS[preparation.VIDEO_IDS[-1]] - 1
    assert all(row.fold_index == preparation.VIDEO_IDS.index(row.video_id) // 2 for row in rows)

    changed_times = times.copy()
    changed_times[10] += 0.25
    with pytest.raises(preparation.PreparationValidationError, match="identities"):
        preparation.build_row_index(ids, changed_times)
    with pytest.raises(preparation.PreparationValidationError, match="schema"):
        preparation.validate_raw_identities(ids, times.astype("<f4"))
    changed_ids = ids.copy()
    changed_ids[0] = "video_bad"
    with pytest.raises(preparation.PreparationValidationError, match="identities"):
        preparation.validate_raw_identities(changed_ids, times)


def test_half_cycle_is_ceil_shift_bijection_with_exact_inverse(rows: tuple[preparation.RowIndex, ...]) -> None:
    mapping = preparation.half_cycle_derangement(rows)
    replay = preparation.half_cycle_derangement(rows)
    np.testing.assert_array_equal(mapping, replay)
    assert not mapping.flags.writeable
    assert sorted(mapping.tolist()) == list(range(453))
    assert np.all(mapping != np.arange(453))
    for video_id in preparation.VIDEO_IDS:
        ordinals = [row.ordinal for row in rows if row.video_id == video_id]
        shift = (len(ordinals) + 1) // 2
        assert [int(mapping[value]) for value in ordinals] == [
            ordinals[(index + shift) % len(ordinals)] for index in range(len(ordinals))
        ]
    inverse = preparation.inverse_derangement(mapping, rows)
    np.testing.assert_array_equal(mapping[inverse], np.arange(453))

    broken = mapping.copy()
    broken[0] = 0
    with pytest.raises(preparation.PreparationValidationError, match="fixed-point"):
        preparation.validate_derangement(broken, rows)

    # A different within-video fixed-point-free permutation is still noncanonical.
    alternate = mapping.copy()
    first_video = [row.ordinal for row in rows if row.video_id == preparation.VIDEO_IDS[0]]
    alternate[first_video] = np.roll(np.asarray(first_video, dtype="<i8"), 1)
    with pytest.raises(preparation.PreparationValidationError, match="ceil-shift"):
        preparation.validate_derangement(alternate, rows)


def test_normalizers_use_only_training_video_r8_current_frames(
    synthetic_frames: np.ndarray, rows: tuple[preparation.RowIndex, ...]
) -> None:
    original = preparation.fit_fold_normalizers(synthetic_frames, rows)
    assert len(original) == 4
    assert all(not value.mean.flags.writeable and not value.scale.flags.writeable for value in original)
    assert all(np.all(value.scale >= preparation.SCALE_FLOOR) for value in original)

    ids, _ = preparation.expected_raw_identities()
    fold = preparation.FOLDS[0]
    held_out_mutation = synthetic_frames.copy()
    held_out_mutation[np.isin(ids, fold.test_ids)] = 255 - held_out_mutation[np.isin(ids, fold.test_ids)]
    held_out_refit = preparation.fit_fold_normalizers(held_out_mutation, rows)[0]
    np.testing.assert_array_equal(original[0].mean, held_out_refit.mean)
    np.testing.assert_array_equal(original[0].scale, held_out_refit.scale)
    assert original[0].fingerprint == held_out_refit.fingerprint

    # Frames outside q=1..N-3 are never a normalizer current in any fold.
    noncurrent_mutation = synthetic_frames.copy()
    offset = 0
    for video_id in preparation.VIDEO_IDS:
        count = preparation.RAW_COUNTS[video_id]
        for local in (0, count - 2, count - 1):
            noncurrent_mutation[offset + local] = 255 - noncurrent_mutation[offset + local]
        offset += count
    noncurrent_refit = preparation.fit_fold_normalizers(noncurrent_mutation, rows)
    for left, right in zip(original, noncurrent_refit, strict=True):
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.scale, right.scale)

    training_mutation = synthetic_frames.copy()
    training_row = next(row for row in rows if row.video_id in fold.train_ids)
    training_mutation[training_row.current_index] = 0
    changed = preparation.fit_fold_normalizers(training_mutation, rows)[0]
    assert changed.fingerprint != original[0].fingerprint

    constant = np.zeros_like(synthetic_frames)
    floored = preparation.fit_fold_normalizers(constant, rows)
    for normalizer in floored:
        np.testing.assert_array_equal(normalizer.scale, np.full(3, preparation.SCALE_FLOOR))


def test_parent_frame_schema_accepts_synthetic_only_when_pins_are_disabled(
    raw_identities: tuple[np.ndarray, np.ndarray], synthetic_frames: np.ndarray
) -> None:
    arrays = {
        "video_ids": raw_identities[0],
        "timestamps": raw_identities[1],
        "frames_uint8": synthetic_frames,
    }
    preparation.validate_parent_frame_arrays(arrays, require_pins=False)
    with pytest.raises(preparation.PreparationValidationError, match="pins"):
        preparation.validate_parent_frame_arrays(arrays, require_pins=True)


def test_source_and_target_rows_are_detached_and_future_mutations_are_one_way(
    synthetic_frames: np.ndarray,
    rows: tuple[preparation.RowIndex, ...],
    normalizers: tuple[preparation.FoldNormalizer, ...],
) -> None:
    derangement = preparation.half_cycle_derangement(rows)
    source = preparation.construct_source_row(synthetic_frames, 0, rows, normalizers, derangement)
    target = preparation.construct_target_row(synthetic_frames, 0, rows, normalizers, derangement)
    preparation.validate_detached_pair(source, target)

    assert set(source) == set(preparation.SOURCE_ROW_SCHEMA)
    assert set(target) == set(preparation.TARGET_ROW_SCHEMA)
    assert all("future" not in name and "target" not in name for name in source)
    assert "normalizer_mean" not in target and "previous" not in target and "current" not in target
    assert all(not array.flags.writeable for array in (*source.values(), *target.values()))
    assert not any(np.shares_memory(left, right) for left in source.values() for right in target.values())

    source_bytes = preparation.source_row_npz_bytes(source)
    target_bytes = preparation.target_row_npz_bytes(target)
    assert source_bytes == preparation.source_row_npz_bytes(source)
    assert target_bytes == preparation.target_row_npz_bytes(target)

    mutated_target = {name: value.copy() for name, value in target.items()}
    mutated_target["future"][0, 0, 0] += 0.125
    checked_mutation = preparation.validate_target_row_arrays(mutated_target)
    assert preparation.source_row_npz_bytes(source) == source_bytes
    assert preparation.target_row_npz_bytes(checked_mutation) != target_bytes

    source_manifest = preparation.source_row_manifest(source, protocol_sha256=TEST_PROTOCOL_SHA256)
    target_manifest = preparation.target_row_manifest(target, protocol_sha256=TEST_PROTOCOL_SHA256)
    assert source_manifest["protocol_sha256"] == TEST_PROTOCOL_SHA256
    assert target_manifest["protocol_sha256"] == TEST_PROTOCOL_SHA256
    assert not set(source_manifest["arrays"]).intersection({"future", "deranged_future"})  # type: ignore[arg-type]


def test_row_npz_schema_roundtrip_and_cross_identity_fail_closed(
    tmp_path: Path,
    synthetic_frames: np.ndarray,
    rows: tuple[preparation.RowIndex, ...],
    normalizers: tuple[preparation.FoldNormalizer, ...],
) -> None:
    derangement = preparation.half_cycle_derangement(rows)
    source = preparation.construct_source_row(synthetic_frames, 1, rows, normalizers, derangement)
    target = preparation.construct_target_row(synthetic_frames, 1, rows, normalizers, derangement)
    source_path = tmp_path / "source" / "000001.npz"
    target_path = tmp_path / "target" / "000001.npz"
    preparation.write_source_row_npz(source_path, source)
    preparation.write_target_row_npz(target_path, target)
    assert stat.S_IMODE(source_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(target_path.stat().st_mode) == 0o444
    loaded_source = preparation.load_source_row_npz(source_path)
    loaded_target = preparation.load_target_row_npz(target_path)
    assert preparation.source_row_npz_bytes(loaded_source) == preparation.source_row_npz_bytes(source)
    assert preparation.target_row_npz_bytes(loaded_target) == preparation.target_row_npz_bytes(target)

    with zipfile.ZipFile(source_path) as archive:
        assert archive.namelist() == [f"{name}.npy" for name in sorted(preparation.SOURCE_ROW_SCHEMA)]
    with pytest.raises(FileExistsError):
        preparation.write_source_row_npz(source_path, source)

    missing = dict(source)
    del missing["current"]
    with pytest.raises(records.RecordValidationError, match="membership"):
        preparation.validate_source_row_arrays(missing)
    wrong_dtype = {name: value.copy() for name, value in source.items()}
    wrong_dtype["current"] = wrong_dtype["current"].astype("<f4")
    with pytest.raises(records.RecordValidationError, match="dtype"):
        preparation.validate_source_row_arrays(wrong_dtype)
    nonfinite = {name: value.copy() for name, value in target.items()}
    nonfinite["future"][0, 0, 0] = np.nan
    with pytest.raises(records.RecordValidationError, match="nonfinite"):
        preparation.validate_target_row_arrays(nonfinite)

    other_target = preparation.construct_target_row(synthetic_frames, 2, rows, normalizers, derangement)
    with pytest.raises(preparation.PreparationValidationError, match="identities"):
        preparation.validate_detached_pair(source, other_target)


def test_exclusive_immutable_helpers_reject_duplicates_and_symlink_components(tmp_path: Path) -> None:
    path = tmp_path / "sealed" / "record.json"
    record = records.write_immutable_json_exclusive(path, {"status": "synthetic"})
    assert record["mode"] == 0o444
    assert path.read_bytes() == b'{"status":"synthetic"}\n'
    with pytest.raises(FileExistsError):
        records.write_immutable_bytes_exclusive(path, b"replacement")

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(records.RecordValidationError, match="directory"):
        records.write_immutable_bytes_exclusive(alias / "forbidden.bin", b"no")


def test_stable_reader_rejects_same_size_path_replacement_between_stat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    replacement = tmp_path / "replacement.bin"
    source.write_bytes(b"A" * 4096)
    replacement.write_bytes(b"B" * 4096)
    original_regular_stat = records._regular_stat  # type: ignore[attr-defined]
    replaced = False

    def replace_after_stat(path: Path) -> os.stat_result:
        nonlocal replaced
        metadata = original_regular_stat(path)
        if path == source and not replaced:
            replaced = True
            os.replace(replacement, source)
        return metadata

    monkeypatch.setattr(records, "_regular_stat", replace_after_stat)
    with pytest.raises(records.RecordValidationError, match="opened path"):
        records.read_regular_bytes(source, maximum_bytes=4096)
    assert replaced
    assert source.read_bytes() == b"B" * 4096


@pytest.mark.parametrize("operation", ("read", "digest", "copy"))
def test_stable_file_consumers_reject_same_size_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    source = tmp_path / f"{operation}.bin"
    destination = tmp_path / f"{operation}-copy.bin"
    source.write_bytes(b"A" * 4096)
    before = source.stat()
    original_read = os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            with source.open("r+b") as handle:
                handle.write(b"B" * 4096)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return chunk

    monkeypatch.setattr(records.os, "read", mutate_after_read)
    with pytest.raises(records.RecordValidationError, match="identity changed"):
        if operation == "read":
            records.read_regular_bytes(source, maximum_bytes=4096)
        elif operation == "digest":
            records.file_sha256(source)
        else:
            records.copy_opaque_immutable_exclusive(source, destination)
    assert mutated
    assert source.read_bytes() == b"B" * 4096
    if operation == "copy":
        assert not destination.exists()
