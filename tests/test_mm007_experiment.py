"""Lifecycle, provenance, and fail-closed tests for MM-007."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_resolution_diagnostics import experiment, method
from bench.multimodal_warp_diagnostics import experiment as mm006_experiment

_FAKE_IDS = np.asarray(
    [
        video_id
        for video_id, count in {
            "video_10993": 63,
            "video_1580": 64,
            "video_2564": 59,
            "video_3501": 65,
            "video_6860": 65,
            "video_8241": 48,
            "video_874": 66,
            "video_9253": 47,
        }.items()
        for _ in range(count)
    ],
    dtype="<U11",
)
_FAKE_TIMES = np.concatenate(
    [
        np.arange(1.0, 1.0 + 0.5 * count, 0.5, dtype=np.float64)
        for count in (63, 64, 59, 65, 65, 48, 66, 47)
    ]
)
_FAKE_FRAMES = np.zeros((477, 64, 64, 3), dtype=np.uint8)
_FAKE_FRAMES[0, 0, 0, 0] = 1
_FAKE_FRAMES[0, 0, 1, 0] = 2
_FAKE_PIXELS: dict[str, np.ndarray] = {
    "video_ids": _FAKE_IDS,
    "timestamps": _FAKE_TIMES,
    "pixel_current": np.zeros((477, 3, 8, 8), dtype=np.float32),
    "pixel_target": np.zeros((477, 3, 8, 8), dtype=np.float32),
}
_FAKE_EVIDENCE: dict[str, object] = {
    "schema_version": "mm007-test-evidence-v1",
    "alignment": {"rows": 453, "identity_sha256": "a" * 64},
    "normalizer_rows": [],
    "real_metric_rows": [],
    "synthetic_rows": [],
    "parent_classification": experiment.PARENT_CLASSIFICATION,
}
_FAKE_SUMMARY: dict[str, object] = {
    "schema_version": "mm007-test-summary-v1",
    "experiment_id": "MM-007",
    "decision": {"classification": "MM007_resolution_diagnostic_inconclusive", "recommended_next_step": "replicate"},
}


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def live_snapshot() -> dict[str, object]:
    mm006_root = experiment.REPO_ROOT / experiment.MM006_ROOT
    mm004_root = experiment.REPO_ROOT / experiment.MM004_ROOT
    return {
        "parent": {
            "experiment_id": "MM-006",
            "classification": experiment.PARENT_CLASSIFICATION,
            "verification": {
                "status": "verified",
                "outcomes": "verified_results",
                "classification": experiment.PARENT_CLASSIFICATION,
                "artifact_count": len(mm006_experiment.ARTIFACT_FILES),
            },
            "live_path": str(experiment.MM006_ROOT),
            "copy_path": str(experiment.MM006_COPY_ROOT),
            "files": experiment._records(mm006_root, mm006_experiment.COMPLETED_FILES),
            "selected_files": [str(path) for path in experiment.MM006_SELECTED],
            "pinned": {str(path): value for path, value in experiment.MM006_PINS.items()},
            "replay": {"classification": experiment.PARENT_CLASSIFICATION},
            "scientific_relationship": (
                "outcome-informed direct child reusing the same eight videos; not independent evidence"
            ),
        },
        "media_lineage": {
            "experiment_id": "MM-004",
            "classification": experiment.MM004_CLASSIFICATION,
            "verification": {
                "status": "verified",
                "outcomes": "verified_results",
                "classification": experiment.MM004_CLASSIFICATION,
                "artifact_count": len(experiment.mm004_experiment.ARTIFACT_FILES),
            },
            "live_path": str(experiment.MM004_ROOT),
            "copy_path": str(experiment.MM004_COPY_ROOT),
            "files": experiment._records(mm004_root, experiment.mm004_experiment.COMPLETED_FILES),
            "selected_files": [str(path) for path in experiment.MM004_SELECTED],
            "pinned": {str(path): value for path, value in experiment.MM004_PINS.items()},
            "authenticated_by_mm006_input_manifest": experiment._file_record(
                mm004_root / experiment.INPUT_MANIFEST_FILE
            ),
        },
    }


def _patch_fake_lifecycle(
    monkeypatch: pytest.MonkeyPatch, snapshot: dict[str, object]
) -> None:
    fake_arrays = {
        "video_ids": _FAKE_IDS,
        "timestamps": _FAKE_TIMES,
        "frames_uint8": _FAKE_FRAMES,
    }
    fake_array_sha256 = {
        name: experiment._array_sha256(value)
        for name, value in fake_arrays.items()
    }
    monkeypatch.setattr(experiment, "FRAME_ARRAY_SHA256", fake_array_sha256)
    monkeypatch.setattr(
        experiment,
        "FRAMES_UINT8_SHA256",
        fake_array_sha256["frames_uint8"],
    )
    monkeypatch.setattr(
        experiment,
        "FRAME_PACKAGE_SHA256",
        sha256(experiment._npz_bytes(fake_arrays)).hexdigest(),
    )
    monkeypatch.setattr(experiment, "_verify_live_inputs", lambda: deepcopy(snapshot))
    monkeypatch.setattr(
        experiment,
        "_source_hashes",
        lambda: {f"source-{index:02d}": f"{index:064x}" for index in range(68)},
    )

    def validate_receipts(output: Path, current: Mapping[str, object]) -> dict[str, object]:
        parent = cast(Mapping[str, Any], current["parent"])
        lineage = cast(Mapping[str, Any], current["media_lineage"])
        for relative in experiment.MM006_SELECTED:
            actual = experiment._file_record(output / experiment.MM006_COPY_ROOT / relative)
            if actual != parent["files"][str(relative)]:
                raise experiment.InvalidMM007ParentParity(f"copied MM-006 file differs: {relative}")
        for relative in experiment.MM004_SELECTED:
            actual = experiment._file_record(output / experiment.MM004_COPY_ROOT / relative)
            if actual != lineage["files"][str(relative)]:
                raise experiment.InvalidMM007ParentParity(f"copied MM-004 file differs: {relative}")
        return deepcopy(cast(dict[str, object], parent["replay"]))

    monkeypatch.setattr(experiment, "_validate_receipts", validate_receipts)
    monkeypatch.setattr(experiment, "_strict_pixel_arrays", lambda path: {k: v.copy() for k, v in _FAKE_PIXELS.items()})
    monkeypatch.setattr(
        experiment,
        "_decode_frame_arrays",
        lambda pixels, manifest: {
            "video_ids": _FAKE_IDS.copy(),
            "timestamps": _FAKE_TIMES.copy(),
            "frames_uint8": _FAKE_FRAMES.copy(),
        },
    )

    def load_inputs(output: Path) -> tuple[object, dict[str, object], dict[str, object]]:
        frames = experiment._load_npz(output / experiment.FRAME_FILE, experiment.FRAME_SCHEMA)
        frame_array_sha256 = experiment._validate_frame_array_digests(frames)
        frame_package_sha256 = experiment._validate_frame_package_digest(
            output / experiment.FRAME_FILE
        )
        validation: dict[str, object] = {
            "frame_file": experiment._file_record(output / experiment.FRAME_FILE),
            "frame_array_sha256": frame_array_sha256,
            "frame_package_sha256": frame_package_sha256,
            "frames_uint8_sha256": frame_array_sha256["frames_uint8"],
            "frame_schema": {"fake": {"sha256": "f" * 64}},
            "media_contract": {"fake": True},
            "media_contract_sha256": "c" * 64,
            "low_resolution_parity": {"current_rows": 477, "current_exact": True},
            "parent_alignment_sha256": "d" * 64,
        }
        return {"table": "fake"}, {"parent": "fake"}, validation

    monkeypatch.setattr(experiment, "_load_analysis_inputs", load_inputs)
    monkeypatch.setattr(method, "execute", lambda table, parent: deepcopy(_FAKE_EVIDENCE))

    def validate(value: object) -> dict[str, object]:
        if value != _FAKE_EVIDENCE:
            raise ValueError("fake evidence differs")
        return deepcopy(_FAKE_EVIDENCE)

    monkeypatch.setattr(method, "validate_evidence", validate, raising=False)
    monkeypatch.setattr(method, "summarize", lambda evidence: deepcopy(_FAKE_SUMMARY), raising=False)
    monkeypatch.setattr(
        method,
        "report_text",
        lambda summary: json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
        raising=False,
    )
    monkeypatch.setattr(
        method,
        "frozen_config",
        lambda: {"schema_version": "mm007-test-config-v1"},
        raising=False,
    )


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> Path:
    output = tmp_path / "MM-007"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch, snapshot)
    assert experiment.prepare(output)["outcomes"] == "prepared_only"
    return output


def _complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> Path:
    output = _prepare(tmp_path, monkeypatch, snapshot)
    assert experiment.run(output)["status"] == "completed"
    return output


def test_exact_source_superset_receipts_and_pins() -> None:
    parent_sources = set(mm006_experiment._source_paths())
    sources = set(experiment._source_paths())
    assert len(parent_sources) == 61
    assert len(sources) == 68
    assert parent_sources < sources
    assert {str(path) for path in sources - parent_sources} == {
        "bench/multimodal_resolution_diagnostics/__init__.py",
        "bench/multimodal_resolution_diagnostics/__main__.py",
        "bench/multimodal_resolution_diagnostics/experiment.py",
        "bench/multimodal_resolution_diagnostics/method.py",
        "docs/research/2026-07-15-mm007-physically-matched-resolution-protocol.md",
        "tests/test_mm007_experiment.py",
        "tests/test_mm007_method.py",
    }
    assert len(experiment.PARENT_COPY_FILES) == 6
    assert experiment.FRAMES_UINT8_SHA256 == (
        "46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9"
    )
    assert experiment.FRAME_ARRAY_SHA256 == {
        "video_ids": "06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50",
        "timestamps": "128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55",
        "frames_uint8": experiment.FRAMES_UINT8_SHA256,
    }
    assert experiment.FRAME_PACKAGE_SHA256 == (
        "fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f"
    )
    for relative in experiment.MM006_SELECTED:
        actual = experiment._file_hash(experiment.REPO_ROOT / experiment.MM006_ROOT / relative)
        assert actual == experiment.MM006_PINS[relative]
    for relative in experiment.MM004_SELECTED:
        actual = experiment._file_hash(experiment.REPO_ROOT / experiment.MM004_ROOT / relative)
        assert actual == experiment.MM004_PINS[relative]


def test_prepare_has_six_receipts_uint8_frames_and_no_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, live_snapshot)
    counts = (
        len(experiment.PREPARED_FILES),
        len(experiment.ARTIFACT_FILES),
        len(experiment.COMPLETED_FILES),
    )
    assert counts == (9, 13, 14)
    assert not (output / experiment.STARTED_FILE).exists()
    frames = experiment._load_npz(output / experiment.FRAME_FILE, experiment.FRAME_SCHEMA)
    assert frames["frames_uint8"].dtype == np.uint8
    assert frames["frames_uint8"].shape == (477, 64, 64, 3)
    manifest = cast(dict[str, Any], experiment._read_json(output / experiment.INPUT_MANIFEST_FILE))
    assert manifest["input_validation"]["frame_array_sha256"] == experiment.FRAME_ARRAY_SHA256
    assert manifest["input_validation"]["frame_package_sha256"] == experiment.FRAME_PACKAGE_SHA256
    assert manifest["config"]["lifecycle"]["frame_array_sha256"] == experiment.FRAME_ARRAY_SHA256
    assert manifest["config"]["lifecycle"]["frame_package_sha256"] == experiment.FRAME_PACKAGE_SHA256
    files, directories = experiment._tree_members(output)
    assert files == set(experiment.PREPARED_FILES)
    assert directories == experiment._expected_directories(experiment.PREPARED_FILES)


def test_prepare_never_calls_scientific_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = tmp_path / "MM-007"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch, live_snapshot)

    def forbidden(*args: object) -> object:
        del args
        raise AssertionError("prepare called resolution search")

    monkeypatch.setattr(method, "execute", forbidden)
    assert experiment.prepare(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()


def test_marker_is_immediately_before_execute_and_interruption_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, live_snapshot)

    def fail(table: object, parent: object) -> object:
        del table, parent
        marker = output / experiment.STARTED_FILE
        assert marker.exists() and stat_mode(marker) == 0o444
        raise RuntimeError("resolution search failed after marker")

    monkeypatch.setattr(method, "execute", fail)
    with pytest.raises(experiment.InvalidMM007Package, match="after marker"):
        experiment.run(output)
    with pytest.raises(experiment.InvalidMM007Package, match="membership mismatch"):
        experiment.run(output)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_completed_fast_regenerates_and_semantic_redecodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    calls = {"execute": 0, "decode": 0}
    output = _prepare(tmp_path, monkeypatch, live_snapshot)

    def execute(table: object, parent: object) -> dict[str, object]:
        del table, parent
        calls["execute"] += 1
        return deepcopy(_FAKE_EVIDENCE)

    def decode(pixels: object, manifest: object) -> dict[str, np.ndarray]:
        del pixels, manifest
        calls["decode"] += 1
        return {"video_ids": _FAKE_IDS.copy(), "timestamps": _FAKE_TIMES.copy(), "frames_uint8": _FAKE_FRAMES.copy()}

    monkeypatch.setattr(method, "execute", execute)
    monkeypatch.setattr(experiment, "_decode_frame_arrays", decode)
    experiment.run(output)
    assert calls == {"execute": 1, "decode": 0}
    assert experiment.verify(output)["outcomes"] == "verified_results"
    assert calls == {"execute": 2, "decode": 0}
    semantic = experiment.verify_semantic(output)
    assert semantic["outcomes"] == "verified_semantic_results"
    assert calls == {"execute": 3, "decode": 1}
    marker = cast(dict[str, Any], experiment._read_json(output / experiment.STARTED_FILE))
    assert marker["mm006_receipt_sha256"] == {
        str(path): value for path, value in experiment.MM006_PINS.items()
    }
    assert marker["mm004_receipt_sha256"] == {
        str(path): value for path, value in experiment.MM004_PINS.items()
    }
    assert marker["frame_array_sha256"] == experiment.FRAME_ARRAY_SHA256
    assert marker["frame_package_sha256"] == experiment.FRAME_PACKAGE_SHA256


def test_within_pooling_block_permutation_fails_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, live_snapshot)
    path = output / experiment.FRAME_FILE
    arrays = experiment._load_npz(path, experiment.FRAME_SCHEMA)
    original_pool = experiment._pool_8x8(arrays["frames_uint8"])
    changed = arrays["frames_uint8"].copy()
    changed[0, 0, 0, 0], changed[0, 0, 1, 0] = (
        changed[0, 0, 1, 0],
        changed[0, 0, 0, 0],
    )
    assert not np.array_equal(changed, arrays["frames_uint8"])
    assert np.array_equal(experiment._pool_8x8(changed), original_pool)
    arrays["frames_uint8"] = changed
    path.write_bytes(experiment._npz_bytes(arrays))
    with pytest.raises(experiment.InvalidMM007Package, match="frames_uint8 digest differs"):
        experiment.verify(output)
    with pytest.raises(experiment.InvalidMM007Package, match="frames_uint8 digest differs"):
        experiment.run(output)
    assert not (output / experiment.STARTED_FILE).exists()


def test_semantic_redecode_byte_mismatch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, live_snapshot)
    changed = _FAKE_FRAMES.copy()
    changed[0, 0, 0, 0] = 3
    monkeypatch.setattr(
        experiment,
        "_decode_frame_arrays",
        lambda pixels, manifest: {
            "video_ids": _FAKE_IDS.copy(),
            "timestamps": _FAKE_TIMES.copy(),
            "frames_uint8": changed,
        },
    )
    with pytest.raises(experiment.InvalidMM007Package, match="semantic redecode differs.*frames_uint8"):
        experiment.verify_semantic(output)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation,message,completed,classification",
    [
        ("artifact", "artifact manifest", True, "invalid_MM007_package"),
        ("extra", "membership mismatch", False, "invalid_MM007_package"),
        ("symlink", "non-regular file", False, "invalid_MM007_package"),
        ("source", "recompute", False, "invalid_MM007_package"),
        ("parent", "copied MM-006 file differs", False, "invalid_MM007_parent_parity"),
        ("frames", "frame package digest differs", False, "invalid_MM007_package"),
        ("marker_mode", "formal marker", True, "invalid_MM007_package"),
    ],
)
def test_integrity_mutations_fail_closed(
    mutation: str,
    message: str,
    completed: bool,
    classification: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = (
        _complete(tmp_path, monkeypatch, live_snapshot)
        if completed
        else _prepare(tmp_path, monkeypatch, live_snapshot)
    )
    if mutation == "artifact":
        path = output / experiment.RESULT_FILE
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "extra":
        (output / "unexpected.txt").write_text("x", encoding="utf-8")
    elif mutation == "symlink":
        path = output / experiment.MM004_COPY_ROOT / experiment.INPUT_MANIFEST_FILE
        path.unlink()
        path.symlink_to(experiment.REPO_ROOT / experiment.MM004_ROOT / experiment.INPUT_MANIFEST_FILE)
    elif mutation == "source":
        hashes = experiment._source_hashes()
        monkeypatch.setattr(experiment, "_source_hashes", lambda: {**hashes, "drift.py": "0" * 64})
    elif mutation == "parent":
        path = output / experiment.MM006_COPY_ROOT / experiment.MM006_SELECTED[-1]
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "frames":
        path = output / experiment.FRAME_FILE
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "marker_mode":
        os.chmod(output / experiment.STARTED_FILE, 0o644)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises((experiment.InvalidMM007Package, experiment.InvalidMM007ParentParity), match=message) as invalid:
        experiment.verify(output)
    assert cast(Any, invalid.value).classification == classification


def test_strict_npz_and_exact_area_pool(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 256, size=(3, 64, 64, 3), dtype=np.uint8)
    pooled = experiment._pool_8x8(frames)
    expected = np.stack(
        [experiment.mm004_experiment._area_pool_frame(frame.astype(np.float32) / np.float32(255.0)) for frame in frames]
    )
    assert np.array_equal(pooled, expected)
    wrong = tmp_path / "wrong.npz"
    wrong.write_bytes(experiment._npz_bytes({"video_ids": _FAKE_IDS, "timestamps": _FAKE_TIMES}))
    with pytest.raises(ValueError, match="keys differ"):
        experiment._load_npz(wrong, experiment.FRAME_SCHEMA)


def test_media_contract_is_exact_and_rejects_ffmpeg_drift() -> None:
    path = experiment.REPO_ROOT / experiment.MM004_ROOT / experiment.INPUT_MANIFEST_FILE
    manifest = cast(dict[str, Any], experiment._read_json(path))
    contract = experiment._media_contract(manifest)
    assert contract["media_sha256"] == dict(experiment.mm001_dataset.EXTRACTED_MP4_SHA256)
    assert cast(Mapping[str, object], contract["ffmpeg"])["sha256"] == experiment.FFMPEG_SHA256
    changed = deepcopy(manifest)
    changed["input_validation"]["pixel_preparation"]["media"]["ffmpeg"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="media contract differs"):
        experiment._media_contract(changed)


def test_rehashed_evidence_tamper_reaches_primitive_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, live_snapshot)
    evidence_path = output / experiment.EVIDENCE_FILE
    evidence = cast(dict[str, Any], experiment._read_json(evidence_path))
    evidence["alignment"] = {"rows": 452, "identity_sha256": "0" * 64}
    evidence_path.write_bytes(experiment._json_bytes(evidence))
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_json_exclusive(output / experiment.ARTIFACT_MANIFEST_FILE, experiment._artifact_manifest(output))
    with pytest.raises(experiment.InvalidMM007Package, match="fake evidence differs"):
        experiment.verify(output)


def test_canonical_path_overlap_and_exclusive_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = tmp_path / "canonical" / "MM-007"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", expected)
    with pytest.raises(experiment.InvalidMM007Package, match="canonical path"):
        experiment.verify(tmp_path / "other")
    parent = experiment.REPO_ROOT / experiment.MM006_ROOT
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", parent)
    with pytest.raises(ValueError, match="overlaps a protected parent"):
        experiment._assert_expected_output(parent)
    path = tmp_path / "created.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment._write_json_exclusive(path, {"replacement": True})
