"""Lifecycle, receipt, and fail-closed tests for MM-004."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_diagnostics import experiment as mm002_experiment
from bench.multimodal_preflight import experiment as mm001_experiment
from bench.multimodal_spatial_diagnostics import experiment, method
from bench.multimodal_transform_diagnostics import experiment as mm003_experiment

_FAKE_PREFLIGHT: dict[str, object] = {
    "passed": True,
    "rows_compared": 16,
    "rtol": 1e-12,
    "atol": 1e-12,
    "max_absolute_error": 0.0,
}
_FAKE_EVIDENCE: dict[str, object] = {
    "schema_version": "mm004-test-evidence-v1",
    "parent_preflight": _FAKE_PREFLIGHT,
    "synthetic_panels": [{"panel_seed": seed} for seed in method.SYNTHETIC_SEEDS],
    "primitive_rows": [],
}
_FAKE_SUMMARY: dict[str, object] = {
    "schema_version": "mm004-test-summary-v1",
    "experiment_id": "MM-004",
    "decision": {
        "classification": "MM004_diagnostic_inconclusive",
        "recommended_fix": "enlarge the independently sampled video panel",
    },
}


def _fake_pixels(feature_table: Any, manifest: object) -> dict[str, np.ndarray]:
    del manifest
    video_ids = np.asarray(feature_table.video_ids, dtype="<U11")
    timestamps = np.asarray(feature_table.timestamps, dtype="<f8")
    current = np.empty((477, 3, 8, 8), dtype=np.float32)
    for row, timestamp in enumerate(timestamps):
        current[row].fill(np.float32((float(timestamp) % 10.0) / 10.0))
    target = np.asarray(np.minimum(current + np.float32(0.01), np.float32(1.0)), dtype=np.float32)
    return {
        "video_ids": video_ids,
        "timestamps": timestamps,
        "pixel_current": current,
        "pixel_target": target,
    }


def _patch_fake_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = experiment._verify_live_parent()
    monkeypatch.setattr(experiment, "_verify_live_parent", lambda: deepcopy(snapshot))
    monkeypatch.setattr(experiment, "_extract_pixel_arrays", _fake_pixels)
    monkeypatch.setattr(
        experiment,
        "_raw_grid_table",
        lambda video_ids, timestamps, current, target: {
            "video_ids": np.asarray(video_ids).copy(),
            "timestamps": np.asarray(timestamps).copy(),
            "current": np.asarray(current).copy(),
            "target": np.asarray(target).copy(),
        },
    )
    monkeypatch.setattr(experiment, "_parent_preflight", lambda table, parent: deepcopy(_FAKE_PREFLIGHT))
    monkeypatch.setattr(experiment, "_execute", lambda taesd, pixels, parent: deepcopy(_FAKE_EVIDENCE))
    monkeypatch.setattr(method, "history_table", lambda table: table)
    monkeypatch.setattr(
        method,
        "synthetic_panel",
        lambda table, seed: (table, {"panel_seed": seed}),
    )

    def validate_evidence(value: object) -> dict[str, object]:
        if value != _FAKE_EVIDENCE:
            raise ValueError("fake evidence differs")
        return deepcopy(_FAKE_EVIDENCE)

    monkeypatch.setattr(method, "validate_evidence", validate_evidence)
    monkeypatch.setattr(experiment, "_summarize", lambda evidence, parent: deepcopy(_FAKE_SUMMARY))
    monkeypatch.setattr(
        method,
        "report_text",
        lambda summary: json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
    )
    monkeypatch.setattr(method, "config_record", lambda: {"schema_version": "mm004-test-config-v1"})


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "MM-004"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch)
    result = experiment.prepare(output)
    assert result == {
        "status": "prepared_only",
        "outcomes": "prepared_only",
        "classification": "no_outcomes_before_formal_marker",
        "artifact_count": len(experiment.PREPARED_FILES),
    }
    return output


def _complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = _prepare(tmp_path, monkeypatch)
    result = experiment.run(output)
    assert result["status"] == "completed"
    return output


def test_parent_chain_and_exact_source_superset_remain_sealed() -> None:
    assert mm001_experiment.verify(mm001_experiment.DEFAULT_OUTPUT)["outcomes"] == "verified_results"
    assert mm002_experiment.verify(mm002_experiment.DEFAULT_OUTPUT)["outcomes"] == "verified_results"
    assert mm003_experiment.verify(mm003_experiment.DEFAULT_OUTPUT)["outcomes"] == "verified_results"

    parent_sources = set(mm003_experiment._source_paths())
    sources = set(experiment._source_paths())
    assert len(parent_sources) == 40
    assert len(sources) == 47
    assert parent_sources < sources
    assert {str(path) for path in sources - parent_sources} == {
        "bench/multimodal_spatial_diagnostics/__init__.py",
        "bench/multimodal_spatial_diagnostics/__main__.py",
        "bench/multimodal_spatial_diagnostics/experiment.py",
        "bench/multimodal_spatial_diagnostics/method.py",
        "docs/research/2026-07-15-mm004-spatial-history-signal-isolation-protocol.md",
        "tests/test_mm004_experiment.py",
        "tests/test_mm004_method.py",
    }
    for parent in (mm001_experiment, mm002_experiment, mm003_experiment):
        assert not any("multimodal_spatial_diagnostics" in str(path) for path in parent._source_paths())


def test_prepare_has_exact_receipts_pixels_and_no_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path, monkeypatch)

    assert len(experiment.PREPARED_FILES) == 10
    assert len(experiment.ARTIFACT_FILES) == 14
    assert len(experiment.COMPLETED_FILES) == 15
    assert experiment.verify(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()
    files, directories = experiment._tree_members(output)
    assert files == set(experiment.PREPARED_FILES)
    assert directories == experiment._expected_directories(experiment.PREPARED_FILES)
    parent_files, _ = experiment._tree_members(output / experiment.MM003_COPY_ROOT)
    assert parent_files == set(experiment.MM003_SELECTED)

    pixels = experiment._load_npz(output / experiment.PIXEL_FILE, experiment.PIXEL_SCHEMA)
    assert set(pixels) == experiment.PIXEL_KEYS
    manifest = cast(dict[str, Any], experiment._read_json(output / experiment.INPUT_MANIFEST_FILE))
    assert manifest["source_count"] == 47
    assert len(manifest["parent"]["files"]) == 19
    assert manifest["input_validation"]["parent_preflight"] == _FAKE_PREFLIGHT
    assert manifest["input_validation"]["pixel_preparation"]["file"] == experiment._file_record(
        output / experiment.PIXEL_FILE
    )
    assert (output / experiment.PROTOCOL_COPY_FILE).stat().st_mode & 0o777 == 0o644
    assert (output / experiment.INPUT_MANIFEST_FILE).stat().st_mode & 0o777 == 0o644
    assert (output / experiment.PIXEL_FILE).stat().st_mode & 0o777 == 0o644


def test_formal_marker_is_read_only_and_interruption_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path, monkeypatch)

    class ExpectedFailure(RuntimeError):
        pass

    def fail_after_marker(*args: object) -> object:
        del args
        marker = output / experiment.STARTED_FILE
        assert marker.is_file()
        assert marker.stat().st_mode & 0o777 == 0o444
        raise ExpectedFailure("fit failed after formal marker")

    monkeypatch.setattr(experiment, "_execute", fail_after_marker)
    with pytest.raises(experiment.InvalidMM004Package, match="after formal marker"):
        experiment.run(output)
    assert (output / experiment.STARTED_FILE).stat().st_mode & 0o777 == 0o444
    with pytest.raises(experiment.InvalidMM004Package, match="membership mismatch"):
        experiment.run(output)
    with pytest.raises(experiment.InvalidMM004Package, match="membership mismatch"):
        experiment.verify(output)


def test_fake_completed_fast_and_semantic_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete(tmp_path, monkeypatch)

    assert experiment.verify(output) == {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": "MM004_diagnostic_inconclusive",
        "artifact_count": len(experiment.ARTIFACT_FILES),
    }
    assert experiment.verify_semantic(output)["outcomes"] == "verified_semantic_results"
    assert (output / experiment.STARTED_FILE).stat().st_mode & 0o777 == 0o444
    assert (output / experiment.ARTIFACT_MANIFEST_FILE).stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    "mutation,message,completed",
    [
        ("artifact_tamper", "artifact manifest", True),
        ("extra", "membership mismatch", False),
        ("symlink", "non-regular file", False),
        ("source_drift", "recompute", False),
        ("parent_copy_drift", "live sealed parent", False),
        ("pixel_mode", "generated file mode", False),
        ("marker_mode", "formal marker", True),
        ("manifest_mode", "generated file mode", True),
    ],
)
def test_integrity_defects_fail_closed(
    mutation: str,
    message: str,
    completed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete(tmp_path, monkeypatch) if completed else _prepare(tmp_path, monkeypatch)
    if mutation == "artifact_tamper":
        path = output / experiment.RESULT_FILE
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "extra":
        (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "symlink":
        copied = output / experiment.MM003_COPY_ROOT / experiment.MM003_SELECTED[1]
        copied.unlink()
        copied.symlink_to(experiment.REPO_ROOT / experiment.MM003_ROOT / experiment.MM003_SELECTED[1])
    elif mutation == "source_drift":
        source = experiment._source_hashes()
        monkeypatch.setattr(
            experiment,
            "_source_hashes",
            lambda: {**source, "synthetic/source-drift.py": "0" * 64},
        )
    elif mutation == "parent_copy_drift":
        copied = output / experiment.MM003_COPY_ROOT / experiment.MM003_SELECTED[-1]
        copied.write_bytes(copied.read_bytes() + b" ")
    elif mutation == "pixel_mode":
        os.chmod(output / experiment.PIXEL_FILE, 0o600)
    elif mutation == "marker_mode":
        os.chmod(output / experiment.STARTED_FILE, 0o644)
    elif mutation == "manifest_mode":
        os.chmod(output / experiment.ARTIFACT_MANIFEST_FILE, 0o600)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(experiment.InvalidMM004Package, match=message) as invalid:
        experiment.verify(output)
    assert invalid.value.classification == "invalid_MM004_package"


def test_rehashed_evidence_tamper_reaches_strict_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete(tmp_path, monkeypatch)
    evidence_path = output / experiment.EVIDENCE_FILE
    evidence = cast(dict[str, object], experiment._read_json(evidence_path))
    evidence["primitive_rows"] = ["tampered"]
    evidence_path.write_bytes(experiment._json_bytes(evidence))
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_json_exclusive(
        output / experiment.ARTIFACT_MANIFEST_FILE,
        experiment._artifact_manifest(output),
        0o644,
    )

    with pytest.raises(experiment.InvalidMM004Package, match="fake evidence differs"):
        experiment.verify(output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("parent_preflight", "parent preflight differs"),
        ("synthetic_panels", "synthetic panel provenance"),
    ],
)
def test_rehashed_provenance_tamper_fails_fast_verification(
    mutation: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete(tmp_path, monkeypatch)
    evidence_path = output / experiment.EVIDENCE_FILE
    evidence = cast(dict[str, Any], experiment._read_json(evidence_path))
    if mutation == "parent_preflight":
        cast(dict[str, Any], evidence["parent_preflight"])["max_absolute_error"] = 1e-6
    else:
        cast(list[dict[str, Any]], evidence["synthetic_panels"])[0]["panel_seed"] = -1
    evidence_path.write_bytes(experiment._json_bytes(evidence))
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_json_exclusive(
        output / experiment.ARTIFACT_MANIFEST_FILE,
        experiment._artifact_manifest(output),
        0o644,
    )
    monkeypatch.setattr(method, "validate_evidence", lambda value: cast(dict[str, object], value))

    with pytest.raises(experiment.InvalidMM004Package, match=message):
        experiment.verify(output)


def test_parent_parity_failure_has_dedicated_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        method,
        "parent_preflight_record",
        lambda table, parent: (_ for _ in ()).throw(ValueError("parent rows differ")),
    )

    with pytest.raises(experiment.InvalidMM004ParentParity, match="parent rows differ") as invalid:
        experiment._parent_preflight({}, {})
    assert invalid.value.classification == "invalid_MM004_parent_parity"


def test_canonical_path_and_parent_overlap_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "canonical" / "MM-004"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", expected)
    with pytest.raises(experiment.InvalidMM004Package, match="canonical path"):
        experiment.verify(tmp_path / "other")

    parent = experiment.REPO_ROOT / experiment.MM003_ROOT
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", parent)
    with pytest.raises(ValueError, match="overlaps a protected parent"):
        experiment._assert_expected_output(parent)


def test_exclusive_writer_refuses_an_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "already-created.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment._write_json_exclusive(path, {"replacement": True})
    assert path.read_text(encoding="utf-8") == "existing"


def test_strict_npz_rejects_wrong_dtype_and_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "pixels.npz"
    arrays = {
        "video_ids": np.full(477, "video_10993", dtype="<U11"),
        "timestamps": np.zeros(477, dtype=np.float64),
        "pixel_current": np.zeros((477, 3, 8, 8), dtype=np.float64),
        "pixel_target": np.zeros((477, 3, 8, 8), dtype=np.float32),
        "unexpected": np.zeros(1, dtype=np.float32),
    }
    path.write_bytes(experiment._npz_bytes(arrays))
    with pytest.raises(ValueError, match="keys differ"):
        experiment._load_npz(path, experiment.PIXEL_SCHEMA)


def test_rehashed_prepared_pixels_must_remain_in_rgb_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path, monkeypatch)
    pixel_path = output / experiment.PIXEL_FILE
    arrays = experiment._load_npz(pixel_path, experiment.PIXEL_SCHEMA)
    arrays["pixel_current"][0, 0, 0, 0] = np.float32(1.01)
    pixel_path.write_bytes(experiment._npz_bytes(arrays))

    manifest_path = output / experiment.INPUT_MANIFEST_FILE
    manifest = cast(dict[str, Any], experiment._read_json(manifest_path))
    pixel_record = cast(
        dict[str, Any],
        cast(dict[str, Any], manifest["input_validation"])["pixel_preparation"],
    )
    pixel_record["file"] = experiment._file_record(pixel_path)
    pixel_record["schema"] = {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": experiment._array_sha256(array),
        }
        for name, array in sorted(arrays.items())
    }
    manifest_path.write_bytes(experiment._json_bytes(manifest))

    with pytest.raises(experiment.InvalidMM004Package, match=r"must remain in \[0,1\]"):
        experiment.verify(output)


def test_pixel_extraction_uses_authenticated_decoder_contract_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = experiment.REPO_ROOT / experiment.MM003_ROOT
    feature_table, _ = mm001_experiment._load_feature_table(parent / "inputs/MM-001/MM-001-features.npz")
    mm001_manifest = experiment._read_json(parent / "inputs/MM-001/input-manifest.json")
    contract = experiment._media_contract(mm001_manifest)
    expected_hashes = cast(dict[str, str], contract["media_sha256"])
    frame_counts = cast(dict[str, int], contract["frame_count_2fps"])

    def fake_hashes(
        cache_path: Path,
        *,
        expected_hashes: dict[str, str],
        expected_sizes: dict[str, int],
    ) -> dict[str, str]:
        del cache_path, expected_sizes
        return dict(expected_hashes)

    def fake_decode(path: Path, *, ffmpeg: str) -> np.ndarray:
        del ffmpeg
        video_id = path.stem
        count = frame_counts[video_id]
        levels = np.arange(count, dtype=np.float32) / np.float32(100.0)
        return np.broadcast_to(levels[:, None, None, None], (count, 64, 64, 3)).copy()

    monkeypatch.setattr(experiment.mm001_dataset, "validate_media_hashes", fake_hashes)
    monkeypatch.setattr(experiment.mm001_backends, "decode_video_frames", fake_decode)
    first = experiment._extract_pixel_arrays(feature_table, mm001_manifest)
    second = experiment._extract_pixel_arrays(feature_table, mm001_manifest)

    assert set(first) == experiment.PIXEL_KEYS
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["pixel_current"].dtype == np.float32
    assert first["pixel_current"].shape == (477, 3, 8, 8)
    assert first["pixel_target"].shape == (477, 3, 8, 8)
    assert np.all(first["pixel_target"] >= first["pixel_current"])
    assert set(expected_hashes) == set(experiment.mm001_dataset.SAMPLE_VIDEO_IDS)
