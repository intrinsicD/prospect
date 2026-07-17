"""Lifecycle, receipt, alignment, and fail-closed tests for MM-005."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_horizon_diagnostics import experiment, method
from bench.multimodal_preflight import experiment as mm001_experiment
from bench.multimodal_spatial_diagnostics import experiment as mm004_experiment

_FAKE_EVIDENCE: dict[str, object] = {
    "schema_version": "mm005-test-evidence-v1",
    "alignment": {"identity_sha256": "a" * 64},
    "synthetic_panels": [{"panel_seed": seed} for seed in method.SYNTHETIC_SEEDS],
    "primitive_rows": [],
}
_FAKE_SUMMARY: dict[str, object] = {
    "schema_version": "mm005-test-summary-v1",
    "experiment_id": "MM-005",
    "decision": {
        "classification": "MM005_diagnostic_inconclusive",
        "recommended_next_step": "repair or replicate the assay",
    },
}
_FAKE_ALIGNMENT: dict[str, object] = {
    "passed": True,
    "raw_rows": 477,
    "matched_rows": 453,
    "matched_identity_sha256": experiment.MATCHED_IDENTITY_SHA256,
}


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def live_parent_snapshot() -> dict[str, object]:
    root = experiment.REPO_ROOT / experiment.MM004_ROOT
    replay = experiment._replay_parent(root)
    return {
        "experiment_id": "MM-004",
        "classification": experiment.PARENT_CLASSIFICATION,
        "verification": {
            "status": "verified",
            "outcomes": "verified_results",
            "classification": experiment.PARENT_CLASSIFICATION,
            "artifact_count": len(mm004_experiment.ARTIFACT_FILES),
        },
        "live_path": str(experiment.MM004_ROOT),
        "copy_path": str(experiment.MM004_COPY_ROOT),
        "files": experiment._records(root, mm004_experiment.COMPLETED_FILES),
        "selected_files": [str(path) for path in experiment.MM004_SELECTED],
        "pinned": {str(path): digest for path, digest in experiment.MM004_PINS.items()},
        "replay": replay,
        "scientific_relationship": (
            "outcome-informed direct child reusing the same eight videos; not independent evidence"
        ),
    }


def _patch_fake_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> None:
    frozen_snapshot = deepcopy(snapshot)
    validation: dict[str, object] = {
        "feature_file": {"sha256": "1" * 64, "bytes": 1, "mode": 0o664},
        "feature_schema": {},
        "component_file": {"sha256": "2" * 64, "bytes": 1, "mode": 0o664},
        "component_schema": {},
        "pixel_file": {"sha256": "3" * 64, "bytes": 1, "mode": 0o644},
        "pixel_schema": {},
        "parent_replay": deepcopy(cast(dict[str, object], snapshot["replay"])),
        "alignment": deepcopy(_FAKE_ALIGNMENT),
        "alignment_sha256": experiment._canonical_json_sha256(_FAKE_ALIGNMENT),
    }
    monkeypatch.setattr(experiment, "_verify_live_parent", lambda: deepcopy(frozen_snapshot))
    monkeypatch.setattr(
        experiment,
        "_load_analysis_inputs",
        lambda output: ({"domain": "taesd"}, {"domain": "pixel"}, deepcopy(validation)),
    )
    monkeypatch.setattr(experiment, "_execute", lambda taesd, pixels: deepcopy(_FAKE_EVIDENCE))

    def validate_provenance(taesd: object, pixels: object, evidence: dict[str, object]) -> None:
        del taesd, pixels
        if evidence.get("alignment") != _FAKE_EVIDENCE["alignment"]:
            raise ValueError("fake evidence alignment differs")
        if evidence.get("synthetic_panels") != _FAKE_EVIDENCE["synthetic_panels"]:
            raise ValueError("fake synthetic panel provenance differs")

    monkeypatch.setattr(experiment, "_validate_evidence_provenance", validate_provenance)

    def validate_evidence(value: object) -> dict[str, object]:
        if value != _FAKE_EVIDENCE:
            raise ValueError("fake evidence differs")
        return deepcopy(_FAKE_EVIDENCE)

    monkeypatch.setattr(method, "validate_evidence", validate_evidence)
    monkeypatch.setattr(experiment, "_summarize", lambda evidence: deepcopy(_FAKE_SUMMARY))
    monkeypatch.setattr(
        method,
        "report_text",
        lambda summary: json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
    )
    monkeypatch.setattr(method, "config_record", lambda: {"schema_version": "mm005-test-config-v1"})


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> Path:
    output = tmp_path / "MM-005"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch, snapshot)
    result = experiment.prepare(output)
    assert result == {
        "status": "prepared_only",
        "outcomes": "prepared_only",
        "classification": "no_outcomes_before_formal_marker",
        "artifact_count": len(experiment.PREPARED_FILES),
    }
    return output


def _complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> Path:
    output = _prepare(tmp_path, monkeypatch, snapshot)
    result = experiment.run(output)
    assert result["status"] == "completed"
    return output


def test_parent_chain_and_exact_source_superset_remain_sealed(
    live_parent_snapshot: dict[str, object],
) -> None:
    assert cast(dict[str, object], live_parent_snapshot["verification"])["outcomes"] == "verified_results"
    parent_sources = set(mm004_experiment._source_paths())
    sources = set(experiment._source_paths())
    assert len(parent_sources) == 47
    assert len(sources) == 54
    assert parent_sources < sources
    assert {str(path) for path in sources - parent_sources} == {
        "bench/multimodal_horizon_diagnostics/__init__.py",
        "bench/multimodal_horizon_diagnostics/__main__.py",
        "bench/multimodal_horizon_diagnostics/experiment.py",
        "bench/multimodal_horizon_diagnostics/method.py",
        "docs/research/2026-07-15-mm005-matched-half-horizon-replay-protocol.md",
        "tests/test_mm005_experiment.py",
        "tests/test_mm005_method.py",
    }
    assert not any("multimodal_horizon_diagnostics" in str(path) for path in parent_sources)


def test_real_parent_alignment_is_exact_before_marker() -> None:
    root = experiment.REPO_ROOT / experiment.MM004_ROOT
    feature_table, _ = mm001_experiment._load_feature_table(root / experiment.FEATURE_RELATIVE)
    components = experiment._strict_component_arrays(root / experiment.COMPONENT_RELATIVE)
    pixels = experiment._strict_pixel_arrays(root / experiment.PIXEL_RELATIVE)
    record = cast(dict[str, Any], experiment._alignment_record(feature_table, components, pixels))

    assert record["passed"] is True
    assert record["matched_rows"] == 453
    assert record["matched_counts"] == experiment.MATCHED_COUNTS
    assert record["matched_identity_sha256"] == experiment.MATCHED_IDENTITY_SHA256
    assert record["fold_train_rows"] == [332, 335, 346, 346]
    assert record["taesd"]["bit_exact_saved_target_parity"] is True
    assert record["pixel"]["bit_exact_saved_target_parity"] is True
    assert all(len(value) == 64 for value in record["taesd"]["array_sha256"].values())
    assert all(len(value) == 64 for value in record["pixel"]["array_sha256"].values())


def test_alignment_tamper_has_dedicated_pre_marker_classification() -> None:
    root = experiment.REPO_ROOT / experiment.MM004_ROOT
    feature_table, _ = mm001_experiment._load_feature_table(root / experiment.FEATURE_RELATIVE)
    components = experiment._strict_component_arrays(root / experiment.COMPONENT_RELATIVE)
    pixels = experiment._strict_pixel_arrays(root / experiment.PIXEL_RELATIVE)
    tampered = {name: value.copy() for name, value in components.items()}
    tampered["target_taesd_latents"][0, 0, 0, 0] += np.float32(0.125)

    with pytest.raises(experiment.InvalidMM005ParentAlignment, match="bit-exactly") as invalid:
        experiment._alignment_record(feature_table, tampered, pixels)
    assert invalid.value.classification == "invalid_MM005_parent_alignment"


def test_cadence_tamper_has_dedicated_pre_marker_classification() -> None:
    root = experiment.REPO_ROOT / experiment.MM004_ROOT
    feature_table, _ = mm001_experiment._load_feature_table(root / experiment.FEATURE_RELATIVE)
    components = experiment._strict_component_arrays(root / experiment.COMPONENT_RELATIVE)
    pixels = experiment._strict_pixel_arrays(root / experiment.PIXEL_RELATIVE)
    timestamps = np.asarray(feature_table.timestamps, dtype=np.float64).copy()
    timestamps[1] += 0.125
    pixels["timestamps"] = timestamps.copy()

    with pytest.raises(experiment.InvalidMM005ParentAlignment, match="cadence"):
        experiment._alignment_record(replace(feature_table, timestamps=timestamps), components, pixels)


def test_fast_provenance_replays_arrays_and_synthetic_panels_without_fits() -> None:
    root = experiment.REPO_ROOT / experiment.MM004_ROOT
    feature_table, _ = mm001_experiment._load_feature_table(root / experiment.FEATURE_RELATIVE)
    components = experiment._strict_component_arrays(root / experiment.COMPONENT_RELATIVE)
    pixels = experiment._strict_pixel_arrays(root / experiment.PIXEL_RELATIVE)
    ids = np.asarray(feature_table.video_ids, dtype="<U11")
    times = np.asarray(feature_table.timestamps, dtype="<f8")
    taesd_raw = experiment._raw_grid_table(
        ids,
        times,
        components["taesd_latents"],
        components["target_taesd_latents"],
    )
    pixel_raw = experiment._raw_grid_table(
        ids,
        times,
        pixels["pixel_current"],
        pixels["pixel_target"],
    )
    taesd_panel = method.matched_panel(taesd_raw)
    pixel_panel = method.matched_panel(pixel_raw)
    records = [method.synthetic_panel(taesd_panel, seed)[1] for seed in method.SYNTHETIC_SEEDS]
    evidence: dict[str, object] = {
        "alignment": method.alignment_record(taesd_panel, pixel_panel),
        "synthetic_panels": records,
    }
    experiment._validate_evidence_provenance(taesd_raw, pixel_raw, evidence)

    tampered = deepcopy(evidence)
    cast(dict[str, Any], tampered["alignment"])["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="alignment differs"):
        experiment._validate_evidence_provenance(taesd_raw, pixel_raw, tampered)

    tampered = deepcopy(evidence)
    cast(list[dict[str, Any]], tampered["synthetic_panels"])[0]["panel_seed"] = -1
    with pytest.raises(ValueError, match="synthetic panel provenance"):
        experiment._validate_evidence_provenance(taesd_raw, pixel_raw, tampered)


def test_prepare_has_exact_receipts_and_no_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, live_parent_snapshot)

    assert len(experiment.PREPARED_FILES) == 9
    assert len(experiment.ARTIFACT_FILES) == 13
    assert len(experiment.COMPLETED_FILES) == 14
    assert experiment.verify(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()
    files, directories = experiment._tree_members(output)
    assert files == set(experiment.PREPARED_FILES)
    assert directories == experiment._expected_directories(experiment.PREPARED_FILES)
    parent_files, _ = experiment._tree_members(output / experiment.MM004_COPY_ROOT)
    assert parent_files == set(experiment.MM004_SELECTED)

    manifest = cast(dict[str, Any], experiment._read_json(output / experiment.INPUT_MANIFEST_FILE))
    assert manifest["source_count"] == 54
    assert len(manifest["parent"]["files"]) == 15
    assert manifest["input_validation"]["alignment"] == _FAKE_ALIGNMENT
    assert (output / experiment.PROTOCOL_COPY_FILE).stat().st_mode & 0o777 == 0o644
    assert (output / experiment.INPUT_MANIFEST_FILE).stat().st_mode & 0o777 == 0o644
    for relative in experiment.MM004_SELECTED:
        source_mode = (experiment.REPO_ROOT / experiment.MM004_ROOT / relative).stat().st_mode & 0o777
        copied_mode = (output / experiment.MM004_COPY_ROOT / relative).stat().st_mode & 0o777
        assert copied_mode == source_mode


def test_prepare_never_invokes_the_scientific_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = tmp_path / "MM-005"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch, live_parent_snapshot)

    def forbidden_execute(*args: object) -> object:
        del args
        raise AssertionError("preparation invoked an MM-005 fit")

    monkeypatch.setattr(experiment, "_execute", forbidden_execute)
    result = experiment.prepare(output)
    assert result["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()


def test_formal_marker_is_read_only_and_interruption_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, live_parent_snapshot)

    class ExpectedFailure(RuntimeError):
        pass

    def fail_after_marker(*args: object) -> object:
        del args
        marker = output / experiment.STARTED_FILE
        assert marker.is_file()
        assert marker.stat().st_mode & 0o777 == 0o444
        raise ExpectedFailure("fit failed after formal marker")

    monkeypatch.setattr(experiment, "_execute", fail_after_marker)
    with pytest.raises(experiment.InvalidMM005Package, match="after formal marker"):
        experiment.run(output)
    assert (output / experiment.STARTED_FILE).stat().st_mode & 0o777 == 0o444
    with pytest.raises(experiment.InvalidMM005Package, match="membership mismatch"):
        experiment.run(output)
    with pytest.raises(experiment.InvalidMM005Package, match="membership mismatch"):
        experiment.verify(output)


def test_fake_completed_fast_and_semantic_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, live_parent_snapshot)

    assert experiment.verify(output) == {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": "MM005_diagnostic_inconclusive",
        "artifact_count": len(experiment.ARTIFACT_FILES),
    }
    semantic = experiment.verify_semantic(output)
    assert semantic["outcomes"] == "verified_semantic_results"
    assert semantic["semantic_regeneration"] == "all 200 synthetic and real fits reproduced from copied arrays"
    assert (output / experiment.STARTED_FILE).stat().st_mode & 0o777 == 0o444
    assert (output / experiment.ARTIFACT_MANIFEST_FILE).stat().st_mode & 0o777 == 0o644
    marker = cast(dict[str, Any], experiment._read_json(output / experiment.STARTED_FILE))
    assert marker["alignment_sha256"] == experiment._canonical_json_sha256(_FAKE_ALIGNMENT)
    assert marker["mm004_receipt_sha256"] == {
        str(path): experiment.MM004_PINS[path] for path in experiment.MM004_SELECTED
    }


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation,message,completed,classification",
    [
        ("artifact_tamper", "artifact manifest", True, "invalid_MM005_package"),
        ("extra", "membership mismatch", False, "invalid_MM005_package"),
        ("symlink", "non-regular file", False, "invalid_MM005_package"),
        ("source_drift", "recompute", False, "invalid_MM005_package"),
        ("parent_copy_drift", "live sealed parent", False, "invalid_MM005_parent_alignment"),
        ("copied_mode", "live sealed parent", False, "invalid_MM005_parent_alignment"),
        ("marker_mode", "formal marker", True, "invalid_MM005_package"),
        ("manifest_mode", "generated file mode", True, "invalid_MM005_package"),
    ],
)
def test_integrity_defects_fail_closed(
    mutation: str,
    message: str,
    completed: bool,
    classification: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = (
        _complete(tmp_path, monkeypatch, live_parent_snapshot)
        if completed
        else _prepare(tmp_path, monkeypatch, live_parent_snapshot)
    )
    if mutation == "artifact_tamper":
        path = output / experiment.RESULT_FILE
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "extra":
        (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "symlink":
        copied = output / experiment.MM004_COPY_ROOT / experiment.MM004_SELECTED[1]
        copied.unlink()
        copied.symlink_to(experiment.REPO_ROOT / experiment.MM004_ROOT / experiment.MM004_SELECTED[1])
    elif mutation == "source_drift":
        source = experiment._source_hashes()
        monkeypatch.setattr(
            experiment,
            "_source_hashes",
            lambda: {**source, "synthetic/source-drift.py": "0" * 64},
        )
    elif mutation == "parent_copy_drift":
        copied = output / experiment.MM004_COPY_ROOT / experiment.MM004_SELECTED[-1]
        copied.write_bytes(copied.read_bytes() + b" ")
    elif mutation == "copied_mode":
        os.chmod(output / experiment.MM004_COPY_ROOT / experiment.FEATURE_RELATIVE, 0o600)
    elif mutation == "marker_mode":
        os.chmod(output / experiment.STARTED_FILE, 0o644)
    elif mutation == "manifest_mode":
        os.chmod(output / experiment.ARTIFACT_MANIFEST_FILE, 0o600)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(
        (experiment.InvalidMM005Package, experiment.InvalidMM005ParentAlignment),
        match=message,
    ) as invalid:
        experiment.verify(output)
    if isinstance(invalid.value, experiment.InvalidMM005Package):
        observed_classification = invalid.value.classification
    else:
        assert isinstance(invalid.value, experiment.InvalidMM005ParentAlignment)
        observed_classification = invalid.value.classification
    assert observed_classification == classification


def test_live_parent_pin_failure_is_classified_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "MM-005"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    monkeypatch.setattr(
        mm004_experiment,
        "verify",
        lambda path: {
            "status": "verified",
            "outcomes": "verified_results",
            "classification": experiment.PARENT_CLASSIFICATION,
        },
    )
    bad_pins = dict(experiment.MM004_PINS)
    bad_pins[Path("artifact-manifest.json")] = "0" * 64
    monkeypatch.setattr(experiment, "MM004_PINS", bad_pins)

    with pytest.raises(experiment.InvalidMM005ParentAlignment, match="pinned MM-004") as invalid:
        experiment.prepare(output)
    assert invalid.value.classification == "invalid_MM005_parent_alignment"
    assert not (output / experiment.STARTED_FILE).exists()


def test_rehashed_evidence_tamper_reaches_strict_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, live_parent_snapshot)
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

    with pytest.raises(experiment.InvalidMM005Package, match="fake evidence differs"):
        experiment.verify(output)


def test_rehashed_provenance_tamper_reaches_no_fit_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_parent_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, live_parent_snapshot)
    evidence_path = output / experiment.EVIDENCE_FILE
    evidence = cast(dict[str, Any], experiment._read_json(evidence_path))
    cast(dict[str, object], evidence["alignment"])["identity_sha256"] = "0" * 64
    evidence_path.write_bytes(experiment._json_bytes(evidence))
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_json_exclusive(
        output / experiment.ARTIFACT_MANIFEST_FILE,
        experiment._artifact_manifest(output),
        0o644,
    )
    monkeypatch.setattr(method, "validate_evidence", lambda value: cast(dict[str, object], value))

    with pytest.raises(experiment.InvalidMM005Package, match="fake evidence alignment differs"):
        experiment.verify(output)


def test_canonical_path_parent_overlap_and_exclusive_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "canonical" / "MM-005"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", expected)
    with pytest.raises(experiment.InvalidMM005Package, match="canonical path"):
        experiment.verify(tmp_path / "other")

    parent = experiment.REPO_ROOT / experiment.MM004_ROOT
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", parent)
    with pytest.raises(ValueError, match="overlaps a protected parent"):
        experiment._assert_expected_output(parent)

    path = tmp_path / "already-created.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment._write_json_exclusive(path, {"replacement": True})
    assert path.read_text(encoding="utf-8") == "existing"


def test_lifecycle_source_has_no_media_or_model_execution_hooks() -> None:
    source = (experiment.REPO_ROOT / "bench/multimodal_horizon_diagnostics/experiment.py").read_text(encoding="utf-8")
    assert "decode_video_frames" not in source
    assert "validate_media_hashes" not in source
    assert "_extract_pixel_arrays" not in source
    assert "torch" not in source


def test_nested_creation_and_exclusive_write_fsync_file_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: calls.append(descriptor))
    directory = tmp_path / "nested" / "receipt"
    experiment._mkdir_fsynced(directory)
    experiment._write_bytes_exclusive(directory / "artifact.bin", b"sealed", 0o644)

    assert len(calls) == 4
    assert (directory / "artifact.bin").read_bytes() == b"sealed"
