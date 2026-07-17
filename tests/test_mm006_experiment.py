"""Lifecycle, provenance, and fail-closed tests for MM-006."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from bench.multimodal_horizon_diagnostics import experiment as mm005_experiment
from bench.multimodal_warp_diagnostics import experiment, method

_FAKE_ALIGNMENT: dict[str, object] = {"rows": 453, "identity_sha256": "a" * 64}
_FAKE_EVIDENCE: dict[str, object] = {
    "schema_version": "mm006-test-evidence-v1",
    "alignment": _FAKE_ALIGNMENT,
    "normalizer_rows": [],
    "synthetic_panel_rows": [],
    "synthetic_metric_rows": [],
    "real_metric_rows": [],
    "activity_rows": [],
}
_FAKE_SUMMARY: dict[str, object] = {
    "schema_version": "mm006-test-summary-v1",
    "experiment_id": "MM-006",
    "decision": {
        "classification": "MM006_diagnostic_inconclusive",
        "recommended_next_step": "replicate",
    },
}


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def parent_snapshot() -> dict[str, object]:
    root = experiment.REPO_ROOT / experiment.MM005_ROOT
    replay = experiment._replay_parent(root)
    return {
        "experiment_id": "MM-005",
        "classification": experiment.PARENT_CLASSIFICATION,
        "verification": {
            "status": "verified",
            "outcomes": "verified_results",
            "classification": experiment.PARENT_CLASSIFICATION,
            "artifact_count": len(mm005_experiment.ARTIFACT_FILES),
        },
        "live_path": str(experiment.MM005_ROOT),
        "copy_path": str(experiment.MM005_COPY_ROOT),
        "files": experiment._records(root, mm005_experiment.COMPLETED_FILES),
        "selected_files": [str(path) for path in experiment.MM005_SELECTED],
        "pinned": {str(path): digest for path, digest in experiment.MM005_PINS.items()},
        "replay": replay,
        "scientific_relationship": (
            "outcome-informed direct child reusing the same eight videos; not independent evidence"
        ),
    }


def _patch_fake_lifecycle(monkeypatch: pytest.MonkeyPatch, snapshot: dict[str, object]) -> None:
    validation: dict[str, object] = {
        "alignment": deepcopy(_FAKE_ALIGNMENT),
        "alignment_sha256": experiment._canonical_json_sha256(_FAKE_ALIGNMENT),
    }
    monkeypatch.setattr(experiment, "_verify_live_parent", lambda: deepcopy(snapshot))
    monkeypatch.setattr(
        experiment,
        "_load_analysis_inputs",
        lambda output: ({"domain": "taesd"}, {"domain": "pixel"}, deepcopy(validation)),
    )
    monkeypatch.setattr(method, "execute", lambda taesd, pixel: deepcopy(_FAKE_EVIDENCE))

    def validate(value: object) -> dict[str, object]:
        if value != _FAKE_EVIDENCE:
            raise ValueError("fake evidence differs")
        return deepcopy(_FAKE_EVIDENCE)

    monkeypatch.setattr(method, "validate_evidence", validate)
    monkeypatch.setattr(method, "summarize", lambda evidence: deepcopy(_FAKE_SUMMARY))
    monkeypatch.setattr(
        method,
        "report_text",
        lambda summary: json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
    )
    monkeypatch.setattr(method, "config_record", lambda: {"schema_version": "mm006-test-config-v1"})

    def provenance(taesd: object, pixel: object, evidence: Mapping[str, object]) -> None:
        del taesd, pixel
        if evidence.get("alignment") != _FAKE_ALIGNMENT:
            raise ValueError("fake evidence alignment differs")

    monkeypatch.setattr(experiment, "_validate_evidence_provenance", provenance)


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> Path:
    output = tmp_path / "MM-006"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch, snapshot)
    result = experiment.prepare(output)
    assert result["outcomes"] == "prepared_only"
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


def test_exact_source_superset_and_parent_pins() -> None:
    parent_sources = set(mm005_experiment._source_paths())
    sources = set(experiment._source_paths())
    assert len(parent_sources) == 54
    assert len(sources) == 61
    assert parent_sources < sources
    assert {str(path) for path in sources - parent_sources} == {
        "bench/multimodal_warp_diagnostics/__init__.py",
        "bench/multimodal_warp_diagnostics/__main__.py",
        "bench/multimodal_warp_diagnostics/experiment.py",
        "bench/multimodal_warp_diagnostics/method.py",
        "docs/research/2026-07-15-mm006-causal-warp-ceiling-protocol.md",
        "tests/test_mm006_experiment.py",
        "tests/test_mm006_method.py",
    }
    root = experiment.REPO_ROOT / experiment.MM005_ROOT
    assert (
        experiment._file_hash(root / Path("artifact-manifest.json"))
        == experiment.MM005_PINS[Path("artifact-manifest.json")]
    )
    for relative in experiment.MM005_SELECTED:
        assert experiment._file_hash(root / relative) == experiment.MM005_PINS[relative]


def test_prepare_has_receipts_and_no_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, parent_snapshot)
    assert len(experiment.PREPARED_FILES) == 9
    assert len(experiment.ARTIFACT_FILES) == 13
    assert len(experiment.COMPLETED_FILES) == 14
    assert experiment.verify(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()
    files, directories = experiment._tree_members(output)
    assert files == set(experiment.PREPARED_FILES)
    assert directories == experiment._expected_directories(experiment.PREPARED_FILES)
    manifest = cast(dict[str, Any], experiment._read_json(output / experiment.INPUT_MANIFEST_FILE))
    assert manifest["source_count"] == 61
    assert manifest["input_validation"]["alignment"] == _FAKE_ALIGNMENT


def test_prepare_never_calls_scientific_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = tmp_path / "MM-006"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    _patch_fake_lifecycle(monkeypatch, parent_snapshot)

    def forbidden(*args: object) -> object:
        del args
        raise AssertionError("prepare called warp search")

    monkeypatch.setattr(method, "execute", forbidden)
    assert experiment.prepare(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()


def test_interruption_after_marker_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = _prepare(tmp_path, monkeypatch, parent_snapshot)

    def fail(*args: object) -> object:
        del args
        assert (output / experiment.STARTED_FILE).stat().st_mode & 0o777 == 0o444
        raise RuntimeError("warp failed after marker")

    monkeypatch.setattr(method, "execute", fail)
    with pytest.raises(experiment.InvalidMM006Package, match="after marker"):
        experiment.run(output)
    with pytest.raises(experiment.InvalidMM006Package, match="membership mismatch"):
        experiment.run(output)


def test_fake_completed_fast_and_semantic_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, parent_snapshot)
    assert experiment.verify(output) == {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": "MM006_diagnostic_inconclusive",
        "artifact_count": len(experiment.ARTIFACT_FILES),
    }
    semantic = experiment.verify_semantic(output)
    assert semantic["outcomes"] == "verified_semantic_results"
    assert "causal/oracle searches" in str(semantic["semantic_regeneration"])
    marker = cast(dict[str, Any], experiment._read_json(output / experiment.STARTED_FILE))
    assert marker["alignment_sha256"] == experiment._canonical_json_sha256(_FAKE_ALIGNMENT)
    assert marker["mm005_receipt_sha256"] == {
        str(path): experiment.MM005_PINS[path] for path in experiment.MM005_SELECTED
    }


def test_semantic_verifier_rejects_regenerated_evidence_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, parent_snapshot)
    regenerated = deepcopy(_FAKE_EVIDENCE)
    regenerated["alignment"] = {"rows": 453, "identity_sha256": "b" * 64}
    monkeypatch.setattr(method, "validate_evidence", lambda value: cast(dict[str, object], value))
    monkeypatch.setattr(method, "execute", lambda taesd, pixel: deepcopy(regenerated))
    with pytest.raises(experiment.InvalidMM006Package, match=r"semantic evidence\.alignment\.identity_sha256 differs"):
        experiment.verify_semantic(output)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation,message,completed,classification",
    [
        ("artifact", "artifact manifest", True, "invalid_MM006_package"),
        ("extra", "membership mismatch", False, "invalid_MM006_package"),
        ("symlink", "non-regular file", False, "invalid_MM006_package"),
        ("source", "recompute", False, "invalid_MM006_package"),
        ("parent", "live parent", False, "invalid_MM006_parent_alignment"),
        ("marker_mode", "formal marker", True, "invalid_MM006_package"),
    ],
)
def test_integrity_defects_fail_closed(
    mutation: str,
    message: str,
    completed: bool,
    classification: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = (
        _complete(tmp_path, monkeypatch, parent_snapshot)
        if completed
        else _prepare(tmp_path, monkeypatch, parent_snapshot)
    )
    if mutation == "artifact":
        path = output / experiment.RESULT_FILE
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "extra":
        (output / "unexpected.txt").write_text("x", encoding="utf-8")
    elif mutation == "symlink":
        copied = output / experiment.MM005_COPY_ROOT / experiment.MM005_SELECTED[1]
        copied.unlink()
        copied.symlink_to(experiment.REPO_ROOT / experiment.MM005_ROOT / experiment.MM005_SELECTED[1])
    elif mutation == "source":
        source = experiment._source_hashes()
        monkeypatch.setattr(experiment, "_source_hashes", lambda: {**source, "drift.py": "0" * 64})
    elif mutation == "parent":
        copied = output / experiment.MM005_COPY_ROOT / experiment.MM005_SELECTED[-1]
        copied.write_bytes(copied.read_bytes() + b" ")
    elif mutation == "marker_mode":
        os.chmod(output / experiment.STARTED_FILE, 0o644)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(
        (experiment.InvalidMM006Package, experiment.InvalidMM006ParentAlignment),
        match=message,
    ) as invalid:
        experiment.verify(output)
    error = cast(
        experiment.InvalidMM006Package | experiment.InvalidMM006ParentAlignment,
        invalid.value,
    )
    assert error.classification == classification


def test_rehashed_evidence_and_provenance_tamper_reach_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_snapshot: dict[str, object],
) -> None:
    output = _complete(tmp_path, monkeypatch, parent_snapshot)
    evidence_path = output / experiment.EVIDENCE_FILE
    evidence = cast(dict[str, Any], experiment._read_json(evidence_path))
    evidence["alignment"] = {"rows": 453, "identity_sha256": "0" * 64}
    evidence_path.write_bytes(experiment._json_bytes(evidence))
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_json_exclusive(
        output / experiment.ARTIFACT_MANIFEST_FILE,
        experiment._artifact_manifest(output),
        0o644,
    )
    monkeypatch.setattr(method, "validate_evidence", lambda value: cast(dict[str, object], value))
    with pytest.raises(experiment.InvalidMM006Package, match="fake evidence alignment"):
        experiment.verify(output)


def test_canonical_path_overlap_and_exclusive_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = tmp_path / "canonical" / "MM-006"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", expected)
    with pytest.raises(experiment.InvalidMM006Package, match="canonical path"):
        experiment.verify(tmp_path / "other")
    parent = experiment.REPO_ROOT / experiment.MM005_ROOT
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", parent)
    with pytest.raises(ValueError, match="overlaps a protected parent"):
        experiment._assert_expected_output(parent)
    path = tmp_path / "created.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment._write_json_exclusive(path, {"replacement": True})


def test_lifecycle_has_no_media_model_or_opencv_hooks() -> None:
    source = (experiment.REPO_ROOT / "bench/multimodal_warp_diagnostics/experiment.py").read_text(encoding="utf-8")
    assert "decode_video_frames" not in source
    assert "torch" not in source
    assert "cv2" not in source
