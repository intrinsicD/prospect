"""Lifecycle and preservation tests for MM-003."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from bench.multimodal_diagnostics import experiment as mm002_experiment
from bench.multimodal_preflight import experiment as mm001_experiment
from bench.multimodal_transform_diagnostics import experiment, method

_FAKE_EVIDENCE: dict[str, object] = {"schema_version": "mm003-test-evidence-v1", "rows": []}
_FAKE_SUMMARY: dict[str, object] = {
    "schema_version": method.SCHEMA_VERSION,
    "experiment_id": "MM-003",
    "decision": {
        "classification": "tested_projection_scale_factors_not_supported",
        "recommended_fix": "do not adopt a transform",
    },
}


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "MM-003"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    result = experiment.prepare(output)
    assert result == {
        "status": "prepared_only",
        "outcomes": "prepared_only",
        "classification": "no_outcomes_before_formal_marker",
        "artifact_count": len(experiment.PREPARED_FILES),
    }
    return output


def _patch_fake_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_execute(
        raw: method.VisualTable,
        projection: Any,
        parent_evidence: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, object]]:
        del parent_evidence
        _, records, arrays = method.fit_all_transforms(raw, projection)
        return arrays, records, deepcopy(_FAKE_EVIDENCE)

    def validate_evidence(value: object) -> dict[str, object]:
        if value != _FAKE_EVIDENCE:
            raise ValueError("fake evidence differs")
        return deepcopy(_FAKE_EVIDENCE)

    monkeypatch.setattr(method, "execute", fake_execute)
    monkeypatch.setattr(method, "validate_evidence", validate_evidence)
    monkeypatch.setattr(method, "assert_parent_parity", lambda evidence, parent: {"passed": True})
    monkeypatch.setattr(
        method,
        "summarize",
        lambda evidence, records, parent: deepcopy(_FAKE_SUMMARY),
    )
    monkeypatch.setattr(
        method,
        "report_text",
        lambda summary: json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
    )


def _complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = _prepare(tmp_path, monkeypatch)
    _patch_fake_analysis(monkeypatch)
    result = experiment.run(output)
    assert result["status"] == "completed"
    return output


def test_parents_still_verify_and_do_not_bind_mm003_sources() -> None:
    assert mm001_experiment.verify(mm001_experiment.DEFAULT_OUTPUT)["outcomes"] == "verified_results"
    assert mm002_experiment.verify(mm002_experiment.DEFAULT_OUTPUT)["outcomes"] == "verified_results"
    mm002_paths = {str(path) for path in mm002_experiment._source_paths()}
    assert not any(path.startswith("bench/multimodal_transform_diagnostics/") for path in mm002_paths)
    assert not any(Path(path).name.startswith("test_mm003_") for path in mm002_paths)
    assert mm002_paths < {str(path) for path in experiment._source_paths()}


def test_prepare_copies_only_selected_receipts_without_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path, monkeypatch)

    assert experiment.verify(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()
    assert not (output / experiment.EVIDENCE_FILE).exists()
    assert {path.name for path in (output / experiment.MM001_COPY_ROOT).iterdir()} == {
        path.name for path in experiment.MM001_SELECTED
    }
    assert {path.name for path in (output / experiment.MM002_COPY_ROOT).iterdir()} == {
        path.name for path in experiment.MM002_SELECTED
    }


def test_formal_marker_is_atomic_read_only_and_interruption_is_terminal(
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
        assert marker.stat().st_mode & 0o222 == 0
        raise ExpectedFailure("training failed after formal marker")

    monkeypatch.setattr(method, "execute", fail_after_marker)
    with pytest.raises(experiment.InvalidMM003Package, match="after formal marker"):
        experiment.run(output)
    assert (output / experiment.STARTED_FILE).stat().st_mode & 0o222 == 0
    with pytest.raises(experiment.InvalidMM003Package, match="membership mismatch"):
        experiment.run(output)
    with pytest.raises(experiment.InvalidMM003Package, match="membership mismatch"):
        experiment.verify(output)


def test_fake_completed_roundtrip_and_semantic_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete(tmp_path, monkeypatch)

    assert experiment.verify(output) == {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": "tested_projection_scale_factors_not_supported",
        "artifact_count": len(experiment.ARTIFACT_FILES),
    }
    assert experiment.verify_semantic(output)["outcomes"] == "verified_semantic_results"


@pytest.mark.parametrize(
    "mutation",
    ["artifact_tamper", "extra", "symlink", "source_drift", "parent_copy_drift"],
)
def test_integrity_defects_fail_closed(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if mutation == "artifact_tamper":
        output = _complete(tmp_path, monkeypatch)
        result_path = output / experiment.RESULT_FILE
        result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        message = "artifact manifest"
    else:
        output = _prepare(tmp_path, monkeypatch)
        if mutation == "extra":
            (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            message = "membership mismatch"
        elif mutation == "symlink":
            copied = output / experiment.MM001_COPY_ROOT / experiment.MM001_SELECTED[1]
            copied.unlink()
            copied.symlink_to(experiment.REPO_ROOT / experiment.MM001_ROOT / experiment.MM001_SELECTED[1])
            message = "non-regular file"
        elif mutation == "source_drift":
            original = experiment._source_hashes()
            monkeypatch.setattr(
                experiment,
                "_source_hashes",
                lambda: {**original, "synthetic/source-drift.py": "0" * 64},
            )
            message = "recompute"
        elif mutation == "parent_copy_drift":
            copied = output / experiment.MM002_COPY_ROOT / experiment.MM002_SELECTED[-1]
            os.chmod(copied, 0o644)
            copied.write_bytes(copied.read_bytes() + b" ")
            message = "differs from live sealed parent"
        else:  # pragma: no cover
            raise AssertionError(mutation)

    with pytest.raises(experiment.InvalidMM003Package, match=message) as invalid:
        experiment.verify(output)
    assert invalid.value.classification == "invalid_MM003_package"


def test_rehashed_transform_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _complete(tmp_path, monkeypatch)
    records_path = output / experiment.TRANSFORM_RECORDS_FILE
    records = cast(list[dict[str, object]], experiment._read_json(records_path))
    records[0]["fit_scope"] = "tampered"
    records_path.write_bytes(experiment._json_bytes(records))
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_json_exclusive(
        output / experiment.ARTIFACT_MANIFEST_FILE,
        experiment._artifact_manifest(output),
    )

    with pytest.raises(experiment.InvalidMM003Package, match="transform records"):
        experiment.verify(output)


def test_parent_parity_failure_has_dedicated_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        method,
        "assert_parent_parity",
        lambda evidence, parent: (_ for _ in ()).throw(ValueError("parity differs")),
    )

    with pytest.raises(experiment.InvalidMM003ParentParity, match="parity differs") as invalid:
        experiment._assert_parent_parity({}, {})
    assert invalid.value.classification == "invalid_MM003_parent_parity"
