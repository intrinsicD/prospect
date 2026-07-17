"""Lifecycle and preservation tests for the feature-only MM-002 diagnostic."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from bench.multimodal_diagnostics import experiment
from bench.multimodal_preflight import experiment as parent_experiment

_FAKE_EVIDENCE: dict[str, object] = {
    "schema_version": "mm002-test-evidence-v1",
    "world_rows": [],
    "codec_rows": [],
}
_FAKE_SUMMARY: dict[str, object] = {
    "decision": {"classification": "tested_factors_not_supported"},
    "world": {"test_double": True},
    "codec": {"test_double": True},
}


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "MM-002"
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
    def validate_evidence(value: object) -> dict[str, object]:
        if value != _FAKE_EVIDENCE:
            raise ValueError("fake evidence differs")
        return deepcopy(_FAKE_EVIDENCE)

    def summarize(
        evidence: Mapping[str, object], parent_rows: list[dict[str, Any]]
    ) -> dict[str, object]:
        assert evidence == _FAKE_EVIDENCE
        assert parent_rows
        return deepcopy(_FAKE_SUMMARY)

    monkeypatch.setattr(experiment.method, "execute", lambda table: deepcopy(_FAKE_EVIDENCE))
    monkeypatch.setattr(experiment.method, "validate_evidence", validate_evidence)
    monkeypatch.setattr(experiment.method, "assert_parent_parity", lambda evidence, rows: None)
    monkeypatch.setattr(experiment.method, "summarize", summarize)
    monkeypatch.setattr(
        experiment.method,
        "report_text",
        lambda results: json.dumps(results, sort_keys=True, allow_nan=False) + "\n",
    )


def _complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = _prepare(tmp_path, monkeypatch)
    _patch_fake_analysis(monkeypatch)
    result = experiment.run(output)
    assert result["status"] == "completed"
    assert cast(dict[str, object], result["summary"])["decision"] == _FAKE_SUMMARY["decision"]
    return output


def test_mm001_still_verifies_and_does_not_bind_mm002_sources() -> None:
    verification = parent_experiment.verify(parent_experiment.DEFAULT_OUTPUT)

    assert verification["outcomes"] == "verified_results"
    parent_paths = {str(path) for path in parent_experiment._source_paths()}
    assert not any(path.startswith("bench/multimodal_diagnostics/") for path in parent_paths)
    assert not any(Path(path).name.startswith("test_mm002_") for path in parent_paths)
    assert parent_paths < {str(path) for path in experiment._source_paths()}


def test_prepare_copies_and_verifies_parent_without_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _prepare(tmp_path, monkeypatch)

    assert experiment.verify(output)["outcomes"] == "prepared_only"
    assert not (output / experiment.STARTED_FILE).exists()
    assert not (output / experiment.EVIDENCE_FILE).exists()
    assert (output / experiment.PROTOCOL_COPY_FILE).read_bytes() == (
        experiment.REPO_ROOT / experiment.PROTOCOL_DOC
    ).read_bytes()
    copied = output / experiment.PARENT_COPY_ROOT
    live = experiment.REPO_ROOT / experiment.PARENT_OUTPUT
    assert {
        relative: (copied / relative).read_bytes()
        for relative in experiment.PARENT_PACKAGE_FILES
    } == {
        relative: (live / relative).read_bytes()
        for relative in experiment.PARENT_PACKAGE_FILES
    }


def test_formal_marker_is_atomic_read_only_and_interrupted_run_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _prepare(tmp_path, monkeypatch)

    class ExpectedFailure(RuntimeError):
        pass

    def fail_after_marker(table: object) -> dict[str, object]:
        del table
        marker = output / experiment.STARTED_FILE
        assert marker.is_file()
        assert marker.stat().st_mode & 0o222 == 0
        raise ExpectedFailure("training failed after formal marker")

    monkeypatch.setattr(experiment.method, "execute", fail_after_marker)
    with pytest.raises(experiment.InvalidMM002Package, match="after formal marker"):
        experiment.run(output)

    marker = output / experiment.STARTED_FILE
    assert marker.is_file()
    assert marker.stat().st_mode & 0o222 == 0
    with pytest.raises(experiment.InvalidMM002Package, match="membership mismatch"):
        experiment.run(output)
    with pytest.raises(experiment.InvalidMM002Package, match="membership mismatch"):
        experiment.verify(output)


def test_fake_completed_roundtrip_and_semantic_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _complete(tmp_path, monkeypatch)

    verification = experiment.verify(output)
    assert verification == {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": "tested_factors_not_supported",
        "artifact_count": len(experiment.ARTIFACT_FILES),
    }
    marker = output / experiment.STARTED_FILE
    assert marker.stat().st_mode & 0o222 == 0
    assert experiment.verify_semantic(output)["outcomes"] == "verified_semantic_results"


@pytest.mark.parametrize(
    "mutation",
    ["artifact_tamper", "extra", "symlink", "source_drift", "parent_copy_drift"],
)
def test_completed_or_prepared_integrity_defects_fail_closed(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mutation == "artifact_tamper":
        output = _complete(tmp_path, monkeypatch)
        result_path = output / experiment.RESULT_FILE
        result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        message = "artifact manifest"
    else:
        output = _prepare(tmp_path, monkeypatch)
        message = ""
        if mutation == "extra":
            (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            message = "membership mismatch"
        elif mutation == "symlink":
            copied = output / experiment.PARENT_COPY_ROOT / experiment.PARENT_PACKAGE_FILES[0]
            copied.unlink()
            copied.symlink_to(
                experiment.REPO_ROOT / experiment.PARENT_OUTPUT / experiment.PARENT_PACKAGE_FILES[0]
            )
            message = "non-regular file"
        elif mutation == "source_drift":
            original = experiment._source_hashes()
            monkeypatch.setattr(
                experiment,
                "_source_hashes",
                lambda: {**original, "synthetic/source-drift.py": "0" * 64},
            )
            message = "does not recompute"
        elif mutation == "parent_copy_drift":
            copied = output / experiment.PARENT_COPY_ROOT / parent_experiment.RESULT_FILE
            os.chmod(copied, 0o644)
            copied.write_bytes(copied.read_bytes() + b" ")
            message = "differs from the live sealed parent"
        else:  # pragma: no cover - exhaustive guard for future parametrization edits
            raise AssertionError(mutation)

    with pytest.raises(experiment.InvalidMM002Package, match=message) as invalid:
        experiment.verify(output)
    assert invalid.value.classification == "invalid_MM002_package"


def test_rehashed_evidence_tamper_is_rejected_by_semantic_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _complete(tmp_path, monkeypatch)
    evidence_path = output / experiment.EVIDENCE_FILE
    tampered = deepcopy(_FAKE_EVIDENCE)
    tampered["unexpected"] = True
    experiment._write_json(evidence_path, tampered)
    (output / experiment.ARTIFACT_MANIFEST_FILE).unlink()
    experiment._write_artifact_manifest(output)

    with pytest.raises(experiment.InvalidMM002Package, match="fake evidence differs"):
        experiment.verify(output)
