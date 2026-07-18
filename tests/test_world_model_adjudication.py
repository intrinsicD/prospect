from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle.adjudication import (
    ADJUDICATION_MANIFEST_NAME,
    AUDITOR_SOURCE_PATH,
    COPIED_AUDIT_NAME,
    COPIED_SEMANTIC_REVIEW_NAME,
    AdjudicationError,
    create_adjudication_package,
    main,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _write_producer_manifest(root: Path, *, lane: str) -> bytes:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _digest(path.read_bytes()),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "producer-manifest.json"
    ]
    return _write_json(
        root / "producer-manifest.json",
        {
            "schema": "prospect.wm001.producer-manifest.v1",
            "experiment_id": "WM-001",
            "lane": lane,
            "status": "completed",
            "started_at_utc": "2026-07-17T00:00:00Z",
            "completed_at_utc": "2026-07-17T01:00:00Z",
            "error": None,
            "manifest_excludes": ["producer-manifest.json"],
            "file_count": len(rows),
            "files": rows,
        },
    )


def _make_evidence(
    tmp_path: Path,
    *,
    lane: str = "formal",
    gates_pass: bool = True,
    binding_mismatch: bool = False,
    auditor_snapshot_mismatch: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    producer = tmp_path / "producer"
    producer.mkdir()

    formal_binding_sha256: str | None = None
    if lane == "formal":
        binding_payload = _write_json(
            producer / "formal-binding.json",
            {
                "schema": "prospect.world-model-lifecycle.formal-binding.v3",
                "experiment_id": "WM-001",
            },
        )
        formal_binding_sha256 = _digest(binding_payload)
        snapshot = producer / "source" / "bench" / "world_model_lifecycle" / "artifact_audit.py"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(b"different auditor\n" if auditor_snapshot_mismatch else AUDITOR_SOURCE_PATH.read_bytes())

    result = {
        "schema": "prospect.world-model-lifecycle.raw-result.v3",
        "experiment_id": "WM-001",
        "lane": lane,
        "claim_eligible": lane == "formal",
        "formal_binding_sha256": ("0" * 64 if binding_mismatch else formal_binding_sha256),
        "gate_results": [
            {
                "gate": f"K{index}",
                "passed": gates_pass or index < 7,
            }
            for index in range(8)
        ],
    }
    result_payload = _write_json(producer / "result.json", result)
    producer_manifest_payload = _write_producer_manifest(producer, lane=lane)

    audit = {
        "schema": "prospect.world-model-lifecycle.artifact-audit.v1",
        "artifact_root": str(producer.resolve()),
        "result_file": "result.json",
        "result_sha256": _digest(result_payload),
        "integrity_passed": True,
        "complete_for_claim": True,
        "passed": True,
        "check_counts": {
            "passed": 100,
            "failed": 0,
            "coverage_gaps": 0,
        },
        "custody": {
            "producer_manifest_checked": True,
            "producer_manifest_status": "completed",
            "producer_manifest_sha256": _digest(producer_manifest_payload),
        },
        "findings": [],
        "coverage_gaps": [],
        "independence_limitations": [],
    }
    audit_path = tmp_path / "audit.json"
    _write_json(audit_path, audit)
    return producer, audit_path, audit


def _make_semantic_review(
    tmp_path: Path,
    *,
    producer: Path,
    audit_path: Path,
    verdict: str,
    fatal_findings: list[object] | None = None,
) -> Path:
    review_path = tmp_path / f"semantic-review-{verdict}.json"
    _write_json(
        review_path,
        {
            "schema": "prospect.wm001.semantic-review.v1",
            "artifact_root": str(producer.resolve()),
            "result_sha256": _digest((producer / "result.json").read_bytes()),
            "independent_audit_sha256": _digest(audit_path.read_bytes()),
            "reviewer": "independent-semantic-review-fixture",
            "reviewed_gates": [f"K{index}" for index in range(8)],
            "verdict": verdict,
            "fatal_findings": fatal_findings or [],
            "conclusion": f"fixture semantic verdict: {verdict}",
        },
    )
    return review_path


def test_formal_acceptance_creates_canonical_external_package(tmp_path: Path) -> None:
    producer, audit_path, _ = _make_evidence(tmp_path)
    review_path = _make_semantic_review(
        tmp_path,
        producer=producer,
        audit_path=audit_path,
        verdict="accepted",
    )
    output = tmp_path / "adjudication"
    audit_payload = audit_path.read_bytes()
    review_payload = review_path.read_bytes()

    manifest = create_adjudication_package(
        producer_root=producer,
        audit_report=audit_path,
        output_directory=output,
        disposition="accepted",
        semantic_review=review_path,
    )

    stored_manifest = (output / ADJUDICATION_MANIFEST_NAME).read_bytes()
    assert stored_manifest == _canonical(manifest)
    assert (output / COPIED_AUDIT_NAME).read_bytes() == audit_payload
    assert manifest["schema"] == "prospect.wm001.adjudication-package.v3"
    assert manifest["lane"] == "formal"
    assert manifest["disposition"] == "accepted"
    assert manifest["audit_clean_for_claim"] is True
    assert manifest["producer_manifest_sha256"] == _digest((producer / "producer-manifest.json").read_bytes())
    assert manifest["result_sha256"] == _digest((producer / "result.json").read_bytes())
    assert manifest["audit_sha256"] == _digest(audit_payload)
    assert manifest["auditor_source_sha256"] == _digest(AUDITOR_SOURCE_PATH.read_bytes())
    assert manifest["semantic_review_sha256"] == _digest(review_payload)
    assert manifest["formal_binding_sha256"] == _digest((producer / "formal-binding.json").read_bytes())
    assert (output / COPIED_SEMANTIC_REVIEW_NAME).read_bytes() == review_payload


def test_formal_acceptance_requires_clean_semantic_review(tmp_path: Path) -> None:
    producer, audit_path, _ = _make_evidence(tmp_path)
    with pytest.raises(AdjudicationError, match="semantic review"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "missing-review",
            disposition="accepted",
        )

    review_path = _make_semantic_review(
        tmp_path,
        producer=producer,
        audit_path=audit_path,
        verdict="accepted",
        fatal_findings=[{"severity": "fatal"}],
    )
    with pytest.raises(AdjudicationError, match="fatal findings"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            semantic_review=review_path,
            output_directory=tmp_path / "fatal-review",
            disposition="accepted",
        )


def test_development_can_be_pending_but_cannot_be_accepted(tmp_path: Path) -> None:
    producer, audit_path, _ = _make_evidence(tmp_path, lane="development")
    pending = tmp_path / "pending"

    manifest = create_adjudication_package(
        producer_root=producer,
        audit_report=audit_path,
        output_directory=pending,
        disposition="pending",
    )

    assert manifest["formal_binding_file"] is None
    assert manifest["formal_binding_sha256"] is None
    with pytest.raises(AdjudicationError, match="development evidence"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "accepted",
            disposition="accepted",
        )
    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize("failure", ["integrity", "completeness"])
@pytest.mark.parametrize("disposition", ["pending", "accepted"])
def test_non_rejected_dispositions_require_a_clean_audit(
    tmp_path: Path,
    failure: str,
    disposition: str,
) -> None:
    producer, audit_path, audit = _make_evidence(tmp_path)
    if failure == "integrity":
        audit["integrity_passed"] = False
        audit["passed"] = False
        audit["check_counts"]["failed"] = 1
        audit["findings"] = [{"severity": "error", "code": "tampered"}]
    else:
        audit["complete_for_claim"] = False
        audit["passed"] = False
        audit["check_counts"]["coverage_gaps"] = 1
        audit["coverage_gaps"] = [{"severity": "blocker", "code": "missing"}]
    _write_json(audit_path, audit)

    review_path = _make_semantic_review(
        tmp_path,
        producer=producer,
        audit_path=audit_path,
        verdict=disposition,
    )
    with pytest.raises(AdjudicationError, match="requires an independent audit"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "package",
            disposition=disposition,
            semantic_review=review_path if disposition == "accepted" else None,
        )
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("failure", ["integrity", "completeness"])
def test_rejected_package_can_preserve_non_clean_audit_with_fatal_review(
    tmp_path: Path,
    failure: str,
) -> None:
    producer, audit_path, audit = _make_evidence(tmp_path)
    if failure == "integrity":
        audit["integrity_passed"] = False
        audit["passed"] = False
        audit["check_counts"]["failed"] = 1
        audit["findings"] = [{"severity": "error", "code": "tampered"}]
    else:
        audit["complete_for_claim"] = False
        audit["passed"] = False
        audit["check_counts"]["coverage_gaps"] = 1
        audit["coverage_gaps"] = [{"severity": "blocker", "code": "missing"}]
    audit_payload = _write_json(audit_path, audit)
    review_path = _make_semantic_review(
        tmp_path,
        producer=producer,
        audit_path=audit_path,
        verdict="rejected",
        fatal_findings=[{"severity": "fatal", "code": f"audit_{failure}"}],
    )
    output = tmp_path / "package"

    manifest = create_adjudication_package(
        producer_root=producer,
        audit_report=audit_path,
        output_directory=output,
        disposition="rejected",
        semantic_review=review_path,
    )

    assert manifest["schema"] == "prospect.wm001.adjudication-package.v3"
    assert manifest["disposition"] == "rejected"
    assert manifest["audit_clean_for_claim"] is False
    assert (output / COPIED_AUDIT_NAME).read_bytes() == audit_payload
    assert (output / COPIED_SEMANTIC_REVIEW_NAME).read_bytes() == review_path.read_bytes()


def test_rejected_non_clean_audit_requires_fatal_semantic_finding(tmp_path: Path) -> None:
    producer, audit_path, audit = _make_evidence(tmp_path)
    audit["integrity_passed"] = False
    audit["passed"] = False
    audit["check_counts"]["failed"] = 1
    audit["findings"] = [{"severity": "error", "code": "failed_check"}]
    _write_json(audit_path, audit)
    review_path = _make_semantic_review(
        tmp_path,
        producer=producer,
        audit_path=audit_path,
        verdict="rejected",
    )

    with pytest.raises(AdjudicationError, match="at least one fatal semantic finding"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "package",
            disposition="rejected",
            semantic_review=review_path,
        )
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failed_count", 2),
        ("gap_count", 2),
        ("integrity_passed", True),
        ("complete_for_claim", True),
        ("passed", True),
    ],
)
def test_rejected_package_refuses_internally_inconsistent_audit_status(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    producer, audit_path, audit = _make_evidence(tmp_path)
    audit["integrity_passed"] = False
    audit["complete_for_claim"] = False
    audit["passed"] = False
    audit["check_counts"]["failed"] = 1
    audit["check_counts"]["coverage_gaps"] = 1
    audit["findings"] = [{"severity": "error", "code": "failed_check"}]
    audit["coverage_gaps"] = [{"severity": "blocker", "code": "missing_check"}]
    if field == "failed_count":
        audit["check_counts"]["failed"] = value
    elif field == "gap_count":
        audit["check_counts"]["coverage_gaps"] = value
    else:
        audit[field] = value
    _write_json(audit_path, audit)
    review_path = _make_semantic_review(
        tmp_path,
        producer=producer,
        audit_path=audit_path,
        verdict="rejected",
        fatal_findings=[{"severity": "fatal", "code": "audit_invalid"}],
    )

    with pytest.raises(AdjudicationError, match="internally inconsistent"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "package",
            disposition="rejected",
            semantic_review=review_path,
        )
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_root", "/not/the/producer", "artifact_root is invalid"),
        ("result_file", "../result.json", "different producer root or result"),
        ("result_sha256", "0" * 64, "different producer root or result"),
    ],
)
@pytest.mark.parametrize("disposition", ["pending", "rejected"])
def test_audit_semantic_identity_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
    disposition: str,
) -> None:
    producer, audit_path, audit = _make_evidence(tmp_path)
    audit[field] = value
    _write_json(audit_path, audit)
    review_path = (
        _make_semantic_review(
            tmp_path,
            producer=producer,
            audit_path=audit_path,
            verdict="rejected",
            fatal_findings=[{"severity": "fatal", "code": "audit_identity"}],
        )
        if disposition == "rejected"
        else None
    )

    with pytest.raises(AdjudicationError, match=message):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "package",
            disposition=disposition,
            semantic_review=review_path,
        )


@pytest.mark.parametrize("disposition", ["pending", "rejected"])
def test_custody_identity_mismatch_is_rejected(tmp_path: Path, disposition: str) -> None:
    producer, audit_path, audit = _make_evidence(tmp_path)
    audit["custody"]["producer_manifest_sha256"] = "0" * 64
    _write_json(audit_path, audit)
    review_path = (
        _make_semantic_review(
            tmp_path,
            producer=producer,
            audit_path=audit_path,
            verdict="rejected",
            fatal_findings=[{"severity": "fatal", "code": "audit_custody"}],
        )
        if disposition == "rejected"
        else None
    )

    with pytest.raises(AdjudicationError, match="custody"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "package",
            disposition=disposition,
            semantic_review=review_path,
        )


def test_external_paths_symlinks_and_overwrite_are_refused(tmp_path: Path) -> None:
    producer, audit_path, _ = _make_evidence(tmp_path)

    with pytest.raises(AdjudicationError, match="outside the producer"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=producer / "adjudication",
            disposition="pending",
        )

    audit_link = tmp_path / "audit-link.json"
    audit_link.symlink_to(audit_path)
    with pytest.raises(AdjudicationError, match="symbolic link"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_link,
            output_directory=tmp_path / "symlink-package",
            disposition="pending",
        )

    package = tmp_path / "package"
    create_adjudication_package(
        producer_root=producer,
        audit_report=audit_path,
        output_directory=package,
        disposition="pending",
    )
    original_manifest = (package / ADJUDICATION_MANIFEST_NAME).read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=package,
            disposition="rejected",
        )
    assert (package / ADJUDICATION_MANIFEST_NAME).read_bytes() == original_manifest


def test_formal_acceptance_rejects_failed_gate_binding_and_auditor_identity(
    tmp_path: Path,
) -> None:
    failed_root = tmp_path / "failed-gate"
    failed_root.mkdir()
    producer, audit_path, _ = _make_evidence(failed_root, gates_pass=False)
    with pytest.raises(AdjudicationError, match="K0 through K7"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "failed-package",
            disposition="accepted",
        )

    bad_binding_root = tmp_path / "bad-binding"
    bad_binding_root.mkdir()
    producer, audit_path, _ = _make_evidence(
        bad_binding_root,
        binding_mismatch=True,
    )
    with pytest.raises(AdjudicationError, match="formal-binding bytes"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "binding-package",
            disposition="pending",
        )

    bad_auditor_root = tmp_path / "bad-auditor"
    bad_auditor_root.mkdir()
    producer, audit_path, _ = _make_evidence(
        bad_auditor_root,
        auditor_snapshot_mismatch=True,
    )
    with pytest.raises(AdjudicationError, match="auditor source"):
        create_adjudication_package(
            producer_root=producer,
            audit_report=audit_path,
            output_directory=tmp_path / "auditor-package",
            disposition="rejected",
        )


def test_cli_creates_package_and_reports_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    producer, audit_path, _ = _make_evidence(tmp_path, lane="development")
    output = tmp_path / "cli-package"
    arguments = [
        "--producer",
        str(producer),
        "--audit",
        str(audit_path),
        "--output",
        str(output),
        "--disposition",
        "pending",
    ]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["disposition"] == "pending"
    assert main(arguments) == 2
    assert "refusing to replace" in capsys.readouterr().err
