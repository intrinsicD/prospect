from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

import bench.world_model_lifecycle.artifact_audit as artifact_audit
from bench.world_model_lifecycle.artifact_audit import ArtifactAuditError


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


def _row(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_attempt(
    path: Path,
    *,
    kind: str,
    lane: str | None,
    primary: Mapping[str, object],
    inputs: list[dict[str, object]],
    files: Mapping[str, bytes],
    finalize: bool = True,
) -> None:
    path.mkdir(parents=True)
    file_rows: list[dict[str, object]] = []
    for filename, payload in sorted(files.items()):
        member = path / filename
        member.write_bytes(payload)
        file_rows.append(
            {
                "path": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema": "prospect.wm001.operator-attempt.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.5.0",
        "assurance": dict(artifact_audit._ASSURANCE),
        "kind": kind,
        "lane": lane,
        "status": "accepted",
        "inputs": inputs,
        "primary": dict(primary),
        "error": None,
        "files": file_rows,
        "file_count": len(file_rows),
        "manifest_excludes": ["operator-attempt.json"],
    }
    terminal = path / "operator-attempt.json"
    terminal.write_bytes(_canonical(manifest))
    if finalize:
        marker = artifact_audit._OUTER_COMPLETIONS_ROOT / (
            hashlib.sha256(str(terminal).encode("utf-8")).hexdigest()
            + ".json"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        os.link(terminal, marker, follow_symlinks=False)


@pytest.fixture
def authorization_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    completion_root = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "outer-completions"
        / "v1.5"
    )
    completion_root.mkdir(parents=True)
    monkeypatch.setattr(
        artifact_audit,
        "_OUTER_COMPLETIONS_ROOT",
        completion_root,
    )
    return repository, completion_root


@pytest.mark.parametrize("substitute", [False, True])
def test_formal_binding_authorization_reconstructs_exact_ordered_inputs(
    authorization_space: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    substitute: bool,
) -> None:
    repository, _ = authorization_space
    artifact_root = repository / "formal-artifact"
    artifact_root.mkdir()
    binding_payload = _canonical(
        {
            "schema": "prospect.world-model-lifecycle.formal-binding.v5",
            "experiment_id": "WM-001",
        }
    )
    preformal_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "preformal-test-report-v1.5.0.json"
    )
    closure_path = preformal_path.with_name(
        "development-closure-v1.5.0.json"
    )
    preformal_payload = b"preformal\n"
    closure_payload = b"closure\n"
    expected = [
        _row(preformal_path, preformal_payload),
        _row(closure_path, closure_payload),
    ]
    observed = [dict(row) for row in expected]
    if substitute:
        observed[0]["sha256"] = "f" * 64
    binding_attempt_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.5"
        / "bindings"
        / "formal-binding-v1.5.0"
    )
    _write_attempt(
        binding_attempt_path,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        inputs=observed,
        files={"formal-binding.json": binding_payload},
    )
    monkeypatch.setattr(
        artifact_audit,
        "_authorization_preformal_rows",
        lambda *_args, **_kwargs: expected[:1],
    )
    monkeypatch.setattr(
        artifact_audit,
        "_authorization_development_closure",
        lambda *_args, **_kwargs: (expected[1:], None),
    )

    if substitute:
        with pytest.raises(
            ArtifactAuditError,
            match="authorization inputs differ",
        ):
            artifact_audit._validate_formal_authorization_lineage(
                repository=repository,
                artifact_root=artifact_root,
                binding={},
                binding_payload=binding_payload,
            )
    else:
        attempt = artifact_audit._validate_formal_authorization_lineage(
            repository=repository,
            artifact_root=artifact_root,
            binding={},
            binding_payload=binding_payload,
        )
        assert attempt.root == binding_attempt_path


@pytest.mark.parametrize("substitute", [False, True])
def test_development_closure_authorization_reconstructs_producer_and_audit(
    authorization_space: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    substitute: bool,
) -> None:
    repository, _ = authorization_space
    results = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
    )
    producer = results / "development" / "qualification-v1.5.0"
    producer_manifest = producer / "producer-manifest.json"
    producer_result = producer / "result.json"
    producer_rows = [
        _row(producer_manifest, b"manifest\n"),
        _row(producer_result, b"result\n"),
    ]
    audit_path = (
        results
        / "operator-v1.5"
        / "audits"
        / "development-audit-v1.5.0"
    )
    _write_attempt(
        audit_path,
        kind="audit",
        lane="development",
        primary={},
        inputs=producer_rows,
        files={"independent-audit.json": _canonical({"passed": True})},
    )
    audit_attempt = artifact_audit._authorization_attempt(
        audit_path,
        kind="audit",
        lane="development",
        primary={},
        label="test development audit",
    )
    monkeypatch.setattr(
        artifact_audit,
        "_authorization_development_producer",
        lambda _repository: (producer, producer_rows),
    )
    monkeypatch.setattr(
        artifact_audit,
        "_authorization_development_audit",
        lambda *_args, **_kwargs: audit_attempt,
    )

    closure_path = (
        results
        / "development"
        / "development-closure-v1.5.0.json"
    )
    qualification_archive = {
        "format": "ustar-uncompressed-v1",
        "file": "development-qualification-v1.5.0.tar",
    }
    closure = {
        field: None
        for field in artifact_audit._AUTHORIZATION_CLOSURE_FIELDS
    }
    closure.update(
        {
            "schema": "prospect.wm001.development-closure.v2",
            "experiment_id": "WM-001",
            "protocol_version": "1.5.0",
            "producer_root": str(producer),
            "qualification_archive": qualification_archive,
            "engineering_verified": True,
            "audit_reproduced": True,
            "performance_values_bound": False,
        }
    )
    closure_payload = _canonical(closure)
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    closure_path.write_bytes(closure_payload)
    closure_row = _row(closure_path, closure_payload)

    closure_attempt_path = (
        results
        / "operator-v1.5"
        / "closures"
        / "development-closure-v1.5.0"
    )
    audit_terminal_row = next(
        row
        for row in audit_attempt.member_rows
        if row["path"] == str(audit_attempt.terminal)
    )
    reference = {
        "schema": "prospect.wm001.closure-reference.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.5.0",
        "closure_marker": str(closure_path),
        "closure_sha256": closure_row["sha256"],
        "qualification_archive": qualification_archive,
        "producer_root": str(producer),
        "audit_attempt": str(audit_path),
        "audit_attempt_manifest_sha256": audit_terminal_row["sha256"],
    }
    expected_inputs = [
        *producer_rows,
        *audit_attempt.member_rows,
        dict(audit_attempt.completion_row),
    ]
    observed_inputs = [dict(row) for row in expected_inputs]
    if substitute:
        observed_inputs[-2]["sha256"] = "e" * 64
    _write_attempt(
        closure_attempt_path,
        kind="closure",
        lane="development",
        primary={"closure_reference_file": "closure-reference.json"},
        inputs=observed_inputs,
        files={"closure-reference.json": _canonical(reference)},
    )
    binding = {
        "development_qualification": {
            "closure_bytes": len(closure_payload),
            "closure_sha256": hashlib.sha256(
                closure_payload
            ).hexdigest(),
        }
    }

    if substitute:
        with pytest.raises(
            ArtifactAuditError,
            match="closure inputs differ",
        ):
            artifact_audit._authorization_development_closure(
                repository,
                binding=binding,
            )
    else:
        rows, attempt = (
            artifact_audit._authorization_development_closure(
                repository,
                binding=binding,
            )
        )
        assert rows[0] == closure_row
        assert attempt.root == closure_attempt_path


def test_authorization_rejects_unfinalized_or_sibling_attempt(
    authorization_space: tuple[Path, Path],
) -> None:
    repository, _ = authorization_space
    expected = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.5"
        / "bindings"
        / "formal-binding-v1.5.0"
    )
    sibling = expected.with_name("sibling-binding")
    binding_payload = _canonical({"binding": True})
    _write_attempt(
        sibling,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        inputs=[],
        files={"formal-binding.json": binding_payload},
    )
    with pytest.raises(ArtifactAuditError, match="cannot be resolved"):
        artifact_audit._validate_formal_authorization_lineage(
            repository=repository,
            artifact_root=repository,
            binding={},
            binding_payload=binding_payload,
        )

    _write_attempt(
        expected,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        inputs=[],
        files={"formal-binding.json": binding_payload},
        finalize=False,
    )
    with pytest.raises(ArtifactAuditError, match="exactly 2 hard link"):
        artifact_audit._authorization_attempt(
            expected,
            kind="binding",
            lane=None,
            primary={"binding_file": "formal-binding.json"},
            label="unfinalized binding",
        )


def test_authorization_rejects_aliased_input_path(
    authorization_space: tuple[Path, Path],
) -> None:
    repository, _ = authorization_space
    canonical = repository / "canonical.json"
    canonical.write_bytes(b"{}\n")
    alias = repository / "alias.json"
    alias.symlink_to(canonical)

    with pytest.raises(ArtifactAuditError, match="aliased"):
        artifact_audit._authorization_file_row(
            alias,
            label="aliased authorization input",
            expected_nlink=1,
        )


def test_independent_preformal_review_command_uses_module_entrypoint() -> None:
    commands = artifact_audit._preformal_expected_commands(
        qa_executable="/qa/python",
        runtime_executable="/runtime/python",
        source={
            "implementation_files": [
                {
                    "path": "tests/test_epistemic_contract.py",
                    "bytes": 1,
                    "sha256": "1" * 64,
                },
                {
                    "path": "tests/test_world_model_audit_runner.py",
                    "bytes": 1,
                    "sha256": "2" * 64,
                },
            ]
        },
        repository_cwd="/repo",
        runtime_seal_path="/repo/runtime-seal.json",
        development_closure_path="/repo/development-closure.json",
        prospective_review_path="/repo/review.json",
        device="cuda",
    )

    assert commands[7] == (
        "prospective-harness-review",
        "qa",
        (
            "/qa/python",
            "-I",
            "-B",
            "-m",
            "bench.world_model_lifecycle.preformal",
            "verify-prospective-review",
            "--review",
            "/repo/review.json",
        ),
    )
