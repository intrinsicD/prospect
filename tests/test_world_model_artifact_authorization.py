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


def _preflight_receipt(binding_payload: bytes) -> bytes:
    return _canonical(
        {
            "schema": "prospect.wm001.formal-input-preflight.v1",
            "experiment_id": "WM-001",
            "protocol_version": "1.15.0",
            "binding_bytes": len(binding_payload),
            "binding_sha256": hashlib.sha256(
                binding_payload
            ).hexdigest(),
            "preformal_report_sha256": "1" * 64,
            "development_closure_sha256": "2" * 64,
            "accepted_closure_evidence_sha256": "3" * 64,
            "runtime_conformance_sha256": "4" * 64,
            "auditor_source_sha256": (
                artifact_audit._AUDITOR_SOURCE_SHA256
            ),
            "passed": True,
        }
    )


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
        "protocol_version": "1.15.0",
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
        / "v1.15"
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
            "schema": "prospect.world-model-lifecycle.formal-binding.v10",
            "experiment_id": "WM-001",
        }
    )
    preformal_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "v1.15.0"
        / "preformal"
        / "preformal-test-report-v1.15.0.json"
    )
    closure_path = (
        preformal_path.parents[2]
        / "development-closure-v1.15.0.json"
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
        / "operator-v1.15"
        / "bindings"
        / "formal-binding-v1.15.0"
    )
    preflight_payload = _preflight_receipt(binding_payload)
    (artifact_root / "formal-input-preflight.json").write_bytes(
        preflight_payload
    )
    _write_attempt(
        binding_attempt_path,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        inputs=observed,
        files={
            "formal-binding.json": binding_payload,
            "formal-input-preflight.json": preflight_payload,
        },
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


@pytest.mark.parametrize(
    "mutation",
    ["missing", "malformed", "copied-different"],
)
def test_formal_binding_authorization_requires_exact_preflight_receipt(
    authorization_space: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repository, _ = authorization_space
    artifact_root = repository / "formal-artifact"
    artifact_root.mkdir()
    binding_payload = _canonical(
        {
            "schema": "prospect.world-model-lifecycle.formal-binding.v10",
            "experiment_id": "WM-001",
        }
    )
    preflight_payload = _preflight_receipt(binding_payload)
    live_preflight = preflight_payload
    if mutation == "malformed":
        value = json.loads(preflight_payload)
        value["binding_sha256"] = "f" * 64
        live_preflight = _canonical(value)
    copied_preflight = (
        _canonical({"different": True})
        if mutation == "copied-different"
        else live_preflight
    )
    (artifact_root / "formal-input-preflight.json").write_bytes(
        copied_preflight
    )
    binding_attempt_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.15"
        / "bindings"
        / "formal-binding-v1.15.0"
    )
    files = {"formal-binding.json": binding_payload}
    if mutation != "missing":
        files["formal-input-preflight.json"] = live_preflight
    _write_attempt(
        binding_attempt_path,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        inputs=[],
        files=files,
    )
    monkeypatch.setattr(
        artifact_audit,
        "_authorization_preformal_rows",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        artifact_audit,
        "_authorization_development_closure",
        lambda *_args, **_kwargs: ([], None),
    )

    with pytest.raises(
        ArtifactAuditError,
        match="preflight",
    ):
        artifact_audit._validate_formal_authorization_lineage(
            repository=repository,
            artifact_root=artifact_root,
            binding={},
            binding_payload=binding_payload,
        )


@pytest.mark.parametrize("aliased_bytes", [True, 1.0])
def test_formal_input_preflight_rejects_numeric_binding_byte_aliases(
    aliased_bytes: object,
) -> None:
    binding_payload = b"x"
    receipt = json.loads(_preflight_receipt(binding_payload))
    receipt["binding_bytes"] = aliased_bytes

    with pytest.raises(
        ArtifactAuditError,
        match="preflight receipt is malformed or misbound",
    ):
        artifact_audit._formal_input_preflight_receipt(
            _canonical(receipt),
            binding_payload=binding_payload,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "input-substitution",
        "producer-manifest-role",
        "raw-result-role",
        "audit-role",
        "reproduction-role",
        "runtime-role",
        "invocation-role",
        "stderr-role",
    ],
)
def test_development_closure_authorization_reconstructs_producer_and_audit(
    authorization_space: tuple[Path, Path],
    mutation: str,
) -> None:
    repository, completion_root = authorization_space
    results = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
    )
    producer = results / "development" / "qualification-v1.15.0"
    producer.mkdir(parents=True)
    producer_manifest = producer / "producer-manifest.json"
    producer_result = producer / "result.json"
    result_payload = _canonical({"result": "development"})
    producer_result.write_bytes(result_payload)
    manifest_payload = _canonical(
        {
            "schema": "prospect.wm001.producer-manifest.v1",
            "experiment_id": "WM-001",
            "lane": "development",
            "status": "completed",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "completed_at_utc": "2026-01-01T00:01:00Z",
            "error": None,
            "manifest_excludes": ["producer-manifest.json"],
            "file_count": 1,
            "files": [
                {
                    "path": "result.json",
                    "bytes": len(result_payload),
                    "sha256": hashlib.sha256(
                        result_payload
                    ).hexdigest(),
                }
            ],
        }
    )
    producer_manifest.write_bytes(manifest_payload)
    producer_completion = completion_root / (
        hashlib.sha256(
            str(producer_manifest).encode("utf-8")
        ).hexdigest()
        + ".json"
    )
    os.link(producer_manifest, producer_completion)
    producer_rows = [
        _row(producer_manifest, manifest_payload),
        _row(producer_result, result_payload),
    ]
    audit_path = (
        results
        / "operator-v1.15"
        / "audits"
        / "development-audit-v1.15.0"
    )
    audit_payload = _canonical({"passed": True})
    runtime_payload = _canonical({"runtime": "sealed"})
    invocation_payload = _canonical({"invocation": "sealed"})
    stderr_payload = b""
    bootstrap_sha256 = "1" * 64
    auditor_sha256 = "2" * 64
    runner_sha256 = "3" * 64
    support_files: list[object] = []
    runtime_file = "audit-execution-02.runtime.json"
    invocation_file = "audit-invocation-manifest.json"
    stderr_file = "audit-stderr.txt"

    def execution_receipt() -> bytes:
        return _canonical(
            {
                "schema": (
                    "prospect.wm001.captured-audit-execution.v1"
                ),
                "returncode": 0,
                "passed": True,
                "source_mode": "descriptor",
                "command": ["python", "artifact_audit.py"],
                "stdout_file": "independent-audit.json",
                "stderr_file": stderr_file,
                "runtime_manifest_file": runtime_file,
                "invocation_manifest_file": invocation_file,
                "stdout_bytes": len(audit_payload),
                "stdout_sha256": hashlib.sha256(
                    audit_payload
                ).hexdigest(),
                "stderr_bytes": len(stderr_payload),
                "stderr_sha256": hashlib.sha256(
                    stderr_payload
                ).hexdigest(),
                "runtime_manifest_bytes": len(runtime_payload),
                "runtime_manifest_sha256": hashlib.sha256(
                    runtime_payload
                ).hexdigest(),
                "invocation_manifest_bytes": len(invocation_payload),
                "invocation_manifest_sha256": hashlib.sha256(
                    invocation_payload
                ).hexdigest(),
                "bootstrap_sha256": bootstrap_sha256,
                "auditor_source_sha256": auditor_sha256,
                "support_files": support_files,
            }
        )

    reproduction_payload = _canonical(
        {
            "schema": "prospect.wm001.audit-reproduction.v2",
            "experiment_id": "WM-001",
            "protocol_version": "1.15.0",
            "supplied_audit_sha256": hashlib.sha256(
                audit_payload
            ).hexdigest(),
            "reproduced_audit_sha256": hashlib.sha256(
                audit_payload
            ).hexdigest(),
            "byte_identical": True,
            "returncode": 0,
            "source_mode": "descriptor",
            "stdout_bytes": len(audit_payload),
            "stderr_file": stderr_file,
            "stderr_bytes": len(stderr_payload),
            "stderr_sha256": hashlib.sha256(
                stderr_payload
            ).hexdigest(),
            "runtime_manifest_file": runtime_file,
            "runtime_manifest_bytes": len(runtime_payload),
            "runtime_manifest_sha256": hashlib.sha256(
                runtime_payload
            ).hexdigest(),
            "invocation_manifest_file": invocation_file,
            "invocation_manifest_bytes": len(invocation_payload),
            "invocation_manifest_sha256": hashlib.sha256(
                invocation_payload
            ).hexdigest(),
            "bootstrap_sha256": bootstrap_sha256,
            "runner_source_sha256": runner_sha256,
            "auditor_source_sha256": auditor_sha256,
            "support_files": support_files,
            "passed": True,
        }
    )
    audit_files = {
        "audit-execution-01.execution.json": execution_receipt(),
        "audit-execution-02.execution.json": execution_receipt(),
        runtime_file: runtime_payload,
        "audit-reproduction.json": reproduction_payload,
        invocation_file: invocation_payload,
        stderr_file: stderr_payload,
        "independent-audit.json": audit_payload,
    }
    audit_primary = {
        "producer_root": str(producer),
        "audit_file": "independent-audit.json",
        "executions": [
            "audit-execution-01.execution.json",
            "audit-execution-02.execution.json",
        ],
        "execution_failures": [],
        "reproduction_file": "audit-reproduction.json",
        "reproduction_runtime_file": runtime_file,
        "claim_file": None,
    }
    _write_attempt(
        audit_path,
        kind="audit",
        lane="development",
        primary=audit_primary,
        inputs=producer_rows,
        files=audit_files,
    )
    audit_attempt = artifact_audit._authorization_development_audit(
        repository,
        producer=producer,
        producer_rows=producer_rows,
    )

    closure_path = (
        results
        / "development"
        / "development-closure-v1.15.0.json"
    )
    qualification_archive = {
        "format": "ustar-uncompressed-v1",
        "file": "development-qualification-v1.15.0.tar",
        "members": [
            {
                "path": "producer/producer-manifest.json",
                "sha256": producer_rows[0]["sha256"],
            },
            {
                "path": "producer/result.json",
                "sha256": producer_rows[1]["sha256"],
            },
            {
                "path": "evidence/independent-audit.json",
                "sha256": hashlib.sha256(
                    audit_files["independent-audit.json"]
                ).hexdigest(),
            },
            {
                "path": "evidence/audit-reproduction.json",
                "sha256": hashlib.sha256(
                    audit_files["audit-reproduction.json"]
                ).hexdigest(),
            },
            {
                "path": f"evidence/{runtime_file}",
                "sha256": hashlib.sha256(
                    audit_files[runtime_file]
                ).hexdigest(),
            },
            {
                "path": f"evidence/{invocation_file}",
                "sha256": hashlib.sha256(
                    audit_files[invocation_file]
                ).hexdigest(),
            },
            {
                "path": f"evidence/{stderr_file}",
                "sha256": hashlib.sha256(
                    audit_files[stderr_file]
                ).hexdigest(),
            },
        ],
    }
    role_member_by_mutation = {
        "producer-manifest-role": "producer/producer-manifest.json",
        "raw-result-role": "producer/result.json",
        "audit-role": "evidence/independent-audit.json",
        "reproduction-role": "evidence/audit-reproduction.json",
        "runtime-role": f"evidence/{runtime_file}",
        "invocation-role": f"evidence/{invocation_file}",
        "stderr-role": f"evidence/{stderr_file}",
    }
    if mutation in role_member_by_mutation:
        member = next(
            row
            for row in qualification_archive["members"]
            if row["path"] == role_member_by_mutation[mutation]
        )
        member["sha256"] = "f" * 64
    closure = {
        field: None
        for field in artifact_audit._AUTHORIZATION_CLOSURE_FIELDS
    }
    closure.update(
        {
            "schema": "prospect.wm001.development-closure.v2",
            "experiment_id": "WM-001",
            "protocol_version": "1.15.0",
            "producer_root": str(producer),
            "producer_manifest_member": "producer/producer-manifest.json",
            "raw_result_member": "producer/result.json",
            "independent_audit_member": "evidence/independent-audit.json",
            "audit_reproduction_member": "evidence/audit-reproduction.json",
            "audit_runtime_manifest_member": (
                f"evidence/{runtime_file}"
            ),
            "audit_invocation_manifest_member": (
                f"evidence/{invocation_file}"
            ),
            "audit_stderr_member": f"evidence/{stderr_file}",
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
        / "operator-v1.15"
        / "closures"
        / "development-closure-v1.15.0"
    )
    audit_terminal_row = next(
        row
        for row in audit_attempt.member_rows
        if row["path"] == str(audit_attempt.terminal)
    )
    fresh_reopen = {
        "schema": (
            "prospect.wm001.development-closure-fresh-reopen.v1"
        ),
        "experiment_id": "WM-001",
        "protocol_version": "1.15.0",
        "mode": "fresh-closure-reopen",
        "challenge": "1" * 64,
        "requesting_process_id": 100,
        "verifier_process_id": 101,
        "matrix_contract_sha256": (
            artifact_audit._DEVELOPMENT_MATRIX_CONTRACT_SHA256
        ),
        "development_closure_sha256": closure_row["sha256"],
        "producer_manifest_sha256": producer_rows[0]["sha256"],
        "raw_result_sha256": producer_rows[1]["sha256"],
        "passed": True,
    }
    fresh_payload = _canonical(fresh_reopen)
    reference = {
        "schema": "prospect.wm001.closure-reference.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.15.0",
        "closure_marker": str(closure_path),
        "closure_sha256": closure_row["sha256"],
        "qualification_archive": qualification_archive,
        "producer_root": str(producer),
        "audit_attempt": str(audit_path),
        "audit_attempt_manifest_sha256": audit_terminal_row["sha256"],
        "fresh_reopen_file": "fresh-runtime-reopen.json",
        "fresh_reopen_sha256": hashlib.sha256(
            fresh_payload
        ).hexdigest(),
    }
    expected_inputs = [
        *producer_rows,
        *audit_attempt.member_rows,
        dict(audit_attempt.completion_row),
    ]
    observed_inputs = [dict(row) for row in expected_inputs]
    if mutation == "input-substitution":
        observed_inputs[-2]["sha256"] = "e" * 64
    _write_attempt(
        closure_attempt_path,
        kind="closure",
        lane="development",
        primary={"closure_reference_file": "closure-reference.json"},
        inputs=observed_inputs,
        files={
            "closure-reference.json": _canonical(reference),
            "fresh-runtime-reopen.json": fresh_payload,
        },
    )
    binding = {
        "development_qualification": {
            "closure_bytes": len(closure_payload),
            "closure_sha256": hashlib.sha256(
                closure_payload
            ).hexdigest(),
        }
    }

    if mutation == "input-substitution":
        with pytest.raises(
            ArtifactAuditError,
            match="closure inputs differ",
        ):
            artifact_audit._authorization_development_closure(
                repository,
                binding=binding,
            )
    elif mutation in role_member_by_mutation:
        with pytest.raises(
            ArtifactAuditError,
            match="archive roles differ from live closure inputs",
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
        / "operator-v1.15"
        / "bindings"
        / "formal-binding-v1.15.0"
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


@pytest.mark.parametrize("aliased_count", [True, 1.0])
def test_authorization_attempt_rejects_numeric_file_count_aliases(
    authorization_space: tuple[Path, Path],
    aliased_count: object,
) -> None:
    repository, _ = authorization_space
    attempt_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.15"
        / "bindings"
        / "formal-binding-v1.15.0"
    )
    _write_attempt(
        attempt_path,
        kind="binding",
        lane=None,
        primary={"binding_file": "formal-binding.json"},
        inputs=[],
        files={"formal-binding.json": b"binding"},
    )
    terminal = attempt_path / "operator-attempt.json"
    manifest = json.loads(terminal.read_bytes())
    manifest["file_count"] = aliased_count
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        ArtifactAuditError,
        match="not exactly accepted",
    ):
        artifact_audit._authorization_attempt(
            attempt_path,
            kind="binding",
            lane=None,
            primary={"binding_file": "formal-binding.json"},
            label="numeric-alias attempt",
        )


@pytest.mark.parametrize("aliased_bytes", [True, 1.0])
def test_authorization_reference_rejects_numeric_byte_aliases(
    authorization_space: tuple[Path, Path],
    aliased_bytes: object,
) -> None:
    repository, _ = authorization_space
    attempt_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.15"
        / "audits"
        / "development-audit-v1.15.0"
    )
    payload = b"x"
    _write_attempt(
        attempt_path,
        kind="audit",
        lane="development",
        primary={},
        inputs=[],
        files={"payload.bin": payload},
    )
    attempt = artifact_audit._authorization_attempt(
        attempt_path,
        kind="audit",
        lane="development",
        primary={},
        label="numeric-alias reference",
    )
    reference = {
        "stdout_file": "payload.bin",
        "stdout_bytes": aliased_bytes,
        "stdout_sha256": hashlib.sha256(payload).hexdigest(),
    }

    with pytest.raises(ArtifactAuditError, match="differs from its identity"):
        artifact_audit._authorization_referenced_payload(
            attempt,
            reference,
            prefix="stdout",
            label="numeric-alias payload",
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
        closure_attempt_path="/repo/development-closure-attempt",
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
    assert commands[8] == (
        "runtime-accepted-closure-evidence",
        "runtime",
        (
            "/runtime/python",
            "-I",
            "-S",
            "-B",
            "/repo/bench/world_model_lifecycle/launch_bootstrap.py",
            "--bootstrap",
            "/repo/bench/world_model_lifecycle/producer_bootstrap.py",
            "--runtime-seal",
            "/repo/runtime-seal.json",
            "preformal-runtime",
            "accepted-closure-evidence",
            "--development-closure",
            "/repo/development-closure.json",
            "--closure-attempt",
            "/repo/development-closure-attempt",
        ),
    )
