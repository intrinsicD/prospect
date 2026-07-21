from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle import audit_runner as audit_runner_module
from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle import experiment as experiment_module
from bench.world_model_lifecycle import operator as operator_module
from bench.world_model_lifecycle import preformal
from bench.world_model_lifecycle import verify as verify_module

_GIT_IDENTITY = {
    "commit": "a" * 40,
    "tree": "b" * 40,
    "worktree_clean": True,
}
_RUNTIME_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "LAZY_LEGACY_OP": "False",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
    "SDL_AUDIODRIVER": "dsp",
    "TZ": "UTC",
}


@pytest.fixture(autouse=True)
def _isolated_outer_completion_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outer-completions"
    root.mkdir()
    monkeypatch.setattr(operator_module, "OUTER_COMPLETIONS_ROOT", root)


def _canonical(value: object) -> bytes:
    return preformal._canonical_json_bytes(value) + b"\n"


def _binding_schema_example(
    schema: dict[str, Any],
    root: dict[str, Any],
    *,
    path: tuple[str, ...] = (),
    item_index: int = 0,
) -> Any:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        return _binding_schema_example(
            root["$defs"][reference.removeprefix("#/$defs/")],
            root,
            path=path,
            item_index=item_index,
        )
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "enum" in schema:
        return copy.deepcopy(schema["enum"][0])
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "null" in schema_type:
            return None
        schema_type = schema_type[0]
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        return {
            name: _binding_schema_example(
                properties[name],
                root,
                path=(*path, name),
                item_index=item_index,
            )
            for name in schema.get("required", [])
        }
    if schema_type == "array":
        return [
            _binding_schema_example(
                schema["items"],
                root,
                path=(*path, "item"),
                item_index=index,
            )
            for index in range(int(schema.get("minItems", 0)))
        ]
    if schema_type == "integer":
        return int(schema.get("minimum", 0))
    if schema_type == "number":
        return float(schema.get("minimum", 0.0))
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    assert schema_type == "string"
    pattern = str(schema.get("pattern", ""))
    if pattern == "^[0-9a-f]{40}$":
        return "a" * 40
    if pattern == "^[0-9a-f]{64}$":
        return "a" * 64
    if "development-qualification-[0-9a-f]{16}" in pattern:
        archive = "development-qualification-" + "a" * 16 + ".tar"
        if pattern.startswith("^bench/"):
            return (
                "bench/world_model_lifecycle/results/development/"
                + archive
            )
        return archive
    if pattern.startswith("^/"):
        return f"/fixture/{item_index}"
    if schema.get("format") == "date-time":
        return "2026-07-21T00:00:00Z"
    if path and path[-1] == "path":
        return f"fixture-{item_index:02d}.log"
    return "fixture"


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _result_free_inventory() -> dict[str, object]:
    root = {
        "semantics_id": "prospect.wm001.package-root.v2",
        "path": "/runtime/site-packages",
        "file_count": 7,
        "directory_count": 3,
        "total_bytes": 70,
        "inventory_sha256": "a" * 64,
    }
    return {
        "packages": [
            {
                "name": "python",
                "version": "3.12.0",
                "distribution_sha256": "b" * 64,
                "declared_file_count": 1,
                "editable": False,
            },
            {
                "name": "prospect",
                "version": "0.1.0",
                "distribution_sha256": "c" * 64,
                "declared_file_count": 5,
                "editable": False,
            },
        ],
        "package_roots": [root],
        "standard_library": {
            "semantics_id": "prospect.wm001.standard-library.v2",
            "path": "/runtime/stdlib",
            "file_count": 11,
            "directory_count": 4,
            "total_bytes": 110,
            "inventory_sha256": "d" * 64,
        },
        "package_ownership": {
            "semantics_id": "prospect.wm001.package-ownership.v1",
            "root": root["path"],
            "file_count": root["file_count"],
            "directory_count": root["directory_count"],
            "shared_file_count": 0,
            "identity_sha256": "e" * 64,
        },
    }


def _fresh_identity_conformance() -> dict[str, object]:
    return {
        "schema": "prospect.wm001.fresh-runtime-identity-conformance.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "mode": "fresh-identity-conformance",
        "challenge": "7" * 64,
        "requesting_process_id": 101,
        "verifier_process_id": 202,
        "matrix_contract_sha256": (
            preformal._DEVELOPMENT_MATRIX_CONTRACT_SHA256
        ),
        "passed": True,
    }


def _runtime_conformance(
    device: str = "cpu",
) -> dict[str, object]:
    inventory = _result_free_inventory()
    fresh_identity = _fresh_identity_conformance()
    return {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "bootstrap-inventory-conformance",
        "device": device,
        "passed": True,
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            preformal._canonical_json_bytes(inventory)
        ).hexdigest(),
        "conformance_sha256": "2" * 64,
        "fresh_runtime_identity_conformance": fresh_identity,
        "fresh_runtime_identity_conformance_sha256": hashlib.sha256(
            preformal._canonical_json_bytes(fresh_identity)
        ).hexdigest(),
        "restart_runtime_conformance_report_sha256": "3" * 64,
        "restart_runtime_execution_receipt_sha256": "4" * 64,
        "restart_runtime_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "restart_runtime_repeat_count": 3,
        "restart_runtime_path_descriptor_equal": True,
        "repeat_count": 3,
        "path_descriptor_equal": True,
    }


def _accepted_closure_evidence(
    *,
    development_closure: Path,
    closure_terminal: Path,
    closure_completion: Path,
) -> dict[str, object]:
    return {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "accepted-closure-evidence",
        "passed": True,
        "development_closure_sha256": hashlib.sha256(
            development_closure.read_bytes()
        ).hexdigest(),
        "producer_manifest_sha256": "5" * 64,
        "raw_result_sha256": "6" * 64,
        "closure_attempt_manifest_sha256": hashlib.sha256(
            closure_terminal.read_bytes()
        ).hexdigest(),
        "closure_outer_completion_sha256": hashlib.sha256(
            closure_completion.read_bytes()
        ).hexdigest(),
    }


def _development_closure(producer_root: Path) -> dict[str, object]:
    member_digests = {
        "evidence/audit-invocation.json": "a" * 64,
        "evidence/audit-reproduction.json": "b" * 64,
        "evidence/audit-runtime.json": "c" * 64,
        "evidence/audit-stderr.log": "d" * 64,
        "evidence/development-result-qualification.json": hashlib.sha256(
            b'{"fixture":"development-result-qualification"}\n'
        ).hexdigest(),
        "evidence/independent-audit.json": "f" * 64,
        "evidence/launch-bootstrap.py": "1" * 64,
        "evidence/producer-bootstrap.py": "2" * 64,
        "evidence/producer-runtime-seal.json": "3" * 64,
        "producer/producer-manifest.json": "5" * 64,
        "producer/result.json": "6" * 64,
    }
    archive_sha256 = "9" * 64
    archive_file = (
        f"development-qualification-{archive_sha256[:16]}.tar"
    )
    return {
        "schema": "prospect.wm001.development-closure.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "source": {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "worktree_clean": True,
            "dependency_lock_sha256": "1" * 64,
            "producer_bootstrap_sha256": "2" * 64,
            "launch_bootstrap_sha256": "3" * 64,
            "runner_source_sha256": "4" * 64,
            "auditor_source_sha256": "5" * 64,
        },
        "producer_root": str(producer_root),
        "producer_manifest_member": (
            "producer/producer-manifest.json"
        ),
        "raw_result_member": "producer/result.json",
        "result_qualification_member": (
            "evidence/development-result-qualification.json"
        ),
        "independent_audit_member": (
            "evidence/independent-audit.json"
        ),
        "audit_reproduction_member": (
            "evidence/audit-reproduction.json"
        ),
        "audit_runtime_manifest_member": (
            "evidence/audit-runtime.json"
        ),
        "audit_invocation_manifest_member": (
            "evidence/audit-invocation.json"
        ),
        "audit_stderr_member": "evidence/audit-stderr.log",
        "producer_execution": {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "worktree_clean": True,
            "dependency_lock_sha256": "1" * 64,
            "python_executable": "/runtime/bin/python",
            "python_executable_sha256": "4" * 64,
            "python_version": "3.12.0",
            "platform": "Linux-runtime",
            "machine": "x86_64",
            "device": "cpu",
            "python_flags": dict(preformal._RUNTIME_FLAGS),
            "process_environment": dict(_RUNTIME_ENVIRONMENT),
            "accelerator": None,
            "thread_count": 1,
            "interop_thread_count": 1,
            "cuda_runtime": None,
            "cuda_driver": None,
            "cublas_workspace_config": None,
            "deterministic_algorithms": True,
            "runtime_seal_sha256": "3" * 64,
            "runtime_seal_descriptor_custody": True,
            "producer_bootstrap_sha256": "2" * 64,
            "bootstrap_descriptor_custody": True,
            "package_roots": [
                {
                    "environment": "runtime-only",
                    "packages": ["prospect", "torch"],
                }
            ],
            "standard_library": {"environment": "runtime-only"},
        },
        "producer_custody": {
            "runtime_seal_member": (
                "evidence/producer-runtime-seal.json"
            ),
            "runtime_seal_sha256": "3" * 64,
            "producer_bootstrap_member": (
                "evidence/producer-bootstrap.py"
            ),
            "producer_bootstrap_sha256": "2" * 64,
            "launch_bootstrap_member": (
                "evidence/launch-bootstrap.py"
            ),
            "launch_bootstrap_sha256": "1" * 64,
            "package_ownership": {"environment": "runtime-only"},
        },
        "audit_execution": {
            "receipt_sha256": "b" * 64,
            "runtime_manifest_sha256": "c" * 64,
            "invocation_manifest_sha256": "a" * 64,
            "stderr_sha256": "d" * 64,
            "bootstrap_sha256": "2" * 64,
            "runner_source_sha256": "7" * 64,
            "auditor_source_sha256": "8" * 64,
            "support_files": [
                {
                    "path": "producer_bootstrap.py",
                    "bytes": 1,
                    "sha256": "2" * 64,
                },
                {
                    "path": "protocol.json",
                    "bytes": 1,
                    "sha256": "3" * 64,
                },
                {
                    "path": "schemas/raw-result.schema.json",
                    "bytes": 1,
                    "sha256": "4" * 64,
                },
            ],
            "source_mode": "descriptor",
        },
        "qualification_archive": {
            "format": "ustar-uncompressed-v1",
            "file": archive_file,
            "canonical_path": (
                "bench/world_model_lifecycle/results/development/"
                f"{archive_file}"
            ),
            "bytes": 10_240,
            "sha256": archive_sha256,
            "members": [
                {
                    "path": path,
                    "bytes": 1,
                    "sha256": digest,
                }
                for path, digest in sorted(member_digests.items())
            ],
        },
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }


def _rewrite_report(path: Path, report: dict[str, Any]) -> None:
    path.write_bytes(_canonical(report))


def _qa_closure() -> dict[str, object]:
    closure: dict[str, object] = {
        "schema": "prospect.wm001.qa-closure.v1",
        "sys_path": ["/qa/site-packages", "/qa/stdlib"],
        "distributions": [
            {
                "name": "mypy",
                "version": "1.18.2",
                "editable": False,
                "declared_file_count": 3,
                "total_bytes": 17,
                "distribution_sha256": "c" * 64,
            },
            {
                "name": "prospect",
                "version": "0.0.1",
                "editable": False,
                "declared_file_count": 5,
                "total_bytes": 29,
                "distribution_sha256": "e" * 64,
            },
            {
                "name": "pytest",
                "version": "8.4.2",
                "editable": False,
                "declared_file_count": 4,
                "total_bytes": 23,
                "distribution_sha256": "d" * 64,
            },
        ],
    }
    closure["inventory_sha256"] = hashlib.sha256(
        preformal._canonical_json_bytes(closure)
    ).hexdigest()
    return closure


def _review(
    implementation_files: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows = [] if implementation_files is None else copy.deepcopy(implementation_files)
    return {
        "schema": preformal.REVIEW_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "implementation_files": rows,
        "implementation_manifest_sha256": hashlib.sha256(
            preformal._canonical_json_bytes(rows)
        ).hexdigest(),
        "reviewer": {
            "kind": "independent-adversarial-referee",
            "identifier": "test-referee",
        },
        "disposition": "accepted",
        "unresolved_blockers": [],
        "findings": [],
    }


def _runtime_seal(runtime_executable: Path) -> dict[str, object]:
    identity = preformal._executable_identity(runtime_executable)
    return {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "assurance": {
            "trust_model_id": "prospect.wm001.trust-model.v1",
            "tamper_resistant": False,
            "external_attestation": False,
            "exclusive_path_use_required": True,
        },
        "git_commit": _GIT_IDENTITY["commit"],
        "git_tree": _GIT_IDENTITY["tree"],
        "worktree_clean": True,
        "python": {
            "executable": str(runtime_executable),
            "resolved_executable": identity["resolved_path"],
            "sha256": identity["sha256"],
            "version": list(sys.version_info[:3]),
        },
        "required_flags": dict(preformal._RUNTIME_FLAGS),
        "process_environment": dict(_RUNTIME_ENVIRONMENT),
        "bootstrap_source_sha256": preformal._file_identity(
            preformal.PRODUCER_BOOTSTRAP_PATH,
            label="producer bootstrap",
            expected_nlink=preformal._SINGLE_LINK_CUSTODY,
        )["sha256"],
        "standard_library": {"identity": "stdlib"},
        "package_roots": [{"identity": "packages"}],
        "package_ownership": {"identity": "ownership"},
    }


def _write_completed_runtime_seal(
    directory: Path,
    seal: dict[str, object],
) -> Path:
    path = directory / "runtime-seal.json"
    path.write_bytes(_canonical(seal))
    marker = operator_module.outer_completion_marker(path)
    os.link(path, marker)
    return path


def test_runtime_seal_requires_exact_assurance_and_two_link_custody(
    tmp_path: Path,
) -> None:
    runtime_executable = Path(sys.executable)
    seal = _runtime_seal(runtime_executable)
    completed = tmp_path / "completed"
    completed.mkdir()
    path = _write_completed_runtime_seal(completed, seal)

    verified, environment = preformal._validated_runtime_seal(
        path,
        runtime_executable=runtime_executable,
    )
    assert verified == seal
    assert environment == _RUNTIME_ENVIRONMENT

    unfinalized = tmp_path / "unfinalized.json"
    unfinalized.write_bytes(_canonical(seal))
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="aliased or not a regular file",
    ):
        preformal._validated_runtime_seal(
            unfinalized,
            runtime_executable=runtime_executable,
        )


def test_runtime_seal_rejects_noncanonical_second_link(
    tmp_path: Path,
) -> None:
    runtime_executable = Path(sys.executable)
    seal = _runtime_seal(runtime_executable)
    path = tmp_path / "runtime-seal.json"
    path.write_bytes(_canonical(seal))
    os.link(path, tmp_path / "untrusted-alias.json")

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="deterministic outer completion",
    ):
        preformal._validated_runtime_seal(
            path,
            runtime_executable=runtime_executable,
        )


@pytest.mark.parametrize(
    ("assurance", "case"),
    [
        (None, "missing"),
        (
            {
                "trust_model_id": "prospect.wm001.trust-model.v1",
                "tamper_resistant": True,
                "external_attestation": False,
                "exclusive_path_use_required": True,
            },
            "overstated",
        ),
    ],
)
def test_runtime_seal_rejects_missing_or_overstated_assurance(
    tmp_path: Path,
    assurance: dict[str, object] | None,
    case: str,
) -> None:
    runtime_executable = Path(sys.executable)
    seal = _runtime_seal(runtime_executable)
    if assurance is None:
        del seal["assurance"]
    else:
        seal["assurance"] = assurance
    directory = tmp_path / case
    directory.mkdir()
    path = _write_completed_runtime_seal(directory, seal)

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="runtime seal identity is malformed",
    ):
        preformal._validated_runtime_seal(
            path,
            runtime_executable=runtime_executable,
        )


def test_live_bootstrap_custody_rejects_assurance_overstatement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = _runtime_seal(Path(sys.executable))
    payload = _canonical(seal)
    monkeypatch.setattr(
        preformal,
        "_captured_payload",
        lambda prefix: payload if prefix == "runtime_seal" else b"bootstrap",
    )
    recomputations = 0

    def verify_live_closure() -> dict[str, object]:
        nonlocal recomputations
        recomputations += 1
        return {"runtime_seal": seal}

    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        verify_live_closure,
    )
    assert preformal._verify_live_bootstrap_custody() == seal
    assert recomputations == 1

    assurance = seal["assurance"]
    assert isinstance(assurance, dict)
    assurance["external_attestation"] = True
    payload = _canonical(seal)
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="captured runtime seal is malformed",
    ):
        preformal._verify_live_bootstrap_custody()


def test_live_bootstrap_custody_translates_exact_recomputation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = _runtime_seal(Path(sys.executable))
    payload = _canonical(seal)
    monkeypatch.setattr(
        preformal,
        "_captured_payload",
        lambda prefix: payload if prefix == "runtime_seal" else b"bootstrap",
    )

    def reject_live_closure() -> dict[str, object]:
        raise RuntimeError("package ownership differs")

    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        reject_live_closure,
    )
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="live runtime closure differs from its pre-import bootstrap seal",
    ):
        preformal._verify_live_bootstrap_custody()


def test_live_bootstrap_custody_rejects_mismatched_recomputed_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = _runtime_seal(Path(sys.executable))
    payload = _canonical(seal)
    monkeypatch.setattr(
        preformal,
        "_captured_payload",
        lambda prefix: payload if prefix == "runtime_seal" else b"bootstrap",
    )
    mismatched = dict(seal)
    mismatched["protocol_version"] = "unexpected"
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: {"runtime_seal": mismatched},
    )

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="recomputed runtime closure returned a different captured seal",
    ):
        preformal._verify_live_bootstrap_custody()


def _prepare_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_command: str | None = None,
    stderr_command: str | None = None,
) -> tuple[Path, list[str], dict[str, Path]]:
    directory = tmp_path / "v1.16.0" / "preformal"
    monkeypatch.setattr(
        preformal,
        "PREFORMAL_REPORT_PATH",
        directory / preformal.REPORT_NAME,
    )
    calls: list[str] = []
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    development_closure = inputs / "development-closure.json"
    review_path = inputs / "review.json"
    runtime_executable = Path(sys.executable)
    seal = _runtime_seal(runtime_executable)
    runtime_seal_path = _write_completed_runtime_seal(inputs, seal)
    development_closure.write_bytes(
        _canonical(_development_closure(inputs / "runtime-producer"))
    )
    review_path.write_bytes(_canonical(_review()))
    closure_attempt = inputs / "development-closure-attempt"
    closure_attempt.mkdir()
    closure_terminal = closure_attempt / "operator-attempt.json"
    closure_terminal.write_bytes(_canonical({"accepted": True}))
    closure_completion = operator_module.outer_completion_marker(
        closure_terminal
    )
    closure_completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(closure_terminal, closure_completion)

    monkeypatch.setattr(preformal, "REVIEW_PATH", review_path)
    monkeypatch.setattr(
        preformal,
        "RUNTIME_SEAL_PATH",
        runtime_seal_path,
    )
    monkeypatch.setattr(
        preformal,
        "DEVELOPMENT_CLOSURE_PATH",
        development_closure,
    )
    monkeypatch.setattr(
        preformal,
        "CLOSURE_ATTEMPT_PATH",
        closure_attempt,
    )
    monkeypatch.setattr(
        preformal,
        "_git_identity",
        lambda *, environment: dict(_GIT_IDENTITY),
    )
    monkeypatch.setattr(
        preformal,
        "_capture_qa_closure",
        lambda *, executable, environment: _qa_closure(),
    )
    monkeypatch.setattr(
        preformal,
        "_validated_runtime_seal",
        lambda path, *, runtime_executable: (
            dict(seal),
            dict(_RUNTIME_ENVIRONMENT),
        ),
    )
    monkeypatch.setattr(
        preformal,
        "verify_prospective_review",
        lambda path: dict(_review()),
    )
    def run_command(
        specification: preformal.CommandSpec,
        *,
        environment: dict[str, str],
    ) -> tuple[int, bytes, bytes]:
        assert not directory.exists()
        expected_environment = (
            preformal._sanitized_environment()
            if specification.role == "qa"
            else _RUNTIME_ENVIRONMENT
        )
        assert environment == expected_environment
        calls.append(specification.name)
        exit_code = 7 if specification.name == failed_command else 0
        stdout = (
            _canonical(_runtime_conformance())
            if specification.name
            == "runtime-bootstrap-inventory-conformance"
            else (
                _canonical(
                    _accepted_closure_evidence(
                        development_closure=development_closure,
                        closure_terminal=closure_terminal,
                        closure_completion=closure_completion,
                    )
                )
                if specification.name
                == "runtime-accepted-closure-evidence"
                else f"stdout:{specification.name}\n".encode()
            )
        )
        stderr = (
            f"stderr:{specification.name}\n".encode()
            if specification.name == stderr_command
            else b""
        )
        return (
            exit_code,
            stdout,
            stderr,
        )

    monkeypatch.setattr(preformal, "_run_command", run_command)
    return (
        directory / preformal.REPORT_NAME,
        calls,
        {
            "runtime_executable": runtime_executable,
            "runtime_seal": runtime_seal_path,
            "development_closure": development_closure,
            "closure_attempt": closure_attempt,
            "prospective_review": review_path,
        },
    )


def _generate(report_path: Path, inputs: dict[str, Path]) -> Path:
    return preformal.generate_preformal_report(
        report_path,
        runtime_executable=inputs["runtime_executable"],
        runtime_seal=inputs["runtime_seal"],
        development_closure=inputs["development_closure"],
        closure_attempt=inputs["closure_attempt"],
        prospective_review=inputs["prospective_review"],
    )


def _projection_implementation_rows() -> list[dict[str, object]]:
    relative_paths = {
        preformal.SOURCE_RELATIVE_PATH,
        "bench/world_model_lifecycle/launch_bootstrap.py",
        "bench/world_model_lifecycle/producer_bootstrap.py",
        *preformal._test_files("test_epistemic_*.py", label="epistemic"),
        *preformal._test_files("test_world_model_*.py", label="WM-001"),
    }
    rows: list[dict[str, object]] = []
    for relative in sorted(relative_paths):
        payload = (preformal.REPO / relative).read_bytes()
        rows.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def _prepare_preserved_preformal_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    inventory = _result_free_inventory()
    audit_execution: dict[str, object] = {
        "restart_runtime_conformance_report_sha256": "3" * 64,
        "restart_runtime_execution_receipt_sha256": "4" * 64,
        "restart_runtime_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "restart_runtime_repeat_count": 3,
        "restart_runtime_path_descriptor_equal": True,
    }
    runtime_receipt = _runtime_conformance()
    runtime_receipt["inventory"] = inventory
    runtime_receipt["inventory_sha256"] = hashlib.sha256(
        preformal._canonical_json_bytes(inventory)
    ).hexdigest()
    runtime_receipt["conformance_sha256"] = hashlib.sha256(
        preformal._canonical_json_bytes(audit_execution)
    ).hexdigest()
    for name in (
        "restart_runtime_conformance_report_sha256",
        "restart_runtime_execution_receipt_sha256",
        "restart_runtime_support_files",
        "restart_runtime_repeat_count",
        "restart_runtime_path_descriptor_equal",
    ):
        runtime_receipt[name] = copy.deepcopy(audit_execution[name])
    monkeypatch.setattr(
        sys.modules[__name__],
        "_runtime_conformance",
        lambda device="cpu": {**copy.deepcopy(runtime_receipt), "device": device},
    )

    implementation_without_review = _projection_implementation_rows()
    review = _review(implementation_without_review)
    review_path = inputs["prospective_review"]
    review_path.write_bytes(_canonical(review))
    monkeypatch.setattr(
        preformal,
        "verify_prospective_review",
        lambda _path: copy.deepcopy(review),
    )

    seal_path = inputs["runtime_seal"]
    seal = json.loads(seal_path.read_bytes())
    seal["standard_library"] = inventory["standard_library"]
    seal["package_roots"] = inventory["package_roots"]
    seal["package_ownership"] = inventory["package_ownership"]
    seal_path.write_bytes(_canonical(seal))
    monkeypatch.setattr(
        preformal,
        "_validated_runtime_seal",
        lambda _path, *, runtime_executable: (
            copy.deepcopy(seal),
            dict(_RUNTIME_ENVIRONMENT),
        ),
    )

    _generate(report_path, inputs)
    report = binding_module.verify_canonical_machine_test_report(report_path)
    assert report == _read_report(report_path)
    report_payload = report_path.read_bytes()
    log_rows, log_payloads = binding_module._capture_preformal_logs(
        report_path,
        report,
    )
    assert len(log_rows) == len(log_payloads) == 20
    snapshot = {
        report_path.name: report_payload,
        **{
            str(row["path"]): payload
            for row, payload in zip(log_rows, log_payloads, strict=True)
        },
    }
    assert len(snapshot) == 21

    review_payload = review_path.read_bytes()
    implementation_rows = sorted(
        [
            *implementation_without_review,
            {
                "path": preformal.REVIEW_RELATIVE_PATH,
                "bytes": len(review_payload),
                "sha256": hashlib.sha256(review_payload).hexdigest(),
            },
        ],
        key=lambda row: str(row["path"]),
    )
    assert report["prospective_review"] == review
    assert report["generator_source_before"] == next(
        row
        for row in implementation_rows
        if row["path"] == preformal.SOURCE_RELATIVE_PATH
    )
    execution_sources = {
        filename: binding_module.sha256_file(
            Path(binding_module.__file__).with_name(filename)
        )
        for filename in binding_module.EXECUTION_SOURCE_FILES
    }
    execution_sources["producer_bootstrap.py"] = seal[
        "bootstrap_source_sha256"
    ]

    package = report_path.parent.parent / "binding-package"
    package.mkdir()
    for filename, payload in snapshot.items():
        (package / filename).write_bytes(payload)
    assert {
        path.name: path.read_bytes()
        for path in package.iterdir()
        if path.is_file()
    } == snapshot
    copied_report = package / report_path.name
    copied_report_payload = copied_report.read_bytes()
    copied_report_value = _read_report(copied_report)
    copied_log_rows = binding_module.preformal_log_rows(
        copied_report,
        copied_report_value,
    )
    assert copied_log_rows == log_rows
    accepted_row = next(
        row
        for row in copied_report_value["commands"]
        if row["name"] == "runtime-accepted-closure-evidence"
    )
    accepted_receipt = json.loads(
        (package / accepted_row["stdout"]["file"]).read_bytes()
    )
    runtime_executable = copied_report_value["runtime_executable_before"]
    source: dict[str, object] = {
        "git_commit": copied_report_value["git_before"]["commit"],
        "git_tree": copied_report_value["git_before"]["tree"],
        "worktree_clean": True,
        "implementation_files": implementation_rows,
        "execution_source_sha256": execution_sources,
        "test_report_file": copied_report.name,
        "test_report_bytes": len(copied_report_payload),
        "test_report_sha256": hashlib.sha256(
            copied_report_payload
        ).hexdigest(),
        "test_log_files": copied_log_rows,
    }
    binding: dict[str, object] = {
        "source": source,
        "dependencies": {
            **inventory,
            "python_executable": runtime_executable["invocation_path"],
            "python_executable_sha256": runtime_executable["sha256"],
        },
        "runtime": {
            "device": copied_report_value["device"],
            "process_environment": dict(_RUNTIME_ENVIRONMENT),
        },
        "development_qualification": {
            "closure_bytes": copied_report_value["input_files_before"][
                "development_closure"
            ]["bytes"],
            "closure_sha256": accepted_receipt[
                "development_closure_sha256"
            ],
            "producer_manifest_sha256": accepted_receipt[
                "producer_manifest_sha256"
            ],
            "raw_result_sha256": accepted_receipt["raw_result_sha256"],
        },
        "audit_execution": audit_execution,
    }
    (package / "formal-binding.json").write_bytes(b"{}\n")
    (package / "coverage-conformance-fixture.json").write_bytes(b"{}\n")
    return copied_report, binding, snapshot


def _refresh_preserved_projection_binding(
    report_path: Path,
    binding: dict[str, object],
) -> dict[str, Any]:
    report = _read_report(report_path)
    payload = report_path.read_bytes()
    source = binding["source"]
    assert isinstance(source, dict)
    source["test_report_bytes"] = len(payload)
    source["test_report_sha256"] = hashlib.sha256(payload).hexdigest()
    source["test_log_files"] = binding_module.preformal_log_rows(
        report_path,
        report,
    )
    return report


def _strict_audit_execution_fixture(
    *,
    repository: Path,
    inventory: dict[str, object],
) -> tuple[dict[str, object], dict[str, bytes]]:
    python_path = Path(sys.executable)
    python_identity = {
        "executable": sys.executable,
        "resolved_executable": str(python_path.resolve(strict=True)),
        "sha256": hashlib.sha256(python_path.read_bytes()).hexdigest(),
        "version": list(sys.version_info[:3]),
    }
    bootstrap = audit_runner_module.bootstrap_source_bytes()
    bootstrap_sha256 = hashlib.sha256(bootstrap).hexdigest()
    auditor = verify_module.HERE / "artifact_audit.py"
    auditor_sha256 = hashlib.sha256(auditor.read_bytes()).hexdigest()
    request_identity = "1" * 64
    request_payload = _canonical(
        {"schema": "prospect.wm001.prebinding-conformance-request.v2"}
    )
    path_runtime = {
        "source": {"mode": "path"},
        "support_files": [],
        "python": python_identity,
    }
    descriptor_runtime = copy.deepcopy(path_runtime)
    descriptor_runtime["source"]["mode"] = "descriptor"
    path_runtime_payload = _canonical(path_runtime)
    descriptor_runtime_payload = _canonical(descriptor_runtime)
    path_invocation_payload = _canonical(
        {
            "runtime_manifest_sha256": hashlib.sha256(
                path_runtime_payload
            ).hexdigest()
        }
    )
    descriptor_invocation_payload = _canonical(
        {
            "runtime_manifest_sha256": hashlib.sha256(
                descriptor_runtime_payload
            ).hexdigest()
        }
    )
    conformance_payload = _canonical(
        {
            "schema": "prospect.wm001.prebinding-conformance.v2",
            "request_sha256": request_identity,
            "passed": True,
        }
    )
    empty_stream = {
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    repeat_count = 3
    modes = ["path"] * repeat_count + ["descriptor"] * repeat_count

    def execution_row(
        ordinal: int,
        mode: str,
    ) -> dict[str, object]:
        runtime_payload = (
            path_runtime_payload
            if mode == "path"
            else descriptor_runtime_payload
        )
        invocation_payload = (
            path_invocation_payload
            if mode == "path"
            else descriptor_invocation_payload
        )
        return {
            "ordinal": ordinal,
            "source_mode": mode,
            "returncode": 0,
            "stdout": {
                "bytes": len(conformance_payload),
                "sha256": hashlib.sha256(conformance_payload).hexdigest(),
            },
            "stderr": dict(empty_stream),
            "runtime_manifest": {
                "bytes": len(runtime_payload),
                "sha256": hashlib.sha256(runtime_payload).hexdigest(),
            },
            "invocation_manifest": {
                "bytes": len(invocation_payload),
                "sha256": hashlib.sha256(invocation_payload).hexdigest(),
            },
            "bootstrap_sha256": bootstrap_sha256,
            "auditor_source_sha256": auditor_sha256,
            "support_files": [],
            "auditor_report_passed": True,
        }

    prebinding_rows = [
        execution_row(ordinal, mode)
        for ordinal, mode in enumerate(modes, start=1)
    ]
    prebinding_receipt_payload = _canonical(
        {
            "schema": "prospect.wm001.audit-conformance-receipt.v1",
            "repeat_count": repeat_count,
            "execution_count": len(prebinding_rows),
            "executions": prebinding_rows,
            "report_sha256": hashlib.sha256(conformance_payload).hexdigest(),
            "path_descriptor_byte_identical": True,
            "execution_conformance_passed": True,
        }
    )
    support_sources = (
        ("producer_bootstrap.py", verify_module.HERE / "producer_bootstrap.py"),
        ("protocol.json", verify_module.PROTOCOL_PATH),
        (
            "schemas/raw-result.schema.json",
            verify_module.RESULT_SCHEMA_PATH,
        ),
    )
    support_rows = [
        {
            "path": relative,
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        for relative, source in support_sources
    ]
    safe_environment = {
        name: value
        for name, value in _RUNTIME_ENVIRONMENT.items()
        if name != "PATH"
    }
    outcome_runtime = {
        "schema": "prospect.wm001.audit-runtime-manifest.v1",
        "assurance": copy.deepcopy(binding_module.ASSURANCE),
        "bootstrap_sha256": bootstrap_sha256,
        "python": python_identity,
        "required_flags": dict(preformal._RUNTIME_FLAGS),
        "source": {
            "mode": "descriptor",
            "path": "artifact_audit.py",
            "bytes": auditor.stat().st_size,
            "sha256": auditor_sha256,
        },
        "support_files": support_rows,
        "closure_import_roots": inventory["package_roots"],
        "standard_library": inventory["standard_library"],
        "environment": safe_environment,
        "limits": {
            "timeout_seconds": 600,
            "stdout_bytes": 64 << 20,
            "stderr_bytes": 16 << 20,
        },
    }
    outcome_runtime_payload = _canonical(outcome_runtime)
    restart_report = {
        "schema": "prospect.wm001.restart-runtime-conformance.v1",
        "protocol_version": "1.16.0",
        "support_files": support_rows,
        "branches": {
            "development": {
                "source_block_present": False,
                "captured_bootstrap_bound": True,
                "passed": True,
            },
            "formal": {
                "source_block_present": True,
                "source_snapshot_bound": True,
                "captured_bootstrap_bound": True,
                "passed": True,
            },
        },
        "negative_cases": [
            {"case_id": case_id, "rejected": True}
            for case_id in (
                "missing-bootstrap-support",
                "extra-bootstrap-support",
                "mutated-bootstrap-identity",
                "development-formal-branch-substitution",
                "formal-development-branch-substitution",
            )
        ],
        "failure_code": None,
        "passed": True,
    }
    restart_report_payload = _canonical(restart_report)
    path_restart_runtime = copy.deepcopy(outcome_runtime)
    path_restart_runtime["source"]["mode"] = "path"
    restart_runtime_payloads = {
        "path": _canonical(path_restart_runtime),
        "descriptor": outcome_runtime_payload,
    }
    restart_arguments = [
        "--restart-runtime-conformance",
        "--producer-bootstrap",
        "@captured/producer_bootstrap.py",
        "--expected-producer-bootstrap-sha256",
        support_rows[0]["sha256"],
    ]
    restart_invocation_payloads = {
        mode: _canonical(
            {
                "schema": "prospect.wm001.audit-invocation-manifest.v1",
                "runtime_manifest_sha256": hashlib.sha256(
                    runtime_payload
                ).hexdigest(),
                "working_directory": str(repository),
                "auditor_argv": restart_arguments,
            }
        )
        for mode, runtime_payload in restart_runtime_payloads.items()
    }
    restart_rows = [
        {
            "ordinal": ordinal,
            "source_mode": mode,
            "returncode": 0,
            "stdout": {
                "bytes": len(restart_report_payload),
                "sha256": hashlib.sha256(restart_report_payload).hexdigest(),
            },
            "stderr": dict(empty_stream),
            "runtime_manifest": {
                "bytes": len(restart_runtime_payloads[mode]),
                "sha256": hashlib.sha256(
                    restart_runtime_payloads[mode]
                ).hexdigest(),
            },
            "invocation_manifest": {
                "bytes": len(restart_invocation_payloads[mode]),
                "sha256": hashlib.sha256(
                    restart_invocation_payloads[mode]
                ).hexdigest(),
            },
            "bootstrap_sha256": bootstrap_sha256,
            "auditor_source_sha256": auditor_sha256,
            "support_files": support_rows,
            "auditor_report_passed": True,
        }
        for ordinal, mode in enumerate(modes, start=1)
    ]
    restart_receipt_payload = _canonical(
        {
            "schema": "prospect.wm001.audit-conformance-receipt.v1",
            "repeat_count": repeat_count,
            "execution_count": len(restart_rows),
            "executions": restart_rows,
            "report_sha256": hashlib.sha256(
                restart_report_payload
            ).hexdigest(),
            "path_descriptor_byte_identical": True,
            "execution_conformance_passed": True,
        }
    )
    prefixed_payloads = {
        "bootstrap_source": bootstrap,
        "prebinding_request": request_payload,
        "prebinding_path_runtime_manifest": path_runtime_payload,
        "prebinding_descriptor_runtime_manifest": descriptor_runtime_payload,
        "prebinding_path_invocation_manifest": path_invocation_payload,
        "prebinding_descriptor_invocation_manifest": (
            descriptor_invocation_payload
        ),
        "prebinding_conformance_report": conformance_payload,
        "prebinding_execution_receipt": prebinding_receipt_payload,
        "outcome_runtime_manifest": outcome_runtime_payload,
        "restart_runtime_conformance_report": restart_report_payload,
        "restart_runtime_execution_receipt": restart_receipt_payload,
    }
    stems = {
        "bootstrap_source": ("audit-bootstrap", ".py"),
        "prebinding_request": ("audit-prebinding-request", ".json"),
        "prebinding_path_runtime_manifest": (
            "audit-prebinding-path-runtime",
            ".json",
        ),
        "prebinding_descriptor_runtime_manifest": (
            "audit-prebinding-descriptor-runtime",
            ".json",
        ),
        "prebinding_path_invocation_manifest": (
            "audit-prebinding-path-invocation",
            ".json",
        ),
        "prebinding_descriptor_invocation_manifest": (
            "audit-prebinding-descriptor-invocation",
            ".json",
        ),
        "prebinding_conformance_report": (
            "audit-prebinding-conformance",
            ".json",
        ),
        "prebinding_execution_receipt": (
            "audit-prebinding-execution-receipt",
            ".json",
        ),
        "outcome_runtime_manifest": ("audit-outcome-runtime", ".json"),
        "restart_runtime_conformance_report": (
            "audit-restart-runtime-conformance",
            ".json",
        ),
        "restart_runtime_execution_receipt": (
            "audit-restart-runtime-execution-receipt",
            ".json",
        ),
    }
    payloads: dict[str, bytes] = {}
    block: dict[str, object] = {
        "runner_source_sha256": hashlib.sha256(
            (verify_module.HERE / "audit_runner.py").read_bytes()
        ).hexdigest(),
        "auditor_source_sha256": auditor_sha256,
        "adjudicator_source_sha256": hashlib.sha256(
            (verify_module.HERE / "adjudication.py").read_bytes()
        ).hexdigest(),
        "prebinding_request_identity_sha256": request_identity,
        "restart_runtime_support_files": [row[0] for row in support_sources],
        "restart_runtime_repeat_count": repeat_count,
        "restart_runtime_path_descriptor_equal": True,
        "outcome_source_mode": "descriptor",
        "outcome_support_files": [row[0] for row in support_sources],
        "outcome_argv_role": [
            "<canonical-producer-root>",
            "--producer-bootstrap",
            "<captured-producer-bootstrap>",
        ],
        "outcome_working_directory": str(repository),
        "interpreter_flags": ["-I", "-S", "-B"],
        "repeat_count": repeat_count,
        "path_descriptor_equal": True,
        "passed": True,
    }
    for prefix, payload in prefixed_payloads.items():
        stem, suffix = stems[prefix]
        filename = binding_module._content_addressed_filename(
            stem,
            payload,
            suffix,
        )
        payloads[filename] = payload
        block[f"{prefix}_file"] = filename
        block[f"{prefix}_bytes"] = len(payload)
        block[f"{prefix}_sha256"] = hashlib.sha256(payload).hexdigest()
    assert len(payloads) == 11
    return block, payloads


def test_test_discovery_rejects_untracked_or_ignored_matching_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    tracked = tests / "test_world_model_tracked.py"
    ignored = tests / "test_world_model_ignored.py"
    tracked.write_text("# tracked\n", encoding="utf-8")
    ignored.write_text("# ignored\n", encoding="utf-8")
    monkeypatch.setattr(preformal, "REPO", tmp_path)
    monkeypatch.setattr(
        preformal,
        "_tracked_implementation_paths",
        lambda **_kwargs: ("tests/test_world_model_tracked.py",),
    )

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="untracked, ignored, missing, or aliased",
    ):
        preformal._test_files(
            "test_world_model_*.py",
            label="WM-001",
        )


def test_generation_requires_completely_absent_versioned_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, calls, inputs = _prepare_generation(
        tmp_path,
        monkeypatch,
    )
    report_path.parent.mkdir(parents=True)
    (report_path.parent / "unbound.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="version namespace",
    ):
        _generate(report_path, inputs)
    assert calls == []


def test_interrupted_preformal_publication_leaves_a_nonretryable_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, calls, inputs = _prepare_generation(
        tmp_path,
        monkeypatch,
    )
    real_write = preformal._atomic_write_exclusive
    writes = 0

    def interrupted_write(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected publication interruption")
        real_write(path, payload)

    monkeypatch.setattr(
        preformal,
        "_atomic_write_exclusive",
        interrupted_write,
    )
    with pytest.raises(OSError, match="injected publication interruption"):
        _generate(report_path, inputs)

    staging = report_path.parent.parent / (
        f".{report_path.parent.name}.staging"
    )
    assert calls == list(preformal._COMMAND_NAMES)
    assert not report_path.parent.exists()
    assert staging.is_dir()
    with pytest.raises(
        FileExistsError,
        match="hidden one-shot claim",
    ):
        _generate(report_path, inputs)


def test_preformal_claim_is_fsynced_before_the_first_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(
        tmp_path,
        monkeypatch,
    )
    fsynced: list[Path] = []
    real_run = preformal._run_command

    monkeypatch.setattr(
        preformal,
        "_fsync_directory",
        lambda path: fsynced.append(path),
    )

    def checked_run(
        specification: preformal.CommandSpec,
        *,
        environment: dict[str, str],
    ) -> tuple[int, bytes, bytes]:
        assert report_path.parent.parent in fsynced
        return real_run(specification, environment=environment)

    monkeypatch.setattr(preformal, "_run_command", checked_run)
    _generate(report_path, inputs)

    assert fsynced[:2] == [
        report_path.parent.parent.parent,
        report_path.parent.parent,
    ]


def test_live_verifier_rejects_unbound_bundle_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    (report_path.parent / "unbound.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="missing or extra members",
    ):
        preformal.verify_preformal_report(report_path)


def test_live_verifier_rejects_alternate_bootstrap_input_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    alternate = tmp_path / "alternate-launch-bootstrap.py"
    alternate.write_bytes(preformal.LAUNCH_BOOTSTRAP_PATH.read_bytes())
    identity = preformal._file_identity(
        alternate,
        label="alternate launch bootstrap",
        expected_nlink=1,
    )
    report["input_files_before"]["launch_bootstrap"] = identity
    report["input_files_after"]["launch_bootstrap"] = identity
    _rewrite_report(report_path, report)

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="not canonical",
    ):
        preformal.verify_preformal_report(report_path)


def test_live_verifier_rejects_after_input_numeric_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    report["input_files_after"] = copy.deepcopy(
        report["input_files_before"]
    )
    byte_count = report["input_files_after"]["launch_bootstrap"]["bytes"]
    report["input_files_after"]["launch_bootstrap"]["bytes"] = float(
        byte_count
    )
    _rewrite_report(report_path, report)

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match=(
            "launch_bootstrap after identity is malformed|"
            "input file identities are incomplete or changed"
        ),
    ):
        preformal.verify_preformal_report(report_path)


def test_live_verifier_rejects_distinct_two_link_closure_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    terminal = Path(
        report["input_files_before"]["closure_attempt_terminal"]["path"]
    )
    completion = Path(
        report["input_files_before"]["closure_outer_completion"]["path"]
    )
    payload = terminal.read_bytes()
    completion.unlink()
    completion.write_bytes(payload)
    os.link(terminal, tmp_path / "terminal-extra.json")
    os.link(completion, tmp_path / "completion-extra.json")

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="not canonical",
    ):
        preformal.verify_preformal_report(report_path)


def test_live_verifier_rejects_distinct_two_link_runtime_seal_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    seal = inputs["runtime_seal"]
    completion = operator_module.outer_completion_marker(seal)
    payload = seal.read_bytes()
    completion.unlink()
    completion.write_bytes(payload)
    os.link(seal, tmp_path / "seal-extra.json")
    os.link(completion, tmp_path / "seal-completion-extra.json")

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="not canonical",
    ):
        preformal.verify_preformal_report(report_path)


def test_exact_preformal_contract_has_ten_role_separated_commands() -> None:
    specifications = preformal.required_commands(
        runtime_executable_path=sys.executable,
    )

    assert [row.name for row in specifications] == list(preformal._COMMAND_NAMES)
    assert [row.role for row in specifications] == ["qa"] * 8 + ["runtime"] * 2
    assert "bench/world_model_lifecycle/verify.py" not in specifications[3].argv
    assert specifications[6].argv[-2:] == (
        "tests/test_world_model_audit_runner.py",
        "tests/test_world_model_prebinding_audit.py",
    )
    assert specifications[7].argv[1:5] == (
        "-I",
        "-B",
        "-m",
        "bench.world_model_lifecycle.preformal",
    )
    assert specifications[8].argv[-5:] == (
        "accepted-closure-evidence",
        "--development-closure",
        str(preformal.DEVELOPMENT_CLOSURE_PATH),
        "--closure-attempt",
        str(preformal.CLOSURE_ATTEMPT_PATH),
    )


def test_generates_and_strictly_verifies_exact_preformal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-enter-report")
    report_path, calls, inputs = _prepare_generation(tmp_path, monkeypatch)

    assert _generate(report_path, inputs) == report_path
    report = preformal.verify_preformal_report(report_path)

    expected_names = list(preformal._COMMAND_NAMES)
    assert calls == expected_names
    assert [row["name"] for row in report["commands"]] == expected_names
    assert [row["role"] for row in report["commands"]] == ["qa"] * 8 + [
        "runtime"
    ] * 2
    assert report["schema"] == "prospect.wm001.preformal-test-report.v2"
    assert report["all_pass"] is True
    assert report["qa_closure_before"] == report["qa_closure_after"]
    assert report["input_files_before"] == report["input_files_after"]
    qa_packages = {
        row["name"]
        for row in report["qa_closure_before"]["distributions"]
    }
    command_10 = next(
        row
        for row in report["commands"]
        if row["name"] == "runtime-bootstrap-inventory-conformance"
    )
    runtime_output = json.loads(
        (
            report_path.parent / command_10["stdout"]["file"]
        ).read_bytes()
    )
    runtime_packages = {
        row["name"] for row in runtime_output["inventory"]["packages"]
    }
    assert {"mypy", "pytest"} <= qa_packages
    assert {"mypy", "pytest"}.isdisjoint(runtime_packages)
    assert qa_packages != runtime_packages
    variables = {
        row["name"]: row["value"]
        for row in report["qa_environment"]["variables"]
    }
    assert "AWS_SECRET_ACCESS_KEY" not in variables
    assert variables["PYGAME_HIDE_SUPPORT_PROMPT"] == "hide"
    assert variables["PYTHONNOUSERSITE"] == "1"
    assert variables["SDL_AUDIODRIVER"] == "dsp"
    logs = {
        path.name
        for path in report_path.parent.iterdir()
        if path.name.startswith(preformal.LOG_PREFIX)
    }
    assert len(logs) == 20


def test_nonempty_command_1_stderr_prevents_preformal_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, _, inputs = _prepare_generation(
        tmp_path,
        monkeypatch,
        stderr_command="protocol-seal-continuity",
    )
    exit_code = preformal.main(
        [
            "generate-report",
            "--output",
            str(report_path),
            "--runtime-executable",
            str(inputs["runtime_executable"]),
            "--runtime-seal",
            str(inputs["runtime_seal"]),
            "--development-closure",
            str(inputs["development_closure"]),
            "--closure-attempt",
            str(inputs["closure_attempt"]),
            "--prospective-review",
            str(inputs["prospective_review"]),
        ]
    )
    generation = json.loads(capsys.readouterr().out)
    report = _read_report(report_path)

    assert exit_code == 1
    assert generation["passed"] is False
    assert generation["failed_commands"] == []
    assert generation["failed_checks"] == [
        "command_01_stderr_not_empty"
    ]
    assert report["commands"][0]["passed"] is True
    assert report["all_pass"] is False
    assert preformal._semantic_failure_checks(
        report_path.parent,
        report,
    ) == ["command_01_stderr_not_empty"]
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="protocol-seal-continuity stderr is not exactly empty",
    ):
        preformal.verify_preformal_report(report_path)


def test_preserved_preformal_bound_projection_accepts_mixed_sidecars_without_ambient_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, snapshot = (
        _prepare_preserved_preformal_projection(tmp_path, monkeypatch)
    )
    assert len(snapshot) == 21

    with pytest.raises(
        RuntimeError,
        match="complete fixed preformal check set",
    ):
        binding_module.verify_canonical_machine_test_report(copied_report)

    def forbidden_ambient(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bound verifier reopened ambient origin evidence")

    for name in (
        "_capture_qa_closure",
        "_executable_identity",
        "_file_identity",
        "_git_identity",
        "_source_identity",
        "_validated_runtime_seal",
        "_verified_closure_member_digests",
        "verify_prospective_review",
        "verify_recorded_preformal_report",
    ):
        monkeypatch.setattr(preformal, name, forbidden_ambient)

    verified = binding_module.verify_bound_machine_test_report(
        copied_report,
        binding,
    )
    assert verified["all_pass"] is True


def test_preserved_preformal_bound_projection_rejects_extra_reserved_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    (copied_report.parent / "preformal-unbound.log").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="extra reserved members"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_log_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    source = binding["source"]
    first_log = copied_report.parent / source["test_log_files"][0]["path"]
    first_log.write_bytes(first_log.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="bytes changed"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param("missing", id="missing"),
        pytest.param("symlink", id="symlink"),
        pytest.param("hardlink", id="hardlink"),
    ),
)
def test_preserved_preformal_bound_projection_rejects_missing_or_aliased_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    source = binding["source"]
    first_log = copied_report.parent / source["test_log_files"][0]["path"]
    payload = first_log.read_bytes()
    first_log.unlink()
    if mutation != "missing":
        alias_source = tmp_path / f"{mutation}-source.log"
        alias_source.write_bytes(payload)
        if mutation == "symlink":
            first_log.symlink_to(alias_source)
        else:
            first_log.hardlink_to(alias_source)

    with pytest.raises(
        RuntimeError,
        match=(
            "missing, aliased, or non-canonical"
            if mutation != "hardlink"
            else "1-link custody contract"
        ),
    ):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_reordered_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    report["commands"][0], report["commands"][1] = (
        report["commands"][1],
        report["commands"][0],
    )
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(RuntimeError, match="command 1 differs"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


@pytest.mark.parametrize("mutation", ("reordered", "duplicate"))
def test_preserved_preformal_bound_projection_rejects_changed_log_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    source = binding["source"]
    logs = source["test_log_files"]
    if mutation == "reordered":
        logs[0], logs[1] = logs[1], logs[0]
    else:
        logs[1] = copy.deepcopy(logs[0])

    with pytest.raises(RuntimeError, match="command-log custody differs"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_noncanonical_report_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    payload = (json.dumps(report, indent=2) + "\n").encode()
    copied_report.write_bytes(payload)
    source = binding["source"]
    source["test_report_bytes"] = len(payload)
    source["test_report_sha256"] = hashlib.sha256(payload).hexdigest()

    with pytest.raises(RuntimeError, match="not canonical JSON"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_rehashed_closure_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    accepted_row = next(
        row
        for row in report["commands"]
        if row["name"] == "runtime-accepted-closure-evidence"
    )
    old_log = copied_report.parent / accepted_row["stdout"]["file"]
    receipt = json.loads(old_log.read_bytes())
    receipt["producer_manifest_sha256"] = "0" * 64
    payload = _canonical(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    filename = preformal._log_filename(
        accepted_row["ordinal"],
        accepted_row["name"],
        "stdout",
        digest,
    )
    old_log.unlink()
    (copied_report.parent / filename).write_bytes(payload)
    accepted_row["stdout"] = {
        "file": filename,
        "bytes": len(payload),
        "sha256": digest,
    }
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(
        RuntimeError,
        match="accepted-closure receipt differs from the binding",
    ):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_boolean_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    report["commands"][0]["ordinal"] = True
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(RuntimeError, match="command 1 differs"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


@pytest.mark.parametrize(
    "command_name",
    (
        "protocol-seal-continuity",
        "runtime-accepted-closure-evidence",
    ),
)
def test_preserved_preformal_bound_projection_rejects_any_command_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    command = next(
        row
        for row in report["commands"]
        if row["name"] == command_name
    )
    old_path = copied_report.parent / command["stderr"]["file"]
    diagnostic = b"unexpected runtime diagnostic\n"
    digest = hashlib.sha256(diagnostic).hexdigest()
    filename = preformal._log_filename(
        command["ordinal"],
        command["name"],
        "stderr",
        digest,
    )
    old_path.unlink()
    (copied_report.parent / filename).write_bytes(diagnostic)
    command["stderr"] = {
        "file": filename,
        "bytes": len(diagnostic),
        "sha256": digest,
    }
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(
        RuntimeError,
        match=rf"command {command['ordinal']} stderr",
    ):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


@pytest.mark.parametrize(
    ("target", "message"),
    (
        pytest.param("repository", "repository identity", id="repository"),
        pytest.param("runtime-seal", "input path", id="input"),
        pytest.param("review", "input path", id="review"),
    ),
)
def test_preserved_preformal_bound_projection_rejects_rebound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    if target == "repository":
        report["repository_cwd"] = str(tmp_path / "alternate-repository")
    else:
        input_name = "runtime_seal" if target == "runtime-seal" else "prospective_review"
        for inputs in (
            report["input_files_before"],
            report["input_files_after"],
        ):
            inputs[input_name]["path"] = str(tmp_path / f"alternate-{target}")
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(RuntimeError, match=message):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_after_numeric_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    report["input_files_after"] = copy.deepcopy(
        report["input_files_before"]
    )
    byte_count = report["input_files_after"]["launch_bootstrap"]["bytes"]
    report["input_files_after"]["launch_bootstrap"]["bytes"] = float(
        byte_count
    )
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(
        RuntimeError,
        match="input identity is malformed|input identities are incomplete",
    ):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


@pytest.mark.parametrize(
    "mutation",
    ("extra-field", "python-version", "python-version-bool"),
)
def test_preserved_preformal_bound_projection_rejects_rehashed_seal_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    seal = report["runtime_seal"]
    if mutation == "extra-field":
        seal["unexpected"] = True
    elif mutation == "python-version-bool":
        seal["python"]["version"][0] = True
    else:
        seal["python"]["version"] = [0, 0, 0]
    seal_payload = _canonical(seal)
    seal_identity = {
        "bytes": len(seal_payload),
        "sha256": hashlib.sha256(seal_payload).hexdigest(),
    }
    for inputs in (
        report["input_files_before"],
        report["input_files_after"],
    ):
        inputs["runtime_seal"].update(seal_identity)
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(RuntimeError, match="runtime seal differs"):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_preserved_preformal_bound_projection_rejects_reordered_review_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_report, binding, _ = _prepare_preserved_preformal_projection(
        tmp_path,
        monkeypatch,
    )
    report = _read_report(copied_report)
    review = report["prospective_review"]
    rows = review["implementation_files"]
    rows[0], rows[1] = rows[1], rows[0]
    review["implementation_manifest_sha256"] = hashlib.sha256(
        preformal._canonical_json_bytes(rows)
    ).hexdigest()
    review_payload = _canonical(review)
    review_identity = {
        "bytes": len(review_payload),
        "sha256": hashlib.sha256(review_payload).hexdigest(),
    }
    for inputs in (
        report["input_files_before"],
        report["input_files_after"],
    ):
        inputs["prospective_review"].update(review_identity)
    implementation_rows = binding["source"]["implementation_files"]
    review_row = next(
        row
        for row in implementation_rows
        if row["path"] == preformal.REVIEW_RELATIVE_PATH
    )
    review_row.update(review_identity)
    _rewrite_report(copied_report, report)
    _refresh_preserved_projection_binding(copied_report, binding)

    with pytest.raises(
        RuntimeError,
        match="prospective review is malformed or misbound",
    ):
        binding_module.verify_bound_machine_test_report(
            copied_report,
            binding,
        )


def test_capture_preformal_logs_retains_once_captured_bytes_after_origin_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = tmp_path / "command.stdout.log"
    stderr = tmp_path / "command.stderr.log"
    stdout_payload = b"captured stdout\n"
    stdout.write_bytes(stdout_payload)
    stderr.write_bytes(b"")
    report = {
        "commands": [
            {
                "stdout": {
                    "file": stdout.name,
                    "bytes": len(stdout_payload),
                    "sha256": hashlib.sha256(stdout_payload).hexdigest(),
                },
                "stderr": {
                    "file": stderr.name,
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            }
        ]
    }
    report_path = tmp_path / "report.json"
    report_path.write_bytes(_canonical(report))
    original_reader = binding_module._stable_regular_payload
    mutated = False

    def read_then_mutate(
        path: Path,
        *,
        label: str,
        expected_nlink: int = 1,
    ) -> bytes:
        nonlocal mutated
        payload = original_reader(
            path,
            label=label,
            expected_nlink=expected_nlink,
        )
        if path == stdout and not mutated:
            stdout.write_bytes(b"changed after capture\n")
            mutated = True
        return payload

    monkeypatch.setattr(
        binding_module,
        "_stable_regular_payload",
        read_then_mutate,
    )
    rows, payloads = binding_module._capture_preformal_logs(
        report_path,
        report,
    )

    assert mutated is True
    assert stdout.read_bytes() != stdout_payload
    assert rows[0] == {
        "path": stdout.name,
        "bytes": len(stdout_payload),
        "sha256": hashlib.sha256(stdout_payload).hexdigest(),
    }
    assert payloads == [stdout_payload, b""]


def test_create_formal_binding_then_real_verify_binding_preserved_preformal_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    inventory = _result_free_inventory()
    package_names = [
        "python",
        *sorted(verify_module.REQUIRED_PACKAGES - {"python"}),
    ]
    inventory["packages"] = [
        {
            "name": name,
            "version": "3.12.0" if name == "python" else "1.0",
            "distribution_sha256": f"{index + 1:x}" * 64,
            "declared_file_count": 1,
            "editable": False,
        }
        for index, name in enumerate(package_names)
    ]
    fixture_repository = tmp_path / "strict-repository"
    fixture_repository.mkdir()
    audit_execution, audit_payloads = _strict_audit_execution_fixture(
        repository=fixture_repository,
        inventory=inventory,
    )

    implementation_without_review = _projection_implementation_rows()
    review = _review(implementation_without_review)
    inputs["prospective_review"].write_bytes(_canonical(review))
    monkeypatch.setattr(
        preformal,
        "verify_prospective_review",
        lambda _path: copy.deepcopy(review),
    )

    seal_path = inputs["runtime_seal"]
    seal = json.loads(seal_path.read_bytes())
    seal["standard_library"] = inventory["standard_library"]
    seal["package_roots"] = inventory["package_roots"]
    seal["package_ownership"] = inventory["package_ownership"]
    seal_path.write_bytes(_canonical(seal))
    monkeypatch.setattr(
        preformal,
        "_validated_runtime_seal",
        lambda _path, *, runtime_executable: (
            copy.deepcopy(seal),
            dict(_RUNTIME_ENVIRONMENT),
        ),
    )

    runtime_receipt = _runtime_conformance()
    runtime_receipt["inventory"] = inventory
    runtime_receipt["inventory_sha256"] = hashlib.sha256(
        preformal._canonical_json_bytes(inventory)
    ).hexdigest()
    runtime_receipt["conformance_sha256"] = hashlib.sha256(
        preformal._canonical_json_bytes(audit_execution)
    ).hexdigest()
    for name in (
        "restart_runtime_conformance_report_sha256",
        "restart_runtime_execution_receipt_sha256",
        "restart_runtime_support_files",
        "restart_runtime_repeat_count",
        "restart_runtime_path_descriptor_equal",
    ):
        runtime_receipt[name] = audit_execution[name]
    monkeypatch.setattr(
        sys.modules[__name__],
        "_runtime_conformance",
        lambda device="cpu": {
            **copy.deepcopy(runtime_receipt),
            "device": device,
        },
    )
    _generate(report_path, inputs)
    canonical_report = binding_module.verify_canonical_machine_test_report(
        report_path
    )
    review_payload = inputs["prospective_review"].read_bytes()
    implementation_rows = sorted(
        [
            *implementation_without_review,
            {
                "path": preformal.REVIEW_RELATIVE_PATH,
                "bytes": len(review_payload),
                "sha256": hashlib.sha256(review_payload).hexdigest(),
            },
        ],
        key=lambda row: str(row["path"]),
    )
    assert canonical_report["prospective_review"] == review
    for row in implementation_without_review:
        relative = str(row["path"])
        destination = fixture_repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((preformal.REPO / relative).read_bytes())
    review_destination = (
        fixture_repository / preformal.REVIEW_RELATIVE_PATH
    )
    review_destination.parent.mkdir(parents=True, exist_ok=True)
    review_destination.write_bytes(review_payload)

    closure = json.loads(inputs["development_closure"].read_bytes())
    lockfile = fixture_repository / "requirements-wm001.lock"
    lockfile.write_text(
        "fixture==1 --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "binding-package" / "formal-binding.json"

    monkeypatch.setattr(binding_module, "REPO", fixture_repository)
    monkeypatch.setattr(verify_module, "REPO", fixture_repository)
    monkeypatch.setattr(binding_module, "LOCKFILE", lockfile)
    monkeypatch.setattr(
        verify_module,
        "verify_protocol",
        lambda: {
            "lanes": {
                "formal": {
                    "master_seeds": list(verify_module.FORMAL_SEEDS),
                }
            }
        },
    )
    monkeypatch.setattr(binding_module, "source_is_clean", lambda: True)
    monkeypatch.setattr(
        binding_module,
        "require_formal_python_flags",
        lambda: dict(preformal._RUNTIME_FLAGS),
    )
    monkeypatch.setattr(
        binding_module,
        "require_formal_process_environment",
        lambda: dict(_RUNTIME_ENVIRONMENT),
    )
    monkeypatch.setattr(
        binding_module,
        "run_pendulum_conformance",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        binding_module,
        "_created_conformance_satisfies_formal_contract",
        lambda _report: True,
    )
    monkeypatch.setattr(
        binding_module,
        "run_independent_phase_oscillator_conformance",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        binding_module,
        "run_coverage_conformance",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(
        binding_module,
        "verify_development_closure",
        lambda _path, *, include_result_qualification=False: (
            (
                copy.deepcopy(closure),
                b'{"fixture":"development-result-qualification"}\n',
            )
            if include_result_qualification
            else copy.deepcopy(closure)
        ),
    )
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: copy.deepcopy(implementation_rows),
    )
    monkeypatch.setattr(
        binding_module,
        "verify_installed_source_snapshot",
        lambda: None,
    )
    monkeypatch.setattr(binding_module, "package_roots", lambda: (tmp_path,))
    monkeypatch.setattr(
        binding_module,
        "package_root_inventory",
        lambda _root: copy.deepcopy(inventory["package_roots"][0]),
    )
    monkeypatch.setattr(
        binding_module,
        "standard_library_inventory",
        lambda: copy.deepcopy(inventory["standard_library"]),
    )
    monkeypatch.setattr(
        binding_module,
        "package_root_ownership",
        lambda: copy.deepcopy(inventory["package_ownership"]),
    )
    monkeypatch.setattr(
        binding_module,
        "installed_package_rows",
        lambda: copy.deepcopy(inventory["packages"]),
    )
    monkeypatch.setattr(binding_module, "verify_lockfile_rows", lambda _rows: None)
    monkeypatch.setattr(
        binding_module,
        "build_bound_audit_execution",
        lambda **_kwargs: (
            copy.deepcopy(audit_execution),
            copy.deepcopy(audit_payloads),
        ),
    )
    monkeypatch.setattr(
        binding_module,
        "git_output",
        lambda *arguments: (
            _GIT_IDENTITY["commit"]
            if arguments == ("rev-parse", "HEAD")
            else _GIT_IDENTITY["tree"]
        ),
    )
    monkeypatch.setattr(
        binding_module,
        "distribution_sha256",
        lambda _name: "c" * 64,
    )
    monkeypatch.setattr(
        binding_module,
        "checkpoint_implementation_sha256",
        lambda: "d" * 64,
    )
    monkeypatch.setattr(
        binding_module,
        "manifest_schema_sha256",
        lambda: "e" * 64,
    )
    monkeypatch.setattr(
        binding_module.torch,
        "use_deterministic_algorithms",
        lambda _enabled: None,
    )
    monkeypatch.setattr(
        binding_module.torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(binding_module.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(
        binding_module.torch,
        "get_num_interop_threads",
        lambda: 1,
    )

    created = binding_module.create_formal_binding(
        output_path=output,
        test_report_path=report_path,
        development_closure_path=inputs["development_closure"],
        device="cpu",
    )
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    verify_module._validate_json_schema(
        created,
        schema,
        label="created preserved-preformal binding",
    )
    monkeypatch.setattr(
        verify_module,
        "_verify_pendulum_conformance_report",
        lambda _report: None,
    )
    monkeypatch.setattr(
        verify_module,
        "_expected_oscillator_conformance",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(
        verify_module,
        "_verify_bound_coverage_runtime_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        verify_module,
        "_verify_coverage_conformance_report",
        lambda *_args, **_kwargs: None,
    )
    member_digests = {
        str(row["path"]): str(row["sha256"])
        for row in closure["qualification_archive"]["members"]
    }
    monkeypatch.setattr(
        verify_module,
        "_recorded_development_closure_identity",
        lambda _path: (
            copy.deepcopy(closure),
            copy.deepcopy(created["development_qualification"]),
            copy.deepcopy(member_digests),
        ),
    )

    verified = verify_module.verify_binding(output)
    assert verified == created


def test_recorded_report_verifier_uses_explicit_qa_not_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import binding

    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    recorded_qa = report["qa_executable_before"]["invocation_path"]
    original_identity = preformal._executable_identity
    identity_paths: list[str] = []
    closure_paths: list[str] = []

    def explicit_identity(
        path: str | Path | None = None,
        *,
        implementation: str | None = None,
        version: str | None = None,
    ) -> dict[str, object]:
        if path is None:
            raise AssertionError(
                "recorded verifier consulted the caller executable"
            )
        identity_paths.append(str(path))
        return original_identity(
            path,
            implementation=implementation,
            version=version,
        )

    def explicit_closure(
        *,
        executable: str,
        environment: dict[str, str],
    ) -> dict[str, object]:
        assert environment == preformal._environment_from_identity(
            report["qa_environment"],
            role="qa",
        )
        closure_paths.append(executable)
        return _qa_closure()

    monkeypatch.setattr(
        preformal,
        "_executable_identity",
        explicit_identity,
    )
    monkeypatch.setattr(
        preformal,
        "_capture_qa_closure",
        explicit_closure,
    )

    def forbidden_ambient_inventory(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        raise AssertionError(
            "recorded verifier re-entered an ambient runtime inventory API"
        )

    for name in (
        "installed_package_rows",
        "package_roots",
        "verify_development_closure",
        "verify_lockfile_rows",
    ):
        monkeypatch.setattr(
            binding,
            name,
            forbidden_ambient_inventory,
        )

    assert (
        preformal.verify_recorded_preformal_report(report_path)["all_pass"]
        is True
    )
    assert identity_paths
    assert set(identity_paths) == {recorded_qa}
    assert closure_paths == [recorded_qa]


def test_preformal_generation_and_verification_reject_same_name_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, calls, inputs = _prepare_generation(tmp_path, monkeypatch)
    sibling_directory = tmp_path / "alternate"
    sibling_directory.mkdir()
    sibling = sibling_directory / preformal.REPORT_NAME

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="sole canonical",
    ):
        _generate(sibling, inputs)
    assert calls == []

    _generate(report_path, inputs)
    sibling.write_bytes(report_path.read_bytes())
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="missing or aliased",
    ):
        preformal.verify_preformal_report(sibling)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("argv", ["/untrusted/python", "-c", "pass"]),
        ("cwd", "/tmp"),
        ("environment_sha256", "0" * 64),
        ("role", "runtime"),
        ("name", "ruff"),
        ("ordinal", 99),
    ),
)
def test_verifier_rejects_caller_selected_command_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    report["commands"][0][field] = replacement
    _rewrite_report(report_path, report)

    with pytest.raises(preformal.PreformalEvidenceError):
        preformal.verify_preformal_report(report_path)


def test_verifier_rejects_changed_qa_closure_and_runtime_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    report["qa_closure_before"]["distributions"][0]["version"] = "tampered"
    _rewrite_report(report_path, report)
    with pytest.raises(preformal.PreformalEvidenceError, match="QA closure"):
        preformal.verify_preformal_report(report_path)

    report = _read_report(report_path)
    report["qa_closure_before"] = _qa_closure()
    _rewrite_report(report_path, report)
    inputs["development_closure"].write_bytes(b"changed")
    with pytest.raises(preformal.PreformalEvidenceError, match="input changed"):
        preformal.verify_preformal_report(report_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("passed", False),
        ("development_closure_sha256", "4" * 64),
        ("producer_manifest_sha256", "7" * 64),
        ("raw_result_sha256", "8" * 64),
        ("closure_attempt_manifest_sha256", "9" * 64),
        ("closure_outer_completion_sha256", "a" * 64),
    ),
)
def test_verifier_semantically_rejects_mutated_accepted_closure_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    row = next(
        item
        for item in report["commands"]
        if item["name"] == "runtime-accepted-closure-evidence"
    )
    old_path = report_path.parent / row["stdout"]["file"]
    value = json.loads(old_path.read_bytes())
    value[field] = replacement
    payload = _canonical(value)
    digest = hashlib.sha256(payload).hexdigest()
    new_name = preformal._log_filename(
        row["ordinal"],
        row["name"],
        "stdout",
        digest,
    )
    new_path = report_path.parent / new_name
    new_path.write_bytes(payload)
    old_path.unlink()
    row["stdout"] = {
        "file": new_name,
        "bytes": len(payload),
        "sha256": digest,
    }
    _rewrite_report(report_path, report)

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="accepted_closure_semantics_failed",
    ):
        preformal.verify_preformal_report(report_path)


def test_qa_closure_parser_rejects_digest_role_and_member_mutations(
    tmp_path: Path,
) -> None:
    pristine = _development_closure(tmp_path / "runtime-producer")

    mutations: list[tuple[str, Any]] = []

    invalid_digest = json.loads(json.dumps(pristine))
    invalid_digest["qualification_archive"]["members"][0][
        "sha256"
    ] = "z" * 64
    mutations.append(("digest", invalid_digest))

    unsafe_role = json.loads(json.dumps(pristine))
    unsafe_role["audit_runtime_manifest_member"] = (
        "evidence/../audit-runtime.json"
    )
    mutations.append(("role", unsafe_role))

    missing_member = json.loads(json.dumps(pristine))
    missing_member["qualification_archive"]["members"] = [
        row
        for row in missing_member["qualification_archive"]["members"]
        if row["path"] != "producer/result.json"
    ]
    mutations.append(("member", missing_member))

    unordered = json.loads(json.dumps(pristine))
    unordered["qualification_archive"]["members"].reverse()
    mutations.append(("order", unordered))

    for name, closure in mutations:
        path = tmp_path / f"{name}.json"
        path.write_bytes(_canonical(closure))
        with pytest.raises(preformal.PreformalEvidenceError):
            preformal._verified_closure_member_digests(path)


def test_qa_closure_parser_never_reenters_runtime_inventory_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import binding

    closure = _development_closure(tmp_path / "runtime-producer")
    execution = closure["producer_execution"]
    assert isinstance(execution, dict)
    roots = execution["package_roots"]
    assert isinstance(roots, list)
    assert "pytest" not in json.dumps(roots)
    assert preformal.importlib.metadata.version("pytest")
    path = tmp_path / "development-closure.json"
    path.write_bytes(_canonical(closure))

    def forbidden_runtime_reentry(_path: Path) -> dict[str, object]:
        raise AssertionError("QA parser re-entered runtime inventory verifier")

    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        forbidden_runtime_reentry,
    )

    parsed, manifest_sha256, result_sha256 = (
        preformal._verified_closure_member_digests(path)
    )
    assert parsed == closure
    assert manifest_sha256 == "5" * 64
    assert result_sha256 == "6" * 64


def test_command_10_semantics_require_empty_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    row = next(
        item
        for item in report["commands"]
        if item["name"]
        == "runtime-bootstrap-inventory-conformance"
    )
    old_path = report_path.parent / row["stderr"]["file"]
    payload = b"unexpected runtime diagnostic\n"
    digest = hashlib.sha256(payload).hexdigest()
    new_name = preformal._log_filename(
        row["ordinal"],
        row["name"],
        "stderr",
        digest,
    )
    (report_path.parent / new_name).write_bytes(payload)
    old_path.unlink()
    row["stderr"] = {
        "file": new_name,
        "bytes": len(payload),
        "sha256": digest,
    }
    _rewrite_report(report_path, report)

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="stderr is not exactly empty",
    ):
        preformal.verify_preformal_report(report_path)


@pytest.mark.parametrize(
    "mutation",
    ("inventory-structure", "inventory-digest", "fresh-report"),
)
def test_command_10_semantics_reject_recorded_identity_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    row = next(
        item
        for item in report["commands"]
        if item["name"]
        == "runtime-bootstrap-inventory-conformance"
    )
    old_path = report_path.parent / row["stdout"]["file"]
    value = json.loads(old_path.read_bytes())
    if mutation == "inventory-structure":
        value["inventory"]["package_ownership"]["file_count"] += 1
        value["inventory_sha256"] = hashlib.sha256(
            preformal._canonical_json_bytes(value["inventory"])
        ).hexdigest()
    elif mutation == "inventory-digest":
        value["inventory_sha256"] = "f" * 64
    else:
        value["fresh_runtime_identity_conformance"]["passed"] = False
        value["fresh_runtime_identity_conformance_sha256"] = (
            hashlib.sha256(
                preformal._canonical_json_bytes(
                    value["fresh_runtime_identity_conformance"]
                )
            ).hexdigest()
        )
    payload = _canonical(value)
    digest = hashlib.sha256(payload).hexdigest()
    new_name = preformal._log_filename(
        row["ordinal"],
        row["name"],
        "stdout",
        digest,
    )
    (report_path.parent / new_name).write_bytes(payload)
    old_path.unlink()
    row["stdout"] = {
        "file": new_name,
        "bytes": len(payload),
        "sha256": digest,
    }
    _rewrite_report(report_path, report)

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="runtime_conformance_semantics_failed",
    ):
        preformal.verify_preformal_report(report_path)


def test_verifier_rejects_tampered_missing_extra_and_aliased_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    _generate(report_path, inputs)
    report = _read_report(report_path)
    first_log = report_path.parent / report["commands"][0]["stdout"]["file"]
    original = first_log.read_bytes()
    first_log.write_bytes(b"tampered")
    with pytest.raises(preformal.PreformalEvidenceError, match="identity changed"):
        preformal.verify_preformal_report(report_path)

    first_log.write_bytes(original)
    extra = report_path.parent / f"{preformal.LOG_PREFIX}99-extra.stdout.{'0' * 64}.log"
    extra.write_bytes(b"")
    with pytest.raises(preformal.PreformalEvidenceError, match="missing or extra"):
        preformal.verify_preformal_report(report_path)
    extra.unlink()

    first_log.unlink()
    target = tmp_path / "external.log"
    target.write_bytes(original)
    first_log.symlink_to(target)
    with pytest.raises(preformal.PreformalEvidenceError):
        preformal.verify_preformal_report(report_path)


def test_failed_command_is_preserved_but_never_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, calls, inputs = _prepare_generation(
        tmp_path,
        monkeypatch,
        failed_command="mypy-wm001",
    )
    _generate(report_path, inputs)
    report = _read_report(report_path)

    assert calls == list(preformal._COMMAND_NAMES)
    assert report["all_pass"] is False
    assert [
        (row["name"], row["exit_code"])
        for row in report["commands"]
        if row["passed"] is False
    ] == [("mypy-wm001", 7)]
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match=r"failed commands: 4:mypy-wm001\(exit=7\)",
    ):
        preformal.verify_preformal_report(report_path)


@pytest.mark.parametrize(
    ("failed_command", "expected_exit", "expected_passed"),
    [
        (None, 0, True),
        ("pytest-wm001", 1, False),
    ],
)
def test_generate_report_cli_returns_the_report_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_command: str | None,
    expected_exit: int,
    expected_passed: bool,
) -> None:
    report_path, _, inputs = _prepare_generation(
        tmp_path,
        monkeypatch,
        failed_command=failed_command,
    )

    exit_code = preformal.main(
        [
            "generate-report",
            "--output",
            str(report_path),
            "--runtime-executable",
            str(inputs["runtime_executable"]),
            "--runtime-seal",
            str(inputs["runtime_seal"]),
            "--development-closure",
            str(inputs["development_closure"]),
            "--closure-attempt",
            str(inputs["closure_attempt"]),
            "--prospective-review",
            str(inputs["prospective_review"]),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == expected_exit
    assert output["schema"] == (
        "prospect.wm001.preformal-test-report-generation.v2"
    )
    assert output["passed"] is expected_passed
    assert output["failed_checks"] == []
    if failed_command is None:
        assert output["failed_commands"] == []
    else:
        assert output["failed_commands"] == [
            {
                "ordinal": 6,
                "name": "pytest-wm001",
                "exit_code": 7,
            }
        ]


def test_generate_report_cli_truthfully_names_noncommand_identity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    identities = [
        dict(_GIT_IDENTITY),
        {**_GIT_IDENTITY, "worktree_clean": False},
    ]
    monkeypatch.setattr(
        preformal,
        "_git_identity",
        lambda *, environment: identities.pop(0),
    )

    exit_code = preformal.main(
        [
            "generate-report",
            "--output",
            str(report_path),
            "--runtime-executable",
            str(inputs["runtime_executable"]),
            "--runtime-seal",
            str(inputs["runtime_seal"]),
            "--development-closure",
            str(inputs["development_closure"]),
            "--closure-attempt",
            str(inputs["closure_attempt"]),
            "--prospective-review",
            str(inputs["prospective_review"]),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["passed"] is False
    assert output["failed_commands"] == []
    assert output["failed_checks"] == [
        "post_run_worktree_not_clean",
        "pre_post_identity_drift",
    ]


def test_generate_report_cli_truthfully_rejects_semantic_runtime_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preformal,
        "_verified_closure_member_digests",
        lambda _path: ({}, "7" * 64, "8" * 64),
    )

    exit_code = preformal.main(
        [
            "generate-report",
            "--output",
            str(report_path),
            "--runtime-executable",
            str(inputs["runtime_executable"]),
            "--runtime-seal",
            str(inputs["runtime_seal"]),
            "--development-closure",
            str(inputs["development_closure"]),
            "--closure-attempt",
            str(inputs["closure_attempt"]),
            "--prospective-review",
            str(inputs["prospective_review"]),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    report = _read_report(report_path)

    assert exit_code == 1
    assert output["passed"] is False
    assert output["failed_commands"] == []
    assert output["failed_checks"] == [
        "accepted_closure_semantics_failed",
    ]
    assert report["all_pass"] is False


@pytest.mark.parametrize(
    ("verifier_name", "failure_id"),
    (
        (
            "_accepted_closure_evidence_from_report",
            "accepted_closure_semantics_failed",
        ),
        (
            "_runtime_bootstrap_conformance_from_report",
            "runtime_conformance_semantics_failed",
        ),
    ),
)
def test_unexpected_semantic_exception_becomes_exact_failed_report_and_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verifier_name: str,
    failure_id: str,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)

    def unexpected_failure(
        directory: Path,
        report: object,
    ) -> dict[str, object]:
        raise RuntimeError("injected QA/runtime inventory mismatch")

    monkeypatch.setattr(
        preformal,
        verifier_name,
        unexpected_failure,
    )

    exit_code = preformal.main(
        [
            "generate-report",
            "--output",
            str(report_path),
            "--runtime-executable",
            str(inputs["runtime_executable"]),
            "--runtime-seal",
            str(inputs["runtime_seal"]),
            "--development-closure",
            str(inputs["development_closure"]),
            "--closure-attempt",
            str(inputs["closure_attempt"]),
            "--prospective-review",
            str(inputs["prospective_review"]),
        ]
    )
    envelope = json.loads(capsys.readouterr().out)
    report = _read_report(report_path)

    assert exit_code == 1
    assert report["all_pass"] is False
    assert envelope["passed"] is False
    assert envelope["failed_commands"] == []
    assert envelope["failed_checks"] == [failure_id]


def test_semantic_checks_do_not_swallow_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _, inputs = _prepare_generation(tmp_path, monkeypatch)

    def interrupted(
        directory: Path,
        report: object,
    ) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        preformal,
        "_accepted_closure_evidence_from_report",
        interrupted,
    )
    with pytest.raises(KeyboardInterrupt):
        _generate(report_path, inputs)


def test_public_runtime_parser_excludes_obsolete_development_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        preformal._runtime_parser().parse_args(
            [
                "development-evidence",
                "--development-closure",
                "/tmp/closure.json",
            ]
        )

    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_versioned_preformal_bundle_ignores_retained_prior_version_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_root = tmp_path / "development"
    legacy_root.mkdir()
    legacy_report = (
        legacy_root / "preformal-test-report-v1.11.0.json"
    )
    legacy_report.write_text(
        "{}\n",
        encoding="utf-8",
    )
    legacy_log = (
        legacy_root
        / "preformal-v1.11.0-command-01.stdout.legacy.log"
    )
    legacy_log.write_bytes(
        b"retained failure evidence"
    )
    legacy_link = legacy_root / "retained-report-completion.json"
    os.link(legacy_report, legacy_link)

    def legacy_fingerprint() -> list[tuple[object, ...]]:
        rows: list[tuple[object, ...]] = []
        for path in (legacy_report, legacy_log, legacy_link):
            metadata = path.stat()
            payload = path.read_bytes()
            rows.append(
                (
                    path.name,
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_size,
                    hashlib.sha256(payload).hexdigest(),
                )
            )
        return rows

    before = legacy_fingerprint()
    bundle_root = legacy_root / "v1.16.0" / "preformal"
    bundle_root.parent.mkdir(parents=True)
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    report_path, _, inputs = _prepare_generation(fixture_root, monkeypatch)
    monkeypatch.setattr(
        preformal,
        "PREFORMAL_REPORT_PATH",
        bundle_root / preformal.REPORT_NAME,
    )

    assert (
        preformal.generate_preformal_report(
            bundle_root / preformal.REPORT_NAME,
            runtime_executable=inputs["runtime_executable"],
            runtime_seal=inputs["runtime_seal"],
            development_closure=inputs["development_closure"],
            closure_attempt=inputs["closure_attempt"],
            prospective_review=inputs["prospective_review"],
        )
        == bundle_root / preformal.REPORT_NAME
    )
    assert report_path.parent != bundle_root
    assert legacy_fingerprint() == before
    assert os.path.samefile(legacy_report, legacy_link)


def test_dirty_worktree_blocks_before_any_command_or_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, calls, inputs = _prepare_generation(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preformal,
        "_git_identity",
        lambda *, environment: {**_GIT_IDENTITY, "worktree_clean": False},
    )

    with pytest.raises(preformal.PreformalEvidenceError, match="clean worktree"):
        _generate(report_path, inputs)
    assert calls == []
    assert not report_path.parent.exists()


def test_prospective_review_requires_exact_manifest_and_independent_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"x = 1\n")
    rows = [
        {
            "path": "source.py",
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    review = {
        "schema": preformal.REVIEW_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "implementation_files": rows,
        "implementation_manifest_sha256": hashlib.sha256(
            preformal._canonical_json_bytes(rows)
        ).hexdigest(),
        "reviewer": {
            "kind": "independent-adversarial-referee",
            "identifier": "independent-test-referee",
        },
        "disposition": "accepted",
        "unresolved_blockers": [],
        "findings": [],
    }
    path = tmp_path / "review.json"
    path.write_bytes(_canonical(review))
    monkeypatch.setattr(preformal, "_implementation_files", lambda **kwargs: rows)
    monkeypatch.setattr(
        preformal,
        "_installed_preformal_identity",
        lambda: {
            "path": "/qa/bench/world_model_lifecycle/preformal.py",
            "bytes": 1,
            "sha256": "f" * 64,
        },
    )

    assert preformal.verify_prospective_review(path) == review
    review["implementation_files"] = []
    path.write_bytes(_canonical(review))
    with pytest.raises(preformal.PreformalEvidenceError, match="accepted exact-source"):
        preformal.verify_prospective_review(path)


def test_prospective_manifest_binds_v1160_plan_and_runbook() -> None:
    paths = {
        str(row["path"])
        for row in preformal._implementation_files()
    }

    assert {
        "docs/wm001-v1160-confirmation-plan.md",
        "docs/wm001-v1160-operator-runbook.md",
    } <= paths


def test_isolated_review_rejects_stale_installed_preformal_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    live = repository / preformal.SOURCE_RELATIVE_PATH
    live.parent.mkdir(parents=True)
    live.write_bytes(b"live reviewed source\n")
    installed_root = tmp_path / "site-packages"
    installed = installed_root / preformal.SOURCE_RELATIVE_PATH
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"stale installed source\n")

    class Distribution:
        files = (Path(preformal.SOURCE_RELATIVE_PATH),)

        @staticmethod
        def read_text(name: str) -> str | None:
            if name == "direct_url.json":
                return json.dumps({"dir_info": {"editable": False}})
            return None

        @staticmethod
        def locate_file(entry: object) -> Path:
            return installed_root / str(entry)

    monkeypatch.setattr(preformal, "REPO", repository)
    monkeypatch.setattr(
        preformal.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="differs from the live reviewed source",
    ):
        preformal._installed_preformal_identity()

    installed.write_bytes(live.read_bytes())
    identity = preformal._installed_preformal_identity()
    assert identity["sha256"] == hashlib.sha256(live.read_bytes()).hexdigest()


def _install_captured_descriptor(
    path: Path,
    payload: bytes,
    *,
    prefix: str,
    link_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    path.write_bytes(payload)
    if link_count == 2:
        os.link(path, path.with_name(f"{path.name}.completion"))
    descriptor = os.open(path, os.O_RDONLY)
    identity = preformal._descriptor_identity(os.fstat(descriptor))
    monkeypatch.setattr(
        sys,
        f"_prospect_wm001_{prefix}_fd",
        descriptor,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        f"_prospect_wm001_{prefix}_payload",
        payload,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        f"_prospect_wm001_{prefix}_identity",
        identity,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        f"_prospect_wm001_{prefix}_sha256",
        hashlib.sha256(payload).hexdigest(),
        raising=False,
    )
    return descriptor


def _formal_runtime_custody() -> dict[str, object]:
    return {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "experiment_id": "WM-001",
        "assurance": copy.deepcopy(binding_module.ASSURANCE),
        "protocol": {
            "version": "1.16.0",
            "sha256": "1" * 64,
            "raw_result_schema_sha256": "2" * 64,
            "binding_schema_sha256": "3" * 64,
        },
    }


@pytest.mark.parametrize(
    ("custody_kind", "link_count"),
    (("prospective", 2), ("formal-binding", 1)),
)
def test_live_bootstrap_custody_accepts_both_typed_runtime_seals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custody_kind: str,
    link_count: int,
) -> None:
    custody = (
        _runtime_seal(Path(sys.executable))
        if custody_kind == "prospective"
        else _formal_runtime_custody()
    )
    runtime_payload = _canonical(custody)
    runtime_descriptor = _install_captured_descriptor(
        tmp_path / "runtime-custody.json",
        runtime_payload,
        prefix="runtime_seal",
        link_count=link_count,
        monkeypatch=monkeypatch,
    )
    bootstrap_payload = b"captured bootstrap\n"
    bootstrap_descriptor = _install_captured_descriptor(
        tmp_path / "producer-bootstrap.py",
        bootstrap_payload,
        prefix="bootstrap",
        link_count=1,
        monkeypatch=monkeypatch,
    )
    recomputations = 0

    def verify_experiment_custody() -> dict[str, object]:
        nonlocal recomputations
        recomputations += 1
        return {"runtime_seal": copy.deepcopy(custody)}

    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        verify_experiment_custody,
    )
    try:
        assert preformal._verify_live_bootstrap_custody() == custody
        assert recomputations == 1
    finally:
        os.close(bootstrap_descriptor)
        os.close(runtime_descriptor)


@pytest.mark.parametrize(
    ("custody_kind", "wrong_link_count"),
    (("prospective", 1), ("formal-binding", 2)),
)
def test_live_bootstrap_custody_rejects_cross_typed_link_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custody_kind: str,
    wrong_link_count: int,
) -> None:
    custody = (
        _runtime_seal(Path(sys.executable))
        if custody_kind == "prospective"
        else _formal_runtime_custody()
    )
    descriptor = _install_captured_descriptor(
        tmp_path / "runtime-custody.json",
        _canonical(custody),
        prefix="runtime_seal",
        link_count=wrong_link_count,
        monkeypatch=monkeypatch,
    )
    try:
        with pytest.raises(
            preformal.PreformalEvidenceError,
            match="custody changed",
        ):
            preformal._captured_payload("runtime_seal")
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("custody_kind", "mutation"),
    (
        ("prospective", "protocol"),
        ("prospective", "assurance"),
        ("formal-binding", "protocol"),
        ("formal-binding", "assurance"),
    ),
)
def test_runtime_custody_requires_exact_protocol_and_assurance(
    custody_kind: str,
    mutation: str,
) -> None:
    custody = (
        _runtime_seal(Path(sys.executable))
        if custody_kind == "prospective"
        else _formal_runtime_custody()
    )
    if mutation == "protocol":
        if custody_kind == "prospective":
            custody["protocol_version"] = "1.15.0"
        else:
            custody["protocol"] = {"version": "1.15.0"}
    else:
        custody["assurance"] = {
            **copy.deepcopy(binding_module.ASSURANCE),
            "tamper_resistant": 0,
        }

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="custody|runtime seal",
    ):
        preformal._runtime_custody_value(_canonical(custody))


def test_formal_binding_custody_rejects_recomputed_experiment_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custody = _formal_runtime_custody()
    runtime_descriptor = _install_captured_descriptor(
        tmp_path / "formal-binding.json",
        _canonical(custody),
        prefix="runtime_seal",
        link_count=1,
        monkeypatch=monkeypatch,
    )
    bootstrap_descriptor = _install_captured_descriptor(
        tmp_path / "producer-bootstrap.py",
        b"captured bootstrap\n",
        prefix="bootstrap",
        link_count=1,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: {
            "runtime_seal": {
                **copy.deepcopy(custody),
                "protocol": {"version": "1.15.0"},
            }
        },
    )
    try:
        with pytest.raises(
            preformal.PreformalEvidenceError,
            match="returned a different captured seal",
        ):
            preformal._verify_live_bootstrap_custody()
    finally:
        os.close(bootstrap_descriptor)
        os.close(runtime_descriptor)


def test_captured_descriptor_custody_rejects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "seal.json"
    payload = _canonical(_runtime_seal(Path(sys.executable)))
    descriptor = _install_captured_descriptor(
        path,
        payload,
        prefix="runtime_seal",
        link_count=2,
        monkeypatch=monkeypatch,
    )
    try:
        assert preformal._captured_payload("runtime_seal") == payload
        path.write_bytes(b"changed")
        with pytest.raises(preformal.PreformalEvidenceError, match="custody changed"):
            preformal._captured_payload("runtime_seal")
    finally:
        os.close(descriptor)


def _write_outer_finalized_development_producer(
    directory: Path,
) -> tuple[Path, dict[str, object]]:
    from bench.world_model_lifecycle import artifact

    producer = directory / "producer"
    producer.mkdir()
    result_path = producer / "result.json"
    result_payload = _canonical({"fixture": "development-result"})
    result_path.write_bytes(result_payload)
    manifest: dict[str, object] = {
        "schema": "prospect.wm001.producer-manifest.v1",
        "experiment_id": "WM-001",
        "lane": "development",
        "status": "completed",
        "started_at_utc": "2026-07-19T00:00:00Z",
        "completed_at_utc": "2026-07-19T00:01:00Z",
        "error": None,
        "manifest_excludes": ["producer-manifest.json"],
        "file_count": 1,
        "files": [
            {
                "path": "result.json",
                "bytes": len(result_payload),
                "sha256": hashlib.sha256(result_payload).hexdigest(),
            }
        ],
    }
    manifest_path = producer / artifact.MANIFEST_NAME
    manifest_path.write_bytes(_canonical(manifest))
    os.link(
        manifest_path,
        operator_module.outer_completion_marker(manifest_path),
    )
    assert artifact.verify_producer_manifest(producer) == manifest
    return producer, manifest


def test_runtime_development_evidence_accepts_outer_finalized_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import binding, verify

    producer, manifest = _write_outer_finalized_development_producer(tmp_path)
    closure_path = tmp_path / "development-closure.json"
    closure_path.write_bytes(_canonical({"fixture": "closure"}))
    closure = {
        "producer_root": str(producer),
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda path: closure,
    )
    monkeypatch.setattr(
        verify,
        "verify_result",
        lambda path, binding_path: {
            "lane": "development",
            "claim_eligible": False,
        },
    )
    monkeypatch.setattr(
        preformal,
        "_verify_live_bootstrap_custody",
        lambda: {"schema": "captured-runtime"},
    )

    observed = preformal._runtime_development_evidence(closure_path)

    assert observed == {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "development-evidence",
        "passed": True,
        "development_closure_sha256": hashlib.sha256(
            closure_path.read_bytes()
        ).hexdigest(),
        "producer_manifest_sha256": hashlib.sha256(
            _canonical(manifest)
        ).hexdigest(),
        "raw_result_sha256": hashlib.sha256(
            (producer / "result.json").read_bytes()
        ).hexdigest(),
    }


def test_fresh_closure_reopen_executes_inherited_seal_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = tmp_path / "development-closure.json"
    closure.write_bytes(_canonical({"closure": True}))
    custody = {"schema": "captured-runtime"}
    challenge = "7" * 64
    child_report = {
        "schema": (
            "prospect.wm001.development-closure-fresh-reopen.v1"
        ),
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "mode": "fresh-closure-reopen",
        "challenge": challenge,
        "requesting_process_id": os.getpid(),
        "verifier_process_id": os.getpid() + 1,
        "matrix_contract_sha256": "8" * 64,
        "development_closure_sha256": "9" * 64,
        "producer_manifest_sha256": "a" * 64,
        "raw_result_sha256": "b" * 64,
        "passed": True,
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        preformal,
        "_verify_live_bootstrap_custody",
        lambda: custody,
    )
    monkeypatch.setattr(
        preformal,
        "_canonical_existing_file",
        lambda path, *, label: path,
    )
    monkeypatch.setattr(os, "urandom", lambda count: bytes.fromhex(challenge))
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_bootstrap_fd",
        11,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_runtime_seal_fd",
        12,
        raising=False,
    )
    monkeypatch.setattr(
        preformal,
        "validate_fresh_closure_reopen_report",
        lambda value, *, development_closure: value,
    )

    def run(
        command: tuple[str, ...],
        **arguments: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed.update(arguments)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_canonical(child_report),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert (
        preformal.fresh_runtime_development_closure_reopen(closure)
        == child_report
    )
    command = observed["command"]
    assert isinstance(command, tuple)
    assert command[:5] == (
        sys.executable,
        "-I",
        "-S",
        "-B",
        "/proc/self/fd/11",
    )
    assert "launch_bootstrap.py" not in " ".join(command)
    assert command[-7:] == (
        "fresh-closure-reopen",
        "--development-closure",
        str(closure),
        "--challenge",
        challenge,
        "--requesting-process-id",
        str(os.getpid()),
    )
    assert observed["pass_fds"] == (11, 12)
    assert (
        observed["timeout"]
        == preformal._FRESH_CLOSURE_REOPEN_TIMEOUT_SECONDS
    )


def test_fresh_closure_reopen_validator_uses_archive_member_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import binding

    closure_path = tmp_path / "development-closure.json"
    closure = _development_closure(tmp_path / "runtime-producer")
    closure_payload = _canonical(closure)
    closure_path.write_bytes(closure_payload)
    manifest_sha256 = "5" * 64
    result_sha256 = "6" * 64
    report = {
        "schema": (
            "prospect.wm001.development-closure-fresh-reopen.v1"
        ),
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "mode": "fresh-closure-reopen",
        "challenge": "7" * 64,
        "requesting_process_id": 101,
        "verifier_process_id": 202,
        "matrix_contract_sha256": "8" * 64,
        "development_closure_sha256": hashlib.sha256(
            closure_payload
        ).hexdigest(),
        "producer_manifest_sha256": manifest_sha256,
        "raw_result_sha256": result_sha256,
        "passed": True,
    }
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda path: pytest.fail(
            "fresh report QA validation re-entered runtime verifier"
        ),
    )
    monkeypatch.setattr(
        binding,
        "_development_matrix_contract_sha256",
        lambda: "8" * 64,
    )

    assert (
        preformal.validate_fresh_closure_reopen_report(
            report,
            development_closure=closure_path,
        )
        == report
    )
    mutated = dict(report)
    mutated["raw_result_sha256"] = "c" * 64
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="live sealed evidence",
    ):
        preformal.validate_fresh_closure_reopen_report(
            mutated,
            development_closure=closure_path,
        )


def test_fresh_identity_conformance_uses_nested_descriptor_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import binding

    custody = {"schema": "captured-runtime"}
    challenge = "7" * 64
    child_report = {
        "schema": (
            "prospect.wm001.fresh-runtime-identity-conformance.v1"
        ),
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "mode": "fresh-identity-conformance",
        "challenge": challenge,
        "requesting_process_id": os.getpid(),
        "verifier_process_id": os.getpid() + 1,
        "matrix_contract_sha256": "8" * 64,
        "passed": True,
    }
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        preformal,
        "_verify_live_bootstrap_custody",
        lambda: custody,
    )
    monkeypatch.setattr(os, "urandom", lambda count: bytes.fromhex(challenge))
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_bootstrap_fd",
        11,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_runtime_seal_fd",
        12,
        raising=False,
    )
    monkeypatch.setattr(
        binding,
        "_development_matrix_contract_sha256",
        lambda: "8" * 64,
    )

    def run(
        command: tuple[str, ...],
        **arguments: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed.update(arguments)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_canonical(child_report),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert (
        preformal.fresh_runtime_identity_conformance()
        == child_report
    )
    command = observed["command"]
    assert isinstance(command, tuple)
    assert command[:5] == (
        sys.executable,
        "-I",
        "-S",
        "-B",
        "/proc/self/fd/11",
    )
    assert "launch_bootstrap.py" not in " ".join(command)
    assert command[-5:] == (
        "fresh-identity-conformance",
        "--challenge",
        challenge,
        "--requesting-process-id",
        str(os.getpid()),
    )
    assert observed["pass_fds"] == (11, 12)


def test_fresh_identity_conformance_child_rejects_same_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preformal,
        "_verify_live_bootstrap_custody",
        lambda: {"schema": "captured-runtime"},
    )

    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="challenge",
    ):
        preformal._runtime_fresh_identity_conformance(
            challenge="7" * 64,
            requesting_process_id=os.getpid(),
        )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    (
        (2, b"", b"child failed"),
        (0, b"not canonical json\n", b""),
        (
            0,
            b"x"
            * (
                preformal._FRESH_CLOSURE_REOPEN_MAX_OUTPUT_BYTES
                + 1
            ),
            b"",
        ),
    ),
)
def test_fresh_closure_reopen_rejects_failed_or_malformed_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> None:
    closure = tmp_path / "development-closure.json"
    closure.write_bytes(_canonical({"closure": True}))
    monkeypatch.setattr(
        preformal,
        "_verify_live_bootstrap_custody",
        lambda: {"schema": "captured-runtime"},
    )
    monkeypatch.setattr(
        preformal,
        "_canonical_existing_file",
        lambda path, *, label: path,
    )
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_bootstrap_fd",
        11,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_runtime_seal_fd",
        12,
        raising=False,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )

    with pytest.raises(preformal.PreformalEvidenceError):
        preformal.fresh_runtime_development_closure_reopen(closure)


def test_fresh_closure_reopen_rejects_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = tmp_path / "development-closure.json"
    closure.write_bytes(_canonical({"closure": True}))
    monkeypatch.setattr(
        preformal,
        "_verify_live_bootstrap_custody",
        lambda: {"schema": "captured-runtime"},
    )
    monkeypatch.setattr(
        preformal,
        "_canonical_existing_file",
        lambda path, *, label: path,
    )
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_bootstrap_fd",
        11,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_runtime_seal_fd",
        12,
        raising=False,
    )

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("fresh child", 3_600)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="could not start",
    ):
        preformal.fresh_runtime_development_closure_reopen(closure)


def test_bootstrap_inventory_rehearses_gymnasium_before_final_closure_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gymnasium

    from bench.world_model_lifecycle import analysis, binding, experiment

    events: list[str] = []
    custody = {"schema": "captured-runtime"}

    def reject_outcome_path(*_: object, **__: object) -> None:
        raise AssertionError(
            "result-free rehearsal invoked an outcome-producing path"
        )

    for module, names in (
        (experiment, ("run_replicate", "run_experiment")),
        (analysis, ("analyze_result", "evaluate_gates")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, reject_outcome_path)

    class ResultFreePendulum:
        class Spec:
            id = "Pendulum-v1"

        spec = Spec()

        def reset(self, *_: object, **__: object) -> None:
            raise AssertionError("result-free rehearsal must not reset")

        def step(self, *_: object, **__: object) -> None:
            raise AssertionError("result-free rehearsal must not step")

        def close(self) -> None:
            events.append("close")

    def make(environment_id: str) -> ResultFreePendulum:
        assert environment_id == "Pendulum-v1"
        events.append("make")
        return ResultFreePendulum()

    def verify_custody() -> dict[str, str]:
        events.append("closure")
        return dict(custody)

    def require_environment() -> dict[str, str]:
        events.append("environment")
        return dict(_RUNTIME_ENVIRONMENT)

    def build_execution(**arguments: object) -> tuple[dict[str, object], dict[str, bytes]]:
        assert arguments["producer_environment"] == _RUNTIME_ENVIRONMENT
        events.append("conformance")
        restart_report = b"restart-report\n"
        restart_receipt = b"restart-receipt\n"
        return (
            {
                "repeat_count": 3,
                "path_descriptor_equal": True,
                "restart_runtime_conformance_report_file": (
                    "restart-report.json"
                ),
                "restart_runtime_conformance_report_sha256": (
                    hashlib.sha256(restart_report).hexdigest()
                ),
                "restart_runtime_execution_receipt_file": (
                    "restart-receipt.json"
                ),
                "restart_runtime_execution_receipt_sha256": (
                    hashlib.sha256(restart_receipt).hexdigest()
                ),
                "restart_runtime_support_files": [
                    "producer_bootstrap.py",
                    "protocol.json",
                    "schemas/raw-result.schema.json",
                ],
                "restart_runtime_repeat_count": 3,
                "restart_runtime_path_descriptor_equal": True,
                "passed": True,
            },
            {
                **{
                    f"payload-{index}": b""
                    for index in range(9)
                },
                "restart-report.json": restart_report,
                "restart-receipt.json": restart_receipt,
            },
        )

    monkeypatch.setattr(gymnasium, "make", make)
    monkeypatch.setattr(preformal, "_verify_live_bootstrap_custody", verify_custody)
    fresh_identity = {
        "schema": (
            "prospect.wm001.fresh-runtime-identity-conformance.v1"
        ),
        "experiment_id": "WM-001",
        "protocol_version": "1.16.0",
        "mode": "fresh-identity-conformance",
        "challenge": "7" * 64,
        "requesting_process_id": 101,
        "verifier_process_id": 202,
        "matrix_contract_sha256": "8" * 64,
        "passed": True,
    }

    def fresh_conformance() -> dict[str, object]:
        events.append("fresh-identity")
        return fresh_identity

    monkeypatch.setattr(
        preformal,
        "fresh_runtime_identity_conformance",
        fresh_conformance,
    )
    monkeypatch.setattr(binding, "require_formal_process_environment", require_environment)
    monkeypatch.setattr(
        binding,
        "verify_installed_source_snapshot",
        lambda: events.append("sources"),
    )
    monkeypatch.setattr(binding, "package_roots", lambda: ())
    monkeypatch.setattr(binding, "installed_package_rows", lambda: [])
    monkeypatch.setattr(
        binding,
        "verify_lockfile_rows",
        lambda _: events.append("lockfile"),
    )
    monkeypatch.setattr(
        binding,
        "standard_library_inventory",
        lambda: {"identity": "stdlib"},
    )
    monkeypatch.setattr(
        binding,
        "package_root_ownership",
        lambda: {"identity": "ownership"},
    )
    monkeypatch.setattr(binding, "build_bound_audit_execution", build_execution)

    report = preformal._runtime_bootstrap_inventory_conformance("cpu")

    assert report["passed"] is True
    assert report[
        "fresh_runtime_identity_conformance_sha256"
    ] == hashlib.sha256(
        preformal._canonical_json_bytes(fresh_identity)
    ).hexdigest()
    assert report["restart_runtime_repeat_count"] == 3
    assert report["restart_runtime_path_descriptor_equal"] is True
    assert events[:4] == ["closure", "make", "close", "closure"]
    assert "fresh-identity" in events
    assert events[-2:] == ["conformance", "closure"]
