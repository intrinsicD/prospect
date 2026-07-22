from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle import (
    binding as binding_module,
)
from bench.world_model_lifecycle import (
    launch_bootstrap,
    producer_bootstrap,
)
from bench.world_model_lifecycle import (
    operator as operator_module,
)
from bench.world_model_lifecycle import (
    rehearsal as rehearsal_module,
)
from bench.world_model_lifecycle.audit_runner import (
    INVOCATION_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_SCHEMA,
    AuditExecution,
    CapturedFileIdentity,
    bootstrap_source_sha256,
)

_ASSURANCE = {
    "trust_model_id": "prospect.wm001.trust-model.v1",
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}


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


def _repository_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    results = repository / "bench" / "world_model_lifecycle" / "results"
    completion_root = results / "outer-completions" / "v1.20"
    completion_root.mkdir(parents=True)
    terminal = results / "operator-v1.20" / "attempt-001" / "terminal-manifest.json"
    terminal.parent.mkdir(parents=True)
    terminal.write_bytes(
        _canonical(
            {
                "experiment_id": "WM-001",
                "schema": "prospect.wm001.operator-attempt-terminal.v1",
                "status": "accepted",
            }
        )
    )
    return repository, completion_root, terminal


def _receipt(
    terminal: Path,
    *,
    logical_exit_code: int = 0,
) -> dict[str, object]:
    payload = terminal.read_bytes()
    return {
        "schema": "prospect.wm001.outer-terminal-receipt.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.20.0",
        "assurance": dict(_ASSURANCE),
        "trust_model": "trusted-single-principal-cooperative-lock-v1",
        "terminal_path": str(terminal),
        "terminal_bytes": len(payload),
        "terminal_sha256": hashlib.sha256(payload).hexdigest(),
        "logical_exit_code": logical_exit_code,
    }


def _completion_marker(completion_root: Path, terminal: Path) -> Path:
    return completion_root / (hashlib.sha256(str(terminal).encode("utf-8")).hexdigest() + ".json")


def _zero_sha256(byte_count: int) -> str:
    """Hash sparse-fixture contents without materializing a second large file."""

    digest = hashlib.sha256()
    block = b"\0" * (1 << 20)
    remaining = byte_count
    while remaining:
        chunk_bytes = min(len(block), remaining)
        digest.update(block[:chunk_bytes])
        remaining -= chunk_bytes
    return digest.hexdigest()


def _development_producer(
    tmp_path: Path,
    *,
    files: Mapping[str, int | bytes],
) -> tuple[Path, Path, Path]:
    """Create one outer-finalized producer with sparse or literal members."""

    repository = tmp_path / "repository"
    results = repository / "bench" / "world_model_lifecycle" / "results"
    completion_root = results / "outer-completions" / "v1.20"
    completion_root.mkdir(parents=True)
    producer = results / "development" / "qualification-v1.20.0"
    producer.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for relative, value in sorted(files.items()):
        path = producer / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
            byte_count = len(value)
            digest = hashlib.sha256(value).hexdigest()
        else:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.ftruncate(descriptor, value)
            finally:
                os.close(descriptor)
            byte_count = value
            digest = _zero_sha256(value)
        rows.append(
            {
                "path": relative,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    manifest = producer / "producer-manifest.json"
    manifest.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.producer-manifest.v1",
                "experiment_id": "WM-001",
                "lane": "development",
                "status": "completed",
                "started_at_utc": "2026-01-01T00:00:00Z",
                "completed_at_utc": "2026-01-01T00:01:00Z",
                "error": None,
                "manifest_excludes": ["producer-manifest.json"],
                "file_count": len(rows),
                "files": rows,
            }
        )
    )
    os.link(manifest, _completion_marker(completion_root, manifest))
    return repository, completion_root, producer


@pytest.mark.parametrize(
    "payload",
    (
        b'{"a":1,"a":1}\n',
        b'{ "a": 1 }\n',
        b'{"a":1}',
        b'{"a":NaN}\n',
    ),
    ids=("duplicate-key", "whitespace", "missing-lf", "non-finite"),
)
def test_outer_canonical_json_parser_rejects_ambiguous_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._canonical_object(
            payload,
            label="adversarial fixture",
        )


def _formal_binding_attempt(
    repository: Path,
    completion_root: Path,
    *,
    binding: dict[str, object],
    terminal_state: str,
    producer_files: dict[str, int | bytes] | None = None,
) -> tuple[Path, Path, Path]:
    results = repository / "bench" / "world_model_lifecycle" / "results"
    report = results / "development" / "v1.20.0" / "preformal" / "preformal-test-report-v1.20.0.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    packages = [
        {
            "name": "python",
            "version": "3.12.0",
            "distribution_sha256": "1" * 64,
            "declared_file_count": 1,
            "editable": False,
        }
    ]
    package_root = {
        "semantics_id": "prospect.wm001.package-root.v2",
        "path": "/runtime/site-packages",
        "file_count": 7,
        "directory_count": 3,
        "total_bytes": 70,
        "inventory_sha256": "2" * 64,
    }
    standard_library = {
        "semantics_id": "prospect.wm001.standard-library.v2",
        "path": "/runtime/stdlib",
        "file_count": 11,
        "directory_count": 4,
        "total_bytes": 110,
        "inventory_sha256": "3" * 64,
    }
    ownership = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": package_root["path"],
        "file_count": package_root["file_count"],
        "directory_count": package_root["directory_count"],
        "shared_file_count": 0,
        "identity_sha256": "4" * 64,
    }
    provided_dependencies = binding.get("dependencies")
    if isinstance(provided_dependencies, dict):
        provided_roots = provided_dependencies.get("package_roots")
        if isinstance(provided_roots, list) and len(provided_roots) == 1 and isinstance(provided_roots[0], dict):
            package_root = provided_roots[0]
        provided_standard_library = provided_dependencies.get("standard_library")
        if isinstance(provided_standard_library, dict):
            standard_library = provided_standard_library
        provided_ownership = provided_dependencies.get("package_ownership")
        if isinstance(provided_ownership, dict):
            ownership = provided_ownership
    inventory = {
        "packages": packages,
        "package_roots": [package_root],
        "standard_library": standard_library,
        "package_ownership": ownership,
    }
    fresh_identity = {
        "schema": "prospect.wm001.fresh-runtime-identity-conformance.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.20.0",
        "mode": "fresh-identity-conformance",
        "challenge": "5" * 64,
        "requesting_process_id": 101,
        "verifier_process_id": 202,
        "matrix_contract_sha256": (launch_bootstrap._DEVELOPMENT_MATRIX_CONTRACT_SHA256),
        "passed": True,
    }
    audit_execution = {
        "schema": "prospect.wm001.bound-audit-execution.v1",
        "restart_runtime_conformance_report_sha256": "7" * 64,
        "restart_runtime_execution_receipt_sha256": "8" * 64,
        "restart_runtime_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "restart_runtime_repeat_count": 3,
        "restart_runtime_path_descriptor_equal": True,
        "repeat_count": 3,
        "path_descriptor_equal": True,
        "passed": True,
    }
    runtime_outputs = {
        "runtime-accepted-closure-evidence": {
            "schema": "prospect.wm001.preformal-runtime-check.v1",
            "mode": "accepted-closure-evidence",
            "passed": True,
            "development_closure_sha256": "9" * 64,
            "producer_manifest_sha256": "a" * 64,
            "raw_result_sha256": "b" * 64,
            "closure_attempt_manifest_sha256": "c" * 64,
            "closure_outer_completion_sha256": "d" * 64,
        },
        "runtime-bootstrap-inventory-conformance": {
            "schema": "prospect.wm001.preformal-runtime-check.v1",
            "mode": "bootstrap-inventory-conformance",
            "device": "cpu",
            "passed": True,
            "inventory": inventory,
            "inventory_sha256": launch_bootstrap._canonical_digest(inventory),
            "conformance_sha256": launch_bootstrap._canonical_digest(audit_execution),
            "fresh_runtime_identity_conformance": fresh_identity,
            "fresh_runtime_identity_conformance_sha256": (launch_bootstrap._canonical_digest(fresh_identity)),
            "restart_runtime_conformance_report_sha256": "7" * 64,
            "restart_runtime_execution_receipt_sha256": "8" * 64,
            "restart_runtime_support_files": [
                "producer_bootstrap.py",
                "protocol.json",
                "schemas/raw-result.schema.json",
            ],
            "restart_runtime_repeat_count": 3,
            "restart_runtime_path_descriptor_equal": True,
            "repeat_count": 3,
            "path_descriptor_equal": True,
        },
    }
    log_paths: dict[str, Path] = {}
    commands: list[dict[str, object]] = []
    for ordinal, name in enumerate(
        launch_bootstrap._PREFORMAL_COMMAND_NAMES,  # noqa: SLF001
        start=1,
    ):
        row: dict[str, object] = {"name": name}
        if name in runtime_outputs:
            payload = _canonical(runtime_outputs[name])
            digest = hashlib.sha256(payload).hexdigest()
            log_path = report.with_name(f"preformal-v1.20.0-command-{ordinal:02d}-{name}.stdout.{digest}.log")
            log_path.write_bytes(payload)
            log_paths[f"{name}:stdout"] = log_path
            row["stdout"] = {
                "file": log_path.name,
                "bytes": len(payload),
                "sha256": digest,
            }
            stderr_path = report.with_name(
                f"preformal-v1.20.0-command-{ordinal:02d}-{name}.stderr.{launch_bootstrap._SHA256_EMPTY}.log"
            )
            stderr_path.write_bytes(b"")
            log_paths[f"{name}:stderr"] = stderr_path
            row.update(
                {
                    "exit_code": 0,
                    "passed": True,
                    "stderr": {
                        "file": stderr_path.name,
                        "bytes": 0,
                        "sha256": launch_bootstrap._SHA256_EMPTY,
                    },
                }
            )
        commands.append(row)
    report.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.preformal-test-report.v2",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "device": "cpu",
                "all_pass": True,
                "commands": commands,
            }
        )
    )
    producer = results / "development" / "qualification-v1.20.0"
    producer.mkdir()
    values = producer_files or {
        "result.json": _canonical(
            {
                "schema": "prospect.world-model-lifecycle.raw-result.v9",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "lane": "development",
            }
        )
    }
    manifest_rows: list[dict[str, object]] = []
    for relative, value in sorted(values.items()):
        path = producer / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
            byte_count = len(value)
            digest = hashlib.sha256(value).hexdigest()
        else:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.ftruncate(descriptor, value)
            finally:
                os.close(descriptor)
            byte_count = value
            digest = _zero_sha256(value)
        manifest_rows.append(
            {
                "path": relative,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    result = producer / "result.json"
    result_manifest_row = next(row for row in manifest_rows if row["path"] == "result.json")
    result_sha256 = str(result_manifest_row["sha256"])
    producer_execution = {"fixture": "development-execution"}
    qualification_payload = _canonical(
        {
            "schema": ("prospect.wm001.development-result-qualification.v1"),
            "experiment_id": "WM-001",
            "protocol_version": "1.20.0",
            "protocol_sha256": "e" * 64,
            "raw_result_sha256": result_sha256,
            "lane": "development",
            "claim_eligible": False,
            "replicates": [
                {
                    "replicate_id": f"development-{master_seed}",
                    "master_seed": master_seed,
                    "episodes": 496,
                    "transitions": 99_200,
                    "predictive_metrics": 12,
                    "policy_runs": 20,
                    "updates": 6,
                    "optimizer_batch_manifests": 5,
                }
                for master_seed in launch_bootstrap._DEVELOPMENT_SEEDS
            ],
            "matrix_contract_sha256": (launch_bootstrap._DEVELOPMENT_MATRIX_CONTRACT_SHA256),
            "producer_execution": producer_execution,
        }
    )
    producer_terminal = producer / "producer-manifest.json"
    producer_terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.producer-manifest.v1",
                "experiment_id": "WM-001",
                "lane": "development",
                "status": "completed",
                "started_at_utc": "2026-01-01T00:00:00Z",
                "completed_at_utc": "2026-01-01T00:01:00Z",
                "error": None,
                "manifest_excludes": ["producer-manifest.json"],
                "file_count": len(manifest_rows),
                "files": manifest_rows,
            }
        )
    )
    producer_completion = _completion_marker(
        completion_root,
        producer_terminal,
    )
    os.link(producer_terminal, producer_completion)
    producer_terminal_payload = producer_terminal.read_bytes()
    producer_rows = [
        {
            "path": str(producer_terminal),
            "bytes": len(producer_terminal_payload),
            "sha256": hashlib.sha256(producer_terminal_payload).hexdigest(),
        },
        {
            "path": str(result),
            "bytes": result_manifest_row["bytes"],
            "sha256": result_sha256,
        },
    ]
    audit_attempt = results / "operator-v1.20" / "audits" / "development-audit-v1.20.0"
    audit_attempt.mkdir(parents=True)
    audit = {"schema": "test.independent-audit.v1", "passed": True}
    audit_payload = _canonical(audit)
    auditor_source = Path(binding_module.__file__).with_name("artifact_audit.py")
    support_rows = binding_module._expected_development_audit_support_rows()
    runtime_payload = _canonical(
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "execution_role": "outcome_audit",
            "assurance": dict(_ASSURANCE),
            "bootstrap_sha256": bootstrap_source_sha256(),
            "python": {},
            "required_flags": {},
            "source": {
                "mode": "descriptor",
                "path": "artifact_audit.py",
                "bytes": auditor_source.stat().st_size,
                "sha256": binding_module.sha256_file(auditor_source),
            },
            "support_files": support_rows,
            "closure_import_roots": [],
            "standard_library": {},
            "environment": {},
            "limits": {
                "timeout_seconds": 10_800,
                "stdout_bytes": 64 << 20,
                "stderr_bytes": 16 << 20,
            },
        }
    )
    runtime_sha256 = hashlib.sha256(runtime_payload).hexdigest()
    invocation_payload = _canonical(
        {
            "schema": INVOCATION_MANIFEST_SCHEMA,
            "runtime_manifest_sha256": runtime_sha256,
            "working_directory": str(repository),
            "auditor_argv": [
                str(producer),
                "--producer-bootstrap",
                "@captured/producer_bootstrap.py",
            ],
        }
    )
    execution = AuditExecution(
        command=(
            sys.executable,
            "-I",
            "-S",
            "-B",
            "/proc/self/fd/9",
        ),
        returncode=0,
        stdout=audit_payload,
        stderr=b"",
        report=audit,
        runtime_manifest=runtime_payload,
        runtime_manifest_sha256=runtime_sha256,
        invocation_manifest=invocation_payload,
        invocation_manifest_sha256=hashlib.sha256(invocation_payload).hexdigest(),
        bootstrap_sha256=bootstrap_source_sha256(),
        auditor_source_sha256=binding_module.sha256_file(auditor_source),
        support_files=tuple(
            CapturedFileIdentity(
                relative_path=str(row["path"]),
                bytes=int(row["bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in support_rows
        ),
        source_mode="descriptor",
        subprocess_elapsed_ns=1,
    )
    operator_module._write_execution(
        audit_attempt,
        prefix="audit-execution-01",
        execution=execution,
    )
    operator_module._write_execution(
        audit_attempt,
        prefix="audit-execution-02",
        execution=execution,
    )
    independent_audit = audit_attempt / "independent-audit.json"
    independent_audit.write_bytes(audit_payload)
    reproduction = binding_module.create_audit_reproduction_receipt(
        supplied_audit_path=independent_audit,
        first_execution=execution,
        execution=execution,
        first_execution_receipt_path=(
            audit_attempt / "audit-execution-01.execution.json"
        ),
        replay_execution_receipt_path=(
            audit_attempt / "audit-execution-02.execution.json"
        ),
        producer_manifest=json.loads(producer_terminal_payload),
        producer_manifest_sha256=hashlib.sha256(
            producer_terminal_payload
        ).hexdigest(),
        output_path=audit_attempt / "audit-reproduction.json",
    )
    reproduction_runtime_file = reproduction["runtime_manifest_file"]
    assert isinstance(reproduction_runtime_file, str)
    assert len(tuple(audit_attempt.iterdir())) == 15
    audit_files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(audit_attempt.iterdir(), key=lambda item: item.name)
    ]
    audit_terminal = audit_attempt / "operator-attempt.json"
    audit_terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.operator-attempt.v1",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "assurance": dict(_ASSURANCE),
                "kind": "audit",
                "lane": "development",
                "status": "accepted",
                "inputs": producer_rows,
                "primary": {
                    "producer_root": str(producer),
                    "audit_file": "independent-audit.json",
                    "executions": [
                        "audit-execution-01.execution.json",
                        "audit-execution-02.execution.json",
                    ],
                    "execution_failures": [],
                    "reproduction_file": "audit-reproduction.json",
                    "reproduction_runtime_file": (reproduction_runtime_file),
                    "claim_file": None,
                },
                "error": None,
                "files": audit_files,
                "file_count": len(audit_files),
                "manifest_excludes": ["operator-attempt.json"],
            }
        )
    )
    audit_completion = _completion_marker(
        completion_root,
        audit_terminal,
    )
    os.link(audit_terminal, audit_completion)
    audit_input_rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (
            *sorted(audit_attempt.iterdir(), key=lambda item: item.name),
            audit_completion,
        )
    ]
    closure = results / "development" / "development-closure-v1.20.0.json"
    closure_value = {
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
        "producer_root": str(producer),
        "result_qualification_member": ("evidence/development-result-qualification.json"),
        "producer_execution": producer_execution,
        "qualification_archive": {
            "canonical_path": "qualification.tar",
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
                    "path": ("evidence/development-result-qualification.json"),
                    "sha256": hashlib.sha256(qualification_payload).hexdigest(),
                },
            ],
        },
    }
    closure.write_bytes(_canonical(closure_value))
    closure_attempt = results / "operator-v1.20" / "closures" / "development-closure-v1.20.0"
    closure_attempt.mkdir(parents=True)
    fresh_reopen = {
        "schema": "prospect.wm001.development-closure-fresh-reopen.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.20.0",
        "mode": "fresh-closure-reopen",
        "challenge": "1" * 64,
        "requesting_process_id": 100,
        "verifier_process_id": 101,
        "matrix_contract_sha256": ("09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"),
        "development_closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
        "producer_manifest_sha256": producer_rows[0]["sha256"],
        "raw_result_sha256": producer_rows[1]["sha256"],
        "passed": True,
    }
    fresh_path = closure_attempt / "fresh-runtime-reopen.json"
    fresh_path.write_bytes(_canonical(fresh_reopen))
    fresh_payload = fresh_path.read_bytes()
    closure_reference = {
        "schema": "prospect.wm001.closure-reference.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.20.0",
        "closure_marker": str(closure),
        "closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
        "qualification_archive": closure_value["qualification_archive"],
        "producer_root": closure_value["producer_root"],
        "audit_attempt": str(audit_attempt),
        "audit_attempt_manifest_sha256": hashlib.sha256(audit_terminal.read_bytes()).hexdigest(),
        "fresh_reopen_file": "fresh-runtime-reopen.json",
        "fresh_reopen_sha256": hashlib.sha256(fresh_payload).hexdigest(),
    }
    reference_path = closure_attempt / "closure-reference.json"
    reference_path.write_bytes(_canonical(closure_reference))
    reference_payload = reference_path.read_bytes()
    closure_terminal = closure_attempt / "operator-attempt.json"
    closure_terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.operator-attempt.v1",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "assurance": dict(_ASSURANCE),
                "kind": "closure",
                "lane": "development",
                "status": "accepted",
                "inputs": [*producer_rows, *audit_input_rows],
                "primary": {"closure_reference_file": "closure-reference.json"},
                "error": None,
                "files": [
                    {
                        "path": "closure-reference.json",
                        "bytes": len(reference_payload),
                        "sha256": hashlib.sha256(reference_payload).hexdigest(),
                    },
                    {
                        "path": "fresh-runtime-reopen.json",
                        "bytes": len(fresh_payload),
                        "sha256": hashlib.sha256(fresh_payload).hexdigest(),
                    },
                ],
                "file_count": 2,
                "manifest_excludes": ["operator-attempt.json"],
            }
        )
    )
    closure_completion = _completion_marker(
        completion_root,
        closure_terminal,
    )
    os.link(closure_terminal, closure_completion)

    accepted_closure = runtime_outputs["runtime-accepted-closure-evidence"]
    accepted_closure.update(
        {
            "development_closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
            "producer_manifest_sha256": producer_rows[0]["sha256"],
            "raw_result_sha256": producer_rows[1]["sha256"],
            "closure_attempt_manifest_sha256": hashlib.sha256(closure_terminal.read_bytes()).hexdigest(),
            "closure_outer_completion_sha256": hashlib.sha256(closure_completion.read_bytes()).hexdigest(),
        }
    )
    accepted_payload = _canonical(accepted_closure)
    accepted_digest = hashlib.sha256(accepted_payload).hexdigest()
    accepted_key = "runtime-accepted-closure-evidence:stdout"
    log_paths[accepted_key].unlink()
    accepted_log = report.with_name(
        f"preformal-v1.20.0-command-09-runtime-accepted-closure-evidence.stdout.{accepted_digest}.log"
    )
    accepted_log.write_bytes(accepted_payload)
    log_paths[accepted_key] = accepted_log
    accepted_command = next(row for row in commands if row["name"] == "runtime-accepted-closure-evidence")
    accepted_command["stdout"] = {
        "file": accepted_log.name,
        "bytes": len(accepted_payload),
        "sha256": accepted_digest,
    }
    report.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.preformal-test-report.v2",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "device": "cpu",
                "all_pass": True,
                "commands": commands,
            }
        )
    )

    binding = dict(binding)
    protocol = binding.get("protocol")
    protocol = dict(protocol) if isinstance(protocol, dict) else {}
    protocol.setdefault("sha256", "e" * 64)
    binding["protocol"] = protocol
    dependencies = binding.get("dependencies")
    dependencies = dict(dependencies) if isinstance(dependencies, dict) else {}
    dependencies.setdefault("packages", packages)
    dependencies.setdefault("package_roots", [package_root])
    dependencies.setdefault("standard_library", standard_library)
    dependencies.setdefault("package_ownership", ownership)
    binding["dependencies"] = dependencies
    runtime = binding.get("runtime")
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    runtime["device"] = "cpu"
    binding["runtime"] = runtime
    binding.setdefault("audit_execution", audit_execution)
    source = binding.get("source")
    source = dict(source) if isinstance(source, dict) else {}
    source.update(
        {
            "implementation_files": [
                {
                    "path": ("bench/world_model_lifecycle/artifact_audit.py"),
                    "bytes": 1,
                    "sha256": "a" * 64,
                }
            ],
            "test_report_file": report.name,
            "test_report_bytes": report.stat().st_size,
            "test_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "test_log_files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in log_paths.values()
            ],
        }
    )
    binding["source"] = source
    development = binding.get("development_qualification")
    development = dict(development) if isinstance(development, dict) else {}
    development_defaults: dict[str, object] = {
        "closure_schema": "prospect.wm001.development-closure.v2",
        "closure_file": "development-closure-aaaaaaaaaaaaaaaa.json",
        "qualification_archive_file": "development-qualification-aaaaaaaaaaaaaaaa.tar",
        "qualification_archive_path": (
            "bench/world_model_lifecycle/results/development/"
            "development-qualification-aaaaaaaaaaaaaaaa.tar"
        ),
        "qualification_archive_bytes": 10240,
        "qualification_archive_sha256": "1" * 64,
        "qualification_archive_members_sha256": "2" * 64,
        "independent_audit_sha256": "3" * 64,
        "audit_reproduction_sha256": "4" * 64,
        "audit_runtime_manifest_sha256": "5" * 64,
        "audit_invocation_manifest_sha256": "6" * 64,
        "audit_stderr_sha256": "7" * 64,
        "source_identity_sha256": "8" * 64,
        "producer_custody_identity_sha256": "9" * 64,
        "audit_execution_identity_sha256": "a" * 64,
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    for field, value in development_defaults.items():
        development.setdefault(field, value)
    development.setdefault(
        "producer_manifest_sha256",
        producer_rows[0]["sha256"],
    )
    development.setdefault(
        "raw_result_sha256",
        producer_rows[1]["sha256"],
    )
    development.setdefault(
        "result_qualification_sha256",
        hashlib.sha256(qualification_payload).hexdigest(),
    )
    development.setdefault(
        "producer_execution_identity_sha256",
        launch_bootstrap._canonical_digest(producer_execution),
    )
    development.update(
        {
            "closure_bytes": closure.stat().st_size,
            "closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
        }
    )
    binding["development_qualification"] = development
    attempt = results / "operator-v1.20" / "bindings" / "formal-binding-v1.20.0"
    attempt.mkdir(parents=True)
    binding_path = attempt / "formal-binding.json"
    binding_path.write_bytes(_canonical(binding))
    binding_payload = binding_path.read_bytes()
    preflight_path = attempt / "formal-input-preflight.json"
    preflight_payload = _canonical(
        {
            "schema": "prospect.wm001.formal-input-preflight.v1",
            "experiment_id": "WM-001",
            "protocol_version": "1.20.0",
            "binding_bytes": len(binding_payload),
            "binding_sha256": hashlib.sha256(binding_payload).hexdigest(),
            "preformal_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "development_closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
            "accepted_closure_evidence_sha256": hashlib.sha256(
                _canonical(runtime_outputs["runtime-accepted-closure-evidence"])[:-1]
            ).hexdigest(),
            "runtime_conformance_sha256": hashlib.sha256(
                _canonical(runtime_outputs["runtime-bootstrap-inventory-conformance"])[:-1]
            ).hexdigest(),
            "auditor_source_sha256": "a" * 64,
            "passed": True,
        }
    )
    preflight_path.write_bytes(preflight_payload)
    qualification_path = attempt / "development-result-qualification.json"
    qualification_path.write_bytes(qualification_payload)
    terminal = attempt / "operator-attempt.json"
    terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.operator-attempt.v1",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "assurance": dict(_ASSURANCE),
                "kind": "binding",
                "lane": None,
                "status": "accepted",
                "inputs": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (
                        report,
                        *log_paths.values(),
                        closure,
                        closure_terminal,
                        closure_completion,
                    )
                ],
                "primary": {"binding_file": "formal-binding.json"},
                "error": None,
                "files": [
                    {
                        "path": "development-result-qualification.json",
                        "bytes": len(qualification_payload),
                        "sha256": hashlib.sha256(qualification_payload).hexdigest(),
                    },
                    {
                        "path": "formal-binding.json",
                        "bytes": len(binding_payload),
                        "sha256": hashlib.sha256(binding_payload).hexdigest(),
                    },
                    {
                        "path": "formal-input-preflight.json",
                        "bytes": len(preflight_payload),
                        "sha256": hashlib.sha256(preflight_payload).hexdigest(),
                    },
                ],
                "file_count": 3,
                "manifest_excludes": ["operator-attempt.json"],
            }
        )
    )
    marker = _completion_marker(completion_root, terminal)
    if terminal_state == "finalized":
        os.link(terminal, marker)
    elif terminal_state == "copied-marker":
        stray_terminal = repository / "stray-binding-terminal.json"
        os.link(terminal, stray_terminal)
        shutil.copyfile(terminal, marker)
        stray_marker = completion_root / "stray-binding-marker.json"
        os.link(marker, stray_marker)
    elif terminal_state != "unfinalized":
        raise AssertionError(f"unknown terminal state: {terminal_state}")
    return binding_path, terminal, marker


def _minimal_formal_binding() -> dict[str, object]:
    return {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "experiment_id": "WM-001",
        "assurance": dict(_ASSURANCE),
    }


def _development_audit_context(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    list[dict[str, object]],
]:
    repository, completion_root, _ = _repository_paths(tmp_path)
    _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    audit_attempt = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.20"
        / "audits"
        / "development-audit-v1.20.0"
    )
    producer_rows = launch_bootstrap._verify_development_producer(
        repository=repository,
        completion_root=completion_root,
    )
    return repository, completion_root, audit_attempt, producer_rows


def _rebind_audit_terminal(audit_attempt: Path) -> None:
    terminal = audit_attempt / "operator-attempt.json"
    manifest = json.loads(terminal.read_bytes())
    manifest["files"] = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(audit_attempt.iterdir(), key=lambda item: item.name)
        if path != terminal
    ]
    manifest["file_count"] = len(manifest["files"])
    terminal.write_bytes(_canonical(manifest))


def _rehearsal_output(binding: dict[str, object]) -> bytes:
    dependencies = binding["dependencies"]
    runtime = binding["runtime"]
    audit_execution = binding["audit_execution"]
    assert isinstance(dependencies, dict)
    assert isinstance(runtime, dict)
    assert isinstance(audit_execution, dict)
    inventory = {
        "packages": dependencies["packages"],
        "package_roots": dependencies["package_roots"],
        "standard_library": dependencies["standard_library"],
        "package_ownership": dependencies["package_ownership"],
    }
    fresh = {
        "schema": "prospect.wm001.fresh-runtime-identity-conformance.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.20.0",
        "mode": "fresh-identity-conformance",
        "challenge": "6" * 64,
        "requesting_process_id": 303,
        "verifier_process_id": 404,
        "matrix_contract_sha256": (
            launch_bootstrap._DEVELOPMENT_MATRIX_CONTRACT_SHA256
        ),
        "passed": True,
    }
    return _canonical(
        {
            "schema": "prospect.wm001.preformal-runtime-check.v1",
            "mode": "bootstrap-inventory-conformance",
            "device": runtime["device"],
            "passed": True,
            "inventory": inventory,
            "inventory_sha256": launch_bootstrap._canonical_digest(inventory),
            "conformance_sha256": launch_bootstrap._canonical_digest(audit_execution),
            "fresh_runtime_identity_conformance": fresh,
            "fresh_runtime_identity_conformance_sha256": (launch_bootstrap._canonical_digest(fresh)),
            "restart_runtime_conformance_report_sha256": (audit_execution["restart_runtime_conformance_report_sha256"]),
            "restart_runtime_execution_receipt_sha256": (audit_execution["restart_runtime_execution_receipt_sha256"]),
            "restart_runtime_support_files": audit_execution["restart_runtime_support_files"],
            "restart_runtime_repeat_count": 3,
            "restart_runtime_path_descriptor_equal": True,
            "repeat_count": 3,
            "path_descriptor_equal": True,
        }
    )


def _configure_independent_rehearsal(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository: Path,
    binding_path: Path,
    binding: dict[str, object],
) -> None:
    results = repository / "bench" / "world_model_lifecycle" / "results"
    operator_root = results / "operator-v1.20"
    attempt = operator_root / "rehearsals" / "accepted-binding-rehearsal-v1.20.0"
    claim_root = results / "rehearsals" / "v1.20"
    completion_root = results / "outer-completions" / "v1.20"
    lifecycle = repository / "bench" / "world_model_lifecycle"
    replacements = {
        "REPO": repository,
        "RESULTS_ROOT": results,
        "OPERATOR_RESULTS_ROOT": operator_root,
        "REHEARSAL_ATTEMPTS_ROOT": attempt.parent,
        "REHEARSAL_ATTEMPT_PATH": attempt,
        "REHEARSAL_CLAIM_ROOT": claim_root,
        "OUTER_COMPLETIONS_ROOT": completion_root,
        "FORMAL_BINDING_PATH": binding_path,
        "LAUNCH_BOOTSTRAP_PATH": lifecycle / "launch_bootstrap.py",
        "PRODUCER_BOOTSTRAP_PATH": lifecycle / "producer_bootstrap.py",
        "CLAIM_PATH": attempt / rehearsal_module.CLAIM_NAME,
        "TERMINAL_PATH": attempt / rehearsal_module.TERMINAL_NAME,
        "STDOUT_PATH": attempt / rehearsal_module.STDOUT_NAME,
        "STDERR_PATH": attempt / rehearsal_module.STDERR_NAME,
        "OUTER_RECEIPT_PATH": (attempt / rehearsal_module.OUTER_RECEIPT_NAME),
    }
    for name, value in replacements.items():
        monkeypatch.setattr(rehearsal_module, name, value)
    monkeypatch.setattr(
        rehearsal_module,
        "_verified_binding",
        lambda path: binding if path == binding_path else None,
    )


def _rehearsal_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[str, object]]:
    repository, completion_root, _ = _repository_paths(tmp_path)
    lifecycle = repository / "bench" / "world_model_lifecycle"
    shutil.copyfile(
        Path(launch_bootstrap.__file__),
        lifecycle / "launch_bootstrap.py",
    )
    shutil.copyfile(
        Path(producer_bootstrap.__file__),
        lifecycle / "producer_bootstrap.py",
    )
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    binding = json.loads(binding_path.read_bytes())
    _configure_independent_rehearsal(
        monkeypatch,
        repository=repository,
        binding_path=binding_path,
        binding=binding,
    )
    return repository, completion_root, binding_path, binding


def _rehearsal_arguments(
    repository: Path,
    binding_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        bootstrap=(repository / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"),
        runtime_seal=None,
        create_runtime_seal=None,
        rehearse_accepted_binding=binding_path,
        producer_arguments=[],
    )


def _prospective_runtime_seal_value() -> dict[str, object]:
    return {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.20.0",
        "assurance": dict(_ASSURANCE),
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "worktree_clean": True,
        "python": {"version": [3, 11, 0]},
        "required_flags": {"isolated": 1},
        "process_environment": {"LC_ALL": "C.UTF-8"},
        "bootstrap_source_sha256": "c" * 64,
        "standard_library": {"sha256": "d" * 64},
        "package_roots": [],
        "package_ownership": {},
    }


def _producer_receipt(
    repository: Path,
    terminal: Path,
    *,
    logical_exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    monkeypatch.chdir(repository)
    registration = producer_bootstrap._capture_outer_terminal(
        terminal,
        logical_exit_code=logical_exit_code,
    )
    read_descriptor, write_descriptor = os.pipe()
    try:
        producer_bootstrap._emit_outer_receipt(
            write_descriptor,
            registration,
        )
        os.close(write_descriptor)
        write_descriptor = -1
        receipt_payload = launch_bootstrap._read_receipt(read_descriptor)
        assert isinstance(receipt_payload, bytes)
        return receipt_payload
    finally:
        if write_descriptor >= 0:
            os.close(write_descriptor)
        os.close(read_descriptor)
        terminal_descriptor = registration["descriptor"]
        assert isinstance(terminal_descriptor, int)
        os.close(terminal_descriptor)


def test_producer_receipt_commits_exact_terminal_as_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    receipt = _producer_receipt(
        repository,
        terminal,
        logical_exit_code=1,
        monkeypatch=monkeypatch,
    )

    assert (
        launch_bootstrap._commit_outer_receipt(
            receipt,
            repository=repository,
            completion_root=completion_root,
        )
        == 1
    )

    marker = _completion_marker(completion_root, terminal)
    assert marker.read_bytes() == terminal.read_bytes()
    assert os.path.samefile(marker, terminal)
    assert terminal.stat().st_nlink == marker.stat().st_nlink == 2


def _remove_assurance(receipt: dict[str, object]) -> None:
    del receipt["assurance"]


def _overstate_tamper_resistance(receipt: dict[str, object]) -> None:
    assurance = receipt["assurance"]
    assert isinstance(assurance, dict)
    assurance["tamper_resistant"] = True


def _overstate_external_attestation(receipt: dict[str, object]) -> None:
    assurance = receipt["assurance"]
    assert isinstance(assurance, dict)
    assurance["external_attestation"] = True


def _alias_tamper_resistance(receipt: dict[str, object]) -> None:
    assurance = receipt["assurance"]
    assert isinstance(assurance, dict)
    assurance["tamper_resistant"] = 0


@pytest.mark.parametrize("legacy_mode", [None, "True"])
def test_both_bootstraps_require_exact_bound_torchrl_legacy_mode(
    monkeypatch: pytest.MonkeyPatch,
    legacy_mode: str | None,
) -> None:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    if legacy_mode is not None:
        environment["LAZY_LEGACY_OP"] = legacy_mode
    monkeypatch.setattr(launch_bootstrap.os, "environ", environment)

    with pytest.raises(launch_bootstrap.LaunchError, match="exact safe runtime"):
        launch_bootstrap._environment()
    with pytest.raises(producer_bootstrap.BootstrapError, match="exact safe runtime"):
        producer_bootstrap._environment()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("PYGAME_HIDE_SUPPORT_PROMPT", None, id="missing-pygame-prompt"),
        pytest.param("PYGAME_HIDE_SUPPORT_PROMPT", "show", id="wrong-pygame-prompt"),
        pytest.param("SDL_AUDIODRIVER", None, id="missing-sdl-audio"),
        pytest.param("SDL_AUDIODRIVER", "alsa", id="wrong-sdl-audio"),
    ],
)
def test_both_bootstraps_require_exact_bound_gymnasium_defaults(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
) -> None:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    if value is None:
        del environment[name]
    else:
        environment[name] = value
    monkeypatch.setattr(launch_bootstrap.os, "environ", environment)

    with pytest.raises(launch_bootstrap.LaunchError, match="exact safe runtime"):
        launch_bootstrap._environment()
    with pytest.raises(producer_bootstrap.BootstrapError, match="exact safe runtime"):
        producer_bootstrap._environment()


@pytest.mark.parametrize(
    ("module", "error_type"),
    [
        pytest.param(
            launch_bootstrap,
            launch_bootstrap.LaunchError,
            id="outer-launcher",
        ),
        pytest.param(
            producer_bootstrap,
            producer_bootstrap.BootstrapError,
            id="producer-bootstrap",
        ),
    ],
)
def test_bootstraps_drop_absent_search_entries_and_reject_ambient_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    error_type: type[Exception],
) -> None:
    stdlib = tmp_path / "stdlib"
    dynload = stdlib / "lib-dynload"
    ambient = tmp_path / "ambient"
    dynload.mkdir(parents=True)
    ambient.mkdir()
    absent_zip = tmp_path / "python312.zip"
    monkeypatch.setattr(
        module.sysconfig,
        "get_path",
        lambda name: str(stdlib),
    )
    monkeypatch.setattr(
        module.sys,
        "path",
        [str(absent_zip), str(stdlib), str(dynload)],
    )

    assert module._sanitize_module_search_path() == (
        str(stdlib),
        str(dynload),
    )
    assert module.sys.path == [str(stdlib), str(dynload)]

    module.sys.path[:] = [str(stdlib), str(ambient)]
    with pytest.raises(
        error_type,
        match="ambient import root",
    ):
        module._sanitize_module_search_path()


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(_remove_assurance, id="missing-assurance"),
        pytest.param(
            _overstate_tamper_resistance,
            id="overstated-tamper-resistance",
        ),
        pytest.param(
            _overstate_external_attestation,
            id="overstated-external-attestation",
        ),
        pytest.param(
            _alias_tamper_resistance,
            id="numeric-assurance-alias",
        ),
    ],
)
def test_outer_commit_rejects_missing_or_overstated_assurance(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    receipt = _receipt(terminal)
    mutation(receipt)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="receipt is malformed",
    ):
        launch_bootstrap._commit_outer_receipt(
            _canonical(receipt),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert not _completion_marker(completion_root, terminal).exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("terminal_bytes", 999, "differs from its identity"),
        ("terminal_sha256", "0" * 64, "differs from its identity"),
        ("terminal_path", "/tmp/not-a-wm001-result.json", "outside results"),
    ],
)
def test_outer_commit_rejects_terminal_identity_or_path_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    receipt = _receipt(terminal)
    receipt[field] = replacement

    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._commit_outer_receipt(
            _canonical(receipt),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert not _completion_marker(completion_root, terminal).exists()


def test_copied_completion_file_cannot_substitute_for_hardlink(
    tmp_path: Path,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    marker = _completion_marker(completion_root, terminal)
    shutil.copyfile(terminal, marker)
    assert marker.read_bytes() == terminal.read_bytes()
    assert not os.path.samefile(marker, terminal)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="completion marker already exists",
    ):
        launch_bootstrap._commit_outer_receipt(
            _canonical(_receipt(terminal)),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert marker.stat().st_nlink == 1
    assert not os.path.samefile(marker, terminal)


def test_hardlink_failure_leaves_terminal_unfinalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    marker = _completion_marker(completion_root, terminal)

    def fail_link(
        _source: os.PathLike[str] | str,
        _destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        del src_dir_fd, dst_dir_fd, follow_symlinks
        raise OSError("injected hardlink failure")

    monkeypatch.setattr(launch_bootstrap.os, "link", fail_link)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="completion hardlink failed",
    ):
        launch_bootstrap._commit_outer_receipt(
            _canonical(_receipt(terminal)),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert not marker.exists()


@pytest.mark.parametrize("marker_state", ["missing", "wrong-inode"])
def test_completed_runtime_seal_rejects_missing_or_forged_marker(
    tmp_path: Path,
    marker_state: str,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    results = completion_root.parents[1]
    runtime_seal = results / "development" / "runtime-seal-v1.20.0.json"
    runtime_seal.parent.mkdir()
    runtime_seal.write_bytes(_canonical(_prospective_runtime_seal_value()))
    stray_seal_link = runtime_seal.with_name("stray-seal-link.json")
    os.link(runtime_seal, stray_seal_link)
    assert runtime_seal.stat().st_nlink == 2

    marker = _completion_marker(completion_root, runtime_seal)
    expected_message = "cannot be resolved"
    if marker_state == "wrong-inode":
        shutil.copyfile(runtime_seal, marker)
        stray_marker_link = completion_root / "stray-marker-link.json"
        os.link(marker, stray_marker_link)
        assert marker.stat().st_nlink == 2
        assert not os.path.samefile(runtime_seal, marker)
        expected_message = "not the terminal inode"

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match=expected_message,
    ):
        launch_bootstrap._open_typed_runtime_custody(
            runtime_seal,
            repository=repository,
            completion_root=completion_root,
        )

    assert runtime_seal.stat().st_nlink == 2
    assert terminal.stat().st_nlink == 1


def test_outer_accepts_only_exact_canonical_prospective_runtime_seal(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    runtime_seal = launch_bootstrap._prospective_runtime_seal(repository)
    runtime_seal.parent.mkdir()
    runtime_seal.write_bytes(_canonical(_prospective_runtime_seal_value()))
    marker = _completion_marker(completion_root, runtime_seal)
    os.link(runtime_seal, marker)

    descriptor, payload, _, expected_nlink = launch_bootstrap._open_typed_runtime_custody(
        runtime_seal,
        repository=repository,
        completion_root=completion_root,
    )
    try:
        assert payload == runtime_seal.read_bytes()
        assert expected_nlink == 2
    finally:
        os.close(descriptor)

    sibling = runtime_seal.with_name("alias-runtime-seal-v1.20.0.json")
    shutil.copyfile(runtime_seal, sibling)
    sibling_link = sibling.with_name("alias-runtime-seal-link.json")
    os.link(sibling, sibling_link)
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="not one canonical",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            sibling,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda value: value.pop("package_ownership"),
            id="missing-field",
        ),
        pytest.param(
            lambda value: value.__setitem__("unexpected", True),
            id="extra-field",
        ),
        pytest.param(
            lambda value: value.__setitem__("protocol_version", "1.5"),
            id="wrong-version",
        ),
    ],
)
def test_prospective_runtime_seal_requires_exact_versioned_field_set(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    runtime_seal = launch_bootstrap._prospective_runtime_seal(repository)
    runtime_seal.parent.mkdir()
    value = _prospective_runtime_seal_value()
    mutation(value)
    runtime_seal.write_bytes(_canonical(value))
    os.link(runtime_seal, _completion_marker(completion_root, runtime_seal))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="schema or assurance",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            runtime_seal,
            repository=repository,
            completion_root=completion_root,
        )


def test_runtime_seal_creation_rejects_every_noncanonical_output_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    called = False

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("producer must not execute")

    monkeypatch.setattr(launch_bootstrap.subprocess, "run", unexpected_run)
    arguments = argparse.Namespace(
        bootstrap=repository / "missing-bootstrap.py",
        create_runtime_seal=(
            repository / "bench" / "world_model_lifecycle" / "results" / "development" / "runtime-seal.json"
        ),
        runtime_seal=None,
        producer_arguments=[],
    )
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="sole canonical",
    ):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert called is False


def test_outer_accepts_only_finalized_canonical_formal_binding_attempt(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, marker = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )

    descriptor, payload, _, expected_nlink = launch_bootstrap._open_typed_runtime_custody(
        binding_path,
        repository=repository,
        completion_root=completion_root,
    )
    try:
        assert payload == binding_path.read_bytes()
        assert expected_nlink == 1
    finally:
        os.close(descriptor)
    assert binding_path.stat().st_nlink == 1
    assert terminal.stat().st_nlink == 2
    assert os.path.samefile(terminal, marker)


def test_outer_accepts_real_writer_production_shaped_audit_reproduction(
    tmp_path: Path,
) -> None:
    repository, completion_root, audit_attempt, producer_rows = _development_audit_context(tmp_path)
    terminal = audit_attempt / "operator-attempt.json"
    manifest = json.loads(terminal.read_bytes())
    reproduction = json.loads((audit_attempt / "audit-reproduction.json").read_bytes())

    assert manifest["file_count"] == 15
    assert len(manifest["files"]) == 15
    assert manifest["primary"]["reproduction_runtime_file"] == (reproduction["runtime_manifest_file"])
    assert reproduction["runtime_manifest_file"].startswith("development-audit-runtime-")
    launch_bootstrap._verify_development_audit(
        repository=repository,
        completion_root=completion_root,
        producer_rows=producer_rows,
    )


@pytest.mark.parametrize(
    "mutation",
    ("old-execution-alias", "unsafe-invocation", "non-addressed-stderr"),
)
def test_outer_rejects_noncanonical_reproduction_sidecar_name(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, completion_root, audit_attempt, producer_rows = _development_audit_context(tmp_path)
    terminal = audit_attempt / "operator-attempt.json"
    manifest = json.loads(terminal.read_bytes())
    receipt_path = audit_attempt / "audit-reproduction.json"
    receipt = json.loads(receipt_path.read_bytes())

    if mutation == "old-execution-alias":
        source = audit_attempt / receipt["runtime_manifest_file"]
        filename = "audit-execution-02.runtime.json"
        source.rename(audit_attempt / filename)
        receipt["runtime_manifest_file"] = filename
        manifest["primary"]["reproduction_runtime_file"] = filename
    elif mutation == "unsafe-invocation":
        receipt["invocation_manifest_file"] = "../audit-invocation.json"
    elif mutation == "non-addressed-stderr":
        source = audit_attempt / receipt["stderr_file"]
        filename = "development-audit-stderr.log"
        source.rename(audit_attempt / filename)
        receipt["stderr_file"] = filename
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(f"unknown mutation: {mutation}")

    receipt_path.write_bytes(_canonical(receipt))
    terminal.write_bytes(_canonical(manifest))
    _rebind_audit_terminal(audit_attempt)
    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._verify_development_audit(
            repository=repository,
            completion_root=completion_root,
            producer_rows=producer_rows,
        )


@pytest.mark.parametrize(
    ("role", "identity"),
    (
        ("runtime_manifest", "bytes"),
        ("runtime_manifest", "sha256"),
        ("invocation_manifest", "bytes"),
        ("invocation_manifest", "sha256"),
        ("stderr", "bytes"),
        ("stderr", "sha256"),
    ),
)
def test_outer_rejects_reproduction_sidecar_identity_mismatch(
    tmp_path: Path,
    role: str,
    identity: str,
) -> None:
    repository, completion_root, audit_attempt, producer_rows = _development_audit_context(tmp_path)
    receipt_path = audit_attempt / "audit-reproduction.json"
    receipt = json.loads(receipt_path.read_bytes())
    field = f"{role}_{identity}"
    if identity == "bytes":
        receipt[field] += 1
    else:
        receipt[field] = "f" * 64
    receipt_path.write_bytes(_canonical(receipt))
    _rebind_audit_terminal(audit_attempt)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match=f"reproduction {role} is misbound",
    ):
        launch_bootstrap._verify_development_audit(
            repository=repository,
            completion_root=completion_root,
            producer_rows=producer_rows,
        )


def test_outer_rejects_rebound_cross_role_reproduction_payload(
    tmp_path: Path,
) -> None:
    repository, completion_root, audit_attempt, producer_rows = _development_audit_context(tmp_path)
    terminal = audit_attempt / "operator-attempt.json"
    manifest = json.loads(terminal.read_bytes())
    receipt_path = audit_attempt / "audit-reproduction.json"
    receipt = json.loads(receipt_path.read_bytes())
    old_runtime = audit_attempt / receipt["runtime_manifest_file"]
    invocation_payload = (audit_attempt / receipt["invocation_manifest_file"]).read_bytes()
    invocation_sha256 = hashlib.sha256(invocation_payload).hexdigest()
    substituted_name = f"development-audit-runtime-{invocation_sha256[:16]}.json"
    old_runtime.unlink()
    (audit_attempt / substituted_name).write_bytes(invocation_payload)
    receipt.update(
        {
            "runtime_manifest_file": substituted_name,
            "runtime_manifest_bytes": len(invocation_payload),
            "runtime_manifest_sha256": invocation_sha256,
        }
    )
    manifest["primary"]["reproduction_runtime_file"] = substituted_name
    receipt_path.write_bytes(_canonical(receipt))
    terminal.write_bytes(_canonical(manifest))
    _rebind_audit_terminal(audit_attempt)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="reproduction runtime_manifest is misbound",
    ):
        launch_bootstrap._verify_development_audit(
            repository=repository,
            completion_root=completion_root,
            producer_rows=producer_rows,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-audit-execution",
        "inventory",
        "conformance-digest",
        "fresh-identity",
        "fresh-identity-digest",
        "restart-report-digest",
        "restart-receipt-digest",
        "restart-support",
        "restart-repeat-count",
        "restart-path-descriptor",
        "repeat-count",
        "path-descriptor",
    ),
)
def test_recorded_runtime_conformance_is_cross_linked_to_binding(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    binding = json.loads(binding_path.read_bytes())
    dependencies = binding["dependencies"]
    audit_execution: object = binding["audit_execution"]
    report_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "v1.20.0"
        / "preformal"
        / "preformal-test-report-v1.20.0.json"
    )
    report = json.loads(report_path.read_bytes())
    command = next(row for row in report["commands"] if row["name"] == "runtime-bootstrap-inventory-conformance")
    value = json.loads((report_path.parent / command["stdout"]["file"]).read_bytes())

    if mutation == "missing-audit-execution":
        audit_execution = None
    elif mutation == "inventory":
        value["inventory"]["packages"][0]["version"] = "3.12.1"
        value["inventory_sha256"] = launch_bootstrap._canonical_digest(value["inventory"])
    elif mutation == "conformance-digest":
        value["conformance_sha256"] = "f" * 64
    elif mutation == "fresh-identity":
        value["fresh_runtime_identity_conformance"]["passed"] = False
        value["fresh_runtime_identity_conformance_sha256"] = launch_bootstrap._canonical_digest(
            value["fresh_runtime_identity_conformance"]
        )
    elif mutation == "fresh-identity-digest":
        value["fresh_runtime_identity_conformance_sha256"] = "f" * 64
    else:
        assert isinstance(audit_execution, dict)
        if mutation == "restart-report-digest":
            audit_execution["restart_runtime_conformance_report_sha256"] = "f" * 64
        elif mutation == "restart-receipt-digest":
            audit_execution["restart_runtime_execution_receipt_sha256"] = "f" * 64
        elif mutation == "restart-support":
            audit_execution["restart_runtime_support_files"] = [
                "producer_bootstrap.py",
                "protocol.json",
            ]
        elif mutation == "restart-repeat-count":
            audit_execution["restart_runtime_repeat_count"] = 4
        elif mutation == "restart-path-descriptor":
            audit_execution["restart_runtime_path_descriptor_equal"] = False
        elif mutation == "repeat-count":
            audit_execution["repeat_count"] = 4
        elif mutation == "path-descriptor":
            audit_execution["path_descriptor_equal"] = False
        else:  # pragma: no cover - parametrization is closed above
            raise AssertionError(f"unknown mutation: {mutation}")
        value["conformance_sha256"] = launch_bootstrap._canonical_digest(audit_execution)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="formal binding",
    ):
        launch_bootstrap._recorded_runtime_conformance(
            value,
            dependencies=dependencies,
            audit_execution=audit_execution,
            device="cpu",
        )


def test_outer_rejects_rebound_nonempty_command10_stderr(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    results = repository / "bench" / "world_model_lifecycle" / "results"
    report_path = results / "development" / "v1.20.0" / "preformal" / "preformal-test-report-v1.20.0.json"
    report = json.loads(report_path.read_bytes())
    command = next(row for row in report["commands"] if row["name"] == "runtime-bootstrap-inventory-conformance")
    stderr_path = report_path.parent / command["stderr"]["file"]
    stderr_payload = b"unexpected runtime stderr\n"
    stderr_path.write_bytes(stderr_payload)
    command["stderr"]["bytes"] = len(stderr_payload)
    command["stderr"]["sha256"] = hashlib.sha256(stderr_payload).hexdigest()
    report_path.write_bytes(_canonical(report))

    binding = json.loads(binding_path.read_bytes())
    source = binding["source"]
    source["test_report_bytes"] = report_path.stat().st_size
    source["test_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    stderr_row = next(row for row in source["test_log_files"] if row["path"] == stderr_path.name)
    stderr_row["bytes"] = len(stderr_payload)
    stderr_row["sha256"] = hashlib.sha256(stderr_payload).hexdigest()
    binding_path.write_bytes(_canonical(binding))

    preflight_path = binding_path.with_name("formal-input-preflight.json")
    preflight = json.loads(preflight_path.read_bytes())
    preflight["binding_bytes"] = binding_path.stat().st_size
    preflight["binding_sha256"] = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    preflight["preformal_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    preflight_path.write_bytes(_canonical(preflight))

    terminal_value = json.loads(terminal.read_bytes())
    for row in terminal_value["inputs"]:
        path = Path(row["path"])
        if path in {report_path, stderr_path}:
            row["bytes"] = path.stat().st_size
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in terminal_value["files"]:
        path = binding_path.with_name(row["path"])
        row["bytes"] = path.stat().st_size
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    terminal.write_bytes(_canonical(terminal_value))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="runtime preformal output is malformed",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-field",
        "extra-field",
        "passed",
        "development-closure",
        "producer-manifest",
        "raw-result",
        "closure-attempt",
        "closure-completion",
    ),
)
def test_recorded_accepted_closure_evidence_is_complete_and_cross_linked(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    results = repository / "bench" / "world_model_lifecycle" / "results"
    report_path = results / "development" / "v1.20.0" / "preformal" / "preformal-test-report-v1.20.0.json"
    report = json.loads(report_path.read_bytes())
    command = next(row for row in report["commands"] if row["name"] == "runtime-accepted-closure-evidence")
    value = json.loads((report_path.parent / command["stdout"]["file"]).read_bytes())
    closure = results / "development" / "development-closure-v1.20.0.json"
    producer = results / "development" / "qualification-v1.20.0"
    closure_terminal = results / "operator-v1.20" / "closures" / "development-closure-v1.20.0" / "operator-attempt.json"
    expected = {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "accepted-closure-evidence",
        "passed": True,
        "development_closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
        "producer_manifest_sha256": hashlib.sha256((producer / "producer-manifest.json").read_bytes()).hexdigest(),
        "raw_result_sha256": hashlib.sha256((producer / "result.json").read_bytes()).hexdigest(),
        "closure_attempt_manifest_sha256": hashlib.sha256(closure_terminal.read_bytes()).hexdigest(),
        "closure_outer_completion_sha256": hashlib.sha256(
            _completion_marker(
                completion_root,
                closure_terminal,
            ).read_bytes()
        ).hexdigest(),
    }
    assert binding_path.is_file()
    assert value == expected
    assert (
        launch_bootstrap._recorded_accepted_closure_evidence(
            value,
            expected=expected,
        )
        == value
    )

    mutated = json.loads(json.dumps(value))
    if mutation == "missing-field":
        mutated.pop("producer_manifest_sha256")
    elif mutation == "extra-field":
        mutated["unbound"] = True
    elif mutation == "passed":
        mutated["passed"] = False
    else:
        digest_fields = {
            "development-closure": "development_closure_sha256",
            "producer-manifest": "producer_manifest_sha256",
            "raw-result": "raw_result_sha256",
            "closure-attempt": "closure_attempt_manifest_sha256",
            "closure-completion": "closure_outer_completion_sha256",
        }
        mutated[digest_fields[mutation]] = "f" * 64

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="accepted-closure evidence is malformed or misbound",
    ):
        launch_bootstrap._recorded_accepted_closure_evidence(
            mutated,
            expected=expected,
        )


@pytest.mark.parametrize(
    "field",
    ("producer_manifest_sha256", "raw_result_sha256"),
)
def test_closure_authorization_cross_links_binding_development_digests(
    tmp_path: Path,
    field: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    binding = json.loads(binding_path.read_bytes())
    binding["development_qualification"][field] = "f" * 64

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="closure authorization inputs differ",
    ):
        launch_bootstrap._verify_closure_authorization(
            repository=repository,
            completion_root=completion_root,
            binding=binding,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "started_at_utc",
            "2026-02-30T00:00:00Z",
            "real UTC timestamp",
        ),
        (
            "completed_at_utc",
            "2025-12-31T23:59:59Z",
            "producer manifest is malformed",
        ),
        ("file_count", 1.0, "producer manifest is malformed"),
    ],
)
def test_outer_rejects_malformed_producer_time_or_numeric_count(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    manifest_path = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "qualification-v1.20.0"
        / "producer-manifest.json"
    )
    manifest = json.loads(manifest_path.read_bytes())
    manifest[field] = value
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "byte_count",
    (64 << 20, (64 << 20) + 1),
    ids=("control-boundary", "control-boundary-plus-one"),
)
def test_outer_streams_valid_result_at_and_above_control_limit(
    tmp_path: Path,
    byte_count: int,
) -> None:
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={"result.json": byte_count},
    )

    rows = launch_bootstrap._verify_development_producer(
        repository=repository,
        completion_root=completion_root,
    )

    assert rows[1] == {
        "path": str(producer / "result.json"),
        "bytes": byte_count,
        "sha256": _zero_sha256(byte_count),
    }


def test_outer_streams_production_scale_five_role_producer_once_per_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_files = {
        "checkpoints/development-3626676950.pt": 78_204_724,
        "checkpoints/development-2572962267.pt": 78_205_824,
        "replicates/development-3626676950.json": 160_421_650,
        "replicates/development-2572962267.json": 160_537_118,
        "result.json": 320_977_868,
    }
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files=production_files,
    )
    calls: list[str] = []
    original = launch_bootstrap._stream_regular_row

    def observed(
        path: Path,
        *,
        label: str,
        maximum_bytes: int,
        expected_nlink: int = 1,
    ) -> tuple[dict[str, object], tuple[int, ...]]:
        calls.append(path.relative_to(producer).as_posix())
        return original(
            path,
            label=label,
            maximum_bytes=maximum_bytes,
            expected_nlink=expected_nlink,
        )

    monkeypatch.setattr(
        launch_bootstrap,
        "_stream_regular_row",
        observed,
    )
    rows = launch_bootstrap._verify_development_producer(
        repository=repository,
        completion_root=completion_root,
    )

    assert calls == sorted(production_files)
    assert len(calls) == len(set(calls)) == 5
    assert rows[1] == {
        "path": str(producer / "result.json"),
        "bytes": production_files["result.json"],
        "sha256": _zero_sha256(production_files["result.json"]),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("boolean-bytes", "manifest row"),
        ("float-bytes", "manifest row"),
        ("negative-bytes", "manifest row"),
        ("file-limit", "manifest row"),
        ("invalid-digest", "manifest row"),
        ("digest-mismatch", "file changed"),
        ("duplicate-path", "manifest row"),
        ("unsorted-paths", "manifest row"),
    ),
)
def test_outer_streaming_manifest_rejects_malformed_identity_metadata(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={"result.json": b"result", "sidecar.bin": b"sidecar"},
    )
    manifest_path = producer / "producer-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    rows = manifest["files"]
    if mutation == "boolean-bytes":
        rows[0]["bytes"] = True
    elif mutation == "float-bytes":
        rows[0]["bytes"] = float(rows[0]["bytes"])
    elif mutation == "negative-bytes":
        rows[0]["bytes"] = -1
    elif mutation == "file-limit":
        rows[0]["bytes"] = launch_bootstrap._MAX_PRODUCER_FILE_BYTES + 1
    elif mutation == "invalid-digest":
        rows[0]["sha256"] = "g" * 64
    elif mutation == "digest-mismatch":
        rows[0]["sha256"] = "0" * 64
    elif mutation == "duplicate-path":
        rows[1]["path"] = rows[0]["path"]
    elif mutation == "unsorted-paths":
        rows.reverse()
    else:  # pragma: no cover - parameter exhaustiveness
        raise AssertionError(mutation)
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_streaming_manifest_enforces_exact_producer_limits(
    tmp_path: Path,
) -> None:
    assert launch_bootstrap._MAX_PRODUCER_FILE_BYTES == 4 << 30
    assert launch_bootstrap._MAX_RESULT_BYTES == 2 << 30
    assert launch_bootstrap._MAX_PRODUCER_TOTAL_BYTES == 8 << 30
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={f"member-{index}.bin": b"x" for index in range(4)} | {"result.json": b"result"},
    )
    manifest_path = producer / "producer-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for row in manifest["files"]:
        row["bytes"] = (
            1
            if row["path"] == "result.json"
            else launch_bootstrap._MAX_PRODUCER_FILE_BYTES
        )
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="aggregate byte limit",
    ):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_streaming_manifest_enforces_capacity_result_limit(
    tmp_path: Path,
) -> None:
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={"result.json": b"result"},
    )
    manifest_path = producer / "producer-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0]["bytes"] = launch_bootstrap._MAX_RESULT_BYTES + 1
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(launch_bootstrap.LaunchError, match="manifest row"):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


def test_producer_tree_snapshot_bounds_entries_before_sorting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "producer"
    root.mkdir()
    for index in range(3):
        (root / f"member-{index}.bin").write_bytes(b"x")
    monkeypatch.setattr(
        launch_bootstrap,
        "_MAX_PRODUCER_TREE_ENTRIES",
        2,
    )

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="tree exceeds its entry limit",
    ):
        launch_bootstrap._producer_tree_snapshot(root)


def test_outer_streaming_rejects_short_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={"result.json": b"abcdefgh"},
    )
    path = producer / "result.json"
    target_inode = path.stat().st_ino
    original = os.pread

    def short_read(
        descriptor: int,
        count: int,
        offset: int,
    ) -> bytes:
        if os.fstat(descriptor).st_ino != target_inode:
            return original(descriptor, count, offset)
        if offset >= 4:
            return b""
        return original(descriptor, min(count, 4), offset)

    monkeypatch.setattr(launch_bootstrap.os, "pread", short_read)
    with pytest.raises(launch_bootstrap.LaunchError, match="ended while streamed"):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "mutation",
    ("growth", "shrink", "inode-swap"),
)
def test_outer_streaming_rejects_in_read_path_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    payload = b"x" * 8192
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={"result.json": payload},
    )
    path = producer / "result.json"
    target_inode = path.stat().st_ino
    original = os.pread
    mutated = False

    def mutating_read(
        descriptor: int,
        count: int,
        offset: int,
    ) -> bytes:
        nonlocal mutated
        chunk = original(descriptor, count, offset)
        if not mutated and os.fstat(descriptor).st_ino == target_inode:
            mutated = True
            if mutation == "growth":
                with path.open("ab") as stream:
                    stream.write(b"y")
            elif mutation == "shrink":
                os.truncate(path, len(payload) // 2)
            else:
                displaced = path.with_name("displaced.bin")
                path.rename(displaced)
                path.write_bytes(payload)
        return chunk

    monkeypatch.setattr(launch_bootstrap.os, "pread", mutating_read)
    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize("mutation", ("symlink", "hardlink"))
def test_outer_streaming_rejects_symlink_and_hardlink(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, completion_root, producer = _development_producer(
        tmp_path,
        files={"result.json": b"payload"},
    )
    result = producer / "result.json"
    if mutation == "symlink":
        target = tmp_path / "target.bin"
        result.rename(target)
        result.symlink_to(target)
    else:
        os.link(result, tmp_path / "hardlink.bin")

    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._verify_development_producer(
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize("mutation", ("omitted", "tampered", "extra-link"))
def test_outer_rejects_invalid_terminal_bound_qualification_custody(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    qualification = binding_path.with_name("development-result-qualification.json")
    if mutation == "omitted":
        qualification.unlink()
    elif mutation == "tampered":
        qualification.write_bytes(qualification.read_bytes() + b"x")
    else:
        os.link(qualification, tmp_path / "qualification-hardlink.json")

    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "protocol",
        "raw-result",
        "lane",
        "claim-eligible",
        "master-seed",
        "budget-count",
        "producer-execution",
    ),
)
def test_terminal_bound_qualification_rejects_semantic_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    binding = json.loads(binding_path.read_bytes())
    qualification_path = binding_path.with_name("development-result-qualification.json")
    qualification = json.loads(qualification_path.read_bytes())
    expected_raw_result = qualification["raw_result_sha256"]
    if mutation == "schema":
        qualification["schema"] = "prospect.invalid"
    elif mutation == "protocol":
        qualification["protocol_version"] = "0.0.0"
    elif mutation == "raw-result":
        qualification["raw_result_sha256"] = "0" * 64
    elif mutation == "lane":
        qualification["lane"] = "formal"
    elif mutation == "claim-eligible":
        qualification["claim_eligible"] = True
    elif mutation == "master-seed":
        qualification["replicates"][0]["master_seed"] += 1
    elif mutation == "budget-count":
        qualification["replicates"][0]["transitions"] -= 1
    elif mutation == "producer-execution":
        qualification["producer_execution"]["fixture"] = "substituted"
    else:  # pragma: no cover - parameter exhaustiveness
        raise AssertionError(mutation)
    payload = _canonical(qualification)
    binding["development_qualification"]["result_qualification_sha256"] = hashlib.sha256(payload).hexdigest()

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="qualification is malformed or misbound",
    ):
        launch_bootstrap._recorded_result_qualification(
            payload,
            binding=binding,
            raw_result_sha256=expected_raw_result,
        )


def test_outer_rejects_binding_attempt_without_preflight_receipt(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    preflight = binding_path.with_name("formal-input-preflight.json")
    preflight.unlink()
    manifest = json.loads(terminal.read_bytes())
    manifest["files"] = [row for row in manifest["files"] if row["path"] != "formal-input-preflight.json"]
    manifest["file_count"] = len(manifest["files"])
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="formal input preflight receipt",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize("alias", [True, "float"])
def test_outer_rejects_numeric_preflight_binding_byte_alias(
    tmp_path: Path,
    alias: object,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    preflight_path = binding_path.with_name("formal-input-preflight.json")
    receipt = json.loads(preflight_path.read_bytes())
    receipt["binding_bytes"] = True if alias is True else float(binding_path.stat().st_size)
    payload = _canonical(receipt)
    preflight_path.write_bytes(payload)
    manifest = json.loads(terminal.read_bytes())
    row = next(item for item in manifest["files"] if item["path"] == "formal-input-preflight.json")
    row["bytes"] = len(payload)
    row["sha256"] = hashlib.sha256(payload).hexdigest()
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="preflight receipt is malformed or misbound",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_rejects_terminal_bound_false_preflight_receipt(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    preflight = binding_path.with_name("formal-input-preflight.json")
    receipt = json.loads(preflight.read_bytes())
    receipt["passed"] = False
    payload = _canonical(receipt)
    preflight.write_bytes(payload)
    manifest = json.loads(terminal.read_bytes())
    row = next(item for item in manifest["files"] if item["path"] == "formal-input-preflight.json")
    row["bytes"] = len(payload)
    row["sha256"] = hashlib.sha256(payload).hexdigest()
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="malformed or misbound",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "field",
    (
        "accepted_closure_evidence_sha256",
        "runtime_conformance_sha256",
    ),
)
def test_outer_rejects_terminal_bound_preflight_runtime_digest_substitution(
    tmp_path: Path,
    field: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    preflight = binding_path.with_name("formal-input-preflight.json")
    receipt = json.loads(preflight.read_bytes())
    receipt[field] = "f" * 64
    payload = _canonical(receipt)
    preflight.write_bytes(payload)
    manifest = json.loads(terminal.read_bytes())
    row = next(item for item in manifest["files"] if item["path"] == "formal-input-preflight.json")
    row["bytes"] = len(payload)
    row["sha256"] = hashlib.sha256(payload).hexdigest()
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="authorization inputs differ",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_rejects_substituted_binding_authorization_input(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    manifest = json.loads(terminal.read_bytes())
    manifest["inputs"][0]["sha256"] = "0" * 64
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="authorization inputs differ",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_rejects_substituted_closure_authorization_input(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, binding_terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    results = repository / "bench" / "world_model_lifecycle" / "results"
    closure_terminal = results / "operator-v1.20" / "closures" / "development-closure-v1.20.0" / "operator-attempt.json"
    closure_completion = _completion_marker(
        completion_root,
        closure_terminal,
    )
    closure_manifest = json.loads(closure_terminal.read_bytes())
    closure_manifest["inputs"][0]["sha256"] = "0" * 64
    closure_terminal.write_bytes(_canonical(closure_manifest))
    closure_payload = closure_terminal.read_bytes()

    binding_manifest = json.loads(binding_terminal.read_bytes())
    for row in binding_manifest["inputs"]:
        if row["path"] in {
            str(closure_terminal),
            str(closure_completion),
        }:
            row["bytes"] = len(closure_payload)
            row["sha256"] = hashlib.sha256(closure_payload).hexdigest()
    binding_terminal.write_bytes(_canonical(binding_manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="closure authorization inputs differ",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    ("terminal_state", "message"),
    [
        ("unfinalized", "exactly 2 link"),
        ("copied-marker", "not the terminal inode"),
    ],
)
def test_outer_rejects_unfinalized_or_copied_binding_completion(
    tmp_path: Path,
    terminal_state: str,
    message: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state=terminal_state,
    )

    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_rejects_direct_or_self_finalized_formal_binding_copy(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    results = completion_root.parents[1]
    copied = results / "formal-binding-copy.json"
    copied.write_bytes(_canonical(_minimal_formal_binding()))

    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._open_typed_runtime_custody(
            copied,
            repository=repository,
            completion_root=completion_root,
        )

    copied_marker = _completion_marker(completion_root, copied)
    os.link(copied, copied_marker)
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="not one canonical",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            copied,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--bootstrap",
            "/tmp/producer-a.py",
            "--bootstrap",
            "/tmp/producer-b.py",
            "--runtime-seal",
            "/tmp/runtime.json",
            "preformal-runtime",
        ],
        [
            "--bootstrap",
            "/tmp/producer.py",
            "--runtime-seal",
            "/tmp/runtime-a.json",
            "--runtime-seal",
            "/tmp/runtime-b.json",
            "preformal-runtime",
        ],
        [
            "--bootstrap",
            "/tmp/producer.py",
            "--runtime-seal",
            "/tmp/runtime.json",
            "--rehearse-accepted-binding",
            "/tmp/binding.json",
        ],
        [
            "--bootstrap",
            "/tmp/producer.py",
            "--rehearse-accepted-binding",
            "/tmp/binding-a.json",
            "--rehearse-accepted-binding",
            "/tmp/binding-b.json",
        ],
    ],
)
def test_outer_parser_rejects_repeated_custody_options(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        launch_bootstrap._parser().parse_args(arguments)


@pytest.mark.parametrize(
    ("producer_arguments", "message"),
    [
        (["--", "--", "preformal-runtime"], "ambiguous separator"),
        (
            ["--", "--runtime-seal=/tmp/second.json", "preformal-runtime"],
            "second custody source",
        ),
        (
            ["--", "--bootstrap-fd", "7", "preformal-runtime"],
            "second custody source",
        ),
    ],
)
def test_outer_run_rejects_ambiguous_or_mixed_custody_before_open(
    tmp_path: Path,
    producer_arguments: list[str],
    message: str,
) -> None:
    arguments = argparse.Namespace(
        bootstrap=tmp_path / "absent-producer.py",
        runtime_seal=tmp_path / "absent-runtime.json",
        create_runtime_seal=None,
        producer_arguments=producer_arguments,
    )
    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=tmp_path,
            completion_root=tmp_path,
        )


def _captured_rehearsal_child(
    stdout: bytes,
    calls: list[list[str]],
    *,
    returncode: int = 0,
    stderr: bytes = b"",
    outer_receipt: bytes = b"",
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(
        command: list[str],
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        receipt_index = command.index("--outer-receipt-fd") + 1
        receipt_descriptor = int(command[receipt_index])
        if outer_receipt:
            os.write(receipt_descriptor, outer_receipt)
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def _accepted_rehearsal(
    *,
    repository: Path,
    completion_root: Path,
    binding_path: Path,
    binding: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(
            _rehearsal_output(binding),
            calls,
        ),
    )
    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 0
    )
    assert len(calls) == 1


def _binding_arguments(
    repository: Path,
    binding_path: Path,
    producer_arguments: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        bootstrap=(repository / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"),
        runtime_seal=binding_path,
        create_runtime_seal=None,
        rehearse_accepted_binding=None,
        producer_arguments=producer_arguments,
    )


def test_accepted_binding_rehearsal_publishes_one_authenticated_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    development = binding["development_qualification"]
    assert isinstance(development, dict)
    binding_schema = json.loads(
        (
            Path(binding_module.__file__).with_name("schemas")
            / "formal-binding.schema.json"
        ).read_bytes()
    )
    development_schema = binding_schema["properties"]["development_qualification"]
    assert development_schema["additionalProperties"] is False
    assert set(development_schema["required"]) == set(development_schema["properties"])
    assert set(development) == set(development_schema["required"])
    assert "matrix_contract_sha256" not in development
    expected_stdout = _rehearsal_output(binding)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(
            expected_stdout,
            calls,
        ),
    )

    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0][-4:] == [
        "preformal-runtime",
        "bootstrap-inventory-conformance",
        "--device",
        "cpu",
    ]
    stored_stdout = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.20"
        / "rehearsals"
        / "accepted-binding-rehearsal-v1.20.0"
        / "rehearsal.stdout.json"
    ).read_bytes()
    assert stored_stdout == expected_stdout
    local = launch_bootstrap._verify_accepted_binding_rehearsal(
        binding_path,
        repository=repository,
        completion_root=completion_root,
    )
    independent_terminal = rehearsal_module.verify_accepted_binding_rehearsal(binding_path)
    independent = rehearsal_module.accepted_binding_rehearsal_identity_rows(binding_path)
    assert independent_terminal["status"] == "accepted"
    assert local == independent


def test_rehearsal_claim_precedes_full_authorization_and_failure_has_no_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, _ = _rehearsal_context(
        tmp_path,
        monkeypatch,
    )
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    paths = launch_bootstrap._rehearsal_paths(
        repository,
        binding_sha256=binding_sha256,
        completion_root=completion_root,
    )

    def refuse_after_claim(*_args: object, **_kwargs: object) -> None:
        assert os.path.samefile(paths["claim"], paths["claim_marker"])
        assert paths["claim"].stat().st_nlink == 2
        raise launch_bootstrap.LaunchError("forced authorization failure")

    monkeypatch.setattr(
        launch_bootstrap,
        "_open_typed_runtime_custody",
        refuse_after_claim,
    )
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("child must not run"),
    )

    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 2
    )
    terminal = rehearsal_module.verify_accepted_binding_rehearsal(binding_path)
    assert terminal["status"] == "failed"
    assert terminal["child_started"] is False
    assert terminal["phase"] == "pre_dispatch"


@pytest.mark.parametrize(
    ("returncode", "stderr", "outer_receipt", "stdout_mutation", "phase"),
    (
        (1, b"", b"", False, "output_validation"),
        (0, b"warning\n", b"", False, "output_validation"),
        (0, b"", b"unexpected", False, "output_validation"),
        (0, b"", b"", True, "output_validation"),
    ),
    ids=("returncode", "stderr", "outer-receipt", "semantic-stdout"),
)
def test_rehearsal_child_or_output_failure_is_terminal_and_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: bytes,
    outer_receipt: bytes,
    stdout_mutation: bool,
    phase: str,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    stdout = _rehearsal_output(binding)
    if stdout_mutation:
        value = json.loads(stdout)
        value["repeat_count"] = 2
        stdout = _canonical(value)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(
            stdout,
            calls,
            returncode=returncode,
            stderr=stderr,
            outer_receipt=outer_receipt,
        ),
    )
    arguments = _rehearsal_arguments(repository, binding_path)

    assert (
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 2
    )
    assert len(calls) == 1
    terminal = rehearsal_module.verify_accepted_binding_rehearsal(binding_path)
    assert terminal["status"] == "failed"
    assert terminal["phase"] == phase
    with pytest.raises(launch_bootstrap.LaunchError, match="already finalized"):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("failpoint", "recovery_status", "recovery_code", "child_calls"),
    (
        ("before_claim_marker", "failed", 2, 0),
        ("after_claim_marker", "failed", 2, 0),
        ("after_child_start", "failed", 2, 0),
        ("after_sidecars", "failed", 2, 1),
        ("after_terminal_staging", "failed", 2, 1),
        ("after_terminal_hardlink", "accepted", 0, 1),
        ("after_staging_cleanup", "accepted", 0, 1),
    ),
)
def test_rehearsal_crash_recovery_never_dispatches_a_second_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
    recovery_status: str,
    recovery_code: int,
    child_calls: int,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(_rehearsal_output(binding), calls),
    )

    def crash(name: str) -> None:
        if name == failpoint:
            raise KeyboardInterrupt(name)

    monkeypatch.setattr(launch_bootstrap, "_rehearsal_failpoint", crash)
    arguments = _rehearsal_arguments(repository, binding_path)
    with pytest.raises(KeyboardInterrupt, match=failpoint):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(calls) == child_calls

    monkeypatch.setattr(
        launch_bootstrap,
        "_rehearsal_failpoint",
        lambda _name: None,
    )
    assert (
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == recovery_code
    )
    assert len(calls) == child_calls
    terminal = rehearsal_module.verify_accepted_binding_rehearsal(binding_path)
    assert terminal["status"] == recovery_status
    if recovery_status == "accepted":
        normal_arguments = argparse.Namespace(
            bootstrap=(repository / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"),
            runtime_seal=binding_path,
            create_runtime_seal=None,
            rehearse_accepted_binding=None,
            producer_arguments=["preformal-runtime"],
        )
        assert (
            launch_bootstrap._run_locked(
                normal_arguments,
                environment={},
                repository=repository,
                completion_root=completion_root,
            )
            == 0
        )
        assert len(calls) == child_calls + 1


def test_rehearsal_crash_after_completion_never_reopens_or_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(_rehearsal_output(binding), calls),
    )

    def crash(name: str) -> None:
        if name == "after_completion_hardlink":
            raise KeyboardInterrupt(name)

    monkeypatch.setattr(launch_bootstrap, "_rehearsal_failpoint", crash)
    arguments = _rehearsal_arguments(repository, binding_path)
    with pytest.raises(KeyboardInterrupt, match="after_completion_hardlink"):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(calls) == 1
    assert rehearsal_module.verify_accepted_binding_rehearsal(binding_path)["status"] == "accepted"
    with pytest.raises(launch_bootstrap.LaunchError, match="already finalized"):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(calls) == 1
    normal_arguments = argparse.Namespace(
        bootstrap=(repository / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"),
        runtime_seal=binding_path,
        create_runtime_seal=None,
        rehearse_accepted_binding=None,
        producer_arguments=["preformal-runtime"],
    )
    assert (
        launch_bootstrap._run_locked(
            normal_arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 0
    )
    assert len(calls) == 2


def test_attempt_directory_only_crash_is_terminally_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(_rehearsal_output(binding), calls),
    )

    def crash(name: str) -> None:
        if name == "after_attempt_mkdir":
            raise KeyboardInterrupt(name)

    monkeypatch.setattr(launch_bootstrap, "_rehearsal_failpoint", crash)
    arguments = _rehearsal_arguments(repository, binding_path)
    with pytest.raises(KeyboardInterrupt, match="after_attempt_mkdir"):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    monkeypatch.setattr(
        launch_bootstrap,
        "_rehearsal_failpoint",
        lambda _name: None,
    )
    with pytest.raises(launch_bootstrap.LaunchError, match="no durable claim"):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert calls == []
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    paths = launch_bootstrap._rehearsal_paths(
        repository,
        binding_sha256=binding_sha256,
        completion_root=completion_root,
    )
    assert not paths["terminal"].exists()
    assert not paths["completion"].exists()


def test_rehearsal_attempt_sibling_refuses_before_claim_or_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, _ = _rehearsal_context(
        tmp_path,
        monkeypatch,
    )
    attempts = repository / "bench" / "world_model_lifecycle" / "results" / "operator-v1.20" / "rehearsals"
    attempts.mkdir(parents=True)
    (attempts / ".stale-rehearsal").write_bytes(b"stale\n")
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("contaminated attempts root must prevent child dispatch"),
    )
    with pytest.raises(launch_bootstrap.LaunchError, match="contaminated"):
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )


def test_accepted_unfinalized_recovery_refuses_new_formal_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(_rehearsal_output(binding), calls),
    )

    def crash(name: str) -> None:
        if name == "after_terminal_hardlink":
            raise KeyboardInterrupt(name)

    monkeypatch.setattr(launch_bootstrap, "_rehearsal_failpoint", crash)
    arguments = _rehearsal_arguments(repository, binding_path)
    with pytest.raises(KeyboardInterrupt, match="after_terminal_hardlink"):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    formal = (
        repository / "bench" / "world_model_lifecycle" / "results" / "formal" / "other-binding" / "confirmation-v1.20.0"
    )
    formal.mkdir(parents=True)
    monkeypatch.setattr(
        launch_bootstrap,
        "_rehearsal_failpoint",
        lambda _name: None,
    )
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="found formal authority",
    ):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(calls) == 1
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    paths = launch_bootstrap._rehearsal_paths(
        repository,
        binding_sha256=binding_sha256,
        completion_root=completion_root,
    )
    assert paths["terminal"].exists()
    assert paths["terminal"].stat().st_nlink == 1
    assert not paths["completion"].exists()


@pytest.mark.parametrize(
    "mutation",
    ("stdout", "stdout-alias", "completion-copy", "extra-file"),
)
def test_launcher_and_independent_rehearsal_verifiers_reject_same_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(_rehearsal_output(binding), calls),
    )
    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 0
    )
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    paths = launch_bootstrap._rehearsal_paths(
        repository,
        binding_sha256=binding_sha256,
        completion_root=completion_root,
    )
    if mutation == "stdout":
        paths["stdout"].write_bytes(b"{}\n")
    elif mutation == "stdout-alias":
        os.link(paths["stdout"], repository / "rehearsal-stdout-alias")
    elif mutation == "completion-copy":
        paths["completion"].unlink()
        shutil.copyfile(paths["terminal"], paths["completion"])
    elif mutation == "extra-file":
        (paths["attempt"] / "unexpected").write_bytes(b"")
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)

    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._verify_accepted_binding_rehearsal(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )
    with pytest.raises(rehearsal_module.RehearsalEvidenceError):
        rehearsal_module.verify_accepted_binding_rehearsal(binding_path)


def test_normal_formal_dispatch_requires_accepted_unmutated_rehearsal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    normal_arguments = argparse.Namespace(
        bootstrap=(repository / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"),
        runtime_seal=binding_path,
        create_runtime_seal=None,
        rehearse_accepted_binding=None,
        producer_arguments=["preformal-runtime"],
    )
    calls: list[list[str]] = []

    def unexpected_formal_child(*_args: object, **_kwargs: object) -> None:
        calls.append([])
        pytest.fail("formal child ran before accepted rehearsal")

    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        unexpected_formal_child,
    )
    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._run_locked(
            normal_arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert calls == []

    rehearsal_calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(
            _rehearsal_output(binding),
            rehearsal_calls,
        ),
    )
    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 0
    )
    assert len(rehearsal_calls) == 1

    normal_calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(b"{}\n", normal_calls),
    )
    assert (
        launch_bootstrap._run_locked(
            normal_arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 0
    )
    assert len(normal_calls) == 1

    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    paths = launch_bootstrap._rehearsal_paths(
        repository,
        binding_sha256=binding_sha256,
        completion_root=completion_root,
    )
    paths["stdout"].write_bytes(b"changed\n")
    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._run_locked(
            normal_arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(normal_calls) == 1


def test_failed_rehearsal_never_authorizes_normal_formal_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    invalid = json.loads(_rehearsal_output(binding))
    invalid["repeat_count"] = 2
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(_canonical(invalid), calls),
    )
    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 2
    )
    assert len(calls) == 1
    normal_arguments = argparse.Namespace(
        bootstrap=(repository / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"),
        runtime_seal=binding_path,
        create_runtime_seal=None,
        rehearse_accepted_binding=None,
        producer_arguments=["preformal-runtime"],
    )
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("failed rehearsal must not authorize a child"),
    )
    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._run_locked(
            normal_arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "candidate",
    (
        "current-binding-root",
        "alternate-binding",
        "nested-confirmation",
        "symlink-confirmation",
        "audit-staging",
        "adjudication-staging",
    ),
)
def test_rehearsal_formal_absence_scan_rejects_every_current_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str,
) -> None:
    repository, completion_root, binding_path, _ = _rehearsal_context(
        tmp_path,
        monkeypatch,
    )
    results = repository / "bench" / "world_model_lifecycle" / "results"
    formal = results / "formal"
    formal.mkdir(parents=True)
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    if candidate == "current-binding-root":
        (formal / binding_sha256).mkdir()
    elif candidate == "alternate-binding":
        path = formal / ("b" * 64) / "confirmation-v1.20.0"
        path.mkdir(parents=True)
    elif candidate == "nested-confirmation":
        path = formal / "nonhex-root" / "confirmation-v1.20.0"
        path.mkdir(parents=True)
    elif candidate == "symlink-confirmation":
        root = formal / "other-binding"
        root.mkdir()
        (repository / "target").mkdir()
        os.symlink(
            repository / "target",
            root / "confirmation-v1.20.0",
        )
    elif candidate == "audit-staging":
        path = results / "operator-v1.20" / "audits" / ".formal-audit-v1.20.0.staging"
        path.mkdir(parents=True)
    elif candidate == "adjudication-staging":
        path = results / "adjudication-v1.20" / ".formal-adjudication-v1.20.0.staging-123"
        path.mkdir(parents=True)
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(candidate)
    assert not launch_bootstrap._formal_rehearsal_paths_absent(
        repository,
        binding_sha256=binding_sha256,
    )
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("current formal path must prevent rehearsal child dispatch"),
    )
    assert (
        launch_bootstrap._run_locked(
            _rehearsal_arguments(repository, binding_path),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 2
    )
    assert rehearsal_module.verify_accepted_binding_rehearsal(binding_path)["status"] == "failed"


def test_rehearsal_formal_absence_scan_preserves_retired_evidence_and_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    formal = repository / "bench" / "world_model_lifecycle" / "results" / "formal"
    retired = formal / ("b" * 64) / "confirmation-v1.17.0"
    retired.mkdir(parents=True)
    (formal / "formal-launch-v1.17.0.json").write_bytes(b"retired\n")
    assert launch_bootstrap._formal_rehearsal_paths_absent(
        repository,
        binding_sha256="a" * 64,
    )

    (formal / "second-retired-entry").write_bytes(b"")
    monkeypatch.setattr(
        launch_bootstrap,
        "_MAX_PRODUCER_TREE_ENTRIES",
        1,
    )
    assert not launch_bootstrap._formal_rehearsal_paths_absent(
        repository,
        binding_sha256="a" * 64,
    )


def test_rehearsal_formal_absence_scan_fails_closed_on_scan_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    formal = repository / "bench" / "world_model_lifecycle" / "results" / "formal"
    formal.mkdir(parents=True)
    original = os.scandir

    def fail(path: str | os.PathLike[str]) -> Any:
        if Path(path) == formal:
            raise OSError("forced scan failure")
        return original(path)

    monkeypatch.setattr(launch_bootstrap.os, "scandir", fail)
    assert not launch_bootstrap._formal_rehearsal_paths_absent(
        repository,
        binding_sha256="a" * 64,
    )


def test_formal_launch_dispatch_rechecks_one_pristine_prepared_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    _accepted_rehearsal(
        repository=repository,
        completion_root=completion_root,
        binding_path=binding_path,
        binding=binding,
        monkeypatch=monkeypatch,
    )
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    formal_root = repository / "bench" / "world_model_lifecycle" / "results" / "formal"
    (formal_root / binding_sha256).mkdir(parents=True)
    retired = formal_root / ("b" * 64) / "confirmation-v1.17.0"
    retired.mkdir(parents=True)
    (formal_root / "formal-launch-v1.17.0.json").write_bytes(b"retired\n")

    readiness_calls: list[str] = []
    original_ready = launch_bootstrap._formal_launch_paths_ready

    def record_ready(
        path: Path,
        *,
        binding_sha256: str,
    ) -> bool:
        readiness_calls.append(binding_sha256)
        return original_ready(path, binding_sha256=binding_sha256)

    monkeypatch.setattr(
        launch_bootstrap,
        "_formal_launch_paths_ready",
        record_ready,
    )
    child_calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(
            b"",
            child_calls,
            returncode=1,
        ),
    )

    assert (
        launch_bootstrap._run_locked(
            _binding_arguments(repository, binding_path, ["formal"]),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 1
    )
    assert readiness_calls == [binding_sha256, binding_sha256]
    assert len(child_calls) == 1


@pytest.mark.parametrize(
    "candidate",
    (
        "formal-root-symlink",
        "binding-root-symlink",
        "current-child",
        "root-confirmation",
        "alternate-confirmation",
        "symlink-confirmation",
        "formal-marker",
        "audit-final",
        "audit-staging",
        "semantic-review",
        "adjudication-marker",
        "adjudication-final",
        "adjudication-staging",
    ),
)
def test_formal_launch_contamination_refuses_before_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    _accepted_rehearsal(
        repository=repository,
        completion_root=completion_root,
        binding_path=binding_path,
        binding=binding,
        monkeypatch=monkeypatch,
    )
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    results = repository / "bench" / "world_model_lifecycle" / "results"
    formal_root = results / "formal"
    binding_root = formal_root / binding_sha256
    if candidate == "formal-root-symlink":
        target = repository / "formal-target"
        (target / binding_sha256).mkdir(parents=True)
        formal_root.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, formal_root, target_is_directory=True)
    else:
        formal_root.mkdir(parents=True)
        if candidate == "binding-root-symlink":
            target = repository / "binding-target"
            target.mkdir()
            os.symlink(target, binding_root, target_is_directory=True)
        else:
            binding_root.mkdir()

    if candidate in {"formal-root-symlink", "binding-root-symlink"}:
        pass
    elif candidate == "current-child":
        (binding_root / "unexpected").write_bytes(b"")
    elif candidate == "root-confirmation":
        (formal_root / "confirmation-v1.20.0").mkdir()
    elif candidate == "alternate-confirmation":
        (formal_root / ("c" * 64) / "confirmation-v1.20.0").mkdir(parents=True)
    elif candidate == "symlink-confirmation":
        alternate = formal_root / "alternate"
        target = repository / "confirmation-target"
        alternate.mkdir()
        target.mkdir()
        os.symlink(
            target,
            alternate / "confirmation-v1.20.0",
            target_is_directory=True,
        )
    elif candidate == "formal-marker":
        (formal_root / "formal-launch-v1.20.0.json").write_bytes(b"")
    elif candidate == "audit-final":
        (results / "operator-v1.20" / "audits" / "formal-audit-v1.20.0").mkdir(parents=True)
    elif candidate == "audit-staging":
        (results / "operator-v1.20" / "audits" / ".formal-audit-v1.20.0.staging").mkdir(parents=True)
    elif candidate == "semantic-review":
        review = repository / "artifacts" / "wm001-reviews" / "formal-v1.20.0.json"
        review.parent.mkdir(parents=True)
        review.write_bytes(b"")
    elif candidate == "adjudication-marker":
        (formal_root / "formal-adjudication-v1.20.0.json").write_bytes(b"")
    elif candidate == "adjudication-final":
        (results / "adjudication-v1.20" / "formal-adjudication-v1.20.0").mkdir(parents=True)
    elif candidate == "adjudication-staging":
        (results / "adjudication-v1.20" / ".formal-adjudication-v1.20.0.staging-race").mkdir(parents=True)
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(candidate)

    assert not launch_bootstrap._formal_launch_paths_ready(
        repository,
        binding_sha256=binding_sha256,
    )
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("contaminated formal namespace must prevent child dispatch"),
    )
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="formal launch namespace is not pristine",
    ):
        launch_bootstrap._run_locked(
            _binding_arguments(repository, binding_path, ["formal"]),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )


def test_formal_launch_second_readiness_check_closes_dispatch_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    _accepted_rehearsal(
        repository=repository,
        completion_root=completion_root,
        binding_path=binding_path,
        binding=binding,
        monkeypatch=monkeypatch,
    )
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    binding_root = repository / "bench" / "world_model_lifecycle" / "results" / "formal" / binding_sha256
    binding_root.mkdir(parents=True)
    original_ready = launch_bootstrap._formal_launch_paths_ready
    readiness_calls = 0

    def race_ready(
        path: Path,
        *,
        binding_sha256: str,
    ) -> bool:
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls == 2:
            (binding_root / "raced").write_bytes(b"")
        return original_ready(path, binding_sha256=binding_sha256)

    monkeypatch.setattr(
        launch_bootstrap,
        "_formal_launch_paths_ready",
        race_ready,
    )
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("second readiness check must prevent child dispatch"),
    )
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="changed before child dispatch",
    ):
        launch_bootstrap._run_locked(
            _binding_arguments(repository, binding_path, ["formal"]),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert readiness_calls == 2


@pytest.mark.parametrize(
    "producer_arguments",
    (["--audit-entry", "formal"], ["--adjudication-entry"]),
    ids=("formal-audit", "adjudication"),
)
def test_later_publishers_do_not_reenter_preformal_readiness_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer_arguments: list[str],
) -> None:
    repository, completion_root, binding_path, binding = _rehearsal_context(tmp_path, monkeypatch)
    _accepted_rehearsal(
        repository=repository,
        completion_root=completion_root,
        binding_path=binding_path,
        binding=binding,
        monkeypatch=monkeypatch,
    )
    binding_sha256 = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    confirmation = (
        repository / "bench" / "world_model_lifecycle" / "results" / "formal" / binding_sha256 / "confirmation-v1.20.0"
    )
    confirmation.mkdir(parents=True)
    monkeypatch.setattr(
        launch_bootstrap,
        "_formal_launch_paths_ready",
        lambda *_args, **_kwargs: pytest.fail("later publishers own their post-formal namespace gates"),
    )
    child_calls: list[list[str]] = []
    monkeypatch.setattr(
        launch_bootstrap.subprocess,
        "run",
        _captured_rehearsal_child(
            b"",
            child_calls,
            returncode=1,
        ),
    )
    assert (
        launch_bootstrap._run_locked(
            _binding_arguments(
                repository,
                binding_path,
                producer_arguments,
            ),
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
        == 1
    )
    assert len(child_calls) == 1


def test_producer_rejects_second_custody_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "producer_bootstrap.py",
            "--binding-entry",
            "--outer-receipt-fd=91",
        ],
    )
    with pytest.raises(
        producer_bootstrap.BootstrapError,
        match="second custody source",
    ):
        producer_bootstrap._reject_second_custody_source()


def test_runtime_lock_is_repository_wide_across_working_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "canonical-repository"
    repository.mkdir()
    first_descriptor = launch_bootstrap._acquire_runtime_lock(repository)
    other_working_directory = tmp_path / "other-working-directory"
    other_working_directory.mkdir()
    monkeypatch.chdir(other_working_directory)
    try:
        with pytest.raises(
            launch_bootstrap.LaunchError,
            match="another WM-001 v1.20 outer invocation",
        ):
            launch_bootstrap._acquire_runtime_lock(repository)
    finally:
        fcntl.flock(first_descriptor, fcntl.LOCK_UN)
        os.close(first_descriptor)


def _run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed: {command!r}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def _install_minimal_distribution(virtualenv: Path) -> None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    package_root = virtualenv / "lib" / version / "site-packages"
    bench = package_root / "bench"
    lifecycle = bench / "world_model_lifecycle"
    metadata = package_root / "wm001_bootstrap_fixture-1.0.dist-info"
    lifecycle.mkdir(parents=True)
    metadata.mkdir()
    (bench / "__init__.py").write_text(
        '"""WM-001 bootstrap fixture namespace."""\n',
        encoding="utf-8",
    )
    (lifecycle / "__init__.py").write_text(
        '"""WM-001 bootstrap fixture lifecycle."""\n',
        encoding="utf-8",
    )
    shutil.copyfile(
        Path(producer_bootstrap.__file__),
        lifecycle / "producer_bootstrap.py",
    )
    (lifecycle / "_fixture_entry.py").write_text(
        "import hashlib\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from bench.world_model_lifecycle.producer_bootstrap import (\n"
        "    register_outer_terminal,\n"
        ")\n"
        "\n"
        "\n"
        "LOGICAL_CODES = {\n"
        "    'binding': 0,\n"
        "    'audit': 1,\n"
        "    'closure': 2,\n"
        "    'adjudication': 0,\n"
        "    'default-run': 1,\n"
        "}\n"
        "\n"
        "\n"
        "def _payload(value):\n"
        "    return (\n"
        "        json.dumps(value, sort_keys=True, separators=(',', ':'))\n"
        "        + '\\n'\n"
        "    )\n"
        "\n"
        "\n"
        "def _digest(value):\n"
        "    return hashlib.sha256(json.dumps(\n"
        "        value, sort_keys=True, separators=(',', ':')\n"
        "    ).encode('utf-8')).hexdigest()\n"
        "\n"
        "\n"
        "def runtime_conformance():\n"
        "    binding = json.loads(sys._prospect_wm001_runtime_seal_payload)\n"
        "    dependencies = binding['dependencies']\n"
        "    runtime = binding['runtime']\n"
        "    audit = binding['audit_execution']\n"
        "    inventory = {\n"
        "        'packages': dependencies['packages'],\n"
        "        'package_roots': dependencies['package_roots'],\n"
        "        'standard_library': dependencies['standard_library'],\n"
        "        'package_ownership': dependencies['package_ownership'],\n"
        "    }\n"
        "    fresh = {\n"
        "        'schema': 'prospect.wm001.fresh-runtime-identity-conformance.v1',\n"
        "        'experiment_id': 'WM-001',\n"
        "        'protocol_version': '1.20.0',\n"
        "        'mode': 'fresh-identity-conformance',\n"
        "        'challenge': '6' * 64,\n"
        "        'requesting_process_id': 303,\n"
        "        'verifier_process_id': 404,\n"
        "        'matrix_contract_sha256': '09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84',\n"
        "        'passed': True,\n"
        "    }\n"
        "    sys.stdout.write(_payload({\n"
        "        'schema': 'prospect.wm001.preformal-runtime-check.v1',\n"
        "        'mode': 'bootstrap-inventory-conformance',\n"
        "        'device': runtime['device'],\n"
        "        'passed': True,\n"
        "        'inventory': inventory,\n"
        "        'inventory_sha256': _digest(inventory),\n"
        "        'conformance_sha256': _digest(audit),\n"
        "        'fresh_runtime_identity_conformance': fresh,\n"
        "        'fresh_runtime_identity_conformance_sha256': _digest(fresh),\n"
        "        'restart_runtime_conformance_report_sha256': audit['restart_runtime_conformance_report_sha256'],\n"
        "        'restart_runtime_execution_receipt_sha256': audit['restart_runtime_execution_receipt_sha256'],\n"
        "        'restart_runtime_support_files': audit['restart_runtime_support_files'],\n"
        "        'restart_runtime_repeat_count': 3,\n"
        "        'restart_runtime_path_descriptor_equal': True,\n"
        "        'repeat_count': 3,\n"
        "        'path_descriptor_equal': True,\n"
        "    }))\n"
        "    return 0\n"
        "\n"
        "\n"
        "def read_only(entry):\n"
        "    sys.stdout.write(_payload({\n"
        "        'entry': entry,\n"
        "        'schema': 'prospect.wm001.bootstrap-fixture.v1',\n"
        "    }))\n"
        "    return 0\n"
        "\n"
        "\n"
        "def publish(entry):\n"
        "    logical_exit_code = LOGICAL_CODES[entry]\n"
        "    terminal = (\n"
        "        Path.cwd()\n"
        "        / 'bench'\n"
        "        / 'world_model_lifecycle'\n"
        "        / 'results'\n"
        "        / 'fixture-dispatch'\n"
        "        / entry\n"
        "        / 'terminal-manifest.json'\n"
        "    )\n"
        "    terminal.parent.mkdir(parents=True)\n"
        "    terminal.write_text(_payload({\n"
        "        'entry': entry,\n"
        "        'logical_exit_code': logical_exit_code,\n"
        "        'schema': 'prospect.wm001.bootstrap-fixture-terminal.v1',\n"
        "    }), encoding='utf-8')\n"
        "    register_outer_terminal(\n"
        "        terminal,\n"
        "        logical_exit_code=logical_exit_code,\n"
        "    )\n"
        "    return logical_exit_code\n",
        encoding="utf-8",
    )
    (lifecycle / "restore_eval.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import read_only\n"
        "\n"
        "\n"
        "def main():\n"
        "    return read_only('restore-eval')\n",
        encoding="utf-8",
    )
    (lifecycle / "operator.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import publish\n"
        "\n"
        "\n"
        "def binding_main():\n"
        "    return publish('binding')\n"
        "\n"
        "\n"
        "def audit_main():\n"
        "    return publish('audit')\n"
        "\n"
        "\n"
        "def closure_main():\n"
        "    return publish('closure')\n",
        encoding="utf-8",
    )
    (lifecycle / "adjudication.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import publish\n"
        "\n"
        "\n"
        "def main():\n"
        "    return publish('adjudication')\n",
        encoding="utf-8",
    )
    (lifecycle / "preformal.py").write_text(
        "import sys\n"
        "\n"
        "from bench.world_model_lifecycle._fixture_entry import (\n"
        "    read_only,\n"
        "    runtime_conformance,\n"
        ")\n"
        "\n"
        "\n"
        "def runtime_main():\n"
        "    if sys.argv[1:2] == ['bootstrap-inventory-conformance']:\n"
        "        return runtime_conformance()\n"
        "    return read_only('preformal-runtime')\n",
        encoding="utf-8",
    )
    (lifecycle / "run.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import publish\n"
        "\n"
        "\n"
        "def main():\n"
        "    return publish('default-run')\n",
        encoding="utf-8",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: wm001-bootstrap-fixture\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "bench/__init__.py,,\n"
        "bench/world_model_lifecycle/__init__.py,,\n"
        "bench/world_model_lifecycle/_fixture_entry.py,,\n"
        "bench/world_model_lifecycle/adjudication.py,,\n"
        "bench/world_model_lifecycle/operator.py,,\n"
        "bench/world_model_lifecycle/preformal.py,,\n"
        "bench/world_model_lifecycle/producer_bootstrap.py,,\n"
        "bench/world_model_lifecycle/restore_eval.py,,\n"
        "bench/world_model_lifecycle/run.py,,\n"
        "wm001_bootstrap_fixture-1.0.dist-info/METADATA,,\n"
        "wm001_bootstrap_fixture-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )


def _formal_binding_from_runtime_seal(
    seal: dict[str, object],
) -> dict[str, object]:
    python = seal["python"]
    assert isinstance(python, dict)
    return {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "experiment_id": "WM-001",
        "assurance": dict(_ASSURANCE),
        "source": {
            "git_commit": seal["git_commit"],
            "git_tree": seal["git_tree"],
            "worktree_clean": seal["worktree_clean"],
            "execution_source_sha256": {
                "producer_bootstrap.py": seal["bootstrap_source_sha256"],
            },
        },
        "dependencies": {
            "python_executable": python["executable"],
            "python_executable_sha256": python["sha256"],
            "standard_library": seal["standard_library"],
            "package_roots": seal["package_roots"],
            "package_ownership": seal["package_ownership"],
        },
        "runtime": {
            "python_flags": seal["required_flags"],
            "process_environment": seal["process_environment"],
        },
    }


def test_real_outer_launcher_rehearses_accepted_binding_with_large_producer(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "canonical-repository"
    lifecycle = repository / "bench" / "world_model_lifecycle"
    lifecycle.mkdir(parents=True)
    launcher = lifecycle / "launch_bootstrap.py"
    producer = lifecycle / "producer_bootstrap.py"
    shutil.copyfile(Path(launch_bootstrap.__file__), launcher)
    shutil.copyfile(Path(producer_bootstrap.__file__), producer)
    (lifecycle / "protocol.json").write_text("{}\n", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "/bench/world_model_lifecycle/results/\n",
        encoding="utf-8",
    )

    _run_checked(["git", "init", "--quiet"], cwd=repository)
    _run_checked(
        ["git", "config", "user.email", "wm001-fixture@example.invalid"],
        cwd=repository,
    )
    _run_checked(
        ["git", "config", "user.name", "WM-001 fixture"],
        cwd=repository,
    )
    _run_checked(["git", "add", "."], cwd=repository)
    _run_checked(
        ["git", "commit", "--quiet", "-m", "bootstrap fixture"],
        cwd=repository,
    )

    virtualenv = tmp_path / "isolated-runtime"
    _run_checked(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            str(virtualenv),
        ]
    )
    _install_minimal_distribution(virtualenv)

    runtime_seal = lifecycle / "results" / "development" / "runtime-seal-v1.20.0.json"
    runtime_seal.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    completed = subprocess.run(
        [
            str(virtualenv / "bin" / "python"),
            "-I",
            "-S",
            "-B",
            str(launcher),
            "--bootstrap",
            str(producer),
            "--create-runtime-seal",
            str(runtime_seal),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"

    seal = json.loads(runtime_seal.read_bytes())
    assert seal["schema"] == "prospect.wm001.runtime-seal.v1"
    assert seal["assurance"] == _ASSURANCE
    assert seal["worktree_clean"] is True
    completion_root = lifecycle / "results" / "outer-completions" / "v1.20"
    marker = _completion_marker(completion_root, runtime_seal)
    assert marker.read_bytes() == runtime_seal.read_bytes()
    assert os.path.samefile(marker, runtime_seal)
    assert runtime_seal.stat().st_nlink == 2

    runtime_prefix = [
        str(virtualenv / "bin" / "python"),
        "-I",
        "-S",
        "-B",
        str(launcher),
        "--bootstrap",
        str(producer),
        "--runtime-seal",
        str(runtime_seal),
        "--",
    ]
    read_only_entries = [
        (["--restore-eval-entry"], "restore-eval"),
        (["preformal-runtime"], "preformal-runtime"),
    ]
    for producer_arguments, entry in read_only_entries:
        markers_before = set(completion_root.glob("*.json"))
        runtime = subprocess.run(
            [*runtime_prefix, *producer_arguments],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert runtime.returncode == 0, f"entry: {entry}\nstdout:\n{runtime.stdout}\nstderr:\n{runtime.stderr}"
        assert json.loads(runtime.stdout) == {
            "entry": entry,
            "schema": "prospect.wm001.bootstrap-fixture.v1",
        }
        assert runtime.stderr == ""
        assert set(completion_root.glob("*.json")) == markers_before
        assert os.path.samefile(marker, runtime_seal)
        assert runtime_seal.stat().st_nlink == 2

    formal_binding = _formal_binding_from_runtime_seal(seal)
    formal_binding_path, binding_terminal, binding_marker = _formal_binding_attempt(
        repository,
        completion_root,
        binding=formal_binding,
        terminal_state="unfinalized",
        producer_files={
            "checkpoints/development-2572962267.pt": 78_205_824,
            "checkpoints/development-3626676950.pt": 78_204_724,
            "replicates/development-2572962267.json": 160_537_118,
            "replicates/development-3626676950.json": 160_421_650,
            "result.json": 320_977_868,
        },
    )

    def invoke_formal(path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(virtualenv / "bin" / "python"),
                "-I",
                "-S",
                "-B",
                str(launcher),
                "--bootstrap",
                str(producer),
                "--runtime-seal",
                str(path),
                "--",
                "preformal-runtime",
                "bootstrap-inventory-conformance",
                "--device",
                "cpu",
            ],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    unfinalized = invoke_formal(formal_binding_path)
    assert unfinalized.returncode != 0
    assert not binding_marker.exists()

    stray_terminal = repository / "stray-binding-terminal.json"
    os.link(binding_terminal, stray_terminal)
    shutil.copyfile(binding_terminal, binding_marker)
    stray_marker = completion_root / "stray-binding-marker.json"
    os.link(binding_marker, stray_marker)
    copied_marker = invoke_formal(formal_binding_path)
    assert copied_marker.returncode != 0
    assert not os.path.samefile(binding_terminal, binding_marker)
    stray_marker.unlink()
    binding_marker.unlink()
    stray_terminal.unlink()
    os.link(binding_terminal, binding_marker)

    direct_copy = lifecycle / "results" / "formal-binding-copy.json"
    shutil.copyfile(formal_binding_path, direct_copy)
    direct = invoke_formal(direct_copy)
    assert direct.returncode != 0
    direct_marker = _completion_marker(completion_root, direct_copy)
    os.link(direct_copy, direct_marker)
    self_finalized_copy = invoke_formal(direct_copy)
    assert self_finalized_copy.returncode != 0
    direct_marker.unlink()

    formal_root = lifecycle / "results" / "formal"
    assert not formal_root.exists()
    missing_rehearsal = invoke_formal(formal_binding_path)
    assert missing_rehearsal.returncode != 0

    rehearsal_runtime = subprocess.run(
        [
            str(virtualenv / "bin" / "python"),
            "-I",
            "-S",
            "-B",
            str(launcher),
            "--bootstrap",
            str(producer),
            "--rehearse-accepted-binding",
            str(formal_binding_path),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert rehearsal_runtime.returncode == 0, f"stdout:\n{rehearsal_runtime.stdout}stderr:\n{rehearsal_runtime.stderr}"
    assert rehearsal_runtime.stdout == ""
    assert rehearsal_runtime.stderr == ""
    rehearsal_attempt = lifecycle / "results" / "operator-v1.20" / "rehearsals" / "accepted-binding-rehearsal-v1.20.0"
    rehearsal_terminal = rehearsal_attempt / "rehearsal-terminal.json"
    rehearsal_completion = _completion_marker(
        completion_root,
        rehearsal_terminal,
    )
    assert json.loads(rehearsal_terminal.read_bytes())["status"] == "accepted"
    assert os.path.samefile(rehearsal_terminal, rehearsal_completion)
    assert not formal_root.exists()

    markers_before = set(completion_root.glob("*.json"))
    formal_runtime = invoke_formal(formal_binding_path)
    assert formal_runtime.returncode == 0, f"stdout:\n{formal_runtime.stdout}\nstderr:\n{formal_runtime.stderr}"
    assert formal_runtime.stdout.encode("utf-8") == _rehearsal_output(json.loads(formal_binding_path.read_bytes()))
    assert formal_runtime.stderr == ""
    assert set(completion_root.glob("*.json")) == markers_before
    assert not formal_root.exists()
    assert formal_binding_path.stat().st_nlink == 1
    assert binding_terminal.stat().st_nlink == 2
    assert os.path.samefile(binding_terminal, binding_marker)

    publisher_entries = [
        (["--binding-entry"], "binding", 0),
        (["--audit-entry"], "audit", 1),
        (["--closure-entry"], "closure", 2),
        (["--adjudication-entry"], "adjudication", 0),
        (["default-run"], "default-run", 1),
    ]
    for producer_arguments, entry, logical_exit_code in publisher_entries:
        markers_before = set(completion_root.glob("*.json"))
        runtime = subprocess.run(
            [*runtime_prefix, *producer_arguments],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert runtime.returncode == logical_exit_code, (
            f"entry: {entry}\nstdout:\n{runtime.stdout}\nstderr:\n{runtime.stderr}"
        )
        assert runtime.stdout == ""
        assert runtime.stderr == ""

        terminal = lifecycle / "results" / "fixture-dispatch" / entry / "terminal-manifest.json"
        assert json.loads(terminal.read_bytes()) == {
            "entry": entry,
            "logical_exit_code": logical_exit_code,
            "schema": "prospect.wm001.bootstrap-fixture-terminal.v1",
        }
        terminal_marker = _completion_marker(completion_root, terminal)
        assert terminal_marker.read_bytes() == terminal.read_bytes()
        assert os.path.samefile(terminal_marker, terminal)
        assert terminal.stat().st_nlink == 2
        assert set(completion_root.glob("*.json")) == {
            *markers_before,
            terminal_marker,
        }
