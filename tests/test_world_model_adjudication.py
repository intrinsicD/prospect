from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from bench.world_model_lifecycle import adjudication, producer_bootstrap
from bench.world_model_lifecycle import operator as operator_module
from bench.world_model_lifecycle import rehearsal as rehearsal_module
from bench.world_model_lifecycle.adjudication import (
    ADJUDICATION_CLAIM_NAME,
    ADJUDICATION_MANIFEST_NAME,
    AUDIT_FAILURE_NAME,
    COPIED_SEMANTIC_REVIEW_NAME,
    INPUT_FAILURE_NAME,
    AdjudicationError,
    create_adjudication_package,
    inspect_adjudication_evidence,
    main,
    recover_adjudication_package,
    verify_adjudication_package,
    verify_semantic_review_for_adjudication,
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


def _fixture_rehearsal_identity(
    binding_path: Path,
) -> dict[str, dict[str, object]]:
    binding_sha256 = _digest(binding_path.read_bytes())
    root = binding_path.parent / "accepted-binding-rehearsal" / binding_sha256

    def row(name: str, payload: bytes) -> dict[str, object]:
        return {
            "path": str(root / name),
            "bytes": len(payload),
            "sha256": _digest(payload),
        }

    claim = f"claim:{binding_sha256}".encode("ascii")
    terminal = f"terminal:{binding_sha256}".encode("ascii")
    return {
        "claim": row("rehearsal-claim.json", claim),
        "claim_marker": row(f"accepted-binding-{binding_sha256}.json", claim),
        "terminal": row("rehearsal-terminal.json", terminal),
        "outer_completion": row("outer-completion.json", terminal),
    }


def _write(path: Path, value: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    path.write_bytes(payload)
    return payload


def _rows(root: Path, *, exclude: set[str]) -> list[dict[str, object]]:
    return [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _digest(path.read_bytes()),
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in exclude
    ]


def _audit_report(
    upstream: adjudication._Upstream,
    *,
    passed: bool,
) -> dict[str, object]:
    findings = [] if passed else [{"code": "fixture-failure"}]
    return {
        "schema": "prospect.world-model-lifecycle.artifact-audit.v2",
        "artifact_root": str(upstream.root),
        "result_file": "result.json",
        "result_sha256": _digest(upstream.result_payload),
        "lane": "formal",
        "integrity_passed": passed,
        "engineering_complete": True,
        "complete_for_claim": True,
        "passed": passed,
        "check_counts": {
            "passed": 8,
            "failed": len(findings),
            "coverage_gaps": 0,
        },
        "findings": findings,
        "coverage_gaps": [],
        "custody": {
            "producer_manifest_checked": True,
            "producer_manifest_status": "completed",
            "producer_manifest_sha256": _digest(upstream.producer_manifest_payload),
        },
        "audit_implementation": {
            "auditor_source_sha256": _digest(upstream.source_payloads["artifact_audit.py"]),
            "bound_auditor_source_sha256": _digest(upstream.source_payloads["artifact_audit.py"]),
            "formal_test_report_sha256": "1" * 64,
            "coverage_conformance_report_sha256": "2" * 64,
            "auditor_source_matches_binding": True,
            "coverage_conformance_verified": True,
            "audit_execution_conformance_verified": True,
        },
    }


@dataclass
class Harness:
    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    upstream: adjudication._Upstream
    audit_attempt: Path
    audit_claim_marker: Path
    package: Path
    adjudication_marker: Path
    outer_root: Path
    replay_calls: list[Path] = field(default_factory=list)
    registrations: list[tuple[Path, int]] = field(default_factory=list)
    report_payload: bytes | None = None
    finalize_registration: bool = True

    def make_attempt(
        self,
        *,
        status: str = "accepted",
        outer_finalized: bool = True,
        partial_failure: bool = False,
    ) -> None:
        self.audit_attempt.mkdir(parents=True)
        claim = {
            "schema": "prospect.wm001.formal-audit-claim.v1",
            "experiment_id": "WM-001",
            "protocol_version": "1.17.0",
            "claim_status": "consumed",
            "attempt_path": str(self.audit_attempt),
            "marker_path": str(self.audit_claim_marker),
            "producer_root": str(self.upstream.root),
            "producer_manifest_sha256": _digest(self.upstream.producer_manifest_payload),
            "raw_result_sha256": _digest(self.upstream.result_payload),
            "formal_binding_sha256": _digest(self.upstream.binding_payload),
            "launch_bootstrap_sha256": _digest(self.upstream.source_payloads["launch_bootstrap.py"]),
        }
        _write(self.audit_attempt / "formal-audit-claim.json", claim)
        self.audit_claim_marker.parent.mkdir(parents=True, exist_ok=True)
        os.link(
            self.audit_attempt / "formal-audit-claim.json",
            self.audit_claim_marker,
        )

        include_report = status in {"accepted", "rejected"}
        executions: list[str] = []
        execution_failures: list[str] = []
        audit_file: str | None = None
        if include_report:
            passed = status != "rejected"
            report = _audit_report(self.upstream, passed=passed)
            self.report_payload = _write(
                self.audit_attempt / "independent-audit.json",
                report,
            )
            runtime = self.upstream.outcome_runtime_payload
            invocation = self.upstream.outcome_invocation_payload
            stdout_name = "audit-execution-01.stdout.json"
            stderr_name = "audit-execution-01.stderr.log"
            runtime_name = "audit-execution-01.runtime.json"
            invocation_name = "audit-execution-01.invocation.json"
            (self.audit_attempt / stdout_name).write_bytes(self.report_payload)
            (self.audit_attempt / stderr_name).write_bytes(b"")
            (self.audit_attempt / runtime_name).write_bytes(runtime)
            (self.audit_attempt / invocation_name).write_bytes(invocation)
            receipt = {
                "schema": "prospect.wm001.captured-audit-execution.v1",
                "returncode": 0 if passed else 1,
                "passed": passed,
                "source_mode": "descriptor",
                "command": ["/python", "-I", "-S", "-B", "/proc/self/fd/9"],
                "stdout_file": stdout_name,
                "stderr_file": stderr_name,
                "runtime_manifest_file": runtime_name,
                "invocation_manifest_file": invocation_name,
                "stdout_bytes": len(self.report_payload),
                "stdout_sha256": _digest(self.report_payload),
                "stderr_bytes": 0,
                "stderr_sha256": _digest(b""),
                "runtime_manifest_bytes": len(runtime),
                "runtime_manifest_sha256": _digest(runtime),
                "invocation_manifest_bytes": len(invocation),
                "invocation_manifest_sha256": _digest(invocation),
                "bootstrap_sha256": _digest(self.upstream.bootstrap_payload),
                "auditor_source_sha256": _digest(self.upstream.source_payloads["artifact_audit.py"]),
                "support_files": [],
            }
            receipt_name = "audit-execution-01.execution.json"
            _write(self.audit_attempt / receipt_name, receipt)
            executions.append(receipt_name)
            audit_file = "independent-audit.json"
        if status == "failure" and partial_failure:
            partial_payloads = {
                "stdout": b"partial audit stdout",
                "stderr": b"partial audit stderr",
                "runtime_manifest": self.upstream.outcome_runtime_payload,
                "invocation_manifest": self.upstream.outcome_invocation_payload,
            }
            partial_names = {
                "stdout": "audit-execution-01.partial.stdout",
                "stderr": "audit-execution-01.partial.stderr",
                "runtime_manifest": "audit-execution-01.partial.runtime.json",
                "invocation_manifest": "audit-execution-01.partial.invocation.json",
            }
            for prefix, payload in partial_payloads.items():
                (self.audit_attempt / partial_names[prefix]).write_bytes(payload)
            failure_receipt = {
                "schema": "prospect.wm001.captured-audit-execution-failure.v1",
                "phase": "timeout",
                "returncode": None,
                "source_mode": "descriptor",
                "command": ["/python", "-I", "-S", "-B", "/proc/self/fd/9"],
                **{f"{prefix}_file": partial_names[prefix] for prefix in partial_payloads},
                **{f"{prefix}_bytes": len(payload) for prefix, payload in partial_payloads.items()},
                **{f"{prefix}_sha256": _digest(payload) for prefix, payload in partial_payloads.items()},
                "bootstrap_sha256": _digest(self.upstream.bootstrap_payload),
                "auditor_source_sha256": _digest(self.upstream.source_payloads["artifact_audit.py"]),
                "support_files": [],
            }
            failure_receipt_name = "audit-execution-01.failure.json"
            _write(
                self.audit_attempt / failure_receipt_name,
                failure_receipt,
            )
            execution_failures.append(failure_receipt_name)

        failure_file: str | None = None
        error: dict[str, object] | None = None
        if status == "failure":
            failure_file = "execution-failure.json"
            failure = {
                "schema": "prospect.wm001.operator-execution-failure.v2",
                "experiment_id": "WM-001",
                "protocol_version": "1.17.0",
                "kind": "audit",
                "lane": "formal",
                "phase": "audit_execution",
                "error_type": "AuditRunnerError",
                "failure_code": "audit_runner_error",
                "error_message_bytes": 0,
                "error_message_sha256": hashlib.sha256(b"").hexdigest(),
                "passed": False,
            }
            _write(self.audit_attempt / failure_file, failure)
            error = {
                "failure_code": "audit_runner_error",
                "error_type": "AuditRunnerError",
            }
        primary: dict[str, object] = {
            "producer_root": str(self.upstream.root),
            "audit_file": audit_file,
            "executions": executions,
            "execution_failures": execution_failures,
            "reproduction_file": None,
            "reproduction_runtime_file": None,
            "claim_file": "formal-audit-claim.json",
        }
        if failure_file is not None:
            primary["execution_failure_file"] = failure_file
        files = _rows(
            self.audit_attempt,
            exclude={"operator-attempt.json"},
        )
        manifest = {
            "schema": "prospect.wm001.operator-attempt.v1",
            "experiment_id": "WM-001",
            "protocol_version": "1.17.0",
            "assurance": adjudication.assurance_record(),
            "kind": "audit",
            "lane": "formal",
            "status": status,
            "inputs": [],
            "primary": primary,
            "error": error,
            "files": files,
            "file_count": len(files),
            "manifest_excludes": ["operator-attempt.json"],
        }
        _write(self.audit_attempt / "operator-attempt.json", manifest)
        if outer_finalized:
            marker = operator_module.outer_completion_marker(self.audit_attempt / "operator-attempt.json")
            marker.parent.mkdir(parents=True, exist_ok=True)
            os.link(self.audit_attempt / "operator-attempt.json", marker)

    def review(self, *, disposition: str) -> Path:
        evidence = inspect_adjudication_evidence(self.audit_attempt)
        fields = {
            key: evidence[key]
            for key in (
                "schema",
                "experiment_id",
                "protocol_version",
                "assurance",
                "evidence_kind",
                "artifact_root",
                "result_sha256",
                "audit_attempt_path",
                "audit_attempt_manifest_sha256",
                "formal_audit_claim_sha256",
                "independent_audit_sha256",
                "execution_failure_sha256",
            )
        }
        failure = evidence["evidence_kind"] == "execution_failure"
        review = {
            **fields,
            "reviewer": "independent-fixture-reviewer",
            "reviewed_gates": [] if failure else [f"K{i}" for i in range(8)],
            "verdict": disposition,
            "fatal_findings": ([{"code": "terminal-rejection"}] if disposition == "rejected" else []),
            "conclusion": "fixture semantic review",
        }
        return_path = self.tmp_path / "artifacts" / "wm001-reviews" / "formal-v1.17.0.json"
        _write(return_path, review)
        return return_path

    def install_replay(
        self,
        *,
        payload: bytes | None = None,
        failure: adjudication._ExecutionFailure | None = None,
    ) -> None:
        def run(
            upstream: adjudication._Upstream,
        ) -> tuple[
            adjudication._Replay | None,
            adjudication._ExecutionFailure | None,
        ]:
            self.replay_calls.append(upstream.root)
            if failure is not None:
                return None, failure
            stdout = payload if payload is not None else self.report_payload
            assert stdout is not None
            report = json.loads(stdout)
            return (
                adjudication._Replay(
                    returncode=0 if report["passed"] else 1,
                    command=(
                        "/python",
                        "-I",
                        "-S",
                        "-B",
                        "/proc/self/fd/9",
                    ),
                    stdout=stdout,
                    stderr=b"",
                    report=report,
                    runtime_manifest=upstream.outcome_runtime_payload,
                    invocation_manifest=upstream.outcome_invocation_payload,
                    bootstrap_sha256=_digest(upstream.bootstrap_payload),
                    auditor_source_sha256=_digest(upstream.source_payloads["artifact_audit.py"]),
                    support_files=[],
                    source_mode="descriptor",
                ),
                None,
            )

        self.monkeypatch.setattr(adjudication, "_run_bound_replay", run)

    def create(self, review: Path, disposition: str) -> dict[str, object]:
        return create_adjudication_package(
            audit_attempt=self.audit_attempt,
            disposition=disposition,  # type: ignore[arg-type]
            semantic_review=review,
        )


@pytest.fixture
def harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Harness:
    producer = tmp_path / "formal" / ("b" * 64) / "attempt-fixture"
    producer.mkdir(parents=True)
    source_payloads = {
        name: f"# fixture {name}\n".encode()
        for name in (
            "audit_runner.py",
            "artifact_audit.py",
            "adjudication.py",
            "binding.py",
            "launch_bootstrap.py",
            "operator.py",
            "preformal.py",
            "producer_bootstrap.py",
            "run.py",
            "verify.py",
        )
    }
    producer_manifest = _canonical({"schema": "fixture.producer", "status": "completed", "lane": "formal"})
    result = {
        "schema": "prospect.world-model-lifecycle.raw-result.v9",
        "gate_results": [{"gate": f"K{index}", "passed": True} for index in range(8)],
    }
    result_payload = _canonical(result)
    runtime_payload = _canonical(
        {
            "schema": "fixture.runtime",
            "support_files": [],
            "limits": {
                "timeout_seconds": 600,
                "stdout_bytes": 64 << 20,
                "stderr_bytes": 16 << 20,
            },
        }
    )
    invocation_payload = _canonical({"schema": "fixture.invocation"})
    bootstrap_payload = b"# fixture bootstrap\n"
    binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "dependencies": {
            "python_executable": "/python",
            "python_executable_sha256": "a" * 64,
        },
        "coverage_arithmetic": {
            "formal_test_report_sha256": "1" * 64,
            "conformance_report_sha256": "2" * 64,
        },
        "audit_execution": {
            "runner_source_sha256": _digest(source_payloads["audit_runner.py"]),
            "auditor_source_sha256": _digest(source_payloads["artifact_audit.py"]),
            "adjudicator_source_sha256": _digest(source_payloads["adjudication.py"]),
        },
    }
    binding_payload = _canonical(binding)
    launch = {
        "schema": "prospect.wm001.formal-launch.v3",
        "accepted_binding_rehearsal": {},
    }
    launch_payload = _canonical(launch)
    upstream = adjudication._Upstream(
        root=producer,
        producer_manifest=cast_dict(json.loads(producer_manifest)),
        producer_manifest_payload=producer_manifest,
        result=cast_dict(result),
        result_payload=result_payload,
        binding=cast_dict(binding),
        binding_payload=binding_payload,
        launch=cast_dict(launch),
        launch_payload=launch_payload,
        audit_execution=cast_dict(binding["audit_execution"]),
        outcome_runtime_payload=runtime_payload,
        outcome_invocation_payload=invocation_payload,
        source_payloads=source_payloads,
        bootstrap_payload=bootstrap_payload,
    )
    audit_attempt = tmp_path / "operator" / "audits" / "formal-audit-v1.17.0"
    audit_claim = tmp_path / "formal" / "formal-audit-v1.17.0.json"
    package = tmp_path / "adjudication" / "formal-adjudication-v1.17.0"
    adjudication_marker = tmp_path / "formal" / "formal-adjudication-v1.17.0.json"
    outer_root = tmp_path / "outer-completions" / "v1.17"
    value = Harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        upstream=upstream,
        audit_attempt=audit_attempt,
        audit_claim_marker=audit_claim,
        package=package,
        adjudication_marker=adjudication_marker,
        outer_root=outer_root,
    )
    monkeypatch.setattr(adjudication, "FORMAL_AUDIT_ATTEMPT_PATH", audit_attempt)
    monkeypatch.setattr(
        adjudication,
        "FORMAL_ADJUDICATION_PACKAGE_PATH",
        package,
    )
    monkeypatch.setattr(
        adjudication,
        "FORMAL_ADJUDICATION_CLAIM_MARKER",
        adjudication_marker,
    )
    monkeypatch.setattr(
        adjudication,
        "FORMAL_SEMANTIC_REVIEW_PATH",
        tmp_path / "artifacts" / "wm001-reviews" / "formal-v1.17.0.json",
    )
    monkeypatch.setattr(operator_module, "FORMAL_AUDIT_ATTEMPT_PATH", audit_attempt)
    monkeypatch.setattr(operator_module, "FORMAL_AUDIT_CLAIM_MARKER", audit_claim)
    monkeypatch.setattr(operator_module, "OUTER_COMPLETIONS_ROOT", outer_root)
    monkeypatch.setattr(adjudication, "_require_sealed_entry", lambda: None)
    monkeypatch.setattr(
        adjudication,
        "_load_upstream",
        lambda path, *, require_live_sources: (
            upstream if path == producer else pytest.fail(f"unexpected producer {path}")
        ),
    )

    def read_manifest(path: Path) -> dict[str, object]:
        operator_module.verify_outer_completion(path / "operator-attempt.json")
        return cast_dict(json.loads((path / "operator-attempt.json").read_bytes()))

    def read_unfinalized(path: Path) -> dict[str, object]:
        terminal = path / "operator-attempt.json"
        assert not os.path.lexists(operator_module.outer_completion_marker(terminal))
        return {
            "manifest": cast_dict(json.loads(terminal.read_bytes())),
            "outer_finalized": False,
        }

    monkeypatch.setattr(operator_module, "verify_operator_attempt", read_manifest)
    monkeypatch.setattr(
        operator_module,
        "inspect_unfinalized_operator_attempt",
        read_unfinalized,
    )

    def register(path: Path, *, logical_exit_code: int) -> None:
        value.registrations.append((path, logical_exit_code))
        if value.finalize_registration:
            marker = operator_module.outer_completion_marker(path)
            marker.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, marker)

    monkeypatch.setattr(producer_bootstrap, "register_outer_terminal", register)
    return value


def cast_dict(value: Any) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def test_launch_v3_authenticates_binding_attempt_and_accepted_rehearsal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import artifact_audit
    from bench.world_model_lifecycle import verify as verify_module
    from bench.world_model_lifecycle.artifact import (
        FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
        FORMAL_BINDING_OUTER_COMPLETION_NAME,
        FORMAL_CONFIRMATION_NAME,
        FORMAL_LAUNCH_MARKER_NAME,
    )

    repo = tmp_path / "repo"
    lifecycle = repo / "bench" / "world_model_lifecycle"
    binding_root = lifecycle / "results" / "operator-v1.17" / "bindings"
    binding_attempt = binding_root / "formal-binding-v1.17.0"
    completion_root = lifecycle / "results" / "outer-completions" / "v1.17"
    binding_root.mkdir(parents=True)
    monkeypatch.setattr(operator_module, "REPO", repo)
    monkeypatch.setattr(adjudication, "REPO", repo)
    monkeypatch.setattr(
        operator_module,
        "BINDING_ATTEMPTS_ROOT",
        binding_root,
    )
    monkeypatch.setattr(
        operator_module,
        "FORMAL_BINDING_ATTEMPT_PATH",
        binding_attempt,
    )
    monkeypatch.setattr(
        operator_module,
        "OUTER_COMPLETIONS_ROOT",
        completion_root,
    )
    binding_value = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "fixture": "real-outer-finalized-attempt",
    }
    binding_payload = _canonical(binding_value)
    monkeypatch.setattr(
        verify_module,
        "verify_binding",
        lambda path: cast_dict(json.loads(path.read_bytes())),
    )
    monkeypatch.setattr(
        operator_module,
        "verify_binding_authorization_inputs",
        lambda _path, inputs: list(inputs),
    )
    preflight_receipt = {
        "schema": "prospect.wm001.formal-input-preflight.v1",
        "binding_sha256": _digest(binding_payload),
        "passed": True,
    }
    monkeypatch.setattr(
        artifact_audit,
        "preflight_formal_input_package",
        lambda _path: preflight_receipt,
    )
    monkeypatch.setattr(
        producer_bootstrap,
        "register_outer_terminal",
        lambda _path, *, logical_exit_code: None,
    )
    attempt = operator_module._Attempt(  # noqa: SLF001
        final=binding_attempt,
        kind="binding",
        lane=None,
        inputs=[],
    )
    (attempt.staging / "formal-binding.json").write_bytes(binding_payload)
    _write(attempt.staging / "formal-input-preflight.json", preflight_receipt)
    attempt.finish(
        status="accepted",
        primary={"binding_file": "formal-binding.json"},
        error=None,
        final_check=lambda: None,
    )
    binding_terminal = binding_attempt / "operator-attempt.json"
    binding_completion = operator_module.outer_completion_marker(binding_terminal)
    binding_completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(
        binding_terminal,
        binding_completion,
        follow_symlinks=False,
    )
    operator_module.verify_operator_attempt(binding_attempt)
    terminal_payload = binding_terminal.read_bytes()
    monkeypatch.setattr(
        rehearsal_module,
        "accepted_binding_rehearsal_identity",
        _fixture_rehearsal_identity,
    )

    binding_sha256 = _digest(binding_payload)
    producer = (
        lifecycle
        / "results"
        / "formal"
        / binding_sha256
        / FORMAL_CONFIRMATION_NAME
    )
    producer.mkdir(parents=True)
    copied_attempt = producer / FORMAL_BINDING_ATTEMPT_MANIFEST_NAME
    copied_completion = producer / FORMAL_BINDING_OUTER_COMPLETION_NAME
    copied_attempt.write_bytes(terminal_payload)
    copied_completion.write_bytes(terminal_payload)
    record: dict[str, object] = {
        "schema": "prospect.wm001.formal-launch.v3",
        "experiment_id": "WM-001",
        "protocol_version": "1.17.0",
        "formal_binding_sha256": binding_sha256,
        "formal_binding_attempt_path": str(binding_attempt),
        "formal_binding_attempt_manifest_file": (FORMAL_BINDING_ATTEMPT_MANIFEST_NAME),
        "formal_binding_attempt_manifest_sha256": _digest(terminal_payload),
        "formal_binding_outer_completion_file": (FORMAL_BINDING_OUTER_COMPLETION_NAME),
        "formal_binding_outer_completion_marker": str(binding_completion),
        "formal_binding_outer_completion_sha256": _digest(terminal_payload),
        "accepted_binding_rehearsal": _fixture_rehearsal_identity(
            binding_attempt / "formal-binding.json"
        ),
        "attempt_directory": producer.name,
        "global_marker_file": FORMAL_LAUNCH_MARKER_NAME,
        "claimed_at_utc": "2026-07-19T00:00:00Z",
        "git_commit": "1" * 40,
        "git_tree": "2" * 40,
    }
    record["record_sha256"] = _digest(_canonical(record)[:-1])
    launch_payload = _canonical(record)
    launch_path = producer / "formal-launch.json"
    launch_path.write_bytes(launch_payload)
    launch_marker = producer.parent.parent / FORMAL_LAUNCH_MARKER_NAME
    os.link(launch_path, launch_marker, follow_symlinks=False)
    producer_manifest = {
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _digest(path.read_bytes()),
            }
            for path in sorted((copied_attempt, copied_completion, launch_path))
        ]
    }
    result = {
        "execution": {
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "formal_launch_file": "formal-launch.json",
            "formal_launch_sha256": _digest(launch_payload),
        }
    }

    launch, payload = adjudication._verify_launch(  # noqa: SLF001
        root=producer,
        producer_manifest=producer_manifest,
        result=result,
        binding_payload=binding_payload,
    )

    assert launch == record
    assert payload == launch_payload

    def reject_rehearsal(_path: Path) -> dict[str, dict[str, object]]:
        raise rehearsal_module.RehearsalEvidenceError("failed rehearsal")

    monkeypatch.setattr(
        rehearsal_module,
        "accepted_binding_rehearsal_identity",
        reject_rehearsal,
    )
    with pytest.raises(
        AdjudicationError,
        match="accepted outer-finalized binding rehearsal",
    ):
        adjudication._verify_launch(  # noqa: SLF001
            root=producer,
            producer_manifest=producer_manifest,
            result=result,
            binding_payload=binding_payload,
        )
    monkeypatch.setattr(
        rehearsal_module,
        "accepted_binding_rehearsal_identity",
        _fixture_rehearsal_identity,
    )
    copied_completion.write_bytes(b"forged completion\n")
    with pytest.raises(
        AdjudicationError,
        match="unique v1.17 launch record",
    ):
        adjudication._verify_launch(  # noqa: SLF001
            root=producer,
            producer_manifest=producer_manifest,
            result=result,
            binding_payload=binding_payload,
        )
    copied_completion.write_bytes(terminal_payload)

    launch_row = next(
        row
        for row in producer_manifest["files"]
        if row["path"] == "formal-launch.json"
    )
    other_binding = tmp_path / "other-binding.json"
    other_binding.write_bytes(b"other canonical binding bytes\n")
    for mutation in (
        "missing-role",
        "extra-role",
        "path",
        "boolean-bytes",
        "sha256",
        "cross-binding",
    ):
        mutated_record = json.loads(json.dumps(record))
        identity = mutated_record["accepted_binding_rehearsal"]
        assert isinstance(identity, dict)
        if mutation == "missing-role":
            del identity["claim_marker"]
        elif mutation == "extra-role":
            identity["unbound"] = identity["claim"]
        elif mutation == "cross-binding":
            mutated_record["accepted_binding_rehearsal"] = (
                _fixture_rehearsal_identity(other_binding)
            )
        else:
            terminal_identity = identity["terminal"]
            assert isinstance(terminal_identity, dict)
            if mutation == "path":
                terminal_identity["path"] = f"{terminal_identity['path']}.alias"
            elif mutation == "boolean-bytes":
                terminal_identity["bytes"] = True
            else:
                terminal_identity["sha256"] = "f" * 64
        mutated_body = dict(mutated_record)
        mutated_body.pop("record_sha256")
        mutated_record["record_sha256"] = _digest(
            _canonical(mutated_body)[:-1]
        )
        mutated_payload = _canonical(mutated_record)
        launch_path.write_bytes(mutated_payload)
        result["execution"]["formal_launch_sha256"] = _digest(mutated_payload)
        launch_row["bytes"] = len(mutated_payload)
        launch_row["sha256"] = _digest(mutated_payload)
        with pytest.raises(
            AdjudicationError,
            match="unique v1.17 launch record",
        ):
            adjudication._verify_launch(  # noqa: SLF001
                root=producer,
                producer_manifest=producer_manifest,
                result=result,
                binding_payload=binding_payload,
            )
    launch_path.write_bytes(launch_payload)
    result["execution"]["formal_launch_sha256"] = _digest(launch_payload)
    launch_row["bytes"] = len(launch_payload)
    launch_row["sha256"] = _digest(launch_payload)

    sibling = producer.with_name("wrong-child")
    producer.rename(sibling)
    sibling_launch = sibling / "formal-launch.json"
    mutated_record = dict(record)
    mutated_record["attempt_directory"] = sibling.name
    mutated_body = dict(mutated_record)
    mutated_body.pop("record_sha256")
    mutated_record["record_sha256"] = _digest(
        _canonical(mutated_body)[:-1],
    )
    mutated_launch_payload = _canonical(mutated_record)
    sibling_launch.write_bytes(mutated_launch_payload)
    result["execution"]["formal_launch_sha256"] = _digest(
        mutated_launch_payload,
    )
    launch_row["bytes"] = len(mutated_launch_payload)
    launch_row["sha256"] = _digest(mutated_launch_payload)

    with pytest.raises(
        AdjudicationError,
        match="unique v1.17 launch record",
    ):
        adjudication._verify_launch(  # noqa: SLF001
            root=sibling,
            producer_manifest=producer_manifest,
            result=result,
            binding_payload=binding_payload,
        )


def test_accepted_report_runs_one_replay_and_outer_finalizes(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()

    manifest = harness.create(review, "accepted")

    assert manifest["disposition"] == "accepted"
    assert manifest["outcome_kind"] == "audit_report"
    assert harness.replay_calls == [harness.upstream.root]
    assert harness.registrations == [(harness.package / ADJUDICATION_MANIFEST_NAME, 0)]
    assert verify_adjudication_package(harness.package) == manifest
    claim = harness.package / ADJUDICATION_CLAIM_NAME
    assert os.path.samefile(claim, harness.adjudication_marker)
    assert claim.stat().st_nlink == 2


def test_rejected_report_is_terminal_after_one_replay(
    harness: Harness,
) -> None:
    harness.make_attempt(status="rejected")
    review = harness.review(disposition="rejected")
    harness.install_replay()

    manifest = harness.create(review, "rejected")

    assert manifest["disposition"] == "rejected"
    assert manifest["outcome_kind"] == "audit_report"
    assert len(harness.replay_calls) == 1
    assert harness.registrations[0][1] == 1
    verify_adjudication_package(harness.package)


@pytest.mark.parametrize("partial", [False, True])
def test_official_operator_failure_is_rejected_with_zero_replay(
    harness: Harness,
    partial: bool,
) -> None:
    harness.make_attempt(status="failure", partial_failure=partial)
    review = harness.review(disposition="rejected")
    harness.install_replay()

    manifest = harness.create(review, "rejected")

    assert manifest["outcome_kind"] == "formal_audit_execution_failure"
    assert manifest["disposition"] == "rejected"
    assert harness.replay_calls == []
    assert (harness.package / INPUT_FAILURE_NAME).is_file()
    assert (
        any(row["source_file"] == "audit-execution-01.failure.json" for row in manifest["audit_attempt_files"])
        is partial
    )
    verify_adjudication_package(harness.package)


def test_outer_completion_absent_consumes_claim_with_zero_replay(
    harness: Harness,
) -> None:
    harness.make_attempt(status="accepted", outer_finalized=False)
    review = harness.review(disposition="rejected")
    harness.install_replay()

    manifest = harness.create(review, "rejected")

    failure = json.loads((harness.package / INPUT_FAILURE_NAME).read_bytes())
    assert failure["failure_code"] == "outer_completion_absent"
    assert manifest["outcome_kind"] == "formal_audit_execution_failure"
    assert harness.replay_calls == []
    assert harness.adjudication_marker.exists()
    verify_adjudication_package(harness.package)


def test_execution_failure_rejects_unbound_partial_replay_file(
    harness: Harness,
) -> None:
    harness.make_attempt(status="failure")
    review = harness.review(disposition="rejected")
    harness.install_replay()
    harness.create(review, "rejected")

    os.chmod(harness.package, 0o700)
    extra = harness.package / adjudication.PARTIAL_REPLAY_STDOUT_NAME
    extra.write_bytes(b"unbound partial replay")
    terminal = harness.package / ADJUDICATION_MANIFEST_NAME
    manifest = json.loads(terminal.read_bytes())
    manifest["files"] = _rows(
        harness.package,
        exclude={ADJUDICATION_MANIFEST_NAME},
    )
    manifest["file_count"] = len(manifest["files"])
    os.chmod(terminal, 0o600)
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        AdjudicationError,
        match="execution-failure package is inconsistent",
    ):
        verify_adjudication_package(harness.package)


def test_replay_failure_preserves_pre_replay_review(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay(
        failure=adjudication._ExecutionFailure(
            code="audit-timeout",
            error_type="AuditRunnerError",
        )
    )

    manifest = harness.create(review, "accepted")

    assert manifest["disposition"] == "rejected"
    assert manifest["outcome_kind"] == "adjudication_replay_failure"
    assert manifest["semantic_review_role"] == "pre_replay_supplied_audit_review"
    assert (harness.package / AUDIT_FAILURE_NAME).is_file()
    assert (harness.package / COPIED_SEMANTIC_REVIEW_NAME).read_bytes() == review.read_bytes()
    verify_adjudication_package(harness.package)


def test_actual_runner_failure_preserves_all_bounded_partial_evidence(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle.audit_runner import AuditExecutionFailure

    harness.make_attempt()
    review = harness.review(disposition="accepted")
    partial = {
        "stdout": b"bounded partial stdout",
        "stderr": b"bounded partial stderr",
        "runtime": harness.upstream.outcome_runtime_payload,
        "invocation": harness.upstream.outcome_invocation_payload,
    }
    calls: list[Path] = []

    def fail(root: Path) -> object:
        calls.append(root)
        raise AuditExecutionFailure(
            "fixture timeout",
            phase="timeout",
            command=("/python", "-I", "-S", "-B", "/proc/self/fd/9"),
            returncode=None,
            stdout=partial["stdout"],
            stderr=partial["stderr"],
            runtime_manifest=partial["runtime"],
            invocation_manifest=partial["invocation"],
            bootstrap_sha256=_digest(harness.upstream.bootstrap_payload),
            auditor_source_sha256=_digest(harness.upstream.source_payloads["artifact_audit.py"]),
            support_files=(),
            source_mode="descriptor",
        )

    binding_stub = ModuleType("bench.world_model_lifecycle.binding")
    binding_stub.run_bound_outcome_audit = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "bench.world_model_lifecycle.binding",
        binding_stub,
    )

    manifest = harness.create(review, "accepted")

    assert calls == [harness.upstream.root]
    assert manifest["outcome_kind"] == "adjudication_replay_failure"
    failure = json.loads((harness.package / AUDIT_FAILURE_NAME).read_bytes())
    assert failure["partial_execution"]["phase"] == "timeout"
    assert failure["execution_completed"] is False
    assert (harness.package / adjudication.PARTIAL_REPLAY_STDOUT_NAME).read_bytes() == partial["stdout"]
    assert (harness.package / adjudication.PARTIAL_REPLAY_STDERR_NAME).read_bytes() == partial["stderr"]
    assert (harness.package / adjudication.PARTIAL_REPLAY_RUNTIME_NAME).read_bytes() == partial["runtime"]
    assert (harness.package / adjudication.PARTIAL_REPLAY_INVOCATION_NAME).read_bytes() == partial["invocation"]
    verify_adjudication_package(harness.package)


def test_valid_but_byte_distinct_replay_is_terminally_rejected(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    assert harness.report_payload is not None
    replay_report = json.loads(harness.report_payload)
    replay_report["fixture_replay_nonce"] = 1
    harness.install_replay(payload=_canonical(replay_report))

    manifest = harness.create(review, "accepted")

    assert manifest["outcome_kind"] == "audit_replay_mismatch"
    assert manifest["disposition"] == "rejected"
    assert manifest["semantic_review_role"] == "pre_replay_supplied_audit_review"
    assert len(harness.replay_calls) == 1
    verify_adjudication_package(harness.package)


def test_second_attempt_is_blocked_before_runner(harness: Harness) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.create(review, "accepted")
    calls = list(harness.replay_calls)

    with pytest.raises(AdjudicationError, match="already consumed"):
        harness.create(review, "accepted")

    assert harness.replay_calls == calls


def test_preclaim_staging_mutation_does_not_consume_or_replay(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()

    def mutate_review() -> None:
        os.chmod(review, 0o600)
        review.write_bytes(b"{}\n")

    harness.monkeypatch.setattr(
        adjudication,
        "_before_adjudication_claim",
        mutate_review,
    )

    with pytest.raises(AdjudicationError, match="changed"):
        harness.create(review, "accepted")

    assert not harness.adjudication_marker.exists()
    assert harness.replay_calls == []
    assert not harness.package.exists()
    assert list(harness.package.parent.glob(".formal-adjudication-v1.17.0.staging-*")) == []


def test_crash_after_claim_consumes_version_without_replay(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_after_adjudication_claim",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture crash")),
    )

    manifest = harness.create(review, "accepted")

    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    assert manifest["disposition"] == "rejected"
    assert harness.replay_calls == []
    assert harness.adjudication_marker.exists()
    assert harness.adjudication_marker.stat().st_nlink == 2
    verify_adjudication_package(harness.package)


def test_fault_after_claim_link_preserves_two_link_consumed_marker(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_after_adjudication_claim_link",
        lambda: (_ for _ in ()).throw(RuntimeError("post-link fault")),
    )

    manifest = harness.create(review, "accepted")

    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    assert manifest["disposition"] == "rejected"
    assert harness.replay_calls == []
    assert harness.adjudication_marker.stat().st_nlink == 2
    assert os.path.samefile(
        harness.package / ADJUDICATION_CLAIM_NAME,
        harness.adjudication_marker,
    )
    verify_adjudication_package(harness.package)


def test_canonical_semantic_review_path_rejects_same_basename_sibling(
    harness: Harness,
) -> None:
    harness.make_attempt()
    canonical = harness.review(disposition="accepted")
    sibling = canonical.parent.parent / "other" / canonical.name
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(canonical.read_bytes())
    harness.install_replay()

    with pytest.raises(AdjudicationError, match="sole canonical semantic review"):
        harness.create(sibling, "accepted")

    assert not harness.adjudication_marker.exists()
    assert harness.replay_calls == []


def test_semantic_review_preflight_is_read_only_and_accepts_exact_review(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    expected = json.loads(review.read_bytes())

    verified = verify_semantic_review_for_adjudication(
        harness.audit_attempt,
        review,
        "accepted",
    )

    assert verified == expected
    assert not harness.adjudication_marker.exists()
    assert not harness.package.exists()
    assert harness.replay_calls == []


def test_semantic_review_preflight_rejects_inspector_only_fields(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    malformed = json.loads(review.read_bytes())
    malformed["required_verdict"] = "accepted_or_rejected"
    malformed["execution_failure_record"] = None
    review.write_bytes(_canonical(malformed))

    with pytest.raises(
        AdjudicationError,
        match="semantic review v2 identity",
    ):
        verify_semantic_review_for_adjudication(
            harness.audit_attempt,
            review,
            "accepted",
        )

    assert not harness.adjudication_marker.exists()
    assert not harness.package.exists()
    assert harness.replay_calls == []


@pytest.mark.parametrize(
    ("hook_name", "expected_prior_state"),
    [
        ("_before_adjudication_replay", "started"),
        ("_after_adjudication_replay", "started"),
    ],
)
def test_replay_boundary_fault_recovers_without_second_replay(
    harness: Harness,
    hook_name: str,
    expected_prior_state: str,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        hook_name,
        lambda _path: (_ for _ in ()).throw(RuntimeError("replay boundary fault")),
    )

    manifest = harness.create(review, "accepted")

    expected_calls = 0 if hook_name == "_before_adjudication_replay" else 1
    assert len(harness.replay_calls) == expected_calls
    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    failure = json.loads((harness.package / adjudication.ADJUDICATION_RECOVERY_FAILURE_NAME).read_bytes())
    assert failure["prior_replay_state"] == expected_prior_state
    assert failure["recovery_replay_performed"] is False
    verify_adjudication_package(harness.package)


def test_unexpected_runner_fault_recovers_without_replay_retry(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")

    def fault(upstream: adjudication._Upstream) -> object:
        harness.replay_calls.append(upstream.root)
        raise RuntimeError("unexpected runner fault")

    harness.monkeypatch.setattr(adjudication, "_run_bound_replay", fault)

    manifest = harness.create(review, "accepted")

    assert harness.replay_calls == [harness.upstream.root]
    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    assert manifest["disposition"] == "rejected"
    verify_adjudication_package(harness.package)


def test_partial_replay_start_write_recovers_conservatively_without_runner(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    real_write = adjudication._write_private_file

    def partial_write(path: Path, payload: bytes) -> None:
        if path.name == adjudication.ADJUDICATION_REPLAY_STARTED_NAME:
            path.write_bytes(b"{")
            raise OSError("partial replay-start receipt")
        real_write(path, payload)

    harness.monkeypatch.setattr(
        adjudication,
        "_write_private_file",
        partial_write,
    )

    manifest = harness.create(review, "accepted")

    failure = json.loads((harness.package / adjudication.ADJUDICATION_RECOVERY_FAILURE_NAME).read_bytes())
    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    assert failure["prior_replay_state"] == "untrusted_or_partial"
    assert harness.replay_calls == []
    verify_adjudication_package(harness.package)


@pytest.mark.parametrize(
    "hook_name",
    [
        "_after_package_publish",
        "_before_outer_registration",
    ],
)
def test_renamed_exact_package_is_only_validated_and_finalized_after_fault(
    harness: Harness,
    hook_name: str,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        hook_name,
        lambda _path: (_ for _ in ()).throw(RuntimeError("post-rename fault")),
    )

    manifest = harness.create(review, "accepted")

    assert manifest["outcome_kind"] == "audit_report"
    assert manifest["disposition"] == "accepted"
    assert harness.replay_calls == [harness.upstream.root]
    assert len(harness.registrations) == 1
    verify_adjudication_package(harness.package)


def test_fault_after_outer_registration_preserves_exact_terminal(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_after_outer_registration",
        lambda _path: (_ for _ in ()).throw(RuntimeError("post-registration fault")),
    )

    with pytest.raises(RuntimeError, match="post-registration fault"):
        harness.create(review, "accepted")

    assert harness.replay_calls == [harness.upstream.root]
    verify_adjudication_package(harness.package)
    with pytest.raises(AdjudicationError, match="already outer-finalized"):
        recover_adjudication_package()
    assert harness.replay_calls == [harness.upstream.root]


def test_explicit_recovery_only_finalizes_unfinalized_exact_package(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.finalize_registration = False

    manifest = harness.create(review, "accepted")
    before = {path.name: path.read_bytes() for path in harness.package.iterdir()}
    harness.finalize_registration = True
    recovered = recover_adjudication_package()

    assert recovered == manifest
    assert {path.name: path.read_bytes() for path in harness.package.iterdir()} == before
    assert harness.replay_calls == [harness.upstream.root]
    verify_adjudication_package(harness.package)


def test_explicit_recovery_refuses_to_rewrite_invalid_renamed_package(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.finalize_registration = False
    harness.create(review, "accepted")
    copied_review = harness.package / COPIED_SEMANTIC_REVIEW_NAME
    os.chmod(copied_review, 0o600)
    copied_review.write_bytes(b"{}\n")
    corrupted = {path.name: path.read_bytes() for path in harness.package.iterdir()}
    registrations = list(harness.registrations)

    with pytest.raises(AdjudicationError):
        recover_adjudication_package()

    assert {path.name: path.read_bytes() for path in harness.package.iterdir()} == corrupted
    assert harness.registrations == registrations
    assert harness.replay_calls == [harness.upstream.root]


def test_marker_only_explicit_recovery_is_zero_replay_and_single_use(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_after_adjudication_claim",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        harness.create(review, "accepted")

    holders = list(harness.package.parent.glob(".formal-adjudication-v1.17.0.staging-*/" + ADJUDICATION_CLAIM_NAME))
    assert len(holders) == 1
    shutil.rmtree(holders[0].parent)
    assert harness.adjudication_marker.stat().st_nlink == 1

    manifest = recover_adjudication_package()

    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    assert manifest["disposition"] == "rejected"
    assert harness.replay_calls == []
    verify_adjudication_package(harness.package)
    with pytest.raises(AdjudicationError, match="already outer-finalized"):
        recover_adjudication_package()
    assert harness.replay_calls == []


def test_recovery_cli_is_zero_replay_and_outer_finalizes(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_after_adjudication_claim",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        harness.create(review, "accepted")
    harness.monkeypatch.setattr(adjudication, "_after_adjudication_claim", lambda: None)

    assert main(["--recover"]) == 1

    assert harness.replay_calls == []
    assert harness.registrations == [(harness.package / ADJUDICATION_MANIFEST_NAME, 1)]
    verify_adjudication_package(harness.package)


def test_older_marker_does_not_consume_v117(harness: Harness) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    older = harness.adjudication_marker.with_name("formal-adjudication-v1.8.0.json")
    older.parent.mkdir(parents=True, exist_ok=True)
    older.write_text("{}\n", encoding="utf-8")

    manifest = harness.create(review, "accepted")

    assert manifest["disposition"] == "accepted"
    assert older.exists()


def test_claim_mutation_is_detected(harness: Harness) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.create(review, "accepted")
    claim = harness.package / ADJUDICATION_CLAIM_NAME
    os.chmod(claim, 0o600)
    claim.write_bytes(b"{}\n")

    with pytest.raises(AdjudicationError):
        verify_adjudication_package(harness.package)


def test_manifest_optional_reference_tamper_is_detected(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.create(review, "accepted")
    terminal = harness.package / ADJUDICATION_MANIFEST_NAME
    value = json.loads(terminal.read_bytes())
    value["audit_execution_file"] = None
    os.chmod(terminal, 0o600)
    terminal.write_bytes(_canonical(value))

    with pytest.raises(AdjudicationError, match="reference"):
        verify_adjudication_package(harness.package)


def test_recovery_failure_semantic_tamper_is_detected(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_after_adjudication_claim",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture crash")),
    )
    harness.create(review, "accepted")
    failure_path = harness.package / adjudication.ADJUDICATION_RECOVERY_FAILURE_NAME
    failure = json.loads(failure_path.read_bytes())
    failure["recovery_replay_performed"] = True
    os.chmod(failure_path, 0o600)
    failure_path.write_bytes(_canonical(failure))
    terminal = harness.package / ADJUDICATION_MANIFEST_NAME
    manifest = json.loads(terminal.read_bytes())
    manifest["files"] = _rows(
        harness.package,
        exclude={ADJUDICATION_MANIFEST_NAME},
    )
    manifest["file_count"] = len(manifest["files"])
    manifest["recovery_failure_sha256"] = _digest(failure_path.read_bytes())
    os.chmod(terminal, 0o600)
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(AdjudicationError, match="recovery failure is inconsistent"):
        verify_adjudication_package(harness.package)


def test_public_verifier_requires_live_canonical_review_bytes(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.create(review, "accepted")
    os.chmod(review, 0o600)
    review.write_bytes(b"{}\n")

    with pytest.raises(AdjudicationError, match="canonical live review"):
        verify_adjudication_package(harness.package)


def test_staged_mutation_race_consumes_claim_and_refuses_publish(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()

    def mutate(staging: Path) -> None:
        target = staging / COPIED_SEMANTIC_REVIEW_NAME
        os.chmod(target, 0o600)
        target.write_bytes(b"{}\n")

    harness.monkeypatch.setattr(
        adjudication,
        "_before_package_publish",
        mutate,
    )

    manifest = harness.create(review, "accepted")

    assert manifest["outcome_kind"] == "adjudication_recovery_failure"
    assert manifest["disposition"] == "rejected"
    assert harness.adjudication_marker.exists()
    assert harness.replay_calls == [harness.upstream.root]
    verify_adjudication_package(harness.package)


def test_unsealed_entry_fails_before_claim_and_runner(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.monkeypatch.setattr(
        adjudication,
        "_require_sealed_entry",
        lambda: (_ for _ in ()).throw(AdjudicationError("sealed bootstrap required")),
    )

    with pytest.raises(AdjudicationError, match="sealed bootstrap"):
        harness.create(review, "accepted")

    assert not harness.adjudication_marker.exists()
    assert harness.replay_calls == []


def test_public_verifier_rejects_absent_and_forged_outer_completion(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.finalize_registration = False
    harness.create(review, "accepted")

    with pytest.raises(AdjudicationError, match="outer completion"):
        verify_adjudication_package(harness.package)

    terminal = harness.package / ADJUDICATION_MANIFEST_NAME
    forged = operator_module.outer_completion_marker(terminal)
    forged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(terminal, forged)
    with pytest.raises(AdjudicationError, match="outer completion"):
        verify_adjudication_package(harness.package)


def test_public_verifier_rejects_copy_and_extra_terminal_link(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    harness.create(review, "accepted")
    copied = harness.tmp_path / "copied-package"
    shutil.copytree(harness.package, copied)

    with pytest.raises(AdjudicationError, match="canonical"):
        verify_adjudication_package(copied)

    extra = harness.tmp_path / "extra-terminal-link.json"
    os.link(harness.package / ADJUDICATION_MANIFEST_NAME, extra)
    with pytest.raises(AdjudicationError, match="outer completion"):
        verify_adjudication_package(harness.package)


def test_semantic_review_rejects_assurance_overstatement(
    harness: Harness,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    value = json.loads(review.read_bytes())
    value["assurance"]["tamper_resistant"] = True
    _write(review, value)
    harness.install_replay()

    with pytest.raises(AdjudicationError, match="trust-model"):
        harness.create(review, "accepted")

    assert not harness.adjudication_marker.exists()
    assert harness.replay_calls == []


def test_main_uses_real_outer_registration_contract(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.make_attempt()
    review = harness.review(disposition="accepted")
    harness.install_replay()
    real_register = producer_bootstrap.register_outer_terminal

    def registrar(path: Path, *, logical_exit_code: int) -> None:
        harness.registrations.append((path, logical_exit_code))
        marker = operator_module.outer_completion_marker(path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, marker)

    monkeypatch.setattr(
        sys,
        "_prospect_wm001_register_outer_terminal",
        registrar,
        raising=False,
    )
    monkeypatch.setattr(
        producer_bootstrap,
        "register_outer_terminal",
        real_register,
    )

    assert (
        main(
            [
                "--audit-attempt",
                str(harness.audit_attempt),
                "--semantic-review",
                str(review),
                "--disposition",
                "accepted",
            ]
        )
        == 0
    )
    assert harness.registrations == [(harness.package / ADJUDICATION_MANIFEST_NAME, 0)]
    verify_adjudication_package(harness.package)
