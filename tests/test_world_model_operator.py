from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from bench.world_model_lifecycle import binding, preformal
from bench.world_model_lifecycle import operator as operator_module
from bench.world_model_lifecycle.assurance import ASSURANCE
from bench.world_model_lifecycle.audit_runner import (
    AuditExecution,
    AuditExecutionFailure,
)
from bench.world_model_lifecycle.operator import OperatorError


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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else _canonical(value))


@dataclass(frozen=True)
class OperatorSpace:
    repo: Path
    binding_root: Path
    audit_root: Path
    closure_root: Path
    completion_root: Path
    formal_binding: Path
    development_audit: Path
    formal_audit: Path
    formal_claim: Path
    development_qualification: Path
    development_closure: Path
    closure: Path
    registrations: list[tuple[Path, int]]


@pytest.fixture
def operator_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> OperatorSpace:
    repo = tmp_path / "repo"
    lifecycle = repo / "bench" / "world_model_lifecycle"
    lifecycle.mkdir(parents=True)
    _write(lifecycle / "protocol.json", {"version": "1.5.0"})
    operator_root = lifecycle / "results" / "operator-v1.5"
    binding_root = operator_root / "bindings"
    audit_root = operator_root / "audits"
    closure_root = operator_root / "closures"
    completion_root = lifecycle / "results" / "outer-completions" / "v1.5"
    formal_binding = binding_root / "formal-binding-v1.5.0"
    development_audit = audit_root / "development-audit-v1.5.0"
    formal_audit = audit_root / "formal-audit-v1.5.0"
    formal_claim = lifecycle / "results" / "formal" / "formal-audit-v1.5.0.json"
    development_qualification = (
        lifecycle / "results" / "development" / "qualification-v1.5.0"
    )
    development_closure = lifecycle / "results" / "development" / "development-closure-v1.5.0.json"
    closure = closure_root / "development-closure-v1.5.0"
    registrations: list[tuple[Path, int]] = []

    for name, value in {
        "REPO": repo,
        "OPERATOR_RESULTS_ROOT": operator_root,
        "BINDING_ATTEMPTS_ROOT": binding_root,
        "AUDIT_ATTEMPTS_ROOT": audit_root,
        "CLOSURE_ATTEMPTS_ROOT": closure_root,
        "OUTER_COMPLETIONS_ROOT": completion_root,
        "FORMAL_BINDING_ATTEMPT_PATH": formal_binding,
        "DEVELOPMENT_AUDIT_ATTEMPT_PATH": development_audit,
        "FORMAL_AUDIT_ATTEMPT_PATH": formal_audit,
        "FORMAL_AUDIT_CLAIM_MARKER": formal_claim,
        "DEVELOPMENT_RESULTS_ROOT": development_qualification.parent,
        "DEVELOPMENT_QUALIFICATION_PATH": development_qualification,
        "CLOSURE_ATTEMPT_PATH": closure,
    }.items():
        monkeypatch.setattr(operator_module, name, value)
    monkeypatch.setattr(
        operator_module,
        "_require_sealed_entry",
        lambda: None,
    )

    def registrar(path: Path, *, logical_exit_code: int) -> None:
        registrations.append((path, logical_exit_code))

    monkeypatch.setattr(
        sys,
        "_prospect_wm001_register_outer_terminal",
        registrar,
        raising=False,
    )
    monkeypatch.setattr(binding, "REPO", repo)
    monkeypatch.setattr(
        binding,
        "DEVELOPMENT_CLOSURE_PATH",
        development_closure,
    )
    monkeypatch.setattr(
        preformal,
        "PREFORMAL_REPORT_PATH",
        development_qualification.parent / preformal.PREFORMAL_REPORT_NAME,
    )
    return OperatorSpace(
        repo=repo,
        binding_root=binding_root,
        audit_root=audit_root,
        closure_root=closure_root,
        completion_root=completion_root,
        formal_binding=formal_binding,
        development_audit=development_audit,
        formal_audit=formal_audit,
        formal_claim=formal_claim,
        development_qualification=development_qualification,
        development_closure=development_closure,
        closure=closure,
        registrations=registrations,
    )


def _finalize(path: Path) -> dict[str, object]:
    terminal = path / "operator-attempt.json"
    marker = operator_module.outer_completion_marker(terminal)
    marker.parent.mkdir(parents=True, exist_ok=True)
    os.link(terminal, marker, follow_symlinks=False)
    assert os.path.samefile(terminal, marker)
    assert terminal.stat().st_nlink == 2
    return operator_module.verify_operator_attempt(path)


def _execution(
    *,
    passed: bool = True,
    variant: str = "stable",
) -> AuditExecution:
    report = {
        "passed": passed,
        "schema": "test.audit.v1",
        "variant": variant,
    }
    stdout = _canonical(report)
    runtime = _canonical(
        {
            "schema": "test.runtime.v1",
            "assurance": dict(ASSURANCE),
        }
    )
    invocation = _canonical({"schema": "test.invocation.v1"})
    return AuditExecution(
        command=("/python", "-I", "-S", "-B", "/proc/self/fd/9"),
        returncode=0 if passed else 1,
        stdout=stdout,
        stderr=b"",
        report=MappingProxyType(report),
        runtime_manifest=runtime,
        runtime_manifest_sha256=hashlib.sha256(runtime).hexdigest(),
        invocation_manifest=invocation,
        invocation_manifest_sha256=hashlib.sha256(invocation).hexdigest(),
        bootstrap_sha256="1" * 64,
        auditor_source_sha256="2" * 64,
        support_files=(),
        source_mode="descriptor",
    )


def _execution_failure() -> AuditExecutionFailure:
    return AuditExecutionFailure(
        "isolated auditor failed before producing a report",
        phase="subprocess",
        command=("/python", "-I", "-S", "-B", "/proc/self/fd/9"),
        returncode=125,
        stdout=b"partial stdout",
        stderr=b"partial stderr",
        runtime_manifest=_canonical(
            {
                "schema": "test.runtime.v1",
                "assurance": dict(ASSURANCE),
            }
        ),
        invocation_manifest=_canonical({"schema": "test.invocation.v1"}),
        bootstrap_sha256="1" * 64,
        auditor_source_sha256="2" * 64,
        support_files=(),
        source_mode="descriptor",
    )


def _patch_finalized_producer(
    monkeypatch: pytest.MonkeyPatch,
    space: OperatorSpace,
    *,
    lane: str,
) -> tuple[Path, dict[str, object]]:
    from bench.world_model_lifecycle import artifact, verify

    producer = (
        space.development_qualification
        if lane == "development"
        else space.repo / "producers" / lane
    )
    execution_identity: dict[str, object] = {
        "process_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        }
    }
    _write(
        producer / "producer-manifest.json",
        {"status": "completed", "lane": lane, "error": None},
    )
    _write(
        producer / "result.json",
        {
            "lane": lane,
            "execution": execution_identity,
        },
    )
    if lane == "formal":
        _write(
            producer / "formal-binding.json",
            {
                "schema": "prospect.world-model-lifecycle.formal-binding.v5",
                "assurance": dict(ASSURANCE),
            },
        )
    terminal = producer / "producer-manifest.json"
    completion = operator_module.outer_completion_marker(terminal)
    completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(terminal, completion, follow_symlinks=False)

    monkeypatch.setattr(
        artifact,
        "verify_producer_manifest",
        lambda root: _load(root / "producer-manifest.json"),
    )
    monkeypatch.setattr(
        verify,
        "verify_result",
        lambda result_path, _binding_path: _load(result_path),
    )
    monkeypatch.setattr(
        binding,
        "_validate_execution_identity",
        lambda value, *, require_live_identity: (
            value
            if value == execution_identity and require_live_identity
            else pytest.fail("unexpected execution identity")
        ),
    )
    monkeypatch.setattr(binding, "package_roots", lambda: (space.repo,))
    monkeypatch.setattr(
        binding,
        "_audit_environment",
        lambda environment: dict(environment),
    )
    return producer, execution_identity


def _write_reproduction_receipt(
    *,
    supplied_audit_path: Path,
    execution: AuditExecution,
    output_path: Path,
) -> dict[str, object]:
    audit_payload = supplied_audit_path.read_bytes()
    assert audit_payload == execution.stdout
    receipt: dict[str, object] = {
        "schema": "prospect.wm001.audit-reproduction.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.5.0",
        "supplied_audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "reproduced_audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "byte_identical": True,
        "returncode": 0,
        "source_mode": "descriptor",
        "stdout_bytes": len(audit_payload),
        "stderr_file": "audit-execution-02.stderr.log",
        "stderr_bytes": len(execution.stderr),
        "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        "runtime_manifest_file": "audit-execution-02.runtime.json",
        "runtime_manifest_bytes": len(execution.runtime_manifest),
        "runtime_manifest_sha256": execution.runtime_manifest_sha256,
        "invocation_manifest_file": "audit-execution-02.invocation.json",
        "invocation_manifest_bytes": len(execution.invocation_manifest),
        "invocation_manifest_sha256": execution.invocation_manifest_sha256,
        "bootstrap_sha256": execution.bootstrap_sha256,
        "runner_source_sha256": "3" * 64,
        "auditor_source_sha256": execution.auditor_source_sha256,
        "support_files": [],
        "passed": True,
    }
    _write(output_path, receipt)
    return receipt


def _make_failure_attempt(
    space: OperatorSpace,
) -> Path:
    final = space.formal_binding
    final.parent.mkdir(parents=True, exist_ok=True)
    attempt = operator_module._Attempt(  # noqa: SLF001
        final=final,
        kind="binding",
        lane=None,
        inputs=[],
    )
    error = RuntimeError("controlled failure")
    record = operator_module._failure_record(  # noqa: SLF001
        kind="binding",
        lane=None,
        phase="test",
        error=error,
    )
    operator_module._write_json(  # noqa: SLF001
        attempt.staging / "execution-failure.json",
        record,
    )
    attempt.finish(
        status="failure",
        primary={"execution_failure_file": "execution-failure.json"},
        error={
            "failure_code": record["failure_code"],
            "error_type": record["error_type"],
        },
        final_check=lambda: None,
    )
    return final


def _run_development_audit(
    monkeypatch: pytest.MonkeyPatch,
    space: OperatorSpace,
    *,
    passed: bool = True,
) -> tuple[Path, Path]:
    from bench.world_model_lifecycle import audit_runner

    producer, _ = _patch_finalized_producer(
        monkeypatch,
        space,
        lane="development",
    )
    execution = _execution(passed=passed)
    calls: list[dict[str, object]] = []

    def runner(_source: Path, **kwargs: object) -> AuditExecution:
        calls.append(kwargs)
        return execution

    monkeypatch.setattr(audit_runner, "run_captured_auditor", runner)
    if passed:
        monkeypatch.setattr(
            binding,
            "create_audit_reproduction_receipt",
            _write_reproduction_receipt,
        )
    output = space.development_audit
    assert operator_module.audit_main(
        [
            "development",
            "--producer",
            str(producer),
            "--output",
            str(output),
        ]
    ) == (0 if passed else 1)
    assert len(calls) == 2
    _finalize(output)
    return producer, output


def _install_closure_fakes(
    monkeypatch: pytest.MonkeyPatch,
    space: OperatorSpace,
) -> tuple[
    Path,
    Callable[[Path, Path, Path], dict[str, object]],
    list[tuple[Path, Path, Path]],
]:
    marker = space.development_closure
    archive = marker.with_name("development-qualification-v1.5.0.tar")
    calls: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(binding, "DEVELOPMENT_CLOSURE_PATH", marker)

    def build(
        producer_root: Path,
        audit_path: Path,
        reproduction_path: Path,
    ) -> dict[str, object]:
        closure: dict[str, object] = {
            "schema": "prospect.wm001.development-closure.v2",
            "producer_root": str(producer_root),
            "engineering_verified": True,
            "audit_reproduced": True,
            "performance_values_bound": False,
            "qualification_archive": {
                "canonical_path": archive.relative_to(space.repo).as_posix(),
                "members": [
                    {
                        "path": "evidence/independent-audit.json",
                        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                    },
                    {
                        "path": "evidence/audit-reproduction.json",
                        "sha256": hashlib.sha256(reproduction_path.read_bytes()).hexdigest(),
                    },
                ],
            },
        }
        _write(archive, b"authenticated archive\n")
        _write(marker, closure)
        return closure

    def create(
        *,
        producer_root: Path,
        audit_path: Path,
        audit_reproduction_path: Path,
        runtime_manifest_path: Path,
    ) -> dict[str, object]:
        assert runtime_manifest_path.name == "audit-execution-02.runtime.json"
        calls.append((producer_root, audit_path, audit_reproduction_path))
        return build(producer_root, audit_path, audit_reproduction_path)

    monkeypatch.setattr(binding, "create_development_closure", create)
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda path: _load(path),
    )
    return marker, build, calls


def test_attempt_requires_outer_hardlink_and_exact_assurance(
    operator_space: OperatorSpace,
) -> None:
    attempt = _make_failure_attempt(operator_space)
    terminal = attempt / "operator-attempt.json"
    unfinalized = operator_module.inspect_unfinalized_operator_attempt(attempt)
    assert unfinalized["outer_finalized"] is False
    assert unfinalized["manifest"]["assurance"] == ASSURANCE
    assert operator_space.registrations == [(terminal, 2)]
    with pytest.raises(OperatorError, match="exactly 2 link"):
        operator_module.verify_operator_attempt(attempt)

    manifest = _finalize(attempt)
    assert manifest["status"] == "failure"
    assert manifest["assurance"] == ASSURANCE


def test_copied_completion_marker_is_not_a_logical_commit(
    operator_space: OperatorSpace,
) -> None:
    attempt = _make_failure_attempt(operator_space)
    terminal = attempt / "operator-attempt.json"
    marker = operator_module.outer_completion_marker(terminal)
    marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(terminal, marker)

    assert not os.path.samefile(terminal, marker)
    with pytest.raises(OperatorError):
        operator_module.verify_operator_attempt(attempt)
    with pytest.raises(OperatorError, match="completion marker"):
        operator_module.inspect_unfinalized_operator_attempt(attempt)


def test_manifest_omission_or_assurance_overstatement_is_rejected(
    operator_space: OperatorSpace,
) -> None:
    attempt = _make_failure_attempt(operator_space)
    _finalize(attempt)
    terminal = attempt / "operator-attempt.json"
    manifest = _load(terminal)
    manifest["assurance"]["tamper_resistant"] = True
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(OperatorError, match="malformed"):
        operator_module.verify_operator_attempt(attempt)


@pytest.mark.parametrize("kind", ["binding", "closure"])
def test_sibling_binding_or_closure_attempt_is_never_canonical(
    operator_space: OperatorSpace,
    kind: str,
) -> None:
    if kind == "binding":
        canonical = _make_failure_attempt(operator_space)
        sibling = operator_space.binding_root / "sibling-binding"
    else:
        canonical = operator_space.closure
        canonical.parent.mkdir(parents=True, exist_ok=True)
        attempt = operator_module._Attempt(  # noqa: SLF001
            final=canonical,
            kind="closure",
            lane="development",
            inputs=[],
        )
        error = RuntimeError("controlled closure failure")
        record = operator_module._failure_record(  # noqa: SLF001
            kind="closure",
            lane="development",
            phase="test",
            error=error,
        )
        operator_module._write_json(  # noqa: SLF001
            attempt.staging / "execution-failure.json",
            record,
        )
        attempt.finish(
            status="failure",
            primary={
                "execution_failure_file": "execution-failure.json"
            },
            error={
                "failure_code": record["failure_code"],
                "error_type": record["error_type"],
            },
            final_check=lambda: None,
        )
        sibling = operator_space.closure_root / "sibling-closure"
    shutil.copytree(canonical, sibling)

    with pytest.raises(OperatorError, match="canonical protocol-1.5"):
        operator_module.inspect_unfinalized_operator_attempt(sibling)

    sibling_terminal = sibling / "operator-attempt.json"
    sibling_completion = operator_module.outer_completion_marker(
        sibling_terminal
    )
    sibling_completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(sibling_terminal, sibling_completion)
    with pytest.raises(OperatorError, match="canonical protocol-1.5"):
        operator_module.verify_operator_attempt(sibling)


def test_staged_output_mutation_prevents_attempt_publication(
    operator_space: OperatorSpace,
) -> None:
    final = operator_space.binding_root / "mutated-before-publication"
    final.parent.mkdir(parents=True, exist_ok=True)
    attempt = operator_module._Attempt(  # noqa: SLF001
        final=final,
        kind="binding",
        lane=None,
        inputs=[],
    )
    error = RuntimeError("controlled failure")
    record = operator_module._failure_record(  # noqa: SLF001
        kind="binding",
        lane=None,
        phase="test",
        error=error,
    )
    operator_module._write_json(  # noqa: SLF001
        attempt.staging / "execution-failure.json",
        record,
    )
    _write(attempt.staging / "partial-output.json", {"before": True})

    with pytest.raises(OperatorError, match="changed before attempt publication"):
        attempt.finish(
            status="failure",
            primary={
                "execution_failure_file": "execution-failure.json",
            },
            error={
                "failure_code": record["failure_code"],
                "error_type": record["error_type"],
            },
            final_check=lambda: _write(
                attempt.staging / "partial-output.json",
                {"after": True},
            ),
        )
    attempt.cleanup()
    assert not final.exists()
    assert operator_space.registrations == []


def test_outer_finalized_producer_enters_real_audit_custody(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import audit_runner, verify
    from bench.world_model_lifecycle.artifact import (
        MANIFEST_NAME,
        ProducerAttempt,
        atomic_write_exclusive,
        verify_producer_manifest,
    )

    producer = operator_space.development_qualification
    producer.parent.mkdir(parents=True)
    execution_identity: dict[str, object] = {
        "process_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        }
    }
    with ProducerAttempt(producer, lane="development"):
        atomic_write_exclusive(
            producer / "result.json",
            _canonical(
                {
                    "lane": "development",
                    "execution": execution_identity,
                }
            ),
        )
    terminal = producer / MANIFEST_NAME
    completion = operator_module.outer_completion_marker(terminal)
    completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(terminal, completion, follow_symlinks=False)
    assert terminal.stat().st_nlink == 2
    assert verify_producer_manifest(producer)["status"] == "completed"

    monkeypatch.setattr(
        verify,
        "verify_result",
        lambda result_path, _binding_path: _load(result_path),
    )
    monkeypatch.setattr(
        binding,
        "_validate_execution_identity",
        lambda value, *, require_live_identity: (
            value
            if value == execution_identity and require_live_identity
            else pytest.fail("unexpected execution identity")
        ),
    )
    monkeypatch.setattr(
        binding,
        "package_roots",
        lambda: (operator_space.repo,),
    )
    monkeypatch.setattr(
        binding,
        "_audit_environment",
        lambda environment: dict(environment),
    )
    execution = _execution()
    calls: list[Path] = []

    def runner(source: Path, **_kwargs: object) -> AuditExecution:
        calls.append(source)
        return execution

    monkeypatch.setattr(audit_runner, "run_captured_auditor", runner)
    monkeypatch.setattr(
        binding,
        "create_audit_reproduction_receipt",
        _write_reproduction_receipt,
    )
    output = operator_space.development_audit

    assert (
        operator_module.audit_main(
            [
                "development",
                "--producer",
                str(producer),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    audit_source = Path(operator_module.__file__).with_name("artifact_audit.py")
    assert calls == [audit_source, audit_source]
    manifest = operator_module.inspect_unfinalized_operator_attempt(output)["manifest"]
    assert manifest["status"] == "accepted"
    assert any(row["path"] == str(terminal) for row in manifest["inputs"])
    assert terminal.stat().st_nlink == 2


def test_development_audit_rejects_noncanonical_attempt_name(
    operator_space: OperatorSpace,
) -> None:
    with pytest.raises(OperatorError, match="sole canonical"):
        operator_module.audit_main(
            [
                "development",
                "--producer",
                str(operator_space.development_qualification),
                "--output",
                str(operator_space.audit_root / "another-development-audit"),
            ]
        )
    assert not operator_space.development_audit.exists()


def test_development_audit_is_retired_by_closure_marker(
    operator_space: OperatorSpace,
) -> None:
    _write(operator_space.development_closure, {"closed": True})

    with pytest.raises(OperatorError, match="retired after development closure"):
        operator_module.audit_main(
            [
                "development",
                "--producer",
                str(operator_space.development_qualification),
                "--output",
                str(operator_space.development_audit),
            ]
        )
    assert not operator_space.development_audit.exists()


def test_closure_rejects_noncanonical_development_audit(
    operator_space: OperatorSpace,
) -> None:
    with pytest.raises(OperatorError, match="canonical protocol-1.5 audit"):
        operator_module.closure_main(
            [
                "--producer",
                str(operator_space.development_qualification),
                "--audit-attempt",
                str(operator_space.audit_root / "rogue-development-audit"),
                "--output",
                str(operator_space.closure),
            ]
        )
    assert not operator_space.closure.exists()


@pytest.mark.parametrize("entry", ["audit", "closure"])
def test_development_authority_rejects_sibling_qualification_producer(
    operator_space: OperatorSpace,
    entry: str,
) -> None:
    sibling = (
        operator_space.development_qualification.parent
        / "qualification-v1.5.0-copy"
    )
    if entry == "audit":
        arguments = [
            "development",
            "--producer",
            str(sibling),
            "--output",
            str(operator_space.development_audit),
        ]
        invoke = operator_module.audit_main
    else:
        arguments = [
            "--producer",
            str(sibling),
            "--audit-attempt",
            str(operator_space.development_audit),
            "--output",
            str(operator_space.closure),
        ]
        invoke = operator_module.closure_main

    with pytest.raises(OperatorError, match="sole canonical qualification producer"):
        invoke(arguments)


@pytest.mark.parametrize(
    ("passed", "expected_status", "expected_code"),
    [(True, "accepted", 0), (False, "rejected", 1)],
)
def test_development_audit_always_replays_and_preserves_outcome(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    expected_status: str,
    expected_code: int,
) -> None:
    from bench.world_model_lifecycle import audit_runner

    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="development",
    )
    execution = _execution(passed=passed)
    calls: list[dict[str, object]] = []

    def runner(_source: Path, **kwargs: object) -> AuditExecution:
        calls.append(kwargs)
        return execution

    monkeypatch.setattr(audit_runner, "run_captured_auditor", runner)
    if passed:
        monkeypatch.setattr(
            binding,
            "create_audit_reproduction_receipt",
            _write_reproduction_receipt,
        )
    else:
        monkeypatch.setattr(
            binding,
            "create_audit_reproduction_receipt",
            lambda **_kwargs: pytest.fail("rejected audit must not be qualified"),
        )
    output = operator_space.development_audit

    assert (
        operator_module.audit_main(
            [
                "development",
                "--producer",
                str(producer),
                "--output",
                str(output),
            ]
        )
        == expected_code
    )
    assert len(calls) == 2
    assert calls[0]["source_mode"] == "descriptor"
    assert calls[1]["runtime_manifest"] == execution.runtime_manifest
    assert calls[1]["invocation_manifest"] == execution.invocation_manifest
    unfinalized = operator_module.inspect_unfinalized_operator_attempt(output)
    manifest = unfinalized["manifest"]
    assert manifest["status"] == expected_status
    primary = manifest["primary"]
    assert len(primary["executions"]) == 2
    assert primary["execution_failures"] == []
    assert (primary["reproduction_file"] is not None) is passed
    assert operator_space.registrations[-1][1] == expected_code

    assert _finalize(output) == manifest


def test_replay_mismatch_becomes_authenticated_failure_with_both_runs(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import audit_runner

    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="development",
    )
    executions = [
        _execution(variant="first"),
        _execution(variant="different"),
    ]
    calls = 0

    def runner(_source: Path, **_kwargs: object) -> AuditExecution:
        nonlocal calls
        execution = executions[calls]
        calls += 1
        return execution

    monkeypatch.setattr(audit_runner, "run_captured_auditor", runner)
    output = operator_space.development_audit

    assert (
        operator_module.audit_main(
            [
                "development",
                "--producer",
                str(producer),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    manifest = operator_module.inspect_unfinalized_operator_attempt(output)["manifest"]
    assert manifest["status"] == "failure"
    assert manifest["primary"]["executions"] == [
        "audit-execution-01.execution.json",
        "audit-execution-02.execution.json",
    ]
    assert manifest["primary"]["execution_failures"] == []
    assert _finalize(output) == manifest


def test_second_auditor_bootstrap_failure_preserves_partial_evidence(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import audit_runner

    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="development",
    )
    first = _execution()
    calls = 0

    def runner(_source: Path, **_kwargs: object) -> AuditExecution:
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise _execution_failure()

    monkeypatch.setattr(audit_runner, "run_captured_auditor", runner)
    output = operator_space.development_audit

    assert (
        operator_module.audit_main(
            [
                "development",
                "--producer",
                str(producer),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    manifest = operator_module.inspect_unfinalized_operator_attempt(output)["manifest"]
    primary = manifest["primary"]
    assert primary["executions"] == ["audit-execution-01.execution.json"]
    assert primary["execution_failures"] == ["audit-execution-02.failure.json"]
    assert (output / "audit-execution-02.partial.stdout").read_bytes() == (b"partial stdout")
    assert _finalize(output) == manifest


def _patch_formal_binding_verifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    launch_digest: str | None = None,
) -> str:
    from bench.world_model_lifecycle import verify

    live_digest = hashlib.sha256(
        Path(operator_module.__file__).with_name("launch_bootstrap.py").read_bytes()
    ).hexdigest()
    bound_digest = live_digest if launch_digest is None else launch_digest
    monkeypatch.setattr(
        verify,
        "verify_binding",
        lambda _path: {
            "source": {
                "execution_source_sha256": {
                    "launch_bootstrap.py": bound_digest,
                }
            }
        },
    )
    return live_digest


@pytest.mark.parametrize(
    ("passed", "expected_status", "expected_code"),
    [(True, "accepted", 0), (False, "rejected", 1)],
)
def test_formal_audit_consumes_one_claim_and_runs_once(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
    expected_status: str,
    expected_code: int,
) -> None:
    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="formal",
    )
    _patch_formal_binding_verifier(monkeypatch)
    execution = _execution(passed=passed)
    calls: list[Path] = []

    def runner(root: Path) -> AuditExecution:
        calls.append(root)
        return execution

    monkeypatch.setattr(binding, "run_bound_outcome_audit", runner)
    assert (
        operator_module.audit_main(
            [
                "formal",
                "--producer",
                str(producer),
                "--output",
                str(operator_space.formal_audit),
            ]
        )
        == expected_code
    )
    assert calls == [producer]
    claim = operator_space.formal_audit / "formal-audit-claim.json"
    assert os.path.samefile(claim, operator_space.formal_claim)
    assert claim.stat().st_nlink == 2
    manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.formal_audit)["manifest"]
    assert manifest["status"] == expected_status
    assert len(manifest["primary"]["executions"]) == 1
    _finalize(operator_space.formal_audit)

    with pytest.raises(OperatorError, match="already consumed"):
        operator_module.audit_main(
            [
                "formal",
                "--producer",
                str(producer),
                "--output",
                str(operator_space.formal_audit),
            ]
        )


def test_formal_audit_failure_preserves_claim_and_partial_execution(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="formal",
    )
    _patch_formal_binding_verifier(monkeypatch)
    monkeypatch.setattr(
        binding,
        "run_bound_outcome_audit",
        lambda _root: (_ for _ in ()).throw(_execution_failure()),
    )

    assert (
        operator_module.audit_main(
            [
                "formal",
                "--producer",
                str(producer),
                "--output",
                str(operator_space.formal_audit),
            ]
        )
        == 2
    )
    manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.formal_audit)["manifest"]
    assert manifest["status"] == "failure"
    assert manifest["primary"]["executions"] == []
    assert manifest["primary"]["execution_failures"] == ["audit-execution-01.failure.json"]
    assert os.path.samefile(
        operator_space.formal_claim,
        operator_space.formal_audit / "formal-audit-claim.json",
    )
    _finalize(operator_space.formal_audit)


def test_formal_claim_post_link_fault_publishes_failure_without_runner(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="formal",
    )
    _patch_formal_binding_verifier(monkeypatch)
    runner_calls: list[Path] = []

    def runner(root: Path) -> AuditExecution:
        runner_calls.append(root)
        return _execution()

    monkeypatch.setattr(binding, "run_bound_outcome_audit", runner)
    monkeypatch.setattr(
        operator_module,
        "_after_formal_audit_claim_link",
        lambda: (_ for _ in ()).throw(OSError("injected post-link fault")),
    )

    assert (
        operator_module.audit_main(
            [
                "formal",
                "--producer",
                str(producer),
                "--output",
                str(operator_space.formal_audit),
            ]
        )
        == 2
    )
    assert runner_calls == []
    claim = operator_space.formal_audit / "formal-audit-claim.json"
    assert os.path.samefile(claim, operator_space.formal_claim)
    assert claim.stat().st_nlink == 2
    manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.formal_audit)["manifest"]
    assert manifest["status"] == "failure"
    assert manifest["primary"]["claim_file"] == "formal-audit-claim.json"
    assert manifest["primary"]["executions"] == []
    assert manifest["primary"]["execution_failures"] == []
    assert manifest["error"]["error_type"] == "OSError"
    assert _finalize(operator_space.formal_audit) == manifest


def test_formal_claim_rejects_unbound_live_launcher_before_consumption(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _patch_finalized_producer(
        monkeypatch,
        operator_space,
        lane="formal",
    )
    _patch_formal_binding_verifier(
        monkeypatch,
        launch_digest="0" * 64,
    )
    monkeypatch.setattr(
        binding,
        "run_bound_outcome_audit",
        lambda _root: pytest.fail("runner must not start"),
    )

    with pytest.raises(OperatorError, match="does not bind"):
        operator_module.audit_main(
            [
                "formal",
                "--producer",
                str(producer),
                "--output",
                str(operator_space.formal_audit),
            ]
        )
    assert not operator_space.formal_claim.exists()
    assert not operator_space.formal_audit.exists()


def test_formal_audit_rejects_noncanonical_attempt_name(
    operator_space: OperatorSpace,
) -> None:
    with pytest.raises(OperatorError, match="sole canonical"):
        operator_module.audit_main(
            [
                "formal",
                "--producer",
                str(operator_space.repo / "missing-producer"),
                "--output",
                str(operator_space.audit_root / "another-formal-attempt"),
            ]
        )


@pytest.mark.parametrize("recover_existing_marker", [False, True])
def test_development_closure_then_binding_end_to_end(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
    recover_existing_marker: bool,
) -> None:
    from bench.world_model_lifecycle import verify

    producer, audit_attempt = _run_development_audit(
        monkeypatch,
        operator_space,
    )
    marker, build_closure, closure_calls = _install_closure_fakes(
        monkeypatch,
        operator_space,
    )
    audit_path = audit_attempt / "independent-audit.json"
    reproduction_path = audit_attempt / "audit-reproduction.json"
    if recover_existing_marker:
        build_closure(producer, audit_path, reproduction_path)

    assert (
        operator_module.closure_main(
            [
                "--producer",
                str(producer),
                "--audit-attempt",
                str(audit_attempt),
                "--output",
                str(operator_space.closure),
            ]
        )
        == 0
    )
    assert len(closure_calls) == (0 if recover_existing_marker else 1)
    closure_manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.closure)["manifest"]
    assert closure_manifest["status"] == "accepted"
    _finalize(operator_space.closure)
    closure_terminal = (
        operator_space.closure / "operator-attempt.json"
    )
    original_closure_terminal = closure_terminal.read_bytes()
    substituted_closure = json.loads(original_closure_terminal)
    substituted_closure["inputs"][0]["sha256"] = "0" * 64
    closure_terminal.write_bytes(_canonical(substituted_closure))
    with pytest.raises(
        OperatorError,
        match="closure authorization inputs differ",
    ):
        operator_module.verify_operator_attempt(operator_space.closure)
    closure_terminal.write_bytes(original_closure_terminal)
    assert (
        operator_module.verify_operator_attempt(operator_space.closure)
        == closure_manifest
    )

    report = marker.with_name("preformal-test-report-v1.5.0.json")
    _write(report, {"passed": True})
    monkeypatch.setattr(
        binding,
        "verify_machine_test_report",
        lambda path: _load(path),
    )
    monkeypatch.setattr(
        binding,
        "preformal_log_rows",
        lambda _path, _report: [],
    )
    expected_binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v5",
        "assurance": dict(ASSURANCE),
        "source": {
            "test_report_file": report.name,
            "test_report_bytes": report.stat().st_size,
            "test_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "test_log_files": [],
        },
        "development_qualification": {
            "closure_bytes": marker.stat().st_size,
            "closure_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        },
    }

    def create_binding(**kwargs: object) -> dict[str, object]:
        _write(Path(str(kwargs["output_path"])), expected_binding)
        return expected_binding

    monkeypatch.setattr(binding, "create_formal_binding", create_binding)
    monkeypatch.setattr(
        verify,
        "verify_binding",
        lambda path: _load(path),
    )
    assert (
        operator_module.binding_main(
            [
                "--output",
                str(operator_space.formal_binding),
                "--test-report",
                str(report),
                "--development-closure",
                str(marker),
                "--closure-attempt",
                str(operator_space.closure),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    binding_manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.formal_binding)["manifest"]
    assert binding_manifest["status"] == "accepted"
    input_paths = {row["path"] for row in binding_manifest["inputs"]}
    assert str(closure_terminal) in input_paths
    assert str(operator_module.outer_completion_marker(closure_terminal)) in (input_paths)
    assert _finalize(operator_space.formal_binding) == binding_manifest

    extra_terminal_link = operator_space.repo / "extra-closure-terminal.json"
    os.link(closure_terminal, extra_terminal_link)
    with pytest.raises(OperatorError, match="exactly 2 link"):
        operator_module.verify_operator_attempt(operator_space.formal_binding)
    extra_terminal_link.unlink()
    assert operator_module.verify_operator_attempt(operator_space.formal_binding) == binding_manifest

    binding_terminal = operator_space.formal_binding / "operator-attempt.json"
    tampered = _load(binding_terminal)
    tampered["inputs"][0]["sha256"] = "0" * 64
    binding_terminal.write_bytes(_canonical(tampered))
    with pytest.raises(
        OperatorError,
        match="authorization inputs differ",
    ):
        operator_module.verify_operator_attempt(operator_space.formal_binding)


def test_binding_rejects_unfinalized_closure_attempt(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, audit_attempt = _run_development_audit(
        monkeypatch,
        operator_space,
    )
    marker, _, _ = _install_closure_fakes(monkeypatch, operator_space)
    assert (
        operator_module.closure_main(
            [
                "--producer",
                str(producer),
                "--audit-attempt",
                str(audit_attempt),
                "--output",
                str(operator_space.closure),
            ]
        )
        == 0
    )
    report = marker.with_name(preformal.PREFORMAL_REPORT_NAME)
    _write(report, {"passed": True})

    with pytest.raises(OperatorError, match="exactly 2 link"):
        operator_module.binding_main(
            [
                "--output",
                str(operator_space.formal_binding),
                "--test-report",
                str(report),
                "--development-closure",
                str(marker),
                "--closure-attempt",
                str(operator_space.closure),
                "--device",
                "cpu",
            ]
        )
    assert not operator_space.formal_binding.exists()


def test_binding_failure_preserves_partial_output_atomically(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, audit_attempt = _run_development_audit(
        monkeypatch,
        operator_space,
    )
    marker, _, _ = _install_closure_fakes(monkeypatch, operator_space)
    assert (
        operator_module.closure_main(
            [
                "--producer",
                str(producer),
                "--audit-attempt",
                str(audit_attempt),
                "--output",
                str(operator_space.closure),
            ]
        )
        == 0
    )
    _finalize(operator_space.closure)
    report = marker.with_name(preformal.PREFORMAL_REPORT_NAME)
    _write(report, {"passed": True})
    monkeypatch.setattr(
        binding,
        "verify_machine_test_report",
        lambda path: _load(path),
    )
    monkeypatch.setattr(
        binding,
        "preformal_log_rows",
        lambda _path, _report: [],
    )

    def fail_after_partial(**kwargs: object) -> dict[str, object]:
        _write(
            Path(str(kwargs["output_path"])),
            {"partial": True},
        )
        raise RuntimeError("binding construction failed")

    monkeypatch.setattr(
        binding,
        "create_formal_binding",
        fail_after_partial,
    )
    assert (
        operator_module.binding_main(
            [
                "--output",
                str(operator_space.formal_binding),
                "--test-report",
                str(report),
                "--development-closure",
                str(marker),
                "--closure-attempt",
                str(operator_space.closure),
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.formal_binding)["manifest"]
    assert manifest["status"] == "failure"
    assert (operator_space.formal_binding / "formal-binding.json").is_file()
    assert _finalize(operator_space.formal_binding) == manifest


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--output",
            "/tmp/a",
            "--output",
            "/tmp/b",
            "--test-report",
            "/tmp/t",
            "--development-closure",
            "/tmp/c",
            "--closure-attempt",
            "/tmp/o",
            "--device",
            "cpu",
        ],
        [
            "development",
            "--producer",
            "/tmp/p",
            "--output",
            "/tmp/a",
            "--output",
            "/tmp/b",
        ],
    ],
)
def test_operator_parsers_reject_repeated_custody_options(
    arguments: list[str],
) -> None:
    entry = operator_module.audit_main if arguments[0] == "development" else operator_module.binding_main
    with pytest.raises(SystemExit):
        entry(arguments)
