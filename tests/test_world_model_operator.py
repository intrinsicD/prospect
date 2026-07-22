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

from bench.world_model_lifecycle import artifact_audit, binding, preformal
from bench.world_model_lifecycle import operator as operator_module
from bench.world_model_lifecycle.assurance import ASSURANCE
from bench.world_model_lifecycle.audit_runner import (
    INVOCATION_MANIFEST_SCHEMA,
    RUNTIME_MANIFEST_SCHEMA,
    AuditExecution,
    AuditExecutionFailure,
    CapturedFileIdentity,
    bootstrap_source_sha256,
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


def _write_preformal_logs(
    report_path: Path,
    *,
    command_count: int,
) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    for ordinal in range(1, command_count + 1):
        streams: dict[str, dict[str, object]] = {}
        for stream, payload in (
            ("stdout", f"command-{ordinal:02d}: passed\n".encode()),
            ("stderr", b""),
        ):
            filename = f"command-{ordinal:02d}.{stream}.log"
            _write(report_path.with_name(filename), payload)
            streams[stream] = {
                "file": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        commands.append(
            {
                "ordinal": ordinal,
                "stdout": streams["stdout"],
                "stderr": streams["stderr"],
            }
        )
    report: dict[str, object] = {"commands": commands}
    _write(report_path, report)
    return report


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
    _write(lifecycle / "protocol.json", {"version": "1.19.0"})
    operator_root = lifecycle / "results" / "operator-v1.19"
    binding_root = operator_root / "bindings"
    audit_root = operator_root / "audits"
    closure_root = operator_root / "closures"
    completion_root = lifecycle / "results" / "outer-completions" / "v1.19"
    formal_binding = binding_root / "formal-binding-v1.19.0"
    development_audit = audit_root / "development-audit-v1.19.0"
    formal_audit = audit_root / "formal-audit-v1.19.0"
    formal_claim = lifecycle / "results" / "formal" / "formal-audit-v1.19.0.json"
    development_qualification = lifecycle / "results" / "development" / "qualification-v1.19.0"
    development_closure = lifecycle / "results" / "development" / "development-closure-v1.19.0.json"
    closure = closure_root / "development-closure-v1.19.0"
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
        development_qualification.parent / "v1.19.0" / "preformal" / preformal.PREFORMAL_REPORT_NAME,
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
    stderr: bytes = b"",
    subprocess_elapsed_ns: int = 100_000,
) -> AuditExecution:
    report = {
        "passed": passed,
        "schema": "test.audit.v1",
        "variant": variant,
    }
    stdout = _canonical(report)
    auditor = Path(binding.__file__).with_name("artifact_audit.py")
    support_rows = binding._expected_development_audit_support_rows()
    bootstrap_sha256 = bootstrap_source_sha256()
    runtime = _canonical(
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "execution_role": "outcome_audit",
            "assurance": dict(ASSURANCE),
            "bootstrap_sha256": bootstrap_sha256,
            "source": {
                "mode": "descriptor",
                "path": "artifact_audit.py",
                "bytes": auditor.stat().st_size,
                "sha256": binding.sha256_file(auditor),
            },
            "support_files": support_rows,
            "limits": {
                "timeout_seconds": 10_800,
                "stdout_bytes": 64 << 20,
                "stderr_bytes": 16 << 20,
            },
            "variant": variant,
        }
    )
    runtime_sha256 = hashlib.sha256(runtime).hexdigest()
    invocation = _canonical(
        {
            "schema": INVOCATION_MANIFEST_SCHEMA,
            "runtime_manifest_sha256": runtime_sha256,
            "variant": variant,
        }
    )
    return AuditExecution(
        command=(sys.executable, "-I", "-S", "-B", "/proc/self/fd/9"),
        returncode=0 if passed else 1,
        stdout=stdout,
        stderr=stderr,
        report=MappingProxyType(report),
        runtime_manifest=runtime,
        runtime_manifest_sha256=runtime_sha256,
        invocation_manifest=invocation,
        invocation_manifest_sha256=hashlib.sha256(invocation).hexdigest(),
        bootstrap_sha256=bootstrap_sha256,
        auditor_source_sha256=binding.sha256_file(auditor),
        support_files=tuple(
            CapturedFileIdentity(
                relative_path=str(row["path"]),
                bytes=int(row["bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in support_rows
        ),
        source_mode="descriptor",
        subprocess_elapsed_ns=subprocess_elapsed_ns,
    )


def _capacity_manifest(
    *,
    result_payload: bytes,
    additional_files: tuple[tuple[str, bytes], ...] = (),
) -> tuple[dict[str, object], str]:
    files = [
        {
            "path": "result.json",
            "bytes": len(result_payload),
            "sha256": hashlib.sha256(result_payload).hexdigest(),
        },
        *[
            {
                "path": path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in additional_files
        ],
    ]
    manifest: dict[str, object] = {
        "status": "completed",
        "lane": "development",
        "error": None,
        "files": files,
    }
    return manifest, hashlib.sha256(_canonical(manifest)).hexdigest()


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

    producer = space.development_qualification if lane == "development" else space.repo / "producers" / lane
    execution_identity: dict[str, object] = {
        "process_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LAZY_LEGACY_OP": "False",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
            "SDL_AUDIODRIVER": "dsp",
            "TZ": "UTC",
        }
    }
    result_payload = _canonical(
        {
            "lane": lane,
            "execution": execution_identity,
        }
    )
    _write(producer / "result.json", result_payload)
    additional_files: tuple[tuple[str, bytes], ...] = ()
    if lane == "formal":
        formal_binding_payload = _canonical(
            {
                "schema": "prospect.world-model-lifecycle.formal-binding.v10",
                "assurance": dict(ASSURANCE),
            }
        )
        _write(producer / "formal-binding.json", formal_binding_payload)
        additional_files = (("formal-binding.json", formal_binding_payload),)
    producer_manifest, _ = _capacity_manifest(
        result_payload=result_payload,
        additional_files=additional_files,
    )
    producer_manifest["lane"] = lane
    _write(producer / "producer-manifest.json", producer_manifest)
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
        phase="binding",
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


def test_failure_record_binds_bounded_message_diagnostic() -> None:
    message = (
        "development result exceeds its byte limit " * (operator_module._FAILURE_MESSAGE_CHUNK_CHARACTERS // 8)
        + "\udcff"
    )
    record = operator_module._failure_record(  # noqa: SLF001
        kind="closure",
        lane="development",
        phase="development_closure",
        error=RuntimeError(message),
    )

    assert record["schema"] == "prospect.wm001.operator-execution-failure.v2"
    message_payload = message.encode(
        "utf-8",
        errors="backslashreplace",
    )
    assert record["error_message_bytes"] == len(message_payload)
    assert record["error_message_sha256"] == hashlib.sha256(message_payload).hexdigest()
    assert "error_message" not in record


def test_failure_record_rejects_noncanonical_phase() -> None:
    with pytest.raises(OperatorError, match="canonical kind, phase"):
        operator_module._failure_record(  # noqa: SLF001
            kind="closure",
            lane="development",
            phase="binding",
            error=RuntimeError("controlled failure"),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"error_message": "raw diagnostic leakage"},
        {"phase": "test"},
        {
            "error_type": "",
            "failure_code": "",
        },
        {
            "error_type": "Bad Type",
            "failure_code": "bad_type",
        },
        {
            "error_type": "DifferentError",
            "failure_code": "runtime_error",
        },
        {"failure_code": "other_error"},
        {"error_message_bytes": False},
        {"error_message_bytes": 18.0},
        {"error_message_sha256": "g" * 64},
    ],
    ids=[
        "raw-message",
        "wrong-phase",
        "empty-type-code",
        "invalid-identifier-shape",
        "type-code-mismatch",
        "code-mismatch",
        "boolean-message-bytes",
        "float-message-bytes",
        "malformed-message-digest",
    ],
)
def test_public_failure_verifier_rejects_diagnostic_mutations(
    operator_space: OperatorSpace,
    mutation: dict[str, object],
) -> None:
    attempt = _make_failure_attempt(operator_space)
    failure_path = attempt / "execution-failure.json"
    failure = _load(failure_path)
    failure.update(mutation)
    failure_path.write_bytes(_canonical(failure))
    terminal = attempt / "operator-attempt.json"
    manifest = _load(terminal)
    manifest["error"] = {
        "failure_code": failure.get("failure_code"),
        "error_type": failure.get("error_type"),
    }
    manifest["files"] = operator_module._file_rows(  # noqa: SLF001
        attempt,
        exclude={"operator-attempt.json"},
    )
    manifest["file_count"] = len(manifest["files"])
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        OperatorError,
        match="failure record is malformed",
    ):
        _finalize(attempt)


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
    for call in calls:
        assert call["execution_role"] == "outcome_audit"
        auditor_arguments = call["auditor_arguments"]
        assert isinstance(auditor_arguments, tuple)
        assert auditor_arguments[1:] == (
            "--producer-bootstrap",
            "@captured/producer_bootstrap.py",
        )
        support_files = call["support_files"]
        assert isinstance(support_files, dict)
        assert sorted(support_files) == [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ]
    if passed:
        manifest = operator_module.inspect_unfinalized_operator_attempt(output)["manifest"]
        primary = manifest["primary"]
        assert isinstance(primary, dict)
        receipt = _load(output / "audit-reproduction.json")
        runtime_name = receipt["runtime_manifest_file"]
        invocation_name = receipt["invocation_manifest_file"]
        stderr_name = receipt["stderr_file"]
        assert primary["reproduction_runtime_file"] == runtime_name
        assert isinstance(runtime_name, str)
        assert isinstance(invocation_name, str)
        assert isinstance(stderr_name, str)
        assert runtime_name.startswith("development-audit-runtime-")
        assert invocation_name.startswith("development-audit-invocation-")
        assert stderr_name.startswith("development-audit-stderr-")
        assert runtime_name != "audit-execution-02.runtime.json"
        assert invocation_name != "audit-execution-02.invocation.json"
        assert stderr_name != "audit-execution-02.stderr.log"
        assert (output / runtime_name).read_bytes() == (output / "audit-execution-02.runtime.json").read_bytes()
        assert (output / invocation_name).read_bytes() == (output / "audit-execution-02.invocation.json").read_bytes()
        assert (output / stderr_name).read_bytes() == (output / "audit-execution-02.stderr.log").read_bytes()
        assert not os.path.samefile(
            output / runtime_name,
            output / "audit-execution-02.runtime.json",
        )
    _finalize(output)
    return producer, output


def test_reproduction_sidecar_names_are_payload_derived_and_consumed(
    tmp_path: Path,
) -> None:
    observed_names: list[tuple[str, str, str]] = []
    for variant, stderr in (
        ("alpha", b"alpha stderr\n"),
        ("beta", b"beta stderr\n"),
    ):
        root = tmp_path / variant
        root.mkdir()
        first_execution = _execution(
            variant=variant,
            stderr=stderr,
            subprocess_elapsed_ns=90_000,
        )
        execution = _execution(
            variant=variant,
            stderr=stderr,
            subprocess_elapsed_ns=100_000,
        )
        operator_module._write_execution(  # noqa: SLF001
            root,
            prefix="audit-execution-01",
            execution=first_execution,
        )
        operator_module._write_execution(  # noqa: SLF001
            root,
            prefix="audit-execution-02",
            execution=execution,
        )
        audit_path = root / "independent-audit.json"
        audit_path.write_bytes(execution.stdout)
        receipt_path = root / "audit-reproduction.json"
        producer_manifest, producer_manifest_sha256 = _capacity_manifest(
            result_payload=_canonical(
                {
                    "lane": "development",
                    "observations": ["small-fixture"] * 32,
                }
            )
        )
        receipt = binding.create_audit_reproduction_receipt(
            supplied_audit_path=audit_path,
            first_execution=first_execution,
            execution=execution,
            first_execution_receipt_path=(
                root / "audit-execution-01.execution.json"
            ),
            replay_execution_receipt_path=(
                root / "audit-execution-02.execution.json"
            ),
            producer_manifest=producer_manifest,
            producer_manifest_sha256=producer_manifest_sha256,
            output_path=receipt_path,
        )
        first = operator_module._verify_execution_receipt(  # noqa: SLF001
            root,
            "audit-execution-01.execution.json",
        )
        replay = operator_module._verify_execution_receipt(  # noqa: SLF001
            root,
            "audit-execution-02.execution.json",
        )
        verified = operator_module._verify_reproduction_receipt(  # noqa: SLF001
            root,
            receipt_path.name,
            audit_payload=execution.stdout,
            first=first,
            replay=replay,
            producer_manifest=producer_manifest,
            producer_manifest_sha256=producer_manifest_sha256,
        )

        names = (
            str(receipt["runtime_manifest_file"]),
            str(receipt["invocation_manifest_file"]),
            str(receipt["stderr_file"]),
        )
        assert verified == receipt
        assert receipt["schema"] == "prospect.wm001.audit-reproduction.v3"
        capacity = receipt["capacity"]
        assert isinstance(capacity, dict)
        assert capacity["schema"] == "prospect.wm001.audit-capacity.v1"
        assert capacity["producer_manifest_sha256"] == producer_manifest_sha256
        assert capacity["first_elapsed_ns"] == 90_000
        assert capacity["replay_elapsed_ns"] == 100_000
        assert capacity["combined_required_ns"] <= capacity["available_timeout_ns"]
        execution_receipt = _load(root / "audit-execution-02.execution.json")
        assert execution_receipt["schema"] == ("prospect.wm001.captured-audit-execution.v2")
        assert execution_receipt["subprocess_elapsed_ns"] == 100_000
        runtime_manifest = json.loads(execution.runtime_manifest)
        assert runtime_manifest["schema"] == ("prospect.wm001.audit-runtime-manifest.v2")
        assert runtime_manifest["execution_role"] == "outcome_audit"
        assert runtime_manifest["limits"]["timeout_seconds"] == 10_800
        assert names[0] == (f"development-audit-runtime-{execution.runtime_manifest_sha256[:16]}.json")
        assert names[1] == (f"development-audit-invocation-{execution.invocation_manifest_sha256[:16]}.json")
        assert names[2] == (f"development-audit-stderr-{hashlib.sha256(stderr).hexdigest()[:16]}.log")
        assert (root / names[0]).read_bytes() == execution.runtime_manifest
        assert (root / names[1]).read_bytes() == execution.invocation_manifest
        assert (root / names[2]).read_bytes() == stderr
        assert not os.path.samefile(
            root / names[0],
            root / "audit-execution-02.runtime.json",
        )
        observed_names.append(names)

    assert all(
        left != right
        for left, right in zip(
            observed_names[0],
            observed_names[1],
            strict=True,
        )
    )


def _install_closure_fakes(
    monkeypatch: pytest.MonkeyPatch,
    space: OperatorSpace,
) -> tuple[
    Path,
    Callable[[Path, Path, Path], dict[str, object]],
    list[tuple[Path, Path, Path]],
]:
    marker = space.development_closure
    archive = marker.with_name("development-qualification-v1.19.0.tar")
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
        reproduction = _load(audit_reproduction_path)
        runtime_name = reproduction["runtime_manifest_file"]
        assert isinstance(runtime_name, str)
        assert runtime_manifest_path.name == runtime_name
        assert runtime_name.startswith("development-audit-runtime-")
        execution_capture = audit_reproduction_path.parent / "audit-execution-02.runtime.json"
        assert runtime_manifest_path.read_bytes() == (execution_capture.read_bytes())
        assert not os.path.samefile(
            runtime_manifest_path,
            execution_capture,
        )
        calls.append((producer_root, audit_path, audit_reproduction_path))
        return build(producer_root, audit_path, audit_reproduction_path)

    monkeypatch.setattr(binding, "create_development_closure", create)
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda path: _load(path),
    )
    fresh_report = {
        "schema": "prospect.wm001.development-closure-fresh-reopen.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.19.0",
        "mode": "fresh-closure-reopen",
        "challenge": "1" * 64,
        "requesting_process_id": 100,
        "verifier_process_id": 101,
        "matrix_contract_sha256": (binding._DEVELOPMENT_MATRIX_CONTRACT_SHA256),
        "development_closure_sha256": "2" * 64,
        "producer_manifest_sha256": "3" * 64,
        "raw_result_sha256": "4" * 64,
        "passed": True,
    }
    monkeypatch.setattr(
        preformal,
        "fresh_runtime_development_closure_reopen",
        lambda path: dict(fresh_report),
    )
    monkeypatch.setattr(
        preformal,
        "validate_fresh_closure_reopen_report",
        lambda value, *, development_closure: value,
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
            phase="development_closure",
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
        sibling = operator_space.closure_root / "sibling-closure"
    shutil.copytree(canonical, sibling)

    with pytest.raises(OperatorError, match="canonical protocol-1.19"):
        operator_module.inspect_unfinalized_operator_attempt(sibling)

    sibling_terminal = sibling / "operator-attempt.json"
    sibling_completion = operator_module.outer_completion_marker(sibling_terminal)
    sibling_completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(sibling_terminal, sibling_completion)
    with pytest.raises(OperatorError, match="canonical protocol-1.19"):
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
        phase="binding",
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


def test_hidden_staging_claim_forbids_operator_retry(
    operator_space: OperatorSpace,
) -> None:
    final = operator_space.formal_binding
    final.parent.mkdir(parents=True, exist_ok=True)
    stranded = final.parent / f".{final.name}.staging"
    stranded.mkdir()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        operator_module._attempt_output(  # noqa: SLF001
            final,
            root=operator_space.binding_root,
            label="binding attempt",
        )
    assert not final.exists()
    assert stranded.is_dir()


def test_operator_staging_claim_is_immediately_fsynced(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = operator_space.formal_binding
    final.parent.mkdir(parents=True, exist_ok=True)
    fsynced: list[Path] = []
    monkeypatch.setattr(
        operator_module,
        "_fsync_directory",
        lambda path: fsynced.append(path),
    )

    attempt = operator_module._Attempt(  # noqa: SLF001
        final=final,
        kind="binding",
        lane=None,
        inputs=[],
    )

    assert fsynced == [final.parent]
    attempt.cleanup()


def test_operator_namespace_is_durably_created_from_absent_root(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsynced: list[Path] = []
    monkeypatch.setattr(
        operator_module,
        "_fsync_directory",
        lambda path: fsynced.append(path),
    )

    output = operator_module._attempt_output(  # noqa: SLF001
        operator_space.formal_binding,
        root=operator_space.binding_root,
        label="binding attempt",
    )

    results_root = operator_space.repo / "bench" / "world_model_lifecycle" / "results"
    assert output == operator_space.formal_binding
    assert fsynced == [
        results_root.parent,
        results_root,
        results_root / "operator-v1.19",
    ]
    assert operator_space.binding_root.is_dir()


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
            "LAZY_LEGACY_OP": "False",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
            "SDL_AUDIODRIVER": "dsp",
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
    with pytest.raises(OperatorError, match="canonical protocol-1.19 audit"):
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
    sibling = operator_space.development_qualification.parent / "qualification-v1.19.0-copy"
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
    if not passed:
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


def test_replay_stderr_mismatch_becomes_authenticated_failure(
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
        _execution(stderr=b"first stderr\n"),
        _execution(stderr=b"different stderr\n"),
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
    manifest = operator_module.inspect_unfinalized_operator_attempt(output)[
        "manifest"
    ]
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


def test_development_closure_then_binding_end_to_end(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import verify

    producer, audit_attempt = _run_development_audit(
        monkeypatch,
        operator_space,
    )
    marker, _, closure_calls = _install_closure_fakes(
        monkeypatch,
        operator_space,
    )
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
    assert len(closure_calls) == 1
    closure_manifest = operator_module.inspect_unfinalized_operator_attempt(operator_space.closure)["manifest"]
    assert closure_manifest["status"] == "accepted"
    _finalize(operator_space.closure)
    closure_terminal = operator_space.closure / "operator-attempt.json"
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
    assert operator_module.verify_operator_attempt(operator_space.closure) == closure_manifest

    report = preformal.PREFORMAL_REPORT_PATH
    report_value = _write_preformal_logs(report, command_count=10)
    monkeypatch.setattr(
        binding,
        "verify_canonical_machine_test_report",
        lambda _path: report_value,
    )
    log_rows = binding.preformal_log_rows(report, report_value)
    expected_binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "assurance": dict(ASSURANCE),
        "source": {
            "test_report_file": report.name,
            "test_report_bytes": report.stat().st_size,
            "test_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "test_log_files": log_rows,
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
    monkeypatch.setattr(
        verify,
        "_recorded_development_closure_identity",
        lambda path: (
            _load(path),
            {
                "closure_bytes": path.stat().st_size,
                "closure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            {},
        ),
    )

    def preflight_receipt(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "schema": "prospect.wm001.formal-input-preflight.v1",
            "binding_bytes": len(payload),
            "binding_sha256": hashlib.sha256(payload).hexdigest(),
            "passed": True,
        }

    monkeypatch.setattr(
        artifact_audit,
        "preflight_formal_input_package",
        preflight_receipt,
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
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda _path: pytest.fail("QA binding authorization must not replay live closure verification"),
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

    preflight_path = operator_space.formal_binding / "formal-input-preflight.json"
    original_preflight = preflight_path.read_bytes()
    preflight_path.write_bytes(b'{"passed":false}\n')
    with pytest.raises(
        OperatorError,
        match="files differ from the terminal manifest",
    ):
        operator_module.verify_operator_attempt(operator_space.formal_binding)
    preflight_path.write_bytes(original_preflight)
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


def test_existing_closure_marker_forbids_closure_resume(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, audit_attempt = _run_development_audit(
        monkeypatch,
        operator_space,
    )
    marker, build_closure, closure_calls = _install_closure_fakes(
        monkeypatch,
        operator_space,
    )
    build_closure(
        producer,
        audit_attempt / "independent-audit.json",
        audit_attempt / "audit-reproduction.json",
    )

    with pytest.raises(
        OperatorError,
        match="marker already consumes the one-shot closure",
    ):
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
    assert marker.is_file()
    assert closure_calls == []
    assert not operator_space.closure.exists()


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
    report = preformal.PREFORMAL_REPORT_PATH
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


def test_binding_rejects_log_manifest_schema_mismatch_before_attempt(
    operator_space: OperatorSpace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = preformal.PREFORMAL_REPORT_PATH
    report = _write_preformal_logs(report_path, command_count=9)
    _write(operator_space.development_closure, {"fixture": "closure"})
    operator_space.closure.mkdir(parents=True)
    monkeypatch.setattr(
        operator_module,
        "verify_operator_attempt",
        lambda _path: {
            "kind": "closure",
            "lane": "development",
            "status": "accepted",
            "primary": {"closure_reference_file": "closure-reference.json"},
        },
    )
    monkeypatch.setattr(
        binding,
        "verify_canonical_machine_test_report",
        lambda _path: report,
    )
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda _path: pytest.fail("closure verification must follow log-schema preclaim validation"),
    )

    with pytest.raises(
        RuntimeError,
        match="preclaim log manifest is incompatible with the formal binding schema",
    ):
        operator_module.binding_main(
            [
                "--output",
                str(operator_space.formal_binding),
                "--test-report",
                str(report_path),
                "--development-closure",
                str(operator_space.development_closure),
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
    report = preformal.PREFORMAL_REPORT_PATH
    report_value = _write_preformal_logs(report, command_count=10)
    monkeypatch.setattr(
        binding,
        "verify_canonical_machine_test_report",
        lambda _path: report_value,
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
