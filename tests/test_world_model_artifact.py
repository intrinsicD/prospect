from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench.world_model_lifecycle import artifact as artifact_module
from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle import experiment as experiment_module
from bench.world_model_lifecycle import operator as operator_module
from bench.world_model_lifecycle import producer_bootstrap as producer_bootstrap_module
from bench.world_model_lifecycle import run as run_module
from bench.world_model_lifecycle import verify as verify_module
from bench.world_model_lifecycle.artifact import (
    FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
    FORMAL_BINDING_OUTER_COMPLETION_NAME,
    FORMAL_CONFIRMATION_NAME,
    FORMAL_LAUNCH_MARKER_NAME,
    FORMAL_LAUNCH_NAME,
    MANIFEST_NAME,
    ProducerAttempt,
    _verify_producer_manifest_precommit,
    atomic_write_exclusive,
    claim_formal_launch,
    formal_launch_marker_path,
    sha256_file,
    verify_producer_manifest,
)
from bench.world_model_lifecycle.experiment import IsolatedConformanceReports


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


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _implementation_row(relative: str = "Makefile") -> dict[str, object]:
    path = binding_module.REPO / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": binding_module.sha256_file(path),
    }


def _write_preservable_binding(
    directory: Path,
    *,
    implementation_sha256: str | None = None,
) -> tuple[Path, set[str]]:
    directory.mkdir(exist_ok=True)

    def evidence(name: str, payload: bytes = b"evidence\n") -> Path:
        path = directory / name
        path.write_bytes(payload)
        return path

    test_report = evidence("preformal-report.json")
    test_logs = [evidence(f"preformal-{index:02d}.log") for index in range(20)]
    conformance = evidence("pendulum-conformance.json")
    oscillator = evidence("oscillator-conformance.json")
    coverage = evidence("coverage-conformance.json")
    closure = evidence("development-closure.json")
    audit_prefixes = (
        "bootstrap_source",
        "prebinding_request",
        "prebinding_path_runtime_manifest",
        "prebinding_descriptor_runtime_manifest",
        "prebinding_path_invocation_manifest",
        "prebinding_descriptor_invocation_manifest",
        "prebinding_conformance_report",
        "prebinding_execution_receipt",
        "outcome_runtime_manifest",
        "restart_runtime_conformance_report",
        "restart_runtime_execution_receipt",
    )
    audit_files = {prefix: evidence(f"{prefix}.evidence") for prefix in audit_prefixes}
    implementation = _implementation_row()
    if implementation_sha256 is not None:
        implementation["sha256"] = implementation_sha256
    value = {
        "source": {
            "implementation_files": [implementation],
            "test_report_file": test_report.name,
            "test_log_files": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in test_logs
            ],
        },
        "environment": {
            "conformance_report_file": conformance.name,
        },
        "irrelevant_control": {
            "conformance_report_file": oscillator.name,
        },
        "coverage_arithmetic": {
            "conformance_report_file": coverage.name,
        },
        "development_qualification": {
            "closure_file": closure.name,
        },
        "audit_execution": {f"{prefix}_file": path.name for prefix, path in audit_files.items()},
    }
    binding_path = directory / "formal-binding.json"
    binding_path.write_text(json.dumps(value), encoding="utf-8")
    return binding_path, {
        test_report.name,
        *(path.name for path in test_logs),
        conformance.name,
        oscillator.name,
        coverage.name,
        closure.name,
        *(path.name for path in audit_files.values()),
    }


def _write_minimal_formal_binding(path: Path, *, git_commit: str) -> str:
    value = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v9",
        "experiment_id": "WM-001",
        "assurance": {
            "trust_model_id": "prospect.wm001.trust-model.v1",
            "tamper_resistant": False,
            "external_attestation": False,
            "exclusive_path_use_required": True,
        },
        "protocol": {"version": "1.10.0"},
        "source": {
            "git_commit": git_commit,
            "git_tree": "2" * 40,
        },
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _install_binding_attempt_stub(
    monkeypatch: pytest.MonkeyPatch,
    binding_path: Path,
) -> bytes:
    attempt = binding_path.parent
    terminal = attempt / "operator-attempt.json"
    completion_root = attempt / "outer-completions"
    completion_root.mkdir()
    monkeypatch.setattr(
        operator_module,
        "OUTER_COMPLETIONS_ROOT",
        completion_root,
    )
    completion = operator_module.outer_completion_marker(terminal)
    terminal_payload = b'{"kind":"binding","status":"accepted"}\n'
    terminal.write_bytes(terminal_payload)
    completion.hardlink_to(terminal)
    (attempt / artifact_module.FORMAL_INPUT_PREFLIGHT_NAME).write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.formal-input-preflight.v1",
                "passed": True,
            }
        )
    )
    monkeypatch.setattr(
        operator_module,
        "FORMAL_BINDING_ATTEMPT_PATH",
        attempt,
    )
    monkeypatch.setattr(
        operator_module,
        "verify_operator_attempt",
        lambda candidate: (
            {
                "kind": "binding",
                "lane": None,
                "status": "accepted",
                "primary": {"binding_file": "formal-binding.json"},
            }
            if candidate == attempt
            else (_ for _ in ()).throw(ValueError("wrong binding attempt"))
        ),
    )

    def evidence(payload: bytes) -> tuple[Path, bytes, str, Path, str]:
        if payload != binding_path.read_bytes():
            raise ValueError("formal binding differs from the accepted canonical binding attempt")
        digest = hashlib.sha256(terminal_payload).hexdigest()
        return attempt, terminal_payload, digest, completion, digest

    monkeypatch.setattr(
        artifact_module,
        "_formal_binding_attempt_evidence",
        evidence,
    )
    return terminal_payload


def _preserve_stubbed_binding_attempt(
    output: Path,
    payload: bytes,
) -> None:
    (output / FORMAL_BINDING_ATTEMPT_MANIFEST_NAME).write_bytes(payload)
    (output / FORMAL_BINDING_OUTER_COMPLETION_NAME).write_bytes(payload)


@contextmanager
def _formal_claim_authorization(binding_path: Path):
    bootstrap_path = binding_path.with_name(f".{binding_path.name}.bootstrap.py")
    bootstrap_path.write_bytes(b"# captured bootstrap\n")
    runtime_descriptor = os.open(binding_path, os.O_RDONLY)
    bootstrap_descriptor = os.open(bootstrap_path, os.O_RDONLY)
    runtime_metadata = os.fstat(runtime_descriptor)
    identity = (
        runtime_metadata.st_dev,
        runtime_metadata.st_ino,
        runtime_metadata.st_mode,
        runtime_metadata.st_nlink,
        runtime_metadata.st_uid,
        runtime_metadata.st_gid,
        runtime_metadata.st_size,
        runtime_metadata.st_mtime_ns,
        runtime_metadata.st_ctime_ns,
    )
    names = (
        "_prospect_wm001_runtime_seal_fd",
        "_prospect_wm001_runtime_seal_identity",
        "_prospect_wm001_bootstrap_fd",
    )
    missing = object()
    previous = {name: getattr(sys, name, missing) for name in names}
    setattr(sys, names[0], runtime_descriptor)
    setattr(sys, names[1], identity)
    setattr(sys, names[2], bootstrap_descriptor)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                delattr(sys, name)
            else:
                setattr(sys, name, value)
        os.close(bootstrap_descriptor)
        os.close(runtime_descriptor)


def _isolated_conformance_reports(
    *,
    failed: str | None = None,
) -> IsolatedConformanceReports:
    return IsolatedConformanceReports(
        pendulum_conformance={"passed": failed != "pendulum", "cases": 256},
        oscillator_conformance={
            "passed": failed != "oscillator",
            "cases": 32,
            "trajectory_sha256": "0" * 64,
        },
        coverage_conformance={"passed": failed != "coverage", "cases": 1024},
        runner_verification={
            "passed": failed != "runner",
            "source_mode": "descriptor",
            "single_launch_replay": True,
            "matches_bound_prebinding_reports": True,
        },
    )


def test_protocol_wide_formal_launch_claim_is_atomic_across_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results" / "formal"
    monkeypatch.setattr(
        artifact_module,
        "FORMAL_RESULTS_ROOT",
        results_root,
    )
    first_binding = tmp_path / "binding-attempt" / "formal-binding.json"
    first_binding.parent.mkdir()
    first_digest = _write_minimal_formal_binding(first_binding, git_commit="1" * 40)
    attempt_payload = _install_binding_attempt_stub(monkeypatch, first_binding)
    first_output = results_root / first_digest / FORMAL_CONFIRMATION_NAME
    first_output.mkdir(parents=True)
    _preserve_stubbed_binding_attempt(first_output, attempt_payload)
    retired_v14_marker = results_root / FORMAL_LAUNCH_NAME
    retired_v14_marker.write_bytes(b"retired-v1.4-marker\n")

    with _formal_claim_authorization(first_binding):
        marker = claim_formal_launch(
            first_binding,
            first_output,
            formal_results_root=results_root,
        )
    assert marker == results_root / FORMAL_LAUNCH_MARKER_NAME
    producer_record = first_output / FORMAL_LAUNCH_NAME
    copied = _read_json(producer_record)
    assert copied["formal_binding_sha256"] == first_digest
    assert copied["attempt_directory"] == FORMAL_CONFIRMATION_NAME
    assert copied["schema"] == "prospect.wm001.formal-launch.v2"
    assert copied["protocol_version"] == "1.10.0"
    assert copied["global_marker_file"] == FORMAL_LAUNCH_MARKER_NAME
    assert marker.read_bytes() == producer_record.read_bytes()
    assert os.path.samefile(marker, producer_record)
    assert marker.stat().st_ino == producer_record.stat().st_ino
    assert marker.stat().st_dev == producer_record.stat().st_dev
    assert retired_v14_marker.read_bytes() == b"retired-v1.4-marker\n"
    verify_module._verify_formal_launch_record(
        producer_record,
        binding_sha256=first_digest,
        execution={
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
        },
    )
    wrong_output = results_root / first_digest / "wrong-child"
    wrong_output.mkdir()
    _preserve_stubbed_binding_attempt(wrong_output, attempt_payload)
    wrong_record = dict(copied)
    wrong_record["attempt_directory"] = wrong_output.name
    wrong_body = dict(wrong_record)
    wrong_body.pop("record_sha256")
    wrong_record["record_sha256"] = hashlib.sha256(
        json.dumps(
            wrong_body,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (wrong_output / FORMAL_LAUNCH_NAME).write_bytes(
        json.dumps(
            wrong_record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(
        verify_module.Violation,
        match="formal launch record identity",
    ):
        verify_module._verify_formal_launch_record(
            wrong_output / FORMAL_LAUNCH_NAME,
            binding_sha256=first_digest,
            execution={
                "git_commit": "1" * 40,
                "git_tree": "2" * 40,
            },
        )
    copied_completion = first_output / FORMAL_BINDING_OUTER_COMPLETION_NAME
    copied_completion.write_bytes(b"tampered\n")
    with pytest.raises(
        verify_module.Violation,
        match="formal launch record identity",
    ):
        verify_module._verify_formal_launch_record(
            producer_record,
            binding_sha256=first_digest,
            execution={
                "git_commit": "1" * 40,
                "git_tree": "2" * 40,
            },
        )
    copied_completion.write_bytes(attempt_payload)

    retry_output = first_output
    with _formal_claim_authorization(first_binding):
        with pytest.raises(RuntimeError, match="already has a formal launch claim"):
            claim_formal_launch(
                first_binding,
                retry_output,
                formal_results_root=results_root,
            )

    second_binding = tmp_path / "binding-b" / "formal-binding.json"
    second_binding.parent.mkdir()
    second_digest = _write_minimal_formal_binding(second_binding, git_commit="3" * 40)
    second_output = results_root / second_digest / FORMAL_CONFIRMATION_NAME
    second_output.mkdir(parents=True)
    second_payload = _install_binding_attempt_stub(monkeypatch, second_binding)
    _preserve_stubbed_binding_attempt(second_output, second_payload)
    with _formal_claim_authorization(second_binding):
        with pytest.raises(RuntimeError, match="already has a formal launch claim"):
            claim_formal_launch(
                second_binding,
                second_output,
                formal_results_root=results_root,
            )


def test_formal_launch_claim_rejects_noncanonical_output(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    binding_digest = _write_minimal_formal_binding(
        binding,
        git_commit="1" * 40,
    )
    results_root = tmp_path / "results" / "formal"

    invalid_outputs = (
        tmp_path / "arbitrary-output",
        results_root / binding_digest / "attempt-a",
        results_root / ("0" * 64) / FORMAL_CONFIRMATION_NAME,
    )
    for output in invalid_outputs:
        with pytest.raises(
            ValueError,
            match="results/formal/<binding-sha256>/confirmation-v1.10.0",
        ):
            formal_launch_marker_path(
                binding,
                output,
                formal_results_root=results_root,
            )
    assert not results_root.exists()


def test_formal_launch_path_validation_creates_no_directories(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    binding_digest = _write_minimal_formal_binding(binding, git_commit="1" * 40)
    results_root = tmp_path / "not-created" / "results" / "formal"
    output = results_root / binding_digest / FORMAL_CONFIRMATION_NAME

    marker, actual_digest = formal_launch_marker_path(
        binding,
        output,
        formal_results_root=results_root,
    )

    assert actual_digest == binding_digest
    assert marker == results_root / FORMAL_LAUNCH_MARKER_NAME
    assert not results_root.exists()
    assert not output.exists()
    with pytest.raises(ValueError, match="aliased"):
        formal_launch_marker_path(
            binding,
            results_root
            / binding_digest
            / "unused"
            / ".."
            / FORMAL_CONFIRMATION_NAME,
            formal_results_root=results_root,
        )


def test_formal_launch_path_validation_rejects_binding_and_output_aliases(
    tmp_path: Path,
) -> None:
    binding = tmp_path / "binding.json"
    binding_digest = _write_minimal_formal_binding(binding, git_commit="1" * 40)
    binding_alias = tmp_path / "binding-alias.json"
    binding_alias.symlink_to(binding)
    results_root = tmp_path / "results" / "formal"
    canonical_binding_root = results_root / binding_digest
    aliased_target = tmp_path / "aliased-binding-root"
    aliased_target.mkdir()
    results_root.mkdir(parents=True)
    canonical_binding_root.symlink_to(aliased_target, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symbolic-link"):
        formal_launch_marker_path(
            binding_alias,
            results_root / binding_digest / FORMAL_CONFIRMATION_NAME,
            formal_results_root=results_root,
        )
    with pytest.raises(ValueError, match="aliased"):
        formal_launch_marker_path(
            binding,
            results_root / binding_digest / FORMAL_CONFIRMATION_NAME,
            formal_results_root=results_root,
        )


def test_formal_launch_prepublication_failure_leaves_marker_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = tmp_path / "binding-attempt" / "formal-binding.json"
    binding.parent.mkdir()
    binding_digest = _write_minimal_formal_binding(binding, git_commit="1" * 40)
    attempt_payload = _install_binding_attempt_stub(monkeypatch, binding)
    results_root = tmp_path / "results" / "formal"
    output = results_root / binding_digest / FORMAL_CONFIRMATION_NAME
    output.mkdir(parents=True)
    _preserve_stubbed_binding_attempt(output, attempt_payload)
    marker = results_root / FORMAL_LAUNCH_MARKER_NAME
    original_atomic_write = artifact_module.atomic_write_exclusive

    def fail_producer_record(path: Path, payload: bytes) -> None:
        if path == output / FORMAL_LAUNCH_NAME:
            raise OSError("injected producer-record failure")
        original_atomic_write(path, payload)

    monkeypatch.setattr(
        artifact_module,
        "atomic_write_exclusive",
        fail_producer_record,
    )

    with _formal_claim_authorization(binding):
        with pytest.raises(OSError, match="injected producer-record failure"):
            claim_formal_launch(
                binding,
                output,
                formal_results_root=results_root,
            )
    assert not marker.exists()
    assert not (output / FORMAL_LAUNCH_NAME).exists()


def test_formal_launch_link_failure_preserves_unclaimed_producer_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = tmp_path / "binding-attempt" / "formal-binding.json"
    binding.parent.mkdir()
    binding_digest = _write_minimal_formal_binding(binding, git_commit="1" * 40)
    attempt_payload = _install_binding_attempt_stub(monkeypatch, binding)
    results_root = tmp_path / "results" / "formal"
    output = results_root / binding_digest / FORMAL_CONFIRMATION_NAME
    output.mkdir(parents=True)
    _preserve_stubbed_binding_attempt(output, attempt_payload)
    marker = results_root / FORMAL_LAUNCH_MARKER_NAME
    original_link = artifact_module.os.link

    def fail_global_link(source: Path, destination: Path) -> None:
        if Path(destination) == marker:
            raise OSError("injected global-link failure")
        original_link(source, destination)

    monkeypatch.setattr(artifact_module.os, "link", fail_global_link)

    with _formal_claim_authorization(binding):
        with pytest.raises(OSError, match="injected global-link failure"):
            claim_formal_launch(
                binding,
                output,
                formal_results_root=results_root,
            )
    assert not marker.exists()
    assert (output / FORMAL_LAUNCH_NAME).is_file()


def test_atomic_write_exclusive_never_replaces_evidence(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"

    atomic_write_exclusive(path, b"first")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        atomic_write_exclusive(path, b"second")
    assert path.read_bytes() == b"first"
    assert not list(tmp_path.glob(".*.tmp"))


def test_completed_attempt_tees_logs_and_manifests_every_file(tmp_path: Path) -> None:
    output = tmp_path / "attempt"

    with ProducerAttempt(output, lane="development"):
        print("durable stdout")
        atomic_write_exclusive(output / "evidence.bin", b"evidence")

    manifest = _read_json(output / MANIFEST_NAME)
    assert manifest["status"] == "completed"
    assert manifest["error"] is None
    rows = {
        str(row["path"]): row
        for row in manifest["files"]  # type: ignore[index,union-attr]
    }
    assert MANIFEST_NAME not in rows
    assert "durable stdout" in (output / "main.stdout.log").read_text(encoding="utf-8")
    assert rows["evidence.bin"]["sha256"] == sha256_file(output / "evidence.bin")
    assert manifest["file_count"] == len(rows)
    assert _verify_producer_manifest_precommit(output) == manifest
    with pytest.raises(ValueError, match="outer-completion"):
        verify_producer_manifest(output)

    (output / "evidence.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest changed"):
        _verify_producer_manifest_precommit(output)

    with pytest.raises(FileExistsError):
        with ProducerAttempt(output, lane="development"):
            pass


def test_public_producer_verifier_requires_same_inode_outer_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "attempt"
    with ProducerAttempt(output, lane="development"):
        atomic_write_exclusive(output / "evidence.bin", b"evidence")

    completion_root = tmp_path / "outer-completions"
    completion_root.mkdir()
    monkeypatch.setattr(
        operator_module,
        "OUTER_COMPLETIONS_ROOT",
        completion_root,
    )
    terminal = output / MANIFEST_NAME
    marker = operator_module.outer_completion_marker(terminal)
    marker.hardlink_to(terminal)

    assert verify_producer_manifest(output)["status"] == "completed"
    marker.unlink()
    with pytest.raises(ValueError, match="outer-completion"):
        verify_producer_manifest(output)


def test_failed_attempt_retains_partial_evidence_and_traceback(tmp_path: Path) -> None:
    output = tmp_path / "failed"

    with pytest.raises(RuntimeError, match="deliberate failure"):
        with ProducerAttempt(output, lane="formal"):
            atomic_write_exclusive(output / "partial.bin", b"partial")
            raise RuntimeError("deliberate failure")

    manifest = _read_json(output / MANIFEST_NAME)
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "message": "deliberate failure",
        "type": "RuntimeError",
    }
    assert (output / "partial.bin").read_bytes() == b"partial"
    stderr = (output / "main.stderr.log").read_text(encoding="utf-8")
    assert "RuntimeError: deliberate failure" in stderr


def test_formal_attempt_preserves_all_pre_outcome_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_directory = tmp_path / "binding"
    binding_path, evidence_names = _write_preservable_binding(binding_directory)
    _install_binding_attempt_stub(monkeypatch, binding_path)
    output = tmp_path / "attempt"

    with ProducerAttempt(output, lane="formal") as attempt:
        copied_binding = attempt.preserve_formal_inputs(binding_path)

    assert copied_binding == output / "formal-binding.json"
    expected = {
        "formal-binding.json",
        FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
        FORMAL_BINDING_OUTER_COMPLETION_NAME,
        artifact_module.FORMAL_INPUT_PREFLIGHT_NAME,
        "protocol.json",
        "SEALED_PROTOCOL.sha256",
        "schemas/formal-binding.schema.json",
        "schemas/raw-result.schema.json",
        "requirements-wm001.lock",
        *evidence_names,
        "source/Makefile",
    }
    assert expected <= {
        candidate.relative_to(output).as_posix() for candidate in output.rglob("*") if candidate.is_file()
    }


def test_formal_cli_verifies_live_copied_binding_before_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_directory = tmp_path / "binding"
    binding_path, _ = _write_preservable_binding(binding_directory)
    _install_binding_attempt_stub(monkeypatch, binding_path)
    output = tmp_path / "formal-attempt"
    calls: list[tuple[str, Path]] = []
    launch_marker = tmp_path / FORMAL_LAUNCH_MARKER_NAME

    def verify_live(path: Path, *, device: str) -> dict[str, object]:
        calls.append((device, path))
        assert path == output / "formal-binding.json"
        assert (output / "protocol.json").is_file()
        return _read_json(path)

    def run_experiment(
        config: object,
        *,
        output_directory: Path,
        formal_binding_path: Path | None,
        output_prepared: bool,
    ) -> tuple[dict[str, object], Path]:
        assert formal_binding_path is not None
        verify_live(
            formal_binding_path,
            device=str(config.device),
        )
        assert calls
        assert output_prepared is True
        assert formal_binding_path == output / "formal-binding.json"
        claim_launch(formal_binding_path, output_directory)
        result_path = output_directory / "result.json"
        atomic_write_exclusive(result_path, b"{}\n")
        return {}, result_path

    def claim_launch(_binding: Path, _output: Path) -> Path:
        assert _binding == output / "formal-binding.json"
        assert _output == output
        producer_record = output / FORMAL_LAUNCH_NAME
        atomic_write_exclusive(producer_record, b"{}\n")
        os.link(producer_record, launch_marker)
        return launch_marker

    monkeypatch.setattr(binding_module, "verify_live_binding", verify_live)
    monkeypatch.setattr(
        artifact_module,
        "formal_launch_marker_path",
        lambda _binding, _output, *, formal_results_root: (
            launch_marker,
            "0" * 64,
        ),
    )
    monkeypatch.setattr(artifact_module, "claim_formal_launch", claim_launch)
    monkeypatch.setattr(run_module, "_ensure_deterministic_cuda_environment", lambda: None)
    monkeypatch.setattr(
        experiment_module.ExperimentConfig,
        "formal",
        staticmethod(
            lambda *, device=None: SimpleNamespace(
                device=device or "cpu",
                validate=lambda: None,
            )
        ),
    )
    monkeypatch.setattr(experiment_module, "run_experiment", run_experiment)
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: {},
    )
    registered: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        producer_bootstrap_module,
        "register_outer_terminal",
        lambda path, *, logical_exit_code: registered.append((path, logical_exit_code)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "wm001",
            "formal",
            "--binding",
            str(binding_path),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    try:
        assert run_module.main() == 0
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
    assert calls == [("cpu", output / "formal-binding.json")]
    manifest = _read_json(output / MANIFEST_NAME)
    assert manifest["status"] == "completed"
    assert (output / "result.json").is_file()
    assert (output / FORMAL_LAUNCH_NAME).read_bytes() == b"{}\n"
    assert os.path.samefile(output / FORMAL_LAUNCH_NAME, launch_marker)
    assert registered == [(output / MANIFEST_NAME, 0)]


@pytest.mark.parametrize("output_kind", ("omitted", "wrong-child"))
def test_formal_cli_refuses_noncanonical_or_omitted_output_before_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: str,
) -> None:
    binding_path = (
        tmp_path
        / "binding-attempt"
        / "formal-binding.json"
    )
    binding_path.parent.mkdir()
    binding_digest = _write_minimal_formal_binding(
        binding_path,
        git_commit="1" * 40,
    )
    _install_binding_attempt_stub(monkeypatch, binding_path)
    results_root = tmp_path / "results" / "formal"
    wrong_output = results_root / binding_digest / "wrong-child"

    monkeypatch.setattr(
        artifact_module,
        "FORMAL_RESULTS_ROOT",
        results_root,
    )
    monkeypatch.setattr(
        run_module,
        "_ensure_deterministic_cuda_environment",
        lambda: None,
    )
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: (_ for _ in ()).throw(
            AssertionError("formal output refusal must precede runtime custody")
        ),
    )
    monkeypatch.setattr(
        artifact_module,
        "ProducerAttempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("formal output refusal must precede producer custody")
        ),
    )
    argv = [
        "wm001",
        "formal",
        "--binding",
        str(binding_path),
        "--device",
        "cpu",
    ]
    if output_kind == "wrong-child":
        argv.extend(("--output", str(wrong_output)))
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as raised:
        run_module.main()

    assert raised.value.code == 2
    assert not results_root.exists()


@pytest.mark.parametrize(
    "failure_stage",
    (
        "preserve_inputs",
        "formal_preflight",
    ),
)
def test_formal_cli_preclaim_failures_leave_v110_marker_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    binding_path = tmp_path / "binding-attempt" / "formal-binding.json"
    binding_path.parent.mkdir()
    _write_minimal_formal_binding(binding_path, git_commit="1" * 40)
    _install_binding_attempt_stub(monkeypatch, binding_path)
    output = tmp_path / "attempt"
    launch_marker = tmp_path / FORMAL_LAUNCH_MARKER_NAME
    retired_v14_marker = tmp_path / FORMAL_LAUNCH_NAME
    retired_v14_marker.write_bytes(b"retired-v1.4-marker\n")
    claim_calls: list[Path] = []

    def preserve_inputs(self: ProducerAttempt, source: Path) -> Path:
        if failure_stage == "preserve_inputs":
            raise RuntimeError("injected input-preservation failure")
        destination = self.output_directory / "formal-binding.json"
        atomic_write_exclusive(destination, source.read_bytes())
        return destination

    def claim_launch(_binding: Path, _output: Path) -> Path:
        claim_calls.append(_output)
        atomic_write_exclusive(launch_marker, b"should-not-be-published\n")
        return launch_marker

    monkeypatch.setattr(
        artifact_module,
        "formal_launch_marker_path",
        lambda _binding, _output, *, formal_results_root: (
            launch_marker,
            "0" * 64,
        ),
    )
    monkeypatch.setattr(artifact_module, "claim_formal_launch", claim_launch)
    monkeypatch.setattr(ProducerAttempt, "preserve_formal_inputs", preserve_inputs)
    monkeypatch.setattr(run_module, "_ensure_deterministic_cuda_environment", lambda: None)
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: {},
    )
    monkeypatch.setattr(
        experiment_module.ExperimentConfig,
        "formal",
        staticmethod(
            lambda *, device=None: SimpleNamespace(
                device=device or "cpu",
                validate=lambda: None,
            )
        ),
    )

    def fail_preflight(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected formal preflight failure")

    monkeypatch.setattr(experiment_module, "run_experiment", fail_preflight)
    registered: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        producer_bootstrap_module,
        "register_outer_terminal",
        lambda path, *, logical_exit_code: registered.append((path, logical_exit_code)),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "wm001",
            "formal",
            "--binding",
            str(binding_path),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    try:
        assert run_module.main() == 1
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
    assert claim_calls == []
    assert not launch_marker.exists()
    assert retired_v14_marker.read_bytes() == b"retired-v1.4-marker\n"
    assert registered == [(output / MANIFEST_NAME, 1)]


def test_formal_experiment_requires_launch_claim_before_any_replicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prepared-formal-attempt"
    output.mkdir()
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / FORMAL_LAUNCH_NAME).write_bytes(b"retired-v1.4-marker\n")
    atomic_write_exclusive(
        output / "attempt-metadata.json",
        json.dumps(
            {
                "schema": "prospect.wm001.producer-attempt.v1",
                "lane": "formal",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n",
    )
    replicate_entered = False

    def run_replicate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal replicate_entered
        replicate_entered = True
        raise AssertionError("formal replicate must not start without the launch claim")

    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: {
            "runtime_seal": {"schema": "prospect.world-model-lifecycle.formal-binding.v9"},
            "runtime_seal_payload": binding_path.read_bytes(),
        },
    )
    monkeypatch.setattr(
        experiment_module,
        "_run_formal_preflight",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        experiment_module,
        "_validate_formal_preflight_reports",
        lambda *_args, **_kwargs: (
            {"passed": True},
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        artifact_module,
        "claim_formal_launch_with_digest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no unique protocol-wide launch claim")),
    )
    monkeypatch.setattr(experiment_module, "run_replicate", run_replicate)

    with pytest.raises(RuntimeError, match="no unique protocol-wide launch claim"):
        experiment_module.run_experiment(
            experiment_module.ExperimentConfig.formal(device="cpu"),
            output_directory=output,
            formal_binding_path=binding_path,
            output_prepared=True,
        )
    assert replicate_entered is False
    assert (tmp_path / FORMAL_LAUNCH_NAME).read_bytes() == b"retired-v1.4-marker\n"


def test_formal_experiment_consumes_preclaim_reports_without_rerunning_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "prepared-formal-attempt"
    output.mkdir()
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_text("{}\n", encoding="utf-8")
    atomic_write_exclusive(
        output / "attempt-metadata.json",
        json.dumps(
            {
                "schema": "prospect.wm001.producer-attempt.v1",
                "lane": "formal",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n",
    )
    producer_record = output / FORMAL_LAUNCH_NAME
    marker = tmp_path / FORMAL_LAUNCH_MARKER_NAME
    atomic_write_exclusive(producer_record, b"{}\n")
    os.link(producer_record, marker)

    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: {
            "runtime_seal": {"schema": "prospect.world-model-lifecycle.formal-binding.v9"},
            "runtime_seal_payload": binding_path.read_bytes(),
        },
    )
    reports = _isolated_conformance_reports()
    monkeypatch.setattr(
        experiment_module,
        "_run_formal_preflight",
        lambda *_args, **_kwargs: reports,
    )
    monkeypatch.setattr(
        experiment_module,
        "_validate_formal_preflight_reports",
        lambda supplied, **_kwargs: (
            supplied.pendulum_conformance,
            supplied.oscillator_conformance,
        ),
    )
    monkeypatch.setattr(
        artifact_module,
        "claim_formal_launch_with_digest",
        lambda *_args, **_kwargs: (marker, "0" * 64),
    )
    monkeypatch.setattr(
        experiment_module,
        "run_pendulum_conformance",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("formal Pendulum preflight was rerun after claim")),
    )
    monkeypatch.setattr(
        experiment_module,
        "run_independent_phase_oscillator_conformance",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("formal oscillator preflight was rerun after claim")),
    )
    monkeypatch.setattr(
        experiment_module,
        "run_replicate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("formal replicate reached")),
    )

    with pytest.raises(RuntimeError, match="formal replicate reached"):
        experiment_module.run_experiment(
            experiment_module.ExperimentConfig.formal(device="cpu"),
            output_directory=output,
            formal_binding_path=binding_path,
            output_prepared=True,
        )


def test_formal_input_source_snapshot_rejects_bound_digest_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_directory = tmp_path / "binding"
    binding_path, _ = _write_preservable_binding(
        binding_directory,
        implementation_sha256="0" * 64,
    )
    _install_binding_attempt_stub(monkeypatch, binding_path)

    with ProducerAttempt(tmp_path / "attempt", lane="formal") as attempt:
        with pytest.raises(ValueError, match="digest changed"):
            attempt.preserve_formal_inputs(binding_path)


def test_experiment_entrypoint_refuses_unowned_or_existing_output(
    tmp_path: Path,
) -> None:
    formal = experiment_module.ExperimentConfig.formal(device="cpu")
    with pytest.raises(RuntimeError, match="sealed producer bootstrap"):
        experiment_module.run_experiment(
            formal,
            output_directory=tmp_path / "formal",
            formal_binding_path=tmp_path / "missing-binding.json",
        )

    development = experiment_module.ExperimentConfig.development(
        master_seeds=(1647437737,),
        device="cpu",
    )
    assert (
        development.collection_episodes,
        development.validation_episodes,
        development.behavior_episodes,
        development.optimizer_steps,
    ) == (8, 8, 32, 2000)
    with pytest.raises(RuntimeError, match="sealed producer bootstrap"):
        experiment_module.run_experiment(
            development,
            output_directory=tmp_path,
        )
