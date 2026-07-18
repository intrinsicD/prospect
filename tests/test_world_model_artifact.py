from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench.world_model_lifecycle import artifact as artifact_module
from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle import experiment as experiment_module
from bench.world_model_lifecycle import run as run_module
from bench.world_model_lifecycle.artifact import (
    MANIFEST_NAME,
    ProducerAttempt,
    atomic_write_exclusive,
    claim_formal_launch,
    copy_file_exclusive,
    formal_launch_marker_path,
    sha256_file,
    verify_producer_manifest,
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


def _write_minimal_formal_binding(path: Path, *, git_commit: str) -> str:
    value = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v4",
        "experiment_id": "WM-001",
        "protocol": {"version": "1.4.0"},
        "source": {
            "git_commit": git_commit,
            "git_tree": "2" * 40,
        },
    }
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_protocol_wide_formal_launch_claim_is_atomic_across_bindings(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results" / "formal"
    first_binding = tmp_path / "binding-a.json"
    first_digest = _write_minimal_formal_binding(first_binding, git_commit="1" * 40)
    first_output = results_root / first_digest / "attempt-a"

    marker = claim_formal_launch(
        first_binding,
        first_output,
        formal_results_root=results_root,
    )
    assert marker == results_root / "formal-launch.json"
    first_output.mkdir()
    copy_file_exclusive(marker, first_output / "formal-launch.json")
    copied = _read_json(first_output / "formal-launch.json")
    assert copied["formal_binding_sha256"] == first_digest
    assert copied["attempt_directory"] == "attempt-a"

    with pytest.raises(RuntimeError, match="already has a formal launch claim"):
        claim_formal_launch(
            first_binding,
            results_root / first_digest / "attempt-b",
            formal_results_root=results_root,
        )

    second_binding = tmp_path / "binding-b.json"
    second_digest = _write_minimal_formal_binding(second_binding, git_commit="3" * 40)
    with pytest.raises(RuntimeError, match="already has a formal launch claim"):
        claim_formal_launch(
            second_binding,
            results_root / second_digest / "attempt-c",
            formal_results_root=results_root,
        )


def test_formal_launch_claim_rejects_noncanonical_output(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    _write_minimal_formal_binding(binding, git_commit="1" * 40)

    with pytest.raises(ValueError, match="results/formal"):
        formal_launch_marker_path(
            binding,
            tmp_path / "arbitrary-output",
            formal_results_root=tmp_path / "results" / "formal",
        )


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
    assert verify_producer_manifest(output) == manifest

    (output / "evidence.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest changed"):
        verify_producer_manifest(output)

    with pytest.raises(FileExistsError):
        with ProducerAttempt(output, lane="development"):
            pass


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


def test_formal_attempt_preserves_all_pre_outcome_inputs(tmp_path: Path) -> None:
    binding_directory = tmp_path / "binding"
    binding_directory.mkdir()
    test_report = binding_directory / "tests.txt"
    conformance = binding_directory / "conformance.json"
    oscillator_conformance = binding_directory / "oscillator-conformance.json"
    coverage_conformance = binding_directory / "coverage-conformance.json"
    test_report.write_text("tests passed\n", encoding="utf-8")
    conformance.write_text('{"passed":true}\n', encoding="utf-8")
    oscillator_conformance.write_text('{"passed":true}\n', encoding="utf-8")
    coverage_conformance.write_text('{"passed":true}\n', encoding="utf-8")
    binding_path = binding_directory / "binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "source": {
                    "implementation_files": [_implementation_row()],
                    "test_report_file": test_report.name,
                },
                "environment": {"conformance_report_file": conformance.name},
                "irrelevant_control": {
                    "conformance_report_file": oscillator_conformance.name,
                },
                "coverage_arithmetic": {
                    "conformance_report_file": coverage_conformance.name,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "attempt"

    with ProducerAttempt(output, lane="formal") as attempt:
        copied_binding = attempt.preserve_formal_inputs(binding_path)

    assert copied_binding == output / "formal-binding.json"
    expected = {
        "formal-binding.json",
        "protocol.json",
        "SEALED_PROTOCOL.sha256",
        "schemas/formal-binding.schema.json",
        "schemas/raw-result.schema.json",
        "requirements-wm001.lock",
        test_report.name,
        conformance.name,
        oscillator_conformance.name,
        coverage_conformance.name,
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
    binding_directory.mkdir()
    (binding_directory / "tests.txt").write_text("passed\n", encoding="utf-8")
    (binding_directory / "conformance.json").write_text(
        '{"passed":true}\n',
        encoding="utf-8",
    )
    (binding_directory / "oscillator-conformance.json").write_text(
        '{"passed":true}\n',
        encoding="utf-8",
    )
    (binding_directory / "coverage-conformance.json").write_text(
        '{"passed":true}\n',
        encoding="utf-8",
    )
    binding_path = binding_directory / "binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "source": {
                    "implementation_files": [_implementation_row()],
                    "test_report_file": "tests.txt",
                },
                "environment": {"conformance_report_file": "conformance.json"},
                "irrelevant_control": {
                    "conformance_report_file": "oscillator-conformance.json",
                },
                "coverage_arithmetic": {
                    "conformance_report_file": "coverage-conformance.json",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "formal-attempt"
    calls: list[tuple[str, Path]] = []
    launch_marker = tmp_path / "formal-launch.json"

    def verify_live(path: Path, *, device: str) -> dict[str, object]:
        calls.append((device, path))
        assert path == output / "formal-binding.json"
        assert (output / "protocol.json").is_file()
        return {}

    def run_experiment(
        config: object,
        *,
        output_directory: Path,
        formal_binding_path: Path | None,
        output_prepared: bool,
    ) -> tuple[dict[str, object], Path]:
        assert calls
        assert output_prepared is True
        assert formal_binding_path == output / "formal-binding.json"
        result_path = output_directory / "result.json"
        atomic_write_exclusive(result_path, b"{}\n")
        return {}, result_path

    def claim_launch(_binding: Path, _output: Path) -> Path:
        assert _binding == binding_path
        assert _output == output
        atomic_write_exclusive(launch_marker, b"{}\n")
        return launch_marker

    monkeypatch.setattr(binding_module, "verify_live_binding", verify_live)
    monkeypatch.setattr(
        artifact_module,
        "formal_launch_marker_path",
        lambda _binding, _output: (launch_marker, "0" * 64),
    )
    monkeypatch.setattr(artifact_module, "claim_formal_launch", claim_launch)
    monkeypatch.setattr(run_module, "_ensure_deterministic_cuda_environment", lambda: None)
    monkeypatch.setattr(
        experiment_module.ExperimentConfig,
        "formal",
        staticmethod(lambda *, device=None: SimpleNamespace(device=device or "cpu")),
    )
    monkeypatch.setattr(experiment_module, "run_experiment", run_experiment)
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
    assert (output / "formal-launch.json").read_bytes() == b"{}\n"


def test_formal_experiment_requires_launch_claim_before_any_replicate(
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
    replicate_entered = False

    def run_replicate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal replicate_entered
        replicate_entered = True
        raise AssertionError("formal replicate must not start without the launch claim")

    monkeypatch.setattr(binding_module, "verify_live_binding", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        experiment_module,
        "formal_launch_marker_path",
        lambda _binding, _output: (tmp_path / "missing-formal-launch.json", "0" * 64),
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


def test_formal_input_source_snapshot_rejects_bound_digest_change(
    tmp_path: Path,
) -> None:
    binding_directory = tmp_path / "binding"
    binding_directory.mkdir()
    (binding_directory / "tests.txt").write_text("passed\n", encoding="utf-8")
    (binding_directory / "conformance.json").write_text(
        '{"passed":true}\n',
        encoding="utf-8",
    )
    (binding_directory / "oscillator-conformance.json").write_text(
        '{"passed":true}\n',
        encoding="utf-8",
    )
    (binding_directory / "coverage-conformance.json").write_text(
        '{"passed":true}\n',
        encoding="utf-8",
    )
    row = _implementation_row()
    row["sha256"] = "0" * 64
    binding_path = binding_directory / "binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "source": {
                    "implementation_files": [row],
                    "test_report_file": "tests.txt",
                },
                "environment": {
                    "conformance_report_file": "conformance.json",
                },
                "irrelevant_control": {
                    "conformance_report_file": "oscillator-conformance.json",
                },
                "coverage_arithmetic": {
                    "conformance_report_file": "coverage-conformance.json",
                },
            }
        ),
        encoding="utf-8",
    )

    with ProducerAttempt(tmp_path / "attempt", lane="formal") as attempt:
        with pytest.raises(ValueError, match="digest changed"):
            attempt.preserve_formal_inputs(binding_path)


def test_experiment_entrypoint_refuses_unowned_or_existing_output(
    tmp_path: Path,
) -> None:
    formal = experiment_module.ExperimentConfig.formal(device="cpu")
    with pytest.raises(ValueError, match="ProducerAttempt"):
        experiment_module.run_experiment(
            formal,
            output_directory=tmp_path / "formal",
            formal_binding_path=tmp_path / "missing-binding.json",
        )

    development = experiment_module.ExperimentConfig.development(
        master_seeds=(2439054559,),
        device="cpu",
    )
    assert (
        development.collection_episodes,
        development.validation_episodes,
        development.behavior_episodes,
        development.optimizer_steps,
    ) == (8, 8, 32, 2000)
    with pytest.raises(FileExistsError):
        experiment_module.run_experiment(
            development,
            output_directory=tmp_path,
        )
