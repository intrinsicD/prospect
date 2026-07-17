"""Focused lifecycle and consumer-semantic tests for LCV-001."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.sealed_lineage_verifier import canary_probe, experiment, runtime_probe, supervisor


def _audit(path: Path) -> Path:
    _, protocol = experiment._stable_read(experiment.PROTOCOL_DOC)
    _, config = experiment._stable_read(experiment.CONFIG_DOC)
    _, source = experiment._source_snapshot()
    binding = experiment._audit_binding(
        config_sha256=str(config["sha256"]),
        protocol_sha256=str(protocol["sha256"]),
        source_sha256=experiment._source_sha256(source),
    )
    value = {
        "binding": binding,
        "checks": {
            "canonical_output_absent": True,
            "claim_boundary_accepted": True,
            "parent_pins_independently_checked": True,
            "runtime_closure_independently_checked": True,
            "source_tests_audited": True,
        },
        "decision": "GO",
        "experiment_id": "LCV-001",
        "independent": True,
        "reviewer": "pytest-independent-reviewer",
        "schema_version": "lcv001-pre-real-audit-v1",
    }
    experiment._write_exclusive(path, experiment._json_bytes(value))
    return path


def _prepare(tmp_path: Path) -> Path:
    output = tmp_path / "LCV-001"
    experiment._prepare_at(output, experiment.MM007_ROOT, _audit(tmp_path / "audit.json"))
    return output


def _make_writable(root: Path) -> None:
    for directory, names, _ in os.walk(root):
        os.chmod(directory, 0o755)
        for name in names:
            path = Path(directory) / name
            if path.is_dir():
                os.chmod(path, 0o755)


def _parent_state() -> tuple[dict[str, object], dict[str, object]]:
    snapshot = experiment._read_parent_snapshot(experiment.MM007_ROOT, copied=False)
    artifact = experiment._parse_parent_json(snapshot, "artifact-manifest.json")
    manifest = experiment._parse_parent_json(snapshot, "input-manifest.json")
    marker = experiment._parse_parent_json(snapshot, "formal-start.json")
    evidence = experiment._parse_parent_json(snapshot, "MM-007-evidence.json")
    result = experiment._parse_parent_json(snapshot, "MM-007-results.json")
    experiment._validate_mm007_crosslinks(artifact, manifest, marker, evidence, result)
    arrays = experiment._load_frame_arrays(snapshot[Path("MM-007-frames-64x64.npz")])
    _semantics, state = experiment._consumer_semantics(arrays, manifest, evidence, result)
    state.update({"artifact": artifact, "manifest": manifest, "marker": marker})
    return {"manifest": manifest, "marker": marker, "evidence": evidence, "result": result}, state


def test_canonical_output_is_absent_during_development() -> None:
    assert not experiment.EXPECTED_OUTPUT.exists()


def test_prepare_copies_all_opaque_parent_bytes_without_parent_parser_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "_validate_parent_closure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parent semantic verifier ran in prepare")),
    )
    monkeypatch.setattr(
        experiment,
        "_load_frame_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("NPZ parser ran in prepare")),
    )
    output = _prepare(tmp_path)
    assert experiment._tree_census(output)[0] == tuple(sorted(experiment.PREPARED_FILES, key=str))
    assert all(experiment._file_record(output / path)["mode"] == 0o444 for path in experiment.PREPARED_FILES)
    assert output.stat().st_mode & 0o777 == 0o555
    assert (output / experiment.PREPARED_ROOT).stat().st_mode & 0o777 == 0o555
    assert (output / experiment.OUTCOMES_ROOT).stat().st_mode & 0o777 == 0o755
    assert all(
        (output / path).stat().st_mode & 0o777 == (0o755 if path == experiment.OUTCOMES_ROOT else 0o555)
        for path in experiment._expected_directories(experiment.PREPARED_FILES)
    )
    for relative, (digest, size, _) in experiment.MM007_PINS.items():
        assert experiment._file_record(output / experiment.PARENT_COPY_ROOT / relative) == {
            "bytes": size,
            "mode": 0o444,
            "sha256": digest,
        }


def test_prepared_subtree_prevents_path_replacement_before_formal(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    target = output / experiment.CONFIG_COPY
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(target.read_bytes())
    with pytest.raises(PermissionError):
        os.replace(replacement, target)
    with pytest.raises(PermissionError):
        os.rename(output / experiment.PREPARED_ROOT, tmp_path / "retired-prepared")
    experiment._validate_prepared(output)


def test_staging_failure_after_parent_copy_keeps_destination_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "LCV-001"
    audit = _audit(tmp_path / "audit.json")
    original_write = experiment._write_exclusive

    def injected_write(path: Path, payload: bytes, mode: int = 0o444) -> None:
        if path.as_posix().endswith(experiment.RUNTIME_EMPTY_PARENT.as_posix()):
            raise experiment.InvalidLCV001Artifact("injected after sealed parent copy")
        original_write(path, payload, mode)

    monkeypatch.setattr(experiment, "_write_exclusive", injected_write)
    with pytest.raises(experiment.InvalidLCV001Artifact, match="injected after sealed parent copy"):
        experiment._prepare_at(output, experiment.MM007_ROOT, audit)
    assert not output.exists()
    assert list(tmp_path.glob(".LCV-001.prepare-*")) == []


@pytest.mark.parametrize("error", (KeyboardInterrupt(), SystemExit(23), FileExistsError("injected")))
def test_preparation_owned_workspace_interrupt_is_cleaned_before_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    output = tmp_path / "LCV-001"

    def interrupt_after_ownership(name: str, _path: Path) -> None:
        if name == "preparation_after_mkdir":
            raise error

    monkeypatch.setattr(experiment, "_workspace_allocation_checkpoint", interrupt_after_ownership)
    expected = experiment.InvalidLCV001Runtime if isinstance(error, OSError) else type(error)
    with pytest.raises(expected):
        experiment._prepare_at(output, experiment.MM007_ROOT, _audit(tmp_path / "audit.json"))
    assert not output.exists()
    assert list(tmp_path.glob(".LCV-001.prepare-*")) == []


def test_preparation_workspace_chmod_failure_cleans_owned_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "LCV-001"
    real_chmod = experiment.os.chmod
    injected = False

    def fail_first_staging_chmod(path: Any, mode: int, *, follow_symlinks: bool = True) -> None:
        nonlocal injected
        if ".LCV-001.prepare-" in os.fspath(path) and not injected:
            injected = True
            raise OSError("injected preparation workspace chmod failure")
        real_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(experiment.os, "chmod", fail_first_staging_chmod)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="private workspace creation failed"):
        experiment._prepare_at(output, experiment.MM007_ROOT, _audit(tmp_path / "audit.json"))
    assert injected is True
    assert not output.exists()
    assert list(tmp_path.glob(".LCV-001.prepare-*")) == []


def test_workspace_collision_never_removes_preexisting_directory(tmp_path: Path) -> None:
    existing = tmp_path / ".LCV-001.prepare-foreign"
    existing.mkdir()
    marker = existing / "owner-marker.txt"
    marker.write_text("foreign", encoding="utf-8")
    ownership = [False]
    with pytest.raises(experiment.InvalidLCV001Runtime, match="appeared before creation"):
        experiment._create_owned_workspace(
            existing,
            mode=0o700,
            ownership=ownership,
            checkpoint="preparation_after_mkdir",
        )
    assert ownership == [False]
    assert marker.read_text(encoding="utf-8") == "foreign"


def test_consumer_semantics_replay_exact_fold_before_pool_normalizers(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    closure, _ = experiment._validate_parent_closure(output / experiment.PARENT_COPY_ROOT)
    semantics = closure["semantics"]
    assert isinstance(semantics, dict)
    panel = semantics["successor_panel"]
    normalizers = semantics["normalizers"]
    assert panel["rows"] == 453
    assert panel["current_identity_sha256"] == experiment.MATCHED_IDENTITY_SHA256
    assert [row["train_rows"] for row in normalizers["folds"]] == [332, 335, 346, 346]
    assert [row["rows"] for row in normalizers["training_current_indices"]] == [332, 335, 346, 346]
    assert all(row["r8_strides"] == [768, 256, 32, 4] for row in normalizers["training_current_indices"])
    assert normalizers["full_current_r8_sha256"] == experiment.R8_CURRENT_SHA256


def test_parent_copy_same_size_mutation_fails_hard_pin(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    target = output / experiment.PARENT_COPY_ROOT / "MM-007-evidence.json"
    os.chmod(target, 0o644)
    descriptor = os.open(target, os.O_RDWR | os.O_NOFOLLOW)
    try:
        first = os.pread(descriptor, 1, 0)
        os.pwrite(descriptor, bytes((first[0] ^ 1,)), 0)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(target, 0o444)
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_parent_closure(output / experiment.PARENT_COPY_ROOT)


def test_wrong_fold_indices_and_layout_are_rejected() -> None:
    ids = np.asarray(
        [name for name in experiment.VIDEO_IDS for _ in range(experiment.MATCHED_COUNTS[name])], dtype="<U11"
    )
    current = np.arange(1, 454, dtype="<i8")
    expected = experiment._expected_train_current_indices(experiment.FOLDS[0], current, ids)
    wrong = expected.copy()
    wrong[0] += 1
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_train_current_indices(wrong, experiment.FOLDS[0], current, ids)
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_r8_layout(np.asfortranarray(np.zeros((332, 3, 8, 8), dtype="<f4")), 332)


def test_full_shadow_runtime_run_seals_and_semantically_verifies(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    assert experiment._run_at(output)["classification"] == "PASS"
    assert experiment.verify(output)["outcomes"] == "verified_results"
    assert experiment.verify_semantic(output)["outcomes"] == "verified_semantic_results"
    files, directories = experiment._tree_census(output)
    assert files == tuple(sorted(experiment.COMPLETED_FILES, key=str))
    assert all(experiment._file_record(output / path)["mode"] == 0o444 for path in files)
    assert all((output / path).stat().st_mode & 0o777 == 0o555 for path in directories)
    assert output.stat().st_mode & 0o777 == 0o555
    runtime = experiment._read_output_json(output, experiment.RUNTIME_RECEIPT)
    mutations: list[tuple[str, Any]] = [
        ("extra", True),
        ("canary.u_sha256", "0" * 64),
        ("canary.shape", [1, 1]),
        ("dependency_hashes.openblas", "0" * 64),
        ("loaded_openblas", []),
        ("numpy_file", "/tmp/numpy.py"),
        ("symlink_chain", []),
        ("environment", {}),
        ("exec_prefix", "/tmp"),
    ]
    for dotted, value in mutations:
        mutated = copy.deepcopy(runtime)
        target: dict[str, Any] = mutated
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        with pytest.raises(experiment.InvalidLCV001Runtime):
            experiment._validate_runtime_receipt(mutated)


def test_completed_tree_extra_file_and_formal_retry_fail(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    experiment._run_at(output)
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._run_at(output)
    _make_writable(output)
    experiment._write_exclusive(output / "extra.json", b"{}\n")
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment.verify(output)


def test_normalizer_fingerprint_mutation_is_rejected(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    _, state = experiment._validate_parent_closure(output / experiment.PARENT_COPY_ROOT)
    evidence = copy.deepcopy(state["evidence"])
    assert isinstance(evidence, dict)
    evidence["normalizer_rows"][0]["fingerprint"] = "0" * 64
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_normalizer_rows(
            evidence["normalizer_rows"], cast(Sequence[Mapping[str, object]], state["normalizers"])
        )


def test_concrete_wrong_layout_endpoints_are_preregistered_and_rejected() -> None:
    _receipts, state = _parent_state()
    arrays = cast(Mapping[str, np.ndarray], state["arrays"])
    wrong, endpoint = experiment._wrong_select_after_pool_rows(
        arrays["frames_uint8"],
        cast(np.ndarray, state["current"]),
        cast(np.ndarray, state["matched_ids"]),
        cast(Sequence[Mapping[str, object]], state["normalizers"]),
    )
    assert endpoint == {
        "max_abs_difference": list(experiment.WRONG_NORMALIZER_MAX_ABS),
        "max_ulp_difference": list(experiment.WRONG_NORMALIZER_MAX_ULP),
        "strides": list(experiment.WRONG_NORMALIZER_STRIDES),
        "wrong_fingerprints": list(experiment.WRONG_NORMALIZER_FINGERPRINTS),
    }
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_normalizer_rows(
            wrong,
            cast(Sequence[Mapping[str, object]], state["normalizers"]),
        )


def test_consumed_mm007_receipts_are_strict_and_all_duplicate_anchors_are_checked() -> None:
    receipts, state = _parent_state()
    manifest = cast(dict[str, Any], receipts["manifest"])
    marker = cast(dict[str, Any], receipts["marker"])
    evidence = cast(dict[str, Any], receipts["evidence"])
    result = cast(dict[str, Any], receipts["result"])
    artifact = cast(dict[str, Any], state["artifact"])

    extra_manifest = copy.deepcopy(manifest)
    extra_manifest["unexpected"] = True
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_mm007_crosslinks(artifact, extra_manifest, marker, evidence, result)
    for field in ("frames_uint8_sha256", "frame_schema_sha256", "parent_alignment_sha256"):
        mutated = copy.deepcopy(marker)
        mutated[field] = "0" * 64
        with pytest.raises(experiment.InvalidLCV001Artifact):
            experiment._validate_mm007_crosslinks(artifact, manifest, mutated, evidence, result)
    for field in ("frame_file_sha256", "frames_uint8_sha256", "parent_alignment_sha256"):
        mutated = copy.deepcopy(result)
        mutated[field] = "0" * 64
        with pytest.raises(experiment.InvalidLCV001Artifact):
            experiment._validate_mm007_crosslinks(artifact, manifest, marker, evidence, mutated)
    wrong_boolean = copy.deepcopy(evidence["normalizer_rows"])
    wrong_boolean[0]["uses_target"] = 0
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_normalizer_rows(
            wrong_boolean,
            cast(Sequence[Mapping[str, object]], state["normalizers"]),
        )
    bad_r8 = copy.deepcopy(evidence)
    bad_r8["alignment"]["resolutions"]["8"]["current_sha256"] = "0" * 64
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._consumer_semantics(cast(Mapping[str, np.ndarray], state["arrays"]), manifest, bad_r8, result)


def _cleanup_receipt(unit: str) -> dict[str, object]:
    return {
        "actions": [
            {"action": action, "returncode": 1, "stderr": "", "stdout": ""}
            for action in ("kill", "stop", "reset-failed")
        ],
        "cgroup_paths": [],
        "is_active_returncode": 4,
        "schema_version": "lcv001-cgroup-cleanup-v1",
        "state": "unknown",
        "status": "inactive_and_cgroup_absent",
        "unit": unit,
    }


def test_post_marker_verifier_failure_is_terminal_sealed_and_not_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def failed_child(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        stderr = '{"classification":"invalid_LCV001_verifier","error":"injected"}\n'
        return supervisor.SupervisedCompletedProcess(tuple(command), 2, "", stderr, _cleanup_receipt(unit))

    monkeypatch.setattr(supervisor, "run", failed_child)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"
    files, directories = experiment._tree_census(output)
    assert experiment.FORMAL_START in files
    assert experiment.TERMINAL_FAILURE in files
    assert all(experiment._file_record(output / path)["mode"] == 0o444 for path in files)
    assert output.stat().st_mode & 0o777 == 0o555
    assert all((output / path).stat().st_mode & 0o777 == 0o555 for path in directories)
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._run_at(output)


def test_real_provisional_cannot_be_promoted_with_mismatched_cleanup_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    real_run = supervisor.run

    def mismatched_cleanup(command: Sequence[str], **kwargs: Any) -> supervisor.SupervisedCompletedProcess:
        completed = real_run(command, **kwargs)
        if kwargs.get("role") != "formal":
            return completed
        receipt = copy.deepcopy(completed.cleanup_receipt)
        receipt["unit"] = "lcv001-custody-formal-999-888-777.service"
        return supervisor.SupervisedCompletedProcess(
            completed.args,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            receipt,
        )

    monkeypatch.setattr(supervisor, "run", mismatched_cleanup)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="different formal cgroup"):
        experiment._run_at(output)
    assert not (output / experiment.RESULTS_FILE).exists()
    assert not (output / experiment.ARTIFACT_MANIFEST).exists()
    assert (output / experiment.TERMINAL_FAILURE).exists()
    provisional = experiment._read_output_json(output, experiment.PROVISIONAL_RESULT)
    assert provisional["classification"] == "PENDING_CGROUP_CLEANUP"
    assert output.stat().st_mode & 0o777 == 0o555
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._run_at(output)


@pytest.mark.parametrize(
    "phase",
    (
        "after_cleanup_validation",
        "after_candidate_copy",
        "after_cleanup_receipt_write",
        "after_result_write",
        "after_report_write",
        "after_artifact_manifest_write",
        "before_candidate_seal",
        "after_candidate_seal",
        "after_candidate_verify",
        "before_atomic_exchange",
    ),
)
def test_every_precommit_finalization_fault_terminalizes_only_provisional_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    output = _prepare(tmp_path)

    def fail_at_phase(name: str) -> None:
        if name == phase:
            raise RuntimeError(f"injected finalization fault: {phase}")

    monkeypatch.setattr(experiment, "_finalization_checkpoint", fail_at_phase)
    with pytest.raises(experiment.InvalidLCV001Verifier, match="parent finalization failed"):
        experiment._run_at(output)

    files, directories = experiment._tree_census(output)
    assert set(files) == {*experiment.PROVISIONAL_FILES, experiment.TERMINAL_FAILURE}
    final_only = {
        experiment.CLEANUP_RECEIPT,
        experiment.RESULTS_FILE,
        experiment.REPORT_FILE,
        experiment.ARTIFACT_MANIFEST,
    }
    assert not (final_only & set(files))
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"
    assert all(experiment._file_record(output / path)["mode"] == 0o444 for path in files)
    assert output.stat().st_mode & 0o777 == 0o555
    assert all((output / path).stat().st_mode & 0o777 == 0o555 for path in directories)
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._run_at(output)


def test_precommit_host_io_fault_is_terminal_runtime_not_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)

    def fail_at_phase(name: str) -> None:
        if name == "after_result_write":
            raise OSError("injected host I/O fault")

    monkeypatch.setattr(experiment, "_finalization_checkpoint", fail_at_phase)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="parent finalization failed"):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"
    assert set(experiment._tree_census(output)[0]) == {*experiment.PROVISIONAL_FILES, experiment.TERMINAL_FAILURE}


@pytest.mark.parametrize(
    "error",
    (
        OSError("injected completion allocation I/O"),
        FileExistsError("injected post-ownership collision"),
        KeyboardInterrupt(),
        SystemExit(23),
    ),
)
def test_completion_owned_workspace_failure_is_cleaned_before_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    output = _prepare(tmp_path)

    def fail_after_ownership(name: str, _path: Path) -> None:
        if name == "completion_after_mkdir":
            raise error

    monkeypatch.setattr(experiment, "_workspace_allocation_checkpoint", fail_after_ownership)
    expected = (
        KeyboardInterrupt
        if isinstance(error, KeyboardInterrupt)
        else experiment.InvalidLCV001Runtime
        if isinstance(error, OSError)
        else experiment.InvalidLCV001Verifier
    )
    with pytest.raises(expected):
        experiment._run_at(output)
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == (
        "invalid_LCV001_verifier" if isinstance(error, SystemExit) else "invalid_LCV001_runtime"
    )


def test_completion_workspace_chmod_failure_cleans_before_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    real_chmod = experiment.os.chmod
    injected = False

    def fail_first_completion_chmod(path: Any, mode: int, *, follow_symlinks: bool = True) -> None:
        nonlocal injected
        if ".LCV-001.completion-" in str(path) and not injected:
            injected = True
            raise OSError("injected completion workspace chmod failure")
        real_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(experiment.os, "chmod", fail_first_completion_chmod)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="private workspace creation failed"):
        experiment._run_at(output)
    assert injected is True
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"


def test_postcommit_faults_do_not_reclassify_verified_canonical_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)

    def fail_after_exchange(name: str) -> None:
        if name == "after_atomic_exchange":
            raise OSError("injected post-commit parent fsync fault")

    monkeypatch.setattr(experiment, "_finalization_checkpoint", fail_after_exchange)
    result = experiment._run_at(output)
    commit = cast(Mapping[str, Any], result["commit"])
    assert result["classification"] == "PASS"
    assert commit["atomic_exchange"] is True
    assert commit["parent_directory_fsync"] is False
    assert commit["retired_provisional_removed"] is True
    assert commit["warnings"] == [{"error_type": "OSError", "phase": "after_atomic_exchange"}]
    assert experiment.verify(output)["classification"] == "PASS"
    assert not (output / experiment.TERMINAL_FAILURE).exists()
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []


@pytest.mark.parametrize("error", (KeyboardInterrupt(), SystemExit(23), MemoryError("injected")))
def test_commit_receipt_return_boundary_never_terminalizes_completed_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    output = _prepare(tmp_path)

    def fail_before_receipt_return(name: str) -> None:
        if name == "before_commit_receipt_return":
            raise error

    monkeypatch.setattr(experiment, "_finalization_checkpoint", fail_before_receipt_return)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="exact commit receipt is unavailable"):
        experiment._run_at(output)
    assert experiment.verify(output)["classification"] == "PASS"
    assert not (output / experiment.TERMINAL_FAILURE).exists()
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []


def test_indeterminate_canonical_phase_is_left_untouched_for_manual_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"
    unexpected = experiment.OUTCOMES_ROOT / "unexpected-empty-directory"

    def indeterminate_child(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        (output / unexpected).mkdir()
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            0,
            "not-json\n",
            "",
            _cleanup_receipt(unit),
        )

    monkeypatch.setattr(supervisor, "run", indeterminate_child)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="canonical phase is indeterminate"):
        experiment._run_at(output)
    assert (output / unexpected).is_dir()
    assert not (output / experiment.TERMINAL_FAILURE).exists()


def test_retired_provisional_cleanup_is_postcommit_and_non_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)

    def fail_retired_cleanup(name: str) -> None:
        if name == "retired_provisional_cleanup":
            raise OSError("injected retired-tree cleanup fault")

    monkeypatch.setattr(experiment, "_finalization_checkpoint", fail_retired_cleanup)
    result = experiment._run_at(output)
    workspaces = list(output.parent.glob(experiment._completion_workspace_pattern(output)))
    try:
        commit = cast(Mapping[str, Any], result["commit"])
        assert result["classification"] == "PASS"
        assert commit["retired_provisional_removed"] is False
        assert len(workspaces) == 1 and workspaces[0].exists()
        assert experiment.verify(output)["classification"] == "PASS"
        assert not (output / experiment.TERMINAL_FAILURE).exists()
    finally:
        for workspace in workspaces:
            if workspace.exists():
                experiment.custody.remove_created_tree(workspace)


def test_completion_workspace_never_removes_a_preexisting_foreign_tree(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    foreign = output.parent / f".{output.name}.completion-foreign"
    foreign.mkdir()
    marker = foreign / "owner-marker.txt"
    marker.write_text("not created by LCV-001", encoding="utf-8")
    try:
        result = experiment._run_at(output)
        assert result["classification"] == "PASS"
        assert marker.read_text(encoding="utf-8") == "not created by LCV-001"
        assert experiment.verify(output)["classification"] == "PASS"
    finally:
        if foreign.exists():
            experiment.custody.remove_created_tree(foreign)


def test_exchange_success_then_wrapper_exception_is_detected_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    real_exchange = experiment._rename_exchange

    def exchange_then_raise(source: Path, destination: Path) -> None:
        real_exchange(source, destination)
        raise RuntimeError("injected immediately after successful exchange")

    monkeypatch.setattr(experiment, "_rename_exchange", exchange_then_raise)
    result = experiment._run_at(output)
    commit = cast(Mapping[str, Any], result["commit"])
    assert result["classification"] == "PASS"
    assert commit == {
        "atomic_exchange": True,
        "parent_directory_fsync": True,
        "retired_provisional_removed": True,
        "warnings": [{"error_type": "RuntimeError", "phase": "exchange_return_boundary"}],
    }
    assert experiment.verify(output)["classification"] == "PASS"
    assert not (output / experiment.TERMINAL_FAILURE).exists()
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []


def test_exchange_wrapper_exception_before_swap_is_precommit_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)

    def fail_before_exchange(_source: Path, _destination: Path) -> None:
        raise RuntimeError("injected before exchange syscall")

    monkeypatch.setattr(experiment, "_rename_exchange", fail_before_exchange)
    with pytest.raises(experiment.InvalidLCV001Verifier, match="parent finalization failed"):
        experiment._run_at(output)
    assert set(experiment._tree_census(output)[0]) == {
        *experiment.PROVISIONAL_FILES,
        experiment.TERMINAL_FAILURE,
    }
    assert experiment._read_output_json(output, experiment.TERMINAL_FAILURE)["classification"] == (
        "invalid_LCV001_verifier"
    )
    assert list(output.parent.glob(experiment._completion_workspace_pattern(output))) == []


@pytest.mark.parametrize(
    "error",
    (supervisor.SupervisorError("injected"), RuntimeError("injected"), KeyboardInterrupt()),
)
def test_raw_post_marker_supervisor_or_interrupt_is_terminal_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def failed_supervisor(_command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        error.__dict__["lcv001_cleanup_receipt"] = _cleanup_receipt(unit)
        raise error

    monkeypatch.setattr(supervisor, "run", failed_supervisor)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else experiment.InvalidLCV001Runtime
    with pytest.raises(expected):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"
    assert set(experiment._tree_census(output)[0]) <= {*experiment.PROVISIONAL_FILES, experiment.TERMINAL_FAILURE}


def test_supervisor_failure_does_not_write_into_indeterminate_canonical_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"
    unexpected = output / experiment.OUTCOMES_ROOT / "unexpected-empty-directory"

    def failed_supervisor(_command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        unexpected.mkdir()
        error = supervisor.SupervisorError("injected supervisor failure")
        error.__dict__["lcv001_cleanup_receipt"] = _cleanup_receipt(unit)
        raise error

    monkeypatch.setattr(supervisor, "run", failed_supervisor)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="canonical phase is indeterminate"):
        experiment._run_at(output)
    assert unexpected.is_dir()
    assert not (output / experiment.TERMINAL_FAILURE).exists()


@pytest.mark.parametrize("invalid_value", (float("nan"), object()))
def test_raw_supervisor_malformed_cleanup_cannot_escape_terminal_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: object,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def failed_supervisor(_command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        cleanup = _cleanup_receipt(unit)
        cast(dict[str, Any], cast(list[object], cleanup["actions"])[0])["stdout"] = invalid_value
        error = supervisor.SupervisorError("injected supervisor failure")
        error.__dict__["lcv001_cleanup_receipt"] = cleanup
        raise error

    monkeypatch.setattr(supervisor, "run", failed_supervisor)
    with pytest.raises(experiment.InvalidLCV001Runtime):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"
    assert terminal["cleanup"] == {
        "receipt_sha256": None,
        "schema_version": None,
        "status": "invalid_untrusted_cleanup_receipt",
        "unit": None,
    }


def test_cyclic_cleanup_value_is_reduced_to_bounded_invalid_summary() -> None:
    cleanup: dict[str, object] = {}
    cleanup["self"] = cleanup
    payload = experiment._terminal_failure_payload(experiment.InvalidLCV001Runtime("injected"), cleanup)
    assert cast(Mapping[str, object], payload["cleanup"])["status"] == "invalid_untrusted_cleanup_receipt"


def test_untyped_post_marker_child_crash_is_terminal_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def crashed_child(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            2,
            "",
            "Traceback (most recent call last):\nRuntimeError: injected\n",
            _cleanup_receipt(unit),
        )

    monkeypatch.setattr(supervisor, "run", crashed_child)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"


@pytest.mark.parametrize("error", (OSError("injected child host I/O"), KeyboardInterrupt()))
def test_formal_child_host_failure_is_serialized_and_parent_classifies_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    def host_failure(_output: Path) -> dict[str, object]:
        raise error

    monkeypatch.setattr(experiment, "_formal_run_in_process", host_failure)
    assert experiment.main(["_formal-run", "--output", str(tmp_path / "LCV-001")]) == (
        experiment.RUNTIME_TRANSPORT_EXIT
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["classification"] == "invalid_LCV001_runtime"
    classified = experiment._classified_child_failure(
        captured.err,
        stdout="",
        returncode=experiment.RUNTIME_TRANSPORT_EXIT,
        default_runtime=False,
    )
    assert isinstance(classified, experiment.InvalidLCV001Runtime)


class _FaultingCLIStream:
    def __init__(self, phase: str, error: BaseException) -> None:
        self.phase = phase
        self.error = error
        self.payload = ""

    def write(self, payload: str) -> int:
        if self.phase == "write":
            raise self.error
        if self.phase == "short":
            return max(0, len(payload) - 1)
        self.payload += payload
        return len(payload)

    def flush(self) -> None:
        if self.phase == "flush":
            raise self.error


class _ToxicBoundaryError(BaseException):
    def __str__(self) -> str:
        raise RuntimeError("injected toxic exception text")

    def __getattribute__(self, name: str) -> Any:
        if name == "lcv001_cleanup_receipt":
            raise RuntimeError("injected toxic cleanup attribute")
        return super().__getattribute__(name)


@pytest.mark.parametrize(
    ("phase", "error"),
    (
        ("write", BrokenPipeError("injected child stdout write")),
        ("flush", BrokenPipeError("injected child stdout flush")),
        ("write", ValueError("injected closed child stdout write")),
        ("flush", ValueError("injected closed child stdout flush")),
        ("write", KeyboardInterrupt()),
        ("flush", KeyboardInterrupt()),
        ("short", OSError("unused short-write sentinel")),
    ),
)
def test_formal_child_success_output_write_and_flush_failures_are_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    error: BaseException,
) -> None:
    monkeypatch.setattr(experiment, "_formal_run_in_process", lambda _output: {"status": "success"})
    stream = _FaultingCLIStream(phase, error)
    monkeypatch.setattr(experiment.sys, "stdout", stream)
    assert experiment.main(["_formal-run", "--output", str(tmp_path / "LCV-001")]) == (
        experiment.RUNTIME_TRANSPORT_EXIT
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["classification"] == "invalid_LCV001_runtime"
    assert isinstance(
        experiment._classified_child_failure(
            captured.err,
            stdout=stream.payload,
            returncode=experiment.RUNTIME_TRANSPORT_EXIT,
            default_runtime=False,
        ),
        experiment.InvalidLCV001Runtime,
    )


@pytest.mark.parametrize(
    "error",
    (
        BrokenPipeError("injected stdout and diagnostic failure"),
        ValueError("injected closed stdout and diagnostic"),
        KeyboardInterrupt(),
    ),
)
def test_formal_child_double_output_failure_uses_runtime_transport_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    monkeypatch.setattr(experiment, "_formal_run_in_process", lambda _output: {"status": "success"})

    def failed_channel(_payload: str, _stream: object) -> None:
        raise error

    monkeypatch.setattr(experiment, "_write_cli_line", failed_channel)
    assert experiment.main(["_formal-run", "--output", str(tmp_path / "LCV-001")]) == (
        experiment.RUNTIME_TRANSPORT_EXIT
    )
    assert isinstance(
        experiment._classified_child_failure(
            "",
            stdout="",
            returncode=experiment.RUNTIME_TRANSPORT_EXIT,
            default_runtime=False,
        ),
        experiment.InvalidLCV001Runtime,
    )


@pytest.mark.parametrize("returncode", sorted(experiment.RUNTIME_TRANSPORT_RETURN_CODES))
def test_unstructured_runtime_transport_return_codes_do_not_become_verifier_failures(returncode: int) -> None:
    assert isinstance(
        experiment._classified_child_failure("", stdout="", returncode=returncode, default_runtime=False),
        experiment.InvalidLCV001Runtime,
    )
    mismatch = experiment._cli_json_line(
        {"classification": "invalid_LCV001_artifact", "error": "injected"},
        compact=False,
    )
    runtime = _diagnostic("invalid_LCV001_runtime")
    pass_claim = experiment._cli_json_line({"classification": "PASS", "status": "impossible"}, compact=True)
    pending_claim = experiment._cli_json_line(
        {"classification": "PENDING_CGROUP_CLEANUP"},
        compact=False,
    )
    nonstring_claim = experiment._cli_json_line({"classification": 1}, compact=False)
    for contradictory in (
        mismatch,
        mismatch + "shutdown noise\n",
        mismatch + runtime,
        pass_claim,
        pending_claim,
        nonstring_claim,
    ):
        assert isinstance(
            experiment._classified_child_failure(
                contradictory,
                stdout="",
                returncode=returncode,
                default_runtime=False,
            ),
            experiment.InvalidLCV001Verifier,
        )


def _diagnostic(classification: str) -> str:
    return experiment._cli_json_line({"classification": classification, "error": "injected"}, compact=False)


@pytest.mark.parametrize(
    ("classification", "expected"),
    (
        ("invalid_LCV001_artifact", experiment.InvalidLCV001Artifact),
        ("invalid_LCV001_runtime", experiment.InvalidLCV001Runtime),
        ("invalid_LCV001_verifier", experiment.InvalidLCV001Verifier),
    ),
)
def test_classified_exit_accepts_only_exact_diagnostic_and_empty_stdout(
    classification: str,
    expected: type[BaseException],
) -> None:
    assert isinstance(
        experiment._classified_child_failure(
            _diagnostic(classification),
            stdout="",
            returncode=experiment.CLASSIFIED_CHILD_EXIT,
            default_runtime=False,
        ),
        expected,
    )
    assert isinstance(
        experiment._classified_child_failure(
            _diagnostic(classification),
            stdout="unexpected\n",
            returncode=experiment.CLASSIFIED_CHILD_EXIT,
            default_runtime=True,
        ),
        experiment.InvalidLCV001Verifier,
    )


@pytest.mark.parametrize(
    "stderr",
    (
        "",
        "noise\n" + _diagnostic("invalid_LCV001_runtime"),
        '{"classification":"invalid_LCV001_runtime","classification":"invalid_LCV001_runtime","error":"x"}\n',
        '{"classification":"invalid_LCV001_runtime","error":"x","extra":true}\n',
        '{"classification":"invalid_LCV001_runtime"}\n',
        '{"classification":"invalid_LCV001_runtime","error":1}\n',
        '{"classification":"invalid_LCV001_runtime","error":NaN}\n',
        "9" * 5000 + "\n",
    ),
)
def test_malformed_classified_diagnostic_is_verifier_owned_and_parser_safe(stderr: str) -> None:
    assert isinstance(
        experiment._classified_child_failure(
            stderr,
            stdout="",
            returncode=experiment.CLASSIFIED_CHILD_EXIT,
            default_runtime=True,
        ),
        experiment.InvalidLCV001Verifier,
    )


def test_arbitrary_return_code_cannot_claim_artifact_or_runtime_ownership() -> None:
    for classification in ("invalid_LCV001_artifact", "invalid_LCV001_runtime"):
        assert isinstance(
            experiment._classified_child_failure(
                _diagnostic(classification),
                stdout="",
                returncode=1,
                default_runtime=True,
            ),
            experiment.InvalidLCV001Verifier,
        )
    assert isinstance(
        experiment._classified_child_failure("traceback\n", stdout="", returncode=1, default_runtime=True),
        experiment.InvalidLCV001Runtime,
    )


def test_semantic_child_raw_unexpected_exit_is_verifier_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashed_semantic(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            1,
            "",
            "Traceback (most recent call last):\nRuntimeError: injected\n",
            _cleanup_receipt("lcv001-custody-semantic-123-456-789.service"),
        )

    monkeypatch.setattr(supervisor, "run", crashed_semantic)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._launch_lexical("_semantic-verify", tmp_path / "LCV-001")


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    (
        ('{"status":"accepted"}\n', _diagnostic("invalid_LCV001_runtime")),
        ("not-json\n", ""),
        ('{"a":1,"a":1}\n', ""),
        ('{"value":NaN}\n', ""),
        ("9" * 5000 + "\n", ""),
        ('{ "status": "accepted" }\n', ""),
        ('{"status":"accepted"}\nnoise\n', ""),
    ),
)
def test_formal_success_transport_defects_terminalize_as_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    stderr: str,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def defective_success(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            0,
            stdout,
            stderr,
            _cleanup_receipt(unit),
        )

    monkeypatch.setattr(supervisor, "run", defective_success)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"


@pytest.mark.parametrize("error", (KeyboardInterrupt(), SystemExit(23)))
def test_post_supervisor_transport_interruption_or_exit_terminalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def nominal_success(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            0,
            '{"status":"accepted"}\n',
            "",
            _cleanup_receipt(unit),
        )

    def interrupted_parse(_stdout: str, _stderr: str, _label: str) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(supervisor, "run", nominal_success)
    monkeypatch.setattr(experiment, "_validated_child_success", interrupted_parse)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else experiment.InvalidLCV001Verifier
    with pytest.raises(expected):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == (
        "invalid_LCV001_runtime" if isinstance(error, KeyboardInterrupt) else "invalid_LCV001_verifier"
    )


def test_immediate_post_supervisor_return_interrupt_is_inside_terminal_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def nominal_child(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            0,
            '{"status":"accepted"}\n',
            "",
            _cleanup_receipt(unit),
        )

    def interrupt_return_boundary(_action: str, _output: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(supervisor, "run", nominal_child)
    monkeypatch.setattr(experiment, "_post_supervisor_return_checkpoint", interrupt_return_boundary)
    with pytest.raises(KeyboardInterrupt):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"


def test_malformed_supervisor_return_is_terminal_verifier_not_handler_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)

    def malformed_return(_command: Sequence[str], **_kwargs: object) -> Any:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return object()

    monkeypatch.setattr(supervisor, "run", malformed_return)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"


@pytest.mark.parametrize("defect", ("boolean_returncode", "wrong_args"))
def test_malformed_supervised_completed_contract_is_terminal_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def malformed_contract(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            ("wrong",) if defect == "wrong_args" else tuple(command),
            False if defect == "boolean_returncode" else 0,
            '{"status":"accepted"}\n',
            "",
            _cleanup_receipt(unit),
        )

    monkeypatch.setattr(supervisor, "run", malformed_contract)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"


@pytest.mark.parametrize("error", (KeyboardInterrupt(), SystemExit(23)))
def test_cleanup_validator_baseexception_is_bounded_inside_terminal_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def nominal_child(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            0,
            '{"status":"accepted"}\n',
            "",
            _cleanup_receipt(unit),
        )

    def toxic_validator(_value: object, *, role: str) -> dict[str, object]:
        del role
        raise error

    monkeypatch.setattr(supervisor, "run", nominal_child)
    monkeypatch.setattr(supervisor, "validate_cleanup_receipt", toxic_validator)
    with pytest.raises(experiment.InvalidLCV001Runtime):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"
    assert cast(Mapping[str, object], terminal["cleanup"])["status"] == (
        "invalid_untrusted_cleanup_receipt"
    )


def test_toxic_post_return_exception_text_cannot_escape_terminal_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    unit = "lcv001-custody-formal-123-456-789.service"

    def nominal_child(command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        return supervisor.SupervisedCompletedProcess(
            tuple(command),
            0,
            '{"status":"accepted"}\n',
            "",
            _cleanup_receipt(unit),
        )

    monkeypatch.setattr(supervisor, "run", nominal_child)
    def toxic_return_boundary(_action: str, _output: Path) -> None:
        raise _ToxicBoundaryError

    monkeypatch.setattr(experiment, "_post_supervisor_return_checkpoint", toxic_return_boundary)
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_verifier"


def test_toxic_supervisor_error_cleanup_attribute_and_text_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)

    def toxic_supervisor(_command: Sequence[str], **_kwargs: object) -> supervisor.SupervisedCompletedProcess:
        experiment._write_exclusive(output / experiment.FORMAL_START, b"{}\n")
        raise _ToxicBoundaryError

    monkeypatch.setattr(supervisor, "run", toxic_supervisor)
    with pytest.raises(experiment.InvalidLCV001Runtime):
        experiment._run_at(output)
    terminal = experiment._read_output_json(output, experiment.TERMINAL_FAILURE)
    assert terminal["classification"] == "invalid_LCV001_runtime"
    assert terminal["cleanup"] is None


@pytest.mark.parametrize(
    "error",
    (
        OSError("injected canary spawn failure"),
        subprocess.TimeoutExpired(("canary",), 60),
        KeyboardInterrupt(),
    ),
)
def test_nested_canary_spawn_timeout_and_interrupt_are_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    def failed_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise error

    monkeypatch.setattr(experiment.subprocess, "run", failed_run)
    with pytest.raises(experiment.InvalidLCV001Runtime):
        experiment._canary_child(tmp_path, Path("/python"), Path("/site"), 1)


@pytest.mark.parametrize("returncode", sorted(experiment.RUNTIME_TRANSPORT_RETURN_CODES))
def test_nested_canary_transport_return_codes_are_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(("canary",), returncode, b"partial", b""),
    )
    with pytest.raises(experiment.InvalidLCV001Runtime):
        experiment._canary_child(tmp_path, Path("/python"), Path("/site"), 1)


def test_nested_canary_transport_contradiction_and_arbitrary_code_are_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (
            experiment.RUNTIME_TRANSPORT_EXIT,
            _diagnostic("invalid_LCV001_artifact").encode(),
        ),
        (experiment.CLASSIFIED_CHILD_EXIT, _diagnostic("invalid_LCV001_runtime").encode()),
        (1, _diagnostic("invalid_LCV001_runtime").encode()),
    )
    for returncode, stderr in cases:
        monkeypatch.setattr(
            experiment.subprocess,
            "run",
            lambda *_args, returncode=returncode, stderr=stderr, **_kwargs: subprocess.CompletedProcess(
                ("canary"), returncode, b"", stderr
            ),
        )
        with pytest.raises(experiment.InvalidLCV001Verifier):
            experiment._canary_child(tmp_path, Path("/python"), Path("/site"), 1)


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    (
        (b"not-json\n", b""),
        (b'{"a":1,"a":1}\n', b""),
        (b'{"value":NaN}\n', b""),
        (b"\xff\n", b""),
        (b'{"bundle_sha256":"ok"}\n', b"noise\n"),
    ),
)
def test_nested_canary_success_transport_defects_are_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
) -> None:
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(("canary",), 0, stdout, stderr),
    )
    with pytest.raises(experiment.InvalidLCV001Verifier):
        experiment._canary_child(tmp_path, Path("/python"), Path("/site"), 1)


def test_nested_canary_accepts_one_canonical_success_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = b'{"bundle_sha256":"ok"}\n'
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(("canary",), 0, stdout, b""),
    )
    assert experiment._canary_child(tmp_path, Path("/python"), Path("/site"), 1) == {
        "bundle_sha256": "ok"
    }


@pytest.mark.parametrize(
    ("phase", "error"),
    (
        ("write", BrokenPipeError("injected canary stdout write")),
        ("flush", BrokenPipeError("injected canary stdout flush")),
        ("write", ValueError("injected closed canary stdout write")),
        ("flush", ValueError("injected closed canary stdout flush")),
        ("write", KeyboardInterrupt()),
        ("flush", KeyboardInterrupt()),
        ("short", OSError("unused short-write sentinel")),
    ),
)
def test_canary_probe_output_transport_failures_are_typed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    error: BaseException,
) -> None:
    monkeypatch.setattr(canary_probe.sys, "argv", ["canary_probe.py", "/site"])
    monkeypatch.setattr(canary_probe.sys, "path", list(canary_probe.sys.path))
    monkeypatch.setattr(runtime_probe, "svd_canary", lambda: {"bundle_sha256": "ok"})
    monkeypatch.setattr(canary_probe.sys, "stdout", _FaultingCLIStream(phase, error))
    assert canary_probe.main() == canary_probe.RUNTIME_TRANSPORT_EXIT
    diagnostic = json.loads(capsys.readouterr().err)
    assert diagnostic["classification"] == "invalid_LCV001_runtime"


def test_canary_probe_double_channel_failure_preserves_transport_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(canary_probe.sys, "argv", ["canary_probe.py", "/site"])
    monkeypatch.setattr(canary_probe.sys, "path", list(canary_probe.sys.path))
    monkeypatch.setattr(runtime_probe, "svd_canary", lambda: {"bundle_sha256": "ok"})

    def failed_channel(_payload: str, _stream: object) -> None:
        raise ValueError("injected closed canary stdout and stderr")

    monkeypatch.setattr(canary_probe, "_write_line", failed_channel)
    assert canary_probe.main() == canary_probe.RUNTIME_TRANSPORT_EXIT


def test_live_orchestrator_source_drift_is_rejected_before_formal_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    real_snapshot = experiment._source_snapshot

    def drifted_snapshot() -> tuple[dict[Path, bytes], dict[str, dict[str, object]]]:
        payloads, records = real_snapshot()
        changed = copy.deepcopy(records)
        changed[str(experiment.SOURCE_FILES[0])]["sha256"] = "0" * 64
        return payloads, changed

    monkeypatch.setattr(experiment, "_source_snapshot", drifted_snapshot)
    with pytest.raises(experiment.InvalidLCV001Artifact, match="live orchestration source differs"):
        experiment._run_at(output)
    assert not (output / experiment.FORMAL_START).exists()


def test_source_and_copied_runtime_source_are_intrinsically_crosslinked(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    manifest, snapshot = experiment._validated_prepared_snapshot(output)
    mutated = copy.deepcopy(manifest)
    source = cast(dict[str, Any], mutated["source"])
    source[str(experiment.SOURCE_FILES[0])]["sha256"] = "0" * 64
    mutated["source_sha256"] = experiment._source_sha256(source)
    freeze = experiment._read_output_json(output, experiment.FREEZE_RECORD)
    with pytest.raises(experiment.InvalidLCV001Artifact, match="source/runtime-source cross-link differs"):
        experiment._validate_input_manifest(mutated, snapshot, freeze)


def test_preparation_postrename_fsync_failure_is_explicit_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "LCV-001"
    audit = _audit(tmp_path / "audit.json")
    real_fsync = experiment._fsync_directory

    def fail_only_after_publish(path: Path) -> None:
        if path == output.parent and output.exists():
            raise OSError("injected post-rename fsync fault")
        real_fsync(path)

    monkeypatch.setattr(experiment, "_fsync_directory", fail_only_after_publish)
    result = experiment._prepare_at(output, experiment.MM007_ROOT, audit)
    assert result["status"] == "prepared_namespace_committed_durability_unconfirmed"
    assert result["parent_directory_fsync"] is False
    assert result["publication_warnings"] == [
        {"error_type": "OSError", "phase": "postrename_parent_fsync"}
    ]
    experiment._validate_prepared(output)


def test_preparation_success_then_publish_wrapper_exception_is_detected_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "LCV-001"
    audit = _audit(tmp_path / "audit.json")
    real_publish = experiment._rename_noreplace

    def publish_then_raise(source: Path, destination: Path) -> None:
        real_publish(source, destination)
        raise RuntimeError("injected immediately after successful publication")

    monkeypatch.setattr(experiment, "_rename_noreplace", publish_then_raise)
    result = experiment._prepare_at(output, experiment.MM007_ROOT, audit)
    assert result["status"] == "prepared_namespace_committed_durability_unconfirmed"
    assert result["parent_directory_fsync"] is False
    assert result["publication_warnings"] == [
        {"error_type": "RuntimeError", "phase": "publication_return_boundary"}
    ]
    experiment._validate_prepared(output)
    assert list(tmp_path.glob(".LCV-001.prepare-*")) == []


def test_preparation_publish_wrapper_exception_before_rename_cleans_only_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "LCV-001"
    audit = _audit(tmp_path / "audit.json")

    def fail_before_publish(_source: Path, _destination: Path) -> None:
        raise RuntimeError("injected before publication")

    monkeypatch.setattr(experiment, "_rename_noreplace", fail_before_publish)
    with pytest.raises(RuntimeError, match="before publication"):
        experiment._prepare_at(output, experiment.MM007_ROOT, audit)
    assert not output.exists()
    assert list(tmp_path.glob(".LCV-001.prepare-*")) == []


def test_formal_refuses_when_prepared_parent_directory_cannot_be_fsynced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _prepare(tmp_path)
    real_fsync = experiment._fsync_directory

    def fail_output_parent(path: Path) -> None:
        if path == output.parent:
            raise OSError("injected durability failure")
        real_fsync(path)

    monkeypatch.setattr(experiment, "_fsync_directory", fail_output_parent)
    with pytest.raises(experiment.InvalidLCV001Runtime, match="durability could not be confirmed"):
        experiment._run_at(output)
    assert not (output / experiment.FORMAL_START).exists()


def test_public_prepare_and_run_reject_noncanonical_paths(tmp_path: Path) -> None:
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment.prepare(tmp_path / "wrong", experiment.MM007_ROOT, tmp_path / "audit")
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment.run(tmp_path / "wrong")


def test_pre_real_audit_requires_independent_exact_go(tmp_path: Path) -> None:
    _, protocol = experiment._stable_read(experiment.PROTOCOL_DOC)
    _, config = experiment._stable_read(experiment.CONFIG_DOC)
    _, source = experiment._source_snapshot()
    binding = experiment._audit_binding(
        config_sha256=str(config["sha256"]),
        protocol_sha256=str(protocol["sha256"]),
        source_sha256=experiment._source_sha256(source),
    )
    bad: dict[str, Any] = {
        "binding": binding,
        "checks": {},
        "decision": "GO",
        "experiment_id": "LCV-001",
        "independent": False,
        "reviewer": "",
        "schema_version": "lcv001-pre-real-audit-v1",
    }
    with pytest.raises(experiment.InvalidLCV001Artifact):
        experiment._validate_pre_real_audit(bad, binding)
