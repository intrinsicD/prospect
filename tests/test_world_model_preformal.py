from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle import experiment as experiment_module
from bench.world_model_lifecycle import operator as operator_module
from bench.world_model_lifecycle import preformal

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


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _runtime_conformance(
    device: str = "cpu",
) -> dict[str, object]:
    return {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "bootstrap-inventory-conformance",
        "device": device,
        "passed": True,
        "inventory_sha256": "1" * 64,
        "conformance_sha256": "2" * 64,
        "fresh_runtime_identity_conformance_sha256": "9" * 64,
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


def _review() -> dict[str, object]:
    return {
        "schema": preformal.REVIEW_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "implementation_files": [],
        "implementation_manifest_sha256": hashlib.sha256(b"[]").hexdigest(),
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
        "protocol_version": "1.10.0",
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
) -> tuple[Path, list[str], dict[str, Path]]:
    directory = tmp_path / "v1.10.0" / "preformal"
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
    development_closure.write_bytes(_canonical({"development": "verified"}))
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
    monkeypatch.setattr(
        preformal,
        "_verified_closure_member_digests",
        lambda path: ({}, "5" * 64, "6" * 64),
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
            b""
            if specification.name
            == "runtime-accepted-closure-evidence"
            else f"stderr:{specification.name}\n".encode()
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
        ("producer_manifest_sha256", "7" * 64),
        ("raw_result_sha256", "8" * 64),
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
    (legacy_root / "preformal-test-report-v1.9.0.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (legacy_root / "preformal-v1.9.0-command-01.stdout.legacy.log").write_bytes(
        b"retained failure evidence"
    )
    bundle_root = legacy_root / "v1.10.0" / "preformal"
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
    assert (legacy_root / "preformal-test-report-v1.9.0.json").is_file()


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
        "protocol_version": "1.10.0",
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


def test_prospective_manifest_binds_v1100_plan_and_runbook() -> None:
    paths = {
        str(row["path"])
        for row in preformal._implementation_files()
    }

    assert {
        "docs/wm001-v1100-confirmation-plan.md",
        "docs/wm001-v1100-operator-runbook.md",
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


def test_captured_descriptor_custody_rejects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "seal.json"
    marker = tmp_path / "seal-completion.json"
    payload = b"sealed\n"
    path.write_bytes(payload)
    os.link(path, marker)
    descriptor = os.open(path, os.O_RDONLY)
    metadata = os.fstat(descriptor)
    identity = preformal._descriptor_identity(metadata)
    monkeypatch.setattr(sys, "_prospect_wm001_runtime_seal_fd", descriptor, raising=False)
    monkeypatch.setattr(sys, "_prospect_wm001_runtime_seal_payload", payload, raising=False)
    monkeypatch.setattr(sys, "_prospect_wm001_runtime_seal_identity", identity, raising=False)
    monkeypatch.setattr(
        sys,
        "_prospect_wm001_runtime_seal_sha256",
        hashlib.sha256(payload).hexdigest(),
        raising=False,
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
        "protocol_version": "1.10.0",
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
    closure_payload = _canonical({"fixture": "closure"})
    closure_path.write_bytes(closure_payload)
    manifest_sha256 = "a" * 64
    result_sha256 = "b" * 64
    closure = {
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "qualification_archive": {
            "members": [
                {
                    "path": "producer/producer-manifest.json",
                    "sha256": manifest_sha256,
                },
                {
                    "path": "producer/result.json",
                    "sha256": result_sha256,
                },
            ],
        },
    }
    report = {
        "schema": (
            "prospect.wm001.development-closure-fresh-reopen.v1"
        ),
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
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
        preformal,
        "_canonical_existing_file",
        lambda path, *, label: path,
    )
    monkeypatch.setattr(
        preformal,
        "_file_identity",
        lambda path, *, label, expected_nlink: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(
        binding,
        "verify_development_closure",
        lambda path: closure,
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
        "protocol_version": "1.10.0",
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

    from bench.world_model_lifecycle import binding

    events: list[str] = []
    custody = {"schema": "captured-runtime"}

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
        "protocol_version": "1.10.0",
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
