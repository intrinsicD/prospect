from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

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
        "protocol_version": "1.5.0",
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
        "protocol_version": "1.5.0",
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
    assert preformal._verify_live_bootstrap_custody() == seal

    assurance = seal["assurance"]
    assert isinstance(assurance, dict)
    assurance["external_attestation"] = True
    payload = _canonical(seal)
    with pytest.raises(
        preformal.PreformalEvidenceError,
        match="captured runtime seal is malformed",
    ):
        preformal._verify_live_bootstrap_custody()


def _prepare_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_command: str | None = None,
) -> tuple[Path, list[str], dict[str, Path]]:
    directory = tmp_path / "evidence"
    directory.mkdir()
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

    monkeypatch.setattr(preformal, "REVIEW_PATH", review_path)
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
        expected_environment = (
            preformal._sanitized_environment()
            if specification.role == "qa"
            else _RUNTIME_ENVIRONMENT
        )
        assert environment == expected_environment
        calls.append(specification.name)
        exit_code = 7 if specification.name == failed_command else 0
        return (
            exit_code,
            f"stdout:{specification.name}\n".encode(),
            f"stderr:{specification.name}\n".encode(),
        )

    monkeypatch.setattr(preformal, "_run_command", run_command)
    return (
        directory / preformal.REPORT_NAME,
        calls,
        {
            "runtime_executable": runtime_executable,
            "runtime_seal": runtime_seal_path,
            "development_closure": development_closure,
            "prospective_review": review_path,
        },
    )


def _generate(report_path: Path, inputs: dict[str, Path]) -> Path:
    return preformal.generate_preformal_report(
        report_path,
        runtime_executable=inputs["runtime_executable"],
        runtime_seal=inputs["runtime_seal"],
        development_closure=inputs["development_closure"],
        prospective_review=inputs["prospective_review"],
    )


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
    assert specifications[8].argv[-3:] == (
        "development-evidence",
        "--development-closure",
        str(preformal.DEVELOPMENT_CLOSURE_PATH),
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
    assert variables["PYTHONNOUSERSITE"] == "1"
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
    with pytest.raises(preformal.PreformalEvidenceError, match="identity differs"):
        preformal.verify_preformal_report(report_path)


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
    assert list(report_path.parent.iterdir()) == []


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
        "protocol_version": "1.5.0",
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
