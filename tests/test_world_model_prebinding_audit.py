from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from bench.world_model_lifecycle import artifact_audit, preformal

HERE = Path(__file__).resolve().parents[1]
WM001 = HERE / "bench" / "world_model_lifecycle"
PROTOCOL = WM001 / "protocol.json"
SCIENTIFIC_SOURCES = {name: WM001 / name for name in artifact_audit._PREBINDING_SCIENTIFIC_SOURCES}


def _python_row() -> dict[str, object]:
    return {
        "name": "python",
        "version": artifact_audit.platform.python_version(),
        "distribution_sha256": artifact_audit._sha256_file(Path(artifact_audit.sys.executable).resolve()),
        "declared_file_count": 1,
        "editable": False,
    }


def test_independent_preformal_contract_matches_live_producer_contract() -> None:
    producer = preformal.required_commands(
        qa_executable_path=artifact_audit.sys.executable,
        runtime_executable_path=artifact_audit.sys.executable,
        device="cpu",
    )
    completed = subprocess.run(
        (
            "git",
            "ls-files",
            "--",
            "tests/test_epistemic_*.py",
            "tests/test_world_model_*.py",
        ),
        cwd=HERE,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        text=True,
    )
    source = {
        "implementation_files": [
            {
                "path": relative,
                "bytes": (HERE / relative).stat().st_size,
                "sha256": hashlib.sha256(
                    (HERE / relative).read_bytes()
                ).hexdigest(),
            }
            for relative in completed.stdout.splitlines()
        ]
    }
    independent = artifact_audit._preformal_expected_commands(
        qa_executable=producer[0].argv[0],
        runtime_executable=producer[8].argv[0],
        source=source,
        repository_cwd=str(preformal.REPO),
        runtime_seal_path=str(
            preformal.REPO
            / "bench/world_model_lifecycle/results/development/"
            "runtime-seal-v1.10.0.json"
        ),
        development_closure_path=str(
            preformal.DEVELOPMENT_CLOSURE_PATH
        ),
        closure_attempt_path=str(preformal.CLOSURE_ATTEMPT_PATH),
        prospective_review_path=str(preformal.REVIEW_PATH),
        device="cpu",
    )

    expected_inputs = frozenset(
        {
            "closure_attempt_terminal",
            "closure_outer_completion",
            "development_closure",
            "launch_bootstrap",
            "producer_bootstrap",
            "prospective_review",
            "runtime_seal",
        }
    )
    assert artifact_audit._PREFORMAL_INPUT_FIELDS == expected_inputs
    assert preformal._PREFORMAL_INPUT_FIELDS == expected_inputs
    assert independent == tuple(
        (row.name, row.role, row.argv)
        for row in producer
    )


def _request(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "closure"
    root.mkdir(parents=True)
    (root / "module.py").write_bytes(b"VALUE = 7\n")
    nested = root / "data"
    nested.mkdir()
    (nested / "identity.bin").write_bytes(b"\x00WM-001\xff")
    return artifact_audit.build_prebinding_conformance_request(
        PROTOCOL,
        scientific_source_paths=SCIENTIFIC_SOURCES,
        root_paths={"closure": root},
        device="cpu",
        package_rows=[_python_row()],
    )


def _patch_live_packages(
    monkeypatch: pytest.MonkeyPatch,
    request: dict[str, Any],
) -> None:
    expected = copy.deepcopy(request["packages"])
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_live_package_rows",
        lambda: copy.deepcopy(expected),
    )
    runtime = request["runtime"]
    for key, value in runtime["producer_process_environment"].items():
        if key != "PATH":
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_live_python_flags",
        lambda: dict(artifact_audit._PREBINDING_AUDITOR_FLAGS),
    )


def _patch_expensive_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("pendulum", "oscillator"):
        monkeypatch.setattr(
            artifact_audit,
            f"_prebinding_{name}_component",
            lambda selected=name: {
                "identity_sha256": hashlib.sha256(selected.encode("ascii")).hexdigest(),
                "passed": True,
            },
        )


def _components(
    report: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], report["components"])


def test_fixed_environment_and_arithmetic_corpora_pass() -> None:
    pendulum = artifact_audit._prebinding_pendulum_component()
    oscillator = artifact_audit._prebinding_oscillator_component()
    coverage = artifact_audit._prebinding_coverage_component()

    assert pendulum["cases"] == 1024
    assert pendulum["samples_per_task"] == 512
    assert pendulum["passed"] is True
    assert oscillator["cases"] == 512
    assert oscillator["steps_per_case"] == 200
    assert oscillator["passed"] is True
    assert coverage["cases"] == 10
    assert coverage["passed"] is True


def test_active_protocol_seed_universe_has_no_declared_collision() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    audit = artifact_audit._Audit()

    artifact_audit._audit_protocol_seed_contract(audit, protocol)

    assert audit.failed_checks == 0
    assert audit.passed_checks == 1


def test_prebinding_protocol_requires_exact_v19_supersession_lineage(
    tmp_path: Path,
) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["experiment"]["revision"]["superseded_protocol_sha256"] = "0" * 64
    changed = tmp_path / "protocol.json"
    changed.write_bytes(artifact_audit._canonical_json_bytes(protocol) + b"\n")
    request = artifact_audit.build_prebinding_conformance_request(
        changed,
        scientific_source_paths=SCIENTIFIC_SOURCES,
        root_paths={"closure": tmp_path},
        device="cpu",
        package_rows=[_python_row()],
    )

    with pytest.raises(
        artifact_audit._PrebindingConformanceError,
        match="protocol_lineage_mismatch",
    ):
        artifact_audit._prebinding_protocol_component(request["protocol"])


def test_complete_prebinding_request_passes_without_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _patch_live_packages(monkeypatch, request)

    report = artifact_audit.audit_prebinding_conformance(request)

    assert report["schema"] == "prospect.wm001.prebinding-conformance.v1"
    assert report["passed"] is True
    assert all(component["passed"] is True for component in _components(report).values())
    encoded = artifact_audit._canonical_json_bytes(report) + b"\n"
    assert str(tmp_path).encode() not in encoded
    assert b"result.json" not in encoded
    assert b"artifact_root" not in encoded


def test_prebinding_validates_runtime_before_importing_pendulum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    calls: list[str] = []

    def runtime_component(_: object) -> dict[str, object]:
        calls.append("runtime")
        return {
            "identity_sha256": "a" * 64,
            "passed": True,
        }

    def pendulum_component() -> dict[str, object]:
        calls.append("pendulum")
        return {
            "identity_sha256": "b" * 64,
            "passed": True,
        }

    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_runtime_component",
        runtime_component,
    )
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_pendulum_component",
        pendulum_component,
    )
    _patch_expensive_semantics(monkeypatch)
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_pendulum_component",
        pendulum_component,
    )

    artifact_audit.audit_prebinding_conformance(request)

    assert calls == ["runtime", "pendulum"]


def test_changed_complete_root_inventory_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _patch_live_packages(monkeypatch, request)
    _patch_expensive_semantics(monkeypatch)
    root = Path(request["root_inventories"][0]["path"])
    (root / "module.py").write_bytes(b"VALUE = 8\n")

    report = artifact_audit.audit_prebinding_conformance(request)

    assert report["passed"] is False
    assert _components(report)["root_inventories"]["passed"] is False


def test_standard_library_inventory_binds_cache_bytecode_and_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stdlib"
    cache = root / "__pycache__"
    cache.mkdir(parents=True)
    (root / "module.py").write_bytes(b"VALUE = 1\n")
    bytecode = cache / "module.cpython-312.pyc"
    bytecode.write_bytes(b"compiled-before")

    before = artifact_audit._prebinding_root_inventory(
        "standard-library",
        str(root.resolve()),
        kind="standard_library",
    )
    bytecode.write_bytes(b"compiled-after")
    after = artifact_audit._prebinding_root_inventory(
        "standard-library",
        str(root.resolve()),
        kind="standard_library",
    )

    assert before["file_count"] == after["file_count"] == 2
    assert before["inventory_sha256"] != after["inventory_sha256"]


def test_standard_library_inventory_rejects_symlinked_pruned_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stdlib"
    root.mkdir()
    (root / "module.py").write_bytes(b"VALUE = 1\n")
    target = tmp_path / "site-packages-target"
    target.mkdir()
    (root / "site-packages").symlink_to(
        target,
        target_is_directory=True,
    )

    with pytest.raises(
        artifact_audit._PrebindingConformanceError,
        match="root_inventory_non_regular_entry",
    ):
        artifact_audit._prebinding_root_inventory(
            "standard-library",
            str(root.resolve()),
            kind="standard_library",
        )


def test_package_root_inventory_binds_cache_bytecode_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "site-packages"
    cache = root / "__pycache__"
    cache.mkdir(parents=True)
    (root / "module.py").write_bytes(b"VALUE = 1\n")
    (cache / "module.cpython-312.pyc").write_bytes(b"compiled")

    inventory = artifact_audit._prebinding_root_inventory(
        "package-root-0000",
        str(root.resolve()),
        kind="package_root",
    )

    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == len(b"VALUE = 1\ncompiled")


def test_root_inventory_hashes_files_in_global_relative_path_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "site-packages"
    nested = root / "a"
    nested.mkdir(parents=True)
    payloads = {
        "a/nested.py": b"nested\n",
        "z-root.py": b"root\n",
    }
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)

    inventory = artifact_audit._prebinding_root_inventory(
        "package-root-0000",
        str(root.resolve()),
        kind="package_root",
    )
    digest = hashlib.sha256(artifact_audit._PREBINDING_PACKAGE_ROOT_DOMAIN)
    digest.update(b"a\0directory\0")
    for relative in sorted(payloads):
        payload = payloads[relative]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")

    assert inventory["file_count"] == 2
    assert inventory["inventory_sha256"] == digest.hexdigest()


def test_execution_source_manifest_includes_descriptor_launcher(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "bench" / "world_model_lifecycle"
    source_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    execution_sources: dict[str, str] = {}
    for filename in artifact_audit._FORMAL_EXECUTION_SOURCE_FILES:
        payload = f"# {filename}\n".encode()
        (source_root / filename).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "path": (f"bench/world_model_lifecycle/{filename}"),
                "bytes": len(payload),
                "sha256": digest,
            }
        )
        execution_sources[filename] = digest
    source: dict[str, object] = {
        "implementation_files": rows,
        "execution_source_sha256": execution_sources,
    }

    assert (
        artifact_audit._validate_bound_execution_source_manifest(
            tmp_path,
            source,
        )
        == execution_sources
    )

    changed = copy.deepcopy(source)
    changed_sources = cast(
        dict[str, str],
        changed["execution_source_sha256"],
    )
    changed_sources["launch_bootstrap.py"] = "0" * 64
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="differ from the source snapshot",
    ):
        artifact_audit._validate_bound_execution_source_manifest(
            tmp_path,
            changed,
        )


@pytest.mark.parametrize(
    ("mutator", "failed_component"),
    [
        (
            lambda request: request["protocol"].__setitem__(
                "scientific_kernel_sha256",
                "0" * 64,
            ),
            "protocol",
        ),
        (
            lambda request: request["protocol"]["scientific_source_files"][0].__setitem__("sha256", "1" * 64),
            "protocol",
        ),
        (
            lambda request: request["packages"][0].__setitem__(
                "distribution_sha256",
                "2" * 64,
            ),
            "packages",
        ),
        (
            lambda request: request["representative_tensor"].__setitem__(
                "cpu_sha256",
                "3" * 64,
            ),
            "representative_tensor",
        ),
    ],
)
def test_changed_scientific_package_and_determinism_identities_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict[str, Any]], None],
    failed_component: str,
) -> None:
    request = _request(tmp_path)
    live_packages = copy.deepcopy(request["packages"])
    _patch_expensive_semantics(monkeypatch)
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_live_package_rows",
        lambda: copy.deepcopy(live_packages),
    )
    for key, value in request["runtime"]["producer_process_environment"].items():
        if key != "PATH":
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_live_python_flags",
        lambda: dict(artifact_audit._PREBINDING_AUDITOR_FLAGS),
    )
    mutator(request)

    report = artifact_audit.audit_prebinding_conformance(request)

    assert report["passed"] is False
    assert _components(report)[failed_component]["passed"] is False


def test_within_process_tensor_nondeterminism_is_a_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _patch_live_packages(monkeypatch, request)
    _patch_expensive_semantics(monkeypatch)

    def nondeterministic(_: str) -> dict[str, object]:
        raise artifact_audit._PrebindingConformanceError("cpu_tensor_nondeterministic")

    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_representative_tensor_identity",
        nondeterministic,
    )
    report = artifact_audit.audit_prebinding_conformance(request)

    assert report["passed"] is False
    assert _components(report)["representative_tensor"] == {
        "code": "cpu_tensor_nondeterministic",
        "identity_sha256": None,
        "passed": False,
    }


def test_request_must_be_bounded_canonical_json(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(request, indent=2),
        encoding="utf-8",
    )

    report = artifact_audit.audit_prebinding_conformance_file(str(noncanonical))

    assert report["passed"] is False
    assert _components(report)["request"]["code"] == ("request_not_canonical")


def test_report_is_independent_of_equivalent_request_and_root_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _request(tmp_path / "first")
    first_root = Path(first["root_inventories"][0]["path"])
    second_root = tmp_path / "second" / "closure"
    second_root.mkdir(parents=True)
    (second_root / "module.py").write_bytes((first_root / "module.py").read_bytes())
    (second_root / "data").mkdir()
    (second_root / "data" / "identity.bin").write_bytes((first_root / "data" / "identity.bin").read_bytes())
    second = copy.deepcopy(first)
    second["root_inventories"][0]["path"] = str(second_root.resolve())
    _patch_live_packages(monkeypatch, first)
    _patch_expensive_semantics(monkeypatch)

    first_path = tmp_path / "first-request.json"
    second_path = tmp_path / "nested" / "second-request.json"
    second_path.parent.mkdir()
    first_path.write_bytes(artifact_audit.canonical_prebinding_request_bytes(first))
    second_path.write_bytes(artifact_audit.canonical_prebinding_request_bytes(second))

    first_report = artifact_audit.audit_prebinding_conformance_file(str(first_path))
    second_report = artifact_audit.audit_prebinding_conformance_file(str(second_path))

    assert first_report == second_report
    assert first_report["passed"] is True


def test_relative_captured_support_locators_are_reopened_from_request_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = tmp_path / "closure"
    closure.mkdir()
    (closure / "module.py").write_bytes(b"VALUE = 7\n")
    request = artifact_audit.build_prebinding_conformance_request(
        PROTOCOL,
        scientific_source_paths=SCIENTIFIC_SOURCES,
        root_paths={"closure": closure},
        device="cpu",
        support_locator_root=WM001,
        package_rows=[_python_row()],
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    for name, source in SCIENTIFIC_SOURCES.items():
        (capture / name).write_bytes(source.read_bytes())
    request_path = capture / "prebinding-request.json"
    request_path.write_bytes(artifact_audit.canonical_prebinding_request_bytes(request))
    _patch_live_packages(monkeypatch, request)
    _patch_expensive_semantics(monkeypatch)

    report = artifact_audit.audit_prebinding_conformance_file(str(request_path))

    assert report["passed"] is True


def _preformal_v2_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    bytes,
    dict[str, Any],
    dict[str, object],
    dict[str, object],
    dict[str, Any],
]:
    executable = Path(artifact_audit.sys.executable)
    resolved_executable = executable.resolve(strict=True)
    executable_payload = resolved_executable.read_bytes()
    executable_digest = hashlib.sha256(executable_payload).hexdigest()
    executable_identity = {
        "invocation_path": str(executable),
        "invocation_symlink_target": (os.readlink(executable) if executable.is_symlink() else None),
        "resolved_path": str(resolved_executable),
        "bytes": len(executable_payload),
        "sha256": executable_digest,
        "implementation": artifact_audit.platform.python_implementation(),
        "version": artifact_audit.platform.python_version(),
    }
    source_payloads = {
        "bench/world_model_lifecycle/audit_runner.py": b"# audit runner\n",
        "bench/world_model_lifecycle/launch_bootstrap.py": b"# launch\n",
        "bench/world_model_lifecycle/preformal.py": (b"def generate_preformal_report(): ...\n"),
        "bench/world_model_lifecycle/producer_bootstrap.py": b"# producer\n",
        "bench/world_model_lifecycle/protocol.json": b"{}\n",
        "bench/world_model_lifecycle/schemas/raw-result.schema.json": (
            b"{}\n"
        ),
        "tests/test_epistemic_contract.py": b"# epistemic\n",
        "tests/test_world_model_alpha.py": b"# wm001\n",
        "tests/test_world_model_audit_runner.py": b"# runner\n",
        "tests/test_world_model_prebinding_audit.py": b"# prebinding\n",
    }
    reviewed_files = [
        {
            "path": path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content in sorted(source_payloads.items())
    ]
    review: dict[str, object] = {
        "schema": "prospect.wm001.prospective-harness-review.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "implementation_files": reviewed_files,
        "implementation_manifest_sha256": hashlib.sha256(
            artifact_audit._canonical_json_bytes(reviewed_files)
        ).hexdigest(),
        "reviewer": {
            "kind": "independent-adversarial-referee",
            "identifier": "fixture-independent-referee",
        },
        "disposition": "accepted",
        "unresolved_blockers": [],
        "findings": [],
    }
    review_payload = artifact_audit._canonical_json_bytes(review) + b"\n"
    implementation_files = sorted(
        [
            *reviewed_files,
            {
                "path": artifact_audit._PREFORMAL_REVIEW_PATH,
                "bytes": len(review_payload),
                "sha256": hashlib.sha256(review_payload).hexdigest(),
            },
        ],
        key=lambda row: cast(str, row["path"]),
    )
    source: dict[str, Any] = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "implementation_files": implementation_files,
        "execution_source_sha256": {
            "audit_runner.py": hashlib.sha256(
                source_payloads[
                    "bench/world_model_lifecycle/audit_runner.py"
                ]
            ).hexdigest(),
            "launch_bootstrap.py": hashlib.sha256(
                source_payloads[
                    "bench/world_model_lifecycle/launch_bootstrap.py"
                ]
            ).hexdigest(),
            "producer_bootstrap.py": hashlib.sha256(
                source_payloads["bench/world_model_lifecycle/producer_bootstrap.py"]
            ).hexdigest(),
        },
    }
    process_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    qa_environment = {
        **artifact_audit._PREFORMAL_FIXED_ENVIRONMENT,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PATH": "/usr/bin:/bin",
    }
    qa_variables = [{"name": name, "value": value} for name, value in sorted(qa_environment.items())]
    qa_environment_identity = {
        "variables": qa_variables,
        "sha256": hashlib.sha256(artifact_audit._canonical_json_bytes(qa_variables)).hexdigest(),
    }
    runtime_variables = [{"name": name, "value": value} for name, value in sorted(process_environment.items())]
    runtime_environment_identity = {
        "variables": runtime_variables,
        "sha256": hashlib.sha256(artifact_audit._canonical_json_bytes(runtime_variables)).hexdigest(),
    }
    qa_closure: dict[str, object] = {
        "schema": "prospect.wm001.qa-closure.v1",
        "sys_path": ["/qa/site-packages", "/qa/stdlib"],
        "distributions": [
            {
                "name": "prospect",
                "version": "0.1.0",
                "editable": False,
                "declared_file_count": 3,
                "total_bytes": 100,
                "distribution_sha256": "9" * 64,
            }
        ],
    }
    qa_closure["inventory_sha256"] = hashlib.sha256(artifact_audit._canonical_json_bytes(qa_closure)).hexdigest()
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(
        artifact_audit,
        "_LIVE_REPOSITORY_ROOT",
        repository,
    )
    repository_cwd = str(repository)
    standard_library = {"identity": "stdlib"}
    package_roots = [{"identity": "package-root"}]
    package_ownership = {"identity": "ownership"}
    runtime_seal: dict[str, object] = {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "assurance": dict(artifact_audit._ASSURANCE),
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "worktree_clean": True,
        "python": {
            "executable": str(executable),
            "resolved_executable": str(resolved_executable),
            "sha256": executable_digest,
            "version": [
                artifact_audit.sys.version_info.major,
                artifact_audit.sys.version_info.minor,
                artifact_audit.sys.version_info.micro,
            ],
        },
        "required_flags": dict(artifact_audit._PREFORMAL_RUNTIME_FLAGS),
        "process_environment": process_environment,
        "bootstrap_source_sha256": source["execution_source_sha256"]["producer_bootstrap.py"],
        "standard_library": standard_library,
        "package_roots": package_roots,
        "package_ownership": package_ownership,
    }
    runtime_seal_payload = (
        artifact_audit._canonical_json_bytes(runtime_seal) + b"\n"
    )
    development_root = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
    )
    development_root.mkdir(parents=True)
    runtime_seal_path = development_root / "runtime-seal-v1.10.0.json"
    runtime_seal_path.write_bytes(runtime_seal_payload)
    runtime_seal_completion = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "outer-completions"
        / "v1.10"
        / (
            hashlib.sha256(
                str(runtime_seal_path).encode("utf-8")
            ).hexdigest()
            + ".json"
        )
    )
    runtime_seal_completion.parent.mkdir(parents=True)
    os.link(runtime_seal_path, runtime_seal_completion)
    development_closure = {
        "schema": "prospect.wm001.development-closure.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "qualification_archive": {
            "members": [
                {
                    "path": "producer/producer-manifest.json",
                    "bytes": 17,
                    "sha256": "5" * 64,
                },
                {
                    "path": "producer/result.json",
                    "bytes": 19,
                    "sha256": "6" * 64,
                },
            ],
        },
    }
    development_payload = (
        artifact_audit._canonical_json_bytes(development_closure) + b"\n"
    )
    development_path = (
        development_root / "development-closure-v1.10.0.json"
    )
    development_path.write_bytes(development_payload)
    closure_terminal = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.10"
        / "closures"
        / "development-closure-v1.10.0"
        / "operator-attempt.json"
    )
    closure_terminal.parent.mkdir(parents=True)
    closure_terminal_payload = b'{"accepted":true}\n'
    closure_terminal.write_bytes(closure_terminal_payload)
    closure_completion = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "outer-completions"
        / "v1.10"
        / (
            hashlib.sha256(
                str(closure_terminal).encode("utf-8")
            ).hexdigest()
            + ".json"
        )
    )
    closure_completion.parent.mkdir(parents=True, exist_ok=True)
    os.link(closure_terminal, closure_completion)
    closure_terminal_sha256 = hashlib.sha256(
        closure_terminal_payload
    ).hexdigest()
    input_files = {
        "closure_attempt_terminal": {
            "path": str(closure_terminal),
            "bytes": len(closure_terminal_payload),
            "sha256": closure_terminal_sha256,
        },
        "closure_outer_completion": {
            "path": str(closure_completion),
            "bytes": len(closure_terminal_payload),
            "sha256": closure_terminal_sha256,
        },
        "development_closure": {
            "path": str(development_path),
            "bytes": len(development_payload),
            "sha256": hashlib.sha256(development_payload).hexdigest(),
        },
        "launch_bootstrap": {
            "path": (f"{repository_cwd}/bench/world_model_lifecycle/launch_bootstrap.py"),
            **next(
                {
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in implementation_files
                if row["path"] == "bench/world_model_lifecycle/launch_bootstrap.py"
            ),
        },
        "producer_bootstrap": {
            "path": (f"{repository_cwd}/bench/world_model_lifecycle/producer_bootstrap.py"),
            **next(
                {
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in implementation_files
                if row["path"] == "bench/world_model_lifecycle/producer_bootstrap.py"
            ),
        },
        "prospective_review": {
            "path": f"{repository_cwd}/{artifact_audit._PREFORMAL_REVIEW_PATH}",
            "bytes": len(review_payload),
            "sha256": hashlib.sha256(review_payload).hexdigest(),
        },
        "runtime_seal": {
            "path": str(runtime_seal_path),
            "bytes": len(runtime_seal_payload),
            "sha256": hashlib.sha256(runtime_seal_payload).hexdigest(),
        },
    }
    expected_commands = artifact_audit._preformal_expected_commands(
        qa_executable=str(executable),
        runtime_executable=str(executable),
        source=source,
        repository_cwd=repository_cwd,
        runtime_seal_path=cast(str, input_files["runtime_seal"]["path"]),
        development_closure_path=cast(
            str,
            input_files["development_closure"]["path"],
        ),
        closure_attempt_path=str(closure_terminal.parent),
        prospective_review_path=cast(
            str,
            input_files["prospective_review"]["path"],
        ),
        device="cpu",
    )
    commands: list[dict[str, Any]] = []
    log_rows: list[dict[str, object]] = []
    for ordinal, (name, role, argv) in enumerate(
        expected_commands,
        start=1,
    ):
        references: dict[str, dict[str, object]] = {}
        for stream in ("stdout", "stderr"):
            if (
                name == "runtime-accepted-closure-evidence"
                and stream == "stdout"
            ):
                log_payload = (
                    artifact_audit._canonical_json_bytes(
                        {
                            "schema": (
                                "prospect.wm001.preformal-runtime-check.v1"
                            ),
                            "mode": "accepted-closure-evidence",
                            "passed": True,
                            "development_closure_sha256": (
                                input_files[
                                    "development_closure"
                                ]["sha256"]
                            ),
                            "producer_manifest_sha256": "5" * 64,
                            "raw_result_sha256": "6" * 64,
                            "closure_attempt_manifest_sha256": (
                                closure_terminal_sha256
                            ),
                            "closure_outer_completion_sha256": (
                                closure_terminal_sha256
                            ),
                        }
                    )
                    + b"\n"
                )
            elif (
                name == "runtime-accepted-closure-evidence"
                and stream == "stderr"
            ):
                log_payload = b""
            elif (
                name == "runtime-bootstrap-inventory-conformance"
                and stream == "stdout"
            ):
                log_payload = (
                    artifact_audit._canonical_json_bytes(
                    {
                        "schema": (
                            "prospect.wm001.preformal-runtime-check.v1"
                        ),
                        "mode": (
                            "bootstrap-inventory-conformance"
                        ),
                        "device": "cpu",
                        "passed": True,
                        "inventory_sha256": "1" * 64,
                        "conformance_sha256": "2" * 64,
                        "fresh_runtime_identity_conformance_sha256": (
                            "9" * 64
                        ),
                        "restart_runtime_conformance_report_sha256": (
                            "3" * 64
                        ),
                        "restart_runtime_execution_receipt_sha256": (
                            "4" * 64
                        ),
                        "restart_runtime_support_files": [
                            "producer_bootstrap.py",
                            "protocol.json",
                            "schemas/raw-result.schema.json",
                        ],
                        "restart_runtime_repeat_count": 3,
                        "restart_runtime_path_descriptor_equal": (
                            True
                        ),
                        "repeat_count": 3,
                        "path_descriptor_equal": True,
                    }
                    )
                    + b"\n"
                )
            else:
                log_payload = f"{ordinal}:{name}:{stream}\n".encode()
            digest = hashlib.sha256(log_payload).hexdigest()
            filename = f"{artifact_audit._PREFORMAL_LOG_PREFIX}{ordinal:02d}-{name}.{stream}.{digest}.log"
            (tmp_path / filename).write_bytes(log_payload)
            reference = {
                "file": filename,
                "bytes": len(log_payload),
                "sha256": digest,
            }
            references[stream] = reference
            log_rows.append(
                {
                    "path": filename,
                    "bytes": len(log_payload),
                    "sha256": digest,
                }
            )
        commands.append(
            {
                "ordinal": ordinal,
                "name": name,
                "role": role,
                "argv": list(argv),
                "cwd": repository_cwd,
                "environment_sha256": (
                    qa_environment_identity["sha256"] if role == "qa" else runtime_environment_identity["sha256"]
                ),
                "exit_code": 0,
                "passed": True,
                **references,
            }
        )
    git_identity = {
        "commit": source["git_commit"],
        "tree": source["git_tree"],
        "worktree_clean": True,
    }
    generator_identity = next(
        row for row in implementation_files if row["path"] == "bench/world_model_lifecycle/preformal.py"
    )
    report: dict[str, Any] = {
        "schema": "prospect.wm001.preformal-test-report.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "repository_cwd": repository_cwd,
        "device": "cpu",
        "qa_environment": qa_environment_identity,
        "runtime_environment": runtime_environment_identity,
        "git_before": git_identity,
        "git_after": git_identity,
        "qa_executable_before": executable_identity,
        "qa_executable_after": executable_identity,
        "runtime_executable_before": executable_identity,
        "runtime_executable_after": executable_identity,
        "qa_closure_before": qa_closure,
        "qa_closure_after": qa_closure,
        "runtime_seal": runtime_seal,
        "prospective_review": review,
        "input_files_before": input_files,
        "input_files_after": input_files,
        "generator_source_before": generator_identity,
        "generator_source_after": generator_identity,
        "identities_stable": True,
        "commands": commands,
        "all_pass": True,
    }
    payload = artifact_audit._canonical_json_bytes(report) + b"\n"
    source.update(
        {
            "test_report_file": artifact_audit._PREFORMAL_REPORT_NAME,
            "test_report_bytes": len(payload),
            "test_report_sha256": hashlib.sha256(payload).hexdigest(),
            "test_log_files": log_rows,
        }
    )
    (tmp_path / artifact_audit._PREFORMAL_REPORT_NAME).write_bytes(payload)
    dependencies: dict[str, object] = {
        "lockfile_sha256": "4" * 64,
        "python_executable": str(executable),
        "python_executable_sha256": executable_digest,
        "standard_library": standard_library,
        "package_roots": package_roots,
        "package_ownership": package_ownership,
    }
    runtime: dict[str, object] = {
        "platform": "fixture-platform",
        "machine": "fixture-machine",
        "device": "cpu",
        "python_flags": dict(artifact_audit._PREFORMAL_RUNTIME_FLAGS),
        "process_environment": process_environment,
        "accelerator": None,
        "thread_count": 1,
        "interop_thread_count": 1,
        "cuda_runtime": None,
        "cuda_driver": None,
        "cublas_workspace_config": None,
    }
    return payload, source, dependencies, runtime, report


def _rewrite_preformal_v2_report(
    tmp_path: Path,
    report: dict[str, Any],
    source: dict[str, Any],
) -> bytes:
    payload = artifact_audit._canonical_json_bytes(report) + b"\n"
    source["test_report_bytes"] = len(payload)
    source["test_report_sha256"] = hashlib.sha256(payload).hexdigest()
    (tmp_path / artifact_audit._PREFORMAL_REPORT_NAME).write_bytes(payload)
    return payload


def _replace_preformal_v2_log(
    tmp_path: Path,
    *,
    report: dict[str, Any],
    source: dict[str, Any],
    command_index: int,
    stream: str,
    payload: bytes,
) -> None:
    row = report["commands"][command_index]
    previous = row[stream]
    previous_name = previous["file"]
    digest = hashlib.sha256(payload).hexdigest()
    filename = (
        f"{artifact_audit._PREFORMAL_LOG_PREFIX}"
        f"{row['ordinal']:02d}-{row['name']}.{stream}.{digest}.log"
    )
    (tmp_path / filename).write_bytes(payload)
    (tmp_path / previous_name).unlink()
    replacement = {
        "file": filename,
        "bytes": len(payload),
        "sha256": digest,
    }
    row[stream] = replacement
    source_row = next(
        item
        for item in source["test_log_files"]
        if item["path"] == previous_name
    )
    source_row.update(
        {
            "path": filename,
            "bytes": len(payload),
            "sha256": digest,
        }
    )


def test_v2_machine_test_receipt_reopens_exact_command_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, source, dependencies, runtime, _ = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )

    artifact_audit._validate_preformal_test_report_v2(
        payload,
        root=tmp_path,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
    )


def test_v2_machine_test_receipt_rejects_noncanonical_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    report["repository_cwd"] = str(tmp_path)
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="identity, repository, or status",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_repository_symlink_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    alias = tmp_path / "repository-alias"
    alias.symlink_to(Path(report["repository_cwd"]), target_is_directory=True)
    report["repository_cwd"] = str(alias)
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="identity, repository, or status",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


@pytest.mark.parametrize(
    "field",
    ["development_closure", "runtime_seal"],
)
def test_v2_machine_test_receipt_rejects_noncanonical_live_input_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    report["input_files_before"][field]["path"] = str(
        tmp_path / f"aliased-{field}.json"
    )
    report["input_files_after"] = copy.deepcopy(
        report["input_files_before"]
    )
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="runtime seal or development closure is not canonical",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_equal_bytes_on_distinct_two_link_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, source, dependencies, runtime, report = (
        _preformal_v2_fixture(tmp_path, monkeypatch)
    )
    terminal = Path(
        report["input_files_before"]["closure_attempt_terminal"]["path"]
    )
    completion = Path(
        report["input_files_before"]["closure_outer_completion"]["path"]
    )
    terminal_payload = terminal.read_bytes()
    completion.unlink()
    completion.write_bytes(terminal_payload)
    os.link(terminal, tmp_path / "terminal-extra-link.json")
    os.link(completion, tmp_path / "completion-extra-link.json")

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="same-inode outer-finalized",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_three_link_closure_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, source, dependencies, runtime, report = (
        _preformal_v2_fixture(tmp_path, monkeypatch)
    )
    terminal = Path(
        report["input_files_before"]["closure_attempt_terminal"]["path"]
    )
    os.link(terminal, tmp_path / "third-closure-link.json")

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="same-inode outer-finalized",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema", "wrong"),
        ("mode", "development-evidence"),
        ("passed", False),
        ("passed", 1),
        ("development_closure_sha256", "f" * 64),
        ("producer_manifest_sha256", "f" * 64),
        ("raw_result_sha256", "f" * 64),
        ("closure_attempt_manifest_sha256", "f" * 64),
        ("closure_outer_completion_sha256", "f" * 64),
    ],
)
def test_v2_machine_test_receipt_rejects_command9_semantic_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    row = report["commands"][8]
    value = json.loads((tmp_path / row["stdout"]["file"]).read_bytes())
    value[field] = replacement
    _replace_preformal_v2_log(
        tmp_path,
        report=report,
        source=source,
        command_index=8,
        stream="stdout",
        payload=artifact_audit._canonical_json_bytes(value) + b"\n",
    )
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="accepted-closure",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "noncanonical"])
def test_v2_machine_test_receipt_rejects_command9_shape_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    row = report["commands"][8]
    value = json.loads((tmp_path / row["stdout"]["file"]).read_bytes())
    if mutation == "missing":
        del value["raw_result_sha256"]
        replacement = artifact_audit._canonical_json_bytes(value) + b"\n"
    elif mutation == "extra":
        value["extra"] = True
        replacement = artifact_audit._canonical_json_bytes(value) + b"\n"
    else:
        replacement = json.dumps(value, indent=2).encode("utf-8") + b"\n"
    _replace_preformal_v2_log(
        tmp_path,
        report=report,
        source=source,
        command_index=8,
        stream="stdout",
        payload=replacement,
    )
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="accepted-closure",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_command9_nonempty_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    _replace_preformal_v2_log(
        tmp_path,
        report=report,
        source=source,
        command_index=8,
        stream="stderr",
        payload=b"unexpected diagnostic\n",
    )
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="accepted-closure",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_changed_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    report["commands"][0]["argv"].append("--quiet")
    payload = _rewrite_preformal_v2_report(
        tmp_path,
        report,
        source,
    )

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="differs from its fixed contract",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_reordered_binding_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, source, dependencies, runtime, _ = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    source["test_log_files"][0], source["test_log_files"][1] = (
        source["test_log_files"][1],
        source["test_log_files"][0],
    )

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="references differ",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_tampered_or_extra_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, source, dependencies, runtime, _ = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    first_log = tmp_path / source["test_log_files"][0]["path"]
    first_payload = first_log.read_bytes()
    first_log.write_bytes(first_payload + b"tampered")
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="log bytes changed",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )

    first_log.write_bytes(first_payload)
    (tmp_path / "preformal-unbound.log").write_bytes(b"extra")
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="missing or extra",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_has_ten_commands_and_twenty_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, source, dependencies, runtime, report = (
        _preformal_v2_fixture(tmp_path, monkeypatch)
    )

    artifact_audit._validate_preformal_test_report_v2(
        payload,
        root=tmp_path,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
    )

    assert len(report["commands"]) == 10
    assert [row["role"] for row in report["commands"]] == [
        "qa",
        "qa",
        "qa",
        "qa",
        "qa",
        "qa",
        "qa",
        "qa",
        "runtime",
        "runtime",
    ]
    assert len(source["test_log_files"]) == 20


def test_v2_machine_test_receipt_rejects_qa_closure_digest_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    report["qa_closure_before"]["distributions"][0]["version"] = "changed"
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="QA closure",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_qa_without_installed_prospect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    distribution = report["qa_closure_before"]["distributions"][0]
    distribution["name"] = "pytest"
    report["qa_closure_before"]["inventory_sha256"] = hashlib.sha256(
        artifact_audit._canonical_json_bytes(
            {
                key: value
                for key, value in report["qa_closure_before"].items()
                if key != "inventory_sha256"
            }
        )
    ).hexdigest()
    report["qa_closure_after"] = copy.deepcopy(report["qa_closure_before"])
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="QA closure",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_review_manifest_self_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    report["prospective_review"]["implementation_files"].pop()
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="accepted exact-source review",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def test_v2_machine_test_receipt_rejects_qa_runtime_role_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    report["commands"][0]["role"] = "runtime"
    report["commands"][0]["environment_sha256"] = report["runtime_environment"]["sha256"]
    payload = _rewrite_preformal_v2_report(tmp_path, report, source)

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="differs from its fixed contract",
    ):
        artifact_audit._validate_preformal_test_report_v2(
            payload,
            root=tmp_path,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def _preflight_package_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str | None = None,
) -> tuple[Path, bytes, bytes, list[bytes]]:
    report_payload = b'{"preformal":"bound"}\n'
    closure_payload = b'{"closure":"bound"}\n'
    (tmp_path / artifact_audit._PREFORMAL_REPORT_NAME).write_bytes(
        report_payload
    )
    (tmp_path / "development-closure-v1.10.0.json").write_bytes(
        closure_payload
    )
    audit_execution = {
        "restart_runtime_conformance_report_sha256": "5" * 64,
        "restart_runtime_execution_receipt_sha256": "6" * 64,
        "restart_runtime_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "restart_runtime_repeat_count": 3,
        "restart_runtime_path_descriptor_equal": True,
    }
    development = {
        "closure_file": "development-closure-v1.10.0.json",
        "closure_sha256": hashlib.sha256(
            closure_payload
        ).hexdigest(),
        "producer_manifest_sha256": "7" * 64,
        "raw_result_sha256": "8" * 64,
    }
    accepted = {
        "development_closure_sha256": development["closure_sha256"],
        "producer_manifest_sha256": development[
            "producer_manifest_sha256"
        ],
        "raw_result_sha256": development["raw_result_sha256"],
    }
    runtime_conformance = {
        "conformance_sha256": hashlib.sha256(
            artifact_audit._canonical_json_bytes(audit_execution)
        ).hexdigest(),
        **audit_execution,
    }
    if mutation == "accepted-producer":
        accepted["producer_manifest_sha256"] = "f" * 64
    elif mutation == "accepted-raw":
        accepted["raw_result_sha256"] = "f" * 64
    elif mutation == "runtime-conformance":
        runtime_conformance["conformance_sha256"] = "f" * 64
    elif mutation == "runtime-restart":
        runtime_conformance[
            "restart_runtime_conformance_report_sha256"
        ] = "f" * 64
    binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v9",
        "experiment_id": "WM-001",
        "assurance": dict(artifact_audit._ASSURANCE),
        "protocol": {"version": "1.10.0"},
        "source": {
            "test_report_file": artifact_audit._PREFORMAL_REPORT_NAME,
        },
        "dependencies": {},
        "runtime": {},
        "development_qualification": development,
        "audit_execution": audit_execution,
    }
    binding_path = tmp_path / "formal-binding.json"
    binding_payload = (
        artifact_audit._canonical_json_bytes(binding) + b"\n"
    )
    binding_path.write_bytes(binding_payload)
    monkeypatch.setattr(
        artifact_audit,
        "_validate_preformal_test_report_v2",
        lambda *_args, **_kwargs: (
            {"runtime_seal": {}},
            runtime_conformance,
            accepted,
        ),
    )
    qualification_calls: list[bytes] = []
    monkeypatch.setattr(
        artifact_audit,
        "_validate_development_qualification",
        lambda payload, **_kwargs: qualification_calls.append(payload),
    )
    return (
        binding_path,
        binding_payload,
        closure_payload,
        qualification_calls,
    )


def test_exact_formal_input_preflight_composes_and_binds_all_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        binding_path,
        binding_payload,
        closure_payload,
        qualification_calls,
    ) = (
        _preflight_package_fixture(tmp_path, monkeypatch)
    )

    receipt = artifact_audit.preflight_formal_input_package(
        binding_path
    )

    assert receipt["passed"] is True
    assert receipt["binding_sha256"] == hashlib.sha256(
        binding_payload
    ).hexdigest()
    assert receipt["development_closure_sha256"] == hashlib.sha256(
        closure_payload
    ).hexdigest()
    assert qualification_calls == [closure_payload]


@pytest.mark.parametrize(
    "mutation",
    [
        "accepted-producer",
        "accepted-raw",
        "runtime-conformance",
        "runtime-restart",
    ],
)
def test_exact_formal_input_preflight_rejects_cross_link_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    binding_path, _, _, _ = _preflight_package_fixture(
        tmp_path,
        monkeypatch,
        mutation=mutation,
    )

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="differ from their sealed preformal rehearsal",
    ):
        artifact_audit.preflight_formal_input_package(binding_path)


def test_full_result_runtime_is_bound_field_for_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    for name, value in process_environment.items():
        if name != "PATH":
            monkeypatch.setenv(name, value)
    shared_runtime = {
        "platform": "test-platform",
        "machine": "test-machine",
        "device": "cpu",
        "accelerator": None,
        "deterministic_algorithms": True,
        "thread_count": 3,
        "interop_thread_count": 2,
        "cuda_runtime": None,
        "cuda_driver": None,
        "cublas_workspace_config": None,
    }
    runtime = {
        **shared_runtime,
        "python_flags": dict(artifact_audit._PREBINDING_PRODUCER_FLAGS),
        "process_environment": process_environment,
    }
    execution = copy.deepcopy(runtime)
    standard_library = {
        "path": str(tmp_path / "stdlib"),
        "semantics_id": "prospect.wm001.standard-library.v2",
        "file_count": 5,
        "directory_count": 2,
        "total_bytes": 50,
        "inventory_sha256": "1" * 64,
    }
    package_root = {
        "path": str(tmp_path / "site-packages"),
        "semantics_id": "prospect.wm001.package-root.v2",
        "file_count": 7,
        "directory_count": 3,
        "total_bytes": 70,
        "inventory_sha256": "2" * 64,
    }
    package_ownership = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": package_root["path"],
        "file_count": 7,
        "directory_count": 3,
        "shared_file_count": 0,
        "identity_sha256": "3" * 64,
    }
    packages = [_python_row()]
    dependencies = {
        "python_executable": artifact_audit.sys.executable,
        "python_executable_sha256": artifact_audit._sha256_file(Path(artifact_audit.sys.executable).resolve()),
        "standard_library": standard_library,
        "package_roots": [package_root],
        "package_ownership": package_ownership,
        "packages": packages,
    }

    monkeypatch.setattr(
        artifact_audit,
        "_live_runtime_identity",
        lambda device: dict(shared_runtime),
    )
    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_live_python_flags",
        lambda: dict(artifact_audit._PREBINDING_AUDITOR_FLAGS),
    )

    def root_inventory(
        identifier: str,
        raw_path: object,
        *,
        kind: str,
    ) -> dict[str, object]:
        expected = standard_library if kind == "standard_library" else package_root
        assert raw_path == expected["path"]
        return {
            "id": identifier,
            "kind": kind,
            "semantics_id": expected["semantics_id"],
            "file_count": expected["file_count"],
            "directory_count": expected["directory_count"],
            "total_bytes": expected["total_bytes"],
            "inventory_sha256": expected["inventory_sha256"],
        }

    monkeypatch.setattr(
        artifact_audit,
        "_prebinding_root_inventory",
        root_inventory,
    )
    monkeypatch.setattr(
        artifact_audit,
        "_live_bound_package_rows",
        lambda rows: copy.deepcopy(packages),
    )
    monkeypatch.setattr(
        artifact_audit,
        "_live_package_ownership",
        lambda _root: copy.deepcopy(package_ownership),
    )

    audit = artifact_audit._Audit()
    assert (
        artifact_audit._audit_formal_runtime_binding(
            audit,
            runtime=runtime,
            dependencies=dependencies,
            execution=execution,
        )
        == "cpu"
    )
    assert audit.failed_checks == 0

    tampered_execution = copy.deepcopy(execution)
    tampered_execution["machine"] = "different-machine"
    tampered = artifact_audit._Audit()
    artifact_audit._audit_formal_runtime_binding(
        tampered,
        runtime=runtime,
        dependencies=dependencies,
        execution=tampered_execution,
    )
    assert "formal_result_runtime_binding_mismatch" in {finding["code"] for finding in tampered.findings}


def test_restart_restore_runtime_is_reopened_and_bound_to_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replicate_id = "wm001-formal-123"
    package_root = tmp_path / "site-packages"
    package_root.mkdir()
    package_root_inventory = {
        "path": str(package_root),
        "semantics_id": "prospect.wm001.package-root.v2",
        "file_count": 1,
        "directory_count": 0,
        "total_bytes": 7,
        "inventory_sha256": "1" * 64,
    }
    standard_library_root = tmp_path / "stdlib"
    standard_library_root.mkdir()
    standard_library = {
        "path": str(standard_library_root),
        "semantics_id": "prospect.wm001.standard-library.v2",
        "file_count": 2,
        "directory_count": 0,
        "total_bytes": 11,
        "inventory_sha256": "2" * 64,
    }
    package_ownership = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(package_root),
        "file_count": 1,
        "directory_count": 0,
        "shared_file_count": 0,
        "identity_sha256": "4" * 64,
    }
    bootstrap_payload = b"raise SystemExit(main())\n"
    bootstrap_snapshot = tmp_path / "source" / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"
    bootstrap_snapshot.parent.mkdir(parents=True)
    bootstrap_snapshot.write_bytes(bootstrap_payload)
    bootstrap_digest = hashlib.sha256(bootstrap_payload).hexdigest()
    executable_digest = artifact_audit._sha256_file(Path(artifact_audit.sys.executable).resolve())
    process_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    flags = dict(artifact_audit._PREBINDING_PRODUCER_FLAGS)
    runtime_seal_digest = "3" * 64
    execution = {
        "python_executable": artifact_audit.sys.executable,
        "python_executable_sha256": executable_digest,
        "python_version": artifact_audit.platform.python_version(),
        "python_flags": flags,
        "process_environment": process_environment,
        "package_roots": [package_root_inventory],
        "package_ownership": package_ownership,
        "standard_library": standard_library,
        "runtime_seal_sha256": runtime_seal_digest,
        "runtime_seal_descriptor_custody": True,
        "producer_bootstrap_sha256": bootstrap_digest,
        "bootstrap_descriptor_custody": True,
        "deterministic_algorithms": True,
    }
    binding_runtime = copy.deepcopy(execution)
    dependencies = {
        "python_executable": artifact_audit.sys.executable,
        "python_executable_sha256": executable_digest,
        "package_roots": [package_root_inventory],
        "package_ownership": package_ownership,
        "standard_library": standard_library,
    }
    source = {
        "implementation_files": [
            {
                "path": ("bench/world_model_lifecycle/producer_bootstrap.py"),
                "bytes": len(bootstrap_payload),
                "sha256": bootstrap_digest,
            }
        ]
    }
    body = {
        "schema": "prospect.wm001.restart-runtime.v2",
        "python_executable": artifact_audit.sys.executable,
        "python_executable_sha256": executable_digest,
        "python_version": artifact_audit.platform.python_version(),
        "python_flags": flags,
        "process_environment": process_environment,
        "package_root": str(package_root),
        "package_root_inventory": package_root_inventory,
        "package_ownership": package_ownership,
        "standard_library": standard_library,
        "runtime_seal_sha256": runtime_seal_digest,
        "runtime_seal_descriptor_custody": True,
        "bootstrap_source_sha256": bootstrap_digest,
        "bootstrap_descriptor_custody": True,
        "deterministic_algorithms": True,
    }
    payload = artifact_audit._canonical_json_bytes(body) + b"\n"
    filename = f"{replicate_id}-restore-runtime.json"
    (tmp_path / filename).write_bytes(payload)
    reference = {
        **body,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "filename": filename,
    }

    # Reproduce the exact development-audit topology that retired v1.6: the
    # captured auditor has no ambient package siblings, and formal source
    # custody is not available. The explicit captured digest is sufficient.
    empty_capture = tmp_path / "captured-auditor"
    empty_capture.mkdir()
    monkeypatch.setattr(artifact_audit, "HERE", empty_capture)
    artifact_audit._validate_restart_restore_runtime(
        root=tmp_path,
        reference=reference,
        replicate_id=replicate_id,
        execution=execution,
        producer_bootstrap_sha256=bootstrap_digest,
        expected_branch="development",
        binding_runtime=None,
        dependencies=None,
        source=None,
    )

    # The formal branch must additionally prove equality with the retained
    # source snapshot and its implementation-manifest row.
    artifact_audit._validate_restart_restore_runtime(
        root=tmp_path,
        reference=reference,
        replicate_id=replicate_id,
        execution=execution,
        producer_bootstrap_sha256=bootstrap_digest,
        expected_branch="formal",
        binding_runtime=binding_runtime,
        dependencies=dependencies,
        source=source,
    )

    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="captured producer bootstrap differs",
    ):
        artifact_audit._validate_restart_restore_runtime(
            root=tmp_path,
            reference=reference,
            replicate_id=replicate_id,
            execution=execution,
            producer_bootstrap_sha256="f" * 64,
            expected_branch="formal",
            binding_runtime=binding_runtime,
            dependencies=dependencies,
            source=source,
        )

    changed_body = {
        **body,
        "package_root": str(tmp_path / "other-root"),
    }
    changed_payload = artifact_audit._canonical_json_bytes(changed_body) + b"\n"
    (tmp_path / filename).write_bytes(changed_payload)
    changed_reference = {
        **changed_body,
        "bytes": len(changed_payload),
        "sha256": hashlib.sha256(changed_payload).hexdigest(),
        "filename": filename,
    }
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="differs from the parent",
    ):
        artifact_audit._validate_restart_restore_runtime(
            root=tmp_path,
            reference=changed_reference,
            replicate_id=replicate_id,
            execution=execution,
            producer_bootstrap_sha256=bootstrap_digest,
            expected_branch="formal",
            binding_runtime=binding_runtime,
            dependencies=dependencies,
            source=source,
        )


def _development_qualification_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: dict[str, Any],
    dependencies: dict[str, object],
    runtime: dict[str, object],
    preformal_runtime_seal: Mapping[str, object],
) -> tuple[
    bytes,
    dict[str, object],
    dict[str, object],
    str,
    bytes,
]:
    execution_sources = cast(
        dict[str, str], source["execution_source_sha256"]
    )
    implementation_by_path = {
        row["path"]: row
        for row in cast(
            list[dict[str, object]], source["implementation_files"]
        )
    }
    producer_bootstrap_row = implementation_by_path[
        "bench/world_model_lifecycle/producer_bootstrap.py"
    ]
    protocol_row = implementation_by_path[
        "bench/world_model_lifecycle/protocol.json"
    ]
    raw_schema_row = implementation_by_path[
        "bench/world_model_lifecycle/schemas/raw-result.schema.json"
    ]
    runtime_seal_payload = (
        artifact_audit._canonical_json_bytes(preformal_runtime_seal) + b"\n"
    )
    runtime_seal_sha256 = hashlib.sha256(runtime_seal_payload).hexdigest()
    runtime_seal_member = "evidence/producer-runtime-seal.json"
    monkeypatch.setattr(
        artifact_audit,
        "_verify_development_qualification_archive",
        lambda *_args, **_kwargs: {
            runtime_seal_member: runtime_seal_payload,
        },
    )
    audit_bootstrap_sha256 = "6" * 64
    bound_audit_execution = {
        "bootstrap_source_sha256": audit_bootstrap_sha256,
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
    closure_source = {
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "worktree_clean": True,
        "dependency_lock_sha256": "4" * 64,
        "producer_bootstrap_sha256": execution_sources["producer_bootstrap.py"],
        "launch_bootstrap_sha256": execution_sources["launch_bootstrap.py"],
        "runner_source_sha256": execution_sources["audit_runner.py"],
        "auditor_source_sha256": (artifact_audit._AUDITOR_SOURCE_SHA256),
    }
    producer_execution = {
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "worktree_clean": True,
        "dependency_lock_sha256": dependencies["lockfile_sha256"],
        "python_executable": dependencies["python_executable"],
        "python_executable_sha256": dependencies["python_executable_sha256"],
        "python_version": "3.12.9",
        "platform": runtime["platform"],
        "machine": runtime["machine"],
        "device": runtime["device"],
        "python_flags": runtime["python_flags"],
        "process_environment": runtime["process_environment"],
        "accelerator": runtime["accelerator"],
        "thread_count": runtime["thread_count"],
        "interop_thread_count": runtime["interop_thread_count"],
        "cuda_runtime": runtime["cuda_runtime"],
        "cuda_driver": runtime["cuda_driver"],
        "cublas_workspace_config": runtime["cublas_workspace_config"],
        "deterministic_algorithms": True,
        "runtime_seal_sha256": runtime_seal_sha256,
        "runtime_seal_descriptor_custody": True,
        "producer_bootstrap_sha256": execution_sources["producer_bootstrap.py"],
        "bootstrap_descriptor_custody": True,
        "package_roots": dependencies["package_roots"],
        "standard_library": dependencies["standard_library"],
    }
    producer_custody = {
        "runtime_seal_member": runtime_seal_member,
        "runtime_seal_sha256": producer_execution["runtime_seal_sha256"],
        "producer_bootstrap_member": "evidence/producer-bootstrap.py",
        "producer_bootstrap_sha256": execution_sources["producer_bootstrap.py"],
        "launch_bootstrap_member": "evidence/launch-bootstrap.py",
        "launch_bootstrap_sha256": execution_sources["launch_bootstrap.py"],
        "package_ownership": dependencies["package_ownership"],
    }
    audit_execution = {
        "receipt_sha256": "8" * 64,
        "runtime_manifest_sha256": "9" * 64,
        "invocation_manifest_sha256": "a" * 64,
        "stderr_sha256": "b" * 64,
        "bootstrap_sha256": audit_bootstrap_sha256,
        "runner_source_sha256": execution_sources["audit_runner.py"],
        "auditor_source_sha256": (artifact_audit._AUDITOR_SOURCE_SHA256),
        "support_files": [
            {
                "path": "producer_bootstrap.py",
                "bytes": producer_bootstrap_row["bytes"],
                "sha256": producer_bootstrap_row["sha256"],
            },
            {
                "path": "protocol.json",
                "bytes": protocol_row["bytes"],
                "sha256": protocol_row["sha256"],
            },
            {
                "path": "schemas/raw-result.schema.json",
                "bytes": raw_schema_row["bytes"],
                "sha256": raw_schema_row["sha256"],
            },
        ],
        "source_mode": "descriptor",
    }
    role_members = {
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "result_qualification_member": ("evidence/development-result-qualification.json"),
        "independent_audit_member": "evidence/independent-audit.json",
        "audit_reproduction_member": "evidence/audit-reproduction.json",
        "audit_runtime_manifest_member": (
            f"evidence/development-audit-runtime-{audit_execution['runtime_manifest_sha256'][:16]}.json"
        ),
        "audit_invocation_manifest_member": (
            f"evidence/development-audit-invocation-{audit_execution['invocation_manifest_sha256'][:16]}.json"
        ),
        "audit_stderr_member": (f"evidence/development-audit-stderr-{audit_execution['stderr_sha256'][:16]}.log"),
    }
    member_rows = [
        {
            "path": path,
            "bytes": (
                len(runtime_seal_payload)
                if path == runtime_seal_member
                else index + 1
            ),
            "sha256": (
                runtime_seal_sha256
                if path == runtime_seal_member
                else f"{index + 5:064x}"
            ),
        }
        for index, path in enumerate(
            sorted({*role_members.values(), runtime_seal_member})
        )
    ]
    digests = {row["path"]: row["sha256"] for row in member_rows}
    archive = {
        "format": "ustar-uncompressed-v1",
        "file": "development-qualification-9999999999999999.tar",
        "canonical_path": (
            "bench/world_model_lifecycle/results/development/development-qualification-9999999999999999.tar"
        ),
        "bytes": 10_240,
        "sha256": "9" * 64,
        "members": member_rows,
    }
    closure = {
        "schema": "prospect.wm001.development-closure.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "source": closure_source,
        "producer_root": ("/repo/bench/world_model_lifecycle/results/development/run"),
        **role_members,
        "producer_execution": producer_execution,
        "producer_custody": producer_custody,
        "audit_execution": audit_execution,
        "qualification_archive": archive,
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    payload = artifact_audit._canonical_json_bytes(closure) + b"\n"
    block = {
        "closure_schema": closure["schema"],
        "closure_file": (f"development-closure-{hashlib.sha256(payload).hexdigest()[:16]}.json"),
        "closure_bytes": len(payload),
        "closure_sha256": hashlib.sha256(payload).hexdigest(),
        "qualification_archive_file": archive["file"],
        "qualification_archive_path": archive["canonical_path"],
        "qualification_archive_bytes": archive["bytes"],
        "qualification_archive_sha256": archive["sha256"],
        "qualification_archive_members_sha256": hashlib.sha256(
            artifact_audit._canonical_json_bytes({"members": member_rows})
        ).hexdigest(),
        "producer_manifest_sha256": digests[role_members["producer_manifest_member"]],
        "raw_result_sha256": digests[role_members["raw_result_member"]],
        "result_qualification_sha256": digests[role_members["result_qualification_member"]],
        "independent_audit_sha256": digests[role_members["independent_audit_member"]],
        "audit_reproduction_sha256": digests[role_members["audit_reproduction_member"]],
        "audit_runtime_manifest_sha256": digests[role_members["audit_runtime_manifest_member"]],
        "audit_invocation_manifest_sha256": digests[role_members["audit_invocation_manifest_member"]],
        "audit_stderr_sha256": digests[role_members["audit_stderr_member"]],
        "source_identity_sha256": hashlib.sha256(artifact_audit._canonical_json_bytes(closure_source)).hexdigest(),
        "producer_execution_identity_sha256": hashlib.sha256(
            artifact_audit._canonical_json_bytes(producer_execution)
        ).hexdigest(),
        "producer_custody_identity_sha256": hashlib.sha256(
            artifact_audit._canonical_json_bytes(producer_custody)
        ).hexdigest(),
        "audit_execution_identity_sha256": hashlib.sha256(
            artifact_audit._canonical_json_bytes(audit_execution)
        ).hexdigest(),
        "git_commit": closure_source["git_commit"],
        "git_tree": closure_source["git_tree"],
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }

    return (
        payload,
        block,
        bound_audit_execution,
        runtime_seal_member,
        runtime_seal_payload,
    )


def test_development_qualification_is_linked_field_for_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_sources = {
        "audit_runner.py": "1" * 64,
        "producer_bootstrap.py": "2" * 64,
        "launch_bootstrap.py": "3" * 64,
    }
    source: dict[str, Any] = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "execution_source_sha256": execution_sources,
        "implementation_files": [
            {
                "path": "bench/world_model_lifecycle/producer_bootstrap.py",
                "bytes": 303,
                "sha256": execution_sources["producer_bootstrap.py"],
            },
            {
                "path": "bench/world_model_lifecycle/protocol.json",
                "bytes": 101,
                "sha256": "c" * 64,
            },
            {
                "path": (
                    "bench/world_model_lifecycle/schemas/"
                    "raw-result.schema.json"
                ),
                "bytes": 202,
                "sha256": "d" * 64,
            },
        ],
    }
    dependencies: dict[str, object] = {
        "lockfile_sha256": "4" * 64,
        "python_executable": "/venv/bin/python",
        "python_executable_sha256": "5" * 64,
        "package_roots": [{"path": "/venv/site-packages"}],
        "standard_library": {"path": "/stdlib"},
        "package_ownership": {"semantics_id": "fixture"},
    }
    runtime: dict[str, object] = {
        "platform": "fixture-platform",
        "machine": "fixture-machine",
        "device": "cpu",
        "python_flags": {"isolated": 1},
        "process_environment": {"LC_ALL": "C.UTF-8"},
        "accelerator": None,
        "thread_count": 1,
        "interop_thread_count": 1,
        "cuda_runtime": None,
        "cuda_driver": None,
        "cublas_workspace_config": None,
    }
    preformal_runtime_seal = {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.10.0",
        "assurance": dict(artifact_audit._ASSURANCE),
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "worktree_clean": True,
        "python": {
            "executable": dependencies["python_executable"],
            "resolved_executable": "/venv/bin/python3.12",
            "sha256": dependencies["python_executable_sha256"],
            "version": [3, 12, 9],
        },
        "required_flags": runtime["python_flags"],
        "process_environment": runtime["process_environment"],
        "bootstrap_source_sha256": execution_sources[
            "producer_bootstrap.py"
        ],
        "standard_library": dependencies["standard_library"],
        "package_roots": dependencies["package_roots"],
        "package_ownership": dependencies["package_ownership"],
    }
    (
        payload,
        block,
        bound_audit_execution,
        runtime_seal_member,
        runtime_seal_payload,
    ) = _development_qualification_fixture(
        monkeypatch,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
        preformal_runtime_seal=preformal_runtime_seal,
    )

    artifact_audit._validate_development_qualification(
        payload,
        block=block,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
        bound_audit_execution=bound_audit_execution,
        preformal_runtime_seal=preformal_runtime_seal,
    )

    overstated_runtime_seal = {
        **preformal_runtime_seal,
        "assurance": {
            **artifact_audit._ASSURANCE,
            "tamper_resistant": True,
        },
    }
    monkeypatch.setattr(
        artifact_audit,
        "_verify_development_qualification_archive",
        lambda *_args, **_kwargs: {
            runtime_seal_member: (
                artifact_audit._canonical_json_bytes(
                    overstated_runtime_seal
                )
                + b"\n"
            ),
        },
    )
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="exact preformal assured seal",
    ):
        artifact_audit._validate_development_qualification(
            payload,
            block=block,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
            bound_audit_execution=bound_audit_execution,
            preformal_runtime_seal=preformal_runtime_seal,
        )

    monkeypatch.setattr(
        artifact_audit,
        "_verify_development_qualification_archive",
        lambda *_args, **_kwargs: {
            runtime_seal_member: runtime_seal_payload,
        },
    )
    block["raw_result_sha256"] = "6" * 64
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="projection differs",
    ):
        artifact_audit._validate_development_qualification(
            payload,
            block=block,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
            bound_audit_execution=bound_audit_execution,
            preformal_runtime_seal=preformal_runtime_seal,
        )


def test_formal_input_preflight_runs_both_substantive_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, dependencies, runtime, report = _preformal_v2_fixture(
        tmp_path,
        monkeypatch,
    )
    preformal_runtime_seal = cast(
        Mapping[str, object], report["runtime_seal"]
    )
    (
        closure_payload,
        development,
        audit_execution,
        _,
        _,
    ) = _development_qualification_fixture(
        monkeypatch,
        source=source,
        dependencies=dependencies,
        runtime=runtime,
        preformal_runtime_seal=preformal_runtime_seal,
    )

    live_closure_identity = report["input_files_before"][
        "development_closure"
    ]
    live_closure_path = Path(live_closure_identity["path"])
    live_closure_path.write_bytes(closure_payload)
    live_closure_identity.update(
        {
            "bytes": len(closure_payload),
            "sha256": hashlib.sha256(closure_payload).hexdigest(),
        }
    )
    report["input_files_after"] = copy.deepcopy(
        report["input_files_before"]
    )

    accepted_closure = json.loads(
        (
            tmp_path
            / report["commands"][8]["stdout"]["file"]
        ).read_bytes()
    )
    accepted_closure.update(
        {
            "development_closure_sha256": development[
                "closure_sha256"
            ],
            "producer_manifest_sha256": development[
                "producer_manifest_sha256"
            ],
            "raw_result_sha256": development["raw_result_sha256"],
        }
    )
    _replace_preformal_v2_log(
        tmp_path,
        report=report,
        source=source,
        command_index=8,
        stream="stdout",
        payload=(
            artifact_audit._canonical_json_bytes(accepted_closure)
            + b"\n"
        ),
    )

    runtime_conformance = json.loads(
        (
            tmp_path
            / report["commands"][9]["stdout"]["file"]
        ).read_bytes()
    )
    runtime_conformance.update(
        {
            "conformance_sha256": hashlib.sha256(
                artifact_audit._canonical_json_bytes(audit_execution)
            ).hexdigest(),
            "restart_runtime_conformance_report_sha256": (
                audit_execution[
                    "restart_runtime_conformance_report_sha256"
                ]
            ),
            "restart_runtime_execution_receipt_sha256": (
                audit_execution[
                    "restart_runtime_execution_receipt_sha256"
                ]
            ),
            "restart_runtime_support_files": audit_execution[
                "restart_runtime_support_files"
            ],
            "restart_runtime_repeat_count": audit_execution[
                "restart_runtime_repeat_count"
            ],
            "restart_runtime_path_descriptor_equal": audit_execution[
                "restart_runtime_path_descriptor_equal"
            ],
        }
    )
    _replace_preformal_v2_log(
        tmp_path,
        report=report,
        source=source,
        command_index=9,
        stream="stdout",
        payload=(
            artifact_audit._canonical_json_bytes(runtime_conformance)
            + b"\n"
        ),
    )
    report_payload = _rewrite_preformal_v2_report(
        tmp_path,
        report,
        source,
    )

    preserved_closure = tmp_path / cast(
        str, development["closure_file"]
    )
    preserved_closure.write_bytes(closure_payload)
    binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v9",
        "experiment_id": "WM-001",
        "assurance": dict(artifact_audit._ASSURANCE),
        "protocol": {"version": "1.10.0"},
        "source": source,
        "dependencies": dependencies,
        "runtime": runtime,
        "development_qualification": development,
        "audit_execution": audit_execution,
    }
    binding_payload = (
        artifact_audit._canonical_json_bytes(binding) + b"\n"
    )
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_bytes(binding_payload)

    receipt = artifact_audit.preflight_formal_input_package(
        binding_path
    )

    assert receipt["passed"] is True
    assert receipt["binding_sha256"] == hashlib.sha256(
        binding_payload
    ).hexdigest()
    assert receipt["preformal_report_sha256"] == hashlib.sha256(
        report_payload
    ).hexdigest()
    assert receipt["development_closure_sha256"] == hashlib.sha256(
        closure_payload
    ).hexdigest()


def test_development_archive_rejects_hidden_gnu_longname_header(
    tmp_path: Path,
) -> None:
    payload = b"x"
    canonical_path = tmp_path / "canonical.tar"
    canonical_name = "producer/result.json"
    with artifact_audit.tarfile.open(
        canonical_path,
        mode="w",
        format=artifact_audit.tarfile.USTAR_FORMAT,
    ) as archive:
        member = artifact_audit.tarfile.TarInfo(canonical_name)
        member.size = len(payload)
        member.mode = 0o444
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        member.mtime = 0
        member.type = artifact_audit.tarfile.REGTYPE
        archive.addfile(member, io.BytesIO(payload))
    descriptor = os.open(canonical_path, os.O_RDONLY)
    try:
        artifact_audit._verify_canonical_development_ustar(
            descriptor,
            archive_bytes=canonical_path.stat().st_size,
            members=[
                {
                    "path": canonical_name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        )
    finally:
        os.close(descriptor)

    long_name = "producer/" + "x" * 150 + ".json"
    gnu_path = tmp_path / "gnu-longname.tar"
    with artifact_audit.tarfile.open(
        gnu_path,
        mode="w",
        format=artifact_audit.tarfile.GNU_FORMAT,
    ) as archive:
        member = artifact_audit.tarfile.TarInfo(long_name)
        member.size = len(payload)
        member.mode = 0o444
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        member.mtime = 0
        member.type = artifact_audit.tarfile.REGTYPE
        archive.addfile(member, io.BytesIO(payload))
    descriptor = os.open(gnu_path, os.O_RDONLY)
    try:
        with pytest.raises(
            artifact_audit.ArtifactAuditError,
            match="canonical USTAR|noncanonical or hidden",
        ):
            artifact_audit._verify_canonical_development_ustar(
                descriptor,
                archive_bytes=gnu_path.stat().st_size,
                members=[
                    {
                        "path": long_name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            )
    finally:
        os.close(descriptor)


def test_bound_prebinding_execution_requires_complete_passing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _patch_live_packages(monkeypatch, request)
    _patch_expensive_semantics(monkeypatch)
    report = artifact_audit.audit_prebinding_conformance(request)
    assert report["passed"] is True
    request_payload = artifact_audit.canonical_prebinding_request_bytes(request)
    report_payload = artifact_audit._canonical_json_bytes(report) + b"\n"
    bootstrap_payload = b"raise SystemExit(main())\n"
    request_runtime = request["runtime"]
    inventory_root = request["root_inventories"][0]
    protocol_payload = PROTOCOL.read_bytes()
    raw_schema_payload = (WM001 / "schemas" / "raw-result.schema.json").read_bytes()
    (tmp_path / "protocol.json").write_bytes(protocol_payload)
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "raw-result.schema.json").write_bytes(raw_schema_payload)
    source_root = tmp_path / "source" / "bench" / "world_model_lifecycle"
    source_root.mkdir(parents=True)
    source_payloads = {
        name: (WM001 / name).read_bytes()
        for name in (
            "adjudication.py",
            "artifact_audit.py",
            "audit_runner.py",
            "learning.py",
            "model.py",
            "planning.py",
            "producer_bootstrap.py",
            "runtime_lane.py",
        )
    }
    runner_tree = artifact_audit.ast.parse(source_payloads["audit_runner.py"].decode("utf-8"))
    bootstrap_payload = cast(
        bytes,
        next(
            artifact_audit.ast.literal_eval(node.value)
            for node in runner_tree.body
            if isinstance(node, artifact_audit.ast.Assign)
            and any(
                isinstance(target, artifact_audit.ast.Name) and target.id == "_BOOTSTRAP_SOURCE"
                for target in node.targets
            )
        ),
    )
    for name, payload in source_payloads.items():
        (source_root / name).write_bytes(payload)
    source = {
        "implementation_files": [
            {
                "path": f"bench/world_model_lifecycle/{name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(source_payloads.items())
        ]
    }
    dependencies = {
        "python_executable": artifact_audit.sys.executable,
        "python_executable_sha256": request_runtime["python_executable_sha256"],
        "package_roots": [
            {
                "path": inventory_root["path"],
                "file_count": inventory_root["file_count"],
                "total_bytes": inventory_root["total_bytes"],
                "inventory_sha256": inventory_root["inventory_sha256"],
            }
        ],
        "standard_library": {
            "path": str(tmp_path / "stdlib"),
            "file_count": 11,
            "total_bytes": 111,
            "inventory_sha256": "f" * 64,
        },
    }
    runtime = {"process_environment": request_runtime["producer_process_environment"]}
    safe_environment = {
        key: value
        for key, value in runtime["process_environment"].items()
        if key in artifact_audit._PREBINDING_SHARED_ENVIRONMENT_KEYS
    }
    auditor_payload = source_payloads["artifact_audit.py"]

    def support_rows(
        payloads: Mapping[str, bytes],
    ) -> list[dict[str, object]]:
        return [
            {
                "path": path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(payloads.items())
        ]

    common_manifest = {
        "schema": "prospect.wm001.audit-runtime-manifest.v1",
        "assurance": dict(artifact_audit._ASSURANCE),
        "bootstrap_sha256": hashlib.sha256(bootstrap_payload).hexdigest(),
        "python": {
            "executable": artifact_audit.sys.executable,
            "resolved_executable": str(Path(artifact_audit.sys.executable).resolve()),
            "sha256": request_runtime["python_executable_sha256"],
            "version": [
                artifact_audit.sys.version_info.major,
                artifact_audit.sys.version_info.minor,
                artifact_audit.sys.version_info.micro,
            ],
        },
        "required_flags": dict(artifact_audit._PREBINDING_AUDITOR_FLAGS),
        "closure_import_roots": dependencies["package_roots"],
        "standard_library": dependencies["standard_library"],
        "environment": safe_environment,
        "limits": {
            "timeout_seconds": 600,
            "stdout_bytes": 64 << 20,
            "stderr_bytes": 16 << 20,
        },
    }
    prebinding_supports = {
        "prebinding-request.json": request_payload,
        "protocol.json": protocol_payload,
        **{
            name: source_payloads[name]
            for name in (
                "learning.py",
                "model.py",
                "planning.py",
                "runtime_lane.py",
            )
        },
    }
    outcome_supports = {
        "producer_bootstrap.py": source_payloads["producer_bootstrap.py"],
        "protocol.json": protocol_payload,
        "schemas/raw-result.schema.json": raw_schema_payload,
    }

    def runtime_manifest(
        mode: str,
        supports: Mapping[str, bytes],
    ) -> bytes:
        value = {
            **common_manifest,
            "source": {
                "mode": mode,
                "path": "artifact_audit.py",
                "bytes": len(auditor_payload),
                "sha256": hashlib.sha256(auditor_payload).hexdigest(),
            },
            "support_files": support_rows(supports),
        }
        return artifact_audit._canonical_json_bytes(value) + b"\n"

    path_runtime_payload = runtime_manifest(
        "path",
        prebinding_supports,
    )
    descriptor_runtime_payload = runtime_manifest(
        "descriptor",
        prebinding_supports,
    )
    outcome_runtime_payload = runtime_manifest(
        "descriptor",
        outcome_supports,
    )
    working_directory = str(Path.cwd().resolve())

    def invocation_manifest(runtime_payload: bytes) -> bytes:
        return (
            artifact_audit._canonical_json_bytes(
                {
                    "schema": ("prospect.wm001.audit-invocation-manifest.v1"),
                    "runtime_manifest_sha256": hashlib.sha256(runtime_payload).hexdigest(),
                    "working_directory": working_directory,
                    "auditor_argv": [
                        "--prebinding-conformance",
                        "@captured/prebinding-request.json",
                    ],
                }
            )
            + b"\n"
        )

    path_invocation_payload = invocation_manifest(path_runtime_payload)
    descriptor_invocation_payload = invocation_manifest(descriptor_runtime_payload)
    repeat_count = 3

    def execution_row(ordinal: int, mode: str) -> dict[str, object]:
        runtime_payload = path_runtime_payload if mode == "path" else descriptor_runtime_payload
        invocation_payload = path_invocation_payload if mode == "path" else descriptor_invocation_payload
        return {
            "ordinal": ordinal,
            "source_mode": mode,
            "returncode": 0,
            "stdout": {
                "bytes": len(report_payload),
                "sha256": hashlib.sha256(report_payload).hexdigest(),
            },
            "stderr": {
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            "runtime_manifest": {
                "bytes": len(runtime_payload),
                "sha256": hashlib.sha256(runtime_payload).hexdigest(),
            },
            "invocation_manifest": {
                "bytes": len(invocation_payload),
                "sha256": hashlib.sha256(invocation_payload).hexdigest(),
            },
            "bootstrap_sha256": hashlib.sha256(bootstrap_payload).hexdigest(),
            "auditor_source_sha256": hashlib.sha256(auditor_payload).hexdigest(),
            "support_files": support_rows(prebinding_supports),
            "auditor_report_passed": True,
        }

    receipt_rows = [
        execution_row(ordinal, mode)
        for ordinal, mode in enumerate(
            [
                *(["path"] * repeat_count),
                *(["descriptor"] * repeat_count),
            ],
            start=1,
        )
    ]
    execution_receipt_payload = (
        artifact_audit._canonical_json_bytes(
            {
                "schema": ("prospect.wm001.audit-conformance-receipt.v1"),
                "repeat_count": repeat_count,
                "execution_count": len(receipt_rows),
                "executions": receipt_rows,
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "path_descriptor_byte_identical": True,
                "execution_conformance_passed": True,
            }
        )
        + b"\n"
    )
    restart_report_payload = (
        artifact_audit._canonical_json_bytes(
            {
                "schema": (
                    "prospect.wm001.restart-runtime-conformance.v1"
                ),
                "protocol_version": "1.10.0",
                "support_files": support_rows(outcome_supports),
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
        )
        + b"\n"
    )
    restart_path_runtime_payload = runtime_manifest(
        "path",
        outcome_supports,
    )

    def restart_invocation_manifest(
        runtime_payload: bytes,
    ) -> bytes:
        return (
            artifact_audit._canonical_json_bytes(
                {
                    "schema": (
                        "prospect.wm001.audit-invocation-manifest.v1"
                    ),
                    "runtime_manifest_sha256": hashlib.sha256(
                        runtime_payload
                    ).hexdigest(),
                    "working_directory": working_directory,
                    "auditor_argv": [
                        "--restart-runtime-conformance",
                        "--producer-bootstrap",
                        "@captured/producer_bootstrap.py",
                        "--expected-producer-bootstrap-sha256",
                        hashlib.sha256(
                            source_payloads["producer_bootstrap.py"]
                        ).hexdigest(),
                    ],
                }
            )
            + b"\n"
        )

    restart_invocations = {
        "path": restart_invocation_manifest(
            restart_path_runtime_payload
        ),
        "descriptor": restart_invocation_manifest(
            outcome_runtime_payload
        ),
    }
    restart_runtimes = {
        "path": restart_path_runtime_payload,
        "descriptor": outcome_runtime_payload,
    }
    restart_receipt_rows = [
        {
            "ordinal": ordinal,
            "source_mode": mode,
            "returncode": 0,
            "stdout": {
                "bytes": len(restart_report_payload),
                "sha256": hashlib.sha256(
                    restart_report_payload
                ).hexdigest(),
            },
            "stderr": {
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            "runtime_manifest": {
                "bytes": len(restart_runtimes[mode]),
                "sha256": hashlib.sha256(
                    restart_runtimes[mode]
                ).hexdigest(),
            },
            "invocation_manifest": {
                "bytes": len(restart_invocations[mode]),
                "sha256": hashlib.sha256(
                    restart_invocations[mode]
                ).hexdigest(),
            },
            "bootstrap_sha256": hashlib.sha256(
                bootstrap_payload
            ).hexdigest(),
            "auditor_source_sha256": hashlib.sha256(
                auditor_payload
            ).hexdigest(),
            "support_files": support_rows(outcome_supports),
            "auditor_report_passed": True,
        }
        for ordinal, mode in enumerate(
            [
                *(["path"] * repeat_count),
                *(["descriptor"] * repeat_count),
            ],
            start=1,
        )
    ]
    restart_receipt_payload = (
        artifact_audit._canonical_json_bytes(
            {
                "schema": (
                    "prospect.wm001.audit-conformance-receipt.v1"
                ),
                "repeat_count": repeat_count,
                "execution_count": len(restart_receipt_rows),
                "executions": restart_receipt_rows,
                "report_sha256": hashlib.sha256(
                    restart_report_payload
                ).hexdigest(),
                "path_descriptor_byte_identical": True,
                "execution_conformance_passed": True,
            }
        )
        + b"\n"
    )

    def content_name(
        prefix: str,
        payload: bytes,
        suffix: str,
    ) -> str:
        return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}{suffix}"

    block = {
        "runner_source_sha256": hashlib.sha256(source_payloads["audit_runner.py"]).hexdigest(),
        "auditor_source_sha256": hashlib.sha256(auditor_payload).hexdigest(),
        "adjudicator_source_sha256": hashlib.sha256(source_payloads["adjudication.py"]).hexdigest(),
        "bootstrap_source_file": content_name(
            "audit-bootstrap",
            bootstrap_payload,
            ".py",
        ),
        "bootstrap_source_bytes": len(bootstrap_payload),
        "bootstrap_source_sha256": hashlib.sha256(bootstrap_payload).hexdigest(),
        "prebinding_request_file": content_name(
            "audit-prebinding-request",
            request_payload,
            ".json",
        ),
        "prebinding_request_bytes": len(request_payload),
        "prebinding_request_sha256": hashlib.sha256(request_payload).hexdigest(),
        "prebinding_request_identity_sha256": report["request_sha256"],
        "prebinding_path_runtime_manifest_file": content_name(
            "audit-prebinding-path-runtime",
            path_runtime_payload,
            ".json",
        ),
        "prebinding_path_runtime_manifest_bytes": len(path_runtime_payload),
        "prebinding_path_runtime_manifest_sha256": hashlib.sha256(path_runtime_payload).hexdigest(),
        "prebinding_descriptor_runtime_manifest_file": content_name(
            "audit-prebinding-descriptor-runtime",
            descriptor_runtime_payload,
            ".json",
        ),
        "prebinding_descriptor_runtime_manifest_bytes": len(descriptor_runtime_payload),
        "prebinding_descriptor_runtime_manifest_sha256": hashlib.sha256(descriptor_runtime_payload).hexdigest(),
        "prebinding_path_invocation_manifest_file": content_name(
            "audit-prebinding-path-invocation",
            path_invocation_payload,
            ".json",
        ),
        "prebinding_path_invocation_manifest_bytes": len(path_invocation_payload),
        "prebinding_path_invocation_manifest_sha256": hashlib.sha256(path_invocation_payload).hexdigest(),
        "prebinding_descriptor_invocation_manifest_file": content_name(
            "audit-prebinding-descriptor-invocation",
            descriptor_invocation_payload,
            ".json",
        ),
        "prebinding_descriptor_invocation_manifest_bytes": len(descriptor_invocation_payload),
        "prebinding_descriptor_invocation_manifest_sha256": (hashlib.sha256(descriptor_invocation_payload).hexdigest()),
        "prebinding_conformance_report_file": content_name(
            "audit-prebinding-conformance",
            report_payload,
            ".json",
        ),
        "prebinding_conformance_report_bytes": len(report_payload),
        "prebinding_conformance_report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "prebinding_execution_receipt_file": content_name(
            "audit-prebinding-execution-receipt",
            execution_receipt_payload,
            ".json",
        ),
        "prebinding_execution_receipt_bytes": len(execution_receipt_payload),
        "prebinding_execution_receipt_sha256": hashlib.sha256(execution_receipt_payload).hexdigest(),
        "outcome_runtime_manifest_file": content_name(
            "audit-outcome-runtime",
            outcome_runtime_payload,
            ".json",
        ),
        "outcome_runtime_manifest_bytes": len(outcome_runtime_payload),
        "outcome_runtime_manifest_sha256": hashlib.sha256(outcome_runtime_payload).hexdigest(),
        "restart_runtime_conformance_report_file": content_name(
            "audit-restart-runtime-conformance",
            restart_report_payload,
            ".json",
        ),
        "restart_runtime_conformance_report_bytes": len(
            restart_report_payload
        ),
        "restart_runtime_conformance_report_sha256": (
            hashlib.sha256(restart_report_payload).hexdigest()
        ),
        "restart_runtime_execution_receipt_file": content_name(
            "audit-restart-runtime-execution-receipt",
            restart_receipt_payload,
            ".json",
        ),
        "restart_runtime_execution_receipt_bytes": len(
            restart_receipt_payload
        ),
        "restart_runtime_execution_receipt_sha256": (
            hashlib.sha256(restart_receipt_payload).hexdigest()
        ),
        "restart_runtime_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "restart_runtime_repeat_count": repeat_count,
        "restart_runtime_path_descriptor_equal": True,
        "outcome_source_mode": "descriptor",
        "outcome_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "outcome_argv_role": [
            "<canonical-producer-root>",
            "--producer-bootstrap",
            "<captured-producer-bootstrap>",
        ],
        "outcome_working_directory": working_directory,
        "interpreter_flags": ["-I", "-S", "-B"],
        "repeat_count": repeat_count,
        "path_descriptor_equal": True,
        "passed": True,
    }

    artifact_audit._validate_audit_execution_conformance(
        block=block,
        bootstrap_payload=bootstrap_payload,
        request_payload=request_payload,
        path_runtime_manifest_payload=path_runtime_payload,
        descriptor_runtime_manifest_payload=(descriptor_runtime_payload),
        path_invocation_manifest_payload=path_invocation_payload,
        descriptor_invocation_manifest_payload=(descriptor_invocation_payload),
        report_payload=report_payload,
        execution_receipt_payload=execution_receipt_payload,
        outcome_runtime_manifest_payload=outcome_runtime_payload,
        restart_runtime_report_payload=restart_report_payload,
        restart_runtime_receipt_payload=restart_receipt_payload,
        dependencies=dependencies,
        runtime=runtime,
        source=source,
        root=tmp_path,
        preformal_repository_cwd=working_directory,
        verify_live_outcome_runtime=False,
    )

    changed_invocation = json.loads(path_invocation_payload)
    changed_invocation["auditor_argv"] = []
    changed_invocation_payload = artifact_audit._canonical_json_bytes(changed_invocation) + b"\n"
    changed_invocation_digest = hashlib.sha256(changed_invocation_payload).hexdigest()
    changed_invocation_block = dict(block)
    changed_invocation_block["prebinding_path_invocation_manifest_file"] = (
        f"audit-prebinding-path-invocation-{changed_invocation_digest[:16]}.json"
    )
    changed_invocation_block["prebinding_path_invocation_manifest_bytes"] = len(changed_invocation_payload)
    changed_invocation_block["prebinding_path_invocation_manifest_sha256"] = changed_invocation_digest
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="differs from the exact prebinding invocation",
    ):
        artifact_audit._validate_audit_execution_conformance(
            block=changed_invocation_block,
            bootstrap_payload=bootstrap_payload,
            request_payload=request_payload,
            path_runtime_manifest_payload=path_runtime_payload,
            descriptor_runtime_manifest_payload=(descriptor_runtime_payload),
            path_invocation_manifest_payload=(changed_invocation_payload),
            descriptor_invocation_manifest_payload=(descriptor_invocation_payload),
            report_payload=report_payload,
            execution_receipt_payload=execution_receipt_payload,
            outcome_runtime_manifest_payload=outcome_runtime_payload,
            restart_runtime_report_payload=restart_report_payload,
            restart_runtime_receipt_payload=restart_receipt_payload,
            dependencies=dependencies,
            runtime=runtime,
            source=source,
            root=tmp_path,
            preformal_repository_cwd=working_directory,
            verify_live_outcome_runtime=False,
        )

    changed_receipt = json.loads(execution_receipt_payload)
    changed_receipt["executions"][1]["ordinal"] = 1
    changed_receipt_payload = artifact_audit._canonical_json_bytes(changed_receipt) + b"\n"
    changed_receipt_digest = hashlib.sha256(changed_receipt_payload).hexdigest()
    changed_receipt_block = dict(block)
    changed_receipt_block["prebinding_execution_receipt_file"] = (
        f"audit-prebinding-execution-receipt-{changed_receipt_digest[:16]}.json"
    )
    changed_receipt_block["prebinding_execution_receipt_bytes"] = len(changed_receipt_payload)
    changed_receipt_block["prebinding_execution_receipt_sha256"] = changed_receipt_digest
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="does not prove every exact path and descriptor execution",
    ):
        artifact_audit._validate_audit_execution_conformance(
            block=changed_receipt_block,
            bootstrap_payload=bootstrap_payload,
            request_payload=request_payload,
            path_runtime_manifest_payload=path_runtime_payload,
            descriptor_runtime_manifest_payload=(descriptor_runtime_payload),
            path_invocation_manifest_payload=path_invocation_payload,
            descriptor_invocation_manifest_payload=(descriptor_invocation_payload),
            report_payload=report_payload,
            execution_receipt_payload=changed_receipt_payload,
            outcome_runtime_manifest_payload=outcome_runtime_payload,
            restart_runtime_report_payload=restart_report_payload,
            restart_runtime_receipt_payload=restart_receipt_payload,
            dependencies=dependencies,
            runtime=runtime,
            source=source,
            root=tmp_path,
            preformal_repository_cwd=working_directory,
            verify_live_outcome_runtime=False,
        )

    aliased_prebinding_receipts: list[dict[str, Any]] = []
    for field in ("repeat_count", "execution_count"):
        changed = copy.deepcopy(
            json.loads(execution_receipt_payload)
        )
        changed[field] = float(cast(int, changed[field]))
        aliased_prebinding_receipts.append(changed)
    false_returncode = copy.deepcopy(
        json.loads(execution_receipt_payload)
    )
    false_returncode["executions"][0]["returncode"] = False
    aliased_prebinding_receipts.append(false_returncode)
    float_stdout_bytes = copy.deepcopy(
        json.loads(execution_receipt_payload)
    )
    float_stdout_bytes["executions"][0]["stdout"]["bytes"] = float(
        float_stdout_bytes["executions"][0]["stdout"]["bytes"]
    )
    aliased_prebinding_receipts.append(float_stdout_bytes)
    for aliased_receipt in aliased_prebinding_receipts:
        aliased_payload = (
            artifact_audit._canonical_json_bytes(aliased_receipt)
            + b"\n"
        )
        aliased_digest = hashlib.sha256(
            aliased_payload
        ).hexdigest()
        aliased_block = dict(block)
        aliased_block["prebinding_execution_receipt_file"] = (
            "audit-prebinding-execution-receipt-"
            f"{aliased_digest[:16]}.json"
        )
        aliased_block["prebinding_execution_receipt_bytes"] = len(
            aliased_payload
        )
        aliased_block["prebinding_execution_receipt_sha256"] = (
            aliased_digest
        )
        with pytest.raises(
            artifact_audit.ArtifactAuditError,
            match=(
                "prebinding execution receipt does not prove every "
                "exact path and descriptor execution"
            ),
        ):
            artifact_audit._validate_audit_execution_conformance(
                block=aliased_block,
                bootstrap_payload=bootstrap_payload,
                request_payload=request_payload,
                path_runtime_manifest_payload=path_runtime_payload,
                descriptor_runtime_manifest_payload=(
                    descriptor_runtime_payload
                ),
                path_invocation_manifest_payload=(
                    path_invocation_payload
                ),
                descriptor_invocation_manifest_payload=(
                    descriptor_invocation_payload
                ),
                report_payload=report_payload,
                execution_receipt_payload=aliased_payload,
                outcome_runtime_manifest_payload=(
                    outcome_runtime_payload
                ),
                restart_runtime_report_payload=(
                    restart_report_payload
                ),
                restart_runtime_receipt_payload=(
                    restart_receipt_payload
                ),
                dependencies=dependencies,
                runtime=runtime,
                source=source,
                root=tmp_path,
                preformal_repository_cwd=working_directory,
                verify_live_outcome_runtime=False,
            )

    changed_outcome = json.loads(outcome_runtime_payload)
    changed_outcome["source"]["mode"] = "path"
    changed_outcome_payload = artifact_audit._canonical_json_bytes(changed_outcome) + b"\n"
    changed_outcome_digest = hashlib.sha256(changed_outcome_payload).hexdigest()
    changed_outcome_block = dict(block)
    changed_outcome_block["outcome_runtime_manifest_file"] = f"audit-outcome-runtime-{changed_outcome_digest[:16]}.json"
    changed_outcome_block["outcome_runtime_manifest_bytes"] = len(changed_outcome_payload)
    changed_outcome_block["outcome_runtime_manifest_sha256"] = changed_outcome_digest
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="outcome descriptor runtime manifest differs",
    ):
        artifact_audit._validate_audit_execution_conformance(
            block=changed_outcome_block,
            bootstrap_payload=bootstrap_payload,
            request_payload=request_payload,
            path_runtime_manifest_payload=path_runtime_payload,
            descriptor_runtime_manifest_payload=(descriptor_runtime_payload),
            path_invocation_manifest_payload=(path_invocation_payload),
            descriptor_invocation_manifest_payload=(descriptor_invocation_payload),
            report_payload=report_payload,
            execution_receipt_payload=execution_receipt_payload,
            outcome_runtime_manifest_payload=changed_outcome_payload,
            restart_runtime_report_payload=restart_report_payload,
            restart_runtime_receipt_payload=restart_receipt_payload,
            dependencies=dependencies,
            runtime=runtime,
            source=source,
            root=tmp_path,
            preformal_repository_cwd=working_directory,
            verify_live_outcome_runtime=False,
        )

    rejected = copy.deepcopy(report)
    rejected["passed"] = False
    rejected_payload = artifact_audit._canonical_json_bytes(rejected) + b"\n"
    rejected_block = dict(block)
    rejected_block["prebinding_conformance_report_bytes"] = len(rejected_payload)
    rejected_digest = hashlib.sha256(rejected_payload).hexdigest()
    rejected_block["prebinding_conformance_report_sha256"] = rejected_digest
    rejected_block["prebinding_conformance_report_file"] = f"audit-prebinding-conformance-{rejected_digest[:16]}.json"
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="not a complete pass",
    ):
        artifact_audit._validate_audit_execution_conformance(
            block=rejected_block,
            bootstrap_payload=bootstrap_payload,
            request_payload=request_payload,
            path_runtime_manifest_payload=path_runtime_payload,
            descriptor_runtime_manifest_payload=(descriptor_runtime_payload),
            path_invocation_manifest_payload=(path_invocation_payload),
            descriptor_invocation_manifest_payload=(descriptor_invocation_payload),
            report_payload=rejected_payload,
            execution_receipt_payload=execution_receipt_payload,
            outcome_runtime_manifest_payload=outcome_runtime_payload,
            restart_runtime_report_payload=restart_report_payload,
            restart_runtime_receipt_payload=restart_receipt_payload,
            dependencies=dependencies,
            runtime=runtime,
            source=source,
            root=tmp_path,
            preformal_repository_cwd=working_directory,
            verify_live_outcome_runtime=False,
        )

    def validate_restart_evidence(
        *,
        changed_block: dict[str, object],
        changed_report_payload: bytes = restart_report_payload,
        changed_receipt_payload: bytes = restart_receipt_payload,
    ) -> None:
        artifact_audit._validate_audit_execution_conformance(
            block=changed_block,
            bootstrap_payload=bootstrap_payload,
            request_payload=request_payload,
            path_runtime_manifest_payload=path_runtime_payload,
            descriptor_runtime_manifest_payload=descriptor_runtime_payload,
            path_invocation_manifest_payload=path_invocation_payload,
            descriptor_invocation_manifest_payload=descriptor_invocation_payload,
            report_payload=report_payload,
            execution_receipt_payload=execution_receipt_payload,
            outcome_runtime_manifest_payload=outcome_runtime_payload,
            restart_runtime_report_payload=changed_report_payload,
            restart_runtime_receipt_payload=changed_receipt_payload,
            dependencies=dependencies,
            runtime=runtime,
            source=source,
            root=tmp_path,
            preformal_repository_cwd=working_directory,
            verify_live_outcome_runtime=False,
        )

    boolean_report = json.loads(restart_report_payload)
    boolean_report["negative_cases"][0]["rejected"] = 1
    boolean_report_payload = (
        artifact_audit._canonical_json_bytes(boolean_report) + b"\n"
    )
    boolean_report_digest = hashlib.sha256(
        boolean_report_payload
    ).hexdigest()
    boolean_report_block = dict(block)
    boolean_report_block["restart_runtime_conformance_report_file"] = (
        f"audit-restart-runtime-conformance-{boolean_report_digest[:16]}.json"
    )
    boolean_report_block["restart_runtime_conformance_report_bytes"] = len(
        boolean_report_payload
    )
    boolean_report_block["restart_runtime_conformance_report_sha256"] = (
        boolean_report_digest
    )
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="restart-runtime conformance report is not one complete",
    ):
        validate_restart_evidence(
            changed_block=boolean_report_block,
            changed_report_payload=boolean_report_payload,
        )

    boolean_receipt = json.loads(restart_receipt_payload)
    boolean_receipt["executions"][0]["returncode"] = False
    boolean_receipt_payload = (
        artifact_audit._canonical_json_bytes(boolean_receipt) + b"\n"
    )
    boolean_receipt_digest = hashlib.sha256(
        boolean_receipt_payload
    ).hexdigest()
    boolean_receipt_block = dict(block)
    boolean_receipt_block["restart_runtime_execution_receipt_file"] = (
        "audit-restart-runtime-execution-receipt-"
        f"{boolean_receipt_digest[:16]}.json"
    )
    boolean_receipt_block["restart_runtime_execution_receipt_bytes"] = len(
        boolean_receipt_payload
    )
    boolean_receipt_block["restart_runtime_execution_receipt_sha256"] = (
        boolean_receipt_digest
    )
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="restart-runtime execution receipt does not prove",
    ):
        validate_restart_evidence(
            changed_block=boolean_receipt_block,
            changed_receipt_payload=boolean_receipt_payload,
        )

    float_receipt = json.loads(restart_receipt_payload)
    float_receipt["executions"][0]["stdout"]["bytes"] = float(
        float_receipt["executions"][0]["stdout"]["bytes"]
    )
    float_receipt_payload = (
        artifact_audit._canonical_json_bytes(float_receipt) + b"\n"
    )
    float_receipt_digest = hashlib.sha256(float_receipt_payload).hexdigest()
    float_receipt_block = dict(block)
    float_receipt_block["restart_runtime_execution_receipt_file"] = (
        "audit-restart-runtime-execution-receipt-"
        f"{float_receipt_digest[:16]}.json"
    )
    float_receipt_block["restart_runtime_execution_receipt_bytes"] = len(
        float_receipt_payload
    )
    float_receipt_block["restart_runtime_execution_receipt_sha256"] = (
        float_receipt_digest
    )
    with pytest.raises(
        artifact_audit.ArtifactAuditError,
        match="restart-runtime execution receipt does not prove",
    ):
        validate_restart_evidence(
            changed_block=float_receipt_block,
            changed_receipt_payload=float_receipt_payload,
        )

    for field in ("repeat_count", "execution_count"):
        float_count_receipt = json.loads(restart_receipt_payload)
        float_count_receipt[field] = float(
            float_count_receipt[field]
        )
        float_count_payload = (
            artifact_audit._canonical_json_bytes(
                float_count_receipt
            )
            + b"\n"
        )
        float_count_digest = hashlib.sha256(
            float_count_payload
        ).hexdigest()
        float_count_block = dict(block)
        float_count_block[
            "restart_runtime_execution_receipt_file"
        ] = (
            "audit-restart-runtime-execution-receipt-"
            f"{float_count_digest[:16]}.json"
        )
        float_count_block[
            "restart_runtime_execution_receipt_bytes"
        ] = len(float_count_payload)
        float_count_block[
            "restart_runtime_execution_receipt_sha256"
        ] = float_count_digest
        with pytest.raises(
            artifact_audit.ArtifactAuditError,
            match="restart-runtime execution receipt does not prove",
        ):
            validate_restart_evidence(
                changed_block=float_count_block,
                changed_receipt_payload=float_count_payload,
            )


def test_cli_mode_is_mutually_exclusive_with_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        artifact_audit.main(
            [
                str(tmp_path),
                "--prebinding-conformance",
                str(tmp_path / "request.json"),
            ]
        )
