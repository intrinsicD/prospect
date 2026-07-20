from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from bench.world_model_lifecycle.audit_runner import (
    BOOTSTRAP_SHA256,
    AuditExecutionFailure,
    AuditRunnerError,
    bootstrap_source_bytes,
    bootstrap_source_sha256,
    build_invocation_manifest,
    build_runtime_manifest,
    captured_support_argument,
    conformance_receipt_bytes,
    run_captured_auditor,
    run_source_mode_conformance,
)


def _write_auditor(path: Path, body: str) -> Path:
    path.write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "import sys\n"
        "def emit(value: object) -> None:\n"
        "    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))\n"
        "    sys.stdout.buffer.write(payload.encode('utf-8') + b'\\n')\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _write_distribution(
    root: Path,
    *,
    distribution: str,
    version: str,
    package: str,
    package_body: str,
) -> None:
    package_root = root / package
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(package_body, encoding="utf-8")
    metadata_root = root / f"{distribution.replace('-', '_')}-{version}.dist-info"
    metadata_root.mkdir()
    (metadata_root / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_restart_runtime_conformance_is_result_free_repeated_and_adversarial(
    tmp_path: Path,
) -> None:
    lifecycle = (
        Path(__file__).resolve().parents[1]
        / "bench"
        / "world_model_lifecycle"
    )
    package_root = tmp_path / "minimal-packages"
    numpy_root = package_root / "numpy"
    numpy_root.mkdir(parents=True)
    (numpy_root / "__init__.py").write_text(
        "__all__ = []\n",
        encoding="utf-8",
    )
    (numpy_root / "typing.py").write_text(
        "__all__ = []\n",
        encoding="utf-8",
    )
    supports = {
        "producer_bootstrap.py": (
            lifecycle / "producer_bootstrap.py"
        ),
        "protocol.json": lifecycle / "protocol.json",
        "schemas/raw-result.schema.json": (
            lifecycle / "schemas" / "raw-result.schema.json"
        ),
    }
    expected_bootstrap_sha256 = hashlib.sha256(
        supports["producer_bootstrap.py"].read_bytes()
    ).hexdigest()
    arguments = (
        "--restart-runtime-conformance",
        "--producer-bootstrap",
        captured_support_argument("producer_bootstrap.py"),
        "--expected-producer-bootstrap-sha256",
        expected_bootstrap_sha256,
    )
    conformance = run_source_mode_conformance(
        lifecycle / "artifact_audit.py",
        auditor_arguments=arguments,
        support_files=supports,
        closure_import_roots=(package_root,),
        working_directory=Path.cwd().resolve(),
        environment={},
        repeat_count=3,
    )
    report = dict(conformance.path_execution.report)
    receipt = json.loads(conformance_receipt_bytes(conformance))
    assert report == {
        "schema": "prospect.wm001.restart-runtime-conformance.v1",
        "protocol_version": "1.10.0",
        "support_files": [
            {
                "path": path,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
            }
            for path, source in sorted(supports.items())
        ],
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
    assert receipt["repeat_count"] == 3
    assert receipt["execution_count"] == 6
    assert [
        row["source_mode"] for row in receipt["executions"]
    ] == ["path"] * 3 + ["descriptor"] * 3
    assert all(
        row["stdout"]
        == receipt["executions"][0]["stdout"]
        and row["support_files"] == report["support_files"]
        for row in receipt["executions"]
    )
    encoded = conformance.path_execution.stdout
    assert conformance.descriptor_execution.stdout == encoded
    assert str(tmp_path).encode() not in encoded
    assert b"prospect-audit-runner-" not in encoded
    assert b"/proc/self/fd/" not in encoded
    assert b"/dev/fd/" not in encoded

    missing = dict(supports)
    del missing["producer_bootstrap.py"]
    with pytest.raises(
        AuditRunnerError,
        match="unknown captured support",
    ):
        run_captured_auditor(
            lifecycle / "artifact_audit.py",
            auditor_arguments=arguments,
            support_files=missing,
            closure_import_roots=(package_root,),
            working_directory=Path.cwd().resolve(),
            environment={},
        )

    extra = tmp_path / "unexpected.py"
    extra.write_text("unexpected = True\n", encoding="utf-8")
    extra_execution = run_captured_auditor(
        lifecycle / "artifact_audit.py",
        auditor_arguments=arguments,
        support_files={**supports, "unexpected.py": extra},
        closure_import_roots=(package_root,),
        working_directory=Path.cwd().resolve(),
        environment={},
    )
    assert extra_execution.returncode == 1
    assert extra_execution.report["passed"] is False
    assert (
        extra_execution.report["failure_code"]
        == "captured_support_invalid"
    )

    mutated = tmp_path / "producer_bootstrap.py"
    mutated.write_bytes(
        supports["producer_bootstrap.py"].read_bytes()
        + b"\n# mutation\n"
    )
    mutated_execution = run_captured_auditor(
        lifecycle / "artifact_audit.py",
        auditor_arguments=arguments,
        support_files={
            **supports,
            "producer_bootstrap.py": mutated,
        },
        closure_import_roots=(package_root,),
        working_directory=Path.cwd().resolve(),
        environment={},
    )
    assert mutated_execution.returncode == 1
    assert mutated_execution.report["passed"] is False
    assert (
        mutated_execution.report["failure_code"]
        == "captured_support_invalid"
    )


def test_path_and_descriptor_modes_are_byte_identical_and_repeatable(
    tmp_path: Path,
) -> None:
    support = tmp_path / "input.json"
    support.write_text('{"answer":42}\n', encoding="utf-8")
    auditor = _write_auditor(
        tmp_path / "tiny_auditor.py",
        "from pathlib import Path\n"
        "support = Path(__file__).resolve().parent / 'data/input.json'\n"
        "emit({'answer': json.loads(support.read_text())['answer'], "
        "'arguments': sys.argv[1:], 'passed': True})",
    )
    conformance = run_source_mode_conformance(
        auditor,
        auditor_arguments=("artifact", "--strict"),
        support_files={"data/input.json": support},
        working_directory=tmp_path,
    )
    repeated = run_captured_auditor(
        auditor,
        auditor_arguments=("artifact", "--strict"),
        support_files={"data/input.json": support},
        source_mode="descriptor",
        working_directory=tmp_path,
    )

    assert conformance.path_execution.stdout == conformance.descriptor_execution.stdout
    assert conformance.descriptor_execution.stdout == repeated.stdout
    assert conformance.descriptor_execution.runtime_manifest == repeated.runtime_manifest
    assert conformance.repeat_count == 3
    assert len(conformance.path_executions) == len(conformance.descriptor_executions) == 3
    assert {execution.stdout for execution in conformance.path_executions} == {repeated.stdout}
    assert {execution.stdout for execution in conformance.descriptor_executions} == {repeated.stdout}
    assert conformance.report_sha256 == hashlib.sha256(repeated.stdout).hexdigest()
    assert repeated.report == {
        "answer": 42,
        "arguments": ["artifact", "--strict"],
        "passed": True,
    }
    assert tuple(repeated.command[1:4]) == ("-I", "-S", "-B")
    assert len(repeated.command) == 5
    assert repeated.command[-1].startswith(("/proc/self/fd/", "/dev/fd/"))


def test_source_mode_conformance_rejects_process_dependent_reports(
    tmp_path: Path,
) -> None:
    auditor = _write_auditor(
        tmp_path / "process-dependent.py",
        "import os\nemit({'passed': True, 'process_id': os.getpid()})",
    )

    with pytest.raises(AuditRunnerError, match="not byte-identical"):
        run_source_mode_conformance(
            auditor,
            working_directory=tmp_path,
            repeat_count=2,
        )


def test_runtime_manifest_is_stable_prebindable_and_bootstrap_bound(
    tmp_path: Path,
) -> None:
    auditor = _write_auditor(
        tmp_path / "audit.py",
        "emit({'passed': True})",
    )
    first = build_runtime_manifest(
        auditor,
        source_mode="descriptor",
        environment={"OMP_NUM_THREADS": "1"},
    )
    second = build_runtime_manifest(
        auditor,
        source_mode="descriptor",
        environment={"OMP_NUM_THREADS": "1"},
    )
    execution = run_captured_auditor(
        auditor,
        auditor_arguments=("evidence",),
        source_mode="descriptor",
        working_directory=tmp_path,
        environment={"OMP_NUM_THREADS": "1"},
        runtime_manifest=first,
    )

    assert first == second == execution.runtime_manifest
    assert execution.runtime_manifest_sha256 == hashlib.sha256(first).hexdigest()
    assert json.loads(execution.invocation_manifest) == {
        "schema": "prospect.wm001.audit-invocation-manifest.v1",
        "runtime_manifest_sha256": execution.runtime_manifest_sha256,
        "working_directory": str(tmp_path),
        "auditor_argv": ["evidence"],
    }
    assert execution.bootstrap_sha256 == BOOTSTRAP_SHA256
    assert bootstrap_source_sha256() == BOOTSTRAP_SHA256
    assert hashlib.sha256(bootstrap_source_bytes()).hexdigest() == BOOTSTRAP_SHA256
    manifest = json.loads(first)
    assert manifest["bootstrap_sha256"] == BOOTSTRAP_SHA256
    assert manifest["assurance"] == {
        "trust_model_id": "prospect.wm001.trust-model.v1",
        "tamper_resistant": False,
        "external_attestation": False,
        "exclusive_path_use_required": True,
    }
    assert manifest["source"]["mode"] == "descriptor"
    assert manifest["environment"] == {"LC_ALL": "C.UTF-8", "OMP_NUM_THREADS": "1"}
    assert manifest["required_flags"] == {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }

    changed = json.loads(first)
    changed["auditor_argv"] = ["different"]
    tampered = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with pytest.raises(AuditRunnerError, match="does not exactly match"):
        run_captured_auditor(
            auditor,
            auditor_arguments=("evidence",),
            source_mode="descriptor",
            working_directory=tmp_path,
            environment={"OMP_NUM_THREADS": "1"},
            runtime_manifest=tampered,
        )

    other_directory = tmp_path / "future-producer"
    other_directory.mkdir()
    other_invocation = build_invocation_manifest(
        runtime_manifest=first,
        auditor_arguments=("different-artifact",),
        working_directory=other_directory,
    )
    assert (
        build_runtime_manifest(
            auditor,
            source_mode="descriptor",
            environment={"OMP_NUM_THREADS": "1"},
        )
        == first
    )
    assert other_invocation != execution.invocation_manifest

    with pytest.raises(AuditRunnerError, match="invocation manifest"):
        run_captured_auditor(
            auditor,
            auditor_arguments=("evidence",),
            source_mode="descriptor",
            working_directory=tmp_path,
            environment={"OMP_NUM_THREADS": "1"},
            runtime_manifest=first,
            invocation_manifest=other_invocation,
        )


def test_captured_support_arguments_resolve_only_inside_private_capture(
    tmp_path: Path,
) -> None:
    request = tmp_path / "mutable-original-request.json"
    request.write_text('{"case":"prebinding"}\n', encoding="utf-8")
    auditor = _write_auditor(
        tmp_path / "request-auditor.py",
        "from pathlib import Path\n"
        "request_path = Path(sys.argv[1])\n"
        "emit({'case': json.loads(request_path.read_text())['case'], "
        "'passed': True, 'private': request_path != Path(sys.argv[2])})",
    )
    token = captured_support_argument("requests/prebinding-request.json")

    execution = run_captured_auditor(
        auditor,
        auditor_arguments=(token, str(request)),
        support_files={"requests/prebinding-request.json": request},
        working_directory=tmp_path,
    )

    assert execution.report == {
        "case": "prebinding",
        "passed": True,
        "private": True,
    }
    invocation = json.loads(execution.invocation_manifest)
    assert invocation["auditor_argv"][0] == token
    assert str(request) not in invocation["auditor_argv"][0]


@pytest.mark.parametrize(
    "token",
    [
        "@captured",
        "@captured/",
        "@captured/../escape.json",
        "@captured/missing.json",
    ],
)
def test_unknown_or_unsafe_captured_support_tokens_are_rejected(
    tmp_path: Path,
    token: str,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    auditor = _write_auditor(tmp_path / "audit.py", "emit({'passed': True})")

    with pytest.raises(AuditRunnerError, match="captured-support|captured support"):
        run_captured_auditor(
            auditor,
            auditor_arguments=(token,),
            support_files={"request.json": request},
            working_directory=tmp_path,
        )


def test_declared_user_site_like_transitive_closure_works_without_site_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary_root = tmp_path / "primary-root"
    user_site = tmp_path / "fake-user-site" / "lib" / "python3.12" / "site-packages"
    ambient_root = tmp_path / "ambient-pythonpath"
    primary_root.mkdir()
    user_site.mkdir(parents=True)
    ambient_root.mkdir()
    sentinel_pth = tmp_path / "pth-executed"
    sentinel_site = tmp_path / "sitecustomize-executed"
    sentinel_ambient = tmp_path / "ambient-sitecustomize-executed"

    _write_distribution(
        primary_root,
        distribution="primary-dist",
        version="1.2.3",
        package="primary_pkg",
        package_body="from transitive_pkg import VALUE\n",
    )
    _write_distribution(
        user_site,
        distribution="transitive-dist",
        version="4.5.6",
        package="transitive_pkg",
        package_body="VALUE = 'closure-loaded'\n",
    )
    (user_site / "hostile.pth").write_text(
        f"import pathlib; pathlib.Path({str(sentinel_pth)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (user_site / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel_site)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (ambient_root / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel_ambient)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(ambient_root))
    monkeypatch.setenv("PYTHONSTARTUP", str(ambient_root / "sitecustomize.py"))

    auditor = _write_auditor(
        tmp_path / "closure_auditor.py",
        "import importlib.metadata\n"
        "import os\n"
        "import primary_pkg\n"
        "emit({\n"
        "    'ambient_python': sorted(key for key in os.environ if key.startswith('PYTHON')),\n"
        "    'internal_controls': sorted(key for key in os.environ "
        "if key.startswith('PROSPECT_AUDIT_RUNNER_')),\n"
        "    'passed': True,\n"
        "    'primary': importlib.metadata.version('primary-dist'),\n"
        "    'transitive': importlib.metadata.version('transitive-dist'),\n"
        "    'value': primary_pkg.VALUE,\n"
        "})",
    )

    execution = run_captured_auditor(
        auditor,
        closure_import_roots=(primary_root, user_site),
        working_directory=tmp_path,
    )

    assert execution.report == {
        "ambient_python": [],
        "internal_controls": [],
        "passed": True,
        "primary": "1.2.3",
        "transitive": "4.5.6",
        "value": "closure-loaded",
    }
    assert not sentinel_pth.exists()
    assert not sentinel_site.exists()
    assert not sentinel_ambient.exists()


def test_exact_interpreter_flags_argv_and_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "ambient"))
    monkeypatch.setenv("HOME", str(tmp_path / "ambient-home"))
    auditor = _write_auditor(
        tmp_path / "flags_auditor.py",
        "import os\n"
        "import sysconfig\n"
        "from pathlib import Path\n"
        "cmdline = Path('/proc/self/cmdline').read_bytes().split(b'\\x00')[:-1]\n"
        "emit({\n"
        "    'arguments': sys.argv[1:],\n"
        "    'cmdline': [item.decode() for item in cmdline],\n"
        "    'environment': dict(sorted(os.environ.items())),\n"
        "    'flags': {\n"
        "        'dont_write_bytecode': sys.flags.dont_write_bytecode,\n"
        "        'ignore_environment': sys.flags.ignore_environment,\n"
        "        'isolated': sys.flags.isolated,\n"
        "        'no_site': sys.flags.no_site,\n"
        "        'no_user_site': sys.flags.no_user_site,\n"
        "        'safe_path': sys.flags.safe_path,\n"
        "    },\n"
        "    'search_path': list(sys.path),\n"
        "    'stdlib': sysconfig.get_path('stdlib'),\n"
        "    'passed': True,\n"
        "})",
    )

    execution = run_captured_auditor(
        auditor,
        auditor_arguments=("one", "two"),
        working_directory=tmp_path,
        environment={
            "OMP_NUM_THREADS": "2",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
            "SDL_AUDIODRIVER": "dsp",
        },
    )

    assert execution.report["arguments"] == ["one", "two"]
    assert execution.report["environment"] == {
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "2",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
    }
    assert execution.report["flags"] == {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    stdlib = Path(str(execution.report["stdlib"]))
    search_path = execution.report["search_path"]
    assert isinstance(search_path, list)
    assert search_path
    assert search_path[0] == str(stdlib)
    assert all(
        Path(str(entry)).is_dir()
        and Path(str(entry)).is_relative_to(stdlib)
        for entry in search_path
    )
    assert not any(str(entry).endswith(".zip") for entry in search_path)
    cmdline = execution.report["cmdline"]
    assert isinstance(cmdline, list)
    assert cmdline[:4] == [sys.executable, "-I", "-S", "-B"]
    assert len(cmdline) == 5
    assert str(cmdline[-1]).startswith(("/proc/self/fd/", "/dev/fd/"))

    with pytest.raises(AuditRunnerError, match="not explicitly safe"):
        run_captured_auditor(
            auditor,
            working_directory=tmp_path,
            environment={"PYTHONPATH": str(tmp_path)},
        )


def test_missing_import_root_fails_closed(tmp_path: Path) -> None:
    auditor = _write_auditor(tmp_path / "audit.py", "emit({'passed': True})")

    with pytest.raises(AuditRunnerError, match="closure import root"):
        run_captured_auditor(
            auditor,
            closure_import_roots=(tmp_path / "missing",),
            working_directory=tmp_path,
        )


@pytest.mark.parametrize("target_kind", ["source", "support", "root", "cwd"])
def test_symbolic_links_are_rejected(
    tmp_path: Path,
    target_kind: str,
) -> None:
    real_source = _write_auditor(
        tmp_path / "real-audit.py",
        "emit({'passed': True})",
    )
    real_support = tmp_path / "real-support.txt"
    real_support.write_text("support\n", encoding="utf-8")
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    (real_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    linked_source = tmp_path / "linked-audit.py"
    linked_source.symlink_to(real_source)
    linked_support = tmp_path / "linked-support.txt"
    linked_support.symlink_to(real_support)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    source = linked_source if target_kind == "source" else real_source
    supports = {"support.txt": linked_support} if target_kind == "support" else {"support.txt": real_support}
    roots = (linked_root,) if target_kind == "root" else (real_root,)
    cwd = linked_root if target_kind == "cwd" else tmp_path

    with pytest.raises(AuditRunnerError, match="symbolic-link"):
        run_captured_auditor(
            source,
            support_files=supports,
            closure_import_roots=roots,
            working_directory=cwd,
        )


def test_failed_audit_report_and_returncode_are_consistent(tmp_path: Path) -> None:
    auditor = _write_auditor(
        tmp_path / "rejected.py",
        "emit({'finding': 'expected fixture rejection', 'passed': False})\nraise SystemExit(1)",
    )

    execution = run_captured_auditor(auditor, working_directory=tmp_path)

    assert execution.returncode == 1
    assert execution.report["passed"] is False
    assert execution.stdout == (b'{"finding":"expected fixture rejection","passed":false}\n')


@pytest.mark.parametrize(
    ("passed", "exit_code"),
    [(True, 1), (False, 0), (True, 2)],
)
def test_report_returncode_mismatch_fails_closed(
    tmp_path: Path,
    passed: bool,
    exit_code: int,
) -> None:
    auditor = _write_auditor(
        tmp_path / f"mismatch-{passed}-{exit_code}.py",
        f"emit({{'passed': {passed!r}}})\nraise SystemExit({exit_code})",
    )

    with pytest.raises(AuditRunnerError, match="return code is inconsistent"):
        run_captured_auditor(auditor, working_directory=tmp_path)


def test_failed_execution_preserves_bounded_authenticated_evidence(
    tmp_path: Path,
) -> None:
    support = tmp_path / "protocol.json"
    support.write_text('{"version":"fixture"}\n', encoding="utf-8")
    auditor = _write_auditor(
        tmp_path / "mismatch-with-evidence.py",
        "sys.stderr.buffer.write(b'fixture diagnostic\\n')\nemit({'passed': True})\nraise SystemExit(1)",
    )

    with pytest.raises(AuditExecutionFailure) as captured:
        run_captured_auditor(
            auditor,
            support_files={"protocol.json": support},
            source_mode="descriptor",
            working_directory=tmp_path,
        )

    failure = captured.value
    assert failure.phase == "report_validation"
    assert failure.returncode == 1
    assert failure.stdout == b'{"passed":true}\n'
    assert failure.stderr == b"fixture diagnostic\n"
    assert tuple(failure.command[1:4]) == ("-I", "-S", "-B")
    assert failure.source_mode == "descriptor"
    assert failure.runtime_manifest_sha256 == hashlib.sha256(failure.runtime_manifest).hexdigest()
    assert failure.invocation_manifest_sha256 == hashlib.sha256(failure.invocation_manifest).hexdigest()
    assert failure.bootstrap_sha256 == BOOTSTRAP_SHA256
    assert failure.auditor_source_sha256 == hashlib.sha256(auditor.read_bytes()).hexdigest()
    assert len(failure.support_files) == 1
    assert failure.support_files[0].relative_path == "protocol.json"
    assert failure.support_files[0].bytes == support.stat().st_size
    assert failure.support_files[0].sha256 == hashlib.sha256(support.read_bytes()).hexdigest()


@pytest.mark.parametrize("target", ["source", "support", "manifest", "invocation"])
def test_private_source_support_and_manifest_are_rechecked_after_execution(
    tmp_path: Path,
    target: str,
) -> None:
    support = tmp_path / "support.txt"
    support.write_text("sealed support\n", encoding="utf-8")
    mutation = {
        "source": ("victim = Path(__file__).resolve()\nvictim.chmod(0o600)\nvictim.write_text('# changed source\\n')"),
        "support": (
            "victim = Path(__file__).resolve().parent / 'support.txt'\n"
            "victim.chmod(0o600)\n"
            "victim.write_text('changed support\\n')"
        ),
        "manifest": (
            "victim = next(Path('/proc/self/fd').joinpath(name).resolve() "
            "for name in os.listdir('/proc/self/fd') "
            "if Path('/proc/self/fd').joinpath(name).resolve().name == 'runtime-manifest.json')\n"
            "victim.chmod(0o600)\n"
            "victim.write_text('{}\\n')"
        ),
        "invocation": (
            "victim = next(Path('/proc/self/fd').joinpath(name).resolve() "
            "for name in os.listdir('/proc/self/fd') "
            "if Path('/proc/self/fd').joinpath(name).resolve().name == 'invocation-manifest.json')\n"
            "victim.chmod(0o600)\n"
            "victim.write_text('{}\\n')"
        ),
    }[target]
    auditor = _write_auditor(
        tmp_path / f"mutate-{target}.py",
        f"import os\nfrom pathlib import Path\n{mutation}\nemit({{'passed': True}})",
    )

    with pytest.raises(AuditRunnerError, match="changed|bootstrap rejected"):
        run_captured_auditor(
            auditor,
            support_files={"support.txt": support},
            working_directory=tmp_path,
        )


@pytest.mark.parametrize("target", ["source", "support"])
def test_original_source_and_support_are_rechecked_after_execution(
    tmp_path: Path,
    target: str,
) -> None:
    support = tmp_path / "original-support.txt"
    support.write_text("original support\n", encoding="utf-8")
    auditor = _write_auditor(
        tmp_path / f"mutate-original-{target}.py",
        "from pathlib import Path\n"
        "victim = Path(sys.argv[1])\n"
        "victim.write_text('mutated original\\n')\n"
        "emit({'passed': True})",
    )
    victim = auditor if target == "source" else support

    with pytest.raises(AuditRunnerError, match="changed during isolated execution"):
        run_captured_auditor(
            auditor,
            auditor_arguments=(str(victim),),
            support_files={"support.txt": support},
            working_directory=tmp_path,
        )
