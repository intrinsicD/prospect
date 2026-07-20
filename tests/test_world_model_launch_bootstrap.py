from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle import launch_bootstrap, producer_bootstrap

_ASSURANCE = {
    "trust_model_id": "prospect.wm001.trust-model.v1",
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}


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


def _repository_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    results = repository / "bench" / "world_model_lifecycle" / "results"
    completion_root = results / "outer-completions" / "v1.9"
    completion_root.mkdir(parents=True)
    terminal = results / "operator-v1.9" / "attempt-001" / "terminal-manifest.json"
    terminal.parent.mkdir(parents=True)
    terminal.write_bytes(
        _canonical(
            {
                "experiment_id": "WM-001",
                "schema": "prospect.wm001.operator-attempt-terminal.v1",
                "status": "accepted",
            }
        )
    )
    return repository, completion_root, terminal


def _receipt(
    terminal: Path,
    *,
    logical_exit_code: int = 0,
) -> dict[str, object]:
    payload = terminal.read_bytes()
    return {
        "schema": "prospect.wm001.outer-terminal-receipt.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.9.0",
        "assurance": dict(_ASSURANCE),
        "trust_model": "trusted-single-principal-cooperative-lock-v1",
        "terminal_path": str(terminal),
        "terminal_bytes": len(payload),
        "terminal_sha256": hashlib.sha256(payload).hexdigest(),
        "logical_exit_code": logical_exit_code,
    }


def _completion_marker(completion_root: Path, terminal: Path) -> Path:
    return completion_root / (hashlib.sha256(str(terminal).encode("utf-8")).hexdigest() + ".json")


def _formal_binding_attempt(
    repository: Path,
    completion_root: Path,
    *,
    binding: dict[str, object],
    terminal_state: str,
) -> tuple[Path, Path, Path]:
    results = repository / "bench" / "world_model_lifecycle" / "results"
    report = (
        results
        / "development"
        / "preformal-test-report-v1.9.0.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(_canonical({"passed": True}))
    producer = results / "development" / "qualification-v1.9.0"
    producer.mkdir()
    result = producer / "result.json"
    result.write_bytes(
        _canonical(
            {
                "schema": "prospect.world-model-lifecycle.raw-result.v9",
                "experiment_id": "WM-001",
                "protocol_version": "1.9.0",
                "lane": "development",
            }
        )
    )
    result_payload = result.read_bytes()
    producer_terminal = producer / "producer-manifest.json"
    producer_terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.producer-manifest.v1",
                "experiment_id": "WM-001",
                "lane": "development",
                "status": "completed",
                "started_at_utc": "2026-01-01T00:00:00Z",
                "completed_at_utc": "2026-01-01T00:01:00Z",
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
        )
    )
    producer_completion = _completion_marker(
        completion_root,
        producer_terminal,
    )
    os.link(producer_terminal, producer_completion)
    producer_rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (producer_terminal, result)
    ]
    audit_attempt = (
        results
        / "operator-v1.9"
        / "audits"
        / "development-audit-v1.9.0"
    )
    audit_attempt.mkdir(parents=True)
    audit_members = {
        "audit-execution-01.execution.json": {"execution": 1},
        "audit-execution-02.execution.json": {"execution": 2},
        "audit-execution-02.runtime.json": {"runtime": 2},
        "audit-reproduction.json": {"reproduced": True},
        "independent-audit.json": {"passed": True},
    }
    for name, value in audit_members.items():
        (audit_attempt / name).write_bytes(_canonical(value))
    audit_files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(audit_attempt.iterdir(), key=lambda item: item.name)
    ]
    audit_terminal = audit_attempt / "operator-attempt.json"
    audit_terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.operator-attempt.v1",
                "experiment_id": "WM-001",
                "protocol_version": "1.9.0",
                "assurance": dict(_ASSURANCE),
                "kind": "audit",
                "lane": "development",
                "status": "accepted",
                "inputs": producer_rows,
                "primary": {
                    "producer_root": str(producer),
                    "audit_file": "independent-audit.json",
                    "executions": [
                        "audit-execution-01.execution.json",
                        "audit-execution-02.execution.json",
                    ],
                    "execution_failures": [],
                    "reproduction_file": "audit-reproduction.json",
                    "reproduction_runtime_file": "audit-execution-02.runtime.json",
                    "claim_file": None,
                },
                "error": None,
                "files": audit_files,
                "file_count": len(audit_files),
                "manifest_excludes": ["operator-attempt.json"],
            }
        )
    )
    audit_completion = _completion_marker(
        completion_root,
        audit_terminal,
    )
    os.link(audit_terminal, audit_completion)
    audit_input_rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (
            *sorted(audit_attempt.iterdir(), key=lambda item: item.name),
            audit_completion,
        )
    ]
    closure = results / "development" / "development-closure-v1.9.0.json"
    closure_value = {
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
        "producer_root": str(producer),
        "qualification_archive": {
            "canonical_path": "qualification.tar",
            "members": [
                {
                    "path": "producer/producer-manifest.json",
                    "sha256": producer_rows[0]["sha256"],
                },
                {
                    "path": "producer/result.json",
                    "sha256": producer_rows[1]["sha256"],
                },
            ],
        },
    }
    closure.write_bytes(_canonical(closure_value))
    closure_attempt = results / "operator-v1.9" / "closures" / "development-closure-v1.9.0"
    closure_attempt.mkdir(parents=True)
    fresh_reopen = {
        "schema": "prospect.wm001.development-closure-fresh-reopen.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.9.0",
        "mode": "fresh-closure-reopen",
        "challenge": "1" * 64,
        "requesting_process_id": 100,
        "verifier_process_id": 101,
        "matrix_contract_sha256": (
            "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
        ),
        "development_closure_sha256": hashlib.sha256(
            closure.read_bytes()
        ).hexdigest(),
        "producer_manifest_sha256": producer_rows[0]["sha256"],
        "raw_result_sha256": producer_rows[1]["sha256"],
        "passed": True,
    }
    fresh_path = closure_attempt / "fresh-runtime-reopen.json"
    fresh_path.write_bytes(_canonical(fresh_reopen))
    fresh_payload = fresh_path.read_bytes()
    closure_reference = {
        "schema": "prospect.wm001.closure-reference.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.9.0",
        "closure_marker": str(closure),
        "closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
        "qualification_archive": closure_value["qualification_archive"],
        "producer_root": closure_value["producer_root"],
        "audit_attempt": str(audit_attempt),
        "audit_attempt_manifest_sha256": hashlib.sha256(
            audit_terminal.read_bytes()
        ).hexdigest(),
        "fresh_reopen_file": "fresh-runtime-reopen.json",
        "fresh_reopen_sha256": hashlib.sha256(
            fresh_payload
        ).hexdigest(),
    }
    reference_path = closure_attempt / "closure-reference.json"
    reference_path.write_bytes(_canonical(closure_reference))
    reference_payload = reference_path.read_bytes()
    closure_terminal = closure_attempt / "operator-attempt.json"
    closure_terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.operator-attempt.v1",
                "experiment_id": "WM-001",
                "protocol_version": "1.9.0",
                "assurance": dict(_ASSURANCE),
                "kind": "closure",
                "lane": "development",
                "status": "accepted",
                "inputs": [*producer_rows, *audit_input_rows],
                "primary": {"closure_reference_file": "closure-reference.json"},
                "error": None,
                "files": [
                    {
                        "path": "closure-reference.json",
                        "bytes": len(reference_payload),
                        "sha256": hashlib.sha256(reference_payload).hexdigest(),
                    },
                    {
                        "path": "fresh-runtime-reopen.json",
                        "bytes": len(fresh_payload),
                        "sha256": hashlib.sha256(fresh_payload).hexdigest(),
                    },
                ],
                "file_count": 2,
                "manifest_excludes": ["operator-attempt.json"],
            }
        )
    )
    closure_completion = _completion_marker(
        completion_root,
        closure_terminal,
    )
    os.link(closure_terminal, closure_completion)

    binding = dict(binding)
    source = binding.get("source")
    source = dict(source) if isinstance(source, dict) else {}
    source.update(
        {
            "test_report_file": report.name,
            "test_report_bytes": report.stat().st_size,
            "test_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "test_log_files": [],
        }
    )
    binding["source"] = source
    development = binding.get("development_qualification")
    development = dict(development) if isinstance(development, dict) else {}
    development.update(
        {
            "closure_bytes": closure.stat().st_size,
            "closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
        }
    )
    binding["development_qualification"] = development
    attempt = results / "operator-v1.9" / "bindings" / "formal-binding-v1.9.0"
    attempt.mkdir(parents=True)
    binding_path = attempt / "formal-binding.json"
    binding_path.write_bytes(_canonical(binding))
    binding_payload = binding_path.read_bytes()
    terminal = attempt / "operator-attempt.json"
    terminal.write_bytes(
        _canonical(
            {
                "schema": "prospect.wm001.operator-attempt.v1",
                "experiment_id": "WM-001",
                "protocol_version": "1.9.0",
                "assurance": dict(_ASSURANCE),
                "kind": "binding",
                "lane": None,
                "status": "accepted",
                "inputs": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (
                        report,
                        closure,
                        closure_terminal,
                        closure_completion,
                    )
                ],
                "primary": {"binding_file": "formal-binding.json"},
                "error": None,
                "files": [
                    {
                        "path": "formal-binding.json",
                        "bytes": len(binding_payload),
                        "sha256": hashlib.sha256(binding_payload).hexdigest(),
                    }
                ],
                "file_count": 1,
                "manifest_excludes": ["operator-attempt.json"],
            }
        )
    )
    marker = _completion_marker(completion_root, terminal)
    if terminal_state == "finalized":
        os.link(terminal, marker)
    elif terminal_state == "copied-marker":
        stray_terminal = repository / "stray-binding-terminal.json"
        os.link(terminal, stray_terminal)
        shutil.copyfile(terminal, marker)
        stray_marker = completion_root / "stray-binding-marker.json"
        os.link(marker, stray_marker)
    elif terminal_state != "unfinalized":
        raise AssertionError(f"unknown terminal state: {terminal_state}")
    return binding_path, terminal, marker


def _minimal_formal_binding() -> dict[str, object]:
    return {
        "schema": "prospect.world-model-lifecycle.formal-binding.v9",
        "experiment_id": "WM-001",
        "assurance": dict(_ASSURANCE),
    }


def _prospective_runtime_seal_value() -> dict[str, object]:
    return {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.9.0",
        "assurance": dict(_ASSURANCE),
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "worktree_clean": True,
        "python": {"version": [3, 11, 0]},
        "required_flags": {"isolated": 1},
        "process_environment": {"LC_ALL": "C.UTF-8"},
        "bootstrap_source_sha256": "c" * 64,
        "standard_library": {"sha256": "d" * 64},
        "package_roots": [],
        "package_ownership": {},
    }


def _producer_receipt(
    repository: Path,
    terminal: Path,
    *,
    logical_exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    monkeypatch.chdir(repository)
    registration = producer_bootstrap._capture_outer_terminal(
        terminal,
        logical_exit_code=logical_exit_code,
    )
    read_descriptor, write_descriptor = os.pipe()
    try:
        producer_bootstrap._emit_outer_receipt(
            write_descriptor,
            registration,
        )
        os.close(write_descriptor)
        write_descriptor = -1
        receipt_payload = launch_bootstrap._read_receipt(read_descriptor)
        assert isinstance(receipt_payload, bytes)
        return receipt_payload
    finally:
        if write_descriptor >= 0:
            os.close(write_descriptor)
        os.close(read_descriptor)
        terminal_descriptor = registration["descriptor"]
        assert isinstance(terminal_descriptor, int)
        os.close(terminal_descriptor)


def test_producer_receipt_commits_exact_terminal_as_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    receipt = _producer_receipt(
        repository,
        terminal,
        logical_exit_code=1,
        monkeypatch=monkeypatch,
    )

    assert (
        launch_bootstrap._commit_outer_receipt(
            receipt,
            repository=repository,
            completion_root=completion_root,
        )
        == 1
    )

    marker = _completion_marker(completion_root, terminal)
    assert marker.read_bytes() == terminal.read_bytes()
    assert os.path.samefile(marker, terminal)
    assert terminal.stat().st_nlink == marker.stat().st_nlink == 2


def _remove_assurance(receipt: dict[str, object]) -> None:
    del receipt["assurance"]


def _overstate_tamper_resistance(receipt: dict[str, object]) -> None:
    assurance = receipt["assurance"]
    assert isinstance(assurance, dict)
    assurance["tamper_resistant"] = True


def _overstate_external_attestation(receipt: dict[str, object]) -> None:
    assurance = receipt["assurance"]
    assert isinstance(assurance, dict)
    assurance["external_attestation"] = True


@pytest.mark.parametrize("legacy_mode", [None, "True"])
def test_both_bootstraps_require_exact_bound_torchrl_legacy_mode(
    monkeypatch: pytest.MonkeyPatch,
    legacy_mode: str | None,
) -> None:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    if legacy_mode is not None:
        environment["LAZY_LEGACY_OP"] = legacy_mode
    monkeypatch.setattr(launch_bootstrap.os, "environ", environment)

    with pytest.raises(launch_bootstrap.LaunchError, match="exact safe runtime"):
        launch_bootstrap._environment()
    with pytest.raises(producer_bootstrap.BootstrapError, match="exact safe runtime"):
        producer_bootstrap._environment()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("PYGAME_HIDE_SUPPORT_PROMPT", None, id="missing-pygame-prompt"),
        pytest.param("PYGAME_HIDE_SUPPORT_PROMPT", "show", id="wrong-pygame-prompt"),
        pytest.param("SDL_AUDIODRIVER", None, id="missing-sdl-audio"),
        pytest.param("SDL_AUDIODRIVER", "alsa", id="wrong-sdl-audio"),
    ],
)
def test_both_bootstraps_require_exact_bound_gymnasium_defaults(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
) -> None:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    if value is None:
        del environment[name]
    else:
        environment[name] = value
    monkeypatch.setattr(launch_bootstrap.os, "environ", environment)

    with pytest.raises(launch_bootstrap.LaunchError, match="exact safe runtime"):
        launch_bootstrap._environment()
    with pytest.raises(producer_bootstrap.BootstrapError, match="exact safe runtime"):
        producer_bootstrap._environment()


@pytest.mark.parametrize(
    ("module", "error_type"),
    [
        pytest.param(
            launch_bootstrap,
            launch_bootstrap.LaunchError,
            id="outer-launcher",
        ),
        pytest.param(
            producer_bootstrap,
            producer_bootstrap.BootstrapError,
            id="producer-bootstrap",
        ),
    ],
)
def test_bootstraps_drop_absent_search_entries_and_reject_ambient_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    error_type: type[Exception],
) -> None:
    stdlib = tmp_path / "stdlib"
    dynload = stdlib / "lib-dynload"
    ambient = tmp_path / "ambient"
    dynload.mkdir(parents=True)
    ambient.mkdir()
    absent_zip = tmp_path / "python312.zip"
    monkeypatch.setattr(
        module.sysconfig,
        "get_path",
        lambda name: str(stdlib),
    )
    monkeypatch.setattr(
        module.sys,
        "path",
        [str(absent_zip), str(stdlib), str(dynload)],
    )

    assert module._sanitize_module_search_path() == (
        str(stdlib),
        str(dynload),
    )
    assert module.sys.path == [str(stdlib), str(dynload)]

    module.sys.path[:] = [str(stdlib), str(ambient)]
    with pytest.raises(
        error_type,
        match="ambient import root",
    ):
        module._sanitize_module_search_path()


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(_remove_assurance, id="missing-assurance"),
        pytest.param(
            _overstate_tamper_resistance,
            id="overstated-tamper-resistance",
        ),
        pytest.param(
            _overstate_external_attestation,
            id="overstated-external-attestation",
        ),
    ],
)
def test_outer_commit_rejects_missing_or_overstated_assurance(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    receipt = _receipt(terminal)
    mutation(receipt)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="receipt is malformed",
    ):
        launch_bootstrap._commit_outer_receipt(
            _canonical(receipt),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert not _completion_marker(completion_root, terminal).exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("terminal_bytes", 999, "differs from its identity"),
        ("terminal_sha256", "0" * 64, "differs from its identity"),
        ("terminal_path", "/tmp/not-a-wm001-result.json", "outside results"),
    ],
)
def test_outer_commit_rejects_terminal_identity_or_path_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    receipt = _receipt(terminal)
    receipt[field] = replacement

    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._commit_outer_receipt(
            _canonical(receipt),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert not _completion_marker(completion_root, terminal).exists()


def test_copied_completion_file_cannot_substitute_for_hardlink(
    tmp_path: Path,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    marker = _completion_marker(completion_root, terminal)
    shutil.copyfile(terminal, marker)
    assert marker.read_bytes() == terminal.read_bytes()
    assert not os.path.samefile(marker, terminal)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="completion marker already exists",
    ):
        launch_bootstrap._commit_outer_receipt(
            _canonical(_receipt(terminal)),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert marker.stat().st_nlink == 1
    assert not os.path.samefile(marker, terminal)


def test_hardlink_failure_leaves_terminal_unfinalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    marker = _completion_marker(completion_root, terminal)

    def fail_link(
        _source: os.PathLike[str] | str,
        _destination: os.PathLike[str] | str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        del src_dir_fd, dst_dir_fd, follow_symlinks
        raise OSError("injected hardlink failure")

    monkeypatch.setattr(launch_bootstrap.os, "link", fail_link)

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="completion hardlink failed",
    ):
        launch_bootstrap._commit_outer_receipt(
            _canonical(_receipt(terminal)),
            repository=repository,
            completion_root=completion_root,
        )

    assert terminal.stat().st_nlink == 1
    assert not marker.exists()


@pytest.mark.parametrize("marker_state", ["missing", "wrong-inode"])
def test_completed_runtime_seal_rejects_missing_or_forged_marker(
    tmp_path: Path,
    marker_state: str,
) -> None:
    repository, completion_root, terminal = _repository_paths(tmp_path)
    results = completion_root.parents[1]
    runtime_seal = (
        results
        / "development"
        / "runtime-seal-v1.9.0.json"
    )
    runtime_seal.parent.mkdir()
    runtime_seal.write_bytes(_canonical(_prospective_runtime_seal_value()))
    stray_seal_link = runtime_seal.with_name("stray-seal-link.json")
    os.link(runtime_seal, stray_seal_link)
    assert runtime_seal.stat().st_nlink == 2

    marker = _completion_marker(completion_root, runtime_seal)
    expected_message = "cannot be resolved"
    if marker_state == "wrong-inode":
        shutil.copyfile(runtime_seal, marker)
        stray_marker_link = completion_root / "stray-marker-link.json"
        os.link(marker, stray_marker_link)
        assert marker.stat().st_nlink == 2
        assert not os.path.samefile(runtime_seal, marker)
        expected_message = "not the terminal inode"

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match=expected_message,
    ):
        launch_bootstrap._open_typed_runtime_custody(
            runtime_seal,
            repository=repository,
            completion_root=completion_root,
        )

    assert runtime_seal.stat().st_nlink == 2
    assert terminal.stat().st_nlink == 1


def test_outer_accepts_only_exact_canonical_prospective_runtime_seal(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    runtime_seal = launch_bootstrap._prospective_runtime_seal(repository)
    runtime_seal.parent.mkdir()
    runtime_seal.write_bytes(_canonical(_prospective_runtime_seal_value()))
    marker = _completion_marker(completion_root, runtime_seal)
    os.link(runtime_seal, marker)

    descriptor, payload, _, expected_nlink = (
        launch_bootstrap._open_typed_runtime_custody(
            runtime_seal,
            repository=repository,
            completion_root=completion_root,
        )
    )
    try:
        assert payload == runtime_seal.read_bytes()
        assert expected_nlink == 2
    finally:
        os.close(descriptor)

    sibling = runtime_seal.with_name("alias-runtime-seal-v1.9.0.json")
    shutil.copyfile(runtime_seal, sibling)
    sibling_link = sibling.with_name("alias-runtime-seal-link.json")
    os.link(sibling, sibling_link)
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="not one canonical",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            sibling,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda value: value.pop("package_ownership"),
            id="missing-field",
        ),
        pytest.param(
            lambda value: value.__setitem__("unexpected", True),
            id="extra-field",
        ),
        pytest.param(
            lambda value: value.__setitem__("protocol_version", "1.5"),
            id="wrong-version",
        ),
    ],
)
def test_prospective_runtime_seal_requires_exact_versioned_field_set(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    runtime_seal = launch_bootstrap._prospective_runtime_seal(repository)
    runtime_seal.parent.mkdir()
    value = _prospective_runtime_seal_value()
    mutation(value)
    runtime_seal.write_bytes(_canonical(value))
    os.link(runtime_seal, _completion_marker(completion_root, runtime_seal))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="schema or assurance",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            runtime_seal,
            repository=repository,
            completion_root=completion_root,
        )


def test_runtime_seal_creation_rejects_every_noncanonical_output_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    called = False

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("producer must not execute")

    monkeypatch.setattr(launch_bootstrap.subprocess, "run", unexpected_run)
    arguments = argparse.Namespace(
        bootstrap=repository / "missing-bootstrap.py",
        create_runtime_seal=(
            repository
            / "bench"
            / "world_model_lifecycle"
            / "results"
            / "development"
            / "runtime-seal.json"
        ),
        runtime_seal=None,
        producer_arguments=[],
    )
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="sole canonical",
    ):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=repository,
            completion_root=completion_root,
        )
    assert called is False


def test_outer_accepts_only_finalized_canonical_formal_binding_attempt(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, marker = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )

    descriptor, payload, _, expected_nlink = launch_bootstrap._open_typed_runtime_custody(
        binding_path,
        repository=repository,
        completion_root=completion_root,
    )
    try:
        assert payload == binding_path.read_bytes()
        assert expected_nlink == 1
    finally:
        os.close(descriptor)
    assert binding_path.stat().st_nlink == 1
    assert terminal.stat().st_nlink == 2
    assert os.path.samefile(terminal, marker)


def test_outer_rejects_substituted_binding_authorization_input(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    manifest = json.loads(terminal.read_bytes())
    manifest["inputs"][0]["sha256"] = "0" * 64
    terminal.write_bytes(_canonical(manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="authorization inputs differ",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_rejects_substituted_closure_authorization_input(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, binding_terminal, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state="finalized",
    )
    results = repository / "bench" / "world_model_lifecycle" / "results"
    closure_terminal = (
        results
        / "operator-v1.9"
        / "closures"
        / "development-closure-v1.9.0"
        / "operator-attempt.json"
    )
    closure_completion = _completion_marker(
        completion_root,
        closure_terminal,
    )
    closure_manifest = json.loads(closure_terminal.read_bytes())
    closure_manifest["inputs"][0]["sha256"] = "0" * 64
    closure_terminal.write_bytes(_canonical(closure_manifest))
    closure_payload = closure_terminal.read_bytes()

    binding_manifest = json.loads(binding_terminal.read_bytes())
    for row in binding_manifest["inputs"]:
        if row["path"] in {
            str(closure_terminal),
            str(closure_completion),
        }:
            row["bytes"] = len(closure_payload)
            row["sha256"] = hashlib.sha256(closure_payload).hexdigest()
    binding_terminal.write_bytes(_canonical(binding_manifest))

    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="closure authorization inputs differ",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    ("terminal_state", "message"),
    [
        ("unfinalized", "exactly 2 link"),
        ("copied-marker", "not the terminal inode"),
    ],
)
def test_outer_rejects_unfinalized_or_copied_binding_completion(
    tmp_path: Path,
    terminal_state: str,
    message: str,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    binding_path, _, _ = _formal_binding_attempt(
        repository,
        completion_root,
        binding=_minimal_formal_binding(),
        terminal_state=terminal_state,
    )

    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._open_typed_runtime_custody(
            binding_path,
            repository=repository,
            completion_root=completion_root,
        )


def test_outer_rejects_direct_or_self_finalized_formal_binding_copy(
    tmp_path: Path,
) -> None:
    repository, completion_root, _ = _repository_paths(tmp_path)
    results = completion_root.parents[1]
    copied = results / "formal-binding-copy.json"
    copied.write_bytes(_canonical(_minimal_formal_binding()))

    with pytest.raises(launch_bootstrap.LaunchError):
        launch_bootstrap._open_typed_runtime_custody(
            copied,
            repository=repository,
            completion_root=completion_root,
        )

    copied_marker = _completion_marker(completion_root, copied)
    os.link(copied, copied_marker)
    with pytest.raises(
        launch_bootstrap.LaunchError,
        match="not one canonical",
    ):
        launch_bootstrap._open_typed_runtime_custody(
            copied,
            repository=repository,
            completion_root=completion_root,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--bootstrap",
            "/tmp/producer-a.py",
            "--bootstrap",
            "/tmp/producer-b.py",
            "--runtime-seal",
            "/tmp/runtime.json",
            "preformal-runtime",
        ],
        [
            "--bootstrap",
            "/tmp/producer.py",
            "--runtime-seal",
            "/tmp/runtime-a.json",
            "--runtime-seal",
            "/tmp/runtime-b.json",
            "preformal-runtime",
        ],
    ],
)
def test_outer_parser_rejects_repeated_custody_options(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        launch_bootstrap._parser().parse_args(arguments)


@pytest.mark.parametrize(
    ("producer_arguments", "message"),
    [
        (["--", "--", "preformal-runtime"], "ambiguous separator"),
        (
            ["--", "--runtime-seal=/tmp/second.json", "preformal-runtime"],
            "second custody source",
        ),
        (
            ["--", "--bootstrap-fd", "7", "preformal-runtime"],
            "second custody source",
        ),
    ],
)
def test_outer_run_rejects_ambiguous_or_mixed_custody_before_open(
    tmp_path: Path,
    producer_arguments: list[str],
    message: str,
) -> None:
    arguments = argparse.Namespace(
        bootstrap=tmp_path / "absent-producer.py",
        runtime_seal=tmp_path / "absent-runtime.json",
        create_runtime_seal=None,
        producer_arguments=producer_arguments,
    )
    with pytest.raises(launch_bootstrap.LaunchError, match=message):
        launch_bootstrap._run_locked(
            arguments,
            environment={},
            repository=tmp_path,
            completion_root=tmp_path,
        )


def test_producer_rejects_second_custody_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "producer_bootstrap.py",
            "--binding-entry",
            "--outer-receipt-fd=91",
        ],
    )
    with pytest.raises(
        producer_bootstrap.BootstrapError,
        match="second custody source",
    ):
        producer_bootstrap._reject_second_custody_source()


def test_runtime_lock_is_repository_wide_across_working_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "canonical-repository"
    repository.mkdir()
    first_descriptor = launch_bootstrap._acquire_runtime_lock(repository)
    other_working_directory = tmp_path / "other-working-directory"
    other_working_directory.mkdir()
    monkeypatch.chdir(other_working_directory)
    try:
        with pytest.raises(
            launch_bootstrap.LaunchError,
            match="another WM-001 v1.9 outer invocation",
        ):
            launch_bootstrap._acquire_runtime_lock(repository)
    finally:
        fcntl.flock(first_descriptor, fcntl.LOCK_UN)
        os.close(first_descriptor)


def _run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed: {command!r}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def _install_minimal_distribution(virtualenv: Path) -> None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    package_root = virtualenv / "lib" / version / "site-packages"
    bench = package_root / "bench"
    lifecycle = bench / "world_model_lifecycle"
    metadata = package_root / "wm001_bootstrap_fixture-1.0.dist-info"
    lifecycle.mkdir(parents=True)
    metadata.mkdir()
    (bench / "__init__.py").write_text(
        '"""WM-001 bootstrap fixture namespace."""\n',
        encoding="utf-8",
    )
    (lifecycle / "__init__.py").write_text(
        '"""WM-001 bootstrap fixture lifecycle."""\n',
        encoding="utf-8",
    )
    shutil.copyfile(
        Path(producer_bootstrap.__file__),
        lifecycle / "producer_bootstrap.py",
    )
    (lifecycle / "_fixture_entry.py").write_text(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from bench.world_model_lifecycle.producer_bootstrap import (\n"
        "    register_outer_terminal,\n"
        ")\n"
        "\n"
        "\n"
        "LOGICAL_CODES = {\n"
        "    'binding': 0,\n"
        "    'audit': 1,\n"
        "    'closure': 2,\n"
        "    'adjudication': 0,\n"
        "    'default-run': 1,\n"
        "}\n"
        "\n"
        "\n"
        "def _payload(value):\n"
        "    return (\n"
        "        json.dumps(value, sort_keys=True, separators=(',', ':'))\n"
        "        + '\\n'\n"
        "    )\n"
        "\n"
        "\n"
        "def read_only(entry):\n"
        "    sys.stdout.write(_payload({\n"
        "        'entry': entry,\n"
        "        'schema': 'prospect.wm001.bootstrap-fixture.v1',\n"
        "    }))\n"
        "    return 0\n"
        "\n"
        "\n"
        "def publish(entry):\n"
        "    logical_exit_code = LOGICAL_CODES[entry]\n"
        "    terminal = (\n"
        "        Path.cwd()\n"
        "        / 'bench'\n"
        "        / 'world_model_lifecycle'\n"
        "        / 'results'\n"
        "        / 'fixture-dispatch'\n"
        "        / entry\n"
        "        / 'terminal-manifest.json'\n"
        "    )\n"
        "    terminal.parent.mkdir(parents=True)\n"
        "    terminal.write_text(_payload({\n"
        "        'entry': entry,\n"
        "        'logical_exit_code': logical_exit_code,\n"
        "        'schema': 'prospect.wm001.bootstrap-fixture-terminal.v1',\n"
        "    }), encoding='utf-8')\n"
        "    register_outer_terminal(\n"
        "        terminal,\n"
        "        logical_exit_code=logical_exit_code,\n"
        "    )\n"
        "    return logical_exit_code\n",
        encoding="utf-8",
    )
    (lifecycle / "restore_eval.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import read_only\n"
        "\n"
        "\n"
        "def main():\n"
        "    return read_only('restore-eval')\n",
        encoding="utf-8",
    )
    (lifecycle / "operator.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import publish\n"
        "\n"
        "\n"
        "def binding_main():\n"
        "    return publish('binding')\n"
        "\n"
        "\n"
        "def audit_main():\n"
        "    return publish('audit')\n"
        "\n"
        "\n"
        "def closure_main():\n"
        "    return publish('closure')\n",
        encoding="utf-8",
    )
    (lifecycle / "adjudication.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import publish\n"
        "\n"
        "\n"
        "def main():\n"
        "    return publish('adjudication')\n",
        encoding="utf-8",
    )
    (lifecycle / "preformal.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import read_only\n"
        "\n"
        "\n"
        "def runtime_main():\n"
        "    return read_only('preformal-runtime')\n",
        encoding="utf-8",
    )
    (lifecycle / "run.py").write_text(
        "from bench.world_model_lifecycle._fixture_entry import publish\n"
        "\n"
        "\n"
        "def main():\n"
        "    return publish('default-run')\n",
        encoding="utf-8",
    )
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: wm001-bootstrap-fixture\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "bench/__init__.py,,\n"
        "bench/world_model_lifecycle/__init__.py,,\n"
        "bench/world_model_lifecycle/_fixture_entry.py,,\n"
        "bench/world_model_lifecycle/adjudication.py,,\n"
        "bench/world_model_lifecycle/operator.py,,\n"
        "bench/world_model_lifecycle/preformal.py,,\n"
        "bench/world_model_lifecycle/producer_bootstrap.py,,\n"
        "bench/world_model_lifecycle/restore_eval.py,,\n"
        "bench/world_model_lifecycle/run.py,,\n"
        "wm001_bootstrap_fixture-1.0.dist-info/METADATA,,\n"
        "wm001_bootstrap_fixture-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )


def _formal_binding_from_runtime_seal(
    seal: dict[str, object],
) -> dict[str, object]:
    python = seal["python"]
    assert isinstance(python, dict)
    return {
        "schema": "prospect.world-model-lifecycle.formal-binding.v9",
        "experiment_id": "WM-001",
        "assurance": dict(_ASSURANCE),
        "source": {
            "git_commit": seal["git_commit"],
            "git_tree": seal["git_tree"],
            "worktree_clean": seal["worktree_clean"],
            "execution_source_sha256": {
                "producer_bootstrap.py": seal["bootstrap_source_sha256"],
            },
        },
        "dependencies": {
            "python_executable": python["executable"],
            "python_executable_sha256": python["sha256"],
            "standard_library": seal["standard_library"],
            "package_roots": seal["package_roots"],
            "package_ownership": seal["package_ownership"],
        },
        "runtime": {
            "python_flags": seal["required_flags"],
            "process_environment": seal["process_environment"],
        },
    }


def test_real_outer_launcher_dispatches_actual_producer_bootstrap(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "canonical-repository"
    lifecycle = repository / "bench" / "world_model_lifecycle"
    lifecycle.mkdir(parents=True)
    launcher = lifecycle / "launch_bootstrap.py"
    producer = lifecycle / "producer_bootstrap.py"
    shutil.copyfile(Path(launch_bootstrap.__file__), launcher)
    shutil.copyfile(Path(producer_bootstrap.__file__), producer)
    (lifecycle / "protocol.json").write_text("{}\n", encoding="utf-8")
    (repository / ".gitignore").write_text(
        "/bench/world_model_lifecycle/results/\n",
        encoding="utf-8",
    )

    _run_checked(["git", "init", "--quiet"], cwd=repository)
    _run_checked(
        ["git", "config", "user.email", "wm001-fixture@example.invalid"],
        cwd=repository,
    )
    _run_checked(
        ["git", "config", "user.name", "WM-001 fixture"],
        cwd=repository,
    )
    _run_checked(["git", "add", "."], cwd=repository)
    _run_checked(
        ["git", "commit", "--quiet", "-m", "bootstrap fixture"],
        cwd=repository,
    )

    virtualenv = tmp_path / "isolated-runtime"
    _run_checked(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            str(virtualenv),
        ]
    )
    _install_minimal_distribution(virtualenv)

    runtime_seal = (
        lifecycle
        / "results"
        / "development"
        / "runtime-seal-v1.9.0.json"
    )
    runtime_seal.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    completed = subprocess.run(
        [
            str(virtualenv / "bin" / "python"),
            "-I",
            "-S",
            "-B",
            str(launcher),
            "--bootstrap",
            str(producer),
            "--create-runtime-seal",
            str(runtime_seal),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"

    seal = json.loads(runtime_seal.read_bytes())
    assert seal["schema"] == "prospect.wm001.runtime-seal.v1"
    assert seal["assurance"] == _ASSURANCE
    assert seal["worktree_clean"] is True
    completion_root = lifecycle / "results" / "outer-completions" / "v1.9"
    marker = _completion_marker(completion_root, runtime_seal)
    assert marker.read_bytes() == runtime_seal.read_bytes()
    assert os.path.samefile(marker, runtime_seal)
    assert runtime_seal.stat().st_nlink == 2

    runtime_prefix = [
        str(virtualenv / "bin" / "python"),
        "-I",
        "-S",
        "-B",
        str(launcher),
        "--bootstrap",
        str(producer),
        "--runtime-seal",
        str(runtime_seal),
        "--",
    ]
    read_only_entries = [
        (["--restore-eval-entry"], "restore-eval"),
        (["preformal-runtime"], "preformal-runtime"),
    ]
    for producer_arguments, entry in read_only_entries:
        markers_before = set(completion_root.glob("*.json"))
        runtime = subprocess.run(
            [*runtime_prefix, *producer_arguments],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert runtime.returncode == 0, f"entry: {entry}\nstdout:\n{runtime.stdout}\nstderr:\n{runtime.stderr}"
        assert json.loads(runtime.stdout) == {
            "entry": entry,
            "schema": "prospect.wm001.bootstrap-fixture.v1",
        }
        assert runtime.stderr == ""
        assert set(completion_root.glob("*.json")) == markers_before
        assert os.path.samefile(marker, runtime_seal)
        assert runtime_seal.stat().st_nlink == 2

    formal_binding = _formal_binding_from_runtime_seal(seal)
    formal_binding_path, binding_terminal, binding_marker = _formal_binding_attempt(
        repository,
        completion_root,
        binding=formal_binding,
        terminal_state="unfinalized",
    )

    def invoke_formal(path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(virtualenv / "bin" / "python"),
                "-I",
                "-S",
                "-B",
                str(launcher),
                "--bootstrap",
                str(producer),
                "--runtime-seal",
                str(path),
                "--",
                "preformal-runtime",
            ],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    unfinalized = invoke_formal(formal_binding_path)
    assert unfinalized.returncode != 0
    assert not binding_marker.exists()

    stray_terminal = repository / "stray-binding-terminal.json"
    os.link(binding_terminal, stray_terminal)
    shutil.copyfile(binding_terminal, binding_marker)
    stray_marker = completion_root / "stray-binding-marker.json"
    os.link(binding_marker, stray_marker)
    copied_marker = invoke_formal(formal_binding_path)
    assert copied_marker.returncode != 0
    assert not os.path.samefile(binding_terminal, binding_marker)
    stray_marker.unlink()
    binding_marker.unlink()
    stray_terminal.unlink()
    os.link(binding_terminal, binding_marker)

    direct_copy = lifecycle / "results" / "formal-binding-copy.json"
    shutil.copyfile(formal_binding_path, direct_copy)
    direct = invoke_formal(direct_copy)
    assert direct.returncode != 0
    direct_marker = _completion_marker(completion_root, direct_copy)
    os.link(direct_copy, direct_marker)
    self_finalized_copy = invoke_formal(direct_copy)
    assert self_finalized_copy.returncode != 0
    direct_marker.unlink()

    markers_before = set(completion_root.glob("*.json"))
    formal_runtime = invoke_formal(formal_binding_path)
    assert formal_runtime.returncode == 0, f"stdout:\n{formal_runtime.stdout}\nstderr:\n{formal_runtime.stderr}"
    assert json.loads(formal_runtime.stdout) == {
        "entry": "preformal-runtime",
        "schema": "prospect.wm001.bootstrap-fixture.v1",
    }
    assert formal_runtime.stderr == ""
    assert set(completion_root.glob("*.json")) == markers_before
    assert formal_binding_path.stat().st_nlink == 1
    assert binding_terminal.stat().st_nlink == 2
    assert os.path.samefile(binding_terminal, binding_marker)

    publisher_entries = [
        (["--binding-entry"], "binding", 0),
        (["--audit-entry"], "audit", 1),
        (["--closure-entry"], "closure", 2),
        (["--adjudication-entry"], "adjudication", 0),
        (["default-run"], "default-run", 1),
    ]
    for producer_arguments, entry, logical_exit_code in publisher_entries:
        markers_before = set(completion_root.glob("*.json"))
        runtime = subprocess.run(
            [*runtime_prefix, *producer_arguments],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert runtime.returncode == logical_exit_code, (
            f"entry: {entry}\nstdout:\n{runtime.stdout}\nstderr:\n{runtime.stderr}"
        )
        assert runtime.stdout == ""
        assert runtime.stderr == ""

        terminal = lifecycle / "results" / "fixture-dispatch" / entry / "terminal-manifest.json"
        assert json.loads(terminal.read_bytes()) == {
            "entry": entry,
            "logical_exit_code": logical_exit_code,
            "schema": "prospect.wm001.bootstrap-fixture-terminal.v1",
        }
        terminal_marker = _completion_marker(completion_root, terminal)
        assert terminal_marker.read_bytes() == terminal.read_bytes()
        assert os.path.samefile(terminal_marker, terminal)
        assert terminal.stat().st_nlink == 2
        assert set(completion_root.glob("*.json")) == {
            *markers_before,
            terminal_marker,
        }
