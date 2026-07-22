from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle import rehearsal
from bench.world_model_lifecycle.assurance import ASSURANCE


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


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)[:-1]).hexdigest()


def _row(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _absolute_row(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path.resolve(strict=True)
    source = repo / "bench" / "world_model_lifecycle"
    source.mkdir(parents=True)
    launch = source / "launch_bootstrap.py"
    bootstrap = source / "producer_bootstrap.py"
    launch.write_bytes(b"# sealed launch fixture\n")
    bootstrap.write_bytes(b"# sealed producer fixture\n")
    results = source / "results"
    operator_root = results / "operator-v1.17"
    attempts = operator_root / "rehearsals"
    attempt = attempts / "accepted-binding-rehearsal-v1.17.0"
    claim_root = results / "rehearsals" / "v1.17"
    completions = results / "outer-completions" / "v1.17"
    binding_path = (
        operator_root
        / "bindings"
        / "formal-binding-v1.17.0"
        / "formal-binding.json"
    )
    values = {
        "REPO": repo,
        "RESULTS_ROOT": results,
        "OPERATOR_RESULTS_ROOT": operator_root,
        "REHEARSAL_ATTEMPTS_ROOT": attempts,
        "REHEARSAL_ATTEMPT_PATH": attempt,
        "REHEARSAL_CLAIM_ROOT": claim_root,
        "OUTER_COMPLETIONS_ROOT": completions,
        "FORMAL_BINDING_PATH": binding_path,
        "LAUNCH_BOOTSTRAP_PATH": launch,
        "PRODUCER_BOOTSTRAP_PATH": bootstrap,
        "CLAIM_PATH": attempt / rehearsal.CLAIM_NAME,
        "TERMINAL_PATH": attempt / rehearsal.TERMINAL_NAME,
        "STDOUT_PATH": attempt / rehearsal.STDOUT_NAME,
        "STDERR_PATH": attempt / rehearsal.STDERR_NAME,
        "OUTER_RECEIPT_PATH": attempt / rehearsal.OUTER_RECEIPT_NAME,
    }
    for name, value in values.items():
        monkeypatch.setattr(rehearsal, name, value)


def _binding() -> dict[str, object]:
    return {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "experiment_id": "WM-001",
        "protocol": {"version": "1.17.0"},
        "assurance": dict(ASSURANCE),
        "runtime": {"device": "cpu"},
        "dependencies": {
            "packages": [
                {
                    "name": "python",
                    "version": "3.12.0",
                    "distribution_sha256": "1" * 64,
                    "declared_file_count": 1,
                    "editable": False,
                }
            ],
            "package_roots": [
                {
                    "semantics_id": "prospect.wm001.package-root.v2",
                    "path": "/runtime/site-packages",
                    "file_count": 7,
                    "directory_count": 3,
                    "total_bytes": 70,
                    "inventory_sha256": "2" * 64,
                }
            ],
            "standard_library": {
                "semantics_id": "prospect.wm001.standard-library.v2",
                "path": "/runtime/stdlib",
                "file_count": 11,
                "directory_count": 4,
                "total_bytes": 110,
                "inventory_sha256": "3" * 64,
            },
            "package_ownership": {
                "semantics_id": "prospect.wm001.package-ownership.v1",
                "root": "/runtime/site-packages",
                "file_count": 7,
                "directory_count": 3,
                "shared_file_count": 0,
                "identity_sha256": "4" * 64,
            },
        },
        "development_qualification": {
            "matrix_contract_sha256": "5" * 64,
        },
        "audit_execution": {
            "restart_runtime_conformance_report_sha256": "6" * 64,
            "restart_runtime_execution_receipt_sha256": "7" * 64,
            "restart_runtime_support_files": [
                "producer_bootstrap.py",
                "protocol.json",
                "schemas/raw-result.schema.json",
            ],
            "restart_runtime_repeat_count": 3,
            "restart_runtime_path_descriptor_equal": True,
            "repeat_count": 3,
            "path_descriptor_equal": True,
            "passed": True,
        },
    }


def _stdout(binding: dict[str, object]) -> bytes:
    dependencies = binding["dependencies"]
    assert isinstance(dependencies, dict)
    execution = binding["audit_execution"]
    assert isinstance(execution, dict)
    inventory = {
        "packages": dependencies["packages"],
        "package_roots": dependencies["package_roots"],
        "standard_library": dependencies["standard_library"],
        "package_ownership": dependencies["package_ownership"],
    }
    fresh = {
        "schema": "prospect.wm001.fresh-runtime-identity-conformance.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.17.0",
        "mode": "fresh-identity-conformance",
        "challenge": "8" * 64,
        "requesting_process_id": 100,
        "verifier_process_id": 101,
        "matrix_contract_sha256": "5" * 64,
        "passed": True,
    }
    value = {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "bootstrap-inventory-conformance",
        "device": "cpu",
        "passed": True,
        "inventory": inventory,
        "inventory_sha256": _digest(inventory),
        "conformance_sha256": _digest(execution),
        "fresh_runtime_identity_conformance": fresh,
        "fresh_runtime_identity_conformance_sha256": _digest(fresh),
        "restart_runtime_conformance_report_sha256": "6" * 64,
        "restart_runtime_execution_receipt_sha256": "7" * 64,
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
    return _canonical(value)


def _rewrite_terminal(package: dict[str, Any]) -> None:
    attempt = rehearsal.REHEARSAL_ATTEMPT_PATH
    terminal = package["terminal"]
    terminal.update(
        {
            "claim_bytes": (attempt / rehearsal.CLAIM_NAME).stat().st_size,
            "claim_sha256": hashlib.sha256(
                (attempt / rehearsal.CLAIM_NAME).read_bytes()
            ).hexdigest(),
            "stdout_bytes": (attempt / rehearsal.STDOUT_NAME).stat().st_size,
            "stdout_sha256": hashlib.sha256(
                (attempt / rehearsal.STDOUT_NAME).read_bytes()
            ).hexdigest(),
            "stderr_bytes": (attempt / rehearsal.STDERR_NAME).stat().st_size,
            "stderr_sha256": hashlib.sha256(
                (attempt / rehearsal.STDERR_NAME).read_bytes()
            ).hexdigest(),
            "outer_receipt_bytes": (
                attempt / rehearsal.OUTER_RECEIPT_NAME
            ).stat().st_size,
            "outer_receipt_sha256": hashlib.sha256(
                (attempt / rehearsal.OUTER_RECEIPT_NAME).read_bytes()
            ).hexdigest(),
        }
    )
    terminal["files"] = sorted(
        [
            _row(attempt / name)
            for name in (
                rehearsal.CLAIM_NAME,
                rehearsal.STDOUT_NAME,
                rehearsal.STDERR_NAME,
                rehearsal.OUTER_RECEIPT_NAME,
            )
        ],
        key=lambda row: str(row["path"]),
    )
    terminal["file_count"] = len(terminal["files"])
    rehearsal.TERMINAL_PATH.write_bytes(_canonical(terminal))


def _package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "accepted",
) -> dict[str, Any]:
    _patch_paths(tmp_path, monkeypatch)
    binding = _binding()
    binding_path = rehearsal.FORMAL_BINDING_PATH
    binding_path.parent.mkdir(parents=True)
    binding_path.write_bytes(_canonical(binding))
    monkeypatch.setattr(rehearsal, "_verified_binding", lambda path: binding)
    binding_payload = binding_path.read_bytes()
    binding_sha256 = hashlib.sha256(binding_payload).hexdigest()
    attempt = rehearsal.REHEARSAL_ATTEMPT_PATH
    claim_root = rehearsal.REHEARSAL_CLAIM_ROOT
    completion_root = rehearsal.OUTER_COMPLETIONS_ROOT
    attempt.mkdir(parents=True)
    claim_root.mkdir(parents=True)
    completion_root.mkdir(parents=True)
    marker = rehearsal.rehearsal_claim_marker(binding_sha256)
    claim = {
        "schema": rehearsal.CLAIM_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.17.0",
        "assurance": dict(ASSURANCE),
        "status": "consumed",
        "binding_path": str(binding_path),
        "binding_bytes": len(binding_payload),
        "binding_sha256": binding_sha256,
        "attempt_path": str(attempt),
        "marker_path": str(marker),
        "launch_bootstrap_path": str(rehearsal.LAUNCH_BOOTSTRAP_PATH),
        "launch_bootstrap_bytes": rehearsal.LAUNCH_BOOTSTRAP_PATH.stat().st_size,
        "launch_bootstrap_sha256": hashlib.sha256(
            rehearsal.LAUNCH_BOOTSTRAP_PATH.read_bytes()
        ).hexdigest(),
        "producer_bootstrap_path": str(rehearsal.PRODUCER_BOOTSTRAP_PATH),
        "producer_bootstrap_bytes": rehearsal.PRODUCER_BOOTSTRAP_PATH.stat().st_size,
        "producer_bootstrap_sha256": hashlib.sha256(
            rehearsal.PRODUCER_BOOTSTRAP_PATH.read_bytes()
        ).hexdigest(),
        "command": [
            "preformal-runtime",
            "bootstrap-inventory-conformance",
            "--device",
            "cpu",
        ],
    }
    rehearsal.CLAIM_PATH.write_bytes(_canonical(claim))
    os.link(rehearsal.CLAIM_PATH, marker)
    (attempt / rehearsal.STDOUT_NAME).write_bytes(
        _stdout(binding) if status == "accepted" else b"partial stdout"
    )
    (attempt / rehearsal.STDERR_NAME).write_bytes(
        b"" if status == "accepted" else b"bounded diagnostic"
    )
    (attempt / rehearsal.OUTER_RECEIPT_NAME).write_bytes(b"")
    terminal: dict[str, object] = {
        "schema": rehearsal.TERMINAL_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.17.0",
        "assurance": dict(ASSURANCE),
        "status": status,
        "claim_file": rehearsal.CLAIM_NAME,
        "claim_marker": str(marker),
        "claim_bytes": 0,
        "claim_sha256": "0" * 64,
        "binding_path": str(binding_path),
        "binding_bytes": len(binding_payload),
        "binding_sha256": binding_sha256,
        "child_started": status == "accepted",
        "returncode": 0 if status == "accepted" else None,
        "stdout_file": rehearsal.STDOUT_NAME,
        "stdout_bytes": 0,
        "stdout_sha256": "0" * 64,
        "stderr_file": rehearsal.STDERR_NAME,
        "stderr_bytes": 0,
        "stderr_sha256": "0" * 64,
        "outer_receipt_file": rehearsal.OUTER_RECEIPT_NAME,
        "outer_receipt_bytes": 0,
        "outer_receipt_sha256": "0" * 64,
        "formal_paths_absent_before": True,
        "formal_paths_absent_after": status == "accepted",
        "phase": "complete" if status == "accepted" else "recovery",
        "error_code": None if status == "accepted" else "interrupted_after_claim",
        "files": [],
        "file_count": 0,
        "manifest_excludes": [rehearsal.TERMINAL_NAME],
    }
    package = {
        "binding": binding,
        "claim": claim,
        "marker": marker,
        "terminal": terminal,
    }
    rehearsal.TERMINAL_PATH.write_bytes(b"placeholder")
    _rewrite_terminal(package)
    completion = rehearsal.rehearsal_outer_completion()
    os.link(rehearsal.TERMINAL_PATH, completion)
    package["completion"] = completion
    return package


def test_accepted_rehearsal_verifies_and_exports_authorization_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _package(tmp_path, monkeypatch)

    assert rehearsal.verify_accepted_binding_rehearsal(
        rehearsal.FORMAL_BINDING_PATH
    ) == package["terminal"]
    rows = rehearsal.accepted_binding_rehearsal_identity_rows(
        rehearsal.FORMAL_BINDING_PATH
    )
    assert rows == {
        "claim": _absolute_row(rehearsal.CLAIM_PATH),
        "claim_marker": _absolute_row(package["marker"]),
        "terminal": _absolute_row(rehearsal.TERMINAL_PATH),
        "outer_completion": _absolute_row(package["completion"]),
    }
    assert rehearsal.accepted_binding_rehearsal_identity(
        rehearsal.FORMAL_BINDING_PATH
    ) == rows
    assert rehearsal.accepted_rehearsal_identity(
        rehearsal.FORMAL_BINDING_PATH
    ) == rows

    with rehearsal.hold_accepted_binding_rehearsal(
        rehearsal.FORMAL_BINDING_PATH
    ) as custody:
        assert custody.terminal == package["terminal"]
        assert custody.identity_rows() == rows
        custody.recheck()


@pytest.mark.parametrize(
    "role",
    (
        "binding",
        "claim",
        "terminal",
        "stdout",
        "launch-bootstrap",
        "producer-bootstrap",
    ),
)
def test_held_custody_rejects_evidence_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    _package(tmp_path, monkeypatch)
    mutation_paths = {
        "binding": rehearsal.FORMAL_BINDING_PATH,
        "claim": rehearsal.CLAIM_PATH,
        "terminal": rehearsal.TERMINAL_PATH,
        "stdout": rehearsal.STDOUT_PATH,
        "launch-bootstrap": rehearsal.LAUNCH_BOOTSTRAP_PATH,
        "producer-bootstrap": rehearsal.PRODUCER_BOOTSTRAP_PATH,
    }

    with pytest.raises(rehearsal.RehearsalEvidenceError):
        with rehearsal.hold_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        ) as custody:
            path = mutation_paths[role]
            path.write_bytes(path.read_bytes() + b"mutation")
            custody.recheck()


@pytest.mark.parametrize(
    "namespace",
    ("attempt", "attempts-root", "claim-root", "completion-root"),
)
def test_held_custody_rejects_namespace_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace: str,
) -> None:
    _package(tmp_path, monkeypatch)
    mutation_paths = {
        "attempt": rehearsal.REHEARSAL_ATTEMPT_PATH / "unexpected.json",
        "attempts-root": rehearsal.REHEARSAL_ATTEMPTS_ROOT / "unexpected",
        "claim-root": rehearsal.REHEARSAL_CLAIM_ROOT / "unexpected.json",
        "completion-root": rehearsal.OUTER_COMPLETIONS_ROOT / "unexpected.json",
    }

    with pytest.raises(rehearsal.RehearsalEvidenceError):
        with rehearsal.hold_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        ) as custody:
            mutation_paths[namespace].write_bytes(b"unexpected\n")
            custody.recheck()


def test_held_custody_exit_rechecks_without_masking_body_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="formal publication failed") as raised:
        with rehearsal.hold_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        ):
            rehearsal.STDOUT_PATH.write_bytes(b"changed after publication error")
            raise RuntimeError("formal publication failed")

    notes = getattr(raised.value, "__notes__", [])
    assert any("custody also failed on exit" in note for note in notes)


def test_held_custody_normalizes_external_verifier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path, monkeypatch)

    class SyntheticVerificationFailure(RuntimeError):
        pass

    def fail_verification(path: Path) -> dict[str, object]:
        del path
        raise SyntheticVerificationFailure("strict verifier failed")

    monkeypatch.setattr(rehearsal, "_verified_binding", fail_verification)
    with pytest.raises(
        rehearsal.RehearsalEvidenceError,
        match="custody cannot be established",
    ) as raised:
        with rehearsal.hold_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        ):
            pass
    assert isinstance(raised.value.__cause__, SyntheticVerificationFailure)


def test_failed_rehearsal_is_authenticated_but_never_authorizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _package(tmp_path, monkeypatch, status="failed")

    assert rehearsal.verify_accepted_binding_rehearsal(
        rehearsal.FORMAL_BINDING_PATH
    ) == package["terminal"]
    with pytest.raises(
        rehearsal.RehearsalEvidenceError,
        match="cannot authorize",
    ):
        rehearsal.accepted_binding_rehearsal_identity_rows(
            rehearsal.FORMAL_BINDING_PATH
        )
    with pytest.raises(
        rehearsal.RehearsalEvidenceError,
        match="cannot authorize",
    ):
        with rehearsal.hold_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        ):
            pass


def test_claim_only_state_is_not_a_terminal_rehearsal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _package(tmp_path, monkeypatch)
    rehearsal.TERMINAL_PATH.unlink()
    package["completion"].unlink()

    with pytest.raises(rehearsal.RehearsalEvidenceError):
        rehearsal.verify_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        )


@pytest.mark.parametrize("role", ("claim", "terminal"))
def test_copied_marker_or_completion_cannot_forge_inode_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    package = _package(tmp_path, monkeypatch)
    if role == "claim":
        marker = package["marker"]
        payload = marker.read_bytes()
        marker.unlink()
        marker.write_bytes(payload)
    else:
        completion = package["completion"]
        payload = completion.read_bytes()
        completion.unlink()
        completion.write_bytes(payload)

    with pytest.raises(rehearsal.RehearsalEvidenceError):
        rehearsal.verify_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        )


def test_replayed_or_changed_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path, monkeypatch)
    changed = _binding()
    runtime = changed["runtime"]
    assert isinstance(runtime, dict)
    runtime["device"] = "cuda"
    rehearsal.FORMAL_BINDING_PATH.write_bytes(_canonical(changed))

    with pytest.raises(rehearsal.RehearsalEvidenceError):
        rehearsal.verify_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "stderr",
        "outer-receipt",
        "semantic-stdout",
        "bool-returncode",
        "extra-file",
        "linked-sidecar",
        "extra-terminal-field",
    ),
)
def test_accepted_rehearsal_rejects_stream_type_namespace_and_link_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    package = _package(tmp_path, monkeypatch)
    attempt = rehearsal.REHEARSAL_ATTEMPT_PATH
    if mutation == "stderr":
        (attempt / rehearsal.STDERR_NAME).write_bytes(b"unexpected stderr")
        _rewrite_terminal(package)
    elif mutation == "outer-receipt":
        (attempt / rehearsal.OUTER_RECEIPT_NAME).write_bytes(b"unexpected receipt")
        _rewrite_terminal(package)
    elif mutation == "semantic-stdout":
        value = json.loads((attempt / rehearsal.STDOUT_NAME).read_bytes())
        value["repeat_count"] = 2
        (attempt / rehearsal.STDOUT_NAME).write_bytes(_canonical(value))
        _rewrite_terminal(package)
    elif mutation == "bool-returncode":
        package["terminal"]["returncode"] = False
        _rewrite_terminal(package)
    elif mutation == "extra-file":
        (attempt / "unexpected.json").write_bytes(b"{}\n")
    elif mutation == "linked-sidecar":
        os.link(
            attempt / rehearsal.STDOUT_NAME,
            tmp_path / "unauthorized-stdout-link.json",
        )
    else:
        package["terminal"]["unbound"] = True
        _rewrite_terminal(package)

    with pytest.raises(rehearsal.RehearsalEvidenceError):
        rehearsal.verify_accepted_binding_rehearsal(
            rehearsal.FORMAL_BINDING_PATH
        )


def test_rehearsal_rejects_noncanonical_binding_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package(tmp_path, monkeypatch)
    sibling = rehearsal.FORMAL_BINDING_PATH.with_name("replayed-binding.json")
    sibling.write_bytes(rehearsal.FORMAL_BINDING_PATH.read_bytes())

    with pytest.raises(
        rehearsal.RehearsalEvidenceError,
        match="noncanonical binding path",
    ):
        rehearsal.verify_accepted_binding_rehearsal(sibling)


def test_installed_rehearsal_module_resolves_live_git_worktree_sources(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "prospect-fixture"
    lifecycle = repository / "bench" / "world_model_lifecycle"
    lifecycle.mkdir(parents=True)
    (lifecycle / "protocol.json").write_bytes(b"{}\n")
    (lifecycle / "launch_bootstrap.py").write_bytes(b"# live launcher\n")
    (lifecycle / "producer_bootstrap.py").write_bytes(b"# live producer\n")
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
    )

    installed = tmp_path / "installed"
    installed_package = installed / "bench" / "world_model_lifecycle"
    installed_package.mkdir(parents=True)
    (installed / "bench" / "__init__.py").write_bytes(b"")
    (installed_package / "__init__.py").write_bytes(b"")
    module_path = Path(rehearsal.__file__).resolve(strict=True)
    (installed_package / "rehearsal.py").write_bytes(module_path.read_bytes())
    (installed_package / "assurance.py").write_bytes(
        module_path.with_name("assurance.py").read_bytes()
    )
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from bench.world_model_lifecycle import rehearsal
print(json.dumps({
    "repo": str(rehearsal.REPO),
    "launch": str(rehearsal.LAUNCH_BOOTSTRAP_PATH),
    "producer": str(rehearsal.PRODUCER_BOOTSTRAP_PATH),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {
        "repo": str(repository),
        "launch": str(lifecycle / "launch_bootstrap.py"),
        "producer": str(lifecycle / "producer_bootstrap.py"),
    }
