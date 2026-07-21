from __future__ import annotations

import base64
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from bench.world_model_lifecycle import (
    artifact_audit,
    producer_bootstrap,
)
from bench.world_model_lifecycle import (
    audit_runner as audit_runner_module,
)
from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle import preformal as preformal_module
from bench.world_model_lifecycle import verify as verify_module
from bench.world_model_lifecycle.assurance import TRUST_MODEL_STATEMENT
from bench.world_model_lifecycle.planning import run_pendulum_conformance


def test_bound_audit_support_paths_share_the_explicit_repository_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    support_root = repository / "bench" / "world_model_lifecycle"
    wheel_protocol = tmp_path / "site-packages" / "bench" / "world_model_lifecycle" / "protocol.json"
    observed: dict[str, object] = {}

    class ExpectedCall(Exception):
        pass

    def capture_request(
        protocol_path: Path,
        **arguments: object,
    ) -> dict[str, object]:
        observed["protocol_path"] = protocol_path
        observed.update(arguments)
        raise ExpectedCall

    monkeypatch.setattr(binding_module, "REPO", repository)
    monkeypatch.setattr(binding_module, "PROTOCOL_PATH", wheel_protocol)
    monkeypatch.setattr(
        artifact_audit,
        "build_prebinding_conformance_request",
        capture_request,
    )

    with pytest.raises(ExpectedCall):
        binding_module.build_bound_audit_execution(
            device="cuda",
            packages=[],
            roots=(),
            standard_library={"path": str(tmp_path / "stdlib")},
            producer_environment={
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "LAZY_LEGACY_OP": "False",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
                "SDL_AUDIODRIVER": "dsp",
                "TZ": "UTC",
            },
        )

    assert observed["protocol_path"] == support_root / "protocol.json"
    assert observed["support_locator_root"] == support_root
    scientific_sources = observed["scientific_source_paths"]
    assert isinstance(scientific_sources, dict)
    assert set(scientific_sources.values()) == {
        support_root / "learning.py",
        support_root / "model.py",
        support_root / "planning.py",
        support_root / "runtime_lane.py",
    }


def test_package_ownership_identity_matches_standard_library_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "site-packages"
    package = root / "fixture"
    metadata = root / "fixture-1.0.dist-info"
    package.mkdir(parents=True)
    metadata.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fixture\nVersion: 1.0\n\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "fixture/__init__.py,,\nfixture-1.0.dist-info/METADATA,,\nfixture-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(binding_module, "package_roots", lambda: (root,))

    bootstrap_identity = producer_bootstrap._package_ownership(root)
    imported_identity = binding_module.package_root_ownership()
    identity_value = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "files": [
            {"path": "fixture-1.0.dist-info/METADATA", "owners": ["fixture"]},
            {"path": "fixture-1.0.dist-info/RECORD", "owners": ["fixture"]},
            {"path": "fixture/__init__.py", "owners": ["fixture"]},
        ],
        "directories": ["fixture", "fixture-1.0.dist-info"],
    }

    assert bootstrap_identity == imported_identity
    assert (
        bootstrap_identity["identity_sha256"]
        == hashlib.sha256(binding_module.canonical_json_bytes(identity_value)).hexdigest()
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        pytest.param("PYGAME_HIDE_SUPPORT_PROMPT", None, id="missing-pygame-prompt"),
        pytest.param("PYGAME_HIDE_SUPPORT_PROMPT", "show", id="wrong-pygame-prompt"),
        pytest.param("SDL_AUDIODRIVER", None, id="missing-sdl-audio"),
        pytest.param("SDL_AUDIODRIVER", "alsa", id="wrong-sdl-audio"),
    ],
)
def test_formal_environment_requires_process_start_gymnasium_defaults(
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
    monkeypatch.setattr(binding_module.os, "environ", environment)

    with pytest.raises(RuntimeError, match="exact safe WM-001 environment"):
        binding_module.require_formal_process_environment()


def test_formal_binding_schema_requires_exact_gymnasium_defaults() -> None:
    schema = json.loads(verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"))
    environment = schema["properties"]["runtime"]["properties"]["process_environment"]

    assert {"PYGAME_HIDE_SUPPORT_PROMPT", "SDL_AUDIODRIVER"} <= set(environment["required"])
    assert environment["properties"]["PYGAME_HIDE_SUPPORT_PROMPT"] == {"const": "hide"}
    assert environment["properties"]["SDL_AUDIODRIVER"] == {"const": "dsp"}


def _schema_example(
    schema: dict[str, Any],
    root: dict[str, Any],
    *,
    path: tuple[str, ...] = (),
    item_index: int = 0,
) -> Any:
    """Construct one deterministic value satisfying the binding schema shape."""

    reference = schema.get("$ref")
    if isinstance(reference, str):
        assert reference.startswith("#/$defs/")
        definition = root["$defs"][reference.removeprefix("#/$defs/")]
        return _schema_example(
            definition,
            root,
            path=path,
            item_index=item_index,
        )
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "enum" in schema:
        return copy.deepcopy(schema["enum"][0])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "null" in schema_type:
            return None
        schema_type = schema_type[0]
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        return {
            name: _schema_example(
                properties[name],
                root,
                path=(*path, name),
                item_index=item_index,
            )
            for name in schema.get("required", [])
        }
    if schema_type == "array":
        count = int(schema.get("minItems", 0))
        return [
            _schema_example(
                schema["items"],
                root,
                path=(*path, "item"),
                item_index=index,
            )
            for index in range(count)
        ]
    if schema_type == "integer":
        return int(schema.get("minimum", 0))
    if schema_type == "number":
        return float(schema.get("minimum", 0.0))
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    assert schema_type == "string"
    pattern = str(schema.get("pattern", ""))
    if pattern == "^[0-9a-f]{40}$":
        return "a" * 40
    if pattern == "^[0-9a-f]{64}$":
        return "a" * 64
    if "development-qualification-[0-9a-f]{16}" in pattern:
        archive = "development-qualification-" + "a" * 16 + ".tar"
        if pattern.startswith("^bench/"):
            return "bench/world_model_lifecycle/results/development/" + archive
        return archive
    if pattern.startswith("^/"):
        return f"/fixture/{item_index}"
    if schema.get("format") == "date-time":
        return "2026-07-21T00:00:00Z"
    if path and path[-1] == "path":
        return f"fixture-{item_index:02d}.log"
    return "fixture"


def _binding_schema_candidate(
    log_rows: list[dict[str, object]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    candidate = _schema_example(schema, schema)
    assert isinstance(candidate, dict)
    candidate["source"]["implementation_files"] = [
        {
            "path": "bench/world_model_lifecycle/binding.py",
            "bytes": 1,
            "sha256": hashlib.sha256(b"source").hexdigest(),
        }
    ]
    candidate["source"]["test_log_files"] = copy.deepcopy(log_rows)
    return schema, candidate


def _write_realistic_preformal_logs(
    directory: Path,
    *,
    command_count: int = 10,
) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    directory.mkdir(parents=True)
    commands: list[dict[str, object]] = []
    for ordinal in range(1, command_count + 1):
        references: dict[str, dict[str, object]] = {}
        for stream, payload in (
            ("stdout", f"command-{ordinal:02d}: passed\n".encode()),
            ("stderr", b""),
        ):
            filename = f"command-{ordinal:02d}.{stream}.log"
            (directory / filename).write_bytes(payload)
            references[stream] = {
                "file": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        commands.append(
            {
                "ordinal": ordinal,
                "stdout": references["stdout"],
                "stderr": references["stderr"],
            }
        )
    report: dict[str, object] = {"commands": commands}
    report_path = directory / "preformal-report.json"
    report_path.write_bytes(binding_module.canonical_json_bytes(report) + b"\n")
    rows = binding_module.preformal_log_rows(report_path, report)
    return report_path, report, rows


def test_record_hash_identity_uses_file_hash_fields_not_repr() -> None:
    value = importlib.metadata.FileHash("sha256=YWJj")

    assert str(value) == "<FileHash mode: sha256 value: YWJj>"
    assert binding_module._record_hash_identity(value) == ("sha256", "YWJj")
    assert producer_bootstrap._record_hash_identity(value) == ("sha256", "YWJj")
    assert artifact_audit._live_record_hash_identity(value) == ("sha256", "YWJj")


def test_record_hash_decoder_requires_exact_sha256() -> None:
    digest = hashlib.sha256(b"shared").digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    assert binding_module._record_sha256_hex(("sha256", encoded)) == digest.hex()
    assert producer_bootstrap._record_sha256_hex(("sha256", encoded)) == digest.hex()
    assert artifact_audit._live_record_sha256_hex(("sha256", encoded)) == digest.hex()
    for decode, error in (
        (binding_module._record_sha256_hex, RuntimeError),
        (producer_bootstrap._record_sha256_hex, producer_bootstrap.BootstrapError),
        (artifact_audit._live_record_sha256_hex, artifact_audit.ArtifactAuditError),
    ):
        with pytest.raises(error):
            decode(("md5", encoded))
        with pytest.raises(error):
            decode(("sha256", "not+a+valid+digest"))


def test_protocol_1140_seed_domain_and_master_seeds_are_exact() -> None:
    assert verify_module.DEVELOPMENT_SEEDS == (630481329, 2204125221)
    assert verify_module.FORMAL_SEEDS == (
        900802928,
        2035185068,
        3817247901,
        14769188,
        2670334085,
        2866408483,
        671156171,
        333753598,
    )
    assert [
        verify_module.derive_seed(
            "predictive_validation_irrelevant_episode",
            630481329,
            index,
        )
        for index in range(8)
    ] == [
        696651623,
        2591703586,
        2707355286,
        631431238,
        599776448,
        2787515156,
        2907597599,
        663923271,
    ]
    assert (
        verify_module.derive_seed(
            "predictive_validation_irrelevant_action",
            2204125221,
            0,
        )
        == 2766881974
    )
    assert (
        tuple(verify_module.derive_master_seed("development", index) for index in range(2))
        == verify_module.DEVELOPMENT_SEEDS
    )
    assert tuple(verify_module.derive_master_seed("formal", index) for index in range(8)) == verify_module.FORMAL_SEEDS

    protocol = json.loads(verify_module.PROTOCOL_PATH.read_text(encoding="utf-8"))
    collision_audit = protocol["seed_schedule"]["master_seed_derivation"]["collision_audit"]
    assert collision_audit["current_master_seed_count"] == 10
    assert collision_audit["current_derived_stream_count"] == 1360
    assert collision_audit["unique_current_derived_stream_count"] == 1360
    assert collision_audit["prior_master_seed_count"] == 130
    assert collision_audit["unique_prior_derived_stream_count"] == 17680


def test_protocol_1140_states_the_negative_assurance_boundary() -> None:
    protocol = json.loads(verify_module.PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["trust_model"] == {
        "id": "prospect.wm001.trust-model.v1",
        "tamper_resistant": False,
        "external_attestation": False,
        "exclusive_path_use_required": True,
        "statement": TRUST_MODEL_STATEMENT,
    }


@pytest.mark.parametrize(
    "receipt_prefix",
    (
        "prebinding_execution_receipt",
        "restart_runtime_execution_receipt",
    ),
)
def test_verify_binding_rejects_rebound_nonempty_conformance_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_prefix: str,
) -> None:
    def canonical(value: object) -> bytes:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    bootstrap_payload = audit_runner_module.bootstrap_source_bytes()
    bootstrap_sha256 = hashlib.sha256(
        bootstrap_payload
    ).hexdigest()
    conformance_payload = canonical(
        {
            "schema": "prospect.wm001.prebinding-conformance.v2",
            "request_sha256": "1" * 64,
            "passed": True,
        }
    )
    path_runtime_value = {
        "source": {"mode": "path"},
        "support_files": [],
    }
    descriptor_runtime_value = {
        "source": {"mode": "descriptor"},
        "support_files": [],
    }
    path_runtime_payload = canonical(path_runtime_value)
    descriptor_runtime_payload = canonical(
        descriptor_runtime_value
    )
    path_invocation_payload = canonical(
        {
            "runtime_manifest_sha256": hashlib.sha256(
                path_runtime_payload
            ).hexdigest(),
        }
    )
    descriptor_invocation_payload = canonical(
        {
            "runtime_manifest_sha256": hashlib.sha256(
                descriptor_runtime_payload
            ).hexdigest(),
        }
    )
    outcome_runtime_value = {
        "source": {"mode": "descriptor"},
        "support_files": [],
    }
    outcome_runtime_payload = canonical(outcome_runtime_value)
    restart_report_payload = canonical(
        {
            "schema": (
                "prospect.wm001.restart-runtime-conformance.v1"
            ),
            "protocol_version": "1.14.0",
            "passed": True,
        }
    )
    repeat_count = 3
    modes = [
        *("path" for _ in range(repeat_count)),
        *("descriptor" for _ in range(repeat_count)),
    ]
    empty_stderr = {
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    auditor_sha256 = "2" * 64

    def prebinding_row(
        ordinal: int,
        mode: str,
    ) -> dict[str, object]:
        runtime_payload = (
            path_runtime_payload
            if mode == "path"
            else descriptor_runtime_payload
        )
        invocation_payload = (
            path_invocation_payload
            if mode == "path"
            else descriptor_invocation_payload
        )
        return {
            "ordinal": ordinal,
            "source_mode": mode,
            "returncode": 0,
            "stdout": {
                "bytes": len(conformance_payload),
                "sha256": hashlib.sha256(
                    conformance_payload
                ).hexdigest(),
            },
            "stderr": dict(empty_stderr),
            "runtime_manifest": {
                "bytes": len(runtime_payload),
                "sha256": hashlib.sha256(
                    runtime_payload
                ).hexdigest(),
            },
            "invocation_manifest": {
                "bytes": len(invocation_payload),
                "sha256": hashlib.sha256(
                    invocation_payload
                ).hexdigest(),
            },
            "bootstrap_sha256": bootstrap_sha256,
            "auditor_source_sha256": auditor_sha256,
            "support_files": [],
            "auditor_report_passed": True,
        }

    prebinding_rows = [
        prebinding_row(ordinal, mode)
        for ordinal, mode in enumerate(modes, start=1)
    ]
    prebinding_receipt = {
        "schema": "prospect.wm001.audit-conformance-receipt.v1",
        "repeat_count": repeat_count,
        "execution_count": len(prebinding_rows),
        "executions": prebinding_rows,
        "report_sha256": hashlib.sha256(
            conformance_payload
        ).hexdigest(),
        "path_descriptor_byte_identical": True,
        "execution_conformance_passed": True,
    }
    path_restart_runtime_value = copy.deepcopy(
        outcome_runtime_value
    )
    path_restart_runtime_value["source"]["mode"] = "path"
    path_restart_runtime_payload = canonical(
        path_restart_runtime_value
    )
    restart_runtime_payloads = {
        "path": path_restart_runtime_payload,
        "descriptor": outcome_runtime_payload,
    }
    restart_arguments = [
        "--restart-runtime-conformance",
        "--producer-bootstrap",
        "@captured/producer_bootstrap.py",
        "--expected-producer-bootstrap-sha256",
        hashlib.sha256(
            (
                verify_module.HERE
                / "producer_bootstrap.py"
            ).read_bytes()
        ).hexdigest(),
    ]
    restart_invocation_payloads = {
        mode: canonical(
            {
                "schema": (
                    "prospect.wm001.audit-invocation-manifest.v1"
                ),
                "runtime_manifest_sha256": hashlib.sha256(
                    runtime_payload
                ).hexdigest(),
                "working_directory": str(verify_module.REPO),
                "auditor_argv": restart_arguments,
            }
        )
        for mode, runtime_payload in (
            restart_runtime_payloads.items()
        )
    }
    restart_rows = [
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
            "stderr": dict(empty_stderr),
            "runtime_manifest": {
                "bytes": len(restart_runtime_payloads[mode]),
                "sha256": hashlib.sha256(
                    restart_runtime_payloads[mode]
                ).hexdigest(),
            },
            "invocation_manifest": {
                "bytes": len(
                    restart_invocation_payloads[mode]
                ),
                "sha256": hashlib.sha256(
                    restart_invocation_payloads[mode]
                ).hexdigest(),
            },
            "bootstrap_sha256": bootstrap_sha256,
            "auditor_source_sha256": auditor_sha256,
            "support_files": [],
            "auditor_report_passed": True,
        }
        for ordinal, mode in enumerate(modes, start=1)
    ]
    restart_receipt = {
        "schema": "prospect.wm001.audit-conformance-receipt.v1",
        "repeat_count": repeat_count,
        "execution_count": len(restart_rows),
        "executions": restart_rows,
        "report_sha256": hashlib.sha256(
            restart_report_payload
        ).hexdigest(),
        "path_descriptor_byte_identical": True,
        "execution_conformance_passed": True,
    }
    request_payload = canonical(
        {
            "schema": (
                "prospect.wm001.prebinding-conformance-request.v2"
            )
        }
    )
    evidence = {
        "bootstrap_source": bootstrap_payload,
        "prebinding_request": request_payload,
        "prebinding_path_runtime_manifest": path_runtime_payload,
        "prebinding_descriptor_runtime_manifest": (
            descriptor_runtime_payload
        ),
        "prebinding_path_invocation_manifest": (
            path_invocation_payload
        ),
        "prebinding_descriptor_invocation_manifest": (
            descriptor_invocation_payload
        ),
        "prebinding_conformance_report": conformance_payload,
        "prebinding_execution_receipt": canonical(
            prebinding_receipt
        ),
        "outcome_runtime_manifest": outcome_runtime_payload,
        "restart_runtime_conformance_report": (
            restart_report_payload
        ),
        "restart_runtime_execution_receipt": canonical(
            restart_receipt
        ),
    }
    audit_execution: dict[str, object] = {
        "bootstrap_source_sha256": bootstrap_sha256,
        "auditor_source_sha256": auditor_sha256,
        "repeat_count": repeat_count,
        "restart_runtime_repeat_count": repeat_count,
    }
    for prefix, payload in evidence.items():
        sibling = tmp_path / f"{prefix}.json"
        sibling.write_bytes(payload)
        audit_execution[f"{prefix}_file"] = sibling.name
        audit_execution[f"{prefix}_bytes"] = len(payload)
        audit_execution[f"{prefix}_sha256"] = (
            hashlib.sha256(payload).hexdigest()
        )
    report_path = tmp_path / str(
        audit_execution["prebinding_conformance_report_file"]
    )
    assert report_path.read_bytes() == conformance_payload
    test_report = tmp_path / "preformal-report.json"
    test_report.write_bytes(canonical({"passed": True}))
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_bytes(canonical({"fixture": True}))
    binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v10",
        "experiment_id": "WM-001",
        "assurance": dict(binding_module.ASSURANCE),
        "protocol": {},
        "source": {
            "implementation_files": [],
            "execution_source_sha256": {},
            "test_report_file": test_report.name,
            "test_log_files": [],
        },
        "dependencies": {
            "lockfile": "requirements-wm001.lock",
            "python_executable": sys.executable,
            "standard_library": {},
            "package_roots": [],
            "package_ownership": {},
            "packages": [],
        },
        "runtime": {"process_environment": {}},
        "audit_execution": audit_execution,
    }

    monkeypatch.setattr(verify_module, "verify_protocol", lambda: {})
    monkeypatch.setattr(
        verify_module,
        "_load_json",
        lambda path: binding if path == binding_path else {},
    )
    monkeypatch.setattr(
        verify_module,
        "_validate_json_schema",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        verify_module,
        "_parse_timestamp",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        verify_module,
        "_verify_implementation_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        binding_module,
        "verify_bound_machine_test_report",
        lambda _path, _binding: {},
    )
    monkeypatch.setattr(
        binding_module,
        "preformal_log_rows",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        preformal_module,
        "_runtime_bootstrap_conformance_from_report",
        lambda *_args, **_kwargs: {},
    )

    target_message = (
        "bound prebinding receipt does not preserve every execution identity"
        if receipt_prefix == "prebinding_execution_receipt"
        else (
            "bound restart-runtime receipt does not preserve every exact "
            "path and descriptor execution"
        )
    )

    class ReachedReceiptGate(Exception):
        pass

    def require_target(condition: bool, message: str) -> None:
        if message != target_message:
            return
        if not condition:
            raise verify_module.Violation(message)
        raise ReachedReceiptGate

    monkeypatch.setattr(
        verify_module,
        "_require",
        require_target,
    )
    with pytest.raises(ReachedReceiptGate):
        verify_module.verify_binding(binding_path)

    changed = (
        prebinding_receipt
        if receipt_prefix == "prebinding_execution_receipt"
        else restart_receipt
    )
    diagnostic = b"internally consistent torch warning\n"
    stderr_identity = {
        "bytes": len(diagnostic),
        "sha256": hashlib.sha256(diagnostic).hexdigest(),
    }
    for row in changed["executions"]:
        row["stderr"] = dict(stderr_identity)
    changed_payload = canonical(changed)
    changed_path = tmp_path / str(
        audit_execution[f"{receipt_prefix}_file"]
    )
    changed_path.write_bytes(changed_payload)
    audit_execution[f"{receipt_prefix}_bytes"] = len(
        changed_payload
    )
    audit_execution[f"{receipt_prefix}_sha256"] = (
        hashlib.sha256(changed_payload).hexdigest()
    )

    with pytest.raises(
        verify_module.Violation,
        match=target_message,
    ):
        verify_module.verify_binding(binding_path)


def test_implementation_manifest_binds_reviewed_v1140_documents() -> None:
    paths = {
        str(row["path"])
        for row in binding_module.implementation_files()
    }

    assert {
        "docs/wm001-v1140-confirmation-plan.md",
        "docs/wm001-v1140-operator-runbook.md",
        "docs/wm001-v1140-prospective-harness-review.json",
    } <= paths


def test_protocol_1140_irrelevant_control_contract_is_bound() -> None:
    assert (
        "collect_irrelevant",
        verify_module.TASK_IRRELEVANT,
        "collection_random",
        "cold",
    ) in verify_module.EPISODE_CONTRACTS
    assert (
        "predictive_validation_irrelevant",
        verify_module.TASK_IRRELEVANT,
        "validation_random",
        "irrelevant",
    ) in verify_module.EPISODE_CONTRACTS
    assert (
        "predictive_validation_irrelevant",
        verify_module.TASK_IRRELEVANT,
        "cold",
        "cold",
    ) in verify_module.PREDICTIVE_CONTRACTS
    assert (
        "predictive_validation_irrelevant",
        verify_module.TASK_IRRELEVANT,
        "irrelevant",
        "irrelevant",
    ) in verify_module.PREDICTIVE_CONTRACTS
    assert (
        "predictive_validation_a",
        verify_module.TASK_A,
        "irrelevant",
        "irrelevant",
    ) in verify_module.PREDICTIVE_CONTRACTS
    assert (
        "behavior_evaluation_a",
        verify_module.TASK_A,
        "irrelevant",
        "irrelevant",
    ) in verify_module.EPISODE_CONTRACTS


def test_result_runtime_must_equal_formal_binding_runtime() -> None:
    runtime = {
        "platform": "bound-platform",
        "device": "cuda",
        "deterministic_algorithms": True,
    }
    verify_module._verify_result_runtime_binding(
        {
            "platform": "bound-platform",
            "device": "cuda",
            "deterministic_algorithms": True,
        },
        runtime,
    )
    with pytest.raises(
        verify_module.Violation,
        match="result runtime differs from binding",
    ):
        verify_module._verify_result_runtime_binding(
            {
                "platform": "bound-platform",
                "device": "cpu",
                "deterministic_algorithms": True,
            },
            runtime,
        )


def test_formal_binding_schema_binds_protocol_1140_and_fresh_seeds() -> None:
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"),
    )

    assert schema["$id"].endswith("wm-001-formal-binding-v10.json")
    assert schema["properties"]["schema"]["const"] == "prospect.world-model-lifecycle.formal-binding.v10"
    assert "assurance" in schema["required"]
    assert schema["properties"]["assurance"]["properties"] == {
        "trust_model_id": {
            "const": "prospect.wm001.trust-model.v1",
        },
        "tamper_resistant": {"const": False},
        "external_attestation": {"const": False},
        "exclusive_path_use_required": {"const": True},
    }
    assert schema["properties"]["protocol"]["properties"]["version"]["const"] == "1.14.0"
    assert (
        tuple(
            schema["properties"]["formal_replicate_master_seeds"]["const"],
        )
        == verify_module.FORMAL_SEEDS
    )
    assert "coverage_arithmetic" in schema["required"]
    coverage = schema["properties"]["coverage_arithmetic"]
    assert coverage["properties"]["semantics_id"]["const"] == ("wm001-mixture-pit-binary64-count-v1")
    assert {
        "producer_source_sha256",
        "auditor_source_sha256",
        "formal_test_report_sha256",
        "conformance_report_sha256",
    } <= set(coverage["required"])
    audit_execution = schema["properties"]["audit_execution"]
    restart_fields = {
        "restart_runtime_conformance_report_file",
        "restart_runtime_conformance_report_bytes",
        "restart_runtime_conformance_report_sha256",
        "restart_runtime_execution_receipt_file",
        "restart_runtime_execution_receipt_bytes",
        "restart_runtime_execution_receipt_sha256",
        "restart_runtime_support_files",
        "restart_runtime_repeat_count",
        "restart_runtime_path_descriptor_equal",
    }
    assert restart_fields <= set(audit_execution["required"])
    assert restart_fields <= set(audit_execution["properties"])
    assert audit_execution["properties"]["restart_runtime_support_files"]["const"] == [
        "producer_bootstrap.py",
        "protocol.json",
        "schemas/raw-result.schema.json",
    ]


def test_formal_binding_schema_separates_source_and_stream_file_digests() -> None:
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    source = schema["properties"]["source"]["properties"]

    assert source["implementation_files"]["items"] == {
        "$ref": "#/$defs/fileDigest",
    }
    assert schema["$defs"]["fileDigest"]["properties"]["bytes"] == {
        "type": "integer",
        "minimum": 1,
    }
    assert source["test_log_files"]["items"] == {
        "$ref": "#/$defs/streamFileDigest",
    }
    assert schema["$defs"]["streamFileDigest"]["properties"]["bytes"] == {
        "type": "integer",
        "minimum": 0,
    }


@pytest.mark.parametrize(
    ("definition", "minimum", "message"),
    [
        (
            "fileDigest",
            0,
            "formal binding fileDigest byte contract is not exact",
        ),
        (
            "streamFileDigest",
            -1,
            "formal binding streamFileDigest byte contract is not exact",
        ),
    ],
)
def test_preclaim_requires_exact_digest_byte_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    definition: str,
    minimum: int,
    message: str,
) -> None:
    _, _, rows = _write_realistic_preformal_logs(tmp_path / "preformal")
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    schema["$defs"][definition]["properties"]["bytes"]["minimum"] = minimum
    schema_path = tmp_path / "formal-binding.schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setattr(binding_module, "BINDING_SCHEMA_PATH", schema_path)

    with pytest.raises(RuntimeError, match=message):
        binding_module.verify_preclaim_log_schema_compatibility(rows)


def test_root_binding_schema_accepts_realistic_zero_byte_stderr_logs(
    tmp_path: Path,
) -> None:
    _, _, rows = _write_realistic_preformal_logs(tmp_path / "preformal")
    schema, candidate = _binding_schema_candidate(rows)

    verify_module._validate_json_schema(
        candidate,
        schema,
        label="synthetic v1.14 formal binding",
    )
    assert len(rows) == 20
    assert all(row["bytes"] > 0 for row in rows[0::2])
    assert all(
        row
        == {
            "path": row["path"],
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        for row in rows[1::2]
    )


def test_root_binding_schema_rejects_negative_log_bytes_and_zero_source_bytes(
    tmp_path: Path,
) -> None:
    _, _, rows = _write_realistic_preformal_logs(tmp_path / "preformal")
    schema, candidate = _binding_schema_candidate(rows)

    negative_log = copy.deepcopy(candidate)
    negative_log["source"]["test_log_files"][1]["bytes"] = -1
    with pytest.raises(
        verify_module.Violation,
        match="violates JSON Schema",
    ):
        verify_module._validate_json_schema(
            negative_log,
            schema,
            label="synthetic v1.14 formal binding",
        )

    empty_source = copy.deepcopy(candidate)
    empty_source["source"]["implementation_files"][0]["bytes"] = 0
    with pytest.raises(
        verify_module.Violation,
        match="violates JSON Schema",
    ):
        verify_module._validate_json_schema(
            empty_source,
            schema,
            label="synthetic v1.14 formal binding",
        )


def test_preformal_log_rows_reject_wrong_empty_stream_digest(
    tmp_path: Path,
) -> None:
    report_path, report, rows = _write_realistic_preformal_logs(
        tmp_path / "preformal",
    )
    binding_module.verify_preclaim_log_schema_compatibility(rows)
    commands = report["commands"]
    assert isinstance(commands, list)
    stderr = commands[0]["stderr"]
    assert isinstance(stderr, dict)
    stderr["sha256"] = "f" * 64

    with pytest.raises(
        RuntimeError,
        match="preformal report log bytes differ from their reference",
    ):
        binding_module.preformal_log_rows(report_path, report)


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (False, 0),
        (True, 1),
        (1.0, 1),
        ({"bytes": 3.0}, {"bytes": 3}),
        ([{"rejected": 1}], [{"rejected": True}]),
    ],
)
def test_restart_json_comparison_rejects_python_numeric_aliases(
    observed: object,
    expected: object,
) -> None:
    assert artifact_audit._strict_json_equal(expected, expected)
    assert verify_module._strict_json_equal(expected, expected)
    assert not artifact_audit._strict_json_equal(observed, expected)
    assert not verify_module._strict_json_equal(observed, expected)


def test_raw_result_schema_binds_v1140_heldout_split_and_formal_counts() -> None:
    schema = json.loads(
        verify_module.RESULT_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    replicate_limits = schema["allOf"][0]["then"]["properties"]["replicates"]["items"]["allOf"][1]["properties"]
    predictive_schema = schema["$defs"]["predictiveMetric"]
    predictive_properties = predictive_schema["properties"]
    gate_comparators = schema["$defs"]["gateCheck"]["properties"]["comparator"]["enum"]

    assert schema["$id"].endswith("wm-001-raw-result-v9.json")
    assert schema["properties"]["schema"]["const"] == "prospect.world-model-lifecycle.raw-result.v9"
    assert schema["properties"]["protocol_version"]["const"] == "1.14.0"
    assert "predictive_validation_irrelevant" in schema["$defs"]["episode"]["properties"]["split"]["enum"]
    assert "predictive_validation_irrelevant" in schema["$defs"]["transition"]["properties"]["split"]["enum"]
    assert "predictive_validation_irrelevant" in predictive_properties["split"]["enum"]
    assert {
        "coverage_semantics",
        "interval_90_covered_target_count",
        "coverage_target_count",
    } <= set(predictive_schema["required"])
    assert predictive_properties["coverage_semantics"] == {
        "const": "wm001-mixture-pit-binary64-count-v1",
    }
    assert predictive_properties["interval_90_covered_target_count"] == {
        "type": "integer",
        "minimum": 0,
    }
    assert predictive_properties["coverage_target_count"] == {
        "type": "integer",
        "minimum": 1,
    }
    assert predictive_properties["interval_90_coverage"] == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
    }
    assert gate_comparators == [
        "eq",
        "gt",
        "ge",
        "lt",
        "le",
        "10*C >= 7*T",
        "100*C <= 99*T",
    ]
    assert replicate_limits["derived_seeds"] == {"minItems": 21, "maxItems": 21}
    assert replicate_limits["episodes"] == {"minItems": 496, "maxItems": 496}
    assert replicate_limits["transitions"] == {"minItems": 99200, "maxItems": 99200}
    assert replicate_limits["predictive_metrics"] == {"minItems": 12, "maxItems": 12}
    assert replicate_limits["policy_runs"] == {"minItems": 20, "maxItems": 20}


def test_formal_matrix_verifier_requires_every_exact_v1140_row() -> None:
    episodes: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    for contract, count in verify_module.FORMAL_EPISODE_CONTRACT_COUNTS.items():
        split, task_id, condition, checkpoint_id = contract
        episodes.extend(
            [
                {
                    "split": split,
                    "task_id": task_id,
                    "condition": condition,
                    "checkpoint_id": checkpoint_id,
                }
            ]
            * count
        )
        transitions.extend([{"split": split}] * (count * 200))
    predictive = [
        {
            "split": split,
            "task_id": task_id,
            "condition": condition,
            "checkpoint_id": checkpoint_id,
            "transition_count": 1_600,
            "coverage_target_count": 6_400,
        }
        for split, task_id, condition, checkpoint_id in verify_module.PREDICTIVE_CONTRACTS
    ]
    policy_runs = [
        {
            "split": split,
            "task_id": task_id,
            "condition": condition,
            "checkpoint_id": checkpoint_id,
        }
        for split, task_id, condition, checkpoint_id in verify_module.EPISODE_CONTRACTS
    ]
    updates = [
        {
            "phase": phase,
            "status": "committed",
            "optimizer_steps": 2_000,
        }
        for phase in verify_module.COMMITTED_PHASE_SPLITS
    ]
    updates.append(
        {
            "phase": "rejected_update_probe",
            "status": "rejected",
            "optimizer_steps": 0,
        }
    )
    replicate = {
        "episodes": episodes,
        "transitions": transitions,
        "predictive_metrics": predictive,
        "policy_runs": policy_runs,
        "updates": updates,
        "optimizer_batch_manifests": [{"phase": phase} for phase in verify_module.COMMITTED_PHASE_SPLITS],
    }

    verify_module._verify_formal_matrix(replicate, replicate_id="formal-fixture")
    replicate["predictive_metrics"] = predictive[:-1]
    with pytest.raises(verify_module.Violation, match="predictive matrix"):
        verify_module._verify_formal_matrix(replicate, replicate_id="formal-fixture")


def test_update_eligibility_categorically_excludes_heldout_oscillator_rows() -> None:
    local_transitions = {
        "collect-a": {"split": "collect_a"},
        "oscillator-heldout": {"split": "predictive_validation_irrelevant"},
    }
    valid_update = {
        "phase": "train_a",
        "eligible_splits": ["collect_a"],
        "eligible_transition_count": 1,
        "eligible_transition_ids": ["collect-a"],
    }
    verify_module._verify_update_eligibility(
        valid_update,
        local_transitions=local_transitions,
        replicate_id="fixture",
    )

    contaminated = {
        **valid_update,
        "eligible_transition_count": 2,
        "eligible_transition_ids": ["collect-a", "oscillator-heldout"],
    }
    with pytest.raises(verify_module.Violation, match="held-out or phase-ineligible"):
        verify_module._verify_update_eligibility(
            contaminated,
            local_transitions=local_transitions,
            replicate_id="fixture",
        )


def test_implementation_manifest_ignores_generated_attempt_source_copies() -> None:
    paths = {str(row["path"]) for row in binding_module.implementation_files()}

    assert "bench/world_model_lifecycle/binding.py" in paths
    assert "bench/world_model_lifecycle/assurance.py" in paths
    assert not any(path.startswith("bench/world_model_lifecycle/results/") for path in paths)


def _rehash_conformance(report: dict[str, object]) -> None:
    body = dict(report)
    body.pop("report_sha256", None)
    payload = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(payload).hexdigest()


def test_exact_implementation_manifest_accepts_only_complete_ordered_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [
        {"path": "a.py", "bytes": 1, "sha256": "1" * 64},
        {"path": "b.py", "bytes": 2, "sha256": "2" * 64},
    ]
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: copy.deepcopy(expected),
    )

    verify_module._verify_implementation_manifest(copy.deepcopy(expected))

    adversarial = {
        "omitted": expected[1:],
        "extra": [
            *expected,
            {"path": "c.py", "bytes": 3, "sha256": "3" * 64},
        ],
        "reordered": list(reversed(expected)),
        "digest": [
            {**expected[0], "sha256": "f" * 64},
            expected[1],
        ],
    }
    for manifest in adversarial.values():
        with pytest.raises(
            verify_module.Violation,
            match="exact complete ordered",
        ):
            verify_module._verify_implementation_manifest(copy.deepcopy(manifest))


def test_fixed_formal_conformance_report_is_independently_accepted() -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )

    verify_module._verify_pendulum_conformance_report(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cases", 2),
        ("samples_per_task", 1),
        ("seed", 7),
        ("spec_horizon", 199),
        ("terminated_or_truncated_cases", 1),
        ("reward_atol", 1.0),
    ],
)
def test_fixed_formal_conformance_rejects_rehashed_contract_changes(
    field: str,
    value: object,
) -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )
    report[field] = value
    _rehash_conformance(report)

    with pytest.raises(verify_module.Violation):
        verify_module._verify_pendulum_conformance_report(report)


def test_fixed_formal_conformance_rejects_invalid_self_hash() -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )
    report["report_sha256"] = "0" * 64

    with pytest.raises(verify_module.Violation, match="self-hash"):
        verify_module._verify_pendulum_conformance_report(report)


def test_coverage_conformance_reproduces_endpoints_and_v130_regression() -> None:
    report = binding_module.run_coverage_conformance()

    verify_module._verify_coverage_conformance_report(report)
    assert report["passed"] is True
    assert report["cases"][-1]["case_id"] == "v130-disclosed-boundary-coordinate"
    assert report["cases"][-1]["observed_pit_hex"] == "0x1.999998b3745adp-5"
    assert report["cases"][-1]["observed_covered"] is False


def test_coverage_conformance_rejects_rehashed_boundary_change() -> None:
    report = binding_module.run_coverage_conformance()
    report["cases"][-1]["target_little_endian_f32_hex"] = "00000000"
    corpus = {
        "semantics_id": report["semantics_id"],
        "cases": report["cases"],
    }
    report["corpus_sha256"] = hashlib.sha256(binding_module.canonical_json_bytes(corpus)).hexdigest()
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(binding_module.canonical_json_bytes(body)).hexdigest()

    with pytest.raises(verify_module.Violation, match="boundary regression inputs"):
        verify_module._verify_coverage_conformance_report(report)


def _install_generated_binding_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    report_path, report, log_rows = _write_realistic_preformal_logs(
        tmp_path / "inputs" / "preformal",
    )
    schema, candidate = _binding_schema_candidate(log_rows)
    closure_path = tmp_path / "inputs" / "development-closure.json"
    closure_path.write_bytes(
        binding_module.canonical_json_bytes({"fixture": "closure"}) + b"\n",
    )
    lockfile = tmp_path / "requirements-wm001.lock"
    lockfile.write_text("fixture==1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    output_path = tmp_path / "binding-package" / "formal-binding.json"

    dependencies = candidate["dependencies"]
    runtime = candidate["runtime"]
    audit_execution = candidate["audit_execution"]
    development_identity = candidate["development_qualification"]
    source = candidate["source"]
    inventory = {
        "packages": copy.deepcopy(dependencies["packages"]),
        "package_roots": copy.deepcopy(dependencies["package_roots"]),
        "standard_library": copy.deepcopy(dependencies["standard_library"]),
        "package_ownership": copy.deepcopy(dependencies["package_ownership"]),
    }
    rehearsal = {
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(
            binding_module.canonical_json_bytes(inventory),
        ).hexdigest(),
        "conformance_sha256": hashlib.sha256(
            binding_module.canonical_json_bytes(audit_execution),
        ).hexdigest(),
        "restart_runtime_conformance_report_sha256": audit_execution[
            "restart_runtime_conformance_report_sha256"
        ],
        "restart_runtime_execution_receipt_sha256": audit_execution[
            "restart_runtime_execution_receipt_sha256"
        ],
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

    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "LOCKFILE", lockfile)
    monkeypatch.setattr(verify_module, "verify_protocol", lambda: {})
    monkeypatch.setattr(binding_module, "source_is_clean", lambda: True)
    monkeypatch.setattr(
        binding_module,
        "require_formal_python_flags",
        lambda: copy.deepcopy(runtime["python_flags"]),
    )
    monkeypatch.setattr(
        binding_module,
        "require_formal_process_environment",
        lambda: copy.deepcopy(runtime["process_environment"]),
    )
    monkeypatch.setattr(
        binding_module,
        "run_pendulum_conformance",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        binding_module,
        "_created_conformance_satisfies_formal_contract",
        lambda _report: True,
    )
    monkeypatch.setattr(
        binding_module,
        "run_independent_phase_oscillator_conformance",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        binding_module,
        "run_coverage_conformance",
        lambda: {"passed": True},
    )
    monkeypatch.setattr(
        binding_module,
        "verify_canonical_machine_test_report",
        lambda path: copy.deepcopy(report),
    )
    monkeypatch.setattr(
        binding_module,
        "verify_development_closure",
        lambda _path: {"fixture": "closure"},
    )
    monkeypatch.setattr(
        binding_module,
        "_formal_development_identity",
        lambda *_args, **_kwargs: copy.deepcopy(development_identity),
    )
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: copy.deepcopy(source["implementation_files"]),
    )
    monkeypatch.setattr(binding_module, "verify_installed_source_snapshot", lambda: None)
    monkeypatch.setattr(binding_module, "package_roots", lambda: (tmp_path,))
    monkeypatch.setattr(
        binding_module,
        "package_root_inventory",
        lambda _root: copy.deepcopy(dependencies["package_roots"][0]),
    )
    monkeypatch.setattr(
        binding_module,
        "standard_library_inventory",
        lambda: copy.deepcopy(dependencies["standard_library"]),
    )
    monkeypatch.setattr(
        binding_module,
        "package_root_ownership",
        lambda: copy.deepcopy(dependencies["package_ownership"]),
    )
    monkeypatch.setattr(
        binding_module,
        "installed_package_rows",
        lambda: copy.deepcopy(dependencies["packages"]),
    )
    monkeypatch.setattr(binding_module, "verify_lockfile_rows", lambda _rows: None)
    monkeypatch.setattr(
        binding_module,
        "build_bound_audit_execution",
        lambda **_kwargs: (copy.deepcopy(audit_execution), {}),
    )
    monkeypatch.setattr(
        preformal_module,
        "_runtime_bootstrap_conformance_from_report",
        lambda *_args, **_kwargs: copy.deepcopy(rehearsal),
    )
    monkeypatch.setattr(
        binding_module,
        "git_output",
        lambda *arguments: (
            "a" * 40 if arguments == ("rev-parse", "HEAD") else "b" * 40
        ),
    )
    monkeypatch.setattr(
        binding_module,
        "distribution_sha256",
        lambda _name: "c" * 64,
    )
    monkeypatch.setattr(
        binding_module,
        "checkpoint_implementation_sha256",
        lambda: "d" * 64,
    )
    monkeypatch.setattr(
        binding_module,
        "manifest_schema_sha256",
        lambda: "e" * 64,
    )
    monkeypatch.setattr(binding_module.torch, "use_deterministic_algorithms", lambda _enabled: None)
    monkeypatch.setattr(
        binding_module.torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(binding_module.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(binding_module.torch, "get_num_interop_threads", lambda: 1)

    return output_path, closure_path, schema, candidate


def test_create_formal_binding_root_schema_preflight_accepts_actual_log_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path, closure_path, schema, _ = _install_generated_binding_fakes(
        tmp_path,
        monkeypatch,
    )
    report_path = tmp_path / "inputs" / "preformal" / "preformal-report.json"

    created = binding_module.create_formal_binding(
        output_path=output_path,
        test_report_path=report_path,
        development_closure_path=closure_path,
        device="cpu",
    )

    verify_module._validate_json_schema(
        created,
        schema,
        label="generated v1.14 formal binding",
    )
    assert created["source"]["test_log_files"] == binding_module.preformal_log_rows(
        report_path,
        binding_module.verify_canonical_machine_test_report(report_path),
    )
    assert output_path.is_file()


def test_create_formal_binding_schema_failure_precedes_all_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path, closure_path, _, candidate = _install_generated_binding_fakes(
        tmp_path,
        monkeypatch,
    )
    report_path = tmp_path / "inputs" / "preformal" / "preformal-report.json"
    invalid_manifest = copy.deepcopy(candidate["source"]["implementation_files"])
    invalid_manifest[0]["bytes"] = 0
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: copy.deepcopy(invalid_manifest),
    )

    with pytest.raises(
        RuntimeError,
        match="assembled formal binding is incompatible with the root schema",
    ):
        binding_module.create_formal_binding(
            output_path=output_path,
            test_report_path=report_path,
            development_closure_path=closure_path,
            device="cpu",
        )

    assert not output_path.parent.exists()


def test_create_binding_refuses_nonformal_conformance_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_report = tmp_path / "tests.txt"
    test_report.write_text("passed\n", encoding="utf-8")
    monkeypatch.setattr(verify_module, "verify_protocol", lambda: {})
    monkeypatch.setattr(binding_module, "source_is_clean", lambda: True)
    monkeypatch.setattr(
        binding_module,
        "require_formal_python_flags",
        lambda: {"isolated": 1},
    )
    monkeypatch.setattr(
        binding_module,
        "require_formal_process_environment",
        lambda: {},
    )

    with pytest.raises(ValueError, match="exactly 1,024"):
        binding_module.create_formal_binding(
            output_path=tmp_path / "binding.json",
            test_report_path=test_report,
            conformance_cases=2,
            device="cpu",
        )


def test_create_binding_validates_environment_before_gymnasium_conformance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conformance_called = False

    def reject_environment() -> dict[str, str]:
        raise RuntimeError("process-start environment rejected")

    def unexpected_conformance(**_: object) -> dict[str, object]:
        nonlocal conformance_called
        conformance_called = True
        return {}

    monkeypatch.setattr(
        binding_module,
        "require_formal_python_flags",
        lambda: {"isolated": 1},
    )
    monkeypatch.setattr(
        binding_module,
        "require_formal_process_environment",
        reject_environment,
    )
    monkeypatch.setattr(
        binding_module,
        "run_pendulum_conformance",
        unexpected_conformance,
    )

    with pytest.raises(RuntimeError, match="process-start environment rejected"):
        binding_module.create_formal_binding(
            output_path=tmp_path / "binding.json",
            test_report_path=tmp_path / "report.json",
            development_closure_path=tmp_path / "closure.json",
            device="cpu",
        )

    assert conformance_called is False


def test_live_binding_rechecks_complete_implementation_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {
        "source": {
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "implementation_files": [{"path": "a.py", "bytes": 1, "sha256": "3" * 64}],
        }
    }
    monkeypatch.setattr(verify_module, "verify_binding", lambda path: binding)
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: [{"path": "b.py", "bytes": 1, "sha256": "4" * 64}],
    )
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_bytes(_canonical_payload({"fixture": "binding"}))

    with pytest.raises(RuntimeError, match="complete live manifest"):
        binding_module.verify_live_binding(
            binding_path,
            device="cpu",
        )


def test_live_binding_rejects_qa_runtime_package_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_digests = {
        filename: binding_module.sha256_file(
            Path(binding_module.__file__).with_name(filename)
        )
        for filename in binding_module.EXECUTION_SOURCE_FILES
    }
    binding = {
        "source": {
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "implementation_files": [],
            "execution_source_sha256": source_digests,
        },
        "dependencies": {
            "packages": [{"name": "runtime-only", "version": "1"}],
        },
    }
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_bytes(_canonical_payload({"fixture": "binding"}))
    monkeypatch.setattr(verify_module, "verify_binding", lambda _path: binding)
    monkeypatch.setattr(binding_module, "implementation_files", lambda: [])
    monkeypatch.setattr(binding_module, "source_is_clean", lambda: True)
    monkeypatch.setattr(
        binding_module,
        "git_output",
        lambda *arguments: (
            "1" * 40 if arguments == ("rev-parse", "HEAD") else "2" * 40
        ),
    )
    monkeypatch.setattr(binding_module, "require_formal_python_flags", lambda: None)
    monkeypatch.setattr(binding_module, "require_formal_process_environment", lambda: None)
    monkeypatch.setattr(binding_module, "verify_installed_source_snapshot", lambda: None)
    monkeypatch.setattr(binding_module, "installed_package_rows", lambda: [])
    monkeypatch.setattr(binding_module, "verify_lockfile_rows", lambda _rows: None)

    with pytest.raises(RuntimeError, match="installed package closure"):
        binding_module.verify_live_binding(binding_path, device="cpu")


def _canonical_payload(value: object) -> bytes:
    return bytes(binding_module.canonical_json_bytes(value)) + b"\n"


def _recorded_development_closure_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, str]]:
    role_paths = {
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "result_qualification_member": "evidence/development-result-qualification.json",
        "independent_audit_member": "evidence/independent-audit.json",
        "audit_reproduction_member": "evidence/audit-reproduction.json",
        "audit_runtime_manifest_member": "evidence/development-audit-runtime-fixture.json",
        "audit_invocation_manifest_member": "evidence/development-audit-invocation-fixture.json",
        "audit_stderr_member": "evidence/development-audit-stderr-fixture.log",
        "runtime_seal_member": "evidence/producer-runtime-seal.json",
        "producer_bootstrap_member": "evidence/producer-bootstrap.py",
        "launch_bootstrap_member": "evidence/launch-bootstrap.py",
    }
    digests = {
        member: hashlib.sha256(member.encode("utf-8")).hexdigest()
        for member in role_paths.values()
    }
    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    source: dict[str, object] = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "worktree_clean": True,
        "dependency_lock_sha256": digest("lock"),
        "producer_bootstrap_sha256": digests[role_paths["producer_bootstrap_member"]],
        "launch_bootstrap_sha256": digests[role_paths["launch_bootstrap_member"]],
        "runner_source_sha256": digest("runner"),
        "auditor_source_sha256": digest("auditor"),
    }
    execution: dict[str, object] = {
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "worktree_clean": True,
        "dependency_lock_sha256": source["dependency_lock_sha256"],
        "python_executable": "/sealed-runtime/bin/python",
        "python_executable_sha256": digest("python"),
        "python_version": "3.12.11",
        "platform": "sealed-runtime-platform",
        "machine": "sealed-runtime-machine",
        "device": "cpu",
        "python_flags": {"isolated": 1},
        "process_environment": {"LC_ALL": "C.UTF-8"},
        "accelerator": None,
        "thread_count": 1,
        "interop_thread_count": 1,
        "cuda_runtime": None,
        "cuda_driver": None,
        "cublas_workspace_config": None,
        "deterministic_algorithms": True,
        "runtime_seal_sha256": digests[role_paths["runtime_seal_member"]],
        "runtime_seal_descriptor_custody": True,
        "producer_bootstrap_sha256": source["producer_bootstrap_sha256"],
        "bootstrap_descriptor_custody": True,
        "package_roots": [{"identity": "runtime-only-package-root"}],
        "standard_library": {"identity": "runtime-only-standard-library"},
    }
    custody: dict[str, object] = {
        "runtime_seal_member": role_paths["runtime_seal_member"],
        "runtime_seal_sha256": execution["runtime_seal_sha256"],
        "producer_bootstrap_member": role_paths["producer_bootstrap_member"],
        "producer_bootstrap_sha256": source["producer_bootstrap_sha256"],
        "launch_bootstrap_member": role_paths["launch_bootstrap_member"],
        "launch_bootstrap_sha256": source["launch_bootstrap_sha256"],
        "package_ownership": {"identity": "runtime-only-package-ownership"},
    }
    audit_execution: dict[str, object] = {
        "receipt_sha256": digests[role_paths["audit_reproduction_member"]],
        "runtime_manifest_sha256": digests[role_paths["audit_runtime_manifest_member"]],
        "invocation_manifest_sha256": digests[role_paths["audit_invocation_manifest_member"]],
        "stderr_sha256": digests[role_paths["audit_stderr_member"]],
        "bootstrap_sha256": digest("audit-bootstrap"),
        "runner_source_sha256": source["runner_source_sha256"],
        "auditor_source_sha256": source["auditor_source_sha256"],
        "support_files": [
            {"path": "producer_bootstrap.py", "bytes": 1, "sha256": digest("support-producer")},
            {"path": "protocol.json", "bytes": 1, "sha256": digest("support-protocol")},
            {"path": "schemas/raw-result.schema.json", "bytes": 1, "sha256": digest("support-schema")},
        ],
        "source_mode": "descriptor",
    }
    archive_digest = digest("qualification-archive")
    closure: dict[str, object] = {
        "schema": "prospect.wm001.development-closure.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.14.0",
        "source": source,
        "producer_root": str((tmp_path / "qualification-v1.14.0").resolve()),
        **{
            field: member
            for field, member in role_paths.items()
            if field not in {"runtime_seal_member", "producer_bootstrap_member", "launch_bootstrap_member"}
        },
        "producer_execution": execution,
        "producer_custody": custody,
        "audit_execution": audit_execution,
        "qualification_archive": {
            "format": "ustar-uncompressed-v1",
            "file": f"development-qualification-{archive_digest[:16]}.tar",
            "canonical_path": (
                "bench/world_model_lifecycle/results/development/"
                f"development-qualification-{archive_digest[:16]}.tar"
            ),
            "bytes": 1024,
            "sha256": archive_digest,
            "members": [
                {"path": member, "bytes": 1, "sha256": digests[member]}
                for member in sorted(role_paths.values())
            ],
        },
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    path = tmp_path / "development-closure-fixture.json"
    path.write_bytes(_canonical_payload(closure))
    return path, closure, digests


def test_recorded_closure_and_coverage_verifiers_ignore_qa_ambient_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure_path, closure, digests = _recorded_development_closure_fixture(tmp_path)
    coverage_report = binding_module.run_coverage_conformance()
    recorded_runtime = {
        "semantics_id": verify_module.COVERAGE_SEMANTICS,
        "python_executable": "/sealed-runtime/bin/python",
        "python_implementation": "CPython",
        "python_version": "3.12.11",
        "platform": "sealed-runtime-platform",
        "machine": "sealed-runtime-machine",
    }
    coverage_report.update(recorded_runtime)
    _rehash_conformance(coverage_report)
    dependencies = {
        "python_executable": recorded_runtime["python_executable"],
        "python_executable_sha256": "c" * 64,
    }
    runtime = {
        "platform": recorded_runtime["platform"],
        "machine": recorded_runtime["machine"],
    }
    bound_python = {
        "executable": recorded_runtime["python_executable"],
        "resolved_executable": recorded_runtime["python_executable"],
        "sha256": dependencies["python_executable_sha256"],
        "version": [3, 12, 11],
    }

    def forbidden() -> str:
        raise AssertionError("recorded verifier consulted QA ambient identity")

    monkeypatch.setattr(verify_module.sys, "executable", "/qa/bin/python")
    monkeypatch.setattr(verify_module.platform, "python_implementation", forbidden)
    monkeypatch.setattr(verify_module.platform, "python_version", forbidden)
    monkeypatch.setattr(verify_module.platform, "platform", forbidden)
    monkeypatch.setattr(verify_module.platform, "machine", forbidden)

    observed, identity, member_digests = (
        verify_module._recorded_development_closure_identity(closure_path)
    )
    verify_module._verify_bound_coverage_runtime_identity(
        recorded_runtime,
        dependencies=dependencies,
        runtime=runtime,
        bound_python=bound_python,
    )
    verify_module._verify_coverage_conformance_report(
        coverage_report,
        recorded_runtime=recorded_runtime,
    )

    assert observed == closure
    assert identity["producer_manifest_sha256"] == digests["producer/producer-manifest.json"]
    assert member_digests["producer/result.json"] == digests["producer/result.json"]


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("unsafe-member", id="unsafe-member"),
        pytest.param("duplicate-member", id="duplicate-member"),
        pytest.param("custody-digest", id="custody-digest"),
        pytest.param("status", id="status"),
    ],
)
def test_recorded_closure_verifier_rejects_cross_link_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    closure_path, closure, _ = _recorded_development_closure_fixture(tmp_path)
    mutated = copy.deepcopy(closure)
    archive = mutated["qualification_archive"]
    assert isinstance(archive, dict)
    members = archive["members"]
    assert isinstance(members, list)
    if mutation == "unsafe-member":
        members[0]["path"] = "../escape"
    elif mutation == "duplicate-member":
        members[1]["path"] = members[0]["path"]
    elif mutation == "custody-digest":
        custody = mutated["producer_custody"]
        assert isinstance(custody, dict)
        custody["runtime_seal_sha256"] = "0" * 64
    else:
        mutated["audit_reproduced"] = False
    closure_path.write_bytes(_canonical_payload(mutated))

    with pytest.raises(verify_module.Violation, match="recorded development"):
        verify_module._recorded_development_closure_identity(closure_path)


def test_recorded_accepted_closure_receipt_rejects_digest_substitution(
    tmp_path: Path,
) -> None:
    closure_path, closure, digests = _recorded_development_closure_fixture(tmp_path)
    closure_payload = closure_path.read_bytes()
    receipt = {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "accepted-closure-evidence",
        "passed": True,
        "development_closure_sha256": hashlib.sha256(closure_payload).hexdigest(),
        "producer_manifest_sha256": digests["producer/producer-manifest.json"],
        "raw_result_sha256": digests["producer/result.json"],
        "closure_attempt_manifest_sha256": "d" * 64,
        "closure_outer_completion_sha256": "d" * 64,
    }
    stdout = tmp_path / "accepted-closure.stdout.json"
    stderr = tmp_path / "accepted-closure.stderr.log"
    stdout.write_bytes(_canonical_payload(receipt))
    stderr.write_bytes(b"")
    report_path = tmp_path / "preformal-report.json"
    report = {
        "commands": [
            {
                "name": "runtime-accepted-closure-evidence",
                "stdout": {
                    "file": stdout.name,
                    "bytes": stdout.stat().st_size,
                    "sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
                },
                "stderr": {
                    "file": stderr.name,
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            }
        ]
    }
    report_path.write_bytes(_canonical_payload(report))
    verify_module._recorded_accepted_closure_receipt(
        report_path,
        report,
        closure_sha256=receipt["development_closure_sha256"],
        producer_manifest_sha256=receipt["producer_manifest_sha256"],
        raw_result_sha256=receipt["raw_result_sha256"],
    )

    with pytest.raises(verify_module.Violation, match="recorded closure"):
        verify_module._recorded_accepted_closure_receipt(
            report_path,
            report,
            closure_sha256=receipt["development_closure_sha256"],
            producer_manifest_sha256=receipt["producer_manifest_sha256"],
            raw_result_sha256="0" * 64,
        )


def test_formal_binding_file_requires_single_link_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_bytes(_canonical_payload({"fixture": "binding"}))
    verified = {"schema": "prospect.world-model-lifecycle.formal-binding.v10"}
    monkeypatch.setattr(
        verify_module,
        "verify_binding",
        lambda path: verified,
    )

    assert binding_module._verified_formal_binding_file(binding_path) == verified
    os.link(binding_path, tmp_path / "untrusted-binding-alias.json")
    with pytest.raises(RuntimeError, match="1-link custody"):
        binding_module._verified_formal_binding_file(binding_path)


def _producer_custody_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    ownership = {"identity": "package-ownership"}
    monkeypatch.setattr(
        binding_module,
        "package_root_ownership",
        lambda: ownership,
    )
    producer_bootstrap = Path(binding_module.__file__).with_name("producer_bootstrap.py").read_bytes()
    launch_bootstrap = Path(binding_module.__file__).with_name("launch_bootstrap.py").read_bytes()
    executable = str(Path(sys.executable).resolve(strict=True))
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    execution: dict[str, object] = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "python_executable": sys.executable,
        "python_executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "python_version": python_version,
        "python_flags": {"isolated": 1},
        "process_environment": {"LC_ALL": "C.UTF-8"},
        "package_roots": [{"identity": "package-root"}],
        "standard_library": {"identity": "standard-library"},
        "producer_bootstrap_sha256": hashlib.sha256(producer_bootstrap).hexdigest(),
    }
    seal: dict[str, object] = {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.14.0",
        "assurance": dict(binding_module.ASSURANCE),
        "git_commit": execution["git_commit"],
        "git_tree": execution["git_tree"],
        "worktree_clean": True,
        "python": {
            "executable": sys.executable,
            "resolved_executable": executable,
            "sha256": execution["python_executable_sha256"],
            "version": list(sys.version_info[:3]),
        },
        "required_flags": execution["python_flags"],
        "process_environment": execution["process_environment"],
        "bootstrap_source_sha256": execution["producer_bootstrap_sha256"],
        "standard_library": execution["standard_library"],
        "package_roots": execution["package_roots"],
        "package_ownership": ownership,
    }
    runtime_seal_payload = _canonical_payload(seal)
    execution["runtime_seal_sha256"] = hashlib.sha256(runtime_seal_payload).hexdigest()
    return execution, seal, producer_bootstrap, launch_bootstrap


def test_archived_producer_runtime_seal_requires_exact_assurance_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, seal, producer_bootstrap, launch_bootstrap = _producer_custody_fixture(monkeypatch)
    runtime_seal_payload = _canonical_payload(seal)

    observed = binding_module._validate_producer_custody(
        execution=execution,
        runtime_seal_payload=runtime_seal_payload,
        bootstrap_payload=producer_bootstrap,
        launch_bootstrap_payload=launch_bootstrap,
    )

    assert observed == {
        "runtime_seal_sha256": hashlib.sha256(runtime_seal_payload).hexdigest(),
        "producer_bootstrap_sha256": hashlib.sha256(producer_bootstrap).hexdigest(),
        "launch_bootstrap_sha256": hashlib.sha256(launch_bootstrap).hexdigest(),
        "package_ownership": {"identity": "package-ownership"},
    }


@pytest.mark.parametrize(
    "mutation",
    ("missing-assurance", "overstated-assurance", "extra-field"),
)
def test_archived_producer_runtime_seal_rejects_schema_or_assurance_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    execution, seal, producer_bootstrap, launch_bootstrap = _producer_custody_fixture(monkeypatch)
    if mutation == "missing-assurance":
        del seal["assurance"]
    elif mutation == "overstated-assurance":
        assurance = seal["assurance"]
        assert isinstance(assurance, dict)
        assurance["tamper_resistant"] = True
    else:
        seal["unbound"] = True
    runtime_seal_payload = _canonical_payload(seal)
    execution["runtime_seal_sha256"] = hashlib.sha256(runtime_seal_payload).hexdigest()

    with pytest.raises(RuntimeError, match="runtime seal/bootstrap custody"):
        binding_module._validate_producer_custody(
            execution=execution,
            runtime_seal_payload=runtime_seal_payload,
            bootstrap_payload=producer_bootstrap,
            launch_bootstrap_payload=launch_bootstrap,
        )


def _outer_finalized_development_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    from bench.world_model_lifecycle import artifact, operator

    completion_root = tmp_path / "outer-completions"
    completion_root.mkdir()
    monkeypatch.setattr(operator, "OUTER_COMPLETIONS_ROOT", completion_root)
    producer = tmp_path / "producer"
    producer.mkdir()
    result_path = producer / "result.json"
    result_payload = _canonical_payload({"fixture": "development-result"})
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
    manifest_path.write_bytes(_canonical_payload(manifest))
    os.link(
        manifest_path,
        operator.outer_completion_marker(manifest_path),
    )
    assert artifact.verify_producer_manifest(producer) == manifest
    return producer, manifest


def test_audit_reproduction_receipt_derives_descriptor_execution_identities(
    tmp_path: Path,
) -> None:
    from bench.world_model_lifecycle.audit_runner import (
        INVOCATION_MANIFEST_SCHEMA,
        RUNTIME_MANIFEST_SCHEMA,
        AuditExecution,
        CapturedFileIdentity,
        bootstrap_source_sha256,
    )

    audit = {"passed": True}
    audit_payload = _canonical_payload(audit)
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(audit_payload)
    auditor = Path(binding_module.__file__).with_name("artifact_audit.py")
    support_rows = binding_module._expected_development_audit_support_rows()
    runtime_payload = _canonical_payload(
        {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "bootstrap_sha256": bootstrap_source_sha256(),
            "source": {
                "mode": "descriptor",
                "path": "artifact_audit.py",
                "bytes": auditor.stat().st_size,
                "sha256": binding_module.sha256_file(auditor),
            },
            "support_files": support_rows,
        }
    )
    runtime_sha256 = hashlib.sha256(runtime_payload).hexdigest()
    invocation_payload = _canonical_payload(
        {
            "schema": INVOCATION_MANIFEST_SCHEMA,
            "runtime_manifest_sha256": runtime_sha256,
        }
    )
    execution = AuditExecution(
        command=(sys.executable, "-I", "-S", "-B", "/proc/self/fd/7"),
        returncode=0,
        stdout=audit_payload,
        stderr=b"",
        report=audit,
        runtime_manifest=runtime_payload,
        runtime_manifest_sha256=runtime_sha256,
        invocation_manifest=invocation_payload,
        invocation_manifest_sha256=hashlib.sha256(invocation_payload).hexdigest(),
        bootstrap_sha256=bootstrap_source_sha256(),
        auditor_source_sha256=binding_module.sha256_file(auditor),
        support_files=tuple(
            CapturedFileIdentity(
                relative_path=str(row["path"]),
                bytes=int(row["bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in support_rows
        ),
        source_mode="descriptor",
    )

    receipt_path = tmp_path / "receipt.json"
    receipt = binding_module.create_audit_reproduction_receipt(
        supplied_audit_path=audit_path,
        execution=execution,
        output_path=receipt_path,
    )

    assert receipt["schema"] == "prospect.wm001.audit-reproduction.v2"
    assert receipt["runner_source_sha256"] == binding_module.sha256_file(
        Path(binding_module.__file__).with_name("audit_runner.py")
    )
    assert receipt["runtime_manifest_sha256"] == runtime_sha256
    assert receipt["support_files"] == support_rows
    assert receipt_path.read_bytes() == _canonical_payload(receipt)


def test_audit_reproduction_receipt_rejects_caller_shaped_object(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(_canonical_payload({"passed": True}))

    with pytest.raises(TypeError, match="captured AuditExecution"):
        binding_module.create_audit_reproduction_receipt(
            supplied_audit_path=audit_path,
            execution=object(),
            output_path=tmp_path / "receipt.json",
        )


def test_development_qualification_archive_is_deterministic_and_stream_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "a.bin").write_bytes(b"producer-bytes")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)

    first_path, first_identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": b"audit-bytes"},
    )
    first_payload = first_path.read_bytes()
    retained = binding_module._stream_qualification_archive(
        first_path,
        first_identity,
        retained_members={"producer/a.bin", "evidence/audit.json"},
    )
    assert retained == {
        "producer/a.bin": b"producer-bytes",
        "evidence/audit.json": b"audit-bytes",
    }
    assert [row["path"] for row in first_identity["members"]] == [
        "evidence/audit.json",
        "producer/a.bin",
    ]

    second_path, second_identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": b"audit-bytes"},
    )
    assert second_identity == first_identity
    assert second_path.read_bytes() == first_payload


def test_development_archive_rejects_recomputed_identity_with_nonzero_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "result.json").write_bytes(b"{}\n")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(
        binding_module,
        "DEVELOPMENT_RESULTS_ROOT",
        destination,
    )
    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": b"{}\n"},
    )
    payload = bytearray(archive_path.read_bytes())
    assert payload[-1] == 0
    payload[-1] = 1
    digest = hashlib.sha256(payload).hexdigest()
    mutated_path = destination / (f"development-qualification-{digest[:16]}.tar")
    mutated_path.write_bytes(payload)
    mutated_identity = copy.deepcopy(identity)
    mutated_identity.update(
        {
            "file": mutated_path.name,
            "canonical_path": mutated_path.relative_to(tmp_path).as_posix(),
            "sha256": digest,
        }
    )

    with pytest.raises(RuntimeError, match="terminal records"):
        binding_module._stream_qualification_archive(
            mutated_path,
            mutated_identity,
            retained_members={"evidence/audit.json"},
        )


def test_development_archive_rejects_recomputed_identity_with_nonzero_member_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "result.json").write_bytes(b"{}\n")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(
        binding_module,
        "DEVELOPMENT_RESULTS_ROOT",
        destination,
    )
    audit_payload = b"{}\n"
    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": audit_payload},
    )
    payload = bytearray(archive_path.read_bytes())
    padding_offset = tarfile.BLOCKSIZE + len(audit_payload)
    assert payload[padding_offset] == 0
    payload[padding_offset] = 1
    digest = hashlib.sha256(payload).hexdigest()
    mutated_path = destination / (f"development-qualification-{digest[:16]}.tar")
    mutated_path.write_bytes(payload)
    mutated_identity = copy.deepcopy(identity)
    mutated_identity.update(
        {
            "file": mutated_path.name,
            "canonical_path": mutated_path.relative_to(tmp_path).as_posix(),
            "sha256": digest,
        }
    )

    with pytest.raises(RuntimeError, match="member padding"):
        binding_module._stream_qualification_archive(
            mutated_path,
            mutated_identity,
            retained_members={"evidence/audit.json"},
        )


def test_development_archive_rejects_recomputed_identity_with_noncanonical_header_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "result.json").write_bytes(b"{}\n")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(
        binding_module,
        "DEVELOPMENT_RESULTS_ROOT",
        destination,
    )
    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": b"{}\n"},
    )
    payload = bytearray(archive_path.read_bytes())
    payload[329:337] = b"0000001\0"
    payload[148:156] = b" " * 8
    checksum = sum(payload[: tarfile.BLOCKSIZE])
    payload[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    mutated_path = destination / (f"development-qualification-{digest[:16]}.tar")
    mutated_path.write_bytes(payload)
    mutated_identity = copy.deepcopy(identity)
    mutated_identity.update(
        {
            "file": mutated_path.name,
            "canonical_path": mutated_path.relative_to(tmp_path).as_posix(),
            "sha256": digest,
        }
    )

    with pytest.raises(RuntimeError, match="noncanonical or hidden header"):
        binding_module._stream_qualification_archive(
            mutated_path,
            mutated_identity,
            retained_members={"evidence/audit.json"},
        )


@pytest.mark.parametrize(
    ("member", "replacement"),
    [
        ("evidence/empty.log", False),
        ("evidence/one.log", True),
        ("evidence/empty.log", 0.0),
        ("evidence/one.log", 1.0),
    ],
)
def test_development_archive_rejects_member_size_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member: str,
    replacement: object,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "result.json").write_bytes(b"{}\n")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(
        binding_module,
        "DEVELOPMENT_RESULTS_ROOT",
        destination,
    )
    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={
            "evidence/empty.log": b"",
            "evidence/one.log": b"x",
        },
    )
    mutated = copy.deepcopy(identity)
    row = next(row for row in mutated["members"] if row["path"] == member)
    row["bytes"] = replacement

    with pytest.raises(RuntimeError, match="identity is malformed"):
        binding_module._stream_qualification_archive(
            archive_path,
            mutated,
            retained_members=set(),
        )


def test_development_closure_archive_and_recheck_accept_outer_finalized_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, manifest = _outer_finalized_development_producer(
        tmp_path,
        monkeypatch,
    )
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)

    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": b"audit"},
    )
    retained = binding_module._stream_qualification_archive(
        archive_path,
        identity,
        retained_members={
            "producer/producer-manifest.json",
            "producer/result.json",
        },
    )

    assert (
        binding_module._validate_archived_producer(
            retained,
            identity["members"],
        )
        == manifest
    )
    binding_module._reverify_live_development_producer(
        producer,
        expected_manifest=manifest,
        expected_result_sha256=hashlib.sha256((producer / "result.json").read_bytes()).hexdigest(),
    )
    assert (producer / "producer-manifest.json").stat().st_nlink == 2
    assert (producer / "result.json").stat().st_nlink == 1


def test_terminal_producer_recheck_streams_result_larger_than_64_mib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import artifact

    producer = tmp_path / "producer"
    producer.mkdir()
    result = producer / "result.json"
    result_bytes = (64 << 20) + 1
    with result.open("wb") as stream:
        stream.truncate(result_bytes)
    digest = hashlib.sha256()
    zero_chunk = b"\0" * (1 << 20)
    for _ in range(64):
        digest.update(zero_chunk)
    digest.update(b"\0")
    result_sha256 = digest.hexdigest()
    manifest = {
        "files": [
            {
                "path": "result.json",
                "bytes": result_bytes,
                "sha256": result_sha256,
            }
        ]
    }
    monkeypatch.setattr(
        artifact,
        "verify_producer_manifest",
        lambda _root: manifest,
    )
    monkeypatch.setattr(
        artifact,
        "_regular_producer_files",
        lambda _root: (result,),
    )
    original_read = binding_module.os.read
    requested_bytes: list[int] = []

    def bounded_read(descriptor: int, length: int) -> bytes:
        requested_bytes.append(length)
        return original_read(descriptor, length)

    monkeypatch.setattr(binding_module.os, "read", bounded_read)

    binding_module._reverify_live_development_producer(
        producer,
        expected_manifest=manifest,
        expected_result_sha256=result_sha256,
    )

    assert requested_bytes
    assert max(requested_bytes) <= 1 << 20


def test_streamed_regular_digest_rejects_in_read_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.json"
    target.write_bytes(b"a" * ((1 << 20) + 1))
    original_read = binding_module.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, length)
        if chunk and not mutated:
            mutated = True
            with target.open("r+b") as stream:
                stream.write(b"b")
                stream.flush()
                os.fsync(stream.fileno())
        return chunk

    monkeypatch.setattr(
        binding_module.os,
        "read",
        mutate_after_first_read,
    )

    with pytest.raises(RuntimeError, match="changed while read"):
        binding_module._stable_regular_digest(
            target,
            label="development result",
            maximum_bytes=target.stat().st_size,
        )


def test_streamed_regular_digest_rejects_path_namespace_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = tmp_path / "live"
    replacement = tmp_path / "replacement"
    displaced = tmp_path / "displaced"
    live.mkdir()
    replacement.mkdir()
    payload = b"a" * ((1 << 20) + 1)
    target = live / "result.json"
    target.write_bytes(payload)
    (replacement / "result.json").write_bytes(payload)
    original_read = binding_module.os.read
    replaced = False

    def replace_namespace_after_first_read(
        descriptor: int,
        length: int,
    ) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, length)
        if chunk and not replaced:
            replaced = True
            live.rename(displaced)
            replacement.rename(live)
        return chunk

    monkeypatch.setattr(
        binding_module.os,
        "read",
        replace_namespace_after_first_read,
    )

    with pytest.raises(RuntimeError, match="changed while read"):
        binding_module._stable_regular_digest(
            target,
            label="development result",
            maximum_bytes=len(payload),
        )


def test_streamed_regular_digest_rejects_aliases_and_nonregular_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    target.write_bytes(b"result")
    alias = tmp_path / "result-alias.json"
    alias.symlink_to(target)
    with pytest.raises(RuntimeError, match="aliased"):
        binding_module._stable_regular_digest(
            alias,
            label="development result",
            maximum_bytes=target.stat().st_size,
        )

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="link custody"):
        binding_module._stable_regular_digest(
            directory,
            label="development result",
            maximum_bytes=0,
        )


def test_development_archive_rejects_extra_manifest_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _ = _outer_finalized_development_producer(
        tmp_path,
        monkeypatch,
    )
    os.link(
        producer / "producer-manifest.json",
        tmp_path / "untrusted-manifest-alias.json",
    )
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)

    with pytest.raises(RuntimeError, match="2-link custody"):
        binding_module._write_qualification_archive(
            destination_directory=destination,
            producer_root=producer,
            evidence_payloads={"evidence/audit.json": b"audit"},
        )


def test_development_qualification_archive_rejects_hardlinks_and_row_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "a.bin").write_bytes(b"producer")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)
    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/audit.json": b"audit"},
    )

    tampered = copy.deepcopy(identity)
    tampered["members"][0]["bytes"] += 1
    with pytest.raises(RuntimeError):
        binding_module._stream_qualification_archive(
            archive_path,
            tampered,
            retained_members={"evidence/audit.json"},
        )

    alias = destination / "archive-alias.tar"
    os.link(archive_path, alias)
    with pytest.raises(RuntimeError, match="file limits"):
        binding_module._stream_qualification_archive(
            archive_path,
            identity,
            retained_members={"evidence/audit.json"},
        )


def test_archive_writer_rejects_hardlinked_producer_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    source = producer / "result.json"
    source.write_bytes(b"result")
    os.link(source, tmp_path / "result-alias.json")
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)

    with pytest.raises(RuntimeError, match="1-link custody"):
        binding_module._write_qualification_archive(
            destination_directory=destination,
            producer_root=producer,
            evidence_payloads={"evidence/audit.json": b"audit"},
        )


@pytest.mark.parametrize(
    ("payload", "replacement"),
    [
        (b"", False),
        (b"x", True),
        (b"", 0.0),
        (b"x", 1.0),
    ],
)
def test_receipt_sidecar_rejects_size_numeric_aliases(
    tmp_path: Path,
    payload: bytes,
    replacement: object,
) -> None:
    filename = binding_module._content_addressed_filename(
        "development-audit-stderr",
        payload,
        ".log",
    )
    (tmp_path / filename).write_bytes(payload)
    receipt = {
        "stderr_file": filename,
        "stderr_bytes": replacement,
        "stderr_sha256": hashlib.sha256(payload).hexdigest(),
    }

    with pytest.raises(RuntimeError, match="reference is malformed"):
        binding_module._receipt_sidecar(
            tmp_path / "audit-reproduction.json",
            receipt,
            prefix="stderr",
            label="development audit stderr",
        )


@pytest.mark.parametrize(
    ("payload", "replacement"),
    [
        (b"", False),
        (b"x", True),
        (b"", 0.0),
        (b"x", 1.0),
    ],
)
def test_archived_producer_rejects_member_size_numeric_aliases(
    payload: bytes,
    replacement: object,
) -> None:
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "prospect.wm001.producer-manifest.v1",
        "experiment_id": "WM-001",
        "lane": "development",
        "status": "completed",
        "error": None,
        "manifest_excludes": ["producer-manifest.json"],
        "file_count": 1,
        "files": [
            {
                "path": "payload.bin",
                "bytes": replacement,
                "sha256": payload_sha256,
            }
        ],
    }
    manifest_payload = _canonical_payload(manifest)
    retained = {
        "producer/producer-manifest.json": manifest_payload,
    }
    member_rows = [
        {
            "path": "producer/payload.bin",
            "bytes": len(payload),
            "sha256": payload_sha256,
        },
        {
            "path": "producer/producer-manifest.json",
            "bytes": len(manifest_payload),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        },
    ]

    with pytest.raises(RuntimeError, match="manifest row is malformed"):
        binding_module._validate_archived_producer(
            retained,
            member_rows,
        )


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_archived_producer_rejects_file_count_numeric_alias(
    replacement: object,
) -> None:
    payload = b"x"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "prospect.wm001.producer-manifest.v1",
        "experiment_id": "WM-001",
        "lane": "development",
        "status": "completed",
        "error": None,
        "manifest_excludes": ["producer-manifest.json"],
        "file_count": replacement,
        "files": [
            {
                "path": "payload.bin",
                "bytes": len(payload),
                "sha256": payload_sha256,
            }
        ],
    }
    manifest_payload = _canonical_payload(manifest)
    retained = {
        "producer/producer-manifest.json": manifest_payload,
    }
    member_rows = [
        {
            "path": "producer/payload.bin",
            "bytes": len(payload),
            "sha256": payload_sha256,
        },
        {
            "path": "producer/producer-manifest.json",
            "bytes": len(manifest_payload),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        },
    ]

    with pytest.raises(RuntimeError, match="manifest is incomplete"):
        binding_module._validate_archived_producer(
            retained,
            member_rows,
        )


def test_terminal_producer_recheck_rejects_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.world_model_lifecycle import artifact

    producer = tmp_path / "producer"
    producer.mkdir()
    result = producer / "result.json"
    result.write_bytes(b"result")
    os.link(result, tmp_path / "result-alias.json")
    result_sha256 = hashlib.sha256(b"result").hexdigest()
    manifest = {
        "files": [
            {
                "path": "result.json",
                "bytes": len(b"result"),
                "sha256": result_sha256,
            }
        ]
    }
    monkeypatch.setattr(
        artifact,
        "verify_producer_manifest",
        lambda _root: manifest,
    )
    monkeypatch.setattr(
        artifact,
        "_regular_producer_files",
        lambda _root: (result,),
    )

    with pytest.raises(RuntimeError, match="1-link custody"):
        binding_module._reverify_live_development_producer(
            producer,
            expected_manifest=manifest,
            expected_result_sha256=result_sha256,
        )


def test_archive_streamer_does_not_materialize_unretained_large_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "result.json").write_bytes(b"x" * 64)
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)
    archive_path, identity = binding_module._write_qualification_archive(
        destination_directory=destination,
        producer_root=producer,
        evidence_payloads={"evidence/small.json": b"small"},
    )
    monkeypatch.setattr(
        binding_module,
        "_MAX_RETAINED_QUALIFICATION_MEMBER_BYTES",
        8,
    )

    retained = binding_module._stream_qualification_archive(
        archive_path,
        identity,
        retained_members={"evidence/small.json"},
    )
    assert retained == {"evidence/small.json": b"small"}
    with pytest.raises(RuntimeError, match="retained qualification member"):
        binding_module._stream_qualification_archive(
            archive_path,
            identity,
            retained_members={"producer/result.json"},
        )


def test_result_qualification_binds_only_exact_structural_seed_and_budget_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = {"sealed": True}
    monkeypatch.setattr(
        binding_module,
        "_validate_execution_identity",
        lambda value, require_live_identity: execution,
    )
    result_sha256 = "a" * 64
    value = {
        "schema": "prospect.wm001.development-result-qualification.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.14.0",
        "protocol_sha256": binding_module.sha256_file(binding_module.PROTOCOL_PATH),
        "raw_result_sha256": result_sha256,
        "lane": "development",
        "claim_eligible": False,
        "replicates": [
            {
                "replicate_id": f"development-{index:02d}",
                "master_seed": seed,
                "episodes": 496,
                "transitions": 99_200,
                "predictive_metrics": 12,
                "policy_runs": 20,
                "updates": 6,
                "optimizer_batch_manifests": 5,
            }
            for index, seed in enumerate(verify_module.DEVELOPMENT_SEEDS)
        ],
        "matrix_contract_sha256": (binding_module._development_matrix_contract_sha256()),
        "producer_execution": execution,
    }
    payload = _canonical_payload(value)

    observed, observed_execution = binding_module._validate_result_qualification(
        payload,
        archived_result_sha256=result_sha256,
    )
    assert observed_execution == execution
    assert not any(key.startswith(("k3", "k4", "k5", "k6")) for key in observed)

    for mutation in (
        lambda row: row["replicates"][0].__setitem__("master_seed", 7),
        lambda row: row["replicates"][0].__setitem__(
            "master_seed",
            float(verify_module.DEVELOPMENT_SEEDS[0]),
        ),
        lambda row: row["replicates"][0].__setitem__("transitions", 99_199),
        lambda row: row["replicates"][0].__setitem__(
            "transitions",
            99_200.0,
        ),
        lambda row: row.__setitem__("matrix_contract_sha256", "0" * 64),
    ):
        adversarial = copy.deepcopy(value)
        mutation(adversarial)
        with pytest.raises(RuntimeError, match="exact seeds/budgets/matrix"):
            binding_module._validate_result_qualification(
                _canonical_payload(adversarial),
                archived_result_sha256=result_sha256,
            )


def test_development_matrix_contract_is_sorted_and_golden() -> None:
    value = binding_module._development_matrix_contract_value()

    assert value["predictive_contracts"] == sorted(
        value["predictive_contracts"]
    )
    assert value["policy_contracts"] == sorted(
        value["policy_contracts"]
    )
    assert (
        binding_module._development_matrix_contract_sha256()
        == binding_module._DEVELOPMENT_MATRIX_CONTRACT_SHA256
        == "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
    )


def test_development_matrix_contract_is_stable_across_fresh_interpreters() -> None:
    source = (
        "import sys;"
        f"sys.path.insert(0,{str(binding_module.REPO)!r});"
        "from bench.world_model_lifecycle.binding import "
        "_development_matrix_contract_sha256;"
        "print(_development_matrix_contract_sha256())"
    )
    observed: list[str] = []
    for hash_seed in ("0", "1", "7", "41", "123456789"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            (sys.executable, "-B", "-c", source),
            cwd=binding_module.REPO,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stderr == ""
        observed.append(completed.stdout.strip())
    for _ in range(4):
        completed = subprocess.run(
            (sys.executable, "-I", "-B", "-c", source),
            cwd=binding_module.REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stderr == ""
        observed.append(completed.stdout.strip())

    assert set(observed) == {
        binding_module._DEVELOPMENT_MATRIX_CONTRACT_SHA256
    }


def test_development_matrix_contract_golden_rejects_content_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = binding_module._development_matrix_contract_value()
    mutated = copy.deepcopy(original)
    assert isinstance(mutated["predictive_contracts"], list)
    mutated["predictive_contracts"].pop()
    monkeypatch.setattr(
        binding_module,
        "_development_matrix_contract_value",
        lambda: mutated,
    )

    with pytest.raises(RuntimeError, match="golden identity"):
        binding_module._development_matrix_contract_sha256()


def test_result_qualification_created_in_one_process_reopens_in_two_others(
    tmp_path: Path,
) -> None:
    execution = {"sealed": True}
    result_sha256 = "d" * 64
    qualification = {
        "schema": "prospect.wm001.development-result-qualification.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.14.0",
        "protocol_sha256": binding_module.sha256_file(
            binding_module.PROTOCOL_PATH
        ),
        "raw_result_sha256": result_sha256,
        "lane": "development",
        "claim_eligible": False,
        "replicates": [
            {
                "replicate_id": f"development-{index:02d}",
                "master_seed": seed,
                "episodes": 496,
                "transitions": 99_200,
                "predictive_metrics": 12,
                "policy_runs": 20,
                "updates": 6,
                "optimizer_batch_manifests": 5,
            }
            for index, seed in enumerate(verify_module.DEVELOPMENT_SEEDS)
        ],
        "matrix_contract_sha256": (
            binding_module._development_matrix_contract_sha256()
        ),
        "producer_execution": execution,
    }
    payload_path = tmp_path / "qualification.json"
    payload_path.write_bytes(_canonical_payload(qualification))
    source = (
        "import pathlib,sys;"
        f"sys.path.insert(0,{str(binding_module.REPO)!r});"
        "from bench.world_model_lifecycle import binding as b;"
        "b._validate_execution_identity="
        "(lambda value,require_live_identity:value);"
        "v,_=b._validate_result_qualification("
        "pathlib.Path(sys.argv[1]).read_bytes(),"
        f"archived_result_sha256={result_sha256!r});"
        "print(v['matrix_contract_sha256'])"
    )
    observed = []
    for _ in range(2):
        completed = subprocess.run(
            (
                sys.executable,
                "-I",
                "-B",
                "-c",
                source,
                str(payload_path),
            ),
            cwd=binding_module.REPO,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stderr == ""
        observed.append(completed.stdout.strip())

    assert observed == [
        binding_module._DEVELOPMENT_MATRIX_CONTRACT_SHA256
    ] * 2


def test_development_qualification_archive_rejects_link_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "results" / "development"
    destination.mkdir(parents=True)
    monkeypatch.setattr(binding_module, "REPO", tmp_path)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_RESULTS_ROOT", destination)
    temporary = destination / "temporary.tar"
    with temporary.open("wb") as stream:
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo("evidence/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO())
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    archive_path = destination / f"development-qualification-{digest[:16]}.tar"
    temporary.rename(archive_path)
    identity = {
        "format": "ustar-uncompressed-v1",
        "file": archive_path.name,
        "canonical_path": archive_path.relative_to(tmp_path).as_posix(),
        "bytes": archive_path.stat().st_size,
        "sha256": digest,
        "members": [
            {
                "path": "evidence/link",
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="noncanonical|tar metadata",
    ):
        binding_module._stream_qualification_archive(
            archive_path,
            identity,
            retained_members={"evidence/link"},
        )


def test_development_closure_creator_rejects_any_alternate_marker_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "development-closure-v1.14.0.json"
    monkeypatch.setattr(binding_module, "DEVELOPMENT_CLOSURE_PATH", canonical)

    with pytest.raises(RuntimeError, match="only be published"):
        binding_module.create_development_closure(
            producer_root=tmp_path / "producer",
            audit_path=tmp_path / "audit.json",
            audit_reproduction_path=tmp_path / "receipt.json",
            runtime_manifest_path=tmp_path / "runtime.json",
            output_path=tmp_path / "alternate.json",
        )


def test_preserved_development_closure_name_must_be_content_addressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _canonical_payload({"schema": "fixture"})
    canonical = tmp_path / "development-closure-v1.14.0.json"
    canonical.write_bytes(payload)
    monkeypatch.setattr(binding_module, "DEVELOPMENT_CLOSURE_PATH", canonical)
    assert binding_module._closure_path_mode(canonical, payload) == "canonical"

    copied = tmp_path / (f"development-closure-{hashlib.sha256(payload).hexdigest()[:16]}.json")
    copied.write_bytes(payload)
    assert binding_module._closure_path_mode(copied, payload) == "content-addressed-copy"
    arbitrary = tmp_path / "copied.json"
    arbitrary.write_bytes(payload)
    with pytest.raises(RuntimeError, match="neither canonical"):
        binding_module._closure_path_mode(arbitrary, payload)
