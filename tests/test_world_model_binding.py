from __future__ import annotations

import base64
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from bench.world_model_lifecycle import artifact_audit, producer_bootstrap
from bench.world_model_lifecycle import binding as binding_module
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
        "fixture/__init__.py,,\n"
        "fixture-1.0.dist-info/METADATA,,\n"
        "fixture-1.0.dist-info/RECORD,,\n",
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
    assert bootstrap_identity["identity_sha256"] == hashlib.sha256(
        binding_module.canonical_json_bytes(identity_value)
    ).hexdigest()


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

    assert {"PYGAME_HIDE_SUPPORT_PROMPT", "SDL_AUDIODRIVER"} <= set(
        environment["required"]
    )
    assert environment["properties"]["PYGAME_HIDE_SUPPORT_PROMPT"] == {
        "const": "hide"
    }
    assert environment["properties"]["SDL_AUDIODRIVER"] == {"const": "dsp"}


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


def test_protocol_160_seed_domain_and_master_seeds_are_exact() -> None:
    assert verify_module.DEVELOPMENT_SEEDS == (2999896578, 3783052994)
    assert verify_module.FORMAL_SEEDS == (
        3863790658,
        3900021454,
        1437244820,
        3175470977,
        228708147,
        3835462042,
        3342200973,
        1751060143,
    )
    assert [
        verify_module.derive_seed(
            "predictive_validation_irrelevant_episode",
            2999896578,
            index,
        )
        for index in range(8)
    ] == [
        1845767432,
        3661058938,
        574263981,
        934079335,
        4213171309,
        957315966,
        1549061991,
        3810173688,
    ]
    assert (
        verify_module.derive_seed(
            "predictive_validation_irrelevant_action",
            3783052994,
            0,
        )
        == 2084673469
    )
    assert (
        tuple(verify_module.derive_master_seed("development", index) for index in range(2))
        == verify_module.DEVELOPMENT_SEEDS
    )
    assert tuple(verify_module.derive_master_seed("formal", index) for index in range(8)) == verify_module.FORMAL_SEEDS


def test_protocol_160_states_the_negative_assurance_boundary() -> None:
    protocol = json.loads(verify_module.PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert protocol["trust_model"] == {
        "id": "prospect.wm001.trust-model.v1",
        "tamper_resistant": False,
        "external_attestation": False,
        "exclusive_path_use_required": True,
        "statement": TRUST_MODEL_STATEMENT,
    }


def test_implementation_manifest_binds_prospective_harness_review() -> None:
    rows = binding_module.implementation_files()

    assert [
        row
        for row in rows
        if row["path"] == "docs/wm001-v160-prospective-harness-review.json"
    ]


def test_protocol_160_irrelevant_control_contract_is_bound() -> None:
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


def test_formal_binding_schema_binds_protocol_160_and_fresh_seeds() -> None:
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"),
    )

    assert schema["$id"].endswith("wm-001-formal-binding-v6.json")
    assert schema["properties"]["schema"]["const"] == "prospect.world-model-lifecycle.formal-binding.v6"
    assert "assurance" in schema["required"]
    assert schema["properties"]["assurance"]["properties"] == {
        "trust_model_id": {
            "const": "prospect.wm001.trust-model.v1",
        },
        "tamper_resistant": {"const": False},
        "external_attestation": {"const": False},
        "exclusive_path_use_required": {"const": True},
    }
    assert schema["properties"]["protocol"]["properties"]["version"]["const"] == "1.6.0"
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


def test_raw_result_schema_binds_v160_heldout_split_and_formal_counts() -> None:
    schema = json.loads(
        verify_module.RESULT_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    replicate_limits = schema["allOf"][0]["then"]["properties"]["replicates"]["items"]["allOf"][1]["properties"]
    predictive_schema = schema["$defs"]["predictiveMetric"]
    predictive_properties = predictive_schema["properties"]
    gate_comparators = schema["$defs"]["gateCheck"]["properties"]["comparator"]["enum"]

    assert schema["$id"].endswith("wm-001-raw-result-v6.json")
    assert schema["properties"]["schema"]["const"] == "prospect.world-model-lifecycle.raw-result.v6"
    assert schema["properties"]["protocol_version"]["const"] == "1.6.0"
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


def test_formal_matrix_verifier_requires_every_exact_v160_row() -> None:
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


def _canonical_payload(value: object) -> bytes:
    return bytes(binding_module.canonical_json_bytes(value)) + b"\n"


def test_formal_binding_file_requires_single_link_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_path = tmp_path / "formal-binding.json"
    binding_path.write_bytes(_canonical_payload({"fixture": "binding"}))
    verified = {"schema": "prospect.world-model-lifecycle.formal-binding.v6"}
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
    producer_bootstrap = Path(binding_module.__file__).with_name(
        "producer_bootstrap.py"
    ).read_bytes()
    launch_bootstrap = Path(binding_module.__file__).with_name(
        "launch_bootstrap.py"
    ).read_bytes()
    executable = str(Path(sys.executable).resolve(strict=True))
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    execution: dict[str, object] = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "python_executable": sys.executable,
        "python_executable_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        "python_version": python_version,
        "python_flags": {"isolated": 1},
        "process_environment": {"LC_ALL": "C.UTF-8"},
        "package_roots": [{"identity": "package-root"}],
        "standard_library": {"identity": "standard-library"},
        "producer_bootstrap_sha256": hashlib.sha256(
            producer_bootstrap
        ).hexdigest(),
    }
    seal: dict[str, object] = {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
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
        "bootstrap_source_sha256": execution[
            "producer_bootstrap_sha256"
        ],
        "standard_library": execution["standard_library"],
        "package_roots": execution["package_roots"],
        "package_ownership": ownership,
    }
    runtime_seal_payload = _canonical_payload(seal)
    execution["runtime_seal_sha256"] = hashlib.sha256(
        runtime_seal_payload
    ).hexdigest()
    return execution, seal, producer_bootstrap, launch_bootstrap


def test_archived_producer_runtime_seal_requires_exact_assurance_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, seal, producer_bootstrap, launch_bootstrap = (
        _producer_custody_fixture(monkeypatch)
    )
    runtime_seal_payload = _canonical_payload(seal)

    observed = binding_module._validate_producer_custody(
        execution=execution,
        runtime_seal_payload=runtime_seal_payload,
        bootstrap_payload=producer_bootstrap,
        launch_bootstrap_payload=launch_bootstrap,
    )

    assert observed == {
        "runtime_seal_sha256": hashlib.sha256(
            runtime_seal_payload
        ).hexdigest(),
        "producer_bootstrap_sha256": hashlib.sha256(
            producer_bootstrap
        ).hexdigest(),
        "launch_bootstrap_sha256": hashlib.sha256(
            launch_bootstrap
        ).hexdigest(),
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
    execution, seal, producer_bootstrap, launch_bootstrap = (
        _producer_custody_fixture(monkeypatch)
    )
    if mutation == "missing-assurance":
        del seal["assurance"]
    elif mutation == "overstated-assurance":
        assurance = seal["assurance"]
        assert isinstance(assurance, dict)
        assurance["tamper_resistant"] = True
    else:
        seal["unbound"] = True
    runtime_seal_payload = _canonical_payload(seal)
    execution["runtime_seal_sha256"] = hashlib.sha256(
        runtime_seal_payload
    ).hexdigest()

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

    assert binding_module._validate_archived_producer(
        retained,
        identity["members"],
    ) == manifest
    binding_module._reverify_live_development_producer(
        producer,
        expected_manifest=manifest,
        expected_result_sha256=hashlib.sha256(
            (producer / "result.json").read_bytes()
        ).hexdigest(),
    )
    assert (producer / "producer-manifest.json").stat().st_nlink == 2
    assert (producer / "result.json").stat().st_nlink == 1


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
    manifest = {"status": "completed"}
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
            expected_result_sha256=hashlib.sha256(b"result").hexdigest(),
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
        "protocol_version": "1.6.0",
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
        lambda row: row["replicates"][0].__setitem__("transitions", 99_199),
        lambda row: row.__setitem__("matrix_contract_sha256", "0" * 64),
    ):
        adversarial = copy.deepcopy(value)
        mutation(adversarial)
        with pytest.raises(RuntimeError, match="exact seeds/budgets/matrix"):
            binding_module._validate_result_qualification(
                _canonical_payload(adversarial),
                archived_result_sha256=result_sha256,
            )


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

    with pytest.raises(RuntimeError, match="tar metadata"):
        binding_module._stream_qualification_archive(
            archive_path,
            identity,
            retained_members={"evidence/link"},
        )


def test_development_closure_creator_rejects_any_alternate_marker_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "development-closure-v1.6.0.json"
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
    canonical = tmp_path / "development-closure-v1.6.0.json"
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
