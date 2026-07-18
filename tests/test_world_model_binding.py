from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle import verify as verify_module
from bench.world_model_lifecycle.planning import run_pendulum_conformance


def test_protocol_140_seed_domain_and_master_seeds_are_exact() -> None:
    assert verify_module.DEVELOPMENT_SEEDS == (2439054559, 3246851043)
    assert verify_module.FORMAL_SEEDS == (
        339970590,
        474769515,
        550273937,
        438984650,
        2732731971,
        2253809848,
        2206960337,
        3506881479,
    )
    assert [
        verify_module.derive_seed(
            "predictive_validation_irrelevant_episode",
            2439054559,
            index,
        )
        for index in range(8)
    ] == [
        1501666155,
        608622105,
        3038957295,
        2485648490,
        3416187949,
        228997209,
        2858985894,
        450599828,
    ]
    assert (
        verify_module.derive_seed(
            "predictive_validation_irrelevant_action",
            3246851043,
            0,
        )
        == 11246224
    )
    assert (
        tuple(verify_module.derive_master_seed("development", index) for index in range(2))
        == verify_module.DEVELOPMENT_SEEDS
    )
    assert tuple(verify_module.derive_master_seed("formal", index) for index in range(8)) == verify_module.FORMAL_SEEDS


def test_protocol_140_irrelevant_control_contract_is_bound() -> None:
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


def test_formal_binding_schema_binds_protocol_140_and_fresh_seeds() -> None:
    schema = json.loads(
        verify_module.BINDING_SCHEMA_PATH.read_text(encoding="utf-8"),
    )

    assert schema["$id"].endswith("wm-001-formal-binding-v4.json")
    assert schema["properties"]["schema"]["const"] == "prospect.world-model-lifecycle.formal-binding.v4"
    assert schema["properties"]["protocol"]["properties"]["version"]["const"] == "1.4.0"
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


def test_raw_result_schema_binds_v140_heldout_split_and_formal_counts() -> None:
    schema = json.loads(
        verify_module.RESULT_SCHEMA_PATH.read_text(encoding="utf-8"),
    )
    replicate_limits = schema["allOf"][0]["then"]["properties"]["replicates"]["items"]["allOf"][1]["properties"]
    predictive_schema = schema["$defs"]["predictiveMetric"]
    predictive_properties = predictive_schema["properties"]
    gate_comparators = schema["$defs"]["gateCheck"]["properties"]["comparator"][
        "enum"
    ]

    assert schema["$id"].endswith("wm-001-raw-result-v4.json")
    assert schema["properties"]["schema"]["const"] == "prospect.world-model-lifecycle.raw-result.v4"
    assert schema["properties"]["protocol_version"]["const"] == "1.4.0"
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


def test_formal_matrix_verifier_requires_every_exact_v140_row() -> None:
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

    with pytest.raises(ValueError, match="exactly 1,024"):
        binding_module.create_formal_binding(
            output_path=tmp_path / "binding.json",
            test_report_path=test_report,
            conformance_cases=2,
            device="cpu",
        )


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

    with pytest.raises(RuntimeError, match="complete live manifest"):
        binding_module.verify_live_binding(
            tmp_path / "formal-binding.json",
            device="cpu",
        )
