from __future__ import annotations

import copy
import json
import math
import struct
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import bench.world_model_lifecycle.analysis as analysis_module
from bench.world_model_lifecycle.analysis import (
    T_CRITICAL_N8,
    analyze_result,
    recompute_aggregate_metrics,
)
from bench.world_model_lifecycle.verify import derive_seed

HERE = Path(__file__).resolve().parents[1] / "bench" / "world_model_lifecycle"
TASK_A = "pendulum_normal_torque"
TASK_B = "pendulum_reversed_torque"
TASK_IRRELEVANT = "independent_phase_oscillator"
ZERO = "0" * 64


def test_scientific_revision_uses_fresh_v1130_seed_domain() -> None:
    master = 560_818_116
    assert derive_seed("model_initialization", master, 0) == 2_719_071_855
    assert derive_seed("planner", master, 0) == 2_826_758_534
    assert derive_seed("collection_action", master, 1) == 1_472_847_698
    assert derive_seed("irrelevant_collection_action", master, 0) == 2_240_135_814
    assert derive_seed("collect_irrelevant_episode", master, 0) == 3_225_599_521
    assert derive_seed("predictive_validation_irrelevant_action", master, 0) == 534_162_265
    assert derive_seed("predictive_validation_irrelevant_episode", master, 0) == 1_594_728_968


def _seed(namespace: str, master: int, index: int) -> int:
    payload = f"WM-001|1.13.0|{namespace}|{master}|{index}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _episode(
    replicate_id: str,
    index: int,
    task: str,
    split: str,
    condition: str,
    reset_seed: int,
    episode_return: float,
    parameter_digest: str,
    model_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = f"{replicate_id}:episode:{index}"
    run_id = f"{replicate_id}:{task}:{split}:{condition}"
    transition_ids = [f"{episode_id}:transition:{step}" for step in range(200)]
    held_out = split.startswith(("predictive_validation_", "behavior_evaluation_"))
    episode = {
        "episode_id": episode_id,
        "run_id": run_id,
        "task_id": task,
        "split": split,
        "condition": condition,
        "checkpoint_id": condition,
        "reset_seed": reset_seed,
        "process_id": 11,
        "model_version": model_version,
        "parameter_sha256": parameter_digest,
        "learning_allowed": not held_out,
        "replay_writes_allowed": not held_out,
        "environment_steps": 200,
        "return": episode_return,
        "started_at_utc": "2026-07-17T00:00:00+00:00",
        "completed_at_utc": "2026-07-17T00:01:00+00:00",
        "action_trace_sha256": _canonical_digest(
            {
                "intended": [0.0] * 200,
                "applied": [0.0] * 200,
            }
        ),
        "transition_ids": transition_ids,
    }
    transitions = [
        {
            "transition_id": transition_id,
            "run_id": run_id,
            "episode_id": episode_id,
            "task_id": task,
            "task_context": (0.0 if task == TASK_A else 1.0 if task == TASK_B else 2.0),
            "split": split,
            "step_index": step,
            "real_or_imagined": "real",
            "pre_observation_id": f"{transition_id}:pre",
            "decision_id": f"{transition_id}:decision",
            "executed_action_id": f"{transition_id}:execution",
            "next_observation_id": f"{transition_id}:next",
            "model_version_at_action": model_version,
            "parameter_sha256_at_action": parameter_digest,
            "pre_observation": [1.0, 0.0, 0.0],
            "intended_action": 0.0,
            "applied_action": 0.0,
            "next_observation": [1.0, 0.0, 0.0],
            "reward": episode_return / 200.0,
            "terminated": False,
            "truncated": step == 199,
            "scaled_target": [0.0, 0.0, 0.0, episode_return / 200.0 / 16.2736044],
            "target_sha256": sha256(
                struct.pack(
                    "<4d",
                    0.0,
                    0.0,
                    0.0,
                    episode_return / 200.0 / 16.2736044,
                )
            ).hexdigest(),
        }
        for step, transition_id in enumerate(transition_ids)
    ]
    return episode, transitions


def _update(
    phase: str,
    predecessor_digest: str,
    committed_digest: str,
    predecessor_version: str,
    committed_version: str,
    eligible: list[str],
    consumed: list[str],
    *,
    rejected: bool = False,
) -> dict[str, Any]:
    multiset = sha256(b"".join(value.encode() + b"\n" for value in consumed)).hexdigest()
    before = _digest(f"{phase}:before")
    permutation_digest = _digest("corrupted-permutation") if phase == "train_a_corrupted" else None
    row: dict[str, Any] = {
        "receipt_id": f"receipt:{phase}",
        "phase": phase,
        "status": "rejected" if rejected else "committed",
        "predecessor_parameter_sha256": predecessor_digest,
        "candidate_parameter_sha256": _digest(f"{phase}:rejected-candidate") if rejected else committed_digest,
        "committed_parameter_sha256": predecessor_digest if rejected else committed_digest,
        "predecessor_model_version": predecessor_version,
        "committed_model_version": predecessor_version if rejected else committed_version,
        "eligible_splits": eligible,
        "eligible_transition_count": len(consumed),
        "eligible_transition_ids": consumed,
        "consumed_sample_count": 0 if rejected else 1 * 5 * 256,
        "consumed_multiset_sha256": multiset,
        "sampling_manifest_sha256": None if rejected else _digest(f"batch:{phase}"),
        "target_permutation_sha256": permutation_digest,
        "target_permutation_file": (
            None
            if permutation_digest is None
            else {
                "media_type": "application/octet-stream",
                "bytes": 1,
                "sha256": permutation_digest,
                "filename": "corrupted-permutation.bin",
            }
        ),
        "optimizer_steps": 0 if rejected else 1,
        "live_state_before_sha256": before,
        "live_state_after_sha256": before if rejected else _digest(f"{phase}:after"),
    }
    if phase == "rejected_update_probe":
        full_state_digest = _digest("rejected-probe-full-state")
        row.update(
            {
                "full_state_before_sha256": full_state_digest,
                "full_state_after_sha256": full_state_digest,
                "full_state_before_file": {
                    "media_type": ("application/vnd.prospect.wm001.rejected-probe-state+json"),
                    "bytes": 1,
                    "sha256": full_state_digest,
                    "filename": "rejected-probe-before.json",
                },
                "full_state_after_file": {
                    "media_type": ("application/vnd.prospect.wm001.rejected-probe-state+json"),
                    "bytes": 1,
                    "sha256": full_state_digest,
                    "filename": "rejected-probe-after.json",
                },
            }
        )
    return row


def _development_result() -> dict[str, Any]:
    protocol = json.loads((HERE / "protocol.json").read_text())
    protocol_sha = sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    master = 560_818_116
    replicate_id = f"dev-{master}"
    cold = _digest("cold")
    after_a = _digest("after-a")
    corrupted = _digest("corrupted")
    irrelevant = _digest("irrelevant")
    replay = _digest("replay")
    naive = _digest("naive")
    versions = {
        "cold": "model-0",
        "frozen": "model-0",
        "after_a": "model-1",
        "corrupted": "model-corrupted",
        "irrelevant": "model-irrelevant",
        "after_b_replay": "model-2-replay",
        "after_b_naive": "model-2-naive",
        "random": "random-v1",
        "oracle": "oracle-v1",
        "collection_random": "collection-v1",
        "validation_random": "validation-v1",
    }
    digests = {
        "cold": cold,
        "frozen": cold,
        "after_a": after_a,
        "corrupted": corrupted,
        "irrelevant": irrelevant,
        "after_b_replay": replay,
        "after_b_naive": naive,
        "random": _digest("random"),
        "oracle": _digest("oracle"),
        "collection_random": cold,
        "validation_random": cold,
    }
    returns = {
        (TASK_A, "cold"): -1000.0,
        (TASK_A, "after_a"): -700.0,
        (TASK_A, "frozen"): -1000.0,
        (TASK_A, "corrupted"): -950.0,
        (TASK_A, "irrelevant"): -950.0,
        (TASK_A, "after_b_replay"): -720.0,
        (TASK_A, "after_b_naive"): -900.0,
        (TASK_A, "random"): -1300.0,
        (TASK_A, "oracle"): -300.0,
        (TASK_B, "after_a"): -1100.0,
        (TASK_B, "after_b_replay"): -700.0,
        (TASK_B, "after_b_naive"): -680.0,
        (TASK_B, "random"): -1300.0,
        (TASK_B, "oracle"): -300.0,
    }
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    def add(task: str, split: str, condition: str, seed: int, value: float) -> None:
        checkpoint_id = (
            "cold"
            if split in {"collect_a", "collect_irrelevant"}
            else "after_a"
            if split
            in {
                "collect_b",
                "predictive_validation_a",
                "predictive_validation_b",
            }
            else "irrelevant"
            if split == "predictive_validation_irrelevant"
            else condition
        )
        episode, rows = _episode(
            replicate_id,
            len(episodes),
            task,
            split,
            condition,
            seed,
            value,
            digests[checkpoint_id],
            versions[checkpoint_id],
        )
        episode["checkpoint_id"] = checkpoint_id
        episodes.append(episode)
        transitions.extend(rows)

    add(
        TASK_A,
        "collect_a",
        "collection_random",
        _seed("collect_a_episode", master, 0),
        -1000.0,
    )
    add(
        TASK_B,
        "collect_b",
        "collection_random",
        _seed("collect_b_episode", master, 0),
        -1000.0,
    )
    add(
        TASK_IRRELEVANT,
        "collect_irrelevant",
        "collection_random",
        _seed("collect_irrelevant_episode", master, 0),
        100.0,
    )
    add(
        TASK_A,
        "predictive_validation_a",
        "validation_random",
        _seed("predictive_validation_a_episode", master, 0),
        -1000.0,
    )
    add(
        TASK_B,
        "predictive_validation_b",
        "validation_random",
        _seed("predictive_validation_b_episode", master, 0),
        -1000.0,
    )
    add(
        TASK_IRRELEVANT,
        "predictive_validation_irrelevant",
        "validation_random",
        _seed("predictive_validation_irrelevant_episode", master, 0),
        100.0,
    )
    for condition in (
        "cold",
        "after_a",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_b_replay",
        "after_b_naive",
        "random",
        "oracle",
    ):
        add(
            TASK_A,
            "behavior_evaluation_a",
            condition,
            _seed("behavior_evaluation_a_episode", master, 0),
            returns[(TASK_A, condition)],
        )
    for condition in ("after_a", "after_b_replay", "after_b_naive", "random", "oracle"):
        add(
            TASK_B,
            "behavior_evaluation_b",
            condition,
            _seed("behavior_evaluation_b_episode", master, 0),
            returns[(TASK_B, condition)],
        )

    collect_a_ids = next(episode["transition_ids"] for episode in episodes if episode["split"] == "collect_a")
    collect_b_ids = next(episode["transition_ids"] for episode in episodes if episode["split"] == "collect_b")
    collect_irrelevant_ids = next(
        episode["transition_ids"] for episode in episodes if episode["split"] == "collect_irrelevant"
    )
    a_inputs = list(collect_a_ids)
    b_inputs = list(collect_b_ids)
    irrelevant_inputs = list(collect_irrelevant_ids)
    replay_inputs = [*collect_a_ids, *collect_b_ids]
    updates = [
        _update("train_a", cold, after_a, "model-0", "model-1", ["collect_a"], a_inputs),
        _update(
            "train_a_corrupted",
            cold,
            corrupted,
            "model-0",
            "model-corrupted",
            ["collect_a"],
            a_inputs,
        ),
        _update(
            "train_a_irrelevant",
            cold,
            irrelevant,
            "model-0",
            "model-irrelevant",
            ["collect_irrelevant"],
            irrelevant_inputs,
        ),
        _update(
            "train_b_replay",
            after_a,
            replay,
            "model-1",
            "model-2-replay",
            ["collect_a", "collect_b"],
            replay_inputs,
        ),
        _update(
            "train_b_naive",
            after_a,
            naive,
            "model-1",
            "model-2-naive",
            ["collect_b"],
            b_inputs,
        ),
        _update(
            "rejected_update_probe",
            after_a,
            after_a,
            "model-1",
            "model-1",
            ["collect_a"],
            [],
            rejected=True,
        ),
    ]
    cold_live_state = _digest("cold:compound")
    updates[0]["live_state_before_sha256"] = cold_live_state
    updates[1]["live_state_before_sha256"] = cold_live_state
    updates[2]["live_state_before_sha256"] = cold_live_state
    updates[2]["sampling_manifest_sha256"] = updates[0]["sampling_manifest_sha256"]
    post_a_live_state = updates[0]["live_state_after_sha256"]
    updates[3]["live_state_before_sha256"] = post_a_live_state
    updates[4]["live_state_before_sha256"] = post_a_live_state
    updates[5]["live_state_before_sha256"] = post_a_live_state
    updates[5]["live_state_after_sha256"] = post_a_live_state
    update_by_phase = {row["phase"]: row for row in updates}
    live_states = {
        "cold": update_by_phase["train_a"]["live_state_before_sha256"],
        "frozen": update_by_phase["train_a"]["live_state_before_sha256"],
        "corrupted": update_by_phase["train_a_corrupted"]["live_state_after_sha256"],
        "irrelevant": update_by_phase["train_a_irrelevant"]["live_state_after_sha256"],
        "after_a": update_by_phase["train_a"]["live_state_after_sha256"],
        "after_b_replay": update_by_phase["train_b_replay"]["live_state_after_sha256"],
        "after_b_naive": update_by_phase["train_b_naive"]["live_state_after_sha256"],
    }
    predictive = [
        {
            "task_id": task,
            "condition": condition,
            "checkpoint_id": condition,
            "model_version": versions[condition],
            "parameter_sha256": digests[condition],
            "live_state_sha256": live_states[condition],
            "split": (
                "predictive_validation_a"
                if task == TASK_A
                else "predictive_validation_b"
                if task == TASK_B
                else "predictive_validation_irrelevant"
            ),
            "transition_count": 200,
            "mixture_nll_nats_per_target_dimension": nll,
            "normalized_rmse": 0.1,
            "coverage_semantics": "wm001-mixture-pit-binary64-count-v1",
            "interval_90_covered_target_count": int(coverage * 800),
            "coverage_target_count": 800,
            "interval_90_coverage": coverage,
            "prediction_rows_sha256": _digest(f"prediction:{task}:{condition}"),
            "prediction_evidence_file": f"{task}-{condition}.bin",
            "prediction_evidence_bytes": 1,
        }
        for task, condition, nll, coverage in (
            (TASK_A, "cold", 0.45, 0.7),
            (TASK_A, "after_a", 0.10, 0.90),
            (TASK_A, "frozen", 0.40, 0.7),
            (TASK_A, "corrupted", 0.50, 0.6),
            (TASK_A, "irrelevant", 0.38, 0.7),
            (TASK_A, "after_b_replay", 0.12, 0.9),
            (TASK_A, "after_b_naive", 0.30, 0.8),
            (TASK_B, "after_a", 0.50, 0.7),
            (TASK_B, "after_b_replay", 0.20, 0.9),
            (TASK_B, "after_b_naive", 0.18, 0.9),
            (TASK_IRRELEVANT, "cold", 0.70, 0.7),
            (TASK_IRRELEVANT, "irrelevant", 0.10, 0.9),
        )
    ]
    components = [
        {
            "checkpoint_id": "after_b_replay",
            "component_id": component_id,
            "logical_version": "1",
            "media_type": "application/octet-stream",
            "bytes": 1,
            "sha256": _digest(f"component:{component_id}"),
            "predecessor_sha256": None,
        }
        for component_id in protocol["bindings"]["checkpoint"]["canonical_component_ids"]
    ]
    namespaces = [
        {
            "namespace": namespace,
            "values": [_seed(namespace, master, index) for index in range(declaration["count"])],
        }
        for namespace, declaration in protocol["seed_schedule"]["namespaces"].items()
    ]
    policy_runs: list[dict[str, Any]] = []
    for episode in episodes:
        task_index = 0 if episode["task_id"] == TASK_A else 1
        if episode["split"] == "collect_irrelevant":
            seed_namespace, seed_index = "irrelevant_collection_action", 0
            controller_kind = "uniform_random"
        elif episode["split"] == "predictive_validation_irrelevant":
            seed_namespace, seed_index = "predictive_validation_irrelevant_action", 0
            controller_kind = "uniform_random"
        elif episode["split"] in {"collect_a", "collect_b"}:
            seed_namespace, seed_index = "collection_action", task_index
            controller_kind = "uniform_random"
        elif episode["split"] in {"predictive_validation_a", "predictive_validation_b"}:
            seed_namespace, seed_index = "predictive_validation_action", task_index
            controller_kind = "uniform_random"
        elif episode["condition"] == "random":
            seed_namespace, seed_index = "random_policy_action", task_index
            controller_kind = "uniform_random"
        else:
            seed_namespace, seed_index = "planner", 0
            controller_kind = "cem_oracle" if episode["condition"] == "oracle" else "cem_learned"
        intended = [0.0] * 200
        applied = [0.0] * 200
        paired_rng_label = (
            f"planner:{episode['task_id']}" if controller_kind.startswith("cem_") else str(episode["run_id"])
        )
        policy_runs.append(
            {
                "run_id": episode["run_id"],
                "task_id": episode["task_id"],
                "split": episode["split"],
                "condition": episode["condition"],
                "checkpoint_id": episode["checkpoint_id"],
                "controller_kind": controller_kind,
                "controller_version": (
                    f"wm001-sha256:{episode['parameter_sha256']}"
                    if controller_kind == "cem_learned"
                    else "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1"
                    if controller_kind == "cem_oracle"
                    else "fixture-controller-v1"
                ),
                "seed_namespace": seed_namespace,
                "seed_index": seed_index,
                "seed": _seed(seed_namespace, master, seed_index),
                "reset_seeds": [episode["reset_seed"]],
                "episode_ids": [episode["episode_id"]],
                "rng_start_sha256": _digest(f"{paired_rng_label}:start"),
                "rng_end_sha256": _digest(f"{paired_rng_label}:end"),
                "action_count": 200,
                "action_trace_sha256": _canonical_digest(
                    {
                        "episode_ids": [episode["episode_id"]],
                        "intended": intended,
                        "applied": applied,
                    }
                ),
                "planner_budget": (
                    {
                        "planning_horizon": 10,
                        "optim_steps": 3,
                        "num_candidates": 64,
                        "top_k": 8,
                    }
                    if controller_kind.startswith("cem_")
                    else None
                ),
            }
        )
    evaluated_checkpoints = [
        {
            "condition": condition,
            "model_version": versions[condition],
            "parameter_sha256": digests[condition],
            "live_state_sha256": live_states[condition],
            "media_type": "application/vnd.prospect.wm001.owned-model-state",
            "bytes": 1,
            "sha256": live_states[condition],
            "filename": f"{condition}.model.bin",
        }
        for condition in (
            "cold",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_a",
            "after_b_replay",
            "after_b_naive",
        )
    ]
    replicate = {
        "replicate_id": replicate_id,
        "master_seed": master,
        "derived_seeds": namespaces,
        "episodes": episodes,
        "transitions": transitions,
        "updates": updates,
        "optimizer_batch_manifests": [
            {
                "phase": phase,
                "media_type": "application/json",
                "bytes": 1,
                "sha256": (_digest("batch:train_a") if phase == "train_a_irrelevant" else _digest(f"batch:{phase}")),
                "filename": f"{phase}.bin",
            }
            for phase in (
                "train_a",
                "train_a_corrupted",
                "train_a_irrelevant",
                "train_b_replay",
                "train_b_naive",
            )
        ],
        "predictive_metrics": predictive,
        "policy_runs": policy_runs,
        "evaluated_checkpoints": evaluated_checkpoints,
        "checkpoint_components": components,
        "checkpoint_archive": {
            "media_type": "application/vnd.prospect.checkpoint",
            "bytes": 1,
            "sha256": _digest("checkpoint-archive"),
            "filename": "checkpoint.bin",
        },
        "restart_parity": {
            "checkpoint_manifest_sha256": _digest("manifest"),
            "original_process_id": 11,
            "restored_process_id": 22,
            "fresh_process": True,
            "component_hash_mismatches": [],
            "identity_or_lineage_mismatches": 0,
            "prediction_max_abs_difference": 0.0,
            "action_max_abs_difference": 0.0,
            "episode_return_max_abs_difference": 0.0,
            "live_evaluation": {
                "media_type": "application/vnd.prospect.wm001.restart-evaluation+json",
                "bytes": 1,
                "sha256": _digest("live-evaluation"),
                "filename": "live-evaluation.json",
            },
            "restored_evaluation": {
                "media_type": "application/vnd.prospect.wm001.restart-evaluation+json",
                "bytes": 1,
                "sha256": _digest("restored-evaluation"),
                "filename": "restored-evaluation.json",
            },
        },
    }
    return {
        "schema": "prospect.world-model-lifecycle.raw-result.v9",
        "experiment_id": "WM-001",
        "protocol_version": "1.13.0",
        "protocol_sha256": protocol_sha,
        "lane": "development",
        "claim_eligible": False,
        "formal_binding_sha256": None,
        "started_at_utc": "2026-07-17T00:00:00+00:00",
        "completed_at_utc": "2026-07-17T01:00:00+00:00",
        "execution": {},
        "replicates": [replicate],
        "aggregate_metrics": [{"name": "fabricated", "mean": 999999}],
        "gate_results": [{"gate": "K7", "passed": True}],
        "limitations": [],
    }


def test_analysis_recomputes_rows_and_passes_development_gates_diagnostically() -> None:
    result = _development_result()
    analysis = analyze_result(result)

    assert [gate["gate"] for gate in analysis["gate_results"]] == [f"K{index}" for index in range(8)]
    assert all(gate["passed"] for gate in analysis["gate_results"])
    assert not any(gate["claim_supported"] for gate in analysis["gate_results"])
    metrics = {metric["name"]: metric for metric in analysis["aggregate_metrics"]}
    assert "fabricated" not in metrics
    assert metrics["a_nll_improvement_after_a_vs_irrelevant"]["replicate_values"] == pytest.approx([0.28])
    assert metrics["a_irrelevant_nll_improvement_vs_frozen"]["replicate_values"] == pytest.approx([0.02])
    assert metrics["a_return_improvement_after_a_vs_irrelevant"]["replicate_values"] == pytest.approx([250.0])
    assert metrics["a_irrelevant_return_improvement_vs_frozen"]["replicate_values"] == pytest.approx([50.0])
    retention = metrics["retained_a_gain_fraction"]
    assert retention["replicate_values"] == pytest.approx([280.0 / 300.0])
    assert retention["ci_95_lower"] == pytest.approx(retention["mean"])
    gates = {gate["gate"]: gate for gate in analysis["gate_results"]}
    k3_checks = {check["name"]: check for check in gates["K3"]["checks"]}
    k4_checks = {check["name"]: check for check in gates["K4"]["checks"]}
    assert k3_checks["a_vs_irrelevant_mean_nll_improvement"]["passed"] is True
    assert k3_checks["a_vs_irrelevant_nll_improvement_ci_lower"]["passed"] is True
    assert k4_checks["after_a_vs_irrelevant_mean_return_improvement"]["passed"] is True
    assert k4_checks["after_a_vs_irrelevant_return_improvement_ci_lower"]["passed"] is True
    assert any(finding["code"] == "development_claim_ineligible" for finding in analysis["audit_findings"])


def test_first_failed_numeric_gate_stops_the_ordered_prefix() -> None:
    result = _development_result()
    after_a = next(
        row
        for row in result["replicates"][0]["predictive_metrics"]
        if row["task_id"] == TASK_A and row["condition"] == "after_a"
    )
    after_a["mixture_nll_nats_per_target_dimension"] = 0.60

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0", "K1", "K2", "K3"]
    assert gates[-1]["passed"] is False
    assert gates[-1]["stop_reason"].startswith("Stop.")


def test_k0_rejects_coverage_fraction_or_count_disagreement() -> None:
    result = _development_result()
    after_a = next(
        row
        for row in result["replicates"][0]["predictive_metrics"]
        if row["task_id"] == TASK_A and row["condition"] == "after_a"
    )
    after_a["interval_90_covered_target_count"] += 1

    analysis = analyze_result(result)
    gates = analysis["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0"]
    assert gates[0]["passed"] is False
    assert any(
        "coverage counts, transition count, and fraction disagree" in finding["message"]
        for finding in analysis["audit_findings"]
    )


def test_k3_coverage_bounds_use_exact_integer_cross_products() -> None:
    at_lower = [
        {
            "_a_after_a_interval_90_covered_target_count": 4_480.0,
            "_a_after_a_coverage_target_count": 6_400.0,
        }
        for _ in range(8)
    ]
    below_lower = copy.deepcopy(at_lower)
    below_lower[0]["_a_after_a_interval_90_covered_target_count"] -= 1.0
    at_upper = [
        {
            "_a_after_a_interval_90_covered_target_count": 6_336.0,
            "_a_after_a_coverage_target_count": 6_400.0,
        }
        for _ in range(8)
    ]
    above_upper = copy.deepcopy(at_upper)
    above_upper[0]["_a_after_a_interval_90_covered_target_count"] += 1.0

    assert all(check["passed"] for check in analysis_module._coverage_count_gate_checks(at_lower))
    assert analysis_module._coverage_count_gate_checks(below_lower)[0]["passed"] is False
    assert all(check["passed"] for check in analysis_module._coverage_count_gate_checks(at_upper))
    assert analysis_module._coverage_count_gate_checks(above_upper)[1]["passed"] is False


def test_k3_rejects_irrelevant_evidence_matching_task_a_prediction() -> None:
    result = _development_result()
    irrelevant = next(
        row
        for row in result["replicates"][0]["predictive_metrics"]
        if row["task_id"] == TASK_A and row["condition"] == "irrelevant"
    )
    irrelevant["mixture_nll_nats_per_target_dimension"] = 0.10

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0", "K1", "K2", "K3"]
    assert gates[-1]["passed"] is False
    failed = {check["name"] for check in gates[-1]["checks"] if not check["passed"]}
    assert "a_vs_irrelevant_mean_nll_improvement" in failed
    assert "a_vs_irrelevant_nll_improvement_ci_lower" in failed


def test_k0_rejects_reselected_environment_reset_seed() -> None:
    result = _development_result()
    result["replicates"][0]["episodes"][0]["reset_seed"] += 1

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0"]
    assert gates[0]["passed"] is False
    seed_check = next(check for check in gates[0]["checks"] if check["name"] == "derived_seed_schedule_exact")
    assert seed_check["observed"] >= 1


def test_k0_rejects_episode_outside_the_sealed_run_matrix() -> None:
    result = _development_result()
    result["replicates"][0]["episodes"][0]["checkpoint_id"] = "after_a"

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0"]
    assert gates[0]["passed"] is False


def test_k1_rejects_wrong_timelimit_flag() -> None:
    result = _development_result()
    result["replicates"][0]["transitions"][0]["truncated"] = True

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0", "K1"]
    assert gates[-1]["passed"] is False


def test_k2_rejects_optimizer_state_branch_mismatch() -> None:
    result = _development_result()
    corrupted = next(update for update in result["replicates"][0]["updates"] if update["phase"] == "train_a_corrupted")
    corrupted["live_state_before_sha256"] = _digest("different-optimizer-state")

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0", "K1", "K2"]
    assert gates[-1]["passed"] is False


def test_k2_rejects_full_state_rollback_mismatch() -> None:
    result = _development_result()
    rejected_probe = next(
        update for update in result["replicates"][0]["updates"] if update["phase"] == "rejected_update_probe"
    )
    rejected_probe["full_state_after_sha256"] = _digest("mutated-full-state")

    gates = analyze_result(result)["gate_results"]

    assert [gate["gate"] for gate in gates] == ["K0", "K1", "K2"]
    assert gates[-1]["passed"] is False


def test_formal_n8_ci_uses_the_sealed_student_t_constant() -> None:
    base = _development_result()["replicates"][0]
    replicates = []
    expected_values = []
    for index in range(8):
        replicate = copy.deepcopy(base)
        replicate["replicate_id"] = f"formal-{index}"
        delta = 0.20 + index * 0.01
        expected_values.append(delta)
        for row in replicate["predictive_metrics"]:
            if row["task_id"] == TASK_A and row["condition"] == "frozen":
                row["mixture_nll_nats_per_target_dimension"] = 0.10 + delta
        replicates.append(replicate)
    metrics = recompute_aggregate_metrics({"replicates": replicates})
    metric = next(row for row in metrics if row["name"] == "a_nll_improvement_after_a_vs_frozen")
    mean = sum(expected_values) / 8
    standard_error = math.sqrt(sum((value - mean) ** 2 for value in expected_values) / 7) / math.sqrt(8)

    assert metric["mean"] == pytest.approx(mean)
    assert metric["ci_95_lower"] == pytest.approx(mean - T_CRITICAL_N8 * standard_error)
    assert metric["ci_95_upper"] == pytest.approx(mean + T_CRITICAL_N8 * standard_error)
