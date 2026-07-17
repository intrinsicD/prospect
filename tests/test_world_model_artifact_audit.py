from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bench.world_model_lifecycle.artifact_audit import (
    _GRAPH_RECORD_FIELDS,
    ArtifactAuditError,
    _Audit,
    _audit_bound_source_snapshot,
    _audit_recomputed_analysis,
    _audit_rejected_probe_full_state,
    _decode_sealed_model,
    _derive_seed,
    _independent_recompute_aggregate_metrics,
    _independent_recompute_gate_results,
    _reconstruct_optimizer_sampling,
    _replay_cem_action_trace,
    _validate_domain_graph_structure,
    _validate_formal_conformance_report,
    audit_artifact,
    decode_sampling_manifest,
    recompute_prediction_evidence,
)
from bench.world_model_lifecycle.checkpoint import (
    CANONICAL_COMPONENT_IDS,
    ComponentPayload,
    save_checkpoint,
)
from bench.world_model_lifecycle.learning import WorldModelRuntime
from bench.world_model_lifecycle.model import (
    ProbabilisticEnsemble,
    TransitionBatch,
    WorldModelConfig,
    encode_prediction_evidence,
    evaluate_mixture,
    prepare_candidate,
)
from bench.world_model_lifecycle.planning import (
    CEMController,
    make_learned_model_env,
    make_true_dynamics_env,
    run_pendulum_conformance,
)
from prospect.domain import TimePoint

MAGIC = b"PROSPECT-WM001\0"
TASK_A = "pendulum_normal_torque"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sampling_payload(indices: np.ndarray, transition_ids: list[str]) -> bytes:
    raw_indices = indices.astype("<u4", copy=False).tobytes(order="C")
    identity_payload = json.dumps(
        transition_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    header = _canonical(
        {
            "dtype": "uint32-le",
            "format": "prospect.wm001.bootstrap-manifest.v1",
            "payload_sha256": hashlib.sha256(raw_indices).hexdigest(),
            "shape": list(indices.shape),
            "transition_ids_sha256": hashlib.sha256(identity_payload).hexdigest(),
        }
    )
    return MAGIC + struct.pack(">Q", len(header)) + header + raw_indices


def test_independent_prediction_decoder_recomputes_simple_gaussian_metrics() -> None:
    transition_ids = ("t0", "t1", "t2")
    targets = torch.zeros((3, 4), dtype=torch.float32)
    means = torch.zeros((5, 3, 4), dtype=torch.float32)
    log_variances = torch.zeros_like(means)
    payload = encode_prediction_evidence(
        transition_ids,
        targets,
        means,
        log_variances,
    )

    recomputed = recompute_prediction_evidence(payload)

    assert recomputed.transition_ids == transition_ids
    assert recomputed.mixture_nll_nats_per_target_dimension == pytest.approx(0.5 * math.log(2.0 * math.pi))
    assert recomputed.normalized_rmse == 0.0
    assert recomputed.interval_90_coverage == 1.0

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ArtifactAuditError, match="SHA-256"):
        recompute_prediction_evidence(bytes(corrupted))


def test_independent_sampling_decoder_checks_shape_and_payload_digest() -> None:
    indices = np.arange(5 * 256, dtype="<u4").reshape(1, 5, 256) % 3
    payload = _sampling_payload(indices, ["a", "b", "c"])

    decoded = decode_sampling_manifest(payload)

    assert np.array_equal(decoded.indices, indices)
    assert decoded.payload_sha256 == hashlib.sha256(indices.tobytes(order="C")).hexdigest()

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ArtifactAuditError, match="SHA-256"):
        decode_sampling_manifest(bytes(corrupted))


def test_independent_sampling_replay_matches_balanced_producer_bytes() -> None:
    master_seed = 101
    transition_ids = ("a0", "a1", "b0", "b1")
    transitions = TransitionBatch.from_arrays(
        transition_ids=transition_ids,
        observations=np.tile([1.0, 0.0, 0.0], (4, 1)),
        contexts=[0.0, 0.0, 1.0, 1.0],
        actions=[0.0, 0.0, 0.0, 0.0],
        next_observations=np.tile([1.0, 0.0, 0.0], (4, 1)),
        rewards=[0.0, 0.0, 0.0, 0.0],
    )
    model = ProbabilisticEnsemble(
        WorldModelConfig(hidden_dimensions=(4, 4)),
        initialization_seed=1,
    )
    producer = prepare_candidate(
        model,
        transitions,
        optimizer_steps=2,
        bootstrap_seeds=[_derive_seed("ensemble_bootstrap_b", master_seed, member) for member in range(5)],
        minibatch_order_seed=_derive_seed("minibatch_order_b", master_seed),
        balanced_tasks=True,
    )
    transition_rows = {
        identity: {"task_id": TASK_A if identity.startswith("a") else "pendulum_reversed_torque"}
        for identity in transition_ids
    }

    _, reconstructed = _reconstruct_optimizer_sampling(
        phase="train_b_replay",
        master_seed=master_seed,
        optimizer_steps=2,
        eligible_ids=transition_ids,
        transitions_by_id=transition_rows,
    )

    assert reconstructed == producer.sampling_manifest


def _transition(
    transition_id: str,
    episode_id: str,
    *,
    split: str,
    step: int,
    action: float,
    parameter_sha256: str,
    model_version: str,
    before: list[float],
) -> dict[str, object]:
    theta = math.atan2(before[1], before[0])
    previous_velocity = before[2]
    reward = -(theta * theta + 0.1 * previous_velocity * previous_velocity + 0.001 * action * action)
    angular_velocity = max(
        -8.0,
        min(8.0, previous_velocity + (15.0 * math.sin(theta) + 3.0 * action) * 0.05),
    )
    angle = theta + angular_velocity * 0.05
    after = [math.cos(angle), math.sin(angle), angular_velocity]
    scaled_target = [
        (after[0] - before[0]) / 2.0,
        (after[1] - before[1]) / 2.0,
        (after[2] - before[2]) / 16.0,
        reward / 16.2736044,
    ]
    return {
        "transition_id": transition_id,
        "run_id": f"run:{split}",
        "episode_id": episode_id,
        "task_id": TASK_A,
        "task_context": 0.0,
        "split": split,
        "step_index": step,
        "real_or_imagined": "real",
        "pre_observation_id": f"{transition_id}:pre",
        "decision_id": f"{transition_id}:decision",
        "executed_action_id": f"{transition_id}:action",
        "next_observation_id": f"{transition_id}:next",
        "model_version_at_action": model_version,
        "parameter_sha256_at_action": parameter_sha256,
        "pre_observation": before,
        "intended_action": action,
        "applied_action": action,
        "next_observation": after,
        "reward": reward,
        "terminated": False,
        "truncated": step == 199,
        "scaled_target": scaled_target,
        "target_sha256": hashlib.sha256(struct.pack("<4d", *scaled_target)).hexdigest(),
    }


def _episode(
    episode_id: str,
    *,
    split: str,
    transition_ids: list[str],
    actions: list[float],
    rewards: list[float],
    parameter_sha256: str,
    model_version: str,
    reset_seed: int,
) -> dict[str, object]:
    run_id = f"run:{split}"
    return {
        "episode_id": episode_id,
        "run_id": run_id,
        "task_id": TASK_A,
        "split": split,
        "condition": "collection_random" if split == "collect_a" else "validation_random",
        "checkpoint_id": "cold",
        "reset_seed": reset_seed,
        "process_id": 1,
        "model_version": model_version,
        "parameter_sha256": parameter_sha256,
        "learning_allowed": split == "collect_a",
        "replay_writes_allowed": split == "collect_a",
        "environment_steps": len(transition_ids),
        "return": math.fsum(rewards),
        "started_at_utc": "2026-07-17T00:00:00Z",
        "completed_at_utc": "2026-07-17T00:01:00Z",
        "action_trace_sha256": hashlib.sha256(_canonical({"intended": actions, "applied": actions})).hexdigest(),
        "transition_ids": transition_ids,
    }


def _write_minimal_auditable_artifact(root: Path) -> Path:
    runtime = WorldModelRuntime.initialize(initialization_seed=7)
    owned_state = runtime.owner.snapshot_state()
    parameter_sha256 = runtime.digest
    model_version = runtime.version
    master_seed = 101

    def seed(namespace: str, index: int = 0) -> int:
        return int.from_bytes(
            hashlib.sha256(f"WM-001|1.0.0|{namespace}|{master_seed}|{index}".encode()).digest()[:4],
            "big",
        )

    collect_ids = [f"collect:{index}" for index in range(200)]
    validation_ids = [f"validation:{index}" for index in range(200)]
    collect_seed = seed("collection_action")
    validation_seed = seed("predictive_validation_action")
    collect_reset_seed = seed("collect_a_episode")
    validation_reset_seed = seed("predictive_validation_a_episode")
    collect_rng = np.random.default_rng(collect_seed)
    validation_rng = np.random.default_rng(validation_seed)
    collect_actions = [float(collect_rng.uniform(-2.0, 2.0)) for _ in collect_ids]
    validation_actions = [float(validation_rng.uniform(-2.0, 2.0)) for _ in validation_ids]

    def chained_transitions(
        identities: list[str],
        episode_id: str,
        split: str,
        actions: list[float],
        reset_seed: int,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        reset_rng = np.random.default_rng(reset_seed)
        theta, angular_velocity = reset_rng.uniform(
            low=np.asarray([-math.pi, -1.0]),
            high=np.asarray([math.pi, 1.0]),
        )
        before = np.asarray(
            [math.cos(theta), math.sin(theta), angular_velocity],
            dtype=np.float32,
        ).tolist()
        for index, (identity, action) in enumerate(zip(identities, actions, strict=True)):
            transition = _transition(
                identity,
                episode_id,
                split=split,
                step=index,
                action=action,
                parameter_sha256=parameter_sha256,
                model_version=model_version,
                before=before,
            )
            result.append(transition)
            before = list(transition["next_observation"])
        return result

    collect_transitions = chained_transitions(
        collect_ids,
        "episode:collect",
        "collect_a",
        collect_actions,
        collect_reset_seed,
    )
    validation_transitions = chained_transitions(
        validation_ids,
        "episode:validation",
        "predictive_validation_a",
        validation_actions,
        validation_reset_seed,
    )
    transitions = [*collect_transitions, *validation_transitions]
    episodes = [
        _episode(
            "episode:collect",
            split="collect_a",
            transition_ids=collect_ids,
            actions=collect_actions,
            rewards=[float(row["reward"]) for row in collect_transitions],
            parameter_sha256=parameter_sha256,
            model_version=model_version,
            reset_seed=collect_reset_seed,
        ),
        _episode(
            "episode:validation",
            split="predictive_validation_a",
            transition_ids=validation_ids,
            actions=validation_actions,
            rewards=[float(row["reward"]) for row in validation_transitions],
            parameter_sha256=parameter_sha256,
            model_version=model_version,
            reset_seed=validation_reset_seed,
        ),
    ]

    validation_observations = np.asarray(
        [row["pre_observation"] for row in validation_transitions],
        dtype=np.float32,
    )
    validation_next = np.asarray(
        [row["next_observation"] for row in validation_transitions],
        dtype=np.float32,
    )
    validation_rewards = np.asarray(
        [row["reward"] for row in validation_transitions],
        dtype=np.float32,
    )
    validation_batch = TransitionBatch.from_arrays(
        transition_ids=validation_ids,
        observations=validation_observations,
        contexts=np.zeros((200, 1), dtype=np.float32),
        actions=np.asarray(validation_actions, dtype=np.float32),
        next_observations=validation_next,
        rewards=validation_rewards,
    )
    generated_prediction = evaluate_mixture(runtime.model, validation_batch)
    prediction_payload = generated_prediction.prediction_payload
    prediction_file = root / "predictions.bin"
    prediction_file.write_bytes(prediction_payload)
    prediction = recompute_prediction_evidence(prediction_payload)

    evaluated_checkpoints: list[dict[str, object]] = []
    for condition in (
        "cold",
        "frozen",
        "corrupted",
        "after_a",
        "after_b_replay",
        "after_b_naive",
    ):
        filename = f"{condition}-model-state.bin"
        (root / filename).write_bytes(owned_state.payload)
        evaluated_checkpoints.append(
            {
                "condition": condition,
                "model_version": model_version,
                "parameter_sha256": parameter_sha256,
                "live_state_sha256": owned_state.digest,
                "media_type": "application/vnd.prospect.wm001.owned-model-state",
                "bytes": len(owned_state.payload),
                "sha256": owned_state.digest,
                "filename": filename,
            }
        )

    indices = np.empty((1, 5, 256), dtype="<u4")
    for member in range(5):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed("ensemble_bootstrap_a", member))
        indices[0, member] = (
            torch.randint(
                len(collect_ids),
                (256,),
                generator=generator,
                dtype=torch.long,
            )
            .numpy()
            .astype("<u4", copy=False)
        )
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed("minibatch_order_a"))
    order = torch.randperm(1, generator=order_generator).numpy()
    indices = indices[order].copy()
    manifest_payload = _sampling_payload(indices, collect_ids)
    manifest_file = root / "train-a.bin"
    manifest_file.write_bytes(manifest_payload)
    corrupted_manifest_file = root / "train-a-corrupted.bin"
    corrupted_manifest_file.write_bytes(manifest_payload)
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(seed("corrupted_target_permutation"))
    permutation_payload = (
        torch.randperm(len(collect_ids), generator=permutation_generator)
        .numpy()
        .astype("<u4", copy=False)
        .tobytes(order="C")
    )
    permutation_file = root / "train-a-corrupted-permutation.bin"
    permutation_file.write_bytes(permutation_payload)
    consumed = hashlib.sha256()
    encoded_ids = [identity.encode() + b"\n" for identity in collect_ids]
    for index in indices.reshape(-1):
        consumed.update(encoded_ids[int(index)])

    checkpoint_path = root / "checkpoint.zip"
    components = {
        component_id: ComponentPayload(
            component_id=component_id,
            logical_version=f"version:{component_id}",
            payload=f"payload:{component_id}".encode(),
        )
        for component_id in CANONICAL_COMPONENT_IDS
    }
    checkpoint = save_checkpoint(
        checkpoint_path,
        checkpoint_id="after_b_replay",
        agent_id="prospect-wm001-agent",
        created_at=TimePoint(1),
        components=components,
    )
    checkpoint_payload = checkpoint_path.read_bytes()
    update = {
        "receipt_id": "receipt:train-a",
        "phase": "train_a",
        "status": "committed",
        "predecessor_parameter_sha256": parameter_sha256,
        "candidate_parameter_sha256": parameter_sha256,
        "committed_parameter_sha256": parameter_sha256,
        "predecessor_model_version": model_version,
        "committed_model_version": model_version,
        "eligible_splits": ["collect_a"],
        "eligible_transition_count": len(collect_ids),
        "eligible_transition_ids": collect_ids,
        "consumed_sample_count": int(indices.size),
        "consumed_multiset_sha256": consumed.hexdigest(),
        "sampling_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "target_permutation_sha256": None,
        "target_permutation_file": None,
        "optimizer_steps": 1,
        "live_state_before_sha256": owned_state.digest,
        "live_state_after_sha256": owned_state.digest,
    }
    corrupted_update = {
        **update,
        "receipt_id": "receipt:train-a-corrupted",
        "phase": "train_a_corrupted",
        "target_permutation_sha256": hashlib.sha256(permutation_payload).hexdigest(),
        "target_permutation_file": {
            "media_type": "application/octet-stream",
            "bytes": len(permutation_payload),
            "sha256": hashlib.sha256(permutation_payload).hexdigest(),
            "filename": permutation_file.name,
        },
    }

    def policy_run(
        *,
        namespace: str,
        controller_seed: int,
        split: str,
        episode_id: str,
        actions: list[float],
        reset_seed: int,
    ) -> dict[str, object]:
        rng = np.random.default_rng(controller_seed)
        start = hashlib.sha256(_canonical(rng.bit_generator.state)).hexdigest()
        reproduced = [float(rng.uniform(-2.0, 2.0)) for _ in actions]
        assert reproduced == actions
        end = hashlib.sha256(_canonical(rng.bit_generator.state)).hexdigest()
        trace = {
            "episode_ids": [episode_id],
            "intended": actions,
            "applied": actions,
        }
        return {
            "run_id": f"run:{split}",
            "task_id": TASK_A,
            "split": split,
            "condition": "collection_random" if split == "collect_a" else "validation_random",
            "checkpoint_id": "cold",
            "controller_kind": "uniform_random",
            "controller_version": "wm001-uniform-random-v1",
            "seed_namespace": namespace,
            "seed_index": 0,
            "seed": controller_seed,
            "reset_seeds": [reset_seed],
            "episode_ids": [episode_id],
            "rng_start_sha256": start,
            "rng_end_sha256": end,
            "action_count": len(actions),
            "action_trace_sha256": hashlib.sha256(_canonical(trace)).hexdigest(),
            "planner_budget": None,
        }

    seed_counts = {
        "model_initialization": 1,
        "torch_runtime": 1,
        "collection_action": 2,
        "predictive_validation_action": 2,
        "random_policy_action": 2,
        "ensemble_bootstrap_a": 5,
        "ensemble_bootstrap_b": 5,
        "minibatch_order_a": 1,
        "minibatch_order_b": 1,
        "corrupted_target_permutation": 1,
        "collect_a_episode": 8,
        "predictive_validation_a_episode": 8,
        "behavior_evaluation_a_episode": 32,
        "collect_b_episode": 8,
        "predictive_validation_b_episode": 8,
        "behavior_evaluation_b_episode": 32,
        "planner": 1,
    }
    replicate = {
        "replicate_id": "dev-101",
        "master_seed": master_seed,
        "derived_seeds": [
            {
                "namespace": namespace,
                "values": [seed(namespace, index) for index in range(count)],
            }
            for namespace, count in seed_counts.items()
        ],
        "episodes": episodes,
        "transitions": transitions,
        "updates": [update, corrupted_update],
        "optimizer_batch_manifests": [
            {
                "phase": "train_a",
                "media_type": "application/octet-stream",
                "bytes": len(manifest_payload),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "filename": manifest_file.name,
            },
            {
                "phase": "train_a_corrupted",
                "media_type": "application/octet-stream",
                "bytes": len(manifest_payload),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "filename": corrupted_manifest_file.name,
            },
        ],
        "predictive_metrics": [
            {
                "task_id": TASK_A,
                "condition": "cold",
                "checkpoint_id": "cold",
                "model_version": model_version,
                "parameter_sha256": parameter_sha256,
                "live_state_sha256": owned_state.digest,
                "split": "predictive_validation_a",
                "transition_count": len(validation_ids),
                "mixture_nll_nats_per_target_dimension": (prediction.mixture_nll_nats_per_target_dimension),
                "normalized_rmse": prediction.normalized_rmse,
                "interval_90_coverage": prediction.interval_90_coverage,
                "prediction_rows_sha256": hashlib.sha256(prediction_payload).hexdigest(),
                "prediction_evidence_file": prediction_file.name,
                "prediction_evidence_bytes": len(prediction_payload),
            }
        ],
        "policy_runs": [
            policy_run(
                namespace="collection_action",
                controller_seed=collect_seed,
                split="collect_a",
                episode_id="episode:collect",
                actions=collect_actions,
                reset_seed=collect_reset_seed,
            ),
            policy_run(
                namespace="predictive_validation_action",
                controller_seed=validation_seed,
                split="predictive_validation_a",
                episode_id="episode:validation",
                actions=validation_actions,
                reset_seed=validation_reset_seed,
            ),
        ],
        "evaluated_checkpoints": evaluated_checkpoints,
        "checkpoint_components": list(checkpoint.component_rows()),
        "checkpoint_archive": {
            "media_type": "application/zip",
            "bytes": len(checkpoint_payload),
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "filename": checkpoint_path.name,
        },
        "restart_parity": {
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
        },
    }
    result: dict[str, Any] = {
        "schema": "prospect.world-model-lifecycle.raw-result.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.1.1",
        "protocol_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "bench" / "world_model_lifecycle" / "protocol.json").read_bytes()
        ).hexdigest(),
        "lane": "development",
        "execution": {"device": "cpu"},
        "replicates": [replicate],
    }
    result_path = root / "result.json"
    result_path.write_bytes(_canonical(result) + b"\n")
    return result_path


def test_artifact_audit_recomputes_current_evidence_and_detects_metric_tampering(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is True
    assert report["complete_for_claim"] is False
    assert report["passed"] is True
    assert {gap["code"] for gap in report["coverage_gaps"]} == {
        "cem_action_trace_replay_absent",
        "checkpoint_domain_graph_semantics_unverified",
        "formal_execution_not_present",
        "producer_custody_not_verified",
        "rejected_probe_full_state_unavailable",
    }

    result = json.loads(result_path.read_text())
    result["replicates"][0]["predictive_metrics"][0]["mixture_nll_nats_per_target_dimension"] += 0.5
    result_path.write_bytes(_canonical(result) + b"\n")

    tampered = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert tampered["integrity_passed"] is False
    assert "prediction_nll_mismatch" in {finding["code"] for finding in tampered["findings"]}


def _add_independent_analysis_rows(result_path: Path) -> None:
    result = json.loads(result_path.read_text())
    metrics, replicate_values = _independent_recompute_aggregate_metrics(result)
    result["aggregate_metrics"] = metrics
    result["gate_results"] = _independent_recompute_gate_results(
        metrics,
        replicate_values,
    )
    result_path.write_bytes(_canonical(result) + b"\n")


def _two_replicate_analysis_result() -> dict[str, Any]:
    replicate_returns = (217.45840023805044, 521.5666528679101)
    replicates: list[dict[str, Any]] = []
    for index, after_a_return in enumerate(replicate_returns):
        behavior_returns = {
            "cold": 0.0,
            "after_a": after_a_return,
            "frozen": 0.0,
            "corrupted": 0.0,
            "after_b_replay": after_a_return,
            "after_b_naive": 0.0,
            "random": 0.0,
            "oracle": 1000.0,
        }
        replicates.append(
            {
                "replicate_id": f"synthetic-{index}",
                "episodes": [
                    {
                        "task_id": TASK_A,
                        "split": "behavior_evaluation_a",
                        "condition": condition,
                        "return": value,
                    }
                    for condition, value in behavior_returns.items()
                ],
                "predictive_metrics": [
                    {
                        "task_id": TASK_A,
                        "split": "predictive_validation_a",
                        "condition": condition,
                        "checkpoint_id": condition,
                        "mixture_nll_nats_per_target_dimension": nll,
                        "interval_90_coverage": 0.9,
                    }
                    for condition, nll in (
                        ("after_a", 0.0),
                        ("frozen", 1.0),
                        ("corrupted", 1.0),
                    )
                ],
                "updates": [],
                "checkpoint_components": [],
            }
        )
    return {"replicates": replicates}


def test_two_replicate_ci_preserves_sealed_operation_order() -> None:
    result = _two_replicate_analysis_result()
    metrics, _ = _independent_recompute_aggregate_metrics(result)
    metric = next(row for row in metrics if row["name"] == "a_return_improvement_after_a_vs_cold")
    values = [217.45840023805044, 521.5666528679101]
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(2)
    margin = 12.706204736 * standard_error

    assert metric["replicate_values"] == values
    assert metric["mean"] == mean
    assert metric["ci_95_lower"] == mean - margin
    assert metric["ci_95_upper"] == mean + margin
    assert metric["ci_95_lower"] == -1562.5183333581233


def test_two_replicate_rehashed_aggregate_and_gate_tampering_is_rejected() -> None:
    result = _two_replicate_analysis_result()
    metrics, replicate_values = _independent_recompute_aggregate_metrics(result)
    result["aggregate_metrics"] = metrics
    result["gate_results"] = _independent_recompute_gate_results(
        metrics,
        replicate_values,
    )
    baseline = _Audit()
    _audit_recomputed_analysis(baseline, result)
    assert baseline.failed_checks == 0
    assert baseline.passed_checks == 2

    tampered = json.loads(json.dumps(result))
    metric = next(row for row in tampered["aggregate_metrics"] if row["name"] == "a_return_improvement_after_a_vs_cold")
    metric["ci_95_lower"] = math.nextafter(
        metric["ci_95_lower"],
        math.inf,
    )
    fabricated_evidence_sha256 = hashlib.sha256(_canonical(metric)).hexdigest()
    k4 = next(row for row in tampered["gate_results"] if row["gate"] == "K4")
    for check in k4["checks"][:2]:
        check["raw_evidence_sha256"] = fabricated_evidence_sha256
    k4["checks"][1]["observed"] = metric["ci_95_lower"]

    audit = _Audit()
    _audit_recomputed_analysis(audit, tampered)

    assert audit.failed_checks == 2
    assert {finding["code"] for finding in audit.findings} == {
        "aggregate_metrics_recomputation_mismatch",
        "gate_results_recomputation_mismatch",
    }


def test_artifact_audit_rejects_fabricated_aggregate_values(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    _add_independent_analysis_rows(result_path)
    baseline = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert baseline["integrity_passed"] is True

    result = json.loads(result_path.read_text())
    result["aggregate_metrics"][0]["mean"] += 1.0
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert report["integrity_passed"] is False
    assert "aggregate_metrics_recomputation_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_fabricated_gate_values(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    _add_independent_analysis_rows(result_path)
    baseline = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert baseline["integrity_passed"] is True

    result = json.loads(result_path.read_text())
    result["gate_results"][0]["checks"][0]["observed"] = 1
    result["gate_results"][0]["checks"][0]["raw_evidence_sha256"] = hashlib.sha256(
        _canonical(["fabricated"])
    ).hexdigest()
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert report["integrity_passed"] is False
    assert "gate_results_recomputation_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_reopens_finalized_producer_manifest(tmp_path: Path) -> None:
    _write_minimal_auditable_artifact(tmp_path)
    files = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file() and path.name != "producer-manifest.json"
    ]
    manifest = {
        "schema": "prospect.wm001.producer-manifest.v1",
        "experiment_id": "WM-001",
        "lane": "development",
        "status": "completed",
        "started_at_utc": "2026-07-17T00:00:00Z",
        "completed_at_utc": "2026-07-17T01:00:00Z",
        "error": None,
        "manifest_excludes": ["producer-manifest.json"],
        "file_count": len(files),
        "files": files,
    }
    (tmp_path / "producer-manifest.json").write_bytes(_canonical(manifest) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
    )

    assert report["integrity_passed"] is True
    assert report["custody"]["producer_manifest_checked"] is True
    assert report["custody"]["producer_manifest_status"] == "completed"


def test_artifact_audit_rejects_reset_state_not_generated_by_episode_seed(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    first = result["replicates"][0]["transitions"][0]
    first["pre_observation"][0] = float(first["pre_observation"][0]) + 0.01
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "episode_reset_observation_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_reordered_episode_seed_schedule(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    replicate = result["replicates"][0]
    collect_seeds = next(row["values"] for row in replicate["derived_seeds"] if row["namespace"] == "collect_a_episode")
    collect_episode = next(row for row in replicate["episodes"] if row["split"] == "collect_a")
    collect_policy = next(row for row in replicate["policy_runs"] if row["split"] == "collect_a")
    collect_episode["reset_seed"] = collect_seeds[1]
    collect_policy["reset_seeds"] = [collect_seeds[1]]
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "policy_reset_seed_schedule_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_rehashed_but_wrong_optimizer_rng_trace(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    replicate = result["replicates"][0]
    update = next(row for row in replicate["updates"] if row["phase"] == "train_a")
    manifest = next(row for row in replicate["optimizer_batch_manifests"] if row["phase"] == "train_a")
    path = tmp_path / manifest["filename"]
    decoded = decode_sampling_manifest(path.read_bytes())
    altered = decoded.indices.copy()
    altered[0, 0, 0] = (int(altered[0, 0, 0]) + 1) % int(update["eligible_transition_count"])
    payload = _sampling_payload(altered, update["eligible_transition_ids"])
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest["bytes"] = len(payload)
    manifest["sha256"] = digest
    update["sampling_manifest_sha256"] = digest
    consumed = hashlib.sha256()
    encoded_ids = [identity.encode() + b"\n" for identity in update["eligible_transition_ids"]]
    for sample_index in altered.reshape(-1):
        consumed.update(encoded_ids[int(sample_index)])
    update["consumed_multiset_sha256"] = consumed.hexdigest()
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "optimizer_sampling_seed_replay_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_rehashed_but_wrong_corruption_permutation(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    update = next(row for row in result["replicates"][0]["updates"] if row["phase"] == "train_a_corrupted")
    reference = update["target_permutation_file"]
    path = tmp_path / reference["filename"]
    permutation = np.frombuffer(path.read_bytes(), dtype="<u4").copy()
    permutation[[0, 1]] = permutation[[1, 0]]
    payload = permutation.astype("<u4", copy=False).tobytes(order="C")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    reference["bytes"] = len(payload)
    reference["sha256"] = digest
    update["target_permutation_sha256"] = digest
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "target_permutation_seed_replay_mismatch" in {finding["code"] for finding in report["findings"]}


@pytest.mark.parametrize("oracle", [False, True], ids=["learned", "oracle"])
def test_standalone_cem_replay_matches_producer_actions_exactly(
    oracle: bool,
) -> None:
    runtime = WorldModelRuntime.initialize(initialization_seed=7)
    states = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.9799, -0.3]],
            [[0.99, 0.1, 0.2], [0.3, 0.9539, 0.4]],
        ],
        dtype=np.float32,
    )
    seed = 99173
    planner = CEMController(
        make_true_dynamics_env() if oracle else make_learned_model_env(runtime.model),
        seed=seed,
    )
    start_digest = planner.rng_digest
    expected: list[np.ndarray] = []
    for step_states in states:
        contextual = np.concatenate(
            (step_states, np.zeros((len(step_states), 1), dtype=np.float32)),
            axis=1,
        )
        expected.append(planner.act(contextual).detach().cpu().numpy())
    expected_actions = np.stack(expected)

    replayed, replay_start, replay_end = _replay_cem_action_trace(
        observed_states=states,
        context=0.0,
        seed=seed,
        device="cpu",
        model_tensors=(None if oracle else _decode_sealed_model(runtime.model_bytes)),
    )

    assert np.array_equal(replayed, expected_actions)
    assert replay_start == start_digest
    assert replay_end == planner.rng_digest
    tampered = expected_actions.copy()
    tampered[0, 0, 0] = np.nextafter(
        tampered[0, 0, 0],
        np.float32(math.inf),
    )
    assert not np.array_equal(replayed, tampered)


def _rejected_probe_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    runtime = WorldModelRuntime.initialize(initialization_seed=17)

    def node(index: int, type_name: str) -> dict[str, object]:
        return {
            "ref": f"n{index:08d}",
            "type": type_name,
            "fields": {field: None for field in _GRAPH_RECORD_FIELDS[type_name]},
        }

    graph = {
        "schema": "prospect.wm001.domain-graph.v1",
        "roots": {
            "agent_snapshot": {"$ref": "n00000000"},
            "source_events": {"$tuple": [{"$ref": "n00000001"}]},
            "source_transitions": {"$tuple": [{"$ref": "n00000002"}]},
            "source_updates": {"$tuple": [{"$ref": "n00000003"}]},
            "probe_transitions": {"$tuple": []},
            "probe_updates": {"$tuple": []},
        },
        "nodes": [
            node(0, "AgentSnapshot"),
            node(1, "ExperienceEvent"),
            node(2, "EpistemicTransition"),
            node(3, "UpdateReceipt"),
        ],
        "observation_sequences": [],
    }
    sampling = _sampling_payload(
        np.zeros((1, 5, 256), dtype="<u4"),
        ["t-1"],
    )
    empty_digest = hashlib.sha256(b"fixture").hexdigest()
    train_a_update = {
        "eligible_transition_ids": ["t-1"],
        "consumed_multiset_sha256": empty_digest,
        "predecessor_parameter_sha256": empty_digest,
        "committed_parameter_sha256": runtime.digest,
        "live_state_before_sha256": empty_digest,
        "live_state_after_sha256": runtime.live_state_digest,
        "optimizer_steps": 1,
        "sampling_manifest_sha256": hashlib.sha256(sampling).hexdigest(),
    }
    encoded_marker = base64.b64encode(b"x").decode()
    state = {
        "schema": "prospect.wm001.rejected-probe-full-state.v1",
        "captured_at": ["interaction", 9],
        "model_state": {
            "version": runtime.version,
            "digest": runtime.live_state_digest,
            "payload_base64": base64.b64encode(runtime.owner.snapshot_state().payload).decode(),
        },
        "domain_graph": graph,
        "source_replay_rows": [],
        "probe_replay_rows": [],
        "source_identity_base64": encoded_marker,
        "probe_identity_base64": encoded_marker,
        "collection_rng_state": {"state": 1},
        "process_rng": {
            "python_base64": encoded_marker,
            "numpy_base64": encoded_marker,
            "torch_cpu_base64": encoded_marker,
            "torch_accelerator_base64": encoded_marker,
        },
        "retained_learning_evidence": {
            "phase": "train_a",
            "consumed_transition_ids": ["t-1"],
            "consumed_multiset_sha256": empty_digest,
            "predecessor_parameter_sha256": empty_digest,
            "candidate_parameter_sha256": runtime.digest,
            "predecessor_live_state_sha256": empty_digest,
            "candidate_live_state_sha256": runtime.live_state_digest,
            "optimizer_steps": 1,
            "sampling_manifest_base64": base64.b64encode(sampling).decode(),
            "sampling_manifest_sha256": hashlib.sha256(sampling).hexdigest(),
            "sampled_id_counts": [["t-1", 1280]],
            "target_permutation_sha256": None,
            "target_permutation_base64": None,
            "loss_history": [0.5],
        },
    }
    payload = _canonical(state)
    before_path = root / "probe-before.json"
    after_path = root / "probe-after.json"
    before_path.write_bytes(payload)
    after_path.write_bytes(payload)

    def reference(path: Path) -> dict[str, object]:
        return {
            "media_type": ("application/vnd.prospect.wm001.rejected-probe-state+json"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": path.name,
        }

    rejected_update = {
        "live_state_before_sha256": runtime.live_state_digest,
        "predecessor_model_version": runtime.version,
        "predecessor_parameter_sha256": runtime.digest,
        "full_state_before_sha256": hashlib.sha256(payload).hexdigest(),
        "full_state_after_sha256": hashlib.sha256(payload).hexdigest(),
        "full_state_before_file": reference(before_path),
        "full_state_after_file": reference(after_path),
    }
    return rejected_update, train_a_update


def test_rejected_probe_audit_reopens_and_compares_complete_state(
    tmp_path: Path,
) -> None:
    rejected_update, train_a_update = _rejected_probe_fixture(tmp_path)
    audit = _Audit()

    _audit_rejected_probe_full_state(
        audit,
        tmp_path,
        rejected_update,
        replicate_id="dev-101",
        train_a_update=train_a_update,
    )

    assert audit.failed_checks == 0
    assert audit.coverage_gaps == []

    after_reference = rejected_update["full_state_after_file"]
    after_path = tmp_path / after_reference["filename"]
    after_payload = after_path.read_bytes() + b" "
    after_path.write_bytes(after_payload)
    after_digest = hashlib.sha256(after_payload).hexdigest()
    after_reference["bytes"] = len(after_payload)
    after_reference["sha256"] = after_digest
    rejected_update["full_state_after_sha256"] = after_digest
    tampered = _Audit()

    _audit_rejected_probe_full_state(
        tampered,
        tmp_path,
        rejected_update,
        replicate_id="dev-101",
        train_a_update=train_a_update,
    )

    assert tampered.failed_checks > 0
    assert {finding["code"] for finding in tampered.findings} >= {
        "rejected_probe_full_state_changed",
        "rejected_probe_full_state_invalid",
    }


def test_checkpoint_domain_graph_audit_rejects_unknown_tags_and_external_refs() -> None:
    receipt_fields: dict[str, object] = {field: None for field in _GRAPH_RECORD_FIELDS["UpdateReceipt"]}
    receipt_fields["transitions"] = {
        "$tuple": [
            {"$external": "transition:t-1"},
        ]
    }
    graph = {
        "schema": "prospect.wm001.domain-graph.v1",
        "roots": {
            "receipts": {
                "$tuple": [
                    {"$ref": "n00000000"},
                ]
            }
        },
        "nodes": [
            {
                "ref": "n00000000",
                "type": "UpdateReceipt",
                "fields": receipt_fields,
            }
        ],
        "observation_sequences": [],
    }

    _validate_domain_graph_structure(graph, component_id="update_receipts")

    unknown_tag = json.loads(json.dumps(graph))
    unknown_tag["nodes"][0]["fields"]["transitions"] = {"$pickle": "payload"}
    with pytest.raises(ArtifactAuditError, match="unknown encoded tag"):
        _validate_domain_graph_structure(
            unknown_tag,
            component_id="update_receipts",
        )

    wrong_namespace = json.loads(json.dumps(graph))
    wrong_namespace["nodes"][0]["fields"]["transitions"]["$tuple"][0] = {"$external": "belief:b-1"}
    with pytest.raises(ArtifactAuditError, match="external ref"):
        _validate_domain_graph_structure(
            wrong_namespace,
            component_id="update_receipts",
        )


def _source_row(relative: str, payload: bytes) -> dict[str, object]:
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_formal_source_snapshot_audit_requires_exact_bound_file_set(
    tmp_path: Path,
) -> None:
    makefile_payload = b"all:\n\t@true\n"
    module_payload = b'"""fixture"""\n'
    (tmp_path / "source" / "Makefile").parent.mkdir(parents=True)
    (tmp_path / "source" / "Makefile").write_bytes(makefile_payload)
    module = tmp_path / "source" / "bench" / "fixture.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(module_payload)
    source = {
        "implementation_files": [
            _source_row("Makefile", makefile_payload),
            _source_row("bench/fixture.py", module_payload),
        ]
    }
    audit = _Audit()

    _audit_bound_source_snapshot(audit, tmp_path, source)

    assert audit.passed_checks == 1
    assert audit.failed_checks == 0

    (tmp_path / "source" / "unexpected.py").write_bytes(b"pass\n")
    with pytest.raises(ArtifactAuditError, match="file set differs"):
        _audit_bound_source_snapshot(_Audit(), tmp_path, source)


def test_formal_source_snapshot_audit_rejects_tamper_and_manifest_omission(
    tmp_path: Path,
) -> None:
    makefile_payload = b"all:\n\t@true\n"
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "Makefile").write_bytes(makefile_payload)
    source = {
        "implementation_files": [
            _source_row("Makefile", makefile_payload),
        ]
    }
    (source_root / "Makefile").write_bytes(b"tampered\n")
    with pytest.raises(ArtifactAuditError, match="size/digest changed"):
        _audit_bound_source_snapshot(_Audit(), tmp_path, source)

    (source_root / "Makefile").write_bytes(makefile_payload)
    source["implementation_files"] = []
    with pytest.raises(ArtifactAuditError, match="no implementation manifest"):
        _audit_bound_source_snapshot(_Audit(), tmp_path, source)


def test_artifact_auditor_independently_rejects_rehashed_conformance_bypass() -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )

    _validate_formal_conformance_report(report)

    report["cases"] = 2
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ArtifactAuditError, match="exactly 512 cases"):
        _validate_formal_conformance_report(report)
