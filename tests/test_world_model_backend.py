from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from bench.world_model_lifecycle.model import (
    ModelValidationError,
    OptimizerConfig,
    ProbabilisticEnsemble,
    TransitionBatch,
    WorldModelConfig,
    evaluate_mixture,
    optimizer_from_bytes,
    optimizer_to_bytes,
    prepare_candidate,
)


def _tiny_config() -> WorldModelConfig:
    return WorldModelConfig(ensemble_members=2, hidden_dimensions=(16, 16))


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=0.003,
        batch_size=16,
        gradient_clip_l2=10.0,
    )


def _transitions(rows: int = 64) -> TransitionBatch:
    generator = np.random.default_rng(20260717)
    theta = generator.uniform(-np.pi, np.pi, size=rows)
    velocity = generator.uniform(-1.0, 1.0, size=rows)
    observations = np.column_stack((np.cos(theta), np.sin(theta), velocity)).astype(np.float32)
    contexts = (np.arange(rows) % 2).astype(np.float32)
    actions = generator.uniform(-2.0, 2.0, size=rows).astype(np.float32)
    signed_action = actions * np.where(contexts == 0.0, 1.0, -1.0)
    next_theta = theta + 0.03 * velocity + 0.02 * signed_action
    next_velocity = np.clip(0.97 * velocity + 0.15 * signed_action, -8.0, 8.0)
    next_observations = np.column_stack((np.cos(next_theta), np.sin(next_theta), next_velocity)).astype(np.float32)
    rewards = (-(theta**2) - 0.1 * velocity**2 - 0.001 * actions**2).astype(np.float32)
    return TransitionBatch.from_arrays(
        transition_ids=[f"transition-{index:04d}" for index in range(rows)],
        observations=observations,
        contexts=contexts,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
    )


def test_model_snapshot_is_canonical_safe_and_versioned_by_sha256() -> None:
    model = ProbabilisticEnsemble(_tiny_config(), initialization_seed=71)
    payload = model.to_bytes()
    restored = ProbabilisticEnsemble.from_bytes(payload)

    assert restored.to_bytes() == payload
    assert restored.parameter_sha256 == model.parameter_sha256
    assert restored.version == f"wm001-sha256:{model.parameter_sha256}"

    tampered = bytearray(payload)
    tampered[-1] ^= 1
    with pytest.raises(ModelValidationError, match="checksum"):
        ProbabilisticEnsemble.from_bytes(bytes(tampered))


def test_candidate_training_is_deterministic_non_mutating_and_balanced() -> None:
    transitions = _transitions()
    source = ProbabilisticEnsemble(_tiny_config(), initialization_seed=991)
    source_payload = source.to_bytes()
    config = _optimizer_config()
    arguments = {
        "optimizer_steps": 20,
        "bootstrap_seeds": (101, 211),
        "optimizer_config": config,
        "balanced_tasks": True,
        "device": "cpu",
    }

    first = prepare_candidate(source, transitions, **arguments)
    second = prepare_candidate(source, transitions, **arguments)

    assert source.to_bytes() == source_payload
    assert first.candidate_parameter_sha256 != first.predecessor_parameter_sha256
    assert first.model.to_bytes() == second.model.to_bytes()
    assert first.optimizer_bytes == second.optimizer_bytes
    assert first.sampling_manifest == second.sampling_manifest
    assert first.loss_history == second.loss_history
    counts = dict(first.sampled_id_counts)
    task_a_samples = sum(counts[identity] for identity in transitions.transition_ids[::2])
    task_b_samples = sum(counts[identity] for identity in transitions.transition_ids[1::2])
    assert task_a_samples == task_b_samples == 20 * 2 * config.batch_size // 2

    restored_optimizer = optimizer_from_bytes(
        first.model,
        first.optimizer_bytes,
        expected_config=config,
    )
    assert optimizer_to_bytes(first.model, restored_optimizer, config=config) == first.optimizer_bytes
    with pytest.raises(ModelValidationError, match="different model"):
        optimizer_from_bytes(source, first.optimizer_bytes, expected_config=config)


def test_corrupted_joint_targets_and_mixture_metrics_are_deterministic() -> None:
    transitions = _transitions()
    source = ProbabilisticEnsemble(_tiny_config(), initialization_seed=19)
    arguments = {
        "optimizer_steps": 8,
        "bootstrap_seeds": (307, 401),
        "optimizer_config": _optimizer_config(),
        "training_mode": "joint_target_permuted",
        "target_permutation_seed": 503,
        "device": "cpu",
    }
    first = prepare_candidate(source, transitions, **arguments)
    second = prepare_candidate(source, transitions, **arguments)
    different_permutation = prepare_candidate(
        source,
        transitions,
        **{**arguments, "target_permutation_seed": 509},
    )

    assert first.target_permutation_sha256 == second.target_permutation_sha256
    assert first.model.to_bytes() == second.model.to_bytes()
    assert first.target_permutation_sha256 != different_permutation.target_permutation_sha256
    assert first.model.to_bytes() != different_permutation.model.to_bytes()

    metrics_small_batches = evaluate_mixture(first.model, transitions, batch_size=7)
    metrics_repeat = evaluate_mixture(first.model, transitions, batch_size=7)
    metrics_one_batch = evaluate_mixture(first.model, transitions, batch_size=len(transitions))
    assert metrics_small_batches == metrics_repeat
    assert metrics_small_batches.mixture_nll_nats_per_target_dimension == pytest.approx(
        metrics_one_batch.mixture_nll_nats_per_target_dimension,
        abs=1e-7,
    )
    assert metrics_small_batches.normalized_rmse == pytest.approx(metrics_one_batch.normalized_rmse, abs=1e-7)
    assert metrics_small_batches.interval_90_coverage == metrics_one_batch.interval_90_coverage
    assert math.isfinite(metrics_one_batch.mixture_nll_nats_per_target_dimension)
    assert math.isfinite(metrics_one_batch.normalized_rmse)
    assert 0.0 <= metrics_one_batch.interval_90_coverage <= 1.0
    assert metrics_one_batch.transition_count == len(transitions)
