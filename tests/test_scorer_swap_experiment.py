"""Frozen protocol and decision tests for SS-001."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bench.scorer_swap import experiment


def _synthetic_call_tensors() -> dict[str, np.ndarray]:
    """Build one full-size call with 186 canonical candidates and six repeats."""

    candidate_count = 3 * 64
    sequences = np.zeros((candidate_count, 12, 2), dtype=np.float64)
    sequences[: experiment.EXPECTED_UNIQUE_CANDIDATES, 0, 0] = np.arange(
        1,
        experiment.EXPECTED_UNIQUE_CANDIDATES + 1,
        dtype=np.float64,
    )
    sequences[experiment.EXPECTED_UNIQUE_CANDIDATES :] = sequences[:6]

    generator_scores = -1_000.0 - np.arange(candidate_count, dtype=np.float64)
    generator_scores[0] = 10.0
    generator_scores[64] = 20.0

    exact_scores = np.empty(candidate_count, dtype=np.float64)
    exact_scores[: experiment.EXPECTED_UNIQUE_CANDIDATES] = 1_000.0 - np.arange(
        experiment.EXPECTED_UNIQUE_CANDIDATES, dtype=np.float64
    )
    exact_scores[experiment.EXPECTED_UNIQUE_CANDIDATES :] = exact_scores[:6]

    injected = np.zeros(candidate_count, dtype=bool)
    injected[0] = True

    one_auditor = np.empty(candidate_count, dtype=np.float64)
    one_auditor[: experiment.EXPECTED_UNIQUE_CANDIDATES] = -np.arange(
        experiment.EXPECTED_UNIQUE_CANDIDATES,
        dtype=np.float64,
    )
    one_auditor[experiment.EXPECTED_UNIQUE_CANDIDATES :] = one_auditor[:6]
    auditor_scores = np.repeat(one_auditor[None, :], len(experiment.AUDITOR_SEEDS), axis=0)

    return {
        "sequences": sequences.reshape(3, 64, 12, 2),
        "generator_scores": generator_scores.reshape(3, 64),
        "exact_scores": exact_scores.reshape(3, 64),
        "injected": injected.reshape(3, 64),
        "auditor_scores": auditor_scores.reshape(len(experiment.AUDITOR_SEEDS), 3, 64),
    }


def _fresh_rows(
    *,
    harmful_counts: dict[int, int],
    rejection_counts: dict[int, int] | None = None,
    transfer_counts: dict[int, int] | None = None,
    rescue_counts: dict[int, int] | None = None,
) -> list[dict[str, object]]:
    rejection = rejection_counts or {}
    transfer = transfer_counts or {}
    rescue = rescue_counts or {}
    rows: list[dict[str, object]] = []
    for seed in experiment.GENERATOR_SEEDS:
        harmful = harmful_counts.get(seed, 0)
        for episode in range(4):
            rows.append(
                {
                    "seed": seed,
                    "episode_index": episode,
                    "harmful_signature": episode < harmful,
                    "restart_rejected": episode < rejection.get(seed, 0),
                    "shared_transfer": episode < transfer.get(seed, 0),
                    "exact_rescue": episode < rescue.get(seed, 0),
                }
            )
    return rows


def _calibration_rows(*, passing_seeds: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positive_episodes = {
        8: {0, 1},
        9: {0},
        10: {0, 3},
        11: {0, 2},
        12: {2, 3},
        18: {0, 3},
    }
    for seed in experiment.PARENT_CALIBRATION_SEEDS:
        for episode in range(4):
            exact_improving = episode in positive_episodes.get(seed, set())
            rows.append(
                {
                    "seed": seed,
                    "episode_index": episode,
                    "exact_improving_control": exact_improving,
                    "calibration_call_pass": exact_improving and seed in passing_seeds,
                }
            )
    return rows


def test_protocol_freezes_disjoint_roles_and_descriptive_thresholds() -> None:
    protocol = experiment.protocol_record()
    seeds = cast(dict[str, object], protocol["seeds"])
    thresholds = cast(dict[str, object], protocol["thresholds"])

    assert seeds == {
        "auditors": list(range(20, 32)),
        "fresh_generators": list(range(32, 44)),
        "parent_calibration": list(range(8, 20)),
        "development_excluded": 97,
    }
    assert set(experiment.AUDITOR_SEEDS).isdisjoint(experiment.GENERATOR_SEEDS)
    assert set(experiment.PARENT_CALIBRATION_SEEDS).isdisjoint(experiment.AUDITOR_SEEDS)
    assert set(experiment.PARENT_CALIBRATION_SEEDS).isdisjoint(experiment.GENERATOR_SEEDS)
    assert thresholds["call_vote_floor"] == 9
    assert thresholds["seed_support_floor"] == 10
    assert thresholds["recurrence_start_floor"] == 2
    assert thresholds["within_seed_fraction"] == 0.75
    assert thresholds["calibration_seed_floor"] == 5
    assert "not classical data cross-fitting" in cast(str, protocol["terminology"])


def test_harmful_signature_requires_every_frozen_condition() -> None:
    def supports(
        *,
        b0_injected: bool = True,
        c_iteration: int = 1,
        learned_gain: float = 2.0 * experiment.SCORE_ATOL,
        exact_delta: float = -2.0 * experiment.SCORE_ATOL,
        exact_rank_damage: float = experiment.RANK_DAMAGE_FLOOR,
    ) -> bool:
        return experiment._harmful_signature(
            b0_injected=b0_injected,
            c_iteration=c_iteration,
            learned_gain=learned_gain,
            exact_delta=exact_delta,
            exact_rank_damage=exact_rank_damage,
        )

    assert supports()
    assert not supports(b0_injected=False)
    assert not supports(c_iteration=0)
    assert not supports(learned_gain=experiment.SCORE_ATOL)
    assert not supports(exact_delta=-experiment.SCORE_ATOL)
    assert not supports(exact_rank_damage=experiment.RANK_DAMAGE_FLOOR - 0.5)


def test_call_derivation_uses_only_within_auditor_ranks() -> None:
    tensors = _synthetic_call_tensors()
    baseline = experiment._derive_call(
        source_kind="fresh_primary",
        seed=32,
        episode_index=0,
        **tensors,
    )

    transformed = dict(tensors)
    transformed_scores = np.asarray(tensors["auditor_scores"]).copy()
    for auditor in range(len(experiment.AUDITOR_SEEDS)):
        transformed_scores[auditor] = transformed_scores[auditor] * float(auditor + 1) + 10_000.0 * auditor
    transformed["auditor_scores"] = transformed_scores
    rescaled = experiment._derive_call(
        source_kind="fresh_primary",
        seed=32,
        episode_index=0,
        **transformed,
    )

    assert baseline["unique_candidates"] == experiment.EXPECTED_UNIQUE_CANDIDATES
    assert baseline["b0_flat_index"] == 0
    assert baseline["c_flat_index"] == 64
    assert baseline["c_iteration"] == 1
    assert baseline["harmful_signature"] is True
    assert baseline["auditor_reject_votes"] == 12
    assert baseline["auditor_transfer_votes"] == 0
    assert baseline["x_union_index"] == baseline["b0_union_index"] == 0
    assert baseline["exact_rescue"] is True

    rank_only_fields = (
        "auditor_reject_votes",
        "auditor_transfer_votes",
        "auditor_tie_votes",
        "restart_rejected",
        "shared_transfer",
        "x_union_index",
        "x_sequence_sha256",
        "x_mean_normalized_rank",
        "exact_rescue",
    )
    assert {name: baseline[name] for name in rank_only_fields} == {name: rescaled[name] for name in rank_only_fields}


def test_first_occurrence_deduplication_is_canonical() -> None:
    tensors = _synthetic_call_tensors()
    flat = tensors["sequences"].reshape(-1, 12, 2)
    indices = experiment._first_unique_indices(flat)

    assert len(indices) == experiment.EXPECTED_UNIQUE_CANDIDATES
    assert np.array_equal(
        indices,
        np.arange(experiment.EXPECTED_UNIQUE_CANDIDATES, dtype=np.int64),
    )


def test_recurrence_and_direction_require_ten_of_twelve_fresh_generators() -> None:
    supporting = experiment.GENERATOR_SEEDS[:10]
    harmful = {seed: 2 for seed in supporting}
    rejection = {seed: 2 for seed in supporting}
    rescue = {seed: 2 for seed in supporting}
    summaries = experiment._seed_summaries(
        _fresh_rows(
            harmful_counts=harmful,
            rejection_counts=rejection,
            rescue_counts=rescue,
        ),
        _calibration_rows(passing_seeds=set(experiment.CALIBRATION_POSITIVE_SEEDS)),
    )
    decision = experiment._decision(summaries["fresh"], summaries["calibration"])

    assert decision["classification"] == ("restart_specific_exploitation+cross_seed_rank_rescue")
    assert decision["recurrence_gate"] == {
        "supporting_seeds": 10,
        "required_seeds": 10,
        "passes": True,
    }
    assert decision["fresh_generator_seeds"] == list(range(32, 44))
    assert decision["parent_calibration_seeds_excluded_from_primary"] == list(range(8, 20))

    only_nine = experiment.GENERATOR_SEEDS[:9]
    nine_summaries = experiment._seed_summaries(
        _fresh_rows(
            harmful_counts={seed: 2 for seed in only_nine},
            rejection_counts={seed: 2 for seed in only_nine},
        ),
        _calibration_rows(passing_seeds=set(experiment.CALIBRATION_POSITIVE_SEEDS)),
    )
    nine_decision = experiment._decision(
        nine_summaries["fresh"],
        nine_summaries["calibration"],
    )
    assert nine_decision["classification"] == "parent_signature_not_reproduced"
    assert nine_decision["mechanism"] == "not_interpretable"


def test_within_generator_support_uses_ceiling_of_seventy_five_percent() -> None:
    harmful = {
        experiment.GENERATOR_SEEDS[0]: 4,
        experiment.GENERATOR_SEEDS[1]: 4,
        experiment.GENERATOR_SEEDS[2]: 3,
    }
    rejection = {
        experiment.GENERATOR_SEEDS[0]: 3,
        experiment.GENERATOR_SEEDS[1]: 2,
        experiment.GENERATOR_SEEDS[2]: 2,
    }
    summaries = experiment._seed_summaries(
        _fresh_rows(harmful_counts=harmful, rejection_counts=rejection),
        _calibration_rows(passing_seeds=set(experiment.CALIBRATION_POSITIVE_SEEDS)),
    )["fresh"]

    first, second, third = summaries[:3]
    assert first["within_seed_required_calls"] == 3
    assert first["restart_rejection_support"] is True
    assert second["within_seed_required_calls"] == 3
    assert second["restart_rejection_support"] is False
    assert third["within_seed_required_calls"] == 3
    assert third["restart_rejection_support"] is False


def test_calibration_failure_precedes_otherwise_supported_transfer() -> None:
    harmful = {seed: 2 for seed in experiment.GENERATOR_SEEDS}
    transfer = {seed: 2 for seed in experiment.GENERATOR_SEEDS}
    fresh_rows = _fresh_rows(harmful_counts=harmful, transfer_counts=transfer)
    four_controls = set(experiment.CALIBRATION_POSITIVE_SEEDS[:4])
    summaries = experiment._seed_summaries(
        fresh_rows,
        _calibration_rows(passing_seeds=four_controls),
    )
    failed = experiment._decision(summaries["fresh"], summaries["calibration"])

    assert failed["classification"] == "auditor_direction_control_failed"
    assert failed["mechanism"] == "not_interpretable"
    assert cast(dict[str, object], failed["shared_transfer"])["supporting_seeds"] == 12
    assert cast(dict[str, object], failed["shared_transfer"])["interpretable"] is False

    summaries = experiment._seed_summaries(
        fresh_rows,
        _calibration_rows(passing_seeds=set(experiment.CALIBRATION_POSITIVE_SEEDS)),
    )
    passed = experiment._decision(summaries["fresh"], summaries["calibration"])
    assert passed["classification"] == ("same_data_shared_bias+no_robust_cross_seed_rank_rescue")


def test_array_digest_binds_name_dtype_shape_and_content() -> None:
    base = {"values": np.array([[1.0, 2.0]], dtype=np.float64)}
    same = {"values": np.array([[1.0, 2.0]], dtype=np.float64)}
    content = {"values": np.array([[1.0, 3.0]], dtype=np.float64)}
    shape = {"values": np.array([1.0, 2.0], dtype=np.float64)}
    name = {"other": np.array([[1.0, 2.0]], dtype=np.float64)}

    assert experiment._array_digest(base) == experiment._array_digest(same)
    assert experiment._array_digest(base) != experiment._array_digest(content)
    assert experiment._array_digest(base) != experiment._array_digest(shape)
    assert experiment._array_digest(base) != experiment._array_digest(name)


def test_tensor_package_round_trip_preserves_schema_and_digest(tmp_path: Path) -> None:
    arrays = experiment._empty_arrays()
    for name, array in arrays.items():
        if name not in ("generator_seeds", "auditor_seeds"):
            array.fill(False if array.dtype.kind == "b" else 0)
    experiment._validate_array_shapes(arrays)

    path = tmp_path / experiment.TENSOR_FILE
    experiment._write_arrays(path, arrays)
    loaded = experiment._load_arrays(path)

    experiment._validate_array_shapes(loaded)
    assert experiment._array_digest(loaded) == experiment._array_digest(arrays)
    assert all(np.array_equal(loaded[name], arrays[name]) for name in experiment.ARRAY_NAMES)


def test_prepare_and_formal_start_are_frozen_atomic_and_one_shot(tmp_path: Path) -> None:
    output = tmp_path / "SS-001"
    prepared = experiment.prepare(output)

    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    assert (output / "protocol.json").exists()
    assert (output / "input-manifest.json").exists()
    assert (output / experiment.PARENT_TENSOR_COPY).exists()
    assert (output / experiment.DATASET_COPY).exists()
    with pytest.raises(ValueError, match="results are required"):
        experiment.verify(output, require_results=True)
    record = experiment._mark_formal_started(output)

    assert record == experiment._formal_start_record(output)
    with pytest.raises(FileExistsError):
        experiment._mark_formal_started(output)
    with pytest.raises(ValueError, match="identifier is terminal"):
        experiment.verify(output)


def test_output_safety_rejects_parent_overlaps_and_repository_sources() -> None:
    protected = experiment.REPO_ROOT / experiment.PARENT_OUTPUT
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._assert_safe_output(protected / "nested")
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._assert_safe_output(protected.parent)
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._mark_formal_started(protected / "moved-prepared-package")
    with pytest.raises(ValueError, match="must be below"):
        experiment._assert_safe_output(experiment.REPO_ROOT / "bench/scorer_swap")
    with pytest.raises(ValueError, match="must be below"):
        experiment._assert_safe_output(experiment.REPO_ROOT / "bench/scorer_swap/results")
