"""Frozen protocol and decision tests for CL-001."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bench.candidate_landscape import experiment


def _call_rows(
    *,
    supporting_confirmatory: set[int],
    supporting_episodes: set[int] | None = None,
) -> list[dict[str, object]]:
    episode_support = supporting_episodes if supporting_episodes is not None else set(range(4))
    rows: list[dict[str, object]] = []
    for seed in experiment.ALL_SEEDS:
        seed_supports = seed in supporting_confirmatory or seed in experiment.REPLAY_SEEDS
        phase = "replay" if seed in experiment.REPLAY_SEEDS else "confirmatory"
        for episode in range(4):
            supports = seed_supports and episode in episode_support
            for step in experiment.AUDIT_STEPS:
                learned_rank = 1.0
                if step == 1 and supports:
                    learned_rank = 5.0
                elif step == 2 and supports:
                    learned_rank = 10.0
                rows.append(
                    {
                        "seed": seed,
                        "phase": phase,
                        "episode_index": episode,
                        "step": step,
                        "round0_best_injected": supports and step == 0,
                        "selected_first_iteration": 1 if supports and step == 0 else 0,
                        "refinement_learned_gain": (1.0 if supports and step == 0 else 0.0),
                        "refinement_exact_delta": (-1.0 if supports and step == 0 else 0.0),
                        "refinement_exact_rank_damage": (9.0 if supports and step == 0 else 0.0),
                        "supports_within_call_exploitation": supports and step == 0,
                        "cold_reference_exact_rank": 1.0,
                        "cold_reference_learned_rank": learned_rank,
                        "cold_reference_rank_residual": learned_rank - 1.0,
                        "cold_learned_exact_spearman": 0.5,
                    }
                )
    return rows


def test_confirmatory_decision_uses_ten_of_twelve_and_excludes_replay_seeds() -> None:
    supporting = set(experiment.CONFIRMATORY_SEEDS[:10])
    summaries = experiment._seed_summaries(_call_rows(supporting_confirmatory=supporting))
    decision = experiment._decision(summaries)

    assert decision["classification"] == ("within_call_exploitation_and_statewise_scorer_shift")
    exploitation = cast(dict[str, object], decision["within_call_exploitation"])
    shift = cast(dict[str, object], decision["visited_state_scorer_shift"])
    assert exploitation == {
        "supporting_seeds": 10,
        "required_seeds": 10,
        "passes": True,
    }
    assert shift == {
        "supporting_seeds": 10,
        "required_seeds": 10,
        "passes": True,
    }
    assert decision["replay_seeds_excluded"] == list(experiment.REPLAY_SEEDS)


def test_nine_of_twelve_is_not_confirmatory_support() -> None:
    supporting = set(experiment.CONFIRMATORY_SEEDS[:9])
    decision = experiment._decision(experiment._seed_summaries(_call_rows(supporting_confirmatory=supporting)))
    assert decision["classification"] == "neither_mechanism_supported"


def test_seed_requires_three_jointly_supporting_starts() -> None:
    supporting = set(experiment.CONFIRMATORY_SEEDS)
    three = experiment._seed_summaries(
        _call_rows(
            supporting_confirmatory=supporting,
            supporting_episodes={0, 1, 2},
        )
    )
    two = experiment._seed_summaries(
        _call_rows(
            supporting_confirmatory=supporting,
            supporting_episodes={0, 1},
        )
    )

    assert all(
        cast(dict[str, object], row["within_call_exploitation"])["supports"]
        for row in three
        if row["phase"] == "confirmatory"
    )
    assert all(
        cast(dict[str, object], row["visited_state_scorer_shift"])["supports"]
        for row in three
        if row["phase"] == "confirmatory"
    )
    assert not any(
        cast(dict[str, object], row["within_call_exploitation"])["supports"]
        for row in two
        if row["phase"] == "confirmatory"
    )


def test_within_call_signature_requires_later_refinement() -> None:
    assert experiment._within_call_support(
        round0_best_injected=True,
        selected_first_iteration=1,
        learned_gain=1.0,
        exact_delta=-1.0,
        exact_rank_damage=8.0,
    )
    assert not experiment._within_call_support(
        round0_best_injected=True,
        selected_first_iteration=0,
        learned_gain=1.0,
        exact_delta=-1.0,
        exact_rank_damage=8.0,
    )
    assert not experiment._within_call_support(
        round0_best_injected=True,
        selected_first_iteration=1,
        learned_gain=experiment.SCORE_ATOL,
        exact_delta=-1.0,
        exact_rank_damage=8.0,
    )


def test_state_shift_uses_learned_minus_exact_rank_residual() -> None:
    assert experiment._state_shift_support(
        step0_exact_rank=2.0,
        step2_exact_rank=2.0,
        rank_residual_deterioration=8.0,
    )
    assert not experiment._state_shift_support(
        step0_exact_rank=1.0,
        step2_exact_rank=8.0,
        rank_residual_deterioration=1.0,
    )


def test_common_native_bank_is_deterministic_and_step_reusable() -> None:
    first = experiment._common_native_sequences(seed=8, episode_index=0)
    again = experiment._common_native_sequences(seed=8, episode_index=0)
    next_seed = experiment._common_native_sequences(seed=9, episode_index=0)
    next_episode = experiment._common_native_sequences(seed=8, episode_index=1)

    assert first.shape == (40, 12, 2)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, next_seed)
    assert not np.array_equal(first, next_episode)
    assert np.max(np.abs(first)) <= 1.0


def test_sequence_deduplication_retains_first_occurrence() -> None:
    rows = np.zeros((4, 12, 2), dtype=float)
    rows[1, 0, 0] = 1.0
    rows[2] = rows[1]
    rows[3, 0, 1] = -1.0
    assert np.array_equal(
        experiment._first_unique_indices(rows),
        np.array([0, 1, 3]),
    )


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


def test_tensor_package_round_trip_preserves_dtypes_shapes_and_digest(tmp_path: Path) -> None:
    arrays = experiment._empty_arrays()
    for name, array in arrays.items():
        array.fill(False if array.dtype.kind == "b" else 0)
        if name == "seeds":
            array[:] = np.asarray(experiment.ALL_SEEDS, dtype=np.int64)
    path = tmp_path / experiment.TENSOR_FILE
    experiment._write_arrays(path, arrays)
    loaded = experiment._load_arrays(path)

    experiment._validate_array_shapes(loaded)
    assert experiment._array_digest(loaded) == experiment._array_digest(arrays)
    assert all(np.array_equal(loaded[name], arrays[name]) for name in arrays)


def test_report_and_csv_are_json_round_trip_canonical() -> None:
    supporting = set(experiment.CONFIRMATORY_SEEDS[:10])
    call_rows = _call_rows(supporting_confirmatory=supporting)
    summaries = experiment._seed_summaries(call_rows)
    result = {
        "status": "synthetic",
        "call_rows": call_rows,
        "seed_summaries": summaries,
        "decision": experiment._decision(summaries),
        "evaluation_accounting": {
            "actual_execute_total": {
                "learned": {"sequences": 1, "transitions": 12},
                "oracle": {"sequences": 2, "transitions": 24},
            }
        },
    }
    round_tripped = json.loads(json.dumps(result, sort_keys=True))

    assert experiment._report_text(result) == experiment._report_text(round_tripped)
    assert experiment._csv_text(call_rows) == experiment._csv_text(
        cast(list[dict[str, object]], round_tripped["call_rows"])
    )


def test_protocol_freezes_independent_seed_logic_and_prepares_only(tmp_path: Path) -> None:
    protocol = experiment.protocol_record()
    decision = cast(dict[str, object], protocol["decision"])
    assert protocol["replay_seeds"] == list(range(8))
    assert protocol["confirmatory_seeds"] == list(range(8, 20))
    assert decision["support_floor"] == 10
    assert decision["statistical_role"] == ("descriptive model-seed robustness; not a binomial p-value")

    output = tmp_path / "CL-001"
    prepared = experiment.prepare(output)
    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    assert (output / "protocol.json").exists()
    assert (output / "input-manifest.json").exists()
    assert (output / experiment.INPUT_COPY).exists()
    with pytest.raises(ValueError, match="results are required"):
        experiment.verify(output, require_results=True)


def test_formal_start_is_atomic_one_shot_and_incomplete_is_terminal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "CL-001"
    experiment.prepare(output)
    record = experiment._mark_formal_started(output)

    assert record == experiment._formal_start_record(output)
    with pytest.raises(FileExistsError):
        experiment._mark_formal_started(output)
    with pytest.raises(ValueError, match="identifier is terminal"):
        experiment.verify(output)


def test_output_safety_rejects_protected_overlaps_and_repo_sources() -> None:
    protected = experiment.REPO_ROOT / experiment.PARENT_OUTPUT
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._assert_safe_output(protected / "nested")
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._assert_safe_output(protected.parent)
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._mark_formal_started(protected / "moved-prepared-package")
    with pytest.raises(ValueError, match="must be below"):
        experiment._assert_safe_output(experiment.REPO_ROOT / "bench/candidate_landscape")
