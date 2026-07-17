"""Frozen protocol, derivation, decision, and package-safety tests for VP-001."""

from __future__ import annotations

import math
import stat
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.validator_panel import experiment

EXPECTED_TARGET_CALLS = ((8, 0), (10, 0), (11, 2))
EXPECTED_DIRECTION_CALLS = (
    (8, 0),
    (8, 1),
    (9, 0),
    (10, 0),
    (10, 3),
    (11, 0),
    (11, 2),
    (12, 2),
    (12, 3),
    (18, 0),
    (18, 3),
)
EXPECTED_SENSITIVITY_CALLS = (
    (32, 1),
    (32, 2),
    (32, 3),
    (33, 2),
    (33, 3),
    (35, 0),
    (35, 2),
    (35, 3),
    (36, 2),
    (36, 3),
    (37, 0),
    (38, 1),
    (38, 3),
    (39, 1),
    (39, 3),
    (40, 0),
    (40, 1),
    (40, 2),
    (40, 3),
    (41, 0),
    (41, 2),
    (42, 1),
    (42, 2),
    (42, 3),
    (43, 2),
)
EXPECTED_IDENTITY_SENTINELS = ((9, 0), (12, 2), (12, 3), (18, 0), (18, 3))
EXPECTED_SUBMATERIAL_CALLS = ((8, 1), (10, 3))


def _synthetic_union() -> np.ndarray:
    """Return 192 sequence slots with the frozen 186-first-occurrence geometry."""

    sequence_count = experiment.injection.NATIVE_ITERATIONS * experiment.injection.NATIVE_CANDIDATES
    sequences = np.zeros((sequence_count, experiment.injection.HORIZON, 2), dtype=np.float64)
    sequences[: experiment.EXPECTED_UNIQUE_CANDIDATES, 0, 0] = np.arange(
        1,
        experiment.EXPECTED_UNIQUE_CANDIDATES + 1,
        dtype=np.float64,
    )
    duplicate_count = sequence_count - experiment.EXPECTED_UNIQUE_CANDIDATES
    sequences[experiment.EXPECTED_UNIQUE_CANDIDATES :] = sequences[:duplicate_count]
    return sequences.reshape(
        experiment.injection.NATIVE_ITERATIONS,
        experiment.injection.NATIVE_CANDIDATES,
        experiment.injection.HORIZON,
        2,
    )


def _synthetic_validator_scores() -> np.ndarray:
    """Build finite full-union scores with intentional direction, target, and tie votes."""

    unique_count = experiment.EXPECTED_UNIQUE_CANDIDATES
    sequence_count = experiment.injection.NATIVE_ITERATIONS * experiment.injection.NATIVE_CANDIDATES
    scores = np.empty((len(experiment.VALIDATOR_SEEDS), sequence_count), dtype=np.float64)
    for validator_axis in range(len(experiment.VALIDATOR_SEEDS)):
        unique_scores = -100.0 - np.arange(unique_count, dtype=np.float64)
        if validator_axis < 9:
            unique_scores[:3] = (1.0, 3.0, 2.0)  # C > X > B0.
        elif validator_axis < 11:
            unique_scores[:3] = (2.0, 1.0, 3.0)  # X > B0 > C.
        else:
            unique_scores[:3] = (2.0, 2.0, 2.0)  # Exact three-way tie.
        scores[validator_axis, :unique_count] = unique_scores
        scores[validator_axis, unique_count:] = unique_scores[: sequence_count - unique_count]
    return scores.reshape(
        len(experiment.VALIDATOR_SEEDS),
        experiment.injection.NATIVE_ITERATIONS,
        experiment.injection.NATIVE_CANDIDATES,
    )


def _derive_synthetic_call(
    *,
    seed: int = 8,
    episode_index: int = 0,
    identity_flat_indices: tuple[int, int, int] = (0, 1, 2),
    identity_exact_scores: tuple[float, float, float] = (0.0, 2.0, 1.0),
    validator_scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Call the frozen derivation API with one complete synthetic union."""

    return experiment._derive_call(
        source_kind="parent_visible",
        seed=seed,
        episode_index=episode_index,
        sequences=_synthetic_union(),
        validator_scores=(_synthetic_validator_scores() if validator_scores is None else validator_scores),
        identity_flat_indices=np.asarray(identity_flat_indices, dtype=np.int64),
        identity_exact_scores=np.asarray(identity_exact_scores, dtype=np.float64),
    )


def _base_rows(
    *,
    direction_passing_seeds: set[int] | None = None,
    sensitivity_passing_seeds: set[int] | None = None,
    target_outcomes: dict[tuple[int, int], str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Construct all 96 visible calls while changing only frozen decision fields."""

    direction_support = (
        {seed for seed, _ in EXPECTED_DIRECTION_CALLS} if direction_passing_seeds is None else direction_passing_seeds
    )
    sensitivity_support = (
        {seed for seed, _ in EXPECTED_SENSITIVITY_CALLS}
        if sensitivity_passing_seeds is None
        else sensitivity_passing_seeds
    )
    outcomes = target_outcomes or {key: "reject" for key in EXPECTED_TARGET_CALLS}

    parent_rows: list[dict[str, object]] = []
    for seed in experiment.PARENT_SOURCE_SEEDS:
        for episode in range(4):
            key = (seed, episode)
            target_outcome = outcomes.get(key, "inconclusive")
            is_target = key in EXPECTED_TARGET_CALLS
            is_direction = key in EXPECTED_DIRECTION_CALLS
            direction_passes = is_direction and seed in direction_support
            reject_votes = experiment.CALL_VOTE_FLOOR if is_target and target_outcome == "reject" else 0
            transfer_votes = experiment.CALL_VOTE_FLOOR if is_target and target_outcome == "transfer" else 0
            parent_rows.append(
                {
                    "source_kind": "parent_visible",
                    "seed": seed,
                    "episode_index": episode,
                    "is_target": is_target,
                    "is_direction_control": is_direction,
                    "direction_call_pass": direction_passes,
                    "c_over_b0_votes": experiment.CALL_VOTE_FLOOR if direction_passes else 0,
                    "c_over_x_votes": reject_votes,
                    "x_over_c_votes": transfer_votes,
                    "c_x_tie_votes": len(experiment.VALIDATOR_SEEDS) - reject_votes - transfer_votes,
                    "normalized_c_over_x_rank_gap_median": 0.0,
                    "normalized_c_over_x_rank_gap_q25": 0.0,
                    "normalized_c_over_x_rank_gap_q75": 0.0,
                    "heldout_reject": is_target and target_outcome == "reject",
                    "heldout_transfer": is_target and target_outcome == "transfer",
                }
            )

    fresh_rows: list[dict[str, object]] = []
    for seed in experiment.FRESH_SOURCE_SEEDS:
        for episode in range(4):
            key = (seed, episode)
            fresh_rows.append(
                {
                    "source_kind": "fresh_visible",
                    "seed": seed,
                    "episode_index": episode,
                    "is_sensitivity_control": key in EXPECTED_SENSITIVITY_CALLS,
                    "fixed_x_sensitivity_call_pass": (
                        key in EXPECTED_SENSITIVITY_CALLS and seed in sensitivity_support
                    ),
                }
            )
    return parent_rows, fresh_rows


def _summaries_and_decision(
    *,
    direction_passing_seeds: set[int] | None = None,
    sensitivity_passing_seeds: set[int] | None = None,
    target_outcomes: dict[tuple[int, int], str] | None = None,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    parent_rows, fresh_rows = _base_rows(
        direction_passing_seeds=direction_passing_seeds,
        sensitivity_passing_seeds=sensitivity_passing_seeds,
        target_outcomes=target_outcomes,
    )
    summaries = experiment._seed_summaries(parent_rows, fresh_rows)
    return summaries, experiment._decision(summaries, parent_rows)


def _summary_for_seed(
    summaries: dict[str, list[dict[str, object]]],
    group: str,
    seed: int,
) -> dict[str, object]:
    rows = summaries[group]
    return next(row for row in rows if int(cast(int, row["seed"])) == seed)


def test_protocol_freezes_exact_call_lists_roles_and_thresholds() -> None:
    assert experiment.TARGET_CALLS == EXPECTED_TARGET_CALLS
    assert experiment.DIRECTION_CALLS == EXPECTED_DIRECTION_CALLS
    assert experiment.SENSITIVITY_CALLS == EXPECTED_SENSITIVITY_CALLS
    assert experiment.IDENTITY_SENTINEL_CALLS == EXPECTED_IDENTITY_SENTINELS
    assert experiment.SUBMATERIAL_CALLS == EXPECTED_SUBMATERIAL_CALLS
    assert set(EXPECTED_TARGET_CALLS) < set(EXPECTED_DIRECTION_CALLS)
    assert set(EXPECTED_IDENTITY_SENTINELS) < set(EXPECTED_DIRECTION_CALLS)
    assert set(EXPECTED_SUBMATERIAL_CALLS) < set(EXPECTED_DIRECTION_CALLS)

    assert experiment.VALIDATOR_SEEDS == tuple(range(44, 56))
    assert experiment.PARENT_SOURCE_SEEDS == tuple(range(8, 20))
    assert experiment.FRESH_SOURCE_SEEDS == tuple(range(32, 44))
    assert experiment.DEVELOPMENT_SEED == 97
    formal_roles = (
        set(experiment.VALIDATOR_SEEDS) | set(experiment.PARENT_SOURCE_SEEDS) | set(experiment.FRESH_SOURCE_SEEDS)
    )
    assert experiment.DEVELOPMENT_SEED not in formal_roles
    assert set(experiment.VALIDATOR_SEEDS).isdisjoint(experiment.PARENT_SOURCE_SEEDS)
    assert set(experiment.VALIDATOR_SEEDS).isdisjoint(experiment.FRESH_SOURCE_SEEDS)
    assert experiment.CALL_VOTE_FLOOR == 9
    assert experiment.DIRECTION_SEED_FLOOR == 5
    assert experiment.SENSITIVITY_SEED_FLOOR == 9
    assert experiment.WITHIN_SEED_FRACTION == 0.75

    protocol = experiment.protocol_record()
    fixed = cast(dict[str, object], protocol["fixed_calls"])
    thresholds = cast(dict[str, object], protocol["thresholds"])
    assert fixed["targets"] == [list(key) for key in EXPECTED_TARGET_CALLS]
    assert fixed["direction_controls"] == [list(key) for key in EXPECTED_DIRECTION_CALLS]
    assert fixed["fixed_x_sensitivity_controls"] == [list(key) for key in EXPECTED_SENSITIVITY_CALLS]
    assert fixed["identity_sentinels"] == [list(key) for key in EXPECTED_IDENTITY_SENTINELS]
    assert fixed["submaterial_descriptive"] == [list(key) for key in EXPECTED_SUBMATERIAL_CALLS]
    assert thresholds["call_vote_floor"] == 9
    assert thresholds["within_seed_fraction"] == 0.75
    assert thresholds["direction_seed_floor"] == 5
    assert thresholds["sensitivity_seed_floor"] == 9
    scoring = cast(dict[str, object], protocol["scoring"])
    assert scoring["epistemic_horizon_bound"] is None
    assert scoring["normalized_c_over_x_rank_gap"] == ("(rank(X)-rank(C))/(186-1); positive favors C")
    assert scoring["quantiles"] == "numpy linear method"
    assert cast(dict[str, object], fixed["positive_x_descriptive"]) == {
        "call": [11, 0],
        "x_exact_delta_from_c": experiment.POSITIVE_X_DESCRIPTIVE_EXACT_DELTA,
        "x_exact_rank_gain_from_c": experiment.POSITIVE_X_DESCRIPTIVE_EXACT_RANK_GAIN,
        "x_vs_c_role": "descriptive_non_gating",
    }


def test_call_derivation_is_invariant_to_per_validator_positive_affine_transforms() -> None:
    scores = _synthetic_validator_scores()
    baseline = _derive_synthetic_call(validator_scores=scores)
    transformed = scores.copy()
    for validator_axis in range(len(experiment.VALIDATOR_SEEDS)):
        scale = float(validator_axis + 1)
        offset = float((validator_axis - 5) * 100_000)
        transformed[validator_axis] = scale * transformed[validator_axis] + offset
    rescaled = _derive_synthetic_call(validator_scores=transformed)

    assert baseline == rescaled
    assert baseline["c_over_b0_votes"] == 9
    assert baseline["b0_over_c_votes"] == 2
    assert baseline["b0_c_tie_votes"] == 1
    assert baseline["c_over_x_votes"] == 9
    assert baseline["x_over_c_votes"] == 2
    assert baseline["c_x_tie_votes"] == 1
    assert baseline["direction_call_pass"] is True
    assert baseline["heldout_reject"] is True
    assert baseline["heldout_transfer"] is False
    assert cast(float, baseline["normalized_c_over_x_rank_gap_median"]) > 0.0

    transfer_scores = scores.reshape(len(experiment.VALIDATOR_SEEDS), -1).copy()
    transfer_scores[:, [1, 2]] = transfer_scores[:, [2, 1]]
    transfer_scores[:, [187, 188]] = transfer_scores[:, [188, 187]]
    transfer = _derive_synthetic_call(validator_scores=transfer_scores.reshape(scores.shape))
    assert transfer["heldout_reject"] is False
    assert transfer["heldout_transfer"] is True
    assert cast(float, transfer["normalized_c_over_x_rank_gap_median"]) < 0.0


def test_fresh_sensitivity_derivation_uses_fixed_x_over_c_votes() -> None:
    scores = _synthetic_validator_scores().reshape(len(experiment.VALIDATOR_SEEDS), -1)
    scores[:, [1, 2]] = scores[:, [2, 1]]
    scores[:, [187, 188]] = scores[:, [188, 187]]
    row = experiment._derive_call(
        source_kind="fresh_visible",
        seed=32,
        episode_index=1,
        sequences=_synthetic_union(),
        validator_scores=scores.reshape(
            len(experiment.VALIDATOR_SEEDS),
            experiment.injection.NATIVE_ITERATIONS,
            experiment.injection.NATIVE_CANDIDATES,
        ),
        identity_flat_indices=np.asarray((0, 1, 2), dtype=np.int64),
        identity_exact_scores=np.asarray((0.0, 1.0, 2.0), dtype=np.float64),
    )

    assert row["is_sensitivity_control"] is True
    assert row["x_over_c_votes"] == 9
    assert row["fixed_x_sensitivity_call_pass"] is True


def test_exact_identity_sentinel_is_an_exact_tie_for_every_validator() -> None:
    sentinel = _derive_synthetic_call(
        seed=9,
        episode_index=0,
        identity_flat_indices=(0, 1, 1),
        identity_exact_scores=(0.0, 2.0, 2.0),
    )
    assert sentinel["is_identity_sentinel"] is True
    assert sentinel["c_over_x_votes"] == 0
    assert sentinel["x_over_c_votes"] == 0
    assert sentinel["c_x_tie_votes"] == len(experiment.VALIDATOR_SEEDS)
    assert sentinel["heldout_reject"] is False
    assert sentinel["heldout_transfer"] is False

    context = experiment._parent_context(experiment.REPO_ROOT / experiment.PARENT_OUTPUT)
    experiment._verify_fixed_sets(context)
    parent_rows = cast(dict[tuple[int, int], dict[str, Any]], context["parent_rows"])
    for key in EXPECTED_IDENTITY_SENTINELS:
        row = parent_rows[key]
        assert row["c_union_index"] == row["x_union_index"]
        assert row["c_sequence_sha256"] == row["x_sequence_sha256"]
        assert row["c_exact_score"] == row["x_exact_score"]


def test_seed_summaries_use_ceiling_of_seventy_five_percent() -> None:
    parent_rows, fresh_rows = _base_rows()

    # Direction seed 8 has two eligible calls, so one of two is not enough.
    next(row for row in parent_rows if (row["seed"], row["episode_index"]) == (8, 1))["direction_call_pass"] = False

    # Sensitivity seed 40 has four eligible calls; three pass and one fails.
    next(row for row in fresh_rows if (row["seed"], row["episode_index"]) == (40, 3))[
        "fixed_x_sensitivity_call_pass"
    ] = False

    # Sensitivity seed 32 has three eligible calls; two of three is below ceil(2.25).
    next(row for row in fresh_rows if (row["seed"], row["episode_index"]) == (32, 3))[
        "fixed_x_sensitivity_call_pass"
    ] = False

    summaries = experiment._seed_summaries(parent_rows, fresh_rows)
    direction_8 = _summary_for_seed(summaries, "direction", 8)
    sensitivity_40 = _summary_for_seed(summaries, "sensitivity", 40)
    sensitivity_32 = _summary_for_seed(summaries, "sensitivity", 32)

    assert direction_8["eligible_calls"] == 2
    assert direction_8["required_calls"] == math.ceil(0.75 * 2)
    assert direction_8["passing_calls"] == 1
    assert direction_8["supports"] is False
    assert sensitivity_40["eligible_calls"] == 4
    assert sensitivity_40["required_calls"] == math.ceil(0.75 * 4) == 3
    assert sensitivity_40["passing_calls"] == 3
    assert sensitivity_40["supports"] is True
    assert sensitivity_32["eligible_calls"] == 3
    assert sensitivity_32["required_calls"] == math.ceil(0.75 * 3) == 3
    assert sensitivity_32["passing_calls"] == 2
    assert sensitivity_32["supports"] is False


def test_direction_aggregate_failure_precedes_target_and_sensitivity_branches() -> None:
    _, decision = _summaries_and_decision(
        direction_passing_seeds={8, 10, 11, 12},
        sensitivity_passing_seeds=set(),
        target_outcomes={key: "reject" for key in EXPECTED_TARGET_CALLS},
    )
    assert decision["classification"] == "validator_direction_control_failed"
    assert cast(dict[str, object], decision["direction_control_gate"])["passes"] is False


def test_target_local_direction_failure_precedes_sensitivity_and_target_branch() -> None:
    # Five non-seed-8 direction groups pass, so the aggregate passes while target (8,0) does not.
    _, decision = _summaries_and_decision(
        direction_passing_seeds={9, 10, 11, 12, 18},
        sensitivity_passing_seeds=set(),
        target_outcomes={key: "reject" for key in EXPECTED_TARGET_CALLS},
    )
    assert cast(dict[str, object], decision["direction_control_gate"])["passes"] is True
    assert decision["classification"] == "target_direction_confounded"
    assert cast(dict[str, object], decision["target_direction_gate"])["passing_calls"] == 2


def test_sensitivity_failure_precedes_an_otherwise_decisive_target_branch() -> None:
    sensitivity_seeds = sorted({seed for seed, _ in EXPECTED_SENSITIVITY_CALLS})
    _, decision = _summaries_and_decision(
        sensitivity_passing_seeds=set(sensitivity_seeds[:8]),
        target_outcomes={key: "reject" for key in EXPECTED_TARGET_CALLS},
    )
    assert cast(dict[str, object], decision["direction_control_gate"])["passes"] is True
    assert cast(dict[str, object], decision["target_direction_gate"])["passes"] is True
    assert cast(dict[str, object], decision["fixed_x_sensitivity_gate"])["passes"] is False
    assert decision["classification"] == "validator_fixed_X_sensitivity_control_failed"

    _, boundary = _summaries_and_decision(
        sensitivity_passing_seeds=set(sensitivity_seeds[:9]),
        target_outcomes={key: "reject" for key in EXPECTED_TARGET_CALLS},
    )
    assert cast(dict[str, object], boundary["fixed_x_sensitivity_gate"])["supporting_seeds"] == 9
    assert cast(dict[str, object], boundary["fixed_x_sensitivity_gate"])["passes"] is True
    assert boundary["classification"] == "finite_panel_winners_curse_supported"


@pytest.mark.parametrize(
    ("outcomes", "classification"),
    (
        (
            {key: "reject" for key in EXPECTED_TARGET_CALLS},
            "finite_panel_winners_curse_supported",
        ),
        (
            {key: "transfer" for key in EXPECTED_TARGET_CALLS},
            "same_data_shared_blind_spot_supported",
        ),
        (
            {
                EXPECTED_TARGET_CALLS[0]: "reject",
                EXPECTED_TARGET_CALLS[1]: "transfer",
                EXPECTED_TARGET_CALLS[2]: "inconclusive",
            },
            "heterogeneous_target_failure",
        ),
        (
            {
                EXPECTED_TARGET_CALLS[0]: "reject",
                EXPECTED_TARGET_CALLS[1]: "reject",
                EXPECTED_TARGET_CALLS[2]: "inconclusive",
            },
            "target_panel_inconclusive",
        ),
        (
            {key: "inconclusive" for key in EXPECTED_TARGET_CALLS},
            "target_panel_inconclusive",
        ),
    ),
)
def test_all_frozen_target_branches(
    outcomes: dict[tuple[int, int], str],
    classification: str,
) -> None:
    _, decision = _summaries_and_decision(target_outcomes=outcomes)
    assert decision["classification"] == classification


def test_array_digest_binds_name_dtype_shape_and_content() -> None:
    base = {"values": np.array([[1.0, 2.0]], dtype=np.float64)}
    same = {"values": np.array([[1.0, 2.0]], dtype=np.float64)}
    content = {"values": np.array([[1.0, 3.0]], dtype=np.float64)}
    shape = {"values": np.array([1.0, 2.0], dtype=np.float64)}
    dtype = {"values": np.array([[1.0, 2.0]], dtype=np.float32)}
    name = {"other": np.array([[1.0, 2.0]], dtype=np.float64)}

    assert experiment._array_digest(base) == experiment._array_digest(same)
    assert experiment._array_digest(base) != experiment._array_digest(content)
    assert experiment._array_digest(base) != experiment._array_digest(shape)
    assert experiment._array_digest(base) != experiment._array_digest(dtype)
    assert experiment._array_digest(base) != experiment._array_digest(name)


def test_tensor_package_round_trip_preserves_full_union_schema_and_digest(tmp_path: Path) -> None:
    arrays = experiment._empty_arrays()
    for name, array in arrays.items():
        if name not in ("validator_seeds", "parent_source_seeds", "fresh_source_seeds"):
            array.fill(0)
    experiment._validate_array_shapes(arrays)

    assert arrays["parent_validator_scores"].shape == (12, 12, 4, 3, 64)
    assert arrays["fresh_validator_scores"].shape == (12, 12, 4, 3, 64)
    assert arrays["parent_identity_union_indices"].shape == (12, 4, 3)
    assert arrays["fresh_identity_sequences"].shape == (12, 4, 3, 12, 2)

    path = tmp_path / experiment.TENSOR_FILE
    experiment._write_arrays(path, arrays)
    loaded = experiment._load_arrays(path)

    experiment._validate_array_shapes(loaded)
    assert experiment._array_digest(loaded) == experiment._array_digest(arrays)
    assert all(np.array_equal(loaded[name], arrays[name]) for name in experiment.ARRAY_NAMES)


def test_prepare_and_formal_marker_are_atomic_and_one_shot(tmp_path: Path) -> None:
    output = tmp_path / "VP-001"
    prepared = experiment.prepare(output)

    assert prepared["status"] == "prepared_only"
    assert prepared["outcomes"] == "prepared_only"
    assert (output / "protocol.json").is_file()
    assert (output / "input-manifest.json").is_file()
    assert (output / experiment.PARENT_COPY / experiment.scorer.RESULT_FILE).is_file()
    assert (output / experiment.DATASET_COPY).is_file()
    assert not any((output / relative).exists() for relative in experiment.OUTCOME_PATHS)

    with pytest.raises(FileExistsError, match="absent or an empty directory"):
        experiment.prepare(output)

    unmanifested = output / "unmanifested-claim.txt"
    unmanifested.write_text("not evidence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set drifted"):
        experiment.verify(output)
    unmanifested.unlink()

    symlink = output / "protocol-alias.json"
    symlink.symlink_to(output / "protocol.json")
    with pytest.raises(ValueError, match="forbidden symlink"):
        experiment.verify(output)
    symlink.unlink()

    record = experiment._mark_formal_started(output)
    assert record == experiment._formal_start_record(output)
    assert experiment._read_json(output / experiment.STARTED_FILE) == record
    assert stat.S_IMODE((output / experiment.STARTED_FILE).stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        experiment._mark_formal_started(output)

    with pytest.raises(ValueError, match="identifier is terminal"):
        experiment.verify(output)


def test_result_envelope_rejects_extra_top_level_and_nested_claims() -> None:
    result: dict[str, Any] = {field: "placeholder" for field in experiment.RESULT_FIELDS}
    result.update(
        {
            "formal_start": {},
            "protocol": {},
            "input_manifest": {},
            "tensor_package": {},
            "execution": {},
            "evaluation_accounting": {},
            "repository": {
                "head": "0" * 40,
                "dirty": False,
                "source_sha256": {},
            },
            "versions": {"python": "3", "numpy": "2", "platform": "test"},
            "parent_call_rows": [],
            "fresh_call_rows": [],
            "direction_seed_summaries": [],
            "sensitivity_seed_summaries": [],
            "decision": {},
        }
    )
    experiment._verify_result_schema(result)

    extra_claim = dict(result)
    extra_claim["unmanifested_claim"] = True
    with pytest.raises(ValueError, match="top-level schema"):
        experiment._verify_result_schema(extra_claim)

    nested_claim = dict(result)
    nested_claim["repository"] = {
        **cast(dict[str, object], result["repository"]),
        "unmanifested_claim": True,
    }
    with pytest.raises(ValueError, match="repository result schema"):
        experiment._verify_result_schema(nested_claim)


def test_output_safety_rejects_parent_overlap_sources_and_partial_preparation(tmp_path: Path) -> None:
    protected = experiment.REPO_ROOT / experiment.PARENT_OUTPUT
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._assert_safe_output(protected / "nested")
    with pytest.raises(ValueError, match="protected evidence"):
        experiment._assert_safe_output(protected.parent)
    with pytest.raises(ValueError, match="must be below"):
        experiment._assert_safe_output(experiment.REPO_ROOT / "bench/validator_panel")
    with pytest.raises(ValueError, match="must be below"):
        experiment._assert_safe_output(experiment.REPO_ROOT / "bench/validator_panel/results")

    partial = tmp_path / "partial-VP-001"
    partial.mkdir()
    (partial / "protocol.json").write_text("partial\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="absent or an empty directory"):
        experiment.prepare(partial)
