"""Scientific, leakage, control, and branch tests for MM-007."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from bench.multimodal_preflight import dataset
from bench.multimodal_resolution_diagnostics import method
from bench.multimodal_warp_diagnostics import method as mm006


def _raw_identities() -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(
        [
            video_id
            for video_id in dataset.SAMPLE_VIDEO_IDS
            for _ in range(dataset.EXPECTED_WINDOW_COUNTS[video_id])
        ]
    )
    times = np.concatenate(
        [
            1.0 + 0.5 * np.arange(dataset.EXPECTED_WINDOW_COUNTS[video_id])
            for video_id in dataset.SAMPLE_VIDEO_IDS
        ]
    )
    return ids, times


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def synthetic_rows() -> list[dict[str, Any]]:
    return method._synthetic_rows()


def test_physical_geometry_is_frozen() -> None:
    assert method.RESOLUTIONS == (8, 16, 32, 64)
    assert len(method.NATIVE_CANDIDATES) == 25
    assert method.NATIVE_CANDIDATES[0] == (0.0, 0.0)
    for resolution in method.RESOLUTIONS:
        geometry = method._geometry(resolution)
        scale = resolution // 8
        assert geometry.coords.shape == (36 * scale * scale, 2)
        assert set(geometry.macro_ids.tolist()) == set(range(36))
        assert set(geometry.parities.tolist()) == {0, 1}
        assert [len(set(geometry.macro_ids[geometry.tile_ids == tile])) for tile in range(4)] == [9] * 4


def test_construct_table_and_exact_r8_parity() -> None:
    ids, times = _raw_identities()
    rng = np.random.Generator(np.random.PCG64(7007))
    frames = rng.integers(0, 256, size=(477, 64, 64, 3), dtype=np.uint8)
    expected = method._pool_frames(frames, 8)
    table = method.construct_table(ids, times, frames, expected)
    assert table.video_ids.shape == (453,)
    assert table.current(8).shape == (453, 3, 8, 8)
    assert table.current(64).shape == (453, 3, 64, 64)
    records, normalizers = method._normalizers(table)
    assert len(records) == 16
    for fold in dataset.formal_folds():
        train = np.asarray(
            [video_id in fold.train_ids for video_id in table.video_ids], dtype=bool
        )
        parent = mm006._fit_normalizer(table.current(8)[train])
        shared = [normalizers[(resolution, fold.index)] for resolution in method.RESOLUTIONS]
        assert len({normalizer.fingerprint() for normalizer in shared}) == 1
        assert all(np.array_equal(normalizer.mean, parent.mean) for normalizer in shared)
        assert all(np.array_equal(normalizer.scale, parent.scale) for normalizer in shared)
    bad = expected.copy()
    bad[0, 0, 0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="64-to-8"):
        method.construct_table(ids, times, frames, bad)


@pytest.mark.parametrize("family", method.FLOW_FAMILIES)  # type: ignore[untyped-decorator]
def test_r8_full_and_xfit_replay_mm006(family: str) -> None:
    rng = np.random.Generator(np.random.PCG64(707))
    current = rng.normal(size=(4, 3, 8, 8))
    target = rng.normal(size=(4, 3, 8, 8))
    full = method._estimate_full(current, target, 8, family)
    parent_full = mm006._estimate_oracle_full(current, target, family)
    assert np.array_equal(full.flow / 8.0, parent_full.flow)
    assert np.array_equal(full.prediction.reshape(4, 3, 6, 6), parent_full.prediction)
    xfit = method._estimate_xfit(current, target, 8, family)
    parent_xfit = mm006._estimate_oracle_xfit(current, target, family)
    assert np.array_equal(xfit.flow / 8.0, parent_xfit.flow)
    assert np.array_equal(xfit.prediction.reshape(4, 3, 6, 6), parent_xfit.prediction)


def test_heldout_target_macrocells_cannot_change_their_own_flow() -> None:
    rng = np.random.Generator(np.random.PCG64(77))
    current = rng.normal(size=(3, 3, 32, 32))
    target = rng.normal(size=(3, 3, 32, 32))
    before = method._estimate_xfit(current, target, 32, method.PRIMARY_FAMILY)
    geometry = method._geometry(32)
    held = geometry.parities == 0
    coords = geometry.coords[held].astype(int)
    mutated = target.copy()
    mutated[:, :, coords[:, 0], coords[:, 1]] += 1_000.0
    after = method._estimate_xfit(current, mutated, 32, method.PRIMARY_FAMILY)
    assert np.array_equal(before.flow[:, 0], after.flow[:, 0])
    assert np.array_equal(before.prediction[:, :, held], after.prediction[:, :, held])


def test_macrocell_trimming_preserves_physical_units() -> None:
    residual = np.zeros((2, 3, 20), dtype=np.float64)
    macros = np.repeat(np.arange(5), 4)
    residual[:, :, macros == 4] = 100.0
    # Five macrocell losses trim one whole macrocell, not 25% arbitrary pixels.
    assert np.array_equal(method._macro_trimmed_loss(residual, macros), np.zeros(2))


def test_near_and_far_derangements_are_within_video() -> None:
    ids = np.asarray(["a"] * 5 + ["b"] * 6)
    for mapping in (method._near_derangement(ids), method._far_derangement(ids)):
        assert np.all(mapping != np.arange(len(ids)))
        assert np.array_equal(ids[mapping], ids)
    near = method._near_derangement(ids)
    assert np.max(np.abs(near[:5] - np.arange(5))) <= 2


def test_synthetic_controls_are_discriminating(synthetic_rows: list[dict[str, Any]]) -> None:
    assert method.SYNTHETIC_SEED_MAP == {
        scenario: 700_700 + index
        for index, scenario in enumerate(method.SYNTHETIC_SCENARIOS)
    }
    assert method._validate_synthetic_seed_map(dict(method.SYNTHETIC_SEED_MAP)) == (
        method.SYNTHETIC_SEED_MAP
    )
    tampered_seeds = dict(method.SYNTHETIC_SEED_MAP)
    tampered_seeds["translation"] += 1
    with pytest.raises(ValueError, match="synthetic seed differs"):
        method._validate_synthetic_seed_map(tampered_seeds)
    assert method._validate_synthetic_expectations(
        deepcopy(method.SYNTHETIC_EXPECTATIONS)
    ) == method.SYNTHETIC_EXPECTATIONS
    tampered_expectations = deepcopy(method.SYNTHETIC_EXPECTATIONS)
    tampered_expectations["translation"]["8"][method.PRIMARY_FAMILY][
        "known_native_flow_full_endpoint_mse"
    ] = 1.0
    with pytest.raises(ValueError, match="synthetic expectations differ"):
        method._validate_synthetic_expectations(tampered_expectations)
    for scenario in method.SYNTHETIC_SCENARIOS:
        fingerprints = {
            row["normalizer_fingerprint"]
            for row in synthetic_rows
            if row["synthetic_scenario"] == scenario
        }
        assert len(fingerprints) == 1
    method._validate_synthetic_normalizer_sharing(synthetic_rows)
    tampered_rows = deepcopy(synthetic_rows)
    tampered_rows[0]["normalizer_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="synthetic R8 normalizer"):
        method._validate_synthetic_normalizer_sharing(tampered_rows)
    summary = method._synthetic_summary(synthetic_rows)
    assert summary["positive_passes"] is True
    assert summary["negative_passes"] is True
    assert summary["alias_recovery_passes"] is True
    translation = [
        row for row in synthetic_rows if row["synthetic_scenario"] == "translation"
    ]
    assert all(row["oracle_support"] and row["full_oracle_support"] for row in translation)
    assert all(row["known_native_flow_endpoint_mse"] == 0.0 for row in translation)
    assert all(
        row["known_native_flow_full_endpoint_mse"] == 0.0 for row in translation
    )
    broken_translation = deepcopy(synthetic_rows)
    broken_translation[0]["full_oracle_support"] = False
    assert method._synthetic_summary(broken_translation)["positive_passes"] is False
    broken_full_endpoint = deepcopy(synthetic_rows)
    broken_full_endpoint[0]["known_native_flow_full_endpoint_mse"] = 1.0
    assert method._synthetic_summary(broken_full_endpoint)["positive_passes"] is False
    alias = [
        row
        for row in synthetic_rows
        if row["synthetic_scenario"] == "alias_recovery" and row["family"] == method.PRIMARY_FAMILY
    ]
    assert next(row for row in alias if row["resolution"] == 8)["mse"]["persistence"] == 0.0
    alias_r8 = next(row for row in alias if row["resolution"] == 8)
    assert alias_r8["known_native_flow_endpoint_mse"] == 16.0
    assert alias_r8["known_native_flow_full_endpoint_mse"] == 16.0
    assert all(
        next(row for row in alias if row["resolution"] == resolution)["known_native_flow_endpoint_mse"] == 0.0
        for resolution in (16, 32, 64)
    )
    assert all(
        next(row for row in alias if row["resolution"] == resolution)[
            "known_native_flow_full_endpoint_mse"
        ]
        == 0.0
        for resolution in (16, 32, 64)
    )


def test_metric_validation_replays_predicates_and_numerators(
    synthetic_rows: list[dict[str, Any]],
) -> None:
    row = synthetic_rows[0]
    assert method._validate_metric_row(row, synthetic=True) == row
    tampered = deepcopy(row)
    tampered["oracle_support"] = not bool(tampered["oracle_support"])
    with pytest.raises(ValueError, match="predicate"):
        method._validate_metric_row(tampered, synthetic=True)
    tampered = deepcopy(row)
    tampered["mse"]["oracle_xfit"] += 0.1
    with pytest.raises(ValueError, match="SSE/MSE"):
        method._validate_metric_row(tampered, synthetic=True)


def _family(
    resolution: int,
    family: str,
    *,
    oracle: int,
    full: int,
    null: int = 0,
    performance: int | None = None,
) -> dict[str, Any]:
    performance = oracle if performance is None else performance
    return {
        "resolution": resolution,
        "family": family,
        "oracle_supporting_videos": oracle,
        "oracle_performance_supporting_videos": performance,
        "oracle_passes": oracle >= 6,
        "oracle_performance_passes": performance >= 6,
        "full_oracle_supporting_videos": full,
        "full_oracle_only_videos": full if oracle < 3 else 0,
        "full_oracle_passes": full >= 6,
        "boundary_warning_videos": 0,
        "near_target_null_support": null,
        "far_target_null_support": null,
    }


def _decision_inputs(
    primary: dict[int, tuple[int, int]], *, null: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    families: list[dict[str, Any]] = []
    for resolution in method.RESOLUTIONS:
        oracle, full = primary[resolution]
        families.append(_family(resolution, "global_translation", oracle=0, full=0, null=null))
        families.append(_family(resolution, method.PRIMARY_FAMILY, oracle=oracle, full=full, null=null))
    relative = [
        {"resolution": resolution, "passes": primary[resolution][0] >= 6}
        for resolution in (16, 32, 64)
    ]
    synthetic = {"positive_passes": True, "negative_passes": True, "alias_recovery_passes": True}
    return families, relative, synthetic


def test_frozen_resolution_branch_ladder() -> None:
    families, relative, synthetic = _decision_inputs({8: (0, 1), 16: (8, 8), 32: (8, 8), 64: (8, 8)})
    assert method._decision(families, relative, synthetic)[:2] == ("resolution_recovery_at_16_supported", 16)
    families, relative, synthetic = _decision_inputs({8: (0, 1), 16: (0, 1), 32: (8, 8), 64: (8, 8)})
    assert method._decision(families, relative, synthetic)[:2] == ("resolution_recovery_at_32_supported", 32)
    families, relative, synthetic = _decision_inputs({8: (0, 1), 16: (0, 1), 32: (0, 1), 64: (0, 1)})
    assert method._decision(families, relative, synthetic)[0] == "physically_matched_resolution_failure_supported"


def test_full_only_and_pairing_controls_preempt_claims() -> None:
    families, relative, synthetic = _decision_inputs({8: (0, 1), 16: (0, 4), 32: (0, 1), 64: (0, 1)})
    assert method._decision(families, relative, synthetic)[0] == "MM007_resolution_response_inconclusive"
    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (8, 8), 32: (8, 8), 64: (8, 8)}, null=6
    )
    assert method._decision(families, relative, synthetic)[0] == "invalid_MM007_real_pairing_control"


def test_performance_pairing_disagreement_is_inconclusive() -> None:
    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (0, 1), 32: (0, 1), 64: (0, 1)}
    )
    primary_r16 = next(
        row
        for row in families
        if row["resolution"] == 16 and row["family"] == method.PRIMARY_FAMILY
    )
    primary_r16["oracle_performance_supporting_videos"] = 8
    primary_r16["oracle_performance_passes"] = True
    classification, _, labels = method._decision(families, relative, synthetic)
    assert classification == "MM007_resolution_response_inconclusive"
    assert labels == ["performance_pairing_disagreement_R16"]


def test_focused_inconclusive_decision_cases() -> None:
    synthetic: dict[str, bool]

    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (0, 1), 32: (0, 1), 64: (8, 8)}
    )
    assert method._decision(families, relative, synthetic)[0] == (
        "MM007_resolution_response_inconclusive"
    )

    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (8, 8), 32: (0, 1), 64: (8, 8)}
    )
    assert method._decision(families, relative, synthetic)[0] == (
        "MM007_resolution_response_inconclusive"
    )

    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (8, 8), 32: (8, 8), 64: (8, 8)}
    )
    next(row for row in relative if row["resolution"] == 32)["passes"] = False
    assert method._decision(families, relative, synthetic)[0] == (
        "MM007_resolution_response_inconclusive"
    )

    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (8, 8), 32: (8, 8), 64: (8, 8)}
    )
    primary_r16 = next(
        row
        for row in families
        if row["resolution"] == 16 and row["family"] == method.PRIMARY_FAMILY
    )
    primary_r16["boundary_warning_videos"] = 3
    assert method._decision(families, relative, synthetic)[0] == (
        "MM007_transport_range_inconclusive"
    )

    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (0, 1), 32: (0, 1), 64: (0, 1)}
    )
    global_r16 = next(
        row
        for row in families
        if row["resolution"] == 16 and row["family"] == "global_translation"
    )
    global_r16["oracle_passes"] = True
    classification, _, labels = method._decision(families, relative, synthetic)
    assert classification == "MM007_resolution_response_inconclusive"
    assert labels == ["global_primary_inconsistency_R16"]

    families, relative, synthetic = _decision_inputs(
        {8: (0, 1), 16: (8, 8), 32: (8, 8), 64: (8, 8)}, null=4
    )
    assert method._decision(families, relative, synthetic)[0] == (
        "MM007_real_pairing_control_inconclusive"
    )


def test_config_report_and_scope_exclude_causal_arms() -> None:
    config = method.frozen_config()
    assert config["oracle"]["uses_target"] is True
    assert config["causal_arms_present"] is False
    assert config["deformation_arms_present"] is False
    assert config["residual_arms_present"] is False
    assert config["normalizer"] == {
        "fit": "R8 training-video current only",
        "reference_resolution": 8,
        "shared_across_resolutions": True,
        "scale_floor": method.SCALE_FLOOR,
    }
    assert config["synthetic_seed_map"] == method.SYNTHETIC_SEED_MAP
    assert config["synthetic_expectations"] == method.SYNTHETIC_EXPECTATIONS
    summary: dict[str, Any] = {
        "schema_version": method.SCHEMA_VERSION,
        "experiment_id": method.EXPERIMENT_ID,
        "decision": {
            "classification": "MM007_resolution_response_inconclusive",
            "onset_resolution": None,
            "recommended_next_step": "replicate",
        },
        "synthetic_control": {
            "positive_passes": True,
            "negative_passes": True,
            "alias_recovery_passes": True,
        },
        "families": [],
        "claim_boundary": "oracle diagnostic only",
    }
    report = method.report_text(summary)
    assert "MM007_resolution_response_inconclusive" in report
    assert "oracle diagnostic only" in report
