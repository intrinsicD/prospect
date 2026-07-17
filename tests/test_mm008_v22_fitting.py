"""Focused deterministic tests for the clean MM-008 v2.2 fitting leaf."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting


def _encoded_target() -> np.ndarray:
    channels = np.arange(3, dtype=np.float64)[:, None, None]
    yy = np.arange(64, dtype=np.float64)[None, :, None]
    xx = np.arange(64, dtype=np.float64)[None, None, :]
    return np.asarray(10_000.0 * channels + 100.0 * yy + xx, dtype=np.float64)


def _macro_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    local_ids = fitting.CENTRAL_MACRO_IDS[mask]
    per_site = np.asarray(values[local_ids], dtype=np.float64)
    return np.ascontiguousarray(np.broadcast_to(per_site, (3, len(per_site))))


def test_module_is_a_clean_leaf_without_retired_imports() -> None:
    assert fitting.PROTOCOL_SHA256 == "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
    tree = ast.parse(Path(fitting.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    retired = {
        "bench.multimodal_mechanism_diagnostics.method",
        "bench.multimodal_mechanism_diagnostics.method_v2",
        "bench.multimodal_mechanism_diagnostics.calibration_v2",
        "bench.multimodal_mechanism_diagnostics.synthetic_v2",
        "bench.multimodal_mechanism_diagnostics.control_v2",
    }
    assert imported.isdisjoint(retired)


def test_target_values_follow_central_row_major_order_and_are_immutable() -> None:
    target = _encoded_target()
    mask = fitting.parity_mask(1)
    extracted = fitting.target_values(target, mask)
    coords = fitting.CENTRAL_COORDS[mask]
    expected = target[:, coords[:, 0], coords[:, 1]]
    assert extracted.shape == (3, 1_152)
    assert np.array_equal(extracted, expected)
    assert np.array_equal(coords[0], [8, 16])
    assert np.array_equal(coords[-1], [55, 47])
    with pytest.raises(ValueError):
        extracted[0, 0] = 0.0
    with pytest.raises(ValueError):
        extracted.flags.writeable = True


def test_macro_losses_and_stable_trim_use_physical_ids_and_frozen_counts() -> None:
    full = fitting.full_mask()
    local_ids = fitting.CENTRAL_MACRO_IDS[full]
    residual = np.broadcast_to(local_ids.astype(np.float64), (1, 3, len(local_ids))).copy()
    losses = fitting.macro_losses(residual, full)
    trim = fitting.stable_trim(losses)
    assert losses.macro_ids == tuple(range(36))
    assert np.array_equal(losses.losses, np.arange(36, dtype=np.float64) ** 2)
    assert trim.retained_in_rank_order == tuple(range(27))
    assert trim.retained_macro_ids == tuple(range(27))

    parity = fitting.parity_mask(1)
    zeros = np.zeros((1, 3, int(np.count_nonzero(parity))), dtype=np.float64)
    tied = fitting.stable_trim(fitting.macro_losses(zeros, parity))
    parity_ids = tuple(int(value) for value in np.unique(fitting.CENTRAL_MACRO_IDS[parity]))
    assert len(tied.retained_macro_ids) == 14
    assert tied.retained_in_rank_order == parity_ids[:14]
    assert tied.retained_macro_ids == parity_ids[:14]


def test_ols_uses_raw_covariance_floor_and_bias_before_independent_clipping() -> None:
    x = np.broadcast_to(np.asarray([1.0, 2.0, 1.0, 2.0]), (3, 4)).copy()
    y = 10.0 * x + 1.0
    clipped = fitting.solve_ols(x, y)
    assert np.array_equal(clipped.raw_gains, np.full(3, 10.0))
    assert np.array_equal(clipped.raw_biases, np.ones(3))
    assert np.array_equal(clipped.gains, np.full(3, 4.0))
    assert np.array_equal(clipped.biases, np.ones(3))

    low_variance_x = np.broadcast_to(np.asarray([0.0, 0.001, 0.0, 0.001]), (3, 4)).copy()
    low_variance_y = 2.0 * low_variance_x + 0.5
    floored = fitting.solve_ols(low_variance_x, low_variance_y)
    assert np.allclose(floored.variances, 2.5e-7, rtol=0.0, atol=1e-22)
    assert np.allclose(floored.covariances, 5.0e-7, rtol=0.0, atol=1e-20)
    assert np.allclose(floored.raw_gains, 0.5, rtol=0.0, atol=1e-15)
    assert np.allclose(floored.raw_biases, 0.50075, rtol=0.0, atol=1e-15)

    negative = fitting.solve_ols(x, -3.0 * x - 5.0)
    assert np.array_equal(negative.raw_gains, np.full(3, -3.0))
    assert np.array_equal(negative.raw_biases, np.full(3, -5.0))
    assert np.array_equal(negative.gains, np.full(3, -2.0))
    assert np.array_equal(negative.biases, np.full(3, -4.0))


def test_affine_and_combined_candidate_reducers_require_leading_dimension_one() -> None:
    mask = fitting.parity_mask(0)
    count = int(np.count_nonzero(mask))
    target = np.zeros((3, count), dtype=np.float64)
    one = np.zeros((1, 3, count), dtype=np.float64)
    affine = fitting.affine_objective(one, target, mask)
    combined = fitting.combined_objective(one, target, mask)
    assert affine.objective == 0.0
    assert combined.objective == 0.0
    assert len(affine.trim.retained_macro_ids) == 14
    assert len(combined.retained_macro_ids) == 14

    for invalid in (
        np.zeros((3, count), dtype=np.float64),
        np.zeros((2, 3, count), dtype=np.float64),
    ):
        with pytest.raises(fitting.FittingV22Error, match="shape"):
            fitting.affine_objective(invalid, target, mask)
        with pytest.raises(fitting.FittingV22Error, match="shape"):
            fitting.fit_appearance(invalid, target, mask)


def test_two_pass_appearance_recovers_exact_map_and_keeps_27_or_14_macros() -> None:
    for mask, retained_count in ((fitting.full_mask(), 27), (fitting.parity_mask(0), 14)):
        count = int(np.count_nonzero(mask))
        binary = (np.arange(count, dtype=np.int64) % 2).astype(np.float64)
        source = np.ascontiguousarray(np.broadcast_to(binary, (1, 3, count)))
        gains = np.asarray([2.0, 1.5, 0.5])
        biases = np.asarray([0.5, -0.25, 1.0])
        target = np.ascontiguousarray(gains[:, None] * source[0] + biases[:, None])
        fitted = fitting.fit_appearance(source, target, mask)
        assert np.array_equal(fitted.gains, gains)
        assert np.array_equal(fitted.biases, biases)
        assert np.array_equal(fitted.prediction, target)
        assert fitted.objective == 0.0
        assert len(fitted.retained_macro_ids) == retained_count


def test_appearance_and_bias_objectives_reuse_pass_one_set_without_third_trim() -> None:
    mask = fitting.full_mask()
    values = np.empty(36, dtype=np.float64)
    values[0] = -2.25
    values[1:9] = 0.0
    values[9:] = 0.25
    target = _macro_values(values, mask)
    sampled = np.zeros((1, 3, target.shape[1]), dtype=np.float64)

    appearance = fitting.fit_appearance(sampled, target, mask)
    bias = fitting.fit_bias_only(target, mask, mask)
    expected_retained = tuple(range(1, 28))
    assert appearance.retained_macro_ids == expected_retained
    assert bias.retained_macro_ids == expected_retained

    for fitted in (appearance, bias):
        final = fitted.final_macro_losses.losses
        third_trim = float(np.mean(np.sort(final, kind="stable")[:27], dtype=np.float64))
        assert fitted.objective > third_trim
        by_id = dict(zip(fitted.final_macro_losses.macro_ids, final, strict=True))
        pass_one_objective = float(
            np.mean(
                np.asarray([by_id[index] for index in fitted.retention.retained_in_rank_order]),
                dtype=np.float64,
            )
        )
        assert fitted.objective == pass_one_objective


def test_bias_only_full_and_cross_fit_validate_masks_clipping_and_prediction() -> None:
    full = fitting.full_mask()
    constants = np.asarray([4.0, -4.0, 0.25])
    full_target = np.ascontiguousarray(np.broadcast_to(constants[:, None], (3, 2_304)))
    fitted = fitting.fit_bias_only(full_target, full, full)
    assert np.array_equal(fitted.raw_first_biases, constants)
    assert np.array_equal(fitted.biases, constants)
    assert fitted.retained_macro_ids == tuple(range(27))
    assert fitted.objective == 0.0
    assert np.array_equal(fitted.prediction[:, 0], constants)

    fit_parity = fitting.parity_mask(0)
    output_parity = fitting.parity_mask(1)
    parity_target = np.ascontiguousarray(np.broadcast_to(constants[:, None], (3, 1_152)))
    cross = fitting.fit_bias_only(parity_target, fit_parity, output_parity)
    assert len(cross.retained_macro_ids) == 14
    assert cross.prediction.shape == (3, 1_152)
    with pytest.raises(fitting.FittingV22Error, match="opposite parity"):
        fitting.fit_bias_only(parity_target, fit_parity, fit_parity)


def test_strict_input_and_retained_id_validation_fails_closed() -> None:
    full = fitting.full_mask()
    target = np.zeros((3, 2_304), dtype=np.float64)
    sampled = np.zeros((1, 3, 2_304), dtype=np.float64)
    fitted = fitting.fit_appearance(sampled, target, full)

    invalid_mask = np.array(full, copy=True)
    invalid_mask[0] = False
    with pytest.raises(fitting.FittingV22Error, match="exactly full"):
        fitting.affine_objective(sampled[:, :, 1:], target[:, 1:], invalid_mask)
    with pytest.raises(fitting.FittingV22Error, match="float64"):
        fitting.affine_objective(sampled.astype(np.float32), target, full)
    nonfinite = target.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(fitting.FittingV22Error, match="nonfinite"):
        fitting.fit_appearance(sampled, nonfinite, full)

    corrupted = fitted.retention.retained_macro_ids[:-1] + (35,)
    with pytest.raises(fitting.FittingV22Error, match="persisted retained"):
        replace(fitted.retention, retained_macro_ids=corrupted)
    with pytest.raises(fitting.FittingV22Error, match="0..35"):
        fitting.MacroLosses(tuple(range(17)) + (36,), np.zeros(18))


def test_result_arrays_are_deeply_immutable() -> None:
    full = fitting.full_mask()
    target = np.zeros((3, 2_304), dtype=np.float64)
    sampled = np.zeros((1, 3, 2_304), dtype=np.float64)
    appearance = fitting.fit_appearance(sampled, target, full)
    bias = fitting.fit_bias_only(target, full, full)
    arrays = (
        appearance.gains,
        appearance.biases,
        appearance.prediction,
        appearance.final_macro_losses.losses,
        bias.biases,
        bias.prediction,
        bias.fit_mask,
    )
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
