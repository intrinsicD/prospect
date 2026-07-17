from __future__ import annotations

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration


def _assert_hard_readonly(array: np.ndarray) -> None:
    assert array.flags.c_contiguous
    assert not array.flags.writeable
    assert array.data.readonly
    current = array
    while True:
        with pytest.raises(ValueError):
            current.setflags(write=True)
        if not isinstance(current.base, np.ndarray):
            assert isinstance(current.base, bytes)
            break
        current = current.base
    with pytest.raises(ValueError):
        array.reshape(-1)[0] = 0.0


def _deterministic_source() -> np.ndarray:
    index = np.arange(6 * 3 * 64 * 64, dtype=np.float64).reshape(6, 3, 64, 64)
    return np.sin(index * 0.017) + np.cos(index * 0.0031)


def test_source_only_normalizer_round_trip_and_immutability() -> None:
    source = _deterministic_source()
    normalizer = calibration.fit_source_only_normalizer(source)
    normalized = normalizer.apply(source)
    restored = normalizer.invert(normalized)
    pooled = calibration.area_pool_r8(source)

    np.testing.assert_allclose(restored, source, rtol=0.0, atol=2e-15)
    assert normalizer.mean.dtype == np.dtype("<f8")
    for array in (normalizer.mean, normalizer.scale, pooled, normalized, restored):
        _assert_hard_readonly(array)


def test_broadband_metrics_are_complete_and_constant_input_fails_gate() -> None:
    metrics = calibration.broadband_validity_metrics(np.ones((6, 3, 64, 64)))
    assert len(metrics.row_rms) == 18
    assert len(metrics.central_variance) == 18
    assert len(metrics.lag_correlation) == 36
    assert len(metrics.lag_denominator_positive) == 36
    assert "nonpositive_row_rms" in metrics.failure_reasons()
    assert not metrics.valid


def test_singleton_global_support_formulas_and_strict_dominance() -> None:
    assert calibration.pair_support(1.0, 2.0, 3.0, 2.0, 2.0, 2.0)
    assert calibration.complete_support("combined", 3.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0)
    assert calibration.strong_support(
        "combined", 4.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0, endpoints_pass=True
    )
    assert calibration.no_bias_gain(1.0, 1.0)
    assert calibration.dominates(1.0, 2.0, 1.0, 2.0)
    assert not calibration.dominates(0.0, 0.0, 0.0, 0.0)


def test_error_records_recompute_and_reject_nonfinite_values() -> None:
    record = calibration.error_record(np.array([1.0, 3.0]), np.array([0.0, 1.0]))
    assert record.sse == 5.0
    assert record.count == 2
    assert record.mse == 2.5
    endpoint = calibration.endpoint_record(np.array([1.0, 3.0]), np.array([0.0, 1.0]))
    assert endpoint.max_abs_error == 2.0
    with pytest.raises(calibration.CalibrationV22Error):
        calibration.error_record(np.array([np.nan]), np.array([0.0]))
