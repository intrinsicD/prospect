from __future__ import annotations

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import nongrid_v22 as nongrid

CONFIG_SHA256 = "1" * 64


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


def _source() -> np.ndarray:
    index = np.arange(3 * 64 * 64, dtype=np.float64).reshape(3, 64, 64)
    return np.ascontiguousarray(np.sin(index * 0.017) + np.cos(index * 0.0061))


def test_appearance_recovers_injected_identity_map() -> None:
    source = _source()
    gains = np.asarray((1.25, 0.75, 1.5), dtype=np.float64)
    biases = np.asarray((0.35, -0.25, 0.15), dtype=np.float64)
    target = gains[:, None] * geometry.sample_scalar(source, 0, geometry.FULL_MASK) + biases[:, None]
    estimate = nongrid.fit_appearance(
        source,
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        config_sha256=CONFIG_SHA256,
    )

    np.testing.assert_allclose(estimate.gains, gains, rtol=0.0, atol=5e-15)
    np.testing.assert_allclose(estimate.biases, biases, rtol=0.0, atol=1e-15)
    assert estimate.objective < 1e-28
    assert len(estimate.retained_macro_ids) == 27
    assert np.max(np.abs(estimate.prediction - target)) < 8e-15
    for array in (
        estimate.parameters,
        estimate.gains,
        estimate.biases,
        estimate.fit_prediction,
        estimate.prediction,
    ):
        _assert_hard_readonly(array)


def test_constant_target_appearance_matches_bias_only() -> None:
    source = _source()
    target = np.broadcast_to(np.asarray((0.4, -0.7, 1.2))[:, None], (3, geometry.SITE_COUNT)).copy()
    appearance = nongrid.fit_appearance(
        source,
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        config_sha256=CONFIG_SHA256,
    )
    bias = nongrid.fit_bias_only(
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        config_sha256=CONFIG_SHA256,
    )

    assert abs(appearance.objective - bias.objective) <= 1e-12
    assert np.max(np.abs(appearance.prediction - bias.prediction)) <= 1e-12
    assert appearance.scope_sha256 != bias.scope_sha256
    for array in (bias.first_biases, bias.biases, bias.prediction):
        _assert_hard_readonly(array)


def test_persistence_and_bias_scope_are_source_isolated() -> None:
    source = _source()
    persistence = nongrid.persistence(source, geometry.PARITY_MASKS[0], config_sha256=CONFIG_SHA256)
    np.testing.assert_array_equal(
        persistence.prediction, geometry.sample_scalar(source, 0, geometry.PARITY_MASKS[0])
    )
    _assert_hard_readonly(persistence.prediction)
    target = fitting.target_values(source, geometry.PARITY_MASKS[1])
    with pytest.raises(nongrid.NonGridV22Error, match="neither include nor read"):
        nongrid.nongrid_scope_sha256(
            source,
            target,
            geometry.PARITY_MASKS[1],
            geometry.PARITY_MASKS[0],
            "bias_only",
            config_sha256=CONFIG_SHA256,
        )


def test_nongrid_mask_copies_are_hard_readonly() -> None:
    fitted, output = nongrid._masks(geometry.PARITY_MASKS[1], geometry.PARITY_MASKS[0])
    _assert_hard_readonly(fitted)
    _assert_hard_readonly(output)
