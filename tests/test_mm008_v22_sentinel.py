from __future__ import annotations

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import sentinel_v22 as sentinel
from bench.multimodal_resolution_diagnostics import method as mm007

CONFIG_SHA256 = "2" * 64


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


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(3 * 64 * 64, dtype=np.float64).reshape(3, 64, 64)
    source = np.ascontiguousarray(np.sin(index * 0.017) + np.cos(index * 0.0061))
    target = source.copy()
    coords = geometry.GEOMETRY.coords.astype(np.intp)
    target[:, coords[:, 0], coords[:, 1]] = geometry.sample_scalar(
        source, (4.0, -4.0, 0.0, 0.0, 0.0, 0.0), geometry.FULL_MASK
    )
    return source, target


@pytest.mark.parametrize(
    ("arm", "family"),
    (("global_translation", "global_translation"), ("quadrant_translation", "quadrant_flow")),
)
def test_full_sentinel_is_bit_exact_to_sealed_mm007_math(arm: str, family: str) -> None:
    source, target = _fixture()
    actual = sentinel.fit_sentinel(
        source,
        fitting.target_values(target, geometry.FULL_MASK),
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        arm,  # type: ignore[arg-type]
        config_sha256=CONFIG_SHA256,
    )
    expected = mm007._estimate_full(source[None], target[None], 64, family)

    np.testing.assert_array_equal(actual.flow, expected.flow[0])
    np.testing.assert_array_equal(actual.confidence, expected.confidence[0])
    np.testing.assert_array_equal(actual.prediction, expected.prediction[0])
    _assert_hard_readonly(sentinel.TILE_IDS)
    for array in (actual.flow, actual.confidence, actual.prediction):
        _assert_hard_readonly(array)


def test_checkerboard_sentinel_is_bit_exact_to_sealed_mm007_math() -> None:
    source, target = _fixture()
    predictions = np.empty((3, geometry.SITE_COUNT), dtype=np.float64)
    flows = np.empty((2, 4, 2), dtype=np.float64)
    for output_parity in (0, 1):
        result = sentinel.fit_sentinel(
            source,
            fitting.target_values(target, geometry.PARITY_MASKS[1 - output_parity]),
            geometry.PARITY_MASKS[1 - output_parity],
            geometry.PARITY_MASKS[output_parity],
            "quadrant_translation",
            config_sha256=CONFIG_SHA256,
        )
        predictions[:, geometry.PARITY_MASKS[output_parity]] = result.prediction
        flows[output_parity] = result.flow
    expected = mm007._estimate_xfit(source[None], target[None], 64, "quadrant_flow")
    np.testing.assert_array_equal(flows, expected.flow[0])
    np.testing.assert_array_equal(predictions, expected.prediction[0])


def test_sentinel_fit_target_is_copied_and_scope_is_target_bound() -> None:
    source, target = _fixture()
    values = fitting.target_values(target, geometry.FULL_MASK).copy()
    first = sentinel.fit_sentinel(
        source,
        values,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "global_translation",
        config_sha256=CONFIG_SHA256,
    )
    values += 1.0
    second = sentinel.fit_sentinel(
        source,
        values,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "global_translation",
        config_sha256=CONFIG_SHA256,
    )
    assert first.scope_sha256 != second.scope_sha256


def test_sentinel_mask_copies_are_hard_readonly() -> None:
    fitted, output = sentinel._masks(geometry.PARITY_MASKS[1], geometry.PARITY_MASKS[0])
    _assert_hard_readonly(fitted)
    _assert_hard_readonly(output)
