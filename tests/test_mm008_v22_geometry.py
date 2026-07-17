"""Focused deterministic tests for the standalone MM-008 v2.2 geometry core."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry


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
        array.reshape(-1)[0] = 0


def _deterministic_source() -> np.ndarray:
    channel = np.arange(3, dtype="<f8")[:, None, None]
    y = np.arange(64, dtype="<f8")[None, :, None]
    x = np.arange(64, dtype="<f8")[None, None, :]
    return np.ascontiguousarray(10_000.0 * channel + 100.0 * y + x, dtype="<f8")


def test_frozen_geometry_layout_hashes_and_immutability() -> None:
    assert geometry.PROTOCOL_SHA256 == "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
    assert geometry.GEOMETRY.coords.shape == (2_304, 2)
    assert np.array_equal(geometry.GEOMETRY.coords[[0, -1]], [[8.0, 8.0], [55.0, 55.0]])
    assert np.array_equal(geometry.GEOMETRY.normalized_coords[[0, -1]], [[-1.0, -1.0], [1.0, 1.0]])
    assert np.bincount(geometry.GEOMETRY.macro_ids).tolist() == [64] * 36
    assert np.bincount(geometry.GEOMETRY.parities).tolist() == [1_152, 1_152]
    assert np.array_equal(geometry.FULL_MASK, geometry.PARITY_MASKS[0] | geometry.PARITY_MASKS[1])
    assert not np.any(geometry.PARITY_MASKS[0] & geometry.PARITY_MASKS[1])

    assert geometry.CANDIDATE_ORDER_SHA256 == "dac8a2fcfa35d333f9338f54cd54648ecf0a5a62f96d6d345817b9e2e23d6e79"
    assert geometry.ADMISSIBLE_LIST_SHA256 == "6c7dfa679e7a10f52bcbedbb2bbdbaabd397157d7333350930d193e14876711d"
    assert geometry.INVALID_BITMAP_SHA256 == "cc478d3eba041f34e5153199f9cccf43fd7891672ff26db96e55d15f5e721132"
    assert geometry.GEOMETRY_SHA256 == "759f3f8b0a76984dafd7f93a00fbf755f8d86de0c9f327efaab7a71ea43574d5"

    for array in (
        geometry.GEOMETRY.coords,
        geometry.GEOMETRY.normalized_coords,
        geometry.GEOMETRY.macro_ids,
        geometry.GEOMETRY.parities,
        geometry.FULL_MASK,
        *geometry.PARITY_MASKS,
        geometry.CANONICAL_GRID,
        geometry.ADMISSIBLE_MASK,
        geometry.ADMISSIBLE_INDICES,
        geometry.INVALID_BITMAP,
    ):
        _assert_hard_readonly(array)

    assert isinstance(geometry.ADMISSIBLE_BATCHES, tuple)
    assert all(isinstance(batch, tuple) for batch in geometry.ADMISSIBLE_BATCHES)


def test_canonical_grid_state_lookup_and_byte_order_are_strict() -> None:
    assert geometry.CANONICAL_GRID.shape == (15_625, 6)
    assert geometry.CANONICAL_GRID.dtype.str == "<f8"
    assert hashlib.sha256(geometry.CANONICAL_GRID.tobytes(order="C")).hexdigest() == geometry.CANDIDATE_ORDER_SHA256
    assert geometry.state_index((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == 0
    for index in (0, 1, 97, 8_000, geometry.STATE_COUNT - 1):
        assert geometry.state_index(geometry.CANONICAL_GRID[index]) == index

    keys = [
        (sum(float(value) ** 2 for value in row), *(float(value) for value in row))
        for row in geometry.CANONICAL_GRID
    ]
    assert keys == sorted(keys)
    with pytest.raises(geometry.GeometryValidationError):
        geometry.state_index((0.0, 0.0, 0.0, 0.0, 0.0, -0.0))
    with pytest.raises(geometry.GeometryValidationError):
        geometry.state_index((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    with pytest.raises(geometry.GeometryValidationError):
        geometry.state_index((0.0,) * 5)


def test_admissible_serializations_are_exact_and_fail_closed_at_boundary() -> None:
    assert geometry.ADMISSIBLE_COUNT == 2_809
    assert len(geometry.INVALID_BITMAP) == 1_954
    assert geometry.ADMISSIBLE_INDICES.dtype.str == "<u2"
    expected_list = struct.pack("<H", 2_809) + geometry.ADMISSIBLE_INDICES.tobytes(order="C")
    assert geometry.ADMISSIBLE_LIST_BYTES == expected_list
    assert hashlib.sha256(b"MM008-v2.2-admissible-indices\0" + expected_list).hexdigest() == (
        geometry.ADMISSIBLE_LIST_SHA256
    )
    assert hashlib.sha256(b"MM008-v2.2-invalid-bitmap\0" + geometry.INVALID_BITMAP_BYTES).hexdigest() == (
        geometry.INVALID_BITMAP_SHA256
    )
    unpacked = np.unpackbits(geometry.INVALID_BITMAP, bitorder="little")
    assert np.array_equal(unpacked[: geometry.STATE_COUNT].astype(bool), ~geometry.ADMISSIBLE_MASK)
    assert not np.any(unpacked[geometry.STATE_COUNT :])

    assert geometry.is_admissible((8.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert geometry.is_admissible((-8.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert not geometry.is_admissible((np.nextafter(8.0, np.inf), 0.0, 0.0, 0.0, 0.0, 0.0))
    assert not geometry.is_admissible((np.nextafter(-8.0, -np.inf), 0.0, 0.0, 0.0, 0.0, 0.0))

    assert len(geometry.ADMISSIBLE_BATCHES) == 22
    assert [len(batch) for batch in geometry.ADMISSIBLE_BATCHES] == [128] * 21 + [121]
    assert tuple(index for batch in geometry.ADMISSIBLE_BATCHES for index in batch) == tuple(
        int(index) for index in geometry.ADMISSIBLE_INDICES
    )


def test_scalar_sampling_uses_frozen_backward_bilinear_arithmetic() -> None:
    source = _deterministic_source()
    mask = np.zeros(geometry.SITE_COUNT, dtype=np.bool_)
    site_index = (20 - geometry.CENTRAL_START) * geometry.CENTRAL_SIZE + (21 - geometry.CENTRAL_START)
    mask[site_index] = True
    state = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    sampled = geometry.sample_scalar(source, state, mask)

    u = (20.0 - 31.5) / 23.5
    source_y = 20.0 - 2.0 * u
    y0 = int(np.floor(source_y))
    wy = source_y - y0
    expected = source[:, y0, 21] * (1.0 - wy) + source[:, y0 + 1, 21] * wy
    assert sampled.shape == (3, 1)
    assert np.array_equal(sampled[:, 0], expected)
    assert sampled.dtype.str == "<f8"
    _assert_hard_readonly(sampled)

    zero = geometry.sample_scalar(source, 0, geometry.FULL_MASK)
    expected_identity = source[
        :,
        geometry.GEOMETRY.coords[:, 0].astype(np.intp),
        geometry.GEOMETRY.coords[:, 1].astype(np.intp),
    ]
    assert np.array_equal(zero, expected_identity)


def test_batch_sampling_matches_scalar_bits_for_full_and_final_batches() -> None:
    source = _deterministic_source()
    for batch in (geometry.ADMISSIBLE_BATCHES[0], geometry.ADMISSIBLE_BATCHES[-1]):
        sampled = geometry.sample_batch(source, batch, geometry.PARITY_MASKS[0])
        assert sampled.shape == (len(batch), 3, 1_152)
        assert sampled.dtype.str == "<f8"
        _assert_hard_readonly(sampled)
        for position in (0, len(batch) // 2, len(batch) - 1):
            scalar = geometry.sample_scalar(source, batch[position], geometry.PARITY_MASKS[0])
            assert np.array_equal(sampled[position], scalar)


def test_sampling_validation_rejects_noncanonical_inputs() -> None:
    source = _deterministic_source()
    first_batch = geometry.ADMISSIBLE_BATCHES[0]
    invalid_index = int(np.flatnonzero(~geometry.ADMISSIBLE_MASK)[0])

    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_batch(source, list(first_batch), geometry.FULL_MASK)  # type: ignore[arg-type]
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_batch(source, tuple(reversed(first_batch)), geometry.FULL_MASK)
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_batch(source, (*first_batch, first_batch[-1] + 1), geometry.FULL_MASK)
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_batch(source, (invalid_index,), geometry.FULL_MASK)
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_scalar(source.astype(np.float32), 0, geometry.FULL_MASK)
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_scalar(source, 0, geometry.FULL_MASK.astype(np.uint8))
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_scalar(source, 0, np.zeros(geometry.SITE_COUNT, dtype=np.bool_))
    with pytest.raises(geometry.GeometryValidationError):
        geometry.sample_scalar(source, invalid_index, geometry.FULL_MASK)
