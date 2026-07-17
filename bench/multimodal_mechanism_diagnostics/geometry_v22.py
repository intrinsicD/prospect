"""Frozen geometry, exact affine grid, and source-only sampling for MM-008 v2.2.

This module is deliberately standalone.  It contains no target, objective, random,
dataset, or lifecycle code and does not depend on an earlier MM-008 implementation.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Final, TypeAlias, cast

import numpy as np

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"

CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
CENTRAL_START: Final = 8
CENTRAL_STOP: Final = 56
CENTRAL_SIZE: Final = 48
SITE_COUNT: Final = CENTRAL_SIZE * CENTRAL_SIZE
MACRO_SIDE: Final = 6
MACRO_PIXELS: Final = 8
MACRO_COUNT: Final = MACRO_SIDE * MACRO_SIDE
FLOW_LIMIT: Final = 8.0

PARAMETER_NAMES: Final = ("ty", "tx", "ayy", "ayx", "axy", "axx")
STATE_COUNT: Final = 15_625
EXPECTED_ADMISSIBLE_COUNT: Final = 2_809
BATCH_SIZE: Final = 128
FINAL_BATCH_SIZE: Final = 121
BATCH_COUNT: Final = 22

EXPECTED_CANDIDATE_ORDER_SHA256: Final = "dac8a2fcfa35d333f9338f54cd54648ecf0a5a62f96d6d345817b9e2e23d6e79"
EXPECTED_ADMISSIBLE_LIST_SHA256: Final = "6c7dfa679e7a10f52bcbedbb2bbdbaabd397157d7333350930d193e14876711d"
EXPECTED_INVALID_BITMAP_SHA256: Final = "cc478d3eba041f34e5153199f9cccf43fd7891672ff26db96e55d15f5e721132"
EXPECTED_GEOMETRY_SHA256: Final = "759f3f8b0a76984dafd7f93a00fbf755f8d86de0c9f327efaab7a71ea43574d5"

_ADMISSIBLE_TAG: Final = b"MM008-v2.2-admissible-indices\0"
_INVALID_BITMAP_TAG: Final = b"MM008-v2.2-invalid-bitmap\0"
_GEOMETRY_TAG: Final = b"MM008-v2.2-geometry\0"
_FLOAT64_LE: Final = np.dtype("<f8")
_UINT16_LE: Final = np.dtype("<u2")

StateValues: TypeAlias = tuple[float, float, float, float, float, float]
StateInput: TypeAlias = int | np.integer | Sequence[float] | np.ndarray


class GeometryValidationError(ValueError):
    """Raised when an input is not an exact member of the frozen geometry API."""


@dataclass(frozen=True, slots=True)
class Geometry:
    """Immutable row-major central geometry and physical macrocell membership."""

    coords: np.ndarray
    normalized_coords: np.ndarray
    macro_ids: np.ndarray
    parities: np.ndarray


def _readonly(array: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    contiguous = np.array(array, dtype=dtype, order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype)
    return immutable.reshape(contiguous.shape)


def _make_geometry() -> Geometry:
    yy, xx = np.meshgrid(
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=_FLOAT64_LE),
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=_FLOAT64_LE),
        indexing="ij",
    )
    coords = np.stack((yy.reshape(-1), xx.reshape(-1)), axis=1)
    normalized = np.stack(
        (
            (coords[:, 0] - 31.5) / 23.5,
            (coords[:, 1] - 31.5) / 23.5,
        ),
        axis=1,
    )
    macro_y = ((coords[:, 0].astype(np.int64) - CENTRAL_START) // MACRO_PIXELS).astype(np.uint8)
    macro_x = ((coords[:, 1].astype(np.int64) - CENTRAL_START) // MACRO_PIXELS).astype(np.uint8)
    return Geometry(
        coords=_readonly(coords, _FLOAT64_LE),
        normalized_coords=_readonly(normalized, _FLOAT64_LE),
        macro_ids=_readonly(macro_y * MACRO_SIDE + macro_x, np.dtype("u1")),
        parities=_readonly((macro_y + macro_x) % 2, np.dtype("u1")),
    )


GEOMETRY: Final = _make_geometry()
FULL_MASK: Final = _readonly(np.ones(SITE_COUNT, dtype=np.bool_), np.dtype(np.bool_))
PARITY_MASKS: Final = (
    _readonly(GEOMETRY.parities == 0, np.dtype(np.bool_)),
    _readonly(GEOMETRY.parities == 1, np.dtype(np.bool_)),
)


def _ordered_pairs(values: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    pairs = ((first, second) for first in values for second in values)
    return tuple(sorted(pairs, key=lambda pair: (pair[0] * pair[0] + pair[1] * pair[1], *pair)))


_TRANSLATION_VALUES: Final = (-8.0, -4.0, 0.0, 4.0, 8.0)
_GRADIENT_VALUES: Final = (-4.0, -2.0, 0.0, 2.0, 4.0)
T_BLOCK: Final = _ordered_pairs(_TRANSLATION_VALUES)
U_BLOCK: Final = _ordered_pairs(_GRADIENT_VALUES)
V_BLOCK: Final = U_BLOCK


def _canonical_key(state: StateValues) -> tuple[float, float, float, float, float, float, float]:
    return (sum(component * component for component in state), *state)


def _make_canonical_grid() -> np.ndarray:
    # U=(ayy,axy) and V=(ayx,axx), while serialized theta interleaves them.
    rows = [
        cast(StateValues, (translation[0], translation[1], u_pair[0], v_pair[0], u_pair[1], v_pair[1]))
        for translation, u_pair, v_pair in product(T_BLOCK, U_BLOCK, V_BLOCK)
    ]
    rows.sort(key=_canonical_key)
    if len(rows) != STATE_COUNT or len(set(rows)) != STATE_COUNT:
        raise RuntimeError("MM-008 v2.2 canonical affine grid is not a 15,625-state bijection")
    return _readonly(np.asarray(rows, dtype=_FLOAT64_LE), _FLOAT64_LE)


CANONICAL_GRID: Final = _make_canonical_grid()
_STATE_INDEX = {
    cast(StateValues, tuple(float(component) for component in row)): index
    for index, row in enumerate(CANONICAL_GRID)
}
STATE_INDEX: Final = MappingProxyType(_STATE_INDEX)


def _finite_state_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        state = np.asarray(values, dtype=_FLOAT64_LE)
    except (TypeError, ValueError) as error:
        raise GeometryValidationError("state must contain exactly six finite numeric values") from error
    if state.shape != (len(PARAMETER_NAMES),) or not np.all(np.isfinite(state)):
        raise GeometryValidationError("state must contain exactly six finite numeric values")
    return np.ascontiguousarray(state, dtype=_FLOAT64_LE)


def _checked_index(value: int | np.integer) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GeometryValidationError("state index must be an integer")
    index = int(value)
    if not 0 <= index < STATE_COUNT:
        raise GeometryValidationError("state index is outside the canonical grid")
    return index


def state_index(values: Sequence[float] | np.ndarray) -> int:
    """Return the canonical index for one exact, byte-canonical grid state."""

    state = _finite_state_array(values)
    key = cast(StateValues, tuple(float(component) for component in state))
    index = STATE_INDEX.get(key)
    if index is None or state.tobytes(order="C") != CANONICAL_GRID[index].tobytes(order="C"):
        raise GeometryValidationError("state is not an exact member of the canonical grid")
    return index


def _resolve_state_index(state_or_index: StateInput) -> int:
    if isinstance(state_or_index, (int, np.integer)) and not isinstance(state_or_index, (bool, np.bool_)):
        return _checked_index(state_or_index)
    if isinstance(state_or_index, (bool, np.bool_)):
        raise GeometryValidationError("state index must not be boolean")
    return state_index(cast(Sequence[float] | np.ndarray, state_or_index))


def _admissible_rows(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=_FLOAT64_LE)
    if values.ndim != 2 or values.shape[1] != len(PARAMETER_NAMES):
        raise GeometryValidationError("affine parameter table must have shape [N,6]")
    if not np.all(np.isfinite(values)):
        raise GeometryValidationError("affine parameter table contains nonfinite values")
    u = GEOMETRY.normalized_coords[:, 0]
    v = GEOMETRY.normalized_coords[:, 1]
    valid = np.empty(len(values), dtype=np.bool_)
    for start in range(0, len(values), BATCH_SIZE):
        batch = values[start : start + BATCH_SIZE]
        dy = (batch[:, 0, None] + batch[:, 2, None] * u) + batch[:, 3, None] * v
        dx = (batch[:, 1, None] + batch[:, 4, None] * u) + batch[:, 5, None] * v
        valid[start : start + len(batch)] = (np.max(np.abs(dy), axis=1) <= FLOW_LIMIT) & (
            np.max(np.abs(dx), axis=1) <= FLOW_LIMIT
        )
    return valid


def is_admissible(values: Sequence[float] | np.ndarray) -> bool:
    """Return exact inclusive central-grid admissibility for one finite affine state."""

    state = _finite_state_array(values)
    return bool(_admissible_rows(state[None, :])[0])


ADMISSIBLE_MASK: Final = _readonly(_admissible_rows(CANONICAL_GRID), np.dtype(np.bool_))
ADMISSIBLE_INDICES: Final = _readonly(np.flatnonzero(ADMISSIBLE_MASK), _UINT16_LE)
ADMISSIBLE_COUNT: Final = len(ADMISSIBLE_INDICES)

_invalid_bits = np.zeros(math.ceil(STATE_COUNT / 8) * 8, dtype=np.bool_)
_invalid_bits[:STATE_COUNT] = ~ADMISSIBLE_MASK
INVALID_BITMAP: Final = _readonly(np.packbits(_invalid_bits, bitorder="little"), np.dtype("u1"))
ADMISSIBLE_LIST_BYTES: Final = struct.pack("<H", ADMISSIBLE_COUNT) + ADMISSIBLE_INDICES.tobytes(order="C")
INVALID_BITMAP_BYTES: Final = INVALID_BITMAP.tobytes(order="C")

CANDIDATE_ORDER_SHA256: Final = hashlib.sha256(CANONICAL_GRID.tobytes(order="C")).hexdigest()
ADMISSIBLE_LIST_SHA256: Final = hashlib.sha256(_ADMISSIBLE_TAG + ADMISSIBLE_LIST_BYTES).hexdigest()
INVALID_BITMAP_SHA256: Final = hashlib.sha256(_INVALID_BITMAP_TAG + INVALID_BITMAP_BYTES).hexdigest()

_geometry_digest = hashlib.sha256()
_geometry_digest.update(_GEOMETRY_TAG)
_geometry_digest.update(struct.pack("<H", SITE_COUNT))
_geometry_digest.update(GEOMETRY.coords.tobytes(order="C"))
_geometry_digest.update(GEOMETRY.normalized_coords.tobytes(order="C"))
_geometry_digest.update(GEOMETRY.macro_ids.tobytes(order="C"))
_geometry_digest.update(GEOMETRY.parities.tobytes(order="C"))
GEOMETRY_SHA256: Final = _geometry_digest.hexdigest()

ADMISSIBLE_BATCHES: Final = tuple(
    tuple(int(index) for index in ADMISSIBLE_INDICES[start : start + BATCH_SIZE])
    for start in range(0, ADMISSIBLE_COUNT, BATCH_SIZE)
)


def _validated_source(source: np.ndarray) -> np.ndarray:
    if not isinstance(source, np.ndarray):
        raise GeometryValidationError("source must be a NumPy array")
    if source.shape != (CHANNELS, NATIVE_SIZE, NATIVE_SIZE):
        raise GeometryValidationError("source must have shape [3,64,64]")
    if source.dtype != _FLOAT64_LE or not source.flags.c_contiguous:
        raise GeometryValidationError("source must be C-contiguous little-endian float64")
    if not np.all(np.isfinite(source)):
        raise GeometryValidationError("source contains nonfinite values")
    return source


def _validated_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise GeometryValidationError("central mask must be a NumPy array")
    if mask.shape != (SITE_COUNT,) or mask.dtype != np.dtype(np.bool_) or not mask.flags.c_contiguous:
        raise GeometryValidationError("central mask must be C-contiguous bool with shape [2304]")
    if not np.any(mask):
        raise GeometryValidationError("central mask must select at least one site")
    return mask


def _sampling_coordinates(parameters: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = GEOMETRY.coords[mask]
    normalized = GEOMETRY.normalized_coords[mask]
    u = normalized[:, 0]
    v = normalized[:, 1]
    dy = parameters[:, 0, None] + parameters[:, 2, None] * u + parameters[:, 3, None] * v
    dx = parameters[:, 1, None] + parameters[:, 4, None] * u + parameters[:, 5, None] * v
    source_y = coords[None, :, 0] - dy
    source_x = coords[None, :, 1] - dx
    if (
        np.any(source_y < 0.0)
        or np.any(source_x < 0.0)
        or np.any(source_y > NATIVE_SIZE - 1)
        or np.any(source_x > NATIVE_SIZE - 1)
    ):
        raise GeometryValidationError("admissible affine state sampled outside the 64x64 source")
    return source_y, source_x


def sample_scalar(source: np.ndarray, state_or_index: StateInput, mask: np.ndarray) -> np.ndarray:
    """Sample one admissible canonical state with scalar one-candidate arithmetic.

    The result has shape ``[3, selected_site]`` and is immutable little-endian
    float64.  Sampling is the frozen backward warp at ``(y-dy, x-dx)``.
    """

    values = _validated_source(source)
    selected = _validated_mask(mask)
    index = _resolve_state_index(state_or_index)
    if not bool(ADMISSIBLE_MASK[index]):
        raise GeometryValidationError("cannot sample an inadmissible canonical state")
    parameters = CANONICAL_GRID[index : index + 1]
    source_y_table, source_x_table = _sampling_coordinates(parameters, selected)
    source_y = source_y_table[0]
    source_x = source_x_table[0]
    y0 = np.floor(source_y).astype(np.intp)
    x0 = np.floor(source_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, NATIVE_SIZE - 1)
    x1 = np.minimum(x0 + 1, NATIVE_SIZE - 1)
    wy = source_y - y0
    wx = source_x - x0
    top = values[:, y0, x0] * (1.0 - wx)[None, :] + values[:, y0, x1] * wx[None, :]
    bottom = values[:, y1, x0] * (1.0 - wx)[None, :] + values[:, y1, x1] * wx[None, :]
    sampled = top * (1.0 - wy)[None, :] + bottom * wy[None, :]
    return _readonly(sampled, _FLOAT64_LE)


def _validated_batch_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    if type(indices) is not tuple:
        raise GeometryValidationError("batch indices must be a tuple")
    if not indices or len(indices) > BATCH_SIZE:
        raise GeometryValidationError("batch must contain between 1 and 128 indices")
    checked = tuple(_checked_index(index) for index in indices)
    if any(left >= right for left, right in zip(checked, checked[1:], strict=False)):
        raise GeometryValidationError("batch indices must be strictly increasing")
    if any(not bool(ADMISSIBLE_MASK[index]) for index in checked):
        raise GeometryValidationError("batch contains an inadmissible state")
    return checked


def sample_batch(source: np.ndarray, indices: tuple[int, ...], mask: np.ndarray) -> np.ndarray:
    """Vectorize only sampling across up to 128 ascending admissible states.

    The result has shape ``[member,3,selected_site]``.  No objective, appearance,
    macrocell, or other cross-candidate reduction occurs in this function.
    """

    values = _validated_source(source)
    selected = _validated_mask(mask)
    checked = _validated_batch_indices(indices)
    parameters = CANONICAL_GRID[np.asarray(checked, dtype=np.intp)]
    source_y, source_x = _sampling_coordinates(parameters, selected)
    y0 = np.floor(source_y).astype(np.intp)
    x0 = np.floor(source_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, NATIVE_SIZE - 1)
    x1 = np.minimum(x0 + 1, NATIVE_SIZE - 1)
    wy = source_y - y0
    wx = source_x - x0
    channels = np.arange(CHANNELS, dtype=np.intp)[None, :, None]
    top = (
        values[channels, y0[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :]
        + values[channels, y0[:, None, :], x1[:, None, :]] * wx[:, None, :]
    )
    bottom = (
        values[channels, y1[:, None, :], x0[:, None, :]] * (1.0 - wx)[:, None, :]
        + values[channels, y1[:, None, :], x1[:, None, :]] * wx[:, None, :]
    )
    sampled = top * (1.0 - wy)[:, None, :] + bottom * wy[:, None, :]
    return _readonly(sampled, _FLOAT64_LE)


def validate_frozen_constants() -> None:
    """Fail closed if any generated geometry or serialization differs from v2.2."""

    expected_hashes = (
        ("candidate order", CANDIDATE_ORDER_SHA256, EXPECTED_CANDIDATE_ORDER_SHA256),
        ("admissible list", ADMISSIBLE_LIST_SHA256, EXPECTED_ADMISSIBLE_LIST_SHA256),
        ("invalid bitmap", INVALID_BITMAP_SHA256, EXPECTED_INVALID_BITMAP_SHA256),
        ("geometry", GEOMETRY_SHA256, EXPECTED_GEOMETRY_SHA256),
    )
    for name, actual, expected in expected_hashes:
        if actual != expected:
            raise RuntimeError(f"MM-008 v2.2 {name} hash mismatch: {actual}")
    if ADMISSIBLE_COUNT != EXPECTED_ADMISSIBLE_COUNT:
        raise RuntimeError("MM-008 v2.2 admissible count is not 2,809")
    if len(INVALID_BITMAP) != 1_954 or int(INVALID_BITMAP[-1]) & 0xFE:
        raise RuntimeError("MM-008 v2.2 invalid bitmap framing is invalid")
    unpacked = np.unpackbits(INVALID_BITMAP, bitorder="little")[:STATE_COUNT].astype(np.bool_)
    if not np.array_equal(unpacked, ~ADMISSIBLE_MASK):
        raise RuntimeError("MM-008 v2.2 invalid bitmap differs from admissibility")
    if len(ADMISSIBLE_BATCHES) != BATCH_COUNT or tuple(map(len, ADMISSIBLE_BATCHES)) != (
        *((BATCH_SIZE,) * (BATCH_COUNT - 1)),
        FINAL_BATCH_SIZE,
    ):
        raise RuntimeError("MM-008 v2.2 canonical batch partition is invalid")
    flattened = tuple(index for batch in ADMISSIBLE_BATCHES for index in batch)
    if flattened != tuple(int(index) for index in ADMISSIBLE_INDICES):
        raise RuntimeError("MM-008 v2.2 canonical batches do not partition admissible indices")
    for array in (
        GEOMETRY.coords,
        GEOMETRY.normalized_coords,
        GEOMETRY.macro_ids,
        GEOMETRY.parities,
        FULL_MASK,
        *PARITY_MASKS,
        CANONICAL_GRID,
        ADMISSIBLE_MASK,
        ADMISSIBLE_INDICES,
        INVALID_BITMAP,
    ):
        if array.flags.writeable or not array.flags.c_contiguous:
            raise RuntimeError("MM-008 v2.2 public geometry arrays must be immutable and C-contiguous")


validate_frozen_constants()


__all__ = [
    "ADMISSIBLE_BATCHES",
    "ADMISSIBLE_COUNT",
    "ADMISSIBLE_INDICES",
    "ADMISSIBLE_LIST_BYTES",
    "ADMISSIBLE_LIST_SHA256",
    "ADMISSIBLE_MASK",
    "BATCH_COUNT",
    "BATCH_SIZE",
    "CANDIDATE_ORDER_SHA256",
    "CANONICAL_GRID",
    "CENTRAL_SIZE",
    "CENTRAL_START",
    "CENTRAL_STOP",
    "EXPECTED_ADMISSIBLE_COUNT",
    "FINAL_BATCH_SIZE",
    "FLOW_LIMIT",
    "FULL_MASK",
    "GEOMETRY",
    "GEOMETRY_SHA256",
    "Geometry",
    "GeometryValidationError",
    "INVALID_BITMAP",
    "INVALID_BITMAP_BYTES",
    "INVALID_BITMAP_SHA256",
    "MACRO_COUNT",
    "NATIVE_SIZE",
    "PARAMETER_NAMES",
    "PARITY_MASKS",
    "PROTOCOL_SHA256",
    "SITE_COUNT",
    "STATE_COUNT",
    "STATE_INDEX",
    "T_BLOCK",
    "U_BLOCK",
    "V_BLOCK",
    "is_admissible",
    "sample_batch",
    "sample_scalar",
    "state_index",
    "validate_frozen_constants",
]
