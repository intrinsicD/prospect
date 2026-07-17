"""Independent scalar scientific oracle for the MM-008 v2.2 exact grid.

This module intentionally depends only on the Python standard library and NumPy.
It reconstructs the frozen geometry, canonical grid, scalar sampler, fitting
mathematics, and evidence hashes locally.  Candidate samples are never stacked:
the exhaustive path holds one ``[3, site]`` sample at a time and groups only its
serialized bytes into the normative 128/121 source frames.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

import numpy as np

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-independent-scalar-oracle-v1"

CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
CENTRAL_START: Final = 8
CENTRAL_STOP: Final = 56
CENTRAL_SIZE: Final = CENTRAL_STOP - CENTRAL_START
SITE_COUNT: Final = CENTRAL_SIZE * CENTRAL_SIZE
MACRO_SIDE: Final = 6
MACRO_PIXELS: Final = 8
FLOW_LIMIT: Final = 8.0
STATE_COUNT: Final = 15_625
ADMISSIBLE_COUNT: Final = 2_809
BATCH_SIZE: Final = 128
FINAL_BATCH_SIZE: Final = 121
BATCH_COUNT: Final = 22

TRIM_FRACTION: Final = 0.25
VARIANCE_FLOOR: Final = 1e-6
GAIN_BOUNDS: Final = (-2.0, 4.0)
BIAS_BOUNDS: Final = (-4.0, 4.0)

EXPECTED_CANDIDATE_ORDER_SHA256: Final = "dac8a2fcfa35d333f9338f54cd54648ecf0a5a62f96d6d345817b9e2e23d6e79"
EXPECTED_ADMISSIBLE_LIST_SHA256: Final = "6c7dfa679e7a10f52bcbedbb2bbdbaabd397157d7333350930d193e14876711d"
EXPECTED_INVALID_BITMAP_SHA256: Final = "cc478d3eba041f34e5153199f9cccf43fd7891672ff26db96e55d15f5e721132"
EXPECTED_GEOMETRY_SHA256: Final = "759f3f8b0a76984dafd7f93a00fbf755f8d86de0c9f327efaab7a71ea43574d5"

_ADMISSIBLE_TAG: Final = b"MM008-v2.2-admissible-indices\0"
_INVALID_BITMAP_TAG: Final = b"MM008-v2.2-invalid-bitmap\0"
_GEOMETRY_TAG: Final = b"MM008-v2.2-geometry\0"
_SOURCE_SCOPE_TAG: Final = b"MM008-v2.2-source-grid-scope\0"
_SOURCE_BATCH_TAG: Final = b"MM008-v2.2-source-grid-batch\0"
_SOURCE_PARTITION_TAG: Final = b"MM008-v2.2-source-grid-partition\0"
_SOURCE_SAMPLES_TAG: Final = b"MM008-v2.2-source-grid-samples\0"
_SOURCE_CONTENT_TAG: Final = b"MM008-v2.2-source-grid-content\0"
_OBJECTIVE_SCOPE_TAG: Final = b"MM008-v2.2-objective-scope\0"
_OBJECTIVE_CONTENT_TAG: Final = b"MM008-v2.2-objective-content\0"
_SELECTED_EVALUATION_TAG: Final = b"MM008-v2.2-selected-evaluation\0"
_SELECTED_PREDICTION_TAG: Final = b"MM008-v2.2-selected-prediction\0"

_FLOAT64_LE: Final = np.dtype("<f8")
_UINT16_LE: Final = np.dtype("<u2")
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")

Arm = Literal["affine", "combined"]
State: TypeAlias = tuple[float, float, float, float, float, float]
StateInput: TypeAlias = int | np.integer | tuple[float, ...] | list[float] | np.ndarray
_ARM_BYTE: Final[dict[Arm, int]] = {"affine": 0, "combined": 1}


class OracleV22Error(ValueError):
    """Raised when an oracle input or independently derived invariant fails."""


def _immutable(value: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    array = _immutable(np.asarray(value), _FLOAT64_LE)
    if not bool(np.all(np.isfinite(array))):
        raise OracleV22Error("scientific float64 evidence contains a nonfinite value")
    return array


def _immutable_bool(value: np.ndarray) -> np.ndarray:
    normalized = np.ascontiguousarray(value, dtype=np.uint8)
    if bool(np.any((normalized != 0) & (normalized != 1))):
        raise OracleV22Error("mask contains a value other than zero or one")
    return np.frombuffer(normalized.tobytes(order="C"), dtype=np.bool_).reshape(normalized.shape)


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise OracleV22Error(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _validate_retained_ids(values: tuple[int, ...], allowed_lengths: tuple[int, ...]) -> None:
    if (
        type(values) is not tuple
        or len(values) not in allowed_lengths
        or any(type(value) is not int or not 0 <= value < MACRO_SIDE * MACRO_SIDE for value in values)
        or tuple(sorted(set(values))) != values
    ):
        raise OracleV22Error("retained macro IDs are invalid")


def _same_float(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def _state_key(state: State) -> tuple[float, float, float, float, float, float, float]:
    return (sum(component * component for component in state), *state)


def _ordered_pairs(values: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    pairs = ((first, second) for first in values for second in values)
    return tuple(sorted(pairs, key=lambda pair: (pair[0] * pair[0] + pair[1] * pair[1], *pair)))


def _build_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=_FLOAT64_LE),
        np.arange(CENTRAL_START, CENTRAL_STOP, dtype=_FLOAT64_LE),
        indexing="ij",
    )
    coords = np.stack((yy.reshape(-1), xx.reshape(-1)), axis=1)
    normalized = np.stack(
        ((coords[:, 0] - 31.5) / 23.5, (coords[:, 1] - 31.5) / 23.5), axis=1
    )
    macro_y = ((coords[:, 0].astype(np.int64) - CENTRAL_START) // MACRO_PIXELS).astype(np.uint8)
    macro_x = ((coords[:, 1].astype(np.int64) - CENTRAL_START) // MACRO_PIXELS).astype(np.uint8)
    macro_ids = macro_y * MACRO_SIDE + macro_x
    parities = (macro_y + macro_x) % 2
    return (
        _immutable(coords, _FLOAT64_LE),
        _immutable(normalized, _FLOAT64_LE),
        _immutable(macro_ids, np.dtype("u1")),
        _immutable(parities, np.dtype("u1")),
    )


CENTRAL_COORDS, NORMALIZED_COORDS, MACRO_IDS, MACRO_PARITIES = _build_geometry()
FULL_MASK: Final = _immutable_bool(np.ones(SITE_COUNT, dtype=np.bool_))
PARITY_MASKS: Final = (
    _immutable_bool(MACRO_PARITIES == 0),
    _immutable_bool(MACRO_PARITIES == 1),
)

_TRANSLATION_PAIRS: Final = _ordered_pairs((-8.0, -4.0, 0.0, 4.0, 8.0))
_GRADIENT_PAIRS: Final = _ordered_pairs((-4.0, -2.0, 0.0, 2.0, 4.0))


def _build_states() -> tuple[State, ...]:
    states = [
        cast(State, (translation[0], translation[1], u[0], v[0], u[1], v[1]))
        for translation, u, v in product(
            _TRANSLATION_PAIRS, _GRADIENT_PAIRS, _GRADIENT_PAIRS
        )
    ]
    states.sort(key=_state_key)
    if len(states) != STATE_COUNT or len(set(states)) != STATE_COUNT:
        raise RuntimeError("oracle canonical grid is not a 15,625-state bijection")
    return tuple(states)


CANONICAL_STATES: Final = _build_states()
CANONICAL_GRID: Final = _immutable(np.asarray(CANONICAL_STATES), _FLOAT64_LE)
STATE_INDEX: Final = MappingProxyType({state: index for index, state in enumerate(CANONICAL_STATES)})


def _flow(state: State) -> tuple[np.ndarray, np.ndarray]:
    u = NORMALIZED_COORDS[:, 0]
    v = NORMALIZED_COORDS[:, 1]
    dy = (state[0] + state[2] * u) + state[3] * v
    dx = (state[1] + state[4] * u) + state[5] * v
    return dy, dx


def _is_admissible(state: State) -> bool:
    dy, dx = _flow(state)
    return bool(np.max(np.abs(dy)) <= FLOW_LIMIT and np.max(np.abs(dx)) <= FLOW_LIMIT)


ADMISSIBLE_INDICES: Final = tuple(
    index for index, state in enumerate(CANONICAL_STATES) if _is_admissible(state)
)
ADMISSIBLE_INDEX_SET: Final = frozenset(ADMISSIBLE_INDICES)
ADMISSIBLE_BATCHES: Final = tuple(
    ADMISSIBLE_INDICES[start : start + BATCH_SIZE]
    for start in range(0, len(ADMISSIBLE_INDICES), BATCH_SIZE)
)

_admissible_array = np.asarray(ADMISSIBLE_INDICES, dtype=_UINT16_LE)
ADMISSIBLE_LIST_BYTES: Final = struct.pack("<H", len(ADMISSIBLE_INDICES)) + _admissible_array.tobytes(
    order="C"
)
_invalid_bits = np.zeros(math.ceil(STATE_COUNT / 8) * 8, dtype=np.bool_)
_invalid_bits[:STATE_COUNT] = True
_invalid_bits[_admissible_array.astype(np.intp)] = False
INVALID_BITMAP: Final = _immutable(np.packbits(_invalid_bits, bitorder="little"), np.dtype("u1"))
INVALID_BITMAP_BYTES: Final = INVALID_BITMAP.tobytes(order="C")

CANDIDATE_ORDER_SHA256: Final = hashlib.sha256(CANONICAL_GRID.tobytes(order="C")).hexdigest()
ADMISSIBLE_LIST_SHA256: Final = hashlib.sha256(
    _ADMISSIBLE_TAG + ADMISSIBLE_LIST_BYTES
).hexdigest()
INVALID_BITMAP_SHA256: Final = hashlib.sha256(
    _INVALID_BITMAP_TAG + INVALID_BITMAP_BYTES
).hexdigest()
_geometry_digest = hashlib.sha256()
_geometry_digest.update(_GEOMETRY_TAG)
_geometry_digest.update(struct.pack("<H", SITE_COUNT))
_geometry_digest.update(CENTRAL_COORDS.tobytes(order="C"))
_geometry_digest.update(NORMALIZED_COORDS.tobytes(order="C"))
_geometry_digest.update(MACRO_IDS.tobytes(order="C"))
_geometry_digest.update(MACRO_PARITIES.tobytes(order="C"))
GEOMETRY_SHA256: Final = _geometry_digest.hexdigest()


def validate_frozen_reconstruction() -> None:
    """Fail closed unless every independently reconstructed frozen identity matches."""

    actual = (
        CANDIDATE_ORDER_SHA256,
        ADMISSIBLE_LIST_SHA256,
        INVALID_BITMAP_SHA256,
        GEOMETRY_SHA256,
    )
    expected = (
        EXPECTED_CANDIDATE_ORDER_SHA256,
        EXPECTED_ADMISSIBLE_LIST_SHA256,
        EXPECTED_INVALID_BITMAP_SHA256,
        EXPECTED_GEOMETRY_SHA256,
    )
    if actual != expected:
        raise OracleV22Error(f"independent frozen reconstruction mismatch: {actual!r}")
    if len(ADMISSIBLE_INDICES) != ADMISSIBLE_COUNT:
        raise OracleV22Error("independent admissible count is not 2,809")
    if tuple(len(batch) for batch in ADMISSIBLE_BATCHES) != (BATCH_SIZE,) * 21 + (
        FINAL_BATCH_SIZE,
    ):
        raise OracleV22Error("independent source partition is not 21x128 plus 121")
    if len(INVALID_BITMAP_BYTES) != 1_954 or int(INVALID_BITMAP[-1]) & 0xFE:
        raise OracleV22Error("independent invalid bitmap framing is malformed")


def _coerce_state(state_or_index: StateInput) -> tuple[int, State]:
    if isinstance(state_or_index, (int, np.integer)) and not isinstance(
        state_or_index, (bool, np.bool_)
    ):
        index = int(state_or_index)
        if not 0 <= index < STATE_COUNT:
            raise OracleV22Error("state index is outside the canonical grid")
        return index, CANONICAL_STATES[index]
    if isinstance(state_or_index, (bool, np.bool_)):
        raise OracleV22Error("state index must not be boolean")
    try:
        array = np.asarray(state_or_index, dtype=_FLOAT64_LE)
    except (TypeError, ValueError) as error:
        raise OracleV22Error("state must contain six finite canonical values") from error
    if array.shape != (6,) or not bool(np.all(np.isfinite(array))):
        raise OracleV22Error("state must contain six finite canonical values")
    state = cast(State, tuple(float(value) for value in array))
    resolved_index = STATE_INDEX.get(state)
    if resolved_index is None or array.tobytes(order="C") != CANONICAL_GRID[
        resolved_index
    ].tobytes(order="C"):
        raise OracleV22Error("state is not an exact canonical-grid member")
    return resolved_index, state


def state_index(state: StateInput) -> int:
    """Return the independent zero-based canonical index for one exact state."""

    return _coerce_state(state)[0]


def _validate_source(source: np.ndarray) -> np.ndarray:
    if not isinstance(source, np.ndarray):
        raise OracleV22Error("source must be a NumPy array")
    if source.shape != (CHANNELS, NATIVE_SIZE, NATIVE_SIZE):
        raise OracleV22Error("source must have shape [3,64,64]")
    if source.dtype != np.dtype(np.float64) or not source.flags.c_contiguous:
        raise OracleV22Error("source must be C-contiguous float64")
    return _immutable_float64(source)


def _validate_mask(mask: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise OracleV22Error(f"{label} must be a NumPy array")
    if (
        mask.shape != (SITE_COUNT,)
        or mask.dtype != np.dtype(np.bool_)
        or not mask.flags.c_contiguous
    ):
        raise OracleV22Error(f"{label} must be C-contiguous bool with shape [2304]")
    normalized = _immutable_bool(mask)
    if not any(np.array_equal(normalized, frozen) for frozen in (FULL_MASK, *PARITY_MASKS)):
        raise OracleV22Error(f"{label} is not a frozen full or checkerboard mask")
    return normalized


def _validate_context_masks(
    fit_mask: np.ndarray, output_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fitted = _validate_mask(fit_mask, "fit mask")
    output = _validate_mask(output_mask, "output mask")
    full = np.array_equal(fitted, FULL_MASK) and np.array_equal(output, FULL_MASK)
    cross_fit = any(
        np.array_equal(fitted, PARITY_MASKS[1 - output_parity])
        and np.array_equal(output, PARITY_MASKS[output_parity])
        for output_parity in (0, 1)
    )
    if not full and not cross_fit:
        raise OracleV22Error("fit/output masks are not a frozen full or checkerboard context")
    return fitted, output


def _validate_fit_target(fit_target: np.ndarray, count: int) -> np.ndarray:
    if not isinstance(fit_target, np.ndarray):
        raise OracleV22Error("fit target must be a NumPy array")
    if (
        fit_target.shape != (CHANNELS, count)
        or fit_target.dtype != np.dtype(np.float64)
        or not fit_target.flags.c_contiguous
    ):
        raise OracleV22Error(f"fit target must be C-contiguous float64 with shape [3,{count}]")
    return _immutable_float64(fit_target)


def target_values(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Independently extract central target values in frozen row-major order."""

    values = _validate_source(target)
    selected = _validate_mask(mask, "target mask")
    coords = CENTRAL_COORDS[selected].astype(np.intp)
    return _immutable_float64(values[:, coords[:, 0], coords[:, 1]])


def _sample_state(source: np.ndarray, state: State, mask: np.ndarray) -> np.ndarray:
    coords = CENTRAL_COORDS[mask]
    normalized = NORMALIZED_COORDS[mask]
    u = normalized[:, 0]
    v = normalized[:, 1]
    dy = (state[0] + state[2] * u) + state[3] * v
    dx = (state[1] + state[4] * u) + state[5] * v
    source_y = coords[:, 0] - dy
    source_x = coords[:, 1] - dx
    if (
        bool(np.any(source_y < 0.0))
        or bool(np.any(source_x < 0.0))
        or bool(np.any(source_y > NATIVE_SIZE - 1))
        or bool(np.any(source_x > NATIVE_SIZE - 1))
    ):
        raise OracleV22Error("admissible state sampled outside the source")
    y0 = np.floor(source_y).astype(np.intp)
    x0 = np.floor(source_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, NATIVE_SIZE - 1)
    x1 = np.minimum(x0 + 1, NATIVE_SIZE - 1)
    wy = source_y - y0
    wx = source_x - x0
    top = source[:, y0, x0] * (1.0 - wx)[None, :] + source[:, y0, x1] * wx[None, :]
    bottom = source[:, y1, x0] * (1.0 - wx)[None, :] + source[:, y1, x1] * wx[None, :]
    return np.ascontiguousarray(top * (1.0 - wy)[None, :] + bottom * wy[None, :], dtype=_FLOAT64_LE)


def sample_scalar(source: np.ndarray, state_or_index: StateInput, mask: np.ndarray) -> np.ndarray:
    """Sample exactly one admissible state through the independent bilinear path."""

    values = _validate_source(source)
    selected = _validate_mask(mask, "sample mask")
    index, state = _coerce_state(state_or_index)
    if index not in ADMISSIBLE_INDEX_SET:
        raise OracleV22Error("cannot sample an inadmissible canonical state")
    return _immutable_float64(_sample_state(values, state, selected))


def _macro_losses(residual: np.ndarray, fit_mask: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    local_ids = np.asarray(MACRO_IDS[fit_mask], dtype=np.int64)
    ids = tuple(int(value) for value in np.unique(local_ids))
    error = residual[None, :, :]
    pixel_error = np.asarray(np.mean(error * error, axis=1, dtype=np.float64), dtype=np.float64)
    losses = np.asarray(
        [np.mean(pixel_error[0, local_ids == macro], dtype=np.float64) for macro in ids],
        dtype=np.float64,
    )
    if not bool(np.all(np.isfinite(losses))) or bool(np.any(losses < 0.0)):
        raise OracleV22Error("candidate produced invalid macro losses")
    return ids, losses


def _stable_trim(
    ids: tuple[int, ...], losses: np.ndarray
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    order = np.argsort(losses, kind="stable")
    ranked = tuple(ids[int(position)] for position in order)
    keep = len(ranked) - math.floor(len(ranked) * TRIM_FRACTION)
    retained_ranked = ranked[:keep]
    retained_sorted = tuple(sorted(retained_ranked))
    objective = float(np.mean(losses[order[:keep]], dtype=np.float64))
    return objective, retained_ranked, retained_sorted


def _solve_ols(
    source: np.ndarray, target: np.ndarray, selected: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if selected is None:
        pixel_mask = np.ones(source.shape[1], dtype=np.bool_)
    else:
        pixel_mask = selected
    if not bool(np.any(pixel_mask)):
        raise OracleV22Error("appearance fit selected no pixels")
    x = source[:, pixel_mask]
    y = target[:, pixel_mask]
    mean_x = np.asarray(np.mean(x, axis=1, dtype=np.float64), dtype=np.float64)
    mean_y = np.asarray(np.mean(y, axis=1, dtype=np.float64), dtype=np.float64)
    centered_x = x - mean_x[:, None]
    centered_y = y - mean_y[:, None]
    variance = np.asarray(
        np.mean(centered_x * centered_x, axis=1, dtype=np.float64), dtype=np.float64
    )
    covariance = np.asarray(
        np.mean(centered_x * centered_y, axis=1, dtype=np.float64), dtype=np.float64
    )
    raw_gain = covariance / np.maximum(variance, VARIANCE_FLOOR)
    raw_bias = mean_y - raw_gain * mean_x
    return (
        np.ascontiguousarray(np.clip(raw_gain, *GAIN_BOUNDS), dtype=_FLOAT64_LE),
        np.ascontiguousarray(np.clip(raw_bias, *BIAS_BOUNDS), dtype=_FLOAT64_LE),
    )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Immutable result of one independently evaluated scalar candidate."""

    arm: Arm
    objective: float
    prediction: np.ndarray
    gains: np.ndarray | None
    biases: np.ndarray | None
    retained_macro_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.arm not in _ARM_BYTE
            or type(self.objective) is not float
            or not math.isfinite(self.objective)
            or self.objective < 0.0
            or type(self.retained_macro_ids) is not tuple
        ):
            raise OracleV22Error("candidate evaluation metadata is invalid")
        prediction = _immutable_float64(self.prediction)
        if prediction.ndim != 2 or prediction.shape[0] != CHANNELS:
            raise OracleV22Error("candidate prediction must have shape [3,fit_site]")
        object.__setattr__(self, "prediction", prediction)
        if self.arm == "affine":
            if self.gains is not None or self.biases is not None or self.retained_macro_ids:
                raise OracleV22Error("affine evaluation carries appearance values")
            return
        if self.gains is None or self.biases is None:
            raise OracleV22Error("combined evaluation lacks appearance values")
        gains = _immutable_float64(self.gains)
        biases = _immutable_float64(self.biases)
        if gains.shape != (CHANNELS,) or biases.shape != gains.shape:
            raise OracleV22Error("combined appearance vectors must have shape [3]")
        if bool(np.any(gains < GAIN_BOUNDS[0])) or bool(np.any(gains > GAIN_BOUNDS[1])):
            raise OracleV22Error("combined gains are outside the frozen bounds")
        if bool(np.any(biases < BIAS_BOUNDS[0])) or bool(np.any(biases > BIAS_BOUNDS[1])):
            raise OracleV22Error("combined biases are outside the frozen bounds")
        _validate_retained_ids(self.retained_macro_ids, (14, 27))
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)


def evaluate_candidate(
    arm: Arm, sampled: np.ndarray, fit_target: np.ndarray, fit_mask: np.ndarray
) -> CandidateEvaluation:
    """Evaluate one candidate with locally reconstructed trim and appearance math."""

    if arm not in _ARM_BYTE:
        raise OracleV22Error("arm must be exactly affine or combined")
    selected = _validate_mask(fit_mask, "fit mask")
    count = int(np.count_nonzero(selected))
    source = _validate_fit_target(sampled, count)
    target = _validate_fit_target(fit_target, count)
    if arm == "affine":
        objective, _, _ = _stable_trim(*_macro_losses(source - target, selected))
        return CandidateEvaluation(arm, objective, source, None, None, ())

    first_gain, first_bias = _solve_ols(source, target)
    first_prediction = first_gain[:, None] * source + first_bias[:, None]
    _, retained_ranked, retained_sorted = _stable_trim(
        *_macro_losses(first_prediction - target, selected)
    )
    local_ids = np.asarray(MACRO_IDS[selected], dtype=np.int64)
    retained_pixels = np.asarray(np.isin(local_ids, retained_sorted), dtype=np.bool_)
    final_gain, final_bias = _solve_ols(source, target, retained_pixels)
    prediction = final_gain[:, None] * source + final_bias[:, None]
    final_ids, final_losses = _macro_losses(prediction - target, selected)
    by_id = dict(zip(final_ids, final_losses, strict=True))
    objective = float(
        np.mean(
            np.asarray([by_id[macro] for macro in retained_ranked], dtype=np.float64),
            dtype=np.float64,
        )
    )
    return CandidateEvaluation(
        arm,
        objective,
        prediction,
        final_gain,
        final_bias,
        retained_sorted,
    )


@dataclass(frozen=True, slots=True)
class OracleBatchRecord:
    ordinal: int
    indices: tuple[int, ...]
    shape: tuple[int, int, int]
    dtype: str
    sample_sha256: str
    batch_sha256: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < BATCH_COUNT:
            raise OracleV22Error("oracle batch ordinal is invalid")
        if type(self.indices) is not tuple or self.indices != ADMISSIBLE_BATCHES[self.ordinal]:
            raise OracleV22Error("oracle batch indices differ from the canonical partition")
        if type(self.shape) is not tuple or len(self.shape) != 3:
            raise OracleV22Error("oracle batch shape is invalid")
        expected_shape = (len(self.indices), CHANNELS, self.shape[2])
        if self.shape != expected_shape or self.shape[2] not in (SITE_COUNT, SITE_COUNT // 2):
            raise OracleV22Error("oracle batch shape is invalid")
        if self.dtype != "<f8":
            raise OracleV22Error("oracle batch dtype must be '<f8'")
        _require_sha256(self.sample_sha256, "sample SHA-256")
        _require_sha256(self.batch_sha256, "batch SHA-256")


@dataclass(frozen=True, slots=True)
class OracleSourceGrid:
    scope_sha256: str
    partition_sha256: str
    sample_stream_sha256: str
    content_sha256: str
    batch_records: tuple[OracleBatchRecord, ...]

    def __post_init__(self) -> None:
        for name in (
            "scope_sha256",
            "partition_sha256",
            "sample_stream_sha256",
            "content_sha256",
        ):
            _require_sha256(cast(str, getattr(self, name)), name)
        if (
            type(self.batch_records) is not tuple
            or len(self.batch_records) != BATCH_COUNT
            or any(not isinstance(record, OracleBatchRecord) for record in self.batch_records)
            or tuple(record.ordinal for record in self.batch_records) != tuple(range(BATCH_COUNT))
        ):
            raise OracleV22Error("oracle source grid does not contain 22 canonical batches")
        fit_counts = {record.shape[2] for record in self.batch_records}
        if len(fit_counts) != 1:
            raise OracleV22Error("oracle source batches disagree on fit-site count")
        if self.partition_sha256 != _partition_hash(self.scope_sha256):
            raise OracleV22Error("oracle source partition hash is inconsistent")
        if self.content_sha256 != _source_content_hash(
            self.scope_sha256, self.batch_records, self.sample_stream_sha256
        ):
            raise OracleV22Error("oracle source content hash is inconsistent")


@dataclass(frozen=True, slots=True)
class OracleObjectiveCache:
    arm: Arm
    objectives: np.ndarray
    gains: np.ndarray | None
    biases: np.ndarray | None
    retained_macro_ids: tuple[tuple[int, ...], ...]
    scope_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARM_BYTE:
            raise OracleV22Error("oracle objective-cache arm is invalid")
        _require_sha256(self.scope_sha256, "objective scope SHA-256")
        _require_sha256(self.content_sha256, "objective content SHA-256")
        if type(self.retained_macro_ids) is not tuple:
            raise OracleV22Error("oracle retained-ID cache must be an immutable tuple")
        objectives = _immutable_float64(self.objectives)
        if objectives.shape != (ADMISSIBLE_COUNT,) or bool(np.any(objectives < 0.0)):
            raise OracleV22Error("oracle objective cache is invalid")
        object.__setattr__(self, "objectives", objectives)
        if self.arm == "affine":
            if self.gains is not None or self.biases is not None or self.retained_macro_ids:
                raise OracleV22Error("affine cache carries appearance evidence")
            if self.content_sha256 != _objective_content_hash(
                self.arm,
                self.scope_sha256,
                objectives,
                None,
                None,
                (),
            ):
                raise OracleV22Error("affine objective content hash is inconsistent")
            return
        if self.gains is None or self.biases is None:
            raise OracleV22Error("combined cache lacks appearance evidence")
        gains = _immutable_float64(self.gains)
        biases = _immutable_float64(self.biases)
        if gains.shape != (ADMISSIBLE_COUNT, CHANNELS) or biases.shape != gains.shape:
            raise OracleV22Error("combined cache appearance arrays have the wrong shape")
        if bool(np.any(gains < GAIN_BOUNDS[0])) or bool(np.any(gains > GAIN_BOUNDS[1])):
            raise OracleV22Error("combined cache gains are outside the frozen bounds")
        if bool(np.any(biases < BIAS_BOUNDS[0])) or bool(np.any(biases > BIAS_BOUNDS[1])):
            raise OracleV22Error("combined cache biases are outside the frozen bounds")
        if type(self.retained_macro_ids) is not tuple or len(self.retained_macro_ids) != ADMISSIBLE_COUNT:
            raise OracleV22Error("combined cache retained-ID evidence has the wrong length")
        for values in self.retained_macro_ids:
            _validate_retained_ids(values, (14, 27))
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)
        if self.content_sha256 != _objective_content_hash(
            self.arm,
            self.scope_sha256,
            objectives,
            gains,
            biases,
            self.retained_macro_ids,
        ):
            raise OracleV22Error("combined objective content hash is inconsistent")


@dataclass(frozen=True, slots=True)
class OracleSelected:
    state_index: int
    admissible_rank: int
    parameters: np.ndarray
    objective: float
    gains: np.ndarray | None
    biases: np.ndarray | None
    retained_macro_ids: tuple[int, ...]
    fit_prediction: np.ndarray
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if type(self.state_index) is not int or not 0 <= self.state_index < STATE_COUNT:
            raise OracleV22Error("oracle selected state index is invalid")
        if type(self.admissible_rank) is not int or not 0 <= self.admissible_rank < ADMISSIBLE_COUNT:
            raise OracleV22Error("oracle selected admissible rank is invalid")
        if ADMISSIBLE_INDICES[self.admissible_rank] != self.state_index:
            raise OracleV22Error("oracle selected total and admissible ranks disagree")
        parameters = _immutable_float64(self.parameters)
        if parameters.shape != (6,) or parameters.tobytes(order="C") != CANONICAL_GRID[
            self.state_index
        ].tobytes(order="C"):
            raise OracleV22Error("oracle selected parameters differ from their canonical state")
        if (
            type(self.objective) is not float
            or not math.isfinite(self.objective)
            or self.objective < 0.0
            or type(self.retained_macro_ids) is not tuple
        ):
            raise OracleV22Error("oracle selected objective is invalid")
        prediction = _immutable_float64(self.fit_prediction)
        if prediction.shape not in ((CHANNELS, SITE_COUNT), (CHANNELS, SITE_COUNT // 2)):
            raise OracleV22Error("oracle selected fit prediction has the wrong shape")
        _require_sha256(self.evaluation_sha256, "selected evaluation SHA-256")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "fit_prediction", prediction)
        if self.gains is None:
            if self.biases is not None or self.retained_macro_ids:
                raise OracleV22Error("affine selection carries appearance values")
            return
        if self.biases is None:
            raise OracleV22Error("combined selection lacks biases")
        gains = _immutable_float64(self.gains)
        biases = _immutable_float64(self.biases)
        if gains.shape != (CHANNELS,) or biases.shape != gains.shape:
            raise OracleV22Error("selected appearance values have the wrong shape")
        if bool(np.any(gains < GAIN_BOUNDS[0])) or bool(np.any(gains > GAIN_BOUNDS[1])):
            raise OracleV22Error("selected gains are outside the frozen bounds")
        if bool(np.any(biases < BIAS_BOUNDS[0])) or bool(np.any(biases > BIAS_BOUNDS[1])):
            raise OracleV22Error("selected biases are outside the frozen bounds")
        _validate_retained_ids(self.retained_macro_ids, (14, 27))
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)


@dataclass(frozen=True, slots=True)
class OracleCertificate:
    protocol_sha256: str
    config_sha256: str
    candidate_order_sha256: str
    admissible_list_sha256: str
    invalid_bitmap_sha256: str
    geometry_sha256: str
    source_scope_sha256: str
    source_content_sha256: str
    objective_scope_sha256: str
    objective_content_sha256: str
    candidate_count: int
    admissible_count: int
    inadmissible_count: int
    selected_total_rank: int
    selected_admissible_rank: int
    exact_tie_multiplicity: int
    second_best_objective_gap: float
    second_best_nonflow_gap: float
    selected_evaluation_sha256: str
    selected_prediction_sha256: str
    scalar_replay_bit_exact: bool

    def __post_init__(self) -> None:
        hash_names = (
            "protocol_sha256",
            "config_sha256",
            "candidate_order_sha256",
            "admissible_list_sha256",
            "invalid_bitmap_sha256",
            "geometry_sha256",
            "source_scope_sha256",
            "source_content_sha256",
            "objective_scope_sha256",
            "objective_content_sha256",
            "selected_evaluation_sha256",
            "selected_prediction_sha256",
        )
        for name in hash_names:
            _require_sha256(cast(str, getattr(self, name)), name)
        if self.protocol_sha256 != PROTOCOL_SHA256:
            raise OracleV22Error("oracle certificate protocol hash is invalid")
        if (
            self.candidate_order_sha256,
            self.admissible_list_sha256,
            self.invalid_bitmap_sha256,
            self.geometry_sha256,
        ) != (
            CANDIDATE_ORDER_SHA256,
            ADMISSIBLE_LIST_SHA256,
            INVALID_BITMAP_SHA256,
            GEOMETRY_SHA256,
        ):
            raise OracleV22Error("oracle certificate frozen hashes are invalid")
        counts = (
            self.candidate_count,
            self.admissible_count,
            self.inadmissible_count,
        )
        if any(type(value) is not int for value in counts) or counts != (
            STATE_COUNT,
            ADMISSIBLE_COUNT,
            STATE_COUNT - ADMISSIBLE_COUNT,
        ):
            raise OracleV22Error("oracle certificate candidate counts are invalid")
        if (
            type(self.selected_total_rank) is not int
            or type(self.selected_admissible_rank) is not int
            or not 0 <= self.selected_admissible_rank < ADMISSIBLE_COUNT
            or ADMISSIBLE_INDICES[self.selected_admissible_rank] != self.selected_total_rank
        ):
            raise OracleV22Error("oracle certificate selected ranks are invalid")
        if (
            type(self.exact_tie_multiplicity) is not int
            or not 1 <= self.exact_tie_multiplicity <= ADMISSIBLE_COUNT
        ):
            raise OracleV22Error("oracle certificate exact-tie count is invalid")
        if (
            type(self.second_best_objective_gap) is not float
            or type(self.second_best_nonflow_gap) is not float
            or not math.isfinite(self.second_best_objective_gap)
            or not math.isfinite(self.second_best_nonflow_gap)
            or self.second_best_objective_gap < 0.0
            or self.second_best_nonflow_gap < 0.0
        ):
            raise OracleV22Error("oracle certificate gaps are invalid")
        if type(self.scalar_replay_bit_exact) is not bool or not self.scalar_replay_bit_exact:
            raise OracleV22Error("oracle selected scalar replay is not bit-exact")


@dataclass(frozen=True, slots=True)
class OracleResult:
    """Immutable exhaustive evidence from the standalone scalar implementation."""

    context_key: str
    arm: Arm
    source_grid: OracleSourceGrid
    objective_cache: OracleObjectiveCache
    selected: OracleSelected
    prediction: np.ndarray
    prediction_sha256: str
    certificate: OracleCertificate
    maximum_candidate_tensor_size: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        if not isinstance(self.context_key, str) or not self.context_key or len(self.context_key) > 512:
            raise OracleV22Error("oracle result context key is invalid")
        try:
            self.context_key.encode("ascii")
        except UnicodeEncodeError as error:
            raise OracleV22Error("oracle result context key must be ASCII") from error
        if (
            not isinstance(self.source_grid, OracleSourceGrid)
            or not isinstance(self.objective_cache, OracleObjectiveCache)
            or not isinstance(self.selected, OracleSelected)
            or not isinstance(self.certificate, OracleCertificate)
        ):
            raise OracleV22Error("oracle result contains an invalid evidence record")
        if self.arm not in _ARM_BYTE or self.arm != self.objective_cache.arm:
            raise OracleV22Error("oracle result arm is inconsistent")
        prediction = _immutable_float64(self.prediction)
        if prediction.shape not in ((CHANNELS, SITE_COUNT), (CHANNELS, SITE_COUNT // 2)):
            raise OracleV22Error("oracle result prediction has the wrong shape")
        object.__setattr__(self, "prediction", prediction)
        _require_sha256(self.prediction_sha256, "selected prediction SHA-256")
        fit_count = self.source_grid.batch_records[0].shape[2]
        if self.selected.fit_prediction.shape[1] != fit_count or prediction.shape[1] != fit_count:
            raise OracleV22Error("oracle result fit/output evidence has inconsistent site counts")
        position = self.selected.admissible_rank
        if not _same_float(self.selected.objective, float(self.objective_cache.objectives[position])):
            raise OracleV22Error("oracle selected objective differs from the complete cache")
        if self.arm == "combined":
            assert self.objective_cache.gains is not None and self.objective_cache.biases is not None
            expected_retained_count = 27 if fit_count == SITE_COUNT else 14
            if (
                self.selected.gains is None
                or self.selected.biases is None
                or len(self.selected.retained_macro_ids) != expected_retained_count
                or any(
                    len(values) != expected_retained_count
                    for values in self.objective_cache.retained_macro_ids
                )
                or self.selected.gains.tobytes(order="C")
                != self.objective_cache.gains[position].tobytes(order="C")
                or self.selected.biases.tobytes(order="C")
                != self.objective_cache.biases[position].tobytes(order="C")
                or self.selected.retained_macro_ids
                != self.objective_cache.retained_macro_ids[position]
            ):
                raise OracleV22Error("oracle selected appearance differs from the complete cache")
        selected_position, exact_ties, second_gap, nonflow_gap = _selection_diagnostics(
            self.objective_cache.objectives
        )
        if (
            selected_position != position
            or exact_ties != self.certificate.exact_tie_multiplicity
            or not _same_float(second_gap, self.certificate.second_best_objective_gap)
            or not _same_float(nonflow_gap, self.certificate.second_best_nonflow_gap)
        ):
            raise OracleV22Error("oracle result selection evidence is inconsistent")
        selected_entry = _entry_bytes(
            self.arm,
            self.selected.state_index,
            self.selected.objective,
            self.selected.gains,
            self.selected.biases,
            self.selected.retained_macro_ids,
        )
        evaluation_sha256 = hashlib.sha256(
            _SELECTED_EVALUATION_TAG
            + bytes.fromhex(self.objective_cache.scope_sha256)
            + selected_entry
        ).hexdigest()
        prediction_digest = hashlib.sha256()
        prediction_digest.update(_SELECTED_PREDICTION_TAG)
        prediction_digest.update(bytes.fromhex(self.objective_cache.scope_sha256))
        prediction_digest.update(struct.pack("<H", prediction.shape[1]))
        prediction_digest.update(prediction.tobytes(order="C"))
        if self.selected.evaluation_sha256 != evaluation_sha256:
            raise OracleV22Error("oracle selected evaluation hash is inconsistent")
        if self.prediction_sha256 != prediction_digest.hexdigest():
            raise OracleV22Error("oracle selected prediction hash is inconsistent")
        if (
            self.certificate.source_scope_sha256 != self.source_grid.scope_sha256
            or self.certificate.source_content_sha256 != self.source_grid.content_sha256
            or self.certificate.objective_scope_sha256 != self.objective_cache.scope_sha256
            or self.certificate.objective_content_sha256 != self.objective_cache.content_sha256
            or self.certificate.selected_total_rank != self.selected.state_index
            or self.certificate.selected_admissible_rank != position
            or self.certificate.selected_evaluation_sha256 != evaluation_sha256
            or self.certificate.selected_prediction_sha256 != self.prediction_sha256
        ):
            raise OracleV22Error("oracle result certificate is inconsistent")


def _source_scope(
    source: np.ndarray, fit_mask: np.ndarray, output_mask: np.ndarray, config_sha256: str
) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_SCOPE_TAG)
    digest.update(source.tobytes(order="C"))
    digest.update(np.asarray(fit_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(np.asarray(output_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(bytes.fromhex(CANDIDATE_ORDER_SHA256))
    digest.update(bytes.fromhex(ADMISSIBLE_LIST_SHA256))
    digest.update(struct.pack("<H", BATCH_SIZE))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


def _partition_hash(scope_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_PARTITION_TAG)
    digest.update(bytes.fromhex(scope_sha256))
    digest.update(struct.pack("<H", BATCH_COUNT))
    for ordinal, indices in enumerate(ADMISSIBLE_BATCHES):
        digest.update(struct.pack("<HH", ordinal, len(indices)))
        digest.update(struct.pack(f"<{len(indices)}H", *indices))
    return digest.hexdigest()


def _source_content_hash(
    scope_sha256: str,
    records: tuple[OracleBatchRecord, ...],
    sample_stream_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_CONTENT_TAG)
    digest.update(bytes.fromhex(scope_sha256))
    digest.update(struct.pack("<H", BATCH_COUNT))
    for record in records:
        digest.update(bytes.fromhex(record.batch_sha256))
    digest.update(bytes.fromhex(sample_stream_sha256))
    return digest.hexdigest()


def _batch_record(
    scope_sha256: str,
    ordinal: int,
    indices: tuple[int, ...],
    fit_site_count: int,
    sample_bytes: bytearray,
) -> OracleBatchRecord:
    digest = hashlib.sha256()
    digest.update(_SOURCE_BATCH_TAG)
    digest.update(bytes.fromhex(scope_sha256))
    digest.update(struct.pack("<HH", ordinal, len(indices)))
    digest.update(struct.pack(f"<{len(indices)}H", *indices))
    digest.update(struct.pack("<HI", CHANNELS, fit_site_count))
    digest.update(sample_bytes)
    return OracleBatchRecord(
        ordinal=ordinal,
        indices=indices,
        shape=(len(indices), CHANNELS, fit_site_count),
        dtype="<f8",
        sample_sha256=hashlib.sha256(sample_bytes).hexdigest(),
        batch_sha256=digest.hexdigest(),
    )


def _objective_scope(
    source_grid: OracleSourceGrid,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: Arm,
    config_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_OBJECTIVE_SCOPE_TAG)
    digest.update(bytes.fromhex(source_grid.scope_sha256))
    digest.update(bytes.fromhex(source_grid.content_sha256))
    digest.update(np.asarray(fit_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(np.asarray(output_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(struct.pack("<H", fit_target.shape[1]))
    digest.update(fit_target.tobytes(order="C"))
    digest.update(struct.pack("<B", _ARM_BYTE[arm]))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


def _entry_bytes(
    arm: Arm,
    state_index_value: int,
    objective: float | None,
    gains: np.ndarray | None = None,
    biases: np.ndarray | None = None,
    retained: tuple[int, ...] = (),
) -> bytes:
    if (
        arm not in _ARM_BYTE
        or type(state_index_value) is not int
        or not 0 <= state_index_value < STATE_COUNT
    ):
        raise OracleV22Error("objective entry metadata is invalid")
    payload = bytearray(struct.pack("<HB", state_index_value, 0 if objective is None else 1))
    if objective is None:
        if gains is not None or biases is not None or retained:
            raise OracleV22Error("invalid objective entry carries a valid payload")
        return bytes(payload)
    if type(objective) is not float or not math.isfinite(objective) or objective < 0.0:
        raise OracleV22Error("valid objective entry has an invalid objective")
    payload.extend(struct.pack("<d", objective))
    if arm == "combined":
        if gains is None or biases is None:
            raise OracleV22Error("combined objective entry lacks appearance evidence")
        gain_values = np.asarray(gains, dtype=_FLOAT64_LE, order="C")
        bias_values = np.asarray(biases, dtype=_FLOAT64_LE, order="C")
        if gain_values.shape != (CHANNELS,) or bias_values.shape != gain_values.shape:
            raise OracleV22Error("combined objective entry has malformed appearance vectors")
        _validate_retained_ids(retained, (14, 27))
        payload.extend(gain_values.tobytes(order="C"))
        payload.extend(bias_values.tobytes(order="C"))
        payload.extend(struct.pack("<B", len(retained)))
        payload.extend(bytes(retained))
    elif gains is not None or biases is not None or retained:
        raise OracleV22Error("affine objective entry carries appearance evidence")
    return bytes(payload)


def _objective_content_hash(
    arm: Arm,
    scope_sha256: str,
    objectives: np.ndarray,
    gains: np.ndarray | None,
    biases: np.ndarray | None,
    retained: tuple[tuple[int, ...], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_OBJECTIVE_CONTENT_TAG)
    digest.update(bytes.fromhex(scope_sha256))
    digest.update(struct.pack("<H", STATE_COUNT))
    valid_position = 0
    for total_index in range(STATE_COUNT):
        if total_index not in ADMISSIBLE_INDEX_SET:
            digest.update(_entry_bytes(arm, total_index, None))
            continue
        if arm == "affine":
            entry = _entry_bytes(arm, total_index, float(objectives[valid_position]))
        else:
            assert gains is not None and biases is not None
            entry = _entry_bytes(
                arm,
                total_index,
                float(objectives[valid_position]),
                gains[valid_position],
                biases[valid_position],
                retained[valid_position],
            )
        digest.update(entry)
        valid_position += 1
    if valid_position != ADMISSIBLE_COUNT:
        raise OracleV22Error("objective stream did not contain exactly 2,809 valid entries")
    return digest.hexdigest()


def _selection_diagnostics(objectives: np.ndarray) -> tuple[int, int, float, float]:
    order = sorted(
        range(ADMISSIBLE_COUNT),
        key=lambda position: (
            float(objectives[position]),
            _state_key(CANONICAL_STATES[ADMISSIBLE_INDICES[position]]),
        ),
    )
    selected_position = order[0]
    selected_objective = float(objectives[selected_position])
    exact_ties = int(np.count_nonzero(objectives == selected_objective))
    second_gap = float(objectives[order[1]] - selected_objective)
    selected_state = CANONICAL_STATES[ADMISSIBLE_INDICES[selected_position]]
    selected_dy, selected_dx = _flow(selected_state)
    nonflow_gap: float | None = None
    for position in order[1:]:
        other_dy, other_dx = _flow(CANONICAL_STATES[ADMISSIBLE_INDICES[position]])
        max_difference = max(
            float(np.max(np.abs(other_dy - selected_dy))),
            float(np.max(np.abs(other_dx - selected_dx))),
        )
        if max_difference > 1e-12:
            nonflow_gap = float(objectives[position] - selected_objective)
            break
    if nonflow_gap is None or second_gap < 0.0 or nonflow_gap < 0.0:
        raise OracleV22Error("oracle could not derive finite selection gaps")
    return selected_position, exact_ties, second_gap, nonflow_gap


def fit_scalar_oracle(
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: Arm,
    *,
    context_key: str,
    config_sha256: str,
) -> OracleResult:
    """Enumerate the exact grid with no cross-candidate scientific arithmetic."""

    validate_frozen_reconstruction()
    if arm not in _ARM_BYTE:
        raise OracleV22Error("arm must be exactly affine or combined")
    if not isinstance(context_key, str) or not context_key or len(context_key) > 512:
        raise OracleV22Error("context key must be a nonempty string of at most 512 characters")
    try:
        context_key.encode("ascii")
    except UnicodeEncodeError as error:
        raise OracleV22Error("context key must be ASCII") from error
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    current = _validate_source(source)
    fitted, output = _validate_context_masks(fit_mask, output_mask)
    target = _validate_fit_target(fit_target, int(np.count_nonzero(fitted)))

    source_scope = _source_scope(current, fitted, output, checked_config)
    partition_sha256 = _partition_hash(source_scope)
    sample_stream_digest = hashlib.sha256()
    sample_stream_digest.update(_SOURCE_SAMPLES_TAG)
    sample_stream_digest.update(bytes.fromhex(source_scope))

    objectives = np.empty(ADMISSIBLE_COUNT, dtype=_FLOAT64_LE)
    gains = np.empty((ADMISSIBLE_COUNT, CHANNELS), dtype=_FLOAT64_LE) if arm == "combined" else None
    biases = np.empty((ADMISSIBLE_COUNT, CHANNELS), dtype=_FLOAT64_LE) if arm == "combined" else None
    retained: list[tuple[int, ...]] = []
    batch_records: list[OracleBatchRecord] = []
    frame_indices: list[int] = []
    frame_samples = bytearray()
    admissible_position = 0

    best_key: tuple[float, tuple[float, float, float, float, float, float, float]] | None = None
    best_position: int | None = None
    best_fit: CandidateEvaluation | None = None

    for total_index, state in enumerate(CANONICAL_STATES):
        if not _is_admissible(state):
            continue
        if total_index != ADMISSIBLE_INDICES[admissible_position]:
            raise OracleV22Error("scalar enumeration diverged from the frozen admissible list")
        sampled = _sample_state(current, state, fitted)
        sample_bytes = sampled.tobytes(order="C")
        sample_stream_digest.update(sample_bytes)
        frame_indices.append(total_index)
        frame_samples.extend(sample_bytes)

        candidate = evaluate_candidate(arm, sampled, target, fitted)
        objectives[admissible_position] = candidate.objective
        if arm == "combined":
            assert gains is not None and biases is not None
            assert candidate.gains is not None and candidate.biases is not None
            gains[admissible_position] = candidate.gains
            biases[admissible_position] = candidate.biases
            retained.append(candidate.retained_macro_ids)
        selection_key = (candidate.objective, _state_key(state))
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_position = admissible_position
            best_fit = candidate

        admissible_position += 1
        if len(frame_indices) == BATCH_SIZE or admissible_position == ADMISSIBLE_COUNT:
            ordinal = len(batch_records)
            expected_indices = ADMISSIBLE_BATCHES[ordinal]
            if tuple(frame_indices) != expected_indices:
                raise OracleV22Error("scalar byte frame differs from the canonical partition")
            batch_records.append(
                _batch_record(
                    source_scope,
                    ordinal,
                    expected_indices,
                    sampled.shape[1],
                    frame_samples,
                )
            )
            frame_indices = []
            frame_samples = bytearray()

    if (
        admissible_position != ADMISSIBLE_COUNT
        or len(batch_records) != BATCH_COUNT
        or best_position is None
        or best_fit is None
    ):
        raise OracleV22Error("scalar enumeration was incomplete")

    sample_stream_sha256 = sample_stream_digest.hexdigest()
    batch_record_tuple = tuple(batch_records)
    source_content = _source_content_hash(source_scope, batch_record_tuple, sample_stream_sha256)
    source_grid = OracleSourceGrid(
        source_scope,
        partition_sha256,
        sample_stream_sha256,
        source_content,
        batch_record_tuple,
    )

    retained_tuple = tuple(retained)
    objective_scope = _objective_scope(source_grid, target, fitted, output, arm, checked_config)
    objective_content = _objective_content_hash(
        arm, objective_scope, objectives, gains, biases, retained_tuple
    )
    objective_cache = OracleObjectiveCache(
        arm,
        objectives,
        gains,
        biases,
        retained_tuple,
        objective_scope,
        objective_content,
    )

    selected_position, exact_ties, second_gap, nonflow_gap = _selection_diagnostics(
        objective_cache.objectives
    )
    if selected_position != best_position:
        raise OracleV22Error("streaming scalar argmin differs from complete scalar selection")
    selected_index = ADMISSIBLE_INDICES[selected_position]
    replay_sample = _sample_state(current, CANONICAL_STATES[selected_index], fitted)
    replay_fit = evaluate_candidate(arm, replay_sample, target, fitted)
    replay_matches = (
        _same_float(replay_fit.objective, best_fit.objective)
        and _same_float(
            replay_fit.objective, float(objective_cache.objectives[selected_position])
        )
        and replay_fit.prediction.tobytes(order="C")
        == best_fit.prediction.tobytes(order="C")
        and replay_fit.retained_macro_ids == best_fit.retained_macro_ids
    )
    if arm == "combined":
        assert replay_fit.gains is not None and replay_fit.biases is not None
        assert best_fit.gains is not None and best_fit.biases is not None
        replay_matches = (
            replay_matches
            and replay_fit.gains.tobytes(order="C") == best_fit.gains.tobytes(order="C")
            and replay_fit.biases.tobytes(order="C") == best_fit.biases.tobytes(order="C")
            and objective_cache.gains is not None
            and objective_cache.biases is not None
            and replay_fit.gains.tobytes(order="C")
            == objective_cache.gains[selected_position].tobytes(order="C")
            and replay_fit.biases.tobytes(order="C")
            == objective_cache.biases[selected_position].tobytes(order="C")
            and replay_fit.retained_macro_ids
            == objective_cache.retained_macro_ids[selected_position]
        )
    if not replay_matches:
        raise OracleV22Error("selected scalar replay differs bitwise from scalar enumeration")
    best_fit = replay_fit
    if arm == "affine":
        selected_entry = _entry_bytes(
            arm,
            selected_index,
            float(objective_cache.objectives[selected_position]),
        )
    else:
        assert objective_cache.gains is not None and objective_cache.biases is not None
        selected_entry = _entry_bytes(
            arm,
            selected_index,
            float(objective_cache.objectives[selected_position]),
            objective_cache.gains[selected_position],
            objective_cache.biases[selected_position],
            objective_cache.retained_macro_ids[selected_position],
        )
    evaluation_sha256 = hashlib.sha256(
        _SELECTED_EVALUATION_TAG + bytes.fromhex(objective_scope) + selected_entry
    ).hexdigest()
    selected = OracleSelected(
        selected_index,
        selected_position,
        CANONICAL_GRID[selected_index],
        best_fit.objective,
        best_fit.gains,
        best_fit.biases,
        best_fit.retained_macro_ids,
        best_fit.prediction,
        evaluation_sha256,
    )

    output_sample = _sample_state(current, CANONICAL_STATES[selected_index], output)
    if arm == "combined":
        assert selected.gains is not None and selected.biases is not None
        prediction = selected.gains[:, None] * output_sample + selected.biases[:, None]
    else:
        prediction = output_sample
    prediction = _immutable_float64(prediction)
    prediction_digest = hashlib.sha256()
    prediction_digest.update(_SELECTED_PREDICTION_TAG)
    prediction_digest.update(bytes.fromhex(objective_scope))
    prediction_digest.update(struct.pack("<H", prediction.shape[1]))
    prediction_digest.update(prediction.tobytes(order="C"))
    prediction_sha256 = prediction_digest.hexdigest()

    certificate = OracleCertificate(
        protocol_sha256=PROTOCOL_SHA256,
        config_sha256=checked_config,
        candidate_order_sha256=CANDIDATE_ORDER_SHA256,
        admissible_list_sha256=ADMISSIBLE_LIST_SHA256,
        invalid_bitmap_sha256=INVALID_BITMAP_SHA256,
        geometry_sha256=GEOMETRY_SHA256,
        source_scope_sha256=source_grid.scope_sha256,
        source_content_sha256=source_grid.content_sha256,
        objective_scope_sha256=objective_scope,
        objective_content_sha256=objective_content,
        candidate_count=STATE_COUNT,
        admissible_count=ADMISSIBLE_COUNT,
        inadmissible_count=STATE_COUNT - ADMISSIBLE_COUNT,
        selected_total_rank=selected_index,
        selected_admissible_rank=selected_position,
        exact_tie_multiplicity=exact_ties,
        second_best_objective_gap=second_gap,
        second_best_nonflow_gap=nonflow_gap,
        selected_evaluation_sha256=evaluation_sha256,
        selected_prediction_sha256=prediction_sha256,
        scalar_replay_bit_exact=True,
    )
    return OracleResult(
        context_key,
        arm,
        source_grid,
        objective_cache,
        selected,
        prediction,
        prediction_sha256,
        certificate,
    )


def _array_bytes_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _certificates_bit_exact(left: OracleCertificate, right: OracleCertificate) -> bool:
    return (
        left.protocol_sha256 == right.protocol_sha256
        and left.config_sha256 == right.config_sha256
        and left.candidate_order_sha256 == right.candidate_order_sha256
        and left.admissible_list_sha256 == right.admissible_list_sha256
        and left.invalid_bitmap_sha256 == right.invalid_bitmap_sha256
        and left.geometry_sha256 == right.geometry_sha256
        and left.source_scope_sha256 == right.source_scope_sha256
        and left.source_content_sha256 == right.source_content_sha256
        and left.objective_scope_sha256 == right.objective_scope_sha256
        and left.objective_content_sha256 == right.objective_content_sha256
        and left.candidate_count == right.candidate_count
        and left.admissible_count == right.admissible_count
        and left.inadmissible_count == right.inadmissible_count
        and left.selected_total_rank == right.selected_total_rank
        and left.selected_admissible_rank == right.selected_admissible_rank
        and left.exact_tie_multiplicity == right.exact_tie_multiplicity
        and _same_float(left.second_best_objective_gap, right.second_best_objective_gap)
        and _same_float(left.second_best_nonflow_gap, right.second_best_nonflow_gap)
        and left.selected_evaluation_sha256 == right.selected_evaluation_sha256
        and left.selected_prediction_sha256 == right.selected_prediction_sha256
        and left.scalar_replay_bit_exact == right.scalar_replay_bit_exact
    )


def _results_bit_exact(left: OracleResult, right: OracleResult) -> bool:
    return (
        left.context_key == right.context_key
        and left.arm == right.arm
        and left.source_grid == right.source_grid
        and left.objective_cache.arm == right.objective_cache.arm
        and left.objective_cache.scope_sha256 == right.objective_cache.scope_sha256
        and left.objective_cache.content_sha256 == right.objective_cache.content_sha256
        and _array_bytes_equal(left.objective_cache.objectives, right.objective_cache.objectives)
        and _array_bytes_equal(left.objective_cache.gains, right.objective_cache.gains)
        and _array_bytes_equal(left.objective_cache.biases, right.objective_cache.biases)
        and left.objective_cache.retained_macro_ids == right.objective_cache.retained_macro_ids
        and left.selected.state_index == right.selected.state_index
        and left.selected.admissible_rank == right.selected.admissible_rank
        and _array_bytes_equal(left.selected.parameters, right.selected.parameters)
        and _same_float(left.selected.objective, right.selected.objective)
        and _array_bytes_equal(left.selected.gains, right.selected.gains)
        and _array_bytes_equal(left.selected.biases, right.selected.biases)
        and left.selected.retained_macro_ids == right.selected.retained_macro_ids
        and _array_bytes_equal(left.selected.fit_prediction, right.selected.fit_prediction)
        and left.selected.evaluation_sha256 == right.selected.evaluation_sha256
        and _array_bytes_equal(left.prediction, right.prediction)
        and left.prediction_sha256 == right.prediction_sha256
        and _certificates_bit_exact(left.certificate, right.certificate)
        and left.maximum_candidate_tensor_size == right.maximum_candidate_tensor_size == 1
    )


def validate_oracle_result(
    result: OracleResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Deeply regenerate one persisted oracle result from original scientific inputs."""

    if not isinstance(result, OracleResult):
        raise OracleV22Error("deep oracle validation requires an OracleResult")
    rebuilt = fit_scalar_oracle(
        source,
        fit_target,
        fit_mask,
        output_mask,
        result.arm,
        context_key=result.context_key,
        config_sha256=config_sha256,
    )
    if not _results_bit_exact(result, rebuilt):
        raise OracleV22Error("persisted oracle result differs from a complete scalar rebuild")


__all__ = [
    "ADMISSIBLE_COUNT",
    "ADMISSIBLE_INDICES",
    "ADMISSIBLE_LIST_SHA256",
    "BATCH_COUNT",
    "BATCH_SIZE",
    "CANONICAL_GRID",
    "CANONICAL_STATES",
    "CANDIDATE_ORDER_SHA256",
    "CandidateEvaluation",
    "FULL_MASK",
    "GEOMETRY_SHA256",
    "INVALID_BITMAP_SHA256",
    "MACRO_IDS",
    "OracleBatchRecord",
    "OracleCertificate",
    "OracleObjectiveCache",
    "OracleResult",
    "OracleSelected",
    "OracleSourceGrid",
    "OracleV22Error",
    "PARITY_MASKS",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "STATE_COUNT",
    "evaluate_candidate",
    "fit_scalar_oracle",
    "sample_scalar",
    "state_index",
    "target_values",
    "validate_frozen_reconstruction",
    "validate_oracle_result",
]
