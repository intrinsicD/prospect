"""Pure support primitives for the frozen MM-008 v2.1 calibration protocol.

This module validates caller-supplied metadata and arrays.  It performs no file
I/O, owns no frozen nonce or seed, and never loads or generates real data.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from numbers import Integral, Real
from typing import Final, Literal, cast

import numpy as np

NONCE_SCHEMA_VERSION: Final = "mm008-v2-challenge-nonce-v1"
NONCE_REVIEWER_ID: Final = "/root/mm005_security_review:Hubble"
PROTOCOL_SHA256: Final = "6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f"
NONCE_RECEIPT_KEYS: Final = frozenset(
    {"created_at_utc", "nonce_hex", "protocol_sha256", "reviewer_id", "schema_version"}
)
CHALLENGE_LABEL_PREFIX: Final = "MM-008-v2-independent-challenge:"
CHALLENGE_COUNT: Final = 8

SYNTHETIC_ROWS: Final = 6
CHANNELS: Final = 3
NATIVE_SIZE: Final = 64
POOLED_SIZE: Final = 8
POOL_BLOCK: Final = NATIVE_SIZE // POOLED_SIZE
CENTRAL_START: Final = 8
CENTRAL_STOP: Final = 56
SCALE_FLOOR: Final = 1e-6
BROADBAND_RANK: Final = SYNTHETIC_ROWS * CHANNELS
MIN_SINGULAR_RATIO: Final = 0.50
MIN_CENTRAL_VARIANCE: Final = 0.50
MAX_ABS_LAG_CORRELATION: Final = 0.10

PERSISTENCE_FACTOR: Final = 1.25
PAIRING_FACTOR: Final = 1.10
ENDPOINT_TOLERANCE: Final = 1e-10
FORMAL_VIDEO_COUNT: Final = 8
CLEAN_FAILURE_MAX_SUPPORT: Final = 2

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_UTC_SECONDS: Final = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)

FactorialArm = Literal["affine", "appearance", "combined"]
IterativeArm = Literal["affine", "combined"]
SupportArm = Literal[
    "global_translation", "quadrant_translation", "affine", "appearance", "combined"
]
QLabel = Literal["S", "F", "R"]
QAggregation = Literal["all", "any"]

Q_LABEL_TIE_ORDER: Final[tuple[QLabel, ...]] = ("S", "F", "R")
SINGLETON_Q_LABELS: Final[tuple[QLabel, ...]] = ("S",)
ITERATIVE_Q_LABELS: Final[tuple[QLabel, ...]] = Q_LABEL_TIE_ORDER


def _require_lower_hex_64(value: str, name: str) -> None:
    if _LOWER_HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must contain exactly 64 lowercase hexadecimal characters")


def _require_utc_seconds(value: str) -> None:
    if _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError("created_at_utc must use exact whole-second RFC3339 UTC form")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("created_at_utc is not a valid Gregorian UTC timestamp") from error


@dataclass(frozen=True, slots=True)
class NonceReceipt:
    """One validated nonce receipt copied into the later formal marker."""

    created_at_utc: str
    nonce_hex: str
    protocol_sha256: str
    reviewer_id: str
    schema_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "created_at_utc": self.created_at_utc,
            "nonce_hex": self.nonce_hex,
            "protocol_sha256": self.protocol_sha256,
            "reviewer_id": self.reviewer_id,
            "schema_version": self.schema_version,
        }


def validate_nonce_receipt(
    value: object, *, expected_protocol_sha256: str
) -> NonceReceipt:
    """Validate exact receipt membership, values, and protocol binding."""

    _require_lower_hex_64(expected_protocol_sha256, "expected_protocol_sha256")
    if not isinstance(value, Mapping):
        raise ValueError("nonce receipt must be a JSON object")
    if set(value) != NONCE_RECEIPT_KEYS:
        raise ValueError("nonce receipt must contain exactly the five frozen keys")
    fields: dict[str, str] = {}
    for key in sorted(NONCE_RECEIPT_KEYS):
        item = value[key]
        if not isinstance(item, str):
            raise ValueError("every nonce receipt value must be a JSON string")
        fields[key] = item

    _require_utc_seconds(fields["created_at_utc"])
    _require_lower_hex_64(fields["nonce_hex"], "nonce_hex")
    _require_lower_hex_64(fields["protocol_sha256"], "protocol_sha256")
    if fields["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("nonce receipt protocol SHA-256 differs from the expected protocol")
    if fields["reviewer_id"] != NONCE_REVIEWER_ID:
        raise ValueError("nonce receipt reviewer identity differs")
    if fields["schema_version"] != NONCE_SCHEMA_VERSION:
        raise ValueError("nonce receipt schema version differs")
    return NonceReceipt(
        created_at_utc=fields["created_at_utc"],
        nonce_hex=fields["nonce_hex"],
        protocol_sha256=fields["protocol_sha256"],
        reviewer_id=fields["reviewer_id"],
        schema_version=fields["schema_version"],
    )


def canonical_nonce_receipt_bytes(receipt: NonceReceipt) -> bytes:
    """Return the exact canonical external receipt bytes, including one LF."""

    validated = validate_nonce_receipt(
        receipt.as_dict(), expected_protocol_sha256=receipt.protocol_sha256
    )
    payload = json.dumps(
        validated.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + b"\n"


def parse_canonical_nonce_receipt_bytes(
    payload: bytes, *, expected_protocol_sha256: str
) -> NonceReceipt:
    """Parse a receipt only when its complete bytes are already canonical."""

    if not isinstance(payload, bytes):
        raise ValueError("nonce receipt payload must be immutable bytes")
    try:
        decoded = payload.decode("ascii")
        value = cast(object, json.loads(decoded))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("nonce receipt bytes are not valid ASCII JSON") from error
    receipt = validate_nonce_receipt(value, expected_protocol_sha256=expected_protocol_sha256)
    if payload != canonical_nonce_receipt_bytes(receipt):
        raise ValueError("nonce receipt bytes are not canonical")
    return receipt


def nonce_receipt_sha256(receipt: NonceReceipt) -> str:
    """Hash the complete canonical receipt bytes, including the final LF."""

    return sha256(canonical_nonce_receipt_bytes(receipt)).hexdigest()


def derive_challenge_seed(protocol_sha256: str, nonce_hex: str, label_index: int) -> int:
    """Derive one protocol-bound uint64 challenge seed without instantiating an RNG."""

    _require_lower_hex_64(protocol_sha256, "protocol_sha256")
    _require_lower_hex_64(nonce_hex, "nonce_hex")
    if isinstance(label_index, bool) or not isinstance(label_index, Integral):
        raise ValueError("challenge label index must be an integer")
    index = int(label_index)
    if not 0 <= index < CHALLENGE_COUNT:
        raise ValueError("challenge label index is outside the frozen eight-label bank")
    digest = sha256(
        bytes.fromhex(protocol_sha256)
        + bytes.fromhex(nonce_hex)
        + f"{CHALLENGE_LABEL_PREFIX}{index}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def derive_challenge_seeds(receipt: NonceReceipt) -> tuple[int, ...]:
    """Derive the ordered seed bank only from the frozen protocol's receipt."""

    validate_nonce_receipt(receipt.as_dict(), expected_protocol_sha256=PROTOCOL_SHA256)
    return tuple(
        derive_challenge_seed(receipt.protocol_sha256, receipt.nonce_hex, index)
        for index in range(CHALLENGE_COUNT)
    )


def _as_finite_r64(value: np.ndarray, name: str, *, exact_rows: int | None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[1:] != (CHANNELS, NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError(f"{name} must have shape [N,3,64,64]")
    if exact_rows is not None and len(array) != exact_rows:
        raise ValueError(f"{name} must contain exactly {exact_rows} rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float64, order="C")


def area_pool_r8(value: np.ndarray) -> np.ndarray:
    """Area-pool finite R64 grids to R8 in float64."""

    array = _as_finite_r64(value, "R64 grid", exact_rows=None)
    if len(array) == 0:
        raise ValueError("R64 grid must contain at least one row")
    blocked = array.reshape(
        len(array), CHANNELS, POOLED_SIZE, POOL_BLOCK, POOLED_SIZE, POOL_BLOCK
    )
    return np.asarray(np.mean(blocked, axis=(3, 5), dtype=np.float64), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SourceOnlyNormalizer:
    """Current-only R8 channel normalizer used for R64 synthetic arrays."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        scale = np.asarray(self.scale, dtype=np.float64).copy()
        if mean.shape != (1, CHANNELS, 1, 1) or scale.shape != mean.shape:
            raise ValueError("source-only normalizer arrays must have shape [1,3,1,1]")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("source-only normalizer contains non-finite values")
        if np.any(scale < SCALE_FLOOR):
            raise ValueError("source-only normalizer scale is below the frozen floor")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def apply(self, value: np.ndarray) -> np.ndarray:
        array = _as_finite_r64(value, "normalizer input", exact_rows=None)
        return np.asarray((array - self.mean) / self.scale, dtype=np.float64)

    def invert(self, value: np.ndarray) -> np.ndarray:
        array = _as_finite_r64(value, "normalized input", exact_rows=None)
        output = np.asarray(array * self.scale + self.mean, dtype=np.float64)
        if not np.all(np.isfinite(output)):
            raise ValueError("normalizer inversion produced non-finite values")
        return output


def fit_source_only_normalizer(raw_source: np.ndarray) -> SourceOnlyNormalizer:
    """Fit the frozen channel statistics from one six-row source and nothing else."""

    source = _as_finite_r64(raw_source, "raw_source", exact_rows=SYNTHETIC_ROWS)
    pooled = area_pool_r8(source)
    mean = np.mean(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64)
    scale = np.maximum(
        np.std(pooled, axis=(0, 2, 3), keepdims=True, dtype=np.float64), SCALE_FLOOR
    )
    return SourceOnlyNormalizer(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
    )


def _finite_float_tuple(values: tuple[float, ...], name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class BroadbandMetrics:
    """Complete finite evidence for the frozen broadband generator checks."""

    row_rms: tuple[float, ...]
    matrix_rank: int
    singular_value_ratio: float
    central_variance: tuple[float, ...]
    lag_correlation: tuple[float, ...]
    lag_denominator_positive: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.row_rms) != BROADBAND_RANK:
            raise ValueError("broadband row RMS record must contain 18 values")
        if len(self.central_variance) != BROADBAND_RANK:
            raise ValueError("broadband central-variance record must contain 18 values")
        if len(self.lag_correlation) != 2 * BROADBAND_RANK:
            raise ValueError("broadband lag-correlation record must contain 36 values")
        if len(self.lag_denominator_positive) != 2 * BROADBAND_RANK or not all(
            isinstance(value, bool) for value in self.lag_denominator_positive
        ):
            raise ValueError("broadband lag-denominator record must contain 36 booleans")
        if isinstance(self.matrix_rank, bool) or not isinstance(self.matrix_rank, int):
            raise ValueError("broadband matrix rank must be an integer")
        if not 0 <= self.matrix_rank <= BROADBAND_RANK:
            raise ValueError("broadband matrix rank is outside its possible range")
        _finite_float_tuple(self.row_rms, "broadband row RMS")
        _finite_float_tuple(self.central_variance, "broadband central variance")
        _finite_float_tuple(self.lag_correlation, "broadband lag correlation")
        if not math.isfinite(self.singular_value_ratio):
            raise ValueError("broadband singular-value ratio must be finite")
        if any(value < 0.0 for value in self.row_rms):
            raise ValueError("broadband row RMS cannot be negative")
        if any(value < 0.0 for value in self.central_variance):
            raise ValueError("broadband central variance cannot be negative")
        if self.singular_value_ratio < 0.0:
            raise ValueError("broadband singular-value ratio cannot be negative")

    def failure_reasons(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not all(value > 0.0 for value in self.row_rms):
            failures.append("nonpositive_row_rms")
        if self.matrix_rank != BROADBAND_RANK:
            failures.append("matrix_rank")
        if not self.singular_value_ratio > MIN_SINGULAR_RATIO:
            failures.append("singular_value_ratio")
        if not all(value > MIN_CENTRAL_VARIANCE for value in self.central_variance):
            failures.append("central_variance")
        if not all(self.lag_denominator_positive):
            failures.append("lag_denominator")
        if not all(abs(value) < MAX_ABS_LAG_CORRELATION for value in self.lag_correlation):
            failures.append("lag_correlation")
        return tuple(failures)

    @property
    def valid(self) -> bool:
        return not self.failure_reasons()


def broadband_validity_metrics(normalized_base: np.ndarray) -> BroadbandMetrics:
    """Compute the exact rank, variance, and lag checks on one six-row R64 array."""

    base = _as_finite_r64(
        normalized_base, "normalized_base", exact_rows=SYNTHETIC_ROWS
    )
    rows = base.reshape(BROADBAND_RANK, NATIVE_SIZE * NATIVE_SIZE)
    centered = rows - np.mean(rows, axis=1, keepdims=True, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        row_rms_array = np.sqrt(
            np.mean(centered * centered, axis=1, dtype=np.float64)
        )
    if not np.all(np.isfinite(row_rms_array)):
        raise ValueError("broadband row RMS computation was non-finite")
    positive_rms = row_rms_array > 0.0
    scaled = np.zeros_like(centered)
    np.divide(centered, row_rms_array[:, None], out=scaled, where=positive_rms[:, None])
    try:
        matrix_rank = int(np.linalg.matrix_rank(scaled))
        singular_values = np.linalg.svd(scaled, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("broadband singular-value computation failed") from error
    largest = float(singular_values[0])
    singular_ratio = float(singular_values[-1] / largest) if largest > 0.0 else 0.0

    central = base[:, :, CENTRAL_START:CENTRAL_STOP, CENTRAL_START:CENTRAL_STOP]
    with np.errstate(over="ignore", invalid="ignore"):
        central_variance_array = np.var(
            central, axis=(2, 3), dtype=np.float64
        ).reshape(-1)
    if not np.all(np.isfinite(central_variance_array)) or not math.isfinite(
        singular_ratio
    ):
        raise ValueError("broadband variance or singular-value evidence was non-finite")

    correlations: list[float] = []
    denominator_flags: list[bool] = []
    for grid in base.reshape(BROADBAND_RANK, NATIVE_SIZE, NATIVE_SIZE):
        for left, right in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
            left_vector = left.reshape(-1)
            right_vector = right.reshape(-1)
            left_centered = left_vector - np.mean(left_vector, dtype=np.float64)
            right_centered = right_vector - np.mean(right_vector, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                left_square = float(
                    np.mean(left_centered * left_centered, dtype=np.float64)
                )
                right_square = float(
                    np.mean(right_centered * right_centered, dtype=np.float64)
                )
                numerator = float(
                    np.mean(left_centered * right_centered, dtype=np.float64)
                )
            if not all(math.isfinite(value) for value in (left_square, right_square, numerator)):
                raise ValueError("broadband lag-correlation evidence was non-finite")
            denominator_positive = left_square > 0.0 and right_square > 0.0
            denominator_flags.append(denominator_positive)
            correlation = (
                numerator / math.sqrt(left_square * right_square)
                if denominator_positive
                else 0.0
            )
            if not math.isfinite(correlation):
                raise ValueError("broadband lag correlation was non-finite")
            correlations.append(correlation)

    return BroadbandMetrics(
        row_rms=tuple(float(value) for value in row_rms_array),
        matrix_rank=matrix_rank,
        singular_value_ratio=singular_ratio,
        central_variance=tuple(float(value) for value in central_variance_array),
        lag_correlation=tuple(correlations),
        lag_denominator_positive=tuple(denominator_flags),
    )


def _finite_nonnegative(*values: object) -> tuple[float, ...] | None:
    output: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            return None
        output.append(number)
    return tuple(output)


def _require_support_arm(arm: str) -> SupportArm:
    if arm not in {
        "global_translation",
        "quadrant_translation",
        "affine",
        "appearance",
        "combined",
    }:
        raise ValueError("unknown MM-008 v2 support arm")
    return cast(SupportArm, arm)


def _require_factorial_arm(arm: str) -> FactorialArm:
    if arm not in {"affine", "appearance", "combined"}:
        raise ValueError("unknown MM-008 v2 factorial arm")
    return cast(FactorialArm, arm)


def _require_iterative_arm(arm: str) -> IterativeArm:
    if arm not in {"affine", "combined"}:
        raise ValueError("unknown MM-008 v2.1 iterative arm")
    return cast(IterativeArm, arm)


def applicable_q_labels(arm: SupportArm) -> tuple[QLabel, ...]:
    """Return the exact applicable panel labels for one frozen support arm."""

    checked_arm = _require_support_arm(arm)
    if checked_arm in {"affine", "combined"}:
        return ITERATIVE_Q_LABELS
    return SINGLETON_Q_LABELS


def residual_pairing(
    ordered_mse: float,
    wrong_mse: float,
    true_bias_mse: float,
    wrong_bias_mse: float,
) -> bool:
    """Return one matched-null residual pairing predicate."""

    values = _finite_nonnegative(ordered_mse, wrong_mse, true_bias_mse, wrong_bias_mse)
    if values is None:
        return False
    ordered, wrong, bias_true, bias_wrong = values
    left = PAIRING_FACTOR * ordered * bias_wrong
    right = wrong * bias_true
    return (
        bias_true > 0.0
        and bias_wrong > 0.0
        and math.isfinite(left)
        and math.isfinite(right)
        and left <= right
    )


def pair_support(
    ordered_mse: float,
    near_mse: float,
    far_mse: float,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    """Return the conjunction of near and far residualized pairing."""

    return residual_pairing(ordered_mse, near_mse, true_bias_mse, near_bias_mse) and residual_pairing(
        ordered_mse, far_mse, true_bias_mse, far_bias_mse
    )


def performance_support(persistence_mse: float, ordered_mse: float) -> bool:
    values = _finite_nonnegative(persistence_mse, ordered_mse)
    if values is None:
        return False
    persistence, ordered = values
    return persistence > 0.0 and PERSISTENCE_FACTOR * ordered <= persistence


def beats_bias(ordered_mse: float, true_bias_mse: float) -> bool:
    values = _finite_nonnegative(ordered_mse, true_bias_mse)
    if values is None:
        return False
    ordered, bias_true = values
    return bias_true > 0.0 and PERSISTENCE_FACTOR * ordered <= bias_true


def complete_support(
    arm: SupportArm,
    persistence_mse: float,
    ordered_mse: float,
    near_mse: float,
    far_mse: float,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    """Evaluate the synthetic per-row ``Complete(m)`` predicate."""

    checked_arm = _require_support_arm(arm)
    return (
        performance_support(persistence_mse, ordered_mse)
        and pair_support(
            ordered_mse,
            near_mse,
            far_mse,
            true_bias_mse,
            near_bias_mse,
            far_bias_mse,
        )
        and (
            checked_arm not in {"appearance", "combined"}
            or beats_bias(ordered_mse, true_bias_mse)
        )
    )


def x_support(
    arm: FactorialArm,
    persistence_mse: float,
    ordered_mse: float,
    true_bias_mse: float,
) -> bool:
    """Evaluate the family-specific real cross-fit ``X_m`` predicate."""

    checked_arm = _require_factorial_arm(arm)
    return performance_support(persistence_mse, ordered_mse) and (
        checked_arm == "affine" or beats_bias(ordered_mse, true_bias_mse)
    )


def f_support(
    arm: FactorialArm,
    persistence_mse: float,
    full_mse: float,
    true_bias_full_mse: float,
) -> bool:
    """Evaluate the family-specific real full-fit ``F_m`` predicate."""

    checked_arm = _require_factorial_arm(arm)
    return performance_support(persistence_mse, full_mse) and (
        checked_arm == "affine" or beats_bias(full_mse, true_bias_full_mse)
    )


def c_support(
    arm: FactorialArm,
    persistence_mse: float,
    ordered_mse: float,
    near_mse: float,
    far_mse: float,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    """Evaluate the family-specific real complete ``C_m`` predicate."""

    return x_support(arm, persistence_mse, ordered_mse, true_bias_mse) and pair_support(
        ordered_mse,
        near_mse,
        far_mse,
        true_bias_mse,
        near_bias_mse,
        far_bias_mse,
    )


def wrong_target_hit(
    persistence_mse: float, wrong_mse: float, matched_bias_mse: float
) -> bool:
    """Evaluate either matched near or matched far ``Hit`` predicate."""

    values = _finite_nonnegative(persistence_mse, wrong_mse, matched_bias_mse)
    if values is None:
        return False
    persistence, wrong, bias_wrong = values
    return (
        persistence > 0.0
        and bias_wrong > 0.0
        and PERSISTENCE_FACTOR * wrong <= persistence
        and PERSISTENCE_FACTOR * wrong <= bias_wrong
    )


def marginal(
    persistence_mse: float, appearance_mse: float, true_bias_mse: float
) -> bool:
    """Evaluate the appearance-only per-video ``Marginal`` diagnostic."""

    values = _finite_nonnegative(persistence_mse, appearance_mse, true_bias_mse)
    if values is None:
        return False
    persistence, appearance, bias_true = values
    return (
        persistence > 0.0
        and bias_true > 0.0
        and PERSISTENCE_FACTOR * bias_true <= persistence
        and PERSISTENCE_FACTOR * appearance <= persistence
        and PERSISTENCE_FACTOR * appearance > bias_true
    )


def dominates(
    preferred_full_mse: float,
    comparator_full_mse: float,
    preferred_xfit_mse: float,
    comparator_xfit_mse: float,
) -> bool:
    """Evaluate the strict synthetic conditional-dominance predicate."""

    values = _finite_nonnegative(
        preferred_full_mse,
        comparator_full_mse,
        preferred_xfit_mse,
        comparator_xfit_mse,
    )
    if values is None:
        return False
    preferred_full, comparator_full, preferred_xfit, comparator_xfit = values
    return (
        comparator_full > preferred_full
        and PERSISTENCE_FACTOR * preferred_full <= comparator_full
        and comparator_xfit > preferred_xfit
        and PERSISTENCE_FACTOR * preferred_xfit <= comparator_xfit
    )


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """A finite SSE/count/MSE triple with a recomputable denominator."""

    sse: float
    count: int
    mse: float

    def __post_init__(self) -> None:
        values = _finite_nonnegative(self.sse, self.mse)
        if values is None:
            raise ValueError("error record SSE and MSE must be finite and nonnegative")
        if isinstance(self.count, bool) or type(self.count) is not int or self.count <= 0:
            raise ValueError("error record count must be a positive Python integer")
        sse, mse = values
        expected = sse / self.count
        if not math.isfinite(expected) or not math.isclose(
            mse, expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("error record MSE does not reproduce from SSE/count")
        object.__setattr__(self, "sse", sse)
        object.__setattr__(self, "mse", mse)


@dataclass(frozen=True, slots=True)
class LabeledQError:
    """One recomputable error record under an applicable Q-panel label."""

    label: QLabel
    error: ErrorRecord

    def __post_init__(self) -> None:
        if self.label not in Q_LABEL_TIE_ORDER:
            raise ValueError("Q-panel label must be one of S, F, or R")
        if not isinstance(self.error, ErrorRecord):
            raise ValueError("Q-panel error must be an ErrorRecord")


@dataclass(frozen=True, slots=True)
class QEnvelope:
    """Canonical singleton or three-panel error envelope for one arm."""

    arm: SupportArm
    panels: tuple[LabeledQError, ...]

    def __post_init__(self) -> None:
        checked_arm = _require_support_arm(self.arm)
        if not isinstance(self.panels, tuple):
            raise ValueError("Q-envelope panels must be an immutable tuple")
        expected = applicable_q_labels(checked_arm)
        labels = tuple(panel.label for panel in self.panels)
        if len(labels) != len(set(labels)) or set(labels) != set(expected):
            raise ValueError("Q-envelope panel membership differs from the arm contract")
        by_label = {panel.label: panel for panel in self.panels}
        canonical = tuple(by_label[label] for label in expected)
        if len({panel.error.count for panel in canonical}) != 1:
            raise ValueError("Q-envelope panel counts must agree")
        object.__setattr__(self, "arm", checked_arm)
        object.__setattr__(self, "panels", canonical)

    @property
    def labels(self) -> tuple[QLabel, ...]:
        return tuple(panel.label for panel in self.panels)

    @property
    def minimum(self) -> LabeledQError:
        """Return the minimum MSE with frozen S,F,R tie-breaking."""

        tie_rank = {label: index for index, label in enumerate(Q_LABEL_TIE_ORDER)}
        return min(
            self.panels,
            key=lambda panel: (panel.error.mse, tie_rank[panel.label]),
        )

    @property
    def minimum_label(self) -> QLabel:
        return self.minimum.label

    @property
    def minimum_mse(self) -> float:
        return self.minimum.error.mse

    def by_label(self) -> dict[QLabel, ErrorRecord]:
        return {panel.label: panel.error for panel in self.panels}


def q_envelope(
    arm: SupportArm, panels: Mapping[str, ErrorRecord]
) -> QEnvelope:
    """Build an exact-membership Q envelope and canonicalize panel order."""

    checked_arm = _require_support_arm(arm)
    if not isinstance(panels, Mapping):
        raise ValueError("Q-envelope input must be a label-to-error mapping")
    records: list[LabeledQError] = []
    for label, error in panels.items():
        if label not in Q_LABEL_TIE_ORDER:
            raise ValueError("Q-envelope contains an unknown panel label")
        records.append(LabeledQError(cast(QLabel, label), error))
    return QEnvelope(checked_arm, tuple(records))


@dataclass(frozen=True, slots=True)
class LabeledQDecision:
    """One panel-specific predicate bit."""

    label: QLabel
    passed: bool

    def __post_init__(self) -> None:
        if self.label not in Q_LABEL_TIE_ORDER:
            raise ValueError("Q-decision label must be one of S, F, or R")
        if type(self.passed) is not bool:
            raise ValueError("Q-decision value must be a Python boolean")


@dataclass(frozen=True, slots=True)
class QPredicateRecord:
    """Panel-specific predicate evidence plus its frozen all/any reduction."""

    envelope: QEnvelope
    decisions: tuple[LabeledQDecision, ...]
    aggregation: QAggregation

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, QEnvelope):
            raise ValueError("Q-predicate envelope is invalid")
        if not isinstance(self.decisions, tuple):
            raise ValueError("Q-predicate decisions must be an immutable tuple")
        if self.aggregation not in {"all", "any"}:
            raise ValueError("Q-predicate aggregation must be all or any")
        labels = tuple(item.label for item in self.decisions)
        if len(labels) != len(set(labels)) or set(labels) != set(self.envelope.labels):
            raise ValueError("Q-predicate decision membership differs from its envelope")
        by_label = {item.label: item for item in self.decisions}
        object.__setattr__(
            self,
            "decisions",
            tuple(by_label[label] for label in self.envelope.labels),
        )

    @property
    def passed(self) -> bool:
        values = tuple(item.passed for item in self.decisions)
        return all(values) if self.aggregation == "all" else any(values)

    def by_label(self) -> dict[QLabel, bool]:
        return {item.label: item.passed for item in self.decisions}


def q_residual_pairing(
    ordered_mse: float,
    wrong: QEnvelope,
    true_bias_mse: float,
    wrong_bias_mse: float,
) -> QPredicateRecord:
    """Require residual pairing separately for every applicable Q panel."""

    decisions = tuple(
        LabeledQDecision(
            panel.label,
            residual_pairing(
                ordered_mse,
                panel.error.mse,
                true_bias_mse,
                wrong_bias_mse,
            ),
        )
        for panel in wrong.panels
    )
    return QPredicateRecord(wrong, decisions, "all")


def q_wrong_target_hit(
    persistence_mse: float,
    wrong: QEnvelope,
    matched_bias_mse: float,
) -> QPredicateRecord:
    """Reduce applicable panel hits with OR so a target counts at most once."""

    decisions = tuple(
        LabeledQDecision(
            panel.label,
            wrong_target_hit(persistence_mse, panel.error.mse, matched_bias_mse),
        )
        for panel in wrong.panels
    )
    return QPredicateRecord(wrong, decisions, "any")


@dataclass(frozen=True, slots=True)
class QPairingRecord:
    """Near/far Q-envelope pairing evidence for one arm and scoring unit."""

    near: QPredicateRecord
    far: QPredicateRecord

    def __post_init__(self) -> None:
        if self.near.aggregation != "all" or self.far.aggregation != "all":
            raise ValueError("Q pairing requires conjunction records")
        if self.near.envelope.arm != self.far.envelope.arm:
            raise ValueError("near/far Q-envelope arms differ")

    @property
    def passed(self) -> bool:
        return self.near.passed and self.far.passed


def q_pair_support(
    ordered_mse: float,
    near: QEnvelope,
    far: QEnvelope,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> QPairingRecord:
    """Evaluate the v2.1 near/far residual-pairing conjunction."""

    if near.arm != far.arm:
        raise ValueError("near/far Q-envelope arms differ")
    return QPairingRecord(
        near=q_residual_pairing(
            ordered_mse, near, true_bias_mse, near_bias_mse
        ),
        far=q_residual_pairing(
            ordered_mse, far, true_bias_mse, far_bias_mse
        ),
    )


def q_complete_support(
    arm: SupportArm,
    persistence_mse: float,
    ordered_mse: float,
    near: QEnvelope,
    far: QEnvelope,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    """Evaluate synthetic ``Complete(m)`` with the applicable Q envelope."""

    checked_arm = _require_support_arm(arm)
    if near.arm != checked_arm or far.arm != checked_arm:
        raise ValueError("Q-envelope arm differs from the Complete arm")
    pairing = q_pair_support(
        ordered_mse,
        near,
        far,
        true_bias_mse,
        near_bias_mse,
        far_bias_mse,
    )
    return (
        performance_support(persistence_mse, ordered_mse)
        and pairing.passed
        and (
            checked_arm not in {"appearance", "combined"}
            or beats_bias(ordered_mse, true_bias_mse)
        )
    )


def q_c_support(
    arm: FactorialArm,
    persistence_mse: float,
    ordered_mse: float,
    near: QEnvelope,
    far: QEnvelope,
    true_bias_mse: float,
    near_bias_mse: float,
    far_bias_mse: float,
) -> bool:
    """Evaluate real ``C_m`` with the applicable Q envelope."""

    checked_arm = _require_factorial_arm(arm)
    if near.arm != checked_arm or far.arm != checked_arm:
        raise ValueError("Q-envelope arm differs from the real family arm")
    pairing = q_pair_support(
        ordered_mse,
        near,
        far,
        true_bias_mse,
        near_bias_mse,
        far_bias_mse,
    )
    return x_support(
        checked_arm, persistence_mse, ordered_mse, true_bias_mse
    ) and pairing.passed


@dataclass(frozen=True, slots=True)
class IterativeStartCertificationRecord:
    """Dual-start validity and preemption for one iterative fit context."""

    arm: IterativeArm
    forward_certified: bool
    reverse_certified: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _require_iterative_arm(self.arm))
        if type(self.forward_certified) is not bool or type(self.reverse_certified) is not bool:
            raise ValueError("iterative start certificates must be Python booleans")

    @property
    def preempts(self) -> bool:
        return not (self.forward_certified and self.reverse_certified)

    @property
    def failure_codes(self) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.forward_certified:
            failures.append("forward_terminal_certificate")
        if not self.reverse_certified:
            failures.append("reverse_terminal_certificate")
        return tuple(failures)


@dataclass(frozen=True, slots=True)
class EndpointRecord:
    """One finite per-row, per-context endpoint error record."""

    sse: float
    count: int
    mse: float
    max_abs_error: float

    def __post_init__(self) -> None:
        score = ErrorRecord(self.sse, self.count, self.mse)
        maximum = _finite_nonnegative(self.max_abs_error)
        if maximum is None:
            raise ValueError("endpoint maximum error must be finite and nonnegative")
        object.__setattr__(self, "sse", score.sse)
        object.__setattr__(self, "count", score.count)
        object.__setattr__(self, "mse", score.mse)
        object.__setattr__(self, "max_abs_error", maximum[0])

    def passes(self, tolerance: float = ENDPOINT_TOLERANCE) -> bool:
        values = _finite_nonnegative(tolerance)
        return values is not None and self.max_abs_error <= values[0]


def error_record(actual: np.ndarray, expected: np.ndarray) -> ErrorRecord:
    """Construct a finite score record from equal, nonempty arrays."""

    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("error-record arrays must have equal shape")
    if left.size == 0:
        raise ValueError("error-record arrays cannot be empty")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("error-record arrays must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left - right
        sse = float(np.sum(difference * difference, dtype=np.float64))
    if not math.isfinite(sse):
        raise ValueError("error-record SSE is non-finite")
    count = int(left.size)
    return ErrorRecord(sse=sse, count=count, mse=sse / count)


def endpoint_record(actual: np.ndarray, expected: np.ndarray) -> EndpointRecord:
    """Construct one endpoint record without aggregating rows or fit contexts."""

    score = error_record(actual, expected)
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    maximum = float(np.max(np.abs(left - right)))
    if not math.isfinite(maximum):
        raise ValueError("endpoint maximum error is non-finite")
    return EndpointRecord(
        sse=score.sse,
        count=score.count,
        mse=score.mse,
        max_abs_error=maximum,
    )


@dataclass(frozen=True, slots=True)
class EndpointErrorRecord:
    """Required finite endpoint errors for full and both parity fits."""

    full: float
    parity0: float
    parity1: float

    def __post_init__(self) -> None:
        values = _finite_nonnegative(self.full, self.parity0, self.parity1)
        if values is None:
            raise ValueError("endpoint errors must be finite and nonnegative")
        object.__setattr__(self, "full", values[0])
        object.__setattr__(self, "parity0", values[1])
        object.__setattr__(self, "parity1", values[2])

    def passes(self, tolerance: float = ENDPOINT_TOLERANCE) -> bool:
        values = _finite_nonnegative(tolerance)
        if values is None:
            return False
        threshold = values[0]
        return max(self.full, self.parity0, self.parity1) <= threshold


@dataclass(frozen=True, slots=True)
class DecisionCountRecord:
    """Finite exact eight-video counts consumed by ``CleanFail(m)``."""

    x_support: int
    complete_support: int
    full_support: int
    hit_near: int
    hit_far: int
    marginal: int
    video_count: int = FORMAL_VIDEO_COUNT

    def __post_init__(self) -> None:
        names = (
            "x_support",
            "complete_support",
            "full_support",
            "hit_near",
            "hit_far",
            "marginal",
            "video_count",
        )
        converted: dict[str, int] = {}
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} count must be an integer")
            converted[name] = int(value)
        if converted["video_count"] != FORMAL_VIDEO_COUNT:
            raise ValueError("decision counts must describe exactly eight formal videos")
        for name in names[:-1]:
            if not 0 <= converted[name] <= FORMAL_VIDEO_COUNT:
                raise ValueError(f"{name} count is outside [0,8]")
        if converted["complete_support"] > converted["x_support"]:
            raise ValueError("complete-support count cannot exceed X-support count")
        for name, value in converted.items():
            object.__setattr__(self, name, value)


def clean_fail(
    arm: FactorialArm,
    counts: DecisionCountRecord,
    *,
    denominators_valid: bool,
    controls_valid: bool,
    optimizer_preemption: bool = False,
    order_preemption: bool = False,
    boundary_preemption: bool = False,
    range_preemption: bool = False,
) -> bool:
    """Evaluate the exact per-family ``CleanFail(m)`` aggregate predicate."""

    checked_arm = _require_factorial_arm(arm)
    flags = (
        denominators_valid,
        controls_valid,
        optimizer_preemption,
        order_preemption,
        boundary_preemption,
        range_preemption,
    )
    if not all(isinstance(value, bool) for value in flags):
        return False
    return (
        counts.x_support <= CLEAN_FAILURE_MAX_SUPPORT
        and counts.complete_support <= CLEAN_FAILURE_MAX_SUPPORT
        and counts.full_support <= CLEAN_FAILURE_MAX_SUPPORT
        and counts.hit_near <= CLEAN_FAILURE_MAX_SUPPORT
        and counts.hit_far <= CLEAN_FAILURE_MAX_SUPPORT
        and denominators_valid
        and controls_valid
        and not optimizer_preemption
        and not order_preemption
        and not boundary_preemption
        and not range_preemption
        and (checked_arm != "appearance" or counts.marginal <= CLEAN_FAILURE_MAX_SUPPORT)
    )
