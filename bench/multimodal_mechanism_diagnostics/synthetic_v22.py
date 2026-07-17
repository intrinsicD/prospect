"""Pure frozen synthetic generators for MM-008 v2.2.

Ordinary seeds are declarations only.  Generation always requires an explicit
caller-injected uint64 seed; this module performs no I/O, nonce derivation, fitting,
scoring, or import-time generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-synthetic-v1"

if calibration.PROTOCOL_SHA256 != PROTOCOL_SHA256 or geometry.PROTOCOL_SHA256 != PROTOCOL_SHA256:
    raise RuntimeError("MM-008 v2.2 synthetic dependencies bind a different protocol")

Scenario = Literal[
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "coupled_boundary",
    "constant_target",
]
StateValues = tuple[float, float, float, float, float, float]

SCENARIOS: Final[tuple[Scenario, ...]] = (
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "coupled_boundary",
    "constant_target",
)

FROZEN_SEED_MAP: Final[Mapping[Scenario, int]] = MappingProxyType(
    {
        "translation": 820_800,
        "affine": 820_801,
        "appearance": 820_802,
        "combined": 820_803,
        "stationary": 820_804,
        "independent": 820_805,
        "coupled_boundary": 820_806,
        "constant_target": 820_807,
    }
)

_FLOAT64_LE: Final = np.dtype("<f8")
_INT64_LE: Final = np.dtype("<i8")
_BANK_SHAPE: Final = (
    calibration.SYNTHETIC_ROWS,
    calibration.CHANNELS,
    calibration.NATIVE_SIZE,
    calibration.NATIVE_SIZE,
)
_ROW_SHAPE: Final = (
    calibration.CHANNELS,
    calibration.NATIVE_SIZE,
    calibration.NATIVE_SIZE,
)


class SyntheticV22Error(ValueError):
    """Raised when a supplied synthetic value violates the frozen v2.2 contract."""


def _immutable_array(value: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _FLOAT64_LE or not bool(np.all(np.isfinite(array))):
        raise SyntheticV22Error("synthetic arrays must contain finite little-endian float64 values")
    return _immutable_array(array, _FLOAT64_LE)


def _immutable_int64(value: np.ndarray) -> np.ndarray:
    return _immutable_array(np.asarray(value), _INT64_LE)


def _array_bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _require_scenario(scenario: str) -> Scenario:
    if type(scenario) is not str or scenario not in SCENARIOS:
        raise SyntheticV22Error("unknown MM-008 v2.2 synthetic scenario")
    return cast(Scenario, scenario)


def _require_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise SyntheticV22Error("synthetic seed must be an integer")
    value = int(seed)
    if not 0 <= value < 2**64:
        raise SyntheticV22Error("synthetic seed is outside the uint64 range")
    return value


def _finite_tuple(values: Sequence[float], length: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise SyntheticV22Error(f"{label} must contain exactly {length} finite values")
    try:
        array = np.asarray(values, dtype=_FLOAT64_LE)
    except (TypeError, ValueError) as error:
        raise SyntheticV22Error(f"{label} must contain exactly {length} finite values") from error
    if array.shape != (length,) or not bool(np.all(np.isfinite(array))):
        raise SyntheticV22Error(f"{label} must contain exactly {length} finite values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class TransformTruth:
    """One exact affine and per-channel appearance endpoint."""

    theta: StateValues
    gains: tuple[float, float, float]
    biases: tuple[float, float, float]

    def __post_init__(self) -> None:
        theta = cast(StateValues, _finite_tuple(self.theta, 6, "theta"))
        gains = cast(tuple[float, float, float], _finite_tuple(self.gains, 3, "gains"))
        biases = cast(tuple[float, float, float], _finite_tuple(self.biases, 3, "biases"))
        try:
            geometry.state_index(theta)
        except geometry.GeometryValidationError as error:
            raise SyntheticV22Error("truth theta must be an exact canonical affine state") from error
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)

    def theta_array(self) -> np.ndarray:
        return _immutable_float64(np.asarray(self.theta, dtype=_FLOAT64_LE))

    def gain_array(self) -> np.ndarray:
        return _immutable_float64(np.asarray(self.gains, dtype=_FLOAT64_LE))

    def bias_array(self) -> np.ndarray:
        return _immutable_float64(np.asarray(self.biases, dtype=_FLOAT64_LE))


IDENTITY_TRUTH: Final = TransformTruth(
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 0.0),
)

TRUTHS: Final[Mapping[Scenario, TransformTruth | None]] = MappingProxyType(
    {
        "translation": TransformTruth(
            (4.0, -4.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
        "affine": TransformTruth(
            (0.0, 0.0, 2.0, 0.0, 0.0, -2.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
        "appearance": TransformTruth(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.25, 0.75, 1.5),
            (0.35, -0.25, 0.15),
        ),
        "combined": TransformTruth(
            (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0),
            (1.2, 0.8, 1.4),
            (0.3, -0.2, 0.1),
        ),
        "stationary": IDENTITY_TRUTH,
        "independent": None,
        "coupled_boundary": TransformTruth(
            (4.0, 0.0, 4.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
        ),
        "constant_target": None,
    }
)


def _freeze_bank(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != _BANK_SHAPE:
        raise SyntheticV22Error(f"{label} must have shape [6,3,64,64]")
    return _immutable_float64(array)


def _freeze_row(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != _ROW_SHAPE:
        raise SyntheticV22Error(f"{label} must have shape [3,64,64]")
    return _immutable_float64(array)


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    """One immutable six-row synthetic source/target bank and validity evidence."""

    scenario: Scenario
    seed: int
    raw_source: np.ndarray
    raw_target: np.ndarray
    normalized_base: np.ndarray
    source: np.ndarray
    target: np.ndarray
    normalizer: calibration.SourceOnlyNormalizer
    source_broadband: calibration.BroadbandMetrics
    independent_target_broadband: calibration.BroadbandMetrics | None
    truth: TransformTruth | None

    def __post_init__(self) -> None:
        scenario = _require_scenario(self.scenario)
        seed = _require_seed(self.seed)
        if not isinstance(self.normalizer, calibration.SourceOnlyNormalizer):
            raise SyntheticV22Error("case normalizer is not a source-only normalizer")
        if not isinstance(self.source_broadband, calibration.BroadbandMetrics):
            raise SyntheticV22Error("case source broadband evidence is invalid")
        if self.independent_target_broadband is not None and not isinstance(
            self.independent_target_broadband, calibration.BroadbandMetrics
        ):
            raise SyntheticV22Error("case independent-target broadband evidence is invalid")
        if self.truth is not None and not isinstance(self.truth, TransformTruth):
            raise SyntheticV22Error("case truth is invalid")
        if scenario in {"independent", "constant_target"} and self.truth is not None:
            raise SyntheticV22Error("target-independent scenarios must not declare a truth")
        if scenario not in {"independent", "constant_target"} and self.truth is None:
            raise SyntheticV22Error("positive/stationary scenarios must declare a truth")
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "seed", seed)
        for name in ("raw_source", "raw_target", "normalized_base", "source", "target"):
            object.__setattr__(self, name, _freeze_bank(getattr(self, name), name))

    def generator_failure_codes(self) -> tuple[str, ...]:
        """Return stable fail-closed generator-invalidity codes without redrawing."""

        failures = [f"source_{code}" for code in self.source_broadband.failure_reasons()]
        normalized_source = self.normalizer.apply(self.raw_source)
        if not _array_bits_equal(normalized_source, self.normalized_base):
            failures.append("normalized_base_source_normalizer_mismatch")
        recomputed_source = calibration.broadband_validity_metrics(self.normalized_base)
        if recomputed_source != self.source_broadband:
            failures.append("source_broadband_evidence_mismatch")

        if self.scenario == "independent":
            if self.independent_target_broadband is None:
                failures.append("independent_target_metrics_missing")
            else:
                failures.extend(
                    f"independent_target_{code}"
                    for code in self.independent_target_broadband.failure_reasons()
                )
                recomputed_target = calibration.broadband_validity_metrics(self.target)
                if recomputed_target != self.independent_target_broadband:
                    failures.append("independent_target_broadband_evidence_mismatch")
        elif self.independent_target_broadband is not None:
            failures.append("unexpected_independent_target_metrics")

        if self.scenario == "stationary":
            if not _array_bits_equal(self.raw_source, self.raw_target):
                failures.append("stationary_raw_copy")
            if not _array_bits_equal(self.source, self.target):
                failures.append("stationary_normalized_copy")
        elif not _array_bits_equal(self.normalizer.apply(self.raw_target), self.target):
            failures.append("target_source_normalizer_mismatch")
        return tuple(failures)


@dataclass(frozen=True, slots=True)
class SyntheticRowTargets:
    """One row's source and deterministic true/near/far full-frame targets."""

    row: int
    source: np.ndarray
    true_target: np.ndarray
    near_row: int
    near_target: np.ndarray
    far_row: int
    far_target: np.ndarray

    def __post_init__(self) -> None:
        for name in ("row", "near_row", "far_row"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < calibration.SYNTHETIC_ROWS:
                raise SyntheticV22Error(f"{name} must be a built-in row index in [0,5]")
        if self.near_row == self.row or self.far_row == self.row:
            raise SyntheticV22Error("near/far rows must be deranged from the true row")
        for name in ("source", "true_target", "near_target", "far_target"):
            object.__setattr__(self, name, _freeze_row(getattr(self, name), name))


def _draw_raw(rng: np.random.Generator) -> np.ndarray:
    epsilon = rng.standard_normal(size=_BANK_SHAPE)
    offsets = rng.standard_normal(
        size=(calibration.SYNTHETIC_ROWS, calibration.CHANNELS, 1, 1)
    )
    return np.ascontiguousarray(epsilon + 0.50 * offsets, dtype=_FLOAT64_LE)


def transform_central(source: np.ndarray, truth: TransformTruth) -> np.ndarray:
    """Apply one truth only at central sites using the frozen scalar sampler."""

    bank = _freeze_bank(source, "source")
    if not isinstance(truth, TransformTruth):
        raise SyntheticV22Error("central transform requires a TransformTruth")
    state = geometry.state_index(truth.theta)
    gains = truth.gain_array()[:, None]
    biases = truth.bias_array()[:, None]
    target = np.array(bank, dtype=_FLOAT64_LE, order="C", copy=True)
    coords = geometry.GEOMETRY.coords.astype(np.intp)
    for row in range(calibration.SYNTHETIC_ROWS):
        sampled = geometry.sample_scalar(bank[row], state, geometry.FULL_MASK)
        transformed = gains * sampled + biases
        row_target = target[row]
        row_target[:, coords[:, 0], coords[:, 1]] = transformed
    return _freeze_bank(target, "transformed target")


def generate_case(
    scenario: Scenario,
    *,
    seed: int,
) -> SyntheticCase:
    """Generate one exact six-row v2.2 case from an explicitly injected seed."""

    checked_scenario = _require_scenario(scenario)
    checked_seed = _require_seed(seed)

    rng = np.random.Generator(np.random.PCG64(checked_seed))
    raw_source = _draw_raw(rng)
    normalizer = calibration.fit_source_only_normalizer(raw_source)
    normalized_base = normalizer.apply(raw_source)
    source_metrics = calibration.broadband_validity_metrics(normalized_base)
    truth = TRUTHS[checked_scenario]
    target_metrics: calibration.BroadbandMetrics | None = None

    if checked_scenario == "coupled_boundary":
        half_plane = (np.arange(calibration.NATIVE_SIZE) >= 32).astype(_FLOAT64_LE)
        source = np.asarray(normalized_base * half_plane[None, None, :, None], dtype=_FLOAT64_LE)
    else:
        source = np.array(normalized_base, dtype=_FLOAT64_LE, order="C", copy=True)

    if checked_scenario == "stationary":
        raw_target = raw_source.copy(order="C")
        target = source.copy(order="C")
    elif checked_scenario == "independent":
        raw_target = _draw_raw(rng)
        target = normalizer.apply(raw_target)
        target_metrics = calibration.broadband_validity_metrics(target)
    elif checked_scenario == "constant_target":
        constants = rng.standard_normal(size=(calibration.SYNTHETIC_ROWS, calibration.CHANNELS))
        target_normalized = np.broadcast_to(constants[:, :, None, None], _BANK_SHAPE).copy(order="C")
        raw_target = normalizer.invert(target_normalized)
        target = normalizer.apply(raw_target)
    else:
        if truth is None:
            raise RuntimeError("positive synthetic scenario unexpectedly has no truth")
        target_normalized = transform_central(source, truth)
        raw_target = normalizer.invert(target_normalized)
        target = normalizer.apply(raw_target)

    case = SyntheticCase(
        scenario=checked_scenario,
        seed=checked_seed,
        raw_source=raw_source,
        raw_target=raw_target,
        normalized_base=normalized_base,
        source=source,
        target=target,
        normalizer=normalizer,
        source_broadband=source_metrics,
        independent_target_broadband=target_metrics,
        truth=truth,
    )
    failures = case.generator_failure_codes()
    if failures:
        raise SyntheticV22Error(f"synthetic generator invalid: {','.join(failures)}")
    return case


def generate_independent_case(*, seed: int) -> SyntheticCase:
    """Generate one independent source/target bank from an injected uint64 seed."""

    return generate_case("independent", seed=seed)


def case_replays_exactly(left: SyntheticCase, right: SyntheticCase) -> bool:
    """Return whether two complete cases replay bit-for-bit, including evidence."""

    if not isinstance(left, SyntheticCase) or not isinstance(right, SyntheticCase):
        return False
    return (
        left.scenario == right.scenario
        and left.seed == right.seed
        and left.truth == right.truth
        and left.source_broadband == right.source_broadband
        and left.independent_target_broadband == right.independent_target_broadband
        and _array_bits_equal(left.raw_source, right.raw_source)
        and _array_bits_equal(left.raw_target, right.raw_target)
        and _array_bits_equal(left.normalized_base, right.normalized_base)
        and _array_bits_equal(left.source, right.source)
        and _array_bits_equal(left.target, right.target)
        and _array_bits_equal(left.normalizer.mean, right.normalizer.mean)
        and _array_bits_equal(left.normalizer.scale, right.normalizer.scale)
    )


def validate_case(case: SyntheticCase) -> None:
    """Deeply regenerate one case and reject any forged or stale scientific field."""

    if not isinstance(case, SyntheticCase):
        raise SyntheticV22Error("case validation requires a SyntheticCase")
    regenerated = generate_case(case.scenario, seed=case.seed)
    if not case_replays_exactly(case, regenerated):
        raise SyntheticV22Error("synthetic case differs from exact generator replay")


def derangements(
    rows: int = calibration.SYNTHETIC_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable adjacent-swap near and half-cycle far row mappings."""

    if isinstance(rows, bool) or not isinstance(rows, Integral):
        raise SyntheticV22Error("derangement row count must be an integer")
    count = int(rows)
    if count <= 0 or count % 2:
        raise SyntheticV22Error("derangements require a positive even row count")
    ordered = np.arange(count, dtype=_INT64_LE)
    near = ordered.reshape(-1, 2)[:, ::-1].reshape(-1)
    far = np.roll(ordered, count // 2)
    return _immutable_int64(near), _immutable_int64(far)


def row_targets(case: SyntheticCase, row: int) -> SyntheticRowTargets:
    """Return one strict source/true/near/far row bundle without fitting."""

    if not isinstance(case, SyntheticCase):
        raise SyntheticV22Error("row target lookup requires a SyntheticCase")
    if isinstance(row, bool) or not isinstance(row, Integral):
        raise SyntheticV22Error("synthetic row must be an integer")
    index = int(row)
    if not 0 <= index < calibration.SYNTHETIC_ROWS:
        raise SyntheticV22Error("synthetic row is outside [0,5]")
    near, far = derangements()
    near_index = int(near[index])
    far_index = int(far[index])
    return SyntheticRowTargets(
        row=index,
        source=case.source[index],
        true_target=case.target[index],
        near_row=near_index,
        near_target=case.target[near_index],
        far_row=far_index,
        far_target=case.target[far_index],
    )


def transpose_theta(theta: Sequence[float]) -> StateValues:
    """Apply the frozen affine transpose parameter permutation."""

    values = cast(StateValues, _finite_tuple(theta, 6, "theta"))
    transformed: StateValues = (
        values[1],
        values[0],
        values[5],
        values[4],
        values[3],
        values[2],
    )
    try:
        geometry.state_index(transformed)
    except geometry.GeometryValidationError as error:
        raise SyntheticV22Error("transposed theta is not an exact canonical affine state") from error
    return transformed


def transpose_truth(truth: TransformTruth) -> TransformTruth:
    """Transpose geometry while retaining channel gain and bias endpoints."""

    if not isinstance(truth, TransformTruth):
        raise SyntheticV22Error("truth transpose requires a TransformTruth")
    return TransformTruth(transpose_theta(truth.theta), truth.gains, truth.biases)


def transpose_r64(value: np.ndarray) -> np.ndarray:
    """Transpose spatial axes of one row or one six-row exact R64 bank."""

    if not isinstance(value, np.ndarray):
        raise SyntheticV22Error("R64 transpose input must be a NumPy array")
    if value.shape not in {_ROW_SHAPE, _BANK_SHAPE}:
        raise SyntheticV22Error("R64 transpose input must have shape [3,64,64] or [6,3,64,64]")
    if value.dtype != _FLOAT64_LE or not value.flags.c_contiguous:
        raise SyntheticV22Error("R64 transpose input must be C-contiguous little-endian float64")
    if not bool(np.all(np.isfinite(value))):
        raise SyntheticV22Error("R64 transpose input contains nonfinite values")
    return _immutable_float64(np.swapaxes(value, -2, -1))


def transpose_case(case: SyntheticCase) -> SyntheticCase:
    """Construct a complete immutable spatial-transpose metamorphic case."""

    if not isinstance(case, SyntheticCase):
        raise SyntheticV22Error("case transpose requires a SyntheticCase")
    normalized_base = transpose_r64(case.normalized_base)
    target = transpose_r64(case.target)
    target_metrics = (
        calibration.broadband_validity_metrics(target)
        if case.scenario == "independent"
        else None
    )
    return SyntheticCase(
        scenario=case.scenario,
        seed=case.seed,
        raw_source=transpose_r64(case.raw_source),
        raw_target=transpose_r64(case.raw_target),
        normalized_base=normalized_base,
        source=transpose_r64(case.source),
        target=target,
        normalizer=case.normalizer,
        source_broadband=calibration.broadband_validity_metrics(normalized_base),
        independent_target_broadband=target_metrics,
        truth=transpose_truth(case.truth) if case.truth is not None else None,
    )


__all__ = [
    "FROZEN_SEED_MAP",
    "IDENTITY_TRUTH",
    "PROTOCOL_SHA256",
    "SCENARIOS",
    "SCHEMA_VERSION",
    "TRUTHS",
    "Scenario",
    "StateValues",
    "SyntheticCase",
    "SyntheticRowTargets",
    "SyntheticV22Error",
    "TransformTruth",
    "case_replays_exactly",
    "derangements",
    "generate_case",
    "generate_independent_case",
    "row_targets",
    "transform_central",
    "transpose_case",
    "transpose_r64",
    "transpose_theta",
    "transpose_truth",
    "validate_case",
]
