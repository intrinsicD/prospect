"""Future-blind source-side prediction primitives for MM-011.

The public fitting APIs in this module accept only a normalized previous/current
pair.  They reuse the frozen MM-008 v2.2 estimators to reconstruct ``current``
from ``previous``, then apply each fitted operator exactly once to the *full*
``current`` frame.  No future frame, score, decision rule, or lifecycle state is
accepted here.

The wrapped v2.2 ``prediction`` fields are historical reconstructions.  Forecasts
have distinct field names and a separate, role-bound hash domain so that a
historical prediction hash cannot be mistaken for a causal forecast hash.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import nongrid_v22 as nongrid

SCHEMA_VERSION: Final = "mm011-source-predictor-v1"
V22_PROTOCOL_SHA256: Final = geometry.PROTOCOL_SHA256

Arm = Literal["affine", "appearance", "combined"]
_ARMS: Final[tuple[Arm, ...]] = ("affine", "appearance", "combined")
_GRID_ARMS: Final[tuple[Literal["affine", "combined"], ...]] = (
    "affine",
    "combined",
)
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_SCOPE_SHA256: Final = hashlib.sha256(
    b"MM011-source-side-scientific-array-v1\0"
).hexdigest()
_FULL_CONTEXTS: Final[dict[Literal["affine", "combined"], str]] = {
    "affine": "mm011/source/history/full/affine",
    "combined": "mm011/source/history/full/combined",
}
_IDENTITY_STATE: Final = geometry.state_index((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


class PredictorValidationError(ValueError):
    """Raised when a source-side input or result fails closed."""


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise PredictorValidationError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _readonly_float64(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise PredictorValidationError(f"{label} must have exact float64 dtype")
    if not bool(np.all(np.isfinite(array))):
        raise PredictorValidationError(f"{label} contains a nonfinite value")
    contiguous = np.array(array, dtype="<f8", order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8")
    return immutable.reshape(contiguous.shape)


def _readonly_int64(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"}:
        raise PredictorValidationError(f"{label} must contain integers")
    contiguous = np.array(array, dtype="<i8", order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype="<i8")
    return immutable.reshape(contiguous.shape)


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        isinstance(left, np.ndarray)
        and isinstance(right, np.ndarray)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _require_prediction(value: np.ndarray, label: str) -> np.ndarray:
    prediction = _readonly_float64(value, label)
    if prediction.shape != (geometry.CHANNELS, geometry.SITE_COUNT):
        raise PredictorValidationError(
            f"{label} must have shape [3,{geometry.SITE_COUNT}]"
        )
    return prediction


def _require_vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    vector = _readonly_float64(value, label)
    if vector.shape != (size,):
        raise PredictorValidationError(f"{label} must have shape [{size}]")
    return vector


def validate_normalized_frame(value: np.ndarray, *, label: str = "frame") -> None:
    """Validate the observable contract for one normalized R64 frame.

    Source-only normalization cannot be proved from one frame in isolation.  This
    validator therefore enforces the complete observable contract: exact shape,
    dtype, C order, and finiteness.  The caller remains responsible for applying
    the previously frozen source-only normalizer.
    """

    if not isinstance(value, np.ndarray):
        raise PredictorValidationError(f"{label} must be a NumPy array")
    if value.shape != (
        geometry.CHANNELS,
        geometry.NATIVE_SIZE,
        geometry.NATIVE_SIZE,
    ):
        raise PredictorValidationError(f"{label} must have shape [3,64,64]")
    if value.dtype != np.dtype(np.float64):
        raise PredictorValidationError(f"{label} must have exact float64 dtype")
    if not value.flags.c_contiguous:
        raise PredictorValidationError(f"{label} must be C-contiguous")
    if not bool(np.all(np.isfinite(value))):
        raise PredictorValidationError(f"{label} contains a nonfinite value")


def _freeze_frame(value: np.ndarray, label: str) -> np.ndarray:
    validate_normalized_frame(value, label=label)
    frozen = _readonly_float64(value, label)
    if frozen.shape != (geometry.CHANNELS, geometry.NATIVE_SIZE, geometry.NATIVE_SIZE):
        raise RuntimeError("validated frame changed shape while being frozen")
    return frozen


def array_sha256(value: np.ndarray, *, role: str) -> str:
    """Return a role-bound SHA-256 for one scientific array.

    This deliberately uses an MM-011-specific scope layered over the frozen v2.2
    array grammar.  Consequently, forecast hashes are fresh identities even when
    a stationary forecast has the same bytes as a historical reconstruction.
    """

    try:
        return nongrid.array_sha256(_ARRAY_SCOPE_SHA256, role, value)
    except nongrid.NonGridV22Error as error:
        raise PredictorValidationError(str(error)) from error


def apply_operator_once(
    current: np.ndarray,
    parameters: np.ndarray,
    gains: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """Apply one frozen operator to ``current`` at ``geometry.FULL_MASK``.

    Composition is spatial-first: bilinearly sample the full current frame using
    the selected canonical affine state, then apply per-channel gain and bias once.
    The returned central packed prediction is not clipped.
    """

    frame = _freeze_frame(current, "current")
    theta = _require_vector(parameters, 6, "operator parameters")
    gain = _require_vector(gains, geometry.CHANNELS, "operator gains")
    bias = _require_vector(biases, geometry.CHANNELS, "operator biases")
    try:
        state_index = geometry.state_index(theta)
    except geometry.GeometryValidationError as error:
        raise PredictorValidationError(
            "operator parameters must be one exact canonical affine state"
        ) from error
    if not bool(geometry.ADMISSIBLE_MASK[state_index]):
        raise PredictorValidationError("operator affine state is inadmissible")
    if bool(np.any(gain < fitting.GAIN_BOUNDS[0])) or bool(
        np.any(gain > fitting.GAIN_BOUNDS[1])
    ):
        raise PredictorValidationError("operator gains are outside v2.2 bounds")
    if bool(np.any(bias < fitting.BIAS_BOUNDS[0])) or bool(
        np.any(bias > fitting.BIAS_BOUNDS[1])
    ):
        raise PredictorValidationError("operator biases are outside v2.2 bounds")
    sampled = geometry.sample_scalar(frame, state_index, geometry.FULL_MASK)
    # Deliberately no np.clip: this is one algebraic operator application.
    return _readonly_float64(gain[:, None] * sampled + bias[:, None], "forecast")


@dataclass(frozen=True, slots=True)
class SourceBaselines:
    """Future-blind persistence and value-velocity predictions."""

    previous_sha256: str
    current_sha256: str
    persistence: np.ndarray
    persistence_sha256: str
    velocity: np.ndarray
    velocity_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.previous_sha256, "previous SHA-256")
        _require_sha256(self.current_sha256, "current SHA-256")
        persistence = _require_prediction(self.persistence, "persistence prediction")
        velocity = _require_prediction(self.velocity, "velocity prediction")
        if self.persistence_sha256 != array_sha256(
            persistence, role="persistence_forecast"
        ):
            raise PredictorValidationError("persistence forecast hash differs")
        if self.velocity_sha256 != array_sha256(velocity, role="velocity_forecast"):
            raise PredictorValidationError("velocity forecast hash differs")
        object.__setattr__(self, "persistence", persistence)
        object.__setattr__(self, "velocity", velocity)


def source_baselines(previous: np.ndarray, current: np.ndarray) -> SourceBaselines:
    """Construct persistence ``C`` and value velocity ``2C-P`` on full support."""

    past = _freeze_frame(previous, "previous")
    present = _freeze_frame(current, "current")
    previous_sites = geometry.sample_scalar(past, _IDENTITY_STATE, geometry.FULL_MASK)
    current_sites = geometry.sample_scalar(present, _IDENTITY_STATE, geometry.FULL_MASK)
    persistence = _readonly_float64(current_sites, "persistence prediction")
    # Deliberately no clipping: the baseline is the exact source-side value extrapolator.
    velocity = _readonly_float64(
        2.0 * current_sites - previous_sites, "velocity prediction"
    )
    return SourceBaselines(
        previous_sha256=array_sha256(past, role="previous_normalized"),
        current_sha256=array_sha256(present, role="current_normalized"),
        persistence=persistence,
        persistence_sha256=array_sha256(persistence, role="persistence_forecast"),
        velocity=velocity,
        velocity_sha256=array_sha256(velocity, role="velocity_forecast"),
    )


def validate_source_baselines(
    result: SourceBaselines, previous: np.ndarray, current: np.ndarray
) -> None:
    """Strictly replay a source-baseline result from its two allowed inputs."""

    if not isinstance(result, SourceBaselines):
        raise PredictorValidationError("baseline validation requires SourceBaselines")
    expected = source_baselines(previous, current)
    if (
        result.previous_sha256 != expected.previous_sha256
        or result.current_sha256 != expected.current_sha256
        or result.persistence_sha256 != expected.persistence_sha256
        or result.velocity_sha256 != expected.velocity_sha256
        or not _same_array(result.persistence, expected.persistence)
        or not _same_array(result.velocity, expected.velocity)
    ):
        raise PredictorValidationError("source baselines differ from exact replay")


def _bias_only_history_hash(estimate: nongrid.BiasOnlyEstimate) -> str:
    hashes = dict(estimate.hashes)
    digest = hashes.get("bias_only_prediction")
    if digest is None:
        raise PredictorValidationError(
            "bias-only history lacks its v2.2 prediction hash"
        )
    return _require_sha256(digest, "bias-only history prediction SHA-256")


@dataclass(frozen=True, slots=True)
class BiasOnlyResult:
    """A target-marginal history fit reused as a constant source-only forecast."""

    config_sha256: str
    current_sha256: str
    history_fit: nongrid.BiasOnlyEstimate
    history_reconstruction: np.ndarray
    history_reconstruction_sha256: str
    forecast: np.ndarray
    forecast_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.config_sha256, "config SHA-256")
        _require_sha256(self.current_sha256, "current SHA-256")
        if not isinstance(self.history_fit, nongrid.BiasOnlyEstimate):
            raise PredictorValidationError("bias-only control lacks a BiasOnlyEstimate")
        history = _require_prediction(
            self.history_reconstruction, "bias-only historical reconstruction"
        )
        forecast = _require_prediction(self.forecast, "bias-only constant forecast")
        if not _same_array(history, self.history_fit.prediction):
            raise PredictorValidationError(
                "bias-only historical reconstruction differs from its v2.2 fit"
            )
        if self.history_reconstruction_sha256 != _bias_only_history_hash(
            self.history_fit
        ):
            raise PredictorValidationError(
                "bias-only historical reconstruction hash differs from its v2.2 fit"
            )
        if not _same_array(forecast, self.history_fit.prediction):
            raise PredictorValidationError(
                "bias-only forecast must reuse the fitted target-marginal constant"
            )
        if self.forecast_sha256 != array_sha256(
            forecast, role="bias_only_forecast"
        ):
            raise PredictorValidationError("bias-only forecast hash differs")
        if not np.array_equal(
            forecast,
            np.broadcast_to(forecast[:, :1], forecast.shape),
        ):
            raise PredictorValidationError("bias-only forecast is not spatially constant")
        object.__setattr__(self, "history_reconstruction", history)
        object.__setattr__(self, "forecast", forecast)


def fit_bias_only_control(
    current: np.ndarray,
    *,
    config_sha256: str,
) -> BiasOnlyResult:
    """Fit a full target-marginal control from observed ``current`` only.

    The v2.2 fit predicts one retained per-channel constant.  Its historical
    prediction is preserved, while the identical constant used as a source-only
    future control receives a fresh MM-011 forecast hash.
    """

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    present = _freeze_frame(current, "current")
    historical_target = fitting.target_values(present, geometry.FULL_MASK)
    try:
        history_fit = nongrid.fit_bias_only(
            historical_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=checked_config,
        )
    except nongrid.NonGridV22Error as error:
        raise PredictorValidationError(str(error)) from error
    forecast = _readonly_float64(
        history_fit.prediction, "bias-only constant forecast"
    )
    return BiasOnlyResult(
        config_sha256=checked_config,
        current_sha256=array_sha256(present, role="current_normalized"),
        history_fit=history_fit,
        history_reconstruction=history_fit.prediction,
        history_reconstruction_sha256=_bias_only_history_hash(history_fit),
        forecast=forecast,
        forecast_sha256=array_sha256(forecast, role="bias_only_forecast"),
    )


HistoryFit = exact.GlobalResult | nongrid.AppearanceEstimate


def _appearance_history_hash(estimate: nongrid.AppearanceEstimate) -> str:
    hashes = dict(estimate.hashes)
    digest = hashes.get("appearance_prediction")
    if digest is None:
        raise PredictorValidationError(
            "appearance history lacks its v2.2 prediction hash"
        )
    return _require_sha256(digest, "appearance history prediction SHA-256")


@dataclass(frozen=True, slots=True)
class FullOperatorResult:
    """One historical v2.2 fit and its separately hashed causal forecast."""

    arm: Arm
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    history_fit: HistoryFit
    history_reconstruction: np.ndarray
    history_reconstruction_sha256: str
    forecast: np.ndarray
    forecast_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise PredictorValidationError("operator arm is invalid")
        parameters = _require_vector(self.parameters, 6, "operator parameters")
        gains = _require_vector(self.gains, 3, "operator gains")
        biases = _require_vector(self.biases, 3, "operator biases")
        history = _require_prediction(
            self.history_reconstruction, "historical reconstruction"
        )
        forecast = _require_prediction(self.forecast, "causal forecast")

        expected_parameters: np.ndarray
        expected_gains: np.ndarray
        expected_biases: np.ndarray
        expected_history: np.ndarray
        expected_history_hash: str
        if self.arm in _GRID_ARMS:
            if not isinstance(self.history_fit, exact.GlobalResult):
                raise PredictorValidationError("grid arm lacks a GlobalResult history fit")
            if self.history_fit.arm != self.arm:
                raise PredictorValidationError("grid history fit belongs to another arm")
            if self.history_fit.context_key != _FULL_CONTEXTS[self.arm]:
                raise PredictorValidationError("grid history fit has the wrong context key")
            expected_parameters = self.history_fit.selected.parameters
            expected_history = self.history_fit.prediction
            expected_history_hash = self.history_fit.prediction_sha256
            if self.arm == "affine":
                if (
                    self.history_fit.selected.gains is not None
                    or self.history_fit.selected.biases is not None
                ):
                    raise PredictorValidationError("affine history carries appearance values")
                expected_gains = np.ones(3, dtype=np.float64)
                expected_biases = np.zeros(3, dtype=np.float64)
            else:
                if (
                    self.history_fit.selected.gains is None
                    or self.history_fit.selected.biases is None
                ):
                    raise PredictorValidationError("combined history lacks appearance values")
                expected_gains = self.history_fit.selected.gains
                expected_biases = self.history_fit.selected.biases
        else:
            if not isinstance(self.history_fit, nongrid.AppearanceEstimate):
                raise PredictorValidationError(
                    "appearance arm lacks an AppearanceEstimate history fit"
                )
            expected_parameters = self.history_fit.parameters
            expected_gains = self.history_fit.gains
            expected_biases = self.history_fit.biases
            expected_history = self.history_fit.prediction
            expected_history_hash = _appearance_history_hash(self.history_fit)

        if not _same_array(parameters, expected_parameters):
            raise PredictorValidationError("operator parameters differ from history fit")
        if not _same_array(gains, expected_gains):
            raise PredictorValidationError("operator gains differ from history fit")
        if not _same_array(biases, expected_biases):
            raise PredictorValidationError("operator biases differ from history fit")
        if not _same_array(history, expected_history):
            raise PredictorValidationError(
                "historical reconstruction differs from the v2.2 fit prediction"
            )
        if self.history_reconstruction_sha256 != expected_history_hash:
            raise PredictorValidationError(
                "historical reconstruction hash differs from its v2.2 fit"
            )
        _require_sha256(
            self.history_reconstruction_sha256,
            "historical reconstruction SHA-256",
        )
        expected_forecast_hash = array_sha256(
            forecast, role=f"{self.arm}_forecast"
        )
        if self.forecast_sha256 != expected_forecast_hash:
            raise PredictorValidationError("causal forecast hash differs")

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "history_reconstruction", history)
        object.__setattr__(self, "forecast", forecast)


@dataclass(frozen=True, slots=True)
class SourcePairResult:
    """All three unselected operators and baselines for one source pair."""

    config_sha256: str
    previous_sha256: str
    current_sha256: str
    affine: FullOperatorResult
    appearance: FullOperatorResult
    combined: FullOperatorResult
    bias_only: BiasOnlyResult
    baselines: SourceBaselines

    def __post_init__(self) -> None:
        config = _require_sha256(self.config_sha256, "config SHA-256")
        _require_sha256(self.previous_sha256, "previous SHA-256")
        _require_sha256(self.current_sha256, "current SHA-256")
        operators = (self.affine, self.appearance, self.combined)
        if any(not isinstance(value, FullOperatorResult) for value in operators):
            raise PredictorValidationError("source pair contains an invalid operator result")
        if tuple(value.arm for value in operators) != _ARMS:
            raise PredictorValidationError("source-pair operator order is invalid")
        if not isinstance(self.affine.history_fit, exact.GlobalResult):
            raise PredictorValidationError("affine fit has the wrong type")
        if not isinstance(self.combined.history_fit, exact.GlobalResult):
            raise PredictorValidationError("combined fit has the wrong type")
        if self.affine.history_fit.certificate.config_sha256 != config:
            raise PredictorValidationError("affine fit config hash differs")
        if self.combined.history_fit.certificate.config_sha256 != config:
            raise PredictorValidationError("combined fit config hash differs")
        if not isinstance(self.bias_only, BiasOnlyResult):
            raise PredictorValidationError("source pair lacks a bias-only control")
        if (
            self.bias_only.config_sha256 != config
            or self.bias_only.current_sha256 != self.current_sha256
        ):
            raise PredictorValidationError("bias-only control inputs differ from pair inputs")
        if not isinstance(self.baselines, SourceBaselines):
            raise PredictorValidationError("source pair lacks source baselines")
        if (
            self.baselines.previous_sha256 != self.previous_sha256
            or self.baselines.current_sha256 != self.current_sha256
        ):
            raise PredictorValidationError("baseline inputs differ from pair inputs")

    def operator(self, arm: Arm) -> FullOperatorResult:
        """Return one named operator without selecting or scoring it."""

        if arm not in _ARMS:
            raise PredictorValidationError("operator arm is invalid")
        return cast(FullOperatorResult, getattr(self, arm))


def _full_operator(
    arm: Arm,
    history_fit: HistoryFit,
    current: np.ndarray,
) -> FullOperatorResult:
    if arm == "affine":
        if not isinstance(history_fit, exact.GlobalResult):
            raise RuntimeError("affine integration received a non-grid fit")
        parameters = history_fit.selected.parameters
        gains = np.ones(3, dtype=np.float64)
        biases = np.zeros(3, dtype=np.float64)
        history = history_fit.prediction
        history_hash = history_fit.prediction_sha256
    elif arm == "combined":
        if not isinstance(history_fit, exact.GlobalResult):
            raise RuntimeError("combined integration received a non-grid fit")
        if history_fit.selected.gains is None or history_fit.selected.biases is None:
            raise RuntimeError("combined v2.2 fit unexpectedly lacks appearance values")
        parameters = history_fit.selected.parameters
        gains = history_fit.selected.gains
        biases = history_fit.selected.biases
        history = history_fit.prediction
        history_hash = history_fit.prediction_sha256
    else:
        if not isinstance(history_fit, nongrid.AppearanceEstimate):
            raise RuntimeError("appearance integration received a grid fit")
        parameters = history_fit.parameters
        gains = history_fit.gains
        biases = history_fit.biases
        history = history_fit.prediction
        history_hash = _appearance_history_hash(history_fit)
    forecast = apply_operator_once(current, parameters, gains, biases)
    return FullOperatorResult(
        arm=arm,
        parameters=parameters,
        gains=gains,
        biases=biases,
        history_fit=history_fit,
        history_reconstruction=history,
        history_reconstruction_sha256=history_hash,
        forecast=forecast,
        forecast_sha256=array_sha256(forecast, role=f"{arm}_forecast"),
    )


def fit_source_pair(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    config_sha256: str,
) -> SourcePairResult:
    """Fit all source-side full-context operators and forecast one step.

    ``current`` is solely the historical fit target for ``previous -> current``
    and then the sole source for the one-step forecasts.  There is intentionally
    no future/target argument and no choice among the returned arms.
    """

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    past = _freeze_frame(previous, "previous")
    present = _freeze_frame(current, "current")
    historical_target = fitting.target_values(present, geometry.FULL_MASK)
    requests = tuple(
        exact.FitRequest.create(
            _FULL_CONTEXTS[arm],
            arm,
            historical_target,
        )
        for arm in _GRID_ARMS
    )
    try:
        affine_fit, combined_fit = exact.fit_global_contexts(
            past,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            requests,
            config_sha256=checked_config,
        )
        appearance_fit = nongrid.fit_appearance(
            past,
            historical_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=checked_config,
        )
    except (exact.GlobalV22Error, nongrid.NonGridV22Error) as error:
        raise PredictorValidationError(str(error)) from error

    baselines = source_baselines(past, present)
    bias_only = fit_bias_only_control(present, config_sha256=checked_config)
    return SourcePairResult(
        config_sha256=checked_config,
        previous_sha256=array_sha256(past, role="previous_normalized"),
        current_sha256=array_sha256(present, role="current_normalized"),
        affine=_full_operator("affine", affine_fit, present),
        appearance=_full_operator("appearance", appearance_fit, present),
        combined=_full_operator("combined", combined_fit, present),
        bias_only=bias_only,
        baselines=baselines,
    )


def _appearance_fits_equal(
    left: nongrid.AppearanceEstimate, right: nongrid.AppearanceEstimate
) -> bool:
    return (
        left.scope_sha256 == right.scope_sha256
        and left.retained_macro_ids == right.retained_macro_ids
        and struct.pack("<d", left.objective) == struct.pack("<d", right.objective)
        and left.hashes == right.hashes
        and all(
            _same_array(getattr(left, name), getattr(right, name))
            for name in (
                "parameters",
                "gains",
                "biases",
                "fit_prediction",
                "prediction",
            )
        )
    )


def _bias_only_fits_equal(
    left: nongrid.BiasOnlyEstimate, right: nongrid.BiasOnlyEstimate
) -> bool:
    return (
        left.scope_sha256 == right.scope_sha256
        and left.retained_macro_ids == right.retained_macro_ids
        and struct.pack("<d", left.objective) == struct.pack("<d", right.objective)
        and left.hashes == right.hashes
        and all(
            _same_array(getattr(left, name), getattr(right, name))
            for name in ("first_biases", "biases", "prediction")
        )
    )


def validate_bias_only_control(
    result: BiasOnlyResult,
    current: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Strictly refit and replay one full bias-only source control."""

    if not isinstance(result, BiasOnlyResult):
        raise PredictorValidationError("bias-only validation requires BiasOnlyResult")
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if result.config_sha256 != checked_config:
        raise PredictorValidationError("bias-only result config SHA-256 differs")
    present = _freeze_frame(current, "current")
    if result.current_sha256 != array_sha256(present, role="current_normalized"):
        raise PredictorValidationError("bias-only current hash differs")
    historical_target = fitting.target_values(present, geometry.FULL_MASK)
    try:
        replay = nongrid.fit_bias_only(
            historical_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=checked_config,
        )
    except nongrid.NonGridV22Error as error:
        raise PredictorValidationError(str(error)) from error
    if not _bias_only_fits_equal(result.history_fit, replay):
        raise PredictorValidationError("bias-only history differs from exact replay")
    if (
        not _same_array(result.history_reconstruction, replay.prediction)
        or not _same_array(result.forecast, replay.prediction)
    ):
        raise PredictorValidationError("bias-only arrays differ from exact replay")
    if result.forecast_sha256 != array_sha256(
        result.forecast, role="bias_only_forecast"
    ):
        raise PredictorValidationError("bias-only forecast hash differs on replay")


def _validate_full_operator_replay(
    result: FullOperatorResult, current: np.ndarray
) -> None:
    replay = apply_operator_once(
        current, result.parameters, result.gains, result.biases
    )
    if not _same_array(result.forecast, replay):
        raise PredictorValidationError(
            f"{result.arm} forecast differs from one-step spatial-first replay"
        )
    if result.forecast_sha256 != array_sha256(
        replay, role=f"{result.arm}_forecast"
    ):
        raise PredictorValidationError(f"{result.arm} forecast hash differs on replay")


def validate_source_pair_result(
    result: SourcePairResult,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Deeply validate a full result against only its previous/current pair.

    The two exact-grid fits are completely rebuilt through the v2.2 validator;
    appearance, forecasts, hashes, and baselines are independently replayed.  This
    function is intentionally expensive and should be used at evidence boundaries,
    not on every prediction call.
    """

    if not isinstance(result, SourcePairResult):
        raise PredictorValidationError("deep validation requires SourcePairResult")
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if result.config_sha256 != checked_config:
        raise PredictorValidationError("result config SHA-256 differs")
    past = _freeze_frame(previous, "previous")
    present = _freeze_frame(current, "current")
    previous_hash = array_sha256(past, role="previous_normalized")
    current_hash = array_sha256(present, role="current_normalized")
    if (
        result.previous_sha256 != previous_hash
        or result.current_sha256 != current_hash
    ):
        raise PredictorValidationError("source-pair input hashes differ")
    historical_target = fitting.target_values(present, geometry.FULL_MASK)
    try:
        exact.validate_global_result(
            cast(exact.GlobalResult, result.affine.history_fit),
            past,
            historical_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=checked_config,
        )
        exact.validate_global_result(
            cast(exact.GlobalResult, result.combined.history_fit),
            past,
            historical_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=checked_config,
        )
        appearance_replay = nongrid.fit_appearance(
            past,
            historical_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=checked_config,
        )
    except (exact.GlobalV22Error, nongrid.NonGridV22Error) as error:
        raise PredictorValidationError(str(error)) from error
    if not isinstance(result.appearance.history_fit, nongrid.AppearanceEstimate):
        raise PredictorValidationError("appearance history fit has the wrong type")
    if not _appearance_fits_equal(result.appearance.history_fit, appearance_replay):
        raise PredictorValidationError("appearance history differs from exact replay")
    for operator in (result.affine, result.appearance, result.combined):
        _validate_full_operator_replay(operator, present)
    validate_bias_only_control(
        result.bias_only, present, config_sha256=checked_config
    )
    validate_source_baselines(result.baselines, past, present)


def _checker_context(
    arm: Literal["affine", "combined"], output_parity: int
) -> str:
    return f"mm011/source/history/checkerboard/output-{output_parity}/{arm}"


@dataclass(frozen=True, slots=True)
class CheckerboardArmResult:
    """Two held-parity history fits stitched into full central row-major order."""

    arm: Arm
    output_parity_fits: tuple[HistoryFit, HistoryFit]
    history_reconstruction: np.ndarray
    history_reconstruction_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARMS:
            raise PredictorValidationError("checkerboard arm is invalid")
        if (
            type(self.output_parity_fits) is not tuple
            or len(self.output_parity_fits) != 2
        ):
            raise PredictorValidationError(
                "checkerboard fits must be a two-tuple indexed by output parity"
            )
        reconstruction = _require_prediction(
            self.history_reconstruction, "checkerboard historical reconstruction"
        )
        expected = np.empty(
            (geometry.CHANNELS, geometry.SITE_COUNT), dtype=np.float64
        )
        for output_parity, history_fit in enumerate(self.output_parity_fits):
            output_mask = geometry.PARITY_MASKS[output_parity]
            if self.arm in _GRID_ARMS:
                if not isinstance(history_fit, exact.GlobalResult):
                    raise PredictorValidationError(
                        "checkerboard grid arm lacks a GlobalResult"
                    )
                if history_fit.arm != self.arm:
                    raise PredictorValidationError(
                        "checkerboard grid fit belongs to another arm"
                    )
                if history_fit.context_key != _checker_context(
                    self.arm, output_parity
                ):
                    raise PredictorValidationError(
                        "checkerboard grid fit has the wrong context key"
                    )
                partial = history_fit.prediction
            else:
                if not isinstance(history_fit, nongrid.AppearanceEstimate):
                    raise PredictorValidationError(
                        "checkerboard appearance arm lacks an AppearanceEstimate"
                    )
                partial = history_fit.prediction
            if partial.shape != (
                geometry.CHANNELS,
                geometry.SITE_COUNT // 2,
            ):
                raise PredictorValidationError(
                    "checkerboard held-parity prediction has the wrong shape"
                )
            expected[:, output_mask] = partial
        if not _same_array(reconstruction, expected):
            raise PredictorValidationError(
                "checkerboard reconstruction differs from its two parity fits"
            )
        expected_hash = array_sha256(
            reconstruction,
            role=f"{self.arm}_checkerboard_history_reconstruction",
        )
        if self.history_reconstruction_sha256 != expected_hash:
            raise PredictorValidationError(
                "checkerboard historical reconstruction hash differs"
            )
        object.__setattr__(self, "history_reconstruction", reconstruction)


@dataclass(frozen=True, slots=True)
class CheckerboardBiasOnlyResult:
    """Two opposite-parity target-marginal fits stitched as a history control."""

    config_sha256: str
    current_sha256: str
    output_parity_fits: tuple[
        nongrid.BiasOnlyEstimate,
        nongrid.BiasOnlyEstimate,
    ]
    history_reconstruction: np.ndarray
    history_reconstruction_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.config_sha256, "config SHA-256")
        _require_sha256(self.current_sha256, "current SHA-256")
        if (
            type(self.output_parity_fits) is not tuple
            or len(self.output_parity_fits) != 2
            or any(
                not isinstance(value, nongrid.BiasOnlyEstimate)
                for value in self.output_parity_fits
            )
        ):
            raise PredictorValidationError(
                "checkerboard bias-only fits must be a two-tuple"
            )
        reconstruction = _require_prediction(
            self.history_reconstruction,
            "checkerboard bias-only historical reconstruction",
        )
        expected = np.empty(
            (geometry.CHANNELS, geometry.SITE_COUNT), dtype=np.float64
        )
        for output_parity, history_fit in enumerate(self.output_parity_fits):
            if history_fit.prediction.shape != (
                geometry.CHANNELS,
                geometry.SITE_COUNT // 2,
            ):
                raise PredictorValidationError(
                    "checkerboard bias-only held-parity prediction has the wrong shape"
                )
            expected[:, geometry.PARITY_MASKS[output_parity]] = history_fit.prediction
        if not _same_array(reconstruction, expected):
            raise PredictorValidationError(
                "checkerboard bias-only reconstruction differs from its parity fits"
            )
        expected_hash = array_sha256(
            reconstruction,
            role="bias_only_checkerboard_history_reconstruction",
        )
        if self.history_reconstruction_sha256 != expected_hash:
            raise PredictorValidationError(
                "checkerboard bias-only reconstruction hash differs"
            )
        object.__setattr__(self, "history_reconstruction", reconstruction)


def fit_checkerboard_bias_only_history(
    current: np.ndarray,
    *,
    config_sha256: str,
) -> CheckerboardBiasOnlyResult:
    """Fit two cheap opposite-parity bias-only history controls from ``current``."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    present = _freeze_frame(current, "current")
    fits: list[nongrid.BiasOnlyEstimate] = []
    try:
        for output_parity in (0, 1):
            fit_mask = geometry.PARITY_MASKS[1 - output_parity]
            output_mask = geometry.PARITY_MASKS[output_parity]
            historical_target = fitting.target_values(present, fit_mask)
            fits.append(
                nongrid.fit_bias_only(
                    historical_target,
                    fit_mask,
                    output_mask,
                    config_sha256=checked_config,
                )
            )
    except nongrid.NonGridV22Error as error:
        raise PredictorValidationError(str(error)) from error
    fit_tuple = cast(
        tuple[nongrid.BiasOnlyEstimate, nongrid.BiasOnlyEstimate], tuple(fits)
    )
    reconstruction = np.empty(
        (geometry.CHANNELS, geometry.SITE_COUNT), dtype=np.float64
    )
    for output_parity, history_fit in enumerate(fit_tuple):
        reconstruction[:, geometry.PARITY_MASKS[output_parity]] = history_fit.prediction
    frozen = _readonly_float64(
        reconstruction, "checkerboard bias-only reconstruction"
    )
    return CheckerboardBiasOnlyResult(
        config_sha256=checked_config,
        current_sha256=array_sha256(present, role="current_normalized"),
        output_parity_fits=fit_tuple,
        history_reconstruction=frozen,
        history_reconstruction_sha256=array_sha256(
            frozen,
            role="bias_only_checkerboard_history_reconstruction",
        ),
    )


@dataclass(frozen=True, slots=True)
class CheckerboardHistoryResult:
    """All unselected source-only checkerboard historical reconstructions."""

    config_sha256: str
    previous_sha256: str
    current_sha256: str
    affine: CheckerboardArmResult
    appearance: CheckerboardArmResult
    combined: CheckerboardArmResult
    bias_only: CheckerboardBiasOnlyResult

    def __post_init__(self) -> None:
        checked_config = _require_sha256(self.config_sha256, "config SHA-256")
        _require_sha256(self.previous_sha256, "previous SHA-256")
        _require_sha256(self.current_sha256, "current SHA-256")
        arms = (self.affine, self.appearance, self.combined)
        if any(not isinstance(value, CheckerboardArmResult) for value in arms):
            raise PredictorValidationError("checkerboard result contains an invalid arm")
        if tuple(value.arm for value in arms) != _ARMS:
            raise PredictorValidationError("checkerboard arm order is invalid")
        if not isinstance(self.bias_only, CheckerboardBiasOnlyResult):
            raise PredictorValidationError(
                "checkerboard result lacks a bias-only control"
            )
        if (
            self.bias_only.config_sha256 != checked_config
            or self.bias_only.current_sha256 != self.current_sha256
        ):
            raise PredictorValidationError(
                "checkerboard bias-only inputs differ from checkerboard inputs"
            )
        for arm_result in (self.affine, self.combined):
            for history_fit in arm_result.output_parity_fits:
                if not isinstance(history_fit, exact.GlobalResult):
                    raise PredictorValidationError("checkerboard grid fit type differs")
                if history_fit.certificate.config_sha256 != checked_config:
                    raise PredictorValidationError("checkerboard fit config hash differs")

    def arm(self, arm: Arm) -> CheckerboardArmResult:
        """Return one named checkerboard reconstruction without choosing an arm."""

        if arm not in _ARMS:
            raise PredictorValidationError("checkerboard arm is invalid")
        return cast(CheckerboardArmResult, getattr(self, arm))


def _checker_arm(
    arm: Arm, fits: tuple[HistoryFit, HistoryFit]
) -> CheckerboardArmResult:
    reconstruction = np.empty(
        (geometry.CHANNELS, geometry.SITE_COUNT), dtype=np.float64
    )
    for output_parity, history_fit in enumerate(fits):
        reconstruction[:, geometry.PARITY_MASKS[output_parity]] = history_fit.prediction
    frozen = _readonly_float64(reconstruction, "checkerboard reconstruction")
    return CheckerboardArmResult(
        arm=arm,
        output_parity_fits=fits,
        history_reconstruction=frozen,
        history_reconstruction_sha256=array_sha256(
            frozen, role=f"{arm}_checkerboard_history_reconstruction"
        ),
    )


def fit_checkerboard_history(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    config_sha256: str,
) -> CheckerboardHistoryResult:
    """Fit held-parity ``previous -> current`` reconstructions for all arms.

    This diagnostic is intentionally separate from :func:`fit_source_pair`: each
    output parity requires another exact v2.2 source stream, while it produces no
    causal forecast.  Fits are returned in ``(output parity 0, output parity 1)``
    order and stitched only after both held-parity predictions exist.
    """

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    past = _freeze_frame(previous, "previous")
    present = _freeze_frame(current, "current")
    grid_fits: dict[str, list[HistoryFit]] = {
        "affine": [],
        "combined": [],
    }
    appearance_fits: list[HistoryFit] = []
    try:
        for output_parity in (0, 1):
            fit_mask = geometry.PARITY_MASKS[1 - output_parity]
            output_mask = geometry.PARITY_MASKS[output_parity]
            historical_target = fitting.target_values(present, fit_mask)
            requests = tuple(
                exact.FitRequest.create(
                    _checker_context(arm, output_parity), arm, historical_target
                )
                for arm in _GRID_ARMS
            )
            fitted = exact.fit_global_contexts(
                past,
                fit_mask,
                output_mask,
                requests,
                config_sha256=checked_config,
            )
            grid_fits["affine"].append(fitted[0])
            grid_fits["combined"].append(fitted[1])
            appearance_fits.append(
                nongrid.fit_appearance(
                    past,
                    historical_target,
                    fit_mask,
                    output_mask,
                    config_sha256=checked_config,
                )
            )
    except (exact.GlobalV22Error, nongrid.NonGridV22Error) as error:
        raise PredictorValidationError(str(error)) from error

    affine_fits = cast(tuple[HistoryFit, HistoryFit], tuple(grid_fits["affine"]))
    combined_fits = cast(
        tuple[HistoryFit, HistoryFit], tuple(grid_fits["combined"])
    )
    appearance_tuple = cast(
        tuple[HistoryFit, HistoryFit], tuple(appearance_fits)
    )
    bias_only = fit_checkerboard_bias_only_history(
        present, config_sha256=checked_config
    )
    return CheckerboardHistoryResult(
        config_sha256=checked_config,
        previous_sha256=array_sha256(past, role="previous_normalized"),
        current_sha256=array_sha256(present, role="current_normalized"),
        affine=_checker_arm("affine", affine_fits),
        appearance=_checker_arm("appearance", appearance_tuple),
        combined=_checker_arm("combined", combined_fits),
        bias_only=bias_only,
    )


def _validate_checker_arm_replay(result: CheckerboardArmResult) -> None:
    expected = np.empty(
        (geometry.CHANNELS, geometry.SITE_COUNT), dtype=np.float64
    )
    for output_parity, history_fit in enumerate(result.output_parity_fits):
        expected[:, geometry.PARITY_MASKS[output_parity]] = history_fit.prediction
    if not _same_array(result.history_reconstruction, expected):
        raise PredictorValidationError(
            f"{result.arm} checkerboard reconstruction differs on replay"
        )
    expected_hash = array_sha256(
        result.history_reconstruction,
        role=f"{result.arm}_checkerboard_history_reconstruction",
    )
    if result.history_reconstruction_sha256 != expected_hash:
        raise PredictorValidationError(
            f"{result.arm} checkerboard reconstruction hash differs on replay"
        )


def validate_checkerboard_bias_only_history(
    result: CheckerboardBiasOnlyResult,
    current: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Strictly refit and replay both checkerboard bias-only contexts."""

    if not isinstance(result, CheckerboardBiasOnlyResult):
        raise PredictorValidationError(
            "checkerboard bias-only validation requires CheckerboardBiasOnlyResult"
        )
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if result.config_sha256 != checked_config:
        raise PredictorValidationError(
            "checkerboard bias-only config SHA-256 differs"
        )
    present = _freeze_frame(current, "current")
    if result.current_sha256 != array_sha256(present, role="current_normalized"):
        raise PredictorValidationError("checkerboard bias-only current hash differs")
    replay = fit_checkerboard_bias_only_history(
        present, config_sha256=checked_config
    )
    if any(
        not _bias_only_fits_equal(left, right)
        for left, right in zip(
            result.output_parity_fits,
            replay.output_parity_fits,
            strict=True,
        )
    ):
        raise PredictorValidationError(
            "checkerboard bias-only history differs from exact replay"
        )
    if (
        not _same_array(
            result.history_reconstruction, replay.history_reconstruction
        )
        or result.history_reconstruction_sha256
        != replay.history_reconstruction_sha256
    ):
        raise PredictorValidationError(
            "checkerboard bias-only reconstruction differs from exact replay"
        )


def validate_checkerboard_history(
    result: CheckerboardHistoryResult,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Deeply rebuild both held-parity historical contexts for every arm."""

    if not isinstance(result, CheckerboardHistoryResult):
        raise PredictorValidationError(
            "checkerboard validation requires CheckerboardHistoryResult"
        )
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if result.config_sha256 != checked_config:
        raise PredictorValidationError("checkerboard result config SHA-256 differs")
    past = _freeze_frame(previous, "previous")
    present = _freeze_frame(current, "current")
    if result.previous_sha256 != array_sha256(
        past, role="previous_normalized"
    ) or result.current_sha256 != array_sha256(present, role="current_normalized"):
        raise PredictorValidationError("checkerboard input hashes differ")
    for output_parity in (0, 1):
        fit_mask = geometry.PARITY_MASKS[1 - output_parity]
        output_mask = geometry.PARITY_MASKS[output_parity]
        historical_target = fitting.target_values(present, fit_mask)
        try:
            for arm_result in (result.affine, result.combined):
                history_fit = arm_result.output_parity_fits[output_parity]
                if not isinstance(history_fit, exact.GlobalResult):
                    raise PredictorValidationError(
                        "checkerboard grid history fit has the wrong type"
                    )
                exact.validate_global_result(
                    history_fit,
                    past,
                    historical_target,
                    fit_mask,
                    output_mask,
                    config_sha256=checked_config,
                )
            appearance_replay = nongrid.fit_appearance(
                past,
                historical_target,
                fit_mask,
                output_mask,
                config_sha256=checked_config,
            )
        except (exact.GlobalV22Error, nongrid.NonGridV22Error) as error:
            raise PredictorValidationError(str(error)) from error
        appearance_fit = result.appearance.output_parity_fits[output_parity]
        if not isinstance(appearance_fit, nongrid.AppearanceEstimate):
            raise PredictorValidationError(
                "checkerboard appearance history fit has the wrong type"
            )
        if not _appearance_fits_equal(appearance_fit, appearance_replay):
            raise PredictorValidationError(
                "checkerboard appearance history differs from exact replay"
            )
    for arm_result in (result.affine, result.appearance, result.combined):
        _validate_checker_arm_replay(arm_result)
    validate_checkerboard_bias_only_history(
        result.bias_only, present, config_sha256=checked_config
    )


def _validated_video_axis(
    video_ids: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(video_ids, np.ndarray) or video_ids.ndim != 1:
        raise PredictorValidationError("video_ids must be a one-dimensional NumPy array")
    if video_ids.dtype.kind not in {"U", "S"}:
        raise PredictorValidationError("video_ids must have a string dtype")
    ids = np.ascontiguousarray(video_ids.astype(str, copy=False))
    if len(ids) == 0 or bool(np.any(np.char.str_len(ids) == 0)):
        raise PredictorValidationError("video_ids must be nonempty strings")
    if (
        not isinstance(timestamps, np.ndarray)
        or timestamps.ndim != 1
        or timestamps.shape != ids.shape
        or timestamps.dtype != np.dtype(np.float64)
        or not timestamps.flags.c_contiguous
        or not bool(np.all(np.isfinite(timestamps)))
    ):
        raise PredictorValidationError(
            "timestamps must be finite C-contiguous float64 with the video_ids shape"
        )
    return ids, timestamps


def within_video_half_cycle_indices(
    video_ids: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    """Return a deterministic fixed-point-free half-cycle row mapping.

    Rows are ordered by timestamp with a stable input-order tie break, rolled by
    ``floor(group_size / 2)``, and never cross a video identity.
    """

    ids, times = _validated_video_axis(video_ids, timestamps)
    mapping = np.empty(len(ids), dtype="<i8")
    for video_id in tuple(dict.fromkeys(ids.tolist())):
        rows = np.flatnonzero(ids == video_id)
        if len(rows) < 2:
            raise PredictorValidationError(
                "each video needs at least two rows for a half-cycle mapping"
            )
        ordered = rows[np.argsort(times[rows], kind="stable")]
        mapping[ordered] = np.roll(ordered, len(ordered) // 2)
    expected_rows = np.arange(len(ids), dtype=np.int64)
    if (
        not np.array_equal(np.sort(mapping), expected_rows)
        or bool(np.any(mapping == expected_rows))
        or not np.array_equal(ids[mapping], ids)
    ):
        raise PredictorValidationError(
            "half-cycle indices are not a fixed-point-free within-video permutation"
        )
    return _readonly_int64(mapping, "half-cycle indices")


def validate_half_cycle_indices(
    mapping: np.ndarray, video_ids: np.ndarray, timestamps: np.ndarray
) -> None:
    """Strictly replay and compare a within-video half-cycle mapping."""

    if (
        not isinstance(mapping, np.ndarray)
        or mapping.ndim != 1
        or mapping.dtype != np.dtype(np.int64)
        or not mapping.flags.c_contiguous
    ):
        raise PredictorValidationError(
            "half-cycle mapping must be C-contiguous int64 [row]"
        )
    expected = within_video_half_cycle_indices(video_ids, timestamps)
    if not _same_array(mapping, expected):
        raise PredictorValidationError("half-cycle mapping differs from exact replay")


__all__ = [
    "Arm",
    "BiasOnlyResult",
    "CheckerboardArmResult",
    "CheckerboardBiasOnlyResult",
    "CheckerboardHistoryResult",
    "FullOperatorResult",
    "PredictorValidationError",
    "SCHEMA_VERSION",
    "SourceBaselines",
    "SourcePairResult",
    "V22_PROTOCOL_SHA256",
    "apply_operator_once",
    "array_sha256",
    "fit_bias_only_control",
    "fit_checkerboard_bias_only_history",
    "fit_checkerboard_history",
    "fit_source_pair",
    "source_baselines",
    "validate_checkerboard_history",
    "validate_checkerboard_bias_only_history",
    "validate_bias_only_control",
    "validate_half_cycle_indices",
    "validate_normalized_frame",
    "validate_source_baselines",
    "validate_source_pair_result",
    "within_video_half_cycle_indices",
]
