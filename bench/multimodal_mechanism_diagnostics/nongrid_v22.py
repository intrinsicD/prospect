"""Deterministic non-grid estimators and array identities for MM-008 v2.2."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry

PROTOCOL_SHA256: Final = (
    "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
)
SCHEMA_VERSION: Final = "mm008-v2.2-nongrid-v1"
NONGRID_SCOPE_TAG: Final = b"MM008-v2.2-nongrid-fit-scope\0"
PERSISTENCE_SCOPE_TAG: Final = b"MM008-v2.2-persistence-scope\0"
ARRAY_TAG: Final = b"MM008-v2.2-array\0"

NonGridArm = Literal["appearance", "bias_only"]
_ARM_BYTE: Final[dict[NonGridArm, int]] = {"appearance": 2, "bias_only": 3}
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")


class NonGridV22Error(ValueError):
    """Raised when a deterministic non-grid fit violates the frozen contract."""


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise NonGridV22Error(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, dtype="<f8", order="C", copy=True)
    if not np.all(np.isfinite(contiguous)):
        raise NonGridV22Error("scientific array contains a nonfinite value")
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8")
    return immutable.reshape(contiguous.shape)


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, dtype=np.bool_, order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)
    return immutable.reshape(contiguous.shape)


def _source(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (geometry.CHANNELS, geometry.NATIVE_SIZE, geometry.NATIVE_SIZE)
        or value.dtype != np.dtype(np.float64)
        or not value.flags.c_contiguous
    ):
        raise NonGridV22Error("source must be C-contiguous float64 [3,64,64]")
    return _readonly_float64(value)


def _masks(fit_mask: np.ndarray, output_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fit_mask, np.ndarray) or not isinstance(output_mask, np.ndarray):
        raise NonGridV22Error("fit and output masks must be NumPy arrays")
    if any(
        mask.shape != (geometry.SITE_COUNT,)
        or mask.dtype != np.dtype(np.bool_)
        or not mask.flags.c_contiguous
        for mask in (fit_mask, output_mask)
    ):
        raise NonGridV22Error("fit/output masks must be C-contiguous bool [2304]")
    full = np.array_equal(fit_mask, geometry.FULL_MASK) and np.array_equal(output_mask, geometry.FULL_MASK)
    xfit = any(
        np.array_equal(fit_mask, geometry.PARITY_MASKS[1 - parity])
        and np.array_equal(output_mask, geometry.PARITY_MASKS[parity])
        for parity in (0, 1)
    )
    if not full and not xfit:
        raise NonGridV22Error("fit/output masks are not a frozen full or checkerboard context")
    return _readonly_bool(fit_mask), _readonly_bool(output_mask)


def _fit_target(value: np.ndarray, fit_count: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (geometry.CHANNELS, fit_count)
        or value.dtype != np.dtype(np.float64)
        or not value.flags.c_contiguous
    ):
        raise NonGridV22Error("fit target has the wrong shape, dtype, or order")
    return _readonly_float64(value)


def _mask_bytes(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8, order="C").tobytes(order="C")


def array_sha256(scope_sha256: str, role: str, value: np.ndarray) -> str:
    """Hash one exact scientific ndarray with the generic v2.2 grammar."""

    _require_sha256(scope_sha256, "array scope SHA-256")
    if not isinstance(role, str) or not role:
        raise NonGridV22Error("array role must be nonempty ASCII")
    try:
        role_bytes = role.encode("ascii")
    except UnicodeEncodeError as error:
        raise NonGridV22Error("array role must be nonempty ASCII") from error
    if len(role_bytes) > 65_535:
        raise NonGridV22Error("array role is too long")
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise NonGridV22Error("scientific array must be a C-contiguous NumPy array")
    array = value
    if array.dtype == np.dtype(np.bool_) or array.dtype == np.dtype(np.uint8):
        normalized = np.asarray(array, dtype=np.uint8, order="C")
        if np.any((normalized != 0) & (normalized != 1)) and array.dtype == np.dtype(np.bool_):
            raise NonGridV22Error("boolean array normalization failed")
        dtype_byte = 0
        payload = normalized.tobytes(order="C")
    elif array.dtype == np.dtype(np.uint16):
        dtype_byte = 1
        payload = np.asarray(array, dtype="<u2", order="C").tobytes(order="C")
    elif array.dtype == np.dtype(np.uint32):
        dtype_byte = 2
        payload = np.asarray(array, dtype="<u4", order="C").tobytes(order="C")
    elif array.dtype == np.dtype(np.int64):
        dtype_byte = 3
        payload = np.asarray(array, dtype="<i8", order="C").tobytes(order="C")
    elif array.dtype == np.dtype(np.float64):
        if not np.all(np.isfinite(array)):
            raise NonGridV22Error("float64 scientific array must be finite")
        dtype_byte = 4
        payload = np.asarray(array, dtype="<f8", order="C").tobytes(order="C")
    else:
        raise NonGridV22Error("scientific array dtype is outside the frozen grammar")
    if array.ndim > 255 or any(size > 2**32 - 1 for size in array.shape):
        raise NonGridV22Error("scientific array rank or dimension is too large")
    digest = hashlib.sha256()
    digest.update(ARRAY_TAG)
    digest.update(bytes.fromhex(scope_sha256))
    digest.update(struct.pack("<H", len(role_bytes)))
    digest.update(role_bytes)
    digest.update(struct.pack("<BB", dtype_byte, array.ndim))
    for size in array.shape:
        digest.update(struct.pack("<I", size))
    digest.update(payload)
    return digest.hexdigest()


def nongrid_scope_sha256(
    source: np.ndarray | None,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: NonGridArm,
    *,
    config_sha256: str,
) -> str:
    """Construct the target-bounded non-grid scope without hidden inputs."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    fitted, output = _masks(fit_mask, output_mask)
    target = _fit_target(fit_target, int(np.count_nonzero(fitted)))
    if arm not in _ARM_BYTE:
        raise NonGridV22Error("non-grid arm must be appearance or bias_only")
    if arm == "appearance":
        if source is None:
            raise NonGridV22Error("appearance scope requires its source")
        current = _source(source)
        source_present = 1
    else:
        if source is not None:
            raise NonGridV22Error("bias-only scope may neither include nor read a source")
        current = None
        source_present = 0
    digest = hashlib.sha256()
    digest.update(NONGRID_SCOPE_TAG)
    digest.update(struct.pack("<B", source_present))
    if current is not None:
        digest.update(current.tobytes(order="C"))
    digest.update(_mask_bytes(fitted))
    digest.update(_mask_bytes(output))
    digest.update(struct.pack("<H", target.shape[1]))
    digest.update(target.tobytes(order="C"))
    digest.update(struct.pack("<B", _ARM_BYTE[arm]))
    digest.update(bytes.fromhex(checked_config))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AppearanceEstimate:
    scope_sha256: str
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    retained_macro_ids: tuple[int, ...]
    fit_prediction: np.ndarray
    prediction: np.ndarray
    objective: float
    hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.scope_sha256, "appearance scope SHA-256")
        parameters = _readonly_float64(self.parameters)
        gains = _readonly_float64(self.gains)
        biases = _readonly_float64(self.biases)
        fit_prediction = _readonly_float64(self.fit_prediction)
        prediction = _readonly_float64(self.prediction)
        if parameters.shape != (6,) or not np.array_equal(parameters, np.zeros(6)):
            raise NonGridV22Error("appearance geometry must be the exact identity")
        if gains.shape != (3,) or biases.shape != (3,):
            raise NonGridV22Error("appearance gains/biases must be three-vectors")
        if fit_prediction.ndim != 2 or prediction.ndim != 2 or fit_prediction.shape[0] != 3 or prediction.shape[0] != 3:
            raise NonGridV22Error("appearance predictions have invalid shapes")
        if len(self.retained_macro_ids) not in (14, 27):
            raise NonGridV22Error("appearance retained macro count is invalid")
        if not np.isfinite(self.objective) or self.objective < 0.0:
            raise NonGridV22Error("appearance objective is invalid")
        if tuple(name for name, _ in self.hashes) != tuple(sorted(name for name, _ in self.hashes)):
            raise NonGridV22Error("appearance hash roles must be sorted")
        for _, digest in self.hashes:
            _require_sha256(digest, "appearance array SHA-256")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "fit_prediction", fit_prediction)
        object.__setattr__(self, "prediction", prediction)


@dataclass(frozen=True, slots=True)
class BiasOnlyEstimate:
    scope_sha256: str
    first_biases: np.ndarray
    biases: np.ndarray
    retained_macro_ids: tuple[int, ...]
    prediction: np.ndarray
    objective: float
    hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.scope_sha256, "bias-only scope SHA-256")
        first = _readonly_float64(self.first_biases)
        biases = _readonly_float64(self.biases)
        prediction = _readonly_float64(self.prediction)
        if first.shape != (3,) or biases.shape != (3,) or prediction.ndim != 2 or prediction.shape[0] != 3:
            raise NonGridV22Error("bias-only values have invalid shapes")
        if len(self.retained_macro_ids) not in (14, 27):
            raise NonGridV22Error("bias-only retained macro count is invalid")
        if not np.isfinite(self.objective) or self.objective < 0.0:
            raise NonGridV22Error("bias-only objective is invalid")
        for _, digest in self.hashes:
            _require_sha256(digest, "bias-only array SHA-256")
        object.__setattr__(self, "first_biases", first)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "prediction", prediction)


@dataclass(frozen=True, slots=True)
class PersistenceEstimate:
    scope_sha256: str
    prediction: np.ndarray
    prediction_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.scope_sha256, "persistence scope SHA-256")
        _require_sha256(self.prediction_sha256, "persistence prediction SHA-256")
        prediction = _readonly_float64(self.prediction)
        if prediction.ndim != 2 or prediction.shape[0] != 3:
            raise NonGridV22Error("persistence prediction has an invalid shape")
        object.__setattr__(self, "prediction", prediction)


def fit_appearance(
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> AppearanceEstimate:
    current = _source(source)
    fitted, output = _masks(fit_mask, output_mask)
    target = _fit_target(fit_target, int(np.count_nonzero(fitted)))
    scope = nongrid_scope_sha256(current, target, fitted, output, "appearance", config_sha256=config_sha256)
    fit_sample = geometry.sample_scalar(current, 0, fitted)[None, :, :]
    result = fitting.fit_appearance(fit_sample, target, fitted)
    output_sample = geometry.sample_scalar(current, 0, output)
    prediction = result.gains[:, None] * output_sample + result.biases[:, None]
    parameters = np.zeros(6, dtype=np.float64)
    retained = np.asarray(result.retained_macro_ids, dtype=np.int64)
    values = {
        "appearance_biases": result.biases,
        "appearance_fit_prediction": result.prediction,
        "appearance_gains": result.gains,
        "appearance_parameters": parameters,
        "appearance_prediction": prediction,
        "appearance_retained_ids": retained,
    }
    hashes = tuple((role, array_sha256(scope, role, value)) for role, value in sorted(values.items()))
    return AppearanceEstimate(
        scope,
        parameters,
        result.gains,
        result.biases,
        result.retained_macro_ids,
        result.prediction,
        prediction,
        result.objective,
        hashes,
    )


def fit_bias_only(
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> BiasOnlyEstimate:
    fitted, output = _masks(fit_mask, output_mask)
    target = _fit_target(fit_target, int(np.count_nonzero(fitted)))
    scope = nongrid_scope_sha256(None, target, fitted, output, "bias_only", config_sha256=config_sha256)
    result = fitting.fit_bias_only(target, fitted, output)
    retained = np.asarray(result.retained_macro_ids, dtype=np.int64)
    values = {
        "bias_only_biases": result.biases,
        "bias_only_first_biases": result.first_biases,
        "bias_only_prediction": result.prediction,
        "bias_only_retained_ids": retained,
    }
    hashes = tuple((role, array_sha256(scope, role, value)) for role, value in sorted(values.items()))
    return BiasOnlyEstimate(
        scope,
        result.first_biases,
        result.biases,
        result.retained_macro_ids,
        result.prediction,
        result.objective,
        hashes,
    )


def persistence(source: np.ndarray, output_mask: np.ndarray, *, config_sha256: str) -> PersistenceEstimate:
    current = _source(source)
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if (
        not isinstance(output_mask, np.ndarray)
        or output_mask.shape != (geometry.SITE_COUNT,)
        or output_mask.dtype != np.dtype(np.bool_)
        or not output_mask.flags.c_contiguous
        or not any(np.array_equal(output_mask, mask) for mask in (geometry.FULL_MASK, *geometry.PARITY_MASKS))
    ):
        raise NonGridV22Error("persistence output mask is not full/parity central geometry")
    output = _readonly_bool(output_mask)
    digest = hashlib.sha256()
    digest.update(PERSISTENCE_SCOPE_TAG)
    digest.update(current.tobytes(order="C"))
    digest.update(_mask_bytes(output))
    digest.update(bytes.fromhex(checked_config))
    scope = digest.hexdigest()
    prediction = geometry.sample_scalar(current, 0, output)
    prediction_sha256 = array_sha256(scope, "persistence_prediction", prediction)
    return PersistenceEstimate(scope, prediction, prediction_sha256)


__all__ = [
    "AppearanceEstimate",
    "BiasOnlyEstimate",
    "NonGridV22Error",
    "PersistenceEstimate",
    "array_sha256",
    "fit_appearance",
    "fit_bias_only",
    "nongrid_scope_sha256",
    "persistence",
]
