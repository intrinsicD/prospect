"""Clean MM-007 R64 translation sentinels for the MM-008 v2.2 archive."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import nongrid_v22 as nongrid

PROTOCOL_SHA256: Final = (
    "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
)
SCHEMA_VERSION: Final = "mm008-v2.2-sentinel-v1"
NONGRID_SCOPE_TAG: Final = b"MM008-v2.2-nongrid-fit-scope\0"
LOCAL_REGULARIZER: Final = 0.05
TRIM_FRACTION: Final = 0.25

SentinelArm = Literal["global_translation", "quadrant_translation"]
_ARM_BYTE: Final[dict[SentinelArm, int]] = {"global_translation": 0, "quadrant_translation": 1}
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")


class SentinelV22Error(ValueError):
    """Raised when a sentinel input or result violates the frozen contract."""


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, dtype="<f8", order="C", copy=True)
    if not np.all(np.isfinite(contiguous)):
        raise SentinelV22Error("sentinel scientific array contains a nonfinite value")
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8")
    return immutable.reshape(contiguous.shape)


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, dtype=np.bool_, order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)
    return immutable.reshape(contiguous.shape)


def _readonly_uint8(value: np.ndarray) -> np.ndarray:
    contiguous = np.array(value, dtype=np.uint8, order="C", copy=True)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.uint8)
    return immutable.reshape(contiguous.shape)


def _require_sha256(value: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise SentinelV22Error("config SHA-256 must contain 64 lowercase hexadecimal characters")
    return value


def _source(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (3, 64, 64)
        or value.dtype != np.dtype(np.float64)
        or not value.flags.c_contiguous
    ):
        raise SentinelV22Error("sentinel source must be C-contiguous float64 [3,64,64]")
    return _readonly_float64(value)


def _masks(fit_mask: np.ndarray, output_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fit_mask, np.ndarray) or not isinstance(output_mask, np.ndarray):
        raise SentinelV22Error("sentinel masks must be NumPy arrays")
    if any(
        mask.shape != (geometry.SITE_COUNT,)
        or mask.dtype != np.dtype(np.bool_)
        or not mask.flags.c_contiguous
        for mask in (fit_mask, output_mask)
    ):
        raise SentinelV22Error("sentinel masks must be C-contiguous bool [2304]")
    full = np.array_equal(fit_mask, geometry.FULL_MASK) and np.array_equal(output_mask, geometry.FULL_MASK)
    xfit = any(
        np.array_equal(fit_mask, geometry.PARITY_MASKS[1 - parity])
        and np.array_equal(output_mask, geometry.PARITY_MASKS[parity])
        for parity in (0, 1)
    )
    if not full and not xfit:
        raise SentinelV22Error("sentinel masks are not full or frozen checkerboard cross-fit")
    return _readonly_bool(fit_mask), _readonly_bool(output_mask)


_macro_y = ((geometry.GEOMETRY.coords[:, 0].astype(np.int64) - 8) // 8).astype(np.int64)
_macro_x = ((geometry.GEOMETRY.coords[:, 1].astype(np.int64) - 8) // 8).astype(np.int64)
TILE_IDS: Final = _readonly_uint8((_macro_y >= 3).astype(np.uint8) * 2 + (_macro_x >= 3))

TRANSLATIONS: Final = tuple(
    sorted(
        ((float(dy), float(dx)) for dy in (-8, -4, 0, 4, 8) for dx in (-8, -4, 0, 4, 8)),
        key=lambda pair: (pair[0] * pair[0] + pair[1] * pair[1], *pair),
    )
)


def _trimmed_loss(residual: np.ndarray, macro_ids: np.ndarray) -> float:
    if residual.ndim != 2 or residual.shape[0] != 3 or residual.shape[1] != len(macro_ids):
        raise SentinelV22Error("sentinel residual/macro shapes differ")
    pixel = np.asarray(np.mean(residual * residual, axis=0, dtype=np.float64), dtype=np.float64)
    ids = np.unique(macro_ids)
    losses = np.asarray([np.mean(pixel[macro_ids == macro], dtype=np.float64) for macro in ids])
    keep = len(ids) - math.floor(len(ids) * TRIM_FRACTION)
    return float(np.mean(np.sort(losses, kind="stable")[:keep], dtype=np.float64))


def _theta(translation: tuple[float, float]) -> tuple[float, float, float, float, float, float]:
    return (translation[0], translation[1], 0.0, 0.0, 0.0, 0.0)


def _select(
    source: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    anchor: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    macro_ids = geometry.GEOMETRY.macro_ids[selected]
    if target.shape != (3, int(np.count_nonzero(selected))):
        raise SentinelV22Error("sentinel permitted target does not match its selected cells")
    raw_losses = np.empty(len(TRANSLATIONS), dtype=np.float64)
    for index, translation in enumerate(TRANSLATIONS):
        prediction = geometry.sample_scalar(source, _theta(translation), selected)
        raw_losses[index] = _trimmed_loss(prediction - target, macro_ids)
    identity = raw_losses[TRANSLATIONS.index((0.0, 0.0))]
    losses = raw_losses.copy()
    candidates = np.asarray(TRANSLATIONS, dtype=np.float64)
    if anchor is not None:
        if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
            raise SentinelV22Error("quadrant anchor is invalid")
        distance = np.sum((candidates - anchor[None, :]) ** 2, axis=1, dtype=np.float64) / 64.0
        losses += LOCAL_REGULARIZER * (identity + 1e-12) * distance
    best_index = int(np.argmin(losses))
    best = candidates[best_index]
    separated = np.sum((candidates - best[None, :]) ** 2, axis=1, dtype=np.float64) >= 16.0
    alternative = float(np.min(np.where(separated, losses, np.inf)))
    confidence = alternative - float(losses[best_index])
    if not math.isfinite(confidence) or confidence < 0.0:
        raise SentinelV22Error("sentinel confidence is invalid")
    return _readonly_float64(best), confidence


def _scope(
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: SentinelArm,
    config_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(NONGRID_SCOPE_TAG)
    digest.update(b"\x01")
    digest.update(source.tobytes(order="C"))
    digest.update(np.asarray(fit_mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(np.asarray(output_mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(struct.pack("<H", fit_target.shape[1]))
    digest.update(fit_target.tobytes(order="C"))
    digest.update(struct.pack("<B", _ARM_BYTE[arm]))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SentinelEstimate:
    arm: SentinelArm
    scope_sha256: str
    flow: np.ndarray
    confidence: np.ndarray
    prediction: np.ndarray
    hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.arm not in _ARM_BYTE:
            raise SentinelV22Error("sentinel arm is invalid")
        flow = _readonly_float64(self.flow)
        confidence = _readonly_float64(self.confidence)
        prediction = _readonly_float64(self.prediction)
        family_size = 1 if self.arm == "global_translation" else 4
        if flow.shape != (family_size, 2) or confidence.shape != (family_size,):
            raise SentinelV22Error("sentinel flow/confidence shapes are invalid")
        if prediction.ndim != 2 or prediction.shape[0] != 3:
            raise SentinelV22Error("sentinel prediction has an invalid shape")
        if np.any(confidence < 0.0):
            raise SentinelV22Error("sentinel confidence must be nonnegative")
        if tuple(name for name, _ in self.hashes) != tuple(sorted(name for name, _ in self.hashes)):
            raise SentinelV22Error("sentinel hash roles must be sorted")
        object.__setattr__(self, "flow", flow)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "prediction", prediction)


def fit_sentinel(
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: SentinelArm,
    *,
    config_sha256: str,
) -> SentinelEstimate:
    """Fit one deterministic target-bounded global or quadrant translation sentinel."""

    if arm not in _ARM_BYTE:
        raise SentinelV22Error("sentinel arm must be global_translation or quadrant_translation")
    checked_config = _require_sha256(config_sha256)
    current = _source(source)
    fitted, output = _masks(fit_mask, output_mask)
    if (
        not isinstance(fit_target, np.ndarray)
        or fit_target.shape != (3, int(np.count_nonzero(fitted)))
        or fit_target.dtype != np.dtype(np.float64)
        or not fit_target.flags.c_contiguous
        or not np.all(np.isfinite(fit_target))
    ):
        raise SentinelV22Error("sentinel fit target has the wrong shape, dtype, order, or values")
    target = _readonly_float64(fit_target)
    scope = _scope(current, target, fitted, output, arm, checked_config)
    global_flow, global_confidence = _select(current, target, fitted, None)
    if arm == "global_translation":
        flow = global_flow[None, :]
        confidence = np.asarray((global_confidence,), dtype=np.float64)
        prediction = geometry.sample_scalar(current, _theta(tuple(global_flow)), output)
    else:
        fit_positions = np.flatnonzero(fitted)
        flow = np.empty((4, 2), dtype=np.float64)
        confidence = np.empty(4, dtype=np.float64)
        for tile in range(4):
            selected = fitted & (TILE_IDS == tile)
            local_target = target[:, TILE_IDS[fit_positions] == tile]
            flow[tile], confidence[tile] = _select(current, local_target, selected, global_flow)
        output_positions = np.flatnonzero(output)
        prediction = np.empty((3, len(output_positions)), dtype=np.float64)
        for tile in range(4):
            selected = output & (TILE_IDS == tile)
            prediction[:, TILE_IDS[output_positions] == tile] = geometry.sample_scalar(
                current, _theta(tuple(flow[tile])), selected
            )
    values = {
        "sentinel_confidence": confidence,
        "sentinel_flow": flow,
        "sentinel_prediction": prediction,
    }
    hashes = tuple((role, nongrid.array_sha256(scope, role, value)) for role, value in sorted(values.items()))
    return SentinelEstimate(arm, scope, flow, confidence, prediction, hashes)


__all__ = [
    "SentinelEstimate",
    "SentinelV22Error",
    "TILE_IDS",
    "TRANSLATIONS",
    "fit_sentinel",
]
