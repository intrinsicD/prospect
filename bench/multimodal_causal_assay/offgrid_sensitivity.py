"""Frozen target-free off-grid sensitivity audit for the MM-011 candidate.

This audit is deliberately smaller than the real assay and accepts no frame path.
It asks whether the finite affine grid can identify and directionally extrapolate six
declared continuous transforms before any LCV/MM-007 scientific array is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

import numpy as np

from bench.multimodal_causal_assay import predictor, records
from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry

SCHEMA_VERSION: Final = "mm011-offgrid-sensitivity-v1"
CONFIG_SHA256: Final = hashlib.sha256(b"MM011-offgrid-sensitivity-config-v1\0").hexdigest()
PRIMARY_FACTOR: Final = 1.25
ACTIVITY_SSE_MIN: Final = 1e-4 * geometry.CHANNELS * geometry.SITE_COUNT
NULL_SSE_MAX: Final = 1e-24
HALO_SIZE: Final = 96
HALO_OFFSET: Final = 16
ROW: Final = 0
REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SOURCE_PATHS: Final[tuple[Path, ...]] = tuple(
    Path(name)
    for name in (
        "bench/multimodal_causal_assay/offgrid_sensitivity.py",
        "bench/multimodal_causal_assay/predictor.py",
        "bench/multimodal_causal_assay/records.py",
        "bench/multimodal_mechanism_diagnostics/calibration_v22.py",
        "bench/multimodal_mechanism_diagnostics/fitting_v22.py",
        "bench/multimodal_mechanism_diagnostics/geometry_v22.py",
        "bench/multimodal_mechanism_diagnostics/global_v22.py",
        "bench/multimodal_mechanism_diagnostics/nongrid_v22.py",
    )
)

CASE_SPECS: Final[tuple[tuple[int, str, tuple[float, ...]], ...]] = (
    (991_100, "T1", (1.0, -1.0, 0.0, 0.0, 0.0, 0.0)),
    (991_101, "T2", (2.0, -2.0, 0.0, 0.0, 0.0, 0.0)),
    (991_102, "T3", (3.0, -3.0, 0.0, 0.0, 0.0, 0.0)),
    (991_100, "A1", (0.0, 0.0, 1.0, 0.0, 0.0, -1.0)),
    (991_101, "A2", (0.0, 0.0, 0.0, 1.0, -1.0, 0.0)),
    (991_102, "A3", (0.0, 0.0, 1.0, -1.0, 1.0, 1.0)),
)

JsonValue = records.JsonValue


class OffgridSensitivityError(ValueError):
    """Raised when the generated audit or its durable receipt differs."""


def _require_protocol_sha256(value: str) -> str:
    try:
        return records.require_sha256(value, "off-grid protocol SHA-256")
    except records.RecordValidationError as error:
        raise OffgridSensitivityError(str(error)) from error


def _json_digest(value: JsonValue, *, protocol_sha256: str) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"MM011-offgrid-sensitivity-evidence-v1\0")
    digest.update(bytes.fromhex(_require_protocol_sha256(protocol_sha256)))
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"MM011-offgrid-sensitivity-array-v1\0")
    digest.update(array.shape.__repr__().encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _theta_array(theta: Sequence[float]) -> np.ndarray:
    value = np.ascontiguousarray(theta, dtype="<f8")
    if value.shape != (6,) or not bool(np.all(np.isfinite(value))):
        raise OffgridSensitivityError("off-grid theta must contain six finite float64 values")
    try:
        geometry.state_index(value)
    except geometry.GeometryValidationError:
        return value
    raise OffgridSensitivityError("declared off-grid theta is a canonical grid member")


def _sample_continuous(
    source: np.ndarray,
    theta: Sequence[float],
    y: np.ndarray,
    x: np.ndarray,
    *,
    source_offset: int,
) -> np.ndarray:
    """Bilinearly sample one arbitrary finite theta at declared local coordinates."""

    frame = np.asarray(source)
    if (
        frame.ndim != 3
        or frame.shape[0] != geometry.CHANNELS
        or frame.dtype != np.dtype(np.float64)
        or not frame.flags.c_contiguous
        or not bool(np.all(np.isfinite(frame)))
    ):
        raise OffgridSensitivityError("continuous sampler source differs")
    yy = np.ascontiguousarray(y, dtype="<f8").reshape(-1)
    xx = np.ascontiguousarray(x, dtype="<f8").reshape(-1)
    if yy.shape != xx.shape or not bool(np.all(np.isfinite(yy))) or not bool(np.all(np.isfinite(xx))):
        raise OffgridSensitivityError("continuous sampler coordinates differ")
    parameters = _theta_array(theta)
    u = (yy - 31.5) / 23.5
    v = (xx - 31.5) / 23.5
    dy = parameters[0] + parameters[2] * u + parameters[3] * v
    dx = parameters[1] + parameters[4] * u + parameters[5] * v
    source_y = yy - dy + float(source_offset)
    source_x = xx - dx + float(source_offset)
    if (
        bool(np.any(source_y < 0.0))
        or bool(np.any(source_x < 0.0))
        or bool(np.any(source_y > frame.shape[1] - 1))
        or bool(np.any(source_x > frame.shape[2] - 1))
    ):
        raise OffgridSensitivityError("off-grid fixture is not boundary-safe")
    y0 = np.floor(source_y).astype(np.intp)
    x0 = np.floor(source_x).astype(np.intp)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = source_y - y0
    wx = source_x - x0
    top = frame[:, y0, x0] * (1.0 - wx)[None, :] + frame[:, y0, x1] * wx[None, :]
    bottom = frame[:, y1, x0] * (1.0 - wx)[None, :] + frame[:, y1, x1] * wx[None, :]
    return np.ascontiguousarray(top * (1.0 - wy)[None, :] + bottom * wy[None, :], dtype="<f8")


def _full_coordinates() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(geometry.NATIVE_SIZE, dtype="<f8"),
        np.arange(geometry.NATIVE_SIZE, dtype="<f8"),
        indexing="ij",
    )
    return yy.reshape(-1), xx.reshape(-1)


def _central_coordinates() -> tuple[np.ndarray, np.ndarray]:
    return geometry.GEOMETRY.coords[:, 0], geometry.GEOMETRY.coords[:, 1]


def _fixture(seed: int) -> tuple[np.ndarray, calibration.BroadbandMetrics, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    epsilon = rng.standard_normal(
        size=(calibration.SYNTHETIC_ROWS, geometry.CHANNELS, HALO_SIZE, HALO_SIZE)
    )
    offsets = rng.standard_normal(size=(calibration.SYNTHETIC_ROWS, geometry.CHANNELS, 1, 1))
    raw_halo = np.ascontiguousarray(epsilon + 0.5 * offsets, dtype="<f8")
    raw_crop = np.ascontiguousarray(
        raw_halo[:, :, HALO_OFFSET : HALO_OFFSET + 64, HALO_OFFSET : HALO_OFFSET + 64],
        dtype="<f8",
    )
    normalizer = calibration.fit_source_only_normalizer(raw_crop)
    normalized_crop = normalizer.apply(raw_crop)
    normalized_halo = np.ascontiguousarray(
        (raw_halo - normalizer.mean) / normalizer.scale,
        dtype="<f8",
    )
    metrics = calibration.broadband_validity_metrics(normalized_crop)
    if metrics.failure_reasons():
        raise OffgridSensitivityError(
            f"seed {seed} failed broadband generation: {metrics.failure_reasons()}"
        )
    return normalized_crop[ROW], metrics, normalized_halo[ROW]


def _sse(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype="<f8")
    b = np.asarray(right, dtype="<f8")
    if a.shape != b.shape or not bool(np.all(np.isfinite(a))) or not bool(np.all(np.isfinite(b))):
        raise OffgridSensitivityError("SSE operands differ")
    difference = a - b
    value = float(np.sum(difference * difference, dtype=np.float64))
    if not math.isfinite(value) or value < 0.0:
        raise OffgridSensitivityError("SSE is invalid")
    return value


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise OffgridSensitivityError("positive-case ratio denominator is not positive")
    value = numerator / denominator
    if not math.isfinite(value) or value < 0.0:
        raise OffgridSensitivityError("positive-case ratio is invalid")
    return value


def _broadband_summary(metrics: calibration.BroadbandMetrics) -> dict[str, JsonValue]:
    return {
        "failure_reasons": list(metrics.failure_reasons()),
        "matrix_rank": metrics.matrix_rank,
        "minimum_central_variance": min(metrics.central_variance),
        "minimum_row_rms": min(metrics.row_rms),
        "singular_value_ratio": metrics.singular_value_ratio,
        "maximum_absolute_lag_correlation": max(abs(value) for value in metrics.lag_correlation),
    }


def _positive_case(
    seed: int,
    name: str,
    theta: Sequence[float],
    fixture: tuple[np.ndarray, calibration.BroadbandMetrics, np.ndarray],
) -> dict[str, JsonValue]:
    previous, metrics, halo = fixture
    full_y, full_x = _full_coordinates()
    central_y, central_x = _central_coordinates()
    current = _sample_continuous(
        halo,
        theta,
        full_y,
        full_x,
        source_offset=HALO_OFFSET,
    ).reshape(geometry.CHANNELS, geometry.NATIVE_SIZE, geometry.NATIVE_SIZE)
    future = _sample_continuous(current, theta, central_y, central_x, source_offset=0)
    source_result = predictor.fit_source_pair(previous, current, config_sha256=CONFIG_SHA256)
    predictor.validate_source_pair_result(
        source_result,
        previous,
        current,
        config_sha256=CONFIG_SHA256,
    )
    affine = source_result.affine
    replay = predictor.apply_operator_once(
        current,
        affine.parameters,
        affine.gains,
        affine.biases,
    )
    replay_exact = replay.tobytes(order="C") == affine.forecast.tobytes(order="C")
    previous_sites = geometry.sample_scalar(previous, 0, geometry.FULL_MASK)
    current_sites = geometry.sample_scalar(current, 0, geometry.FULL_MASK)
    identity_sse = _sse(previous_sites, current_sites)
    history_sse = _sse(affine.history_reconstruction, current_sites)
    persistence_sse = _sse(current_sites, future)
    forecast_sse = _sse(affine.forecast, future)
    reverse_persistence_sse = _sse(current_sites, previous_sites)
    reverse_forecast_sse = _sse(affine.forecast, previous_sites)
    predicates: dict[str, JsonValue] = {
        "activity": identity_sse > ACTIVITY_SSE_MIN and persistence_sse > ACTIVITY_SSE_MIN,
        "directionality": PRIMARY_FACTOR * reverse_forecast_sse > reverse_persistence_sse,
        "forecast_replay_bit_exact": replay_exact,
        "forward_margin": PRIMARY_FACTOR * forecast_sse <= persistence_sse,
        "historical_margin": PRIMARY_FACTOR * history_sse <= identity_sse,
    }
    return {
        "broadband": _broadband_summary(metrics),
        "case": name,
        "errors": {
            "forecast": forecast_sse,
            "historical": history_sse,
            "identity": identity_sse,
            "persistence": persistence_sse,
            "reverse_forecast": reverse_forecast_sse,
            "reverse_persistence": reverse_persistence_sse,
        },
        "input_sha256": {
            "current": _array_sha256(current),
            "future": _array_sha256(future),
            "previous": _array_sha256(previous),
        },
        "passed": all(value is True for value in predicates.values()),
        "predicates": predicates,
        "ratios": {
            "forward": _ratio(forecast_sse, persistence_sse),
            "historical": _ratio(history_sse, identity_sse),
            "reversal": _ratio(reverse_forecast_sse, reverse_persistence_sse),
        },
        "seed": seed,
        "selected_theta": [float(value) for value in affine.parameters],
        "truth_theta": [float(value) for value in _theta_array(theta)],
    }


def _null_control(
    name: str,
    frame: np.ndarray,
    *,
    require_every_arm: bool,
) -> dict[str, JsonValue]:
    source_result = predictor.fit_source_pair(frame, frame, config_sha256=CONFIG_SHA256)
    predictor.validate_source_pair_result(
        source_result,
        frame,
        frame,
        config_sha256=CONFIG_SHA256,
    )
    target = geometry.sample_scalar(frame, 0, geometry.FULL_MASK)
    previous_sites = target
    current_sites = target
    identity_sse = _sse(previous_sites, current_sites)
    persistence_sse = _sse(current_sites, target)
    arm_sse = {
        arm: _sse(source_result.operator(cast(predictor.Arm, arm)).forecast, target)
        for arm in ("affine", "appearance", "combined")
    }
    identity_theta = [0.0] * 6
    predicates: dict[str, JsonValue] = {
        "affine_identity": list(source_result.affine.parameters) == identity_theta,
        "affine_null_sse": (
            _sse(source_result.affine.history_reconstruction, target) <= NULL_SSE_MAX
            and arm_sse["affine"] <= NULL_SSE_MAX
        ),
        "ineligible_zero_activity": (
            identity_sse == 0.0
            and persistence_sse == 0.0
            and not (identity_sse > ACTIVITY_SSE_MIN and persistence_sse > ACTIVITY_SSE_MIN)
        ),
    }
    if require_every_arm:
        predicates["every_arm_null_sse"] = all(value <= NULL_SSE_MAX for value in arm_sse.values())
    return {
        "arm_forecast_sse": cast(dict[str, JsonValue], arm_sse),
        "control": name,
        "identity_sse": identity_sse,
        "passed": all(value is True for value in predicates.values()),
        "persistence_sse": persistence_sse,
        "predicates": predicates,
        "selected_theta": [float(value) for value in source_result.affine.parameters],
    }


def config_record() -> dict[str, JsonValue]:
    return {
        "activity_sse_min": ACTIVITY_SSE_MIN,
        "case_specs": cast(
            list[JsonValue],
            [
                {"case": name, "seed": seed, "truth_theta": list(theta)}
                for seed, name, theta in CASE_SPECS
            ],
        ),
        "config_sha256": CONFIG_SHA256,
        "generator": {
            "crop": [HALO_OFFSET, HALO_OFFSET + 64],
            "halo_shape": [calibration.SYNTHETIC_ROWS, geometry.CHANNELS, HALO_SIZE, HALO_SIZE],
            "normalizer": "v2.2_source_only_fit_on_six_crops_then_applied_to_halo",
            "offset_scale": 0.5,
            "row": ROW,
            "sampler": "finite_float64_bilinear_no_padding_reflection_or_clipping",
        },
        "null_sse_max": NULL_SSE_MAX,
        "primary_factor": PRIMARY_FACTOR,
        "required_positive_passes": len(CASE_SPECS),
        "schema_version": SCHEMA_VERSION,
        "source_sha256": {
            str(relative): records.file_sha256(REPO_ROOT / relative)
            for relative in SOURCE_PATHS
        },
    }


def run(*, protocol_sha256: str) -> dict[str, JsonValue]:
    """Execute the frozen generated audit and return canonicalizable evidence."""

    protocol = _require_protocol_sha256(protocol_sha256)
    fixtures = {seed: _fixture(seed) for seed in sorted({item[0] for item in CASE_SPECS})}
    positives = [
        _positive_case(seed, name, theta, fixtures[seed])
        for seed, name, theta in CASE_SPECS
    ]
    identity_frame = fixtures[991_100][0]
    constant = np.empty((geometry.CHANNELS, 64, 64), dtype="<f8")
    constant[:] = np.asarray((-0.25, 0.0, 0.25), dtype="<f8")[:, None, None]
    controls = [
        _null_control("broadband_identity", identity_frame, require_every_arm=False),
        _null_control("constant_low_texture", constant, require_every_arm=True),
    ]
    failure_codes = [
        f"positive:{cast(str, item['case'])}"
        for item in positives
        if item["passed"] is not True
    ]
    failure_codes.extend(
        f"control:{cast(str, item['control'])}"
        for item in controls
        if item["passed"] is not True
    )
    body: dict[str, JsonValue] = {
        "claim_boundary": (
            "generated target-free finite-grid sensitivity only; no real LCV/MM-007 frame, "
            "population, learned-model, or end-to-end Prospect claim"
        ),
        "config": config_record(),
        "controls": cast(list[JsonValue], controls),
        "decision": (
            "ABANDON_FINITE_GRID_BEFORE_REAL_DATA"
            if failure_codes
            else "PERMIT_MM011_PRE_REAL_FREEZE"
        ),
        "experiment_id": "MM-011-OFFGRID",
        "failure_codes": cast(list[JsonValue], failure_codes),
        "positive_cases": cast(list[JsonValue], positives),
        "protocol_sha256": protocol,
        "real_scientific_frame_inputs": "forbidden_not_accepted_by_api",
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
    }
    return {**body, "evidence_sha256": _json_digest(body, protocol_sha256=protocol)}


def validate_receipt(value: object, *, protocol_sha256: str) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise OffgridSensitivityError("off-grid receipt must be an object")
    receipt = cast(dict[str, JsonValue], value)
    evidence = receipt.get("evidence_sha256")
    body = {name: item for name, item in receipt.items() if name != "evidence_sha256"}
    if (
        set(receipt) != {
            "claim_boundary",
            "config",
            "controls",
            "decision",
            "evidence_sha256",
            "experiment_id",
            "failure_codes",
            "positive_cases",
            "protocol_sha256",
            "real_scientific_frame_inputs",
            "schema_version",
            "status",
        }
        or receipt.get("protocol_sha256") != _require_protocol_sha256(protocol_sha256)
        or receipt.get("config") != config_record()
        or evidence != _json_digest(body, protocol_sha256=protocol_sha256)
    ):
        raise OffgridSensitivityError("off-grid receipt structure or digest differs")
    return receipt


def write_receipt(path: Path, value: Mapping[str, JsonValue]) -> None:
    """Write one exclusive immutable receipt so UI output is never the authority."""

    try:
        records.write_immutable_json_exclusive(path, dict(value))
    except (OSError, records.RecordValidationError) as error:
        raise OffgridSensitivityError(f"could not write off-grid receipt: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run(protocol_sha256=arguments.protocol_sha256)
    validate_receipt(result, protocol_sha256=arguments.protocol_sha256)
    write_receipt(arguments.output, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "evidence_sha256": result["evidence_sha256"],
                "output": str(arguments.output),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVITY_SSE_MIN",
    "CASE_SPECS",
    "CONFIG_SHA256",
    "NULL_SSE_MAX",
    "OffgridSensitivityError",
    "config_record",
    "run",
    "validate_receipt",
    "write_receipt",
]
