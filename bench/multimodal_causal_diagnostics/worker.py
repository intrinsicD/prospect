"""One-row, source-only MM-009 prediction worker.

The formal launcher imports this module only after establishing Landlock/seccomp.
It consumes one detached source row, emits compact fit/apply evidence plus fixed-order
central predictions, and has no target/scoring/decision import.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np

from bench.multimodal_causal_diagnostics import predictor, preparation, records
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import nongrid_v22 as nongrid

SCHEMA_VERSION: Final = "mm009-source-worker-v1"
PREDICTION_FILE: Final = "predictions.npy"
COMMIT_FILE: Final = "prediction.json"
PREDICTION_ROLES: Final = (
    "history_identity",
    "history_xfit_affine",
    "history_xfit_appearance",
    "history_xfit_combined",
    "history_shuffle_xfit_affine",
    "history_shuffle_xfit_appearance",
    "history_shuffle_xfit_combined",
    "history_bias",
    "persistence",
    "velocity",
    "forecast_affine",
    "forecast_appearance",
    "forecast_combined",
    "forecast_shuffle_affine",
    "forecast_shuffle_appearance",
    "forecast_shuffle_combined",
    "forecast_reverse_affine",
    "forecast_reverse_appearance",
    "forecast_reverse_combined",
    "forecast_bias",
)
PREDICTION_SHAPE: Final = (len(PREDICTION_ROLES), geometry.CHANNELS, geometry.SITE_COUNT)


class WorkerError(ValueError):
    """Raised when a source-only worker input or compact output fails closed."""


def _reject_constant(value: str) -> None:
    raise WorkerError(f"nonfinite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise WorkerError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def _read_config(path: str | Path) -> dict[str, object]:
    candidate = Path(path)
    try:
        payload = records.read_regular_bytes(candidate, maximum_bytes=64 * 1024)
        value = json.loads(
            payload.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, records.RecordValidationError) as error:
        raise WorkerError("worker config is not strict ASCII JSON") from error
    if type(value) is not dict:
        raise WorkerError("worker config must be an object")
    try:
        if payload != records.json_file_bytes(cast(records.JsonValue, value)):
            raise WorkerError("worker config is not canonical JSON bytes")
    except records.RecordValidationError as error:
        raise WorkerError("worker config is outside canonical JSON") from error
    expected = {
        "config_sha256",
        "prediction_roles",
        "protocol_sha256",
        "schema_version",
    }
    if set(value) != expected or value.get("schema_version") != "mm009-worker-config-v1":
        raise WorkerError("worker config schema differs")
    records.require_sha256(value.get("config_sha256"), "scientific config SHA-256")
    protocol_sha256 = records.require_sha256(value.get("protocol_sha256"), "protocol SHA-256")
    if protocol_sha256 != records.PROTOCOL_SHA256 or protocol_sha256 != preparation.PROTOCOL_SHA256:
        raise WorkerError("worker config protocol binding differs")
    roles = value.get("prediction_roles")
    if type(roles) is not list or tuple(roles) != PREDICTION_ROLES:
        raise WorkerError("worker prediction role order differs")
    return cast(dict[str, object], value)


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise WorkerError(f"{label} must be finite")
    return float(value)


def _array_list(value: np.ndarray) -> list[records.JsonValue]:
    array = np.asarray(value)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise WorkerError("compact parameter arrays must be finite vectors")
    return [float(item) for item in array]


def _global_fit_record(result: exact.GlobalResult) -> dict[str, records.JsonValue]:
    selected = result.selected
    certificate = result.certificate
    return {
        "admissible_rank": selected.admissible_rank,
        "arm": result.arm,
        "biases": None if selected.biases is None else _array_list(selected.biases),
        "certificate": {
            "admissible_count": certificate.admissible_count,
            "admissible_list_sha256": certificate.admissible_list_sha256,
            "candidate_count": certificate.candidate_count,
            "candidate_order_sha256": certificate.candidate_order_sha256,
            "config_sha256": certificate.config_sha256,
            "exact_tie_multiplicity": certificate.exact_tie_multiplicity,
            "geometry_sha256": certificate.geometry_sha256,
            "inadmissible_count": certificate.inadmissible_count,
            "invalid_bitmap_sha256": certificate.invalid_bitmap_sha256,
            "objective_content_sha256": certificate.objective_content_sha256,
            "objective_scope_sha256": certificate.objective_scope_sha256,
            "protocol_sha256": certificate.protocol_sha256,
            "scalar_replay_bit_exact": certificate.scalar_replay_bit_exact,
            "second_best_nonflow_gap": _float(certificate.second_best_nonflow_gap, "second-best nonflow gap"),
            "second_best_objective_gap": _float(certificate.second_best_objective_gap, "second-best objective gap"),
            "selected_admissible_rank": certificate.selected_admissible_rank,
            "selected_evaluation_sha256": certificate.selected_evaluation_sha256,
            "selected_prediction_sha256": certificate.selected_prediction_sha256,
            "selected_total_rank": certificate.selected_total_rank,
            "source_content_sha256": certificate.source_content_sha256,
            "source_scope_sha256": certificate.source_scope_sha256,
        },
        "context_key": result.context_key,
        "gains": None if selected.gains is None else _array_list(selected.gains),
        "history_prediction_sha256": result.prediction_sha256,
        "objective": _float(selected.objective, "selected objective"),
        "parameters": _array_list(selected.parameters),
        "retained_macro_ids": list(selected.retained_macro_ids),
        "scientific_fingerprint": exact.scientific_fingerprint(result),
        "source_grid": {
            "content_sha256": result.source_grid.content_sha256,
            "partition_sha256": result.source_grid.partition_sha256,
            "sample_stream_sha256": result.source_grid.sample_stream_sha256,
            "scope_sha256": result.source_grid.scope_sha256,
        },
        "state_index": selected.state_index,
    }


def _appearance_fit_record(result: nongrid.AppearanceEstimate) -> dict[str, records.JsonValue]:
    return {
        "biases": _array_list(result.biases),
        "gains": _array_list(result.gains),
        "hashes": {name: digest for name, digest in result.hashes},
        "objective": _float(result.objective, "appearance objective"),
        "parameters": _array_list(result.parameters),
        "retained_macro_ids": list(result.retained_macro_ids),
        "scope_sha256": result.scope_sha256,
    }


def _bias_fit_record(result: nongrid.BiasOnlyEstimate) -> dict[str, records.JsonValue]:
    return {
        "biases": _array_list(result.biases),
        "first_biases": _array_list(result.first_biases),
        "hashes": {name: digest for name, digest in result.hashes},
        "objective": _float(result.objective, "bias-only objective"),
        "retained_macro_ids": list(result.retained_macro_ids),
        "scope_sha256": result.scope_sha256,
    }


def _operator_record(result: predictor.FullOperatorResult) -> dict[str, records.JsonValue]:
    history: dict[str, records.JsonValue]
    if isinstance(result.history_fit, exact.GlobalResult):
        history = _global_fit_record(result.history_fit)
    else:
        history = _appearance_fit_record(result.history_fit)
    return {
        "arm": result.arm,
        "biases": _array_list(result.biases),
        "forecast_sha256": result.forecast_sha256,
        "gains": _array_list(result.gains),
        "history_fit": history,
        "history_reconstruction_sha256": result.history_reconstruction_sha256,
        "parameters": _array_list(result.parameters),
    }


def _source_pair_record(result: predictor.SourcePairResult) -> dict[str, records.JsonValue]:
    return {
        "affine": _operator_record(result.affine),
        "appearance": _operator_record(result.appearance),
        "combined": _operator_record(result.combined),
        "current_sha256": result.current_sha256,
        "previous_sha256": result.previous_sha256,
    }


def _reverse_pair_record(
    result: predictor.SourcePairResult,
    current: np.ndarray,
    forecasts: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, records.JsonValue]:
    """Bind the C->P fit separately from its required one-shot application to C."""

    record = _source_pair_record(result)
    record["control_application"] = {
        "applied_array_sha256": predictor.array_sha256(current, role="reverse:actual-current"),
        "forecast_sha256": {
            arm: predictor.array_sha256(forecast, role=f"reverse-control:{arm}")
            for arm, forecast in zip(("affine", "appearance", "combined"), forecasts, strict=True)
        },
        "semantics": "fit_current_to_previous_then_apply_frozen_fit_to_actual_current",
    }
    return record


def _checker_fit_record(
    result: exact.GlobalResult | nongrid.AppearanceEstimate,
) -> dict[str, records.JsonValue]:
    if isinstance(result, exact.GlobalResult):
        return _global_fit_record(result)
    return _appearance_fit_record(result)


def _checker_record(result: predictor.CheckerboardHistoryResult) -> dict[str, records.JsonValue]:
    arms: dict[str, records.JsonValue] = {}
    for arm_name in ("affine", "appearance", "combined"):
        arm = result.arm(cast(predictor.Arm, arm_name))
        arms[arm_name] = {
            "history_reconstruction_sha256": arm.history_reconstruction_sha256,
            "output_parity_fits": [_checker_fit_record(value) for value in arm.output_parity_fits],
        }
    return {
        "arms": arms,
        "bias_only": {
            "history_reconstruction_sha256": result.bias_only.history_reconstruction_sha256,
            "output_parity_fits": [_bias_fit_record(value) for value in result.bias_only.output_parity_fits],
        },
        "current_sha256": result.current_sha256,
        "previous_sha256": result.previous_sha256,
    }


def _is_bounded(operator: predictor.FullOperatorResult) -> bool:
    parameters = np.asarray(operator.parameters)
    translation = bool(np.any(np.isin(parameters[:2], (-8.0, 8.0))))
    gradients = bool(np.any(np.isin(parameters[2:], (-4.0, 4.0))))
    appearance = bool(np.any(np.isin(operator.gains, (-2.0, 4.0))) or np.any(np.isin(operator.biases, (-4.0, 4.0))))
    if operator.arm == "affine":
        return translation or gradients
    if operator.arm == "appearance":
        return appearance
    return translation or gradients or appearance


def _prediction_hashes(
    predictions: np.ndarray,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    return {
        role: {
            "mm009_sha256": records.scientific_array_sha256(
                f"prediction:{role}", predictions[index], protocol_sha256=protocol_sha256
            ),
            "predictor_sha256": predictor.array_sha256(predictions[index], role=f"bundle:{role}"),
        }
        for index, role in enumerate(PREDICTION_ROLES)
    }


def _write_npy_exclusive(path: Path, value: np.ndarray) -> dict[str, records.JsonValue]:
    if path.exists() or path.is_symlink():
        raise WorkerError("prediction array path already exists")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return cast(dict[str, records.JsonValue], records.file_record(path))


def run_source_row(
    source_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, records.JsonValue]:
    """Run all frozen source fits for exactly one detached row."""

    config = _read_config(config_path)
    config_sha256 = cast(str, config["config_sha256"])
    protocol_sha256 = cast(str, config["protocol_sha256"])
    source = preparation.load_source_row_npz(source_path)
    previous = source["previous"]
    current = source["current"]
    shuffled_previous = source["shuffled_previous"]

    ordered = predictor.fit_source_pair(previous, current, config_sha256=config_sha256)
    ordered_predictions = tuple(
        ordered.operator(cast(predictor.Arm, arm)).forecast for arm in ("affine", "appearance", "combined")
    )
    baselines = (ordered.baselines.persistence, ordered.baselines.velocity)
    ordered_record = _source_pair_record(ordered)
    bounded: dict[str, records.JsonValue] = {
        arm: _is_bounded(ordered.operator(cast(predictor.Arm, arm))) for arm in ("affine", "appearance", "combined")
    }
    del ordered
    gc.collect()

    shuffled = predictor.fit_source_pair(shuffled_previous, current, config_sha256=config_sha256)
    shuffled_predictions = tuple(
        shuffled.operator(cast(predictor.Arm, arm)).forecast for arm in ("affine", "appearance", "combined")
    )
    shuffled_record = _source_pair_record(shuffled)
    del shuffled
    gc.collect()

    reverse = predictor.fit_source_pair(current, previous, config_sha256=config_sha256)
    reverse_predictions = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple(
            predictor.apply_operator_once(
                current,
                reverse.operator(cast(predictor.Arm, arm)).parameters,
                reverse.operator(cast(predictor.Arm, arm)).gains,
                reverse.operator(cast(predictor.Arm, arm)).biases,
            )
            for arm in ("affine", "appearance", "combined")
        ),
    )
    reverse_record = _reverse_pair_record(reverse, current, reverse_predictions)
    del reverse
    gc.collect()

    history = predictor.fit_checkerboard_history(previous, current, config_sha256=config_sha256)
    history_predictions = tuple(
        history.arm(cast(predictor.Arm, arm)).history_reconstruction for arm in ("affine", "appearance", "combined")
    )
    history_bias = history.bias_only.history_reconstruction
    history_record = _checker_record(history)
    del history
    gc.collect()

    shuffled_history = predictor.fit_checkerboard_history(shuffled_previous, current, config_sha256=config_sha256)
    shuffled_history_predictions = tuple(
        shuffled_history.arm(cast(predictor.Arm, arm)).history_reconstruction
        for arm in ("affine", "appearance", "combined")
    )
    shuffled_history_record = _checker_record(shuffled_history)
    del shuffled_history
    gc.collect()

    bias = predictor.fit_bias_only_control(current, config_sha256=config_sha256)
    forecast_bias = bias.forecast
    bias_record: dict[str, records.JsonValue] = {
        "current_sha256": bias.current_sha256,
        "forecast_sha256": bias.forecast_sha256,
        "history_fit": _bias_fit_record(bias.history_fit),
        "history_reconstruction_sha256": bias.history_reconstruction_sha256,
    }
    identity = geometry.sample_scalar(previous, 0, geometry.FULL_MASK)
    prediction_array = np.ascontiguousarray(
        np.stack(
            (
                identity,
                *history_predictions,
                *shuffled_history_predictions,
                history_bias,
                *baselines,
                *ordered_predictions,
                *shuffled_predictions,
                *reverse_predictions,
                forecast_bias,
            ),
            axis=0,
        ),
        dtype="<f8",
    )
    if prediction_array.shape != PREDICTION_SHAPE or not np.all(np.isfinite(prediction_array)):
        raise WorkerError("assembled prediction bundle shape or finiteness differs")

    output = Path(output_directory)
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise WorkerError("worker output must be an empty real directory")
    prediction_record = _write_npy_exclusive(output / PREDICTION_FILE, prediction_array)
    commit: dict[str, records.JsonValue] = {
        "bias_only": bias_record,
        "bounded": bounded,
        "config_sha256": config_sha256,
        "fold_index": int(source["fold_index"][0]),
        "normalizer_fingerprint": str(source["normalizer_fingerprint"][0]),
        "ordered": ordered_record,
        "prediction_file": prediction_record,
        "prediction_hashes": _prediction_hashes(prediction_array, protocol_sha256),
        "prediction_roles": list(PREDICTION_ROLES),
        "prediction_shape": list(PREDICTION_SHAPE),
        "protocol_sha256": protocol_sha256,
        "reverse": reverse_record,
        "row_ordinal": int(source["row_ordinal"][0]),
        "schema_version": SCHEMA_VERSION,
        "shuffle_row_ordinal": int(source["shuffle_row_ordinal"][0]),
        "shuffled": shuffled_record,
        "shuffled_history": shuffled_history_record,
        "source_file_sha256": records.file_sha256(source_path),
        "video_id": str(source["video_id"][0]),
        "history": history_record,
    }
    commit["evidence_sha256"] = records.canonical_json_sha256(commit, protocol_sha256=protocol_sha256)
    records.write_immutable_json_exclusive(output / COMMIT_FILE, commit)
    validate_worker_output(output, config, source, source_path=source_path)
    return commit


def load_prediction_array(path: str | Path) -> np.ndarray:
    candidate = Path(path)
    try:
        payload = records.read_regular_bytes(candidate, maximum_bytes=16 * 1024 * 1024)
        stream = io.BytesIO(payload)
        value = np.load(stream, allow_pickle=False)
        if stream.tell() != len(payload):
            raise WorkerError("prediction array carries trailing bytes")
    except (OSError, ValueError, records.RecordValidationError) as error:
        raise WorkerError("prediction array is not a valid NPY file") from error
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f8")
        or value.shape != PREDICTION_SHAPE
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise WorkerError("prediction array schema differs")
    return cast(np.ndarray, records.immutable_array(value))


def load_commit(path: str | Path) -> dict[str, object]:
    candidate = Path(path)
    try:
        payload = records.read_regular_bytes(candidate, maximum_bytes=16 * 1024 * 1024)
        value = json.loads(
            payload.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, records.RecordValidationError) as error:
        raise WorkerError("prediction commit is not strict JSON") from error
    if type(value) is not dict:
        raise WorkerError("prediction commit must be an object")
    try:
        if payload != records.json_file_bytes(cast(records.JsonValue, value)):
            raise WorkerError("prediction commit is not canonical JSON bytes")
    except records.RecordValidationError as error:
        raise WorkerError("prediction commit is outside canonical JSON") from error
    return cast(dict[str, object], value)


def _record_vector(value: object, size: int, label: str) -> np.ndarray:
    if (
        type(value) is not list
        or len(value) != size
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise WorkerError(f"{label} must contain exactly {size} values")
    try:
        array = np.ascontiguousarray(value, dtype="<f8")
    except (TypeError, ValueError) as error:
        raise WorkerError(f"{label} is not a finite vector") from error
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise WorkerError(f"{label} is not a finite vector")
    return cast(np.ndarray, array)


def _strict_dict(value: object, expected: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise WorkerError(f"{label} schema differs")
    return cast(dict[str, object], value)


def _strict_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WorkerError(f"{label} is outside its frozen integer range")
    return cast(int, value)


def _retained_ids(value: object, expected_count: int, label: str) -> tuple[int, ...]:
    if type(value) is not list or len(value) != expected_count:
        raise WorkerError(f"{label} retained-ID count differs")
    retained = tuple(value)
    if (
        any(type(item) is not int or not 0 <= item < geometry.MACRO_COUNT for item in retained)
        or tuple(sorted(set(retained))) != retained
    ):
        raise WorkerError(f"{label} retained IDs differ from the frozen grammar")
    return cast(tuple[int, ...], retained)


def _hash_map(value: object, expected: set[str], label: str) -> dict[str, str]:
    record = _strict_dict(value, expected, f"{label} hash map")
    for role, digest in record.items():
        records.require_sha256(digest, f"{label}/{role} SHA-256")
    return cast(dict[str, str], record)


def _same_vector(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.tobytes(order="C") == right.tobytes(order="C")


def _target_values(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(geometry.GEOMETRY.coords[np.asarray(mask, dtype=np.bool_)], dtype=np.intp)
    return cast(
        np.ndarray,
        np.ascontiguousarray(frame[:, coordinates[:, 0], coordinates[:, 1]], dtype="<f8"),
    )


def _global_source_scope(
    source: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    config_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(exact.SOURCE_SCOPE_TAG)
    digest.update(np.asarray(source, dtype="<f8", order="C").tobytes(order="C"))
    digest.update(np.asarray(fit_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(np.asarray(output_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(bytes.fromhex(geometry.CANDIDATE_ORDER_SHA256))
    digest.update(bytes.fromhex(geometry.ADMISSIBLE_LIST_SHA256))
    digest.update(struct.pack("<H", geometry.BATCH_SIZE))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


def _global_partition_sha256(source_scope_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(exact.SOURCE_PARTITION_TAG)
    digest.update(bytes.fromhex(source_scope_sha256))
    digest.update(struct.pack("<H", geometry.BATCH_COUNT))
    for ordinal, indices in enumerate(geometry.ADMISSIBLE_BATCHES):
        digest.update(struct.pack("<HH", ordinal, len(indices)))
        digest.update(struct.pack(f"<{len(indices)}H", *indices))
    return digest.hexdigest()


def _global_objective_scope(
    source_scope_sha256: str,
    source_content_sha256: str,
    target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: str,
    config_sha256: str,
) -> str:
    fit_target = _target_values(target, fit_mask)
    digest = hashlib.sha256()
    digest.update(exact.OBJECTIVE_SCOPE_TAG)
    digest.update(bytes.fromhex(source_scope_sha256))
    digest.update(bytes.fromhex(source_content_sha256))
    digest.update(np.asarray(fit_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(np.asarray(output_mask, dtype=np.uint8, order="C").tobytes(order="C"))
    digest.update(struct.pack("<H", fit_target.shape[1]))
    digest.update(fit_target.tobytes(order="C"))
    digest.update(struct.pack("<B", {"affine": 0, "combined": 1}[arm]))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


def _selected_evaluation_sha256(
    objective_scope_sha256: str,
    arm: str,
    state_index: int,
    objective: float,
    gains: np.ndarray,
    biases: np.ndarray,
    retained: tuple[int, ...],
) -> str:
    entry = bytearray(struct.pack("<HBd", state_index, 1, objective))
    if arm == "combined":
        entry.extend(gains.tobytes(order="C"))
        entry.extend(biases.tobytes(order="C"))
        entry.extend(struct.pack("<B", len(retained)))
        entry.extend(bytes(retained))
    return hashlib.sha256(
        exact.SELECTED_EVALUATION_TAG + bytes.fromhex(objective_scope_sha256) + bytes(entry)
    ).hexdigest()


def _selected_prediction_sha256(objective_scope_sha256: str, prediction: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(exact.SELECTED_PREDICTION_TAG)
    digest.update(bytes.fromhex(objective_scope_sha256))
    digest.update(struct.pack("<H", prediction.shape[1]))
    digest.update(np.asarray(prediction, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _validated_global_fit_record(
    value: object,
    *,
    arm: str,
    context_key: str,
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    config_sha256: str,
    expected_prediction: np.ndarray | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    expected = {
        "admissible_rank",
        "arm",
        "biases",
        "certificate",
        "context_key",
        "gains",
        "history_prediction_sha256",
        "objective",
        "parameters",
        "retained_macro_ids",
        "scientific_fingerprint",
        "source_grid",
        "state_index",
    }
    record = _strict_dict(value, expected, f"{label} global fit")
    if record["arm"] != arm or record["context_key"] != context_key:
        raise WorkerError(f"{label} global arm/context differs")
    state_index = _strict_int(
        record["state_index"],
        f"{label} state index",
        minimum=0,
        maximum=geometry.STATE_COUNT - 1,
    )
    admissible_rank = _strict_int(
        record["admissible_rank"],
        f"{label} admissible rank",
        minimum=0,
        maximum=geometry.ADMISSIBLE_COUNT - 1,
    )
    if int(geometry.ADMISSIBLE_INDICES[admissible_rank]) != state_index:
        raise WorkerError(f"{label} selected state/rank relation differs")
    parameters = _record_vector(record["parameters"], 6, f"{label} selected parameters")
    if not _same_vector(parameters, np.asarray(geometry.CANONICAL_GRID[state_index], dtype="<f8")):
        raise WorkerError(f"{label} selected parameters differ from the canonical grid")
    expected_retained = 27 if int(np.count_nonzero(fit_mask)) == geometry.SITE_COUNT else 14
    if arm == "affine":
        if record["gains"] is not None or record["biases"] is not None:
            raise WorkerError(f"{label} affine fit carries appearance parameters")
        retained = _retained_ids(record["retained_macro_ids"], 0, label)
        gains = np.ones(geometry.CHANNELS, dtype="<f8")
        biases = np.zeros(geometry.CHANNELS, dtype="<f8")
    else:
        gains = _record_vector(record["gains"], geometry.CHANNELS, f"{label} selected gains")
        biases = _record_vector(record["biases"], geometry.CHANNELS, f"{label} selected biases")
        if np.any(gains < -2.0) or np.any(gains > 4.0) or np.any(biases < -4.0) or np.any(biases > 4.0):
            raise WorkerError(f"{label} selected appearance values exceed frozen bounds")
        retained = _retained_ids(record["retained_macro_ids"], expected_retained, label)
    objective = _float(record["objective"], f"{label} objective")
    if objective < 0.0:
        raise WorkerError(f"{label} objective must be nonnegative")

    source_grid = _strict_dict(
        record["source_grid"],
        {"content_sha256", "partition_sha256", "sample_stream_sha256", "scope_sha256"},
        f"{label} source-grid record",
    )
    for name, digest in source_grid.items():
        records.require_sha256(digest, f"{label} source-grid {name}")
    source_scope = _global_source_scope(source, fit_mask, output_mask, config_sha256)
    if source_grid["scope_sha256"] != source_scope or source_grid["partition_sha256"] != _global_partition_sha256(
        source_scope
    ):
        raise WorkerError(f"{label} source-grid scope/partition differs")
    source_content = cast(str, source_grid["content_sha256"])
    objective_scope = _global_objective_scope(
        source_scope,
        source_content,
        target,
        fit_mask,
        output_mask,
        arm,
        config_sha256,
    )
    certificate_expected = {
        "admissible_count",
        "admissible_list_sha256",
        "candidate_count",
        "candidate_order_sha256",
        "config_sha256",
        "exact_tie_multiplicity",
        "geometry_sha256",
        "inadmissible_count",
        "invalid_bitmap_sha256",
        "objective_content_sha256",
        "objective_scope_sha256",
        "protocol_sha256",
        "scalar_replay_bit_exact",
        "second_best_nonflow_gap",
        "second_best_objective_gap",
        "selected_admissible_rank",
        "selected_evaluation_sha256",
        "selected_prediction_sha256",
        "selected_total_rank",
        "source_content_sha256",
        "source_scope_sha256",
    }
    certificate = _strict_dict(record["certificate"], certificate_expected, f"{label} certificate")
    for name in (
        "admissible_list_sha256",
        "candidate_order_sha256",
        "config_sha256",
        "geometry_sha256",
        "invalid_bitmap_sha256",
        "objective_content_sha256",
        "objective_scope_sha256",
        "protocol_sha256",
        "selected_evaluation_sha256",
        "selected_prediction_sha256",
        "source_content_sha256",
        "source_scope_sha256",
    ):
        records.require_sha256(certificate[name], f"{label} certificate {name}")
    if (
        certificate["protocol_sha256"] != exact.PROTOCOL_SHA256
        or certificate["config_sha256"] != config_sha256
        or certificate["candidate_order_sha256"] != geometry.CANDIDATE_ORDER_SHA256
        or certificate["admissible_list_sha256"] != geometry.ADMISSIBLE_LIST_SHA256
        or certificate["invalid_bitmap_sha256"] != geometry.INVALID_BITMAP_SHA256
        or certificate["geometry_sha256"] != geometry.GEOMETRY_SHA256
        or certificate["source_scope_sha256"] != source_scope
        or certificate["source_content_sha256"] != source_content
        or certificate["objective_scope_sha256"] != objective_scope
    ):
        raise WorkerError(f"{label} certificate static/scope binding differs")
    if (
        _strict_int(certificate["candidate_count"], f"{label} candidate count", minimum=0, maximum=geometry.STATE_COUNT)
        != geometry.STATE_COUNT
        or _strict_int(
            certificate["admissible_count"],
            f"{label} admissible count",
            minimum=0,
            maximum=geometry.STATE_COUNT,
        )
        != geometry.ADMISSIBLE_COUNT
        or _strict_int(
            certificate["inadmissible_count"],
            f"{label} inadmissible count",
            minimum=0,
            maximum=geometry.STATE_COUNT,
        )
        != geometry.STATE_COUNT - geometry.ADMISSIBLE_COUNT
        or _strict_int(
            certificate["selected_total_rank"],
            f"{label} selected total rank",
            minimum=0,
            maximum=geometry.STATE_COUNT - 1,
        )
        != state_index
        or _strict_int(
            certificate["selected_admissible_rank"],
            f"{label} selected admissible rank",
            minimum=0,
            maximum=geometry.ADMISSIBLE_COUNT - 1,
        )
        != admissible_rank
        or not 1
        <= _strict_int(
            certificate["exact_tie_multiplicity"],
            f"{label} exact-tie multiplicity",
            minimum=1,
            maximum=geometry.ADMISSIBLE_COUNT,
        )
        <= geometry.ADMISSIBLE_COUNT
        or certificate["scalar_replay_bit_exact"] is not True
    ):
        raise WorkerError(f"{label} certificate counts/booleans differ")
    if any(
        _float(certificate[name], f"{label} {name}") < 0.0
        for name in ("second_best_objective_gap", "second_best_nonflow_gap")
    ):
        raise WorkerError(f"{label} certificate gaps must be nonnegative")

    sampled = geometry.sample_scalar(source, state_index, output_mask)
    prediction = np.ascontiguousarray(gains[:, None] * sampled + biases[:, None], dtype="<f8")
    if expected_prediction is not None and not np.array_equal(prediction, expected_prediction):
        raise WorkerError(f"{label} selected prediction differs from its emitted parity")
    selected_evaluation = _selected_evaluation_sha256(
        objective_scope,
        arm,
        state_index,
        objective,
        gains,
        biases,
        retained,
    )
    selected_prediction = _selected_prediction_sha256(objective_scope, prediction)
    if (
        certificate["selected_evaluation_sha256"] != selected_evaluation
        or certificate["selected_prediction_sha256"] != selected_prediction
        or record["history_prediction_sha256"] != selected_prediction
    ):
        raise WorkerError(f"{label} selected evaluation/prediction digest differs")
    fingerprint = hashlib.sha256()
    fingerprint.update(b"MM008-v2.2-global-result\0")
    fingerprint.update(context_key.encode("ascii"))
    fingerprint.update(arm.encode("ascii"))
    for digest in (
        source_scope,
        cast(str, source_grid["partition_sha256"]),
        cast(str, source_grid["sample_stream_sha256"]),
        source_content,
        objective_scope,
        cast(str, certificate["objective_content_sha256"]),
        selected_evaluation,
        selected_prediction,
    ):
        fingerprint.update(bytes.fromhex(digest))
    if record["scientific_fingerprint"] != fingerprint.hexdigest():
        raise WorkerError(f"{label} scientific fingerprint differs")
    return parameters, gains, biases, selected_prediction


def _validated_appearance_fit_record(
    value: object,
    *,
    source: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    config_sha256: str,
    expected_prediction: np.ndarray | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    expected = {"biases", "gains", "hashes", "objective", "parameters", "retained_macro_ids", "scope_sha256"}
    record = _strict_dict(value, expected, f"{label} appearance fit")
    parameters = _record_vector(record["parameters"], 6, f"{label} appearance parameters")
    if not np.array_equal(parameters, np.zeros(6, dtype="<f8")):
        raise WorkerError(f"{label} appearance geometry is not identity")
    gains = _record_vector(record["gains"], 3, f"{label} appearance gains")
    biases = _record_vector(record["biases"], 3, f"{label} appearance biases")
    if np.any(gains < -2.0) or np.any(gains > 4.0) or np.any(biases < -4.0) or np.any(biases > 4.0):
        raise WorkerError(f"{label} appearance values exceed frozen bounds")
    expected_retained = 27 if int(np.count_nonzero(fit_mask)) == geometry.SITE_COUNT else 14
    retained = _retained_ids(record["retained_macro_ids"], expected_retained, label)
    objective = _float(record["objective"], f"{label} appearance objective")
    if objective < 0.0:
        raise WorkerError(f"{label} appearance objective must be nonnegative")
    scope = records.require_sha256(record["scope_sha256"], f"{label} appearance scope")
    expected_scope = nongrid.nongrid_scope_sha256(
        source,
        _target_values(target, fit_mask),
        fit_mask,
        output_mask,
        "appearance",
        config_sha256=config_sha256,
    )
    if scope != expected_scope:
        raise WorkerError(f"{label} appearance scope differs")
    hashes = _hash_map(
        record["hashes"],
        {
            "appearance_biases",
            "appearance_fit_prediction",
            "appearance_gains",
            "appearance_parameters",
            "appearance_prediction",
            "appearance_retained_ids",
        },
        label,
    )
    fit_prediction = np.ascontiguousarray(
        gains[:, None] * geometry.sample_scalar(source, 0, fit_mask) + biases[:, None],
        dtype="<f8",
    )
    prediction = np.ascontiguousarray(
        gains[:, None] * geometry.sample_scalar(source, 0, output_mask) + biases[:, None],
        dtype="<f8",
    )
    arrays = {
        "appearance_biases": biases,
        "appearance_fit_prediction": fit_prediction,
        "appearance_gains": gains,
        "appearance_parameters": parameters,
        "appearance_prediction": prediction,
        "appearance_retained_ids": np.asarray(retained, dtype="<i8"),
    }
    if any(hashes[role] != nongrid.array_sha256(scope, role, array) for role, array in arrays.items()):
        raise WorkerError(f"{label} appearance array digest differs")
    if expected_prediction is not None and not np.array_equal(prediction, expected_prediction):
        raise WorkerError(f"{label} appearance prediction differs from its emitted parity")
    return parameters, gains, biases, hashes["appearance_prediction"]


def _validated_bias_fit_record(
    value: object,
    *,
    target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    config_sha256: str,
    expected_prediction: np.ndarray | None,
    label: str,
) -> tuple[np.ndarray, str, np.ndarray]:
    expected = {"biases", "first_biases", "hashes", "objective", "retained_macro_ids", "scope_sha256"}
    record = _strict_dict(value, expected, f"{label} bias-only fit")
    first_biases = _record_vector(record["first_biases"], 3, f"{label} first biases")
    biases = _record_vector(record["biases"], 3, f"{label} biases")
    if np.any(first_biases < -4.0) or np.any(first_biases > 4.0) or np.any(biases < -4.0) or np.any(biases > 4.0):
        raise WorkerError(f"{label} bias-only values exceed frozen bounds")
    expected_retained = 27 if int(np.count_nonzero(fit_mask)) == geometry.SITE_COUNT else 14
    retained = _retained_ids(record["retained_macro_ids"], expected_retained, label)
    objective = _float(record["objective"], f"{label} bias-only objective")
    if objective < 0.0:
        raise WorkerError(f"{label} bias-only objective must be nonnegative")
    scope = records.require_sha256(record["scope_sha256"], f"{label} bias-only scope")
    expected_scope = nongrid.nongrid_scope_sha256(
        None,
        _target_values(target, fit_mask),
        fit_mask,
        output_mask,
        "bias_only",
        config_sha256=config_sha256,
    )
    if scope != expected_scope:
        raise WorkerError(f"{label} bias-only scope differs")
    hashes = _hash_map(
        record["hashes"],
        {"bias_only_biases", "bias_only_first_biases", "bias_only_prediction", "bias_only_retained_ids"},
        label,
    )
    output_count = int(np.count_nonzero(output_mask))
    prediction = np.ascontiguousarray(
        np.broadcast_to(biases[:, None], (geometry.CHANNELS, output_count)),
        dtype="<f8",
    )
    arrays = {
        "bias_only_biases": biases,
        "bias_only_first_biases": first_biases,
        "bias_only_prediction": prediction,
        "bias_only_retained_ids": np.asarray(retained, dtype="<i8"),
    }
    if any(hashes[role] != nongrid.array_sha256(scope, role, array) for role, array in arrays.items()):
        raise WorkerError(f"{label} bias-only array digest differs")
    if expected_prediction is not None and not np.array_equal(prediction, expected_prediction):
        raise WorkerError(f"{label} bias-only prediction differs from emitted values")
    return biases, hashes["bias_only_prediction"], prediction


def _validated_operator_record(
    value: object,
    arm: str,
    label: str,
    *,
    source: np.ndarray,
    target: np.ndarray,
    config_sha256: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    expected = {
        "arm",
        "biases",
        "forecast_sha256",
        "gains",
        "history_fit",
        "history_reconstruction_sha256",
        "parameters",
    }
    if type(value) is not dict or set(value) != expected or value.get("arm") != arm:
        raise WorkerError(f"{label} operator record schema differs")
    record = cast(dict[str, object], value)
    parameters = _record_vector(record["parameters"], 6, f"{label} parameters")
    gains = _record_vector(record["gains"], 3, f"{label} gains")
    biases = _record_vector(record["biases"], 3, f"{label} biases")
    records.require_sha256(record["forecast_sha256"], f"{label} forecast SHA-256")
    records.require_sha256(record["history_reconstruction_sha256"], f"{label} history SHA-256")
    if arm == "appearance":
        fit_parameters, fit_gains, fit_biases, history_hash = _validated_appearance_fit_record(
            record["history_fit"],
            source=source,
            target=target,
            fit_mask=geometry.FULL_MASK,
            output_mask=geometry.FULL_MASK,
            config_sha256=config_sha256,
            expected_prediction=None,
            label=f"{label}/history-fit",
        )
    else:
        fit_parameters, fit_gains, fit_biases, history_hash = _validated_global_fit_record(
            record["history_fit"],
            arm=arm,
            context_key=f"mm009/source/history/full/{arm}",
            source=source,
            target=target,
            fit_mask=geometry.FULL_MASK,
            output_mask=geometry.FULL_MASK,
            config_sha256=config_sha256,
            expected_prediction=None,
            label=f"{label}/history-fit",
        )
    if (
        not _same_vector(parameters, fit_parameters)
        or not _same_vector(gains, fit_gains)
        or not _same_vector(biases, fit_biases)
        or record["history_reconstruction_sha256"] != history_hash
    ):
        raise WorkerError(f"{label} top-level parameters/history hash differ from selected fit")
    native_forecast = predictor.apply_operator_once(target, parameters, gains, biases)
    if record["forecast_sha256"] != predictor.array_sha256(native_forecast, role=f"{arm}_forecast"):
        raise WorkerError(f"{label} native forecast hash differs")
    return record, parameters, gains, biases


def _validate_pair_record(
    value: object,
    *,
    previous: np.ndarray,
    current: np.ndarray,
    application: np.ndarray,
    predictions: np.ndarray,
    role_prefix: str,
    config_sha256: str,
    reverse: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    expected = {"affine", "appearance", "combined", "current_sha256", "previous_sha256"}
    if reverse:
        expected.add("control_application")
    if type(value) is not dict or set(value) != expected:
        raise WorkerError(f"{role_prefix} source-pair record schema differs")
    record = cast(dict[str, object], value)
    if record["previous_sha256"] != predictor.array_sha256(previous, role="previous_normalized") or record[
        "current_sha256"
    ] != predictor.array_sha256(current, role="current_normalized"):
        raise WorkerError(f"{role_prefix} source-pair input hashes differ")
    role_lookup = {role: index for index, role in enumerate(PREDICTION_ROLES)}
    control_hashes: dict[str, object] | None = None
    if reverse:
        control = record["control_application"]
        if type(control) is not dict or set(control) != {
            "applied_array_sha256",
            "forecast_sha256",
            "semantics",
        }:
            raise WorkerError("reverse control application record differs")
        if (
            control["applied_array_sha256"] != predictor.array_sha256(application, role="reverse:actual-current")
            or control["semantics"] != "fit_current_to_previous_then_apply_frozen_fit_to_actual_current"
            or type(control["forecast_sha256"]) is not dict
        ):
            raise WorkerError("reverse control application binding differs")
        control_hashes = cast(dict[str, object], control["forecast_sha256"])
        if set(control_hashes) != {"affine", "appearance", "combined"}:
            raise WorkerError("reverse control forecast-hash membership differs")
    parameters_by_arm: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for arm in ("affine", "appearance", "combined"):
        operator, parameters, gains, biases = _validated_operator_record(
            record[arm],
            arm,
            f"{role_prefix}/{arm}",
            source=previous,
            target=current,
            config_sha256=config_sha256,
        )
        parameters_by_arm[arm] = (parameters, gains, biases)
        expected_prediction = predictor.apply_operator_once(
            application,
            parameters,
            gains,
            biases,
        )
        actual = predictions[role_lookup[f"{role_prefix}_{arm}"]]
        if not np.array_equal(actual, expected_prediction):
            raise WorkerError(f"{role_prefix}/{arm} application replay differs")
        if reverse:
            assert control_hashes is not None
            if control_hashes.get(arm) != predictor.array_sha256(actual, role=f"reverse-control:{arm}"):
                raise WorkerError(f"reverse/{arm} control forecast hash differs")
        elif operator["forecast_sha256"] != predictor.array_sha256(actual, role=f"{arm}_forecast"):
            raise WorkerError(f"{role_prefix}/{arm} forecast hash differs")
    return parameters_by_arm


def _validate_checker_record(
    value: object,
    *,
    previous: np.ndarray,
    current: np.ndarray,
    predictions: np.ndarray,
    role_prefix: str,
    config_sha256: str,
) -> None:
    expected = {"arms", "bias_only", "current_sha256", "previous_sha256"}
    if type(value) is not dict or set(value) != expected:
        raise WorkerError(f"{role_prefix} checkerboard record schema differs")
    record = cast(dict[str, object], value)
    arms = record["arms"]
    if (
        record["previous_sha256"] != predictor.array_sha256(previous, role="previous_normalized")
        or record["current_sha256"] != predictor.array_sha256(current, role="current_normalized")
        or type(arms) is not dict
        or set(arms) != {"affine", "appearance", "combined"}
    ):
        raise WorkerError(f"{role_prefix} checkerboard binding differs")
    role_lookup = {role: index for index, role in enumerate(PREDICTION_ROLES)}
    for arm in ("affine", "appearance", "combined"):
        item = arms[arm]
        if (
            type(item) is not dict
            or set(item) != {"history_reconstruction_sha256", "output_parity_fits"}
            or type(item["output_parity_fits"]) is not list
            or len(item["output_parity_fits"]) != 2
        ):
            raise WorkerError(f"{role_prefix}/{arm} checkerboard evidence differs")
        actual = predictions[role_lookup[f"{role_prefix}_{arm}"]]
        parity_fits = cast(list[object], item["output_parity_fits"])
        for output_parity, fit_record in enumerate(parity_fits):
            output_mask = geometry.PARITY_MASKS[output_parity]
            fit_mask = geometry.PARITY_MASKS[1 - output_parity]
            expected_partial = np.ascontiguousarray(actual[:, output_mask], dtype="<f8")
            if arm == "appearance":
                _validated_appearance_fit_record(
                    fit_record,
                    source=previous,
                    target=current,
                    fit_mask=fit_mask,
                    output_mask=output_mask,
                    config_sha256=config_sha256,
                    expected_prediction=expected_partial,
                    label=f"{role_prefix}/{arm}/parity-{output_parity}",
                )
            else:
                _validated_global_fit_record(
                    fit_record,
                    arm=arm,
                    context_key=f"mm009/source/history/checkerboard/output-{output_parity}/{arm}",
                    source=previous,
                    target=current,
                    fit_mask=fit_mask,
                    output_mask=output_mask,
                    config_sha256=config_sha256,
                    expected_prediction=expected_partial,
                    label=f"{role_prefix}/{arm}/parity-{output_parity}",
                )
        if item["history_reconstruction_sha256"] != predictor.array_sha256(
            actual, role=f"{arm}_checkerboard_history_reconstruction"
        ):
            raise WorkerError(f"{role_prefix}/{arm} checkerboard hash differs")
    bias_item = _strict_dict(
        record["bias_only"],
        {"history_reconstruction_sha256", "output_parity_fits"},
        f"{role_prefix} checkerboard bias-only evidence",
    )
    bias_fits = bias_item["output_parity_fits"]
    if type(bias_fits) is not list or len(bias_fits) != 2:
        raise WorkerError(f"{role_prefix} checkerboard bias parity membership differs")
    bias_reconstruction = np.empty((geometry.CHANNELS, geometry.SITE_COUNT), dtype="<f8")
    emitted_bias = predictions[role_lookup["history_bias"]] if role_prefix == "history_xfit" else None
    for output_parity, fit_record in enumerate(bias_fits):
        output_mask = geometry.PARITY_MASKS[output_parity]
        fit_mask = geometry.PARITY_MASKS[1 - output_parity]
        expected_bias_partial = (
            None if emitted_bias is None else np.ascontiguousarray(emitted_bias[:, output_mask], dtype="<f8")
        )
        _, _, partial = _validated_bias_fit_record(
            fit_record,
            target=current,
            fit_mask=fit_mask,
            output_mask=output_mask,
            config_sha256=config_sha256,
            expected_prediction=expected_bias_partial,
            label=f"{role_prefix}/bias-only/parity-{output_parity}",
        )
        bias_reconstruction[:, output_mask] = partial
    if emitted_bias is not None and not np.array_equal(bias_reconstruction, emitted_bias):
        raise WorkerError(f"{role_prefix} checkerboard bias reconstruction differs")
    if bias_item["history_reconstruction_sha256"] != predictor.array_sha256(
        bias_reconstruction,
        role="bias_only_checkerboard_history_reconstruction",
    ):
        raise WorkerError(f"{role_prefix} checkerboard bias hash differs")


def validate_worker_output(
    output_directory: str | Path,
    config: Mapping[str, object],
    source: Mapping[str, np.ndarray],
    *,
    source_path: str | Path,
) -> None:
    output = Path(output_directory)
    if set(path.name for path in output.iterdir()) != {PREDICTION_FILE, COMMIT_FILE}:
        raise WorkerError("worker output membership differs")
    predictions = load_prediction_array(output / PREDICTION_FILE)
    commit = load_commit(output / COMMIT_FILE)
    expected_top = {
        "bias_only",
        "bounded",
        "config_sha256",
        "evidence_sha256",
        "fold_index",
        "history",
        "normalizer_fingerprint",
        "ordered",
        "prediction_file",
        "prediction_hashes",
        "prediction_roles",
        "prediction_shape",
        "protocol_sha256",
        "reverse",
        "row_ordinal",
        "schema_version",
        "shuffle_row_ordinal",
        "shuffled",
        "shuffled_history",
        "source_file_sha256",
        "video_id",
    }
    if set(commit) != expected_top or commit.get("schema_version") != SCHEMA_VERSION:
        raise WorkerError("prediction commit schema differs")
    evidence_sha256 = records.require_sha256(commit["evidence_sha256"], "worker evidence SHA-256")
    evidence_body = cast(dict[str, records.JsonValue], dict(commit))
    del evidence_body["evidence_sha256"]
    protocol_sha256 = cast(str, config["protocol_sha256"])
    if evidence_sha256 != records.canonical_json_sha256(evidence_body, protocol_sha256=protocol_sha256):
        raise WorkerError("worker evidence digest differs")
    if (
        commit.get("config_sha256") != config.get("config_sha256")
        or commit.get("protocol_sha256") != config.get("protocol_sha256")
        or commit.get("prediction_roles") != list(PREDICTION_ROLES)
        or commit.get("prediction_shape") != list(PREDICTION_SHAPE)
        or commit.get("row_ordinal") != int(source["row_ordinal"][0])
        or commit.get("video_id") != str(source["video_id"][0])
        or commit.get("fold_index") != int(source["fold_index"][0])
        or commit.get("normalizer_fingerprint") != str(source["normalizer_fingerprint"][0])
        or commit.get("shuffle_row_ordinal") != int(source["shuffle_row_ordinal"][0])
        or commit.get("source_file_sha256") != records.file_sha256(source_path)
    ):
        raise WorkerError("prediction commit binding differs")
    file_record = commit.get("prediction_file")
    if (
        type(file_record) is not dict
        or file_record != records.file_record(output / PREDICTION_FILE)
        or file_record.get("mode") != 0o444
        or records.file_record(output / COMMIT_FILE)["mode"] != 0o444
    ):
        raise WorkerError("prediction file hash differs")
    hashes = commit.get("prediction_hashes")
    if type(hashes) is not dict or set(hashes) != set(PREDICTION_ROLES):
        raise WorkerError("prediction array hash membership differs")
    for index, role in enumerate(PREDICTION_ROLES):
        item = hashes.get(role)
        if type(item) is not dict or set(item) != {"mm009_sha256", "predictor_sha256"}:
            raise WorkerError("prediction array hash record differs")
        if item.get("mm009_sha256") != records.scientific_array_sha256(
            f"prediction:{role}", predictions[index], protocol_sha256=protocol_sha256
        ) or item.get("predictor_sha256") != predictor.array_sha256(predictions[index], role=f"bundle:{role}"):
            raise WorkerError(f"prediction hash differs for role {role}")
    roles = {role: index for index, role in enumerate(PREDICTION_ROLES)}
    identity = geometry.sample_scalar(source["previous"], 0, geometry.FULL_MASK)
    baselines = predictor.source_baselines(source["previous"], source["current"])
    if (
        not np.array_equal(predictions[roles["history_identity"]], identity)
        or not np.array_equal(predictions[roles["persistence"]], baselines.persistence)
        or not np.array_equal(predictions[roles["velocity"]], baselines.velocity)
    ):
        raise WorkerError("source baseline replay differs")
    config_sha256 = cast(str, config["config_sha256"])
    ordered_operators = _validate_pair_record(
        commit["ordered"],
        previous=source["previous"],
        current=source["current"],
        application=source["current"],
        predictions=predictions,
        role_prefix="forecast",
        config_sha256=config_sha256,
    )
    _validate_pair_record(
        commit["shuffled"],
        previous=source["shuffled_previous"],
        current=source["current"],
        application=source["current"],
        predictions=predictions,
        role_prefix="forecast_shuffle",
        config_sha256=config_sha256,
    )
    _validate_pair_record(
        commit["reverse"],
        previous=source["current"],
        current=source["previous"],
        application=source["current"],
        predictions=predictions,
        role_prefix="forecast_reverse",
        config_sha256=config_sha256,
        reverse=True,
    )
    _validate_checker_record(
        commit["history"],
        previous=source["previous"],
        current=source["current"],
        predictions=predictions,
        role_prefix="history_xfit",
        config_sha256=config_sha256,
    )
    _validate_checker_record(
        commit["shuffled_history"],
        previous=source["shuffled_previous"],
        current=source["current"],
        predictions=predictions,
        role_prefix="history_shuffle_xfit",
        config_sha256=config_sha256,
    )
    bias = commit["bias_only"]
    if (
        type(bias) is not dict
        or set(bias) != {"current_sha256", "forecast_sha256", "history_fit", "history_reconstruction_sha256"}
        or bias["current_sha256"] != predictor.array_sha256(source["current"], role="current_normalized")
        or bias["forecast_sha256"]
        != predictor.array_sha256(predictions[roles["forecast_bias"]], role="bias_only_forecast")
    ):
        raise WorkerError("bias-only evidence differs")
    _, bias_history_hash, _ = _validated_bias_fit_record(
        bias["history_fit"],
        target=source["current"],
        fit_mask=geometry.FULL_MASK,
        output_mask=geometry.FULL_MASK,
        config_sha256=config_sha256,
        expected_prediction=predictions[roles["forecast_bias"]],
        label="forecast/bias-only/history-fit",
    )
    if bias["history_reconstruction_sha256"] != bias_history_hash:
        raise WorkerError("bias-only history hash differs from selected fit")
    bounded = commit["bounded"]
    if (
        type(bounded) is not dict
        or set(bounded) != {"affine", "appearance", "combined"}
        or any(type(value) is not bool for value in bounded.values())
    ):
        raise WorkerError("bounded evidence differs")
    expected_bounded: dict[str, bool] = {}
    for arm, (parameters, gains, biases) in ordered_operators.items():
        translation = bool(np.any(np.isin(parameters[:2], (-8.0, 8.0))))
        gradients = bool(np.any(np.isin(parameters[2:], (-4.0, 4.0))))
        appearance = bool(np.any(np.isin(gains, (-2.0, 4.0))) or np.any(np.isin(biases, (-4.0, 4.0))))
        expected_bounded[arm] = (
            appearance
            if arm == "appearance"
            else (translation or gradients if arm == "affine" else translation or gradients or appearance)
        )
    if bounded != expected_bounded:
        raise WorkerError("bounded evidence differs from selected operator parameters")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("config")
    parser.add_argument("output")
    arguments = parser.parse_args(argv)
    run_source_row(arguments.source, arguments.config, arguments.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMMIT_FILE",
    "PREDICTION_FILE",
    "PREDICTION_ROLES",
    "PREDICTION_SHAPE",
    "SCHEMA_VERSION",
    "WorkerError",
    "load_commit",
    "load_prediction_array",
    "run_source_row",
    "validate_worker_output",
]
