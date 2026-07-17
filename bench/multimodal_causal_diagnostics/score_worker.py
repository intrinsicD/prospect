"""One-row, fitting-free target scorer for frozen MM-009 predictions.

This module intentionally imports only custody/preparation/scoring primitives.  It
does not import the source worker, predictor, or any v2.2 fitting implementation.
The caller must first authenticate the complete prediction chain; this process then
revalidates the exact row-local custody links it consumes and binds its score to the
supervisor commit hash.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np

from bench.multimodal_causal_diagnostics import preparation, records, scoring

SCHEMA_VERSION: Final = "mm009-score-worker-v1"
SCORE_SCHEMA_VERSION: Final = "mm009-row-score-v1"
SUPERVISOR_SCHEMA_VERSION: Final = "mm009-supervisor-prediction-commit-v1"
WORKER_SCHEMA_VERSION: Final = "mm009-source-worker-v1"
WORKER_CONFIG_SCHEMA_VERSION: Final = "mm009-worker-config-v1"

COMMIT_FILE: Final = "commit.json"
PREDICTION_FILE: Final = "predictions.npy"
WORKER_EVIDENCE_FILE: Final = "worker-evidence.json"
SCORE_FILE: Final = "score.json"

# Deliberately duplicated here so importing the scorer cannot import the fitting
# worker merely to learn a serialization constant.
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
PREDICTION_SHAPE: Final = (len(PREDICTION_ROLES), *scoring.CENTRAL_SHAPE)
METRICS: Final = ("i", "a", "q", "p", "c", "h", "r", "z", "d", "pd", "u", "b", "bd")

_SUPERVISOR_KEYS: Final = {
    "config_sha256",
    "fold_index",
    "predecessor_sha256",
    "protocol_sha256",
    "row_ordinal",
    "schema_version",
    "source_file",
    "video_id",
    "video_row",
    "worker_evidence_file",
    "worker_prediction_file",
}
_WORKER_EVIDENCE_KEYS: Final = {
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


class ScoreWorkerError(ValueError):
    """Raised when score custody or one-row scoring fails closed."""


def _reject_constant(value: str) -> None:
    raise ScoreWorkerError(f"nonfinite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ScoreWorkerError(f"duplicate JSON key is forbidden: {key}")
        output[key] = value
    return output


def _load_canonical_json(path: str | Path, label: str) -> dict[str, object]:
    candidate = Path(path)
    try:
        metadata = records.file_record(candidate)
    except (OSError, records.RecordValidationError) as error:
        raise ScoreWorkerError(f"{label} must be a unique regular file") from error
    if metadata["mode"] != 0o444:
        raise ScoreWorkerError(f"{label} must be immutable mode 0444")
    try:
        payload = records.read_regular_bytes(candidate, maximum_bytes=100_000_000)
        value = json.loads(
            payload.decode("ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScoreWorkerError(f"{label} is not strict ASCII JSON") from error
    if type(value) is not dict:
        raise ScoreWorkerError(f"{label} must be a JSON object")
    checked = cast(dict[str, object], value)
    try:
        canonical = records.json_file_bytes(cast(records.JsonValue, checked))
    except records.RecordValidationError as error:
        raise ScoreWorkerError(f"{label} is outside canonical JSON") from error
    if payload != canonical:
        raise ScoreWorkerError(f"{label} is not canonical JSON bytes")
    return checked


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ScoreWorkerError(f"{label} must be an integer >= {minimum}")
    return value


def _require_file_record(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"bytes", "mode", "sha256"}:
        raise ScoreWorkerError(f"{label} file record schema differs")
    checked = cast(dict[str, object], value)
    _require_int(checked["bytes"], f"{label} bytes")
    if checked["mode"] != 0o444:
        raise ScoreWorkerError(f"{label} file record must bind mode 0444")
    try:
        records.require_sha256(checked["sha256"], f"{label} SHA-256")
    except records.RecordValidationError as error:
        raise ScoreWorkerError(f"{label} file record SHA-256 differs") from error
    return checked


def _require_actual_file_record(path: str | Path, claimed: object, label: str) -> dict[str, object]:
    expected = _require_file_record(claimed, label)
    try:
        actual = records.file_record(path)
    except (OSError, records.RecordValidationError) as error:
        raise ScoreWorkerError(f"{label} must be a unique regular file") from error
    if actual != expected:
        raise ScoreWorkerError(f"{label} file record differs from custody binding")
    return expected


def _load_config(path: str | Path) -> dict[str, object]:
    config = _load_canonical_json(path, "worker config")
    expected = {"config_sha256", "prediction_roles", "protocol_sha256", "schema_version"}
    if set(config) != expected or config.get("schema_version") != WORKER_CONFIG_SCHEMA_VERSION:
        raise ScoreWorkerError("worker config schema differs")
    try:
        records.require_sha256(config.get("config_sha256"), "scientific config SHA-256")
        protocol_sha256 = records.require_sha256(config.get("protocol_sha256"), "protocol SHA-256")
    except records.RecordValidationError as error:
        raise ScoreWorkerError("worker config hash binding differs") from error
    if protocol_sha256 not in {
        records.PROTOCOL_SHA256,
        preparation.PROTOCOL_SHA256,
        scoring.PROTOCOL_SHA256,
    } or len(
        {
            protocol_sha256,
            records.PROTOCOL_SHA256,
            preparation.PROTOCOL_SHA256,
            scoring.PROTOCOL_SHA256,
        }
    ) != 1:
        raise ScoreWorkerError("worker config protocol binding differs")
    roles = config.get("prediction_roles")
    if type(roles) is not list or tuple(roles) != PREDICTION_ROLES:
        raise ScoreWorkerError("worker config prediction role order differs")
    return config


def _load_supervisor_commit(path: str | Path) -> dict[str, object]:
    commit = _load_canonical_json(path, "supervisor commit")
    if set(commit) != _SUPERVISOR_KEYS or commit.get("schema_version") != SUPERVISOR_SCHEMA_VERSION:
        raise ScoreWorkerError("supervisor commit schema differs")
    try:
        records.require_sha256(commit.get("config_sha256"), "scientific config SHA-256")
        records.require_sha256(commit.get("protocol_sha256"), "protocol SHA-256")
        records.require_sha256(commit.get("predecessor_sha256"), "predecessor SHA-256")
    except records.RecordValidationError as error:
        raise ScoreWorkerError("supervisor commit hash binding differs") from error
    _require_int(commit.get("row_ordinal"), "row ordinal")
    _require_int(commit.get("fold_index"), "fold index")
    _require_int(commit.get("video_row"), "video row")
    if type(commit.get("video_id")) is not str:
        raise ScoreWorkerError("supervisor commit video ID differs")
    for name in ("source_file", "worker_prediction_file", "worker_evidence_file"):
        _require_file_record(commit.get(name), name)
    return commit


def _load_predictions(path: str | Path) -> np.ndarray:
    candidate = Path(path)
    try:
        payload = records.read_regular_bytes(candidate, maximum_bytes=100_000_000)
        stream = io.BytesIO(payload)
        value = np.load(stream, allow_pickle=False)
        if stream.tell() != len(payload):
            raise ScoreWorkerError("prediction file carries trailing bytes")
    except (OSError, ValueError, records.RecordValidationError) as error:
        raise ScoreWorkerError("prediction file is not a valid NPY array") from error
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype("<f8")
        or value.shape != PREDICTION_SHAPE
        or not value.flags.c_contiguous
        or not bool(np.all(np.isfinite(value)))
    ):
        raise ScoreWorkerError("prediction array schema differs")
    return cast(np.ndarray, records.immutable_array(value))


def _validate_worker_evidence(
    evidence: Mapping[str, object],
    *,
    config: Mapping[str, object],
    commit: Mapping[str, object],
    source_sha256: str,
    prediction_record: Mapping[str, object],
    predictions: np.ndarray,
) -> None:
    if set(evidence) != _WORKER_EVIDENCE_KEYS or evidence.get("schema_version") != WORKER_SCHEMA_VERSION:
        raise ScoreWorkerError("worker evidence schema differs")
    protocol_sha256 = cast(str, config["protocol_sha256"])
    try:
        claimed_evidence_sha256 = records.require_sha256(
            evidence.get("evidence_sha256"), "worker evidence SHA-256"
        )
        evidence_body = cast(
            dict[str, records.JsonValue],
            {key: value for key, value in evidence.items() if key != "evidence_sha256"},
        )
        expected_evidence_sha256 = records.canonical_json_sha256(
            evidence_body,
            protocol_sha256=protocol_sha256,
        )
    except records.RecordValidationError as error:
        raise ScoreWorkerError("worker evidence digest grammar differs") from error
    if claimed_evidence_sha256 != expected_evidence_sha256:
        raise ScoreWorkerError("worker evidence digest differs")
    if (
        evidence.get("config_sha256") != config.get("config_sha256")
        or evidence.get("protocol_sha256") != config.get("protocol_sha256")
        or evidence.get("row_ordinal") != commit.get("row_ordinal")
        or evidence.get("video_id") != commit.get("video_id")
        or evidence.get("fold_index") != commit.get("fold_index")
        or evidence.get("source_file_sha256") != source_sha256
        or evidence.get("prediction_file") != prediction_record
        or evidence.get("prediction_roles") != list(PREDICTION_ROLES)
        or evidence.get("prediction_shape") != list(PREDICTION_SHAPE)
    ):
        raise ScoreWorkerError("worker evidence row/hash/role binding differs")
    hashes = evidence.get("prediction_hashes")
    if type(hashes) is not dict or set(hashes) != set(PREDICTION_ROLES):
        raise ScoreWorkerError("worker evidence prediction hash membership differs")
    for index, role in enumerate(PREDICTION_ROLES):
        item = hashes.get(role)
        if type(item) is not dict or set(item) != {"mm009_sha256", "predictor_sha256"}:
            raise ScoreWorkerError(f"worker evidence prediction hash record differs for {role}")
        try:
            records.require_sha256(item.get("predictor_sha256"), f"{role} predictor SHA-256")
        except records.RecordValidationError as error:
            raise ScoreWorkerError(f"worker evidence predictor hash differs for {role}") from error
        expected = records.scientific_array_sha256(
            f"prediction:{role}", predictions[index], protocol_sha256=protocol_sha256
        )
        if item.get("mm009_sha256") != expected:
            raise ScoreWorkerError(f"worker evidence MM-009 prediction hash differs for {role}")


def _central_frame(value: np.ndarray, label: str) -> np.ndarray:
    if value.dtype != np.dtype("<f8") or value.shape != (3, 64, 64) or not value.flags.c_contiguous:
        raise ScoreWorkerError(f"{label} full-frame schema differs")
    central = np.ascontiguousarray(value[:, 8:56, 8:56].reshape(scoring.CENTRAL_SHAPE), dtype="<f8")
    if not np.all(np.isfinite(central)):
        raise ScoreWorkerError(f"{label} central target contains a nonfinite value")
    return cast(np.ndarray, central)


def _error_json(value: scoring.ErrorPrimitive) -> dict[str, records.JsonValue]:
    return {"count": value.count, "sse": value.sse}


def _row_scores_json(value: scoring.RowScores) -> dict[str, records.JsonValue]:
    return {name: _error_json(cast(scoring.ErrorPrimitive, getattr(value, name))) for name in METRICS}


def _score_families(
    predictions: np.ndarray,
    source: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    *,
    video_id: str,
    fold_index: int,
    video_row: int,
) -> dict[str, records.JsonValue]:
    roles = {role: predictions[index] for index, role in enumerate(PREDICTION_ROLES)}
    current = _central_frame(source["current"], "current")
    future = _central_frame(target["future"], "future")
    deranged = _central_frame(target["deranged_future"], "deranged future")
    output: dict[str, records.JsonValue] = {}
    for family in scoring.FAMILIES:
        inputs = scoring.RowScoreInputs(
            video_id=video_id,
            fold=fold_index,
            row_index=video_row,
            family=family,
            current_target=current,
            future_target=future,
            deranged_future_target=deranged,
            history_identity=roles["history_identity"],
            history_xfit=roles[f"history_xfit_{family}"],
            history_shuffle_xfit=roles[f"history_shuffle_xfit_{family}"],
            persistence=roles["persistence"],
            forecast=roles[f"forecast_{family}"],
            forecast_shuffle=roles[f"forecast_shuffle_{family}"],
            forecast_reverse=roles[f"forecast_reverse_{family}"],
            velocity=roles["velocity"],
            history_bias=roles["history_bias"],
            forecast_bias=roles["forecast_bias"],
        )
        output[family] = _row_scores_json(scoring.score_row(inputs))
    return output


def run_score_row(
    source_path: str | Path,
    target_path: str | Path,
    prediction_path: str | Path,
    supervisor_commit_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, records.JsonValue]:
    """Validate one frozen row, score all families, and write one immutable record."""

    prediction_candidate = Path(prediction_path)
    commit_candidate = Path(supervisor_commit_path)
    if (
        prediction_candidate.name != PREDICTION_FILE
        or commit_candidate.name != COMMIT_FILE
        or prediction_candidate.parent != commit_candidate.parent
    ):
        raise ScoreWorkerError("prediction and supervisor commit are not one canonical row directory")
    output = Path(output_directory)
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
        raise ScoreWorkerError("score output must be an empty real directory")

    config = _load_config(config_path)
    commit = _load_supervisor_commit(commit_candidate)
    if (
        commit.get("config_sha256") != config.get("config_sha256")
        or commit.get("protocol_sha256") != config.get("protocol_sha256")
    ):
        raise ScoreWorkerError("supervisor commit and worker config binding differs")

    source_record = _require_actual_file_record(source_path, commit["source_file"], "source file")
    prediction_record = _require_actual_file_record(
        prediction_candidate, commit["worker_prediction_file"], "worker prediction file"
    )
    evidence_path = commit_candidate.parent / WORKER_EVIDENCE_FILE
    _require_actual_file_record(evidence_path, commit["worker_evidence_file"], "worker evidence file")

    predictions = _load_predictions(prediction_candidate)
    evidence = _load_canonical_json(evidence_path, "worker evidence")
    source_sha256 = cast(str, source_record["sha256"])
    _validate_worker_evidence(
        evidence,
        config=config,
        commit=commit,
        source_sha256=source_sha256,
        prediction_record=prediction_record,
        predictions=predictions,
    )

    try:
        source = preparation.load_source_row_npz(source_path)
        target = preparation.load_target_row_npz(target_path)
        preparation.validate_detached_pair(source, target)
    except (records.RecordValidationError, preparation.PreparationValidationError) as error:
        raise ScoreWorkerError("detached source/target custody differs") from error
    if records.read_regular_bytes(source_path, maximum_bytes=100_000_000) != preparation.source_row_npz_bytes(source):
        raise ScoreWorkerError("source row is not canonical NPZ bytes")
    if records.read_regular_bytes(target_path, maximum_bytes=100_000_000) != preparation.target_row_npz_bytes(target):
        raise ScoreWorkerError("target row is not canonical NPZ bytes")
    target_record = records.file_record(target_path)
    if target_record["mode"] != 0o444:
        raise ScoreWorkerError("target row must be immutable mode 0444")

    ordinal = cast(int, commit["row_ordinal"])
    canonical = preparation.canonical_row_index()
    if ordinal not in range(len(canonical)):
        raise ScoreWorkerError("supervisor row ordinal is outside the frozen panel")
    row = canonical[ordinal]
    identity = (row.video_id, row.fold_index, row.video_row)
    if identity != (commit["video_id"], commit["fold_index"], commit["video_row"]):
        raise ScoreWorkerError("supervisor commit canonical row identity differs")
    if ordinal != int(source["row_ordinal"][0]) or ordinal != int(target["row_ordinal"][0]):
        raise ScoreWorkerError("supervisor commit row ordinal differs from detached inputs")

    families = _score_families(
        predictions,
        source,
        target,
        video_id=row.video_id,
        fold_index=row.fold_index,
        video_row=row.video_row,
    )
    result: dict[str, records.JsonValue] = {
        "config_sha256": cast(str, config["config_sha256"]),
        "families": families,
        "fold_index": row.fold_index,
        "prediction_file_sha256": cast(str, prediction_record["sha256"]),
        "protocol_sha256": cast(str, config["protocol_sha256"]),
        "row_ordinal": ordinal,
        "schema_version": SCORE_SCHEMA_VERSION,
        "source_file_sha256": source_sha256,
        "supervisor_commit_sha256": records.file_sha256(commit_candidate),
        "target_file_sha256": cast(str, target_record["sha256"]),
        "video_id": row.video_id,
        "video_row": row.video_row,
    }
    records.write_immutable_json_exclusive(output / SCORE_FILE, result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("predictions")
    parser.add_argument("supervisor_commit")
    parser.add_argument("config")
    parser.add_argument("output")
    arguments = parser.parse_args(argv)
    run_score_row(
        arguments.source,
        arguments.target,
        arguments.predictions,
        arguments.supervisor_commit,
        arguments.config,
        arguments.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMMIT_FILE",
    "PREDICTION_FILE",
    "PREDICTION_ROLES",
    "PREDICTION_SHAPE",
    "SCHEMA_VERSION",
    "SCORE_FILE",
    "SCORE_SCHEMA_VERSION",
    "ScoreWorkerError",
    "WORKER_EVIDENCE_FILE",
    "main",
    "run_score_row",
]
