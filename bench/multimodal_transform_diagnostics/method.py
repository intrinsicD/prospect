"""Pure NumPy scientific engine for MM-003.

This module consumes sealed TAESD arrays and already-computed MM-002 evidence.
It performs no media or neural-frontend inference and owns no filesystem writes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final, cast

import numpy as np

from bench.multimodal_diagnostics import method as mm002
from bench.multimodal_preflight import core, dataset
from prospect.types import LatentState, Transition
from prospect.world_model import FlatWorldModel

SCHEMA_VERSION: Final = "mm003-method-v1"
PARITY_RTOL: Final = 1e-12
PARITY_ATOL: Final = 1e-12
ALIGNMENT_RTOL: Final = 1e-6
ALIGNMENT_ATOL: Final = 1e-6
PROBE_INVARIANCE_RTOL: Final = 1e-8
PROBE_INVARIANCE_ATOL: Final = 1e-10
PROBE_PENALTY: Final = 1e-3
SCALE_FLOOR: Final = 1e-6
QR_PROJECTOR_TOLERANCE: Final = 1e-10
PCA_RANK_RATIO_MIN: Final = 1e-8
PCA_BOUNDARY_GAP_MIN: Final = 1e-6
INTEGRITY_MIN_STD: Final = mm002.INTEGRITY_MIN_STD
INTEGRITY_MIN_EFFECTIVE_RANK: Final = mm002.INTEGRITY_MIN_EFFECTIVE_RANK
CHECKPOINTS: Final = (300, 600, 1_500)
PREDICTOR_IDS: Final = ("absolute_target", "residual_delta")


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    """One frozen representation intervention."""

    representation_id: str
    output_dim: int
    fitted: bool
    world_tested: bool


REPRESENTATIONS: Final = (
    RepresentationSpec("r32_native", 32, False, True),
    RepresentationSpec("r32_postz", 32, True, True),
    RepresentationSpec("raw256_native", 256, False, False),
    RepresentationSpec("raw256_postz", 256, True, False),
    RepresentationSpec("r32_qr_postz", 32, True, True),
    RepresentationSpec("pca32_postz", 32, True, True),
)
WORLD_REPRESENTATION_IDS: Final = tuple(spec.representation_id for spec in REPRESENTATIONS if spec.world_tested)
REPRESENTATION_BY_ID: Final = {spec.representation_id: spec for spec in REPRESENTATIONS}

MATCHED_ROWS: Final = mm002.MATCHED_ROWS
MATCHED_COUNTS: Final = dict(mm002.MATCHED_COUNTS)

PROBE_METRICS: Final = (
    "persistence_mse",
    "ridge_mse",
    "shuffle_ridge_mse",
    "ridge_ratio",
    "shuffle_advantage",
)
WORLD_METRICS: Final = mm002.WORLD_METRICS

PROBE_ROW_KEYS: Final = {
    "representation_id",
    "predictor_id",
    "output_dim",
    "fold",
    "video_id",
    "train_rows",
    "test_rows",
    "transform_fingerprint",
    *PROBE_METRICS,
}
PARENT_RIDGE_ROW_KEYS: Final = {
    "fold",
    "video_id",
    "train_rows",
    "test_rows",
    "raw_persistence_mse",
    "raw_ridge_mse",
    "raw_shuffle_ridge_mse",
}
WORLD_ROW_KEYS: Final = {
    "representation_id",
    "output_dim",
    "evidence_origin",
    "fold",
    "seed",
    "video_id",
    "updates",
    "train_rows",
    "test_rows",
    "transform_fingerprint",
    "model_fingerprint",
    "shuffle_model_fingerprint",
    *WORLD_METRICS,
}
INTEGRITY_ROW_KEYS: Final = {
    "representation_id",
    "output_dim",
    "evidence_origin",
    "fold",
    "seed",
    "updates",
    "model_role",
    "encoder_role",
    "pooled_test_rows",
    "transform_fingerprint",
    "model_fingerprint",
    "latent_std_min",
    "latent_effective_rank",
    "prediction_min_variance",
    "prediction_finite",
    "prediction_variance_positive",
}
TRANSFORM_RECORD_KEYS: Final = {
    "fold",
    "representation_id",
    "output_dim",
    "fitted",
    "fit_scope",
    "train_video_ids",
    "excluded_video_ids",
    "train_rows",
    "fit_identity_sha256",
    "fit_matrix_sha256",
    "transform_fingerprint",
    "parameter_arrays",
    "qr_projector_max_abs_error",
    "pca_retained_variance_fraction",
    "pca_rank_below_32",
    "pca_boundary_degenerate",
    "pca_32_to_1_ratio",
    "pca_32_33_relative_gap",
}


@dataclass(frozen=True, slots=True)
class VisualTable:
    """Generic-dimensional visual current/target rows with group identities."""

    video_ids: np.ndarray
    timestamps: np.ndarray
    current: np.ndarray
    target: np.ndarray

    def validate(self, expected_dim: int | None = None) -> None:
        n = len(self.video_ids)
        if np.asarray(self.video_ids).shape != (n,) or np.asarray(self.timestamps).shape != (n,):
            raise ValueError("visual identities must be one-dimensional and row aligned")
        current = np.asarray(self.current, dtype=float)
        target = np.asarray(self.target, dtype=float)
        if current.ndim != 2 or target.shape != current.shape or current.shape[0] != n:
            raise ValueError("visual current/target arrays must have equal two-dimensional shapes")
        if expected_dim is not None and current.shape[1] != expected_dim:
            raise ValueError(f"visual table must have dimension {expected_dim}")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
            raise ValueError("visual table contains non-finite values")
        if not np.all(np.isfinite(np.asarray(self.timestamps, dtype=float))):
            raise ValueError("visual timestamps contain non-finite values")
        if tuple(sorted(set(np.asarray(self.video_ids, dtype=str)))) != tuple(dataset.SAMPLE_VIDEO_IDS):
            raise ValueError("visual table does not contain the exact formal video panel")

    @property
    def dim(self) -> int:
        return int(np.asarray(self.current).shape[1])

    def subset(self, video_ids: Sequence[str]) -> VisualTable:
        wanted = set(video_ids)
        mask = np.asarray([str(value) in wanted for value in self.video_ids], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"no rows for videos {sorted(wanted)}")
        return VisualTable(
            video_ids=np.asarray(self.video_ids)[mask].copy(),
            timestamps=np.asarray(self.timestamps, dtype=float)[mask].copy(),
            current=np.asarray(self.current, dtype=float)[mask].copy(),
            target=np.asarray(self.target, dtype=float)[mask].copy(),
        )


@dataclass(frozen=True, slots=True)
class FittedTransform:
    """A fully frozen affine/projective representation map."""

    representation_id: str
    input_mean: np.ndarray
    input_scale: np.ndarray
    projection: np.ndarray | None
    output_mean: np.ndarray
    output_scale: np.ndarray
    spectrum: np.ndarray
    qr_projector_max_abs_error: float | None
    pca_retained_variance_fraction: float | None
    pca_rank_below_32: bool
    pca_boundary_degenerate: bool
    pca_32_to_1_ratio: float | None
    pca_32_33_relative_gap: float | None

    @property
    def output_dim(self) -> int:
        return len(self.output_mean)

    def apply(self, values: np.ndarray) -> np.ndarray:
        rows = (np.asarray(values, dtype=float) - self.input_mean) / self.input_scale
        if self.projection is not None:
            rows = rows @ self.projection
        return np.asarray((rows - self.output_mean) / self.output_scale, dtype=float)

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "input_mean": np.asarray(self.input_mean, dtype=np.float64),
            "input_scale": np.asarray(self.input_scale, dtype=np.float64),
            "output_mean": np.asarray(self.output_mean, dtype=np.float64),
            "output_scale": np.asarray(self.output_scale, dtype=np.float64),
            "spectrum": np.asarray(self.spectrum, dtype=np.float64),
        }
        if self.projection is not None:
            arrays["projection"] = np.asarray(self.projection, dtype=np.float64)
        return arrays

    def fingerprint(self) -> str:
        digest = sha256(f"mm003-transform-v1:{self.representation_id}".encode("ascii"))
        for name, value in sorted(self.parameter_arrays().items()):
            array = np.asarray(value, dtype="<f8", order="C")
            digest.update(name.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return sha256(array.tobytes(order="C")).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _scale(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.maximum(np.std(np.asarray(values, dtype=float), axis=0, ddof=0), SCALE_FLOOR),
        dtype=float,
    )


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=float).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0.0:
            output[:, column] *= -1.0
    return output


def validate_raw_table(table: VisualTable) -> None:
    table.validate(expected_dim=256)
    expected: list[tuple[str, float]] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        expected.extend((video_id, 1.0 + 0.5 * index) for index in range(dataset.EXPECTED_WINDOW_COUNTS[video_id]))
    actual = list(
        zip(
            np.asarray(table.video_ids, dtype=str).tolist(),
            np.asarray(table.timestamps, dtype=float).tolist(),
            strict=True,
        )
    )
    if len(actual) != 477 or any(
        left_id != right_id or not math.isclose(left_time, right_time, rel_tol=0.0, abs_tol=1e-12)
        for (left_id, left_time), (right_id, right_time) in zip(actual, expected, strict=True)
    ):
        raise ValueError("raw TAESD rows differ from the frozen identity grid")


def alignment_record(
    table: VisualTable,
    projection: np.ndarray,
    saved_current: np.ndarray,
    saved_target: np.ndarray,
) -> dict[str, object]:
    """Prove that flattened raw latents reproduce MM-001 projected features."""

    validate_raw_table(table)
    matrix = np.asarray(projection, dtype=float)
    if matrix.shape != (256, 32) or not np.all(np.isfinite(matrix)):
        raise ValueError("MM-001 vision projection must be finite [256,32]")
    current_error = float(np.max(np.abs(table.current @ matrix - np.asarray(saved_current, dtype=float))))
    target_error = float(np.max(np.abs(table.target @ matrix - np.asarray(saved_target, dtype=float))))
    if not np.allclose(
        table.current @ matrix,
        saved_current,
        rtol=ALIGNMENT_RTOL,
        atol=ALIGNMENT_ATOL,
    ) or not np.allclose(
        table.target @ matrix,
        saved_target,
        rtol=ALIGNMENT_RTOL,
        atol=ALIGNMENT_ATOL,
    ):
        raise ValueError("raw TAESD latents do not reproduce MM-001 fixed features")
    return {
        "rows": len(table.video_ids),
        "raw_dim": table.dim,
        "projected_dim": matrix.shape[1],
        "current_max_abs_error": current_error,
        "target_max_abs_error": target_error,
        "rtol": ALIGNMENT_RTOL,
        "atol": ALIGNMENT_ATOL,
    }


def matched_table(table: VisualTable) -> VisualTable:
    """Return MM-002's exact 461-source one-second panel."""

    validate_raw_table(table)
    ids = np.asarray(table.video_ids, dtype=str)
    times = np.asarray(table.timestamps, dtype=float)
    selected: list[int] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        rows = np.flatnonzero(ids == video_id)
        ordered = rows[np.argsort(times[rows], kind="stable")]
        selected.extend(int(value) for value in ordered[:-2])
    indices = np.asarray(selected, dtype=int)
    output = VisualTable(
        video_ids=ids[indices].copy(),
        timestamps=times[indices].copy(),
        current=np.asarray(table.current, dtype=float)[indices].copy(),
        target=np.asarray(table.target, dtype=float)[indices].copy(),
    )
    output.validate(expected_dim=256)
    counts = {video_id: int(np.sum(output.video_ids == video_id)) for video_id in dataset.SAMPLE_VIDEO_IDS}
    if len(output.video_ids) != MATCHED_ROWS or counts != MATCHED_COUNTS:
        raise ValueError("matched raw panel does not reproduce MM-002 membership")
    return output


def fit_transform(
    representation_id: str,
    train_current: np.ndarray,
    projection: np.ndarray,
) -> FittedTransform:
    """Fit one transform using current training rows only."""

    raw = np.asarray(train_current, dtype=float)
    matrix = np.asarray(projection, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 256 or matrix.shape != (256, 32):
        raise ValueError("transform inputs must be [n,256] and [256,32]")
    zeros256 = np.zeros(256, dtype=float)
    ones256 = np.ones(256, dtype=float)
    spectrum = np.empty(0, dtype=float)
    kwargs: dict[str, Any] = {
        "spectrum": spectrum,
        "qr_projector_max_abs_error": None,
        "pca_retained_variance_fraction": None,
        "pca_rank_below_32": False,
        "pca_boundary_degenerate": False,
        "pca_32_to_1_ratio": None,
        "pca_32_33_relative_gap": None,
    }
    if representation_id == "r32_native":
        return FittedTransform(representation_id, zeros256, ones256, matrix.copy(), np.zeros(32), np.ones(32), **kwargs)
    if representation_id == "r32_postz":
        scores = raw @ matrix
        return FittedTransform(
            representation_id,
            zeros256,
            ones256,
            matrix.copy(),
            scores.mean(axis=0),
            _scale(scores),
            **kwargs,
        )
    if representation_id == "raw256_native":
        return FittedTransform(representation_id, zeros256, ones256, None, np.zeros(256), np.ones(256), **kwargs)
    if representation_id == "raw256_postz":
        return FittedTransform(
            representation_id,
            raw.mean(axis=0),
            _scale(raw),
            None,
            np.zeros(256),
            np.ones(256),
            **kwargs,
        )
    if representation_id == "r32_qr_postz":
        q, _ = np.linalg.qr(matrix, mode="reduced")
        q = _canonicalize_columns(q)
        projector_error = float(np.max(np.abs(q @ q.T - matrix @ np.linalg.pinv(matrix))))
        if projector_error > QR_PROJECTOR_TOLERANCE:
            raise ValueError("QR basis does not preserve the fixed projection subspace")
        scores = raw @ q
        return FittedTransform(
            representation_id,
            zeros256,
            ones256,
            q,
            scores.mean(axis=0),
            _scale(scores),
            qr_projector_max_abs_error=projector_error,
            spectrum=spectrum,
            pca_retained_variance_fraction=None,
            pca_rank_below_32=False,
            pca_boundary_degenerate=False,
            pca_32_to_1_ratio=None,
            pca_32_33_relative_gap=None,
        )
    if representation_id == "pca32_postz":
        mean = raw.mean(axis=0)
        centered = raw - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = _canonicalize_columns(vt[:32].T)
        scores = centered @ components
        ratio = float(singular_values[31] / singular_values[0])
        gap = float((singular_values[31] - singular_values[32]) / singular_values[31])
        retained = float(np.sum(singular_values[:32] ** 2) / np.sum(singular_values**2))
        return FittedTransform(
            representation_id,
            mean,
            ones256,
            components,
            scores.mean(axis=0),
            _scale(scores),
            spectrum=singular_values,
            qr_projector_max_abs_error=None,
            pca_retained_variance_fraction=retained,
            pca_rank_below_32=ratio < PCA_RANK_RATIO_MIN,
            pca_boundary_degenerate=gap < PCA_BOUNDARY_GAP_MIN,
            pca_32_to_1_ratio=ratio,
            pca_32_33_relative_gap=gap,
        )
    raise ValueError(f"unknown representation {representation_id!r}")


def _parameter_records(transform: FittedTransform) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": np.asarray(value, dtype=np.float64).dtype.str,
            "shape": list(np.asarray(value).shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(transform.parameter_arrays().items())
    }


def _identity_sha256(table: VisualTable) -> str:
    identities = [
        [str(video_id), float(timestamp)] for video_id, timestamp in zip(table.video_ids, table.timestamps, strict=True)
    ]
    return _canonical_json_sha256(identities)


def fit_all_transforms(
    table: VisualTable,
    projection: np.ndarray,
) -> tuple[dict[tuple[int, str], FittedTransform], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit and serialize every outer-fold transform."""

    validate_raw_table(table)
    transforms: dict[tuple[int, str], FittedTransform] = {}
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for fold in dataset.formal_folds():
        train = table.subset(fold.train_ids)
        for spec in REPRESENTATIONS:
            transform = fit_transform(spec.representation_id, train.current, projection)
            transforms[(fold.index, spec.representation_id)] = transform
            parameter_records = _parameter_records(transform)
            prefix = f"fold_{fold.index}__{spec.representation_id}__"
            for name, value in sorted(transform.parameter_arrays().items()):
                arrays[prefix + name] = np.asarray(value, dtype=np.float64)
            records.append(
                {
                    "fold": fold.index,
                    "representation_id": spec.representation_id,
                    "output_dim": spec.output_dim,
                    "fitted": spec.fitted,
                    "fit_scope": "outer_train_current_rows_only" if spec.fitted else "fixed_no_data_fit",
                    "train_video_ids": list(fold.train_ids),
                    "excluded_video_ids": list(fold.test_ids),
                    "train_rows": len(train.video_ids),
                    "fit_identity_sha256": _identity_sha256(train),
                    "fit_matrix_sha256": _array_sha256(train.current),
                    "transform_fingerprint": transform.fingerprint(),
                    "parameter_arrays": parameter_records,
                    "qr_projector_max_abs_error": transform.qr_projector_max_abs_error,
                    "pca_retained_variance_fraction": transform.pca_retained_variance_fraction,
                    "pca_rank_below_32": transform.pca_rank_below_32,
                    "pca_boundary_degenerate": transform.pca_boundary_degenerate,
                    "pca_32_to_1_ratio": transform.pca_32_to_1_ratio,
                    "pca_32_33_relative_gap": transform.pca_32_33_relative_gap,
                }
            )
    return transforms, validate_transform_records(records), arrays


def transformed_table(table: VisualTable, transform: FittedTransform) -> VisualTable:
    output = VisualTable(
        video_ids=np.asarray(table.video_ids, dtype=str).copy(),
        timestamps=np.asarray(table.timestamps, dtype=float).copy(),
        current=transform.apply(table.current),
        target=transform.apply(table.target),
    )
    output.validate(expected_dim=transform.output_dim)
    return output


def _ridge_weights(x: np.ndarray, y: np.ndarray, penalty: float = PROBE_PENALTY) -> np.ndarray:
    design = np.c_[np.asarray(x, dtype=float), np.ones(len(x))]
    regularizer = penalty * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ np.asarray(y, dtype=float))


def _ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(np.c_[np.asarray(x, dtype=float), np.ones(len(x))] @ weights, dtype=float)


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean((np.asarray(first, dtype=float) - np.asarray(second, dtype=float)) ** 2))


def _probe_rows_for(
    representation_id: str,
    predictor_id: str,
    fold: dataset.DatasetFold,
    table: VisualTable,
    transform: FittedTransform,
) -> list[dict[str, Any]]:
    transformed = transformed_table(table, transform)
    train = transformed.subset(fold.train_ids)
    x_mean = train.current.mean(axis=0)
    x_scale = _scale(train.current)
    x_train = (train.current - x_mean) / x_scale
    target_mean = train.target.mean(axis=0)
    target_scale = _scale(train.target)
    ordered_target = (train.target - target_mean) / target_scale
    derangement = core.temporal_derangement(
        core.FeatureTable(
            video_ids=train.video_ids,
            timestamps=train.timestamps,
            vision=np.zeros((len(train.video_ids), 32)),
            audio=np.zeros((len(train.video_ids), 32)),
            text=np.zeros((len(train.video_ids), 32)),
            target_vision=np.zeros((len(train.video_ids), 32)),
            annotation_present=np.zeros(len(train.video_ids), dtype=bool),
        )
    )
    if predictor_id == "absolute_target":
        ordered_fit_target = ordered_target
        shuffled_fit_target = (train.target[derangement] - target_mean) / target_scale
        output_mean = None
        output_scale = None
    elif predictor_id == "residual_delta":
        delta = train.target - train.current
        output_mean = delta.mean(axis=0)
        output_scale = _scale(delta)
        ordered_fit_target = (delta - output_mean) / output_scale
        shuffled_delta = train.target[derangement] - train.current
        shuffled_fit_target = (shuffled_delta - output_mean) / output_scale
    else:
        raise ValueError(f"unknown probe predictor {predictor_id!r}")
    ordered_weights = _ridge_weights(x_train, ordered_fit_target)
    shuffled_weights = _ridge_weights(x_train, shuffled_fit_target)
    rows: list[dict[str, Any]] = []
    for video_id in fold.test_ids:
        test = transformed.subset([video_id])
        x_test = (test.current - x_mean) / x_scale
        ordered_prediction = _ridge_predict(x_test, ordered_weights)
        shuffled_prediction = _ridge_predict(x_test, shuffled_weights)
        if predictor_id == "residual_delta":
            assert output_mean is not None and output_scale is not None
            ordered_prediction = (
                test.current + ordered_prediction * output_scale + output_mean - target_mean
            ) / target_scale
            shuffled_prediction = (
                test.current + shuffled_prediction * output_scale + output_mean - target_mean
            ) / target_scale
        truth = (test.target - target_mean) / target_scale
        persistence = (test.current - target_mean) / target_scale
        persistence_mse = _mse(persistence, truth)
        ridge_mse = _mse(ordered_prediction, truth)
        shuffle_mse = _mse(shuffled_prediction, truth)
        rows.append(
            {
                "representation_id": representation_id,
                "predictor_id": predictor_id,
                "output_dim": transform.output_dim,
                "fold": fold.index,
                "video_id": video_id,
                "train_rows": len(train.video_ids),
                "test_rows": len(test.video_ids),
                "transform_fingerprint": transform.fingerprint(),
                "persistence_mse": persistence_mse,
                "ridge_mse": ridge_mse,
                "shuffle_ridge_mse": shuffle_mse,
                "ridge_ratio": ridge_mse / persistence_mse,
                "shuffle_advantage": shuffle_mse / ridge_mse,
            }
        )
    return rows


def _parent_ridge_rows(
    matched: VisualTable,
    projection: np.ndarray,
) -> list[dict[str, Any]]:
    fixed = transformed_table(
        matched,
        fit_transform("r32_native", matched.current, projection),
    )
    rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        train = fixed.subset(fold.train_ids)
        weights = _ridge_weights(train.current, train.target)
        derangement = core.temporal_derangement(
            core.FeatureTable(
                video_ids=train.video_ids,
                timestamps=train.timestamps,
                vision=np.zeros((len(train.video_ids), 32)),
                audio=np.zeros((len(train.video_ids), 32)),
                text=np.zeros((len(train.video_ids), 32)),
                target_vision=np.zeros((len(train.video_ids), 32)),
                annotation_present=np.zeros(len(train.video_ids), dtype=bool),
            )
        )
        shuffle_weights = _ridge_weights(train.current, train.target[derangement])
        for video_id in fold.test_ids:
            test = fixed.subset([video_id])
            rows.append(
                {
                    "fold": fold.index,
                    "video_id": video_id,
                    "train_rows": len(train.video_ids),
                    "test_rows": len(test.video_ids),
                    "raw_persistence_mse": _mse(test.current, test.target),
                    "raw_ridge_mse": _mse(_ridge_predict(test.current, weights), test.target),
                    "raw_shuffle_ridge_mse": _mse(_ridge_predict(test.current, shuffle_weights), test.target),
                }
            )
    return rows


def parent_preflight_record(
    table: VisualTable,
    projection: np.ndarray,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Check every inherited parent evidence identity before the formal marker."""

    parent = mm002.validate_evidence(parent_evidence)
    actual_rows = _parent_ridge_rows(matched_table(table), projection)
    expected_rows = [row for row in parent["raw_probe_rows"] if row["probe_id"] == "matched_1s"]
    if len(actual_rows) != len(expected_rows):
        raise ValueError("MM-003 parent ridge preflight row count differs")
    max_error = 0.0
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        for name in PARENT_RIDGE_ROW_KEYS:
            left = actual[name]
            right = expected[name]
            if isinstance(left, (int, float)) and not isinstance(left, bool):
                error = abs(float(left) - float(cast(float, right)))
                max_error = max(max_error, error)
                if not math.isclose(
                    float(left),
                    float(cast(float, right)),
                    rel_tol=PARITY_RTOL,
                    abs_tol=PARITY_ATOL,
                ):
                    raise ValueError(f"MM-003 parent ridge preflight differs for {name}")
            elif left != right:
                raise ValueError(f"MM-003 parent ridge preflight differs for {name}")
    world_rows = [row for row in parent["world_rows"] if row["variant_id"] == "full_1s_1500"]
    integrity_rows = [
        row for row in parent["integrity_rows"] if row["trajectory_id"] == "full_1s" and row["updates"] in CHECKPOINTS
    ]
    if len(world_rows) != 24 or len(integrity_rows) != 144:
        raise ValueError("MM-003 inherited world/integrity preflight membership differs")
    return {
        "passed": True,
        "parent_ridge_rows": len(actual_rows),
        "inherited_world_rows": len(world_rows),
        "inherited_integrity_rows": len(integrity_rows),
        "max_absolute_error": max_error,
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
    }


def _world_model(obs_dim: int, seed: int) -> FlatWorldModel:
    return FlatWorldModel(
        obs_dim=obs_dim,
        action_dim=1,
        latent_dim=core.LATENT_DIM,
        hidden=core.WORLD_HIDDEN,
        ensemble=core.WORLD_ENSEMBLE,
        lr=core.WORLD_LR,
        ema_tau=core.WORLD_EMA_TAU,
        w_reward=core.WORLD_W_REWARD,
        w_inverse=core.WORLD_W_INVERSE,
        w_var=core.WORLD_W_VAR,
        w_cov=core.WORLD_W_COV,
        seed=seed,
    )


def _fit_world_trajectory(
    table: VisualTable,
    seed: int,
    *,
    shuffled: bool,
) -> dict[int, FlatWorldModel]:
    targets = np.asarray(table.target, dtype=float)
    if shuffled:
        control = core.FeatureTable(
            video_ids=table.video_ids,
            timestamps=table.timestamps,
            vision=np.zeros((len(table.video_ids), 32)),
            audio=np.zeros((len(table.video_ids), 32)),
            text=np.zeros((len(table.video_ids), 32)),
            target_vision=np.zeros((len(table.video_ids), 32)),
            annotation_present=np.zeros(len(table.video_ids), dtype=bool),
        )
        targets = targets[core.temporal_derangement(control)]
    transitions = [
        Transition(
            state=LatentState(z=np.asarray(source, dtype=float)),
            action=core.NULL_ACTION,
            next_state=LatentState(z=np.asarray(target, dtype=float)),
            reward=0.0,
        )
        for source, target in zip(table.current, targets, strict=True)
    ]
    model = _world_model(table.dim, seed)
    rng = np.random.default_rng(seed + core.WORLD_SAMPLE_SEED_OFFSET)
    batch_size = min(core.WORLD_BATCH, len(transitions))
    output: dict[int, FlatWorldModel] = {}
    for completed in range(1, CHECKPOINTS[-1] + 1):
        indices = rng.integers(0, len(transitions), size=batch_size)
        model.update([transitions[int(index)] for index in indices])
        if completed in CHECKPOINTS:
            output[completed] = deepcopy(model)
    return output


def _target_latents(model: FlatWorldModel, values: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.encode_target(row).z, dtype=float) for row in values])


def _online_latents(model: FlatWorldModel, values: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.encode(row).z, dtype=float) for row in values])


def _predictions(model: FlatWorldModel, latents: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(model.predict(LatentState(z=row), core.NULL_ACTION).mean) for row in latents])


def _latent_ridge(model: FlatWorldModel, train: VisualTable, test: VisualTable) -> np.ndarray:
    weights = _ridge_weights(train.current, _target_latents(model, train.target), core.RIDGE_PENALTY)
    return _ridge_predict(test.current, weights)


def _world_metrics(
    model: FlatWorldModel,
    shuffled_model: FlatWorldModel,
    train: VisualTable,
    test: VisualTable,
) -> dict[str, float]:
    targets = _target_latents(model, test.target)
    current = _target_latents(model, test.current)
    prediction = _predictions(model, _online_latents(model, test.current))
    shuffled_targets = _target_latents(shuffled_model, test.target)
    shuffled_current = _target_latents(shuffled_model, test.current)
    shuffled_prediction = _predictions(shuffled_model, _online_latents(shuffled_model, test.current))
    return {
        "world_mse": _mse(prediction, targets),
        "persistence_mse": _mse(current, targets),
        "ridge_mse": _mse(_latent_ridge(model, train, test), targets),
        "shuffle_model_mse": _mse(shuffled_prediction, shuffled_targets),
        "shuffle_model_persistence_mse": _mse(shuffled_current, shuffled_targets),
    }


def _effective_rank(latents: np.ndarray) -> float:
    covariance = np.cov(np.asarray(latents, dtype=float).T) + 1e-8 * np.eye(latents.shape[1])
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2))


def _integrity_rows(
    representation_id: str,
    output_dim: int,
    origin: str,
    fold: dataset.DatasetFold,
    seed: int,
    update: int,
    role: str,
    model: FlatWorldModel,
    pooled_test: VisualTable,
    transform_fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fingerprint = core.model_fingerprint(model)
    for encoder_role in ("online", "ema_target"):
        states = (
            [model.encode(row) for row in pooled_test.current]
            if encoder_role == "online"
            else [model.encode_target(row) for row in pooled_test.current]
        )
        latents = np.stack([np.asarray(state.z, dtype=float) for state in states])
        predictions = [model.predict(state, core.NULL_ACTION) for state in states]
        variances = np.stack([np.asarray(prediction.var, dtype=float) for prediction in predictions])
        finite = all(
            np.all(np.isfinite(prediction.mean))
            and np.all(np.isfinite(prediction.var))
            and math.isfinite(float(prediction.epistemic))
            and math.isfinite(float(prediction.aleatoric))
            for prediction in predictions
        )
        rows.append(
            {
                "representation_id": representation_id,
                "output_dim": output_dim,
                "evidence_origin": origin,
                "fold": fold.index,
                "seed": seed,
                "updates": update,
                "model_role": role,
                "encoder_role": encoder_role,
                "pooled_test_rows": len(pooled_test.video_ids),
                "transform_fingerprint": transform_fingerprint,
                "model_fingerprint": fingerprint,
                "latent_std_min": float(np.std(latents, axis=0, ddof=0).min()),
                "latent_effective_rank": _effective_rank(latents),
                "prediction_min_variance": float(variances.min()),
                "prediction_finite": bool(finite),
                "prediction_variance_positive": bool(np.all(variances > 0.0)),
            }
        )
    return rows


def _inherit_world_rows(
    parent_evidence: Mapping[str, object],
    transforms: Mapping[tuple[int, str], FittedTransform],
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[tuple[int, int, int, str, str], dict[str, Any]]]:
    world_lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in cast(Sequence[Mapping[str, Any]], parent_evidence["world_rows"]):
        if row["variant_id"] != "full_1s_1500":
            continue
        fold = int(row["fold"])
        mapped = {
            "representation_id": "r32_native",
            "output_dim": 32,
            "evidence_origin": "inherited_MM002",
            "fold": fold,
            "seed": int(row["seed"]),
            "video_id": str(row["video_id"]),
            "updates": 1_500,
            "train_rows": int(row["train_rows"]),
            "test_rows": int(row["test_rows"]),
            "transform_fingerprint": transforms[(fold, "r32_native")].fingerprint(),
            "model_fingerprint": row["model_fingerprint"],
            "shuffle_model_fingerprint": row["shuffle_model_fingerprint"],
            **{name: float(row[name]) for name in WORLD_METRICS},
        }
        world_lookup[(fold, int(row["seed"]), str(row["video_id"]))] = mapped
    integrity_lookup: dict[tuple[int, int, int, str, str], dict[str, Any]] = {}
    for row in cast(Sequence[Mapping[str, Any]], parent_evidence["integrity_rows"]):
        if row["trajectory_id"] != "full_1s" or int(row["updates"]) not in CHECKPOINTS:
            continue
        fold = int(row["fold"])
        mapped = {
            "representation_id": "r32_native",
            "output_dim": 32,
            "evidence_origin": "inherited_MM002",
            "fold": fold,
            "seed": int(row["seed"]),
            "updates": int(row["updates"]),
            "model_role": row["model_role"],
            "encoder_role": row["encoder_role"],
            "pooled_test_rows": int(row["pooled_test_rows"]),
            "transform_fingerprint": transforms[(fold, "r32_native")].fingerprint(),
            "model_fingerprint": row["model_fingerprint"],
            "latent_std_min": float(row["latent_std_min"]),
            "latent_effective_rank": float(row["latent_effective_rank"]),
            "prediction_min_variance": float(row["prediction_min_variance"]),
            "prediction_finite": bool(row["prediction_finite"]),
            "prediction_variance_positive": bool(row["prediction_variance_positive"]),
        }
        key = (
            fold,
            int(row["seed"]),
            int(row["updates"]),
            str(row["model_role"]),
            str(row["encoder_role"]),
        )
        integrity_lookup[key] = mapped
    return world_lookup, integrity_lookup


def execute(
    table: VisualTable,
    projection: np.ndarray,
    parent_evidence: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, object]]:
    """Execute all frozen transforms, probes, and new world trajectories."""

    validate_raw_table(table)
    transforms, transform_records, transform_arrays = fit_all_transforms(table, projection)
    matched = matched_table(table)
    probe_rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        for spec in REPRESENTATIONS:
            transform = transforms[(fold.index, spec.representation_id)]
            for predictor_id in PREDICTOR_IDS:
                probe_rows.extend(
                    _probe_rows_for(
                        spec.representation_id,
                        predictor_id,
                        fold,
                        matched,
                        transform,
                    )
                )
    parent_ridge_rows = _parent_ridge_rows(matched, projection)
    inherited_world, inherited_integrity = _inherit_world_rows(parent_evidence, transforms)
    world_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        for seed in core.SEEDS:
            for representation_id in WORLD_REPRESENTATION_IDS:
                transform = transforms[(fold.index, representation_id)]
                if representation_id == "r32_native":
                    for video_id in fold.test_ids:
                        world_rows.append(inherited_world[(fold.index, seed, video_id)])
                    for update in CHECKPOINTS:
                        for role in ("primary", "shuffle"):
                            for encoder_role in ("online", "ema_target"):
                                integrity_rows.append(
                                    inherited_integrity[(fold.index, seed, update, role, encoder_role)]
                                )
                    continue
                transformed = transformed_table(table, transform)
                train = transformed.subset(fold.train_ids)
                pooled_test = transformed.subset(fold.test_ids)
                primary = _fit_world_trajectory(train, seed, shuffled=False)
                shuffled = _fit_world_trajectory(train, seed, shuffled=True)
                for update in CHECKPOINTS:
                    for role, model in (
                        ("primary", primary[update]),
                        ("shuffle", shuffled[update]),
                    ):
                        integrity_rows.extend(
                            _integrity_rows(
                                representation_id,
                                transform.output_dim,
                                "trained_MM003",
                                fold,
                                seed,
                                update,
                                role,
                                model,
                                pooled_test,
                                transform.fingerprint(),
                            )
                        )
                model = primary[1_500]
                shuffled_model = shuffled[1_500]
                for video_id in fold.test_ids:
                    test = transformed.subset([video_id])
                    world_rows.append(
                        {
                            "representation_id": representation_id,
                            "output_dim": transform.output_dim,
                            "evidence_origin": "trained_MM003",
                            "fold": fold.index,
                            "seed": seed,
                            "video_id": video_id,
                            "updates": 1_500,
                            "train_rows": len(train.video_ids),
                            "test_rows": len(test.video_ids),
                            "transform_fingerprint": transform.fingerprint(),
                            "model_fingerprint": core.model_fingerprint(model),
                            "shuffle_model_fingerprint": core.model_fingerprint(shuffled_model),
                            **_world_metrics(model, shuffled_model, train, test),
                        }
                    )
    evidence = validate_evidence(
        {
            "schema_version": SCHEMA_VERSION,
            "probe_rows": probe_rows,
            "parent_ridge_rows": parent_ridge_rows,
            "world_rows": world_rows,
            "integrity_rows": integrity_rows,
        }
    )
    return transform_arrays, transform_records, evidence


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_transform_records(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("transform records must be an array")
    records: list[dict[str, Any]] = []
    identities: list[tuple[int, str]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != TRANSFORM_RECORD_KEYS:
            raise ValueError("transform record schema does not match MM-003")
        row = dict(raw)
        representation_id = row["representation_id"]
        if type(row["fold"]) is not int or representation_id not in REPRESENTATION_BY_ID:
            raise ValueError("transform identity is invalid")
        spec = REPRESENTATION_BY_ID[cast(str, representation_id)]
        fold = dataset.formal_folds()[row["fold"]]
        if (
            row["output_dim"] != spec.output_dim
            or row["fitted"] is not spec.fitted
            or row["train_video_ids"] != list(fold.train_ids)
            or row["excluded_video_ids"] != list(fold.test_ids)
            or row["train_rows"] != sum(dataset.EXPECTED_WINDOW_COUNTS[item] for item in fold.train_ids)
        ):
            raise ValueError("transform record conflicts with its frozen fold/spec")
        for name in ("fit_identity_sha256", "fit_matrix_sha256", "transform_fingerprint"):
            _fingerprint(row[name], name)
        parameters = row["parameter_arrays"]
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("transform parameter records are invalid")
        for name, record in parameters.items():
            if (
                not isinstance(name, str)
                or not isinstance(record, dict)
                or set(record)
                != {
                    "dtype",
                    "shape",
                    "sha256",
                }
            ):
                raise ValueError("transform array record is invalid")
            if record["dtype"] != "<f8" or not isinstance(record["shape"], list):
                raise ValueError("transform arrays must be little-endian float64")
            _fingerprint(record["sha256"], "parameter sha256")
        for name in (
            "qr_projector_max_abs_error",
            "pca_retained_variance_fraction",
            "pca_32_to_1_ratio",
            "pca_32_33_relative_gap",
        ):
            if row[name] is not None:
                _finite(row[name], name)
        for name in ("pca_rank_below_32", "pca_boundary_degenerate"):
            if type(row[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        identities.append((row["fold"], cast(str, representation_id)))
        records.append(row)
    expected = [(fold.index, spec.representation_id) for fold in dataset.formal_folds() for spec in REPRESENTATIONS]
    if identities != expected:
        raise ValueError("transform records are incomplete, duplicated, or reordered")
    return records


def validate_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "probe_rows",
        "parent_ridge_rows",
        "world_rows",
        "integrity_rows",
    }:
        raise ValueError("MM-003 evidence schema is invalid")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("MM-003 evidence schema version differs")
    probe_rows = _validate_probe_rows(value["probe_rows"])
    parent_rows = _validate_parent_ridge_rows(value["parent_ridge_rows"])
    world_rows = _validate_world_rows(value["world_rows"])
    integrity_rows = _validate_integrity_rows(value["integrity_rows"])
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_rows": probe_rows,
        "parent_ridge_rows": parent_rows,
        "world_rows": world_rows,
        "integrity_rows": integrity_rows,
    }


def _validate_probe_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("probe rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, str, str, str]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != PROBE_ROW_KEYS:
            raise ValueError("probe row schema does not match MM-003")
        row = dict(raw)
        rep = row["representation_id"]
        predictor = row["predictor_id"]
        if rep not in REPRESENTATION_BY_ID or predictor not in PREDICTOR_IDS:
            raise ValueError("unknown probe representation/predictor")
        if type(row["fold"]) is not int or not isinstance(row["video_id"], str):
            raise ValueError("probe identity fields are invalid")
        fold = dataset.formal_folds()[row["fold"]]
        if row["video_id"] not in fold.test_ids or row["output_dim"] != REPRESENTATION_BY_ID[rep].output_dim:
            raise ValueError("probe row conflicts with frozen fold/spec")
        expected_train = sum(MATCHED_COUNTS[item] for item in fold.train_ids)
        if row["train_rows"] != expected_train or row["test_rows"] != MATCHED_COUNTS[row["video_id"]]:
            raise ValueError("probe row counts differ from matched panel")
        _fingerprint(row["transform_fingerprint"], "transform_fingerprint")
        for name in PROBE_METRICS:
            _finite(row[name], name, positive=True)
        if not math.isclose(
            row["ridge_ratio"], row["ridge_mse"] / row["persistence_mse"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("probe ridge ratio does not recompute")
        if not math.isclose(
            row["shuffle_advantage"], row["shuffle_ridge_mse"] / row["ridge_mse"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("probe shuffle advantage does not recompute")
        identities.append((row["fold"], rep, predictor, row["video_id"]))
        rows.append(row)
    expected = [
        (fold.index, spec.representation_id, predictor, video_id)
        for fold in dataset.formal_folds()
        for spec in REPRESENTATIONS
        for predictor in PREDICTOR_IDS
        for video_id in fold.test_ids
    ]
    if identities != expected:
        raise ValueError("probe rows are incomplete, duplicated, or reordered")
    return rows


def _validate_parent_ridge_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("parent ridge rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, str]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != PARENT_RIDGE_ROW_KEYS:
            raise ValueError("parent ridge row schema differs")
        row = dict(raw)
        if type(row["fold"]) is not int or not isinstance(row["video_id"], str):
            raise ValueError("parent ridge identity is invalid")
        fold = dataset.formal_folds()[row["fold"]]
        if row["video_id"] not in fold.test_ids:
            raise ValueError("parent ridge test video differs")
        if (
            row["train_rows"] != sum(MATCHED_COUNTS[item] for item in fold.train_ids)
            or row["test_rows"] != MATCHED_COUNTS[row["video_id"]]
        ):
            raise ValueError("parent ridge row counts differ")
        for name in ("raw_persistence_mse", "raw_ridge_mse", "raw_shuffle_ridge_mse"):
            _finite(row[name], name, positive=True)
        identities.append((row["fold"], row["video_id"]))
        rows.append(row)
    expected = [(fold.index, video_id) for fold in dataset.formal_folds() for video_id in fold.test_ids]
    if identities != expected:
        raise ValueError("parent ridge rows are incomplete or reordered")
    return rows


def _validate_world_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("world rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, int, str, str]] = []
    fingerprints: dict[tuple[int, int, str], tuple[str, str]] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != WORLD_ROW_KEYS:
            raise ValueError("world row schema does not match MM-003")
        row = dict(raw)
        rep = row["representation_id"]
        if rep not in WORLD_REPRESENTATION_IDS or type(row["fold"]) is not int or type(row["seed"]) is not int:
            raise ValueError("world identity is invalid")
        fold = dataset.formal_folds()[row["fold"]]
        if row["video_id"] not in fold.test_ids or row["seed"] not in core.SEEDS:
            raise ValueError("world row fold/seed/video differs")
        if row["updates"] != 1_500 or row["output_dim"] != 32:
            raise ValueError("world row budget/dimension differs")
        if (
            row["train_rows"] != sum(dataset.EXPECTED_WINDOW_COUNTS[item] for item in fold.train_ids)
            or row["test_rows"] != dataset.EXPECTED_WINDOW_COUNTS[row["video_id"]]
        ):
            raise ValueError("world row counts differ")
        expected_origin = "inherited_MM002" if rep == "r32_native" else "trained_MM003"
        if row["evidence_origin"] != expected_origin:
            raise ValueError("world evidence origin differs")
        transform_fp = _fingerprint(row["transform_fingerprint"], "transform_fingerprint")
        primary = _fingerprint(row["model_fingerprint"], "model_fingerprint")
        shuffled = _fingerprint(row["shuffle_model_fingerprint"], "shuffle_model_fingerprint")
        key = (row["fold"], row["seed"], rep)
        expected_fp = (transform_fp + primary, shuffled)
        if key in fingerprints and fingerprints[key] != expected_fp:
            raise ValueError("world fingerprints differ within a run")
        fingerprints[key] = expected_fp
        for name in WORLD_METRICS:
            _finite(row[name], name, positive=True)
        identities.append((row["fold"], row["seed"], rep, row["video_id"]))
        rows.append(row)
    expected = [
        (fold.index, seed, rep, video_id)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for rep in WORLD_REPRESENTATION_IDS
        for video_id in fold.test_ids
    ]
    if identities != expected:
        raise ValueError("world rows are incomplete, duplicated, or reordered")
    return rows


def _validate_integrity_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("integrity rows must be an array")
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, int, str, int, str, str]] = []
    run_fingerprints: dict[tuple[int, int, str, int, str], tuple[str, str]] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != INTEGRITY_ROW_KEYS:
            raise ValueError("integrity row schema does not match MM-003")
        row = dict(raw)
        rep = row["representation_id"]
        if rep not in WORLD_REPRESENTATION_IDS or row["updates"] not in CHECKPOINTS:
            raise ValueError("integrity representation/checkpoint differs")
        if row["model_role"] not in ("primary", "shuffle") or row["encoder_role"] not in ("online", "ema_target"):
            raise ValueError("integrity roles differ")
        fold = dataset.formal_folds()[row["fold"]]
        expected_pooled = sum(dataset.EXPECTED_WINDOW_COUNTS[item] for item in fold.test_ids)
        if row["pooled_test_rows"] != expected_pooled or row["output_dim"] != 32:
            raise ValueError("integrity pooled count/dimension differs")
        expected_origin = "inherited_MM002" if rep == "r32_native" else "trained_MM003"
        if row["evidence_origin"] != expected_origin:
            raise ValueError("integrity evidence origin differs")
        transform_fp = _fingerprint(row["transform_fingerprint"], "transform_fingerprint")
        model_fp = _fingerprint(row["model_fingerprint"], "model_fingerprint")
        key = (row["fold"], row["seed"], rep, row["updates"], row["model_role"])
        pair = (transform_fp, model_fp)
        if key in run_fingerprints and run_fingerprints[key] != pair:
            raise ValueError("integrity fingerprints differ between encoders")
        run_fingerprints[key] = pair
        for name in ("latent_std_min", "latent_effective_rank", "prediction_min_variance"):
            _finite(row[name], name, positive=name == "prediction_min_variance")
        for name in ("prediction_finite", "prediction_variance_positive"):
            if type(row[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        identities.append((row["fold"], row["seed"], rep, row["updates"], row["model_role"], row["encoder_role"]))
        rows.append(row)
    expected = [
        (fold.index, seed, rep, update, role, encoder)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for rep in WORLD_REPRESENTATION_IDS
        for update in CHECKPOINTS
        for role in ("primary", "shuffle")
        for encoder in ("online", "ema_target")
    ]
    if identities != expected:
        raise ValueError("integrity rows are incomplete, duplicated, or reordered")
    return rows


def assert_parent_parity(
    evidence: object,
    parent_evidence: Mapping[str, object],
) -> dict[str, object]:
    normalized = validate_evidence(evidence)
    parent = mm002.validate_evidence(parent_evidence)
    max_error = 0.0

    def same(left: object, right: object, label: str) -> None:
        nonlocal max_error
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            error = abs(float(left) - float(right))
            max_error = max(max_error, error)
            if not math.isclose(float(left), float(right), rel_tol=PARITY_RTOL, abs_tol=PARITY_ATOL):
                raise ValueError(f"MM-003 parent parity failed for {label}")
        elif left != right:
            raise ValueError(f"MM-003 parent parity failed for {label}")

    parent_raw = {
        (row["fold"], row["video_id"]): row for row in parent["raw_probe_rows"] if row["probe_id"] == "matched_1s"
    }
    for row in cast(Sequence[Mapping[str, Any]], normalized["parent_ridge_rows"]):
        old = parent_raw[(row["fold"], row["video_id"])]
        for name in PARENT_RIDGE_ROW_KEYS:
            if name in ("train_rows", "test_rows", "fold", "video_id"):
                same(row[name], old[name], f"raw {row['video_id']} {name}")
            else:
                same(row[name], old[name], f"raw {row['video_id']} {name}")

    parent_world = {
        (row["fold"], row["seed"], row["video_id"]): row
        for row in parent["world_rows"]
        if row["variant_id"] == "full_1s_1500"
    }
    inherited_world = [
        row
        for row in cast(Sequence[Mapping[str, Any]], normalized["world_rows"])
        if row["representation_id"] == "r32_native"
    ]
    for row in inherited_world:
        old = parent_world[(row["fold"], row["seed"], row["video_id"])]
        for new_name, old_name in (
            ("model_fingerprint", "model_fingerprint"),
            ("shuffle_model_fingerprint", "shuffle_model_fingerprint"),
            ("world_mse", "world_mse"),
            ("persistence_mse", "persistence_mse"),
            ("ridge_mse", "ridge_mse"),
            ("shuffle_model_mse", "shuffle_model_mse"),
            ("shuffle_model_persistence_mse", "shuffle_model_persistence_mse"),
        ):
            same(row[new_name], old[old_name], f"world {row['fold']}/{row['seed']}/{row['video_id']} {new_name}")

    parent_integrity = {
        (row["fold"], row["seed"], row["updates"], row["model_role"], row["encoder_role"]): row
        for row in parent["integrity_rows"]
        if row["trajectory_id"] == "full_1s" and row["updates"] in CHECKPOINTS
    }
    inherited_integrity = [
        row
        for row in cast(Sequence[Mapping[str, Any]], normalized["integrity_rows"])
        if row["representation_id"] == "r32_native"
    ]
    for row in inherited_integrity:
        key = (row["fold"], row["seed"], row["updates"], row["model_role"], row["encoder_role"])
        old = parent_integrity[key]
        for name in (
            "model_fingerprint",
            "pooled_test_rows",
            "latent_std_min",
            "latent_effective_rank",
            "prediction_min_variance",
            "prediction_finite",
            "prediction_variance_positive",
        ):
            same(row[name], old[name], f"integrity {key} {name}")
    return {
        "passed": True,
        "parent_ridge_rows_compared": len(cast(Sequence[Mapping[str, Any]], normalized["parent_ridge_rows"])),
        "parent_world_rows_compared": len(inherited_world),
        "parent_integrity_rows_compared": len(inherited_integrity),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "max_absolute_error": max_error,
    }


def _probe_support(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["ridge_mse"]) * 1.2 <= float(row["persistence_mse"])
        and float(row["ridge_mse"]) * 1.1 <= float(row["shuffle_ridge_mse"])
    )


def _world_support(row: Mapping[str, Any]) -> bool:
    world = float(row["world_mse"])
    persistence = float(row["persistence_mse"])
    return bool(
        world * core.VISUAL_PERSISTENCE_FACTOR <= persistence
        and world < float(row["ridge_mse"])
        and (world / persistence) * core.VISUAL_SHUFFLE_MARGIN
        <= float(row["shuffle_model_mse"]) / float(row["shuffle_model_persistence_mse"])
    )


def _world_video_medians(rows: Sequence[Mapping[str, Any]], rep: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        group = [row for row in rows if row["representation_id"] == rep and row["video_id"] == video_id]
        output.append(
            {
                "representation_id": rep,
                "video_id": video_id,
                **{name: float(np.median([float(row[name]) for row in group])) for name in WORLD_METRICS},
            }
        )
    return output


def _diagnostic_decision(
    probe_passes: Mapping[str, bool],
    world_passes: Mapping[str, bool],
    world_healthy: Mapping[str, bool],
    contrast_improvements: Mapping[tuple[str, str], bool],
    *,
    pca_stable: bool,
) -> tuple[list[str], list[str], str]:
    """Apply the preregistered causal branches to compact boolean evidence."""

    labels: list[str] = []
    full_probe = bool(probe_passes["raw256_native"] or probe_passes["raw256_postz"])
    fixed_subspace_probe = any(probe_passes[rep] for rep in ("r32_native", "r32_postz", "r32_qr_postz"))
    pca_probe = bool(probe_passes["pca32_postz"])
    if pca_stable and pca_probe and not fixed_subspace_probe and full_probe:
        labels.append("fixed_random_subspace_linear_signal_loss_supported")
    if full_probe and not fixed_subspace_probe and not pca_probe:
        labels.append("tested_32d_compression_linear_signal_loss_supported")
    if not full_probe:
        labels.append("no_linear_full_taesd_signal_at_frozen_margin")

    def rescue(candidate: str, comparator: str) -> bool:
        return bool(
            world_healthy[candidate] and world_passes[candidate] and contrast_improvements[(candidate, comparator)]
        )

    if any(world_passes[rep] and not world_healthy[rep] for rep in WORLD_REPRESENTATION_IDS):
        labels.append("apparent_rescue_via_representation_collapse")

    primary: list[str] = []
    if rescue("r32_postz", "r32_native") and (not world_passes["r32_native"] or not world_healthy["r32_native"]):
        primary.append("mm001_coordinate_scale_cause_supported")
    if rescue("r32_qr_postz", "r32_postz") and (not world_passes["r32_postz"] or not world_healthy["r32_postz"]):
        primary.append("fixed_subspace_basis_conditioning_cause_supported")
    pca_world_rescue = rescue("pca32_postz", "r32_qr_postz")
    if pca_world_rescue and pca_stable and pca_probe and full_probe and not fixed_subspace_probe:
        primary.append("fixed_random_subspace_information_loss_supported")
    elif pca_world_rescue:
        labels.append("pca_subspace_world_sensitivity_only")

    corresponding_path_supported = any(
        probe_passes[rep] and world_healthy[rep] and world_passes[rep] for rep in WORLD_REPRESENTATION_IDS
    )
    any_32_probe = any(probe_passes[rep] for rep in WORLD_REPRESENTATION_IDS)
    if any_32_probe and not corresponding_path_supported:
        labels.append("linear_temporal_signal_present_world_path_not_supported")
    if full_probe and not primary:
        labels.append("full_information_signal_present_no_compatible_32d_fix")
    if not primary:
        labels.append("tested_projection_scale_factors_not_supported")

    if len(primary) > 1:
        classification = "multiple_projection_scale_mechanisms_supported"
    elif primary:
        classification = primary[0]
    elif "pca_subspace_world_sensitivity_only" in labels:
        classification = "pca_subspace_world_sensitivity_only"
    elif "linear_temporal_signal_present_world_path_not_supported" in labels:
        classification = "linear_temporal_signal_present_world_path_not_supported"
    elif "full_information_signal_present_no_compatible_32d_fix" in labels:
        classification = "full_information_signal_present_no_compatible_32d_fix"
    elif "no_linear_full_taesd_signal_at_frozen_margin" in labels:
        classification = "no_linear_full_taesd_signal_at_frozen_margin"
    else:
        classification = "tested_projection_scale_factors_not_supported"
    return labels, primary, classification


def summarize(
    evidence: object,
    transform_records: object,
    parent_evidence: Mapping[str, object],
) -> dict[str, Any]:
    normalized = validate_evidence(evidence)
    records = validate_transform_records(transform_records)
    transform_fingerprints = {(row["fold"], row["representation_id"]): row["transform_fingerprint"] for row in records}
    for section in ("probe_rows", "world_rows", "integrity_rows"):
        for row in cast(Sequence[Mapping[str, Any]], normalized[section]):
            key = (row["fold"], row["representation_id"])
            if row["transform_fingerprint"] != transform_fingerprints[key]:
                raise ValueError(f"{section} transform fingerprint differs from its fitted transform")
    parity = assert_parent_parity(normalized, parent_evidence)
    probe_rows = cast(list[dict[str, Any]], normalized["probe_rows"])
    probe_summary: list[dict[str, Any]] = []
    for spec in REPRESENTATIONS:
        for predictor in PREDICTOR_IDS:
            rows = [
                row
                for row in probe_rows
                if row["representation_id"] == spec.representation_id and row["predictor_id"] == predictor
            ]
            support = sum(_probe_support(row) for row in rows)
            probe_summary.append(
                {
                    "representation_id": spec.representation_id,
                    "predictor_id": predictor,
                    "supporting_videos": support,
                    "passes": support >= core.REQUIRED_VIDEO_SUPPORT,
                    "video_rows": rows,
                }
            )
    probes = {(row["representation_id"], row["predictor_id"]): row for row in probe_summary}

    invariant_pairs: list[dict[str, Any]] = []
    for left, right in (("r32_native", "r32_postz"), ("raw256_native", "raw256_postz")):
        max_error = 0.0
        for predictor in PREDICTOR_IDS:
            left_rows = {
                (row["fold"], row["video_id"]): row
                for row in probe_rows
                if row["representation_id"] == left and row["predictor_id"] == predictor
            }
            right_rows = {
                (row["fold"], row["video_id"]): row
                for row in probe_rows
                if row["representation_id"] == right and row["predictor_id"] == predictor
            }
            for key in left_rows:
                for metric in PROBE_METRICS:
                    a = float(left_rows[key][metric])
                    b = float(right_rows[key][metric])
                    max_error = max(max_error, abs(a - b))
                    if not np.isclose(a, b, rtol=PROBE_INVARIANCE_RTOL, atol=PROBE_INVARIANCE_ATOL):
                        raise ValueError(f"scale-neutral probe invariance failed for {left}/{right}")
        invariant_pairs.append({"left": left, "right": right, "passed": True, "max_absolute_error": max_error})

    world_rows = cast(list[dict[str, Any]], normalized["world_rows"])
    world_summary: list[dict[str, Any]] = []
    integrity_rows = cast(list[dict[str, Any]], normalized["integrity_rows"])
    for rep in WORLD_REPRESENTATION_IDS:
        videos = _world_video_medians(world_rows, rep)
        support = sum(_world_support(row) for row in videos)
        applicable = [row for row in integrity_rows if row["representation_id"] == rep]
        failures = [
            row
            for row in applicable
            if float(row["latent_std_min"]) < INTEGRITY_MIN_STD
            or float(row["latent_effective_rank"]) < INTEGRITY_MIN_EFFECTIVE_RANK
            or not row["prediction_finite"]
            or not row["prediction_variance_positive"]
        ]
        world_summary.append(
            {
                "representation_id": rep,
                "supporting_videos": support,
                "passes": support >= core.REQUIRED_VIDEO_SUPPORT,
                "healthy": not failures,
                "failed_integrity_rows": len(failures),
                "minimum_latent_std": min(float(row["latent_std_min"]) for row in applicable),
                "minimum_effective_rank": min(float(row["latent_effective_rank"]) for row in applicable),
                "video_medians": videos,
            }
        )
    worlds = {row["representation_id"]: row for row in world_summary}

    contrasts: list[dict[str, Any]] = []
    for candidate, comparator in (
        ("r32_postz", "r32_native"),
        ("r32_qr_postz", "r32_postz"),
        ("pca32_postz", "r32_qr_postz"),
    ):
        candidate_videos = {row["video_id"]: row for row in worlds[candidate]["video_medians"]}
        comparator_videos = {row["video_id"]: row for row in worlds[comparator]["video_medians"]}
        support = 0
        paired_rows: list[dict[str, Any]] = []
        for video_id in dataset.SAMPLE_VIDEO_IDS:
            c_ratio = candidate_videos[video_id]["world_mse"] / candidate_videos[video_id]["persistence_mse"]
            b_ratio = comparator_videos[video_id]["world_mse"] / comparator_videos[video_id]["persistence_mse"]
            improved = c_ratio * 1.1 <= b_ratio
            support += int(improved)
            paired_rows.append(
                {
                    "video_id": video_id,
                    "candidate_world_ratio": c_ratio,
                    "comparator_world_ratio": b_ratio,
                    "material_improvement": improved,
                }
            )
        contrasts.append(
            {
                "candidate": candidate,
                "comparator": comparator,
                "supporting_videos": support,
                "material_improvement": support >= core.REQUIRED_VIDEO_SUPPORT,
                "paired_videos": paired_rows,
            }
        )
    contrast_map = {(row["candidate"], row["comparator"]): row for row in contrasts}

    def any_probe(rep: str) -> bool:
        return any(bool(probes[(rep, predictor)]["passes"]) for predictor in PREDICTOR_IDS)

    pca_records = [row for row in records if row["representation_id"] == "pca32_postz"]
    pca_stable = not any(row["pca_rank_below_32"] or row["pca_boundary_degenerate"] for row in pca_records)
    labels: list[str] = []
    for spec in REPRESENTATIONS:
        if (
            probes[(spec.representation_id, "residual_delta")]["passes"]
            and not probes[(spec.representation_id, "absolute_target")]["passes"]
        ):
            labels.append(f"persistence_aware_parameterization_rescue_{spec.representation_id}")
    branch_labels, primary, classification = _diagnostic_decision(
        {spec.representation_id: any_probe(spec.representation_id) for spec in REPRESENTATIONS},
        {rep: bool(worlds[rep]["passes"]) for rep in WORLD_REPRESENTATION_IDS},
        {rep: bool(worlds[rep]["healthy"]) for rep in WORLD_REPRESENTATION_IDS},
        {key: bool(row["material_improvement"]) for key, row in contrast_map.items()},
        pca_stable=pca_stable,
    )
    labels.extend(branch_labels)

    recommendations = {
        "mm001_coordinate_scale_cause_supported": "freeze train-current mean/std after the existing 32-D projection",
        "fixed_subspace_basis_conditioning_cause_supported": (
            "replace the fixed matrix by its canonical orthonormal basis and freeze score statistics"
        ),
        "fixed_random_subspace_information_loss_supported": (
            "fit PCA32 on training-current raw TAESD latents and freeze components and score statistics"
        ),
        "multiple_projection_scale_mechanisms_supported": (
            "apply the simplest supported same-information fix first, then revalidate PCA only if needed"
        ),
        "full_information_signal_present_no_compatible_32d_fix": (
            "test a wider or supervised temporal projection before changing the world model"
        ),
        "no_linear_full_taesd_signal_at_frozen_margin": (
            "stop projection tuning and test spatial/difference-aware visual objectives or more dynamic data"
        ),
        "linear_temporal_signal_present_world_path_not_supported": (
            "repair world-representation optimization before changing the visual frontend"
        ),
        "pca_subspace_world_sensitivity_only": (
            "treat PCA as sensitivity only and seek independent predictive-subspace evidence"
        ),
        "tested_projection_scale_factors_not_supported": "do not adopt any tested transform as a fix",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "MM-003",
        "scope": "outcome_informed_feature_only_diagnostic",
        "parent_parity": parity,
        "probe_invariance": invariant_pairs,
        "pca_stable": pca_stable,
        "linear_probes": probe_summary,
        "world": world_summary,
        "contrasts": contrasts,
        "diagnosis_labels": [*labels, *primary],
        "decision": {
            "classification": classification,
            "primary_cause_labels": primary,
            "recommended_fix": recommendations[classification],
        },
        "claim_boundary": (
            "MM-003 identifies proximate sensitivity on eight outcome-visible videos only; "
            "a failed linear probe does not exclude nonlinear or spatial temporal signal."
        ),
    }


def report_text(summary: Mapping[str, Any]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("experiment_id") != "MM-003":
        raise ValueError("summary is not an MM-003 result")
    decision = cast(Mapping[str, Any], summary["decision"])
    parity = cast(Mapping[str, Any], summary["parent_parity"])
    lines = [
        "# MM-003 TAESD projection/scale isolation report",
        "",
        "MM-003 is outcome-informed and does not reclassify MM-001.",
        "",
        f"Decision classification: `{decision['classification']}`.",
        "",
        f"Recommended next fix: {decision['recommended_fix']}.",
        "",
        "## Parent and transform integrity",
        "",
        f"Parent parity: **{'PASS' if parity['passed'] else 'FAIL'}** "
        f"({parity['parent_ridge_rows_compared']} ridge, "
        f"{parity['parent_world_rows_compared']} world, and "
        f"{parity['parent_integrity_rows_compared']} integrity rows).",
        f"PCA boundary/rank stability: **{'PASS' if summary['pca_stable'] else 'FAIL'}**.",
        "",
        "## Scale-neutral linear probes",
        "",
        "| Representation | Predictor | Supporting videos |",
        "|---|---|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], summary["linear_probes"]):
        lines.append(f"| `{row['representation_id']}` | `{row['predictor_id']}` | {row['supporting_videos']}/8 |")
    lines.extend(
        [
            "",
            "## End-to-end world test",
            "",
            "| Representation | Supporting videos | Integrity | Failed probes |",
            "|---|---:|---|---:|",
        ]
    )
    for row in cast(Sequence[Mapping[str, Any]], summary["world"]):
        lines.append(
            f"| `{row['representation_id']}` | {row['supporting_videos']}/8 | "
            f"{'healthy' if row['healthy'] else 'unhealthy'} | {row['failed_integrity_rows']} |"
        )
    labels = cast(Sequence[str], summary["diagnosis_labels"])
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            ", ".join(f"`{label}`" for label in labels) if labels else "No diagnosis labels.",
            "",
            str(summary["claim_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def config_record() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_dim": 256,
        "representations": [
            {
                "representation_id": spec.representation_id,
                "output_dim": spec.output_dim,
                "fitted": spec.fitted,
                "world_tested": spec.world_tested,
            }
            for spec in REPRESENTATIONS
        ],
        "predictors": list(PREDICTOR_IDS),
        "folds": [
            {"fold": fold.index, "train_ids": list(fold.train_ids), "test_ids": list(fold.test_ids)}
            for fold in dataset.formal_folds()
        ],
        "seeds": list(core.SEEDS),
        "matched_rows": MATCHED_ROWS,
        "matched_counts": MATCHED_COUNTS,
        "scale_floor": SCALE_FLOOR,
        "probe_penalty": PROBE_PENALTY,
        "probe_target": "absolute_and_residual; train-only standardized",
        "qr_projector_tolerance": QR_PROJECTOR_TOLERANCE,
        "pca_rank_ratio_min": PCA_RANK_RATIO_MIN,
        "pca_boundary_gap_min": PCA_BOUNDARY_GAP_MIN,
        "world": {
            "updates": CHECKPOINTS[-1],
            "checkpoints": list(CHECKPOINTS),
            "batch": core.WORLD_BATCH,
            "hidden": core.WORLD_HIDDEN,
            "ensemble": core.WORLD_ENSEMBLE,
            "learning_rate": core.WORLD_LR,
            "ema_tau": core.WORLD_EMA_TAU,
            "variance_weight": core.WORLD_W_VAR,
            "covariance_weight": core.WORLD_W_COV,
            "reward_weight": core.WORLD_W_REWARD,
            "inverse_weight": core.WORLD_W_INVERSE,
            "sample_seed_offset": core.WORLD_SAMPLE_SEED_OFFSET,
        },
        "thresholds": {
            "persistence_factor": core.VISUAL_PERSISTENCE_FACTOR,
            "shuffle_margin": core.VISUAL_SHUFFLE_MARGIN,
            "material_improvement_margin": 1.1,
            "required_video_support": core.REQUIRED_VIDEO_SUPPORT,
            "integrity_min_std": INTEGRITY_MIN_STD,
            "integrity_min_effective_rank": INTEGRITY_MIN_EFFECTIVE_RANK,
        },
        "parity": {
            "rtol": PARITY_RTOL,
            "atol": PARITY_ATOL,
            "alignment_rtol": ALIGNMENT_RTOL,
            "alignment_atol": ALIGNMENT_ATOL,
            "probe_invariance_rtol": PROBE_INVARIANCE_RTOL,
            "probe_invariance_atol": PROBE_INVARIANCE_ATOL,
        },
        "row_schemas": {
            "transform_records": sorted(TRANSFORM_RECORD_KEYS),
            "probe_rows": sorted(PROBE_ROW_KEYS),
            "parent_ridge_rows": sorted(PARENT_RIDGE_ROW_KEYS),
            "world_rows": sorted(WORLD_ROW_KEYS),
            "integrity_rows": sorted(INTEGRITY_ROW_KEYS),
        },
    }


__all__ = [
    "CHECKPOINTS",
    "FittedTransform",
    "REPRESENTATIONS",
    "SCHEMA_VERSION",
    "VisualTable",
    "alignment_record",
    "assert_parent_parity",
    "config_record",
    "execute",
    "fit_all_transforms",
    "fit_transform",
    "matched_table",
    "parent_preflight_record",
    "report_text",
    "summarize",
    "transformed_table",
    "validate_evidence",
    "validate_raw_table",
    "validate_transform_records",
]
