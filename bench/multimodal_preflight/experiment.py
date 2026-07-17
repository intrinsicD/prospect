"""Formal MM-001 orchestration, packaging, and fast verification.

The optional neural/media stack is imported only through the small backend
wrappers below.  Importing this module therefore remains safe in the numpy-only
test environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np

from . import core, dataset

SCHEMA_VERSION = "mm001-formal-v1"
EXPERIMENT_ID = "MM-001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/multimodal_preflight/results/MM-001")
EXPECTED_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT
DEFAULT_HF_CACHE = Path("/tmp/prospect_component_smoke/hf")
DEFAULT_DEVICE = "cuda"
PROTOCOL_DOC = Path("docs/research/2026-07-15-mm001-small-real-multimodal-preflight-protocol.md")

INPUT_MANIFEST_FILE = Path("input-manifest.json")
STARTED_FILE = Path("formal-start.json")
PROTOCOL_COPY_FILE = Path("MM-001-protocol.md")
FEATURE_FILE = Path("MM-001-features.npz")
COMPONENT_AUDIT_FILE = Path("MM-001-component-audit.npz")
PROJECTION_FILE = Path("MM-001-projections.npz")
PAIRING_FILE = Path("MM-001-training-pairings.npz")
COMPONENT_ROWS_FILE = Path("MM-001-component-rows.json")
WINDOW_ROWS_FILE = Path("MM-001-window-rows.json")
EXTRACTION_METADATA_FILE = Path("MM-001-extraction-metadata.json")
INTEGRATION_ROWS_FILE = Path("MM-001-integration-rows.json")
RESULT_FILE = Path("MM-001-results.json")
REPORT_FILE = Path("MM-001-report.md")
ARTIFACT_MANIFEST_FILE = Path("artifact-manifest.json")

ARTIFACT_FILES = (
    INPUT_MANIFEST_FILE,
    STARTED_FILE,
    PROTOCOL_COPY_FILE,
    FEATURE_FILE,
    COMPONENT_AUDIT_FILE,
    PROJECTION_FILE,
    PAIRING_FILE,
    COMPONENT_ROWS_FILE,
    WINDOW_ROWS_FILE,
    EXTRACTION_METADATA_FILE,
    INTEGRATION_ROWS_FILE,
    RESULT_FILE,
    REPORT_FILE,
)
PACKAGE_FILES = (*ARTIFACT_FILES, ARTIFACT_MANIFEST_FILE)
FEATURE_ARRAYS = (
    "video_ids",
    "timestamps",
    "vision",
    "audio",
    "text",
    "target_vision",
    "annotation_present",
)
WINDOW_ROW_KEYS = {
    "video_id",
    "timestamp",
    "target_timestamp",
    "annotation_text",
    "annotation_present",
    "taesd_matched_mse",
    "taesd_spatial_mean_mse",
    "taesd_half_cycle_mse",
    "taesd_shuffled_latent_mse",
    "snac_matched_mse",
    "snac_cross_video_mse",
    "snac_temporally_permuted_mse",
    "t5_correct_target_nll",
    "t5_cross_video_target_nll",
    "t5_deranged_context_nll",
    "t5_generation_length",
    "t5_generation_finite",
    "t5_generation_parseable",
    "t5_generation_exact",
    "input_frame_index",
    "target_frame_index",
    "audio_start_sample",
    "audio_stop_sample",
    "cross_video_index",
    "t5_mask_key",
    "t5_mask_start",
    "t5_mask_stop",
}
WINDOW_FLOAT_METRICS = (
    "taesd_matched_mse",
    "taesd_spatial_mean_mse",
    "taesd_half_cycle_mse",
    "taesd_shuffled_latent_mse",
    "snac_matched_mse",
    "snac_cross_video_mse",
    "snac_temporally_permuted_mse",
    "t5_correct_target_nll",
    "t5_cross_video_target_nll",
    "t5_deranged_context_nll",
)
WINDOW_BOOL_METRICS = (
    "annotation_present",
    "t5_generation_finite",
    "t5_generation_parseable",
    "t5_generation_exact",
)
COMPONENT_BACKEND_ARRAYS = (
    "taesd_latents",
    "target_taesd_latents",
    "snac_code_ids",
    "t5_pooled_states",
    "t5_masked_input_ids",
    "t5_target_ids",
    "t5_generated_ids",
)
PROJECTION_ARRAYS = (
    "vision_projection_matrix",
    "audio_projection_matrix",
    "text_projection_matrix",
)
PAIRING_ARRAYS = tuple(
    name
    for fold_index in range(4)
    for name in (
        f"fold_{fold_index}_temporal_shuffle_indices",
        f"fold_{fold_index}_cross_video_indices",
    )
)
EXTRACTION_ARRAYS = (*COMPONENT_BACKEND_ARRAYS, *PROJECTION_ARRAYS)
COMPONENT_AUDIT_ARRAYS = COMPONENT_BACKEND_ARRAYS
PROJECTION_SPECS = {
    "vision_projection_matrix": {"input_dim": 256, "seed": 12_001},
    "audio_projection_matrix": {"input_dim": 84, "seed": 12_002},
    "text_projection_matrix": {"input_dim": 256, "seed": 12_003},
}
T5_PAD_TOKEN_ID = 0
T5_EOS_TOKEN_ID = 1
T5_SENTINEL_0 = 32_099
T5_SENTINEL_1 = 32_098
T5_VOCAB_SIZE = 32_128
T5_INPUT_MAX_TOKENS = 96
T5_GENERATION_MAX_TOKENS = 32
T5_MASK_FRACTION = 0.15
SNAC_DECODE_SEED = 12_004
SNAC_FRONTEND_BATCH = 8
T5_FRONTEND_BATCH = 16
REPEAT_RTOL = 1e-6
REPEAT_ATOL = 1e-6
MODEL_CONTRACTS: dict[str, dict[str, object]] = {
    "taesd": {
        "model_id": "madebyollin/taesd",
        "revision": "614f76814bbe30edbe2e627ace1c2234c81a2c0e",
        "parameter_count": 2_445_063,
        "device": "cuda",
        "dtype": "torch.float16",
    },
    "snac": {
        "model_id": "hubertsiuzdak/snac_24khz",
        "revision": "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec",
        "parameter_count": 19_842_914,
        "device": "cuda",
        "dtype": "torch.float16",
    },
    "t5": {
        "model_id": "google/t5-efficient-tiny",
        "revision": "3441d7e8bf3f89841f366d39452b95200416e4a9",
        "parameter_count": 15_570_688,
        "device": "cpu",
        "dtype": "torch.float32",
    },
}
FORMAL_GPU_NAME = "NVIDIA GeForce RTX 3050"
COMPONENT_ROW_KEYS = {
    "video_id",
    "window_count",
    "taesd_matched_mse",
    "taesd_spatial_mean_mse",
    "taesd_half_cycle_mse",
    "taesd_shuffled_latent_mse",
    "snac_matched_mse",
    "snac_cross_video_mse",
    "snac_temporally_permuted_mse",
    "t5_correct_target_nll",
    "t5_cross_video_target_nll",
    "t5_deranged_context_nll",
    "t5_generation_max_tokens",
    "t5_generation_finite_rate",
    "t5_generation_parseable_rate",
    "t5_generation_exact_rate",
}
INTEGRATION_METRICS = {
    "incumbent_mse",
    "persistence_mse",
    "ridge_mse",
    "shuffle_model_mse",
    "shuffle_model_persistence_mse",
    "annotation_coverage",
    "vision_mse",
    "vision_latent_mse",
    "vision_feature_decode_mse",
    "vision_agent_wiring",
    "audio_mse",
    "audio_latent_mse",
    "audio_feature_decode_mse",
    "audio_agent_wiring",
    "audio_deranged_mse",
    "audio_deranged_latent_mse",
    "audio_constant_mse",
    "text_mse",
    "text_latent_mse",
    "text_feature_decode_mse",
    "text_agent_wiring",
    "text_deranged_mse",
    "text_deranged_latent_mse",
    "text_constant_mse",
    "actual_nll",
    "temporal_deranged_nll",
    "prediction_finite",
}
INTEGRATION_ROW_KEYS = {
    "video_id",
    "fold",
    "seed",
    "model_fingerprint",
    *INTEGRATION_METRICS,
}

_P = ParamSpec("_P")
_T = TypeVar("_T")


class InvalidMM001Package(ValueError):
    """Named fail-closed classification for any public MM-001 integrity defect."""

    classification = "invalid_MM001_package"


def _integrity_boundary(function: Callable[_P, _T]) -> Callable[_P, _T]:
    """Expose one automation-stable invalid-package classification."""

    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return function(*args, **kwargs)
        except InvalidMM001Package:
            raise
        except Exception as error:
            raise InvalidMM001Package(f"invalid_MM001_package: {error}") from error

    return guarded


def inspect_backend_inputs(hf_cache: Path, *, device: str) -> dict[str, Any]:
    """Call the heavyweight backend lazily so core CI needs no optional deps."""

    backend = importlib.import_module("bench.multimodal_preflight.backends")
    function = cast(Any, backend.inspect_backend_inputs)
    return cast(dict[str, Any], function(hf_cache, device=device))


def inspect_media_inputs(cache_path: Path) -> dict[str, Any]:
    """Validate exact decoded media grids without running a neural encoder."""

    backend = importlib.import_module("bench.multimodal_preflight.backends")
    function = cast(Any, backend.inspect_media_inputs)
    return cast(dict[str, Any], function(cache_path))


def extract_sample(cache_path: Path, hf_cache: Path, *, device: str) -> Any:
    """Run frozen feature extraction through the lazily imported backend."""

    backend = importlib.import_module("bench.multimodal_preflight.backends")
    function = cast(Any, backend.extract_sample)
    return function(cache_path, hf_cache, device=device)


def component_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the pure component decision without loading model weights."""

    backend = importlib.import_module("bench.multimodal_preflight.backends")
    function = cast(Any, backend.component_decision)
    return cast(dict[str, Any], function(rows))


def _json_value(value: object) -> object:
    """Convert backend/numpy values to strict finite JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite values are forbidden in MM-001 JSON")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("MM-001 JSON object keys must be strings")
            output[key] = _json_value(item)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported MM-001 JSON value {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fsync_file_and_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _file_hash(path: Path) -> str:
    return dataset.sha256_file(path)


def _source_paths() -> tuple[Path, ...]:
    paths = {
        Path("Makefile"),
        Path("pyproject.toml"),
        PROTOCOL_DOC,
        *(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "src/prospect").rglob("*.py")),
        *(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "bench/multimodal_preflight").glob("*.py")),
        *(path.relative_to(REPO_ROOT) for path in (REPO_ROOT / "tests").glob("test_multimodal_preflight*.py")),
    }
    return tuple(sorted(paths, key=str))


def _source_hashes() -> dict[str, str]:
    return {str(path): _file_hash(REPO_ROOT / path) for path in _source_paths()}


def _dependency_versions() -> dict[str, object]:
    packages = ("prospect", "numpy", "torch", "transformers", "diffusers", "snac")
    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _projection_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, spec in PROJECTION_SPECS.items():
        input_dim = int(spec["input_dim"])
        seed = int(spec["seed"])
        signs = np.random.default_rng(seed).integers(0, 2, size=(input_dim, core.FEATURE_DIM), dtype=np.int8)
        arrays[name] = (2.0 * signs.astype(np.float64) - 1.0) / np.sqrt(input_dim)
    return arrays


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return sha256(array.tobytes(order="C")).hexdigest()


def _projection_record() -> dict[str, object]:
    arrays = _projection_arrays()
    return {
        name: {
            **PROJECTION_SPECS[name],
            "output_dim": core.FEATURE_DIM,
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "sha256": _array_sha256(arrays[name]),
        }
        for name in sorted(arrays)
    }


def _fold_record() -> list[dict[str, object]]:
    return [
        {
            "index": fold.index,
            "train_ids": list(fold.train_ids),
            "test_ids": list(fold.test_ids),
        }
        for fold in dataset.formal_folds()
    ]


def _canonical_identity_table() -> core.FeatureTable:
    """Build the frozen row identity grid without using any model outcome."""

    video_ids: list[str] = []
    timestamps: list[float] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        for index in range(int(dataset.EXPECTED_WINDOW_COUNTS[video_id])):
            video_ids.append(video_id)
            timestamps.append(dataset.AUDIO_HISTORY_SECONDS + index * dataset.TIMESTAMP_STEP_SECONDS)
    row_count = len(video_ids)
    zeros = np.zeros((row_count, core.FEATURE_DIM), dtype=np.float64)
    return core.FeatureTable(
        video_ids=np.asarray(video_ids, dtype=str),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        vision=zeros.copy(),
        audio=zeros.copy(),
        text=zeros.copy(),
        target_vision=zeros.copy(),
        annotation_present=np.zeros(row_count, dtype=np.bool_),
    )


def _pairing_arrays(table: core.FeatureTable | None = None) -> dict[str, np.ndarray]:
    """Reconstruct every training control from canonical video/time identities."""

    canonical = _canonical_identity_table()
    source = canonical if table is None else table
    actual_identity = list(
        zip(
            np.asarray(source.video_ids, dtype=str).tolist(),
            np.asarray(source.timestamps, dtype=np.float64).tolist(),
            strict=True,
        )
    )
    canonical_identity = list(
        zip(
            np.asarray(canonical.video_ids, dtype=str).tolist(),
            np.asarray(canonical.timestamps, dtype=np.float64).tolist(),
            strict=True,
        )
    )
    if actual_identity != canonical_identity:
        raise ValueError("training control identities differ from the frozen row grid")
    arrays: dict[str, np.ndarray] = {}
    for fold in dataset.formal_folds():
        train = source.subset(list(fold.train_ids))
        arrays[f"fold_{fold.index}_temporal_shuffle_indices"] = np.asarray(
            core.temporal_derangement(train), dtype=np.int64
        )
        arrays[f"fold_{fold.index}_cross_video_indices"] = np.asarray(
            core.cross_video_derangement(train), dtype=np.int64
        )
    if tuple(arrays) != PAIRING_ARRAYS:
        raise AssertionError("training pairing array order drifted")
    return arrays


def _integer_array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<i8", order="C")
    return sha256(array.tobytes(order="C")).hexdigest()


def _pairing_record() -> dict[str, object]:
    arrays = _pairing_arrays()
    return {
        name: {
            "dtype": arrays[name].dtype.str,
            "shape": list(arrays[name].shape),
            "sha256": _integer_array_sha256(arrays[name]),
            "index_space": f"fold_{name.split('_')[1]}_training_rows_in_canonical_order",
        }
        for name in PAIRING_ARRAYS
    }


def _schema_record() -> dict[str, object]:
    row_count = sum(int(value) for value in dataset.EXPECTED_WINDOW_COUNTS.values())
    canonical_features = _feature_arrays(_canonical_identity_table())
    feature_schemas = _array_metadata(canonical_features)
    component_schemas: dict[str, object] = {
        "taesd_latents": {"dtype": "<f4", "shape": [row_count, 4, 8, 8]},
        "target_taesd_latents": {"dtype": "<f4", "shape": [row_count, 4, 8, 8]},
        "snac_code_ids": {"dtype": "<i8", "shape": [row_count, 84]},
        "t5_pooled_states": {"dtype": "<f4", "shape": [row_count, 256]},
        "t5_masked_input_ids": {
            "dtype": "<i8",
            "shape_prefix": [row_count],
            "width_range_inclusive": [2, 83],
            "token_id_range_inclusive": [0, T5_VOCAB_SIZE - 1],
        },
        "t5_target_ids": {
            "dtype": "<i8",
            "shape_prefix": [row_count],
            "width_range_inclusive": [4, 18],
            "token_id_range_or_padding": [0, T5_VOCAB_SIZE - 1, -100],
        },
        "t5_generated_ids": {
            "dtype": "<i8",
            "shape_prefix": [row_count],
            "width_range_inclusive": [1, 33],
            "token_id_range_inclusive": [0, T5_VOCAB_SIZE - 1],
        },
    }
    projection_schemas = {
        name: {"dtype": array.dtype.str, "shape": list(array.shape)} for name, array in _projection_arrays().items()
    }
    pairing_schemas = {
        name: {"dtype": array.dtype.str, "shape": list(array.shape)} for name, array in _pairing_arrays().items()
    }
    return {
        "feature_arrays": feature_schemas,
        "component_audit_arrays": component_schemas,
        "projection_arrays": projection_schemas,
        "training_pairing_arrays": pairing_schemas,
        "window_rows": {"count": row_count, "keys": sorted(WINDOW_ROW_KEYS)},
        "component_rows": {"count": core.FORMAL_VIDEO_COUNT, "keys": sorted(COMPONENT_ROW_KEYS)},
        "integration_rows": {
            "count": len(_expected_integration_identities()),
            "keys": sorted(INTEGRATION_ROW_KEYS),
        },
    }


def _config_record(device: str) -> dict[str, object]:
    return {
        "output_path": str(EXPECTED_OUTPUT.resolve()),
        "device": device,
        "feature_dim": core.FEATURE_DIM,
        "latent_dim": core.LATENT_DIM,
        "seeds": list(core.SEEDS),
        "world_steps": core.WORLD_STEPS,
        "world_batch": core.WORLD_BATCH,
        "codec_steps": core.CODEC_STEPS,
        "codec_batch": core.CODEC_BATCH,
        "world_model": {
            "obs_dim": core.FEATURE_DIM,
            "action_dim": 1,
            "latent_dim": core.LATENT_DIM,
            "hidden": core.WORLD_HIDDEN,
            "ensemble": core.WORLD_ENSEMBLE,
            "learning_rate": core.WORLD_LR,
            "ema_tau": core.WORLD_EMA_TAU,
            "w_reward": core.WORLD_W_REWARD,
            "w_inverse": core.WORLD_W_INVERSE,
            "w_var": core.WORLD_W_VAR,
            "w_cov": core.WORLD_W_COV,
            "updates": core.WORLD_STEPS,
            "batch_size": core.WORLD_BATCH,
            "sampling": "with_replacement",
            "sampling_seed": "seed + 1",
            "null_action": [0.0],
        },
        "codec": {
            "modality_dims": {modality.value: core.FEATURE_DIM for modality in core.MODALITIES},
            "latent_dim": core.LATENT_DIM,
            "token_dim": core.CODEC_TOKEN_DIM,
            "hidden": core.CODEC_HIDDEN,
            "learning_rate": core.CODEC_LR,
            "initialization_seed": "seed + 1",
            "cycles": core.CODEC_STEPS,
            "batch_size": core.CODEC_BATCH,
            "sampling": "with_replacement",
            "sampling_seed": "seed + 303",
            "full_fold_initialization_order": [modality.value for modality in core.MODALITIES],
        },
        "ridge": {"penalty": core.RIDGE_PENALTY, "penalize_intercept": False},
        "controls": {
            "temporal_shuffle": "within_video_circular_floor_half",
            "audio_text_derangement": "next_sorted_video_nearest_normalized_progress_python_round",
            "content_free": "training_mean_incumbent_latent",
            "taesd_half_cycle": "floor(n_frames / 2)",
            "snac_half_cycle": "floor(level_length / 2)",
        },
        "thresholds": {
            "required_videos": core.REQUIRED_VIDEO_SUPPORT,
            "total_videos": core.FORMAL_VIDEO_COUNT,
            "visual_persistence_factor": core.VISUAL_PERSISTENCE_FACTOR,
            "visual_ridge_strict": True,
            "visual_shuffle_margin": core.VISUAL_SHUFFLE_MARGIN,
            "vision_codec_factor": core.VISION_CODEC_FACTOR,
            "audio_text_substitution_margin": core.SUBSTITUTION_MARGIN,
            "component_relative_checks_strict": True,
        },
        "frontend": {
            "frame_rate": 2,
            "frame_size": 64,
            "audio_sample_rate": 24_000,
            "audio_samples_per_window": 24_000,
            "snac_batch_size": SNAC_FRONTEND_BATCH,
            "snac_codebook_size": 4_096,
            "snac_level_lengths": [12, 24, 48],
            "t5_batch_size": T5_FRONTEND_BATCH,
            "t5_input_max_tokens": T5_INPUT_MAX_TOKENS,
            "t5_generation_max_new_tokens": T5_GENERATION_MAX_TOKENS,
            "t5_mask_fraction": T5_MASK_FRACTION,
            "t5_vocab_size": T5_VOCAB_SIZE,
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
            "snac_decode_seed": SNAC_DECODE_SEED,
            "snac_seed_schedule": "SNAC_DECODE_SEED + chunk_start",
            "repeat_rtol": REPEAT_RTOL,
            "repeat_atol": REPEAT_ATOL,
            "token_ids_exact": True,
        },
        "timestamp_step_seconds": dataset.TIMESTAMP_STEP_SECONDS,
        "audio_history_seconds": dataset.AUDIO_HISTORY_SECONDS,
        "visual_target_horizon_seconds": dataset.VISUAL_TARGET_HORIZON_SECONDS,
        "development_smoke_video": dataset.DEVELOPMENT_VIDEO_ID,
        "development_smoke_enters_formal_decision": False,
        "formal_video_ids": list(dataset.SAMPLE_VIDEO_IDS),
        "expected_window_counts": dict(dataset.EXPECTED_WINDOW_COUNTS),
        "expected_artifacts": [str(path) for path in PACKAGE_FILES],
        "schemas": _schema_record(),
        "projections": _projection_record(),
        "training_pairings": _pairing_record(),
    }


def _dataset_record(cache_path: Path, media_inspection: Mapping[str, object]) -> dict[str, object]:
    annotation_path = cache_path / "annotations" / "sample.json"
    media = {video_id: _file_hash(cache_path / "videos" / f"{video_id}.mp4") for video_id in dataset.SAMPLE_VIDEO_IDS}
    return {
        "cache_path": str(cache_path.resolve()),
        "video_ids": list(dataset.SAMPLE_VIDEO_IDS),
        "sample_videos_url": dataset.OFFICIAL_SAMPLE_VIDEOS_URL,
        "sample_videos_archive_sha256": dataset.OFFICIAL_SAMPLE_VIDEOS_SHA256,
        "sample_videos_archive_bytes": (cache_path / "sample_videos.zip").stat().st_size,
        "sample_annotations_url": dataset.OFFICIAL_SAMPLE_ANNOTATIONS_URL,
        "sample_annotations_archive_sha256": dataset.OFFICIAL_SAMPLE_ANNOTATIONS_SHA256,
        "sample_annotations_archive_bytes": (cache_path / "sample_annotations.zip").stat().st_size,
        "annotation_file_sha256": _file_hash(annotation_path),
        "annotation_file_bytes": annotation_path.stat().st_size,
        "media_sha256": media,
        "media_inspection": _json_value(media_inspection),
    }


def _input_manifest(
    cache_path: Path,
    hf_cache: Path,
    device: str,
    backend_inputs: Mapping[str, object],
    media_inspection: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "validated_before_formal_start",
        "protocol": {
            "path": str(PROTOCOL_DOC),
            "copy_path": str(PROTOCOL_COPY_FILE),
            "sha256": _file_hash(REPO_ROOT / PROTOCOL_DOC),
        },
        "source": _source_hashes(),
        "dataset": _dataset_record(cache_path, media_inspection),
        "models": _json_value(backend_inputs),
        "dependencies": _dependency_versions(),
        "folds": _fold_record(),
        "config": {
            **_config_record(device),
            "hf_cache": str(hf_cache.resolve()),
        },
    }


def _stable_model_identities(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("backend model metadata must be an object")
    models_value = value.get("models")
    if not isinstance(models_value, Mapping) or set(models_value) != {"taesd", "snac", "t5"}:
        raise ValueError("backend metadata must identify exactly TAESD, SNAC, and T5")
    fields = (
        "model_id",
        "requested_revision",
        "resolved_revision",
        "parameter_count",
        "device",
        "dtype",
        "versions",
        "snapshot_files",
    )
    identities: dict[str, object] = {}
    for name in ("taesd", "snac", "t5"):
        model_value = models_value[name]
        if not isinstance(model_value, Mapping):
            raise ValueError(f"{name} model metadata must be an object")
        if any(field not in model_value for field in fields):
            raise ValueError(f"{name} model metadata is missing stable identity fields")
        snapshot_files = model_value["snapshot_files"]
        if not isinstance(snapshot_files, Mapping) or not snapshot_files:
            raise ValueError(f"{name} model snapshot manifest must be non-empty")
        identities[name] = {field: _json_value(model_value[field]) for field in fields}
        if name == "t5":
            tokenizer_ids = model_value.get("tokenizer_ids")
            expected_ids = {
                "pad": T5_PAD_TOKEN_ID,
                "eos": T5_EOS_TOKEN_ID,
                "sentinel_0": T5_SENTINEL_0,
                "sentinel_1": T5_SENTINEL_1,
            }
            if tokenizer_ids != expected_ids:
                raise ValueError("T5 tokenizer IDs do not match the frozen native-sentinel contract")
            cast(dict[str, object], identities[name])["tokenizer_ids"] = expected_ids
    return identities


def _validate_formal_backend_inputs(value: object) -> dict[str, object]:
    """Fail closed unless the inspected models and runtime match MM-001 exactly."""

    identities = _stable_model_identities(value)
    for name, contract in MODEL_CONTRACTS.items():
        identity = identities[name]
        if not isinstance(identity, Mapping):
            raise ValueError(f"{name} stable identity is invalid")
        expected = {
            "model_id": contract["model_id"],
            "requested_revision": contract["revision"],
            "resolved_revision": contract["revision"],
            "parameter_count": contract["parameter_count"],
            "device": contract["device"],
            "dtype": contract["dtype"],
        }
        if any(identity.get(field) != expected_value for field, expected_value in expected.items()):
            raise ValueError(f"{name} does not match the frozen MM-001 model contract")

    if not isinstance(value, Mapping):
        raise ValueError("backend input record must be an object")
    environment = value.get("environment")
    if not isinstance(environment, Mapping) or environment.get("requested_device") != DEFAULT_DEVICE:
        raise ValueError("formal runtime must report requested_device='cuda'")
    torch_value = environment.get("torch")
    if not isinstance(torch_value, Mapping):
        raise ValueError("formal runtime must include torch/CUDA identity")
    required_torch = {
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if any(torch_value.get(field) != expected for field, expected in required_torch.items()):
        raise ValueError("formal torch runtime does not match the deterministic MM-001 contract")
    gpu = torch_value.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("name") != FORMAL_GPU_NAME:
        raise ValueError(f"formal MM-001 requires exact GPU {FORMAL_GPU_NAME!r}")
    return identities


def _validate_extraction_metadata(metadata: object, manifest: Mapping[str, object]) -> dict[str, Any]:
    normalized = _json_value(metadata)
    if not isinstance(normalized, dict):
        raise ValueError("extraction metadata must be a JSON object")
    inspected = manifest.get("models")
    if _stable_model_identities(normalized) != _stable_model_identities(inspected):
        raise ValueError("extraction model identity differs from pre-start inspection")
    if not isinstance(inspected, Mapping) or normalized.get("environment") != inspected.get("environment"):
        raise ValueError("extraction runtime identity differs from pre-start inspection")
    projection_digests = normalized.get("projection_digests")
    expected_digests = {
        name.removesuffix("_projection_matrix"): cast(dict[str, object], record)["sha256"]
        for name, record in _projection_record().items()
    }
    if projection_digests != expected_digests:
        raise ValueError("extraction projection digests do not match the frozen matrices")
    if normalized.get("projection_seeds") != {
        "vision": 12_001,
        "audio": 12_002,
        "text": 12_003,
    }:
        raise ValueError("extraction projection seeds do not match MM-001")
    if normalized.get("video_ids") != list(dataset.SAMPLE_VIDEO_IDS) or normalized.get("window_counts") != dict(
        dataset.EXPECTED_WINDOW_COUNTS
    ):
        raise ValueError("extraction video IDs or window counts do not match the frozen sample")
    if normalized.get("media") != {
        "frame_rate": 2,
        "frame_height": 64,
        "frame_width": 64,
        "letterboxed": True,
        "audio_sample_rate": 24_000,
        "audio_channels": 1,
    }:
        raise ValueError("extraction media transform metadata does not match MM-001")
    dataset_manifest = manifest.get("dataset")
    if not isinstance(dataset_manifest, Mapping):
        raise ValueError("formal dataset manifest is missing")
    media_inspection = dataset_manifest.get("media_inspection")
    decoded_media = normalized.get("decoded_media")
    if not isinstance(media_inspection, Mapping) or not isinstance(decoded_media, Mapping):
        raise ValueError("pre-start or extraction decoded-media metadata is missing")
    inspected_videos = media_inspection.get("videos")
    if not isinstance(inspected_videos, Mapping) or set(decoded_media) != set(dataset.SAMPLE_VIDEO_IDS):
        raise ValueError("decoded-media video identities do not match")
    stable_media_fields = ("frame_count_2fps", "audio_samples_24khz", "file_size_bytes", "ffprobe")
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        before = inspected_videos.get(video_id)
        after = decoded_media.get(video_id)
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"decoded-media metadata is invalid for {video_id}")
        if any(before.get(field) != after.get(field) for field in stable_media_fields):
            raise ValueError(f"decoded-media identity differs from pre-start inspection for {video_id}")
    return cast(dict[str, Any], normalized)


def _formal_start_record(manifest: Mapping[str, object], manifest_sha256: str) -> dict[str, object]:
    bindings = {
        name: _canonical_json_sha256(manifest[name])
        for name in ("protocol", "source", "dataset", "models", "dependencies", "folds", "config")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "input_manifest_sha256": manifest_sha256,
        "bindings": bindings,
    }


def _mark_formal_started(output: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    """Atomically and durably consume the MM-001 experiment identifier."""

    record = _formal_start_record(manifest, _file_hash(output / INPUT_MANIFEST_FILE))
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path = output / STARTED_FILE
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return record


def _assert_safe_new_output(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("MM-001 output must be a real directory path")
    resolved = output.resolve()
    repository = REPO_ROOT.resolve()
    owned = (REPO_ROOT / "bench/multimodal_preflight/results").resolve()
    if resolved == repository or resolved in repository.parents:
        raise ValueError("MM-001 output cannot be the repository root or an ancestor")
    if repository in resolved.parents and resolved != owned and owned not in resolved.parents:
        raise ValueError("in-repository output must be below bench/multimodal_preflight/results")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("MM-001 output must be absent or empty; formal execution is one-shot")


def _assert_expected_output(output: Path) -> None:
    if output.resolve() != EXPECTED_OUTPUT.resolve():
        raise ValueError(f"MM-001 is bound to the single output path {EXPECTED_OUTPUT}")


def _require_formal_device(device: str) -> None:
    if device != DEFAULT_DEVICE:
        raise ValueError("formal MM-001 execution is frozen to device='cuda'")


def _feature_arrays(table: core.FeatureTable) -> dict[str, np.ndarray]:
    arrays = {
        "video_ids": np.asarray(table.video_ids, dtype=str),
        "timestamps": np.asarray(table.timestamps, dtype=np.float64),
        "vision": np.asarray(table.vision, dtype=np.float64),
        "audio": np.asarray(table.audio, dtype=np.float64),
        "text": np.asarray(table.text, dtype=np.float64),
        "target_vision": np.asarray(table.target_vision, dtype=np.float64),
        "annotation_present": np.asarray(table.annotation_present, dtype=np.bool_),
    }
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise TypeError("object arrays are forbidden in the MM-001 feature package")
    return arrays


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {name: {"dtype": value.dtype.str, "shape": list(value.shape)} for name, value in sorted(arrays.items())}


def _write_array_package(
    path: Path, arrays: Mapping[str, np.ndarray], expected_names: Sequence[str]
) -> dict[str, object]:
    if set(arrays) != set(expected_names):
        raise ValueError(f"{path.name} array schema does not match MM-001")
    values = {name: np.asarray(arrays[name]) for name in expected_names}
    if any(value.dtype.kind == "O" for value in values.values()):
        raise TypeError(f"object arrays are forbidden in {path.name}")
    cast(Any, np.savez_compressed)(path, **values)
    return _array_metadata(values)


def _load_array_package(path: Path, expected_names: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        if tuple(sorted(package.files)) != tuple(sorted(expected_names)):
            raise ValueError(f"{path.name} has unexpected array names")
        return {name: np.asarray(package[name]).copy() for name in expected_names}


def _validate_extraction_arrays(value: object, row_count: int) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(EXTRACTION_ARRAYS):
        raise ValueError("ExtractionResult.component_arrays must contain the ten frozen arrays")
    arrays = {name: np.asarray(value[name]) for name in EXTRACTION_ARRAYS}
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise TypeError("component/projection object arrays are forbidden")
    expected_fixed = {
        "taesd_latents": (np.dtype(np.float32), (row_count, 4, 8, 8)),
        "target_taesd_latents": (np.dtype(np.float32), (row_count, 4, 8, 8)),
        "snac_code_ids": (np.dtype(np.int64), (row_count, 84)),
        "t5_pooled_states": (np.dtype(np.float32), (row_count, 256)),
    }
    for name, (dtype, shape) in expected_fixed.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    for name in ("taesd_latents", "target_taesd_latents", "t5_pooled_states"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name} must be finite")
    codes = arrays["snac_code_ids"]
    if np.any(codes < 0) or np.any(codes >= 4_096):
        raise ValueError("snac_code_ids contains an out-of-codebook value")
    token_width_bounds = {
        "t5_masked_input_ids": (2, 83),
        "t5_target_ids": (4, 18),
        "t5_generated_ids": (1, 33),
    }
    for name, (minimum_width, maximum_width) in token_width_bounds.items():
        array = arrays[name]
        if array.dtype != np.dtype(np.int64) or array.ndim != 2 or array.shape[0] != row_count:
            raise ValueError(f"{name} must be an int64 matrix with one row per window")
        if array.shape[1] < minimum_width or array.shape[1] > maximum_width:
            raise ValueError(f"{name} has an invalid padded token width")
    if any(
        np.any((arrays[name] < 0) | (arrays[name] >= T5_VOCAB_SIZE))
        for name in ("t5_masked_input_ids", "t5_generated_ids")
    ):
        raise ValueError("T5 input/generated token IDs must be inside the frozen vocabulary")
    target_ids = arrays["t5_target_ids"]
    if np.any(((target_ids < 0) & (target_ids != -100)) | (target_ids >= T5_VOCAB_SIZE)):
        raise ValueError("T5 target IDs must use the frozen vocabulary or exactly -100 padding")
    expected_projections = _projection_arrays()
    for name in PROJECTION_ARRAYS:
        expected = expected_projections[name]
        if arrays[name].dtype != np.dtype(np.float64) or arrays[name].shape != expected.shape:
            raise ValueError(f"{name} dtype or shape does not match the frozen projection")
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"{name} does not reproduce from its frozen seed")
    return {name: np.asarray(arrays[name]).copy() for name in EXTRACTION_ARRAYS}


def _validate_component_array_alignment(
    arrays: Mapping[str, np.ndarray],
    table: core.FeatureTable,
    window_rows: Sequence[Mapping[str, object]],
) -> None:
    vision_projection = np.asarray(arrays["vision_projection_matrix"], dtype=np.float64)
    vision_raw = np.asarray(arrays["taesd_latents"], dtype=np.float64).reshape(len(table.video_ids), -1)
    if not np.allclose(vision_raw @ vision_projection, table.vision, rtol=1e-6, atol=1e-6):
        raise ValueError("projected TAESD audit latents do not reproduce VISION features")
    target_vision_raw = np.asarray(arrays["target_taesd_latents"], dtype=np.float64).reshape(len(table.video_ids), -1)
    if not np.allclose(
        target_vision_raw @ vision_projection,
        table.target_vision,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("projected target TAESD audit latents do not reproduce target VISION features")
    audio_projection = np.asarray(arrays["audio_projection_matrix"], dtype=np.float64)
    audio_ids = np.asarray(arrays["snac_code_ids"], dtype=np.float64)
    normalized_audio = 2.0 * audio_ids / 4_095.0 - 1.0
    if not np.allclose(normalized_audio @ audio_projection, table.audio, rtol=1e-6, atol=1e-6):
        raise ValueError("projected SNAC audit codes do not reproduce AUDIO features")
    text_projection = np.asarray(arrays["text_projection_matrix"], dtype=np.float64)
    text_raw = np.asarray(arrays["t5_pooled_states"], dtype=np.float64)
    if not np.allclose(text_raw @ text_projection, table.text, rtol=1e-6, atol=1e-6):
        raise ValueError("projected T5 audit states do not reproduce TEXT features")
    masked_inputs = np.asarray(arrays["t5_masked_input_ids"], dtype=np.int64)
    target_ids = np.asarray(arrays["t5_target_ids"], dtype=np.int64)
    generated_ids = np.asarray(arrays["t5_generated_ids"], dtype=np.int64)
    for index, row in enumerate(window_rows):
        masked = [int(token) for token in masked_inputs[index] if int(token) != T5_PAD_TOKEN_ID]
        target = [int(token) for token in target_ids[index] if int(token) != -100]
        generated = [int(token) for token in generated_ids[index] if int(token) != T5_PAD_TOKEN_ID]
        if T5_EOS_TOKEN_ID in generated:
            generated = generated[: generated.index(T5_EOS_TOKEN_ID) + 1]
        if not masked or masked[-1] != T5_EOS_TOKEN_ID or masked.count(T5_SENTINEL_0) != 1 or T5_SENTINEL_1 in masked:
            raise ValueError("retained T5 masked input does not preserve the native-sentinel schema")
        if len(target) < 4 or target[0] != T5_SENTINEL_0 or target[-2:] != [T5_SENTINEL_1, T5_EOS_TOKEN_ID]:
            raise ValueError("retained T5 target does not preserve the native-sentinel schema")
        start = int(cast(int, row["t5_mask_start"]))
        stop = int(cast(int, row["t5_mask_stop"]))
        span_length = len(target) - 3
        original_length = len(masked) + span_length - 2
        expected_span_length = max(1, int(math.ceil(T5_MASK_FRACTION * original_length)))
        choices = original_length - expected_span_length + 1
        key = str(row["t5_mask_key"])
        expected_start = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "little") % choices
        if (
            stop - start != span_length
            or span_length != expected_span_length
            or start != expected_start
            or masked.index(T5_SENTINEL_0) != start
        ):
            raise ValueError("retained T5 mask span does not reproduce from the frozen key")
        first_sentinel = generated.index(T5_SENTINEL_0) if T5_SENTINEL_0 in generated else -1
        second_sentinel = generated.index(T5_SENTINEL_1) if T5_SENTINEL_1 in generated else -1
        parseable = (
            0 <= len(generated) <= T5_GENERATION_MAX_TOKENS
            and first_sentinel == 0
            and second_sentinel > first_sentinel
            and T5_EOS_TOKEN_ID in generated
        )
        if (
            int(cast(int, row["t5_generation_length"])) != len(generated)
            or row["t5_generation_finite"] is not True
            or bool(row["t5_generation_parseable"]) is not parseable
            or bool(row["t5_generation_exact"]) is not (generated == target)
        ):
            raise ValueError("retained T5 generation diagnostics do not recompute from token IDs")


def _write_feature_table(path: Path, table: core.FeatureTable) -> dict[str, object]:
    _validate_feature_table(table)
    arrays = _feature_arrays(table)
    cast(Any, np.savez_compressed)(path, **arrays)
    return _array_metadata(arrays)


def _load_feature_table(path: Path) -> tuple[core.FeatureTable, dict[str, object]]:
    with np.load(path, allow_pickle=False) as package:
        if tuple(sorted(package.files)) != tuple(sorted(FEATURE_ARRAYS)):
            raise ValueError("MM-001 feature package has unexpected array names")
        arrays = {name: np.asarray(package[name]).copy() for name in FEATURE_ARRAYS}
    if arrays["video_ids"].dtype.kind not in ("U", "S"):
        raise ValueError("MM-001 video_ids must be a non-object string array")
    table = core.FeatureTable(
        video_ids=np.asarray(arrays["video_ids"], dtype=str),
        timestamps=np.asarray(arrays["timestamps"]),
        vision=np.asarray(arrays["vision"]),
        audio=np.asarray(arrays["audio"]),
        text=np.asarray(arrays["text"]),
        target_vision=np.asarray(arrays["target_vision"]),
        annotation_present=np.asarray(arrays["annotation_present"]),
    )
    _validate_feature_table(table)
    return table, _array_metadata(arrays)


def _validate_feature_table(table: core.FeatureTable) -> None:
    table.validate()
    count = len(table.video_ids)
    target = np.asarray(table.target_vision)
    if target.shape != (count, core.FEATURE_DIM) or not np.all(np.isfinite(target)):
        raise ValueError("target_vision must be a finite 32-D row for every window")
    annotation = np.asarray(table.annotation_present)
    if annotation.shape != (count,) or annotation.dtype.kind != "b":
        raise ValueError("annotation_present must be a one-dimensional boolean array")
    video_ids = np.asarray(table.video_ids, dtype=str)
    if set(video_ids) != set(dataset.SAMPLE_VIDEO_IDS):
        raise ValueError("formal feature table must contain exactly the eight frozen video IDs")
    expected_order = sorted(range(count), key=lambda index: (video_ids[index], table.timestamps[index]))
    if expected_order != list(range(count)):
        raise ValueError("formal feature rows must be sorted by video_id then timestamp")
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        times = np.asarray(table.timestamps)[video_ids == video_id]
        expected_count = int(dataset.EXPECTED_WINDOW_COUNTS[video_id])
        if len(times) != expected_count or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{video_id} must have exactly {expected_count} strictly ordered windows")
        if not math.isclose(float(times[0]), dataset.AUDIO_HISTORY_SECONDS, abs_tol=1e-12):
            raise ValueError(f"{video_id} does not begin on the frozen 1.0 s input anchor")
        if not np.allclose(np.diff(times), dataset.TIMESTAMP_STEP_SECONDS, atol=1e-12, rtol=0.0):
            raise ValueError(f"{video_id} timestamps do not follow the frozen 0.5 s grid")


def _validate_component_rows(rows: object) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(dataset.SAMPLE_VIDEO_IDS):
        raise ValueError("component rows must contain exactly one row per formal video")
    normalized = cast(list[dict[str, Any]], _json_value(rows))
    if [row.get("video_id") for row in normalized] != list(dataset.SAMPLE_VIDEO_IDS):
        raise ValueError("component rows must be in frozen video-ID order")
    for row in normalized:
        if set(row) != COMPONENT_ROW_KEYS:
            raise ValueError("component row schema does not match MM-001")
        count = row["window_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("component window_count must be a positive integer")
        for name in COMPONENT_ROW_KEYS - {"video_id", "window_count"}:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"component metric {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"component metric {name} must be finite")
    return normalized


def _validate_window_rows(rows: object, table: core.FeatureTable) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(table.video_ids):
        raise ValueError("window metadata must contain one row per feature row")
    normalized = cast(list[dict[str, Any]], _json_value(rows))
    expected_ids = list(np.asarray(table.video_ids, dtype=str))
    expected_times = list(np.asarray(table.timestamps, dtype=float))
    expected_annotations = list(np.asarray(table.annotation_present, dtype=bool))
    cross_indices = _cross_video_indices(table)
    for index, row in enumerate(normalized):
        if not isinstance(row, dict) or set(row) != WINDOW_ROW_KEYS:
            raise ValueError("window metadata row schema does not match MM-001")
        timestamp = expected_times[index]
        if row["video_id"] != expected_ids[index] or not math.isclose(
            float(row["timestamp"]), timestamp, abs_tol=1e-12
        ):
            raise ValueError("window metadata identity does not align with the feature table")
        if not math.isclose(
            float(row["target_timestamp"]),
            timestamp + dataset.VISUAL_TARGET_HORIZON_SECONDS,
            abs_tol=1e-12,
        ):
            raise ValueError("window target timestamp violates the frozen horizon")
        if type(row["annotation_present"]) is not bool or row["annotation_present"] is not bool(
            expected_annotations[index]
        ):
            raise ValueError("window annotation flag does not align with the feature table")
        annotation_text = row["annotation_text"]
        if (
            not isinstance(annotation_text, str)
            or not annotation_text.startswith("action: ")
            or "; sound: " not in annotation_text
            or not annotation_text.endswith(".")
        ):
            raise ValueError("window annotation text is not canonical")
        integer_fields = (
            "input_frame_index",
            "target_frame_index",
            "audio_start_sample",
            "audio_stop_sample",
            "cross_video_index",
            "t5_mask_start",
            "t5_mask_stop",
            "t5_generation_length",
        )
        if any(type(row[name]) is not int for name in integer_fields):
            raise ValueError("window index and length fields must be integers")
        if row["input_frame_index"] != int(round(timestamp * 2.0)):
            raise ValueError("window input frame index does not match the 2 fps grid")
        if row["target_frame_index"] != int(round((timestamp + dataset.VISUAL_TARGET_HORIZON_SECONDS) * 2.0)):
            raise ValueError("window target frame index does not match the 2 fps grid")
        if row["audio_start_sample"] != int(round((timestamp - dataset.AUDIO_HISTORY_SECONDS) * 24_000)) or row[
            "audio_stop_sample"
        ] != int(round(timestamp * 24_000)):
            raise ValueError("window audio indices do not match the causal 24 kHz interval")
        if row["cross_video_index"] != int(cross_indices[index]):
            raise ValueError("window cross-video index does not match nearest normalized progress")
        expected_key = f"MM-001|{expected_ids[index]}|{timestamp:.1f}"
        if row["t5_mask_key"] != expected_key:
            raise ValueError("window T5 mask key does not match the frozen deterministic key")
        if row["t5_mask_start"] < 0 or row["t5_mask_stop"] <= row["t5_mask_start"]:
            raise ValueError("window T5 mask span is invalid")
        if row["t5_generation_length"] < 0 or row["t5_generation_length"] > T5_GENERATION_MAX_TOKENS:
            raise ValueError("window T5 generation length violates the frozen 32-token limit")
        for name in WINDOW_FLOAT_METRICS:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"window metric {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"window metric {name} must be finite and non-negative")
        for name in WINDOW_BOOL_METRICS[1:]:
            if type(row[name]) is not bool:
                raise ValueError(f"window diagnostic {name} must be boolean")
    return normalized


def _cross_video_indices(table: core.FeatureTable) -> np.ndarray:
    video_ids = np.asarray(table.video_ids, dtype=str)
    groups = {video_id: np.flatnonzero(video_ids == video_id) for video_id in dataset.SAMPLE_VIDEO_IDS}
    output = np.empty(len(video_ids), dtype=np.int64)
    for position, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        source = groups[video_id]
        target = groups[dataset.SAMPLE_VIDEO_IDS[(position + 1) % len(dataset.SAMPLE_VIDEO_IDS)]]
        for rank, row_index in enumerate(source):
            progress = rank / max(len(source) - 1, 1)
            target_rank = int(round(progress * (len(target) - 1)))
            output[row_index] = target[target_rank]
    if np.any(video_ids == video_ids[output]):
        raise ValueError("cross-video indices contain a same-video pairing")
    return output


def _component_rows_from_windows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        group = [row for row in rows if row["video_id"] == video_id]
        if not group:
            raise ValueError(f"window metadata is missing {video_id}")
        aggregate: dict[str, Any] = {"video_id": video_id, "window_count": len(group)}
        for name in WINDOW_FLOAT_METRICS:
            aggregate[name] = float(np.median([float(cast(float, row[name])) for row in group]))
        aggregate["t5_generation_max_tokens"] = max(int(cast(int, row["t5_generation_length"])) for row in group)
        for name in ("t5_generation_finite", "t5_generation_parseable", "t5_generation_exact"):
            aggregate[f"{name}_rate"] = float(np.mean([bool(row[name]) for row in group]))
        output.append(aggregate)
    return _validate_component_rows(output)


def _validate_component_decision(decision: object) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise ValueError("component decision must be a JSON object")
    normalized = cast(dict[str, Any], _json_value(decision))
    names = (
        "taesd_image_pass",
        "taesd_framewise_video_pass",
        "snac_audio_pass",
        "t5_text_pass",
        "all_pass",
    )
    for name in names:
        if type(normalized.get(name)) is not bool:
            raise ValueError(f"component decision {name} must be boolean")
    expected_all = all(bool(normalized[name]) for name in names[:-1])
    if normalized["all_pass"] is not expected_all:
        raise ValueError("component all_pass does not match its four frozen checks")
    return normalized


def _expected_integration_identities() -> list[tuple[int, int, str]]:
    return [
        (fold.index, seed, video_id)
        for fold in dataset.formal_folds()
        for seed in core.SEEDS
        for video_id in fold.test_ids
    ]


def _validate_integration_rows(rows: object) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("integration rows must be a JSON array")
    normalized = cast(list[dict[str, Any]], _json_value(rows))
    expected = _expected_integration_identities()
    actual: list[tuple[int, int, str]] = []
    fingerprints: dict[tuple[int, int], str] = {}
    for row in normalized:
        if not isinstance(row, dict) or set(row) != INTEGRATION_ROW_KEYS:
            raise ValueError("integration row schema does not match MM-001")
        fold = row["fold"]
        seed = row["seed"]
        video_id = row["video_id"]
        if type(fold) is not int or type(seed) is not int or not isinstance(video_id, str):
            raise ValueError("integration row identity fields have invalid types")
        actual.append((fold, seed, video_id))
        fingerprint = row["model_fingerprint"]
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("integration model fingerprint is not lowercase SHA-256")
        key = (fold, seed)
        if key in fingerprints and fingerprints[key] != fingerprint:
            raise ValueError("held-out videos in one fold/seed must share the primary model")
        fingerprints[key] = fingerprint
        for name in INTEGRATION_METRICS:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"integration metric {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"integration metric {name} must be finite")
        if float(row["persistence_mse"]) <= 0.0:
            raise ValueError("persistence_mse must be positive for the normalized shuffle gate")
        if float(row["shuffle_model_persistence_mse"]) <= 0.0:
            raise ValueError("shuffle_model_persistence_mse must be positive for the normalized shuffle gate")
    if actual != expected:
        raise ValueError("integration rows are incomplete, duplicated, or out of frozen order")
    return normalized


def protocol_decision(components: Mapping[str, object], integration: Mapping[str, object]) -> dict[str, object]:
    """Apply the frozen component-aware branch order."""

    component = _validate_component_decision(components)
    passes_value = integration.get("passes")
    if not isinstance(passes_value, Mapping):
        raise ValueError("integration decision is missing its passes object")
    passes = cast(Mapping[str, object], passes_value)
    required = (
        "real_visual_dynamics",
        "vision_codec_migration",
        "audio_substitution",
        "text_substitution",
    )
    if any(type(passes.get(name)) is not bool for name in required):
        raise ValueError("integration decision has invalid protocol pass flags")

    vision_component = bool(component["taesd_image_pass"]) and bool(component["taesd_framewise_video_pass"])
    audio_component = bool(component["snac_audio_pass"])
    text_component = bool(component["t5_text_pass"])
    audio_supported = audio_component and bool(passes["audio_substitution"])
    text_supported = text_component and bool(passes["text_substitution"])

    if not vision_component:
        classification = "vision_component_not_supported"
    elif not bool(passes["real_visual_dynamics"]):
        classification = "real_visual_temporal_prediction_not_supported"
    elif not bool(passes["vision_codec_migration"]):
        classification = "vision_codec_migration_not_supported"
    elif audio_supported and text_supported:
        classification = "three_seam_predictive_substitution_supported"
    elif audio_supported:
        classification = "vision_audio_predictive_substitution_only"
    elif text_supported:
        classification = "vision_text_predictive_substitution_only"
    else:
        classification = "real_visual_prediction_only"
    return {
        "classification": classification,
        "component_eligibility": {
            "vision": vision_component,
            "audio": audio_component,
            "text": text_component,
        },
        "substitution_support": {
            "audio": audio_supported,
            "text": text_supported,
        },
    }


def protocol_branch(components: Mapping[str, object], integration: Mapping[str, object]) -> str:
    """Return only the frozen branch label for callers testing the truth table."""

    return cast(str, protocol_decision(components, integration)["classification"])


def _execute(table: core.FeatureTable) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in dataset.formal_folds():
        train = table.subset(fold.train_ids)
        for seed in core.SEEDS:
            model = core.fit_world_model(train, seed, steps=core.WORLD_STEPS)
            shuffled = core.fit_world_model(train, seed, shuffled=True, steps=core.WORLD_STEPS)
            codec = core.fit_codec(model, train, seed, steps=core.CODEC_STEPS)
            deranged = core.fit_codec(
                model,
                train,
                seed,
                deranged_audio_text=True,
                steps=core.CODEC_STEPS,
            )
            fingerprint = core.model_fingerprint(model)
            for video_id in fold.test_ids:
                metrics = core.evaluate_video(model, shuffled, codec, deranged, train, table.subset([video_id]))
                rows.append(
                    {
                        "video_id": video_id,
                        "fold": fold.index,
                        "seed": seed,
                        "model_fingerprint": fingerprint,
                        **metrics,
                    }
                )
    return _validate_integration_rows(rows)


def _result_record(
    output: Path,
    input_manifest: Mapping[str, object],
    formal_start: Mapping[str, object],
    array_metadata: Mapping[str, object],
    component_array_metadata: Mapping[str, object],
    projection_array_metadata: Mapping[str, object],
    pairing_array_metadata: Mapping[str, object],
    component_result: Mapping[str, object],
    integration_result: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_small_real_multimodal_preflight",
        "interpretation_scope": "eight-video engineering preflight; not independent confirmation",
        "input_manifest_sha256": _file_hash(output / INPUT_MANIFEST_FILE),
        "formal_start": formal_start,
        "formal_start_sha256": _file_hash(output / STARTED_FILE),
        "protocol_copy": {
            "path": str(PROTOCOL_COPY_FILE),
            "sha256": _file_hash(output / PROTOCOL_COPY_FILE),
        },
        "feature_package": {
            "path": str(FEATURE_FILE),
            "sha256": _file_hash(output / FEATURE_FILE),
            "arrays": array_metadata,
        },
        "component_audit_package": {
            "path": str(COMPONENT_AUDIT_FILE),
            "sha256": _file_hash(output / COMPONENT_AUDIT_FILE),
            "arrays": component_array_metadata,
        },
        "projection_package": {
            "path": str(PROJECTION_FILE),
            "sha256": _file_hash(output / PROJECTION_FILE),
            "arrays": projection_array_metadata,
        },
        "training_pairing_package": {
            "path": str(PAIRING_FILE),
            "sha256": _file_hash(output / PAIRING_FILE),
            "arrays": pairing_array_metadata,
        },
        "evidence_files": {
            str(path): _file_hash(output / path)
            for path in (
                COMPONENT_ROWS_FILE,
                WINDOW_ROWS_FILE,
                EXTRACTION_METADATA_FILE,
                INTEGRATION_ROWS_FILE,
            )
        },
        "folds": input_manifest["folds"],
        "seeds": list(core.SEEDS),
        "integration_row_count": len(_expected_integration_identities()),
        "component_decision": component_result,
        "integration_decision": integration_result,
        "decision": decision,
    }


def _report_text(results: Mapping[str, object]) -> str:
    decision = cast(Mapping[str, object], results["decision"])
    integration = cast(Mapping[str, object], results["integration_decision"])
    counts = cast(Mapping[str, object], integration["supporting_videos"])
    components = cast(Mapping[str, object], results["component_decision"])
    lines = [
        "# MM-001 small real-multimodal preflight",
        "",
        f"Decision: `{decision['classification']}`.",
        "",
        "This is an eight-video engineering preflight and not independent confirmation.",
        "The video_9253 development smoke did not contribute thresholds or smoke outcomes;",
        "its clean formal fold result is retained under the same 6/8 rule.",
        "",
        "## Component checks",
        "",
        f"- TAESD frame/image: {components['taesd_image_pass']}",
        f"- TAESD framewise-video: {components['taesd_framewise_video_pass']}",
        f"- SNAC-24k audio: {components['snac_audio_pass']}",
        f"- T5 masked-span decoder: {components['t5_text_pass']}",
        "",
        "## Integration support",
        "",
    ]
    for name in sorted(counts):
        lines.append(f"- {name}: {counts[name]}/8 videos")
    lines.extend(
        [
            "",
            "All raw visual, persistence, ridge, shuffle-model, and shuffle-persistence",
            "metrics remain in the integration evidence. The shuffle comparison is made",
            "only after normalizing each independently learned latent scale by persistence.",
            "",
            "No simultaneous fusion, control, imitation, planning, full-dataset generalization,",
            "or production capability is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifest(output: Path) -> None:
    _write_json(
        output / ARTIFACT_MANIFEST_FILE,
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "artifacts": {
                str(path): {"sha256": _file_hash(output / path), "bytes": (output / path).stat().st_size}
                for path in ARTIFACT_FILES
            },
        },
    )


@_integrity_boundary
def smoke(
    cache_path: Path = dataset.DEFAULT_CACHE_PATH,
    hf_cache: Path = DEFAULT_HF_CACHE,
    *,
    device: str = DEFAULT_DEVICE,
) -> dict[str, object]:
    """Validate formal dataset and backend inputs without computing evidence."""

    _require_formal_device(device)
    dataset.validate_sample_cache(cache_path)
    media_inspection = inspect_media_inputs(cache_path)
    inspected = inspect_backend_inputs(hf_cache, device=device)
    _validate_formal_backend_inputs(inspected)
    return {
        "status": "inputs_valid",
        "experiment_id": EXPERIMENT_ID,
        "dataset": _dataset_record(cache_path, media_inspection),
        "backend": _json_value(inspected),
        "formal_inference_performed": False,
    }


@_integrity_boundary
def run(
    output: Path = DEFAULT_OUTPUT,
    *,
    cache_path: Path = dataset.DEFAULT_CACHE_PATH,
    hf_cache: Path = DEFAULT_HF_CACHE,
    device: str = DEFAULT_DEVICE,
) -> dict[str, object]:
    """Execute all four folds and three seeds once, without outcome stopping."""

    _require_formal_device(device)
    _assert_expected_output(output)
    _assert_safe_new_output(output)
    dataset.validate_sample_cache(cache_path)
    media_inspection = inspect_media_inputs(cache_path)
    inspected = inspect_backend_inputs(hf_cache, device=device)
    _validate_formal_backend_inputs(inspected)
    manifest = _input_manifest(cache_path, hf_cache, device, inspected, media_inspection)
    output.mkdir(parents=True, exist_ok=True)
    (output / PROTOCOL_COPY_FILE).write_bytes((REPO_ROOT / PROTOCOL_DOC).read_bytes())
    projection_arrays = _projection_arrays()
    projection_array_metadata = _write_array_package(output / PROJECTION_FILE, projection_arrays, PROJECTION_ARRAYS)
    pairing_arrays = _pairing_arrays()
    pairing_array_metadata = _write_array_package(output / PAIRING_FILE, pairing_arrays, PAIRING_ARRAYS)
    _write_json(output / INPUT_MANIFEST_FILE, manifest)
    for path in (PROTOCOL_COPY_FILE, PROJECTION_FILE, PAIRING_FILE, INPUT_MANIFEST_FILE):
        _fsync_file_and_directory(output / path)
    formal_start = _mark_formal_started(output, manifest)

    extraction = extract_sample(cache_path, hf_cache, device=device)
    table = cast(core.FeatureTable, extraction.table)
    _validate_feature_table(table)
    component_rows = _validate_component_rows(extraction.component_rows)
    window_rows = _validate_window_rows(extraction.window_rows, table)
    recomputed_component_rows = _component_rows_from_windows(window_rows)
    if component_rows != recomputed_component_rows:
        raise ValueError("backend component rows do not recompute from window evidence")
    extraction_arrays = _validate_extraction_arrays(extraction.component_arrays, len(table.video_ids))
    _validate_component_array_alignment(extraction_arrays, table, window_rows)
    metadata_value = _validate_extraction_metadata(extraction.metadata, manifest)

    array_metadata = _write_feature_table(output / FEATURE_FILE, table)
    component_array_metadata = _write_array_package(
        output / COMPONENT_AUDIT_FILE,
        {name: extraction_arrays[name] for name in COMPONENT_AUDIT_ARRAYS},
        COMPONENT_AUDIT_ARRAYS,
    )
    _write_json(output / COMPONENT_ROWS_FILE, recomputed_component_rows)
    _write_json(output / WINDOW_ROWS_FILE, window_rows)
    _write_json(output / EXTRACTION_METADATA_FILE, metadata_value)

    integration_rows = _execute(table)
    _write_json(output / INTEGRATION_ROWS_FILE, integration_rows)
    component_result = _validate_component_decision(component_decision(recomputed_component_rows))
    integration_result = cast(dict[str, Any], _json_value(core.integration_decision(integration_rows)))
    decision = protocol_decision(component_result, integration_result)
    results = _result_record(
        output,
        manifest,
        formal_start,
        array_metadata,
        component_array_metadata,
        projection_array_metadata,
        pairing_array_metadata,
        component_result,
        integration_result,
        decision,
    )
    _write_json(output / RESULT_FILE, results)
    (output / REPORT_FILE).write_text(_report_text(results), encoding="utf-8")
    _write_artifact_manifest(output)
    verify(output)
    return results


@_integrity_boundary
def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Fast-verify hashes, schemas, saved evidence, decisions, branch, and report."""

    _assert_expected_output(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("MM-001 output must be a real completed directory")
    entries = list(output.iterdir())
    actual_files = {path.name for path in entries}
    if actual_files != {str(path) for path in PACKAGE_FILES} or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("MM-001 package is partial, contains extras, or uses symlinks")

    artifact_manifest = _read_json(output / ARTIFACT_MANIFEST_FILE)
    expected_artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": {
            str(path): {"sha256": _file_hash(output / path), "bytes": (output / path).stat().st_size}
            for path in ARTIFACT_FILES
        },
    }
    if artifact_manifest != expected_artifact_manifest:
        raise ValueError("MM-001 artifact manifest or artifact hash does not match")

    manifest_value = _read_json(output / INPUT_MANIFEST_FILE)
    if not isinstance(manifest_value, dict):
        raise ValueError("MM-001 input manifest is not an object")
    manifest = cast(dict[str, object], manifest_value)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("MM-001 input manifest identity does not match")
    _validate_formal_backend_inputs(manifest.get("models"))
    protocol = cast(Mapping[str, object], manifest.get("protocol"))
    if protocol.get("sha256") != _file_hash(REPO_ROOT / PROTOCOL_DOC):
        raise ValueError("MM-001 frozen protocol source has drifted")
    if (output / PROTOCOL_COPY_FILE).read_bytes() != (REPO_ROOT / PROTOCOL_DOC).read_bytes():
        raise ValueError("MM-001 canonical protocol copy differs from the frozen source")
    if manifest.get("source") != _source_hashes():
        raise ValueError("MM-001 source tree has drifted from formal start")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("MM-001 input manifest config is invalid")
    expected_config = _config_record(DEFAULT_DEVICE)
    if set(config) != {*expected_config, "hf_cache"} or any(
        config.get(name) != value for name, value in expected_config.items()
    ):
        raise ValueError("MM-001 frozen configuration drifted")
    if manifest.get("folds") != _fold_record():
        raise ValueError("MM-001 frozen folds drifted")

    formal_start_value = _read_json(output / STARTED_FILE)
    if not isinstance(formal_start_value, dict):
        raise ValueError("MM-001 formal-start record is not an object")
    formal_start = cast(dict[str, object], formal_start_value)
    expected_start = _formal_start_record(manifest, _file_hash(output / INPUT_MANIFEST_FILE))
    if formal_start != expected_start:
        raise ValueError("MM-001 formal-start bindings do not match")

    table, array_metadata = _load_feature_table(output / FEATURE_FILE)
    schemas = config.get("schemas")
    if not isinstance(schemas, Mapping) or array_metadata != schemas.get("feature_arrays"):
        raise ValueError("MM-001 feature dtype/shape metadata differs from the bound schema")
    pairing_arrays = _load_array_package(output / PAIRING_FILE, PAIRING_ARRAYS)
    expected_pairings = _pairing_arrays(table)
    if any(not np.array_equal(pairing_arrays[name], expected_pairings[name]) for name in PAIRING_ARRAYS):
        raise ValueError("MM-001 training-control pairings do not reconstruct from canonical row identities")
    pairing_array_metadata = _array_metadata(pairing_arrays)
    if pairing_array_metadata != schemas.get("training_pairing_arrays"):
        raise ValueError("MM-001 training-pairing dtype/shape metadata differs from the bound schema")
    window_rows = _validate_window_rows(_read_json(output / WINDOW_ROWS_FILE), table)
    component_rows = _validate_component_rows(_read_json(output / COMPONENT_ROWS_FILE))
    recomputed_component_rows = _component_rows_from_windows(window_rows)
    if component_rows != recomputed_component_rows:
        raise ValueError("MM-001 component rows do not recompute from window evidence")
    component_arrays = _load_array_package(output / COMPONENT_AUDIT_FILE, COMPONENT_AUDIT_ARRAYS)
    projection_arrays = _load_array_package(output / PROJECTION_FILE, PROJECTION_ARRAYS)
    combined_arrays = _validate_extraction_arrays({**component_arrays, **projection_arrays}, len(table.video_ids))
    _validate_component_array_alignment(combined_arrays, table, window_rows)
    component_array_metadata = _array_metadata(component_arrays)
    projection_array_metadata = _array_metadata(projection_arrays)
    _validate_extraction_metadata(_read_json(output / EXTRACTION_METADATA_FILE), manifest)
    integration_rows = _validate_integration_rows(_read_json(output / INTEGRATION_ROWS_FILE))
    component_result = _validate_component_decision(component_decision(component_rows))
    integration_result = cast(dict[str, Any], _json_value(core.integration_decision(integration_rows)))
    decision = protocol_decision(component_result, integration_result)

    expected_results = _result_record(
        output,
        manifest,
        formal_start,
        array_metadata,
        component_array_metadata,
        projection_array_metadata,
        pairing_array_metadata,
        component_result,
        integration_result,
        decision,
    )
    saved_results = _read_json(output / RESULT_FILE)
    if saved_results != expected_results:
        raise ValueError("MM-001 saved result does not recompute from raw evidence")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != _report_text(expected_results):
        raise ValueError("MM-001 report is not canonical")
    return {
        "status": "verified",
        "outcomes": "verified_results",
        "classification": decision["classification"],
        "artifact_count": len(ARTIFACT_FILES),
    }


def _assert_array_mapping_close(
    saved: Mapping[str, np.ndarray], regenerated: Mapping[str, np.ndarray], *, label: str
) -> None:
    if set(saved) != set(regenerated):
        raise ValueError(f"semantic {label} array names differ")
    for name in saved:
        left = np.asarray(saved[name])
        right = np.asarray(regenerated[name])
        if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
            raise ValueError(f"semantic {label} array schema differs for {name}")
        if left.dtype.kind == "f":
            matches = np.allclose(left, right, rtol=1e-6, atol=1e-6)
        else:
            matches = np.array_equal(left, right)
        if not matches:
            raise ValueError(f"semantic {label} array differs for {name}")


def _assert_rows_close(
    saved: Sequence[Mapping[str, object]],
    regenerated: Sequence[Mapping[str, object]],
    *,
    label: str,
    ignored_fields: frozenset[str] = frozenset(),
) -> None:
    if len(saved) != len(regenerated):
        raise ValueError(f"semantic {label} row count differs")
    for saved_row, regenerated_row in zip(saved, regenerated, strict=True):
        if set(saved_row) != set(regenerated_row):
            raise ValueError(f"semantic {label} row schema differs")
        for name in set(saved_row) - ignored_fields:
            left = saved_row[name]
            right = regenerated_row[name]
            if isinstance(left, float) or isinstance(right, float):
                if not math.isclose(float(cast(float, left)), float(cast(float, right)), rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"semantic {label} differs in {name}")
            elif left != right:
                raise ValueError(f"semantic {label} differs in {name}")


@_integrity_boundary
def verify_semantic(
    output: Path = DEFAULT_OUTPUT,
    *,
    cache_path: Path = dataset.DEFAULT_CACHE_PATH,
    hf_cache: Path = DEFAULT_HF_CACHE,
    device: str = DEFAULT_DEVICE,
) -> dict[str, object]:
    """Regenerate frozen features and retrain all folds in memory, without package writes."""

    _require_formal_device(device)
    verification = verify(output)
    dataset.validate_sample_cache(cache_path)
    manifest = cast(dict[str, object], _read_json(output / INPUT_MANIFEST_FILE))
    media_inspection = inspect_media_inputs(cache_path)
    dataset_manifest = manifest.get("dataset")
    if not isinstance(dataset_manifest, Mapping) or dataset_manifest.get("media_inspection") != _json_value(
        media_inspection
    ):
        raise ValueError("semantic verification media identity differs from formal start")
    inspected = inspect_backend_inputs(hf_cache, device=device)
    _validate_formal_backend_inputs(inspected)
    if _stable_model_identities(inspected) != _stable_model_identities(manifest["models"]):
        raise ValueError("semantic verification model identity differs from formal start")
    extraction = extract_sample(cache_path, hf_cache, device=device)
    regenerated_table = cast(core.FeatureTable, extraction.table)
    _validate_feature_table(regenerated_table)
    regenerated_windows = _validate_window_rows(extraction.window_rows, regenerated_table)
    regenerated_components = _component_rows_from_windows(regenerated_windows)
    backend_components = _validate_component_rows(extraction.component_rows)
    if regenerated_components != backend_components:
        raise ValueError("semantic component rows do not recompute from regenerated windows")
    regenerated_arrays = _validate_extraction_arrays(extraction.component_arrays, len(regenerated_table.video_ids))
    _validate_component_array_alignment(regenerated_arrays, regenerated_table, regenerated_windows)
    _validate_extraction_metadata(extraction.metadata, manifest)

    saved_table, _ = _load_feature_table(output / FEATURE_FILE)
    _assert_array_mapping_close(_feature_arrays(saved_table), _feature_arrays(regenerated_table), label="feature")
    saved_windows = _validate_window_rows(_read_json(output / WINDOW_ROWS_FILE), saved_table)
    _assert_rows_close(saved_windows, regenerated_windows, label="window")
    saved_component_arrays = _load_array_package(output / COMPONENT_AUDIT_FILE, COMPONENT_AUDIT_ARRAYS)
    _assert_array_mapping_close(
        saved_component_arrays,
        {name: regenerated_arrays[name] for name in COMPONENT_AUDIT_ARRAYS},
        label="component audit",
    )

    saved_integration = _validate_integration_rows(_read_json(output / INTEGRATION_ROWS_FILE))
    regenerated_integration = _execute(regenerated_table)
    _assert_rows_close(
        saved_integration,
        regenerated_integration,
        label="integration",
    )
    component_result = _validate_component_decision(component_decision(regenerated_components))
    integration_result = cast(dict[str, Any], _json_value(core.integration_decision(regenerated_integration)))
    decision = protocol_decision(component_result, integration_result)
    saved_results = cast(dict[str, object], _read_json(output / RESULT_FILE))
    if (
        saved_results["component_decision"] != component_result
        or saved_results["integration_decision"] != integration_result
        or saved_results["decision"] != decision
    ):
        raise ValueError("semantic decisions differ from the formal package")
    return {
        **verification,
        "outcomes": "verified_semantic_results",
        "semantic_regeneration": "features, components, and all 4x3 fits reproduced in memory",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MM-001 small real-multimodal preflight")
    parser.add_argument("command", choices=("run", "verify", "verify-semantic", "smoke"))
    parser.add_argument("--cache", type=Path, default=dataset.DEFAULT_CACHE_PATH)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.output, cache_path=args.cache, hf_cache=args.hf_cache, device=args.device)
        print(f"run: {cast(Mapping[str, object], result['decision'])['classification']}")
    elif args.command == "verify":
        result = verify(args.output)
        print(f"verify: {result['classification']}")
    elif args.command == "verify-semantic":
        result = verify_semantic(
            args.output,
            cache_path=args.cache,
            hf_cache=args.hf_cache,
            device=args.device,
        )
        print(f"verify-semantic: {result['classification']}")
    else:
        result = smoke(args.cache, args.hf_cache, device=args.device)
        print(f"smoke: {result['status']}")
    return 0


__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_HF_CACHE",
    "DEFAULT_OUTPUT",
    "EXPERIMENT_ID",
    "InvalidMM001Package",
    "main",
    "protocol_branch",
    "protocol_decision",
    "run",
    "smoke",
    "verify",
    "verify_semantic",
]
