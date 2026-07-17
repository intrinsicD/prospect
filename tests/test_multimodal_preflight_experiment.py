"""Lifecycle tests for the MM-001 formal runner without neural weights."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_preflight import core, dataset, experiment


def _table(rows_per_video: int = 6) -> core.FeatureTable:
    video_ids: list[str] = []
    timestamps: list[float] = []
    vision: list[np.ndarray] = []
    audio: list[np.ndarray] = []
    text: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    angles = np.linspace(0.0, 2.0 * np.pi, core.FEATURE_DIM, endpoint=False)
    for video_index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        phase = 0.17 * video_index
        for row in range(rows_per_video):
            timestamp = 1.0 + 0.5 * row
            current = np.sin(angles + 0.13 * timestamp + phase)
            future = np.sin(angles + 0.13 * (timestamp + 1.0) + phase)
            video_ids.append(video_id)
            timestamps.append(timestamp)
            vision.append(current)
            audio.append(np.cos(angles + 0.13 * timestamp + phase))
            text.append(np.tanh(1.7 * current))
            targets.append(future)
    return core.FeatureTable(
        video_ids=np.asarray(video_ids),
        timestamps=np.asarray(timestamps),
        vision=np.asarray(vision),
        audio=np.asarray(audio),
        text=np.asarray(text),
        target_vision=np.asarray(targets),
        annotation_present=np.ones(len(video_ids), dtype=bool),
    )


def _component_rows() -> list[dict[str, object]]:
    return [
        {
            "video_id": video_id,
            "window_count": 6,
            "taesd_matched_mse": 0.1,
            "taesd_spatial_mean_mse": 0.2,
            "taesd_half_cycle_mse": 0.2,
            "taesd_shuffled_latent_mse": 0.2,
            "snac_matched_mse": 0.1,
            "snac_cross_video_mse": 0.2,
            "snac_temporally_permuted_mse": 0.2,
            "t5_correct_target_nll": 1.0,
            "t5_cross_video_target_nll": 2.0,
            "t5_deranged_context_nll": 2.0,
            "t5_generation_max_tokens": 4,
            "t5_generation_finite_rate": 1.0,
            "t5_generation_parseable_rate": 1.0,
            "t5_generation_exact_rate": 1.0,
        }
        for video_id in dataset.SAMPLE_VIDEO_IDS
    ]


def _fake_models() -> dict[str, object]:
    models: dict[str, object] = {
        name: {
            "model_id": contract["model_id"],
            "requested_revision": contract["revision"],
            "resolved_revision": contract["revision"],
            "parameter_count": contract["parameter_count"],
            "device": contract["device"],
            "dtype": contract["dtype"],
            "versions": {"fake": "1"},
            "snapshot_files": {"config.json": "b" * 64},
        }
        for name, contract in experiment.MODEL_CONTRACTS.items()
    }
    cast(dict[str, object], models["t5"])["tokenizer_ids"] = {
        "pad": experiment.T5_PAD_TOKEN_ID,
        "eos": experiment.T5_EOS_TOKEN_ID,
        "sentinel_0": experiment.T5_SENTINEL_0,
        "sentinel_1": experiment.T5_SENTINEL_1,
    }
    return models


def _fake_environment() -> dict[str, object]:
    return {
        "requested_device": "cuda",
        "ffmpeg": {"path": "/fake/ffmpeg", "sha256": "c" * 64},
        "ffprobe": {"path": "/fake/ffprobe", "sha256": "d" * 64},
        "torch": {
            "version": "fake",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "gpu": {"name": experiment.FORMAL_GPU_NAME},
        },
    }


def _fake_media_inspection() -> dict[str, object]:
    videos = {
        video_id: {
            "frame_count_2fps": 20,
            "audio_samples_24khz": 240_000,
            "file_size_bytes": len(f"fake:{video_id}".encode()),
            "ffprobe": {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]},
            "window_count": 6,
        }
        for video_id in dataset.SAMPLE_VIDEO_IDS
    }
    return {
        "video_ids": list(dataset.SAMPLE_VIDEO_IDS),
        "window_counts": {video_id: 6 for video_id in dataset.SAMPLE_VIDEO_IDS},
        "videos": videos,
        "environment": {"requested_device": "media-only"},
        "dataset_inference_performed": False,
    }


def _extraction() -> SimpleNamespace:
    base = _table()
    row_count = len(base.video_ids)
    projections = experiment._projection_arrays()
    rng = np.random.default_rng(17)
    taesd_latents = rng.normal(0.0, 0.1, size=(row_count, 4, 8, 8)).astype(np.float32)
    target_taesd_latents = rng.normal(0.0, 0.1, size=(row_count, 4, 8, 8)).astype(np.float32)
    snac_code_ids = rng.integers(0, 4_096, size=(row_count, 84), dtype=np.int64)
    t5_pooled_states = rng.normal(0.0, 0.1, size=(row_count, 256)).astype(np.float32)
    vision = taesd_latents.reshape(row_count, -1).astype(np.float64) @ projections["vision_projection_matrix"]
    target_vision = (
        target_taesd_latents.reshape(row_count, -1).astype(np.float64) @ projections["vision_projection_matrix"]
    )
    normalized_codes = 2.0 * snac_code_ids.astype(np.float64) / 4_095.0 - 1.0
    audio = normalized_codes @ projections["audio_projection_matrix"]
    text = t5_pooled_states.astype(np.float64) @ projections["text_projection_matrix"]
    table = core.FeatureTable(
        video_ids=base.video_ids,
        timestamps=base.timestamps,
        vision=vision,
        audio=audio,
        text=text,
        target_vision=target_vision,
        annotation_present=base.annotation_present,
    )
    cross_indices = experiment._cross_video_indices(table)
    window_rows: list[dict[str, object]] = []
    masked_inputs: list[list[int]] = []
    target_ids: list[list[int]] = []
    generated_ids: list[list[int]] = []
    for index, (video_id, timestamp) in enumerate(zip(table.video_ids, table.timestamps, strict=True)):
        key = f"MM-001|{video_id}|{float(timestamp):.1f}"
        start = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % 3
        original = [10, 11, 12]
        masked = [*original[:start], experiment.T5_SENTINEL_0, *original[start + 1 :], experiment.T5_EOS_TOKEN_ID]
        target = [
            experiment.T5_SENTINEL_0,
            original[start],
            experiment.T5_SENTINEL_1,
            experiment.T5_EOS_TOKEN_ID,
        ]
        masked_inputs.append(masked)
        target_ids.append(target)
        generated_ids.append([experiment.T5_PAD_TOKEN_ID, *target])
        window_rows.append(
            {
                "video_id": str(video_id),
                "timestamp": float(timestamp),
                "target_timestamp": float(timestamp + 1.0),
                "annotation_text": "action: fake; sound: fake.",
                "annotation_present": True,
                "taesd_matched_mse": 0.1,
                "taesd_spatial_mean_mse": 0.2,
                "taesd_half_cycle_mse": 0.2,
                "taesd_shuffled_latent_mse": 0.2,
                "snac_matched_mse": 0.1,
                "snac_cross_video_mse": 0.2,
                "snac_temporally_permuted_mse": 0.2,
                "t5_correct_target_nll": 1.0,
                "t5_cross_video_target_nll": 2.0,
                "t5_deranged_context_nll": 2.0,
                "t5_generation_length": 4,
                "t5_generation_finite": True,
                "t5_generation_parseable": True,
                "t5_generation_exact": True,
                "input_frame_index": int(round(float(timestamp) * 2.0)),
                "target_frame_index": int(round((float(timestamp) + 1.0) * 2.0)),
                "audio_start_sample": int(round((float(timestamp) - 1.0) * 24_000)),
                "audio_stop_sample": int(round(float(timestamp) * 24_000)),
                "cross_video_index": int(cross_indices[index]),
                "t5_mask_key": key,
                "t5_mask_start": start,
                "t5_mask_stop": start + 1,
            }
        )
    component_arrays = {
        "taesd_latents": taesd_latents,
        "target_taesd_latents": target_taesd_latents,
        "snac_code_ids": snac_code_ids,
        "t5_pooled_states": t5_pooled_states,
        "t5_masked_input_ids": np.asarray(masked_inputs, dtype=np.int64),
        "t5_target_ids": np.asarray(target_ids, dtype=np.int64),
        "t5_generated_ids": np.asarray(generated_ids, dtype=np.int64),
        **projections,
    }
    projection_digests = {
        name.removesuffix("_projection_matrix"): cast(dict[str, object], record)["sha256"]
        for name, record in experiment._projection_record().items()
    }
    return SimpleNamespace(
        table=table,
        component_rows=_component_rows(),
        window_rows=window_rows,
        component_arrays=component_arrays,
        metadata={
            "models": _fake_models(),
            "environment": _fake_environment(),
            "projection_seeds": {"vision": 12_001, "audio": 12_002, "text": 12_003},
            "projection_digests": projection_digests,
            "video_ids": list(dataset.SAMPLE_VIDEO_IDS),
            "window_counts": {video_id: 6 for video_id in dataset.SAMPLE_VIDEO_IDS},
            "media": {
                "frame_rate": 2,
                "frame_height": 64,
                "frame_width": 64,
                "letterboxed": True,
                "audio_sample_rate": 24_000,
                "audio_channels": 1,
            },
            "decoded_media": {
                video_id: {
                    field: value
                    for field, value in cast(dict[str, object], metadata).items()
                    if field in {"frame_count_2fps", "audio_samples_24khz", "file_size_bytes", "ffprobe"}
                }
                for video_id, metadata in cast(dict[str, object], _fake_media_inspection()["videos"]).items()
            },
            "runtime": {"seconds": 0.0},
        },
    )


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    (cache / "annotations").mkdir(parents=True)
    (cache / "videos").mkdir()
    annotations = {video_id: {"metadata": {"video_id": video_id}} for video_id in dataset.SAMPLE_VIDEO_IDS}
    (cache / "annotations" / "sample.json").write_text(json.dumps(annotations, sort_keys=True), encoding="utf-8")
    (cache / "sample_videos.zip").write_bytes(b"fake video archive")
    (cache / "sample_annotations.zip").write_bytes(b"fake annotation archive")
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        (cache / "videos" / f"{video_id}.mp4").write_bytes(f"fake:{video_id}".encode())
    return cache


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, extraction: object) -> None:
    monkeypatch.setattr(
        experiment.dataset,
        "EXPECTED_WINDOW_COUNTS",
        {video_id: 6 for video_id in dataset.SAMPLE_VIDEO_IDS},
    )
    monkeypatch.setattr(
        experiment.dataset,
        "validate_sample_cache",
        lambda cache_path: {video_id: {} for video_id in dataset.SAMPLE_VIDEO_IDS},
    )
    monkeypatch.setattr(
        experiment,
        "inspect_media_inputs",
        lambda cache_path: _fake_media_inspection(),
    )
    monkeypatch.setattr(
        experiment,
        "inspect_backend_inputs",
        lambda hf_cache, *, device: {
            "models": _fake_models(),
            "environment": _fake_environment(),
            "device": device,
            "hf_cache": str(hf_cache),
            "dataset_inference_performed": False,
        },
    )
    monkeypatch.setattr(
        experiment,
        "extract_sample",
        lambda cache_path, hf_cache, *, device: extraction,
    )


def _components(**overrides: bool) -> dict[str, bool]:
    values = {
        "taesd_image_pass": True,
        "taesd_framewise_video_pass": True,
        "snac_audio_pass": True,
        "t5_text_pass": True,
    }
    values.update(overrides)
    return {**values, "all_pass": all(values.values())}


def _integration(**overrides: bool) -> dict[str, object]:
    passes = {
        "real_visual_dynamics": True,
        "vision_codec_migration": True,
        "audio_substitution": True,
        "text_substitution": True,
    }
    passes.update(overrides)
    return {"passes": passes}


@pytest.mark.parametrize(
    ("components", "integration", "expected"),
    [
        (
            _components(taesd_image_pass=False),
            _integration(real_visual_dynamics=False),
            "vision_component_not_supported",
        ),
        (
            _components(),
            _integration(real_visual_dynamics=False, vision_codec_migration=False),
            "real_visual_temporal_prediction_not_supported",
        ),
        (
            _components(),
            _integration(vision_codec_migration=False),
            "vision_codec_migration_not_supported",
        ),
        (_components(), _integration(), "three_seam_predictive_substitution_supported"),
        (
            _components(t5_text_pass=False),
            _integration(),
            "vision_audio_predictive_substitution_only",
        ),
        (
            _components(snac_audio_pass=False),
            _integration(),
            "vision_text_predictive_substitution_only",
        ),
        (
            _components(snac_audio_pass=False, t5_text_pass=False),
            _integration(),
            "real_visual_prediction_only",
        ),
    ],
)
def test_protocol_branch_truth_table(
    components: dict[str, bool], integration: dict[str, object], expected: str
) -> None:
    assert experiment.protocol_branch(components, integration) == expected


def test_formal_start_is_atomic_and_precedes_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache(tmp_path)

    class ExtractionFailure(RuntimeError):
        pass

    _patch_inputs(monkeypatch, ExtractionFailure("unused"))

    def fail_after_start(cache_path: Path, hf_cache: Path, *, device: str) -> Any:
        del cache_path, hf_cache, device
        assert (tmp_path / "output" / experiment.STARTED_FILE).is_file()
        raise ExtractionFailure("backend failed after formal start")

    monkeypatch.setattr(experiment, "extract_sample", fail_after_start)
    output = tmp_path / "output"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    with pytest.raises(experiment.InvalidMM001Package, match="after formal start"):
        experiment.run(output, cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")

    started = output / experiment.STARTED_FILE
    assert started.is_file()
    assert started.stat().st_mode & 0o222 == 0
    with pytest.raises(experiment.InvalidMM001Package, match="invalid_MM001_package.*one-shot") as invalid:
        experiment.run(output, cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")
    assert invalid.value.classification == "invalid_MM001_package"


def test_formal_run_rejects_non_cuda_and_any_second_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(tmp_path)
    _patch_inputs(monkeypatch, _extraction())
    expected = tmp_path / "expected"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", expected)
    with pytest.raises(ValueError, match="device='cuda'"):
        experiment.run(expected, cache_path=cache, hf_cache=tmp_path / "hf", device="cpu")
    with pytest.raises(ValueError, match="single output path"):
        experiment.run(tmp_path / "other", cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")
    assert not expected.exists()


def test_fake_formal_roundtrip_and_artifact_tamper_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache(tmp_path)
    _patch_inputs(monkeypatch, _extraction())
    monkeypatch.setattr(core, "WORLD_STEPS", 1)
    monkeypatch.setattr(core, "CODEC_STEPS", 1)
    output = tmp_path / "results"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)

    result = experiment.run(output, cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")

    assert result["status"] == "completed_small_real_multimodal_preflight"
    assert experiment.verify(output)["outcomes"] == "verified_results"
    assert (output / experiment.PROTOCOL_COPY_FILE).read_bytes() == (
        experiment.REPO_ROOT / experiment.PROTOCOL_DOC
    ).read_bytes()
    assert (
        experiment.verify_semantic(output, cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")["outcomes"]
        == "verified_semantic_results"
    )
    with np.load(output / experiment.FEATURE_FILE, allow_pickle=False) as package:
        assert set(package.files) == set(experiment.FEATURE_ARRAYS)
        assert package["vision"].shape[1] == core.FEATURE_DIM
    with np.load(output / experiment.COMPONENT_AUDIT_FILE, allow_pickle=False) as package:
        assert set(package.files) == set(experiment.COMPONENT_AUDIT_ARRAYS)
    with np.load(output / experiment.PROJECTION_FILE, allow_pickle=False) as package:
        assert set(package.files) == set(experiment.PROJECTION_ARRAYS)
    pairing_path = output / experiment.PAIRING_FILE
    original_pairing_bytes = pairing_path.read_bytes()
    with np.load(pairing_path, allow_pickle=False) as package:
        assert set(package.files) == set(experiment.PAIRING_ARRAYS)
        tampered_pairings = {name: np.asarray(package[name]).copy() for name in experiment.PAIRING_ARRAYS}
    first_pairing = experiment.PAIRING_ARRAYS[0]
    tampered_pairings[first_pairing] = np.roll(tampered_pairings[first_pairing], 1)
    experiment._write_array_package(pairing_path, tampered_pairings, experiment.PAIRING_ARRAYS)
    experiment._write_artifact_manifest(output)
    with pytest.raises(ValueError, match="training-control pairings"):
        experiment.verify(output)
    pairing_path.write_bytes(original_pairing_bytes)
    experiment._write_artifact_manifest(output)

    with np.load(pairing_path, allow_pickle=False) as package:
        wrong_dtype_pairings = {
            name: np.asarray(package[name], dtype=np.int32).copy() for name in experiment.PAIRING_ARRAYS
        }
    experiment._write_array_package(pairing_path, wrong_dtype_pairings, experiment.PAIRING_ARRAYS)
    experiment._write_artifact_manifest(output)
    with pytest.raises(ValueError, match="training-pairing dtype/shape"):
        experiment.verify(output)
    pairing_path.write_bytes(original_pairing_bytes)
    experiment._write_artifact_manifest(output)

    feature_path = output / experiment.FEATURE_FILE
    original_feature_bytes = feature_path.read_bytes()
    with np.load(feature_path, allow_pickle=False) as package:
        wrong_dtype_features = {name: np.asarray(package[name]).copy() for name in experiment.FEATURE_ARRAYS}
    wrong_dtype_features["vision"] = wrong_dtype_features["vision"].astype(np.float32)
    experiment._write_array_package(feature_path, wrong_dtype_features, experiment.FEATURE_ARRAYS)
    experiment._write_artifact_manifest(output)
    with pytest.raises(ValueError, match="feature dtype/shape"):
        experiment.verify(output)
    feature_path.write_bytes(original_feature_bytes)
    experiment._write_artifact_manifest(output)

    rows_path = output / experiment.INTEGRATION_ROWS_FILE
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact manifest"):
        experiment.verify(output)


def test_fast_verify_rejects_rehashed_cross_video_schema_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _cache(tmp_path)
    _patch_inputs(monkeypatch, _extraction())
    monkeypatch.setattr(core, "WORLD_STEPS", 1)
    monkeypatch.setattr(core, "CODEC_STEPS", 1)
    output = tmp_path / "results"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)
    experiment.run(output, cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")

    rows_path = output / experiment.WINDOW_ROWS_FILE
    rows = cast(list[dict[str, object]], json.loads(rows_path.read_text(encoding="utf-8")))
    rows[0]["cross_video_index"] = 0
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    experiment._write_artifact_manifest(output)

    with pytest.raises(ValueError, match="cross-video index"):
        experiment.verify(output)


def test_zero_token_t5_generation_is_a_valid_unparseable_component_outcome() -> None:
    extraction = _extraction()
    table = cast(core.FeatureTable, extraction.table)
    arrays = cast(dict[str, np.ndarray], extraction.component_arrays)
    rows = cast(list[dict[str, object]], extraction.window_rows)
    arrays["t5_generated_ids"][0] = experiment.T5_PAD_TOKEN_ID
    rows[0]["t5_generation_length"] = 0
    rows[0]["t5_generation_finite"] = True
    rows[0]["t5_generation_parseable"] = False
    rows[0]["t5_generation_exact"] = False

    validated_rows = experiment._validate_window_rows(rows, table)
    validated_arrays = experiment._validate_extraction_arrays(arrays, len(table.video_ids))
    experiment._validate_component_array_alignment(validated_arrays, table, validated_rows)


def test_t5_ids_outside_frozen_vocabulary_are_rejected() -> None:
    extraction = _extraction()
    table = cast(core.FeatureTable, extraction.table)
    arrays = cast(dict[str, np.ndarray], extraction.component_arrays)
    arrays["t5_generated_ids"][0, 1] = experiment.T5_VOCAB_SIZE

    with pytest.raises(ValueError, match="frozen vocabulary"):
        experiment._validate_extraction_arrays(arrays, len(table.video_ids))


def test_exact_gpu_contract_is_rejected_before_formal_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache(tmp_path)
    _patch_inputs(monkeypatch, _extraction())
    backend = {
        "models": _fake_models(),
        "environment": deepcopy(_fake_environment()),
        "device": "cuda",
        "hf_cache": str(tmp_path / "hf"),
        "dataset_inference_performed": False,
    }
    cast(dict[str, object], cast(dict[str, object], backend["environment"])["torch"])["gpu"] = {
        "name": "NVIDIA GeForce RTX 4090"
    }
    monkeypatch.setattr(experiment, "inspect_backend_inputs", lambda hf_cache, *, device: backend)
    output = tmp_path / "results"
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", output)

    with pytest.raises(ValueError, match="exact GPU"):
        experiment.run(output, cache_path=cache, hf_cache=tmp_path / "hf", device="cuda")
    assert not output.exists()
