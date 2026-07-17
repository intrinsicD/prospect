"""No-download tests for the MM-001 media and frozen-backend seams."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from bench.multimodal_preflight import backends, core, dataset


def test_cuda_determinism_binds_the_required_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeCudnn:
        benchmark = True
        deterministic = False

    class FakeBackends:
        cudnn = FakeCudnn()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()
        deterministic = False

        @classmethod
        def use_deterministic_algorithms(cls, enabled: bool) -> None:
            cls.deterministic = enabled

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    backends._configure_torch(FakeTorch, "cuda")
    assert backends.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert FakeTorch.deterministic
    assert not FakeTorch.backends.cudnn.benchmark
    assert FakeTorch.backends.cudnn.deterministic


def test_runtime_metadata_binds_media_and_torch_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVersion:
        cuda = "12.8"

    class FakeCudnn:
        @staticmethod
        def version() -> int:
            return 90_100

    class FakeBackends:
        cudnn = FakeCudnn()

    class FakeTorch:
        __version__ = "2.9.0"
        version = FakeVersion()
        backends = FakeBackends()

        @staticmethod
        def are_deterministic_algorithms_enabled() -> bool:
            return True

    monkeypatch.setattr(
        backends,
        "binary_identity",
        lambda executable, version_argument="-version": {
            "path": f"/resolved/{Path(executable).name}",
            "version": f"{Path(executable).name} 1.2.3",
            "size_bytes": 123,
            "sha256": "a" * 64,
        },
    )
    metadata = backends.runtime_environment("cpu", torch=FakeTorch())
    assert metadata["ffmpeg"] == {
        "path": "/resolved/ffmpeg",
        "version": "ffmpeg 1.2.3",
        "size_bytes": 123,
        "sha256": "a" * 64,
    }
    assert metadata["ffprobe"] == {
        "path": "/resolved/ffprobe",
        "version": "ffprobe 1.2.3",
        "size_bytes": 123,
        "sha256": "a" * 64,
    }
    assert metadata["torch"]["cuda_runtime"] == "12.8"  # type: ignore[index]
    assert metadata["torch"]["cudnn_version"] == 90_100  # type: ignore[index]


def test_ffmpeg_helpers_decode_letterboxed_rgb_and_mono_float(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rgb = np.arange(2 * 64 * 64 * 3, dtype=np.uint8).reshape(2, 64, 64, 3)
    audio = np.linspace(-0.5, 0.5, 24_000, dtype="<f4")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool, capture_output: bool) -> subprocess.CompletedProcess[bytes]:
        assert check and capture_output
        calls.append(command)
        payload = rgb.tobytes() if "rawvideo" in command else audio.tobytes()
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    frames = backends.decode_video_frames(tmp_path / "sample.mp4")
    waveform = backends.decode_audio_mono_24k(tmp_path / "sample.mp4")

    assert frames.shape == (2, 64, 64, 3)
    assert frames.dtype == np.float32
    assert np.all((0.0 <= frames) & (frames <= 1.0))
    assert waveform.shape == (24_000,)
    assert waveform.dtype == np.float32
    assert "fps=2" in calls[0][calls[0].index("-vf") + 1]
    assert "flags=bicubic" in calls[0][calls[0].index("-vf") + 1]
    assert "pad=64:64" in calls[0][calls[0].index("-vf") + 1]
    assert calls[1][calls[1].index("-ac") + 1] == "1"
    assert calls[1][calls[1].index("-ar") + 1] == "24000"


def test_snac_code_normalization_and_length_crop_are_explicit() -> None:
    levels = [
        np.zeros((2, 12), dtype=np.int64),
        np.full((2, 24), 2_048, dtype=np.int64),
        np.full((2, 48), 4_095, dtype=np.int64),
    ]
    normalized = backends.normalize_snac_codes(levels)
    assert normalized.shape == (2, 84)
    assert np.all(normalized[:, :12] == -1.0)
    assert np.allclose(normalized[:, 12:36], 2.0 * 2_048 / 4_095 - 1.0)
    assert np.all(normalized[:, 36:] == 1.0)

    decoded = np.arange(2 * 24_576, dtype=np.float32).reshape(2, 24_576)
    assert backends.crop_or_pad_audio(decoded, 24_000).shape == (2, 24_000)
    short = backends.crop_or_pad_audio(decoded[:, :12], 16)
    assert short.shape == (2, 16)
    assert np.all(short[:, 12:] == 0.0)


def test_media_indices_fail_closed_and_taesd_post_map_does_not_clip() -> None:
    assert backends.frame_index_at(1.5, 4) == 3
    with pytest.raises(ValueError, match="2-fps grid"):
        backends.frame_index_at(1.25, 10)
    with pytest.raises(ValueError, match="unavailable"):
        backends.frame_index_at(2.0, 4)

    assert backends.audio_sample_bounds(0.5, 1.5, 36_000) == (12_000, 36_000)
    with pytest.raises(ValueError, match="24-kHz window"):
        backends.audio_sample_bounds(0.1, 1.10001, 48_000)
    with pytest.raises(ValueError, match="unavailable"):
        backends.audio_sample_bounds(1.0, 2.0, 47_999)

    mapped = backends.taesd_post_map(np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32))
    assert mapped.tolist() == [-0.5, 0.0, 1.0, 1.5]


def test_projection_matrices_match_core_and_have_frozen_digests() -> None:
    fixtures = {
        "vision": (256, 12_001, "ccf87240a00a4a41c45b6c6b9da54faa9857392e7e86e180f05b2c7e283a681d"),
        "audio": (84, 12_002, "0c78770a48ca723b6ad8af9db1d495bc008d1dd7582e608e0efa4fc710ef8286"),
        "text": (256, 12_003, "4b2d568468f70dcb4282dcc25ef4e8609d400dfe7409eb12121a6f5974a0cf70"),
    }
    for input_dim, seed, digest in fixtures.values():
        matrix = backends.projection_matrix(input_dim, seed)
        values = np.arange(2 * input_dim, dtype=float).reshape(2, input_dim)
        assert np.array_equal(values @ matrix, core.fixed_projection(values, core.FEATURE_DIM, seed))
        assert backends.projection_digest(matrix) == digest


def test_snac_warmup_consumes_encode_and_seeded_decode_once_per_batch_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    encode_shapes: list[tuple[int, ...]] = []
    decode_calls: list[tuple[int, int]] = []

    class FakeModel:
        def encode(self, values: np.ndarray) -> list[np.ndarray]:
            encode_shapes.append(values.shape)
            return [np.zeros((len(values), length), dtype=np.int64) for length in (12, 24, 48)]

    class FakeTorch:
        float32 = np.float32

        @staticmethod
        def zeros(shape: tuple[int, ...], *, device: str, dtype: object) -> np.ndarray:
            assert device == "cpu" and dtype is np.float32
            return np.zeros(shape, dtype=np.float32)

        @staticmethod
        def inference_mode() -> nullcontext[None]:
            return nullcontext()

    backend = backends.SNACBackend(
        FakeModel(),
        FakeTorch(),
        device="cpu",
        dtype=np.float32,
        cache_dir=tmp_path,
        load_seconds=0.0,
        snac_version="fake",
    )

    def fake_decode(codes: Sequence[np.ndarray], original_samples: int, *, seed: int) -> np.ndarray:
        decode_calls.append((len(codes[0]), seed))
        return np.zeros((len(codes[0]), original_samples), dtype=np.float32)

    monkeypatch.setattr(
        backend,
        "_decode",
        fake_decode,
    )

    backend._warm_encoder(8, 24_000)
    backend._warm_encoder(8, 24_000)
    backend._warm_encoder(3, 24_000)
    assert encode_shapes == [(8, 1, 24_000), (3, 1, 24_000)]
    assert decode_calls == [(8, backends.SNAC_DECODE_SEED), (3, backends.SNAC_DECODE_SEED)]


def test_masked_span_uses_only_frozen_sha256_key_and_native_sentinels() -> None:
    tokens = [10, 11, 12, 13, 14, 15, 16]
    key = "MM-001|video_10993|1.0"
    span = backends.build_masked_span(
        tokens,
        sentinel_0=32_099,
        sentinel_1=32_098,
        eos_token_id=1,
        key=key,
    )
    expected_length = max(1, int(np.ceil(0.15 * len(tokens))))
    expected_start = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (
        len(tokens) - expected_length + 1
    )

    assert (span.start, span.stop) == (expected_start, expected_start + expected_length)
    assert span.input_ids.count(32_099) == 1
    assert span.input_ids[-1] == 1
    assert span.target_ids[0] == 32_099
    assert span.target_ids[-2:] == (32_098, 1)
    assert span == backends.build_masked_span(
        tokens,
        sentinel_0=32_099,
        sentinel_1=32_098,
        eos_token_id=1,
        key=key,
    )

    long_span = backends.build_masked_span(
        list(range(60)),
        sentinel_0=32_099,
        sentinel_1=32_098,
        eos_token_id=1,
        key="MM-001|video_10993|1.5",
    )
    assert long_span.stop - long_span.start == 9


def test_teacher_forced_nll_and_generation_diagnostics_are_per_example() -> None:
    logits = np.zeros((2, 3, 5), dtype=float)
    logits[0, 0, 2] = 4.0
    logits[0, 1, 3] = 4.0
    labels = np.array([[2, 3, -100], [1, 1, 1]], dtype=np.int64)
    nll = backends.per_example_teacher_forced_nll(logits, labels)
    assert nll.shape == (2,)
    assert nll[0] < nll[1]

    generated = np.array([[0, 9, 4, 8, 1, 0], [0, 9, 4, 1, 0, 0]], dtype=np.int64)
    targets = np.array([[9, 4, 8, 1], [9, 4, 8, 1]], dtype=np.int64)
    lengths, finite, parseable, exact = backends.generation_diagnostics(
        generated,
        targets,
        pad_token_id=0,
        eos_token_id=1,
        sentinel_0=9,
        sentinel_1=8,
    )
    assert lengths.tolist() == [4, 3]
    assert finite.tolist() == [True, True]
    assert parseable.tolist() == [True, False]
    assert exact.tolist() == [True, False]


def test_cross_video_control_uses_next_video_nearest_progress() -> None:
    ids = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    times = [1.0, 1.5, 2.0, 1.0, 2.0, 1.0, 1.5, 2.0, 2.5]
    indices = backends.cross_video_nearest_progress_indices(ids, times)
    values = np.asarray(ids)
    assert np.all(values != values[indices])
    assert indices[:3].tolist() == [3, 3, 4]
    assert indices[3:5].tolist() == [5, 8]
    assert indices[5:].tolist() == [0, 1, 1, 2]


def _component_rows(support: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        wins = index < support
        matched = 1.0 if wins else 3.0
        control = 2.0
        rows.append(
            {
                "video_id": video_id,
                "taesd_matched_mse": matched,
                "taesd_spatial_mean_mse": control,
                "taesd_half_cycle_mse": control,
                "snac_matched_mse": matched,
                "snac_cross_video_mse": control,
                "t5_correct_target_nll": matched,
                "t5_cross_video_target_nll": control,
                "t5_generation_finite_rate": 1.0,
                "t5_generation_parseable_rate": 1.0,
                "t5_generation_max_tokens": 12,
            }
        )
    return rows


def test_component_decision_requires_six_of_eight_videos() -> None:
    passing = backends.component_decision(_component_rows(6))
    assert passing["all_pass"] is True
    assert passing["taesd_image_supporting_videos"] == 6
    assert passing["taesd_framewise_video_supporting_videos"] == 6
    assert passing["snac_audio_supporting_videos"] == 6
    assert passing["t5_text_supporting_videos"] == 6

    failing = backends.component_decision(_component_rows(5))
    assert failing["all_pass"] is False
    assert failing["taesd_image_pass"] is False
    assert failing["snac_audio_pass"] is False
    assert failing["t5_text_pass"] is False


class _FakeVision:
    def embed(self, frames: np.ndarray) -> np.ndarray:
        means = np.mean(frames, axis=(1, 2, 3))
        return np.repeat(means[:, None], core.FEATURE_DIM, axis=1)

    def embed_with_latents(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.embed(frames), np.zeros((len(frames), 4, 8, 8), dtype=np.float32)

    def analyze(self, frames: np.ndarray) -> backends.VisionBatch:
        count = len(frames)
        features = self.embed(frames)
        return backends.VisionBatch(
            features=features,
            latents=np.zeros((count, 4, 8, 8), dtype=np.float32),
            reconstruction=frames.copy(),
            matched_mse=np.arange(count, dtype=float) + 0.1,
            spatial_mean_mse=np.arange(count, dtype=float) + 1.0,
            half_cycle_mse=np.arange(count, dtype=float) + 2.0,
            shuffled_latent_mse=np.arange(count, dtype=float) + 2.0,
        )

    def model_metadata(self) -> dict[str, object]:
        return {"model_id": backends.TAESD_MODEL_ID, "fake": True}


class _FakeAudio:
    def analyze(self, waveforms: np.ndarray) -> backends.AudioBatch:
        features = np.repeat(np.mean(waveforms, axis=1)[:, None], core.FEATURE_DIM, axis=1)
        reconstruction = waveforms * 0.5
        return backends.AudioBatch(
            features=features,
            code_ids=np.tile(np.arange(84, dtype=np.int64), (len(waveforms), 1)),
            reconstruction=reconstruction,
            matched_mse=np.mean(np.square(waveforms - reconstruction), axis=1),
            temporally_permuted_reconstruction=-reconstruction,
            temporally_permuted_mse=np.mean(np.square(waveforms + reconstruction), axis=1),
        )

    def model_metadata(self) -> dict[str, object]:
        return {"model_id": backends.SNAC_MODEL_ID, "fake": True}


class _FakeText:
    def analyze(self, texts: Sequence[str], keys: Sequence[str], cross_video_indices: np.ndarray) -> backends.TextBatch:
        assert all(key.startswith("MM-001|") for key in keys)
        count = len(texts)
        features = np.repeat(np.arange(count, dtype=float)[:, None], core.FEATURE_DIM, axis=1)
        return backends.TextBatch(
            features=features,
            pooled_states=np.zeros((count, 256), dtype=np.float32),
            masked_input_ids=np.tile(np.array([9, 3, 1]), (count, 1)),
            target_ids=np.tile(np.array([9, 4, 8, 1]), (count, 1)),
            generated_ids=np.tile(np.array([0, 9, 4, 8, 1]), (count, 1)),
            mask_start=np.zeros(count, dtype=np.int64),
            mask_stop=np.ones(count, dtype=np.int64),
            correct_target_nll=np.full(count, 1.0),
            cross_video_target_nll=np.full(count, 2.0),
            deranged_context_nll=np.full(count, 3.0),
            generation_length=np.full(count, 4),
            generation_finite=np.ones(count, dtype=bool),
            generation_parseable=np.ones(count, dtype=bool),
            generation_exact=np.zeros(count, dtype=bool),
        )

    def model_metadata(self) -> dict[str, object]:
        return {"model_id": backends.T5_MODEL_ID, "fake": True}


def test_extract_with_fake_backends_builds_table_and_median_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video_ids = ("video_a", "video_b")
    specs = [
        dataset.WindowSpec(
            video_id=video_id,
            audio_start_seconds=timestamp - 1.0,
            audio_end_seconds=timestamp,
            frame_seconds=timestamp,
            target_seconds=timestamp + 1.0,
            duration_seconds=3.5,
            annotation_text=f"action: event-{video_id}; sound: none.",
        )
        for video_id in video_ids
        for timestamp in (1.0, 1.5, 2.0)
    ]
    frames = np.linspace(0.0, 1.0, 8 * 64 * 64 * 3, dtype=np.float32).reshape(8, 64, 64, 3)
    audio = np.linspace(-1.0, 1.0, 4 * 24_000, dtype=np.float32)
    monkeypatch.setattr(backends, "decode_video_frames", lambda path, ffmpeg="ffmpeg": frames)
    monkeypatch.setattr(backends, "decode_audio_mono_24k", lambda path, ffmpeg="ffmpeg": audio)
    monkeypatch.setattr(backends, "runtime_environment", lambda device, ffmpeg="ffmpeg", torch=None: {"fake": True})

    result = backends.extract_with_backends(
        tmp_path,
        specs,
        backends.FrozenBackends(taesd=_FakeVision(), snac=_FakeAudio(), t5=_FakeText()),
        expected_video_ids=video_ids,
    )

    assert result.table.vision.shape == (6, core.FEATURE_DIM)
    assert result.table.audio.shape == (6, core.FEATURE_DIM)
    assert result.table.text.shape == (6, core.FEATURE_DIM)
    assert result.table.target_vision.shape == (6, core.FEATURE_DIM)
    assert len(result.window_rows) == 6
    assert len(result.component_rows) == 2
    assert result.component_rows[0]["taesd_matched_mse"] == pytest.approx(1.1)
    assert result.component_rows[0]["t5_generation_parseable_rate"] == 1.0
    assert result.metadata["projection_seeds"] == {"vision": 12_001, "audio": 12_002, "text": 12_003}
    assert result.window_rows[0]["input_frame_index"] == 2
    assert result.window_rows[0]["target_frame_index"] == 4
    assert result.window_rows[0]["audio_start_sample"] == 0
    assert result.window_rows[0]["audio_stop_sample"] == 24_000
    assert result.window_rows[0]["t5_mask_key"] == "MM-001|video_a|1.0"
    assert set(result.component_arrays) == {
        "taesd_latents",
        "target_taesd_latents",
        "snac_code_ids",
        "t5_pooled_states",
        "t5_masked_input_ids",
        "t5_target_ids",
        "t5_generated_ids",
        "vision_projection_matrix",
        "audio_projection_matrix",
        "text_projection_matrix",
    }
    assert result.component_arrays["taesd_latents"].shape == (6, 4, 8, 8)
    assert result.component_arrays["target_taesd_latents"].shape == (6, 4, 8, 8)
    assert result.component_arrays["snac_code_ids"].shape == (6, 84)
    assert result.component_arrays["t5_pooled_states"].shape == (6, 256)
    assert result.component_arrays["vision_projection_matrix"].shape == (256, 32)
    projection_digests = result.metadata["projection_digests"]
    assert isinstance(projection_digests, dict)
    assert set(projection_digests) == {"vision", "audio", "text"}
