"""Frozen media/model adapters for the MM-001 multimodal preflight.

Only NumPy and repository-local, NumPy-only modules are imported at module import
time.  PyTorch, Diffusers, SNAC, Transformers, and Hugging Face Hub stay behind
explicit loader calls so the ordinary Prospect test and gate paths remain light.

The text arm is deliberately a T5 span-denoising task.  It never presents the
pretrained checkpoint as an identity autoencoder: features are masked-context
encoder states and the raw control is teacher-forced native-sentinel target NLL.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from bench.multimodal_preflight import core, dataset

TAESD_MODEL_ID = "madebyollin/taesd"
TAESD_REVISION = "614f76814bbe30edbe2e627ace1c2234c81a2c0e"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
T5_MODEL_ID = "google/t5-efficient-tiny"
T5_REVISION = "3441d7e8bf3f89841f366d39452b95200416e4a9"

FRAME_RATE = 2
FRAME_SIZE = 64
AUDIO_SAMPLE_RATE = 24_000
SNAC_CODEBOOK_SIZE = 4_096
SNAC_LEVEL_LENGTHS = (12, 24, 48)
VISION_PROJECTION_SEED = 12_001
AUDIO_PROJECTION_SEED = 12_002
TEXT_PROJECTION_SEED = 12_003
SNAC_DECODE_SEED = 12_004
T5_MAX_TOKENS = 96
T5_GENERATION_MAX_TOKENS = 32
T5_MASK_FRACTION = 0.15
SNAC_BATCH_SIZE = 8
T5_BATCH_SIZE = 16


@dataclass(frozen=True, slots=True)
class MaskedSpan:
    """One deterministic T5 span-corruption example."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class VisionBatch:
    """Projected TAESD features and per-frame reconstruction controls."""

    features: np.ndarray
    latents: np.ndarray
    reconstruction: np.ndarray
    matched_mse: np.ndarray
    spatial_mean_mse: np.ndarray
    half_cycle_mse: np.ndarray
    shuffled_latent_mse: np.ndarray


@dataclass(frozen=True, slots=True)
class AudioBatch:
    """Projected SNAC codes and length-correct reconstruction controls."""

    features: np.ndarray
    code_ids: np.ndarray
    reconstruction: np.ndarray
    matched_mse: np.ndarray
    temporally_permuted_reconstruction: np.ndarray
    temporally_permuted_mse: np.ndarray


@dataclass(frozen=True, slots=True)
class TextBatch:
    """Projected T5 encoder states and explicitly seq2seq NLL controls."""

    features: np.ndarray
    pooled_states: np.ndarray
    masked_input_ids: np.ndarray
    target_ids: np.ndarray
    generated_ids: np.ndarray
    mask_start: np.ndarray
    mask_stop: np.ndarray
    correct_target_nll: np.ndarray
    cross_video_target_nll: np.ndarray
    deranged_context_nll: np.ndarray
    generation_length: np.ndarray
    generation_finite: np.ndarray
    generation_parseable: np.ndarray
    generation_exact: np.ndarray


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Serializable audit rows alongside the array-valued core feature table."""

    table: core.FeatureTable
    component_rows: list[dict[str, object]]
    window_rows: list[dict[str, object]]
    component_arrays: dict[str, np.ndarray]
    metadata: dict[str, object]


class VisionBackend(Protocol):
    def embed(self, frames: np.ndarray) -> np.ndarray: ...

    def embed_with_latents(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def analyze(self, frames: np.ndarray) -> VisionBatch: ...

    def model_metadata(self) -> dict[str, object]: ...


class AudioBackend(Protocol):
    def analyze(self, waveforms: np.ndarray) -> AudioBatch: ...

    def model_metadata(self) -> dict[str, object]: ...


class TextBackend(Protocol):
    def analyze(
        self,
        texts: Sequence[str],
        keys: Sequence[str],
        cross_video_indices: np.ndarray,
    ) -> TextBatch: ...

    def model_metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class FrozenBackends:
    taesd: VisionBackend
    snac: AudioBackend
    t5: TextBackend


@dataclass(slots=True)
class _WindowData:
    spec: dataset.WindowSpec
    vision: np.ndarray
    target_vision: np.ndarray
    audio: np.ndarray
    source_waveform: np.ndarray
    audio_reconstruction: np.ndarray
    taesd_latent: np.ndarray
    target_taesd_latent: np.ndarray
    snac_code_ids: np.ndarray
    input_frame_index: int
    target_frame_index: int
    audio_start_sample: int
    audio_stop_sample: int
    taesd_matched_mse: float
    taesd_spatial_mean_mse: float
    taesd_half_cycle_mse: float
    taesd_shuffled_latent_mse: float
    snac_matched_mse: float
    snac_temporally_permuted_mse: float


def _optional_module(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name.split(".")[0]:
            raise
        raise RuntimeError(f"optional backend {name!r} is unavailable; install {install_hint}") from error


def _hub_cache_path(hf_cache: Path) -> Path:
    cache = Path(hf_cache).expanduser()
    return cache if cache.name == "hub" else cache / "hub"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_manifest(cache_dir: Path, model_id: str, revision: str) -> dict[str, str]:
    repository = cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not repository.is_dir():
        return {}
    return {
        str(path.relative_to(repository)): _sha256_file(path)
        for path in sorted(repository.rglob("*"))
        if path.is_file()
    }


def projection_matrix(input_dim: int, seed: int) -> np.ndarray:
    """Recreate the exact untrained Rademacher fixture used by ``core``."""

    if input_dim < 1:
        raise ValueError("projection input_dim must be positive")
    signs = np.random.default_rng(seed).integers(0, 2, size=(input_dim, core.FEATURE_DIM), dtype=np.int8)
    return np.asarray((2.0 * signs.astype(float) - 1.0) / np.sqrt(input_dim), dtype=np.float64)


def projection_digest(matrix: np.ndarray) -> str:
    """Hash a projection in a platform-independent little-endian representation."""

    canonical = np.asarray(matrix, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _parameter_count(model: Any) -> int:
    return int(sum(int(parameter.numel()) for parameter in model.parameters()))


def _module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _configure_torch(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        # PyTorch's deterministic CUDA matmul contract requires this to be set
        # before the first cuBLAS operation.  Backend loading happens before any
        # dataset inference, so bind it here rather than relying on caller state.
        configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if configured != ":4096:8":
            raise RuntimeError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if device.startswith("cuda") and not bool(torch.cuda.is_available()):
        raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seed_torch(torch: Any, seed: int) -> None:
    """Set the global RNG because SNAC's eval decoder injects random noise."""

    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _to_numpy(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.detach().float().cpu().numpy())


def _validate_rgb_frames(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (FRAME_SIZE, FRAME_SIZE, 3):
        raise ValueError(f"frames must have shape (n, {FRAME_SIZE}, {FRAME_SIZE}, 3)")
    if len(values) == 0:
        raise ValueError("frames must be non-empty")
    if FRAME_SIZE % 8 != 0:
        raise AssertionError("TAESD frame dimensions must be multiples of eight")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames contain non-finite values")
    if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
        raise ValueError("frames must be RGB floats in [0, 1]")
    return np.ascontiguousarray(values)


def _validate_waveforms(waveforms: np.ndarray) -> np.ndarray:
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != AUDIO_SAMPLE_RATE:
        raise ValueError(f"waveforms must have shape (n, {AUDIO_SAMPLE_RATE})")
    if len(values) == 0:
        raise ValueError("waveforms must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("waveforms contain non-finite values")
    return np.ascontiguousarray(values)


def taesd_post_map(samples: np.ndarray) -> np.ndarray:
    """Map TAESD decoder samples to RGB units without hiding excursions by clipping."""

    mapped = np.asarray((np.asarray(samples, dtype=np.float32) + 1.0) / 2.0, dtype=np.float32)
    if not np.all(np.isfinite(mapped)):
        raise RuntimeError("TAESD decoder returned non-finite post-map RGB values")
    return mapped


def _mse_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("MSE inputs must have the same batched shape")
    return np.asarray(np.mean(np.square(a - b), axis=tuple(range(1, a.ndim))), dtype=np.float64)


def _half_cycle_indices(length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("a temporal control needs at least two examples")
    shift = max(1, length // 2)
    return np.roll(np.arange(length), shift)


def normalize_snac_codes(
    levels: Sequence[np.ndarray],
    *,
    codebook_size: int = SNAC_CODEBOOK_SIZE,
) -> np.ndarray:
    """Concatenate 12/24/48 SNAC IDs after mapping their range to [-1, 1]."""

    if codebook_size < 2:
        raise ValueError("codebook_size must be at least two")
    if len(levels) != len(SNAC_LEVEL_LENGTHS):
        raise ValueError("SNAC must return exactly three code levels")
    normalized: list[np.ndarray] = []
    batch_size: int | None = None
    for level, expected_length in zip(levels, SNAC_LEVEL_LENGTHS, strict=True):
        ids = np.asarray(level)
        if ids.ndim != 2 or ids.shape[1] != expected_length:
            raise ValueError(f"SNAC code level must have shape (n, {expected_length})")
        if batch_size is None:
            batch_size = len(ids)
        elif len(ids) != batch_size:
            raise ValueError("SNAC code levels have inconsistent batch sizes")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("SNAC code IDs must be integers")
        if np.any(ids < 0) or np.any(ids >= codebook_size):
            raise ValueError("SNAC code ID is outside the codebook")
        normalized.append(2.0 * ids.astype(np.float64) / float(codebook_size - 1) - 1.0)
    return np.concatenate(normalized, axis=1)


def crop_or_pad_audio(values: np.ndarray, original_samples: int) -> np.ndarray:
    """Restore an explicit original audio length after SNAC's padded decode."""

    audio = np.asarray(values)
    if audio.ndim != 2 or original_samples < 1:
        raise ValueError("audio must be a non-empty (batch, samples) array")
    if audio.shape[1] >= original_samples:
        return np.asarray(audio[:, :original_samples], dtype=np.float32)
    return np.pad(audio, ((0, 0), (0, original_samples - audio.shape[1]))).astype(np.float32)


def build_masked_span(
    token_ids: Sequence[int],
    *,
    sentinel_0: int,
    sentinel_1: int,
    eos_token_id: int,
    key: str,
) -> MaskedSpan:
    """Build one deterministic native-sentinel T5 span-corruption example."""

    tokens = tuple(int(token_id) for token_id in token_ids)
    if not tokens:
        raise ValueError("masked-span construction needs at least one token")
    span_length = max(1, int(math.ceil(T5_MASK_FRACTION * len(tokens))))
    choices = len(tokens) - span_length + 1
    start = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % choices
    stop = start + span_length
    encoder = (*tokens[:start], int(sentinel_0), *tokens[stop:], int(eos_token_id))
    target = (int(sentinel_0), *tokens[start:stop], int(sentinel_1), int(eos_token_id))
    return MaskedSpan(input_ids=encoder, target_ids=target, start=start, stop=stop)


def pad_token_sequences(sequences: Sequence[Sequence[int]], pad_value: int) -> np.ndarray:
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("token sequences must be non-empty")
    width = max(len(sequence) for sequence in sequences)
    out = np.full((len(sequences), width), int(pad_value), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        out[row, : len(sequence)] = np.asarray(sequence, dtype=np.int64)
    return out


def per_example_teacher_forced_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute token-mean NLL per row, ignoring the standard ``-100`` pads."""

    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 3 or targets.shape != scores.shape[:2]:
        raise ValueError("logits must be (batch, tokens, vocab) and labels (batch, tokens)")
    mask = targets != -100
    if np.any(np.sum(mask, axis=1) == 0):
        raise ValueError("each example needs at least one target token")
    safe = np.where(mask, targets, 0)
    if np.any(safe < 0) or np.any(safe >= scores.shape[-1]):
        raise ValueError("label is outside the logits vocabulary")
    maximum = np.max(scores, axis=-1)
    log_partition = maximum + np.log(np.sum(np.exp(scores - maximum[..., None]), axis=-1))
    selected = np.take_along_axis(scores, safe[..., None], axis=-1)[..., 0]
    token_nll = log_partition - selected
    return np.asarray(np.sum(np.where(mask, token_nll, 0.0), axis=1) / np.sum(mask, axis=1), dtype=np.float64)


def generation_diagnostics(
    generated_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    pad_token_id: int,
    eos_token_id: int,
    sentinel_0: int,
    sentinel_1: int,
    max_new_tokens: int = T5_GENERATION_MAX_TOKENS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Audit bounded greedy T5 outputs without treating exact match as a gate."""

    generated = np.asarray(generated_ids)
    targets = np.asarray(target_ids, dtype=np.int64)
    if generated.ndim != 2 or targets.ndim != 2 or len(generated) != len(targets):
        raise ValueError("generated_ids and target_ids must be matching batches")
    lengths: list[int] = []
    finite: list[bool] = []
    parseable: list[bool] = []
    exact: list[bool] = []
    for generated_row, target_row in zip(generated, targets, strict=True):
        row_finite = bool(np.all(np.isfinite(generated_row)))
        tokens = [int(token) for token in generated_row if int(token) != pad_token_id]
        if eos_token_id in tokens:
            tokens = tokens[: tokens.index(eos_token_id) + 1]
        target = [int(token) for token in target_row if int(token) != -100]
        first_sentinel = tokens.index(sentinel_0) if sentinel_0 in tokens else -1
        second_sentinel = tokens.index(sentinel_1) if sentinel_1 in tokens else -1
        is_parseable = (
            row_finite
            and 0 <= len(tokens) <= max_new_tokens
            and first_sentinel == 0
            and second_sentinel > first_sentinel
            and eos_token_id in tokens
        )
        lengths.append(len(tokens))
        finite.append(row_finite)
        parseable.append(is_parseable)
        exact.append(tokens == target)
    return (
        np.asarray(lengths, dtype=int),
        np.asarray(finite, dtype=bool),
        np.asarray(parseable, dtype=bool),
        np.asarray(exact, dtype=bool),
    )


def cross_video_nearest_progress_indices(video_ids: Sequence[str], timestamps: Sequence[float]) -> np.ndarray:
    """Map each row to the nearest-progress row in the next sorted video."""

    ids = np.asarray(video_ids, dtype=str)
    times = np.asarray(timestamps, dtype=float)
    if ids.shape != times.shape or ids.ndim != 1:
        raise ValueError("video_ids and timestamps must be matching vectors")
    unique_ids = sorted(set(ids))
    if len(unique_ids) < 2:
        raise ValueError("cross-video control needs at least two videos")
    groups: dict[str, np.ndarray] = {}
    for video_id in unique_ids:
        indices = np.flatnonzero(ids == video_id)
        groups[video_id] = indices[np.argsort(times[indices], kind="stable")]
    out = np.empty(len(ids), dtype=int)
    for position, video_id in enumerate(unique_ids):
        source = groups[video_id]
        target = groups[unique_ids[(position + 1) % len(unique_ids)]]
        for rank, row in enumerate(source):
            progress = rank / max(len(source) - 1, 1)
            target_rank = int(round(progress * (len(target) - 1)))
            out[row] = int(target[target_rank])
    if np.any(ids == ids[out]):
        raise AssertionError("cross-video mapping contains a same-video row")
    return out


def _run_ffmpeg(command: list[str]) -> bytes:
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"ffmpeg executable not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {command[-2]}: {detail}") from error
    return completed.stdout


def _resolve_executable(executable: str) -> Path:
    resolved_raw = shutil.which(executable)
    if resolved_raw is None:
        candidate = Path(executable).expanduser()
        if not candidate.is_file():
            raise RuntimeError(f"executable not found: {executable}")
        resolved_raw = str(candidate)
    return Path(resolved_raw).resolve()


def binary_identity(executable: str, *, version_argument: str = "-version") -> dict[str, object]:
    """Bind an executable to its resolved path, version banner, size, and SHA-256."""

    resolved = _resolve_executable(executable)
    completed = subprocess.run([str(resolved), version_argument], check=True, capture_output=True, text=True)
    banner = (completed.stdout or completed.stderr).splitlines()
    return {
        "path": str(resolved),
        "version": banner[0] if banner else "",
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def runtime_environment(device: str, *, ffmpeg: str = "ffmpeg", torch: Any | None = None) -> dict[str, object]:
    """Record stable software, accelerator, driver, and media-tool identity."""

    ffmpeg_identity = binary_identity(ffmpeg)
    ffprobe_name = str(Path(cast(str, ffmpeg_identity["path"])).with_name("ffprobe"))
    ffprobe_identity = binary_identity(ffprobe_name)
    runtime: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "ffmpeg": ffmpeg_identity,
        "ffprobe": ffprobe_identity,
        "requested_device": device,
    }
    if torch is None:
        runtime["torch"] = None
        return runtime
    torch_info: dict[str, object] = {
        "version": _module_version(torch),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": (
            bool(torch.backends.cudnn.deterministic)
            if getattr(getattr(torch.backends, "cudnn", None), "deterministic", None) is not None
            else None
        ),
        "cudnn_benchmark": (
            bool(torch.backends.cudnn.benchmark)
            if getattr(getattr(torch.backends, "cudnn", None), "benchmark", None) is not None
            else None
        ),
    }
    if device.startswith("cuda"):
        index = int(torch.device(device).index or 0)
        properties = torch.cuda.get_device_properties(index)
        torch_info["gpu"] = {
            "index": index,
            "name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise RuntimeError("CUDA formal runtime requires nvidia-smi for driver binding")
        driver = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        torch_info["driver_version"] = driver[index].strip()
        torch_info["nvidia_smi"] = binary_identity(nvidia_smi, version_argument="--version")
    runtime["torch"] = torch_info
    return runtime


def decode_video_frames(path: str | Path, *, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode an entire video to 2-fps, letterboxed 64x64 RGB float frames."""

    media_path = Path(path)
    filter_graph = (
        "fps=2,scale=64:64:force_original_aspect_ratio=decrease:flags=bicubic,pad=64:64:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    payload = _run_ffmpeg(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vf",
            filter_graph,
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    pixels_per_frame = FRAME_SIZE * FRAME_SIZE * 3
    if not payload or len(payload) % pixels_per_frame != 0:
        raise RuntimeError(f"ffmpeg returned an invalid RGB payload for {media_path}")
    frames = np.frombuffer(payload, dtype=np.uint8).reshape(-1, FRAME_SIZE, FRAME_SIZE, 3)
    return np.asarray(frames, dtype=np.float32) / np.float32(255.0)


def decode_audio_mono_24k(path: str | Path, *, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Decode an entire media file to mono 24-kHz float32 PCM."""

    media_path = Path(path)
    payload = _run_ffmpeg(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    if not payload or len(payload) % np.dtype("<f4").itemsize != 0:
        raise RuntimeError(f"ffmpeg returned an invalid audio payload for {media_path}")
    audio = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
    if not np.all(np.isfinite(audio)):
        raise RuntimeError(f"ffmpeg returned non-finite audio for {media_path}")
    return audio


def probe_media(path: str | Path, *, ffprobe: str = "ffprobe") -> dict[str, object]:
    """Return JSON stream metadata needed to audit decoded media alignment."""

    resolved = str(_resolve_executable(ffprobe))
    completed = subprocess.run(
        [
            resolved,
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,"
                "avg_frame_rate,nb_frames,sample_rate,channels,duration"
            ),
            "-of",
            "json",
            str(Path(path)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("streams"), list):
        raise RuntimeError(f"ffprobe returned no stream list for {path}")
    streams = cast(list[object], raw["streams"])
    stream_types = {stream.get("codec_type") for stream in streams if isinstance(stream, dict)}
    if "video" not in stream_types or "audio" not in stream_types:
        raise RuntimeError(f"media must contain video and audio streams: {path}")
    return cast(dict[str, object], raw)


def inspect_media_inputs(
    cache_path: Path,
    specs: Iterable[dataset.WindowSpec] | None = None,
    *,
    ffmpeg: str = "ffmpeg",
) -> dict[str, object]:
    """Validate all decoded frame/sample intervals without running any encoder."""

    if specs is None:
        annotations = dataset.validate_sample_cache(cache_path)
        formal_specs = [
            spec
            for video_id in dataset.SAMPLE_VIDEO_IDS
            for spec in dataset.windows_for_video(video_id, annotations[video_id])
        ]
    else:
        formal_specs = list(specs)
    ordered = _validate_specs(formal_specs, dataset.SAMPLE_VIDEO_IDS)
    ffmpeg_path = cast(str, binary_identity(ffmpeg)["path"])
    ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe"))
    videos: dict[str, object] = {}
    for video_id in dataset.SAMPLE_VIDEO_IDS:
        media_path = Path(cache_path) / "videos" / f"{video_id}.mp4"
        frames = decode_video_frames(media_path, ffmpeg=ffmpeg_path)
        audio = decode_audio_mono_24k(media_path, ffmpeg=ffmpeg_path)
        video_specs = [spec for spec in ordered if spec.video_id == video_id]
        input_indices = [frame_index_at(spec.frame_seconds, len(frames)) for spec in video_specs]
        target_indices = [frame_index_at(spec.target_seconds, len(frames)) for spec in video_specs]
        sample_bounds = [
            audio_sample_bounds(spec.audio_start_seconds, spec.audio_end_seconds, len(audio)) for spec in video_specs
        ]
        videos[video_id] = {
            "file_size_bytes": media_path.stat().st_size,
            "frame_count_2fps": len(frames),
            "audio_samples_24khz": len(audio),
            "window_count": len(video_specs),
            "input_frame_index_min": min(input_indices),
            "input_frame_index_max": max(input_indices),
            "target_frame_index_min": min(target_indices),
            "target_frame_index_max": max(target_indices),
            "audio_start_sample_min": min(start for start, _ in sample_bounds),
            "audio_stop_sample_max": max(stop for _, stop in sample_bounds),
            "ffprobe": probe_media(media_path, ffprobe=ffprobe_path),
        }
    return {
        "video_ids": list(dataset.SAMPLE_VIDEO_IDS),
        "window_count": len(ordered),
        "window_counts": dict(dataset.EXPECTED_WINDOW_COUNTS),
        "videos": videos,
        "environment": runtime_environment("media-only", ffmpeg=ffmpeg_path),
        "dataset_inference_performed": False,
    }


def frame_index_at(timestamp: float, frame_count: int) -> int:
    """Return an exact 2-fps index, rejecting fractional or unavailable frames."""

    scaled = float(timestamp) * FRAME_RATE
    index = int(round(scaled))
    if not math.isfinite(scaled) or timestamp < 0.0 or not math.isclose(scaled, index, abs_tol=1e-9):
        raise ValueError("frame timestamp must lie exactly on the 2-fps grid")
    if frame_count < 1 or index >= frame_count:
        raise ValueError(f"frame index {index} is unavailable from {frame_count} decoded frames")
    return index


def _frame_at(frames: np.ndarray, timestamp: float) -> tuple[np.ndarray, int]:
    index = frame_index_at(timestamp, len(frames))
    return np.asarray(frames[index], dtype=np.float32), index


def audio_sample_bounds(start_seconds: float, end_seconds: float, sample_count: int) -> tuple[int, int]:
    """Return exact half-open 24-kHz bounds and reject missing source samples."""

    scaled_start = float(start_seconds) * AUDIO_SAMPLE_RATE
    scaled_stop = float(end_seconds) * AUDIO_SAMPLE_RATE
    start = int(round(scaled_start))
    stop = int(round(scaled_stop))
    if (
        not math.isfinite(scaled_start)
        or not math.isfinite(scaled_stop)
        or not math.isclose(scaled_start, start, abs_tol=1e-6)
        or not math.isclose(scaled_stop, stop, abs_tol=1e-6)
        or start < 0
        or stop - start != AUDIO_SAMPLE_RATE
    ):
        raise ValueError("audio interval must be an exact non-negative one-second 24-kHz window")
    if stop > sample_count:
        raise ValueError(f"audio stop sample {stop} is unavailable from {sample_count} decoded samples")
    return start, stop


def _audio_interval(audio: np.ndarray, start_seconds: float, end_seconds: float) -> tuple[np.ndarray, int, int]:
    start, stop = audio_sample_bounds(start_seconds, end_seconds, len(audio))
    return np.asarray(audio[start:stop], dtype=np.float32), start, stop


class TAESDBackend:
    """Thin verified wrapper around ``diffusers.AutoencoderTiny``."""

    def __init__(
        self,
        model: Any,
        torch: Any,
        *,
        device: str,
        dtype: Any,
        cache_dir: Path,
        load_seconds: float,
        diffusers_version: str,
    ) -> None:
        self._model = model
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._cache_dir = cache_dir
        self._load_seconds = load_seconds
        self._inference_seconds = 0.0
        self._diffusers_version = diffusers_version

    @classmethod
    def load(cls, hf_cache: Path, *, device: str, local_files_only: bool) -> TAESDBackend:
        torch = _optional_module("torch", "torch")
        diffusers = _optional_module("diffusers", "diffusers")
        _configure_torch(torch, device)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        cache_dir = _hub_cache_path(hf_cache)
        started = time.perf_counter()
        model = diffusers.AutoencoderTiny.from_pretrained(
            TAESD_MODEL_ID,
            revision=TAESD_REVISION,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
        model = model.to(device=device, dtype=dtype).eval()
        return cls(
            model,
            torch,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            load_seconds=time.perf_counter() - started,
            diffusers_version=_module_version(diffusers),
        )

    def _encode(self, frames: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
        values = _validate_rgb_frames(frames)
        tensor = self._torch.from_numpy(values).permute(0, 3, 1, 2).to(self._device, dtype=self._dtype)
        tensor = tensor * 2.0 - 1.0
        started = time.perf_counter()
        with self._torch.inference_mode():
            latents = self._model.encode(tensor).latents
        _synchronize(self._torch, self._device)
        self._inference_seconds += time.perf_counter() - started
        latent_values = _to_numpy(latents)
        if latent_values.shape != (len(values), 4, 8, 8):
            raise RuntimeError(f"TAESD returned unexpected latent shape {latent_values.shape}")
        features = core.fixed_projection(
            latent_values.reshape(len(values), -1), core.FEATURE_DIM, VISION_PROJECTION_SEED
        )
        return latents, features, np.asarray(latent_values, dtype=np.float32)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        return self._encode(frames)[1]

    def embed_with_latents(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, features, latents = self._encode(frames)
        return features, latents

    def analyze(self, frames: np.ndarray) -> VisionBatch:
        values = _validate_rgb_frames(frames)
        if len(values) < 2:
            raise ValueError("TAESD framewise-video control needs at least two frames")
        latents, features, latent_values = self._encode(values)
        permutation = _half_cycle_indices(len(values))
        tensor_permutation = self._torch.as_tensor(permutation, device=self._device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            reconstruction = self._model.decode(latents).sample
            shuffled = self._model.decode(latents[tensor_permutation]).sample
        _synchronize(self._torch, self._device)
        self._inference_seconds += time.perf_counter() - started
        decoded = taesd_post_map(_to_numpy(reconstruction).transpose(0, 2, 3, 1))
        shuffled_decoded = taesd_post_map(_to_numpy(shuffled).transpose(0, 2, 3, 1))
        spatial_mean = np.mean(values, axis=(1, 2), keepdims=True)
        return VisionBatch(
            features=features,
            latents=latent_values,
            reconstruction=decoded,
            matched_mse=_mse_rows(values, decoded),
            spatial_mean_mse=_mse_rows(values, np.broadcast_to(spatial_mean, values.shape)),
            half_cycle_mse=_mse_rows(values, decoded[permutation]),
            shuffled_latent_mse=_mse_rows(values, shuffled_decoded),
        )

    def model_metadata(self) -> dict[str, object]:
        return {
            "model_id": TAESD_MODEL_ID,
            "requested_revision": TAESD_REVISION,
            "resolved_revision": TAESD_REVISION,
            "parameter_count": _parameter_count(self._model),
            "device": self._device,
            "dtype": str(self._dtype),
            "load_seconds": self._load_seconds,
            "inference_seconds": self._inference_seconds,
            "rgb_input_range": [0.0, 1.0],
            "decoder_post_map": "(sample + 1) / 2",
            "decoder_post_map_clipping": False,
            "versions": {"torch": _module_version(self._torch), "diffusers": self._diffusers_version},
            "snapshot_files": _snapshot_manifest(self._cache_dir, TAESD_MODEL_ID, TAESD_REVISION),
        }


class SNACBackend:
    """Verified 24-kHz SNAC wrapper with explicit length and RNG contracts."""

    def __init__(
        self,
        model: Any,
        torch: Any,
        *,
        device: str,
        dtype: Any,
        cache_dir: Path,
        load_seconds: float,
        snac_version: str,
    ) -> None:
        self._model = model
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self._cache_dir = cache_dir
        self._load_seconds = load_seconds
        self._inference_seconds = 0.0
        self._snac_version = snac_version
        self._warmed_batch_sizes: set[int] = set()
        self._determinism_checks = 0

    @classmethod
    def load(cls, hf_cache: Path, *, device: str, local_files_only: bool) -> SNACBackend:
        torch = _optional_module("torch", "torch")
        snac_module = _optional_module("snac", "snac")
        _configure_torch(torch, device)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        cache_dir = _hub_cache_path(hf_cache)
        started = time.perf_counter()
        model = snac_module.SNAC.from_pretrained(
            SNAC_MODEL_ID,
            revision=SNAC_REVISION,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
        model = model.to(device=device, dtype=dtype).eval()
        return cls(
            model,
            torch,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            load_seconds=time.perf_counter() - started,
            snac_version=_module_version(snac_module),
        )

    def _decode(self, codes: Sequence[Any], original_samples: int, *, seed: int) -> np.ndarray:
        _seed_torch(self._torch, seed)
        with self._torch.inference_mode():
            decoded = self._model.decode(list(codes))
        values = _to_numpy(decoded)
        if values.ndim != 3 or values.shape[1] != 1:
            raise RuntimeError(f"SNAC returned unexpected decoded shape {values.shape}")
        return crop_or_pad_audio(values[:, 0, :], original_samples)

    def _warm_encoder(self, batch_size: int, original_samples: int) -> None:
        """Consume SNAC encode/decode first-call paths for each observed batch shape."""

        if batch_size in self._warmed_batch_sizes:
            return
        zeros = self._torch.zeros((batch_size, 1, original_samples), device=self._device, dtype=self._dtype)
        with self._torch.inference_mode():
            warm_codes = list(self._model.encode(zeros))
        self._decode(warm_codes, original_samples, seed=SNAC_DECODE_SEED)
        _synchronize(self._torch, self._device)
        self._warmed_batch_sizes.add(batch_size)

    def analyze(self, waveforms: np.ndarray) -> AudioBatch:
        values = _validate_waveforms(waveforms)
        started = time.perf_counter()
        normalized_rows: list[np.ndarray] = []
        code_rows: list[np.ndarray] = []
        reconstruction_rows: list[np.ndarray] = []
        permuted_rows: list[np.ndarray] = []
        for start in range(0, len(values), SNAC_BATCH_SIZE):
            stop = min(start + SNAC_BATCH_SIZE, len(values))
            tensor = self._torch.from_numpy(values[start:stop, None, :]).to(self._device, dtype=self._dtype)
            self._warm_encoder(stop - start, values.shape[1])
            _seed_torch(self._torch, SNAC_DECODE_SEED + start)
            with self._torch.inference_mode():
                first_codes = list(self._model.encode(tensor))
                codes = list(self._model.encode(tensor))
            if len(first_codes) != len(codes) or any(
                not bool(self._torch.equal(first, second)) for first, second in zip(first_codes, codes, strict=True)
            ):
                raise RuntimeError("SNAC token IDs did not reproduce exactly after CUDA warm-up")
            level_values = [_to_numpy(level).astype(np.int64) for level in codes]
            normalized_rows.append(normalize_snac_codes(level_values))
            code_rows.append(np.concatenate(level_values, axis=1))
            reconstruction = self._decode(codes, values.shape[1], seed=SNAC_DECODE_SEED + start)
            repeated_reconstruction = self._decode(codes, values.shape[1], seed=SNAC_DECODE_SEED + start)
            if not np.allclose(reconstruction, repeated_reconstruction, rtol=1e-6, atol=1e-6):
                raise RuntimeError("SNAC matched decode did not reproduce within tolerance")
            reconstruction_rows.append(reconstruction)
            permuted_codes = [self._torch.roll(level, shifts=max(1, level.shape[-1] // 2), dims=-1) for level in codes]
            permuted_reconstruction = self._decode(permuted_codes, values.shape[1], seed=SNAC_DECODE_SEED + start)
            repeated_permuted = self._decode(permuted_codes, values.shape[1], seed=SNAC_DECODE_SEED + start)
            if not np.allclose(permuted_reconstruction, repeated_permuted, rtol=1e-6, atol=1e-6):
                raise RuntimeError("SNAC control decode did not reproduce within tolerance")
            permuted_rows.append(permuted_reconstruction)
            self._determinism_checks += 1
        normalized = np.concatenate(normalized_rows, axis=0)
        code_ids = np.concatenate(code_rows, axis=0)
        reconstruction = np.concatenate(reconstruction_rows, axis=0)
        permuted_reconstruction = np.concatenate(permuted_rows, axis=0)
        features = core.fixed_projection(normalized, core.FEATURE_DIM, AUDIO_PROJECTION_SEED)
        _synchronize(self._torch, self._device)
        self._inference_seconds += time.perf_counter() - started
        return AudioBatch(
            features=features,
            code_ids=code_ids,
            reconstruction=reconstruction,
            matched_mse=_mse_rows(values, reconstruction),
            temporally_permuted_reconstruction=permuted_reconstruction,
            temporally_permuted_mse=_mse_rows(values, permuted_reconstruction),
        )

    def model_metadata(self) -> dict[str, object]:
        return {
            "model_id": SNAC_MODEL_ID,
            "requested_revision": SNAC_REVISION,
            "resolved_revision": SNAC_REVISION,
            "parameter_count": _parameter_count(self._model),
            "device": self._device,
            "dtype": str(self._dtype),
            "load_seconds": self._load_seconds,
            "inference_seconds": self._inference_seconds,
            "decode_seed": SNAC_DECODE_SEED,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "determinism_checks": self._determinism_checks,
            "batch_size": SNAC_BATCH_SIZE,
            "warmed_batch_sizes": sorted(self._warmed_batch_sizes),
            "repeat_rtol": 1e-6,
            "repeat_atol": 1e-6,
            "seed_schedule": "SNAC_DECODE_SEED + chunk_start",
            "versions": {"torch": _module_version(self._torch), "snac": self._snac_version},
            "snapshot_files": _snapshot_manifest(self._cache_dir, SNAC_MODEL_ID, SNAC_REVISION),
        }


class T5Backend:
    """T5 span-denoising encoder/teacher-forcing adapter; not an autoencoder."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch: Any,
        *,
        cache_dir: Path,
        load_seconds: float,
        transformers_version: str,
        batch_size: int = T5_BATCH_SIZE,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._device = "cpu"
        self._dtype = torch.float32
        self._cache_dir = cache_dir
        self._load_seconds = load_seconds
        self._inference_seconds = 0.0
        self._transformers_version = transformers_version
        self._batch_size = batch_size

    @classmethod
    def load(cls, hf_cache: Path, *, local_files_only: bool) -> T5Backend:
        torch = _optional_module("torch", "torch")
        transformers = _optional_module("transformers", "transformers")
        _configure_torch(torch, "cpu")
        cache_dir = _hub_cache_path(hf_cache)
        started = time.perf_counter()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            T5_MODEL_ID,
            revision=T5_REVISION,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = (
            transformers.AutoModelForSeq2SeqLM.from_pretrained(
                T5_MODEL_ID,
                revision=T5_REVISION,
                cache_dir=str(cache_dir),
                local_files_only=local_files_only,
                dtype=torch.float32,
            )
            .to("cpu")
            .eval()
        )
        return cls(
            model,
            tokenizer,
            torch,
            cache_dir=cache_dir,
            load_seconds=time.perf_counter() - started,
            transformers_version=_module_version(transformers),
        )

    def _examples(
        self, texts: Sequence[str], keys: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(texts) != len(keys) or not texts:
            raise ValueError("texts and keys must be matching non-empty sequences")
        sentinel_0 = int(self._tokenizer.convert_tokens_to_ids("<extra_id_0>"))
        sentinel_1 = int(self._tokenizer.convert_tokens_to_ids("<extra_id_1>"))
        eos_token_id = int(self._tokenizer.eos_token_id)
        spans: list[MaskedSpan] = []
        for text, key in zip(texts, keys, strict=True):
            token_ids = cast(
                list[int],
                self._tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=T5_MAX_TOKENS,
                ),
            )
            spans.append(
                build_masked_span(
                    token_ids,
                    sentinel_0=sentinel_0,
                    sentinel_1=sentinel_1,
                    eos_token_id=eos_token_id,
                    key=key,
                )
            )
        input_ids = pad_token_sequences([span.input_ids for span in spans], int(self._tokenizer.pad_token_id))
        attention_mask = input_ids != int(self._tokenizer.pad_token_id)
        labels = pad_token_sequences([span.target_ids for span in spans], -100)
        starts = np.asarray([span.start for span in spans], dtype=np.int64)
        stops = np.asarray([span.stop for span in spans], dtype=np.int64)
        return input_ids, attention_mask.astype(np.int64), labels, starts, stops

    def _uncorrupted_inputs(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        sequences: list[list[int]] = []
        for text in texts:
            token_ids = cast(
                list[int],
                self._tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=T5_MAX_TOKENS,
                ),
            )
            if not token_ids:
                raise ValueError("canonical text must tokenize to at least one token")
            sequences.append([*token_ids, int(self._tokenizer.eos_token_id)])
        input_ids = pad_token_sequences(sequences, int(self._tokenizer.pad_token_id))
        return input_ids, (input_ids != int(self._tokenizer.pad_token_id)).astype(np.int64)

    def _nll(self, input_ids: Any, attention_mask: Any, labels: Any) -> np.ndarray:
        output = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return per_example_teacher_forced_nll(_to_numpy(output.logits), _to_numpy(labels).astype(np.int64))

    def analyze(
        self,
        texts: Sequence[str],
        keys: Sequence[str],
        cross_video_indices: np.ndarray,
    ) -> TextBatch:
        input_values, mask_values, label_values, mask_starts, mask_stops = self._examples(texts, keys)
        original_input_values, original_mask_values = self._uncorrupted_inputs(texts)
        permutation = np.asarray(cross_video_indices, dtype=int)
        if permutation.shape != (len(texts),) or np.any(permutation < 0) or np.any(permutation >= len(texts)):
            raise ValueError("cross_video_indices must index every text example")
        pooled_rows: list[np.ndarray] = []
        correct_rows: list[np.ndarray] = []
        cross_target_rows: list[np.ndarray] = []
        deranged_context_rows: list[np.ndarray] = []
        generated_rows: list[np.ndarray] = []
        started = time.perf_counter()
        with self._torch.inference_mode():
            for start in range(0, len(texts), self._batch_size):
                stop = min(start + self._batch_size, len(texts))
                rows = np.arange(start, stop)
                paired = permutation[rows]
                input_ids = self._torch.from_numpy(input_values[rows]).to("cpu")
                attention_mask = self._torch.from_numpy(mask_values[rows]).to("cpu")
                original_input_ids = self._torch.from_numpy(original_input_values[rows]).to("cpu")
                original_attention_mask = self._torch.from_numpy(original_mask_values[rows]).to("cpu")
                labels = self._torch.from_numpy(label_values[rows]).to("cpu")
                paired_labels = self._torch.from_numpy(label_values[paired]).to("cpu")
                paired_inputs = self._torch.from_numpy(input_values[paired]).to("cpu")
                paired_masks = self._torch.from_numpy(mask_values[paired]).to("cpu")
                hidden = self._model.get_encoder()(
                    input_ids=original_input_ids,
                    attention_mask=original_attention_mask,
                ).last_hidden_state
                weights = original_attention_mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                pooled_rows.append(_to_numpy(pooled))
                correct_rows.append(self._nll(input_ids, attention_mask, labels))
                cross_target_rows.append(self._nll(input_ids, attention_mask, paired_labels))
                deranged_context_rows.append(self._nll(paired_inputs, paired_masks, labels))
                generated_rows.append(
                    _to_numpy(
                        self._model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=False,
                            max_new_tokens=T5_GENERATION_MAX_TOKENS,
                        )
                    ).astype(np.int64)
                )
        self._inference_seconds += time.perf_counter() - started
        pooled_values = np.concatenate(pooled_rows, axis=0)
        generated_values = pad_token_sequences(
            [row.tolist() for batch in generated_rows for row in batch],
            int(self._tokenizer.pad_token_id),
        )
        lengths, finite, parseable, exact = generation_diagnostics(
            generated_values,
            label_values,
            pad_token_id=int(self._tokenizer.pad_token_id),
            eos_token_id=int(self._tokenizer.eos_token_id),
            sentinel_0=int(self._tokenizer.convert_tokens_to_ids("<extra_id_0>")),
            sentinel_1=int(self._tokenizer.convert_tokens_to_ids("<extra_id_1>")),
        )
        return TextBatch(
            features=core.fixed_projection(pooled_values, core.FEATURE_DIM, TEXT_PROJECTION_SEED),
            pooled_states=np.asarray(pooled_values, dtype=np.float32),
            masked_input_ids=input_values,
            target_ids=label_values,
            generated_ids=generated_values,
            mask_start=mask_starts,
            mask_stop=mask_stops,
            correct_target_nll=np.concatenate(correct_rows),
            cross_video_target_nll=np.concatenate(cross_target_rows),
            deranged_context_nll=np.concatenate(deranged_context_rows),
            generation_length=lengths,
            generation_finite=finite,
            generation_parseable=parseable,
            generation_exact=exact,
        )

    def model_metadata(self) -> dict[str, object]:
        return {
            "model_id": T5_MODEL_ID,
            "requested_revision": T5_REVISION,
            "resolved_revision": T5_REVISION,
            "parameter_count": _parameter_count(self._model),
            "device": "cpu",
            "dtype": str(self._dtype),
            "load_seconds": self._load_seconds,
            "inference_seconds": self._inference_seconds,
            "task": "masked-span seq2seq teacher forcing (not identity autoencoding)",
            "mask_placement": "sha256('MM-001|{video_id}|{timestamp:.1f}')",
            "tokenizer_ids": {
                "pad": int(self._tokenizer.pad_token_id),
                "eos": int(self._tokenizer.eos_token_id),
                "sentinel_0": int(self._tokenizer.convert_tokens_to_ids("<extra_id_0>")),
                "sentinel_1": int(self._tokenizer.convert_tokens_to_ids("<extra_id_1>")),
            },
            "versions": {"torch": _module_version(self._torch), "transformers": self._transformers_version},
            "snapshot_files": _snapshot_manifest(self._cache_dir, T5_MODEL_ID, T5_REVISION),
        }


def load_frozen_backends(
    hf_cache: Path,
    *,
    device: str,
    local_files_only: bool,
) -> FrozenBackends:
    """Load the three exact frozen revisions with the formal device policy."""

    return FrozenBackends(
        taesd=TAESDBackend.load(hf_cache, device=device, local_files_only=local_files_only),
        snac=SNACBackend.load(hf_cache, device=device, local_files_only=local_files_only),
        t5=T5Backend.load(hf_cache, local_files_only=local_files_only),
    )


def inspect_backend_inputs(hf_cache: Path, *, device: str) -> dict[str, object]:
    """Download/load and bind exact model inputs without performing dataset inference."""

    started = time.perf_counter()
    backends = load_frozen_backends(hf_cache, device=device, local_files_only=False)
    models = {
        "taesd": backends.taesd.model_metadata(),
        "snac": backends.snac.model_metadata(),
        "t5": backends.t5.model_metadata(),
    }
    if any(not cast(dict[str, object], metadata)["snapshot_files"] for metadata in models.values()):
        raise RuntimeError("could not bind every exact model snapshot in the requested HF cache")
    return {
        "models": models,
        "environment": runtime_environment(device, torch=getattr(backends.taesd, "_torch", None)),
        "inspection_seconds": time.perf_counter() - started,
        "hf_cache": str(_hub_cache_path(hf_cache).resolve()),
        "dataset_inference_performed": False,
    }


def _validate_specs(specs: Iterable[dataset.WindowSpec], expected_video_ids: Sequence[str]) -> list[dataset.WindowSpec]:
    ordered = sorted(specs, key=lambda spec: (spec.video_id, spec.frame_seconds))
    if not ordered:
        raise ValueError("at least one WindowSpec is required")
    actual_ids = tuple(sorted({spec.video_id for spec in ordered}))
    expected_ids = tuple(sorted(expected_video_ids))
    if actual_ids != expected_ids:
        raise ValueError(f"window videos mismatch: expected={expected_ids}, actual={actual_ids}")
    keys = [(spec.video_id, spec.frame_seconds) for spec in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("WindowSpec video/timestamp pairs must be unique")
    for video_id in actual_ids:
        if sum(spec.video_id == video_id for spec in ordered) < 2:
            raise ValueError(f"video {video_id} needs at least two windows for controls")
    if expected_ids == dataset.SAMPLE_VIDEO_IDS:
        ordered = list(dataset.validate_formal_window_specs(ordered))
    return ordered


def extract_with_backends(
    cache_path: Path,
    specs: Iterable[dataset.WindowSpec],
    backends: FrozenBackends,
    *,
    expected_video_ids: Sequence[str] = dataset.SAMPLE_VIDEO_IDS,
    ffmpeg: str = "ffmpeg",
) -> ExtractionResult:
    """Extract a table from already-bound backends (the no-download test seam)."""

    started = time.perf_counter()
    ordered = _validate_specs(specs, expected_video_ids)
    data: list[_WindowData] = []
    decoded_media: dict[str, dict[str, object]] = {}
    ffmpeg_seconds = 0.0
    for video_id in sorted(expected_video_ids):
        media_path = Path(cache_path) / "videos" / f"{video_id}.mp4"
        decode_started = time.perf_counter()
        frames = decode_video_frames(media_path, ffmpeg=ffmpeg)
        audio = decode_audio_mono_24k(media_path, ffmpeg=ffmpeg)
        decoded_media[video_id] = {
            "frame_count_2fps": len(frames),
            "audio_samples_24khz": len(audio),
            "file_size_bytes": media_path.stat().st_size if media_path.is_file() else 0,
        }
        if tuple(sorted(expected_video_ids)) == dataset.SAMPLE_VIDEO_IDS:
            ffmpeg_path = _resolve_executable(ffmpeg)
            decoded_media[video_id]["ffprobe"] = probe_media(media_path, ffprobe=str(ffmpeg_path.with_name("ffprobe")))
        ffmpeg_seconds += time.perf_counter() - decode_started
        video_specs = [spec for spec in ordered if spec.video_id == video_id]
        current_items = [_frame_at(frames, spec.frame_seconds) for spec in video_specs]
        target_items = [_frame_at(frames, spec.target_seconds) for spec in video_specs]
        audio_items = [_audio_interval(audio, spec.audio_start_seconds, spec.audio_end_seconds) for spec in video_specs]
        current_frames = np.stack([item[0] for item in current_items])
        target_frames = np.stack([item[0] for item in target_items])
        waveforms = np.stack([item[0] for item in audio_items])
        vision = backends.taesd.analyze(current_frames)
        target_vision, target_latents = backends.taesd.embed_with_latents(target_frames)
        audio_batch = backends.snac.analyze(waveforms)
        if vision.features.shape != (len(video_specs), core.FEATURE_DIM):
            raise RuntimeError("TAESD backend returned invalid feature shape")
        if target_vision.shape != (len(video_specs), core.FEATURE_DIM):
            raise RuntimeError("TAESD target backend returned invalid feature shape")
        if target_latents.shape != (len(video_specs), 4, 8, 8):
            raise RuntimeError("TAESD target backend returned invalid raw latent shape")
        if audio_batch.features.shape != (len(video_specs), core.FEATURE_DIM):
            raise RuntimeError("SNAC backend returned invalid feature shape")
        if vision.latents.shape != (len(video_specs), 4, 8, 8):
            raise RuntimeError("TAESD backend returned invalid raw latent shape")
        if audio_batch.code_ids.shape != (len(video_specs), sum(SNAC_LEVEL_LENGTHS)):
            raise RuntimeError("SNAC backend returned invalid raw code shape")
        for index, spec in enumerate(video_specs):
            data.append(
                _WindowData(
                    spec=spec,
                    vision=np.asarray(vision.features[index], dtype=float),
                    target_vision=np.asarray(target_vision[index], dtype=float),
                    audio=np.asarray(audio_batch.features[index], dtype=float),
                    source_waveform=np.asarray(waveforms[index], dtype=np.float32),
                    audio_reconstruction=np.asarray(audio_batch.reconstruction[index], dtype=np.float32),
                    taesd_latent=np.asarray(vision.latents[index], dtype=np.float32),
                    target_taesd_latent=np.asarray(target_latents[index], dtype=np.float32),
                    snac_code_ids=np.asarray(audio_batch.code_ids[index], dtype=np.int64),
                    input_frame_index=current_items[index][1],
                    target_frame_index=target_items[index][1],
                    audio_start_sample=audio_items[index][1],
                    audio_stop_sample=audio_items[index][2],
                    taesd_matched_mse=float(vision.matched_mse[index]),
                    taesd_spatial_mean_mse=float(vision.spatial_mean_mse[index]),
                    taesd_half_cycle_mse=float(vision.half_cycle_mse[index]),
                    taesd_shuffled_latent_mse=float(vision.shuffled_latent_mse[index]),
                    snac_matched_mse=float(audio_batch.matched_mse[index]),
                    snac_temporally_permuted_mse=float(audio_batch.temporally_permuted_mse[index]),
                )
            )
    data.sort(key=lambda row: (row.spec.video_id, row.spec.frame_seconds))
    video_ids = [row.spec.video_id for row in data]
    timestamps = [row.spec.frame_seconds for row in data]
    cross_indices = cross_video_nearest_progress_indices(video_ids, timestamps)
    texts = [row.spec.annotation_text for row in data]
    keys = [f"MM-001|{row.spec.video_id}|{row.spec.frame_seconds:.1f}" for row in data]
    text_batch = backends.t5.analyze(texts, keys, cross_indices)
    if text_batch.features.shape != (len(data), core.FEATURE_DIM):
        raise RuntimeError("T5 backend returned invalid feature shape")
    if text_batch.pooled_states.shape != (len(data), 256):
        raise RuntimeError("T5 backend returned invalid pooled-state shape")

    # The frozen SNAC identity control compares each source window with the next
    # video's nearest-progress reconstruction, not with a within-video shuffle.
    source_waveforms = np.stack([row.source_waveform for row in data])
    reconstructions = np.stack([row.audio_reconstruction for row in data])
    snac_cross_video_mse = _mse_rows(source_waveforms, reconstructions[cross_indices])

    window_rows: list[dict[str, object]] = []
    for index, row in enumerate(data):
        window_rows.append(
            {
                "video_id": row.spec.video_id,
                "timestamp": row.spec.frame_seconds,
                "target_timestamp": row.spec.target_seconds,
                "input_frame_index": row.input_frame_index,
                "target_frame_index": row.target_frame_index,
                "audio_start_sample": row.audio_start_sample,
                "audio_stop_sample": row.audio_stop_sample,
                "cross_video_index": int(cross_indices[index]),
                "annotation_text": row.spec.annotation_text,
                "annotation_present": row.spec.annotation_text != dataset.NO_EVENT_MARKER,
                "taesd_matched_mse": row.taesd_matched_mse,
                "taesd_spatial_mean_mse": row.taesd_spatial_mean_mse,
                "taesd_half_cycle_mse": row.taesd_half_cycle_mse,
                "taesd_shuffled_latent_mse": row.taesd_shuffled_latent_mse,
                "snac_matched_mse": row.snac_matched_mse,
                "snac_cross_video_mse": float(snac_cross_video_mse[index]),
                "snac_temporally_permuted_mse": row.snac_temporally_permuted_mse,
                "t5_correct_target_nll": float(text_batch.correct_target_nll[index]),
                "t5_cross_video_target_nll": float(text_batch.cross_video_target_nll[index]),
                "t5_deranged_context_nll": float(text_batch.deranged_context_nll[index]),
                "t5_generation_length": int(text_batch.generation_length[index]),
                "t5_generation_finite": bool(text_batch.generation_finite[index]),
                "t5_generation_parseable": bool(text_batch.generation_parseable[index]),
                "t5_generation_exact": bool(text_batch.generation_exact[index]),
                "t5_mask_key": keys[index],
                "t5_mask_start": int(text_batch.mask_start[index]),
                "t5_mask_stop": int(text_batch.mask_stop[index]),
            }
        )

    metric_names = (
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
    component_rows: list[dict[str, object]] = []
    for video_id in sorted(expected_video_ids):
        rows = [row for row in window_rows if row["video_id"] == video_id]
        aggregate: dict[str, object] = {"video_id": video_id, "window_count": len(rows)}
        for name in metric_names:
            aggregate[name] = float(np.median([float(cast(float, row[name])) for row in rows]))
        aggregate["t5_generation_max_tokens"] = max(int(cast(int, row["t5_generation_length"])) for row in rows)
        for name in ("t5_generation_finite", "t5_generation_parseable", "t5_generation_exact"):
            aggregate[f"{name}_rate"] = float(np.mean([bool(row[name]) for row in rows]))
        component_rows.append(aggregate)

    table = core.FeatureTable(
        video_ids=np.asarray(video_ids),
        timestamps=np.asarray(timestamps, dtype=float),
        vision=np.stack([row.vision for row in data]),
        audio=np.stack([row.audio for row in data]),
        text=np.asarray(text_batch.features, dtype=float),
        target_vision=np.stack([row.target_vision for row in data]),
        annotation_present=np.asarray(
            [row.spec.annotation_text != dataset.NO_EVENT_MARKER for row in data], dtype=bool
        ),
    )
    table.validate()
    if table.target_vision.shape != (len(data), core.FEATURE_DIM) or not np.all(np.isfinite(table.target_vision)):
        raise ValueError("target_vision must be a finite 32-D feature per window")
    vision_matrix = projection_matrix(4 * 8 * 8, VISION_PROJECTION_SEED)
    audio_matrix = projection_matrix(sum(SNAC_LEVEL_LENGTHS), AUDIO_PROJECTION_SEED)
    text_matrix = projection_matrix(256, TEXT_PROJECTION_SEED)
    component_arrays: dict[str, np.ndarray] = {
        "taesd_latents": np.stack([row.taesd_latent for row in data]).astype(np.float32),
        "target_taesd_latents": np.stack([row.target_taesd_latent for row in data]).astype(np.float32),
        "snac_code_ids": np.stack([row.snac_code_ids for row in data]).astype(np.int64),
        "t5_pooled_states": np.asarray(text_batch.pooled_states, dtype=np.float32),
        "t5_masked_input_ids": np.asarray(text_batch.masked_input_ids, dtype=np.int64),
        "t5_target_ids": np.asarray(text_batch.target_ids, dtype=np.int64),
        "t5_generated_ids": np.asarray(text_batch.generated_ids, dtype=np.int64),
        "vision_projection_matrix": vision_matrix,
        "audio_projection_matrix": audio_matrix,
        "text_projection_matrix": text_matrix,
    }
    metadata: dict[str, object] = {
        "models": {
            "taesd": backends.taesd.model_metadata(),
            "snac": backends.snac.model_metadata(),
            "t5": backends.t5.model_metadata(),
        },
        "runtime": {
            "ffmpeg_seconds": ffmpeg_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "environment": runtime_environment(
            str(getattr(backends.taesd, "_device", "injected")),
            ffmpeg=ffmpeg,
            torch=getattr(backends.taesd, "_torch", None),
        ),
        "projection_seeds": {
            "vision": VISION_PROJECTION_SEED,
            "audio": AUDIO_PROJECTION_SEED,
            "text": TEXT_PROJECTION_SEED,
        },
        "projection_digests": {
            "vision": projection_digest(vision_matrix),
            "audio": projection_digest(audio_matrix),
            "text": projection_digest(text_matrix),
        },
        "media": {
            "frame_rate": FRAME_RATE,
            "frame_height": FRAME_SIZE,
            "frame_width": FRAME_SIZE,
            "letterboxed": True,
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "audio_channels": 1,
        },
        "video_ids": list(sorted(expected_video_ids)),
        "window_counts": {
            video_id: sum(row.spec.video_id == video_id for row in data) for video_id in sorted(expected_video_ids)
        },
        "decoded_media": decoded_media,
    }
    return ExtractionResult(
        table=table,
        component_rows=component_rows,
        window_rows=window_rows,
        component_arrays=component_arrays,
        metadata=metadata,
    )


def extract_sample(cache_path: Path, hf_cache: Path, *, device: str = "cuda") -> ExtractionResult:
    """Run formal extraction over all eight integrity-checked sample videos."""

    annotations = dataset.validate_sample_cache(cache_path)
    specs = [
        spec
        for video_id in dataset.SAMPLE_VIDEO_IDS
        for spec in dataset.windows_for_video(video_id, annotations[video_id])
    ]
    backends = load_frozen_backends(hf_cache, device=device, local_files_only=True)
    return extract_with_backends(cache_path, specs, backends)


def _mean_component(rows: Sequence[Mapping[str, object]], name: str) -> float:
    values = np.asarray([float(cast(float, row[name])) for row in rows], dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"component metric {name!r} must be non-empty and finite")
    return float(np.mean(values))


def component_decision(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require each frozen control to win on at least six of eight videos."""

    if len(rows) != len(dataset.SAMPLE_VIDEO_IDS):
        raise ValueError(f"component decision requires {len(dataset.SAMPLE_VIDEO_IDS)} per-video rows")

    matched_vision = _mean_component(rows, "taesd_matched_mse")
    spatial_mean = _mean_component(rows, "taesd_spatial_mean_mse")
    half_cycle = _mean_component(rows, "taesd_half_cycle_mse")
    matched_audio = _mean_component(rows, "snac_matched_mse")
    cross_audio = _mean_component(rows, "snac_cross_video_mse")
    correct_text = _mean_component(rows, "t5_correct_target_nll")
    cross_text = _mean_component(rows, "t5_cross_video_target_nll")
    generation_finite = _mean_component(rows, "t5_generation_finite_rate")
    generation_parseable = _mean_component(rows, "t5_generation_parseable_rate")
    generation_max_tokens = max(int(cast(int, row["t5_generation_max_tokens"])) for row in rows)
    support = {
        "taesd_image_supporting_videos": sum(
            float(cast(float, row["taesd_matched_mse"])) < float(cast(float, row["taesd_spatial_mean_mse"]))
            for row in rows
        ),
        "taesd_framewise_video_supporting_videos": sum(
            float(cast(float, row["taesd_matched_mse"])) < float(cast(float, row["taesd_half_cycle_mse"]))
            for row in rows
        ),
        "snac_audio_supporting_videos": sum(
            float(cast(float, row["snac_matched_mse"])) < float(cast(float, row["snac_cross_video_mse"]))
            for row in rows
        ),
        "t5_text_supporting_videos": sum(
            float(cast(float, row["t5_correct_target_nll"])) < float(cast(float, row["t5_cross_video_target_nll"]))
            and float(cast(float, row["t5_generation_finite_rate"])) == 1.0
            and float(cast(float, row["t5_generation_parseable_rate"])) == 1.0
            and int(cast(int, row["t5_generation_max_tokens"])) <= 32
            for row in rows
        ),
    }
    decisions = {
        "taesd_image_pass": support["taesd_image_supporting_videos"] >= 6,
        "taesd_framewise_video_pass": support["taesd_framewise_video_supporting_videos"] >= 6,
        "snac_audio_pass": support["snac_audio_supporting_videos"] >= 6,
        "t5_text_pass": support["t5_text_supporting_videos"] >= 6,
    }
    return {
        **decisions,
        **support,
        "all_pass": all(decisions.values()),
        "taesd_matched_mse": matched_vision,
        "taesd_spatial_mean_mse": spatial_mean,
        "taesd_image_delta": spatial_mean - matched_vision,
        "taesd_half_cycle_mse": half_cycle,
        "taesd_framewise_video_delta": half_cycle - matched_vision,
        "snac_matched_mse": matched_audio,
        "snac_cross_video_mse": cross_audio,
        "snac_audio_delta": cross_audio - matched_audio,
        "t5_correct_target_nll": correct_text,
        "t5_cross_video_target_nll": cross_text,
        "t5_text_delta": cross_text - correct_text,
        "t5_generation_finite_rate": generation_finite,
        "t5_generation_parseable_rate": generation_parseable,
        "t5_generation_max_tokens": generation_max_tokens,
    }


__all__ = [
    "AUDIO_PROJECTION_SEED",
    "AUDIO_SAMPLE_RATE",
    "AudioBatch",
    "ExtractionResult",
    "FRAME_RATE",
    "FRAME_SIZE",
    "FrozenBackends",
    "MaskedSpan",
    "SNACBackend",
    "SNAC_MODEL_ID",
    "SNAC_REVISION",
    "T5Backend",
    "T5_MODEL_ID",
    "T5_REVISION",
    "TAESDBackend",
    "TAESD_MODEL_ID",
    "TAESD_REVISION",
    "TEXT_PROJECTION_SEED",
    "TextBatch",
    "VISION_PROJECTION_SEED",
    "VisionBatch",
    "build_masked_span",
    "audio_sample_bounds",
    "binary_identity",
    "component_decision",
    "crop_or_pad_audio",
    "cross_video_nearest_progress_indices",
    "decode_audio_mono_24k",
    "decode_video_frames",
    "extract_sample",
    "extract_with_backends",
    "frame_index_at",
    "generation_diagnostics",
    "inspect_backend_inputs",
    "inspect_media_inputs",
    "load_frozen_backends",
    "normalize_snac_codes",
    "pad_token_sequences",
    "per_example_teacher_forced_nll",
    "probe_media",
    "projection_digest",
    "projection_matrix",
    "runtime_environment",
    "taesd_post_map",
]
