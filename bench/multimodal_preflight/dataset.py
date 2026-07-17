"""Perception Test sample metadata, integrity, and causal alignment helpers.

This module deliberately performs no downloads and imports no media or learning
libraries.  It treats the official Perception Test sample as immutable input,
validates its identity, and describes causal windows for downstream decoders.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

OFFICIAL_SAMPLE_VIDEOS_URL: Final = "https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip"
OFFICIAL_SAMPLE_ANNOTATIONS_URL: Final = (
    "https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip"
)
OFFICIAL_SAMPLE_VIDEOS_SHA256: Final = "68e5cca8c3064859f273ec7a52c15d5eb4a337032d12e6d2f379079825a72537"
OFFICIAL_SAMPLE_ANNOTATIONS_SHA256: Final = "2d919a4b7154bf66a71ea3964be4b413c1995e6c90de54d3da9422f3ebfdfaea"
OFFICIAL_SAMPLE_JSON_SHA256: Final = "8d67bceda5a21f0e32919dd4631f4142d85eabe2fc92c03df118f2ef8978a8ce"
OFFICIAL_SAMPLE_VIDEOS_SIZE_BYTES: Final = 225_370_639
OFFICIAL_SAMPLE_ANNOTATIONS_SIZE_BYTES: Final = 2_305_618
OFFICIAL_SAMPLE_JSON_SIZE_BYTES: Final = 19_931_707

EXTRACTED_MP4_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "video_10993": "9ccfbf39dade7e6108b610e37eb7d478a3b5ad5463d169dfd3ad69dbbfd1921b",
        "video_1580": "3650937f112216580776780a6b3bf8e11587bfe2bc526c5348128383eaec5cd1",
        "video_2564": "43aa9f4c7eec12c0599c915fc3e6fe29b8825bc1c9d09ebe5ad96d4082b55950",
        "video_3501": "8de619c6bc3ae7b7f676b3d33fba85b954a69a544c4245bf3b3282ec380a4d16",
        "video_6860": "97224e8f3990e173c47fd7c82ab1094d3cfd27b9726985c2cc26fe2022ee60ee",
        "video_8241": "81ff2f3f93af9ba5b7b23e21b4c6daf8b9a7e4931df9fc947eaa4d6349b59345",
        "video_874": "208eb2914766296d4af10361c3683381d330480b9edb5fbff20238740af18a0e",
        "video_9253": "6dbe9e42d0351220c1e81ade592ad85f8d40f5112fb2744a8205012c41699270",
    }
)
EXTRACTED_MP4_SIZE_BYTES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "video_10993": 91_078_692,
        "video_1580": 20_523_562,
        "video_2564": 36_790_707,
        "video_3501": 18_717_639,
        "video_6860": 22_082_352,
        "video_8241": 15_780_090,
        "video_874": 6_377_635,
        "video_9253": 13_228_873,
    }
)
SAMPLE_VIDEO_IDS: Final = tuple(sorted(EXTRACTED_MP4_SHA256))
EXPECTED_WINDOW_COUNTS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "video_10993": 63,
        "video_1580": 64,
        "video_2564": 59,
        "video_3501": 65,
        "video_6860": 65,
        "video_8241": 48,
        "video_874": 66,
        "video_9253": 47,
    }
)
DEFAULT_CACHE_PATH: Final = Path.home() / ".cache" / "prospect" / "perception_test_sample"
DEVELOPMENT_VIDEO_ID: Final = "video_9253"

TIMESTAMP_STEP_SECONDS: Final = 0.5
AUDIO_HISTORY_SECONDS: Final = 1.0
VISUAL_TARGET_HORIZON_SECONDS: Final = 1.0
VISUAL_SAMPLE_INTERVAL_SECONDS: Final = 0.5
NO_EVENT_MARKER: Final = "action: none; sound: none."


class DatasetValidationError(ValueError):
    """The staged sample does not match the expected immutable dataset."""


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """One causal multimodal prediction window anchored at ``frame_seconds``.

    Audio is restricted to the causal half-open interval ``[t - 1, t)``.  The input
    frame and annotation text are evaluated at ``t``; the visual prediction
    target is evaluated at ``t + 1``.
    """

    video_id: str
    audio_start_seconds: float
    audio_end_seconds: float
    frame_seconds: float
    target_seconds: float
    duration_seconds: float
    annotation_text: str

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must be non-empty")
        times = (
            self.audio_start_seconds,
            self.audio_end_seconds,
            self.frame_seconds,
            self.target_seconds,
            self.duration_seconds,
        )
        if not all(math.isfinite(value) for value in times):
            raise ValueError("window times must be finite")
        if self.audio_start_seconds < 0.0:
            raise ValueError("audio history cannot start before the video")
        if not math.isclose(
            self.audio_end_seconds - self.audio_start_seconds,
            AUDIO_HISTORY_SECONDS,
            abs_tol=1e-12,
        ):
            raise ValueError("audio history must be exactly one second")
        if not math.isclose(self.audio_end_seconds, self.frame_seconds, abs_tol=1e-12):
            raise ValueError("audio history must end at the input frame")
        if not math.isclose(
            self.target_seconds - self.frame_seconds,
            VISUAL_TARGET_HORIZON_SECONDS,
            abs_tol=1e-12,
        ):
            raise ValueError("visual target must be exactly one second after the input frame")
        if self.target_seconds + VISUAL_SAMPLE_INTERVAL_SECONDS > self.duration_seconds + 1e-12:
            raise ValueError("visual target must have a complete sampled-frame interval before duration")
        if not self.annotation_text:
            raise ValueError("annotation_text must be non-empty")

    @property
    def t_seconds(self) -> float:
        """Alias for the causal anchor timestamp."""

        return self.frame_seconds


@dataclass(frozen=True, slots=True)
class DatasetFold:
    """One deterministic held-out-video fold."""

    index: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("fold index must be non-negative")
        if len(self.test_ids) != 2:
            raise ValueError("each fold must have exactly two test videos")
        if len(self.train_ids) != 6:
            raise ValueError("each fold must have exactly six train videos")
        if set(self.train_ids) & set(self.test_ids):
            raise ValueError("fold train and test IDs must be disjoint")
        if tuple(sorted(self.train_ids)) != self.train_ids or tuple(sorted(self.test_ids)) != self.test_ids:
            raise ValueError("fold IDs must be lexicographically sorted")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of ``path`` without loading it whole."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_json(
    path: str | Path,
    *,
    expected_ids: Iterable[str] = SAMPLE_VIDEO_IDS,
) -> dict[str, dict[str, Any]]:
    """Load annotations and require exactly the expected top-level video IDs."""

    annotation_path = Path(path)
    try:
        payload = annotation_path.read_bytes()
        loaded = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"cannot load annotations at {annotation_path}: {error}") from error
    return _validate_loaded_sample_json(loaded, expected_ids=expected_ids)


def _validate_loaded_sample_json(loaded: object, *, expected_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(loaded, dict):
        raise DatasetValidationError("sample annotations must be a top-level JSON object")

    expected = tuple(sorted(expected_ids))
    actual = tuple(sorted(loaded))
    if len(expected) != len(set(expected)):
        raise ValueError("expected_ids must be unique")
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise DatasetValidationError(f"annotation video IDs mismatch: missing={missing}, extra={extra}")
    if not all(isinstance(video_id, str) and isinstance(value, dict) for video_id, value in loaded.items()):
        raise DatasetValidationError("each annotation video ID must map to a JSON object")
    return cast(dict[str, dict[str, Any]], loaded)


def validate_media_hashes(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    *,
    expected_hashes: Mapping[str, str] = EXTRACTED_MP4_SHA256,
    expected_sizes: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Validate exact directory membership, sizes, and every extracted digest."""

    videos_path = Path(cache_path) / "videos"
    if not videos_path.is_dir():
        raise DatasetValidationError(f"missing videos directory: {videos_path}")
    expected_names = {f"{video_id}.mp4" for video_id in expected_hashes}
    actual_names = {path.name for path in videos_path.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise DatasetValidationError(f"videos directory membership mismatch: missing={missing}, extra={extra}")
    if expected_sizes is not None and set(expected_sizes) != set(expected_hashes):
        raise ValueError("expected_sizes IDs must exactly match expected_hashes IDs")
    observed: dict[str, str] = {}
    for video_id, expected_digest in sorted(expected_hashes.items()):
        media_path = videos_path / f"{video_id}.mp4"
        if media_path.is_symlink() or not media_path.is_file():
            raise DatasetValidationError(f"missing media file: {media_path}")
        if expected_sizes is not None and media_path.stat().st_size != expected_sizes[video_id]:
            raise DatasetValidationError(
                f"media size mismatch for {video_id}: expected {expected_sizes[video_id]}, "
                f"got {media_path.stat().st_size}"
            )
        actual_digest = sha256_file(media_path)
        if actual_digest != expected_digest.lower():
            raise DatasetValidationError(
                f"media hash mismatch for {video_id}: expected {expected_digest.lower()}, got {actual_digest}"
            )
        observed[video_id] = actual_digest
    return observed


def validate_frozen_file(path: str | Path, expected_sha256: str, *, expected_size_bytes: int | None = None) -> str:
    """Require one staged external input to match its frozen SHA-256 digest."""

    file_path = Path(path)
    if file_path.is_symlink() or not file_path.is_file():
        raise DatasetValidationError(f"missing frozen input: {file_path}")
    if expected_size_bytes is not None and file_path.stat().st_size != expected_size_bytes:
        raise DatasetValidationError(
            f"input size mismatch for {file_path}: expected {expected_size_bytes}, got {file_path.stat().st_size}"
        )
    observed = sha256_file(file_path)
    if observed != expected_sha256.lower():
        raise DatasetValidationError(
            f"input hash mismatch for {file_path}: expected {expected_sha256.lower()}, got {observed}"
        )
    return observed


def load_sample_annotations(cache_path: str | Path = DEFAULT_CACHE_PATH) -> dict[str, dict[str, Any]]:
    """Authenticate and parse the exact same official JSON byte payload once."""

    path = Path(cache_path) / "annotations" / "sample.json"
    if path.is_symlink():
        raise DatasetValidationError(f"authenticated annotations cannot be a symlink: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DatasetValidationError(f"cannot load annotations at {path}: {error}") from error
    if len(payload) != OFFICIAL_SAMPLE_JSON_SIZE_BYTES:
        raise DatasetValidationError(
            f"input size mismatch for {path}: expected {OFFICIAL_SAMPLE_JSON_SIZE_BYTES}, got {len(payload)}"
        )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != OFFICIAL_SAMPLE_JSON_SHA256:
        raise DatasetValidationError(
            f"input hash mismatch for {path}: expected {OFFICIAL_SAMPLE_JSON_SHA256}, got {observed}"
        )
    try:
        loaded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"cannot load annotations at {path}: {error}") from error
    return _validate_loaded_sample_json(loaded, expected_ids=SAMPLE_VIDEO_IDS)


def validate_sample_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> dict[str, dict[str, Any]]:
    """Validate the official annotation identity and all eight extracted MP4s."""

    root = Path(cache_path)
    validate_frozen_file(
        root / "sample_videos.zip",
        OFFICIAL_SAMPLE_VIDEOS_SHA256,
        expected_size_bytes=OFFICIAL_SAMPLE_VIDEOS_SIZE_BYTES,
    )
    validate_frozen_file(
        root / "sample_annotations.zip",
        OFFICIAL_SAMPLE_ANNOTATIONS_SHA256,
        expected_size_bytes=OFFICIAL_SAMPLE_ANNOTATIONS_SIZE_BYTES,
    )
    annotations = load_sample_annotations(root)
    validate_media_hashes(
        root,
        expected_hashes=EXTRACTED_MP4_SHA256,
        expected_sizes=EXTRACTED_MP4_SIZE_BYTES,
    )
    return annotations


def duration_seconds(metadata: Mapping[str, Any]) -> float:
    """Derive visual duration from the official ``num_frames / frame_rate`` metadata."""

    try:
        num_frames_raw = metadata["num_frames"]
        frame_rate_raw = metadata["frame_rate"]
    except KeyError as error:
        raise DatasetValidationError(f"metadata is missing {error.args[0]!r}") from error
    if isinstance(num_frames_raw, bool) or not isinstance(num_frames_raw, (int, float)):
        raise DatasetValidationError("num_frames must be numeric")
    if isinstance(frame_rate_raw, bool) or not isinstance(frame_rate_raw, (int, float)):
        raise DatasetValidationError("frame_rate must be numeric")
    num_frames = float(num_frames_raw)
    frame_rate = float(frame_rate_raw)
    if not math.isfinite(num_frames) or num_frames <= 0.0:
        raise DatasetValidationError("num_frames must be finite and positive")
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise DatasetValidationError("frame_rate must be finite and positive")
    return num_frames / frame_rate


def _active_labels(
    video_annotations: Mapping[str, Any],
    field: str,
    kind: str,
    timestamp_us: int,
) -> set[tuple[str, str]]:
    raw_segments = video_annotations.get(field, ())
    if not isinstance(raw_segments, (list, tuple)):
        raise DatasetValidationError(f"{field} must be an array")
    active: set[tuple[str, str]] = set()
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            raise DatasetValidationError(f"{field} entries must be objects")
        raw_interval = segment.get("timestamps")
        label = segment.get("label")
        if (
            not isinstance(raw_interval, (list, tuple))
            or len(raw_interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_interval)
        ):
            raise DatasetValidationError(f"{field} timestamps must be numeric [start_us, end_us] pairs")
        start_us, end_us = (int(raw_interval[0]), int(raw_interval[1]))
        if start_us > end_us:
            raise DatasetValidationError(f"{field} interval starts after it ends")
        if not isinstance(label, str) or not label.split():
            raise DatasetValidationError(f"{field} label must be a non-empty string")
        if start_us <= timestamp_us < end_us:
            active.add((kind, " ".join(label.split())))
    return active


def annotation_text_at(video_annotations: Mapping[str, Any], t_seconds: float) -> str:
    """Return deterministic action/sound text active at ``t`` only.

    Labels that start after ``t`` are never included, even when they are active at
    the visual target timestamp.  Duplicate labels are collapsed and the result is
    sorted by annotation kind and label, making it independent of JSON array order.
    """

    if not math.isfinite(t_seconds) or t_seconds < 0.0:
        raise ValueError("t_seconds must be finite and non-negative")
    timestamp_us = int(round(t_seconds * 1_000_000.0))
    actions = sorted(
        label for _, label in _active_labels(video_annotations, "action_localisation", "action", timestamp_us)
    )
    sounds = sorted(
        label for _, label in _active_labels(video_annotations, "sound_localisation", "sound", timestamp_us)
    )
    action_text = " | ".join(actions) if actions else "none"
    sound_text = " | ".join(sounds) if sounds else "none"
    return f"action: {action_text}; sound: {sound_text}."


def generate_window_specs(
    video_id: str,
    media_duration_seconds: float,
    *,
    annotations: Mapping[str, Any] | None = None,
) -> tuple[WindowSpec, ...]:
    """Generate half-second causal anchors that have full past and future context."""

    if not video_id:
        raise ValueError("video_id must be non-empty")
    duration = float(media_duration_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("media_duration_seconds must be finite and positive")

    windows: list[WindowSpec] = []
    index = 0
    while True:
        frame_seconds = AUDIO_HISTORY_SECONDS + index * TIMESTAMP_STEP_SECONDS
        target_seconds = frame_seconds + VISUAL_TARGET_HORIZON_SECONDS
        if target_seconds + VISUAL_SAMPLE_INTERVAL_SECONDS > duration + 1e-12:
            break
        annotation_text = NO_EVENT_MARKER if annotations is None else annotation_text_at(annotations, frame_seconds)
        windows.append(
            WindowSpec(
                video_id=video_id,
                audio_start_seconds=frame_seconds - AUDIO_HISTORY_SECONDS,
                audio_end_seconds=frame_seconds,
                frame_seconds=frame_seconds,
                target_seconds=target_seconds,
                duration_seconds=duration,
                annotation_text=annotation_text,
            )
        )
        index += 1
    return tuple(windows)


def windows_for_video(video_id: str, video_annotations: Mapping[str, Any]) -> tuple[WindowSpec, ...]:
    """Generate aligned windows directly from one official per-video annotation object."""

    metadata = video_annotations.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DatasetValidationError("video annotations must contain a metadata object")
    metadata_video_id = metadata.get("video_id")
    if metadata_video_id is not None and metadata_video_id != video_id:
        raise DatasetValidationError(f"metadata video_id {metadata_video_id!r} does not match {video_id!r}")
    return generate_window_specs(
        video_id,
        duration_seconds(metadata),
        annotations=video_annotations,
    )


def validate_formal_window_specs(specs: Iterable[WindowSpec]) -> tuple[WindowSpec, ...]:
    """Require the frozen video order, counts, and half-second timestamps exactly."""

    ordered = tuple(sorted(specs, key=lambda spec: (spec.video_id, spec.frame_seconds)))
    expected_pairs = tuple(
        (video_id, AUDIO_HISTORY_SECONDS + index * TIMESTAMP_STEP_SECONDS)
        for video_id in SAMPLE_VIDEO_IDS
        for index in range(EXPECTED_WINDOW_COUNTS[video_id])
    )
    actual_pairs = tuple((spec.video_id, spec.frame_seconds) for spec in ordered)
    if actual_pairs != expected_pairs:
        actual_counts = {video_id: sum(spec.video_id == video_id for spec in ordered) for video_id in SAMPLE_VIDEO_IDS}
        raise DatasetValidationError(
            f"formal window order/count mismatch: expected={dict(EXPECTED_WINDOW_COUNTS)}, actual={actual_counts}"
        )
    return ordered


def formal_folds() -> tuple[DatasetFold, ...]:
    """Return four fixed two-video test folds over all official sample IDs."""

    all_ids = SAMPLE_VIDEO_IDS
    folds: list[DatasetFold] = []
    for index in range(4):
        test_ids = all_ids[2 * index : 2 * index + 2]
        test_set = set(test_ids)
        train_ids = tuple(video_id for video_id in all_ids if video_id not in test_set)
        folds.append(DatasetFold(index=index, train_ids=train_ids, test_ids=test_ids))
    return tuple(folds)


def development_video_id() -> str:
    """Return the single convenience ID for smoke development, never formal evaluation."""

    return DEVELOPMENT_VIDEO_ID


__all__ = [
    "AUDIO_HISTORY_SECONDS",
    "DEFAULT_CACHE_PATH",
    "DEVELOPMENT_VIDEO_ID",
    "DatasetFold",
    "DatasetValidationError",
    "EXTRACTED_MP4_SHA256",
    "EXTRACTED_MP4_SIZE_BYTES",
    "EXPECTED_WINDOW_COUNTS",
    "NO_EVENT_MARKER",
    "OFFICIAL_SAMPLE_ANNOTATIONS_SHA256",
    "OFFICIAL_SAMPLE_ANNOTATIONS_SIZE_BYTES",
    "OFFICIAL_SAMPLE_ANNOTATIONS_URL",
    "OFFICIAL_SAMPLE_JSON_SHA256",
    "OFFICIAL_SAMPLE_JSON_SIZE_BYTES",
    "OFFICIAL_SAMPLE_VIDEOS_SHA256",
    "OFFICIAL_SAMPLE_VIDEOS_SIZE_BYTES",
    "OFFICIAL_SAMPLE_VIDEOS_URL",
    "SAMPLE_VIDEO_IDS",
    "TIMESTAMP_STEP_SECONDS",
    "VISUAL_TARGET_HORIZON_SECONDS",
    "VISUAL_SAMPLE_INTERVAL_SECONDS",
    "WindowSpec",
    "annotation_text_at",
    "development_video_id",
    "duration_seconds",
    "formal_folds",
    "generate_window_specs",
    "load_sample_annotations",
    "sha256_file",
    "validate_media_hashes",
    "validate_frozen_file",
    "validate_formal_window_specs",
    "validate_sample_cache",
    "validate_sample_json",
    "windows_for_video",
]
