"""Tests for immutable Perception Test metadata and causal sample alignment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bench.multimodal_preflight import dataset
from bench.multimodal_preflight.dataset import (
    AUDIO_HISTORY_SECONDS,
    DEFAULT_CACHE_PATH,
    DEVELOPMENT_VIDEO_ID,
    EXPECTED_WINDOW_COUNTS,
    EXTRACTED_MP4_SHA256,
    EXTRACTED_MP4_SIZE_BYTES,
    NO_EVENT_MARKER,
    OFFICIAL_SAMPLE_ANNOTATIONS_SHA256,
    OFFICIAL_SAMPLE_ANNOTATIONS_SIZE_BYTES,
    OFFICIAL_SAMPLE_ANNOTATIONS_URL,
    OFFICIAL_SAMPLE_JSON_SHA256,
    OFFICIAL_SAMPLE_JSON_SIZE_BYTES,
    OFFICIAL_SAMPLE_VIDEOS_SHA256,
    OFFICIAL_SAMPLE_VIDEOS_SIZE_BYTES,
    OFFICIAL_SAMPLE_VIDEOS_URL,
    SAMPLE_VIDEO_IDS,
    TIMESTAMP_STEP_SECONDS,
    VISUAL_TARGET_HORIZON_SECONDS,
    DatasetValidationError,
    WindowSpec,
    annotation_text_at,
    development_video_id,
    duration_seconds,
    formal_folds,
    generate_window_specs,
    validate_formal_window_specs,
    validate_frozen_file,
    validate_media_hashes,
    validate_sample_cache,
    validate_sample_json,
    windows_for_video,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_official_identity_constants_are_complete() -> None:
    assert OFFICIAL_SAMPLE_VIDEOS_URL.endswith("/zip_data/sample_videos.zip")
    assert OFFICIAL_SAMPLE_ANNOTATIONS_URL.endswith("/zip_data/sample_annotations.zip")
    assert OFFICIAL_SAMPLE_VIDEOS_SHA256 == "68e5cca8c3064859f273ec7a52c15d5eb4a337032d12e6d2f379079825a72537"
    assert OFFICIAL_SAMPLE_ANNOTATIONS_SHA256 == ("2d919a4b7154bf66a71ea3964be4b413c1995e6c90de54d3da9422f3ebfdfaea")
    assert OFFICIAL_SAMPLE_JSON_SHA256 == ("8d67bceda5a21f0e32919dd4631f4142d85eabe2fc92c03df118f2ef8978a8ce")
    assert OFFICIAL_SAMPLE_VIDEOS_SIZE_BYTES == 225_370_639
    assert OFFICIAL_SAMPLE_ANNOTATIONS_SIZE_BYTES == 2_305_618
    assert OFFICIAL_SAMPLE_JSON_SIZE_BYTES == 19_931_707
    assert SAMPLE_VIDEO_IDS == (
        "video_10993",
        "video_1580",
        "video_2564",
        "video_3501",
        "video_6860",
        "video_8241",
        "video_874",
        "video_9253",
    )
    assert tuple(sorted(EXTRACTED_MP4_SHA256)) == SAMPLE_VIDEO_IDS
    assert len(EXTRACTED_MP4_SHA256) == 8
    assert tuple(sorted(EXTRACTED_MP4_SIZE_BYTES)) == SAMPLE_VIDEO_IDS
    assert EXPECTED_WINDOW_COUNTS == {
        "video_10993": 63,
        "video_1580": 64,
        "video_2564": 59,
        "video_3501": 65,
        "video_6860": 65,
        "video_8241": 48,
        "video_874": 66,
        "video_9253": 47,
    }
    assert all(len(digest) == 64 for digest in EXTRACTED_MP4_SHA256.values())
    assert EXTRACTED_MP4_SHA256["video_9253"] == ("6dbe9e42d0351220c1e81ade592ad85f8d40f5112fb2744a8205012c41699270")
    assert DEFAULT_CACHE_PATH == Path.home() / ".cache" / "prospect" / "perception_test_sample"


def test_sample_json_and_media_hash_validation_use_synthetic_files(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations" / "sample.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(json.dumps({"video_a": {}, "video_b": {}}), encoding="utf-8")
    loaded = validate_sample_json(annotation_path, expected_ids=("video_b", "video_a"))
    assert tuple(sorted(loaded)) == ("video_a", "video_b")

    videos_path = tmp_path / "videos"
    videos_path.mkdir()
    payloads = {"video_a": b"synthetic-a", "video_b": b"synthetic-b"}
    for video_id, payload in payloads.items():
        (videos_path / f"{video_id}.mp4").write_bytes(payload)
    expected = {video_id: _sha256(payload) for video_id, payload in payloads.items()}
    sizes = {video_id: len(payload) for video_id, payload in payloads.items()}
    assert validate_media_hashes(tmp_path, expected_hashes=expected, expected_sizes=sizes) == expected
    wrong_sizes = {**sizes, "video_a": sizes["video_a"] + 1}
    with pytest.raises(DatasetValidationError, match="media size mismatch"):
        validate_media_hashes(tmp_path, expected_hashes=expected, expected_sizes=wrong_sizes)

    frozen = tmp_path / "frozen.bin"
    frozen.write_bytes(b"frozen-input")
    assert validate_frozen_file(frozen, _sha256(b"frozen-input")) == _sha256(b"frozen-input")
    with pytest.raises(DatasetValidationError, match="input hash mismatch"):
        validate_frozen_file(frozen, _sha256(b"different"))


def test_sample_cache_authenticates_archives_json_and_exact_video_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video_archive = b"video-archive"
    annotation_archive = b"annotation-archive"
    json_payload = json.dumps({"video_a": {}}).encode()
    media_payload = b"authenticated-mp4"
    (tmp_path / "sample_videos.zip").write_bytes(video_archive)
    (tmp_path / "sample_annotations.zip").write_bytes(annotation_archive)
    annotations_path = tmp_path / "annotations"
    annotations_path.mkdir()
    (annotations_path / "sample.json").write_bytes(json_payload)
    videos_path = tmp_path / "videos"
    videos_path.mkdir()
    (videos_path / "video_a.mp4").write_bytes(media_payload)

    monkeypatch.setattr(dataset, "SAMPLE_VIDEO_IDS", ("video_a",))
    monkeypatch.setattr(dataset, "OFFICIAL_SAMPLE_VIDEOS_SHA256", _sha256(video_archive))
    monkeypatch.setattr(dataset, "OFFICIAL_SAMPLE_ANNOTATIONS_SHA256", _sha256(annotation_archive))
    monkeypatch.setattr(dataset, "OFFICIAL_SAMPLE_JSON_SHA256", _sha256(json_payload))
    monkeypatch.setattr(dataset, "OFFICIAL_SAMPLE_VIDEOS_SIZE_BYTES", len(video_archive))
    monkeypatch.setattr(dataset, "OFFICIAL_SAMPLE_ANNOTATIONS_SIZE_BYTES", len(annotation_archive))
    monkeypatch.setattr(dataset, "OFFICIAL_SAMPLE_JSON_SIZE_BYTES", len(json_payload))
    monkeypatch.setattr(dataset, "EXTRACTED_MP4_SHA256", {"video_a": _sha256(media_payload)})
    monkeypatch.setattr(dataset, "EXTRACTED_MP4_SIZE_BYTES", {"video_a": len(media_payload)})

    assert validate_sample_cache(tmp_path) == {"video_a": {}}
    (videos_path / "extra.mp4").write_bytes(b"extra")
    with pytest.raises(DatasetValidationError, match="membership mismatch"):
        validate_sample_cache(tmp_path)


def test_identity_validation_rejects_wrong_ids_missing_media_and_wrong_hash(tmp_path: Path) -> None:
    annotation_path = tmp_path / "sample.json"
    annotation_path.write_text(json.dumps({"video_a": {}}), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="missing=.*video_b"):
        validate_sample_json(annotation_path, expected_ids=("video_a", "video_b"))

    videos_path = tmp_path / "videos"
    videos_path.mkdir()
    expected = {"video_a": _sha256(b"expected")}
    with pytest.raises(DatasetValidationError, match="membership mismatch"):
        validate_media_hashes(tmp_path, expected_hashes=expected)

    (videos_path / "video_a.mp4").write_bytes(b"tampered")
    with pytest.raises(DatasetValidationError, match="media hash mismatch"):
        validate_media_hashes(tmp_path, expected_hashes=expected)

    (videos_path / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="extra=.*extra.txt"):
        validate_media_hashes(tmp_path, expected_hashes=expected)


def test_causal_windows_use_past_audio_and_never_cross_duration() -> None:
    windows = generate_window_specs("video_x", 3.25)

    assert [window.frame_seconds for window in windows] == [1.0, 1.5]
    assert [window.target_seconds for window in windows] == [2.0, 2.5]
    assert all(window.annotation_text == NO_EVENT_MARKER for window in windows)
    for previous, current in zip(windows, windows[1:], strict=False):
        assert current.frame_seconds - previous.frame_seconds == TIMESTAMP_STEP_SECONDS
    for window in windows:
        assert window.audio_start_seconds == window.frame_seconds - AUDIO_HISTORY_SECONDS
        assert window.audio_end_seconds == window.frame_seconds
        assert window.target_seconds == window.frame_seconds + VISUAL_TARGET_HORIZON_SECONDS
        assert window.audio_end_seconds <= window.frame_seconds < window.target_seconds
        assert window.target_seconds + 0.5 <= window.duration_seconds

    assert generate_window_specs("video_x", 2.499) == ()
    exact = generate_window_specs("video_x", 2.5)
    assert len(exact) == 1
    assert exact[0].target_seconds + 0.5 == exact[0].duration_seconds


def test_window_spec_rejects_noncausal_or_out_of_range_timing() -> None:
    common: dict[str, object] = {
        "video_id": "video_x",
        "audio_start_seconds": 0.0,
        "audio_end_seconds": 1.0,
        "frame_seconds": 1.0,
        "target_seconds": 2.0,
        "duration_seconds": 2.5,
        "annotation_text": NO_EVENT_MARKER,
    }
    assert WindowSpec(**common).t_seconds == 1.0  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="end at the input frame"):
        WindowSpec(
            **{**common, "audio_start_seconds": 0.1, "audio_end_seconds": 1.1}  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="complete sampled-frame interval"):
        WindowSpec(**{**common, "duration_seconds": 2.49})  # type: ignore[arg-type]


def test_annotation_text_is_active_only_at_t_and_deterministic() -> None:
    current_action = {"label": "mixing", "timestamps": [500_000, 1_000_000]}
    duplicate_action = {"label": "mixing", "timestamps": [0, 2_000_000]}
    future_action = {"label": "pouring", "timestamps": [1_500_000, 2_500_000]}
    current_sound = {"label": "fluid", "timestamps": [750_000, 1_250_000]}
    future_sound = {"label": "impact", "timestamps": [1_000_001, 2_000_000]}
    annotations = {
        "action_localisation": [future_action, current_action, duplicate_action],
        "sound_localisation": [future_sound, current_sound],
    }

    expected = "action: mixing; sound: fluid."
    assert annotation_text_at(annotations, 1.0) == expected
    reversed_annotations = {
        "action_localisation": list(reversed(annotations["action_localisation"])),
        "sound_localisation": list(reversed(annotations["sound_localisation"])),
    }
    assert annotation_text_at(reversed_annotations, 1.0) == expected
    assert "pouring" not in annotation_text_at(annotations, 1.0)
    assert "impact" not in annotation_text_at(annotations, 1.0)
    assert annotation_text_at(annotations, 3.0) == NO_EVENT_MARKER

    whitespace_annotations = {
        "action_localisation": [{"label": "  hand\t  mixing  ", "timestamps": [0, 2_000_000]}],
        "sound_localisation": [{"label": "fluid\n flow", "timestamps": [0, 2_000_000]}],
    }
    assert annotation_text_at(whitespace_annotations, 1.0) == "action: hand mixing; sound: fluid flow."

    windows = generate_window_specs("video_x", 2.5, annotations=annotations)
    assert len(windows) == 1
    assert windows[0].annotation_text == expected


def test_metadata_duration_and_window_alignment() -> None:
    video_annotations: dict[str, Any] = {
        "metadata": {"video_id": "video_x", "num_frames": 90, "frame_rate": 30.0},
        "action_localisation": [{"label": "first", "timestamps": [0, 1_250_000]}],
        "sound_localisation": [],
    }
    assert duration_seconds(video_annotations["metadata"]) == 3.0
    windows = windows_for_video("video_x", video_annotations)
    assert [window.frame_seconds for window in windows] == [1.0, 1.5]
    assert windows[0].annotation_text == "action: first; sound: none."
    assert windows[1].annotation_text == NO_EVENT_MARKER
    assert windows[-1].target_seconds == 2.5


def test_frozen_formal_window_counts_and_order_are_exact() -> None:
    specs = [
        spec
        for video_id in SAMPLE_VIDEO_IDS
        for spec in generate_window_specs(
            video_id,
            2.5 + (EXPECTED_WINDOW_COUNTS[video_id] - 1) * TIMESTAMP_STEP_SECONDS,
        )
    ]
    validated = validate_formal_window_specs(specs)
    assert len(validated) == sum(EXPECTED_WINDOW_COUNTS.values()) == 477
    assert validated[0].video_id == SAMPLE_VIDEO_IDS[0]
    assert validated[-1].video_id == SAMPLE_VIDEO_IDS[-1]
    with pytest.raises(DatasetValidationError, match="order/count mismatch"):
        validate_formal_window_specs(specs[:-1])


def test_formal_folds_are_deterministic_disjoint_and_cover_all_ids() -> None:
    folds = formal_folds()
    assert folds == formal_folds()
    assert [fold.index for fold in folds] == [0, 1, 2, 3]
    assert [fold.test_ids for fold in folds] == [
        SAMPLE_VIDEO_IDS[0:2],
        SAMPLE_VIDEO_IDS[2:4],
        SAMPLE_VIDEO_IDS[4:6],
        SAMPLE_VIDEO_IDS[6:8],
    ]

    held_out: list[str] = []
    for fold in folds:
        assert len(fold.train_ids) == 6
        assert len(fold.test_ids) == 2
        assert set(fold.train_ids).isdisjoint(fold.test_ids)
        assert set(fold.train_ids) | set(fold.test_ids) == set(SAMPLE_VIDEO_IDS)
        held_out.extend(fold.test_ids)
    assert tuple(sorted(held_out)) == SAMPLE_VIDEO_IDS


def test_development_id_is_a_convenience_not_a_formal_exclusion() -> None:
    assert development_video_id() == DEVELOPMENT_VIDEO_ID == "video_9253"
    assert DEVELOPMENT_VIDEO_ID in SAMPLE_VIDEO_IDS
    assert sum(DEVELOPMENT_VIDEO_ID in fold.test_ids for fold in formal_folds()) == 1
    assert sum(DEVELOPMENT_VIDEO_ID in fold.train_ids for fold in formal_folds()) == 3
