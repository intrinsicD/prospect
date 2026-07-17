"""Scientific-method tests for MM-003 projection/scale isolation."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bench.multimodal_preflight import core, dataset
from bench.multimodal_transform_diagnostics import method


def _formal_raw_table() -> method.VisualTable:
    video_ids: list[str] = []
    timestamps: list[float] = []
    current: list[np.ndarray] = []
    target: list[np.ndarray] = []
    axis = np.linspace(-0.5, 0.5, 256)
    for video_index, video_id in enumerate(dataset.SAMPLE_VIDEO_IDS):
        count = dataset.EXPECTED_WINDOW_COUNTS[video_id]
        for row in range(count):
            time = 1.0 + 0.5 * row
            base = np.sin(axis * (row + 1)) + 0.05 * video_index
            delta = 0.02 * np.cos(axis * (row + 2))
            video_ids.append(video_id)
            timestamps.append(time)
            current.append(base)
            target.append(base + delta)
    return method.VisualTable(
        video_ids=np.asarray(video_ids),
        timestamps=np.asarray(timestamps),
        current=np.asarray(current),
        target=np.asarray(target),
    )


def _projection() -> np.ndarray:
    rng = np.random.default_rng(12_001)
    signs = rng.integers(0, 2, size=(256, 32), dtype=np.int8)
    return np.asarray((2.0 * signs.astype(float) - 1.0) / np.sqrt(256), dtype=float)


def test_raw_alignment_and_matched_panel_are_exact() -> None:
    table = _formal_raw_table()
    projection = _projection()
    projected_current = table.current @ projection
    projected_target = table.target @ projection

    alignment = method.alignment_record(
        table,
        projection,
        projected_current,
        projected_target,
    )
    matched = method.matched_table(table)

    assert alignment["rows"] == 477
    assert alignment["current_max_abs_error"] == 0.0
    assert len(matched.video_ids) == 461
    assert {
        video_id: int(np.sum(matched.video_ids == video_id)) for video_id in dataset.SAMPLE_VIDEO_IDS
    } == method.MATCHED_COUNTS


def test_transform_pairs_preserve_declared_information_and_ignore_test_values() -> None:
    table = _formal_raw_table()
    projection = _projection()
    fold = dataset.formal_folds()[0]
    train = table.subset(fold.train_ids)

    native = method.fit_transform("r32_native", train.current, projection)
    postz = method.fit_transform("r32_postz", train.current, projection)
    qr = method.fit_transform("r32_qr_postz", train.current, projection)
    pca = method.fit_transform("pca32_postz", train.current, projection)

    np.testing.assert_allclose(
        native.apply(table.current),
        table.current @ projection,
        rtol=0.0,
        atol=0.0,
    )
    assert qr.qr_projector_max_abs_error is not None
    assert qr.qr_projector_max_abs_error <= method.QR_PROJECTOR_TOLERANCE
    assert pca.projection is not None and pca.projection.shape == (256, 32)
    np.testing.assert_allclose(pca.projection.T @ pca.projection, np.eye(32), atol=1e-12)

    # Native and post-z are related by an exactly invertible affine map.
    reconstructed_native = postz.apply(table.current) * postz.output_scale + postz.output_mean
    np.testing.assert_allclose(reconstructed_native, native.apply(table.current), atol=1e-12)

    # Held-out mutations cannot enter a fit API that receives training current rows only.
    mutated = table.current.copy()
    test_mask = np.isin(table.video_ids, fold.test_ids)
    mutated[test_mask] += 1e6
    refit = method.fit_transform("pca32_postz", mutated[~test_mask], projection)
    assert refit.fingerprint() == pca.fingerprint()


@pytest.mark.parametrize("predictor", method.PREDICTOR_IDS)
def test_scale_neutral_probe_is_invariant_for_fixed_subspace_scale_pair(
    predictor: str,
) -> None:
    table = method.matched_table(_formal_raw_table())
    projection = _projection()
    fold = dataset.formal_folds()[0]
    full_train = _formal_raw_table().subset(fold.train_ids)
    native = method.fit_transform("r32_native", full_train.current, projection)
    postz = method.fit_transform("r32_postz", full_train.current, projection)

    native_rows = method._probe_rows_for("r32_native", predictor, fold, table, native)
    postz_rows = method._probe_rows_for("r32_postz", predictor, fold, table, postz)

    for left, right in zip(native_rows, postz_rows, strict=True):
        for metric in method.PROBE_METRICS:
            assert left[metric] == pytest.approx(
                right[metric],
                rel=method.PROBE_INVARIANCE_RTOL,
                abs=method.PROBE_INVARIANCE_ATOL,
            )


def test_fit_all_transforms_records_train_only_forensic_bindings() -> None:
    table = _formal_raw_table()
    transforms, records, arrays = method.fit_all_transforms(table, _projection())

    assert len(transforms) == 4 * 6
    assert len(records) == 4 * 6
    assert arrays
    for record in records:
        fold = dataset.formal_folds()[record["fold"]]
        assert record["train_video_ids"] == list(fold.train_ids)
        assert record["excluded_video_ids"] == list(fold.test_ids)
        assert set(record["train_video_ids"]).isdisjoint(record["excluded_video_ids"])
        assert len(record["fit_identity_sha256"]) == 64
        assert len(record["fit_matrix_sha256"]) == 64
        assert len(record["transform_fingerprint"]) == 64


def test_world_trajectory_checkpoint_copy_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _formal_raw_table()
    projection = _projection()
    fold = dataset.formal_folds()[0]
    transform = method.fit_transform("r32_postz", table.subset(fold.train_ids).current, projection)
    train = method.transformed_table(table, transform).subset(fold.train_ids)
    monkeypatch.setattr(method, "CHECKPOINTS", (2, 4))

    first = method._fit_world_trajectory(train, 0, shuffled=False)
    second = method._fit_world_trajectory(train, 0, shuffled=False)

    assert set(first) == {2, 4}
    assert core.model_fingerprint(first[2]) == core.model_fingerprint(second[2])
    assert core.model_fingerprint(first[4]) == core.model_fingerprint(second[4])
    assert core.model_fingerprint(first[2]) != core.model_fingerprint(first[4])


def test_transform_fingerprint_binds_every_parameter_array() -> None:
    table = _formal_raw_table()
    transform = method.fit_transform("pca32_postz", table.current, _projection())
    original = transform.fingerprint()
    altered_projection = np.asarray(transform.projection).copy()
    altered_projection[0, 0] += 1e-9
    altered = method.FittedTransform(
        representation_id=transform.representation_id,
        input_mean=transform.input_mean,
        input_scale=transform.input_scale,
        projection=altered_projection,
        output_mean=transform.output_mean,
        output_scale=transform.output_scale,
        spectrum=transform.spectrum,
        qr_projector_max_abs_error=transform.qr_projector_max_abs_error,
        pca_retained_variance_fraction=transform.pca_retained_variance_fraction,
        pca_rank_below_32=transform.pca_rank_below_32,
        pca_boundary_degenerate=transform.pca_boundary_degenerate,
        pca_32_to_1_ratio=transform.pca_32_to_1_ratio,
        pca_32_33_relative_gap=transform.pca_32_33_relative_gap,
    )

    assert original != altered.fingerprint()
    assert len(original) == hashlib.sha256().digest_size * 2


def _branch_inputs() -> tuple[
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
    dict[tuple[str, str], bool],
]:
    probes = {spec.representation_id: False for spec in method.REPRESENTATIONS}
    worlds = {rep: False for rep in method.WORLD_REPRESENTATION_IDS}
    healthy = {rep: True for rep in method.WORLD_REPRESENTATION_IDS}
    contrasts = {
        ("r32_postz", "r32_native"): False,
        ("r32_qr_postz", "r32_postz"): False,
        ("pca32_postz", "r32_qr_postz"): False,
    }
    return probes, worlds, healthy, contrasts


def test_information_loss_branch_requires_every_fixed_subspace_probe_to_fail() -> None:
    probes, worlds, healthy, contrasts = _branch_inputs()
    probes["r32_native"] = True
    probes["pca32_postz"] = True
    probes["raw256_postz"] = True

    labels, primary, _ = method._diagnostic_decision(
        probes,
        worlds,
        healthy,
        contrasts,
        pca_stable=True,
    )

    assert "fixed_random_subspace_linear_signal_loss_supported" not in labels
    assert "tested_32d_compression_linear_signal_loss_supported" not in labels
    assert "fixed_random_subspace_information_loss_supported" not in primary


def test_world_path_label_requires_support_in_the_same_probe_passing_arm() -> None:
    probes, worlds, healthy, contrasts = _branch_inputs()
    probes["r32_postz"] = True
    worlds["pca32_postz"] = True  # Unrelated arm has no linear-probe support.

    labels, _, classification = method._diagnostic_decision(
        probes,
        worlds,
        healthy,
        contrasts,
        pca_stable=True,
    )

    assert "linear_temporal_signal_present_world_path_not_supported" in labels
    assert classification == "linear_temporal_signal_present_world_path_not_supported"


def test_unhealthy_gate_crossing_is_labeled_and_cannot_be_a_rescue() -> None:
    probes, worlds, healthy, contrasts = _branch_inputs()
    worlds["r32_postz"] = True
    healthy["r32_postz"] = False
    contrasts[("r32_postz", "r32_native")] = True

    labels, primary, _ = method._diagnostic_decision(
        probes,
        worlds,
        healthy,
        contrasts,
        pca_stable=True,
    )

    assert "apparent_rescue_via_representation_collapse" in labels
    assert "mm001_coordinate_scale_cause_supported" not in primary


def test_same_subspace_scale_rescue_yields_primary_fix() -> None:
    probes, worlds, healthy, contrasts = _branch_inputs()
    worlds["r32_postz"] = True
    healthy["r32_native"] = False
    contrasts[("r32_postz", "r32_native")] = True

    _, primary, classification = method._diagnostic_decision(
        probes,
        worlds,
        healthy,
        contrasts,
        pca_stable=True,
    )

    assert primary == ["mm001_coordinate_scale_cause_supported"]
    assert classification == "mm001_coordinate_scale_cause_supported"


def test_pca_information_cause_requires_stable_differential_probe_and_world_rescue() -> None:
    probes, worlds, healthy, contrasts = _branch_inputs()
    probes["pca32_postz"] = True
    probes["raw256_postz"] = True
    worlds["pca32_postz"] = True
    contrasts[("pca32_postz", "r32_qr_postz")] = True

    labels, primary, classification = method._diagnostic_decision(
        probes,
        worlds,
        healthy,
        contrasts,
        pca_stable=True,
    )

    assert "fixed_random_subspace_linear_signal_loss_supported" in labels
    assert primary == ["fixed_random_subspace_information_loss_supported"]
    assert classification == "fixed_random_subspace_information_loss_supported"


def test_full_information_label_is_suppressed_by_admissible_scale_fix() -> None:
    probes, worlds, healthy, contrasts = _branch_inputs()
    probes["raw256_postz"] = True
    worlds["r32_postz"] = True
    contrasts[("r32_postz", "r32_native")] = True

    labels, primary, classification = method._diagnostic_decision(
        probes,
        worlds,
        healthy,
        contrasts,
        pca_stable=True,
    )

    assert primary == ["mm001_coordinate_scale_cause_supported"]
    assert "full_information_signal_present_no_compatible_32d_fix" not in labels
    assert classification == "mm001_coordinate_scale_cause_supported"
