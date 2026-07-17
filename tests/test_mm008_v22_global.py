from __future__ import annotations

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact

CONFIG_SHA256 = "0" * 64


def _source() -> np.ndarray:
    index = np.arange(3 * 64 * 64, dtype=np.float64).reshape(3, 64, 64)
    return np.ascontiguousarray(
        np.sin(index * 0.017) + np.cos(index * 0.0031) + 0.2 * np.sin(index * 0.111),
        dtype=np.float64,
    )


def _central_target(source: np.ndarray, theta: tuple[float, ...], gains: np.ndarray, biases: np.ndarray) -> np.ndarray:
    transformed = geometry.sample_scalar(source, theta, geometry.FULL_MASK)
    target = source.copy()
    coords = geometry.GEOMETRY.coords.astype(np.intp)
    target[:, coords[:, 0], coords[:, 1]] = gains[:, None] * transformed + biases[:, None]
    return fitting.target_values(target, geometry.FULL_MASK)


def test_affine_identity_complete_grid_has_stable_certificate() -> None:
    source = _source()
    target = fitting.target_values(source.copy(), geometry.FULL_MASK)
    result = exact.fit_global(
        source,
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "affine",
        context_key="kat/identity/full",
        config_sha256=CONFIG_SHA256,
    )

    assert result.selected.state_index == geometry.state_index((0.0,) * 6)
    assert result.selected.objective == 0.0
    assert result.certificate.candidate_count == 15_625
    assert result.certificate.admissible_count == 2_809
    assert result.certificate.inadmissible_count == 12_816
    assert result.certificate.exact_tie_multiplicity == 1
    assert result.certificate.scalar_replay_bit_exact
    assert np.array_equal(result.prediction, target)
    assert result.source_grid.content_sha256 == "b971c5eb0e7467bf72078965b81a3b05381ec59e565f3027ed5839e20e967da3"
    assert result.objective_cache.content_sha256 == "d2cc544c108e51a9825194b072352a28a5681f954719f814cafc5123586fea54"
    exact.validate_global_result(
        result,
        source,
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        config_sha256=CONFIG_SHA256,
    )


def test_combined_truth_recovery_and_cross_arm_source_reuse() -> None:
    source = _source()
    truth = (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0)
    target = _central_target(
        source,
        truth,
        np.asarray((1.2, 0.8, 1.4), dtype=np.float64),
        np.asarray((0.3, -0.2, 0.1), dtype=np.float64),
    )
    requests = (
        exact.FitRequest.create("kat/combined/affine", "affine", target),
        exact.FitRequest.create("kat/combined/combined", "combined", target),
    )
    affine, combined = exact.fit_global_contexts(
        source,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        requests,
        config_sha256=CONFIG_SHA256,
    )

    assert affine.source_grid == combined.source_grid
    assert combined.selected.state_index == geometry.state_index(truth)
    assert combined.selected.gains is not None and combined.selected.biases is not None
    np.testing.assert_allclose(
        combined.selected.gains, np.asarray((1.2, 0.8, 1.4)), rtol=0.0, atol=6e-15
    )
    np.testing.assert_allclose(
        combined.selected.biases, np.asarray((0.3, -0.2, 0.1)), rtol=0.0, atol=1e-15
    )
    assert len(combined.selected.retained_macro_ids) == 27
    assert combined.selected.objective < 2e-29
    assert combined.certificate.second_best_nonflow_gap > 1e-12
    assert np.max(np.abs(combined.prediction - target)) < 1.5e-14


def test_exact_objective_ties_use_full_canonical_state_key() -> None:
    source = np.zeros((3, 64, 64), dtype=np.float64)
    target = fitting.target_values(source, geometry.FULL_MASK)
    result = exact.fit_global(
        source,
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "affine",
        context_key="kat/all-tied/full",
        config_sha256=CONFIG_SHA256,
    )

    assert result.selected.state_index == 0
    assert result.certificate.exact_tie_multiplicity == geometry.ADMISSIBLE_COUNT
    assert result.certificate.second_best_objective_gap == 0.0
    assert result.certificate.second_best_nonflow_gap == 0.0


def test_context_validation_fails_closed_without_enumeration() -> None:
    source = _source()
    target = fitting.target_values(source, geometry.FULL_MASK)
    mutable_target = target.copy()
    request = exact.FitRequest("duplicate", "affine", mutable_target)
    before = request.fit_target.tobytes(order="C")
    mutable_target.fill(123.0)
    assert request.fit_target.tobytes(order="C") == before
    assert not request.fit_target.flags.writeable
    with pytest.raises(exact.GlobalV22Error, match="unique"):
        exact.fit_global_contexts(
            source,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            (request, request),
            config_sha256=CONFIG_SHA256,
        )
    with pytest.raises(exact.GlobalV22Error, match="config SHA"):
        exact.fit_global_contexts(
            source,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            (request,),
            config_sha256="not-a-hash",
        )
    with pytest.raises(exact.GlobalV22Error, match="fit target"):
        exact.FitRequest.create("bad", "affine", target[:, :-1].copy(order="F"))
    with pytest.raises(exact.GlobalV22Error, match="context key"):
        exact.FitRequest("", "affine", target)
