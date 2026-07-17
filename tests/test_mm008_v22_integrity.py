"""Production-backed integrity tests for the MM-008 v2.2 exact grid."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import integrity_v22 as integrity

CONFIG_SHA256 = "4" * 64


def _source() -> np.ndarray:
    index = np.arange(3 * 64 * 64, dtype=np.float64).reshape(3, 64, 64)
    return np.ascontiguousarray(
        np.sin(index * 0.017)
        + np.cos(index * 0.0031)
        + 0.2 * np.sin(index * 0.111),
        dtype=np.float64,
    )


def _combined_target(source: np.ndarray) -> np.ndarray:
    sampled = geometry.sample_scalar(
        source,
        (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0),
        geometry.FULL_MASK,
    )
    target = source.copy()
    coords = geometry.GEOMETRY.coords.astype(np.intp)
    target[:, coords[:, 0], coords[:, 1]] = (
        np.asarray((1.2, 0.8, 1.4), dtype=np.float64)[:, None] * sampled
        + np.asarray((0.3, -0.2, 0.1), dtype=np.float64)[:, None]
    )
    return fitting.target_values(target, geometry.FULL_MASK)


@dataclass(frozen=True)
class _Completed:
    source: np.ndarray
    changed_source: np.ndarray
    target: np.ndarray
    changed_target: np.ndarray
    result: exact.GlobalResult


@pytest.fixture(scope="module")
def completed() -> _Completed:
    source = _source()
    changed_source = source.copy()
    changed_source[0, 0, 0] = np.nextafter(changed_source[0, 0, 0], np.inf)
    target = _combined_target(source)
    changed_target = target.copy()
    changed_target[0, 0] = np.nextafter(changed_target[0, 0], np.inf)
    result = exact.fit_global(
        source,
        target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "combined",
        context_key="integrity/authenticated/combined",
        config_sha256=CONFIG_SHA256,
    )
    return _Completed(source, changed_source, target, changed_target, result)


@pytest.fixture(scope="module")
def complete_evidence(completed: _Completed) -> integrity.IntegrityEvidence:
    return integrity.run_integrity_controls(
        completed.result,
        completed.source,
        completed.changed_source,
        completed.target,
        completed.changed_target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        config_sha256=CONFIG_SHA256,
    )


def test_delivery_collector_is_package_private_and_public_fit_api_is_unchanged() -> None:
    assert "_ObjectiveEntry" not in exact.__all__
    assert "_CompleteDelivery" not in exact.__all__
    assert "_collect_complete_delivery" not in exact.__all__
    assert tuple(inspect.signature(exact.fit_global).parameters) == (
        "source",
        "fit_target",
        "fit_mask",
        "output_mask",
        "arm",
        "context_key",
        "config_sha256",
    )
    assert tuple(inspect.signature(exact.fit_global_contexts).parameters) == (
        "source",
        "fit_mask",
        "output_mask",
        "requests",
        "config_sha256",
    )
    assert (
        inspect.signature(exact.fit_global).parameters["context_key"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(exact.fit_global_contexts).parameters["config_sha256"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_global_scientific_arrays_and_internal_masks_are_irreversibly_immutable(
    completed: _Completed,
) -> None:
    request = exact.FitRequest.create(
        "integrity/immutable/request", "combined", completed.target
    )
    checked_source = exact._validate_source(completed.source)
    checked_fit, checked_output = exact._validate_context_masks(
        geometry.FULL_MASK, geometry.FULL_MASK
    )
    checked_uint8 = exact._readonly_uint8(np.arange(8, dtype=np.uint8))
    cache = completed.result.objective_cache
    selected = completed.result.selected
    assert cache.gains is not None and cache.biases is not None
    assert selected.gains is not None and selected.biases is not None
    arrays = (
        request.fit_target,
        checked_source,
        checked_fit,
        checked_output,
        checked_uint8,
        cache.objectives,
        cache.gains,
        cache.biases,
        selected.parameters,
        selected.gains,
        selected.biases,
        selected.fit_prediction,
        completed.result.prediction,
    )
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)

    duplicated_ids = tuple(
        sorted((selected.retained_macro_ids[0], *selected.retained_macro_ids[1:-1], selected.retained_macro_ids[0]))
    )
    with pytest.raises(exact.GlobalV22Error, match="retained macro IDs"):
        replace(selected, retained_macro_ids=duplicated_ids)


def test_complete_delivery_forgery_cache_and_scalar_controls_pass(
    completed: _Completed,
    complete_evidence: integrity.IntegrityEvidence,
) -> None:
    evidence = complete_evidence
    assert evidence.passed
    assert tuple(item.order for item in evidence.delivery.orders) == (
        "canonical",
        "reversed",
        "even_then_odd",
    )
    assert {
        item.objective_content_sha256 for item in evidence.delivery.orders
    } == {completed.result.objective_cache.content_sha256}
    assert tuple(item.name for item in evidence.delivery.rejections) == (
        "missing",
        "duplicate",
        "index_status",
        "admissibility_bit",
    )
    assert all(item.rejected for item in evidence.delivery.rejections)
    assert evidence.delivery.batch_sizes == (*((128,) * 21), 121)
    assert evidence.delivery.scalar_replay_bit_exact

    assert tuple(item.key for item in evidence.forgery_matrix.witnesses) == integrity.FORGERY_KEYS
    assert set(integrity.NORMATIVE_HASH_FORGERY_KEYS).issubset(
        item.key for item in evidence.forgery_matrix.witnesses
    )
    assert all(
        item.rejected_at in {"construction", "deep_compare"}
        for item in evidence.forgery_matrix.witnesses
    )
    assert tuple(check.name for check in evidence.cache_identities.checks) == (
        integrity.CACHE_CHECK_NAMES
    )
    assert all(check.passed for check in evidence.cache_identities.checks)


def test_forgery_matrix_uses_one_independently_rebuilt_reference(
    completed: _Completed,
) -> None:
    with patch.object(
        integrity.exact,
        "fit_global",
        wraps=exact.fit_global,
    ) as fit_global:
        evidence = integrity.audit_deep_forgery_matrix(
            completed.result,
            completed.source,
            completed.target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=CONFIG_SHA256,
        )
    assert evidence.passed
    assert fit_global.call_count == 1


def test_plausible_all_pass_integrity_evidence_is_rejected_by_complete_replay(
    completed: _Completed,
    complete_evidence: integrity.IntegrityEvidence,
) -> None:
    original = complete_evidence
    canonical = original.delivery.orders[0]
    forged_rank = (
        canonical.selected_admissible_rank + 1
    ) % geometry.ADMISSIBLE_COUNT
    forged_state = int(geometry.ADMISSIBLE_INDICES[forged_rank])
    forged_orders = tuple(
        replace(
            witness,
            selected_state_index=forged_state,
            selected_admissible_rank=forged_rank,
        )
        for witness in original.delivery.orders
    )
    forged_delivery = replace(original.delivery, orders=forged_orders)

    first_forgery = original.forgery_matrix.witnesses[0]
    alternate_stage: Literal["construction", "deep_compare"] = (
        "construction"
        if first_forgery.rejected_at == "deep_compare"
        else "deep_compare"
    )
    forged_matrix = replace(
        original.forgery_matrix,
        witnesses=(
            replace(first_forgery, rejected_at=alternate_stage),
            *original.forgery_matrix.witnesses[1:],
        ),
    )
    forged_cache = replace(
        original.cache_identities,
        identities=tuple(
            (name, "f" * 64) for name, _ in original.cache_identities.identities
        ),
    )
    forged = integrity.IntegrityEvidence(
        forged_delivery,
        forged_matrix,
        forged_cache,
    )
    assert forged.passed
    with pytest.raises(integrity.IntegrityV22Error, match="complete replay"):
        integrity.validate_integrity_evidence(
            forged,
            completed.result,
            completed.source,
            completed.changed_source,
            completed.target,
            completed.changed_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            config_sha256=CONFIG_SHA256,
        )
