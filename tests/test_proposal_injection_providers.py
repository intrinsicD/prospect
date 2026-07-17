from __future__ import annotations

import numpy as np
import pytest

from bench.proposal_injection.providers import ExactReferenceProvider, provider_summary


@pytest.fixture(scope="module")
def start() -> np.ndarray:
    return np.array([-0.84, -0.08, -0.75], dtype=float)


def test_privileged_provider_is_deterministic_and_accounted(start: np.ndarray) -> None:
    first = ExactReferenceProvider(3, "privileged")
    second = ExactReferenceProvider(3, "privileged")

    left = first(start, 8, 12, 2)
    right = second(start, 8, 12, 2)

    assert np.array_equal(left, right)
    assert first.calls == second.calls
    assert first.calls[0].reference_exact_scores == first.calls[0].output_exact_scores
    assert first.calls[0].oracle_sequence_evaluations == 2_704
    assert first.calls[0].oracle_transition_evaluations == 2_704 * 12
    summary = provider_summary(first)
    assert summary["call_count"] == 1
    assert summary["mean_exact_score_delta"] == pytest.approx(0.0)


def test_action_permutation_uses_same_reference_bank(start: np.ndarray) -> None:
    privileged = ExactReferenceProvider(4, "privileged")
    permuted = ExactReferenceProvider(4, "action_permuted")

    original = privileged(start, 8, 12, 2)
    transformed = permuted(start, 8, 12, 2)

    assert np.array_equal(transformed, original[:, :, ::-1])
    assert privileged.calls[0].bank_sha256 == permuted.calls[0].bank_sha256
    assert privileged.calls[0].bank_seed == permuted.calls[0].bank_seed
    assert permuted.calls[0].mode == "action_permuted"


def test_time_permutation_is_non_identity_and_preserves_values(start: np.ndarray) -> None:
    privileged = ExactReferenceProvider(5, "privileged")
    shifted = ExactReferenceProvider(5, "time_permuted")
    original = privileged(start, 8, 12, 2)
    output = shifted(start, 8, 12, 2)

    assert shifted.calls[0].time_shift == 1
    assert np.array_equal(output, np.roll(original, shift=1, axis=1))
    assert np.array_equal(np.sort(output, axis=1), np.sort(original, axis=1))


def test_provider_rejects_non_protocol_shapes(start: np.ndarray) -> None:
    provider = ExactReferenceProvider(0)
    with pytest.raises(ValueError, match="count=8"):
        provider(start, 4, 12, 2)
    with pytest.raises(ValueError, match="finite BridgeControl"):
        provider(np.array([0.0, np.nan, 1.0]), 8, 12, 2)
