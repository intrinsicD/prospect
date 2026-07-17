"""Deterministic gates for the bench-only proposal-injection planner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from bench.proposal_injection import PlannerState, ProposalInjectionPlanner
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, Prediction


class _FirstActionModel:
    """Protocol-only model whose candidate return is determined by each action."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        value = float(np.asarray(action.data, dtype=float).reshape(-1)[0])
        return Prediction(
            mean=np.asarray(state.z, dtype=float),
            var=np.ones_like(np.asarray(state.z, dtype=float)),
            epistemic=0.0,
            aleatoric=1.0,
            reward=value,
        )

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, action) for action in actions]


class _FlatCapturePlanner(FlatPlanner):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(_FirstActionModel(), **kwargs)  # type: ignore[arg-type]
        self.pools: list[np.ndarray] = []

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        self.pools.append(sequences.copy())
        return sequences[:, 0, 0].copy()


class _InjectionCapturePlanner(ProposalInjectionPlanner):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(_FirstActionModel(), **kwargs)  # type: ignore[arg-type]
        self.pools: list[np.ndarray] = []

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        self.pools.append(sequences.copy())
        return sequences[:, 0, 0].copy()


def _state(raw: float = 7.0) -> PlannerState:
    return PlannerState(LatentState(z=np.array([0.0, 0.0])), np.array([raw, -1.0, 2.0]))


def _constant_provider(
    raw: np.ndarray, count: int, horizon: int, action_dim: int
) -> np.ndarray:
    return np.full((count, horizon, action_dim), raw[0] / 10.0)


def test_zero_injection_is_exact_flat_planner_parity_across_warm_calls() -> None:
    kwargs: dict[str, Any] = dict(
        action_dim=2,
        action_low=-1.5,
        action_high=1.5,
        horizon=5,
        candidates=9,
        elites=3,
        iterations=3,
        discount=0.97,
        uncertainty_penalty=0.0,
        seed=1234,
        colored_beta=1.5,
        keep_elite_fraction=2.0 / 3.0,
        temperature=0.7,
    )
    native = FlatPlanner(_FirstActionModel(), **kwargs)
    injected = ProposalInjectionPlanner(
        _FirstActionModel(),
        **kwargs,
        injection_count=0,
        # A provider may be configured for an ablation object but must not run.
        sequence_provider=lambda *_: (_ for _ in ()).throw(AssertionError("provider called")),
    )

    for raw in (3.0, 4.0):
        state = _state(raw)
        native_action = native.plan(state.latent)
        injected_action = injected.plan(state)
        assert np.array_equal(injected_action.data, native_action.data)
        assert injected._warm_mean is not None
        assert native._warm_mean is not None
        assert injected._warm_elites is not None
        assert native._warm_elites is not None
        assert np.array_equal(injected._warm_mean, native._warm_mean)
        assert np.array_equal(injected._warm_elites, native._warm_elites)

    # A subsequent native draw proves the complete bit-generator schedule matches.
    assert np.array_equal(
        injected._sample_colored_noise(4),
        native._sample_colored_noise(4),
    )
    assert len(injected.diagnostics) == 2
    assert injected.last_diagnostics is not None
    assert injected.last_diagnostics.injected_count == 0
    assert injected.last_diagnostics.injected_top_elite_count == 0
    assert injected.last_diagnostics.first_round_best_injected is False
    assert injected.last_diagnostics.best_sequence_was_injected is False
    assert injected.last_diagnostics.selected_first_action_source == "native"
    assert injected.last_diagnostics.candidate_eval_count == 27
    assert injected.last_diagnostics.candidate_transition_eval_count == 27 * 5


def test_replacement_uses_last_fresh_rows_and_preserves_native_rng_schedule() -> None:
    provider_calls: list[tuple[np.ndarray, int, int, int]] = []

    def provider(raw: np.ndarray, count: int, horizon: int, action_dim: int) -> np.ndarray:
        provider_calls.append((raw.copy(), count, horizon, action_dim))
        rows = np.arange(count, dtype=float)[:, None, None] + 0.75
        return np.broadcast_to(rows, (count, horizon, action_dim)).copy()

    kwargs = dict(
        action_dim=2,
        action_low=-2.0,
        action_high=2.0,
        horizon=4,
        candidates=8,
        elites=3,
        iterations=1,
        uncertainty_penalty=0.0,
        keep_elite_fraction=0.0,
        seed=91,
    )
    native = _FlatCapturePlanner(**kwargs)
    injected = _InjectionCapturePlanner(
        **kwargs,
        injection_count=2,
        sequence_provider=provider,
    )
    native.plan(_state().latent)
    injected.plan(_state())

    native_pool = native.pools[0]
    injected_pool = injected.pools[0]
    assert np.array_equal(injected_pool[:-2], native_pool[:-2])
    assert np.array_equal(injected_pool[-2], np.full((4, 2), 0.75))
    assert np.array_equal(injected_pool[-1], np.full((4, 2), 1.75))
    assert len(provider_calls) == 1
    raw, count, horizon, action_dim = provider_calls[0]
    assert np.array_equal(raw, _state().raw_sidecar)
    assert (count, horizon, action_dim) == (2, 4, 2)

    # Replacing already-sampled rows consumes no planner RNG of its own.
    assert np.array_equal(
        injected._sample_colored_noise(5),
        native._sample_colored_noise(5),
    )


def test_injected_provenance_diagnostics_survive_within_call_elite_retention() -> None:
    class _ZeroNoisePlanner(_InjectionCapturePlanner):
        def _sample_colored_noise(self, count: int) -> np.ndarray:
            return np.zeros((count, self.horizon, self.action_dim))

    def provider(
        raw: np.ndarray, count: int, horizon: int, action_dim: int
    ) -> np.ndarray:
        del raw
        values = np.array([1.0, 2.0])[:count, None, None]
        return np.broadcast_to(values, (count, horizon, action_dim)).copy()

    planner = _ZeroNoisePlanner(
        horizon=3,
        candidates=6,
        elites=2,
        iterations=2,
        keep_elite_fraction=1.0,
        temperature=0.5,
        action_low=-3.0,
        action_high=3.0,
        injection_count=2,
        sequence_provider=provider,
    )
    action = planner.plan(_state())

    assert action.data[0] == pytest.approx(2.0)
    diagnostics = planner.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.injected_count == 2
    assert diagnostics.injected_top_elite_count == 2
    assert diagnostics.first_round_best_injected
    assert diagnostics.best_sequence_injected
    assert diagnostics.best_sequence_was_injected
    assert diagnostics.selected_first_action_source == "injected"
    assert diagnostics.candidate_eval_count == 6 * 2
    assert diagnostics.candidate_transition_eval_count == 6 * 2 * 3


def test_first_round_transfer_is_distinct_from_later_native_selection() -> None:
    class _ScriptedPlanner(_InjectionCapturePlanner):
        def __init__(self, scores: list[np.ndarray], **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._scores = scores

        def _imagined_returns(
            self, state: LatentState, sequences: np.ndarray
        ) -> np.ndarray:
            self.pools.append(sequences.copy())
            return self._scores[len(self.pools) - 1].copy()

    planner = _ScriptedPlanner(
        [
            np.array([0.0, 1.0, 2.0, 3.0, 9.0]),
            np.array([10.0, 0.0, 0.0, 0.0, 0.0]),
        ],
        horizon=2,
        candidates=5,
        elites=2,
        iterations=2,
        keep_elite_fraction=0.0,
        action_low=-2.0,
        action_high=2.0,
        injection_count=1,
        sequence_provider=_constant_provider,
    )
    planner.plan(_state())

    diagnostics = planner.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.first_round_best_injected
    assert not diagnostics.best_sequence_injected
    assert diagnostics.selected_first_action_source == "native"


def test_warm_start_keeps_carried_tail_outside_replacement_and_reset_clears_it() -> None:
    planner = _InjectionCapturePlanner(
        horizon=3,
        candidates=7,
        elites=3,
        iterations=1,
        keep_elite_fraction=2.0 / 3.0,
        action_low=-2.0,
        action_high=2.0,
        seed=18,
        injection_count=2,
        sequence_provider=_constant_provider,
    )
    planner.plan(_state(7.0))
    assert planner._warm_elites is not None
    prior_elites = planner._warm_elites.copy()

    planner.plan(_state(8.0))
    warm_pool = planner.pools[1]
    keep_count = 2
    fresh_count = planner.candidates - keep_count
    expected_carried = planner._shift_sequences(prior_elites[:keep_count])
    assert np.array_equal(warm_pool[-keep_count:], expected_carried)
    assert np.array_equal(
        warm_pool[fresh_count - 2 : fresh_count],
        np.full((2, planner.horizon, planner.action_dim), 0.8),
    )

    diagnostic_count = len(planner.diagnostics)
    planner.reset()
    assert planner._warm_mean is None
    assert planner._warm_elites is None
    # Reset has FlatPlanner semantics: it clears warm state, not audit history.
    assert len(planner.diagnostics) == diagnostic_count
    planner.plan(_state(9.0))
    reset_pool = planner.pools[2]
    assert np.array_equal(
        reset_pool[-2:],
        np.full((2, planner.horizon, planner.action_dim), 0.9),
    )


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (lambda _r, _n, h, d: np.zeros((1, h, d)), "must return shape"),
        (lambda _r, n, h, d: np.full((n, h, d), np.nan), "non-finite"),
        (lambda _r, n, h, d: np.full((n, h, d), 4.0), "outside"),
    ],
)
def test_provider_output_is_strictly_validated(provider: object, message: str) -> None:
    planner = ProposalInjectionPlanner(
        _FirstActionModel(),
        horizon=2,
        candidates=4,
        elites=1,
        iterations=1,
        action_low=-1.0,
        action_high=1.0,
        injection_count=2,
        sequence_provider=provider,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match=message):
        planner.plan(_state())


def test_constructor_and_sidecar_shape_validation() -> None:
    common: dict[str, Any] = dict(
        horizon=2, candidates=4, elites=2, keep_elite_fraction=1.0
    )
    with pytest.raises(TypeError, match="integer"):
        ProposalInjectionPlanner(
            _FirstActionModel(), **common, injection_count=True, sequence_provider=_constant_provider
        )
    with pytest.raises(ValueError, match="non-negative"):
        ProposalInjectionPlanner(
            _FirstActionModel(), **common, injection_count=-1, sequence_provider=_constant_provider
        )
    with pytest.raises(ValueError, match="fresh-candidate count"):
        ProposalInjectionPlanner(
            _FirstActionModel(), **common, injection_count=3, sequence_provider=_constant_provider
        )
    with pytest.raises(ValueError, match="required"):
        ProposalInjectionPlanner(_FirstActionModel(), **common, injection_count=1)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        PlannerState(LatentState(z=np.zeros(2)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="finite"):
        PlannerState(LatentState(z=np.zeros(2)), np.array([0.0, np.inf]))

    planner = ProposalInjectionPlanner(
        _FirstActionModel(),
        **common,
        injection_count=1,
        sequence_provider=_constant_provider,
    )
    with pytest.raises(ValueError, match="raw_sidecar"):
        planner.plan(LatentState(z=np.zeros(2)))
