"""Unit tests for FlatPlanner (P2-001/U-002): iCEM optimization through the
protocol fallback, the ADR-0007 exploit-mode epistemic penalty, and the
receding-horizon warm start. Stub world models keep these fast and exact."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from prospect import interfaces
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, MemberRollout, Prediction


class _ConcaveRewardModel:
    """Protocol-only stub: reward peaks at action 0.7, state never moves."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        a = float(np.asarray(action.data).ravel()[0])
        return Prediction(
            mean=np.asarray(state.z, dtype=float),
            var=np.ones(2),
            epistemic=0.0,
            aleatoric=1.0,
            reward=1.0 - (a - 0.7) ** 2,
        )

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, a) for a in actions]


class _RiskyArmModel(_ConcaveRewardModel):
    """Positive actions promise more reward but with huge epistemic uncertainty."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        a = float(np.asarray(action.data).ravel()[0])
        risky = a > 0
        return Prediction(
            mean=np.asarray(state.z, dtype=float),
            var=np.ones(2),
            epistemic=5.0 if risky else 0.0,
            aleatoric=1.0,
            reward=1.0 if risky else 0.5,
        )


class _TrajectorySamplingModel(_ConcaveRewardModel):
    """Two deterministic members whose trajectories separate one unit per step."""

    def __init__(self) -> None:
        self.member_calls = 0

    def predict_member_batch(
        self,
        member_latents: np.ndarray,
        actions: np.ndarray,
        initial_ood: float | None = None,
    ) -> MemberRollout:
        self.member_calls += 1
        states = np.asarray(member_latents, dtype=float)
        if states.ndim == 2:
            states = np.repeat(states[None, :, :], 2, axis=0)
        next_states = states.copy()
        next_states[0] += 1.0
        next_states[1] -= 1.0
        variances = np.full_like(next_states, 0.25)
        rewards = np.ones((2, len(actions)))
        epistemic = next_states.var(axis=0).mean(axis=1)
        if initial_ood is not None:
            epistemic *= 1.0 + initial_ood
        return MemberRollout(
            states=next_states, variances=variances, rewards=rewards, epistemic=epistemic
        )


class _OODAwareProtocolModel(_ConcaveRewardModel):
    def predict(self, state: LatentState, action: Action) -> Prediction:
        prediction = super().predict(state, action)
        return Prediction(
            mean=prediction.mean,
            var=prediction.var,
            epistemic=1.0 + (state.ood or 0.0),
            aleatoric=prediction.aleatoric,
            reward=prediction.reward,
        )


class _FirstActionScorePlanner(FlatPlanner):
    """Rigged iCEM loop: deterministic proposals scored by their first action."""

    def __init__(self, **kwargs: float) -> None:
        super().__init__(_ConcaveRewardModel(), **kwargs)  # type: ignore[arg-type]
        self.candidate_pools: list[np.ndarray] = []

    def _sample_colored_noise(self, count: int) -> np.ndarray:
        grid = np.linspace(-0.4, 0.4, count) if count else np.empty(0)
        return np.broadcast_to(
            grid[:, None, None], (count, self.horizon, self.action_dim)
        ).copy()

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        self.candidate_pools.append(sequences.copy())
        return sequences[:, 0, 0].copy()


class _ScriptedScorePlanner(_FirstActionScorePlanner):
    """Rigged scores make the global best occur before the final iteration."""

    def __init__(self, scores: list[np.ndarray], **kwargs: float) -> None:
        super().__init__(**kwargs)
        self._scores = scores

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        self.candidate_pools.append(sequences.copy())
        scores = self._scores[len(self.candidate_pools) - 1]
        assert len(scores) == len(sequences)
        return scores.copy()


class _CandidateCapturePlanner(FlatPlanner):
    """Use the production sampler while recording the population scored by plan()."""

    def __init__(self, **kwargs: float) -> None:
        super().__init__(_ConcaveRewardModel(), **kwargs)  # type: ignore[arg-type]
        self.candidate_pools: list[np.ndarray] = []

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        self.candidate_pools.append(sequences.copy())
        return np.zeros(len(sequences))


def _planner(model: object, **kwargs: float) -> FlatPlanner:
    defaults: dict[str, float] = {"horizon": 4, "candidates": 32, "elites": 4, "iterations": 3}
    return FlatPlanner(model, **(defaults | kwargs))  # type: ignore[arg-type]


def test_planner_satisfies_protocol_with_stub_model() -> None:
    assert isinstance(_planner(_ConcaveRewardModel()), interfaces.Planner)


def test_cem_finds_the_concave_optimum_via_protocol_fallback() -> None:
    # Enough CEM budget to converge (few iterations leave the elite spread wide).
    planner = _planner(_ConcaveRewardModel(), candidates=64, elites=6, iterations=6,
                       uncertainty_penalty=0.0)
    action = planner.plan(LatentState(z=np.zeros(2)))
    assert abs(float(np.asarray(action.data).ravel()[0]) - 0.7) < 0.15


def test_epistemic_penalty_repels_the_risky_arm() -> None:
    state = LatentState(z=np.zeros(2))
    greedy = _planner(_RiskyArmModel(), uncertainty_penalty=0.0, seed=1)
    cautious = _planner(_RiskyArmModel(), uncertainty_penalty=1.0, seed=1)
    assert float(np.asarray(greedy.plan(state).data).ravel()[0]) > 0  # chases reward
    assert float(np.asarray(cautious.plan(state).data).ravel()[0]) < 0  # ADR-0007 exploit mode


def test_warm_start_persists_and_resets() -> None:
    planner = _planner(_ConcaveRewardModel())
    state = LatentState(z=np.zeros(2))
    planner.plan(state)
    assert planner._warm_mean is not None
    assert planner._warm_elites is not None
    planner.reset()
    assert planner._warm_mean is None
    assert planner._warm_elites is None


@pytest.mark.parametrize(("horizon", "minimum_delta"), [(4, 0.15), (32, 0.5)])
def test_colored_noise_is_temporally_correlated_without_losing_offsets(
    horizon: int, minimum_delta: float
) -> None:
    common = {
        "horizon": horizon,
        "candidates": 4096,
        "elites": 1,
        "iterations": 1,
        "seed": 1729,
    }
    colored = _CandidateCapturePlanner(colored_beta=2.0, **common)
    white = _CandidateCapturePlanner(colored_beta=0.0, **common)
    state = LatentState(z=np.zeros(2))
    colored.plan(state)
    white.plan(state)
    colored_samples = colored.candidate_pools[0][:, :, 0]
    white_samples = white.candidate_pools[0][:, :, 0]

    def lag_one(samples: np.ndarray) -> float:
        return float(
            np.corrcoef(samples[:, :-1].ravel(), samples[:, 1:].ravel())[0, 1]
        )

    assert lag_one(colored_samples) > lag_one(white_samples) + minimum_delta
    raw_colored = _planner(
        _ConcaveRewardModel(), horizon=horizon, colored_beta=2.0, seed=1729
    )._sample_colored_noise(4096)[:, :, 0]
    assert float(raw_colored.std()) == pytest.approx(1.0, abs=0.04)
    # A zeroed DC bin would forbid sustained positive/negative action proposals.
    assert float(raw_colored.mean(axis=1).std()) > 0.1


def test_icem_keeps_elites_with_constant_candidate_budget_and_shifts_them() -> None:
    planner = _FirstActionScorePlanner(
        horizon=3,
        candidates=6,
        elites=4,
        iterations=2,
        keep_elite_fraction=0.5,
        temperature=0.5,
    )
    state = LatentState(z=np.zeros(2))
    planner.plan(state)

    first_pool, second_pool = planner.candidate_pools
    assert first_pool.shape == second_pool.shape == (6, 3, 1)
    first_scores = first_pool[:, 0, 0]
    elite_indices = np.argsort(first_scores)[-4:][::-1]
    first_elite = first_pool[elite_indices]
    first_weights = planner._softmax_weights(first_scores[elite_indices], planner.temperature)
    expected_mean = np.sum(first_weights[:, None, None] * first_elite, axis=0)
    expected_variance = np.sum(
        first_weights[:, None, None] * (first_elite - expected_mean) ** 2, axis=0
    )
    expected_std = np.maximum(np.sqrt(expected_variance), 0.05)
    fresh_noise = np.broadcast_to(
        np.linspace(-0.4, 0.4, 4)[:, None, None], (4, 3, 1)
    )
    expected_fresh = np.clip(
        expected_mean + expected_std * fresh_noise,
        planner.action_low,
        planner.action_high,
    )
    kept = first_elite[:2]
    assert np.allclose(second_pool[:4], expected_fresh)
    assert np.allclose(second_pool[-2:], kept)

    assert planner._warm_elites is not None
    shifted = planner._shift_sequences(planner._warm_elites[:2].copy())
    planner.plan(state)
    next_call_first_pool = planner.candidate_pools[2]
    assert next_call_first_pool.shape == (6, 3, 1)
    assert np.allclose(next_call_first_pool[-2:], shifted)


def test_icem_executes_global_best_and_uses_softmax_elite_update() -> None:
    planner = _FirstActionScorePlanner(
        horizon=3,
        candidates=5,
        elites=3,
        iterations=1,
        keep_elite_fraction=0.0,
        temperature=0.5,
    )
    action = planner.plan(LatentState(z=np.zeros(2)))
    pool = planner.candidate_pools[0]
    scores = pool[:, 0, 0]
    elite_indices = np.argsort(scores)[-3:][::-1]
    elite = pool[elite_indices]
    weights = planner._softmax_weights(scores[elite_indices], planner.temperature)
    expected_mean = np.sum(weights[:, None, None] * elite, axis=0)

    assert float(action.data[0]) == pytest.approx(float(scores.max()))
    assert planner._warm_mean is not None
    assert np.allclose(planner._warm_mean, expected_mean)
    assert not np.isclose(float(action.data[0]), float(expected_mean[0, 0]))


def test_icem_executes_the_best_sequence_seen_across_iterations() -> None:
    scores = [
        np.array([0.0, 100.0, 1.0, 2.0, 3.0]),
        np.array([-5.0, -4.0, -3.0, -2.0, -1.0]),
    ]
    planner = _ScriptedScorePlanner(
        scores,
        horizon=3,
        candidates=5,
        elites=2,
        iterations=2,
        keep_elite_fraction=0.0,
        temperature=0.5,
    )
    action = planner.plan(LatentState(z=np.zeros(2)))

    first_iteration_best = planner.candidate_pools[0][1, 0, 0]
    final_iteration_best = planner.candidate_pools[1][-1, 0, 0]
    assert float(action.data[0]) == pytest.approx(float(first_iteration_best))
    assert not np.isclose(float(action.data[0]), float(final_iteration_best))


def test_softmax_elite_weights_are_stable_and_recover_the_arithmetic_mean() -> None:
    scores = np.array([-1000.0, 0.0, 1000.0])
    weights = FlatPlanner._softmax_weights(scores, temperature=0.5)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(1.0)

    uniform = FlatPlanner._softmax_weights(scores, temperature=np.inf)
    elite = np.array([[[-1.0]], [[2.0]], [[8.0]]])
    weighted_mean = np.sum(uniform[:, None, None] * elite, axis=0)
    assert np.allclose(uniform, np.full(3, 1.0 / 3.0))
    assert np.allclose(weighted_mean, elite.mean(axis=0))
    with pytest.raises(ValueError, match="scores must be finite"):
        FlatPlanner._softmax_weights(np.array([-np.inf, 0.0]), temperature=0.5)


def test_ts_infinity_scoring_and_optional_epistemic_truncation() -> None:
    state = LatentState(z=np.zeros(2))
    sequences = np.zeros((3, 4, 1))

    full_model = _TrajectorySamplingModel()
    full = FlatPlanner(
        full_model, horizon=4, candidates=3, elites=1, iterations=1,
        discount=1.0, uncertainty_penalty=0.0, epistemic_horizon_bound=None,
    )
    assert np.allclose(full._imagined_returns(state, sequences), 4.0)
    assert full_model.member_calls == 4  # bound=None keeps the full horizon

    bounded_model = _TrajectorySamplingModel()
    bounded = FlatPlanner(
        bounded_model, horizon=4, candidates=3, elites=1, iterations=1,
        discount=1.0, uncertainty_penalty=0.0, epistemic_horizon_bound=0.5,
    )
    # First-step spread is 1.0, so that crossing step is scored and the rest is cut.
    assert np.allclose(bounded._imagined_returns(state, sequences), 1.0)
    assert bounded_model.member_calls == 1


def test_planner_consumes_the_model_owned_effective_epistemic() -> None:
    model = _TrajectorySamplingModel()
    planner = FlatPlanner(
        model, horizon=1, candidates=1, elites=1, iterations=1,
        discount=1.0, uncertainty_penalty=1.0,
    )
    score = planner._imagined_returns(
        LatentState(z=np.zeros(2), ood=9.0), np.zeros((1, 1, 1))
    )
    # Raw member spread is 1; the model's OOD-aware effective signal is 10.
    assert score[0] == pytest.approx(1.0 - 10.0)


def test_protocol_fallback_preserves_the_root_ood_signal() -> None:
    planner = FlatPlanner(
        _OODAwareProtocolModel(), horizon=1, candidates=1, elites=1, iterations=1,
        discount=1.0, uncertainty_penalty=1.0,
    )
    score = planner._imagined_returns(
        LatentState(z=np.zeros(2), ood=9.0), np.zeros((1, 1, 1))
    )
    # Concave stub reward at a=0 is 0.51; OOD-aware epistemic is 10.
    assert score[0] == pytest.approx(0.51 - 10.0)
