"""Unit tests for FlatPlanner (P2-001): CEM optimization through the protocol
fallback, the ADR-0007 exploit-mode epistemic penalty, and the receding-horizon
warm start. Stub world models keep these fast and exact."""
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
    planner.reset()
    assert planner._warm_mean is None


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
