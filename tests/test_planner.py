"""Unit tests for FlatPlanner (P2-001): CEM optimization through the protocol
fallback, the ADR-0007 exploit-mode epistemic penalty, and the receding-horizon
warm start. Stub world models keep these fast and exact."""
from __future__ import annotations

import numpy as np

from prospect import interfaces
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, Prediction


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

    def imagine(self, state: LatentState, actions: list[Action]) -> list[Prediction]:
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
