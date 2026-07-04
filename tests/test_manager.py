"""Unit tests for the HierarchicalManager (P5-002): planning (not reaction) over
the jumpy model, the exploit-mode epistemic penalty, and VoE termination."""
from __future__ import annotations

import numpy as np
import pytest

from prospect import interfaces
from prospect.planning import HierarchicalManager
from prospect.types import Action, LatentState, Option, Prediction, Transition

GREEDY = Option(name="greedy", horizon=1)
DETOUR = Option(name="detour", horizon=1)
RISKY = Option(name="risky", horizon=1)


class _TwoRoomModel:
    """Stub option-model: latent is (room,). In room 0, 'greedy' pays 1 and stays;
    'detour' pays 0 and moves to room 1, where 'greedy' pays 10. 'risky' promises
    5 anywhere but with huge epistemic."""

    def predict_option(self, state: LatentState, option: Option) -> Prediction:
        room = float(np.asarray(state.z).ravel()[0])
        if option is RISKY:
            return self._make(room, reward=5.0, epistemic=50.0)
        if option is DETOUR:
            return self._make(1.0, reward=0.0)
        return self._make(room, reward=10.0 if room >= 1.0 else 1.0)

    def _make(self, room: float, reward: float, epistemic: float = 0.01) -> Prediction:
        return Prediction(mean=np.array([room]), var=np.array([0.01]),
                          epistemic=epistemic, aleatoric=0.01, reward=reward, duration=1.0)


def _manager(depth: int, options: list[Option], penalty: float = 1.0) -> HierarchicalManager:
    return HierarchicalManager(_TwoRoomModel(), options, depth=depth,
                               uncertainty_penalty=penalty)


def test_conforms_and_requires_configuration() -> None:
    assert isinstance(HierarchicalManager(), interfaces.HierarchicalPlanner)
    with pytest.raises(ValueError, match="option model"):
        HierarchicalManager().plan_option(LatentState(z=np.zeros(1)))


def test_depth_one_is_myopic_but_depth_two_plans_the_detour() -> None:
    state = LatentState(z=np.zeros(1))
    myopic = _manager(depth=1, options=[GREEDY, DETOUR])
    assert myopic.plan_option(state) is GREEDY  # 1 now beats 0 now
    planner = _manager(depth=2, options=[GREEDY, DETOUR])
    assert planner.plan_option(state) is DETOUR  # 0 + 10 beats 1 + 1: planning, not reaction


def test_exploit_mode_penalty_avoids_the_uncertain_option() -> None:
    state = LatentState(z=np.zeros(1))
    greedy_about_risk = _manager(depth=1, options=[GREEDY, RISKY], penalty=0.0)
    assert greedy_about_risk.plan_option(state) is RISKY  # 5 beats 1 when risk is free
    cautious = _manager(depth=1, options=[GREEDY, RISKY], penalty=1.0)
    assert cautious.plan_option(state) is GREEDY  # 5 - 50 loses (ADR-0006/0007)


def test_voe_termination_fires_on_violation_only() -> None:
    manager = HierarchicalManager(surprise_threshold=10.0)
    prediction = Prediction(mean=np.zeros(2), var=np.full(2, 0.01),
                            epistemic=0.0, aleatoric=0.01)

    def transition(observed: np.ndarray) -> Transition:
        return Transition(state=LatentState(z=np.zeros(2)), action=Action(data=np.zeros(1)),
                          next_state=LatentState(z=observed), reward=0.0,
                          prediction=prediction)

    assert manager.should_terminate(transition(np.array([0.01, -0.01]))) is False
    assert manager.should_terminate(transition(np.array([2.0, -2.0]))) is True
    no_expectation = Transition(state=LatentState(z=np.zeros(2)), action=Action(data=np.zeros(1)),
                                next_state=LatentState(z=np.full(2, 9.9)), reward=0.0)
    assert manager.should_terminate(no_expectation) is False
