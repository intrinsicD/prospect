"""Unit tests for the SkillRouter (P4-001): simulate-to-match ranking, the
epistemic term as the predictive precondition, and competence gating."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from prospect import interfaces
from prospect.skills import SkillRouter
from prospect.types import Action, Competence, LatentState, Option, Prediction, Subgoal, Surprise, Transition


class _ShiftModel:
    """Transparent stub: the latent moves by exactly the action each step."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        z = np.asarray(state.z, dtype=float)
        a = float(np.asarray(action.data).ravel()[0])
        return Prediction(mean=z + a, var=np.ones_like(z), epistemic=0.01, aleatoric=0.1)

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, a) for a in actions]


class _RiskyPositiveModel(_ShiftModel):
    """Positive moves are unpredictable: huge epistemic (precondition violated)."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        prediction = super().predict(state, action)
        if float(np.asarray(action.data).ravel()[0]) > 0:
            return Prediction(mean=prediction.mean, var=prediction.var,
                              epistemic=50.0, aleatoric=0.1)
        return prediction


class _MasterySubsetMonitor:
    """CompetenceMonitor stub: a fixed set of skills counts as mastered."""

    def __init__(self, mastered: set[str]) -> None:
        self._mastered = mastered

    def surprise(self, prediction: Prediction, observed: LatentState) -> Surprise:
        return Surprise(total=0.0, epistemic=0.0, aleatoric=0.0)

    def update(self, transition: Transition) -> None:
        return None

    def competence(self, skill: str) -> Competence:
        return Competence(skill=skill, epistemic=0.0, learning_progress=0.0,
                          mastered=skill in self._mastered)

    def is_mastered(self, skill: str) -> bool:
        return skill in self._mastered

    def is_forgetting(self, skill: str) -> bool:
        return False


def _const(value: float) -> Option:
    def policy(latent: LatentState) -> Action:
        return Action(data=np.array([value]))

    return Option(name=f"move{value:+.0f}", policy=policy, horizon=3)


def _state(z: float = 0.0) -> LatentState:
    return LatentState(z=np.array([z]))


def _goal(z: float) -> Subgoal:
    return Subgoal(target=LatentState(z=np.array([z])))


def test_router_satisfies_protocol_and_rejects_policy_less_options() -> None:
    router = SkillRouter(_ShiftModel())
    assert isinstance(router, interfaces.SkillLibrary)
    with pytest.raises(ValueError, match="has no policy"):
        router.add(Option(name="ghost"))


def test_simulate_to_match_ranks_the_reaching_skill_first() -> None:
    router = SkillRouter(_ShiftModel())
    left, right = _const(-1.0), _const(+1.0)
    router.add(left)
    router.add(right)
    assert router.propose(_state(), _goal(+3.0))[0] is right  # 3 steps of +1 land at +3
    assert router.propose(_state(), _goal(-3.0))[0] is left


def test_epistemic_term_is_the_predictive_precondition() -> None:
    # Both skills land equally far from the goal at 0; the unpredictable one loses.
    router = SkillRouter(_RiskyPositiveModel(), uncertainty_weight=1.0)
    left, right = _const(-1.0), _const(+1.0)
    router.add(left)
    router.add(right)
    assert router.propose(_state(), _goal(0.0))[0] is left


def test_competence_gating_and_cold_start() -> None:
    left, right = _const(-1.0), _const(+1.0)
    gated = SkillRouter(_ShiftModel(), monitor=_MasterySubsetMonitor({left.name}))
    gated.add(left)
    gated.add(right)
    proposals = gated.propose(_state(), _goal(+3.0))
    assert [o.name for o in proposals] == [left.name]  # only mastered offered upward
    cold = SkillRouter(_ShiftModel(), monitor=_MasterySubsetMonitor(set()))
    cold.add(left)
    cold.add(right)
    assert len(cold.propose(_state(), _goal(+3.0))) == 2  # nothing mastered: offer all


def test_simulate_returns_landing_prediction_and_epistemic() -> None:
    router = SkillRouter(_ShiftModel())
    prediction, epistemic = router.simulate(_state(0.0), _const(+1.0))
    assert float(np.asarray(prediction.mean)[0]) == pytest.approx(3.0)
    assert epistemic == pytest.approx(0.03)