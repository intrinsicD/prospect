"""Unit tests for the SurpriseCompetenceMonitor (P3-001): decomposition that sums
exactly, attribution that follows the uncertainty source, and the mastery state
machine (low epistemic + flattened progress), per ADR-0002."""
from __future__ import annotations

from math import isinf

import numpy as np
import pytest

from prospect.types import (
    Action,
    Competence,
    LatentState,
    Mode,
    Option,
    Prediction,
    Surprise,
    Transition,
)
from prospect.voe import LearningProgressCurriculum, SurpriseCompetenceMonitor


def _pred(epistemic: float, aleatoric: float) -> Prediction:
    return Prediction(mean=np.zeros(2), var=np.ones(2), epistemic=epistemic, aleatoric=aleatoric)


def _transition(epistemic: float, skill: str | None = None,
                with_prediction: bool = True, error: float = 0.0) -> Transition:
    # `error` sets how far the observed next-state sits from the prediction mean
    # (mean is zeros), i.e. the prediction error the monitor tracks for forgetting.
    return Transition(
        state=LatentState(z=np.zeros(2)),
        action=Action(data=np.zeros(1)),
        next_state=LatentState(z=np.full(2, error**0.5)),
        reward=0.0,
        prediction=_pred(epistemic, 0.1) if with_prediction else None,
        option=Option(name=skill) if skill is not None else None,
    )


def test_decomposition_sums_exactly_to_total() -> None:
    monitor = SurpriseCompetenceMonitor()
    s = monitor.surprise(_pred(0.3, 0.7), LatentState(z=np.array([0.5, -0.5])))
    assert s.total == pytest.approx(-_pred(0.3, 0.7).log_prob(np.array([0.5, -0.5])))
    assert s.epistemic + s.aleatoric == pytest.approx(s.total)


def test_attribution_follows_the_uncertainty_source() -> None:
    monitor = SurpriseCompetenceMonitor()
    observed = LatentState(z=np.array([1.5, -1.5]))
    epistemic_dominated = monitor.surprise(_pred(0.9, 0.1), observed)
    aleatoric_dominated = monitor.surprise(_pred(0.1, 0.9), observed)
    assert epistemic_dominated.epistemic == pytest.approx(0.9 * epistemic_dominated.total)
    assert aleatoric_dominated.aleatoric == pytest.approx(0.9 * aleatoric_dominated.total)


def test_unseen_skill_is_maximally_unknown() -> None:
    competence = SurpriseCompetenceMonitor().competence("never-practiced")
    assert isinf(competence.epistemic)
    assert competence.mastered is False


def test_mastery_lifecycle() -> None:
    monitor = SurpriseCompetenceMonitor(
        mastery_epistemic=0.1, flat_progress=0.01, min_updates=5,
        fast_rate=0.5, slow_rate=0.3,
    )
    skill = "swing"
    for _ in range(10):  # plateau at high epistemic: unlearned
        monitor.update(_transition(1.0, skill))
    assert monitor.is_mastered(skill) is False
    for _ in range(4):  # falling fast: below threshold but progress not yet flat
        monitor.update(_transition(0.02, skill))
    competence = monitor.competence(skill)
    assert competence.epistemic < 0.1
    assert competence.learning_progress > 0.01  # still improving -> not mastered
    assert competence.mastered is False
    for _ in range(40):  # converged low and flat: mastered
        monitor.update(_transition(0.02, skill))
    assert monitor.is_mastered(skill) is True


def test_skills_are_isolated() -> None:
    monitor = SurpriseCompetenceMonitor(min_updates=1, fast_rate=0.5, slow_rate=0.5)
    for _ in range(30):
        monitor.update(_transition(0.001, "practiced"))
    assert monitor.is_mastered("practiced") is True
    assert monitor.is_mastered("other") is False
    assert isinf(monitor.competence("other").epistemic)


def test_prediction_less_transitions_are_ignored() -> None:
    monitor = SurpriseCompetenceMonitor()
    monitor.update(_transition(1.0, "skill", with_prediction=False))
    assert isinf(monitor.competence("skill").epistemic)  # nothing was expected


class _FixedMasteryMonitor:
    """CompetenceMonitor-conforming stub with a fixed mastery verdict."""

    def __init__(self, mastered: bool) -> None:
        self._mastered = mastered

    def surprise(self, prediction: Prediction, observed: LatentState) -> Surprise:
        return Surprise(total=0.0, epistemic=0.0, aleatoric=0.0)

    def update(self, transition: Transition) -> None:
        return None

    def competence(self, skill: str) -> Competence:
        return Competence(skill=skill, epistemic=0.0, learning_progress=0.0,
                          mastered=self._mastered)

    def is_mastered(self, skill: str) -> bool:
        return self._mastered

    def is_forgetting(self, skill: str) -> bool:
        return False


def test_curriculum_owns_the_sign() -> None:
    # ADR-0007: the sign applied to epistemic uncertainty is the curriculum's
    # decision alone — negative bonus while learning, positive penalty once mastered.
    learning = LearningProgressCurriculum(_FixedMasteryMonitor(False),
                                          explore_bonus=2.0, exploit_penalty=3.0)
    assert learning.mode() is Mode.EXPLORE
    assert learning.uncertainty_coefficient() == -2.0
    mastered = LearningProgressCurriculum(_FixedMasteryMonitor(True),
                                          explore_bonus=2.0, exploit_penalty=3.0)
    assert mastered.mode() is Mode.EXPLOIT
    assert mastered.uncertainty_coefficient() == 3.0


def test_is_forgetting_lifecycle() -> None:
    # P7-001: forgetting = prediction ERROR rising on a once-mastered skill. It
    # keys on error, not epistemic, because a confidently-wrong ensemble under
    # shift keeps epistemic LOW even as the skill decays (ADR-0002).
    monitor = SurpriseCompetenceMonitor(
        mastery_epistemic=0.1, flat_progress=0.02, min_updates=5,
        fast_rate=0.5, slow_rate=0.4, forget_factor=3.0, error_floor=0.01,
    )
    skill = "reach"
    assert monitor.is_forgetting(skill) is False  # never practiced -> not forgetting
    for _ in range(30):  # master it: low epistemic, low error
        monitor.update(_transition(0.02, skill, error=0.005))
    assert monitor.is_mastered(skill) is True
    assert monitor.is_forgetting(skill) is False  # mastered and accurate -> not forgetting
    # The skill decays: the model is now CONFIDENTLY WRONG — epistemic stays low
    # but prediction error climbs. Epistemic alone would miss this.
    for _ in range(30):
        monitor.update(_transition(0.02, skill, error=1.0))
    assert monitor.is_forgetting(skill) is True  # error risen far above the mastered floor


def test_is_forgetting_needs_prior_mastery_and_is_isolated() -> None:
    monitor = SurpriseCompetenceMonitor(min_updates=1, fast_rate=0.5, slow_rate=0.5)
    for _ in range(20):  # high epistemic throughout: never mastered
        monitor.update(_transition(1.0, "never"))
    assert monitor.is_forgetting("never") is False
    for _ in range(20):
        monitor.update(_transition(0.001, "solid"))  # mastered and stays low
    assert monitor.is_forgetting("solid") is False
