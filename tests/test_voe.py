"""Unit tests for the SurpriseCompetenceMonitor (P3-001): decomposition that sums
exactly, attribution that follows the uncertainty source, and the mastery state
machine (low epistemic + flattened progress), per ADR-0002."""
from __future__ import annotations

from math import isinf

import numpy as np
import pytest

from prospect.types import Action, LatentState, Option, Prediction, Transition
from prospect.voe import SurpriseCompetenceMonitor


def _pred(epistemic: float, aleatoric: float) -> Prediction:
    return Prediction(mean=np.zeros(2), var=np.ones(2), epistemic=epistemic, aleatoric=aleatoric)


def _transition(epistemic: float, skill: str | None = None,
                with_prediction: bool = True) -> Transition:
    return Transition(
        state=LatentState(z=np.zeros(2)),
        action=Action(data=np.zeros(1)),
        next_state=LatentState(z=np.zeros(2)),
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
