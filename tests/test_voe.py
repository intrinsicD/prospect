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
from prospect.voe import AdaptiveThreshold, LearningProgressCurriculum, SurpriseCompetenceMonitor


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


def test_adaptive_threshold_controls_the_tail_rate_and_tracks_a_shift() -> None:
    alpha = 0.1
    tracker = AdaptiveThreshold(alpha=alpha, eta=0.5)
    rng = np.random.default_rng(7)
    before = rng.normal(size=5_000)
    after = rng.normal(loc=3.0, size=10_000)
    triggered: list[bool] = []

    for score in before:
        triggered.append(float(score) > tracker.value)
        tracker.update(float(score))
    settled_before = tracker.value
    for score in after:
        triggered.append(float(score) > tracker.value)
        tracker.update(float(score))

    assert settled_before == pytest.approx(float(np.quantile(before, 1.0 - alpha)), abs=0.15)
    assert tracker.value == pytest.approx(float(np.quantile(after, 1.0 - alpha)), abs=0.15)
    assert float(np.mean(triggered[-5_000:])) == pytest.approx(alpha, abs=0.015)


def test_adaptive_threshold_uses_strict_pre_update_decisions_and_decaying_steps() -> None:
    tracker = AdaptiveThreshold(alpha=0.25, eta=2.0, initial_value=1.0)
    tracker.update(1.0)  # equality is not a trigger: 1 - 2 * .25
    assert tracker.value == pytest.approx(0.5)
    assert tracker.trigger_rate == 0.0

    tracker.update(0.6)  # judged against 0.5, then eta / sqrt(2) is applied
    assert tracker.value == pytest.approx(0.5 + 2.0 / np.sqrt(2.0) * 0.75)
    assert tracker.updates == 2
    assert tracker.trigger_rate == pytest.approx(0.5)


def test_adaptive_threshold_controls_a_one_percent_tail() -> None:
    alpha = 0.01
    tracker = AdaptiveThreshold(alpha=alpha, eta=0.5)
    scores = np.random.default_rng(17).normal(size=50_000)
    triggered: list[bool] = []

    for score in scores:
        triggered.append(float(score) > tracker.value)
        tracker.update(float(score))

    assert tracker.value == pytest.approx(float(np.quantile(scores, 1.0 - alpha)), abs=0.08)
    assert float(np.mean(triggered[-25_000:])) == pytest.approx(alpha, abs=0.002)


@pytest.mark.parametrize(
    ("alpha", "eta", "message"),
    [(0.0, 0.1, "alpha"), (1.0, 0.1, "alpha"), (0.1, 0.0, "eta")],
)
def test_adaptive_threshold_rejects_invalid_configuration(
    alpha: float, eta: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveThreshold(alpha=alpha, eta=eta)
    tracker = AdaptiveThreshold(alpha=0.1, eta=0.1)
    with pytest.raises(ValueError, match="score"):
        tracker.update(float("nan"))
    with pytest.raises(ValueError, match="initial_value"):
        AdaptiveThreshold(alpha=0.1, eta=0.1, initial_value=float("inf"))


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
    mastered_error = monitor._skills[skill].mastered_error
    # U-003 adapts termination/retrieval gates, never this latched reference.
    tracker = AdaptiveThreshold(alpha=0.1, eta=0.5)
    for score in np.linspace(0.0, 10.0, 100):
        tracker.update(float(score))
    assert monitor._skills[skill].mastered_error == mastered_error
    # The skill decays: the model is now CONFIDENTLY WRONG — epistemic stays low
    # but prediction error climbs. Epistemic alone would miss this.
    for _ in range(30):
        monitor.update(_transition(0.02, skill, error=1.0))
    assert monitor._skills[skill].mastered_error == mastered_error
    assert monitor.is_forgetting(skill) is True  # error risen far above the mastered floor


def test_is_forgetting_needs_prior_mastery_and_is_isolated() -> None:
    monitor = SurpriseCompetenceMonitor(min_updates=1, fast_rate=0.5, slow_rate=0.5)
    for _ in range(20):  # high epistemic throughout: never mastered
        monitor.update(_transition(1.0, "never"))
    assert monitor.is_forgetting("never") is False
    for _ in range(20):
        monitor.update(_transition(0.001, "solid"))  # mastered and stays low
    assert monitor.is_forgetting("solid") is False
