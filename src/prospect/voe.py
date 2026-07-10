"""Violation of expectation — the unifying signal (R3, R7). See ADR-0002.
Implemented across P3-001 (surprise, decomposition, mastery), P3-002 (the
learning-progress curriculum that owns the ADR-0007 mode flag) and P7-001
(error-based forgetting detection).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from .interfaces import CompetenceMonitor
from .types import Competence, LatentState, Mode, Prediction, Surprise, Transition


class AdaptiveThreshold:
    """Online adaptive-conformal threshold with a target exceedance rate.

    Before each update, consumers compare their score with :attr:`value`.  The
    update then raises the threshold after a trigger and lowers it after a
    non-trigger, using the task's decaying ``eta / sqrt(t)`` policy to reduce
    late-stream oscillation around the target quantile (U-003).
    """

    def __init__(self, alpha: float, eta: float, *, initial_value: float = 0.0) -> None:
        if not isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        if not isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not isfinite(initial_value):
            raise ValueError("initial_value must be finite")
        self.alpha = alpha
        self.eta = eta
        self._value = initial_value
        self._updates = 0
        self._triggers = 0

    @property
    def value(self) -> float:
        """The threshold to use for the next score."""
        return self._value

    @property
    def updates(self) -> int:
        """Number of nominal scores consumed."""
        return self._updates

    @property
    def trigger_rate(self) -> float:
        """Online pre-update exceedance rate over the consumed stream."""
        return self._triggers / self._updates if self._updates else 0.0

    def update(self, score: float) -> None:
        """Consume one score after testing it against the current value."""
        if not isfinite(score):
            raise ValueError("score must be finite")
        triggered = int(score > self._value)
        self._updates += 1
        self._triggers += triggered
        self._value += self.eta / sqrt(self._updates) * (triggered - self.alpha)


@dataclass
class _SkillStats:
    """Per-skill EMAs. Epistemic fast/slow (their gap is learning progress) drive
    MASTERY; a fast EMA of prediction ERROR drives FORGETTING. `mastered_error`
    latches the error at first mastery — the floor `is_forgetting` measures a rise
    against (`inf` = never mastered)."""
    fast: float = 0.0
    slow: float = 0.0
    fast_error: float = 0.0
    updates: int = 0
    mastered_epistemic: float = float("inf")
    mastered_error: float = float("inf")

    def push(self, epistemic: float, error: float, fast_rate: float, slow_rate: float) -> None:
        if self.updates == 0:
            self.fast = self.slow = epistemic
            self.fast_error = error
        else:
            self.fast += fast_rate * (epistemic - self.fast)
            self.slow += slow_rate * (epistemic - self.slow)
            self.fast_error += fast_rate * (error - self.fast_error)
        self.updates += 1


class SurpriseCompetenceMonitor:
    """Computes calibrated surprise and tracks per-skill competence (P3-001).

    Discipline (ADR-0002): surprise is -log_prob under the predicted
    *distribution*, returned as a decomposed `Surprise` (P0-002) — never a bare
    float. Attribution is by predictive-variance share, so
    `epistemic + aleatoric == total` exactly: a mastered skill's surprise reads
    as mostly aleatoric (noise), an unfamiliar state's as mostly epistemic.
    Known limit (ADR-0002): under hard distribution shift the ensemble can be
    confidently wrong — the expected-vs-violated differential rides on `total`;
    mastery rides on epistemic *uncertainty*, not error.

    "Learned" = low epistemic uncertainty + flattened learning progress: per-skill
    fast/slow EMAs of `prediction.epistemic`; progress = slow − fast (positive
    while improving). `update()` consumes latent-space transitions that carry the
    model's act-time `prediction`; per-skill attribution via `Transition.option`
    (P0-002); transitions without a prediction are ignored (nothing was expected).
    Unseen skill ⇒ epistemic `inf`, unmastered: never practiced = maximally
    unknown. `is_forgetting` (P7-001): prediction ERROR risen back up on a
    once-mastered skill — the signal that, in the full system, triggers rehearsal
    (generative replay, ADR-0002/0006). Forgetting keys on *error*, not epistemic,
    because under distribution shift the ensemble is often confidently WRONG — it
    agrees on a wrong answer, so epistemic (disagreement) does not rise even as the
    skill decays (ADR-0002, resolving the P0-010-flagged P7 concern). Mastery still
    keys on epistemic (learned = low uncertainty).

    Contract: interfaces.CompetenceMonitor.
    """

    DEFAULT_SKILL = "__task__"

    def __init__(
        self,
        mastery_epistemic: float = 0.05,
        flat_progress: float = 0.005,
        min_updates: int = 20,
        fast_rate: float = 0.1,
        slow_rate: float = 0.02,
        forget_factor: float = 3.0,
        error_floor: float = 0.01,
    ) -> None:
        self.mastery_epistemic = mastery_epistemic
        self.flat_progress = flat_progress
        self.min_updates = min_updates
        self.fast_rate, self.slow_rate = fast_rate, slow_rate
        self.forget_factor = forget_factor
        self.error_floor = error_floor
        self._skills: dict[str, _SkillStats] = {}

    def _mastered(self, stats: _SkillStats) -> bool:
        return (
            stats.updates >= self.min_updates
            and stats.fast <= self.mastery_epistemic
            and abs(stats.slow - stats.fast) <= self.flat_progress
        )

    @staticmethod
    def _prediction_error(prediction: Prediction, observed: LatentState) -> float:
        """Mean squared latent prediction error (pure Python; core stays
        dependency-free). Mismatched shapes ⇒ 0.0 (nothing to compare)."""
        mean = [float(m) for m in prediction.mean]
        obs = [float(x) for x in observed.z]
        if len(mean) != len(obs) or not mean:
            return 0.0
        return sum((m - x) ** 2 for m, x in zip(mean, obs, strict=True)) / len(mean)

    def surprise(self, prediction: Prediction, observed: LatentState) -> Surprise:
        total = -prediction.log_prob(observed.z)
        epistemic = max(prediction.epistemic, 0.0)
        aleatoric = max(prediction.aleatoric, 0.0)
        weight = epistemic / (epistemic + aleatoric) if epistemic + aleatoric > 0 else 0.0
        return Surprise(total=total, epistemic=weight * total, aleatoric=(1.0 - weight) * total)

    def update(self, transition: Transition) -> None:
        if transition.prediction is None:
            return  # nothing was expected — no violation to measure
        skill = transition.option.name if transition.option is not None else self.DEFAULT_SKILL
        stats = self._skills.setdefault(skill, _SkillStats())
        epistemic = max(transition.prediction.epistemic, 0.0)
        error = self._prediction_error(transition.prediction, transition.next_state)
        stats.push(epistemic, error, self.fast_rate, self.slow_rate)
        if stats.mastered_epistemic == float("inf") and self._mastered(stats):
            stats.mastered_epistemic = stats.fast  # latch the mastery floors...
            stats.mastered_error = stats.fast_error  # ...for forgetting detection

    def competence(self, skill: str) -> Competence:
        stats = self._skills.get(skill)
        if stats is None:
            return Competence(skill=skill, epistemic=float("inf"), learning_progress=0.0)
        return Competence(skill=skill, epistemic=stats.fast, learning_progress=stats.slow - stats.fast,
                          mastered=self._mastered(stats))

    def is_mastered(self, skill: str) -> bool:
        return self.competence(skill).mastered

    def is_forgetting(self, skill: str) -> bool:
        """Prediction ERROR risen back up on a once-mastered skill (ADR-0002). A
        skill that never mastered cannot be forgetting; a mastered one is forgetting
        when its fast-EMA error exceeds `forget_factor` x its mastered-error floor
        (floored by `error_floor` so a near-zero mastered error can't trip on noise).
        Error, not epistemic: a confidently-wrong ensemble under shift keeps
        epistemic low even as the skill decays (ADR-0002)."""
        stats = self._skills.get(skill)
        if stats is None or stats.mastered_error == float("inf"):
            return False
        return stats.fast_error > self.forget_factor * max(stats.mastered_error, self.error_floor)


class LearningProgressCurriculum:
    """The ADR-0007 arbiter (P3-002): decides the explore/exploit mode from
    learning progress; consumers read the mode (or the signed coefficient) and
    never pick the sign themselves.

    Rule: EXPLORE while the skill is still being learned; EXPLOIT once the
    monitor reports it mastered (low epistemic + flattened progress, ADR-0002).
    The explore bonus applies to *epistemic* uncertainty only — `Prediction`
    separates it from aleatoric, so noise is never rewarded (the noisy-TV
    defense, ADR-0002/0006).
    """

    def __init__(
        self,
        monitor: CompetenceMonitor,
        skill: str = SurpriseCompetenceMonitor.DEFAULT_SKILL,
        explore_bonus: float = 0.03,
        exploit_penalty: float = 0.03,
    ) -> None:
        self._monitor = monitor
        self._skill = skill
        self.explore_bonus = explore_bonus
        self.exploit_penalty = exploit_penalty

    def mode(self) -> Mode:
        return Mode.EXPLOIT if self._monitor.is_mastered(self._skill) else Mode.EXPLORE

    def uncertainty_coefficient(self) -> float:
        """The signed coefficient consumers apply to per-step epistemic
        uncertainty (e.g. `FlatPlanner.uncertainty_penalty`): positive = penalty
        (exploit-mode planning), negative = bonus (explore-mode collection)."""
        return self.exploit_penalty if self.mode() is Mode.EXPLOIT else -self.explore_bonus
