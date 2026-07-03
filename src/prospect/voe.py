"""Violation of expectation — the unifying signal (R3, R7). See ADR-0002.
Implemented in P3-001 (surprise, decomposition, mastery); forgetting detection is
P7-001.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Competence, LatentState, Prediction, Surprise, Transition


@dataclass
class _SkillStats:
    """Fast/slow EMAs of epistemic uncertainty; their gap is learning progress."""
    fast: float = 0.0
    slow: float = 0.0
    updates: int = 0

    def push(self, epistemic: float, fast_rate: float, slow_rate: float) -> None:
        if self.updates == 0:
            self.fast = self.slow = epistemic
        else:
            self.fast += fast_rate * (epistemic - self.fast)
            self.slow += slow_rate * (epistemic - self.slow)
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
    unknown. `is_forgetting` arrives with P7-001.

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
    ) -> None:
        self.mastery_epistemic = mastery_epistemic
        self.flat_progress = flat_progress
        self.min_updates = min_updates
        self.fast_rate, self.slow_rate = fast_rate, slow_rate
        self._skills: dict[str, _SkillStats] = {}

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
        stats.push(max(transition.prediction.epistemic, 0.0), self.fast_rate, self.slow_rate)

    def competence(self, skill: str) -> Competence:
        stats = self._skills.get(skill)
        if stats is None:
            return Competence(skill=skill, epistemic=float("inf"), learning_progress=0.0)
        progress = stats.slow - stats.fast
        mastered = (
            stats.updates >= self.min_updates
            and stats.fast <= self.mastery_epistemic
            and abs(progress) <= self.flat_progress
        )
        return Competence(skill=skill, epistemic=stats.fast, learning_progress=progress,
                          mastered=mastered)

    def is_mastered(self, skill: str) -> bool:
        return self.competence(skill).mastered

    def is_forgetting(self, skill: str) -> bool:
        raise NotImplementedError("P7-001")
