"""Violation of expectation — the unifying signal (R3, R7). See ADR-0002.
Tasks: P3-001 (mastery test), P7-001 (forgetting detection).
"""
from __future__ import annotations

from .types import Competence, LatentState, Prediction, Transition


class SurpriseCompetenceMonitor:
    """Computes calibrated surprise and tracks per-skill competence.

    Discipline (ADR-0002): surprise is -log_prob under the predicted *distribution*,
    and EPISTEMIC uncertainty (reducible) is tracked separately from aleatoric
    (noise). A skill is 'mastered' when epistemic is low and learning progress has
    flattened; 'forgetting' is epistemic rising again on a mastered skill -> trigger
    rehearsal (generative replay).

    Contract: interfaces.CompetenceMonitor.
    """

    def surprise(self, prediction: Prediction, observed: LatentState) -> float:
        raise NotImplementedError("P3-001")

    def update(self, transition: Transition) -> None:
        raise NotImplementedError("P3-001")

    def competence(self, skill: str) -> Competence:
        raise NotImplementedError("P3-001")

    def is_mastered(self, skill: str) -> bool:
        raise NotImplementedError("P3-001")

    def is_forgetting(self, skill: str) -> bool:
        raise NotImplementedError("P7-001")
