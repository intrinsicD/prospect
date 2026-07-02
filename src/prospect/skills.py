"""Skill library + router (R5). Options carry a *predictive precondition*; selection
is simulate-to-match: roll each candidate forward and pick the lowest-uncertainty
match to the subgoal. Only competence-gated (mastered) skills are offered upward.
See ADR-0002 (competence) and ADR-0003 (options as the high-level action space).
Task: P4-001.
"""
from __future__ import annotations

from .types import LatentState, Option, Subgoal


class SkillRouter:
    """Contract: interfaces.SkillLibrary."""

    def propose(self, state: LatentState, subgoal: Subgoal) -> list[Option]:
        raise NotImplementedError("P4-001")

    def add(self, option: Option) -> None:
        raise NotImplementedError("P4-001")
