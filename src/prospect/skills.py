"""Skill library + router (R5) — implemented in P4-001. Options carry a policy
and horizon (types.Option); selection is simulate-to-match: roll each candidate
forward under the world model and pick the lowest-cost match to the subgoal.
Only competence-gated (mastered) skills are offered upward. See ADR-0002
(competence, skill trust) and ADR-0003 (options as the high-level action space).
"""
from __future__ import annotations

import numpy as np

from .interfaces import CompetenceMonitor, WorldModel
from .types import LatentState, Option, Prediction, Subgoal


class SkillRouter:
    """Skill library + simulate-to-select router (P4-001).

    Selection *is* the predictive precondition (ADR-0002/0003): each candidate
    option is rolled forward under the world model for its horizon, and its score
    is `||imagined landing − subgoal||² + uncertainty_weight · Σ epistemic`. An
    option is applicable where its outcome is predictable (low epistemic) and
    useful (lands near the goal) — no stored predicate could say more.

    Gating (ADR-0003): with a monitor attached, only mastered skills are offered
    upward — except cold start (nothing mastered yet), where everything is
    offered so competence can accrue. No monitor = no gating.

    `simulate()` (the flat-rollout landing prediction) is the primitive that
    P5's learned jumpy option-model replaces.

    Contract: interfaces.SkillLibrary.
    """

    def __init__(
        self,
        world_model: WorldModel | None = None,
        monitor: CompetenceMonitor | None = None,
        uncertainty_weight: float = 1.0,
    ) -> None:
        self._model = world_model
        self._monitor = monitor
        self.uncertainty_weight = uncertainty_weight
        self._options: dict[str, Option] = {}

    def add(self, option: Option) -> None:
        if option.policy is None:
            raise ValueError(
                f"option {option.name!r} has no policy — it cannot be simulated or executed"
            )
        self._options[option.name] = option

    def propose(self, state: LatentState, subgoal: Subgoal) -> list[Option]:
        """Candidates ranked by simulate-to-match score (best first)."""
        if self._model is None:
            raise ValueError("simulate-to-select needs a world model to roll options forward")
        candidates = list(self._options.values())
        if self._monitor is not None:
            mastered = [o for o in candidates if self._monitor.is_mastered(o.name)]
            if mastered:  # cold start: nothing mastered yet -> offer everything
                candidates = mastered
        return sorted(candidates, key=lambda option: self._score(state, option, subgoal))

    def simulate(self, state: LatentState, option: Option) -> tuple[Prediction, float]:
        """Roll the option's policy forward `horizon` steps in imagination:
        returns (the landing-step Prediction, accumulated epistemic along the way).
        This is what execution is later judged against (misapplication = VoE spike)."""
        if self._model is None:
            raise ValueError("simulate-to-select needs a world model to roll options forward")
        if option.policy is None:
            raise ValueError(f"option {option.name!r} has no policy")
        latent = state
        epistemic = 0.0
        prediction: Prediction | None = None
        for _ in range(option.horizon):
            prediction = self._model.predict(latent, option.policy(latent))
            epistemic += prediction.epistemic
            latent = LatentState(z=prediction.mean)
        assert prediction is not None  # horizon >= 1
        return prediction, epistemic

    def _score(self, state: LatentState, option: Option, subgoal: Subgoal) -> float:
        prediction, epistemic = self.simulate(state, option)
        landing = np.asarray(prediction.mean, dtype=float)
        target = np.asarray(subgoal.target.z, dtype=float)
        return float(np.mean((landing - target) ** 2)) + self.uncertainty_weight * epistemic
