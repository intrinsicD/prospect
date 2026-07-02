"""Planning (R1, R2). Flat MPC in imagination, plus the hierarchical manager over a
jumpy option-model. See ADR-0001, ADR-0003. Tasks: P2-001, P5-001, P5-002.
"""
from __future__ import annotations

from .types import Action, LatentState, Option, Prediction, Subgoal, Transition


class FlatPlanner:
    """Phase-2: optimise an imagined action sequence (CEM/MPC) over the world model,
    then act on the first action (receding horizon). Contract: interfaces.Planner."""

    def plan(self, state: LatentState, goal: Subgoal | None = None) -> Action:
        raise NotImplementedError("P2-001")


class JumpyOptionModel:
    """Phase-5: the temporally-abstract model — predicts the outcome of committing to
    an option (landing latent, cumulative reward, duration, uncertainty). This is what
    turns hierarchy from reactive control into hierarchical *planning* (ADR-0003).

    Contract: interfaces.OptionModel.
    """

    def predict_option(self, state: LatentState, option: Option) -> Prediction:
        raise NotImplementedError("P5-001")


class HierarchicalManager:
    """Phase-5: search over the JumpyOptionModel, emit an option/subgoal; the worker
    executes it; VoE terminates it early on a surprise spike.

    Contract: interfaces.HierarchicalPlanner.
    """

    def plan_option(self, state: LatentState) -> Option:
        raise NotImplementedError("P5-002")

    def should_terminate(self, transition: Transition) -> bool:
        # Terminate when the option's predicted trajectory is violated (VoE).
        raise NotImplementedError("P5-002")
