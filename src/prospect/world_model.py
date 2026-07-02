"""Latent predictive world model (R1, R4). See ADR-0001, ADR-0002.
First implementation task: tasks/P1-001-flat-world-model.md.
"""
from __future__ import annotations

from collections.abc import Sequence

from .types import Action, LatentState, Prediction


class FlatWorldModel:
    """Skeleton for the Phase-1 flat latent dynamics model.

    Contract (interfaces.WorldModel):
      - predict(): return a Prediction with a real distribution and an
        epistemic/aleatoric split (use an ensemble for epistemic).
      - Predict in *latent* space, not pixels (ADR-0001).
    """

    def predict(self, state: LatentState, action: Action) -> Prediction:
        raise NotImplementedError("P1-001")

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        raise NotImplementedError("P1-001")
