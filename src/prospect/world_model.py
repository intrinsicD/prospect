"""Latent predictive world model (R1, R4). See ADR-0001, ADR-0002.
First implementation task: tasks/P1-001-flat-world-model.md.
"""
from __future__ import annotations

from collections.abc import Sequence

from .types import Action, LatentState, Prediction, Transition


class FlatWorldModel:
    """Skeleton for the Phase-1 flat latent dynamics model.

    Contracts (interfaces.WorldModel + interfaces.Learner):
      - predict(): return a Prediction with a real distribution and an
        epistemic/aleatoric split (use an ensemble for epistemic).
      - Predict in *latent* space, not pixels (ADR-0001).
      - update(): train from a batch of transitions; return the metrics dict
        (losses + integrity stats) the harness logs for the sentinels (P0-003).
    """

    def predict(self, state: LatentState, action: Action) -> Prediction:
        raise NotImplementedError("P1-001")

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        raise NotImplementedError("P1-001")

    def update(self, batch: Sequence[Transition]) -> dict[str, float]:
        raise NotImplementedError("P1-001")
