"""Backend-neutral behavior contracts over :mod:`prospect.domain.records`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .records import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    DecisionRecord,
    EpistemicEffect,
    EpistemicTarget,
    EpistemicTransition,
    ExperienceEvent,
    Goal,
    InformationValue,
    Prediction,
    ProperScore,
    TimePoint,
    UpdateReceipt,
)


@runtime_checkable
class BeliefUpdater(Protocol):
    """Assimilate real experience into a linked posterior belief."""

    def assimilate(self, prior: Belief, experience: ExperienceEvent) -> BeliefUpdate: ...


@runtime_checkable
class PredictiveModel(Protocol):
    """Produce a qualified action-conditional prediction from a prior belief."""

    def predict(
        self,
        belief: Belief,
        action: Action,
        target: EpistemicTarget,
        horizon_end: TimePoint,
    ) -> Prediction: ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """Select and justify an intended action under a goal."""

    def decide(self, snapshot: AgentSnapshot, goal: Goal) -> DecisionRecord: ...


@runtime_checkable
class Learner(Protocol):
    """Propose or apply a persistent update from completed epistemic transitions."""

    def update(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
    ) -> UpdateReceipt: ...


@runtime_checkable
class ExperienceStore(Protocol):
    """Append-only canonical real-experience storage."""

    def append(self, event: ExperienceEvent) -> None: ...

    def get(self, experience_id: str) -> ExperienceEvent: ...

    def history(self, agent_id: str, as_of: TimePoint) -> Sequence[ExperienceEvent]: ...


@runtime_checkable
class Scorer(Protocol):
    """Score a prospective distribution against linked realized evidence."""

    def score(self, prediction: Prediction, experience: ExperienceEvent) -> ProperScore: ...


@runtime_checkable
class InformationEvaluator(Protocol):
    """Estimate prospective information value and assess realized belief change."""

    def value(
        self,
        belief: Belief,
        action: Action,
        target: EpistemicTarget,
        horizon_end: TimePoint,
    ) -> InformationValue: ...

    def effect(self, update: BeliefUpdate) -> EpistemicEffect: ...


@runtime_checkable
class SnapshotProvider(Protocol):
    """Capture a coherent immutable agent-state view."""

    def snapshot(self, agent_id: str, at: TimePoint) -> AgentSnapshot: ...


__all__ = (
    "BeliefUpdater",
    "DecisionPolicy",
    "ExperienceStore",
    "InformationEvaluator",
    "Learner",
    "PredictiveModel",
    "Scorer",
    "SnapshotProvider",
)
