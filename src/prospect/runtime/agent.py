"""One authoritative runtime for decision, collection, assimilation, and learning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Never, Protocol, runtime_checkable

from prospect.decision import CounterIdentitySource
from prospect.domain import (
    AgentSnapshot,
    BeliefUpdate,
    BeliefUpdater,
    DecisionPolicy,
    DecisionRecord,
    EpistemicEffect,
    EpistemicTransition,
    ExecutedAction,
    ExperienceEvent,
    ExperienceKind,
    ExperienceStore,
    Goal,
    IntendedAction,
    Learner,
    Observation,
    Outcome,
    Scorer,
    TimePoint,
    UpdateReceipt,
)
from prospect.storage import EpistemicLedger

from .journal import (
    LifecycleJournal,
    LifecycleRecord,
    LifecycleStage,
)
from .state import AgentState, StateTransitionError


class RuntimeError(Exception):
    """Base error for lifecycle orchestration failures."""


class UnknownDecisionError(RuntimeError):
    """Observation handling did not receive the exact pending decision."""


class StepAlreadyObservedError(RuntimeError):
    """A run/episode/step identity was submitted more than once."""

    def __init__(
        self,
        message: str,
        *,
        lifecycle: tuple[LifecycleRecord, ...],
    ) -> None:
        super().__init__(message)
        self.lifecycle = lifecycle


class LifecycleFailureError(RuntimeError):
    """A step stopped after its lifecycle failure was recorded."""

    def __init__(self, record: LifecycleRecord) -> None:
        super().__init__(
            f"interaction failed while attempting {record.attempted_stage}: "
            f"{record.failure_type}: {record.failure_detail}"
        )
        self.record = record


@dataclass(frozen=True, slots=True)
class InteractionContext:
    """Stable identity and bootstrapping context for one environment step."""

    run_id: str
    task_id: str
    episode_id: str
    step_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("task_id", self.task_id),
            ("episode_id", self.episode_id),
        ):
            if not value or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if self.step_index < 0:
            raise ValueError("step_index must be nonnegative")

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.run_id, self.episode_id, self.step_index)


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    """Real environment response to one explicit intended action."""

    execution: ExecutedAction
    observation: Observation
    outcome: Outcome
    closed_at: TimePoint
    terminated: bool = False
    truncated: bool = False
    discount: float = 1.0

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("an environment step cannot be both terminated and truncated")
        if not isfinite(self.discount) or self.discount < 0.0:
            raise ValueError("discount must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class InteractionResult:
    """The complete real-experience and epistemic records created by one step."""

    decision: DecisionRecord
    experience: ExperienceEvent
    transition: EpistemicTransition


@runtime_checkable
class RuntimeEnvironment(Protocol):
    """Execute one intention and return externally evidenced results."""

    def step(self, intention: IntendedAction) -> EnvironmentStep: ...


@runtime_checkable
class EffectAssessor(Protocol):
    """Assess one realized posterior change."""

    def effect(self, update: BeliefUpdate) -> EpistemicEffect: ...


@runtime_checkable
class ReplayIndex(Protocol):
    """Optional lossy sampling index fed from the canonical experience store."""

    def add(self, event: ExperienceEvent) -> None: ...


@runtime_checkable
class StepExperienceStore(Protocol):
    """Canonical store that can look up an exact environment step."""

    def find_step(
        self,
        agent_id: str,
        run_id: str,
        episode_id: str,
        step_index: int,
    ) -> ExperienceEvent | None: ...


class EpistemicAgent:
    """Composition root with no hidden "last prediction" semantics.

    Decisions are returned as immutable records and must be passed back to
    :meth:`observe`.  A pending registry prevents replay or mismatch, but the
    prediction used for scoring always comes from the supplied decision itself.
    """

    def __init__(
        self,
        *,
        state: AgentState,
        policy: DecisionPolicy,
        belief_updater: BeliefUpdater,
        scorer: Scorer,
        effect_assessor: EffectAssessor,
        learner: Learner,
        experience_store: ExperienceStore,
        ledger: EpistemicLedger,
        identities: CounterIdentitySource,
        replay: ReplayIndex | None = None,
        lifecycle_journal: LifecycleJournal | None = None,
    ) -> None:
        self._state = state
        self._policy = policy
        self._belief_updater = belief_updater
        self._scorer = scorer
        self._effect_assessor = effect_assessor
        self._learner = learner
        self._experience_store = experience_store
        self._ledger = ledger
        self._identities = identities
        self._replay = replay
        self._lifecycle_journal = lifecycle_journal or LifecycleJournal()

    @property
    def agent_id(self) -> str:
        return self._state.agent_id

    def snapshot(self, at: TimePoint) -> AgentSnapshot:
        """Return the current frozen state without exposing the mutable custodian."""

        return self._state.snapshot(self.agent_id, at)

    def lifecycle(
        self,
        context: InteractionContext,
    ) -> tuple[LifecycleRecord, ...]:
        """Return immutable progress evidence; automatic continuation is not implemented."""

        return self._lifecycle_journal.history(
            agent_id=self.agent_id,
            run_id=context.run_id,
            episode_id=context.episode_id,
            step_index=context.step_index,
        )

    def decide(self, goal: Goal, *, at: TimePoint) -> DecisionRecord:
        """Select an action from exactly the state captured at ``at``."""

        snapshot = self._state.snapshot(self.agent_id, at)
        decision = self._policy.decide(snapshot, goal)
        if (
            decision.decided_at.clock_id != snapshot.captured_at.clock_id
            or decision.decided_at.tick < snapshot.captured_at.tick
        ):
            raise RuntimeError("policy decision precedes its frozen snapshot")
        try:
            self._state.register_decision(decision)
        except StateTransitionError as error:
            raise RuntimeError("policy returned an invalid decision") from error
        return decision

    def observe(
        self,
        decision: DecisionRecord,
        step: EnvironmentStep,
        *,
        context: InteractionContext,
    ) -> InteractionResult:
        """Persist one real step and derive its linked epistemic transition."""

        self._reject_stored_step(context)
        if context.task_id != decision.goal.task_id:
            raise RuntimeError("interaction task does not match the decision goal")
        if step.execution.intention.intention_id != decision.intended_action.intention_id:
            raise UnknownDecisionError("environment execution does not match the supplied decision")

        event = ExperienceEvent(
            experience_id=self._identities.next("experience"),
            agent_id=self.agent_id,
            run_id=context.run_id,
            task_id=context.task_id,
            episode_id=context.episode_id,
            step_index=context.step_index,
            kind=ExperienceKind.INTERACTION,
            observation=step.observation,
            outcome=step.outcome,
            terminated=step.terminated,
            truncated=step.truncated,
            discount=step.discount,
            behavior_policy_version=decision.policy_version,
            closed_at=step.closed_at,
            decision=decision,
            execution=step.execution,
        )
        try:
            self._state.consume_decision(decision)
        except StateTransitionError as error:
            raise UnknownDecisionError("observe requires the exact unconsumed pending decision") from error

        try:
            self._experience_store.append(event)
        except Exception as error:
            existing = self._find_stored_step(context)
            if existing is not None:
                self._record_discovered_experience(context, existing)
                self._raise_already_observed(context, existing)
            self._raise_lifecycle_failure(
                context=context,
                decision=decision,
                attempted_stage=LifecycleStage.EXPERIENCE_STORED,
                error=error,
                recorded_at=step.closed_at,
            )
        self._record_stage(
            context=context,
            decision=decision,
            stage=LifecycleStage.EXPERIENCE_STORED,
            recorded_at=event.closed_at,
            experience=event,
        )

        try:
            update = self._belief_updater.assimilate(decision.belief, event)
            score = self._scorer.score(
                decision.selected_assessment.prediction,
                event,
            )
            effect = self._effect_assessor.effect(update)
            created_at = _latest_time(
                "epistemic transition",
                (
                    event.closed_at,
                    update.updated_at,
                    score.scored_at,
                    effect.evaluated_at,
                ),
            )
            transition = EpistemicTransition(
                transition_id=self._identities.next("transition"),
                experience=event,
                belief_update=update,
                proper_scores=(score,),
                effects=(effect,),
                created_at=created_at,
            )
            self._ledger.append_transition(transition)
        except Exception as error:
            self._raise_lifecycle_failure(
                context=context,
                decision=decision,
                attempted_stage=LifecycleStage.TRANSITION_STORED,
                error=error,
                recorded_at=event.closed_at,
                experience=event,
            )
        self._record_stage(
            context=context,
            decision=decision,
            stage=LifecycleStage.TRANSITION_STORED,
            recorded_at=transition.created_at,
            experience=event,
            transition=transition,
        )

        try:
            self._state.assimilate(update)
            self._state.add_resource(
                "environment_steps",
                1.0,
                "step",
                at=created_at,
            )
        except Exception as error:
            self._raise_lifecycle_failure(
                context=context,
                decision=decision,
                attempted_stage=LifecycleStage.STATE_APPLIED,
                error=error,
                recorded_at=transition.created_at,
                experience=event,
                transition=transition,
            )
        self._record_stage(
            context=context,
            decision=decision,
            stage=LifecycleStage.STATE_APPLIED,
            recorded_at=transition.created_at,
            experience=event,
            transition=transition,
        )

        if self._replay is not None:
            try:
                self._replay.add(event)
            except Exception as error:
                self._raise_lifecycle_failure(
                    context=context,
                    decision=decision,
                    attempted_stage=LifecycleStage.REPLAY_INDEXED,
                    error=error,
                    recorded_at=transition.created_at,
                    experience=event,
                    transition=transition,
                )
            self._record_stage(
                context=context,
                decision=decision,
                stage=LifecycleStage.REPLAY_INDEXED,
                recorded_at=transition.created_at,
                experience=event,
                transition=transition,
            )
        return InteractionResult(
            decision=decision,
            experience=event,
            transition=transition,
        )

    def interact(
        self,
        environment: RuntimeEnvironment,
        goal: Goal,
        *,
        context: InteractionContext,
        decide_at: TimePoint,
    ) -> InteractionResult:
        """Run the sole authoritative decide → step → observe path."""

        decision = self.decide(goal, at=decide_at)
        step = environment.step(decision.intended_action)
        return self.observe(decision, step, context=context)

    def learn(
        self,
        transitions: Sequence[EpistemicTransition],
        *,
        at: TimePoint,
    ) -> UpdateReceipt:
        """Apply an explicit learner update to a declared transition sequence."""

        inputs = tuple(transitions)
        if not inputs:
            raise ValueError("learning requires at least one epistemic transition")
        snapshot = self._state.snapshot(self.agent_id, at)
        receipt = self._learner.update(snapshot, inputs)
        if len(receipt.transitions) != len(inputs) or any(
            received is not supplied for received, supplied in zip(receipt.transitions, inputs, strict=True)
        ):
            raise RuntimeError("learner receipt does not identify exactly the supplied transitions")
        if (
            receipt.started_at.clock_id != snapshot.captured_at.clock_id
            or receipt.started_at.tick < snapshot.captured_at.tick
        ):
            raise RuntimeError("learner receipt starts before its frozen input snapshot")
        try:
            self._state.validate_update(receipt)
        except StateTransitionError as error:
            raise RuntimeError("learner returned an invalid version transition") from error
        self._ledger.append_update(receipt)
        try:
            self._state.apply_update(receipt)
        except StateTransitionError as error:
            raise RuntimeError("prevalidated learner receipt became invalid before state apply") from error
        return receipt

    def _reject_stored_step(self, context: InteractionContext) -> None:
        stored = self._find_stored_step(context)
        history = self.lifecycle(context)
        if stored is None and not history:
            return
        if stored is not None and not history:
            self._record_discovered_experience(context, stored)
        self._raise_already_observed(context, stored)

    def _find_stored_step(
        self,
        context: InteractionContext,
    ) -> ExperienceEvent | None:
        if not isinstance(self._experience_store, StepExperienceStore):
            return None
        return self._experience_store.find_step(
            self.agent_id,
            context.run_id,
            context.episode_id,
            context.step_index,
        )

    def _record_discovered_experience(
        self,
        context: InteractionContext,
        event: ExperienceEvent,
    ) -> None:
        if self.lifecycle(context):
            return
        decision_id = "unknown-decision" if event.decision is None else event.decision.decision_id
        self._lifecycle_journal.append_stage(
            agent_id=self.agent_id,
            run_id=event.run_id,
            task_id=event.task_id,
            episode_id=event.episode_id,
            step_index=event.step_index,
            decision_id=decision_id,
            stage=LifecycleStage.EXPERIENCE_STORED,
            recorded_at=event.closed_at,
            experience_id=event.experience_id,
        )

    def _raise_already_observed(
        self,
        context: InteractionContext,
        event: ExperienceEvent | None,
    ) -> Never:
        history = self.lifecycle(context)
        experience_id = (
            next(
                (record.experience_id for record in reversed(history) if record.experience_id is not None),
                None,
            )
            if event is None
            else event.experience_id
        )
        suffix = "" if experience_id is None else f" as {experience_id}"
        raise StepAlreadyObservedError(
            f"step {context.run_id}/{context.episode_id}/{context.step_index} "
            f"was already stored{suffix}; automatic continuation is not implemented",
            lifecycle=history,
        )

    def _record_stage(
        self,
        *,
        context: InteractionContext,
        decision: DecisionRecord,
        stage: LifecycleStage,
        recorded_at: TimePoint,
        experience: ExperienceEvent,
        transition: EpistemicTransition | None = None,
    ) -> LifecycleRecord:
        return self._lifecycle_journal.append_stage(
            agent_id=self.agent_id,
            run_id=context.run_id,
            task_id=context.task_id,
            episode_id=context.episode_id,
            step_index=context.step_index,
            decision_id=decision.decision_id,
            stage=stage,
            recorded_at=recorded_at,
            experience_id=experience.experience_id,
            transition_id=(None if transition is None else transition.transition_id),
        )

    def _raise_lifecycle_failure(
        self,
        *,
        context: InteractionContext,
        decision: DecisionRecord,
        attempted_stage: LifecycleStage,
        error: BaseException,
        recorded_at: TimePoint,
        experience: ExperienceEvent | None = None,
        transition: EpistemicTransition | None = None,
    ) -> Never:
        record = self._lifecycle_journal.append_failure(
            agent_id=self.agent_id,
            run_id=context.run_id,
            task_id=context.task_id,
            episode_id=context.episode_id,
            step_index=context.step_index,
            decision_id=decision.decision_id,
            attempted_stage=attempted_stage,
            error=error,
            recorded_at=recorded_at,
            experience_id=(None if experience is None else experience.experience_id),
            transition_id=(None if transition is None else transition.transition_id),
        )
        raise LifecycleFailureError(record) from error


def _latest_time(label: str, points: Sequence[TimePoint]) -> TimePoint:
    clocks = {point.clock_id for point in points}
    if len(clocks) != 1:
        raise RuntimeError(f"{label} inputs use different clocks: {sorted(clocks)}")
    latest = max(points, key=lambda point: point.tick)
    return TimePoint(tick=latest.tick, clock_id=latest.clock_id)


__all__ = (
    "EffectAssessor",
    "EnvironmentStep",
    "EpistemicAgent",
    "InteractionContext",
    "InteractionResult",
    "LifecycleFailureError",
    "ReplayIndex",
    "RuntimeEnvironment",
    "RuntimeError",
    "StepAlreadyObservedError",
    "StepExperienceStore",
    "UnknownDecisionError",
)
