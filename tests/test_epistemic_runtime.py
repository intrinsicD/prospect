from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from prospect.decision import CounterIdentitySource, MaxValuePolicy
from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    CandidateAssessment,
    DecisionRecord,
    Distribution,
    EpistemicEffect,
    EpistemicEffectKind,
    EpistemicTarget,
    EpistemicTransition,
    Evidence,
    EvidenceLineage,
    EvidenceOrigin,
    ExecutedAction,
    ExecutionStatus,
    ExperienceEvent,
    Goal,
    InformationSet,
    InformationValue,
    IntendedAction,
    Observation,
    Outcome,
    Prediction,
    ProperScore,
    Provenance,
    ResourceLedger,
    TimePoint,
    TrustLevel,
    UpdateReceipt,
    UpdateStatus,
    Utility,
)
from prospect.runtime import (
    AgentState,
    EnvironmentStep,
    EpistemicAgent,
    InteractionContext,
    LifecycleFailureError,
    LifecycleStage,
    StepAlreadyObservedError,
    UnknownDecisionError,
)
from prospect.runtime import (
    RuntimeError as AgentRuntimeError,
)
from prospect.storage import EpistemicLedger, InMemoryExperienceStore


def _time(tick: int) -> TimePoint:
    return TimePoint(tick=tick)


def _evidence(evidence_id: str, *, tick: int, payload: object) -> Evidence:
    point = _time(tick)
    return Evidence(
        evidence_id=evidence_id,
        payload=payload,
        occurred_at=point,
        available_at=point,
        lineage=EvidenceLineage(
            evidence_id=evidence_id,
            origin=EvidenceOrigin.OBSERVED,
            provenance=Provenance(
                source_id="runtime-test-environment",
                trust=TrustLevel.VERIFIED,
                source_kind="fixture",
            ),
        ),
    )


def _initial_snapshot() -> AgentSnapshot:
    target = EpistemicTarget(
        target_id="binary-outcome",
        description="binary outcome after the selected action",
        target_kind="future_outcome",
    )
    initial_observation = Observation(
        observation_id="observation-initial",
        agent_id="agent",
        modality="state",
        evidence=_evidence(
            "observation-initial",
            tick=0,
            payload={"index": 0},
        ),
    )
    information = InformationSet(
        information_set_id="information-initial",
        agent_id="agent",
        as_of=_time(0),
        observations=(initial_observation,),
        memory_version="memory-v0",
    )
    belief = Belief(
        belief_id="belief-initial",
        agent_id="agent",
        target=target,
        information_set=information,
        distribution=Distribution(
            distribution_id="belief-distribution-initial",
            family="categorical",
            support="binary",
            parameters=(0.5, 0.5),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        formed_at=_time(0),
        model_version="model-v1",
        representation_version="representation-v1",
    )
    return AgentSnapshot(
        snapshot_id="snapshot-initial",
        agent_id="agent",
        captured_at=_time(0),
        belief=belief,
        configuration_version="configuration-v1",
        memory_version="memory-v0",
        knowledge_version="knowledge-v1",
        model_version="model-v1",
        representation_version="representation-v1",
        policy_version="policy-v1",
        resources=ResourceLedger(
            ledger_id="resources-initial",
            started_at=_time(0),
            completed_at=_time(0),
        ),
    )


def _goal(target: EpistemicTarget) -> Goal:
    return Goal(
        goal_id="goal",
        task_id="task",
        target=target,
        description="choose the likely binary outcome",
        issued_at=_time(0),
        preference_version="preference-v1",
    )


class _Assessor:
    def __init__(self) -> None:
        self.calls = 0

    def assess(
        self,
        snapshot: AgentSnapshot,
        goal: Goal,
    ) -> tuple[CandidateAssessment, ...]:
        candidate_index = self.calls
        self.calls += 1
        suffix = f"{snapshot.belief.belief_id}-{candidate_index}"
        action = Action(
            action_id=f"action-{candidate_index}",
            action_kind="choose",
            parameters={"index": 1},
        )
        prediction = Prediction(
            prediction_id=f"prediction-{suffix}",
            prior_belief=snapshot.belief,
            action=action,
            target=goal.target,
            distribution=Distribution(
                distribution_id=f"prediction-distribution-{suffix}",
                family="categorical",
                support="binary",
                parameters=(0.25, 0.75),
                representation_version=snapshot.representation_version,
                event_shape=(2,),
            ),
            issued_at=snapshot.captured_at,
            horizon_end=_time(snapshot.captured_at.tick + 2),
            model_version=snapshot.model_version,
            representation_version=snapshot.representation_version,
            calibration_version="uncalibrated",
        )
        utility = Utility(
            utility_id=f"utility-{suffix}",
            goal_id=goal.goal_id,
            prediction_id=prediction.prediction_id,
            expected_value=0.75,
            unit="decision_value",
            evaluator_version="utility-v1",
            assessed_at=snapshot.captured_at,
        )
        information = InformationValue(
            information_value_id=f"information-value-{suffix}",
            prior_belief_id=snapshot.belief.belief_id,
            action_id=action.action_id,
            target_id=goal.target.target_id,
            expected_reduction=0.0,
            expected_cost=0.0,
            unit="decision_value",
            evaluator_version="information-v1",
            assessed_at=snapshot.captured_at,
        )
        return (
            CandidateAssessment(
                assessment_id=f"assessment-{suffix}",
                action=action,
                prediction=prediction,
                utility=utility,
                information_value=information,
                expected_action_cost=0.0,
                expected_risk=0.0,
                admissible=True,
                constraint_reasons=(),
                constraint_penalty=0.0,
                total_value=0.75,
                unit="decision_value",
                evaluator_version="candidate-v1",
                assessed_at=snapshot.captured_at,
            ),
        )


class _Updater:
    def __init__(self, *, failure: str | None = None) -> None:
        self.failure = failure

    def assimilate(
        self,
        prior: Belief,
        experience: ExperienceEvent,
    ) -> BeliefUpdate:
        if self.failure is not None:
            raise ValueError(self.failure)
        observations = (*prior.information_set.observations, experience.observation)
        information = InformationSet(
            information_set_id=f"information-{experience.experience_id}",
            agent_id=prior.agent_id,
            as_of=experience.closed_at,
            observations=observations,
            memory_version=f"memory-step-{experience.step_index + 1}",
        )
        posterior = Belief(
            belief_id=f"belief-{experience.experience_id}",
            agent_id=prior.agent_id,
            target=prior.target,
            information_set=information,
            distribution=Distribution(
                distribution_id=f"belief-distribution-{experience.experience_id}",
                family="categorical",
                support="binary",
                parameters=(0.1, 0.9),
                representation_version=prior.representation_version,
                event_shape=(2,),
            ),
            formed_at=_time(experience.closed_at.tick + 1),
            model_version=prior.model_version,
            representation_version=prior.representation_version,
        )
        return BeliefUpdate(
            update_id=f"belief-update-{experience.experience_id}",
            prior=prior,
            experience=experience,
            posterior=posterior,
            updater_version="updater-v1",
            updated_at=posterior.formed_at,
        )


class _TrackingScorer:
    def __init__(self) -> None:
        self.predictions: list[Prediction] = []

    def score(
        self,
        prediction: Prediction,
        experience: ExperienceEvent,
    ) -> ProperScore:
        self.predictions.append(prediction)
        return ProperScore(
            score_id=f"score-{experience.experience_id}",
            prediction_id=prediction.prediction_id,
            realized_evidence_id=experience.outcome.evidence.evidence_id,
            rule="log_score",
            value=0.25,
            unit="nats",
            scorer_version="scorer-v1",
            scored_at=experience.closed_at,
        )


@dataclass(frozen=True, slots=True)
class _EffectAssessor:
    def effect(self, update: BeliefUpdate) -> EpistemicEffect:
        return EpistemicEffect(
            effect_id=f"effect-{update.update_id}",
            belief_update_id=update.update_id,
            target_id=update.prior.target.target_id,
            kind=EpistemicEffectKind.INFORMATION_GAIN,
            measure="categorical_entropy",
            before=1.0,
            after=0.5,
            improvement=0.5,
            higher_is_better=False,
            evaluator_version="effect-v1",
            evaluated_at=update.updated_at,
        )


class _Learner:
    def __init__(self, *, stale_previous: bool = False) -> None:
        self.stale_previous = stale_previous

    def update(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
    ) -> UpdateReceipt:
        return UpdateReceipt(
            receipt_id="receipt-valid",
            agent_id=snapshot.agent_id,
            transitions=tuple(transitions),
            learner_version="learner-v1",
            status=UpdateStatus.APPLIED,
            previous_configuration_version=(
                "configuration-stale" if self.stale_previous else snapshot.configuration_version
            ),
            new_configuration_version="configuration-v2",
            previous_model_version=snapshot.model_version,
            new_model_version=snapshot.model_version,
            previous_representation_version=snapshot.representation_version,
            new_representation_version=snapshot.representation_version,
            previous_policy_version=snapshot.policy_version,
            new_policy_version=snapshot.policy_version,
            started_at=snapshot.captured_at,
            completed_at=_time(snapshot.captured_at.tick + 1),
        )


class _DroppingEvidenceLearner:
    def update(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
    ) -> UpdateReceipt:
        source = transitions[-1].belief_update.posterior
        completed_at = _time(snapshot.captured_at.tick + 1)
        resulting = replace(
            source,
            belief_id="belief-result-dropping-current-evidence",
            distribution=replace(
                source.distribution,
                distribution_id="distribution-result-model-v2",
            ),
            formed_at=completed_at,
            model_version="model-v2",
        )
        return UpdateReceipt(
            receipt_id="receipt-dropping-current-evidence",
            agent_id=snapshot.agent_id,
            transitions=tuple(transitions),
            learner_version="learner-v1",
            status=UpdateStatus.APPLIED,
            previous_configuration_version=snapshot.configuration_version,
            new_configuration_version="configuration-v2",
            previous_model_version=snapshot.model_version,
            new_model_version="model-v2",
            previous_representation_version=snapshot.representation_version,
            new_representation_version=snapshot.representation_version,
            previous_policy_version=snapshot.policy_version,
            new_policy_version=snapshot.policy_version,
            started_at=snapshot.captured_at,
            completed_at=completed_at,
            resulting_belief=resulting,
        )


class _Replay:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.events: list[ExperienceEvent] = []

    def add(self, event: ExperienceEvent) -> None:
        self.calls += 1
        if self.fail:
            raise ValueError("api_key=super-secret replay unavailable")
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class _RuntimeFixture:
    agent: EpistemicAgent
    state: AgentState
    store: InMemoryExperienceStore
    ledger: EpistemicLedger
    scorer: _TrackingScorer
    replay: _Replay
    goal: Goal


def _runtime(
    *,
    updater_failure: str | None = None,
    replay_failure: bool = False,
    learner: _Learner | _DroppingEvidenceLearner | None = None,
) -> _RuntimeFixture:
    initial = _initial_snapshot()
    state = AgentState(initial)
    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    scorer = _TrackingScorer()
    replay = _Replay(fail=replay_failure)
    identities = CounterIdentitySource("runtime")
    policy = MaxValuePolicy(
        agent_id=initial.agent_id,
        policy_version=initial.policy_version,
        assessor=_Assessor(),
        identities=identities,
    )
    agent = EpistemicAgent(
        state=state,
        policy=policy,
        belief_updater=_Updater(failure=updater_failure),
        scorer=scorer,
        effect_assessor=_EffectAssessor(),
        learner=_Learner() if learner is None else learner,
        experience_store=store,
        ledger=ledger,
        identities=identities,
        replay=replay,
    )
    return _RuntimeFixture(
        agent=agent,
        state=state,
        store=store,
        ledger=ledger,
        scorer=scorer,
        replay=replay,
        goal=_goal(initial.belief.target),
    )


def _step(
    decision: DecisionRecord,
    *,
    suffix: str,
    closed_at: int,
    terminated: bool = False,
    truncated: bool = False,
) -> EnvironmentStep:
    return _step_for_intention(
        decision.intended_action,
        suffix=suffix,
        closed_at=closed_at,
        terminated=terminated,
        truncated=truncated,
    )


def _step_for_intention(
    intention: IntendedAction,
    *,
    suffix: str,
    closed_at: int,
    terminated: bool = False,
    truncated: bool = False,
) -> EnvironmentStep:
    execution = ExecutedAction(
        execution_id=f"execution-{suffix}",
        intention=intention,
        status=ExecutionStatus.SUCCEEDED,
        started_at=_time(closed_at - 1),
        ended_at=_time(closed_at),
        realized_action=intention.action,
    )
    observation = Observation(
        observation_id=f"observation-{suffix}",
        agent_id=intention.agent_id,
        modality="state",
        evidence=_evidence(
            f"observation-{suffix}",
            tick=closed_at,
            payload={"index": 1},
        ),
    )
    outcome = Outcome(
        outcome_id=f"outcome-{suffix}",
        execution_id=execution.execution_id,
        evidence=_evidence(
            f"outcome-evidence-{suffix}",
            tick=closed_at,
            payload={"index": 1},
        ),
    )
    return EnvironmentStep(
        execution=execution,
        observation=observation,
        outcome=outcome,
        closed_at=_time(closed_at),
        terminated=terminated,
        truncated=truncated,
        discount=0.99,
    )


class _Environment:
    def __init__(self) -> None:
        self.received: IntendedAction | None = None

    def step(self, intention: IntendedAction) -> EnvironmentStep:
        self.received = intention
        return _step_for_intention(
            intention,
            suffix="interact",
            closed_at=2,
        )


def test_interact_runs_the_authoritative_decide_step_observe_path() -> None:
    fixture = _runtime()
    environment = _Environment()

    result = fixture.agent.interact(
        environment,
        fixture.goal,
        context=InteractionContext("run", "task", "episode", 0),
        decide_at=_time(0),
    )

    assert environment.received is result.decision.intended_action
    assert result.experience.execution is not None
    assert result.experience.execution.intention is environment.received
    assert result.transition.experience is result.experience


def test_decide_step_observe_is_exact_canonical_and_advances_state() -> None:
    fixture = _runtime()
    context = InteractionContext(
        run_id="run",
        task_id="task",
        episode_id="episode",
        step_index=0,
    )
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    step = _step(decision, suffix="0", closed_at=2)

    result = fixture.agent.observe(decision, step, context=context)
    snapshot = fixture.agent.snapshot(_time(3))

    assert result.experience.decision is decision
    assert result.experience.execution is step.execution
    assert fixture.store.get(result.experience.experience_id) is result.experience
    assert fixture.store.get_step("agent", "run", "episode", 0) is result.experience
    assert fixture.ledger.get_transition(result.transition.transition_id) is result.transition
    assert result.transition.belief_update.experience is result.experience
    assert result.transition.proper_scores[0].prediction_id == (decision.selected_assessment.prediction.prediction_id)
    assert fixture.scorer.predictions == [decision.selected_assessment.prediction]
    assert snapshot.belief is result.transition.belief_update.posterior
    assert snapshot.memory_version == snapshot.belief.information_set.memory_version
    assert snapshot.resources.uses[0].amount == 1.0
    assert fixture.replay.events == [result.experience]
    assert [record.stage for record in fixture.agent.lifecycle(context)] == [
        LifecycleStage.EXPERIENCE_STORED,
        LifecycleStage.TRANSITION_STORED,
        LifecycleStage.STATE_APPLIED,
        LifecycleStage.REPLAY_INDEXED,
    ]


def test_runtime_has_no_hidden_last_decision_scoring_semantics() -> None:
    fixture = _runtime()
    first = fixture.agent.decide(fixture.goal, at=_time(0))
    second = fixture.agent.decide(fixture.goal, at=_time(0))
    assert first.selected_assessment.prediction.prediction_id != (second.selected_assessment.prediction.prediction_id)

    result = fixture.agent.observe(
        first,
        _step(first, suffix="first", closed_at=2),
        context=InteractionContext("run", "task", "episode", 0),
    )

    assert fixture.scorer.predictions == [first.selected_assessment.prediction]
    assert result.transition.proper_scores[0].prediction_id != (second.selected_assessment.prediction.prediction_id)
    assert not hasattr(fixture.agent, "_last_prediction")
    assert not hasattr(fixture.agent, "_last_decision")


def test_snapshot_is_a_pure_deterministic_read() -> None:
    fixture = _runtime()

    first = fixture.agent.snapshot(_time(0))
    second = fixture.agent.snapshot(_time(0))

    assert first == second
    assert first.snapshot_id == second.snapshot_id
    fixture.agent.decide(fixture.goal, at=_time(0))
    with_pending = fixture.agent.snapshot(_time(0))
    repeated = fixture.agent.snapshot(_time(0))
    assert with_pending == repeated
    assert with_pending.snapshot_id != first.snapshot_id


def test_mismatched_and_consumed_decisions_cannot_create_experience() -> None:
    fixture = _runtime()
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    forged = replace(decision, policy_version="policy-forged")
    step = _step(decision, suffix="0", closed_at=2)
    context = InteractionContext("run", "task", "episode", 0)

    with pytest.raises(UnknownDecisionError, match="exact unconsumed"):
        fixture.agent.observe(forged, step, context=context)
    assert len(fixture.store) == 0
    assert fixture.agent.lifecycle(context) == ()

    result = fixture.agent.observe(decision, step, context=context)
    with pytest.raises(UnknownDecisionError, match="exact unconsumed"):
        fixture.agent.observe(
            decision,
            _step(decision, suffix="1", closed_at=4),
            context=InteractionContext("run", "task", "episode-2", 0),
        )
    assert len(fixture.store) == 1
    assert fixture.store.get(result.experience.experience_id) is result.experience


def test_duplicate_step_exposes_immutable_lifecycle_without_second_append() -> None:
    fixture = _runtime()
    context = InteractionContext("run", "task", "episode", 0)
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    step = _step(decision, suffix="0", closed_at=2)
    fixture.agent.observe(decision, step, context=context)
    before = fixture.agent.lifecycle(context)

    with pytest.raises(StepAlreadyObservedError) as captured:
        fixture.agent.observe(decision, step, context=context)

    assert captured.value.lifecycle == before
    assert fixture.agent.lifecycle(context) == before
    assert len(fixture.store) == 1


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    ((True, False), (False, True)),
)
def test_terminated_and_truncated_are_distinct(
    terminated: bool,
    truncated: bool,
) -> None:
    fixture = _runtime()
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    result = fixture.agent.observe(
        decision,
        _step(
            decision,
            suffix="terminal",
            closed_at=2,
            terminated=terminated,
            truncated=truncated,
        ),
        context=InteractionContext("run", "task", "episode", 0),
    )

    assert result.experience.terminated is terminated
    assert result.experience.truncated is truncated


def test_environment_step_rejects_ambiguous_or_nonfinite_episode_values() -> None:
    fixture = _runtime()
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    valid = _step(decision, suffix="0", closed_at=2)

    with pytest.raises(ValueError, match="both terminated and truncated"):
        replace(valid, terminated=True, truncated=True)
    for discount in (float("inf"), float("nan"), -0.1):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            replace(valid, discount=discount)


def test_learn_receipt_is_explicit_and_advances_versions_after_ledger_append() -> None:
    fixture = _runtime()
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    interaction = fixture.agent.observe(
        decision,
        _step(decision, suffix="0", closed_at=2),
        context=InteractionContext("run", "task", "episode", 0),
    )

    receipt = fixture.agent.learn((interaction.transition,), at=_time(4))
    snapshot = fixture.agent.snapshot(_time(5))

    assert fixture.ledger.get_update(receipt.receipt_id) is receipt
    assert snapshot.latest_update is receipt
    assert snapshot.configuration_version == "configuration-v2"
    assert snapshot.belief is interaction.transition.belief_update.posterior


def test_invalid_learn_receipt_is_rejected_before_canonical_ledger_append() -> None:
    fixture = _runtime(learner=_Learner(stale_previous=True))
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    interaction = fixture.agent.observe(
        decision,
        _step(decision, suffix="0", closed_at=2),
        context=InteractionContext("run", "task", "episode", 0),
    )

    with pytest.raises(AgentRuntimeError, match="invalid version transition"):
        fixture.agent.learn((interaction.transition,), at=_time(4))

    assert fixture.ledger.update_count == 0
    snapshot = fixture.agent.snapshot(_time(5))
    assert snapshot.configuration_version == "configuration-v1"
    assert snapshot.latest_update is None


def test_resulting_belief_cannot_drop_evidence_newer_than_receipt_batch() -> None:
    fixture = _runtime(learner=_DroppingEvidenceLearner())
    first_decision = fixture.agent.decide(fixture.goal, at=_time(0))
    first = fixture.agent.observe(
        first_decision,
        _step(first_decision, suffix="0", closed_at=2),
        context=InteractionContext("run", "task", "episode", 0),
    )
    second_decision = fixture.agent.decide(fixture.goal, at=_time(3))
    second = fixture.agent.observe(
        second_decision,
        _step(second_decision, suffix="1", closed_at=5),
        context=InteractionContext("run", "task", "episode", 1),
    )

    with pytest.raises(AgentRuntimeError, match="invalid version transition"):
        fixture.agent.learn((first.transition,), at=_time(7))

    assert fixture.ledger.update_count == 0
    assert fixture.agent.snapshot(_time(8)).belief is (second.transition.belief_update.posterior)


def test_updater_failure_keeps_canonical_experience_and_records_failed_stage() -> None:
    fixture = _runtime(
        updater_failure="api_key=super-secret updater unavailable",
    )
    context = InteractionContext("run", "task", "episode", 0)
    decision = fixture.agent.decide(fixture.goal, at=_time(0))
    step = _step(decision, suffix="0", closed_at=2)

    with pytest.raises(LifecycleFailureError) as captured:
        fixture.agent.observe(decision, step, context=context)

    history = fixture.agent.lifecycle(context)
    assert [record.stage for record in history] == [
        LifecycleStage.EXPERIENCE_STORED,
        LifecycleStage.FAILED,
    ]
    assert captured.value.record is history[-1]
    assert history[-1].attempted_stage is LifecycleStage.TRANSITION_STORED
    assert history[-1].failure_type == "ValueError"
    assert "super-secret" not in history[-1].failure_detail
    assert "<redacted>" in history[-1].failure_detail
    assert len(fixture.store) == 1
    assert fixture.ledger.transition_count == 0
    assert fixture.replay.calls == 0

    with pytest.raises(
        StepAlreadyObservedError,
        match="automatic continuation is not implemented",
    ):
        fixture.agent.observe(decision, step, context=context)
    assert len(fixture.store) == 1


def test_replay_failure_occurs_only_after_transition_and_state_are_durable() -> None:
    fixture = _runtime(replay_failure=True)
    context = InteractionContext("run", "task", "episode", 0)
    decision = fixture.agent.decide(fixture.goal, at=_time(0))

    with pytest.raises(LifecycleFailureError):
        fixture.agent.observe(
            decision,
            _step(decision, suffix="0", closed_at=2),
            context=context,
        )

    history = fixture.agent.lifecycle(context)
    assert [record.stage for record in history] == [
        LifecycleStage.EXPERIENCE_STORED,
        LifecycleStage.TRANSITION_STORED,
        LifecycleStage.STATE_APPLIED,
        LifecycleStage.FAILED,
    ]
    assert history[-1].attempted_stage is LifecycleStage.REPLAY_INDEXED
    assert len(fixture.store) == 1
    assert fixture.ledger.transition_count == 1
    transition = fixture.ledger.transitions("agent", _time(3))[0]
    assert fixture.agent.snapshot(_time(3)).belief is (transition.belief_update.posterior)
    assert fixture.replay.calls == 1
