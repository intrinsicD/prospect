from __future__ import annotations

from dataclasses import dataclass

import pytest

from prospect.decision import (
    CounterIdentitySource,
    MaxValuePolicy,
    NoAdmissibleActionError,
)
from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    CandidateAssessment,
    Distribution,
    EpistemicTarget,
    Goal,
    InformationSet,
    InformationValue,
    Prediction,
    ResourceLedger,
    TimePoint,
    Utility,
)


def _time(tick: int) -> TimePoint:
    return TimePoint(tick=tick)


@dataclass(frozen=True, slots=True)
class _DecisionFixture:
    snapshot: AgentSnapshot
    goal: Goal


def _fixture() -> _DecisionFixture:
    target = EpistemicTarget(
        target_id="target",
        description="binary action outcome",
        target_kind="future_outcome",
    )
    information = InformationSet(
        information_set_id="information-0",
        agent_id="agent",
        as_of=_time(0),
        memory_version="memory-v1",
    )
    belief = Belief(
        belief_id="belief-0",
        agent_id="agent",
        target=target,
        information_set=information,
        distribution=Distribution(
            distribution_id="belief-distribution-0",
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
    snapshot = AgentSnapshot(
        snapshot_id="snapshot-0",
        agent_id="agent",
        captured_at=_time(0),
        belief=belief,
        configuration_version="configuration-v1",
        memory_version="memory-v1",
        knowledge_version="knowledge-v1",
        model_version="model-v1",
        representation_version="representation-v1",
        policy_version="policy-v1",
        resources=ResourceLedger(
            ledger_id="resources-0",
            started_at=_time(0),
            completed_at=_time(0),
        ),
    )
    goal = Goal(
        goal_id="goal-0",
        task_id="task-0",
        target=target,
        description="maximize the binary outcome",
        issued_at=_time(0),
        preference_version="preference-v1",
    )
    return _DecisionFixture(snapshot=snapshot, goal=goal)


def _assessment(
    fixture: _DecisionFixture,
    *,
    action_id: str,
    total_value: float,
    admissible: bool = True,
    reasons: tuple[str, ...] = (),
) -> CandidateAssessment:
    action = Action(
        action_id=action_id,
        action_kind="choose",
        parameters={"choice": action_id},
    )
    prediction = Prediction(
        prediction_id=f"prediction-{action_id}",
        prior_belief=fixture.snapshot.belief,
        action=action,
        target=fixture.goal.target,
        distribution=Distribution(
            distribution_id=f"distribution-{action_id}",
            family="categorical",
            support="binary",
            parameters=(0.5, 0.5),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        issued_at=_time(0),
        horizon_end=_time(1),
        model_version="model-v1",
        representation_version="representation-v1",
        calibration_version="uncalibrated",
    )
    utility = Utility(
        utility_id=f"utility-{action_id}",
        goal_id=fixture.goal.goal_id,
        prediction_id=prediction.prediction_id,
        expected_value=total_value,
        unit="decision_value",
        evaluator_version="utility-v1",
        assessed_at=_time(0),
    )
    information_value = InformationValue(
        information_value_id=f"information-value-{action_id}",
        prior_belief_id=fixture.snapshot.belief.belief_id,
        action_id=action_id,
        target_id=fixture.goal.target.target_id,
        expected_reduction=0.0,
        expected_cost=0.0,
        unit="decision_value",
        evaluator_version="information-v1",
        assessed_at=_time(0),
    )
    return CandidateAssessment(
        assessment_id=f"assessment-{action_id}",
        action=action,
        prediction=prediction,
        utility=utility,
        information_value=information_value,
        expected_action_cost=0.0,
        expected_risk=0.0,
        admissible=admissible,
        constraint_reasons=reasons,
        constraint_penalty=0.0,
        total_value=total_value,
        unit="decision_value",
        evaluator_version="candidate-v1",
        assessed_at=_time(0),
    )


@dataclass(frozen=True, slots=True)
class _StaticAssessor:
    alternatives: tuple[CandidateAssessment, ...]

    def assess(
        self,
        snapshot: AgentSnapshot,
        goal: Goal,
    ) -> tuple[CandidateAssessment, ...]:
        del snapshot, goal
        return self.alternatives


def test_max_value_selection_filters_hard_inadmissibility() -> None:
    fixture = _fixture()
    low = _assessment(fixture, action_id="low", total_value=0.2)
    high = _assessment(fixture, action_id="high", total_value=0.8)
    prohibited = _assessment(
        fixture,
        action_id="prohibited",
        total_value=100.0,
        admissible=False,
        reasons=("violates safety envelope",),
    )
    policy = MaxValuePolicy(
        agent_id="agent",
        policy_version="policy-v1",
        assessor=_StaticAssessor((low, prohibited, high)),
        identities=CounterIdentitySource("decision-test"),
    )

    decision = policy.decide(fixture.snapshot, fixture.goal)

    assert decision.alternatives == (low, prohibited, high)
    assert decision.selected_assessment is high
    assert decision.intended_action.action is high.action


def test_equal_value_tie_is_deterministic_by_action_id() -> None:
    fixture = _fixture()
    action_z = _assessment(fixture, action_id="z-action", total_value=0.5)
    action_a = _assessment(fixture, action_id="a-action", total_value=0.5)
    identities = CounterIdentitySource("tie")
    policy = MaxValuePolicy(
        agent_id="agent",
        policy_version="policy-v1",
        assessor=_StaticAssessor((action_z, action_a)),
        identities=identities,
    )

    decision = policy.decide(fixture.snapshot, fixture.goal)

    assert decision.selected_assessment is action_a
    assert decision.intended_action.intention_id == "tie:intention:0"
    assert decision.decision_id == "tie:decision:1"


def test_all_hard_inadmissible_candidates_fail_before_allocating_ids() -> None:
    fixture = _fixture()
    identities = CounterIdentitySource("blocked")
    policy = MaxValuePolicy(
        agent_id="agent",
        policy_version="policy-v1",
        assessor=_StaticAssessor(
            (
                _assessment(
                    fixture,
                    action_id="unsafe-b",
                    total_value=2.0,
                    admissible=False,
                    reasons=("unsafe-b",),
                ),
                _assessment(
                    fixture,
                    action_id="unsafe-a",
                    total_value=1.0,
                    admissible=False,
                    reasons=("unsafe-a",),
                ),
            )
        ),
        identities=identities,
    )

    with pytest.raises(
        NoAdmissibleActionError,
        match=r"unsafe-a, unsafe-b",
    ):
        policy.decide(fixture.snapshot, fixture.goal)
    assert identities.next_counter == 0


def test_counter_identity_checkpoint_roundtrip_continues_exact_sequence() -> None:
    original = CounterIdentitySource("run")
    assert original.next("intention") == "run:intention:0"
    assert original.next("decision") == "run:decision:1"
    payload = original.checkpoint_bytes()

    restarted = CounterIdentitySource("run", next_counter=99)
    restarted.restore_bytes(payload)

    assert restarted.next_counter == original.next_counter == 2
    assert restarted.next("experience") == original.next("experience")
    assert restarted.checkpoint_bytes() == original.checkpoint_bytes()
