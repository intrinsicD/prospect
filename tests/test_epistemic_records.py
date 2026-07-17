from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    CandidateAssessment,
    DecisionRecord,
    Distribution,
    DomainInvariantError,
    EpistemicEffect,
    EpistemicEffectKind,
    EpistemicTarget,
    EpistemicTransition,
    EvaluationMetric,
    EvaluationRecord,
    Evidence,
    EvidenceLineage,
    EvidenceOrigin,
    ExecutedAction,
    ExecutionStatus,
    ExperienceEvent,
    ExperienceKind,
    ExperienceStore,
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
    ResourceUse,
    TimePoint,
    TrustLevel,
    UncertaintyEstimate,
    UncertaintyKind,
    UpdateReceipt,
    UpdateStatus,
    Utility,
)


def _time(tick: int) -> TimePoint:
    return TimePoint(tick=tick)


def _evidence(
    evidence_id: str,
    *,
    occurred: int,
    available: int,
    payload: object,
    origin: EvidenceOrigin = EvidenceOrigin.OBSERVED,
    producer_version: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        payload=payload,
        occurred_at=_time(occurred),
        available_at=_time(available),
        lineage=EvidenceLineage(
            evidence_id=evidence_id,
            origin=origin,
            provenance=Provenance(
                source_id="environment",
                trust=TrustLevel.HIGH,
                source_kind="sensor",
            ),
            producer_version=producer_version,
        ),
    )


@dataclass(frozen=True, slots=True)
class _Graph:
    prior_observation: Observation
    observation: Observation
    prior_information: InformationSet
    posterior_information: InformationSet
    prior: Belief
    posterior: Belief
    action: Action
    intention: IntendedAction
    prediction: Prediction
    assessment: CandidateAssessment
    alternative_assessment: CandidateAssessment
    decision: DecisionRecord
    execution: ExecutedAction
    outcome: Outcome
    experience: ExperienceEvent
    belief_update: BeliefUpdate
    score: ProperScore
    effect: EpistemicEffect
    transition: EpistemicTransition
    receipt: UpdateReceipt
    snapshot: AgentSnapshot


def _graph() -> _Graph:
    target = EpistemicTarget(
        target_id="pendulum-angle",
        description="pendulum angle after the selected action",
        target_kind="future_outcome",
    )
    prior_observation = Observation(
        observation_id="obs-0",
        agent_id="agent-1",
        modality="state",
        evidence=_evidence("obs-0", occurred=0, available=0, payload=(0.0, 0.0)),
    )
    prior_information = InformationSet(
        information_set_id="info-0",
        agent_id="agent-1",
        as_of=_time(0),
        observations=(prior_observation,),
        memory_version="memory-v1",
    )
    prior_distribution = Distribution(
        distribution_id="belief-dist-0",
        family="categorical",
        support="angle-cells",
        parameters=(0.25, 0.75),
        representation_version="representation-v1",
        event_shape=(2,),
    )
    prior = Belief(
        belief_id="belief-0",
        agent_id="agent-1",
        target=target,
        information_set=prior_information,
        distribution=prior_distribution,
        formed_at=_time(1),
        model_version="model-v1",
        representation_version="representation-v1",
    )
    action = Action(action_id="action-1", action_kind="torque", parameters=(0.5,))
    intention = IntendedAction(
        intention_id="intention-1",
        agent_id="agent-1",
        action=action,
        intended_at=_time(2),
    )
    uncertainty = UncertaintyEstimate(
        estimate_id="uncertainty-1",
        kind=UncertaintyKind.MODEL,
        measure="ensemble_variance",
        value=0.1,
        unit="variance",
        target_id=target.target_id,
        estimator_version="uncertainty-v1",
        assessed_at=_time(1),
        calibration_version="calibration-v1",
    )
    prediction = Prediction(
        prediction_id="prediction-1",
        prior_belief=prior,
        action=action,
        target=target,
        distribution=Distribution(
            distribution_id="prediction-dist-1",
            family="categorical",
            support="angle-cells",
            parameters=(0.7, 0.3),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        issued_at=_time(1),
        horizon_end=_time(4),
        model_version="model-v1",
        representation_version="representation-v1",
        calibration_version="calibration-v1",
        uncertainties=(uncertainty,),
    )
    goal = Goal(
        goal_id="goal-1",
        task_id="task-1",
        target=target,
        description="move the pendulum toward the target angle",
        issued_at=_time(1),
        preference_version="preference-v1",
        deadline=_time(8),
    )
    utility = Utility(
        utility_id="utility-1",
        goal_id=goal.goal_id,
        prediction_id=prediction.prediction_id,
        expected_value=0.8,
        unit="decision_value",
        evaluator_version="utility-v1",
        assessed_at=_time(2),
    )
    information_value = InformationValue(
        information_value_id="information-value-1",
        prior_belief_id=prior.belief_id,
        action_id=action.action_id,
        target_id=target.target_id,
        expected_reduction=0.2,
        expected_cost=0.01,
        unit="decision_value",
        evaluator_version="information-v1",
        assessed_at=_time(2),
    )
    assessment = CandidateAssessment(
        assessment_id="assessment-1",
        action=action,
        prediction=prediction,
        utility=utility,
        information_value=information_value,
        expected_action_cost=0.05,
        expected_risk=0.02,
        admissible=True,
        constraint_reasons=(),
        constraint_penalty=0.0,
        total_value=0.92,
        unit="decision_value",
        evaluator_version="candidate-v1",
        assessed_at=_time(2),
    )
    alternative_action = Action(
        action_id="action-2",
        action_kind="torque",
        parameters=(-0.5,),
    )
    alternative_prediction = replace(
        prediction,
        prediction_id="prediction-2",
        action=alternative_action,
        distribution=replace(
            prediction.distribution,
            distribution_id="prediction-dist-2",
            parameters=(0.4, 0.6),
        ),
        uncertainties=(),
    )
    alternative_utility = Utility(
        utility_id="utility-2",
        goal_id=goal.goal_id,
        prediction_id=alternative_prediction.prediction_id,
        expected_value=0.6,
        unit="decision_value",
        evaluator_version="utility-v1",
        assessed_at=_time(2),
    )
    alternative_information_value = InformationValue(
        information_value_id="information-value-2",
        prior_belief_id=prior.belief_id,
        action_id=alternative_action.action_id,
        target_id=target.target_id,
        expected_reduction=0.3,
        expected_cost=0.02,
        unit="decision_value",
        evaluator_version="information-v1",
        assessed_at=_time(2),
    )
    alternative_assessment = CandidateAssessment(
        assessment_id="assessment-2",
        action=alternative_action,
        prediction=alternative_prediction,
        utility=alternative_utility,
        information_value=alternative_information_value,
        expected_action_cost=0.03,
        expected_risk=0.1,
        admissible=True,
        constraint_reasons=(),
        constraint_penalty=0.1,
        total_value=0.65,
        unit="decision_value",
        evaluator_version="candidate-v1",
        assessed_at=_time(2),
    )
    decision = DecisionRecord(
        decision_id="decision-1",
        agent_id="agent-1",
        belief=prior,
        goal=goal,
        intended_action=intention,
        alternatives=(assessment, alternative_assessment),
        selected_assessment=assessment,
        policy_version="policy-v1",
        decided_at=_time(2),
    )
    execution = ExecutedAction(
        execution_id="execution-1",
        intention=intention,
        status=ExecutionStatus.SUCCEEDED,
        started_at=_time(3),
        ended_at=_time(4),
        realized_action=action,
    )
    observation = Observation(
        observation_id="obs-1",
        agent_id="agent-1",
        modality="state",
        evidence=_evidence("obs-1", occurred=4, available=5, payload=(0.4, 0.1)),
    )
    outcome = Outcome(
        outcome_id="outcome-1",
        execution_id=execution.execution_id,
        evidence=_evidence("outcome-evidence-1", occurred=4, available=5, payload={"cost": 0.1}),
    )
    experience = ExperienceEvent(
        experience_id="experience-1",
        agent_id="agent-1",
        run_id="run-1",
        task_id="task-1",
        episode_id="episode-1",
        step_index=0,
        kind=ExperienceKind.INTERACTION,
        observation=observation,
        outcome=outcome,
        terminated=False,
        truncated=False,
        discount=0.99,
        behavior_policy_version="policy-v1",
        decision=decision,
        execution=execution,
        closed_at=_time(5),
    )
    posterior_information = InformationSet(
        information_set_id="info-1",
        agent_id="agent-1",
        as_of=_time(5),
        observations=(prior_observation, observation),
        memory_version="memory-v2",
    )
    posterior = Belief(
        belief_id="belief-1",
        agent_id="agent-1",
        target=target,
        information_set=posterior_information,
        distribution=Distribution(
            distribution_id="belief-dist-1",
            family="categorical",
            support="angle-cells",
            parameters=(0.9, 0.1),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        formed_at=_time(6),
        model_version="model-v1",
        representation_version="representation-v1",
    )
    belief_update = BeliefUpdate(
        update_id="belief-update-1",
        prior=prior,
        experience=experience,
        posterior=posterior,
        updater_version="updater-v1",
        updated_at=_time(6),
    )
    score = ProperScore(
        score_id="score-1",
        prediction_id=prediction.prediction_id,
        realized_evidence_id=outcome.evidence.evidence_id,
        rule="log_score",
        value=0.4,
        unit="nats",
        scorer_version="scorer-v1",
        scored_at=_time(6),
    )
    effect = EpistemicEffect(
        effect_id="effect-1",
        belief_update_id=belief_update.update_id,
        target_id=target.target_id,
        kind=EpistemicEffectKind.PREDICTIVE_RISK_CHANGE,
        measure="expected_log_score",
        before=1.0,
        after=0.4,
        improvement=0.6,
        higher_is_better=False,
        evaluator_version="effect-v1",
        evaluated_at=_time(7),
        externally_calibrated=True,
    )
    transition = EpistemicTransition(
        transition_id="transition-1",
        experience=experience,
        belief_update=belief_update,
        proper_scores=(score,),
        effects=(effect,),
        created_at=_time(7),
    )
    receipt = UpdateReceipt(
        receipt_id="receipt-1",
        agent_id="agent-1",
        transitions=(transition,),
        learner_version="learner-v1",
        status=UpdateStatus.APPLIED,
        previous_configuration_version="configuration-v1",
        new_configuration_version="configuration-v2",
        previous_model_version="model-v1",
        new_model_version="model-v1",
        previous_representation_version="representation-v1",
        new_representation_version="representation-v1",
        previous_policy_version="policy-v1",
        new_policy_version="policy-v1",
        started_at=_time(8),
        completed_at=_time(9),
        metrics=(("loss", 0.2),),
    )
    resources = ResourceLedger(
        ledger_id="resources-through-snapshot",
        started_at=_time(0),
        completed_at=_time(9),
        uses=(ResourceUse(resource="environment_steps", amount=1.0, unit="step"),),
    )
    snapshot = AgentSnapshot(
        snapshot_id="snapshot-1",
        agent_id="agent-1",
        captured_at=_time(10),
        belief=posterior,
        configuration_version="configuration-v2",
        memory_version="memory-v2",
        knowledge_version="knowledge-v1",
        model_version="model-v1",
        representation_version="representation-v1",
        policy_version="policy-v1",
        resources=resources,
        latest_update=receipt,
    )
    return _Graph(
        prior_observation=prior_observation,
        observation=observation,
        prior_information=prior_information,
        posterior_information=posterior_information,
        prior=prior,
        posterior=posterior,
        action=action,
        intention=intention,
        prediction=prediction,
        assessment=assessment,
        alternative_assessment=alternative_assessment,
        decision=decision,
        execution=execution,
        outcome=outcome,
        experience=experience,
        belief_update=belief_update,
        score=score,
        effect=effect,
        transition=transition,
        receipt=receipt,
        snapshot=snapshot,
    )


def test_complete_epistemic_transition_is_linked_and_has_no_scalar_epistemic_alias() -> None:
    graph = _graph()

    assert graph.transition.belief_update.prior is graph.prior
    assert graph.transition.belief_update.posterior is graph.posterior
    assert graph.transition.experience.decision is graph.decision
    assert graph.snapshot.latest_update is graph.receipt
    assert not hasattr(graph.prediction, "epistemic")
    assert not hasattr(graph.transition, "epistemic")
    assert not hasattr(graph.outcome, "terminal")


def test_records_are_frozen_and_slotted() -> None:
    graph = _graph()

    assert not hasattr(graph.transition, "__dict__")
    with pytest.raises(FrozenInstanceError):
        graph.transition.transition_id = "mutated"  # type: ignore[misc]


def test_information_set_rejects_future_observation() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="cutoff.*temporal order"):
        InformationSet(
            information_set_id="future-leaking-info",
            agent_id="agent-1",
            as_of=_time(4),
            observations=(graph.observation,),
            memory_version="memory-v1",
        )


def test_imagined_evidence_cannot_be_observation_or_outcome() -> None:
    imagined = _evidence(
        "imagined-1",
        occurred=2,
        available=2,
        payload=(1.0,),
        origin=EvidenceOrigin.IMAGINED,
        producer_version="model-v1",
    )

    with pytest.raises(DomainInvariantError, match="imagined evidence"):
        Observation(
            observation_id="imagined-1",
            agent_id="agent-1",
            modality="state",
            evidence=imagined,
        )
    with pytest.raises(DomainInvariantError, match="imagined evidence"):
        Outcome(outcome_id="imagined-outcome", evidence=imagined)


def test_prediction_rejects_model_and_representation_version_mismatch() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="model versions"):
        replace(graph.prediction, prediction_id="bad-model", model_version="model-v2")
    with pytest.raises(DomainInvariantError, match="representation versions"):
        replace(
            graph.prediction,
            prediction_id="bad-representation",
            representation_version="representation-v2",
        )


def test_prediction_uncertainty_is_explicit_and_may_be_empty() -> None:
    graph = _graph()

    uncalibrated = replace(
        graph.prediction,
        prediction_id="uncalibrated-prediction",
        calibration_version="uncalibrated",
        uncertainties=(),
    )
    assert uncalibrated.uncertainties == ()

    with pytest.raises(DomainInvariantError, match="calibration_version"):
        replace(
            graph.prediction,
            prediction_id="missing-calibration-identity",
            calibration_version=" ",
            uncertainties=(),
        )


def test_prediction_rejects_ambiguous_or_unlinked_uncertainty() -> None:
    graph = _graph()
    estimate = graph.prediction.uncertainties[0]
    duplicate_measure = replace(estimate, estimate_id="uncertainty-2")

    with pytest.raises(DomainInvariantError, match="duplicate uncertainty"):
        replace(
            graph.prediction,
            prediction_id="duplicate-uncertainty",
            uncertainties=(estimate, duplicate_measure),
        )
    with pytest.raises(DomainInvariantError, match="different target"):
        replace(
            graph.prediction,
            prediction_id="wrong-uncertainty-target",
            uncertainties=(
                replace(
                    estimate,
                    estimate_id="wrong-target",
                    target_id="another-target",
                ),
            ),
        )
    with pytest.raises(DomainInvariantError, match="calibration versions"):
        replace(
            graph.prediction,
            prediction_id="wrong-uncertainty-calibration",
            uncertainties=(
                replace(
                    estimate,
                    estimate_id="wrong-calibration",
                    calibration_version="calibration-v2",
                ),
            ),
        )
    with pytest.raises(DomainInvariantError, match="prediction issuance.*temporal order"):
        replace(
            graph.prediction,
            prediction_id="future-uncertainty",
            uncertainties=(
                replace(
                    estimate,
                    estimate_id="future-estimate",
                    assessed_at=_time(2),
                ),
            ),
        )


def test_decision_rejects_prediction_action_mismatch() -> None:
    graph = _graph()
    other_action = Action(action_id="action-2", action_kind="torque", parameters=(-0.5,))

    with pytest.raises(DomainInvariantError, match="prediction action ids differ"):
        replace(
            graph.assessment,
            assessment_id="mismatched-candidate",
            action=other_action,
        )
    with pytest.raises(DomainInvariantError, match="intended action ids differ"):
        replace(
            graph.decision,
            decision_id="bad-decision",
            selected_assessment=graph.alternative_assessment,
        )


def test_candidate_total_and_hard_constraints_are_enforced() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="assessment unit"):
        replace(
            graph.assessment,
            assessment_id="mixed-units",
            utility=replace(graph.assessment.utility, unit="different-unit"),
        )
    with pytest.raises(DomainInvariantError, match="different target"):
        replace(
            graph.assessment,
            assessment_id="wrong-information-target",
            information_value=replace(
                graph.assessment.information_value,
                target_id="another-target",
            ),
        )
    with pytest.raises(DomainInvariantError, match="utility assessment.*temporal order"):
        replace(
            graph.assessment,
            assessment_id="premature-utility",
            utility=replace(graph.assessment.utility, assessed_at=_time(0)),
        )
    with pytest.raises(DomainInvariantError, match="total_value disagrees"):
        replace(
            graph.assessment,
            assessment_id="wrong-total",
            total_value=100.0,
        )
    with pytest.raises(DomainInvariantError, match="requires a hard-constraint reason"):
        replace(
            graph.assessment,
            assessment_id="unexplained-prohibition",
            admissible=False,
        )
    with pytest.raises(DomainInvariantError, match="cannot have hard-constraint reasons"):
        replace(
            graph.assessment,
            assessment_id="contradictory-admissibility",
            constraint_reasons=("unsafe torque",),
        )
    with pytest.raises(DomainInvariantError, match="constraint reason must be nonempty"):
        replace(
            graph.assessment,
            assessment_id="blank-constraint-reason",
            admissible=False,
            constraint_reasons=(" ",),
        )
    with pytest.raises(DomainInvariantError, match="constraint reasons must be unique"):
        replace(
            graph.assessment,
            assessment_id="duplicate-constraint-reasons",
            admissible=False,
            constraint_reasons=("unsafe torque", "unsafe torque"),
        )

    inadmissible = replace(
        graph.assessment,
        assessment_id="inadmissible-candidate",
        admissible=False,
        constraint_reasons=("unsafe torque",),
    )
    assert not inadmissible.admissible


def test_decision_requires_unique_linked_candidates_and_admissible_selection() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="at least one candidate"):
        replace(
            graph.decision,
            decision_id="empty-candidate-set",
            alternatives=(),
        )
    duplicate_action = replace(
        graph.assessment,
        assessment_id="duplicate-action-assessment",
    )
    with pytest.raises(DomainInvariantError, match="candidate action ids"):
        replace(
            graph.decision,
            decision_id="duplicate-action",
            alternatives=(graph.assessment, duplicate_action),
        )
    with pytest.raises(DomainInvariantError, match="not linked to the alternatives"):
        replace(
            graph.decision,
            decision_id="missing-selected-candidate",
            alternatives=(graph.alternative_assessment,),
        )
    inadmissible = replace(
        graph.assessment,
        admissible=False,
        constraint_reasons=("violates actuator envelope",),
    )
    with pytest.raises(DomainInvariantError, match="selected assessment must be admissible"):
        replace(
            graph.decision,
            decision_id="selected-hard-prohibition",
            alternatives=(inadmissible, graph.alternative_assessment),
            selected_assessment=inadmissible,
        )


def test_experience_rejects_execution_and_outcome_id_mismatch() -> None:
    graph = _graph()
    other_execution = replace(graph.execution, execution_id="execution-2")

    with pytest.raises(DomainInvariantError, match="outcome does not match"):
        replace(
            graph.experience,
            experience_id="bad-experience",
            execution=other_execution,
        )


def test_experience_enforces_e0_episode_and_behavior_links() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="step_index must be nonnegative"):
        replace(
            graph.experience,
            experience_id="negative-step",
            step_index=-1,
        )
    with pytest.raises(DomainInvariantError, match="both terminated and truncated"):
        replace(
            graph.experience,
            experience_id="ambiguous-terminal",
            terminated=True,
            truncated=True,
        )
    for discount in (-0.1, float("inf"), float("nan")):
        with pytest.raises(DomainInvariantError, match="discount"):
            replace(
                graph.experience,
                experience_id=f"bad-discount-{discount}",
                discount=discount,
            )
    with pytest.raises(DomainInvariantError, match="task does not match"):
        replace(
            graph.experience,
            experience_id="wrong-task",
            task_id="task-2",
        )
    with pytest.raises(DomainInvariantError, match="behavior policy does not match"):
        replace(
            graph.experience,
            experience_id="wrong-behavior-policy",
            behavior_policy_version="policy-v2",
        )


def test_belief_update_rejects_missing_new_evidence_and_version_change() -> None:
    graph = _graph()
    incomplete_information = replace(
        graph.posterior_information,
        information_set_id="incomplete-info",
        observations=(graph.prior_observation,),
    )
    incomplete_posterior = replace(
        graph.posterior,
        belief_id="incomplete-posterior",
        information_set=incomplete_information,
    )

    with pytest.raises(DomainInvariantError, match="omits the new observation"):
        replace(
            graph.belief_update,
            update_id="bad-update-evidence",
            posterior=incomplete_posterior,
        )

    changed_model_posterior = replace(
        graph.posterior,
        belief_id="changed-model-posterior",
        model_version="model-v2",
    )
    with pytest.raises(DomainInvariantError, match="changed model version"):
        replace(
            graph.belief_update,
            update_id="bad-update-version",
            posterior=changed_model_posterior,
        )


def test_effect_direction_and_transition_links_are_enforced() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="before/after direction"):
        replace(graph.effect, effect_id="bad-effect", improvement=-0.6)

    wrong_score = replace(
        graph.score,
        score_id="wrong-score",
        prediction_id="unrelated-prediction",
    )
    with pytest.raises(DomainInvariantError, match="different prediction"):
        replace(
            graph.transition,
            transition_id="bad-transition",
            proper_scores=(wrong_score,),
        )


def test_update_receipt_enforces_status_and_causal_version_rules() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="must change at least one version"):
        replace(
            graph.receipt,
            receipt_id="no-op-applied",
            new_configuration_version=graph.receipt.previous_configuration_version,
        )
    with pytest.raises(DomainInvariantError, match="rejected update cannot change"):
        replace(graph.receipt, receipt_id="changed-rejection", status=UpdateStatus.REJECTED)
    with pytest.raises(DomainInvariantError, match="only an applied update"):
        replace(
            graph.receipt,
            receipt_id="rejected-with-result",
            status=UpdateStatus.REJECTED,
            new_configuration_version=graph.receipt.previous_configuration_version,
            resulting_belief=replace(
                graph.posterior,
                belief_id="belief-from-rejected-update",
                formed_at=_time(9),
            ),
        )
    with pytest.raises(DomainInvariantError, match="learning start.*temporal order"):
        replace(
            graph.receipt,
            receipt_id="future-consuming-update",
            started_at=_time(6),
            completed_at=_time(9),
        )


def test_model_update_links_a_rebased_resulting_belief_and_snapshot() -> None:
    graph = _graph()
    rebased = replace(
        graph.posterior,
        belief_id="belief-2",
        distribution=replace(
            graph.posterior.distribution,
            distribution_id="belief-dist-2",
            representation_version="representation-v2",
        ),
        formed_at=_time(9),
        model_version="model-v2",
        representation_version="representation-v2",
    )
    receipt = replace(
        graph.receipt,
        receipt_id="receipt-model-v2",
        new_model_version="model-v2",
        new_representation_version="representation-v2",
        resulting_belief=rebased,
    )
    snapshot = replace(
        graph.snapshot,
        snapshot_id="snapshot-model-v2",
        belief=rebased,
        model_version="model-v2",
        representation_version="representation-v2",
        latest_update=receipt,
    )

    assert receipt.resulting_belief is rebased
    assert snapshot.belief.belief_id == rebased.belief_id


def test_model_update_requires_a_causally_valid_resulting_belief() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="requires a resulting_belief"):
        replace(
            graph.receipt,
            receipt_id="model-change-without-belief",
            new_model_version="model-v2",
        )

    rebased = replace(
        graph.posterior,
        belief_id="belief-2",
        distribution=replace(
            graph.posterior.distribution,
            distribution_id="belief-dist-2",
            representation_version="representation-v2",
        ),
        formed_at=_time(9),
        model_version="model-v2",
        representation_version="representation-v2",
    )
    with pytest.raises(DomainInvariantError, match="new model version"):
        replace(
            graph.receipt,
            receipt_id="wrong-result-model",
            new_model_version="model-v2",
            new_representation_version="representation-v2",
            resulting_belief=replace(
                rebased,
                belief_id="belief-wrong-model",
                model_version="model-v3",
            ),
        )
    with pytest.raises(DomainInvariantError, match="changed epistemic target"):
        replace(
            graph.receipt,
            receipt_id="wrong-result-target",
            new_model_version="model-v2",
            new_representation_version="representation-v2",
            resulting_belief=replace(
                rebased,
                belief_id="belief-wrong-target",
                target=EpistemicTarget(
                    target_id="another-target",
                    description="an unrelated future outcome",
                ),
            ),
        )
    other_agent_information = InformationSet(
        information_set_id="other-agent-information",
        agent_id="agent-2",
        as_of=_time(5),
        memory_version="memory-v2",
    )
    with pytest.raises(DomainInvariantError, match="another agent"):
        replace(
            graph.receipt,
            receipt_id="wrong-result-agent",
            new_model_version="model-v2",
            new_representation_version="representation-v2",
            resulting_belief=replace(
                rebased,
                belief_id="belief-wrong-agent",
                agent_id="agent-2",
                information_set=other_agent_information,
            ),
        )

    information_loss = replace(
        graph.posterior_information,
        information_set_id="information-loss",
        observations=(graph.observation,),
    )
    with pytest.raises(DomainInvariantError, match="lost source-belief information"):
        replace(
            graph.receipt,
            receipt_id="information-losing-result",
            new_model_version="model-v2",
            new_representation_version="representation-v2",
            resulting_belief=replace(
                rebased,
                belief_id="belief-information-loss",
                information_set=information_loss,
            ),
        )
    with pytest.raises(DomainInvariantError, match="formation.*temporal order"):
        replace(
            graph.receipt,
            receipt_id="premature-result",
            new_model_version="model-v2",
            new_representation_version="representation-v2",
            resulting_belief=replace(
                rebased,
                belief_id="belief-before-update-completed",
                formed_at=_time(8),
            ),
        )


def test_snapshot_must_use_latest_updates_resulting_belief_identity() -> None:
    graph = _graph()
    rebased = replace(
        graph.posterior,
        belief_id="belief-2",
        distribution=replace(
            graph.posterior.distribution,
            distribution_id="belief-dist-2",
            representation_version="representation-v2",
        ),
        formed_at=_time(9),
        model_version="model-v2",
        representation_version="representation-v2",
    )
    receipt = replace(
        graph.receipt,
        receipt_id="receipt-model-v2",
        new_model_version="model-v2",
        new_representation_version="representation-v2",
        resulting_belief=rebased,
    )
    unrelated = replace(rebased, belief_id="unrelated-belief")

    with pytest.raises(DomainInvariantError, match="resulting belief"):
        replace(
            graph.snapshot,
            snapshot_id="snapshot-with-unrelated-belief",
            belief=unrelated,
            model_version="model-v2",
            representation_version="representation-v2",
            latest_update=receipt,
        )


def test_snapshot_rejects_version_mismatch() -> None:
    graph = _graph()

    with pytest.raises(DomainInvariantError, match="snapshot and belief model versions"):
        replace(graph.snapshot, snapshot_id="bad-snapshot", model_version="model-v2")


def test_evaluation_enforces_resource_window_and_no_learning_guard() -> None:
    graph = _graph()
    resources = ResourceLedger(
        ledger_id="evaluation-resources",
        started_at=_time(11),
        completed_at=_time(12),
        uses=(ResourceUse(resource="environment_steps", amount=10.0, unit="step"),),
    )
    valid = EvaluationRecord(
        evaluation_id="evaluation-1",
        agent_id="agent-1",
        task_id="task-1",
        evaluator_version="evaluator-v1",
        snapshot=graph.snapshot,
        started_at=_time(11),
        completed_at=_time(12),
        metrics=(EvaluationMetric(name="return", value=1.0, unit="score"),),
        resources=resources,
        transition_ids=(graph.transition.transition_id,),
    )
    assert valid.metrics[0].value == 1.0

    with pytest.raises(DomainInvariantError, match="forbids training updates"):
        replace(
            valid,
            evaluation_id="evaluation-with-hidden-update",
            update_receipts=(graph.receipt,),
        )


def test_experience_store_protocol_is_structural() -> None:
    graph = _graph()

    class MemoryStore:
        def __init__(self) -> None:
            self.events: dict[str, ExperienceEvent] = {}

        def append(self, event: ExperienceEvent) -> None:
            self.events[event.experience_id] = event

        def get(self, experience_id: str) -> ExperienceEvent:
            return self.events[experience_id]

        def history(self, agent_id: str, as_of: TimePoint) -> tuple[ExperienceEvent, ...]:
            return tuple(
                event
                for event in self.events.values()
                if event.agent_id == agent_id and event.closed_at.tick <= as_of.tick
            )

    store = MemoryStore()
    assert isinstance(store, ExperienceStore)
    store.append(graph.experience)
    assert store.get(graph.experience.experience_id) is graph.experience
