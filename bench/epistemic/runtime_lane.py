"""Integration lane through Prospect's authoritative runtime and canonical stores.

The exact lifecycle oracle in :mod:`bench.epistemic.lifecycle` is intentionally
small enough to audit, but it must not become a second production collector.  This
module therefore drives the same finite diagnostic problem through the real
``EpistemicAgent.interact`` and ``EpistemicAgent.learn`` path.  Passing this lane
means the selected prediction, real environment event, belief update, proper score,
transition, and update receipt are the exact canonical objects held by storage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from bench.epistemic.lifecycle import (
    PROBES,
    BinaryRuleTask,
    ProbeAction,
    diagnose_probe,
    posterior_after_signal,
)
from prospect.decision import CounterIdentitySource, MaxValuePolicy
from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    CandidateAssessment,
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
from prospect.epistemics.information import entropy, predictive_distribution
from prospect.epistemics.scoring import categorical_log_score
from prospect.runtime import (
    AgentState,
    EnvironmentStep,
    EpistemicAgent,
    InteractionContext,
    InteractionResult,
)
from prospect.storage import EpistemicLedger, InMemoryExperienceStore

_MODEL_VERSION = "runtime-exact-channel-v1"
_REPRESENTATION_VERSION = "runtime-binary-categorical-v1"
_POLICY_VERSION = "runtime-exact-voi-v1"


@dataclass(frozen=True, slots=True)
class RuntimeIntegrityResult:
    """Inspectable evidence that one complete lifecycle used the real runtime."""

    interaction: InteractionResult
    receipt: UpdateReceipt
    final_snapshot: AgentSnapshot
    store: InMemoryExperienceStore
    ledger: EpistemicLedger
    canonical_experience: bool
    canonical_transition: bool
    canonical_update: bool
    selected_relevant_probe: bool
    learned_configuration: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.canonical_experience,
                self.canonical_transition,
                self.canonical_update,
                self.selected_relevant_probe,
                self.learned_configuration,
                len(self.store) == 1,
                self.ledger.transition_count == 1,
                self.ledger.update_count == 1,
            )
        )


class _ExactCandidateAssessor:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def assess(
        self,
        snapshot: AgentSnapshot,
        goal: Goal,
    ) -> Sequence[CandidateAssessment]:
        prior_rule_one = _belief_rule_one(snapshot.belief)
        alternatives: list[CandidateAssessment] = []
        for probe in PROBES:
            action = Action(
                action_id=self._identities.next(f"action-{probe.action.value}"),
                action_kind=f"probe:{probe.action.value}",
                parameters={
                    "probe_kind": probe.action.value,
                    "accuracy": probe.accuracy,
                    "cost": probe.cost,
                },
            )
            probabilities = predictive_distribution(
                (1.0 - prior_rule_one, prior_rule_one),
                probe.likelihoods,
            )
            prediction = Prediction(
                prediction_id=self._identities.next(f"prediction-{probe.action.value}"),
                prior_belief=snapshot.belief,
                action=action,
                target=goal.target,
                distribution=Distribution(
                    distribution_id=self._identities.next(f"prediction-distribution-{probe.action.value}"),
                    family="categorical",
                    support="signal={0,1}",
                    parameters=probabilities,
                    representation_version=_REPRESENTATION_VERSION,
                    event_shape=(2,),
                ),
                issued_at=snapshot.captured_at,
                horizon_end=TimePoint(snapshot.captured_at.tick + 2),
                model_version=_MODEL_VERSION,
                representation_version=_REPRESENTATION_VERSION,
                calibration_version="runtime-exact-calibration-v1",
            )
            diagnostics = diagnose_probe(prior_rule_one, probe)
            utility = Utility(
                utility_id=self._identities.next(f"utility-{probe.action.value}"),
                goal_id=goal.goal_id,
                prediction_id=prediction.prediction_id,
                expected_value=max(1.0 - prior_rule_one, prior_rule_one, 0.65),
                unit="external_utility",
                evaluator_version="runtime-terminal-value-v1",
                assessed_at=snapshot.captured_at,
            )
            information_value = InformationValue(
                information_value_id=self._identities.next(f"information-value-{probe.action.value}"),
                prior_belief_id=snapshot.belief.belief_id,
                action_id=action.action_id,
                target_id=goal.target.target_id,
                expected_reduction=diagnostics.expected_decision_value,
                expected_cost=probe.cost,
                unit="external_utility",
                evaluator_version="runtime-exact-evsi-v1",
                assessed_at=snapshot.captured_at,
            )
            alternatives.append(
                CandidateAssessment(
                    assessment_id=self._identities.next(f"assessment-{probe.action.value}"),
                    action=action,
                    prediction=prediction,
                    utility=utility,
                    information_value=information_value,
                    expected_action_cost=0.0,
                    expected_risk=0.0,
                    admissible=True,
                    constraint_reasons=(),
                    constraint_penalty=0.0,
                    total_value=(
                        utility.expected_value + information_value.expected_reduction - information_value.expected_cost
                    ),
                    unit="external_utility",
                    evaluator_version="runtime-exact-assessor-v1",
                    assessed_at=snapshot.captured_at,
                )
            )
        return tuple(alternatives)


class _ExactEnvironment:
    def __init__(
        self,
        task: BinaryRuleTask,
        identities: CounterIdentitySource,
    ) -> None:
        self._task = task
        self._identities = identities

    def step(self, intention: IntendedAction) -> EnvironmentStep:
        if intention.action.action_kind != "probe:relevant":
            raise RuntimeError("exact runtime policy failed to select the relevant probe")
        started = TimePoint(intention.intended_at.tick + 1)
        ended = TimePoint(started.tick + 1)
        execution = ExecutedAction(
            execution_id=self._identities.next("execution"),
            intention=intention,
            status=ExecutionStatus.SUCCEEDED,
            started_at=started,
            ended_at=ended,
            realized_action=intention.action,
        )
        signal = self._task.rule
        observation_id = self._identities.next("observation")
        observation = Observation(
            observation_id=observation_id,
            agent_id="runtime-reference-agent",
            modality="diagnostic_signal",
            evidence=Evidence(
                evidence_id=observation_id,
                payload={
                    "task_id": self._task.task_id,
                    "probe_kind": ProbeAction.RELEVANT.value,
                    "signal": signal,
                    "accuracy": 0.8,
                },
                occurred_at=ended,
                available_at=ended,
                lineage=EvidenceLineage(
                    evidence_id=observation_id,
                    origin=EvidenceOrigin.OBSERVED,
                    provenance=Provenance(
                        source_id="runtime-binary-environment",
                        trust=TrustLevel.VERIFIED,
                        source_kind="diagnostic_sensor",
                    ),
                ),
            ),
        )
        outcome_evidence_id = self._identities.next("outcome-evidence")
        outcome = Outcome(
            outcome_id=self._identities.next("outcome"),
            execution_id=execution.execution_id,
            evidence=Evidence(
                evidence_id=outcome_evidence_id,
                payload={"probe_cost": 0.04},
                occurred_at=ended,
                available_at=ended,
                lineage=EvidenceLineage(
                    evidence_id=outcome_evidence_id,
                    origin=EvidenceOrigin.OBSERVED,
                    provenance=Provenance(
                        source_id="runtime-binary-environment",
                        trust=TrustLevel.VERIFIED,
                        source_kind="resource_meter",
                    ),
                ),
            ),
        )
        return EnvironmentStep(
            execution=execution,
            observation=observation,
            outcome=outcome,
            closed_at=ended,
        )


class _ExactBeliefUpdater:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def assimilate(
        self,
        prior: Belief,
        experience: ExperienceEvent,
    ) -> BeliefUpdate:
        signal = _experience_signal(experience)
        action = experience.decision
        if action is None:
            raise ValueError("runtime reference experience is missing its decision")
        selected_kind = action.selected_assessment.action.action_kind
        if selected_kind != "probe:relevant":
            raise ValueError("runtime reference updater expected a relevant probe")
        prior_rule_one = _belief_rule_one(prior)
        posterior_rule_one = posterior_after_signal(
            prior_rule_one,
            PROBES[0],
            signal,
        )
        formed_at = TimePoint(experience.closed_at.tick + 1)
        posterior = Belief(
            belief_id=self._identities.next("posterior-belief"),
            agent_id=prior.agent_id,
            target=prior.target,
            information_set=InformationSet(
                information_set_id=self._identities.next("posterior-information"),
                agent_id=prior.agent_id,
                as_of=experience.closed_at,
                observations=(
                    *prior.information_set.observations,
                    experience.observation,
                ),
                memory_version="runtime-memory-v1",
            ),
            distribution=Distribution(
                distribution_id=self._identities.next("posterior-distribution"),
                family="categorical",
                support="rule={0,1}",
                parameters=(1.0 - posterior_rule_one, posterior_rule_one),
                representation_version=_REPRESENTATION_VERSION,
                event_shape=(2,),
            ),
            formed_at=formed_at,
            model_version=_MODEL_VERSION,
            representation_version=_REPRESENTATION_VERSION,
        )
        return BeliefUpdate(
            update_id=self._identities.next("belief-update"),
            prior=prior,
            experience=experience,
            posterior=posterior,
            updater_version="runtime-exact-bayes-v1",
            updated_at=formed_at,
        )


class _ExactScorer:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def score(
        self,
        prediction: Prediction,
        experience: ExperienceEvent,
    ) -> ProperScore:
        probabilities = _distribution_probabilities(prediction.distribution)
        return ProperScore(
            score_id=self._identities.next("proper-score"),
            prediction_id=prediction.prediction_id,
            realized_evidence_id=experience.observation.evidence.evidence_id,
            rule="categorical_log_score",
            value=categorical_log_score(
                probabilities,
                _experience_signal(experience),
            ),
            unit="nats",
            scorer_version="runtime-categorical-log-score-v1",
            scored_at=TimePoint(experience.closed_at.tick + 1),
        )


class _ExactEffectAssessor:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def effect(self, update: BeliefUpdate) -> EpistemicEffect:
        before = entropy(_belief_probabilities(update.prior))
        after = entropy(_belief_probabilities(update.posterior))
        return EpistemicEffect(
            effect_id=self._identities.next("epistemic-effect"),
            belief_update_id=update.update_id,
            target_id=update.prior.target.target_id,
            kind=EpistemicEffectKind.INFORMATION_GAIN,
            measure="realized_posterior_entropy_reduction",
            before=before,
            after=after,
            improvement=before - after,
            higher_is_better=False,
            evaluator_version="runtime-exact-entropy-v1",
            evaluated_at=update.updated_at,
        )


class _ConfigurationLearner:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def update(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
    ) -> UpdateReceipt:
        inputs = tuple(transitions)
        if not inputs:
            raise ValueError("runtime reference learner requires a transition")
        completed = TimePoint(snapshot.captured_at.tick + 1)
        return UpdateReceipt(
            receipt_id=self._identities.next("update-receipt"),
            agent_id=snapshot.agent_id,
            transitions=inputs,
            learner_version="runtime-reference-learner-v1",
            status=UpdateStatus.APPLIED,
            previous_configuration_version=snapshot.configuration_version,
            new_configuration_version="runtime-config-v1",
            previous_model_version=snapshot.model_version,
            new_model_version=snapshot.model_version,
            previous_representation_version=snapshot.representation_version,
            new_representation_version=snapshot.representation_version,
            previous_policy_version=snapshot.policy_version,
            new_policy_version=snapshot.policy_version,
            started_at=completed,
            completed_at=completed,
            metrics=(("consumed_transitions", float(len(inputs))),),
        )


def run_runtime_integrity_lane() -> RuntimeIntegrityResult:
    """Drive one exact interaction and update through the authoritative runtime."""

    agent_id = "runtime-reference-agent"
    task = BinaryRuleTask("runtime-task-000", 0)
    identities = CounterIdentitySource("runtime-lane")
    initial_belief = Belief(
        belief_id="runtime-initial-belief",
        agent_id=agent_id,
        target=EpistemicTarget(
            target_id=f"binary-rule:{task.task_id}",
            description="persistent XOR rule in the runtime integrity lane",
            target_kind="latent_law",
        ),
        information_set=InformationSet(
            information_set_id="runtime-initial-information",
            agent_id=agent_id,
            as_of=TimePoint(0),
            observations=(),
            memory_version="runtime-memory-v0",
        ),
        distribution=Distribution(
            distribution_id="runtime-initial-distribution",
            family="categorical",
            support="rule={0,1}",
            parameters=(0.5, 0.5),
            representation_version=_REPRESENTATION_VERSION,
            event_shape=(2,),
        ),
        formed_at=TimePoint(0),
        model_version=_MODEL_VERSION,
        representation_version=_REPRESENTATION_VERSION,
    )
    initial_snapshot = AgentSnapshot(
        snapshot_id="runtime-initial-snapshot",
        agent_id=agent_id,
        captured_at=TimePoint(0),
        belief=initial_belief,
        configuration_version="runtime-config-v0",
        memory_version="runtime-memory-v0",
        knowledge_version="runtime-knowledge-v0",
        model_version=_MODEL_VERSION,
        representation_version=_REPRESENTATION_VERSION,
        policy_version=_POLICY_VERSION,
        resources=ResourceLedger(
            ledger_id="runtime-initial-resources",
            started_at=TimePoint(0),
            completed_at=TimePoint(0),
        ),
    )
    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    policy = MaxValuePolicy(
        agent_id=agent_id,
        policy_version=_POLICY_VERSION,
        assessor=_ExactCandidateAssessor(identities),
        identities=identities,
    )
    runtime = EpistemicAgent(
        state=AgentState(initial_snapshot),
        policy=policy,
        belief_updater=_ExactBeliefUpdater(identities),
        scorer=_ExactScorer(identities),
        effect_assessor=_ExactEffectAssessor(identities),
        learner=_ConfigurationLearner(identities),
        experience_store=store,
        ledger=ledger,
        identities=identities,
    )
    goal = Goal(
        goal_id="runtime-diagnostic-goal",
        task_id=task.task_id,
        target=EpistemicTarget(
            target_id=f"probe-signal:{task.task_id}",
            description="next diagnostic signal in the runtime integrity lane",
            target_kind="future_observation",
        ),
        description="acquire decision-relevant rule evidence",
        issued_at=TimePoint(1),
        preference_version="runtime-binary-utility-v1",
        deadline=TimePoint(10),
    )
    interaction = runtime.interact(
        _ExactEnvironment(task, identities),
        goal,
        context=InteractionContext(
            run_id="runtime-integrity-run",
            task_id=task.task_id,
            episode_id="runtime-integrity-episode",
            step_index=0,
        ),
        decide_at=TimePoint(1),
    )
    receipt = runtime.learn((interaction.transition,), at=TimePoint(5))
    final_snapshot = runtime.snapshot(TimePoint(7))
    return RuntimeIntegrityResult(
        interaction=interaction,
        receipt=receipt,
        final_snapshot=final_snapshot,
        store=store,
        ledger=ledger,
        canonical_experience=store.get(interaction.experience.experience_id) is interaction.experience,
        canonical_transition=ledger.get_transition(interaction.transition.transition_id) is interaction.transition,
        canonical_update=ledger.get_update(receipt.receipt_id) is receipt,
        selected_relevant_probe=(interaction.decision.selected_assessment.action.action_kind == "probe:relevant"),
        learned_configuration=(
            final_snapshot.configuration_version == "runtime-config-v1" and final_snapshot.latest_update is receipt
        ),
    )


def _belief_rule_one(belief: Belief) -> float:
    return _belief_probabilities(belief)[1]


def _belief_probabilities(belief: Belief) -> tuple[float, float]:
    return _distribution_probabilities(belief.distribution)


def _distribution_probabilities(distribution: Distribution) -> tuple[float, float]:
    parameters = distribution.parameters
    if (
        not isinstance(parameters, tuple)
        or len(parameters) != 2
        or not all(isinstance(value, float) for value in parameters)
    ):
        raise TypeError("runtime reference distribution is not a float pair")
    return cast(tuple[float, float], parameters)


def _experience_signal(experience: ExperienceEvent) -> int:
    payload = experience.observation.evidence.payload
    if not isinstance(payload, dict):
        raise TypeError("runtime reference observation payload is not an object")
    signal = payload.get("signal")
    if not isinstance(signal, int) or isinstance(signal, bool) or signal not in (0, 1):
        raise ValueError("runtime reference signal must be 0 or 1")
    return signal


__all__ = ("RuntimeIntegrityResult", "run_runtime_integrity_lane")
