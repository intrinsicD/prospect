"""Deterministic reference lifecycle for causal adaptive-agent evaluation.

The environment contains task-scoped binary rules.  A cue ``c`` is mapped to the
correct terminal response by ``c XOR rule``.  Learning the rule therefore transfers
to held-out cues without putting the rule itself in an observation payload.

Three probes deliberately have the same acquisition cost:

* ``relevant`` is a calibrated noisy channel about the rule;
* ``irrelevant`` is a fair nuisance bit independent of the rule;
* ``noise`` is a fair random observation whose entropy stays maximal.

The exact policy compares expected information gain, expected value of sample
information, and net value of information.  It never substitutes raw observation
entropy for information about the named rule.  The implementation is dependency
light so it can serve as an auditable semantic oracle for learned approximations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
from typing import Final, cast

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
    EvaluationMetric,
    EvaluationRecord,
    Evidence,
    EvidenceLineage,
    EvidenceOrigin,
    ExecutedAction,
    ExecutionStatus,
    ExperienceEvent,
    ExperienceKind,
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
    UpdateReceipt,
    UpdateStatus,
    Utility,
)
from prospect.epistemics.information import (
    InformationValueResult,
    bayes_posterior,
    entropy,
    expected_value_of_sample_information,
    predictive_distribution,
)
from prospect.epistemics.scoring import brier_score, categorical_log_score

_REPRESENTATION_VERSION: Final = "categorical-binary-v1"
_MODEL_VERSION: Final = "exact-binary-channel-v1"
_POLICY_VERSION: Final = "exact-voi-v1"
_LEARNER_VERSION: Final = "exact-bayes-v1"
_CHECKPOINT_SCHEMA: Final = "prospect.epistemic.reference.v1"
_RULE_UTILITIES: Final = ((1.0, 0.0), (0.0, 1.0), (0.65, 0.65))
_TOL: Final = 1e-12


class ProbeAction(StrEnum):
    """Diagnostic and terminal action kinds in the reference problem."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    NOISE = "noise"
    EXPLOIT = "exploit"


class CollectionPolicy(StrEnum):
    """Policies compared under a matched acquisition budget."""

    EXACT_VOI = "exact_voi"
    RAW_ENTROPY = "raw_entropy"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class BinaryRuleTask:
    """One persistent task identity and its environment-hidden binary rule."""

    task_id: str
    rule: int

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be nonempty")
        if self.rule not in (0, 1):
            raise ValueError("rule must be 0 or 1")

    def label(self, cue: int) -> int:
        """Return the correct terminal label for a held-out cue."""

        if cue not in (0, 1):
            raise ValueError("cue must be 0 or 1")
        return cue ^ self.rule


@dataclass(frozen=True, slots=True)
class ProbeSpecification:
    """Known observation model and acquisition cost for one probe."""

    action: ProbeAction
    accuracy: float
    cost: float

    def __post_init__(self) -> None:
        if self.action is ProbeAction.EXPLOIT:
            raise ValueError("exploit is not a probe specification")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be a probability")
        if self.cost < 0.0:
            raise ValueError("probe cost must be non-negative")
        if self.action is ProbeAction.RELEVANT and self.accuracy <= 0.5:
            raise ValueError("relevant probe accuracy must exceed chance")
        if self.action is not ProbeAction.RELEVANT and self.accuracy != 0.5:
            raise ValueError("irrelevant and noise probes must be independent of the rule")

    @property
    def likelihoods(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return rows ``P(signal | rule=0)`` and ``P(signal | rule=1)``."""

        if self.action is ProbeAction.RELEVANT:
            return (
                (self.accuracy, 1.0 - self.accuracy),
                (1.0 - self.accuracy, self.accuracy),
            )
        return ((0.5, 0.5), (0.5, 0.5))


RELEVANT_PROBE: Final = ProbeSpecification(ProbeAction.RELEVANT, accuracy=0.8, cost=0.04)
IRRELEVANT_PROBE: Final = ProbeSpecification(ProbeAction.IRRELEVANT, accuracy=0.5, cost=0.04)
NOISE_PROBE: Final = ProbeSpecification(ProbeAction.NOISE, accuracy=0.5, cost=0.04)
PROBES: Final = (RELEVANT_PROBE, IRRELEVANT_PROBE, NOISE_PROBE)


@dataclass(frozen=True, slots=True)
class ProbeDiagnostics:
    """Exact pre-observation quantities for one candidate probe."""

    action: ProbeAction
    observation_entropy_nats: float
    expected_information_gain_nats: float
    expected_decision_value: float
    acquisition_cost: float
    net_value: float


@dataclass(frozen=True, slots=True)
class LearningStep:
    """One real interaction and its optional persistent learning records."""

    experience: ExperienceEvent
    transition: EpistemicTransition | None
    receipt: UpdateReceipt | None
    posterior_rule_one: float


@dataclass(frozen=True, slots=True)
class _TaskState:
    posterior_rule_one: float = 0.5
    observations: tuple[Observation, ...] = ()


def probe_specification(action: ProbeAction) -> ProbeSpecification:
    """Return the immutable model for a diagnostic action."""

    for probe in PROBES:
        if probe.action is action:
            return probe
    raise ValueError(f"{action.value!r} is not a diagnostic probe")


def diagnose_probe(
    posterior_rule_one: float,
    probe: ProbeSpecification,
) -> ProbeDiagnostics:
    """Compute exact observation entropy, EIG, EVSI, and net VOI."""

    prior = (1.0 - posterior_rule_one, posterior_rule_one)
    result = expected_value_of_sample_information(
        prior,
        probe.likelihoods,
        _RULE_UTILITIES,
        acquisition_cost=probe.cost,
    )
    return _diagnostics_from_result(prior, probe, result)


def select_probe_by_voi(posterior_rule_one: float) -> ProbeAction:
    """Choose the strictly positive-net-VOI probe, otherwise exploit."""

    best = ProbeAction.EXPLOIT
    best_value = 0.0
    for probe in PROBES:
        value = diagnose_probe(posterior_rule_one, probe).net_value
        if value > best_value + _TOL:
            best = probe.action
            best_value = value
    return best


def select_probe_by_raw_entropy(posterior_rule_one: float) -> ProbeAction:
    """Choose by observation entropy alone.

    ``NOISE`` comes first intentionally: at an uninformative prior all three outputs
    are equiprobable, so a deterministic raw-entropy policy has no semantic basis for
    preferring the diagnostic channel.  Once the rule belief sharpens, noise remains
    maximally unpredictable and the failure becomes strict rather than a tie.
    """

    ordered = (NOISE_PROBE, IRRELEVANT_PROBE, RELEVANT_PROBE)
    return max(
        ordered,
        key=lambda probe: diagnose_probe(posterior_rule_one, probe).observation_entropy_nats,
    ).action


def posterior_after_signal(
    posterior_rule_one: float,
    probe: ProbeSpecification,
    signal: int,
) -> float:
    """Return the exact posterior after one causally available probe signal."""

    if signal not in (0, 1):
        raise ValueError("signal must be 0 or 1")
    return bayes_posterior(
        (1.0 - posterior_rule_one, posterior_rule_one),
        probe.likelihoods,
        signal,
    )[1]


def expected_external_utility(
    posterior_rule_one: float,
    task: BinaryRuleTask,
) -> float:
    """Exact frozen-policy utility on held-out cues, including optional probe cost."""

    selected = select_probe_by_voi(posterior_rule_one)
    if selected is ProbeAction.EXPLOIT:
        return _terminal_utility(posterior_rule_one, task.rule)

    probe = probe_specification(selected)
    expected = 0.0
    for signal, probability in enumerate(probe.likelihoods[task.rule]):
        transient_posterior = posterior_after_signal(posterior_rule_one, probe, signal)
        expected += probability * _terminal_utility(transient_posterior, task.rule)
    return expected - probe.cost


class ExactRuleAgent:
    """Small persistent learner whose inputs and outputs are linked domain records.

    The reference agent owns an append-only interaction trace and task-scoped
    posteriors.  Evaluation routines only read this state.  Checkpoints contain the
    posteriors, evidence history, replay-consumption identities, clock, and resource
    counters needed to reproduce its frozen behavior after restart.
    """

    def __init__(self, agent_id: str = "reference-agent") -> None:
        if not agent_id.strip():
            raise ValueError("agent_id must be nonempty")
        self.agent_id = agent_id
        self._states: dict[str, _TaskState] = {}
        self._experiences: list[ExperienceEvent] = []
        self._transitions: list[EpistemicTransition] = []
        self._receipts: list[UpdateReceipt] = []
        self._experience_ids: set[str] = set()
        self._consumed_ids: set[str] = set()
        self._configuration_revision = 0
        self._tick = 0
        self._total_probe_cost = 0.0
        self._total_environment_steps = 0

    @property
    def experiences(self) -> tuple[ExperienceEvent, ...]:
        return tuple(self._experiences)

    @property
    def transitions(self) -> tuple[EpistemicTransition, ...]:
        return tuple(self._transitions)

    @property
    def receipts(self) -> tuple[UpdateReceipt, ...]:
        return tuple(self._receipts)

    @property
    def total_probe_cost(self) -> float:
        return self._total_probe_cost

    @property
    def total_environment_steps(self) -> int:
        return self._total_environment_steps

    @property
    def configuration_version(self) -> str:
        return f"reference-config-r{self._configuration_revision}"

    def posterior(self, task_id: str) -> float:
        """Return the task posterior without creating or mutating task state."""

        state = self._states.get(task_id)
        return 0.5 if state is None else state.posterior_rule_one

    def select_action(self, task_id: str) -> ProbeAction:
        """Select an exact-net-VOI action from current persistent knowledge."""

        return select_probe_by_voi(self.posterior(task_id))

    def interact(
        self,
        task: BinaryRuleTask,
        action: ProbeAction,
        signal: int,
        *,
        experience_id: str,
        update: bool,
        require_optimal: bool = False,
        origin: EvidenceOrigin = EvidenceOrigin.OBSERVED,
        source_id: str = "binary-rule-environment",
        source_kind: str = "diagnostic_sensor",
        trust: TrustLevel = TrustLevel.HIGH,
        producer_version: str | None = None,
        parent_evidence_ids: tuple[str, ...] = (),
        behavior_policy_version: str | None = None,
    ) -> LearningStep:
        """Execute, record, and optionally learn from one diagnostic interaction."""

        if action is ProbeAction.EXPLOIT:
            raise ValueError("interact requires a diagnostic probe")
        if signal not in (0, 1):
            raise ValueError("signal must be 0 or 1")
        if not experience_id.strip() or experience_id in self._experience_ids:
            raise ValueError("experience_id must be nonempty and unique")
        if require_optimal and self.select_action(task.task_id) is not action:
            raise ValueError("assigned action is not the exact net-VOI selection")
        policy_version = behavior_policy_version or (
            _POLICY_VERSION if require_optimal else "assigned-probe-control-v1"
        )

        probe = probe_specification(action)
        prior_state = self._states.get(task.task_id, _TaskState())
        base = self._tick + 1
        prior = self._belief(task.task_id, prior_state, formed_at=base, suffix=f"{experience_id}-prior")
        experience = self._experience(
            task=task,
            probe=probe,
            signal=signal,
            experience_id=experience_id,
            prior=prior,
            base=base,
            origin=origin,
            source_id=source_id,
            source_kind=source_kind,
            trust=trust,
            producer_version=producer_version,
            parent_evidence_ids=parent_evidence_ids,
            policy_version=policy_version,
        )
        self._experience_ids.add(experience_id)
        self._experiences.append(experience)
        self._total_probe_cost += probe.cost
        self._total_environment_steps += 1

        posterior = posterior_after_signal(prior_state.posterior_rule_one, probe, signal)
        next_state = _TaskState(
            posterior_rule_one=posterior if update else prior_state.posterior_rule_one,
            observations=(*prior_state.observations, experience.observation),
        )

        if not update:
            self._states[task.task_id] = next_state
            self._tick = base + 3
            return LearningStep(
                experience=experience,
                transition=None,
                receipt=None,
                posterior_rule_one=next_state.posterior_rule_one,
            )

        if experience_id in self._consumed_ids:
            raise ValueError("experience was already consumed")
        transition = self._transition(
            experience=experience,
            prior=prior,
            posterior_probability=posterior,
            posterior_observations=next_state.observations,
            base=base,
        )
        previous_configuration = self.configuration_version
        self._configuration_revision += 1
        receipt = UpdateReceipt(
            receipt_id=f"receipt-{experience_id}",
            agent_id=self.agent_id,
            transitions=(transition,),
            learner_version=_LEARNER_VERSION,
            status=UpdateStatus.APPLIED,
            previous_configuration_version=previous_configuration,
            new_configuration_version=self.configuration_version,
            previous_model_version=_MODEL_VERSION,
            new_model_version=_MODEL_VERSION,
            previous_representation_version=_REPRESENTATION_VERSION,
            new_representation_version=_REPRESENTATION_VERSION,
            previous_policy_version=_POLICY_VERSION,
            new_policy_version=_POLICY_VERSION,
            started_at=TimePoint(base + 5),
            completed_at=TimePoint(base + 5),
            metrics=(
                ("posterior_rule_one", posterior),
                (
                    "realized_entropy_reduction_nats",
                    entropy((1.0 - prior_state.posterior_rule_one, prior_state.posterior_rule_one))
                    - entropy((1.0 - posterior, posterior)),
                ),
            ),
        )
        self._consumed_ids.add(experience_id)
        self._states[task.task_id] = next_state
        self._transitions.append(transition)
        self._receipts.append(receipt)
        self._tick = base + 5
        return LearningStep(
            experience=experience,
            transition=transition,
            receipt=receipt,
            posterior_rule_one=posterior,
        )

    def snapshot(self, task_id: str, *, snapshot_id: str) -> AgentSnapshot:
        """Create an immutable current-state view without mutating the agent."""

        captured = self._tick + 1
        state = self._states.get(task_id, _TaskState())
        belief = self._belief(task_id, state, formed_at=captured, suffix=f"{snapshot_id}-active")
        return AgentSnapshot(
            snapshot_id=snapshot_id,
            agent_id=self.agent_id,
            captured_at=TimePoint(captured),
            belief=belief,
            configuration_version=self.configuration_version,
            memory_version=f"reference-memory-{len(self._experience_ids)}",
            knowledge_version=f"sha256:{self.state_digest()}",
            model_version=_MODEL_VERSION,
            representation_version=_REPRESENTATION_VERSION,
            policy_version=_POLICY_VERSION,
            resources=ResourceLedger(
                ledger_id=f"training-resources-{snapshot_id}",
                started_at=TimePoint(0),
                completed_at=TimePoint(self._tick),
                uses=(
                    ResourceUse(
                        resource="environment_steps",
                        amount=float(self._total_environment_steps),
                        unit="step",
                    ),
                    ResourceUse(
                        resource="probe_cost",
                        amount=self._total_probe_cost,
                        unit="utility",
                    ),
                ),
            ),
            latest_update=self._receipts[-1] if self._receipts else None,
        )

    def checkpoint_json(self) -> str:
        """Return a canonical, complete reference-state checkpoint."""

        task_states: list[dict[str, object]] = []
        for task_id in sorted(self._states):
            state = self._states[task_id]
            task_states.append(
                {
                    "task_id": task_id,
                    "posterior_rule_one": state.posterior_rule_one,
                    "observations": [_serialize_observation(observation) for observation in state.observations],
                }
            )
        payload: dict[str, object] = {
            "schema": _CHECKPOINT_SCHEMA,
            "agent_id": self.agent_id,
            "configuration_revision": self._configuration_revision,
            "tick": self._tick,
            "total_probe_cost": self._total_probe_cost,
            "total_environment_steps": self._total_environment_steps,
            "experience_ids": sorted(self._experience_ids),
            "consumed_ids": sorted(self._consumed_ids),
            "task_states": task_states,
        }
        return json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def state_digest(self) -> str:
        """Return a stable digest used to prove frozen evaluation and restart parity."""

        return sha256(self.checkpoint_json().encode("utf-8")).hexdigest()

    def save_checkpoint(self, path: Path) -> None:
        """Persist the canonical checkpoint to ``path``."""

        path.write_text(self.checkpoint_json(), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path: Path) -> ExactRuleAgent:
        """Restart an agent from a persisted canonical checkpoint."""

        return cls.from_checkpoint_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_checkpoint_json(cls, checkpoint: str) -> ExactRuleAgent:
        """Restart an agent from :meth:`checkpoint_json` output."""

        decoded: object = json.loads(checkpoint)
        root = _mapping(decoded, "checkpoint")
        if _string(root, "schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        agent = cls(_string(root, "agent_id"))
        agent._configuration_revision = _nonnegative_int(root, "configuration_revision")
        agent._tick = _nonnegative_int(root, "tick")
        agent._total_probe_cost = _nonnegative_float(root, "total_probe_cost")
        agent._total_environment_steps = _nonnegative_int(root, "total_environment_steps")
        agent._experience_ids = set(_string_list(root, "experience_ids"))
        agent._consumed_ids = set(_string_list(root, "consumed_ids"))
        if not agent._consumed_ids.issubset(agent._experience_ids):
            raise ValueError("checkpoint consumed ids are not experience ids")

        state_items = _sequence(root, "task_states")
        for raw_state in state_items:
            state = _mapping(raw_state, "task state")
            task_id = _string(state, "task_id")
            if task_id in agent._states:
                raise ValueError("checkpoint contains duplicate task ids")
            posterior = _probability(state, "posterior_rule_one")
            observations = tuple(
                _deserialize_observation(item, agent.agent_id) for item in _sequence(state, "observations")
            )
            agent._states[task_id] = _TaskState(posterior, observations)
        if agent.checkpoint_json() != checkpoint:
            raise ValueError("checkpoint is non-canonical or internally inconsistent")
        return agent

    def _belief(
        self,
        task_id: str,
        state: _TaskState,
        *,
        formed_at: int,
        suffix: str,
    ) -> Belief:
        target = _rule_target(task_id)
        return Belief(
            belief_id=f"belief-{suffix}",
            agent_id=self.agent_id,
            target=target,
            information_set=InformationSet(
                information_set_id=f"information-{suffix}",
                agent_id=self.agent_id,
                as_of=TimePoint(formed_at),
                observations=state.observations,
                memory_version=f"reference-memory-{len(self._experience_ids)}",
            ),
            distribution=Distribution(
                distribution_id=f"belief-distribution-{suffix}",
                family="categorical",
                support="rule={0,1}",
                parameters=(1.0 - state.posterior_rule_one, state.posterior_rule_one),
                representation_version=_REPRESENTATION_VERSION,
                event_shape=(2,),
            ),
            formed_at=TimePoint(formed_at),
            model_version=_MODEL_VERSION,
            representation_version=_REPRESENTATION_VERSION,
        )

    def _experience(
        self,
        *,
        task: BinaryRuleTask,
        probe: ProbeSpecification,
        signal: int,
        experience_id: str,
        prior: Belief,
        base: int,
        origin: EvidenceOrigin,
        source_id: str,
        source_kind: str,
        trust: TrustLevel,
        producer_version: str | None,
        parent_evidence_ids: tuple[str, ...],
        policy_version: str,
    ) -> ExperienceEvent:
        action = Action(
            action_id=f"action-{experience_id}",
            action_kind=f"probe:{probe.action.value}",
            parameters={
                "task_id": task.task_id,
                "probe_kind": probe.action.value,
                "accuracy": probe.accuracy,
                "cost": probe.cost,
            },
        )
        prediction_probabilities = predictive_distribution(
            _belief_probabilities(prior),
            probe.likelihoods,
        )
        signal_target = EpistemicTarget(
            target_id=f"probe-signal:{task.task_id}",
            description=f"next diagnostic signal for task {task.task_id}",
            target_kind="future_observation",
        )
        prediction = Prediction(
            prediction_id=f"prediction-{experience_id}",
            prior_belief=prior,
            action=action,
            target=signal_target,
            distribution=Distribution(
                distribution_id=f"prediction-distribution-{experience_id}",
                family="categorical",
                support="signal={0,1}",
                parameters=prediction_probabilities,
                representation_version=_REPRESENTATION_VERSION,
                event_shape=(2,),
            ),
            issued_at=TimePoint(base),
            horizon_end=TimePoint(base + 3),
            model_version=_MODEL_VERSION,
            representation_version=_REPRESENTATION_VERSION,
            calibration_version="exact-channel-calibration-v1",
        )
        diagnostics = diagnose_probe(_belief_rule_one(prior), probe)
        goal = Goal(
            goal_id=f"goal-{experience_id}",
            task_id=task.task_id,
            target=signal_target,
            description="acquire decision-relevant evidence before terminal exploitation",
            issued_at=TimePoint(base),
            preference_version="binary-rule-utility-v1",
            deadline=TimePoint(base + 10),
        )
        intention = IntendedAction(
            intention_id=f"intention-{experience_id}",
            agent_id=self.agent_id,
            action=action,
            intended_at=TimePoint(base + 1),
        )
        utility = Utility(
            utility_id=f"utility-{experience_id}",
            goal_id=goal.goal_id,
            prediction_id=prediction.prediction_id,
            expected_value=_best_terminal_value(_belief_rule_one(prior)),
            unit="external_utility",
            evaluator_version="exact-terminal-utility-v1",
            assessed_at=TimePoint(base + 1),
        )
        information_value = InformationValue(
            information_value_id=f"information-value-{experience_id}",
            prior_belief_id=prior.belief_id,
            action_id=action.action_id,
            target_id=signal_target.target_id,
            expected_reduction=diagnostics.expected_decision_value,
            expected_cost=probe.cost,
            unit="external_utility",
            evaluator_version="exact-evsi-v1",
            assessed_at=TimePoint(base + 1),
        )
        assessment = CandidateAssessment(
            assessment_id=f"assessment-{experience_id}",
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
            evaluator_version="exact-value-decomposition-v1",
            assessed_at=TimePoint(base + 1),
        )
        decision = DecisionRecord(
            decision_id=f"decision-{experience_id}",
            agent_id=self.agent_id,
            belief=prior,
            goal=goal,
            intended_action=intention,
            alternatives=(assessment,),
            selected_assessment=assessment,
            policy_version=policy_version,
            decided_at=TimePoint(base + 1),
        )
        execution = ExecutedAction(
            execution_id=f"execution-{experience_id}",
            intention=intention,
            status=ExecutionStatus.SUCCEEDED,
            started_at=TimePoint(base + 2),
            ended_at=TimePoint(base + 3),
            realized_action=action,
        )
        observation_evidence_id = f"observation-{experience_id}"
        observation = Observation(
            observation_id=observation_evidence_id,
            agent_id=self.agent_id,
            modality="diagnostic_signal",
            evidence=Evidence(
                evidence_id=observation_evidence_id,
                payload={
                    "task_id": task.task_id,
                    "probe_kind": probe.action.value,
                    "signal": signal,
                    "accuracy": probe.accuracy,
                },
                occurred_at=TimePoint(base + 3),
                available_at=TimePoint(base + 3),
                lineage=EvidenceLineage(
                    evidence_id=observation_evidence_id,
                    origin=origin,
                    provenance=Provenance(
                        source_id=source_id,
                        trust=trust,
                        source_kind=source_kind,
                    ),
                    parent_evidence_ids=parent_evidence_ids,
                    producer_version=producer_version,
                ),
            ),
        )
        outcome_evidence_id = f"outcome-{experience_id}"
        outcome = Outcome(
            outcome_id=f"outcome-record-{experience_id}",
            execution_id=execution.execution_id,
            evidence=Evidence(
                evidence_id=outcome_evidence_id,
                payload={"probe_cost": probe.cost},
                occurred_at=TimePoint(base + 3),
                available_at=TimePoint(base + 3),
                lineage=EvidenceLineage(
                    evidence_id=outcome_evidence_id,
                    origin=EvidenceOrigin.OBSERVED,
                    provenance=Provenance(
                        source_id="binary-rule-environment",
                        trust=TrustLevel.VERIFIED,
                        source_kind="resource_meter",
                    ),
                ),
            ),
        )
        return ExperienceEvent(
            experience_id=experience_id,
            agent_id=self.agent_id,
            run_id=f"run-{self.agent_id}",
            task_id=task.task_id,
            episode_id=f"episode-{task.task_id}",
            step_index=len(prior.information_set.observations),
            kind=ExperienceKind.INTERACTION,
            observation=observation,
            outcome=outcome,
            terminated=False,
            truncated=False,
            discount=1.0,
            behavior_policy_version=policy_version,
            closed_at=TimePoint(base + 3),
            decision=decision,
            execution=execution,
        )

    def _transition(
        self,
        *,
        experience: ExperienceEvent,
        prior: Belief,
        posterior_probability: float,
        posterior_observations: tuple[Observation, ...],
        base: int,
    ) -> EpistemicTransition:
        suffix = experience.experience_id
        posterior = self._belief(
            prior.target.target_id.removeprefix("binary-rule:"),
            _TaskState(posterior_probability, posterior_observations),
            formed_at=base + 4,
            suffix=f"{suffix}-posterior",
        )
        update = BeliefUpdate(
            update_id=f"belief-update-{suffix}",
            prior=prior,
            experience=experience,
            posterior=posterior,
            updater_version=_LEARNER_VERSION,
            updated_at=TimePoint(base + 4),
        )
        prediction = _required_decision(experience).selected_assessment.prediction
        prediction_probabilities = _distribution_probabilities(prediction.distribution)
        signal = _experience_signal(experience)
        score = ProperScore(
            score_id=f"proper-score-{suffix}",
            prediction_id=prediction.prediction_id,
            realized_evidence_id=experience.observation.evidence.evidence_id,
            rule="categorical_log_score",
            value=categorical_log_score(prediction_probabilities, signal),
            unit="nats",
            scorer_version="categorical-log-score-v1",
            scored_at=TimePoint(base + 4),
        )
        prior_entropy = entropy(_belief_probabilities(prior))
        posterior_entropy = entropy((1.0 - posterior_probability, posterior_probability))
        effect = EpistemicEffect(
            effect_id=f"epistemic-effect-{suffix}",
            belief_update_id=update.update_id,
            target_id=prior.target.target_id,
            kind=EpistemicEffectKind.INFORMATION_GAIN,
            measure="realized_posterior_entropy_reduction",
            before=prior_entropy,
            after=posterior_entropy,
            improvement=prior_entropy - posterior_entropy,
            higher_is_better=False,
            evaluator_version="exact-entropy-v1",
            evaluated_at=TimePoint(base + 4),
            externally_calibrated=False,
        )
        return EpistemicTransition(
            transition_id=f"transition-{suffix}",
            experience=experience,
            belief_update=update,
            proper_scores=(score,),
            effects=(effect,),
            created_at=TimePoint(base + 4),
        )


def evaluate_frozen(
    agent: ExactRuleAgent,
    tasks: Sequence[BinaryRuleTask],
    *,
    evaluation_id: str,
) -> EvaluationRecord:
    """Externally evaluate a snapshot without changing any agent state."""

    if not tasks:
        raise ValueError("evaluation needs at least one task")
    digest_before = agent.state_digest()
    log_scores = tuple(
        categorical_log_score(
            (1.0 - agent.posterior(task.task_id), agent.posterior(task.task_id)),
            task.rule,
        )
        for task in tasks
    )
    brier_scores = tuple(
        brier_score(
            (1.0 - agent.posterior(task.task_id), agent.posterior(task.task_id)),
            task.rule,
        )
        for task in tasks
    )
    utilities = tuple(expected_external_utility(agent.posterior(task.task_id), task) for task in tasks)
    snapshot = agent.snapshot(tasks[0].task_id, snapshot_id=f"snapshot-{evaluation_id}")
    started = snapshot.captured_at.tick + 1
    completed = started + 1
    record = EvaluationRecord(
        evaluation_id=evaluation_id,
        agent_id=agent.agent_id,
        task_id="binary-rule-heldout-cues",
        evaluator_version="exact-frozen-evaluator-v1",
        snapshot=snapshot,
        started_at=TimePoint(started),
        completed_at=TimePoint(completed),
        metrics=(
            EvaluationMetric("mean_log_score", _mean(log_scores), "nats"),
            EvaluationMetric("mean_brier_score", _mean(brier_scores), "score"),
            EvaluationMetric("mean_external_utility", _mean(utilities), "utility"),
            EvaluationMetric("mean_regret", 1.0 - _mean(utilities), "utility"),
            EvaluationMetric("heldout_cues", float(2 * len(tasks)), "cue"),
        ),
        resources=ResourceLedger(
            ledger_id=f"evaluation-resources-{evaluation_id}",
            started_at=TimePoint(started),
            completed_at=TimePoint(completed),
            uses=(
                ResourceUse(
                    resource="heldout_environment_steps",
                    amount=float(2 * len(tasks)),
                    unit="step",
                ),
            ),
        ),
        transition_ids=(),
        training_updates_allowed=False,
        update_receipts=(),
    )
    if agent.state_digest() != digest_before:
        raise RuntimeError("frozen evaluation mutated agent state")
    return record


def evaluation_metric(record: EvaluationRecord, name: str) -> float:
    """Extract one named metric from an evaluation record."""

    for metric in record.metrics:
        if metric.name == name:
            return metric.value
    raise KeyError(name)


def empirically_exact_training_signals(task: BinaryRuleTask, round_index: int) -> int:
    """Return a deterministic 0.8-accurate two-probe joint schedule.

    Within each block of 25 tasks there are 16 ``correct/correct`` pairs, eight
    mixed pairs, and one ``wrong/wrong`` pair.  Those frequencies are exactly the
    joint probabilities of two independent 0.8-accurate signals.
    """

    if round_index not in (0, 1):
        raise ValueError("round_index must be 0 or 1")
    numeric_suffix = int(task.task_id.rsplit("-", maxsplit=1)[-1])
    within_rule_block = (numeric_suffix // 2) % 25
    if within_rule_block < 16:
        correct = True
    elif within_rule_block < 24:
        correct = round_index == (within_rule_block % 2)
    else:
        correct = False
    return task.rule if correct else 1 - task.rule


def make_balanced_task_suite(prefix: str = "task", *, tasks_per_rule: int = 25) -> tuple[BinaryRuleTask, ...]:
    """Create paired rule-0/rule-1 tasks with stable numeric identities."""

    if tasks_per_rule <= 0:
        raise ValueError("tasks_per_rule must be positive")
    tasks: list[BinaryRuleTask] = []
    for index in range(tasks_per_rule):
        tasks.append(BinaryRuleTask(f"{prefix}-{2 * index:03d}", 0))
        tasks.append(BinaryRuleTask(f"{prefix}-{2 * index + 1:03d}", 1))
    return tuple(tasks)


def _diagnostics_from_result(
    prior: tuple[float, float],
    probe: ProbeSpecification,
    result: InformationValueResult,
) -> ProbeDiagnostics:
    return ProbeDiagnostics(
        action=probe.action,
        observation_entropy_nats=entropy(predictive_distribution(prior, probe.likelihoods)),
        expected_information_gain_nats=result.expected_information_gain_nats,
        expected_decision_value=result.expected_decision_value,
        acquisition_cost=result.acquisition_cost,
        net_value=result.net_value,
    )


def _best_terminal_value(posterior_rule_one: float) -> float:
    return max(1.0 - posterior_rule_one, posterior_rule_one, 0.65)


def _terminal_utility(posterior_rule_one: float, true_rule: int) -> float:
    values = (1.0 - posterior_rule_one, posterior_rule_one, 0.65)
    selected = max(range(len(values)), key=values.__getitem__)
    if selected == 2:
        return 0.65
    return float(selected == true_rule)


def _rule_target(task_id: str) -> EpistemicTarget:
    return EpistemicTarget(
        target_id=f"binary-rule:{task_id}",
        description=f"persistent XOR rule for task {task_id}",
        target_kind="latent_law",
    )


def _belief_probabilities(belief: Belief) -> tuple[float, float]:
    return _distribution_probabilities(belief.distribution)


def _belief_rule_one(belief: Belief) -> float:
    return _belief_probabilities(belief)[1]


def _distribution_probabilities(distribution: Distribution) -> tuple[float, float]:
    parameters = distribution.parameters
    if (
        not isinstance(parameters, tuple)
        or len(parameters) != 2
        or not all(isinstance(value, float) for value in parameters)
    ):
        raise TypeError("reference categorical distribution parameters are not a float pair")
    first, second = cast(tuple[float, float], parameters)
    if not isclose(first + second, 1.0, rel_tol=0.0, abs_tol=_TOL):
        raise ValueError("reference categorical probabilities are not normalized")
    return first, second


def _required_decision(experience: ExperienceEvent) -> DecisionRecord:
    if experience.decision is None:
        raise ValueError("reference interaction is missing its decision")
    return experience.decision


def _experience_signal(experience: ExperienceEvent) -> int:
    payload = _mapping(experience.observation.evidence.payload, "observation payload")
    signal = payload.get("signal")
    if not isinstance(signal, int) or isinstance(signal, bool) or signal not in (0, 1):
        raise ValueError("reference observation signal must be 0 or 1")
    return signal


def _serialize_observation(observation: Observation) -> dict[str, object]:
    payload = _mapping(observation.evidence.payload, "observation payload")
    return {
        "observation_id": observation.observation_id,
        "payload": {
            "task_id": _string(payload, "task_id"),
            "probe_kind": _string(payload, "probe_kind"),
            "signal": _binary_int(payload, "signal"),
            "accuracy": _probability(payload, "accuracy"),
        },
        "occurred_at": observation.evidence.occurred_at.tick,
        "available_at": observation.evidence.available_at.tick,
        "origin": observation.evidence.lineage.origin.value,
        "source_id": observation.evidence.lineage.provenance.source_id,
        "source_kind": observation.evidence.lineage.provenance.source_kind,
        "trust": int(observation.evidence.lineage.provenance.trust),
        "parent_evidence_ids": list(observation.evidence.lineage.parent_evidence_ids),
        "producer_version": observation.evidence.lineage.producer_version,
    }


def _deserialize_observation(raw: object, agent_id: str) -> Observation:
    item = _mapping(raw, "observation")
    observation_id = _string(item, "observation_id")
    raw_payload = _mapping(item.get("payload"), "observation payload")
    payload: dict[str, object] = {
        "task_id": _string(raw_payload, "task_id"),
        "probe_kind": _string(raw_payload, "probe_kind"),
        "signal": _binary_int(raw_payload, "signal"),
        "accuracy": _probability(raw_payload, "accuracy"),
    }
    raw_producer = item.get("producer_version")
    if raw_producer is not None and not isinstance(raw_producer, str):
        raise ValueError("producer_version must be a string or null")
    producer_version = raw_producer
    raw_origin = _string(item, "origin")
    try:
        origin = EvidenceOrigin(raw_origin)
    except ValueError as error:
        raise ValueError("unknown observation origin") from error
    raw_trust = _nonnegative_int(item, "trust")
    try:
        trust = TrustLevel(raw_trust)
    except ValueError as error:
        raise ValueError("unknown trust level") from error
    parents = tuple(_string_list(item, "parent_evidence_ids"))
    return Observation(
        observation_id=observation_id,
        agent_id=agent_id,
        modality="diagnostic_signal",
        evidence=Evidence(
            evidence_id=observation_id,
            payload=payload,
            occurred_at=TimePoint(_nonnegative_int(item, "occurred_at")),
            available_at=TimePoint(_nonnegative_int(item, "available_at")),
            lineage=EvidenceLineage(
                evidence_id=observation_id,
                origin=origin,
                provenance=Provenance(
                    source_id=_string(item, "source_id"),
                    trust=trust,
                    source_kind=_string(item, "source_kind"),
                ),
                parent_evidence_ids=parents,
                producer_version=producer_version,
            ),
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return cast(Sequence[object], value)


def _string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _string_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = _sequence(mapping, key)
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{key} must contain nonempty strings")
    result = cast(tuple[str, ...], tuple(values))
    if len(result) != len(set(result)):
        raise ValueError(f"{key} must not contain duplicates")
    return result


def _nonnegative_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _binary_int(mapping: Mapping[str, object], key: str) -> int:
    value = _nonnegative_int(mapping, key)
    if value not in (0, 1):
        raise ValueError(f"{key} must be 0 or 1")
    return value


def _nonnegative_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return result


def _probability(mapping: Mapping[str, object], key: str) -> float:
    result = _nonnegative_float(mapping, key)
    if result > 1.0:
        raise ValueError(f"{key} must be a probability")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


__all__ = (
    "BinaryRuleTask",
    "CollectionPolicy",
    "ExactRuleAgent",
    "IRRELEVANT_PROBE",
    "LearningStep",
    "NOISE_PROBE",
    "PROBES",
    "ProbeAction",
    "ProbeDiagnostics",
    "ProbeSpecification",
    "RELEVANT_PROBE",
    "diagnose_probe",
    "empirically_exact_training_signals",
    "evaluate_frozen",
    "evaluation_metric",
    "expected_external_utility",
    "make_balanced_task_suite",
    "posterior_after_signal",
    "probe_specification",
    "select_probe_by_raw_entropy",
    "select_probe_by_voi",
)
