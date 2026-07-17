"""Backend-neutral records for Prospect's structured epistemic transition.

The domain layer describes identities, temporal ordering, evidence lineage, beliefs,
predictions, decisions, experience, and learning receipts.  It deliberately contains
no tensor, optimizer, replay, planner, or distribution-family implementation.

All records are frozen and slotted.  Opaque payloads are owned by their producing
backend; domain records link them without interpreting or mutating them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isclose, isfinite


class DomainInvariantError(ValueError):
    """A linked domain record is internally inconsistent."""


def _require_id(name: str, value: str) -> None:
    if not value or not value.strip():
        raise DomainInvariantError(f"{name} must be a nonempty identifier")


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise DomainInvariantError(f"{name} must be nonempty")


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise DomainInvariantError(f"{name} must be finite")


def _same_clock(label: str, *points: TimePoint) -> None:
    clocks = {point.clock_id for point in points}
    if len(clocks) > 1:
        raise DomainInvariantError(f"{label} uses different clocks: {sorted(clocks)}")


def _not_before(label: str, later: TimePoint, earlier: TimePoint) -> None:
    _same_clock(label, later, earlier)
    if later.tick < earlier.tick:
        raise DomainInvariantError(f"{label} violates temporal order: {later.tick} is before {earlier.tick}")


def _strictly_after(label: str, later: TimePoint, earlier: TimePoint) -> None:
    _same_clock(label, later, earlier)
    if later.tick <= earlier.tick:
        raise DomainInvariantError(f"{label} must be strictly later: {later.tick} is not after {earlier.tick}")


def _unique_ids(label: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise DomainInvariantError(f"{label} must not contain duplicate identifiers")


@dataclass(frozen=True, slots=True)
class TimePoint:
    """A logical instant on a named clock.

    ``tick`` supplies the causal ordering used by domain invariants.  A backend may
    separately retain wall-clock or simulator time in an opaque payload.
    """

    tick: int
    clock_id: str = "interaction"

    def __post_init__(self) -> None:
        _require_id("clock_id", self.clock_id)
        if self.tick < 0:
            raise DomainInvariantError("time tick must be nonnegative")


class TrustLevel(IntEnum):
    """Declared trust in a source, not a probability that its content is true."""

    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


@dataclass(frozen=True, slots=True)
class Provenance:
    """Who supplied evidence and under which declared trust and custody context."""

    source_id: str
    trust: TrustLevel = TrustLevel.UNTRUSTED
    source_kind: str = "unspecified"
    detail: str = ""

    def __post_init__(self) -> None:
        _require_id("source_id", self.source_id)
        _require_text("source_kind", self.source_kind)


class EvidenceOrigin(StrEnum):
    """How an evidence payload entered the agent's information boundary."""

    OBSERVED = "observed"
    RETRIEVED = "retrieved"
    COMPUTED = "computed"
    REPORTED = "reported"
    DERIVED = "derived"
    IMAGINED = "imagined"


@dataclass(frozen=True, slots=True)
class EvidenceLineage:
    """Immutable lineage for one evidence item."""

    evidence_id: str
    origin: EvidenceOrigin
    provenance: Provenance
    parent_evidence_ids: tuple[str, ...] = ()
    producer_version: str | None = None

    def __post_init__(self) -> None:
        _require_id("evidence_id", self.evidence_id)
        for parent_id in self.parent_evidence_ids:
            _require_id("parent_evidence_id", parent_id)
        _unique_ids("parent_evidence_ids", self.parent_evidence_ids)
        if self.evidence_id in self.parent_evidence_ids:
            raise DomainInvariantError("evidence cannot be its own lineage parent")
        if self.origin in {EvidenceOrigin.DERIVED, EvidenceOrigin.COMPUTED, EvidenceOrigin.IMAGINED}:
            if self.producer_version is None:
                raise DomainInvariantError(f"{self.origin.value} evidence requires a producer_version")
        if self.producer_version is not None:
            _require_id("producer_version", self.producer_version)


@dataclass(frozen=True, slots=True)
class Evidence:
    """A payload with causal availability and explicit lineage."""

    evidence_id: str
    payload: object
    occurred_at: TimePoint
    available_at: TimePoint
    lineage: EvidenceLineage

    def __post_init__(self) -> None:
        _require_id("evidence_id", self.evidence_id)
        if self.lineage.evidence_id != self.evidence_id:
            raise DomainInvariantError("evidence_id does not match its lineage")
        _not_before("evidence availability", self.available_at, self.occurred_at)


@dataclass(frozen=True, slots=True)
class Observation:
    """A real information event delivered to an agent.

    Imagined evidence is intentionally not an observation.  It may support planning,
    but cannot silently enter experience as if the environment supplied it.
    """

    observation_id: str
    agent_id: str
    modality: str
    evidence: Evidence

    def __post_init__(self) -> None:
        _require_id("observation_id", self.observation_id)
        _require_id("agent_id", self.agent_id)
        _require_text("modality", self.modality)
        if self.observation_id != self.evidence.evidence_id:
            raise DomainInvariantError("observation_id does not match its evidence")
        if self.evidence.lineage.origin is EvidenceOrigin.IMAGINED:
            raise DomainInvariantError("imagined evidence cannot be recorded as an observation")


@dataclass(frozen=True, slots=True)
class InformationSet:
    """The evidence available to one agent at a declared causal cutoff."""

    information_set_id: str
    agent_id: str
    as_of: TimePoint
    observations: tuple[Observation, ...] = ()
    memory_version: str = "none"

    def __post_init__(self) -> None:
        _require_id("information_set_id", self.information_set_id)
        _require_id("agent_id", self.agent_id)
        _require_id("memory_version", self.memory_version)
        ids = tuple(observation.observation_id for observation in self.observations)
        _unique_ids("information-set observations", ids)
        for observation in self.observations:
            if observation.agent_id != self.agent_id:
                raise DomainInvariantError("information set contains another agent's observation")
            _not_before(
                "information-set cutoff",
                self.as_of,
                observation.evidence.available_at,
            )


@dataclass(frozen=True, slots=True)
class EpistemicTarget:
    """A named object, proposition, state, law, or future outcome being represented."""

    target_id: str
    description: str
    target_kind: str = "unspecified"

    def __post_init__(self) -> None:
        _require_id("target_id", self.target_id)
        _require_text("description", self.description)
        _require_text("target_kind", self.target_kind)


@dataclass(frozen=True, slots=True)
class Distribution:
    """An opaque probability or uncertainty representation supplied by a backend."""

    distribution_id: str
    family: str
    support: str
    parameters: object
    representation_version: str
    event_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_id("distribution_id", self.distribution_id)
        _require_text("family", self.family)
        _require_text("support", self.support)
        _require_id("representation_version", self.representation_version)
        if any(size < 0 for size in self.event_shape):
            raise DomainInvariantError("distribution event_shape must be nonnegative")


@dataclass(frozen=True, slots=True)
class Belief:
    """An agent's stance about a target under a named information set."""

    belief_id: str
    agent_id: str
    target: EpistemicTarget
    information_set: InformationSet
    distribution: Distribution
    formed_at: TimePoint
    model_version: str
    representation_version: str

    def __post_init__(self) -> None:
        _require_id("belief_id", self.belief_id)
        _require_id("agent_id", self.agent_id)
        _require_id("model_version", self.model_version)
        _require_id("representation_version", self.representation_version)
        if self.information_set.agent_id != self.agent_id:
            raise DomainInvariantError("belief and information set have different agent ids")
        if self.distribution.representation_version != self.representation_version:
            raise DomainInvariantError("belief and distribution representation versions differ")
        _not_before("belief formation", self.formed_at, self.information_set.as_of)


@dataclass(frozen=True, slots=True)
class Action:
    """A typed action value independent of whether it was intended or executed."""

    action_id: str
    action_kind: str
    parameters: object

    def __post_init__(self) -> None:
        _require_id("action_id", self.action_id)
        _require_text("action_kind", self.action_kind)


class UncertaintyKind(StrEnum):
    """The source or decision role qualified by an uncertainty estimate."""

    PREDICTIVE = "predictive"
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    DATA = "data"
    SOURCE = "source"
    MODEL = "model"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    """One named uncertainty measurement, without a universal scalar alias."""

    estimate_id: str
    kind: UncertaintyKind
    measure: str
    value: float
    unit: str
    target_id: str
    estimator_version: str
    assessed_at: TimePoint
    calibration_version: str | None = None

    def __post_init__(self) -> None:
        _require_id("uncertainty estimate_id", self.estimate_id)
        _require_text("uncertainty measure", self.measure)
        _require_finite("uncertainty value", self.value)
        _require_text("uncertainty unit", self.unit)
        _require_id("uncertainty target_id", self.target_id)
        _require_id("uncertainty estimator_version", self.estimator_version)
        if self.calibration_version is not None:
            _require_id("uncertainty calibration_version", self.calibration_version)


@dataclass(frozen=True, slots=True)
class IntendedAction:
    """An action selected by an agent but not yet asserted to have occurred."""

    intention_id: str
    agent_id: str
    action: Action
    intended_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("intention_id", self.intention_id)
        _require_id("agent_id", self.agent_id)


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ExecutedAction:
    """The environment-side execution linked to an agent intention."""

    execution_id: str
    intention: IntendedAction
    status: ExecutionStatus
    started_at: TimePoint
    ended_at: TimePoint
    realized_action: Action | None = None
    deviation_reason: str = ""

    def __post_init__(self) -> None:
        _require_id("execution_id", self.execution_id)
        _not_before("action start", self.started_at, self.intention.intended_at)
        _not_before("action end", self.ended_at, self.started_at)
        if self.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.PARTIAL}:
            if self.realized_action is None:
                raise DomainInvariantError(f"{self.status.value} execution requires a realized_action")
        if self.status in {ExecutionStatus.REJECTED, ExecutionStatus.CANCELLED}:
            if self.realized_action is not None:
                raise DomainInvariantError(f"{self.status.value} execution cannot claim a realized_action")
        if (
            self.realized_action is not None
            and self.realized_action.action_id != self.intention.action.action_id
            and not self.deviation_reason.strip()
        ):
            raise DomainInvariantError("executed action differs from the intention without a deviation reason")


@dataclass(frozen=True, slots=True)
class Prediction:
    """An action-conditional prospective distribution under one prior belief."""

    prediction_id: str
    prior_belief: Belief
    action: Action
    target: EpistemicTarget
    distribution: Distribution
    issued_at: TimePoint
    horizon_end: TimePoint
    model_version: str
    representation_version: str
    calibration_version: str
    uncertainties: tuple[UncertaintyEstimate, ...] = ()

    def __post_init__(self) -> None:
        _require_id("prediction_id", self.prediction_id)
        _require_id("model_version", self.model_version)
        _require_id("representation_version", self.representation_version)
        _require_id("calibration_version", self.calibration_version)
        if self.model_version != self.prior_belief.model_version:
            raise DomainInvariantError("prediction and prior belief model versions differ")
        if self.representation_version != self.prior_belief.representation_version:
            raise DomainInvariantError("prediction and prior belief representation versions differ")
        if self.distribution.representation_version != self.representation_version:
            raise DomainInvariantError("prediction and distribution representation versions differ")
        uncertainty_ids = tuple(estimate.estimate_id for estimate in self.uncertainties)
        _unique_ids("prediction uncertainty estimate ids", uncertainty_ids)
        uncertainty_keys = tuple(
            (estimate.kind, estimate.measure, estimate.target_id) for estimate in self.uncertainties
        )
        if len(uncertainty_keys) != len(set(uncertainty_keys)):
            raise DomainInvariantError("prediction cannot contain duplicate uncertainty measurements")
        for estimate in self.uncertainties:
            if estimate.target_id != self.target.target_id:
                raise DomainInvariantError("prediction uncertainty refers to a different target")
            if estimate.calibration_version is not None and estimate.calibration_version != self.calibration_version:
                raise DomainInvariantError("prediction and uncertainty calibration versions differ")
            _not_before(
                "uncertainty assessment",
                estimate.assessed_at,
                self.prior_belief.formed_at,
            )
            _not_before(
                "prediction issuance",
                self.issued_at,
                estimate.assessed_at,
            )
        _not_before("prediction issuance", self.issued_at, self.prior_belief.formed_at)
        _strictly_after("prediction horizon", self.horizon_end, self.issued_at)


@dataclass(frozen=True, slots=True)
class Goal:
    """A desired target supplied independently of the predictive distribution."""

    goal_id: str
    task_id: str
    target: EpistemicTarget
    description: str
    issued_at: TimePoint
    preference_version: str
    deadline: TimePoint | None = None

    def __post_init__(self) -> None:
        _require_id("goal_id", self.goal_id)
        _require_id("task_id", self.task_id)
        _require_text("description", self.description)
        _require_id("preference_version", self.preference_version)
        if self.deadline is not None:
            _strictly_after("goal deadline", self.deadline, self.issued_at)


@dataclass(frozen=True, slots=True)
class Utility:
    """Expected external goal value assigned to one prediction."""

    utility_id: str
    goal_id: str
    prediction_id: str
    expected_value: float
    unit: str
    evaluator_version: str
    assessed_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("utility_id", self.utility_id)
        _require_id("goal_id", self.goal_id)
        _require_id("prediction_id", self.prediction_id)
        _require_text("unit", self.unit)
        _require_id("evaluator_version", self.evaluator_version)
        _require_finite("expected_value", self.expected_value)


@dataclass(frozen=True, slots=True)
class InformationValue:
    """Expected decision-relevant reducible-risk reduction for one action."""

    information_value_id: str
    prior_belief_id: str
    action_id: str
    target_id: str
    expected_reduction: float
    expected_cost: float
    unit: str
    evaluator_version: str
    assessed_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("information_value_id", self.information_value_id)
        _require_id("prior_belief_id", self.prior_belief_id)
        _require_id("action_id", self.action_id)
        _require_id("target_id", self.target_id)
        _require_text("unit", self.unit)
        _require_id("evaluator_version", self.evaluator_version)
        _require_finite("expected_reduction", self.expected_reduction)
        _require_finite("expected_cost", self.expected_cost)
        if self.expected_reduction < 0.0:
            raise DomainInvariantError("expected information reduction must be nonnegative")
        if self.expected_cost < 0.0:
            raise DomainInvariantError("expected information cost must be nonnegative")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """Auditable value decomposition for one candidate action.

    ``InformationValue.expected_cost`` is the cost of obtaining or processing the
    information itself.  ``expected_action_cost`` is the independently estimated
    cost of executing the action.  All additive terms share ``unit``.
    """

    assessment_id: str
    action: Action
    prediction: Prediction
    utility: Utility
    information_value: InformationValue
    expected_action_cost: float
    expected_risk: float
    admissible: bool
    constraint_reasons: tuple[str, ...]
    constraint_penalty: float
    total_value: float
    unit: str
    evaluator_version: str
    assessed_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("assessment_id", self.assessment_id)
        _require_text("candidate assessment unit", self.unit)
        _require_id("candidate evaluator_version", self.evaluator_version)
        for name, value in (
            ("expected_action_cost", self.expected_action_cost),
            ("expected_risk", self.expected_risk),
            ("constraint_penalty", self.constraint_penalty),
            ("total_value", self.total_value),
        ):
            _require_finite(name, value)
        if min(self.expected_action_cost, self.expected_risk, self.constraint_penalty) < 0.0:
            raise DomainInvariantError("candidate costs, risk, and penalties must be nonnegative")
        for reason in self.constraint_reasons:
            _require_text("constraint reason", reason)
        if len(self.constraint_reasons) != len(set(self.constraint_reasons)):
            raise DomainInvariantError("constraint reasons must be unique")
        if self.admissible and self.constraint_reasons:
            raise DomainInvariantError("admissible candidate cannot have hard-constraint reasons")
        if not self.admissible and not self.constraint_reasons:
            raise DomainInvariantError("inadmissible candidate requires a hard-constraint reason")
        if self.prediction.action.action_id != self.action.action_id:
            raise DomainInvariantError("candidate action and prediction action ids differ")
        if self.utility.prediction_id != self.prediction.prediction_id:
            raise DomainInvariantError("candidate utility refers to a different prediction")
        if self.information_value.prior_belief_id != self.prediction.prior_belief.belief_id:
            raise DomainInvariantError("candidate information value uses a different prior belief")
        if self.information_value.action_id != self.action.action_id:
            raise DomainInvariantError("candidate information value refers to a different action")
        if self.information_value.target_id != self.prediction.target.target_id:
            raise DomainInvariantError("candidate information value refers to a different target")
        if self.utility.unit != self.unit or self.information_value.unit != self.unit:
            raise DomainInvariantError("candidate value terms must use the assessment unit")
        _not_before(
            "candidate utility assessment",
            self.utility.assessed_at,
            self.prediction.issued_at,
        )
        _not_before(
            "candidate information assessment",
            self.information_value.assessed_at,
            self.prediction.prior_belief.formed_at,
        )
        _not_before("candidate assessment", self.assessed_at, self.prediction.issued_at)
        _not_before("candidate assessment", self.assessed_at, self.utility.assessed_at)
        _not_before(
            "candidate assessment",
            self.assessed_at,
            self.information_value.assessed_at,
        )
        expected_total = (
            self.utility.expected_value
            + self.information_value.expected_reduction
            - self.information_value.expected_cost
            - self.expected_action_cost
            - self.expected_risk
            - self.constraint_penalty
        )
        if not isclose(self.total_value, expected_total, rel_tol=1e-9, abs_tol=1e-12):
            raise DomainInvariantError(
                "candidate total_value disagrees with utility, information, cost, risk, and penalty"
            )


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Why one intended action was selected from one pre-outcome information state."""

    decision_id: str
    agent_id: str
    belief: Belief
    goal: Goal
    intended_action: IntendedAction
    alternatives: tuple[CandidateAssessment, ...]
    selected_assessment: CandidateAssessment
    policy_version: str
    decided_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("decision_id", self.decision_id)
        _require_id("agent_id", self.agent_id)
        _require_id("policy_version", self.policy_version)
        if self.belief.agent_id != self.agent_id:
            raise DomainInvariantError("decision and belief have different agent ids")
        if self.intended_action.agent_id != self.agent_id:
            raise DomainInvariantError("decision and intention have different agent ids")
        if not self.alternatives:
            raise DomainInvariantError("decision requires at least one candidate assessment")
        assessment_ids = tuple(assessment.assessment_id for assessment in self.alternatives)
        action_ids = tuple(assessment.action.action_id for assessment in self.alternatives)
        prediction_ids = tuple(assessment.prediction.prediction_id for assessment in self.alternatives)
        _unique_ids("decision candidate assessment ids", assessment_ids)
        _unique_ids("decision candidate action ids", action_ids)
        _unique_ids("decision candidate prediction ids", prediction_ids)
        selected_matches = [
            assessment
            for assessment in self.alternatives
            if assessment.assessment_id == self.selected_assessment.assessment_id
        ]
        if len(selected_matches) != 1:
            raise DomainInvariantError("selected assessment is not linked to the alternatives")
        linked_selected = selected_matches[0]
        selected_links = (
            linked_selected.action.action_id,
            linked_selected.prediction.prediction_id,
            linked_selected.utility.utility_id,
            linked_selected.information_value.information_value_id,
        )
        supplied_links = (
            self.selected_assessment.action.action_id,
            self.selected_assessment.prediction.prediction_id,
            self.selected_assessment.utility.utility_id,
            self.selected_assessment.information_value.information_value_id,
        )
        if supplied_links != selected_links or not isclose(
            self.selected_assessment.total_value,
            linked_selected.total_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise DomainInvariantError("selected assessment content differs from its linked alternative")
        if (
            self.selected_assessment.admissible != linked_selected.admissible
            or self.selected_assessment.constraint_reasons != linked_selected.constraint_reasons
        ):
            raise DomainInvariantError("selected assessment constraints differ from its linked alternative")
        if not self.selected_assessment.admissible:
            raise DomainInvariantError("selected assessment must be admissible")
        for assessment in self.alternatives:
            if assessment.prediction.prior_belief.belief_id != self.belief.belief_id:
                raise DomainInvariantError("candidate prediction does not use the decision belief")
            if assessment.prediction.target.target_id != self.goal.target.target_id:
                raise DomainInvariantError("candidate prediction and goal targets differ")
            if assessment.utility.goal_id != self.goal.goal_id:
                raise DomainInvariantError("candidate utility refers to a different goal")
            _not_before("decision time", self.decided_at, assessment.assessed_at)
        if self.selected_assessment.action.action_id != self.intended_action.action.action_id:
            raise DomainInvariantError("selected assessment and intended action ids differ")
        _not_before("decision time", self.decided_at, self.belief.formed_at)
        _not_before("decision time", self.decided_at, self.goal.issued_at)
        _not_before("decision time", self.decided_at, self.intended_action.intended_at)


@dataclass(frozen=True, slots=True)
class Outcome:
    """An externally evidenced consequence, distinct from its utility."""

    outcome_id: str
    evidence: Evidence
    execution_id: str | None = None

    def __post_init__(self) -> None:
        _require_id("outcome_id", self.outcome_id)
        if self.evidence.lineage.origin is EvidenceOrigin.IMAGINED:
            raise DomainInvariantError("imagined evidence cannot be recorded as an outcome")
        if self.execution_id is not None:
            _require_id("execution_id", self.execution_id)


class ExperienceKind(StrEnum):
    INTERACTION = "interaction"
    PASSIVE = "passive"


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    """A real, closed event from which learner-specific views may be derived."""

    experience_id: str
    agent_id: str
    run_id: str
    task_id: str
    episode_id: str
    step_index: int
    kind: ExperienceKind
    observation: Observation
    outcome: Outcome
    terminated: bool
    truncated: bool
    discount: float
    behavior_policy_version: str
    closed_at: TimePoint
    decision: DecisionRecord | None = None
    execution: ExecutedAction | None = None

    def __post_init__(self) -> None:
        _require_id("experience_id", self.experience_id)
        _require_id("agent_id", self.agent_id)
        _require_id("run_id", self.run_id)
        _require_id("task_id", self.task_id)
        _require_id("episode_id", self.episode_id)
        _require_id("behavior_policy_version", self.behavior_policy_version)
        if self.step_index < 0:
            raise DomainInvariantError("experience step_index must be nonnegative")
        if self.terminated and self.truncated:
            raise DomainInvariantError("experience cannot be both terminated and truncated")
        _require_finite("experience discount", self.discount)
        if self.discount < 0.0:
            raise DomainInvariantError("experience discount must be nonnegative")
        if self.observation.agent_id != self.agent_id:
            raise DomainInvariantError("experience and observation have different agent ids")
        if self.kind is ExperienceKind.INTERACTION:
            if self.decision is None or self.execution is None:
                raise DomainInvariantError("interaction experience requires both decision and execution")
        elif self.decision is not None or self.execution is not None:
            raise DomainInvariantError("passive experience cannot claim a decision or executed action")
        if self.decision is not None and self.execution is not None:
            if self.decision.agent_id != self.agent_id:
                raise DomainInvariantError("experience and decision have different agent ids")
            if self.decision.goal.task_id != self.task_id:
                raise DomainInvariantError("experience task does not match its decision goal")
            if self.decision.policy_version != self.behavior_policy_version:
                raise DomainInvariantError("experience behavior policy does not match its decision")
            if self.execution.intention.intention_id != self.decision.intended_action.intention_id:
                raise DomainInvariantError("experience execution does not match its decision")
            if self.outcome.execution_id != self.execution.execution_id:
                raise DomainInvariantError("outcome does not match the executed action")
            _not_before(
                "experience execution",
                self.execution.started_at,
                self.decision.decided_at,
            )
            _not_before(
                "experience outcome",
                self.outcome.evidence.occurred_at,
                self.execution.started_at,
            )
            _not_before(
                "experience observation",
                self.observation.evidence.occurred_at,
                self.execution.started_at,
            )
            _not_before("experience close", self.closed_at, self.execution.ended_at)
        elif self.outcome.execution_id is not None:
            raise DomainInvariantError("passive experience outcome cannot reference an execution")
        _not_before(
            "experience close",
            self.closed_at,
            self.observation.evidence.available_at,
        )
        _not_before(
            "experience close",
            self.closed_at,
            self.outcome.evidence.available_at,
        )


@dataclass(frozen=True, slots=True)
class BeliefUpdate:
    """Assimilation of one real experience into a linked posterior belief."""

    update_id: str
    prior: Belief
    experience: ExperienceEvent
    posterior: Belief
    updater_version: str
    updated_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("update_id", self.update_id)
        _require_id("updater_version", self.updater_version)
        if self.prior.agent_id != self.experience.agent_id:
            raise DomainInvariantError("belief update prior belongs to a different agent")
        if self.posterior.agent_id != self.experience.agent_id:
            raise DomainInvariantError("belief update posterior belongs to a different agent")
        if self.experience.decision is not None:
            if self.experience.decision.belief.belief_id != self.prior.belief_id:
                raise DomainInvariantError("belief update prior does not match the decision belief")
        if self.posterior.belief_id == self.prior.belief_id:
            raise DomainInvariantError("prior and posterior beliefs require different ids")
        if self.posterior.target.target_id != self.prior.target.target_id:
            raise DomainInvariantError("belief update changed epistemic target")
        if self.posterior.model_version != self.prior.model_version:
            raise DomainInvariantError("belief assimilation changed model version")
        if self.posterior.representation_version != self.prior.representation_version:
            raise DomainInvariantError("belief assimilation changed representation version")
        prior_evidence = {observation.observation_id for observation in self.prior.information_set.observations}
        posterior_evidence = {observation.observation_id for observation in self.posterior.information_set.observations}
        if not prior_evidence.issubset(posterior_evidence):
            raise DomainInvariantError("posterior information set dropped prior evidence")
        if self.experience.observation.observation_id not in posterior_evidence:
            raise DomainInvariantError("posterior information set omits the new observation")
        _not_before(
            "posterior formation",
            self.posterior.formed_at,
            self.experience.closed_at,
        )
        _not_before("belief update", self.updated_at, self.posterior.formed_at)


@dataclass(frozen=True, slots=True)
class ProperScore:
    """A named proper-score result for one prediction and realized evidence item."""

    score_id: str
    prediction_id: str
    realized_evidence_id: str
    rule: str
    value: float
    unit: str
    scorer_version: str
    scored_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("score_id", self.score_id)
        _require_id("prediction_id", self.prediction_id)
        _require_id("realized_evidence_id", self.realized_evidence_id)
        _require_text("rule", self.rule)
        _require_text("unit", self.unit)
        _require_id("scorer_version", self.scorer_version)
        _require_finite("proper score", self.value)


class EpistemicEffectKind(StrEnum):
    INFORMATION_GAIN = "information_gain"
    CALIBRATION_CHANGE = "calibration_change"
    PREDICTIVE_RISK_CHANGE = "predictive_risk_change"
    KNOWLEDGE_POTENTIAL_CHANGE = "knowledge_potential_change"


@dataclass(frozen=True, slots=True)
class EpistemicEffect:
    """One explicitly named projection of a belief update."""

    effect_id: str
    belief_update_id: str
    target_id: str
    kind: EpistemicEffectKind
    measure: str
    before: float
    after: float
    improvement: float
    higher_is_better: bool
    evaluator_version: str
    evaluated_at: TimePoint
    externally_calibrated: bool = False

    def __post_init__(self) -> None:
        _require_id("effect_id", self.effect_id)
        _require_id("belief_update_id", self.belief_update_id)
        _require_id("target_id", self.target_id)
        _require_text("measure", self.measure)
        _require_id("evaluator_version", self.evaluator_version)
        _require_finite("effect before", self.before)
        _require_finite("effect after", self.after)
        _require_finite("effect improvement", self.improvement)
        expected = self.after - self.before if self.higher_is_better else self.before - self.after
        if not isclose(self.improvement, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise DomainInvariantError("epistemic effect improvement disagrees with before/after direction")


@dataclass(frozen=True, slots=True)
class EpistemicTransition:
    """Aggregate linking prediction, reality, belief revision, and typed effects."""

    transition_id: str
    experience: ExperienceEvent
    belief_update: BeliefUpdate
    proper_scores: tuple[ProperScore, ...]
    effects: tuple[EpistemicEffect, ...]
    created_at: TimePoint

    def __post_init__(self) -> None:
        _require_id("transition_id", self.transition_id)
        if self.experience.kind is not ExperienceKind.INTERACTION:
            raise DomainInvariantError(
                "an epistemic transition with an action prediction requires interaction experience"
            )
        if self.experience.decision is None:
            raise DomainInvariantError("epistemic transition is missing its decision")
        if self.belief_update.experience.experience_id != self.experience.experience_id:
            raise DomainInvariantError("epistemic transition links a different experience")
        prediction = self.experience.decision.selected_assessment.prediction
        realized_ids = {
            self.experience.observation.evidence.evidence_id,
            self.experience.outcome.evidence.evidence_id,
        }
        score_ids = tuple(score.score_id for score in self.proper_scores)
        effect_ids = tuple(effect.effect_id for effect in self.effects)
        _unique_ids("proper score ids", score_ids)
        _unique_ids("epistemic effect ids", effect_ids)
        for score in self.proper_scores:
            if score.prediction_id != prediction.prediction_id:
                raise DomainInvariantError("proper score refers to a different prediction")
            if score.realized_evidence_id not in realized_ids:
                raise DomainInvariantError("proper score refers to unrelated realized evidence")
            _not_before(
                "proper scoring",
                score.scored_at,
                self.experience.outcome.evidence.available_at,
            )
            _not_before("transition creation", self.created_at, score.scored_at)
        for effect in self.effects:
            if effect.belief_update_id != self.belief_update.update_id:
                raise DomainInvariantError("epistemic effect refers to a different belief update")
            if effect.target_id != self.belief_update.prior.target.target_id:
                raise DomainInvariantError("epistemic effect refers to a different target")
            _not_before("epistemic effect", effect.evaluated_at, self.belief_update.updated_at)
            _not_before("transition creation", self.created_at, effect.evaluated_at)
        _not_before(
            "transition creation",
            self.created_at,
            self.belief_update.updated_at,
        )


class UpdateStatus(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class UpdateReceipt:
    """Auditable persistent-configuration update caused by completed transitions."""

    receipt_id: str
    agent_id: str
    transitions: tuple[EpistemicTransition, ...]
    learner_version: str
    status: UpdateStatus
    previous_configuration_version: str
    new_configuration_version: str
    previous_model_version: str
    new_model_version: str
    previous_representation_version: str
    new_representation_version: str
    previous_policy_version: str
    new_policy_version: str
    started_at: TimePoint
    completed_at: TimePoint
    resulting_belief: Belief | None = None
    rollback_of: str | None = None
    metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _require_id("receipt_id", self.receipt_id)
        _require_id("agent_id", self.agent_id)
        _require_id("learner_version", self.learner_version)
        version_values = (
            self.previous_configuration_version,
            self.new_configuration_version,
            self.previous_model_version,
            self.new_model_version,
            self.previous_representation_version,
            self.new_representation_version,
            self.previous_policy_version,
            self.new_policy_version,
        )
        for version in version_values:
            _require_id("update version", version)
        _not_before("learning update", self.completed_at, self.started_at)
        transition_ids = tuple(transition.transition_id for transition in self.transitions)
        _unique_ids("update transitions", transition_ids)
        for transition in self.transitions:
            if transition.experience.agent_id != self.agent_id:
                raise DomainInvariantError("update consumes another agent's transition")
            _not_before("learning start", self.started_at, transition.created_at)
        model_changed = self.previous_model_version != self.new_model_version
        representation_changed = self.previous_representation_version != self.new_representation_version
        changes = (
            self.previous_configuration_version != self.new_configuration_version,
            model_changed,
            representation_changed,
            self.previous_policy_version != self.new_policy_version,
        )
        if self.status is UpdateStatus.APPLIED and not any(changes):
            raise DomainInvariantError("applied update must change at least one version")
        if self.status is UpdateStatus.REJECTED and any(changes):
            raise DomainInvariantError("rejected update cannot change persistent versions")
        if (
            self.status is UpdateStatus.APPLIED
            and (model_changed or representation_changed)
            and self.resulting_belief is None
        ):
            raise DomainInvariantError("applied model or representation update requires a resulting_belief")
        if self.resulting_belief is not None and self.status is not UpdateStatus.APPLIED:
            raise DomainInvariantError("only an applied update can produce a resulting_belief")
        if self.resulting_belief is not None:
            if not self.transitions:
                raise DomainInvariantError("resulting_belief requires at least one source transition")
            source_belief = self.transitions[-1].belief_update.posterior
            resulting = self.resulting_belief
            if resulting.agent_id != self.agent_id:
                raise DomainInvariantError("resulting belief belongs to another agent")
            if resulting.target.target_id != source_belief.target.target_id:
                raise DomainInvariantError("resulting belief changed epistemic target")
            if resulting.belief_id == source_belief.belief_id:
                raise DomainInvariantError("resulting belief requires a new identity after learning")
            if resulting.model_version != self.new_model_version:
                raise DomainInvariantError("resulting belief does not use the new model version")
            if resulting.representation_version != self.new_representation_version:
                raise DomainInvariantError("resulting belief does not use the new representation version")
            source_evidence = {observation.observation_id for observation in source_belief.information_set.observations}
            resulting_evidence = {observation.observation_id for observation in resulting.information_set.observations}
            if not source_evidence.issubset(resulting_evidence):
                raise DomainInvariantError("resulting belief lost source-belief information")
            _not_before(
                "resulting belief information",
                resulting.information_set.as_of,
                source_belief.information_set.as_of,
            )
            _not_before(
                "resulting belief formation",
                resulting.formed_at,
                self.completed_at,
            )
        if self.status is UpdateStatus.ROLLED_BACK:
            if self.rollback_of is None:
                raise DomainInvariantError("rolled-back update requires rollback_of")
        elif self.rollback_of is not None:
            raise DomainInvariantError("rollback_of is valid only for a rolled-back update")
        names = tuple(name for name, _ in self.metrics)
        _unique_ids("update metric names", names)
        for name, value in self.metrics:
            _require_text("update metric name", name)
            _require_finite(f"update metric {name}", value)


@dataclass(frozen=True, slots=True)
class ResourceUse:
    """One nonnegative resource quantity in a declared unit."""

    resource: str
    amount: float
    unit: str

    def __post_init__(self) -> None:
        _require_text("resource", self.resource)
        _require_text("resource unit", self.unit)
        _require_finite("resource amount", self.amount)
        if self.amount < 0.0:
            raise DomainInvariantError("resource amount must be nonnegative")


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    """Immutable resource accounting over one causal interval."""

    ledger_id: str
    started_at: TimePoint
    completed_at: TimePoint
    uses: tuple[ResourceUse, ...] = ()

    def __post_init__(self) -> None:
        _require_id("ledger_id", self.ledger_id)
        _not_before("resource ledger", self.completed_at, self.started_at)
        keys = tuple((use.resource, use.unit) for use in self.uses)
        if len(keys) != len(set(keys)):
            raise DomainInvariantError("resource ledger contains duplicate resource/unit entries")


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """A coherent immutable view of the complete persistent agent state."""

    snapshot_id: str
    agent_id: str
    captured_at: TimePoint
    belief: Belief
    configuration_version: str
    memory_version: str
    knowledge_version: str
    model_version: str
    representation_version: str
    policy_version: str
    resources: ResourceLedger
    pending_intentions: tuple[IntendedAction, ...] = ()
    latest_update: UpdateReceipt | None = None

    def __post_init__(self) -> None:
        _require_id("snapshot_id", self.snapshot_id)
        _require_id("agent_id", self.agent_id)
        for version in (
            self.configuration_version,
            self.memory_version,
            self.knowledge_version,
            self.model_version,
            self.representation_version,
            self.policy_version,
        ):
            _require_id("snapshot version", version)
        if self.belief.agent_id != self.agent_id:
            raise DomainInvariantError("snapshot belief belongs to another agent")
        if self.belief.model_version != self.model_version:
            raise DomainInvariantError("snapshot and belief model versions differ")
        if self.belief.representation_version != self.representation_version:
            raise DomainInvariantError("snapshot and belief representation versions differ")
        _not_before("snapshot capture", self.captured_at, self.belief.formed_at)
        _not_before("snapshot resources", self.captured_at, self.resources.completed_at)
        intention_ids = tuple(intention.intention_id for intention in self.pending_intentions)
        _unique_ids("pending intention ids", intention_ids)
        for intention in self.pending_intentions:
            if intention.agent_id != self.agent_id:
                raise DomainInvariantError("snapshot contains another agent's intention")
            _not_before("snapshot capture", self.captured_at, intention.intended_at)
        if self.latest_update is not None:
            receipt = self.latest_update
            if receipt.agent_id != self.agent_id:
                raise DomainInvariantError("snapshot update belongs to another agent")
            _not_before("snapshot capture", self.captured_at, receipt.completed_at)
            expected_versions = (
                (self.configuration_version, receipt.new_configuration_version),
                (self.model_version, receipt.new_model_version),
                (self.representation_version, receipt.new_representation_version),
                (self.policy_version, receipt.new_policy_version),
            )
            if any(snapshot != receipt for snapshot, receipt in expected_versions):
                raise DomainInvariantError("snapshot versions do not match its latest update")
            if receipt.resulting_belief is not None and self.belief.belief_id != receipt.resulting_belief.belief_id:
                raise DomainInvariantError("snapshot belief does not match the update's resulting belief")


@dataclass(frozen=True, slots=True)
class EvaluationMetric:
    """One externally computed evaluation quantity."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        _require_text("evaluation metric name", self.name)
        _require_text("evaluation metric unit", self.unit)
        _require_finite(f"evaluation metric {self.name}", self.value)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Externally scored behavior under an explicit resource and update policy."""

    evaluation_id: str
    agent_id: str
    task_id: str
    evaluator_version: str
    snapshot: AgentSnapshot
    started_at: TimePoint
    completed_at: TimePoint
    metrics: tuple[EvaluationMetric, ...]
    resources: ResourceLedger
    transition_ids: tuple[str, ...] = ()
    training_updates_allowed: bool = False
    update_receipts: tuple[UpdateReceipt, ...] = ()

    def __post_init__(self) -> None:
        _require_id("evaluation_id", self.evaluation_id)
        _require_id("agent_id", self.agent_id)
        _require_id("task_id", self.task_id)
        _require_id("evaluator_version", self.evaluator_version)
        if self.snapshot.agent_id != self.agent_id:
            raise DomainInvariantError("evaluation snapshot belongs to another agent")
        _not_before("evaluation start", self.started_at, self.snapshot.captured_at)
        _not_before("evaluation completion", self.completed_at, self.started_at)
        _not_before("evaluation resources", self.resources.started_at, self.started_at)
        _not_before(
            "evaluation completion",
            self.completed_at,
            self.resources.completed_at,
        )
        metric_names = tuple(metric.name for metric in self.metrics)
        _unique_ids("evaluation metric names", metric_names)
        _unique_ids("evaluation transition ids", self.transition_ids)
        if not self.training_updates_allowed and self.update_receipts:
            raise DomainInvariantError("evaluation forbids training updates but includes update receipts")
        for receipt in self.update_receipts:
            if receipt.agent_id != self.agent_id:
                raise DomainInvariantError("evaluation includes another agent's update")
            _not_before("evaluation update", receipt.started_at, self.started_at)
            _not_before(
                "evaluation completion",
                self.completed_at,
                receipt.completed_at,
            )
