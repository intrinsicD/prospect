"""Authoritative Prospect runtime components for the WM-002 Q1 fixture.

This module implements the runtime lane only.  It does not schedule Q1 episodes,
derive private seeds, checkpoint state, write qualification artifacts, or authorize
an experiment.  Every environment interaction must still be initiated by a
separately qualified Q1 producer.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Final, cast

from bench.active_acquisition.oracle import FractionOracle
from bench.active_acquisition.policies import SHUFFLED_INFORMATION_SOURCE_BY_ACTION
from bench.active_acquisition.problem import (
    ACQUISITION_ACTIONS,
    TERMINAL_ACTIONS,
    AcquisitionAction,
    AcquisitionDiagnostics,
    HiddenActuatorProblem,
    TerminalAction,
)
from prospect.decision import CounterIdentitySource, DecisionError, MaxValuePolicy
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
from prospect.epistemics.assessments import categorical_probabilities
from prospect.epistemics.information import entropy
from prospect.epistemics.scoring import categorical_log_score
from prospect.runtime import (
    AgentState,
    EnvironmentStep,
    EpistemicAgent,
    ModelState,
    PreparedLearningUpdate,
    VersionedModelOwner,
)
from prospect.storage import EpistemicLedger, InMemoryExperienceStore

POSTERIOR_MODEL_SCHEMA: Final = "prospect.wm002.posterior-model.v1"
LIKELIHOOD_VERSION: Final = "wm002-hidden-actuator-true-v1"
MODEL_VERSION_PREFIX: Final = "wm002-model-sha256:"
CONFIGURATION_VERSION_PREFIX: Final = "wm002-config-sha256:"
REPRESENTATION_VERSION: Final = "wm002-categorical-regime-v1"
CALIBRATION_VERSION: Final = "wm002-known-likelihood-v1"
LEARNER_VERSION: Final = "wm002-exact-posterior-transactional-learner-v1"
ASSIMILATOR_VERSION: Final = "wm002-no-op-observation-assimilator-v1"
SCORER_VERSION: Final = "wm002-acquisition-terminal-log-scorer-v1"
EFFECT_VERSION: Final = "wm002-no-op-assimilation-effect-v1"
AGENT_ID: Final = "wm002-agent"
TASK_ID: Final = "wm002-hidden-actuator"
TARGET_ID: Final = "wm002-acquisition-observation-and-terminal-success"
TARGET_DESCRIPTION: Final = (
    "phase-qualified acquisition observation, inferred hidden actuator regime, "
    "and executed terminal success; the belief distribution represents the regime "
    "while action-conditional predictions represent the phase-qualified observable"
)
TARGET_KIND: Final = "composite_observation_regime_and_outcome"

AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS: Final = frozenset({"ordinal", "phase", "semantic_action"})
AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS: Final = frozenset({"observed_symbol", "phase", "semantic_action"})
AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS: Final = frozenset(
    {
        "information_acquisition_cost",
        "net_reward",
        "physical_action_cost",
        "task_payoff",
    }
)
AGENT_VISIBLE_TERMINAL_ACTION_FIELDS: Final = frozenset({"ordinal", "phase", "terminal_action"})
AGENT_VISIBLE_TERMINAL_OBSERVATION_FIELDS: Final = frozenset({"phase"})
AGENT_VISIBLE_TERMINAL_OUTCOME_FIELDS: Final = frozenset({"executed_terminal_action", "success", "terminal_reward"})


_ACQUISITION_ACTION_BY_SEMANTIC: Final = {action.action_id: action for action in ACQUISITION_ACTIONS}
_ACQUISITION_ORDINAL: Final = {action.action_id: ordinal for ordinal, action in enumerate(ACQUISITION_ACTIONS)}
_TERMINAL_ORDINAL: Final = {
    TerminalAction.DIRECT: 0,
    TerminalAction.REVERSED: 1,
}


class ArmMode(StrEnum):
    """The exact seven declared Q1 acquisition-selection modes."""

    PROSPECT = "prospect_expected_return"
    ORACLE = "independent_fraction_oracle"
    GOAL_ONLY = "goal_only"
    RAW_ENTROPY = "raw_observation_entropy"
    EIG_ONLY = "eig_only"
    SHUFFLED_INFORMATION = "shuffled_information"
    UNIFORM_RANDOM = "uniform_random"


_SELECTION_KIND: Final = {
    ArmMode.PROSPECT: "expected_episode_net_return",
    ArmMode.ORACLE: "independent_expected_episode_net_return",
    ArmMode.GOAL_ONLY: "goal_only_expected_return",
    ArmMode.RAW_ENTROPY: "raw_observation_entropy",
    ArmMode.EIG_ONLY: "expected_information_gain",
    ArmMode.SHUFFLED_INFORMATION: "shuffled_information_expected_episode_net_return",
    ArmMode.UNIFORM_RANDOM: "uniform_random",
}


@dataclass(frozen=True, slots=True)
class ExactPosteriorModel:
    """Decoded, non-executable posterior-model state."""

    evidence_count: int
    last_experience_id: str | None
    last_transition_id: str | None
    posterior_direct: Fraction
    likelihood_version: str = LIKELIHOOD_VERSION
    schema: str = POSTERIOR_MODEL_SCHEMA


def encode_posterior_model(model: ExactPosteriorModel) -> bytes:
    """Encode a posterior model as finite, sorted-key canonical UTF-8 JSON."""

    _validate_decoded_model(model)
    payload = {
        "evidence_count": model.evidence_count,
        "last_experience_id": model.last_experience_id,
        "last_transition_id": model.last_transition_id,
        "likelihood_version": model.likelihood_version,
        "posterior_direct": {
            "denominator": model.posterior_direct.denominator,
            "numerator": model.posterior_direct.numerator,
        },
        "schema": model.schema,
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_posterior_model(payload: bytes) -> ExactPosteriorModel:
    """Strictly decode and canonicality-check posterior bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("posterior model payload must be bytes")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("posterior model is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "evidence_count",
        "last_experience_id",
        "last_transition_id",
        "likelihood_version",
        "posterior_direct",
        "schema",
    }:
        raise ValueError("posterior model has an invalid top-level schema")

    evidence_count = raw["evidence_count"]
    if isinstance(evidence_count, bool) or not isinstance(evidence_count, int):
        raise ValueError("posterior model evidence_count must be an integer")
    posterior = raw["posterior_direct"]
    if not isinstance(posterior, dict) or set(posterior) != {"denominator", "numerator"}:
        raise ValueError("posterior_direct has an invalid schema")
    numerator = posterior["numerator"]
    denominator = posterior["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
    ):
        raise ValueError("posterior fraction fields must be integers")
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("posterior fraction must be in [0, 1] with a positive denominator")
    if math.gcd(numerator, denominator) != 1:
        raise ValueError("posterior fraction must be reduced")

    last_experience_id = _nullable_identifier(raw["last_experience_id"], "last_experience_id")
    last_transition_id = _nullable_identifier(raw["last_transition_id"], "last_transition_id")
    model = ExactPosteriorModel(
        evidence_count=evidence_count,
        last_experience_id=last_experience_id,
        last_transition_id=last_transition_id,
        likelihood_version=_required_string(raw["likelihood_version"], "likelihood_version"),
        posterior_direct=Fraction(numerator, denominator),
        schema=_required_string(raw["schema"], "schema"),
    )
    _validate_decoded_model(model)
    if payload != encode_posterior_model(model):
        raise ValueError("posterior model is not canonical JSON")
    return model


def model_version_for_payload(payload: bytes) -> str:
    return f"{MODEL_VERSION_PREFIX}{hashlib.sha256(payload).hexdigest()}"


def configuration_version_for_digest(digest: str) -> str:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("model digest must be lowercase SHA-256 hexadecimal")
    return f"{CONFIGURATION_VERSION_PREFIX}{digest}"


def initial_posterior_model_state(prior_direct: Fraction = Fraction(1, 2)) -> ModelState:
    """Return the canonical independent-episode predecessor model."""

    model = ExactPosteriorModel(
        evidence_count=0,
        last_experience_id=None,
        last_transition_id=None,
        posterior_direct=prior_direct,
    )
    payload = encode_posterior_model(model)
    return ModelState(version=model_version_for_payload(payload), payload=payload)


def validate_posterior_model_state(
    state: ModelState,
    predecessor: ModelState | None = None,
) -> None:
    """Validate structural state and, when supplied, exact count monotonicity.

    ``VersionedModelOwner`` calls this as a unary structural validator.  The
    transactional learner additionally supplies the immutable predecessor,
    composing the predecessor-relative invariant with source-digest custody.
    """

    decoded = decode_posterior_model(state.payload)
    if state.version != model_version_for_payload(state.payload):
        raise ValueError("posterior model version does not bind its canonical payload digest")
    if predecessor is not None:
        previous = decode_posterior_model(predecessor.payload)
        if predecessor.version != model_version_for_payload(predecessor.payload):
            raise ValueError("predecessor model version does not bind its payload digest")
        if decoded.evidence_count != previous.evidence_count + 1:
            raise ValueError("posterior model evidence_count must advance by exactly one")


def _validate_decoded_model(model: ExactPosteriorModel) -> None:
    if model.schema != POSTERIOR_MODEL_SCHEMA:
        raise ValueError("posterior model schema is not allowlisted")
    if model.likelihood_version != LIKELIHOOD_VERSION:
        raise ValueError("posterior model likelihood version is not allowlisted")
    if model.evidence_count < 0:
        raise ValueError("posterior model evidence_count must be nonnegative")
    if not Fraction(0) <= model.posterior_direct <= Fraction(1):
        raise ValueError("posterior probability must be in [0, 1]")
    if model.evidence_count == 0:
        if model.last_experience_id is not None or model.last_transition_id is not None:
            raise ValueError("zero-evidence model cannot claim consumed lineage")
    elif model.last_experience_id is None or model.last_transition_id is None:
        raise ValueError("nonzero-evidence model requires consumed lineage")


@dataclass(frozen=True, slots=True)
class CandidateDiagnosticRow:
    """Arm-specific selection sidecar linked to one truthful core assessment."""

    assessment_id: str
    semantic_action: str
    expected_episode_value: float
    expected_immediate_payoff: float
    expected_terminal_value: float
    expected_decision_value: float
    expected_information_gain_nats: float
    raw_observation_entropy_nats: float
    physical_action_cost: float
    information_acquisition_cost: float
    arm_selection_score: float | None
    selection_unit: str | None


def candidate_diagnostic_bytes(rows: Sequence[CandidateDiagnosticRow]) -> bytes:
    """Canonical bytes for the complete ordered five-row selection sidecar."""

    ordered = tuple(rows)
    if tuple(row.semantic_action for row in ordered) != tuple(action.action_id for action in ACQUISITION_ACTIONS):
        raise ValueError("candidate diagnostics do not follow the canonical five-action order")
    return (
        json.dumps(
            [asdict(row) for row in ordered],
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def candidate_diagnostic_digest(rows: Sequence[CandidateDiagnosticRow]) -> str:
    return hashlib.sha256(candidate_diagnostic_bytes(rows)).hexdigest()


def acquisition_public_probe_bytes(
    *,
    candidate_rows: Sequence[CandidateDiagnosticRow],
    decision: DecisionRecord,
    transition: EpistemicTransition,
    receipt: UpdateReceipt,
    model_state: ModelState,
    identity_next_counter: int,
    experience_count: int,
    transition_count: int,
    update_count: int,
) -> bytes:
    """Canonicalize acquisition state that may depend only on visible facts."""

    if transition.experience.decision is not decision:
        raise ValueError("probe transition does not retain the exact decision")
    resulting_belief = receipt.resulting_belief
    if receipt.transitions != (transition,) or resulting_belief is None:
        raise ValueError("probe receipt does not bind exactly one acquisition transition")
    validate_posterior_model_state(model_state)
    counts = (identity_next_counter, experience_count, transition_count, update_count)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("probe counters must be nonnegative integers")

    action = _parse_acquisition_action(decision.intended_action.action)
    experience = transition.experience
    if experience.observation.modality != "wm002_acquisition_symbol":
        raise ValueError("probe requires an acquisition observation")
    observation = _require_mapping(
        experience.observation.evidence.payload,
        set(AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS),
        "probe acquisition observation",
    )
    outcome = _require_mapping(
        experience.outcome.evidence.payload,
        set(AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS),
        "probe acquisition outcome",
    )
    action_parameters = _require_mapping(
        decision.intended_action.action.parameters,
        set(AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS),
        "probe acquisition action",
    )
    if observation["semantic_action"] != action.action_id:
        raise ValueError("probe observation does not match its selected action")

    payload = {
        "action_parameters": dict(action_parameters),
        "belief_update": {
            "posterior": transition.belief_update.posterior.distribution.parameters,
            "prior": transition.belief_update.prior.distribution.parameters,
        },
        "candidate_diagnostics": json.loads(candidate_diagnostic_bytes(candidate_rows)),
        "decision_id": decision.decision_id,
        "experience_count": experience_count,
        "identity_next_counter": identity_next_counter,
        "model": json.loads(model_state.payload),
        "model_sha256": model_state.digest,
        "observation": dict(observation),
        "outcome": dict(outcome),
        "receipt": {
            "metrics": dict(receipt.metrics),
            "new_model_version": receipt.new_model_version,
            "previous_model_version": receipt.previous_model_version,
            "receipt_id": receipt.receipt_id,
            "resulting_belief": resulting_belief.distribution.parameters,
        },
        "selected_action": decision.selected_assessment.action.action_id,
        "transition_count": transition_count,
        "transition_id": transition.transition_id,
        "update_count": update_count,
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def assert_acquisition_public_noninterference(reference: bytes, candidate: bytes) -> None:
    """Reject any byte-level effect of changed acquisition-inaccessible inputs."""

    if type(reference) is not bytes or type(candidate) is not bytes:
        raise TypeError("public noninterference probes must be bytes")
    if reference != candidate:
        raise ValueError("private acquisition inputs changed public acquisition state")


class HiddenActuatorCandidateAssessor:
    """Build five truthful return-valued assessments plus arm diagnostics."""

    def __init__(
        self,
        *,
        model_owner: VersionedModelOwner,
        identities: CounterIdentitySource,
        arm: ArmMode,
        uniform_ordinal: int | None = None,
        uniform_selector_digest: str | None = None,
    ) -> None:
        if arm is ArmMode.UNIFORM_RANDOM:
            if (
                isinstance(uniform_ordinal, bool)
                or not isinstance(uniform_ordinal, int)
                or not 0 <= uniform_ordinal < len(ACQUISITION_ACTIONS)
            ):
                raise ValueError("uniform_random requires a precomputed ordinal in [0, 5)")
            _require_sha256(uniform_selector_digest, "uniform_selector_digest")
        elif uniform_ordinal is not None or uniform_selector_digest is not None:
            raise ValueError("uniform selector fields are valid only for uniform_random")
        self._model_owner = model_owner
        self._identities = identities
        self._arm = arm
        self._uniform_ordinal = uniform_ordinal
        self._uniform_selector_digest = uniform_selector_digest

    @property
    def arm(self) -> ArmMode:
        return self._arm

    @property
    def uniform_ordinal(self) -> int | None:
        return self._uniform_ordinal

    @property
    def uniform_selector_digest(self) -> str | None:
        return self._uniform_selector_digest

    def assess(self, snapshot: AgentSnapshot, goal: Goal) -> Sequence[CandidateAssessment]:
        assessments, _ = self.assess_with_diagnostics(snapshot, goal)
        return assessments

    def assess_with_diagnostics(
        self,
        snapshot: AgentSnapshot,
        goal: Goal,
    ) -> tuple[tuple[CandidateAssessment, ...], tuple[CandidateDiagnosticRow, ...]]:
        model_state = self._model_owner.snapshot_state()
        decoded = _require_snapshot_model(snapshot, model_state)
        _require_composite_target(snapshot.belief.target)
        if goal.task_id != TASK_ID or goal.target is not snapshot.belief.target:
            raise ValueError("acquisition goal must reuse the exact WM-002 composite target")
        if decoded.evidence_count != 0 or snapshot.latest_update is not None:
            raise ValueError("acquisition assessment requires an independent zero-evidence prior")
        problem = HiddenActuatorProblem(prior_direct=float(decoded.posterior_direct))
        diagnostics = tuple(problem.diagnose(action) for action in problem.acquisition_actions)
        selection_scores, selection_unit = self._selection_scores(decoded.posterior_direct, diagnostics)

        assessments: list[CandidateAssessment] = []
        rows: list[CandidateDiagnosticRow] = []
        for action_spec, diagnostic, selection_score in zip(
            problem.acquisition_actions,
            diagnostics,
            selection_scores,
            strict=True,
        ):
            action = _acquisition_domain_action(action_spec)
            prediction = Prediction(
                prediction_id=self._identities.next(f"prediction-acquisition-{action_spec.action_id}"),
                prior_belief=snapshot.belief,
                action=action,
                target=goal.target,
                distribution=Distribution(
                    distribution_id=self._identities.next(
                        f"prediction-distribution-acquisition-{action_spec.action_id}"
                    ),
                    family="categorical",
                    support=f"observed_symbol={problem.outcomes(action_spec)!r}",
                    parameters=diagnostic.outcome_probabilities,
                    representation_version=snapshot.representation_version,
                    event_shape=(len(diagnostic.outcome_probabilities),),
                ),
                issued_at=snapshot.captured_at,
                horizon_end=_offset(snapshot.captured_at, 2),
                model_version=snapshot.model_version,
                representation_version=snapshot.representation_version,
                calibration_version=CALIBRATION_VERSION,
            )
            utility = Utility(
                utility_id=self._identities.next(f"utility-acquisition-{action_spec.action_id}"),
                goal_id=goal.goal_id,
                prediction_id=prediction.prediction_id,
                expected_value=diagnostic.prior_terminal_value + diagnostic.expected_immediate_payoff,
                unit="return",
                evaluator_version="wm002-known-return-utility-v1",
                assessed_at=snapshot.captured_at,
            )
            information_value = InformationValue(
                information_value_id=self._identities.next(f"information-value-acquisition-{action_spec.action_id}"),
                prior_belief_id=snapshot.belief.belief_id,
                action_id=action.action_id,
                target_id=goal.target.target_id,
                expected_reduction=diagnostic.expected_decision_value,
                expected_cost=diagnostic.expected_acquisition_cost,
                unit="return",
                evaluator_version="wm002-known-decision-value-v1",
                assessed_at=snapshot.captured_at,
            )
            assessment = CandidateAssessment(
                assessment_id=self._identities.next(f"assessment-acquisition-{action_spec.action_id}"),
                action=action,
                prediction=prediction,
                utility=utility,
                information_value=information_value,
                expected_action_cost=diagnostic.expected_action_cost,
                expected_risk=0.0,
                admissible=True,
                constraint_reasons=(),
                constraint_penalty=0.0,
                total_value=diagnostic.expected_episode_value,
                unit="return",
                evaluator_version="wm002-truthful-candidate-assessor-v1",
                assessed_at=snapshot.captured_at,
            )
            assessments.append(assessment)
            rows.append(
                CandidateDiagnosticRow(
                    assessment_id=assessment.assessment_id,
                    semantic_action=action_spec.action_id,
                    expected_episode_value=diagnostic.expected_episode_value,
                    expected_immediate_payoff=diagnostic.expected_immediate_payoff,
                    expected_terminal_value=(diagnostic.prior_terminal_value + diagnostic.expected_decision_value),
                    expected_decision_value=diagnostic.expected_decision_value,
                    expected_information_gain_nats=diagnostic.expected_information_gain_nats,
                    raw_observation_entropy_nats=diagnostic.observation_entropy_nats,
                    physical_action_cost=diagnostic.expected_action_cost,
                    information_acquisition_cost=diagnostic.expected_acquisition_cost,
                    arm_selection_score=selection_score,
                    selection_unit=selection_unit,
                )
            )
        return tuple(assessments), tuple(rows)

    def _selection_scores(
        self,
        prior_direct: Fraction,
        diagnostics: Sequence[AcquisitionDiagnostics],
    ) -> tuple[tuple[float | None, ...], str | None]:
        typed = tuple(diagnostics)
        if self._arm is ArmMode.PROSPECT:
            return tuple(row.expected_episode_value for row in typed), "return"
        if self._arm is ArmMode.ORACLE:
            exact = FractionOracle().decide(prior_direct=prior_direct)
            return tuple(float(row.expected_episode_value) for row in exact.evaluations), "return"
        if self._arm is ArmMode.GOAL_ONLY:
            return (
                tuple(
                    row.prior_terminal_value
                    + row.expected_immediate_payoff
                    - row.expected_action_cost
                    - row.expected_acquisition_cost
                    for row in typed
                ),
                "return",
            )
        if self._arm is ArmMode.RAW_ENTROPY:
            return tuple(row.observation_entropy_nats for row in typed), "nats"
        if self._arm is ArmMode.EIG_ONLY:
            return tuple(row.expected_information_gain_nats for row in typed), "nats"
        if self._arm is ArmMode.SHUFFLED_INFORMATION:
            decision_value_by_action = {
                action.action_id: row.expected_decision_value
                for action, row in zip(ACQUISITION_ACTIONS, typed, strict=True)
            }
            return (
                tuple(
                    row.prior_terminal_value
                    + row.expected_immediate_payoff
                    + decision_value_by_action[SHUFFLED_INFORMATION_SOURCE_BY_ACTION[action.action_id]]
                    - row.expected_action_cost
                    - row.expected_acquisition_cost
                    for action, row in zip(ACQUISITION_ACTIONS, typed, strict=True)
                ),
                "return",
            )
        return (None,) * len(typed), None


class ArmDecisionPolicy:
    """Select by one declared arm score while preserving core assessments."""

    def __init__(
        self,
        *,
        agent_id: str,
        policy_version: str,
        assessor: HiddenActuatorCandidateAssessor,
        identities: CounterIdentitySource,
    ) -> None:
        if not agent_id.strip() or not policy_version.strip():
            raise ValueError("agent_id and policy_version must be nonempty")
        self._agent_id = agent_id
        self._policy_version = policy_version
        self._assessor = assessor
        self._identities = identities
        self._last_rows: tuple[CandidateDiagnosticRow, ...] | None = None
        self._last_digest: str | None = None

    @property
    def arm(self) -> ArmMode:
        return self._assessor.arm

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def selection_kind(self) -> str:
        return _SELECTION_KIND[self.arm]

    @property
    def last_diagnostic_rows(self) -> tuple[CandidateDiagnosticRow, ...]:
        if self._last_rows is None:
            raise RuntimeError("policy has not made an acquisition decision")
        return self._last_rows

    @property
    def last_diagnostic_digest(self) -> str:
        if self._last_digest is None:
            raise RuntimeError("policy has not made an acquisition decision")
        return self._last_digest

    def decide(self, snapshot: AgentSnapshot, goal: Goal) -> DecisionRecord:
        if snapshot.agent_id != self._agent_id:
            raise DecisionError("policy received another agent's snapshot")
        if snapshot.policy_version != self._policy_version:
            raise DecisionError("policy version does not match the frozen snapshot")
        alternatives, rows = self._assessor.assess_with_diagnostics(snapshot, goal)
        if tuple(row.assessment_id for row in rows) != tuple(row.assessment_id for row in alternatives):
            raise DecisionError("diagnostic rows do not link one-to-one to candidate assessments")
        if self.arm is ArmMode.UNIFORM_RANDOM:
            selected_index = self._assessor.uniform_ordinal
            if selected_index is None:
                raise DecisionError("uniform_random assessor has no frozen selection ordinal")
        else:
            selected_index = max(
                range(len(rows)),
                key=lambda index: _required_score(rows[index]),
            )
        selected = alternatives[selected_index]
        decided_at = _latest_time(
            snapshot.captured_at,
            goal.issued_at,
            *(assessment.assessed_at for assessment in alternatives),
        )
        intention = IntendedAction(
            intention_id=self._identities.next("intention-acquisition"),
            agent_id=self._agent_id,
            action=selected.action,
            intended_at=decided_at,
        )
        self._last_rows = rows
        self._last_digest = candidate_diagnostic_digest(rows)
        return DecisionRecord(
            decision_id=self._identities.next("decision-acquisition"),
            agent_id=self._agent_id,
            belief=snapshot.belief,
            goal=goal,
            intended_action=intention,
            alternatives=alternatives,
            selected_assessment=selected,
            policy_version=self._policy_version,
            decided_at=decided_at,
        )


class HiddenActuatorAcquisitionEnvironment:
    """One-use gateway for one exact, privately precomputed observation symbol.

    The private regime and 256-bit realization digest never enter the agent
    runtime graph.  The permissioned producer computes the selected action's
    symbol with exact integer arithmetic and supplies only that visible fact.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        observed_symbol: int,
        identities: CounterIdentitySource,
        problem: HiddenActuatorProblem | None = None,
    ) -> None:
        if type(observed_symbol) is not int:
            raise TypeError("observed_symbol must be an integer")
        self._agent_id = agent_id
        self._observed_symbol = observed_symbol
        self._identities = identities
        self._problem = problem or HiddenActuatorProblem()
        self._used = False

    def step(self, intention: IntendedAction) -> EnvironmentStep:
        if self._used:
            raise RuntimeError("acquisition environment is single-use")
        if intention.agent_id != self._agent_id:
            raise ValueError("acquisition intention belongs to another agent")
        action = _parse_acquisition_action(intention.action)
        if self._observed_symbol not in self._problem.outcomes(action):
            raise ValueError("observed_symbol is outside the selected action support")
        task_payoff = 1.0 if action.is_pulse and self._observed_symbol == 1 else 0.0
        action_cost = action.action_cost
        acquisition_cost = action.acquisition_cost
        net_reward = task_payoff - action_cost - acquisition_cost
        self._used = True
        started = _offset(intention.intended_at, 1)
        ended = _offset(started, 1)
        execution = ExecutedAction(
            execution_id=self._identities.next("execution-acquisition"),
            intention=intention,
            status=ExecutionStatus.SUCCEEDED,
            started_at=started,
            ended_at=ended,
            realized_action=intention.action,
        )
        observation_id = self._identities.next("observation-acquisition")
        observation = Observation(
            observation_id=observation_id,
            agent_id=self._agent_id,
            modality="wm002_acquisition_symbol",
            evidence=_observed_evidence(
                evidence_id=observation_id,
                payload={
                    "phase": "acquisition",
                    "semantic_action": action.action_id,
                    "observed_symbol": self._observed_symbol,
                },
                at=ended,
                source_id="wm002-acquisition-environment",
                source_kind="hidden_actuator_sensor",
            ),
        )
        outcome_evidence_id = self._identities.next("outcome-evidence-acquisition")
        outcome = Outcome(
            outcome_id=self._identities.next("outcome-acquisition"),
            execution_id=execution.execution_id,
            evidence=_observed_evidence(
                evidence_id=outcome_evidence_id,
                payload={
                    "task_payoff": task_payoff,
                    "physical_action_cost": action_cost,
                    "information_acquisition_cost": acquisition_cost,
                    "net_reward": net_reward,
                },
                at=ended,
                source_id="wm002-acquisition-environment",
                source_kind="executed_return_meter",
            ),
        )
        step = EnvironmentStep(
            execution=execution,
            observation=observation,
            outcome=outcome,
            closed_at=ended,
        )
        validate_agent_visible_environment_step(step)
        return step


class HiddenActuatorTerminalEnvironment:
    """One-use gateway for one exact, privately precomputed success bit."""

    def __init__(
        self,
        *,
        agent_id: str,
        terminal_success: bool,
        identities: CounterIdentitySource,
    ) -> None:
        if type(terminal_success) is not bool:
            raise TypeError("terminal_success must be a boolean")
        self._agent_id = agent_id
        self._terminal_success = terminal_success
        self._identities = identities
        self._used = False

    def step(self, intention: IntendedAction) -> EnvironmentStep:
        if self._used:
            raise RuntimeError("terminal environment is single-use")
        if intention.agent_id != self._agent_id:
            raise ValueError("terminal intention belongs to another agent")
        action = _parse_terminal_action(intention.action)
        self._used = True
        started = _offset(intention.intended_at, 1)
        ended = _offset(started, 1)
        execution = ExecutedAction(
            execution_id=self._identities.next("execution-terminal"),
            intention=intention,
            status=ExecutionStatus.SUCCEEDED,
            started_at=started,
            ended_at=ended,
            realized_action=intention.action,
        )
        observation_id = self._identities.next("observation-terminal")
        observation = Observation(
            observation_id=observation_id,
            agent_id=self._agent_id,
            modality="wm002_terminal_neutral",
            evidence=_observed_evidence(
                evidence_id=observation_id,
                payload={"phase": "terminal"},
                at=ended,
                source_id="wm002-terminal-environment",
                source_kind="terminal_marker",
            ),
        )
        outcome_evidence_id = self._identities.next("outcome-evidence-terminal")
        outcome = Outcome(
            outcome_id=self._identities.next("outcome-terminal"),
            execution_id=execution.execution_id,
            evidence=_observed_evidence(
                evidence_id=outcome_evidence_id,
                payload={
                    "executed_terminal_action": int(action),
                    "success": self._terminal_success,
                    "terminal_reward": float(self._terminal_success),
                },
                at=ended,
                source_id="wm002-terminal-environment",
                source_kind="terminal_success_meter",
            ),
        )
        step = EnvironmentStep(
            execution=execution,
            observation=observation,
            outcome=outcome,
            closed_at=ended,
            terminated=True,
            discount=0.0,
        )
        validate_agent_visible_environment_step(step)
        return step


def validate_agent_visible_environment_step(step: EnvironmentStep) -> None:
    """Fail closed on every field crossing an environment-to-agent boundary."""

    action = step.execution.intention.action
    observation_payload = step.observation.evidence.payload
    outcome_payload = step.outcome.evidence.payload
    if step.observation.modality == "wm002_acquisition_symbol":
        acquisition = _parse_acquisition_action(action)
        observation = _require_mapping(
            observation_payload,
            set(AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS),
            "agent-visible acquisition observation",
        )
        outcome = _require_mapping(
            outcome_payload,
            set(AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS),
            "agent-visible acquisition outcome",
        )
        _require_mapping(
            action.parameters,
            set(AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS),
            "agent-visible acquisition action",
        )
        symbol = observation["observed_symbol"]
        if (
            observation["phase"] != "acquisition"
            or observation["semantic_action"] != acquisition.action_id
            or type(symbol) is not int
            or symbol not in HiddenActuatorProblem().outcomes(acquisition)
        ):
            raise ValueError("agent-visible acquisition observation is invalid")
        if any(type(outcome[field]) is not float for field in AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS):
            raise ValueError("agent-visible acquisition outcome values must be floats")
        return

    if step.observation.modality == "wm002_terminal_neutral":
        terminal = _parse_terminal_action(action)
        observation = _require_mapping(
            observation_payload,
            set(AGENT_VISIBLE_TERMINAL_OBSERVATION_FIELDS),
            "agent-visible terminal observation",
        )
        outcome = _require_mapping(
            outcome_payload,
            set(AGENT_VISIBLE_TERMINAL_OUTCOME_FIELDS),
            "agent-visible terminal outcome",
        )
        _require_mapping(
            action.parameters,
            set(AGENT_VISIBLE_TERMINAL_ACTION_FIELDS),
            "agent-visible terminal action",
        )
        if observation["phase"] != "terminal":
            raise ValueError("agent-visible terminal observation is invalid")
        if (
            outcome["executed_terminal_action"] != int(terminal)
            or type(outcome["success"]) is not bool
            or type(outcome["terminal_reward"]) is not float
            or outcome["terminal_reward"] != float(outcome["success"])
        ):
            raise ValueError("agent-visible terminal outcome is invalid")
        return

    raise ValueError("environment step has a non-allowlisted observation modality")


class NoOpObservationAssimilator:
    """Register real observations without performing persistent inference."""

    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def assimilate(self, prior: Belief, experience: ExperienceEvent) -> BeliefUpdate:
        _visible_phase(experience)
        memory_version = self._identities.next("memory")
        information = InformationSet(
            information_set_id=self._identities.next("information-set"),
            agent_id=prior.agent_id,
            as_of=experience.closed_at,
            observations=(*prior.information_set.observations, experience.observation),
            memory_version=memory_version,
        )
        posterior = Belief(
            belief_id=self._identities.next("belief-assimilation"),
            agent_id=prior.agent_id,
            target=prior.target,
            information_set=information,
            distribution=Distribution(
                distribution_id=self._identities.next("belief-distribution-assimilation"),
                family=prior.distribution.family,
                support=prior.distribution.support,
                parameters=prior.distribution.parameters,
                representation_version=prior.representation_version,
                event_shape=prior.distribution.event_shape,
            ),
            formed_at=experience.closed_at,
            model_version=prior.model_version,
            representation_version=prior.representation_version,
        )
        return BeliefUpdate(
            update_id=self._identities.next("belief-update-assimilation"),
            prior=prior,
            experience=experience,
            posterior=posterior,
            updater_version=ASSIMILATOR_VERSION,
            updated_at=experience.closed_at,
        )


class AcquisitionTerminalScorer:
    """Score acquisition observations and terminal success in their own phases."""

    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def score(self, prediction: Prediction, experience: ExperienceEvent) -> ProperScore:
        probabilities = categorical_probabilities(prediction.distribution)
        phase = _action_phase(prediction.action)
        if phase == "acquisition":
            acquisition_action = _parse_acquisition_action(prediction.action)
            payload = _require_mapping(
                experience.observation.evidence.payload,
                {"phase", "semantic_action", "observed_symbol"},
                "acquisition observation",
            )
            if payload["phase"] != "acquisition" or payload["semantic_action"] != acquisition_action.action_id:
                raise ValueError("acquisition observation does not match its prediction action")
            symbol = payload["observed_symbol"]
            if isinstance(symbol, bool) or not isinstance(symbol, int):
                raise ValueError("acquisition observed_symbol must be an integer")
            try:
                observed_index = HiddenActuatorProblem().outcomes(acquisition_action).index(symbol)
            except ValueError as error:
                raise ValueError("acquisition observed_symbol is outside the action support") from error
            evidence = experience.observation.evidence
        elif phase == "terminal":
            terminal_action = _parse_terminal_action(prediction.action)
            payload = _require_mapping(
                experience.outcome.evidence.payload,
                {"executed_terminal_action", "success", "terminal_reward"},
                "terminal outcome",
            )
            if payload["executed_terminal_action"] != int(terminal_action) or not isinstance(payload["success"], bool):
                raise ValueError("terminal outcome does not match its prediction action")
            observed_index = int(payload["success"])
            evidence = experience.outcome.evidence
        else:
            raise ValueError("scorer received an unsupported phase")
        return ProperScore(
            score_id=self._identities.next(f"proper-score-{phase}"),
            prediction_id=prediction.prediction_id,
            realized_evidence_id=evidence.evidence_id,
            rule="categorical_log_score",
            value=categorical_log_score(probabilities, observed_index),
            unit="nats",
            scorer_version=SCORER_VERSION,
            scored_at=experience.closed_at,
        )


class NoOpAssimilationEffect:
    """Record that assimilation itself made no posterior entropy change."""

    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def effect(self, update: BeliefUpdate) -> EpistemicEffect:
        before = entropy(categorical_probabilities(update.prior.distribution))
        after = entropy(categorical_probabilities(update.posterior.distribution))
        if update.prior.distribution.parameters != update.posterior.distribution.parameters:
            raise ValueError("no-op assimilation changed categorical probabilities")
        return EpistemicEffect(
            effect_id=self._identities.next("epistemic-effect-assimilation"),
            belief_update_id=update.update_id,
            target_id=update.prior.target.target_id,
            kind=EpistemicEffectKind.INFORMATION_GAIN,
            measure="assimilation_only_categorical_entropy",
            before=before,
            after=after,
            improvement=before - after,
            higher_is_better=False,
            evaluator_version=EFFECT_VERSION,
            evaluated_at=update.updated_at,
            externally_calibrated=False,
        )


class ExactPosteriorTransactionalLearner:
    """Prepare one exact posterior update from one canonical acquisition transition."""

    def __init__(
        self,
        *,
        identities: CounterIdentitySource,
        expected_episode_id: str,
        expected_arm: ArmMode,
        expected_policy_version: str,
    ) -> None:
        self._identities = identities
        self._expected_episode_id = expected_episode_id
        self._expected_arm = expected_arm
        self._expected_policy_version = expected_policy_version

    def prepare(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
        current_model: ModelState,
    ) -> PreparedLearningUpdate:
        inputs = tuple(transitions)
        if len(inputs) != 1:
            raise ValueError("WM-002 learner requires exactly one acquisition transition")
        transition = inputs[0]
        action, observed_symbol = self._require_transition(snapshot, transition, current_model)
        previous = decode_posterior_model(current_model.payload)
        if previous.evidence_count != 0 or snapshot.latest_update is not None:
            raise ValueError("WM-002 permits exactly one acquisition update per independent episode")
        posterior = _exact_posterior(previous.posterior_direct, action, observed_symbol)
        candidate_decoded = ExactPosteriorModel(
            evidence_count=previous.evidence_count + 1,
            last_experience_id=transition.experience.experience_id,
            last_transition_id=transition.transition_id,
            posterior_direct=posterior,
        )
        candidate_payload = encode_posterior_model(candidate_decoded)
        candidate = ModelState(
            version=model_version_for_payload(candidate_payload),
            payload=candidate_payload,
        )
        validate_posterior_model_state(candidate, predecessor=current_model)
        completed_at = _offset(snapshot.captured_at, 1)
        source_belief = transition.belief_update.posterior
        resulting_belief = Belief(
            belief_id=self._identities.next("belief-learning"),
            agent_id=snapshot.agent_id,
            target=source_belief.target,
            information_set=source_belief.information_set,
            distribution=Distribution(
                distribution_id=self._identities.next("belief-distribution-learning"),
                family="categorical",
                support="actuator_regime={-1,+1}",
                parameters=(float(1 - posterior), float(posterior)),
                representation_version=snapshot.representation_version,
                event_shape=(2,),
            ),
            formed_at=completed_at,
            model_version=candidate.version,
            representation_version=snapshot.representation_version,
        )
        before_entropy = _fraction_entropy(previous.posterior_direct)
        after_entropy = _fraction_entropy(posterior)
        receipt = UpdateReceipt(
            receipt_id=self._identities.next("update-receipt-acquisition"),
            agent_id=snapshot.agent_id,
            transitions=(transition,),
            learner_version=LEARNER_VERSION,
            status=UpdateStatus.APPLIED,
            previous_configuration_version=snapshot.configuration_version,
            new_configuration_version=configuration_version_for_digest(candidate.digest),
            previous_model_version=current_model.version,
            new_model_version=candidate.version,
            previous_representation_version=snapshot.representation_version,
            new_representation_version=snapshot.representation_version,
            previous_policy_version=snapshot.policy_version,
            new_policy_version=snapshot.policy_version,
            started_at=completed_at,
            completed_at=completed_at,
            resulting_belief=resulting_belief,
            metrics=(
                ("posterior_direct_before", float(previous.posterior_direct)),
                ("posterior_direct_after", float(posterior)),
                ("entropy_before_nats", before_entropy),
                ("entropy_after_nats", after_entropy),
                ("entropy_reduction_nats", before_entropy - after_entropy),
                ("consumed_transition_count", 1.0),
                ("evidence_count_before", float(previous.evidence_count)),
                ("evidence_count_after", float(candidate_decoded.evidence_count)),
            ),
        )
        return PreparedLearningUpdate(
            receipt=receipt,
            source_model_digest=current_model.digest,
            candidate_model=candidate,
        )

    def _require_transition(
        self,
        snapshot: AgentSnapshot,
        transition: EpistemicTransition,
        current_model: ModelState,
    ) -> tuple[AcquisitionAction, int]:
        validate_posterior_model_state(current_model)
        _require_snapshot_model(snapshot, current_model)
        experience = transition.experience
        if (
            experience.agent_id != snapshot.agent_id
            or experience.task_id != TASK_ID
            or experience.episode_id != self._expected_episode_id
            or experience.step_index != 0
            or experience.terminated
            or experience.truncated
        ):
            raise ValueError("learner received a transition outside its acquisition episode/step")
        decision = experience.decision
        execution = experience.execution
        if decision is None or execution is None:
            raise ValueError("acquisition transition lacks decision or execution")
        _require_composite_target(decision.belief.target)
        if decision.goal.target is not decision.belief.target:
            raise ValueError("acquisition decision does not reuse the exact composite target")
        if (
            decision.policy_version != self._expected_policy_version
            or decision.policy_version != snapshot.policy_version
            or self._expected_arm.value not in decision.policy_version
        ):
            raise ValueError("acquisition transition does not identify the expected arm policy")
        if execution.intention is not decision.intended_action:
            raise ValueError("execution does not carry the canonical intended action")
        if execution.realized_action is not decision.intended_action.action:
            raise ValueError("execution did not realize the selected canonical action")
        if transition.belief_update.experience is not experience:
            raise ValueError("belief update does not carry the canonical experience")
        if transition.belief_update.prior is not decision.belief:
            raise ValueError("belief update prior is not the canonical decision belief")
        if transition.belief_update.posterior is not snapshot.belief:
            raise ValueError("learner snapshot is not the canonical no-op posterior")
        update = transition.belief_update
        if update.updater_version != ASSIMILATOR_VERSION or categorical_probabilities(
            update.prior.distribution
        ) != categorical_probabilities(update.posterior.distribution):
            raise ValueError("acquisition transition is not the canonical no-op assimilation")
        if len(transition.proper_scores) != 1 or transition.proper_scores[0].scorer_version != SCORER_VERSION:
            raise ValueError("acquisition transition lacks its canonical observation score")
        if (
            len(transition.effects) != 1
            or transition.effects[0].evaluator_version != EFFECT_VERSION
            or transition.effects[0].improvement != 0.0
        ):
            raise ValueError("acquisition transition lacks its zero assimilation effect")
        if decision.belief.model_version != current_model.version:
            raise ValueError("acquisition decision does not name the predecessor model version")

        action = _parse_acquisition_action(decision.intended_action.action)
        if experience.observation.modality != "wm002_acquisition_symbol":
            raise ValueError("acquisition transition has the wrong observation modality")
        payload = _require_mapping(
            experience.observation.evidence.payload,
            {"phase", "semantic_action", "observed_symbol"},
            "acquisition observation",
        )
        if payload["phase"] != "acquisition" or payload["semantic_action"] != action.action_id:
            raise ValueError("acquisition observation does not match the executed action")
        symbol = payload["observed_symbol"]
        if isinstance(symbol, bool) or not isinstance(symbol, int):
            raise ValueError("acquisition observed_symbol must be an integer")
        if symbol not in HiddenActuatorProblem().outcomes(action):
            raise ValueError("acquisition observed_symbol is outside the executed action support")
        return action, symbol


class TerminalCandidateAssessor:
    """Assess terminal actions from the committed exact posterior."""

    def __init__(
        self,
        *,
        model_owner: VersionedModelOwner,
        identities: CounterIdentitySource,
    ) -> None:
        self._model_owner = model_owner
        self._identities = identities

    def assess(self, snapshot: AgentSnapshot, goal: Goal) -> Sequence[CandidateAssessment]:
        state = self._model_owner.snapshot_state()
        decoded = _require_snapshot_model(snapshot, state)
        _require_composite_target(snapshot.belief.target)
        if goal.target is not snapshot.belief.target:
            raise ValueError("terminal goal must reuse the snapshot's exact composite target")
        receipt = _require_committed_acquisition_receipt(snapshot)
        source = receipt.transitions[0]
        if (
            decoded.evidence_count != 1
            or decoded.last_experience_id != source.experience.experience_id
            or decoded.last_transition_id != source.transition_id
        ):
            raise ValueError("terminal model bytes do not bind the acquisition receipt lineage")

        problem = HiddenActuatorProblem(prior_direct=float(decoded.posterior_direct))
        assessments: list[CandidateAssessment] = []
        for terminal_action in TERMINAL_ACTIONS:
            action = _terminal_domain_action(terminal_action)
            success_probability = problem.expected_terminal_payoff(terminal_action)
            prediction = Prediction(
                prediction_id=self._identities.next(f"prediction-terminal-{_terminal_label(terminal_action)}"),
                prior_belief=snapshot.belief,
                action=action,
                target=goal.target,
                distribution=Distribution(
                    distribution_id=self._identities.next(
                        f"prediction-distribution-terminal-{_terminal_label(terminal_action)}"
                    ),
                    family="categorical",
                    support="terminal_success={false,true}",
                    parameters=(1.0 - success_probability, success_probability),
                    representation_version=snapshot.representation_version,
                    event_shape=(2,),
                ),
                issued_at=snapshot.captured_at,
                horizon_end=_offset(snapshot.captured_at, 2),
                model_version=snapshot.model_version,
                representation_version=snapshot.representation_version,
                calibration_version=CALIBRATION_VERSION,
            )
            utility = Utility(
                utility_id=self._identities.next(f"utility-terminal-{_terminal_label(terminal_action)}"),
                goal_id=goal.goal_id,
                prediction_id=prediction.prediction_id,
                expected_value=success_probability,
                unit="return",
                evaluator_version="wm002-known-terminal-success-v1",
                assessed_at=snapshot.captured_at,
            )
            information = InformationValue(
                information_value_id=self._identities.next(
                    f"information-value-terminal-{_terminal_label(terminal_action)}"
                ),
                prior_belief_id=snapshot.belief.belief_id,
                action_id=action.action_id,
                target_id=goal.target.target_id,
                expected_reduction=0.0,
                expected_cost=0.0,
                unit="return",
                evaluator_version="wm002-terminal-no-information-v1",
                assessed_at=snapshot.captured_at,
            )
            assessments.append(
                CandidateAssessment(
                    assessment_id=self._identities.next(f"assessment-terminal-{_terminal_label(terminal_action)}"),
                    action=action,
                    prediction=prediction,
                    utility=utility,
                    information_value=information,
                    expected_action_cost=0.0,
                    expected_risk=0.0,
                    admissible=True,
                    constraint_reasons=(),
                    constraint_penalty=0.0,
                    total_value=success_probability,
                    unit="return",
                    evaluator_version="wm002-terminal-candidate-assessor-v1",
                    assessed_at=snapshot.captured_at,
                )
            )
        return tuple(assessments)


@dataclass(frozen=True, slots=True)
class EpisodeRuntimeComposition:
    """Fresh per-episode Prospect custodians and phase-specific agent façades."""

    agent_id: str
    episode_id: str
    arm: ArmMode
    policy_version: str
    identities: CounterIdentitySource
    state: AgentState
    store: InMemoryExperienceStore
    ledger: EpistemicLedger
    model_owner: VersionedModelOwner
    acquisition_assessor: HiddenActuatorCandidateAssessor
    acquisition_policy: ArmDecisionPolicy
    terminal_assessor: TerminalCandidateAssessor
    acquisition_agent: EpistemicAgent
    terminal_agent: EpistemicAgent
    acquisition_goal: Goal
    terminal_goal: Goal


@dataclass(frozen=True, slots=True)
class RestoredTerminalRuntimeComposition:
    """Fresh terminal-only component graph over verified restored custodians."""

    agent_id: str
    episode_id: str
    arm: ArmMode
    policy_version: str
    identities: CounterIdentitySource
    state: AgentState
    store: InMemoryExperienceStore
    ledger: EpistemicLedger
    model_owner: VersionedModelOwner
    belief_updater: NoOpObservationAssimilator
    scorer: AcquisitionTerminalScorer
    effect_assessor: NoOpAssimilationEffect
    learner: ExactPosteriorTransactionalLearner
    terminal_assessor: TerminalCandidateAssessor
    terminal_policy: MaxValuePolicy
    terminal_agent: EpistemicAgent
    terminal_goal: Goal


def compose_episode_runtime(
    *,
    episode_id: str,
    arm: ArmMode,
    identity_namespace: str,
    identity_next_counter: int = 0,
    uniform_ordinal: int | None = None,
    uniform_selector_digest: str | None = None,
    agent_id: str = AGENT_ID,
    prior_direct: Fraction = Fraction(1, 2),
) -> EpisodeRuntimeComposition:
    """Construct a fresh independent-prior runtime graph without interacting."""

    if not episode_id.strip():
        raise ValueError("episode_id must be nonempty")
    identities = CounterIdentitySource(identity_namespace, next_counter=identity_next_counter)
    initial_model = initial_posterior_model_state(prior_direct)
    model_owner = VersionedModelOwner(initial_model, validator=validate_posterior_model_state)
    policy_version = f"wm002-q1:{arm.value}:policy-v1"
    target = EpistemicTarget(
        target_id=TARGET_ID,
        description=TARGET_DESCRIPTION,
        target_kind=TARGET_KIND,
    )
    initial_information = InformationSet(
        information_set_id=f"{episode_id}:information:initial",
        agent_id=agent_id,
        as_of=TimePoint(0),
        observations=(),
        memory_version=f"{episode_id}:memory:initial",
    )
    initial_belief = Belief(
        belief_id=f"{episode_id}:belief:initial",
        agent_id=agent_id,
        target=target,
        information_set=initial_information,
        distribution=Distribution(
            distribution_id=f"{episode_id}:distribution:initial",
            family="categorical",
            support="actuator_regime={-1,+1}",
            parameters=(float(1 - prior_direct), float(prior_direct)),
            representation_version=REPRESENTATION_VERSION,
            event_shape=(2,),
        ),
        formed_at=TimePoint(0),
        model_version=initial_model.version,
        representation_version=REPRESENTATION_VERSION,
    )
    initial_snapshot = AgentSnapshot(
        snapshot_id=f"{episode_id}:snapshot:initial",
        agent_id=agent_id,
        captured_at=TimePoint(0),
        belief=initial_belief,
        configuration_version=configuration_version_for_digest(initial_model.digest),
        memory_version=initial_information.memory_version,
        knowledge_version="wm002-known-fixture-v1",
        model_version=initial_model.version,
        representation_version=REPRESENTATION_VERSION,
        policy_version=policy_version,
        resources=ResourceLedger(
            ledger_id=f"{episode_id}:resources:initial",
            started_at=TimePoint(0),
            completed_at=TimePoint(0),
        ),
    )
    state = AgentState(initial_snapshot)
    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    assimilation = NoOpObservationAssimilator(identities)
    scorer = AcquisitionTerminalScorer(identities)
    effect = NoOpAssimilationEffect(identities)
    learner = ExactPosteriorTransactionalLearner(
        identities=identities,
        expected_episode_id=episode_id,
        expected_arm=arm,
        expected_policy_version=policy_version,
    )
    acquisition_assessor = HiddenActuatorCandidateAssessor(
        model_owner=model_owner,
        identities=identities,
        arm=arm,
        uniform_ordinal=uniform_ordinal,
        uniform_selector_digest=uniform_selector_digest,
    )
    acquisition_policy = ArmDecisionPolicy(
        agent_id=agent_id,
        policy_version=policy_version,
        assessor=acquisition_assessor,
        identities=identities,
    )
    terminal_assessor = TerminalCandidateAssessor(
        model_owner=model_owner,
        identities=identities,
    )
    terminal_policy = MaxValuePolicy(
        agent_id=agent_id,
        policy_version=policy_version,
        assessor=terminal_assessor,
        identities=identities,
    )
    acquisition_agent = EpistemicAgent(
        state=state,
        policy=acquisition_policy,
        belief_updater=assimilation,
        scorer=scorer,
        effect_assessor=effect,
        learner=learner,
        experience_store=store,
        ledger=ledger,
        identities=identities,
        model=model_owner,
    )
    terminal_agent = EpistemicAgent(
        state=state,
        policy=terminal_policy,
        belief_updater=assimilation,
        scorer=scorer,
        effect_assessor=effect,
        learner=learner,
        experience_store=store,
        ledger=ledger,
        identities=identities,
        model=model_owner,
    )
    acquisition_goal = Goal(
        goal_id=f"{episode_id}:goal:acquisition",
        task_id=TASK_ID,
        target=target,
        description="maximize expected executed episode net return",
        issued_at=TimePoint(0),
        preference_version="wm002-episode-return-v1",
    )
    terminal_goal = Goal(
        goal_id=f"{episode_id}:goal:terminal",
        task_id=TASK_ID,
        target=target,
        description="maximize expected terminal success",
        issued_at=TimePoint(0),
        preference_version="wm002-terminal-success-v1",
    )
    return EpisodeRuntimeComposition(
        agent_id=agent_id,
        episode_id=episode_id,
        arm=arm,
        policy_version=policy_version,
        identities=identities,
        state=state,
        store=store,
        ledger=ledger,
        model_owner=model_owner,
        acquisition_assessor=acquisition_assessor,
        acquisition_policy=acquisition_policy,
        terminal_assessor=terminal_assessor,
        acquisition_agent=acquisition_agent,
        terminal_agent=terminal_agent,
        acquisition_goal=acquisition_goal,
        terminal_goal=terminal_goal,
    )


def compose_restored_terminal_runtime(
    *,
    episode_id: str,
    arm: ArmMode,
    restored_at: TimePoint,
    identities: CounterIdentitySource,
    state: AgentState,
    store: InMemoryExperienceStore,
    ledger: EpistemicLedger,
    model_owner: VersionedModelOwner,
    agent_id: str = AGENT_ID,
) -> RestoredTerminalRuntimeComposition:
    """Construct fresh terminal components over an integrity-checked restore.

    The caller owns checkpoint verification and canonical record rehydration.
    This function verifies the resulting state/model boundary and creates no
    environment, private draw, or interaction.
    """

    if not episode_id.strip():
        raise ValueError("episode_id must be nonempty")
    snapshot = state.snapshot(agent_id, restored_at)
    _require_snapshot_model(snapshot, model_owner.snapshot_state())
    _require_composite_target(snapshot.belief.target)
    _require_committed_acquisition_receipt(snapshot)
    policy_version = f"wm002-q1:{arm.value}:policy-v1"
    if snapshot.policy_version != policy_version:
        raise ValueError("restored snapshot policy version does not match the declared arm")

    belief_updater = NoOpObservationAssimilator(identities)
    scorer = AcquisitionTerminalScorer(identities)
    effect_assessor = NoOpAssimilationEffect(identities)
    learner = ExactPosteriorTransactionalLearner(
        identities=identities,
        expected_episode_id=episode_id,
        expected_arm=arm,
        expected_policy_version=policy_version,
    )
    terminal_assessor = TerminalCandidateAssessor(
        model_owner=model_owner,
        identities=identities,
    )
    terminal_policy = MaxValuePolicy(
        agent_id=agent_id,
        policy_version=policy_version,
        assessor=terminal_assessor,
        identities=identities,
    )
    terminal_agent = EpistemicAgent(
        state=state,
        policy=terminal_policy,
        belief_updater=belief_updater,
        scorer=scorer,
        effect_assessor=effect_assessor,
        learner=learner,
        experience_store=store,
        ledger=ledger,
        identities=identities,
        model=model_owner,
    )
    terminal_goal = Goal(
        goal_id=f"{episode_id}:goal:terminal",
        task_id=TASK_ID,
        target=snapshot.belief.target,
        description="maximize expected terminal success",
        issued_at=TimePoint(0, clock_id=restored_at.clock_id),
        preference_version="wm002-terminal-success-v1",
    )
    return RestoredTerminalRuntimeComposition(
        agent_id=agent_id,
        episode_id=episode_id,
        arm=arm,
        policy_version=policy_version,
        identities=identities,
        state=state,
        store=store,
        ledger=ledger,
        model_owner=model_owner,
        belief_updater=belief_updater,
        scorer=scorer,
        effect_assessor=effect_assessor,
        learner=learner,
        terminal_assessor=terminal_assessor,
        terminal_policy=terminal_policy,
        terminal_agent=terminal_agent,
        terminal_goal=terminal_goal,
    )


def _acquisition_domain_action(action: AcquisitionAction) -> Action:
    ordinal = _ACQUISITION_ORDINAL[action.action_id]
    return Action(
        action_id=f"acquisition:{ordinal:02d}:{action.action_id}",
        action_kind="wm002_acquisition",
        parameters={
            "phase": "acquisition",
            "ordinal": ordinal,
            "semantic_action": action.action_id,
        },
    )


def _terminal_domain_action(action: TerminalAction) -> Action:
    ordinal = _TERMINAL_ORDINAL[action]
    return Action(
        action_id=f"terminal:{ordinal:02d}:{int(action):+d}",
        action_kind="wm002_terminal",
        parameters={
            "phase": "terminal",
            "ordinal": ordinal,
            "terminal_action": int(action),
        },
    )


def _parse_acquisition_action(action: Action) -> AcquisitionAction:
    payload = _require_mapping(
        action.parameters,
        {"phase", "ordinal", "semantic_action"},
        "acquisition action parameters",
    )
    semantic = payload["semantic_action"]
    if not isinstance(semantic, str) or semantic not in _ACQUISITION_ACTION_BY_SEMANTIC:
        raise ValueError("action has an unknown acquisition semantic ID")
    expected = _acquisition_domain_action(_ACQUISITION_ACTION_BY_SEMANTIC[semantic])
    if action != expected:
        raise ValueError("action does not match its canonical phase-qualified acquisition form")
    return _ACQUISITION_ACTION_BY_SEMANTIC[semantic]


def _parse_terminal_action(action: Action) -> TerminalAction:
    payload = _require_mapping(
        action.parameters,
        {"phase", "ordinal", "terminal_action"},
        "terminal action parameters",
    )
    raw = payload["terminal_action"]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("terminal action value must be an integer")
    try:
        terminal = TerminalAction(raw)
    except ValueError as error:
        raise ValueError("terminal action value must be -1 or +1") from error
    if action != _terminal_domain_action(terminal):
        raise ValueError("action does not match its canonical phase-qualified terminal form")
    return terminal


def _action_phase(action: Action) -> str:
    if not isinstance(action.parameters, Mapping):
        raise ValueError("action parameters must be a mapping")
    phase = action.parameters.get("phase")
    if not isinstance(phase, str):
        raise ValueError("action parameters lack a phase")
    return phase


def _visible_phase(experience: ExperienceEvent) -> str:
    decision = experience.decision
    if decision is None:
        raise ValueError("WM-002 assimilation requires an interaction decision")
    phase = _action_phase(decision.intended_action.action)
    payload = experience.observation.evidence.payload
    if not isinstance(payload, Mapping) or payload.get("phase") != phase:
        raise ValueError("observation phase does not match the selected action")
    return phase


def _require_composite_target(target: EpistemicTarget) -> None:
    if target.target_id != TARGET_ID or target.description != TARGET_DESCRIPTION or target.target_kind != TARGET_KIND:
        raise ValueError("epistemic target does not match the WM-002 composite target")


def _require_committed_acquisition_receipt(
    snapshot: AgentSnapshot,
) -> UpdateReceipt:
    receipt = snapshot.latest_update
    if (
        receipt is None
        or receipt.learner_version != LEARNER_VERSION
        or receipt.status is not UpdateStatus.APPLIED
        or len(receipt.transitions) != 1
        or receipt.transitions[0].experience.step_index != 0
        or receipt.new_model_version != snapshot.model_version
        or receipt.resulting_belief is not snapshot.belief
    ):
        raise ValueError("terminal assessment requires the committed acquisition receipt")
    return receipt


def _require_snapshot_model(snapshot: AgentSnapshot, state: ModelState) -> ExactPosteriorModel:
    validate_posterior_model_state(state)
    if snapshot.model_version != state.version:
        raise ValueError("snapshot model version differs from the live model owner")
    if snapshot.configuration_version != configuration_version_for_digest(state.digest):
        raise ValueError("snapshot configuration version does not bind the live model digest")
    decoded = decode_posterior_model(state.payload)
    direct = categorical_probabilities(snapshot.belief.distribution)[1]
    if not math.isclose(direct, float(decoded.posterior_direct), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("snapshot posterior disagrees with the exact owned model")
    return decoded


def _exact_posterior(
    prior_direct: Fraction,
    action: AcquisitionAction,
    observed_symbol: int,
) -> Fraction:
    oracle = FractionOracle()
    return oracle.posterior_direct(
        action,
        observed_symbol,
        prior_direct=prior_direct,
    )


def _fraction_entropy(probability: Fraction) -> float:
    return entropy((float(1 - probability), float(probability)))


def _observed_evidence(
    *,
    evidence_id: str,
    payload: object,
    at: TimePoint,
    source_id: str,
    source_kind: str,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        payload=payload,
        occurred_at=at,
        available_at=at,
        lineage=EvidenceLineage(
            evidence_id=evidence_id,
            origin=EvidenceOrigin.OBSERVED,
            provenance=Provenance(
                source_id=source_id,
                trust=TrustLevel.VERIFIED,
                source_kind=source_kind,
            ),
        ),
    )


def _required_score(row: CandidateDiagnosticRow) -> float:
    if row.arm_selection_score is None:
        raise DecisionError("ranked arm diagnostic is missing a selection score")
    return row.arm_selection_score


def _require_mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} has an invalid field set")
    return cast(Mapping[str, object], value)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    text = _required_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be lowercase SHA-256 hexadecimal")
    return text


def _nullable_identifier(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label)


def _offset(point: TimePoint, ticks: int) -> TimePoint:
    return TimePoint(tick=point.tick + ticks, clock_id=point.clock_id)


def _latest_time(*points: TimePoint) -> TimePoint:
    clocks = {point.clock_id for point in points}
    if len(clocks) != 1:
        raise DecisionError("decision inputs use different logical clocks")
    latest = max(points, key=lambda point: point.tick)
    return TimePoint(tick=latest.tick, clock_id=latest.clock_id)


def _terminal_label(action: TerminalAction) -> str:
    return "direct" if action is TerminalAction.DIRECT else "reversed"


__all__ = (
    "AGENT_ID",
    "AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS",
    "AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS",
    "AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS",
    "AGENT_VISIBLE_TERMINAL_ACTION_FIELDS",
    "AGENT_VISIBLE_TERMINAL_OBSERVATION_FIELDS",
    "AGENT_VISIBLE_TERMINAL_OUTCOME_FIELDS",
    "ASSIMILATOR_VERSION",
    "AcquisitionTerminalScorer",
    "ArmDecisionPolicy",
    "ArmMode",
    "CALIBRATION_VERSION",
    "CandidateDiagnosticRow",
    "CONFIGURATION_VERSION_PREFIX",
    "EFFECT_VERSION",
    "EpisodeRuntimeComposition",
    "RestoredTerminalRuntimeComposition",
    "ExactPosteriorModel",
    "ExactPosteriorTransactionalLearner",
    "HiddenActuatorAcquisitionEnvironment",
    "HiddenActuatorCandidateAssessor",
    "HiddenActuatorTerminalEnvironment",
    "LEARNER_VERSION",
    "LIKELIHOOD_VERSION",
    "MODEL_VERSION_PREFIX",
    "NoOpAssimilationEffect",
    "NoOpObservationAssimilator",
    "POSTERIOR_MODEL_SCHEMA",
    "REPRESENTATION_VERSION",
    "SCORER_VERSION",
    "TASK_ID",
    "TARGET_DESCRIPTION",
    "TARGET_ID",
    "TARGET_KIND",
    "TerminalCandidateAssessor",
    "acquisition_public_probe_bytes",
    "assert_acquisition_public_noninterference",
    "candidate_diagnostic_bytes",
    "candidate_diagnostic_digest",
    "compose_episode_runtime",
    "compose_restored_terminal_runtime",
    "configuration_version_for_digest",
    "decode_posterior_model",
    "encode_posterior_model",
    "initial_posterior_model_state",
    "model_version_for_payload",
    "validate_agent_visible_environment_step",
    "validate_posterior_model_state",
)
