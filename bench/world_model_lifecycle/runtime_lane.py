"""WM-001 integration with Prospect's authoritative real-experience lifecycle.

The neural backend and planner remain optional benchmark dependencies.  This
module translates their numeric inputs and outputs into Prospect's immutable
domain records, drives every real Gymnasium step through ``EpistemicAgent``, and
exposes the exact canonical transitions used by learning.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

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
    Observation,
    Outcome,
    Prediction,
    ProperScore,
    Provenance,
    ResourceLedger,
    TimePoint,
    TrustLevel,
    UncertaintyEstimate,
    UncertaintyKind,
    Utility,
)
from prospect.runtime import (
    AgentState,
    EnvironmentStep,
    EpistemicAgent,
    InteractionContext,
    InteractionResult,
)
from prospect.runtime.learning import OwnedModel, TransactionalLearner
from prospect.storage import EpistemicLedger, InMemoryExperienceStore

AGENT_ID = "prospect-wm001-agent"
REPRESENTATION_VERSION = "wm001-physical-state-v1"
POLICY_VERSION = "wm001-controller-policy-v1"
CALIBRATION_VERSION = "wm001-uncalibrated-v1"
TARGET = EpistemicTarget(
    target_id="wm001:next-physical-observation",
    description="next Pendulum physical observation conditional on intended torque and observed task context",
    target_kind="one_step_environment_state",
)
TASK_CONTEXTS = {
    "pendulum_normal_torque": 0.0,
    "pendulum_reversed_torque": 1.0,
    "independent_phase_oscillator": 2.0,
}
INDEPENDENT_OSCILLATOR_TASK = "independent_phase_oscillator"
INDEPENDENT_OSCILLATOR_SOURCE = "prospect:IndependentPhaseOscillator-v1"


class PredictiveBackend(Protocol):
    """Numeric prediction view used at the decision boundary."""

    @property
    def version(self) -> str: ...

    @property
    def digest(self) -> str: ...

    def predict_ensemble(
        self,
        observation: np.ndarray,
        context: float,
        action: float,
    ) -> tuple[object, object]: ...


class Controller(Protocol):
    """Choose one intended torque and report its predicted trajectory value."""

    @property
    def version(self) -> str: ...

    def select(self, observation: np.ndarray, context: float) -> tuple[float, float]: ...


class _NoLearning:
    """Sentinel accepted by evaluation agents that never invoke ``learn``."""

    def update(self, snapshot: AgentSnapshot, transitions: Sequence[EpistemicTransition]) -> object:
        raise RuntimeError("learning is disabled for this evaluation agent")


class _UnusedRuntimeComponent:
    """A branch learner never calls decision/observation components."""


class CanonicalReplayIndex:
    """Lossless write-through index over canonical real ExperienceEvent objects."""

    def __init__(self) -> None:
        self._events_by_id: dict[str, ExperienceEvent] = {}
        self._ordered_ids: list[str] = []

    def add(self, event: ExperienceEvent) -> None:
        existing = self._events_by_id.get(event.experience_id)
        if existing is not None:
            if existing is event:
                raise RuntimeError("canonical replay event was indexed twice")
            raise RuntimeError("replay experience identity collision")
        self._events_by_id[event.experience_id] = event
        self._ordered_ids.append(event.experience_id)

    def require_transitions(
        self,
        transitions: Sequence[EpistemicTransition],
    ) -> tuple[EpistemicTransition, ...]:
        resolved: list[EpistemicTransition] = []
        for transition in transitions:
            event = self._events_by_id.get(transition.experience.experience_id)
            if event is None:
                raise RuntimeError(f"transition {transition.transition_id!r} has no canonical replay event")
            if event is not transition.experience:
                raise RuntimeError("replay index does not reference the canonical experience object")
            resolved.append(transition)
        return tuple(resolved)

    @property
    def event_count(self) -> int:
        return len(self._ordered_ids)

    def get(self, experience_id: str) -> ExperienceEvent:
        """Return the exact canonical replay event for identity checks."""

        try:
            return self._events_by_id[experience_id]
        except KeyError as error:
            raise KeyError(f"canonical replay has no experience {experience_id!r}") from error

    def events(self) -> tuple[ExperienceEvent, ...]:
        """Return all canonical events in replay insertion order."""

        return tuple(self._events_by_id[event_id] for event_id in self._ordered_ids)

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "experience_id": event.experience_id,
                "run_id": event.run_id,
                "task_id": event.task_id,
                "episode_id": event.episode_id,
                "step_index": event.step_index,
                "closed_at": [event.closed_at.clock_id, event.closed_at.tick],
            }
            for event_id in self._ordered_ids
            for event in (self._events_by_id[event_id],)
        ]


@dataclass(frozen=True, slots=True)
class RuntimeCustody:
    """Shared canonical custody for a collection phase."""

    store: InMemoryExperienceStore
    ledger: EpistemicLedger
    identities: CounterIdentitySource
    replay: CanonicalReplayIndex

    @classmethod
    def create(cls, namespace: str) -> RuntimeCustody:
        store = InMemoryExperienceStore()
        return cls(
            store=store,
            ledger=EpistemicLedger(store),
            identities=CounterIdentitySource(namespace),
            replay=CanonicalReplayIndex(),
        )

    @classmethod
    def branch(
        cls,
        source: RuntimeCustody,
        namespace: str,
        transitions: Sequence[EpistemicTransition],
    ) -> RuntimeCustody:
        """Create an isolated control ledger over exact canonical source records."""

        ledger = EpistemicLedger(source.store)
        replay = CanonicalReplayIndex()
        for transition in transitions:
            ledger.append_transition(transition)
            replay.add(transition.experience)
        return cls(
            store=source.store,
            ledger=ledger,
            identities=CounterIdentitySource(namespace),
            replay=replay,
        )


@dataclass(frozen=True, slots=True)
class EpisodeEvidence:
    """Real executed episode plus the records needed for audit and learning."""

    episode_id: str
    task_id: str
    reset_seed: int
    undiscounted_return: float
    transitions: tuple[EpistemicTransition, ...]
    intended_actions: tuple[float, ...]
    applied_actions: tuple[float, ...]
    prediction_ids: tuple[str, ...]
    final_tick: int
    agent: EpistemicAgent


@dataclass(frozen=True, slots=True)
class CollectionEvidence:
    """Several whole episodes sharing one persistent model and canonical ledger."""

    run_id: str
    task_id: str
    episodes: tuple[EpisodeEvidence, ...]
    transitions: tuple[EpistemicTransition, ...]
    custody: RuntimeCustody
    final_agent: EpistemicAgent
    next_tick: int


class UniformRandomController:
    """Seeded, checkpointable uniform exploratory or control policy."""

    def __init__(self, seed: int, *, version: str = "wm001-uniform-random-v1") -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._version = version
        self._actions_emitted = 0

    @property
    def version(self) -> str:
        return self._version

    def select(self, observation: np.ndarray, context: float) -> tuple[float, float]:
        del observation, context
        action = float(self._rng.uniform(-2.0, 2.0))
        self._actions_emitted += 1
        return action, 0.0

    def state(self) -> dict[str, object]:
        return cast(dict[str, object], self._rng.bit_generator.state)

    @property
    def actions_emitted(self) -> int:
        return self._actions_emitted

    @property
    def rng_digest(self) -> str:
        payload = json.dumps(
            self.state(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class PresetActionController:
    """One-shot controller used when planning happens outside the runtime step.

    The caller presets exactly one action/value pair immediately before
    ``EpistemicAgent.interact``.  ``select`` consumes it, so a missing plan or
    accidental reuse cannot silently repeat a previous action.
    """

    def __init__(self, *, version: str = "wm001-external-preset-controller-v1") -> None:
        if not version or not version.strip():
            raise ValueError("controller version must be nonempty")
        self._version = version
        self._selection: tuple[float, float] | None = None

    @property
    def version(self) -> str:
        return self._version

    def preset(self, action: float, predicted_value: float = 0.0) -> None:
        """Stage one bounded Pendulum torque and its planner-reported value."""

        torque = float(action)
        value = float(predicted_value)
        if not math.isfinite(torque) or not -2.0 <= torque <= 2.0:
            raise ValueError("preset action must be finite and within [-2, 2]")
        if not math.isfinite(value):
            raise ValueError("preset predicted value must be finite")
        if self._selection is not None:
            raise RuntimeError("a preset action is already waiting to be consumed")
        self._selection = (torque, value)

    def select(self, observation: np.ndarray, context: float) -> tuple[float, float]:
        del observation, context
        selection = self._selection
        if selection is None:
            raise RuntimeError("external controller has no preset action")
        self._selection = None
        return selection

    def discard(self) -> None:
        """Discard an unconsumed preset after a failed interaction."""

        self._selection = None


class _WorldModelCandidateAssessor:
    def __init__(
        self,
        *,
        backend: PredictiveBackend,
        controller: Controller,
        identities: CounterIdentitySource,
    ) -> None:
        self._backend = backend
        self._controller = controller
        self._identities = identities

    def assess(self, snapshot: AgentSnapshot, goal: Goal) -> Sequence[CandidateAssessment]:
        if snapshot.model_version != self._backend.version:
            raise RuntimeError("decision snapshot and live predictive backend have different model versions")
        context = _task_context(goal.task_id)
        observation = _belief_observation(snapshot.belief)
        action_value, plan_value = self._controller.select(observation.copy(), context)
        intended_torque = float(np.clip(action_value, -2.0, 2.0))
        member_means_raw, member_variances_raw = self._backend.predict_ensemble(
            observation.copy(),
            context,
            intended_torque,
        )
        member_means = np.asarray(member_means_raw, dtype=np.float64)
        member_variances = np.asarray(member_variances_raw, dtype=np.float64)
        if member_means.ndim != 2 or member_means.shape[1] != 3:
            raise ValueError("predictive backend means must have shape [ensemble, 3]")
        if member_variances.shape != member_means.shape:
            raise ValueError("predictive backend variances must match the member means")
        if not np.isfinite(member_means).all() or not np.isfinite(member_variances).all():
            raise ValueError("predictive backend returned non-finite values")
        if np.any(member_variances <= 0.0):
            raise ValueError("predictive backend variances must be positive")

        action = Action(
            action_id=self._identities.next("action"),
            action_kind="continuous_torque",
            parameters={
                "intended_torque": intended_torque,
                "task_context": context,
                "parameter_digest_at_action": self._backend.digest,
            },
        )
        issued_at = snapshot.captured_at
        predictive_mean = member_means.mean(axis=0)
        total_variance = np.mean(member_variances + np.square(member_means), axis=0) - np.square(predictive_mean)
        total_variance = np.maximum(total_variance, 1e-12)
        aleatoric = float(np.mean(member_variances))
        epistemic = float(np.mean(np.var(member_means, axis=0)))
        uncertainties = (
            UncertaintyEstimate(
                estimate_id=self._identities.next("aleatoric-uncertainty"),
                kind=UncertaintyKind.ALEATORIC,
                measure="mean_member_variance",
                value=aleatoric,
                unit="physical_state_squared",
                target_id=goal.target.target_id,
                estimator_version="wm001-ensemble-decomposition-v1",
                assessed_at=issued_at,
            ),
            UncertaintyEstimate(
                estimate_id=self._identities.next("epistemic-uncertainty"),
                kind=UncertaintyKind.EPISTEMIC,
                measure="between_member_mean_variance",
                value=epistemic,
                unit="physical_state_squared",
                target_id=goal.target.target_id,
                estimator_version="wm001-ensemble-decomposition-v1",
                assessed_at=issued_at,
            ),
        )
        prediction = Prediction(
            prediction_id=self._identities.next("prediction"),
            prior_belief=snapshot.belief,
            action=action,
            target=goal.target,
            distribution=Distribution(
                distribution_id=self._identities.next("prediction-distribution"),
                family="equal_weight_diagonal_gaussian_mixture",
                support="R^3",
                parameters={
                    "member_means": tuple(tuple(float(value) for value in row) for row in member_means),
                    "member_variances": tuple(tuple(float(value) for value in row) for row in member_variances),
                    "mixture_mean": tuple(float(value) for value in predictive_mean),
                    "mixture_variance": tuple(float(value) for value in total_variance),
                    "parameter_digest": self._backend.digest,
                },
                representation_version=REPRESENTATION_VERSION,
                event_shape=(3,),
            ),
            issued_at=issued_at,
            horizon_end=TimePoint(issued_at.tick + 1, issued_at.clock_id),
            model_version=snapshot.model_version,
            representation_version=REPRESENTATION_VERSION,
            calibration_version=CALIBRATION_VERSION,
            uncertainties=uncertainties,
        )
        utility = Utility(
            utility_id=self._identities.next("utility"),
            goal_id=goal.goal_id,
            prediction_id=prediction.prediction_id,
            expected_value=float(plan_value),
            unit="predicted_environment_return",
            evaluator_version=self._controller.version,
            assessed_at=issued_at,
        )
        information = InformationValue(
            information_value_id=self._identities.next("information-value"),
            prior_belief_id=snapshot.belief.belief_id,
            action_id=action.action_id,
            target_id=goal.target.target_id,
            expected_reduction=0.0,
            expected_cost=0.0,
            unit="predicted_environment_return",
            evaluator_version="wm001-no-intrinsic-reward-v1",
            assessed_at=issued_at,
        )
        assessment = CandidateAssessment(
            assessment_id=self._identities.next("assessment"),
            action=action,
            prediction=prediction,
            utility=utility,
            information_value=information,
            expected_action_cost=0.0,
            expected_risk=0.0,
            admissible=True,
            constraint_reasons=(),
            constraint_penalty=0.0,
            total_value=float(plan_value),
            unit="predicted_environment_return",
            evaluator_version=f"wm001-assessor:{self._controller.version}",
            assessed_at=issued_at,
        )
        return (assessment,)


class _PhysicalBeliefUpdater:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def assimilate(self, prior: Belief, experience: object) -> BeliefUpdate:
        event = cast(Any, experience)
        next_observation = _event_observation(event)
        context = float(event.observation.evidence.payload["task_context"])
        updated_at = event.closed_at
        observations = (*prior.information_set.observations, event.observation)
        posterior = Belief(
            belief_id=self._identities.next("belief"),
            agent_id=prior.agent_id,
            target=prior.target,
            information_set=InformationSet(
                information_set_id=self._identities.next("information-set"),
                agent_id=prior.agent_id,
                as_of=updated_at,
                observations=observations,
                memory_version=f"wm001-memory:{len(observations)}",
            ),
            distribution=_state_distribution(
                self._identities,
                next_observation,
                context,
            ),
            formed_at=updated_at,
            model_version=prior.model_version,
            representation_version=prior.representation_version,
        )
        return BeliefUpdate(
            update_id=self._identities.next("belief-update"),
            prior=prior,
            experience=event,
            posterior=posterior,
            updater_version="wm001-observed-state-assimilation-v1",
            updated_at=updated_at,
        )


class _MixtureScorer:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def score(self, prediction: Prediction, experience: object) -> ProperScore:
        event = cast(Any, experience)
        realized = _event_observation(event)
        parameters = cast(dict[str, object], prediction.distribution.parameters)
        means = np.asarray(parameters["member_means"], dtype=np.float64)
        variances = np.asarray(parameters["member_variances"], dtype=np.float64)
        component_log_prob = -0.5 * np.sum(
            np.log(2.0 * math.pi * variances) + np.square(realized[None, :] - means) / variances,
            axis=1,
        )
        maximum = float(component_log_prob.max())
        mixture_log_prob = maximum + math.log(float(np.exp(component_log_prob - maximum).mean()))
        return ProperScore(
            score_id=self._identities.next("proper-score"),
            prediction_id=prediction.prediction_id,
            realized_evidence_id=event.observation.evidence.evidence_id,
            rule="diagonal-gaussian-mixture-negative-log-likelihood",
            value=-mixture_log_prob / 3.0,
            unit="nats_per_physical_dimension",
            scorer_version="wm001-mixture-nll-v1",
            scored_at=event.closed_at,
        )


class _UncalibratedEffectAssessor:
    def __init__(self, identities: CounterIdentitySource) -> None:
        self._identities = identities

    def effect(self, update: BeliefUpdate) -> EpistemicEffect:
        return EpistemicEffect(
            effect_id=self._identities.next("effect"),
            belief_update_id=update.update_id,
            target_id=update.prior.target.target_id,
            kind=EpistemicEffectKind.PREDICTIVE_RISK_CHANGE,
            measure="unassessed_at_assimilation",
            before=0.0,
            after=0.0,
            improvement=0.0,
            higher_is_better=False,
            evaluator_version="wm001-no-posthoc-effect-claim-v1",
            evaluated_at=update.updated_at,
            externally_calibrated=False,
        )


class IndependentPhaseOscillatorEnvironment:
    """Deterministic action-independent nuisance process for a causal control.

    The reset seed selects an initial phase and angular velocity.  Thereafter
    the oscillator advances autonomously for exactly 200 steps:

    ``phase[t+1] = wrap(phase[t] + STEP_SECONDS * velocity)``

    Its observation is ``[cos(phase), sin(phase), velocity]`` and its reward is
    ``cos(phase)`` after the transition.  The action argument is deliberately
    ignored, making the process causally independent of intended torque while
    retaining the same observation/reward shapes as the two Pendulum tasks.
    """

    STEP_SECONDS = 0.05
    HORIZON = 200

    def __init__(self) -> None:
        self._phase: float | None = None
        self._velocity: float | None = None
        self._step_index = 0
        self._closed = False

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("independent oscillator reset requires a nonnegative integer seed")
        digest = hashlib.sha256(f"{INDEPENDENT_OSCILLATOR_SOURCE}:{seed}".encode("ascii")).digest()
        phase_unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        velocity_unit = int.from_bytes(digest[8:16], "big") / float(1 << 64)
        self._phase = (2.0 * phase_unit - 1.0) * math.pi
        self._velocity = 0.5 + velocity_unit
        self._step_index = 0
        self._closed = False
        return self._observation(), {"reset_seed": seed}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        if self._closed:
            raise RuntimeError("cannot step a closed independent oscillator")
        if self._phase is None or self._velocity is None:
            raise RuntimeError("independent oscillator must be reset before stepping")
        if self._step_index >= self.HORIZON:
            raise RuntimeError("independent oscillator cannot step past its 200-step horizon")
        self._phase = math.remainder(
            self._phase + self.STEP_SECONDS * self._velocity,
            2.0 * math.pi,
        )
        self._step_index += 1
        observation = self._observation()
        reward = float(math.cos(self._phase))
        return (
            observation,
            reward,
            False,
            self._step_index == self.HORIZON,
            {"step_index": self._step_index},
        )

    def close(self) -> None:
        self._closed = True

    def _observation(self) -> np.ndarray:
        if self._phase is None or self._velocity is None:
            raise RuntimeError("independent oscillator has no state before reset")
        return np.asarray(
            [math.cos(self._phase), math.sin(self._phase), self._velocity],
            dtype=np.float64,
        )


def run_independent_phase_oscillator_conformance(
    *,
    cases: int = 512,
    seed: int = 20260718,
) -> dict[str, object]:
    """Exercise deterministic reset, dynamics, horizon, and action independence."""

    if cases < 1 or isinstance(cases, bool):
        raise ValueError("oscillator conformance requires at least one case")
    if seed < 0 or isinstance(seed, bool):
        raise ValueError("oscillator conformance seed must be nonnegative")
    trajectory = hashlib.sha256()
    max_reset_difference = 0.0
    max_observation_difference = 0.0
    max_reward_difference = 0.0
    premature_or_missing_truncations = 0
    unexpected_terminations = 0
    for case_index in range(cases):
        reset_seed = seed + case_index
        negative = IndependentPhaseOscillatorEnvironment()
        positive = IndependentPhaseOscillatorEnvironment()
        negative_reset, _ = negative.reset(seed=reset_seed)
        positive_reset, _ = positive.reset(seed=reset_seed)
        max_reset_difference = max(
            max_reset_difference,
            float(np.max(np.abs(negative_reset - positive_reset), initial=0.0)),
        )
        trajectory.update(reset_seed.to_bytes(8, "big", signed=False))
        trajectory.update(np.asarray(negative_reset, dtype="<f8").tobytes())
        for step_index in range(IndependentPhaseOscillatorEnvironment.HORIZON):
            negative_row = negative.step(np.asarray([-2.0], dtype=np.float64))
            positive_row = positive.step(np.asarray([2.0], dtype=np.float64))
            negative_observation, negative_reward, negative_terminated, negative_truncated, _ = negative_row
            positive_observation, positive_reward, positive_terminated, positive_truncated, _ = positive_row
            max_observation_difference = max(
                max_observation_difference,
                float(
                    np.max(
                        np.abs(negative_observation - positive_observation),
                        initial=0.0,
                    )
                ),
            )
            max_reward_difference = max(
                max_reward_difference,
                abs(float(negative_reward) - float(positive_reward)),
            )
            unexpected_terminations += int(negative_terminated or positive_terminated)
            expected_truncation = step_index == IndependentPhaseOscillatorEnvironment.HORIZON - 1
            premature_or_missing_truncations += int(
                negative_truncated is not expected_truncation or positive_truncated is not expected_truncation
            )
            trajectory.update(np.asarray(negative_observation, dtype="<f8").tobytes())
            trajectory.update(np.asarray([negative_reward], dtype="<f8").tobytes())
            trajectory.update(bytes((int(negative_terminated), int(negative_truncated))))
        negative.close()
        positive.close()
    report: dict[str, object] = {
        "schema": "prospect.wm001.independent-phase-oscillator-conformance.v1",
        "source_id": INDEPENDENT_OSCILLATOR_SOURCE,
        "cases": cases,
        "steps_per_case": IndependentPhaseOscillatorEnvironment.HORIZON,
        "seed": seed,
        "max_reset_absolute_difference": max_reset_difference,
        "max_action_pair_observation_absolute_difference": max_observation_difference,
        "max_action_pair_reward_absolute_difference": max_reward_difference,
        "unexpected_terminations": unexpected_terminations,
        "premature_or_missing_truncations": premature_or_missing_truncations,
        "trajectory_sha256": trajectory.hexdigest(),
        "passed": (
            max_reset_difference == 0.0
            and max_observation_difference == 0.0
            and max_reward_difference == 0.0
            and unexpected_terminations == 0
            and premature_or_missing_truncations == 0
        ),
    }
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


class PendulumRuntimeEnvironment:
    """Turn one task-aware WM-001 environment step into canonical records."""

    def __init__(
        self,
        *,
        environment: object,
        agent_id: str,
        task_id: str,
        identities: CounterIdentitySource,
    ) -> None:
        self._environment = environment
        self._agent_id = agent_id
        self._task_id = task_id
        self._context = _task_context(task_id)
        self._identities = identities
        if task_id == INDEPENDENT_OSCILLATOR_TASK:
            self._modality = "independent_phase_oscillator_state"
            self._state_source_id = INDEPENDENT_OSCILLATOR_SOURCE
            self._state_source_kind = "simulated_distractor_environment"
            self._reward_source_id = INDEPENDENT_OSCILLATOR_SOURCE
            self._reward_source_kind = "simulated_distractor_environment_reward"
        else:
            self._modality = "gymnasium_pendulum_state"
            self._state_source_id = "gymnasium:Pendulum-v1"
            self._state_source_kind = "simulated_environment"
            self._reward_source_id = "gymnasium:Pendulum-v1"
            self._reward_source_kind = "simulated_environment_reward"

    def step(self, intention: object) -> EnvironmentStep:
        intended = cast(Any, intention)
        parameters = cast(dict[str, object], intended.action.parameters)
        command = float(parameters["intended_torque"])
        context = float(parameters["task_context"])
        if context != self._context:
            raise RuntimeError("intended action carries the wrong observed task context")
        if self._task_id == INDEPENDENT_OSCILLATOR_TASK:
            applied = 0.0
        else:
            applied = command if context == 0.0 else -command
        next_observation, reward, terminated, truncated, _ = cast(Any, self._environment).step(
            np.asarray([applied], dtype=np.float32)
        )
        started = TimePoint(intended.intended_at.tick + 1, intended.intended_at.clock_id)
        ended = TimePoint(started.tick + 1, started.clock_id)
        execution = ExecutedAction(
            execution_id=self._identities.next("execution"),
            intention=intended,
            status=ExecutionStatus.SUCCEEDED,
            started_at=started,
            ended_at=ended,
            realized_action=intended.action,
        )
        observation_id = self._identities.next("observation")
        observation = Observation(
            observation_id=observation_id,
            agent_id=self._agent_id,
            modality=self._modality,
            evidence=Evidence(
                evidence_id=observation_id,
                payload={
                    "physical_observation": tuple(float(value) for value in next_observation),
                    "task_context": context,
                    "task_id": self._task_id,
                },
                occurred_at=ended,
                available_at=ended,
                lineage=EvidenceLineage(
                    evidence_id=observation_id,
                    origin=EvidenceOrigin.OBSERVED,
                    provenance=Provenance(
                        source_id=self._state_source_id,
                        trust=TrustLevel.VERIFIED,
                        source_kind=self._state_source_kind,
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
                payload={
                    "reward": float(reward),
                    "intended_torque": command,
                    "applied_torque": applied,
                    "task_context": context,
                },
                occurred_at=ended,
                available_at=ended,
                lineage=EvidenceLineage(
                    evidence_id=outcome_evidence_id,
                    origin=EvidenceOrigin.OBSERVED,
                    provenance=Provenance(
                        source_id=self._reward_source_id,
                        trust=TrustLevel.VERIFIED,
                        source_kind=self._reward_source_kind,
                    ),
                ),
            ),
        )
        return EnvironmentStep(
            execution=execution,
            observation=observation,
            outcome=outcome,
            closed_at=ended,
            terminated=bool(terminated),
            truncated=bool(truncated),
            discount=1.0,
        )


class PendulumEpisodeSession:
    """One live Pendulum episode with externally supplied actions.

    Sessions may share ``RuntimeCustody`` and be stepped in any interleaving.
    The caller owns the global logical clock and must pass ticks that keep
    shared custody append-only.  Every accepted action still travels through
    the normal assessor → policy → ``EpistemicAgent.interact`` path.
    """

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        episode_id: str,
        reset_seed: int,
        backend: PredictiveBackend,
        custody: RuntimeCustody | None = None,
        start_tick: int = 0,
        model_owner: OwnedModel | None = None,
        learner: TransactionalLearner | None = None,
        controller: PresetActionController | None = None,
    ) -> None:
        if start_tick < 0:
            raise ValueError("start_tick must be nonnegative")
        self._run_id = run_id
        self._task_id = task_id
        self._episode_id = episode_id
        self._reset_seed = int(reset_seed)
        self._context = _task_context(task_id)
        self._custody = custody or RuntimeCustody.create(f"{run_id}:{episode_id}")
        self._controller = controller or PresetActionController()
        (
            self._environment,
            self._runtime_environment,
            self._agent,
            self._current_observation,
        ) = _initialize_episode_runtime(
            run_id=run_id,
            task_id=task_id,
            episode_id=episode_id,
            reset_seed=self._reset_seed,
            controller=self._controller,
            backend=backend,
            custody=self._custody,
            start_tick=start_tick,
            model_owner=model_owner,
            learner=learner,
        )
        self._transitions: list[EpistemicTransition] = []
        self._intended_actions: list[float] = []
        self._applied_actions: list[float] = []
        self._predictions: list[str] = []
        self._total_return = 0.0
        self._last_tick = start_tick
        self._done = False
        self._closed = False
        self._evidence: EpisodeEvidence | None = None

    @property
    def current_observation(self) -> np.ndarray:
        """Return a copy of the latest authoritative physical observation."""

        return self._current_observation.copy()

    @property
    def context(self) -> float:
        """Return the observed task context supplied to the world model."""

        return self._context

    @property
    def step_index(self) -> int:
        return len(self._transitions)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def agent(self) -> EpistemicAgent:
        return self._agent

    @property
    def custody(self) -> RuntimeCustody:
        return self._custody

    def step(
        self,
        action: float,
        tick: int,
        *,
        predicted_value: float = 0.0,
    ) -> InteractionResult:
        """Execute one externally planned action at a caller-owned global tick."""

        if self._closed:
            raise RuntimeError("cannot step a closed episode session")
        if self._done:
            raise RuntimeError("cannot step a completed episode session")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < self._last_tick:
            raise ValueError("tick must be an integer not before this session's last causal tick")
        if self.step_index >= 200:
            raise RuntimeError("WM-001 sessions cannot exceed 200 accepted actions")
        self._controller.preset(action, predicted_value)
        at = TimePoint(tick)
        goal = Goal(
            goal_id=self._custody.identities.next("goal"),
            task_id=self._task_id,
            target=TARGET,
            description="maximize undiscounted Pendulum-v1 return using the committed world model",
            issued_at=at,
            preference_version="wm001-pendulum-return-v1",
        )
        try:
            result = self._agent.interact(
                self._runtime_environment,
                goal,
                context=InteractionContext(
                    run_id=self._run_id,
                    task_id=self._task_id,
                    episode_id=self._episode_id,
                    step_index=self.step_index,
                ),
                decide_at=at,
            )
        except BaseException:
            self._controller.discard()
            raise
        self._transitions.append(result.transition)
        action_payload = cast(dict[str, object], result.decision.intended_action.action.parameters)
        outcome_payload = cast(dict[str, object], result.experience.outcome.evidence.payload)
        self._intended_actions.append(float(action_payload["intended_torque"]))
        self._applied_actions.append(float(outcome_payload["applied_torque"]))
        self._total_return += float(outcome_payload["reward"])
        self._predictions.append(result.decision.selected_assessment.prediction.prediction_id)
        self._current_observation = _event_observation(result.experience)
        self._last_tick = result.transition.created_at.tick
        self._done = bool(result.experience.terminated or result.experience.truncated)
        if self._done and (self.step_index != 200 or result.experience.terminated or not result.experience.truncated):
            raise RuntimeError("WM-001 requires exactly 200 accepted actions and a TimeLimit truncation")
        return result

    def finish(self) -> EpisodeEvidence:
        """Close a completed session and return its immutable episode evidence."""

        if self._evidence is not None:
            return self._evidence
        if not self._done or self.step_index != 200 or not self._transitions[-1].experience.truncated:
            raise RuntimeError("cannot finish an incomplete WM-001 episode session")
        if not self._closed:
            cast(Any, self._environment).close()
            self._closed = True
        self._evidence = EpisodeEvidence(
            episode_id=self._episode_id,
            task_id=self._task_id,
            reset_seed=self._reset_seed,
            undiscounted_return=self._total_return,
            transitions=tuple(self._transitions),
            intended_actions=tuple(self._intended_actions),
            applied_actions=tuple(self._applied_actions),
            prediction_ids=tuple(self._predictions),
            final_tick=self._transitions[-1].created_at.tick,
            agent=self._agent,
        )
        return self._evidence

    def close(self) -> EpisodeEvidence:
        """Alias for ``finish`` for explicit resource-lifecycle call sites."""

        return self.finish()


def _initialize_episode_runtime(
    *,
    run_id: str,
    task_id: str,
    episode_id: str,
    reset_seed: int,
    controller: Controller,
    backend: PredictiveBackend,
    custody: RuntimeCustody,
    start_tick: int,
    model_owner: OwnedModel | None,
    learner: TransactionalLearner | None,
) -> tuple[object, PendulumRuntimeEnvironment, EpistemicAgent, np.ndarray]:
    del run_id, episode_id
    if task_id == INDEPENDENT_OSCILLATOR_TASK:
        environment: object = IndependentPhaseOscillatorEnvironment()
    else:
        import gymnasium as gym

        environment = gym.make("Pendulum-v1")
    reset_observation, _ = cast(Any, environment).reset(seed=int(reset_seed))
    initial_observation = np.asarray(reset_observation, dtype=np.float64)
    initial = initial_snapshot(
        observation=initial_observation,
        context=_task_context(task_id),
        model_version=backend.version,
        identities=custody.identities,
        at=TimePoint(start_tick),
        task_id=task_id,
    )
    assessor = _WorldModelCandidateAssessor(
        backend=backend,
        controller=controller,
        identities=custody.identities,
    )
    policy = MaxValuePolicy(
        agent_id=AGENT_ID,
        policy_version=POLICY_VERSION,
        assessor=assessor,
        identities=custody.identities,
    )
    agent = EpistemicAgent(
        state=AgentState(initial),
        policy=policy,
        belief_updater=_PhysicalBeliefUpdater(custody.identities),
        scorer=_MixtureScorer(custody.identities),
        effect_assessor=_UncalibratedEffectAssessor(custody.identities),
        learner=learner if learner is not None else cast(Any, _NoLearning()),
        experience_store=custody.store,
        ledger=custody.ledger,
        identities=custody.identities,
        model=model_owner,
        replay=(custody.replay if model_owner is not None else None),
    )
    runtime_environment = PendulumRuntimeEnvironment(
        environment=environment,
        agent_id=AGENT_ID,
        task_id=task_id,
        identities=custody.identities,
    )
    return environment, runtime_environment, agent, initial_observation


def collect_episodes(
    *,
    run_id: str,
    task_id: str,
    episode_seeds: Sequence[int],
    controller_factory: Callable[[int], Controller],
    backend: PredictiveBackend,
    custody: RuntimeCustody,
    start_tick: int = 0,
    model_owner: OwnedModel | None = None,
    learner: TransactionalLearner | None = None,
) -> CollectionEvidence:
    """Collect whole reset episodes through one shared canonical custody chain."""

    if not episode_seeds:
        raise ValueError("collection requires at least one whole episode")
    if (model_owner is None) != (learner is None):
        raise ValueError("transactional collection requires both model_owner and learner")
    tick = start_tick
    episodes: list[EpisodeEvidence] = []
    all_transitions: list[EpistemicTransition] = []
    final_agent: EpistemicAgent | None = None
    for episode_index, seed in enumerate(episode_seeds):
        controller = controller_factory(seed)
        episode, final_agent = run_episode(
            run_id=run_id,
            task_id=task_id,
            episode_id=f"{run_id}:{task_id}:episode:{episode_index}",
            reset_seed=seed,
            controller=controller,
            backend=backend,
            custody=custody,
            start_tick=tick,
            model_owner=model_owner,
            learner=learner,
        )
        episodes.append(episode)
        all_transitions.extend(episode.transitions)
        tick = episode.final_tick + 1
    assert final_agent is not None
    return CollectionEvidence(
        run_id=run_id,
        task_id=task_id,
        episodes=tuple(episodes),
        transitions=tuple(all_transitions),
        custody=custody,
        final_agent=final_agent,
        next_tick=tick,
    )


def make_branch_learning_agent(
    *,
    source_agent: EpistemicAgent,
    at: TimePoint,
    model_owner: OwnedModel,
    learner: TransactionalLearner,
    custody: RuntimeCustody,
) -> EpistemicAgent:
    """Fork runtime state at an episode boundary for a controlled learner arm."""

    snapshot = source_agent.snapshot(at)
    return EpistemicAgent(
        state=AgentState(snapshot),
        policy=cast(Any, _UnusedRuntimeComponent()),
        belief_updater=cast(Any, _UnusedRuntimeComponent()),
        scorer=cast(Any, _UnusedRuntimeComponent()),
        effect_assessor=cast(Any, _UnusedRuntimeComponent()),
        learner=learner,
        experience_store=custody.store,
        ledger=custody.ledger,
        identities=custody.identities,
        model=model_owner,
        replay=custody.replay,
    )


def run_episode(
    *,
    run_id: str,
    task_id: str,
    episode_id: str,
    reset_seed: int,
    controller: Controller,
    backend: PredictiveBackend,
    custody: RuntimeCustody | None = None,
    start_tick: int = 0,
    model_owner: OwnedModel | None = None,
    learner: TransactionalLearner | None = None,
) -> tuple[EpisodeEvidence, EpistemicAgent]:
    """Execute exactly one 200-step Gymnasium episode through Prospect."""

    active_custody = custody or RuntimeCustody.create(f"{run_id}:{episode_id}")
    environment, runtime_environment, agent, _ = _initialize_episode_runtime(
        run_id=run_id,
        task_id=task_id,
        episode_id=episode_id,
        reset_seed=reset_seed,
        controller=controller,
        backend=backend,
        custody=active_custody,
        start_tick=start_tick,
        model_owner=model_owner,
        learner=learner,
    )
    transitions: list[EpistemicTransition] = []
    intended_actions: list[float] = []
    applied_actions: list[float] = []
    predictions: list[str] = []
    total_return = 0.0
    tick = start_tick
    try:
        for step_index in range(200):
            at = TimePoint(tick)
            goal = Goal(
                goal_id=active_custody.identities.next("goal"),
                task_id=task_id,
                target=TARGET,
                description="maximize undiscounted Pendulum-v1 return using the committed world model",
                issued_at=at,
                preference_version="wm001-pendulum-return-v1",
            )
            result = agent.interact(
                runtime_environment,
                goal,
                context=InteractionContext(
                    run_id=run_id,
                    task_id=task_id,
                    episode_id=episode_id,
                    step_index=step_index,
                ),
                decide_at=at,
            )
            transitions.append(result.transition)
            action_payload = cast(dict[str, object], result.decision.intended_action.action.parameters)
            outcome_payload = cast(dict[str, object], result.experience.outcome.evidence.payload)
            intended_actions.append(float(action_payload["intended_torque"]))
            applied_actions.append(float(outcome_payload["applied_torque"]))
            total_return += float(outcome_payload["reward"])
            predictions.append(result.decision.selected_assessment.prediction.prediction_id)
            tick = result.transition.created_at.tick + 1
            if result.experience.terminated or result.experience.truncated:
                break
    finally:
        cast(Any, environment).close()
    if len(transitions) != 200 or not transitions[-1].experience.truncated:
        raise RuntimeError("WM-001 requires exactly 200 accepted actions and a TimeLimit truncation")
    episode = EpisodeEvidence(
        episode_id=episode_id,
        task_id=task_id,
        reset_seed=int(reset_seed),
        undiscounted_return=total_return,
        transitions=tuple(transitions),
        intended_actions=tuple(intended_actions),
        applied_actions=tuple(applied_actions),
        prediction_ids=tuple(predictions),
        final_tick=transitions[-1].created_at.tick,
        agent=agent,
    )
    return episode, agent


def initial_snapshot(
    *,
    observation: np.ndarray,
    context: float,
    model_version: str,
    identities: CounterIdentitySource,
    at: TimePoint,
    task_id: str | None = None,
) -> AgentSnapshot:
    """Create an episode-local recurrent state while preserving persistent versions."""

    oscillator_reset = task_id == INDEPENDENT_OSCILLATOR_TASK
    reset_id = identities.next("reset-observation")
    reset_observation = Observation(
        observation_id=reset_id,
        agent_id=AGENT_ID,
        modality=("independent_phase_oscillator_reset_state" if oscillator_reset else "gymnasium_pendulum_reset_state"),
        evidence=Evidence(
            evidence_id=reset_id,
            payload={
                "physical_observation": tuple(float(value) for value in observation),
                "task_context": float(context),
            },
            occurred_at=at,
            available_at=at,
            lineage=EvidenceLineage(
                evidence_id=reset_id,
                origin=EvidenceOrigin.OBSERVED,
                provenance=Provenance(
                    source_id=(
                        f"{INDEPENDENT_OSCILLATOR_SOURCE}:reset" if oscillator_reset else "gymnasium:Pendulum-v1:reset"
                    ),
                    trust=TrustLevel.VERIFIED,
                    source_kind=(
                        "simulated_distractor_environment_reset" if oscillator_reset else "simulated_environment_reset"
                    ),
                ),
            ),
        ),
    )
    information = InformationSet(
        information_set_id=identities.next("information-set"),
        agent_id=AGENT_ID,
        as_of=at,
        observations=(reset_observation,),
        memory_version="wm001-memory:1",
    )
    belief = Belief(
        belief_id=identities.next("belief"),
        agent_id=AGENT_ID,
        target=TARGET,
        information_set=information,
        distribution=_state_distribution(identities, observation, context),
        formed_at=at,
        model_version=model_version,
        representation_version=REPRESENTATION_VERSION,
    )
    return AgentSnapshot(
        snapshot_id=identities.next("initial-snapshot"),
        agent_id=AGENT_ID,
        captured_at=at,
        belief=belief,
        configuration_version=f"wm001-config:{model_version}",
        memory_version=information.memory_version,
        knowledge_version="wm001-knowledge:none",
        model_version=model_version,
        representation_version=REPRESENTATION_VERSION,
        policy_version=POLICY_VERSION,
        resources=ResourceLedger(
            ledger_id=identities.next("resource-ledger"),
            started_at=at,
            completed_at=at,
        ),
    )


def transition_arrays(
    transitions: Sequence[EpistemicTransition],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Decode exact canonical transitions into model inputs and targets."""

    observations: list[np.ndarray] = []
    contexts: list[float] = []
    actions: list[float] = []
    targets: list[np.ndarray] = []
    transition_ids: list[str] = []
    for transition in transitions:
        decision = transition.experience.decision
        if decision is None:
            raise ValueError("WM-001 learning requires action-linked transitions")
        before = _belief_observation(decision.belief)
        after = _event_observation(transition.experience)
        action_payload = cast(dict[str, object], decision.intended_action.action.parameters)
        outcome_payload = cast(dict[str, object], transition.experience.outcome.evidence.payload)
        observations.append(before)
        contexts.append(float(action_payload["task_context"]))
        actions.append(float(action_payload["intended_torque"]))
        targets.append(np.concatenate((after - before, [float(outcome_payload["reward"])])))
        transition_ids.append(transition.transition_id)
    return (
        np.stack(observations),
        np.asarray(contexts, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        np.stack(targets),
        tuple(transition_ids),
    )


def transition_lineage_row(transition: EpistemicTransition, *, split: str) -> dict[str, object]:
    """Return the protocol-declared causal envelope for one real transition."""

    event = transition.experience
    decision = event.decision
    execution = event.execution
    if decision is None or execution is None:
        raise ValueError("WM-001 lineage requires an interaction transition")
    action_payload = cast(dict[str, object], decision.intended_action.action.parameters)
    outcome_payload = cast(dict[str, object], event.outcome.evidence.payload)
    prediction = decision.selected_assessment.prediction
    distribution = cast(dict[str, object], prediction.distribution.parameters)
    prior_observations = decision.belief.information_set.observations
    return {
        "split": split,
        "run_id": event.run_id,
        "task_id": event.task_id,
        "episode_id": event.episode_id,
        "step_index": event.step_index,
        "transition_id": transition.transition_id,
        "experience_id": event.experience_id,
        "pre_action_observation_id": prior_observations[-1].observation_id,
        "task_context": float(action_payload["task_context"]),
        "decision_id": decision.decision_id,
        "intended_action": float(action_payload["intended_torque"]),
        "executed_action_id": execution.execution_id,
        "applied_action": float(outcome_payload["applied_torque"]),
        "next_observation_id": event.observation.observation_id,
        "reward": float(outcome_payload["reward"]),
        "terminated": event.terminated,
        "truncated": event.truncated,
        "model_version_at_action": prediction.model_version,
        "parameter_digest_at_action": str(distribution["parameter_digest"]),
    }


def digest_rows(rows: Sequence[dict[str, object]]) -> str:
    payload = json.dumps(
        list(rows),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _task_context(task_id: str) -> float:
    try:
        return TASK_CONTEXTS[task_id]
    except KeyError as error:
        raise ValueError(f"unknown WM-001 task {task_id!r}") from error


def _state_distribution(
    identities: CounterIdentitySource,
    observation: np.ndarray,
    context: float,
) -> Distribution:
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Pendulum observation must be a finite length-three vector")
    return Distribution(
        distribution_id=identities.next("belief-distribution"),
        family="observed_point_state",
        support="Pendulum-v1 observation space x observed task context",
        parameters={
            "physical_observation": tuple(float(value) for value in values),
            "task_context": float(context),
        },
        representation_version=REPRESENTATION_VERSION,
        event_shape=(4,),
    )


def _belief_observation(belief: Belief) -> np.ndarray:
    parameters = cast(dict[str, object], belief.distribution.parameters)
    values = np.asarray(parameters["physical_observation"], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("belief physical observation has the wrong shape")
    return values


def _event_observation(event: object) -> np.ndarray:
    payload = cast(dict[str, object], cast(Any, event).observation.evidence.payload)
    values = np.asarray(payload["physical_observation"], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("experience physical observation has the wrong shape")
    return values


__all__ = (
    "AGENT_ID",
    "CALIBRATION_VERSION",
    "CanonicalReplayIndex",
    "CollectionEvidence",
    "Controller",
    "EpisodeEvidence",
    "INDEPENDENT_OSCILLATOR_SOURCE",
    "INDEPENDENT_OSCILLATOR_TASK",
    "run_independent_phase_oscillator_conformance",
    "IndependentPhaseOscillatorEnvironment",
    "POLICY_VERSION",
    "PendulumEpisodeSession",
    "PendulumRuntimeEnvironment",
    "PresetActionController",
    "PredictiveBackend",
    "REPRESENTATION_VERSION",
    "RuntimeCustody",
    "TASK_CONTEXTS",
    "TARGET",
    "UniformRandomController",
    "collect_episodes",
    "digest_rows",
    "initial_snapshot",
    "make_branch_learning_agent",
    "run_episode",
    "transition_arrays",
    "transition_lineage_row",
)
