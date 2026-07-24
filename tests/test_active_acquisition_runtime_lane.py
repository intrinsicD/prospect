from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import cast

import pytest

from bench.active_acquisition import q1 as q1_producer
from bench.active_acquisition.checkpoint import (
    EpisodeAccumulator,
    QualificationBinding,
    dump_q1_checkpoint,
    load_q1_checkpoint,
)
from bench.active_acquisition.runtime_lane import (
    AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS,
    AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS,
    AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS,
    AGENT_VISIBLE_TERMINAL_ACTION_FIELDS,
    AGENT_VISIBLE_TERMINAL_OBSERVATION_FIELDS,
    AGENT_VISIBLE_TERMINAL_OUTCOME_FIELDS,
    EFFECT_VERSION,
    LEARNER_VERSION,
    MODEL_VERSION_PREFIX,
    POSTERIOR_MODEL_SCHEMA,
    AcquisitionTerminalScorer,
    ArmMode,
    EpisodeRuntimeComposition,
    ExactPosteriorModel,
    ExactPosteriorTransactionalLearner,
    HiddenActuatorAcquisitionEnvironment,
    HiddenActuatorCandidateAssessor,
    HiddenActuatorTerminalEnvironment,
    NoOpAssimilationEffect,
    NoOpObservationAssimilator,
    candidate_diagnostic_digest,
    compose_episode_runtime,
    compose_restored_terminal_runtime,
    decode_posterior_model,
    encode_posterior_model,
    initial_posterior_model_state,
    model_version_for_payload,
    validate_agent_visible_environment_step,
    validate_posterior_model_state,
)
from bench.active_acquisition.seeding import PrivateEpisodeSeedMaterial
from prospect.decision import DecisionError
from prospect.domain import EpistemicEffectKind, TimePoint, UpdateReceipt, UpdateStatus
from prospect.runtime import InteractionContext, InteractionResult, ModelState


def _uniform_digest(master: int = 0, episode: int = 0) -> str:
    payload = f"WM-002|0.3.0-q1|q1v3-uniform|{master}|{episode}".encode()
    return hashlib.sha256(payload).hexdigest()


def _runtime(
    arm: ArmMode = ArmMode.PROSPECT,
    *,
    ordinal: int | None = None,
    namespace: str = "wm002-runtime-test",
) -> EpisodeRuntimeComposition:
    return compose_episode_runtime(
        episode_id=f"{namespace}:episode",
        arm=arm,
        identity_namespace=namespace,
        uniform_ordinal=ordinal,
        uniform_selector_digest=(_uniform_digest() if arm is ArmMode.UNIFORM_RANDOM else None),
    )


def _acquire(
    runtime: EpisodeRuntimeComposition,
    *,
    observed_symbol: int = 1,
) -> InteractionResult:
    environment = HiddenActuatorAcquisitionEnvironment(
        agent_id=runtime.agent_id,
        observed_symbol=observed_symbol,
        identities=runtime.identities,
    )
    return runtime.acquisition_agent.interact(
        environment,
        runtime.acquisition_goal,
        context=InteractionContext(
            run_id="wm002-unit-rehearsal",
            task_id=runtime.acquisition_goal.task_id,
            episode_id=runtime.episode_id,
            step_index=0,
        ),
        decide_at=TimePoint(1),
    )


def _learn(
    runtime: EpisodeRuntimeComposition,
    acquisition: InteractionResult,
) -> UpdateReceipt:
    return runtime.acquisition_agent.learn(
        (acquisition.transition,),
        at=TimePoint(acquisition.transition.created_at.tick + 1),
    )


def _terminal(
    runtime: EpisodeRuntimeComposition,
    receipt: UpdateReceipt,
    *,
    terminal_success: bool = True,
) -> InteractionResult:
    environment = HiddenActuatorTerminalEnvironment(
        agent_id=runtime.agent_id,
        terminal_success=terminal_success,
        identities=runtime.identities,
    )
    return runtime.terminal_agent.interact(
        environment,
        runtime.terminal_goal,
        context=InteractionContext(
            run_id="wm002-unit-rehearsal",
            task_id=runtime.terminal_goal.task_id,
            episode_id=runtime.episode_id,
            step_index=1,
        ),
        decide_at=TimePoint(receipt.completed_at.tick + 1),
    )


def test_exact_model_codec_is_canonical_and_self_binding() -> None:
    state = initial_posterior_model_state()
    decoded = decode_posterior_model(state.payload)

    assert decoded == ExactPosteriorModel(
        evidence_count=0,
        last_experience_id=None,
        last_transition_id=None,
        posterior_direct=Fraction(1, 2),
    )
    assert encode_posterior_model(decoded) == state.payload
    assert state.version == f"{MODEL_VERSION_PREFIX}{state.digest}"
    validate_posterior_model_state(state)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"schema": "wrong"}, "schema"),
        ({"likelihood_version": "wrong"}, "likelihood"),
        ({"evidence_count": -1}, "nonnegative"),
        ({"last_experience_id": "orphan"}, "zero-evidence"),
    ],
)
def test_exact_model_codec_rejects_invalid_top_level_state(
    mutation: dict[str, object],
    match: str,
) -> None:
    raw = json.loads(initial_posterior_model_state().payload)
    raw.update(mutation)
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match=match):
        decode_posterior_model(payload)


@pytest.mark.parametrize(
    "numerator,denominator,match",
    [
        (2, 4, "reduced"),
        (-1, 2, r"in \[0, 1\]"),
        (3, 2, r"in \[0, 1\]"),
        (1, 0, "positive denominator"),
    ],
)
def test_exact_model_codec_rejects_invalid_fraction(
    numerator: int,
    denominator: int,
    match: str,
) -> None:
    raw = json.loads(initial_posterior_model_state().payload)
    raw["posterior_direct"] = {
        "numerator": numerator,
        "denominator": denominator,
    }
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match=match):
        decode_posterior_model(payload)


def test_exact_model_codec_rejects_noncanonical_json_and_unbound_version() -> None:
    state = initial_posterior_model_state()
    raw = json.loads(state.payload)
    noncanonical = (json.dumps(raw, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="canonical"):
        decode_posterior_model(noncanonical)

    wrong_version = ModelState(version="wm002-model-sha256:" + "0" * 64, payload=state.payload)
    with pytest.raises(ValueError, match="does not bind"):
        validate_posterior_model_state(wrong_version)


def test_composed_validator_enforces_predecessor_relative_evidence_count() -> None:
    predecessor = initial_posterior_model_state()
    skipped_count = ExactPosteriorModel(
        evidence_count=2,
        last_experience_id="experience",
        last_transition_id="transition",
        posterior_direct=Fraction(1, 2),
    )
    payload = encode_posterior_model(skipped_count)
    candidate = ModelState(version=model_version_for_payload(payload), payload=payload)

    # The unary owner validator is structural by design.
    validate_posterior_model_state(candidate)
    with pytest.raises(ValueError, match="advance by exactly one"):
        validate_posterior_model_state(candidate, predecessor=predecessor)


@pytest.mark.parametrize(
    ("arm", "ordinal", "expected_action", "selection_unit"),
    [
        (ArmMode.PROSPECT, None, "strong", "return"),
        (ArmMode.ORACLE, None, "strong", "return"),
        (ArmMode.GOAL_ONLY, None, "skip", "return"),
        (ArmMode.RAW_ENTROPY, None, "nuisance", "nats"),
        (ArmMode.EIG_ONLY, None, "overpowered", "nats"),
        (ArmMode.SHUFFLED_INFORMATION, None, "weak", "return"),
        (ArmMode.UNIFORM_RANDOM, 3, "overpowered", None),
    ],
)
def test_all_seven_arms_select_from_truthful_five_candidate_assessments(
    arm: ArmMode,
    ordinal: int | None,
    expected_action: str,
    selection_unit: str | None,
) -> None:
    runtime = _runtime(arm, ordinal=ordinal, namespace=f"arm-{arm.value}")
    decision = runtime.acquisition_agent.decide(runtime.acquisition_goal, at=TimePoint(1))
    rows = runtime.acquisition_policy.last_diagnostic_rows

    assert [assessment.action.action_id for assessment in decision.alternatives] == [
        "acquisition:00:skip",
        "acquisition:01:weak",
        "acquisition:02:strong",
        "acquisition:03:overpowered",
        "acquisition:04:nuisance",
    ]
    assert [assessment.total_value for assessment in decision.alternatives] == pytest.approx(
        [0.50, 0.63, 0.74, 0.45, 0.49],
        abs=1e-15,
    )
    assert [row.semantic_action for row in rows] == [
        "skip",
        "weak",
        "strong",
        "overpowered",
        "nuisance",
    ]
    assert [row.assessment_id for row in rows] == [assessment.assessment_id for assessment in decision.alternatives]
    selected_parameters = cast(dict[str, object], decision.selected_assessment.action.parameters)
    assert selected_parameters["semantic_action"] == expected_action
    assert all(row.selection_unit == selection_unit for row in rows)
    if arm is ArmMode.UNIFORM_RANDOM:
        assert all(row.arm_selection_score is None for row in rows)
    else:
        assert all(row.arm_selection_score is not None for row in rows)
    assert runtime.acquisition_policy.last_diagnostic_digest == candidate_diagnostic_digest(rows)


def test_candidate_costs_and_value_decomposition_are_truthful_for_every_arm() -> None:
    expected_physical = [0.0, 0.53, 0.58, 0.95, 0.0]
    expected_information = [0.0, 0.0, 0.0, 0.0, 0.01]
    reference: tuple[tuple[float, float, float, float], ...] | None = None

    for arm in ArmMode:
        ordinal = 0 if arm is ArmMode.UNIFORM_RANDOM else None
        runtime = _runtime(arm, ordinal=ordinal, namespace=f"truth-{arm.value}")
        decision = runtime.acquisition_agent.decide(runtime.acquisition_goal, at=TimePoint(1))
        decomposition = tuple(
            (
                assessment.utility.expected_value,
                assessment.information_value.expected_reduction,
                assessment.information_value.expected_cost,
                assessment.expected_action_cost,
            )
            for assessment in decision.alternatives
        )
        if reference is None:
            reference = decomposition
        assert decomposition == reference
        assert [row.expected_action_cost for row in decision.alternatives] == expected_physical
        assert [row.information_value.expected_cost for row in decision.alternatives] == expected_information
        for assessment in decision.alternatives:
            assert assessment.total_value == pytest.approx(
                assessment.utility.expected_value
                + assessment.information_value.expected_reduction
                - assessment.information_value.expected_cost
                - assessment.expected_action_cost,
                abs=1e-15,
            )


def test_uniform_selection_requires_precomputed_successor_ordinal_and_digest() -> None:
    with pytest.raises(ValueError, match="precomputed ordinal"):
        _runtime(ArmMode.UNIFORM_RANDOM)
    with pytest.raises(ValueError, match="SHA-256"):
        compose_episode_runtime(
            episode_id="uniform-invalid",
            arm=ArmMode.UNIFORM_RANDOM,
            identity_namespace="uniform-invalid",
            uniform_ordinal=2,
            uniform_selector_digest="legacy-seed",
        )
    with pytest.raises(ValueError, match=r"\[0, 5\)"):
        compose_episode_runtime(
            episode_id="uniform-invalid-ordinal",
            arm=ArmMode.UNIFORM_RANDOM,
            identity_namespace="uniform-invalid-ordinal",
            uniform_ordinal=5,
            uniform_selector_digest=_uniform_digest(),
        )


def test_uniform_policy_fails_closed_if_frozen_ordinal_disappears() -> None:
    runtime = _runtime(
        ArmMode.UNIFORM_RANDOM,
        ordinal=2,
        namespace="uniform-missing-ordinal",
    )
    object.__setattr__(runtime.acquisition_assessor, "_uniform_ordinal", None)

    with pytest.raises(DecisionError, match="has no frozen selection ordinal"):
        runtime.acquisition_agent.decide(runtime.acquisition_goal, at=TimePoint(1))


def test_runtime_uses_one_exact_composite_target_across_belief_and_both_phases() -> None:
    runtime = _runtime()
    snapshot = runtime.acquisition_agent.snapshot(TimePoint(0))

    assert snapshot.belief.target is runtime.acquisition_goal.target
    assert runtime.acquisition_goal.target is runtime.terminal_goal.target
    assert snapshot.belief.target.target_id == "wm002-acquisition-observation-and-terminal-success"
    decision = runtime.acquisition_agent.decide(runtime.acquisition_goal, at=TimePoint(1))
    assert all(
        assessment.prediction.target is snapshot.belief.target
        and assessment.information_value.target_id == snapshot.belief.target.target_id
        for assessment in decision.alternatives
    )


def test_episode_composition_binds_keyed_initial_identity_counter() -> None:
    runtime = compose_episode_runtime(
        episode_id="identity-counter:episode",
        arm=ArmMode.PROSPECT,
        identity_namespace="identity-counter",
        identity_next_counter=37,
    )

    assert runtime.identities.next_counter == 37


def test_agent_visible_acquisition_environment_has_exact_allowlisted_shapes() -> None:
    runtime = _runtime()
    decision = runtime.acquisition_agent.decide(runtime.acquisition_goal, at=TimePoint(1))
    environment = HiddenActuatorAcquisitionEnvironment(
        agent_id=runtime.agent_id,
        observed_symbol=1,
        identities=runtime.identities,
    )
    step = environment.step(decision.intended_action)
    assert AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS == frozenset({"ordinal", "phase", "semantic_action"})
    assert AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS == frozenset({"observed_symbol", "phase", "semantic_action"})
    assert AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS == frozenset(
        {"information_acquisition_cost", "net_reward", "physical_action_cost", "task_payoff"}
    )
    assert (
        set(cast(dict[str, object], decision.intended_action.action.parameters))
        == AGENT_VISIBLE_ACQUISITION_ACTION_FIELDS
    )
    assert (
        set(cast(dict[str, object], step.observation.evidence.payload)) == AGENT_VISIBLE_ACQUISITION_OBSERVATION_FIELDS
    )
    assert set(cast(dict[str, object], step.outcome.evidence.payload)) == AGENT_VISIBLE_ACQUISITION_OUTCOME_FIELDS
    validate_agent_visible_environment_step(step)

    assert step.observation.evidence.payload == {
        "phase": "acquisition",
        "semantic_action": "strong",
        "observed_symbol": 1,
    }
    outcome = cast(dict[str, float], step.outcome.evidence.payload)
    assert outcome["task_payoff"] == 1.0
    assert outcome["physical_action_cost"] == 0.58
    assert outcome["information_acquisition_cost"] == 0.0
    assert outcome["net_reward"] == pytest.approx(0.42, abs=1e-15)
    visible = json.dumps(
        {
            "action": decision.intended_action.action.parameters,
            "observation": step.observation.evidence.payload,
            "outcome": step.outcome.evidence.payload,
        },
        sort_keys=True,
    )
    assert all(
        forbidden not in visible for forbidden in ("theta", "regime", "salt", "secret", "counterfactual", "future")
    )
    with pytest.raises(RuntimeError, match="single-use"):
        environment.step(decision.intended_action)


def test_acquisition_interaction_is_no_op_until_transactional_learning() -> None:
    runtime = _runtime()
    predecessor = runtime.model_owner.snapshot_state()
    acquisition = _acquire(runtime)
    transition = acquisition.transition

    assert acquisition.decision.selected_assessment.action.action_id == "acquisition:02:strong"
    assert transition.belief_update.prior.distribution.parameters == (0.5, 0.5)
    assert transition.belief_update.posterior.distribution.parameters == (0.5, 0.5)
    assert transition.belief_update.posterior.belief_id != transition.belief_update.prior.belief_id
    assert (
        transition.belief_update.posterior.information_set.information_set_id
        != transition.belief_update.prior.information_set.information_set_id
    )
    assert transition.belief_update.posterior.model_version == predecessor.version
    assert transition.effects[0].kind is EpistemicEffectKind.INFORMATION_GAIN
    assert transition.effects[0].measure == "assimilation_only_categorical_entropy"
    assert transition.effects[0].evaluator_version == EFFECT_VERSION
    assert transition.effects[0].improvement == 0.0
    assert transition.proper_scores[0].realized_evidence_id == (acquisition.experience.observation.evidence.evidence_id)
    assert runtime.model_owner.snapshot_state() == predecessor
    assert len(runtime.store) == runtime.ledger.transition_count == 1
    assert runtime.ledger.update_count == 0


def test_transactional_learning_updates_exact_model_and_receipt_lineage() -> None:
    runtime = _runtime()
    predecessor = runtime.model_owner.snapshot_state()
    acquisition = _acquire(runtime)
    receipt = _learn(runtime, acquisition)
    candidate = runtime.model_owner.snapshot_state()
    decoded = decode_posterior_model(candidate.payload)
    snapshot = runtime.acquisition_agent.snapshot(receipt.completed_at)
    metrics = dict(receipt.metrics)

    assert receipt.status is UpdateStatus.APPLIED
    assert receipt.learner_version == LEARNER_VERSION
    assert receipt.transitions == (acquisition.transition,)
    assert receipt.transitions[0] is acquisition.transition
    assert receipt.previous_model_version == predecessor.version
    assert receipt.new_model_version == candidate.version
    assert candidate != predecessor
    assert decoded.posterior_direct == Fraction(9, 10)
    assert decoded.evidence_count == 1
    assert decoded.last_experience_id == acquisition.experience.experience_id
    assert decoded.last_transition_id == acquisition.transition.transition_id
    assert snapshot.model_version == candidate.version
    assert snapshot.belief is receipt.resulting_belief
    assert snapshot.latest_update is receipt
    assert snapshot.belief.distribution.parameters == pytest.approx((0.1, 0.9), abs=1e-15)
    assert metrics == pytest.approx(
        {
            "posterior_direct_before": 0.5,
            "posterior_direct_after": 0.9,
            "entropy_before_nats": math.log(2.0),
            "entropy_after_nats": -(0.1 * math.log(0.1) + 0.9 * math.log(0.9)),
            "entropy_reduction_nats": math.log(2.0) + 0.1 * math.log(0.1) + 0.9 * math.log(0.9),
            "consumed_transition_count": 1.0,
            "evidence_count_before": 0.0,
            "evidence_count_after": 1.0,
        },
        abs=1e-15,
    )
    assert runtime.ledger.update_count == 1
    assert runtime.ledger.get_update(receipt.receipt_id) is receipt


@pytest.mark.parametrize(
    ("ordinal", "expected_posterior"),
    [
        (0, Fraction(1, 2)),
        (1, Fraction(7, 10)),
        (2, Fraction(9, 10)),
        (3, Fraction(1, 1)),
        (4, Fraction(1, 2)),
    ],
)
def test_learner_uses_true_executed_action_likelihood_for_all_candidates(
    ordinal: int,
    expected_posterior: Fraction,
) -> None:
    runtime = _runtime(
        ArmMode.UNIFORM_RANDOM,
        ordinal=ordinal,
        namespace=f"likelihood-{ordinal}",
    )
    acquisition = _acquire(runtime, observed_symbol=(0 if ordinal == 0 else 1))
    _learn(runtime, acquisition)

    assert decode_posterior_model(runtime.model_owner.snapshot_state().payload).posterior_direct == expected_posterior


def test_shuffled_arm_changes_selection_only_and_learns_true_weak_likelihood() -> None:
    runtime = _runtime(ArmMode.SHUFFLED_INFORMATION, namespace="shuffled-true-likelihood")
    acquisition = _acquire(runtime, observed_symbol=1)
    receipt = _learn(runtime, acquisition)

    parameters = cast(dict[str, object], acquisition.decision.selected_assessment.action.parameters)
    assert parameters["semantic_action"] == "weak"
    assert decode_posterior_model(runtime.model_owner.snapshot_state().payload).posterior_direct == Fraction(7, 10)
    assert receipt.transitions == (acquisition.transition,)


def test_learner_rejects_wrong_cardinality_and_wrong_episode_or_arm() -> None:
    runtime = _runtime()
    acquisition = _acquire(runtime)
    snapshot = runtime.acquisition_agent.snapshot(TimePoint(acquisition.transition.created_at.tick + 1))
    current_model = runtime.model_owner.snapshot_state()
    learner = ExactPosteriorTransactionalLearner(
        identities=runtime.identities,
        expected_episode_id=runtime.episode_id,
        expected_arm=runtime.arm,
        expected_policy_version=runtime.policy_version,
    )

    with pytest.raises(ValueError, match="exactly one"):
        learner.prepare(snapshot, (), current_model)
    with pytest.raises(ValueError, match="exactly one"):
        learner.prepare(
            snapshot,
            (acquisition.transition, acquisition.transition),
            current_model,
        )
    wrong_episode = ExactPosteriorTransactionalLearner(
        identities=runtime.identities,
        expected_episode_id="another-episode",
        expected_arm=runtime.arm,
        expected_policy_version=runtime.policy_version,
    )
    with pytest.raises(ValueError, match="episode/step"):
        wrong_episode.prepare(snapshot, (acquisition.transition,), current_model)
    wrong_arm = ExactPosteriorTransactionalLearner(
        identities=runtime.identities,
        expected_episode_id=runtime.episode_id,
        expected_arm=ArmMode.GOAL_ONLY,
        expected_policy_version=runtime.policy_version,
    )
    with pytest.raises(ValueError, match="expected arm policy"):
        wrong_arm.prepare(snapshot, (acquisition.transition,), current_model)


def test_restored_terminal_composition_builds_fresh_phase_components() -> None:
    runtime = _runtime(namespace="restored-terminal")
    acquisition = _acquire(runtime)
    receipt = _learn(runtime, acquisition)
    restored = compose_restored_terminal_runtime(
        episode_id=runtime.episode_id,
        arm=runtime.arm,
        restored_at=receipt.completed_at,
        identities=runtime.identities,
        state=runtime.state,
        store=runtime.store,
        ledger=runtime.ledger,
        model_owner=runtime.model_owner,
    )

    snapshot = restored.terminal_agent.snapshot(receipt.completed_at)
    assert restored.terminal_goal.target is snapshot.belief.target
    assert restored.terminal_agent is not runtime.terminal_agent
    assert restored.terminal_assessor is not runtime.terminal_assessor
    decision = restored.terminal_agent.decide(
        restored.terminal_goal,
        at=TimePoint(receipt.completed_at.tick + 1),
    )
    assert decision.selected_assessment.action.action_id == "terminal:00:+1"
    assert all(assessment.prediction.target is snapshot.belief.target for assessment in decision.alternatives)


def test_checkpoint_restore_replays_identical_terminal_decision_in_fresh_graph() -> None:
    runtime = _runtime(namespace="checkpoint-terminal-replay")
    predecessor = runtime.model_owner.snapshot_state()
    acquisition = _acquire(runtime)
    receipt = _learn(runtime, acquisition)
    snapshot = runtime.acquisition_agent.snapshot(receipt.completed_at)
    outcome = cast(dict[str, float], acquisition.experience.outcome.evidence.payload)
    binding = QualificationBinding(
        protocol_version="0.3.0-q1",
        protocol_sha256=hashlib.sha256(b"protocol").hexdigest(),
        implementation_sha256=hashlib.sha256(b"implementation").hexdigest(),
        q0_report_sha256=hashlib.sha256(b"q0").hexdigest(),
        entry_qualification_sha256=hashlib.sha256(b"entry").hexdigest(),
        salt_commitment_sha256=hashlib.sha256(b"salt").hexdigest(),
    )
    artifact = dump_q1_checkpoint(
        checkpoint_id="checkpoint-terminal-replay",
        agent_id=runtime.agent_id,
        created_at=snapshot.captured_at,
        model_owner=runtime.model_owner,
        predecessor_model_sha256=predecessor.digest,
        snapshot=snapshot,
        experience=acquisition.experience,
        transition=acquisition.transition,
        receipt=receipt,
        identity_source=runtime.identities,
        accumulator=EpisodeAccumulator(
            task_payoff=outcome["task_payoff"],
            physical_action_cost=outcome["physical_action_cost"],
            information_acquisition_cost=outcome["information_acquisition_cost"],
        ),
        binding=binding,
    )
    loaded = load_q1_checkpoint(
        artifact.payload,
        expected_agent_id=runtime.agent_id,
        expected_aggregate_sha256=artifact.report.aggregate_sha256,
        expected_binding=binding,
        model_validator=validate_posterior_model_state,
    )
    restored = compose_restored_terminal_runtime(
        episode_id=runtime.episode_id,
        arm=runtime.arm,
        restored_at=loaded.snapshot.captured_at,
        identities=loaded.identity_source,
        state=loaded.agent_state,
        store=loaded.experience_store,
        ledger=loaded.ledger,
        model_owner=loaded.model_owner,
    )
    decision_at = TimePoint(receipt.completed_at.tick + 1)
    live_decision = runtime.terminal_agent.decide(runtime.terminal_goal, at=decision_at)
    restored_decision = restored.terminal_agent.decide(restored.terminal_goal, at=decision_at)

    assert restored_decision == live_decision
    assert loaded.identity_source.next_counter == runtime.identities.next_counter
    assert loaded.snapshot.belief is loaded.receipt.resulting_belief


def test_terminal_assessment_requires_committed_acquisition_receipt() -> None:
    runtime = _runtime()
    snapshot = runtime.terminal_agent.snapshot(TimePoint(0))
    with pytest.raises(ValueError, match="committed acquisition receipt"):
        runtime.terminal_assessor.assess(snapshot, runtime.terminal_goal)


def test_skip_posterior_uses_explicit_direct_tie_order_at_terminal() -> None:
    runtime = _runtime(ArmMode.GOAL_ONLY, namespace="terminal-tie")
    acquisition = _acquire(runtime, observed_symbol=0)
    receipt = _learn(runtime, acquisition)
    terminal = _terminal(runtime, receipt)

    assert decode_posterior_model(runtime.model_owner.snapshot_state().payload).posterior_direct == Fraction(1, 2)
    assert [assessment.action.action_id for assessment in terminal.decision.alternatives] == [
        "terminal:00:+1",
        "terminal:01:-1",
    ]
    assert terminal.decision.selected_assessment.action.action_id == "terminal:00:+1"


def test_end_to_end_unit_rehearsal_has_two_transitions_and_one_update() -> None:
    runtime = _runtime(namespace="end-to-end-unit")
    acquisition = _acquire(runtime, observed_symbol=1)
    receipt = _learn(runtime, acquisition)
    terminal = _terminal(
        runtime,
        receipt,
        terminal_success=True,
    )
    final = runtime.terminal_agent.snapshot(TimePoint(terminal.transition.created_at.tick + 1))
    acquisition_outcome = cast(dict[str, float], acquisition.experience.outcome.evidence.payload)
    terminal_outcome = cast(dict[str, object], terminal.experience.outcome.evidence.payload)
    assert AGENT_VISIBLE_TERMINAL_ACTION_FIELDS == frozenset({"ordinal", "phase", "terminal_action"})
    assert AGENT_VISIBLE_TERMINAL_OBSERVATION_FIELDS == frozenset({"phase"})
    assert AGENT_VISIBLE_TERMINAL_OUTCOME_FIELDS == frozenset(
        {"executed_terminal_action", "success", "terminal_reward"}
    )
    assert (
        set(cast(dict[str, object], terminal.decision.intended_action.action.parameters))
        == AGENT_VISIBLE_TERMINAL_ACTION_FIELDS
    )
    assert (
        set(cast(dict[str, object], terminal.experience.observation.evidence.payload))
        == AGENT_VISIBLE_TERMINAL_OBSERVATION_FIELDS
    )
    assert set(terminal_outcome) == AGENT_VISIBLE_TERMINAL_OUTCOME_FIELDS
    reconstructed_return = (
        acquisition_outcome["task_payoff"]
        - acquisition_outcome["physical_action_cost"]
        - acquisition_outcome["information_acquisition_cost"]
        + cast(float, terminal_outcome["terminal_reward"])
    )

    assert terminal.experience.observation.evidence.payload == {"phase": "terminal"}
    assert terminal_outcome == {
        "executed_terminal_action": 1,
        "success": True,
        "terminal_reward": 1.0,
    }
    assert terminal.transition.proper_scores[0].realized_evidence_id == (
        terminal.experience.outcome.evidence.evidence_id
    )
    assert terminal.experience.terminated
    assert reconstructed_return == pytest.approx(1.42, abs=1e-15)
    assert len(runtime.store) == 2
    assert runtime.ledger.transition_count == 2
    assert runtime.ledger.update_count == 1
    assert final.latest_update is receipt
    assert final.model_version == receipt.new_model_version
    assert final.belief.distribution.parameters == pytest.approx((0.1, 0.9), abs=1e-15)


def test_acquisition_public_state_is_byte_identical_under_private_noninterference() -> None:
    protocol_sha256 = hashlib.sha256(b"synthetic-noninterference-protocol").hexdigest()
    implementation_sha256 = hashlib.sha256(b"synthetic-noninterference-implementation").hexdigest()
    q0_report_sha256 = hashlib.sha256(b"synthetic-noninterference-q0").hexdigest()
    entry_sha256 = hashlib.sha256(b"synthetic-noninterference-entry").hexdigest()
    salt_commitment_sha256 = hashlib.sha256(b"synthetic-noninterference-salt-commitment").hexdigest()
    binding = QualificationBinding(
        protocol_version="0.3.0-q1",
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        entry_qualification_sha256=entry_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    baseline = q1_producer._run_synthetic_episode(
        arm=ArmMode.PROSPECT,
        synthetic_ordinal=0,
        binding=binding,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
    )

    q1_producer._exercise_synthetic_noninterference(
        artifact=baseline,
        binding=binding,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
        public_samples={},
    )


def _noninterference_fixture() -> tuple[QualificationBinding, dict[str, str], object]:
    """Build one accepted baseline the mutation controls can corrupt."""

    digests = {
        name: hashlib.sha256(f"synthetic-noninterference-{name}".encode()).hexdigest()
        for name in ("protocol", "implementation", "q0", "entry", "salt")
    }
    binding = QualificationBinding(
        protocol_version="0.3.0-q1",
        protocol_sha256=digests["protocol"],
        implementation_sha256=digests["implementation"],
        q0_report_sha256=digests["q0"],
        entry_qualification_sha256=digests["entry"],
        salt_commitment_sha256=digests["salt"],
    )
    baseline = q1_producer._run_synthetic_episode(
        arm=ArmMode.PROSPECT,
        synthetic_ordinal=0,
        binding=binding,
        protocol_sha256=digests["protocol"],
        implementation_sha256=digests["implementation"],
        q0_report_sha256=digests["q0"],
        salt_commitment_sha256=digests["salt"],
    )
    return binding, digests, baseline


def _exercise_noninterference(binding: QualificationBinding, digests: dict[str, str], baseline: object) -> None:
    q1_producer._exercise_synthetic_noninterference(
        artifact=cast(q1_producer.EpisodeArtifacts, baseline),
        binding=binding,
        protocol_sha256=digests["protocol"],
        implementation_sha256=digests["implementation"],
        q0_report_sha256=digests["q0"],
        salt_commitment_sha256=digests["salt"],
        public_samples={},
    )


@pytest.mark.parametrize(
    ("leaked_field", "expected_message"),
    [
        ("acquisition", "changed the acquisition/public trace projection"),
        ("checkpoint", "changed the acquisition/public trace projection"),
    ],
)
def test_noninterference_probe_detects_private_state_leaking_into_public_rows(
    monkeypatch: pytest.MonkeyPatch,
    leaked_field: str,
    expected_message: str,
) -> None:
    """A probe that cannot fail proves nothing; make the leak and require detection."""

    binding, digests, baseline = _noninterference_fixture()
    real_run_live_episode = q1_producer._run_live_episode

    def leaking_episode(**kwargs: object) -> q1_producer.EpisodeArtifacts:
        artifact = real_run_live_episode(**kwargs)  # type: ignore[arg-type]
        private = cast(PrivateEpisodeSeedMaterial, kwargs["private_audit"])
        raw = dict(artifact.raw_trace)
        section = dict(cast(dict, raw[leaked_field]))
        section["observed_symbol_count"] = private.theta
        raw[leaked_field] = section
        return q1_producer.EpisodeArtifacts(
            raw_trace=raw,
            private_audit=artifact.private_audit,
            checkpoint_index=artifact.checkpoint_index,
            checkpoint_payload=artifact.checkpoint_payload,
            acquisition_public_probe=artifact.acquisition_public_probe,
        )

    monkeypatch.setattr(q1_producer, "_run_live_episode", leaking_episode)
    with pytest.raises(q1_producer.Q1ExecutionError, match=expected_message):
        _exercise_noninterference(binding, digests, baseline)


def test_noninterference_probe_detects_a_private_dependent_checkpoint_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, digests, baseline = _noninterference_fixture()
    real_run_live_episode = q1_producer._run_live_episode

    def leaking_episode(**kwargs: object) -> q1_producer.EpisodeArtifacts:
        artifact = real_run_live_episode(**kwargs)  # type: ignore[arg-type]
        private = cast(PrivateEpisodeSeedMaterial, kwargs["private_audit"])
        return q1_producer.EpisodeArtifacts(
            raw_trace=artifact.raw_trace,
            private_audit=artifact.private_audit,
            checkpoint_index=artifact.checkpoint_index,
            checkpoint_payload=artifact.checkpoint_payload + str(private.theta).encode("ascii"),
            acquisition_public_probe=artifact.acquisition_public_probe,
        )

    monkeypatch.setattr(q1_producer, "_run_live_episode", leaking_episode)
    with pytest.raises(q1_producer.Q1ExecutionError, match="changed the preterminal checkpoint payload"):
        _exercise_noninterference(binding, digests, baseline)


def test_noninterference_probe_rejects_a_private_variant_that_never_varied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal public bytes are only evidence when the private input actually changed."""

    binding, digests, baseline = _noninterference_fixture()
    real_run_live_episode = q1_producer._run_live_episode
    reference_material: list[object] = []

    def frozen_private_episode(**kwargs: object) -> q1_producer.EpisodeArtifacts:
        if not reference_material:
            reference_material.append(kwargs["private_audit"])
        kwargs["private_audit"] = reference_material[0]
        return real_run_live_episode(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(q1_producer, "_run_live_episode", frozen_private_episode)
    with pytest.raises(q1_producer.Q1ExecutionError, match="private material"):
        _exercise_noninterference(binding, digests, baseline)


def test_noninterference_probe_rejects_an_unvaried_private_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two arms of the probe must derive genuinely different private material."""

    binding, digests, baseline = _noninterference_fixture()
    real_material = q1_producer._synthetic_private_material

    def constant_material(**kwargs: object) -> PrivateEpisodeSeedMaterial:
        kwargs.pop("private_variant", None)
        kwargs.pop("theta", None)
        return real_material(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(q1_producer, "_synthetic_private_material", constant_material)
    with pytest.raises(
        q1_producer.Q1ExecutionError,
        match="did not vary every private theta/HMAC field",
    ):
        _exercise_noninterference(binding, digests, baseline)


def test_noninterference_probe_rejects_an_insensitive_terminal_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal control must move executed success, or the probe is vacuous."""

    binding, digests, baseline = _noninterference_fixture()
    real_run_live_episode = q1_producer._run_live_episode

    def insensitive_episode(**kwargs: object) -> q1_producer.EpisodeArtifacts:
        callback = cast(object, kwargs["terminal_success"])
        kwargs["terminal_success"] = lambda decision: bool(callback(decision)) and False  # type: ignore[operator]
        return real_run_live_episode(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(q1_producer, "_run_live_episode", insensitive_episode)
    with pytest.raises(q1_producer.Q1ExecutionError, match="terminal sensitivity control did not vary"):
        _exercise_noninterference(binding, digests, baseline)


def test_runtime_components_satisfy_prospect_behavior_protocols() -> None:
    runtime = _runtime()
    identities = runtime.identities

    assert isinstance(runtime.acquisition_assessor, HiddenActuatorCandidateAssessor)
    assert isinstance(NoOpObservationAssimilator(identities), NoOpObservationAssimilator)
    assert isinstance(AcquisitionTerminalScorer(identities), AcquisitionTerminalScorer)
    assert isinstance(NoOpAssimilationEffect(identities), NoOpAssimilationEffect)
    assert POSTERIOR_MODEL_SCHEMA == "prospect.wm002.posterior-model.v1"
