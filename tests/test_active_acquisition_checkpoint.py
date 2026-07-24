from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from bench.active_acquisition.checkpoint import (
    Q1_COMPONENT_IDS,
    Q1_MAX_IDENTITY_NEXT_COUNTER,
    EpisodeAccumulator,
    Q1CheckpointArtifact,
    Q1CheckpointError,
    QualificationBinding,
    _decode_identity_source,
    _reject_private_value_keys,
    dump_q1_checkpoint,
    load_q1_checkpoint,
)
from prospect.decision import CounterIdentitySource
from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    CandidateAssessment,
    DecisionRecord,
    Distribution,
    EpistemicTarget,
    EpistemicTransition,
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
    Provenance,
    ResourceLedger,
    TimePoint,
    TrustLevel,
    UpdateReceipt,
    UpdateStatus,
    Utility,
)
from prospect.runtime import ModelState, VersionedModelOwner
from prospect.storage import (
    CheckpointComponent,
    CheckpointCoordinator,
    CheckpointFormatError,
    encode_domain_graph,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _point(tick: int) -> TimePoint:
    return TimePoint(tick=tick)


def _evidence(identity: str, *, tick: int, payload: object) -> Evidence:
    return Evidence(
        evidence_id=identity,
        payload=payload,
        occurred_at=_point(tick),
        available_at=_point(tick),
        lineage=EvidenceLineage(
            evidence_id=identity,
            origin=EvidenceOrigin.OBSERVED,
            provenance=Provenance(
                source_id="wm002-test-environment",
                trust=TrustLevel.VERIFIED,
                source_kind="fixture",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Fixture:
    owner: VersionedModelOwner
    predecessor_sha256: str
    snapshot: AgentSnapshot
    experience: ExperienceEvent
    transition: EpistemicTransition
    receipt: UpdateReceipt
    identities: CounterIdentitySource
    accumulator: EpisodeAccumulator
    binding: QualificationBinding


def _model_payload(*, posterior_numerator: int, evidence_count: int) -> bytes:
    value = {
        "evidence_count": evidence_count,
        "last_experience_id": "experience-acquisition" if evidence_count else None,
        "last_transition_id": "transition-acquisition" if evidence_count else None,
        "likelihood_version": "wm002-hidden-actuator-true-v1",
        "posterior_direct": {
            "denominator": 4 if evidence_count else 2,
            "numerator": posterior_numerator,
        },
        "schema": "prospect.wm002.posterior-model.v1",
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _validate_model(state: ModelState) -> None:
    decoded = json.loads(state.payload)
    assert decoded["schema"] == "prospect.wm002.posterior-model.v1"
    assert state.version == f"wm002-model-sha256:{state.digest}"


def _fixture() -> _Fixture:
    prior_payload = _model_payload(posterior_numerator=1, evidence_count=0)
    candidate_payload = _model_payload(posterior_numerator=3, evidence_count=1)
    prior_digest = hashlib.sha256(prior_payload).hexdigest()
    candidate_digest = hashlib.sha256(candidate_payload).hexdigest()
    prior_version = f"wm002-model-sha256:{prior_digest}"
    candidate_version = f"wm002-model-sha256:{candidate_digest}"
    target = EpistemicTarget(
        target_id="hidden-actuator-sign",
        description="posterior over the hidden actuator sign",
        target_kind="latent_state",
    )
    prior_information = InformationSet(
        information_set_id="information-prior",
        agent_id="wm002-agent",
        as_of=_point(0),
        memory_version="memory-0",
    )
    prior = Belief(
        belief_id="belief-prior",
        agent_id="wm002-agent",
        target=target,
        information_set=prior_information,
        distribution=Distribution(
            distribution_id="distribution-prior",
            family="categorical",
            support="{-1,+1}",
            parameters=(0.5, 0.5),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        formed_at=_point(0),
        model_version=prior_version,
        representation_version="representation-v1",
    )
    action = Action(
        action_id="acquisition:02:strong",
        action_kind="wm002_acquisition",
        parameters={"ordinal": 2, "phase": "acquisition", "semantic_action": "strong"},
    )
    prediction = Prediction(
        prediction_id="prediction-strong",
        prior_belief=prior,
        action=action,
        target=target,
        distribution=Distribution(
            distribution_id="prediction-strong-distribution",
            family="categorical",
            support="{-1,+1}",
            parameters=(0.25, 0.75),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        issued_at=_point(0),
        horizon_end=_point(2),
        model_version=prior_version,
        representation_version="representation-v1",
        calibration_version="known-fixture-v1",
    )
    goal = Goal(
        goal_id="goal-terminal-success",
        task_id="wm002-task",
        target=target,
        description="maximize reconstructed episode return",
        issued_at=_point(0),
        preference_version="wm002-preference-v1",
    )
    utility = Utility(
        utility_id="utility-strong",
        goal_id=goal.goal_id,
        prediction_id=prediction.prediction_id,
        expected_value=1.07,
        unit="episode_return",
        evaluator_version="wm002-assessor-v1",
        assessed_at=_point(0),
    )
    information_value = InformationValue(
        information_value_id="information-value-strong",
        prior_belief_id=prior.belief_id,
        action_id=action.action_id,
        target_id=target.target_id,
        expected_reduction=0.25,
        expected_cost=0.0,
        unit="episode_return",
        evaluator_version="wm002-assessor-v1",
        assessed_at=_point(0),
    )
    assessment = CandidateAssessment(
        assessment_id="assessment-strong",
        action=action,
        prediction=prediction,
        utility=utility,
        information_value=information_value,
        expected_action_cost=0.58,
        expected_risk=0.0,
        admissible=True,
        constraint_reasons=(),
        constraint_penalty=0.0,
        total_value=0.74,
        unit="episode_return",
        evaluator_version="wm002-assessor-v1",
        assessed_at=_point(0),
    )
    intention = IntendedAction(
        intention_id="intention-strong",
        agent_id="wm002-agent",
        action=action,
        intended_at=_point(0),
    )
    decision = DecisionRecord(
        decision_id="decision-acquisition",
        agent_id="wm002-agent",
        belief=prior,
        goal=goal,
        intended_action=intention,
        alternatives=(assessment,),
        selected_assessment=assessment,
        policy_version="wm002-policy-v1",
        decided_at=_point(0),
    )
    execution = ExecutedAction(
        execution_id="execution-acquisition",
        intention=intention,
        status=ExecutionStatus.SUCCEEDED,
        started_at=_point(1),
        ended_at=_point(2),
        realized_action=action,
    )
    observation = Observation(
        observation_id="observation-acquisition",
        agent_id="wm002-agent",
        modality="hidden-actuator-symbol",
        evidence=_evidence(
            "observation-acquisition",
            tick=2,
            payload={
                "observed_symbol": 1,
                "phase": "acquisition",
                "semantic_action": "strong",
            },
        ),
    )
    accumulator = EpisodeAccumulator(
        task_payoff=1.0,
        physical_action_cost=0.58,
        information_acquisition_cost=0.0,
    )
    outcome = Outcome(
        outcome_id="outcome-acquisition",
        execution_id=execution.execution_id,
        evidence=_evidence(
            "outcome-acquisition-evidence",
            tick=2,
            payload={
                "information_acquisition_cost": 0.0,
                "net_reward": 0.42,
                "physical_action_cost": 0.58,
                "task_payoff": 1.0,
            },
        ),
    )
    experience = ExperienceEvent(
        experience_id="experience-acquisition",
        agent_id="wm002-agent",
        run_id="run-000",
        task_id="wm002-task",
        episode_id="episode-000",
        step_index=0,
        kind=ExperienceKind.INTERACTION,
        observation=observation,
        outcome=outcome,
        terminated=False,
        truncated=False,
        discount=1.0,
        behavior_policy_version="wm002-policy-v1",
        closed_at=_point(2),
        decision=decision,
        execution=execution,
    )
    assimilated_information = InformationSet(
        information_set_id="information-assimilated",
        agent_id="wm002-agent",
        as_of=_point(2),
        observations=(observation,),
        memory_version="memory-1",
    )
    assimilated = Belief(
        belief_id="belief-assimilated",
        agent_id="wm002-agent",
        target=target,
        information_set=assimilated_information,
        distribution=replace(
            prior.distribution,
            distribution_id="distribution-assimilated",
        ),
        formed_at=_point(3),
        model_version=prior_version,
        representation_version="representation-v1",
    )
    update = BeliefUpdate(
        update_id="belief-update-acquisition",
        prior=prior,
        experience=experience,
        posterior=assimilated,
        updater_version="wm002-noop-assimilator-v1",
        updated_at=_point(3),
    )
    transition = EpistemicTransition(
        transition_id="transition-acquisition",
        experience=experience,
        belief_update=update,
        proper_scores=(),
        effects=(),
        created_at=_point(3),
    )
    resulting = Belief(
        belief_id="belief-persistent-posterior",
        agent_id="wm002-agent",
        target=target,
        information_set=assimilated_information,
        distribution=Distribution(
            distribution_id="distribution-persistent-posterior",
            family="categorical",
            support="{-1,+1}",
            parameters=(0.25, 0.75),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        formed_at=_point(5),
        model_version=candidate_version,
        representation_version="representation-v1",
    )
    receipt = UpdateReceipt(
        receipt_id="receipt-acquisition",
        agent_id="wm002-agent",
        transitions=(transition,),
        learner_version="wm002-exact-posterior-v1",
        status=UpdateStatus.APPLIED,
        previous_configuration_version=f"wm002-configuration-sha256:{prior_digest}",
        new_configuration_version=f"wm002-configuration-sha256:{candidate_digest}",
        previous_model_version=prior_version,
        new_model_version=candidate_version,
        previous_representation_version="representation-v1",
        new_representation_version="representation-v1",
        previous_policy_version="wm002-policy-v1",
        new_policy_version="wm002-policy-v1",
        started_at=_point(4),
        completed_at=_point(5),
        resulting_belief=resulting,
        metrics=(("posterior_after", 0.75),),
    )
    snapshot = AgentSnapshot(
        snapshot_id="snapshot-preterminal",
        agent_id="wm002-agent",
        captured_at=_point(5),
        belief=resulting,
        configuration_version=f"wm002-configuration-sha256:{candidate_digest}",
        memory_version="memory-1",
        knowledge_version="knowledge-v1",
        model_version=candidate_version,
        representation_version="representation-v1",
        policy_version="wm002-policy-v1",
        resources=ResourceLedger(
            ledger_id="resources-preterminal",
            started_at=_point(0),
            completed_at=_point(5),
        ),
        latest_update=receipt,
    )
    identities = CounterIdentitySource("wm002-run-000-episode-000")
    identities.next("decision")
    return _Fixture(
        owner=VersionedModelOwner.from_checkpoint(
            version=candidate_version,
            payload=candidate_payload,
            validator=_validate_model,
        ),
        predecessor_sha256=prior_digest,
        snapshot=snapshot,
        experience=experience,
        transition=transition,
        receipt=receipt,
        identities=identities,
        accumulator=accumulator,
        binding=QualificationBinding(
            protocol_version="0.3.0-q1",
            protocol_sha256=_digest("protocol"),
            implementation_sha256=_digest("implementation"),
            q0_report_sha256=_digest("q0-report"),
            entry_qualification_sha256=_digest("entry-qualification"),
            salt_commitment_sha256=_digest("salt-commitment"),
        ),
    )


def _artifact(fixture: _Fixture) -> Q1CheckpointArtifact:
    return dump_q1_checkpoint(
        checkpoint_id="checkpoint-run-000-episode-000",
        agent_id="wm002-agent",
        created_at=_point(5),
        model_owner=fixture.owner,
        predecessor_model_sha256=fixture.predecessor_sha256,
        snapshot=fixture.snapshot,
        experience=fixture.experience,
        transition=fixture.transition,
        receipt=fixture.receipt,
        identity_source=fixture.identities,
        accumulator=fixture.accumulator,
        binding=fixture.binding,
    )


@pytest.mark.parametrize(
    "private_key",
    (
        "hidden_regime",
        "secret_salt",
        "terminal_draw",
        "theta",
        "theta_order_hmac_sha256",
        "counterfactual_terminal_outcome",
    ),
)
def test_q1_checkpoint_rejects_private_keys_encoded_as_mapping_entries(private_key: str) -> None:
    encoded = encode_domain_graph({"opaque_record_mapping": {private_key: 1}})

    with pytest.raises(Q1CheckpointError, match="private keys"):
        _reject_private_value_keys(encoded)


def test_q1_checkpoint_is_deterministic_and_restores_fresh_canonical_graph() -> None:
    fixture = _fixture()
    first = _artifact(fixture)
    second = _artifact(fixture)

    assert first.payload == second.payload
    assert first.report == second.report
    assert tuple(row.component_id for row in first.report.component_digests) == Q1_COMPONENT_IDS
    assert first.report.aggregate_sha256 == hashlib.sha256(first.payload).hexdigest()
    restored = load_q1_checkpoint(
        first.payload,
        expected_agent_id="wm002-agent",
        expected_aggregate_sha256=first.report.aggregate_sha256,
        expected_binding=fixture.binding,
        model_validator=_validate_model,
    )

    assert restored.report == first.report
    assert restored.model_owner.checkpoint_bytes() == fixture.owner.checkpoint_bytes()
    assert restored.model_owner.version == fixture.owner.version
    assert restored.model_predecessor_sha256 == fixture.predecessor_sha256
    assert restored.experience == fixture.experience
    assert restored.experience is not fixture.experience
    assert restored.experience_store.get(restored.experience.experience_id) is restored.experience
    assert restored.ledger.get_transition(restored.transition.transition_id) is restored.transition
    assert restored.ledger.get_update(restored.receipt.receipt_id) is restored.receipt
    assert restored.transition.experience is restored.experience
    assert restored.receipt.transitions == (restored.transition,)
    assert restored.receipt.transitions[0] is restored.transition
    assert restored.snapshot.latest_update is restored.receipt
    assert restored.snapshot.belief is restored.receipt.resulting_belief
    assert restored.identity_source.namespace == fixture.identities.namespace
    assert restored.identity_source.next_counter == fixture.identities.next_counter
    assert restored.accumulator == fixture.accumulator


def test_q1_identity_counter_cap_accepts_boundary_and_rejects_successor() -> None:
    namespace = "wm002-q1-counter-boundary"
    boundary = CounterIdentitySource(namespace, next_counter=Q1_MAX_IDENTITY_NEXT_COUNTER)
    restored = _decode_identity_source(boundary.checkpoint_bytes())

    assert restored.namespace == namespace
    assert restored.next_counter == Q1_MAX_IDENTITY_NEXT_COUNTER
    above_boundary = CounterIdentitySource(namespace, next_counter=Q1_MAX_IDENTITY_NEXT_COUNTER + 1)
    with pytest.raises(Q1CheckpointError, match=r"counter must not exceed 64"):
        _decode_identity_source(above_boundary.checkpoint_bytes())


def test_q1_identity_counter_rejects_enormous_integer_before_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enormous_counter = 10**1000
    payload = (
        json.dumps(
            {
                "namespace": "wm002-q1-enormous-counter",
                "next_counter": enormous_counter,
                "schema_version": 1,
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )

    def fail_if_restored(_source: CounterIdentitySource, _payload: bytes) -> None:
        raise AssertionError("oversized counter reached CounterIdentitySource.restore_bytes")

    monkeypatch.setattr(CounterIdentitySource, "restore_bytes", fail_if_restored)
    with pytest.raises(Q1CheckpointError, match=r"counter must not exceed 64"):
        _decode_identity_source(payload)


def test_generic_in_memory_checkpoint_codec_matches_file_codec_and_is_canonical(tmp_path: Path) -> None:
    coordinator = CheckpointCoordinator(max_component_bytes=1024, max_total_bytes=4096)
    components = {
        "component": CheckpointComponent(
            name="component",
            version="v1",
            payload=b"fixture",
        )
    }
    payload = coordinator.dump_bytes(
        checkpoint_id="checkpoint",
        agent_id="agent",
        created_at=_point(1),
        components=components,
        versions={"fixture": "v1"},
    )
    path = tmp_path / "checkpoint.prospect"
    coordinator.save(
        path,
        checkpoint_id="checkpoint",
        agent_id="agent",
        created_at=_point(1),
        components=components,
        versions={"fixture": "v1"},
    )

    assert payload == path.read_bytes()
    assert coordinator.load_bytes(payload).payload("component") == b"fixture"
    with pytest.raises(CheckpointFormatError, match="not canonically encoded"):
        coordinator.load_bytes(payload + b"trailing-data")


def test_q1_checkpoint_rejects_external_binding_and_aggregate_mismatch() -> None:
    fixture = _fixture()
    artifact = _artifact(fixture)

    with pytest.raises(Q1CheckpointError, match="aggregate SHA-256"):
        load_q1_checkpoint(
            artifact.payload,
            expected_agent_id="wm002-agent",
            expected_aggregate_sha256=_digest("wrong-aggregate"),
            expected_binding=fixture.binding,
            model_validator=_validate_model,
        )
    with pytest.raises(Q1CheckpointError, match="metadata differs"):
        load_q1_checkpoint(
            artifact.payload,
            expected_agent_id="wm002-agent",
            expected_aggregate_sha256=artifact.report.aggregate_sha256,
            expected_binding=replace(fixture.binding, salt_commitment_sha256=_digest("another-salt")),
            model_validator=_validate_model,
        )


def test_q1_checkpoint_rejects_private_or_extra_components_before_restore() -> None:
    fixture = _fixture()
    artifact = _artifact(fixture)
    coordinator = CheckpointCoordinator(max_component_bytes=16 * 1024 * 1024, max_total_bytes=32 * 1024 * 1024)
    loaded = coordinator.load_bytes(artifact.payload)
    entries = {entry.name: entry for entry in loaded.manifest.components}
    components = {
        name: CheckpointComponent(
            name=name,
            version=entries[name].version,
            payload=payload,
            media_type=entries[name].media_type,
        )
        for name, payload in loaded.payloads
    }
    components["hidden_regime"] = CheckpointComponent(
        name="hidden_regime",
        version="private-v1",
        payload=b"1",
    )
    malicious = coordinator.dump_bytes(
        checkpoint_id=loaded.manifest.checkpoint_id,
        agent_id=loaded.manifest.agent_id,
        created_at=loaded.manifest.created_at,
        components=components,
        versions=dict(loaded.manifest.versions),
        metadata=dict(loaded.manifest.metadata),
    )

    with pytest.raises(Q1CheckpointError, match="forbidden private components"):
        load_q1_checkpoint(
            malicious,
            expected_agent_id="wm002-agent",
            expected_aggregate_sha256=hashlib.sha256(malicious).hexdigest(),
            expected_binding=fixture.binding,
            model_validator=_validate_model,
        )


def test_q1_checkpoint_rejects_primitive_accumulator_mismatch_before_encoding() -> None:
    fixture = _fixture()
    with pytest.raises(Q1CheckpointError, match="differs from primitive executed"):
        dump_q1_checkpoint(
            checkpoint_id="bad-accumulator",
            agent_id="wm002-agent",
            created_at=_point(5),
            model_owner=fixture.owner,
            predecessor_model_sha256=fixture.predecessor_sha256,
            snapshot=fixture.snapshot,
            experience=fixture.experience,
            transition=fixture.transition,
            receipt=fixture.receipt,
            identity_source=fixture.identities,
            accumulator=replace(fixture.accumulator, physical_action_cost=0.57),
            binding=fixture.binding,
        )


def test_q1_checkpoint_rejects_private_fields_inside_agent_visible_records() -> None:
    fixture = _fixture()
    original_observation = fixture.experience.observation
    bad_observation = replace(
        original_observation,
        evidence=replace(
            original_observation.evidence,
            payload={
                "observed_symbol": 1,
                "phase": "acquisition",
                "seed": 17,
                "semantic_action": "strong",
            },
        ),
    )
    bad_experience = replace(fixture.experience, observation=bad_observation)
    bad_update = replace(
        fixture.transition.belief_update,
        experience=bad_experience,
    )
    bad_transition = replace(
        fixture.transition,
        experience=bad_experience,
        belief_update=bad_update,
    )
    bad_receipt = replace(fixture.receipt, transitions=(bad_transition,))
    bad_snapshot = replace(fixture.snapshot, latest_update=bad_receipt)

    with pytest.raises(Q1CheckpointError, match="acquisition observation payload must contain exactly"):
        dump_q1_checkpoint(
            checkpoint_id="private-observation-field",
            agent_id="wm002-agent",
            created_at=_point(5),
            model_owner=fixture.owner,
            predecessor_model_sha256=fixture.predecessor_sha256,
            snapshot=bad_snapshot,
            experience=bad_experience,
            transition=bad_transition,
            receipt=bad_receipt,
            identity_source=fixture.identities,
            accumulator=fixture.accumulator,
            binding=fixture.binding,
        )


def test_q1_checkpoint_loads_in_a_fresh_interpreter(tmp_path: Path) -> None:
    fixture = _fixture()
    artifact = _artifact(fixture)
    path = tmp_path / "q1-frame.prospect"
    path.write_bytes(artifact.payload)
    script = (
        "from pathlib import Path;"
        "from bench.active_acquisition.checkpoint import QualificationBinding,load_q1_checkpoint;"
        f"b=QualificationBinding({fixture.binding.protocol_version!r},{fixture.binding.protocol_sha256!r},"
        f"{fixture.binding.implementation_sha256!r},{fixture.binding.q0_report_sha256!r},"
        f"{fixture.binding.entry_qualification_sha256!r},{fixture.binding.salt_commitment_sha256!r});"
        f"r=load_q1_checkpoint(Path({str(path)!r}).read_bytes(),expected_agent_id='wm002-agent',"
        f"expected_aggregate_sha256={artifact.report.aggregate_sha256!r},expected_binding=b,"
        "model_validator=lambda state: None);"
        "print(r.report.aggregate_sha256);"
        "print(len(r.experience_store),r.ledger.transition_count,r.ledger.update_count)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        artifact.report.aggregate_sha256,
        "1 1 1",
    ]
