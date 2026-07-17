from __future__ import annotations

import copy
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from prospect.domain import (
    Action,
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
    TimePoint,
    TrustLevel,
    UpdateReceipt,
    UpdateStatus,
    Utility,
)
from prospect.storage import (
    CausalOrderError,
    CheckpointComponent,
    CheckpointCoordinator,
    CheckpointFormatError,
    CheckpointIntegrityError,
    DuplicateRecordError,
    EpistemicLedger,
    InMemoryExperienceStore,
    LedgerIntegrityError,
    RecordNotFoundError,
    TensorDictExperienceReplay,
    torchrl_available,
)


def _time(tick: int, clock_id: str = "interaction") -> TimePoint:
    return TimePoint(tick=tick, clock_id=clock_id)


def _evidence(
    evidence_id: str,
    *,
    tick: int,
    payload: object,
    clock_id: str = "interaction",
) -> Evidence:
    point = _time(tick, clock_id)
    return Evidence(
        evidence_id=evidence_id,
        payload=payload,
        occurred_at=point,
        available_at=point,
        lineage=EvidenceLineage(
            evidence_id=evidence_id,
            origin=EvidenceOrigin.OBSERVED,
            provenance=Provenance(
                source_id="test-environment",
                trust=TrustLevel.VERIFIED,
                source_kind="fixture",
            ),
        ),
    )


def _passive_experience(
    experience_id: str,
    *,
    agent_id: str,
    tick: int,
    clock_id: str = "interaction",
    episode_id: str | None = None,
    step_index: int | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> ExperienceEvent:
    observation_id = f"{experience_id}-observation"
    outcome_evidence_id = f"{experience_id}-outcome-evidence"
    return ExperienceEvent(
        experience_id=experience_id,
        agent_id=agent_id,
        run_id="fixture-run",
        task_id="fixture-task",
        episode_id=episode_id or experience_id,
        step_index=tick if step_index is None else step_index,
        kind=ExperienceKind.PASSIVE,
        observation=Observation(
            observation_id=observation_id,
            agent_id=agent_id,
            modality="fixture",
            evidence=_evidence(
                observation_id,
                tick=tick,
                payload={"observation": experience_id},
                clock_id=clock_id,
            ),
        ),
        outcome=Outcome(
            outcome_id=f"{experience_id}-outcome",
            evidence=_evidence(
                outcome_evidence_id,
                tick=tick,
                payload={"outcome": experience_id},
                clock_id=clock_id,
            ),
        ),
        terminated=terminated,
        truncated=truncated,
        discount=0.0 if terminated else 1.0,
        behavior_policy_version="passive-v1",
        closed_at=_time(tick, clock_id),
    )


@dataclass(frozen=True, slots=True)
class _TransitionFixture:
    experience: ExperienceEvent
    transition: EpistemicTransition
    receipt: UpdateReceipt


def _transition_fixture(
    prefix: str,
    *,
    agent_id: str = "agent-a",
    base: int = 0,
) -> _TransitionFixture:
    target = EpistemicTarget(
        target_id=f"{prefix}-target",
        description="binary outcome after an action",
        target_kind="future-outcome",
    )
    prior_observation_id = f"{prefix}-prior-observation"
    prior_observation = Observation(
        observation_id=prior_observation_id,
        agent_id=agent_id,
        modality="state",
        evidence=_evidence(
            prior_observation_id,
            tick=base,
            payload=0,
        ),
    )
    prior_information = InformationSet(
        information_set_id=f"{prefix}-prior-information",
        agent_id=agent_id,
        as_of=_time(base),
        observations=(prior_observation,),
        memory_version=f"{prefix}-memory-0",
    )
    prior = Belief(
        belief_id=f"{prefix}-prior",
        agent_id=agent_id,
        target=target,
        information_set=prior_information,
        distribution=Distribution(
            distribution_id=f"{prefix}-prior-distribution",
            family="categorical",
            support="binary",
            parameters=(0.5, 0.5),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        formed_at=_time(base),
        model_version="model-v1",
        representation_version="representation-v1",
    )
    action = Action(
        action_id=f"{prefix}-action",
        action_kind="choose",
        parameters=1,
    )
    intention = IntendedAction(
        intention_id=f"{prefix}-intention",
        agent_id=agent_id,
        action=action,
        intended_at=_time(base),
    )
    prediction = Prediction(
        prediction_id=f"{prefix}-prediction",
        prior_belief=prior,
        action=action,
        target=target,
        distribution=Distribution(
            distribution_id=f"{prefix}-prediction-distribution",
            family="categorical",
            support="binary",
            parameters=(0.4, 0.6),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        issued_at=_time(base),
        horizon_end=_time(base + 2),
        model_version="model-v1",
        representation_version="representation-v1",
        calibration_version="calibration-v1",
    )
    goal = Goal(
        goal_id=f"{prefix}-goal",
        task_id=f"{prefix}-task",
        target=target,
        description="choose the correct binary outcome",
        issued_at=_time(base),
        preference_version="preference-v1",
    )
    utility = Utility(
        utility_id=f"{prefix}-utility",
        goal_id=goal.goal_id,
        prediction_id=prediction.prediction_id,
        expected_value=0.6,
        unit="value",
        evaluator_version="utility-v1",
        assessed_at=_time(base),
    )
    information_value = InformationValue(
        information_value_id=f"{prefix}-information-value",
        prior_belief_id=prior.belief_id,
        action_id=action.action_id,
        target_id=target.target_id,
        expected_reduction=0.1,
        expected_cost=0.01,
        unit="value",
        evaluator_version="information-v1",
        assessed_at=_time(base),
    )
    assessment = CandidateAssessment(
        assessment_id=f"{prefix}-assessment",
        action=action,
        prediction=prediction,
        utility=utility,
        information_value=information_value,
        expected_action_cost=0.02,
        expected_risk=0.03,
        admissible=True,
        constraint_reasons=(),
        constraint_penalty=0.0,
        total_value=0.64,
        unit="value",
        evaluator_version="candidate-v1",
        assessed_at=_time(base),
    )
    decision = DecisionRecord(
        decision_id=f"{prefix}-decision",
        agent_id=agent_id,
        belief=prior,
        goal=goal,
        intended_action=intention,
        alternatives=(assessment,),
        selected_assessment=assessment,
        policy_version="policy-v1",
        decided_at=_time(base),
    )
    execution = ExecutedAction(
        execution_id=f"{prefix}-execution",
        intention=intention,
        status=ExecutionStatus.SUCCEEDED,
        started_at=_time(base + 1),
        ended_at=_time(base + 2),
        realized_action=action,
    )
    observation_id = f"{prefix}-observation"
    observation = Observation(
        observation_id=observation_id,
        agent_id=agent_id,
        modality="state",
        evidence=_evidence(observation_id, tick=base + 2, payload=1),
    )
    outcome = Outcome(
        outcome_id=f"{prefix}-outcome",
        evidence=_evidence(
            f"{prefix}-outcome-evidence",
            tick=base + 2,
            payload={"correct": True},
        ),
        execution_id=execution.execution_id,
    )
    experience = ExperienceEvent(
        experience_id=f"{prefix}-experience",
        agent_id=agent_id,
        run_id="fixture-run",
        task_id=goal.task_id,
        episode_id=f"{prefix}-episode",
        step_index=0,
        kind=ExperienceKind.INTERACTION,
        observation=observation,
        outcome=outcome,
        terminated=True,
        truncated=False,
        discount=0.0,
        behavior_policy_version=decision.policy_version,
        closed_at=_time(base + 2),
        decision=decision,
        execution=execution,
    )
    posterior_information = InformationSet(
        information_set_id=f"{prefix}-posterior-information",
        agent_id=agent_id,
        as_of=_time(base + 2),
        observations=(prior_observation, observation),
        memory_version=f"{prefix}-memory-1",
    )
    posterior = Belief(
        belief_id=f"{prefix}-posterior",
        agent_id=agent_id,
        target=target,
        information_set=posterior_information,
        distribution=Distribution(
            distribution_id=f"{prefix}-posterior-distribution",
            family="categorical",
            support="binary",
            parameters=(0.1, 0.9),
            representation_version="representation-v1",
            event_shape=(2,),
        ),
        formed_at=_time(base + 3),
        model_version="model-v1",
        representation_version="representation-v1",
    )
    belief_update = BeliefUpdate(
        update_id=f"{prefix}-belief-update",
        prior=prior,
        experience=experience,
        posterior=posterior,
        updater_version="updater-v1",
        updated_at=_time(base + 3),
    )
    transition = EpistemicTransition(
        transition_id=f"{prefix}-transition",
        experience=experience,
        belief_update=belief_update,
        proper_scores=(
            ProperScore(
                score_id=f"{prefix}-score",
                prediction_id=prediction.prediction_id,
                realized_evidence_id=outcome.evidence.evidence_id,
                rule="log-score",
                value=0.51,
                unit="nats",
                scorer_version="scorer-v1",
                scored_at=_time(base + 3),
            ),
        ),
        effects=(
            EpistemicEffect(
                effect_id=f"{prefix}-effect",
                belief_update_id=belief_update.update_id,
                target_id=target.target_id,
                kind=EpistemicEffectKind.INFORMATION_GAIN,
                measure="entropy",
                before=0.69,
                after=0.33,
                improvement=0.36,
                higher_is_better=False,
                evaluator_version="effect-v1",
                evaluated_at=_time(base + 3),
            ),
        ),
        created_at=_time(base + 3),
    )
    receipt = UpdateReceipt(
        receipt_id=f"{prefix}-receipt",
        agent_id=agent_id,
        transitions=(transition,),
        learner_version="learner-v1",
        status=UpdateStatus.APPLIED,
        previous_configuration_version=f"{prefix}-configuration-0",
        new_configuration_version=f"{prefix}-configuration-1",
        previous_model_version="model-v1",
        new_model_version="model-v1",
        previous_representation_version="representation-v1",
        new_representation_version="representation-v1",
        previous_policy_version="policy-v1",
        new_policy_version="policy-v1",
        started_at=_time(base + 4),
        completed_at=_time(base + 5),
    )
    return _TransitionFixture(
        experience=experience,
        transition=transition,
        receipt=receipt,
    )


def test_experience_store_is_append_only_ordered_and_protocol_conformant() -> None:
    store = InMemoryExperienceStore()
    first = _passive_experience("a-1", agent_id="agent-a", tick=1)
    same_tick = _passive_experience("a-2", agent_id="agent-a", tick=1)
    later = _passive_experience("a-3", agent_id="agent-a", tick=3)
    other_agent = _passive_experience("b-1", agent_id="agent-b", tick=0)

    for event in (first, same_tick, later, other_agent):
        store.append(event)

    assert isinstance(store, ExperienceStore)
    assert len(store) == 4
    assert store.get(first.experience_id) is first
    assert store.get_step("agent-a", "fixture-run", "a-1", 1) is first
    assert store.find_step("agent-a", "fixture-run", "missing", 1) is None
    assert store.history("agent-a", _time(1)) == (first, same_tick)
    assert store.history("agent-a", _time(3)) == (first, same_tick, later)
    assert store.history("agent-b", _time(0)) == (other_agent,)
    assert store.history("unknown-agent", _time(999, "another-clock")) == ()


def test_experience_store_rejects_duplicates_regression_and_clock_mix() -> None:
    store = InMemoryExperienceStore()
    later = _passive_experience("later", agent_id="agent-a", tick=5)
    store.append(later)

    with pytest.raises(DuplicateRecordError):
        store.append(later)
    with pytest.raises(CausalOrderError, match="before"):
        store.append(_passive_experience("earlier", agent_id="agent-a", tick=4))
    with pytest.raises(CausalOrderError, match="different clocks"):
        store.append(
            _passive_experience(
                "wrong-clock",
                agent_id="agent-a",
                tick=6,
                clock_id="simulator",
            )
        )
    with pytest.raises(CausalOrderError, match="different clocks"):
        store.history("agent-a", _time(5, "simulator"))
    with pytest.raises(RecordNotFoundError):
        store.get("missing")
    assert len(store) == 1


def test_experience_store_rejects_step_regression_and_closed_episode() -> None:
    store = InMemoryExperienceStore()
    first = _passive_experience(
        "episode-step-2",
        agent_id="agent-a",
        tick=2,
        episode_id="episode-a",
        step_index=2,
    )
    store.append(first)

    with pytest.raises(CausalOrderError, match="not after"):
        store.append(
            _passive_experience(
                "episode-step-1-late",
                agent_id="agent-a",
                tick=3,
                episode_id="episode-a",
                step_index=1,
            )
        )

    terminal = _passive_experience(
        "episode-terminal",
        agent_id="agent-a",
        tick=4,
        episode_id="episode-a",
        step_index=3,
        terminated=True,
    )
    store.append(terminal)
    with pytest.raises(CausalOrderError, match="terminated or truncated"):
        store.append(
            _passive_experience(
                "episode-after-terminal",
                agent_id="agent-a",
                tick=5,
                episode_id="episode-a",
                step_index=4,
            )
        )
    assert store.history("agent-a", _time(5)) == (first, terminal)


def test_experience_store_rejects_duplicate_runtime_step_identity() -> None:
    store = InMemoryExperienceStore()
    first = _passive_experience(
        "canonical-step",
        agent_id="agent-a",
        tick=1,
        episode_id="episode-a",
        step_index=0,
    )
    duplicate_step = _passive_experience(
        "different-experience-id",
        agent_id="agent-a",
        tick=2,
        episode_id="episode-a",
        step_index=0,
    )
    store.append(first)

    with pytest.raises(DuplicateRecordError, match="step identity"):
        store.append(duplicate_step)
    with pytest.raises(RecordNotFoundError, match="experience step"):
        store.get_step("agent-a", "fixture-run", "episode-a", 99)
    with pytest.raises(ValueError, match="nonnegative"):
        store.find_step("agent-a", "fixture-run", "episode-a", -1)
    assert store.get_step("agent-a", "fixture-run", "episode-a", 0) is first
    assert len(store) == 1


def test_epistemic_ledger_requires_canonical_links_and_filters_history() -> None:
    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    first = _transition_fixture("first", base=0)
    second = _transition_fixture("second", base=10)
    absent = _transition_fixture("absent", base=20)
    store.append(first.experience)
    store.append(second.experience)

    ledger.append_transition(first.transition)
    ledger.append_update(first.receipt)
    ledger.append_transition(second.transition)
    ledger.append_update(second.receipt)

    assert ledger.transition_count == 2
    assert ledger.update_count == 2
    assert ledger.get_transition(first.transition.transition_id) is first.transition
    assert ledger.get_update(second.receipt.receipt_id) is second.receipt
    assert ledger.transitions("agent-a", _time(3)) == (first.transition,)
    assert ledger.transitions("agent-a", _time(13)) == (
        first.transition,
        second.transition,
    )
    assert ledger.updates("agent-a", _time(5)) == (first.receipt,)
    assert ledger.transitions("another-agent", _time(999, "other-clock")) == ()

    with pytest.raises(DuplicateRecordError):
        ledger.append_transition(first.transition)
    with pytest.raises(DuplicateRecordError):
        ledger.append_update(first.receipt)
    with pytest.raises(LedgerIntegrityError, match="absent"):
        ledger.append_transition(absent.transition)
    with pytest.raises(RecordNotFoundError):
        ledger.get_transition("missing-transition")


def test_epistemic_ledger_explicitly_rehydrates_restart_copies() -> None:
    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    fixture = _transition_fixture("restart")
    store.append(fixture.experience)

    transition_copy = copy.deepcopy(fixture.transition)
    assert transition_copy.experience is not fixture.experience
    rehydrated_transition = ledger.append_rehydrated_transition(transition_copy)
    assert rehydrated_transition.experience is fixture.experience
    assert fixture.experience.decision is not None
    assert rehydrated_transition.belief_update.prior is fixture.experience.decision.belief

    receipt_copy = copy.deepcopy(fixture.receipt)
    rehydrated_receipt = ledger.append_rehydrated_update(receipt_copy)
    assert rehydrated_receipt.transitions == (rehydrated_transition,)
    assert rehydrated_receipt.transitions[0] is rehydrated_transition


def test_checkpoint_bundle_is_deterministic_atomic_and_restartable(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "agent.prospect-checkpoint"
    coordinator = CheckpointCoordinator()
    components = {
        "model": CheckpointComponent(
            name="model",
            version="model-v2",
            payload=b"model state",
        ),
        "replay": CheckpointComponent(
            name="replay",
            version="memory-v4",
            payload=b"replay state",
            media_type="application/x-prospect-replay",
        ),
    }
    manifest = coordinator.save(
        checkpoint_path,
        checkpoint_id="checkpoint-1",
        agent_id="agent-a",
        created_at=_time(42),
        components=components,
        versions={
            "configuration": "configuration-v7",
            "model": "model-v2",
        },
        metadata={"experiment": "E5-restart"},
    )
    first_archive_bytes = checkpoint_path.read_bytes()
    loaded = coordinator.load(checkpoint_path, expected_agent_id="agent-a")

    assert loaded.manifest == manifest
    assert loaded.payload("model") == b"model state"
    assert loaded.payload("replay") == b"replay state"
    assert manifest.version_map["model"] == "model-v2"
    assert manifest.metadata_map["experiment"] == "E5-restart"

    coordinator.save(
        checkpoint_path,
        checkpoint_id="checkpoint-1",
        agent_id="agent-a",
        created_at=_time(42),
        components=components,
        versions={
            "configuration": "configuration-v7",
            "model": "model-v2",
        },
        metadata={"experiment": "E5-restart"},
    )
    assert checkpoint_path.read_bytes() == first_archive_bytes

    restored: dict[str, tuple[bytes, str]] = {}
    returned = coordinator.restore(
        checkpoint_path,
        {
            "model": lambda payload, entry: restored.__setitem__(entry.name, (payload, entry.version)),
            "replay": lambda payload, entry: restored.__setitem__(entry.name, (payload, entry.version)),
        },
        expected_agent_id="agent-a",
    )
    assert returned == manifest
    assert restored == {
        "model": (b"model state", "model-v2"),
        "replay": (b"replay state", "memory-v4"),
    }


def test_checkpoint_restore_preflights_callbacks_before_mutation(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "agent.prospect-checkpoint"
    coordinator = CheckpointCoordinator()
    coordinator.save(
        checkpoint_path,
        checkpoint_id="checkpoint-1",
        agent_id="agent-a",
        created_at=_time(1),
        components={
            "model": CheckpointComponent("model", "v1", b"model"),
            "policy": CheckpointComponent("policy", "v1", b"policy"),
        },
        versions={"configuration": "v1"},
    )
    called: list[str] = []
    with pytest.raises(KeyError, match="missing restorers"):
        coordinator.restore(
            checkpoint_path,
            {"model": lambda _payload, entry: called.append(entry.name)},
        )
    assert called == []


def test_checkpoint_rejects_corruption_wrong_agent_and_nonarchives(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "agent.prospect-checkpoint"
    coordinator = CheckpointCoordinator()
    coordinator.save(
        checkpoint_path,
        checkpoint_id="checkpoint-1",
        agent_id="agent-a",
        created_at=_time(1),
        components={"model": CheckpointComponent("model", "v1", b"MODEL")},
        versions={"configuration": "v1"},
    )
    with pytest.raises(CheckpointIntegrityError, match="belongs to agent"):
        coordinator.load(checkpoint_path, expected_agent_id="agent-b")

    with zipfile.ZipFile(checkpoint_path, "r") as archive:
        manifest_bytes = archive.read("manifest.json")
    with zipfile.ZipFile(checkpoint_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("components/model.bin", b"OTHER")
    with pytest.raises(CheckpointIntegrityError, match="sha256"):
        coordinator.load(checkpoint_path)

    nonarchive = tmp_path / "not-a-checkpoint"
    nonarchive.write_bytes(b"not a zip file")
    with pytest.raises(CheckpointFormatError, match="valid ZIP"):
        coordinator.load(nonarchive)


def test_checkpoint_validation_failure_does_not_replace_prior_bundle(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "agent.prospect-checkpoint"
    coordinator = CheckpointCoordinator(max_component_bytes=8, max_total_bytes=16)
    coordinator.save(
        checkpoint_path,
        checkpoint_id="good",
        agent_id="agent-a",
        created_at=_time(1),
        components={"model": CheckpointComponent("model", "v1", b"good")},
        versions={"configuration": "v1"},
    )
    before = checkpoint_path.read_bytes()

    with pytest.raises(ValueError, match="exceeds"):
        coordinator.save(
            checkpoint_path,
            checkpoint_id="bad",
            agent_id="agent-a",
            created_at=_time(2),
            components={"model": CheckpointComponent("model", "v2", b"far too large")},
            versions={"configuration": "v2"},
        )
    assert checkpoint_path.read_bytes() == before
    assert coordinator.load(checkpoint_path).manifest.checkpoint_id == "good"


@pytest.mark.skipif(  # type: ignore[untyped-decorator]
    not torchrl_available(), reason="optional TorchRL runtime absent"
)
def test_torchrl_replay_is_lazy_codec_driven_and_not_canonical_storage() -> None:
    import torch
    from tensordict import TensorDict  # type: ignore[import-untyped]

    class FixtureCodec:
        version = "fixture-codec-v1"

        def __init__(self) -> None:
            self._next_key = 0
            self._events: dict[int, ExperienceEvent] = {}

        def encode(self, event: ExperienceEvent) -> object:
            key = self._next_key
            self._next_key += 1
            self._events[key] = event
            return TensorDict(
                {"event_key": torch.tensor(key, dtype=torch.int64)},
                batch_size=[],
            )

        def decode(self, encoded: object) -> ExperienceEvent:
            assert isinstance(encoded, TensorDict)
            key = int(encoded.get("event_key").item())
            return self._events[key]

    codec = FixtureCodec()
    replay = TensorDictExperienceReplay(capacity=2, codec=codec)
    first = _passive_experience("event-1", agent_id="agent-a", tick=1)
    second = _passive_experience("event-2", agent_id="agent-a", tick=2)
    third = _passive_experience("event-3", agent_id="agent-a", tick=3)
    replay.add(first)
    replay.add(second)
    replay.add(third)

    sampled = replay.sample(20)
    assert len(replay) == 2
    assert len(sampled) == 20
    assert {event.experience_id for event in sampled}.issubset({second.experience_id, third.experience_id})
    with pytest.raises(DuplicateRecordError):
        replay.add(first)
    with pytest.raises(ValueError, match="positive"):
        replay.sample(0)


@pytest.mark.skipif(  # type: ignore[untyped-decorator]
    not torchrl_available(), reason="optional TorchRL runtime absent"
)
def test_torchrl_replay_checkpoint_restores_storage_rng_and_seen_ids() -> None:
    import torch
    from tensordict import TensorDict

    events = {
        index: _passive_experience(
            f"checkpoint-event-{index}",
            agent_id="agent-a",
            tick=index,
        )
        for index in range(1, 5)
    }

    class RegistryCodec:
        version = "registry-codec-v1"

        def encode(self, event: ExperienceEvent) -> object:
            key = int(event.experience_id.rsplit("-", 1)[1])
            return TensorDict(
                {"event_key": torch.tensor(key, dtype=torch.int64)},
                batch_size=[],
            )

        def decode(self, encoded: object) -> ExperienceEvent:
            assert isinstance(encoded, TensorDict)
            return events[int(encoded.get("event_key").item())]

    original = TensorDictExperienceReplay(
        capacity=3,
        codec=RegistryCodec(),
        seed=119,
    )
    for event in events.values():
        original.add(event)
    checkpoint_bytes = original.dump_checkpoint_bytes()
    expected_after_restart = [event.experience_id for event in original.sample(12)]

    restarted = TensorDictExperienceReplay(
        capacity=3,
        codec=RegistryCodec(),
        seed=999,
    )
    restarted.load_checkpoint_bytes(checkpoint_bytes)
    actual_after_restart = [event.experience_id for event in restarted.sample(12)]

    assert len(restarted) == 3
    assert actual_after_restart == expected_after_restart
    with pytest.raises(DuplicateRecordError):
        restarted.add(events[1])


def test_ledger_rejects_update_whose_transition_was_not_appended() -> None:
    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    fixture = _transition_fixture("orphan")
    store.append(fixture.experience)

    with pytest.raises(LedgerIntegrityError, match="unknown transition"):
        ledger.append_update(fixture.receipt)

    altered_receipt = replace(
        fixture.receipt,
        receipt_id="orphan-rejected",
        status=UpdateStatus.REJECTED,
        new_configuration_version=fixture.receipt.previous_configuration_version,
    )
    with pytest.raises(LedgerIntegrityError, match="unknown transition"):
        ledger.append_update(altered_receipt)
