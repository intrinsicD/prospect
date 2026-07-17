from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from threading import Barrier, RLock, Thread
from typing import Any, cast

import pytest

from prospect.decision import CounterIdentitySource
from prospect.domain import (
    AgentSnapshot,
    EpistemicTransition,
    TimePoint,
    UpdateReceipt,
    UpdateStatus,
)
from prospect.runtime import (
    AgentState,
    EpistemicAgent,
    ModelState,
    PreparedLearningUpdate,
    RuntimeError,
    StateTransitionError,
    VersionedModelOwner,
)
from prospect.storage import EpistemicLedger


@dataclass(frozen=True, slots=True)
class _Transition:
    transition_id: str


@dataclass(frozen=True, slots=True)
class _Snapshot:
    agent_id: str
    captured_at: TimePoint
    configuration_version: str
    model_version: str
    representation_version: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class _Receipt:
    receipt_id: str
    agent_id: str
    transitions: tuple[EpistemicTransition, ...]
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


class _State:
    def __init__(self) -> None:
        self.agent_id = "agent"
        self.configuration_version = "configuration-v1"
        self.model_version = "model-v1"
        self.representation_version = "representation-v1"
        self.policy_version = "policy-v1"
        self.latest_update: UpdateReceipt | None = None
        self.fail_commit: str | None = None
        self.preflight_barrier: Barrier | None = None
        self._lock = RLock()

    def snapshot(self, agent_id: str, at: TimePoint) -> AgentSnapshot:
        with self._lock:
            if agent_id != self.agent_id:
                raise StateTransitionError("wrong agent")
            return cast(
                AgentSnapshot,
                _Snapshot(
                    agent_id=self.agent_id,
                    captured_at=at,
                    configuration_version=self.configuration_version,
                    model_version=self.model_version,
                    representation_version=self.representation_version,
                    policy_version=self.policy_version,
                ),
            )

    def transaction_lock(self) -> RLock:
        return self._lock

    def validate_update(self, receipt: UpdateReceipt) -> None:
        with self._lock:
            self._validate_update(receipt)
        if self.preflight_barrier is not None:
            self.preflight_barrier.wait(timeout=5)

    def apply_update(self, receipt: UpdateReceipt) -> None:
        prepared = self.prepare_update(receipt)
        self.commit_prepared_update(prepared)

    def prepare_update(self, receipt: UpdateReceipt) -> tuple[UpdateReceipt, tuple[object, ...]]:
        self._validate_update(receipt)
        return (receipt, self._persistent_state())

    def commit_prepared_update(
        self,
        prepared: tuple[UpdateReceipt, tuple[object, ...]],
    ) -> None:
        receipt, _ = prepared
        if self.fail_commit == "before":
            raise ValueError("state commit failed before apply")
        self.configuration_version = receipt.new_configuration_version
        self.model_version = receipt.new_model_version
        self.representation_version = receipt.new_representation_version
        self.policy_version = receipt.new_policy_version
        self.latest_update = receipt
        if self.fail_commit == "after":
            raise ValueError("state commit failed after apply")

    def rollback_prepared_update(
        self,
        prepared: tuple[UpdateReceipt, tuple[object, ...]],
    ) -> None:
        _, previous = prepared
        self.configuration_version = cast(str, previous[0])
        self.model_version = cast(str, previous[1])
        self.representation_version = cast(str, previous[2])
        self.policy_version = cast(str, previous[3])
        self.latest_update = cast(UpdateReceipt | None, previous[4])

    def _persistent_state(self) -> tuple[object, ...]:
        return (
            self.configuration_version,
            self.model_version,
            self.representation_version,
            self.policy_version,
            self.latest_update,
        )

    def _validate_update(self, receipt: UpdateReceipt) -> None:
        current = (
            self.configuration_version,
            self.model_version,
            self.representation_version,
            self.policy_version,
        )
        previous = (
            receipt.previous_configuration_version,
            receipt.previous_model_version,
            receipt.previous_representation_version,
            receipt.previous_policy_version,
        )
        if current != previous:
            raise StateTransitionError("stale receipt")


class _Ledger:
    def __init__(self, transition: EpistemicTransition) -> None:
        self.canonical = transition
        self.updates: list[UpdateReceipt] = []
        self.fail_append = False
        self.fail_commit: str | None = None
        self._lock = RLock()

    def transaction_lock(self) -> RLock:
        return self._lock

    def get_transition(self, transition_id: str) -> EpistemicTransition:
        if transition_id != self.canonical.transition_id:
            raise KeyError(transition_id)
        return self.canonical

    def append_update(self, receipt: UpdateReceipt) -> None:
        prepared = self.prepare_update(receipt)
        self.commit_prepared_update(prepared)

    def prepare_update(self, receipt: UpdateReceipt) -> tuple[UpdateReceipt, int]:
        if receipt.transitions != (self.canonical,) or receipt.transitions[0] is not self.canonical:
            raise ValueError("noncanonical receipt")
        if any(update.receipt_id == receipt.receipt_id for update in self.updates):
            raise ValueError("duplicate receipt")
        return (receipt, len(self.updates))

    def commit_prepared_update(self, prepared: tuple[UpdateReceipt, int]) -> None:
        receipt, _ = prepared
        if self.fail_append:
            raise ValueError("durable ledger unavailable")
        if self.fail_commit == "before":
            raise ValueError("ledger commit failed before append")
        self.updates.append(receipt)
        if self.fail_commit == "after":
            raise ValueError("ledger commit failed after append")

    def rollback_prepared_update(self, prepared: tuple[UpdateReceipt, int]) -> None:
        _, previous_length = prepared
        del self.updates[previous_length:]


class _Learner:
    def __init__(
        self,
        *,
        source_digest: str | None = None,
        receipt_model_version: str = "model-v2",
        candidate_version: str = "model-v2",
        candidate_payload: bytes = b"trained-model",
        receipt_transition: EpistemicTransition | None = None,
        failure: BaseException | None = None,
        barrier: Barrier | None = None,
    ) -> None:
        self.source_digest = source_digest
        self.receipt_model_version = receipt_model_version
        self.candidate_version = candidate_version
        self.candidate_payload = candidate_payload
        self.receipt_transition = receipt_transition
        self.failure = failure
        self.barrier = barrier
        self.calls = 0
        self.seen_model: ModelState | None = None

    def prepare(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
        current_model: ModelState,
    ) -> PreparedLearningUpdate:
        self.calls += 1
        self.seen_model = current_model
        if self.failure is not None:
            raise self.failure
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        receipt_transitions = tuple(transitions) if self.receipt_transition is None else (self.receipt_transition,)
        receipt = cast(
            UpdateReceipt,
            _Receipt(
                receipt_id="receipt-v2",
                agent_id=snapshot.agent_id,
                transitions=receipt_transitions,
                status=UpdateStatus.APPLIED,
                previous_configuration_version=snapshot.configuration_version,
                new_configuration_version="configuration-v2",
                previous_model_version=snapshot.model_version,
                new_model_version=self.receipt_model_version,
                previous_representation_version=snapshot.representation_version,
                new_representation_version=snapshot.representation_version,
                previous_policy_version=snapshot.policy_version,
                new_policy_version=snapshot.policy_version,
                started_at=snapshot.captured_at,
                completed_at=TimePoint(
                    tick=snapshot.captured_at.tick + 1,
                    clock_id=snapshot.captured_at.clock_id,
                ),
            ),
        )
        return PreparedLearningUpdate(
            receipt=receipt,
            source_model_digest=(current_model.digest if self.source_digest is None else self.source_digest),
            candidate_model=ModelState(
                version=self.candidate_version,
                payload=self.candidate_payload,
            ),
        )


@dataclass(slots=True)
class _Fixture:
    agent: EpistemicAgent
    state: _State
    ledger: _Ledger
    model: VersionedModelOwner
    learner: _Learner
    transition: EpistemicTransition


class _FailingModelOwner(VersionedModelOwner):
    def __init__(self, initial: ModelState, *, failure: str) -> None:
        super().__init__(initial)
        self.failure = failure

    def commit_swap(self, swap: Any) -> None:
        if self.failure == "before":
            raise ValueError("model commit failed before swap")
        super().commit_swap(swap)
        if self.failure == "after":
            raise ValueError("model commit failed after swap")


def _fixture(
    learner: _Learner | None = None,
    *,
    validator: Any = None,
    model_failure: str | None = None,
) -> _Fixture:
    transition = cast(EpistemicTransition, _Transition("transition-1"))
    state = _State()
    ledger = _Ledger(transition)
    transactional_learner = _Learner() if learner is None else learner
    initial_model = ModelState(version="model-v1", payload=b"cold-model")
    model: VersionedModelOwner
    if model_failure is None:
        model = VersionedModelOwner(initial_model, validator=validator)
    else:
        if validator is not None:
            raise ValueError("failure model fixture does not support a validator")
        model = _FailingModelOwner(initial_model, failure=model_failure)
    unused = cast(Any, object())
    agent = EpistemicAgent(
        state=cast(AgentState, state),
        policy=unused,
        belief_updater=unused,
        scorer=unused,
        effect_assessor=unused,
        learner=transactional_learner,
        experience_store=unused,
        ledger=cast(EpistemicLedger, ledger),
        identities=cast(CounterIdentitySource, unused),
        model=model,
    )
    return _Fixture(
        agent=agent,
        state=state,
        ledger=ledger,
        model=model,
        learner=transactional_learner,
        transition=transition,
    )


def _persistent_state(fixture: _Fixture) -> tuple[object, ...]:
    return (
        fixture.model.version,
        fixture.model.digest,
        fixture.model.checkpoint_bytes(),
        fixture.state.configuration_version,
        fixture.state.model_version,
        fixture.state.latest_update,
        tuple(fixture.ledger.updates),
    )


def test_transaction_commits_exact_candidate_bytes_version_and_lineage() -> None:
    fixture = _fixture()
    before = fixture.model.snapshot_state()

    receipt = fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert fixture.learner.seen_model is before
    assert receipt.transitions == (fixture.transition,)
    assert receipt.transitions[0] is fixture.transition
    assert fixture.ledger.updates == [receipt]
    assert fixture.state.latest_update is receipt
    assert fixture.state.configuration_version == "configuration-v2"
    assert fixture.state.model_version == "model-v2"
    assert fixture.model.version == "model-v2"
    assert fixture.model.checkpoint_bytes() == b"trained-model"
    assert fixture.model.digest == ModelState("model-v2", b"trained-model").digest

    restored = VersionedModelOwner.from_checkpoint(
        version=fixture.model.version,
        payload=fixture.model.checkpoint_bytes(),
    )
    assert restored.snapshot_state() == fixture.model.snapshot_state()


def test_learner_receives_only_immutable_checkpoint_state() -> None:
    state = ModelState(version="model-v1", payload=b"cold-model")

    with pytest.raises(FrozenInstanceError):
        state.version = "forged"  # type: ignore[misc]
    with pytest.raises(TypeError):
        state.payload[0] = 0  # type: ignore[index]


@pytest.mark.parametrize(
    ("learner", "message"),
    (
        (
            _Learner(source_digest="0" * 64),
            "different model digest",
        ),
        (
            _Learner(receipt_model_version="model-v3"),
            "does not identify the candidate",
        ),
        (
            _Learner(candidate_payload=b"cold-model"),
            "must change the model checkpoint bytes",
        ),
    ),
)
def test_invalid_prepared_update_leaves_all_persistent_state_unchanged(
    learner: _Learner,
    message: str,
) -> None:
    fixture = _fixture(learner)
    before = _persistent_state(fixture)

    with pytest.raises(RuntimeError, match=message):
        fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert _persistent_state(fixture) == before


def test_backend_candidate_rejection_happens_before_any_persistent_commit() -> None:
    def validate(state: ModelState) -> None:
        if state.version == "model-v2":
            raise ValueError("candidate cannot be deserialized")

    fixture = _fixture(validator=validate)
    before = _persistent_state(fixture)

    with pytest.raises(RuntimeError, match="failed owner validation") as captured:
        fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert isinstance(captured.value.__cause__, ValueError)
    assert _persistent_state(fixture) == before


def test_learner_failure_cannot_mutate_owned_model_or_runtime_records() -> None:
    fixture = _fixture(_Learner(failure=ValueError("optimizer failed")))
    before = _persistent_state(fixture)

    with pytest.raises(ValueError, match="optimizer failed"):
        fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert _persistent_state(fixture) == before


def test_ledger_failure_aborts_reservation_and_allows_a_later_clean_commit() -> None:
    fixture = _fixture()
    fixture.ledger.fail_append = True
    before = _persistent_state(fixture)

    with pytest.raises(ValueError, match="ledger unavailable"):
        fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert _persistent_state(fixture) == before
    fixture.ledger.fail_append = False
    fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))
    assert fixture.model.version == "model-v2"


@pytest.mark.parametrize(
    ("stage", "timing"),
    (
        ("ledger", "before"),
        ("ledger", "after"),
        ("state", "before"),
        ("state", "after"),
        ("model", "before"),
        ("model", "after"),
    ),
)
def test_failure_before_or_after_every_commit_stage_restores_all_participants(
    stage: str,
    timing: str,
) -> None:
    fixture = _fixture(model_failure=timing if stage == "model" else None)
    fixture.ledger.fail_commit = timing if stage == "ledger" else None
    fixture.state.fail_commit = timing if stage == "state" else None
    before = _persistent_state(fixture)

    with pytest.raises(ValueError, match=rf"{stage} commit failed {timing}"):
        fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert _persistent_state(fixture) == before

    fixture.ledger.fail_commit = None
    fixture.state.fail_commit = None
    if isinstance(fixture.model, _FailingModelOwner):
        fixture.model.failure = "disabled"
    fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))
    assert fixture.model.version == "model-v2"
    assert fixture.state.model_version == "model-v2"
    assert len(fixture.ledger.updates) == 1


def test_concurrent_learners_cannot_commit_a_preflight_from_stale_state() -> None:
    learner = _Learner(barrier=Barrier(2))
    fixture = _fixture(learner)
    fixture.state.preflight_barrier = Barrier(2)
    receipts: list[UpdateReceipt] = []
    failures: list[BaseException] = []

    def learn() -> None:
        try:
            receipts.append(
                fixture.agent.learn(
                    (fixture.transition,),
                    at=TimePoint(tick=4),
                )
            )
        except BaseException as error:
            failures.append(error)

    threads = (Thread(target=learn), Thread(target=learn))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(receipts) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "became invalid" in str(failures[0])
    assert fixture.ledger.updates == receipts
    assert fixture.state.latest_update is receipts[0]
    assert fixture.state.model_version == fixture.model.version == "model-v2"
    assert fixture.model.checkpoint_bytes() == b"trained-model"


def test_noncanonical_transition_is_rejected_before_the_learner_runs() -> None:
    fixture = _fixture()
    forged = cast(EpistemicTransition, _Transition(fixture.transition.transition_id))
    before = _persistent_state(fixture)

    with pytest.raises(RuntimeError, match="exact canonical"):
        fixture.agent.learn((forged,), at=TimePoint(tick=4))

    assert fixture.learner.calls == 0
    assert _persistent_state(fixture) == before


def test_receipt_must_return_the_exact_supplied_transition_objects() -> None:
    forged = cast(EpistemicTransition, _Transition("transition-1"))
    fixture = _fixture(_Learner(receipt_transition=forged))
    before = _persistent_state(fixture)

    with pytest.raises(RuntimeError, match="exactly the supplied transitions"):
        fixture.agent.learn((fixture.transition,), at=TimePoint(tick=4))

    assert _persistent_state(fixture) == before
