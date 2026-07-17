"""Mutable runtime custody for immutable agent snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import RLock

from prospect.domain import (
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    DecisionRecord,
    DomainInvariantError,
    IntendedAction,
    ResourceLedger,
    ResourceUse,
    TimePoint,
    UpdateReceipt,
    UpdateStatus,
)


class StateTransitionError(RuntimeError):
    """A runtime state mutation does not follow its immutable input records."""


@dataclass(frozen=True, slots=True)
class _LearningState:
    """Exact mutable fields touched by one learning receipt."""

    belief: Belief
    configuration_version: str
    memory_version: str
    model_version: str
    representation_version: str
    policy_version: str
    latest_update: UpdateReceipt | None
    last_time: TimePoint


@dataclass(frozen=True, slots=True)
class PreparedStateUpdate:
    """Opaque preflight token for one rollback-safe state update."""

    receipt: UpdateReceipt
    _owner_identity: object
    _generation: int
    _before: _LearningState


class AgentState:
    """Own the current belief, versions, pending intentions, and resource totals.

    This is intentionally the small mutable center of the runtime.  Every boundary
    crossing still produces an immutable domain record; callers and evaluators see
    only :class:`AgentSnapshot` values.
    """

    def __init__(self, initial: AgentSnapshot) -> None:
        self._agent_id = initial.agent_id
        self._belief = initial.belief
        self._configuration_version = initial.configuration_version
        self._memory_version = initial.memory_version
        self._knowledge_version = initial.knowledge_version
        self._model_version = initial.model_version
        self._representation_version = initial.representation_version
        self._policy_version = initial.policy_version
        self._resource_started_at = initial.resources.started_at
        self._resources = {(use.resource, use.unit): use.amount for use in initial.resources.uses}
        self._pending: dict[str, DecisionRecord] = {}
        self._latest_update = initial.latest_update
        self._last_time = initial.captured_at
        self._lock = RLock()
        self._transaction_identity = object()
        self._generation = 0
        for intention in initial.pending_intentions:
            raise ValueError(
                "initializing mutable state with orphan pending intentions is unsupported; "
                f"missing decision for {intention.intention_id!r}"
            )

    @property
    def agent_id(self) -> str:
        return str(self._agent_id)

    @property
    def current_belief_id(self) -> str:
        with self._lock:
            return str(self._belief.belief_id)

    def transaction_lock(self) -> AbstractContextManager[bool]:
        """Return the state lock for a short composed runtime transaction."""

        return self._lock

    def snapshot(self, agent_id: str, at: TimePoint) -> AgentSnapshot:
        """Capture a coherent immutable view at or after the latest mutation."""

        if agent_id != self._agent_id:
            raise StateTransitionError("cannot snapshot another agent")
        with self._lock:
            self._require_not_before("snapshot", at, self._last_time)
            pending = tuple(decision.intended_action for _, decision in sorted(self._pending.items()))
            resource_uses = tuple(
                ResourceUse(resource=resource, amount=amount, unit=unit)
                for (resource, unit), amount in sorted(self._resources.items())
            )
            fingerprint = self._snapshot_fingerprint(
                at=at,
                pending=pending,
                resource_uses=resource_uses,
            )
            return AgentSnapshot(
                snapshot_id=f"{self._agent_id}:snapshot:{fingerprint}",
                agent_id=self._agent_id,
                captured_at=at,
                belief=self._belief,
                configuration_version=self._configuration_version,
                memory_version=self._memory_version,
                knowledge_version=self._knowledge_version,
                model_version=self._model_version,
                representation_version=self._representation_version,
                policy_version=self._policy_version,
                resources=ResourceLedger(
                    ledger_id=f"{self._agent_id}:resources:{fingerprint}",
                    started_at=self._resource_started_at,
                    completed_at=at,
                    uses=resource_uses,
                ),
                pending_intentions=pending,
                latest_update=self._latest_update,
            )

    def register_decision(self, decision: DecisionRecord) -> None:
        """Register an explicit pending decision so it can be consumed once."""

        with self._lock:
            if decision.agent_id != self._agent_id:
                raise StateTransitionError("decision belongs to another agent")
            if decision.belief.belief_id != self._belief.belief_id:
                raise StateTransitionError("decision was not made from the current belief")
            if decision.policy_version != self._policy_version:
                raise StateTransitionError("decision uses a stale policy version")
            if decision.decision_id in self._pending:
                raise StateTransitionError("decision is already pending")
            if any(
                pending.intended_action.intention_id == decision.intended_action.intention_id
                for pending in self._pending.values()
            ):
                raise StateTransitionError("intention identity is already pending")
            self._require_not_before("decision registration", decision.decided_at, self._last_time)
            self._pending[decision.decision_id] = decision
            self._last_time = decision.decided_at
            self._generation += 1

    def consume_decision(self, decision: DecisionRecord) -> None:
        """Consume exactly the pending record supplied to observation handling."""

        with self._lock:
            canonical = self._pending.get(decision.decision_id)
            if canonical is None:
                raise StateTransitionError("decision is not pending")
            if canonical is not decision:
                raise StateTransitionError("decision is not the pending canonical record")
            del self._pending[decision.decision_id]
            self._generation += 1

    def assimilate(self, update: BeliefUpdate) -> None:
        """Advance the current posterior after one real experience."""

        with self._lock:
            if update.prior.belief_id != self._belief.belief_id:
                raise StateTransitionError("belief update uses a stale prior")
            if update.posterior.agent_id != self._agent_id:
                raise StateTransitionError("belief update belongs to another agent")
            self._require_not_before("belief update", update.updated_at, self._last_time)
            self._belief = update.posterior
            self._memory_version = update.posterior.information_set.memory_version
            self._last_time = update.updated_at
            self._generation += 1

    def apply_update(self, receipt: UpdateReceipt) -> None:
        """Apply only the version transition documented by a learning receipt."""

        with self._lock:
            self._validate_update_locked(receipt)
            self._apply_update_locked(receipt)
            self._generation += 1

    def validate_update(self, receipt: UpdateReceipt) -> None:
        """Preflight a learning receipt without mutating runtime state."""

        with self._lock:
            self._validate_update_locked(receipt)

    def prepare_update(self, receipt: UpdateReceipt) -> PreparedStateUpdate:
        """Preflight an update and capture the exact state needed for rollback.

        Callers that compose this with another mutable custodian must hold
        :meth:`transaction_lock` continuously through prepare, commit, and either
        success or rollback. The generation check also rejects a stale token if a
        caller fails to honor that rule.
        """

        with self._lock:
            self._validate_update_locked(receipt)
            return PreparedStateUpdate(
                receipt=receipt,
                _owner_identity=self._transaction_identity,
                _generation=self._generation,
                _before=self._learning_state_locked(),
            )

    def commit_prepared_update(self, prepared: PreparedStateUpdate) -> None:
        """Apply a preflighted update while retaining enough data to undo it."""

        with self._lock:
            self._require_prepared_owner(prepared)
            if self._generation != prepared._generation:
                raise StateTransitionError("runtime state changed after update preflight")
            try:
                self._apply_update_locked(prepared.receipt)
            except BaseException:
                self._restore_learning_state_locked(prepared._before)
                raise
            self._generation += 1

    def rollback_prepared_update(self, prepared: PreparedStateUpdate) -> None:
        """Restore a prepared update whether it failed before or after commit."""

        with self._lock:
            self._require_prepared_owner(prepared)
            if self._generation == prepared._generation:
                return
            if self._generation != prepared._generation + 1 or self._latest_update is not prepared.receipt:
                raise StateTransitionError("cannot rollback state after a later mutation")
            self._restore_learning_state_locked(prepared._before)
            self._generation = prepared._generation

    def add_resource(self, resource: str, amount: float, unit: str, *, at: TimePoint) -> None:
        """Accumulate a nonnegative resource quantity at a causal time."""

        use = ResourceUse(resource=resource, amount=amount, unit=unit)
        with self._lock:
            self._require_not_before("resource use", at, self._last_time)
            key = (use.resource, use.unit)
            self._resources[key] = self._resources.get(key, 0.0) + use.amount
            self._last_time = at
            self._generation += 1

    def pending_decisions(self) -> Iterable[DecisionRecord]:
        with self._lock:
            return tuple(self._pending.values())

    def _validate_update_locked(self, receipt: UpdateReceipt) -> None:
        if receipt.agent_id != self._agent_id:
            raise StateTransitionError("learning receipt belongs to another agent")
        current = (
            self._configuration_version,
            self._model_version,
            self._representation_version,
            self._policy_version,
        )
        previous = (
            receipt.previous_configuration_version,
            receipt.previous_model_version,
            receipt.previous_representation_version,
            receipt.previous_policy_version,
        )
        if current != previous:
            raise StateTransitionError("learning receipt previous versions do not match current state")
        self._require_not_before(
            "learning receipt",
            receipt.completed_at,
            self._last_time,
        )
        resulting = receipt.resulting_belief
        if resulting is not None:
            current_evidence = {observation.observation_id for observation in self._belief.information_set.observations}
            resulting_evidence = {observation.observation_id for observation in resulting.information_set.observations}
            if not current_evidence.issubset(resulting_evidence):
                raise StateTransitionError("learning receipt resulting belief drops current-state evidence")
            self._require_not_before(
                "resulting belief information",
                resulting.information_set.as_of,
                self._belief.information_set.as_of,
            )

    def _apply_update_locked(self, receipt: UpdateReceipt) -> None:
        if receipt.status is not UpdateStatus.REJECTED:
            self._configuration_version = receipt.new_configuration_version
            self._model_version = receipt.new_model_version
            self._representation_version = receipt.new_representation_version
            self._policy_version = receipt.new_policy_version
            if receipt.resulting_belief is not None:
                self._belief = receipt.resulting_belief
                self._memory_version = receipt.resulting_belief.information_set.memory_version
        self._latest_update = receipt
        self._last_time = receipt.completed_at

    def _learning_state_locked(self) -> _LearningState:
        return _LearningState(
            belief=self._belief,
            configuration_version=self._configuration_version,
            memory_version=self._memory_version,
            model_version=self._model_version,
            representation_version=self._representation_version,
            policy_version=self._policy_version,
            latest_update=self._latest_update,
            last_time=self._last_time,
        )

    def _restore_learning_state_locked(self, previous: _LearningState) -> None:
        self._belief = previous.belief
        self._configuration_version = previous.configuration_version
        self._memory_version = previous.memory_version
        self._model_version = previous.model_version
        self._representation_version = previous.representation_version
        self._policy_version = previous.policy_version
        self._latest_update = previous.latest_update
        self._last_time = previous.last_time

    def _require_prepared_owner(self, prepared: PreparedStateUpdate) -> None:
        if prepared._owner_identity is not self._transaction_identity:
            raise StateTransitionError("state update was not prepared by this owner")

    def _snapshot_fingerprint(
        self,
        *,
        at: TimePoint,
        pending: tuple[IntendedAction, ...],
        resource_uses: tuple[ResourceUse, ...],
    ) -> str:
        state = {
            "agent_id": self._agent_id,
            "at": [at.clock_id, at.tick],
            "belief_id": self._belief.belief_id,
            "versions": [
                self._configuration_version,
                self._memory_version,
                self._knowledge_version,
                self._model_version,
                self._representation_version,
                self._policy_version,
            ],
            "pending_intention_ids": [intention.intention_id for intention in pending],
            "resources": [[use.resource, use.unit, use.amount] for use in resource_uses],
            "resource_started_at": [
                self._resource_started_at.clock_id,
                self._resource_started_at.tick,
            ],
            "latest_update_id": (None if self._latest_update is None else self._latest_update.receipt_id),
        }
        payload = json.dumps(
            state,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()[:24]

    @staticmethod
    def _require_not_before(label: str, point: TimePoint, previous: TimePoint) -> None:
        if point.clock_id != previous.clock_id:
            raise DomainInvariantError(f"{label} uses clock {point.clock_id!r}, expected {previous.clock_id!r}")
        if point.tick < previous.tick:
            raise StateTransitionError(f"{label} at tick {point.tick} precedes runtime state at tick {previous.tick}")


__all__ = ("AgentState", "PreparedStateUpdate", "StateTransitionError")
