"""Append-only linkage ledger for epistemic transitions and learning updates."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from threading import RLock

from prospect.domain import (
    EpistemicTransition,
    ExperienceEvent,
    ExperienceStore,
    TimePoint,
    UpdateReceipt,
)

from .memory import (
    CausalOrderError,
    DuplicateRecordError,
    RecordNotFoundError,
    StorageError,
    _ensure_same_clock,
    _require_identifier,
)


class LedgerIntegrityError(StorageError):
    """A ledger record does not link to its canonical prerequisite."""


@dataclass(frozen=True, slots=True)
class PreparedLedgerUpdate:
    """Opaque preflight token for one rollback-safe update append."""

    receipt: UpdateReceipt
    _owner_identity: object
    _generation: int
    _previous_last: TimePoint | None


class EpistemicLedger:
    """Canonical runtime ledger above the real-experience store.

    Transitions may optionally be anchored to an ``ExperienceStore``.  When
    anchored, the transition must reference the exact canonical immutable event.
    Update receipts likewise reference exact transitions already in this ledger.
    These identity checks prevent reconstructed records with reused identifiers
    from silently changing evidence lineage.
    """

    def __init__(self, experience_store: ExperienceStore | None = None) -> None:
        self._experience_store = experience_store
        self._transitions_by_id: dict[str, EpistemicTransition] = {}
        self._transitions_by_agent: dict[str, list[EpistemicTransition]] = {}
        self._transition_last: dict[str, TimePoint] = {}
        self._updates_by_id: dict[str, UpdateReceipt] = {}
        self._updates_by_agent: dict[str, list[UpdateReceipt]] = {}
        self._update_last: dict[str, TimePoint] = {}
        self._lock = RLock()
        self._transaction_identity = object()
        self._generation = 0

    @property
    def transition_count(self) -> int:
        with self._lock:
            return len(self._transitions_by_id)

    @property
    def update_count(self) -> int:
        with self._lock:
            return len(self._updates_by_id)

    def transaction_lock(self) -> AbstractContextManager[bool]:
        """Return the ledger lock for a short composed runtime transaction."""

        return self._lock

    def append_transition(self, transition: EpistemicTransition) -> None:
        """Append a completed transition after checking evidence custody and order."""

        agent_id = transition.experience.agent_id
        with self._lock:
            if transition.transition_id in self._transitions_by_id:
                raise DuplicateRecordError(f"transition id {transition.transition_id!r} is already stored")
            if self._experience_store is not None:
                try:
                    canonical = self._experience_store.get(transition.experience.experience_id)
                except KeyError as error:
                    raise LedgerIntegrityError(
                        "transition experience is absent from the canonical experience store"
                    ) from error
                if canonical is not transition.experience:
                    raise LedgerIntegrityError("transition does not reference the canonical experience object")
            self._check_order(
                label="transition append",
                agent_id=agent_id,
                point=transition.created_at,
                previous=self._transition_last.get(agent_id),
                record_id=transition.transition_id,
            )
            self._transitions_by_id[transition.transition_id] = transition
            self._transitions_by_agent.setdefault(agent_id, []).append(transition)
            self._transition_last[agent_id] = transition.created_at
            self._generation += 1

    def append_rehydrated_transition(self, transition: EpistemicTransition) -> EpistemicTransition:
        """Relink a deserialized transition to canonical experience after restart.

        The supplied copy's episode/evidence/execution envelope must identify the
        same event.  Its experience object and prior belief are then discarded in
        favor of the canonical store graph before normal append checks run.
        """

        if self._experience_store is None:
            raise LedgerIntegrityError("rehydration requires a canonical experience store")
        try:
            canonical = self._experience_store.get(transition.experience.experience_id)
        except KeyError as error:
            raise LedgerIntegrityError("transition experience is absent from the canonical experience store") from error
        if _experience_envelope(canonical) != _experience_envelope(transition.experience):
            raise LedgerIntegrityError("deserialized transition experience disagrees with its canonical envelope")
        if canonical.decision is None:
            raise LedgerIntegrityError("epistemic transition canonical experience has no decision")
        belief_update = replace(
            transition.belief_update,
            prior=canonical.decision.belief,
            experience=canonical,
        )
        rehydrated = replace(
            transition,
            experience=canonical,
            belief_update=belief_update,
        )
        self.append_transition(rehydrated)
        return rehydrated

    def append_update(self, receipt: UpdateReceipt) -> None:
        """Append a learning receipt whose input transitions already exist."""

        with self._lock:
            self._validate_update_locked(receipt)
            self._append_update_locked(receipt)
            self._generation += 1

    def prepare_update(self, receipt: UpdateReceipt) -> PreparedLedgerUpdate:
        """Preflight an append and capture the ordering state needed for rollback."""

        with self._lock:
            self._validate_update_locked(receipt)
            return PreparedLedgerUpdate(
                receipt=receipt,
                _owner_identity=self._transaction_identity,
                _generation=self._generation,
                _previous_last=self._update_last.get(receipt.agent_id),
            )

    def commit_prepared_update(self, prepared: PreparedLedgerUpdate) -> None:
        """Append a preflighted receipt while retaining an exact rollback path."""

        with self._lock:
            self._require_prepared_owner(prepared)
            if self._generation != prepared._generation:
                raise LedgerIntegrityError("ledger changed after update preflight")
            try:
                self._append_update_locked(prepared.receipt)
            except BaseException:
                self._remove_appended_update_locked(prepared)
                raise
            self._generation += 1

    def rollback_prepared_update(self, prepared: PreparedLedgerUpdate) -> None:
        """Remove a prepared append whether failure occurred before or after commit."""

        with self._lock:
            self._require_prepared_owner(prepared)
            if self._generation == prepared._generation:
                return
            if self._generation != prepared._generation + 1:
                raise LedgerIntegrityError("cannot rollback ledger after a later mutation")
            if self._updates_by_id.get(prepared.receipt.receipt_id) is not prepared.receipt:
                raise LedgerIntegrityError("cannot rollback ledger after an unrelated mutation")
            self._remove_appended_update_locked(prepared)
            self._generation = prepared._generation

    def append_rehydrated_update(self, receipt: UpdateReceipt) -> UpdateReceipt:
        """Relink a deserialized receipt to canonical transitions after restart."""

        canonical_transitions: list[EpistemicTransition] = []
        with self._lock:
            for transition in receipt.transitions:
                canonical = self._transitions_by_id.get(transition.transition_id)
                if canonical is None:
                    raise LedgerIntegrityError(f"update references unknown transition {transition.transition_id!r}")
                if _transition_envelope(canonical) != _transition_envelope(transition):
                    raise LedgerIntegrityError("deserialized update transition disagrees with its canonical envelope")
                canonical_transitions.append(canonical)
        rehydrated = replace(receipt, transitions=tuple(canonical_transitions))
        self.append_update(rehydrated)
        return rehydrated

    def get_transition(self, transition_id: str) -> EpistemicTransition:
        _require_identifier("transition_id", transition_id)
        with self._lock:
            try:
                return self._transitions_by_id[transition_id]
            except KeyError as error:
                raise RecordNotFoundError(f"transition id {transition_id!r} is not stored") from error

    def get_update(self, receipt_id: str) -> UpdateReceipt:
        _require_identifier("receipt_id", receipt_id)
        with self._lock:
            try:
                return self._updates_by_id[receipt_id]
            except KeyError as error:
                raise RecordNotFoundError(f"update receipt id {receipt_id!r} is not stored") from error

    def transitions(self, agent_id: str, as_of: TimePoint) -> Sequence[EpistemicTransition]:
        _require_identifier("agent_id", agent_id)
        with self._lock:
            records = self._transitions_by_agent.get(agent_id)
            if not records:
                return ()
            _ensure_same_clock("transition history cutoff", as_of, records[0].created_at)
            return tuple(record for record in records if record.created_at.tick <= as_of.tick)

    def updates(self, agent_id: str, as_of: TimePoint) -> Sequence[UpdateReceipt]:
        _require_identifier("agent_id", agent_id)
        with self._lock:
            records = self._updates_by_agent.get(agent_id)
            if not records:
                return ()
            _ensure_same_clock("update history cutoff", as_of, records[0].completed_at)
            return tuple(record for record in records if record.completed_at.tick <= as_of.tick)

    def _validate_update_locked(self, receipt: UpdateReceipt) -> None:
        if receipt.receipt_id in self._updates_by_id:
            raise DuplicateRecordError(f"update receipt id {receipt.receipt_id!r} is already stored")
        for transition in receipt.transitions:
            canonical = self._transitions_by_id.get(transition.transition_id)
            if canonical is None:
                raise LedgerIntegrityError(f"update references unknown transition {transition.transition_id!r}")
            if canonical is not transition:
                raise LedgerIntegrityError("update does not reference the canonical transition object")
        if receipt.rollback_of is not None:
            rolled_back = self._updates_by_id.get(receipt.rollback_of)
            if rolled_back is None:
                raise LedgerIntegrityError(f"rollback references unknown update {receipt.rollback_of!r}")
            if rolled_back.agent_id != receipt.agent_id:
                raise LedgerIntegrityError("rollback references another agent's update")
        self._check_order(
            label="update append",
            agent_id=receipt.agent_id,
            point=receipt.completed_at,
            previous=self._update_last.get(receipt.agent_id),
            record_id=receipt.receipt_id,
        )

    def _append_update_locked(self, receipt: UpdateReceipt) -> None:
        self._updates_by_id[receipt.receipt_id] = receipt
        self._updates_by_agent.setdefault(receipt.agent_id, []).append(receipt)
        self._update_last[receipt.agent_id] = receipt.completed_at

    def _remove_appended_update_locked(self, prepared: PreparedLedgerUpdate) -> None:
        receipt = prepared.receipt
        stored = self._updates_by_id.get(receipt.receipt_id)
        if stored is None:
            return
        records = self._updates_by_agent.get(receipt.agent_id)
        if stored is not receipt:
            raise LedgerIntegrityError("cannot rollback a nonterminal or noncanonical update")
        if records:
            if records[-1] is not receipt:
                raise LedgerIntegrityError("cannot rollback a nonterminal or noncanonical update")
            records.pop()
        if records == []:
            del self._updates_by_agent[receipt.agent_id]
        del self._updates_by_id[receipt.receipt_id]
        if prepared._previous_last is None:
            self._update_last.pop(receipt.agent_id, None)
        else:
            self._update_last[receipt.agent_id] = prepared._previous_last

    def _require_prepared_owner(self, prepared: PreparedLedgerUpdate) -> None:
        if prepared._owner_identity is not self._transaction_identity:
            raise LedgerIntegrityError("ledger update was not prepared by this owner")

    @staticmethod
    def _check_order(
        *,
        label: str,
        agent_id: str,
        point: TimePoint,
        previous: TimePoint | None,
        record_id: str,
    ) -> None:
        if previous is None:
            return
        _ensure_same_clock(label, point, previous)
        if point.tick < previous.tick:
            raise CausalOrderError(
                f"{record_id!r} is at tick {point.tick}, before {agent_id!r}'s "
                f"last {label.removesuffix(' append')} at tick {previous.tick}"
            )


def _experience_envelope(experience: ExperienceEvent) -> tuple[object, ...]:
    decision = experience.decision
    execution = experience.execution
    return (
        experience.experience_id,
        experience.agent_id,
        experience.run_id,
        experience.task_id,
        experience.episode_id,
        experience.step_index,
        experience.kind,
        experience.terminated,
        experience.truncated,
        experience.discount,
        experience.behavior_policy_version,
        experience.closed_at,
        experience.observation.observation_id,
        experience.observation.evidence.evidence_id,
        experience.outcome.outcome_id,
        experience.outcome.evidence.evidence_id,
        None if decision is None else decision.decision_id,
        None if execution is None else execution.execution_id,
    )


def _transition_envelope(
    transition: EpistemicTransition,
) -> tuple[object, ...]:
    return (
        transition.transition_id,
        _experience_envelope(transition.experience),
        transition.belief_update.update_id,
        transition.belief_update.prior.belief_id,
        transition.belief_update.posterior.belief_id,
        tuple(score.score_id for score in transition.proper_scores),
        tuple(effect.effect_id for effect in transition.effects),
        transition.created_at,
    )


__all__ = ("EpistemicLedger", "LedgerIntegrityError", "PreparedLedgerUpdate")
