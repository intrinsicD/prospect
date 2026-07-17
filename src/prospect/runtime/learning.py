"""Failure-atomic custody for opaque, versioned model state."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol, runtime_checkable

from prospect.domain import AgentSnapshot, EpistemicTransition, UpdateReceipt

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ModelTransactionError(RuntimeError):
    """A model candidate cannot be reserved or committed as declared."""


@dataclass(frozen=True, slots=True)
class ModelState:
    """An immutable, self-digesting checkpoint of one predictive model.

    The payload is deliberately opaque to Prospect. Backend adapters may encode a
    framework state dict, optimizer-free inference artifact, or any other canonical
    representation, while the runtime can still establish byte identity.
    """

    version: str
    payload: bytes
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.version or not self.version.strip():
            raise ValueError("model state version must be nonempty")
        if not isinstance(self.payload, bytes):
            raise TypeError("model state payload must be bytes")
        object.__setattr__(self, "digest", hashlib.sha256(self.payload).hexdigest())


@dataclass(frozen=True, slots=True)
class PreparedLearningUpdate:
    """A learner proposal built without access to the live mutable model."""

    receipt: UpdateReceipt
    source_model_digest: str
    candidate_model: ModelState

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.source_model_digest) is None:
            raise ValueError("source_model_digest must be lowercase SHA-256 hexadecimal")


@runtime_checkable
class TransactionalLearner(Protocol):
    """Prepare new persistent state from an immutable model checkpoint."""

    def prepare(
        self,
        snapshot: AgentSnapshot,
        transitions: Sequence[EpistemicTransition],
        current_model: ModelState,
    ) -> PreparedLearningUpdate: ...


@dataclass(frozen=True, slots=True)
class PreparedModelSwap:
    """Opaque reservation accepted by exactly one model owner."""

    expected: ModelState
    candidate: ModelState
    _owner_identity: object
    _reservation_id: int


@runtime_checkable
class OwnedModel(Protocol):
    """Exclusive custody and atomic replacement of persistent model bytes."""

    def snapshot_state(self) -> ModelState:
        """Return the immutable state currently used by the owner."""
        ...

    def transaction_lock(self) -> AbstractContextManager[bool]:
        """Return the owner lock for a short composed runtime transaction."""
        ...

    def prepare_swap(
        self,
        expected: ModelState,
        candidate: ModelState,
    ) -> PreparedModelSwap:
        """Validate and reserve a swap without changing persistent state."""
        ...

    def commit_swap(self, swap: PreparedModelSwap) -> None:
        """Commit a valid reservation with no backend work or fallible mutation."""
        ...

    def abort_swap(self, swap: PreparedModelSwap) -> None:
        """Release an uncommitted reservation without changing persistent state."""
        ...

    def rollback_swap(self, swap: PreparedModelSwap) -> None:
        """Restore the expected state before or after a reserved swap commits."""
        ...


ModelStateValidator = Callable[[ModelState], None]


class VersionedModelOwner:
    """Own one opaque model checkpoint and replace it by reserved pointer swap.

    Candidate deserialization or backend-specific validation belongs in
    ``validator`` and runs during :meth:`prepare_swap`. Once that method returns,
    :meth:`commit_swap` performs only two in-memory assignments for the exact
    reservation. It never calls backend code.
    """

    def __init__(
        self,
        initial: ModelState,
        *,
        validator: ModelStateValidator | None = None,
    ) -> None:
        if validator is not None:
            validator(initial)
        self._state = initial
        self._validator = validator
        self._identity = object()
        self._next_reservation_id = 0
        self._reservation: PreparedModelSwap | None = None
        self._lock = RLock()

    @classmethod
    def from_checkpoint(
        cls,
        *,
        version: str,
        payload: bytes,
        validator: ModelStateValidator | None = None,
    ) -> VersionedModelOwner:
        """Construct fresh ownership from externally integrity-checked bytes."""

        return cls(ModelState(version=version, payload=payload), validator=validator)

    @property
    def version(self) -> str:
        return self.snapshot_state().version

    @property
    def digest(self) -> str:
        return self.snapshot_state().digest

    def checkpoint_bytes(self) -> bytes:
        """Return the exact current bytes for a checkpoint component."""

        return self.snapshot_state().payload

    def transaction_lock(self) -> AbstractContextManager[bool]:
        """Return the owner lock for a short composed runtime transaction."""

        return self._lock

    def snapshot_state(self) -> ModelState:
        with self._lock:
            return self._state

    def prepare_swap(
        self,
        expected: ModelState,
        candidate: ModelState,
    ) -> PreparedModelSwap:
        with self._lock:
            if self._reservation is not None:
                raise ModelTransactionError("another model swap is already reserved")
            if self._state != expected:
                raise ModelTransactionError("live model state changed after the learner snapshot")
            if candidate.version == expected.version:
                raise ModelTransactionError("changed model state requires a new model version")
            if candidate.payload == expected.payload:
                raise ModelTransactionError("new model version must change checkpoint bytes")
            if self._validator is not None:
                self._validator(candidate)
            if self._reservation is not None or self._state != expected:
                raise ModelTransactionError("live model state changed during candidate validation")
            reservation = PreparedModelSwap(
                expected=expected,
                candidate=candidate,
                _owner_identity=self._identity,
                _reservation_id=self._next_reservation_id,
            )
            self._next_reservation_id += 1
            self._reservation = reservation
            return reservation

    def commit_swap(self, swap: PreparedModelSwap) -> None:
        """Commit an accepted reservation without invoking fallible backend code."""

        with self._lock:
            if swap._owner_identity is not self._identity or self._reservation is not swap:
                raise ModelTransactionError("model swap was not reserved by this owner")
            self._state = swap.candidate
            self._reservation = None

    def abort_swap(self, swap: PreparedModelSwap) -> None:
        with self._lock:
            if swap._owner_identity is not self._identity or self._reservation is not swap:
                raise ModelTransactionError("model swap was not reserved by this owner")
            self._reservation = None

    def rollback_swap(self, swap: PreparedModelSwap) -> None:
        """Restore ``expected`` whether a reserved swap is pending or committed.

        The transaction coordinator holds :meth:`transaction_lock`, so no later
        model mutation can be mistaken for the commit being compensated.
        """

        with self._lock:
            if swap._owner_identity is not self._identity:
                raise ModelTransactionError("model swap was not reserved by this owner")
            if self._reservation is swap:
                if self._state not in (swap.expected, swap.candidate):
                    raise ModelTransactionError("reserved model state cannot be rolled back")
                self._state = swap.expected
                self._reservation = None
                return
            if self._reservation is None and self._state == swap.candidate:
                self._state = swap.expected
                return
            raise ModelTransactionError("model swap is neither pending nor the latest commit")


__all__ = (
    "ModelState",
    "ModelStateValidator",
    "ModelTransactionError",
    "OwnedModel",
    "PreparedLearningUpdate",
    "PreparedModelSwap",
    "TransactionalLearner",
    "VersionedModelOwner",
)
