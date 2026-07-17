"""Decision policies over explicit utility and information-value decompositions."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from threading import RLock
from typing import Protocol, runtime_checkable

from prospect.domain import (
    AgentSnapshot,
    CandidateAssessment,
    DecisionRecord,
    Goal,
    IntendedAction,
    TimePoint,
)

_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class DecisionError(RuntimeError):
    """A policy cannot produce a causally and semantically valid decision."""


class NoAdmissibleActionError(DecisionError):
    """Every assessed candidate violates a hard constraint."""


@runtime_checkable
class CandidateAssessor(Protocol):
    """Build complete candidate assessments from a frozen agent snapshot."""

    def assess(self, snapshot: AgentSnapshot, goal: Goal) -> Sequence[CandidateAssessment]: ...


class CounterIdentitySource:
    """Deterministic, checkpointable record identities.

    UUIDs are useful for globally distributed systems, but an incrementing
    namespace is easier to replay and compare in a controlled research run.  The
    counter is itself state and therefore has explicit dump/restore methods.
    """

    def __init__(self, namespace: str, *, next_counter: int = 0) -> None:
        if not namespace or not namespace.strip():
            raise ValueError("identity namespace must be nonempty")
        if next_counter < 0:
            raise ValueError("identity counter must be nonnegative")
        self._namespace = namespace
        self._next_counter = next_counter
        self._lock = RLock()

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def next_counter(self) -> int:
        with self._lock:
            return self._next_counter

    def next(self, label: str) -> str:
        """Return one stable identity and advance the persisted counter."""

        if _LABEL.fullmatch(label) is None:
            raise ValueError("identity label must be alphanumeric with optional dot, underscore, or hyphen")
        with self._lock:
            identity = f"{self._namespace}:{label}:{self._next_counter}"
            self._next_counter += 1
            return identity

    def checkpoint_bytes(self) -> bytes:
        """Return canonical JSON suitable for a checkpoint component payload."""

        with self._lock:
            state = {
                "namespace": self._namespace,
                "next_counter": self._next_counter,
                "schema_version": 1,
            }
        return (
            json.dumps(
                state,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )

    def restore_bytes(self, payload: bytes) -> None:
        """Restore a counter after strict schema and namespace validation."""

        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("identity checkpoint is not valid JSON") from error
        if (
            not isinstance(raw, dict)
            or set(raw) != {"namespace", "next_counter", "schema_version"}
            or raw.get("schema_version") != 1
            or raw.get("namespace") != self._namespace
        ):
            raise ValueError("identity checkpoint schema or namespace mismatch")
        counter = raw.get("next_counter")
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            raise ValueError("identity checkpoint counter must be a nonnegative integer")
        if payload != self._canonical_payload(counter):
            raise ValueError("identity checkpoint is not canonical JSON")
        with self._lock:
            self._next_counter = counter

    def _canonical_payload(self, counter: int) -> bytes:
        state = {
            "namespace": self._namespace,
            "next_counter": counter,
            "schema_version": 1,
        }
        return (
            json.dumps(
                state,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )


class MaxValuePolicy:
    """Select the highest-total admissible assessment with deterministic ties."""

    def __init__(
        self,
        *,
        agent_id: str,
        policy_version: str,
        assessor: CandidateAssessor,
        identities: CounterIdentitySource,
    ) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must be nonempty")
        if not policy_version or not policy_version.strip():
            raise ValueError("policy_version must be nonempty")
        self._agent_id = agent_id
        self._policy_version = policy_version
        self._assessor = assessor
        self._identities = identities

    @property
    def policy_version(self) -> str:
        return self._policy_version

    def decide(self, snapshot: AgentSnapshot, goal: Goal) -> DecisionRecord:
        """Evaluate every candidate, reject prohibited actions, and record why."""

        if snapshot.agent_id != self._agent_id:
            raise DecisionError("policy received another agent's snapshot")
        if snapshot.policy_version != self._policy_version:
            raise DecisionError("policy version does not match the frozen decision snapshot")
        alternatives = tuple(self._assessor.assess(snapshot, goal))
        if not alternatives:
            raise DecisionError("candidate assessor returned no alternatives")
        admissible = tuple(assessment for assessment in alternatives if assessment.admissible)
        if not admissible:
            reasons = sorted({reason for assessment in alternatives for reason in assessment.constraint_reasons})
            suffix = f": {', '.join(reasons)}" if reasons else ""
            raise NoAdmissibleActionError(f"all candidate actions are prohibited{suffix}")

        # Sorting action IDs first gives a deterministic lexical tie break.
        selected = min(
            admissible,
            key=lambda assessment: (
                -assessment.total_value,
                assessment.action.action_id,
            ),
        )
        decided_at = _latest_time(
            "decision",
            (
                snapshot.captured_at,
                goal.issued_at,
                *(assessment.assessed_at for assessment in alternatives),
            ),
        )
        intention = IntendedAction(
            intention_id=self._identities.next("intention"),
            agent_id=self._agent_id,
            action=selected.action,
            intended_at=decided_at,
        )
        return DecisionRecord(
            decision_id=self._identities.next("decision"),
            agent_id=self._agent_id,
            belief=snapshot.belief,
            goal=goal,
            intended_action=intention,
            alternatives=alternatives,
            selected_assessment=selected,
            policy_version=self._policy_version,
            decided_at=decided_at,
        )


def _latest_time(label: str, points: Sequence[TimePoint]) -> TimePoint:
    if not points:
        raise ValueError(f"{label} requires at least one time point")
    clocks = {point.clock_id for point in points}
    if len(clocks) != 1:
        raise DecisionError(f"{label} inputs use different clocks: {sorted(clocks)}")
    latest = max(points, key=lambda point: point.tick)
    return TimePoint(tick=latest.tick, clock_id=latest.clock_id)


__all__ = (
    "CandidateAssessor",
    "CounterIdentitySource",
    "DecisionError",
    "MaxValuePolicy",
    "NoAdmissibleActionError",
)
