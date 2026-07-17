"""Append-only lifecycle evidence for partially completed interaction steps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock

from prospect.domain import TimePoint

_SECRET_ASSIGNMENT = re.compile(r"(?i)\b(token|password|secret|api[_-]?key)\b\s*[:=]\s*\S+")
_MAX_FAILURE_DETAIL = 240


class LifecycleStage(StrEnum):
    """Durable boundaries reached by one runtime interaction."""

    EXPERIENCE_STORED = "experience_stored"
    TRANSITION_STORED = "transition_stored"
    STATE_APPLIED = "state_applied"
    REPLAY_INDEXED = "replay_indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    """One immutable success or failure fact for an interaction step."""

    sequence: int
    agent_id: str
    run_id: str
    task_id: str
    episode_id: str
    step_index: int
    decision_id: str
    stage: LifecycleStage
    recorded_at: TimePoint
    experience_id: str | None = None
    transition_id: str | None = None
    attempted_stage: LifecycleStage | None = None
    failure_type: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("lifecycle sequence must be nonnegative")
        for name, value in (
            ("agent_id", self.agent_id),
            ("run_id", self.run_id),
            ("task_id", self.task_id),
            ("episode_id", self.episode_id),
            ("decision_id", self.decision_id),
        ):
            if not value or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if self.step_index < 0:
            raise ValueError("lifecycle step_index must be nonnegative")
        for name, optional_value in (
            ("experience_id", self.experience_id),
            ("transition_id", self.transition_id),
        ):
            if optional_value is not None and (not optional_value or not optional_value.strip()):
                raise ValueError(f"{name} must be nonempty when supplied")
        if self.stage is LifecycleStage.FAILED:
            if self.attempted_stage in {None, LifecycleStage.FAILED}:
                raise ValueError("failed lifecycle record requires an attempted stage")
            if not self.failure_type or not self.failure_detail:
                raise ValueError("failed lifecycle record requires bounded failure details")
            if len(self.failure_detail) > _MAX_FAILURE_DETAIL:
                raise ValueError("lifecycle failure detail exceeds its size bound")
            if self.attempted_stage is not LifecycleStage.EXPERIENCE_STORED and self.experience_id is None:
                raise ValueError("downstream lifecycle failure requires an experience_id")
            if (
                self.attempted_stage
                in {
                    LifecycleStage.STATE_APPLIED,
                    LifecycleStage.REPLAY_INDEXED,
                }
                and self.transition_id is None
            ):
                raise ValueError("post-transition lifecycle failure requires a transition_id")
        elif self.attempted_stage is not None or self.failure_type or self.failure_detail:
            raise ValueError("successful lifecycle record cannot contain failure details")
        elif self.experience_id is None:
            raise ValueError("successful lifecycle record requires an experience_id")
        elif (
            self.stage
            in {
                LifecycleStage.TRANSITION_STORED,
                LifecycleStage.STATE_APPLIED,
                LifecycleStage.REPLAY_INDEXED,
            }
            and self.transition_id is None
        ):
            raise ValueError("post-experience lifecycle stage requires a transition_id")

    @property
    def step_key(self) -> tuple[str, str, str, int]:
        return (
            self.agent_id,
            self.run_id,
            self.episode_id,
            self.step_index,
        )


class LifecycleJournal:
    """Thread-safe append-only evidence of interaction progress.

    The journal exposes partial completion; it does not automatically resume a
    failed step. External orchestration can inspect immutable history and choose a
    recovery policy without guessing which canonical writes already occurred.
    """

    _NEXT_STAGE = {
        None: LifecycleStage.EXPERIENCE_STORED,
        LifecycleStage.EXPERIENCE_STORED: LifecycleStage.TRANSITION_STORED,
        LifecycleStage.TRANSITION_STORED: LifecycleStage.STATE_APPLIED,
        LifecycleStage.STATE_APPLIED: LifecycleStage.REPLAY_INDEXED,
    }

    def __init__(self) -> None:
        self._records: list[LifecycleRecord] = []
        self._by_step: dict[tuple[str, str, str, int], list[LifecycleRecord]] = {}
        self._lock = RLock()

    def append_stage(
        self,
        *,
        agent_id: str,
        run_id: str,
        task_id: str,
        episode_id: str,
        step_index: int,
        decision_id: str,
        stage: LifecycleStage,
        recorded_at: TimePoint,
        experience_id: str | None = None,
        transition_id: str | None = None,
    ) -> LifecycleRecord:
        """Append the next successful stage for one step."""

        if stage is LifecycleStage.FAILED:
            raise ValueError("use append_failure for failed lifecycle records")
        key = (agent_id, run_id, episode_id, step_index)
        with self._lock:
            history = self._by_step.get(key, [])
            previous = None if not history else history[-1].stage
            expected = self._NEXT_STAGE.get(previous)
            if expected is not stage:
                raise ValueError(
                    f"lifecycle stage {stage.value!r} does not follow {None if previous is None else previous.value!r}"
                )
            _require_causal_append(history, recorded_at)
            record = LifecycleRecord(
                sequence=len(self._records),
                agent_id=agent_id,
                run_id=run_id,
                task_id=task_id,
                episode_id=episode_id,
                step_index=step_index,
                decision_id=decision_id,
                stage=stage,
                recorded_at=recorded_at,
                experience_id=experience_id,
                transition_id=transition_id,
            )
            self._append(key, record)
            return record

    def append_failure(
        self,
        *,
        agent_id: str,
        run_id: str,
        task_id: str,
        episode_id: str,
        step_index: int,
        decision_id: str,
        attempted_stage: LifecycleStage,
        error: BaseException,
        recorded_at: TimePoint,
        experience_id: str | None = None,
        transition_id: str | None = None,
    ) -> LifecycleRecord:
        """Append sanitized failure evidence without storing a traceback."""

        if attempted_stage is LifecycleStage.FAILED:
            raise ValueError("failure cannot attempt the FAILED stage")
        key = (agent_id, run_id, episode_id, step_index)
        failure_type, failure_detail = _failure_details(error)
        with self._lock:
            history = self._by_step.get(key, [])
            if history and history[-1].stage is LifecycleStage.FAILED:
                raise ValueError("failed lifecycle step is already closed")
            previous = None if not history else history[-1].stage
            expected = self._NEXT_STAGE.get(previous)
            if expected is not attempted_stage:
                raise ValueError(
                    f"failure for {attempted_stage.value!r} does not follow "
                    f"{None if previous is None else previous.value!r}"
                )
            _require_causal_append(history, recorded_at)
            record = LifecycleRecord(
                sequence=len(self._records),
                agent_id=agent_id,
                run_id=run_id,
                task_id=task_id,
                episode_id=episode_id,
                step_index=step_index,
                decision_id=decision_id,
                stage=LifecycleStage.FAILED,
                recorded_at=recorded_at,
                experience_id=experience_id,
                transition_id=transition_id,
                attempted_stage=attempted_stage,
                failure_type=failure_type,
                failure_detail=failure_detail,
            )
            self._append(key, record)
            return record

    def history(
        self,
        *,
        agent_id: str,
        run_id: str,
        episode_id: str,
        step_index: int,
    ) -> tuple[LifecycleRecord, ...]:
        """Return an immutable insertion-ordered history for one step."""

        key = (agent_id, run_id, episode_id, step_index)
        with self._lock:
            return tuple(self._by_step.get(key, ()))

    def _append(
        self,
        key: tuple[str, str, str, int],
        record: LifecycleRecord,
    ) -> None:
        self._records.append(record)
        self._by_step.setdefault(key, []).append(record)


def _failure_details(error: BaseException) -> tuple[str, str]:
    failure_type = type(error).__name__
    message = " ".join(str(error).split()) or "no exception message"
    message = _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", message)
    if len(message) > _MAX_FAILURE_DETAIL:
        message = f"{message[: _MAX_FAILURE_DETAIL - 1]}…"
    return failure_type, message


def _require_causal_append(
    history: list[LifecycleRecord],
    recorded_at: TimePoint,
) -> None:
    if not history:
        return
    previous = history[-1].recorded_at
    if recorded_at.clock_id != previous.clock_id:
        raise ValueError("lifecycle records for one step must use one clock")
    if recorded_at.tick < previous.tick:
        raise ValueError("lifecycle record time cannot precede the previous stage")


__all__ = ("LifecycleJournal", "LifecycleRecord", "LifecycleStage")
