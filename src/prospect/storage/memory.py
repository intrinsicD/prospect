"""Canonical in-memory storage for real experience."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock

from prospect.domain import ExperienceEvent, TimePoint


class StorageError(RuntimeError):
    """Base error for canonical storage failures."""


class DuplicateRecordError(StorageError):
    """An append attempted to reuse an immutable record identifier."""


class CausalOrderError(StorageError):
    """An append or query mixed clocks or violated per-agent causal order."""


class RecordNotFoundError(StorageError, KeyError):
    """A requested canonical record does not exist."""


def _require_identifier(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a nonempty identifier")


def _ensure_same_clock(label: str, left: TimePoint, right: TimePoint) -> None:
    if left.clock_id != right.clock_id:
        raise CausalOrderError(f"{label} uses different clocks: {left.clock_id!r} and {right.clock_id!r}")


class InMemoryExperienceStore:
    """Thread-safe, append-only canonical store implementing ``ExperienceStore``.

    An agent's events must be appended in nondecreasing ``closed_at`` order on one
    logical clock.  Equal ticks preserve insertion order.  Steps must increase
    within each agent/run/episode, and no event may follow a terminated or truncated
    episode.  ``history`` uses an inclusive causal cutoff and never returns another
    agent's records.

    This store is intentionally canonical rather than capacity bounded: replay
    buffers are lossy sampling indexes and must not replace the evidence ledger.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, ExperienceEvent] = {}
        self._by_step: dict[tuple[str, str, str, int], ExperienceEvent] = {}
        self._by_agent: dict[str, list[ExperienceEvent]] = {}
        self._last_close: dict[str, TimePoint] = {}
        self._last_step: dict[tuple[str, str, str], int] = {}
        self._closed_episodes: set[tuple[str, str, str]] = set()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def append(self, event: ExperienceEvent) -> None:
        """Append one real, closed experience without replacing prior state."""

        with self._lock:
            if event.experience_id in self._by_id:
                raise DuplicateRecordError(f"experience id {event.experience_id!r} is already stored")
            step_key = (
                event.agent_id,
                event.run_id,
                event.episode_id,
                event.step_index,
            )
            if step_key in self._by_step:
                raise DuplicateRecordError(
                    "experience step identity "
                    f"{event.agent_id!r}/{event.run_id!r}/{event.episode_id!r}/"
                    f"{event.step_index} is already stored"
                )
            previous = self._last_close.get(event.agent_id)
            if previous is not None:
                _ensure_same_clock("experience append", event.closed_at, previous)
                if event.closed_at.tick < previous.tick:
                    raise CausalOrderError(
                        f"experience {event.experience_id!r} closes at "
                        f"{event.closed_at.tick}, before the agent's last close at {previous.tick}"
                    )
            episode_key = (
                event.agent_id,
                event.run_id,
                event.episode_id,
            )
            if episode_key in self._closed_episodes:
                raise CausalOrderError(f"experience {event.experience_id!r} follows a terminated or truncated episode")
            previous_step = self._last_step.get(episode_key)
            if previous_step is not None and event.step_index <= previous_step:
                raise CausalOrderError(
                    f"experience {event.experience_id!r} has step {event.step_index}, "
                    f"not after the episode's last step {previous_step}"
                )

            self._by_id[event.experience_id] = event
            self._by_step[step_key] = event
            self._by_agent.setdefault(event.agent_id, []).append(event)
            self._last_close[event.agent_id] = event.closed_at
            self._last_step[episode_key] = event.step_index
            if event.terminated or event.truncated:
                self._closed_episodes.add(episode_key)

    def get(self, experience_id: str) -> ExperienceEvent:
        """Return the canonical record for ``experience_id``."""

        _require_identifier("experience_id", experience_id)
        with self._lock:
            try:
                return self._by_id[experience_id]
            except KeyError as error:
                raise RecordNotFoundError(f"experience id {experience_id!r} is not stored") from error

    def get_step(
        self,
        agent_id: str,
        run_id: str,
        episode_id: str,
        step_index: int,
    ) -> ExperienceEvent:
        """Return the canonical event for one exact runtime step identity."""

        event = self.find_step(agent_id, run_id, episode_id, step_index)
        if event is None:
            raise RecordNotFoundError(
                f"experience step {agent_id!r}/{run_id!r}/{episode_id!r}/{step_index} is not stored"
            )
        return event

    def find_step(
        self,
        agent_id: str,
        run_id: str,
        episode_id: str,
        step_index: int,
    ) -> ExperienceEvent | None:
        """Find one exact runtime step, returning ``None`` when it is absent."""

        _require_identifier("agent_id", agent_id)
        _require_identifier("run_id", run_id)
        _require_identifier("episode_id", episode_id)
        if step_index < 0:
            raise ValueError("step_index must be nonnegative")
        with self._lock:
            return self._by_step.get((agent_id, run_id, episode_id, step_index))

    def history(self, agent_id: str, as_of: TimePoint) -> Sequence[ExperienceEvent]:
        """Return an immutable, insertion-stable history through ``as_of`` inclusive."""

        _require_identifier("agent_id", agent_id)
        with self._lock:
            events = self._by_agent.get(agent_id)
            if not events:
                return ()
            _ensure_same_clock("experience history cutoff", as_of, events[0].closed_at)
            return tuple(event for event in events if event.closed_at.tick <= as_of.tick)
