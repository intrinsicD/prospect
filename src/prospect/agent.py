"""The composition root (P2-002): the one place the core components meet.

`Agent` wires seams, not implementations: an `encode` callable (the Codec seam —
the P6 universal-codec swap passes through this single point) and a `Planner`
(which holds the world model it plans over). Keeping the wiring here means gate
evals and future phases extend one loop instead of re-inventing it.

Where the next phases plug in (the seams already exist):
- P3-001: the `CompetenceMonitor` reads each `observe()`d transition (VoE).
- P3-003: the `EpisodicMemory` buffers each `observe()`d transition (replay).
- P8: retrieval-as-action joins `act()` via the `MemoryRouter` (ADR-0004).
"""
from __future__ import annotations

from collections.abc import Callable

from .interfaces import Planner
from .types import Action, LatentState, Observation, Option, Prediction, Transition


class Agent:
    """Act–observe loop over the assembled components. Not a `Protocol` — this is
    the concrete composition root, not a seam to swap."""

    def __init__(self, encode: Callable[[Observation], LatentState], planner: Planner) -> None:
        self._encode = encode
        self._planner = planner

    def act(self, obs: Observation) -> Action:
        """Encode the observation into the shared latent, then plan (ADR-0001)."""
        return self._planner.plan(self._encode(obs))

    def observe(
        self,
        obs: Observation,
        action: Action,
        next_obs: Observation,
        reward: float,
        prediction: Prediction | None = None,
        option: Option | None = None,
    ) -> Transition:
        """Record one step of experience. Transitions carry RAW modality data
        (P0-011): replay stays re-encodable under a future codec; models encode
        internally when they learn."""
        return Transition(
            state=LatentState(z=obs.data),
            action=action,
            next_state=LatentState(z=next_obs.data),
            reward=reward,
            prediction=prediction,
            option=option,
        )

    def reset(self) -> None:
        """Start of episode: clear planner state (e.g. a receding-horizon warm start)."""
        planner_reset = getattr(self._planner, "reset", None)
        if planner_reset is not None:
            planner_reset()
