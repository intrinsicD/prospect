"""The composition root (P2-002): the one place the core components meet.

`Agent` wires seams, not implementations: an `encode` callable (the Codec seam —
the P6 universal-codec swap passes through this single point) and a `Planner`
(which holds the world model it plans over). Keeping the wiring here means gate
evals and future phases extend one loop instead of re-inventing it.

Plugged in so far:
- P3-001: pass (world_model, monitor) and the loop feeds the `CompetenceMonitor`
  a latent-space transition carrying the act-time prediction (VoE).
- P3-003: pass `memory` and every `observe()`d transition (raw modality, P0-011)
  is buffered in the `EpisodicMemory`.
Where the next phases plug in (the seams already exist):
- P8: retrieval-as-action joins `act()` via the `MemoryRouter` (ADR-0004).
"""
from __future__ import annotations

from collections.abc import Callable

from .interfaces import CompetenceMonitor, EpisodicMemory, Planner, WorldModel
from .types import Action, LatentState, Observation, Option, Prediction, Transition


class Agent:
    """Act–observe loop over the assembled components. Not a `Protocol` — this is
    the concrete composition root, not a seam to swap."""

    def __init__(
        self,
        encode: Callable[[Observation], LatentState],
        planner: Planner,
        world_model: WorldModel | None = None,
        monitor: CompetenceMonitor | None = None,
        memory: EpisodicMemory | None = None,
    ) -> None:
        self._encode = encode
        self._planner = planner
        self._model = world_model
        self._monitor = monitor
        self._memory = memory
        self._last: tuple[LatentState, Prediction] | None = None  # act-time expectation

    def act(self, obs: Observation) -> Action:
        """Encode the observation into the shared latent, then plan (ADR-0001).
        With a (world_model, monitor) pair attached, also record what the model
        expects the chosen action to do — `observe()` turns that into VoE."""
        latent = self._encode(obs)
        action = self._planner.plan(latent)
        if self._model is not None and self._monitor is not None:
            self._last = (latent, self._model.predict(latent, action))
        return action

    def observe(
        self,
        obs: Observation,
        action: Action,
        next_obs: Observation,
        reward: float,
        prediction: Prediction | None = None,
        option: Option | None = None,
    ) -> Transition:
        """Record one step of experience. The returned transition carries RAW
        modality data (P0-011): replay stays re-encodable under a future codec;
        models encode internally when they learn. A monitored agent additionally
        feeds the monitor a latent-space transition with the act-time prediction
        (the monitor lives in the space prediction error lives in, ADR-0001)."""
        expected = prediction
        if self._monitor is not None and self._last is not None:
            latent, act_prediction = self._last
            expected = prediction if prediction is not None else act_prediction
            self._monitor.update(
                Transition(state=latent, action=action, next_state=self._encode(next_obs),
                           reward=reward, prediction=expected, option=option)
            )
            self._last = None
        stored = Transition(
            state=LatentState(z=obs.data),
            action=action,
            next_state=LatentState(z=next_obs.data),
            reward=reward,
            prediction=expected,
            option=option,
        )
        if self._memory is not None:
            self._memory.add(stored)
        return stored

    def reset(self) -> None:
        """Start of episode: clear planner state (e.g. a receding-horizon warm
        start) and any pending act-time expectation."""
        self._last = None
        planner_reset = getattr(self._planner, "reset", None)
        if planner_reset is not None:
            planner_reset()
