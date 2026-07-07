"""Imitation from observation (R5, R7, ADR-0012). Task: P14-001.

`ObservationImitator` reproduces a behaviour the agent has only **watched** — a
demonstration given as an observation trajectory, with NO expert actions. It recovers the
demonstrated actions from observation with an **inverse-dynamics** model grounded on a
little of the agent's own labelled experience, then clones a reactive policy that
reproduces the behaviour. This is the observe→repeat step: watching supplies a
goal-reaching behaviour that exploration at the same interaction budget cannot find
(measured on cartpole swingup, the non-gated hard-benchmark tier — ADR-0011/0012).

Why inverse dynamics rather than P13's latent-action inference (ADR-0010): when a small
grounding budget exists, a supervised inverse-dynamics model recovers the *real* action
directly and robustly. The latent-action route is for the fully action-free limit (no
grounding labels at all); it is the arc-faithful but higher-variance route, kept in
`observation.LatentActionModel` and measured alongside this one in the harness.

Backend: numpy only (the `learn` extra), reusing the world model's `_MLP`.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import Transition
from .world_model import _MLP


class ObservationImitator:
    """Inverse-dynamics action recovery + behavioural cloning, to reproduce a demonstrated
    behaviour from observation alone. `ground()` and `clone()` are single-step training
    verbs the harness loops (P0-003). Contracts: interfaces.ImitationLearner.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # (o_t, o_{t+1}) -> action: recover the action that produced a transition.
        self._inverse = _MLP([2 * obs_dim, hidden, action_dim], np.random.default_rng(seed * 3 + 1), lr)
        # o_t -> action: the reactive clone of the demonstrated behaviour.
        self._policy = _MLP([obs_dim, hidden, action_dim], np.random.default_rng(seed * 3 + 2), lr)

    def ground(self, batch: Sequence[Transition]) -> dict[str, float]:
        """One supervised inverse-dynamics step on the agent's OWN labelled transitions —
        the little acting that grounds action recovery."""
        o = np.stack([np.asarray(t.state.z, dtype=float) for t in batch])
        a = np.stack([np.asarray(t.action.data, dtype=float).ravel()[: self.action_dim] for t in batch])
        n = np.stack([np.asarray(t.next_state.z, dtype=float) for t in batch])
        pred, cache = self._inverse.forward(np.concatenate([o, n], axis=1))
        self._inverse.zero_grad()
        loss = float(np.mean((pred - a) ** 2))
        self._inverse.backward(2.0 * (pred - a) / len(o), cache)
        self._inverse.step()
        return {"loss_inverse": loss}

    def recover(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """Recover the action that produced obs -> next_obs (one row per pair). Accepts a
        single pair (1-D) or a batch (2-D)."""
        o = np.atleast_2d(np.asarray(obs, dtype=float))
        n = np.atleast_2d(np.asarray(next_obs, dtype=float))
        a, _ = self._inverse.forward(np.concatenate([o, n], axis=1))
        return a if np.ndim(obs) > 1 else a[0]

    def clone(self, observations: np.ndarray, next_observations: np.ndarray) -> dict[str, float]:
        """One behavioural-cloning step: recover the demo's actions from its observations
        (frozen inverse — no grad through recovery) and fit the reactive policy toward them."""
        o = np.atleast_2d(np.asarray(observations, dtype=float))
        n = np.atleast_2d(np.asarray(next_observations, dtype=float))
        target = np.atleast_2d(self.recover(o, n))
        pred, cache = self._policy.forward(o)
        self._policy.zero_grad()
        loss = float(np.mean((pred - target) ** 2))
        self._policy.backward(2.0 * (pred - target) / len(o), cache)
        self._policy.step()
        return {"loss_clone": loss}

    def act(self, obs: np.ndarray) -> np.ndarray:
        """The reproduced behaviour: the cloned reactive policy's action for an observation."""
        o = np.atleast_2d(np.asarray(obs, dtype=float))
        a, _ = self._policy.forward(o)
        return a[0] if np.ndim(obs) <= 1 else a
