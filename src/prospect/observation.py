"""Learning from action-free observation (R7, ADR-0010). Task: P13-001.

`LatentActionModel` learns a predictive world model from a stream of observations with
NO actions and NO rewards — "watching". An **inverse model** infers a low-dimensional
**latent action** between consecutive observations; a **forward model** predicts the next
observation from it. They train jointly to reconstruct the next observation, so the model
learns the dynamics *and* — via the latent-action bottleneck — the action structure,
without ever seeing an action.

The load-bearing detail is identifiability (ADR-0010): a naive bottleneck captures a
state-*dependent* feature of the next observation (e.g. the next velocity), not the action.
A **decorrelation penalty** pushes the latent action to be uncorrelated with the current
observation, so it captures the state-*independent* controllable factor — the action.

Backend: numpy only (the `learn` extra), reusing the world model's `_MLP` (tanh MLP + Adam).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .world_model import _MLP


class LatentActionModel:
    """Inverse + forward models over consecutive observations, with a decorrelation
    penalty for action identifiability (ADR-0010). Operates on raw observation vectors
    (state or, for video, frozen-encoder embeddings — ADR-0009); no reward, no action.

    Contracts: interfaces.ObservationLearner.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_action_dim: int = 1,
        hidden: int = 64,
        decorrelation: float = 15.0,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.latent_action_dim = latent_action_dim
        self.decorrelation = decorrelation
        self._inverse = _MLP([2 * obs_dim, hidden, latent_action_dim],
                             np.random.default_rng(seed * 3 + 1), lr)   # (o_t, o_{t+1}) -> latent action
        self._forward = _MLP([obs_dim + latent_action_dim, hidden, obs_dim],
                            np.random.default_rng(seed * 3 + 2), lr)     # (o_t, latent action) -> o_{t+1}

    def infer_action(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        """The inverse model: the latent action that explains obs -> next_obs (one row per
        pair). Accepts a single pair (1-D) or a batch (2-D)."""
        o = np.atleast_2d(np.asarray(obs, dtype=float))
        n = np.atleast_2d(np.asarray(next_obs, dtype=float))
        z, _ = self._inverse.forward(np.concatenate([o, n], axis=1))
        return z if np.ndim(obs) > 1 else z[0]

    def predict(self, obs: np.ndarray, latent_action: np.ndarray) -> np.ndarray:
        """The forward model: the next observation implied by a latent action."""
        o = np.atleast_2d(np.asarray(obs, dtype=float))
        z = np.atleast_2d(np.asarray(latent_action, dtype=float))
        pred, _ = self._forward.forward(np.concatenate([o, z], axis=1))
        return pred if np.ndim(obs) > 1 else pred[0]

    def ground(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> dict[str, float]:
        """Ground the (action-free-pretrained) latent action to a real, directly-executable
        action with a little LABELLED experience — one supervised fine-tuning step on the
        inverse model toward the true action (ADR-0010: watching is the prior, a little acting
        grounds it). Recovery is then `infer_action` directly, with no separate, extrapolating
        calibration — the fix for imitation reliability (the calibration's grounding→demo bias
        was what made the latent route unreliable). Requires latent_action_dim == action_dim;
        the harness loops this after `observe` pretraining."""
        o = np.atleast_2d(np.asarray(obs, dtype=float))
        n = np.atleast_2d(np.asarray(next_obs, dtype=float))
        a = np.asarray(action, dtype=float).reshape(len(o), -1)
        z, cache = self._inverse.forward(np.concatenate([o, n], axis=1))
        self._inverse.zero_grad()
        loss = float(np.mean((z - a) ** 2))
        self._inverse.backward(2.0 * (z - a) / len(o), cache)
        self._inverse.step()
        return {"loss_ground": loss}

    def observe(self, obs: np.ndarray, next_obs: np.ndarray) -> dict[str, float]:
        """One joint training step on a batch of (obs, next_obs) pairs — the action-free
        learning verb. Reconstructs next_obs through the latent-action bottleneck, with the
        decorrelation penalty keeping the latent action state-independent (ADR-0010)."""
        o = np.asarray(obs, dtype=float)
        n = np.asarray(next_obs, dtype=float)
        batch = len(o)
        z, inv_cache = self._inverse.forward(np.concatenate([o, n], axis=1))
        pred, fwd_cache = self._forward.forward(np.concatenate([o, z], axis=1))

        self._inverse.zero_grad()
        self._forward.zero_grad()
        recon = float(np.mean((pred - n) ** 2))
        d_in = self._forward.backward(2.0 * (pred - n) / batch, fwd_cache)  # dL/d(o_t, z)
        dz = d_in[:, self.obs_dim:]  # reconstruction gradient into the latent action

        # Decorrelation (ADR-0010): penalize batch covariance between the latent action and
        # the current observation, so the latent action captures the state-INDEPENDENT
        # controllable factor (the action), not a feature of the next observation.
        z_centered = z - z.mean(axis=0)
        o_centered = o - o.mean(axis=0)
        cov = (z_centered.T @ o_centered) / batch  # (latent_action_dim, obs_dim)
        decorr = float(np.sum(cov ** 2))
        dz = dz + self.decorrelation * (o_centered @ cov.T) / batch

        self._inverse.backward(dz, inv_cache)
        self._forward.step()
        self._inverse.step()
        return {"loss_recon": recon, "decorrelation": decorr}

    def observe_batch(self, batch: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
        """Convenience: `observe` over a sequence of (obs, next_obs) pairs stacked into a
        batch — the shape the harness collects a watched stream into."""
        obs = np.stack([np.asarray(o, dtype=float) for o, _ in batch])
        nxt = np.stack([np.asarray(n, dtype=float) for _, n in batch])
        return self.observe(obs, nxt)
