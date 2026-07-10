"""Latent predictive world model (R1, R4). See ADR-0001, ADR-0002, ADR-0006.

`FlatWorldModel` is the Phase-1 implementation (task P1-001): a small MLP encoder
maps the (single toy modality, low-dim) observation into a latent; an **ensemble**
of Gaussian dynamics heads predicts the next-latent distribution (disagreement =
epistemic, predicted variance = aleatoric); collapse guards per ADR-0006 are built
in from the start — EMA target encoder with stop-gradient, VICReg-style variance +
covariance regularization, and inverse-dynamics + reward auxiliary heads.

Space conventions (P1, single modality):
- `predict()` / `imagine()` consume and produce **latents** (`LatentState.z` is the
  encoded latent; use `encode()` / `encode_target()` to get one from a raw
  observation vector). Prediction error lives in latent space (ADR-0001).
- `update()` consumes transitions whose `.state.z` / `.next_state.z` hold the **raw
  observation vectors**: replay stores raw experience so it stays re-encodable
  under a future codec (P0-011); the model encodes internally while training.

Backend: numpy only (the `learn` extra) — the core package metadata stays
dependency-free; heavier backends arrive only when an implementation needs them.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import Action, LatentState, MemberRollout, Prediction, Transition

_LOGVAR_MIN, _LOGVAR_MAX = -6.0, 2.0
_CLIP_NORM = 1.0  # global gradient-norm clip per network
_LOG_TAU = float(np.log(2.0 * np.pi))
_DIST_WEIGHT = 1.0  # distance-aware epistemic: boost factor for the pre-encoder OOD score


class _MLP:
    """Minimal tanh MLP (linear output) with explicit-cache backprop and Adam."""

    def __init__(self, sizes: Sequence[int], rng: np.random.Generator, lr: float) -> None:
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / (m + n)), size=(m, n))
            for m, n in zip(sizes[:-1], sizes[1:], strict=True)
        ]
        self.biases = [np.zeros(n) for n in sizes[1:]]
        self.lr = lr
        self._grad_w = [np.zeros_like(w) for w in self.weights]
        self._grad_b = [np.zeros_like(b) for b in self.biases]
        self._adam_m = [np.zeros_like(p) for p in self.weights + self.biases]
        self._adam_v = [np.zeros_like(p) for p in self.weights + self.biases]
        self._adam_t = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns (output, cache); cache holds post-activation values per layer."""
        cache = [x]
        h = x
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if i < last:
                h = np.tanh(h)
            cache.append(h)
        return h, cache

    def backward(self, dout: np.ndarray, cache: list[np.ndarray]) -> np.ndarray:
        """Accumulates parameter gradients; returns dL/dx for the cached forward."""
        grad = dout
        for i in range(len(self.weights) - 1, -1, -1):
            if i < len(self.weights) - 1:  # undo tanh of this layer's output
                grad = grad * (1.0 - cache[i + 1] ** 2)
            self._grad_w[i] += cache[i].T @ grad
            self._grad_b[i] += grad.sum(axis=0)
            grad = grad @ self.weights[i].T
        return grad

    def zero_grad(self) -> None:
        for g in self._grad_w + self._grad_b:
            g[...] = 0.0

    def step(self) -> None:
        """One Adam step over all parameters, then clears gradients. Gradients are
        clipped to a global norm first: abrupt coordinated moves (not steady
        learning) are what transiently collapse the latent's effective rank."""
        self._adam_t += 1
        params = self.weights + self.biases
        grads = self._grad_w + self._grad_b
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if norm > _CLIP_NORM:
            grads = [g * (_CLIP_NORM / norm) for g in grads]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(params, grads, self._adam_m, self._adam_v, strict=True):
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            m_hat = m / (1 - b1**self._adam_t)
            v_hat = v / (1 - b2**self._adam_t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.zero_grad()


class FlatWorldModel:
    """Contracts: interfaces.WorldModel + interfaces.Learner (see module docstring)."""

    def __init__(
        self,
        obs_dim: int = 3,
        action_dim: int = 1,
        latent_dim: int = 8,
        hidden: int = 64,
        ensemble: int = 5,
        lr: float = 3e-3,
        ema_tau: float = 0.995,
        seed: int = 0,
        w_reward: float = 1.0,
        w_inverse: float = 1.0,
        w_var: float = 25.0,
        w_cov: float = 1.0,
    ) -> None:
        self.obs_dim, self.action_dim, self.latent_dim = obs_dim, action_dim, latent_dim
        self.ema_tau = ema_tau
        self.w_reward, self.w_inverse, self.w_var, self.w_cov = w_reward, w_inverse, w_var, w_cov
        rng = np.random.default_rng(seed)
        self._rng = rng
        self.encoder = _MLP([obs_dim, hidden, latent_dim], rng, lr)
        self._target_w = [w.copy() for w in self.encoder.weights]
        self._target_b = [b.copy() for b in self.encoder.biases]
        # Independent member inits (decorrelated seeds) — ADR-0006 ensemble diversity.
        self.members = [
            _MLP([latent_dim + action_dim, hidden, 2 * latent_dim], np.random.default_rng(seed * 1009 + 7 * k + 1), lr)
            for k in range(ensemble)
        ]
        self.reward_head = _MLP([latent_dim + action_dim, hidden // 2, 1], rng, lr)
        self.inverse_head = _MLP([2 * latent_dim, hidden // 2, action_dim], rng, lr)
        # Running input standardization (EMA, updated in update(), shared by online
        # and target encoders): without it the raw observation's variance structure
        # (e.g. one high-variance dim) passes straight into the latent and caps its
        # effective rank — a degenerate representation the sentinel rightly flags.
        self._obs_mean = np.zeros(obs_dim)
        self._obs_var = np.ones(obs_dim)
        self._obs_stats_ready = False

    # ------------------------------------------------------------------ encode
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._obs_mean) / np.sqrt(self._obs_var + 1e-6)

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        if not self._obs_stats_ready:
            self._obs_mean = obs.mean(axis=0)
            self._obs_var = obs.var(axis=0) + 1e-6
            self._obs_stats_ready = True
        else:
            self._obs_mean = 0.99 * self._obs_mean + 0.01 * obs.mean(axis=0)
            self._obs_var = 0.99 * self._obs_var + 0.01 * obs.var(axis=0)

    def encode(self, obs: object) -> LatentState:
        """Online encoder: raw observation vector -> latent (the model's input space).

        The latent carries an `ood` score — the standardized input's excess energy over
        the training distribution's unit variance (0 in-distribution, rising out of it),
        measured on the raw input *before* the encoder. Ensemble disagreement alone
        under-detects OOD once the tanh encoder saturates (P9-005): this pre-encoder
        distance is what makes epistemic OOD-reliable, and it rides on the latent so
        `predict` can use it without re-plumbing the raw observation everywhere."""
        x = self._standardize(np.asarray(obs, dtype=float).reshape(1, -1))
        ood = max(0.0, float(np.mean(x**2)) - 1.0) if self._obs_stats_ready else 0.0
        h, _ = self.encoder.forward(x)
        return LatentState(z=h[0], ood=ood)

    def encode_target(self, obs: object) -> LatentState:
        """EMA target encoder (stop-gradient) — the space prediction error lives in."""
        x = self._standardize(np.asarray(obs, dtype=float).reshape(1, -1))
        return LatentState(z=self._target_forward(x)[0])

    def _target_forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        last = len(self._target_w) - 1
        for i, (w, b) in enumerate(zip(self._target_w, self._target_b, strict=True)):
            h = h @ w + b
            if i < last:
                h = np.tanh(h)
        return h

    # ----------------------------------------------------------------- predict
    def _member_forward(self, h: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All members on (h, a): returns (mus, variances), each (k, n, latent_dim)."""
        x = np.concatenate([h, a], axis=1)
        mus, variances = [], []
        for member in self.members:
            out, _ = member.forward(x)
            delta, logvar = out[:, : self.latent_dim], out[:, self.latent_dim :]
            logvar = np.clip(logvar, _LOGVAR_MIN, _LOGVAR_MAX)
            mus.append(h + delta)  # residual parameterization
            variances.append(np.exp(logvar))
        return np.stack(mus), np.stack(variances)

    def predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized `predict()` for planning rollouts (n latent/action pairs).

        Returns (mean (n,d), total var (n,d), epistemic (n,), aleatoric (n,),
        reward (n,)) — the exact per-sample quantities `predict()` wraps. Not part
        of the `WorldModel` protocol; planners duck-type it and fall back to
        per-sample `predict()` for protocol-only models.
        """
        mus, variances = self._member_forward(latents, actions)
        mean = mus.mean(axis=0)
        aleatoric = variances.mean(axis=0)  # within-member spread
        epistemic = mus.var(axis=0)  # between-member disagreement
        reward_out, _ = self.reward_head.forward(np.concatenate([latents, actions], axis=1))
        return (
            mean,
            aleatoric + epistemic,  # total predictive variance (moment-matched mixture)
            epistemic.mean(axis=1),
            aleatoric.mean(axis=1),
            reward_out[:, 0],
        )

    def predict_member_batch(
        self,
        member_latents: np.ndarray,
        actions: np.ndarray,
        initial_ood: float | None = None,
    ) -> MemberRollout:
        """Advance one trajectory per ensemble member (TS∞, task U-001).

        ``member_latents`` is either ``(candidates, latent_dim)`` on the first
        rollout step (the common start is expanded across members here), or
        ``(members, candidates, latent_dim)`` on later steps.  Each dynamics
        member advances only its own state, so disagreement can compound across
        the horizon instead of being recomputed around a mean state no member
        actually occupies.

        Returns a :class:`MemberRollout` whose states and variances have shape
        ``(members, candidates, latent_dim)``, whose rewards have shape
        ``(members, candidates)``, and whose effective epistemic signal has shape
        ``(candidates,)``.
        The variances are tracked alongside the deterministic member states;
        aleatoric noise is deliberately not sampled into the trajectories.
        """
        states = np.asarray(member_latents, dtype=float)
        act = np.asarray(actions, dtype=float)
        member_count = len(self.members)
        if states.ndim == 2:
            states = np.repeat(states[None, :, :], member_count, axis=0)
        if states.ndim != 3 or states.shape[0] != member_count:
            raise ValueError(
                "member_latents must have shape (candidates, latent_dim) or "
                f"({member_count}, candidates, latent_dim)"
            )
        if states.shape[2] != self.latent_dim:
            raise ValueError(
                f"member_latents has latent dim {states.shape[2]}, expected {self.latent_dim}"
            )
        if act.ndim != 2 or act.shape != (states.shape[1], self.action_dim):
            raise ValueError(
                "actions must have shape "
                f"({states.shape[1]}, {self.action_dim}), got {act.shape}"
            )

        next_states, variances = [], []
        for member, state_rows in zip(self.members, states, strict=True):
            out, _ = member.forward(np.concatenate([state_rows, act], axis=1))
            delta = out[:, : self.latent_dim]
            logvar = np.clip(out[:, self.latent_dim :], _LOGVAR_MIN, _LOGVAR_MAX)
            next_states.append(state_rows + delta)
            variances.append(np.exp(logvar))

        # Reward is shared rather than ensembled, but it must be evaluated at
        # every member's own current state after the trajectories diverge.
        repeated_actions = np.broadcast_to(act, (member_count, *act.shape))
        reward_input = np.concatenate(
            [states, repeated_actions], axis=2
        ).reshape(member_count * states.shape[1], self.latent_dim + self.action_dim)
        reward_out, _ = self.reward_head.forward(reward_input)
        rewards = reward_out[:, 0].reshape(member_count, states.shape[1])
        next_rows = np.stack(next_states)
        epistemic = next_rows.var(axis=0).mean(axis=1)
        if initial_ood is not None:
            epistemic *= 1.0 + _DIST_WEIGHT * initial_ood
        return MemberRollout(
            states=next_rows,
            variances=np.stack(variances),
            rewards=rewards,
            epistemic=epistemic,
        )

    def predict(self, state: LatentState, action: Action) -> Prediction:
        h = np.asarray(state.z, dtype=float).reshape(1, -1)
        a = np.asarray(action.data, dtype=float).reshape(1, -1)
        mean, var, epistemic, aleatoric, reward = self.predict_batch(h, a)
        # Distance-aware epistemic (P9-005): when the latent came from a real
        # observation (encode set `state.ood`), scale ensemble disagreement by the
        # pre-encoder OOD score so epistemic rises out-of-distribution even where the
        # saturated ensemble agrees. Synthesized rollout latents carry no `ood` (None)
        # and are unchanged. Only the scalar is scaled — `var` (log_prob) stays the
        # ensemble's calibrated total.
        epistemic_scalar = float(epistemic[0])
        if state.ood is not None:
            epistemic_scalar *= 1.0 + _DIST_WEIGHT * state.ood
        return Prediction(
            mean=mean[0],
            var=var[0],
            epistemic=epistemic_scalar,
            aleatoric=float(aleatoric[0]),
            reward=float(reward[0]),
        )

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        """Open-loop TS∞ rollout, propagating one state per ensemble member.

        Epistemic uncertainty is the spread of the member trajectories at each
        horizon step.  Per-member aleatoric variance is accumulated alongside the
        trajectories without injecting noise into their states (U-001).
        """
        predictions: list[Prediction] = []
        member_states = np.asarray(state.z, dtype=float).reshape(1, -1)
        accumulated_variance: np.ndarray | None = None
        for step, action in enumerate(actions):
            action_rows = np.asarray(action.data, dtype=float).reshape(1, -1)
            rollout = self.predict_member_batch(
                member_states, action_rows, initial_ood=state.ood if step == 0 else None
            )
            member_states = np.asarray(rollout.states, dtype=float)
            step_variance = np.asarray(rollout.variances, dtype=float)
            member_rewards = np.asarray(rollout.rewards, dtype=float)
            if accumulated_variance is None:
                accumulated_variance = np.zeros_like(step_variance)
            accumulated_variance += step_variance
            mean = member_states.mean(axis=0)[0]
            epistemic = member_states.var(axis=0)[0]
            aleatoric = accumulated_variance.mean(axis=0)[0]
            predictions.append(
                Prediction(
                    mean=mean,
                    var=aleatoric + epistemic,
                    epistemic=float(np.asarray(rollout.epistemic, dtype=float)[0]),
                    aleatoric=float(aleatoric.mean()),
                    reward=float(member_rewards.mean(axis=0)[0]),
                )
            )
        return predictions

    # ------------------------------------------------------------------- learn
    def update(self, batch: Sequence[Transition]) -> dict[str, float]:
        """One training step; returns the metrics dict the harness logs (P0-005)."""
        obs = np.stack([np.asarray(t.state.z, dtype=float) for t in batch])
        act = np.stack([np.asarray(t.action.data, dtype=float) for t in batch])
        nxt = np.stack([np.asarray(t.next_state.z, dtype=float) for t in batch])
        rew = np.array([t.reward for t in batch], dtype=float).reshape(-1, 1)
        n, d = obs.shape[0], self.latent_dim

        self._update_obs_stats(obs)
        obs = self._standardize(obs)
        nxt = self._standardize(nxt)
        latents, cache_h = self.encoder.forward(obs)
        next_target = self._target_forward(nxt)  # stop-grad target (ADR-0006)
        next_online, cache_hn = self.encoder.forward(nxt)

        d_latents = np.zeros_like(latents)
        d_next_online = np.zeros_like(next_online)
        for net in (self.encoder, self.reward_head, self.inverse_head, *self.members):
            net.zero_grad()

        # Ensemble NLL — each member on its own bootstrap resample (decorrelation).
        nll_total = 0.0
        for member in self.members:
            idx = self._rng.integers(0, n, size=n)
            h_k, a_k, target_k = latents[idx], act[idx], next_target[idx]
            out, cache_k = member.forward(np.concatenate([h_k, a_k], axis=1))
            delta, logvar = out[:, :d], np.clip(out[:, d:], _LOGVAR_MIN, _LOGVAR_MAX)
            var = np.exp(logvar)
            mu = h_k + delta
            diff = mu - target_k
            nll_total += float(0.5 * np.mean(np.sum(diff**2 / var + logvar + _LOG_TAU, axis=1)))
            d_mu = diff / var / n
            d_logvar = 0.5 * (1.0 - diff**2 / var) / n
            d_x = member.backward(np.concatenate([d_mu, d_logvar], axis=1), cache_k)
            np.add.at(d_latents, idx, d_x[:, :d] + d_mu)  # input path + residual path
        nll_mean = nll_total / len(self.members)

        # Reward head (auxiliary: forces outcome-relevant content into the latent).
        reward_out, cache_r = self.reward_head.forward(np.concatenate([latents, act], axis=1))
        loss_reward = float(np.mean((reward_out - rew) ** 2))
        d_r = self.w_reward * 2.0 * (reward_out - rew) / n
        d_latents += self.reward_head.backward(d_r, cache_r)[:, :d]

        # Inverse-dynamics head (forces controllable content into the latent).
        inv_out, cache_i = self.inverse_head.forward(np.concatenate([latents, next_online], axis=1))
        loss_inverse = float(np.mean((inv_out - act) ** 2))
        d_a = self.w_inverse * 2.0 * (inv_out - act) / (n * self.action_dim)
        d_inv = self.inverse_head.backward(d_a, cache_i)
        d_latents += d_inv[:, :d]
        d_next_online += d_inv[:, d:]

        # VICReg-style anti-collapse on the online latents (ADR-0006).
        centered = latents - latents.mean(axis=0, keepdims=True)
        std = np.sqrt(centered.var(axis=0) + 1e-4)
        loss_var = float(np.mean(np.maximum(0.0, 1.0 - std)))
        active = (std < 1.0).astype(float)
        d_latents += self.w_var * (-(1.0 / d) * active * centered / (n * std))
        cov = centered.T @ centered / (n - 1)
        off_diag = cov * (1.0 - np.eye(d))
        loss_cov = float(np.sum(off_diag**2) / d)
        d_latents += self.w_cov * (2.0 / (n - 1)) * centered @ ((2.0 / d) * off_diag)

        # Backprop into the encoder (both forwards), then step everything.
        self.encoder.backward(d_latents, cache_h)
        self.encoder.backward(d_next_online, cache_hn)
        for net in (self.encoder, self.reward_head, self.inverse_head, *self.members):
            net.step()

        # EMA target-encoder update (stop-gradient branch).
        for target, online in zip(self._target_w + self._target_b,
                                  self.encoder.weights + self.encoder.biases, strict=True):
            target[...] = self.ema_tau * target + (1.0 - self.ema_tau) * online

        # Integrity metrics for the run log / sentinels (fresh disagreement, full batch).
        mus, _ = self._member_forward(latents, act)
        disagreement = float(mus.var(axis=0).mean())
        eigenvalues = np.linalg.eigvalsh(np.cov(latents.T) + 1e-8 * np.eye(d))
        effective_rank = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        return {
            "loss_nll": nll_mean,
            "loss_reward": loss_reward,
            "loss_inverse": loss_inverse,
            "loss_var_hinge": loss_var,
            "loss_cov": loss_cov,
            "latent_std_min": float(std.min()),
            "latent_effective_rank": effective_rank,
            "ensemble_disagreement": disagreement,
        }
