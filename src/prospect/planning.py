"""Planning (R1, R2). Flat MPC in imagination, the learned jumpy option-model,
and the hierarchical manager over it. See ADR-0001, ADR-0003, ADR-0006/0007.
Tasks: P2-001 (done), P5-001 (done), P5-002.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .interfaces import WorldModel
from .types import Action, LatentState, Option, Prediction, Subgoal, Transition
from .world_model import _LOGVAR_MAX, _LOGVAR_MIN, _MLP  # core-internal reuse


class FlatPlanner:
    """CEM/MPC in imagination over a `WorldModel` (R1, task P2-001).

    Candidate action sequences are rolled out in latent space and scored by
    discounted imagined reward **minus a per-step epistemic penalty** — ADR-0006's
    model-exploitation control. This is the ADR-0007 *exploit-mode* consumer: the
    penalty sign is fixed here; the exploration bonus (sign flip) belongs to the
    curriculum (P3-002), never to the planner.

    Receding horizon: `plan()` returns the first action of the optimized sequence
    and warm-starts the next call with the shifted elite mean; call `reset()`
    between episodes. `goal` is accepted per the Protocol and ignored until
    hierarchical planning lands (P5) — P2 plans on reward.

    Contract: interfaces.Planner. Uses the model's vectorized `predict_batch` when
    offered, falling back to the protocol's per-sample `predict()`.
    """

    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int = 1,
        action_low: float = -2.0,
        action_high: float = 2.0,
        horizon: int = 20,
        candidates: int = 64,
        elites: int = 8,
        iterations: int = 3,
        discount: float = 0.99,
        uncertainty_penalty: float = 1.0,
        seed: int = 0,
    ) -> None:
        self._model = world_model
        self.action_dim, self.action_low, self.action_high = action_dim, action_low, action_high
        self.horizon, self.candidates, self.elites = horizon, candidates, elites
        self.iterations, self.discount = iterations, discount
        self.uncertainty_penalty = uncertainty_penalty
        self._rng = np.random.default_rng(seed)
        self._warm_mean: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the receding-horizon warm start (call between episodes)."""
        self._warm_mean = None

    def plan(self, state: LatentState, goal: Subgoal | None = None) -> Action:
        if self._warm_mean is not None:  # shift last plan by one step
            mean = np.concatenate([self._warm_mean[1:], self._warm_mean[-1:]], axis=0)
        else:
            mean = np.zeros((self.horizon, self.action_dim))
        std = np.full((self.horizon, self.action_dim), 0.5 * (self.action_high - self.action_low))
        for _ in range(self.iterations):
            noise = self._rng.normal(size=(self.candidates, self.horizon, self.action_dim))
            sequences = np.clip(mean + std * noise, self.action_low, self.action_high)
            scores = self._imagined_returns(state, sequences)
            elite = sequences[np.argsort(scores)[-self.elites :]]
            mean = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), 0.05)
        self._warm_mean = mean
        return Action(data=mean[0].copy())

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        """Score (K,H,action_dim) candidate sequences by imagined discounted reward,
        epistemic-penalized per step (mean rollout; bounded horizon per ADR-0006)."""
        k = sequences.shape[0]
        latents = np.repeat(np.asarray(state.z, dtype=float).reshape(1, -1), k, axis=0)
        totals = np.zeros(k)
        discount = 1.0
        for t in range(self.horizon):
            mean, _, epistemic, _, reward = self._predict_batch(latents, sequences[:, t])
            totals += discount * (reward - self.uncertainty_penalty * epistemic)
            latents = mean
            discount *= self.discount
        return totals

    def _predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = getattr(self._model, "predict_batch", None)
        if batch is not None:
            result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = batch(
                latents, actions
            )
            return result
        # Protocol fallback: per-sample predict() for any WorldModel.
        preds = [
            self._model.predict(LatentState(z=z), Action(data=a))
            for z, a in zip(latents, actions, strict=True)
        ]
        return (
            np.stack([np.asarray(p.mean, dtype=float) for p in preds]),
            np.stack([np.asarray(p.var, dtype=float) for p in preds]),
            np.array([p.epistemic for p in preds]),
            np.array([p.aleatoric for p in preds]),
            np.array([p.reward for p in preds]),
        )


class JumpyOptionModel:
    """The learned temporally-abstract model (P5-001, ADR-0003): predicts the
    outcome of committing to an option in ONE jump — landing-latent distribution
    (ensemble: disagreement = epistemic, predicted variance = aleatoric,
    moment-matched total), cumulative discounted reward, and duration. This is
    what turns hierarchy from reactive control into hierarchical *planning*, and
    what bounds ADR-0001's compounding rollout error.

    Lives in the shared latent (ADR-0001) — no encoder of its own. Training
    convention: `update()` consumes latent-space option-transitions (`state.z` /
    `next_state.z` are latents, `option` set, `reward` cumulative discounted;
    duration target = `option.metadata["duration"]` if present — varying
    durations arrive with P5-002's VoE termination — else `option.horizon`).

    Contracts: interfaces.OptionModel + interfaces.Learner.
    """

    def __init__(
        self,
        option_names: Sequence[str] = (),
        latent_dim: int = 8,
        hidden: int = 64,
        ensemble: int = 5,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        self.latent_dim = latent_dim
        self._names = list(option_names)
        self._index = {name: i for i, name in enumerate(self._names)}
        in_dim = latent_dim + len(self._names)
        self._rng = np.random.default_rng(seed)
        self.members = [
            _MLP([in_dim, hidden, 2 * latent_dim], np.random.default_rng(seed * 1013 + 3 * k + 2), lr)
            for k in range(ensemble)
        ]
        self.reward_head = _MLP([in_dim, hidden // 2, 1], self._rng, lr)
        self.duration_head = _MLP([in_dim, hidden // 2, 1], self._rng, lr)

    def _one_hot(self, option: Option) -> np.ndarray:
        if option.name not in self._index:
            raise KeyError(
                f"unknown option {option.name!r}; registered: {', '.join(self._names) or 'none'}"
            )
        encoding = np.zeros(len(self._names))
        encoding[self._index[option.name]] = 1.0
        return encoding

    def _features(self, latents: np.ndarray, options: Sequence[Option]) -> np.ndarray:
        one_hots = np.stack([self._one_hot(o) for o in options])
        return np.concatenate([latents, one_hots], axis=1)

    def predict_option(self, state: LatentState, option: Option) -> Prediction:
        x = self._features(np.asarray(state.z, dtype=float).reshape(1, -1), [option])
        latent = x[:, : self.latent_dim]
        mus, variances = [], []
        for member in self.members:
            out, _ = member.forward(x)
            delta = out[:, : self.latent_dim]
            logvar = np.clip(out[:, self.latent_dim :], _LOGVAR_MIN, _LOGVAR_MAX)
            mus.append(latent + delta)  # residual: landings correlate with starts
            variances.append(np.exp(logvar))
        mu = np.stack(mus)
        aleatoric = np.stack(variances).mean(axis=0)[0]
        epistemic = mu.var(axis=0)[0]
        reward_out, _ = self.reward_head.forward(x)
        duration_out, _ = self.duration_head.forward(x)
        return Prediction(
            mean=mu.mean(axis=0)[0],
            var=aleatoric + epistemic,
            epistemic=float(epistemic.mean()),
            aleatoric=float(aleatoric.mean()),
            reward=float(reward_out[0, 0]),
            duration=float(duration_out[0, 0]),
        )

    def update(self, batch: Sequence[Transition]) -> dict[str, float]:
        """One training step on latent-space option-transitions (see class doc)."""
        for t in batch:
            if t.option is None:
                raise ValueError("option-transition without an option — a jumpy model "
                                 "cannot learn from unattributed jumps")
        options = [t.option for t in batch if t.option is not None]  # narrowed above
        starts = np.stack([np.asarray(t.state.z, dtype=float) for t in batch])
        landings = np.stack([np.asarray(t.next_state.z, dtype=float) for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=float).reshape(-1, 1)
        duration_targets = [float(o.metadata.get("duration", o.horizon)) for o in options]
        durations = np.asarray(duration_targets, dtype=float).reshape(-1, 1)
        x = self._features(starts, options)
        n, d = starts.shape[0], self.latent_dim

        for net in (self.reward_head, self.duration_head, *self.members):
            net.zero_grad()
        nll_total = 0.0
        for member in self.members:
            idx = self._rng.integers(0, n, size=n)  # bootstrap per member
            out, cache = member.forward(x[idx])
            delta = out[:, :d]
            logvar = np.clip(out[:, d:], _LOGVAR_MIN, _LOGVAR_MAX)
            var = np.exp(logvar)
            diff = starts[idx] + delta - landings[idx]
            nll_total += float(0.5 * np.mean(np.sum(diff**2 / var + logvar, axis=1)))
            d_mu = diff / var / n
            d_logvar = 0.5 * (1.0 - diff**2 / var) / n
            member.backward(np.concatenate([d_mu, d_logvar], axis=1), cache)
        reward_out, cache_r = self.reward_head.forward(x)
        loss_reward = float(np.mean((reward_out - rewards) ** 2))
        self.reward_head.backward(2.0 * (reward_out - rewards) / n, cache_r)
        duration_out, cache_d = self.duration_head.forward(x)
        loss_duration = float(np.mean((duration_out - durations) ** 2))
        self.duration_head.backward(2.0 * (duration_out - durations) / n, cache_d)
        for net in (self.reward_head, self.duration_head, *self.members):
            net.step()

        mus = np.stack([
            x[:, :d] + member.forward(x)[0][:, :d] for member in self.members
        ])
        return {
            "loss_nll": nll_total / len(self.members),
            "loss_reward": loss_reward,
            "loss_duration": loss_duration,
            "ensemble_disagreement": float(mus.var(axis=0).mean()),
        }


class HierarchicalManager:
    """Phase-5: search over the JumpyOptionModel, emit an option/subgoal; the worker
    executes it; VoE terminates it early on a surprise spike.

    Contract: interfaces.HierarchicalPlanner.
    """

    def plan_option(self, state: LatentState) -> Option:
        raise NotImplementedError("P5-002")

    def should_terminate(self, transition: Transition) -> bool:
        # Terminate when the option's predicted trajectory is violated (VoE).
        raise NotImplementedError("P5-002")
