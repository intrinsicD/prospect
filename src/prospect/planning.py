"""Planning (R1, R2). Flat MPC in imagination, plus the hierarchical manager over a
jumpy option-model. See ADR-0001, ADR-0003, ADR-0006/0007. Tasks: P2-001 (done),
P5-001, P5-002.
"""
from __future__ import annotations

import numpy as np

from .interfaces import WorldModel
from .types import Action, LatentState, Option, Prediction, Subgoal, Transition


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
    """Phase-5: the temporally-abstract model — predicts the outcome of committing to
    an option (landing latent, cumulative reward, duration, uncertainty). This is what
    turns hierarchy from reactive control into hierarchical *planning* (ADR-0003).

    Contract: interfaces.OptionModel.
    """

    def predict_option(self, state: LatentState, option: Option) -> Prediction:
        raise NotImplementedError("P5-001")


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
