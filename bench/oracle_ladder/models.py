"""Mean-rollout and simulator-prefix models for the OL-001 harness.

These adapters intentionally live outside the production model.  In particular,
neither class exposes ``predict_member_batch``: :class:`FlatPlanner` therefore uses
its recursive ensemble-mean path instead of TS-infinity member propagation.

``MeanOraclePrefixModel`` carries all rollout progress in its latent state rather
than mutable wrapper state.  Its layout is::

    [active learned latent (6), exact raw sidecar (3), exact steps remaining (1)]

That makes prefix depth independent for every candidate and resets it naturally
whenever the harness encodes a real observation for a new MPC call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from bench.bridge_control.fixture import transition_dynamics
from prospect.types import Action, LatentState, Prediction
from prospect.world_model import FlatWorldModel

Refresh = Literal["target", "online"]
RewardSource = Literal["learned", "oracle"]

_LEARNED_LATENT_DIM = 6
_RAW_DIM = 3
_ACTION_DIM = 2
_STATE_DIM = _LEARNED_LATENT_DIM + _RAW_DIM + 1
_SIDECAR_VAR = 1e-12
_EXACT_VAR = 1e-6


class MeanWorldModelAdapter:
    """Expose deterministic recursive ensemble-mean rollouts.

    ``FlatWorldModel.imagine`` is a TS-infinity rollout, so delegating it would not
    implement the mean rung.  One-step and batch predictions are delegated without
    arithmetic changes; only multi-step recurrence is supplied here.  Deliberately
    omitting ``predict_member_batch`` also selects the same mean branch inside
    :class:`prospect.planning.FlatPlanner`.
    """

    def __init__(self, learned: FlatWorldModel) -> None:
        self._learned = learned

    def predict(self, state: LatentState, action: Action) -> Prediction:
        """Return the learned model's one-step mixture-mean prediction unchanged."""

        return self._learned.predict(state, action)

    def predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the learned model's vectorized mixture-mean outputs unchanged."""

        return self._learned.predict_batch(latents, actions)

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        """Roll the ensemble mean forward recursively for an open-loop action list."""

        current = state
        predictions: list[Prediction] = []
        for action in actions:
            prediction = self.predict(current, action)
            predictions.append(prediction)
            current = LatentState(z=np.asarray(prediction.mean, dtype=float))
        return predictions


class MeanOraclePrefixModel:
    """Use exact dynamics for the first ``prefix_steps`` imagined transitions.

    Exact steps update the raw sidecar with ``transition_dynamics`` and refresh the
    active learned latent through either the EMA target encoder or online encoder.
    Once the counter reaches zero, only the active latent is advanced by recursive
    ensemble-mean dynamics; the no-longer-grounded raw sidecar is held fixed.

    Learned reward is always evaluated at the active *current* latent.  Exact reward
    is allowed only when the prefix covers the declared planning horizon, because
    oracle reward after learned latent recursion would require an undefined decoder.
    """

    learned_latent_dim = _LEARNED_LATENT_DIM
    raw_dim = _RAW_DIM
    action_dim = _ACTION_DIM
    state_dim = _STATE_DIM
    latent_dim = _STATE_DIM

    def __init__(
        self,
        learned: FlatWorldModel,
        prefix_steps: int,
        horizon: int,
        *,
        refresh: Refresh = "target",
        reward_source: RewardSource = "learned",
    ) -> None:
        if (
            learned.latent_dim != _LEARNED_LATENT_DIM
            or learned.obs_dim != _RAW_DIM
            or learned.action_dim != _ACTION_DIM
        ):
            raise ValueError("OL-001 requires a FlatWorldModel with obs_dim=3, action_dim=2, and latent_dim=6")
        if isinstance(prefix_steps, bool) or not isinstance(prefix_steps, int):
            raise TypeError("prefix_steps must be an integer")
        if prefix_steps < 0:
            raise ValueError("prefix_steps must be non-negative")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if refresh not in ("target", "online"):
            raise ValueError("refresh must be 'target' or 'online'")
        if reward_source not in ("learned", "oracle"):
            raise ValueError("reward_source must be 'learned' or 'oracle'")
        if reward_source == "oracle" and prefix_steps < horizon:
            raise ValueError("oracle reward is defined only when prefix_steps covers the planning horizon")

        self._learned = learned
        self.prefix_steps = prefix_steps
        self.horizon = horizon
        self.refresh = refresh
        self.reward_source = reward_source

    def initial_state(self, raw: object) -> LatentState:
        """Encode a real observation and reset its candidate-carried prefix counter."""

        raw_row = self._raw_row(raw)
        active = self._learned.encode(raw_row)
        return LatentState(
            z=np.concatenate(
                [
                    np.asarray(active.z, dtype=float),
                    raw_row,
                    np.array([float(self.prefix_steps)]),
                ]
            ),
            ood=active.ood,
        )

    def encode(self, raw: object) -> LatentState:
        """Alias for :meth:`initial_state`, convenient at the Agent codec seam."""

        return self.initial_state(raw)

    @staticmethod
    def raw_state(state: LatentState | np.ndarray) -> np.ndarray:
        """Return a copy of the exact raw sidecar from an augmented state."""

        values = state.z if isinstance(state, LatentState) else state
        row = MeanOraclePrefixModel._state_row(values)
        return row[_LEARNED_LATENT_DIM : _LEARNED_LATENT_DIM + _RAW_DIM].copy()

    @staticmethod
    def remaining_steps(state: LatentState | np.ndarray) -> int:
        """Return the exact number of candidate-carried oracle steps remaining."""

        values = state.z if isinstance(state, LatentState) else state
        row = MeanOraclePrefixModel._state_row(values)
        return MeanOraclePrefixModel._remaining(row[-1])

    def predict(self, state: LatentState, action: Action) -> Prediction:
        """Advance one augmented state while preserving real-state OOD semantics."""

        row = self._state_row(state.z)
        act = self._action_row(action.data)
        remaining = self._remaining(row[-1])
        if self.reward_source == "oracle" and remaining == 0:
            raise ValueError("oracle reward is undefined after the exact prefix is exhausted")

        active = row[:_LEARNED_LATENT_DIM]
        learned = self._learned.predict(LatentState(z=active, ood=state.ood), action)
        next_row, exact_reward, used_oracle = self._advance_row(row, act, np.asarray(learned.mean, dtype=float))

        if self.reward_source == "oracle":
            assert used_oracle and exact_reward is not None
            return Prediction(
                mean=next_row,
                var=np.full(_STATE_DIM, _EXACT_VAR),
                epistemic=0.0,
                aleatoric=_EXACT_VAR,
                reward=exact_reward,
                duration=learned.duration,
            )
        return Prediction(
            mean=next_row,
            var=self._augment_var(np.asarray(learned.var, dtype=float)),
            epistemic=learned.epistemic,
            aleatoric=learned.aleatoric,
            reward=learned.reward,
            duration=learned.duration,
        )

    def predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized planner path for augmented recursive-mean candidates."""

        states = np.asarray(latents, dtype=float)
        acts = np.asarray(actions, dtype=float)
        if states.ndim != 2 or states.shape[1] != _STATE_DIM:
            raise ValueError(f"latents must have shape (n, {_STATE_DIM}), got {states.shape}")
        if acts.ndim != 2 or acts.shape != (states.shape[0], _ACTION_DIM):
            raise ValueError(f"actions must have shape ({states.shape[0]}, {_ACTION_DIM}), got {acts.shape}")

        remaining = np.array([self._remaining(value) for value in states[:, -1]], dtype=int)
        if self.reward_source == "oracle" and np.any(remaining == 0):
            raise ValueError("oracle reward is undefined after the exact prefix is exhausted")

        # Copying the strided augmented-state slice recovers the same contiguous
        # input layout used by the direct mean adapter, which keeps the k=0 path
        # bitwise comparable rather than merely numerically close.
        active = states[:, :_LEARNED_LATENT_DIM].copy()
        learned_mean, learned_var, epistemic, aleatoric, learned_reward = self._learned.predict_batch(active, acts)

        means = states.copy()
        exact_rewards = np.empty(states.shape[0], dtype=float)
        oracle_mask = remaining > 0
        for index, (row, act, next_learned) in enumerate(zip(states, acts, learned_mean, strict=True)):
            next_row, exact_reward, used_oracle = self._advance_row(row, act, next_learned)
            means[index] = next_row
            exact_rewards[index] = exact_reward if used_oracle else np.nan

        variances = np.concatenate(
            [
                np.asarray(learned_var, dtype=float),
                np.full((states.shape[0], _RAW_DIM + 1), _SIDECAR_VAR),
            ],
            axis=1,
        )
        rewards = np.asarray(learned_reward, dtype=float).copy()
        epistemic_out = np.asarray(epistemic, dtype=float).copy()
        aleatoric_out = np.asarray(aleatoric, dtype=float).copy()
        if self.reward_source == "oracle":
            rewards[oracle_mask] = exact_rewards[oracle_mask]
            variances[oracle_mask] = _EXACT_VAR
            epistemic_out[oracle_mask] = 0.0
            aleatoric_out[oracle_mask] = _EXACT_VAR
        return means, variances, epistemic_out, aleatoric_out, rewards

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        """Roll the stateless prefix model forward over an open-loop action list."""

        if self.reward_source == "oracle" and len(actions) > self.prefix_steps:
            raise ValueError("oracle-reward rollout cannot exceed the exact prefix")
        current = state
        predictions: list[Prediction] = []
        for action in actions:
            prediction = self.predict(current, action)
            predictions.append(prediction)
            current = LatentState(z=np.asarray(prediction.mean, dtype=float))
        return predictions

    def _advance_row(
        self, row: np.ndarray, action: np.ndarray, learned_mean: np.ndarray
    ) -> tuple[np.ndarray, float | None, bool]:
        remaining = self._remaining(row[-1])
        if remaining == 0:
            # The sidecar has no raw preimage after learned recursion.  Holding it
            # fixed makes that boundary explicit and prevents accidental oracle use.
            return (
                np.concatenate([np.asarray(learned_mean, dtype=float), row[-4:]]),
                None,
                False,
            )

        raw = row[_LEARNED_LATENT_DIM : _LEARNED_LATENT_DIM + _RAW_DIM]
        next_raw, exact_reward = transition_dynamics(raw, action)
        if self.refresh == "target":
            refreshed = self._learned.encode_target(next_raw)
        else:
            refreshed = self._learned.encode(next_raw)
        next_row = np.concatenate(
            [
                np.asarray(refreshed.z, dtype=float),
                next_raw,
                np.array([float(remaining - 1)]),
            ]
        )
        return next_row, exact_reward, True

    @staticmethod
    def _augment_var(var: np.ndarray) -> np.ndarray:
        values = np.asarray(var, dtype=float).reshape(-1)
        if values.shape != (_LEARNED_LATENT_DIM,):
            raise ValueError(f"learned variance must have shape ({_LEARNED_LATENT_DIM},), got {values.shape}")
        return np.concatenate([values, np.full(_RAW_DIM + 1, _SIDECAR_VAR)])

    @staticmethod
    def _state_row(value: object) -> np.ndarray:
        row = np.asarray(value, dtype=float).reshape(-1)
        if row.shape != (_STATE_DIM,):
            raise ValueError(f"augmented state must have shape ({_STATE_DIM},), got {row.shape}")
        return row

    @staticmethod
    def _raw_row(value: object) -> np.ndarray:
        row = np.asarray(value, dtype=float).reshape(-1)
        if row.shape != (_RAW_DIM,):
            raise ValueError(f"raw state must have shape ({_RAW_DIM},), got {row.shape}")
        return row

    @staticmethod
    def _action_row(value: object) -> np.ndarray:
        row = np.asarray(value, dtype=float).reshape(-1)
        if row.shape != (_ACTION_DIM,):
            raise ValueError(f"action must have shape ({_ACTION_DIM},), got {row.shape}")
        return row

    @staticmethod
    def _remaining(value: float) -> int:
        rounded = int(round(float(value)))
        if rounded < 0 or not np.isclose(float(value), rounded, rtol=0.0, atol=1e-12):
            raise ValueError(f"remaining exact-step counter must be a non-negative integer, got {value}")
        return rounded
