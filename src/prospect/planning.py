"""Planning (R1, R2). Flat iCEM/MPC in imagination, the learned jumpy option-model,
and the hierarchical manager over it. See ADR-0001, ADR-0003, ADR-0006/0007.
Tasks: P2-001 (done), P5-001 (done), P5-002 (done), U-001/U-002 (done).
"""
from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np

from .interfaces import OptionModel, WorldModel
from .types import Action, LatentState, MemberRollout, Option, Prediction, Subgoal, Transition
from .world_model import _LOGVAR_MAX, _LOGVAR_MIN, _MLP  # core-internal reuse


class FlatPlanner:
    """iCEM/MPC in imagination over a `WorldModel` (R1, P2-001/U-002).

    Candidate action sequences are proposed with temporally colored noise, retained
    elites, and a softmax-weighted distribution update, then rolled out in latent
    space and scored by
    discounted imagined reward **minus a per-step epistemic penalty** — ADR-0006's
    model-exploitation control. This is the ADR-0007 *exploit-mode* consumer: the
    penalty sign is fixed here; the exploration bonus (sign flip) belongs to the
    curriculum (P3-002), never to the planner.

    Receding horizon: `plan()` returns the first action of the optimized sequence
    and warm-starts the next call with the shifted elite mean and pool; call
    `reset()` between episodes. `goal` is accepted per the Protocol and ignored until
    hierarchical planning lands (P5) — P2 plans on reward.

    Contract: interfaces.Planner. Uses `TrajectoryWorldModel`-style member batches
    when offered, then the model's vectorized `predict_batch`, and finally falls
    back to the narrow protocol's per-sample `predict()`.
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
        uncertainty_penalty: float = 0.03,
        seed: int = 0,
        epistemic_horizon_bound: float | None = None,
        colored_beta: float = 2.0,
        keep_elite_fraction: float = 0.3,
        temperature: float = 0.5,
    ) -> None:
        if not 1 <= elites <= candidates:
            raise ValueError("elites must be between 1 and candidates")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if colored_beta < 0.0:
            raise ValueError("colored_beta must be non-negative")
        if not 0.0 <= keep_elite_fraction <= 1.0:
            raise ValueError("keep_elite_fraction must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self._model = world_model
        self.action_dim, self.action_low, self.action_high = action_dim, action_low, action_high
        self.horizon, self.candidates, self.elites = horizon, candidates, elites
        self.iterations, self.discount = iterations, discount
        self.uncertainty_penalty = uncertainty_penalty
        if epistemic_horizon_bound is not None and epistemic_horizon_bound < 0.0:
            raise ValueError("epistemic_horizon_bound must be non-negative or None")
        self.epistemic_horizon_bound = epistemic_horizon_bound
        self.colored_beta = colored_beta
        self.keep_elite_fraction = keep_elite_fraction
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)
        self._warm_mean: np.ndarray | None = None
        self._warm_elites: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the receding-horizon warm start (call between episodes)."""
        self._warm_mean = None
        self._warm_elites = None

    def _sample_colored_noise(self, count: int) -> np.ndarray:
        """Temporally correlated Gaussian noise via an f^(-beta/2) FFT filter.

        Horizons below four have too few distinct real-FFT bins for meaningful
        spectral coloring; they still receive a valid unit-Gaussian proposal.
        """
        shape = (count, self.horizon, self.action_dim)
        white = self._rng.normal(size=shape)
        if count == 0 or self.horizon <= 1 or self.colored_beta == 0.0:
            return white
        spectrum = np.fft.rfft(white, axis=1)
        frequencies = np.fft.rfftfreq(self.horizon)
        # Give the DC component the lowest finite frequency's power.  Zeroing DC
        # would force every proposal to have zero temporal mean, excluding useful
        # sustained actions and even making short-horizon samples anti-correlated.
        frequencies[0] = frequencies[1]
        scale = frequencies ** (-self.colored_beta / 2.0)
        # The inverse FFT of the response is the circular filter kernel.  Its L2
        # norm is the exact marginal standard deviation induced by unit white
        # noise, so this deterministic normalization preserves N(0, 1) marginals
        # without normalizing away each proposal's temporal offset.
        kernel = np.fft.irfft(scale, n=self.horizon)
        normalizer = float(np.sqrt(np.sum(kernel**2)))
        spectrum *= scale[None, :, None]
        colored = np.fft.irfft(spectrum, n=self.horizon, axis=1)
        return np.asarray(colored / normalizer, dtype=float)

    @staticmethod
    def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
        """Stable score weights; infinite temperature approaches a uniform mean."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        values = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite")
        logits = (values - float(np.max(values))) / temperature
        weights = np.exp(logits)
        return np.asarray(weights / weights.sum(), dtype=float)

    @staticmethod
    def _shift_sequences(sequences: np.ndarray) -> np.ndarray:
        """Receding-horizon shift, repeating the last action at the open tail."""
        return np.concatenate([sequences[:, 1:], sequences[:, -1:]], axis=1)

    def plan(self, state: LatentState, goal: Subgoal | None = None) -> Action:
        if self._warm_mean is not None:  # shift last plan by one step
            mean = np.concatenate([self._warm_mean[1:], self._warm_mean[-1:]], axis=0)
        else:
            mean = np.zeros((self.horizon, self.action_dim))
        # iCEM's fixed initial sigma is 0.5 in normalized [-1, 1] action
        # coordinates: one quarter of the physical action range.
        std = np.full((self.horizon, self.action_dim), 0.25 * (self.action_high - self.action_low))
        keep_count = min(
            self.elites,
            int(np.ceil(self.keep_elite_fraction * self.elites)),
        )
        carried = (
            self._shift_sequences(self._warm_elites[:keep_count])
            if self._warm_elites is not None
            else np.empty((0, self.horizon, self.action_dim))
        )
        best_sequence: np.ndarray | None = None
        best_score = -np.inf
        elite = np.empty((0, self.horizon, self.action_dim))
        for _ in range(self.iterations):
            carried = carried[: self.candidates]
            fresh_count = self.candidates - len(carried)
            noise = self._sample_colored_noise(fresh_count)
            fresh = np.clip(mean + std * noise, self.action_low, self.action_high)
            sequences = np.concatenate([fresh, carried], axis=0)
            scores = self._imagined_returns(state, sequences)
            best_index = int(np.argmax(scores))
            if float(scores[best_index]) > best_score:
                best_score = float(scores[best_index])
                best_sequence = sequences[best_index].copy()
            elite_indices = np.argsort(scores)[-self.elites :][::-1]
            elite = sequences[elite_indices]
            elite_scores = scores[elite_indices]
            weights = self._softmax_weights(elite_scores, self.temperature)
            mean = np.sum(weights[:, None, None] * elite, axis=0)
            variance = np.sum(weights[:, None, None] * (elite - mean) ** 2, axis=0)
            std = np.maximum(np.sqrt(variance), 0.05)
            carried = elite[:keep_count].copy()
        assert best_sequence is not None  # iterations >= 1, candidates >= elites >= 1
        self._warm_mean = mean
        self._warm_elites = elite.copy()
        return Action(data=best_sequence[0].copy())

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        """Score (K,H,action_dim) candidate sequences by imagined discounted reward,
        using TS∞ when the model exposes per-member trajectories.

        The optional epistemic horizon bound stops adding rewards after a
        candidate's accumulated trajectory spread crosses the calibrated bound.
        The crossing step itself is scored; later, untrusted steps are not.
        """
        k = sequences.shape[0]
        latents = np.repeat(np.asarray(state.z, dtype=float).reshape(1, -1), k, axis=0)
        member_latents = latents
        member_batch = getattr(self._model, "predict_member_batch", None)
        totals = np.zeros(k)
        accumulated_epistemic = np.zeros(k)
        active = np.ones(k, dtype=bool)
        discount = 1.0
        for t in range(self.horizon):
            if member_batch is not None:
                result: MemberRollout = member_batch(
                    member_latents,
                    sequences[:, t],
                    initial_ood=state.ood if t == 0 else None,
                )
                member_latents = np.asarray(result.states, dtype=float)
                member_rewards = np.asarray(result.rewards, dtype=float)
                epistemic = np.asarray(result.epistemic, dtype=float)
                reward = member_rewards.mean(axis=0)
                latents = member_latents.mean(axis=0)
            else:
                mean, _, epistemic, _, reward = self._predict_batch(
                    latents,
                    sequences[:, t],
                    initial_ood=state.ood if t == 0 else None,
                )
                latents = mean
            totals[active] += discount * (
                reward[active] - self.uncertainty_penalty * epistemic[active]
            )
            accumulated_epistemic[active] += epistemic[active]
            if self.epistemic_horizon_bound is not None:
                active &= accumulated_epistemic <= self.epistemic_horizon_bound
                if not np.any(active):
                    break
            discount *= self.discount
        return totals

    def _predict_batch(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        initial_ood: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = getattr(self._model, "predict_batch", None)
        if batch is not None and initial_ood is None:
            result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = batch(
                latents, actions
            )
            return result
        # Protocol fallback: per-sample predict() for any WorldModel.
        preds = [
            self._model.predict(LatentState(z=z, ood=initial_ood), Action(data=a))
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
    """The two-level manager (P5-002, ADR-0003): *plans* over the learned jumpy
    option-model — exhaustive search over option sequences (K^depth stays small
    for a competence-gated library at shallow depth), scoring cumulative
    discounted reward with duration-aware discounting minus the exploit-mode
    epistemic penalty (ADR-0006/0007) — and emits the first option of the best
    sequence. The worker executes it; `should_terminate` cuts it early when
    one-step VoE spikes (the re-planning interrupt, the one signal's job #4).

    Contract: interfaces.HierarchicalPlanner.
    """

    def __init__(
        self,
        option_model: OptionModel | None = None,
        options: Sequence[Option] = (),
        depth: int = 3,
        discount: float = 0.99,
        uncertainty_penalty: float = 1.0,
        surprise_threshold: float = float("inf"),
    ) -> None:
        self._option_model = option_model
        self._options = list(options)
        self.depth = depth
        self.discount = discount
        self.uncertainty_penalty = uncertainty_penalty
        self.surprise_threshold = surprise_threshold

    def plan_option(self, state: LatentState) -> Option:
        if self._option_model is None or not self._options:
            raise ValueError("the manager needs an option model and a non-empty option set")
        best_score, best_first = -float("inf"), self._options[0]
        for sequence in product(self._options, repeat=self.depth):
            latent = state
            score, discount = 0.0, 1.0
            for option in sequence:
                prediction = self._option_model.predict_option(latent, option)
                score += discount * prediction.reward
                score -= self.uncertainty_penalty * prediction.epistemic
                discount *= self.discount ** max(prediction.duration, 1.0)
                latent = LatentState(z=prediction.mean)
            if score > best_score:
                best_score, best_first = score, sequence[0]
        return best_first

    def should_terminate(self, transition: Transition) -> bool:
        """Terminate the running option when its predicted step was violated:
        one-step surprise above the calibrated threshold (VoE, ADR-0003)."""
        if transition.prediction is None:
            return False  # nothing was expected — nothing to violate
        surprise = -transition.prediction.log_prob(transition.next_state.z)
        return surprise > self.surprise_threshold
