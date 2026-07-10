"""Unit tests for the P1 FlatWorldModel (task P1-001): prediction shape/contract,
finite calibrated surprise, learning progress, and the Pendulum reference task."""
from __future__ import annotations

from math import isfinite

import numpy as np
import pytest

from bench.envs import Environment, Pendulum
from prospect import interfaces
from prospect.types import Action, LatentState, Transition
from prospect.world_model import FlatWorldModel


def _random_transitions(n: int, seed: int = 0) -> list[Transition]:
    """Tiny synthetic system: obs' = 0.9*obs + 0.3*a on the first dim + tanh coupling."""
    rng = np.random.default_rng(seed)
    transitions = []
    for _ in range(n):
        obs = rng.uniform(-1, 1, size=3)
        a = float(rng.uniform(-1, 1))
        nxt = 0.9 * obs + 0.3 * a * np.array([1.0, 0.5, -0.5]) + 0.1 * np.tanh(obs[::-1])
        transitions.append(
            Transition(state=LatentState(z=obs), action=Action(data=np.array([a])),
                       next_state=LatentState(z=nxt), reward=float(-np.sum(obs**2)))
        )
    return transitions


def test_predict_returns_full_distribution() -> None:
    model = FlatWorldModel(seed=0)
    state = model.encode(np.array([1.0, 0.0, 0.5]))
    pred = model.predict(state, Action(data=np.array([0.3])))
    assert len(pred.mean) == model.latent_dim
    assert len(pred.var) == model.latent_dim
    assert all(v > 0 for v in pred.var)
    assert pred.epistemic > 0  # fresh ensemble members disagree
    assert pred.aleatoric > 0
    assert pred.duration == 1.0
    target = model.encode_target(np.array([0.9, 0.1, 0.4]))
    assert isfinite(pred.log_prob(target.z))


def test_update_returns_metrics_and_learns() -> None:
    model = FlatWorldModel(seed=1, hidden=32)
    data = _random_transitions(512, seed=1)
    rng = np.random.default_rng(2)

    def run(steps: int) -> list[float]:
        losses = []
        for _ in range(steps):
            idx = rng.integers(0, len(data), size=32)
            metrics = model.update([data[i] for i in idx])
            losses.append(metrics["loss_nll"])
        return losses

    losses = run(300)
    for key in ("loss_nll", "loss_reward", "loss_inverse", "latent_std_min",
                "latent_effective_rank", "ensemble_disagreement"):
        assert isfinite(model.update(data[:32])[key])
    assert np.mean(losses[-50:]) < np.mean(losses[:50])  # NLL falls: it is learning


def test_imagine_rolls_out_open_loop() -> None:
    model = FlatWorldModel(seed=3)
    state = model.encode(np.array([0.5, -0.5, 0.0]))
    actions = [Action(data=np.array([0.1])), Action(data=np.array([-0.2])), Action(data=np.array([0.0]))]
    preds = model.imagine(state, actions)
    assert len(preds) == 3
    assert all(len(p.mean) == model.latent_dim for p in preds)


def test_imagine_propagates_member_trajectories_and_horizon_uncertainty() -> None:
    """TS∞ exposes compounding spread that a mean-state rollout hides (U-001)."""
    model = FlatWorldModel(obs_dim=1, action_dim=1, latent_dim=1, hidden=2, ensemble=2, seed=4)
    # Member 0 moves away from zero; member 1 mirrors it toward/through zero.
    # Around a fixed mean state their one-step disagreement stays constant, while
    # independently propagated member states diverge over the horizon.
    for member, sign in zip(model.members, (1.0, -1.0), strict=True):
        for weight in member.weights:
            weight[...] = 0.0
        for bias in member.biases:
            bias[...] = 0.0
        member.weights[0][0, 0] = 1.0
        member.weights[1][0, 0] = sign

    state = LatentState(z=np.array([0.5]))  # deliberately far from the members' fixed point
    actions = [Action(data=np.zeros(1)) for _ in range(3)]
    ts_predictions = model.imagine(state, actions)

    mean_state = np.asarray(state.z, dtype=float).reshape(1, -1)
    mean_rollout_epistemic = []
    for action in actions:
        mean, _, epistemic, _, _ = model.predict_batch(
            mean_state, np.asarray(action.data, dtype=float).reshape(1, -1)
        )
        mean_rollout_epistemic.append(float(epistemic[0]))
        mean_state = mean

    assert ts_predictions[0].epistemic == pytest.approx(mean_rollout_epistemic[0])
    assert ts_predictions[-1].epistemic > mean_rollout_epistemic[-1]
    assert [p.aleatoric for p in ts_predictions] == pytest.approx([1.0, 2.0, 3.0])


def test_world_model_satisfies_both_contracts() -> None:
    model = FlatWorldModel()
    assert isinstance(model, interfaces.WorldModel)
    assert isinstance(model, interfaces.Learner)


def test_pendulum_is_a_seeded_environment() -> None:
    env = Pendulum()
    assert isinstance(env, Environment)
    first = env.reset(seed=42)
    again = Pendulum().reset(seed=42)
    assert np.allclose(first.data, again.data)  # seeded resets reproduce
    obs, reward, done = env.step(Action(data=np.array([0.5])))
    assert obs.data.shape == (3,)
    assert isfinite(reward)
    assert done is False


def test_pendulum_stochastic_variant_adds_noise() -> None:
    def next_omega(noise: float) -> float:
        env = Pendulum(noise_std=noise)
        env.reset(seed=7)
        obs, _, _ = env.step(Action(data=np.array([0.0])))
        return float(obs.data[2])

    assert next_omega(0.0) == next_omega(0.0)  # deterministic core
    assert next_omega(0.0) != next_omega(0.5)  # noise actually enters the dynamics


def test_distance_aware_epistemic_rises_out_of_distribution() -> None:  # P9-005
    model = FlatWorldModel(seed=0)
    for _ in range(50):  # populate the input-standardization stats (training range ~[-1,1])
        model.update(_random_transitions(64, seed=1))
    action = Action(data=np.array([0.0]))
    in_dist = model.encode(np.array([0.2, -0.1, 0.3]))   # near the training range
    far = model.encode(np.array([12.0, -9.0, 8.0]))       # far outside it
    assert far.ood is not None and far.ood > 0.0          # the far obs is flagged OOD
    assert (in_dist.ood or 0.0) < far.ood                 # and more OOD than the near one
    # predict scales epistemic by the OOD score; a synthesized latent (ood=None) is unscaled
    boosted = model.predict(far, action).epistemic
    plain = model.predict(LatentState(z=far.z), action).epistemic  # same latent, no ood
    assert boosted > plain
