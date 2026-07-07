"""BH-001 adapter smoke test (ADR-0011). Skipped cleanly when the optional
`[bench-hard]` extra is absent — so the numpy-only CI never touches it — and a fast
contract check (no eval, no training) when dm_control is installed."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dm_control", reason="optional [bench-hard] extra not installed")

from bench.envs import Environment  # noqa: E402
from bench.hard.dmc_env import DMCEnvironment  # noqa: E402
from prospect.types import Action, Modality  # noqa: E402


def _env() -> DMCEnvironment:
    return DMCEnvironment("cartpole", "swingup")


def test_conforms_to_environment_protocol() -> None:
    assert isinstance(_env(), Environment)


def test_dims_and_action_box_exposed() -> None:
    env = _env()
    assert env.obs_dim > 0 and env.action_dim > 0
    assert env.action_low < env.action_high


def test_reset_returns_state_observation() -> None:
    env = _env()
    obs = env.reset(seed=0)
    assert obs.modality is Modality.STATE
    assert np.asarray(obs.data).shape == (env.obs_dim,)


def test_step_returns_obs_reward_done() -> None:
    env = _env()
    env.reset(seed=0)
    obs, reward, done = env.step(Action(data=np.zeros(env.action_dim)))
    assert np.asarray(obs.data).shape == (env.obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_reset_is_seed_reproducible() -> None:
    env = _env()
    a = env.reset(seed=3).data.copy()
    b = env.reset(seed=3).data.copy()
    c = env.reset(seed=4).data.copy()
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_action_out_of_box_is_clipped_not_raised() -> None:
    env = _env()
    env.reset(seed=0)
    # An action far outside the spec box must be tolerated (clipped), not error.
    obs, _, _ = env.step(Action(data=np.full(env.action_dim, 1e3)))
    assert np.all(np.isfinite(obs.data))


def test_curiosity_rollout_collects_transitions() -> None:
    """A study: the P3-002 explore path runs on DMC and collects the budget."""
    from bench.hard.curiosity import curious_rollout  # noqa: E402

    env = _env()
    data = curious_rollout(env, budget=512, seed=0)  # 1 random chunk + 1 explore chunk
    assert len(data) == 512
    assert all(t.state.z.shape == (env.obs_dim,) for t in data[:5])


def test_imitation_helpers_recover_a_linear_map() -> None:
    """B study: the regressor learns and action recovery + R^2 behave on synthetic data."""
    from bench.hard.imitation import _mlp_regress, _r2  # noqa: E402

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(256, 1))
    y = 2.0 * x
    net = _mlp_regress([1, 16, 1], x, y, 2000, 0)
    pred, _ = net.forward(np.array([[0.5]]))
    assert abs(float(pred[0, 0]) - 1.0) < 0.2                 # learned y = 2x
    assert _r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) > 0.99
    assert _r2(np.zeros(3), np.array([1.0, 2.0, 3.0])) < 0.5  # a bad predictor scores low
