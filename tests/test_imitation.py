"""Unit tests for ObservationImitator (P14-001, ADR-0012): inverse-dynamics action
recovery, cloning, and shapes — on a tiny synthetic dynamics (fast, deterministic)."""
from __future__ import annotations

import numpy as np

from prospect.imitation import ObservationImitator
from prospect.types import Action, LatentState, Transition


def _batch(rng: np.random.Generator, n: int) -> list[Transition]:
    """Synthetic recoverable dynamics: next_obs = obs + [a, 0.1a, 0], so the action is a
    deterministic function of (obs, next_obs) — an inverse-dynamics model should recover it."""
    o = rng.uniform(-1.0, 1.0, (n, 3))
    a = rng.uniform(-1.0, 1.0, (n, 1))
    n2 = o + np.c_[a, 0.1 * a, np.zeros(n)]
    return [Transition(state=LatentState(z=o[i]), action=Action(data=a[i]),
                       next_state=LatentState(z=n2[i]), reward=0.0) for i in range(n)]


def test_imitator_recovers_actions_from_observation() -> None:  # P14-001
    rng = np.random.default_rng(0)
    imi = ObservationImitator(obs_dim=3, action_dim=1, seed=0)
    data = _batch(rng, 256)
    for _ in range(2500):
        idx = rng.integers(0, len(data), 64)
        imi.ground([data[i] for i in idx])
    ho = rng.uniform(-1.0, 1.0, (64, 3))
    ha = rng.uniform(-1.0, 1.0, (64, 1))
    hn = ho + np.c_[ha, 0.1 * ha, np.zeros(64)]
    rec = np.atleast_2d(imi.recover(ho, hn))
    r2 = 1.0 - np.mean((rec - ha) ** 2) / np.var(ha)
    assert r2 > 0.9  # recovers the hidden action from the transition


def test_imitator_clone_and_act_shapes() -> None:  # P14-001
    rng = np.random.default_rng(1)
    imi = ObservationImitator(obs_dim=3, action_dim=1, seed=1)
    o = rng.uniform(-1.0, 1.0, (32, 3))
    n2 = o + rng.uniform(-1.0, 1.0, (32, 3))
    metrics = imi.clone(o, n2)
    assert "loss_clone" in metrics
    assert np.asarray(imi.act(o[0])).shape == (1,)          # single obs -> single action
    assert np.asarray(imi.act(o)).shape == (32, 1)          # batch -> batch
    assert np.asarray(imi.recover(o[0], n2[0])).shape == (1,)  # single pair -> 1-D
