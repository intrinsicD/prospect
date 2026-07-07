"""Unit tests for LatentActionModel (P13-001, ADR-0010): action-free learning, shapes,
and the decorrelation metric."""
from __future__ import annotations

import numpy as np

from prospect import interfaces
from prospect.observation import LatentActionModel


def test_latent_action_model_learns_and_has_right_shapes() -> None:  # P13-001
    model = LatentActionModel(obs_dim=2, latent_action_dim=1, seed=0)
    assert isinstance(model, interfaces.ObservationLearner)
    rng = np.random.default_rng(0)
    # a toy action-free stream: next = obs + [hidden_action, 0]
    obs = rng.uniform(-1.0, 1.0, (2000, 2))
    hidden = rng.uniform(-1.0, 1.0, (2000, 1))
    nxt = obs + np.concatenate([hidden, np.zeros((2000, 1))], axis=1)

    first = model.observe(obs[:128], nxt[:128])["loss_recon"]
    for _ in range(500):
        idx = rng.integers(0, 2000, 128)
        model.observe(obs[idx], nxt[idx])
    assert model.observe(obs[:128], nxt[:128])["loss_recon"] < first  # it learns to reconstruct

    z = model.infer_action(obs[:5], nxt[:5])
    assert z.shape == (5, 1)  # a batch of latent actions
    assert model.predict(obs[:5], z).shape == (5, 2)  # forward prediction
    z1 = model.infer_action(obs[0], nxt[0])  # a single pair rounds through 1-D
    assert z1.shape == (1,) and model.predict(obs[0], z1).shape == (2,)


def test_observe_reports_reconstruction_and_decorrelation() -> None:  # P13-001
    model = LatentActionModel(obs_dim=3, decorrelation=10.0, seed=1)
    rng = np.random.default_rng(1)
    obs = rng.uniform(-1.0, 1.0, (256, 3))
    metrics = model.observe(obs, obs + 0.1 * rng.uniform(-1.0, 1.0, (256, 3)))
    assert "loss_recon" in metrics and "decorrelation" in metrics  # the ADR-0010 identifiability term


def test_ground_makes_infer_action_recover_the_real_action() -> None:  # P14 reliability fix
    """After a supervised grounding, infer_action returns a directly-executable action
    (no separate calibration) — the Part-2 fix for imitation reliability."""
    model = LatentActionModel(obs_dim=2, latent_action_dim=1, seed=0)
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1.0, 1.0, (512, 2))
    action = rng.uniform(-1.0, 1.0, (512, 1))
    nxt = obs + np.concatenate([action, 0.1 * action], axis=1)
    for _ in range(400):  # watch (action-free)
        idx = rng.integers(0, 512, 64)
        model.observe(obs[idx], nxt[idx])
    for _ in range(1500):  # ground (a little labelled acting)
        idx = rng.integers(0, 512, 64)
        metrics = model.ground(obs[idx], action[idx], nxt[idx])
    assert "loss_ground" in metrics
    rec = np.atleast_2d(model.infer_action(obs, nxt))
    r2 = 1.0 - np.mean((rec - action) ** 2) / np.var(action)
    assert r2 > 0.9  # infer_action now recovers the real action directly
