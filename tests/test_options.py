"""Unit tests for the JumpyOptionModel (P5-001): one-jump outcome prediction —
landing, cumulative reward, duration — learned from option-transitions."""
from __future__ import annotations

import numpy as np
import pytest

from prospect import interfaces
from prospect.planning import JumpyOptionModel
from prospect.types import Action, LatentState, Option, Transition

UP = Option(name="up", horizon=3)
DOWN = Option(name="down", horizon=5)


def _jump(rng: np.random.Generator, option: Option) -> Transition:
    """Synthetic system: 'up' jumps +2 with reward +1; 'down' jumps -2, reward -1."""
    start = rng.uniform(-1.0, 1.0, size=4)
    shift, reward = (2.0, 1.0) if option is UP else (-2.0, -1.0)
    return Transition(
        state=LatentState(z=start),
        action=Action(data=np.zeros(1)),
        next_state=LatentState(z=start + shift),
        reward=reward,
        option=option,
    )


def _trained(steps: int = 400) -> JumpyOptionModel:
    model = JumpyOptionModel(["up", "down"], latent_dim=4, hidden=32, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(steps):
        batch = [_jump(rng, UP if rng.random() < 0.5 else DOWN) for _ in range(32)]
        model.update(batch)
    return model


def test_conforms_to_option_model_and_learner() -> None:
    model = JumpyOptionModel(["up"])
    assert isinstance(model, interfaces.OptionModel)
    assert isinstance(model, interfaces.Learner)


def test_learns_landing_reward_and_duration() -> None:
    model = _trained()
    state = LatentState(z=np.zeros(4))
    up = model.predict_option(state, UP)
    down = model.predict_option(state, DOWN)
    assert np.allclose(np.asarray(up.mean), 2.0, atol=0.3)
    assert np.allclose(np.asarray(down.mean), -2.0, atol=0.3)
    assert up.reward == pytest.approx(1.0, abs=0.2)
    assert down.reward == pytest.approx(-1.0, abs=0.2)
    assert up.duration == pytest.approx(3.0, abs=0.5)
    assert down.duration == pytest.approx(5.0, abs=0.5)
    assert all(v > 0 for v in up.var)


def test_epistemic_falls_with_training() -> None:
    fresh = JumpyOptionModel(["up", "down"], latent_dim=4, hidden=32, seed=0)
    state = LatentState(z=np.zeros(4))
    before = fresh.predict_option(state, UP).epistemic
    after = _trained().predict_option(state, UP).epistemic
    assert after < before


def test_unknown_option_and_missing_option_fail_loudly() -> None:
    model = JumpyOptionModel(["up"])
    with pytest.raises(KeyError, match="unknown option"):
        model.predict_option(LatentState(z=np.zeros(8)), Option(name="sideways"))
    with pytest.raises(ValueError, match="without an option"):
        model.update([Transition(state=LatentState(z=np.zeros(8)),
                                 action=Action(data=np.zeros(1)),
                                 next_state=LatentState(z=np.zeros(8)), reward=0.0)])


def test_duration_metadata_overrides_horizon() -> None:
    # P5-002's early termination will record actual durations in metadata.
    cut = Option(name="up", horizon=3, metadata={"duration": 1.0})
    model = JumpyOptionModel(["up"], latent_dim=4, hidden=32, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(300):
        start = rng.uniform(-1, 1, size=4)
        model.update([Transition(state=LatentState(z=start), action=Action(data=np.zeros(1)),
                                 next_state=LatentState(z=start + 2.0), reward=1.0,
                                 option=cut)] * 16)
    prediction = model.predict_option(LatentState(z=np.zeros(4)), cut)
    assert prediction.duration == pytest.approx(1.0, abs=0.3)
