"""Tests for the harness `Environment` seam (P0-004) and the import-direction rule
(golden rule 3: the harness imports the core; the core never imports the harness).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bench import Environment
from bench.envs import PointMass
from prospect.types import Action, Modality, Observation

_CORE = Path(__file__).resolve().parent.parent / "src" / "prospect"


class _CountdownEnv:
    """Minimal test double: reward 1.0 per step, done after two steps."""

    def __init__(self) -> None:
        self._steps_left = 2

    def reset(self, seed: int | None = None) -> Observation:
        self._steps_left = 2
        return Observation(modality=Modality.STATE, data=[float(seed or 0)])

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        self._steps_left -= 1
        return Observation(modality=Modality.STATE, data=[0.0]), 1.0, self._steps_left <= 0


def test_dummy_env_satisfies_protocol() -> None:
    assert isinstance(_CountdownEnv(), Environment)


def test_env_round_trips_with_core_types() -> None:
    env = _CountdownEnv()
    obs = env.reset(seed=7)
    assert isinstance(obs, Observation)
    assert obs.data == [7.0]
    obs, reward, done = env.step(Action(data=[0.0]))
    assert isinstance(obs, Observation)
    assert reward == 1.0
    assert done is False
    _, _, done = env.step(Action(data=[0.0]))
    assert done is True


def test_pointmass_satisfies_protocol_and_steps() -> None:  # P9-003 second environment
    env = PointMass()
    assert isinstance(env, Environment)
    obs = env.reset(seed=3)
    assert isinstance(obs, Observation) and np.asarray(obs.data).shape == (4,)  # x, y, vx, vy
    obs, reward, done = env.step(Action(data=np.array([1.0, -1.0])))
    assert np.asarray(obs.data).shape == (4,) and isinstance(reward, float) and done is False


def test_pointmass_set_state_and_nonlinear_drag() -> None:  # P9-003
    env = PointMass()
    # set_state places an exact state; a rightward push from rest moves +x, +vx.
    env.set_state(0.0, 0.0, 0.0, 0.0)
    obs, _, _ = env.step(Action(data=np.array([1.0, 0.0])))
    x, _, vx, _ = np.asarray(obs.data)
    assert vx > 0.0 and x > 0.0
    # quadratic drag: the per-step velocity gain is smaller at high speed than at rest.
    env.set_state(0.0, 0.0, 0.0, 0.0)
    gain_rest = np.asarray(env.step(Action(data=np.array([1.0, 0.0])))[0].data)[2]
    env.set_state(0.0, 0.0, 5.0, 0.0)
    gain_fast = np.asarray(env.step(Action(data=np.array([1.0, 0.0])))[0].data)[2] - 5.0
    assert gain_fast < gain_rest  # drag opposes motion more at higher speed (nonlinear)


def test_core_never_imports_the_harness() -> None:
    offenders = [
        path.name
        for path in _CORE.glob("*.py")
        if "import bench" in path.read_text() or "from bench" in path.read_text()
    ]
    assert offenders == []
