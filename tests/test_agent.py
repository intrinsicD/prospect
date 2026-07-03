"""Unit tests for the composition root (P2-002): Agent wiring and the harness
episode driver."""
from __future__ import annotations

import numpy as np

from bench.envs import Pendulum
from bench.loop import Acting, run_episode
from prospect.agent import Agent
from prospect.types import Action, LatentState, Modality, Observation, Option, Subgoal


class _StubPlanner:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.seen: list[LatentState] = []

    def plan(self, state: LatentState, goal: Subgoal | None = None) -> Action:
        self.seen.append(state)
        return Action(data=np.array([0.25]))

    def reset(self) -> None:
        self.reset_calls += 1


def _obs(value: float) -> Observation:
    return Observation(modality=Modality.STATE, data=np.array([value, 0.0]))


def _agent(planner: _StubPlanner) -> Agent:
    return Agent(encode=lambda obs: LatentState(z=np.asarray(obs.data) * 2.0), planner=planner)


def test_act_encodes_then_plans() -> None:
    planner = _StubPlanner()
    action = _agent(planner).act(_obs(1.0))
    assert float(np.asarray(action.data)[0]) == 0.25
    assert np.allclose(planner.seen[0].z, [2.0, 0.0])  # planner saw the ENCODED latent


def test_observe_stores_raw_modality_with_passthrough() -> None:
    agent = _agent(_StubPlanner())
    option = Option(name="skill")
    t = agent.observe(_obs(1.0), Action(data=np.array([0.1])), _obs(3.0), reward=0.5, option=option)
    assert np.allclose(t.state.z, [1.0, 0.0])  # raw obs, NOT encoded (P0-011)
    assert np.allclose(t.next_state.z, [3.0, 0.0])
    assert t.reward == 0.5
    assert t.option is option


def test_reset_reaches_the_planner() -> None:
    planner = _StubPlanner()
    _agent(planner).reset()
    assert planner.reset_calls == 1


def test_run_episode_drives_an_agent_and_collects() -> None:
    planner = _StubPlanner()
    agent = _agent(planner)
    conforming: Acting = agent  # mypy: Agent satisfies the harness's Acting seam
    assert isinstance(conforming, Acting)
    total, transitions = run_episode(Pendulum(), agent, steps=3, seed=5, collect=True)
    assert planner.reset_calls == 1  # run_episode resets at episode start
    assert isinstance(total, float)
    assert len(transitions) == 3
    assert transitions[0].state.z.shape == (3,)  # raw pendulum obs, re-encodable


def test_run_episode_is_seed_reproducible() -> None:
    def run() -> float:
        return run_episode(Pendulum(), _agent(_StubPlanner()), steps=4, seed=11)[0]

    assert run() == run()
