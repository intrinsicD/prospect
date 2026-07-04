"""Unit tests for the composition root (P2-002): Agent wiring and the harness
episode driver — plus the P3-001 monitor hook."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bench.envs import Pendulum
from bench.loop import Acting, run_episode
from prospect.agent import Agent
from prospect.types import (
    Action,
    Competence,
    LatentState,
    Modality,
    Observation,
    Option,
    Prediction,
    Subgoal,
    Surprise,
    Transition,
)


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


class _StubModel:
    """Protocol-only world model: predicts staying put with a marked epistemic."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        return Prediction(mean=np.asarray(state.z, dtype=float), var=np.ones(2),
                          epistemic=0.42, aleatoric=0.1)

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        return [self.predict(state, a) for a in actions]


class _SpyMonitor:
    """CompetenceMonitor conforming spy: records what the Agent feeds it."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def surprise(self, prediction: Prediction, observed: LatentState) -> Surprise:
        return Surprise(total=0.0, epistemic=0.0, aleatoric=0.0)

    def update(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def competence(self, skill: str) -> Competence:
        return Competence(skill=skill, epistemic=0.0, learning_progress=0.0)

    def is_mastered(self, skill: str) -> bool:
        return False

    def is_forgetting(self, skill: str) -> bool:
        return False


def _monitored_agent(monitor: _SpyMonitor) -> Agent:
    return Agent(encode=lambda obs: LatentState(z=np.asarray(obs.data) * 2.0),
                 planner=_StubPlanner(), world_model=_StubModel(), monitor=monitor)


def test_monitored_agent_feeds_latent_transitions_with_act_time_prediction() -> None:
    monitor = _SpyMonitor()
    agent = _monitored_agent(monitor)
    option = Option(name="skill")
    action = agent.act(_obs(1.0))
    stored = agent.observe(_obs(1.0), action, _obs(2.0), reward=1.0, option=option)
    [fed] = monitor.transitions
    assert np.allclose(fed.state.z, [2.0, 0.0])  # monitor sees LATENT space
    assert np.allclose(fed.next_state.z, [4.0, 0.0])
    assert fed.prediction is not None and fed.prediction.epistemic == 0.42
    assert fed.option is option
    assert np.allclose(stored.state.z, [1.0, 0.0])  # storage stays RAW (P0-011)
    assert stored.prediction is fed.prediction  # act-time expectation backfilled


def test_observe_without_act_does_not_feed_the_monitor() -> None:
    monitor = _SpyMonitor()
    agent = _monitored_agent(monitor)
    agent.observe(_obs(1.0), Action(data=np.zeros(1)), _obs(2.0), reward=0.0)
    assert monitor.transitions == []


def test_reset_clears_the_pending_expectation() -> None:
    monitor = _SpyMonitor()
    agent = _monitored_agent(monitor)
    agent.act(_obs(1.0))
    agent.reset()
    agent.observe(_obs(1.0), Action(data=np.zeros(1)), _obs(2.0), reward=0.0)
    assert monitor.transitions == []


class _SpyMemory:
    """EpisodicMemory-conforming spy."""

    def __init__(self) -> None:
        self.added: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.added.append(transition)

    def sample(self, n: int) -> list[Transition]:
        return self.added[:n]

    def generative_replay(self, n: int) -> list[Transition]:
        return self.added[:n]


def test_agent_buffers_observed_transitions() -> None:
    memory = _SpyMemory()
    agent = Agent(encode=lambda obs: LatentState(z=np.asarray(obs.data)),
                  planner=_StubPlanner(), memory=memory)
    stored = agent.observe(_obs(1.0), Action(data=np.zeros(1)), _obs(2.0), reward=0.5)
    assert memory.added == [stored]  # raw-modality transition, buffered (P3-003)
