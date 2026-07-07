"""BH-001 (ADR-0011) — re-run the **P2 claim** on a real MuJoCo task the repo did
not author: does MPC/CEM over a learned `FlatWorldModel` beat a model-free baseline
**at equal env-step budget**?

This is a NON-GATED credibility probe. There is no pass/fail bar that ships a phase;
the deliverable is an honest report (raw returns, seed spread, matched-budget deltas).
A small-budget planner losing to published SAC would be unremarkable — the only
question here is whether it beats a model-free baseline given the *same* env steps,
i.e. whether the model-based machine buys anything on foreign dynamics.

Three agents on identical seeded eval episodes, same learning budget:
- MBRL: BUDGET random env steps -> FlatWorldModel (the P1 recipe) -> FlatPlanner (CEM
  in imagination, exploit-mode epistemic penalty, ADR-0007), wired by `Agent`.
- Model-free: CEM-ES policy search over a small tanh-MLP; every fitness rollout is
  real env interaction counted against the SAME budget.
- Random: the floor.

A light **P1-calibration spot-check** (seed 0) rides along: median ensemble
epistemic on a held-out probe should fall as the training set grows (reported, not
gated). Reuses the harness's P1/P2 machinery unchanged (`_train`, `run_episode`) —
only the environment is new, which is the whole point of the seam.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, Observation, Transition
from prospect.world_model import FlatWorldModel

from ..evals.p1_world_model import STEPS, _train
from ..loop import run_episode
from .dmc_env import DMCEnvironment

# --- probe configuration (matched to P2 where it maps; documented where it differs) ---
# Matched to the shipped P2 gate so this is the *same machine* on foreign dynamics,
# not a re-tuned one: BUDGET, EP_LEN and the CEM-ES search (POP·GENS·EP_LEN = 4000 ≤
# BUDGET, so the model-free baseline is budget-matched) all equal P2's, and the
# planner uses P2's FlatPlanner defaults. Deviating from these to make MBRL look
# better would defeat the point of the probe.
SEEDS = (0, 1, 2, 3, 4)
BUDGET = 4096          # env steps each agent may use to learn (== P2)
EP_LEN = 100           # eval / fitness episode length (== P2)
EVAL_EPISODES = 3      # shared fresh episodes, identical seeds across all agents
HORIZON, CANDIDATES, ELITES, ITERS = 20, 64, 8, 3   # CEM/MPC — P2 FlatPlanner defaults
POP, GENS, HIDDEN = 10, 4, 16                        # model-free CEM-ES policy search (== P2)
CALIB_SMALL = 256      # low-data point for the P1-calibration spot-check

Policy = Callable[[Observation], Action]


@dataclass
class TaskResult:
    """One (domain, task) rung's measured returns; serialized into the report."""

    domain: str
    task: str
    obs_dim: int
    action_dim: int
    seeds: list[int]
    mbrl: list[float]
    model_free: list[float]
    random: list[float]
    baseline_env_steps: int
    calib_epistemic_ratio: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def med_mbrl(self) -> float:
        return float(np.median(self.mbrl))

    @property
    def med_model_free(self) -> float:
        return float(np.median(self.model_free))

    @property
    def med_random(self) -> float:
        return float(np.median(self.random))

    @property
    def mbrl_beats_baseline(self) -> bool:
        return self.med_mbrl > self.med_model_free and self.med_mbrl > self.med_random


class _PolicyAgent:
    """A stateless policy as an `Acting` agent for `run_episode` (mirrors P2)."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    def act(self, obs: Observation) -> Action:
        return self._policy(obs)

    def reset(self) -> None:
        return None


def _rollout(env: DMCEnvironment, n: int, seed: int) -> list[Transition]:
    """Random-policy transitions; raw obs ride in `.state.z` (P0-011)."""
    rng = np.random.default_rng(seed)
    transitions: list[Transition] = []
    obs = env.reset(seed=seed * 7919 + 1)
    for _ in range(n):
        a = rng.uniform(env.action_low, env.action_high, size=env.action_dim)
        action = Action(data=a)
        next_obs, reward, done = env.step(action)
        transitions.append(
            Transition(state=LatentState(z=obs.data), action=action,
                       next_state=LatentState(z=next_obs.data), reward=reward)
        )
        obs = env.reset(seed=seed * 7919 + len(transitions)) if done else next_obs
    return transitions


def _mbrl_agent(env: DMCEnvironment, transitions: list[Transition], seed: int) -> Agent:
    model = FlatWorldModel(obs_dim=env.obs_dim, action_dim=env.action_dim, seed=seed)
    _train(model, transitions, STEPS, np.random.default_rng(seed + 1))
    planner = FlatPlanner(
        model, action_dim=env.action_dim, action_low=env.action_low, action_high=env.action_high,
        horizon=HORIZON, candidates=CANDIDATES, elites=ELITES, iterations=ITERS, seed=seed,
    )
    return Agent(encode=lambda obs: model.encode(obs.data), planner=planner)


def _mlp_policy(params: np.ndarray, obs_dim: int, action_dim: int,
                low: float, high: float) -> Policy:
    n1 = obs_dim * HIDDEN
    w1 = params[:n1].reshape(obs_dim, HIDDEN)
    b1 = params[n1:n1 + HIDDEN]
    w2 = params[n1 + HIDDEN:n1 + HIDDEN + HIDDEN * action_dim].reshape(HIDDEN, action_dim)
    b2 = params[-action_dim:]
    mid, half = (high + low) / 2.0, (high - low) / 2.0

    def act(obs: Observation) -> Action:
        h = np.tanh(np.asarray(obs.data, dtype=float) @ w1 + b1)
        out = np.tanh(h @ w2 + b2)  # (-1, 1)
        return Action(data=(mid + half * out).ravel())

    return act


def _n_params(obs_dim: int, action_dim: int) -> int:
    return obs_dim * HIDDEN + HIDDEN + HIDDEN * action_dim + action_dim


def _es_baseline(env: DMCEnvironment, seed: int) -> tuple[Policy, int]:
    """Model-free CEM-ES over a tanh-MLP policy; returns (policy, env steps used)."""
    rng = np.random.default_rng(seed + 300)
    dim = _n_params(env.obs_dim, env.action_dim)
    mean, std, used = np.zeros(dim), np.full(dim, 0.5), 0
    for gen in range(GENS):
        population = mean + std * rng.normal(size=(POP, dim))
        fitness = []
        for candidate in population:  # common start state per generation (CRN)
            agent = _PolicyAgent(_mlp_policy(candidate, env.obs_dim, env.action_dim,
                                             env.action_low, env.action_high))
            fitness.append(run_episode(env, agent, EP_LEN, seed * 991 + gen)[0])
            used += EP_LEN
        elite = population[np.argsort(fitness)[-max(2, POP // 3):]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 0.05
    policy = _mlp_policy(mean, env.obs_dim, env.action_dim, env.action_low, env.action_high)
    return policy, used


def _random_policy(env: DMCEnvironment, seed: int) -> Policy:
    rng = np.random.default_rng(seed + 700)

    def act(obs: Observation) -> Action:
        return Action(data=rng.uniform(env.action_low, env.action_high, size=env.action_dim))

    return act


def _mean_return(env: DMCEnvironment, agent: _PolicyAgent | Agent, seed: int) -> float:
    """Mean return over the shared fresh eval episodes (identical for every agent)."""
    return float(np.mean([
        run_episode(env, agent, EP_LEN, 9000 + seed * 50 + e)[0]
        for e in range(EVAL_EPISODES)
    ]))


def _median_epistemic(model: FlatWorldModel, probe: list[Transition]) -> float:
    epi = [model.predict(model.encode(t.state.z), t.action).epistemic for t in probe]
    return float(np.median(epi))


def _calibration_ratio(env: DMCEnvironment, transitions: list[Transition],
                       probe: list[Transition], seed: int) -> float:
    """P1-calibration spot-check: epistemic(full budget) / epistemic(small budget).
    < 1 means uncertainty falls with data (well-calibrated); reported, not gated."""
    small = FlatWorldModel(obs_dim=env.obs_dim, action_dim=env.action_dim, seed=seed)
    _train(small, transitions[:CALIB_SMALL], STEPS, np.random.default_rng(seed + 1))
    full = FlatWorldModel(obs_dim=env.obs_dim, action_dim=env.action_dim, seed=seed)
    _train(full, transitions, STEPS, np.random.default_rng(seed + 1))
    e_small = _median_epistemic(small, probe)
    e_full = _median_epistemic(full, probe)
    return float(e_full / e_small) if e_small > 0 else float("nan")


def run_task(domain: str, task: str, seeds: tuple[int, ...] = SEEDS,
             calibrate: bool = True) -> TaskResult:
    env = DMCEnvironment(domain, task)
    mbrl_returns, mf_returns, rand_returns, es_used = [], [], [], 0
    calib: float | None = None
    for seed in seeds:
        train = _rollout(env, BUDGET, seed)
        mbrl_returns.append(_mean_return(env, _mbrl_agent(env, train, seed), seed))
        baseline_policy, es_used = _es_baseline(env, seed)
        mf_returns.append(_mean_return(env, _PolicyAgent(baseline_policy), seed))
        rand_returns.append(_mean_return(env, _PolicyAgent(_random_policy(env, seed)), seed))
        if calibrate and seed == seeds[0]:
            probe = _rollout(env, 256, seed + 500)
            calib = _calibration_ratio(env, train, probe, seed)
    return TaskResult(
        domain=domain, task=task, obs_dim=env.obs_dim, action_dim=env.action_dim,
        seeds=list(seeds), mbrl=mbrl_returns, model_free=mf_returns, random=rand_returns,
        baseline_env_steps=es_used, calib_epistemic_ratio=calib,
        extra={"budget_env_steps": float(BUDGET)},
    )
