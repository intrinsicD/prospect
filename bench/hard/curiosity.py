"""A / non-gated (ADR-0011) — does curiosity-driven data collection fix the swingup
failure the P2-claim probe (`eval.py`) surfaced?

BH-001 found MBRL doesn't beat model-free on cartpole-swingup at equal budget. The cause
(measured): random data never reaches the upright goal region, so the world model is
ignorant exactly where the reward lives. The obvious fix is *directed* exploration — so
this study swaps random collection for the **curiosity curriculum** (P3-002 / ADR-0007):
an explore-mode planner whose epistemic coefficient is a *bonus* (from the live
`LearningProgressCurriculum`), steering collection toward high-uncertainty regions.

It reports, at matched budget, random-data MBRL vs curiosity-data MBRL, plus the goal
coverage of each data source (max reward reached, fraction of steps near the goal). The
finding is honest either way: curiosity *reaches* the goal region random data can't, but
whether that converts to exploit control is the measured question. Core untouched — the
curiosity machinery is the shipped P3-002 curriculum, driven on the DMC `Environment` seam.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import Transition
from prospect.voe import LearningProgressCurriculum, SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

from ..evals.p1_world_model import _train
from ..loop import run_episode
from . import eval as E
from .dmc_env import DMCEnvironment

SEEDS = (0, 1, 2)
CHUNK, TRAIN_PER_ROUND, MONITOR_FEED = 256, 250, 20      # P3-002 active-learning schedule
EXP_HORIZON, EXP_CANDIDATES, EXP_ELITES, EXP_ITERS = 8, 32, 6, 2  # light explore planner


@dataclass
class CuriosityResult:
    task: str
    seeds: list[int]
    mbrl_random: list[float]
    mbrl_curious: list[float]
    cov_random: list[float]      # max reward reached in the random data (goal-coverage proxy)
    cov_curious: list[float]     # max reward reached in the curious data
    goalfrac_random: list[float]
    goalfrac_curious: list[float]
    extra: dict[str, float] = field(default_factory=dict)

    def med(self, xs: list[float]) -> float:
        return float(np.median(xs))


def curious_rollout(env: DMCEnvironment, budget: int, seed: int) -> list[Transition]:
    """Collect `budget` env steps with the P3-002 explore-mode planner (epistemic *bonus*
    from the live curriculum). The seed chunk is random; thereafter collection seeks
    uncertainty. Returns the transitions — a fresh eval model is trained on them exactly
    like the random arm, so the only difference from `eval._rollout` is *where data comes
    from* (ADR-0007: the agent never picks the sign; the curriculum does)."""
    model = FlatWorldModel(obs_dim=env.obs_dim, action_dim=env.action_dim, seed=seed)
    monitor = SurpriseCompetenceMonitor()
    curriculum = LearningProgressCurriculum(monitor)
    planner = FlatPlanner(model, action_dim=env.action_dim, action_low=env.action_low,
                          action_high=env.action_high, horizon=EXP_HORIZON,
                          candidates=EXP_CANDIDATES, elites=EXP_ELITES,
                          iterations=EXP_ITERS, seed=seed)
    explorer = Agent(encode=lambda o: model.encode(o.data), planner=planner)
    rng = np.random.default_rng(seed + 17)
    data: list[Transition] = []
    for r in range(budget // CHUNK):
        if r > 0:  # seed chunk random for both arms; then seek uncertainty
            planner.uncertainty_penalty = curriculum.uncertainty_coefficient()
            _, chunk = run_episode(env, explorer, CHUNK, seed * 131 + r, collect=True)
        else:
            _, chunk = run_episode(env, E._PolicyAgent(E._random_policy(env, seed * 131 + r)),
                                   CHUNK, seed * 131 + r, collect=True)
        data.extend(chunk)
        _train(model, data, TRAIN_PER_ROUND, rng)
        for t in chunk[:MONITOR_FEED]:  # keep the curriculum's mode decision live
            pred = model.predict(model.encode(t.state.z), t.action)
            monitor.update(Transition(state=model.encode(t.state.z), action=t.action,
                                      next_state=model.encode_target(t.next_state.z),
                                      reward=t.reward, prediction=pred))
    return data


def _coverage(data: list[Transition]) -> tuple[float, float]:
    """(max reward reached, fraction of steps near the goal) — reward is the DMC
    uprightness/goal-proximity signal in [0, 1], so it is a coverage proxy."""
    r = np.array([t.reward for t in data])
    return float(r.max()), float(np.mean(r > 0.5))


def run_curiosity_study(task: str = "swingup", seeds: Sequence[int] = SEEDS) -> CuriosityResult:
    env = DMCEnvironment("cartpole", task)
    res = CuriosityResult(task, list(seeds), [], [], [], [], [], [],
                          {"budget_env_steps": float(E.BUDGET)})
    for seed in seeds:
        rand = E._rollout(env, E.BUDGET, seed)
        cur = curious_rollout(env, E.BUDGET, seed)
        # Same downstream as the P2-claim probe (exploit-mode MBRL) — only the collection
        # policy differs, so any gap is attributable to the data distribution.
        res.mbrl_random.append(E._mean_return(env, E._mbrl_agent(env, rand, seed), seed))
        res.mbrl_curious.append(E._mean_return(env, E._mbrl_agent(env, cur, seed), seed))
        cr_max, cr_gf = _coverage(rand)
        cu_max, cu_gf = _coverage(cur)
        res.cov_random.append(cr_max)
        res.cov_curious.append(cu_max)
        res.goalfrac_random.append(cr_gf)
        res.goalfrac_curious.append(cu_gf)
    return res
