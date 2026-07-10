"""P2 eval (task P2-001): planning beats reaction at equal env-step budget.

Three agents on the Pendulum reference task, identical evaluation episodes:
- MBRL: BUDGET random env steps -> FlatWorldModel (P1 recipe; probes logged to run
  `p2`, so the P1-era sentinels judge the model THIS phase trained) -> FlatPlanner
  (CEM in imagination, exploit-mode epistemic penalty per ADR-0007).
- Model-free baseline: CEM-ES direct policy search over a small tanh-MLP policy;
  every fitness rollout is real env interaction counted against the SAME budget.
- Random policy: the floor.

Learning budget = env steps used to learn (training set / fitness rollouts).
Held-out probe sets and evaluation episodes are measurement apparatus for both
agents alike and are not counted. Pass: planner return beats the baseline AND the
floor on every seed (the documented P2 criterion, not only the median).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import Action, Observation
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..loop import run_episode
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe, _rollout, _train

RUN_ID = "p2"
SEEDS = [0, 1, 2]
BUDGET = 4096  # env steps each agent may use for learning
EP_LEN, EVAL_EPISODES = 100, 5
POP, GENS, ES_ELITES, PARAMS = 10, 4, 3, 81  # tanh-MLP policy: (3->16->1)

Policy = Callable[[Observation], Action]


class _PolicyAgent:
    """Adapter: a plain (stateless) policy as an `Acting` agent for `run_episode`."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    def act(self, obs: Observation) -> Action:
        return self._policy(obs)

    def reset(self) -> None:
        return None


def _eval_policy(policy: Policy, seed: int) -> float:
    """Mean return over the shared fresh evaluation episodes (same for all agents)."""
    agent = _PolicyAgent(policy)
    return float(np.mean([
        run_episode(Pendulum(), agent, EP_LEN, 9000 + seed * 50 + e)[0]
        for e in range(EVAL_EPISODES)
    ]))


def _mlp_policy(params: np.ndarray) -> Policy:
    w1 = params[:48].reshape(3, 16)
    b1, w2, b2 = params[48:64], params[64:80].reshape(16, 1), params[80:]

    def act(obs: Observation) -> Action:
        h = np.tanh(np.asarray(obs.data, dtype=float) @ w1 + b1)
        return Action(data=(np.tanh(h @ w2 + b2) * 2.0).ravel())

    return act


def _random_policy(seed: int) -> Policy:
    rng = np.random.default_rng(seed + 700)

    def act(obs: Observation) -> Action:
        return Action(data=rng.uniform(-2.0, 2.0, size=1))

    return act


def _es_baseline(seed: int) -> tuple[Policy, int]:
    """Model-free CEM-ES policy search; returns (policy, env steps consumed)."""
    rng = np.random.default_rng(seed + 300)
    env = Pendulum()
    mean, std, used = np.zeros(PARAMS), np.full(PARAMS, 0.5), 0
    for gen in range(GENS):
        population = mean + std * rng.normal(size=(POP, PARAMS))
        fitness = []
        for candidate in population:  # common start state per generation (CRN)
            agent = _PolicyAgent(_mlp_policy(candidate))
            fitness.append(run_episode(env, agent, EP_LEN, seed * 991 + gen)[0])
            used += EP_LEN
        elite = population[np.argsort(fitness)[-ES_ELITES:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 0.05
    return _mlp_policy(mean), used


def _mpc_agent(seed: int, log: RunLog) -> Agent:
    """World model from BUDGET random steps, then CEM planning via the composition
    root: `Agent` wires the encode seam to the planner (P2-002)."""
    train = _rollout(Pendulum(), BUDGET, seed)
    heldout = _rollout(Pendulum(), 256, seed + 500)
    ood = _rollout(Pendulum(init_omega=14.0, omega_max=16.0), 256, seed + 900)
    model = FlatWorldModel(seed=seed)
    _train(model, train, STEPS, np.random.default_rng(seed + 1),
           probe=_make_probe(heldout, heldout + ood, seed), log=log,
           step_offset=seed * SEED_STEP_OFFSET)
    planner = FlatPlanner(model, seed=seed)
    return Agent(encode=lambda obs: model.encode(obs.data), planner=planner)


def _eval_planner(agent: Agent, seed: int) -> float:
    return float(np.mean([
        run_episode(Pendulum(), agent, EP_LEN, 9000 + seed * 50 + e)[0]
        for e in range(EVAL_EPISODES)
    ]))


@gate_check("P2")
def check_p2() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    planner_returns, baseline_returns, random_returns = [], [], []
    for seed in SEEDS:
        planner_return = _eval_planner(_mpc_agent(seed, log), seed)
        baseline_policy, es_used = _es_baseline(seed)
        baseline_return = _eval_policy(baseline_policy, seed)
        random_return = _eval_policy(_random_policy(seed), seed)
        planner_returns.append(planner_return)
        baseline_returns.append(baseline_return)
        random_returns.append(random_return)
        metrics |= {
            f"planner_return_s{seed}": planner_return,
            f"baseline_return_s{seed}": baseline_return,
            f"random_return_s{seed}": random_return,
            f"baseline_env_steps_s{seed}": float(es_used),
        }
    med_planner = float(np.median(planner_returns))
    med_baseline = float(np.median(baseline_returns))
    med_random = float(np.median(random_returns))
    metrics |= {
        "planner_return_median": med_planner,
        "baseline_return_median": med_baseline,
        "random_return_median": med_random,
        "budget_env_steps": float(BUDGET),
    }
    wins = [
        planner > baseline and planner > random
        for planner, baseline, random in zip(
            planner_returns, baseline_returns, random_returns, strict=True
        )
    ]
    passed = all(wins)
    metrics["wins_every_seed"] = float(passed)
    detail = (
        f"median eval return over {EVAL_EPISODES} shared episodes: planner {med_planner:.2f} "
        f"vs model-free ES baseline {med_baseline:.2f} vs random {med_random:.2f} "
        f"(learning budget {BUDGET} env steps each; ES used "
        f"{int(metrics['baseline_env_steps_s0'])}); planner "
        f"{'wins' if passed else 'does not win'} on every seed"
    )
    return GateResult(phase="P2", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
