"""B / P14 (non-gated, ADR-0011) — imitation-from-observation on cartpole swingup.

The A study (`curiosity.py`) showed exploration — even curiosity-driven — cannot crack
swingup at feasible budgets: it reaches the upright region but can't convert sparse
coverage into control. This study asks the complementary question: can the agent
**reproduce a swingup it never performed, purely from watching** an expert's
*observations* (no expert actions)?

Pipeline (imitation from observation):
1. **Watch** an expert swingup demonstration — its observation trajectory only; the
   expert's actions are hidden (used solely for an oracle upper bound).
2. **Recover the actions from observation** at the same interaction budget a from-scratch
   agent gets (grounding = the agent's own labelled steps). Two routes, both reported:
   - `inverse-dynamics`: a direct model g(obs, next_obs) → action (BCO-style).
   - `latent-action` (P13, ADR-0010): the action-free `LatentActionModel` infers a latent
     action per demo step; a tiny calibration maps latent → real action. This is the
     arc-faithful route (learn behaviour from action-free observation, ground with a little
     acting) — honestly measured against the direct route.
3. **Clone** a closed-loop reactive policy from the (obs, recovered-action) pairs and run it.

Headline: reproduction beats from-scratch MBRL (which fails swingup — the A finding) and a
shuffled-demo negative control, approaching an oracle clone trained on the true actions.
Watching does what exploration at the same budget cannot. The core (`src/prospect/`) is
untouched — this is all harness, on the existing `bench.Environment` seam.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from prospect.observation import LatentActionModel
from prospect.types import Action, Observation
from prospect.world_model import _MLP

from ..loop import run_episode
from . import eval as E
from .dmc_env import DMCEnvironment

SEEDS = (0, 1, 2)
DEMO_SEARCH_SEED, DEMO_SEED, DEMO_EP = 0, 123, 250
N_GROUND = 4096          # agent's own labelled steps == the from-scratch budget
DEMO_POP, DEMO_GENS, DEMO_ELITES = 40, 30, 8
CLONE_STEPS, CAL_STEPS, LAM_STEPS = 4000, 3000, 6000
EVAL_EPISODES = 3


@dataclass
class ImitationResult:
    demo_return: float
    from_scratch: list[float]
    oracle: list[float]
    inverse_dyn: list[float]
    latent_action: list[float]
    shuffled: list[float]
    recovery_r2: dict[str, float]          # median action-recovery R^2 per route

    def med(self, xs: list[float]) -> float:
        return float(np.median(xs))


def _mlp_regress(sizes: list[int], x: np.ndarray, y: np.ndarray, steps: int,
                 seed: int, lr: float = 3e-3, batch: int = 64) -> _MLP:
    """Supervised MSE regressor on the numpy `_MLP` (linear output)."""
    rng = np.random.default_rng(seed)
    net = _MLP(sizes, rng, lr)
    y2 = y.reshape(len(y), -1)
    for _ in range(steps):
        idx = rng.integers(0, len(x), min(batch, len(x)))
        pred, cache = net.forward(x[idx])
        net.zero_grad()
        net.backward(2.0 * (pred - y2[idx]) / len(idx), cache)
        net.step()
    return net


def _generate_demo(env: DMCEnvironment) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """A strong expert swingup via scaled CEM-ES policy search (deterministic). Returns the
    demonstration observation trajectory (O, next-O) and the (hidden) true actions A, plus
    the expert's return. Only O/next-O are used by the imitation routes; A is oracle-only."""
    dim = env.obs_dim * E.HIDDEN + E.HIDDEN + E.HIDDEN * env.action_dim + env.action_dim
    rng = np.random.default_rng(DEMO_SEARCH_SEED + 1)
    mean, std, best, best_fit = np.zeros(dim), np.full(dim, 0.6), np.zeros(dim), -1e9
    for gen in range(DEMO_GENS):
        pop = mean + std * rng.normal(size=(DEMO_POP, dim))
        fit = np.array([run_episode(env, E._PolicyAgent(
            E._mlp_policy(c, env.obs_dim, env.action_dim, env.action_low, env.action_high)),
            DEMO_EP, 7 * DEMO_SEARCH_SEED + gen)[0] for c in pop])
        order = np.argsort(fit)
        if fit[order[-1]] > best_fit:
            best_fit, best = float(fit[order[-1]]), pop[order[-1]].copy()
        elite = pop[order[-DEMO_ELITES:]]
        mean, std = elite.mean(0), elite.std(0) + 0.05
    expert = E._mlp_policy(best, env.obs_dim, env.action_dim, env.action_low, env.action_high)
    obs = env.reset(seed=DEMO_SEED)
    o_list, a_list, o2_list = [], [], []
    for _ in range(DEMO_EP):
        a = expert(obs)
        nobs, _, _ = env.step(a)
        o_list.append(obs.data.copy())
        a_list.append(a.data.copy())
        o2_list.append(nobs.data.copy())
        obs = nobs
    demo_return = float(run_episode(env, E._PolicyAgent(expert), DEMO_EP, DEMO_SEED)[0])
    return np.array(o_list), np.array(a_list), np.array(o2_list), demo_return


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    p, t = pred.reshape(-1), true.reshape(-1)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return float(1 - np.sum((t - p) ** 2) / ss_tot) if ss_tot > 0 else 0.0


def _recover_inverse(env: DMCEnvironment, g_o: np.ndarray, g_a: np.ndarray, g_o2: np.ndarray,
                     demo_o: np.ndarray, demo_o2: np.ndarray, seed: int) -> np.ndarray:
    """Direct inverse dynamics g(obs, next_obs) -> action, fit on grounding labels."""
    inv = _mlp_regress([2 * env.obs_dim, 64, env.action_dim],
                       np.concatenate([g_o, g_o2], 1), g_a, CLONE_STEPS, seed)
    return inv.forward(np.concatenate([demo_o, demo_o2], 1))[0]


def _recover_latent(env: DMCEnvironment, g_o: np.ndarray, g_a: np.ndarray, g_o2: np.ndarray,
                    demo_o: np.ndarray, demo_o2: np.ndarray, seed: int) -> np.ndarray:
    """P13 LatentActionModel (action-free) on demo+grounding streams, then a tiny
    latent -> real-action calibration fit on the grounding labels (ADR-0010)."""
    lam = LatentActionModel(obs_dim=env.obs_dim, latent_action_dim=env.action_dim, seed=seed)
    stream = list(zip(np.concatenate([g_o, demo_o]), np.concatenate([g_o2, demo_o2]), strict=True))
    rng = np.random.default_rng(seed + 3)
    for _ in range(LAM_STEPS):
        idx = rng.integers(0, len(stream), 64)
        lam.observe_batch([stream[i] for i in idx])
    z_g = np.array([lam.infer_action(g_o[i], g_o2[i]) for i in range(len(g_o))])
    cal = _mlp_regress([env.action_dim, 16, env.action_dim], z_g, g_a, CAL_STEPS, seed + 1)
    z_demo = np.array([lam.infer_action(demo_o[i], demo_o2[i]) for i in range(len(demo_o))])
    return cal.forward(z_demo)[0]


def _clone_return(env: DMCEnvironment, demo_o: np.ndarray, target_actions: np.ndarray,
                  seed: int) -> float:
    """Clone a closed-loop reactive policy from (obs, action) and run it from the demo start."""
    pi = _mlp_regress([env.obs_dim, 64, env.action_dim], demo_o, target_actions, CLONE_STEPS, seed)

    def policy(o: Observation) -> Action:
        out, _ = pi.forward(np.asarray(o.data, dtype=float).reshape(1, -1))
        return Action(data=np.clip(out.reshape(-1), env.action_low, env.action_high))

    return float(np.mean([run_episode(env, E._PolicyAgent(policy), DEMO_EP, DEMO_SEED + e)[0]
                          for e in range(EVAL_EPISODES)]))


def run_imitation(seeds: Sequence[int] = SEEDS) -> ImitationResult:
    env = DMCEnvironment("cartpole", "swingup")
    demo_o, demo_a, demo_o2, demo_return = _generate_demo(env)
    shuffle = np.random.default_rng(0).permutation(len(demo_a))
    res = ImitationResult(demo_return, [], [], [], [], [], {})
    r2_inv, r2_lam = [], []
    for seed in seeds:
        gtr = E._rollout(env, N_GROUND, seed)
        g_o = np.array([t.state.z for t in gtr])
        g_a = np.array([t.action.data for t in gtr])
        g_o2 = np.array([t.next_state.z for t in gtr])
        a_inv = _recover_inverse(env, g_o, g_a, g_o2, demo_o, demo_o2, seed)
        a_lam = _recover_latent(env, g_o, g_a, g_o2, demo_o, demo_o2, seed)
        r2_inv.append(_r2(a_inv, demo_a))
        r2_lam.append(_r2(a_lam, demo_a))
        res.from_scratch.append(E._mean_return(env, E._mbrl_agent(env, gtr, seed), seed))
        res.oracle.append(_clone_return(env, demo_o, demo_a, seed + 10))
        res.inverse_dyn.append(_clone_return(env, demo_o, a_inv, seed + 20))
        res.latent_action.append(_clone_return(env, demo_o, a_lam, seed + 30))
        res.shuffled.append(_clone_return(env, demo_o, demo_a[shuffle], seed + 40))
    res.recovery_r2 = {"inverse_dyn": float(np.median(r2_inv)),
                       "latent_action": float(np.median(r2_lam))}
    return res
