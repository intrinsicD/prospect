"""P14 eval: observe → repeat (imitation from observation) — task P14-001, ADR-0012.

The agent reproduces a behaviour it has only **watched** — an expert swing-up given as an
observation trajectory, with NO expert actions. It recovers the demonstrated actions from
observation with an inverse-dynamics model grounded on a little of its OWN labelled,
broad-coverage experience (`imitation.ObservationImitator`), then clones a reactive policy
and reproduces the swing-up.

Setup (per seed): a CEM-ES search produces an expert swing-up on `PendulumSwingup`
(hanging-down start — the numpy analogue of DMC cartpole-swingup); we record ONLY its
observation trajectory (its actions are used ONLY to score recovery). The agent grounds
action-recovery on a random broad-coverage stream (standard `Pendulum`, resets to all
angles — same dynamics), watches the demo, and reproduces it.

Four criteria (median over seeds), scored as (return − floor)/(demo − floor) so 1.0 = the
expert and 0.0 = a do-nothing floor, plus all applicable collapse sentinels on run `p14`:
1. **Reproduces the demo** — the imitation policy's score is high (it swings up).
2. **Recovers actions from observation** — recovered demo actions match the true (hidden)
   actions with R² above a floor (recovery is real, not given).
3. **Specific behaviour (negative control)** — a shuffled-demo clone collapses toward the
   floor (it imitates THIS behaviour, not just "move"; the gate-overfit guard).
4. **Watching is what does it** — imitation (watch the expert) beats cloning the agent's OWN
   random data by a margin: same machinery, same grounding, only the watched stream differs.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prospect.imitation import ObservationImitator
from prospect.types import Action, LatentState, Observation, Transition
from prospect.world_model import _MLP

from ..envs import Pendulum, PendulumSwingup
from ..gates import GateResult, gate_check
from ..loop import run_episode
from ..runlog import RUNS_DIR, RunLog
from .p13_observation import _log_sentinel_model

RUN_ID = "p14"
SEEDS = [0, 1, 2]
EP, MAXT = 120, 2.0                       # eval episode length; torque bound
POP, GENS, ELITES, N_PARAMS = 40, 30, 8, 81   # CEM-ES expert search: (3->16->1) tanh policy
GROUND_N, GROUND_STEPS, CLONE_STEPS = 2048, 3000, 3000
REPRODUCE_MIN = 0.6      # imitation score floor (swings up)
RECOVERY_R2_MIN = 0.5    # recovered-vs-true action R^2 floor on the demo states
SHUFFLE_MAX = 0.35       # shuffled-demo control score ceiling (negative control)
WATCH_MARGIN = 0.4       # imitation score − clone-own-random score (watching's contribution)

Policy = Callable[[Observation], Action]


class _PolicyAgent:
    """A stateless policy as an `Acting` agent for `run_episode`."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    def act(self, obs: Observation) -> Action:
        return self._policy(obs)

    def reset(self) -> None:
        return None


def _mlp_policy(params: np.ndarray) -> Policy:
    w1 = params[:48].reshape(3, 16)
    b1, w2, b2 = params[48:64], params[64:80].reshape(16, 1), params[80:]

    def act(obs: Observation) -> Action:
        h = np.tanh(np.asarray(obs.data, dtype=float) @ w1 + b1)
        return Action(data=(np.tanh(h @ w2 + b2) * MAXT).ravel())

    return act


def _cem_expert(seed: int) -> np.ndarray:
    """CEM-ES policy search for an expert swing-up on PendulumSwingup (deterministic)."""
    rng = np.random.default_rng(seed + 1)
    mean, std = np.zeros(N_PARAMS), np.full(N_PARAMS, 0.6)
    best, best_fit = mean, -1e9
    for gen in range(GENS):
        pop = mean + std * rng.normal(size=(POP, N_PARAMS))
        fit = np.array([run_episode(PendulumSwingup(), _PolicyAgent(_mlp_policy(c)), EP, 7 * seed + gen)[0]
                        for c in pop])
        order = np.argsort(fit)
        if fit[order[-1]] > best_fit:
            best_fit, best = float(fit[order[-1]]), pop[order[-1]].copy()
        elite = pop[order[-ELITES:]]
        mean, std = elite.mean(0), elite.std(0) + 0.05
    return best


def _broad_grounding(n: int, seed: int) -> list[Transition]:
    """Random broad-coverage transitions: standard Pendulum resets to all angles, so the
    inverse-dynamics grounding covers the demo's swing-up states (same dynamics)."""
    env = Pendulum()
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    tr: list[Transition] = []
    for i in range(n):
        if i % 30 == 0 and i:
            obs = env.reset(seed=seed * 7 + i)
        a = Action(data=np.array([rng.uniform(-MAXT, MAXT)]))
        nobs, r, _ = env.step(a)
        tr.append(Transition(state=LatentState(z=obs.data), action=a,
                             next_state=LatentState(z=nobs.data), reward=r))
        obs = nobs
    return tr


def _demo(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Roll out the CEM-ES expert on PendulumSwingup; return (obs, hidden actions, next-obs,
    expert return). Only obs/next-obs feed the imitation; actions are recovery-scoring only."""
    expert = _mlp_policy(_cem_expert(seed))
    env = PendulumSwingup()
    obs = env.reset(seed=seed * 17 + 321)
    o, a, o2 = [], [], []
    for _ in range(EP):
        act = expert(obs)
        nobs, _, _ = env.step(act)
        o.append(obs.data.copy())
        a.append(act.data.copy())
        o2.append(nobs.data.copy())
        obs = nobs
    return np.array(o), np.array(a), np.array(o2), _return(expert, seed)


def _return(policy: Policy, seed: int, n: int = 3) -> float:
    return float(np.mean([run_episode(PendulumSwingup(), _PolicyAgent(policy), EP, 5000 + seed * 10 + e)[0]
                          for e in range(n)]))


def _train_policy(obs: np.ndarray, targets: np.ndarray, seed: int) -> Policy:
    """Behavioural-clone a raw reactive policy toward `targets` (for the control baselines)."""
    net = _MLP([3, 64, 1], np.random.default_rng(seed), 3e-3)
    rng = np.random.default_rng(seed + 1)
    tgt = np.atleast_2d(targets)
    for _ in range(CLONE_STEPS):
        idx = rng.integers(0, len(obs), 64)
        pred, cache = net.forward(obs[idx])
        net.zero_grad()
        net.backward(2.0 * (pred - tgt[idx]) / len(idx), cache)
        net.step()

    def act(o: Observation) -> Action:
        out, _ = net.forward(np.asarray(o.data, dtype=float).reshape(1, -1))
        return Action(data=np.clip(out.ravel(), -MAXT, MAXT))

    return act


def _seed_metrics(seed: int) -> dict[str, float]:
    o, a, o2, demo_ret = _demo(seed)
    floor = _return(lambda _o: Action(data=np.zeros(1)), seed)
    gtr = _broad_grounding(GROUND_N, seed)

    imitator = ObservationImitator(obs_dim=3, action_dim=1, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(GROUND_STEPS):
        idx = rng.integers(0, len(gtr), 64)
        imitator.ground([gtr[i] for i in idx])
    for _ in range(CLONE_STEPS):
        idx = rng.integers(0, len(o), 64)
        imitator.clone(o[idx], o2[idx])

    a_rec = np.atleast_2d(imitator.recover(o, o2))
    recovery_r2 = float(1.0 - np.mean((a_rec - a) ** 2) / np.var(a))
    imit_ret = _return(lambda ob: Action(data=np.clip(imitator.act(ob.data), -MAXT, MAXT)), seed)

    shuffled = np.random.default_rng(seed).permutation(len(a_rec))
    shuf_ret = _return(_train_policy(o, a_rec[shuffled], seed + 9), seed)
    g_o = np.array([t.state.z for t in gtr])
    g_a = np.array([t.action.data for t in gtr])
    clone_own_ret = _return(_train_policy(g_o, g_a, seed + 11), seed)  # clone your OWN random data

    span = demo_ret - floor if demo_ret - floor > 1e-6 else 1.0

    def score(x: float) -> float:
        return (x - floor) / span

    return {
        f"recovery_r2_s{seed}": recovery_r2,
        f"imitation_score_s{seed}": score(imit_ret),
        f"shuffled_score_s{seed}": score(shuf_ret),
        f"clone_own_score_s{seed}": score(clone_own_ret),
        f"demo_return_s{seed}": demo_ret,
        f"floor_return_s{seed}": floor,
    }


@gate_check("P14")
def check_p14() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    for seed in SEEDS:
        _log_sentinel_model(seed, log)
        metrics |= _seed_metrics(seed)

    def med(key: str) -> float:
        return float(np.median([metrics[f"{key}_s{s}"] for s in SEEDS]))

    r2 = med("recovery_r2")
    imit = med("imitation_score")
    shuf = med("shuffled_score")
    clone_own = med("clone_own_score")

    reproduces = imit >= REPRODUCE_MIN
    recovers = r2 >= RECOVERY_R2_MIN
    specific = shuf <= SHUFFLE_MAX
    watching = imit - clone_own >= WATCH_MARGIN
    passed = reproduces and recovers and specific and watching

    metrics |= {"recovery_r2_median": r2, "imitation_score_median": imit,
                "shuffled_score_median": shuf, "clone_own_score_median": clone_own,
                "reproduces_met": float(reproduces), "recovers_met": float(recovers),
                "specific_met": float(specific), "watching_met": float(watching)}
    detail = (
        f"observe→repeat (imitation from observation) — reproduces: imitation score {imit:.2f} "
        f"(>= {REPRODUCE_MIN}): {'MET' if reproduces else 'NOT MET'}. recovers actions from "
        f"observation: R² {r2:.3f} (>= {RECOVERY_R2_MIN}): {'MET' if recovers else 'NOT MET'}. "
        f"specific (shuffled control): {shuf:.2f} (<= {SHUFFLE_MAX}): {'MET' if specific else 'NOT MET'}. "
        f"watching matters: imitation {imit:.2f} − clone-own-random {clone_own:.2f} = {imit - clone_own:.2f} "
        f"(>= {WATCH_MARGIN}): {'MET' if watching else 'NOT MET'}. P14 {'PASS' if passed else 'BLOCKED'}"
    )
    return GateResult(phase="P14", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
