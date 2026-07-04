"""P5 eval (P5-001 + P5-002): the jumpy option-model and two-level planning.

P5-001's record — one learned jump bounds compounding error (ADR-0001's named
limiter): on held-out REAL option executions, jumpy landing MSE beats the P4
router's flat one-step-composed rollout on every seed.

P5-002's capability — two-level planning beats flat at EQUAL compute: the
manager searches option sequences over the jumpy model and re-plans when an
option ends (by horizon or a VoE spike, threshold calibrated to the held-out
q99 one-step surprise). Compute is counted in ensemble member-forward passes
per environment step, VoE monitoring charged to the hierarchy; the flat CEM
arm's candidate count is derived at runtime to match the hierarchy's measured
budget, and a full-compute flat reference is reported for context (not gated).
Data budgets are reported too: the jumpy model's option executions are the
hierarchy's one-time abstraction cost (ADR-0003 trades data for cheap,
long-horizon decisions).

The `option-diversity` sentinel (ADR-0006) reads this eval's run log: per seed,
normalized option-usage entropy, mean EXECUTED option duration, and the minimum
pairwise d' between REAL landing distributions of the options actually used.

Run `p5` carries the P1 probes and replay-fidelity records so all active
sentinels judge this phase's model.
"""
from __future__ import annotations

import numpy as np

from prospect.agent import Agent
from prospect.planning import FlatPlanner, HierarchicalManager, JumpyOptionModel
from prospect.skills import SkillRouter
from prospect.types import LatentState, Option, Transition
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, SentinelResult, gate_check, sentinel_check
from ..loop import run_episode
from ..runlog import RUNS_DIR, RunLog, latest_run, read_run
from .p1_world_model import SEED_STEP_OFFSET, _make_probe, _rollout, _train
from .p3_replay import log_replay_fidelity
from .p4_skills import TRAIN_N, TRAIN_STEPS, _collect, _constant_policy, _env

RUN_ID = "p5"
SEEDS = [0, 1, 2]
EXECUTIONS, HELDOUT_EXECUTIONS = 768, 128
JUMPY_STEPS, JUMPY_BATCH = 800, 64
DISCOUNT = 0.99
EP_LEN, EVAL_EPISODES, MANAGER_DEPTH = 100, 10, 3
OPTION_HORIZON = 3  # finer than P4's: bang-bang control near the target needs cadence
MEMBERS = 5  # ensemble size, the unit of compute accounting
ENTROPY_FLOOR, DURATION_FLOOR, DPRIME_FLOOR = 0.3, 1.0, 0.5


def _skills() -> list[Option]:
    """P5's option set: coarse AND fine torques at a short horizon. The fine
    (+-0.5) skills exist because the option set is the hierarchy's control
    resolution: with bang-bang +-2 only, the manager cannot hold the target as
    tightly as a continuous-action flat planner, and easy near-target episodes
    are lost on that ceiling, not on planning."""
    options = [_constant_policy(t) for t in (-2.0, -0.5, 0.0, 0.5, 2.0)]
    for option in options:
        option.horizon = OPTION_HORIZON
    return options


def _execute_option(theta: float, omega: float, option: Option) -> tuple[np.ndarray, float, int]:
    """Really run the option; returns (landing obs, cumulative discounted reward,
    duration in primitive steps)."""
    env = _env()
    env.reset(seed=0)
    obs = env.set_state(theta, omega)
    assert option.policy is not None
    total, discount = 0.0, 1.0
    for _ in range(option.horizon):
        obs, reward, _ = env.step(option.policy(LatentState(z=obs.data)))
        total += discount * reward
        discount *= DISCOUNT
    return np.asarray(obs.data, dtype=float), total, option.horizon


def _option_transitions(
    model: FlatWorldModel, skills: list[Option], n: int, seed: int
) -> list[Transition]:
    """Latent-space option-transitions from real executions (the P5-001 training
    convention: E(start) in, target-encoded landing out, cumulative reward)."""
    rng = np.random.default_rng(seed)
    transitions = []
    for _ in range(n):
        theta = float(rng.uniform(-np.pi, np.pi))
        omega = float(rng.uniform(-6.0, 6.0))
        option = skills[int(rng.integers(len(skills)))]
        start_obs = np.asarray(_env().set_state(theta, omega).data, dtype=float)
        landing_obs, cumulative, _ = _execute_option(theta, omega, option)
        transitions.append(
            Transition(state=model.encode(start_obs), action=option.policy(model.encode(start_obs)),  # type: ignore[misc]
                       next_state=model.encode_target(landing_obs), reward=cumulative,
                       option=option)
        )
    return transitions


def _surprise_threshold(model: FlatWorldModel, skills: list[Option], seed: int) -> float:
    """q99 of one-step surprise measured DURING option execution — the reference
    distribution VoE termination actually sees (random-walk held-out data is the
    wrong distribution: controlled trajectories surprise differently, and a
    threshold calibrated there cut healthy options at ~2 of 5 steps)."""
    rng = np.random.default_rng(seed + 33)
    surprises = []
    for _ in range(30):
        env = _env()
        env.reset(seed=0)
        obs = env.set_state(float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-6.0, 6.0)))
        option = skills[int(rng.integers(len(skills)))]
        assert option.policy is not None
        for _ in range(option.horizon):
            latent = model.encode(obs.data)
            expected = model.predict(latent, option.policy(latent))
            obs, _, _ = env.step(option.policy(latent))
            surprises.append(-expected.log_prob(model.encode_target(obs.data).z))
    return float(np.quantile(surprises, 0.99))


def _hierarchical_episode(
    model: FlatWorldModel, manager: HierarchicalManager, env_seed: int
) -> tuple[float, dict[str, int], list[int], dict[str, list[np.ndarray]], int, int]:
    """One two-level episode: manager plans options over the jumpy model, the
    worker executes, VoE ends options early. Returns (return, usage, executed
    durations, real landings per option, member-forward cost, terminations)."""
    env = _env()
    obs = env.reset(seed=env_seed)
    plan_cost = (len(manager._options) ** manager.depth) * manager.depth * MEMBERS
    active: Option | None = None
    steps_in, total, cost, terminations = 0, 0.0, 0, 0
    usage: dict[str, int] = {}
    durations: list[int] = []
    landings: dict[str, list[np.ndarray]] = {}
    for _ in range(EP_LEN):
        latent = model.encode(obs.data)
        if active is None:
            active = manager.plan_option(latent)
            usage[active.name] = usage.get(active.name, 0) + 1
            steps_in = 0
            cost += plan_cost
        assert active.policy is not None
        action = active.policy(latent)
        expected = model.predict(latent, action)
        cost += MEMBERS  # VoE monitoring, charged to the hierarchy
        obs, reward, _ = env.step(action)
        total += reward
        steps_in += 1
        ended = manager.should_terminate(
            Transition(state=latent, action=action,
                       next_state=model.encode_target(obs.data), reward=reward,
                       prediction=expected)
        )
        terminations += int(ended)
        if ended or steps_in >= active.horizon:
            durations.append(steps_in)
            landings.setdefault(active.name, []).append(
                np.asarray(model.encode_target(obs.data).z, dtype=float))
            active = None
    if active is not None:
        durations.append(steps_in)
    return total, usage, durations, landings, cost, terminations


def _flat_return(model: FlatWorldModel, planner: FlatPlanner, seed: int) -> float:
    agent = Agent(encode=lambda o: model.encode(o.data), planner=planner)
    return float(np.mean([
        run_episode(_env(), agent, EP_LEN, 9000 + seed * 50 + e)[0]
        for e in range(EVAL_EPISODES)
    ]))


def _diversity(usage: dict[str, int], durations: list[int],
               landings: dict[str, list[np.ndarray]]) -> tuple[float, float, float]:
    counts = np.array(list(usage.values()), dtype=float)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(_skills())))
    mean_duration = float(np.mean(durations))
    names = [name for name, ls in landings.items() if len(ls) >= 2]
    dprimes = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            la, lb = np.stack(landings[a]), np.stack(landings[b])
            pooled = float(np.sqrt((la.var(axis=0) + lb.var(axis=0)) / 2.0).mean()) + 1e-6
            dprimes.append(float(np.linalg.norm(la.mean(axis=0) - lb.mean(axis=0)) / pooled))
    return entropy, mean_duration, (float(min(dprimes)) if dprimes else 0.0)


@gate_check("P5")
def check_p5() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    jumpy_mses, flat_mses = [], []
    hier_wins: list[bool] = []
    hier_returns_all: list[float] = []
    matched_returns_all: list[float] = []
    for seed in SEEDS:
        train = _collect(TRAIN_N, seed)
        heldout = _collect(256, seed + 500)
        ood = _rollout(_env(init_omega=14.0, omega_max=16.0), 256, seed + 900)
        model = FlatWorldModel(seed=seed)
        _train(model, train, TRAIN_STEPS, np.random.default_rng(seed + 1),
               probe=_make_probe(heldout, heldout + ood, seed), log=log,
               step_offset=seed * SEED_STEP_OFFSET)
        log_replay_fidelity(model, train, seed, log,
                            step_offset=seed * SEED_STEP_OFFSET + 50_000)

        skills = _skills()
        jumpy = JumpyOptionModel([s.name for s in skills], latent_dim=model.latent_dim,
                                 seed=seed)
        train_jumps = _option_transitions(model, skills, EXECUTIONS, seed + 1300)
        rng = np.random.default_rng(seed + 5)
        for _ in range(JUMPY_STEPS):
            idx = rng.integers(0, len(train_jumps), size=JUMPY_BATCH)
            jumpy.update([train_jumps[i] for i in idx])

        router = SkillRouter(model)  # the flat-rollout baseline the jumpy model replaces
        for skill in skills:
            router.add(skill)
        heldout_jumps = _option_transitions(model, skills, HELDOUT_EXECUTIONS, seed + 4700)
        jumpy_errors, flat_errors = [], []
        for t in heldout_jumps:
            assert t.option is not None
            target = np.asarray(t.next_state.z, dtype=float)
            jumpy_landing = np.asarray(jumpy.predict_option(t.state, t.option).mean, dtype=float)
            flat_landing = np.asarray(router.simulate(t.state, t.option)[0].mean, dtype=float)
            jumpy_errors.append(float(np.mean((jumpy_landing - target) ** 2)))
            flat_errors.append(float(np.mean((flat_landing - target) ** 2)))
        jumpy_mse, flat_mse = float(np.mean(jumpy_errors)), float(np.mean(flat_errors))
        jumpy_mses.append(jumpy_mse)
        flat_mses.append(flat_mse)
        metrics |= {f"jumpy_landing_mse_s{seed}": jumpy_mse,
                    f"flat_rollout_mse_s{seed}": flat_mse}

        # ---- P5-002: two-level vs compute-matched flat planning ----
        manager = HierarchicalManager(jumpy, skills, depth=MANAGER_DEPTH,
                                      uncertainty_penalty=1.0,
                                      surprise_threshold=_surprise_threshold(model, skills, seed))
        usage_all: dict[str, int] = {}
        durations_all: list[int] = []
        landings_all: dict[str, list[np.ndarray]] = {}
        hier_returns, total_cost, total_terminations = [], 0, 0
        for episode in range(EVAL_EPISODES):
            ret, usage, durations, landings, cost, terminations = _hierarchical_episode(
                model, manager, 9000 + seed * 50 + episode)
            hier_returns.append(ret)
            total_cost += cost
            total_terminations += terminations
            durations_all.extend(durations)
            for name, count in usage.items():
                usage_all[name] = usage_all.get(name, 0) + count
            for name, points in landings.items():
                landings_all.setdefault(name, []).extend(points)
        hier_return = float(np.mean(hier_returns))
        hier_cost_per_step = total_cost / (EVAL_EPISODES * EP_LEN)
        matched_horizon = 5
        matched_candidates = max(2, round(hier_cost_per_step / (matched_horizon * MEMBERS)))
        matched = FlatPlanner(model, horizon=matched_horizon, candidates=matched_candidates,
                              elites=2, iterations=1, seed=seed)
        matched_return = _flat_return(model, matched, seed)
        matched_cost_per_step = matched_candidates * matched_horizon * MEMBERS
        reference = FlatPlanner(model, horizon=12, candidates=32, elites=6, iterations=2,
                                seed=seed)
        reference_return = _flat_return(model, reference, seed)
        hier_wins.append(hier_return > matched_return)
        entropy, mean_duration, min_dprime = _diversity(usage_all, durations_all, landings_all)
        log.log(seed * SEED_STEP_OFFSET + 60_000, {
            "option_usage_entropy": entropy,
            "option_mean_duration": mean_duration,
            "option_min_pairwise_dprime": min_dprime,
            "option_terminations": float(total_terminations),
            "seed": float(seed),
        })
        metrics |= {
            f"hierarchical_return_s{seed}": hier_return,
            f"matched_flat_return_s{seed}": matched_return,
            f"reference_flat_return_s{seed}": reference_return,
            f"hier_cost_per_step_s{seed}": float(hier_cost_per_step),
            f"matched_flat_cost_per_step_s{seed}": float(matched_cost_per_step),
            f"voe_terminations_s{seed}": float(total_terminations),
            f"option_usage_entropy_s{seed}": entropy,
            f"option_mean_duration_s{seed}": mean_duration,
            f"option_min_dprime_s{seed}": min_dprime,
        }
        hier_returns_all.append(hier_return)
        matched_returns_all.append(matched_return)

    jumpy_beats_flat = all(j < f for j, f in zip(jumpy_mses, flat_mses, strict=True))
    two_level_wins = all(hier_wins)
    metrics |= {"jumpy_landing_mse_median": float(np.median(jumpy_mses)),
                "flat_rollout_mse_median": float(np.median(flat_mses)),
                "jumpy_beats_flat": float(jumpy_beats_flat),
                "jumpy_data_cost_env_steps": float(EXECUTIONS * _skills()[0].horizon),
                "two_level_beats_matched_flat": float(two_level_wins)}
    detail = (
        f"two-level return per seed {[round(r, 1) for r in hier_returns_all]} vs "
        f"compute-matched flat {[round(r, 1) for r in matched_returns_all]} "
        f"(~{metrics[f'hier_cost_per_step_s{SEEDS[0]}']:.0f} member-forwards/step) — "
        f"{'WINS on every seed' if two_level_wins else 'does NOT win on every seed'}; "
        f"jumpy landing MSE beats composed flat rollout on every seed: "
        f"{'YES' if jumpy_beats_flat else 'NO'}"
    )
    return GateResult(phase="P5", passed=two_level_wins, metrics=metrics,
                      seeds=list(SEEDS), detail=detail)


@sentinel_check("option-diversity")
def check_option_diversity() -> SentinelResult:
    name = "option-diversity"
    try:
        records = read_run(latest_run())
    except (FileNotFoundError, ValueError) as err:
        return SentinelResult(name=name, healthy=False,
                              detail=f"no readable training run log: {err}")
    rows = [r.metrics for r in records if "option_usage_entropy" in r.metrics]
    if not rows:
        return SentinelResult(name=name, healthy=False,
                              detail="run log has no option-diversity records — run the P5 gate")
    entropy = min(m["option_usage_entropy"] for m in rows)
    duration = min(m["option_mean_duration"] for m in rows)
    dprime = min(m["option_min_pairwise_dprime"] for m in rows)
    healthy = entropy >= ENTROPY_FLOOR and duration > DURATION_FLOOR and dprime >= DPRIME_FLOOR
    return SentinelResult(
        name=name, healthy=healthy,
        metrics={"min_usage_entropy": entropy, "min_mean_duration": duration,
                 "min_pairwise_dprime": dprime},
        detail=(f"worst seed over the two-level eval episodes: usage entropy "
                f"{entropy:.2f} (floor {ENTROPY_FLOOR}), mean executed duration "
                f"{duration:.2f} steps (floor > {DURATION_FLOOR}), min pairwise "
                f"landing d' {dprime:.2f} (floor {DPRIME_FLOOR})"),
    )
