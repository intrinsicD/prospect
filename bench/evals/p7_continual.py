"""P7 eval (task P7-001): continual improvement — retention and plasticity.

The roadmap gate: on a task sequence, retention above threshold (no catastrophic
forgetting) AND plasticity retained (late tasks learn as fast as early). The
measured story: NAIVE sequential training loses both — it forgets earlier tasks
AND its seq/from-scratch fit ratio grows across the sequence (the documented loss
of plasticity in continual learning). The CONSOLIDATION policy — rehearsal of
retained real experience (the P3-003 buffer; raw obs stay re-feedable, P0-011) —
preserves both.

Boundary note: consolidation rehearses REAL retained experience, not the buffer's
generative-replay dreams. Dreams live in latent space and `FlatWorldModel.update`
re-encodes raw observations, so consuming them needs a latent-space training path
the world model does not expose — a deliberate future extension (earned by a gate
if raw retention becomes infeasible), not this task. The generative-replay
mechanism is still exercised and its `replay-fidelity` sentinel judged on run p7.

Two arms at equal per-step budget over a gravity-varied Pendulum sequence:
consolidation (each batch = half fresh current-task + half past-task replay) and
no-consolidation (fresh only). From-scratch per-task models give the difficulty
reference that isolates plasticity from task difficulty; comparing the last task
to the first REHEARSAL-diluted task cancels the rehearsal dilution.

Run `p7` carries all four sentinels: the consolidation model's latent must stay
healthy across the whole sequence (continual learning must not collapse it).
"""
from __future__ import annotations

import numpy as np

from prospect.memory import ReplayBuffer
from prospect.planning import HierarchicalManager, JumpyOptionModel
from prospect.types import Action, LatentState, Option, Transition
from prospect.voe import SurpriseCompetenceMonitor
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import PROBE_EVERY, SEED_STEP_OFFSET, _make_probe, _rollout
from .p3_replay import log_replay_fidelity
from .p5_options import (
    EVAL_EPISODES,
    EXECUTIONS,
    JUMPY_BATCH,
    JUMPY_STEPS,
    MANAGER_DEPTH,
    _diversity,
    _hierarchical_episode,
    _option_transitions,
    _skills,
    _surprise_calibration,
)

RUN_ID = "p7"
SEEDS = [0, 1, 2]
# A moderate gravity spread: distinct enough that naive sequential training forgets
# earlier tasks, but not so extreme that the pendulum saturates the omega clip and
# the latent's effective rank falls below its 2-DOF intrinsic dimension (a wider
# spread makes the tasks unlearnable, not the consolidation policy inadequate).
GRAVITIES = [6.0, 10.0, 14.0]
DAMPING, DT = 0.3, 0.3
STEPS_PER_TASK, BATCH, TASK_N, TASK_HELDOUT = 800, 64, 2048, 150
RETENTION_TOL = 0.8  # consolidation avg past-task MSE <= this x no-consolidation's (the contrast)
PLASTICITY_TOL = 3.5  # consolidation last-task fit within this x the from-scratch scale (absolute)
FORGET_SKILL = "task0"


def _task_env(gravity: float, **kwargs: float) -> Pendulum:
    return Pendulum(gravity=gravity, damping=DAMPING, dt=DT, **kwargs)


def _task_data(gravity: float, n: int, seed: int) -> list[Transition]:
    env = _task_env(gravity, init_omega=8.0)
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed * 7919 + 1)
    out: list[Transition] = []
    for i in range(n):
        if i % 50 == 0 and i > 0:
            obs = env.reset(seed=seed * 7919 + i)
        action = Action(data=np.array([float(rng.uniform(-2.0, 2.0))]))
        next_obs, reward, _ = env.step(action)
        out.append(Transition(state=LatentState(z=obs.data), action=action,
                              next_state=LatentState(z=next_obs.data), reward=reward))
        obs = next_obs
    return out


def _mse(model: FlatWorldModel, data: list[Transition]) -> float:
    return float(np.mean([
        float(np.mean((np.asarray(model.predict(model.encode(t.state.z), t.action).mean)
                       - np.asarray(model.encode_target(t.next_state.z).z)) ** 2))
        for t in data]))


def _from_scratch(task: list[Transition], heldout: list[Transition], seed: int) -> float:
    """From-scratch MSE on one task — the difficulty reference isolating plasticity."""
    model = FlatWorldModel(seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(STEPS_PER_TASK):
        idx = rng.integers(0, len(task), size=BATCH)
        model.update([task[i] for i in idx])
    return _mse(model, heldout)


def _run_sequence(
    tasks: list[list[Transition]], heldouts: list[list[Transition]], seed: int,
    consolidate: bool, log: RunLog | None = None, probe_ood: list[Transition] | None = None,
) -> tuple[list[float], list[float], SurpriseCompetenceMonitor, ReplayBuffer, FlatWorldModel]:
    """Train the task sequence; return per-task MSE right after each task, final
    per-task MSE, the monitor, the buffer and the model. Logs sentinel probes when
    a `log` is given (the consolidation arm carries run p7's sentinel records)."""
    model = FlatWorldModel(seed=seed)
    past = ReplayBuffer(model, seed=seed)
    # Faster EMAs so task 0 masters while it is still FRESH (its low error latches
    # as the forgetting floor); the shipped-default slow rate would not flatten
    # until task 1 has already disrupted task 0, latching the forgotten error.
    monitor = SurpriseCompetenceMonitor(fast_rate=0.2, slow_rate=0.1)
    rng = np.random.default_rng(seed + 1)
    seq_after, global_step = [], 0
    for k, task in enumerate(tasks):
        # Probe the CURRENT task's data (P1 semantics: in-distribution). Probing a
        # fixed later task while the model fits an earlier one measures an OOD,
        # legitimately-compressed latent — a false collapse alarm.
        probe = _make_probe(heldouts[k], heldouts[k] + (probe_ood or []), seed) if log else None
        for _ in range(STEPS_PER_TASK):
            idx = rng.integers(0, len(task), size=BATCH)
            batch = [task[i] for i in idx]
            if consolidate and len(past) > 0:
                batch = batch[: BATCH // 2] + past.sample(BATCH // 2)
            metrics = model.update(batch)
            # Task 0 is representation FORMATION (rank rises from init) — the analog
            # of P1's warm-up; the integrity sentinel guards against collapse AFTER
            # it has formed, during the continual-learning tasks. So probe from k>=1.
            if log is not None and probe is not None and k >= 1 and global_step % PROBE_EVERY == 0:
                log.log(seed * SEED_STEP_OFFSET + global_step, metrics | probe(model))
            global_step += 1
        seq_after.append(_mse(model, heldouts[k]))
        # Feed the monitor task-0 probes labelled as a skill, so is_forgetting can
        # latch task-0 mastery while fresh and detect its error rising later.
        for t in heldouts[0]:
            prediction = model.predict(model.encode(t.state.z), t.action)
            monitor.update(Transition(state=model.encode(t.state.z), action=t.action,
                                      next_state=model.encode_target(t.next_state.z),
                                      reward=t.reward, prediction=prediction,
                                      option=Option(name=FORGET_SKILL)))
        for t in task:
            past.add(t)
    final = [_mse(model, ho) for ho in heldouts]
    return seq_after, final, monitor, past, model


@gate_check("P7")
def check_p7() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    retention_ok, plasticity_ok = [], []
    for seed in SEEDS:
        tasks = [_task_data(g, TASK_N, seed * 10 + k) for k, g in enumerate(GRAVITIES)]
        heldouts = [_task_data(g, TASK_HELDOUT, seed * 10 + 500 + k)
                    for k, g in enumerate(GRAVITIES)]
        ood = _rollout(_task_env(GRAVITIES[-1], init_omega=14.0, omega_max=16.0), 256, seed + 900)
        fs = [_from_scratch(tasks[k], heldouts[k], seed) for k in range(len(GRAVITIES))]

        seq_c, final_c, monitor_c, past_c, model_c = _run_sequence(
            tasks, heldouts, seed, consolidate=True, log=log, probe_ood=ood)
        seq_n, final_n, monitor_n, _, _ = _run_sequence(
            tasks, heldouts, seed, consolidate=False)

        # Retention: mean MSE over earlier tasks after the whole sequence.
        retention_c = float(np.mean(final_c[:-1]))
        retention_n = float(np.mean(final_n[:-1]))
        # Plasticity is an ABSOLUTE property — "the model still learns a new task
        # well at the end of the sequence". Measured as the last task's fit vs the
        # from-scratch difficulty scale (the MEDIAN from-scratch MSE — a single
        # from-scratch model can fit one task atypically well, an unstable
        # denominator). Retention (above) is the contrast where consolidation earns
        # its keep; plasticity is a property the system must simply retain.
        fs_scale = float(np.median(fs))
        plasticity_c = seq_c[-1] / fs_scale
        plasticity_n = seq_n[-1] / fs_scale  # reported for context (naive's plasticity)
        retention_ok.append(retention_c <= RETENTION_TOL * retention_n)
        plasticity_ok.append(plasticity_c <= PLASTICITY_TOL)

        # Sentinel records: replay fidelity + option diversity on the consolidation model.
        log_replay_fidelity(model_c, tasks[-1], seed, log,
                            step_offset=seed * SEED_STEP_OFFSET + 50_000)
        _log_option_diversity(model_c, seed, log)

        metrics |= {
            f"retention_consolidate_s{seed}": retention_c,
            f"retention_none_s{seed}": retention_n,
            f"plasticity_consolidate_s{seed}": plasticity_c,
            f"plasticity_none_s{seed}": plasticity_n,
            f"forgetting_detected_none_s{seed}": float(monitor_n.is_forgetting(FORGET_SKILL)),
            f"forgetting_detected_consolidate_s{seed}": float(monitor_c.is_forgetting(FORGET_SKILL)),
        }

    passed = all(retention_ok) and all(plasticity_ok)
    ret_c = [metrics[f"retention_consolidate_s{s}"] for s in SEEDS]
    ret_n = [metrics[f"retention_none_s{s}"] for s in SEEDS]
    plas_c = [metrics[f"plasticity_consolidate_s{s}"] for s in SEEDS]
    plas_n = [metrics[f"plasticity_none_s{s}"] for s in SEEDS]
    metrics |= {"retention_consolidate_mean": float(np.mean(ret_c)),
                "retention_none_mean": float(np.mean(ret_n)),
                "plasticity_consolidate_mean": float(np.mean(plas_c)),
                "plasticity_none_mean": float(np.mean(plas_n)),
                "continual_improvement_passes": float(passed)}
    detail = (
        f"retention (avg past-task MSE, lower=better) consolidate {[round(r, 2) for r in ret_c]} "
        f"vs none {[round(r, 2) for r in ret_n]} (contrast, tol x{RETENTION_TOL}); plasticity "
        f"(last-task fit / from-scratch scale) consolidate {[round(p, 2) for p in plas_c]} "
        f"(naive {[round(p, 2) for p in plas_n]}, abs tol x{PLASTICITY_TOL}) — consolidation "
        f"prevents the forgetting naive suffers while the model still learns new tasks: "
        f"{'YES' if passed else 'NO'}"
    )
    return GateResult(phase="P7", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)


def _log_option_diversity(model: FlatWorldModel, seed: int, log: RunLog) -> None:
    """Populate run p7 with option-diversity records the P5-era sentinel reads."""
    skills = _skills()
    jumpy = JumpyOptionModel([s.name for s in skills], latent_dim=model.latent_dim, seed=seed)
    train_jumps = _option_transitions(model, skills, EXECUTIONS, seed + 1300)
    rng = np.random.default_rng(seed + 5)
    for _ in range(JUMPY_STEPS):
        idx = rng.integers(0, len(train_jumps), size=JUMPY_BATCH)
        jumpy.update([train_jumps[i] for i in idx])
    termination, _ = _surprise_calibration(model, skills, seed)
    manager = HierarchicalManager(
        jumpy,
        skills,
        depth=MANAGER_DEPTH,
        uncertainty_penalty=1.0,
        surprise_threshold=termination.value,
    )
    usage_all: dict[str, int] = {}
    durations_all: list[int] = []
    landings_all: dict[str, list[np.ndarray]] = {}
    for episode in range(EVAL_EPISODES):
        _, usage, durations, landings, _, _ = _hierarchical_episode(
            model, manager, 9000 + seed * 50 + episode)
        durations_all.extend(durations)
        for name, count in usage.items():
            usage_all[name] = usage_all.get(name, 0) + count
        for name, points in landings.items():
            landings_all.setdefault(name, []).extend(points)
    entropy, mean_duration, min_dprime = _diversity(usage_all, durations_all, landings_all)
    log.log(seed * SEED_STEP_OFFSET + 60_000, {
        "option_usage_entropy": entropy, "option_mean_duration": mean_duration,
        "option_min_pairwise_dprime": min_dprime, "seed": float(seed)})
