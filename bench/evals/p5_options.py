"""P5 eval — partial, from P5-001: the jumpy option-model's own claim.

ADR-0003's reason for the jumpy model to exist is that one learned jump bounds
the compounding error of composing one-step predictions (ADR-0001's named
limiter). Measured here: on held-out REAL option executions (the P4 reference
task and constant-torque skills), the jumpy landing prediction must beat the P4
router's flat one-step-composed rollout — landing MSE in the target-latent
space, lower on every seed.

The full P5 capability (two-level planning beats flat at equal compute) and the
`option-diversity` sentinel are P5-002's; until then this check reports
`passed=False` with the pending half named. Run `p5` carries the P1 probes and
replay-fidelity records so the active sentinels judge this phase's model.
"""
from __future__ import annotations

import numpy as np

from prospect.planning import JumpyOptionModel
from prospect.skills import SkillRouter
from prospect.types import LatentState, Option, Transition
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, _make_probe, _rollout, _train
from .p3_replay import log_replay_fidelity
from .p4_skills import TRAIN_N, TRAIN_STEPS, _collect, _env, _skills

RUN_ID = "p5"
SEEDS = [0, 1, 2]
EXECUTIONS, HELDOUT_EXECUTIONS = 512, 128
JUMPY_STEPS, JUMPY_BATCH = 800, 64
DISCOUNT = 0.99


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


@gate_check("P5")
def check_p5() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    jumpy_mses, flat_mses = [], []
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

    jumpy_beats_flat = all(j < f for j, f in zip(jumpy_mses, flat_mses, strict=True))
    metrics |= {"jumpy_landing_mse_median": float(np.median(jumpy_mses)),
                "flat_rollout_mse_median": float(np.median(flat_mses)),
                "jumpy_beats_flat": float(jumpy_beats_flat)}
    detail = (
        f"jumpy landing MSE per seed {[round(m, 4) for m in jumpy_mses]} vs flat rollout "
        f"{[round(m, 4) for m in flat_mses]} — jumpy beats flat on every seed: "
        f"{'YES' if jumpy_beats_flat else 'NO'}; hierarchical manager pending (P5-002)"
    )
    return GateResult(phase="P5", passed=False, metrics=metrics, seeds=list(SEEDS), detail=detail)
