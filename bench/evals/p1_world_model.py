"""P1 eval (task P1-001): flat world model + calibrated uncertainty.

Capability (GATES["P1"]): latent 1-step prediction beats persistence and a linear
(ridge) baseline on held-out data, AND on a stochastic Pendulum variant epistemic
uncertainty falls with more data while aleatoric persists.

Sentinels (ADR-0006), fed by the run log this eval writes (`bench/runs/p1`):
- representation-integrity: latent per-dim std and effective rank above floors on
  held-out probes throughout training (after a short warm-up).
- uncertainty-reliability: ensemble disagreement rank-correlates with held-out
  error on a mixed in-distribution + out-of-distribution probe set, and stays high
  where error is high.

All thresholds are this eval's precise instantiation of the criteria in
`bench.gates`; seeds are explicit and recorded in the report (P0-006).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prospect.types import Action, LatentState, Transition
from prospect.world_model import FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, SentinelResult, gate_check, sentinel_check
from ..runlog import RUNS_DIR, Record, RunLog, latest_run, read_run

RUN_ID = "p1"
SEEDS = [0, 1, 2]
TRAIN_N, HELDOUT_N, OOD_N = 4096, 256, 256
STEPS, BATCH, PROBE_EVERY, SEED_STEP_OFFSET = 1500, 64, 100, 100_000
NOISE_STD, SIZES, STEPS_PER_SIZE = 0.3, (128, 1024, 8192), 1200
EPISTEMIC_FALL_MAX = 0.7  # epistemic(N_max)/epistemic(N_min) must fall below this
ALEATORIC_BAND = (0.5, 2.0)  # aleatoric ratio must stay inside (persists)
# Rank floor = the task's intrinsic dimension (pendulum: 2 DOF): collapse means
# falling below what the manifold honestly spans, not below an arbitrary bar.
# Warm-up excludes the representation-FORMATION phase (rank rises from init);
# the sentinel guards against collapse: rank falling after it has formed.
STD_FLOOR, RANK_FLOOR, WARMUP_STEPS = 0.3, 2.0, 300
CORR_MIN, HIGH_ERROR_RATIO_MIN = 0.3, 1.0

Probe = Callable[[FlatWorldModel], dict[str, float]]


def _rollout(env: Pendulum, n: int, seed: int) -> list[Transition]:
    """Random-policy transitions; raw observation vectors ride in `.state.z` (P0-011)."""
    rng = np.random.default_rng(seed)
    transitions: list[Transition] = []
    obs = env.reset(seed=seed * 7919 + 1)
    for i in range(n):
        if i % 200 == 0 and i > 0:
            obs = env.reset(seed=seed * 7919 + i)
        torque = float(rng.uniform(-env.max_torque, env.max_torque))
        action = Action(data=np.array([torque]))
        next_obs, reward, _ = env.step(action)
        transitions.append(
            Transition(state=LatentState(z=obs.data), action=action,
                       next_state=LatentState(z=next_obs.data), reward=reward)
        )
        obs = next_obs
    return transitions


def _train(
    model: FlatWorldModel,
    transitions: list[Transition],
    steps: int,
    rng: np.random.Generator,
    probe: Probe | None = None,
    log: RunLog | None = None,
    step_offset: int = 0,
) -> None:
    for step in range(steps):
        idx = rng.integers(0, len(transitions), size=BATCH)
        metrics = model.update([transitions[i] for i in idx])
        if probe is not None and log is not None and (step % PROBE_EVERY == 0 or step == steps - 1):
            log.log(step_offset + step, metrics | probe(model))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _per_sample_stats(
    model: FlatWorldModel, transitions: list[Transition]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(epistemic, aleatoric, sq_error, nll, z_squared) per transition."""
    epi, alea, err, nll, z_sq = [], [], [], [], []
    for t in transitions:
        pred = model.predict(model.encode(t.state.z), t.action)
        target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        mean = np.asarray(pred.mean, dtype=float)
        var = np.asarray(pred.var, dtype=float)
        epi.append(pred.epistemic)
        alea.append(pred.aleatoric)
        err.append(float(np.mean((mean - target) ** 2)))
        nll.append(-pred.log_prob(target))
        z_sq.append(float(np.mean((mean - target) ** 2 / var)))
    return np.array(epi), np.array(alea), np.array(err), np.array(nll), np.array(z_sq)


def _make_probe(heldout: list[Transition], mixed: list[Transition], seed: int) -> Probe:
    heldout_obs = np.stack([np.asarray(t.state.z, dtype=float) for t in heldout])

    def probe(model: FlatWorldModel) -> dict[str, float]:
        latents = np.stack([np.asarray(model.encode(o).z, dtype=float) for o in heldout_obs])
        std = latents.std(axis=0)
        eigenvalues = np.linalg.eigvalsh(np.cov(latents.T) + 1e-8 * np.eye(latents.shape[1]))
        effective_rank = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
        epi, _, err, nll, z_sq = _per_sample_stats(model, mixed)
        high_error = err >= np.quantile(err, 0.9)
        median_epi = float(np.median(epi))
        ratio = float(np.mean(epi[high_error]) / median_epi) if median_epi > 0 else 0.0
        return {
            "latent_std_min": float(std.min()),
            "latent_effective_rank": effective_rank,
            "disagreement_error_rank_corr": _spearman(epi, err),
            "high_error_disagreement_ratio": ratio,
            "heldout_nll": float(np.mean(nll)),
            "calibration_ratio": float(np.mean(z_sq)),
            "seed": float(seed),
        }

    return probe


def _baselines_and_model_mse(
    model: FlatWorldModel, train: list[Transition], heldout: list[Transition]
) -> dict[str, float]:
    """Held-out 1-step MSE in the (frozen) target-latent space: model vs baselines.

    Baselines are honest, not model-assisted: persistence copies the current latent;
    the linear baseline is a ridge regression from the RAW (obs, action) inputs to
    the target latent — same information as the model, restricted to a linear
    function class (it does not get the model's learned nonlinear encoder)."""

    def arrays(ts: list[Transition]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = np.stack([np.asarray(t.state.z, dtype=float) for t in ts])
        h = np.stack([np.asarray(model.encode_target(t.state.z).z, dtype=float) for t in ts])
        a = np.stack([np.asarray(t.action.data, dtype=float) for t in ts])
        hn = np.stack([np.asarray(model.encode_target(t.next_state.z).z, dtype=float) for t in ts])
        return obs, h, a, hn

    obs_tr, _, a_tr, hn_tr = arrays(train)
    obs_ho, h_ho, a_ho, hn_ho = arrays(heldout)
    persistence_mse = float(np.mean((h_ho - hn_ho) ** 2))
    x_tr = np.concatenate([obs_tr, a_tr, np.ones((len(obs_tr), 1))], axis=1)
    x_ho = np.concatenate([obs_ho, a_ho, np.ones((len(obs_ho), 1))], axis=1)
    ridge = np.linalg.solve(x_tr.T @ x_tr + 1e-3 * np.eye(x_tr.shape[1]), x_tr.T @ hn_tr)
    linear_mse = float(np.mean((x_ho @ ridge - hn_ho) ** 2))
    predicted = np.stack(
        [np.asarray(model.predict(model.encode(t.state.z), t.action).mean, dtype=float) for t in heldout]
    )
    model_mse = float(np.mean((predicted - hn_ho) ** 2))
    return {"model_mse": model_mse, "persistence_mse": persistence_mse, "linear_mse": linear_mse}


def _separation(seed: int) -> tuple[float, float]:
    """Train on the stochastic variant at growing dataset sizes; return the
    (epistemic, aleatoric) ratios of largest-to-smallest dataset."""
    heldout = _rollout(Pendulum(noise_std=NOISE_STD), HELDOUT_N, seed + 4000)
    epi_means, alea_means = [], []
    for size in SIZES:
        data = _rollout(Pendulum(noise_std=NOISE_STD), size, seed + size)
        model = FlatWorldModel(seed=seed + size)
        _train(model, data, STEPS_PER_SIZE, np.random.default_rng(seed + size + 1))
        epi, alea, _, _, _ = _per_sample_stats(model, heldout)
        epi_means.append(float(np.mean(epi)))
        alea_means.append(float(np.mean(alea)))
    return epi_means[-1] / epi_means[0], alea_means[-1] / alea_means[0]


@gate_check("P1")
def check_p1() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    model_mses, persistence_mses, linear_mses, epi_ratios, alea_ratios = [], [], [], [], []
    for seed in SEEDS:
        train = _rollout(Pendulum(), TRAIN_N, seed)
        heldout = _rollout(Pendulum(), HELDOUT_N, seed + 500)
        ood = _rollout(Pendulum(init_omega=14.0, omega_max=16.0), OOD_N, seed + 900)
        model = FlatWorldModel(seed=seed)
        _train(model, train, STEPS, np.random.default_rng(seed + 1),
               probe=_make_probe(heldout, heldout + ood, seed), log=log,
               step_offset=seed * SEED_STEP_OFFSET)
        mses = _baselines_and_model_mse(model, train, heldout)
        epi_ratio, alea_ratio = _separation(seed)
        model_mses.append(mses["model_mse"])
        persistence_mses.append(mses["persistence_mse"])
        linear_mses.append(mses["linear_mse"])
        epi_ratios.append(epi_ratio)
        alea_ratios.append(alea_ratio)
        metrics |= {f"{k}_s{seed}": v for k, v in mses.items()}
        metrics |= {f"epistemic_ratio_s{seed}": epi_ratio, f"aleatoric_ratio_s{seed}": alea_ratio}

    med = {
        "model_mse": float(np.median(model_mses)),
        "persistence_mse": float(np.median(persistence_mses)),
        "linear_mse": float(np.median(linear_mses)),
        "epistemic_ratio": float(np.median(epi_ratios)),
        "aleatoric_ratio": float(np.median(alea_ratios)),
    }
    metrics |= {f"{k}_median": v for k, v in med.items()}
    beats_baselines = med["model_mse"] < med["persistence_mse"] and med["model_mse"] < med["linear_mse"]
    separable = (
        med["epistemic_ratio"] < EPISTEMIC_FALL_MAX
        and ALEATORIC_BAND[0] <= med["aleatoric_ratio"] <= ALEATORIC_BAND[1]
    )
    detail = (
        f"median held-out latent MSE {med['model_mse']:.4f} vs persistence "
        f"{med['persistence_mse']:.4f} / linear {med['linear_mse']:.4f}; "
        f"epistemic ratio {med['epistemic_ratio']:.2f} (must be < {EPISTEMIC_FALL_MAX}), "
        f"aleatoric ratio {med['aleatoric_ratio']:.2f} (must stay in {ALEATORIC_BAND})"
    )
    return GateResult(phase="P1", passed=beats_baselines and separable,
                      metrics=metrics, seeds=list(SEEDS), detail=detail)


def _p1_records() -> list[Record]:
    return read_run(latest_run())


@sentinel_check("representation-integrity")
def check_representation_integrity() -> SentinelResult:
    name = "representation-integrity"
    try:
        records = _p1_records()
    except (FileNotFoundError, ValueError) as err:
        return SentinelResult(name=name, healthy=False, detail=f"no readable training run log: {err}")
    probes = [r for r in records
              if "latent_std_min" in r.metrics and r.step % SEED_STEP_OFFSET >= WARMUP_STEPS]
    if not probes:
        return SentinelResult(name=name, healthy=False, detail="run log has no post-warm-up latent probes")
    min_std = min(r.metrics["latent_std_min"] for r in probes)
    min_rank = min(r.metrics["latent_effective_rank"] for r in probes)
    healthy = min_std >= STD_FLOOR and min_rank >= RANK_FLOOR
    return SentinelResult(
        name=name, healthy=healthy,
        metrics={"min_latent_std": min_std, "min_effective_rank": min_rank},
        detail=(f"across {len(probes)} held-out probes (warm-up {WARMUP_STEPS} steps): "
                f"min per-dim std {min_std:.3f} (floor {STD_FLOOR}), "
                f"min effective rank {min_rank:.2f} (floor {RANK_FLOOR})"),
    )


@sentinel_check("uncertainty-reliability")
def check_uncertainty_reliability() -> SentinelResult:
    name = "uncertainty-reliability"
    try:
        records = _p1_records()
    except (FileNotFoundError, ValueError) as err:
        return SentinelResult(name=name, healthy=False, detail=f"no readable training run log: {err}")
    final_by_seed: dict[float, Record] = {}
    for r in records:
        if "disagreement_error_rank_corr" in r.metrics:
            final_by_seed[r.metrics["seed"]] = r  # records are in step order; keep last
    if not final_by_seed:
        return SentinelResult(name=name, healthy=False, detail="run log has no disagreement probes")
    corr = min(r.metrics["disagreement_error_rank_corr"] for r in final_by_seed.values())
    ratio = min(r.metrics["high_error_disagreement_ratio"] for r in final_by_seed.values())
    healthy = corr >= CORR_MIN and ratio >= HIGH_ERROR_RATIO_MIN
    return SentinelResult(
        name=name, healthy=healthy,
        metrics={"min_final_rank_corr": corr, "min_high_error_disagreement_ratio": ratio},
        detail=(f"worst seed at end of training: disagreement-vs-error rank corr {corr:.2f} "
                f"(min {CORR_MIN}), high-error-decile disagreement {ratio:.2f}x median "
                f"(min {HIGH_ERROR_RATIO_MIN}) on a mixed in-dist + OOD probe set"),
    )
