"""P13 eval: learn from passive, action-free observation (task P13-001, ADR-0010).

The agent watches a stream of observations with NO actions and NO rewards and learns a
predictive world model from it — via latent-action inference: an inverse model infers a
bottlenecked latent action between consecutive observations, a forward model predicts the
next observation from it, and a **decorrelation penalty** keeps the latent action
state-independent so it captures the *action* rather than a feature of the next state
(the identifiability fix, ADR-0010).

Setup (per seed): a demonstrator acts randomly in the pendulum; we record ONLY the
observation stream (actions/rewards hidden). A `LatentActionModel` learns from it. The
true (hidden) actions are used ONLY to score recovery — never for training.

Three criteria (median over seeds), plus all applicable collapse sentinels on run `p13`:
1. **Learns dynamics by watching** — the latent-action forward model reconstructs the next
   observation far better than a persistence baseline.
2. **Recovers the hidden actions** — the latent action decodes (linearly) to the true
   action with R² above a floor, and a shuffled-label control is ~0 (negative control).
3. **Watching transfers** — in the low-data regime, a model bootstrapped from action-free
   observation (fit a small true-action → latent-action map, frozen forward model) beats a
   from-scratch action-conditioned model at an equal, small labelled budget.
"""
from __future__ import annotations

import numpy as np

from prospect.observation import LatentActionModel
from prospect.types import Action
from prospect.world_model import _MLP, FlatWorldModel

from ..envs import Pendulum
from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe
from .p3_replay import log_replay_fidelity
from .p7_continual import _log_option_diversity
from .p8_knowledge import FULL, PROBE_N, REGION, TRAIN_N, _region_data

RUN_ID = "p13"
SEEDS = [0, 1, 2]
OBS_DIM, LA_DIM = 3, 1        # pendulum observation; latent-action bottleneck = true action dim
STREAM_N, HELD_N = 6000, 1000  # action-free training stream / held-out
OBS_STEPS, OBS_BATCH = 4000, 128
DECORR = 15.0                 # decorrelation strength (ADR-0010 identifiability)
RECON_MARGIN = 5.0            # recon MSE * this <= persistence MSE (learned dynamics by watching)
RECOVERY_R2_MIN = 0.5         # latent-action -> true-action recovery R^2 floor
RECOVERY_SHUFFLE_MAX = 0.1    # shuffled-label control R^2 ceiling (the negative control)
TRANSFER_N = 40              # small action-labelled budget for the transfer test
TRANSFER_STEPS = 1500
TRANSFER_MARGIN = 1.5        # watch-first MSE * this <= from-scratch MSE at the budget


def _stream(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An action-free observation stream: a demonstrator acts randomly, we keep only the
    observations (the true actions are returned separately, used ONLY to score recovery)."""
    env = Pendulum()
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    o, a, o2 = [], [], []
    for i in range(n):
        if i % 60 == 0:
            obs = env.reset(seed=seed * 13 + i)
        action = float(rng.uniform(-2.0, 2.0))
        nxt, _, _ = env.step(Action(data=np.array([action])))
        o.append(obs.data)
        a.append([action])
        o2.append(nxt.data)
        obs = nxt
    return np.array(o, dtype=float), np.array(a, dtype=float), np.array(o2, dtype=float)


def _recovery(model: LatentActionModel, o: np.ndarray, a: np.ndarray, o2: np.ndarray,
              oh: np.ndarray, ah: np.ndarray, o2h: np.ndarray) -> tuple[float, float]:
    """R^2 of a linear decoder latent-action -> true-action on held-out, and the same for a
    shuffled-label control (the gate-overfit negative control)."""
    z_tr = np.atleast_2d(model.infer_action(o, o2))
    z_h = np.atleast_2d(model.infer_action(oh, o2h))
    x_tr, x_h = np.c_[z_tr, np.ones(len(z_tr))], np.c_[z_h, np.ones(len(z_h))]

    def r2(targets: np.ndarray) -> float:
        w, *_ = np.linalg.lstsq(x_tr, targets, rcond=None)
        pred = x_h @ w
        return float(1.0 - np.mean((pred - ah) ** 2) / np.var(ah))

    return r2(a), r2(np.random.default_rng(0).permutation(a))


def _transfer(model: LatentActionModel, o: np.ndarray, a: np.ndarray, o2: np.ndarray,
              oh: np.ndarray, ah: np.ndarray, o2h: np.ndarray, seed: int) -> tuple[float, float]:
    """Low-data transfer: watch-first (fit a small true-action -> latent-action map on N
    labels, frozen forward model) vs from-scratch (an action-conditioned model on the same N
    labels). Returns (watch_mse, scratch_mse) on held-out 1-step prediction."""
    ol, al, o2l = o[:TRANSFER_N], a[:TRANSFER_N], o2[:TRANSFER_N]
    z_l = np.atleast_2d(model.infer_action(ol, o2l))
    decoder = _MLP([1, 16, LA_DIM], np.random.default_rng(seed + 5), 5e-3)  # true action -> latent action
    rng = np.random.default_rng(seed + 6)
    for _ in range(TRANSFER_STEPS):
        idx = rng.integers(0, TRANSFER_N, min(64, TRANSFER_N))
        pred, cache = decoder.forward(al[idx])
        decoder.zero_grad()
        decoder.backward(2.0 * (pred - z_l[idx]) / len(idx), cache)
        decoder.step()
    z_h, _ = decoder.forward(ah)
    watch = float(np.mean((np.atleast_2d(model.predict(oh, z_h)) - o2h) ** 2))

    scratch = _MLP([OBS_DIM + 1, 64, OBS_DIM], np.random.default_rng(seed + 8), 3e-3)
    rng = np.random.default_rng(seed + 9)
    for _ in range(TRANSFER_STEPS):
        idx = rng.integers(0, TRANSFER_N, min(64, TRANSFER_N))
        pred, cache = scratch.forward(np.concatenate([ol[idx], al[idx]], axis=1))
        scratch.zero_grad()
        scratch.backward(2.0 * (pred - o2l[idx]) / len(idx), cache)
        scratch.step()
    pred_s, _ = scratch.forward(np.concatenate([oh, ah], axis=1))
    return watch, float(np.mean((pred_s - o2h) ** 2))


def _log_sentinel_model(seed: int, log: RunLog) -> None:
    """Train the phase's pendulum integrity model and log the four data-sentinels to run
    `p13` (the standard fixture)."""
    train = _region_data(REGION, TRAIN_N, seed)
    heldout = _region_data(REGION, PROBE_N, seed + 500)
    ood = _region_data(FULL, PROBE_N, seed + 900)
    model = FlatWorldModel(seed=seed)
    rng = np.random.default_rng(seed + 1)
    probe = _make_probe(heldout, heldout + ood, seed)
    for step in range(STEPS):
        idx = rng.integers(0, len(train), size=64)
        step_metrics = model.update([train[i] for i in idx])
        if step % 100 == 0:
            log.log(seed * SEED_STEP_OFFSET + step, step_metrics | probe(model))
    log_replay_fidelity(model, train, seed, log, step_offset=seed * SEED_STEP_OFFSET + 50_000)
    _log_option_diversity(model, seed, log)


def _seed_metrics(seed: int) -> dict[str, float]:
    o, a, o2 = _stream(STREAM_N, seed)
    oh, ah, o2h = _stream(HELD_N, seed + 999)
    model = LatentActionModel(obs_dim=OBS_DIM, latent_action_dim=LA_DIM, decorrelation=DECORR, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(OBS_STEPS):
        idx = rng.integers(0, len(o), size=OBS_BATCH)
        model.observe(o[idx], o2[idx])

    # 1. learns dynamics by watching: reconstruct next-obs (from the inferred latent action)
    z_h = model.infer_action(oh, o2h)
    recon = float(np.mean((np.atleast_2d(model.predict(oh, z_h)) - o2h) ** 2))
    persist = float(np.mean((oh - o2h) ** 2))
    # 2. recovers the hidden actions (+ shuffled control)
    r2, r2_shuffled = _recovery(model, o, a, o2, oh, ah, o2h)
    # 3. watching transfers (low-data regime)
    watch, scratch = _transfer(model, o, a, o2, oh, ah, o2h, seed)
    return {
        f"recon_mse_s{seed}": recon, f"persist_mse_s{seed}": persist,
        f"recovery_r2_s{seed}": r2, f"recovery_shuffled_r2_s{seed}": r2_shuffled,
        f"transfer_watch_mse_s{seed}": watch, f"transfer_scratch_mse_s{seed}": scratch,
    }


@gate_check("P13")
def check_p13() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    for seed in SEEDS:
        _log_sentinel_model(seed, log)
        metrics |= _seed_metrics(seed)

    def med(key: str) -> float:
        return float(np.median([metrics[f"{key}_s{s}"] for s in SEEDS]))

    recon, persist = med("recon_mse"), med("persist_mse")
    r2, r2_shuffled = med("recovery_r2"), med("recovery_shuffled_r2")
    watch, scratch = med("transfer_watch_mse"), med("transfer_scratch_mse")

    learns = recon * RECON_MARGIN <= persist
    recovers = r2 >= RECOVERY_R2_MIN and r2_shuffled <= RECOVERY_SHUFFLE_MAX
    transfers = watch * TRANSFER_MARGIN <= scratch
    passed = learns and recovers and transfers

    metrics |= {"recon_mse_median": recon, "persist_mse_median": persist,
                "recovery_r2_median": r2, "recovery_shuffled_r2_median": r2_shuffled,
                "transfer_watch_mse_median": watch, "transfer_scratch_mse_median": scratch,
                "learns_met": float(learns), "recovers_met": float(recovers),
                "transfers_met": float(transfers)}
    detail = (
        f"learn from passive observation — dynamics: recon {recon:.4f} vs persistence {persist:.4f} "
        f"(>= x{RECON_MARGIN}): {'MET' if learns else 'NOT MET'}. recovers hidden actions: R² {r2:.3f} "
        f"(>= {RECOVERY_R2_MIN}, shuffled {r2_shuffled:+.3f} <= {RECOVERY_SHUFFLE_MAX}): "
        f"{'MET' if recovers else 'NOT MET'}. transfers (N={TRANSFER_N} labels): watch-first {watch:.4f} "
        f"vs from-scratch {scratch:.4f} (>= x{TRANSFER_MARGIN}): {'MET' if transfers else 'NOT MET'}. "
        f"P13 {'PASS' if passed else 'BLOCKED'}"
    )
    return GateResult(phase="P13", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
