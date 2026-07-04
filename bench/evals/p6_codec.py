"""P6 eval (task P6-001): any-to-any codec via distillation into the incumbent
latent, and the swap that preserves the core loop.

The migration is the point (P0-011): the dynamics model is NOT retrained. The
UniversalCodec's `encode` is distilled to reproduce the frozen incumbent encoder
on the shared STATE modality, so swapping it in preserves 1-step prediction
(planning is then preserved by construction — same latent, same plan). A second
modality (IMAGE, a rasterized sensor view of the same situation) is distilled to
the SAME latent, so the frozen core loop predicts from an image exactly as from a
state vector — any-to-any, measured.

Pass: codec-swapped held-out 1-step MSE stays within TOL x the incumbent MSE for
BOTH modalities on every seed. Cross-modality latent agreement and STATE decode
reconstruction are reported.

Run `p6` carries all four applicable sentinels' records (one model per seed feeds
the P1 probes, replay fidelity, a hierarchy rollout for option-diversity, and the
codec swap) — every active integrity check judges this phase's model.
"""
from __future__ import annotations

import numpy as np

from prospect.codec import UniversalCodec
from prospect.planning import HierarchicalManager, JumpyOptionModel
from prospect.types import Modality, Observation, Transition
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, _make_probe, _rollout, _train
from .p3_replay import log_replay_fidelity
from .p4_skills import TRAIN_N, TRAIN_STEPS, _collect, _env
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
    _surprise_threshold,
)

RUN_ID = "p6"
SEEDS = [0, 1, 2]
IMAGE_ANGULAR_BINS, IMAGE_VELOCITY_BINS = 8, 4
IMAGE_DIM = IMAGE_ANGULAR_BINS + IMAGE_VELOCITY_BINS
DISTILL_N, DISTILL_STEPS, DISTILL_BATCH = 3072, 800, 128
HELDOUT_N = 256
# The option-diversity measurement reuses P5's episode budget — the same options
# measured d'=2.12 there; fewer episodes just undersample the landing distributions.
HIER_EPISODES = EVAL_EPISODES
TOL = 1.5  # codec-swapped MSE must stay within this factor of the incumbent MSE


def _image_obs(theta: float, omega: float) -> np.ndarray:
    """A rasterized sensor view of the same situation — a genuinely different
    modality (an 'image'): angular Gaussian bins around the circle + velocity
    Gaussian bins. Deterministic function of (theta, omega)."""
    centers = np.linspace(-np.pi, np.pi, IMAGE_ANGULAR_BINS, endpoint=False)
    ang_dist = (centers - theta + np.pi) % (2.0 * np.pi) - np.pi
    angular = np.exp(-(ang_dist**2) / 0.5)
    vel_centers = np.linspace(-8.0, 8.0, IMAGE_VELOCITY_BINS)
    velocity = np.exp(-((vel_centers - omega) ** 2) / 8.0)
    return np.concatenate([angular, velocity])


def _theta_omega(state_obs: np.ndarray) -> tuple[float, float]:
    cos_theta, sin_theta, omega = np.asarray(state_obs, dtype=float)
    return float(np.arctan2(sin_theta, cos_theta)), float(omega)


def _distill_codec(model: FlatWorldModel, seed: int) -> UniversalCodec:
    """Distil a UniversalCodec into the incumbent latent: both STATE and IMAGE map
    to `model.encode` on wide, paired coverage of the state space (P0-011)."""
    codec = UniversalCodec({Modality.STATE: 3, Modality.IMAGE: IMAGE_DIM},
                           latent_dim=model.latent_dim, seed=seed)
    rng = np.random.default_rng(seed + 202)
    states = np.stack([np.asarray(_env().set_state(
        float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-8.0, 8.0))).data, dtype=float)
        for _ in range(DISTILL_N)])
    images = np.stack([_image_obs(*_theta_omega(s)) for s in states])
    targets = np.stack([np.asarray(model.encode(s).z, dtype=float) for s in states])
    fit_rng = np.random.default_rng(seed + 303)
    for _ in range(DISTILL_STEPS):
        idx = fit_rng.integers(0, DISTILL_N, size=DISTILL_BATCH)
        codec.distill_encode(states[idx], Modality.STATE, targets[idx])
        codec.distill_encode(images[idx], Modality.IMAGE, targets[idx])
        codec.fit_decode(targets[idx], states[idx], Modality.STATE)
    return codec


def _swap_mses(
    model: FlatWorldModel, codec: UniversalCodec, heldout: list[Transition]
) -> tuple[float, float, float, float, float]:
    """Held-out 1-step MSE with the incumbent encoder vs the codec-swapped encoder
    (STATE and IMAGE), plus cross-modality latent MSE and STATE reconstruction."""
    inc, state_swap, image_swap, cross, recon = [], [], [], [], []
    for t in heldout:
        state_obs = np.asarray(t.state.z, dtype=float)
        image = _image_obs(*_theta_omega(state_obs))
        target = np.asarray(model.encode_target(t.next_state.z).z, dtype=float)
        z_inc = model.encode(state_obs)
        z_state = codec.encode(Observation(Modality.STATE, state_obs))
        z_image = codec.encode(Observation(Modality.IMAGE, image))
        inc.append(float(np.mean((model.predict(z_inc, t.action).mean - target) ** 2)))
        state_swap.append(float(np.mean((model.predict(z_state, t.action).mean - target) ** 2)))
        image_swap.append(float(np.mean((model.predict(z_image, t.action).mean - target) ** 2)))
        cross.append(float(np.mean((np.asarray(z_state.z) - np.asarray(z_image.z)) ** 2)))
        recon_obs = np.asarray(codec.decode(z_inc, Modality.STATE).data, dtype=float)
        mean, std = codec._stats[Modality.STATE]
        recon.append(float(np.mean((recon_obs * std + mean - state_obs) ** 2)))
    return (float(np.mean(inc)), float(np.mean(state_swap)), float(np.mean(image_swap)),
            float(np.mean(cross)), float(np.mean(recon)))


def _log_option_diversity(model: FlatWorldModel, seed: int, log: RunLog) -> None:
    """Populate run `p6` with the option-diversity records the P5-era sentinel
    reads (the codec swap is orthogonal to the hierarchy; it still runs)."""
    skills = _skills()
    jumpy = JumpyOptionModel([s.name for s in skills], latent_dim=model.latent_dim, seed=seed)
    train_jumps = _option_transitions(model, skills, EXECUTIONS, seed + 1300)
    rng = np.random.default_rng(seed + 5)
    for _ in range(JUMPY_STEPS):
        idx = rng.integers(0, len(train_jumps), size=JUMPY_BATCH)
        jumpy.update([train_jumps[i] for i in idx])
    manager = HierarchicalManager(jumpy, skills, depth=MANAGER_DEPTH, uncertainty_penalty=1.0,
                                  surprise_threshold=_surprise_threshold(model, skills, seed))
    usage_all: dict[str, int] = {}
    durations_all: list[int] = []
    landings_all: dict[str, list[np.ndarray]] = {}
    for episode in range(HIER_EPISODES):
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


@gate_check("P6")
def check_p6() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    state_ok, image_ok = [], []
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
        _log_option_diversity(model, seed, log)

        codec = _distill_codec(model, seed)
        probe = _collect(HELDOUT_N, seed + 800)
        inc, state_swap, image_swap, cross, recon = _swap_mses(model, codec, probe)
        state_ok.append(state_swap <= TOL * inc)
        image_ok.append(image_swap <= TOL * inc)
        metrics |= {
            f"incumbent_mse_s{seed}": inc,
            f"state_swap_mse_s{seed}": state_swap,
            f"image_swap_mse_s{seed}": image_swap,
            f"state_swap_ratio_s{seed}": state_swap / inc,
            f"image_swap_ratio_s{seed}": image_swap / inc,
            f"cross_modality_latent_mse_s{seed}": cross,
            f"state_reconstruction_mse_s{seed}": recon,
        }

    passed = all(state_ok) and all(image_ok)
    state_ratios = [metrics[f"state_swap_ratio_s{s}"] for s in SEEDS]
    image_ratios = [metrics[f"image_swap_ratio_s{s}"] for s in SEEDS]
    metrics |= {"state_swap_ratio_max": float(max(state_ratios)),
                "image_swap_ratio_max": float(max(image_ratios)),
                "codec_preserves_core_loop": float(passed)}
    detail = (
        f"codec-swapped 1-step MSE / incumbent — STATE {[round(r, 2) for r in state_ratios]}, "
        f"IMAGE {[round(r, 2) for r in image_ratios]} (tolerance x{TOL}); "
        f"the swap preserves the core loop and a second modality drives it "
        f"identically on every seed: {'YES' if passed else 'NO'}"
    )
    return GateResult(phase="P6", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
