"""P12 eval: swappable visual perception — the first omni-modal seam (ADR-0009).

The agent sees through a FROZEN encoder: a frame → embedding → codec → shared latent, and
the world model predicts over what it sees and is surprised when wrong. And vision is a
*swappable* module: a second, different frozen encoder distils into the incumbent latent
without retraining the core (P0-011).

Backend note (Path B): a *real* pretrained encoder only makes sense on real image content,
which a numpy CI gate has none of. So the encoders here are deterministic **stand-in**
frozen modules (fixed random-feature projections) — which is exactly what "swappable"
means: a real DINOv2/CLIP encoder swaps in via the identical distill path, generating its
embeddings offline under the `[vision]` extra (the local real-video regen). The gate proves
the *mechanism* (see → predict → swap → surprise); the stand-in is pure numpy, so the gate
is deterministic and needs no backend.

Content: a blob orbits the image centre under a FIXED global flow (rotation by DTHETA),
so the next frame is determined by the current one (single-frame prediction is well-posed);
clips differ by orbit radius/phase. "Novel" frames are a structurally different scene (two
blobs) whose embeddings sit outside the training distribution — the surprise probe.

Three criteria (median over seeds), plus all applicable collapse sentinels on run `p12`:
1. **Sees and predicts** — next-visual-latent MSE beats a persistence baseline.
2. **Swappable** — a second frozen encoder distils into the incumbent latent and preserves
   the core-loop 1-step MSE within tolerance (a better vision module drops in for free).
3. **Surprise is calibrated** — epistemic VoE higher on novel frames than familiar ones.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from prospect.codec import UniversalCodec
from prospect.types import Action, LatentState, Modality, Observation, Transition
from prospect.world_model import FlatWorldModel

from ..gates import GateResult, gate_check
from ..runlog import RUNS_DIR, RunLog
from .p1_world_model import SEED_STEP_OFFSET, STEPS, _make_probe
from .p3_replay import log_replay_fidelity
from .p7_continual import _log_option_diversity
from .p8_knowledge import FULL, PROBE_N, REGION, TRAIN_N, _region_data

RUN_ID = "p12"
SEEDS = [0, 1, 2]
G = 12                      # render grid (G x G); a frame is a flattened G*G vector
FRAME_DIM = G * G
EMB_DIM, ENC_HIDDEN = 32, 64  # frozen stand-in encoder: frame -> embedding
CLIP_LEN = 40
N_TRAIN_CLIPS, N_HELD_CLIPS, N_NOVEL_CLIPS = 48, 12, 12
DTHETA = 0.25              # fixed global rotation per step (single-frame prediction is well-posed)
BLOB_SIGMA = 0.08
VISION_STEPS, VISION_BATCH = 1500, 64
DISTILL_STEPS, DISTILL_BATCH = 600, 128
PREDICT_MARGIN = 1.2       # wm_mse * this <= persistence_mse (it learned visual dynamics)
SWAP_TOL = 1.5             # swapped-encoder core-loop MSE <= this x incumbent
SURPRISE_MARGIN = 1.5      # novel-frame epistemic >= this x familiar-frame epistemic

_GX, _GY = np.meshgrid(np.linspace(0.0, 1.0, G), np.linspace(0.0, 1.0, G))
_NULL = Action(data=np.array([0.0]))


def _render(x: float, y: float) -> np.ndarray:
    """A Gaussian blob at (x, y) on the G x G grid, flattened."""
    return np.exp(-((_GX - x) ** 2 + (_GY - y) ** 2) / (2 * BLOB_SIGMA**2)).ravel()


def _clip(seed: int) -> np.ndarray:
    """A blob orbiting the centre under the fixed global flow (rotation by DTHETA)."""
    rng = np.random.default_rng(seed)
    r, phi = float(rng.uniform(0.15, 0.4)), float(rng.uniform(0.0, 2 * np.pi))
    return np.stack([_render(0.5 + r * np.cos(phi + t * DTHETA), 0.5 + r * np.sin(phi + t * DTHETA))
                     for t in range(CLIP_LEN)])


def _novel_clip(seed: int) -> np.ndarray:
    """A structurally novel scene — TWO blobs — whose embeddings sit outside the training
    distribution (the surprise probe)."""
    rng = np.random.default_rng(seed)
    r1, p1 = float(rng.uniform(0.15, 0.4)), float(rng.uniform(0.0, 2 * np.pi))
    r2, p2 = float(rng.uniform(0.15, 0.4)), float(rng.uniform(0.0, 2 * np.pi))
    out = []
    for t in range(CLIP_LEN):
        a = _render(0.5 + r1 * np.cos(p1 + t * DTHETA), 0.5 + r1 * np.sin(p1 + t * DTHETA))
        b = _render(0.5 + r2 * np.cos(p2 - t * DTHETA), 0.5 + r2 * np.sin(p2 - t * DTHETA))
        out.append(np.maximum(a, b))
    return np.stack(out)


def _encoder(seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """A FROZEN stand-in vision encoder: a fixed random-feature projection frame ->
    embedding. Stands in for a real pretrained encoder (which swaps in via the same distill
    path); pure numpy, so the gate is deterministic and backend-free."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 1.0 / np.sqrt(FRAME_DIM), (FRAME_DIM, ENC_HIDDEN))
    b1 = rng.normal(0.0, 0.1, ENC_HIDDEN)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(ENC_HIDDEN), (ENC_HIDDEN, EMB_DIM))

    def encode(frames: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(frames, dtype=float) @ w1 + b1) @ w2

    return encode


def _transitions(clips: list[np.ndarray], encode: Callable[[np.ndarray], np.ndarray]) -> list[Transition]:
    """Embedding-space transitions (emb_t, null, emb_{t+1}) — autonomous visual dynamics."""
    out: list[Transition] = []
    for clip in clips:
        emb = encode(clip)
        for t in range(len(emb) - 1):
            out.append(Transition(state=LatentState(z=emb[t]), action=_NULL,
                                  next_state=LatentState(z=emb[t + 1]), reward=0.0))
    return out


def _mean_epistemic(clips: list[np.ndarray], encode: Callable[[np.ndarray], np.ndarray],
                    model: FlatWorldModel) -> float:
    vals = [model.predict(model.encode(e), _NULL).epistemic
            for clip in clips for e in encode(clip)]
    return float(np.mean(vals))


def _log_sentinel_model(seed: int, log: RunLog) -> None:
    """Train the phase's pendulum integrity model and log the four data-sentinels to run
    `p12` (the standard fixture — representation/uncertainty/replay/option-diversity are
    pendulum-coupled; the vision capability is measured separately)."""
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


def _vision_metrics(seed: int) -> dict[str, float]:
    enc_a, enc_b = _encoder(seed * 7 + 1), _encoder(seed * 7 + 2)  # two frozen encoders
    train_clips = [_clip(seed * 100 + i) for i in range(N_TRAIN_CLIPS)]
    held_clips = [_clip(seed * 100 + 500 + i) for i in range(N_HELD_CLIPS)]
    novel_clips = [_novel_clip(seed * 100 + 900 + i) for i in range(N_NOVEL_CLIPS)]

    model = FlatWorldModel(obs_dim=EMB_DIM, action_dim=1, seed=seed)
    train_tr = _transitions(train_clips, enc_a)
    rng = np.random.default_rng(seed + 1)
    for _ in range(VISION_STEPS):
        idx = rng.integers(0, len(train_tr), size=VISION_BATCH)
        model.update([train_tr[i] for i in idx])

    # 1. sees and predicts: beat persistence on held-out next-visual-latent MSE
    wm_err, persist_err = [], []
    for tr in _transitions(held_clips, enc_a):
        target = np.asarray(model.encode_target(tr.next_state.z).z, dtype=float)
        pred = np.asarray(model.predict(model.encode(tr.state.z), _NULL).mean, dtype=float)
        persist = np.asarray(model.encode_target(tr.state.z).z, dtype=float)  # predict "no change"
        wm_err.append(float(np.mean((pred - target) ** 2)))
        persist_err.append(float(np.mean((persist - target) ** 2)))

    # 2. swappable: distil BOTH encoders' embeddings into the incumbent latent (P0-011),
    # then swap encoder B in via its codec and check the frozen core loop is preserved.
    all_frames = np.concatenate(train_clips)
    emb_a, emb_b = enc_a(all_frames), enc_b(all_frames)
    incumbent = np.stack([np.asarray(model.encode(e).z, dtype=float) for e in emb_a])
    codec_b = UniversalCodec({Modality.VISION: EMB_DIM}, latent_dim=model.latent_dim, seed=seed + 1)
    drng = np.random.default_rng(seed + 303)
    for _ in range(DISTILL_STEPS):
        idx = drng.integers(0, len(all_frames), size=DISTILL_BATCH)
        codec_b.distill_encode(emb_b[idx], Modality.VISION, incumbent[idx])
    inc_err, swap_err = [], []
    for clip in held_clips:
        fa, fb = enc_a(clip), enc_b(clip)
        for t in range(len(clip) - 1):
            target = np.asarray(model.encode_target(fa[t + 1]).z, dtype=float)
            z_inc = model.encode(fa[t])                                    # incumbent encoder
            z_swap = codec_b.encode(Observation(Modality.VISION, fb[t]))   # encoder B, swapped in
            inc_err.append(float(np.mean((np.asarray(model.predict(z_inc, _NULL).mean, dtype=float) - target) ** 2)))
            swap_err.append(float(np.mean((np.asarray(model.predict(z_swap, _NULL).mean, dtype=float) - target) ** 2)))

    return {
        f"wm_mse_s{seed}": float(np.mean(wm_err)),
        f"persist_mse_s{seed}": float(np.mean(persist_err)),
        f"inc_mse_s{seed}": float(np.mean(inc_err)),
        f"swap_mse_s{seed}": float(np.mean(swap_err)),
        f"indist_epi_s{seed}": _mean_epistemic(held_clips, enc_a, model),
        f"novel_epi_s{seed}": _mean_epistemic(novel_clips, enc_a, model),
    }


@gate_check("P12")
def check_p12() -> GateResult:
    (RUNS_DIR / RUN_ID / "metrics.jsonl").unlink(missing_ok=True)
    log = RunLog(RUN_ID)
    metrics: dict[str, float] = {}
    for seed in SEEDS:
        _log_sentinel_model(seed, log)
        metrics |= _vision_metrics(seed)

    def med(key: str) -> float:
        return float(np.median([metrics[f"{key}_s{s}"] for s in SEEDS]))

    wm, persist = med("wm_mse"), med("persist_mse")
    inc, swap = med("inc_mse"), med("swap_mse")
    indist_epi, novel_epi = med("indist_epi"), med("novel_epi")

    sees = wm * PREDICT_MARGIN <= persist
    swappable = swap <= inc * SWAP_TOL
    surprise = novel_epi >= indist_epi * SURPRISE_MARGIN
    passed = sees and swappable and surprise

    metrics |= {"wm_mse_median": wm, "persist_mse_median": persist, "inc_mse_median": inc,
                "swap_mse_median": swap, "indist_epi_median": indist_epi,
                "novel_epi_median": novel_epi, "sees_met": float(sees),
                "swappable_met": float(swappable), "surprise_met": float(surprise)}
    detail = (
        f"swappable visual perception — sees & predicts: wm {wm:.4f} vs persistence {persist:.4f} "
        f"(>= x{PREDICT_MARGIN}): {'MET' if sees else 'NOT MET'}. swappable: encoder-B via codec "
        f"{swap:.4f} vs incumbent {inc:.4f} (<= x{SWAP_TOL}): {'MET' if swappable else 'NOT MET'}. "
        f"surprise: novel-frame epistemic {novel_epi:.4f} vs familiar {indist_epi:.4f} "
        f"(>= x{SURPRISE_MARGIN}): {'MET' if surprise else 'NOT MET'}. P12 "
        f"{'PASS' if passed else 'BLOCKED'} (stand-in encoders; real vision swaps in via the "
        f"same distill path — ADR-0009)"
    )
    return GateResult(phase="P12", passed=passed, metrics=metrics, seeds=list(SEEDS), detail=detail)
