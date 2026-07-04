"""Replay-fidelity sentinel (P3-003, ADR-0006): generative-replay collapse is
measured, not hoped for.

`log_replay_fidelity` (called per seed by the P3 gate eval) builds a ReplayBuffer
over the real training data and the trained model, regenerates rehearsal batches
several times, and logs per regeneration: the measured real-anchor fraction,
dreamed-sample diversity vs real (mean per-dim std of dreamed next-latents over
that of target-encoded real next-latents), the max lineage depth observed, and
whether any dream was stored. The zero-arg sentinel reads the run back.
"""
from __future__ import annotations

import numpy as np

from prospect.memory import ReplayBuffer
from prospect.types import Transition
from prospect.world_model import FlatWorldModel

from ..gates import SentinelResult, sentinel_check
from ..runlog import Record, RunLog, latest_run, read_run  # noqa: F401  (Record used in annotations)

REGENERATIONS, REPLAY_BATCH = 5, 128
REAL_FRACTION_FLOOR = 0.3
DIVERSITY_FLOOR = 0.3
SHRINK_FLOOR = 0.5  # last/first diversity across regenerations must stay above
DEPTH_CAP = 3


def log_replay_fidelity(
    model: FlatWorldModel, data: list[Transition], seed: int, log: RunLog, step_offset: int
) -> None:
    buffer = ReplayBuffer(model, max_dream_depth=DEPTH_CAP, seed=seed)
    for t in data:
        buffer.add(t)
    real_next = np.stack(
        [np.asarray(model.encode_target(t.next_state.z).z, dtype=float) for t in data[:256]]
    )
    real_spread = float(real_next.std(axis=0).mean())
    for regeneration in range(REGENERATIONS):
        before = len(buffer)
        batch = buffer.generative_replay(REPLAY_BATCH)
        dreams = [t for t in batch
                  if t.option is not None and t.option.name == ReplayBuffer.DREAM_SKILL]
        dream_next = np.stack([np.asarray(t.next_state.z, dtype=float) for t in dreams])
        log.log(step_offset + regeneration, {
            "replay_real_fraction": (len(batch) - len(dreams)) / len(batch),
            "replay_dream_diversity": float(dream_next.std(axis=0).mean()) / real_spread,
            "replay_max_depth": float(max(
                t.option.metadata["depth"] for t in dreams if t.option is not None
            )),
            "replay_dreams_stored": float(len(buffer) - before),
            "seed": float(seed),
        })


@sentinel_check("replay-fidelity")
def check_replay_fidelity() -> SentinelResult:
    name = "replay-fidelity"
    try:
        records = read_run(latest_run())
    except (FileNotFoundError, ValueError) as err:
        return SentinelResult(name=name, healthy=False,
                              detail=f"no readable training run log: {err}")
    per_seed: dict[float, list[dict[str, float]]] = {}
    for record in records:
        if "replay_real_fraction" in record.metrics:
            per_seed.setdefault(record.metrics["seed"], []).append(record.metrics)
    if not per_seed:
        return SentinelResult(name=name, healthy=False,
                              detail="run log has no replay-fidelity records — run the P3 gate")
    series = list(per_seed.values())
    min_fraction = min(m["replay_real_fraction"] for ms in series for m in ms)
    min_diversity = min(m["replay_dream_diversity"] for ms in series for m in ms)
    worst_shrink = min(
        ms[-1]["replay_dream_diversity"] / ms[0]["replay_dream_diversity"] for ms in series
    )
    max_depth = max(m["replay_max_depth"] for ms in series for m in ms)
    stored = max(m["replay_dreams_stored"] for ms in series for m in ms)
    healthy = (
        min_fraction >= REAL_FRACTION_FLOOR
        and min_diversity >= DIVERSITY_FLOOR
        and worst_shrink >= SHRINK_FLOOR
        and max_depth <= DEPTH_CAP
        and stored == 0.0
    )
    return SentinelResult(
        name=name, healthy=healthy,
        metrics={"min_real_fraction": min_fraction, "min_dream_diversity": min_diversity,
                 "worst_diversity_shrink": worst_shrink, "max_dream_depth": max_depth},
        detail=(f"across {REGENERATIONS} regenerations x {len(per_seed)} seed(s): "
                f"real fraction min {min_fraction:.2f} (floor {REAL_FRACTION_FLOOR}), "
                f"dream diversity min {min_diversity:.2f} (floor {DIVERSITY_FLOOR}), "
                f"diversity shrink {worst_shrink:.2f} (floor {SHRINK_FLOOR}), "
                f"lineage depth max {int(max_depth)} (cap {DEPTH_CAP}), "
                f"dreams stored: {int(stored)}"),
    )
