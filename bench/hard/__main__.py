"""`python -m bench.hard` (via `make bench-hard`) — run the non-gated harder-benchmark
probe (BH-001, ADR-0011) and write the consolidated report under `bench/hard/results/`.

Three parts, all on cartpole via the `bench.Environment` seam:
1. the **P2-claim probe** (MPC-over-a-learned-model vs a budget-matched model-free
   baseline) on balance + swingup, plus reachability rungs;
2. **A** — does the curiosity curriculum (P3-002) fix the swingup failure? (`curiosity.py`)
3. **B** — does imitation-from-observation reproduce swingup? (`imitation.py`)

Requires the optional `[bench-hard]` extra (dm_control + mujoco); it is never run by
the numpy-only core CI. Exit 0 on a completed run regardless of the numbers — this is
a probe, not a gate, so there is no pass/fail to signal.
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md

import numpy as np

from .curiosity import run_curiosity_study
from .eval import run_task
from .imitation import run_imitation
from .report import write_report

# Informative rungs (full multi-seed comparison): one task both learners saturate
# (sanity) and the harder swingup where the P2 claim is actually tested.
RUNGS = [("cartpole", "balance"), ("cartpole", "swingup")]
# Reachability rungs (1 seed): different domains / 2-D actions — the adapter handles
# them unchanged, but they score ~0 at this budget (below the probe's resolution).
REACH = [("reacher", "easy"), ("point_mass", "easy"), ("finger", "spin")]


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("dm_control", "mujoco", "numpy"):
        try:
            out[pkg] = _md.version(pkg)
        except _md.PackageNotFoundError:  # pragma: no cover
            out[pkg] = "?"
    return out


def main() -> int:
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    results = [run_task(domain, task, calibrate=(i == 0))
               for i, (domain, task) in enumerate(RUNGS)]
    reach = [run_task(domain, task, seeds=(0,), calibrate=False) for domain, task in REACH]
    # The swingup follow-up: A (does curiosity fix it?) then B (does watching a demo?).
    curiosity = run_curiosity_study("swingup")
    imitation = run_imitation()
    path = write_report(results, reach, stamp, _versions(), curiosity=curiosity, imitation=imitation)
    print(f"BH-001 report written -> {path}")
    for r in results:
        print(f"  {r.domain}-{r.task}: MBRL={r.med_mbrl:.1f}  "
              f"model-free={r.med_model_free:.1f}  random={r.med_random:.1f}  "
              f"(MBRL≥both: {r.mbrl_beats_baseline})")
    print(f"  A curiosity (swingup): MBRL random={np.median(curiosity.mbrl_random):.1f}  "
          f"curious={np.median(curiosity.mbrl_curious):.1f}")
    print(f"  B imitation (swingup): inverse-dyn={np.median(imitation.inverse_dyn):.1f}  "
          f"latent(P13)={np.median(imitation.latent_action):.1f}  "
          f"from-scratch={np.median(imitation.from_scratch):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
