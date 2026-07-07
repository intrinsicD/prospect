"""`python -m bench.hard` (via `make bench-hard`) — run the non-gated harder-benchmark
probe (BH-001, ADR-0011) and write the report artifact under `bench/hard/results/`.

Requires the optional `[bench-hard]` extra (dm_control + mujoco); it is never run by
the numpy-only core CI. Exit 0 on a completed run regardless of the numbers — this is
a probe, not a gate, so there is no pass/fail to signal.
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md

from .eval import run_task
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
    path = write_report(results, reach, stamp, _versions())
    print(f"BH-001 report written -> {path}")
    for r in results:
        print(f"  {r.domain}-{r.task}: MBRL={r.med_mbrl:.1f}  "
              f"model-free={r.med_model_free:.1f}  random={r.med_random:.1f}  "
              f"(MBRL≥both: {r.mbrl_beats_baseline})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
