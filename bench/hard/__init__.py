"""Optional harder-benchmark tier (BH-001, ADR-0011) — a **non-gated** credibility
probe that runs the composed agent on a real MuJoCo control task instead of the
toy environments the repo authored itself.

Deliberately fenced off from the numpy-only core CI:
- `dm_control` / `mujoco` are the optional `[bench-hard]` extra, never in `[dev,learn]`;
- this package is **never** imported by the gate registry (`bench.evals`), so
  `make gate-all` / `python -m bench` never touch it and the ratchet stays numpy-only;
- everything here skips cleanly when the extra is absent.

Run it on demand: `make bench-hard` (or `python -m bench.hard`). It writes a report
artifact under `bench/hard/results/` — a committed record, not a gate that can pass.
"""
from __future__ import annotations
