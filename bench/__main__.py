"""CLI: `python -m bench <phase>` (via `make gate PHASE=<phase>`) — run a kill-gate.

Prints the GateReport and exits 0 whether the phase passes or is BLOCKED (this is
an inspection tool; strict exit semantics arrive with `make gate-all`, P0-007).
Unknown/missing phase prints a friendly message and exits 2 — no traceback.
"""
from __future__ import annotations

import sys

from .gates import GATES, run_gate


def main(argv: list[str]) -> int:
    known = ", ".join(GATES)
    if len(argv) != 1:
        print(f"usage: python -m bench <phase>  (or: make gate PHASE=<phase>); known phases: {known}")
        return 2
    phase = argv[0]
    if phase not in GATES:
        print(f"unknown phase {phase!r}; known phases: {known}")
        return 2
    print(run_gate(phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
