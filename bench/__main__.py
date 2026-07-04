"""CLI: `python -m bench <phase>` (via `make gate PHASE=<phase>`) — run a kill-gate;
`python -m bench --all` (via `make gate-all`) — the regression ratchet (P0-007).

Single-phase mode is an inspection tool: it prints the GateReport and exits 0
whether the phase passes or is BLOCKED. Ratchet mode is strict: exit 0 only if
every shipped gate (bench/SHIPPED) still passes, 1 on any regression, 2 on usage
errors or a malformed SHIPPED file. No tracebacks either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .gates import GATES, run_gate, run_shipped_gates


def main(
    argv: list[str],
    shipped_path: Path | None = None,
    results_dir: Path | None = None,
) -> int:
    known = ", ".join(GATES)
    if argv == ["--all"]:
        try:
            reports = run_shipped_gates(shipped_path, results_dir=results_dir)
        except ValueError as err:
            print(err)
            return 2
        if not reports:
            print("no shipped phases (bench/SHIPPED is empty) — nothing to ratchet")
            return 0
        for report in reports:
            print(report)
        blocked = [r.phase for r in reports if not r.passed]
        if blocked:
            print(f"RATCHET FAILED — shipped gate(s) regressed: {', '.join(blocked)}")
            return 1
        print(f"ratchet ok — {len(reports)} shipped gate(s) still green")
        return 0
    if len(argv) != 1:
        print(
            f"usage: python -m bench <phase> | --all  "
            f"(or: make gate PHASE=<phase> / make gate-all); known phases: {known}"
        )
        return 2
    phase = argv[0]
    if phase not in GATES:
        print(f"unknown phase {phase!r}; known phases: {known}")
        return 2
    print(run_gate(phase, results_dir=results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
