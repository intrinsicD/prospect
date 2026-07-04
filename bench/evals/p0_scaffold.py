"""P0 capability check: the scaffold is healthy iff imports are clean and the smoke
test suite passes (the roadmap's P0 criterion).

Runs pytest in a subprocess with PROSPECT_IN_P0_GATE=1 so the suite's own
P0-gate test skips itself (one level of nesting, no recursion). Deterministic —
no seeds to record.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..gates import GateResult, gate_check

_REPO_ROOT = Path(__file__).resolve().parents[2]


@gate_check("P0")
def check_p0() -> GateResult:
    env = dict(os.environ, PROSPECT_IN_P0_GATE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        check=False,
    )
    out = (proc.stdout or proc.stderr).strip()
    detail = out.splitlines()[-1] if out else "no pytest output"
    return GateResult(
        phase="P0",
        passed=proc.returncode == 0,
        metrics={"pytest_exit_code": float(proc.returncode)},
        detail=detail,
    )
