"""Harness: task-specific evaluation, kept OUT of the core package (ADR-0005)."""
from __future__ import annotations

# Eval modules self-register their checks on import; they import bench.gates
# directly, so this import is position-independent (isort keeps it first).
from . import evals  # noqa: F401
from .envs import Environment  # noqa: F401
from .gates import (  # noqa: F401
    GATES,
    RESULTS_DIR,
    SENTINELS,
    SHIPPED_FILE,
    Gate,
    GateReport,
    GateResult,
    Sentinel,
    SentinelResult,
    applicable_sentinels,
    gate_check,
    run_gate,
    run_sentinels,
    run_shipped_gates,
    sentinel_check,
    shipped_phases,
)
from .runlog import Record, RunLog, latest_run, read_run  # noqa: F401
