"""Harness: task-specific evaluation, kept OUT of the core package (ADR-0005)."""
from __future__ import annotations

from .envs import Environment  # noqa: F401
from .runlog import Record, RunLog, latest_run, read_run  # noqa: F401
from .gates import (  # noqa: F401
    GATES,
    RESULTS_DIR,
    SENTINELS,
    Gate,
    GateReport,
    GateResult,
    Sentinel,
    SentinelResult,
    SHIPPED_FILE,
    applicable_sentinels,
    gate_check,
    run_gate,
    run_sentinels,
    run_shipped_gates,
    sentinel_check,
    shipped_phases,
)

# Import last: eval modules self-register their checks against the registries above.
from . import evals  # noqa: E402, F401
