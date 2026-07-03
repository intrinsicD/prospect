"""Harness: task-specific evaluation, kept OUT of the core package (ADR-0005)."""
from __future__ import annotations

from .envs import Environment  # noqa: F401
from .gates import (  # noqa: F401
    GATES,
    SENTINELS,
    Gate,
    GateReport,
    GateResult,
    Sentinel,
    SentinelResult,
    applicable_sentinels,
    run_gate,
    run_sentinels,
)
