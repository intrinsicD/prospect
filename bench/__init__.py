"""Harness: task-specific evaluation, kept OUT of the core package (ADR-0005)."""
from __future__ import annotations

from .gates import (  # noqa: F401
    FLOORS,
    GATES,
    SENTINELS,
    Floor,
    FloorResult,
    Gate,
    GateReport,
    GateResult,
    Sentinel,
    SentinelResult,
    applicable_floors,
    applicable_sentinels,
    run_floors,
    run_gate,
    run_sentinels,
)
