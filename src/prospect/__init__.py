"""Prospect — an auditable adaptive-agent research runtime."""

from __future__ import annotations

__version__ = "0.0.1"

from . import decision, domain, epistemics, runtime, storage
from .decision import CounterIdentitySource, MaxValuePolicy
from .runtime import AgentState, EpistemicAgent

__all__ = (
    "AgentState",
    "CounterIdentitySource",
    "EpistemicAgent",
    "MaxValuePolicy",
    "decision",
    "domain",
    "epistemics",
    "runtime",
    "storage",
)
