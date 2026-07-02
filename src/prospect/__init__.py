"""Prospect — a predictive-world-model agent research scaffold.

The core is task-unspecific. The one rule that binds it together: prediction error
(violation of expectation) is the single signal reused across the system, and the
world model speaks in distributions with an epistemic/aleatoric split, never bare
point estimates. See docs/architecture.md and docs/adr/0002.
"""
from __future__ import annotations

__version__ = "0.0.1"

from . import interfaces, types  # noqa: F401
