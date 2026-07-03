"""The harness's environment seam (P0-004). Environments are task-specific, so they
live in the harness (golden rule 3): gate evals and training loops target this one
contract instead of re-inventing wiring per phase.

Import direction is one-way: the harness may import core types (`prospect.types`);
the core must never import `bench` (guarded by a test).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from prospect.types import Action, Observation


@runtime_checkable
class Environment(Protocol):
    """A task the agent acts in. Deliberately gym-shaped, dependency-free.

    `seed` is mandatory plumbing, not a nicety: gate criteria quantify effects
    "over N seeds" (ADR-0005), and reproducibility starts at the environment.
    """

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Advance one step; returns (observation, reward, done)."""
        ...
