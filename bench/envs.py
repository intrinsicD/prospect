"""The harness's environment seam (P0-004) and the P1 reference task (P1-001).
Environments are task-specific, so they live in the harness (golden rule 3): gate
evals and training loops target this one contract instead of re-inventing wiring
per phase.

Import direction is one-way: the harness may import core types (`prospect.types`);
the core must never import `bench` (guarded by a test).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from prospect.types import Action, Modality, Observation


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


class Pendulum:
    """Torque-controlled pendulum — the P1 reference task. obs = (cosθ, sinθ, ω).

    Nonlinear (sinθ) so a learned nonlinear model can beat the linear baseline.
    Deterministic by default; `noise_std > 0` adds Gaussian noise to the angular
    velocity — irreducible by construction (aleatoric), for the P1
    epistemic/aleatoric separation experiment. `init_omega` widens the initial
    velocity range (used to build out-of-distribution probes). Satisfies
    `bench.Environment`.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        dt: float = 0.15,
        gravity: float = 10.0,
        damping: float = 0.15,
        max_torque: float = 2.0,
        omega_max: float = 8.0,
        init_omega: float = 1.0,
    ) -> None:
        self.noise_std, self.dt, self.gravity = noise_std, dt, gravity
        self.damping, self.max_torque, self.omega_max = damping, max_torque, omega_max
        self.init_omega = init_omega
        self._rng = np.random.default_rng(0)
        self._theta, self._omega = 0.0, 0.0

    def reset(self, seed: int | None = None) -> Observation:
        self._rng = np.random.default_rng(seed)
        self._theta = float(self._rng.uniform(-np.pi, np.pi))
        self._omega = float(self._rng.uniform(-self.init_omega, self.init_omega))
        return self._obs()

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        torque = float(np.clip(np.asarray(action.data, dtype=float).ravel()[0],
                               -self.max_torque, self.max_torque))
        omega = self._omega + self.dt * (
            self.gravity * np.sin(self._theta) - self.damping * self._omega + torque
        )
        if self.noise_std > 0.0:
            omega += self.noise_std * float(self._rng.normal())
        self._omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        theta = self._theta + self.dt * self._omega
        self._theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)  # wrap to (-pi, pi]
        reward = -(self._theta**2 + 0.1 * self._omega**2 + 0.01 * torque**2) / 10.0
        return self._obs(), float(reward), False

    def _obs(self) -> Observation:
        data = np.array([np.cos(self._theta), np.sin(self._theta), self._omega], dtype=float)
        return Observation(modality=Modality.STATE, data=data)
