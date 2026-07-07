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

    def set_state(self, theta: float, omega: float) -> Observation:
        """Harness surface for counterfactual probes (P3): place the pendulum at an
        exact state so the same premise can be stepped under different physics."""
        self._theta, self._omega = float(theta), float(omega)
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


class PendulumSwingup(Pendulum):
    """The Pendulum started HANGING DOWN (θ≈π): reaching the upright reward requires an
    energy-pumping swing-up, because `max_torque` is far below the torque needed to lift the
    pole statically. The numpy analogue of DMC cartpole-swingup and an **exploration-hard**
    task — random torque almost never reaches upright, so a from-scratch learner fails at a
    small budget while *watching* a demonstration succeeds (the P14 imitation gate). Only the
    initial state differs from `Pendulum`; the dynamics and reward are inherited. Satisfies
    `bench.Environment`."""

    def reset(self, seed: int | None = None) -> Observation:
        self._rng = np.random.default_rng(seed)
        theta = float(np.pi + self._rng.uniform(-0.3, 0.3))
        self._theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)  # wrap to (-π, π]
        self._omega = float(self._rng.uniform(-0.3, 0.3))
        return self._obs()


class PointMass:
    """2D point mass with nonlinear (quadratic) drag — the P9-003 second environment.

    Structurally different from the Pendulum: Cartesian, 4-dim state (x, y, vx, vy),
    2-dim action (ax, ay force), no rotational/trig structure. The nonlinearity is
    quadratic drag (`drag * v * |v|`): negligible at low speed, dominant at high speed,
    so a model trained on a limited-velocity region is confident there and uncertain
    outside it — the seen/OOD split P8/P9-style retrieval needs. No spring: the agent
    must apply force to reach and hold the origin. Satisfies `bench.Environment`.

    The point of a *second* environment is to run the SAME core (world model, planner,
    retrieval) on a genuinely different task with only recalibrated thresholds — a
    capability that survives is real, one that collapses was a Pendulum artifact (P9-003).
    """

    def __init__(
        self,
        dt: float = 0.1,
        force_scale: float = 4.0,
        drag: float = 0.4,
        max_force: float = 1.0,
        vel_max: float = 8.0,
        pos_max: float = 8.0,
        init_pos: float = 2.0,
        init_vel: float = 1.0,
    ) -> None:
        self.dt, self.force_scale, self.drag = dt, force_scale, drag
        self.max_force, self.vel_max, self.pos_max = max_force, vel_max, pos_max
        self.init_pos, self.init_vel = init_pos, init_vel
        self._rng = np.random.default_rng(0)
        self._state = np.zeros(4)  # x, y, vx, vy

    def reset(self, seed: int | None = None) -> Observation:
        self._rng = np.random.default_rng(seed)
        self._state = np.array([
            self._rng.uniform(-self.init_pos, self.init_pos),
            self._rng.uniform(-self.init_pos, self.init_pos),
            self._rng.uniform(-self.init_vel, self.init_vel),
            self._rng.uniform(-self.init_vel, self.init_vel),
        ])
        return self._obs()

    def set_state(self, x: float, y: float, vx: float, vy: float) -> Observation:
        """Harness surface for region-exact probes (P9-003): place the mass at a state
        so the seen/OOD velocity boundary is exact, not visitation-dependent."""
        self._state = np.array([x, y, vx, vy], dtype=float)
        return self._obs()

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        a = np.clip(np.asarray(action.data, dtype=float).ravel()[:2], -self.max_force, self.max_force)
        s = self._state
        vx = float(np.clip(s[2] + self.dt * (self.force_scale * a[0] - self.drag * s[2] * abs(s[2])),
                           -self.vel_max, self.vel_max))
        vy = float(np.clip(s[3] + self.dt * (self.force_scale * a[1] - self.drag * s[3] * abs(s[3])),
                           -self.vel_max, self.vel_max))
        x = float(np.clip(s[0] + self.dt * vx, -self.pos_max, self.pos_max))
        y = float(np.clip(s[1] + self.dt * vy, -self.pos_max, self.pos_max))
        self._state = np.array([x, y, vx, vy])
        reward = -(0.1 * (x**2 + y**2) + 0.01 * (vx**2 + vy**2) + 0.001 * float(a @ a))
        return self._obs(), float(reward), False

    def _obs(self) -> Observation:
        return Observation(modality=Modality.STATE, data=self._state.copy())
