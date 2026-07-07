"""A DeepMind Control Suite task exposed through the harness's `bench.Environment`
Protocol (BH-001, ADR-0011).

The point of this adapter is what it does *not* require: the core (`src/prospect/`)
needs zero changes to act in a real MuJoCo task — `FlatWorldModel`, `FlatPlanner` and
`Agent` already take `obs_dim`/`action_dim`/`action_low`/`action_high`, so the
Environment seam (P0-004) is the only glue. **State observations only** (the DMC
observation dict, flattened) — no pixels, hence no GL/render dependency, so it steps
headless. `dm_control` is the optional `[bench-hard]` extra.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prospect.types import Action, Modality, Observation


def _load_suite() -> Any:
    """Import dm_control lazily with a clear pointer to the optional extra."""
    try:
        from dm_control import suite
    except ImportError as err:  # pragma: no cover - the extra-absent path
        raise ImportError(
            "the harder benchmark needs the optional `bench-hard` extra: "
            "pip install -e '.[bench-hard]'  (installs dm_control + mujoco). "
            "It is deliberately kept out of the numpy-only core install."
        ) from err
    return suite


class DMCEnvironment:
    """A DeepMind Control Suite task as a `bench.Environment`.

    `reset(seed)` reloads the task with that seed so episodes are reproducible
    (ADR-0005 — reproducibility starts at the environment). Actions are the task's
    native `action_spec` box; `action_low`/`action_high` (and `obs_dim`/`action_dim`)
    are exposed so the planner and the model-free baseline size themselves to the task
    with no per-task code. Satisfies `bench.Environment`.
    """

    def __init__(self, domain: str = "cartpole", task: str = "swingup") -> None:
        self.domain, self.task = domain, task
        self._suite = _load_suite()
        self._env = self._load(0)
        spec = self._env.action_spec()
        self.action_dim = int(np.prod(spec.shape))
        self._act_min = np.broadcast_to(np.asarray(spec.minimum, dtype=float), spec.shape).ravel()
        self._act_max = np.broadcast_to(np.asarray(spec.maximum, dtype=float), spec.shape).ravel()
        self.action_low = float(self._act_min.min())
        self.action_high = float(self._act_max.max())
        self.obs_dim = int(self._flatten(self._env.reset().observation).shape[0])

    def _load(self, seed: int) -> Any:
        return self._suite.load(domain_name=self.domain, task_name=self.task,
                                task_kwargs={"random": int(seed)})

    @staticmethod
    def _flatten(observation: Any) -> np.ndarray:
        """DMC returns an ordered obs dict; concatenate into one state vector."""
        return np.concatenate([np.asarray(v, dtype=float).ravel() for v in observation.values()])

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._reseed(seed)
        ts = self._env.reset()
        return Observation(modality=Modality.STATE, data=self._flatten(ts.observation))

    def _reseed(self, seed: int) -> None:
        """Reseed the loaded task in place so episodes are reproducible without
        rebuilding the MuJoCo model each reset (suite tasks read `task._random`);
        fall back to a full reload if the task does not expose its RNG."""
        task = getattr(self._env, "task", None)
        if task is not None and hasattr(task, "_random"):
            task._random = np.random.RandomState(seed)
        else:  # pragma: no cover - task without an exposed RNG
            self._env = self._load(seed)

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        a = np.asarray(action.data, dtype=float).ravel()
        if a.shape[0] < self.action_dim:  # tolerate a shorter action vector
            a = np.pad(a, (0, self.action_dim - a.shape[0]))
        a = np.clip(a[: self.action_dim], self._act_min, self._act_max)
        ts = self._env.step(a)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        obs = Observation(modality=Modality.STATE, data=self._flatten(ts.observation))
        return obs, reward, bool(ts.last())
