"""The harness's episode driver (P2-002): runs anything with `act`/`reset` on an
`Environment`. The one act/step loop gate evals share — behavior-identical to the
loops it replaced (the shipped P2 gate reproduces its exact metrics through it).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from prospect.types import Action, LatentState, Observation, Transition

from .envs import Environment


@runtime_checkable
class Acting(Protocol):
    """What `run_episode` needs: `prospect.agent.Agent` satisfies it; plain
    policies wrap in a trivial adapter."""

    def act(self, obs: Observation) -> Action: ...
    def reset(self) -> None: ...


def run_episode(
    env: Environment,
    agent: Acting,
    steps: int,
    seed: int,
    collect: bool = False,
) -> tuple[float, list[Transition]]:
    """One seeded episode; returns (total reward, transitions if collect else []).

    Collected transitions carry raw modality data in `.state.z` (P0-011) so
    replay stays re-encodable under a future codec."""
    agent.reset()
    obs = env.reset(seed=seed)
    total = 0.0
    transitions: list[Transition] = []
    for _ in range(steps):
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        total += reward
        if collect:
            transitions.append(
                Transition(
                    state=LatentState(z=obs.data), action=action,
                    next_state=LatentState(z=next_obs.data), reward=reward,
                )
            )
        obs = next_obs
        if done:
            break
    return total, transitions
