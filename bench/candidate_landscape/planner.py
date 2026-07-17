"""Non-invasive candidate-pool capture for the PI-003 injected planner.

The sealed proposal-injection implementation is intentionally left untouched.  This
subclass observes its score calls, reconstructs the already-defined injection
provenance, and evaluates identical action sequences with an experiment-owned exact
scorer.  Observation happens after the learned scores are computed and never changes
the planner's candidates, moments, warm state, action, or random-number schedule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench.oracle_ladder.audit import exact_discounted_scores
from bench.proposal_injection.planner import ProposalInjectionPlanner
from prospect.types import Action, LatentState

ExactScorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _frozen_array(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CandidatePoolAudit:
    """One unchanged iCEM candidate pool scored in learned and exact worlds."""

    call_index: int
    episode_index: int
    step: int
    iteration: int
    raw_state: np.ndarray
    latent_state: np.ndarray
    latent_ood: float | None
    sequences: np.ndarray
    learned_scores: np.ndarray
    exact_scores: np.ndarray
    injected: np.ndarray

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_state, dtype=np.float64)
        latent = np.asarray(self.latent_state, dtype=np.float64)
        sequences = np.asarray(self.sequences, dtype=np.float64)
        learned = np.asarray(self.learned_scores, dtype=np.float64)
        exact = np.asarray(self.exact_scores, dtype=np.float64)
        injected = np.asarray(self.injected, dtype=bool)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            raise ValueError("raw_state must be finite with shape (3,)")
        if latent.ndim != 1 or latent.size == 0 or not np.all(np.isfinite(latent)):
            raise ValueError("latent_state must be a non-empty finite vector")
        if self.latent_ood is not None and not np.isfinite(self.latent_ood):
            raise ValueError("latent_ood must be finite or None")
        if sequences.ndim != 3 or sequences.shape[1:] != (12, 2):
            raise ValueError("sequences must have shape (candidates, 12, 2)")
        if learned.shape != (len(sequences),) or exact.shape != learned.shape:
            raise ValueError("one learned and exact score is required per candidate")
        if injected.shape != learned.shape:
            raise ValueError("one injection flag is required per candidate")
        if not np.all(np.isfinite(sequences)):
            raise ValueError("candidate sequences must be finite")
        if not np.all(np.isfinite(learned)) or not np.all(np.isfinite(exact)):
            raise ValueError("candidate scores must be finite")
        object.__setattr__(self, "raw_state", _frozen_array(raw, dtype=np.float64))
        object.__setattr__(self, "latent_state", _frozen_array(latent, dtype=np.float64))
        object.__setattr__(self, "sequences", _frozen_array(sequences, dtype=np.float64))
        object.__setattr__(self, "learned_scores", _frozen_array(learned, dtype=np.float64))
        object.__setattr__(self, "exact_scores", _frozen_array(exact, dtype=np.float64))
        object.__setattr__(self, "injected", _frozen_array(injected, dtype=bool))


class CandidateLandscapePlanner(ProposalInjectionPlanner):
    """Capture selected PI-003 calls without perturbing the sealed planner.

    ``reset()`` retains the monotonically increasing call index because PI-003's
    exact-reference provider also retains its call index across the four episodes of
    one model seed.  This is required to replay the original provider seed schedule.
    """

    def __init__(
        self,
        *args: object,
        episode_steps: int = 14,
        audit_steps: Iterable[int] = (0, 1, 2),
        exact_scorer: ExactScorer = exact_discounted_scores,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        steps = tuple(sorted(set(audit_steps)))
        if not steps or steps[0] < 0 or steps[-1] >= episode_steps:
            raise ValueError("audit_steps must be unique steps inside the episode")
        if self.injection_count <= 0:
            raise ValueError("candidate-landscape capture requires positive injection")
        if not callable(exact_scorer):
            raise TypeError("exact_scorer must be callable")
        self.episode_steps = episode_steps
        self.audit_steps = steps
        self._exact_scorer = exact_scorer
        self._pool_audits: list[CandidatePoolAudit] = []
        self._call_index = 0
        self._active_call: tuple[int, int, int, np.ndarray] | None = None
        self._active_iteration = 0
        self._active_injected = np.empty(0, dtype=bool)

    @property
    def pool_audits(self) -> tuple[CandidatePoolAudit, ...]:
        """Captured pools in model-seed, episode, step, iteration order."""

        return tuple(self._pool_audits)

    def _plan_with_injection(self, state: LatentState, raw_sidecar: np.ndarray) -> Action:
        call_index = self._call_index
        self._call_index += 1
        episode_index, step = divmod(call_index, self.episode_steps)
        capture = step in self.audit_steps
        if capture:
            keep_count = min(
                self.elites,
                int(np.ceil(self.keep_elite_fraction * self.elites)),
            )
            carried_count = (
                min(len(self._warm_elites), keep_count, self.candidates) if self._warm_elites is not None else 0
            )
            fresh_count = self.candidates - carried_count
            if self.injection_count > fresh_count:
                raise ValueError("injection exceeds the captured first-round fresh pool")
            injected = np.zeros(self.candidates, dtype=bool)
            injected[fresh_count - self.injection_count : fresh_count] = True
            self._active_call = (
                call_index,
                episode_index,
                step,
                np.asarray(raw_sidecar, dtype=np.float64).copy(),
            )
            self._active_iteration = 0
            self._active_injected = injected
        else:
            self._active_call = None
            self._active_injected = np.empty(0, dtype=bool)

        try:
            action = super()._plan_with_injection(state, raw_sidecar)
            if capture and self._active_iteration != self.iterations:
                raise RuntimeError("candidate capture missed an iCEM iteration")
            return action
        finally:
            self._active_call = None
            self._active_injected = np.empty(0, dtype=bool)

    def _imagined_returns(self, state: LatentState, sequences: np.ndarray) -> np.ndarray:
        learned = np.asarray(super()._imagined_returns(state, sequences), dtype=np.float64)
        if self._active_call is None:
            return learned

        if self._active_iteration >= self.iterations:
            raise RuntimeError("candidate capture observed too many iCEM iterations")
        if self._active_injected.shape != learned.shape:
            raise RuntimeError("candidate provenance shape does not match the learned pool")
        call_index, episode_index, step, raw_state = self._active_call
        exact = np.asarray(self._exact_scorer(raw_state.copy(), sequences.copy()), dtype=np.float64)
        self._pool_audits.append(
            CandidatePoolAudit(
                call_index=call_index,
                episode_index=episode_index,
                step=step,
                iteration=self._active_iteration,
                raw_state=raw_state,
                latent_state=np.asarray(state.z, dtype=np.float64),
                latent_ood=state.ood,
                sequences=sequences,
                learned_scores=learned,
                exact_scores=exact,
                injected=self._active_injected,
            )
        )

        elite_indices = np.argsort(learned)[-self.elites :][::-1]
        keep_count = min(
            self.elites,
            int(np.ceil(self.keep_elite_fraction * self.elites)),
        )
        carried_injected = self._active_injected[elite_indices[:keep_count]]
        next_injected = np.zeros(self.candidates, dtype=bool)
        next_injected[self.candidates - keep_count :] = carried_injected
        self._active_injected = next_injected
        self._active_iteration += 1
        return learned


__all__ = ["CandidateLandscapePlanner", "CandidatePoolAudit", "ExactScorer"]
