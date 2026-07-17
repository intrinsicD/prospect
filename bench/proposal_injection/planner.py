"""Bench-only iCEM planner with controlled first-round proposal replacement.

The production :class:`prospect.planning.FlatPlanner` intentionally has no oracle
seam.  This module supplies that seam for causal search experiments while keeping
the learned planner budget and random-number schedule fixed.  In particular, all
native first-round noise is sampled before any proposal is replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from prospect.interfaces import WorldModel
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, Subgoal

SequenceProvider = Callable[[np.ndarray, int, int, int], np.ndarray]
"""Generate ``(count, horizon, action_dim)`` sequences from a raw sidecar."""


@dataclass(frozen=True)
class ProposalInjectionState:
    """A learned planning latent paired with its exact real-state sidecar.

    The sidecar is deliberately not appended to ``latent.z``: the learned model
    sees exactly the same latent as the native arm, while only the proposal
    provider receives an immutable copy of the exact raw state.
    """

    latent: LatentState
    raw_sidecar: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.latent, LatentState):
            raise TypeError("latent must be a LatentState")
        raw = np.asarray(self.raw_sidecar, dtype=float)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError(f"raw_sidecar must be a non-empty 1-D array, got {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("raw_sidecar must contain only finite values")
        frozen = raw.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "raw_sidecar", frozen)


@dataclass(frozen=True)
class ProposalInjectionDiagnostics:
    """Learned-budget and provenance diagnostics for one call to ``plan``."""

    injected_count: int
    injected_top_elite_count: int
    first_round_best_injected: bool
    best_sequence_injected: bool
    selected_first_action_source: Literal["native", "injected"]
    candidate_eval_count: int
    candidate_transition_eval_count: int

    @property
    def best_sequence_was_injected(self) -> bool:
        """Readable alias used by result serializers and reports."""

        return self.best_sequence_injected


class ProposalInjectionPlanner(FlatPlanner):
    """FlatPlanner replica with first-iteration proposal replacement.

    With ``injection_count == 0``, :meth:`plan` delegates directly to
    :class:`FlatPlanner`; actions, warm starts, scoring calls, and RNG state are
    therefore exactly native.  When enabled, the last ``injection_count`` *fresh*
    candidates in the first iCEM iteration are replaced after all native noise for
    that iteration has been sampled.  Retained candidates remain at the end of the
    population and are never overwritten.

    Candidate provenance is followed through within-call elite retention.  Warm
    candidates inherited from a previous MPC call start as native for the new
    call, so every diagnostic describes only proposals injected at that call.
    """

    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int = 1,
        action_low: float = -2.0,
        action_high: float = 2.0,
        horizon: int = 20,
        candidates: int = 64,
        elites: int = 8,
        iterations: int = 3,
        discount: float = 0.99,
        uncertainty_penalty: float = 0.03,
        seed: int = 0,
        epistemic_horizon_bound: float | None = None,
        colored_beta: float = 2.0,
        keep_elite_fraction: float = 0.3,
        temperature: float = 0.5,
        *,
        injection_count: int = 0,
        sequence_provider: SequenceProvider | None = None,
    ) -> None:
        super().__init__(
            world_model=world_model,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            discount=discount,
            uncertainty_penalty=uncertainty_penalty,
            seed=seed,
            epistemic_horizon_bound=epistemic_horizon_bound,
            colored_beta=colored_beta,
            keep_elite_fraction=keep_elite_fraction,
            temperature=temperature,
        )
        if isinstance(injection_count, bool) or not isinstance(injection_count, int):
            raise TypeError("injection_count must be an integer")
        if injection_count < 0:
            raise ValueError("injection_count must be non-negative")
        keep_count = min(elites, int(np.ceil(keep_elite_fraction * elites)))
        minimum_fresh_count = candidates - keep_count
        if injection_count > minimum_fresh_count:
            raise ValueError(
                "injection_count must not exceed the fresh-candidate count in a "
                f"warm-started first iteration ({minimum_fresh_count})"
            )
        if injection_count > 0 and sequence_provider is None:
            raise ValueError("sequence_provider is required when injection_count is positive")
        if sequence_provider is not None and not callable(sequence_provider):
            raise TypeError("sequence_provider must be callable")

        self.injection_count = injection_count
        self.sequence_provider = sequence_provider
        self._diagnostics: list[ProposalInjectionDiagnostics] = []

    @property
    def diagnostics(self) -> tuple[ProposalInjectionDiagnostics, ...]:
        """All completed per-plan diagnostics in call order."""

        return tuple(self._diagnostics)

    @property
    def last_diagnostics(self) -> ProposalInjectionDiagnostics | None:
        """Diagnostics from the most recently completed plan call, if any."""

        return self._diagnostics[-1] if self._diagnostics else None

    def clear_diagnostics(self) -> None:
        """Clear audit records without changing RNG or receding-horizon state."""

        self._diagnostics.clear()

    def plan(
        self,
        state: LatentState | ProposalInjectionState,
        goal: Subgoal | None = None,
    ) -> Action:
        """Plan from a learned latent, using the sidecar only for replacements."""

        if isinstance(state, ProposalInjectionState):
            latent = state.latent
            raw_sidecar: np.ndarray | None = state.raw_sidecar
        elif isinstance(state, LatentState):
            latent = state
            raw_sidecar = None
        else:
            raise TypeError("state must be a LatentState or ProposalInjectionState")

        if self.injection_count == 0:
            action = super().plan(latent, goal)
            self._diagnostics.append(
                ProposalInjectionDiagnostics(
                    injected_count=0,
                    injected_top_elite_count=0,
                    first_round_best_injected=False,
                    best_sequence_injected=False,
                    selected_first_action_source="native",
                    candidate_eval_count=self.candidates * self.iterations,
                    candidate_transition_eval_count=(
                        self.candidates * self.iterations * self.horizon
                    ),
                )
            )
            return action

        if raw_sidecar is None:
            raise ValueError(
                "ProposalInjectionState with a raw_sidecar is required when injection is enabled"
            )
        assert self.sequence_provider is not None  # validated in __init__
        return self._plan_with_injection(latent, raw_sidecar)

    def _plan_with_injection(self, state: LatentState, raw_sidecar: np.ndarray) -> Action:
        """Mirror FlatPlanner.plan while replacing first-round fresh proposals."""

        if self._warm_mean is not None:
            mean = np.concatenate([self._warm_mean[1:], self._warm_mean[-1:]], axis=0)
        else:
            mean = np.zeros((self.horizon, self.action_dim))
        std = np.full(
            (self.horizon, self.action_dim),
            0.25 * (self.action_high - self.action_low),
        )
        keep_count = min(
            self.elites,
            int(np.ceil(self.keep_elite_fraction * self.elites)),
        )
        carried = (
            self._shift_sequences(self._warm_elites[:keep_count])
            if self._warm_elites is not None
            else np.empty((0, self.horizon, self.action_dim))
        )
        # Provenance is local to this MPC call.  Prior-call warm candidates are
        # useful native warm starts here, not members of this call's injected set.
        carried_injected = np.zeros(len(carried), dtype=bool)
        best_sequence: np.ndarray | None = None
        best_score = -np.inf
        best_sequence_injected = False
        first_round_best_injected = False
        injected_top_elite_count = 0
        candidate_eval_count = 0
        elite = np.empty((0, self.horizon, self.action_dim))

        for iteration_index in range(self.iterations):
            carried = carried[: self.candidates]
            carried_injected = carried_injected[: self.candidates]
            fresh_count = self.candidates - len(carried)

            # Consume the complete native draw before replacing any row.  The
            # oracle/control provider therefore cannot perturb FlatPlanner's RNG
            # schedule even though its returned rows alter later elite moments.
            noise = self._sample_colored_noise(fresh_count)
            fresh = np.clip(mean + std * noise, self.action_low, self.action_high)
            fresh_injected = np.zeros(fresh_count, dtype=bool)
            if iteration_index == 0:
                replacements = self._replacement_sequences(raw_sidecar)
                start = fresh_count - self.injection_count
                fresh[start:] = replacements
                fresh_injected[start:] = True

            sequences = np.concatenate([fresh, carried], axis=0)
            sequence_injected = np.concatenate([fresh_injected, carried_injected])
            scores = self._imagined_returns(state, sequences)
            candidate_eval_count += len(sequences)
            best_index = int(np.argmax(scores))
            if iteration_index == 0:
                first_round_best_injected = bool(sequence_injected[best_index])
            if float(scores[best_index]) > best_score:
                best_score = float(scores[best_index])
                best_sequence = sequences[best_index].copy()
                best_sequence_injected = bool(sequence_injected[best_index])

            elite_indices = np.argsort(scores)[-self.elites :][::-1]
            elite = sequences[elite_indices]
            elite_injected = sequence_injected[elite_indices]
            if iteration_index == 0:
                injected_top_elite_count = int(np.count_nonzero(elite_injected))
            elite_scores = scores[elite_indices]
            weights = self._softmax_weights(elite_scores, self.temperature)
            mean = np.sum(weights[:, None, None] * elite, axis=0)
            variance = np.sum(weights[:, None, None] * (elite - mean) ** 2, axis=0)
            std = np.maximum(np.sqrt(variance), 0.05)
            carried = elite[:keep_count].copy()
            carried_injected = elite_injected[:keep_count].copy()

        assert best_sequence is not None
        self._warm_mean = mean
        self._warm_elites = elite.copy()
        selected_source: Literal["native", "injected"] = (
            "injected" if best_sequence_injected else "native"
        )
        self._diagnostics.append(
            ProposalInjectionDiagnostics(
                injected_count=self.injection_count,
                injected_top_elite_count=injected_top_elite_count,
                first_round_best_injected=first_round_best_injected,
                best_sequence_injected=best_sequence_injected,
                selected_first_action_source=selected_source,
                candidate_eval_count=candidate_eval_count,
                candidate_transition_eval_count=candidate_eval_count * self.horizon,
            )
        )
        return Action(data=best_sequence[0].copy())

    def _replacement_sequences(self, raw_sidecar: np.ndarray) -> np.ndarray:
        """Call and strictly validate the experiment-owned sequence provider."""

        assert self.sequence_provider is not None
        proposals = np.asarray(
            self.sequence_provider(
                raw_sidecar.copy(),
                self.injection_count,
                self.horizon,
                self.action_dim,
            ),
            dtype=float,
        )
        expected = (self.injection_count, self.horizon, self.action_dim)
        if proposals.shape != expected:
            raise ValueError(f"sequence_provider must return shape {expected}, got {proposals.shape}")
        if not np.all(np.isfinite(proposals)):
            raise ValueError("sequence_provider returned non-finite actions")
        if np.any(proposals < self.action_low) or np.any(proposals > self.action_high):
            raise ValueError(
                "sequence_provider returned actions outside the planner's action bounds"
            )
        return proposals.copy()


# Concise aliases make experiment manifests readable while retaining explicit
# public class names in serialized metadata.
PlannerState = ProposalInjectionState
PlanDiagnostics = ProposalInjectionDiagnostics


__all__ = [
    "PlanDiagnostics",
    "PlannerState",
    "ProposalInjectionDiagnostics",
    "ProposalInjectionPlanner",
    "ProposalInjectionState",
    "SequenceProvider",
]
