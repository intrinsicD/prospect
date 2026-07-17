"""Deterministic exact-reference providers for PI-001.

All simulator optimization happens here, outside the learned planner.  The provider
returns only replacement sequences and keeps a complete accounting record so oracle
diagnostic compute cannot be mistaken for learned planning compute.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Literal

import numpy as np

from bench.oracle_ladder.audit import (
    REFERENCE_SEARCH,
    build_fixed_bank,
    exact_discounted_scores,
)

ProviderMode = Literal["privileged", "action_permuted", "time_permuted"]
PROVIDER_SEED_SALT = 0x50490001


@dataclass(frozen=True)
class ReferenceCall:
    """One exact-reference generation and optional negative transformation."""

    call_index: int
    bank_seed: int
    raw_sha256: str
    bank_sha256: str
    mode: ProviderMode
    time_shift: int
    reference_exact_scores: tuple[float, ...]
    output_exact_scores: tuple[float, ...]
    oracle_sequence_evaluations: int
    oracle_transition_evaluations: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ExactReferenceProvider:
    """Generate current-state exact elites under a frozen deterministic seed schedule."""

    def __init__(self, model_seed: int, mode: ProviderMode = "privileged") -> None:
        if isinstance(model_seed, bool) or not isinstance(model_seed, int):
            raise TypeError("model_seed must be an integer")
        if model_seed < 0:
            raise ValueError("model_seed must be non-negative")
        if mode not in ("privileged", "action_permuted", "time_permuted"):
            raise ValueError("unknown exact-reference provider mode")
        self.model_seed = model_seed
        self.mode = mode
        self._calls: list[ReferenceCall] = []

    @property
    def calls(self) -> tuple[ReferenceCall, ...]:
        return tuple(self._calls)

    def __call__(
        self,
        raw_sidecar: np.ndarray,
        count: int,
        horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        raw = np.asarray(raw_sidecar, dtype=np.float64).reshape(-1)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            raise ValueError("PI-001 raw sidecar must be a finite BridgeControl state")
        if count != 8 or horizon != 12 or action_dim != 2:
            raise ValueError("PI-001 provider requires count=8, horizon=12, action_dim=2")

        call_index = len(self._calls)
        bank_seed = self._bank_seed(call_index)
        bank = build_fixed_bank(raw, seed=bank_seed)
        references = np.asarray(bank.sequences[bank.reference_indices], dtype=np.float64).copy()
        output, time_shift = self._transform(references, call_index)
        reference_scores = exact_discounted_scores(raw, references)
        output_scores = exact_discounted_scores(raw, output)
        oracle_sequences = REFERENCE_SEARCH.evaluated_sequences + len(bank.sequences) + 2 * count
        # The final two `count` terms are the explicit reference/output manipulation
        # scores above.  build_fixed_bank already accounts for its own final 128 bank.
        record = ReferenceCall(
            call_index=call_index,
            bank_seed=bank_seed,
            raw_sha256=self._raw_hash(raw),
            bank_sha256=bank.sha256,
            mode=self.mode,
            time_shift=time_shift,
            reference_exact_scores=tuple(float(value) for value in reference_scores),
            output_exact_scores=tuple(float(value) for value in output_scores),
            oracle_sequence_evaluations=oracle_sequences,
            oracle_transition_evaluations=oracle_sequences * horizon,
        )
        self._calls.append(record)
        return output

    def _bank_seed(self, call_index: int) -> int:
        sequence = np.random.SeedSequence(
            [PROVIDER_SEED_SALT, self.model_seed, call_index]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _transform(
        self,
        references: np.ndarray,
        call_index: int,
    ) -> tuple[np.ndarray, int]:
        if self.mode == "privileged":
            return references.copy(), 0
        if self.mode == "action_permuted":
            return references[:, :, ::-1].copy(), 0
        shift = 1 + call_index % (references.shape[1] - 1)
        return np.roll(references, shift=shift, axis=1).copy(), shift

    @staticmethod
    def _raw_hash(raw: np.ndarray) -> str:
        return sha256(np.asarray(raw, dtype="<f8", order="C").tobytes()).hexdigest()


def provider_summary(provider: ExactReferenceProvider) -> dict[str, object]:
    """Return aggregate accounting and manipulation statistics."""

    calls = provider.calls
    reference_scores = np.array(
        [score for call in calls for score in call.reference_exact_scores],
        dtype=float,
    )
    output_scores = np.array(
        [score for call in calls for score in call.output_exact_scores],
        dtype=float,
    )
    return {
        "mode": provider.mode,
        "call_count": len(calls),
        "oracle_sequence_evaluations": sum(
            call.oracle_sequence_evaluations for call in calls
        ),
        "oracle_transition_evaluations": sum(
            call.oracle_transition_evaluations for call in calls
        ),
        "mean_reference_exact_score": (
            float(np.mean(reference_scores)) if len(reference_scores) else None
        ),
        "mean_output_exact_score": (
            float(np.mean(output_scores)) if len(output_scores) else None
        ),
        "mean_exact_score_delta": (
            float(np.mean(output_scores - reference_scores))
            if len(reference_scores)
            else None
        ),
        "calls": [call.as_dict() for call in calls],
    }


__all__ = [
    "ExactReferenceProvider",
    "PROVIDER_SEED_SALT",
    "ProviderMode",
    "ReferenceCall",
    "provider_summary",
]
