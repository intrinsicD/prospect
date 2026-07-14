"""Deterministic fixed-bank diagnostics for the OL-001 oracle ladder.

The online iCEM return is the experiment's control outcome.  This module supplies
the complementary, common-candidate audit: every rung scores exactly the same
bank, including a small set of sequences found by a deliberately larger exact-
model search.  The helpers are pure and do not train models or run episodes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import numpy as np
import numpy.typing as npt

from bench.bridge_control.fixture import transition_dynamics

AUDIT_SCHEMA_VERSION: Final = "oracle-ladder-fixed-bank-v1"
HORIZON: Final = 12
ACTION_DIM: Final = 2
ACTION_LOW: Final = -1.0
ACTION_HIGH: Final = 1.0
DISCOUNT: Final = 0.99
STANDARD_PROPOSAL_COUNT: Final = 120
REFERENCE_ELITE_COUNT: Final = 8
REFERENCE_CANDIDATES: Final = 512
REFERENCE_SEARCH_ELITES: Final = 32
REFERENCE_ITERATIONS: Final = 5
COLORED_BETA: Final = 2.0
KEEP_ELITE_FRACTION: Final = 0.3
TEMPERATURE: Final = 0.5
INITIAL_STD: Final = 0.5
MIN_STD: Final = 0.05
REFERENCE_SEED_SALT: Final = 0x4F4C0001

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "ACTION_DIM",
    "AUDIT_SCHEMA_VERSION",
    "DISCOUNT",
    "FixedBank",
    "HORIZON",
    "REFERENCE_ELITE_COUNT",
    "REFERENCE_SEARCH",
    "RankAudit",
    "STANDARD_PROPOSAL_COUNT",
    "average_tie_ranks",
    "build_fixed_bank",
    "build_fixed_banks",
    "canonical_bank_sha256",
    "exact_discounted_scores",
    "pearson_correlation",
    "rank_audit",
    "spearman_correlation",
]


@dataclass(frozen=True)
class ReferenceSearchSpec:
    """Frozen compute specification for the exact-model reference search."""

    candidates: int = REFERENCE_CANDIDATES
    elites: int = REFERENCE_SEARCH_ELITES
    iterations: int = REFERENCE_ITERATIONS
    returned_elites: int = REFERENCE_ELITE_COUNT
    horizon: int = HORIZON
    discount: float = DISCOUNT
    colored_beta: float = COLORED_BETA
    keep_elite_fraction: float = KEEP_ELITE_FRACTION
    temperature: float = TEMPERATURE

    @property
    def evaluated_sequences(self) -> int:
        return self.candidates * self.iterations

    def as_dict(self) -> dict[str, int | float]:
        return {
            "candidates": self.candidates,
            "elites": self.elites,
            "iterations": self.iterations,
            "returned_elites": self.returned_elites,
            "evaluated_sequences": self.evaluated_sequences,
            "horizon": self.horizon,
            "discount": self.discount,
            "colored_beta": self.colored_beta,
            "keep_elite_fraction": self.keep_elite_fraction,
            "temperature": self.temperature,
        }


REFERENCE_SEARCH: Final = ReferenceSearchSpec()


def _frozen_float_array(values: npt.ArrayLike) -> FloatArray:
    array = np.array(values, dtype=np.float64, order="C", copy=True)
    array.setflags(write=False)
    return array


def _finite_vector(values: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=np.float64)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _array_bytes(values: npt.ArrayLike) -> bytes:
    """Canonical C-order, little-endian float64 representation."""

    return np.asarray(values, dtype="<f8", order="C").tobytes(order="C")


def canonical_bank_sha256(
    start: npt.ArrayLike,
    sequences: npt.ArrayLike,
    sources: Sequence[str],
) -> str:
    """Return a content hash for a start-conditioned candidate bank.

    Scores are intentionally excluded: this digest identifies the candidate
    evidence itself.  Exact and rung scores are separately stored and verified.
    Shapes, source labels, and the start state are included, so reshaping or
    relabelling identical numeric bytes cannot preserve the digest.
    """

    start_array = np.asarray(start, dtype=np.float64)
    sequence_array = np.asarray(sequences, dtype=np.float64)
    if start_array.shape != (3,):
        raise ValueError("start must have shape (3,)")
    if sequence_array.ndim != 3 or sequence_array.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"sequences must have shape (n, {HORIZON}, {ACTION_DIM})")
    if len(sources) != len(sequence_array):
        raise ValueError("one source label is required per sequence")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(sequence_array)):
        raise ValueError("bank arrays must contain only finite values")

    header = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dtype": "<f8",
        "start_shape": list(start_array.shape),
        "sequence_shape": list(sequence_array.shape),
        "sources": list(sources),
    }
    digest = sha256()
    digest.update(b"OL-001 fixed candidate bank\0")
    header_bytes = _canonical_json_bytes(header)
    digest.update(len(header_bytes).to_bytes(8, byteorder="big"))
    digest.update(header_bytes)
    digest.update(_array_bytes(start_array))
    digest.update(_array_bytes(sequence_array))
    return digest.hexdigest()


def exact_discounted_scores(
    start: npt.ArrayLike,
    sequences: npt.ArrayLike,
    *,
    discount: float = DISCOUNT,
) -> FloatArray:
    """Score every H=12 action sequence with frozen BridgeControl dynamics."""

    start_array = np.asarray(start, dtype=np.float64)
    sequence_array = np.asarray(sequences, dtype=np.float64)
    if start_array.shape != (3,):
        raise ValueError("start must have shape (3,)")
    if sequence_array.ndim != 3 or sequence_array.shape[1:] != (HORIZON, ACTION_DIM):
        raise ValueError(f"sequences must have shape (n, {HORIZON}, {ACTION_DIM})")
    if len(sequence_array) == 0:
        raise ValueError("at least one sequence is required")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(sequence_array)):
        raise ValueError("exact-score inputs must contain only finite values")

    scores = np.zeros(len(sequence_array), dtype=np.float64)
    states = np.repeat(start_array.reshape(1, 3), len(sequence_array), axis=0)
    weight = 1.0
    for step in range(HORIZON):
        next_states: list[FloatArray] = []
        rewards: list[float] = []
        for state, action in zip(states, sequence_array[:, step], strict=True):
            next_state, reward = transition_dynamics(state, action)
            next_states.append(np.asarray(next_state, dtype=np.float64))
            rewards.append(float(reward))
        states = np.stack(next_states)
        scores += weight * np.asarray(rewards, dtype=np.float64)
        weight *= discount
    return np.asarray(scores, dtype=np.float64)


def _sample_colored_noise(
    rng: np.random.Generator,
    count: int,
    *,
    beta: float = COLORED_BETA,
) -> FloatArray:
    """Mirror FlatPlanner's standard f^(-beta/2) proposal coloring."""

    if count < 0:
        raise ValueError("proposal count must be non-negative")
    if beta < 0.0:
        raise ValueError("colored-noise beta must be non-negative")
    white = rng.normal(size=(count, HORIZON, ACTION_DIM))
    if count == 0 or beta == 0.0:
        return np.asarray(white, dtype=np.float64)
    spectrum = np.fft.rfft(white, axis=1)
    frequencies = np.fft.rfftfreq(HORIZON)
    frequencies[0] = frequencies[1]
    scale = frequencies ** (-beta / 2.0)
    kernel = np.fft.irfft(scale, n=HORIZON)
    normalizer = float(np.sqrt(np.sum(kernel**2)))
    spectrum *= scale[None, :, None]
    return np.asarray(np.fft.irfft(spectrum, n=HORIZON, axis=1) / normalizer, dtype=np.float64)


def _standard_colored_proposals(seed: int) -> FloatArray:
    # Using default_rng(seed) makes this block directly parity-testable against
    # FlatPlanner's first, zero-mean proposal round at the same seed.
    rng = np.random.default_rng(seed)
    noise = _sample_colored_noise(rng, STANDARD_PROPOSAL_COUNT)
    return np.asarray(np.clip(INITIAL_STD * noise, ACTION_LOW, ACTION_HIGH), dtype=np.float64)


def _descending_indices(scores: FloatArray) -> npt.NDArray[np.int64]:
    """Stable score-descending order with the lower candidate index as tie break."""

    indices = np.arange(len(scores), dtype=np.int64)
    return np.asarray(np.lexsort((indices, -scores)), dtype=np.int64)


def _softmax_weights(scores: FloatArray) -> FloatArray:
    logits = (scores - float(np.max(scores))) / TEMPERATURE
    weights = np.exp(logits)
    return np.asarray(weights / np.sum(weights), dtype=np.float64)


def _sequence_key(sequence: npt.ArrayLike) -> bytes:
    return _array_bytes(sequence)


def _exact_reference_elites(
    start: FloatArray,
    seed: int,
    excluded: set[bytes],
) -> FloatArray:
    """Run the frozen larger exact iCEM search and return its best unique eight."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, REFERENCE_SEED_SALT]))
    mean = np.zeros((HORIZON, ACTION_DIM), dtype=np.float64)
    std = np.full((HORIZON, ACTION_DIM), INITIAL_STD, dtype=np.float64)
    keep_count = min(
        REFERENCE_SEARCH_ELITES,
        int(np.ceil(KEEP_ELITE_FRACTION * REFERENCE_SEARCH_ELITES)),
    )
    carried = np.empty((0, HORIZON, ACTION_DIM), dtype=np.float64)
    evaluated_sequences: list[FloatArray] = []
    evaluated_scores: list[FloatArray] = []

    for _ in range(REFERENCE_ITERATIONS):
        carried = carried[:REFERENCE_CANDIDATES]
        fresh_count = REFERENCE_CANDIDATES - len(carried)
        noise = _sample_colored_noise(rng, fresh_count)
        fresh = np.clip(mean + std * noise, ACTION_LOW, ACTION_HIGH)
        sequences = np.concatenate([fresh, carried], axis=0)
        scores = exact_discounted_scores(start, sequences)
        evaluated_sequences.append(np.asarray(sequences, dtype=np.float64))
        evaluated_scores.append(scores)

        elite_indices = _descending_indices(scores)[:REFERENCE_SEARCH_ELITES]
        elite = sequences[elite_indices]
        elite_scores = scores[elite_indices]
        weights = _softmax_weights(elite_scores)
        mean = np.sum(weights[:, None, None] * elite, axis=0)
        variance = np.sum(weights[:, None, None] * (elite - mean) ** 2, axis=0)
        std = np.maximum(np.sqrt(variance), MIN_STD)
        carried = elite[:keep_count].copy()

    all_sequences = np.concatenate(evaluated_sequences, axis=0)
    all_scores = np.concatenate(evaluated_scores, axis=0)
    selected: list[FloatArray] = []
    seen = set(excluded)
    for index in _descending_indices(all_scores):
        sequence = all_sequences[int(index)]
        key = _sequence_key(sequence)
        if key in seen:
            continue
        selected.append(np.asarray(sequence, dtype=np.float64).copy())
        seen.add(key)
        if len(selected) == REFERENCE_ELITE_COUNT:
            break
    if len(selected) != REFERENCE_ELITE_COUNT:
        raise RuntimeError("exact reference search did not produce eight unique sequences")
    return np.stack(selected)


@dataclass(frozen=True)
class FixedBank:
    """One immutable, start-conditioned 120+8 common-candidate bank."""

    start: FloatArray
    sequences: FloatArray
    sources: tuple[str, ...]
    exact_scores: FloatArray
    seed: int

    def __post_init__(self) -> None:
        start = _frozen_float_array(self.start)
        sequences = _frozen_float_array(self.sequences)
        exact_scores = _frozen_float_array(self.exact_scores)
        if start.shape != (3,):
            raise ValueError("start must have shape (3,)")
        expected_shape = (STANDARD_PROPOSAL_COUNT + REFERENCE_ELITE_COUNT, HORIZON, ACTION_DIM)
        if sequences.shape != expected_shape:
            raise ValueError(f"fixed bank must have shape {expected_shape}")
        if exact_scores.shape != (len(sequences),):
            raise ValueError("exact scores must contain one value per sequence")
        if len(self.sources) != len(sequences):
            raise ValueError("fixed bank must contain one source label per sequence")
        expected_sources = ("standard_colored",) * STANDARD_PROPOSAL_COUNT + (
            "exact_reference",
        ) * REFERENCE_ELITE_COUNT
        if self.sources != expected_sources:
            raise ValueError("fixed bank source labels or ordering do not match the frozen design")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(sequences)):
            raise ValueError("fixed bank arrays must contain only finite values")
        if not np.all(np.isfinite(exact_scores)):
            raise ValueError("fixed bank exact scores must be finite")
        recomputed = exact_discounted_scores(start, sequences)
        if not np.array_equal(exact_scores, recomputed):
            raise ValueError("fixed bank exact scores do not match its candidate sequences")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "exact_scores", exact_scores)

    @property
    def sha256(self) -> str:
        return canonical_bank_sha256(self.start, self.sequences, self.sources)

    @property
    def reference_indices(self) -> npt.NDArray[np.int64]:
        return np.arange(
            STANDARD_PROPOSAL_COUNT,
            STANDARD_PROPOSAL_COUNT + REFERENCE_ELITE_COUNT,
            dtype=np.int64,
        )

    def as_dict(self, *, include_sequences: bool = True) -> dict[str, object]:
        """Return a canonical JSON-ready record for results or reporting."""

        record: dict[str, object] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "seed": self.seed,
            "start": self.start.tolist(),
            "horizon": HORIZON,
            "action_dim": ACTION_DIM,
            "bank_sha256": self.sha256,
            "standard_proposal_count": STANDARD_PROPOSAL_COUNT,
            "reference_elite_count": REFERENCE_ELITE_COUNT,
            "reference_search": REFERENCE_SEARCH.as_dict(),
            "sources": list(self.sources),
            "exact_scores": self.exact_scores.tolist(),
        }
        if include_sequences:
            record["sequences"] = self.sequences.tolist()
        return record


def build_fixed_bank(start: npt.ArrayLike, *, seed: int) -> FixedBank:
    """Build the deterministic H=12 bank for one evaluation start."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    start_array = np.asarray(start, dtype=np.float64)
    if start_array.shape != (3,) or not np.all(np.isfinite(start_array)):
        raise ValueError("start must be a finite array with shape (3,)")
    standard = _standard_colored_proposals(seed)
    excluded = {_sequence_key(sequence) for sequence in standard}
    reference = _exact_reference_elites(start_array, seed, excluded)
    sequences = np.concatenate([standard, reference], axis=0)
    sources = ("standard_colored",) * STANDARD_PROPOSAL_COUNT + ("exact_reference",) * REFERENCE_ELITE_COUNT
    scores = exact_discounted_scores(start_array, sequences)
    return FixedBank(
        start=start_array,
        sequences=sequences,
        sources=sources,
        exact_scores=scores,
        seed=seed,
    )


def build_fixed_banks(starts: npt.ArrayLike, *, seed: int) -> tuple[FixedBank, ...]:
    """Build per-start banks with common random standard/reference proposals."""

    start_array = np.asarray(starts, dtype=np.float64)
    if start_array.ndim != 2 or start_array.shape[1] != 3 or len(start_array) == 0:
        raise ValueError("starts must have shape (n, 3)")
    return tuple(build_fixed_bank(start, seed=seed) for start in start_array)


def average_tie_ranks(values: npt.ArrayLike, *, descending: bool = True) -> FloatArray:
    """Return one-based ranks, assigning tied values their average occupied rank."""

    array = _finite_vector(values, name="rank values")
    order = np.argsort(-array if descending else array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Occupied one-based ranks are start+1 through stop.
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def pearson_correlation(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    """Finite Pearson correlation; return zero when either vector is constant."""

    x = _finite_vector(left, name="left correlation values")
    y = _finite_vector(right, name="right correlation values")
    if x.shape != y.shape:
        raise ValueError("correlation vectors must have identical shapes")
    if len(x) < 2:
        raise ValueError("correlation requires at least two values")
    centered_x = x - float(np.mean(x))
    centered_y = y - float(np.mean(y))
    denominator = float(np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)))
    if denominator == 0.0:
        return 0.0
    correlation = float(np.dot(centered_x, centered_y) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))


def spearman_correlation(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    """Spearman correlation using average ranks for ties."""

    x = _finite_vector(left, name="left correlation values")
    y = _finite_vector(right, name="right correlation values")
    if x.shape != y.shape:
        raise ValueError("correlation vectors must have identical shapes")
    return pearson_correlation(
        average_tie_ranks(x, descending=False),
        average_tie_ranks(y, descending=False),
    )


@dataclass(frozen=True)
class RankAudit:
    """Recomputable score/rank diagnostics for one rung on one fixed bank."""

    candidate_count: int
    selected_index: int
    selected_source: str
    selected_score: float
    selected_exact_score: float
    exact_best_score: float
    exact_worst_score: float
    exact_selected_rank: float
    exact_selected_rank_normalized: float
    normalized_selected_regret: float
    pearson: float
    spearman: float
    reference_count: int
    reference_best_rank: float
    reference_mean_rank: float
    reference_median_rank: float
    reference_worst_rank: float
    reference_mean_rank_normalized: float
    reference_top_k: int
    reference_top_k_fraction: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "candidate_count": self.candidate_count,
            "selected_index": self.selected_index,
            "selected_source": self.selected_source,
            "selected_score": self.selected_score,
            "selected_exact_score": self.selected_exact_score,
            "exact_best_score": self.exact_best_score,
            "exact_worst_score": self.exact_worst_score,
            "exact_selected_rank": self.exact_selected_rank,
            "exact_selected_rank_normalized": self.exact_selected_rank_normalized,
            "normalized_selected_regret": self.normalized_selected_regret,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "reference_count": self.reference_count,
            "reference_best_rank": self.reference_best_rank,
            "reference_mean_rank": self.reference_mean_rank,
            "reference_median_rank": self.reference_median_rank,
            "reference_worst_rank": self.reference_worst_rank,
            "reference_mean_rank_normalized": self.reference_mean_rank_normalized,
            "reference_top_k": self.reference_top_k,
            "reference_top_k_fraction": self.reference_top_k_fraction,
        }


def rank_audit(
    bank: FixedBank,
    candidate_scores: npt.ArrayLike,
    *,
    selected_index: int | None = None,
    reference_top_k: int | None = None,
) -> RankAudit:
    """Compare one rung's scores with exact scores on the common bank.

    If ``selected_index`` is omitted, the first maximum under the rung scorer is
    treated as selected.  Normalized ranks/regret use 0 for best and 1 for worst.
    """

    scores = _finite_vector(candidate_scores, name="candidate scores")
    if len(scores) != len(bank.sequences):
        raise ValueError("candidate scores must contain one value per bank sequence")
    selected = int(np.argmax(scores)) if selected_index is None else selected_index
    if not 0 <= selected < len(scores):
        raise ValueError("selected index is outside the candidate bank")
    top_k = REFERENCE_ELITE_COUNT if reference_top_k is None else reference_top_k
    if not 1 <= top_k <= len(scores):
        raise ValueError("reference top-k must be between one and the bank size")

    exact = bank.exact_scores
    exact_best = float(np.max(exact))
    exact_worst = float(np.min(exact))
    exact_range = exact_best - exact_worst
    normalized_regret = 0.0 if exact_range == 0.0 else (exact_best - float(exact[selected])) / exact_range

    exact_ranks = average_tie_ranks(exact)
    selected_exact_rank = float(exact_ranks[selected])
    rank_denominator = len(scores) - 1
    selected_rank_normalized = 0.0 if rank_denominator == 0 else (selected_exact_rank - 1.0) / rank_denominator

    rung_ranks = average_tie_ranks(scores)
    reference_ranks = rung_ranks[bank.reference_indices]
    reference_mean_rank = float(np.mean(reference_ranks))
    reference_mean_normalized = 0.0 if rank_denominator == 0 else (reference_mean_rank - 1.0) / rank_denominator
    return RankAudit(
        candidate_count=len(scores),
        selected_index=selected,
        selected_source=bank.sources[selected],
        selected_score=float(scores[selected]),
        selected_exact_score=float(exact[selected]),
        exact_best_score=exact_best,
        exact_worst_score=exact_worst,
        exact_selected_rank=selected_exact_rank,
        exact_selected_rank_normalized=float(selected_rank_normalized),
        normalized_selected_regret=float(normalized_regret),
        pearson=pearson_correlation(exact, scores),
        spearman=spearman_correlation(exact, scores),
        reference_count=len(reference_ranks),
        reference_best_rank=float(np.min(reference_ranks)),
        reference_mean_rank=reference_mean_rank,
        reference_median_rank=float(np.median(reference_ranks)),
        reference_worst_rank=float(np.max(reference_ranks)),
        reference_mean_rank_normalized=float(reference_mean_normalized),
        reference_top_k=top_k,
        reference_top_k_fraction=float(np.mean(reference_ranks <= top_k)),
    )
