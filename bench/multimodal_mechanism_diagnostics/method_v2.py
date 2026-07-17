"""Pure MM-008 v2 optimizer, cache, bias comparator, and oracle core.

This module deliberately has no lifecycle, filesystem, dataset, or random-generator
code.  A :class:`FitContext` owns exactly one permitted fit target; held/scoring
target values cannot enter its objective cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import method as v1

SCHEMA_VERSION: Final = "mm008-v2.1-method-v1"
PROTOCOL_SHA256: Final = "6bd9f35d13a36394ea2a17cdd951a0ea0adf0365909228e73671cc9484c19b5f"
STATE_COUNT: Final = 15_625
FLOW_EQUIVALENCE_ATOL: Final = 1e-12
PREDICTION_AGREEMENT_ATOL: Final = 1e-12
CONFIG_TAG: Final = b"MM008-v2-config\0"
CACHE_SCOPE_TAG: Final = b"MM008-v2-cache-scope\0"
CACHE_CONTENT_TAG: Final = b"MM008-v2-cache-content\0"
CACHE_REQUEST_TAG: Final = b"MM008-v2-cache-requests\0"

Arm = Literal["affine", "combined"]
StartName = Literal["F", "R"]
BlockName = Literal["T", "U", "V", "TU", "TV", "UV"]
CertificationMode = Literal["claim", "null"]
QLabel = Literal["S", "F", "R"]
StateValues = tuple[float, float, float, float, float, float]
Q_LABEL_ORDER: Final[tuple[QLabel, ...]] = ("S", "F", "R")

CONFIG: Final[dict[str, object]] = {
    "schema_version": SCHEMA_VERSION,
    "protocol_sha256": PROTOCOL_SHA256,
    "flow_limit": 8.0,
    "gradient_values": [0.0, -2.0, 2.0, -4.0, 4.0],
    "parameter_names": list(v1.PARAMETER_NAMES),
    "single_orders": {"F": ["T", "U", "V"], "R": ["V", "U", "T"]},
    "pair_orders": {"F": ["TU", "TV", "UV"], "R": ["UV", "TV", "TU"]},
    "state_count": STATE_COUNT,
    "tolerance_absolute": 1e-12,
    "tolerance_relative": 1e-10,
    "translation_candidates": [list(value) for value in v1.INITIAL_TRANSLATIONS],
}
_CONFIG_JSON: Final = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
CONFIG_SHA256: Final = hashlib.sha256(CONFIG_TAG + _CONFIG_JSON).hexdigest()


class V2ValidationError(ValueError):
    """Raised when a v2 evidence seam fails closed."""


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=np.float64, order="C", copy=True)
    output.setflags(write=False)
    return output


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=bool, order="C", copy=True)
    output.setflags(write=False)
    return output


def optimizer_tolerance(objective: float) -> float:
    """Return the frozen absolute-or-relative optimizer tolerance."""

    if not math.isfinite(objective):
        raise V2ValidationError("optimizer objective is nonfinite")
    return max(1e-12, 1e-10 * abs(objective))


@dataclass(frozen=True, slots=True)
class AffineState:
    """One point of the frozen six-parameter affine grid."""

    ty: float
    tx: float
    ayy: float
    ayx: float
    axy: float
    axx: float

    @property
    def values(self) -> StateValues:
        return (self.ty, self.tx, self.ayy, self.ayx, self.axy, self.axx)

    @property
    def canonical_key(self) -> tuple[float, float, float, float, float, float, float]:
        values = self.values
        return (sum(value * value for value in values), *values)

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


def _ordered_pairs(values: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    pairs = [(first, second) for first in values for second in values]
    return tuple(sorted(pairs, key=lambda item: (item[0] ** 2 + item[1] ** 2, *item)))


T_BLOCK: Final = tuple(
    sorted(
        ((float(y), float(x)) for y, x in v1.INITIAL_TRANSLATIONS),
        key=lambda item: (item[0] ** 2 + item[1] ** 2, *item),
    )
)
GRADIENT_VALUES: Final = (0.0, -2.0, 2.0, -4.0, 4.0)
U_BLOCK: Final = _ordered_pairs(GRADIENT_VALUES)
V_BLOCK: Final = U_BLOCK


def _make_states() -> tuple[AffineState, ...]:
    states = [AffineState(t[0], t[1], u[0], w[0], u[1], w[1]) for t, u, w in product(T_BLOCK, U_BLOCK, V_BLOCK)]
    states.sort(key=lambda state: state.canonical_key)
    if len(states) != STATE_COUNT or len({state.values for state in states}) != STATE_COUNT:
        raise RuntimeError("MM-008 v2 canonical grid is not a 15,625-state bijection")
    return tuple(states)


CANONICAL_STATES: Final = _make_states()
STATE_INDEX: Final = {state.values: index for index, state in enumerate(CANONICAL_STATES)}
CANONICAL_GRID: Final = np.ascontiguousarray(np.asarray([state.values for state in CANONICAL_STATES], dtype="<f8"))
CANONICAL_GRID.setflags(write=False)
CANDIDATE_ORDER_SHA256: Final = hashlib.sha256(CANONICAL_GRID.tobytes(order="C")).hexdigest()
ZERO_STATE_INDEX: Final = STATE_INDEX[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]

_BLOCK_COORDINATES: Final[dict[BlockName, tuple[int, ...]]] = {
    "T": (0, 1),
    "U": (2, 4),
    "V": (3, 5),
    "TU": (0, 1, 2, 4),
    "TV": (0, 1, 3, 5),
    "UV": (2, 4, 3, 5),
}
_SINGLE_ORDERS: Final[dict[StartName, tuple[BlockName, ...]]] = {
    "F": ("T", "U", "V"),
    "R": ("V", "U", "T"),
}
_PAIR_ORDERS: Final[dict[StartName, tuple[BlockName, ...]]] = {
    "F": ("TU", "TV", "UV"),
    "R": ("UV", "TV", "TU"),
}
_TERMINAL_ORDER: Final = ("T", "U", "V", "TU", "TV", "UV")


def state_index(values: StateValues | np.ndarray) -> int:
    """Return the canonical-grid index for an exact v2 state."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (6,) or not np.all(np.isfinite(raw)):
        raise V2ValidationError("state must be six finite values")
    key = cast(StateValues, tuple(float(value) for value in raw))
    try:
        return STATE_INDEX[key]
    except KeyError as error:
        raise V2ValidationError("state is not on the frozen v2 grid") from error


def neighborhood_indices(current_index: int, block: BlockName) -> tuple[int, ...]:
    """Return a block replacement neighborhood in canonical-grid index order."""

    if not 0 <= current_index < STATE_COUNT:
        raise V2ValidationError("current state index is outside the canonical grid")
    current = list(CANONICAL_STATES[current_index].values)
    coordinates = _BLOCK_COORDINATES[block]
    if block == "T":
        replacements = (tuple(value) for value in T_BLOCK)
    elif block == "U":
        replacements = (tuple(value) for value in U_BLOCK)
    elif block == "V":
        replacements = (tuple(value) for value in V_BLOCK)
    else:
        first_name = cast(BlockName, block[0])
        second_name = cast(BlockName, block[1])
        first = T_BLOCK if first_name == "T" else U_BLOCK
        second = T_BLOCK if second_name == "T" else U_BLOCK
        replacements = (tuple((*left, *right)) for left, right in product(first, second))
    indices: list[int] = []
    for replacement in replacements:
        candidate = current.copy()
        for coordinate, value in zip(coordinates, replacement, strict=True):
            candidate[coordinate] = value
        indices.append(STATE_INDEX[cast(StateValues, tuple(candidate))])
    expected = 25 if len(coordinates) == 2 else 625
    if len(indices) != expected or len(set(indices)) != expected:
        raise RuntimeError("v2 block neighborhood is not a bijection")
    return tuple(sorted(indices))


@dataclass(frozen=True, slots=True)
class FitContext:
    """Leakage-resistant objective context containing one permitted target view."""

    source: np.ndarray
    fit_target: np.ndarray
    fit_mask: np.ndarray
    output_mask: np.ndarray
    arm: Arm
    config_sha256: str

    @classmethod
    def create(
        cls,
        source: np.ndarray,
        fit_target: np.ndarray,
        fit_mask: np.ndarray,
        output_mask: np.ndarray,
        arm: Arm,
        *,
        config_sha256: str = CONFIG_SHA256,
    ) -> FitContext:
        if arm not in ("affine", "combined"):
            raise V2ValidationError("cache arm must be affine or combined")
        if config_sha256 != CONFIG_SHA256:
            raise V2ValidationError("fit-context config hash does not match this implementation")
        current = _readonly_float64(source)
        target = _readonly_float64(fit_target)
        fitted = _readonly_bool(fit_mask)
        output = _readonly_bool(output_mask)
        sites = len(v1.GEOMETRY.coords)
        if current.shape != (v1.CHANNELS, v1.NATIVE_SIZE, v1.NATIVE_SIZE):
            raise V2ValidationError("fit-context source must have shape [3,64,64]")
        if fitted.shape != (sites,) or output.shape != (sites,):
            raise V2ValidationError("fit/output masks must cover the 2,304 central sites")
        if target.shape != (v1.CHANNELS, int(np.count_nonzero(fitted))):
            raise V2ValidationError("fit target does not match the permitted fit mask")
        if not np.any(fitted) or not np.any(output):
            raise V2ValidationError("fit/output masks must be nonempty")
        full = np.all(fitted) and np.all(output)
        xfit = (
            not np.any(fitted & output)
            and np.all(fitted | output)
            and (
                (
                    np.array_equal(fitted, v1.GEOMETRY.parities == 0)
                    and np.array_equal(output, v1.GEOMETRY.parities == 1)
                )
                or (
                    np.array_equal(fitted, v1.GEOMETRY.parities == 1)
                    and np.array_equal(output, v1.GEOMETRY.parities == 0)
                )
            )
        )
        if not full and not xfit:
            raise V2ValidationError("fit/output masks are neither full nor frozen checkerboard xfit")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
            raise V2ValidationError("fit context contains nonfinite values")
        return cls(current, target, fitted, output, arm, config_sha256)

    @property
    def scope_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(CACHE_SCOPE_TAG)
        digest.update(self.source.tobytes(order="C"))
        digest.update(self.fit_target.tobytes(order="C"))
        digest.update(np.asarray(self.fit_mask, dtype=np.uint8).tobytes(order="C"))
        digest.update(np.asarray(self.output_mask, dtype=np.uint8).tobytes(order="C"))
        digest.update(self.arm.encode("ascii"))
        digest.update(bytes.fromhex(self.config_sha256))
        return digest.hexdigest()


def _validate_single_grids(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(source, dtype=np.float64)
    future = np.asarray(target, dtype=np.float64)
    required = (v1.CHANNELS, v1.NATIVE_SIZE, v1.NATIVE_SIZE)
    if current.shape != required or future.shape != required:
        raise V2ValidationError("v2 source and target must each have shape [3,64,64]")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(future)):
        raise V2ValidationError("v2 source or target contains nonfinite values")
    return current, future


def make_full_context(source: np.ndarray, target: np.ndarray, arm: Arm) -> FitContext:
    current, future = _validate_single_grids(source, target)
    mask = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    return FitContext.create(current, v1._target_values(future[None], mask)[0], mask, mask, arm)


def make_xfit_context(source: np.ndarray, target: np.ndarray, arm: Arm, *, output_parity: int) -> FitContext:
    current, future = _validate_single_grids(source, target)
    if output_parity not in (0, 1):
        raise V2ValidationError("output parity must be zero or one")
    output_mask = v1.GEOMETRY.parities == output_parity
    fit_mask = v1.GEOMETRY.parities == 1 - output_parity
    return FitContext.create(
        current,
        v1._target_values(future[None], fit_mask)[0],
        fit_mask,
        output_mask,
        arm,
    )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One memoized objective entry; inadmissible objectives are represented by ``None``."""

    state_index: int
    status: Literal["inadmissible", "valid"]
    objective: float | None
    gains: np.ndarray | None = None
    biases: np.ndarray | None = None
    retained_macros: tuple[int, ...] = ()

    @property
    def state(self) -> AffineState:
        return CANONICAL_STATES[self.state_index]

    @property
    def selection_objective(self) -> float:
        return math.inf if self.objective is None else self.objective

    def validate(self, arm: Arm) -> None:
        if not 0 <= self.state_index < STATE_COUNT:
            raise V2ValidationError("cache entry has an invalid state index")
        if self.status == "inadmissible":
            if self.objective is not None or self.gains is not None or self.biases is not None:
                raise V2ValidationError("inadmissible cache entry carries fitted values")
            return
        if self.status != "valid":
            raise V2ValidationError("cache entry has an invalid status")
        if self.objective is None or not math.isfinite(self.objective):
            raise V2ValidationError("valid cache entry has no finite objective")
        if arm == "affine":
            if self.gains is not None or self.biases is not None or self.retained_macros:
                raise V2ValidationError("affine cache entry carries appearance values")
        else:
            if self.gains is None or self.biases is None:
                raise V2ValidationError("combined cache entry lacks appearance values")
            if self.gains.shape != (v1.CHANNELS,) or self.biases.shape != (v1.CHANNELS,):
                raise V2ValidationError("combined cache appearance shape is invalid")
            if (
                self.gains.dtype != np.dtype(np.float64)
                or self.biases.dtype != np.dtype(np.float64)
                or self.gains.flags.writeable
                or self.biases.flags.writeable
                or not self.gains.flags.c_contiguous
                or not self.biases.flags.c_contiguous
            ):
                raise V2ValidationError("combined cache appearance values are not immutable C-order")
            if not np.all(np.isfinite(self.gains)) or not np.all(np.isfinite(self.biases)):
                raise V2ValidationError("combined cache appearance values are nonfinite")
            if tuple(sorted(set(self.retained_macros))) != self.retained_macros or any(
                macro not in set(int(value) for value in v1.GEOMETRY.macro_ids) for macro in self.retained_macros
            ):
                raise V2ValidationError("combined retained macro IDs are invalid")


def evaluate_candidate(context: FitContext, index: int) -> CandidateEvaluation:
    """Evaluate one frozen-grid state without reading any target outside ``context``."""

    if not 0 <= index < STATE_COUNT:
        raise V2ValidationError("candidate index is outside the canonical grid")
    state = CANONICAL_STATES[index]
    parameters = state.array()[None]
    if not bool(v1._admissible(parameters)[0]):
        result = CandidateEvaluation(index, "inadmissible", None)
        result.validate(context.arm)
        return result
    sampled = v1._sample_affine(context.source[None], parameters, context.fit_mask)
    target = context.fit_target[None]
    macro_ids = v1.GEOMETRY.macro_ids[context.fit_mask]
    if context.arm == "affine":
        objective = float(v1._macro_trimmed_loss(sampled - target, macro_ids)[0])
        result = CandidateEvaluation(index, "valid", objective)
    else:
        fitted = v1._fit_appearance(sampled, target, macro_ids)
        gains = _readonly_float64(fitted.gains[0])
        biases = _readonly_float64(fitted.biases[0])
        retained = tuple(sorted(int(value) for value in fitted.retained_macros[0]))
        result = CandidateEvaluation(
            index,
            "valid",
            float(fitted.objective[0]),
            gains,
            biases,
            retained,
        )
    result.validate(context.arm)
    return result


class ObjectiveCache:
    """Per-context deterministic objective cache with separate F/R request traces."""

    def __init__(self, context: FitContext) -> None:
        validate_fit_context(context)
        self.context = context
        self._entries: dict[int, CandidateEvaluation] = {}
        self._requests: dict[StartName, list[int]] = {"F": [], "R": []}

    @property
    def entries(self) -> dict[int, CandidateEvaluation]:
        return dict(self._entries)

    @property
    def unique_count(self) -> int:
        return len(self._entries)

    def requested_count(self, start: StartName) -> int:
        return len(self._requests[start])

    def requested_indices(self, start: StartName) -> tuple[int, ...]:
        return tuple(self._requests[start])

    def evaluate(self, index: int, *, start: StartName | None = None) -> CandidateEvaluation:
        if not 0 <= index < STATE_COUNT:
            raise V2ValidationError("cache request index is outside the canonical grid")
        if start is not None:
            self._requests[start].append(index)
        if index not in self._entries:
            if len(self._entries) >= STATE_COUNT:
                raise V2ValidationError("objective cache exceeded the finite state space")
            self._entries[index] = evaluate_candidate(self.context, index)
        return self._entries[index]

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(CACHE_CONTENT_TAG)
        digest.update(bytes.fromhex(self.context.scope_sha256))
        for index in sorted(self._entries):
            entry = self._entries[index]
            entry.validate(self.context.arm)
            digest.update(struct.pack("<H", index))
            if entry.status == "inadmissible":
                digest.update(b"\x00")
                continue
            digest.update(b"\x01")
            assert entry.objective is not None
            digest.update(struct.pack("<d", entry.objective))
            if self.context.arm == "combined":
                assert entry.gains is not None and entry.biases is not None
                digest.update(np.asarray(entry.gains, dtype="<f8").tobytes(order="C"))
                digest.update(np.asarray(entry.biases, dtype="<f8").tobytes(order="C"))
                if len(entry.retained_macros) > 255:
                    raise V2ValidationError("retained macro count does not fit uint8")
                digest.update(struct.pack("<B", len(entry.retained_macros)))
                digest.update(bytes(entry.retained_macros))
        return digest.hexdigest()

    def request_sha256(self, start: StartName) -> str:
        digest = hashlib.sha256()
        digest.update(CACHE_REQUEST_TAG)
        digest.update(bytes.fromhex(self.context.scope_sha256))
        digest.update(start.encode("ascii"))
        for index in self._requests[start]:
            digest.update(struct.pack("<H", index))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NeighborhoodRecord:
    """One exhaustive one- or two-block replacement probe."""

    block: BlockName
    current_state_index: int
    minimum_state_index: int
    current_objective: float
    minimum_objective: float
    best_improvement: float
    tolerance: float
    candidate_count: int
    admissible_count: int
    requested_indices: tuple[int, ...]
    changed: bool


@dataclass(frozen=True, slots=True)
class AcceptedState:
    """A state accepted by one independent search history."""

    stage: str
    state_index: int
    objective: float


@dataclass(frozen=True, slots=True)
class StartTrace:
    """Complete deterministic trace for one of the two zero starts."""

    start: StartName
    endpoint_state_index: int
    endpoint_objective: float
    accepted_states: tuple[AcceptedState, ...]
    objective_history: tuple[float, ...]
    neighborhoods: tuple[NeighborhoodRecord, ...]
    terminal_neighborhoods: tuple[NeighborhoodRecord, ...]
    requested_indices: tuple[int, ...]
    request_sha256: str
    requested_count: int
    unique_count: int
    certified: bool


@dataclass(frozen=True, slots=True)
class CacheSummary:
    scope_sha256: str
    content_sha256: str
    evaluated_count: int
    admissible_count: int
    inadmissible_count: int


def _same_evaluation(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    return bool(
        left.state_index == right.state_index
        and left.status == right.status
        and left.objective == right.objective
        and left.retained_macros == right.retained_macros
        and (
            (left.gains is None and right.gains is None)
            or (left.gains is not None and right.gains is not None and np.array_equal(left.gains, right.gains))
        )
        and (
            (left.biases is None and right.biases is None)
            or (left.biases is not None and right.biases is not None and np.array_equal(left.biases, right.biases))
        )
    )


def _validate_immutable_prediction(prediction: np.ndarray, context: FitContext, *, label: str) -> None:
    expected_shape = (v1.CHANNELS, int(np.count_nonzero(context.output_mask)))
    if prediction.shape != expected_shape:
        raise V2ValidationError(f"{label} prediction shape does not match output mask")
    if prediction.dtype != np.dtype(np.float64) or prediction.flags.writeable or not prediction.flags.c_contiguous:
        raise V2ValidationError(f"{label} prediction must be an immutable C-order array")
    if not np.all(np.isfinite(prediction)):
        raise V2ValidationError(f"{label} prediction contains nonfinite values")


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Certified two-order optimizer output for one leakage-resistant context."""

    protocol_sha256: str
    config_sha256: str
    candidate_order_sha256: str
    context_scope_sha256: str
    forward: StartTrace
    reverse: StartTrace
    forward_evaluation: CandidateEvaluation
    reverse_evaluation: CandidateEvaluation
    forward_prediction: np.ndarray
    reverse_prediction: np.ndarray
    selected_start: StartName
    selected_evaluation: CandidateEvaluation
    selected_prediction: np.ndarray
    objective_agreement: bool
    prediction_agreement: bool
    flow_agreement: bool
    certified: bool
    failure_codes: tuple[str, ...]
    cache: CacheSummary

    @property
    def null_certified(self) -> bool:
        """Whether both independent starts have valid terminal certificates."""

        return self.forward.certified and self.reverse.certified

    def evaluation_for(self, label: QLabel) -> CandidateEvaluation:
        if label == "S":
            return self.selected_evaluation
        if label == "F":
            return self.forward_evaluation
        if label == "R":
            return self.reverse_evaluation
        raise V2ValidationError("unknown Q endpoint label")

    def prediction_for(self, label: QLabel) -> np.ndarray:
        if label == "S":
            return self.selected_prediction
        if label == "F":
            return self.forward_prediction
        if label == "R":
            return self.reverse_prediction
        raise V2ValidationError("unknown Q prediction label")

    def validate(
        self,
        context: FitContext,
        *,
        require_certified: bool = True,
        certification_mode: CertificationMode = "claim",
    ) -> None:
        """Replay and authenticate every persisted optimizer field fail closed."""

        _validate_certification_mode(certification_mode)
        validate_fit_context(context)
        _validate_optimizer_identity(self, context)
        replayed, replay_cache = _build_optimizer_result(context)
        _validate_optimizer_live(replayed, context, replay_cache)
        _compare_optimizer_results(self, replayed, context=context)
        _require_optimizer_certification(
            self,
            certification_mode=certification_mode,
            require_certified=require_certified,
        )


def _validate_certification_mode(mode: CertificationMode) -> None:
    if mode not in ("claim", "null"):
        raise V2ValidationError("unknown optimizer certification mode")


def _validate_optimizer_identity(result: OptimizerResult, context: FitContext) -> None:
    if result.protocol_sha256 != PROTOCOL_SHA256:
        raise V2ValidationError("optimizer result protocol hash mismatch")
    if result.config_sha256 != CONFIG_SHA256:
        raise V2ValidationError("optimizer result config hash mismatch")
    if result.candidate_order_sha256 != CANDIDATE_ORDER_SHA256:
        raise V2ValidationError("optimizer result candidate-order hash mismatch")
    if result.context_scope_sha256 != context.scope_sha256:
        raise V2ValidationError("optimizer result belongs to another fit context")


def _require_optimizer_certification(
    result: OptimizerResult,
    *,
    certification_mode: CertificationMode,
    require_certified: bool,
) -> None:
    if not require_certified:
        return
    valid = result.certified if certification_mode == "claim" else result.null_certified
    if not valid:
        raise V2ValidationError(
            f"optimizer {certification_mode} context is not certified: " + ",".join(result.failure_codes)
        )


def _assert_evaluation_equal(actual: CandidateEvaluation, expected: CandidateEvaluation, *, label: str) -> None:
    if not _same_evaluation(actual, expected):
        raise V2ValidationError(f"{label} candidate evaluation differs from deterministic replay")


def _compare_optimizer_results(actual: OptimizerResult, expected: OptimizerResult, *, context: FitContext) -> None:
    """Compare every persisted field against an independently regenerated result."""

    scalar_fields = (
        "protocol_sha256",
        "config_sha256",
        "candidate_order_sha256",
        "context_scope_sha256",
        "forward",
        "reverse",
        "selected_start",
        "objective_agreement",
        "prediction_agreement",
        "flow_agreement",
        "certified",
        "failure_codes",
        "cache",
    )
    for field in scalar_fields:
        if getattr(actual, field) != getattr(expected, field):
            raise V2ValidationError(f"optimizer {field} differs from deterministic replay")
    for label, actual_evaluation, expected_evaluation in (
        ("forward", actual.forward_evaluation, expected.forward_evaluation),
        ("reverse", actual.reverse_evaluation, expected.reverse_evaluation),
        ("selected", actual.selected_evaluation, expected.selected_evaluation),
    ):
        actual_evaluation.validate(context.arm)
        expected_evaluation.validate(context.arm)
        _assert_evaluation_equal(actual_evaluation, expected_evaluation, label=label)
    for label, actual_prediction, expected_prediction in (
        ("forward", actual.forward_prediction, expected.forward_prediction),
        ("reverse", actual.reverse_prediction, expected.reverse_prediction),
        ("selected", actual.selected_prediction, expected.selected_prediction),
    ):
        if not np.array_equal(actual_prediction, expected_prediction):
            raise V2ValidationError(f"optimizer {label} prediction differs from deterministic replay")
        _validate_immutable_prediction(actual_prediction, context, label=label)


def _validate_optimizer_live(result: OptimizerResult, context: FitContext, cache: ObjectiveCache) -> None:
    """Check a newly constructed result against its live cache without replaying."""

    _validate_optimizer_identity(result, context)
    _validate_start_trace(
        result.forward,
        expected_start="F",
        scope_sha256=context.scope_sha256,
    )
    _validate_start_trace(
        result.reverse,
        expected_start="R",
        scope_sha256=context.scope_sha256,
    )
    _validate_trace_against_live_cache(result.forward, cache)
    _validate_trace_against_live_cache(result.reverse, cache)
    if result.forward.requested_indices != cache.requested_indices("F"):
        raise V2ValidationError("forward request sequence differs from live cache")
    if result.reverse.requested_indices != cache.requested_indices("R"):
        raise V2ValidationError("reverse request sequence differs from live cache")
    if result.forward.request_sha256 != cache.request_sha256("F"):
        raise V2ValidationError("forward request hash differs from live cache")
    if result.reverse.request_sha256 != cache.request_sha256("R"):
        raise V2ValidationError("reverse request hash differs from live cache")
    entries = cache.entries
    requested_union = set(result.forward.requested_indices) | set(result.reverse.requested_indices)
    if set(entries) != requested_union:
        raise V2ValidationError("optimizer cache is not the exact union of F/R requests")
    admissible_count = sum(entry.status == "valid" for entry in entries.values())
    expected_cache = CacheSummary(
        scope_sha256=context.scope_sha256,
        content_sha256=cache.content_sha256(),
        evaluated_count=len(entries),
        admissible_count=admissible_count,
        inadmissible_count=len(entries) - admissible_count,
    )
    if result.cache != expected_cache:
        raise V2ValidationError("optimizer cache summary differs from live cache")
    for label, trace, evaluation, prediction in (
        ("forward", result.forward, result.forward_evaluation, result.forward_prediction),
        ("reverse", result.reverse, result.reverse_evaluation, result.reverse_prediction),
    ):
        cached = cache.evaluate(trace.endpoint_state_index)
        _assert_evaluation_equal(evaluation, cached, label=label)
        _validate_immutable_prediction(prediction, context, label=label)
        if not np.array_equal(prediction, _prediction_for_evaluation(context, cached)):
            raise V2ValidationError(f"{label} prediction differs from live endpoint")
    forward_key = _evaluation_key(result.forward_evaluation)
    reverse_key = _evaluation_key(result.reverse_evaluation)
    expected_start: StartName = "F" if forward_key <= reverse_key else "R"
    expected_evaluation = result.forward_evaluation if expected_start == "F" else result.reverse_evaluation
    expected_prediction = result.forward_prediction if expected_start == "F" else result.reverse_prediction
    if result.selected_start != expected_start:
        raise V2ValidationError("optimizer selected-start label is not canonical")
    _assert_evaluation_equal(result.selected_evaluation, expected_evaluation, label="selected")
    _validate_immutable_prediction(result.selected_prediction, context, label="selected")
    if not np.array_equal(result.selected_prediction, expected_prediction):
        raise V2ValidationError("selected prediction differs from selected endpoint")
    assert result.selected_evaluation.objective is not None
    expected_objective_agreement = objective_order_agrees(
        result.forward.endpoint_objective,
        result.reverse.endpoint_objective,
        result.selected_evaluation.objective,
    )
    expected_prediction_agreement = bool(
        np.allclose(
            result.forward_prediction,
            result.reverse_prediction,
            rtol=0.0,
            atol=PREDICTION_AGREEMENT_ATOL,
        )
    )
    endpoint_flows = v1._affine_flow(
        np.asarray(
            [result.forward_evaluation.state.values, result.reverse_evaluation.state.values],
            dtype=np.float64,
        )
    )
    expected_flow_agreement = bool(
        np.allclose(endpoint_flows[0], endpoint_flows[1], rtol=0.0, atol=FLOW_EQUIVALENCE_ATOL)
    )
    expected_failures: list[str] = []
    if not result.forward.certified:
        expected_failures.append("forward_terminal_certificate")
    if not result.reverse.certified:
        expected_failures.append("reverse_terminal_certificate")
    if not expected_objective_agreement:
        expected_failures.append("objective_order_disagreement")
    if not expected_prediction_agreement:
        expected_failures.append("prediction_order_disagreement")
    if not expected_flow_agreement:
        expected_failures.append("flow_order_disagreement")
    if (
        result.objective_agreement != expected_objective_agreement
        or result.prediction_agreement != expected_prediction_agreement
        or result.flow_agreement != expected_flow_agreement
        or result.failure_codes != tuple(expected_failures)
        or result.certified != (not expected_failures)
    ):
        raise V2ValidationError("optimizer agreement/certification summary is inconsistent")


def _validate_trace_against_live_cache(trace: StartTrace, cache: ObjectiveCache) -> None:
    """Bind trace objectives, minima, and admissible counts to cached evaluations."""

    entries = cache.entries
    for accepted in trace.accepted_states:
        evaluation = entries.get(accepted.state_index)
        if evaluation is None or evaluation.objective != accepted.objective:
            raise V2ValidationError("optimizer accepted state differs from live cache")
    for record in (*trace.neighborhoods, *trace.terminal_neighborhoods):
        try:
            evaluations = [entries[index] for index in record.requested_indices]
        except KeyError as error:
            raise V2ValidationError("optimizer trace request is absent from live cache") from error
        valid = [evaluation for evaluation in evaluations if evaluation.status == "valid"]
        if len(valid) != record.admissible_count:
            raise V2ValidationError("optimizer admissible count differs from live cache")
        current = entries[record.current_state_index]
        if current.objective != record.current_objective:
            raise V2ValidationError("optimizer current objective differs from live cache")
        minimum = min(valid, key=_evaluation_key)
        if minimum.state_index != record.minimum_state_index or minimum.objective != record.minimum_objective:
            raise V2ValidationError("optimizer neighborhood minimum differs from live cache")


def validate_fit_context(context: FitContext) -> None:
    """Reject contexts that bypassed the copying/validation constructor."""

    rebuilt = FitContext.create(
        context.source,
        context.fit_target,
        context.fit_mask,
        context.output_mask,
        context.arm,
        config_sha256=context.config_sha256,
    )
    if rebuilt.scope_sha256 != context.scope_sha256:
        raise V2ValidationError("fit-context scope is not canonical")
    for value in (context.source, context.fit_target, context.fit_mask, context.output_mask):
        if value.flags.writeable or not value.flags.c_contiguous:
            raise V2ValidationError("fit-context arrays must be read-only C-order copies")


def _evaluation_key(
    evaluation: CandidateEvaluation,
) -> tuple[float, tuple[float, float, float, float, float, float, float]]:
    return evaluation.selection_objective, evaluation.state.canonical_key


def _probe_neighborhood(
    cache: ObjectiveCache,
    start: StartName,
    current_index: int,
    block: BlockName,
) -> NeighborhoodRecord:
    requested = neighborhood_indices(current_index, block)
    evaluations = [cache.evaluate(index, start=start) for index in requested]
    valid = [evaluation for evaluation in evaluations if evaluation.status == "valid"]
    if not valid:
        raise V2ValidationError("optimizer neighborhood has no admissible state")
    current = evaluations[requested.index(current_index)]
    if current.objective is None:
        raise V2ValidationError("optimizer reached an inadmissible current state")
    minimum = min(valid, key=_evaluation_key)
    assert minimum.objective is not None
    changed = _evaluation_key(minimum) < _evaluation_key(current)
    improvement = max(current.objective - minimum.objective, 0.0)
    return NeighborhoodRecord(
        block=block,
        current_state_index=current_index,
        minimum_state_index=minimum.state_index,
        current_objective=current.objective,
        minimum_objective=minimum.objective,
        best_improvement=improvement,
        tolerance=optimizer_tolerance(current.objective),
        candidate_count=len(requested),
        admissible_count=len(valid),
        requested_indices=requested,
        changed=changed,
    )


def _accept_state(
    cache: ObjectiveCache,
    start: StartName,
    current_index: int,
    next_index: int,
    stage: str,
    accepted: list[AcceptedState],
    seen: set[int],
) -> int:
    current = cache.evaluate(current_index)
    candidate = cache.evaluate(next_index)
    if not _evaluation_key(candidate) < _evaluation_key(current):
        raise V2ValidationError("accepted state did not strictly lower (objective,key)")
    if next_index in seen:
        raise V2ValidationError(f"{start} search repeated an accepted state")
    if len(seen) >= STATE_COUNT:
        raise V2ValidationError(f"{start} search exceeded the finite state space")
    if candidate.objective is None or not math.isfinite(candidate.objective):
        raise V2ValidationError(f"{start} search selected a nonfinite objective")
    seen.add(next_index)
    accepted.append(AcceptedState(stage, next_index, candidate.objective))
    return next_index


def _single_sweep(
    cache: ObjectiveCache,
    start: StartName,
    current_index: int,
    accepted: list[AcceptedState],
    seen: set[int],
) -> tuple[int, list[NeighborhoodRecord], bool]:
    changed = False
    records: list[NeighborhoodRecord] = []
    for block in _SINGLE_ORDERS[start]:
        record = _probe_neighborhood(cache, start, current_index, block)
        records.append(record)
        if record.changed:
            current_index = _accept_state(
                cache,
                start,
                current_index,
                record.minimum_state_index,
                block,
                accepted,
                seen,
            )
            changed = True
    return current_index, records, changed


def _converge_singles(
    cache: ObjectiveCache,
    start: StartName,
    current_index: int,
    accepted: list[AcceptedState],
    seen: set[int],
) -> tuple[int, list[NeighborhoodRecord]]:
    records: list[NeighborhoodRecord] = []
    for _ in range(STATE_COUNT):
        current_index, sweep, changed = _single_sweep(cache, start, current_index, accepted, seen)
        records.extend(sweep)
        if not changed:
            return current_index, records
    raise V2ValidationError(f"{start} single-block search did not terminate")


def _run_start(cache: ObjectiveCache, start: StartName) -> StartTrace:
    initial = cache.evaluate(ZERO_STATE_INDEX, start=start)
    if initial.objective is None or not math.isfinite(initial.objective):
        raise V2ValidationError("zero state is not a finite admissible start")
    accepted = [AcceptedState("zero", ZERO_STATE_INDEX, initial.objective)]
    seen = {ZERO_STATE_INDEX}
    all_records: list[NeighborhoodRecord] = []
    current, records = _converge_singles(cache, start, ZERO_STATE_INDEX, accepted, seen)
    all_records.extend(records)

    for _ in range(STATE_COUNT):
        escaped = False
        for block in _PAIR_ORDERS[start]:
            record = _probe_neighborhood(cache, start, current, block)
            all_records.append(record)
            if record.changed:
                current = _accept_state(
                    cache,
                    start,
                    current,
                    record.minimum_state_index,
                    block,
                    accepted,
                    seen,
                )
                current, polish = _converge_singles(cache, start, current, accepted, seen)
                all_records.extend(polish)
                escaped = True
                break
        if escaped:
            continue
        current, final_sweep, final_changed = _single_sweep(cache, start, current, accepted, seen)
        all_records.extend(final_sweep)
        if final_changed:
            current, polish = _converge_singles(cache, start, current, accepted, seen)
            all_records.extend(polish)
            continue
        break
    else:
        raise V2ValidationError(f"{start} pair-escape search did not terminate")

    terminal = tuple(_probe_neighborhood(cache, start, current, cast(BlockName, block)) for block in _TERMINAL_ORDER)
    endpoint = cache.evaluate(current)
    if endpoint.objective is None or not math.isfinite(endpoint.objective):
        raise V2ValidationError(f"{start} terminal objective is nonfinite")
    tolerance = optimizer_tolerance(endpoint.objective)
    certified = all(not record.changed and record.best_improvement <= tolerance for record in terminal)
    requested = cache.requested_indices(start)
    trace = StartTrace(
        start=start,
        endpoint_state_index=current,
        endpoint_objective=endpoint.objective,
        accepted_states=tuple(accepted),
        objective_history=tuple(item.objective for item in accepted),
        neighborhoods=tuple(all_records),
        terminal_neighborhoods=terminal,
        requested_indices=requested,
        request_sha256=cache.request_sha256(start),
        requested_count=len(requested),
        unique_count=len(set(requested)),
        certified=certified,
    )
    _validate_start_trace(
        trace,
        expected_start=start,
        scope_sha256=cache.context.scope_sha256,
    )
    return trace


def _validate_start_trace(
    trace: StartTrace,
    *,
    expected_start: StartName | None = None,
    scope_sha256: str | None = None,
) -> None:
    if trace.start not in ("F", "R"):
        raise V2ValidationError("optimizer trace has an invalid start name")
    if expected_start is not None and trace.start != expected_start:
        raise V2ValidationError("optimizer trace has the wrong F/R start label")
    if trace.requested_count != len(trace.requested_indices):
        raise V2ValidationError("optimizer request count does not match its trace")
    if trace.unique_count != len(set(trace.requested_indices)):
        raise V2ValidationError("optimizer unique request count does not match its trace")
    if any(not 0 <= index < STATE_COUNT for index in trace.requested_indices):
        raise V2ValidationError("optimizer trace requested an invalid state index")
    if not trace.accepted_states or trace.accepted_states[0].state_index != ZERO_STATE_INDEX:
        raise V2ValidationError("optimizer trace is not a zero start")
    if trace.accepted_states[0].stage != "zero":
        raise V2ValidationError("optimizer trace has an invalid zero-start stage")
    if len(trace.accepted_states) != len(trace.objective_history):
        raise V2ValidationError("optimizer accepted-state/objective history lengths differ")
    if trace.objective_history != tuple(item.objective for item in trace.accepted_states):
        raise V2ValidationError("optimizer objective history differs from accepted states")
    accepted_indices = [item.state_index for item in trace.accepted_states]
    if len(accepted_indices) != len(set(accepted_indices)) or len(accepted_indices) > STATE_COUNT:
        raise V2ValidationError("optimizer trace repeated or exceeded accepted states")
    previous_key: tuple[float, tuple[float, float, float, float, float, float, float]] | None = None
    for item in trace.accepted_states:
        if not 0 <= item.state_index < STATE_COUNT:
            raise V2ValidationError("optimizer history contains an invalid state index")
        if not math.isfinite(item.objective):
            raise V2ValidationError("optimizer history contains a nonfinite objective")
        key = item.objective, CANONICAL_STATES[item.state_index].canonical_key
        if previous_key is not None and not key < previous_key:
            raise V2ValidationError("optimizer history is not strictly lexicographically decreasing")
        previous_key = key
    if trace.endpoint_state_index != trace.accepted_states[-1].state_index:
        raise V2ValidationError("optimizer endpoint is not its last accepted state")
    if trace.endpoint_objective != trace.accepted_states[-1].objective:
        raise V2ValidationError("optimizer endpoint objective differs from its history")
    expected_blocks = cast(tuple[BlockName, ...], _TERMINAL_ORDER)
    if tuple(record.block for record in trace.terminal_neighborhoods) != expected_blocks:
        raise V2ValidationError("optimizer terminal neighborhood set is incomplete")
    current_index = ZERO_STATE_INDEX
    reconstructed_accepted = [trace.accepted_states[0]]
    all_records = (*trace.neighborhoods, *trace.terminal_neighborhoods)
    for position, record in enumerate(all_records):
        terminal_record = position >= len(trace.neighborhoods)
        if record.block not in _BLOCK_COORDINATES:
            raise V2ValidationError("optimizer neighborhood has an invalid block")
        if not 0 <= record.current_state_index < STATE_COUNT or not 0 <= (record.minimum_state_index) < STATE_COUNT:
            raise V2ValidationError("optimizer neighborhood contains an invalid state index")
        expected_current = trace.endpoint_state_index if terminal_record else current_index
        if record.current_state_index != expected_current:
            raise V2ValidationError("optimizer neighborhood history is discontinuous")
        expected_requests = neighborhood_indices(record.current_state_index, record.block)
        expected_count = 25 if len(_BLOCK_COORDINATES[record.block]) == 2 else 625
        if record.candidate_count != expected_count or len(record.requested_indices) != (expected_count):
            raise V2ValidationError("optimizer neighborhood candidate count is invalid")
        if record.requested_indices != expected_requests:
            raise V2ValidationError("optimizer neighborhood requests are not canonical")
        if record.current_state_index not in record.requested_indices:
            raise V2ValidationError("optimizer neighborhood omitted its current state")
        if record.minimum_state_index not in record.requested_indices:
            raise V2ValidationError("optimizer neighborhood minimum is outside its requests")
        if not 0 < record.admissible_count <= record.candidate_count:
            raise V2ValidationError("optimizer neighborhood admissible count is invalid")
        if not all(
            math.isfinite(value)
            for value in (
                record.current_objective,
                record.minimum_objective,
                record.best_improvement,
                record.tolerance,
            )
        ):
            raise V2ValidationError("optimizer neighborhood contains a nonfinite value")
        current_key = (
            record.current_objective,
            CANONICAL_STATES[record.current_state_index].canonical_key,
        )
        minimum_key = (
            record.minimum_objective,
            CANONICAL_STATES[record.minimum_state_index].canonical_key,
        )
        if minimum_key > current_key:
            raise V2ValidationError("optimizer neighborhood minimum is worse than current")
        expected_changed = minimum_key < current_key
        if record.changed != expected_changed:
            raise V2ValidationError("optimizer neighborhood changed flag is inconsistent")
        if record.best_improvement != max(record.current_objective - record.minimum_objective, 0.0):
            raise V2ValidationError("optimizer neighborhood improvement is inconsistent")
        if record.tolerance != optimizer_tolerance(record.current_objective):
            raise V2ValidationError("optimizer neighborhood tolerance is inconsistent")
        if record.changed and not terminal_record:
            current_index = record.minimum_state_index
            reconstructed_accepted.append(AcceptedState(record.block, current_index, record.minimum_objective))

    if tuple(reconstructed_accepted) != trace.accepted_states:
        raise V2ValidationError("optimizer accepted states differ from neighborhood history")
    if current_index != trace.endpoint_state_index:
        raise V2ValidationError("optimizer terminal neighborhoods changed the endpoint")
    endpoint_tolerance = optimizer_tolerance(trace.endpoint_objective)
    expected_certified = all(
        not record.changed and record.best_improvement <= endpoint_tolerance for record in trace.terminal_neighborhoods
    )
    if trace.certified != expected_certified:
        raise V2ValidationError("optimizer terminal certification is inconsistent")

    expected_requested = (
        (ZERO_STATE_INDEX,)
        + tuple(index for record in trace.neighborhoods for index in record.requested_indices)
        + tuple(index for record in trace.terminal_neighborhoods for index in record.requested_indices)
    )
    if trace.requested_indices != expected_requested:
        raise V2ValidationError("optimizer request trace differs from neighborhood history")
    if scope_sha256 is not None:
        try:
            scope = bytes.fromhex(scope_sha256)
        except ValueError as error:
            raise V2ValidationError("optimizer request scope hash is invalid") from error
        if len(scope) != 32:
            raise V2ValidationError("optimizer request scope hash is invalid")
        digest = hashlib.sha256()
        digest.update(CACHE_REQUEST_TAG)
        digest.update(scope)
        digest.update(trace.start.encode("ascii"))
        for index in trace.requested_indices:
            digest.update(struct.pack("<H", index))
        if trace.request_sha256 != digest.hexdigest():
            raise V2ValidationError("optimizer request hash differs from its trace")


def _prediction_for_evaluation(context: FitContext, evaluation: CandidateEvaluation) -> np.ndarray:
    if evaluation.status != "valid":
        raise V2ValidationError("cannot predict from an inadmissible evaluation")
    sampled = v1._sample_affine(context.source[None], evaluation.state.array()[None], context.output_mask)[0]
    if context.arm == "combined":
        assert evaluation.gains is not None and evaluation.biases is not None
        prediction = evaluation.gains[:, None] * sampled + evaluation.biases[:, None]
    else:
        prediction = sampled
    if not np.all(np.isfinite(prediction)):
        raise V2ValidationError("optimizer prediction contains nonfinite values")
    return _readonly_float64(prediction)


def objective_order_agrees(
    forward_objective: float,
    reverse_objective: float,
    selected_objective: float,
) -> bool:
    """Apply the selected-final-objective tolerance to F/R agreement."""

    if not all(math.isfinite(value) for value in (forward_objective, reverse_objective, selected_objective)):
        raise V2ValidationError("order-agreement objective is nonfinite")
    return abs(forward_objective - reverse_objective) <= optimizer_tolerance(selected_objective)


def _build_optimizer_result(context: FitContext) -> tuple[OptimizerResult, ObjectiveCache]:
    """Construct one optimizer result and retain its live cache for private validation."""

    validate_fit_context(context)
    cache = ObjectiveCache(context)
    forward = _run_start(cache, "F")
    if forward.request_sha256 != cache.request_sha256("F"):
        raise V2ValidationError("forward request trace hash drifted")
    reverse = _run_start(cache, "R")
    if reverse.request_sha256 != cache.request_sha256("R"):
        raise V2ValidationError("reverse request trace hash drifted")

    forward_evaluation = cache.evaluate(forward.endpoint_state_index)
    reverse_evaluation = cache.evaluate(reverse.endpoint_state_index)
    selected_start: StartName = (
        "F" if _evaluation_key(forward_evaluation) <= _evaluation_key(reverse_evaluation) else "R"
    )
    selected = forward_evaluation if selected_start == "F" else reverse_evaluation
    forward_prediction = _prediction_for_evaluation(context, forward_evaluation)
    reverse_prediction = _prediction_for_evaluation(context, reverse_evaluation)
    if selected.objective is None:
        raise V2ValidationError("selected endpoint has no finite objective")
    objective_agreement = objective_order_agrees(
        forward.endpoint_objective,
        reverse.endpoint_objective,
        selected.objective,
    )
    prediction_agreement = bool(
        np.allclose(
            forward_prediction,
            reverse_prediction,
            rtol=0.0,
            atol=PREDICTION_AGREEMENT_ATOL,
        )
    )
    forward_state = CANONICAL_STATES[forward.endpoint_state_index]
    reverse_state = CANONICAL_STATES[reverse.endpoint_state_index]
    if forward_state.values == reverse_state.values:
        flow_agreement = True
    else:
        flows = v1._affine_flow(np.asarray([forward_state.values, reverse_state.values], dtype=np.float64))
        flow_agreement = bool(np.allclose(flows[0], flows[1], rtol=0.0, atol=FLOW_EQUIVALENCE_ATOL))
    failures: list[str] = []
    if not forward.certified:
        failures.append("forward_terminal_certificate")
    if not reverse.certified:
        failures.append("reverse_terminal_certificate")
    if not objective_agreement:
        failures.append("objective_order_disagreement")
    if not prediction_agreement:
        failures.append("prediction_order_disagreement")
    if not flow_agreement:
        failures.append("flow_order_disagreement")
    entries = cache.entries
    admissible_count = sum(entry.status == "valid" for entry in entries.values())
    summary = CacheSummary(
        scope_sha256=context.scope_sha256,
        content_sha256=cache.content_sha256(),
        evaluated_count=len(entries),
        admissible_count=admissible_count,
        inadmissible_count=len(entries) - admissible_count,
    )
    result = OptimizerResult(
        protocol_sha256=PROTOCOL_SHA256,
        config_sha256=CONFIG_SHA256,
        candidate_order_sha256=CANDIDATE_ORDER_SHA256,
        context_scope_sha256=context.scope_sha256,
        forward=forward,
        reverse=reverse,
        forward_evaluation=forward_evaluation,
        reverse_evaluation=reverse_evaluation,
        forward_prediction=forward_prediction,
        reverse_prediction=reverse_prediction,
        selected_start=selected_start,
        selected_evaluation=selected,
        selected_prediction=_prediction_for_evaluation(context, selected),
        objective_agreement=objective_agreement,
        prediction_agreement=prediction_agreement,
        flow_agreement=flow_agreement,
        certified=not failures,
        failure_codes=tuple(failures),
        cache=summary,
    )
    return result, cache


def fit_optimizer(
    context: FitContext,
    *,
    require_certified: bool = True,
    certification_mode: CertificationMode = "claim",
) -> OptimizerResult:
    """Run the exact F/R all-pairs local search for one fit-only context."""

    _validate_certification_mode(certification_mode)
    result, cache = _build_optimizer_result(context)
    _validate_optimizer_live(result, context, cache)
    _require_optimizer_certification(
        result,
        certification_mode=certification_mode,
        require_certified=require_certified,
    )
    return result


def fit_null_optimizer(context: FitContext) -> OptimizerResult:
    """Fit a declared wrong/null context, requiring both terminal certificates."""

    return fit_optimizer(context, certification_mode="null")


@dataclass(frozen=True, slots=True)
class CarryEstimate:
    """One immutable combined-family nesting witness."""

    kind: Literal["affine_carry", "appearance_carry"]
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    retained_macros: tuple[int, ...]
    prediction: np.ndarray
    fit_objective: float | None


@dataclass(frozen=True, slots=True)
class DirectionEstimate:
    """A full fit or one independently fitted checkerboard output direction."""

    context: FitContext
    optimizer: OptimizerResult
    certification_mode: CertificationMode
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    retained_macros: tuple[int, ...]
    prediction: np.ndarray
    affine_carry: CarryEstimate | None
    appearance_carry: CarryEstimate | None

    @property
    def certified(self) -> bool:
        if self.certification_mode == "claim":
            return self.optimizer.certified
        if self.certification_mode == "null":
            return self.optimizer.null_certified
        raise V2ValidationError("unknown optimizer certification mode")

    def evaluation_for(self, label: QLabel) -> CandidateEvaluation:
        return self.optimizer.evaluation_for(label)

    def prediction_for(self, label: QLabel) -> np.ndarray:
        return self.optimizer.prediction_for(label)

    def validate(self, *, require_certified: bool = True) -> None:
        _validate_certification_mode(self.certification_mode)
        self.optimizer.validate(
            self.context,
            require_certified=require_certified,
            certification_mode=self.certification_mode,
        )
        _validate_direction_fields(self)


def _validate_immutable_float64_array(value: np.ndarray, expected_shape: tuple[int, ...], *, label: str) -> None:
    if value.shape != expected_shape:
        raise V2ValidationError(f"{label} shape is invalid")
    if value.dtype != np.dtype(np.float64) or value.flags.writeable or not value.flags.c_contiguous:
        raise V2ValidationError(f"{label} must be an immutable float64 C-order array")
    if not np.all(np.isfinite(value)):
        raise V2ValidationError(f"{label} contains nonfinite values")


def _compare_carry_estimates(
    actual: CarryEstimate,
    expected: CarryEstimate,
    context: FitContext,
) -> None:
    if (
        actual.kind != expected.kind
        or actual.retained_macros != expected.retained_macros
        or actual.fit_objective != expected.fit_objective
    ):
        raise V2ValidationError(f"{expected.kind} metadata differs from deterministic replay")
    _validate_immutable_float64_array(actual.parameters, (6,), label=f"{expected.kind} parameters")
    _validate_immutable_float64_array(actual.gains, (v1.CHANNELS,), label=f"{expected.kind} gains")
    _validate_immutable_float64_array(actual.biases, (v1.CHANNELS,), label=f"{expected.kind} biases")
    _validate_immutable_prediction(actual.prediction, context, label=expected.kind)
    for field in ("parameters", "gains", "biases", "prediction"):
        if not np.array_equal(getattr(actual, field), getattr(expected, field)):
            raise V2ValidationError(f"{expected.kind} {field} differs from deterministic replay")


def _validate_direction_fields(direction: DirectionEstimate) -> None:
    """Validate direction aggregates and carries without rerunning its optimizer."""

    _validate_certification_mode(direction.certification_mode)
    validate_fit_context(direction.context)
    _validate_optimizer_identity(direction.optimizer, direction.context)
    _validate_immutable_float64_array(direction.parameters, (6,), label="direction parameters")
    _validate_immutable_float64_array(direction.gains, (v1.CHANNELS,), label="direction gains")
    _validate_immutable_float64_array(direction.biases, (v1.CHANNELS,), label="direction biases")
    _validate_immutable_prediction(direction.prediction, direction.context, label="direction")
    selected = direction.optimizer.selected_evaluation
    selected.validate(direction.context.arm)
    if not np.array_equal(direction.parameters, selected.state.array()):
        raise V2ValidationError("direction parameters differ from optimizer endpoint")
    if not np.array_equal(direction.prediction, direction.optimizer.selected_prediction):
        raise V2ValidationError("direction prediction differs from optimizer endpoint")
    if direction.context.arm == "affine":
        if (
            not np.array_equal(direction.gains, np.ones(v1.CHANNELS))
            or not np.array_equal(direction.biases, np.zeros(v1.CHANNELS))
            or direction.retained_macros
        ):
            raise V2ValidationError("affine direction has nonidentity appearance")
        if direction.affine_carry is not None or direction.appearance_carry is not None:
            raise V2ValidationError("affine direction unexpectedly contains combined carries")
        return
    if direction.affine_carry is None or direction.appearance_carry is None:
        raise V2ValidationError("combined direction is missing nesting carries")
    assert selected.gains is not None and selected.biases is not None
    if (
        not np.array_equal(direction.gains, selected.gains)
        or not np.array_equal(direction.biases, selected.biases)
        or direction.retained_macros != selected.retained_macros
    ):
        raise V2ValidationError("combined direction differs from selected appearance fit")
    expected_affine, expected_appearance = _combined_carries(direction.context, selected)
    _compare_carry_estimates(direction.affine_carry, expected_affine, direction.context)
    _compare_carry_estimates(direction.appearance_carry, expected_appearance, direction.context)


@dataclass(frozen=True, slots=True)
class QPanel:
    """One complete public S/F/R prediction panel."""

    label: QLabel
    prediction: np.ndarray


@dataclass(frozen=True, slots=True)
class QMinimum:
    """Frozen-order minimum over a complete S/F/R scalar mapping."""

    label: QLabel
    value: float


def select_q_minimum(values: Mapping[QLabel, float]) -> QMinimum:
    """Select a finite S/F/R minimum with frozen tie order S,F,R."""

    if set(values) != set(Q_LABEL_ORDER):
        raise V2ValidationError("iterative Q values must contain exactly S, F, and R")
    for value in values.values():
        if not math.isfinite(value):
            raise V2ValidationError("Q value is nonfinite")
    label = min(Q_LABEL_ORDER, key=lambda item: (values[item], Q_LABEL_ORDER.index(item)))
    return QMinimum(label, float(values[label]))


@dataclass(frozen=True, slots=True)
class Estimate:
    """One full or two-direction checkerboard-cross-fitted v2 estimate."""

    arm: Arm
    mode: Literal["full", "xfit"]
    certification_mode: CertificationMode
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    prediction: np.ndarray
    objectives: np.ndarray
    directions: tuple[DirectionEstimate, ...]

    @property
    def certified(self) -> bool:
        _validate_certification_mode(self.certification_mode)
        if self.arm not in ("affine", "combined"):
            raise V2ValidationError("estimate has an invalid arm")
        expected_count = 1 if self.mode == "full" else 2 if self.mode == "xfit" else 0
        if expected_count == 0 or len(self.directions) != expected_count:
            raise V2ValidationError("estimate has an invalid mode or direction count")
        return all(direction.certified for direction in self.directions)

    def prediction_for(self, label: QLabel) -> np.ndarray:
        """Assemble one complete S/F/R panel without refitting or private access."""

        if label not in Q_LABEL_ORDER:
            raise V2ValidationError("unknown Q prediction label")
        expected_count = 1 if self.mode == "full" else 2 if self.mode == "xfit" else 0
        if expected_count == 0 or len(self.directions) != expected_count:
            raise V2ValidationError("estimate has an invalid mode or direction count")
        panel = np.full((v1.CHANNELS, len(v1.GEOMETRY.coords)), np.nan, dtype=np.float64)
        for direction in self.directions:
            panel[:, direction.context.output_mask] = direction.prediction_for(label)
        if not np.all(np.isfinite(panel)):
            raise V2ValidationError("Q panel did not predict every central site")
        return _readonly_float64(panel)

    def q_panels(self) -> tuple[QPanel, ...]:
        return tuple(QPanel(label, self.prediction_for(label)) for label in Q_LABEL_ORDER)

    def validate(self, *, require_certified: bool = True) -> None:
        """Authenticate aggregate fields and deeply replay every fitted direction."""

        _validate_estimate_fields(self)
        for direction in self.directions:
            direction.validate(require_certified=require_certified)


def _validate_estimate_fields(estimate: Estimate) -> None:
    """Validate estimate roles, masks, aggregates, and Q panels without refitting."""

    _validate_certification_mode(estimate.certification_mode)
    if estimate.arm not in ("affine", "combined"):
        raise V2ValidationError("estimate has an invalid arm")
    if estimate.mode == "full":
        expected_count = 1
    elif estimate.mode == "xfit":
        expected_count = 2
    else:
        raise V2ValidationError("estimate has an invalid mode")
    if len(estimate.directions) != expected_count:
        raise V2ValidationError("estimate direction count does not match its mode")

    reference_source = estimate.directions[0].context.source
    for index, direction in enumerate(estimate.directions):
        _validate_direction_fields(direction)
        if direction.context.arm != estimate.arm:
            raise V2ValidationError("estimate direction arm is incompatible")
        if direction.certification_mode != estimate.certification_mode:
            raise V2ValidationError("estimate direction certification role is incompatible")
        if direction.context.config_sha256 != CONFIG_SHA256:
            raise V2ValidationError("estimate direction config is incompatible")
        if not np.array_equal(direction.context.source, reference_source):
            raise V2ValidationError("estimate directions do not share an exact source")
        if estimate.mode == "full":
            expected_output = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
            expected_fit = expected_output
        else:
            expected_output = v1.GEOMETRY.parities == index
            expected_fit = v1.GEOMETRY.parities == 1 - index
        if not np.array_equal(direction.context.output_mask, expected_output) or not (
            np.array_equal(direction.context.fit_mask, expected_fit)
        ):
            raise V2ValidationError("estimate direction masks are incompatible with its mode")

    _validate_immutable_float64_array(estimate.parameters, (expected_count, 6), label="estimate parameters")
    _validate_immutable_float64_array(
        estimate.gains,
        (expected_count, v1.CHANNELS),
        label="estimate gains",
    )
    _validate_immutable_float64_array(
        estimate.biases,
        (expected_count, v1.CHANNELS),
        label="estimate biases",
    )
    _validate_immutable_float64_array(estimate.objectives, (expected_count,), label="estimate objectives")
    _validate_immutable_float64_array(
        estimate.prediction,
        (v1.CHANNELS, len(v1.GEOMETRY.coords)),
        label="estimate prediction",
    )
    expected_parameters = np.stack([direction.parameters for direction in estimate.directions])
    expected_gains = np.stack([direction.gains for direction in estimate.directions])
    expected_biases = np.stack([direction.biases for direction in estimate.directions])
    objective_values: list[float] = []
    for direction in estimate.directions:
        objective = direction.optimizer.selected_evaluation.objective
        if objective is None:
            raise V2ValidationError("estimate direction has no finite objective")
        objective_values.append(objective)
    expected_objectives = np.asarray(objective_values, dtype=np.float64)
    if (
        not np.array_equal(estimate.parameters, expected_parameters)
        or not np.array_equal(estimate.gains, expected_gains)
        or not np.array_equal(estimate.biases, expected_biases)
        or not np.array_equal(estimate.objectives, expected_objectives)
    ):
        raise V2ValidationError("estimate aggregates differ from fitted directions")
    for panel in estimate.q_panels():
        _validate_immutable_float64_array(
            panel.prediction,
            (v1.CHANNELS, len(v1.GEOMETRY.coords)),
            label=f"estimate Q[{panel.label}] prediction",
        )
        if panel.label == "S" and not np.array_equal(estimate.prediction, panel.prediction):
            raise V2ValidationError("estimate prediction differs from its selected Q panel")


def _validate_estimate_live(estimate: Estimate, *, require_certified: bool) -> None:
    """Validate a newly constructed estimate without replaying any optimizer."""

    _validate_estimate_fields(estimate)
    for direction in estimate.directions:
        _require_optimizer_certification(
            direction.optimizer,
            certification_mode=direction.certification_mode,
            require_certified=require_certified,
        )


def _combined_carries(context: FitContext, selected: CandidateEvaluation) -> tuple[CarryEstimate, CarryEstimate]:
    selected_parameters = selected.state.array()
    affine_prediction = v1._sample_affine(context.source[None], selected_parameters[None], context.output_mask)[0]
    affine = CarryEstimate(
        kind="affine_carry",
        parameters=_readonly_float64(selected_parameters),
        gains=_readonly_float64(np.ones(v1.CHANNELS)),
        biases=_readonly_float64(np.zeros(v1.CHANNELS)),
        retained_macros=(),
        prediction=_readonly_float64(affine_prediction),
        fit_objective=None,
    )

    zero_parameters = np.zeros((1, 6), dtype=np.float64)
    fit_sampled = v1._sample_affine(context.source[None], zero_parameters, context.fit_mask)
    appearance_fit = v1._fit_appearance(
        fit_sampled,
        context.fit_target[None],
        v1.GEOMETRY.macro_ids[context.fit_mask],
    )
    output_sampled = v1._sample_affine(context.source[None], zero_parameters, context.output_mask)[0]
    gains = appearance_fit.gains[0]
    biases = appearance_fit.biases[0]
    appearance_prediction = gains[:, None] * output_sampled + biases[:, None]
    appearance = CarryEstimate(
        kind="appearance_carry",
        parameters=_readonly_float64(zero_parameters[0]),
        gains=_readonly_float64(gains),
        biases=_readonly_float64(biases),
        retained_macros=tuple(sorted(int(value) for value in appearance_fit.retained_macros[0])),
        prediction=_readonly_float64(appearance_prediction),
        fit_objective=float(appearance_fit.objective[0]),
    )
    return affine, appearance


def fit_direction(
    context: FitContext,
    *,
    require_certified: bool = True,
    certification_mode: CertificationMode = "claim",
) -> DirectionEstimate:
    """Fit and materialize one context, including combined nesting witnesses."""

    _validate_certification_mode(certification_mode)
    optimizer = fit_optimizer(
        context,
        require_certified=require_certified,
        certification_mode=certification_mode,
    )
    selected = optimizer.selected_evaluation
    parameters = _readonly_float64(selected.state.array())
    if context.arm == "combined":
        assert selected.gains is not None and selected.biases is not None
        gains = _readonly_float64(selected.gains)
        biases = _readonly_float64(selected.biases)
        affine_carry, appearance_carry = _combined_carries(context, selected)
    else:
        gains = _readonly_float64(np.ones(v1.CHANNELS))
        biases = _readonly_float64(np.zeros(v1.CHANNELS))
        affine_carry = None
        appearance_carry = None
    result = DirectionEstimate(
        context=context,
        optimizer=optimizer,
        certification_mode=certification_mode,
        parameters=parameters,
        gains=gains,
        biases=biases,
        retained_macros=selected.retained_macros,
        prediction=optimizer.selected_prediction,
        affine_carry=affine_carry,
        appearance_carry=appearance_carry,
    )
    _validate_direction_fields(result)
    _require_optimizer_certification(
        optimizer,
        certification_mode=certification_mode,
        require_certified=require_certified,
    )
    return result


def estimate_full(
    source: np.ndarray,
    target: np.ndarray,
    arm: Arm,
    *,
    require_certified: bool = True,
    certification_mode: CertificationMode = "claim",
) -> Estimate:
    if arm not in ("affine", "combined"):
        raise V2ValidationError("estimate arm must be affine or combined")
    _validate_certification_mode(certification_mode)
    context = make_full_context(source, target, arm)
    direction = fit_direction(
        context,
        require_certified=require_certified,
        certification_mode=certification_mode,
    )
    selected = direction.optimizer.selected_evaluation
    assert selected.objective is not None
    result = Estimate(
        arm=arm,
        mode="full",
        certification_mode=certification_mode,
        parameters=_readonly_float64(direction.parameters[None]),
        gains=_readonly_float64(direction.gains[None]),
        biases=_readonly_float64(direction.biases[None]),
        prediction=direction.prediction,
        objectives=_readonly_float64(np.asarray([selected.objective])),
        directions=(direction,),
    )
    _validate_estimate_live(result, require_certified=require_certified)
    return result


def estimate_xfit(
    source: np.ndarray,
    target: np.ndarray,
    arm: Arm,
    *,
    require_certified: bool = True,
    certification_mode: CertificationMode = "claim",
) -> Estimate:
    if arm not in ("affine", "combined"):
        raise V2ValidationError("estimate arm must be affine or combined")
    _validate_certification_mode(certification_mode)
    directions = tuple(
        fit_direction(
            make_xfit_context(source, target, arm, output_parity=parity),
            require_certified=require_certified,
            certification_mode=certification_mode,
        )
        for parity in (0, 1)
    )
    prediction = np.full((v1.CHANNELS, len(v1.GEOMETRY.coords)), np.nan, dtype=np.float64)
    for parity, direction in enumerate(directions):
        prediction[:, v1.GEOMETRY.parities == parity] = direction.prediction
    if not np.all(np.isfinite(prediction)):
        raise V2ValidationError("cross-fit did not predict every central site")
    objectives = []
    for direction in directions:
        objective = direction.optimizer.selected_evaluation.objective
        assert objective is not None
        objectives.append(objective)
    result = Estimate(
        arm=arm,
        mode="xfit",
        certification_mode=certification_mode,
        parameters=_readonly_float64(np.stack([direction.parameters for direction in directions])),
        gains=_readonly_float64(np.stack([direction.gains for direction in directions])),
        biases=_readonly_float64(np.stack([direction.biases for direction in directions])),
        prediction=_readonly_float64(prediction),
        objectives=_readonly_float64(np.asarray(objectives)),
        directions=cast(tuple[DirectionEstimate, ...], directions),
    )
    _validate_estimate_live(result, require_certified=require_certified)
    return result


def estimate_null_xfit(source: np.ndarray, target: np.ndarray, arm: Arm) -> Estimate:
    """Fit a complete declared wrong/null S/F/R cross-fitted panel."""

    return estimate_xfit(source, target, arm, certification_mode="null")


@dataclass(frozen=True, slots=True)
class BiasDirectionEstimate:
    """Two-pass target-only mean comparator for one permitted target view."""

    fit_mask: np.ndarray
    output_mask: np.ndarray
    first_biases: np.ndarray
    biases: np.ndarray
    retained_macros: tuple[int, ...]
    objective: float
    prediction: np.ndarray


@dataclass(frozen=True, slots=True)
class BiasEstimate:
    mode: Literal["full", "xfit"]
    biases: np.ndarray
    prediction: np.ndarray
    objectives: np.ndarray
    directions: tuple[BiasDirectionEstimate, ...]


def fit_bias_only(fit_target: np.ndarray, fit_mask: np.ndarray, output_mask: np.ndarray) -> BiasDirectionEstimate:
    """Fit the frozen two-pass bias-only comparator with no third trim."""

    target = np.asarray(fit_target, dtype=np.float64)
    fitted = np.asarray(fit_mask, dtype=bool)
    output = np.asarray(output_mask, dtype=bool)
    sites = len(v1.GEOMETRY.coords)
    if fitted.shape != (sites,) or output.shape != (sites,):
        raise V2ValidationError("bias fit/output masks must cover the central grid")
    if target.shape != (v1.CHANNELS, int(np.count_nonzero(fitted))):
        raise V2ValidationError("bias fit target does not match its fit mask")
    if not np.all(np.isfinite(target)) or not np.any(fitted) or not np.any(output):
        raise V2ValidationError("bias-only context is empty or nonfinite")
    macro_ids = v1.GEOMETRY.macro_ids[fitted]
    first_biases = np.clip(np.mean(target, axis=1, dtype=np.float64), *v1.BIAS_BOUNDS)
    first_residual = first_biases[None, :, None] - target[None]
    macros, first_losses = v1._macro_losses(first_residual, macro_ids)
    keep = len(macros) - math.floor(len(macros) * v1.TRIM_FRACTION)
    order = np.argsort(first_losses[0], kind="stable")[:keep]
    retained_in_rank_order = macros[order]
    retained_pixels = np.isin(macro_ids, retained_in_rank_order)
    if not np.any(retained_pixels):
        raise V2ValidationError("bias-only pass one retained no pixels")
    biases = np.clip(np.mean(target[:, retained_pixels], axis=1, dtype=np.float64), *v1.BIAS_BOUNDS)
    final_residual = biases[None, :, None] - target[None]
    final_macros, final_losses = v1._macro_losses(final_residual, macro_ids)
    loss_by_macro = {int(macro): float(final_losses[0, index]) for index, macro in enumerate(final_macros)}
    objective = float(np.mean([loss_by_macro[int(macro)] for macro in retained_in_rank_order]))
    if not math.isfinite(objective):
        raise V2ValidationError("bias-only objective is nonfinite")
    prediction = np.broadcast_to(biases[:, None], (v1.CHANNELS, int(np.count_nonzero(output)))).copy()
    return BiasDirectionEstimate(
        fit_mask=_readonly_bool(fitted),
        output_mask=_readonly_bool(output),
        first_biases=_readonly_float64(first_biases),
        biases=_readonly_float64(biases),
        retained_macros=tuple(sorted(int(value) for value in retained_in_rank_order)),
        objective=objective,
        prediction=_readonly_float64(prediction),
    )


def estimate_bias_full(target: np.ndarray) -> BiasEstimate:
    future = np.asarray(target, dtype=np.float64)
    if future.shape != (v1.CHANNELS, v1.NATIVE_SIZE, v1.NATIVE_SIZE):
        raise V2ValidationError("bias-only target must have shape [3,64,64]")
    mask = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    values = v1._target_values(future[None], mask)[0]
    direction = fit_bias_only(values, mask, mask)
    return BiasEstimate(
        mode="full",
        biases=direction.biases[None],
        prediction=direction.prediction,
        objectives=_readonly_float64(np.asarray([direction.objective])),
        directions=(direction,),
    )


def estimate_bias_xfit(target: np.ndarray) -> BiasEstimate:
    future = np.asarray(target, dtype=np.float64)
    if future.shape != (v1.CHANNELS, v1.NATIVE_SIZE, v1.NATIVE_SIZE):
        raise V2ValidationError("bias-only target must have shape [3,64,64]")
    directions = []
    prediction = np.full((v1.CHANNELS, len(v1.GEOMETRY.coords)), np.nan, dtype=np.float64)
    for parity in (0, 1):
        output_mask = v1.GEOMETRY.parities == parity
        fit_mask = v1.GEOMETRY.parities == 1 - parity
        fit_target = v1._target_values(future[None], fit_mask)[0]
        direction = fit_bias_only(fit_target, fit_mask, output_mask)
        directions.append(direction)
        prediction[:, output_mask] = direction.prediction
    if not np.all(np.isfinite(prediction)):
        raise V2ValidationError("bias-only cross-fit did not predict every central site")
    return BiasEstimate(
        mode="xfit",
        biases=_readonly_float64(np.stack([item.biases for item in directions])),
        prediction=_readonly_float64(prediction),
        objectives=_readonly_float64(np.asarray([item.objective for item in directions])),
        directions=tuple(directions),
    )


@dataclass(frozen=True, slots=True)
class ExhaustiveOracle:
    """Global frozen-grid enumeration record for a declared synthetic context."""

    context_scope_sha256: str
    candidate_order_sha256: str
    candidate_count: int
    admissible_count: int
    selected_state_index: int
    selected_admissible_rank: int
    truth_state_index: int
    truth_admissible_rank: int
    minimum_objective: float
    truth_objective: float
    second_best_nonequivalent_gap: float
    cache_content_sha256: str

    @property
    def selected_total_rank(self) -> int:
        """Zero-based rank in the complete canonical list."""

        return self.selected_state_index

    @property
    def truth_total_rank(self) -> int:
        """Injected truth's zero-based rank in the complete canonical list."""

        return self.truth_state_index

    def validate(
        self,
        context: FitContext,
        *,
        require_truth: bool = True,
        require_separated_flow: bool = True,
    ) -> None:
        if self.context_scope_sha256 != context.scope_sha256:
            raise V2ValidationError("exhaustive oracle scope mismatch")
        if self.candidate_order_sha256 != CANDIDATE_ORDER_SHA256:
            raise V2ValidationError("exhaustive oracle candidate-order hash mismatch")
        if self.candidate_count != STATE_COUNT:
            raise V2ValidationError("exhaustive oracle did not enumerate the full grid")
        if not 0 < self.admissible_count <= STATE_COUNT:
            raise V2ValidationError("exhaustive oracle admissible count is invalid")
        if not math.isfinite(self.minimum_objective) or not math.isfinite(self.truth_objective):
            raise V2ValidationError("exhaustive oracle objective is nonfinite")
        if require_truth and self.selected_state_index != self.truth_state_index:
            raise V2ValidationError("exhaustive global minimum does not recover injected truth")
        if self.minimum_objective > self.truth_objective + optimizer_tolerance(self.minimum_objective):
            raise V2ValidationError("exhaustive minimum is worse than direct truth")
        if require_separated_flow and self.second_best_nonequivalent_gap <= 1e-12:
            raise V2ValidationError("exhaustive minimum lacks a separated flow endpoint")


def exhaustive_oracle(
    context: FitContext,
    truth: StateValues | np.ndarray,
    *,
    optimizer: OptimizerResult | None = None,
    require_truth: bool = True,
    require_separated_flow: bool = True,
) -> ExhaustiveOracle:
    """Enumerate all 15,625 states and certify the first global lexicographic minimum."""

    validate_fit_context(context)
    truth_index = state_index(truth)
    cache = ObjectiveCache(context)
    evaluations = [cache.evaluate(index) for index in range(STATE_COUNT)]
    valid = [item for item in evaluations if item.status == "valid"]
    if not valid:
        raise V2ValidationError("exhaustive oracle found no admissible state")
    selected = min(valid, key=_evaluation_key)
    assert selected.objective is not None
    truth_evaluation = evaluations[truth_index]
    if truth_evaluation.objective is None:
        raise V2ValidationError("exhaustive-oracle truth is inadmissible")
    admissible_indices = [item.state_index for item in valid]
    selected_rank = admissible_indices.index(selected.state_index)
    truth_rank = admissible_indices.index(truth_index)
    selected_flow = v1._affine_flow(selected.state.array()[None])[0]
    second_gap = math.inf
    for candidate in sorted(valid, key=_evaluation_key):
        candidate_flow = v1._affine_flow(candidate.state.array()[None])[0]
        if not np.allclose(
            candidate_flow,
            selected_flow,
            rtol=0.0,
            atol=FLOW_EQUIVALENCE_ATOL,
        ):
            assert candidate.objective is not None
            second_gap = candidate.objective - selected.objective
            break
    if not math.isfinite(second_gap):
        raise V2ValidationError("exhaustive oracle found no non-equivalent flow")
    result = ExhaustiveOracle(
        context_scope_sha256=context.scope_sha256,
        candidate_order_sha256=CANDIDATE_ORDER_SHA256,
        candidate_count=len(evaluations),
        admissible_count=len(valid),
        selected_state_index=selected.state_index,
        selected_admissible_rank=selected_rank,
        truth_state_index=truth_index,
        truth_admissible_rank=truth_rank,
        minimum_objective=selected.objective,
        truth_objective=truth_evaluation.objective,
        second_best_nonequivalent_gap=second_gap,
        cache_content_sha256=cache.content_sha256(),
    )
    result.validate(
        context,
        require_truth=require_truth,
        require_separated_flow=require_separated_flow,
    )
    if optimizer is not None:
        optimizer.validate(context)
        if optimizer.selected_evaluation.state_index != result.selected_state_index:
            raise V2ValidationError("optimizer endpoint differs from exhaustive global minimum")
    return result
