"""Exact canonical-grid estimator and certificate core for MM-008 v2.2.

Only source sampling is vectorized.  Every target-conditioned reduction receives a
leading candidate dimension of exactly one.  Source batches are streamed once,
shared by explicitly declared target/arm consumers, and discarded immediately.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry

PROTOCOL_SHA256: Final = (
    "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
)
SCHEMA_VERSION: Final = "mm008-v2.2-exact-global-v1"

SOURCE_SCOPE_TAG: Final = b"MM008-v2.2-source-grid-scope\0"
SOURCE_BATCH_TAG: Final = b"MM008-v2.2-source-grid-batch\0"
SOURCE_PARTITION_TAG: Final = b"MM008-v2.2-source-grid-partition\0"
SOURCE_SAMPLES_TAG: Final = b"MM008-v2.2-source-grid-samples\0"
SOURCE_CONTENT_TAG: Final = b"MM008-v2.2-source-grid-content\0"
OBJECTIVE_SCOPE_TAG: Final = b"MM008-v2.2-objective-scope\0"
OBJECTIVE_CONTENT_TAG: Final = b"MM008-v2.2-objective-content\0"
SELECTED_EVALUATION_TAG: Final = b"MM008-v2.2-selected-evaluation\0"
SELECTED_PREDICTION_TAG: Final = b"MM008-v2.2-selected-prediction\0"

Arm = Literal["affine", "combined"]
_ARM_BYTE: Final[dict[Arm, int]] = {"affine": 0, "combined": 1}
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")


class GlobalV22Error(ValueError):
    """Raised when an exact-global input or evidence seam fails closed."""


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise GlobalV22Error(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    if not np.all(np.isfinite(array)):
        raise GlobalV22Error("scientific float64 array contains a nonfinite value")
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _readonly_uint8(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(array.shape)


def _mask_bytes(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8, order="C").tobytes(order="C")


def _validate_context_masks(fit_mask: np.ndarray, output_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fit_mask, np.ndarray) or not isinstance(output_mask, np.ndarray):
        raise GlobalV22Error("fit and output masks must be NumPy arrays")
    if (
        fit_mask.shape != (geometry.SITE_COUNT,)
        or output_mask.shape != (geometry.SITE_COUNT,)
        or fit_mask.dtype != np.dtype(np.bool_)
        or output_mask.dtype != np.dtype(np.bool_)
        or not fit_mask.flags.c_contiguous
        or not output_mask.flags.c_contiguous
    ):
        raise GlobalV22Error("fit/output masks must be C-contiguous bool arrays of shape [2304]")
    full = np.array_equal(fit_mask, geometry.FULL_MASK) and np.array_equal(output_mask, geometry.FULL_MASK)
    xfit = any(
        np.array_equal(fit_mask, geometry.PARITY_MASKS[1 - output_parity])
        and np.array_equal(output_mask, geometry.PARITY_MASKS[output_parity])
        for output_parity in (0, 1)
    )
    if not full and not xfit:
        raise GlobalV22Error("fit/output masks are not a frozen full or checkerboard context")
    fitted_bytes = np.ascontiguousarray(fit_mask, dtype=np.uint8).tobytes(order="C")
    output_bytes = np.ascontiguousarray(output_mask, dtype=np.uint8).tobytes(order="C")
    fitted = np.frombuffer(fitted_bytes, dtype=np.bool_)
    output = np.frombuffer(output_bytes, dtype=np.bool_)
    return fitted, output


def _validate_source(source: np.ndarray) -> np.ndarray:
    if not isinstance(source, np.ndarray):
        raise GlobalV22Error("source must be a NumPy array")
    if source.shape != (geometry.CHANNELS, geometry.NATIVE_SIZE, geometry.NATIVE_SIZE):
        raise GlobalV22Error("source must have shape [3,64,64]")
    if source.dtype != np.dtype(np.float64) or not source.flags.c_contiguous:
        raise GlobalV22Error("source must be C-contiguous float64")
    return _readonly_float64(source)


@dataclass(frozen=True, slots=True)
class FitRequest:
    """One permitted target consumer of a target-free source-grid stream."""

    context_key: str
    arm: Arm
    fit_target: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.context_key, str) or not self.context_key or len(self.context_key) > 512:
            raise GlobalV22Error("context key must be a nonempty string of at most 512 characters")
        try:
            self.context_key.encode("ascii")
        except UnicodeEncodeError as error:
            raise GlobalV22Error("context key must be ASCII") from error
        if self.arm not in _ARM_BYTE:
            raise GlobalV22Error("grid arm must be exactly affine or combined")
        if not isinstance(self.fit_target, np.ndarray) or self.fit_target.ndim != 2:
            raise GlobalV22Error("fit target must be a [3,fit_site] NumPy array")
        if self.fit_target.shape not in (
            (geometry.CHANNELS, geometry.SITE_COUNT),
            (geometry.CHANNELS, geometry.SITE_COUNT // 2),
        ) or self.fit_target.dtype != np.dtype(np.float64):
            raise GlobalV22Error("fit target must have exact float64 dtype and three channels")
        if not self.fit_target.flags.c_contiguous:
            raise GlobalV22Error("fit target must be C-contiguous")
        object.__setattr__(self, "fit_target", _readonly_float64(self.fit_target))

    @classmethod
    def create(cls, context_key: str, arm: Arm, fit_target: np.ndarray) -> FitRequest:
        return cls(context_key, arm, fit_target)


@dataclass(frozen=True, slots=True)
class SourceBatchRecord:
    ordinal: int
    indices: tuple[int, ...]
    shape: tuple[int, int, int]
    dtype: str
    sample_sha256: str
    batch_sha256: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < geometry.BATCH_COUNT:
            raise GlobalV22Error("source batch ordinal is invalid")
        expected_indices = geometry.ADMISSIBLE_BATCHES[self.ordinal]
        if self.indices != expected_indices:
            raise GlobalV22Error("source batch indices differ from the canonical partition")
        expected_shape = (len(self.indices), geometry.CHANNELS, self.shape[2])
        if self.shape != expected_shape or self.shape[2] not in (geometry.SITE_COUNT, geometry.SITE_COUNT // 2):
            raise GlobalV22Error("source batch shape is invalid")
        if self.dtype != "<f8":
            raise GlobalV22Error("source batch dtype must be '<f8'")
        _require_sha256(self.sample_sha256, "sample SHA-256")
        _require_sha256(self.batch_sha256, "batch SHA-256")


@dataclass(frozen=True, slots=True)
class SourceGridRecord:
    scope_sha256: str
    partition_sha256: str
    sample_stream_sha256: str
    content_sha256: str
    batch_records: tuple[SourceBatchRecord, ...]

    def __post_init__(self) -> None:
        for name in ("scope_sha256", "partition_sha256", "sample_stream_sha256", "content_sha256"):
            _require_sha256(cast(str, getattr(self, name)), name)
        if len(self.batch_records) != geometry.BATCH_COUNT:
            raise GlobalV22Error("source-grid record must contain exactly 22 batches")
        if tuple(record.ordinal for record in self.batch_records) != tuple(range(geometry.BATCH_COUNT)):
            raise GlobalV22Error("source-grid batches are not in canonical order")


@dataclass(frozen=True, slots=True)
class CompleteObjectiveCache:
    """Compact, complete representation of all 15,625 canonical entries."""

    arm: Arm
    objectives: np.ndarray
    gains: np.ndarray | None
    biases: np.ndarray | None
    retained_macro_ids: tuple[tuple[int, ...], ...]
    scope_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in _ARM_BYTE:
            raise GlobalV22Error("objective cache arm is invalid")
        objectives = _readonly_float64(self.objectives)
        if objectives.shape != (geometry.ADMISSIBLE_COUNT,) or np.any(objectives < 0.0):
            raise GlobalV22Error("objective cache must have 2,809 finite nonnegative values")
        object.__setattr__(self, "objectives", objectives)
        _require_sha256(self.scope_sha256, "objective scope SHA-256")
        _require_sha256(self.content_sha256, "objective content SHA-256")
        if self.arm == "affine":
            if self.gains is not None or self.biases is not None or self.retained_macro_ids:
                raise GlobalV22Error("affine objective cache carries appearance values")
            return
        if self.gains is None or self.biases is None:
            raise GlobalV22Error("combined objective cache lacks appearance values")
        gains = _readonly_float64(self.gains)
        biases = _readonly_float64(self.biases)
        if gains.shape != (geometry.ADMISSIBLE_COUNT, geometry.CHANNELS) or biases.shape != gains.shape:
            raise GlobalV22Error("combined appearance cache has the wrong shape")
        if len(self.retained_macro_ids) != geometry.ADMISSIBLE_COUNT:
            raise GlobalV22Error("combined retained-ID cache has the wrong length")
        for retained in self.retained_macro_ids:
            if (
                type(retained) is not tuple
                or len(retained) not in (14, 27)
                or tuple(sorted(set(retained))) != retained
                or any(type(value) is not int or not 0 <= value < 36 for value in retained)
            ):
                raise GlobalV22Error("combined retained macro IDs are invalid")
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)


_EntryStatus = Literal["admissible", "inadmissible"]


@dataclass(frozen=True, slots=True)
class _ObjectiveEntry:
    """One immutable total-grid delivery before canonical collection."""

    arm: Arm
    state_index: int
    status: _EntryStatus
    invalid_bitmap_bit: bool
    objective: float | None
    gains: np.ndarray | None
    biases: np.ndarray | None
    retained_macro_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.arm not in _ARM_BYTE:
            raise GlobalV22Error("delivered objective entry arm is invalid")
        if (
            type(self.state_index) is not int
            or not 0 <= self.state_index < geometry.STATE_COUNT
        ):
            raise GlobalV22Error("delivered objective entry index is invalid")
        if self.status not in {"admissible", "inadmissible"}:
            raise GlobalV22Error("delivered objective entry status is invalid")
        if type(self.invalid_bitmap_bit) is not bool:
            raise GlobalV22Error("delivered invalid-bitmap bit must be a built-in boolean")
        if type(self.retained_macro_ids) is not tuple:
            raise GlobalV22Error("delivered retained IDs must be an immutable tuple")
        if self.status == "inadmissible":
            if (
                self.objective is not None
                or self.gains is not None
                or self.biases is not None
                or self.retained_macro_ids
            ):
                raise GlobalV22Error("inadmissible delivery carries a fitted payload")
            return
        if (
            type(self.objective) is not float
            or not math.isfinite(self.objective)
            or self.objective < 0.0
        ):
            raise GlobalV22Error("admissible delivery objective is invalid")
        if self.arm == "affine":
            if self.gains is not None or self.biases is not None or self.retained_macro_ids:
                raise GlobalV22Error("affine delivery carries appearance evidence")
            return
        if self.gains is None or self.biases is None:
            raise GlobalV22Error("combined delivery lacks appearance evidence")
        gains = _readonly_float64(self.gains)
        biases = _readonly_float64(self.biases)
        if gains.shape != (geometry.CHANNELS,) or biases.shape != gains.shape:
            raise GlobalV22Error("combined delivery appearance vectors have the wrong shape")
        if bool(np.any(gains < fitting.GAIN_BOUNDS[0])) or bool(
            np.any(gains > fitting.GAIN_BOUNDS[1])
        ):
            raise GlobalV22Error("combined delivery gains are outside the frozen bounds")
        if bool(np.any(biases < fitting.BIAS_BOUNDS[0])) or bool(
            np.any(biases > fitting.BIAS_BOUNDS[1])
        ):
            raise GlobalV22Error("combined delivery biases are outside the frozen bounds")
        retained = self.retained_macro_ids
        if (
            len(retained) not in (14, 27)
            or tuple(sorted(set(retained))) != retained
            or any(type(value) is not int or not 0 <= value < geometry.MACRO_COUNT for value in retained)
        ):
            raise GlobalV22Error("combined delivery retained macro IDs are invalid")
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "biases", biases)


def _invalid_bitmap_bit(state_index: int) -> bool:
    byte = int(geometry.INVALID_BITMAP[state_index // 8])
    return bool(byte & (1 << (state_index % 8)))


def _validate_delivered_entry(
    entry: _ObjectiveEntry,
    arm: Arm,
    expected_retained_count: int,
) -> None:
    if entry.arm != arm:
        raise GlobalV22Error("objective delivery contains another arm")
    expected_admissible = bool(geometry.ADMISSIBLE_MASK[entry.state_index])
    expected_status: _EntryStatus = (
        "admissible" if expected_admissible else "inadmissible"
    )
    if entry.status != expected_status:
        raise GlobalV22Error("objective delivery index/status differs from admissibility")
    if entry.invalid_bitmap_bit is not _invalid_bitmap_bit(entry.state_index):
        raise GlobalV22Error("objective delivery invalid-bitmap bit is inconsistent")
    if expected_admissible and len(entry.retained_macro_ids) != expected_retained_count:
        raise GlobalV22Error("objective delivery retained-ID count differs from its fit context")


@dataclass(frozen=True, slots=True)
class _CompleteDelivery:
    """Canonical immutable collection of every total-grid status and payload."""

    arm: Arm
    expected_retained_count: int
    entries: tuple[_ObjectiveEntry, ...]

    def __post_init__(self) -> None:
        if self.arm not in _ARM_BYTE:
            raise GlobalV22Error("complete objective delivery arm is invalid")
        valid_retained_counts = {0} if self.arm == "affine" else {14, 27}
        if self.expected_retained_count not in valid_retained_counts:
            raise GlobalV22Error("complete objective delivery retained count is invalid")
        if (
            type(self.entries) is not tuple
            or len(self.entries) != geometry.STATE_COUNT
            or any(not isinstance(entry, _ObjectiveEntry) for entry in self.entries)
        ):
            raise GlobalV22Error("complete objective delivery must contain 15,625 entries")
        if tuple(entry.state_index for entry in self.entries) != tuple(
            range(geometry.STATE_COUNT)
        ):
            raise GlobalV22Error("complete objective delivery is not canonical by index")
        for entry in self.entries:
            _validate_delivered_entry(entry, self.arm, self.expected_retained_count)
        if len(self.admissible_entries) != geometry.ADMISSIBLE_COUNT:
            raise GlobalV22Error("complete objective delivery has the wrong admissible count")

    @property
    def admissible_entries(self) -> tuple[_ObjectiveEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status == "admissible")


def _collect_complete_delivery(
    arm: Arm,
    delivered: tuple[_ObjectiveEntry, ...],
    *,
    expected_retained_count: int,
) -> _CompleteDelivery:
    """Validate exact total-grid membership, then canonicalize delivery order."""

    if arm not in _ARM_BYTE:
        raise GlobalV22Error("objective delivery collector arm is invalid")
    if type(delivered) is not tuple or len(delivered) != geometry.STATE_COUNT:
        raise GlobalV22Error("objective delivery is missing a total-grid entry")
    if any(not isinstance(entry, _ObjectiveEntry) for entry in delivered):
        raise GlobalV22Error("objective delivery contains a malformed entry")
    indices = tuple(entry.state_index for entry in delivered)
    if len(set(indices)) != geometry.STATE_COUNT:
        raise GlobalV22Error("objective delivery contains a duplicate state index")
    if set(indices) != set(range(geometry.STATE_COUNT)):
        raise GlobalV22Error("objective delivery index membership is incomplete")
    for entry in delivered:
        _validate_delivered_entry(entry, arm, expected_retained_count)
    canonical = tuple(sorted(delivered, key=lambda entry: entry.state_index))
    return _CompleteDelivery(arm, expected_retained_count, canonical)


@dataclass(frozen=True, slots=True)
class SelectedEvaluation:
    state_index: int
    admissible_rank: int
    parameters: np.ndarray
    objective: float
    gains: np.ndarray | None
    biases: np.ndarray | None
    retained_macro_ids: tuple[int, ...]
    fit_prediction: np.ndarray
    evaluation_sha256: str

    def __post_init__(self) -> None:
        if type(self.retained_macro_ids) is not tuple:
            raise GlobalV22Error("selected retained IDs must be an immutable tuple")
        if type(self.state_index) is not int or not 0 <= self.state_index < geometry.STATE_COUNT:
            raise GlobalV22Error("selected state index is invalid")
        if type(self.admissible_rank) is not int or not 0 <= self.admissible_rank < geometry.ADMISSIBLE_COUNT:
            raise GlobalV22Error("selected admissible rank is invalid")
        if int(geometry.ADMISSIBLE_INDICES[self.admissible_rank]) != self.state_index:
            raise GlobalV22Error("selected total and admissible ranks disagree")
        parameters = _readonly_float64(self.parameters)
        if parameters.shape != (6,) or not np.array_equal(parameters, geometry.CANONICAL_GRID[self.state_index]):
            raise GlobalV22Error("selected parameters differ from their canonical state")
        if not math.isfinite(self.objective) or self.objective < 0.0:
            raise GlobalV22Error("selected objective must be finite and nonnegative")
        prediction = _readonly_float64(self.fit_prediction)
        if prediction.ndim != 2 or prediction.shape[0] != geometry.CHANNELS:
            raise GlobalV22Error("selected fit prediction has the wrong shape")
        if self.gains is None:
            if self.biases is not None or self.retained_macro_ids:
                raise GlobalV22Error("affine selection carries appearance values")
        else:
            if self.biases is None:
                raise GlobalV22Error("combined selection lacks biases")
            gains = _readonly_float64(self.gains)
            biases = _readonly_float64(self.biases)
            if gains.shape != (geometry.CHANNELS,) or biases.shape != gains.shape:
                raise GlobalV22Error("selected appearance values have the wrong shape")
            if bool(np.any(gains < fitting.GAIN_BOUNDS[0])) or bool(
                np.any(gains > fitting.GAIN_BOUNDS[1])
            ):
                raise GlobalV22Error("selected gains are outside the frozen bounds")
            if bool(np.any(biases < fitting.BIAS_BOUNDS[0])) or bool(
                np.any(biases > fitting.BIAS_BOUNDS[1])
            ):
                raise GlobalV22Error("selected biases are outside the frozen bounds")
            retained = self.retained_macro_ids
            if (
                len(retained) not in (14, 27)
                or tuple(sorted(set(retained))) != retained
                or any(
                    type(value) is not int or not 0 <= value < geometry.MACRO_COUNT
                    for value in retained
                )
            ):
                raise GlobalV22Error("selected retained macro IDs are invalid")
            object.__setattr__(self, "gains", gains)
            object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "fit_prediction", prediction)
        _require_sha256(self.evaluation_sha256, "selected evaluation SHA-256")


@dataclass(frozen=True, slots=True)
class GlobalCertificate:
    protocol_sha256: str
    config_sha256: str
    candidate_order_sha256: str
    admissible_list_sha256: str
    invalid_bitmap_sha256: str
    geometry_sha256: str
    source_scope_sha256: str
    source_content_sha256: str
    objective_scope_sha256: str
    objective_content_sha256: str
    candidate_count: int
    admissible_count: int
    inadmissible_count: int
    selected_total_rank: int
    selected_admissible_rank: int
    exact_tie_multiplicity: int
    second_best_objective_gap: float
    second_best_nonflow_gap: float
    selected_evaluation_sha256: str
    selected_prediction_sha256: str
    scalar_replay_bit_exact: bool

    def __post_init__(self) -> None:
        hashes = (
            "protocol_sha256",
            "config_sha256",
            "candidate_order_sha256",
            "admissible_list_sha256",
            "invalid_bitmap_sha256",
            "geometry_sha256",
            "source_scope_sha256",
            "source_content_sha256",
            "objective_scope_sha256",
            "objective_content_sha256",
            "selected_evaluation_sha256",
            "selected_prediction_sha256",
        )
        for name in hashes:
            _require_sha256(cast(str, getattr(self, name)), name)
        if self.protocol_sha256 != PROTOCOL_SHA256:
            raise GlobalV22Error("certificate protocol hash differs from v2.2")
        if (self.candidate_count, self.admissible_count, self.inadmissible_count) != (
            geometry.STATE_COUNT,
            geometry.ADMISSIBLE_COUNT,
            geometry.STATE_COUNT - geometry.ADMISSIBLE_COUNT,
        ):
            raise GlobalV22Error("certificate candidate counts are invalid")
        if not 0 <= self.selected_total_rank < geometry.STATE_COUNT:
            raise GlobalV22Error("selected total rank is invalid")
        if not 0 <= self.selected_admissible_rank < geometry.ADMISSIBLE_COUNT:
            raise GlobalV22Error("selected admissible rank is invalid")
        if self.exact_tie_multiplicity < 1:
            raise GlobalV22Error("exact tie multiplicity must be positive")
        if (
            not math.isfinite(self.second_best_objective_gap)
            or not math.isfinite(self.second_best_nonflow_gap)
            or self.second_best_objective_gap < 0.0
            or self.second_best_nonflow_gap < 0.0
        ):
            raise GlobalV22Error("certificate gaps must be finite and nonnegative")
        if type(self.scalar_replay_bit_exact) is not bool or not self.scalar_replay_bit_exact:
            raise GlobalV22Error("selected scalar replay did not pass bit-exactly")


@dataclass(frozen=True, slots=True)
class GlobalResult:
    context_key: str
    arm: Arm
    source_grid: SourceGridRecord
    objective_cache: CompleteObjectiveCache
    selected: SelectedEvaluation
    prediction: np.ndarray
    prediction_sha256: str
    certificate: GlobalCertificate

    def __post_init__(self) -> None:
        prediction = _readonly_float64(self.prediction)
        if prediction.ndim != 2 or prediction.shape[0] != geometry.CHANNELS:
            raise GlobalV22Error("selected prediction has the wrong shape")
        if self.arm != self.objective_cache.arm:
            raise GlobalV22Error("result arm differs from objective-cache arm")
        if self.prediction_sha256 != self.certificate.selected_prediction_sha256:
            raise GlobalV22Error("selected prediction hashes disagree")
        if self.selected.evaluation_sha256 != self.certificate.selected_evaluation_sha256:
            raise GlobalV22Error("selected evaluation hashes disagree")
        object.__setattr__(self, "prediction", prediction)
        _require_sha256(self.prediction_sha256, "selected prediction SHA-256")


@dataclass(slots=True)
class _Consumer:
    request: FitRequest
    delivered: list[_ObjectiveEntry]
    best_entry: _ObjectiveEntry | None = None
    best_prediction: np.ndarray | None = None


def _source_scope(
    source: np.ndarray, fit_mask: np.ndarray, output_mask: np.ndarray, config_sha256: str
) -> str:
    digest = hashlib.sha256()
    digest.update(SOURCE_SCOPE_TAG)
    digest.update(np.asarray(source, dtype="<f8", order="C").tobytes(order="C"))
    digest.update(_mask_bytes(fit_mask))
    digest.update(_mask_bytes(output_mask))
    digest.update(bytes.fromhex(geometry.CANDIDATE_ORDER_SHA256))
    digest.update(bytes.fromhex(geometry.ADMISSIBLE_LIST_SHA256))
    digest.update(struct.pack("<H", geometry.BATCH_SIZE))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


def _partition_sha256(source_scope_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(SOURCE_PARTITION_TAG)
    digest.update(bytes.fromhex(source_scope_sha256))
    digest.update(struct.pack("<H", geometry.BATCH_COUNT))
    for ordinal, indices in enumerate(geometry.ADMISSIBLE_BATCHES):
        digest.update(struct.pack("<HH", ordinal, len(indices)))
        digest.update(struct.pack(f"<{len(indices)}H", *indices))
    return digest.hexdigest()


def _batch_record(
    source_scope_sha256: str, ordinal: int, indices: tuple[int, ...], sampled: np.ndarray
) -> SourceBatchRecord:
    sample_bytes = np.asarray(sampled, dtype="<f8", order="C").tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(SOURCE_BATCH_TAG)
    digest.update(bytes.fromhex(source_scope_sha256))
    digest.update(struct.pack("<HH", ordinal, len(indices)))
    digest.update(struct.pack(f"<{len(indices)}H", *indices))
    digest.update(struct.pack("<HI", geometry.CHANNELS, sampled.shape[2]))
    digest.update(sample_bytes)
    return SourceBatchRecord(
        ordinal=ordinal,
        indices=indices,
        shape=cast(tuple[int, int, int], sampled.shape),
        dtype="<f8",
        sample_sha256=hashlib.sha256(sample_bytes).hexdigest(),
        batch_sha256=digest.hexdigest(),
    )


def _evaluate(arm: Arm, sampled_one: np.ndarray, target: np.ndarray, fit_mask: np.ndarray) -> object:
    return fitting.reduce_candidate(arm, sampled_one, target, fit_mask)


def _entry_bytes(
    arm: Arm,
    state_index: int,
    objective: float | None,
    gains: np.ndarray | None = None,
    biases: np.ndarray | None = None,
    retained: tuple[int, ...] = (),
) -> bytes:
    prefix = struct.pack("<HB", state_index, 0 if objective is None else 1)
    if objective is None:
        if gains is not None or biases is not None or retained:
            raise GlobalV22Error("inadmissible entry carries fitted values")
        return prefix
    if not math.isfinite(objective) or objective < 0.0:
        raise GlobalV22Error("valid entry objective is invalid")
    payload = bytearray(prefix)
    payload.extend(struct.pack("<d", objective))
    if arm == "combined":
        if gains is None or biases is None or len(retained) not in (14, 27):
            raise GlobalV22Error("combined entry appearance payload is invalid")
        payload.extend(np.asarray(gains, dtype="<f8", order="C").tobytes(order="C"))
        payload.extend(np.asarray(biases, dtype="<f8", order="C").tobytes(order="C"))
        payload.extend(struct.pack("<B", len(retained)))
        payload.extend(bytes(retained))
    elif gains is not None or biases is not None or retained:
        raise GlobalV22Error("affine entry carries appearance values")
    return bytes(payload)


def _objective_scope(
    source_grid: SourceGridRecord,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: Arm,
    config_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(OBJECTIVE_SCOPE_TAG)
    digest.update(bytes.fromhex(source_grid.scope_sha256))
    digest.update(bytes.fromhex(source_grid.content_sha256))
    digest.update(_mask_bytes(fit_mask))
    digest.update(_mask_bytes(output_mask))
    digest.update(struct.pack("<H", fit_target.shape[1]))
    digest.update(np.asarray(fit_target, dtype="<f8", order="C").tobytes(order="C"))
    digest.update(struct.pack("<B", _ARM_BYTE[arm]))
    digest.update(bytes.fromhex(config_sha256))
    return digest.hexdigest()


def _objective_content(
    scope_sha256: str,
    delivery: _CompleteDelivery,
) -> str:
    digest = hashlib.sha256()
    digest.update(OBJECTIVE_CONTENT_TAG)
    digest.update(bytes.fromhex(scope_sha256))
    digest.update(struct.pack("<H", geometry.STATE_COUNT))
    for entry in delivery.entries:
        digest.update(
            _entry_bytes(
                entry.arm,
                entry.state_index,
                entry.objective,
                entry.gains,
                entry.biases,
                entry.retained_macro_ids,
            )
        )
    return digest.hexdigest()


def _delivery_cache_payload(
    delivery: _CompleteDelivery,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    tuple[tuple[int, ...], ...],
]:
    admissible = delivery.admissible_entries
    objectives = np.asarray(
        [cast(float, entry.objective) for entry in admissible], dtype="<f8"
    )
    if delivery.arm == "affine":
        return objectives, None, None, ()
    gains = np.stack([cast(np.ndarray, entry.gains) for entry in admissible])
    biases = np.stack([cast(np.ndarray, entry.biases) for entry in admissible])
    retained = tuple(entry.retained_macro_ids for entry in admissible)
    return objectives, gains, biases, retained


def _delivery_from_cache(
    cache: CompleteObjectiveCache,
    *,
    expected_retained_count: int,
) -> _CompleteDelivery:
    """Reconstruct total-grid deliveries from an already authenticated cache."""

    if not isinstance(cache, CompleteObjectiveCache):
        raise GlobalV22Error("delivery reconstruction requires a complete objective cache")
    delivered: list[_ObjectiveEntry] = []
    admissible_position = 0
    for state_index in range(geometry.STATE_COUNT):
        if not bool(geometry.ADMISSIBLE_MASK[state_index]):
            delivered.append(
                _ObjectiveEntry(
                    cache.arm,
                    state_index,
                    "inadmissible",
                    _invalid_bitmap_bit(state_index),
                    None,
                    None,
                    None,
                    (),
                )
            )
            continue
        gains = (
            None if cache.gains is None else cache.gains[admissible_position]
        )
        biases = (
            None if cache.biases is None else cache.biases[admissible_position]
        )
        retained = (
            ()
            if cache.arm == "affine"
            else cache.retained_macro_ids[admissible_position]
        )
        delivered.append(
            _ObjectiveEntry(
                cache.arm,
                state_index,
                "admissible",
                _invalid_bitmap_bit(state_index),
                float(cache.objectives[admissible_position]),
                gains,
                biases,
                retained,
            )
        )
        admissible_position += 1
    if admissible_position != geometry.ADMISSIBLE_COUNT:
        raise GlobalV22Error("cache delivery reconstruction was incomplete")
    return _collect_complete_delivery(
        cache.arm,
        tuple(delivered),
        expected_retained_count=expected_retained_count,
    )


def _flow(state_index: int) -> np.ndarray:
    state = geometry.CANONICAL_GRID[state_index]
    u = geometry.GEOMETRY.normalized_coords[:, 0]
    v = geometry.GEOMETRY.normalized_coords[:, 1]
    dy = state[0] + state[2] * u + state[3] * v
    dx = state[1] + state[4] * u + state[5] * v
    return np.stack((dy, dx), axis=1)


def _selection_diagnostics(delivery: _CompleteDelivery) -> tuple[int, int, float, float]:
    admissible = delivery.admissible_entries
    order = sorted(
        range(geometry.ADMISSIBLE_COUNT),
        key=lambda position: (
            cast(float, admissible[position].objective),
            admissible[position].state_index,
        ),
    )
    if len(order) < 2:
        raise GlobalV22Error("complete grid has no second valid entry")
    selected_position = order[0]
    selected_objective = cast(float, admissible[selected_position].objective)
    exact_ties = sum(
        cast(float, entry.objective) == selected_objective for entry in admissible
    )
    second_gap = cast(float, admissible[order[1]].objective) - selected_objective
    selected_index = admissible[selected_position].state_index
    selected_flow = _flow(selected_index)
    nonflow_gap: float | None = None
    for position in order[1:]:
        index = admissible[position].state_index
        if float(np.max(np.abs(_flow(index) - selected_flow))) > 1e-12:
            nonflow_gap = cast(float, admissible[position].objective) - selected_objective
            break
    if nonflow_gap is None or second_gap < 0.0 or nonflow_gap < 0.0:
        raise GlobalV22Error("complete grid has no valid non-flow-equivalent second entry")
    return selected_position, exact_ties, second_gap, nonflow_gap


def _selected_entry_bytes(entry: _ObjectiveEntry) -> bytes:
    return _entry_bytes(
        entry.arm,
        entry.state_index,
        entry.objective,
        entry.gains,
        entry.biases,
        entry.retained_macro_ids,
    )


def _same_float(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def _finish_consumer(
    consumer: _Consumer,
    source: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    source_grid: SourceGridRecord,
    config_sha256: str,
) -> GlobalResult:
    request = consumer.request
    expected_retained_count = (
        0
        if request.arm == "affine"
        else (27 if int(np.count_nonzero(fit_mask)) == geometry.SITE_COUNT else 14)
    )
    delivery = _collect_complete_delivery(
        request.arm,
        tuple(consumer.delivered),
        expected_retained_count=expected_retained_count,
    )
    objectives, cached_gains, cached_biases, retained_tuple = _delivery_cache_payload(
        delivery
    )
    objective_scope = _objective_scope(
        source_grid, request.fit_target, fit_mask, output_mask, request.arm, config_sha256
    )
    objective_content = _objective_content(objective_scope, delivery)
    cache = CompleteObjectiveCache(
        arm=request.arm,
        objectives=objectives,
        gains=cached_gains,
        biases=cached_biases,
        retained_macro_ids=retained_tuple,
        scope_sha256=objective_scope,
        content_sha256=objective_content,
    )
    selected_position, exact_ties, second_gap, nonflow_gap = _selection_diagnostics(
        delivery
    )
    selected_delivery = delivery.admissible_entries[selected_position]
    selected_index = selected_delivery.state_index
    if (
        consumer.best_entry is None
        or consumer.best_entry.state_index != selected_index
        or consumer.best_prediction is None
    ):
        raise GlobalV22Error("streaming argmin differs from complete global selection")

    scalar_sample = geometry.sample_scalar(source, selected_index, fit_mask)[None, :, :]
    scalar_fit = _evaluate(request.arm, scalar_sample, request.fit_target, fit_mask)
    scalar_objective = float(cast(object, scalar_fit).objective)  # type: ignore[attr-defined]
    if not _same_float(scalar_objective, float(cache.objectives[selected_position])):
        raise GlobalV22Error("selected scalar objective differs bitwise from the batched reduction")
    scalar_prediction = np.asarray(cast(object, scalar_fit).prediction, dtype=np.float64)  # type: ignore[attr-defined]
    if scalar_prediction.tobytes(order="C") != consumer.best_prediction.tobytes(order="C"):
        raise GlobalV22Error("selected scalar fit prediction differs bitwise from the batched reduction")

    gains: np.ndarray | None = None
    biases: np.ndarray | None = None
    retained: tuple[int, ...] = ()
    if request.arm == "combined":
        gains = np.asarray(cast(object, scalar_fit).gains, dtype=np.float64)  # type: ignore[attr-defined]
        biases = np.asarray(cast(object, scalar_fit).biases, dtype=np.float64)  # type: ignore[attr-defined]
        retained = cast(tuple[int, ...], cast(object, scalar_fit).retained_macro_ids)  # type: ignore[attr-defined]
        assert cache.gains is not None and cache.biases is not None
        if (
            gains.tobytes(order="C") != cache.gains[selected_position].tobytes(order="C")
            or biases.tobytes(order="C") != cache.biases[selected_position].tobytes(order="C")
            or retained != cache.retained_macro_ids[selected_position]
        ):
            raise GlobalV22Error("selected appearance replay differs from the complete cache")

    selected_entry = _selected_entry_bytes(selected_delivery)
    evaluation_digest = hashlib.sha256(
        SELECTED_EVALUATION_TAG + bytes.fromhex(objective_scope) + selected_entry
    ).hexdigest()
    selected = SelectedEvaluation(
        state_index=selected_index,
        admissible_rank=selected_position,
        parameters=geometry.CANONICAL_GRID[selected_index],
        objective=scalar_objective,
        gains=gains,
        biases=biases,
        retained_macro_ids=retained,
        fit_prediction=scalar_prediction,
        evaluation_sha256=evaluation_digest,
    )

    output_sample = geometry.sample_scalar(source, selected_index, output_mask)
    if request.arm == "combined":
        assert gains is not None and biases is not None
        prediction = gains[:, None] * output_sample + biases[:, None]
    else:
        prediction = output_sample
    prediction = _readonly_float64(prediction)
    prediction_digest = hashlib.sha256()
    prediction_digest.update(SELECTED_PREDICTION_TAG)
    prediction_digest.update(bytes.fromhex(objective_scope))
    prediction_digest.update(struct.pack("<H", prediction.shape[1]))
    prediction_digest.update(prediction.tobytes(order="C"))
    prediction_sha256 = prediction_digest.hexdigest()

    certificate = GlobalCertificate(
        protocol_sha256=PROTOCOL_SHA256,
        config_sha256=config_sha256,
        candidate_order_sha256=geometry.CANDIDATE_ORDER_SHA256,
        admissible_list_sha256=geometry.ADMISSIBLE_LIST_SHA256,
        invalid_bitmap_sha256=geometry.INVALID_BITMAP_SHA256,
        geometry_sha256=geometry.GEOMETRY_SHA256,
        source_scope_sha256=source_grid.scope_sha256,
        source_content_sha256=source_grid.content_sha256,
        objective_scope_sha256=objective_scope,
        objective_content_sha256=objective_content,
        candidate_count=geometry.STATE_COUNT,
        admissible_count=geometry.ADMISSIBLE_COUNT,
        inadmissible_count=geometry.STATE_COUNT - geometry.ADMISSIBLE_COUNT,
        selected_total_rank=selected_index,
        selected_admissible_rank=selected_position,
        exact_tie_multiplicity=exact_ties,
        second_best_objective_gap=second_gap,
        second_best_nonflow_gap=nonflow_gap,
        selected_evaluation_sha256=evaluation_digest,
        selected_prediction_sha256=prediction_sha256,
        scalar_replay_bit_exact=True,
    )
    return GlobalResult(
        context_key=request.context_key,
        arm=request.arm,
        source_grid=source_grid,
        objective_cache=cache,
        selected=selected,
        prediction=prediction,
        prediction_sha256=prediction_sha256,
        certificate=certificate,
    )


def fit_global_contexts(
    source: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    requests: tuple[FitRequest, ...],
    *,
    config_sha256: str,
) -> tuple[GlobalResult, ...]:
    """Fit one or more consumers through one target-free 22-batch source stream."""

    geometry.validate_frozen_constants()
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    current = _validate_source(source)
    fitted, output = _validate_context_masks(fit_mask, output_mask)
    if type(requests) is not tuple or not requests:
        raise GlobalV22Error("requests must be a nonempty tuple")
    if any(not isinstance(request, FitRequest) for request in requests):
        raise GlobalV22Error("every request must be a FitRequest")
    keys = tuple(request.context_key for request in requests)
    if len(set(keys)) != len(keys):
        raise GlobalV22Error("fit request context keys must be unique")
    fit_count = int(np.count_nonzero(fitted))
    if any(request.fit_target.shape != (geometry.CHANNELS, fit_count) for request in requests):
        raise GlobalV22Error("fit request target shape differs from the selected fit mask")

    invalid_deliveries: dict[Arm, tuple[_ObjectiveEntry, ...]] = {}
    for arm in ("affine", "combined"):
        if any(request.arm == arm for request in requests):
            invalid_deliveries[arm] = tuple(
                _ObjectiveEntry(
                    arm=arm,
                    state_index=state_index,
                    status="inadmissible",
                    invalid_bitmap_bit=_invalid_bitmap_bit(state_index),
                    objective=None,
                    gains=None,
                    biases=None,
                    retained_macro_ids=(),
                )
                for state_index in range(geometry.STATE_COUNT)
                if not bool(geometry.ADMISSIBLE_MASK[state_index])
            )

    consumers: list[_Consumer] = []
    for request in requests:
        consumers.append(
            _Consumer(
                request=request,
                delivered=list(invalid_deliveries[request.arm]),
            )
        )

    source_scope = _source_scope(current, fitted, output, checked_config)
    partition_sha256 = _partition_sha256(source_scope)
    samples_digest = hashlib.sha256()
    samples_digest.update(SOURCE_SAMPLES_TAG)
    samples_digest.update(bytes.fromhex(source_scope))
    batch_records: list[SourceBatchRecord] = []
    admissible_position = 0
    for ordinal, indices in enumerate(geometry.ADMISSIBLE_BATCHES):
        sampled = geometry.sample_batch(current, indices, fitted)
        record = _batch_record(source_scope, ordinal, indices, sampled)
        batch_records.append(record)
        samples_digest.update(sampled.tobytes(order="C"))
        for member, state_index in enumerate(indices):
            sampled_one = sampled[member : member + 1]
            for consumer in consumers:
                fitted_candidate = _evaluate(
                    consumer.request.arm,
                    sampled_one,
                    consumer.request.fit_target,
                    fitted,
                )
                objective = float(cast(object, fitted_candidate).objective)  # type: ignore[attr-defined]
                if not math.isfinite(objective) or objective < 0.0:
                    raise GlobalV22Error("candidate reduction produced an invalid objective")
                gains: np.ndarray | None = None
                biases: np.ndarray | None = None
                retained: tuple[int, ...] = ()
                if consumer.request.arm == "combined":
                    gains = np.asarray(cast(object, fitted_candidate).gains, dtype=np.float64)  # type: ignore[attr-defined]
                    biases = np.asarray(cast(object, fitted_candidate).biases, dtype=np.float64)  # type: ignore[attr-defined]
                    retained = cast(
                        tuple[int, ...], cast(object, fitted_candidate).retained_macro_ids  # type: ignore[attr-defined]
                    )
                entry = _ObjectiveEntry(
                    arm=consumer.request.arm,
                    state_index=state_index,
                    status="admissible",
                    invalid_bitmap_bit=_invalid_bitmap_bit(state_index),
                    objective=objective,
                    gains=gains,
                    biases=biases,
                    retained_macro_ids=retained,
                )
                consumer.delivered.append(entry)
                # Canonical indices are the zero-based order of the frozen complete
                # state key, so ``(objective,index)`` is exactly the normative
                # ``(objective,canonical_state_key)`` ordering.
                selection_key = (objective, state_index)
                if consumer.best_entry is None:
                    is_better = True
                else:
                    best_key = (
                        cast(float, consumer.best_entry.objective),
                        consumer.best_entry.state_index,
                    )
                    is_better = selection_key < best_key
                if is_better:
                    consumer.best_entry = entry
                    consumer.best_prediction = _readonly_float64(
                        np.asarray(cast(object, fitted_candidate).prediction, dtype=np.float64)  # type: ignore[attr-defined]
                    )
            admissible_position += 1
        del sampled
    if admissible_position != geometry.ADMISSIBLE_COUNT:
        raise GlobalV22Error("source stream did not enumerate exactly 2,809 admissible states")

    sample_stream_sha256 = samples_digest.hexdigest()
    content_digest = hashlib.sha256()
    content_digest.update(SOURCE_CONTENT_TAG)
    content_digest.update(bytes.fromhex(source_scope))
    content_digest.update(struct.pack("<H", geometry.BATCH_COUNT))
    for record in batch_records:
        content_digest.update(bytes.fromhex(record.batch_sha256))
    content_digest.update(bytes.fromhex(sample_stream_sha256))
    source_grid = SourceGridRecord(
        scope_sha256=source_scope,
        partition_sha256=partition_sha256,
        sample_stream_sha256=sample_stream_sha256,
        content_sha256=content_digest.hexdigest(),
        batch_records=tuple(batch_records),
    )
    return tuple(
        _finish_consumer(consumer, current, fitted, output, source_grid, checked_config)
        for consumer in consumers
    )


def fit_global(
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    arm: Arm,
    *,
    context_key: str,
    config_sha256: str,
) -> GlobalResult:
    """Fit one mandatory exact-global ``G`` result with no uncertified seam."""

    request = FitRequest.create(context_key, arm, fit_target)
    return fit_global_contexts(
        source,
        fit_mask,
        output_mask,
        (request,),
        config_sha256=config_sha256,
    )[0]


def scientific_fingerprint(result: GlobalResult) -> str:
    """Return a compact diagnostic identity over every normative result hash."""

    if not isinstance(result, GlobalResult):
        raise GlobalV22Error("scientific fingerprint requires a GlobalResult")
    digest = hashlib.sha256()
    digest.update(b"MM008-v2.2-global-result\0")
    digest.update(result.context_key.encode("ascii"))
    digest.update(result.arm.encode("ascii"))
    for value in (
        result.source_grid.scope_sha256,
        result.source_grid.partition_sha256,
        result.source_grid.sample_stream_sha256,
        result.source_grid.content_sha256,
        result.objective_cache.scope_sha256,
        result.objective_cache.content_sha256,
        result.selected.evaluation_sha256,
        result.prediction_sha256,
    ):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _array_bytes_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _results_bit_exact(left: GlobalResult, right: GlobalResult) -> bool:
    return (
        left.context_key == right.context_key
        and left.arm == right.arm
        and left.source_grid == right.source_grid
        and left.objective_cache.arm == right.objective_cache.arm
        and left.objective_cache.scope_sha256 == right.objective_cache.scope_sha256
        and left.objective_cache.content_sha256 == right.objective_cache.content_sha256
        and _array_bytes_equal(left.objective_cache.objectives, right.objective_cache.objectives)
        and _array_bytes_equal(left.objective_cache.gains, right.objective_cache.gains)
        and _array_bytes_equal(left.objective_cache.biases, right.objective_cache.biases)
        and left.objective_cache.retained_macro_ids == right.objective_cache.retained_macro_ids
        and left.selected.state_index == right.selected.state_index
        and left.selected.admissible_rank == right.selected.admissible_rank
        and _array_bytes_equal(left.selected.parameters, right.selected.parameters)
        and _same_float(left.selected.objective, right.selected.objective)
        and _array_bytes_equal(left.selected.gains, right.selected.gains)
        and _array_bytes_equal(left.selected.biases, right.selected.biases)
        and left.selected.retained_macro_ids == right.selected.retained_macro_ids
        and _array_bytes_equal(left.selected.fit_prediction, right.selected.fit_prediction)
        and left.selected.evaluation_sha256 == right.selected.evaluation_sha256
        and _array_bytes_equal(left.prediction, right.prediction)
        and left.prediction_sha256 == right.prediction_sha256
        and left.certificate == right.certificate
    )


def validate_global_result(
    result: GlobalResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Deeply rebuild and compare every scientific field of one persisted result.

    The validator accepts no cached source/objective seam and does not trust a PASS
    label or selected-only replay.  It repeats all 22 source batches and all 15,625
    canonical statuses from the supplied immutable scientific inputs.
    """

    if not isinstance(result, GlobalResult):
        raise GlobalV22Error("deep validation requires a GlobalResult")
    rebuilt = fit_global(
        source,
        fit_target,
        fit_mask,
        output_mask,
        result.arm,
        context_key=result.context_key,
        config_sha256=config_sha256,
    )
    if not _results_bit_exact(result, rebuilt):
        raise GlobalV22Error("persisted global result differs from a complete deep rebuild")


__all__ = [
    "CompleteObjectiveCache",
    "FitRequest",
    "GlobalCertificate",
    "GlobalResult",
    "GlobalV22Error",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "SelectedEvaluation",
    "SourceBatchRecord",
    "SourceGridRecord",
    "fit_global",
    "fit_global_contexts",
    "scientific_fingerprint",
    "validate_global_result",
]
