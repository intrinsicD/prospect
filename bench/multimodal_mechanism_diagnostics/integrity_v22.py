"""Production-backed scientific-integrity controls for MM-008 v2.2 exact grids.

Every control consumes explicit scientific arrays or a completed production result.
It owns no alternate fitter, cache, result constructor, filesystem, lifecycle, or
randomness seam.  Delivery-order controls authenticate a production result by deep
replay before exercising the package-private collector used by production itself.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Final, Literal

import numpy as np

from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-exact-integrity-v1"

if any(
    dependency.PROTOCOL_SHA256 != PROTOCOL_SHA256
    for dependency in (geometry, exact)
):
    raise RuntimeError("MM-008 v2.2 integrity dependency binds a different protocol")

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_DeliveryOrder = Literal["canonical", "reversed", "even_then_odd"]
_RejectionName = Literal[
    "missing",
    "duplicate",
    "index_status",
    "admissibility_bit",
]
_ForgeryStage = Literal["construction", "deep_compare"]

DELIVERY_REJECTION_NAMES: Final[tuple[_RejectionName, ...]] = (
    "missing",
    "duplicate",
    "index_status",
    "admissibility_bit",
)

NORMATIVE_HASH_FORGERY_KEYS: Final[tuple[str, ...]] = (
    "hash/source_grid/scope_sha256",
    "hash/source_grid/partition_sha256",
    "hash/source_grid/sample_stream_sha256",
    "hash/source_grid/content_sha256",
    "hash/objective_cache/scope_sha256",
    "hash/objective_cache/content_sha256",
    "hash/selected/evaluation_sha256",
    "hash/result/prediction_sha256",
    "hash/certificate/protocol_sha256",
    "hash/certificate/config_sha256",
    "hash/certificate/candidate_order_sha256",
    "hash/certificate/admissible_list_sha256",
    "hash/certificate/invalid_bitmap_sha256",
    "hash/certificate/geometry_sha256",
    "hash/certificate/source_scope_sha256",
    "hash/certificate/source_content_sha256",
    "hash/certificate/objective_scope_sha256",
    "hash/certificate/objective_content_sha256",
    "hash/certificate/selected_evaluation_sha256",
    "hash/certificate/selected_prediction_sha256",
)

FORGERY_KEYS: Final[tuple[str, ...]] = (
    "payload/objective",
    "payload/gain",
    "payload/bias",
    "payload/retained_id",
    "payload/selected_fit_prediction",
    "payload/selected_output_prediction",
    "certificate/count",
    "certificate/rank",
    *NORMATIVE_HASH_FORGERY_KEYS,
)

CACHE_CHECK_NAMES: Final[tuple[str, ...]] = (
    "shared_source_object_reused",
    "independent_source_object_not_reused",
    "shared_vs_independent_affine_bits",
    "shared_vs_independent_combined_bits",
    "changed_source_cache_miss",
    "changed_fit_mask_cache_miss",
    "changed_output_mask_cache_miss",
    "target_specific_objective_scope",
    "arm_specific_objective_scope",
    "context_key_excluded_from_objective_scope",
    "context_key_excluded_from_objective_content",
)


class IntegrityV22Error(ValueError):
    """Raised when an exact-grid integrity control fails closed."""


def _require_sha256(value: str, label: str) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise IntegrityV22Error(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _float_bits_equal(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def _array_bits_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _scientific_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return _array_bits_equal(left, right)
    if isinstance(left, np.generic) and isinstance(right, np.generic):
        return left.dtype == right.dtype and left.tobytes() == right.tobytes()
    if isinstance(left, float) and isinstance(right, float):
        return _float_bits_equal(left, right)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _scientific_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if is_dataclass(left) and is_dataclass(right) and not isinstance(left, type):
        return all(
            _scientific_equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    equality = left == right
    return type(equality) is bool and equality


def _expected_retained_count(arm: exact.Arm, fit_mask: np.ndarray) -> int:
    if arm == "affine":
        return 0
    return 27 if int(np.count_nonzero(fit_mask)) == geometry.SITE_COUNT else 14


def _authenticate(
    result: exact.GlobalResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    config_sha256: str,
) -> exact.GlobalResult:
    try:
        rebuilt = exact.fit_global(
            source,
            fit_target,
            fit_mask,
            output_mask,
            result.arm,
            context_key=result.context_key,
            config_sha256=config_sha256,
        )
    except exact.GlobalV22Error as error:
        raise IntegrityV22Error("production result failed complete deep authentication") from error
    if not exact._results_bit_exact(result, rebuilt):
        raise IntegrityV22Error("production result differs from complete deep rebuild")
    return rebuilt


@dataclass(frozen=True, slots=True)
class DeliveryOrderWitness:
    order: _DeliveryOrder
    total_count: int
    admissible_count: int
    selected_state_index: int
    selected_admissible_rank: int
    exact_tie_multiplicity: int
    second_best_objective_gap: float
    second_best_nonflow_gap: float
    objective_content_sha256: str

    def __post_init__(self) -> None:
        if self.order not in {"canonical", "reversed", "even_then_odd"}:
            raise IntegrityV22Error("delivery order witness has an invalid order")
        if (self.total_count, self.admissible_count) != (
            geometry.STATE_COUNT,
            geometry.ADMISSIBLE_COUNT,
        ):
            raise IntegrityV22Error("delivery order witness has invalid complete counts")
        if (
            type(self.selected_state_index) is not int
            or type(self.selected_admissible_rank) is not int
            or not 0 <= self.selected_state_index < geometry.STATE_COUNT
            or not 0 <= self.selected_admissible_rank < geometry.ADMISSIBLE_COUNT
            or int(geometry.ADMISSIBLE_INDICES[self.selected_admissible_rank])
            != self.selected_state_index
        ):
            raise IntegrityV22Error("delivery order witness has inconsistent selected ranks")
        if (
            type(self.exact_tie_multiplicity) is not int
            or not 1
            <= self.exact_tie_multiplicity
            <= geometry.ADMISSIBLE_COUNT
        ):
            raise IntegrityV22Error("delivery order witness has an invalid tie count")
        if (
            type(self.second_best_objective_gap) is not float
            or type(self.second_best_nonflow_gap) is not float
            or not np.isfinite(self.second_best_objective_gap)
            or not np.isfinite(self.second_best_nonflow_gap)
            or self.second_best_objective_gap < 0.0
            or self.second_best_nonflow_gap < 0.0
        ):
            raise IntegrityV22Error("delivery order witness has invalid selection gaps")
        _require_sha256(self.objective_content_sha256, "delivery objective content SHA-256")


@dataclass(frozen=True, slots=True)
class RejectionWitness:
    name: _RejectionName
    rejected: bool

    def __post_init__(self) -> None:
        if self.name not in DELIVERY_REJECTION_NAMES or type(self.rejected) is not bool:
            raise IntegrityV22Error("delivery rejection witness is invalid")


@dataclass(frozen=True, slots=True)
class DeliveryIntegrityEvidence:
    orders: tuple[DeliveryOrderWitness, ...]
    rejections: tuple[RejectionWitness, ...]
    batch_sizes: tuple[int, ...]
    scalar_replay_bit_exact: bool

    def __post_init__(self) -> None:
        if tuple(item.order for item in self.orders) != (
            "canonical",
            "reversed",
            "even_then_odd",
        ):
            raise IntegrityV22Error("delivery evidence lacks the three frozen orders")
        canonical = self.orders[0]
        for witness in self.orders[1:]:
            if (
                witness.total_count != canonical.total_count
                or witness.admissible_count != canonical.admissible_count
                or witness.selected_state_index != canonical.selected_state_index
                or witness.selected_admissible_rank
                != canonical.selected_admissible_rank
                or witness.exact_tie_multiplicity
                != canonical.exact_tie_multiplicity
                or not _float_bits_equal(
                    witness.second_best_objective_gap,
                    canonical.second_best_objective_gap,
                )
                or not _float_bits_equal(
                    witness.second_best_nonflow_gap,
                    canonical.second_best_nonflow_gap,
                )
                or witness.objective_content_sha256
                != canonical.objective_content_sha256
            ):
                raise IntegrityV22Error("canonical argmin depends on delivery order")
        if tuple(item.name for item in self.rejections) != DELIVERY_REJECTION_NAMES or not all(
            item.rejected for item in self.rejections
        ):
            raise IntegrityV22Error("malformed full-entry delivery did not fail closed")
        if self.batch_sizes != (*((128,) * 21), 121):
            raise IntegrityV22Error("source stream is not partitioned into 128/121 batches")
        if self.scalar_replay_bit_exact is not True:
            raise IntegrityV22Error("selected scalar replay is not bit-exact")

    @property
    def passed(self) -> bool:
        return True


def _delivery_witness(
    order: _DeliveryOrder,
    delivery: exact._CompleteDelivery,
    objective_scope_sha256: str,
) -> DeliveryOrderWitness:
    position, ties, second_gap, nonflow_gap = exact._selection_diagnostics(delivery)
    selected = delivery.admissible_entries[position]
    return DeliveryOrderWitness(
        order=order,
        total_count=len(delivery.entries),
        admissible_count=len(delivery.admissible_entries),
        selected_state_index=selected.state_index,
        selected_admissible_rank=position,
        exact_tie_multiplicity=ties,
        second_best_objective_gap=second_gap,
        second_best_nonflow_gap=nonflow_gap,
        objective_content_sha256=exact._objective_content(
            objective_scope_sha256, delivery
        ),
    )


def _collector_rejects(
    arm: exact.Arm,
    delivered: tuple[exact._ObjectiveEntry, ...],
    expected_retained_count: int,
) -> bool:
    try:
        exact._collect_complete_delivery(
            arm,
            delivered,
            expected_retained_count=expected_retained_count,
        )
    except exact.GlobalV22Error:
        return True
    return False


def audit_delivery_integrity(
    result: exact.GlobalResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> DeliveryIntegrityEvidence:
    """Authenticate one result, then audit production collection under three orders."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    _authenticate(
        result,
        source,
        fit_target,
        fit_mask,
        output_mask,
        checked_config,
    )
    expected_retained_count = _expected_retained_count(result.arm, fit_mask)
    canonical = exact._delivery_from_cache(
        result.objective_cache,
        expected_retained_count=expected_retained_count,
    )
    reversed_delivery = exact._collect_complete_delivery(
        result.arm,
        tuple(reversed(canonical.entries)),
        expected_retained_count=expected_retained_count,
    )
    fixed_permutation = canonical.entries[::2] + canonical.entries[1::2]
    permuted_delivery = exact._collect_complete_delivery(
        result.arm,
        fixed_permutation,
        expected_retained_count=expected_retained_count,
    )
    orders = (
        _delivery_witness(
            "canonical", canonical, result.objective_cache.scope_sha256
        ),
        _delivery_witness(
            "reversed", reversed_delivery, result.objective_cache.scope_sha256
        ),
        _delivery_witness(
            "even_then_odd", permuted_delivery, result.objective_cache.scope_sha256
        ),
    )
    if (
        orders[0].objective_content_sha256
        != result.objective_cache.content_sha256
        or orders[0].selected_state_index != result.selected.state_index
        or orders[0].selected_admissible_rank != result.selected.admissible_rank
        or orders[0].exact_tie_multiplicity
        != result.certificate.exact_tie_multiplicity
        or not _float_bits_equal(
            orders[0].second_best_objective_gap,
            result.certificate.second_best_objective_gap,
        )
        or not _float_bits_equal(
            orders[0].second_best_nonflow_gap,
            result.certificate.second_best_nonflow_gap,
        )
    ):
        raise IntegrityV22Error("authenticated result differs from its complete delivery")

    entries = canonical.entries
    first_admissible_position = next(
        position for position, entry in enumerate(entries) if entry.status == "admissible"
    )
    admissible_entry = entries[first_admissible_position]
    status_forgery = replace(
        admissible_entry,
        status="inadmissible",
        objective=None,
        gains=None,
        biases=None,
        retained_macro_ids=(),
    )
    status_entries = (
        entries[:first_admissible_position]
        + (status_forgery,)
        + entries[first_admissible_position + 1 :]
    )
    bit_forgery = replace(
        entries[0], invalid_bitmap_bit=not entries[0].invalid_bitmap_bit
    )
    bit_entries = (bit_forgery, *entries[1:])
    rejections = (
        RejectionWitness(
            "missing",
            _collector_rejects(
                result.arm, entries[:-1], expected_retained_count
            ),
        ),
        RejectionWitness(
            "duplicate",
            _collector_rejects(
                result.arm,
                (*entries[:-1], entries[0]),
                expected_retained_count,
            ),
        ),
        RejectionWitness(
            "index_status",
            _collector_rejects(
                result.arm, status_entries, expected_retained_count
            ),
        ),
        RejectionWitness(
            "admissibility_bit",
            _collector_rejects(
                result.arm, bit_entries, expected_retained_count
            ),
        ),
    )
    return DeliveryIntegrityEvidence(
        orders=orders,
        rejections=rejections,
        batch_sizes=tuple(
            len(record.indices) for record in result.source_grid.batch_records
        ),
        scalar_replay_bit_exact=result.certificate.scalar_replay_bit_exact,
    )


@dataclass(frozen=True, slots=True)
class ForgeryWitness:
    key: str
    rejected_at: _ForgeryStage

    def __post_init__(self) -> None:
        if self.key not in FORGERY_KEYS or self.rejected_at not in {
            "construction",
            "deep_compare",
        }:
            raise IntegrityV22Error("forgery witness is invalid")


@dataclass(frozen=True, slots=True)
class ForgeryMatrixEvidence:
    witnesses: tuple[ForgeryWitness, ...]

    def __post_init__(self) -> None:
        if tuple(item.key for item in self.witnesses) != FORGERY_KEYS:
            raise IntegrityV22Error("deep forgery matrix coverage is incomplete")

    @property
    def passed(self) -> bool:
        return True


def _changed_hash(value: str) -> str:
    _require_sha256(value, "forged source hash")
    return ("1" if value[0] == "0" else "0") + value[1:]


def _changed_array(value: np.ndarray) -> np.ndarray:
    changed = value.copy()
    flat = changed.reshape(-1)
    flat[0] = np.nextafter(flat[0], np.inf)
    return changed


def _changed_bounded_array(
    value: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    changed = value.copy()
    flat = changed.reshape(-1)
    midpoint = (lower + upper) / 2.0
    replacement = midpoint if flat[0] != midpoint else midpoint + 0.25
    flat[0] = replacement
    return changed


def _changed_retained_ids(values: tuple[int, ...]) -> tuple[int, ...]:
    replacement = next(value for value in range(geometry.MACRO_COUNT) if value not in values)
    changed = (replacement, *values[1:])
    return tuple(sorted(changed))


def _assess_forgery(
    key: str,
    result: exact.GlobalResult,
    rebuilt: exact.GlobalResult,
    mutation: Callable[[exact.GlobalResult], exact.GlobalResult],
) -> ForgeryWitness:
    try:
        forged = mutation(result)
    except exact.GlobalV22Error:
        return ForgeryWitness(key, "construction")
    if exact._results_bit_exact(result, forged):
        raise IntegrityV22Error(f"forgery escaped complete comparison: {key}")
    if exact._results_bit_exact(forged, rebuilt):
        raise IntegrityV22Error(f"forgery escaped shared complete deep replay: {key}")
    return ForgeryWitness(key, "deep_compare")


def _mutate_cache_objective(result: exact.GlobalResult) -> exact.GlobalResult:
    cache = replace(
        result.objective_cache,
        objectives=_changed_array(result.objective_cache.objectives),
    )
    return replace(result, objective_cache=cache)


def _mutate_cache_gain(result: exact.GlobalResult) -> exact.GlobalResult:
    assert result.objective_cache.gains is not None
    cache = replace(
        result.objective_cache,
        gains=_changed_bounded_array(result.objective_cache.gains, -2.0, 4.0),
    )
    return replace(result, objective_cache=cache)


def _mutate_cache_bias(result: exact.GlobalResult) -> exact.GlobalResult:
    assert result.objective_cache.biases is not None
    cache = replace(
        result.objective_cache,
        biases=_changed_bounded_array(result.objective_cache.biases, -4.0, 4.0),
    )
    return replace(result, objective_cache=cache)


def _mutate_cache_retained(result: exact.GlobalResult) -> exact.GlobalResult:
    retained = result.objective_cache.retained_macro_ids
    changed = (
        _changed_retained_ids(retained[0]),
        *retained[1:],
    )
    cache = replace(result.objective_cache, retained_macro_ids=changed)
    return replace(result, objective_cache=cache)


def _mutate_selected_fit_prediction(result: exact.GlobalResult) -> exact.GlobalResult:
    selected = replace(
        result.selected,
        fit_prediction=_changed_array(result.selected.fit_prediction),
    )
    return replace(result, selected=selected)


def _mutate_selected_output_prediction(result: exact.GlobalResult) -> exact.GlobalResult:
    return replace(result, prediction=_changed_array(result.prediction))


def _mutate_count(result: exact.GlobalResult) -> exact.GlobalResult:
    certificate = replace(
        result.certificate,
        candidate_count=result.certificate.candidate_count - 1,
    )
    return replace(result, certificate=certificate)


def _mutate_rank(result: exact.GlobalResult) -> exact.GlobalResult:
    rank = (result.certificate.selected_admissible_rank + 1) % geometry.ADMISSIBLE_COUNT
    certificate = replace(
        result.certificate,
        selected_admissible_rank=rank,
        selected_total_rank=int(geometry.ADMISSIBLE_INDICES[rank]),
    )
    return replace(result, certificate=certificate)


def _mutate_hash(result: exact.GlobalResult, key: str) -> exact.GlobalResult:
    if key == "hash/source_grid/scope_sha256":
        source_grid = replace(
            result.source_grid,
            scope_sha256=_changed_hash(result.source_grid.scope_sha256),
        )
        return replace(result, source_grid=source_grid)
    if key == "hash/source_grid/partition_sha256":
        source_grid = replace(
            result.source_grid,
            partition_sha256=_changed_hash(result.source_grid.partition_sha256),
        )
        return replace(result, source_grid=source_grid)
    if key == "hash/source_grid/sample_stream_sha256":
        source_grid = replace(
            result.source_grid,
            sample_stream_sha256=_changed_hash(
                result.source_grid.sample_stream_sha256
            ),
        )
        return replace(result, source_grid=source_grid)
    if key == "hash/source_grid/content_sha256":
        source_grid = replace(
            result.source_grid,
            content_sha256=_changed_hash(result.source_grid.content_sha256),
        )
        return replace(result, source_grid=source_grid)
    if key == "hash/objective_cache/scope_sha256":
        cache = replace(
            result.objective_cache,
            scope_sha256=_changed_hash(result.objective_cache.scope_sha256),
        )
        return replace(result, objective_cache=cache)
    if key == "hash/objective_cache/content_sha256":
        cache = replace(
            result.objective_cache,
            content_sha256=_changed_hash(result.objective_cache.content_sha256),
        )
        return replace(result, objective_cache=cache)
    if key == "hash/selected/evaluation_sha256":
        digest = _changed_hash(result.selected.evaluation_sha256)
        selected = replace(result.selected, evaluation_sha256=digest)
        certificate = replace(
            result.certificate, selected_evaluation_sha256=digest
        )
        return replace(result, selected=selected, certificate=certificate)
    if key == "hash/result/prediction_sha256":
        digest = _changed_hash(result.prediction_sha256)
        certificate = replace(
            result.certificate, selected_prediction_sha256=digest
        )
        return replace(
            result,
            prediction_sha256=digest,
            certificate=certificate,
        )
    certificate = result.certificate
    if key == "hash/certificate/protocol_sha256":
        changed = replace(
            certificate,
            protocol_sha256=_changed_hash(certificate.protocol_sha256),
        )
    elif key == "hash/certificate/config_sha256":
        changed = replace(
            certificate,
            config_sha256=_changed_hash(certificate.config_sha256),
        )
    elif key == "hash/certificate/candidate_order_sha256":
        changed = replace(
            certificate,
            candidate_order_sha256=_changed_hash(
                certificate.candidate_order_sha256
            ),
        )
    elif key == "hash/certificate/admissible_list_sha256":
        changed = replace(
            certificate,
            admissible_list_sha256=_changed_hash(
                certificate.admissible_list_sha256
            ),
        )
    elif key == "hash/certificate/invalid_bitmap_sha256":
        changed = replace(
            certificate,
            invalid_bitmap_sha256=_changed_hash(
                certificate.invalid_bitmap_sha256
            ),
        )
    elif key == "hash/certificate/geometry_sha256":
        changed = replace(
            certificate,
            geometry_sha256=_changed_hash(certificate.geometry_sha256),
        )
    elif key == "hash/certificate/source_scope_sha256":
        changed = replace(
            certificate,
            source_scope_sha256=_changed_hash(certificate.source_scope_sha256),
        )
    elif key == "hash/certificate/source_content_sha256":
        changed = replace(
            certificate,
            source_content_sha256=_changed_hash(
                certificate.source_content_sha256
            ),
        )
    elif key == "hash/certificate/objective_scope_sha256":
        changed = replace(
            certificate,
            objective_scope_sha256=_changed_hash(
                certificate.objective_scope_sha256
            ),
        )
    elif key == "hash/certificate/objective_content_sha256":
        changed = replace(
            certificate,
            objective_content_sha256=_changed_hash(
                certificate.objective_content_sha256
            ),
        )
    elif key == "hash/certificate/selected_evaluation_sha256":
        changed = replace(
            certificate,
            selected_evaluation_sha256=_changed_hash(
                certificate.selected_evaluation_sha256
            ),
        )
    elif key == "hash/certificate/selected_prediction_sha256":
        changed = replace(
            certificate,
            selected_prediction_sha256=_changed_hash(
                certificate.selected_prediction_sha256
            ),
        )
    else:
        raise IntegrityV22Error("unknown normative hash owner")
    return replace(result, certificate=changed)


def audit_deep_forgery_matrix(
    result: exact.GlobalResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> ForgeryMatrixEvidence:
    """Authenticate once, then exercise every nested deep-comparison surface."""

    if result.arm != "combined":
        raise IntegrityV22Error("deep gain/bias/retained forgery matrix requires combined")
    checked_config = _require_sha256(config_sha256, "config SHA-256")
    rebuilt = _authenticate(
        result,
        source,
        fit_target,
        fit_mask,
        output_mask,
        checked_config,
    )
    mutations: tuple[
        tuple[str, Callable[[exact.GlobalResult], exact.GlobalResult]], ...
    ] = (
        ("payload/objective", _mutate_cache_objective),
        ("payload/gain", _mutate_cache_gain),
        ("payload/bias", _mutate_cache_bias),
        ("payload/retained_id", _mutate_cache_retained),
        ("payload/selected_fit_prediction", _mutate_selected_fit_prediction),
        (
            "payload/selected_output_prediction",
            _mutate_selected_output_prediction,
        ),
        ("certificate/count", _mutate_count),
        ("certificate/rank", _mutate_rank),
        *(
            (
                key,
                lambda current, hash_key=key: _mutate_hash(current, hash_key),
            )
            for key in NORMATIVE_HASH_FORGERY_KEYS
        ),
    )
    if tuple(key for key, _ in mutations) != FORGERY_KEYS:
        raise IntegrityV22Error("forgery mutation registry differs from frozen coverage")
    return ForgeryMatrixEvidence(
        tuple(
            _assess_forgery(
                key,
                result,
                rebuilt,
                mutation,
            )
            for key, mutation in mutations
        )
    )


@dataclass(frozen=True, slots=True)
class IntegrityCheck:
    name: str
    passed: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or not self.name.isascii():
            raise IntegrityV22Error("integrity check name must be nonempty ASCII")
        if type(self.passed) is not bool:
            raise IntegrityV22Error("integrity check result must be a built-in boolean")


@dataclass(frozen=True, slots=True)
class CacheIdentityEvidence:
    checks: tuple[IntegrityCheck, ...]
    identities: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if tuple(check.name for check in self.checks) != CACHE_CHECK_NAMES or not all(
            check.passed for check in self.checks
        ):
            raise IntegrityV22Error("cache identity controls did not all pass")
        if (
            type(self.identities) is not tuple
            or len({name for name, _ in self.identities}) != len(self.identities)
        ):
            raise IntegrityV22Error("cache identity labels are duplicated")
        for name, digest in self.identities:
            if type(name) is not str or not name or not name.isascii():
                raise IntegrityV22Error("cache identity label is invalid")
            _require_sha256(digest, f"cache identity {name}")

    @property
    def passed(self) -> bool:
        return True


def audit_cache_identities(
    source: np.ndarray,
    changed_source: np.ndarray,
    fit_target: np.ndarray,
    changed_target: np.ndarray,
    *,
    config_sha256: str,
) -> CacheIdentityEvidence:
    """Exercise source reuse, cache misses, and target/arm scope ownership."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    requests = (
        exact.FitRequest.create("integrity/base/affine", "affine", fit_target),
        exact.FitRequest.create("integrity/duplicate/affine", "affine", fit_target),
        exact.FitRequest.create(
            "integrity/changed-target/affine", "affine", changed_target
        ),
        exact.FitRequest.create("integrity/base/combined", "combined", fit_target),
    )
    base, duplicate, target_changed, combined = exact.fit_global_contexts(
        source,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        requests,
        config_sha256=checked_config,
    )
    independent_affine = exact.fit_global(
        source,
        fit_target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "affine",
        context_key=base.context_key,
        config_sha256=checked_config,
    )
    independent_combined = exact.fit_global(
        source,
        fit_target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "combined",
        context_key=combined.context_key,
        config_sha256=checked_config,
    )
    source_changed = exact.fit_global(
        changed_source,
        fit_target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        "affine",
        context_key="integrity/changed-source/affine",
        config_sha256=checked_config,
    )
    p0_target = np.ascontiguousarray(
        fit_target[:, geometry.PARITY_MASKS[1]], dtype=np.float64
    )
    p1_target = np.ascontiguousarray(
        fit_target[:, geometry.PARITY_MASKS[0]], dtype=np.float64
    )
    p0 = exact.fit_global(
        source,
        p0_target,
        geometry.PARITY_MASKS[1],
        geometry.PARITY_MASKS[0],
        "affine",
        context_key="integrity/p0/affine",
        config_sha256=checked_config,
    )
    p1 = exact.fit_global(
        source,
        p1_target,
        geometry.PARITY_MASKS[0],
        geometry.PARITY_MASKS[1],
        "affine",
        context_key="integrity/p1/affine",
        config_sha256=checked_config,
    )
    shared = (base, duplicate, target_changed, combined)
    checks = (
        IntegrityCheck(
            "shared_source_object_reused",
            all(result.source_grid is base.source_grid for result in shared),
        ),
        IntegrityCheck(
            "independent_source_object_not_reused",
            independent_affine.source_grid is not base.source_grid
            and independent_combined.source_grid is not combined.source_grid,
        ),
        IntegrityCheck(
            "shared_vs_independent_affine_bits",
            exact._results_bit_exact(base, independent_affine),
        ),
        IntegrityCheck(
            "shared_vs_independent_combined_bits",
            exact._results_bit_exact(combined, independent_combined),
        ),
        IntegrityCheck(
            "changed_source_cache_miss",
            source_changed.source_grid.scope_sha256
            != base.source_grid.scope_sha256,
        ),
        IntegrityCheck(
            "changed_fit_mask_cache_miss",
            p0.source_grid.scope_sha256 != base.source_grid.scope_sha256,
        ),
        IntegrityCheck(
            "changed_output_mask_cache_miss",
            p1.source_grid.scope_sha256 != p0.source_grid.scope_sha256,
        ),
        IntegrityCheck(
            "target_specific_objective_scope",
            target_changed.objective_cache.scope_sha256
            != base.objective_cache.scope_sha256,
        ),
        IntegrityCheck(
            "arm_specific_objective_scope",
            combined.objective_cache.scope_sha256
            != base.objective_cache.scope_sha256,
        ),
        IntegrityCheck(
            "context_key_excluded_from_objective_scope",
            duplicate.objective_cache.scope_sha256
            == base.objective_cache.scope_sha256,
        ),
        IntegrityCheck(
            "context_key_excluded_from_objective_content",
            duplicate.objective_cache.content_sha256
            == base.objective_cache.content_sha256
            and _array_bits_equal(
                duplicate.objective_cache.objectives,
                base.objective_cache.objectives,
            ),
        ),
    )
    identities = (
        ("source/base", base.source_grid.scope_sha256),
        ("source/changed", source_changed.source_grid.scope_sha256),
        ("source/p0", p0.source_grid.scope_sha256),
        ("source/p1", p1.source_grid.scope_sha256),
        ("objective/base-affine", base.objective_cache.scope_sha256),
        (
            "objective/changed-target-affine",
            target_changed.objective_cache.scope_sha256,
        ),
        ("objective/base-combined", combined.objective_cache.scope_sha256),
        ("objective/duplicate-affine", duplicate.objective_cache.scope_sha256),
    )
    return CacheIdentityEvidence(checks, identities)


def validate_delivery_integrity_evidence(
    evidence: DeliveryIntegrityEvidence,
    result: exact.GlobalResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Rerun delivery controls and compare every evidence field bit-exactly."""

    if not isinstance(evidence, DeliveryIntegrityEvidence):
        raise IntegrityV22Error("delivery validation requires delivery evidence")
    regenerated = audit_delivery_integrity(
        result,
        source,
        fit_target,
        fit_mask,
        output_mask,
        config_sha256=config_sha256,
    )
    if not _scientific_equal(evidence, regenerated):
        raise IntegrityV22Error("delivery evidence differs from complete replay")


def validate_forgery_matrix_evidence(
    evidence: ForgeryMatrixEvidence,
    result: exact.GlobalResult,
    source: np.ndarray,
    fit_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Rerun the complete forgery matrix and compare every witness."""

    if not isinstance(evidence, ForgeryMatrixEvidence):
        raise IntegrityV22Error("forgery validation requires matrix evidence")
    regenerated = audit_deep_forgery_matrix(
        result,
        source,
        fit_target,
        fit_mask,
        output_mask,
        config_sha256=config_sha256,
    )
    if not _scientific_equal(evidence, regenerated):
        raise IntegrityV22Error("forgery evidence differs from complete replay")


def validate_cache_identity_evidence(
    evidence: CacheIdentityEvidence,
    source: np.ndarray,
    changed_source: np.ndarray,
    fit_target: np.ndarray,
    changed_target: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Rerun all cache-identity controls and compare every witness."""

    if not isinstance(evidence, CacheIdentityEvidence):
        raise IntegrityV22Error("cache validation requires cache identity evidence")
    regenerated = audit_cache_identities(
        source,
        changed_source,
        fit_target,
        changed_target,
        config_sha256=config_sha256,
    )
    if not _scientific_equal(evidence, regenerated):
        raise IntegrityV22Error("cache identity evidence differs from complete replay")


@dataclass(frozen=True, slots=True)
class IntegrityEvidence:
    delivery: DeliveryIntegrityEvidence
    forgery_matrix: ForgeryMatrixEvidence
    cache_identities: CacheIdentityEvidence

    def __post_init__(self) -> None:
        if (
            not isinstance(self.delivery, DeliveryIntegrityEvidence)
            or not isinstance(self.forgery_matrix, ForgeryMatrixEvidence)
            or not isinstance(self.cache_identities, CacheIdentityEvidence)
        ):
            raise IntegrityV22Error("aggregate integrity evidence has an invalid member")

    @property
    def passed(self) -> bool:
        return (
            self.delivery.passed
            and self.forgery_matrix.passed
            and self.cache_identities.passed
        )


def run_integrity_controls(
    result: exact.GlobalResult,
    source: np.ndarray,
    changed_source: np.ndarray,
    fit_target: np.ndarray,
    changed_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> IntegrityEvidence:
    """Run the complete production-backed exact-grid integrity panel."""

    return IntegrityEvidence(
        delivery=audit_delivery_integrity(
            result,
            source,
            fit_target,
            fit_mask,
            output_mask,
            config_sha256=config_sha256,
        ),
        forgery_matrix=audit_deep_forgery_matrix(
            result,
            source,
            fit_target,
            fit_mask,
            output_mask,
            config_sha256=config_sha256,
        ),
        cache_identities=audit_cache_identities(
            source,
            changed_source,
            fit_target,
            changed_target,
            config_sha256=config_sha256,
        ),
    )


def validate_integrity_evidence(
    evidence: IntegrityEvidence,
    result: exact.GlobalResult,
    source: np.ndarray,
    changed_source: np.ndarray,
    fit_target: np.ndarray,
    changed_target: np.ndarray,
    fit_mask: np.ndarray,
    output_mask: np.ndarray,
    *,
    config_sha256: str,
) -> None:
    """Rerun the full panel; no stored PASS label grants integrity coverage."""

    if not isinstance(evidence, IntegrityEvidence):
        raise IntegrityV22Error("aggregate validation requires integrity evidence")
    regenerated = run_integrity_controls(
        result,
        source,
        changed_source,
        fit_target,
        changed_target,
        fit_mask,
        output_mask,
        config_sha256=config_sha256,
    )
    if not _scientific_equal(evidence, regenerated):
        raise IntegrityV22Error("aggregate integrity evidence differs from complete replay")


__all__ = [
    "CACHE_CHECK_NAMES",
    "DELIVERY_REJECTION_NAMES",
    "FORGERY_KEYS",
    "NORMATIVE_HASH_FORGERY_KEYS",
    "CacheIdentityEvidence",
    "DeliveryIntegrityEvidence",
    "DeliveryOrderWitness",
    "ForgeryMatrixEvidence",
    "ForgeryWitness",
    "IntegrityCheck",
    "IntegrityEvidence",
    "IntegrityV22Error",
    "PROTOCOL_SHA256",
    "RejectionWitness",
    "SCHEMA_VERSION",
    "audit_cache_identities",
    "audit_deep_forgery_matrix",
    "audit_delivery_integrity",
    "run_integrity_controls",
    "validate_cache_identity_evidence",
    "validate_delivery_integrity_evidence",
    "validate_forgery_matrix_evidence",
    "validate_integrity_evidence",
]
