"""Pure real-gate arithmetic for the frozen MM-008 v2.2 protocol.

The module consumes only caller-supplied aggregate scientific primitives.  It has
no dataset, decoder, filesystem, lifecycle, seed, RNG, or result-claim authority.
Every MSE, predicate, count, fold gate, relative gate, preemption, and decision is
recomputed from immutable SSE/count and control primitives.

Protocol wording mentions both a ``boundary`` and ``range`` preemption, while the
inherited v1 arithmetic defines one per-video boundary warning and only one
aggregate threshold: range preemption at three warned videos.  This module exposes
both the primitive warning count and that defined aggregate; it deliberately does
not invent a second boundary threshold.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Final, Literal, cast

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-real-decision-v1"

Family = Literal["affine", "appearance", "combined"]
BoundaryKind = Literal["site_flow", "gradient", "gain", "bias"]
RelativeGate = Literal[
    "affine_over_global_translation",
    "combined_over_affine",
    "combined_over_appearance",
]
DecisionLabel = Literal[
    "smooth_affine_family_supported",
    "global_appearance_supported",
    "joint_affine_appearance_recovery_supported",
    "mixed_mechanism_nonidentifiable",
    "tested_affine_appearance_ceiling_failure_supported",
    "MM008_mechanism_factorial_inconclusive",
]

FAMILIES: Final[tuple[Family, ...]] = ("affine", "appearance", "combined")
FOLDS: Final[tuple[int, ...]] = (0, 1, 2, 3)
VIDEO_SPECS: Final[tuple[tuple[str, int, int], ...]] = (
    ("video_10993", 0, 60),
    ("video_1580", 0, 61),
    ("video_2564", 1, 56),
    ("video_3501", 1, 62),
    ("video_6860", 2, 62),
    ("video_8241", 2, 45),
    ("video_874", 3, 63),
    ("video_9253", 3, 44),
)

REAL_ROWS: Final = 453
SCORED_ELEMENTS_PER_ROW: Final = 3 * 48 * 48
CONTEXTS_PER_FAMILY_ROW: Final = 7
SITE_FLOW_VALUES_PER_CONTEXT: Final = 2 * 48 * 48
GRADIENT_VALUES_PER_CONTEXT: Final = 4
GAIN_VALUES_PER_CONTEXT: Final = 3
BIAS_VALUES_PER_CONTEXT: Final = 3

EXPECTED_REAL_CENSUS: Final = (6_342, 9_513, 2_718, 3_171, 453)
BOUNDARY_WARNING_THRESHOLD: Final = 0.25
RANGE_WARNING_VIDEO_THRESHOLD: Final = 3


class DecisionV22Error(ValueError):
    """Raised when primitive evidence or a claimed summary fails closed."""


class InvalidRealAssayError(DecisionV22Error):
    """Raised when valid primitives establish a protocol-level invalid assay."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise DecisionV22Error(f"{label} must be a built-in boolean")
    return cast(bool, value)


def _require_nonnegative_float(value: object, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise DecisionV22Error(f"{label} must be a built-in finite number")
    converted = float(cast(float, value))
    if not math.isfinite(converted) or converted < 0.0:
        raise DecisionV22Error(f"{label} must be finite and nonnegative")
    return converted


def _scaled_le(scale: float, left: float, right: float, label: str) -> bool:
    scaled = scale * left
    if not math.isfinite(scaled):
        raise DecisionV22Error(f"{label} overflowed finite decision arithmetic")
    return scaled <= right


def _scaled_product_le(
    scale: float,
    left_a: float,
    left_b: float,
    right_a: float,
    right_b: float,
    label: str,
) -> bool:
    left = scale * left_a * left_b
    right = right_a * right_b
    if not math.isfinite(left) or not math.isfinite(right):
        raise DecisionV22Error(f"{label} overflowed finite decision arithmetic")
    return left <= right


@dataclass(frozen=True, slots=True)
class ErrorPrimitive:
    """One aggregate error numerator and its exact element denominator."""

    sse: float
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sse", _require_nonnegative_float(self.sse, "error SSE"))
        if type(self.count) is not int or self.count <= 0:
            raise DecisionV22Error("error count must be a positive built-in integer")
        if not math.isfinite(self.sse / self.count):
            raise DecisionV22Error("derived error MSE is non-finite")

    @property
    def mse(self) -> float:
        return self.sse / self.count


@dataclass(frozen=True, slots=True)
class BiasPrimitives:
    """Shared matched bias-only errors for one video."""

    true_full: ErrorPrimitive
    true_xfit: ErrorPrimitive
    near_xfit: ErrorPrimitive
    far_xfit: ErrorPrimitive

    def __post_init__(self) -> None:
        if any(
            type(value) is not ErrorPrimitive
            for value in (self.true_full, self.true_xfit, self.near_xfit, self.far_xfit)
        ):
            raise DecisionV22Error("bias evidence must contain four ErrorPrimitive values")


@dataclass(frozen=True, slots=True)
class CertificatePrimitive:
    """Completeness and independent validity for one exact-grid fit context."""

    context_index: int
    complete: bool
    valid: bool

    def __post_init__(self) -> None:
        if type(self.context_index) is not int or self.context_index < 0:
            raise DecisionV22Error("certificate context index must be a nonnegative built-in integer")
        complete = _require_bool(self.complete, "certificate completeness")
        valid = _require_bool(self.valid, "certificate validity")
        if valid and not complete:
            raise DecisionV22Error("an incomplete certificate cannot be marked valid")


@dataclass(frozen=True, slots=True)
class BoundaryPrimitive:
    """One saved boundary numerator and its complete parameter denominator."""

    kind: BoundaryKind
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.kind not in ("site_flow", "gradient", "gain", "bias"):
            raise DecisionV22Error("unknown boundary primitive kind")
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise DecisionV22Error("boundary numerator and denominator must be built-in integers")
        if self.denominator <= 0 or not 0 <= self.numerator <= self.denominator:
            raise DecisionV22Error("boundary numerator/denominator is outside its valid range")

    @property
    def fraction(self) -> float:
        return self.numerator / self.denominator


@dataclass(frozen=True, slots=True)
class FamilyVideoPrimitives:
    """All decision-relevant primitives for one family and video."""

    family: Family
    true_full: ErrorPrimitive
    true_xfit: ErrorPrimitive
    near_xfit: ErrorPrimitive
    far_xfit: ErrorPrimitive
    certificates: tuple[CertificatePrimitive, ...]
    boundaries: tuple[BoundaryPrimitive, ...]

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise DecisionV22Error("family evidence has an unknown family")
        if any(
            type(value) is not ErrorPrimitive
            for value in (self.true_full, self.true_xfit, self.near_xfit, self.far_xfit)
        ):
            raise DecisionV22Error("family evidence must contain four ErrorPrimitive values")
        if type(self.certificates) is not tuple or any(
            type(value) is not CertificatePrimitive for value in self.certificates
        ):
            raise DecisionV22Error("certificate evidence must be an immutable typed tuple")
        if type(self.boundaries) is not tuple or any(
            type(value) is not BoundaryPrimitive for value in self.boundaries
        ):
            raise DecisionV22Error("boundary evidence must be an immutable typed tuple")


@dataclass(frozen=True, slots=True)
class VideoPrimitives:
    """Complete decision primitives for one frozen real video."""

    video_id: str
    fold: int
    row_count: int
    persistence: ErrorPrimitive
    global_translation_xfit: ErrorPrimitive
    bias: BiasPrimitives
    families: tuple[FamilyVideoPrimitives, ...]

    def __post_init__(self) -> None:
        if type(self.video_id) is not str or not self.video_id:
            raise DecisionV22Error("video ID must be a nonempty built-in string")
        if type(self.fold) is not int or type(self.row_count) is not int:
            raise DecisionV22Error("video fold and row count must be built-in integers")
        if type(self.persistence) is not ErrorPrimitive or type(self.global_translation_xfit) is not ErrorPrimitive:
            raise DecisionV22Error("video errors must be ErrorPrimitive values")
        if type(self.bias) is not BiasPrimitives:
            raise DecisionV22Error("video bias evidence must be BiasPrimitives")
        if type(self.families) is not tuple or any(
            type(value) is not FamilyVideoPrimitives for value in self.families
        ):
            raise DecisionV22Error("video family evidence must be an immutable typed tuple")


@dataclass(frozen=True, slots=True)
class PrerequisitePrimitives:
    """Independent pre-real controls required before applying the real ladder."""

    parent_replay_valid: bool
    synthetic_controls_valid: bool
    exact_global_controls_valid: bool
    boundary_controls_valid: bool
    consistency_checks_valid: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_bool(getattr(self, field.name), field.name)


@dataclass(frozen=True, slots=True)
class RealPrimitiveEvidence:
    """The exact eight-video primitive input to the pure decision transform."""

    prerequisites: PrerequisitePrimitives
    videos: tuple[VideoPrimitives, ...]

    def __post_init__(self) -> None:
        if type(self.prerequisites) is not PrerequisitePrimitives:
            raise DecisionV22Error("real evidence prerequisites have the wrong type")
        if type(self.videos) is not tuple or any(type(value) is not VideoPrimitives for value in self.videos):
            raise DecisionV22Error("real videos must be an immutable typed tuple")


@dataclass(frozen=True, slots=True)
class PrimitiveCensus:
    exact_grid_contexts: int
    primary_contexts: int
    translation_contexts: int
    bias_contexts: int
    persistence_records: int


@dataclass(frozen=True, slots=True)
class VideoFamilyFacts:
    video_id: str
    fold: int
    family: Family
    x_support: bool
    full_support: bool
    residual_near: bool
    residual_far: bool
    complete_support: bool
    any_improvement: bool
    full_any_improvement: bool
    full_only: bool
    hit_near: bool
    hit_far: bool
    marginal: bool
    denominators_valid: bool
    certificates_valid: bool
    boundary_fraction: float
    boundary_warning: bool


@dataclass(frozen=True, slots=True)
class FamilyAggregate:
    family: Family
    x_count: int
    complete_count: int
    full_count: int
    any_improvement_count: int
    full_any_improvement_count: int
    full_only_count: int
    hit_near_count: int
    hit_far_count: int
    marginal_count: int
    complete_folds: tuple[int, ...]
    full_folds: tuple[int, ...]
    denominators_valid: bool
    global_certificates_valid: bool
    boundary_warning_count: int
    range_preemption: bool
    passes: bool
    clean_fail: bool


@dataclass(frozen=True, slots=True)
class RelativeAggregate:
    gate: RelativeGate
    margin_count: int
    improvement_count: int
    margin_folds: tuple[int, ...]
    passes: bool


@dataclass(frozen=True, slots=True)
class RealDecisionSummary:
    schema_version: str
    census: PrimitiveCensus
    video_facts: tuple[VideoFamilyFacts, ...]
    families: tuple[FamilyAggregate, ...]
    relatives: tuple[RelativeAggregate, ...]
    inconclusive_preemptions: tuple[str, ...]
    decision: DecisionLabel


def _validate_error(error: ErrorPrimitive, expected_count: int, label: str) -> None:
    if type(error) is not ErrorPrimitive:
        raise DecisionV22Error(f"{label} has the wrong primitive type")
    _require_nonnegative_float(error.sse, f"{label} SSE")
    if type(error.count) is not int or error.count != expected_count:
        raise DecisionV22Error(f"{label} count differs from the exact scored-element census")
    if not math.isfinite(error.mse):
        raise DecisionV22Error(f"{label} MSE is non-finite")


def _boundary_spec(family: Family, row_count: int) -> tuple[tuple[BoundaryKind, int], ...]:
    per_context = CONTEXTS_PER_FAMILY_ROW * row_count
    all_specs: dict[BoundaryKind, int] = {
        "site_flow": per_context * SITE_FLOW_VALUES_PER_CONTEXT,
        "gradient": per_context * GRADIENT_VALUES_PER_CONTEXT,
        "gain": per_context * GAIN_VALUES_PER_CONTEXT,
        "bias": per_context * BIAS_VALUES_PER_CONTEXT,
    }
    if family == "affine":
        names: tuple[BoundaryKind, ...] = ("site_flow", "gradient")
    elif family == "appearance":
        names = ("gain", "bias")
    else:
        names = ("site_flow", "gradient", "gain", "bias")
    return tuple((name, all_specs[name]) for name in names)


def _validate_family(family: FamilyVideoPrimitives, row_count: int, expected_error_count: int) -> None:
    for name in ("true_full", "true_xfit", "near_xfit", "far_xfit"):
        _validate_error(getattr(family, name), expected_error_count, f"{family.family} {name}")
    expected_certificate_count = 0 if family.family == "appearance" else CONTEXTS_PER_FAMILY_ROW * row_count
    if type(family.certificates) is not tuple or len(family.certificates) != expected_certificate_count:
        raise DecisionV22Error(f"{family.family} certificate context census is missing or extra")
    for expected_index, certificate in enumerate(family.certificates):
        if type(certificate) is not CertificatePrimitive or certificate.context_index != expected_index:
            raise DecisionV22Error(f"{family.family} certificates are duplicated, missing, or out of order")
        _require_bool(certificate.complete, "certificate completeness")
        _require_bool(certificate.valid, "certificate validity")
        if certificate.valid and not certificate.complete:
            raise DecisionV22Error("an incomplete certificate cannot be marked valid")
    expected_boundaries = _boundary_spec(family.family, row_count)
    actual_boundaries = tuple((item.kind, item.denominator) for item in family.boundaries)
    if actual_boundaries != expected_boundaries:
        raise DecisionV22Error(f"{family.family} boundary primitives are missing, extra, duplicated, or miscounted")
    for item in family.boundaries:
        if type(item) is not BoundaryPrimitive or not 0 <= item.numerator <= item.denominator:
            raise DecisionV22Error(f"{family.family} boundary primitive is invalid")


def _validate_evidence(evidence: RealPrimitiveEvidence) -> PrimitiveCensus:
    if type(evidence) is not RealPrimitiveEvidence:
        raise DecisionV22Error("decision input must be RealPrimitiveEvidence")
    if type(evidence.prerequisites) is not PrerequisitePrimitives:
        raise DecisionV22Error("prerequisite evidence has the wrong type")
    prerequisite_failures = tuple(
        field.name for field in fields(evidence.prerequisites) if not _require_bool(
            getattr(evidence.prerequisites, field.name), field.name
        )
    )
    if prerequisite_failures:
        raise InvalidRealAssayError(tuple(f"prerequisite:{name}" for name in prerequisite_failures))
    if type(evidence.videos) is not tuple or len(evidence.videos) != len(VIDEO_SPECS):
        raise DecisionV22Error("real evidence must contain exactly eight videos")
    actual_specs = tuple((video.video_id, video.fold, video.row_count) for video in evidence.videos)
    if actual_specs != VIDEO_SPECS:
        raise DecisionV22Error("real video evidence is missing, extra, duplicated, reordered, or miscounted")
    for video in evidence.videos:
        if type(video) is not VideoPrimitives:
            raise DecisionV22Error("real video evidence has the wrong type")
        expected_error_count = video.row_count * SCORED_ELEMENTS_PER_ROW
        _validate_error(video.persistence, expected_error_count, f"{video.video_id} persistence")
        _validate_error(
            video.global_translation_xfit,
            expected_error_count,
            f"{video.video_id} global translation xfit",
        )
        if type(video.bias) is not BiasPrimitives:
            raise DecisionV22Error("bias evidence has the wrong type")
        for name in ("true_full", "true_xfit", "near_xfit", "far_xfit"):
            _validate_error(getattr(video.bias, name), expected_error_count, f"{video.video_id} bias {name}")
        if tuple(family.family for family in video.families) != FAMILIES:
            raise DecisionV22Error("video family evidence is missing, extra, duplicated, or out of order")
        for family in video.families:
            _validate_family(family, video.row_count, expected_error_count)
    rows = sum(video.row_count for video in evidence.videos)
    census = PrimitiveCensus(
        exact_grid_contexts=rows * 14,
        primary_contexts=rows * 21,
        translation_contexts=rows * 6,
        bias_contexts=rows * 7,
        persistence_records=rows,
    )
    if (
        census.exact_grid_contexts,
        census.primary_contexts,
        census.translation_contexts,
        census.bias_contexts,
        census.persistence_records,
    ) != EXPECTED_REAL_CENSUS:
        raise DecisionV22Error("derived real primitive census differs from 6342/9513/2718/3171/453")
    return census


def _facts(video: VideoPrimitives, family: FamilyVideoPrimitives) -> VideoFamilyFacts:
    p = video.persistence.mse
    f = family.true_full.mse
    o = family.true_xfit.mse
    n = family.near_xfit.mse
    s = family.far_xfit.mse
    b_t_full = video.bias.true_full.mse
    b_t = video.bias.true_xfit.mse
    b_n = video.bias.near_xfit.mse
    b_f = video.bias.far_xfit.mse

    denominators_valid = b_t > 0.0 and b_n > 0.0 and b_f > 0.0
    if family.family != "affine":
        denominators_valid = denominators_valid and b_t_full > 0.0
    residual_near = b_t > 0.0 and b_n > 0.0 and _scaled_product_le(
        1.10,
        o,
        b_n,
        n,
        b_t,
        f"{video.video_id}/{family.family}/residual-near",
    )
    residual_far = b_t > 0.0 and b_f > 0.0 and _scaled_product_le(
        1.10,
        o,
        b_f,
        s,
        b_t,
        f"{video.video_id}/{family.family}/residual-far",
    )
    if family.family == "affine":
        x_support = p > 0.0 and _scaled_le(1.25, o, p, f"{video.video_id}/affine/X")
        full_support = p > 0.0 and _scaled_le(1.25, f, p, f"{video.video_id}/affine/F")
        any_improvement = o < p
        full_any_improvement = f < p
    else:
        x_support = (
            p > 0.0
            and _scaled_le(1.25, o, p, f"{video.video_id}/{family.family}/X-persistence")
            and b_t > 0.0
            and _scaled_le(1.25, o, b_t, f"{video.video_id}/{family.family}/X-bias")
        )
        full_support = (
            p > 0.0
            and _scaled_le(1.25, f, p, f"{video.video_id}/{family.family}/F-persistence")
            and b_t_full > 0.0
            and _scaled_le(1.25, f, b_t_full, f"{video.video_id}/{family.family}/F-bias")
        )
        any_improvement = o < min(p, b_t)
        full_any_improvement = f < min(p, b_t_full)
    complete_support = x_support and residual_near and residual_far
    full_only = full_support and not x_support
    hit_near = (
        p > 0.0
        and b_n > 0.0
        and _scaled_le(1.25, n, p, f"{video.video_id}/{family.family}/near-hit-persistence")
        and _scaled_le(1.25, n, b_n, f"{video.video_id}/{family.family}/near-hit-bias")
    )
    hit_far = (
        p > 0.0
        and b_f > 0.0
        and _scaled_le(1.25, s, p, f"{video.video_id}/{family.family}/far-hit-persistence")
        and _scaled_le(1.25, s, b_f, f"{video.video_id}/{family.family}/far-hit-bias")
    )
    marginal = (
        family.family == "appearance"
        and p > 0.0
        and b_t > 0.0
        and _scaled_le(1.25, b_t, p, f"{video.video_id}/appearance/marginal-bias")
        and _scaled_le(1.25, o, p, f"{video.video_id}/appearance/marginal-persistence")
        and not _scaled_le(1.25, o, b_t, f"{video.video_id}/appearance/marginal-above-bias")
    )
    certificates_valid = all(item.complete and item.valid for item in family.certificates)
    boundary_fraction = max(item.fraction for item in family.boundaries)
    return VideoFamilyFacts(
        video_id=video.video_id,
        fold=video.fold,
        family=family.family,
        x_support=x_support,
        full_support=full_support,
        residual_near=residual_near,
        residual_far=residual_far,
        complete_support=complete_support,
        any_improvement=any_improvement,
        full_any_improvement=full_any_improvement,
        full_only=full_only,
        hit_near=hit_near,
        hit_far=hit_far,
        marginal=marginal,
        denominators_valid=denominators_valid,
        certificates_valid=certificates_valid,
        boundary_fraction=boundary_fraction,
        boundary_warning=boundary_fraction >= BOUNDARY_WARNING_THRESHOLD,
    )


def _count(records: tuple[VideoFamilyFacts, ...], name: str) -> int:
    return sum(bool(getattr(record, name)) for record in records)


def _folds(records: tuple[VideoFamilyFacts, ...], name: str) -> tuple[int, ...]:
    return tuple(sorted({record.fold for record in records if bool(getattr(record, name))}))


def _aggregate_family(family: Family, facts: tuple[VideoFamilyFacts, ...]) -> FamilyAggregate:
    records = tuple(record for record in facts if record.family == family)
    if len(records) != 8:
        raise DecisionV22Error("family fact census is not exactly eight videos")
    x_count = _count(records, "x_support")
    complete_count = _count(records, "complete_support")
    full_count = _count(records, "full_support")
    any_count = _count(records, "any_improvement")
    full_any_count = _count(records, "full_any_improvement")
    full_only_count = _count(records, "full_only")
    hit_near_count = _count(records, "hit_near")
    hit_far_count = _count(records, "hit_far")
    marginal_count = _count(records, "marginal") if family == "appearance" else 0
    complete_folds = _folds(records, "complete_support")
    full_folds = _folds(records, "full_support")
    denominators_valid = all(record.denominators_valid for record in records)
    certificates_valid = all(record.certificates_valid for record in records)
    boundary_warning_count = _count(records, "boundary_warning")
    range_preemption = boundary_warning_count >= RANGE_WARNING_VIDEO_THRESHOLD
    passes = (
        x_count >= 6
        and complete_count >= 6
        and any_count >= 7
        and complete_folds == FOLDS
        and full_count >= 6
        and full_any_count >= 7
        and full_folds == FOLDS
        and full_only_count < 3
        and denominators_valid
        and certificates_valid
        and not range_preemption
    )
    clean_fail = (
        x_count < 3
        and complete_count < 3
        and full_count < 3
        and hit_near_count < 3
        and hit_far_count < 3
        and denominators_valid
        and certificates_valid
        and not range_preemption
        and (family != "appearance" or marginal_count < 3)
    )
    return FamilyAggregate(
        family=family,
        x_count=x_count,
        complete_count=complete_count,
        full_count=full_count,
        any_improvement_count=any_count,
        full_any_improvement_count=full_any_count,
        full_only_count=full_only_count,
        hit_near_count=hit_near_count,
        hit_far_count=hit_far_count,
        marginal_count=marginal_count,
        complete_folds=complete_folds,
        full_folds=full_folds,
        denominators_valid=denominators_valid,
        global_certificates_valid=certificates_valid,
        boundary_warning_count=boundary_warning_count,
        range_preemption=range_preemption,
        passes=passes,
        clean_fail=clean_fail,
    )


def _relative(
    gate: RelativeGate,
    evidence: RealPrimitiveEvidence,
    candidate: Family,
    comparator: Family | Literal["global_translation"],
) -> RelativeAggregate:
    margins: list[bool] = []
    improvements: list[bool] = []
    folds_seen: set[int] = set()
    for video in evidence.videos:
        by_family = {item.family: item for item in video.families}
        candidate_mse = by_family[candidate].true_xfit.mse
        comparator_mse = (
            video.global_translation_xfit.mse
            if comparator == "global_translation"
            else by_family[comparator].true_xfit.mse
        )
        margin = video.persistence.mse > 0.0 and _scaled_le(
            1.10,
            candidate_mse,
            comparator_mse,
            f"{video.video_id}/{gate}",
        )
        improvement = video.persistence.mse > 0.0 and candidate_mse < comparator_mse
        margins.append(margin)
        improvements.append(improvement)
        if margin:
            folds_seen.add(video.fold)
    margin_count = sum(margins)
    improvement_count = sum(improvements)
    margin_folds = tuple(sorted(folds_seen))
    return RelativeAggregate(
        gate=gate,
        margin_count=margin_count,
        improvement_count=improvement_count,
        margin_folds=margin_folds,
        passes=margin_count >= 6 and improvement_count >= 7 and margin_folds == FOLDS,
    )


def _preemptions(families: tuple[FamilyAggregate, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for aggregate in families:
        prefix = aggregate.family
        if not aggregate.denominators_valid:
            found.append(f"{prefix}:required_denominator")
        if not aggregate.global_certificates_valid:
            found.append(f"{prefix}:global_certificate")
        if aggregate.range_preemption:
            found.append(f"{prefix}:range_warning")
        for label, count in (
            ("X", aggregate.x_count),
            ("C", aggregate.complete_count),
            ("F", aggregate.full_count),
        ):
            if 3 <= count <= 5:
                found.append(f"{prefix}:{label}_support_3_to_5")
        if aggregate.x_count >= 6 and aggregate.complete_count < 6:
            found.append(f"{prefix}:residual_pairing_after_performance")
        if aggregate.full_only_count >= 3:
            found.append(f"{prefix}:full_xfit_inversion")
        if aggregate.complete_count >= 6 and (
            aggregate.any_improvement_count < 7 or aggregate.complete_folds != FOLDS
        ):
            found.append(f"{prefix}:complete_gate_improvement_or_fold")
        if aggregate.full_count >= 6 and (
            aggregate.full_any_improvement_count < 7 or aggregate.full_folds != FOLDS
        ):
            found.append(f"{prefix}:full_gate_improvement_or_fold")
        if aggregate.family == "appearance" and aggregate.marginal_count >= 3:
            found.append("appearance:target_marginal_only")
        if 3 <= aggregate.hit_near_count <= 5:
            found.append(f"{prefix}:near_null_hits_3_to_5")
        if 3 <= aggregate.hit_far_count <= 5:
            found.append(f"{prefix}:far_null_hits_3_to_5")
    return tuple(found)


def derive_decision(evidence: RealPrimitiveEvidence) -> RealDecisionSummary:
    """Recompute the exact MM-008 v2.2 real gates and frozen decision ladder."""

    census = _validate_evidence(evidence)
    video_facts = tuple(_facts(video, family) for video in evidence.videos for family in video.families)
    families = tuple(_aggregate_family(family, video_facts) for family in FAMILIES)
    invalid_reasons = tuple(
        f"{aggregate.family}:{mapping}_null_hits_ge_6"
        for aggregate in families
        for mapping, count in (("near", aggregate.hit_near_count), ("far", aggregate.hit_far_count))
        if count >= 6
    )
    if invalid_reasons:
        raise InvalidRealAssayError(invalid_reasons)
    relatives = (
        _relative("affine_over_global_translation", evidence, "affine", "global_translation"),
        _relative("combined_over_affine", evidence, "combined", "affine"),
        _relative("combined_over_appearance", evidence, "combined", "appearance"),
    )
    by_family = {item.family: item for item in families}
    by_relative = {item.gate: item for item in relatives}
    preemptions = _preemptions(families)
    if preemptions:
        decision: DecisionLabel = "MM008_mechanism_factorial_inconclusive"
    elif (
        by_family["affine"].passes
        and by_family["combined"].passes
        and by_relative["affine_over_global_translation"].passes
        and by_family["appearance"].clean_fail
    ):
        decision = "smooth_affine_family_supported"
    elif (
        by_family["appearance"].passes
        and by_family["combined"].passes
        and by_family["affine"].clean_fail
    ):
        decision = "global_appearance_supported"
    elif (
        by_family["combined"].passes
        and by_relative["combined_over_affine"].passes
        and by_relative["combined_over_appearance"].passes
        and by_family["affine"].clean_fail
        and by_family["appearance"].clean_fail
    ):
        decision = "joint_affine_appearance_recovery_supported"
    elif (
        all(by_family[family].passes for family in FAMILIES)
        and by_relative["affine_over_global_translation"].passes
    ):
        decision = "mixed_mechanism_nonidentifiable"
    elif all(by_family[family].clean_fail for family in FAMILIES):
        decision = "tested_affine_appearance_ceiling_failure_supported"
    else:
        decision = "MM008_mechanism_factorial_inconclusive"
    return RealDecisionSummary(
        schema_version=SCHEMA_VERSION,
        census=census,
        video_facts=video_facts,
        families=families,
        relatives=relatives,
        inconclusive_preemptions=preemptions,
        decision=decision,
    )


def _deep_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("<d", left) == struct.pack("<d", cast(float, right))
    if isinstance(left, tuple):
        right_tuple = cast(tuple[object, ...], right)
        return len(left) == len(right_tuple) and all(
            _deep_equal(a, b) for a, b in zip(left, right_tuple, strict=True)
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(_deep_equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))
    return left == right


def validate_decision(evidence: RealPrimitiveEvidence, claimed: RealDecisionSummary) -> None:
    """Reject any missing, extra, nullable, nonfinite, or forged decision summary."""

    if type(claimed) is not RealDecisionSummary:
        raise DecisionV22Error("claimed decision has the wrong exact summary type")
    expected = derive_decision(evidence)
    if not _deep_equal(claimed, expected):
        raise DecisionV22Error("claimed decision differs from deep primitive recomputation")


__all__ = [
    "BIAS_VALUES_PER_CONTEXT",
    "BOUNDARY_WARNING_THRESHOLD",
    "BoundaryPrimitive",
    "CertificatePrimitive",
    "CONTEXTS_PER_FAMILY_ROW",
    "DecisionLabel",
    "DecisionV22Error",
    "EXPECTED_REAL_CENSUS",
    "ErrorPrimitive",
    "FAMILIES",
    "FamilyAggregate",
    "FamilyVideoPrimitives",
    "GAIN_VALUES_PER_CONTEXT",
    "GRADIENT_VALUES_PER_CONTEXT",
    "InvalidRealAssayError",
    "PrimitiveCensus",
    "PrerequisitePrimitives",
    "PROTOCOL_SHA256",
    "RANGE_WARNING_VIDEO_THRESHOLD",
    "REAL_ROWS",
    "RealDecisionSummary",
    "RealPrimitiveEvidence",
    "RelativeAggregate",
    "SCHEMA_VERSION",
    "SCORED_ELEMENTS_PER_ROW",
    "SITE_FLOW_VALUES_PER_CONTEXT",
    "VIDEO_SPECS",
    "VideoFamilyFacts",
    "VideoPrimitives",
    "derive_decision",
    "validate_decision",
]
