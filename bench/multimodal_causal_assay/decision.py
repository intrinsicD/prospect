"""Pure eight-video gate arithmetic for frozen MM-011 scoring primitives.

The decision layer consumes only typed, already-aggregated score evidence and
independent completeness/range primitives.  It owns no arrays, fitting, filesystem,
lifecycle, seed, report, or claim-promotion authority.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Final, Literal, cast

from bench.multimodal_causal_assay import scoring

PROTOCOL_SHA256: Final[str] = "ca39f7cea6a2a5b041956b419bf3530dd54eb8403096963a044d7fcf1e2121cc"
SCHEMA_VERSION: Final = "mm011-decision-v1"

if scoring.PROTOCOL_SHA256 != PROTOCOL_SHA256:
    raise RuntimeError("MM-011 decision and scoring bind different protocol bytes")

Family = scoring.Family
DecisionLabel = Literal[
    "invalid_MM011",
    "invalid_MM011_real_negative_control",
    "MM011_inconclusive",
    "causal_affine_family_supported",
    "causal_appearance_family_supported",
    "causal_combined_family_supported",
    "joint_affine_appearance_causal_operator_supported",
    "causal_family_nonidentifiable",
    "tested_causal_operator_failure_supported",
]
FamilyFailureDiagnosis = Literal[
    "tested_family_identifiability_failure",
    "historically_identifiable_but_complete_future_gate_failed",
]

FAMILIES: Final = scoring.FAMILIES
FOLDS: Final = (0, 1, 2, 3)
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

ACTIVITY_MSE_MIN: Final = 1e-4
PRIMARY_FACTOR: Final = 1.25
CONTROL_FACTOR: Final = 1.10
REQUIRED_SUPPORT: Final = 6
REQUIRED_DIRECTIONAL: Final = 7
RANGE_WARNING_VIDEOS: Final = 3
NULL_INVALID_COUNT: Final = 6
NULL_INCONCLUSIVE_MIN: Final = 3


class DecisionError(ValueError):
    """Raised when MM-011 decision evidence violates its strict schema."""


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise DecisionError(f"{label} must be a built-in boolean")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DecisionError(f"{label} must be a nonnegative built-in integer")
    return value


def _scaled_le(scale: float, left: scoring.ErrorPrimitive, right: scoring.ErrorPrimitive, label: str) -> bool:
    if type(left) is not scoring.ErrorPrimitive or type(right) is not scoring.ErrorPrimitive:
        raise DecisionError(f"{label} requires exact ErrorPrimitive operands")
    if left.count != right.count:
        raise DecisionError(f"{label} operands have different scalar counts")
    scaled = scale * left.sse
    if not math.isfinite(scaled):
        raise DecisionError(f"{label} overflowed finite decision arithmetic")
    return scaled <= right.sse


def _strict_lt(left: scoring.ErrorPrimitive, right: scoring.ErrorPrimitive, label: str) -> bool:
    if type(left) is not scoring.ErrorPrimitive or type(right) is not scoring.ErrorPrimitive:
        raise DecisionError(f"{label} requires exact ErrorPrimitive operands")
    if left.count != right.count:
        raise DecisionError(f"{label} operands have different scalar counts")
    return left.sse < right.sse


def _active(error: scoring.ErrorPrimitive, label: str) -> bool:
    threshold = ACTIVITY_MSE_MIN * error.count
    if not math.isfinite(threshold):
        raise DecisionError(f"{label} activity threshold overflowed")
    return error.sse > threshold


@dataclass(frozen=True, slots=True)
class CompletenessPrimitives:
    """Independent non-score checks required for one family/video record."""

    rows_complete: bool
    certificates_valid: bool
    apply_replays_valid: bool
    hashes_valid: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_bool(getattr(self, field.name), field.name)

    @property
    def valid(self) -> bool:
        return all(cast(bool, getattr(self, field.name)) for field in fields(self))


@dataclass(frozen=True, slots=True)
class FamilyVideoEvidence:
    """One exact family/video score aggregate plus non-score gate primitives."""

    scores: scoring.VideoScores
    bounded_rows: int
    completeness: CompletenessPrimitives

    def __post_init__(self) -> None:
        if type(self.scores) is not scoring.VideoScores:
            raise DecisionError("family/video scores have the wrong type")
        bounded = _require_nonnegative_int(self.bounded_rows, "bounded row count")
        if bounded > self.scores.row_count:
            raise DecisionError("bounded row count exceeds the exact video row count")
        if type(self.completeness) is not CompletenessPrimitives:
            raise DecisionError("family/video completeness has the wrong type")

    @property
    def range_warning(self) -> bool:
        """Return exact ``bounded_rows / rows >= 0.25`` without float division."""

        return 4 * self.bounded_rows >= self.scores.row_count


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    family: Family
    videos: tuple[FamilyVideoEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in FAMILIES:
            raise DecisionError("family evidence has an unknown family")
        if type(self.videos) is not tuple or any(type(video) is not FamilyVideoEvidence for video in self.videos):
            raise DecisionError("family videos must be an immutable typed tuple")


@dataclass(frozen=True, slots=True)
class PrerequisitePrimitives:
    parent_alignment_valid: bool
    synthetic_controls_valid: bool
    future_isolation_valid: bool
    source_binding_valid: bool
    package_integrity_valid: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_bool(getattr(self, field.name), field.name)


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    prerequisites: PrerequisitePrimitives
    families: tuple[FamilyEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.prerequisites) is not PrerequisitePrimitives:
            raise DecisionError("decision prerequisites have the wrong type")
        if type(self.families) is not tuple or any(type(family) is not FamilyEvidence for family in self.families):
            raise DecisionError("decision families must be an immutable typed tuple")


@dataclass(frozen=True, slots=True)
class VideoFamilyFacts:
    video_id: str
    fold: int
    family: Family
    activity: bool
    historical_support: bool
    future_support: bool
    joint_support: bool
    directional_improvement: bool
    history_shuffle_null: bool
    forecast_shuffle_null: bool
    forecast_reverse_null: bool
    future_derangement_null: bool
    evidence_complete: bool
    range_warning: bool


@dataclass(frozen=True, slots=True)
class FamilyAggregate:
    family: Family
    activity_count: int
    historical_support_count: int
    future_support_count: int
    joint_support_count: int
    directional_improvement_count: int
    joint_support_folds: tuple[int, ...]
    history_shuffle_null_count: int
    forecast_shuffle_null_count: int
    forecast_reverse_null_count: int
    future_derangement_null_count: int
    complete: bool
    range_warning_count: int
    passes: bool
    failure_diagnosis: FamilyFailureDiagnosis | None
    clean_fail: bool


@dataclass(frozen=True, slots=True)
class DominanceAggregate:
    margin_count: int
    strict_improvement_count: int
    margin_folds: tuple[int, ...]
    passes: bool


@dataclass(frozen=True, slots=True)
class DecisionSummary:
    schema_version: str
    video_facts: tuple[VideoFamilyFacts, ...]
    families: tuple[FamilyAggregate, ...]
    combined_dominance: DominanceAggregate
    invalid_reasons: tuple[str, ...]
    inconclusive_reasons: tuple[str, ...]
    passing_families: tuple[Family, ...]
    go: bool
    decision: DecisionLabel


def _validate_video_scores(scores: scoring.VideoScores, spec: tuple[str, int, int], family: Family) -> None:
    if type(scores) is not scoring.VideoScores:
        raise DecisionError("video scores have the wrong type")
    if (scores.video_id, scores.fold, scores.row_count) != spec or scores.family != family:
        raise DecisionError("video scores are missing, extra, reordered, or bound to the wrong family")
    expected_count = scores.row_count * scoring.ELEMENTS_PER_ROW
    for name in ("i", "a", "q", "p", "c", "h", "r", "z", "d", "pd", "u", "b", "bd"):
        error = getattr(scores, name)
        if type(error) is not scoring.ErrorPrimitive or error.count != expected_count:
            raise DecisionError(f"video metric {name} has an invalid type or count")


def _validate_evidence(evidence: DecisionEvidence) -> None:
    if type(evidence) is not DecisionEvidence:
        raise DecisionError("decision input must be exact DecisionEvidence")
    if type(evidence.prerequisites) is not PrerequisitePrimitives:
        raise DecisionError("decision prerequisites have the wrong type")
    if tuple(family.family for family in evidence.families) != FAMILIES:
        raise DecisionError("family evidence is missing, extra, duplicated, or reordered")
    for family in evidence.families:
        if len(family.videos) != len(VIDEO_SPECS):
            raise DecisionError("each family must contain exactly eight videos")
        for item, spec in zip(family.videos, VIDEO_SPECS, strict=True):
            if type(item) is not FamilyVideoEvidence:
                raise DecisionError("family/video evidence has the wrong type")
            _validate_video_scores(item.scores, spec, family.family)
            if type(item.completeness) is not CompletenessPrimitives:
                raise DecisionError("completeness evidence has the wrong type")
            _require_nonnegative_int(item.bounded_rows, "bounded row count")
    for video_index in range(len(VIDEO_SPECS)):
        shared = tuple(family.videos[video_index].scores for family in evidence.families)
        for name in ("i", "p", "z", "pd", "u", "b", "bd"):
            reference = getattr(shared[0], name)
            if any(not _error_bits_equal(reference, getattr(scores, name)) for scores in shared[1:]):
                raise DecisionError(f"shared metric {name} differs across families")


def _error_bits_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if type(left) is not scoring.ErrorPrimitive or type(right) is not scoring.ErrorPrimitive:
        return False
    return left.count == right.count and struct.pack("<d", left.sse) == struct.pack("<d", right.sse)


def _bias(scores: scoring.VideoScores, name: Literal["u", "b", "bd"]) -> scoring.ErrorPrimitive:
    value = getattr(scores, name)
    if type(value) is not scoring.ErrorPrimitive:
        raise DecisionError(f"{scores.family} requires exact {name} bias evidence")
    return value


def _facts(item: FamilyVideoEvidence) -> VideoFamilyFacts:
    scores = item.scores
    activity = _active(scores.i, f"{scores.video_id}/{scores.family}/history") and _active(
        scores.p, f"{scores.video_id}/{scores.family}/future"
    )
    historical_support = (
        activity
        and _scaled_le(PRIMARY_FACTOR, scores.a, scores.i, "historical/persistence")
        and _scaled_le(CONTROL_FACTOR, scores.a, scores.q, "historical/shuffle")
    )
    future_support = (
        activity
        and _scaled_le(PRIMARY_FACTOR, scores.c, scores.p, "future/persistence")
        and _scaled_le(CONTROL_FACTOR, scores.c, scores.h, "future/history-shuffle")
        and _scaled_le(CONTROL_FACTOR, scores.c, scores.r, "future/reverse")
        and _scaled_le(CONTROL_FACTOR, scores.c, scores.z, "future/velocity")
        and _scaled_le(CONTROL_FACTOR, scores.c, scores.d, "future/derangement")
    )
    directional = _strict_lt(scores.a, scores.i, "historical direction") and _strict_lt(
        scores.c, scores.p, "future direction"
    )
    history_shuffle_null = activity and _scaled_le(PRIMARY_FACTOR, scores.q, scores.i, "historical shuffled null")
    forecast_shuffle_null = (
        activity
        and _scaled_le(PRIMARY_FACTOR, scores.h, scores.p, "future shuffled null/persistence")
        and _scaled_le(CONTROL_FACTOR, scores.h, scores.r, "future shuffled null/reverse")
        and _scaled_le(CONTROL_FACTOR, scores.h, scores.z, "future shuffled null/velocity")
        and _scaled_le(CONTROL_FACTOR, scores.h, scores.d, "future shuffled null/derangement")
    )
    forecast_reverse_null = (
        activity
        and _scaled_le(PRIMARY_FACTOR, scores.r, scores.p, "future reverse null/persistence")
        and _scaled_le(CONTROL_FACTOR, scores.r, scores.h, "future reverse null/shuffle")
        and _scaled_le(CONTROL_FACTOR, scores.r, scores.z, "future reverse null/velocity")
        and _scaled_le(CONTROL_FACTOR, scores.r, scores.d, "future reverse null/derangement")
    )
    future_derangement_null = _active(scores.pd, f"{scores.video_id}/{scores.family}/deranged") and _scaled_le(
        PRIMARY_FACTOR, scores.d, scores.pd, "future derangement null/persistence"
    )
    u = _bias(scores, "u")
    b = _bias(scores, "b")
    bd = _bias(scores, "bd")
    source_credit = u.sse > 0.0 and _scaled_le(PRIMARY_FACTOR, scores.a, u, "historical/bias")
    future_credit = b.sse > 0.0 and _scaled_le(PRIMARY_FACTOR, scores.c, b, "future/bias")
    historical_support = historical_support and source_credit
    future_support = future_support and future_credit
    directional = (
        directional
        and _strict_lt(scores.a, u, "historical bias direction")
        and _strict_lt(scores.c, b, "future bias direction")
    )
    history_shuffle_null = (
        history_shuffle_null
        and u.sse > 0.0
        and _scaled_le(PRIMARY_FACTOR, scores.q, u, "historical shuffled null/bias")
    )
    forecast_shuffle_null = (
        forecast_shuffle_null and b.sse > 0.0 and _scaled_le(PRIMARY_FACTOR, scores.h, b, "future shuffled null/bias")
    )
    forecast_reverse_null = (
        forecast_reverse_null and b.sse > 0.0 and _scaled_le(PRIMARY_FACTOR, scores.r, b, "future reverse null/bias")
    )
    future_derangement_null = (
        future_derangement_null
        and bd.sse > 0.0
        and _scaled_le(PRIMARY_FACTOR, scores.d, bd, "future derangement null/bias")
    )
    return VideoFamilyFacts(
        video_id=scores.video_id,
        fold=scores.fold,
        family=scores.family,
        activity=activity,
        historical_support=historical_support,
        future_support=future_support,
        joint_support=historical_support and future_support,
        directional_improvement=directional,
        history_shuffle_null=history_shuffle_null,
        forecast_shuffle_null=forecast_shuffle_null,
        forecast_reverse_null=forecast_reverse_null,
        future_derangement_null=future_derangement_null,
        evidence_complete=item.completeness.valid,
        range_warning=item.range_warning,
    )


def _count(facts: tuple[VideoFamilyFacts, ...], field_name: str) -> int:
    return sum(bool(getattr(fact, field_name)) for fact in facts)


def _aggregate(family: Family, all_facts: tuple[VideoFamilyFacts, ...]) -> FamilyAggregate:
    facts = tuple(fact for fact in all_facts if fact.family == family)
    if len(facts) != len(VIDEO_SPECS):
        raise DecisionError("family fact census differs from eight videos")
    activity_count = _count(facts, "activity")
    historical_count = _count(facts, "historical_support")
    future_count = _count(facts, "future_support")
    joint_count = _count(facts, "joint_support")
    directional_count = _count(facts, "directional_improvement")
    support_folds = tuple(sorted({fact.fold for fact in facts if fact.joint_support}))
    history_null = _count(facts, "history_shuffle_null")
    shuffle_null = _count(facts, "forecast_shuffle_null")
    reverse_null = _count(facts, "forecast_reverse_null")
    deranged_null = _count(facts, "future_derangement_null")
    complete = all(fact.evidence_complete for fact in facts)
    range_count = _count(facts, "range_warning")
    null_counts = (history_null, shuffle_null, reverse_null, deranged_null)
    passes = (
        activity_count == len(VIDEO_SPECS)
        and joint_count >= REQUIRED_SUPPORT
        and directional_count >= REQUIRED_DIRECTIONAL
        and support_folds == FOLDS
        and complete
        and range_count < RANGE_WARNING_VIDEOS
        and all(count < NULL_INCONCLUSIVE_MIN for count in null_counts)
    )
    clean_failure_eligible = (
        activity_count == len(VIDEO_SPECS)
        and future_count <= 2
        and joint_count <= 2
        and complete
        and range_count < RANGE_WARNING_VIDEOS
        and all(count < NULL_INCONCLUSIVE_MIN for count in null_counts)
    )
    failure_diagnosis: FamilyFailureDiagnosis | None = None
    if clean_failure_eligible and historical_count <= 2:
        failure_diagnosis = "tested_family_identifiability_failure"
    elif clean_failure_eligible and historical_count >= REQUIRED_SUPPORT:
        failure_diagnosis = "historically_identifiable_but_complete_future_gate_failed"
    clean_fail = failure_diagnosis is not None
    return FamilyAggregate(
        family=family,
        activity_count=activity_count,
        historical_support_count=historical_count,
        future_support_count=future_count,
        joint_support_count=joint_count,
        directional_improvement_count=directional_count,
        joint_support_folds=support_folds,
        history_shuffle_null_count=history_null,
        forecast_shuffle_null_count=shuffle_null,
        forecast_reverse_null_count=reverse_null,
        future_derangement_null_count=deranged_null,
        complete=complete,
        range_warning_count=range_count,
        passes=passes,
        failure_diagnosis=failure_diagnosis,
        clean_fail=clean_fail,
    )


def _dominance(evidence: DecisionEvidence) -> DominanceAggregate:
    lookup = {
        family.family: {item.scores.video_id: item.scores for item in family.videos} for family in evidence.families
    }
    margins = 0
    improvements = 0
    folds: set[int] = set()
    for video_id, fold, _ in VIDEO_SPECS:
        affine = lookup["affine"][video_id].c
        appearance = lookup["appearance"][video_id].c
        combined = lookup["combined"][video_id].c
        margin = _scaled_le(CONTROL_FACTOR, combined, affine, "combined/affine dominance") and _scaled_le(
            CONTROL_FACTOR, combined, appearance, "combined/appearance dominance"
        )
        improvement = _strict_lt(combined, affine, "combined/affine improvement") and _strict_lt(
            combined, appearance, "combined/appearance improvement"
        )
        margins += int(margin)
        improvements += int(improvement)
        if margin:
            folds.add(fold)
    margin_folds = tuple(sorted(folds))
    return DominanceAggregate(
        margin_count=margins,
        strict_improvement_count=improvements,
        margin_folds=margin_folds,
        passes=margins >= REQUIRED_SUPPORT and improvements >= REQUIRED_DIRECTIONAL and margin_folds == FOLDS,
    )


def _prerequisite_failures(prerequisites: PrerequisitePrimitives) -> tuple[str, ...]:
    return tuple(
        f"prerequisite:{field.name}"
        for field in fields(prerequisites)
        if not _require_bool(getattr(prerequisites, field.name), field.name)
    )


def _incomplete_reasons(families: tuple[FamilyAggregate, ...]) -> tuple[str, ...]:
    return tuple(f"{family.family}:incomplete_evidence" for family in families if not family.complete)


def _invalid_null_reasons(families: tuple[FamilyAggregate, ...]) -> tuple[str, ...]:
    names = (
        "history_shuffle",
        "forecast_shuffle",
        "forecast_reverse",
        "future_derangement",
    )
    return tuple(
        f"{family.family}:{name}_null_ge_6"
        for family in families
        for name, count in zip(
            names,
            (
                family.history_shuffle_null_count,
                family.forecast_shuffle_null_count,
                family.forecast_reverse_null_count,
                family.future_derangement_null_count,
            ),
            strict=True,
        )
        if count >= NULL_INVALID_COUNT
    )


def _inconclusive_reasons(families: tuple[FamilyAggregate, ...]) -> tuple[str, ...]:
    reasons: list[str] = []
    names = (
        "history_shuffle",
        "forecast_shuffle",
        "forecast_reverse",
        "future_derangement",
    )
    for family in families:
        counts = (
            family.history_shuffle_null_count,
            family.forecast_shuffle_null_count,
            family.forecast_reverse_null_count,
            family.future_derangement_null_count,
        )
        for name, count in zip(names, counts, strict=True):
            if NULL_INCONCLUSIVE_MIN <= count < NULL_INVALID_COUNT:
                reasons.append(f"{family.family}:{name}_null_3_to_5")
        if family.range_warning_count >= RANGE_WARNING_VIDEOS:
            reasons.append(f"{family.family}:range_warning")
        if family.activity_count < len(VIDEO_SPECS):
            reasons.append(f"{family.family}:activity_ambiguity")
        decision_controls_clear = (
            family.activity_count == len(VIDEO_SPECS)
            and family.complete
            and family.range_warning_count < RANGE_WARNING_VIDEOS
            and all(count < NULL_INCONCLUSIVE_MIN for count in counts)
        )
        if decision_controls_clear and not family.passes and not family.clean_fail:
            if 3 <= family.joint_support_count <= 5:
                reasons.append(f"{family.family}:joint_support_3_to_5")
            elif family.joint_support_count >= REQUIRED_SUPPORT and (
                family.directional_improvement_count < REQUIRED_DIRECTIONAL or family.joint_support_folds != FOLDS
            ):
                reasons.append(f"{family.family}:directional_or_fold_gate")
            else:
                family_reason_count = len(reasons)
                if 3 <= family.historical_support_count <= 5:
                    reasons.append(f"{family.family}:historical_support_3_to_5")
                if 3 <= family.future_support_count <= 5:
                    reasons.append(f"{family.family}:future_support_3_to_5")
                if family.future_support_count >= REQUIRED_SUPPORT and family.joint_support_count < REQUIRED_SUPPORT:
                    reasons.append(f"{family.family}:future_support_without_joint_support")
                if len(reasons) == family_reason_count:
                    reasons.append(f"{family.family}:mixed_gate")
    return tuple(dict.fromkeys(reasons))


def derive_decision(evidence: DecisionEvidence) -> DecisionSummary:
    """Recompute all MM-011 family gates and apply the frozen precedence ladder."""

    _validate_evidence(evidence)
    video_facts = tuple(_facts(item) for family in evidence.families for item in family.videos)
    families = tuple(_aggregate(family, video_facts) for family in FAMILIES)
    dominance = _dominance(evidence)
    prerequisite_failures = _prerequisite_failures(evidence.prerequisites)
    incomplete = _incomplete_reasons(families)
    invalid_nulls = _invalid_null_reasons(families)
    passing = tuple(family.family for family in families if family.passes)
    inconclusive = _inconclusive_reasons(families)

    invalid_reasons: tuple[str, ...] = ()
    if prerequisite_failures or incomplete:
        invalid_reasons = prerequisite_failures + incomplete
        passing = ()
        inconclusive = ()
        decision: DecisionLabel = "invalid_MM011"
    elif invalid_nulls:
        invalid_reasons = invalid_nulls
        passing = ()
        inconclusive = ()
        decision = "invalid_MM011_real_negative_control"
    elif inconclusive:
        passing = ()
        decision = "MM011_inconclusive"
    elif passing:
        if "combined" in passing and dominance.passes:
            decision = "joint_affine_appearance_causal_operator_supported"
        elif len(passing) > 1:
            decision = "causal_family_nonidentifiable"
        elif passing == ("affine",):
            decision = "causal_affine_family_supported"
        elif passing == ("appearance",):
            decision = "causal_appearance_family_supported"
        else:
            decision = "causal_combined_family_supported"
    elif all(family.clean_fail for family in families):
        decision = "tested_causal_operator_failure_supported"
    else:
        decision = "MM011_inconclusive"
        inconclusive = ("unclassified_mixed_gate",)

    return DecisionSummary(
        schema_version=SCHEMA_VERSION,
        video_facts=video_facts,
        families=families,
        combined_dominance=dominance,
        invalid_reasons=invalid_reasons,
        inconclusive_reasons=inconclusive,
        passing_families=passing,
        go=bool(passing),
        decision=decision,
    )


def _deep_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("<d", left) == struct.pack("<d", cast(float, right))
    if isinstance(left, tuple):
        other = cast(tuple[object, ...], right)
        return len(left) == len(other) and all(_deep_equal(a, b) for a, b in zip(left, other, strict=True))
    if is_dataclass(left) and not isinstance(left, type):
        return all(_deep_equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))
    return left == right


def validate_decision(evidence: DecisionEvidence, claimed: DecisionSummary) -> None:
    """Recompute and deeply compare a claimed decision summary."""

    if type(claimed) is not DecisionSummary or not _deep_equal(derive_decision(evidence), claimed):
        raise DecisionError("claimed MM-011 decision differs from pure recomputation")


__all__ = [
    "ACTIVITY_MSE_MIN",
    "FAMILIES",
    "FOLDS",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "VIDEO_SPECS",
    "CompletenessPrimitives",
    "DecisionError",
    "DecisionEvidence",
    "DecisionLabel",
    "DecisionSummary",
    "DominanceAggregate",
    "FamilyAggregate",
    "FamilyFailureDiagnosis",
    "FamilyEvidence",
    "FamilyVideoEvidence",
    "PrerequisitePrimitives",
    "VideoFamilyFacts",
    "derive_decision",
    "validate_decision",
]
