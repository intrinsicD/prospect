"""Authored-fixture tests for pure MM-008 v2.2 real decision arithmetic."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from typing import Literal, cast

import pytest

from bench.multimodal_mechanism_diagnostics import decision_v22 as decision


@dataclass(frozen=True, slots=True)
class _Numbers:
    full: float
    xfit: float
    near: float
    far: float


@dataclass(frozen=True, slots=True)
class _BiasNumbers:
    full: float = 100.0
    xfit: float = 100.0
    near: float = 100.0
    far: float = 100.0


_GOOD = _Numbers(40.0, 40.0, 90.0, 90.0)
_BAD = _Numbers(100.0, 100.0, 100.0, 100.0)
_JOINT = _Numbers(30.0, 30.0, 90.0, 90.0)
_IMPROVEMENT_ONLY = _Numbers(90.0, 90.0, 100.0, 100.0)
_ALL_VALID = decision.PrerequisitePrimitives(True, True, True, True, True)


def _uniform(value: _Numbers) -> tuple[_Numbers, ...]:
    return (value,) * len(decision.VIDEO_SPECS)


def _error(mse: float, rows: int) -> decision.ErrorPrimitive:
    count = rows * decision.SCORED_ELEMENTS_PER_ROW
    return decision.ErrorPrimitive(sse=mse * count, count=count)


def _boundaries(
    family: decision.Family,
    rows: int,
    *,
    warning: Literal["none", "edge", "below"] = "none",
) -> tuple[decision.BoundaryPrimitive, ...]:
    contexts = rows * decision.CONTEXTS_PER_FAMILY_ROW
    denominators: dict[decision.BoundaryKind, int] = {
        "site_flow": contexts * decision.SITE_FLOW_VALUES_PER_CONTEXT,
        "gradient": contexts * decision.GRADIENT_VALUES_PER_CONTEXT,
        "gain": contexts * decision.GAIN_VALUES_PER_CONTEXT,
        "bias": contexts * decision.BIAS_VALUES_PER_CONTEXT,
    }
    names: tuple[decision.BoundaryKind, ...]
    if family == "affine":
        names = ("site_flow", "gradient")
    elif family == "appearance":
        names = ("gain", "bias")
    else:
        names = ("site_flow", "gradient", "gain", "bias")
    values: list[decision.BoundaryPrimitive] = []
    for position, raw_name in enumerate(names):
        name = cast(decision.BoundaryKind, raw_name)
        denominator = denominators[name]
        numerator = 0
        if position == 0 and warning == "edge":
            numerator = denominator // 4
        elif position == 0 and warning == "below":
            numerator = denominator // 4 - 1
        values.append(decision.BoundaryPrimitive(name, numerator, denominator))
    return tuple(values)


def _certificates(
    family: decision.Family,
    rows: int,
    status: Literal["valid", "incomplete", "invalid"] = "valid",
) -> tuple[decision.CertificatePrimitive, ...]:
    if family == "appearance":
        return ()
    count = rows * decision.CONTEXTS_PER_FAMILY_ROW
    records = [decision.CertificatePrimitive(index, True, True) for index in range(count)]
    if status == "incomplete":
        records[0] = decision.CertificatePrimitive(0, False, False)
    elif status == "invalid":
        records[0] = decision.CertificatePrimitive(0, True, False)
    return tuple(records)


def _evidence(
    *,
    affine: tuple[_Numbers, ...] | None = None,
    appearance: tuple[_Numbers, ...] | None = None,
    combined: tuple[_Numbers, ...] | None = None,
    persistence: tuple[float, ...] = (100.0,) * 8,
    global_translation: tuple[float, ...] = (60.0,) * 8,
    biases: tuple[_BiasNumbers, ...] = (_BiasNumbers(),) * 8,
    boundary_modes: dict[tuple[decision.Family, int], Literal["none", "edge", "below"]] | None = None,
    certificate_modes: dict[tuple[decision.Family, int], Literal["valid", "incomplete", "invalid"]]
    | None = None,
    prerequisites: decision.PrerequisitePrimitives = _ALL_VALID,
) -> decision.RealPrimitiveEvidence:
    by_family = {
        "affine": _uniform(_BAD) if affine is None else affine,
        "appearance": _uniform(_BAD) if appearance is None else appearance,
        "combined": _uniform(_BAD) if combined is None else combined,
    }
    if any(len(values) != 8 for values in by_family.values()):
        raise AssertionError("test fixture family length must be eight")
    if not (len(persistence) == len(global_translation) == len(biases) == 8):
        raise AssertionError("test fixture video length must be eight")
    boundary_modes = {} if boundary_modes is None else boundary_modes
    certificate_modes = {} if certificate_modes is None else certificate_modes
    videos: list[decision.VideoPrimitives] = []
    for video_index, (video_id, fold, rows) in enumerate(decision.VIDEO_SPECS):
        bias_values = biases[video_index]
        family_records: list[decision.FamilyVideoPrimitives] = []
        for family in decision.FAMILIES:
            numbers = by_family[family][video_index]
            family_records.append(
                decision.FamilyVideoPrimitives(
                    family=family,
                    true_full=_error(numbers.full, rows),
                    true_xfit=_error(numbers.xfit, rows),
                    near_xfit=_error(numbers.near, rows),
                    far_xfit=_error(numbers.far, rows),
                    certificates=_certificates(
                        family,
                        rows,
                        certificate_modes.get((family, video_index), "valid"),
                    ),
                    boundaries=_boundaries(
                        family,
                        rows,
                        warning=boundary_modes.get((family, video_index), "none"),
                    ),
                )
            )
        videos.append(
            decision.VideoPrimitives(
                video_id=video_id,
                fold=fold,
                row_count=rows,
                persistence=_error(persistence[video_index], rows),
                global_translation_xfit=_error(global_translation[video_index], rows),
                bias=decision.BiasPrimitives(
                    true_full=_error(bias_values.full, rows),
                    true_xfit=_error(bias_values.xfit, rows),
                    near_xfit=_error(bias_values.near, rows),
                    far_xfit=_error(bias_values.far, rows),
                ),
                families=tuple(family_records),
            )
        )
    return decision.RealPrimitiveEvidence(prerequisites=prerequisites, videos=tuple(videos))


def _family(summary: decision.RealDecisionSummary, family: decision.Family) -> decision.FamilyAggregate:
    return next(item for item in summary.families if item.family == family)


def _fact(
    summary: decision.RealDecisionSummary,
    family: decision.Family,
    video_index: int = 0,
) -> decision.VideoFamilyFacts:
    video_id = decision.VIDEO_SPECS[video_index][0]
    return next(item for item in summary.video_facts if item.video_id == video_id and item.family == family)


def _relative(summary: decision.RealDecisionSummary, gate: decision.RelativeGate) -> decision.RelativeAggregate:
    return next(item for item in summary.relatives if item.gate == gate)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    (
        (
            _evidence(affine=_uniform(_GOOD), combined=_uniform(_GOOD)),
            "smooth_affine_family_supported",
        ),
        (
            _evidence(appearance=_uniform(_GOOD), combined=_uniform(_GOOD)),
            "global_appearance_supported",
        ),
        (
            _evidence(combined=_uniform(_JOINT)),
            "joint_affine_appearance_recovery_supported",
        ),
        (
            _evidence(affine=_uniform(_GOOD), appearance=_uniform(_GOOD), combined=_uniform(_GOOD)),
            "mixed_mechanism_nonidentifiable",
        ),
        (
            _evidence(),
            "tested_affine_appearance_ceiling_failure_supported",
        ),
        (
            _evidence(affine=_uniform(_GOOD)),
            "MM008_mechanism_factorial_inconclusive",
        ),
    ),
)
def test_exact_decision_ladder_reaches_every_valid_branch(
    evidence: decision.RealPrimitiveEvidence,
    expected: decision.DecisionLabel,
) -> None:
    summary = decision.derive_decision(evidence)
    assert summary.decision == expected
    decision.validate_decision(evidence, summary)


def test_exact_real_primitive_census_is_recomputed_from_frozen_video_rows() -> None:
    summary = decision.derive_decision(_evidence())
    assert summary.census == decision.PrimitiveCensus(6_342, 9_513, 2_718, 3_171, 453)
    assert sum(spec[2] for spec in decision.VIDEO_SPECS) == decision.REAL_ROWS


def test_per_video_predicates_include_all_exact_equality_and_strict_edges() -> None:
    edge = _Numbers(full=80.0, xfit=80.0, near=88.0, far=88.0)
    summary = decision.derive_decision(_evidence(affine=(edge, *_uniform(_BAD)[1:])))
    facts = _fact(summary, "affine")
    assert facts.x_support
    assert facts.full_support
    assert facts.residual_near
    assert facts.residual_far
    assert facts.complete_support
    assert facts.any_improvement
    assert facts.full_any_improvement

    below_pair = replace(edge, near=math.nextafter(88.0, -math.inf))
    below_summary = decision.derive_decision(_evidence(affine=(below_pair, *_uniform(_BAD)[1:])))
    assert not _fact(below_summary, "affine").residual_near

    no_strict_improvement = replace(edge, xfit=100.0, near=110.0, far=110.0)
    strict_summary = decision.derive_decision(_evidence(affine=(no_strict_improvement, *_uniform(_BAD)[1:])))
    assert not _fact(strict_summary, "affine").x_support
    assert not _fact(strict_summary, "affine").any_improvement


def test_matched_bias_joint_denominators_and_hit_thresholds_are_exact() -> None:
    edge = _Numbers(full=80.0, xfit=80.0, near=80.0, far=80.0)
    summary = decision.derive_decision(_evidence(appearance=(edge, *_uniform(_BAD)[1:])))
    facts = _fact(summary, "appearance")
    assert facts.x_support
    assert facts.full_support
    assert facts.hit_near
    assert facts.hit_far

    above = replace(edge, near=math.nextafter(80.0, math.inf), far=math.nextafter(80.0, math.inf))
    above_summary = decision.derive_decision(_evidence(appearance=(above, *_uniform(_BAD)[1:])))
    above_facts = _fact(above_summary, "appearance")
    assert not above_facts.hit_near
    assert not above_facts.hit_far


def test_full_only_is_f_and_not_x_and_three_videos_preempt() -> None:
    inversion = _Numbers(full=40.0, xfit=100.0, near=100.0, far=100.0)
    values = (inversion, inversion, inversion, *_uniform(_BAD)[3:])
    summary = decision.derive_decision(_evidence(affine=values))
    aggregate = _family(summary, "affine")
    assert aggregate.full_count == 3
    assert aggregate.x_count == 0
    assert aggregate.full_only_count == 3
    assert "affine:full_xfit_inversion" in summary.inconclusive_preemptions
    assert summary.decision == "MM008_mechanism_factorial_inconclusive"


def test_family_gate_accepts_exactly_six_supports_seven_improvements_and_all_folds() -> None:
    values = (_GOOD, _GOOD, _GOOD, _GOOD, _GOOD, _IMPROVEMENT_ONLY, _GOOD, _BAD)
    summary = decision.derive_decision(_evidence(affine=values))
    aggregate = _family(summary, "affine")
    assert aggregate.x_count == aggregate.complete_count == aggregate.full_count == 6
    assert aggregate.any_improvement_count == aggregate.full_any_improvement_count == 7
    assert aggregate.complete_folds == aggregate.full_folds == decision.FOLDS
    assert aggregate.passes


@pytest.mark.parametrize(
    ("values", "preemption"),
    (
        ((_GOOD,) * 5 + (_BAD,) * 3, "affine:X_support_3_to_5"),
        ((_GOOD,) * 6 + (_BAD,) * 2, "affine:complete_gate_improvement_or_fold"),
        ((_GOOD, _GOOD, _GOOD, _GOOD, _GOOD, _IMPROVEMENT_ONLY, _BAD, _BAD), "affine:X_support_3_to_5"),
    ),
)
def test_support_improvement_and_fold_threshold_failures_preempt(
    values: tuple[_Numbers, ...],
    preemption: str,
) -> None:
    summary = decision.derive_decision(_evidence(affine=values))
    assert not _family(summary, "affine").passes
    assert preemption in summary.inconclusive_preemptions
    assert summary.decision == "MM008_mechanism_factorial_inconclusive"


def test_failed_residual_pairing_after_performance_preempts_even_below_three_c() -> None:
    unpaired = _Numbers(full=40.0, xfit=40.0, near=60.0, far=60.0)
    large_wrong_bias = (_BiasNumbers(near=200.0, far=200.0),) * 8
    summary = decision.derive_decision(
        _evidence(
            affine=_uniform(unpaired),
            persistence=(50.0,) * 8,
            biases=large_wrong_bias,
        )
    )
    aggregate = _family(summary, "affine")
    assert aggregate.x_count == 8
    assert aggregate.complete_count == 0
    assert "affine:residual_pairing_after_performance" in summary.inconclusive_preemptions


def test_relative_gates_use_exact_ten_percent_six_seven_and_fold_edges() -> None:
    affine = _uniform(_Numbers(50.0, 50.0, 90.0, 90.0))
    exact = decision.derive_decision(
        _evidence(affine=affine, global_translation=(55.00000000000001,) * 8)
    )
    gate = _relative(exact, "affine_over_global_translation")
    assert gate.margin_count == gate.improvement_count == 8
    assert gate.margin_folds == decision.FOLDS
    assert gate.passes

    below = decision.derive_decision(_evidence(affine=affine, global_translation=(54.999999999,) * 8))
    assert _relative(below, "affine_over_global_translation").margin_count == 0

    six_margin = (55.00000000000001,) * 6 + (54.0, 54.0)
    missing_fold = decision.derive_decision(_evidence(affine=affine, global_translation=six_margin))
    fold_gate = _relative(missing_fold, "affine_over_global_translation")
    assert fold_gate.margin_count == 6
    assert fold_gate.improvement_count == 8
    assert fold_gate.margin_folds == (0, 1, 2)
    assert not fold_gate.passes


def test_combined_relative_gates_are_separate_and_require_seven_improvements() -> None:
    combined = (_JOINT,) * 7 + (_Numbers(30.0, 0.0, 90.0, 90.0),)
    affine = _uniform(_BAD)
    appearance_values = (_BAD,) * 7 + (_Numbers(100.0, 0.0, 100.0, 100.0),)
    summary = decision.derive_decision(
        _evidence(affine=affine, appearance=appearance_values, combined=combined)
    )
    assert _relative(summary, "combined_over_affine").passes
    appearance_gate = _relative(summary, "combined_over_appearance")
    assert appearance_gate.margin_count == 8
    assert appearance_gate.improvement_count == 7
    assert appearance_gate.passes

    combined = (_JOINT,) * 6 + (_Numbers(30.0, 0.0, 90.0, 90.0),) * 2
    appearance_values = (_BAD,) * 6 + (_Numbers(100.0, 0.0, 100.0, 100.0),) * 2
    failed = decision.derive_decision(
        _evidence(affine=affine, appearance=appearance_values, combined=combined)
    )
    assert _relative(failed, "combined_over_appearance").improvement_count == 6
    assert not _relative(failed, "combined_over_appearance").passes


def test_boundary_fraction_warns_at_exact_quarter_and_range_preempts_at_three_videos() -> None:
    one_edge = decision.derive_decision(
        _evidence(boundary_modes={("affine", 0): "edge", ("affine", 1): "below"})
    )
    assert _fact(one_edge, "affine", 0).boundary_fraction == 0.25
    assert _fact(one_edge, "affine", 0).boundary_warning
    assert _fact(one_edge, "affine", 1).boundary_fraction < 0.25
    assert not _fact(one_edge, "affine", 1).boundary_warning
    assert _family(one_edge, "affine").boundary_warning_count == 1
    assert not _family(one_edge, "affine").range_preemption

    three = decision.derive_decision(
        _evidence(boundary_modes={("affine", 0): "edge", ("affine", 1): "edge", ("affine", 2): "edge"})
    )
    assert _family(three, "affine").boundary_warning_count == 3
    assert _family(three, "affine").range_preemption
    assert "affine:range_warning" in three.inconclusive_preemptions


def test_nonpositive_required_denominator_and_incomplete_certificate_preempt_clean_failure() -> None:
    zero_bias = (_BiasNumbers(xfit=0.0),) + (_BiasNumbers(),) * 7
    denominator = decision.derive_decision(_evidence(biases=zero_bias))
    assert not _family(denominator, "affine").denominators_valid
    assert not _family(denominator, "appearance").clean_fail
    assert "affine:required_denominator" in denominator.inconclusive_preemptions

    certificate = decision.derive_decision(
        _evidence(certificate_modes={("affine", 0): "incomplete"})
    )
    assert not _family(certificate, "affine").global_certificates_valid
    assert not _family(certificate, "affine").clean_fail
    assert "affine:global_certificate" in certificate.inconclusive_preemptions


def test_near_and_far_null_hits_are_never_pooled_and_have_three_six_edges() -> None:
    near_hit = _Numbers(100.0, 100.0, 80.0, 100.0)
    far_hit = _Numbers(100.0, 100.0, 100.0, 80.0)
    three_each = (near_hit,) * 3 + (far_hit,) * 3 + (_BAD,) * 2
    summary = decision.derive_decision(_evidence(affine=three_each))
    aggregate = _family(summary, "affine")
    assert aggregate.hit_near_count == 3
    assert aggregate.hit_far_count == 3
    assert "affine:near_null_hits_3_to_5" in summary.inconclusive_preemptions
    assert "affine:far_null_hits_3_to_5" in summary.inconclusive_preemptions

    six_near = (near_hit,) * 6 + (_BAD,) * 2
    with pytest.raises(decision.InvalidRealAssayError, match="affine:near_null_hits_ge_6"):
        decision.derive_decision(_evidence(affine=six_near))


def test_three_marginal_videos_preempt_only_appearance_clean_failure() -> None:
    marginal = _Numbers(200.0, 90.0, 200.0, 200.0)
    appearance = (marginal,) * 3 + (_Numbers(200.0, 200.0, 200.0, 200.0),) * 5
    persistence = (200.0,) * 8
    bad_for_large_p = _uniform(_Numbers(200.0, 200.0, 200.0, 200.0))
    summary = decision.derive_decision(
        _evidence(
            affine=bad_for_large_p,
            appearance=appearance,
            combined=bad_for_large_p,
            persistence=persistence,
        )
    )
    aggregate = _family(summary, "appearance")
    assert aggregate.marginal_count == 3
    assert not aggregate.clean_fail
    assert "appearance:target_marginal_only" in summary.inconclusive_preemptions
    assert _family(summary, "affine").clean_fail
    assert _family(summary, "combined").clean_fail


def test_failed_pre_real_prerequisite_invalidates_instead_of_minting_a_label() -> None:
    prerequisites = replace(_ALL_VALID, synthetic_controls_valid=False)
    with pytest.raises(decision.InvalidRealAssayError, match="prerequisite:synthetic_controls_valid"):
        decision.derive_decision(_evidence(prerequisites=prerequisites))


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_missing_extra_and_duplicate_video_evidence_fail_closed(mutation: str) -> None:
    evidence = _evidence()
    if mutation == "missing":
        videos = evidence.videos[:-1]
    elif mutation == "extra":
        videos = (*evidence.videos, evidence.videos[-1])
    else:
        videos = (*evidence.videos[:-1], evidence.videos[0])
    with pytest.raises(decision.DecisionV22Error):
        decision.derive_decision(replace(evidence, videos=videos))


def test_family_certificate_boundary_and_error_censuses_fail_closed() -> None:
    evidence = _evidence()
    video = evidence.videos[0]
    affine = video.families[0]

    missing_certificate = replace(affine, certificates=affine.certificates[:-1])
    changed_video = replace(video, families=(missing_certificate, *video.families[1:]))
    with pytest.raises(decision.DecisionV22Error, match="certificate context census"):
        decision.derive_decision(replace(evidence, videos=(changed_video, *evidence.videos[1:])))

    duplicated = replace(
        affine,
        certificates=(affine.certificates[0], affine.certificates[0], *affine.certificates[2:]),
    )
    changed_video = replace(video, families=(duplicated, *video.families[1:]))
    with pytest.raises(decision.DecisionV22Error, match="duplicated"):
        decision.derive_decision(replace(evidence, videos=(changed_video, *evidence.videos[1:])))

    boundary = affine.boundaries[0]
    wrong_boundary = replace(boundary, denominator=boundary.denominator + 1)
    changed_affine = replace(affine, boundaries=(wrong_boundary, *affine.boundaries[1:]))
    changed_video = replace(video, families=(changed_affine, *video.families[1:]))
    with pytest.raises(decision.DecisionV22Error, match="boundary primitives"):
        decision.derive_decision(replace(evidence, videos=(changed_video, *evidence.videos[1:])))

    wrong_count = replace(affine.true_xfit, count=affine.true_xfit.count + 1)
    changed_affine = replace(affine, true_xfit=wrong_count)
    changed_video = replace(video, families=(changed_affine, *video.families[1:]))
    with pytest.raises(decision.DecisionV22Error, match="scored-element census"):
        decision.derive_decision(replace(evidence, videos=(changed_video, *evidence.videos[1:])))


def test_nonfinite_nullable_and_extra_family_primitives_are_rejected() -> None:
    with pytest.raises(decision.DecisionV22Error, match="finite"):
        decision.ErrorPrimitive(float("nan"), 1)

    with pytest.raises(decision.DecisionV22Error, match="immutable typed tuple"):
        decision.RealPrimitiveEvidence(_ALL_VALID, None)  # type: ignore[arg-type]

    evidence = _evidence()
    video = evidence.videos[0]
    changed_video = replace(video, families=(*video.families, video.families[-1]))
    with pytest.raises(decision.DecisionV22Error, match="missing, extra, duplicated"):
        decision.derive_decision(replace(evidence, videos=(changed_video, *evidence.videos[1:])))


def test_finite_primitives_that_overflow_derived_gate_arithmetic_fail_closed() -> None:
    evidence = _evidence()
    video = evidence.videos[0]
    affine = video.families[0]
    huge_xfit = replace(affine.true_xfit, sse=sys.float_info.max)
    huge_near_bias = replace(video.bias.near_xfit, sse=sys.float_info.max)
    changed_affine = replace(affine, true_xfit=huge_xfit)
    changed_video = replace(
        video,
        bias=replace(video.bias, near_xfit=huge_near_bias),
        families=(changed_affine, *video.families[1:]),
    )
    with pytest.raises(decision.DecisionV22Error, match="overflowed finite decision arithmetic"):
        decision.derive_decision(replace(evidence, videos=(changed_video, *evidence.videos[1:])))


def test_deep_recomputation_rejects_forged_missing_extra_and_nested_summaries() -> None:
    evidence = _evidence(affine=_uniform(_GOOD), combined=_uniform(_GOOD))
    summary = decision.derive_decision(evidence)
    decision.validate_decision(evidence, summary)

    forged_label = replace(summary, decision="MM008_mechanism_factorial_inconclusive")
    with pytest.raises(decision.DecisionV22Error, match="deep primitive recomputation"):
        decision.validate_decision(evidence, forged_label)

    forged_count = replace(summary.families[0], x_count=7)
    forged_nested = replace(summary, families=(forged_count, *summary.families[1:]))
    with pytest.raises(decision.DecisionV22Error, match="deep primitive recomputation"):
        decision.validate_decision(evidence, forged_nested)

    for forged_facts in (summary.video_facts[:-1], (*summary.video_facts, summary.video_facts[-1])):
        with pytest.raises(decision.DecisionV22Error, match="deep primitive recomputation"):
            decision.validate_decision(evidence, replace(summary, video_facts=forged_facts))
