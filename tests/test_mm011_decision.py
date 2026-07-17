from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

import pytest

from bench.multimodal_causal_assay import decision, scoring

Mode = Literal["support", "direction", "fail"]
FamilyModes = dict[scoring.Family, tuple[Mode, ...]]
Overrides = dict[tuple[scoring.Family, int], dict[str, float]]
Coordinates = set[tuple[scoring.Family, int]]


def _values(mode: Mode) -> dict[str, float]:
    if mode == "support":
        return {
            "i": 1.0,
            "a": 0.5,
            "q": 0.9,
            "p": 1.0,
            "c": 0.5,
            "h": 0.9,
            "r": 0.9,
            "z": 1.2,
            "d": 0.9,
            "pd": 1.0,
            "u": 1.0,
            "b": 1.0,
            "bd": 1.0,
        }
    if mode == "direction":
        return {
            "i": 1.0,
            "a": 0.9,
            "q": 1.2,
            "p": 1.0,
            "c": 0.9,
            "h": 1.2,
            "r": 1.2,
            "z": 1.2,
            "d": 1.2,
            "pd": 1.0,
            "u": 1.0,
            "b": 1.0,
            "bd": 1.0,
        }
    return {
        "i": 1.0,
        "a": 1.0,
        "q": 1.2,
        "p": 1.0,
        "c": 1.0,
        "h": 1.2,
        "r": 1.2,
        "z": 1.2,
        "d": 1.2,
        "pd": 1.0,
        "u": 1.0,
        "b": 1.0,
        "bd": 1.0,
    }


def _error(mse: float, count: int) -> scoring.ErrorPrimitive:
    return scoring.ErrorPrimitive(mse * count, count)


def _scores(
    family: scoring.Family,
    video_index: int,
    mode: Mode,
    overrides: dict[str, float] | None = None,
) -> scoring.VideoScores:
    video_id, fold, rows = decision.VIDEO_SPECS[video_index]
    values = _values(mode)
    if overrides is not None:
        values.update(overrides)
    count = rows * scoring.ELEMENTS_PER_ROW
    required = {name: _error(values[name], count) for name in ("i", "a", "q", "p", "c", "h", "r", "z", "d", "pd")}
    return scoring.VideoScores(
        video_id=video_id,
        fold=fold,
        family=family,
        row_count=rows,
        **required,
        u=_error(values["u"], count),
        b=_error(values["b"], count),
        bd=_error(values["bd"], count),
    )


def _complete(valid: bool = True) -> decision.CompletenessPrimitives:
    return decision.CompletenessPrimitives(valid, valid, valid, valid)


def _evidence(
    modes: FamilyModes,
    *,
    overrides: Overrides | None = None,
    bounded: Coordinates | None = None,
    incomplete: Coordinates | None = None,
    prerequisites: decision.PrerequisitePrimitives | None = None,
) -> decision.DecisionEvidence:
    changed = {} if overrides is None else overrides
    warned = set() if bounded is None else bounded
    missing = set() if incomplete is None else incomplete
    families: list[decision.FamilyEvidence] = []
    for family in scoring.FAMILIES:
        family_modes = modes[family]
        assert len(family_modes) == 8
        videos = tuple(
            decision.FamilyVideoEvidence(
                scores=_scores(family, index, family_modes[index], changed.get((family, index))),
                bounded_rows=(decision.VIDEO_SPECS[index][2] + 3) // 4 if (family, index) in warned else 0,
                completeness=_complete((family, index) not in missing),
            )
            for index in range(8)
        )
        families.append(decision.FamilyEvidence(family, videos))
    ready = prerequisites or decision.PrerequisitePrimitives(True, True, True, True, True)
    return decision.DecisionEvidence(ready, tuple(families))


def _all(mode: Mode) -> tuple[Mode, ...]:
    return cast(tuple[Mode, ...], (mode,) * 8)


def _single_affine_go() -> FamilyModes:
    # Six joint supports cover all four folds; one additional directional-only
    # video supplies the frozen 7/8 paired directional gate.
    affine = cast(
        tuple[Mode, ...],
        ("support", "support", "support", "support", "support", "direction", "support", "fail"),
    )
    return {"affine": affine, "appearance": _all("fail"), "combined": _all("fail")}


def test_independent_affine_gate_go_without_best_arm_envelope() -> None:
    summary = decision.derive_decision(_evidence(_single_affine_go()))
    assert summary.decision == "causal_affine_family_supported"
    assert summary.go is True
    assert summary.passing_families == ("affine",)
    affine = summary.families[0]
    assert affine.joint_support_count == 6
    assert affine.directional_improvement_count == 7
    assert affine.joint_support_folds == decision.FOLDS


def test_affine_primary_gate_requires_source_free_bias_credit() -> None:
    overrides: Overrides = {(family, index): {"u": 0.4, "b": 0.4} for family in scoring.FAMILIES for index in range(8)}
    summary = decision.derive_decision(_evidence(_single_affine_go(), overrides=overrides))
    affine = summary.families[0]
    assert affine.historical_support_count == 0
    assert affine.future_support_count == 0
    assert affine.directional_improvement_count == 0
    assert affine.failure_diagnosis == "tested_family_identifiability_failure"
    assert summary.decision == "tested_causal_operator_failure_supported"


def test_all_families_clean_no_go() -> None:
    modes = {family: _all("fail") for family in scoring.FAMILIES}
    summary = decision.derive_decision(_evidence(modes))
    assert summary.decision == "tested_causal_operator_failure_supported"
    assert summary.go is False
    assert all(family.clean_fail for family in summary.families)
    assert all(family.failure_diagnosis == "tested_family_identifiability_failure" for family in summary.families)


def test_historical_nonstationarity_is_a_clean_diagnosis() -> None:
    overrides: Overrides = {(family, index): {"a": 0.5, "q": 0.9} for family in scoring.FAMILIES for index in range(6)}
    modes = {family: _all("fail") for family in scoring.FAMILIES}
    summary = decision.derive_decision(_evidence(modes, overrides=overrides))
    assert summary.decision == "tested_causal_operator_failure_supported"
    assert all(family.clean_fail for family in summary.families)
    assert all(
        family.failure_diagnosis == "historically_identifiable_but_complete_future_gate_failed"
        for family in summary.families
    )


def test_mixed_clean_failure_diagnoses_still_support_no_go() -> None:
    overrides: Overrides = {("affine", index): {"a": 0.5, "q": 0.9} for index in range(6)}
    modes = {family: _all("fail") for family in scoring.FAMILIES}
    summary = decision.derive_decision(_evidence(modes, overrides=overrides))
    assert summary.decision == "tested_causal_operator_failure_supported"
    assert tuple(family.failure_diagnosis for family in summary.families) == (
        "historically_identifiable_but_complete_future_gate_failed",
        "tested_family_identifiability_failure",
        "tested_family_identifiability_failure",
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {("affine", index): {"a": 0.5, "q": 0.9} for index in range(4)},
            "affine:historical_support_3_to_5",
        ),
        (
            {("affine", index): {"c": 0.5, "h": 0.9, "r": 0.9, "d": 0.9} for index in range(4)},
            "affine:future_support_3_to_5",
        ),
        (
            {("affine", index): {"c": 0.5, "h": 0.9, "r": 0.9, "d": 0.9} for index in range(6)},
            "affine:future_support_without_joint_support",
        ),
    ],
)
def test_intermediate_or_future_only_support_is_inconclusive(
    overrides: Overrides,
    reason: str,
) -> None:
    modes: FamilyModes = {family: _all("fail") for family in scoring.FAMILIES}
    summary = decision.derive_decision(_evidence(modes, overrides=overrides))
    assert summary.decision == "MM011_inconclusive"
    assert reason in summary.inconclusive_reasons
    assert summary.families[0].failure_diagnosis is None


def test_three_to_five_joint_support_is_inconclusive() -> None:
    affine = cast(tuple[Mode, ...], ("support", "fail", "support", "fail", "support", "fail", "support", "fail"))
    modes: FamilyModes = {"affine": affine, "appearance": _all("fail"), "combined": _all("fail")}
    summary = decision.derive_decision(_evidence(modes))
    assert summary.decision == "MM011_inconclusive"
    assert "affine:joint_support_3_to_5" in summary.inconclusive_reasons


@pytest.mark.parametrize(
    ("metric", "reason_name"),
    [
        ("q", "history_shuffle"),
        ("h", "forecast_shuffle"),
        ("r", "forecast_reverse"),
        ("d", "future_derangement"),
    ],
)
@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (6, "invalid_MM011_real_negative_control"),
        (4, "MM011_inconclusive"),
    ],
)
def test_each_real_null_preempts_at_frozen_counts(
    metric: str,
    reason_name: str,
    count: int,
    expected: str,
) -> None:
    overrides: Overrides = {("affine", index): {metric: 0.7} for index in range(count)}
    summary = decision.derive_decision(_evidence(_single_affine_go(), overrides=overrides))
    assert summary.decision == expected
    if count == 6:
        assert f"affine:{reason_name}_null_ge_6" in summary.invalid_reasons
    else:
        assert f"affine:{reason_name}_null_3_to_5" in summary.inconclusive_reasons


@pytest.mark.parametrize(
    ("null_metric", "bias_metric", "count_field"),
    [
        ("q", "u", "history_shuffle_null_count"),
        ("h", "b", "forecast_shuffle_null_count"),
        ("r", "b", "forecast_reverse_null_count"),
        ("d", "bd", "future_derangement_null_count"),
    ],
)
def test_matched_bias_control_prevents_false_affine_null(
    null_metric: str,
    bias_metric: str,
    count_field: str,
) -> None:
    modes = _single_affine_go()
    overrides: Overrides = {}
    # Keep the directional-only video untouched; use the already-failing eighth
    # video as the sixth null candidate so the family pass remains diagnostic.
    for index in (0, 1, 2, 3, 4, 7):
        for family in scoring.FAMILIES:
            overrides[(family, index)] = {bias_metric: 0.8}
        overrides[("affine", index)][null_metric] = 0.7
    summary = decision.derive_decision(_evidence(modes, overrides=overrides))
    assert summary.decision == "causal_affine_family_supported"
    assert getattr(summary.families[0], count_field) == 0


def test_fold_activity_and_range_each_preempt_go() -> None:
    fold_modes: FamilyModes = {
        "affine": cast(
            tuple[Mode, ...],
            ("support", "support", "support", "support", "support", "support", "direction", "fail"),
        ),
        "appearance": _all("fail"),
        "combined": _all("fail"),
    }
    fold_summary = decision.derive_decision(_evidence(fold_modes))
    assert fold_summary.decision == "MM011_inconclusive"
    assert "affine:directional_or_fold_gate" in fold_summary.inconclusive_reasons

    activity_overrides = {(family, 0): {"i": 0.0} for family in scoring.FAMILIES}
    activity_summary = decision.derive_decision(_evidence(_single_affine_go(), overrides=activity_overrides))
    assert activity_summary.decision == "MM011_inconclusive"
    assert any(reason.endswith("activity_ambiguity") for reason in activity_summary.inconclusive_reasons)

    warned: Coordinates = {("affine", index) for index in range(3)}
    range_summary = decision.derive_decision(_evidence(_single_affine_go(), bounded=warned))
    assert range_summary.decision == "MM011_inconclusive"
    assert "affine:range_warning" in range_summary.inconclusive_reasons


def test_invalid_prerequisite_precedes_real_null_and_incomplete_is_invalid() -> None:
    overrides: Overrides = {("affine", index): {"q": 0.7} for index in range(6)}
    prerequisites = decision.PrerequisitePrimitives(False, True, True, True, True)
    summary = decision.derive_decision(_evidence(_single_affine_go(), overrides=overrides, prerequisites=prerequisites))
    assert summary.decision == "invalid_MM011"
    assert summary.invalid_reasons == ("prerequisite:parent_alignment_valid",)

    incomplete = decision.derive_decision(_evidence(_single_affine_go(), incomplete={("affine", 0)}))
    assert incomplete.decision == "invalid_MM011"
    assert "affine:incomplete_evidence" in incomplete.invalid_reasons


def test_combined_dominance_and_multiple_pass_classification() -> None:
    all_support: FamilyModes = {family: _all("support") for family in scoring.FAMILIES}
    nonidentifiable = decision.derive_decision(_evidence(all_support))
    assert nonidentifiable.decision == "causal_family_nonidentifiable"
    assert nonidentifiable.passing_families == scoring.FAMILIES

    overrides: Overrides = {("combined", index): {"c": 0.4} for index in range(8)}
    joint = decision.derive_decision(_evidence(all_support, overrides=overrides))
    assert joint.combined_dominance.passes is True
    assert joint.decision == "joint_affine_appearance_causal_operator_supported"
    assert joint.go is True


def test_shared_primitives_schema_and_claim_tampering_fail_closed() -> None:
    evidence = _evidence(_single_affine_go())
    appearance = evidence.families[1]
    first = appearance.videos[0]
    altered_scores = replace(
        first.scores,
        p=scoring.ErrorPrimitive(first.scores.p.sse + 1.0, first.scores.p.count),
    )
    altered_appearance = replace(
        appearance,
        videos=(replace(first, scores=altered_scores), *appearance.videos[1:]),
    )
    with pytest.raises(decision.DecisionError, match="shared metric p"):
        decision.derive_decision(
            replace(evidence, families=(evidence.families[0], altered_appearance, evidence.families[2]))
        )

    affine = evidence.families[0]
    affine_first = affine.videos[0]
    altered_bias_scores = replace(
        affine_first.scores,
        u=scoring.ErrorPrimitive(affine_first.scores.u.sse + 1.0, affine_first.scores.u.count),
    )
    altered_affine = replace(
        affine,
        videos=(replace(affine_first, scores=altered_bias_scores), *affine.videos[1:]),
    )
    with pytest.raises(decision.DecisionError, match="shared metric u"):
        decision.derive_decision(
            replace(evidence, families=(altered_affine, evidence.families[1], evidence.families[2]))
        )

    summary = decision.derive_decision(evidence)
    forged = replace(summary, decision="MM011_inconclusive")
    with pytest.raises(decision.DecisionError, match="pure recomputation"):
        decision.validate_decision(evidence, forged)
