"""Parity and authority tests for selective MM-008 v2.2 transpose controls."""

from __future__ import annotations

import inspect
import os
import struct
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from bench.multimodal_mechanism_diagnostics import controls_v22 as controls
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic
from bench.multimodal_mechanism_diagnostics import transpose_v22 as transpose

CONFIG_SHA256 = "d" * 64


@dataclass(frozen=True)
class _LeftLookalike:
    value: int


@dataclass(frozen=True)
class _RightLookalike:
    value: int


@pytest.fixture(scope="module")
def translation_row_zero() -> tuple[
    synthetic.SyntheticCase,
    scoring.RowScore,
    scoring.RowScore,
    transpose.TransposedGridControlRow,
]:
    case = synthetic.generate_case("translation", seed=12_345)
    native = scoring.score_row(case, 0, config_sha256=CONFIG_SHA256)
    legacy = scoring.score_transposed_row(case, 0, config_sha256=CONFIG_SHA256)
    selective = transpose.score_transposed_grid_control_row(
        case,
        0,
        config_sha256=CONFIG_SHA256,
    )
    return case, native, legacy, selective


def test_selective_api_has_no_caller_work_or_formal_authority() -> None:
    signature = inspect.signature(transpose.score_transposed_grid_control_row)
    assert tuple(signature.parameters) == ("original_case", "row", "config_sha256")
    assert signature.parameters["config_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    assert transpose.TRANSPOSE_SCENARIOS == (
        "translation",
        "affine",
        "appearance",
        "combined",
    )
    assert transpose.TRUE_CONTEXTS == ("true_full", "true_p0", "true_p1")
    assert tuple(transpose.TRANSPOSE_GRID_ARMS.items()) == (
        ("translation", ("affine", "combined")),
        ("affine", ("affine", "combined")),
        ("appearance", ("combined",)),
        ("combined", ("combined",)),
    )
    assert sum(len(arms) * 6 * 3 for arms in transpose.TRANSPOSE_GRID_ARMS.values()) == 108
    assert transpose.CLAIM_SCOPE == "exposed-seed-scientific-control-only"
    assert not any(
        name in transpose.__all__
        for name in ("passed", "formal", "nonce", "seed", "path", "serialize")
    )

    out_of_panel = synthetic.generate_case("independent", seed=12_345)
    with pytest.raises(transpose.TransposeV22Error, match="four-scenario"):
        transpose.score_transposed_grid_control_row(
            out_of_panel,
            0,
            config_sha256=CONFIG_SHA256,
        )
    with pytest.raises(transpose.TransposeV22Error, match="integer"):
        transpose.score_transposed_grid_control_row(
            out_of_panel,
            True,
            config_sha256=CONFIG_SHA256,
        )


def test_control_panels_and_deep_comparator_reject_structural_lookalikes() -> None:
    assert not controls._scientific_equal(_LeftLookalike(1), _RightLookalike(1))

    fake_oracles = tuple(SimpleNamespace(spec=spec) for spec in controls.ORACLE_SPECS)
    with pytest.raises(controls.ControlsV22Error, match="exact ordered 22"):
        controls.OraclePanelEvidence(
            CONFIG_SHA256,
            fake_oracles,  # type: ignore[arg-type]
            controls.CONTROL_COVERAGE.sha256,
        )
    fake_mutations = tuple(SimpleNamespace(spec=spec) for spec in controls.MUTATION_SPECS)
    with pytest.raises(controls.ControlsV22Error, match="exact ordered four"):
        controls.MutationPanelEvidence(
            CONFIG_SHA256,
            fake_mutations,  # type: ignore[arg-type]
            controls.CONTROL_COVERAGE.sha256,
        )
    fake_transposes = tuple(SimpleNamespace(spec=spec) for spec in controls.TRANSPOSE_SPECS)
    with pytest.raises(controls.ControlsV22Error, match="exact ordered 108"):
        controls.TransposePanelEvidence(
            CONFIG_SHA256,
            fake_transposes,  # type: ignore[arg-type]
            controls.CONTROL_COVERAGE.sha256,
        )


def test_selective_six_contexts_are_bit_identical_to_legacy_full_row(
    translation_row_zero: tuple[
        synthetic.SyntheticCase,
        scoring.RowScore,
        scoring.RowScore,
        transpose.TransposedGridControlRow,
    ],
) -> None:
    _, native, legacy, selective = translation_row_zero
    assert len(selective.contexts) == 6
    assert tuple((item.arm, item.plan.name) for item in selective.contexts) == (
        ("affine", "true_full"),
        ("affine", "true_p0"),
        ("affine", "true_p1"),
        ("combined", "true_full"),
        ("combined", "true_p0"),
        ("combined", "true_p1"),
    )

    compared = 0
    for spec in controls.TRANSPOSE_SPECS:
        if spec.scenario != "translation" or spec.row != 0:
            continue
        selected = selective.context(spec.arm, spec.context)
        legacy_context = next(
            item for item in legacy.arm(spec.arm).contexts if item.plan.name == spec.context
        )
        legacy_result = cast(exact.GlobalResult, legacy_context.estimate)
        assert exact._results_bit_exact(selected.result, legacy_result)
        assert controls._scientific_equal(selected.plan, legacy_context.plan)
        assert selected.error.count == legacy_context.error.count
        assert struct.pack("<d", selected.error.sse) == struct.pack(
            "<d", legacy_context.error.sse
        )
        assert struct.pack("<d", selected.error.mse) == struct.pack(
            "<d", legacy_context.error.mse
        )

        legacy_comparison = controls.compare_transpose_context(spec, native, legacy)
        selective_comparison = controls.compare_selective_transpose_context(
            spec,
            native,
            selective,
        )
        assert controls._scientific_equal(legacy_comparison, selective_comparison)
        compared += 1
    assert compared == 6


def test_selective_replay_rejects_nested_objective_cache_tamper_without_recompute(
    translation_row_zero: tuple[
        synthetic.SyntheticCase,
        scoring.RowScore,
        scoring.RowScore,
        transpose.TransposedGridControlRow,
    ],
) -> None:
    case, _, _, canonical = translation_row_zero
    contexts = list(canonical.contexts)
    original = contexts[-1]
    forged_cache = replace(original.result.objective_cache, content_sha256="e" * 64)
    forged_result = replace(original.result, objective_cache=forged_cache)
    contexts[-1] = replace(original, result=forged_result)
    forged = replace(canonical, contexts=tuple(contexts))

    with patch.object(
        transpose,
        "score_transposed_grid_control_row",
        return_value=canonical,
    ) as rebuild:
        with pytest.raises(transpose.TransposeV22Error, match="complete replay"):
            transpose.validate_transposed_grid_control_row(forged, case)
    rebuild.assert_called_once_with(case, 0, config_sha256=CONFIG_SHA256)


def test_transpose_comparison_rejects_context_mismatch_and_alias_fracture(
    translation_row_zero: tuple[
        synthetic.SyntheticCase,
        scoring.RowScore,
        scoring.RowScore,
        transpose.TransposedGridControlRow,
    ],
) -> None:
    _, native, _, selective = translation_row_zero
    specs = tuple(
        spec
        for spec in controls.TRANSPOSE_SPECS
        if spec.scenario == "translation" and spec.row == 0
    )
    comparisons = tuple(
        controls.compare_selective_transpose_context(spec, native, selective)
        for spec in specs
    )

    original = comparisons[0]
    wrong_native_plan = replace(original.native_plan, target_row=1)
    wrong_transposed_plan = replace(original.transposed_plan, target_row=1)
    wrong_diagnostics = controls._transpose_diagnostics(
        original.spec,
        wrong_native_plan,
        wrong_transposed_plan,
        original.native,
        original.transposed,
        original.native_error,
        original.transposed_error,
    )
    assert not wrong_diagnostics.context_identity_exact
    with pytest.raises(controls.ControlsV22Error, match="context identity or mask"):
        replace(
            original,
            native_plan=wrong_native_plan,
            transposed_plan=wrong_transposed_plan,
            diagnostics=wrong_diagnostics,
        )

    combined_full_index = next(
        index
        for index, entry in enumerate(comparisons)
        if entry.spec.arm == "combined" and entry.spec.context == "true_full"
    )
    combined_full = comparisons[combined_full_index]
    copied_source_grid = replace(combined_full.transposed.source_grid)
    assert copied_source_grid is not combined_full.transposed.source_grid
    assert controls._scientific_equal(
        copied_source_grid,
        combined_full.transposed.source_grid,
    )
    forged_result = replace(
        combined_full.transposed,
        source_grid=copied_source_grid,
    )
    forged_entry = replace(combined_full, transposed=forged_result)
    fractured = list(comparisons)
    fractured[combined_full_index] = forged_entry
    with pytest.raises(controls.ControlsV22Error, match="fractures.*aliases"):
        controls._validate_transpose_source_aliases(tuple(fractured))


@pytest.mark.skipif(
    os.environ.get("PROSPECT_MM008_EXHAUSTIVE_TRANSPOSE_PARITY") != "1",
    reason="16-minute exposed-seed exhaustive parity is opt-in",
)
def test_all_108_selective_results_match_the_legacy_full_row_path() -> None:
    compared = 0
    legacy_contexts = 0
    selective_contexts = 0
    for scenario in transpose.TRANSPOSE_SCENARIOS:
        case = synthetic.generate_case(scenario, seed=12_345)
        for row in range(6):
            legacy = scoring.score_transposed_row(
                case,
                row,
                config_sha256=CONFIG_SHA256,
            )
            selective = transpose.score_transposed_grid_control_row(
                case,
                row,
                config_sha256=CONFIG_SHA256,
            )
            legacy_contexts += sum(
                len(legacy.arm(arm).contexts)
                for arm in scoring.SCENARIO_GRID_ARMS[scenario]
            )
            selective_contexts += len(selective.contexts)
            for spec in controls.TRANSPOSE_SPECS:
                if spec.scenario != scenario or spec.row != row:
                    continue
                selected = selective.context(spec.arm, spec.context)
                old_context = next(
                    item
                    for item in legacy.arm(spec.arm).contexts
                    if item.plan.name == spec.context
                )
                old_result = cast(exact.GlobalResult, old_context.estimate)
                assert exact._results_bit_exact(selected.result, old_result)
                assert controls._scientific_equal(selected.plan, old_context.plan)
                assert controls._scientific_equal(selected.error, old_context.error)
                compared += 1
    assert (compared, legacy_contexts, selective_contexts) == (108, 336, 108)
