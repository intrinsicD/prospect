"""Focused exposed-seed tests for pure MM-008 v2.2 scientific controls."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import controls_v22 as controls
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import oracle_v22 as oracle
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

EXPOSED_SEED = 12_345
CONFIG_SHA256 = "7" * 64


@dataclass(frozen=True, slots=True)
class _OraclePanelRun:
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase]
    scores: dict[synthetic.Scenario, scoring.RowScore]
    evidence: controls.OraclePanelEvidence


@dataclass(frozen=True, slots=True)
class _MutationRun:
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase]
    evidence: controls.MutationPanelEvidence


@dataclass(frozen=True, slots=True)
class _TransposeRun:
    native: scoring.RowScore
    transposed: scoring.RowScore
    entries: tuple[controls.TransposeComparisonEvidence, ...]


@pytest.fixture(scope="module")
def oracle_panel_run() -> _OraclePanelRun:
    scenarios: tuple[synthetic.Scenario, ...] = (
        "translation",
        "affine",
        "combined",
        "coupled_boundary",
        "independent",
        "constant_target",
    )
    cases = {scenario: synthetic.generate_case(scenario, seed=EXPOSED_SEED) for scenario in scenarios}
    scores = {scenario: scoring.score_row(case, 0, config_sha256=CONFIG_SHA256) for scenario, case in cases.items()}
    evidence = controls.assemble_oracle_panel(cases, scores, config_sha256=CONFIG_SHA256)
    return _OraclePanelRun(cases, scores, evidence)


@pytest.fixture(scope="module")
def mutation_run() -> _MutationRun:
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase] = {
        "affine": synthetic.generate_case("affine", seed=EXPOSED_SEED),
        "combined": synthetic.generate_case("combined", seed=EXPOSED_SEED),
    }
    evidence = controls.run_mutation_panel(cases, config_sha256=CONFIG_SHA256)
    return _MutationRun(cases, evidence)


@pytest.fixture(scope="module")
def transpose_run() -> _TransposeRun:
    case = synthetic.generate_case("combined", seed=EXPOSED_SEED)
    native = scoring.score_row(case, 0, config_sha256=CONFIG_SHA256)
    transposed = scoring.score_transposed_row(case, 0, config_sha256=CONFIG_SHA256)
    specs = tuple(
        spec
        for spec in controls.TRANSPOSE_SPECS
        if spec.scenario == "combined" and spec.row == 0 and spec.arm == "combined"
    )
    entries = tuple(controls.compare_transpose_context(spec, native, transposed) for spec in specs)
    return _TransposeRun(native, transposed, entries)


def test_module_is_no_io_and_declares_exact_unique_coverage() -> None:
    module_path = Path(controls.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_import_roots = {
        "argparse",
        "importlib",
        "io",
        "json",
        "os",
        "pathlib",
        "pickle",
        "random",
        "secrets",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"open", "print", "input", "exec", "eval"}
    assert imported_roots.isdisjoint(forbidden_import_roots)

    coverage = controls.CONTROL_COVERAGE
    assert coverage.counts == (22, 4, 108, 134)
    assert coverage.sha256 == ("4b64856d2fd75305768aa60edbe9ae799d639ce8c96eda99e5317da7111d9fc8")
    assert coverage.all_keys == controls.EXPECTED_COVERAGE_KEYS
    assert len(set(coverage.all_keys)) == 134
    assert tuple(spec.key for spec in controls.ORACLE_SPECS) == coverage.oracle_keys
    assert tuple(spec.key for spec in controls.MUTATION_SPECS) == coverage.mutation_keys
    assert tuple(spec.key for spec in controls.TRANSPOSE_SPECS) == coverage.transpose_keys


def test_named_spec_order_is_exact_and_tamper_closed() -> None:
    assert tuple(spec.key for spec in controls.ORACLE_SPECS) == (
        "oracle/translation/row-0/affine/true_full",
        "oracle/translation/row-0/affine/true_p0",
        "oracle/translation/row-0/affine/true_p1",
        "oracle/affine/row-0/affine/true_full",
        "oracle/affine/row-0/affine/true_p0",
        "oracle/affine/row-0/affine/true_p1",
        "oracle/combined/row-0/combined/true_full",
        "oracle/combined/row-0/combined/true_p0",
        "oracle/combined/row-0/combined/true_p1",
        "oracle/coupled_boundary/row-0/affine/true_full",
        "oracle/coupled_boundary/row-0/affine/true_p0",
        "oracle/coupled_boundary/row-0/affine/true_p1",
        "oracle/independent/row-0/combined/true_full",
        "oracle/independent/row-0/combined/true_p0",
        "oracle/independent/row-0/combined/true_p1",
        "oracle/constant_target/row-0/combined/true_full",
        "oracle/constant_target/row-0/combined/true_p0",
        "oracle/constant_target/row-0/combined/true_p1",
        "oracle/translation/row-0/affine/near_p0",
        "oracle/translation/row-0/affine/far_p1",
        "oracle/translation/row-0/combined/near_p0",
        "oracle/translation/row-0/combined/far_p1",
    )
    assert tuple(spec.key for spec in controls.MUTATION_SPECS) == (
        "mutation/affine/row-0/affine/true_p0",
        "mutation/affine/row-0/affine/true_p1",
        "mutation/combined/row-0/combined/true_p0",
        "mutation/combined/row-0/combined/true_p1",
    )
    assert [spec.key for spec in controls.TRANSPOSE_SPECS[:4]] == [
        "transpose/translation/row-0/affine/true_full",
        "transpose/translation/row-0/affine/true_p0",
        "transpose/translation/row-0/affine/true_p1",
        "transpose/translation/row-1/affine/true_full",
    ]
    assert controls.TRANSPOSE_SPECS[-1].key == ("transpose/combined/row-5/combined/true_p1")
    with pytest.raises(controls.ControlsV22Error):
        replace(controls.ORACLE_SPECS[0], key="oracle/not-canonical")
    with pytest.raises(controls.ControlsV22Error):
        replace(controls.ORACLE_SPECS[12], requires_truth=True)
    with pytest.raises(controls.ControlsV22Error):
        replace(controls.MUTATION_SPECS[0], output_parity=1)
    with pytest.raises(controls.ControlsV22Error):
        replace(controls.TRANSPOSE_SPECS[0], row=6)


def test_held_target_mutator_changes_only_all_channels_of_output_sites() -> None:
    index = np.arange(3 * 64 * 64, dtype=np.float64).reshape(3, 64, 64)
    target = np.ascontiguousarray(index)
    changed = controls.mutate_held_target(target, 0)
    difference = changed - target
    central = geometry.GEOMETRY.coords.astype(np.intp)
    central_difference = difference[:, central[:, 0], central[:, 1]]
    assert np.array_equal(
        central_difference[:, geometry.PARITY_MASKS[0]],
        np.full((3, geometry.SITE_COUNT // 2), controls.MUTATION_DELTA),
    )
    assert np.array_equal(
        central_difference[:, geometry.PARITY_MASKS[1]],
        np.zeros((3, geometry.SITE_COUNT // 2)),
    )
    outside = np.ones((64, 64), dtype=np.bool_)
    outside[8:56, 8:56] = False
    assert np.array_equal(difference[:, outside], np.zeros((3, int(outside.sum()))))
    assert not changed.flags.writeable
    with pytest.raises(ValueError):
        changed.setflags(write=True)


def test_all_four_mutations_leave_every_fit_artifact_bit_exact(
    mutation_run: _MutationRun,
) -> None:
    panel = mutation_run.evidence
    assert tuple(entry.spec for entry in panel.entries) == controls.MUTATION_SPECS
    for entry in panel.entries:
        diagnostic = entry.diagnostics
        assert all(controls.fields_as_values(diagnostic))
        assert entry.mutated_error.mse == controls.MUTATION_DELTA**2
        assert entry.original_error.mse < 1e-24
        assert entry.original_fit_target.tobytes(order="C") == entry.mutated_fit_target.tobytes(order="C")
        assert entry.original_result.prediction.tobytes(order="C") == entry.mutated_result.prediction.tobytes(order="C")
        assert (
            entry.original_result.objective_cache.content_sha256 == entry.mutated_result.objective_cache.content_sha256
        )
        assert entry.original_bias.scope_sha256 == entry.mutated_bias.scope_sha256

    changed_diagnostic = replace(panel.entries[0].diagnostics, replay_bit_exact=False)
    with pytest.raises(controls.ControlsV22Error):
        replace(panel.entries[0], diagnostics=changed_diagnostic)


def test_mutation_panel_is_deeply_replay_validatable(mutation_run: _MutationRun) -> None:
    controls.validate_mutation_panel(mutation_run.evidence, mutation_run.cases)


def test_real_scalar_oracle_matches_every_production_field_and_truth(
    oracle_panel_run: _OraclePanelRun,
) -> None:
    panel = oracle_panel_run.evidence
    assert tuple(entry.spec for entry in panel.entries) == controls.ORACLE_SPECS
    assert len(panel.entries) == 22
    assert all(entry.diagnostics.all_fields_bit_exact for entry in panel.entries)
    assert all(
        entry.diagnostics.truth_selected_bit_exact is True and entry.diagnostics.nonflow_separated is True
        for entry in panel.entries[:12]
    )
    assert all(
        entry.diagnostics.truth_state_index is None
        and entry.diagnostics.truth_admissible_rank is None
        and entry.diagnostics.truth_selected_bit_exact is None
        and entry.diagnostics.nonflow_separated is None
        for entry in panel.entries[12:]
    )

    evidence = panel.entries[0]
    diagnostic = evidence.diagnostics
    assert evidence.spec.key == "oracle/translation/row-0/affine/true_full"
    assert diagnostic.all_fields_bit_exact
    assert diagnostic.truth_state_index == geometry.state_index(
        synthetic.TRUTHS["translation"].theta  # type: ignore[union-attr]
    )
    assert diagnostic.truth_selected_bit_exact is True
    assert diagnostic.nonflow_separated is True
    assert diagnostic.production_nonflow_gap > 1e-12
    assert diagnostic.production_nonflow_gap == diagnostic.independent_nonflow_gap
    assert evidence.production.source_grid.content_sha256 == evidence.independent.source_grid.content_sha256
    assert evidence.production.objective_cache.objectives.tobytes(
        order="C"
    ) == evidence.independent.objective_cache.objectives.tobytes(order="C")
    assert evidence.production.prediction.tobytes(order="C") == evidence.independent.prediction.tobytes(order="C")
    changed = replace(diagnostic, all_fields_bit_exact=False)
    with pytest.raises(controls.ControlsV22Error):
        replace(evidence, diagnostics=changed)


def test_scalar_oracle_completes_when_production_batch_builder_and_evaluator_raise(
    oracle_panel_run: _OraclePanelRun,
) -> None:
    case = oracle_panel_run.cases["translation"]
    bundle = synthetic.row_targets(case, 0)
    fit_target = fitting.target_values(bundle.true_target, geometry.FULL_MASK)
    production_only = AssertionError("production exact-grid path must remain unused")
    with (
        patch.object(exact, "_batch_record", side_effect=production_only),
        patch.object(exact, "_evaluate", side_effect=production_only),
    ):
        independent = oracle.fit_scalar_oracle(
            bundle.source,
            fit_target,
            geometry.FULL_MASK,
            geometry.FULL_MASK,
            "affine",
            context_key="oracle-independence/translation/row-0/affine/true_full",
            config_sha256=CONFIG_SHA256,
        )
    assert independent.maximum_candidate_tensor_size == 1
    assert independent.selected.state_index == geometry.state_index(
        synthetic.TRUTHS["translation"].theta  # type: ignore[union-attr]
    )


def test_hardest_transpose_kat_preserves_only_the_normative_discrete_rule(
    transpose_run: _TransposeRun,
) -> None:
    assert tuple(entry.spec.context for entry in transpose_run.entries) == (
        "true_full",
        "true_p0",
        "true_p1",
    )
    for entry in transpose_run.entries:
        diagnostic = entry.diagnostics
        assert diagnostic.passes
        assert diagnostic.unique_minima
        assert diagnostic.theta_permutation_exact
        assert diagnostic.context_identity_exact
        assert diagnostic.fit_mask_transpose_exact
        assert diagnostic.output_mask_transpose_exact

        # These continuous/reduction-order diagnostics expose the actual mechanism:
        # they drift at machine precision and are deliberately not PASS predicates.
        assert not diagnostic.prediction_transpose_bit_exact
        assert diagnostic.prediction_max_abs_delta > 0.0
        assert not diagnostic.fit_prediction_transpose_bit_exact
        assert diagnostic.fit_prediction_max_abs_delta > 0.0
        assert not diagnostic.objective_bit_exact
        assert diagnostic.objective_abs_delta > 0.0
        assert diagnostic.gains_bit_exact is False
        assert diagnostic.gains_max_abs_delta is not None
        assert diagnostic.gains_max_abs_delta > 0.0
        assert diagnostic.biases_bit_exact is False
        assert diagnostic.biases_max_abs_delta is not None
        assert diagnostic.biases_max_abs_delta > 0.0
        assert not diagnostic.error_bit_exact
        assert diagnostic.error_max_abs_delta > 0.0

        assert not replace(diagnostic, unique_minima=False).passes
        assert not replace(diagnostic, theta_permutation_exact=False).passes
        assert replace(
            diagnostic,
            prediction_transpose_bit_exact=not diagnostic.prediction_transpose_bit_exact,
            objective_bit_exact=not diagnostic.objective_bit_exact,
            error_bit_exact=not diagnostic.error_bit_exact,
        ).passes

    assert [entry.diagnostics.retained_ids_transpose_exact for entry in transpose_run.entries] == [
        False,
        True,
        False,
    ]
    assert transpose_run.entries[0].diagnostics.second_gap_bit_exact
    assert transpose_run.entries[1].diagnostics.second_gap_bit_exact
    assert not transpose_run.entries[2].diagnostics.second_gap_bit_exact
    assert not transpose_run.entries[2].diagnostics.nonflow_gap_bit_exact


def test_transpose_evidence_rejects_forged_pass_diagnostics(
    transpose_run: _TransposeRun,
) -> None:
    entry = transpose_run.entries[0]
    changed = replace(entry.diagnostics, unique_minima=False)
    with pytest.raises(controls.ControlsV22Error):
        replace(entry, diagnostics=changed)
