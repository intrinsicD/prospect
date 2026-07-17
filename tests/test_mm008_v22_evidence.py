"""Cheap structural tests for the ephemeral MM-008 v2.2 science aggregate.

These tests use explicit object shells to exercise graph authority and tamper
rejection without pretending to be a completed numerical run.  The full exposed
collector is intentionally left to the dedicated development experiment.
"""

from __future__ import annotations

import inspect
import pickle
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Any, TypeVar, cast
from unittest.mock import patch

import pytest

from bench.multimodal_mechanism_diagnostics import bank_v22 as bank
from bench.multimodal_mechanism_diagnostics import controls_v22 as controls
from bench.multimodal_mechanism_diagnostics import evidence_v22 as evidence
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import integrity_v22 as integrity
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

CONFIG_SHA256 = "4" * 64
_T = TypeVar("_T")


def _shell(cls: type[_T], **values: object) -> _T:
    result = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _clone_shell(value: _T, **changes: object) -> _T:
    values = {
        field.name: getattr(value, field.name)
        for field in fields(cast(Any, value))
    }
    values.update(changes)
    return _shell(type(value), **values)


def _result_shell() -> exact.GlobalResult:
    batch_records = (
        *(SimpleNamespace(indices=tuple(range(128))) for _ in range(21)),
        SimpleNamespace(indices=tuple(range(121))),
    )
    source_grid = SimpleNamespace(scope_sha256="a" * 64, batch_records=batch_records)
    objective_cache = SimpleNamespace(
        arm="combined",
        scope_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    selected = SimpleNamespace(state_index=0, admissible_rank=0)
    certificate = SimpleNamespace(
        exact_tie_multiplicity=1,
        second_best_objective_gap=1.0,
        second_best_nonflow_gap=2.0,
        scalar_replay_bit_exact=True,
    )
    return _shell(
        exact.GlobalResult,
        context_key="constructed-only",
        arm="combined",
        source_grid=source_grid,
        objective_cache=objective_cache,
        selected=selected,
        prediction=SimpleNamespace(name="constructed-prediction"),
        prediction_sha256="d" * 64,
        certificate=certificate,
    )


def _context_shell(
    arm: scoring.Arm,
    context: scoring.ContextName,
    estimate: object,
) -> scoring.ArmContextScore:
    return _shell(
        scoring.ArmContextScore,
        plan=SimpleNamespace(name=context),
        arm=arm,
        estimate=estimate,
        error=SimpleNamespace(name="constructed-error"),
        endpoints=(),
        carries=None,
    )


def _arm_shell(arm: scoring.Arm, grid_result: exact.GlobalResult) -> scoring.ArmScores:
    estimate: object = grid_result if arm in scoring.GRID_ARMS else SimpleNamespace(arm=arm)
    contexts = tuple(_context_shell(arm, context, estimate) for context in scoring.CONTEXT_ORDER)
    placeholder = SimpleNamespace(name="constructed-aggregate")
    return _shell(
        scoring.ArmScores,
        arm=arm,
        contexts=contexts,
        persistence=placeholder,
        true_full=placeholder,
        true_xfit=placeholder,
        near_xfit=placeholder,
        far_xfit=placeholder,
        endpoints=(),
        endpoints_pass=False,
        true_prediction_bit_exact=False,
        predicates=placeholder,
        carries=None,
    )


def _row_shell(
    scenario: synthetic.Scenario,
    row: int,
    grid_result: exact.GlobalResult,
) -> scoring.RowScore:
    arms = tuple(_arm_shell(arm, grid_result) for arm in scoring.scenario_arms(scenario))
    return _shell(
        scoring.RowScore,
        scenario=scenario,
        seed=evidence.DEV_SEED,
        row=row,
        config_sha256=CONFIG_SHA256,
        input_orientation="native",
        persistence_estimate=SimpleNamespace(name="constructed-persistence"),
        persistence=SimpleNamespace(name="constructed-persistence-error"),
        bias=SimpleNamespace(contexts=tuple(range(7))),
        arms=arms,
        grid_streams=(),
        dominance=(),
        expectations=(),
    )


def _bank_shell(grid_result: exact.GlobalResult) -> bank.OrdinaryBankEvidence:
    rows = tuple(
        _row_shell(scenario, row, grid_result)
        for scenario in synthetic.SCENARIOS
        for row in range(6)
    )
    return _shell(
        bank.OrdinaryBankEvidence,
        protocol_sha256=evidence.PROTOCOL_SHA256,
        schema_version=bank.SCHEMA_VERSION,
        config_sha256=CONFIG_SHA256,
        seed_authority=bank.exposed_seed_authority(),
        rows=rows,
        primitive_ledger=bank.EXPECTED_PRIMITIVE_LEDGER,
        scenario_expectations=(),
        boundary_occupancy=tuple(SimpleNamespace(index=index) for index in range(5)),
    )


def _row_by_key(
    native_bank: bank.OrdinaryBankEvidence,
) -> dict[tuple[synthetic.Scenario, int], scoring.RowScore]:
    return {(row.scenario, row.row): row for row in native_bank.rows}


def _grid_result(
    native_bank: bank.OrdinaryBankEvidence,
    scenario: synthetic.Scenario,
    row: int,
    arm: controls.GridArm,
    context: scoring.ContextName,
) -> exact.GlobalResult:
    arm_scores = _row_by_key(native_bank)[(scenario, row)].arm(arm)
    context_score = next(item for item in arm_scores.contexts if item.plan.name == context)
    return cast(exact.GlobalResult, context_score.estimate)


def _controls_shell(native_bank: bank.OrdinaryBankEvidence) -> controls.ScientificControlsEvidence:
    oracle_entries = tuple(
        SimpleNamespace(
            spec=spec,
            production=_grid_result(
                native_bank,
                spec.scenario,
                spec.row,
                spec.arm,
                spec.context,
            ),
        )
        for spec in controls.ORACLE_SPECS
    )
    transpose_entries = tuple(
        SimpleNamespace(
            spec=spec,
            native=_grid_result(
                native_bank,
                spec.scenario,
                spec.row,
                spec.arm,
                spec.context,
            ),
        )
        for spec in controls.TRANSPOSE_SPECS
    )
    oracle_panel = _shell(
        controls.OraclePanelEvidence,
        config_sha256=CONFIG_SHA256,
        entries=oracle_entries,
        coverage_sha256=controls.CONTROL_COVERAGE.sha256,
    )
    mutation_panel = _shell(
        controls.MutationPanelEvidence,
        config_sha256=CONFIG_SHA256,
        entries=tuple(SimpleNamespace(index=index) for index in range(4)),
        coverage_sha256=controls.CONTROL_COVERAGE.sha256,
    )
    transpose_panel = _shell(
        controls.TransposePanelEvidence,
        config_sha256=CONFIG_SHA256,
        entries=transpose_entries,
        coverage_sha256=controls.CONTROL_COVERAGE.sha256,
    )
    return _shell(
        controls.ScientificControlsEvidence,
        oracle_panel=oracle_panel,
        mutation_panel=mutation_panel,
        transpose_panel=transpose_panel,
        coverage=controls.CONTROL_COVERAGE,
    )


def _integrity_shell(bound: exact.GlobalResult) -> integrity.IntegrityEvidence:
    certificate = bound.certificate
    canonical = SimpleNamespace(
        selected_state_index=bound.selected.state_index,
        selected_admissible_rank=bound.selected.admissible_rank,
        exact_tie_multiplicity=certificate.exact_tie_multiplicity,
        second_best_objective_gap=certificate.second_best_objective_gap,
        second_best_nonflow_gap=certificate.second_best_nonflow_gap,
        objective_content_sha256=bound.objective_cache.content_sha256,
    )
    delivery = _shell(
        integrity.DeliveryIntegrityEvidence,
        orders=(canonical, canonical, canonical),
        rejections=tuple(SimpleNamespace(index=index) for index in range(4)),
        batch_sizes=tuple(len(record.indices) for record in bound.source_grid.batch_records),
        scalar_replay_bit_exact=True,
    )
    forgery = _shell(
        integrity.ForgeryMatrixEvidence,
        witnesses=tuple(SimpleNamespace(index=index) for index in range(28)),
    )
    cache = _shell(
        integrity.CacheIdentityEvidence,
        checks=tuple(SimpleNamespace(index=index) for index in range(11)),
        identities=(
            ("source/base", bound.source_grid.scope_sha256),
            ("objective/base-combined", bound.objective_cache.scope_sha256),
        ),
    )
    return _shell(
        integrity.IntegrityEvidence,
        delivery=delivery,
        forgery_matrix=forgery,
        cache_identities=cache,
    )


def _aggregate(
    native_bank: bank.OrdinaryBankEvidence,
    scientific_controls: controls.ScientificControlsEvidence,
) -> evidence.ExposedSeedScienceEvidence:
    bound = _grid_result(native_bank, "combined", 0, "combined", "true_full")
    return evidence.ExposedSeedScienceEvidence(
        protocol_sha256=evidence.PROTOCOL_SHA256,
        schema_version=evidence.SCHEMA_VERSION,
        config_sha256=CONFIG_SHA256,
        dev_seed=evidence.DEV_SEED,
        claim_scope=evidence.CLAIM_SCOPE,
        bank=native_bank,
        controls=scientific_controls,
        integrity_binding=evidence.INTEGRITY_BINDING,
        integrity=_integrity_shell(bound),
        census=evidence.EXPECTED_CENSUS,
    )


def _fixture() -> evidence.ExposedSeedScienceEvidence:
    native_bank = _bank_shell(_result_shell())
    return _aggregate(native_bank, _controls_shell(native_bank))


def test_exact_census_binding_scope_and_public_api_have_no_formal_authority() -> None:
    assert tuple(getattr(evidence.EXPECTED_CENSUS, field.name) for field in fields(evidence.EvidenceCensus)) == (
        8,
        48,
        34,
        204,
        1_428,
        630,
        294,
        336,
        336,
        48,
        5,
        22,
        12,
        10,
        608,
        4,
        108,
        134,
        1,
        3,
        4,
        28,
        11,
        "declared-22-context-subset-not-full-independent-coverage",
    )
    assert evidence.INTEGRITY_BINDING == evidence.IntegrityBinding(
        "combined", 0, "combined", "true_full", 0, 0, "nextafter-positive"
    )
    assert tuple(field.name for field in fields(evidence.ExposedSeedScienceEvidence)) == (
        "protocol_sha256",
        "schema_version",
        "config_sha256",
        "dev_seed",
        "claim_scope",
        "bank",
        "controls",
        "integrity_binding",
        "integrity",
        "census",
    )
    assert tuple(inspect.signature(evidence.collect_exposed_seed_science).parameters) == (
        "config_sha256",
    )
    assert (
        inspect.signature(evidence.collect_exposed_seed_science).parameters["config_sha256"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    with pytest.raises(evidence.EvidenceV22Error):
        replace(evidence.EXPECTED_CENSUS, oracle_contexts=630)
    with pytest.raises(evidence.EvidenceV22Error):
        replace(evidence.INTEGRITY_BINDING, source_flat_index=1)


def test_aggregate_is_ephemeral_and_enforces_all_12345_authority() -> None:
    aggregate = _fixture()
    assert not hasattr(aggregate, "passed")
    with pytest.raises(TypeError, match="ephemeral and non-serializable"):
        pickle.dumps(aggregate)

    wrong_bank = _clone_shell(
        aggregate.bank,
        seed_authority=bank.uniform_seed_authority(evidence.DEV_SEED + 1),
    )
    with pytest.raises(evidence.EvidenceV22Error, match="all-12345"):
        _aggregate(wrong_bank, aggregate.controls)


@pytest.mark.parametrize("panel_name", ("oracle", "transpose"))
def test_bank_result_aliases_are_identity_requirements(panel_name: str) -> None:
    aggregate = _fixture()
    scientific_controls = aggregate.controls
    if panel_name == "oracle":
        oracle_entries = scientific_controls.oracle_panel.entries
        copied = _clone_shell(oracle_entries[0].production)
        forged_entry = SimpleNamespace(spec=oracle_entries[0].spec, production=copied)
        forged_oracle_panel = _clone_shell(
            scientific_controls.oracle_panel,
            entries=(forged_entry, *oracle_entries[1:]),
        )
        forged_controls = _clone_shell(
            scientific_controls,
            oracle_panel=forged_oracle_panel,
        )
    else:
        transpose_entries = scientific_controls.transpose_panel.entries
        copied = _clone_shell(transpose_entries[0].native)
        forged_entry = SimpleNamespace(spec=transpose_entries[0].spec, native=copied)
        forged_transpose_panel = _clone_shell(
            scientific_controls.transpose_panel,
            entries=(forged_entry, *transpose_entries[1:]),
        )
        forged_controls = _clone_shell(
            scientific_controls,
            transpose_panel=forged_transpose_panel,
        )
    with pytest.raises(evidence.EvidenceV22Error, match="bank-owned context object"):
        _aggregate(aggregate.bank, forged_controls)


def test_complete_replay_rejects_nested_production_only_objective_cache_tamper() -> None:
    canonical = _fixture()
    rows = list(canonical.bank.rows)
    row_index = synthetic.SCENARIOS.index("stationary") * 6 + 5
    original_row = rows[row_index]
    arms = list(original_row.arms)
    arm_index = next(index for index, item in enumerate(arms) if item.arm == "combined")
    original_arm = arms[arm_index]
    contexts = list(original_arm.contexts)
    context_index = scoring.CONTEXT_ORDER.index("far_p1")
    original_context = contexts[context_index]
    original_result = cast(exact.GlobalResult, original_context.estimate)
    forged_cache = SimpleNamespace(
        arm=original_result.objective_cache.arm,
        scope_sha256=original_result.objective_cache.scope_sha256,
        content_sha256="e" * 64,
    )
    forged_result = _clone_shell(original_result, objective_cache=forged_cache)
    contexts[context_index] = _clone_shell(original_context, estimate=forged_result)
    arms[arm_index] = _clone_shell(original_arm, contexts=tuple(contexts))
    rows[row_index] = _clone_shell(original_row, arms=tuple(arms))
    forged_bank = _clone_shell(canonical.bank, rows=tuple(rows))

    # Stationary/far_p1 is outside both the 22-context oracle subset and the
    # 108 transpose comparisons, so aggregate construction must not relabel it
    # as independently checked.  Complete regeneration must still find it.
    forged = _aggregate(forged_bank, canonical.controls)
    with patch.object(
        evidence,
        "collect_exposed_seed_science",
        return_value=canonical,
    ) as recollect:
        with pytest.raises(evidence.EvidenceV22Error, match="complete scientific replay"):
            evidence.validate_exposed_seed_science(forged)
    recollect.assert_called_once_with(config_sha256=CONFIG_SHA256)


def test_invalid_validation_type_fails_before_collection() -> None:
    with patch.object(evidence, "collect_exposed_seed_science") as recollect:
        with pytest.raises(evidence.EvidenceV22Error, match="requires"):
            evidence.validate_exposed_seed_science(cast(evidence.ExposedSeedScienceEvidence, object()))
    recollect.assert_not_called()
