"""Ephemeral exposed-seed science aggregation for MM-008 v2.2.

This module joins the already exposed ``PCG64(12345)`` native bank, scientific
controls, and one production-backed integrity panel.  It deliberately owns no
filesystem, serialization, lifecycle, runtime, nonce, challenge, reserved-seed,
or real-data behavior.  The aggregate is development science evidence only; it
is not a sealed-runtime DEV receipt or a formal result.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, fields, is_dataclass
from typing import Final, Literal, NoReturn, SupportsIndex

import numpy as np

from bench.multimodal_mechanism_diagnostics import bank_v22 as bank
from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import controls_v22 as controls
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import integrity_v22 as integrity
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-exposed-seed-science-v1"
DEV_SEED: Final = 12_345
CLAIM_SCOPE: Final = "exposed-seed-science-only-not-sealed-runtime-dev"
ORACLE_SCOPE: Final = "declared-22-context-subset-not-full-independent-coverage"

for _dependency in (
    bank,
    calibration,
    controls,
    fitting,
    geometry,
    exact,
    integrity,
    scoring,
    synthetic,
):
    if _dependency.PROTOCOL_SHA256 != PROTOCOL_SHA256:
        raise RuntimeError("MM-008 v2.2 exposed-seed evidence dependency binds a different protocol")

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_CENSUS_VALUES: Final = (
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
)


class EvidenceV22Error(ValueError):
    """Raised when exposed-seed evidence is incomplete, aliased wrongly, or stale."""


def _require_config_sha256(value: str) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise EvidenceV22Error("evidence config SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _float_bits_equal(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def _scientific_equal(left: object, right: object) -> bool:
    """Compare a nested immutable evidence graph without NumPy truth coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
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


@dataclass(frozen=True, slots=True)
class EvidenceCensus:
    """Exact disclosure of independent and production-only development coverage."""

    scenario_cases: int
    native_rows: int
    scenario_arm_banks: int
    row_arms: int
    fitted_contexts: int
    exact_grid_contexts: int
    affine_grid_contexts: int
    combined_grid_contexts: int
    bias_contexts: int
    persistence_records: int
    boundary_records: int
    oracle_contexts: int
    oracle_truth_contexts: int
    oracle_diagnostic_contexts: int
    production_only_grid_contexts: int
    mutation_controls: int
    transpose_comparisons: int
    named_controls: int
    integrity_bound_contexts: int
    delivery_orders: int
    delivery_rejections: int
    forgery_cases: int
    cache_checks: int
    oracle_scope: str

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field.name) for field in fields(self)[:-1])
        if values != _CENSUS_VALUES or any(type(value) is not int for value in values):
            raise EvidenceV22Error("evidence census differs from the exact exposed-seed disclosure")
        if self.oracle_scope != ORACLE_SCOPE:
            raise EvidenceV22Error("evidence census overstates or obscures independent oracle coverage")


EXPECTED_CENSUS: Final = EvidenceCensus(*_CENSUS_VALUES, ORACLE_SCOPE)


@dataclass(frozen=True, slots=True)
class IntegrityBinding:
    """Fixed bank context and one-ULP mutations used by the integrity panel."""

    scenario: Literal["combined"]
    row: int
    arm: Literal["combined"]
    context: Literal["true_full"]
    source_flat_index: int
    fit_target_flat_index: int
    mutation: Literal["nextafter-positive"]

    def __post_init__(self) -> None:
        if (
            self.scenario,
            self.row,
            self.arm,
            self.context,
            self.source_flat_index,
            self.fit_target_flat_index,
            self.mutation,
        ) != (
            "combined",
            0,
            "combined",
            "true_full",
            0,
            0,
            "nextafter-positive",
        ):
            raise EvidenceV22Error("integrity binding differs from the fixed exposed-seed context")


INTEGRITY_BINDING: Final = IntegrityBinding(
    "combined",
    0,
    "combined",
    "true_full",
    0,
    0,
    "nextafter-positive",
)


def _bank_rows(
    evidence: bank.OrdinaryBankEvidence,
) -> dict[tuple[synthetic.Scenario, int], scoring.RowScore]:
    rows = {(row.scenario, row.row): row for row in evidence.rows}
    if len(rows) != len(evidence.rows):
        raise EvidenceV22Error("native bank row identities are duplicated")
    return rows


def _grid_result(
    row: scoring.RowScore,
    arm: controls.GridArm,
    context: scoring.ContextName,
) -> exact.GlobalResult:
    try:
        arm_scores = row.arm(arm)
        context_score = next(item for item in arm_scores.contexts if item.plan.name == context)
    except (StopIteration, scoring.ScoringV22Error) as error:
        raise EvidenceV22Error("bank lacks a required exact-grid context") from error
    if not isinstance(context_score.estimate, exact.GlobalResult):
        raise EvidenceV22Error("bank exact-grid context has a non-G estimate")
    return context_score.estimate


def _bank_grid_results(
    evidence: bank.OrdinaryBankEvidence,
) -> dict[tuple[synthetic.Scenario, int, controls.GridArm, scoring.ContextName], exact.GlobalResult]:
    results: dict[
        tuple[synthetic.Scenario, int, controls.GridArm, scoring.ContextName],
        exact.GlobalResult,
    ] = {}
    for row in evidence.rows:
        for arm in scoring.SCENARIO_GRID_ARMS[row.scenario]:
            for context in scoring.CONTEXT_ORDER:
                results[(row.scenario, row.row, arm, context)] = _grid_result(row, arm, context)
    return results


def _derive_census(
    native_bank: bank.OrdinaryBankEvidence,
    scientific_controls: controls.ScientificControlsEvidence,
    integrity_evidence: integrity.IntegrityEvidence,
) -> EvidenceCensus:
    grid_results = _bank_grid_results(native_bank)
    affine_count = sum(key[2] == "affine" for key in grid_results)
    combined_count = sum(key[2] == "combined" for key in grid_results)
    truth_oracles = sum(entry.spec.requires_truth for entry in scientific_controls.oracle_panel.entries)
    oracle_count = len(scientific_controls.oracle_panel.entries)
    return EvidenceCensus(
        scenario_cases=len({row.scenario for row in native_bank.rows}),
        native_rows=len(native_bank.rows),
        scenario_arm_banks=sum(len(scoring.scenario_arms(scenario)) for scenario in synthetic.SCENARIOS),
        row_arms=sum(len(row.arms) for row in native_bank.rows),
        fitted_contexts=sum(len(arm.contexts) for row in native_bank.rows for arm in row.arms),
        exact_grid_contexts=len(grid_results),
        affine_grid_contexts=affine_count,
        combined_grid_contexts=combined_count,
        bias_contexts=sum(len(row.bias.contexts) for row in native_bank.rows),
        persistence_records=len(native_bank.rows),
        boundary_records=len(native_bank.boundary_occupancy),
        oracle_contexts=oracle_count,
        oracle_truth_contexts=truth_oracles,
        oracle_diagnostic_contexts=oracle_count - truth_oracles,
        production_only_grid_contexts=len(grid_results) - oracle_count,
        mutation_controls=len(scientific_controls.mutation_panel.entries),
        transpose_comparisons=len(scientific_controls.transpose_panel.entries),
        named_controls=len(scientific_controls.coverage.all_keys),
        integrity_bound_contexts=1,
        delivery_orders=len(integrity_evidence.delivery.orders),
        delivery_rejections=len(integrity_evidence.delivery.rejections),
        forgery_cases=len(integrity_evidence.forgery_matrix.witnesses),
        cache_checks=len(integrity_evidence.cache_identities.checks),
        oracle_scope=ORACLE_SCOPE,
    )


def _validate_aliases(
    native_bank: bank.OrdinaryBankEvidence,
    scientific_controls: controls.ScientificControlsEvidence,
) -> None:
    grid_results = _bank_grid_results(native_bank)
    oracle_keys: set[tuple[synthetic.Scenario, int, controls.GridArm, scoring.ContextName]] = set()
    for oracle_entry in scientific_controls.oracle_panel.entries:
        key = (
            oracle_entry.spec.scenario,
            oracle_entry.spec.row,
            oracle_entry.spec.arm,
            oracle_entry.spec.context,
        )
        expected = grid_results.get(key)
        if expected is None or oracle_entry.production is not expected:
            raise EvidenceV22Error("oracle production result is not the bank-owned context object")
        oracle_keys.add(key)
    if len(oracle_keys) != 22 or len(grid_results) - len(oracle_keys) != 608:
        raise EvidenceV22Error("oracle disclosure is not exactly 22 independent and 608 production-only contexts")

    transpose_keys: set[tuple[synthetic.Scenario, int, controls.GridArm, scoring.ContextName]] = set()
    for transpose_entry in scientific_controls.transpose_panel.entries:
        key = (
            transpose_entry.spec.scenario,
            transpose_entry.spec.row,
            transpose_entry.spec.arm,
            transpose_entry.spec.context,
        )
        expected = grid_results.get(key)
        if expected is None or transpose_entry.native is not expected:
            raise EvidenceV22Error("transpose native result is not the bank-owned context object")
        transpose_keys.add(key)
    if len(transpose_keys) != 108:
        raise EvidenceV22Error("transpose native alias coverage is not exactly 108 contexts")


def _validate_integrity_binding(
    native_bank: bank.OrdinaryBankEvidence,
    binding: IntegrityBinding,
    integrity_evidence: integrity.IntegrityEvidence,
) -> None:
    rows = _bank_rows(native_bank)
    try:
        row = rows[(binding.scenario, binding.row)]
    except KeyError as error:
        raise EvidenceV22Error("integrity binding row is absent from the native bank") from error
    bound = _grid_result(row, binding.arm, binding.context)
    canonical = integrity_evidence.delivery.orders[0]
    certificate = bound.certificate
    if not (
        canonical.selected_state_index == bound.selected.state_index
        and canonical.selected_admissible_rank == bound.selected.admissible_rank
        and canonical.exact_tie_multiplicity == certificate.exact_tie_multiplicity
        and _float_bits_equal(canonical.second_best_objective_gap, certificate.second_best_objective_gap)
        and _float_bits_equal(canonical.second_best_nonflow_gap, certificate.second_best_nonflow_gap)
        and canonical.objective_content_sha256 == bound.objective_cache.content_sha256
        and integrity_evidence.delivery.batch_sizes
        == tuple(len(record.indices) for record in bound.source_grid.batch_records)
        and integrity_evidence.delivery.scalar_replay_bit_exact
        is certificate.scalar_replay_bit_exact
    ):
        raise EvidenceV22Error("integrity evidence is not structurally bound to the fixed bank result")
    identities = dict(integrity_evidence.cache_identities.identities)
    if (
        identities.get("source/base") != bound.source_grid.scope_sha256
        or identities.get("objective/base-combined") != bound.objective_cache.scope_sha256
    ):
        raise EvidenceV22Error("integrity cache identities are not bound to the fixed bank scopes")


@dataclass(frozen=True, slots=True)
class ExposedSeedScienceEvidence:
    """In-memory exposed-seed evidence graph; never a receipt or serial artifact."""

    protocol_sha256: str
    schema_version: str
    config_sha256: str
    dev_seed: int
    claim_scope: str
    bank: bank.OrdinaryBankEvidence
    controls: controls.ScientificControlsEvidence
    integrity_binding: IntegrityBinding
    integrity: integrity.IntegrityEvidence
    census: EvidenceCensus

    def __post_init__(self) -> None:
        if self.protocol_sha256 != PROTOCOL_SHA256 or self.schema_version != SCHEMA_VERSION:
            raise EvidenceV22Error("exposed-seed evidence protocol or schema identity is invalid")
        checked_config = _require_config_sha256(self.config_sha256)
        if type(self.dev_seed) is not int or self.dev_seed != DEV_SEED:
            raise EvidenceV22Error("exposed-seed evidence must use only the declared seed 12345")
        if self.claim_scope != CLAIM_SCOPE:
            raise EvidenceV22Error("exposed-seed evidence claim scope overstates its authority")
        if (
            not isinstance(self.bank, bank.OrdinaryBankEvidence)
            or not isinstance(self.controls, controls.ScientificControlsEvidence)
            or not isinstance(self.integrity_binding, IntegrityBinding)
            or not isinstance(self.integrity, integrity.IntegrityEvidence)
            or not isinstance(self.census, EvidenceCensus)
        ):
            raise EvidenceV22Error("exposed-seed evidence contains an invalid nested member")
        if (
            self.bank.config_sha256 != checked_config
            or self.controls.oracle_panel.config_sha256 != checked_config
            or self.controls.mutation_panel.config_sha256 != checked_config
            or self.controls.transpose_panel.config_sha256 != checked_config
        ):
            raise EvidenceV22Error("exposed-seed evidence mixes config scopes")
        expected_authority = bank.exposed_seed_authority()
        if self.bank.seed_authority != expected_authority or any(
            row.seed != DEV_SEED for row in self.bank.rows
        ):
            raise EvidenceV22Error("native bank differs from the exact all-12345 seed authority")
        if self.integrity_binding != INTEGRITY_BINDING:
            raise EvidenceV22Error("exposed-seed evidence uses another integrity binding")
        _validate_aliases(self.bank, self.controls)
        _validate_integrity_binding(self.bank, self.integrity_binding, self.integrity)
        derived = _derive_census(self.bank, self.controls, self.integrity)
        if self.census != derived or derived != EXPECTED_CENSUS:
            raise EvidenceV22Error("exposed-seed evidence census is not recomputable")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("ExposedSeedScienceEvidence is ephemeral and non-serializable")


def _cases() -> tuple[synthetic.SyntheticCase, ...]:
    return tuple(
        synthetic.generate_case(scenario, seed=DEV_SEED)
        for scenario in synthetic.SCENARIOS
    )


def _assemble_transpose_panel(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
    rows: dict[tuple[synthetic.Scenario, int], scoring.RowScore],
    *,
    config_sha256: str,
) -> controls.TransposePanelEvidence:
    """Score only retained transposed contexts and release each row after comparison."""

    return controls.assemble_selective_transpose_panel(
        cases,
        rows,
        config_sha256=config_sha256,
    )


def _integrity_inputs(
    cases: dict[synthetic.Scenario, synthetic.SyntheticCase],
    rows: dict[tuple[synthetic.Scenario, int], scoring.RowScore],
) -> tuple[exact.GlobalResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binding = INTEGRITY_BINDING
    case = cases[binding.scenario]
    row = rows[(binding.scenario, binding.row)]
    result = _grid_result(row, binding.arm, binding.context)
    bundle = synthetic.row_targets(case, binding.row)
    source = np.ascontiguousarray(bundle.source, dtype=np.float64)
    fit_target = fitting.target_values(bundle.true_target, geometry.FULL_MASK)
    changed_source = source.copy()
    changed_target = fit_target.copy()
    source_flat = changed_source.reshape(-1)
    target_flat = changed_target.reshape(-1)
    source_flat[binding.source_flat_index] = np.nextafter(
        source_flat[binding.source_flat_index], np.inf
    )
    target_flat[binding.fit_target_flat_index] = np.nextafter(
        target_flat[binding.fit_target_flat_index], np.inf
    )
    return result, source, changed_source, fit_target, changed_target


def collect_exposed_seed_science(*, config_sha256: str) -> ExposedSeedScienceEvidence:
    """Collect the complete in-memory exposed-seed science graph with no injection seam."""

    checked_config = _require_config_sha256(config_sha256)
    generated_cases = _cases()
    case_by_scenario = {case.scenario: case for case in generated_cases}
    authority = bank.exposed_seed_authority()
    expected_seeds: dict[str, int] = {
        scenario: seed for scenario, seed in authority.entries
    }
    native_bank = bank.collect_bank(
        generated_cases,
        expected_seed_by_scenario=expected_seeds,
        config_sha256=checked_config,
    )
    row_by_key = _bank_rows(native_bank)
    row_zero = {scenario: row_by_key[(scenario, 0)] for scenario in synthetic.SCENARIOS}
    oracle_panel = controls.assemble_oracle_panel(
        case_by_scenario,
        row_zero,
        config_sha256=checked_config,
    )
    mutation_panel = controls.run_mutation_panel(
        case_by_scenario,
        config_sha256=checked_config,
    )
    transpose_panel = _assemble_transpose_panel(
        case_by_scenario,
        row_by_key,
        config_sha256=checked_config,
    )
    scientific_controls = controls.ScientificControlsEvidence(
        oracle_panel,
        mutation_panel,
        transpose_panel,
        controls.CONTROL_COVERAGE,
    )
    result, source, changed_source, fit_target, changed_target = _integrity_inputs(
        case_by_scenario,
        row_by_key,
    )
    integrity_evidence = integrity.run_integrity_controls(
        result,
        source,
        changed_source,
        fit_target,
        changed_target,
        geometry.FULL_MASK,
        geometry.FULL_MASK,
        config_sha256=checked_config,
    )
    census = _derive_census(native_bank, scientific_controls, integrity_evidence)
    return ExposedSeedScienceEvidence(
        protocol_sha256=PROTOCOL_SHA256,
        schema_version=SCHEMA_VERSION,
        config_sha256=checked_config,
        dev_seed=DEV_SEED,
        claim_scope=CLAIM_SCOPE,
        bank=native_bank,
        controls=scientific_controls,
        integrity_binding=INTEGRITY_BINDING,
        integrity=integrity_evidence,
        census=census,
    )


def validate_exposed_seed_science(evidence: ExposedSeedScienceEvidence) -> None:
    """Regenerate the complete bank and every panel, then compare every nested bit."""

    if not isinstance(evidence, ExposedSeedScienceEvidence):
        raise EvidenceV22Error("exposed-seed validation requires ExposedSeedScienceEvidence")
    regenerated = collect_exposed_seed_science(config_sha256=evidence.config_sha256)
    if not _scientific_equal(evidence, regenerated):
        raise EvidenceV22Error("exposed-seed evidence differs from complete scientific replay")


__all__ = [
    "CLAIM_SCOPE",
    "DEV_SEED",
    "EXPECTED_CENSUS",
    "EvidenceCensus",
    "EvidenceV22Error",
    "ExposedSeedScienceEvidence",
    "INTEGRITY_BINDING",
    "IntegrityBinding",
    "ORACLE_SCOPE",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "collect_exposed_seed_science",
    "validate_exposed_seed_science",
]
