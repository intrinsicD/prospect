"""Pure ordinary-scenario bank collection for MM-008 v2.2.

This module owns no filesystem, lifecycle, nonce, challenge, reserved-seed, or
real-data operation.  A caller supplies eight already generated cases, an exact
per-scenario seed authority, and a config hash.  Native row results are collected
or authenticated, sorted into the frozen scenario-major/row-major order, and
reduced to the exact ordinary-bank census.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-ordinary-bank-v1"
EXPOSED_DEV_SEED: Final = 12_345

if any(
    dependency.PROTOCOL_SHA256 != PROTOCOL_SHA256
    for dependency in (calibration, fitting, geometry, scoring, synthetic)
):
    raise RuntimeError("MM-008 v2.2 bank dependency binds a different protocol")

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_BOUNDARY_SCENARIOS: Final[tuple[synthetic.Scenario, ...]] = (
    "translation",
    "affine",
    "appearance",
    "combined",
    "coupled_boundary",
)
_SCENARIO_POSITION: Final = MappingProxyType(
    {scenario: index for index, scenario in enumerate(synthetic.SCENARIOS)}
)


class BankV22Error(ValueError):
    """Raised when ordinary-bank authority, coverage, or evidence fails closed."""


def _require_seed(seed: int, label: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise BankV22Error(f"{label} must be a built-in uint64 integer")
    return seed


def _require_config_sha256(value: str) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise BankV22Error("bank config SHA-256 must be 64 lowercase hexadecimal characters")
    return value


@dataclass(frozen=True, slots=True)
class SeedAuthority:
    """Immutable exact scenario-to-seed authority supplied by the caller."""

    entries: tuple[tuple[synthetic.Scenario, int], ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise BankV22Error("seed authority entries must be an immutable tuple")
        expected_scenarios = synthetic.SCENARIOS
        if tuple(scenario for scenario, _ in self.entries) != expected_scenarios:
            raise BankV22Error("seed authority must contain every scenario exactly once in order")
        for scenario, seed in self.entries:
            if scenario not in expected_scenarios:
                raise BankV22Error("seed authority contains an unknown scenario")
            _require_seed(seed, f"seed authority for {scenario}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, int]) -> SeedAuthority:
        """Copy one exact eight-key mapping into immutable canonical order."""

        if not isinstance(value, Mapping):
            raise BankV22Error("expected_seed_by_scenario must be a mapping")
        keys = set(value.keys())
        expected = set(synthetic.SCENARIOS)
        if keys != expected:
            raise BankV22Error("seed authority has missing or extra scenario keys")
        entries = tuple(
            (scenario, _require_seed(value[scenario], f"seed authority for {scenario}"))
            for scenario in synthetic.SCENARIOS
        )
        return cls(entries)

    def seed_for(self, scenario: synthetic.Scenario) -> int:
        if scenario not in synthetic.SCENARIOS:
            raise BankV22Error("seed lookup requires a known ordinary scenario")
        return next(seed for name, seed in self.entries if name == scenario)

    def as_mapping(self) -> Mapping[synthetic.Scenario, int]:
        """Return a fresh read-only view without exposing mutable authority."""

        return MappingProxyType(dict(self.entries))


def uniform_seed_authority(seed: int) -> SeedAuthority:
    """Construct an all-scenario authority from one explicit caller seed."""

    checked = _require_seed(seed, "uniform bank seed")
    return SeedAuthority(tuple((scenario, checked) for scenario in synthetic.SCENARIOS))


def exposed_seed_authority() -> SeedAuthority:
    """Return the declared all-12345 exposed-development authority without running it."""

    return uniform_seed_authority(EXPOSED_DEV_SEED)


@dataclass(frozen=True, slots=True)
class PrimitiveLedger:
    """Exact primitive census derived from the collected native row evidence."""

    scenario_arm_banks: int
    row_arms: int
    fitted_contexts: int
    grid_contexts: int
    bias_contexts: int
    persistence_records: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.scenario_arm_banks,
                self.row_arms,
                self.fitted_contexts,
                self.grid_contexts,
                self.bias_contexts,
                self.persistence_records,
            )
        ):
            raise BankV22Error("primitive ledger counts must be positive built-in integers")


EXPECTED_PRIMITIVE_LEDGER: Final = PrimitiveLedger(34, 204, 1_428, 630, 336, 48)
if (
    scoring.ORDINARY_COVERAGE.scenario_arm_banks,
    scoring.ORDINARY_COVERAGE.row_arms,
    scoring.ORDINARY_COVERAGE.fitted_contexts,
    scoring.ORDINARY_COVERAGE.grid_contexts,
    scoring.ORDINARY_COVERAGE.bias_contexts,
    scoring.ORDINARY_COVERAGE.persistence_records,
) != (
    EXPECTED_PRIMITIVE_LEDGER.scenario_arm_banks,
    EXPECTED_PRIMITIVE_LEDGER.row_arms,
    EXPECTED_PRIMITIVE_LEDGER.fitted_contexts,
    EXPECTED_PRIMITIVE_LEDGER.grid_contexts,
    EXPECTED_PRIMITIVE_LEDGER.bias_contexts,
    EXPECTED_PRIMITIVE_LEDGER.persistence_records,
):
    raise RuntimeError("MM-008 v2.2 scorer and bank ordinary censuses disagree")


@dataclass(frozen=True, slots=True)
class ScenarioExpectation:
    """Six-row conjunction for every required expectation in one scenario."""

    scenario: synthetic.Scenario
    expectation_names: tuple[str, ...]
    row_failures: tuple[tuple[str, ...], ...]
    row_passes: tuple[bool, ...]
    passed: bool

    def __post_init__(self) -> None:
        if self.scenario not in synthetic.SCENARIOS:
            raise BankV22Error("scenario expectation has an unknown scenario")
        if (
            type(self.expectation_names) is not tuple
            or not self.expectation_names
            or len(set(self.expectation_names)) != len(self.expectation_names)
            or any(type(name) is not str or not name or not name.isascii() for name in self.expectation_names)
        ):
            raise BankV22Error("scenario expectation names are invalid")
        if (
            type(self.row_failures) is not tuple
            or len(self.row_failures) != calibration.SYNTHETIC_ROWS
            or any(type(failures) is not tuple for failures in self.row_failures)
        ):
            raise BankV22Error("scenario expectation must contain six failure tuples")
        for failures in self.row_failures:
            expected_order = tuple(name for name in self.expectation_names if name in set(failures))
            if failures != expected_order or len(set(failures)) != len(failures):
                raise BankV22Error("row expectation failures are not an ordered subset")
        expected_passes = tuple(not failures for failures in self.row_failures)
        if self.row_passes != expected_passes or any(type(value) is not bool for value in self.row_passes):
            raise BankV22Error("row expectation pass flags are not recomputable")
        if type(self.passed) is not bool or self.passed is not all(expected_passes):
            raise BankV22Error("scenario expectation is not the exact six-row conjunction")


@dataclass(frozen=True, slots=True)
class BoundaryOccupancy:
    """Truth-level sampler, site-flow, and gradient boundary evidence."""

    scenario: synthetic.Scenario
    clip_count: int
    clip_denominator: int
    clip_occupancy: float
    site_flow_boundary_count: int
    site_flow_denominator: int
    site_flow_boundary_fraction: float
    gradient_boundary_count: int
    gradient_denominator: int
    gradient_boundary_fraction: float
    site_flow_boundary: bool
    gradient_boundary: bool
    max_abs_flow: float
    max_abs_gradient: float

    def __post_init__(self) -> None:
        if self.scenario not in _BOUNDARY_SCENARIOS:
            raise BankV22Error("boundary evidence has an undeclared scenario")
        count_denominators = (
            (self.clip_count, self.clip_denominator),
            (self.site_flow_boundary_count, self.site_flow_denominator),
            (self.gradient_boundary_count, self.gradient_denominator),
        )
        if any(
            type(count) is not int
            or type(denominator) is not int
            or denominator <= 0
            or not 0 <= count <= denominator
            for count, denominator in count_denominators
        ):
            raise BankV22Error("boundary count or denominator is invalid")
        fractions = (
            (self.clip_occupancy, self.clip_count / self.clip_denominator),
            (
                self.site_flow_boundary_fraction,
                self.site_flow_boundary_count / self.site_flow_denominator,
            ),
            (
                self.gradient_boundary_fraction,
                self.gradient_boundary_count / self.gradient_denominator,
            ),
        )
        if any(
            type(actual) is not float or not math.isfinite(actual) or actual != expected
            for actual, expected in fractions
        ):
            raise BankV22Error("boundary occupancy fraction is not recomputable")
        if (
            type(self.site_flow_boundary) is not bool
            or self.site_flow_boundary is not (self.site_flow_boundary_count > 0)
            or type(self.gradient_boundary) is not bool
            or self.gradient_boundary is not (self.gradient_boundary_count > 0)
        ):
            raise BankV22Error("boundary occupancy tags are not recomputable")
        if (
            type(self.max_abs_flow) is not float
            or type(self.max_abs_gradient) is not float
            or not math.isfinite(self.max_abs_flow)
            or not math.isfinite(self.max_abs_gradient)
            or self.max_abs_flow < 0.0
            or self.max_abs_gradient < 0.0
        ):
            raise BankV22Error("boundary maxima are invalid")

    @property
    def expectation_passes(self) -> bool:
        expected_boundary = self.scenario == "coupled_boundary"
        return (
            self.clip_count == 0
            and self.site_flow_boundary is expected_boundary
            and self.gradient_boundary is expected_boundary
            and (
                self.max_abs_flow == geometry.FLOW_LIMIT
                and self.max_abs_gradient == 4.0
                if expected_boundary
                else self.max_abs_flow < geometry.FLOW_LIMIT and self.max_abs_gradient < 4.0
            )
        )


def _boundary_occupancy(case: synthetic.SyntheticCase) -> BoundaryOccupancy:
    if case.scenario not in _BOUNDARY_SCENARIOS or case.truth is None:
        raise BankV22Error("boundary evidence requires a declared truth-bearing scenario")
    theta = case.truth.theta_array()
    gains = case.truth.gain_array()
    biases = case.truth.bias_array()
    normalized = geometry.GEOMETRY.normalized_coords
    u = normalized[:, 0]
    v = normalized[:, 1]
    dy = (theta[0] + theta[2] * u) + theta[3] * v
    dx = (theta[1] + theta[4] * u) + theta[5] * v
    flow = np.stack((dy, dx), axis=1)
    gain_at_clip = np.isclose(gains, fitting.GAIN_BOUNDS[0], rtol=0.0, atol=1e-12) | np.isclose(
        gains, fitting.GAIN_BOUNDS[1], rtol=0.0, atol=1e-12
    )
    bias_at_clip = np.isclose(biases, fitting.BIAS_BOUNDS[0], rtol=0.0, atol=1e-12) | np.isclose(
        biases, fitting.BIAS_BOUNDS[1], rtol=0.0, atol=1e-12
    )
    clip_count = int(
        np.count_nonzero(gain_at_clip) + np.count_nonzero(bias_at_clip)
    )
    site_count = int(
        np.count_nonzero(
            np.isclose(np.abs(flow), geometry.FLOW_LIMIT, rtol=0.0, atol=1e-12)
        )
    )
    gradients = theta[2:]
    gradient_count = int(
        np.count_nonzero(np.isclose(np.abs(gradients), 4.0, rtol=0.0, atol=1e-12))
    )
    clip_denominator = 2 * geometry.CHANNELS
    site_denominator = geometry.SITE_COUNT * 2
    gradient_denominator = 4
    return BoundaryOccupancy(
        scenario=case.scenario,
        clip_count=clip_count,
        clip_denominator=clip_denominator,
        clip_occupancy=float(clip_count / clip_denominator),
        site_flow_boundary_count=site_count,
        site_flow_denominator=site_denominator,
        site_flow_boundary_fraction=float(site_count / site_denominator),
        gradient_boundary_count=gradient_count,
        gradient_denominator=gradient_denominator,
        gradient_boundary_fraction=float(gradient_count / gradient_denominator),
        site_flow_boundary=site_count > 0,
        gradient_boundary=gradient_count > 0,
        max_abs_flow=float(np.max(np.abs(flow))),
        max_abs_gradient=float(np.max(np.abs(gradients))),
    )


def _expected_row_keys() -> tuple[tuple[synthetic.Scenario, int], ...]:
    return tuple(
        (scenario, row)
        for scenario in synthetic.SCENARIOS
        for row in range(calibration.SYNTHETIC_ROWS)
    )


def _sort_rows(rows: tuple[scoring.RowScore, ...]) -> tuple[scoring.RowScore, ...]:
    if type(rows) is not tuple or any(not isinstance(row, scoring.RowScore) for row in rows):
        raise BankV22Error("bank rows must be an immutable tuple of RowScore evidence")
    keys = tuple((row.scenario, row.row) for row in rows)
    if len(set(keys)) != len(keys):
        raise BankV22Error("bank row collection contains a duplicate scenario/row key")
    expected = _expected_row_keys()
    if set(keys) != set(expected):
        raise BankV22Error("bank row collection has missing or extra scenario/row keys")
    ordered = tuple(sorted(rows, key=lambda row: (_SCENARIO_POSITION[row.scenario], row.row)))
    if tuple((row.scenario, row.row) for row in ordered) != expected:
        raise BankV22Error("bank rows could not be placed in canonical order")
    return ordered


def _validate_cases(
    cases: tuple[synthetic.SyntheticCase, ...], authority: SeedAuthority
) -> tuple[synthetic.SyntheticCase, ...]:
    if type(cases) is not tuple or any(not isinstance(case, synthetic.SyntheticCase) for case in cases):
        raise BankV22Error("bank cases must be an immutable tuple of SyntheticCase values")
    scenarios = tuple(case.scenario for case in cases)
    if len(set(scenarios)) != len(scenarios):
        raise BankV22Error("bank case collection contains a duplicate scenario")
    if set(scenarios) != set(synthetic.SCENARIOS):
        raise BankV22Error("bank case collection has missing or extra scenarios")
    ordered = tuple(sorted(cases, key=lambda case: _SCENARIO_POSITION[case.scenario]))
    for expected_scenario, case in zip(synthetic.SCENARIOS, ordered, strict=True):
        if case.scenario != expected_scenario or case.seed != authority.seed_for(expected_scenario):
            raise BankV22Error("bank case identity differs from seed authority")
        try:
            synthetic.validate_case(case)
        except synthetic.SyntheticV22Error as error:
            raise BankV22Error("bank case failed exact generator replay") from error
    return ordered


def derive_primitive_ledger(rows: tuple[scoring.RowScore, ...]) -> PrimitiveLedger:
    """Derive the ordinary primitive census from complete nested row evidence."""

    ordered = _sort_rows(rows)
    scenario_arm_pairs: set[tuple[synthetic.Scenario, scoring.Arm]] = set()
    row_arm_count = 0
    fitted_count = 0
    grid_count = 0
    bias_count = 0
    for row in ordered:
        expected_arms = scoring.scenario_arms(row.scenario)
        if tuple(arm.arm for arm in row.arms) != expected_arms:
            raise BankV22Error("row arm membership differs from its scenario scope")
        row_arm_count += len(row.arms)
        for arm in row.arms:
            if tuple(context.plan.name for context in arm.contexts) != scoring.CONTEXT_ORDER:
                raise BankV22Error("row fitted contexts differ from the frozen seven-key order")
            scenario_arm_pairs.add((row.scenario, arm.arm))
            fitted_count += len(arm.contexts)
        if tuple(context.plan.name for context in row.bias.contexts) != scoring.CONTEXT_ORDER:
            raise BankV22Error("row bias contexts differ from the frozen seven-key order")
        bias_count += len(row.bias.contexts)
        row_grid_count = sum(len(stream.consumer_keys) for stream in row.grid_streams)
        fitted_grid_count = sum(
            len(arm.contexts) for arm in row.arms if arm.arm in scoring.GRID_ARMS
        )
        if row_grid_count != fitted_grid_count:
            raise BankV22Error("row grid stream census differs from its fitted grid contexts")
        grid_count += row_grid_count
    return PrimitiveLedger(
        scenario_arm_banks=len(scenario_arm_pairs),
        row_arms=row_arm_count,
        fitted_contexts=fitted_count,
        grid_contexts=grid_count,
        bias_contexts=bias_count,
        persistence_records=len(ordered),
    )


def _scenario_expectations(
    rows: tuple[scoring.RowScore, ...],
) -> tuple[ScenarioExpectation, ...]:
    by_scenario = {
        scenario: tuple(row for row in rows if row.scenario == scenario)
        for scenario in synthetic.SCENARIOS
    }
    summaries: list[ScenarioExpectation] = []
    for scenario in synthetic.SCENARIOS:
        scenario_rows = by_scenario[scenario]
        if len(scenario_rows) != calibration.SYNTHETIC_ROWS:
            raise BankV22Error("scenario does not contain exactly six row scores")
        expectation_names = tuple(check.name for check in scenario_rows[0].expectations)
        if not expectation_names:
            raise BankV22Error("scenario row contains no required expectation checks")
        if any(
            tuple(check.name for check in row.expectations) != expectation_names
            for row in scenario_rows
        ):
            raise BankV22Error("scenario rows disagree on required expectation membership")
        failures = tuple(row.expectation_failures for row in scenario_rows)
        passes = tuple(not row_failures for row_failures in failures)
        summaries.append(
            ScenarioExpectation(
                scenario=scenario,
                expectation_names=expectation_names,
                row_failures=failures,
                row_passes=passes,
                passed=all(passes),
            )
        )
    return tuple(summaries)


@dataclass(frozen=True, slots=True)
class OrdinaryBankEvidence:
    """Immutable complete native ordinary-bank evidence and derived controls."""

    protocol_sha256: str
    schema_version: str
    config_sha256: str
    seed_authority: SeedAuthority
    rows: tuple[scoring.RowScore, ...]
    primitive_ledger: PrimitiveLedger
    scenario_expectations: tuple[ScenarioExpectation, ...]
    boundary_occupancy: tuple[BoundaryOccupancy, ...]

    def __post_init__(self) -> None:
        if self.protocol_sha256 != PROTOCOL_SHA256 or self.schema_version != SCHEMA_VERSION:
            raise BankV22Error("bank evidence protocol or schema identity is invalid")
        _require_config_sha256(self.config_sha256)
        if not isinstance(self.seed_authority, SeedAuthority):
            raise BankV22Error("bank evidence lacks immutable seed authority")
        ordered = _sort_rows(self.rows)
        if any(left is not right for left, right in zip(ordered, self.rows, strict=True)):
            raise BankV22Error("persisted bank rows are not scenario-major and row-major")
        for row in self.rows:
            if row.config_sha256 != self.config_sha256:
                raise BankV22Error("bank row config differs from bank authority")
            if row.seed != self.seed_authority.seed_for(row.scenario):
                raise BankV22Error("bank row seed differs from bank authority")
            if row.input_orientation != "native":
                raise BankV22Error("ordinary bank rejects non-native row provenance")
        derived_ledger = derive_primitive_ledger(self.rows)
        if self.primitive_ledger != derived_ledger or derived_ledger != EXPECTED_PRIMITIVE_LEDGER:
            raise BankV22Error("bank primitive ledger differs from exact ordinary coverage")
        expected_summaries = _scenario_expectations(self.rows)
        if self.scenario_expectations != expected_summaries:
            raise BankV22Error("bank scenario conjunctions are not recomputable")
        if (
            type(self.boundary_occupancy) is not tuple
            or tuple(record.scenario for record in self.boundary_occupancy) != _BOUNDARY_SCENARIOS
            or any(not isinstance(record, BoundaryOccupancy) for record in self.boundary_occupancy)
        ):
            raise BankV22Error("bank boundary evidence membership is invalid")

    @property
    def expectations_pass(self) -> bool:
        return all(summary.passed for summary in self.scenario_expectations) and all(
            record.expectation_passes for record in self.boundary_occupancy
        )

    @property
    def expectation_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        for summary in self.scenario_expectations:
            for row, names in enumerate(summary.row_failures):
                failures.extend(f"{summary.scenario}/row-{row}/{name}" for name in names)
        failures.extend(
            f"{record.scenario}/boundary_occupancy"
            for record in self.boundary_occupancy
            if not record.expectation_passes
        )
        return tuple(failures)


def _build_bank(
    rows: tuple[scoring.RowScore, ...],
    cases: tuple[synthetic.SyntheticCase, ...],
    authority: SeedAuthority,
    config_sha256: str,
) -> OrdinaryBankEvidence:
    ordered_rows = _sort_rows(rows)
    case_by_scenario = {case.scenario: case for case in cases}
    boundaries = tuple(_boundary_occupancy(case_by_scenario[scenario]) for scenario in _BOUNDARY_SCENARIOS)
    return OrdinaryBankEvidence(
        protocol_sha256=PROTOCOL_SHA256,
        schema_version=SCHEMA_VERSION,
        config_sha256=config_sha256,
        seed_authority=authority,
        rows=ordered_rows,
        primitive_ledger=derive_primitive_ledger(ordered_rows),
        scenario_expectations=_scenario_expectations(ordered_rows),
        boundary_occupancy=boundaries,
    )


def collect_bank(
    cases: tuple[synthetic.SyntheticCase, ...],
    *,
    expected_seed_by_scenario: Mapping[str, int],
    config_sha256: str,
) -> OrdinaryBankEvidence:
    """Score all 48 native rows in exact scenario-major/row-major order."""

    authority = SeedAuthority.from_mapping(expected_seed_by_scenario)
    checked_config = _require_config_sha256(config_sha256)
    ordered_cases = _validate_cases(cases, authority)
    rows = tuple(
        scoring.score_row(case, row, config_sha256=checked_config)
        for case in ordered_cases
        for row in range(calibration.SYNTHETIC_ROWS)
    )
    return _build_bank(rows, ordered_cases, authority, checked_config)


def assemble_bank(
    rows: tuple[scoring.RowScore, ...],
    cases: tuple[synthetic.SyntheticCase, ...],
    *,
    expected_seed_by_scenario: Mapping[str, int],
    config_sha256: str,
) -> OrdinaryBankEvidence:
    """Authenticate unordered completed native rows, reject gaps, and collect them."""

    authority = SeedAuthority.from_mapping(expected_seed_by_scenario)
    checked_config = _require_config_sha256(config_sha256)
    ordered_cases = _validate_cases(cases, authority)
    ordered_rows = _sort_rows(rows)
    case_by_scenario = {case.scenario: case for case in ordered_cases}
    for row in ordered_rows:
        if row.config_sha256 != checked_config:
            raise BankV22Error("completed row config differs from collection authority")
        if row.seed != authority.seed_for(row.scenario):
            raise BankV22Error("completed row seed differs from collection authority")
        if row.input_orientation != "native":
            raise BankV22Error("ordinary collection rejects transposed row evidence")
        try:
            scoring.validate_row_score(row, case_by_scenario[row.scenario])
        except scoring.ScoringV22Error as error:
            raise BankV22Error("completed row failed complete deep replay") from error
    return _build_bank(ordered_rows, ordered_cases, authority, checked_config)


def validate_bank(
    evidence: OrdinaryBankEvidence,
    cases: tuple[synthetic.SyntheticCase, ...],
    *,
    expected_seed_by_scenario: Mapping[str, int],
    config_sha256: str,
) -> None:
    """Deeply replay every native row and reconstruct every derived bank field."""

    if not isinstance(evidence, OrdinaryBankEvidence):
        raise BankV22Error("bank validation requires OrdinaryBankEvidence")
    rebuilt = assemble_bank(
        evidence.rows,
        cases,
        expected_seed_by_scenario=expected_seed_by_scenario,
        config_sha256=config_sha256,
    )
    if evidence.config_sha256 != rebuilt.config_sha256 or evidence.seed_authority != rebuilt.seed_authority:
        raise BankV22Error("persisted bank authority differs from validation authority")
    if any(left is not right for left, right in zip(evidence.rows, rebuilt.rows, strict=True)):
        raise BankV22Error("persisted bank row collection differs from canonical collection")
    if (
        evidence.primitive_ledger != rebuilt.primitive_ledger
        or evidence.scenario_expectations != rebuilt.scenario_expectations
        or evidence.boundary_occupancy != rebuilt.boundary_occupancy
    ):
        raise BankV22Error("persisted bank derived evidence differs from deep reconstruction")


__all__ = [
    "BankV22Error",
    "BoundaryOccupancy",
    "EXPECTED_PRIMITIVE_LEDGER",
    "EXPOSED_DEV_SEED",
    "OrdinaryBankEvidence",
    "PROTOCOL_SHA256",
    "PrimitiveLedger",
    "SCHEMA_VERSION",
    "ScenarioExpectation",
    "SeedAuthority",
    "assemble_bank",
    "collect_bank",
    "derive_primitive_ledger",
    "exposed_seed_authority",
    "uniform_seed_authority",
    "validate_bank",
]
