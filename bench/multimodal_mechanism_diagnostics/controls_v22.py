"""Pure scientific controls for the exposed MM-008 v2.2 diagnostics.

The module owns no filesystem, lifecycle, runtime, RNG, nonce, challenge-seed,
or real-data behavior.  Callers supply already generated synthetic cases, or ask
the pure runners below to score those supplied cases.  Every evidence record is
immutable and can be rebuilt from its scientific inputs.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Final, Literal, TypeAlias, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import nongrid_v22 as nongrid
from bench.multimodal_mechanism_diagnostics import oracle_v22 as oracle
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic
from bench.multimodal_mechanism_diagnostics import transpose_v22 as transpose

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-scientific-controls-v1"
MUTATION_DELTA: Final = 123.0

for _dependency in (
    calibration,
    fitting,
    geometry,
    exact,
    nongrid,
    oracle,
    scoring,
    synthetic,
    transpose,
):
    if _dependency.PROTOCOL_SHA256 != PROTOCOL_SHA256:
        raise RuntimeError("MM-008 v2.2 controls dependency binds a different protocol")

_FLOAT64_LE: Final = np.dtype("<f8")
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_COVERAGE_TAG: Final = b"MM008-v2.2-scientific-control-coverage\0"

GridArm = Literal["affine", "combined"]
TrueContext = Literal["true_full", "true_p0", "true_p1"]
OracleContext = Literal["true_full", "true_p0", "true_p1", "near_p0", "far_p1"]
ScoreKey: TypeAlias = tuple[synthetic.Scenario, int]

TRUE_CONTEXTS: Final[tuple[TrueContext, ...]] = (
    "true_full",
    "true_p0",
    "true_p1",
)
TRANSPOSE_PAIRS: Final[tuple[tuple[synthetic.Scenario, GridArm], ...]] = (
    ("translation", "affine"),
    ("translation", "combined"),
    ("affine", "affine"),
    ("affine", "combined"),
    ("appearance", "combined"),
    ("combined", "combined"),
)
MUTATION_PAIRS: Final[tuple[tuple[synthetic.Scenario, GridArm], ...]] = (
    ("affine", "affine"),
    ("combined", "combined"),
)


class ControlsV22Error(ValueError):
    """Raised when a scientific control or its evidence fails closed."""


def _require_sha256(value: str, label: str) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise ControlsV22Error(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _FLOAT64_LE or not bool(np.all(np.isfinite(array))):
        raise ControlsV22Error("control array must be finite little-endian float64")
    contiguous = np.ascontiguousarray(array, dtype=_FLOAT64_LE)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=_FLOAT64_LE).reshape(contiguous.shape)


def _array_bits_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.shape == right.shape and left.dtype == right.dtype and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _float_bits_equal(left: float, right: float) -> bool:
    return struct.pack("<d", left) == struct.pack("<d", right)


def _scientific_equal(left: object, right: object) -> bool:
    """Compare immutable scientific values, retaining all IEEE-754 bits."""

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
            _scientific_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    if is_dataclass(left) and is_dataclass(right) and not isinstance(left, type) and not isinstance(right, type):
        left_names = tuple(field.name for field in fields(left))
        right_names = tuple(field.name for field in fields(right))
        return left_names == right_names and all(
            _scientific_equal(getattr(left, name), getattr(right, name)) for name in left_names
        )
    equality = left == right
    return type(equality) is bool and equality


def _coverage_sha256(categories: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_COVERAGE_TAG)
    digest.update(struct.pack("<H", len(categories)))
    for category, keys in categories:
        category_bytes = category.encode("ascii")
        digest.update(struct.pack("<H", len(category_bytes)))
        digest.update(category_bytes)
        digest.update(struct.pack("<H", len(keys)))
        for key in keys:
            payload = key.encode("ascii")
            digest.update(struct.pack("<H", len(payload)))
            digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OracleControlSpec:
    key: str
    scenario: synthetic.Scenario
    row: int
    arm: GridArm
    context: OracleContext
    requires_truth: bool

    def __post_init__(self) -> None:
        if self.scenario not in synthetic.SCENARIOS:
            raise ControlsV22Error("oracle scenario is invalid")
        if self.row != 0 or self.arm not in {"affine", "combined"}:
            raise ControlsV22Error("oracle control must be a row-0 exact-grid fit")
        if self.context not in {
            "true_full",
            "true_p0",
            "true_p1",
            "near_p0",
            "far_p1",
        }:
            raise ControlsV22Error("oracle context is invalid")
        expected_key = f"oracle/{self.scenario}/row-{self.row}/{self.arm}/{self.context}"
        if self.key != expected_key:
            raise ControlsV22Error("oracle coverage key is not canonical")
        expected_truth = (self.scenario, self.arm) in {
            ("translation", "affine"),
            ("affine", "affine"),
            ("combined", "combined"),
            ("coupled_boundary", "affine"),
        } and self.context in TRUE_CONTEXTS
        if type(self.requires_truth) is not bool or self.requires_truth is not expected_truth:
            raise ControlsV22Error("oracle truth requirement differs from the named panel")


@dataclass(frozen=True, slots=True)
class MutationControlSpec:
    key: str
    scenario: synthetic.Scenario
    row: int
    arm: GridArm
    output_parity: int

    @property
    def context(self) -> TrueContext:
        return cast(TrueContext, f"true_p{self.output_parity}")

    def __post_init__(self) -> None:
        if (
            (self.scenario, self.arm) not in {("affine", "affine"), ("combined", "combined")}
            or self.row != 0
            or type(self.output_parity) is not int
            or self.output_parity not in {0, 1}
        ):
            raise ControlsV22Error("held-target mutation specification is invalid")
        expected_key = f"mutation/{self.scenario}/row-{self.row}/{self.arm}/true_p{self.output_parity}"
        if self.key != expected_key:
            raise ControlsV22Error("mutation coverage key is not canonical")


@dataclass(frozen=True, slots=True)
class TransposeControlSpec:
    key: str
    scenario: synthetic.Scenario
    row: int
    arm: GridArm
    context: TrueContext

    def __post_init__(self) -> None:
        if (
            (self.scenario, self.arm) not in TRANSPOSE_PAIRS
            or type(self.row) is not int
            or not 0 <= self.row < calibration.SYNTHETIC_ROWS
            or self.context not in TRUE_CONTEXTS
        ):
            raise ControlsV22Error("transpose specification is invalid")
        expected_key = f"transpose/{self.scenario}/row-{self.row}/{self.arm}/{self.context}"
        if self.key != expected_key:
            raise ControlsV22Error("transpose coverage key is not canonical")


def _oracle_spec(
    scenario: synthetic.Scenario,
    arm: GridArm,
    context: OracleContext,
) -> OracleControlSpec:
    key = f"oracle/{scenario}/row-0/{arm}/{context}"
    requires_truth = (scenario, arm) in {
        ("translation", "affine"),
        ("affine", "affine"),
        ("combined", "combined"),
        ("coupled_boundary", "affine"),
    } and context in TRUE_CONTEXTS
    return OracleControlSpec(key, scenario, 0, arm, context, requires_truth)


ORACLE_SPECS: Final[tuple[OracleControlSpec, ...]] = (
    *(_oracle_spec("translation", "affine", context) for context in TRUE_CONTEXTS),
    *(_oracle_spec("affine", "affine", context) for context in TRUE_CONTEXTS),
    *(_oracle_spec("combined", "combined", context) for context in TRUE_CONTEXTS),
    *(_oracle_spec("coupled_boundary", "affine", context) for context in TRUE_CONTEXTS),
    *(_oracle_spec("independent", "combined", context) for context in TRUE_CONTEXTS),
    *(_oracle_spec("constant_target", "combined", context) for context in TRUE_CONTEXTS),
    _oracle_spec("translation", "affine", "near_p0"),
    _oracle_spec("translation", "affine", "far_p1"),
    _oracle_spec("translation", "combined", "near_p0"),
    _oracle_spec("translation", "combined", "far_p1"),
)

MUTATION_SPECS: Final[tuple[MutationControlSpec, ...]] = tuple(
    MutationControlSpec(
        f"mutation/{scenario}/row-0/{arm}/true_p{parity}",
        scenario,
        0,
        arm,
        parity,
    )
    for scenario, arm in MUTATION_PAIRS
    for parity in (0, 1)
)

TRANSPOSE_SPECS: Final[tuple[TransposeControlSpec, ...]] = tuple(
    TransposeControlSpec(
        f"transpose/{scenario}/row-{row}/{arm}/{context}",
        scenario,
        row,
        arm,
        context,
    )
    for scenario, arm in TRANSPOSE_PAIRS
    for row in range(calibration.SYNTHETIC_ROWS)
    for context in TRUE_CONTEXTS
)


@dataclass(frozen=True, slots=True)
class ControlCoverage:
    oracle_keys: tuple[str, ...]
    mutation_keys: tuple[str, ...]
    transpose_keys: tuple[str, ...]
    sha256: str

    @property
    def all_keys(self) -> tuple[str, ...]:
        return self.oracle_keys + self.mutation_keys + self.transpose_keys

    @property
    def counts(self) -> tuple[int, int, int, int]:
        return (
            len(self.oracle_keys),
            len(self.mutation_keys),
            len(self.transpose_keys),
            len(self.all_keys),
        )

    def __post_init__(self) -> None:
        expected = (
            tuple(spec.key for spec in ORACLE_SPECS),
            tuple(spec.key for spec in MUTATION_SPECS),
            tuple(spec.key for spec in TRANSPOSE_SPECS),
        )
        if (self.oracle_keys, self.mutation_keys, self.transpose_keys) != expected:
            raise ControlsV22Error("control coverage differs from the declared specifications")
        if self.counts != (22, 4, 108, 134):
            raise ControlsV22Error("control coverage census must be exactly 22/4/108/134")
        if len(set(self.all_keys)) != 134:
            raise ControlsV22Error("control coverage keys are not globally unique")
        for key in self.all_keys:
            if not key or not key.isascii() or len(key.encode("ascii")) > 0xFFFF:
                raise ControlsV22Error("control coverage key is malformed")
        actual = _coverage_sha256(
            (
                ("oracle", self.oracle_keys),
                ("mutation", self.mutation_keys),
                ("transpose", self.transpose_keys),
            )
        )
        if self.sha256 != actual:
            raise ControlsV22Error("control coverage hash is not recomputable")


_ORACLE_KEYS: Final = tuple(spec.key for spec in ORACLE_SPECS)
_MUTATION_KEYS: Final = tuple(spec.key for spec in MUTATION_SPECS)
_TRANSPOSE_KEYS: Final = tuple(spec.key for spec in TRANSPOSE_SPECS)
CONTROL_COVERAGE: Final = ControlCoverage(
    _ORACLE_KEYS,
    _MUTATION_KEYS,
    _TRANSPOSE_KEYS,
    _coverage_sha256(
        (
            ("oracle", _ORACLE_KEYS),
            ("mutation", _MUTATION_KEYS),
            ("transpose", _TRANSPOSE_KEYS),
        )
    ),
)
EXPECTED_COVERAGE_KEYS: Final = CONTROL_COVERAGE.all_keys


def _admissible_rank(state_index: int) -> int:
    positions = np.flatnonzero(geometry.ADMISSIBLE_INDICES == state_index)
    if positions.shape != (1,):
        raise ControlsV22Error("truth state is not exactly one admissible grid member")
    return int(positions[0])


def _parallel_oracle_equal(production: exact.GlobalResult, independent: oracle.OracleResult) -> bool:
    """Explicitly compare every corresponding field across both implementations."""

    source_fields = (
        "scope_sha256",
        "partition_sha256",
        "sample_stream_sha256",
        "content_sha256",
    )
    batch_fields = (
        "ordinal",
        "indices",
        "shape",
        "dtype",
        "sample_sha256",
        "batch_sha256",
    )
    objective_fields = (
        "arm",
        "objectives",
        "gains",
        "biases",
        "retained_macro_ids",
        "scope_sha256",
        "content_sha256",
    )
    selected_fields = (
        "state_index",
        "admissible_rank",
        "parameters",
        "objective",
        "gains",
        "biases",
        "retained_macro_ids",
        "fit_prediction",
        "evaluation_sha256",
    )
    certificate_fields = (
        "protocol_sha256",
        "config_sha256",
        "candidate_order_sha256",
        "admissible_list_sha256",
        "invalid_bitmap_sha256",
        "geometry_sha256",
        "source_scope_sha256",
        "source_content_sha256",
        "objective_scope_sha256",
        "objective_content_sha256",
        "candidate_count",
        "admissible_count",
        "inadmissible_count",
        "selected_total_rank",
        "selected_admissible_rank",
        "exact_tie_multiplicity",
        "second_best_objective_gap",
        "second_best_nonflow_gap",
        "selected_evaluation_sha256",
        "selected_prediction_sha256",
        "scalar_replay_bit_exact",
    )

    def corresponding_fields_equal(left: object, right: object, names: tuple[str, ...]) -> bool:
        return all(_scientific_equal(getattr(left, name), getattr(right, name)) for name in names)

    production_batches = production.source_grid.batch_records
    independent_batches = independent.source_grid.batch_records
    return (
        production.context_key == independent.context_key
        and production.arm == independent.arm
        and corresponding_fields_equal(production.source_grid, independent.source_grid, source_fields)
        and len(production_batches) == len(independent_batches)
        and all(
            corresponding_fields_equal(left, right, batch_fields)
            for left, right in zip(production_batches, independent_batches, strict=True)
        )
        and corresponding_fields_equal(
            production.objective_cache,
            independent.objective_cache,
            objective_fields,
        )
        and corresponding_fields_equal(production.selected, independent.selected, selected_fields)
        and _array_bits_equal(production.prediction, independent.prediction)
        and production.prediction_sha256 == independent.prediction_sha256
        and corresponding_fields_equal(production.certificate, independent.certificate, certificate_fields)
    )


@dataclass(frozen=True, slots=True)
class OracleComparisonDiagnostics:
    all_fields_bit_exact: bool
    production_state_index: int
    independent_state_index: int
    production_admissible_rank: int
    independent_admissible_rank: int
    production_exact_ties: int
    independent_exact_ties: int
    production_second_gap: float
    independent_second_gap: float
    production_nonflow_gap: float
    independent_nonflow_gap: float
    truth_state_index: int | None
    truth_admissible_rank: int | None
    truth_selected_bit_exact: bool | None
    nonflow_separated: bool | None

    def __post_init__(self) -> None:
        booleans = (self.all_fields_bit_exact,)
        if any(type(value) is not bool for value in booleans):
            raise ControlsV22Error("oracle equality diagnostic is invalid")
        integers = (
            self.production_state_index,
            self.independent_state_index,
            self.production_admissible_rank,
            self.independent_admissible_rank,
            self.production_exact_ties,
            self.independent_exact_ties,
        )
        if any(type(value) is not int for value in integers):
            raise ControlsV22Error("oracle integer diagnostic is invalid")
        floats = (
            self.production_second_gap,
            self.independent_second_gap,
            self.production_nonflow_gap,
            self.independent_nonflow_gap,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in floats):
            raise ControlsV22Error("oracle floating diagnostic is nonfinite")
        optional = (self.truth_selected_bit_exact, self.nonflow_separated)
        if any(value is not None and type(value) is not bool for value in optional):
            raise ControlsV22Error("oracle truth diagnostic is invalid")


def _oracle_diagnostics(
    spec: OracleControlSpec,
    production: exact.GlobalResult,
    independent: oracle.OracleResult,
) -> OracleComparisonDiagnostics:
    truth_state: int | None = None
    truth_rank: int | None = None
    truth_selected: bool | None = None
    separated: bool | None = None
    if spec.requires_truth:
        truth = synthetic.TRUTHS[spec.scenario]
        if truth is None:
            raise ControlsV22Error("truth-required oracle scenario has no injected truth")
        truth_state = geometry.state_index(truth.theta)
        truth_rank = _admissible_rank(truth_state)
        truth_selected = (
            production.selected.state_index == truth_state
            and independent.selected.state_index == truth_state
            and production.selected.admissible_rank == truth_rank
            and independent.selected.admissible_rank == truth_rank
        )
        separated = (
            production.certificate.second_best_nonflow_gap > 1e-12
            and independent.certificate.second_best_nonflow_gap > 1e-12
        )
    return OracleComparisonDiagnostics(
        all_fields_bit_exact=_parallel_oracle_equal(production, independent),
        production_state_index=production.selected.state_index,
        independent_state_index=independent.selected.state_index,
        production_admissible_rank=production.selected.admissible_rank,
        independent_admissible_rank=independent.selected.admissible_rank,
        production_exact_ties=production.certificate.exact_tie_multiplicity,
        independent_exact_ties=independent.certificate.exact_tie_multiplicity,
        production_second_gap=production.certificate.second_best_objective_gap,
        independent_second_gap=independent.certificate.second_best_objective_gap,
        production_nonflow_gap=production.certificate.second_best_nonflow_gap,
        independent_nonflow_gap=independent.certificate.second_best_nonflow_gap,
        truth_state_index=truth_state,
        truth_admissible_rank=truth_rank,
        truth_selected_bit_exact=truth_selected,
        nonflow_separated=separated,
    )


@dataclass(frozen=True, slots=True)
class OracleComparisonEvidence:
    spec: OracleControlSpec
    production: exact.GlobalResult
    independent: oracle.OracleResult
    diagnostics: OracleComparisonDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.spec, OracleControlSpec):
            raise ControlsV22Error("oracle evidence has an invalid specification")
        if not isinstance(self.production, exact.GlobalResult) or not isinstance(self.independent, oracle.OracleResult):
            raise ControlsV22Error("oracle evidence contains invalid result types")
        expected = _oracle_diagnostics(self.spec, self.production, self.independent)
        if not _scientific_equal(self.diagnostics, expected):
            raise ControlsV22Error("oracle diagnostics are not recomputable")
        if not expected.all_fields_bit_exact:
            raise ControlsV22Error("independent scalar oracle differs from production G")
        if self.spec.requires_truth and (
            expected.truth_selected_bit_exact is not True or expected.nonflow_separated is not True
        ):
            raise ControlsV22Error("positive oracle control lacks exact truth recovery/separation")
        if not math.isfinite(self.production.selected.objective) or not math.isfinite(
            self.independent.selected.objective
        ):
            raise ControlsV22Error("oracle selected objective is nonfinite")


def _context_inputs(
    case: synthetic.SyntheticCase,
    row: int,
    context: scoring.ContextName,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    bundle = synthetic.row_targets(case, row)
    if context == "true_full":
        fit_mask = geometry.FULL_MASK
        output_mask = geometry.FULL_MASK
    else:
        output_parity = 0 if context.endswith("p0") else 1
        fit_mask = geometry.PARITY_MASKS[1 - output_parity]
        output_mask = geometry.PARITY_MASKS[output_parity]
    target_kind = context.split("_", maxsplit=1)[0]
    if target_kind == "true":
        target = bundle.true_target
        target_row = row
    elif target_kind == "near":
        target = bundle.near_target
        target_row = bundle.near_row
    elif target_kind == "far":
        target = bundle.far_target
        target_row = bundle.far_row
    else:
        raise ControlsV22Error("unknown fitted-target kind")
    return (
        bundle.source,
        fitting.target_values(target, fit_mask),
        fit_mask,
        output_mask,
        target_row,
    )


def _grid_context(score: scoring.RowScore, arm: GridArm, context: scoring.ContextName) -> scoring.ArmContextScore:
    try:
        value = next(item for item in score.arm(arm).contexts if item.plan.name == context)
    except (StopIteration, scoring.ScoringV22Error) as error:
        raise ControlsV22Error("row score lacks the required exact-grid context") from error
    if not isinstance(value.estimate, exact.GlobalResult):
        raise ControlsV22Error("required row-score context is not a production G result")
    return value


def compare_oracle_context(
    spec: OracleControlSpec,
    case: synthetic.SyntheticCase,
    row_score: scoring.RowScore,
    *,
    config_sha256: str,
) -> OracleComparisonEvidence:
    """Run one named standalone scalar-oracle comparison."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if (
        not isinstance(case, synthetic.SyntheticCase)
        or not isinstance(row_score, scoring.RowScore)
        or case.scenario != spec.scenario
        or row_score.scenario != spec.scenario
        or row_score.seed != case.seed
        or row_score.row != spec.row
        or row_score.input_orientation != "native"
        or row_score.config_sha256 != checked_config
    ):
        raise ControlsV22Error("oracle case/score identity differs from its specification")
    context = _grid_context(row_score, spec.arm, spec.context)
    source, fit_target, fit_mask, output_mask, target_row = _context_inputs(case, spec.row, spec.context)
    if (
        context.plan.target_row != target_row
        or not np.array_equal(context.plan.fit_mask, fit_mask)
        or not np.array_equal(context.plan.output_mask, output_mask)
    ):
        raise ControlsV22Error("production context differs from independently derived inputs")
    production = cast(exact.GlobalResult, context.estimate)
    independent = oracle.fit_scalar_oracle(
        source,
        fit_target,
        fit_mask,
        output_mask,
        spec.arm,
        context_key=production.context_key,
        config_sha256=checked_config,
    )
    return OracleComparisonEvidence(
        spec,
        production,
        independent,
        _oracle_diagnostics(spec, production, independent),
    )


@dataclass(frozen=True, slots=True)
class OraclePanelEvidence:
    config_sha256: str
    entries: tuple[OracleComparisonEvidence, ...]
    coverage_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.config_sha256, "oracle-panel config SHA-256")
        if (
            type(self.entries) is not tuple
            or any(type(entry) is not OracleComparisonEvidence for entry in self.entries)
            or tuple(entry.spec for entry in self.entries) != ORACLE_SPECS
            or len({entry.spec.key for entry in self.entries}) != 22
        ):
            raise ControlsV22Error("oracle panel is not the exact ordered 22-control panel")
        if any(
            entry.production.certificate.config_sha256 != self.config_sha256
            or entry.independent.certificate.config_sha256 != self.config_sha256
            for entry in self.entries
        ):
            raise ControlsV22Error("oracle panel mixes config scopes")
        if self.coverage_sha256 != CONTROL_COVERAGE.sha256:
            raise ControlsV22Error("oracle panel does not bind the complete control coverage")


def assemble_oracle_panel(
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
    row_scores: Mapping[synthetic.Scenario, scoring.RowScore],
    *,
    config_sha256: str,
) -> OraclePanelEvidence:
    """Assemble the exact ordered 22-row0 oracle panel from native row scores."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    entries: list[OracleComparisonEvidence] = []
    for spec in ORACLE_SPECS:
        case = cases.get(spec.scenario)
        score = row_scores.get(spec.scenario)
        if case is None or score is None:
            raise ControlsV22Error("oracle panel is missing a required case or row score")
        entries.append(compare_oracle_context(spec, case, score, config_sha256=checked_config))
    return OraclePanelEvidence(checked_config, tuple(entries), CONTROL_COVERAGE.sha256)


def run_oracle_panel(
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
    *,
    config_sha256: str,
) -> OraclePanelEvidence:
    """Score the six required supplied cases and execute all 22 oracle controls."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    scenarios = tuple(dict.fromkeys(spec.scenario for spec in ORACLE_SPECS))
    scores: dict[synthetic.Scenario, scoring.RowScore] = {}
    for scenario in scenarios:
        case = cases.get(scenario)
        if case is None:
            raise ControlsV22Error("oracle runner is missing a required supplied case")
        scores[scenario] = scoring.score_row(case, 0, config_sha256=checked_config)
    return assemble_oracle_panel(cases, scores, config_sha256=checked_config)


def validate_oracle_panel(
    evidence: OraclePanelEvidence,
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    """Deeply rebuild every production score and independent oracle entry."""

    if not isinstance(evidence, OraclePanelEvidence):
        raise ControlsV22Error("oracle-panel validation requires OraclePanelEvidence")
    rebuilt = run_oracle_panel(cases, config_sha256=evidence.config_sha256)
    if not _scientific_equal(evidence, rebuilt):
        raise ControlsV22Error("oracle panel differs from complete scientific replay")


def mutate_held_target(target: np.ndarray, output_parity: int) -> np.ndarray:
    """Add the frozen +123 mutation only to held central parity sites/channels."""

    if (
        not isinstance(target, np.ndarray)
        or target.shape != (geometry.CHANNELS, geometry.NATIVE_SIZE, geometry.NATIVE_SIZE)
        or target.dtype != _FLOAT64_LE
        or not target.flags.c_contiguous
        or not bool(np.all(np.isfinite(target)))
        or type(output_parity) is not int
        or output_parity not in {0, 1}
    ):
        raise ControlsV22Error("held-target mutation input is invalid")
    changed = np.array(target, dtype=_FLOAT64_LE, order="C", copy=True)
    coords = geometry.GEOMETRY.coords[geometry.PARITY_MASKS[output_parity]].astype(np.intp)
    changed[:, coords[:, 0], coords[:, 1]] += MUTATION_DELTA
    return _immutable_float64(changed)


def _global_equal_ignoring_context(left: exact.GlobalResult, right: exact.GlobalResult) -> bool:
    return (
        left.arm == right.arm
        and _scientific_equal(left.source_grid, right.source_grid)
        and _scientific_equal(left.objective_cache, right.objective_cache)
        and _scientific_equal(left.selected, right.selected)
        and _array_bits_equal(left.prediction, right.prediction)
        and left.prediction_sha256 == right.prediction_sha256
        and _scientific_equal(left.certificate, right.certificate)
    )


@dataclass(frozen=True, slots=True)
class MutationDiagnostics:
    fit_target_bit_exact: bool
    source_grid_bit_exact: bool
    objective_cache_bit_exact: bool
    selected_parameters_bit_exact: bool
    selected_appearance_bit_exact: bool
    selection_bit_exact: bool
    held_prediction_bit_exact: bool
    bias_only_bit_exact: bool
    replay_bit_exact: bool
    output_target_changed: bool
    error_changed: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in fields_as_values(self)):
            raise ControlsV22Error("mutation diagnostics must be built-in booleans")


def fields_as_values(value: object) -> tuple[object, ...]:
    """Return dataclass values in declaration order without mutable reflection output."""

    if not is_dataclass(value) or isinstance(value, type):
        raise ControlsV22Error("field extraction requires a dataclass instance")
    return tuple(getattr(value, field.name) for field in fields(value))


def _mutation_diagnostics(
    original_fit_target: np.ndarray,
    mutated_fit_target: np.ndarray,
    original_output_target: np.ndarray,
    mutated_output_target: np.ndarray,
    original_result: exact.GlobalResult,
    mutated_result: exact.GlobalResult,
    replay_original_result: exact.GlobalResult,
    replay_mutated_result: exact.GlobalResult,
    original_bias: nongrid.BiasOnlyEstimate,
    mutated_bias: nongrid.BiasOnlyEstimate,
    replay_original_bias: nongrid.BiasOnlyEstimate,
    replay_mutated_bias: nongrid.BiasOnlyEstimate,
    original_error: calibration.ErrorRecord,
    mutated_error: calibration.ErrorRecord,
) -> MutationDiagnostics:
    left_selected = original_result.selected
    right_selected = mutated_result.selected
    selected_appearance = (
        _array_bits_equal(left_selected.gains, right_selected.gains)
        and _array_bits_equal(left_selected.biases, right_selected.biases)
        and left_selected.retained_macro_ids == right_selected.retained_macro_ids
    )
    replay = (
        _scientific_equal(original_result, replay_original_result)
        and _scientific_equal(mutated_result, replay_mutated_result)
        and _scientific_equal(original_bias, replay_original_bias)
        and _scientific_equal(mutated_bias, replay_mutated_bias)
    )
    return MutationDiagnostics(
        fit_target_bit_exact=_array_bits_equal(original_fit_target, mutated_fit_target),
        source_grid_bit_exact=_scientific_equal(original_result.source_grid, mutated_result.source_grid),
        objective_cache_bit_exact=_scientific_equal(original_result.objective_cache, mutated_result.objective_cache),
        selected_parameters_bit_exact=_array_bits_equal(left_selected.parameters, right_selected.parameters),
        selected_appearance_bit_exact=selected_appearance,
        selection_bit_exact=(
            _scientific_equal(left_selected, right_selected)
            and _scientific_equal(original_result.certificate, mutated_result.certificate)
        ),
        held_prediction_bit_exact=_array_bits_equal(original_result.prediction, mutated_result.prediction),
        bias_only_bit_exact=_scientific_equal(original_bias, mutated_bias),
        replay_bit_exact=replay,
        output_target_changed=not _array_bits_equal(original_output_target, mutated_output_target),
        error_changed=not _scientific_equal(original_error, mutated_error),
    )


@dataclass(frozen=True, slots=True)
class MutationComparisonEvidence:
    spec: MutationControlSpec
    original_fit_target: np.ndarray
    mutated_fit_target: np.ndarray
    original_output_target: np.ndarray
    mutated_output_target: np.ndarray
    original_result: exact.GlobalResult
    mutated_result: exact.GlobalResult
    replay_original_result: exact.GlobalResult
    replay_mutated_result: exact.GlobalResult
    original_bias: nongrid.BiasOnlyEstimate
    mutated_bias: nongrid.BiasOnlyEstimate
    replay_original_bias: nongrid.BiasOnlyEstimate
    replay_mutated_bias: nongrid.BiasOnlyEstimate
    original_error: calibration.ErrorRecord
    mutated_error: calibration.ErrorRecord
    diagnostics: MutationDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.spec, MutationControlSpec):
            raise ControlsV22Error("mutation evidence has an invalid specification")
        for name in (
            "original_fit_target",
            "mutated_fit_target",
            "original_output_target",
            "mutated_output_target",
        ):
            object.__setattr__(self, name, _immutable_float64(getattr(self, name)))
        result_values = (
            self.original_result,
            self.mutated_result,
            self.replay_original_result,
            self.replay_mutated_result,
        )
        if any(not isinstance(value, exact.GlobalResult) for value in result_values):
            raise ControlsV22Error("mutation evidence contains an invalid G result")
        bias_values = (
            self.original_bias,
            self.mutated_bias,
            self.replay_original_bias,
            self.replay_mutated_bias,
        )
        if any(not isinstance(value, nongrid.BiasOnlyEstimate) for value in bias_values):
            raise ControlsV22Error("mutation evidence contains an invalid bias-only fit")
        expected_original_error = calibration.error_record(self.original_result.prediction, self.original_output_target)
        expected_mutated_error = calibration.error_record(self.mutated_result.prediction, self.mutated_output_target)
        if not _scientific_equal(self.original_error, expected_original_error) or not (
            _scientific_equal(self.mutated_error, expected_mutated_error)
        ):
            raise ControlsV22Error("mutation errors do not reproduce from outputs")
        expected = _mutation_diagnostics(
            self.original_fit_target,
            self.mutated_fit_target,
            self.original_output_target,
            self.mutated_output_target,
            self.original_result,
            self.mutated_result,
            self.replay_original_result,
            self.replay_mutated_result,
            self.original_bias,
            self.mutated_bias,
            self.replay_original_bias,
            self.replay_mutated_bias,
            self.original_error,
            self.mutated_error,
        )
        if not _scientific_equal(self.diagnostics, expected):
            raise ControlsV22Error("mutation diagnostics are not recomputable")
        required = (
            expected.fit_target_bit_exact,
            expected.source_grid_bit_exact,
            expected.objective_cache_bit_exact,
            expected.selected_parameters_bit_exact,
            expected.selected_appearance_bit_exact,
            expected.selection_bit_exact,
            expected.held_prediction_bit_exact,
            expected.bias_only_bit_exact,
            expected.replay_bit_exact,
            expected.output_target_changed,
            expected.error_changed,
        )
        if not all(required) or not _global_equal_ignoring_context(self.original_result, self.mutated_result):
            raise ControlsV22Error("held-target mutation changed a fit or failed to reach scoring")


def run_mutation_control(
    spec: MutationControlSpec,
    case: synthetic.SyntheticCase,
    *,
    config_sha256: str,
) -> MutationComparisonEvidence:
    """Execute one held-target mutation and an independent deterministic replay."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    if not isinstance(case, synthetic.SyntheticCase) or case.scenario != spec.scenario:
        raise ControlsV22Error("mutation case identity differs from its specification")
    synthetic.validate_case(case)
    bundle = synthetic.row_targets(case, spec.row)
    fit_mask = geometry.PARITY_MASKS[1 - spec.output_parity]
    output_mask = geometry.PARITY_MASKS[spec.output_parity]
    mutated_target = mutate_held_target(bundle.true_target, spec.output_parity)
    original_fit = fitting.target_values(bundle.true_target, fit_mask)
    mutated_fit = fitting.target_values(mutated_target, fit_mask)
    original_output = fitting.target_values(bundle.true_target, output_mask)
    mutated_output = fitting.target_values(mutated_target, output_mask)

    requests = (
        exact.FitRequest.create(f"{spec.key}/original", spec.arm, original_fit),
        exact.FitRequest.create(f"{spec.key}/mutated", spec.arm, mutated_fit),
    )
    original_result, mutated_result = exact.fit_global_contexts(
        bundle.source,
        fit_mask,
        output_mask,
        requests,
        config_sha256=checked_config,
    )
    replay_original, replay_mutated = exact.fit_global_contexts(
        bundle.source,
        fit_mask,
        output_mask,
        requests,
        config_sha256=checked_config,
    )
    original_bias = nongrid.fit_bias_only(original_fit, fit_mask, output_mask, config_sha256=checked_config)
    mutated_bias = nongrid.fit_bias_only(mutated_fit, fit_mask, output_mask, config_sha256=checked_config)
    replay_original_bias = nongrid.fit_bias_only(original_fit, fit_mask, output_mask, config_sha256=checked_config)
    replay_mutated_bias = nongrid.fit_bias_only(mutated_fit, fit_mask, output_mask, config_sha256=checked_config)
    original_error = calibration.error_record(original_result.prediction, original_output)
    mutated_error = calibration.error_record(mutated_result.prediction, mutated_output)
    diagnostics = _mutation_diagnostics(
        original_fit,
        mutated_fit,
        original_output,
        mutated_output,
        original_result,
        mutated_result,
        replay_original,
        replay_mutated,
        original_bias,
        mutated_bias,
        replay_original_bias,
        replay_mutated_bias,
        original_error,
        mutated_error,
    )
    return MutationComparisonEvidence(
        spec,
        original_fit,
        mutated_fit,
        original_output,
        mutated_output,
        original_result,
        mutated_result,
        replay_original,
        replay_mutated,
        original_bias,
        mutated_bias,
        replay_original_bias,
        replay_mutated_bias,
        original_error,
        mutated_error,
        diagnostics,
    )


@dataclass(frozen=True, slots=True)
class MutationPanelEvidence:
    config_sha256: str
    entries: tuple[MutationComparisonEvidence, ...]
    coverage_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.config_sha256, "mutation-panel config SHA-256")
        if (
            type(self.entries) is not tuple
            or any(type(entry) is not MutationComparisonEvidence for entry in self.entries)
            or tuple(entry.spec for entry in self.entries) != MUTATION_SPECS
            or len({entry.spec.key for entry in self.entries}) != 4
        ):
            raise ControlsV22Error("mutation panel is not the exact ordered four-control panel")
        if any(entry.original_result.certificate.config_sha256 != self.config_sha256 for entry in self.entries):
            raise ControlsV22Error("mutation panel mixes config scopes")
        if self.coverage_sha256 != CONTROL_COVERAGE.sha256:
            raise ControlsV22Error("mutation panel does not bind complete control coverage")


def run_mutation_panel(
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
    *,
    config_sha256: str,
) -> MutationPanelEvidence:
    """Execute all four exact held-target mutation controls."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    entries: list[MutationComparisonEvidence] = []
    for spec in MUTATION_SPECS:
        case = cases.get(spec.scenario)
        if case is None:
            raise ControlsV22Error("mutation panel is missing a required supplied case")
        entries.append(run_mutation_control(spec, case, config_sha256=checked_config))
    return MutationPanelEvidence(checked_config, tuple(entries), CONTROL_COVERAGE.sha256)


def validate_mutation_panel(
    evidence: MutationPanelEvidence,
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    """Deeply replay all four held-target mutation controls."""

    if not isinstance(evidence, MutationPanelEvidence):
        raise ControlsV22Error("mutation-panel validation requires MutationPanelEvidence")
    rebuilt = run_mutation_panel(cases, config_sha256=evidence.config_sha256)
    if not _scientific_equal(evidence, rebuilt):
        raise ControlsV22Error("mutation panel differs from complete scientific replay")


def _transpose_mask(mask: np.ndarray) -> np.ndarray:
    reshaped = np.asarray(mask, dtype=np.bool_).reshape(geometry.CENTRAL_SIZE, geometry.CENTRAL_SIZE)
    return np.ascontiguousarray(reshaped.T.reshape(-1), dtype=np.bool_)


def _transpose_masked_prediction(
    prediction: np.ndarray,
    native_mask: np.ndarray,
    transposed_mask: np.ndarray,
) -> np.ndarray:
    native_count = int(np.count_nonzero(native_mask))
    transposed_count = int(np.count_nonzero(transposed_mask))
    if prediction.shape != (geometry.CHANNELS, native_count) or native_count != transposed_count:
        raise ControlsV22Error("masked prediction cannot be transposed")
    canvas = np.zeros(
        (geometry.CHANNELS, geometry.CENTRAL_SIZE, geometry.CENTRAL_SIZE),
        dtype=_FLOAT64_LE,
    )
    canvas[:, native_mask.reshape(geometry.CENTRAL_SIZE, geometry.CENTRAL_SIZE)] = prediction
    transposed = np.swapaxes(canvas, -2, -1)
    selected = transposed[:, transposed_mask.reshape(geometry.CENTRAL_SIZE, geometry.CENTRAL_SIZE)]
    return _immutable_float64(selected)


def _transpose_retained(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted((value % 6) * 6 + value // 6 for value in values))


def _max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        raise ControlsV22Error("max-absolute diagnostic arrays have incompatible shapes")
    value = float(np.max(np.abs(left - right)))
    if not math.isfinite(value):
        raise ControlsV22Error("max-absolute diagnostic is nonfinite")
    return value


@dataclass(frozen=True, slots=True)
class TransposeDiagnostics:
    expected_transposed_state_index: int
    unique_minima: bool
    theta_permutation_exact: bool
    context_identity_exact: bool
    fit_mask_transpose_exact: bool
    output_mask_transpose_exact: bool
    retained_ids_transpose_exact: bool | None
    prediction_transpose_bit_exact: bool
    prediction_max_abs_delta: float
    fit_prediction_transpose_bit_exact: bool
    fit_prediction_max_abs_delta: float
    objective_bit_exact: bool
    objective_abs_delta: float
    gains_bit_exact: bool | None
    gains_max_abs_delta: float | None
    biases_bit_exact: bool | None
    biases_max_abs_delta: float | None
    second_gap_bit_exact: bool
    second_gap_abs_delta: float
    nonflow_gap_bit_exact: bool
    nonflow_gap_abs_delta: float
    error_bit_exact: bool
    error_max_abs_delta: float

    @property
    def passes(self) -> bool:
        """Return the exact Section-8 predicate, with no floating tolerance."""

        return self.unique_minima and self.theta_permutation_exact

    def __post_init__(self) -> None:
        if type(self.expected_transposed_state_index) is not int:
            raise ControlsV22Error("transpose expected state index is invalid")
        required_bools = (
            self.unique_minima,
            self.theta_permutation_exact,
            self.context_identity_exact,
            self.fit_mask_transpose_exact,
            self.output_mask_transpose_exact,
            self.prediction_transpose_bit_exact,
            self.fit_prediction_transpose_bit_exact,
            self.objective_bit_exact,
            self.second_gap_bit_exact,
            self.nonflow_gap_bit_exact,
            self.error_bit_exact,
        )
        if any(type(value) is not bool for value in required_bools):
            raise ControlsV22Error("transpose bit-equality diagnostic is invalid")
        optional_bools = (
            self.retained_ids_transpose_exact,
            self.gains_bit_exact,
            self.biases_bit_exact,
        )
        if any(value is not None and type(value) is not bool for value in optional_bools):
            raise ControlsV22Error("transpose optional diagnostic is invalid")
        floats = (
            self.prediction_max_abs_delta,
            self.fit_prediction_max_abs_delta,
            self.objective_abs_delta,
            self.second_gap_abs_delta,
            self.nonflow_gap_abs_delta,
            self.error_max_abs_delta,
        )
        if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in floats):
            raise ControlsV22Error("transpose max-absolute diagnostic is invalid")
        optional_floats = (self.gains_max_abs_delta, self.biases_max_abs_delta)
        if any(
            value is not None and (type(value) is not float or not math.isfinite(value) or value < 0.0)
            for value in optional_floats
        ):
            raise ControlsV22Error("transpose appearance diagnostic is invalid")


def _transpose_diagnostics(
    spec: TransposeControlSpec,
    native_plan: scoring.ContextPlan,
    transposed_plan: scoring.ContextPlan,
    native: exact.GlobalResult,
    transposed: exact.GlobalResult,
    native_error: calibration.ErrorRecord,
    transposed_error: calibration.ErrorRecord,
) -> TransposeDiagnostics:
    expected_theta = synthetic.transpose_theta(tuple(float(value) for value in native.selected.parameters))
    expected_parameters = np.asarray(expected_theta, dtype=_FLOAT64_LE)
    expected_state = geometry.state_index(expected_theta)
    expected_prediction = _transpose_masked_prediction(
        native.prediction, native_plan.output_mask, transposed_plan.output_mask
    )
    expected_fit_prediction = _transpose_masked_prediction(
        native.selected.fit_prediction,
        native_plan.fit_mask,
        transposed_plan.fit_mask,
    )
    if native.selected.gains is None or transposed.selected.gains is None:
        gains_bits: bool | None = None
        gains_delta: float | None = None
    else:
        gains_bits = _array_bits_equal(native.selected.gains, transposed.selected.gains)
        gains_delta = _max_abs_delta(native.selected.gains, transposed.selected.gains)
    if native.selected.biases is None or transposed.selected.biases is None:
        biases_bits: bool | None = None
        biases_delta: float | None = None
    else:
        biases_bits = _array_bits_equal(native.selected.biases, transposed.selected.biases)
        biases_delta = _max_abs_delta(native.selected.biases, transposed.selected.biases)
    retained_exact: bool | None = None
    if spec.arm == "combined":
        retained_exact = (
            _transpose_retained(native.selected.retained_macro_ids) == transposed.selected.retained_macro_ids
        )
    error_delta = max(
        abs(native_error.sse - transposed_error.sse),
        abs(native_error.mse - transposed_error.mse),
    )
    return TransposeDiagnostics(
        expected_transposed_state_index=expected_state,
        unique_minima=(
            native.certificate.exact_tie_multiplicity == 1 and transposed.certificate.exact_tie_multiplicity == 1
        ),
        theta_permutation_exact=(
            transposed.selected.state_index == expected_state
            and _array_bits_equal(transposed.selected.parameters, expected_parameters)
        ),
        context_identity_exact=(
            native_plan.name == transposed_plan.name == spec.context
            and native_plan.target_kind == transposed_plan.target_kind == "true"
            and native_plan.target_row == transposed_plan.target_row == spec.row
        ),
        fit_mask_transpose_exact=np.array_equal(_transpose_mask(native_plan.fit_mask), transposed_plan.fit_mask),
        output_mask_transpose_exact=np.array_equal(
            _transpose_mask(native_plan.output_mask), transposed_plan.output_mask
        ),
        retained_ids_transpose_exact=retained_exact,
        prediction_transpose_bit_exact=_array_bits_equal(expected_prediction, transposed.prediction),
        prediction_max_abs_delta=_max_abs_delta(expected_prediction, transposed.prediction),
        fit_prediction_transpose_bit_exact=_array_bits_equal(
            expected_fit_prediction, transposed.selected.fit_prediction
        ),
        fit_prediction_max_abs_delta=_max_abs_delta(expected_fit_prediction, transposed.selected.fit_prediction),
        objective_bit_exact=_float_bits_equal(native.selected.objective, transposed.selected.objective),
        objective_abs_delta=abs(native.selected.objective - transposed.selected.objective),
        gains_bit_exact=gains_bits,
        gains_max_abs_delta=gains_delta,
        biases_bit_exact=biases_bits,
        biases_max_abs_delta=biases_delta,
        second_gap_bit_exact=_float_bits_equal(
            native.certificate.second_best_objective_gap,
            transposed.certificate.second_best_objective_gap,
        ),
        second_gap_abs_delta=abs(
            native.certificate.second_best_objective_gap - transposed.certificate.second_best_objective_gap
        ),
        nonflow_gap_bit_exact=_float_bits_equal(
            native.certificate.second_best_nonflow_gap,
            transposed.certificate.second_best_nonflow_gap,
        ),
        nonflow_gap_abs_delta=abs(
            native.certificate.second_best_nonflow_gap - transposed.certificate.second_best_nonflow_gap
        ),
        error_bit_exact=_scientific_equal(native_error, transposed_error),
        error_max_abs_delta=error_delta,
    )


@dataclass(frozen=True, slots=True)
class TransposeComparisonEvidence:
    spec: TransposeControlSpec
    native_plan: scoring.ContextPlan
    transposed_plan: scoring.ContextPlan
    native: exact.GlobalResult
    transposed: exact.GlobalResult
    native_error: calibration.ErrorRecord
    transposed_error: calibration.ErrorRecord
    diagnostics: TransposeDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TransposeControlSpec):
            raise ControlsV22Error("transpose evidence has an invalid specification")
        if not isinstance(self.native_plan, scoring.ContextPlan) or not isinstance(
            self.transposed_plan, scoring.ContextPlan
        ):
            raise ControlsV22Error("transpose evidence contains an invalid context plan")
        if not isinstance(self.native, exact.GlobalResult) or not isinstance(self.transposed, exact.GlobalResult):
            raise ControlsV22Error("transpose evidence contains an invalid G result")
        if self.native.arm != self.transposed.arm or self.native.arm != self.spec.arm:
            raise ControlsV22Error("transpose result arm differs from its specification")
        expected = _transpose_diagnostics(
            self.spec,
            self.native_plan,
            self.transposed_plan,
            self.native,
            self.transposed,
            self.native_error,
            self.transposed_error,
        )
        if not _scientific_equal(self.diagnostics, expected):
            raise ControlsV22Error("transpose diagnostics are not recomputable")
        if not (
            expected.context_identity_exact
            and expected.fit_mask_transpose_exact
            and expected.output_mask_transpose_exact
        ):
            raise ControlsV22Error("transpose control context identity or mask mapping is invalid")
        if not expected.passes:
            raise ControlsV22Error("transpose control lacks unique minima or exact canonical theta permutation")


def compare_transpose_context(
    spec: TransposeControlSpec,
    native_score: scoring.RowScore,
    transposed_score: scoring.RowScore,
) -> TransposeComparisonEvidence:
    """Compare one native/transposed true context under the exact Section-8 rule."""

    if (
        not isinstance(native_score, scoring.RowScore)
        or not isinstance(transposed_score, scoring.RowScore)
        or native_score.scenario != transposed_score.scenario
        or native_score.scenario != spec.scenario
        or native_score.seed != transposed_score.seed
        or native_score.row != transposed_score.row
        or native_score.row != spec.row
        or native_score.config_sha256 != transposed_score.config_sha256
        or native_score.input_orientation != "native"
        or transposed_score.input_orientation != "transposed"
    ):
        raise ControlsV22Error("transpose score identities differ from the specification")
    native_context = _grid_context(native_score, spec.arm, spec.context)
    transposed_context = _grid_context(transposed_score, spec.arm, spec.context)
    native = cast(exact.GlobalResult, native_context.estimate)
    transposed = cast(exact.GlobalResult, transposed_context.estimate)
    diagnostics = _transpose_diagnostics(
        spec,
        native_context.plan,
        transposed_context.plan,
        native,
        transposed,
        native_context.error,
        transposed_context.error,
    )
    return TransposeComparisonEvidence(
        spec,
        native_context.plan,
        transposed_context.plan,
        native,
        transposed,
        native_context.error,
        transposed_context.error,
        diagnostics,
    )


def compare_selective_transpose_context(
    spec: TransposeControlSpec,
    native_score: scoring.RowScore,
    transposed_score: transpose.TransposedGridControlRow,
) -> TransposeComparisonEvidence:
    """Compare one bank-owned native context to the closed selective transpose path."""

    if (
        not isinstance(native_score, scoring.RowScore)
        or not isinstance(transposed_score, transpose.TransposedGridControlRow)
        or native_score.scenario != transposed_score.scenario
        or native_score.scenario != spec.scenario
        or native_score.seed != transposed_score.seed
        or native_score.row != transposed_score.row
        or native_score.row != spec.row
        or native_score.config_sha256 != transposed_score.config_sha256
        or native_score.input_orientation != "native"
        or transposed_score.input_orientation != "transposed"
    ):
        raise ControlsV22Error("selective transpose identities differ from the specification")
    native_context = _grid_context(native_score, spec.arm, spec.context)
    transposed_context = transposed_score.context(spec.arm, spec.context)
    native = cast(exact.GlobalResult, native_context.estimate)
    diagnostics = _transpose_diagnostics(
        spec,
        native_context.plan,
        transposed_context.plan,
        native,
        transposed_context.result,
        native_context.error,
        transposed_context.error,
    )
    return TransposeComparisonEvidence(
        spec,
        native_context.plan,
        transposed_context.plan,
        native,
        transposed_context.result,
        native_context.error,
        transposed_context.error,
        diagnostics,
    )


@dataclass(frozen=True, slots=True)
class TransposePanelEvidence:
    config_sha256: str
    entries: tuple[TransposeComparisonEvidence, ...]
    coverage_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.config_sha256, "transpose-panel config SHA-256")
        if (
            type(self.entries) is not tuple
            or any(type(entry) is not TransposeComparisonEvidence for entry in self.entries)
            or tuple(entry.spec for entry in self.entries) != TRANSPOSE_SPECS
            or len({entry.spec.key for entry in self.entries}) != 108
        ):
            raise ControlsV22Error("transpose panel is not the exact ordered 108-control panel")
        if any(
            entry.native.certificate.config_sha256 != self.config_sha256
            or entry.transposed.certificate.config_sha256 != self.config_sha256
            for entry in self.entries
        ):
            raise ControlsV22Error("transpose panel mixes config scopes")
        _validate_transpose_source_aliases(self.entries)
        if self.coverage_sha256 != CONTROL_COVERAGE.sha256:
            raise ControlsV22Error("transpose panel does not bind complete control coverage")


def _validate_transpose_source_aliases(
    entries: tuple[TransposeComparisonEvidence, ...],
) -> None:
    """Require same-stream affine/combined records to retain source-grid identity."""

    groups: dict[
        tuple[synthetic.Scenario, int, TrueContext],
        list[TransposeComparisonEvidence],
    ] = {}
    for entry in entries:
        key = (entry.spec.scenario, entry.spec.row, entry.spec.context)
        groups.setdefault(key, []).append(entry)
    for group in groups.values():
        if len(group) < 2:
            continue
        native_source = group[0].native.source_grid
        transposed_source = group[0].transposed.source_grid
        if any(
            entry.native.source_grid is not native_source
            or entry.transposed.source_grid is not transposed_source
            for entry in group[1:]
        ):
            raise ControlsV22Error("transpose panel fractures same-stream source-grid aliases")


def assemble_transpose_panel(
    native_scores: Mapping[ScoreKey, scoring.RowScore],
    transposed_scores: Mapping[ScoreKey, scoring.RowScore],
    *,
    config_sha256: str,
) -> TransposePanelEvidence:
    """Assemble all 108 comparisons from exact native/transposed row scores."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    entries: list[TransposeComparisonEvidence] = []
    for spec in TRANSPOSE_SPECS:
        key = (spec.scenario, spec.row)
        native = native_scores.get(key)
        transposed = transposed_scores.get(key)
        if native is None or transposed is None:
            raise ControlsV22Error("transpose panel is missing a required row-score pair")
        if native.config_sha256 != checked_config or transposed.config_sha256 != checked_config:
            raise ControlsV22Error("transpose row-score config differs from the panel")
        entries.append(compare_transpose_context(spec, native, transposed))
    return TransposePanelEvidence(checked_config, tuple(entries), CONTROL_COVERAGE.sha256)


def assemble_selective_transpose_panel(
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
    native_scores: Mapping[ScoreKey, scoring.RowScore],
    *,
    config_sha256: str,
) -> TransposePanelEvidence:
    """Assemble 108 comparisons while scoring only the 108 retained transposed contexts."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    by_key: dict[str, TransposeComparisonEvidence] = {}
    scenarios = tuple(dict.fromkeys(scenario for scenario, _ in TRANSPOSE_PAIRS))
    for scenario in scenarios:
        case = cases.get(scenario)
        if case is None or case.scenario != scenario:
            raise ControlsV22Error("selective transpose panel is missing a required case")
        for row in range(calibration.SYNTHETIC_ROWS):
            native = native_scores.get((scenario, row))
            if native is None or native.config_sha256 != checked_config:
                raise ControlsV22Error(
                    "selective transpose panel is missing a config-matched native row"
                )
            transposed = transpose.score_transposed_grid_control_row(
                case,
                row,
                config_sha256=checked_config,
            )
            for spec in TRANSPOSE_SPECS:
                if spec.scenario == scenario and spec.row == row:
                    by_key[spec.key] = compare_selective_transpose_context(
                        spec,
                        native,
                        transposed,
                    )
            del transposed
    if set(by_key) != {spec.key for spec in TRANSPOSE_SPECS}:
        raise ControlsV22Error("selective transpose panel did not produce the exact 108 keys")
    entries = tuple(by_key[spec.key] for spec in TRANSPOSE_SPECS)
    return TransposePanelEvidence(checked_config, entries, CONTROL_COVERAGE.sha256)


def validate_selective_transpose_panel(
    evidence: TransposePanelEvidence,
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
    native_scores: Mapping[ScoreKey, scoring.RowScore],
) -> None:
    """Deeply regenerate the closed 108-context transpose panel and compare every bit."""

    if not isinstance(evidence, TransposePanelEvidence):
        raise ControlsV22Error("selective transpose validation requires TransposePanelEvidence")
    rebuilt = assemble_selective_transpose_panel(
        cases,
        native_scores,
        config_sha256=evidence.config_sha256,
    )
    if not _scientific_equal(evidence, rebuilt):
        raise ControlsV22Error("selective transpose panel differs from complete replay")


def run_transpose_panel(
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
    *,
    config_sha256: str,
) -> TransposePanelEvidence:
    """Score the supplied cases and execute all 108 transpose comparisons."""

    checked_config = _require_sha256(config_sha256, "config SHA-256")
    native: dict[ScoreKey, scoring.RowScore] = {}
    transposed: dict[ScoreKey, scoring.RowScore] = {}
    scenarios = tuple(dict.fromkeys(scenario for scenario, _ in TRANSPOSE_PAIRS))
    for scenario in scenarios:
        case = cases.get(scenario)
        if case is None:
            raise ControlsV22Error("transpose runner is missing a required supplied case")
        for row in range(calibration.SYNTHETIC_ROWS):
            key = (scenario, row)
            native[key] = scoring.score_row(case, row, config_sha256=checked_config)
            transposed[key] = scoring.score_transposed_row(case, row, config_sha256=checked_config)
    return assemble_transpose_panel(native, transposed, config_sha256=checked_config)


def validate_transpose_panel(
    evidence: TransposePanelEvidence,
    cases: Mapping[synthetic.Scenario, synthetic.SyntheticCase],
) -> None:
    """Deeply regenerate all 48 row scores and all 108 comparisons."""

    if not isinstance(evidence, TransposePanelEvidence):
        raise ControlsV22Error("transpose-panel validation requires TransposePanelEvidence")
    rebuilt = run_transpose_panel(cases, config_sha256=evidence.config_sha256)
    if not _scientific_equal(evidence, rebuilt):
        raise ControlsV22Error("transpose panel differs from complete scientific replay")


@dataclass(frozen=True, slots=True)
class ScientificControlsEvidence:
    oracle_panel: OraclePanelEvidence
    mutation_panel: MutationPanelEvidence
    transpose_panel: TransposePanelEvidence
    coverage: ControlCoverage

    def __post_init__(self) -> None:
        if (
            type(self.oracle_panel) is not OraclePanelEvidence
            or type(self.mutation_panel) is not MutationPanelEvidence
            or type(self.transpose_panel) is not TransposePanelEvidence
        ):
            raise ControlsV22Error("complete controls evidence contains an invalid panel")
        if self.coverage != CONTROL_COVERAGE:
            raise ControlsV22Error("complete controls evidence has the wrong coverage")
        if not (
            self.oracle_panel.config_sha256 == self.mutation_panel.config_sha256 == self.transpose_panel.config_sha256
        ):
            raise ControlsV22Error("complete controls evidence mixes config scopes")


__all__ = [
    "CONTROL_COVERAGE",
    "EXPECTED_COVERAGE_KEYS",
    "MUTATION_DELTA",
    "MUTATION_PAIRS",
    "MUTATION_SPECS",
    "ORACLE_SPECS",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "TRANSPOSE_PAIRS",
    "TRANSPOSE_SPECS",
    "TRUE_CONTEXTS",
    "ControlCoverage",
    "ControlsV22Error",
    "MutationComparisonEvidence",
    "MutationControlSpec",
    "MutationDiagnostics",
    "MutationPanelEvidence",
    "OracleComparisonDiagnostics",
    "OracleComparisonEvidence",
    "OracleControlSpec",
    "OraclePanelEvidence",
    "ScientificControlsEvidence",
    "TransposeComparisonEvidence",
    "TransposeControlSpec",
    "TransposeDiagnostics",
    "TransposePanelEvidence",
    "assemble_oracle_panel",
    "assemble_selective_transpose_panel",
    "assemble_transpose_panel",
    "compare_oracle_context",
    "compare_selective_transpose_context",
    "compare_transpose_context",
    "mutate_held_target",
    "run_mutation_control",
    "run_mutation_panel",
    "run_oracle_panel",
    "run_transpose_panel",
    "validate_mutation_panel",
    "validate_oracle_panel",
    "validate_selective_transpose_panel",
    "validate_transpose_panel",
]
