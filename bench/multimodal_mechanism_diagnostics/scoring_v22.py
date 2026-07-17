"""Strict one-row synthetic scoring orchestration for MM-008 v2.2.

This module accepts one already-generated and independently replay-validated
``SyntheticCase``.  It owns no RNG, filesystem, lifecycle, candidate subset, cache
injection, scoring-target override, or target-selected prediction envelope.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import nongrid_v22 as nongrid
from bench.multimodal_mechanism_diagnostics import sentinel_v22 as sentinel
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-scoring-v1"

for _dependency in (calibration, fitting, geometry, exact, nongrid, sentinel, synthetic):
    if _dependency.PROTOCOL_SHA256 != PROTOCOL_SHA256:
        raise RuntimeError("MM-008 v2.2 scoring dependency binds a different protocol")

Arm = calibration.SupportArm
ContextName = Literal[
    "true_full",
    "true_p0",
    "true_p1",
    "near_p0",
    "near_p1",
    "far_p0",
    "far_p1",
]
TargetKind = Literal["true", "near", "far"]
StreamName = Literal["full", "p0", "p1"]
InputOrientation = Literal["native", "transposed"]
Estimate: TypeAlias = (
    sentinel.SentinelEstimate | exact.GlobalResult | nongrid.AppearanceEstimate
)

ARM_ORDER: Final[tuple[Arm, ...]] = (
    "global_translation",
    "quadrant_translation",
    "affine",
    "appearance",
    "combined",
)
CONTEXT_ORDER: Final[tuple[ContextName, ...]] = (
    "true_full",
    "true_p0",
    "true_p1",
    "near_p0",
    "near_p1",
    "far_p0",
    "far_p1",
)
GRID_ARMS: Final[tuple[exact.Arm, ...]] = ("affine", "combined")
SCENARIO_ARMS: Final[Mapping[synthetic.Scenario, tuple[Arm, ...]]] = MappingProxyType(
    {
        "translation": ARM_ORDER,
        "affine": ARM_ORDER,
        "appearance": ARM_ORDER,
        "combined": ARM_ORDER,
        "stationary": ARM_ORDER,
        "independent": ARM_ORDER,
        "coupled_boundary": ("affine", "combined"),
        "constant_target": ("appearance", "combined"),
    }
)
SCENARIO_GRID_ARMS: Final[
    Mapping[synthetic.Scenario, tuple[exact.Arm, ...]]
] = MappingProxyType(
    {
        "translation": GRID_ARMS,
        "affine": GRID_ARMS,
        "appearance": GRID_ARMS,
        "combined": GRID_ARMS,
        "stationary": GRID_ARMS,
        "independent": GRID_ARMS,
        "coupled_boundary": GRID_ARMS,
        "constant_target": ("combined",),
    }
)
_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT64_LE: Final = np.dtype("<f8")
_INT64_LE: Final = np.dtype("<i8")


class ScoringV22Error(ValueError):
    """Raised when one-row scoring inputs or evidence fail closed."""


@dataclass(frozen=True, slots=True)
class CoverageCounts:
    scenario_arm_banks: int
    row_arms: int
    fitted_contexts: int
    grid_contexts: int
    bias_contexts: int
    persistence_records: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (
            self.scenario_arm_banks,
            self.row_arms,
            self.fitted_contexts,
            self.grid_contexts,
            self.bias_contexts,
            self.persistence_records,
        )):
            raise ScoringV22Error("ordinary coverage counts must be positive built-in integers")


ORDINARY_COVERAGE: Final = CoverageCounts(
    scenario_arm_banks=sum(len(arms) for arms in SCENARIO_ARMS.values()),
    row_arms=sum(len(arms) for arms in SCENARIO_ARMS.values()) * calibration.SYNTHETIC_ROWS,
    fitted_contexts=sum(len(arms) for arms in SCENARIO_ARMS.values())
    * calibration.SYNTHETIC_ROWS
    * len(CONTEXT_ORDER),
    grid_contexts=sum(len(arms) for arms in SCENARIO_GRID_ARMS.values())
    * calibration.SYNTHETIC_ROWS
    * len(CONTEXT_ORDER),
    bias_contexts=len(synthetic.SCENARIOS) * calibration.SYNTHETIC_ROWS * len(CONTEXT_ORDER),
    persistence_records=len(synthetic.SCENARIOS) * calibration.SYNTHETIC_ROWS,
)
if ORDINARY_COVERAGE != CoverageCounts(34, 204, 1_428, 630, 336, 48):
    raise RuntimeError("MM-008 v2.2 ordinary scoring coverage differs from the frozen census")


def scenario_arms(scenario: synthetic.Scenario) -> tuple[Arm, ...]:
    """Return the exact ordered fitted-arm membership for one ordinary scenario."""

    if scenario not in synthetic.SCENARIOS:
        raise ScoringV22Error("unknown synthetic scenario arm scope")
    return SCENARIO_ARMS[scenario]


def _require_config_sha256(value: str) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise ScoringV22Error("raw config SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _immutable_float64(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _FLOAT64_LE or not bool(np.all(np.isfinite(array))):
        raise ScoringV22Error("scientific evidence must be finite little-endian float64")
    contiguous = np.ascontiguousarray(array, dtype=_FLOAT64_LE)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=_FLOAT64_LE).reshape(contiguous.shape)


def _immutable_bool(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (geometry.SITE_COUNT,)
        or value.dtype != np.dtype(np.bool_)
        or not value.flags.c_contiguous
    ):
        raise ScoringV22Error("context mask must be C-contiguous bool [2304]")
    payload = np.asarray(value, dtype=np.uint8, order="C").tobytes(order="C")
    return np.frombuffer(payload, dtype=np.bool_)


def _array_bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _prediction(estimate: Estimate) -> np.ndarray:
    return cast(np.ndarray, estimate.prediction)


@dataclass(frozen=True, slots=True)
class ContextPlan:
    """One frozen target-bounded fit/output direction."""

    name: ContextName
    target_kind: TargetKind
    target_row: int
    fit_mask: np.ndarray
    output_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.name not in CONTEXT_ORDER:
            raise ScoringV22Error("context name is outside the frozen order")
        if self.target_kind not in {"true", "near", "far"}:
            raise ScoringV22Error("context target kind is invalid")
        if type(self.target_row) is not int or not 0 <= self.target_row < calibration.SYNTHETIC_ROWS:
            raise ScoringV22Error("context target row is outside [0,5]")
        expected_kind = self.name.split("_", maxsplit=1)[0]
        if expected_kind != self.target_kind:
            raise ScoringV22Error("context target kind differs from its name")
        fitted = _immutable_bool(self.fit_mask)
        output = _immutable_bool(self.output_mask)
        if self.name == "true_full":
            valid_masks = np.array_equal(fitted, geometry.FULL_MASK) and np.array_equal(
                output, geometry.FULL_MASK
            )
        else:
            output_parity = 0 if self.name.endswith("p0") else 1
            valid_masks = np.array_equal(
                fitted, geometry.PARITY_MASKS[1 - output_parity]
            ) and np.array_equal(output, geometry.PARITY_MASKS[output_parity])
        if not valid_masks:
            raise ScoringV22Error("context masks differ from the frozen full/checkerboard direction")
        object.__setattr__(self, "fit_mask", fitted)
        object.__setattr__(self, "output_mask", output)


@dataclass(frozen=True, slots=True)
class NamedEndpoint:
    name: str
    record: calibration.EndpointRecord

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or not self.name.isascii():
            raise ScoringV22Error("endpoint name must be nonempty ASCII")
        if not isinstance(self.record, calibration.EndpointRecord):
            raise ScoringV22Error("endpoint record has the wrong type")


@dataclass(frozen=True, slots=True)
class CarryEvidence:
    """Both immutable nesting witnesses owned by one combined objective scope."""

    owner_scope_sha256: str
    affine_parameters: np.ndarray
    affine_gains: np.ndarray
    affine_biases: np.ndarray
    affine_prediction: np.ndarray
    appearance_parameters: np.ndarray
    appearance_gains: np.ndarray
    appearance_biases: np.ndarray
    appearance_retained_macro_ids: tuple[int, ...]
    appearance_prediction: np.ndarray
    hashes: tuple[tuple[str, str], ...]
    affine_error: calibration.ErrorRecord
    appearance_error: calibration.ErrorRecord

    def __post_init__(self) -> None:
        _require_config_sha256(self.owner_scope_sha256)
        shapes = {
            "affine_parameters": (6,),
            "affine_gains": (3,),
            "affine_biases": (3,),
            "appearance_parameters": (6,),
            "appearance_gains": (3,),
            "appearance_biases": (3,),
        }
        for name, shape in shapes.items():
            value = _immutable_float64(getattr(self, name))
            if value.shape != shape:
                raise ScoringV22Error(f"{name} has the wrong shape")
            object.__setattr__(self, name, value)
        for name in ("affine_prediction", "appearance_prediction"):
            value = _immutable_float64(getattr(self, name))
            if value.ndim != 2 or value.shape[0] != geometry.CHANNELS:
                raise ScoringV22Error("carry prediction has the wrong shape")
            object.__setattr__(self, name, value)
        retained = self.appearance_retained_macro_ids
        if (
            type(retained) is not tuple
            or len(retained) not in (14, 27)
            or tuple(sorted(set(retained))) != retained
        ):
            raise ScoringV22Error("appearance carry retained IDs are invalid")
        if tuple(name for name, _ in self.hashes) != tuple(sorted(name for name, _ in self.hashes)):
            raise ScoringV22Error("carry hash roles are not sorted")
        if any(_LOWER_HEX_64.fullmatch(digest) is None for _, digest in self.hashes):
            raise ScoringV22Error("carry hash is invalid")
        if not isinstance(self.affine_error, calibration.ErrorRecord) or not isinstance(
            self.appearance_error, calibration.ErrorRecord
        ):
            raise ScoringV22Error("carry errors are invalid")


@dataclass(frozen=True, slots=True)
class ArmContextScore:
    plan: ContextPlan
    arm: Arm
    estimate: Estimate
    error: calibration.ErrorRecord
    endpoints: tuple[NamedEndpoint, ...]
    carries: CarryEvidence | None

    def __post_init__(self) -> None:
        checked_arm = calibration.require_support_arm(self.arm)
        object.__setattr__(self, "arm", checked_arm)
        valid_type = (
            checked_arm in {"global_translation", "quadrant_translation"}
            and isinstance(self.estimate, sentinel.SentinelEstimate)
            or checked_arm in {"affine", "combined"}
            and isinstance(self.estimate, exact.GlobalResult)
            or checked_arm == "appearance"
            and isinstance(self.estimate, nongrid.AppearanceEstimate)
        )
        if not valid_type:
            raise ScoringV22Error("arm context estimate type does not match its arm")
        if isinstance(self.estimate, (sentinel.SentinelEstimate, exact.GlobalResult)):
            if self.estimate.arm != checked_arm:
                raise ScoringV22Error("arm context estimate declares another arm")
        expected_count = geometry.CHANNELS * int(np.count_nonzero(self.plan.output_mask))
        if self.error.count != expected_count:
            raise ScoringV22Error("arm context error count differs from its output mask")
        if self.plan.target_kind != "true" and self.endpoints:
            raise ScoringV22Error("wrong-target context must not claim injected endpoints")
        if (checked_arm == "combined") != (self.carries is not None):
            raise ScoringV22Error("combined carry membership differs from its arm")


@dataclass(frozen=True, slots=True)
class BiasContextScore:
    plan: ContextPlan
    estimate: nongrid.BiasOnlyEstimate
    error: calibration.ErrorRecord

    def __post_init__(self) -> None:
        if not isinstance(self.estimate, nongrid.BiasOnlyEstimate):
            raise ScoringV22Error("bias context estimate is invalid")
        expected_count = geometry.CHANNELS * int(np.count_nonzero(self.plan.output_mask))
        if self.error.count != expected_count:
            raise ScoringV22Error("bias context error count differs from its output mask")


@dataclass(frozen=True, slots=True)
class BiasScores:
    contexts: tuple[BiasContextScore, ...]
    true_full: calibration.ErrorRecord
    true_xfit: calibration.ErrorRecord
    near_xfit: calibration.ErrorRecord
    far_xfit: calibration.ErrorRecord

    def __post_init__(self) -> None:
        if tuple(item.plan.name for item in self.contexts) != CONTEXT_ORDER:
            raise ScoringV22Error("bias contexts differ from the frozen seven-context order")
        for record in (self.true_full, self.true_xfit, self.near_xfit, self.far_xfit):
            if record.count != geometry.CHANNELS * geometry.SITE_COUNT:
                raise ScoringV22Error("aggregate bias record count differs from 6,912")


@dataclass(frozen=True, slots=True)
class CarryScores:
    affine_full: calibration.ErrorRecord
    affine_xfit: calibration.ErrorRecord
    appearance_full: calibration.ErrorRecord
    appearance_xfit: calibration.ErrorRecord

    def __post_init__(self) -> None:
        for record in (
            self.affine_full,
            self.affine_xfit,
            self.appearance_full,
            self.appearance_xfit,
        ):
            if record.count != geometry.CHANNELS * geometry.SITE_COUNT:
                raise ScoringV22Error("aggregate carry record count differs from 6,912")


@dataclass(frozen=True, slots=True)
class SupportPredicates:
    pair: bool
    performance: bool
    beats_bias: bool
    complete: bool
    strong: bool
    no_bias_gain: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in (
            self.pair,
            self.performance,
            self.beats_bias,
            self.complete,
            self.strong,
            self.no_bias_gain,
        )):
            raise ScoringV22Error("support predicates must be built-in booleans")


@dataclass(frozen=True, slots=True)
class ArmScores:
    arm: Arm
    contexts: tuple[ArmContextScore, ...]
    persistence: calibration.ErrorRecord
    true_full: calibration.ErrorRecord
    true_xfit: calibration.ErrorRecord
    near_xfit: calibration.ErrorRecord
    far_xfit: calibration.ErrorRecord
    endpoints: tuple[NamedEndpoint, ...]
    endpoints_pass: bool
    true_prediction_bit_exact: bool
    predicates: SupportPredicates
    carries: CarryScores | None

    def __post_init__(self) -> None:
        checked_arm = calibration.require_support_arm(self.arm)
        if tuple(item.plan.name for item in self.contexts) != CONTEXT_ORDER:
            raise ScoringV22Error("arm contexts differ from the frozen seven-context order")
        if any(item.arm != checked_arm for item in self.contexts):
            raise ScoringV22Error("arm context membership differs")
        for record in (
            self.persistence,
            self.true_full,
            self.true_xfit,
            self.near_xfit,
            self.far_xfit,
        ):
            if record.count != geometry.CHANNELS * geometry.SITE_COUNT:
                raise ScoringV22Error("aggregate arm record count differs from 6,912")
        if self.endpoints != tuple(
            endpoint for context in self.contexts[:3] for endpoint in context.endpoints
        ):
            raise ScoringV22Error("aggregate endpoints differ from true full/p0/p1 contexts")
        expected_endpoint_pass = bool(self.endpoints) and all(
            endpoint.record.passes() for endpoint in self.endpoints
        )
        if self.endpoints_pass is not expected_endpoint_pass:
            raise ScoringV22Error("endpoint pass summary is not recomputable")
        if type(self.true_prediction_bit_exact) is not bool:
            raise ScoringV22Error("prediction bit-equality flag must be a built-in boolean")
        if (checked_arm == "combined") != (self.carries is not None):
            raise ScoringV22Error("aggregate carry membership differs from its arm")


@dataclass(frozen=True, slots=True)
class GridStreamEvidence:
    stream: StreamName
    grid_arms: tuple[exact.Arm, ...]
    consumer_keys: tuple[str, ...]
    source_grid: exact.SourceGridRecord

    def __post_init__(self) -> None:
        if self.stream not in {"full", "p0", "p1"}:
            raise ScoringV22Error("grid stream name is invalid")
        if self.grid_arms not in {("affine", "combined"), ("combined",)}:
            raise ScoringV22Error("grid stream arm membership is invalid")
        expected_count = len(self.grid_arms) * (1 if self.stream == "full" else 3)
        if len(self.consumer_keys) != expected_count or len(set(self.consumer_keys)) != expected_count:
            raise ScoringV22Error("grid stream consumer membership is invalid")
        if not isinstance(self.source_grid, exact.SourceGridRecord):
            raise ScoringV22Error("grid stream record is invalid")


@dataclass(frozen=True, slots=True)
class DominanceRecord:
    preferred: Arm
    comparator: Arm
    passed: bool

    def __post_init__(self) -> None:
        calibration.require_support_arm(self.preferred)
        calibration.require_support_arm(self.comparator)
        if self.preferred == self.comparator or type(self.passed) is not bool:
            raise ScoringV22Error("dominance record is invalid")


@dataclass(frozen=True, slots=True)
class ExpectationCheck:
    name: str
    passed: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or not self.name.isascii():
            raise ScoringV22Error("expectation name must be nonempty ASCII")
        if type(self.passed) is not bool:
            raise ScoringV22Error("expectation result must be a built-in boolean")


@dataclass(frozen=True, slots=True)
class RowScore:
    scenario: synthetic.Scenario
    seed: int
    row: int
    config_sha256: str
    input_orientation: InputOrientation
    persistence_estimate: nongrid.PersistenceEstimate
    persistence: calibration.ErrorRecord
    bias: BiasScores
    arms: tuple[ArmScores, ...]
    grid_streams: tuple[GridStreamEvidence, ...]
    dominance: tuple[DominanceRecord, ...]
    expectations: tuple[ExpectationCheck, ...]

    def __post_init__(self) -> None:
        if self.scenario not in synthetic.SCENARIOS:
            raise ScoringV22Error("row score scenario is invalid")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ScoringV22Error("row score seed is invalid")
        if type(self.row) is not int or not 0 <= self.row < calibration.SYNTHETIC_ROWS:
            raise ScoringV22Error("row score index is invalid")
        _require_config_sha256(self.config_sha256)
        if self.input_orientation not in {"native", "transposed"}:
            raise ScoringV22Error("row score input orientation is invalid")
        expected_arms = scenario_arms(self.scenario)
        expected_grid_arms = SCENARIO_GRID_ARMS[self.scenario]
        if tuple(item.arm for item in self.arms) != expected_arms:
            raise ScoringV22Error("row score arms differ from the frozen order")
        if tuple(item.stream for item in self.grid_streams) != ("full", "p0", "p1"):
            raise ScoringV22Error("row score does not contain exactly three ordered grid streams")
        if any(item.grid_arms != expected_grid_arms for item in self.grid_streams):
            raise ScoringV22Error("row score grid streams differ from scenario arm scope")
        expected_dominance = tuple(
            (preferred, comparator)
            for preferred in expected_arms
            for comparator in expected_arms
            if preferred != comparator
        )
        if tuple((item.preferred, item.comparator) for item in self.dominance) != expected_dominance:
            raise ScoringV22Error("row score dominance panel is incomplete")
        if len({check.name for check in self.expectations}) != len(self.expectations):
            raise ScoringV22Error("row score expectation names are duplicated")

    @property
    def expectation_failures(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.expectations if not check.passed)

    @property
    def expectations_pass(self) -> bool:
        return not self.expectation_failures

    def arm(self, name: Arm) -> ArmScores:
        checked = calibration.require_support_arm(name)
        return next(item for item in self.arms if item.arm == checked)

    def dominates(self, preferred: Arm, comparator: Arm) -> bool:
        for record in self.dominance:
            if record.preferred == preferred and record.comparator == comparator:
                return record.passed
        raise ScoringV22Error("dominance lookup requires two distinct known arms")


@dataclass(frozen=True, slots=True)
class _ContextWork:
    plan: ContextPlan
    fit_target: np.ndarray
    score_target: np.ndarray


def _context_key(
    case: synthetic.SyntheticCase,
    row: int,
    input_orientation: InputOrientation,
    arm: Literal["affine", "combined"],
    context: ContextName,
) -> str:
    return (
        f"synthetic/{case.scenario}/seed-{case.seed}/row-{row}/"
        f"{input_orientation}/{arm}/{context}"
    )


def _plans(bundle: synthetic.SyntheticRowTargets) -> tuple[ContextPlan, ...]:
    return (
        ContextPlan("true_full", "true", bundle.row, geometry.FULL_MASK, geometry.FULL_MASK),
        ContextPlan("true_p0", "true", bundle.row, geometry.PARITY_MASKS[1], geometry.PARITY_MASKS[0]),
        ContextPlan("true_p1", "true", bundle.row, geometry.PARITY_MASKS[0], geometry.PARITY_MASKS[1]),
        ContextPlan("near_p0", "near", bundle.near_row, geometry.PARITY_MASKS[1], geometry.PARITY_MASKS[0]),
        ContextPlan("near_p1", "near", bundle.near_row, geometry.PARITY_MASKS[0], geometry.PARITY_MASKS[1]),
        ContextPlan("far_p0", "far", bundle.far_row, geometry.PARITY_MASKS[1], geometry.PARITY_MASKS[0]),
        ContextPlan("far_p1", "far", bundle.far_row, geometry.PARITY_MASKS[0], geometry.PARITY_MASKS[1]),
    )


def _target_for_kind(bundle: synthetic.SyntheticRowTargets, kind: TargetKind) -> np.ndarray:
    if kind == "true":
        return bundle.true_target
    if kind == "near":
        return bundle.near_target
    return bundle.far_target


def _work_items(bundle: synthetic.SyntheticRowTargets) -> tuple[_ContextWork, ...]:
    items: list[_ContextWork] = []
    for plan in _plans(bundle):
        fitted_target = fitting.target_values(_target_for_kind(bundle, plan.target_kind), plan.fit_mask)
        scoring_target = fitting.target_values(bundle.true_target, plan.output_mask)
        items.append(_ContextWork(plan, fitted_target, scoring_target))
    return tuple(items)


def _grid_fit_stream(
    case: synthetic.SyntheticCase,
    row: int,
    input_orientation: InputOrientation,
    source: np.ndarray,
    selected: tuple[_ContextWork, ...],
    stream: StreamName,
    grid_arms: tuple[exact.Arm, ...],
    *,
    config_sha256: str,
) -> tuple[dict[tuple[Arm, ContextName], exact.GlobalResult], GridStreamEvidence]:
    request_members = tuple(
        (cast(Arm, arm), item)
        for arm in grid_arms
        for item in selected
    )
    requests = tuple(
        exact.FitRequest.create(
            _context_key(
                case,
                row,
                input_orientation,
                cast(Literal["affine", "combined"], arm),
                item.plan.name,
            ),
            cast(exact.Arm, arm),
            item.fit_target,
        )
        for arm, item in request_members
    )
    results = exact.fit_global_contexts(
        source,
        selected[0].plan.fit_mask,
        selected[0].plan.output_mask,
        requests,
        config_sha256=config_sha256,
    )
    source_grid = results[0].source_grid
    if any(result.source_grid is not source_grid for result in results):
        raise ScoringV22Error("grid consumers did not share one source-stream record")
    mapped = {
        (arm, item.plan.name): result
        for (arm, item), result in zip(request_members, results, strict=True)
    }
    evidence = GridStreamEvidence(
        stream,
        grid_arms,
        tuple(request.context_key for request in requests),
        source_grid,
    )
    return mapped, evidence


def _named_endpoints(
    plan: ContextPlan,
    arm: Arm,
    estimate: Estimate,
    truth: synthetic.TransformTruth | None,
) -> tuple[NamedEndpoint, ...]:
    if plan.target_kind != "true" or truth is None:
        return ()
    records: list[NamedEndpoint] = []

    def add(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
        records.append(
            NamedEndpoint(
                f"{plan.name}:{name}",
                calibration.endpoint_record(actual, expected),
            )
        )

    if arm in {"global_translation", "quadrant_translation"}:
        sentinel_result = cast(sentinel.SentinelEstimate, estimate)
        expected_flow = np.broadcast_to(truth.theta_array()[:2], sentinel_result.flow.shape)
        add("flow", sentinel_result.flow, expected_flow)
    elif arm == "affine":
        result = cast(exact.GlobalResult, estimate)
        add("theta", result.selected.parameters, truth.theta_array())
    elif arm == "appearance":
        appearance_result = cast(nongrid.AppearanceEstimate, estimate)
        add("theta", appearance_result.parameters, np.zeros(6, dtype=_FLOAT64_LE))
        add("gain", appearance_result.gains, truth.gain_array())
        add("bias", appearance_result.biases, truth.bias_array())
    else:
        result = cast(exact.GlobalResult, estimate)
        if result.selected.gains is None or result.selected.biases is None:
            raise ScoringV22Error("combined selection lacks appearance endpoints")
        add("theta", result.selected.parameters, truth.theta_array())
        add("gain", result.selected.gains, truth.gain_array())
        add("bias", result.selected.biases, truth.bias_array())
    return tuple(records)


def _carry_evidence(
    source: np.ndarray,
    plan: ContextPlan,
    result: exact.GlobalResult,
    fit_target: np.ndarray,
    score_target: np.ndarray,
    *,
    config_sha256: str,
) -> CarryEvidence:
    owner = result.objective_cache.scope_sha256
    appearance = nongrid.fit_appearance(
        source,
        fit_target,
        plan.fit_mask,
        plan.output_mask,
        config_sha256=config_sha256,
    )
    affine_parameters = result.selected.parameters
    affine_gains = np.ones(geometry.CHANNELS, dtype=_FLOAT64_LE)
    affine_biases = np.zeros(geometry.CHANNELS, dtype=_FLOAT64_LE)
    affine_prediction = geometry.sample_scalar(source, result.selected.state_index, plan.output_mask)
    values = {
        "affine_carry_biases": affine_biases,
        "affine_carry_gains": affine_gains,
        "affine_carry_parameters": affine_parameters,
        "affine_carry_prediction": affine_prediction,
        "appearance_carry_biases": appearance.biases,
        "appearance_carry_gains": appearance.gains,
        "appearance_carry_parameters": appearance.parameters,
        "appearance_carry_prediction": appearance.prediction,
        "appearance_carry_retained_ids": np.asarray(
            appearance.retained_macro_ids, dtype=_INT64_LE
        ),
    }
    hashes = tuple(
        (role, nongrid.array_sha256(owner, role, value))
        for role, value in sorted(values.items())
    )
    return CarryEvidence(
        owner_scope_sha256=owner,
        affine_parameters=affine_parameters,
        affine_gains=affine_gains,
        affine_biases=affine_biases,
        affine_prediction=affine_prediction,
        appearance_parameters=appearance.parameters,
        appearance_gains=appearance.gains,
        appearance_biases=appearance.biases,
        appearance_retained_macro_ids=appearance.retained_macro_ids,
        appearance_prediction=appearance.prediction,
        hashes=hashes,
        affine_error=calibration.error_record(affine_prediction, score_target),
        appearance_error=calibration.error_record(appearance.prediction, score_target),
    )


def _assemble_pair(
    first: np.ndarray,
    first_mask: np.ndarray,
    second: np.ndarray,
    second_mask: np.ndarray,
) -> np.ndarray:
    assembled = np.empty((geometry.CHANNELS, geometry.SITE_COUNT), dtype=_FLOAT64_LE)
    assembled[:, first_mask] = first
    assembled[:, second_mask] = second
    return _immutable_float64(assembled)


def _aggregate_xfit(
    first: np.ndarray,
    second: np.ndarray,
    true_full_target: np.ndarray,
) -> calibration.ErrorRecord:
    prediction = _assemble_pair(
        first,
        geometry.PARITY_MASKS[0],
        second,
        geometry.PARITY_MASKS[1],
    )
    return calibration.error_record(prediction, true_full_target)


def _bias_scores(
    contexts: tuple[BiasContextScore, ...],
    true_full_target: np.ndarray,
) -> BiasScores:
    by_name = {item.plan.name: item for item in contexts}
    return BiasScores(
        contexts=contexts,
        true_full=by_name["true_full"].error,
        true_xfit=_aggregate_xfit(
            by_name["true_p0"].estimate.prediction,
            by_name["true_p1"].estimate.prediction,
            true_full_target,
        ),
        near_xfit=_aggregate_xfit(
            by_name["near_p0"].estimate.prediction,
            by_name["near_p1"].estimate.prediction,
            true_full_target,
        ),
        far_xfit=_aggregate_xfit(
            by_name["far_p0"].estimate.prediction,
            by_name["far_p1"].estimate.prediction,
            true_full_target,
        ),
    )


def _arm_scores(
    arm: Arm,
    contexts: tuple[ArmContextScore, ...],
    persistence: calibration.ErrorRecord,
    bias: BiasScores,
    true_full_target: np.ndarray,
) -> ArmScores:
    by_name = {item.plan.name: item for item in contexts}
    true_xfit = _aggregate_xfit(
        _prediction(by_name["true_p0"].estimate),
        _prediction(by_name["true_p1"].estimate),
        true_full_target,
    )
    near_xfit = _aggregate_xfit(
        _prediction(by_name["near_p0"].estimate),
        _prediction(by_name["near_p1"].estimate),
        true_full_target,
    )
    far_xfit = _aggregate_xfit(
        _prediction(by_name["far_p0"].estimate),
        _prediction(by_name["far_p1"].estimate),
        true_full_target,
    )
    endpoints = tuple(endpoint for context in contexts[:3] for endpoint in context.endpoints)
    endpoint_pass = bool(endpoints) and all(endpoint.record.passes() for endpoint in endpoints)
    checked_arm = calibration.require_support_arm(arm)
    pair = calibration.pair_support(
        true_xfit.mse,
        near_xfit.mse,
        far_xfit.mse,
        bias.true_xfit.mse,
        bias.near_xfit.mse,
        bias.far_xfit.mse,
    )
    performance = calibration.performance_support(persistence.mse, true_xfit.mse)
    beats = calibration.beats_bias(true_xfit.mse, bias.true_xfit.mse)
    complete = calibration.complete_support(
        checked_arm,
        persistence.mse,
        true_xfit.mse,
        near_xfit.mse,
        far_xfit.mse,
        bias.true_xfit.mse,
        bias.near_xfit.mse,
        bias.far_xfit.mse,
    )
    strong = calibration.strong_support(
        checked_arm,
        persistence.mse,
        by_name["true_full"].error.mse,
        true_xfit.mse,
        near_xfit.mse,
        far_xfit.mse,
        bias.true_xfit.mse,
        bias.near_xfit.mse,
        bias.far_xfit.mse,
        endpoints_pass=endpoint_pass,
    )
    true_assembled = _assemble_pair(
        _prediction(by_name["true_p0"].estimate),
        geometry.PARITY_MASKS[0],
        _prediction(by_name["true_p1"].estimate),
        geometry.PARITY_MASKS[1],
    )
    true_exact = _array_bits_equal(
        _prediction(by_name["true_full"].estimate), true_full_target
    ) and _array_bits_equal(true_assembled, true_full_target)

    carry_scores: CarryScores | None = None
    if arm == "combined":
        carry_by_name = {item.plan.name: item.carries for item in contexts}
        if any(value is None for value in carry_by_name.values()):
            raise ScoringV22Error("combined arm lacks carry evidence")
        full = cast(CarryEvidence, carry_by_name["true_full"])
        p0 = cast(CarryEvidence, carry_by_name["true_p0"])
        p1 = cast(CarryEvidence, carry_by_name["true_p1"])
        carry_scores = CarryScores(
            affine_full=full.affine_error,
            affine_xfit=_aggregate_xfit(
                p0.affine_prediction, p1.affine_prediction, true_full_target
            ),
            appearance_full=full.appearance_error,
            appearance_xfit=_aggregate_xfit(
                p0.appearance_prediction, p1.appearance_prediction, true_full_target
            ),
        )

    return ArmScores(
        arm=checked_arm,
        contexts=contexts,
        persistence=persistence,
        true_full=by_name["true_full"].error,
        true_xfit=true_xfit,
        near_xfit=near_xfit,
        far_xfit=far_xfit,
        endpoints=endpoints,
        endpoints_pass=endpoint_pass,
        true_prediction_bit_exact=true_exact,
        predicates=SupportPredicates(
            pair=pair,
            performance=performance,
            beats_bias=beats,
            complete=complete,
            strong=strong,
            no_bias_gain=calibration.no_bias_gain(true_xfit.mse, bias.true_xfit.mse),
        ),
        carries=carry_scores,
    )


def _dominance(arms: tuple[ArmScores, ...]) -> tuple[DominanceRecord, ...]:
    records: list[DominanceRecord] = []
    for preferred in arms:
        for comparator in arms:
            if preferred.arm == comparator.arm:
                continue
            records.append(
                DominanceRecord(
                    preferred.arm,
                    comparator.arm,
                    calibration.dominates(
                        preferred.true_full.mse,
                        comparator.true_full.mse,
                        preferred.true_xfit.mse,
                        comparator.true_xfit.mse,
                    ),
                )
            )
    return tuple(records)


_STRONG_ARMS: Final[dict[synthetic.Scenario, tuple[Arm, ...]]] = {
    "translation": ("global_translation", "quadrant_translation", "affine", "combined"),
    "affine": ("affine", "combined"),
    "appearance": ("appearance", "combined"),
    "combined": ("combined",),
    "stationary": (),
    "independent": (),
    "coupled_boundary": (),
    "constant_target": (),
}
_DOMINANCE_REQUIREMENTS: Final[dict[synthetic.Scenario, tuple[tuple[Arm, Arm], ...]]] = {
    "translation": (("global_translation", "appearance"),),
    "affine": (
        ("affine", "global_translation"),
        ("affine", "quadrant_translation"),
        ("affine", "appearance"),
    ),
    "appearance": (
        ("appearance", "global_translation"),
        ("appearance", "quadrant_translation"),
        ("appearance", "affine"),
    ),
    "combined": (
        ("combined", "global_translation"),
        ("combined", "quadrant_translation"),
        ("combined", "affine"),
        ("combined", "appearance"),
    ),
    "stationary": (),
    "independent": (),
    "coupled_boundary": (),
    "constant_target": (),
}


def _expectations(
    scenario: synthetic.Scenario,
    arms: tuple[ArmScores, ...],
    persistence: calibration.ErrorRecord,
    dominance: tuple[DominanceRecord, ...],
    grid_results: dict[tuple[Arm, ContextName], exact.GlobalResult],
    bias: BiasScores,
) -> tuple[ExpectationCheck, ...]:
    by_arm = {item.arm: item for item in arms}
    by_dom = {(item.preferred, item.comparator): item.passed for item in dominance}
    checks: list[ExpectationCheck] = []

    def add(name: str, passed: bool) -> None:
        checks.append(ExpectationCheck(name, bool(passed)))

    for arm in _STRONG_ARMS[scenario]:
        add(f"{arm}:Strong", by_arm[arm].predicates.strong)
    for preferred, comparator in _DOMINANCE_REQUIREMENTS[scenario]:
        add(f"{preferred}>{comparator}:Dom", by_dom[(preferred, comparator)])

    if scenario in {"translation", "affine", "appearance", "combined"}:
        add("combined:BeatsBias", by_arm["combined"].predicates.beats_bias)
    if scenario == "appearance":
        add("appearance:BeatsBias", by_arm["appearance"].predicates.beats_bias)
    if scenario == "combined":
        carries = by_arm["combined"].carries
        add("combined:carry_evidence", carries is not None)
        if carries is not None:
            add(
                "combined>affine_carry:Dom",
                calibration.dominates(
                    by_arm["combined"].true_full.mse,
                    carries.affine_full.mse,
                    by_arm["combined"].true_xfit.mse,
                    carries.affine_xfit.mse,
                ),
            )
            add(
                "combined>appearance_carry:Dom",
                calibration.dominates(
                    by_arm["combined"].true_full.mse,
                    carries.appearance_full.mse,
                    by_arm["combined"].true_xfit.mse,
                    carries.appearance_xfit.mse,
                ),
            )
    elif scenario == "stationary":
        add("stationary:persistence_zero", persistence.sse == 0.0 and persistence.mse == 0.0)
        for arm in ARM_ORDER:
            score = by_arm[arm]
            add(f"stationary:{arm}:not_Perf", not score.predicates.performance)
            add(f"stationary:{arm}:not_Complete", not score.predicates.complete)
            add(f"stationary:{arm}:not_Strong", not score.predicates.strong)
            add(f"stationary:{arm}:prediction_bits", score.true_prediction_bit_exact)
            add(f"stationary:{arm}:identity_endpoints", score.endpoints_pass)
    elif scenario == "independent":
        for arm in ("global_translation", "quadrant_translation", "affine"):
            add(f"independent:{arm}:not_Complete", not by_arm[arm].predicates.complete)
        for arm in ("appearance", "combined"):
            add(f"independent:{arm}:no_bias_gain", by_arm[arm].predicates.no_bias_gain)
            add(f"independent:{arm}:not_Complete", not by_arm[arm].predicates.complete)
    elif scenario == "constant_target":
        bias_by_context = {item.plan.name: item for item in bias.contexts}
        for arm in ("appearance", "combined"):
            score = by_arm[arm]
            add(
                f"constant_target:{arm}:matches_bias_full",
                abs(score.true_full.mse - bias.true_full.mse) <= 1e-12,
            )
            add(
                f"constant_target:{arm}:matches_bias_xfit",
                abs(score.true_xfit.mse - bias.true_xfit.mse) <= 1e-12,
            )
            add(f"constant_target:{arm}:not_BeatsBias", not score.predicates.beats_bias)
            add(f"constant_target:{arm}:not_Complete", not score.predicates.complete)
        for name in ("true_full", "true_p0", "true_p1"):
            appearance_context = next(
                item for item in by_arm["appearance"].contexts if item.plan.name == name
            )
            appearance_estimate = cast(
                nongrid.AppearanceEstimate, appearance_context.estimate
            )
            combined_result = grid_results[("combined", cast(ContextName, name))]
            combined_gains = combined_result.selected.gains
            combined_biases = combined_result.selected.biases
            bias_estimate = bias_by_context[cast(ContextName, name)].estimate
            add(
                f"constant_target:{name}:combined_identity",
                combined_result.selected.state_index == 0,
            )
            add(
                f"constant_target:{name}:appearance_zero_source",
                np.array_equal(appearance_estimate.parameters, np.zeros(6))
                and float(np.max(np.abs(appearance_estimate.gains))) <= 1e-12
                and float(
                    np.max(np.abs(appearance_estimate.biases - bias_estimate.biases))
                )
                <= 1e-12,
            )
            add(
                f"constant_target:{name}:combined_zero_source",
                combined_gains is not None
                and combined_biases is not None
                and float(np.max(np.abs(combined_gains))) <= 1e-12
                and float(np.max(np.abs(combined_biases - bias_estimate.biases))) <= 1e-12,
            )
    elif scenario == "coupled_boundary":
        for arm in GRID_ARMS:
            score = by_arm[arm]
            true_results = tuple(
                grid_results[(cast(Arm, arm), name)]
                for name in ("true_full", "true_p0", "true_p1")
            )
            add(f"coupled_boundary:{arm}:endpoints", score.endpoints_pass)
            add(
                f"coupled_boundary:{arm}:zero_objective",
                max(result.selected.objective for result in true_results) <= 1e-12,
            )
            add(
                f"coupled_boundary:{arm}:certificates",
                all(result.certificate.scalar_replay_bit_exact for result in true_results),
            )
    return tuple(checks)


def _validated_public_inputs(
    case: synthetic.SyntheticCase,
    row: int,
    *,
    config_sha256: str,
) -> tuple[int, str]:
    """Validate an authoritative generated case before any fit or transformation."""

    checked_config = _require_config_sha256(config_sha256)
    if not isinstance(case, synthetic.SyntheticCase):
        raise ScoringV22Error("row scoring requires a SyntheticCase")
    if isinstance(row, bool) or not isinstance(row, Integral):
        raise ScoringV22Error("score row must be an integer")
    checked_row = int(row)
    if not 0 <= checked_row < calibration.SYNTHETIC_ROWS:
        raise ScoringV22Error("score row is outside [0,5]")
    try:
        synthetic.validate_case(case)
    except synthetic.SyntheticV22Error as error:
        raise ScoringV22Error("synthetic case failed independent replay validation") from error
    return checked_row, checked_config


def _score_validated_case(
    case: synthetic.SyntheticCase,
    row: int,
    *,
    config_sha256: str,
    input_orientation: InputOrientation,
) -> RowScore:
    """Score a trusted native or internally derived transpose without regeneration."""

    checked_config = config_sha256
    arms = scenario_arms(case.scenario)
    grid_arms = SCENARIO_GRID_ARMS[case.scenario]
    bundle = synthetic.row_targets(case, row)
    source = bundle.source
    work = _work_items(bundle)
    by_context = {item.plan.name: item for item in work}
    true_full_target = fitting.target_values(bundle.true_target, geometry.FULL_MASK)

    persistence_estimate = nongrid.persistence(
        source, geometry.FULL_MASK, config_sha256=checked_config
    )
    persistence_score = calibration.error_record(
        persistence_estimate.prediction, true_full_target
    )

    grid_results: dict[tuple[Arm, ContextName], exact.GlobalResult] = {}
    stream_evidence: list[GridStreamEvidence] = []
    stream_members = (
        ("full", (by_context["true_full"],)),
        ("p0", (by_context["true_p0"], by_context["near_p0"], by_context["far_p0"])),
        ("p1", (by_context["true_p1"], by_context["near_p1"], by_context["far_p1"])),
    )
    for stream, members in stream_members:
        mapped, evidence = _grid_fit_stream(
            case,
            row,
            input_orientation,
            source,
            members,
            cast(StreamName, stream),
            grid_arms,
            config_sha256=checked_config,
        )
        grid_results.update(mapped)
        stream_evidence.append(evidence)

    bias_estimates: dict[ContextName, nongrid.BiasOnlyEstimate] = {}
    appearance_estimates: dict[ContextName, nongrid.AppearanceEstimate] = {}
    sentinel_estimates: dict[tuple[Arm, ContextName], sentinel.SentinelEstimate] = {}
    for item in work:
        name = item.plan.name
        bias_estimates[name] = nongrid.fit_bias_only(
            item.fit_target,
            item.plan.fit_mask,
            item.plan.output_mask,
            config_sha256=checked_config,
        )
        if "appearance" in arms:
            appearance_estimates[name] = nongrid.fit_appearance(
                source,
                item.fit_target,
                item.plan.fit_mask,
                item.plan.output_mask,
                config_sha256=checked_config,
            )
        for sentinel_arm in tuple(
            candidate
            for candidate in ("global_translation", "quadrant_translation")
            if candidate in arms
        ):
            sentinel_estimates[(cast(Arm, sentinel_arm), name)] = sentinel.fit_sentinel(
                source,
                item.fit_target,
                item.plan.fit_mask,
                item.plan.output_mask,
                cast(sentinel.SentinelArm, sentinel_arm),
                config_sha256=checked_config,
            )

    bias_contexts = tuple(
        BiasContextScore(
            item.plan,
            bias_estimates[item.plan.name],
            calibration.error_record(
                bias_estimates[item.plan.name].prediction, item.score_target
            ),
        )
        for item in work
    )
    bias_scores = _bias_scores(bias_contexts, true_full_target)

    arm_scores: list[ArmScores] = []
    for arm in arms:
        context_scores: list[ArmContextScore] = []
        for item in work:
            name = item.plan.name
            if arm in {"global_translation", "quadrant_translation"}:
                estimate: Estimate = sentinel_estimates[(arm, name)]
            elif arm in {"affine", "combined"}:
                estimate = grid_results[(arm, name)]
            else:
                estimate = appearance_estimates[name]
            carry = (
                _carry_evidence(
                    source,
                    item.plan,
                    cast(exact.GlobalResult, estimate),
                    item.fit_target,
                    item.score_target,
                    config_sha256=checked_config,
                )
                if arm == "combined"
                else None
            )
            context_scores.append(
                ArmContextScore(
                    plan=item.plan,
                    arm=arm,
                    estimate=estimate,
                    error=calibration.error_record(_prediction(estimate), item.score_target),
                    endpoints=_named_endpoints(item.plan, arm, estimate, case.truth),
                    carries=carry,
                )
            )
        arm_scores.append(
            _arm_scores(
                arm,
                tuple(context_scores),
                persistence_score,
                bias_scores,
                true_full_target,
            )
        )

    ordered_arms = tuple(arm_scores)
    dominance = _dominance(ordered_arms)
    expectations = _expectations(
        case.scenario,
        ordered_arms,
        persistence_score,
        dominance,
        grid_results,
        bias_scores,
    )
    return RowScore(
        scenario=case.scenario,
        seed=case.seed,
        row=row,
        config_sha256=checked_config,
        input_orientation=input_orientation,
        persistence_estimate=persistence_estimate,
        persistence=persistence_score,
        bias=bias_scores,
        arms=ordered_arms,
        grid_streams=tuple(stream_evidence),
        dominance=dominance,
        expectations=expectations,
    )


def score_row(
    case: synthetic.SyntheticCase,
    row: int,
    *,
    config_sha256: str,
) -> RowScore:
    """Fit and score one validated native row through its scenario-scoped arms.

    The synthetic case is deeply regenerated before any estimator runs.  The only
    scoring target is the validated case's true row target; wrong targets are bounded
    to their declared fit masks and never select among predictions.
    """

    checked_row, checked_config = _validated_public_inputs(
        case, row, config_sha256=config_sha256
    )
    return _score_validated_case(
        case,
        checked_row,
        config_sha256=checked_config,
        input_orientation="native",
    )


def score_transposed_row(
    original_case: synthetic.SyntheticCase,
    row: int,
    *,
    config_sha256: str,
) -> RowScore:
    """Validate an original case, derive its transpose internally, and score it.

    A caller cannot supply an authoritative transformed case: validation always
    precedes the internally owned spatial transpose.
    """

    checked_row, checked_config = _validated_public_inputs(
        original_case, row, config_sha256=config_sha256
    )
    transposed_case = synthetic.transpose_case(original_case)
    return _score_validated_case(
        transposed_case,
        checked_row,
        config_sha256=checked_config,
        input_orientation="transposed",
    )


def _scientific_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return _array_bits_equal(left, right)
    if isinstance(left, np.generic) and isinstance(right, np.generic):
        return left.dtype == right.dtype and left.tobytes() == right.tobytes()
    if isinstance(left, float) and isinstance(right, float):
        return struct.pack("<d", left) == struct.pack("<d", right)
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


def validate_row_score(score: RowScore, case: synthetic.SyntheticCase) -> None:
    """Fully rerun one native row and reject any stale or forged nested evidence."""

    if not isinstance(score, RowScore):
        raise ScoringV22Error("row-score validation requires a RowScore")
    if not isinstance(case, synthetic.SyntheticCase):
        raise ScoringV22Error("row-score validation requires a SyntheticCase")
    if score.input_orientation != "native":
        raise ScoringV22Error("native row-score validation rejects transposed evidence")
    if score.scenario != case.scenario or score.seed != case.seed:
        raise ScoringV22Error("row score and synthetic case identities differ")
    regenerated = score_row(case, score.row, config_sha256=score.config_sha256)
    if not _scientific_equal(score, regenerated):
        raise ScoringV22Error("row score differs from complete bit-exact replay")


def validate_transposed_row_score(
    score: RowScore, original_case: synthetic.SyntheticCase
) -> None:
    """Fully rerun one trusted transpose path and reject forged nested evidence."""

    if not isinstance(score, RowScore):
        raise ScoringV22Error("transposed row-score validation requires a RowScore")
    if not isinstance(original_case, synthetic.SyntheticCase):
        raise ScoringV22Error("transposed validation requires an original SyntheticCase")
    if score.input_orientation != "transposed":
        raise ScoringV22Error("transposed row-score validation rejects native evidence")
    if score.scenario != original_case.scenario or score.seed != original_case.seed:
        raise ScoringV22Error("row score and original synthetic case identities differ")
    regenerated = score_transposed_row(
        original_case, score.row, config_sha256=score.config_sha256
    )
    if not _scientific_equal(score, regenerated):
        raise ScoringV22Error("transposed row score differs from complete bit-exact replay")


__all__ = [
    "ARM_ORDER",
    "CONTEXT_ORDER",
    "GRID_ARMS",
    "ORDINARY_COVERAGE",
    "SCENARIO_ARMS",
    "SCENARIO_GRID_ARMS",
    "Arm",
    "ArmContextScore",
    "ArmScores",
    "BiasContextScore",
    "BiasScores",
    "CarryEvidence",
    "CarryScores",
    "ContextName",
    "ContextPlan",
    "CoverageCounts",
    "DominanceRecord",
    "ExpectationCheck",
    "GridStreamEvidence",
    "InputOrientation",
    "PROTOCOL_SHA256",
    "RowScore",
    "SCHEMA_VERSION",
    "ScoringV22Error",
    "SupportPredicates",
    "scenario_arms",
    "score_row",
    "score_transposed_row",
    "validate_row_score",
    "validate_transposed_row_score",
]
