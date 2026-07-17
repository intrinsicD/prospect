"""Selective transposed exact-grid controls for MM-008 v2.2 development.

The normative transpose panel retains only the affine/combined ``true_full``,
``true_p0``, and ``true_p1`` contexts for four scenarios.  Ordinary row scoring
also computes wrong-target, sentinel, appearance, bias, carry, dominance, and
expectation records that never enter those 108 comparisons.  This module owns a
closed path that validates one native synthetic case, derives its transpose
internally, and computes exactly the retained grid contexts with the unchanged
exact-global fitter.

It owns no RNG, filesystem, lifecycle, nonce, reserved-seed, or real-data
behavior.  The output is development scientific-control evidence, not formal
runtime evidence.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import fitting_v22 as fitting
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import global_v22 as exact
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

PROTOCOL_SHA256: Final = "300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622"
SCHEMA_VERSION: Final = "mm008-v2.2-selective-transpose-control-v1"
CLAIM_SCOPE: Final = "exposed-seed-scientific-control-only"

for _dependency in (calibration, fitting, geometry, exact, scoring, synthetic):
    if _dependency.PROTOCOL_SHA256 != PROTOCOL_SHA256:
        raise RuntimeError("MM-008 v2.2 selective-transpose dependency binds another protocol")

TransposeScenario = Literal["translation", "affine", "appearance", "combined"]
TrueContext = Literal["true_full", "true_p0", "true_p1"]

TRANSPOSE_SCENARIOS: Final[tuple[TransposeScenario, ...]] = (
    "translation",
    "affine",
    "appearance",
    "combined",
)
TRUE_CONTEXTS: Final[tuple[TrueContext, ...]] = (
    "true_full",
    "true_p0",
    "true_p1",
)
TRANSPOSE_GRID_ARMS: Final[Mapping[TransposeScenario, tuple[exact.Arm, ...]]] = (
    MappingProxyType(
        {
            "translation": ("affine", "combined"),
            "affine": ("affine", "combined"),
            "appearance": ("combined",),
            "combined": ("combined",),
        }
    )
)

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")


class TransposeV22Error(ValueError):
    """Raised when selective transpose authority or evidence is invalid."""


def _require_config_sha256(value: str) -> str:
    if type(value) is not str or _LOWER_HEX_64.fullmatch(value) is None:
        raise TransposeV22Error("config SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _context_key(
    scenario: synthetic.Scenario,
    seed: int,
    row: int,
    arm: exact.Arm,
    context: TrueContext,
) -> str:
    return (
        f"synthetic/{scenario}/seed-{seed}/row-{row}/"
        f"transposed/{arm}/{context}"
    )


def _scientific_equal(left: object, right: object) -> bool:
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


@dataclass(frozen=True, slots=True)
class TransposedGridContextScore:
    """One retained transposed true-context result and its true-target error."""

    plan: scoring.ContextPlan
    arm: exact.Arm
    result: exact.GlobalResult
    error: calibration.ErrorRecord

    def __post_init__(self) -> None:
        if (
            not isinstance(self.plan, scoring.ContextPlan)
            or self.plan.name not in TRUE_CONTEXTS
            or self.plan.target_kind != "true"
        ):
            raise TransposeV22Error("selective transpose context is not a frozen true context")
        if self.arm not in {"affine", "combined"}:
            raise TransposeV22Error("selective transpose context has an invalid grid arm")
        if not isinstance(self.result, exact.GlobalResult) or self.result.arm != self.arm:
            raise TransposeV22Error("selective transpose result differs from its grid arm")
        if not isinstance(self.error, calibration.ErrorRecord):
            raise TransposeV22Error("selective transpose context has an invalid error record")
        expected_count = geometry.CHANNELS * int(np.count_nonzero(self.plan.output_mask))
        if self.error.count != expected_count:
            raise TransposeV22Error("selective transpose error count differs from its output mask")


@dataclass(frozen=True, slots=True)
class TransposedGridControlRow:
    """Closed retained-grid projection of one internally transposed native row."""

    protocol_sha256: str
    schema_version: str
    claim_scope: str
    scenario: TransposeScenario
    seed: int
    row: int
    config_sha256: str
    input_orientation: Literal["transposed"]
    contexts: tuple[TransposedGridContextScore, ...]

    def __post_init__(self) -> None:
        if (
            self.protocol_sha256 != PROTOCOL_SHA256
            or self.schema_version != SCHEMA_VERSION
            or self.claim_scope != CLAIM_SCOPE
        ):
            raise TransposeV22Error("selective transpose identity or claim scope is invalid")
        if self.scenario not in TRANSPOSE_SCENARIOS:
            raise TransposeV22Error("selective transpose scenario is outside the fixed panel")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise TransposeV22Error("selective transpose seed is invalid")
        if type(self.row) is not int or not 0 <= self.row < calibration.SYNTHETIC_ROWS:
            raise TransposeV22Error("selective transpose row is invalid")
        checked_config = _require_config_sha256(self.config_sha256)
        if self.input_orientation != "transposed":
            raise TransposeV22Error("selective transpose orientation must be transposed")
        grid_arms = TRANSPOSE_GRID_ARMS[self.scenario]
        expected_order = tuple(
            (arm, context) for arm in grid_arms for context in TRUE_CONTEXTS
        )
        if (
            type(self.contexts) is not tuple
            or any(not isinstance(item, TransposedGridContextScore) for item in self.contexts)
            or tuple((item.arm, item.plan.name) for item in self.contexts) != expected_order
        ):
            raise TransposeV22Error("selective transpose contexts differ from the exact panel order")
        for item in self.contexts:
            expected_key = _context_key(
                self.scenario,
                self.seed,
                self.row,
                item.arm,
                cast(TrueContext, item.plan.name),
            )
            if (
                item.result.context_key != expected_key
                or item.result.certificate.config_sha256 != checked_config
            ):
                raise TransposeV22Error("selective transpose context key or config scope is invalid")
        for context in TRUE_CONTEXTS:
            members = tuple(item for item in self.contexts if item.plan.name == context)
            if any(item.result.source_grid is not members[0].result.source_grid for item in members):
                raise TransposeV22Error("selective transpose same-stream arms did not share source evidence")

    def context(self, arm: exact.Arm, name: TrueContext) -> TransposedGridContextScore:
        """Return one exact retained member; caller cannot define new work."""

        if arm not in TRANSPOSE_GRID_ARMS[self.scenario] or name not in TRUE_CONTEXTS:
            raise TransposeV22Error("selective transpose context lookup is outside the fixed panel")
        return next(item for item in self.contexts if item.arm == arm and item.plan.name == name)


def _plans(bundle: synthetic.SyntheticRowTargets) -> tuple[scoring.ContextPlan, ...]:
    return (
        scoring.ContextPlan(
            "true_full", "true", bundle.row, geometry.FULL_MASK, geometry.FULL_MASK
        ),
        scoring.ContextPlan(
            "true_p0",
            "true",
            bundle.row,
            geometry.PARITY_MASKS[1],
            geometry.PARITY_MASKS[0],
        ),
        scoring.ContextPlan(
            "true_p1",
            "true",
            bundle.row,
            geometry.PARITY_MASKS[0],
            geometry.PARITY_MASKS[1],
        ),
    )


def score_transposed_grid_control_row(
    original_case: synthetic.SyntheticCase,
    row: int,
    *,
    config_sha256: str,
) -> TransposedGridControlRow:
    """Validate a native case, internally transpose it, and score only 3/6 retained contexts."""

    checked_config = _require_config_sha256(config_sha256)
    if not isinstance(original_case, synthetic.SyntheticCase):
        raise TransposeV22Error("selective transpose scoring requires a SyntheticCase")
    if isinstance(row, bool) or not isinstance(row, Integral):
        raise TransposeV22Error("selective transpose row must be an integer")
    checked_row = int(row)
    if not 0 <= checked_row < calibration.SYNTHETIC_ROWS:
        raise TransposeV22Error("selective transpose row is outside [0,5]")
    try:
        synthetic.validate_case(original_case)
    except synthetic.SyntheticV22Error as error:
        raise TransposeV22Error("selective transpose original case failed replay validation") from error
    if original_case.scenario not in TRANSPOSE_SCENARIOS:
        raise TransposeV22Error("selective transpose case is outside the fixed four-scenario panel")

    transposed_case = synthetic.transpose_case(original_case)
    scenario = cast(TransposeScenario, transposed_case.scenario)
    bundle = synthetic.row_targets(transposed_case, checked_row)
    plans = _plans(bundle)
    grid_arms = TRANSPOSE_GRID_ARMS[scenario]
    true_targets = {
        cast(TrueContext, plan.name): fitting.target_values(
            bundle.true_target, plan.fit_mask
        )
        for plan in plans
    }
    score_targets = {
        cast(TrueContext, plan.name): fitting.target_values(
            bundle.true_target, plan.output_mask
        )
        for plan in plans
    }
    by_key: dict[tuple[exact.Arm, TrueContext], TransposedGridContextScore] = {}
    for plan in plans:
        name = cast(TrueContext, plan.name)
        requests = tuple(
            exact.FitRequest.create(
                _context_key(
                    transposed_case.scenario,
                    transposed_case.seed,
                    checked_row,
                    arm,
                    name,
                ),
                arm,
                true_targets[name],
            )
            for arm in grid_arms
        )
        results = exact.fit_global_contexts(
            bundle.source,
            plan.fit_mask,
            plan.output_mask,
            requests,
            config_sha256=checked_config,
        )
        source_grid = results[0].source_grid
        if any(result.source_grid is not source_grid for result in results):
            raise TransposeV22Error("selective transpose same-stream requests did not share source evidence")
        for arm, result in zip(grid_arms, results, strict=True):
            by_key[(arm, name)] = TransposedGridContextScore(
                plan,
                arm,
                result,
                calibration.error_record(result.prediction, score_targets[name]),
            )
    contexts = tuple(
        by_key[(arm, context)] for arm in grid_arms for context in TRUE_CONTEXTS
    )
    return TransposedGridControlRow(
        PROTOCOL_SHA256,
        SCHEMA_VERSION,
        CLAIM_SCOPE,
        scenario,
        transposed_case.seed,
        checked_row,
        checked_config,
        "transposed",
        contexts,
    )


def validate_transposed_grid_control_row(
    evidence: TransposedGridControlRow,
    original_case: synthetic.SyntheticCase,
) -> None:
    """Deeply regenerate the exact retained projection and compare every nested bit."""

    if not isinstance(evidence, TransposedGridControlRow):
        raise TransposeV22Error("selective transpose validation requires its exact row type")
    if not isinstance(original_case, synthetic.SyntheticCase):
        raise TransposeV22Error("selective transpose validation requires a SyntheticCase")
    if evidence.scenario != original_case.scenario or evidence.seed != original_case.seed:
        raise TransposeV22Error("selective transpose evidence and original case identities differ")
    rebuilt = score_transposed_grid_control_row(
        original_case,
        evidence.row,
        config_sha256=evidence.config_sha256,
    )
    if not _scientific_equal(evidence, rebuilt):
        raise TransposeV22Error("selective transpose evidence differs from complete replay")


__all__ = [
    "CLAIM_SCOPE",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "TRANSPOSE_SCENARIOS",
    "TRANSPOSE_GRID_ARMS",
    "TRUE_CONTEXTS",
    "TransposeScenario",
    "TransposeV22Error",
    "TransposedGridContextScore",
    "TransposedGridControlRow",
    "score_transposed_grid_control_row",
    "validate_transposed_grid_control_row",
]
