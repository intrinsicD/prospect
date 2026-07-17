"""Pure MM-008 v2 synthetic generation, scoring, and expectation support.

Frozen seeds are declarations only.  Every generator call requires an explicit
caller-supplied seed; this module performs no I/O and never derives nonce seeds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Final, Literal, cast

import numpy as np

from bench.multimodal_mechanism_diagnostics import calibration_v2 as calibration
from bench.multimodal_mechanism_diagnostics import method as v1
from bench.multimodal_mechanism_diagnostics import method_v2 as v2

SCHEMA_VERSION: Final = "mm008-v2.1-synthetic-v1"
PROTOCOL_SHA256: Final = calibration.PROTOCOL_SHA256

Scenario = Literal[
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "coupled_boundary",
    "constant_target",
]
Arm = Literal[
    "global_translation", "quadrant_translation", "affine", "appearance", "combined"
]
Mode = Literal["full", "p0", "p1"]

SCENARIOS: Final[tuple[Scenario, ...]] = (
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "coupled_boundary",
    "constant_target",
)
ARMS: Final[tuple[Arm, ...]] = (
    "global_translation",
    "quadrant_translation",
    "affine",
    "appearance",
    "combined",
)
FROZEN_SEED_MAP: Final[dict[Scenario, int]] = {
    scenario: 820_800 + index for index, scenario in enumerate(SCENARIOS)
}


class SyntheticV2Error(ValueError):
    """Raised when a supplied synthetic case fails a frozen v2 contract."""


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=np.float64, order="C", copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class TransformTruth:
    theta: tuple[float, float, float, float, float, float]
    gains: tuple[float, float, float]
    biases: tuple[float, float, float]

    def theta_array(self) -> np.ndarray:
        return np.asarray(self.theta, dtype=np.float64)

    def gain_array(self) -> np.ndarray:
        return np.asarray(self.gains, dtype=np.float64)

    def bias_array(self) -> np.ndarray:
        return np.asarray(self.biases, dtype=np.float64)


IDENTITY_TRUTH: Final = TransformTruth(
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (0.0, 0.0, 0.0),
)
TRUTHS: Final[dict[Scenario, TransformTruth | None]] = {
    "translation": TransformTruth(
        (4.0, -4.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
    "affine": TransformTruth(
        (0.0, 0.0, 2.0, 0.0, 0.0, -2.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
    "appearance": TransformTruth(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.25, 0.75, 1.5),
        (0.35, -0.25, 0.15),
    ),
    "combined": TransformTruth(
        (-4.0, 4.0, 0.0, 2.0, -2.0, 0.0),
        (1.2, 0.8, 1.4),
        (0.3, -0.2, 0.1),
    ),
    "stationary": IDENTITY_TRUTH,
    "independent": None,
    "coupled_boundary": TransformTruth(
        (4.0, 0.0, 4.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ),
    "constant_target": None,
}


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    scenario: Scenario
    seed: int
    raw_source: np.ndarray
    raw_target: np.ndarray
    normalized_base: np.ndarray
    source: np.ndarray
    target: np.ndarray
    normalizer: calibration.SourceOnlyNormalizer
    source_broadband: calibration.BroadbandMetrics
    independent_target_broadband: calibration.BroadbandMetrics | None
    truth: TransformTruth | None

    def generator_failure_codes(self) -> tuple[str, ...]:
        failures = [
            f"source_{code}" for code in self.source_broadband.failure_reasons()
        ]
        if self.scenario == "independent":
            if self.independent_target_broadband is None:
                failures.append("independent_target_metrics_missing")
            else:
                failures.extend(
                    f"independent_target_{code}"
                    for code in self.independent_target_broadband.failure_reasons()
                )
        elif self.independent_target_broadband is not None:
            failures.append("unexpected_independent_target_metrics")
        if self.scenario == "stationary":
            if not np.array_equal(self.raw_source, self.raw_target):
                failures.append("stationary_raw_copy")
            if not np.array_equal(self.source, self.target):
                failures.append("stationary_normalized_copy")
        return tuple(failures)


def _require_scenario(scenario: str) -> Scenario:
    if scenario not in SCENARIOS:
        raise SyntheticV2Error("unknown MM-008 v2 synthetic scenario")
    return cast(Scenario, scenario)


def _require_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise SyntheticV2Error("synthetic seed must be an integer")
    value = int(seed)
    if not 0 <= value < 2**64:
        raise SyntheticV2Error("synthetic seed is outside uint64 range")
    return value


def _draw_raw(rng: np.random.Generator) -> np.ndarray:
    epsilon = rng.normal(
        size=(
            calibration.SYNTHETIC_ROWS,
            calibration.CHANNELS,
            calibration.NATIVE_SIZE,
            calibration.NATIVE_SIZE,
        )
    )
    offsets = rng.normal(
        size=(calibration.SYNTHETIC_ROWS, calibration.CHANNELS, 1, 1)
    )
    return np.asarray(epsilon + 0.50 * offsets, dtype=np.float64)


def _central_target(source: np.ndarray, truth: TransformTruth) -> np.ndarray:
    rows = len(source)
    parameters = np.broadcast_to(truth.theta_array(), (rows, 6)).copy()
    gains = np.broadcast_to(truth.gain_array(), (rows, 3))
    biases = np.broadcast_to(truth.bias_array(), (rows, 3))
    selected = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    transformed = v1._sample_affine(source, parameters, selected)
    transformed = gains[:, :, None] * transformed + biases[:, :, None]
    target = np.asarray(source, dtype=np.float64).copy()
    coords = v1.GEOMETRY.coords.astype(int)
    target[:, :, coords[:, 0], coords[:, 1]] = transformed
    return target


def generate_case(
    scenario: Scenario,
    *,
    seed: int,
    require_valid: bool = True,
) -> SyntheticCase:
    """Generate one exact six-row case from an explicitly injected seed."""

    checked_scenario = _require_scenario(scenario)
    checked_seed = _require_seed(seed)
    rng = np.random.Generator(np.random.PCG64(checked_seed))
    raw_source = _draw_raw(rng)
    normalizer = calibration.fit_source_only_normalizer(raw_source)
    normalized_base = normalizer.apply(raw_source)
    source_metrics = calibration.broadband_validity_metrics(normalized_base)
    truth = TRUTHS[checked_scenario]
    target_metrics: calibration.BroadbandMetrics | None = None

    if checked_scenario == "coupled_boundary":
        mask = (np.arange(calibration.NATIVE_SIZE) >= 32).astype(np.float64)
        source = np.asarray(normalized_base * mask[None, None, :, None], dtype=np.float64)
    else:
        source = normalized_base.copy()

    if checked_scenario == "stationary":
        raw_target = raw_source.copy()
        target = source.copy()
    elif checked_scenario == "independent":
        raw_target = _draw_raw(rng)
        target = normalizer.apply(raw_target)
        target_metrics = calibration.broadband_validity_metrics(target)
    elif checked_scenario == "constant_target":
        constants = rng.normal(size=(calibration.SYNTHETIC_ROWS, calibration.CHANNELS))
        target_normalized = np.broadcast_to(
            constants[:, :, None, None], source.shape
        ).copy()
        raw_target = normalizer.invert(target_normalized)
        target = normalizer.apply(raw_target)
    else:
        assert truth is not None
        target_normalized = _central_target(source, truth)
        raw_target = normalizer.invert(target_normalized)
        target = normalizer.apply(raw_target)

    case = SyntheticCase(
        scenario=checked_scenario,
        seed=checked_seed,
        raw_source=_readonly(raw_source),
        raw_target=_readonly(raw_target),
        normalized_base=_readonly(normalized_base),
        source=_readonly(source),
        target=_readonly(target),
        normalizer=normalizer,
        source_broadband=source_metrics,
        independent_target_broadband=target_metrics,
        truth=truth,
    )
    failures = case.generator_failure_codes()
    if require_valid and failures:
        raise SyntheticV2Error(f"synthetic generator invalid: {','.join(failures)}")
    return case


def generate_independent_case(*, seed: int, require_valid: bool = True) -> SyntheticCase:
    """Generate one independent source/target bank from an injected seed."""

    return generate_case("independent", seed=seed, require_valid=require_valid)


def case_replays_exactly(left: SyntheticCase, right: SyntheticCase) -> bool:
    """Check deterministic replay without hashing or serializing case contents."""

    return (
        left.scenario == right.scenario
        and left.seed == right.seed
        and left.truth == right.truth
        and np.array_equal(left.raw_source, right.raw_source)
        and np.array_equal(left.raw_target, right.raw_target)
        and np.array_equal(left.normalized_base, right.normalized_base)
        and np.array_equal(left.source, right.source)
        and np.array_equal(left.target, right.target)
        and np.array_equal(left.normalizer.mean, right.normalizer.mean)
        and np.array_equal(left.normalizer.scale, right.normalizer.scale)
    )


def derangements(rows: int = calibration.SYNTHETIC_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen adjacent-swap and half-cycle row mappings."""

    if isinstance(rows, bool) or not isinstance(rows, Integral):
        raise SyntheticV2Error("derangement row count must be an integer")
    count = int(rows)
    if count <= 0 or count % 2:
        raise SyntheticV2Error("derangements require a positive even row count")
    near = np.arange(count).reshape(-1, 2)[:, ::-1].reshape(-1)
    far = np.roll(np.arange(count), count // 2)
    return near, far


def _target_values(case: SyntheticCase, row: int) -> np.ndarray:
    if not 0 <= row < calibration.SYNTHETIC_ROWS:
        raise SyntheticV2Error("synthetic row is outside [0,5]")
    selected = np.ones(len(v1.GEOMETRY.coords), dtype=bool)
    values = cast(
        np.ndarray,
        v1._target_values(case.target[row : row + 1], selected),
    )
    return np.asarray(values[0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BiasRowScores:
    true_full: calibration.ErrorRecord
    true_xfit: calibration.ErrorRecord
    near_xfit: calibration.ErrorRecord
    far_xfit: calibration.ErrorRecord


def score_persistence(case: SyntheticCase, row: int) -> calibration.ErrorRecord:
    prediction = v1.persistence_prediction(case.source[row : row + 1])[0]
    return calibration.error_record(prediction, _target_values(case, row))


def score_bias_row(case: SyntheticCase, row: int) -> BiasRowScores:
    """Fit and score true/near/far bias-only comparators for one row."""

    target_values = _target_values(case, row)
    near, far = derangements()
    full = v2.estimate_bias_full(case.target[row])
    true = v2.estimate_bias_xfit(case.target[row])
    near_fit = v2.estimate_bias_xfit(case.target[int(near[row])])
    far_fit = v2.estimate_bias_xfit(case.target[int(far[row])])
    return BiasRowScores(
        true_full=calibration.error_record(full.prediction, target_values),
        true_xfit=calibration.error_record(true.prediction, target_values),
        near_xfit=calibration.error_record(near_fit.prediction, target_values),
        far_xfit=calibration.error_record(far_fit.prediction, target_values),
    )


@dataclass(frozen=True, slots=True)
class NamedEndpoint:
    name: str
    record: calibration.EndpointRecord


@dataclass(frozen=True, slots=True)
class CarryScores:
    affine_full: calibration.ErrorRecord
    affine_xfit: calibration.ErrorRecord
    appearance_full: calibration.ErrorRecord
    appearance_xfit: calibration.ErrorRecord


@dataclass(frozen=True, slots=True)
class DirectionCertificationRecord:
    """Recomputable claim/null certification evidence for one fit direction."""

    starts: calibration.IterativeStartCertificationRecord
    objective_agreement: bool
    prediction_agreement: bool
    flow_agreement: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.starts, calibration.IterativeStartCertificationRecord
        ):
            raise SyntheticV2Error("direction start certification record is invalid")
        flags = (
            self.objective_agreement,
            self.prediction_agreement,
            self.flow_agreement,
        )
        if any(type(value) is not bool for value in flags):
            raise SyntheticV2Error("direction agreement flags must be Python booleans")

    def certified(self, mode: v2.CertificationMode) -> bool:
        if mode == "null":
            return not self.starts.preempts
        if mode != "claim":
            raise SyntheticV2Error("direction certification mode is invalid")
        return (
            not self.starts.preempts
            and self.objective_agreement
            and self.prediction_agreement
            and self.flow_agreement
        )


@dataclass(frozen=True, slots=True)
class FitCertificationScores:
    """Claim/null validity and dual-start evidence for one scored row."""

    true_mode: v2.CertificationMode
    true_full: bool
    true_xfit: bool
    near_xfit: bool
    far_xfit: bool
    true_full_starts: tuple[calibration.IterativeStartCertificationRecord, ...]
    true_xfit_starts: tuple[calibration.IterativeStartCertificationRecord, ...]
    near_xfit_starts: tuple[calibration.IterativeStartCertificationRecord, ...]
    far_xfit_starts: tuple[calibration.IterativeStartCertificationRecord, ...]
    true_full_directions: tuple[DirectionCertificationRecord, ...]
    true_xfit_directions: tuple[DirectionCertificationRecord, ...]
    near_xfit_directions: tuple[DirectionCertificationRecord, ...]
    far_xfit_directions: tuple[DirectionCertificationRecord, ...]

    def __post_init__(self) -> None:
        if self.true_mode not in {"claim", "null"}:
            raise SyntheticV2Error("true fit certification mode is invalid")
        flags = (self.true_full, self.true_xfit, self.near_xfit, self.far_xfit)
        if any(type(value) is not bool for value in flags):
            raise SyntheticV2Error("fit certification flags must be Python booleans")
        groups = (
            self.true_full_starts,
            self.true_xfit_starts,
            self.near_xfit_starts,
            self.far_xfit_starts,
        )
        if any(not isinstance(group, tuple) for group in groups):
            raise SyntheticV2Error("fit start certifications must be immutable tuples")
        if any(
            not isinstance(item, calibration.IterativeStartCertificationRecord)
            for group in groups
            for item in group
        ):
            raise SyntheticV2Error("fit start certification record is invalid")
        direction_groups = (
            self.true_full_directions,
            self.true_xfit_directions,
            self.near_xfit_directions,
            self.far_xfit_directions,
        )
        if any(not isinstance(group, tuple) for group in direction_groups):
            raise SyntheticV2Error("fit direction certifications must be immutable tuples")
        if any(
            not isinstance(item, DirectionCertificationRecord)
            for group in direction_groups
            for item in group
        ):
            raise SyntheticV2Error("fit direction certification record is invalid")
        if any(
            tuple(item.starts for item in direction_group) != start_group
            for start_group, direction_group in zip(groups, direction_groups, strict=True)
        ):
            raise SyntheticV2Error("direction/start certification evidence differs")
        expected_flags = (
            all(item.certified(self.true_mode) for item in self.true_full_directions),
            all(item.certified(self.true_mode) for item in self.true_xfit_directions),
            all(item.certified("null") for item in self.near_xfit_directions),
            all(item.certified("null") for item in self.far_xfit_directions),
        )
        if any(direction_groups) and flags != expected_flags:
            raise SyntheticV2Error("aggregate fit certification differs from directions")
        if not any(direction_groups) and (groups != ((), (), (), ()) or flags != (True,) * 4):
            raise SyntheticV2Error("noniterative certification evidence is not empty/valid")

    @property
    def preempts(self) -> bool:
        return not all((self.true_full, self.true_xfit, self.near_xfit, self.far_xfit))


@dataclass(frozen=True, slots=True)
class ArmRowScores:
    row: int
    arm: Arm
    persistence: calibration.ErrorRecord
    true_full: calibration.ErrorRecord
    true_xfit: calibration.ErrorRecord
    near_xfit: calibration.ErrorRecord
    far_xfit: calibration.ErrorRecord
    true_full_q: calibration.QEnvelope
    true_xfit_q: calibration.QEnvelope
    near_xfit_q: calibration.QEnvelope
    far_xfit_q: calibration.QEnvelope
    pairing: calibration.QPairingRecord
    near_hit: calibration.QPredicateRecord
    far_hit: calibration.QPredicateRecord
    bias: BiasRowScores
    endpoints: tuple[NamedEndpoint, ...]
    complete: bool
    strong: bool
    no_bias_gain: bool
    full_certified: bool
    xfit_certified: bool
    full_objectives: tuple[float, ...]
    xfit_objectives: tuple[float, ...]
    full_prediction_exact: bool
    xfit_prediction_exact: bool
    certifications: FitCertificationScores
    carries: CarryScores | None

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or not isinstance(self.row, Integral):
            raise SyntheticV2Error("row-score index must be an integer")
        if not 0 <= int(self.row) < calibration.SYNTHETIC_ROWS:
            raise SyntheticV2Error("row-score index is outside [0,5]")
        object.__setattr__(self, "row", int(self.row))
        if self.arm not in ARMS:
            raise SyntheticV2Error("unknown MM-008 v2 arm score")
        summary_flags = (
            self.complete,
            self.strong,
            self.no_bias_gain,
            self.full_certified,
            self.xfit_certified,
            self.full_prediction_exact,
            self.xfit_prediction_exact,
        )
        if any(type(value) is not bool for value in summary_flags):
            raise SyntheticV2Error("row-score summary flags must be Python booleans")
        expected_objective_lengths = (
            (0, 0)
            if self.arm in {"global_translation", "quadrant_translation"}
            else (1, 2)
        )
        if (len(self.full_objectives), len(self.xfit_objectives)) != expected_objective_lengths:
            raise SyntheticV2Error("row-score objective membership differs")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (*self.full_objectives, *self.xfit_objectives)
        ):
            raise SyntheticV2Error("row-score objective is nonfinite or negative")
        envelopes = (
            self.true_full_q,
            self.true_xfit_q,
            self.near_xfit_q,
            self.far_xfit_q,
        )
        if any(envelope.arm != self.arm for envelope in envelopes):
            raise SyntheticV2Error("Q-envelope arm differs from row-score arm")
        ordinary_count = calibration.CHANNELS * 48 * 48
        ordinary_records = [
            self.persistence,
            self.true_full,
            self.true_xfit,
            self.near_xfit,
            self.far_xfit,
            self.bias.true_full,
            self.bias.true_xfit,
            self.bias.near_xfit,
            self.bias.far_xfit,
            *(panel.error for envelope in envelopes for panel in envelope.panels),
        ]
        if self.carries is not None:
            ordinary_records.extend(
                (
                    self.carries.affine_full,
                    self.carries.affine_xfit,
                    self.carries.appearance_full,
                    self.carries.appearance_xfit,
                )
            )
        if any(record.count != ordinary_count for record in ordinary_records):
            raise SyntheticV2Error("row-score ordinary MSE count differs from 6912")
        lengths = tuple(
            len(group)
            for group in (
                self.certifications.true_full_starts,
                self.certifications.true_xfit_starts,
                self.certifications.near_xfit_starts,
                self.certifications.far_xfit_starts,
            )
        )
        expected = (1, 2, 2, 2) if self.arm in {"affine", "combined"} else (0, 0, 0, 0)
        if lengths != expected:
            raise SyntheticV2Error("per-direction start certification membership differs")
        direction_lengths = tuple(
            len(group)
            for group in (
                self.certifications.true_full_directions,
                self.certifications.true_xfit_directions,
                self.certifications.near_xfit_directions,
                self.certifications.far_xfit_directions,
            )
        )
        if direction_lengths != expected:
            raise SyntheticV2Error("per-direction certification membership differs")
        if any(
            item.arm != self.arm
            for group in (
                self.certifications.true_full_starts,
                self.certifications.true_xfit_starts,
                self.certifications.near_xfit_starts,
                self.certifications.far_xfit_starts,
            )
            for item in group
        ):
            raise SyntheticV2Error("start certification arm differs from row-score arm")
        selected_full = self.true_full_q.by_label()["S"]
        selected_xfit = self.true_xfit_q.by_label()["S"]
        expected_true_xfit = (
            self.true_xfit_q.minimum.error
            if self.certifications.true_mode == "null"
            else selected_xfit
        )
        if self.true_full != selected_full or self.true_xfit != expected_true_xfit:
            raise SyntheticV2Error("true scalar score differs from its Q-panel contract")
        if (
            self.near_xfit != self.near_xfit_q.minimum.error
            or self.far_xfit != self.far_xfit_q.minimum.error
        ):
            raise SyntheticV2Error("wrong scalar score differs from its Q minimum")
        if (
            self.pairing.near.envelope != self.near_xfit_q
            or self.pairing.far.envelope != self.far_xfit_q
            or self.near_hit.envelope != self.near_xfit_q
            or self.far_hit.envelope != self.far_xfit_q
        ):
            raise SyntheticV2Error("Q predicate evidence differs from row-score panels")
        expected_pairing = calibration.q_pair_support(
            self.true_xfit.mse,
            self.near_xfit_q,
            self.far_xfit_q,
            self.bias.true_xfit.mse,
            self.bias.near_xfit.mse,
            self.bias.far_xfit.mse,
        )
        expected_complete = calibration.q_complete_support(
            self.arm,
            self.persistence.mse,
            self.true_xfit.mse,
            self.near_xfit_q,
            self.far_xfit_q,
            self.bias.true_xfit.mse,
            self.bias.near_xfit.mse,
            self.bias.far_xfit.mse,
        )
        expected_near_hit = calibration.q_wrong_target_hit(
            self.persistence.mse, self.near_xfit_q, self.bias.near_xfit.mse
        )
        expected_far_hit = calibration.q_wrong_target_hit(
            self.persistence.mse, self.far_xfit_q, self.bias.far_xfit.mse
        )
        expected_strong = (
            self.persistence.mse > 0.0
            and 2.0 * self.true_full.mse <= self.persistence.mse
            and 2.0 * self.true_xfit.mse <= self.persistence.mse
            and expected_pairing.passed
            and expected_complete
            and self.endpoints_pass
        )
        expected_no_bias_gain = (
            self.bias.true_xfit.mse > 0.0
            and calibration.PERSISTENCE_FACTOR * self.true_xfit.mse
            > self.bias.true_xfit.mse
        )
        if (
            self.pairing != expected_pairing
            or self.near_hit != expected_near_hit
            or self.far_hit != expected_far_hit
            or self.complete is not expected_complete
            or self.strong is not expected_strong
            or self.no_bias_gain is not expected_no_bias_gain
        ):
            raise SyntheticV2Error("row-score predicate summary is not recomputable")
        if (
            self.full_certified is not self.certifications.true_full
            or self.xfit_certified is not self.certifications.true_xfit
        ):
            raise SyntheticV2Error("row-score certification summary differs")
        if (self.arm == "combined") != (self.carries is not None):
            raise SyntheticV2Error("combined carry evidence membership differs")

    @property
    def endpoints_pass(self) -> bool:
        return bool(self.endpoints) and all(item.record.passes() for item in self.endpoints)


@dataclass(frozen=True, slots=True)
class _FitOutput:
    prediction: np.ndarray
    parameters: np.ndarray
    gains: np.ndarray
    biases: np.ndarray
    objectives: tuple[float, ...]
    certified: bool
    q_panels: tuple[v2.QPanel, ...]
    start_certifications: tuple[calibration.IterativeStartCertificationRecord, ...]
    direction_certifications: tuple[DirectionCertificationRecord, ...]
    affine_carry: np.ndarray | None = None
    appearance_carry: np.ndarray | None = None


def _v2_carry_prediction(estimate: v2.Estimate, kind: str) -> np.ndarray:
    output = np.full((v1.CHANNELS, len(v1.GEOMETRY.coords)), np.nan, dtype=np.float64)
    if estimate.mode == "full":
        carry = getattr(estimate.directions[0], kind)
        assert carry is not None
        return np.asarray(carry.prediction, dtype=np.float64)
    for parity, direction in enumerate(estimate.directions):
        carry = getattr(direction, kind)
        assert carry is not None
        output[:, v1.GEOMETRY.parities == parity] = carry.prediction
    if not np.all(np.isfinite(output)):
        raise SyntheticV2Error("combined carry failed to predict every central site")
    return output


def _singleton_q_panel(prediction: np.ndarray) -> tuple[v2.QPanel, ...]:
    return (v2.QPanel("S", np.asarray(prediction, dtype=np.float64)),)


def _v2_start_certifications(
    estimate: v2.Estimate,
) -> tuple[calibration.IterativeStartCertificationRecord, ...]:
    arm = cast(calibration.IterativeArm, estimate.arm)
    return tuple(
        calibration.IterativeStartCertificationRecord(
            arm,
            direction.optimizer.forward.certified,
            direction.optimizer.reverse.certified,
        )
        for direction in estimate.directions
    )


def _v2_direction_certifications(
    estimate: v2.Estimate,
) -> tuple[DirectionCertificationRecord, ...]:
    starts = _v2_start_certifications(estimate)
    return tuple(
        DirectionCertificationRecord(
            start,
            direction.optimizer.objective_agreement,
            direction.optimizer.prediction_agreement,
            direction.optimizer.flow_agreement,
        )
        for start, direction in zip(starts, estimate.directions, strict=True)
    )


def score_q_panels(
    arm: Arm,
    panels: Sequence[v2.QPanel],
    expected: np.ndarray,
) -> calibration.QEnvelope:
    """Score complete Q panels and fail closed on arm/label/count drift."""

    if arm not in ARMS:
        raise SyntheticV2Error("unknown MM-008 v2 arm")
    labels = tuple(panel.label for panel in panels)
    if len(labels) != len(set(labels)):
        raise SyntheticV2Error("Q prediction panel labels are duplicated")
    errors: dict[str, calibration.ErrorRecord] = {
        panel.label: calibration.error_record(panel.prediction, expected)
        for panel in panels
    }
    return calibration.q_envelope(cast(calibration.SupportArm, arm), errors)


def _fit_full(
    source: np.ndarray,
    target: np.ndarray,
    arm: Arm,
    *,
    require_certified: bool,
    certification_mode: v2.CertificationMode,
) -> _FitOutput:
    if arm in {"global_translation", "quadrant_translation"}:
        sentinel_result = v1.estimate_sentinel_full(
            source[None], target[None], cast(v1.SentinelArm, arm)
        )
        prediction = sentinel_result.prediction[0]
        return _FitOutput(
            prediction=prediction,
            parameters=sentinel_result.flow[0],
            gains=np.ones(3),
            biases=np.zeros(3),
            objectives=(),
            certified=True,
            q_panels=_singleton_q_panel(prediction),
            start_certifications=(),
            direction_certifications=(),
        )
    if arm == "appearance":
        appearance_result = v1.estimate_full(source[None], target[None], "appearance")
        prediction = appearance_result.prediction[0]
        return _FitOutput(
            prediction=prediction,
            parameters=appearance_result.parameters[0],
            gains=appearance_result.gains[0],
            biases=appearance_result.biases[0],
            objectives=(float(appearance_result.objective[0]),),
            certified=True,
            q_panels=_singleton_q_panel(prediction),
            start_certifications=(),
            direction_certifications=(),
        )
    result_v2 = v2.estimate_full(
        source,
        target,
        cast(v2.Arm, arm),
        require_certified=require_certified,
        certification_mode=certification_mode,
    )
    return _FitOutput(
        prediction=result_v2.prediction,
        parameters=result_v2.parameters[0],
        gains=result_v2.gains[0],
        biases=result_v2.biases[0],
        objectives=tuple(float(value) for value in result_v2.objectives),
        certified=result_v2.certified,
        q_panels=result_v2.q_panels(),
        start_certifications=_v2_start_certifications(result_v2),
        direction_certifications=_v2_direction_certifications(result_v2),
        affine_carry=(
            _v2_carry_prediction(result_v2, "affine_carry")
            if arm == "combined"
            else None
        ),
        appearance_carry=(
            _v2_carry_prediction(result_v2, "appearance_carry")
            if arm == "combined"
            else None
        ),
    )


def _fit_xfit(
    source: np.ndarray,
    target: np.ndarray,
    arm: Arm,
    *,
    require_certified: bool,
    certification_mode: v2.CertificationMode,
) -> _FitOutput:
    if arm in {"global_translation", "quadrant_translation"}:
        sentinel_result = v1.estimate_sentinel_xfit(
            source[None], target[None], cast(v1.SentinelArm, arm)
        )
        prediction = sentinel_result.prediction[0]
        return _FitOutput(
            prediction=prediction,
            parameters=sentinel_result.flow[0],
            gains=np.ones((2, 3)),
            biases=np.zeros((2, 3)),
            objectives=(),
            certified=True,
            q_panels=_singleton_q_panel(prediction),
            start_certifications=(),
            direction_certifications=(),
        )
    if arm == "appearance":
        appearance_result = v1.estimate_xfit(source[None], target[None], "appearance")
        prediction = appearance_result.prediction[0]
        return _FitOutput(
            prediction=prediction,
            parameters=appearance_result.parameters[0],
            gains=appearance_result.gains[0],
            biases=appearance_result.biases[0],
            objectives=tuple(float(value) for value in appearance_result.objective[0]),
            certified=True,
            q_panels=_singleton_q_panel(prediction),
            start_certifications=(),
            direction_certifications=(),
        )
    result_v2 = v2.estimate_xfit(
        source,
        target,
        cast(v2.Arm, arm),
        require_certified=require_certified,
        certification_mode=certification_mode,
    )
    return _FitOutput(
        prediction=result_v2.prediction,
        parameters=result_v2.parameters,
        gains=result_v2.gains,
        biases=result_v2.biases,
        objectives=tuple(float(value) for value in result_v2.objectives),
        certified=result_v2.certified,
        q_panels=result_v2.q_panels(),
        start_certifications=_v2_start_certifications(result_v2),
        direction_certifications=_v2_direction_certifications(result_v2),
        affine_carry=(
            _v2_carry_prediction(result_v2, "affine_carry")
            if arm == "combined"
            else None
        ),
        appearance_carry=(
            _v2_carry_prediction(result_v2, "appearance_carry")
            if arm == "combined"
            else None
        ),
    )


def _endpoint_records(
    arm: Arm, truth: TransformTruth | None, full: _FitOutput, xfit: _FitOutput
) -> tuple[NamedEndpoint, ...]:
    if truth is None:
        return ()
    records: list[NamedEndpoint] = []

    def add(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
        records.append(NamedEndpoint(name, calibration.endpoint_record(actual, expected)))

    if arm in {"global_translation", "quadrant_translation"}:
        flow = truth.theta_array()[:2]
        add("flow_full", full.parameters, np.broadcast_to(flow, full.parameters.shape))
        add("flow_p0", xfit.parameters[0], np.broadcast_to(flow, xfit.parameters[0].shape))
        add("flow_p1", xfit.parameters[1], np.broadcast_to(flow, xfit.parameters[1].shape))
    if arm in {"affine", "combined"}:
        theta = truth.theta_array()
        add("theta_full", full.parameters, theta)
        add("theta_p0", xfit.parameters[0], theta)
        add("theta_p1", xfit.parameters[1], theta)
    if arm in {"appearance", "combined"}:
        gains = truth.gain_array()
        biases = truth.bias_array()
        add("gain_full", full.gains, gains)
        add("gain_p0", xfit.gains[0], gains)
        add("gain_p1", xfit.gains[1], gains)
        add("bias_full", full.biases, biases)
        add("bias_p0", xfit.biases[0], biases)
        add("bias_p1", xfit.biases[1], biases)
    return tuple(records)


def score_arm_row(
    case: SyntheticCase,
    row: int,
    arm: Arm,
    *,
    bias: BiasRowScores | None = None,
    require_certified: bool = True,
) -> ArmRowScores:
    """Fit and score one arm against true, near, and far targets for one row."""

    if arm not in ARMS:
        raise SyntheticV2Error("unknown MM-008 v2 arm")
    target_values = _target_values(case, row)
    near, far = derangements()
    target_independent = case.scenario in {"independent", "constant_target"}
    true_mode: v2.CertificationMode = "null" if target_independent else "claim"
    full = _fit_full(
        case.source[row],
        case.target[row],
        arm,
        require_certified=require_certified,
        certification_mode=true_mode,
    )
    xfit = _fit_xfit(
        case.source[row],
        case.target[row],
        arm,
        require_certified=require_certified,
        certification_mode=true_mode,
    )
    near_fit = _fit_xfit(
        case.source[row],
        case.target[int(near[row])],
        arm,
        require_certified=require_certified,
        certification_mode="null",
    )
    far_fit = _fit_xfit(
        case.source[row],
        case.target[int(far[row])],
        arm,
        require_certified=require_certified,
        certification_mode="null",
    )
    bias_scores = score_bias_row(case, row) if bias is None else bias
    persistence = score_persistence(case, row)
    true_full_q = score_q_panels(arm, full.q_panels, target_values)
    true_xfit_q = score_q_panels(arm, xfit.q_panels, target_values)
    near_xfit_q = score_q_panels(arm, near_fit.q_panels, target_values)
    far_xfit_q = score_q_panels(arm, far_fit.q_panels, target_values)
    true_full = true_full_q.by_label()["S"]
    true_xfit = (
        true_xfit_q.minimum.error
        if target_independent
        else true_xfit_q.by_label()["S"]
    )
    near_xfit = near_xfit_q.minimum.error
    far_xfit = far_xfit_q.minimum.error
    endpoints = _endpoint_records(arm, case.truth, full, xfit)
    endpoint_pass = bool(endpoints) and all(item.record.passes() for item in endpoints)
    pairing = calibration.q_pair_support(
        true_xfit.mse,
        near_xfit_q,
        far_xfit_q,
        bias_scores.true_xfit.mse,
        bias_scores.near_xfit.mse,
        bias_scores.far_xfit.mse,
    )
    complete = calibration.q_complete_support(
        arm,
        persistence.mse,
        true_xfit.mse,
        near_xfit_q,
        far_xfit_q,
        bias_scores.true_xfit.mse,
        bias_scores.near_xfit.mse,
        bias_scores.far_xfit.mse,
    )
    near_hit = calibration.q_wrong_target_hit(
        persistence.mse, near_xfit_q, bias_scores.near_xfit.mse
    )
    far_hit = calibration.q_wrong_target_hit(
        persistence.mse, far_xfit_q, bias_scores.far_xfit.mse
    )
    strong = (
        persistence.mse > 0.0
        and 2.0 * true_full.mse <= persistence.mse
        and 2.0 * true_xfit.mse <= persistence.mse
        and pairing.passed
        and complete
        and endpoint_pass
    )
    no_bias_gain = (
        bias_scores.true_xfit.mse > 0.0
        and calibration.PERSISTENCE_FACTOR * true_xfit.mse
        > bias_scores.true_xfit.mse
    )
    carries: CarryScores | None = None
    if arm == "combined":
        assert full.affine_carry is not None and xfit.affine_carry is not None
        assert full.appearance_carry is not None and xfit.appearance_carry is not None
        carries = CarryScores(
            affine_full=calibration.error_record(full.affine_carry, target_values),
            affine_xfit=calibration.error_record(xfit.affine_carry, target_values),
            appearance_full=calibration.error_record(full.appearance_carry, target_values),
            appearance_xfit=calibration.error_record(xfit.appearance_carry, target_values),
        )
    certifications = FitCertificationScores(
        true_mode=true_mode,
        true_full=full.certified,
        true_xfit=xfit.certified,
        near_xfit=near_fit.certified,
        far_xfit=far_fit.certified,
        true_full_starts=full.start_certifications,
        true_xfit_starts=xfit.start_certifications,
        near_xfit_starts=near_fit.start_certifications,
        far_xfit_starts=far_fit.start_certifications,
        true_full_directions=full.direction_certifications,
        true_xfit_directions=xfit.direction_certifications,
        near_xfit_directions=near_fit.direction_certifications,
        far_xfit_directions=far_fit.direction_certifications,
    )
    return ArmRowScores(
        row=row,
        arm=arm,
        persistence=persistence,
        true_full=true_full,
        true_xfit=true_xfit,
        near_xfit=near_xfit,
        far_xfit=far_xfit,
        true_full_q=true_full_q,
        true_xfit_q=true_xfit_q,
        near_xfit_q=near_xfit_q,
        far_xfit_q=far_xfit_q,
        pairing=pairing,
        near_hit=near_hit,
        far_hit=far_hit,
        bias=bias_scores,
        endpoints=endpoints,
        complete=complete,
        strong=strong,
        no_bias_gain=no_bias_gain,
        full_certified=full.certified,
        xfit_certified=xfit.certified,
        full_objectives=full.objectives,
        xfit_objectives=xfit.objectives,
        full_prediction_exact=all(
            np.array_equal(panel.prediction, target_values) for panel in full.q_panels
        ),
        xfit_prediction_exact=all(
            np.array_equal(panel.prediction, target_values) for panel in xfit.q_panels
        ),
        certifications=certifications,
        carries=carries,
    )


@dataclass(frozen=True, slots=True)
class ArmBatchScores:
    arm: Arm
    rows: tuple[ArmRowScores, ...]

    def __post_init__(self) -> None:
        indices = tuple(item.row for item in self.rows)
        if len(indices) != len(set(indices)) or indices != tuple(sorted(indices)):
            raise SyntheticV2Error("batch score rows must be unique and sorted")
        if any(item.arm != self.arm for item in self.rows):
            raise SyntheticV2Error("batch score arm membership differs")

    def by_row(self) -> dict[int, ArmRowScores]:
        return {item.row: item for item in self.rows}


def score_arm_batch(
    case: SyntheticCase,
    arm: Arm,
    *,
    rows: Sequence[int] | None = None,
    require_certified: bool = True,
) -> ArmBatchScores:
    """Score an ordered subset or the complete six-row bank."""

    selected = tuple(range(calibration.SYNTHETIC_ROWS)) if rows is None else tuple(rows)
    if len(selected) != len(set(selected)) or selected != tuple(sorted(selected)):
        raise SyntheticV2Error("requested score rows must be unique and sorted")
    biases = {row: score_bias_row(case, row) for row in selected}
    return ArmBatchScores(
        arm=arm,
        rows=tuple(
            score_arm_row(
                case,
                row,
                arm,
                bias=biases[row],
                require_certified=require_certified,
            )
            for row in selected
        ),
    )


def row_dominates(preferred: ArmRowScores, comparator: ArmRowScores) -> bool:
    if preferred.row != comparator.row:
        raise SyntheticV2Error("dominance rows differ")
    return calibration.dominates(
        preferred.true_full.mse,
        comparator.true_full.mse,
        preferred.true_xfit.mse,
        comparator.true_xfit.mse,
    )


def _carry_dominance(score: ArmRowScores) -> bool:
    if score.carries is None:
        return False
    return calibration.dominates(
        score.true_full.mse,
        score.carries.affine_full.mse,
        score.true_xfit.mse,
        score.carries.affine_xfit.mse,
    ) and calibration.dominates(
        score.true_full.mse,
        score.carries.appearance_full.mse,
        score.true_xfit.mse,
        score.carries.appearance_xfit.mse,
    )


_POSITIVE_ARMS: Final[dict[Scenario, tuple[Arm, ...]]] = {
    "translation": ("global_translation", "quadrant_translation", "affine", "combined"),
    "affine": ("affine", "combined"),
    "appearance": ("appearance", "combined"),
    "combined": ("combined",),
    "stationary": (),
    "independent": (),
    "coupled_boundary": (),
    "constant_target": (),
}
_REQUIRED_ARMS: Final[dict[Scenario, tuple[Arm, ...]]] = {
    "translation": ARMS,
    "affine": ARMS,
    "appearance": ARMS,
    "combined": ARMS,
    "stationary": ARMS,
    "independent": ARMS,
    "coupled_boundary": ("affine", "combined"),
    "constant_target": ("appearance", "combined"),
}
_DOMINANCE: Final[dict[Scenario, tuple[tuple[Arm, Arm], ...]]] = {
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


def scenario_expectation_failures(
    case: SyntheticCase, scores: Mapping[Arm, ArmBatchScores]
) -> tuple[str, ...]:
    """Evaluate frozen scenario expectations from complete primitive score banks."""

    failures = list(case.generator_failure_codes())
    required_arms = _REQUIRED_ARMS[case.scenario]
    if set(scores) != set(required_arms):
        return tuple((*failures, "score_arm_membership"))
    expected_rows = tuple(range(calibration.SYNTHETIC_ROWS))
    for arm in required_arms:
        if tuple(item.row for item in scores[arm].rows) != expected_rows:
            return tuple((*failures, f"{arm}:score_row_membership"))
    by_arm = {arm: scores[arm].by_row() for arm in required_arms}

    for row in expected_rows:
        for arm in required_arms:
            certification = by_arm[arm][row].certifications
            for context_name, valid in (
                ("true_full", certification.true_full),
                ("true_xfit", certification.true_xfit),
                ("near_xfit", certification.near_xfit),
                ("far_xfit", certification.far_xfit),
            ):
                if not valid:
                    failures.append(f"row{row}:{arm}:{context_name}_certification")
        for arm in _POSITIVE_ARMS[case.scenario]:
            if not by_arm[arm][row].strong:
                failures.append(f"row{row}:{arm}:strong")
        for preferred, comparator in _DOMINANCE[case.scenario]:
            if not row_dominates(by_arm[preferred][row], by_arm[comparator][row]):
                failures.append(f"row{row}:{preferred}>{comparator}:dominance")

        if case.scenario == "combined" and not _carry_dominance(by_arm["combined"][row]):
            failures.append(f"row{row}:combined:carry_dominance")
        elif case.scenario == "stationary":
            for arm in ARMS:
                score = by_arm[arm][row]
                if score.persistence.mse != 0.0 or score.complete or score.strong:
                    failures.append(f"row{row}:{arm}:stationary_support")
                if not score.full_prediction_exact or not score.xfit_prediction_exact:
                    failures.append(f"row{row}:{arm}:stationary_prediction")
        elif case.scenario == "independent":
            for arm in ("global_translation", "quadrant_translation", "affine"):
                if by_arm[arm][row].complete:
                    failures.append(f"row{row}:{arm}:independent_complete")
            for arm in ("appearance", "combined"):
                score = by_arm[arm][row]
                if not score.no_bias_gain or score.complete:
                    failures.append(f"row{row}:{arm}:independent_bias_null")
        elif case.scenario == "constant_target":
            for arm in ("appearance", "combined"):
                score = by_arm[arm][row]
                if (
                    abs(score.true_xfit.mse - score.bias.true_xfit.mse) > 1e-12
                    or calibration.beats_bias(score.true_xfit.mse, score.bias.true_xfit.mse)
                    or score.complete
                ):
                    failures.append(f"row{row}:{arm}:constant_target_bias_null")
        elif case.scenario == "coupled_boundary":
            for arm in ("affine", "combined"):
                score = by_arm[arm][row]
                objectives = (*score.full_objectives, *score.xfit_objectives)
                if (
                    not score.endpoints_pass
                    or not score.full_certified
                    or not score.xfit_certified
                    or not objectives
                    or max(objectives) > 1e-12
                ):
                    failures.append(f"row{row}:{arm}:coupled_boundary")
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class ExhaustiveDispatch:
    scenario: Scenario
    arm: v2.Arm
    row: int
    mode: Mode
    context: v2.FitContext
    truth: v2.StateValues


DECLARED_EXHAUSTIVE_PAIRS: Final = frozenset(
    {
        ("translation", "affine"),
        ("affine", "affine"),
        ("combined", "combined"),
        ("coupled_boundary", "affine"),
    }
)
MUTATION_DELTA: Final = 123.0
MUTATION_OUTPUT_PARITIES: Final[tuple[int, ...]] = (0, 1)
DECLARED_MUTATION_PAIRS: Final[tuple[tuple[Scenario, v2.Arm], ...]] = (
    ("affine", "affine"),
    ("combined", "combined"),
)
DECLARED_TRANSPOSE_PAIRS: Final[tuple[tuple[Scenario, v2.Arm], ...]] = (
    ("translation", "affine"),
    ("translation", "combined"),
    ("affine", "affine"),
    ("affine", "combined"),
    ("appearance", "combined"),
    ("combined", "combined"),
)

SYNTHETIC_CONFIG_TAG: Final = b"MM008-v2.1-synthetic-config\0"


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Immutable canonical JSON payload bound by the synthetic config hash."""

    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise SyntheticV2Error("synthetic config must be canonical JSON text")
        try:
            self.canonical_json.encode("ascii")
            parsed = json.loads(self.canonical_json)
            rebuilt = json.dumps(
                parsed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (UnicodeEncodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise SyntheticV2Error("synthetic config JSON is invalid") from error
        if not isinstance(parsed, dict) or rebuilt != self.canonical_json:
            raise SyntheticV2Error("synthetic config JSON is not canonical")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SyntheticConfig:
        if not isinstance(payload, Mapping):
            raise SyntheticV2Error("synthetic config payload must be a mapping")
        try:
            canonical = json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise SyntheticV2Error("synthetic config payload is not finite JSON") from error
        return cls(canonical)

    @property
    def canonical_bytes(self) -> bytes:
        return self.canonical_json.encode("ascii")

    @property
    def sha256(self) -> str:
        return sha256(SYNTHETIC_CONFIG_TAG + self.canonical_bytes).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.canonical_json))


def _synthetic_config_payload() -> dict[str, object]:
    truths: dict[str, object] = {}
    for scenario in SCENARIOS:
        truth = TRUTHS[scenario]
        truths[scenario] = (
            None
            if truth is None
            else {
                "theta": list(truth.theta),
                "gains": list(truth.gains),
                "biases": list(truth.biases),
            }
        )
    exhaustive = sorted(
        DECLARED_EXHAUSTIVE_PAIRS,
        key=lambda item: (SCENARIOS.index(cast(Scenario, item[0])), ARMS.index(cast(Arm, item[1]))),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "method_config_sha256": v2.CONFIG_SHA256,
        "candidate_order_sha256": v2.CANDIDATE_ORDER_SHA256,
        "scenario_order": list(SCENARIOS),
        "arm_order": list(ARMS),
        "ordinary_seed_map": {
            scenario: FROZEN_SEED_MAP[scenario] for scenario in SCENARIOS
        },
        "truths": truths,
        "source_generator": {
            "bit_generator": "PCG64",
            "dtype": "float64",
            "draw_order": ["epsilon", "c"],
            "epsilon_distribution": "standard_normal",
            "epsilon_shape": [6, 3, 64, 64],
            "c_distribution": "standard_normal",
            "c_shape": [6, 3, 1, 1],
            "c_factor": 0.5,
            "raw_source_expression": "epsilon+0.50*c",
        },
        "normalizer": {
            "fit_array": "raw_source_only",
            "pool": "area_mean_R64_to_R8",
            "pool_block": calibration.POOL_BLOCK,
            "pooled_size": calibration.POOLED_SIZE,
            "channelwise": True,
            "statistic_axes": ["row", "r8_y", "r8_x"],
            "mean": "float64_mean",
            "scale": "float64_population_std",
            "scale_floor": calibration.SCALE_FLOOR,
            "apply_resolution": calibration.NATIVE_SIZE,
        },
        "broadband": {
            "evaluated_array": "normalized_base_before_optional_mask",
            "matrix_shape": [calibration.BROADBAND_RANK, 64 * 64],
            "required_rank": calibration.BROADBAND_RANK,
            "require_positive_row_rms": True,
            "min_singular_ratio_strict": calibration.MIN_SINGULAR_RATIO,
            "central_slice": [calibration.CENTRAL_START, calibration.CENTRAL_STOP],
            "min_central_variance_strict": calibration.MIN_CENTRAL_VARIANCE,
            "lag_axes": ["x", "y"],
            "require_positive_lag_denominators": True,
            "max_abs_lag_correlation_strict": calibration.MAX_ABS_LAG_CORRELATION,
        },
        "scenario_construction": {
            "positive_target": "central_replace_then_source_normalizer_invert_reapply",
            "central_slice": [calibration.CENTRAL_START, calibration.CENTRAL_STOP],
            "stationary": "bit_exact_raw_and_normalized_copy",
            "independent": "second_complete_epsilon_c_draw_source_normalized",
            "constant_target": {
                "draw_distribution": "standard_normal",
                "draw_shape": [6, 3],
                "space": "normalized",
                "broadcast_shape": [6, 3, 64, 64],
            },
            "coupled_boundary": {
                "mask": "1[y>=32]",
                "mask_after_normalization": True,
                "mask_before_target": True,
                "truth_flow_range": [0.0, 8.0],
            },
        },
        "derangements": {
            "near": [1, 0, 3, 2, 5, 4],
            "far": [3, 4, 5, 0, 1, 2],
        },
        "required_arms": {
            scenario: list(_REQUIRED_ARMS[scenario]) for scenario in SCENARIOS
        },
        "positive_arms": {
            scenario: list(_POSITIVE_ARMS[scenario]) for scenario in SCENARIOS
        },
        "dominance": {
            scenario: [list(pair) for pair in _DOMINANCE[scenario]]
            for scenario in SCENARIOS
        },
        "combined_carry_dominance": ["affine_carry", "appearance_carry"],
        "q_envelope": {
            "tie_order": list(calibration.Q_LABEL_TIE_ORDER),
            "applicability": {
                arm: list(
                    calibration.applicable_q_labels(cast(calibration.SupportArm, arm))
                )
                for arm in ARMS
            },
            "target_independent_scenarios": ["independent", "constant_target"],
            "claim_true_scenarios": [
                "translation",
                "affine",
                "appearance",
                "combined",
                "stationary",
                "coupled_boundary",
            ],
            "wrong_mode": "null",
            "selection": "complete_panel_minimum_after_target_blind_assembly",
            "pairing_reduction": "ALL",
            "hit_reduction": "ANY_once_per_wrong_target",
        },
        "predicates": {
            "persistence_factor": calibration.PERSISTENCE_FACTOR,
            "pairing_factor": calibration.PAIRING_FACTOR,
            "strong_factor": 2.0,
            "endpoint_tolerance": calibration.ENDPOINT_TOLERANCE,
            "constant_bias_tolerance": 1e-12,
            "coupled_objective_tolerance": 1e-12,
            "flow_equivalence_tolerance": v2.FLOW_EQUIVALENCE_ATOL,
            "prediction_agreement_tolerance": v2.PREDICTION_AGREEMENT_ATOL,
            "formulas": {
                "Pair": (
                    "b_T>0 and b_N>0 and b_F>0 and "
                    "1.10*o_m*b_N<=min_q(n_m^q)*b_T and "
                    "1.10*o_m*b_F<=min_q(s_m^q)*b_T"
                ),
                "Perf": "p>0 and 1.25*o_m<=p",
                "BeatsBias": "b_T>0 and 1.25*o_m<=b_T",
                "Complete": (
                    "Perf(m) and Pair(m) and "
                    "(m not in {appearance,combined} or BeatsBias(m))"
                ),
                "Strong": (
                    "p>0 and 2.0*f_m<=p and 2.0*o_m<=p and Pair(m) and "
                    "Complete(m) and all_applicable_endpoint_error<=1e-10"
                ),
                "Dom": (
                    "f_b>f_a and 1.25*f_a<=f_b and "
                    "o_b>o_a and 1.25*o_a<=o_b"
                ),
                "NoBiasGain": "b_T>0 and 1.25*o_m>b_T",
                "Hit": (
                    "p>0 and b_wrong>0 and 1.25*min_q(wrong_m^q)<=p and "
                    "1.25*min_q(wrong_m^q)<=b_wrong"
                ),
            },
        },
        "scoring": {
            "dtype": "float64",
            "central_shape": [3, 48 * 48],
            "ordinary_element_count": 3 * 48 * 48,
            "optimizer_objective_is_primary_mse": False,
        },
        "exhaustive_pairs": [list(pair) for pair in exhaustive],
        "exhaustive_modes": ["full", "p0", "p1"],
        "exhaustive_rules": {
            "state_count": v2.STATE_COUNT,
            "enumeration": "full_canonical_order",
            "selection_key": ["objective", "canonical_state_key"],
            "selected_equals_first_global_minimum": True,
            "selected_equals_injected_truth": True,
            "objective_tolerance": "max(1e-12,1e-10*abs(final_objective))",
            "selected_vs_truth": "selected<=direct_truth+tolerance",
            "flow_equivalence_atol": 1e-12,
            "second_best_non_equivalent_flow_gap_strict": 1e-12,
            "ranks": ["total_zero_based", "admissible_zero_based"],
            "saved_counts": ["candidate", "admissible"],
        },
        "challenge": {
            "scenario": "independent",
            "count": calibration.CHALLENGE_COUNT,
            "label_prefix": calibration.CHALLENGE_LABEL_PREFIX,
            "nonce_schema_version": calibration.NONCE_SCHEMA_VERSION,
            "digest": "sha256",
            "digest_input_order": [
                "bytes.fromhex(protocol_sha256)",
                "bytes.fromhex(auditor_nonce_hex)",
                "ascii(label_prefix+decimal_index)",
            ],
            "digest_slice": [0, 8],
            "integer_endian": "big",
            "integer_signed": False,
        },
        "metamorphic_controls": {
            "replay": "bit_exact",
            "held_target_mutation": {
                "delta": MUTATION_DELTA,
                "operation": "add_to_all_channels_at_held_output_parity_sites",
                "row": 0,
                "scenario_arm_pairs": [
                    list(pair) for pair in DECLARED_MUTATION_PAIRS
                ],
                "output_parities": list(MUTATION_OUTPUT_PARITIES),
                "preserve": [
                    "parameters",
                    "gains",
                    "biases",
                    "retained_macrocell_ids",
                    "optimizer_histories_and_probes",
                    "bias_only_fit",
                    "held_parity_prediction",
                ],
                "comparison": "bit_exact",
            },
            "q_order_swap_invariant": True,
            "transpose_theta_permutation": [1, 0, 5, 4, 3, 2],
            "transpose_scenario_arm_pairs": [
                list(pair) for pair in DECLARED_TRANSPOSE_PAIRS
            ],
            "transpose_rows": "all_six",
            "transpose_comparison": "bit_exact_unique_minimum",
            "flow_limit_inclusive": 8.0,
            "first_rejected_flow": "nextafter(8,+inf)",
        },
        "boundary_expectations": {
            "sampler_accepts": [-8.0, 8.0],
            "sampler_rejects": ["nextafter(-8,-inf)", "nextafter(8,+inf)"],
            "interior_zero_clip_scenarios": [
                "translation",
                "affine",
                "appearance",
                "combined",
            ],
            "interior_zero_site_flow_boundary_scenarios": [
                "translation",
                "affine",
                "appearance",
                "combined",
            ],
            "interior_zero_gradient_boundary_scenarios": [
                "translation",
                "affine",
                "appearance",
                "combined",
            ],
            "coupled_boundary": {
                "clip_occupancy": 0.0,
                "site_flow_boundary": True,
                "gradient_boundary": True,
                "truth_flow_reaches": 8.0,
                "truth_ayy_reaches": 4.0,
            },
        },
    }


SYNTHETIC_CONFIG: Final = SyntheticConfig.from_payload(_synthetic_config_payload())
SYNTHETIC_CONFIG_SHA256: Final = SYNTHETIC_CONFIG.sha256


def validate_synthetic_config(config: SyntheticConfig) -> None:
    """Reject any canonical payload that differs from the frozen bound config."""

    if not isinstance(config, SyntheticConfig):
        raise SyntheticV2Error("synthetic config record is invalid")
    if (
        config.canonical_json != SYNTHETIC_CONFIG.canonical_json
        or config.sha256 != SYNTHETIC_CONFIG_SHA256
    ):
        raise SyntheticV2Error("synthetic config differs from the frozen payload")


def declared_exhaustive_dispatch(case: SyntheticCase, arm: v2.Arm) -> tuple[ExhaustiveDispatch, ...]:
    """Construct declared first-row contexts without running exhaustive enumeration."""

    if (case.scenario, arm) not in DECLARED_EXHAUSTIVE_PAIRS or case.truth is None:
        raise SyntheticV2Error("scenario/arm is not a declared exhaustive pair")
    source = case.source[0]
    target = case.target[0]
    truth = cast(v2.StateValues, case.truth.theta)
    return (
        ExhaustiveDispatch(
            case.scenario,
            arm,
            0,
            "full",
            v2.make_full_context(source, target, arm),
            truth,
        ),
        ExhaustiveDispatch(
            case.scenario,
            arm,
            0,
            "p0",
            v2.make_xfit_context(source, target, arm, output_parity=0),
            truth,
        ),
        ExhaustiveDispatch(
            case.scenario,
            arm,
            0,
            "p1",
            v2.make_xfit_context(source, target, arm, output_parity=1),
            truth,
        ),
    )


def mutate_held_target(
    target: np.ndarray, *, output_parity: int, delta: float
) -> np.ndarray:
    """Return a copy with only one held output parity changed."""

    future = np.asarray(target, dtype=np.float64)
    if future.shape != (v1.CHANNELS, v1.NATIVE_SIZE, v1.NATIVE_SIZE):
        raise SyntheticV2Error("mutation target must have shape [3,64,64]")
    if output_parity not in (0, 1) or not math.isfinite(delta):
        raise SyntheticV2Error("mutation parity or delta is invalid")
    output = future.copy()
    held = v1.GEOMETRY.parities == output_parity
    coords = v1.GEOMETRY.coords[held].astype(int)
    output[:, coords[:, 0], coords[:, 1]] += delta
    return output


def mutation_context_pair(
    case: SyntheticCase,
    *,
    row: int,
    arm: v2.Arm,
    output_parity: int,
    delta: float,
) -> tuple[v2.FitContext, v2.FitContext]:
    """Build original/mutated leakage contexts without fitting either context."""

    if not 0 <= row < calibration.SYNTHETIC_ROWS:
        raise SyntheticV2Error("mutation row is outside [0,5]")
    original = v2.make_xfit_context(
        case.source[row], case.target[row], arm, output_parity=output_parity
    )
    mutated_target = mutate_held_target(
        case.target[row], output_parity=output_parity, delta=delta
    )
    mutated = v2.make_xfit_context(
        case.source[row], mutated_target, arm, output_parity=output_parity
    )
    return original, mutated


@dataclass(frozen=True, slots=True)
class DeclaredMutationDispatch:
    """One frozen first-row held-target mutation context pair."""

    scenario: Scenario
    arm: v2.Arm
    row: int
    output_parity: int
    delta: float
    original: v2.FitContext
    mutated: v2.FitContext


def declared_mutation_dispatch(
    case: SyntheticCase, arm: v2.Arm
) -> tuple[DeclaredMutationDispatch, ...]:
    """Build the frozen mutation contexts without fitting or scoring them."""

    if (case.scenario, arm) not in DECLARED_MUTATION_PAIRS:
        raise SyntheticV2Error("scenario/arm is not a declared mutation pair")
    output: list[DeclaredMutationDispatch] = []
    for parity in MUTATION_OUTPUT_PARITIES:
        original, mutated = mutation_context_pair(
            case,
            row=0,
            arm=arm,
            output_parity=parity,
            delta=MUTATION_DELTA,
        )
        output.append(
            DeclaredMutationDispatch(
                scenario=case.scenario,
                arm=arm,
                row=0,
                output_parity=parity,
                delta=MUTATION_DELTA,
                original=original,
                mutated=mutated,
            )
        )
    return tuple(output)
