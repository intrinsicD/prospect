"""MM-011-owned pre-real repeated-operator synthetic controls."""

from __future__ import annotations

import math
from typing import Final, cast

import numpy as np

from bench.multimodal_causal_assay import predictor, records
from bench.multimodal_mechanism_diagnostics import geometry_v22 as geometry
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

SCHEMA_VERSION: Final = "mm011-synthetic-controls-v1"
SEEDS: Final = (990_900, 990_901, 990_902)
INDEPENDENT_FUTURE_SEEDS: Final = (990_910, 990_911, 990_912)
PERMUTATION_SEED: Final = 990_919
ALL_SEEDS: Final = (*SEEDS, *INDEPENDENT_FUTURE_SEEDS, PERMUTATION_SEED)
SCENARIOS: Final = (
    "translation",
    "affine",
    "appearance",
    "combined",
    "stationary",
    "independent",
    "constant_target",
    "coupled_boundary",
)
_POSITIVE_ARM: Final = {
    "translation": "affine",
    "affine": "affine",
    "appearance": "appearance",
    "combined": "combined",
}


class SyntheticControlError(ValueError):
    """Raised when a pre-real MM-011 synthetic expectation fails."""


def _sse(left: np.ndarray, right: np.ndarray) -> float:
    residual = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    value = float(np.sum(residual * residual, dtype=np.float64))
    if not math.isfinite(value):
        raise SyntheticControlError("synthetic SSE is nonfinite")
    return value


def _operator_record(operator: predictor.FullOperatorResult) -> dict[str, records.JsonValue]:
    return {
        "biases": [float(value) for value in operator.biases],
        "forecast_sha256": operator.forecast_sha256,
        "gains": [float(value) for value in operator.gains],
        "history_sha256": operator.history_reconstruction_sha256,
        "parameters": [float(value) for value in operator.parameters],
    }


def _positive(
    scenario: str,
    seed: int,
    config_sha256: str,
) -> tuple[dict[str, records.JsonValue], predictor.SourcePairResult, synthetic.SyntheticCase]:
    case = synthetic.generate_case(cast(synthetic.Scenario, scenario), seed=seed)
    if case.truth is None:
        raise SyntheticControlError("positive scenario lacks transform truth")
    result = predictor.fit_source_pair(case.source[0], case.target[0], config_sha256=config_sha256)
    arm = cast(predictor.Arm, _POSITIVE_ARM[scenario])
    operator = result.operator(arm)
    expected = predictor.apply_operator_once(
        case.target[0],
        case.truth.theta_array(),
        case.truth.gain_array(),
        case.truth.bias_array(),
    )
    persistence = geometry.sample_scalar(case.target[0], 0, geometry.FULL_MASK)
    current_target = persistence
    previous_target = geometry.sample_scalar(case.source[0], 0, geometry.FULL_MASK)
    forecast_sse = _sse(operator.forecast, expected)
    persistence_sse = _sse(persistence, expected)
    replayed = predictor.apply_operator_once(case.target[0], operator.parameters, operator.gains, operator.biases)
    if (
        not np.array_equal(operator.parameters, case.truth.theta_array())
        or not np.allclose(operator.gains, case.truth.gain_array(), rtol=0.0, atol=1e-12)
        or not np.allclose(operator.biases, case.truth.bias_array(), rtol=0.0, atol=1e-12)
        or not np.array_equal(operator.forecast, replayed)
        or not np.allclose(operator.forecast, expected, rtol=0.0, atol=1e-12)
        or not persistence_sse > 1e-4 * geometry.CHANNELS * geometry.SITE_COUNT
        or 1.25 * forecast_sse > persistence_sse
    ):
        raise SyntheticControlError(f"{scenario}/{seed} repeated-operator positive failed")

    checker = predictor.fit_checkerboard_history(case.source[0], case.target[0], config_sha256=config_sha256)
    shuffled = predictor.fit_source_pair(case.source[3], case.target[0], config_sha256=config_sha256)
    shuffled_checker = predictor.fit_checkerboard_history(case.source[3], case.target[0], config_sha256=config_sha256)
    reverse = predictor.fit_source_pair(case.target[0], case.source[0], config_sha256=config_sha256)
    reverse_operator = reverse.operator(arm)
    reverse_forecast = predictor.apply_operator_once(
        case.target[0],
        reverse_operator.parameters,
        reverse_operator.gains,
        reverse_operator.biases,
    )
    velocity = 2.0 * current_target - previous_target
    deranged_future = predictor.apply_operator_once(
        case.target[3],
        case.truth.theta_array(),
        case.truth.gain_array(),
        case.truth.bias_array(),
    )
    errors = {
        "i": _sse(previous_target, current_target),
        "a": _sse(checker.arm(arm).history_reconstruction, current_target),
        "q": _sse(shuffled_checker.arm(arm).history_reconstruction, current_target),
        "u": _sse(checker.bias_only.history_reconstruction, current_target),
        "p": persistence_sse,
        "c": forecast_sse,
        "h": _sse(shuffled.operator(arm).forecast, expected),
        "r": _sse(reverse_forecast, expected),
        "z": _sse(velocity, expected),
        "d": _sse(operator.forecast, deranged_future),
        "pd": _sse(persistence, deranged_future),
        "b": _sse(result.bias_only.forecast, expected),
        "bd": _sse(result.bias_only.forecast, deranged_future),
    }
    history_gate = (
        errors["i"] > 1e-4 * geometry.CHANNELS * geometry.SITE_COUNT
        and 1.25 * errors["a"] <= errors["i"]
        and 1.10 * errors["a"] <= errors["q"]
        and errors["u"] > 0.0
        and 1.25 * errors["a"] <= errors["u"]
    )
    future_gate = (
        errors["p"] > 1e-4 * geometry.CHANNELS * geometry.SITE_COUNT
        and 1.25 * errors["c"] <= errors["p"]
        and all(1.10 * errors["c"] <= errors[name] for name in ("h", "r", "z", "d"))
        and errors["b"] > 0.0
        and 1.25 * errors["c"] <= errors["b"]
    )
    directional = (
        errors["a"] < errors["i"]
        and errors["a"] < errors["u"]
        and errors["c"] < errors["p"]
        and errors["c"] < errors["b"]
    )
    if not history_gate or not future_gate or not directional:
        raise SyntheticControlError(f"{scenario}/{seed} complete causal predicates failed")
    return (
        {
            "arm": arm,
            "complete_predicates": {
                "directional": directional,
                "future": future_gate,
                "history": history_gate,
                "joint": history_gate and future_gate,
            },
            "errors": {name: value for name, value in errors.items()},
            "forecast_sse": forecast_sse,
            "operator": _operator_record(operator),
            "persistence_sse": persistence_sse,
            "scenario": scenario,
            "seed": seed,
        },
        result,
        case,
    )


def _stationary(seed: int, config_sha256: str) -> dict[str, records.JsonValue]:
    case = synthetic.generate_case("stationary", seed=seed)
    result = predictor.fit_source_pair(case.source[0], case.target[0], config_sha256=config_sha256)
    target = geometry.sample_scalar(case.target[0], 0, geometry.FULL_MASK)
    errors = {
        arm: _sse(result.operator(cast(predictor.Arm, arm)).forecast, target)
        for arm in ("affine", "appearance", "combined")
    }
    persistence = _sse(result.baselines.persistence, target)
    if persistence != 0.0 or any(value > 1e-24 for value in errors.values()):
        raise SyntheticControlError("stationary non-vacuity fixture differs")
    json_errors: dict[str, records.JsonValue] = {name: value for name, value in errors.items()}
    return {
        "arm_sse": json_errors,
        "eligible": False,
        "persistence_sse": persistence,
        "scenario": "stationary",
        "seed": seed,
    }


def _independent(
    seed: int, future_seed: int, config_sha256: str
) -> tuple[
    dict[str, records.JsonValue],
    dict[str, float],
    float,
    dict[str, float],
    float,
    float,
    float,
]:
    case = synthetic.generate_case("independent", seed=seed)
    future_case = synthetic.generate_case("independent", seed=future_seed)
    result = predictor.fit_source_pair(case.source[0], case.target[0], config_sha256=config_sha256)
    checker = predictor.fit_checkerboard_history(case.source[0], case.target[0], config_sha256=config_sha256)
    future = geometry.sample_scalar(future_case.target[0], 0, geometry.FULL_MASK)
    current = geometry.sample_scalar(case.target[0], 0, geometry.FULL_MASK)
    previous = geometry.sample_scalar(case.source[0], 0, geometry.FULL_MASK)
    persistence_sse = _sse(result.baselines.persistence, future)
    ratios: dict[str, records.JsonValue] = {}
    errors: dict[str, float] = {}
    for arm in ("affine", "appearance", "combined"):
        forecast_sse = _sse(result.operator(cast(predictor.Arm, arm)).forecast, future)
        errors[arm] = forecast_sse
        ratios[arm] = forecast_sse / persistence_sse
    history_errors = {
        arm: _sse(
            checker.arm(cast(predictor.Arm, arm)).history_reconstruction,
            current,
        )
        for arm in ("affine", "appearance", "combined")
    }
    identity_sse = _sse(previous, current)
    bias_sse = _sse(checker.bias_only.history_reconstruction, current)
    future_bias_sse = _sse(result.bias_only.forecast, future)
    return (
        {
            "forecast_sse": {arm: value for arm, value in errors.items()},
            "distinct_future_fixture": True,
            "future_bias_sse": future_bias_sse,
            "history_bias_sse": bias_sse,
            "history_identity_sse": identity_sse,
            "history_xfit_sse": {arm: value for arm, value in history_errors.items()},
            "persistence_sse": persistence_sse,
            "ratios_to_persistence": ratios,
            "scenario": "independent",
            "seed": seed,
            "future_seed": future_seed,
        },
        errors,
        persistence_sse,
        history_errors,
        identity_sse,
        bias_sse,
        future_bias_sse,
    )


def _constant_target(seed: int, config_sha256: str) -> dict[str, records.JsonValue]:
    case = synthetic.generate_case("constant_target", seed=seed)
    result = predictor.fit_source_pair(case.source[0], case.target[0], config_sha256=config_sha256)
    bias = result.bias_only.forecast
    errors: dict[str, records.JsonValue] = {}
    for arm in ("affine", "appearance", "combined"):
        error = _sse(result.operator(cast(predictor.Arm, arm)).forecast, bias)
        errors[arm] = error
        if error > 1e-22:
            raise SyntheticControlError(f"constant target {arm} beat/differed from bias-only")
    return {"bias_match_sse": errors, "scenario": "constant_target", "seed": seed}


def _boundary(seed: int, config_sha256: str) -> dict[str, records.JsonValue]:
    case = synthetic.generate_case("coupled_boundary", seed=seed)
    result = predictor.fit_source_pair(case.source[0], case.target[0], config_sha256=config_sha256)
    bounded: dict[str, records.JsonValue] = {}
    for arm in ("affine", "combined"):
        parameters = result.operator(cast(predictor.Arm, arm)).parameters
        value = bool(np.any(np.isin(parameters[:2], (-8.0, 8.0))) or np.any(np.isin(parameters[2:], (-4.0, 4.0))))
        bounded[arm] = value
        if not value:
            raise SyntheticControlError(f"boundary scenario did not tag {arm}")
    return {"bounded": bounded, "scenario": "coupled_boundary", "seed": seed}


def run_controls(*, config_sha256: str, protocol_sha256: str) -> dict[str, records.JsonValue]:
    """Execute the complete exposed, nonreserved pre-real control panel."""

    records.require_sha256(config_sha256, "config SHA-256")
    records.require_sha256(protocol_sha256, "protocol SHA-256")
    if set(ALL_SEEDS).intersection(synthetic.FROZEN_SEED_MAP.values()):
        raise SyntheticControlError("MM-011 synthetic seeds overlap MM-008 reserved seeds")
    positive_rows: list[records.JsonValue] = []
    reversal_rows: list[records.JsonValue] = []
    for scenario in _POSITIVE_ARM:
        for seed in SEEDS:
            row, result, case = _positive(scenario, seed, config_sha256)
            positive_rows.append(row)
            if scenario in ("affine", "appearance"):
                arm = cast(predictor.Arm, _POSITIVE_ARM[scenario])
                previous = geometry.sample_scalar(case.source[0], 0, geometry.FULL_MASK)
                persistence_sse = _sse(result.baselines.persistence, previous)
                repeated_sse = _sse(result.operator(arm).forecast, previous)
                if 1.25 * repeated_sse <= persistence_sse:
                    raise SyntheticControlError(f"{scenario}/{seed} reversal crossed the positive margin")
                reversal_rows.append(
                    {
                        "arm": arm,
                        "persistence_sse": persistence_sse,
                        "repeated_sse": repeated_sse,
                        "scenario": f"{scenario}_reversal",
                        "seed": seed,
                    }
                )
    stationary: list[records.JsonValue] = [_stationary(seed, config_sha256) for seed in SEEDS]
    independent: list[records.JsonValue] = []
    independent_errors = {arm: 0.0 for arm in ("affine", "appearance", "combined")}
    independent_persistence = 0.0
    independent_history = {arm: 0.0 for arm in ("affine", "appearance", "combined")}
    independent_identity = 0.0
    independent_bias = 0.0
    independent_future_bias = 0.0
    for seed, future_seed in zip(SEEDS, INDEPENDENT_FUTURE_SEEDS, strict=True):
        (
            record,
            errors,
            persistence_sse,
            history_errors,
            identity_sse,
            bias_sse,
            future_bias_sse,
        ) = _independent(seed, future_seed, config_sha256)
        independent.append(record)
        independent_persistence += persistence_sse
        independent_identity += identity_sse
        independent_bias += bias_sse
        independent_future_bias += future_bias_sse
        for family_name, value in errors.items():
            independent_errors[family_name] += value
        for family_name, value in history_errors.items():
            independent_history[family_name] += value
    independent_branches: dict[str, records.JsonValue] = {}
    for family_name, value in independent_history.items():
        identity_credit = 1.25 * value <= independent_identity
        bias_credit = 1.25 * value <= independent_bias
        future_credit = (
            1.25 * independent_errors[family_name] <= independent_persistence
            and 1.25 * independent_errors[family_name] <= independent_future_bias
        )
        historical_credit = identity_credit and bias_credit
        if historical_credit or future_credit:
            raise SyntheticControlError(f"aggregate independent fixture crossed a source-use margin for {family_name}")
        independent_branches[family_name] = {
            "expected_branch": "tested_family_identifiability_failure",
            "future_source_credit": future_credit,
            "historical_source_credit": historical_credit,
            "joint_support": False,
        }
    constant: list[records.JsonValue] = [_constant_target(seed, config_sha256) for seed in SEEDS]
    boundary: list[records.JsonValue] = [_boundary(seed, config_sha256) for seed in SEEDS]

    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    current = np.ascontiguousarray(rng.normal(size=(3, 64, 64)), dtype=np.float64)
    parameters = np.asarray((-4.0, 4.0, 0.0, 2.0, -2.0, 0.0), dtype=np.float64)
    gains = np.asarray((1.2, 0.8, 1.4), dtype=np.float64)
    biases = np.asarray((0.3, -0.2, 0.1), dtype=np.float64)
    original = predictor.apply_operator_once(current, parameters, gains, biases)
    permutation = np.asarray((2, 0, 1), dtype=np.intp)
    permuted = predictor.apply_operator_once(
        np.ascontiguousarray(current[permutation]),
        parameters,
        gains[permutation],
        biases[permutation],
    )
    if not np.array_equal(permuted, original[permutation]):
        raise SyntheticControlError("channel permutation metamorphic control failed")

    evidence: dict[str, records.JsonValue] = {
        "boundary": boundary,
        "channel_permutation_exact": True,
        "config_sha256": config_sha256,
        "constant_target": constant,
        "independent": independent,
        "independent_aggregate_branches": independent_branches,
        "positive": positive_rows,
        "protocol_sha256": protocol_sha256,
        "reserved_seed_overlap": False,
        "reversal": reversal_rows,
        "schema_version": SCHEMA_VERSION,
        "seeds": list(ALL_SEEDS),
        "stationary": stationary,
    }
    evidence["evidence_sha256"] = records.canonical_json_sha256(evidence, protocol_sha256=protocol_sha256)
    return evidence


def _record(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise SyntheticControlError(f"{label} schema differs")
    return cast(dict[str, object], value)


def _rows(value: object, count: int, label: str) -> list[object]:
    if type(value) is not list or len(value) != count:
        raise SyntheticControlError(f"{label} row census differs")
    return cast(list[object], value)


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise SyntheticControlError(f"{label} must be a finite nonnegative float")
    return value


def _float_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    rows = _rows(value, length, label)
    output: list[float] = []
    for index, item in enumerate(rows):
        if type(item) is not float or not math.isfinite(item):
            raise SyntheticControlError(f"{label}[{index}] must be a finite float")
        output.append(item)
    return tuple(output)


def _metric_map(value: object, names: tuple[str, ...], label: str) -> dict[str, float]:
    record = _record(value, set(names), label)
    return {name: _finite_nonnegative(record[name], f"{label}/{name}") for name in names}


def validate_control_evidence(
    value: object,
    *,
    config_sha256: str,
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    """Validate the complete stored control semantics without fitting again."""

    try:
        records.require_sha256(config_sha256, "config SHA-256")
        records.require_sha256(protocol_sha256, "protocol SHA-256")
    except records.RecordValidationError as error:
        raise SyntheticControlError("synthetic evidence expected bindings differ") from error
    top_keys = {
        "boundary",
        "channel_permutation_exact",
        "config_sha256",
        "constant_target",
        "evidence_sha256",
        "independent",
        "independent_aggregate_branches",
        "positive",
        "protocol_sha256",
        "reserved_seed_overlap",
        "reversal",
        "schema_version",
        "seeds",
        "stationary",
    }
    evidence = _record(value, top_keys, "synthetic evidence")
    if (
        evidence["schema_version"] != SCHEMA_VERSION
        or evidence["config_sha256"] != config_sha256
        or evidence["protocol_sha256"] != protocol_sha256
        or evidence["seeds"] != list(ALL_SEEDS)
        or evidence["reserved_seed_overlap"] is not False
        or evidence["channel_permutation_exact"] is not True
    ):
        raise SyntheticControlError("synthetic evidence header or permutation result differs")
    try:
        claimed_digest = records.require_sha256(evidence["evidence_sha256"], "synthetic evidence SHA-256")
        body = cast(
            dict[str, records.JsonValue],
            {name: item for name, item in evidence.items() if name != "evidence_sha256"},
        )
        expected_digest = records.canonical_json_sha256(body, protocol_sha256=protocol_sha256)
    except records.RecordValidationError as error:
        raise SyntheticControlError("synthetic evidence digest grammar differs") from error
    if claimed_digest != expected_digest:
        raise SyntheticControlError("synthetic evidence digest differs")

    error_names = ("i", "a", "q", "u", "p", "c", "h", "r", "z", "d", "pd", "b", "bd")
    positive_specs = tuple((scenario, seed, _POSITIVE_ARM[scenario]) for scenario in _POSITIVE_ARM for seed in SEEDS)
    for raw, (scenario, seed, arm) in zip(
        _rows(evidence["positive"], len(positive_specs), "positive"), positive_specs, strict=True
    ):
        row = _record(
            raw,
            {
                "arm",
                "complete_predicates",
                "errors",
                "forecast_sse",
                "operator",
                "persistence_sse",
                "scenario",
                "seed",
            },
            f"positive/{scenario}/{seed}",
        )
        if (row["scenario"], row["seed"], row["arm"]) != (scenario, seed, arm):
            raise SyntheticControlError(f"positive/{scenario}/{seed} identity differs")
        predicates = _record(
            row["complete_predicates"],
            {"directional", "future", "history", "joint"},
            f"positive/{scenario}/{seed}/predicates",
        )
        if predicates != {"directional": True, "future": True, "history": True, "joint": True}:
            raise SyntheticControlError(f"positive/{scenario}/{seed} declared predicates differ")
        errors = _metric_map(row["errors"], error_names, f"positive/{scenario}/{seed}/errors")
        if (
            _finite_nonnegative(row["forecast_sse"], "positive forecast SSE") != errors["c"]
            or _finite_nonnegative(row["persistence_sse"], "positive persistence SSE") != errors["p"]
            or not errors["i"] > 1e-4 * geometry.CHANNELS * geometry.SITE_COUNT
            or not errors["p"] > 1e-4 * geometry.CHANNELS * geometry.SITE_COUNT
            or 1.25 * errors["a"] > errors["i"]
            or 1.10 * errors["a"] > errors["q"]
            or errors["u"] <= 0.0
            or 1.25 * errors["a"] > errors["u"]
            or 1.25 * errors["c"] > errors["p"]
            or any(1.10 * errors["c"] > errors[name] for name in ("h", "r", "z", "d"))
            or errors["b"] <= 0.0
            or 1.25 * errors["c"] > errors["b"]
            or not (errors["a"] < errors["i"] and errors["a"] < errors["u"])
            or not (errors["c"] < errors["p"] and errors["c"] < errors["b"])
        ):
            raise SyntheticControlError(f"positive/{scenario}/{seed} numeric predicates differ")
        operator = _record(
            row["operator"],
            {"biases", "forecast_sha256", "gains", "history_sha256", "parameters"},
            f"positive/{scenario}/{seed}/operator",
        )
        _float_vector(operator["parameters"], 6, "positive operator parameters")
        _float_vector(operator["gains"], 3, "positive operator gains")
        _float_vector(operator["biases"], 3, "positive operator biases")
        try:
            records.require_sha256(operator["forecast_sha256"], "positive forecast SHA-256")
            records.require_sha256(operator["history_sha256"], "positive history SHA-256")
        except records.RecordValidationError as error:
            raise SyntheticControlError("positive operator digest differs") from error

    reversal_specs = tuple(
        (scenario, seed, _POSITIVE_ARM[scenario])
        for scenario in ("affine", "appearance")
        for seed in SEEDS
    )
    for raw, (scenario, seed, arm) in zip(
        _rows(evidence["reversal"], len(reversal_specs), "reversal"), reversal_specs, strict=True
    ):
        row = _record(
            raw,
            {"arm", "persistence_sse", "repeated_sse", "scenario", "seed"},
            f"reversal/{scenario}/{seed}",
        )
        persistence = _finite_nonnegative(row["persistence_sse"], "reversal persistence SSE")
        repeated = _finite_nonnegative(row["repeated_sse"], "reversal repeated SSE")
        if (
            (row["scenario"], row["seed"], row["arm"]) != (f"{scenario}_reversal", seed, arm)
            or 1.25 * repeated <= persistence
        ):
            raise SyntheticControlError(f"reversal/{scenario}/{seed} branch differs")

    for raw, seed in zip(_rows(evidence["stationary"], len(SEEDS), "stationary"), SEEDS, strict=True):
        row = _record(raw, {"arm_sse", "eligible", "persistence_sse", "scenario", "seed"}, f"stationary/{seed}")
        errors = _metric_map(row["arm_sse"], ("affine", "appearance", "combined"), f"stationary/{seed}")
        if (
            (row["scenario"], row["seed"], row["eligible"]) != ("stationary", seed, False)
            or _finite_nonnegative(row["persistence_sse"], "stationary persistence SSE") != 0.0
            or any(item > 1e-24 for item in errors.values())
        ):
            raise SyntheticControlError(f"stationary/{seed} branch differs")

    aggregate_forecast = {arm: 0.0 for arm in ("affine", "appearance", "combined")}
    aggregate_history = {arm: 0.0 for arm in ("affine", "appearance", "combined")}
    aggregate_persistence = aggregate_identity = aggregate_history_bias = aggregate_future_bias = 0.0
    independent_specs = tuple(zip(SEEDS, INDEPENDENT_FUTURE_SEEDS, strict=True))
    for raw, (seed, future_seed) in zip(
        _rows(evidence["independent"], len(independent_specs), "independent"), independent_specs, strict=True
    ):
        row = _record(
            raw,
            {
                "distinct_future_fixture",
                "forecast_sse",
                "future_bias_sse",
                "future_seed",
                "history_bias_sse",
                "history_identity_sse",
                "history_xfit_sse",
                "persistence_sse",
                "ratios_to_persistence",
                "scenario",
                "seed",
            },
            f"independent/{seed}",
        )
        if (row["scenario"], row["seed"], row["future_seed"], row["distinct_future_fixture"]) != (
            "independent",
            seed,
            future_seed,
            True,
        ):
            raise SyntheticControlError(f"independent/{seed} identity differs")
        forecast = _metric_map(row["forecast_sse"], ("affine", "appearance", "combined"), "independent forecast")
        history = _metric_map(row["history_xfit_sse"], ("affine", "appearance", "combined"), "independent history")
        ratios = _metric_map(
            row["ratios_to_persistence"], ("affine", "appearance", "combined"), "independent ratios"
        )
        persistence = _finite_nonnegative(row["persistence_sse"], "independent persistence SSE")
        identity = _finite_nonnegative(row["history_identity_sse"], "independent identity SSE")
        history_bias = _finite_nonnegative(row["history_bias_sse"], "independent history bias SSE")
        future_bias = _finite_nonnegative(row["future_bias_sse"], "independent future bias SSE")
        if persistence <= 0.0 or any(ratios[arm] != forecast[arm] / persistence for arm in forecast):
            raise SyntheticControlError(f"independent/{seed} ratio evidence differs")
        aggregate_persistence += persistence
        aggregate_identity += identity
        aggregate_history_bias += history_bias
        aggregate_future_bias += future_bias
        for arm in forecast:
            aggregate_forecast[arm] += forecast[arm]
            aggregate_history[arm] += history[arm]

    branches = _record(
        evidence["independent_aggregate_branches"],
        {"affine", "appearance", "combined"},
        "independent aggregate branches",
    )
    for arm in ("affine", "appearance", "combined"):
        historical_credit = (
            1.25 * aggregate_history[arm] <= aggregate_identity
            and 1.25 * aggregate_history[arm] <= aggregate_history_bias
        )
        future_credit = (
            1.25 * aggregate_forecast[arm] <= aggregate_persistence
            and 1.25 * aggregate_forecast[arm] <= aggregate_future_bias
        )
        branch = _record(
            branches[arm],
            {"expected_branch", "future_source_credit", "historical_source_credit", "joint_support"},
            f"independent aggregate/{arm}",
        )
        expected_branch = {
            "expected_branch": "tested_family_identifiability_failure",
            "future_source_credit": future_credit,
            "historical_source_credit": historical_credit,
            "joint_support": False,
        }
        if historical_credit or future_credit or branch != expected_branch:
            raise SyntheticControlError(f"independent aggregate/{arm} branch differs")

    for raw, seed in zip(_rows(evidence["constant_target"], len(SEEDS), "constant target"), SEEDS, strict=True):
        row = _record(raw, {"bias_match_sse", "scenario", "seed"}, f"constant target/{seed}")
        errors = _metric_map(row["bias_match_sse"], ("affine", "appearance", "combined"), "constant target")
        if (row["scenario"], row["seed"]) != ("constant_target", seed) or any(
            item > 1e-22 for item in errors.values()
        ):
            raise SyntheticControlError(f"constant target/{seed} branch differs")

    for raw, seed in zip(_rows(evidence["boundary"], len(SEEDS), "boundary"), SEEDS, strict=True):
        row = _record(raw, {"bounded", "scenario", "seed"}, f"boundary/{seed}")
        bounded = _record(row["bounded"], {"affine", "combined"}, f"boundary/{seed}/bounded")
        if (row["scenario"], row["seed"], bounded) != (
            "coupled_boundary",
            seed,
            {"affine": True, "combined": True},
        ):
            raise SyntheticControlError(f"boundary/{seed} branch differs")
    return cast(dict[str, records.JsonValue], evidence)


__all__ = [
    "ALL_SEEDS",
    "INDEPENDENT_FUTURE_SEEDS",
    "PERMUTATION_SEED",
    "SCENARIOS",
    "SCHEMA_VERSION",
    "SEEDS",
    "SyntheticControlError",
    "run_controls",
    "validate_control_evidence",
]
