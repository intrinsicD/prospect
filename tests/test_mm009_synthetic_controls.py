from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

from bench.multimodal_causal_diagnostics import records, synthetic_controls

PROTOCOL_SHA256 = "1" * 64
CONFIG_SHA256 = "2" * 64
ARMS = ("affine", "appearance", "combined")


def _rehash(value: dict[str, Any]) -> None:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = records.canonical_json_sha256(
        cast(dict[str, records.JsonValue], value),
        protocol_sha256=PROTOCOL_SHA256,
    )


def _valid_evidence() -> dict[str, Any]:
    errors = {
        "i": 10.0,
        "a": 1.0,
        "q": 2.0,
        "u": 2.0,
        "p": 10.0,
        "c": 1.0,
        "h": 2.0,
        "r": 2.0,
        "z": 2.0,
        "d": 2.0,
        "pd": 3.0,
        "b": 2.0,
        "bd": 3.0,
    }
    positives = []
    positive_arms = {
        "translation": "affine",
        "affine": "affine",
        "appearance": "appearance",
        "combined": "combined",
    }
    for scenario, arm in positive_arms.items():
        for seed in synthetic_controls.SEEDS:
            positives.append(
                {
                    "arm": arm,
                    "complete_predicates": {
                        "directional": True,
                        "future": True,
                        "history": True,
                        "joint": True,
                    },
                    "errors": dict(errors),
                    "forecast_sse": 1.0,
                    "operator": {
                        "biases": [0.0, 0.0, 0.0],
                        "forecast_sha256": "3" * 64,
                        "gains": [1.0, 1.0, 1.0],
                        "history_sha256": "4" * 64,
                        "parameters": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "persistence_sse": 10.0,
                    "scenario": scenario,
                    "seed": seed,
                }
            )
    reversal = [
        {
            "arm": positive_arms[scenario],
            "persistence_sse": 1.0,
            "repeated_sse": 1.0,
            "scenario": f"{scenario}_reversal",
            "seed": seed,
        }
        for scenario in ("affine", "appearance")
        for seed in synthetic_controls.SEEDS
    ]
    stationary = [
        {
            "arm_sse": {arm: 0.0 for arm in ARMS},
            "eligible": False,
            "persistence_sse": 0.0,
            "scenario": "stationary",
            "seed": seed,
        }
        for seed in synthetic_controls.SEEDS
    ]
    independent = [
        {
            "distinct_future_fixture": True,
            "forecast_sse": {arm: 10.0 for arm in ARMS},
            "future_bias_sse": 1.0,
            "future_seed": future_seed,
            "history_bias_sse": 1.0,
            "history_identity_sse": 1.0,
            "history_xfit_sse": {arm: 10.0 for arm in ARMS},
            "persistence_sse": 1.0,
            "ratios_to_persistence": {arm: 10.0 for arm in ARMS},
            "scenario": "independent",
            "seed": seed,
        }
        for seed, future_seed in zip(
            synthetic_controls.SEEDS,
            synthetic_controls.INDEPENDENT_FUTURE_SEEDS,
            strict=True,
        )
    ]
    evidence: dict[str, Any] = {
        "boundary": [
            {
                "bounded": {"affine": True, "combined": True},
                "scenario": "coupled_boundary",
                "seed": seed,
            }
            for seed in synthetic_controls.SEEDS
        ],
        "channel_permutation_exact": True,
        "config_sha256": CONFIG_SHA256,
        "constant_target": [
            {
                "bias_match_sse": {arm: 0.0 for arm in ARMS},
                "scenario": "constant_target",
                "seed": seed,
            }
            for seed in synthetic_controls.SEEDS
        ],
        "independent": independent,
        "independent_aggregate_branches": {
            arm: {
                "expected_branch": "tested_family_identifiability_failure",
                "future_source_credit": False,
                "historical_source_credit": False,
                "joint_support": False,
            }
            for arm in ARMS
        },
        "positive": positives,
        "protocol_sha256": PROTOCOL_SHA256,
        "reserved_seed_overlap": False,
        "reversal": reversal,
        "schema_version": synthetic_controls.SCHEMA_VERSION,
        "seeds": list(synthetic_controls.ALL_SEEDS),
        "stationary": stationary,
    }
    _rehash(evidence)
    return evidence


def test_control_evidence_rejects_semantic_forgery_even_with_recomputed_digest() -> None:
    evidence = _valid_evidence()
    assert synthetic_controls.validate_control_evidence(
        evidence,
        config_sha256=CONFIG_SHA256,
        protocol_sha256=PROTOCOL_SHA256,
    ) == evidence

    mutations: list[tuple[str, dict[str, Any]]] = []
    predicate = deepcopy(evidence)
    predicate["positive"][0]["complete_predicates"]["future"] = False
    mutations.append(("declared predicates", predicate))
    numeric = deepcopy(evidence)
    numeric["positive"][0]["errors"]["c"] = 9.0
    mutations.append(("numeric predicates", numeric))
    branch = deepcopy(evidence)
    branch["independent_aggregate_branches"]["affine"]["future_source_credit"] = True
    mutations.append(("branch differs", branch))
    extra = deepcopy(evidence)
    extra["extra"] = True
    mutations.append(("schema differs", extra))
    for message, mutated in mutations:
        _rehash(mutated)
        with pytest.raises(synthetic_controls.SyntheticControlError, match=message):
            synthetic_controls.validate_control_evidence(
                mutated,
                config_sha256=CONFIG_SHA256,
                protocol_sha256=PROTOCOL_SHA256,
            )
