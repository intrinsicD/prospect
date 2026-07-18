#!/usr/bin/env python3
"""Dependency-free integrity checks for the sealed WM-001 evidence package.

This is deliberately not the experiment runner.  It verifies the immutable
scientific protocol now, an implementation binding before a formal launch, and
the causal/evidence envelope of a result after execution.  Metric recomputation
from tensors and episode traces belongs to the later independent audit.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import struct
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = HERE / "protocol.json"
SEAL_PATH = HERE / "SEALED_PROTOCOL.sha256"
RESULT_SCHEMA_PATH = HERE / "schemas" / "raw-result.schema.json"
BINDING_SCHEMA_PATH = HERE / "schemas" / "formal-binding.schema.json"

FORMAL_SEEDS = (
    339970590,
    474769515,
    550273937,
    438984650,
    2732731971,
    2253809848,
    2206960337,
    3506881479,
)
DEVELOPMENT_SEEDS = (2439054559, 3246851043)
COVERAGE_SEMANTICS = "wm001-mixture-pit-binary64-count-v1"
_V130_MASTER_SEEDS = (
    17123296,
    3280610186,
    2725263418,
    3124246399,
    4093604926,
    3908390087,
    3332986400,
    724244869,
    3625750835,
    2671781227,
)
_V130_BOUNDARY_TARGET_F32_HEX = "ac3cdebd"
_V130_BOUNDARY_MEANS_F32_HEX = (
    "8cd85cbb",
    "f032d7bb",
    "d0d5aebc",
    "fcaa09bc",
    "0086a53a",
)
_V130_BOUNDARY_LOG_VARIANCES_F32_HEX = (
    "66b8b3c0",
    "cb11b5c0",
    "d611b2c0",
    "86dcb2c0",
    "9390b2c0",
)
TASK_A = "pendulum_normal_torque"
TASK_B = "pendulum_reversed_torque"
TASK_IRRELEVANT = "independent_phase_oscillator"
INDEPENDENT_OSCILLATOR_SOURCE = "prospect:IndependentPhaseOscillator-v1"
EPISODE_CONTRACTS = frozenset(
    {
        ("collect_a", TASK_A, "collection_random", "cold"),
        ("collect_b", TASK_B, "collection_random", "after_a"),
        (
            "collect_irrelevant",
            TASK_IRRELEVANT,
            "collection_random",
            "cold",
        ),
        (
            "predictive_validation_a",
            TASK_A,
            "validation_random",
            "after_a",
        ),
        (
            "predictive_validation_b",
            TASK_B,
            "validation_random",
            "after_a",
        ),
        (
            "predictive_validation_irrelevant",
            TASK_IRRELEVANT,
            "validation_random",
            "irrelevant",
        ),
        *(
            ("behavior_evaluation_a", TASK_A, condition, condition)
            for condition in (
                "cold",
                "after_a",
                "frozen",
                "corrupted",
                "irrelevant",
                "after_b_replay",
                "after_b_naive",
                "random",
                "oracle",
            )
        ),
        *(
            ("behavior_evaluation_b", TASK_B, condition, condition)
            for condition in (
                "after_a",
                "after_b_replay",
                "after_b_naive",
                "random",
                "oracle",
            )
        ),
    }
)
PREDICTIVE_CONTRACTS = frozenset(
    {
        *(
            ("predictive_validation_a", TASK_A, condition, condition)
            for condition in (
                "cold",
                "after_a",
                "frozen",
                "corrupted",
                "irrelevant",
                "after_b_replay",
                "after_b_naive",
            )
        ),
        *(
            ("predictive_validation_b", TASK_B, condition, condition)
            for condition in (
                "after_a",
                "after_b_replay",
                "after_b_naive",
            )
        ),
        *(
            (
                "predictive_validation_irrelevant",
                TASK_IRRELEVANT,
                condition,
                condition,
            )
            for condition in ("cold", "irrelevant")
        ),
    }
)
FORMAL_EPISODE_CONTRACT_COUNTS = {
    contract: 32 if str(contract[0]).startswith("behavior_evaluation_") else 8 for contract in EPISODE_CONTRACTS
}
FORMAL_PREDICTIVE_CONTRACT_COUNTS = {contract: 1 for contract in PREDICTIVE_CONTRACTS}
COMMITTED_PHASE_SPLITS = {
    "train_a": ("collect_a",),
    "train_a_corrupted": ("collect_a",),
    "train_a_irrelevant": ("collect_irrelevant",),
    "train_b_replay": ("collect_a", "collect_b"),
    "train_b_naive": ("collect_b",),
}
EXPECTED_SEED_COUNTS = {
    "model_initialization": 1,
    "torch_runtime": 1,
    "collection_action": 2,
    "irrelevant_collection_action": 1,
    "predictive_validation_irrelevant_action": 1,
    "predictive_validation_action": 2,
    "random_policy_action": 2,
    "ensemble_bootstrap_a": 5,
    "ensemble_bootstrap_b": 5,
    "minibatch_order_a": 1,
    "minibatch_order_b": 1,
    "corrupted_target_permutation": 1,
    "collect_a_episode": 8,
    "collect_irrelevant_episode": 8,
    "predictive_validation_irrelevant_episode": 8,
    "predictive_validation_a_episode": 8,
    "behavior_evaluation_a_episode": 32,
    "collect_b_episode": 8,
    "predictive_validation_b_episode": 8,
    "behavior_evaluation_b_episode": 32,
    "planner": 1,
}
REQUIRED_PACKAGES = frozenset(
    {
        "python",
        "gymnasium",
        "jsonschema",
        "numpy",
        "torch",
        "torchrl",
        "tensordict",
        "prospect",
    }
)
REQUIRED_COMPONENTS = (
    "world_model",
    "optimizer",
    "model_version_ledger",
    "experience_store",
    "replay_index",
    "replay_sampling_history",
    "update_receipts",
    "agent_runtime",
    "scaling_configuration",
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_accelerator_rng",
    "collection_rng",
    "planner_rng",
)
GATES = tuple(f"K{index}" for index in range(8))
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FORMAL_CONFORMANCE_KEYS = frozenset(
    {
        "schema",
        "environment_id",
        "gymnasium_version",
        "seed",
        "samples_per_task",
        "cases",
        "semantic_parameters",
        "semantic_parameter_absolute_errors",
        "spec_horizon",
        "max_observation_absolute_error",
        "max_reward_absolute_error",
        "planner_dtype",
        "max_planner_observation_absolute_error",
        "max_planner_reward_absolute_error",
        "terminated_or_truncated_cases",
        "observation_atol",
        "reward_atol",
        "planner_observation_atol",
        "planner_reward_atol",
        "passed",
        "report_sha256",
    }
)
FORMAL_CONFORMANCE_PARAMETERS = {
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "dt": 0.05,
    "max_speed": 8.0,
    "max_torque": 2.0,
}
FORMAL_CONFORMANCE_TOLERANCES = {
    "observation_atol": 2e-6,
    "reward_atol": 1e-9,
    "planner_observation_atol": 2e-6,
    "planner_reward_atol": 2e-5,
}


class Violation(ValueError):
    """A protocol, binding, or result integrity violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Violation(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Violation(f"cannot load JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def _validate_json_schema(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    label: str,
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise Violation("jsonschema 4.25.1 is required for binding/result verification") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise Violation(f"{label} schema is invalid: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise Violation(f"{label} violates JSON Schema at {location}: {first.message}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _binding_sibling(path: Path, value: object, field: str) -> Path:
    _require(isinstance(value, str) and value, f"{field} is missing")
    relative = Path(value)
    _require(
        not relative.is_absolute() and len(relative.parts) == 1 and relative.name == value,
        f"{field} must be a safe sibling filename",
    )
    return path.parent / relative


def _require_sha256(value: object, field: str) -> str:
    _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None, f"{field} is not SHA-256")
    return value


def _parse_timestamp(value: object, field: str) -> datetime:
    _require(isinstance(value, str), f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Violation(f"{field} is not an ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, f"{field} must include a UTC offset")
    return parsed


def _verify_implementation_manifest(value: object) -> None:
    """Require the exact complete ordered implementation manifest."""

    from .binding import implementation_files

    expected = implementation_files()
    _require(
        value == expected,
        "binding implementation_files differs from the exact complete ordered live implementation manifest",
    )


def _verify_pendulum_conformance_report(report: object) -> None:
    """Independently enforce the fixed formal Pendulum conformance contract."""

    _require(isinstance(report, dict), "bound Pendulum conformance report is not an object")
    assert isinstance(report, dict)
    _require(
        set(report) == FORMAL_CONFORMANCE_KEYS,
        "bound Pendulum conformance report fields differ from the formal contract",
    )
    _require(
        report.get("schema") == "prospect.wm001.pendulum-conformance.v1",
        "bound Pendulum conformance report has the wrong schema",
    )
    _require(
        report.get("environment_id") == "Pendulum-v1",
        "bound Pendulum conformance report has the wrong environment",
    )
    _require(
        isinstance(report.get("gymnasium_version"), str) and bool(report["gymnasium_version"]),
        "bound Pendulum conformance report has no Gymnasium version",
    )
    _require(
        report.get("seed") == 20260717 and report.get("samples_per_task") == 512 and report.get("cases") == 1024,
        "bound Pendulum conformance report does not contain exactly 512 cases per task from seed 20260717",
    )
    _require(
        report.get("semantic_parameters") == FORMAL_CONFORMANCE_PARAMETERS,
        "bound Pendulum semantic parameters changed",
    )
    _require(
        report.get("semantic_parameter_absolute_errors") == {name: 0.0 for name in FORMAL_CONFORMANCE_PARAMETERS},
        "bound Pendulum semantic parameters differ from Gymnasium",
    )
    _require(
        report.get("spec_horizon") == 200,
        "bound Pendulum horizon changed",
    )
    _require(
        report.get("terminated_or_truncated_cases") == 0,
        "bound Pendulum conformance cases terminated or truncated",
    )
    _require(
        all(report.get(field) == expected for field, expected in FORMAL_CONFORMANCE_TOLERANCES.items()),
        "bound Pendulum conformance tolerances changed",
    )
    _require(
        report.get("planner_dtype") == "float32",
        "bound Pendulum planner conformance dtype changed",
    )
    error_limits = {
        "max_observation_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["observation_atol"],
        "max_reward_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["reward_atol"],
        "max_planner_observation_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["planner_observation_atol"],
        "max_planner_reward_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["planner_reward_atol"],
    }
    for field, limit in error_limits.items():
        value = report.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= limit,
            f"bound Pendulum conformance {field} exceeds its fixed tolerance",
        )
    _require(
        report.get("passed") is True,
        "bound Pendulum conformance report did not pass",
    )
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    _require_sha256(report_sha256, "Pendulum conformance report_sha256")
    _require(
        report_sha256 == _canonical_sha256(body),
        "bound Pendulum conformance self-hash changed",
    )


def _binary64_pit_covered(pit: float) -> bool:
    _require(math.isfinite(pit), "coverage conformance PIT is non-finite")
    numerator, denominator = pit.as_integer_ratio()
    return 20 * numerator >= denominator and 20 * numerator <= 19 * denominator


def _verify_coverage_conformance_report(report: object) -> None:
    _require(isinstance(report, dict), "bound coverage conformance report is not an object")
    _require(
        report.get("schema") == "prospect.wm001.coverage-conformance.v1"
        and report.get("semantics_id") == COVERAGE_SEMANTICS
        and report.get("python_executable") == sys.executable
        and report.get("python_implementation") == platform.python_implementation() == "CPython"
        and report.get("python_version") == platform.python_version()
        and report.get("platform") == platform.platform()
        and report.get("machine") == platform.machine(),
        "bound coverage conformance runtime identity changed",
    )
    rows = report.get("cases")
    _require(isinstance(rows, list), "bound coverage conformance cases are missing")
    direct_expected = (
        ("lower-binary64", 0.05, True),
        ("lower-predecessor", math.nextafter(0.05, -math.inf), False),
        ("lower-successor", math.nextafter(0.05, math.inf), True),
        ("upper-binary64", 0.95, True),
        ("upper-predecessor", math.nextafter(0.95, -math.inf), True),
        ("upper-successor", math.nextafter(0.95, math.inf), False),
        ("central", 0.5, True),
        ("zero-tail", 0.0, False),
        ("one-tail", 1.0, False),
    )
    _require(len(rows) == len(direct_expected) + 1, "coverage conformance case count changed")
    for row, (case_id, pit, expected) in zip(rows[: len(direct_expected)], direct_expected, strict=True):
        _require(
            isinstance(row, dict)
            and row.get("case_id") == case_id
            and row.get("kind") == "binary64_pit"
            and row.get("pit_hex") == pit.hex()
            and row.get("expected_covered") is expected
            and row.get("observed_covered") is expected
            and row.get("passed") is True
            and _binary64_pit_covered(float.fromhex(str(row.get("pit_hex")))) is expected,
            f"coverage conformance direct case {case_id} changed or failed",
        )
    regression = rows[-1]
    _require(isinstance(regression, dict), "coverage regression case is malformed")
    _require(
        regression.get("case_id") == "v130-disclosed-boundary-coordinate"
        and regression.get("kind") == "float32_mixture_inputs"
        and regression.get("target_little_endian_f32_hex") == _V130_BOUNDARY_TARGET_F32_HEX
        and tuple(regression.get("member_means_little_endian_f32_hex", ())) == _V130_BOUNDARY_MEANS_F32_HEX
        and tuple(regression.get("member_log_variances_little_endian_f32_hex", ()))
        == _V130_BOUNDARY_LOG_VARIANCES_F32_HEX,
        "coverage v1.3 boundary regression inputs changed",
    )
    target = float(struct.unpack("<f", bytes.fromhex(_V130_BOUNDARY_TARGET_F32_HEX))[0])
    means = tuple(float(struct.unpack("<f", bytes.fromhex(value))[0]) for value in _V130_BOUNDARY_MEANS_F32_HEX)
    log_variances = tuple(
        float(struct.unpack("<f", bytes.fromhex(value))[0]) for value in _V130_BOUNDARY_LOG_VARIANCES_F32_HEX
    )
    member_cdfs = []
    for mean, log_variance in zip(means, log_variances, strict=True):
        z_score = (target - mean) * math.exp(-0.5 * log_variance)
        member_cdfs.append(0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0))))
    pit = math.fsum(member_cdfs) / 5
    _require(
        pit.hex() == "0x1.999998b3745adp-5"
        and regression.get("expected_pit_hex") == pit.hex()
        and regression.get("observed_pit_hex") == pit.hex()
        and regression.get("expected_covered") is False
        and regression.get("observed_covered") is False
        and regression.get("passed") is True
        and not _binary64_pit_covered(pit),
        "coverage v1.3 boundary regression did not reproduce exactly",
    )
    corpus = {"semantics_id": COVERAGE_SEMANTICS, "cases": rows}
    _require(
        report.get("corpus_sha256") == _canonical_sha256(corpus),
        "coverage conformance corpus digest changed",
    )
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    _require_sha256(report_sha256, "coverage conformance report_sha256")
    _require(
        report_sha256 == _canonical_sha256(body) and report.get("passed") is True,
        "coverage conformance report self-hash or pass status changed",
    )


def derive_seed(namespace: str, master_seed: int, index: int) -> int:
    """Derive the exact protocol-1.4.0 uint32 seed."""

    payload = f"WM-001|1.4.0|{namespace}|{master_seed}|{index}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)


def derive_master_seed(lane: str, index: int) -> int:
    """Derive one protocol-1.4.0 lane master from its prospective index."""

    if lane not in {"development", "formal"} or index < 0:
        raise ValueError("invalid WM-001 master-seed lane or index")
    payload = f"WM-001|1.4.0|{lane}-master|{index}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)


def _expected_oscillator_conformance() -> dict[str, object]:
    """Recompute the bound oscillator report without producer runtime imports."""

    cases = 512
    steps = 200
    seed = 20260718
    trajectory = sha256()
    for case_index in range(cases):
        reset_seed = seed + case_index
        digest = sha256(f"{INDEPENDENT_OSCILLATOR_SOURCE}:{reset_seed}".encode("ascii")).digest()
        phase_unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        velocity_unit = int.from_bytes(digest[8:16], "big") / float(1 << 64)
        phase = (2.0 * phase_unit - 1.0) * math.pi
        velocity = 0.5 + velocity_unit
        reset_observation = (math.cos(phase), math.sin(phase), velocity)
        trajectory.update(reset_seed.to_bytes(8, "big", signed=False))
        trajectory.update(struct.pack("<3d", *reset_observation))
        for step_index in range(steps):
            phase = math.remainder(phase + 0.05 * velocity, 2.0 * math.pi)
            observation = (math.cos(phase), math.sin(phase), velocity)
            reward = math.cos(phase)
            trajectory.update(struct.pack("<3d", *observation))
            trajectory.update(struct.pack("<d", reward))
            trajectory.update(bytes((0, int(step_index == steps - 1))))
    report: dict[str, object] = {
        "schema": "prospect.wm001.independent-phase-oscillator-conformance.v1",
        "source_id": INDEPENDENT_OSCILLATOR_SOURCE,
        "cases": cases,
        "steps_per_case": steps,
        "seed": seed,
        "max_reset_absolute_difference": 0.0,
        "max_action_pair_observation_absolute_difference": 0.0,
        "max_action_pair_reward_absolute_difference": 0.0,
        "unexpected_terminations": 0,
        "premature_or_missing_truncations": 0,
        "trajectory_sha256": trajectory.hexdigest(),
        "passed": True,
    }
    report["report_sha256"] = sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def verify_protocol() -> dict[str, Any]:
    """Verify the raw-byte seal and non-negotiable internal invariants."""

    protocol = _load_json(PROTOCOL_PATH)
    result_schema = _load_json(RESULT_SCHEMA_PATH)
    binding_schema = _load_json(BINDING_SCHEMA_PATH)

    try:
        seal_rows = [line.split() for line in SEAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise Violation(f"cannot load protocol seal: {exc}") from exc
    _require(all(len(row) == 2 for row in seal_rows), "invalid protocol seal format")
    seals = {relative: digest for digest, relative in seal_rows}
    expected_sealed_files = {
        "protocol.json": PROTOCOL_PATH,
        "schemas/formal-binding.schema.json": BINDING_SCHEMA_PATH,
        "schemas/raw-result.schema.json": RESULT_SCHEMA_PATH,
    }
    _require(set(seals) == set(expected_sealed_files), "sealed evidence file set changed")
    for relative, path in expected_sealed_files.items():
        _require_sha256(seals[relative], f"{relative} seal")
        _require(_file_sha256(path) == seals[relative], f"{relative} bytes do not match SEALED_PROTOCOL.sha256")

    experiment = protocol.get("experiment", {})
    _require(protocol.get("schema") == "prospect.world-model-lifecycle.protocol.v4", "wrong protocol schema")
    _require(experiment.get("id") == "WM-001", "wrong experiment ID")
    _require(experiment.get("protocol_version") == "1.4.0", "wrong protocol version")
    _require(experiment.get("status") == "sealed_before_formal_outcomes", "protocol is not marked sealed")
    _require(experiment.get("thresholds_sealed_before_outcomes") is True, "experiment thresholds are not sealed")
    _require(protocol.get("thresholds", {}).get("sealed_before_outcomes") is True, "threshold block is not sealed")

    _require(protocol.get("splits", {}).get("unit") == "whole_episode", "splits are not whole-episode")
    formal_splits = protocol.get("splits", {}).get("formal", {})
    for split in (
        "collect_a",
        "collect_b",
        "collect_irrelevant",
        "predictive_validation_a",
        "predictive_validation_b",
        "predictive_validation_irrelevant",
    ):
        _require(formal_splits.get(split, {}).get("episodes_per_replicate") == 8, f"{split} episode budget changed")
        _require(formal_splits.get(split, {}).get("transitions_per_replicate") == 1600, f"{split} step budget changed")
    _require(
        formal_splits.get("collect_irrelevant", {}).get("task") == "task_irrelevant",
        "irrelevant collection split is not bound to task_irrelevant",
    )
    _require(
        formal_splits.get("predictive_validation_irrelevant", {}).get("task") == "task_irrelevant",
        "irrelevant validation split is not bound to task_irrelevant",
    )
    for split in ("behavior_evaluation_a", "behavior_evaluation_b"):
        _require(
            formal_splits.get(split, {}).get("episodes_per_replicate") == 32,
            f"{split} must use 32 reset seeds",
        )

    seed_schedule = protocol.get("seed_schedule", {})
    _require(
        seed_schedule.get("derivation_domain_version") == "1.4.0",
        "seed derivation domain differs from protocol 1.4.0",
    )
    formal_seeds = tuple(seed_schedule.get("formal_replicate_master_seeds", ()))
    development_seeds = tuple(seed_schedule.get("development_replicate_master_seeds", ()))
    _require(formal_seeds == FORMAL_SEEDS, "formal master seeds changed")
    _require(development_seeds == DEVELOPMENT_SEEDS, "development master seeds changed")
    _require(set(formal_seeds).isdisjoint(development_seeds), "development and formal seeds overlap")
    _require(protocol.get("budgets", {}).get("formal_replicates") == 8, "formal replicate count changed")
    namespaces = seed_schedule.get("namespaces", {})
    actual_seed_counts = {
        namespace: declaration.get("count")
        for namespace, declaration in namespaces.items()
        if isinstance(declaration, dict)
    }
    _require(
        actual_seed_counts == EXPECTED_SEED_COUNTS,
        "seed namespace/count schedule differs from protocol 1.4.0",
    )
    master_derivation = seed_schedule.get("master_seed_derivation", {})
    _require(
        isinstance(master_derivation, dict)
        and master_derivation.get("lane_index_domains")
        == {
            "development": [0, 1],
            "formal": [0, 7],
        },
        "master-seed derivation index domains changed",
    )
    _require(
        DEVELOPMENT_SEEDS == tuple(derive_master_seed("development", index) for index in range(2))
        and FORMAL_SEEDS == tuple(derive_master_seed("formal", index) for index in range(8)),
        "master seeds do not match their prospective SHA-256 derivation",
    )
    current_streams = {
        derive_seed(namespace, master_seed, index)
        for master_seed in (*DEVELOPMENT_SEEDS, *FORMAL_SEEDS)
        for namespace, count in EXPECTED_SEED_COUNTS.items()
        for index in range(count)
    }
    old_streams = {
        int.from_bytes(
            sha256(f"WM-001|1.3.0|{namespace}|{master_seed}|{index}".encode()).digest()[:4],
            "big",
            signed=False,
        )
        for master_seed in _V130_MASTER_SEEDS
        for namespace, count in EXPECTED_SEED_COUNTS.items()
        for index in range(count)
    }
    collision_audit = master_derivation.get("collision_audit", {})
    _require(
        isinstance(collision_audit, dict)
        and collision_audit.get("current_master_seed_count") == 10
        and collision_audit.get("current_derived_stream_count") == 1360
        and collision_audit.get("unique_current_derived_stream_count") == len(current_streams) == 1360
        and collision_audit.get("current_internal_collision_count") == 0
        and collision_audit.get("master_overlap_with_v1_3_count") == 0
        and collision_audit.get("derived_stream_overlap_with_v1_3_count") == 0
        and set((*DEVELOPMENT_SEEDS, *FORMAL_SEEDS)).isdisjoint(_V130_MASTER_SEEDS)
        and current_streams.isdisjoint(old_streams),
        "master or derived seed collision audit failed",
    )

    tasks = protocol.get("tasks", {})
    _require(
        tasks.get("context_encoding", {}).get("task_irrelevant") == 2.0,
        "irrelevant task context encoding changed",
    )
    irrelevant_task = tasks.get("task_irrelevant", {})
    _require(
        irrelevant_task.get("id") == TASK_IRRELEVANT and irrelevant_task.get("context") == 2.0,
        "irrelevant task identity or context changed",
    )

    model = protocol.get("representation_and_model", {}).get("world_model", {})
    planner = protocol.get("representation_and_model", {}).get("planner", {})
    _require(model.get("ensemble_members") == 5, "world-model ensemble size changed")
    _require(planner.get("learning_during_evaluation") is False, "evaluation learning must be disabled")
    _require(planner.get("replay_writes_during_evaluation") is False, "evaluation replay writes must be disabled")
    predictive_secondary = protocol.get("metrics", {}).get("prediction", {}).get("secondary", ())
    coverage_metric = next(
        (
            row
            for row in predictive_secondary
            if isinstance(row, dict) and row.get("name") == "heldout_90_percent_interval_coverage"
        ),
        None,
    )
    _require(
        isinstance(coverage_metric, dict)
        and coverage_metric.get("semantics_id") == COVERAGE_SEMANTICS
        and coverage_metric.get("raw_fields")
        == [
            "interval_90_covered_target_count",
            "coverage_target_count",
            "interval_90_coverage",
            "coverage_semantics",
        ],
        "coverage metric semantics or raw fields changed",
    )
    k3_thresholds = protocol.get("thresholds", {}).get("k3_predictive_learning", {})
    _require(
        k3_thresholds.get("after_a_90_percent_interval_coverage_bounds_inclusive") == [0.7, 0.99]
        and "10*C >= 7*T" in str(k3_thresholds.get("after_a_coverage_decision_arithmetic"))
        and "100*C <= 99*T" in str(k3_thresholds.get("after_a_coverage_decision_arithmetic")),
        "coverage thresholds or count-space decision arithmetic changed",
    )
    _require(
        protocol.get("experience_and_learning", {}).get("task_a_update", {}).get("optimizer_steps") == 2000,
        "task-A update budget changed",
    )
    _require(
        protocol.get("experience_and_learning", {}).get("task_a_irrelevant_update", {}).get("optimizer_steps") == 2000,
        "irrelevant-evidence update budget changed",
    )
    _require(
        tuple(
            protocol.get("experience_and_learning", {}).get("task_a_irrelevant_update", {}).get("eligible_replay", ())
        )
        == ("collect_irrelevant",),
        "irrelevant-evidence update eligibility changed",
    )
    _require(
        protocol.get("experience_and_learning", {}).get("task_b_replay_update", {}).get("optimizer_steps") == 2000,
        "task-B update budget changed",
    )

    _require(
        set(protocol.get("controls", {}))
        == {
            "frozen",
            "corrupted_target",
            "irrelevant_evidence",
            "naive_sequential",
            "random_policy",
            "true_dynamics_mpc",
        },
        "control set changed",
    )
    _require(tuple(item.get("gate") for item in protocol.get("killing_order", ())) == GATES, "killing order changed")
    _require(
        tuple(protocol.get("bindings", {}).get("checkpoint", {}).get("canonical_component_ids", ()))
        == REQUIRED_COMPONENTS,
        "checkpoint component contract changed",
    )
    development_lane = protocol.get("lanes", {}).get("development", {})
    formal_lane = protocol.get("lanes", {}).get("formal", {})
    _require(
        development_lane.get("tuning_allowed") is False
        and "K3-K6 development performance values are descriptive" in str(development_lane.get("rule"))
        and tuple(development_lane.get("master_seeds", ())) == DEVELOPMENT_SEEDS,
        "development rehearsal governance changed",
    )
    _require(
        tuple(formal_lane.get("master_seeds", ())) == FORMAL_SEEDS
        and "sole permitted producer attempt" in str(formal_lane.get("launch_rule"))
        and "no same-version formal rerun" in str(formal_lane.get("launch_rule")),
        "single-attempt formal governance changed",
    )
    _require(
        set(protocol.get("bindings", {}).get("coverage_arithmetic", {}).get("required", ()))
        >= {
            "semantics identifier",
            "exact CPython executable and platform identity",
            "content-addressed conformance corpus and report",
            "producer-reference and independent-auditor agreement",
        },
        "coverage arithmetic binding contract is incomplete",
    )
    _require(
        result_schema.get("$id") == "https://prospect.local/schemas/wm-001-raw-result-v4.json",
        "wrong raw-result schema",
    )
    _require(
        binding_schema.get("$id") == "https://prospect.local/schemas/wm-001-formal-binding-v4.json",
        "wrong formal-binding schema",
    )
    return protocol


def verify_binding(path: Path) -> dict[str, Any]:
    """Verify a completed implementation binding before formal execution."""

    protocol = verify_protocol()
    binding = _load_json(path)
    _validate_json_schema(binding, _load_json(BINDING_SCHEMA_PATH), label="formal binding")
    _require(binding.get("schema") == "prospect.world-model-lifecycle.formal-binding.v4", "wrong binding schema")
    _require(binding.get("experiment_id") == "WM-001", "binding has wrong experiment")
    _parse_timestamp(binding.get("sealed_at_utc"), "sealed_at_utc")

    bound_protocol = binding.get("protocol", {})
    _require(bound_protocol.get("version") == "1.4.0", "binding has wrong protocol version")
    _require(bound_protocol.get("sha256") == _file_sha256(PROTOCOL_PATH), "binding has wrong protocol digest")
    _require(
        bound_protocol.get("raw_result_schema_sha256") == _file_sha256(RESULT_SCHEMA_PATH),
        "binding has wrong raw-result schema digest",
    )
    _require(
        bound_protocol.get("binding_schema_sha256") == _file_sha256(BINDING_SCHEMA_PATH),
        "binding has wrong binding-schema digest",
    )

    source = binding.get("source", {})
    _require(
        isinstance(source.get("git_commit"), str) and GIT_SHA_PATTERN.fullmatch(source["git_commit"]) is not None,
        "binding source commit is not a full Git SHA",
    )
    _require(
        isinstance(source.get("git_tree"), str) and GIT_SHA_PATTERN.fullmatch(source["git_tree"]) is not None,
        "binding source tree is not a full Git SHA",
    )
    _require(source.get("worktree_clean") is True, "formal binding requires a clean worktree")
    bound_implementation_files = source.get("implementation_files")
    _require(
        isinstance(bound_implementation_files, list) and bound_implementation_files,
        "binding has no implementation files",
    )
    seen_paths: set[str] = set()
    for index, entry in enumerate(bound_implementation_files):
        _require(isinstance(entry, dict), f"implementation_files[{index}] is not an object")
        relative = entry.get("path")
        _require(isinstance(relative, str) and relative, f"implementation_files[{index}].path is invalid")
        candidate = Path(relative)
        _require(not candidate.is_absolute() and ".." not in candidate.parts, f"unsafe implementation path: {relative}")
        _require(relative not in seen_paths, f"duplicate implementation path: {relative}")
        seen_paths.add(relative)
        actual = REPO / candidate
        _require(actual.is_file(), f"bound implementation file is missing: {relative}")
        _require(entry.get("bytes") == actual.stat().st_size, f"bound byte size changed: {relative}")
        _require(entry.get("sha256") == _file_sha256(actual), f"bound file digest changed: {relative}")
    _verify_implementation_manifest(bound_implementation_files)
    test_report_path = _binding_sibling(
        path,
        source.get("test_report_file"),
        "source.test_report_file",
    )
    _require(test_report_path.is_file(), "bound test report is missing")
    _require(
        source.get("test_report_bytes") == test_report_path.stat().st_size,
        "bound test report byte size changed",
    )
    _require(
        source.get("test_report_sha256") == _file_sha256(test_report_path),
        "bound test report digest changed",
    )

    dependencies = binding.get("dependencies", {})
    lockfile = dependencies.get("lockfile")
    _require(isinstance(lockfile, str) and lockfile, "binding lockfile path is missing")
    lock_path = REPO / lockfile
    _require(lock_path.is_file(), "bound dependency lockfile is missing")
    _require(dependencies.get("lockfile_sha256") == _file_sha256(lock_path), "dependency lockfile digest changed")
    packages = dependencies.get("packages")
    _require(isinstance(packages, list), "binding packages must be an array")
    package_names = {entry.get("name") for entry in packages if isinstance(entry, dict)}
    _require(REQUIRED_PACKAGES <= package_names, "binding is missing a required root package identity")
    _require(len(package_names) == len(packages), "binding package identities are duplicated")
    for package in packages:
        _require(isinstance(package.get("version"), str) and package["version"], "package version is missing")
        _require_sha256(package.get("distribution_sha256"), f"{package.get('name')} distribution_sha256")

    runtime = binding.get("runtime", {})
    _require(runtime.get("deterministic_algorithms") is True, "formal runtime must enable deterministic algorithms")
    if runtime.get("device") == "cuda":
        _require(
            runtime.get("cublas_workspace_config") == ":4096:8",
            "formal CUDA runtime must bind CUBLAS_WORKSPACE_CONFIG=:4096:8",
        )
    else:
        _require(
            runtime.get("cublas_workspace_config") is None,
            "non-CUDA formal runtime must not bind a cuBLAS workspace",
        )
    environment = binding.get("environment", {})
    _require(environment.get("id") == "Pendulum-v1", "binding has wrong environment")
    for field in ("wrapper_source_sha256", "installed_distribution_sha256", "conformance_report_sha256"):
        _require_sha256(environment.get(field), f"environment.{field}")
    conformance_path = _binding_sibling(
        path,
        environment.get("conformance_report_file"),
        "environment.conformance_report_file",
    )
    _require(conformance_path.is_file(), "bound Pendulum conformance report is missing")
    _require(
        environment.get("conformance_report_bytes") == conformance_path.stat().st_size,
        "bound Pendulum conformance report byte size changed",
    )
    _require(
        environment.get("conformance_report_sha256") == _file_sha256(conformance_path),
        "bound Pendulum conformance report digest changed",
    )
    conformance_report = _load_json(conformance_path)
    _verify_pendulum_conformance_report(conformance_report)
    expected_conformance_bytes = (
        json.dumps(
            conformance_report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        conformance_path.read_bytes() == expected_conformance_bytes,
        "bound Pendulum conformance report is not canonical JSON",
    )

    irrelevant_control = binding.get("irrelevant_control", {})
    _require(
        irrelevant_control.get("id") == TASK_IRRELEVANT
        and irrelevant_control.get("source_id") == INDEPENDENT_OSCILLATOR_SOURCE,
        "binding has the wrong irrelevant-control identity",
    )
    oscillator_source = HERE / "runtime_lane.py"
    _require(
        irrelevant_control.get("source_sha256") == _file_sha256(oscillator_source),
        "bound irrelevant-control source digest changed",
    )
    oscillator_conformance_path = _binding_sibling(
        path,
        irrelevant_control.get("conformance_report_file"),
        "irrelevant_control.conformance_report_file",
    )
    _require(
        oscillator_conformance_path.is_file(),
        "bound oscillator conformance report is missing",
    )
    _require(
        irrelevant_control.get("conformance_report_bytes") == oscillator_conformance_path.stat().st_size,
        "bound oscillator conformance report byte size changed",
    )
    _require(
        irrelevant_control.get("conformance_report_sha256") == _file_sha256(oscillator_conformance_path),
        "bound oscillator conformance report digest changed",
    )
    oscillator_conformance = _load_json(oscillator_conformance_path)
    _require(
        oscillator_conformance == _expected_oscillator_conformance(),
        "bound oscillator conformance report differs from independent semantics",
    )
    expected_oscillator_bytes = (
        json.dumps(
            oscillator_conformance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        oscillator_conformance_path.read_bytes() == expected_oscillator_bytes,
        "bound oscillator conformance report is not canonical JSON",
    )

    coverage_arithmetic = binding.get("coverage_arithmetic", {})
    _require(
        coverage_arithmetic.get("semantics_id") == COVERAGE_SEMANTICS
        and coverage_arithmetic.get("python_executable") == sys.executable
        and coverage_arithmetic.get("python_implementation") == platform.python_implementation() == "CPython"
        and coverage_arithmetic.get("python_version") == platform.python_version()
        and coverage_arithmetic.get("platform") == platform.platform()
        and coverage_arithmetic.get("machine") == platform.machine(),
        "bound coverage arithmetic runtime identity changed",
    )
    _require(
        coverage_arithmetic.get("producer_source_sha256") == _file_sha256(HERE / "model.py")
        and coverage_arithmetic.get("auditor_source_sha256") == _file_sha256(HERE / "artifact_audit.py")
        and coverage_arithmetic.get("formal_test_report_sha256") == source.get("test_report_sha256"),
        "bound coverage source or test-report digest changed",
    )
    coverage_conformance_path = _binding_sibling(
        path,
        coverage_arithmetic.get("conformance_report_file"),
        "coverage_arithmetic.conformance_report_file",
    )
    _require(coverage_conformance_path.is_file(), "bound coverage conformance report is missing")
    _require(
        coverage_arithmetic.get("conformance_report_bytes") == coverage_conformance_path.stat().st_size
        and coverage_arithmetic.get("conformance_report_sha256") == _file_sha256(coverage_conformance_path),
        "bound coverage conformance report size or digest changed",
    )
    coverage_conformance = _load_json(coverage_conformance_path)
    _verify_coverage_conformance_report(coverage_conformance)
    expected_coverage_bytes = (
        json.dumps(
            coverage_conformance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(
        coverage_conformance_path.read_bytes() == expected_coverage_bytes,
        "bound coverage conformance report is not canonical JSON",
    )

    checkpoint = binding.get("checkpoint_implementation", {})
    _require(tuple(checkpoint.get("component_ids", ())) == REQUIRED_COMPONENTS, "checkpoint components are incomplete")
    _require(
        tuple(binding.get("formal_replicate_master_seeds", ())) == FORMAL_SEEDS,
        "binding formal seeds do not match protocol",
    )
    _require(
        tuple(protocol["lanes"]["formal"]["master_seeds"]) == FORMAL_SEEDS,
        "protocol formal lane seed disagreement",
    )
    return binding


def _verify_formal_launch_record(
    path: Path,
    *,
    binding_sha256: str,
    execution: Mapping[str, Any],
) -> None:
    _require(path.name == "formal-launch.json", "formal launch record has the wrong filename")
    _require(path.is_file() and not path.is_symlink(), "formal launch record is missing or aliased")
    payload = path.read_bytes()
    record = _load_json(path)
    body = dict(record)
    record_sha256 = body.pop("record_sha256", None)
    _require_sha256(record_sha256, "formal launch record_sha256")
    _require(
        record.get("schema") == "prospect.wm001.formal-launch.v1"
        and record.get("experiment_id") == "WM-001"
        and record.get("protocol_version") == "1.4.0"
        and record.get("formal_binding_sha256") == binding_sha256
        and record.get("attempt_directory") == path.parent.name
        and record.get("git_commit") == execution.get("git_commit")
        and record.get("git_tree") == execution.get("git_tree")
        and record_sha256 == _canonical_sha256(body),
        "formal launch record identity or self-hash changed",
    )
    expected_payload = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _require(payload == expected_payload, "formal launch record is not canonical JSON")


def _verify_result_runtime_binding(
    execution: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    _require(
        execution.get("platform") == runtime.get("platform")
        and execution.get("device") == runtime.get("device")
        and execution.get("deterministic_algorithms") == runtime.get("deterministic_algorithms") is True,
        "result runtime differs from binding",
    )


def verify_result(path: Path, binding_path: Path | None) -> dict[str, Any]:
    """Verify result-envelope invariants and causal custody."""

    protocol = verify_protocol()
    result = _load_json(path)
    _validate_json_schema(result, _load_json(RESULT_SCHEMA_PATH), label="raw result")
    _require(result.get("schema") == "prospect.world-model-lifecycle.raw-result.v4", "wrong result schema")
    _require(result.get("experiment_id") == "WM-001", "result has wrong experiment")
    _require(result.get("protocol_version") == "1.4.0", "result has wrong protocol version")
    _require(result.get("protocol_sha256") == _file_sha256(PROTOCOL_PATH), "result protocol digest mismatch")

    lane = result.get("lane")
    _require(lane in {"development", "formal"}, "result lane must be development or formal")
    if lane == "development":
        _require(result.get("claim_eligible") is False, "development result cannot be claim eligible")
    else:
        _require(result.get("claim_eligible") is True, "formal result must identify its claim-eligible lane")
        _require(binding_path is not None, "formal result verification requires --binding")
        binding = verify_binding(binding_path)
        _require(result.get("formal_binding_sha256") == _file_sha256(binding_path), "formal binding digest mismatch")
        _require(
            _parse_timestamp(result.get("started_at_utc"), "started_at_utc")
            >= _parse_timestamp(binding.get("sealed_at_utc"), "binding.sealed_at_utc"),
            "formal result started before its implementation binding was sealed",
        )
        execution = result.get("execution", {})
        _require(
            execution.get("git_commit") == binding["source"]["git_commit"],
            "result source commit differs from binding",
        )
        _require(execution.get("git_tree") == binding["source"]["git_tree"], "result source tree differs from binding")
        _require(execution.get("worktree_clean") is True, "formal result did not run from a clean worktree")
        _require(
            execution.get("dependency_lock_sha256") == binding["dependencies"]["lockfile_sha256"],
            "result dependency lock differs from binding",
        )
        runtime = binding.get("runtime", {})
        _require(
            isinstance(runtime, dict),
            "formal binding runtime block is malformed",
        )
        _verify_result_runtime_binding(execution, runtime)
        _require(execution.get("deterministic_algorithms") is True, "formal result was nondeterministic")
        launch_path = path.parent / "formal-launch.json"
        _require(
            execution.get("formal_launch_file") == "formal-launch.json"
            and execution.get("formal_launch_sha256") == _file_sha256(launch_path),
            "formal result does not bind its sole-launch record",
        )
        _verify_formal_launch_record(
            launch_path,
            binding_sha256=_file_sha256(binding_path),
            execution=execution,
        )
    if lane == "development":
        execution = result.get("execution", {})
        _require(
            isinstance(execution, dict)
            and execution.get("formal_launch_file") is None
            and execution.get("formal_launch_sha256") is None,
            "development result unexpectedly binds a formal launch",
        )

    started = _parse_timestamp(result.get("started_at_utc"), "started_at_utc")
    completed = _parse_timestamp(result.get("completed_at_utc"), "completed_at_utc")
    _require(completed >= started, "result completed before it started")

    replicates = result.get("replicates")
    _require(isinstance(replicates, list) and replicates, "result has no replicates")
    expected_masters = FORMAL_SEEDS if lane == "formal" else DEVELOPMENT_SEEDS
    if lane == "formal":
        _require(len(replicates) == 8, "formal result must contain exactly 8 replicates")
        _require(
            tuple(item.get("master_seed") for item in replicates) == FORMAL_SEEDS,
            "formal replicate order/seeds changed",
        )
    else:
        _require(all(item.get("master_seed") in expected_masters for item in replicates), "undeclared development seed")

    transition_owner: dict[str, tuple[str, str]] = {}
    for replicate in replicates:
        _verify_replicate(replicate, protocol, transition_owner, lane=lane)

    metrics = result.get("aggregate_metrics")
    _require(isinstance(metrics, list), "aggregate_metrics must be an array")
    for metric in metrics:
        values = metric.get("replicate_values")
        _require(isinstance(values, list), "aggregate metric replicate_values must be an array")
        if lane == "formal":
            _require(len(values) == 8, f"formal aggregate {metric.get('name')} must have 8 values")

    gates = result.get("gate_results")
    _require(isinstance(gates, list) and gates, "result has no gate results")
    gate_ids = tuple(item.get("gate") for item in gates)
    _require(gate_ids == GATES[: len(gate_ids)], "gate results are not a K0-K7 prefix")
    prior_passed = True
    for index, gate in enumerate(gates):
        _require(
            prior_passed,
            "gate results continue after the first failed killing gate",
        )
        checks = gate.get("checks")
        _require(isinstance(checks, list), f"{gate.get('gate')} checks must be an array")
        computed_pass = bool(checks) and all(check.get("passed") is True for check in checks)
        _require(gate.get("passed") is computed_pass, f"{gate.get('gate')} passed value disagrees with checks")
        _require(
            gate.get("claim_supported") is False,
            "producer result claimed support before independent artifact audit",
        )
        if not computed_pass:
            _require(
                index == len(gates) - 1,
                "failed killing gate is not the final reported gate",
            )
        prior_passed = computed_pass
    _require(
        gate_ids[-1] == "K7" or gates[-1].get("passed") is False,
        "passing gate prefix ends before K7",
    )
    if "K7" in gate_ids:
        for replicate in replicates:
            _verify_restart_parity(replicate)
    return result


def _row_contract(row: dict[str, Any]) -> tuple[object, object, object, object]:
    return (
        row.get("split"),
        row.get("task_id"),
        row.get("condition"),
        row.get("checkpoint_id"),
    )


def _verify_formal_matrix(
    replicate: dict[str, Any],
    *,
    replicate_id: str,
) -> None:
    """Require the exact sealed v1.4 formal evidence matrix."""

    episodes = replicate["episodes"]
    actual_episode_counts = Counter(_row_contract(row) for row in episodes)
    _require(
        len(episodes) == 496 and actual_episode_counts == Counter(FORMAL_EPISODE_CONTRACT_COUNTS),
        f"{replicate_id}: formal episode matrix differs from the exact 496-episode contract",
    )

    expected_transition_counts: Counter[object] = Counter()
    for contract, episode_count in FORMAL_EPISODE_CONTRACT_COUNTS.items():
        expected_transition_counts[contract[0]] += episode_count * 200
    transitions = replicate["transitions"]
    actual_transition_counts = Counter(row.get("split") for row in transitions)
    _require(
        len(transitions) == 99_200 and actual_transition_counts == expected_transition_counts,
        f"{replicate_id}: formal transition matrix differs from the exact 99,200-transition contract",
    )

    predictive = replicate.get("predictive_metrics")
    _require(
        isinstance(predictive, list)
        and len(predictive) == 12
        and Counter(_row_contract(row) for row in predictive) == Counter(FORMAL_PREDICTIVE_CONTRACT_COUNTS)
        and all(row.get("transition_count") == 1_600 for row in predictive)
        and all(row.get("coverage_target_count") == 6_400 for row in predictive),
        f"{replicate_id}: formal predictive matrix differs from the exact twelve-row contract",
    )

    policy_runs = replicate.get("policy_runs")
    _require(
        isinstance(policy_runs, list)
        and len(policy_runs) == 20
        and Counter(_row_contract(row) for row in policy_runs)
        == Counter({contract: 1 for contract in EPISODE_CONTRACTS}),
        f"{replicate_id}: formal policy-run matrix differs from the exact twenty-run contract",
    )

    updates = replicate["updates"]
    expected_updates = Counter(
        {(phase, "committed", 2_000): 1 for phase in COMMITTED_PHASE_SPLITS}
        | {("rejected_update_probe", "rejected", 0): 1}
    )
    actual_updates = Counter(
        (
            row.get("phase"),
            row.get("status"),
            row.get("optimizer_steps"),
        )
        for row in updates
    )
    _require(
        len(updates) == 6 and actual_updates == expected_updates,
        f"{replicate_id}: formal update matrix differs from five matched updates plus the rejection probe",
    )

    manifests = replicate.get("optimizer_batch_manifests")
    _require(
        isinstance(manifests, list)
        and len(manifests) == 5
        and Counter(row.get("phase") for row in manifests) == Counter({phase: 1 for phase in COMMITTED_PHASE_SPLITS}),
        f"{replicate_id}: formal optimizer-manifest matrix differs",
    )


def _verify_update_eligibility(
    update: dict[str, Any],
    *,
    local_transitions: dict[str, dict[str, Any]],
    replicate_id: str,
) -> None:
    """Bind one update to its exact collection-only canonical input set."""

    expected_phase_splits = {
        **COMMITTED_PHASE_SPLITS,
        "rejected_update_probe": ("collect_a",),
    }
    phase = update.get("phase")
    consumed = update.get("eligible_transition_ids")
    _require(isinstance(consumed, list), f"{replicate_id}: update eligible IDs must be an array")
    _require(update.get("eligible_transition_count") == len(consumed), f"{replicate_id}: eligible count mismatch")
    _require(
        len(consumed) == len(set(consumed)),
        f"{replicate_id}: {phase} repeats an eligible transition ID",
    )
    expected_splits = expected_phase_splits.get(phase)
    _require(
        expected_splits is not None and tuple(update.get("eligible_splits", ())) == expected_splits,
        f"{replicate_id}: {phase} eligible split declaration differs",
    )
    eligible = set(expected_splits or ())
    for transition_id in consumed:
        _require(transition_id in local_transitions, f"{replicate_id}: update consumes unknown {transition_id}")
        _require(
            local_transitions[transition_id].get("split") in eligible,
            f"{replicate_id}: update consumes held-out or phase-ineligible {transition_id}",
        )
    expected_ids = {
        transition_id for transition_id, transition in local_transitions.items() if transition.get("split") in eligible
    }
    if phase == "rejected_update_probe":
        _require(not consumed, f"{replicate_id}: rejected probe consumed real experience")
    else:
        _require(
            set(consumed) == expected_ids,
            f"{replicate_id}: {phase} does not bind the exact eligible collection set",
        )


def _verify_replicate(
    replicate: dict[str, Any],
    protocol: dict[str, Any],
    global_transition_owner: dict[str, tuple[str, str]],
    *,
    lane: str,
) -> None:
    replicate_id = replicate.get("replicate_id")
    master = replicate.get("master_seed")
    _require(isinstance(replicate_id, str) and replicate_id, "replicate_id is missing")
    _require(isinstance(master, int), f"{replicate_id}: master_seed is missing")

    seed_rows = replicate.get("derived_seeds")
    _require(isinstance(seed_rows, list), f"{replicate_id}: derived_seeds must be an array")
    declared_namespaces = protocol["seed_schedule"]["namespaces"]
    _require(
        len(seed_rows) == len(declared_namespaces)
        and all(isinstance(row, dict) for row in seed_rows)
        and {row.get("namespace") for row in seed_rows} == set(declared_namespaces),
        f"{replicate_id}: derived seed namespace matrix is incomplete, duplicated, or extended",
    )
    seeds_by_namespace = {row.get("namespace"): row.get("values") for row in seed_rows if isinstance(row, dict)}
    for namespace, declaration in declared_namespaces.items():
        values = seeds_by_namespace.get(namespace)
        _require(isinstance(values, list), f"{replicate_id}: missing seed namespace {namespace}")
        expected = [derive_seed(namespace, master, index) for index in range(declaration["count"])]
        _require(values == expected, f"{replicate_id}: derived seeds changed for {namespace}")

    episodes = replicate.get("episodes")
    transitions = replicate.get("transitions")
    updates = replicate.get("updates")
    _require(isinstance(episodes, list), f"{replicate_id}: episodes must be an array")
    _require(isinstance(transitions, list), f"{replicate_id}: transitions must be an array")
    _require(isinstance(updates, list), f"{replicate_id}: updates must be an array")
    if lane == "formal":
        _verify_formal_matrix(replicate, replicate_id=replicate_id)

    local_transitions: dict[str, dict[str, Any]] = {}
    seed_split: dict[tuple[str, int], str] = {}
    for transition in transitions:
        transition_id = transition.get("transition_id")
        _require(isinstance(transition_id, str) and transition_id, f"{replicate_id}: transition ID missing")
        _require(transition_id not in global_transition_owner, f"duplicate real transition ID: {transition_id}")
        _require(
            transition.get("real_or_imagined") == "real",
            f"imagined transition in real namespace: {transition_id}",
        )
        _require(
            isinstance(transition.get("run_id"), str) and transition["run_id"],
            f"{transition_id}: run ID missing",
        )
        expected_context = {
            TASK_A: 0.0,
            TASK_B: 1.0,
            TASK_IRRELEVANT: 2.0,
        }.get(transition.get("task_id"))
        _require(
            expected_context is not None and transition.get("task_context") == expected_context,
            f"{transition_id}: invalid task identity or context",
        )
        scaled_target = transition.get("scaled_target")
        _require(
            isinstance(scaled_target, list)
            and len(scaled_target) == 4
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in scaled_target),
            f"{transition_id}: scaled target is malformed",
        )
        expected_target_sha256 = sha256(struct.pack("<4d", *(float(value) for value in scaled_target))).hexdigest()
        _require(
            transition.get("target_sha256") == expected_target_sha256,
            f"{transition_id}: target digest differs from scaled target",
        )
        pre_observation = transition.get("pre_observation")
        next_observation = transition.get("next_observation")
        _require(
            isinstance(pre_observation, list)
            and len(pre_observation) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in pre_observation),
            f"{transition_id}: pre-observation is malformed",
        )
        _require(
            isinstance(next_observation, list)
            and len(next_observation) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in next_observation),
            f"{transition_id}: next observation is malformed",
        )
        reward = transition.get("reward")
        _require(
            isinstance(reward, (int, float)) and math.isfinite(float(reward)),
            f"{transition_id}: reward is malformed",
        )
        expected_scaled = (
            (float(next_observation[0]) - float(pre_observation[0])) / 2.0,
            (float(next_observation[1]) - float(pre_observation[1])) / 2.0,
            (float(next_observation[2]) - float(pre_observation[2])) / 16.0,
            float(reward) / 16.2736044,
        )
        _require(
            all(
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
                for actual, expected in zip(scaled_target, expected_scaled, strict=True)
            ),
            f"{transition_id}: scaled target differs from raw transition values",
        )
        intended = float(transition.get("intended_action"))
        applied = float(transition.get("applied_action"))
        task_id = transition.get("task_id")
        expected_applied = -intended if task_id == TASK_B else 0.0 if task_id == TASK_IRRELEVANT else intended
        _require(
            math.isclose(applied, expected_applied, rel_tol=0.0, abs_tol=1e-12),
            f"{transition_id}: applied action differs from task semantics",
        )
        global_transition_owner[transition_id] = (replicate_id, transition.get("split"))
        local_transitions[transition_id] = transition

    episode_ids: set[str] = set()
    referenced_transition_ids: Counter[str] = Counter()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        _require(isinstance(episode_id, str) and episode_id, f"{replicate_id}: episode ID missing")
        _require(episode_id not in episode_ids, f"{replicate_id}: duplicate episode ID {episode_id}")
        episode_ids.add(episode_id)
        _require(episode.get("environment_steps") == 200, f"{episode_id}: incomplete episode")
        run_id = episode.get("run_id")
        _require(isinstance(run_id, str) and run_id, f"{episode_id}: run ID missing")
        split = episode.get("split")
        task = episode.get("task_id")
        contract = (
            split,
            task,
            episode.get("condition"),
            episode.get("checkpoint_id"),
        )
        _require(
            contract in EPISODE_CONTRACTS,
            f"{episode_id}: split/task/condition/checkpoint is outside the sealed matrix",
        )
        reset_seed = episode.get("reset_seed")
        _require(isinstance(reset_seed, int), f"{episode_id}: reset seed missing")
        assignment_key = (task, reset_seed)
        previous_split = seed_split.setdefault(assignment_key, split)
        _require(previous_split == split, f"{replicate_id}: reset seed crosses episode splits")
        if str(split).startswith(("predictive_validation_", "behavior_evaluation_")):
            _require(episode.get("learning_allowed") is False, f"{episode_id}: held-out learning was allowed")
            _require(episode.get("replay_writes_allowed") is False, f"{episode_id}: held-out replay write was allowed")
        transition_ids = episode.get("transition_ids")
        _require(isinstance(transition_ids, list) and len(transition_ids) == 200, f"{episode_id}: bad transition count")
        intended_actions: list[float] = []
        applied_actions: list[float] = []
        rewards: list[float] = []
        previous_next: list[float] | None = None
        for expected_step, transition_id in enumerate(transition_ids):
            referenced_transition_ids[str(transition_id)] += 1
            _require(transition_id in local_transitions, f"{episode_id}: unresolved transition {transition_id}")
            transition = local_transitions[transition_id]
            _require(transition.get("episode_id") == episode_id, f"{transition_id}: episode linkage mismatch")
            _require(transition.get("run_id") == run_id, f"{transition_id}: run linkage mismatch")
            _require(transition.get("split") == split, f"{transition_id}: split linkage mismatch")
            _require(transition.get("task_id") == task, f"{transition_id}: task linkage mismatch")
            _require(
                transition.get("model_version_at_action") == episode.get("model_version")
                and transition.get("parameter_sha256_at_action") == episode.get("parameter_sha256"),
                f"{transition_id}: action-time model lineage differs from its episode",
            )
            _require(
                transition.get("step_index") == expected_step,
                f"{transition_id}: step index differs from episode order",
            )
            _require(
                transition.get("terminated") is False,
                f"{transition_id}: Pendulum must not terminate",
            )
            _require(
                transition.get("truncated") is (expected_step == 199),
                f"{transition_id}: TimeLimit truncation flag differs",
            )
            current_pre = transition["pre_observation"]
            if previous_next is not None:
                _require(
                    all(
                        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
                        for left, right in zip(previous_next, current_pre, strict=True)
                    ),
                    f"{transition_id}: physical observation chain is discontinuous",
                )
            previous_next = transition["next_observation"]
            intended_actions.append(float(transition["intended_action"]))
            applied_actions.append(float(transition["applied_action"]))
            rewards.append(float(transition["reward"]))
        _require(
            math.isclose(
                float(episode.get("return")),
                math.fsum(rewards),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            f"{episode_id}: return differs from raw rewards",
        )
        _require(
            episode.get("action_trace_sha256")
            == _canonical_sha256(
                {
                    "intended": intended_actions,
                    "applied": applied_actions,
                }
            ),
            f"{episode_id}: action trace digest differs from raw transitions",
        )

    _require(
        set(referenced_transition_ids) == set(local_transitions)
        and all(count == 1 for count in referenced_transition_ids.values()),
        f"{replicate_id}: every real transition must be referenced by exactly one episode",
    )

    split_reset_namespaces = {
        "collect_a": "collect_a_episode",
        "collect_b": "collect_b_episode",
        "collect_irrelevant": "collect_irrelevant_episode",
        "predictive_validation_a": "predictive_validation_a_episode",
        "predictive_validation_b": "predictive_validation_b_episode",
        "predictive_validation_irrelevant": "predictive_validation_irrelevant_episode",
    }
    for split, namespace in split_reset_namespaces.items():
        grouped = [episode for episode in episodes if episode.get("split") == split]
        actual = [episode.get("reset_seed") for episode in grouped]
        expected = [derive_seed(namespace, master, index) for index in range(len(grouped))]
        _require(
            actual == expected,
            f"{replicate_id}: {split} reset seeds differ from {namespace}",
        )
    behavior_reset_namespaces = {
        "behavior_evaluation_a": "behavior_evaluation_a_episode",
        "behavior_evaluation_b": "behavior_evaluation_b_episode",
    }
    for split, namespace in behavior_reset_namespaces.items():
        conditions = {str(episode.get("condition")) for episode in episodes if episode.get("split") == split}
        for condition in sorted(conditions):
            grouped = [
                episode
                for episode in episodes
                if episode.get("split") == split and episode.get("condition") == condition
            ]
            actual = [episode.get("reset_seed") for episode in grouped]
            expected = [derive_seed(namespace, master, index) for index in range(len(grouped))]
            _require(
                actual == expected,
                f"{replicate_id}: {split}/{condition} reset seeds differ from {namespace}",
            )

    policy_runs = replicate.get("policy_runs")
    _require(isinstance(policy_runs, list), f"{replicate_id}: policy_runs must be an array")
    runs_by_id: dict[str, list[dict[str, Any]]] = {}
    for run in policy_runs:
        run_id = run.get("run_id")
        _require(isinstance(run_id, str) and run_id, f"{replicate_id}: policy run ID missing")
        runs_by_id.setdefault(run_id, []).append(run)
    episode_runs = {str(episode.get("run_id")) for episode in episodes}
    _require(set(runs_by_id) == episode_runs, f"{replicate_id}: policy-run set differs from real executions")
    for run_id, run_rows in runs_by_id.items():
        _require(len(run_rows) == 1, f"{replicate_id}: duplicate policy run {run_id}")
        run = run_rows[0]
        grouped = [episode for episode in episodes if episode.get("run_id") == run_id]
        ordered_ids = [str(episode["episode_id"]) for episode in grouped]
        reset_seeds = [int(episode["reset_seed"]) for episode in grouped]
        _require(run.get("episode_ids") == ordered_ids, f"{run_id}: episode order differs")
        _require(run.get("reset_seeds") == reset_seeds, f"{run_id}: reset seeds differ")
        _require(run.get("action_count") == 200 * len(grouped), f"{run_id}: action count differs")
        for field in ("task_id", "split", "condition", "checkpoint_id"):
            values = {episode.get(field) for episode in grouped}
            _require(len(values) == 1 and run.get(field) in values, f"{run_id}: {field} differs")
        namespace = run.get("seed_namespace")
        seed_index = run.get("seed_index")
        _require(
            isinstance(namespace, str)
            and namespace in protocol["seed_schedule"]["namespaces"]
            and isinstance(seed_index, int)
            and 0 <= seed_index < int(protocol["seed_schedule"]["namespaces"][namespace]["count"]),
            f"{run_id}: seed reference is invalid",
        )
        _require(
            run.get("seed") == derive_seed(namespace, master, seed_index),
            f"{run_id}: seed differs from declared namespace",
        )
        task_id = run.get("task_id")
        task_index = 0 if task_id == TASK_A else 1
        split = str(run.get("split"))
        condition = str(run.get("condition"))
        controller = run.get("controller_kind")
        expected_controller = (
            "uniform_random"
            if condition in {"collection_random", "validation_random", "random"}
            else "cem_oracle"
            if condition == "oracle"
            else "cem_learned"
        )
        _require(
            (
                run.get("split"),
                run.get("task_id"),
                condition,
                run.get("checkpoint_id"),
            )
            in EPISODE_CONTRACTS
            and controller == expected_controller,
            f"{run_id}: execution/controller contract is outside the sealed matrix",
        )
        expected_seed_ref = (
            ("irrelevant_collection_action", 0)
            if split == "collect_irrelevant"
            else ("predictive_validation_irrelevant_action", 0)
            if split == "predictive_validation_irrelevant"
            else ("collection_action", task_index)
            if split in {"collect_a", "collect_b"}
            else ("predictive_validation_action", task_index)
            if split in {"predictive_validation_a", "predictive_validation_b"}
            else ("random_policy_action", task_index)
            if condition == "random"
            else ("planner", 0)
        )
        _require((namespace, seed_index) == expected_seed_ref, f"{run_id}: wrong policy seed namespace")
        intended_actions = [
            float(local_transitions[transition_id]["intended_action"])
            for episode in grouped
            for transition_id in episode["transition_ids"]
        ]
        applied_actions = [
            float(local_transitions[transition_id]["applied_action"])
            for episode in grouped
            for transition_id in episode["transition_ids"]
        ]
        _require(
            run.get("action_trace_sha256")
            == _canonical_sha256(
                {
                    "episode_ids": ordered_ids,
                    "intended": intended_actions,
                    "applied": applied_actions,
                }
            ),
            f"{run_id}: run action trace differs",
        )

    snapshots = replicate.get("evaluated_checkpoints")
    _require(isinstance(snapshots, list), f"{replicate_id}: evaluated_checkpoints must be an array")
    snapshots_by_condition = {
        snapshot.get("condition"): snapshot for snapshot in snapshots if isinstance(snapshot, dict)
    }
    _require(
        len(snapshots) == 7
        and set(snapshots_by_condition)
        == {
            "cold",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_a",
            "after_b_replay",
            "after_b_naive",
        },
        f"{replicate_id}: evaluated checkpoint set differs",
    )
    for condition, snapshot in snapshots_by_condition.items():
        _require(
            snapshot.get("sha256") == snapshot.get("live_state_sha256"),
            f"{replicate_id}: {condition} model artifact does not bind live state",
        )
    for run in policy_runs:
        controller_kind = run.get("controller_kind")
        condition = str(run.get("condition"))
        if controller_kind == "cem_learned":
            snapshot = snapshots_by_condition.get(condition)
            _require(
                isinstance(snapshot, dict)
                and run.get("controller_version") == f"wm001-sha256:{snapshot.get('parameter_sha256')}",
                f"{replicate_id}: learned CEM {condition} is not bound to its evaluated parameter snapshot",
            )
        elif controller_kind == "cem_oracle":
            _require(
                run.get("controller_version") == "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1",
                f"{replicate_id}: oracle CEM controller version differs",
            )
    predictive_metrics = replicate.get("predictive_metrics")
    _require(isinstance(predictive_metrics, list), f"{replicate_id}: predictive_metrics must be an array")
    for metric in predictive_metrics:
        _require(
            (
                metric.get("split"),
                metric.get("task_id"),
                metric.get("condition"),
                metric.get("checkpoint_id"),
            )
            in PREDICTIVE_CONTRACTS,
            f"{replicate_id}: predictive row is outside the sealed matrix",
        )
        snapshot = snapshots_by_condition.get(metric.get("condition"))
        _require(isinstance(snapshot, dict), f"{replicate_id}: predictive model snapshot missing")
        transition_count = metric.get("transition_count")
        covered_target_count = metric.get("interval_90_covered_target_count")
        coverage_target_count = metric.get("coverage_target_count")
        coverage = metric.get("interval_90_coverage")
        _require(
            metric.get("coverage_semantics") == COVERAGE_SEMANTICS,
            f"{replicate_id}: predictive coverage semantics differ from v1.4",
        )
        _require(
            isinstance(transition_count, int)
            and not isinstance(transition_count, bool)
            and isinstance(covered_target_count, int)
            and not isinstance(covered_target_count, bool)
            and isinstance(coverage_target_count, int)
            and not isinstance(coverage_target_count, bool)
            and coverage_target_count == 4 * transition_count
            and 0 <= covered_target_count <= coverage_target_count,
            f"{replicate_id}: predictive coverage counts are invalid",
        )
        _require(
            isinstance(coverage, (int, float))
            and not isinstance(coverage, bool)
            and math.isfinite(float(coverage))
            and float(coverage) == covered_target_count / coverage_target_count,
            f"{replicate_id}: predictive coverage fraction differs from exact counts",
        )
        for field in ("model_version", "parameter_sha256", "live_state_sha256"):
            _require(
                metric.get(field) == snapshot.get(field),
                f"{replicate_id}: predictive {metric.get('condition')} {field} differs from artifact",
            )

    for update in updates:
        _verify_update_eligibility(
            update,
            local_transitions=local_transitions,
            replicate_id=replicate_id,
        )
        steps = update.get("optimizer_steps")
        expected_samples = (
            0 if update.get("status") == "rejected" else int(steps) * 5 * 256 if isinstance(steps, int) else -1
        )
        _require(
            update.get("consumed_sample_count") == expected_samples,
            f"{replicate_id}: consumed sample count differs from optimizer budget",
        )

    updates_by_phase = {update.get("phase"): update for update in updates if isinstance(update, dict)}
    _require(
        len(updates_by_phase) == len(updates)
        and set(updates_by_phase) == {*COMMITTED_PHASE_SPLITS, "rejected_update_probe"},
        f"{replicate_id}: update phase matrix is incomplete, duplicated, or extended",
    )
    train_a = updates_by_phase.get("train_a")
    corrupted = updates_by_phase.get("train_a_corrupted")
    irrelevant = updates_by_phase.get("train_a_irrelevant")
    replay = updates_by_phase.get("train_b_replay")
    naive = updates_by_phase.get("train_b_naive")
    rejected_probe = updates_by_phase.get("rejected_update_probe")
    _require(
        all(
            isinstance(update, dict)
            for update in (
                train_a,
                corrupted,
                irrelevant,
                replay,
                naive,
                rejected_probe,
            )
        ),
        f"{replicate_id}: update and rejected-probe set is incomplete",
    )
    assert isinstance(train_a, dict)
    assert isinstance(corrupted, dict)
    assert isinstance(irrelevant, dict)
    assert isinstance(replay, dict)
    assert isinstance(naive, dict)
    assert isinstance(rejected_probe, dict)
    for phase, update in updates_by_phase.items():
        expected_status = "rejected" if phase == "rejected_update_probe" else "committed"
        _require(
            update.get("status") == expected_status,
            f"{replicate_id}: {phase} has the wrong transaction status",
        )
    _require(
        rejected_probe.get("status") == "rejected"
        and rejected_probe.get("full_state_before_sha256") == rejected_probe.get("full_state_after_sha256"),
        f"{replicate_id}: rejected probe did not preserve full-state bytes",
    )
    for label in ("before", "after"):
        digest = rejected_probe.get(f"full_state_{label}_sha256")
        reference = rejected_probe.get(f"full_state_{label}_file")
        _require(
            isinstance(reference, dict)
            and reference.get("sha256") == digest
            and reference.get("media_type") == "application/vnd.prospect.wm001.rejected-probe-state+json",
            f"{replicate_id}: rejected probe {label} full-state reference differs",
        )
    for field in (
        "predecessor_parameter_sha256",
        "predecessor_model_version",
        "live_state_before_sha256",
    ):
        _require(
            corrupted.get(field) == train_a.get(field),
            f"{replicate_id}: corrupted control cold {field} ancestry differs",
        )
        _require(
            irrelevant.get(field) == train_a.get(field),
            f"{replicate_id}: irrelevant control cold {field} ancestry differs",
        )
    for label, update in (("train_b_replay", replay), ("train_b_naive", naive)):
        for previous, resulting in (
            ("predecessor_parameter_sha256", "committed_parameter_sha256"),
            ("predecessor_model_version", "committed_model_version"),
            ("live_state_before_sha256", "live_state_after_sha256"),
        ):
            _require(
                update.get(previous) == train_a.get(resulting),
                f"{replicate_id}: {label} post-A {previous} ancestry differs",
            )
    snapshot_ancestry = {
        "cold": train_a.get("live_state_before_sha256"),
        "frozen": train_a.get("live_state_before_sha256"),
        "corrupted": corrupted.get("live_state_after_sha256"),
        "irrelevant": irrelevant.get("live_state_after_sha256"),
        "after_a": train_a.get("live_state_after_sha256"),
        "after_b_replay": replay.get("live_state_after_sha256"),
        "after_b_naive": naive.get("live_state_after_sha256"),
    }
    for condition, expected_digest in snapshot_ancestry.items():
        _require(
            snapshots_by_condition[condition].get("live_state_sha256") == expected_digest,
            f"{replicate_id}: {condition} compound snapshot ancestry differs",
        )

    manifests = replicate.get("optimizer_batch_manifests")
    _require(isinstance(manifests, list), f"{replicate_id}: optimizer manifests must be an array")
    manifests_by_phase = {manifest.get("phase"): manifest for manifest in manifests if isinstance(manifest, dict)}
    _require(
        len(manifests) == len(COMMITTED_PHASE_SPLITS)
        and len(manifests_by_phase) == len(manifests)
        and set(manifests_by_phase) == set(COMMITTED_PHASE_SPLITS),
        f"{replicate_id}: optimizer manifest matrix is incomplete, duplicated, or extended",
    )
    for phase, manifest in manifests_by_phase.items():
        _require(
            manifest.get("sha256") == updates_by_phase[phase].get("sampling_manifest_sha256"),
            f"{replicate_id}: {phase} optimizer manifest differs from its update receipt",
        )


def _verify_restart_parity(replicate: dict[str, Any]) -> None:
    replicate_id = replicate.get("replicate_id")
    parity = replicate.get("restart_parity")
    _require(isinstance(parity, dict), f"{replicate_id}: restart parity is missing")
    _require(parity.get("fresh_process") is True, f"{replicate_id}: restore was not fresh-process")
    _require(
        parity.get("original_process_id") != parity.get("restored_process_id"),
        f"{replicate_id}: restore reused the original process",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("protocol", help="verify the sealed scientific protocol")
    binding = subparsers.add_parser("binding", help="verify a pre-run formal implementation binding")
    binding.add_argument("path", type=Path)
    result = subparsers.add_parser("result", help="verify a development or formal raw result")
    result.add_argument("path", type=Path)
    result.add_argument("--binding", type=Path, help="required for a formal result")
    seed = subparsers.add_parser("seed", help="print one declared derived seed")
    seed.add_argument("namespace")
    seed.add_argument("master_seed", type=int)
    seed.add_argument("index", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested verifier."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "protocol":
            verify_protocol()
            print(f"WM-001 protocol valid: {_file_sha256(PROTOCOL_PATH)}")
        elif arguments.command == "binding":
            verify_binding(arguments.path)
            print(f"WM-001 formal binding valid: {_file_sha256(arguments.path)}")
        elif arguments.command == "result":
            verify_result(arguments.path, arguments.binding)
            print(f"WM-001 result envelope valid: {_file_sha256(arguments.path)}")
        else:
            print(derive_seed(arguments.namespace, arguments.master_seed, arguments.index))
    except Violation as exc:
        print(f"WM-001 verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
