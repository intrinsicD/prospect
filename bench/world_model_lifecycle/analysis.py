"""Independent, raw-row analysis for the sealed WM-001 protocol.

The experiment runner is allowed to *store* ``aggregate_metrics`` and
``gate_results``, but those fields are never inputs to this module.  Every
contrast below is reconstructed from replicate predictive-metric and executed
episode rows.  Structural gates are reconstructed from custody, update,
checkpoint, and restart-parity rows.

The public functions do not mutate their arguments:

``recompute_aggregate_metrics(result)``
    Return raw-result-schema-compatible aggregate metric rows.

``evaluate_gates(result)``
    Return the ordered K0--K7 prefix, stopping at the first failed gate.

``analyze_result(result)``
    Return both products plus audit findings.  This is the normal entry point.

Development runs use the same semantics and thresholds but are never
claim-eligible.  Their budgets may be smaller if conditions remain paired.
Formal runs must contain the exact eight predeclared replicates and budgets.
"""

from __future__ import annotations

import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"

TASK_A = "pendulum_normal_torque"
TASK_B = "pendulum_reversed_torque"
TASK_IRRELEVANT = "independent_phase_oscillator"
FORMAL_SEEDS = (
    140647545,
    2239253745,
    3333612762,
    4269572592,
    2151457732,
    4034984701,
    2426483518,
    2833322658,
)
DEVELOPMENT_SEEDS = (560818116, 1392377688)
COVERAGE_SEMANTICS = "wm001-mixture-pit-binary64-count-v1"
T_CRITICAL_N8 = 2.364624251

# Two-sided 95% Student-t critical values indexed by sample size.  WM-001 only
# makes formal inferences at n=8.  The smaller values are descriptive support
# for development runs; n=1 is represented by a zero-width descriptive range.
_T_CRITICAL_BY_N = {
    2: 12.706204736,
    3: 4.30265273,
    4: 3.182446305,
    5: 2.776445105,
    6: 2.570581836,
    7: 2.446911851,
    8: T_CRITICAL_N8,
}

_BEHAVIOR_CONDITIONS = {
    TASK_A: (
        "cold",
        "after_a",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_b_replay",
        "after_b_naive",
        "random",
        "oracle",
    ),
    TASK_B: (
        "after_a",
        "after_b_replay",
        "after_b_naive",
        "random",
        "oracle",
    ),
}
_PREDICTIVE_CONDITIONS = {
    TASK_A: (
        "cold",
        "after_a",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_b_replay",
        "after_b_naive",
    ),
    TASK_B: ("after_a", "after_b_replay", "after_b_naive"),
    TASK_IRRELEVANT: ("cold", "irrelevant"),
}
_BEHAVIOR_SPLIT = {
    TASK_A: "behavior_evaluation_a",
    TASK_B: "behavior_evaluation_b",
}
_PREDICTIVE_SPLIT = {
    TASK_A: "predictive_validation_a",
    TASK_B: "predictive_validation_b",
    TASK_IRRELEVANT: "predictive_validation_irrelevant",
}
_EPISODE_CONTRACTS = frozenset(
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
        *(("behavior_evaluation_a", TASK_A, condition, condition) for condition in _BEHAVIOR_CONDITIONS[TASK_A]),
        *(("behavior_evaluation_b", TASK_B, condition, condition) for condition in _BEHAVIOR_CONDITIONS[TASK_B]),
    }
)
_PREDICTIVE_CONTRACTS = frozenset(
    {
        (
            _PREDICTIVE_SPLIT[task],
            task,
            condition,
            condition,
        )
        for task, conditions in _PREDICTIVE_CONDITIONS.items()
        for condition in conditions
    }
)
_COLLECTION_SPLITS = ("collect_a", "collect_b", "collect_irrelevant")
_VALIDATION_SPLITS = (
    "predictive_validation_a",
    "predictive_validation_b",
    "predictive_validation_irrelevant",
)
_HELD_OUT_SPLITS = frozenset(
    {
        "predictive_validation_a",
        "predictive_validation_b",
        "predictive_validation_irrelevant",
        "behavior_evaluation_a",
        "behavior_evaluation_b",
    }
)
_REQUIRED_COMPONENTS = frozenset(
    {
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
    }
)
_COMMITTED_PHASES = (
    "train_a",
    "train_a_irrelevant",
    "train_a_corrupted",
    "train_b_replay",
    "train_b_naive",
)
_EXPECTED_ELIGIBLE_SPLITS = {
    "train_a": frozenset({"collect_a"}),
    "train_a_corrupted": frozenset({"collect_a"}),
    "train_a_irrelevant": frozenset({"collect_irrelevant"}),
    "train_b_replay": frozenset({"collect_a", "collect_b"}),
    "train_b_naive": frozenset({"collect_b"}),
}
_METRIC_UNITS = {
    "irrelevant_source_nll_improvement_after_irrelevant_vs_cold": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_frozen": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_corrupted": "nats/target-dimension",
    "a_nll_improvement_after_a_vs_irrelevant": "nats/target-dimension",
    "a_irrelevant_nll_improvement_vs_frozen": "nats/target-dimension",
    "a_after_a_interval_90_coverage": "fraction",
    "a_return_improvement_after_a_vs_cold": "return",
    "a_return_improvement_after_a_vs_frozen": "return",
    "a_return_improvement_after_a_vs_irrelevant": "return",
    "a_irrelevant_return_improvement_vs_frozen": "return",
    "a_after_a_oracle_normalized_score": "fraction",
    "a_oracle_vs_random_return_gap": "return",
    "b_nll_improvement_after_b_replay_vs_before_b": "nats/target-dimension",
    "b_return_improvement_after_b_replay_vs_before_b": "return",
    "b_return_improvement_after_b_naive_vs_before_b": "return",
    "a_naive_forgetting_return_drop": "return",
    "retained_a_gain_fraction": "fraction",
    "a_replay_vs_naive_return_advantage": "return",
    "b_replay_minus_naive_return": "return",
    "shared_parameter_violations": "count",
    "restart_component_hash_mismatches": "count",
    "restart_identity_or_lineage_mismatches": "count",
    "restart_prediction_max_abs_difference": "absolute-difference",
    "restart_action_max_abs_difference": "absolute-difference",
    "restart_episode_return_max_abs_difference": "return",
}


def _load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sealed protocol must be a JSON object")
    return value


def _protocol_sha256() -> str:
    return sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe representation, including bad floats."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return f"non-finite:{value!r}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _evidence_digest(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _mean_return(
    replicate: Mapping[str, Any],
    task_id: str,
    condition: str,
) -> float | None:
    split = _BEHAVIOR_SPLIT[task_id]
    rows = [
        row
        for row in _as_rows(replicate.get("episodes"))
        if row.get("task_id") == task_id and row.get("split") == split and row.get("condition") == condition
    ]
    returns = [float(row["return"]) for row in rows if _is_finite_number(row.get("return"))]
    if not rows or len(returns) != len(rows):
        return None
    return statistics.fmean(returns)


def _predictive_row(
    replicate: Mapping[str, Any],
    task_id: str,
    condition: str,
) -> Mapping[str, Any] | None:
    rows = [
        row
        for row in _as_rows(replicate.get("predictive_metrics"))
        if row.get("task_id") == task_id
        and row.get("split") == _PREDICTIVE_SPLIT[task_id]
        and row.get("condition") == condition
        and row.get("checkpoint_id") == condition
    ]
    return rows[0] if len(rows) == 1 else None


def _replicate_numeric_values(replicate: Mapping[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Reconstruct every predeclared replicate-level numerical contrast."""

    replicate_id = str(replicate.get("replicate_id", "<missing>"))
    findings: list[dict[str, Any]] = []
    values: dict[str, float] = {}

    prediction = {
        (task, condition): _predictive_row(replicate, task, condition)
        for task, conditions in _PREDICTIVE_CONDITIONS.items()
        for condition in conditions
    }
    returns = {
        (task, condition): _mean_return(replicate, task, condition)
        for task, conditions in _BEHAVIOR_CONDITIONS.items()
        for condition in conditions
    }

    def nll(task: str, condition: str) -> float | None:
        row = prediction[(task, condition)]
        if row is None or not _is_finite_number(row.get("mixture_nll_nats_per_target_dimension")):
            return None
        return float(row["mixture_nll_nats_per_target_dimension"])

    def coverage(task: str, condition: str) -> float | None:
        row = prediction[(task, condition)]
        if row is None or row.get("coverage_semantics") != COVERAGE_SEMANTICS:
            return None
        covered = row.get("interval_90_covered_target_count")
        total = row.get("coverage_target_count")
        transition_count = row.get("transition_count")
        stored_fraction = row.get("interval_90_coverage")
        if (
            not isinstance(covered, int)
            or isinstance(covered, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or not isinstance(transition_count, int)
            or isinstance(transition_count, bool)
            or total <= 0
            or covered < 0
            or covered > total
            or total != 4 * transition_count
            or not _is_finite_number(stored_fraction)
            or float(stored_fraction) != covered / total
        ):
            return None
        return covered / total

    required_inputs = {
        "irrelevant_source_nll_improvement_after_irrelevant_vs_cold": (
            nll(TASK_IRRELEVANT, "cold"),
            nll(TASK_IRRELEVANT, "irrelevant"),
        ),
        "a_nll_improvement_after_a_vs_frozen": (nll(TASK_A, "frozen"), nll(TASK_A, "after_a")),
        "a_nll_improvement_after_a_vs_corrupted": (nll(TASK_A, "corrupted"), nll(TASK_A, "after_a")),
        "a_nll_improvement_after_a_vs_irrelevant": (
            nll(TASK_A, "irrelevant"),
            nll(TASK_A, "after_a"),
        ),
        "a_irrelevant_nll_improvement_vs_frozen": (
            nll(TASK_A, "frozen"),
            nll(TASK_A, "irrelevant"),
        ),
        "a_return_improvement_after_a_vs_cold": (
            returns[(TASK_A, "after_a")],
            returns[(TASK_A, "cold")],
        ),
        "a_return_improvement_after_a_vs_frozen": (
            returns[(TASK_A, "after_a")],
            returns[(TASK_A, "frozen")],
        ),
        "a_return_improvement_after_a_vs_irrelevant": (
            returns[(TASK_A, "after_a")],
            returns[(TASK_A, "irrelevant")],
        ),
        "a_irrelevant_return_improvement_vs_frozen": (
            returns[(TASK_A, "irrelevant")],
            returns[(TASK_A, "frozen")],
        ),
        "a_oracle_vs_random_return_gap": (
            returns[(TASK_A, "oracle")],
            returns[(TASK_A, "random")],
        ),
        "b_nll_improvement_after_b_replay_vs_before_b": (
            nll(TASK_B, "after_a"),
            nll(TASK_B, "after_b_replay"),
        ),
        "b_return_improvement_after_b_replay_vs_before_b": (
            returns[(TASK_B, "after_b_replay")],
            returns[(TASK_B, "after_a")],
        ),
        "b_return_improvement_after_b_naive_vs_before_b": (
            returns[(TASK_B, "after_b_naive")],
            returns[(TASK_B, "after_a")],
        ),
        "a_naive_forgetting_return_drop": (
            returns[(TASK_A, "after_a")],
            returns[(TASK_A, "after_b_naive")],
        ),
        "a_replay_vs_naive_return_advantage": (
            returns[(TASK_A, "after_b_replay")],
            returns[(TASK_A, "after_b_naive")],
        ),
        "b_replay_minus_naive_return": (
            returns[(TASK_B, "after_b_replay")],
            returns[(TASK_B, "after_b_naive")],
        ),
    }
    for name, operands in required_inputs.items():
        left, right = operands
        if left is not None and right is not None and math.isfinite(left) and math.isfinite(right):
            values[name] = float(left - right)
        else:
            findings.append(
                _finding(
                    "error",
                    "K0",
                    "missing_numeric_source",
                    f"{replicate_id}: cannot reconstruct {name} from complete finite raw rows",
                    replicate_id,
                )
            )

    after_a_coverage = coverage(TASK_A, "after_a")
    if after_a_coverage is not None and math.isfinite(after_a_coverage):
        values["a_after_a_interval_90_coverage"] = after_a_coverage
        after_a_row = prediction[(TASK_A, "after_a")]
        assert after_a_row is not None
        values["_a_after_a_interval_90_covered_target_count"] = float(
            after_a_row["interval_90_covered_target_count"]
        )
        values["_a_after_a_coverage_target_count"] = float(
            after_a_row["coverage_target_count"]
        )
    else:
        findings.append(
            _finding(
                "error",
                "K0",
                "missing_numeric_source",
                f"{replicate_id}: missing finite after-A task-A coverage",
                replicate_id,
            )
        )

    oracle = returns[(TASK_A, "oracle")]
    random_return = returns[(TASK_A, "random")]
    after_a = returns[(TASK_A, "after_a")]
    if oracle is not None and random_return is not None and after_a is not None and oracle != random_return:
        values["a_after_a_oracle_normalized_score"] = (after_a - random_return) / (oracle - random_return)
    else:
        findings.append(
            _finding(
                "error",
                "K4",
                "undefined_oracle_normalization",
                f"{replicate_id}: oracle-normalized task-A score has a missing or zero denominator",
                replicate_id,
            )
        )

    cold = returns[(TASK_A, "cold")]
    after_b_replay_a = returns[(TASK_A, "after_b_replay")]
    if cold is not None and after_a is not None and after_b_replay_a is not None and after_a > cold:
        values["retained_a_gain_fraction"] = (after_b_replay_a - cold) / (after_a - cold)
    else:
        findings.append(
            _finding(
                "error",
                "K6",
                "invalid_retention_denominator",
                f"{replicate_id}: retention requires A_after_A - A_cold > 0",
                replicate_id,
            )
        )

    updates = _phase_index(replicate)
    shared_violations = 0
    train_a = updates.get("train_a")
    for phase in ("train_b_replay", "train_b_naive"):
        update = updates.get(phase)
        if train_a is None or update is None:
            shared_violations += 1
            continue
        if update.get("predecessor_parameter_sha256") != train_a.get("committed_parameter_sha256"):
            shared_violations += 1
        if update.get("predecessor_model_version") != train_a.get("committed_model_version"):
            shared_violations += 1
    for component in _as_rows(replicate.get("checkpoint_components")):
        component_id = str(component.get("component_id", "")).lower()
        if "task_a" in component_id or "task_b" in component_id:
            shared_violations += 1
    values["shared_parameter_violations"] = float(shared_violations)

    parity = replicate.get("restart_parity")
    if isinstance(parity, Mapping):
        component_mismatches = parity.get("component_hash_mismatches")
        if isinstance(component_mismatches, list):
            values["restart_component_hash_mismatches"] = float(len(component_mismatches))
        for source, name in (
            ("identity_or_lineage_mismatches", "restart_identity_or_lineage_mismatches"),
            ("prediction_max_abs_difference", "restart_prediction_max_abs_difference"),
            ("action_max_abs_difference", "restart_action_max_abs_difference"),
            ("episode_return_max_abs_difference", "restart_episode_return_max_abs_difference"),
        ):
            if _is_finite_number(parity.get(source)):
                values[name] = float(parity[source])
    return values, findings


def _aggregate(name: str, values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} requires at least one finite replicate value")
    mean = statistics.fmean(values)
    if len(values) == 1:
        lower = upper = mean
    else:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        critical = _T_CRITICAL_BY_N.get(len(values), 1.959963985)
        margin = critical * standard_error
        lower, upper = mean - margin, mean + margin
    return {
        "name": name,
        "unit": _METRIC_UNITS[name],
        "replicate_values": [float(value) for value in values],
        "mean": float(mean),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def _recompute_metrics_and_findings(
    result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, float]]]:
    replicates = _as_rows(result.get("replicates"))
    replicate_values: list[dict[str, float]] = []
    findings: list[dict[str, Any]] = []
    for replicate in replicates:
        values, local_findings = _replicate_numeric_values(replicate)
        replicate_values.append(values)
        findings.extend(local_findings)

    aggregates: list[dict[str, Any]] = []
    for name in _METRIC_UNITS:
        metric_values = [row[name] for row in replicate_values if name in row]
        if len(metric_values) == len(replicates) and metric_values:
            aggregates.append(_aggregate(name, metric_values))
    return aggregates, findings, replicate_values


def recompute_aggregate_metrics(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recompute aggregate rows, ignoring aggregates already stored in ``result``."""

    metrics, _, _ = _recompute_metrics_and_findings(result)
    return metrics


def _finding(
    severity: str,
    gate: str,
    code: str,
    message: str,
    replicate_id: str | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "gate": gate,
        "code": code,
        "message": message,
        "replicate_id": replicate_id,
    }


def _check(
    name: str,
    observed: int | float | bool | str,
    comparator: str,
    threshold: int | float | bool | str,
    passed: bool,
    evidence: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
        "raw_evidence_sha256": _evidence_digest(evidence),
    }


def _zero_violation_check(name: str, violations: Sequence[str]) -> dict[str, Any]:
    return _check(name, len(violations), "eq", 0, not violations, list(violations))


def _phase_index(replicate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = _as_rows(replicate.get("updates"))
    counts = Counter(str(row.get("phase")) for row in rows)
    return {
        phase: row
        for row in rows
        if isinstance(row.get("phase"), str) and counts[str(row.get("phase"))] == 1
        for phase in (str(row["phase"]),)
    }


def _derive_seed(namespace: str, master_seed: int, index: int) -> int:
    payload = f"WM-001|1.13.0|{namespace}|{master_seed}|{index}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)


def _episode_seed_schedule_violations(
    replicate: Mapping[str, Any],
    master_seed: int,
) -> list[str]:
    """Bind every environment reset, in execution order, to its named stream."""

    replicate_id = str(replicate.get("replicate_id", "<missing>"))
    episodes = _as_rows(replicate.get("episodes"))
    violations: list[str] = []
    split_namespaces = {
        "collect_a": "collect_a_episode",
        "collect_b": "collect_b_episode",
        "collect_irrelevant": "collect_irrelevant_episode",
        "predictive_validation_a": "predictive_validation_a_episode",
        "predictive_validation_b": "predictive_validation_b_episode",
        "predictive_validation_irrelevant": "predictive_validation_irrelevant_episode",
    }
    for split, namespace in split_namespaces.items():
        rows = [episode for episode in episodes if episode.get("split") == split]
        actual = [episode.get("reset_seed") for episode in rows]
        expected = [_derive_seed(namespace, master_seed, index) for index in range(len(rows))]
        if actual != expected:
            violations.append(f"{replicate_id}: {split} reset-seed order differs from {namespace}")

    behavior_namespaces = {
        "behavior_evaluation_a": "behavior_evaluation_a_episode",
        "behavior_evaluation_b": "behavior_evaluation_b_episode",
    }
    for split, namespace in behavior_namespaces.items():
        conditions = {str(episode.get("condition")) for episode in episodes if episode.get("split") == split}
        for condition in sorted(conditions):
            rows = [
                episode
                for episode in episodes
                if episode.get("split") == split and episode.get("condition") == condition
            ]
            actual = [episode.get("reset_seed") for episode in rows]
            expected = [_derive_seed(namespace, master_seed, index) for index in range(len(rows))]
            if actual != expected:
                violations.append(f"{replicate_id}: {split}/{condition} reset-seed order differs from {namespace}")
    return violations


def _run_matrix_violations(replicate: Mapping[str, Any]) -> list[str]:
    """Reject any execution or metric row outside the sealed run matrix."""

    replicate_id = str(replicate.get("replicate_id", "<missing>"))
    violations: list[str] = []
    for episode in _as_rows(replicate.get("episodes")):
        contract = (
            episode.get("split"),
            episode.get("task_id"),
            episode.get("condition"),
            episode.get("checkpoint_id"),
        )
        if contract not in _EPISODE_CONTRACTS:
            violations.append(
                f"{replicate_id}: episode {episode.get('episode_id')} is outside "
                f"the sealed split/task/condition/checkpoint matrix"
            )
    for metric in _as_rows(replicate.get("predictive_metrics")):
        contract = (
            metric.get("split"),
            metric.get("task_id"),
            metric.get("condition"),
            metric.get("checkpoint_id"),
        )
        if contract not in _PREDICTIVE_CONTRACTS:
            violations.append(f"{replicate_id}: predictive row {contract!r} is outside the sealed matrix")
    for run in _as_rows(replicate.get("policy_runs")):
        contract = (
            run.get("split"),
            run.get("task_id"),
            run.get("condition"),
            run.get("checkpoint_id"),
        )
        controller = run.get("controller_kind")
        expected_controller = (
            "uniform_random"
            if run.get("condition") in {"collection_random", "validation_random", "random"}
            else "cem_oracle"
            if run.get("condition") == "oracle"
            else "cem_learned"
        )
        if contract not in _EPISODE_CONTRACTS or controller != expected_controller:
            violations.append(
                f"{replicate_id}: policy run {run.get('run_id')} is outside the sealed execution/controller matrix"
            )
    return violations


def _policy_and_snapshot_violations(
    replicate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Cross-check declared RNG use and evaluated model-byte lineage."""

    replicate_id = str(replicate.get("replicate_id", "<missing>"))
    policy_violations: list[str] = []
    snapshot_violations: list[str] = []
    episodes = _as_rows(replicate.get("episodes"))
    transitions = {
        str(row.get("transition_id")): row
        for row in _as_rows(replicate.get("transitions"))
        if isinstance(row.get("transition_id"), str)
    }
    episode_by_id = {str(row.get("episode_id")): row for row in episodes if isinstance(row.get("episode_id"), str)}
    runs = _as_rows(replicate.get("policy_runs"))
    runs_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_id[str(run.get("run_id"))].append(run)
    episode_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        episode_runs[str(episode.get("run_id"))].append(episode)
    if set(runs_by_id) != set(episode_runs):
        policy_violations.append(f"{replicate_id}: policy-run IDs differ from executed episode-run IDs")

    master_seed = replicate.get("master_seed")
    namespace_declarations = protocol["seed_schedule"]["namespaces"]
    for run_id, grouped_episodes in episode_runs.items():
        candidates = runs_by_id.get(run_id, ())
        if len(candidates) != 1:
            policy_violations.append(f"{replicate_id}: run {run_id} has {len(candidates)} RNG records, expected one")
            continue
        run = candidates[0]
        ordered_episode_ids = [str(row.get("episode_id")) for row in grouped_episodes]
        reset_seeds = [row.get("reset_seed") for row in grouped_episodes]
        for field in ("task_id", "split", "condition", "checkpoint_id"):
            values = {episode.get(field) for episode in grouped_episodes}
            if len(values) != 1 or run.get(field) not in values:
                policy_violations.append(f"{replicate_id}: run {run_id} {field} differs from its episodes")
        if run.get("episode_ids") != ordered_episode_ids:
            policy_violations.append(f"{replicate_id}: run {run_id} episode order differs")
        if run.get("reset_seeds") != reset_seeds:
            policy_violations.append(f"{replicate_id}: run {run_id} reset seeds differ")
        expected_actions = 200 * len(grouped_episodes)
        if run.get("action_count") != expected_actions:
            policy_violations.append(f"{replicate_id}: run {run_id} action count differs from complete episodes")

        namespace = run.get("seed_namespace")
        seed_index = run.get("seed_index")
        if (
            not isinstance(master_seed, int)
            or not isinstance(namespace, str)
            or namespace not in namespace_declarations
            or not isinstance(seed_index, int)
            or seed_index < 0
            or seed_index >= int(namespace_declarations[namespace]["count"])
        ):
            policy_violations.append(f"{replicate_id}: run {run_id} has an invalid seed reference")
        elif run.get("seed") != _derive_seed(namespace, master_seed, seed_index):
            policy_violations.append(f"{replicate_id}: run {run_id} did not use its declared seed")

        split = run.get("split")
        condition = run.get("condition")
        expected_seed_ref = (
            ("irrelevant_collection_action", 0)
            if split == "collect_irrelevant"
            else ("predictive_validation_irrelevant_action", 0)
            if split == "predictive_validation_irrelevant"
            else ("collection_action", 0 if run.get("task_id") == TASK_A else 1)
            if split in {"collect_a", "collect_b"}
            else ("predictive_validation_action", 0 if run.get("task_id") == TASK_A else 1)
            if split in _VALIDATION_SPLITS
            else ("random_policy_action", 0 if run.get("task_id") == TASK_A else 1)
            if condition == "random"
            else ("planner", 0)
        )
        if (namespace, seed_index) != expected_seed_ref:
            policy_violations.append(
                f"{replicate_id}: run {run_id} uses {(namespace, seed_index)!r}, expected {expected_seed_ref!r}"
            )
        planner_budget = run.get("planner_budget")
        if condition == "random" or split in (*_COLLECTION_SPLITS, *_VALIDATION_SPLITS):
            if run.get("controller_kind") != "uniform_random" or planner_budget is not None:
                policy_violations.append(f"{replicate_id}: run {run_id} random controller metadata differs")
        else:
            expected_kind = "cem_oracle" if condition == "oracle" else "cem_learned"
            if run.get("controller_kind") != expected_kind:
                policy_violations.append(f"{replicate_id}: run {run_id} CEM kind differs")
            if planner_budget != {
                "planning_horizon": 10,
                "optim_steps": 3,
                "num_candidates": 64,
                "top_k": 8,
            }:
                policy_violations.append(f"{replicate_id}: run {run_id} planner budget differs")
        for field in ("rng_start_sha256", "rng_end_sha256"):
            value = run.get(field)
            if not isinstance(value, str) or len(value) != 64:
                policy_violations.append(f"{replicate_id}: run {run_id} has invalid {field}")

        intended: list[float] = []
        applied: list[float] = []
        trace_complete = True
        for episode_id in ordered_episode_ids:
            episode = episode_by_id.get(episode_id)
            if episode is None:
                trace_complete = False
                continue
            episode_intended: list[float] = []
            episode_applied: list[float] = []
            transition_ids = episode.get("transition_ids")
            if not isinstance(transition_ids, list):
                trace_complete = False
                continue
            for transition_id in transition_ids:
                transition = transitions.get(str(transition_id))
                if (
                    transition is None
                    or not _is_finite_number(transition.get("intended_action"))
                    or not _is_finite_number(transition.get("applied_action"))
                ):
                    trace_complete = False
                    continue
                episode_intended.append(float(transition["intended_action"]))
                episode_applied.append(float(transition["applied_action"]))
            intended.extend(episode_intended)
            applied.extend(episode_applied)
            episode_trace = {
                "intended": episode_intended,
                "applied": episode_applied,
            }
            if episode.get("action_trace_sha256") != _evidence_digest(episode_trace):
                policy_violations.append(f"{replicate_id}: episode {episode_id} action trace digest differs")
        run_trace = {
            "episode_ids": ordered_episode_ids,
            "intended": intended,
            "applied": applied,
        }
        if not trace_complete or run.get("action_trace_sha256") != _evidence_digest(run_trace):
            policy_violations.append(f"{replicate_id}: run {run_id} action trace digest differs")

    cem_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("controller_kind") in {"cem_learned", "cem_oracle"}:
            cem_by_task[str(run.get("task_id"))].append(run)
    for task_id, task_runs in cem_by_task.items():
        starts = {run.get("rng_start_sha256") for run in task_runs}
        ends = {run.get("rng_end_sha256") for run in task_runs}
        if len(starts) != 1 or len(ends) != 1:
            policy_violations.append(f"{replicate_id}: paired CEM RNG streams differ for {task_id}")

    snapshots = _as_rows(replicate.get("evaluated_checkpoints"))
    snapshot_by_condition = {
        str(row.get("condition")): row for row in snapshots if isinstance(row.get("condition"), str)
    }
    expected_conditions = {
        "cold",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_a",
        "after_b_replay",
        "after_b_naive",
    }
    if len(snapshots) != len(snapshot_by_condition) or set(snapshot_by_condition) != expected_conditions:
        snapshot_violations.append(f"{replicate_id}: evaluated checkpoint set differs")
    for condition, snapshot in snapshot_by_condition.items():
        if snapshot.get("sha256") != snapshot.get("live_state_sha256"):
            snapshot_violations.append(f"{replicate_id}: {condition} artifact digest differs from live-state digest")
    if all(condition in snapshot_by_condition for condition in ("cold", "frozen")):
        for field in ("parameter_sha256", "live_state_sha256", "model_version", "sha256"):
            if snapshot_by_condition["cold"].get(field) != snapshot_by_condition["frozen"].get(field):
                snapshot_violations.append(f"{replicate_id}: frozen {field} differs from cold without an update")

    update_by_phase = _phase_index(replicate)
    for condition, phase in (
        ("corrupted", "train_a_corrupted"),
        ("irrelevant", "train_a_irrelevant"),
        ("after_a", "train_a"),
        ("after_b_replay", "train_b_replay"),
        ("after_b_naive", "train_b_naive"),
    ):
        snapshot = snapshot_by_condition.get(condition)
        update = update_by_phase.get(phase)
        if snapshot is None or update is None:
            continue
        if snapshot.get("parameter_sha256") != update.get("committed_parameter_sha256"):
            snapshot_violations.append(f"{replicate_id}: {condition} parameters differ from {phase} commit")
        if snapshot.get("live_state_sha256") != update.get("live_state_after_sha256"):
            snapshot_violations.append(f"{replicate_id}: {condition} live state differs from {phase} commit")
        if snapshot.get("model_version") != update.get("committed_model_version"):
            snapshot_violations.append(f"{replicate_id}: {condition} model version differs from {phase} commit")
    for row in _as_rows(replicate.get("predictive_metrics")):
        condition = str(row.get("condition"))
        snapshot = snapshot_by_condition.get(condition)
        if snapshot is None:
            snapshot_violations.append(f"{replicate_id}: predictive condition {condition} has no model artifact")
            continue
        for field in ("parameter_sha256", "live_state_sha256", "model_version"):
            if row.get(field) != snapshot.get(field):
                snapshot_violations.append(
                    f"{replicate_id}: predictive {condition} {field} differs from model artifact"
                )
    for run in runs:
        controller_kind = run.get("controller_kind")
        condition = str(run.get("condition"))
        if controller_kind == "cem_learned":
            snapshot = snapshot_by_condition.get(condition)
            parameter_digest = snapshot.get("parameter_sha256") if snapshot is not None else None
            expected_version = f"wm001-sha256:{parameter_digest}" if isinstance(parameter_digest, str) else None
            if run.get("controller_version") != expected_version:
                snapshot_violations.append(
                    f"{replicate_id}: learned CEM {condition} is not bound to its evaluated parameter snapshot"
                )
        elif (
            controller_kind == "cem_oracle"
            and run.get("controller_version") != "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1"
        ):
            snapshot_violations.append(f"{replicate_id}: oracle CEM controller version differs")
    return policy_violations, snapshot_violations


def _expected_condition_checkpoint(condition: str) -> str:
    return condition


def _structural_checks(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    expected_protocol_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    checks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    findings: list[dict[str, Any]] = []
    lane = result.get("lane")
    replicates = _as_rows(result.get("replicates"))

    envelope_violations: list[str] = []
    if result.get("schema") != "prospect.world-model-lifecycle.raw-result.v9":
        envelope_violations.append("wrong raw-result schema")
    if result.get("experiment_id") != "WM-001":
        envelope_violations.append("wrong experiment_id")
    if result.get("protocol_version") != "1.13.0":
        envelope_violations.append("wrong protocol_version")
    if result.get("protocol_sha256") != expected_protocol_sha256:
        envelope_violations.append("protocol SHA-256 does not match sealed raw bytes")
    if lane not in {"development", "formal"}:
        envelope_violations.append("lane is neither development nor formal")
    if lane == "formal":
        if result.get("claim_eligible") is not True:
            envelope_violations.append("formal lane is not claim_eligible")
        binding = result.get("formal_binding_sha256")
        if not isinstance(binding, str) or len(binding) != 64:
            envelope_violations.append("formal binding SHA-256 is missing")
        execution = result.get("execution")
        if not isinstance(execution, Mapping):
            envelope_violations.append("formal execution record is missing")
        else:
            if execution.get("worktree_clean") is not True:
                envelope_violations.append("formal execution did not bind a clean worktree")
            if execution.get("deterministic_algorithms") is not True:
                envelope_violations.append("formal execution did not enable deterministic algorithms")
    elif lane == "development":
        if result.get("claim_eligible") is not False:
            envelope_violations.append("development lane must be claim-ineligible")
        if result.get("formal_binding_sha256") is not None:
            envelope_violations.append("development lane unexpectedly names a formal binding")
    checks["K0"].append(_zero_violation_check("result_envelope_matches_lane", envelope_violations))

    replicate_violations: list[str] = []
    replicate_ids = [row.get("replicate_id") for row in replicates]
    master_seeds = [row.get("master_seed") for row in replicates]
    if len(replicate_ids) != len(set(replicate_ids)):
        replicate_violations.append("replicate IDs are not unique")
    if lane == "formal":
        if len(replicates) != 8:
            replicate_violations.append(f"formal replicate count is {len(replicates)}, expected 8")
        if tuple(master_seeds) != FORMAL_SEEDS:
            replicate_violations.append("formal master seeds or order differ from the sealed schedule")
    elif lane == "development":
        if not replicates:
            replicate_violations.append("development result has no replicates")
        if len(replicates) > len(DEVELOPMENT_SEEDS):
            replicate_violations.append("development result has more than two replicates")
        if len(master_seeds) != len(set(master_seeds)):
            replicate_violations.append("development master seeds are duplicated")
        if any(seed not in DEVELOPMENT_SEEDS for seed in master_seeds):
            replicate_violations.append("development result uses a non-development master seed")
    checks["K0"].append(_zero_violation_check("replicate_schedule_complete", replicate_violations))

    budget_violations: list[str] = []
    seed_violations: list[str] = []
    numeric_source_violations: list[str] = []
    global_episode_ids: set[str] = set()
    global_transition_ids: set[str] = set()
    global_receipt_ids: set[str] = set()

    for replicate in replicates:
        replicate_id = str(replicate.get("replicate_id", "<missing>"))
        master_seed = replicate.get("master_seed")
        episodes = _as_rows(replicate.get("episodes"))
        transitions = _as_rows(replicate.get("transitions"))
        predictive = _as_rows(replicate.get("predictive_metrics"))
        updates = _as_rows(replicate.get("updates"))
        policy_runs = _as_rows(replicate.get("policy_runs"))
        evaluated_checkpoints = _as_rows(replicate.get("evaluated_checkpoints"))
        if lane == "formal":
            exact_counts = {
                "episodes": (len(episodes), 496),
                "transitions": (len(transitions), 99200),
                "predictive rows": (len(predictive), 12),
                "policy runs": (len(policy_runs), 20),
                "evaluated checkpoints": (len(evaluated_checkpoints), 7),
                "updates including rejected probe": (len(updates), 6),
            }
            for label, (observed_count, expected_count) in exact_counts.items():
                if observed_count != expected_count:
                    budget_violations.append(
                        f"{replicate_id}: formal {label} count is {observed_count}, expected {expected_count}"
                    )
        transition_ids_by_split: dict[str, set[str]] = defaultdict(set)
        for transition in transitions:
            transition_id = transition.get("transition_id")
            split = transition.get("split")
            if isinstance(transition_id, str) and isinstance(split, str):
                transition_ids_by_split[split].add(transition_id)
            scaled_target = transition.get("scaled_target")
            if (
                not isinstance(scaled_target, list)
                or len(scaled_target) != 4
                or not all(_is_finite_number(value) for value in scaled_target)
            ):
                numeric_source_violations.append(
                    f"{replicate_id}: {transition_id} has no finite four-value scaled target"
                )
            else:
                target_digest = sha256(struct.pack("<4d", *(float(value) for value in scaled_target))).hexdigest()
                if transition.get("target_sha256") != target_digest:
                    numeric_source_violations.append(
                        f"{replicate_id}: {transition_id} target digest differs from raw target"
                    )
            if not isinstance(transition.get("run_id"), str) or not transition.get("run_id"):
                budget_violations.append(f"{replicate_id}: {transition_id} has no run ID")
            expected_context = {
                TASK_A: 0.0,
                TASK_B: 1.0,
                TASK_IRRELEVANT: 2.0,
            }.get(str(transition.get("task_id")))
            if transition.get("task_context") != expected_context:
                budget_violations.append(f"{replicate_id}: {transition_id} task context differs")
            intended = transition.get("intended_action")
            applied = transition.get("applied_action")
            if _is_finite_number(intended) and _is_finite_number(applied):
                expected_applied = (
                    -float(intended)
                    if transition.get("task_id") == TASK_B
                    else 0.0
                    if transition.get("task_id") == TASK_IRRELEVANT
                    else float(intended)
                )
                if not math.isclose(
                    float(applied),
                    expected_applied,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    budget_violations.append(
                        f"{replicate_id}: {transition_id} applied action differs from task semantics"
                    )

        if isinstance(master_seed, int):
            declared = {
                str(row.get("namespace")): row.get("values") for row in _as_rows(replicate.get("derived_seeds"))
            }
            for namespace, declaration in protocol["seed_schedule"]["namespaces"].items():
                expected = [_derive_seed(namespace, master_seed, index) for index in range(int(declaration["count"]))]
                if declared.get(namespace) != expected:
                    seed_violations.append(f"{replicate_id}: derived namespace {namespace} differs")
            if set(declared) != set(protocol["seed_schedule"]["namespaces"]):
                seed_violations.append(f"{replicate_id}: derived seed namespace set differs")
            seed_violations.extend(_episode_seed_schedule_violations(replicate, master_seed))
        else:
            seed_violations.append(f"{replicate_id}: master_seed is not an integer")
        policy_issues, snapshot_issues = _policy_and_snapshot_violations(
            replicate,
            protocol,
        )
        seed_violations.extend(policy_issues)
        budget_violations.extend(snapshot_issues)
        budget_violations.extend(_run_matrix_violations(replicate))

        # Complete fixed-horizon real episodes are required in both lanes.
        for episode in episodes:
            episode_id = episode.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                budget_violations.append(f"{replicate_id}: episode with missing ID")
                continue
            if episode_id in global_episode_ids:
                budget_violations.append(f"{replicate_id}: duplicate global episode ID {episode_id}")
            global_episode_ids.add(episode_id)
            if episode.get("environment_steps") != 200:
                budget_violations.append(f"{replicate_id}: {episode_id} does not contain 200 environment steps")
            if not _is_finite_number(episode.get("return")):
                numeric_source_violations.append(f"{replicate_id}: {episode_id} has a non-finite return")

        # Training and prediction-validation data are one set of complete
        # episodes.  Behavior budgets are per evaluated condition.
        for split in (*_COLLECTION_SPLITS, *_VALIDATION_SPLITS):
            count = sum(episode.get("split") == split for episode in episodes)
            if lane == "formal" and count != 8:
                budget_violations.append(f"{replicate_id}: {split} has {count} episodes, expected 8")
            elif lane == "development" and not 0 < count <= 8:
                budget_violations.append(f"{replicate_id}: {split} development episode count {count} is invalid")

        task_reset_sequences: dict[str, list[int]] = {}
        for task, conditions in _BEHAVIOR_CONDITIONS.items():
            split = _BEHAVIOR_SPLIT[task]
            for condition in conditions:
                rows = [
                    episode
                    for episode in episodes
                    if episode.get("task_id") == task
                    and episode.get("split") == split
                    and episode.get("condition") == condition
                ]
                expected_count = 32 if lane == "formal" else None
                if expected_count is not None and len(rows) != expected_count:
                    budget_violations.append(
                        f"{replicate_id}: {task}/{condition} has {len(rows)} behavior episodes, expected 32"
                    )
                elif lane == "development" and not 0 < len(rows) <= 32:
                    budget_violations.append(
                        f"{replicate_id}: {task}/{condition} development behavior count {len(rows)} is invalid"
                    )
                if any(row.get("checkpoint_id") != _expected_condition_checkpoint(condition) for row in rows):
                    budget_violations.append(
                        f"{replicate_id}: {task}/{condition} behavior checkpoint ID differs from condition"
                    )
                seeds = [int(row["reset_seed"]) for row in rows if isinstance(row.get("reset_seed"), int)]
                if len(seeds) != len(rows) or len(seeds) != len(set(seeds)):
                    budget_violations.append(
                        f"{replicate_id}: {task}/{condition} behavior reset seeds are missing or duplicated"
                    )
                seed_previous = task_reset_sequences.setdefault(task, seeds)
                if seeds != seed_previous:
                    budget_violations.append(
                        f"{replicate_id}: {task}/{condition} ordered behavior reset seeds are not paired"
                    )

        predictive_counts: dict[str, int] = {}
        for task, conditions in _PREDICTIVE_CONDITIONS.items():
            for condition in conditions:
                rows = [
                    row
                    for row in predictive
                    if row.get("task_id") == task
                    and row.get("split") == _PREDICTIVE_SPLIT[task]
                    and row.get("condition") == condition
                    and row.get("checkpoint_id") == condition
                ]
                if len(rows) != 1:
                    numeric_source_violations.append(
                        f"{replicate_id}: expected one predictive row for {task}/{condition}, found {len(rows)}"
                    )
                    continue
                predictive_row = rows[0]
                count = predictive_row.get("transition_count")
                if not isinstance(count, int) or count <= 0:
                    numeric_source_violations.append(
                        f"{replicate_id}: invalid predictive transition count for {task}/{condition}"
                    )
                else:
                    predictive_counts[f"{task}/{condition}"] = count
                    if lane == "formal" and count != 1600:
                        budget_violations.append(
                            f"{replicate_id}: {task}/{condition} predicts {count} rows, expected 1600"
                        )
                for field in (
                    "mixture_nll_nats_per_target_dimension",
                    "normalized_rmse",
                    "interval_90_coverage",
                ):
                    if not _is_finite_number(predictive_row.get(field)):
                        numeric_source_violations.append(f"{replicate_id}: {task}/{condition} has non-finite {field}")
                covered_target_count = predictive_row.get("interval_90_covered_target_count")
                coverage_target_count = predictive_row.get("coverage_target_count")
                stored_coverage = predictive_row.get("interval_90_coverage")
                if predictive_row.get("coverage_semantics") != COVERAGE_SEMANTICS:
                    numeric_source_violations.append(
                        f"{replicate_id}: {task}/{condition} has the wrong coverage semantics"
                    )
                if (
                    not isinstance(covered_target_count, int)
                    or isinstance(covered_target_count, bool)
                    or not isinstance(coverage_target_count, int)
                    or isinstance(coverage_target_count, bool)
                    or coverage_target_count <= 0
                    or covered_target_count < 0
                    or covered_target_count > coverage_target_count
                ):
                    numeric_source_violations.append(
                        f"{replicate_id}: {task}/{condition} has invalid coverage counts"
                    )
                elif (
                    not isinstance(count, int)
                    or coverage_target_count != 4 * count
                    or not _is_finite_number(stored_coverage)
                    or float(stored_coverage) != covered_target_count / coverage_target_count
                ):
                    numeric_source_violations.append(
                        f"{replicate_id}: {task}/{condition} coverage counts, transition count, and fraction disagree"
                    )
                evidence_file = predictive_row.get("prediction_evidence_file")
                evidence_bytes = predictive_row.get("prediction_evidence_bytes")
                if (
                    not isinstance(evidence_file, str)
                    or not evidence_file
                    or "/" in evidence_file
                    or "\\" in evidence_file
                ):
                    numeric_source_violations.append(
                        f"{replicate_id}: {task}/{condition} predictive evidence filename is unsafe"
                    )
                if not isinstance(evidence_bytes, int) or evidence_bytes <= 0:
                    numeric_source_violations.append(
                        f"{replicate_id}: {task}/{condition} predictive evidence byte count is invalid"
                    )
            task_counts = {
                predictive_counts.get(f"{task}/{condition}")
                for condition in conditions
                if f"{task}/{condition}" in predictive_counts
            }
            if len(task_counts) > 1:
                budget_violations.append(f"{replicate_id}: predictive condition budgets differ for {task}")

        phase_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for update in updates:
            phase_rows[str(update.get("phase"))].append(update)
            receipt_id = update.get("receipt_id")
            if not isinstance(receipt_id, str) or not receipt_id:
                budget_violations.append(f"{replicate_id}: update has no receipt ID")
            elif receipt_id in global_receipt_ids:
                budget_violations.append(f"{replicate_id}: duplicate global receipt ID {receipt_id}")
            else:
                global_receipt_ids.add(receipt_id)
        for phase in (*_COMMITTED_PHASES, "rejected_update_probe"):
            if len(phase_rows.get(phase, ())) != 1:
                budget_violations.append(
                    f"{replicate_id}: phase {phase} has {len(phase_rows.get(phase, ()))} rows, expected 1"
                )

        committed_steps: list[int] = []
        for phase in _COMMITTED_PHASES:
            if len(phase_rows.get(phase, ())) != 1:
                continue
            update = phase_rows[phase][0]
            steps = update.get("optimizer_steps")
            if not isinstance(steps, int):
                budget_violations.append(f"{replicate_id}: {phase} optimizer_steps is not an integer")
                continue
            committed_steps.append(steps)
            if lane == "formal" and steps != 2000:
                budget_violations.append(f"{replicate_id}: {phase} used {steps} optimizer steps, expected 2000")
            elif lane == "development" and not 0 < steps <= 2000:
                budget_violations.append(f"{replicate_id}: {phase} development optimizer budget {steps} is invalid")
            consumed = update.get("eligible_transition_ids")
            expected_consumed = sum(len(transition_ids_by_split[split]) for split in _EXPECTED_ELIGIBLE_SPLITS[phase])
            if not isinstance(consumed, list) or len(consumed) != expected_consumed:
                budget_violations.append(
                    f"{replicate_id}: {phase} consumed "
                    f"{len(consumed) if isinstance(consumed, list) else 'non-list'} IDs, expected {expected_consumed}"
                )
        if committed_steps and len(set(committed_steps)) != 1:
            budget_violations.append(f"{replicate_id}: optimizer budgets are not matched across learned controls")

        probe_rows = phase_rows.get("rejected_update_probe", ())
        if len(probe_rows) == 1 and probe_rows[0].get("optimizer_steps") != 0:
            budget_violations.append(f"{replicate_id}: rejected probe performed optimizer steps")

        manifests = _as_rows(replicate.get("optimizer_batch_manifests"))
        expected_manifests = len(_COMMITTED_PHASES)
        if len(manifests) != expected_manifests:
            budget_violations.append(
                f"{replicate_id}: found {len(manifests)} optimizer manifests, expected {expected_manifests}"
            )
        for index, manifest in enumerate(manifests):
            if manifest.get("phase") not in _COMMITTED_PHASES:
                budget_violations.append(f"{replicate_id}: optimizer manifest {index} has an invalid phase")
            if not isinstance(manifest.get("media_type"), str) or not manifest.get("media_type"):
                budget_violations.append(f"{replicate_id}: optimizer manifest {index} has no media type")
            if not isinstance(manifest.get("bytes"), int) or manifest.get("bytes", 0) <= 0:
                budget_violations.append(f"{replicate_id}: optimizer manifest {index} has no payload bytes")
            digest = manifest.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                budget_violations.append(f"{replicate_id}: optimizer manifest {index} has an invalid digest")
            filename = manifest.get("filename")
            if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
                budget_violations.append(f"{replicate_id}: optimizer manifest {index} has an unsafe filename")
        if {manifest.get("phase") for manifest in manifests} != set(_COMMITTED_PHASES):
            budget_violations.append(f"{replicate_id}: optimizer manifest phases are incomplete or duplicated")
        manifest_by_phase = {str(manifest.get("phase")): manifest for manifest in manifests}
        for phase in _COMMITTED_PHASES:
            phase_updates = phase_rows.get(phase, ())
            if len(phase_updates) != 1 or phase not in manifest_by_phase:
                continue
            update = phase_updates[0]
            manifest = manifest_by_phase[phase]
            if update.get("sampling_manifest_sha256") != manifest.get("sha256"):
                budget_violations.append(f"{replicate_id}: {phase} update and optimizer manifest digests differ")
            permutation_digest = update.get("target_permutation_sha256")
            permutation_file = update.get("target_permutation_file")
            if phase == "train_a_corrupted":
                if not isinstance(permutation_digest, str) or len(permutation_digest) != 64:
                    budget_violations.append(f"{replicate_id}: corrupted control has no target permutation digest")
                if not isinstance(permutation_file, Mapping):
                    budget_violations.append(f"{replicate_id}: corrupted control has no target permutation artifact")
                elif permutation_file.get("sha256") != permutation_digest:
                    budget_violations.append(f"{replicate_id}: corrupted permutation artifact digest differs")
            elif permutation_digest is not None or permutation_file is not None:
                budget_violations.append(f"{replicate_id}: {phase} unexpectedly names a target permutation")
        checkpoint_archive = replicate.get("checkpoint_archive")
        if not isinstance(checkpoint_archive, Mapping):
            budget_violations.append(f"{replicate_id}: checkpoint archive reference is missing")
        else:
            for field in ("filename", "media_type", "sha256"):
                if not isinstance(checkpoint_archive.get(field), str) or not checkpoint_archive.get(field):
                    budget_violations.append(f"{replicate_id}: checkpoint archive {field} is missing")
            if not isinstance(checkpoint_archive.get("bytes"), int) or checkpoint_archive.get("bytes", 0) <= 0:
                budget_violations.append(f"{replicate_id}: checkpoint archive byte count is invalid")

        for transition in transitions:
            transition_id = transition.get("transition_id")
            if isinstance(transition_id, str) and transition_id:
                if transition_id in global_transition_ids:
                    # ID uniqueness is a K1 causal-custody property; K0 only
                    # establishes that the row is syntactically present.
                    pass
                global_transition_ids.add(transition_id)
            else:
                budget_violations.append(f"{replicate_id}: transition with missing ID")

    checks["K0"].append(_zero_violation_check("formal_or_paired_development_budgets", budget_violations))
    checks["K0"].append(_zero_violation_check("derived_seed_schedule_exact", seed_violations))
    checks["K0"].append(_zero_violation_check("raw_numeric_sources_complete_and_finite", numeric_source_violations))

    # K1: identity uniqueness, episode linkage, split isolation, update
    # eligibility, consumed multiset custody, and action-time digest use.
    identity_violations: list[str] = []
    split_violations: list[str] = []
    lineage_violations: list[str] = []
    all_transition_ids: list[str] = []
    all_episode_ids: list[str] = []
    for replicate in replicates:
        replicate_id = str(replicate.get("replicate_id", "<missing>"))
        episodes = _as_rows(replicate.get("episodes"))
        transitions = _as_rows(replicate.get("transitions"))
        transition_by_id: dict[str, Mapping[str, Any]] = {}
        for transition in transitions:
            transition_id = transition.get("transition_id")
            if not isinstance(transition_id, str) or not transition_id:
                continue
            all_transition_ids.append(transition_id)
            if transition_id in transition_by_id:
                identity_violations.append(f"{replicate_id}: duplicate local transition ID {transition_id}")
            transition_by_id[transition_id] = transition
            if transition.get("real_or_imagined") != "real":
                split_violations.append(f"{replicate_id}: imagined row in real namespace {transition_id}")

        referenced: Counter[str] = Counter()
        split_for_reset: dict[tuple[Any, Any], Any] = {}
        for episode in episodes:
            episode_id = episode.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                continue
            all_episode_ids.append(episode_id)
            key = (episode.get("task_id"), episode.get("reset_seed"))
            split_previous = split_for_reset.setdefault(key, episode.get("split"))
            if split_previous != episode.get("split"):
                split_violations.append(f"{replicate_id}: reset {key!r} crosses splits")
            if episode.get("split") in _HELD_OUT_SPLITS:
                if episode.get("learning_allowed") is not False:
                    split_violations.append(f"{replicate_id}: held-out episode {episode_id} allowed learning")
                if episode.get("replay_writes_allowed") is not False:
                    split_violations.append(f"{replicate_id}: held-out episode {episode_id} allowed replay writes")
            transition_ids = episode.get("transition_ids")
            if not isinstance(transition_ids, list) or len(transition_ids) != 200:
                lineage_violations.append(f"{replicate_id}: episode {episode_id} lacks 200 transition IDs")
                continue
            if len(transition_ids) != len(set(transition_ids)):
                identity_violations.append(f"{replicate_id}: episode {episode_id} repeats a transition ID")
            step_indexes: set[int] = set()
            for expected_step, transition_id in enumerate(transition_ids):
                referenced[str(transition_id)] += 1
                episode_transition = transition_by_id.get(str(transition_id))
                if episode_transition is None:
                    lineage_violations.append(
                        f"{replicate_id}: episode {episode_id} references missing transition {transition_id}"
                    )
                    continue
                step = episode_transition.get("step_index")
                if isinstance(step, int):
                    step_indexes.add(step)
                if step != expected_step:
                    lineage_violations.append(f"{replicate_id}: {transition_id} is out of episode step order")
                if episode_transition.get("terminated") is not False:
                    lineage_violations.append(
                        f"{replicate_id}: {transition_id} violates Pendulum's no-termination rule"
                    )
                if episode_transition.get("truncated") is not (expected_step == 199):
                    lineage_violations.append(
                        f"{replicate_id}: {transition_id} has the wrong TimeLimit truncation flag"
                    )
                if episode_transition.get("episode_id") != episode_id:
                    lineage_violations.append(f"{replicate_id}: {transition_id} episode linkage differs")
                if episode_transition.get("task_id") != episode.get("task_id"):
                    lineage_violations.append(f"{replicate_id}: {transition_id} task linkage differs")
                if episode_transition.get("split") != episode.get("split"):
                    lineage_violations.append(f"{replicate_id}: {transition_id} split linkage differs")
                if episode_transition.get("model_version_at_action") != episode.get("model_version"):
                    lineage_violations.append(f"{replicate_id}: {transition_id} action-time model version differs")
                if episode_transition.get("parameter_sha256_at_action") != episode.get("parameter_sha256"):
                    lineage_violations.append(f"{replicate_id}: {transition_id} action-time parameter digest differs")
            if step_indexes != set(range(200)):
                lineage_violations.append(f"{replicate_id}: episode {episode_id} step indexes are not 0..199")
        for transition_id in transition_by_id:
            if referenced[transition_id] != 1:
                lineage_violations.append(
                    f"{replicate_id}: transition {transition_id} is referenced {referenced[transition_id]} times"
                )

        for update in _as_rows(replicate.get("updates")):
            consumed = update.get("eligible_transition_ids")
            if not isinstance(consumed, list):
                lineage_violations.append(f"{replicate_id}: update consumed IDs are not an array")
                continue
            if update.get("eligible_transition_count") != len(consumed):
                lineage_violations.append(f"{replicate_id}: update eligible count differs from its ID list")
            eligible = set(update.get("eligible_splits", ()))
            phase = str(update.get("phase"))
            expected_eligible = _EXPECTED_ELIGIBLE_SPLITS.get(phase)
            if expected_eligible is not None and eligible != set(expected_eligible):
                split_violations.append(f"{replicate_id}: {phase} eligible split declaration differs")
            consumed_digest = update.get("consumed_multiset_sha256")
            if (
                not isinstance(consumed_digest, str)
                or len(consumed_digest) != 64
                or any(character not in "0123456789abcdef" for character in consumed_digest)
            ):
                lineage_violations.append(f"{replicate_id}: {phase} consumed multiset hash is invalid")
            if phase == "rejected_update_probe" and consumed_digest != sha256(b"").hexdigest():
                lineage_violations.append(f"{replicate_id}: rejected probe consumed multiset is not empty")
            if len(consumed) != len(set(consumed)):
                lineage_violations.append(f"{replicate_id}: {phase} receipt repeats a canonical input ID")
            consumed_splits: Counter[str] = Counter()
            for transition_id in consumed:
                consumed_transition = transition_by_id.get(str(transition_id))
                if consumed_transition is None:
                    lineage_violations.append(f"{replicate_id}: {phase} consumes unknown {transition_id}")
                    continue
                split = str(consumed_transition.get("split"))
                consumed_splits[split] += 1
                if split in _HELD_OUT_SPLITS or split not in eligible:
                    split_violations.append(f"{replicate_id}: {phase} consumes ineligible {transition_id}")
            if expected_eligible is not None:
                expected_ids = {
                    transition_id
                    for transition_id, transition in transition_by_id.items()
                    if transition.get("split") in expected_eligible
                }
                if set(str(transition_id) for transition_id in consumed) != expected_ids:
                    lineage_violations.append(
                        f"{replicate_id}: {phase} receipt does not bind the exact eligible canonical input set"
                    )
            if phase == "train_b_replay" and consumed:
                if consumed_splits["collect_a"] != consumed_splits["collect_b"]:
                    split_violations.append(f"{replicate_id}: B replay is not globally balanced A/B")
            optimizer_steps = update.get("optimizer_steps")
            expected_samples = (
                0
                if update.get("status") == "rejected"
                else int(optimizer_steps) * 5 * 256
                if isinstance(optimizer_steps, int)
                else -1
            )
            if update.get("consumed_sample_count") != expected_samples:
                lineage_violations.append(f"{replicate_id}: {phase} consumed sample count differs")

    if len(all_transition_ids) != len(set(all_transition_ids)):
        identity_violations.append("real transition IDs are not globally unique")
    if len(all_episode_ids) != len(set(all_episode_ids)):
        identity_violations.append("episode IDs are not globally unique")
    checks["K1"].append(_zero_violation_check("real_identity_uniqueness", identity_violations))
    checks["K1"].append(_zero_violation_check("heldout_and_replay_split_isolation", split_violations))
    checks["K1"].append(_zero_violation_check("episode_update_and_action_digest_lineage", lineage_violations))

    # K2: exact transaction state change, branch ancestry, rejected probe, and
    # downstream evaluation digest use.
    transaction_violations: list[str] = []
    branch_violations: list[str] = []
    rejection_violations: list[str] = []
    downstream_violations: list[str] = []
    for replicate in replicates:
        replicate_id = str(replicate.get("replicate_id", "<missing>"))
        phases = _phase_index(replicate)
        for phase in _COMMITTED_PHASES:
            phase_update = phases.get(phase)
            if phase_update is None:
                transaction_violations.append(f"{replicate_id}: unique {phase} receipt missing")
                continue
            if phase_update.get("status") != "committed":
                transaction_violations.append(f"{replicate_id}: {phase} is not committed")
            predecessor = phase_update.get("predecessor_parameter_sha256")
            candidate = phase_update.get("candidate_parameter_sha256")
            committed = phase_update.get("committed_parameter_sha256")
            if predecessor == committed:
                transaction_violations.append(f"{replicate_id}: {phase} did not change parameter digest")
            if candidate != committed:
                transaction_violations.append(f"{replicate_id}: {phase} candidate and commit digests differ")
            if phase_update.get("predecessor_model_version") == phase_update.get("committed_model_version"):
                transaction_violations.append(f"{replicate_id}: {phase} did not advance model version")
            if phase_update.get("live_state_before_sha256") == phase_update.get("live_state_after_sha256"):
                transaction_violations.append(f"{replicate_id}: {phase} did not change live-state bytes")

        train_a = phases.get("train_a")
        corrupted = phases.get("train_a_corrupted")
        irrelevant = phases.get("train_a_irrelevant")
        replay = phases.get("train_b_replay")
        naive = phases.get("train_b_naive")
        if train_a is not None and corrupted is not None:
            if corrupted.get("predecessor_parameter_sha256") != train_a.get("predecessor_parameter_sha256"):
                branch_violations.append(f"{replicate_id}: corrupted control did not fork from cold parameters")
            if corrupted.get("predecessor_model_version") != train_a.get("predecessor_model_version"):
                branch_violations.append(f"{replicate_id}: corrupted control did not fork from cold version")
            if corrupted.get("live_state_before_sha256") != train_a.get("live_state_before_sha256"):
                branch_violations.append(f"{replicate_id}: corrupted control did not fork from cold compound state")
        if train_a is not None and irrelevant is not None:
            if irrelevant.get("predecessor_parameter_sha256") != train_a.get("predecessor_parameter_sha256"):
                branch_violations.append(f"{replicate_id}: irrelevant control did not fork from cold parameters")
            if irrelevant.get("predecessor_model_version") != train_a.get("predecessor_model_version"):
                branch_violations.append(f"{replicate_id}: irrelevant control did not fork from cold version")
            if irrelevant.get("live_state_before_sha256") != train_a.get("live_state_before_sha256"):
                branch_violations.append(f"{replicate_id}: irrelevant control did not fork from cold compound state")
        if train_a is not None:
            for name, branch_update in (("train_b_replay", replay), ("train_b_naive", naive)):
                if branch_update is None:
                    continue
                if branch_update.get("predecessor_parameter_sha256") != train_a.get("committed_parameter_sha256"):
                    branch_violations.append(f"{replicate_id}: {name} did not fork from post-A parameters")
                if branch_update.get("predecessor_model_version") != train_a.get("committed_model_version"):
                    branch_violations.append(f"{replicate_id}: {name} did not fork from post-A version")
                if branch_update.get("live_state_before_sha256") != train_a.get("live_state_after_sha256"):
                    branch_violations.append(f"{replicate_id}: {name} did not fork from post-A compound state")

        probe = phases.get("rejected_update_probe")
        if probe is None:
            rejection_violations.append(f"{replicate_id}: unique rejected probe missing")
        else:
            if probe.get("status") != "rejected":
                rejection_violations.append(f"{replicate_id}: rejected probe status is not rejected")
            if probe.get("live_state_before_sha256") != probe.get("live_state_after_sha256"):
                rejection_violations.append(f"{replicate_id}: rejected probe changed live-state bytes")
            before_full_state = probe.get("full_state_before_sha256")
            after_full_state = probe.get("full_state_after_sha256")
            before_reference = probe.get("full_state_before_file")
            after_reference = probe.get("full_state_after_file")
            if before_full_state != after_full_state:
                rejection_violations.append(f"{replicate_id}: rejected probe changed full-state bytes")
            for label, reference, expected_reference_digest in (
                ("before", before_reference, before_full_state),
                ("after", after_reference, after_full_state),
            ):
                if (
                    not isinstance(reference, Mapping)
                    or reference.get("sha256") != expected_reference_digest
                    or reference.get("media_type") != "application/vnd.prospect.wm001.rejected-probe-state+json"
                    or not isinstance(reference.get("bytes"), int)
                    or int(reference.get("bytes", 0)) <= 0
                    or not isinstance(reference.get("filename"), str)
                    or not reference.get("filename")
                    or "/" in str(reference.get("filename"))
                    or "\\" in str(reference.get("filename"))
                ):
                    rejection_violations.append(
                        f"{replicate_id}: rejected probe {label} full-state reference is invalid"
                    )
            if probe.get("predecessor_parameter_sha256") != probe.get("committed_parameter_sha256"):
                rejection_violations.append(f"{replicate_id}: rejected probe changed committed parameters")
            if probe.get("predecessor_model_version") != probe.get("committed_model_version"):
                rejection_violations.append(f"{replicate_id}: rejected probe changed model version")
            if (
                probe.get("optimizer_steps") != 0
                or probe.get("eligible_transition_count") != 0
                or probe.get("consumed_sample_count") != 0
            ):
                rejection_violations.append(f"{replicate_id}: rejected probe consumed training budget")

        expected_digest: dict[str, Any] = {}
        if train_a is not None:
            expected_digest["cold"] = train_a.get("predecessor_parameter_sha256")
            expected_digest["frozen"] = train_a.get("predecessor_parameter_sha256")
            expected_digest["after_a"] = train_a.get("committed_parameter_sha256")
        if corrupted is not None:
            expected_digest["corrupted"] = corrupted.get("committed_parameter_sha256")
        if irrelevant is not None:
            expected_digest["irrelevant"] = irrelevant.get("committed_parameter_sha256")
        if replay is not None:
            expected_digest["after_b_replay"] = replay.get("committed_parameter_sha256")
        if naive is not None:
            expected_digest["after_b_naive"] = naive.get("committed_parameter_sha256")
        for episode in _as_rows(replicate.get("episodes")):
            episode_condition = str(episode.get("condition"))
            if (
                episode_condition in expected_digest
                and episode.get("parameter_sha256") != expected_digest[episode_condition]
            ):
                downstream_violations.append(
                    f"{replicate_id}: {episode_condition} episode did not use its committed parameter digest"
                )
        predictive_rows = _as_rows(replicate.get("predictive_metrics"))
        for condition in (
            "cold",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_a",
            "after_b_replay",
            "after_b_naive",
        ):
            matching = [row for row in predictive_rows if row.get("condition") == condition]
            if not matching:
                downstream_violations.append(f"{replicate_id}: no downstream predictive row for {condition}")
            elif any(row.get("checkpoint_id") != condition for row in matching):
                downstream_violations.append(f"{replicate_id}: {condition} predictive row names a different checkpoint")

    checks["K2"].append(_zero_violation_check("committed_updates_change_exact_candidate_bytes", transaction_violations))
    checks["K2"].append(_zero_violation_check("update_branch_ancestry_exact", branch_violations))
    checks["K2"].append(_zero_violation_check("rejected_probe_is_byte_stable", rejection_violations))
    checks["K2"].append(_zero_violation_check("committed_digest_used_downstream", downstream_violations))

    # K7 row-level checkpoint and fresh-process checks.  Actual component
    # payload bytes must additionally be reopened by the artifact-level audit.
    checkpoint_violations: list[str] = []
    parity_violations: list[str] = []
    for replicate in replicates:
        replicate_id = str(replicate.get("replicate_id", "<missing>"))
        components = _as_rows(replicate.get("checkpoint_components"))
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for component in components:
            grouped[str(component.get("checkpoint_id"))].append(component)
            component_id = component.get("component_id")
            if component_id not in _REQUIRED_COMPONENTS:
                checkpoint_violations.append(f"{replicate_id}: checkpoint contains unknown component {component_id!r}")
            digest = component.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                checkpoint_violations.append(
                    f"{replicate_id}: checkpoint component {component_id!r} has an invalid digest"
                )
        complete_groups = []
        for checkpoint_id, rows in grouped.items():
            component_ids = [row.get("component_id") for row in rows]
            if set(component_ids) == _REQUIRED_COMPONENTS and len(component_ids) == len(_REQUIRED_COMPONENTS):
                complete_groups.append(checkpoint_id)
        if not complete_groups:
            checkpoint_violations.append(f"{replicate_id}: no checkpoint has exactly all 15 canonical components")
        parity = replicate.get("restart_parity")
        if not isinstance(parity, Mapping):
            parity_violations.append(f"{replicate_id}: restart parity row missing")
            continue
        if parity.get("fresh_process") is not True:
            parity_violations.append(f"{replicate_id}: restore is not marked fresh-process")
        if parity.get("original_process_id") == parity.get("restored_process_id"):
            parity_violations.append(f"{replicate_id}: restore reused the original process ID")
        for trace_name in ("live_evaluation", "restored_evaluation"):
            trace = parity.get(trace_name)
            if (
                not isinstance(trace, Mapping)
                or trace.get("media_type") != "application/vnd.prospect.wm001.restart-evaluation+json"
                or not isinstance(trace.get("bytes"), int)
                or int(trace.get("bytes", 0)) <= 0
                or not isinstance(trace.get("sha256"), str)
                or len(str(trace.get("sha256"))) != 64
                or not isinstance(trace.get("filename"), str)
                or not trace.get("filename")
                or "/" in str(trace.get("filename"))
                or "\\" in str(trace.get("filename"))
            ):
                parity_violations.append(f"{replicate_id}: {trace_name} raw trace reference is invalid")
        mismatches = parity.get("component_hash_mismatches")
        if not isinstance(mismatches, list) or mismatches:
            parity_violations.append(f"{replicate_id}: checkpoint component hash mismatches are nonzero")
        if parity.get("identity_or_lineage_mismatches") != 0:
            parity_violations.append(f"{replicate_id}: identity or lineage mismatches are nonzero")
        for field in (
            "prediction_max_abs_difference",
            "action_max_abs_difference",
            "episode_return_max_abs_difference",
        ):
            if parity.get(field) != 0.0:
                parity_violations.append(f"{replicate_id}: {field} is not exactly zero")
    checks["K7"].append(_zero_violation_check("checkpoint_component_set_complete", checkpoint_violations))
    checks["K7"].append(_zero_violation_check("fresh_process_state_and_behavior_parity_exact", parity_violations))

    for gate, groups in (
        (
            "K0",
            (
                envelope_violations,
                replicate_violations,
                budget_violations,
                seed_violations,
                numeric_source_violations,
            ),
        ),
        ("K1", (identity_violations, split_violations, lineage_violations)),
        ("K2", (transaction_violations, branch_violations, rejection_violations, downstream_violations)),
        ("K7", (checkpoint_violations, parity_violations)),
    ):
        for group in groups:
            for message in group:
                findings.append(_finding("error", gate, "structural_violation", message))

    findings.append(
        _finding(
            "warning",
            "K1",
            "optimizer_batch_payloads_not_embedded",
            "Canonical eligible input sets and declared hashes were checked, but exact consumption-order "
            "hashes and per-step replay balance require reopening the content-addressed optimizer manifests.",
        )
    )
    findings.append(
        _finding(
            "warning",
            "K3",
            "corrupted_target_payload_not_embedded",
            "The corrupted control row was compared numerically, but preservation of target marginals "
            "requires reopening its content-addressed permutation evidence.",
        )
    )
    findings.append(
        _finding(
            "warning",
            "K3",
            "prediction_payloads_not_embedded",
            "NLL/RMSE/coverage contrasts were recomputed from predictive metric rows; "
            "the separately content-addressed prediction payloads still require artifact-level reopening.",
        )
    )
    findings.append(
        _finding(
            "warning",
            "K0",
            "binding_payload_not_embedded",
            "A formal result's binding digest can be inspected here, but source, dependency, and "
            "pre-launch timestamp parity require the separately sealed binding artifact.",
        )
    )
    findings.append(
        _finding(
            "warning",
            "K7",
            "checkpoint_payloads_not_embedded",
            "Checkpoint rows and parity declarations were checked structurally; "
            "component bytes still require artifact-level hash recomputation.",
        )
    )
    if lane == "development":
        findings.append(
            _finding(
                "info",
                "K0",
                "development_claim_ineligible",
                "Development K3-K6 values are descriptive only: they cannot support the WM-001 claim "
                "or determine whether a formal binding may launch.",
            )
        )
    return dict(checks), findings


def _metric_index(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(metric["name"]): metric for metric in metrics if isinstance(metric.get("name"), str)}


def _metric_check(
    metrics: Mapping[str, Mapping[str, Any]],
    metric_name: str,
    field: str,
    comparator: str,
    threshold: float,
    check_name: str,
) -> dict[str, Any]:
    metric = metrics.get(metric_name)
    if metric is None or not _is_finite_number(metric.get(field)):
        return _check(check_name, "missing", "eq", "present", False, {"metric": metric_name, "field": field})
    observed = float(metric[field])
    comparisons = {
        "eq": observed == threshold,
        "gt": observed > threshold,
        "ge": observed >= threshold,
        "lt": observed < threshold,
        "le": observed <= threshold,
    }
    return _check(check_name, observed, comparator, threshold, comparisons[comparator], metric)


def _coverage_count_gate_checks(
    replicate_values: Sequence[Mapping[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    count_pairs: list[tuple[int, int]] = []
    for row in replicate_values:
        covered_value = row.get("_a_after_a_interval_90_covered_target_count")
        total_value = row.get("_a_after_a_coverage_target_count")
        if (
            not _is_finite_number(covered_value)
            or not _is_finite_number(total_value)
            or not float(covered_value).is_integer()
            or not float(total_value).is_integer()
        ):
            count_pairs = []
            break
        covered = int(covered_value)
        total = int(total_value)
        if total <= 0 or covered < 0 or covered > total:
            count_pairs = []
            break
        count_pairs.append((covered, total))

    complete = bool(replicate_values) and len(count_pairs) == len(replicate_values)
    covered_sum = sum(covered for covered, _ in count_pairs)
    target_sum = sum(total for _, total in count_pairs)
    evidence = {
        "replicate_counts": [
            {"covered_target_count": covered, "coverage_target_count": total}
            for covered, total in count_pairs
        ],
        "covered_target_count_sum": covered_sum,
        "coverage_target_count_sum": target_sum,
    }
    observed: int | float | bool | str = (
        f"{covered_sum}/{target_sum}" if complete and target_sum > 0 else "missing"
    )
    lower_passed = complete and target_sum > 0 and 10 * covered_sum >= 7 * target_sum
    upper_passed = complete and target_sum > 0 and 100 * covered_sum <= 99 * target_sum
    return (
        _check(
            "after_a_interval_coverage_lower_bound",
            observed,
            "10*C >= 7*T",
            "0.70 inclusive",
            lower_passed,
            {**evidence, "left": 10 * covered_sum, "right": 7 * target_sum},
        ),
        _check(
            "after_a_interval_coverage_upper_bound",
            observed,
            "100*C <= 99*T",
            "0.99 inclusive",
            upper_passed,
            {**evidence, "left": 100 * covered_sum, "right": 99 * target_sum},
        ),
    )


def _numeric_gate_checks(
    aggregates: Sequence[Mapping[str, Any]],
    replicate_values: Sequence[Mapping[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    metrics = _metric_index(aggregates)
    checks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    checks["K3"].extend(
        (
            _metric_check(
                metrics,
                "irrelevant_source_nll_improvement_after_irrelevant_vs_cold",
                "mean",
                "ge",
                0.05,
                "irrelevant_source_vs_cold_mean_nll_improvement",
            ),
            _metric_check(
                metrics,
                "irrelevant_source_nll_improvement_after_irrelevant_vs_cold",
                "ci_95_lower",
                "gt",
                0.0,
                "irrelevant_source_vs_cold_nll_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_nll_improvement_after_a_vs_frozen",
                "mean",
                "ge",
                0.05,
                "a_vs_frozen_mean_nll_improvement",
            ),
            _metric_check(
                metrics,
                "a_nll_improvement_after_a_vs_frozen",
                "ci_95_lower",
                "gt",
                0.0,
                "a_vs_frozen_nll_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_nll_improvement_after_a_vs_corrupted",
                "mean",
                "ge",
                0.05,
                "a_vs_corrupted_mean_nll_improvement",
            ),
            _metric_check(
                metrics,
                "a_nll_improvement_after_a_vs_corrupted",
                "ci_95_lower",
                "gt",
                0.0,
                "a_vs_corrupted_nll_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_nll_improvement_after_a_vs_irrelevant",
                "mean",
                "ge",
                0.05,
                "a_vs_irrelevant_mean_nll_improvement",
            ),
            _metric_check(
                metrics,
                "a_nll_improvement_after_a_vs_irrelevant",
                "ci_95_lower",
                "gt",
                0.0,
                "a_vs_irrelevant_nll_improvement_ci_lower",
            ),
            *_coverage_count_gate_checks(replicate_values),
        )
    )

    checks["K4"].extend(
        (
            _metric_check(
                metrics,
                "a_return_improvement_after_a_vs_cold",
                "mean",
                "ge",
                100.0,
                "after_a_vs_cold_mean_return_improvement",
            ),
            _metric_check(
                metrics,
                "a_return_improvement_after_a_vs_cold",
                "ci_95_lower",
                "gt",
                0.0,
                "after_a_vs_cold_return_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_return_improvement_after_a_vs_frozen",
                "mean",
                "ge",
                100.0,
                "after_a_vs_frozen_mean_return_improvement",
            ),
            _metric_check(
                metrics,
                "a_return_improvement_after_a_vs_frozen",
                "ci_95_lower",
                "gt",
                0.0,
                "after_a_vs_frozen_return_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_return_improvement_after_a_vs_irrelevant",
                "mean",
                "ge",
                100.0,
                "after_a_vs_irrelevant_mean_return_improvement",
            ),
            _metric_check(
                metrics,
                "a_return_improvement_after_a_vs_irrelevant",
                "ci_95_lower",
                "gt",
                0.0,
                "after_a_vs_irrelevant_return_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_after_a_oracle_normalized_score",
                "mean",
                "ge",
                0.2,
                "after_a_oracle_normalized_score",
            ),
            _metric_check(
                metrics,
                "a_oracle_vs_random_return_gap",
                "mean",
                "ge",
                100.0,
                "oracle_vs_random_mean_return_gap",
            ),
        )
    )

    checks["K5"].extend(
        (
            _metric_check(
                metrics,
                "b_nll_improvement_after_b_replay_vs_before_b",
                "mean",
                "ge",
                0.05,
                "after_b_replay_vs_before_b_mean_b_nll_improvement",
            ),
            _metric_check(
                metrics,
                "b_nll_improvement_after_b_replay_vs_before_b",
                "ci_95_lower",
                "gt",
                0.0,
                "after_b_replay_vs_before_b_b_nll_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "b_return_improvement_after_b_replay_vs_before_b",
                "mean",
                "ge",
                100.0,
                "after_b_replay_vs_before_b_mean_b_return_improvement",
            ),
            _metric_check(
                metrics,
                "b_return_improvement_after_b_replay_vs_before_b",
                "ci_95_lower",
                "gt",
                0.0,
                "after_b_replay_vs_before_b_b_return_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "b_return_improvement_after_b_naive_vs_before_b",
                "mean",
                "ge",
                100.0,
                "after_b_naive_vs_before_b_mean_b_return_improvement",
            ),
            _metric_check(
                metrics,
                "b_return_improvement_after_b_naive_vs_before_b",
                "ci_95_lower",
                "gt",
                0.0,
                "after_b_naive_vs_before_b_b_return_improvement_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_naive_forgetting_return_drop",
                "mean",
                "ge",
                50.0,
                "naive_a_forgetting_mean_return_drop",
            ),
            _metric_check(
                metrics,
                "a_naive_forgetting_return_drop",
                "ci_95_lower",
                "gt",
                0.0,
                "naive_a_forgetting_return_drop_ci_lower",
            ),
            _metric_check(
                metrics,
                "shared_parameter_violations",
                "mean",
                "eq",
                0.0,
                "shared_parameter_violations",
            ),
        )
    )

    denominator_passed = bool(replicate_values) and all("retained_a_gain_fraction" in row for row in replicate_values)
    checks["K6"].append(
        _check(
            "retention_denominators_positive",
            denominator_passed,
            "eq",
            True,
            denominator_passed,
            [row.get("retained_a_gain_fraction", "undefined") for row in replicate_values],
        )
    )
    checks["K6"].extend(
        (
            _metric_check(
                metrics,
                "retained_a_gain_fraction",
                "mean",
                "ge",
                0.8,
                "mean_seed_level_retained_a_gain_fraction",
            ),
            _metric_check(
                metrics,
                "retained_a_gain_fraction",
                "ci_95_lower",
                "ge",
                0.65,
                "retained_a_gain_fraction_ci_lower",
            ),
            _metric_check(
                metrics,
                "a_replay_vs_naive_return_advantage",
                "mean",
                "ge",
                50.0,
                "replay_vs_naive_mean_a_return_advantage",
            ),
            _metric_check(
                metrics,
                "a_replay_vs_naive_return_advantage",
                "ci_95_lower",
                "gt",
                0.0,
                "replay_vs_naive_a_return_advantage_ci_lower",
            ),
            _metric_check(
                metrics,
                "b_replay_minus_naive_return",
                "mean",
                "ge",
                -25.0,
                "replay_minus_naive_mean_b_return",
            ),
            _metric_check(
                metrics,
                "b_replay_minus_naive_return",
                "ci_95_lower",
                "ge",
                -75.0,
                "replay_minus_naive_b_return_ci_lower",
            ),
        )
    )
    return dict(checks)


def _ordered_gate_results(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    structural: Mapping[str, Sequence[Mapping[str, Any]]],
    numeric: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    gate_rows: list[dict[str, Any]] = []
    all_prior_passed = True
    for declaration in protocol["killing_order"]:
        gate = str(declaration["gate"])
        checks = [dict(row) for row in (*structural.get(gate, ()), *numeric.get(gate, ()))]
        passed = bool(checks) and all(row.get("passed") is True for row in checks)
        still_supported = all_prior_passed and passed
        gate_rows.append(
            {
                "gate": gate,
                "name": str(declaration["name"]),
                "checks": checks,
                "passed": passed,
                # Producer-side analysis may report that every prespecified gate
                # passed, but the protocol withholds the capability claim until
                # the separately finalized artifact passes independent audit.
                "claim_supported": False,
                "stop_reason": None if passed else str(declaration["on_failure"]),
            }
        )
        if not passed:
            break
        all_prior_passed = still_supported
    return gate_rows


def analyze_result(
    result: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently recompute WM-001 aggregates, gates, and audit findings.

    Existing ``result["aggregate_metrics"]`` and ``result["gate_results"]`` are
    deliberately ignored.
    """

    selected_protocol = dict(protocol) if protocol is not None else _load_protocol()
    selected_sha256 = expected_protocol_sha256 or _protocol_sha256()
    aggregates, numeric_findings, replicate_values = _recompute_metrics_and_findings(result)
    structural, structural_findings = _structural_checks(result, selected_protocol, selected_sha256)
    numeric = _numeric_gate_checks(aggregates, replicate_values)
    gates = _ordered_gate_results(result, selected_protocol, structural, numeric)

    findings = [*numeric_findings, *structural_findings]
    first_failed = next((row["gate"] for row in gates if not row["passed"]), None)
    if first_failed is not None:
        findings.append(
            _finding(
                "error",
                first_failed,
                "killing_order_stop",
                f"Analysis stopped at {first_failed}; later numerical endpoints are descriptive only.",
            )
        )
    return {
        "aggregate_metrics": aggregates,
        "gate_results": gates,
        "audit_findings": findings,
    }


def evaluate_gates(
    result: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    expected_protocol_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Return independently recomputed ordered gates, never stored gates."""

    analysis = analyze_result(
        result,
        protocol=protocol,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    gates = analysis["gate_results"]
    if not isinstance(gates, list):
        raise TypeError("analysis gate_results must be a list")
    return gates


__all__ = [
    "T_CRITICAL_N8",
    "analyze_result",
    "evaluate_gates",
    "recompute_aggregate_metrics",
]
