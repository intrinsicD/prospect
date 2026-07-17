"""PI-001: compute-matched exact-reference proposal injection.

PI-001 is a non-gated, one-shot BridgeControl research harness.  It preserves the
production planner and learned model; simulator-optimized sequences enter only through
the bench-only proposal replacement seam in :mod:`bench.proposal_injection.planner`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

from bench.bridge_control.experiment import _train_model as train_balanced_model
from bench.bridge_control.fixture import (
    EVAL_STARTS,
    GOAL_Y,
    BridgeControlEnv,
    ExactBridgeModel,
    load_dataset,
    semantic_hash,
)
from bench.oracle_ladder import experiment as oracle
from prospect.planning import FlatPlanner
from prospect.types import LatentState
from prospect.world_model import FlatWorldModel

from .planner import PlannerState, ProposalInjectionPlanner
from .providers import ExactReferenceProvider, provider_summary
from .trigger import (
    DEFAULT_INPUT as TRIGGER_PARENT,
)
from .trigger import (
    DEFAULT_OUTPUT as TRIGGER_RESULT,
)
from .trigger import (
    analyze_trigger,
)

SCHEMA_VERSION = "proposal-injection-v1"
EXPERIMENT_ID = "PI-001"
MODEL_SEEDS = tuple(range(8))
DEVELOPMENT_SEED = 97
EVAL_STEPS = 14
HORIZON = 12
NATIVE_CANDIDATES = 64
NATIVE_ELITES = 8
NATIVE_ITERATIONS = 3
INJECTION_COUNT = 8
ENLARGED_CANDIDATES = 512
ENLARGED_ELITES = 32
ENLARGED_ITERATIONS = 5
SUCCESS_FLOOR = 0.80
EXACT_FLOOR = 0.95
SIGN_FLOOR = 7
GAP_FLOOR = 0.50
PARITY_ATOL = 1e-10
STATEWISE_TRANSFER_FLOOR = 0.50
FINAL_RETENTION_FLOOR = 0.50

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/proposal_injection/results/PI-001")
PROTOCOL_DOC = Path("docs/research/2026-07-14-proposal-injection-pi001-protocol.md")
LOOP_PROMPT = Path("docs/research/2026-07-14-evidence-driven-rd-loop-prompt.md")
PARENT_PORTFOLIO = Path("docs/research/2026-07-13-predictive-reliability-portfolio.md")
PARENT_OL002 = Path("bench/oracle_ladder_v2/results/OL-002")
PARENT_OL002_RESULTS = PARENT_OL002 / "OL-002-results.json"
PARENT_OL002_PROTOCOL = PARENT_OL002 / "protocol.json"
PARENT_OL002_MANIFEST = PARENT_OL002 / "artifact-manifest.json"
PARENT_DATASET = Path("bench/bridge_control/results/BC-001/datasets/b1_r1_d8.npz")
INPUT_COPY = Path("inputs/BC-001-b1_r1_d8.npz")

PRIMARY_ARMS = (
    "native_no_penalty",
    "privileged_injection",
    "action_permuted_injection",
    "exact_raw",
)
CONDITIONAL_ARMS = ("enlarged_native_search", "time_permuted_injection")

SOURCE_FILES = (
    Path("bench/proposal_injection/__init__.py"),
    Path("bench/proposal_injection/__main__.py"),
    Path("bench/proposal_injection/experiment.py"),
    Path("bench/proposal_injection/planner.py"),
    Path("bench/proposal_injection/providers.py"),
    Path("bench/proposal_injection/trigger.py"),
    Path("tests/test_proposal_injection_experiment.py"),
    Path("tests/test_proposal_injection_planner.py"),
    Path("tests/test_proposal_injection_providers.py"),
    Path("tests/test_proposal_injection_trigger.py"),
    Path("bench/oracle_ladder/audit.py"),
    Path("bench/oracle_ladder/experiment.py"),
    Path("bench/oracle_ladder/models.py"),
    Path("bench/oracle_ladder_v2/experiment.py"),
    Path("bench/bridge_control/experiment.py"),
    Path("bench/bridge_control/fixture.py"),
    Path("src/prospect/agent.py"),
    Path("src/prospect/planning.py"),
    Path("src/prospect/types.py"),
    Path("src/prospect/world_model.py"),
    PROTOCOL_DOC,
    LOOP_PROMPT,
    PARENT_PORTFOLIO,
)

OUTCOME_PATHS = (
    Path(f"{EXPERIMENT_ID}-results.json"),
    Path(f"{EXPERIMENT_ID}-runs.csv"),
    Path(f"{EXPERIMENT_ID}-report.md"),
    Path("artifact-manifest.json"),
)
ARTIFACT_PATHS = (
    Path("protocol.json"),
    Path("input-manifest.json"),
    INPUT_COPY,
    *OUTCOME_PATHS[:-1],
)


@dataclass(frozen=True)
class Evaluation:
    returns: list[float]
    successes: list[bool]
    final_states: list[list[float]]
    action_traces: list[list[list[float]]]
    plan_diagnostics: list[dict[str, object]]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return cast(
        dict[str, Any],
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant),
    )


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {str(path): _file_hash(REPO_ROOT / path) for path in SOURCE_FILES}


def _success(state: np.ndarray | list[float]) -> bool:
    row = np.asarray(state, dtype=float)
    return bool(row[0] >= 0.75 and abs(row[1] - GOAL_Y) <= 0.20)


def _native_planner(
    model: object,
    seed: int,
    *,
    candidates: int = NATIVE_CANDIDATES,
    elites: int = NATIVE_ELITES,
    iterations: int = NATIVE_ITERATIONS,
) -> FlatPlanner:
    return FlatPlanner(
        cast(Any, model),
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=HORIZON,
        candidates=candidates,
        elites=elites,
        iterations=iterations,
        uncertainty_penalty=0.0,
        seed=seed,
    )


def _injection_planner(
    model: FlatWorldModel,
    seed: int,
    provider: ExactReferenceProvider | None,
) -> ProposalInjectionPlanner:
    return ProposalInjectionPlanner(
        model,
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=HORIZON,
        candidates=NATIVE_CANDIDATES,
        elites=NATIVE_ELITES,
        iterations=NATIVE_ITERATIONS,
        uncertainty_penalty=0.0,
        seed=seed,
        injection_count=INJECTION_COUNT if provider is not None else 0,
        sequence_provider=provider,
    )


def _sidecar_encoder(model: FlatWorldModel) -> Callable[[np.ndarray], object]:
    def encode(raw: np.ndarray) -> object:
        return PlannerState(model.encode(raw), raw)

    return encode


def _evaluate(
    planner: FlatPlanner | ProposalInjectionPlanner,
    encode: Callable[[np.ndarray], object],
) -> Evaluation:
    returns: list[float] = []
    successes: list[bool] = []
    final_states: list[list[float]] = []
    action_traces: list[list[list[float]]] = []
    diagnostics: list[dict[str, object]] = []
    for episode_index, start in enumerate(EVAL_STARTS):
        env = BridgeControlEnv()
        obs = env.set_state(start)
        planner.reset()
        total = 0.0
        trace: list[list[float]] = []
        episode_diagnostics: list[dict[str, object]] = []
        for step in range(EVAL_STEPS):
            state = encode(np.asarray(obs.data, dtype=float))
            action = planner.plan(cast(Any, state))
            trace.append(np.asarray(action.data, dtype=float).tolist())
            if isinstance(planner, ProposalInjectionPlanner):
                latest = planner.last_diagnostics
                assert latest is not None
                episode_diagnostics.append(
                    {
                        "episode_index": episode_index,
                        "step": step,
                        **asdict(latest),
                    }
                )
            obs, reward, _ = env.step(action)
            total += reward
        final = np.asarray(obs.data, dtype=float)
        succeeded = _success(final)
        for item in episode_diagnostics:
            item["episode_success"] = succeeded
        diagnostics.extend(episode_diagnostics)
        returns.append(float(total))
        successes.append(succeeded)
        final_states.append(final.tolist())
        action_traces.append(trace)
    return Evaluation(returns, successes, final_states, action_traces, diagnostics)


def _evaluation_difference(left: Evaluation, right: Evaluation) -> dict[str, object]:
    return {
        "max_return_abs": float(
            np.max(np.abs(np.asarray(left.returns) - np.asarray(right.returns)))
        ),
        "max_final_state_abs": float(
            np.max(
                np.abs(np.asarray(left.final_states) - np.asarray(right.final_states))
            )
        ),
        "max_action_abs": float(
            np.max(
                np.abs(np.asarray(left.action_traces) - np.asarray(right.action_traces))
            )
        ),
        "successes_equal": left.successes == right.successes,
    }


def _assert_parity(name: str, left: Evaluation, right: Evaluation) -> dict[str, object]:
    difference = _evaluation_difference(left, right)
    if not difference["successes_equal"] or any(
        float(cast(float, difference[field])) > PARITY_ATOL
        for field in ("max_return_abs", "max_final_state_abs", "max_action_abs")
    ):
        raise ValueError(f"{name} parity failed: {difference}")
    return difference


def _parent_rows() -> dict[str, dict[int, dict[str, Any]]]:
    results = _read_json(REPO_ROOT / PARENT_OL002_RESULTS)
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for row in cast(list[dict[str, Any]], results["rows"]):
        grouped.setdefault(str(row["rung"]), {})[int(row["seed"])] = row
    return grouped


def _evaluation_from_parent(row: dict[str, Any]) -> Evaluation:
    return Evaluation(
        returns=[float(value) for value in row["episode_returns"]],
        successes=[bool(value) for value in row["episode_successes"]],
        final_states=cast(list[list[float]], row["final_states"]),
        action_traces=cast(list[list[list[float]]], row["action_traces"]),
        plan_diagnostics=[],
    )


def _row(
    arm: str,
    seed: int,
    model_sha256: str,
    evaluation: Evaluation,
    *,
    provider: ExactReferenceProvider | None = None,
    candidates: int = NATIVE_CANDIDATES,
    iterations: int = NATIVE_ITERATIONS,
) -> dict[str, object]:
    learned_sequence_evaluations = len(EVAL_STARTS) * EVAL_STEPS * candidates * iterations
    return {
        "arm": arm,
        "seed": seed,
        "model_sha256": model_sha256,
        "mean_eval_return": evaluation.mean_return,
        "success_rate": evaluation.success_rate,
        "episode_returns": evaluation.returns,
        "episode_successes": evaluation.successes,
        "final_states": evaluation.final_states,
        "action_traces": evaluation.action_traces,
        "plan_diagnostics": evaluation.plan_diagnostics,
        "learned_sequence_evaluations": learned_sequence_evaluations,
        "learned_transition_evaluations": learned_sequence_evaluations * HORIZON,
        "provider": provider_summary(provider) if provider is not None else None,
    }


def _rows_by_arm(rows: list[dict[str, object]]) -> dict[str, dict[int, dict[str, object]]]:
    grouped: dict[str, dict[int, dict[str, object]]] = {}
    for row in rows:
        arm = str(row["arm"])
        seed = int(cast(int, row["seed"]))
        if seed in grouped.setdefault(arm, {}):
            raise ValueError(f"duplicate PI-001 row for {arm}/{seed}")
        grouped[arm][seed] = row
    return grouped


def _aggregate(rows: dict[int, dict[str, object]]) -> dict[str, float]:
    return {
        "mean_return": float(
            np.mean([float(cast(float, row["mean_eval_return"])) for row in rows.values()])
        ),
        "success_rate": float(
            np.mean([float(cast(float, row["success_rate"])) for row in rows.values()])
        ),
    }


def _rescue(
    grouped: dict[str, dict[int, dict[str, object]]],
    treatment: str,
) -> dict[str, object]:
    native = grouped["native_no_penalty"]
    exact = grouped["exact_raw"]
    treated = grouped[treatment]
    differences = [
        float(cast(float, treated[seed]["mean_eval_return"]))
        - float(cast(float, native[seed]["mean_eval_return"]))
        for seed in MODEL_SEEDS
    ]
    exact_gaps = [
        float(cast(float, exact[seed]["mean_eval_return"]))
        - float(cast(float, native[seed]["mean_eval_return"]))
        for seed in MODEL_SEEDS
    ]
    denominator = float(np.mean(exact_gaps))
    gap_fraction = float(np.mean(differences) / denominator) if denominator > 0.0 else float("-inf")
    success_rate = _aggregate(treated)["success_rate"]
    positive = sum(value > 0.0 for value in differences)
    recovered = bool(
        positive >= SIGN_FLOOR
        and gap_fraction >= GAP_FLOOR
        and success_rate >= SUCCESS_FLOOR
    )
    return {
        "treatment": treatment,
        "control": "native_no_penalty",
        "positive_seed_count": positive,
        "seed_return_differences": differences,
        "mean_return_difference": float(np.mean(differences)),
        "oracle_gap_fraction_closed": gap_fraction,
        "success_rate": success_rate,
        "recovered": recovered,
    }


def _commitment_audit(rows: dict[int, dict[str, object]]) -> dict[str, object]:
    calls = [
        cast(dict[str, Any], call)
        for row in rows.values()
        for call in cast(list[dict[str, object]], row["plan_diagnostics"])
    ]
    if not calls:
        raise ValueError("privileged injection rows contain no plan diagnostics")

    top = [int(call["injected_top_elite_count"]) > 0 for call in calls]
    first = [bool(call["first_round_best_injected"]) for call in calls]
    final = [bool(call["best_sequence_injected"]) for call in calls]
    succeeded = [bool(call["episode_success"]) for call in calls]

    def rate(mask: list[bool]) -> float:
        return float(np.mean(mask))

    def conditional_success(mask: list[bool]) -> float | None:
        values = [success for success, selected in zip(succeeded, mask, strict=True) if selected]
        return float(np.mean(values)) if values else None

    top_rate = rate(top)
    final_rate = rate(final)
    if top_rate < STATEWISE_TRANSFER_FLOOR:
        classification = "trigger_not_statewise"
    elif final_rate < FINAL_RETENTION_FLOOR:
        classification = "refinement_or_warm_start_loss"
    elif final_rate >= FINAL_RETENTION_FLOOR:
        classification = "open_loop_closed_loop_mismatch"
    else:
        classification = "mixed_or_unresolved"
    return {
        "call_count": len(calls),
        "injected_top_elite_call_fraction": top_rate,
        "first_round_best_injected_call_fraction": rate(first),
        "final_best_injected_call_fraction": final_rate,
        "success_given_top_elite": conditional_success(top),
        "success_given_first_round_best": conditional_success(first),
        "success_given_final_best": conditional_success(final),
        "classification": classification,
        "thresholds": {
            "statewise_transfer_floor": STATEWISE_TRANSFER_FLOOR,
            "final_retention_floor": FINAL_RETENTION_FLOOR,
        },
    }


def _decision(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped = _rows_by_arm(rows)
    missing = [arm for arm in PRIMARY_ARMS if set(grouped.get(arm, {})) != set(MODEL_SEEDS)]
    if missing:
        raise ValueError(f"PI-001 primary rows are incomplete: {missing}")

    aggregates = {arm: _aggregate(seed_rows) for arm, seed_rows in grouped.items()}
    privileged = _rescue(grouped, "privileged_injection")
    permuted = _rescue(grouped, "action_permuted_injection")
    privileged_recovered = bool(privileged["recovered"])
    permuted_recovered = bool(permuted["recovered"])

    if privileged_recovered and not permuted_recovered:
        branch = "specific_privileged_rescue"
        expected_conditional = "enlarged_native_search"
    elif privileged_recovered and permuted_recovered:
        branch = "non_specific_injection_rescue"
        expected_conditional = "time_permuted_injection"
    else:
        branch = "no_privileged_rescue"
        expected_conditional = None

    conditional: dict[str, object] | None = None
    if expected_conditional is not None and expected_conditional in grouped:
        conditional = _rescue(grouped, expected_conditional)
    audit = (
        _commitment_audit(grouped["privileged_injection"])
        if branch == "no_privileged_rescue"
        else None
    )

    if branch == "specific_privileged_rescue" and conditional is not None:
        classification = (
            "proposal_scarcity_with_search_scale_rescue"
            if conditional["recovered"]
            else "privileged_proposal_quality_without_search_scale_rescue"
        )
    elif branch == "non_specific_injection_rescue" and conditional is not None:
        classification = (
            "generic_structured_proposal_rescue"
            if conditional["recovered"]
            else "action_semantics_or_structure_unresolved"
        )
    elif branch == "no_privileged_rescue" and audit is not None:
        classification = f"no_privileged_rescue:{audit['classification']}"
    else:
        classification = f"pending_conditional:{branch}"

    return {
        "primary_branch": branch,
        "expected_conditional_arm": expected_conditional,
        "classification": classification,
        "aggregates": aggregates,
        "rescues": {
            "privileged_injection": privileged,
            "action_permuted_injection": permuted,
            **({expected_conditional: conditional} if conditional is not None else {}),
        },
        "commitment_audit": audit,
        "thresholds": {
            "positive_seed_count": SIGN_FLOOR,
            "oracle_gap_fraction": GAP_FLOOR,
            "success_rate": SUCCESS_FLOOR,
        },
    }


def _execute(dataset_path: Path) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    if dataset.name != "b1_r1_d8":
        raise ValueError("PI-001 requires the frozen BC-001 b1_r1_d8 dataset")
    parents = _parent_rows()
    rows: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    trained: list[tuple[int, FlatWorldModel, str]] = []

    for seed in MODEL_SEEDS:
        model = train_balanced_model(dataset, seed)
        fingerprint = oracle._model_fingerprint(model)
        trained.append((seed, model, fingerprint))

        native = _evaluate(_native_planner(model, seed), model.encode)
        parent_native = _evaluation_from_parent(parents["learned_tsinf_no_penalty"][seed])
        parent_replay = _assert_parity(f"seed {seed} OL-002 native", native, parent_native)

        disabled = _injection_planner(model, seed, None)
        disabled_eval = _evaluate(disabled, model.encode)
        planner_parity = _assert_parity(
            f"seed {seed} disabled injection", native, disabled_eval
        )

        exact_model = ExactBridgeModel()
        exact = _evaluate(
            _native_planner(exact_model, seed),
            lambda raw: LatentState(z=raw.copy()),
        )
        parent_exact = _evaluation_from_parent(parents["exact_raw"][seed])
        exact_replay = _assert_parity(f"seed {seed} OL-002 exact", exact, parent_exact)

        privileged_provider = ExactReferenceProvider(seed, "privileged")
        privileged_planner = _injection_planner(model, seed, privileged_provider)
        privileged = _evaluate(
            privileged_planner,
            _sidecar_encoder(model),
        )

        permuted_provider = ExactReferenceProvider(seed, "action_permuted")
        permuted_planner = _injection_planner(model, seed, permuted_provider)
        permuted = _evaluate(
            permuted_planner,
            _sidecar_encoder(model),
        )

        rows.extend(
            [
                _row("native_no_penalty", seed, fingerprint, native),
                _row(
                    "privileged_injection",
                    seed,
                    fingerprint,
                    privileged,
                    provider=privileged_provider,
                ),
                _row(
                    "action_permuted_injection",
                    seed,
                    fingerprint,
                    permuted,
                    provider=permuted_provider,
                ),
                _row("exact_raw", seed, fingerprint, exact),
            ]
        )
        parity.append(
            {
                "seed": seed,
                "model_sha256": fingerprint,
                "parent_native_replay": parent_replay,
                "disabled_injection_parity": planner_parity,
                "parent_exact_replay": exact_replay,
            }
        )

    exact_success = _aggregate(_rows_by_arm(rows)["exact_raw"])["success_rate"]
    if exact_success < EXACT_FLOOR:
        raise ValueError("PI-001 exact controller missed the frozen success floor")

    primary_decision = _decision(rows)
    branch = primary_decision["primary_branch"]
    if branch == "specific_privileged_rescue":
        for seed, model, fingerprint in trained:
            enlarged = _evaluate(
                _native_planner(
                    model,
                    seed,
                    candidates=ENLARGED_CANDIDATES,
                    elites=ENLARGED_ELITES,
                    iterations=ENLARGED_ITERATIONS,
                ),
                model.encode,
            )
            rows.append(
                _row(
                    "enlarged_native_search",
                    seed,
                    fingerprint,
                    enlarged,
                    candidates=ENLARGED_CANDIDATES,
                    iterations=ENLARGED_ITERATIONS,
                )
            )
    elif branch == "non_specific_injection_rescue":
        for seed, model, fingerprint in trained:
            provider = ExactReferenceProvider(seed, "time_permuted")
            planner = _injection_planner(model, seed, provider)
            evaluation = _evaluate(
                planner,
                _sidecar_encoder(model),
            )
            rows.append(
                _row(
                    "time_permuted_injection",
                    seed,
                    fingerprint,
                    evaluation,
                    provider=provider,
                )
            )

    decision = _decision(rows)
    return {
        "rows": rows,
        "parity": parity,
        "decision": decision,
        "executed_arms": list(PRIMARY_ARMS)
        + ([str(decision["expected_conditional_arm"])] if decision["expected_conditional_arm"] else []),
    }


def _parent_snapshot() -> dict[str, object]:
    protocol = _read_json(REPO_ROOT / PARENT_OL002_PROTOCOL)
    results = _read_json(REPO_ROOT / PARENT_OL002_RESULTS)
    manifest = _read_json(REPO_ROOT / PARENT_OL002_MANIFEST)
    if results.get("experiment_id") != "OL-002" or results.get("status") != "completed_localization":
        raise ValueError("PI-001 requires the sealed completed OL-002 result")
    if results.get("protocol") != protocol:
        raise ValueError("OL-002 result is not bound to its saved protocol")
    expected_artifacts = {
        name: _file_hash(REPO_ROOT / PARENT_OL002 / name)
        for name in cast(dict[str, str], manifest["artifacts"])
    }
    if expected_artifacts != manifest["artifacts"]:
        raise ValueError("OL-002 artifact manifest does not verify")
    current_parent_sources = {
        path: _file_hash(REPO_ROOT / path)
        for path in cast(dict[str, str], protocol["source_sha256"])
    }
    if current_parent_sources != protocol["source_sha256"]:
        raise ValueError("OL-002 frozen source snapshot has drifted")
    trigger = analyze_trigger(REPO_ROOT / TRIGGER_PARENT)
    saved_trigger = _read_json(REPO_ROOT / TRIGGER_RESULT)
    if trigger != saved_trigger or not trigger["decision"]["trigger_reproduced"]:
        raise ValueError("PI-001 trigger is missing, stale, or not reproduced")
    return {
        "experiment_id": "OL-002",
        "results_sha256": _file_hash(REPO_ROOT / PARENT_OL002_RESULTS),
        "protocol_sha256": _file_hash(REPO_ROOT / PARENT_OL002_PROTOCOL),
        "artifact_manifest_sha256": _file_hash(REPO_ROOT / PARENT_OL002_MANIFEST),
        "trigger_sha256": _file_hash(REPO_ROOT / TRIGGER_RESULT),
        "trigger": trigger["decision"],
        "semantic_verification": (
            "independently executed in iteration 1 under pinned NumPy; this snapshot rechecks hashes"
        ),
    }


def protocol_record() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_formal_execution",
        "protocol_document": str(PROTOCOL_DOC),
        "protocol_document_sha256": _file_hash(REPO_ROOT / PROTOCOL_DOC),
        "loop_prompt": str(LOOP_PROMPT),
        "loop_prompt_sha256": _file_hash(REPO_ROOT / LOOP_PROMPT),
        "source_sha256": _source_hashes(),
        "runtime_constraints": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "formal_model_seeds": list(MODEL_SEEDS),
        "development_seed_excluded": DEVELOPMENT_SEED,
        "training": {"steps": 1_800, "batch_size": 64, "dataset": "BC-001/b1_r1_d8"},
        "evaluation": {
            "starts": EVAL_STARTS.tolist(),
            "steps": EVAL_STEPS,
            "success": {"x_min": 0.75, "goal_y": GOAL_Y, "y_tolerance": 0.20},
        },
        "native_planner": {
            "horizon": HORIZON,
            "candidates": NATIVE_CANDIDATES,
            "elites": NATIVE_ELITES,
            "iterations": NATIVE_ITERATIONS,
            "uncertainty_penalty": 0.0,
        },
        "injection": {
            "count": INJECTION_COUNT,
            "iteration": 0,
            "replacement": "last eight fresh rows after full native RNG draw",
            "privileged": "exact 512x5 reference elites from current raw state",
            "negative": "swap action coordinates after identical reference generation",
        },
        "primary_arms": list(PRIMARY_ARMS),
        "conditional": {
            "specific_privileged_rescue": {
                "arm": "enlarged_native_search",
                "candidates": ENLARGED_CANDIDATES,
                "elites": ENLARGED_ELITES,
                "iterations": ENLARGED_ITERATIONS,
            },
            "non_specific_injection_rescue": {"arm": "time_permuted_injection"},
            "no_privileged_rescue": {"arm": None, "analysis": "action_commitment_audit"},
        },
        "thresholds": {
            "parity_atol": PARITY_ATOL,
            "exact_success_floor": EXACT_FLOOR,
            "recovery_success_floor": SUCCESS_FLOOR,
            "positive_seed_floor": SIGN_FLOOR,
            "oracle_gap_fraction_floor": GAP_FLOOR,
            "statewise_transfer_floor": STATEWISE_TRANSFER_FLOOR,
            "final_retention_floor": FINAL_RETENTION_FLOOR,
        },
        "stop_rules": [
            "invalid on parent drift, replay failure, disabled-injection parity failure, or exact ceiling failure",
            "run exactly one iteration-3 branch selected by the frozen primary decision",
            "never enlarge learned search unless privileged injection specifically rescues",
            "do not add arms, seeds, thresholds, or retries after formal execution begins",
            "do not modify production, tasks, ADRs, gates, or sealed parent packages",
        ],
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    copied = output / INPUT_COPY
    dataset = load_dataset(copied)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256(_canonical_json_bytes(protocol_record())).hexdigest(),
        "parent": _parent_snapshot(),
        "copied_dataset": {
            "path": str(INPUT_COPY),
            "file_sha256": _file_hash(copied),
            "semantic_sha256": semantic_hash(dataset),
            "name": dataset.name,
        },
    }


def _assert_safe_output(output: Path) -> None:
    resolved = output.resolve()
    if resolved == REPO_ROOT.resolve() or resolved in REPO_ROOT.resolve().parents:
        raise ValueError("PI-001 output cannot be the repository root or an ancestor")
    if resolved == (REPO_ROOT / PARENT_OL002).resolve():
        raise ValueError("PI-001 cannot overwrite OL-002")
    if (output / f"{EXPERIMENT_ID}-results.json").exists():
        raise FileExistsError("PI-001 is already formal; preserve it and use a new id")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Freeze PI-001 sources, inputs, protocol, and parent hashes without outcomes."""

    _parent_snapshot()
    _assert_safe_output(output)
    if output.exists():
        if any((output / path).exists() for path in OUTCOME_PATHS):
            raise FileExistsError("refusing to erase a partial PI-001 outcome package")
        shutil.rmtree(output)
    (output / INPUT_COPY.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / PARENT_DATASET, output / INPUT_COPY)
    _write_json(output / "protocol.json", protocol_record())
    _write_json(output / "input-manifest.json", _expected_input_manifest(output))
    result = verify(output)
    return {**result, "status": "prepared_only"}


def _csv_text(rows: list[dict[str, object]]) -> str:
    handle = io.StringIO(newline="")
    fields = (
        "arm",
        "seed",
        "model_sha256",
        "mean_eval_return",
        "success_rate",
        "learned_sequence_evaluations",
        "learned_transition_evaluations",
    )
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return handle.getvalue()


def _report_text(results: dict[str, Any]) -> str:
    decision = cast(dict[str, Any], results["decision"])
    aggregates = cast(dict[str, dict[str, float]], decision["aggregates"])
    lines = [
        "# PI-001 proposal-injection result",
        "",
        f"**Status:** {results['status']}  ",
        "**Scope:** non-gated BridgeControl research evidence",
        "",
        "## Outcome",
        "",
        f"Classification: **{decision['classification']}**.",
        "",
        f"Primary branch: `{decision['primary_branch']}`.",
        "",
        "## Executed arms",
        "",
        "| Arm | Mean return | Success |",
        "|---|---:|---:|",
    ]
    for arm in cast(list[str], results["executed_arms"]):
        aggregate = aggregates[arm]
        lines.append(
            f"| `{arm}` | {aggregate['mean_return']:.6f} | "
            f"{100.0 * aggregate['success_rate']:.2f}% |"
        )
    lines.extend(["", "## Rescue decisions", ""])
    for name, rescue in cast(dict[str, dict[str, Any]], decision["rescues"]).items():
        lines.append(
            f"- `{name}`: {rescue['positive_seed_count']}/8 positive seeds, "
            f"{100.0 * float(rescue['oracle_gap_fraction_closed']):.2f}% gap closure, "
            f"{100.0 * float(rescue['success_rate']):.2f}% success — "
            f"{'recovered' if rescue['recovered'] else 'not recovered'}."
        )
    if decision["commitment_audit"] is not None:
        audit = cast(dict[str, Any], decision["commitment_audit"])
        lines.extend(
            [
                "",
                "## Action-commitment audit",
                "",
                f"Audit classification: `{audit['classification']}`.",
                "",
                f"Injected top-elite call fraction: {100.0 * float(audit['injected_top_elite_call_fraction']):.2f}%.",
                f"Final injected-best call fraction: {100.0 * float(audit['final_best_injected_call_fraction']):.2f}%.",
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This result is limited to the frozen authored BridgeControl fixture. "
            "Oracle candidate generation is diagnostic compute and is not a deployable planner input.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifest(output: Path) -> None:
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "artifacts": {str(path): _file_hash(output / path) for path in ARTIFACT_PATHS},
        },
    )


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Execute the frozen PI-001 primary arms and exactly one conditional branch."""

    verification = verify(output)
    if verification["outcomes"] != "prepared_only":
        raise FileExistsError("PI-001 formal execution is one-shot")
    execution = _execute(output / INPUT_COPY)
    protocol = _read_json(output / "protocol.json")
    manifest = _read_json(output / "input-manifest.json")
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_proposal_injection",
        "interpretation_scope": "non-gated BridgeControl research evidence",
        "protocol_sha256": verification["protocol_sha256"],
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
        "protocol": protocol,
        "input_manifest": manifest,
        "repository": {
            "head": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--short")),
            "source_sha256": protocol["source_sha256"],
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        **execution,
    }
    rows = cast(list[dict[str, object]], results["rows"])
    _write_json(output / f"{EXPERIMENT_ID}-results.json", results)
    (output / f"{EXPERIMENT_ID}-runs.csv").write_text(
        _csv_text(rows), encoding="utf-8"
    )
    (output / f"{EXPERIMENT_ID}-report.md").write_text(
        _report_text(results), encoding="utf-8"
    )
    _write_artifact_manifest(output)
    verify(output, require_results=True)
    return results


def _verify_rows(results: dict[str, Any]) -> list[dict[str, object]]:
    rows = cast(list[dict[str, object]], results["rows"])
    grouped = _rows_by_arm(rows)
    expected = list(PRIMARY_ARMS)
    primary_decision = _decision(
        [row for row in rows if str(row["arm"]) in PRIMARY_ARMS]
    )
    conditional = primary_decision["expected_conditional_arm"]
    if conditional is not None:
        expected.append(str(conditional))
    if set(grouped) != set(expected):
        raise ValueError("executed PI-001 arms do not follow the frozen branch")
    if any(set(grouped[arm]) != set(MODEL_SEEDS) for arm in expected):
        raise ValueError("PI-001 rows do not contain exactly eight seeds per arm")
    if results["executed_arms"] != expected:
        raise ValueError("PI-001 executed-arm ordering is not canonical")
    if results["decision"] != _decision(rows):
        raise ValueError("PI-001 decision does not recompute from raw rows")
    return rows


def _verify_outcomes(output: Path, protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    results = _read_json(output / f"{EXPERIMENT_ID}-results.json")
    if results.get("schema_version") != SCHEMA_VERSION or results.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PI-001 result identity does not match")
    if results.get("status") != "completed_proposal_injection":
        raise ValueError("PI-001 result status does not match")
    protocol_sha = sha256(_canonical_json_bytes(protocol)).hexdigest()
    if results.get("protocol") != protocol or results.get("protocol_sha256") != protocol_sha:
        raise ValueError("PI-001 results are not bound to the frozen protocol")
    if results.get("input_manifest") != manifest or results.get("input_manifest_sha256") != _file_hash(
        output / "input-manifest.json"
    ):
        raise ValueError("PI-001 results are not bound to the input manifest")
    if cast(dict[str, Any], results["repository"])["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("PI-001 result source hashes do not match")
    rows = _verify_rows(results)
    if len(cast(list[dict[str, object]], results["parity"])) != len(MODEL_SEEDS):
        raise ValueError("PI-001 parity evidence is incomplete")
    if (output / f"{EXPERIMENT_ID}-runs.csv").read_text(encoding="utf-8") != _csv_text(rows):
        raise ValueError("PI-001 CSV is not canonical")
    if (output / f"{EXPERIMENT_ID}-report.md").read_text(encoding="utf-8") != _report_text(results):
        raise ValueError("PI-001 report is not canonical")
    artifact_manifest = _read_json(output / "artifact-manifest.json")
    expected = {str(path): _file_hash(output / path) for path in ARTIFACT_PATHS}
    if artifact_manifest != {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": expected,
    }:
        raise ValueError("PI-001 artifact manifest does not match")


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
    semantic: bool = False,
) -> dict[str, object]:
    """Verify hashes/derivations, optionally regenerating all semantic outcomes."""

    protocol = _read_json(output / "protocol.json")
    if protocol != protocol_record():
        raise ValueError("saved PI-001 protocol does not match current sources")
    manifest = _read_json(output / "input-manifest.json")
    if manifest != _expected_input_manifest(output):
        raise ValueError("saved PI-001 input manifest or parent snapshot has drifted")
    result_path = output / f"{EXPERIMENT_ID}-results.json"
    if result_path.exists():
        if not all((output / path).exists() for path in OUTCOME_PATHS):
            raise ValueError("PI-001 outcome package is partial")
        _verify_outcomes(output, protocol, manifest)
        outcomes = "verified_results"
        if semantic:
            saved = _read_json(result_path)
            regenerated = _execute(output / INPUT_COPY)
            for field in ("rows", "parity", "decision", "executed_arms"):
                if saved[field] != regenerated[field]:
                    raise ValueError(f"PI-001 semantic regeneration differs in {field}")
            outcomes = "verified_semantic_results"
    elif any((output / path).exists() for path in OUTCOME_PATHS):
        raise ValueError("PI-001 has partial outcomes without a result")
    else:
        outcomes = "prepared_only"
    if require_results and outcomes == "prepared_only":
        raise ValueError("complete PI-001 results are required")
    return {
        "status": "verified",
        "outcomes": outcomes,
        "protocol_sha256": sha256(_canonical_json_bytes(protocol)).hexdigest(),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
    }


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify(output, require_results=True)
    return _read_json(output / f"{EXPERIMENT_ID}-results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="PI-001 proposal-injection experiment")
    parser.add_argument("command", choices=("prepare", "run", "verify", "verify-semantic", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
        print(f"prepare: {result['status']}")
    elif args.command == "run":
        result = run(args.output)
        decision = cast(dict[str, object], result["decision"])
        print(f"run: {decision['classification']}")
    elif args.command == "analyze":
        result = analyze(args.output)
        decision = cast(dict[str, object], result["decision"])
        print(f"analyze: {decision['classification']}")
    else:
        result = verify(
            args.output,
            require_results=True,
            semantic=args.command == "verify-semantic",
        )
        print(f"verify: {result['outcomes']}")


if __name__ == "__main__":
    main()
