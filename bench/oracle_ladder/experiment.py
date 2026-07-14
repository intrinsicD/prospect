"""Prepare, execute, verify, and analyze OL-001.

OL-001 is a non-gated research harness.  It replays the sealed BC-001 balanced
models and performs a sequential simulator-oracle localization ladder without
changing production code or any BC-001-hashed input.
"""

from __future__ import annotations

import csv
import io
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from bench.bridge_control.experiment import (
    DEFAULT_OUTPUT as BC_OUTPUT,
)
from bench.bridge_control.experiment import (
    _train_model as train_balanced_model,
)
from bench.bridge_control.experiment import (
    verify as verify_bridge_control,
)
from bench.bridge_control.fixture import (
    EVAL_STARTS,
    GOAL_Y,
    BridgeControlEnv,
    ExactBridgeModel,
    load_dataset,
    semantic_hash,
    transition_dynamics,
)
from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import LatentState
from prospect.world_model import FlatWorldModel

from .audit import FixedBank, build_fixed_bank, rank_audit
from .models import MeanOraclePrefixModel, MeanWorldModelAdapter

SCHEMA_VERSION = "oracle-ladder-v1"
EXPERIMENT_ID = "OL-001"
MODEL_SEEDS = tuple(range(8))
DEVELOPMENT_SEED = 97
EVAL_STEPS = 14
PLANNER_HORIZON = 12
PLANNER_CANDIDATES = 64
PLANNER_ELITES = 8
PLANNER_ITERATIONS = 3
UNCERTAINTY_PENALTY = 0.03
AUDIT_BANK_SEED = 714_001
REPLAY_ATOL = 1e-10
PARITY_ATOL = 1e-10
SUCCESS_FLOOR = 0.80
EXACT_FLOOR = 0.95
MATERIAL_SIGN_COUNT = 7
MATERIAL_GAP_FRACTION = 0.20
DOMINANT_GAP_FRACTION = 0.50
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/oracle_ladder/results/OL-001")
PROTOCOL_DOC = Path("docs/research/2026-07-14-oracle-prefix-ladder-protocol.md")
PARENT_PROMPT = Path("docs/research/2026-07-13-transformational-research-prompt.md")
PARENT_PORTFOLIO = Path("docs/research/2026-07-13-predictive-reliability-portfolio.md")
PARENT_RESULTS = BC_OUTPUT / "BC-001-results.json"
PARENT_PROTOCOL = BC_OUTPUT / "protocol.json"
PARENT_DATASET_MANIFEST = BC_OUTPUT / "dataset-manifest.json"
PARENT_ARTIFACT_MANIFEST = BC_OUTPUT / "artifact-manifest.json"
PARENT_DATASET = BC_OUTPUT / "datasets/b1_r1_d8.npz"
INPUT_COPY = Path("inputs/BC-001-b1_r1_d8.npz")

ENDPOINT_RUNGS = (
    "learned_tsinf_penalty",
    "learned_tsinf_no_penalty",
    "learned_mean_no_penalty",
    "exact_target_learned_reward",
    "exact_online_learned_reward",
    "exact_online_oracle_reward",
    "exact_raw",
)
PREFIX_RUNGS = (
    "prefix_1_target_no_penalty",
    "prefix_2_target_no_penalty",
    "prefix_4_target_no_penalty",
    "prefix_8_target_no_penalty",
)
ALL_POSSIBLE_RUNGS = ENDPOINT_RUNGS + PREFIX_RUNGS
CONTRAST_SPECS = (
    ("penalty_removal", "learned_tsinf_no_penalty", "learned_tsinf_penalty"),
    ("mean_vs_tsinf", "learned_mean_no_penalty", "learned_tsinf_no_penalty"),
    ("transition_stack", "exact_target_learned_reward", "learned_mean_no_penalty"),
    ("online_target_interface", "exact_online_learned_reward", "exact_target_learned_reward"),
    ("reward_stack", "exact_online_oracle_reward", "exact_online_learned_reward"),
)

SOURCE_FILES = (
    Path("bench/oracle_ladder/__init__.py"),
    Path("bench/oracle_ladder/__main__.py"),
    Path("bench/oracle_ladder/audit.py"),
    Path("bench/oracle_ladder/experiment.py"),
    Path("bench/oracle_ladder/models.py"),
    Path("tests/test_oracle_ladder.py"),
    Path("bench/bridge_control/__init__.py"),
    Path("bench/bridge_control/__main__.py"),
    Path("bench/bridge_control/experiment.py"),
    Path("bench/bridge_control/fixture.py"),
    Path("bench/bridge_control/report.py"),
    Path("src/prospect/agent.py"),
    Path("src/prospect/planning.py"),
    Path("src/prospect/types.py"),
    Path("src/prospect/world_model.py"),
    PROTOCOL_DOC,
    PARENT_PROMPT,
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
    Path(f"{EXPERIMENT_ID}-results.json"),
    Path(f"{EXPERIMENT_ID}-runs.csv"),
    Path(f"{EXPERIMENT_ID}-report.md"),
)


@dataclass(frozen=True)
class Evaluation:
    """Raw repeated-start evaluation for one trained model and rung."""

    returns: list[float]
    successes: list[bool]
    final_states: list[list[float]]
    action_traces: list[list[list[float]]]

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


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return completed.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {str(path): _file_hash(REPO_ROOT / path) for path in SOURCE_FILES}


def _success(final: list[float] | np.ndarray) -> bool:
    return bool(float(final[0]) >= 0.75 and abs(float(final[1]) - GOAL_Y) <= 0.20)


def _planner(model: object, seed: int, penalty: float) -> FlatPlanner:
    return FlatPlanner(
        cast(Any, model),
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=PLANNER_HORIZON,
        candidates=PLANNER_CANDIDATES,
        elites=PLANNER_ELITES,
        iterations=PLANNER_ITERATIONS,
        uncertainty_penalty=penalty,
        seed=seed,
    )


def _audit_hashes() -> dict[str, str]:
    return {str(index): build_fixed_bank(start, seed=AUDIT_BANK_SEED).sha256 for index, start in enumerate(EVAL_STARTS)}


def protocol_record() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_component_localization",
        "protocol_document": str(PROTOCOL_DOC),
        "protocol_document_sha256": _file_hash(REPO_ROOT / PROTOCOL_DOC),
        "parent_prompt": str(PARENT_PROMPT),
        "parent_portfolio": str(PARENT_PORTFOLIO),
        "parent_experiment": "BC-001",
        "source_sha256": _source_hashes(),
        "runtime_constraints": {"numpy_version": np.__version__},
        "development_seed_excluded": DEVELOPMENT_SEED,
        "formal_model_seeds": list(MODEL_SEEDS),
        "training": {
            "dataset": "BC-001/b1_r1_d8",
            "learner": "prospect.world_model.FlatWorldModel",
            "steps": 1_800,
            "batch_size": 64,
            "formal_execution_train_once_per_seed": True,
            "semantic_verifier_regenerates_models": True,
        },
        "planner": {
            "class": "prospect.planning.FlatPlanner",
            "horizon": PLANNER_HORIZON,
            "candidates": PLANNER_CANDIDATES,
            "elites": PLANNER_ELITES,
            "iterations": PLANNER_ITERATIONS,
            "penalty": UNCERTAINTY_PENALTY,
        },
        "evaluation": {
            "steps": EVAL_STEPS,
            "starts": EVAL_STARTS.tolist(),
            "success": {"x_min": 0.75, "goal_y": GOAL_Y, "y_tolerance": 0.20},
            "independent_block": "model_seed",
            "repeated_measure": "four_fixed_starts",
        },
        "endpoint_rungs": list(ENDPOINT_RUNGS),
        "conditional_prefix_rungs": list(PREFIX_RUNGS),
        "contrasts": [
            {"name": name, "treatment": treatment, "control": control} for name, treatment, control in CONTRAST_SPECS
        ],
        "fixed_audit_bank": {
            "horizon": PLANNER_HORIZON,
            "seed": AUDIT_BANK_SEED,
            "hash_by_start": _audit_hashes(),
            "role": "diagnostic_only_never_injected",
        },
        "thresholds": {
            "replay_atol": REPLAY_ATOL,
            "parity_atol": PARITY_ATOL,
            "exact_success_floor": EXACT_FLOOR,
            "recovery_success_floor": SUCCESS_FLOOR,
            "material_positive_seed_count": MATERIAL_SIGN_COUNT,
            "material_gap_fraction": MATERIAL_GAP_FRACTION,
            "dominant_gap_fraction": DOMINANT_GAP_FRACTION,
        },
        "stop_rules": [
            "invalid on parent drift, replay failure, wrapper parity failure, or exact ceiling failure",
            "run intermediate prefixes only when the C-to-D transition contrast is material",
            "do not inject expert sequences or enlarge search in OL-001",
            "do not alter production, tasks, ADRs, gates, parent prompt, or parent portfolio",
        ],
    }


def _parent_snapshot() -> dict[str, object]:
    verification = verify_bridge_control(REPO_ROOT / BC_OUTPUT, require_results=True)
    results = _read_json(REPO_ROOT / PARENT_RESULTS)
    protocol = _read_json(REPO_ROOT / PARENT_PROTOCOL)
    dataset_manifest = _read_json(REPO_ROOT / PARENT_DATASET_MANIFEST)
    dataset_entry = cast(dict[str, Any], dataset_manifest["datasets"])["b1_r1_d8"]
    dataset = load_dataset(REPO_ROOT / PARENT_DATASET)
    if results.get("experiment_id") != "BC-001" or results.get("schema_version") != "bridge-control-v1":
        raise ValueError("verified parent identity or schema is not the frozen BC-001 package")
    if results.get("status") != "aborted_invalid_fixture":
        raise ValueError("verified parent status is not the sealed BC-001 stop result")
    if verification.get("outcomes") != "verified_results":
        raise ValueError("parent verification did not produce verified_results")
    expected_cell = {"name": "b1_r1_d8", "bridge": True, "full_rank": True, "density": 8}
    if dataset.name != "b1_r1_d8" or dataset.cell is None or dataset.cell.as_dict() != expected_cell:
        raise ValueError("parent selected dataset identity or factor cell has drifted")
    semantic = semantic_hash(dataset)
    if semantic != dataset_entry["semantic_hash"]:
        raise ValueError("parent balanced dataset semantic hash does not match its manifest")
    canonical_protocol = sha256(_canonical_json_bytes(protocol)).hexdigest()
    if canonical_protocol != results["protocol_sha256"]:
        raise ValueError("parent result is not bound to its canonical protocol")
    return {
        "experiment_id": results["experiment_id"],
        "schema_version": results["schema_version"],
        "result_status": results["status"],
        "verification": verification,
        "protocol_canonical_sha256": canonical_protocol,
        "protocol_file_sha256": _file_hash(REPO_ROOT / PARENT_PROTOCOL),
        "dataset_manifest_file_sha256": _file_hash(REPO_ROOT / PARENT_DATASET_MANIFEST),
        "artifact_manifest_file_sha256": _file_hash(REPO_ROOT / PARENT_ARTIFACT_MANIFEST),
        "results_file_sha256": _file_hash(REPO_ROOT / PARENT_RESULTS),
        "source_sha256": protocol["source_sha256"],
        "selected_dataset": {
            "name": dataset.name,
            "source_path": str(PARENT_DATASET),
            "file_sha256": _file_hash(REPO_ROOT / PARENT_DATASET),
            "semantic_sha256": semantic,
        },
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    copied = output / INPUT_COPY
    dataset = load_dataset(copied)
    parent = _parent_snapshot()
    selected = cast(dict[str, object], parent["selected_dataset"])
    copied_file_hash = _file_hash(copied)
    copied_semantic_hash = semantic_hash(dataset)
    if dataset.name != selected["name"]:
        raise ValueError("copied dataset name does not match the sealed parent selection")
    if copied_file_hash != selected["file_sha256"]:
        raise ValueError("copied dataset bytes do not match the sealed parent selection")
    if copied_semantic_hash != selected["semantic_sha256"]:
        raise ValueError("copied dataset semantics do not match the sealed parent selection")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256(_canonical_json_bytes(protocol_record())).hexdigest(),
        "parent": parent,
        "copied_dataset": {
            "path": str(INPUT_COPY),
            "file_sha256": copied_file_hash,
            "semantic_sha256": copied_semantic_hash,
            "name": dataset.name,
        },
    }


def _assert_safe_output(output: Path) -> None:
    resolved = output.resolve()
    repo = REPO_ROOT.resolve()
    parent_output = (REPO_ROOT / BC_OUTPUT).resolve()
    if resolved == repo or resolved in repo.parents:
        raise ValueError("OL-001 output cannot be the repository root or one of its ancestors")
    if resolved == parent_output or parent_output in resolved.parents or resolved in parent_output.parents:
        raise ValueError("OL-001 output cannot overwrite or contain the sealed BC-001 package")
    if not output.exists():
        return
    if (output / f"{EXPERIMENT_ID}-results.json").exists():
        raise FileExistsError("OL-001 already has formal results; preserve them and use a new experiment id")
    if any((output / relative).exists() for relative in OUTCOME_PATHS):
        raise FileExistsError("refusing to erase a partial OL-001 outcome package")
    entries = list(output.iterdir())
    if not entries:
        return
    protocol_path = output / "protocol.json"
    if not protocol_path.exists():
        raise ValueError("refusing to erase an unrecognized non-empty output directory")
    protocol = _read_json(protocol_path)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("refusing to erase an output directory owned by another experiment")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Freeze inputs and protocol without producing formal outcomes."""

    _parent_snapshot()
    _assert_safe_output(output)
    if output.exists():
        shutil.rmtree(output)
    (output / INPUT_COPY.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / PARENT_DATASET, output / INPUT_COPY)
    protocol = protocol_record()
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "input-manifest.json", _expected_input_manifest(output))
    verification = verify(output)
    return {**verification, "status": "prepared_only"}


def _model_fingerprint(model: FlatWorldModel) -> str:
    digest = sha256()

    def add(name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype="<f8", order="C")
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(array.tobytes(order="C"))

    networks = [
        ("encoder", model.encoder),
        ("reward", model.reward_head),
        ("inverse", model.inverse_head),
        *[(f"member_{index}", member) for index, member in enumerate(model.members)],
    ]
    for prefix, network in networks:
        for index, weight in enumerate(network.weights):
            add(f"{prefix}.weight.{index}", weight)
        for index, bias in enumerate(network.biases):
            add(f"{prefix}.bias.{index}", bias)
    for index, weight in enumerate(model._target_w):
        add(f"target.weight.{index}", weight)
    for index, bias in enumerate(model._target_b):
        add(f"target.bias.{index}", bias)
    add("obs.mean", model._obs_mean)
    add("obs.var", model._obs_var)
    digest.update(str(model._obs_stats_ready).encode("ascii"))
    return digest.hexdigest()


def _evaluate(agent: Agent) -> Evaluation:
    returns: list[float] = []
    successes: list[bool] = []
    final_states: list[list[float]] = []
    action_traces: list[list[list[float]]] = []
    for start in EVAL_STARTS:
        env = BridgeControlEnv()
        obs = env.set_state(start)
        agent.reset()
        total = 0.0
        trace: list[list[float]] = []
        for _ in range(EVAL_STEPS):
            action = agent.act(obs)
            trace.append(np.asarray(action.data, dtype=float).tolist())
            obs, reward, _ = env.step(action)
            total += reward
        final = np.asarray(obs.data, dtype=float)
        returns.append(float(total))
        successes.append(_success(final))
        final_states.append(final.tolist())
        action_traces.append(trace)
    return Evaluation(returns, successes, final_states, action_traces)


def _native_agent(model: FlatWorldModel, seed: int, penalty: float) -> Agent:
    return Agent(
        encode=lambda obs: model.encode(obs.data),
        planner=_planner(model, seed, penalty),
    )


def _mean_agent(model: FlatWorldModel, seed: int) -> Agent:
    adapter = MeanWorldModelAdapter(model)
    return Agent(
        encode=lambda obs: model.encode(obs.data),
        planner=_planner(adapter, seed, 0.0),
    )


def _prefix_agent(
    model: FlatWorldModel,
    seed: int,
    prefix_steps: int,
    *,
    refresh: Literal["target", "online"] = "target",
    reward_source: Literal["learned", "oracle"] = "learned",
) -> tuple[Agent, MeanOraclePrefixModel]:
    wrapper = MeanOraclePrefixModel(
        model,
        prefix_steps=prefix_steps,
        horizon=PLANNER_HORIZON,
        refresh=refresh,
        reward_source=reward_source,
    )
    return (
        Agent(
            encode=lambda obs: wrapper.initial_state(obs.data),
            planner=_planner(wrapper, seed, 0.0),
        ),
        wrapper,
    )


def _exact_agent(seed: int) -> Agent:
    exact = ExactBridgeModel()
    return Agent(
        encode=lambda obs: LatentState(z=np.asarray(obs.data, dtype=float)),
        planner=_planner(exact, seed, 0.0),
    )


def _evaluation_difference(left: Evaluation, right: Evaluation) -> dict[str, object]:
    return {
        "max_return_abs": float(np.max(np.abs(np.asarray(left.returns) - np.asarray(right.returns)))),
        "max_final_state_abs": float(np.max(np.abs(np.asarray(left.final_states) - np.asarray(right.final_states)))),
        "max_action_abs": float(np.max(np.abs(np.asarray(left.action_traces) - np.asarray(right.action_traces)))),
        "successes_equal": left.successes == right.successes,
    }


def _assert_evaluation_parity(
    name: str,
    left: Evaluation,
    right: Evaluation,
    *,
    check_actions: bool = True,
) -> dict[str, object]:
    difference = _evaluation_difference(left, right)
    fields = ["max_return_abs", "max_final_state_abs"]
    if check_actions:
        fields.append("max_action_abs")
    if not bool(difference["successes_equal"]) or any(_as_float(difference[field]) > PARITY_ATOL for field in fields):
        raise ValueError(f"{name} evaluation parity failed: {difference}")
    return difference


def _parent_balanced_rows() -> dict[int, dict[str, Any]]:
    parent = _read_json(REPO_ROOT / PARENT_RESULTS)
    rows = [row for row in cast(list[dict[str, Any]], parent["rows"]) if row["arm"] == "b1_r1_d8"]
    if {int(row["seed"]) for row in rows} != set(MODEL_SEEDS):
        raise ValueError("parent result does not contain the eight frozen balanced rows")
    return {int(row["seed"]): row for row in rows}


def _assert_parent_replay(seed: int, evaluation: Evaluation) -> dict[str, object]:
    expected = _parent_balanced_rows()[seed]
    return_difference = float(
        np.max(
            np.abs(np.asarray(evaluation.returns, dtype=float) - np.asarray(expected["episode_returns"], dtype=float))
        )
    )
    final_difference = float(
        np.max(
            np.abs(np.asarray(evaluation.final_states, dtype=float) - np.asarray(expected["final_states"], dtype=float))
        )
    )
    success_equal = evaluation.successes == expected["episode_successes"]
    if return_difference > REPLAY_ATOL or final_difference > REPLAY_ATOL or not success_equal:
        raise ValueError(
            f"seed {seed} failed BC-001 replay: return={return_difference}, "
            f"final={final_difference}, successes={success_equal}"
        )
    return {
        "max_return_abs": return_difference,
        "max_final_state_abs": final_difference,
        "successes_equal": success_equal,
    }


def _rung_agent(model: FlatWorldModel, seed: int, rung: str) -> tuple[Agent, object]:
    if rung == "learned_tsinf_penalty":
        return _native_agent(model, seed, UNCERTAINTY_PENALTY), model
    if rung == "learned_tsinf_no_penalty":
        return _native_agent(model, seed, 0.0), model
    if rung == "learned_mean_no_penalty":
        adapter = MeanWorldModelAdapter(model)
        return (
            Agent(
                encode=lambda obs: model.encode(obs.data),
                planner=_planner(adapter, seed, 0.0),
            ),
            adapter,
        )
    if rung == "exact_target_learned_reward":
        return _prefix_agent(model, seed, PLANNER_HORIZON)
    if rung == "exact_online_learned_reward":
        return _prefix_agent(model, seed, PLANNER_HORIZON, refresh="online")
    if rung == "exact_online_oracle_reward":
        return _prefix_agent(
            model,
            seed,
            PLANNER_HORIZON,
            refresh="online",
            reward_source="oracle",
        )
    if rung == "exact_raw":
        exact = ExactBridgeModel()
        return (
            Agent(
                encode=lambda obs: LatentState(z=np.asarray(obs.data, dtype=float)),
                planner=_planner(exact, seed, 0.0),
            ),
            exact,
        )
    if rung.startswith("prefix_"):
        prefix = int(rung.split("_")[1])
        return _prefix_agent(model, seed, prefix)
    raise ValueError(f"unknown rung: {rung}")


def _initial_latent(model: FlatWorldModel, planning_model: object, start: np.ndarray) -> LatentState:
    if isinstance(planning_model, MeanOraclePrefixModel):
        return planning_model.initial_state(start)
    if isinstance(planning_model, ExactBridgeModel):
        return LatentState(z=np.asarray(start, dtype=float).copy())
    return model.encode(start)


def _row(
    rung: str,
    seed: int,
    fingerprint: str,
    evaluation: Evaluation,
) -> dict[str, object]:
    prefix_steps = PLANNER_HORIZON if rung.startswith("exact_") else 0
    if rung.startswith("prefix_"):
        prefix_steps = int(rung.split("_")[1])
    total_candidate_steps = len(EVAL_STARTS) * EVAL_STEPS * PLANNER_ITERATIONS * PLANNER_HORIZON * PLANNER_CANDIDATES
    exact_candidate_steps = (
        len(EVAL_STARTS) * EVAL_STEPS * PLANNER_ITERATIONS * min(prefix_steps, PLANNER_HORIZON) * PLANNER_CANDIDATES
    )
    return {
        "rung": rung,
        "seed": seed,
        "model_sha256": fingerprint,
        "mean_eval_return": evaluation.mean_return,
        "success_rate": evaluation.success_rate,
        "episode_returns": evaluation.returns,
        "episode_successes": evaluation.successes,
        "final_states": evaluation.final_states,
        "action_traces": evaluation.action_traces,
        "planner_candidate_transition_count": total_candidate_steps,
        "oracle_candidate_transition_count": exact_candidate_steps,
    }


def _rows_by_rung(rows: list[dict[str, object]]) -> dict[str, dict[int, dict[str, object]]]:
    grouped: dict[str, dict[int, dict[str, object]]] = {}
    for row in rows:
        rung = str(row["rung"])
        seed = int(cast(int, row["seed"]))
        if seed in grouped.setdefault(rung, {}):
            raise ValueError(f"duplicate result row for {rung}, seed {seed}")
        grouped[rung][seed] = row
    return grouped


def _rung_aggregates(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped = _rows_by_rung(rows)
    return {
        rung: {
            "mean_return": float(np.mean([_as_float(row["mean_eval_return"]) for row in seeds.values()])),
            "success_rate": float(np.mean([_as_float(row["success_rate"]) for row in seeds.values()])),
        }
        for rung, seeds in grouped.items()
    }


def _audit_regret_by_seed(
    audits: list[dict[str, object]],
    rung: str,
) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for audit in audits:
        if audit["rung"] != rung:
            continue
        values.setdefault(int(cast(int, audit["seed"])), []).append(
            _as_float(cast(dict[str, object], audit["diagnostics"])["normalized_selected_regret"])
        )
    return {seed: float(np.mean(items)) for seed, items in values.items()}


def _gap_closure(
    treatment: dict[int, dict[str, object]],
    control: dict[int, dict[str, object]],
    oracle: dict[int, dict[str, object]],
) -> tuple[list[float], float]:
    deltas = [
        _as_float(treatment[seed]["mean_eval_return"]) - _as_float(control[seed]["mean_eval_return"])
        for seed in MODEL_SEEDS
    ]
    gaps = [
        _as_float(oracle[seed]["mean_eval_return"]) - _as_float(control[seed]["mean_eval_return"])
        for seed in MODEL_SEEDS
    ]
    denominator = float(np.sum(gaps))
    fraction = float(np.sum(deltas) / denominator) if denominator > 1e-12 else 0.0
    return deltas, fraction


def _contrast_summaries(
    rows: list[dict[str, object]],
    audits: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped = _rows_by_rung(rows)
    oracle = grouped["exact_raw"]
    summaries: dict[str, dict[str, object]] = {}
    for name, treatment_name, control_name in CONTRAST_SPECS:
        treatment = grouped[treatment_name]
        control = grouped[control_name]
        deltas, gap_fraction = _gap_closure(treatment, control, oracle)
        treatment_regret = _audit_regret_by_seed(audits, treatment_name)
        control_regret = _audit_regret_by_seed(audits, control_name)
        regret_deltas = [control_regret[seed] - treatment_regret[seed] for seed in MODEL_SEEDS]
        positive_count = sum(delta > 0.0 for delta in deltas)
        median_regret_improvement = float(np.median(regret_deltas))
        material = bool(
            positive_count >= MATERIAL_SIGN_COUNT
            and gap_fraction >= MATERIAL_GAP_FRACTION
            and median_regret_improvement > 0.0
        )
        summaries[name] = {
            "treatment": treatment_name,
            "control": control_name,
            "seed_return_differences": deltas,
            "positive_seed_count": positive_count,
            "mean_return_difference": float(np.mean(deltas)),
            "success_rate_difference": float(
                np.mean([_as_float(treatment[seed]["success_rate"]) for seed in MODEL_SEEDS])
                - np.mean([_as_float(control[seed]["success_rate"]) for seed in MODEL_SEEDS])
            ),
            "oracle_gap_fraction_closed": gap_fraction,
            "seed_regret_improvements": regret_deltas,
            "median_normalized_regret_improvement": median_regret_improvement,
            "material": material,
            "dominant": bool(material and gap_fraction >= DOMINANT_GAP_FRACTION),
        }
    return summaries


def _recovery_summary(
    rows: list[dict[str, object]],
    rung: str,
) -> dict[str, object]:
    grouped = _rows_by_rung(rows)
    baseline = grouped["learned_tsinf_penalty"]
    treatment = grouped[rung]
    oracle = grouped["exact_raw"]
    _, gap_fraction = _gap_closure(treatment, baseline, oracle)
    success_rate = float(np.mean([_as_float(treatment[seed]["success_rate"]) for seed in MODEL_SEEDS]))
    return {
        "success_rate": success_rate,
        "oracle_gap_fraction_closed": gap_fraction,
        "recovered": bool(success_rate >= SUCCESS_FLOOR and gap_fraction >= DOMINANT_GAP_FRACTION),
    }


def _prefix_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped = _rows_by_rung(rows)
    depth_names = {
        0: "learned_mean_no_penalty",
        1: "prefix_1_target_no_penalty",
        2: "prefix_2_target_no_penalty",
        4: "prefix_4_target_no_penalty",
        8: "prefix_8_target_no_penalty",
        12: "exact_target_learned_reward",
    }
    executed_depths = [depth for depth, name in depth_names.items() if name in grouped]
    reversal_pairs: list[dict[str, int]] = []
    for left, right in zip(executed_depths[:-1], executed_depths[1:], strict=True):
        reversed_seeds = sum(
            _as_float(grouped[depth_names[right]][seed]["mean_eval_return"])
            < _as_float(grouped[depth_names[left]][seed]["mean_eval_return"])
            for seed in MODEL_SEEDS
        )
        if reversed_seeds >= MATERIAL_SIGN_COUNT:
            reversal_pairs.append({"from_k": left, "to_k": right, "reversed_seed_count": reversed_seeds})
    recovered_depths = [
        depth for depth in executed_depths if bool(_recovery_summary(rows, depth_names[depth])["recovered"])
    ]
    knee: int | None = None
    full_curve = executed_depths == [0, 1, 2, 4, 8, 12]
    if full_curve and recovered_depths and not reversal_pairs:
        candidate = min(recovered_depths)
        later_depths = [depth for depth in executed_depths if depth >= candidate]
        if all(depth in recovered_depths for depth in later_depths):
            knee = candidate
    return {
        "executed_depths": executed_depths,
        "reversal_pairs": reversal_pairs,
        "recovered_depths": recovered_depths,
        "full_curve_executed": full_curve,
        "minimum_recovery_depth": knee,
    }


def _decision(
    rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    parity: list[dict[str, object]],
) -> dict[str, object]:
    contrasts = _contrast_summaries(rows, audits)
    material = [name for name, result in contrasts.items() if bool(result["material"])]
    dominant = [name for name, result in contrasts.items() if bool(result["dominant"])]
    classification = "unresolved_interaction"
    if len(material) == 1:
        classification = material[0]
    elif material:
        classification = "mixed:" + ",".join(material)
    aggregates = _rung_aggregates(rows)
    recoveries = {rung: _recovery_summary(rows, rung) for rung in aggregates}
    return {
        "valid_harness": True,
        "classification": classification,
        "material_components": material,
        "dominant_components": dominant,
        "contrasts": contrasts,
        "rung_aggregates": aggregates,
        "recoveries": recoveries,
        "prefix": _prefix_decision(rows),
        "parity_seed_count": len(parity),
        "sign_rule_one_sided_null_probability": 9 / 256,
    }


def _rung_penalty(rung: str) -> float:
    return UNCERTAINTY_PENALTY if rung == "learned_tsinf_penalty" else 0.0


def _fixed_banks() -> tuple[FixedBank, ...]:
    return tuple(build_fixed_bank(start, seed=AUDIT_BANK_SEED) for start in EVAL_STARTS)


def _audit_rung(
    model: FlatWorldModel,
    seed: int,
    rung: str,
    banks: tuple[FixedBank, ...],
) -> list[dict[str, object]]:
    _, planning_model = _rung_agent(model, seed, rung)
    planner = _planner(planning_model, seed, _rung_penalty(rung))
    records: list[dict[str, object]] = []
    for start_index, (start, bank) in enumerate(zip(EVAL_STARTS, banks, strict=True)):
        state = _initial_latent(model, planning_model, start)
        scores = planner._imagined_returns(state, bank.sequences)
        diagnostics = rank_audit(bank, scores).as_dict()
        records.append(
            {
                "rung": rung,
                "seed": seed,
                "start_index": start_index,
                "bank_sha256": bank.sha256,
                "candidate_scores": np.asarray(scores, dtype=float).tolist(),
                "diagnostics": diagnostics,
            }
        )
    return records


def _score_parity(
    model: FlatWorldModel,
    seed: int,
    banks: tuple[FixedBank, ...],
) -> dict[str, float]:
    direct_mean = MeanWorldModelAdapter(model)
    prefix_zero = MeanOraclePrefixModel(
        model,
        prefix_steps=0,
        horizon=PLANNER_HORIZON,
    )
    oracle_wrapper = MeanOraclePrefixModel(
        model,
        prefix_steps=PLANNER_HORIZON,
        horizon=PLANNER_HORIZON,
        refresh="online",
        reward_source="oracle",
    )
    exact_raw = ExactBridgeModel()
    mean_max = 0.0
    exact_max = 0.0
    for start, bank in zip(EVAL_STARTS, banks, strict=True):
        direct_scores = _planner(direct_mean, seed, 0.0)._imagined_returns(model.encode(start), bank.sequences)
        wrapped_scores = _planner(prefix_zero, seed, 0.0)._imagined_returns(
            prefix_zero.initial_state(start), bank.sequences
        )
        exact_wrapper_scores = _planner(oracle_wrapper, seed, 0.0)._imagined_returns(
            oracle_wrapper.initial_state(start), bank.sequences
        )
        exact_raw_scores = _planner(exact_raw, seed, 0.0)._imagined_returns(
            LatentState(z=np.asarray(start, dtype=float)), bank.sequences
        )
        mean_max = max(mean_max, float(np.max(np.abs(direct_scores - wrapped_scores))))
        exact_max = max(
            exact_max,
            float(np.max(np.abs(exact_wrapper_scores - exact_raw_scores))),
        )
    if mean_max > PARITY_ATOL or exact_max > PARITY_ATOL:
        raise ValueError(f"seed {seed} fixed-bank parity failed: mean={mean_max}, exact={exact_max}")
    return {"mean_k0_score_max_abs": mean_max, "exact_score_max_abs": exact_max}


def _csv_text(rows: list[dict[str, object]]) -> str:
    handle = io.StringIO(newline="")
    fields = (
        "rung",
        "seed",
        "model_sha256",
        "mean_eval_return",
        "success_rate",
        "planner_candidate_transition_count",
        "oracle_candidate_transition_count",
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return handle.getvalue()


def _report_text(results: dict[str, Any]) -> str:
    decision = cast(dict[str, Any], results["decision"])
    aggregates = cast(dict[str, dict[str, float]], decision["rung_aggregates"])
    contrasts = cast(dict[str, dict[str, object]], decision["contrasts"])
    lines = [
        "# OL-001 simulator-oracle localization result",
        "",
        f"**Status:** {results['status']}  ",
        "**Scope:** non-gated BridgeControl research evidence",
        "",
        "## Outcome",
        "",
        f"Classification: **{decision['classification']}**.",
        "",
        "The result passed BC-001 replay, mean-wrapper parity, and exact-wrapper parity. "
        "Interpretation remains limited to this authored fixture.",
        "",
        "## Endpoint and executed-rung results",
        "",
        "| Rung | Mean return | Success |",
        "|---|---:|---:|",
    ]
    executed = cast(list[str], results["executed_rungs"])
    for rung in executed:
        aggregate = aggregates[rung]
        lines.append(f"| `{rung}` | {aggregate['mean_return']:.6f} | {100.0 * aggregate['success_rate']:.2f}% |")
    lines.extend(
        [
            "",
            "## Frozen contrasts",
            "",
            "| Contrast | Positive seeds | Mean return delta | Gap closed | Regret delta | Decision |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, _, _ in CONTRAST_SPECS:
        result = contrasts[name]
        lines.append(
            f"| `{name}` | {result['positive_seed_count']}/8 | "
            f"{_as_float(result['mean_return_difference']):.6f} | "
            f"{100.0 * _as_float(result['oracle_gap_fraction_closed']):.2f}% | "
            f"{_as_float(result['median_normalized_regret_improvement']):.6f} | "
            f"{'material' if result['material'] else 'not material'} |"
        )
    prefix = cast(dict[str, object], decision["prefix"])
    lines.extend(
        [
            "",
            "## Sequential decision",
            "",
            f"- Material components: {decision['material_components']}",
            f"- Dominant components: {decision['dominant_components']}",
            f"- Prefix depths executed: {prefix['executed_depths']}",
            f"- Minimum recovery depth: {prefix['minimum_recovery_depth']}",
            f"- Not run: {results['not_run']}",
            "",
            "## Causal boundary",
            "",
            "A→B is only a penalty-coefficient test. B→C bundles member propagation with "
            "nonlinear reward averaging. C→D is the transition-mean/recursive-refresh stack. "
            "D→E is the online/EMA interface, and E→F is the learned reward composed with "
            "online encoding—not reward-head weights alone. No learned-dynamics/oracle-reward "
            "decoder arm was fabricated.",
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
            "artifacts": {str(relative): _file_hash(output / relative) for relative in ARTIFACT_PATHS},
        },
    )


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Execute the frozen endpoint ladder and its conditional prefix curve."""

    verification = verify(output)
    if verification["outcomes"] != "prepared_only":
        raise FileExistsError("OL-001 formal execution is one-shot; preserve completed results and use a new id")
    protocol = _read_json(output / "protocol.json")
    input_manifest = _read_json(output / "input-manifest.json")
    dataset = load_dataset(output / INPUT_COPY)
    if dataset.name != "b1_r1_d8":
        raise ValueError("OL-001 input must be the frozen b1_r1_d8 dataset")

    banks = _fixed_banks()
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    trained: list[tuple[int, FlatWorldModel, str]] = []
    gated_evaluations: dict[int, dict[str, Evaluation]] = {}

    # Phase 1: no causal endpoint is interpreted or executed until every model
    # replays BC-001 and both wrapper endpoints pass the all-seed parity/ceiling gate.
    for seed in MODEL_SEEDS:
        model = train_balanced_model(dataset, seed)
        fingerprint = _model_fingerprint(model)
        trained.append((seed, model, fingerprint))
        baseline = _evaluate(_native_agent(model, seed, UNCERTAINTY_PENALTY))
        replay = _assert_parent_replay(seed, baseline)
        mean = _evaluate(_mean_agent(model, seed))
        prefix_zero_agent, _ = _prefix_agent(model, seed, 0)
        mean_parity = _assert_evaluation_parity(
            f"seed {seed} mean k=0",
            mean,
            _evaluate(prefix_zero_agent),
        )
        oracle_agent, _ = _prefix_agent(
            model,
            seed,
            PLANNER_HORIZON,
            refresh="online",
            reward_source="oracle",
        )
        oracle = _evaluate(oracle_agent)
        exact = _evaluate(_exact_agent(seed))
        exact_parity = _assert_evaluation_parity(
            f"seed {seed} exact wrapper",
            oracle,
            exact,
        )
        score_parity = _score_parity(model, seed, banks)
        gated_evaluations[seed] = {
            "learned_tsinf_penalty": baseline,
            "learned_mean_no_penalty": mean,
            "exact_online_oracle_reward": oracle,
            "exact_raw": exact,
        }
        parity.append(
            {
                "seed": seed,
                "model_sha256": fingerprint,
                "parent_replay": replay,
                "mean_k0_evaluation": mean_parity,
                "exact_evaluation": exact_parity,
                "fixed_bank_scores": score_parity,
            }
        )

    exact_success = float(np.mean([gated_evaluations[seed]["exact_raw"].success_rate for seed in MODEL_SEEDS]))
    if exact_success < EXACT_FLOOR:
        raise ValueError(f"exact raw ceiling missed the frozen {EXACT_FLOOR:.0%} success floor")

    # Phase 2: with all gates closed, execute the remaining causal endpoints and
    # generate common-bank diagnostics for the full endpoint set.
    for seed, model, fingerprint in trained:
        evaluations = gated_evaluations[seed]
        for rung in ENDPOINT_RUNGS:
            if rung not in evaluations:
                agent, _ = _rung_agent(model, seed, rung)
                evaluations[rung] = _evaluate(agent)
            rows.append(_row(rung, seed, fingerprint, evaluations[rung]))
            audits.extend(_audit_rung(model, seed, rung, banks))

    endpoint_decision = _decision(rows, audits, parity)
    transition = cast(dict[str, Any], endpoint_decision["contrasts"])["transition_stack"]
    run_prefixes = bool(transition["material"])
    if run_prefixes:
        for seed, model, fingerprint in trained:
            for rung in PREFIX_RUNGS:
                agent, _ = _rung_agent(model, seed, rung)
                evaluation = _evaluate(agent)
                rows.append(_row(rung, seed, fingerprint, evaluation))
                audits.extend(_audit_rung(model, seed, rung, banks))

    decision = _decision(rows, audits, parity)
    executed_rungs = list(ENDPOINT_RUNGS) + (list(PREFIX_RUNGS) if run_prefixes else [])
    not_run = [
        *([] if run_prefixes else list(PREFIX_RUNGS)),
        "privileged-sequence candidate injection",
        "enlarged learned-landscape search",
        "MuJoCo P3 replication",
        "production or task activation",
    ]
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_localization",
        "interpretation_scope": "non-gated BridgeControl research evidence",
        "protocol_sha256": verification["protocol_sha256"],
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
        "protocol": protocol,
        "input_manifest": input_manifest,
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
        "executed_rungs": executed_rungs,
        "rows": rows,
        "fixed_audit_banks": [bank.as_dict(include_sequences=True) for bank in banks],
        "audit_rows": audits,
        "parity": parity,
        "decision": decision,
        "not_run": not_run,
    }
    _write_json(output / f"{EXPERIMENT_ID}-results.json", results)
    (output / f"{EXPERIMENT_ID}-runs.csv").write_text(_csv_text(rows), encoding="utf-8")
    (output / f"{EXPERIMENT_ID}-report.md").write_text(_report_text(results), encoding="utf-8")
    _write_artifact_manifest(output)
    verify(output, require_results=True)
    return results


def _evaluation_from_row(row: dict[str, object]) -> Evaluation:
    return Evaluation(
        returns=cast(list[float], row["episode_returns"]),
        successes=cast(list[bool], row["episode_successes"]),
        final_states=cast(list[list[float]], row["final_states"]),
        action_traces=cast(list[list[list[float]]], row["action_traces"]),
    )


def _replay_action_trace(start: np.ndarray, trace: list[list[float]]) -> tuple[float, list[float]]:
    state = np.asarray(start, dtype=float).copy()
    total = 0.0
    for action_value in trace:
        action = np.asarray(action_value, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("saved action must be a finite two-vector")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("saved action is outside the frozen planner bounds")
        state, reward = transition_dynamics(state, action)
        total += reward
    return float(total), state.tolist()


def _verify_rows(results: dict[str, Any]) -> list[dict[str, object]]:
    rows = cast(list[dict[str, object]], results.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("result rows are missing")
    executed = cast(list[str], results.get("executed_rungs"))
    if executed[: len(ENDPOINT_RUNGS)] != list(ENDPOINT_RUNGS):
        raise ValueError("endpoint rungs do not match the frozen order")
    conditional = executed[len(ENDPOINT_RUNGS) :]
    if conditional not in ([], list(PREFIX_RUNGS)):
        raise ValueError("conditional prefix rungs are partial or out of order")
    grouped = _rows_by_rung(rows)
    if set(grouped) != set(executed):
        raise ValueError("raw rows do not match the declared executed rungs")
    fingerprints: dict[int, str] = {}
    for rung in executed:
        rung_rows = grouped[rung]
        if set(rung_rows) != set(MODEL_SEEDS):
            raise ValueError(f"rung {rung} does not contain all formal model seeds")
        for seed, row in rung_rows.items():
            evaluation = _evaluation_from_row(row)
            if not (
                len(evaluation.returns)
                == len(evaluation.successes)
                == len(evaluation.final_states)
                == len(evaluation.action_traces)
                == len(EVAL_STARTS)
            ):
                raise ValueError(f"row {rung}/{seed} has the wrong repeated-start count")
            if any(len(trace) != EVAL_STEPS for trace in evaluation.action_traces):
                raise ValueError(f"row {rung}/{seed} has the wrong action-trace length")
            if any(np.asarray(trace).shape != (EVAL_STEPS, 2) for trace in evaluation.action_traces):
                raise ValueError(f"row {rung}/{seed} has malformed actions")
            if not np.all(np.isfinite(np.asarray(evaluation.returns, dtype=float))):
                raise ValueError(f"row {rung}/{seed} has non-finite returns")
            if np.asarray(evaluation.final_states, dtype=float).shape != (len(EVAL_STARTS), 3):
                raise ValueError(f"row {rung}/{seed} has malformed final states")
            if not np.all(np.isfinite(np.asarray(evaluation.final_states, dtype=float))):
                raise ValueError(f"row {rung}/{seed} has non-finite final states")
            for start_index, start in enumerate(EVAL_STARTS):
                replay_return, replay_final = _replay_action_trace(
                    start,
                    evaluation.action_traces[start_index],
                )
                if not np.isclose(
                    replay_return,
                    evaluation.returns[start_index],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(f"row {rung}/{seed} return does not replay from actions")
                if not np.allclose(
                    replay_final,
                    evaluation.final_states[start_index],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(f"row {rung}/{seed} final state does not replay from actions")
            recomputed_success = [_success(final) for final in evaluation.final_states]
            if recomputed_success != evaluation.successes:
                raise ValueError(f"row {rung}/{seed} success labels do not match final states")
            if not np.isclose(_as_float(row["mean_eval_return"]), evaluation.mean_return, atol=1e-12):
                raise ValueError(f"row {rung}/{seed} mean return is inconsistent")
            if not np.isclose(_as_float(row["success_rate"]), evaluation.success_rate, atol=1e-12):
                raise ValueError(f"row {rung}/{seed} success aggregate is inconsistent")
            fingerprint = str(row["model_sha256"])
            if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
                raise ValueError(f"row {rung}/{seed} has a malformed model fingerprint")
            if seed in fingerprints and fingerprints[seed] != fingerprint:
                raise ValueError(f"seed {seed} does not reuse one trained model across rungs")
            fingerprints[seed] = fingerprint
            expected_count = len(EVAL_STARTS) * EVAL_STEPS * PLANNER_ITERATIONS * PLANNER_HORIZON * PLANNER_CANDIDATES
            if row["planner_candidate_transition_count"] != expected_count:
                raise ValueError(f"row {rung}/{seed} has inconsistent planner compute")
            prefix_steps = PLANNER_HORIZON if rung.startswith("exact_") else 0
            if rung.startswith("prefix_"):
                prefix_steps = int(rung.split("_")[1])
            expected_oracle = expected_count * min(prefix_steps, PLANNER_HORIZON) // PLANNER_HORIZON
            if row["oracle_candidate_transition_count"] != expected_oracle:
                raise ValueError(f"row {rung}/{seed} has inconsistent oracle-call accounting")

    for seed in MODEL_SEEDS:
        _assert_parent_replay(
            seed,
            _evaluation_from_row(grouped["learned_tsinf_penalty"][seed]),
        )
        _assert_evaluation_parity(
            f"saved seed {seed} exact wrapper",
            _evaluation_from_row(grouped["exact_online_oracle_reward"][seed]),
            _evaluation_from_row(grouped["exact_raw"][seed]),
        )
    if _rung_aggregates(rows)["exact_raw"]["success_rate"] < EXACT_FLOOR:
        raise ValueError("saved exact ceiling misses the frozen success floor")
    return rows


def _verify_audits(
    results: dict[str, Any],
    executed: list[str],
) -> list[dict[str, object]]:
    banks = _fixed_banks()
    saved_banks = cast(list[dict[str, object]], results.get("fixed_audit_banks"))
    expected_banks = [bank.as_dict(include_sequences=True) for bank in banks]
    if saved_banks != expected_banks:
        raise ValueError("saved fixed audit banks do not match the frozen construction")
    audits = cast(list[dict[str, object]], results.get("audit_rows"))
    expected_keys = {
        (rung, seed, start_index)
        for rung in executed
        for seed in MODEL_SEEDS
        for start_index in range(len(EVAL_STARTS))
    }
    observed_keys: set[tuple[str, int, int]] = set()
    for audit in audits:
        key = (
            str(audit["rung"]),
            int(cast(int, audit["seed"])),
            int(cast(int, audit["start_index"])),
        )
        if key in observed_keys:
            raise ValueError(f"duplicate fixed-bank audit row: {key}")
        observed_keys.add(key)
        bank = banks[key[2]]
        if audit["bank_sha256"] != bank.sha256:
            raise ValueError(f"fixed-bank hash mismatch in audit {key}")
        scores = np.asarray(audit["candidate_scores"], dtype=float)
        diagnostics = rank_audit(bank, scores).as_dict()
        if audit["diagnostics"] != diagnostics:
            raise ValueError(f"fixed-bank diagnostics do not recompute for {key}")
    if observed_keys != expected_keys:
        raise ValueError("fixed-bank audit rows do not cover every executed block")
    return audits


def _verify_parity(
    results: dict[str, Any],
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    parity = cast(list[dict[str, object]], results.get("parity"))
    if len(parity) != len(MODEL_SEEDS) or {int(cast(int, item["seed"])) for item in parity} != set(MODEL_SEEDS):
        raise ValueError("parity records do not contain all formal model seeds")
    fingerprints = {
        int(cast(int, row["seed"])): str(row["model_sha256"]) for row in rows if row["rung"] == "learned_tsinf_penalty"
    }
    for item in parity:
        seed = int(cast(int, item["seed"]))
        if item["model_sha256"] != fingerprints[seed]:
            raise ValueError(f"parity fingerprint mismatch for seed {seed}")
        replay = cast(dict[str, object], item["parent_replay"])
        mean = cast(dict[str, object], item["mean_k0_evaluation"])
        exact = cast(dict[str, object], item["exact_evaluation"])
        scores = cast(dict[str, object], item["fixed_bank_scores"])
        scalar_values = [
            *[replay[field] for field in ("max_return_abs", "max_final_state_abs")],
            *[
                record[field]
                for record in (mean, exact)
                for field in ("max_return_abs", "max_final_state_abs", "max_action_abs")
            ],
            *scores.values(),
        ]
        if not all(np.isfinite(_as_float(value)) for value in scalar_values):
            raise ValueError(f"saved parity contains a non-finite scalar for seed {seed}")
        if not bool(replay["successes_equal"]) or any(
            _as_float(replay[field]) > REPLAY_ATOL for field in ("max_return_abs", "max_final_state_abs")
        ):
            raise ValueError(f"saved parent replay gate failed for seed {seed}")
        for name, record in (("mean", mean), ("exact", exact)):
            if not bool(record["successes_equal"]) or any(
                _as_float(record[field]) > PARITY_ATOL
                for field in ("max_return_abs", "max_final_state_abs", "max_action_abs")
            ):
                raise ValueError(f"saved {name} parity gate failed for seed {seed}")
        if any(_as_float(value) > PARITY_ATOL for value in scores.values()):
            raise ValueError(f"saved score parity gate failed for seed {seed}")
    return parity


def _verify_model_evidence(
    output: Path,
    rows: list[dict[str, object]],
    audits: list[dict[str, object]],
    parity: list[dict[str, object]],
    executed: list[str],
) -> None:
    """Deterministically retrain and recompute every model-owned formal outcome."""

    dataset = load_dataset(output / INPUT_COPY)
    grouped = _rows_by_rung(rows)
    saved_audits = {
        (
            str(item["rung"]),
            int(cast(int, item["seed"])),
            int(cast(int, item["start_index"])),
        ): item
        for item in audits
    }
    saved_parity = {int(cast(int, item["seed"])): item for item in parity}
    banks = _fixed_banks()
    for seed in MODEL_SEEDS:
        model = train_balanced_model(dataset, seed)
        fingerprint = _model_fingerprint(model)
        if fingerprint != grouped["learned_tsinf_penalty"][seed]["model_sha256"]:
            raise ValueError(f"seed {seed} model fingerprint does not reproduce")
        for rung in executed:
            agent, _ = _rung_agent(model, seed, rung)
            recomputed_evaluation = _evaluate(agent)
            _assert_evaluation_parity(
                f"semantic rerun {rung}/{seed}",
                recomputed_evaluation,
                _evaluation_from_row(grouped[rung][seed]),
            )
            for recomputed_audit in _audit_rung(model, seed, rung, banks):
                key = (
                    rung,
                    seed,
                    int(cast(int, recomputed_audit["start_index"])),
                )
                if saved_audits[key] != recomputed_audit:
                    raise ValueError(f"model-owned fixed-bank scores do not reproduce for {key}")
        prefix_zero_agent, _ = _prefix_agent(model, seed, 0)
        _assert_evaluation_parity(
            f"semantic mean k=0/{seed}",
            _evaluate(prefix_zero_agent),
            _evaluation_from_row(grouped["learned_mean_no_penalty"][seed]),
        )
        if saved_parity[seed]["fixed_bank_scores"] != _score_parity(model, seed, banks):
            raise ValueError(f"fixed-bank parity does not reproduce for seed {seed}")


def _verify_outcomes(output: Path, protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    result_path = output / f"{EXPERIMENT_ID}-results.json"
    results = _read_json(result_path)
    if results.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema does not match")
    if results.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("result experiment id does not match")
    if results.get("status") != "completed_localization":
        raise ValueError("result status does not match the frozen completed state")
    protocol_sha = sha256(_canonical_json_bytes(protocol)).hexdigest()
    if results.get("protocol_sha256") != protocol_sha or results.get("protocol") != protocol:
        raise ValueError("result is not bound to the current frozen protocol")
    if (
        results.get("input_manifest_sha256") != _file_hash(output / "input-manifest.json")
        or results.get("input_manifest") != manifest
    ):
        raise ValueError("result is not bound to the verified input manifest")
    repository = cast(dict[str, Any], results.get("repository"))
    if repository.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("result source hashes do not match the frozen protocol")
    if not isinstance(repository.get("dirty"), bool):
        raise ValueError("result repository dirty state is missing")

    rows = _verify_rows(results)
    executed = cast(list[str], results["executed_rungs"])
    audits = _verify_audits(results, executed)
    parity = _verify_parity(results, rows)
    _verify_model_evidence(output, rows, audits, parity, executed)
    expected_decision = _decision(rows, audits, parity)
    if results.get("decision") != expected_decision:
        raise ValueError("saved decision does not recompute from raw rows and audits")
    transition_material = bool(cast(dict[str, Any], expected_decision["contrasts"])["transition_stack"]["material"])
    expected_executed = list(ENDPOINT_RUNGS) + (list(PREFIX_RUNGS) if transition_material else [])
    if executed != expected_executed:
        raise ValueError("conditional prefix execution does not follow the frozen stop rule")
    expected_not_run = [
        *([] if transition_material else list(PREFIX_RUNGS)),
        "privileged-sequence candidate injection",
        "enlarged learned-landscape search",
        "MuJoCo P3 replication",
        "production or task activation",
    ]
    if results.get("not_run") != expected_not_run:
        raise ValueError("not-run list does not follow the sequential decision")

    if (output / f"{EXPERIMENT_ID}-runs.csv").read_text(encoding="utf-8") != _csv_text(rows):
        raise ValueError("CSV does not match the canonical machine-result rows")
    if (output / f"{EXPERIMENT_ID}-report.md").read_text(encoding="utf-8") != _report_text(results):
        raise ValueError("report does not match the canonical machine result")
    artifact_manifest = _read_json(output / "artifact-manifest.json")
    expected_artifacts = {str(relative): _file_hash(output / relative) for relative in ARTIFACT_PATHS}
    if artifact_manifest != {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": expected_artifacts,
    }:
        raise ValueError("artifact manifest does not match the complete result package")


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
) -> dict[str, Any]:
    """Verify frozen inputs and, when present, every derived outcome artifact."""

    _parent_snapshot()
    protocol = _read_json(output / "protocol.json")
    current_protocol = protocol_record()
    if protocol != current_protocol:
        raise ValueError("saved protocol does not match current sources or research documents")
    manifest = _read_json(output / "input-manifest.json")
    expected_manifest = _expected_input_manifest(output)
    if manifest != expected_manifest:
        raise ValueError("saved input manifest or copied dataset has drifted")
    result_path = output / f"{EXPERIMENT_ID}-results.json"
    outcome_paths = [output / relative for relative in OUTCOME_PATHS]
    if result_path.exists():
        if not all(path.exists() for path in outcome_paths):
            raise ValueError("result package is partial")
        _verify_outcomes(output, protocol, manifest)
        outcome_status = "verified_results"
    elif any(path.exists() for path in outcome_paths):
        raise ValueError("partial or stale outcome artifacts exist without a result record")
    else:
        outcome_status = "prepared_only"
    if require_results and outcome_status != "verified_results":
        raise ValueError("a complete verified OL-001 result package is required")
    return {
        "status": "verified",
        "outcomes": outcome_status,
        "protocol_sha256": sha256(_canonical_json_bytes(protocol)).hexdigest(),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
        "parent_outcomes": "verified_results",
    }


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Return the verified machine result used by the deterministic report."""

    verify(output, require_results=True)
    return _read_json(output / f"{EXPERIMENT_ID}-results.json")
