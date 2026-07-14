"""Preparation, execution, verification, and analysis for BC-001."""
from __future__ import annotations

import csv
import io
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState, Prediction, Transition
from prospect.world_model import FlatWorldModel

from .fixture import (
    ACTION_CORNERS,
    BRIDGE_SOURCE_REGION,
    DOOR_LANE,
    EVAL_STARTS,
    GOAL_Y,
    REGION_CENTERS,
    SCHEMA_VERSION,
    BridgeControlEnv,
    BridgeDataset,
    ExactBridgeModel,
    FactorCell,
    constant_nuisance_control,
    dataset_diagnostics,
    factor_cells,
    generate_dataset,
    load_dataset,
    permuted_action_control,
    region_id,
    save_dataset,
    semantic_hash,
    transition_dynamics,
)

EXPERIMENT_ID = "BC-001"
MODEL_SEEDS = tuple(range(8))
DEVELOPMENT_SEED = 97
TRAIN_STEPS = 1_800
BATCH_SIZE = 64
EVAL_STEPS = 14
PLANNER_HORIZON = 12
PLANNER_CANDIDATES = 64
PLANNER_ELITES = 8
PLANNER_ITERATIONS = 3
UNCERTAINTY_PENALTY = 0.03
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DOC = Path("docs/research/2026-07-13-bridge-control-protocol.md")
PARENT_PROMPT = Path("docs/research/2026-07-13-transformational-research-prompt.md")
PARENT_PORTFOLIO = Path("docs/research/2026-07-13-predictive-reliability-portfolio.md")
DEFAULT_OUTPUT = Path("bench/bridge_control/results/BC-001")
SOURCE_FILES = (
    Path("bench/bridge_control/__init__.py"),
    Path("bench/bridge_control/__main__.py"),
    Path("bench/bridge_control/experiment.py"),
    Path("bench/bridge_control/fixture.py"),
    Path("bench/bridge_control/report.py"),
    Path("src/prospect/agent.py"),
    Path("src/prospect/planning.py"),
    Path("src/prospect/types.py"),
    Path("src/prospect/world_model.py"),
    Path("tests/test_bridge_control.py"),
    PROTOCOL_DOC,
    PARENT_PROMPT,
    PARENT_PORTFOLIO,
)
OUTCOME_PATHS = (
    Path(f"{EXPERIMENT_ID}-results.json"),
    Path(f"{EXPERIMENT_ID}-runs.csv"),
    Path(f"{EXPERIMENT_ID}-report.md"),
    Path("artifact-manifest.json"),
    Path("plots/control-returns.svg"),
    Path("plots/balanced-final-states.svg"),
    Path("plots/topology.svg"),
)
RENDERED_PATHS = OUTCOME_PATHS[2:3] + OUTCOME_PATHS[4:]
ARTIFACT_PATHS = (
    Path("protocol.json"),
    Path("dataset-manifest.json"),
) + OUTCOME_PATHS[:3] + OUTCOME_PATHS[4:]


@dataclass(frozen=True)
class Evaluation:
    returns: list[float]
    successes: list[bool]
    final_states: list[list[float]]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes))


class ExactTransitionLearnedReward:
    """Diagnostic model: exact raw dynamics, zero epistemic, learned reward.

    This rung tests whether the learned reward is the sole blocker. It deliberately
    changes transition, representation, and uncertainty handling together, so it
    cannot uniquely attribute any rescue among those components.
    """

    def __init__(self, learned: FlatWorldModel) -> None:
        self._learned = learned

    def predict(self, state: LatentState, action: Action) -> Prediction:
        raw = np.asarray(state.z, dtype=float)
        act = np.asarray(action.data, dtype=float)
        next_state, _ = transition_dynamics(raw, act)
        learned = self._learned.predict(self._learned.encode(raw), action)
        return Prediction(
            mean=next_state,
            var=np.full(3, 1e-6),
            epistemic=0.0,
            aleatoric=1e-6,
            reward=learned.reward,
        )

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        current = LatentState(z=np.asarray(state.z, dtype=float).copy())
        predictions: list[Prediction] = []
        for action in actions:
            prediction = self.predict(current, action)
            predictions.append(prediction)
            current = LatentState(z=np.asarray(prediction.mean, dtype=float))
        return predictions

    def predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        exact = [
            transition_dynamics(state, action)
            for state, action in zip(latents, actions, strict=True)
        ]
        learned_latents = np.stack([self._learned.encode(state).z for state in latents])
        _, _, _, _, rewards = self._learned.predict_batch(learned_latents, actions)
        means = np.stack([item[0] for item in exact])
        n = len(means)
        return (
            means,
            np.full_like(means, 1e-6),
            np.zeros(n),
            np.full(n, 1e-6),
            rewards,
        )


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
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def protocol_record() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_control_validation",
        "parent_prompt": str(PARENT_PROMPT),
        "parent_portfolio": str(PARENT_PORTFOLIO),
        "protocol_document": str(PROTOCOL_DOC),
        "protocol_document_sha256": _file_hash(REPO_ROOT / PROTOCOL_DOC),
        "source_sha256": _source_hashes(),
        "development_seed_excluded": DEVELOPMENT_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "learner": {
            "class": "prospect.world_model.FlatWorldModel",
            "obs_dim": 3,
            "action_dim": 2,
            "latent_dim": 6,
            "hidden": 48,
            "ensemble": 5,
            "steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
        },
        "planner": {
            "class": "prospect.planning.FlatPlanner",
            "horizon": PLANNER_HORIZON,
            "candidates": PLANNER_CANDIDATES,
            "elites": PLANNER_ELITES,
            "iterations": PLANNER_ITERATIONS,
            "uncertainty_penalty": UNCERTAINTY_PENALTY,
            "action_low": -1.0,
            "action_high": 1.0,
        },
        "evaluation": {
            "steps": EVAL_STEPS,
            "fixed_starts": EVAL_STARTS.tolist(),
            "success": {"x_min": 0.75, "goal_y": GOAL_Y, "y_tolerance": 0.20},
        },
        "controls": {
            "exact_success_rate_min": 0.95,
            "balanced_success_rate_min": 0.80,
            "balanced_must_beat_random": True,
            "stop_before_factorial_if_control_fails": True,
        },
        "factorial": {
            "bridge_edges": [0, 16],
            "rank_min_singular": [0.05, 0.5],
            "unique_controllable_support": [1, 8],
        },
    }


def _all_datasets() -> list[BridgeDataset]:
    primary = [generate_dataset(cell) for cell in factor_cells()]
    by_name = {dataset.name: dataset for dataset in primary}
    return primary + [
        permuted_action_control(by_name["b1_r1_d8"]),
        constant_nuisance_control(by_name["b1_r1_d1"]),
    ]


def _datasets_by_name() -> dict[str, BridgeDataset]:
    return {dataset.name: dataset for dataset in _all_datasets()}


def _clear_generated_artifacts(output: Path) -> None:
    for relative in OUTCOME_PATHS[:4]:
        (output / relative).unlink(missing_ok=True)
    shutil.rmtree(output / "plots", ignore_errors=True)
    shutil.rmtree(output / "datasets", ignore_errors=True)


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Write fixed data and manipulation checks without model-control outcomes."""

    output.mkdir(parents=True, exist_ok=True)
    _clear_generated_artifacts(output)
    protocol = protocol_record()
    _write_json(output / "protocol.json", protocol)
    entries: dict[str, object] = {}
    for dataset in _all_datasets():
        relative = Path("datasets") / f"{dataset.name}.npz"
        save_dataset(dataset, output / relative)
        entries[dataset.name] = {
            "path": str(relative),
            "semantic_hash": semantic_hash(dataset),
            "cell": None if dataset.cell is None else dataset.cell.as_dict(),
            "control": dataset.control,
            "manipulation_checks": dataset_diagnostics(dataset),
        }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared_no_model_outcomes",
        "protocol_sha256": sha256(_canonical_json_bytes(protocol)).hexdigest(),
        "datasets": entries,
    }
    _write_json(output / "dataset-manifest.json", manifest)
    verify(output)
    return manifest


def _primary_from_manifest(output: Path, manifest: dict[str, Any]) -> dict[str, BridgeDataset]:
    datasets_raw = cast(dict[str, dict[str, Any]], manifest["datasets"])
    regenerated = _datasets_by_name()
    if set(datasets_raw) != set(regenerated):
        raise ValueError("manifest must contain exactly eight primary and two control datasets")
    loaded: dict[str, BridgeDataset] = {}
    for name, entry in datasets_raw.items():
        expected_path = str(Path("datasets") / f"{name}.npz")
        if entry.get("path") != expected_path:
            raise ValueError(f"unexpected dataset path for {name}")
        dataset = load_dataset(output / str(entry["path"]))
        if dataset.name != name:
            raise ValueError(f"embedded dataset name does not match manifest key {name}")
        expected = str(entry["semantic_hash"])
        actual = semantic_hash(dataset)
        if actual != expected:
            raise ValueError(f"semantic hash mismatch for {name}: {actual} != {expected}")
        if actual != semantic_hash(regenerated[name]):
            raise ValueError(f"saved dataset {name} does not match current deterministic generation")
        if entry.get("cell") != (None if dataset.cell is None else dataset.cell.as_dict()):
            raise ValueError(f"manifest cell metadata does not match {name}")
        if entry.get("control") != dataset.control:
            raise ValueError(f"manifest control metadata does not match {name}")
        diagnostics = dataset_diagnostics(dataset)
        if entry.get("manipulation_checks") != diagnostics:
            raise ValueError(f"manifest manipulation checks do not match {name}")
        loaded[name] = dataset
    return loaded


def _assert_manipulations(datasets: dict[str, BridgeDataset]) -> None:
    primary = {name: data for name, data in datasets.items() if data.cell is not None}
    if set(primary) != {cell.name for cell in factor_cells()}:
        raise ValueError("primary factorial does not contain exactly eight cells")
    diagnostics = {name: dataset_diagnostics(data) for name, data in primary.items()}
    reference = diagnostics["b0_r0_d1"]
    matched_fields = (
        "rows",
        "node_coverage",
        "state_region_counts",
        "state_region_lane_counts",
        "nuisance_levels",
        "nuisance_histogram",
        "global_action_coordinate_histograms",
    )
    for name, current in diagnostics.items():
        for field in matched_fields:
            if current[field] != reference[field]:
                raise ValueError(f"{name} does not match {field}")
        cell = primary[name].cell
        assert cell is not None
        expected_edges = 16 if cell.bridge else 0
        if current["bridge_edge_count"] != expected_edges:
            raise ValueError(f"{name} bridge edge count is not {expected_edges}")
        singular = cast(float, current["local_action_min_singular"])
        if cell.full_rank and singular < 0.5:
            raise ValueError(f"{name} full-rank singular value below 0.5")
        if not cell.full_rank and singular > 0.05:
            raise ValueError(f"{name} deficient singular value above 0.05")
        unique_min = cast(int, current["controllable_unique_per_cell_min"])
        unique_max = cast(int, current["controllable_unique_per_cell_max"])
        if unique_min != cell.density or unique_max != cell.density:
            raise ValueError(f"{name} density manipulation is not exactly {cell.density}")

    # For a fixed bridge setting, the bridge-source arrays must not depend on R/D.
    for bridge in (False, True):
        members = [data for data in primary.values() if data.cell and data.cell.bridge == bridge]
        base = members[0]
        base_rows = (base.region_ids == BRIDGE_SOURCE_REGION) & (base.lane_ids == DOOR_LANE)
        for member in members[1:]:
            rows = (member.region_ids == BRIDGE_SOURCE_REGION) & (member.lane_ids == DOOR_LANE)
            for left, right in (
                (base.states[base_rows], member.states[rows]),
                (base.actions[base_rows], member.actions[rows]),
                (base.next_states[base_rows], member.next_states[rows]),
                (base.rewards[base_rows], member.rewards[rows]),
            ):
                if not np.array_equal(left, right):
                    raise ValueError("bridge rows vary with rank or density")


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
) -> dict[str, Any]:
    protocol = _read_json(output / "protocol.json")
    current_protocol = protocol_record()
    if protocol != current_protocol:
        raise ValueError("saved protocol does not match current protocol, source, or research documents")
    manifest = _read_json(output / "dataset-manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset manifest schema does not match")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("dataset manifest experiment id does not match")
    expected_protocol = sha256(_canonical_json_bytes(protocol)).hexdigest()
    if manifest.get("protocol_sha256") != expected_protocol:
        raise ValueError("protocol hash does not match dataset manifest")
    datasets = _primary_from_manifest(output, manifest)
    _assert_manipulations(datasets)
    result_path = output / f"{EXPERIMENT_ID}-results.json"
    outcome_paths = [output / relative for relative in OUTCOME_PATHS]
    if result_path.exists():
        _verify_outcomes(output, protocol, manifest)
        outcome_status = "verified_results"
    elif any(path.exists() for path in outcome_paths):
        raise ValueError("partial or stale outcome artifacts exist without a result record")
    else:
        outcome_status = "prepared_only"
    if require_results and outcome_status != "verified_results":
        raise ValueError("a complete verified result package is required")
    return {
        "protocol_sha256": expected_protocol,
        "dataset_count": len(datasets),
        "primary_count": sum(data.cell is not None for data in datasets.values()),
        "outcomes": outcome_status,
        "status": "verified",
    }


def _planner(model: object, seed: int) -> FlatPlanner:
    return FlatPlanner(
        cast(Any, model),
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=PLANNER_HORIZON,
        candidates=PLANNER_CANDIDATES,
        elites=PLANNER_ELITES,
        iterations=PLANNER_ITERATIONS,
        uncertainty_penalty=UNCERTAINTY_PENALTY,
        seed=seed,
    )


def _evaluate_agent(agent: Agent) -> Evaluation:
    returns: list[float] = []
    successes: list[bool] = []
    final_states: list[list[float]] = []
    for start in EVAL_STARTS:
        env = BridgeControlEnv()
        obs = env.set_state(start)
        agent.reset()
        total = 0.0
        for _ in range(EVAL_STEPS):
            action = agent.act(obs)
            obs, reward, _ = env.step(action)
            total += reward
        final = np.asarray(obs.data, dtype=float)
        returns.append(float(total))
        successes.append(bool(final[0] >= 0.75 and abs(final[1] - GOAL_Y) <= 0.20))
        final_states.append(final.tolist())
    return Evaluation(returns, successes, final_states)


def _evaluate_exact(seed: int) -> Evaluation:
    agent = Agent(
        encode=lambda obs: LatentState(z=np.asarray(obs.data, dtype=float)),
        planner=_planner(ExactBridgeModel(), seed),
    )
    return _evaluate_agent(agent)


def _evaluate_random(seed: int) -> Evaluation:
    returns: list[float] = []
    successes: list[bool] = []
    final_states: list[list[float]] = []
    for start_index, start in enumerate(EVAL_STARTS):
        rng = np.random.default_rng(70_000 + 101 * seed + start_index)
        env = BridgeControlEnv()
        obs = env.set_state(start)
        total = 0.0
        for _ in range(EVAL_STEPS):
            obs, reward, _ = env.step(Action(data=rng.uniform(-1.0, 1.0, size=2)))
            total += reward
        final = np.asarray(obs.data, dtype=float)
        returns.append(float(total))
        successes.append(bool(final[0] >= 0.75 and abs(final[1] - GOAL_Y) <= 0.20))
        final_states.append(final.tolist())
    return Evaluation(returns, successes, final_states)


def _train_model(dataset: BridgeDataset, seed: int) -> FlatWorldModel:
    transitions = dataset.transitions()
    model = FlatWorldModel(
        obs_dim=3,
        action_dim=2,
        latent_dim=6,
        hidden=48,
        ensemble=5,
        seed=seed,
    )
    schedule = np.random.default_rng(seed + 1_000)
    for _ in range(TRAIN_STEPS):
        indices = schedule.integers(0, len(transitions), size=BATCH_SIZE)
        model.update([transitions[int(index)] for index in indices])
    return model


def _evaluate_learned(model: FlatWorldModel, seed: int) -> Evaluation:
    agent = Agent(
        encode=lambda obs: model.encode(obs.data),
        planner=_planner(model, seed),
    )
    return _evaluate_agent(agent)


def _evaluate_reward_hybrid(model: FlatWorldModel, seed: int) -> Evaluation:
    hybrid = ExactTransitionLearnedReward(model)
    agent = Agent(
        encode=lambda obs: LatentState(z=np.asarray(obs.data, dtype=float)),
        planner=_planner(hybrid, seed),
    )
    return _evaluate_agent(agent)


def _heldout_transitions() -> list[Transition]:
    actions = np.array([[-0.8, -0.2], [-0.2, 0.8], [0.4, -0.8], [0.8, 0.4]])
    transitions: list[Transition] = []
    for center in REGION_CENTERS:
        for y_index, y in enumerate((-0.26, -0.06, 0.14, 0.34)):
            state = np.array([center + 0.018, y, (-0.6, -0.2, 0.2, 0.6)[y_index]])
            for action in actions:
                next_state, reward = transition_dynamics(state, action)
                transitions.append(
                    Transition(
                        state=LatentState(z=state.copy()),
                        action=Action(data=action.copy()),
                        next_state=LatentState(z=next_state),
                        reward=reward,
                    )
                )
    return transitions


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _sequence_bank() -> np.ndarray:
    rng = np.random.default_rng(314_159)
    random_bank = rng.uniform(-1.0, 1.0, size=(56, 5, 2))
    structured = np.array(
        [np.tile(action, (5, 1)) for action in ACTION_CORNERS]
        + [
            np.array([[1.0, -1.0]] * 2 + [[1.0, 1.0]] * 3),
            np.array([[1.0, 1.0]] * 2 + [[1.0, -1.0]] * 3),
            np.array([[0.5, -1.0]] * 3 + [[1.0, 0.5]] * 2),
            np.array([[0.5, 1.0]] * 3 + [[1.0, -0.5]] * 2),
        ]
    )
    return np.concatenate([random_bank, structured], axis=0)


def _exact_sequence_score(start: np.ndarray, sequence: np.ndarray) -> float:
    state = start.copy()
    total, discount = 0.0, 1.0
    for action in sequence:
        state, reward = transition_dynamics(state, action)
        total += discount * reward
        discount *= 0.99
    return float(total)


def _candidate_ranking(model: FlatWorldModel) -> dict[str, float]:
    bank = _sequence_bank()
    starts = (np.array([-0.11, 0.0, 0.0]), np.array([0.40, 0.0, 0.0]))
    correlations: list[float] = []
    regrets: list[float] = []
    for start in starts:
        exact = np.array([_exact_sequence_score(start, sequence) for sequence in bank])
        learned_scores: list[float] = []
        for sequence in bank:
            predictions = model.imagine(
                model.encode(start),
                [Action(data=action) for action in sequence],
            )
            discount = 1.0
            score = 0.0
            for prediction in predictions:
                score += discount * (
                    prediction.reward - UNCERTAINTY_PENALTY * prediction.epistemic
                )
                discount *= 0.99
            learned_scores.append(score)
        learned = np.array(learned_scores)
        correlations.append(_spearman(exact, learned))
        regrets.append(float(np.max(exact) - exact[int(np.argmax(learned))]))
    return {
        "candidate_rank_spearman": float(np.mean(correlations)),
        "candidate_action_regret": float(np.mean(regrets)),
    }


def _model_diagnostics(model: FlatWorldModel, dataset: BridgeDataset) -> dict[str, float]:
    heldout = _heldout_transitions()
    target_latent_errors: list[float] = []
    persistence_errors: list[float] = []
    reward_errors: list[float] = []
    epistemic: list[float] = []

    prototype_latents = np.stack([model.encode_target(row).z for row in dataset.next_states])
    decoded_predictions: list[np.ndarray] = []
    true_next: list[np.ndarray] = []
    region_correct: list[bool] = []
    lane_correct: list[bool] = []
    for transition in heldout:
        state_raw = np.asarray(transition.state.z, dtype=float)
        next_raw = np.asarray(transition.next_state.z, dtype=float)
        prediction = model.predict(model.encode(state_raw), transition.action)
        target = np.asarray(model.encode_target(next_raw).z, dtype=float)
        current = np.asarray(model.encode_target(state_raw).z, dtype=float)
        predicted = np.asarray(prediction.mean, dtype=float)
        target_latent_errors.append(float(np.mean((predicted - target) ** 2)))
        persistence_errors.append(float(np.mean((current - target) ** 2)))
        reward_errors.append((prediction.reward - transition.reward) ** 2)
        epistemic.append(prediction.epistemic)
        nearest = int(np.argmin(np.sum((prototype_latents - predicted) ** 2, axis=1)))
        decoded = dataset.next_states[nearest]
        decoded_predictions.append(decoded)
        true_next.append(next_raw)
        region_correct.append(region_id(float(decoded[0])) == region_id(float(next_raw[0])))
        lane_correct.append((abs(decoded[1]) >= 0.55) == (abs(next_raw[1]) >= 0.55))
    decoded_array = np.stack(decoded_predictions)
    true_array = np.stack(true_next)
    normalized = float(np.mean(target_latent_errors)) / max(
        float(np.mean(persistence_errors)), 1e-12
    )
    diagnostics = {
        "reward_rmse": float(np.sqrt(np.mean(reward_errors))),
        "target_latent_mse": float(np.mean(target_latent_errors)),
        "target_latent_mse_over_persistence": normalized,
        "prototype_raw_next_mse": float(np.mean((decoded_array[:, :2] - true_array[:, :2]) ** 2)),
        "next_region_accuracy": float(np.mean(region_correct)),
        "next_lane_accuracy": float(np.mean(lane_correct)),
        "mean_epistemic": float(np.mean(epistemic)),
    }
    return diagnostics | _candidate_ranking(model)


def _row(
    arm: str,
    seed: int,
    evaluation: Evaluation,
    cell: FactorCell | None = None,
    diagnostics: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "arm": arm,
        "seed": seed,
        "cell": None if cell is None else cell.as_dict(),
        "mean_eval_return": evaluation.mean_return,
        "success_rate": evaluation.success_rate,
        "episode_returns": evaluation.returns,
        "episode_successes": evaluation.successes,
        "final_states": evaluation.final_states,
        "diagnostics": diagnostics or {},
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "mean_return": float(
            np.mean([cast(float, row["mean_eval_return"]) for row in rows])
        ),
        "mean_success_rate": float(
            np.mean([cast(float, row["success_rate"]) for row in rows])
        ),
    }


def _factorial_effects(rows: list[dict[str, object]]) -> dict[str, object]:
    effects: dict[str, list[float]] = {"bridge": [], "rank": [], "density": [], "bridge_x_rank": []}
    for seed in MODEL_SEEDS:
        seed_rows = [row for row in rows if cast(int, row["seed"]) == seed]
        values: dict[tuple[int, int, int], float] = {}
        for row in seed_rows:
            cell = cast(dict[str, Any], row["cell"])
            key = (int(bool(cell["bridge"])), int(bool(cell["full_rank"])), int(cell["density"] == 8))
            values[key] = cast(float, row["mean_eval_return"])
        if len(values) != 8:
            continue
        for index, name in enumerate(("bridge", "rank", "density")):
            plus = [value for key, value in values.items() if key[index] == 1]
            minus = [value for key, value in values.items() if key[index] == 0]
            effects[name].append(float(np.mean(plus) - np.mean(minus)))
        rank_when_bridge = np.mean([v for k, v in values.items() if k[0] == 1 and k[1] == 1]) - np.mean(
            [v for k, v in values.items() if k[0] == 1 and k[1] == 0]
        )
        rank_without_bridge = np.mean([v for k, v in values.items() if k[0] == 0 and k[1] == 1]) - np.mean(
            [v for k, v in values.items() if k[0] == 0 and k[1] == 0]
        )
        effects["bridge_x_rank"].append(float(rank_when_bridge - rank_without_bridge))
    return {
        name: {"per_seed": values, "mean": float(np.mean(values)) if values else None}
        for name, values in effects.items()
    }


def _csv_text(rows: list[dict[str, object]]) -> str:
    handle = io.StringIO(newline="")
    fields = ["arm", "seed", "mean_eval_return", "success_rate", "bridge", "full_rank", "density"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        cell = cast(dict[str, object] | None, row["cell"])
        writer.writerow(
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "mean_eval_return": row["mean_eval_return"],
                "success_rate": row["success_rate"],
                "bridge": "" if cell is None else cell["bridge"],
                "full_rank": "" if cell is None else cell["full_rank"],
                "density": "" if cell is None else cell["density"],
            }
        )
    return handle.getvalue()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_csv_text(rows))


def _write_artifact_manifest(output: Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": {
            str(relative): _file_hash(output / relative) for relative in ARTIFACT_PATHS
        },
    }
    _write_json(output / "artifact-manifest.json", payload)


def _verify_outcomes(
    output: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    from .report import write_report_artifacts

    results_path = output / f"{EXPERIMENT_ID}-results.json"
    results = _read_json(results_path)
    if results.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema does not match")
    if results.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("result experiment id does not match")
    protocol_sha = sha256(_canonical_json_bytes(protocol)).hexdigest()
    if results.get("protocol_sha256") != protocol_sha or results.get("protocol") != protocol:
        raise ValueError("result is not bound to the current frozen protocol")
    if results.get("manifest_sha256") != _file_hash(output / "dataset-manifest.json"):
        raise ValueError("result dataset-manifest hash does not match")

    expected_checks = {
        name: cast(dict[str, Any], entry)["manipulation_checks"]
        for name, entry in cast(dict[str, Any], manifest["datasets"]).items()
        if name.startswith("b")
    }
    if results.get("manipulation_checks") != expected_checks:
        raise ValueError("result manipulation checks do not match the verified manifest")

    repository = cast(dict[str, Any], results.get("repository"))
    if not isinstance(repository.get("dirty"), bool):
        raise ValueError("result repository dirty state is missing")
    if repository.get("source_sha256") != protocol.get("source_sha256"):
        raise ValueError("result source hashes do not match the frozen protocol")

    rows = cast(list[dict[str, object]], results.get("rows"))
    evaluation = cast(dict[str, Any], protocol["evaluation"])
    success_spec = cast(dict[str, float], evaluation["success"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        arm = str(row.get("arm"))
        grouped.setdefault(arm, []).append(row)
        returns = cast(list[float], row.get("episode_returns"))
        successes = cast(list[bool], row.get("episode_successes"))
        final_states = cast(list[list[float]], row.get("final_states"))
        if len(returns) != len(EVAL_STARTS) or len(successes) != len(EVAL_STARTS):
            raise ValueError(f"result row {arm} has the wrong evaluation-start count")
        if len(final_states) != len(EVAL_STARTS):
            raise ValueError(f"result row {arm} has the wrong final-state count")
        recomputed_successes = [
            bool(
                final[0] >= success_spec["x_min"]
                and abs(final[1] - success_spec["goal_y"])
                <= success_spec["y_tolerance"]
            )
            for final in final_states
        ]
        if successes != recomputed_successes:
            raise ValueError(f"result row {arm} success labels do not match final states")
        if not np.isclose(
            cast(float, row["mean_eval_return"]), float(np.mean(returns)), atol=1e-12
        ):
            raise ValueError(f"result row {arm} return aggregate is inconsistent")
        if not np.isclose(cast(float, row["success_rate"]), float(np.mean(successes)), atol=1e-12):
            raise ValueError(f"result row {arm} success aggregate is inconsistent")

    decision = cast(dict[str, Any], results.get("control_decision"))
    control_arms = {
        "exact": "exact_dynamics_exact_reward",
        "random": "random_policy",
        "balanced_learned": "b1_r1_d8",
    }
    aggregates: dict[str, dict[str, float]] = {}
    for key, arm in control_arms.items():
        arm_rows = grouped.get(arm, [])
        if len(arm_rows) != len(MODEL_SEEDS):
            raise ValueError(f"result must contain one {arm} row per formal seed")
        if {cast(int, row["seed"]) for row in arm_rows} != set(MODEL_SEEDS):
            raise ValueError(f"result {arm} seeds do not match the frozen blocks")
        aggregates[key] = _aggregate(arm_rows)
        if decision.get(key) != aggregates[key]:
            raise ValueError(f"result {arm} aggregate does not match raw rows")
    hybrid_rows = grouped.get("exact_transition_learned_reward_zero_epistemic", [])
    if len(hybrid_rows) != len(MODEL_SEEDS):
        raise ValueError("result must contain one reward-hybrid row per formal seed")

    controls = cast(dict[str, Any], protocol["controls"])
    controls_pass = bool(
        aggregates["exact"]["mean_success_rate"] >= float(controls["exact_success_rate_min"])
        and aggregates["balanced_learned"]["mean_success_rate"]
        >= float(controls["balanced_success_rate_min"])
        and aggregates["balanced_learned"]["mean_return"] > aggregates["random"]["mean_return"]
    )
    if bool(decision.get("passed")) != controls_pass:
        raise ValueError("result control decision does not follow the frozen thresholds")
    expected_status = "completed_factorial" if controls_pass else "aborted_invalid_fixture"
    if results.get("status") != expected_status:
        raise ValueError("result status does not follow the control decision")
    if bool(decision.get("stop_rule_applied")) != (not controls_pass):
        raise ValueError("result stop-rule flag is inconsistent")

    base_arms = {
        "exact_dynamics_exact_reward",
        "random_policy",
        "exact_transition_learned_reward_zero_epistemic",
    }
    primary_arms = {cell.name for cell in factor_cells()}
    if controls_pass:
        expected_arms = base_arms | primary_arms | {
            "control_action_permuted",
            "control_nuisance_constant",
        }
        factorial_rows = [row for row in rows if str(row["arm"]) in primary_arms]
        if results.get("factorial_effects") != _factorial_effects(factorial_rows):
            raise ValueError("factorial effects do not match raw rows")
        if results.get("not_run") != ["second corridor-length replication"]:
            raise ValueError("completed factorial has an inconsistent replication status")
    else:
        expected_arms = base_arms | {"b1_r1_d8"}
        if results.get("factorial_effects") is not None:
            raise ValueError("stopped experiment cannot report factorial effects")
        expected_not_run = [
            "seven remaining factorial cells",
            "action-label permutation control",
            "nuisance-only control",
            "second corridor-length replication",
        ]
        if results.get("not_run") != expected_not_run:
            raise ValueError("stopped experiment has an inconsistent not-run list")
    if set(grouped) != expected_arms:
        raise ValueError("result arms do not match the sequential control decision")
    cells_by_arm = {cell.name: cell.as_dict() for cell in factor_cells()}
    for arm, arm_rows in grouped.items():
        if len(arm_rows) != len(MODEL_SEEDS):
            raise ValueError(f"result arm {arm} does not contain all formal blocks")
        if {cast(int, row["seed"]) for row in arm_rows} != set(MODEL_SEEDS):
            raise ValueError(f"result arm {arm} has inconsistent formal blocks")
        expected_cell = cells_by_arm.get(arm)
        if any(row.get("cell") != expected_cell for row in arm_rows):
            raise ValueError(f"result arm {arm} has inconsistent factor metadata")

    csv_path = output / f"{EXPERIMENT_ID}-runs.csv"
    if csv_path.read_bytes() != _csv_text(rows).encode("utf-8"):
        raise ValueError("CSV rows do not match the canonical machine result")

    with tempfile.TemporaryDirectory(prefix="bc001-verify-") as temp_dir:
        rendered = Path(temp_dir)
        write_report_artifacts(rendered, results, manifest)
        for relative in RENDERED_PATHS:
            if (output / relative).read_bytes() != (rendered / relative).read_bytes():
                raise ValueError(f"rendered artifact does not match result: {relative}")

    artifact_manifest = _read_json(output / "artifact-manifest.json")
    expected_artifacts = {
        str(relative): _file_hash(output / relative) for relative in ARTIFACT_PATHS
    }
    if artifact_manifest != {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": expected_artifacts,
    }:
        raise ValueError("artifact manifest does not match the complete result package")


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    from .report import write_report_artifacts

    verification = verify(output)
    protocol = _read_json(output / "protocol.json")
    manifest = _read_json(output / "dataset-manifest.json")
    datasets = _primary_from_manifest(output, manifest)
    balanced = datasets["b1_r1_d8"]

    exact_rows = [_row("exact_dynamics_exact_reward", seed, _evaluate_exact(seed)) for seed in MODEL_SEEDS]
    random_rows = [_row("random_policy", seed, _evaluate_random(seed)) for seed in MODEL_SEEDS]
    balanced_rows: list[dict[str, object]] = []
    reward_hybrid_rows: list[dict[str, object]] = []
    for seed in MODEL_SEEDS:
        model = _train_model(balanced, seed)
        balanced_rows.append(
            _row(
                balanced.name,
                seed,
                _evaluate_learned(model, seed),
                balanced.cell,
                _model_diagnostics(model, balanced),
            )
        )
        reward_hybrid_rows.append(
            _row(
                "exact_transition_learned_reward_zero_epistemic",
                seed,
                _evaluate_reward_hybrid(model, seed),
            )
        )

    exact_aggregate = _aggregate(exact_rows)
    random_aggregate = _aggregate(random_rows)
    balanced_aggregate = _aggregate(balanced_rows)
    controls_pass = bool(
        exact_aggregate["mean_success_rate"] >= 0.95
        and balanced_aggregate["mean_success_rate"] >= 0.80
        and balanced_aggregate["mean_return"] > random_aggregate["mean_return"]
    )

    all_rows = exact_rows + random_rows + balanced_rows + reward_hybrid_rows
    factorial_rows: list[dict[str, object]] = list(balanced_rows)
    control_rows: list[dict[str, object]] = []
    if controls_pass:
        for name, dataset in datasets.items():
            if dataset.cell is None or name == balanced.name:
                continue
            for seed in MODEL_SEEDS:
                model = _train_model(dataset, seed)
                row = _row(
                    name,
                    seed,
                    _evaluate_learned(model, seed),
                    dataset.cell,
                    _model_diagnostics(model, dataset),
                )
                factorial_rows.append(row)
                all_rows.append(row)
        for control_name in ("control_action_permuted", "control_nuisance_constant"):
            dataset = datasets[control_name]
            for seed in MODEL_SEEDS:
                model = _train_model(dataset, seed)
                row = _row(
                    control_name,
                    seed,
                    _evaluate_learned(model, seed),
                    diagnostics=_model_diagnostics(model, dataset),
                )
                control_rows.append(row)
                all_rows.append(row)

    status = "completed_factorial" if controls_pass else "aborted_invalid_fixture"
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "interpretation_scope": "non-gated research evidence",
        "protocol_sha256": verification["protocol_sha256"],
        "manifest_sha256": _file_hash(output / "dataset-manifest.json"),
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
        "protocol": protocol,
        "manipulation_checks": {
            name: cast(dict[str, Any], entry)["manipulation_checks"]
            for name, entry in cast(dict[str, Any], manifest["datasets"]).items()
            if name.startswith("b")
        },
        "control_decision": {
            "passed": controls_pass,
            "exact": exact_aggregate,
            "random": random_aggregate,
            "balanced_learned": balanced_aggregate,
            "stop_rule_applied": not controls_pass,
            "reason": (
                "positive controls passed; factorial executed"
                if controls_pass
                else "balanced learned control missed the frozen 80% success floor after one redesign"
            ),
        },
        "rows": all_rows,
        "factorial_effects": _factorial_effects(factorial_rows) if controls_pass else None,
        "not_run": (
            ["second corridor-length replication"]
            if controls_pass
            else [
                "seven remaining factorial cells",
                "action-label permutation control",
                "nuisance-only control",
                "second corridor-length replication",
            ]
        ),
    }
    _write_json(output / f"{EXPERIMENT_ID}-results.json", results)
    _write_csv(output / f"{EXPERIMENT_ID}-runs.csv", all_rows)
    write_report_artifacts(output, results, manifest)
    _write_artifact_manifest(output)
    verify(output, require_results=True)
    return results
