"""BC-001 fixture tests. The research run remains non-gated."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast
from zipfile import BadZipFile

import numpy as np
import pytest

from bench.bridge_control.experiment import (
    DEFAULT_OUTPUT,
    MODEL_SEEDS,
    _evaluate_exact,
    prepare,
    protocol_record,
    verify,
)
from bench.bridge_control.fixture import (
    BRIDGE_SOURCE_REGION,
    DOOR_LANE,
    EVAL_STARTS,
    GOAL_Y,
    BridgeControlEnv,
    ExactBridgeModel,
    FactorCell,
    constant_nuisance_control,
    dataset_diagnostics,
    factor_cells,
    generate_dataset,
    load_dataset,
    permuted_action_control,
    save_dataset,
    semantic_hash,
)
from bench.envs import Environment
from prospect.agent import Agent
from prospect.planning import FlatPlanner
from prospect.types import LatentState


def test_factorial_manipulations_are_orthogonal_and_matched() -> None:
    datasets = {cell.name: generate_dataset(cell) for cell in factor_cells()}
    diagnostics = {name: dataset_diagnostics(dataset) for name, dataset in datasets.items()}
    reference = diagnostics["b0_r0_d1"]
    matched = (
        "rows",
        "node_coverage",
        "state_region_counts",
        "state_region_lane_counts",
        "nuisance_levels",
        "nuisance_histogram",
        "global_action_coordinate_histograms",
    )
    for name, dataset in datasets.items():
        cell = dataset.cell
        assert cell is not None
        current = diagnostics[name]
        assert all(current[field] == reference[field] for field in matched)
        assert current["bridge_edge_count"] == (16 if cell.bridge else 0)
        if cell.full_rank:
            assert cast(float, current["local_action_min_singular"]) >= 0.5
        else:
            assert cast(float, current["local_action_min_singular"]) <= 0.05
        assert current["controllable_unique_per_cell_min"] == cell.density
        assert current["controllable_unique_per_cell_max"] == cell.density

    for bridge in (False, True):
        group = [dataset for dataset in datasets.values() if dataset.cell and dataset.cell.bridge == bridge]
        base = group[0]
        base_rows = (base.region_ids == BRIDGE_SOURCE_REGION) & (base.lane_ids == DOOR_LANE)
        for dataset in group[1:]:
            rows = (dataset.region_ids == BRIDGE_SOURCE_REGION) & (dataset.lane_ids == DOOR_LANE)
            np.testing.assert_array_equal(base.states[base_rows], dataset.states[rows])
            np.testing.assert_array_equal(base.actions[base_rows], dataset.actions[rows])
            np.testing.assert_array_equal(base.next_states[base_rows], dataset.next_states[rows])
            np.testing.assert_array_equal(base.rewards[base_rows], dataset.rewards[rows])


def test_semantic_hash_round_trip_and_mutation(tmp_path: Path) -> None:
    dataset = generate_dataset(FactorCell(True, True, 8))
    regenerated = generate_dataset(FactorCell(True, True, 8))
    assert semantic_hash(dataset) == semantic_hash(regenerated)
    path = tmp_path / "dataset.npz"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert semantic_hash(dataset) == semantic_hash(loaded)

    changed_states = dataset.states.copy()
    changed_states[0, 0] += 1e-12
    mutated = replace(dataset, states=changed_states)
    assert semantic_hash(mutated) != semantic_hash(dataset)


def test_named_controls_preserve_their_required_marginals() -> None:
    balanced = generate_dataset(FactorCell(True, True, 8))
    permuted = permuted_action_control(balanced)
    np.testing.assert_array_equal(permuted.states, balanced.states)
    np.testing.assert_array_equal(permuted.next_states, balanced.next_states)
    np.testing.assert_array_equal(permuted.rewards, balanced.rewards)
    assert dataset_diagnostics(permuted)["global_action_coordinate_histograms"] == (
        dataset_diagnostics(balanced)["global_action_coordinate_histograms"]
    )
    assert semantic_hash(permuted) != semantic_hash(balanced)

    low_density = generate_dataset(FactorCell(True, True, 1))
    constant = constant_nuisance_control(low_density)
    np.testing.assert_array_equal(constant.states[:, :2], low_density.states[:, :2])
    np.testing.assert_array_equal(constant.actions, low_density.actions)
    assert np.all(constant.states[:, 2] == 0.0)


def test_exact_model_and_unchanged_planner_solve_fixed_starts() -> None:
    env: Environment = BridgeControlEnv()
    assert isinstance(env, Environment)
    planner = FlatPlanner(
        ExactBridgeModel(),
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=12,
        candidates=64,
        elites=8,
        iterations=3,
        seed=0,
    )
    agent = Agent(
        encode=lambda obs: LatentState(z=np.asarray(obs.data, dtype=float)),
        planner=planner,
    )
    successes = []
    for start in EVAL_STARTS:
        bridge = BridgeControlEnv()
        obs = bridge.set_state(start)
        agent.reset()
        for _ in range(14):
            obs, _, _ = bridge.step(agent.act(obs))
        final = np.asarray(obs.data, dtype=float)
        successes.append(final[0] >= 0.75 and abs(final[1] - GOAL_Y) <= 0.20)
    assert all(successes)
    assert all(_evaluate_exact(seed).success_rate == 1.0 for seed in MODEL_SEEDS)


def test_prepare_contains_no_model_outcomes_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "stale.svg").write_text("stale", encoding="utf-8")
    (tmp_path / "BC-001-results.json").write_text("{}", encoding="utf-8")
    manifest = prepare(tmp_path)
    serialized = json.dumps(manifest, sort_keys=True)
    for forbidden in ("mean_eval_return", "success_rate", "episode_returns"):
        assert forbidden not in serialized
    assert verify(tmp_path)["status"] == "verified"
    assert not (tmp_path / "BC-001-results.json").exists()
    assert not (tmp_path / "plots").exists()
    protocol = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["source_sha256"] == protocol_record()["source_sha256"]
    with pytest.raises(ValueError, match="complete verified result"):
        verify(tmp_path, require_results=True)

    dataset_entry = manifest["datasets"]["b1_r1_d8"]
    dataset_path = tmp_path / dataset_entry["path"]
    dataset_path.write_bytes(dataset_path.read_bytes()[:-8])
    with pytest.raises((BadZipFile, OSError, ValueError)):
        verify(tmp_path)


def test_verify_rejects_forged_protocol_and_manifest_diagnostics(tmp_path: Path) -> None:
    protocol_dir = tmp_path / "protocol"
    prepare(protocol_dir)
    protocol_path = protocol_dir / "protocol.json"
    manifest_path = protocol_dir / "dataset-manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["protocol_document_sha256"] = "0" * 64
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest["protocol_sha256"] = sha256(canonical).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="saved protocol"):
        verify(protocol_dir)

    diagnostics_dir = tmp_path / "diagnostics"
    prepare(diagnostics_dir)
    diagnostics_manifest_path = diagnostics_dir / "dataset-manifest.json"
    diagnostics_manifest = json.loads(
        diagnostics_manifest_path.read_text(encoding="utf-8")
    )
    diagnostics_manifest["datasets"]["b1_r1_d8"]["manipulation_checks"][
        "bridge_edge_count"
    ] = 999
    diagnostics_manifest_path.write_text(
        json.dumps(diagnostics_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manipulation checks"):
        verify(diagnostics_dir)


def test_verify_requires_named_controls_and_complete_outcomes(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing-control"
    prepare(missing_dir)
    manifest_path = missing_dir / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["datasets"]["control_action_permuted"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly eight primary and two control"):
        verify(missing_dir)

    partial_dir = tmp_path / "partial"
    prepare(partial_dir)
    (partial_dir / "BC-001-report.md").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="partial or stale"):
        verify(partial_dir)


def test_formal_result_package_verifies_and_detects_tampering(tmp_path: Path) -> None:
    if not (DEFAULT_OUTPUT / "artifact-manifest.json").exists():
        pytest.skip("formal BC-001 result package has not been generated")
    copied = tmp_path / "result"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    assert verify(copied, require_results=True)["outcomes"] == "verified_results"

    result_path = copied / "BC-001-results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "fabricated"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="status"):
        verify(copied)
