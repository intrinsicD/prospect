from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bench.bridge_control.experiment import DEFAULT_OUTPUT as BC_OUTPUT
from bench.bridge_control.experiment import verify as verify_bridge_control
from bench.bridge_control.fixture import ExactBridgeModel, transition_dynamics
from bench.oracle_ladder.audit import build_fixed_bank, rank_audit
from bench.oracle_ladder.experiment import _read_json, prepare, verify
from bench.oracle_ladder.models import MeanOraclePrefixModel, MeanWorldModelAdapter
from prospect.planning import FlatPlanner
from prospect.types import Action, LatentState
from prospect.world_model import FlatWorldModel


def _model(seed: int = 3) -> FlatWorldModel:
    return FlatWorldModel(
        obs_dim=3,
        action_dim=2,
        latent_dim=6,
        hidden=12,
        ensemble=3,
        seed=seed,
    )


def _planner(model: object, seed: int, horizon: int = 4) -> FlatPlanner:
    return FlatPlanner(
        model,  # type: ignore[arg-type]
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=horizon,
        candidates=12,
        elites=3,
        iterations=2,
        uncertainty_penalty=0.0,
        seed=seed,
    )


def test_sealed_parent_still_verifies() -> None:
    assert verify_bridge_control(BC_OUTPUT, require_results=True)["outcomes"] == "verified_results"


def test_mean_adapter_deliberately_omits_member_batch() -> None:
    adapter = MeanWorldModelAdapter(_model())
    assert not hasattr(adapter, "predict_member_batch")


def test_k0_wrapper_matches_recursive_mean_scores_actions_and_warm_start() -> None:
    model = _model()
    direct = MeanWorldModelAdapter(model)
    wrapped = MeanOraclePrefixModel(model, prefix_steps=0, horizon=4)
    raw = np.array([-0.9, -0.2, 0.5])
    rng = np.random.default_rng(41)
    sequences = rng.uniform(-1.0, 1.0, size=(12, 4, 2))

    direct_planner = _planner(direct, 19)
    wrapped_planner = _planner(wrapped, 19)
    direct_state = model.encode(raw)
    wrapped_state = wrapped.initial_state(raw)
    assert np.array_equal(
        direct_planner._imagined_returns(direct_state, sequences),
        wrapped_planner._imagined_returns(wrapped_state, sequences),
    )
    direct_action = direct_planner.plan(direct_state)
    wrapped_action = wrapped_planner.plan(wrapped_state)
    assert np.array_equal(direct_action.data, wrapped_action.data)
    assert direct_planner._warm_mean is not None
    assert wrapped_planner._warm_mean is not None
    assert direct_planner._warm_elites is not None
    assert wrapped_planner._warm_elites is not None
    assert np.array_equal(direct_planner._warm_mean, wrapped_planner._warm_mean)
    assert np.array_equal(direct_planner._warm_elites, wrapped_planner._warm_elites)

    corrupted = wrapped_state.z.copy()
    corrupted[6:9] = np.array([0.91, -0.73, 0.11])
    corrupted_scores = _planner(wrapped, 19)._imagined_returns(
        LatentState(z=corrupted, ood=wrapped_state.ood), sequences
    )
    assert np.array_equal(
        direct_planner._imagined_returns(direct_state, sequences),
        corrupted_scores,
    )


def test_full_oracle_wrapper_matches_exact_model() -> None:
    horizon = 4
    model = _model()
    wrapped = MeanOraclePrefixModel(
        model,
        prefix_steps=horizon,
        horizon=horizon,
        reward_source="oracle",
    )
    exact = ExactBridgeModel()
    raw = np.array([-0.9, 0.2, -0.25])
    sequences = np.random.default_rng(52).uniform(-1.0, 1.0, size=(12, horizon, 2))
    wrapped_planner = _planner(wrapped, 23, horizon)
    exact_planner = _planner(exact, 23, horizon)
    assert np.array_equal(
        wrapped_planner._imagined_returns(wrapped.initial_state(raw), sequences),
        exact_planner._imagined_returns(LatentState(z=raw), sequences),
    )
    assert np.array_equal(
        wrapped_planner.plan(wrapped.initial_state(raw)).data,
        exact_planner.plan(LatentState(z=raw)).data,
    )


def test_prefix_boundary_is_candidate_carried_and_raw_sidecar_freezes() -> None:
    model = _model()
    wrapped = MeanOraclePrefixModel(model, prefix_steps=1, horizon=4)
    raw = np.array([-0.9, -0.28, -0.75])
    action = Action(data=np.array([0.7, 0.4]))
    initial = wrapped.initial_state(raw)
    first = wrapped.predict(initial, action)
    expected_raw, _ = transition_dynamics(raw, action.data)
    assert np.allclose(first.mean[6:9], expected_raw)
    assert first.mean[-1] == 0.0
    assert np.allclose(first.mean[:6], model.encode_target(expected_raw).z)

    second = wrapped.predict(LatentState(z=first.mean), action)
    assert np.array_equal(second.mean[6:9], first.mean[6:9])
    assert second.mean[-1] == 0.0
    assert wrapped.initial_state(raw).z[-1] == 1.0


def test_online_and_target_refresh_are_explicitly_distinct_interfaces() -> None:
    model = _model()
    raw = np.array([-0.65, -0.1, 0.25])
    action = Action(data=np.array([0.3, -0.8]))
    next_raw, _ = transition_dynamics(raw, action.data)
    target = MeanOraclePrefixModel(model, 1, 4, refresh="target")
    online = MeanOraclePrefixModel(model, 1, 4, refresh="online")
    target_prediction = target.predict(target.initial_state(raw), action)
    online_prediction = online.predict(online.initial_state(raw), action)
    assert np.allclose(target_prediction.mean[:6], model.encode_target(next_raw).z)
    assert np.allclose(online_prediction.mean[:6], model.encode(next_raw).z)


def test_oracle_reward_requires_a_full_exact_horizon() -> None:
    with pytest.raises(ValueError, match="covers the planning horizon"):
        MeanOraclePrefixModel(_model(), prefix_steps=3, horizon=4, reward_source="oracle")


def test_fixed_bank_is_deterministic_and_self_auditing() -> None:
    start = np.array([-0.9, -0.28, -0.75])
    left = build_fixed_bank(start, seed=714_001)
    right = build_fixed_bank(start, seed=714_001)
    assert left.sha256 == right.sha256
    assert np.array_equal(left.sequences, right.sequences)
    audit = rank_audit(left, left.exact_scores)
    assert audit.normalized_selected_regret == 0.0
    assert audit.pearson == pytest.approx(1.0)
    assert audit.spearman == pytest.approx(1.0)
    assert audit.reference_top_k_fraction == 1.0


def test_prepare_copies_and_binds_parent_dataset(tmp_path: Path) -> None:
    output = tmp_path / "OL-001"
    assert prepare(output)["outcomes"] == "prepared_only"
    assert verify(output)["outcomes"] == "prepared_only"
    strict = subprocess.run(
        [sys.executable, "-m", "bench.oracle_ladder", "verify", "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode != 0
    assert "complete verified OL-001 result package" in strict.stderr
    copied = output / "inputs/BC-001-b1_r1_d8.npz"
    original = copied.read_bytes()
    copied.write_bytes((BC_OUTPUT / "datasets/b0_r0_d1.npz").read_bytes())
    with pytest.raises(ValueError, match="sealed parent selection"):
        verify(output)
    copied.write_bytes(original)
    assert verify(output)["outcomes"] == "prepared_only"
    payload = bytearray(copied.read_bytes())
    payload[-1] ^= 1
    copied.write_bytes(payload)
    with pytest.raises((ValueError, OSError)):
        verify(output)


def test_prepare_refuses_unsafe_or_formal_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository root"):
        prepare(Path("."))
    formal = tmp_path / "formal"
    formal.mkdir()
    (formal / "OL-001-results.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="formal results"):
        prepare(formal)


def test_json_reader_rejects_non_finite_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        _read_json(path)
