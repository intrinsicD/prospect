from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("torchrl")
pytest.importorskip("gymnasium")

from bench.world_model_lifecycle.checkpoint import (
    CANONICAL_COMPONENT_IDS,
    ComponentPayload,
    canonical_json_bytes,
    load_checkpoint,
    save_checkpoint,
    snapshot_numpy_rng,
    snapshot_python_rng,
    snapshot_torch_accelerator_rng,
    snapshot_torch_cpu_rng,
)
from bench.world_model_lifecycle.experiment import (
    ReplicateRows,
    append_collection_rows,
    append_update_row,
    save_replicate_checkpoint,
)
from bench.world_model_lifecycle.learning import (
    TransactionalWorldModelLearner,
    WorldModelRuntime,
)
from bench.world_model_lifecycle.model import (
    SAMPLING_FORMAT,
    FixedScaling,
    WorldModelConfig,
)
from bench.world_model_lifecycle.parity import (
    compare_parity,
    restore_checkpoint_state,
    snapshot_planner_rng,
)
from bench.world_model_lifecycle.planning import (
    CEMController,
    make_learned_model_env,
)
from bench.world_model_lifecycle.runtime_lane import (
    RuntimeCustody,
    UniformRandomController,
    collect_episodes,
)
from prospect.decision import CounterIdentitySource
from prospect.domain import TimePoint

_MAGIC = b"PROSPECT-WM001\0"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _target_digest(
    observation: list[float],
    next_observation: list[float],
    reward: float,
) -> str:
    target = np.asarray(
        _scaled_target(observation, next_observation, reward),
        dtype="<f8",
    )
    return hashlib.sha256(target.tobytes()).hexdigest()


def _scaled_target(
    observation: list[float],
    next_observation: list[float],
    reward: float,
) -> list[float]:
    before = np.asarray(observation, dtype=np.float64)
    after = np.asarray(next_observation, dtype=np.float64)
    return [
        float(value)
        for value in np.asarray(
            [
                (after[0] - before[0]) / 2.0,
                (after[1] - before[1]) / 2.0,
                (after[2] - before[2]) / 16.0,
                reward / 16.2736044,
            ],
            dtype=np.float64,
        )
    ]


def _sampling_manifest(
    transition_ids: list[str],
    indices: np.ndarray,
) -> bytes:
    raw = indices.astype("<u4", copy=False).tobytes(order="C")
    identity_payload = json.dumps(
        transition_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    header = canonical_json_bytes(
        {
            "dtype": "uint32-le",
            "format": SAMPLING_FORMAT,
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(indices.shape),
            "transition_ids_sha256": hashlib.sha256(identity_payload).hexdigest(),
        }
    )
    return _MAGIC + struct.pack(">Q", len(header)) + header + raw


def _multiset_digest(
    transition_ids: list[str],
    indices: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for index in indices.reshape(-1):
        digest.update(transition_ids[int(index)].encode() + b"\n")
    return digest.hexdigest()


def _summary_only_components() -> tuple[dict[str, ComponentPayload], str, int]:
    """Return the pre-graph fixture to prove summary-only restore is rejected."""
    runtime = WorldModelRuntime.initialize(
        initialization_seed=19,
        model_config=WorldModelConfig(
            ensemble_members=1,
            hidden_dimensions=(4, 4),
        ),
        device="cpu",
    )
    planner = CEMController(
        make_learned_model_env(runtime.model, device="cpu"),
        seed=918273,
    )
    planner_digest = planner.rng_digest

    observation_a = [1.0, 0.0, 0.0]
    next_a = [0.99, 0.01, 0.1]
    observation_b = [0.0, 1.0, 0.2]
    next_b = [0.01, 0.99, 0.3]
    reward_a = -0.5
    reward_b = -1.25
    transition_ids = ["transition-a", "transition-b"]
    experience_rows = [
        {
            "transition_id": "transition-a",
            "run_id": "fixture",
            "episode_id": "episode-a",
            "task_id": "pendulum_normal_torque",
            "task_context": 0.0,
            "split": "collect_a",
            "step_index": 0,
            "real_or_imagined": "real",
            "pre_observation_id": "pre-a",
            "decision_id": "decision-a",
            "executed_action_id": "execution-a",
            "next_observation_id": "next-a",
            "model_version_at_action": runtime.version,
            "parameter_sha256_at_action": runtime.digest,
            "pre_observation": observation_a,
            "intended_action": 0.5,
            "applied_action": 0.5,
            "next_observation": next_a,
            "reward": reward_a,
            "terminated": False,
            "truncated": False,
            "scaled_target": _scaled_target(
                observation_a,
                next_a,
                reward_a,
            ),
            "target_sha256": _target_digest(observation_a, next_a, reward_a),
        },
        {
            "transition_id": "transition-b",
            "run_id": "fixture",
            "episode_id": "episode-b",
            "task_id": "pendulum_reversed_torque",
            "task_context": 1.0,
            "split": "collect_b",
            "step_index": 0,
            "real_or_imagined": "real",
            "pre_observation_id": "pre-b",
            "decision_id": "decision-b",
            "executed_action_id": "execution-b",
            "next_observation_id": "next-b",
            "model_version_at_action": runtime.version,
            "parameter_sha256_at_action": runtime.digest,
            "pre_observation": observation_b,
            "intended_action": -0.25,
            "applied_action": 0.25,
            "next_observation": next_b,
            "reward": reward_b,
            "terminated": False,
            "truncated": False,
            "scaled_target": _scaled_target(
                observation_b,
                next_b,
                reward_b,
            ),
            "target_sha256": _target_digest(observation_b, next_b, reward_b),
        },
    ]
    replay_index = {
        "schema": "prospect.wm001.replay-index.v1",
        "canonical_experience_rows": [
            {
                "experience_id": "experience-a",
                "run_id": "fixture",
                "task_id": "pendulum_normal_torque",
                "episode_id": "episode-a",
                "step_index": 0,
                "closed_at": ["interaction", 2],
            },
            {
                "experience_id": "experience-b",
                "run_id": "fixture",
                "task_id": "pendulum_reversed_torque",
                "episode_id": "episode-b",
                "step_index": 0,
                "closed_at": ["interaction", 4],
            },
        ],
        "collect_a": {
            "transition_ids": ["transition-a"],
            "observations": [observation_a],
            "contexts": [0.0],
            "actions": [0.5],
            "next_observations": [next_a],
            "rewards": [reward_a],
        },
        "collect_b": {
            "transition_ids": ["transition-b"],
            "observations": [observation_b],
            "contexts": [1.0],
            "actions": [-0.25],
            "next_observations": [next_b],
            "rewards": [reward_b],
        },
    }
    a_indices = np.zeros((1, 5, 256), dtype="<u4")
    b_indices = (np.arange(5 * 256, dtype="<u4") % 2).reshape(1, 5, 256)
    manifest_a = _sampling_manifest(["transition-a"], a_indices)
    manifest_b = _sampling_manifest(transition_ids, b_indices)
    intermediate_parameter = _digest("after-a-parameters")
    initial_parameter = _digest("cold-parameters")
    intermediate_live = _digest("after-a-live")
    initial_live = _digest("cold-live")
    model_ledger = [
        {
            "phase": "train_a",
            "predecessor_parameter_sha256": initial_parameter,
            "candidate_parameter_sha256": intermediate_parameter,
            "predecessor_live_state_sha256": initial_live,
            "candidate_live_state_sha256": intermediate_live,
        },
        {
            "phase": "train_b_replay",
            "predecessor_parameter_sha256": intermediate_parameter,
            "candidate_parameter_sha256": runtime.digest,
            "predecessor_live_state_sha256": intermediate_live,
            "candidate_live_state_sha256": runtime.live_state_digest,
        },
    ]
    update_receipts = {
        "schema": "prospect.wm001.update-receipts.v1",
        "updates": [
            {
                "receipt_id": "receipt-a",
                "phase": "train_a",
                "status": "committed",
                "predecessor_parameter_sha256": initial_parameter,
                "candidate_parameter_sha256": intermediate_parameter,
                "committed_parameter_sha256": intermediate_parameter,
                "predecessor_model_version": f"wm001-state-sha256:{initial_live}",
                "committed_model_version": f"wm001-state-sha256:{intermediate_live}",
                "eligible_splits": ["collect_a"],
                "eligible_transition_count": 1,
                "eligible_transition_ids": ["transition-a"],
                "consumed_sample_count": 5 * 256,
                "consumed_multiset_sha256": _multiset_digest(
                    ["transition-a"],
                    a_indices,
                ),
                "sampling_manifest_sha256": hashlib.sha256(manifest_a).hexdigest(),
                "target_permutation_sha256": None,
                "target_permutation_file": None,
                "optimizer_steps": 1,
                "live_state_before_sha256": initial_live,
                "live_state_after_sha256": intermediate_live,
            },
            {
                "receipt_id": "receipt-b",
                "phase": "train_b_replay",
                "status": "committed",
                "predecessor_parameter_sha256": intermediate_parameter,
                "candidate_parameter_sha256": runtime.digest,
                "committed_parameter_sha256": runtime.digest,
                "predecessor_model_version": f"wm001-state-sha256:{intermediate_live}",
                "committed_model_version": runtime.version,
                "eligible_splits": ["collect_a", "collect_b"],
                "eligible_transition_count": 2,
                "eligible_transition_ids": transition_ids,
                "consumed_sample_count": 5 * 256,
                "consumed_multiset_sha256": _multiset_digest(
                    transition_ids,
                    b_indices,
                ),
                "sampling_manifest_sha256": hashlib.sha256(manifest_b).hexdigest(),
                "target_permutation_sha256": None,
                "target_permutation_file": None,
                "optimizer_steps": 1,
                "live_state_before_sha256": intermediate_live,
                "live_state_after_sha256": runtime.live_state_digest,
            },
        ],
    }
    identities = CounterIdentitySource("wm001-parity-fixture")
    identities.next("prior")
    agent_runtime = {
        "schema": "prospect.wm001.agent-runtime.v1",
        "identity_namespace": identities.namespace,
        "identity_checkpoint_base64": base64.b64encode(identities.checkpoint_bytes()).decode(),
        "next_tick": 5,
        "agent_id": "prospect-wm001-agent",
        "configuration_version": f"wm001-config:{runtime.version}",
        "memory_version": "memory-v1",
        "knowledge_version": "knowledge-v1",
        "model_version": runtime.version,
        "representation_version": "wm001-physical-state-v1",
        "policy_version": "wm001-controller-policy-v1",
        "belief_id": "belief-v1",
    }
    collection_rng_states = {
        "task_a": np.random.default_rng(1).bit_generator.state,
        "task_b": np.random.default_rng(2).bit_generator.state,
    }
    payloads = {
        "world_model": runtime.model_bytes,
        "optimizer": runtime.optimizer_bytes,
        "model_version_ledger": canonical_json_bytes(model_ledger),
        "experience_store": canonical_json_bytes(
            {
                "schema": "prospect.wm001.experience-custody.v1",
                "transition_rows": experience_rows,
            }
        ),
        "replay_index": canonical_json_bytes(replay_index),
        "replay_sampling_history": canonical_json_bytes(
            {
                "schema": "prospect.wm001.replay-sampling-history.v1",
                "manifests": [
                    {
                        "phase": "train_a",
                        "sha256": hashlib.sha256(manifest_a).hexdigest(),
                        "bytes": len(manifest_a),
                        "payload_base64": base64.b64encode(manifest_a).decode(),
                    },
                    {
                        "phase": "train_b_replay",
                        "sha256": hashlib.sha256(manifest_b).hexdigest(),
                        "bytes": len(manifest_b),
                        "payload_base64": base64.b64encode(manifest_b).decode(),
                    },
                ],
            }
        ),
        "update_receipts": canonical_json_bytes(update_receipts),
        "agent_runtime": canonical_json_bytes(agent_runtime),
        "scaling_configuration": canonical_json_bytes(
            {
                "schema": "prospect.wm001.fixed-scaling.v1",
                **asdict(FixedScaling()),
            }
        ),
        "python_rng": snapshot_python_rng(),
        "numpy_rng": snapshot_numpy_rng(),
        "torch_cpu_rng": snapshot_torch_cpu_rng(),
        "torch_accelerator_rng": snapshot_torch_accelerator_rng(),
        "collection_rng": canonical_json_bytes(
            {
                "schema": "prospect.wm001.collection-rng.v1",
                "states": collection_rng_states,
            }
        ),
        "planner_rng": snapshot_planner_rng(planner),
    }
    assert tuple(payloads) == CANONICAL_COMPONENT_IDS
    metadata = {
        "world_model": (runtime.version, "application/vnd.prospect.wm001.model"),
        "optimizer": (runtime.version, "application/vnd.prospect.wm001.adamw"),
        "model_version_ledger": (runtime.version, "application/json"),
        "experience_store": ("wm001-experience-v1", "application/json"),
        "replay_index": ("wm001-replay-v1", "application/json"),
        "replay_sampling_history": ("wm001-sampling-v1", "application/json"),
        "update_receipts": ("wm001-updates-v1", "application/json"),
        "agent_runtime": (runtime.version, "application/json"),
        "scaling_configuration": ("wm001-fixed-scaling-v1", "application/json"),
        "python_rng": (
            "python-rng-v1",
            "application/vnd.prospect.rng-state+json",
        ),
        "numpy_rng": (
            "numpy-rng-v1",
            "application/vnd.prospect.rng-state+json",
        ),
        "torch_cpu_rng": (
            "torch-rng-v1",
            "application/vnd.prospect.rng-state+json",
        ),
        "torch_accelerator_rng": (
            "torch-rng-v1",
            "application/vnd.prospect.rng-state+json",
        ),
        "collection_rng": ("collection-rng-v1", "application/json"),
        "planner_rng": ("planner-rng-v1", "application/json"),
    }
    components = {
        component_id: ComponentPayload(
            component_id=component_id,
            logical_version=metadata[component_id][0],
            payload=payload,
            media_type=metadata[component_id][1],
        )
        for component_id, payload in payloads.items()
    }
    return components, planner_digest, planner.state_dict()["seed"]


@pytest.fixture(scope="module")
def semantic_components(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, ComponentPayload], str, int]:
    """Build the parity fixture through the real custody and save path."""

    directory = tmp_path_factory.mktemp("wm001-semantic-checkpoint")
    rows = ReplicateRows.create("wm001-parity-fixture", 101)
    runtime = WorldModelRuntime.initialize(
        initialization_seed=19,
        model_config=WorldModelConfig(
            ensemble_members=5,
            hidden_dimensions=(4, 4),
        ),
        device="cpu",
    )
    custody = RuntimeCustody.create("wm001-parity-fixture")

    learner_a = TransactionalWorldModelLearner(
        phase="train_a",
        bootstrap_seeds=(11, 12, 13, 14, 15),
        minibatch_order_seed=16,
        optimizer_steps=1,
        device="cpu",
    )
    controller_a = UniformRandomController(17)
    collection_a = collect_episodes(
        run_id="wm001-parity-fixture:collect-a",
        task_id="pendulum_normal_torque",
        episode_seeds=(18,),
        controller_factory=lambda _: controller_a,
        backend=runtime,
        custody=custody,
        model_owner=runtime.owner,
        learner=learner_a,
    )
    append_collection_rows(
        rows,
        collection_a,
        split="collect_a",
        condition="collection_random",
        checkpoint_id="cold",
        learning_allowed=True,
        replay_writes_allowed=True,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
    )
    receipt_a = collection_a.final_agent.learn(
        custody.replay.require_transitions(collection_a.transitions),
        at=TimePoint(collection_a.next_tick),
    )
    assert learner_a.last_evidence is not None
    append_update_row(
        rows,
        receipt_a,
        learner_a.last_evidence,
        phase="train_a",
        eligible_splits=("collect_a",),
        committed_parameter_sha256=runtime.digest,
        committed_live_state_sha256=runtime.live_state_digest,
        manifest_directory=directory,
    )

    learner_b = TransactionalWorldModelLearner(
        phase="train_b_replay",
        bootstrap_seeds=(21, 22, 23, 24, 25),
        minibatch_order_seed=26,
        optimizer_steps=1,
        balanced_tasks=True,
        device="cpu",
    )
    controller_b = UniformRandomController(27)
    collection_b = collect_episodes(
        run_id="wm001-parity-fixture:collect-b",
        task_id="pendulum_reversed_torque",
        episode_seeds=(28,),
        controller_factory=lambda _: controller_b,
        backend=runtime,
        custody=custody,
        start_tick=receipt_a.completed_at.tick + 2,
        model_owner=runtime.owner,
        learner=learner_b,
    )
    append_collection_rows(
        rows,
        collection_b,
        split="collect_b",
        condition="collection_random",
        checkpoint_id="after_a",
        learning_allowed=True,
        replay_writes_allowed=True,
        started_at="2026-01-01T00:00:02Z",
        completed_at="2026-01-01T00:00:03Z",
    )
    receipt_b = collection_b.final_agent.learn(
        custody.replay.require_transitions((*collection_a.transitions, *collection_b.transitions)),
        at=TimePoint(collection_b.next_tick),
    )
    assert learner_b.last_evidence is not None
    append_update_row(
        rows,
        receipt_b,
        learner_b.last_evidence,
        phase="train_b_replay",
        eligible_splits=("collect_a", "collect_b"),
        committed_parameter_sha256=runtime.digest,
        committed_live_state_sha256=runtime.live_state_digest,
        manifest_directory=directory,
    )

    planner = CEMController(
        make_learned_model_env(runtime.model, device="cpu"),
        seed=918273,
    )
    checkpoint_path = directory / "real-save-replicate.checkpoint"
    save_replicate_checkpoint(
        path=checkpoint_path,
        rows=rows,
        runtime=runtime,
        collection=collection_b,
        collection_a=collection_a.transitions,
        collection_b=collection_b.transitions,
        collection_rng_states={
            "task_a": controller_a.state(),
            "task_b": controller_b.state(),
        },
        planner=planner,
        update_evidence=(learner_a.last_evidence, learner_b.last_evidence),
        created_at=receipt_b.completed_at,
    )
    loaded = load_checkpoint(checkpoint_path)
    metadata = {record.component_id: (record.logical_version, record.media_type) for record in loaded.report.components}
    components = {
        component_id: ComponentPayload(
            component_id=component_id,
            logical_version=metadata[component_id][0],
            payload=loaded.payload(component_id),
            media_type=metadata[component_id][1],
        )
        for component_id in CANONICAL_COMPONENT_IDS
    }
    return components, planner.rng_digest, int(planner.state_dict()["seed"])


def _save_fixture(
    path: Path,
    components: dict[str, ComponentPayload],
) -> None:
    save_checkpoint(
        path,
        checkpoint_id="wm001-parity-fixture",
        agent_id="prospect-wm001-agent",
        created_at=TimePoint(4),
        components=components,
    )


def test_restore_semantically_consumes_all_components_and_planner_state(
    tmp_path: Path,
    semantic_components: tuple[dict[str, ComponentPayload], str, int],
) -> None:
    components, planner_digest, planner_seed = semantic_components
    path = tmp_path / "valid.checkpoint"
    _save_fixture(path, components)

    restored = restore_checkpoint_state(path, device="cpu")

    assert restored.next_tick == restored.snapshot.captured_at.tick + 1
    assert restored.runtime.digest == hashlib.sha256(components["world_model"].payload).hexdigest()
    assert restored.planner.rng_digest == planner_digest
    assert restored.planner.state_dict()["seed"] == planner_seed
    assert tuple(restored.component_hashes) == CANONICAL_COMPONENT_IDS
    assert restored.component_hashes == {
        component_id: hashlib.sha256(components[component_id].payload).hexdigest()
        for component_id in CANONICAL_COMPONENT_IDS
    }
    assert len(restored.custody.store) == 400
    assert restored.custody.ledger.transition_count == 400
    assert restored.custody.ledger.update_count == 2
    assert restored.custody.replay.event_count == 400
    assert len(restored.transitions) == 400
    assert len(restored.updates) == 2
    assert restored.snapshot.latest_update is restored.updates[-1]
    assert restored.snapshot.belief is restored.updates[-1].resulting_belief
    reconstructed_boundary = restored.state.snapshot(
        restored.snapshot.agent_id,
        restored.snapshot.captured_at,
    )
    assert reconstructed_boundary == restored.snapshot
    assert reconstructed_boundary.belief is restored.snapshot.belief
    assert reconstructed_boundary.latest_update is restored.snapshot.latest_update
    for transition in restored.transitions:
        canonical_event = restored.custody.store.get(transition.experience.experience_id)
        assert transition.experience is canonical_event
        assert restored.custody.replay.get(canonical_event.experience_id) is (canonical_event)
        assert restored.custody.ledger.get_transition(transition.transition_id) is transition
    for receipt in restored.updates:
        assert restored.custody.ledger.get_update(receipt.receipt_id) is receipt
        assert all(
            restored.custody.ledger.get_transition(transition.transition_id) is transition
            for transition in receipt.transitions
        )
    collection_rng_value = json.loads(components["collection_rng"].payload)
    for task in ("task_a", "task_b"):
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = collection_rng_value["states"][task]
        np.testing.assert_array_equal(
            restored.collection_rngs[task].integers(0, 2**31, size=8),
            expected_rng.integers(0, 2**31, size=8),
        )


def test_summary_rows_without_domain_custody_graph_are_rejected(
    tmp_path: Path,
) -> None:
    components, _, _ = _summary_only_components()
    path = tmp_path / "summary-only.checkpoint"
    _save_fixture(path, components)

    with pytest.raises(ValueError, match="experience store has an invalid field set"):
        restore_checkpoint_state(path, device="cpu")


def test_restored_custody_accepts_new_experience_and_transactional_learning(
    tmp_path: Path,
    semantic_components: tuple[dict[str, ComponentPayload], str, int],
) -> None:
    components, _, _ = semantic_components
    path = tmp_path / "continued.checkpoint"
    _save_fixture(path, components)
    restored = restore_checkpoint_state(path, device="cpu")
    prior_transition_ids = {transition.transition_id for transition in restored.transitions}
    prior_experience_ids = {transition.experience.experience_id for transition in restored.transitions}
    predecessor_digest = restored.runtime.live_state_digest
    learner = TransactionalWorldModelLearner(
        phase="restart_continuation_probe",
        bootstrap_seeds=(31, 32, 33, 34, 35),
        minibatch_order_seed=36,
        optimizer_steps=1,
        device="cpu",
    )
    collection = collect_episodes(
        run_id="wm001-parity-fixture:continued",
        task_id="pendulum_normal_torque",
        episode_seeds=(37,),
        controller_factory=lambda _: UniformRandomController(38),
        backend=restored.runtime,
        custody=restored.custody,
        start_tick=restored.next_tick,
        model_owner=restored.runtime.owner,
        learner=learner,
    )
    receipt = collection.final_agent.learn(
        restored.custody.replay.require_transitions(collection.transitions),
        at=TimePoint(collection.next_tick),
    )

    assert len(restored.custody.store) == 600
    assert restored.custody.ledger.transition_count == 600
    assert restored.custody.ledger.update_count == 3
    assert restored.custody.replay.event_count == 600
    assert prior_transition_ids.isdisjoint(transition.transition_id for transition in collection.transitions)
    assert prior_experience_ids.isdisjoint(transition.experience.experience_id for transition in collection.transitions)
    assert receipt is restored.custody.ledger.get_update(receipt.receipt_id)
    assert receipt.transitions == collection.transitions
    assert restored.runtime.live_state_digest != predecessor_digest


def test_parity_counts_persistent_boundary_or_custody_divergence() -> None:
    component_hashes = {component_id: _digest(component_id) for component_id in CANONICAL_COMPONENT_IDS}
    original: dict[str, object] = {
        "schema": "prospect.wm001.restart-evaluation.v1",
        "process_id": 10,
        "checkpoint_manifest_sha256": _digest("manifest"),
        "component_hashes": component_hashes,
        "model_version": "model-v1",
        "parameter_sha256": _digest("parameters"),
        "boundary_state": {"belief_id": "belief-v1", "experiences": 400},
        "post_evaluation_custody": {"experiences": 800},
        "tasks": [],
    }
    restored = copy.deepcopy(original)
    restored["process_id"] = 11
    restored["boundary_state"] = {
        "belief_id": "belief-v0",
        "experiences": 0,
    }

    parity = compare_parity(original, restored)

    assert parity["fresh_process"] is True
    assert parity["identity_or_lineage_mismatches"] == 1


@pytest.mark.parametrize("component_id", CANONICAL_COMPONENT_IDS)
def test_integrity_valid_but_semantically_corrupt_component_fails_restore(
    tmp_path: Path,
    semantic_components: tuple[dict[str, ComponentPayload], str, int],
    component_id: str,
) -> None:
    valid, _, _ = semantic_components
    corrupt = dict(valid)
    source = valid[component_id]
    corrupt[component_id] = ComponentPayload(
        component_id=component_id,
        logical_version=source.logical_version,
        payload=b"{}",
        media_type=source.media_type,
    )
    path = tmp_path / f"corrupt-{component_id}.checkpoint"
    _save_fixture(path, corrupt)
    rng_before = (
        snapshot_python_rng(),
        snapshot_numpy_rng(),
        snapshot_torch_cpu_rng(),
        snapshot_torch_accelerator_rng(),
    )

    with pytest.raises((ValueError, RuntimeError)):
        restore_checkpoint_state(path, device="cpu")
    assert (
        snapshot_python_rng(),
        snapshot_numpy_rng(),
        snapshot_torch_cpu_rng(),
        snapshot_torch_accelerator_rng(),
    ) == rng_before


def test_integrity_valid_but_incompatible_component_metadata_fails_restore(
    tmp_path: Path,
    semantic_components: tuple[dict[str, ComponentPayload], str, int],
) -> None:
    valid, _, _ = semantic_components
    incompatible = dict(valid)
    source = valid["planner_rng"]
    incompatible["planner_rng"] = ComponentPayload(
        component_id="planner_rng",
        logical_version=source.logical_version,
        payload=source.payload,
        media_type="application/octet-stream",
    )
    path = tmp_path / "incompatible-metadata.checkpoint"
    _save_fixture(path, incompatible)

    with pytest.raises(ValueError, match="planner_rng checkpoint metadata"):
        restore_checkpoint_state(path, device="cpu")
