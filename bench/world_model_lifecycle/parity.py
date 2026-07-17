"""Live-state versus fresh-process checkpoint parity for WM-001.

K7 compares two genuinely different execution paths:

* the original path continues from the retained in-memory runtime, identity
  source, clock, and planner present at the checkpoint boundary; and
* the restart path verifies and semantically decodes every one of the fifteen
  sealed checkpoint components in a fresh interpreter before acting.

The original path never reloads its own checkpoint.  This matters: comparing
two deserializations would only test serializer determinism, not continuation
of the state that actually produced the checkpoint.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from prospect.decision import CounterIdentitySource
from prospect.domain import (
    AgentSnapshot,
    EpistemicTransition,
    ExperienceEvent,
    TimePoint,
    UpdateReceipt,
    UpdateStatus,
)
from prospect.runtime import AgentState
from prospect.storage import EpistemicLedger, InMemoryExperienceStore

from .checkpoint import (
    CANONICAL_COMPONENT_IDS,
    LoadedWMCheckpoint,
    canonical_json_bytes,
    load_checkpoint,
    restore_numpy_rng,
    restore_python_rng,
    restore_torch_accelerator_rng,
    restore_torch_cpu_rng,
    snapshot_numpy_rng,
    snapshot_python_rng,
    snapshot_torch_accelerator_rng,
    snapshot_torch_cpu_rng,
)
from .domain_graph import (
    MAX_GRAPH_JSON_BYTES,
    belief_external_reference,
    decode_domain_graph,
    encode_domain_graph,
    transition_external_reference,
    update_external_reference,
)
from .learning import CompoundModelState, WorldModelRuntime
from .model import SAMPLING_FORMAT, FixedScaling, TransitionBatch
from .planning import CEMController, make_learned_model_env
from .runtime_lane import CanonicalReplayIndex, RuntimeCustody, run_episode

TASK_A = "pendulum_normal_torque"
TASK_B = "pendulum_reversed_torque"
_TASKS = (TASK_A, TASK_B)
_SHA256_HEX = frozenset("0123456789abcdef")
_SAMPLING_MAGIC = b"PROSPECT-WM001\0"
_PLANNER_SCHEMA = "prospect.wm001.planner-rng.v1"
_PLANNER_STATE_SCHEMA = "prospect.wm001.cem-controller-state.v1"


@dataclass(frozen=True, slots=True)
class RestoredRuntime:
    """Fresh objects reconstructed from all checkpoint components."""

    runtime: WorldModelRuntime
    custody: RuntimeCustody
    state: AgentState
    snapshot: AgentSnapshot
    transitions: tuple[EpistemicTransition, ...]
    updates: tuple[UpdateReceipt, ...]
    collection_rngs: dict[str, np.random.Generator]
    next_tick: int
    planner: CEMController
    checkpoint_manifest_sha256: str
    component_hashes: dict[str, str]


def snapshot_planner_rng(planner: CEMController) -> bytes:
    """Safely encode the planner's exact owned RNG state as canonical JSON."""

    state = planner.state_dict()
    expected = {
        "schema",
        "seed",
        "actions_emitted",
        "rng_state",
        "budget",
        "device_type",
    }
    if set(state) != expected or state.get("schema") != _PLANNER_STATE_SCHEMA:
        raise ValueError("CEM controller returned an unsupported state mapping")
    rng_state = state["rng_state"]
    if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
        raise ValueError("CEM controller returned an invalid RNG tensor")
    raw = rng_state.detach().cpu().contiguous().numpy().tobytes()
    controller_state = {
        "actions_emitted": _require_nonnegative_int(
            state["actions_emitted"],
            label="planner actions_emitted",
        ),
        "budget": _validate_planner_budget(state["budget"]),
        "device_type": _require_nonempty_string(
            state["device_type"],
            label="planner device_type",
        ),
        "rng_state_base64": base64.b64encode(raw).decode("ascii"),
        "rng_state_bytes": len(raw),
        "rng_state_sha256": hashlib.sha256(raw).hexdigest(),
        "schema": _PLANNER_STATE_SCHEMA,
        "seed": _require_int(state["seed"], label="planner seed"),
    }
    return canonical_json_bytes(
        {
            "controller_state": controller_state,
            "schema": _PLANNER_SCHEMA,
        }
    )


def evaluate_live_state(
    *,
    runtime: WorldModelRuntime,
    custody: RuntimeCustody,
    next_tick: int,
    planner: CEMController,
    task_reset_seeds: Mapping[str, int],
    checkpoint_manifest_sha256: str,
    component_hashes: Mapping[str, str],
    boundary_snapshot: AgentSnapshot,
) -> dict[str, object]:
    """Continue directly from retained live state without reading a checkpoint."""

    _validate_task_reset_seeds(task_reset_seeds)
    _require_sha256(checkpoint_manifest_sha256, label="checkpoint manifest")
    hashes = _validate_component_hashes(component_hashes)
    return _evaluate_runtime(
        runtime=runtime,
        custody=custody,
        next_tick=_require_nonnegative_int(next_tick, label="next_tick"),
        planner=planner,
        task_reset_seeds=task_reset_seeds,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        component_hashes=hashes,
        boundary_snapshot=boundary_snapshot,
    )


def restore_checkpoint_state(
    checkpoint_path: str | Path,
    *,
    device: str,
) -> RestoredRuntime:
    """Verify and semantically reconstruct every sealed checkpoint component.

    A failed restore is process-RNG atomic, including failures during lazy model
    construction.  The checkpoint RNG states are applied only after every
    domain graph has been rehydrated and all non-global live components have
    re-encoded to their exact captured bytes.
    """

    entry_rngs = _snapshot_process_rngs()
    try:
        return _restore_checkpoint_state(checkpoint_path, device=device)
    except BaseException:
        _restore_process_rngs(entry_rngs)
        raise


def _restore_checkpoint_state(
    checkpoint_path: str | Path,
    *,
    device: str,
) -> RestoredRuntime:
    loaded = load_checkpoint(checkpoint_path)
    payloads = _capture_all_payloads(loaded)

    # Parameters and optimizer slots are jointly decoded and cross-bound.
    compound = CompoundModelState(
        model_bytes=payloads["world_model"],
        optimizer_bytes=payloads["optimizer"],
    )
    runtime = WorldModelRuntime.from_payload(compound.payload, device=device)
    # Force both lazy deserializers now; a malformed model or optimizer must
    # fail before any environment action.
    _ = runtime.model
    _ = runtime.optimizer_bytes
    _validate_component_metadata(loaded, runtime)

    model_ledger = _validate_model_ledger(payloads["model_version_ledger"], runtime)
    experience_rows, events, serialized_transitions = _validate_experience_store(payloads["experience_store"])

    store = InMemoryExperienceStore()
    ledger = EpistemicLedger(store)
    replay = CanonicalReplayIndex()
    for event in events:
        store.append(event)
    transitions: list[EpistemicTransition] = []
    for serialized in serialized_transitions:
        canonical = ledger.append_rehydrated_transition(serialized)
        replay.add(canonical.experience)
        transitions.append(canonical)

    replay_batches, replay_value = _validate_replay_index(
        payloads["replay_index"],
        experience_rows=experience_rows,
        events=events,
    )
    sampling = _validate_replay_sampling_history(
        payloads["replay_sampling_history"],
        replay_batches=replay_batches,
    )
    update_rows, serialized_updates = _validate_update_receipts(
        payloads["update_receipts"],
        runtime=runtime,
        transitions=transitions,
    )
    updates: list[UpdateReceipt] = []
    for serialized_update in serialized_updates:
        updates.append(ledger.append_rehydrated_update(serialized_update))
    _cross_validate_learning_state(
        ledger=model_ledger,
        sampling=sampling,
        updates=update_rows,
        replay_batches=replay_batches,
    )
    _cross_validate_domain_updates(update_rows, updates)

    runtime_state, snapshot = _validate_agent_runtime(
        payloads["agent_runtime"],
        runtime,
        updates=updates,
    )
    state = AgentState(snapshot)
    _validate_scaling_configuration(payloads["scaling_configuration"])
    collection_rngs = _validate_collection_rng(payloads["collection_rng"])
    planner = _restore_planner(
        payloads["planner_rng"],
        runtime=runtime,
        device=device,
    )

    namespace = cast(str, runtime_state["identity_namespace"])
    identity_payload = _decode_base64_field(
        runtime_state,
        "identity_checkpoint_base64",
    )
    identities = CounterIdentitySource(namespace)
    identities.restore_bytes(identity_payload)
    custody = RuntimeCustody(
        store=store,
        ledger=ledger,
        identities=identities,
        replay=replay,
    )

    # These components are re-encoded from the reconstructed live objects,
    # rather than trusted because their original JSON happened to parse.
    roundtrip_payloads = _reencode_non_global_components(
        payloads=payloads,
        runtime=runtime,
        model_ledger=model_ledger,
        experience_rows=experience_rows,
        events=events,
        transitions=transitions,
        replay_value=replay_value,
        replay_batches=replay_batches,
        sampling=sampling,
        update_rows=update_rows,
        updates=updates,
        runtime_state=runtime_state,
        snapshot=snapshot,
        collection_rngs=collection_rngs,
        planner=planner,
    )
    for component_id, reencoded in roundtrip_payloads.items():
        if reencoded != payloads[component_id]:
            raise ValueError(f"{component_id} live reconstruction does not exactly re-encode")

    # Decode all process-global RNGs without leaving a partial state, then
    # commit the captured states only after complete semantic preflight.
    _preflight_process_rngs(payloads)
    restore_python_rng(payloads["python_rng"])
    restore_numpy_rng(payloads["numpy_rng"])
    restore_torch_cpu_rng(payloads["torch_cpu_rng"])
    restore_torch_accelerator_rng(payloads["torch_accelerator_rng"])
    roundtrip_payloads.update(
        {
            "python_rng": snapshot_python_rng(),
            "numpy_rng": snapshot_numpy_rng(),
            "torch_cpu_rng": snapshot_torch_cpu_rng(),
            "torch_accelerator_rng": snapshot_torch_accelerator_rng(),
        }
    )
    if tuple(roundtrip_payloads) != CANONICAL_COMPONENT_IDS:
        # Reorder by the sealed component contract before digesting.
        roundtrip_payloads = {
            component_id: roundtrip_payloads[component_id] for component_id in CANONICAL_COMPONENT_IDS
        }
    for component_id in CANONICAL_COMPONENT_IDS:
        if roundtrip_payloads[component_id] != payloads[component_id]:
            raise ValueError(f"{component_id} restored state differs from captured bytes")
    component_hashes = {
        component_id: hashlib.sha256(roundtrip_payloads[component_id]).hexdigest()
        for component_id in CANONICAL_COMPONENT_IDS
    }
    manifest_hashes = {component.component_id: component.sha256 for component in loaded.report.components}
    if component_hashes != manifest_hashes:
        raise ValueError("live component hashes differ from the checkpoint manifest")

    return RestoredRuntime(
        runtime=runtime,
        custody=custody,
        state=state,
        snapshot=snapshot,
        transitions=tuple(transitions),
        updates=tuple(updates),
        collection_rngs=collection_rngs,
        next_tick=cast(int, runtime_state["next_tick"]),
        planner=planner,
        checkpoint_manifest_sha256=loaded.report.manifest_sha256,
        component_hashes=component_hashes,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    task_reset_seeds: Mapping[str, int],
    device: str,
) -> dict[str, object]:
    """Restore every component, then evaluate deterministic continuation."""

    restored = restore_checkpoint_state(checkpoint_path, device=device)
    return _evaluate_runtime(
        runtime=restored.runtime,
        custody=restored.custody,
        next_tick=restored.next_tick,
        planner=restored.planner,
        task_reset_seeds=task_reset_seeds,
        checkpoint_manifest_sha256=restored.checkpoint_manifest_sha256,
        component_hashes=restored.component_hashes,
        boundary_snapshot=restored.snapshot,
    )


def _evaluate_runtime(
    *,
    runtime: WorldModelRuntime,
    custody: RuntimeCustody,
    next_tick: int,
    planner: CEMController,
    task_reset_seeds: Mapping[str, int],
    checkpoint_manifest_sha256: str,
    component_hashes: Mapping[str, str],
    boundary_snapshot: AgentSnapshot,
) -> dict[str, object]:
    _validate_task_reset_seeds(task_reset_seeds)
    boundary_state = _boundary_state_row(boundary_snapshot, custody)
    task_rows: list[dict[str, object]] = []
    tick = next_tick
    for task_id in _TASKS:
        reset_seed = int(task_reset_seeds[task_id])
        episode, _ = run_episode(
            run_id=f"wm001-restart-parity:{task_id}",
            task_id=task_id,
            episode_id=f"wm001-restart-parity:{task_id}:{reset_seed}",
            reset_seed=reset_seed,
            controller=planner,
            backend=runtime,
            custody=custody,
            start_tick=tick,
        )
        tick = episode.final_tick + 1
        predictions: list[dict[str, object]] = []
        identities_rows: list[list[str]] = []
        for transition in episode.transitions:
            decision = transition.experience.decision
            if decision is None:
                raise ValueError("restart parity transition lacks its decision")
            parameters = cast(
                dict[str, object],
                decision.selected_assessment.prediction.distribution.parameters,
            )
            predictions.append(
                {
                    "member_means": parameters["member_means"],
                    "member_variances": parameters["member_variances"],
                }
            )
            identities_rows.append(
                [
                    decision.decision_id,
                    decision.selected_assessment.prediction.prediction_id,
                    transition.experience.experience_id,
                    transition.transition_id,
                ]
            )
        task_rows.append(
            {
                "task_id": task_id,
                "reset_seed": reset_seed,
                "return": episode.undiscounted_return,
                "actions": list(episode.intended_actions),
                "predictions": predictions,
                "identities": identities_rows,
            }
        )
    return {
        "schema": "prospect.wm001.restart-evaluation.v1",
        "process_id": os.getpid(),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "component_hashes": dict(component_hashes),
        "model_version": runtime.version,
        "parameter_sha256": runtime.digest,
        "boundary_state": boundary_state,
        "post_evaluation_custody": _custody_count_row(custody),
        "tasks": task_rows,
    }


def _boundary_state_row(
    snapshot: AgentSnapshot,
    custody: RuntimeCustody,
) -> dict[str, object]:
    latest = snapshot.latest_update
    if latest is None:
        raise ValueError("restart boundary snapshot lacks its latest update")
    if custody.ledger.get_update(latest.receipt_id) is not latest:
        raise ValueError("restart boundary does not reference its canonical update")
    if latest.resulting_belief is None or snapshot.belief is not latest.resulting_belief:
        raise ValueError("restart boundary does not reference its canonical belief")
    return {
        "snapshot_id": snapshot.snapshot_id,
        "agent_id": snapshot.agent_id,
        "captured_at": [snapshot.captured_at.clock_id, snapshot.captured_at.tick],
        "belief_id": snapshot.belief.belief_id,
        "latest_update_id": latest.receipt_id,
        "configuration_version": snapshot.configuration_version,
        "memory_version": snapshot.memory_version,
        "knowledge_version": snapshot.knowledge_version,
        "model_version": snapshot.model_version,
        "representation_version": snapshot.representation_version,
        "policy_version": snapshot.policy_version,
        "custody": _custody_count_row(custody),
    }


def _custody_count_row(custody: RuntimeCustody) -> dict[str, int]:
    return {
        "experiences": len(custody.store),
        "transitions": custody.ledger.transition_count,
        "updates": custody.ledger.update_count,
        "replay_events": custody.replay.event_count,
    }


def compare_parity(
    original: dict[str, object],
    restored: dict[str, object],
) -> dict[str, object]:
    """Return the exact K7 differences between two restart evaluations."""

    if original.get("schema") != restored.get("schema"):
        raise ValueError("restart evaluations use different schemas")
    for field in ("checkpoint_manifest_sha256", "model_version", "parameter_sha256"):
        if original.get(field) != restored.get(field):
            raise ValueError(f"restart evaluations differ on {field}")
    original_components = cast(dict[str, str], original["component_hashes"])
    restored_components = cast(dict[str, str], restored["component_hashes"])
    component_mismatches = sorted(
        component_id
        for component_id in CANONICAL_COMPONENT_IDS
        if original_components.get(component_id) != restored_components.get(component_id)
    )
    original_tasks = cast(list[dict[str, Any]], original["tasks"])
    restored_tasks = cast(list[dict[str, Any]], restored["tasks"])
    if len(original_tasks) != len(restored_tasks):
        raise ValueError("restart evaluations have different task counts")
    action_difference = 0.0
    prediction_difference = 0.0
    return_difference = 0.0
    identity_mismatches = 0
    if original.get("boundary_state") != restored.get("boundary_state"):
        identity_mismatches += 1
    original_post = original.get("post_evaluation_custody")
    restored_post = restored.get("post_evaluation_custody")
    if original_post != restored_post:
        identity_mismatches += 1
    for before, after in zip(original_tasks, restored_tasks, strict=True):
        if (before["task_id"], before["reset_seed"]) != (after["task_id"], after["reset_seed"]):
            raise ValueError("restart evaluations have different task/reset assignments")
        before_actions = np.asarray(before["actions"], dtype=np.float64)
        after_actions = np.asarray(after["actions"], dtype=np.float64)
        if before_actions.shape != after_actions.shape:
            raise ValueError("restart action traces have different shapes")
        action_difference = max(
            action_difference,
            float(np.max(np.abs(before_actions - after_actions), initial=0.0)),
        )
        before_predictions = _flatten_predictions(cast(list[dict[str, object]], before["predictions"]))
        after_predictions = _flatten_predictions(cast(list[dict[str, object]], after["predictions"]))
        if before_predictions.shape != after_predictions.shape:
            raise ValueError("restart prediction traces have different shapes")
        prediction_difference = max(
            prediction_difference,
            float(np.max(np.abs(before_predictions - after_predictions), initial=0.0)),
        )
        return_difference = max(
            return_difference,
            abs(float(before["return"]) - float(after["return"])),
        )
        before_ids = cast(list[list[str]], before["identities"])
        after_ids = cast(list[list[str]], after["identities"])
        if len(before_ids) != len(after_ids):
            identity_mismatches += abs(len(before_ids) - len(after_ids))
        identity_mismatches += sum(int(left != right) for left, right in zip(before_ids, after_ids, strict=False))
    original_pid = _require_nonnegative_int(
        original["process_id"],
        label="original process ID",
    )
    restored_pid = _require_nonnegative_int(
        restored["process_id"],
        label="restored process ID",
    )
    return {
        "checkpoint_manifest_sha256": str(original["checkpoint_manifest_sha256"]),
        "original_process_id": original_pid,
        "restored_process_id": restored_pid,
        "fresh_process": original_pid != restored_pid,
        "component_hash_mismatches": component_mismatches,
        "identity_or_lineage_mismatches": identity_mismatches,
        "prediction_max_abs_difference": prediction_difference,
        "action_max_abs_difference": action_difference,
        "episode_return_max_abs_difference": return_difference,
    }


def save_evaluation(path: str | Path, value: dict[str, object]) -> None:
    Path(path).write_bytes(canonical_json_bytes(value) + b"\n")


def load_evaluation(path: str | Path) -> dict[str, object]:
    payload = Path(path).read_bytes()
    if not payload.endswith(b"\n"):
        raise ValueError("restart evaluation file lacks its canonical newline")
    value = _load_canonical_json(payload[:-1], "restart evaluation")
    if not isinstance(value, dict):
        raise ValueError("restart evaluation must be an object")
    return cast(dict[str, object], value)


def _capture_all_payloads(loaded: LoadedWMCheckpoint) -> dict[str, bytes]:
    restored: dict[str, bytes] = {}

    def capture(component_id: str) -> Any:
        def callback(payload: bytes, _: object) -> None:
            restored[component_id] = payload

        return callback

    loaded.restore({component_id: capture(component_id) for component_id in CANONICAL_COMPONENT_IDS})
    if tuple(restored) != CANONICAL_COMPONENT_IDS:
        raise ValueError("checkpoint restore callback order changed")
    return restored


def _validate_model_ledger(
    payload: bytes,
    runtime: WorldModelRuntime,
) -> list[dict[str, Any]]:
    value = _load_canonical_json(payload, "model version ledger")
    if not isinstance(value, list) or not value:
        raise ValueError("model version ledger must be a nonempty array")
    expected = {
        "phase",
        "predecessor_parameter_sha256",
        "candidate_parameter_sha256",
        "predecessor_live_state_sha256",
        "candidate_live_state_sha256",
    }
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _require_exact_object(raw, expected, label=f"model ledger row {index}")
        _require_nonempty_string(row["phase"], label=f"model ledger row {index} phase")
        for field in expected - {"phase"}:
            _require_sha256(row[field], label=f"model ledger row {index} {field}")
        rows.append(row)
    if [row["phase"] for row in rows] != ["train_a", "train_b_replay"]:
        raise ValueError("model ledger must contain exactly the retained A then B updates")
    if rows[-1]["candidate_parameter_sha256"] != runtime.digest:
        raise ValueError("model ledger does not end at the restored parameter state")
    if rows[-1]["candidate_live_state_sha256"] != runtime.live_state_digest:
        raise ValueError("model ledger does not end at the restored compound state")
    for before, after in zip(rows, rows[1:], strict=False):
        if before["candidate_parameter_sha256"] != after["predecessor_parameter_sha256"]:
            raise ValueError("model ledger parameter ancestry is discontinuous")
        if before["candidate_live_state_sha256"] != after["predecessor_live_state_sha256"]:
            raise ValueError("model ledger compound-state ancestry is discontinuous")
    return rows


def _validate_component_metadata(
    loaded: LoadedWMCheckpoint,
    runtime: WorldModelRuntime,
) -> None:
    expected = {
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
    for record in loaded.report.components:
        logical_version, media_type = expected[record.component_id]
        if record.logical_version != logical_version or record.media_type != media_type:
            raise ValueError(f"{record.component_id} checkpoint metadata is incompatible")


def _validate_experience_store(
    payload: bytes,
) -> tuple[
    list[dict[str, Any]],
    tuple[ExperienceEvent, ...],
    tuple[EpistemicTransition, ...],
]:
    if len(payload) > MAX_GRAPH_JSON_BYTES:
        raise ValueError("experience store exceeds the domain-graph byte bound")
    component = _require_exact_object(
        _load_canonical_json(payload, "experience store"),
        {"schema", "transition_rows", "domain_graph"},
        label="experience store",
    )
    if component["schema"] != "prospect.wm001.experience-custody.v1":
        raise ValueError("experience store has an unsupported schema")
    raw_rows = component["transition_rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("experience store must retain at least one transition")
    required = {
        "transition_id",
        "run_id",
        "episode_id",
        "task_id",
        "task_context",
        "split",
        "step_index",
        "real_or_imagined",
        "pre_observation_id",
        "decision_id",
        "executed_action_id",
        "next_observation_id",
        "model_version_at_action",
        "parameter_sha256_at_action",
        "pre_observation",
        "intended_action",
        "applied_action",
        "next_observation",
        "reward",
        "terminated",
        "truncated",
        "scaled_target",
        "target_sha256",
    }
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, raw in enumerate(raw_rows):
        row = _require_exact_object(raw, required, label=f"experience transition {index}")
        transition_id = _require_nonempty_string(
            row["transition_id"],
            label=f"experience transition {index} ID",
        )
        if transition_id in identities:
            raise ValueError("experience store contains a duplicate transition ID")
        identities.add(transition_id)
        for field in (
            "run_id",
            "episode_id",
            "pre_observation_id",
            "decision_id",
            "executed_action_id",
            "next_observation_id",
            "model_version_at_action",
        ):
            _require_nonempty_string(row[field], label=f"experience transition {index} {field}")
        task_id = row["task_id"]
        split = row["split"]
        expected_split = "collect_a" if task_id == TASK_A else "collect_b"
        if task_id not in _TASKS or split != expected_split:
            raise ValueError("experience transition has an invalid task/split assignment")
        expected_context = 0.0 if task_id == TASK_A else 1.0
        if row["task_context"] != expected_context:
            raise ValueError("experience transition has an invalid task context")
        if row["real_or_imagined"] != "real":
            raise ValueError("experience store contains a non-real transition")
        step = _require_nonnegative_int(
            row["step_index"],
            label=f"experience transition {index} step",
        )
        if step >= 200:
            raise ValueError("experience transition step is outside the Pendulum horizon")
        for field in ("intended_action", "applied_action", "reward"):
            _require_finite_number(row[field], label=f"experience transition {index} {field}")
        intended = float(row["intended_action"])
        applied = float(row["applied_action"])
        if not -2.0 <= intended <= 2.0:
            raise ValueError("experience transition intended torque is out of bounds")
        expected_applied = intended if task_id == TASK_A else -intended
        if applied != expected_applied:
            raise ValueError("experience transition applied torque violates task semantics")
        before = row["pre_observation"]
        after = row["next_observation"]
        scaled_target = row["scaled_target"]
        if (
            not isinstance(before, list)
            or len(before) != 3
            or not isinstance(after, list)
            or len(after) != 3
            or not isinstance(scaled_target, list)
            or len(scaled_target) != 4
        ):
            raise ValueError("experience transition numeric vectors are malformed")
        for field, values in (
            ("pre observation", before),
            ("next observation", after),
            ("scaled target", scaled_target),
        ):
            for value in values:
                _require_finite_number(
                    value,
                    label=f"experience transition {index} {field}",
                )
        expected_scaled = np.asarray(
            [
                (float(after[0]) - float(before[0])) / 2.0,
                (float(after[1]) - float(before[1])) / 2.0,
                (float(after[2]) - float(before[2])) / 16.0,
                float(row["reward"]) / 16.2736044,
            ],
            dtype="<f8",
        )
        actual_scaled = np.asarray(scaled_target, dtype="<f8")
        if not np.array_equal(actual_scaled, expected_scaled):
            raise ValueError("experience transition scaled target differs from raw values")
        if not isinstance(row["terminated"], bool) or not isinstance(row["truncated"], bool):
            raise ValueError("experience transition terminal flags must be Boolean")
        _require_sha256(
            row["parameter_sha256_at_action"],
            label=f"experience transition {index} parameter digest",
        )
        target_digest = _require_sha256(
            row["target_sha256"],
            label=f"experience transition {index} target digest",
        )
        if hashlib.sha256(actual_scaled.tobytes(order="C")).hexdigest() != target_digest:
            raise ValueError("experience transition target digest differs")
        rows.append(row)
    roots = decode_domain_graph(component["domain_graph"])
    if set(roots) != {"events", "transitions"}:
        raise ValueError("experience domain graph has an invalid root set")
    raw_events = roots["events"]
    raw_transitions = roots["transitions"]
    if (
        not isinstance(raw_events, tuple)
        or not isinstance(raw_transitions, tuple)
        or any(type(event) is not ExperienceEvent for event in raw_events)
        or any(type(transition) is not EpistemicTransition for transition in raw_transitions)
    ):
        raise ValueError("experience domain graph roots use invalid record types")
    events = cast(tuple[ExperienceEvent, ...], raw_events)
    transitions = cast(tuple[EpistemicTransition, ...], raw_transitions)
    if len(events) != len(rows) or len(transitions) != len(rows):
        raise ValueError("experience domain graph record counts differ from custody rows")
    event_ids: set[str] = set()
    for index, (event, transition, row) in enumerate(zip(events, transitions, rows, strict=True)):
        if event.experience_id in event_ids:
            raise ValueError("experience domain graph contains a duplicate experience ID")
        event_ids.add(event.experience_id)
        if transition.experience is not event:
            raise ValueError("domain transition does not reference its canonical experience")
        if transition.transition_id != row["transition_id"]:
            raise ValueError(f"domain transition {index} differs from its custody row")
        if (
            event.run_id != row["run_id"]
            or event.episode_id != row["episode_id"]
            or event.task_id != row["task_id"]
            or event.step_index != row["step_index"]
            or event.terminated != row["terminated"]
            or event.truncated != row["truncated"]
        ):
            raise ValueError(f"domain experience {index} differs from its custody row")
        decision = event.decision
        execution = event.execution
        if decision is None or execution is None:
            raise ValueError("WM-001 domain experience lacks action lineage")
        if (
            decision.decision_id != row["decision_id"]
            or execution.execution_id != row["executed_action_id"]
            or event.observation.observation_id != row["next_observation_id"]
            or decision.belief.information_set.observations[-1].observation_id != row["pre_observation_id"]
        ):
            raise ValueError(f"domain experience {index} lineage differs from custody")
    return rows, events, transitions


def _validate_replay_index(
    payload: bytes,
    *,
    experience_rows: Sequence[dict[str, Any]],
    events: Sequence[ExperienceEvent],
) -> tuple[dict[str, TransitionBatch], dict[str, Any]]:
    value = _require_exact_object(
        _load_canonical_json(payload, "replay index"),
        {"schema", "canonical_experience_rows", "collect_a", "collect_b"},
        label="replay index",
    )
    if value["schema"] != "prospect.wm001.replay-index.v1":
        raise ValueError("replay index has an unsupported schema")
    _validate_canonical_replay_rows(
        value["canonical_experience_rows"],
        experience_rows=experience_rows,
        events=events,
    )
    result: dict[str, TransitionBatch] = {}
    for split, task_id, expected_context in (
        ("collect_a", TASK_A, 0.0),
        ("collect_b", TASK_B, 1.0),
    ):
        dataset = _require_exact_object(
            value[split],
            {
                "transition_ids",
                "observations",
                "contexts",
                "actions",
                "next_observations",
                "rewards",
            },
            label=f"replay index {split}",
        )
        observations = np.asarray(dataset["observations"], dtype=np.float64)
        contexts = np.asarray(dataset["contexts"], dtype=np.float64)
        actions = np.asarray(dataset["actions"], dtype=np.float64)
        next_observations = np.asarray(
            dataset["next_observations"],
            dtype=np.float64,
        )
        rewards = np.asarray(dataset["rewards"], dtype=np.float64)
        try:
            batch = TransitionBatch.from_arrays(
                transition_ids=dataset["transition_ids"],
                observations=observations,
                contexts=contexts,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"replay index {split} is malformed") from error
        expected_rows = [row for row in experience_rows if row["split"] == split]
        expected_ids = tuple(str(row["transition_id"]) for row in expected_rows)
        if batch.transition_ids != expected_ids:
            raise ValueError(f"replay index {split} IDs differ from experience custody")
        if not np.all(contexts == expected_context):
            raise ValueError(f"replay index {split} contains the wrong observed context")
        actions = actions.reshape(-1)
        rewards = rewards.reshape(-1)
        for index, row in enumerate(expected_rows):
            if float(actions[index]) != float(row["intended_action"]):
                raise ValueError(f"replay index {split} action differs from experience custody")
            if float(rewards[index]) != float(row["reward"]):
                raise ValueError(f"replay index {split} reward differs from experience custody")
        if task_id not in {str(row["task_id"]) for row in expected_rows}:
            raise ValueError(f"replay index {split} has no rows for its declared task")
        result[split] = batch
    return result, value


def _validate_canonical_replay_rows(
    value: object,
    *,
    experience_rows: Sequence[dict[str, Any]],
    events: Sequence[ExperienceEvent],
) -> None:
    if not isinstance(value, list) or len(value) != len(experience_rows) or len(events) != len(experience_rows):
        raise ValueError("canonical replay rows differ in count from experience custody")
    required = {
        "experience_id",
        "run_id",
        "task_id",
        "episode_id",
        "step_index",
        "closed_at",
    }
    experience_ids: set[str] = set()
    for index, (raw, transition, event) in enumerate(zip(value, experience_rows, events, strict=True)):
        row = _require_exact_object(
            raw,
            required,
            label=f"canonical replay row {index}",
        )
        experience_id = _require_nonempty_string(
            row["experience_id"],
            label=f"canonical replay row {index} experience ID",
        )
        if experience_id in experience_ids:
            raise ValueError("canonical replay index contains a duplicate experience ID")
        experience_ids.add(experience_id)
        if experience_id != event.experience_id:
            raise ValueError("canonical replay row names a different domain experience")
        _require_nonempty_string(
            row["run_id"],
            label=f"canonical replay row {index} run ID",
        )
        if (
            row["task_id"] != transition["task_id"]
            or row["episode_id"] != transition["episode_id"]
            or row["step_index"] != transition["step_index"]
            or row["run_id"] != event.run_id
        ):
            raise ValueError("canonical replay row differs from experience custody")
        closed_at = row["closed_at"]
        if (
            not isinstance(closed_at, list)
            or len(closed_at) != 2
            or not isinstance(closed_at[0], str)
            or not closed_at[0]
        ):
            raise ValueError("canonical replay row has malformed causal time")
        _require_nonnegative_int(
            closed_at[1],
            label=f"canonical replay row {index} closed tick",
        )


def _validate_replay_sampling_history(
    payload: bytes,
    *,
    replay_batches: Mapping[str, TransitionBatch],
) -> list[dict[str, Any]]:
    value = _require_exact_object(
        _load_canonical_json(payload, "replay sampling history"),
        {"schema", "manifests"},
        label="replay sampling history",
    )
    if value["schema"] != "prospect.wm001.replay-sampling-history.v1":
        raise ValueError("replay sampling history has an unsupported schema")
    manifests = value["manifests"]
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("replay sampling history must be nonempty")
    decoded: list[dict[str, Any]] = []
    for index, raw in enumerate(manifests):
        row = _require_exact_object(
            raw,
            {"phase", "sha256", "bytes", "payload_base64"},
            label=f"replay sampling manifest {index}",
        )
        phase = _require_nonempty_string(
            row["phase"],
            label=f"replay sampling manifest {index} phase",
        )
        encoded = _decode_base64_field(row, "payload_base64")
        if _require_nonnegative_int(
            row["bytes"],
            label=f"replay sampling manifest {index} bytes",
        ) != len(encoded):
            raise ValueError("replay sampling manifest byte length differs")
        digest = _require_sha256(
            row["sha256"],
            label=f"replay sampling manifest {index} digest",
        )
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise ValueError("replay sampling manifest digest differs")
        transition_ids = (
            replay_batches["collect_a"].transition_ids
            if phase == "train_a"
            else (
                *replay_batches["collect_a"].transition_ids,
                *replay_batches["collect_b"].transition_ids,
            )
        )
        if phase not in {"train_a", "train_b_replay"}:
            raise ValueError("checkpoint sampling history contains an unexpected phase")
        indices = _decode_sampling_manifest(encoded, transition_ids=transition_ids)
        decoded.append(
            {
                "phase": phase,
                "sha256": digest,
                "payload": encoded,
                "indices": indices,
                "transition_ids": transition_ids,
            }
        )
    return decoded


def _decode_sampling_manifest(
    payload: bytes,
    *,
    transition_ids: Sequence[str],
) -> np.ndarray:
    if not payload.startswith(_SAMPLING_MAGIC) or len(payload) < len(_SAMPLING_MAGIC) + 8:
        raise ValueError("replay sampling manifest has the wrong container magic")
    offset = len(_SAMPLING_MAGIC)
    (header_size,) = struct.unpack(">Q", payload[offset : offset + 8])
    offset += 8
    if header_size > 1 << 20 or len(payload) < offset + header_size:
        raise ValueError("replay sampling manifest header is truncated")
    header_payload = payload[offset : offset + header_size]
    offset += header_size
    header = _require_exact_object(
        _load_canonical_json(header_payload, "replay sampling manifest header"),
        {"format", "shape", "dtype", "transition_ids_sha256", "payload_sha256"},
        label="replay sampling manifest header",
    )
    if header["format"] != SAMPLING_FORMAT or header["dtype"] != "uint32-le":
        raise ValueError("replay sampling manifest has unsupported tensor metadata")
    shape = header["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape)
    ):
        raise ValueError("replay sampling manifest shape is invalid")
    raw = payload[offset:]
    if len(raw) != math.prod(shape) * 4:
        raise ValueError("replay sampling manifest payload size differs from its shape")
    if hashlib.sha256(raw).hexdigest() != _require_sha256(
        header["payload_sha256"],
        label="replay sampling manifest payload digest",
    ):
        raise ValueError("replay sampling manifest payload digest differs")
    identity_payload = json.dumps(
        list(transition_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(identity_payload).hexdigest() != _require_sha256(
        header["transition_ids_sha256"],
        label="replay sampling manifest identity digest",
    ):
        raise ValueError("replay sampling manifest is bound to different transition IDs")
    indices = np.frombuffer(raw, dtype="<u4").reshape(tuple(shape))
    if np.any(indices >= len(transition_ids)):
        raise ValueError("replay sampling manifest contains an out-of-range transition index")
    return indices.copy()


def _validate_update_receipts(
    payload: bytes,
    *,
    runtime: WorldModelRuntime,
    transitions: Sequence[EpistemicTransition],
) -> tuple[list[dict[str, Any]], tuple[UpdateReceipt, ...]]:
    if len(payload) > MAX_GRAPH_JSON_BYTES:
        raise ValueError("update receipts exceed the domain-graph byte bound")
    value = _require_exact_object(
        _load_canonical_json(payload, "update receipts"),
        {"schema", "updates", "domain_graph"},
        label="update receipts",
    )
    if value["schema"] != "prospect.wm001.update-receipts.v1":
        raise ValueError("update receipts have an unsupported schema")
    updates = value["updates"]
    if not isinstance(updates, list) or not updates:
        raise ValueError("update receipts must be nonempty")
    required = {
        "receipt_id",
        "phase",
        "status",
        "predecessor_parameter_sha256",
        "candidate_parameter_sha256",
        "committed_parameter_sha256",
        "predecessor_model_version",
        "committed_model_version",
        "eligible_splits",
        "eligible_transition_count",
        "eligible_transition_ids",
        "consumed_sample_count",
        "consumed_multiset_sha256",
        "sampling_manifest_sha256",
        "target_permutation_sha256",
        "target_permutation_file",
        "optimizer_steps",
        "live_state_before_sha256",
        "live_state_after_sha256",
    }
    rows: list[dict[str, Any]] = []
    phases: set[str] = set()
    for index, raw in enumerate(updates):
        row = _require_exact_object(raw, required, label=f"update receipt {index}")
        for field in ("receipt_id", "phase", "predecessor_model_version", "committed_model_version"):
            _require_nonempty_string(row[field], label=f"update receipt {index} {field}")
        phase = cast(str, row["phase"])
        if phase in phases:
            raise ValueError("update receipts contain a duplicate phase")
        phases.add(phase)
        if row["status"] not in {"committed", "rejected"}:
            raise ValueError("update receipt status is invalid")
        for field in (
            "predecessor_parameter_sha256",
            "candidate_parameter_sha256",
            "committed_parameter_sha256",
            "consumed_multiset_sha256",
            "live_state_before_sha256",
            "live_state_after_sha256",
        ):
            _require_sha256(row[field], label=f"update receipt {index} {field}")
        eligible = row["eligible_splits"]
        consumed = row["eligible_transition_ids"]
        if (
            not isinstance(eligible, list)
            or not eligible
            or any(split not in {"collect_a", "collect_b"} for split in eligible)
            or len(eligible) != len(set(eligible))
            or not isinstance(consumed, list)
            or any(not isinstance(identity, str) or not identity for identity in consumed)
            or len(consumed) != len(set(consumed))
        ):
            raise ValueError("update receipt eligible splits or consumed IDs are malformed")
        count = _require_nonnegative_int(
            row["eligible_transition_count"],
            label=f"update receipt {index} eligible count",
        )
        if count != len(consumed):
            raise ValueError("update receipt eligible transition count differs")
        steps = _require_nonnegative_int(
            row["optimizer_steps"],
            label=f"update receipt {index} optimizer steps",
        )
        sample_count = _require_nonnegative_int(
            row["consumed_sample_count"],
            label=f"update receipt {index} consumed sample count",
        )
        if row["status"] == "rejected" and (count != 0 or sample_count != 0 or steps != 0):
            raise ValueError("rejected update receipt reports consumed state")
        if row["status"] == "committed" and sample_count != steps * 5 * 256:
            raise ValueError("update receipt consumed sample count differs from its budget")
        sampling_digest = row["sampling_manifest_sha256"]
        if row["status"] == "committed":
            _require_sha256(
                sampling_digest,
                label=f"update receipt {index} sampling manifest digest",
            )
        elif sampling_digest is not None:
            raise ValueError("rejected update receipt names a sampling manifest")
        if row["target_permutation_sha256"] is not None or row["target_permutation_file"] is not None:
            raise ValueError("retained main-line update unexpectedly declares a target permutation")
        rows.append(row)
    replay = next((row for row in rows if row["phase"] == "train_b_replay"), None)
    if replay is None:
        raise ValueError("update receipts omit the retained train_b_replay update")
    if (
        replay["committed_parameter_sha256"] != runtime.digest
        or replay["live_state_after_sha256"] != runtime.live_state_digest
        or replay["committed_model_version"] != runtime.version
    ):
        raise ValueError("retained update receipt does not identify the restored runtime")
    external_transitions = {
        transition_external_reference(transition.transition_id): transition for transition in transitions
    }
    if len(external_transitions) != len(transitions):
        raise ValueError("canonical transitions contain duplicate stable references")
    roots = decode_domain_graph(
        value["domain_graph"],
        external_objects=external_transitions,
    )
    if set(roots) != {"receipts"}:
        raise ValueError("update domain graph has an invalid root set")
    raw_receipts = roots["receipts"]
    if not isinstance(raw_receipts, tuple) or any(type(receipt) is not UpdateReceipt for receipt in raw_receipts):
        raise ValueError("update domain graph root uses invalid record types")
    receipts = cast(tuple[UpdateReceipt, ...], raw_receipts)
    if len(receipts) != len(rows):
        raise ValueError("update domain graph count differs from receipt rows")
    for index, (row, receipt) in enumerate(zip(rows, receipts, strict=True)):
        if receipt.receipt_id != row["receipt_id"]:
            raise ValueError(f"domain update receipt {index} differs from its summary")
        if any(
            external_transitions.get(transition_external_reference(transition.transition_id)) is not transition
            for transition in receipt.transitions
        ):
            raise ValueError("domain update does not reference canonical transitions")
    return rows, receipts


def _cross_validate_learning_state(
    *,
    ledger: Sequence[dict[str, Any]],
    sampling: Sequence[dict[str, Any]],
    updates: Sequence[dict[str, Any]],
    replay_batches: Mapping[str, TransitionBatch],
) -> None:
    ledger_phases = [str(row["phase"]) for row in ledger]
    sampling_phases = [str(row["phase"]) for row in sampling]
    if ledger_phases != sampling_phases:
        raise ValueError("model ledger and replay sampling phase order differ")
    updates_by_phase = {str(row["phase"]): row for row in updates}
    if set(updates_by_phase) != set(ledger_phases):
        raise ValueError("update receipts differ from the retained model-ledger phases")
    all_replay_ids = {
        "collect_a": set(replay_batches["collect_a"].transition_ids),
        "collect_b": set(replay_batches["collect_b"].transition_ids),
    }
    for ledger_row, manifest in zip(ledger, sampling, strict=True):
        phase = str(ledger_row["phase"])
        update = updates_by_phase.get(phase)
        if update is None:
            raise ValueError(f"checkpoint omits the {phase} update receipt")
        for field in ("predecessor_parameter_sha256", "candidate_parameter_sha256"):
            if ledger_row[field] != update[field]:
                raise ValueError(f"{phase} model ledger and update receipt differ on {field}")
        if ledger_row["predecessor_live_state_sha256"] != update["live_state_before_sha256"]:
            raise ValueError(f"{phase} model ledger and update receipt predecessor states differ")
        if ledger_row["candidate_live_state_sha256"] != update["live_state_after_sha256"]:
            raise ValueError(f"{phase} model ledger and update receipt candidate states differ")
        indices = cast(np.ndarray, manifest["indices"])
        if int(indices.shape[0]) != int(update["optimizer_steps"]):
            raise ValueError(f"{phase} sampling history and optimizer-step count differ")
        if int(indices.size) != int(update["consumed_sample_count"]):
            raise ValueError(f"{phase} sampling history and consumed sample count differ")
        if manifest["sha256"] != update["sampling_manifest_sha256"]:
            raise ValueError(f"{phase} sampling manifest digest differs from its receipt")
        identities = cast(tuple[str, ...], manifest["transition_ids"])
        digest = hashlib.sha256()
        for index in indices.reshape(-1):
            digest.update(identities[int(index)].encode("utf-8") + b"\n")
        if digest.hexdigest() != update["consumed_multiset_sha256"]:
            raise ValueError(f"{phase} consumed multiset digest differs from sampling history")
        eligible_ids = set().union(*(all_replay_ids[split] for split in cast(list[str], update["eligible_splits"])))
        if set(cast(list[str], update["eligible_transition_ids"])) != eligible_ids:
            raise ValueError(f"{phase} eligible IDs differ from its replay set")


def _cross_validate_domain_updates(
    rows: Sequence[dict[str, Any]],
    receipts: Sequence[UpdateReceipt],
) -> None:
    if len(rows) != len(receipts):
        raise ValueError("domain update receipts differ in count from summaries")
    prior_completed: TimePoint | None = None
    for index, (row, receipt) in enumerate(zip(rows, receipts, strict=True)):
        expected_status = UpdateStatus.APPLIED if row["status"] == "committed" else UpdateStatus.REJECTED
        if receipt.status is not expected_status:
            raise ValueError(f"domain update receipt {index} has a different status")
        if (
            receipt.previous_model_version != row["predecessor_model_version"]
            or receipt.new_model_version != row["committed_model_version"]
            or receipt.new_configuration_version != f"wm001-config:{receipt.new_model_version}"
        ):
            raise ValueError(f"domain update receipt {index} has different versions")
        transition_ids = [transition.transition_id for transition in receipt.transitions]
        if transition_ids != row["eligible_transition_ids"]:
            raise ValueError(f"domain update receipt {index} transition order/content differs")
        if receipt.resulting_belief is None:
            raise ValueError("retained committed update lacks its resulting belief")
        if prior_completed is not None and (
            receipt.completed_at.clock_id != prior_completed.clock_id
            or receipt.completed_at.tick < prior_completed.tick
        ):
            raise ValueError("domain update receipt order regresses in causal time")
        prior_completed = receipt.completed_at


def _validate_agent_runtime(
    payload: bytes,
    runtime: WorldModelRuntime,
    *,
    updates: Sequence[UpdateReceipt],
) -> tuple[dict[str, Any], AgentSnapshot]:
    if len(payload) > MAX_GRAPH_JSON_BYTES:
        raise ValueError("agent runtime exceeds the domain-graph byte bound")
    expected = {
        "schema",
        "identity_namespace",
        "identity_checkpoint_base64",
        "next_tick",
        "agent_id",
        "configuration_version",
        "memory_version",
        "knowledge_version",
        "model_version",
        "representation_version",
        "policy_version",
        "belief_id",
        "domain_graph",
    }
    value = _require_exact_object(
        _load_canonical_json(payload, "agent runtime"),
        expected,
        label="agent runtime",
    )
    if value["schema"] != "prospect.wm001.agent-runtime.v1":
        raise ValueError("agent runtime has an unsupported schema")
    for field in expected - {
        "schema",
        "identity_checkpoint_base64",
        "next_tick",
        "domain_graph",
    }:
        _require_nonempty_string(value[field], label=f"agent runtime {field}")
    _require_nonnegative_int(value["next_tick"], label="agent runtime next_tick")
    if value["model_version"] != runtime.version:
        raise ValueError("agent runtime model version differs from restored state")
    if value["configuration_version"] != f"wm001-config:{runtime.version}":
        raise ValueError("agent runtime configuration does not bind the restored model")
    identities = CounterIdentitySource(cast(str, value["identity_namespace"]))
    identities.restore_bytes(_decode_base64_field(value, "identity_checkpoint_base64"))
    external: dict[str, object] = {}
    for receipt in updates:
        update_ref = update_external_reference(receipt.receipt_id)
        if update_ref in external:
            raise ValueError("domain updates contain duplicate stable references")
        external[update_ref] = receipt
        if receipt.resulting_belief is not None:
            belief_ref = belief_external_reference(receipt.resulting_belief.belief_id)
            if belief_ref in external:
                raise ValueError("domain updates contain duplicate belief references")
            external[belief_ref] = receipt.resulting_belief
    roots = decode_domain_graph(
        value["domain_graph"],
        external_objects=external,
    )
    if set(roots) != {"snapshot"} or type(roots["snapshot"]) is not AgentSnapshot:
        raise ValueError("agent runtime domain graph has an invalid snapshot root")
    snapshot = roots["snapshot"]
    if not updates or snapshot.latest_update is not updates[-1]:
        raise ValueError("agent snapshot does not reference the latest canonical update")
    if updates[-1].resulting_belief is None or snapshot.belief is not updates[-1].resulting_belief:
        raise ValueError("agent snapshot does not reference the canonical resulting belief")
    if snapshot.pending_intentions:
        raise ValueError("WM-001 checkpoint cannot retain orphan pending intentions")
    scalar_fields = (
        "agent_id",
        "configuration_version",
        "memory_version",
        "knowledge_version",
        "model_version",
        "representation_version",
        "policy_version",
    )
    if any(getattr(snapshot, field) != value[field] for field in scalar_fields):
        raise ValueError("agent snapshot differs from its runtime summary")
    if snapshot.belief.belief_id != value["belief_id"]:
        raise ValueError("agent snapshot belief differs from its runtime summary")
    if snapshot.captured_at.tick >= cast(int, value["next_tick"]):
        raise ValueError("agent snapshot is not before the next checkpoint tick")
    return value, snapshot


def _reencode_non_global_components(
    *,
    payloads: Mapping[str, bytes],
    runtime: WorldModelRuntime,
    model_ledger: Sequence[dict[str, Any]],
    experience_rows: Sequence[dict[str, Any]],
    events: Sequence[ExperienceEvent],
    transitions: Sequence[EpistemicTransition],
    replay_value: Mapping[str, Any],
    replay_batches: Mapping[str, TransitionBatch],
    sampling: Sequence[dict[str, Any]],
    update_rows: Sequence[dict[str, Any]],
    updates: Sequence[UpdateReceipt],
    runtime_state: Mapping[str, Any],
    snapshot: AgentSnapshot,
    collection_rngs: Mapping[str, np.random.Generator],
    planner: CEMController,
) -> dict[str, bytes]:
    del payloads, replay_batches
    experience_value = {
        "schema": "prospect.wm001.experience-custody.v1",
        "transition_rows": list(experience_rows),
        "domain_graph": encode_domain_graph(
            {
                "events": tuple(events),
                "transitions": tuple(transitions),
            }
        ),
    }
    transition_references = {
        id(transition): transition_external_reference(transition.transition_id) for transition in transitions
    }
    update_value = {
        "schema": "prospect.wm001.update-receipts.v1",
        "updates": list(update_rows),
        "domain_graph": encode_domain_graph(
            {"receipts": tuple(updates)},
            external_references=transition_references,
        ),
    }
    agent_value = {key: value for key, value in runtime_state.items() if key != "domain_graph"}
    if snapshot.latest_update is None:
        raise ValueError("agent snapshot lost its latest update")
    agent_value["domain_graph"] = encode_domain_graph(
        {"snapshot": snapshot},
        external_references={
            id(snapshot.latest_update): update_external_reference(snapshot.latest_update.receipt_id),
            id(snapshot.belief): belief_external_reference(snapshot.belief.belief_id),
        },
    )
    replay_reencoded = {
        "schema": "prospect.wm001.replay-index.v1",
        "canonical_experience_rows": [
            {
                "experience_id": event.experience_id,
                "run_id": event.run_id,
                "task_id": event.task_id,
                "episode_id": event.episode_id,
                "step_index": event.step_index,
                "closed_at": [event.closed_at.clock_id, event.closed_at.tick],
            }
            for event in events
        ],
        "collect_a": _domain_transition_dataset(
            transition
            for row, transition in zip(experience_rows, transitions, strict=True)
            if row["split"] == "collect_a"
        ),
        "collect_b": _domain_transition_dataset(
            transition
            for row, transition in zip(experience_rows, transitions, strict=True)
            if row["split"] == "collect_b"
        ),
    }
    if replay_reencoded["schema"] != replay_value["schema"]:
        raise ValueError("reconstructed replay schema differs")
    sampling_value = {
        "schema": "prospect.wm001.replay-sampling-history.v1",
        "manifests": [
            {
                "phase": row["phase"],
                "sha256": row["sha256"],
                "bytes": len(cast(bytes, row["payload"])),
                "payload_base64": base64.b64encode(cast(bytes, row["payload"])).decode("ascii"),
            }
            for row in sampling
        ],
    }
    scaling = {
        "schema": "prospect.wm001.fixed-scaling.v1",
        **asdict(FixedScaling()),
    }
    collection_rng = {
        "schema": "prospect.wm001.collection-rng.v1",
        "states": {task: collection_rngs[task].bit_generator.state for task in ("task_a", "task_b")},
    }
    return {
        "world_model": runtime.model_bytes,
        "optimizer": runtime.optimizer_bytes,
        "model_version_ledger": canonical_json_bytes(list(model_ledger)),
        "experience_store": canonical_json_bytes(experience_value),
        "replay_index": canonical_json_bytes(replay_reencoded),
        "replay_sampling_history": canonical_json_bytes(sampling_value),
        "update_receipts": canonical_json_bytes(update_value),
        "agent_runtime": canonical_json_bytes(agent_value),
        "scaling_configuration": canonical_json_bytes(scaling),
        "collection_rng": canonical_json_bytes(collection_rng),
        "planner_rng": snapshot_planner_rng(planner),
    }


def _domain_transition_dataset(
    transitions: Iterable[EpistemicTransition],
) -> dict[str, object]:
    transition_rows = tuple(transitions)
    observations: list[list[float]] = []
    contexts: list[float] = []
    actions: list[float] = []
    next_observations: list[list[float]] = []
    rewards: list[float] = []
    identities: list[str] = []
    for transition in transition_rows:
        decision = transition.experience.decision
        if decision is None:
            raise ValueError("replay transition lacks its decision")
        belief_parameters = cast(
            dict[str, object],
            decision.belief.distribution.parameters,
        )
        action_parameters = cast(
            dict[str, object],
            decision.intended_action.action.parameters,
        )
        observation_payload = cast(
            dict[str, object],
            transition.experience.observation.evidence.payload,
        )
        outcome_payload = cast(
            dict[str, object],
            transition.experience.outcome.evidence.payload,
        )
        observations.append(
            [
                float(value)
                for value in cast(
                    Sequence[float],
                    belief_parameters["physical_observation"],
                )
            ]
        )
        contexts.append(
            _require_finite_number(
                action_parameters["task_context"],
                label="domain replay task context",
            )
        )
        actions.append(
            _require_finite_number(
                action_parameters["intended_torque"],
                label="domain replay intended torque",
            )
        )
        next_observations.append(
            [
                float(value)
                for value in cast(
                    Sequence[float],
                    observation_payload["physical_observation"],
                )
            ]
        )
        rewards.append(
            _require_finite_number(
                outcome_payload["reward"],
                label="domain replay reward",
            )
        )
        identities.append(transition.transition_id)
    return {
        "transition_ids": identities,
        "observations": observations,
        "contexts": contexts,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
    }


def _validate_scaling_configuration(payload: bytes) -> None:
    value = _load_canonical_json(payload, "scaling configuration")
    expected = json.loads(
        canonical_json_bytes(
            {
                "schema": "prospect.wm001.fixed-scaling.v1",
                **asdict(FixedScaling()),
            }
        )
    )
    if value != expected:
        raise ValueError("scaling configuration differs from the sealed fixed scaling")


def _validate_collection_rng(payload: bytes) -> dict[str, np.random.Generator]:
    value = _require_exact_object(
        _load_canonical_json(payload, "collection RNG"),
        {"schema", "states"},
        label="collection RNG",
    )
    if value["schema"] != "prospect.wm001.collection-rng.v1":
        raise ValueError("collection RNG has an unsupported schema")
    states = _require_exact_object(
        value["states"],
        {"task_a", "task_b"},
        label="collection RNG states",
    )
    restored: dict[str, np.random.Generator] = {}
    for task in ("task_a", "task_b"):
        generator = np.random.default_rng()
        try:
            generator.bit_generator.state = states[task]
        except (TypeError, ValueError) as error:
            raise ValueError(f"collection RNG {task} state is invalid") from error
        restored[task] = generator
    return restored


def _restore_planner(
    payload: bytes,
    *,
    runtime: WorldModelRuntime,
    device: str,
) -> CEMController:
    value = _require_exact_object(
        _load_canonical_json(payload, "planner RNG"),
        {"schema", "controller_state"},
        label="planner RNG",
    )
    if value["schema"] != _PLANNER_SCHEMA:
        raise ValueError("planner RNG has an unsupported schema")
    state = _require_exact_object(
        value["controller_state"],
        {
            "schema",
            "seed",
            "actions_emitted",
            "rng_state_base64",
            "rng_state_bytes",
            "rng_state_sha256",
            "budget",
            "device_type",
        },
        label="planner controller state",
    )
    if state["schema"] != _PLANNER_STATE_SCHEMA:
        raise ValueError("planner controller state has an unsupported schema")
    raw = _decode_base64_field(state, "rng_state_base64")
    if _require_nonnegative_int(
        state["rng_state_bytes"],
        label="planner RNG byte length",
    ) != len(raw):
        raise ValueError("planner RNG byte length differs")
    if hashlib.sha256(raw).hexdigest() != _require_sha256(
        state["rng_state_sha256"],
        label="planner RNG digest",
    ):
        raise ValueError("planner RNG digest differs")
    if not raw:
        raise ValueError("planner RNG state is empty")
    seed = _require_int(state["seed"], label="planner seed")
    actions_emitted = _require_nonnegative_int(
        state["actions_emitted"],
        label="planner actions emitted",
    )
    budget = _validate_planner_budget(state["budget"])
    device_type = _require_nonempty_string(
        state["device_type"],
        label="planner device type",
    )
    planner = CEMController(
        make_learned_model_env(runtime.model, device=device),
        seed=0,
    )
    planner.load_state_dict(
        {
            "schema": _PLANNER_STATE_SCHEMA,
            "seed": seed,
            "actions_emitted": actions_emitted,
            "rng_state": torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()),
            "budget": budget,
            "device_type": device_type,
        }
    )
    return planner


def _validate_planner_budget(value: object) -> dict[str, int]:
    budget = _require_exact_object(
        value,
        {"planning_horizon", "num_candidates", "top_k", "optim_steps"},
        label="planner budget",
    )
    result: dict[str, int] = {}
    for field, raw in budget.items():
        parsed = _require_nonnegative_int(raw, label=f"planner budget {field}")
        if parsed < 1:
            raise ValueError(f"planner budget {field} must be positive")
        result[field] = parsed
    if result["top_k"] > result["num_candidates"]:
        raise ValueError("planner top_k exceeds its candidate count")
    return result


def _preflight_process_rngs(payloads: Mapping[str, bytes]) -> None:
    previous = _snapshot_process_rngs()
    try:
        restore_python_rng(payloads["python_rng"], random.Random())
        restore_numpy_rng(payloads["numpy_rng"])
        restore_numpy_rng(previous["numpy_rng"])
        restore_torch_cpu_rng(payloads["torch_cpu_rng"])
        restore_torch_cpu_rng(previous["torch_cpu_rng"])
        restore_torch_accelerator_rng(payloads["torch_accelerator_rng"])
        restore_torch_accelerator_rng(previous["torch_accelerator_rng"])
    except BaseException:
        _restore_process_rngs(previous)
        raise


def _snapshot_process_rngs() -> dict[str, bytes]:
    return {
        "python_rng": snapshot_python_rng(),
        "numpy_rng": snapshot_numpy_rng(),
        "torch_cpu_rng": snapshot_torch_cpu_rng(),
        "torch_accelerator_rng": snapshot_torch_accelerator_rng(),
    }


def _restore_process_rngs(payloads: Mapping[str, bytes]) -> None:
    restore_python_rng(payloads["python_rng"])
    restore_numpy_rng(payloads["numpy_rng"])
    restore_torch_cpu_rng(payloads["torch_cpu_rng"])
    restore_torch_accelerator_rng(payloads["torch_accelerator_rng"])


def _validate_task_reset_seeds(task_reset_seeds: Mapping[str, int]) -> None:
    if set(task_reset_seeds) != set(_TASKS):
        raise ValueError("restart evaluation requires exactly both Pendulum task reset seeds")
    for task_id in _TASKS:
        _require_nonnegative_int(
            task_reset_seeds[task_id],
            label=f"{task_id} reset seed",
        )


def _validate_component_hashes(values: Mapping[str, str]) -> dict[str, str]:
    if tuple(values) != CANONICAL_COMPONENT_IDS:
        raise ValueError("component hashes differ from the sealed component order")
    return {
        component_id: _require_sha256(
            values[component_id],
            label=f"{component_id} component digest",
        )
        for component_id in CANONICAL_COMPONENT_IDS
    }


def _load_canonical_json(payload: bytes, label: str) -> Any:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_exact_object(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value) or set(value) != expected:
        raise ValueError(f"{label} has an invalid field set")
    return value


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    parsed = _require_int(value, label=label)
    if parsed < 0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _require_finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _decode_base64_field(value: Mapping[str, Any], field: str) -> bytes:
    encoded = value.get(field)
    if not isinstance(encoded, str):
        raise ValueError(f"{field} is not base64 text")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError(f"{field} is not valid base64") from error


def _flatten_predictions(rows: list[dict[str, object]]) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in rows:
        values.append(np.asarray(row["member_means"], dtype=np.float64).ravel())
        values.append(np.asarray(row["member_variances"], dtype=np.float64).ravel())
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


__all__ = (
    "RestoredRuntime",
    "compare_parity",
    "evaluate_checkpoint",
    "evaluate_live_state",
    "load_evaluation",
    "restore_checkpoint_state",
    "save_evaluation",
    "snapshot_planner_rng",
)
