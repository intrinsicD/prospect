"""End-to-end WM-001 experiment harness.

This module executes the sealed causal sequence.  Formal configuration is
accepted only at the exact protocol budgets; the v1.19 development rehearsal
uses the same budgets but remains permanently claim-ineligible.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import platform
import random
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from prospect.domain import AgentSnapshot, EpistemicTransition, TimePoint, UpdateReceipt

from .artifact import atomic_write_exclusive
from .assurance import ASSURANCE
from .checkpoint import (
    CANONICAL_COMPONENT_IDS,
    ComponentPayload,
    canonical_json_bytes,
    save_checkpoint,
    snapshot_numpy_rng,
    snapshot_python_rng,
    snapshot_torch_accelerator_rng,
    snapshot_torch_cpu_rng,
)
from .domain_graph import (
    belief_external_reference,
    encode_domain_graph,
    transition_external_reference,
    update_external_reference,
)
from .learning import (
    LearningEvidence,
    RejectedUpdateProbeLearner,
    TransactionalWorldModelLearner,
    WorldModelRuntime,
)
from .model import (
    FixedScaling,
    PredictiveMetrics,
    TransitionBatch,
    evaluate_mixture,
)
from .parity import (
    compare_parity,
    evaluate_live_state,
    load_evaluation,
    snapshot_planner_rng,
)
from .planning import (
    CEMController,
    analytic_pendulum_step,
    make_learned_model_env,
    make_true_dynamics_env,
    run_pendulum_conformance,
)
from .runtime_lane import (
    AGENT_ID,
    INDEPENDENT_OSCILLATOR_TASK,
    CollectionEvidence,
    EpisodeEvidence,
    PendulumEpisodeSession,
    PredictiveBackend,
    PresetActionController,
    RuntimeCustody,
    UniformRandomController,
    collect_episodes,
    make_branch_learning_agent,
    run_episode,
    run_independent_phase_oscillator_conformance,
    transition_arrays,
    transition_lineage_row,
)
from .verify import (
    DEVELOPMENT_SEEDS,
    FORMAL_SEEDS,
    PROTOCOL_PATH,
    derive_seed,
)

TASK_A = "pendulum_normal_torque"
TASK_B = "pendulum_reversed_torque"
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
T_CRITICAL_DF7 = 2.364624251
_RESTART_TIMEOUT_SECONDS = 600
_RESTART_LOG_LIMIT_BYTES = 16 << 20


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _runtime_custody_value(payload: bytes) -> tuple[dict[str, object], int]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "captured runtime seal is no longer valid JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or payload != canonical_json_bytes(value) + b"\n"
        or value.get("experiment_id") != "WM-001"
        or value.get("assurance") != ASSURANCE
    ):
        raise RuntimeError(
            "captured runtime seal is not one canonical assured object"
        )
    schema = value.get("schema")
    if schema == "prospect.wm001.runtime-seal.v1":
        return cast(dict[str, object], value), 2
    if schema == "prospect.world-model-lifecycle.formal-binding.v10":
        return cast(dict[str, object], value), 1
    raise RuntimeError("captured runtime seal has an unsupported schema")


def _captured_bootstrap_custody() -> dict[str, object]:
    """Reopen the pre-import bootstrap and seal descriptors without path lookup."""

    fields = {
        "runtime_seal": (
            "_prospect_wm001_runtime_seal_fd",
            "_prospect_wm001_runtime_seal_payload",
            "_prospect_wm001_runtime_seal_identity",
            "_prospect_wm001_runtime_seal_sha256",
        ),
        "bootstrap": (
            "_prospect_wm001_bootstrap_fd",
            "_prospect_wm001_bootstrap_payload",
            "_prospect_wm001_bootstrap_identity",
            "_prospect_wm001_bootstrap_sha256",
        ),
    }
    result: dict[str, object] = {}
    runtime_seal: dict[str, object] | None = None
    for label, (fd_name, payload_name, identity_name, digest_name) in fields.items():
        descriptor = getattr(sys, fd_name, None)
        expected_payload = getattr(sys, payload_name, None)
        expected_identity = getattr(sys, identity_name, None)
        expected_digest = getattr(sys, digest_name, None)
        if (
            type(descriptor) is not int
            or not isinstance(expected_payload, bytes)
            or not isinstance(expected_identity, tuple)
            or len(expected_identity) != 9
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            raise RuntimeError("WM-001 must be entered through the sealed producer bootstrap")
        expected_nlink = 1
        if label == "runtime_seal":
            runtime_seal, expected_nlink = _runtime_custody_value(
                expected_payload
            )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != expected_nlink
            ):
                raise RuntimeError(
                    f"captured {label} does not have its typed link count"
                )
            payload = os.pread(descriptor, before.st_size + 1, 0)
            after = os.fstat(descriptor)
        except OSError as error:
            raise RuntimeError(f"captured {label} descriptor cannot be read") from error
        if (
            len(payload) != before.st_size
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != expected_identity
            or payload != expected_payload
            or hashlib.sha256(payload).hexdigest() != expected_digest
        ):
            raise RuntimeError(f"captured {label} changed after bootstrap verification")
        result[f"{label}_fd"] = descriptor
        result[f"{label}_payload"] = payload
        result[f"{label}_sha256"] = expected_digest
    assert runtime_seal is not None
    result["runtime_seal"] = runtime_seal
    return result


def _descriptor_path(descriptor: int) -> str:
    for prefix in ("/proc/self/fd/", "/dev/fd/"):
        candidate = f"{prefix}{descriptor}"
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError("platform has no inherited-descriptor execution path")


def _verify_live_bootstrap_custody() -> dict[str, object]:
    """Recompute the sealed closure before producer completion is finalized."""

    from .binding import (
        package_root_inventory,
        package_root_ownership,
        package_roots,
        python_flag_identity,
        standard_library_inventory,
    )

    custody = _captured_bootstrap_custody()
    seal = cast(dict[str, object], custody["runtime_seal"])
    if seal.get("schema") == "prospect.wm001.runtime-seal.v1":
        source = seal
        dependencies = {
            "python_executable": cast(dict[str, object], seal.get("python", {})).get("executable"),
            "python_executable_sha256": cast(
                dict[str, object],
                seal.get("python", {}),
            ).get("sha256"),
            "package_roots": seal.get("package_roots"),
            "package_ownership": seal.get("package_ownership"),
            "standard_library": seal.get("standard_library"),
        }
        runtime = {
            "python_flags": seal.get("required_flags"),
            "process_environment": seal.get("process_environment"),
        }
        expected_bootstrap = seal.get("bootstrap_source_sha256")
    elif seal.get("schema") == "prospect.world-model-lifecycle.formal-binding.v10":
        source = cast(dict[str, object], seal.get("source", {}))
        dependencies = cast(dict[str, object], seal.get("dependencies", {}))
        runtime = cast(dict[str, object], seal.get("runtime", {}))
        execution_sources = cast(
            dict[str, object],
            source.get("execution_source_sha256", {}),
        )
        expected_bootstrap = execution_sources.get("producer_bootstrap.py")
    else:
        raise RuntimeError("captured runtime seal has an unsupported schema")
    roots = package_roots()
    current_root_inventories = [package_root_inventory(root) for root in roots]
    current_standard_library = standard_library_inventory()
    executable = Path(sys.executable).resolve(strict=True)
    if (
        source.get("git_commit") != _git_value("rev-parse", "HEAD")
        or source.get("git_tree") != _git_value("rev-parse", "HEAD^{tree}")
        or source.get("worktree_clean") is not True
        or _git_value("status", "--short", "--untracked-files=all")
        or dependencies.get("python_executable") != sys.executable
        or dependencies.get("python_executable_sha256") != hashlib.sha256(executable.read_bytes()).hexdigest()
        or dependencies.get("package_roots") != current_root_inventories
        or dependencies.get("package_ownership") != package_root_ownership()
        or dependencies.get("standard_library") != current_standard_library
        or runtime.get("python_flags") != python_flag_identity()
        or runtime.get("process_environment") != dict(sorted(os.environ.items()))
        or expected_bootstrap != custody["bootstrap_sha256"]
    ):
        raise RuntimeError("live runtime closure differs from its pre-import bootstrap seal")
    return custody


def _recheck_live_bootstrap_custody(
    expected: dict[str, object],
) -> None:
    """Require exact custody again after lazy runtime initialization."""

    if _verify_live_bootstrap_custody() != expected:
        raise RuntimeError(
            "live runtime custody identity changed after conformance"
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Lane-specific execution budgets."""

    lane: str
    master_seeds: tuple[int, ...]
    collection_episodes: int
    validation_episodes: int
    behavior_episodes: int
    optimizer_steps: int
    device: str

    @classmethod
    def development(
        cls,
        *,
        master_seeds: Sequence[int] = DEVELOPMENT_SEEDS,
        device: str | None = None,
    ) -> ExperimentConfig:
        return cls(
            lane="development",
            master_seeds=tuple(master_seeds),
            collection_episodes=8,
            validation_episodes=8,
            behavior_episodes=32,
            optimizer_steps=2000,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

    @classmethod
    def formal(cls, *, device: str | None = None) -> ExperimentConfig:
        return cls(
            lane="formal",
            master_seeds=FORMAL_SEEDS,
            collection_episodes=8,
            validation_episodes=8,
            behavior_episodes=32,
            optimizer_steps=2000,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def validate(self) -> None:
        if self.lane not in {"development", "formal"}:
            raise ValueError("lane must be development or formal")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("unsupported execution device")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if (
            min(
                self.collection_episodes,
                self.validation_episodes,
                self.behavior_episodes,
                self.optimizer_steps,
            )
            < 1
        ):
            raise ValueError("all execution budgets must be positive")
        if self.validation_episodes != 8:
            raise ValueError("raw predictive evidence requires the sealed 1,600-transition validation split")
        if self.lane == "formal":
            expected = self.formal(device=self.device)
            if self != expected:
                raise ValueError("formal WM-001 budgets or seeds differ from the sealed protocol")
        elif not set(self.master_seeds) <= set(DEVELOPMENT_SEEDS):
            raise ValueError("development uses an undeclared master seed")


@dataclass(frozen=True, slots=True)
class IsolatedConformanceReports:
    """Reports reproduced by the bound path/descriptor audit-runner modes."""

    pendulum_conformance: dict[str, Any]
    oscillator_conformance: dict[str, Any]
    coverage_conformance: dict[str, Any]
    runner_verification: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _FormalPreflightReports:
    """Fully verified outcome-free checks completed before the formal claim."""

    binding_sha256: str
    live_binding: dict[str, object]
    isolated_conformance: IsolatedConformanceReports


def _run_formal_preflight(
    binding_path: Path,
    *,
    device: str,
) -> _FormalPreflightReports:
    """Run and validate every outcome-free check before launch custody is consumed."""

    from .binding import run_bound_preflight_conformance, verify_live_binding

    live_binding = verify_live_binding(binding_path, device=device)
    isolated_conformance = cast(
        IsolatedConformanceReports,
        run_bound_preflight_conformance(binding_path, device),
    )
    reports = _FormalPreflightReports(
        binding_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        live_binding=dict(live_binding),
        isolated_conformance=isolated_conformance,
    )
    _validate_formal_preflight_reports(reports, binding_path=binding_path)
    return reports


def _validate_formal_preflight_reports(
    reports: _FormalPreflightReports,
    *,
    binding_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate report identity without repeating any live or environment check."""

    try:
        copied_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal preflight binding is unreadable") from error
    if not isinstance(copied_binding, dict) or reports.live_binding != copied_binding:
        raise ValueError("formal preflight live-binding report differs from the copied binding")
    if reports.binding_sha256 != hashlib.sha256(binding_path.read_bytes()).hexdigest():
        raise ValueError("formal preflight binding digest differs from the copied binding")
    isolated = reports.isolated_conformance
    if isolated.pendulum_conformance.get("passed") is not True:
        raise ValueError("formal preflight Pendulum report did not pass")
    if isolated.oscillator_conformance.get("passed") is not True:
        raise ValueError("formal preflight oscillator report did not pass")
    if isolated.coverage_conformance.get("passed") is not True:
        raise ValueError("formal preflight coverage report did not pass")
    verification = isolated.runner_verification
    if (
        verification.get("passed") is not True
        or verification.get("source_mode") != "descriptor"
        or verification.get("single_launch_replay") is not True
        or verification.get("matches_bound_prebinding_reports") is not True
    ):
        raise ValueError("formal preflight isolated runner verification did not pass")
    return isolated.pendulum_conformance, isolated.oscillator_conformance


class OraclePredictiveBackend(PredictiveBackend):
    """Exact separately namespaced prediction backend for the oracle control."""

    version = "wm001-analytic-pendulum-v1"
    digest = hashlib.sha256(b"WM-001|Gymnasium-Pendulum-v1|analytic-oracle-v1").hexdigest()

    def predict_ensemble(
        self,
        observation: np.ndarray,
        context: float,
        action: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        next_observation, _, _ = analytic_pendulum_step(observation, context, [action])
        values = next_observation.detach().cpu().numpy().astype(np.float64)
        means = np.repeat(values[None, :], 5, axis=0)
        variances = np.full_like(means, 1e-12)
        return means, variances


@dataclass(slots=True)
class ReplicateRows:
    """Mutable raw-evidence accumulator for one replicate."""

    replicate_id: str
    master_seed: int
    derived_seeds: list[dict[str, object]]
    episodes: list[dict[str, object]]
    transitions: list[dict[str, object]]
    updates: list[dict[str, object]]
    optimizer_batch_manifests: list[dict[str, object]]
    predictive_metrics: list[dict[str, object]]
    policy_runs: list[dict[str, object]]
    evaluated_checkpoints: list[dict[str, object]]
    checkpoint_components: list[dict[str, object]]
    checkpoint_archive: dict[str, object] | None = None
    restart_parity: dict[str, object] | None = None

    @classmethod
    def create(cls, replicate_id: str, master_seed: int) -> ReplicateRows:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        declarations = protocol["seed_schedule"]["namespaces"]
        derived = [
            {
                "namespace": namespace,
                "values": [derive_seed(namespace, master_seed, index) for index in range(int(declaration["count"]))],
            }
            for namespace, declaration in declarations.items()
        ]
        return cls(
            replicate_id=replicate_id,
            master_seed=master_seed,
            derived_seeds=derived,
            episodes=[],
            transitions=[],
            updates=[],
            optimizer_batch_manifests=[],
            predictive_metrics=[],
            policy_runs=[],
            evaluated_checkpoints=[],
            checkpoint_components=[],
        )

    def seed(self, namespace: str, index: int = 0) -> int:
        for row in self.derived_seeds:
            if row["namespace"] == namespace:
                return int(cast(list[int], row["values"])[index])
        raise KeyError(namespace)

    def as_dict(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "master_seed": self.master_seed,
            "derived_seeds": self.derived_seeds,
            "episodes": self.episodes,
            "transitions": self.transitions,
            "updates": self.updates,
            "optimizer_batch_manifests": self.optimizer_batch_manifests,
            "predictive_metrics": self.predictive_metrics,
            "policy_runs": self.policy_runs,
            "evaluated_checkpoints": self.evaluated_checkpoints,
            "checkpoint_components": self.checkpoint_components,
            "checkpoint_archive": self.checkpoint_archive,
            "restart_parity": self.restart_parity,
        }


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def append_policy_run(
    rows: ReplicateRows,
    *,
    run_id: str,
    task_id: str,
    split: str,
    condition: str,
    checkpoint_id: str,
    controller_kind: str,
    controller_version: str,
    seed_namespace: str,
    seed_index: int,
    seed: int,
    reset_seeds: Sequence[int],
    episode_ids: Sequence[str],
    intended_actions: Sequence[float],
    applied_actions: Sequence[float],
    rng_start_sha256: str,
    rng_end_sha256: str,
    action_count: int,
    planner_budget: dict[str, int] | None,
) -> None:
    """Bind a whole real-execution run to its declared RNG stream."""

    if len(episode_ids) != len(reset_seeds):
        raise RuntimeError("policy run episode IDs and reset seeds differ in length")
    if len(intended_actions) != len(applied_actions) or len(intended_actions) != action_count:
        raise RuntimeError("policy run action trace differs from its controller draw count")
    trace = {
        "episode_ids": list(episode_ids),
        "intended": [float(value) for value in intended_actions],
        "applied": [float(value) for value in applied_actions],
    }
    rows.policy_runs.append(
        {
            "run_id": run_id,
            "task_id": task_id,
            "split": split,
            "condition": condition,
            "checkpoint_id": checkpoint_id,
            "controller_kind": controller_kind,
            "controller_version": controller_version,
            "seed_namespace": seed_namespace,
            "seed_index": seed_index,
            "seed": seed,
            "reset_seeds": list(reset_seeds),
            "episode_ids": list(episode_ids),
            "rng_start_sha256": rng_start_sha256,
            "rng_end_sha256": rng_end_sha256,
            "action_count": action_count,
            "action_trace_sha256": hashlib.sha256(canonical_json_bytes(trace)).hexdigest(),
            "planner_budget": planner_budget,
        }
    )


def append_collection_policy_run(
    rows: ReplicateRows,
    collection: CollectionEvidence,
    *,
    split: str,
    condition: str,
    checkpoint_id: str,
    controller: UniformRandomController,
    seed_namespace: str,
    seed_index: int,
    seed: int,
    rng_start_sha256: str,
    actions_before: int,
) -> None:
    """Preserve RNG and action-use evidence for a random collection run."""

    episodes = collection.episodes
    append_policy_run(
        rows,
        run_id=collection.run_id,
        task_id=collection.task_id,
        split=split,
        condition=condition,
        checkpoint_id=checkpoint_id,
        controller_kind="uniform_random",
        controller_version=controller.version,
        seed_namespace=seed_namespace,
        seed_index=seed_index,
        seed=seed,
        reset_seeds=[episode.reset_seed for episode in episodes],
        episode_ids=[episode.episode_id for episode in episodes],
        intended_actions=[action for episode in episodes for action in episode.intended_actions],
        applied_actions=[action for episode in episodes for action in episode.applied_actions],
        rng_start_sha256=rng_start_sha256,
        rng_end_sha256=controller.rng_digest,
        action_count=controller.actions_emitted - actions_before,
        planner_budget=None,
    )


def preserve_evaluated_checkpoint(
    rows: ReplicateRows,
    *,
    condition: str,
    runtime: WorldModelRuntime,
    output_directory: Path,
) -> None:
    """Write the exact compound model/optimizer state used by a condition."""

    if any(row["condition"] == condition for row in rows.evaluated_checkpoints):
        raise RuntimeError(f"evaluated checkpoint {condition!r} was preserved twice")
    state = runtime.owner.snapshot_state()
    filename = f"{rows.replicate_id}-{condition}-model-state.bin"
    path = output_directory / filename
    atomic_write_exclusive(path, state.payload)
    rows.evaluated_checkpoints.append(
        {
            "condition": condition,
            "model_version": state.version,
            "parameter_sha256": runtime.digest,
            "live_state_sha256": state.digest,
            "media_type": "application/vnd.prospect.wm001.owned-model-state",
            "bytes": len(state.payload),
            "sha256": hashlib.sha256(state.payload).hexdigest(),
            "filename": filename,
        }
    )


def load_evaluated_checkpoint(
    rows: ReplicateRows,
    *,
    condition: str,
    output_directory: Path,
    device: str,
) -> WorldModelRuntime:
    """Reload and revalidate one immutable evaluation state from its artifact."""

    matches = [row for row in rows.evaluated_checkpoints if row.get("condition") == condition]
    if len(matches) != 1:
        raise RuntimeError(f"expected one preserved evaluated checkpoint for {condition!r}")
    reference = matches[0]
    filename = reference.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError(f"evaluated checkpoint {condition!r} has an unsafe filename")
    path = output_directory / filename
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != reference.get("bytes") or digest != reference.get("sha256"):
        raise RuntimeError(f"evaluated checkpoint {condition!r} file reference differs")
    if digest != reference.get("live_state_sha256"):
        raise RuntimeError(f"evaluated checkpoint {condition!r} live-state digest differs")
    runtime = WorldModelRuntime.from_payload(payload, device=device)
    if (
        runtime.version != reference.get("model_version")
        or runtime.digest != reference.get("parameter_sha256")
        or runtime.live_state_digest != reference.get("live_state_sha256")
    ):
        raise RuntimeError(f"evaluated checkpoint {condition!r} semantic identity differs")
    return runtime


def configure_determinism(seed: int, *, device: str) -> None:
    """Configure every process-global RNG and deterministic Torch path."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")


def to_transition_batch(transitions: Sequence[EpistemicTransition]) -> TransitionBatch:
    observations, contexts, actions, targets, transition_ids = transition_arrays(transitions)
    return TransitionBatch.from_arrays(
        transition_ids=transition_ids,
        observations=observations,
        contexts=contexts,
        actions=actions,
        next_observations=observations + targets[:, :3],
        rewards=targets[:, 3],
    )


def append_collection_rows(
    rows: ReplicateRows,
    collection: CollectionEvidence,
    *,
    split: str,
    condition: str,
    checkpoint_id: str,
    learning_allowed: bool,
    replay_writes_allowed: bool,
    started_at: str,
    completed_at: str,
) -> None:
    for episode in collection.episodes:
        append_episode_rows(
            rows,
            episode,
            split=split,
            condition=condition,
            checkpoint_id=checkpoint_id,
            learning_allowed=learning_allowed,
            replay_writes_allowed=replay_writes_allowed,
            started_at=started_at,
            completed_at=completed_at,
        )


def append_episode_rows(
    rows: ReplicateRows,
    episode: EpisodeEvidence,
    *,
    split: str,
    condition: str,
    checkpoint_id: str,
    learning_allowed: bool,
    replay_writes_allowed: bool,
    started_at: str,
    completed_at: str,
) -> None:
    first_decision = episode.transitions[0].experience.decision
    if first_decision is None:
        raise ValueError("WM-001 episode has no first decision")
    first_prediction = first_decision.selected_assessment.prediction
    first_parameters = cast(dict[str, object], first_prediction.distribution.parameters)
    action_trace = {
        "intended": list(episode.intended_actions),
        "applied": list(episode.applied_actions),
    }
    rows.episodes.append(
        {
            "episode_id": episode.episode_id,
            "run_id": episode.transitions[0].experience.run_id,
            "task_id": episode.task_id,
            "split": split,
            "condition": condition,
            "checkpoint_id": checkpoint_id,
            "reset_seed": episode.reset_seed,
            "process_id": os.getpid(),
            "model_version": first_prediction.model_version,
            "parameter_sha256": str(first_parameters["parameter_digest"]),
            "learning_allowed": learning_allowed,
            "replay_writes_allowed": replay_writes_allowed,
            "environment_steps": len(episode.transitions),
            "return": episode.undiscounted_return,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "action_trace_sha256": hashlib.sha256(canonical_json_bytes(action_trace)).hexdigest(),
            "transition_ids": [transition.transition_id for transition in episode.transitions],
        }
    )
    for transition in episode.transitions:
        lineage = transition_lineage_row(transition, split=split)
        target = _transition_target(transition)
        scaled_target = _scaled_transition_target(transition)
        rows.transitions.append(
            {
                "transition_id": str(lineage["transition_id"]),
                "run_id": str(lineage["run_id"]),
                "episode_id": str(lineage["episode_id"]),
                "task_id": str(lineage["task_id"]),
                "task_context": float(lineage["task_context"]),
                "split": split,
                "step_index": int(lineage["step_index"]),
                "real_or_imagined": "real",
                "pre_observation_id": str(lineage["pre_action_observation_id"]),
                "decision_id": str(lineage["decision_id"]),
                "executed_action_id": str(lineage["executed_action_id"]),
                "next_observation_id": str(lineage["next_observation_id"]),
                "model_version_at_action": str(lineage["model_version_at_action"]),
                "parameter_sha256_at_action": str(lineage["parameter_digest_at_action"]),
                "pre_observation": [float(value) for value in cast(Sequence[float], target["before"])],
                "intended_action": float(lineage["intended_action"]),
                "applied_action": float(lineage["applied_action"]),
                "next_observation": [float(value) for value in cast(Sequence[float], target["next"])],
                "reward": float(lineage["reward"]),
                "terminated": bool(lineage["terminated"]),
                "truncated": bool(lineage["truncated"]),
                "scaled_target": list(scaled_target),
                "target_sha256": _scaled_target_sha256(scaled_target),
            }
        )


def _transition_target(transition: EpistemicTransition) -> dict[str, object]:
    decision = transition.experience.decision
    if decision is None:
        raise ValueError("transition target has no decision")
    before = cast(
        dict[str, object],
        decision.belief.distribution.parameters,
    )["physical_observation"]
    observation_payload = cast(
        dict[str, object],
        transition.experience.observation.evidence.payload,
    )
    outcome_payload = cast(
        dict[str, object],
        transition.experience.outcome.evidence.payload,
    )
    return {
        "before": before,
        "context": outcome_payload["task_context"],
        "intended_action": outcome_payload["intended_torque"],
        "applied_action": outcome_payload["applied_torque"],
        "next": observation_payload["physical_observation"],
        "reward": outcome_payload["reward"],
    }


def _scaled_transition_target(
    transition: EpistemicTransition,
) -> tuple[float, float, float, float]:
    """Return the protocol-fixed scaled target in canonical field order."""

    target = _transition_target(transition)
    before = np.asarray(target["before"], dtype=np.float64)
    after = np.asarray(target["next"], dtype=np.float64)
    reward = target["reward"]
    if not isinstance(reward, (int, float)):
        raise TypeError("transition reward must be numeric")
    return cast(
        tuple[float, float, float, float],
        tuple(
            [
                (after[0] - before[0]) / 2.0,
                (after[1] - before[1]) / 2.0,
                (after[2] - before[2]) / 16.0,
                float(reward) / 16.2736044,
            ]
        ),
    )


def _scaled_target_sha256(
    scaled_target: Sequence[float],
) -> str:
    """Hash four scaled targets as little-endian IEEE-754 float64 bytes."""

    scaled = np.asarray(scaled_target, dtype="<f8")
    if scaled.shape != (4,) or not np.isfinite(scaled).all():
        raise ValueError("scaled WM-001 target must contain four finite values")
    return hashlib.sha256(scaled.tobytes(order="C")).hexdigest()


def append_predictive_metric(
    rows: ReplicateRows,
    metrics: PredictiveMetrics,
    *,
    runtime: WorldModelRuntime,
    task_id: str,
    condition: str,
    checkpoint_id: str,
    split: str,
    evidence_directory: Path,
) -> None:
    evidence_path = evidence_directory / (f"{rows.replicate_id}-{task_id}-{condition}-predictions.bin")
    atomic_write_exclusive(evidence_path, metrics.prediction_payload)
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != metrics.prediction_rows_sha256:
        raise RuntimeError("persisted predictive evidence digest differs from its metric row")
    rows.predictive_metrics.append(
        {
            "task_id": task_id,
            "condition": condition,
            "checkpoint_id": checkpoint_id,
            "model_version": runtime.version,
            "parameter_sha256": runtime.digest,
            "live_state_sha256": runtime.live_state_digest,
            "split": split,
            "transition_count": metrics.transition_count,
            "mixture_nll_nats_per_target_dimension": metrics.mixture_nll_nats_per_target_dimension,
            "normalized_rmse": metrics.normalized_rmse,
            "interval_90_coverage": metrics.interval_90_coverage,
            "interval_90_covered_target_count": metrics.interval_90_covered_target_count,
            "coverage_target_count": metrics.coverage_target_count,
            "coverage_semantics": metrics.coverage_semantics,
            "prediction_rows_sha256": metrics.prediction_rows_sha256,
            "prediction_evidence_file": evidence_path.name,
            "prediction_evidence_bytes": len(metrics.prediction_payload),
        }
    )


def append_update_row(
    rows: ReplicateRows,
    receipt: UpdateReceipt,
    evidence: LearningEvidence,
    *,
    phase: str,
    eligible_splits: Sequence[str],
    committed_parameter_sha256: str,
    committed_live_state_sha256: str,
    manifest_directory: Path,
) -> None:
    manifest_path = manifest_directory / f"{rows.replicate_id}-{phase}-optimizer-batches.bin"
    atomic_write_exclusive(manifest_path, evidence.sampling_manifest)
    rows.optimizer_batch_manifests.append(
        {
            "phase": phase,
            "media_type": "application/vnd.prospect.wm001.bootstrap-manifest",
            "bytes": len(evidence.sampling_manifest),
            "sha256": evidence.sampling_manifest_sha256,
            "filename": manifest_path.name,
        }
    )
    target_permutation_file: dict[str, object] | None = None
    if evidence.target_permutation_payload is not None:
        permutation_path = manifest_directory / (f"{rows.replicate_id}-{phase}-target-permutation.bin")
        atomic_write_exclusive(permutation_path, evidence.target_permutation_payload)
        permutation_sha256 = hashlib.sha256(evidence.target_permutation_payload).hexdigest()
        if permutation_sha256 != evidence.target_permutation_sha256:
            raise RuntimeError("persisted target permutation digest differs from learning evidence")
        target_permutation_file = {
            "media_type": "application/vnd.prospect.wm001.target-permutation",
            "bytes": len(evidence.target_permutation_payload),
            "sha256": permutation_sha256,
            "filename": permutation_path.name,
        }
    rows.updates.append(
        {
            "receipt_id": receipt.receipt_id,
            "phase": phase,
            "status": "committed",
            "predecessor_parameter_sha256": evidence.predecessor_parameter_sha256,
            "candidate_parameter_sha256": evidence.candidate_parameter_sha256,
            "committed_parameter_sha256": committed_parameter_sha256,
            "predecessor_model_version": receipt.previous_model_version,
            "committed_model_version": receipt.new_model_version,
            "eligible_splits": list(eligible_splits),
            "eligible_transition_count": len(evidence.consumed_transition_ids),
            "eligible_transition_ids": list(evidence.consumed_transition_ids),
            "consumed_sample_count": evidence.optimizer_steps * 5 * 256,
            "consumed_multiset_sha256": evidence.consumed_multiset_sha256,
            "sampling_manifest_sha256": evidence.sampling_manifest_sha256,
            "target_permutation_sha256": evidence.target_permutation_sha256,
            "target_permutation_file": target_permutation_file,
            "optimizer_steps": evidence.optimizer_steps,
            "live_state_before_sha256": evidence.predecessor_live_state_sha256,
            "live_state_after_sha256": committed_live_state_sha256,
        }
    )


def append_rejected_probe_row(
    rows: ReplicateRows,
    *,
    receipt_id: str,
    parameter_sha256: str,
    model_version: str,
    live_state_sha256: str,
    state_before: bytes,
    state_after: bytes,
    evidence_directory: Path,
) -> None:
    if state_before != state_after:
        raise RuntimeError("rejected update changed the full live-state snapshot")
    state_digest = hashlib.sha256(state_before).hexdigest()
    before_path = evidence_directory / (f"{rows.replicate_id}-rejected-probe-state-before.json")
    after_path = evidence_directory / (f"{rows.replicate_id}-rejected-probe-state-after.json")
    atomic_write_exclusive(before_path, state_before)
    atomic_write_exclusive(after_path, state_after)

    def reference(path: Path, payload: bytes) -> dict[str, object]:
        return {
            "media_type": ("application/vnd.prospect.wm001.rejected-probe-state+json"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": path.name,
        }

    rows.updates.append(
        {
            "receipt_id": receipt_id,
            "phase": "rejected_update_probe",
            "status": "rejected",
            "predecessor_parameter_sha256": parameter_sha256,
            "candidate_parameter_sha256": parameter_sha256,
            "committed_parameter_sha256": parameter_sha256,
            "predecessor_model_version": model_version,
            "committed_model_version": model_version,
            "eligible_splits": ["collect_a"],
            "eligible_transition_count": 0,
            "eligible_transition_ids": [],
            "consumed_sample_count": 0,
            "consumed_multiset_sha256": hashlib.sha256(b"").hexdigest(),
            "sampling_manifest_sha256": None,
            "target_permutation_sha256": None,
            "target_permutation_file": None,
            "optimizer_steps": 0,
            "live_state_before_sha256": live_state_sha256,
            "live_state_after_sha256": live_state_sha256,
            "full_state_before_sha256": state_digest,
            "full_state_after_sha256": state_digest,
            "full_state_before_file": reference(before_path, state_before),
            "full_state_after_file": reference(after_path, state_after),
        }
    )


def capture_rejected_probe_state(
    *,
    runtime: WorldModelRuntime,
    source_custody: RuntimeCustody,
    probe_custody: RuntimeCustody,
    probe_snapshot: AgentSnapshot,
    at: TimePoint,
    collection_controller: UniformRandomController,
    retained_learning_evidence: LearningEvidence,
) -> bytes:
    """Serialize every live component in scope for the rejected-update probe."""

    graph = encode_domain_graph(
        {
            "agent_snapshot": probe_snapshot,
            "source_events": tuple(source_custody.store.history(AGENT_ID, at)),
            "source_transitions": tuple(source_custody.ledger.transitions(AGENT_ID, at)),
            "source_updates": tuple(source_custody.ledger.updates(AGENT_ID, at)),
            "probe_transitions": tuple(probe_custody.ledger.transitions(AGENT_ID, at)),
            "probe_updates": tuple(probe_custody.ledger.updates(AGENT_ID, at)),
        }
    )
    model_state = runtime.owner.snapshot_state()
    evidence = retained_learning_evidence
    return cast(
        bytes,
        canonical_json_bytes(
            {
                "schema": "prospect.wm001.rejected-probe-full-state.v1",
                "captured_at": [at.clock_id, at.tick],
                "model_state": {
                    "version": model_state.version,
                    "digest": model_state.digest,
                    "payload_base64": base64.b64encode(model_state.payload).decode("ascii"),
                },
                "domain_graph": graph,
                "source_replay_rows": source_custody.replay.rows(),
                "probe_replay_rows": probe_custody.replay.rows(),
                "source_identity_base64": base64.b64encode(source_custody.identities.checkpoint_bytes()).decode(
                    "ascii"
                ),
                "probe_identity_base64": base64.b64encode(probe_custody.identities.checkpoint_bytes()).decode("ascii"),
                "collection_rng_state": collection_controller.state(),
                "process_rng": {
                    "python_base64": base64.b64encode(snapshot_python_rng()).decode("ascii"),
                    "numpy_base64": base64.b64encode(snapshot_numpy_rng()).decode("ascii"),
                    "torch_cpu_base64": base64.b64encode(snapshot_torch_cpu_rng()).decode("ascii"),
                    "torch_accelerator_base64": base64.b64encode(snapshot_torch_accelerator_rng()).decode("ascii"),
                },
                "retained_learning_evidence": {
                    "phase": evidence.phase,
                    "consumed_transition_ids": list(evidence.consumed_transition_ids),
                    "consumed_multiset_sha256": (evidence.consumed_multiset_sha256),
                    "predecessor_parameter_sha256": (evidence.predecessor_parameter_sha256),
                    "candidate_parameter_sha256": (evidence.candidate_parameter_sha256),
                    "predecessor_live_state_sha256": (evidence.predecessor_live_state_sha256),
                    "candidate_live_state_sha256": (evidence.candidate_live_state_sha256),
                    "optimizer_steps": evidence.optimizer_steps,
                    "sampling_manifest_base64": base64.b64encode(evidence.sampling_manifest).decode("ascii"),
                    "sampling_manifest_sha256": (evidence.sampling_manifest_sha256),
                    "sampled_id_counts": [[identity, count] for identity, count in evidence.sampled_id_counts],
                    "target_permutation_sha256": (evidence.target_permutation_sha256),
                    "target_permutation_base64": (
                        None
                        if evidence.target_permutation_payload is None
                        else base64.b64encode(evidence.target_permutation_payload).decode("ascii")
                    ),
                    "loss_history": list(evidence.loss_history),
                },
            },
        ),
    )


def run_behavior_condition(
    rows: ReplicateRows,
    *,
    replicate_id: str,
    task_id: str,
    reset_seeds: Sequence[int],
    condition: str,
    checkpoint_id: str,
    backend: PredictiveBackend,
    controller: UniformRandomController,
    seed_namespace: str,
    seed_index: int,
) -> list[float]:
    """Execute paired real episodes and immediately preserve their raw rows."""

    run_id = f"{replicate_id}:behavior:{task_id}:{condition}"
    custody = RuntimeCustody.create(run_id)
    tick = 0
    returns: list[float] = []
    episode_ids: list[str] = []
    intended_actions: list[float] = []
    applied_actions: list[float] = []
    rng_start_sha256 = controller.rng_digest
    actions_before = controller.actions_emitted
    for index, reset_seed in enumerate(reset_seeds):
        started = utc_now()
        episode, _ = run_episode(
            run_id=run_id,
            task_id=task_id,
            episode_id=f"{replicate_id}:behavior:{task_id}:{condition}:{index}",
            reset_seed=reset_seed,
            controller=controller,
            backend=backend,
            custody=custody,
            start_tick=tick,
        )
        completed = utc_now()
        append_episode_rows(
            rows,
            episode,
            split=("behavior_evaluation_a" if task_id == TASK_A else "behavior_evaluation_b"),
            condition=condition,
            checkpoint_id=checkpoint_id,
            learning_allowed=False,
            replay_writes_allowed=False,
            started_at=started,
            completed_at=completed,
        )
        returns.append(episode.undiscounted_return)
        episode_ids.append(episode.episode_id)
        intended_actions.extend(episode.intended_actions)
        applied_actions.extend(episode.applied_actions)
        tick = episode.final_tick + 1
        del episode
        gc.collect()
    append_policy_run(
        rows,
        run_id=run_id,
        task_id=task_id,
        split=("behavior_evaluation_a" if task_id == TASK_A else "behavior_evaluation_b"),
        condition=condition,
        checkpoint_id=checkpoint_id,
        controller_kind="uniform_random",
        controller_version=controller.version,
        seed_namespace=seed_namespace,
        seed_index=seed_index,
        seed=controller.seed,
        reset_seeds=reset_seeds,
        episode_ids=episode_ids,
        intended_actions=intended_actions,
        applied_actions=applied_actions,
        rng_start_sha256=rng_start_sha256,
        rng_end_sha256=controller.rng_digest,
        action_count=controller.actions_emitted - actions_before,
        planner_budget=None,
    )
    return returns


def run_batched_cem_condition(
    rows: ReplicateRows,
    *,
    replicate_id: str,
    task_id: str,
    reset_seeds: Sequence[int],
    condition: str,
    checkpoint_id: str,
    backend: PredictiveBackend,
    planner: CEMController,
) -> list[float]:
    """Batch only imagined planning while retaining canonical real step custody."""

    expected_controller_version = (
        "wm001-analytic-pendulum-cem-torchrl-0.13.3-v1" if condition == "oracle" else f"wm001-sha256:{backend.digest}"
    )
    if planner.version != expected_controller_version:
        raise RuntimeError(
            "CEM controller is not bound to the evaluated backend: "
            f"expected {expected_controller_version}, found {planner.version}"
        )

    run_id = f"{replicate_id}:behavior:{task_id}:{condition}"
    custody = RuntimeCustody.create(run_id)
    rng_start_sha256 = planner.rng_digest
    actions_before = int(planner.state_dict()["actions_emitted"])
    sessions = [
        PendulumEpisodeSession(
            run_id=run_id,
            task_id=task_id,
            episode_id=f"{replicate_id}:behavior:{task_id}:{condition}:{index}",
            reset_seed=reset_seed,
            backend=backend,
            custody=custody,
            start_tick=0,
            controller=PresetActionController(version=planner.version),
        )
        for index, reset_seed in enumerate(reset_seeds)
    ]
    started = utc_now()
    tick = 0
    for _ in range(200):
        states = np.stack(
            [
                np.concatenate(
                    (
                        session.current_observation.astype(np.float32),
                        np.asarray([session.context], dtype=np.float32),
                    )
                )
                for session in sessions
            ]
        )
        actions = planner.act(states).detach().cpu().numpy().reshape(len(sessions))
        for session, action in zip(sessions, actions, strict=True):
            result = session.step(float(action), tick)
            tick = result.transition.created_at.tick + 1
    completed = utc_now()
    returns: list[float] = []
    episodes: list[EpisodeEvidence] = []
    for session in sessions:
        episode = session.finish()
        episodes.append(episode)
        append_episode_rows(
            rows,
            episode,
            split=("behavior_evaluation_a" if task_id == TASK_A else "behavior_evaluation_b"),
            condition=condition,
            checkpoint_id=checkpoint_id,
            learning_allowed=False,
            replay_writes_allowed=False,
            started_at=started,
            completed_at=completed,
        )
        returns.append(episode.undiscounted_return)
    actions_after = int(planner.state_dict()["actions_emitted"])
    append_policy_run(
        rows,
        run_id=run_id,
        task_id=task_id,
        split=("behavior_evaluation_a" if task_id == TASK_A else "behavior_evaluation_b"),
        condition=condition,
        checkpoint_id=checkpoint_id,
        controller_kind=("cem_oracle" if condition == "oracle" else "cem_learned"),
        controller_version=planner.version,
        seed_namespace="planner",
        seed_index=0,
        seed=planner.seed,
        reset_seeds=reset_seeds,
        episode_ids=[episode.episode_id for episode in episodes],
        intended_actions=[action for episode in episodes for action in episode.intended_actions],
        applied_actions=[action for episode in episodes for action in episode.applied_actions],
        rng_start_sha256=rng_start_sha256,
        rng_end_sha256=planner.rng_digest,
        action_count=actions_after - actions_before,
        planner_budget={key: int(value) for key, value in asdict(planner.budget).items()},
    )
    return returns


def _canonical_transition_dataset(
    transitions: Sequence[EpistemicTransition],
) -> dict[str, object]:
    observations, contexts, actions, targets, transition_ids = transition_arrays(transitions)
    return {
        "transition_ids": list(transition_ids),
        "observations": observations.tolist(),
        "contexts": contexts.tolist(),
        "actions": actions.tolist(),
        "next_observations": (observations + targets[:, :3]).tolist(),
        "rewards": targets[:, 3].tolist(),
    }


def save_replicate_checkpoint(
    *,
    path: Path,
    rows: ReplicateRows,
    runtime: WorldModelRuntime,
    collection: CollectionEvidence,
    collection_a: Sequence[EpistemicTransition],
    collection_b: Sequence[EpistemicTransition],
    collection_rng_states: dict[str, object],
    planner: CEMController,
    update_evidence: Sequence[LearningEvidence],
    created_at: TimePoint,
) -> tuple[str, list[dict[str, object]], int]:
    """Create the exact fifteen-component retained-state bundle."""

    if path.exists():
        raise FileExistsError(f"refusing to replace checkpoint evidence: {path}")
    snapshot = collection.final_agent.snapshot(TimePoint(max(created_at.tick, collection.next_tick)))
    identity_bytes = collection.custody.identities.checkpoint_bytes()
    next_tick = max(created_at.tick, collection.next_tick) + 1
    agent_runtime = {
        "schema": "prospect.wm001.agent-runtime.v1",
        "identity_namespace": collection.custody.identities.namespace,
        "identity_checkpoint_base64": base64.b64encode(identity_bytes).decode("ascii"),
        "next_tick": next_tick,
        "agent_id": snapshot.agent_id,
        "configuration_version": snapshot.configuration_version,
        "memory_version": snapshot.memory_version,
        "knowledge_version": snapshot.knowledge_version,
        "model_version": runtime.version,
        "representation_version": snapshot.representation_version,
        "policy_version": snapshot.policy_version,
        "belief_id": snapshot.belief.belief_id,
    }
    model_ledger = [
        {
            "phase": evidence.phase,
            "predecessor_parameter_sha256": evidence.predecessor_parameter_sha256,
            "candidate_parameter_sha256": evidence.candidate_parameter_sha256,
            "predecessor_live_state_sha256": evidence.predecessor_live_state_sha256,
            "candidate_live_state_sha256": evidence.candidate_live_state_sha256,
        }
        for evidence in update_evidence
    ]
    replay_sampling = {
        "schema": "prospect.wm001.replay-sampling-history.v1",
        "manifests": [
            {
                "phase": evidence.phase,
                "sha256": evidence.sampling_manifest_sha256,
                "bytes": len(evidence.sampling_manifest),
                "payload_base64": base64.b64encode(evidence.sampling_manifest).decode("ascii"),
            }
            for evidence in update_evidence
        ],
    }
    training_rows = [row for row in rows.transitions if row["split"] in {"collect_a", "collect_b"}]
    retained_transitions = (*collection_a, *collection_b)
    retained_events = tuple(transition.experience for transition in retained_transitions)
    experience_store = {
        "schema": "prospect.wm001.experience-custody.v1",
        "transition_rows": training_rows,
        "domain_graph": encode_domain_graph(
            {
                "events": retained_events,
                "transitions": retained_transitions,
            }
        ),
    }
    replay_index = {
        "schema": "prospect.wm001.replay-index.v1",
        "canonical_experience_rows": collection.custody.replay.rows(),
        "collect_a": _canonical_transition_dataset(collection_a),
        "collect_b": _canonical_transition_dataset(collection_b),
    }
    retained_update_phases = {evidence.phase for evidence in update_evidence}
    retained_update_rows = [row for row in rows.updates if row["phase"] in retained_update_phases]
    retained_receipt_ids = {str(row["receipt_id"]) for row in retained_update_rows}
    retained_receipts = tuple(
        receipt
        for receipt in collection.custody.ledger.updates(
            snapshot.agent_id,
            snapshot.captured_at,
        )
        if receipt.receipt_id in retained_receipt_ids
    )
    if len(retained_receipts) != len(retained_update_phases):
        raise RuntimeError("checkpoint custody does not contain every retained update receipt")
    transition_references = {
        id(transition): transition_external_reference(transition.transition_id) for transition in retained_transitions
    }
    update_receipts = {
        "schema": "prospect.wm001.update-receipts.v1",
        "updates": retained_update_rows,
        "domain_graph": encode_domain_graph(
            {"receipts": retained_receipts},
            external_references=transition_references,
        ),
    }
    if snapshot.latest_update is None or snapshot.latest_update.resulting_belief is None:
        raise RuntimeError("checkpoint boundary lacks its canonical learning receipt/belief")
    if all(snapshot.latest_update is not receipt for receipt in retained_receipts):
        raise RuntimeError("checkpoint boundary update is absent from retained update custody")
    agent_runtime["domain_graph"] = encode_domain_graph(
        {"snapshot": snapshot},
        external_references={
            id(snapshot.latest_update): update_external_reference(snapshot.latest_update.receipt_id),
            id(snapshot.belief): belief_external_reference(snapshot.belief.belief_id),
        },
    )
    scaling = {
        "schema": "prospect.wm001.fixed-scaling.v1",
        **asdict(FixedScaling()),
    }
    raw_components = {
        "world_model": ComponentPayload(
            "world_model",
            runtime.version,
            runtime.model_bytes,
            "application/vnd.prospect.wm001.model",
        ),
        "optimizer": ComponentPayload(
            "optimizer",
            runtime.version,
            runtime.optimizer_bytes,
            "application/vnd.prospect.wm001.adamw",
        ),
        "model_version_ledger": ComponentPayload(
            "model_version_ledger",
            runtime.version,
            canonical_json_bytes(model_ledger),
            "application/json",
        ),
        "experience_store": ComponentPayload(
            "experience_store",
            "wm001-experience-v1",
            canonical_json_bytes(experience_store),
            "application/json",
        ),
        "replay_index": ComponentPayload(
            "replay_index",
            "wm001-replay-v1",
            canonical_json_bytes(replay_index),
            "application/json",
        ),
        "replay_sampling_history": ComponentPayload(
            "replay_sampling_history",
            "wm001-sampling-v1",
            canonical_json_bytes(replay_sampling),
            "application/json",
        ),
        "update_receipts": ComponentPayload(
            "update_receipts",
            "wm001-updates-v1",
            canonical_json_bytes(update_receipts),
            "application/json",
        ),
        "agent_runtime": ComponentPayload(
            "agent_runtime",
            runtime.version,
            canonical_json_bytes(agent_runtime),
            "application/json",
        ),
        "scaling_configuration": ComponentPayload(
            "scaling_configuration",
            "wm001-fixed-scaling-v1",
            canonical_json_bytes(scaling),
            "application/json",
        ),
        "python_rng": ComponentPayload(
            "python_rng",
            "python-rng-v1",
            snapshot_python_rng(),
            "application/vnd.prospect.rng-state+json",
        ),
        "numpy_rng": ComponentPayload(
            "numpy_rng",
            "numpy-rng-v1",
            snapshot_numpy_rng(),
            "application/vnd.prospect.rng-state+json",
        ),
        "torch_cpu_rng": ComponentPayload(
            "torch_cpu_rng",
            "torch-rng-v1",
            snapshot_torch_cpu_rng(),
            "application/vnd.prospect.rng-state+json",
        ),
        "torch_accelerator_rng": ComponentPayload(
            "torch_accelerator_rng",
            "torch-rng-v1",
            snapshot_torch_accelerator_rng(),
            "application/vnd.prospect.rng-state+json",
        ),
        "collection_rng": ComponentPayload(
            "collection_rng",
            "collection-rng-v1",
            canonical_json_bytes(
                {
                    "schema": "prospect.wm001.collection-rng.v1",
                    "states": collection_rng_states,
                }
            ),
            "application/json",
        ),
        "planner_rng": ComponentPayload(
            "planner_rng",
            "planner-rng-v1",
            snapshot_planner_rng(planner),
            "application/json",
        ),
    }
    if tuple(raw_components) != CANONICAL_COMPONENT_IDS:
        raise RuntimeError("checkpoint component assembly order differs from the sealed contract")
    report = save_checkpoint(
        path,
        checkpoint_id=f"{rows.replicate_id}:after_b_replay",
        agent_id=AGENT_ID,
        created_at=created_at,
        components=raw_components,
        versions={
            "model": runtime.version,
            "protocol": PROTOCOL_SHA256,
        },
    )
    archive_bytes = path.read_bytes()
    rows.checkpoint_archive = {
        "media_type": "application/vnd.prospect.checkpoint+zip",
        "bytes": len(archive_bytes),
        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "filename": path.name,
    }
    return report.manifest_sha256, list(report.component_rows()), next_tick


def run_restart_parity(
    *,
    checkpoint_path: Path,
    output_directory: Path,
    rows: ReplicateRows,
    runtime: WorldModelRuntime,
    custody: RuntimeCustody,
    next_tick: int,
    planner: CEMController,
    checkpoint_manifest_sha256: str,
    component_rows: Sequence[dict[str, object]],
    task_reset_seeds: dict[str, int],
    device: str,
    boundary_snapshot: AgentSnapshot,
) -> dict[str, object]:
    """Evaluate the bundle here and in a genuinely new interpreter."""

    specification = {
        "schema": "prospect.wm001.restart-spec.v1",
        "task_reset_seeds": task_reset_seeds,
        "device": device,
    }
    spec_path = output_directory / f"{rows.replicate_id}-restart-spec.json"
    original_path = output_directory / f"{rows.replicate_id}-live-evaluation.json"
    restored_path = output_directory / f"{rows.replicate_id}-restored-evaluation.json"
    restore_runtime_path = output_directory / f"{rows.replicate_id}-restore-runtime.json"
    atomic_write_exclusive(spec_path, canonical_json_bytes(specification) + b"\n")
    original = evaluate_live_state(
        runtime=runtime,
        custody=custody,
        next_tick=next_tick,
        planner=planner,
        task_reset_seeds=task_reset_seeds,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        component_hashes={str(row["component_id"]): str(row["sha256"]) for row in component_rows},
        boundary_snapshot=boundary_snapshot,
    )
    atomic_write_exclusive(
        original_path,
        canonical_json_bytes(original) + b"\n",
    )
    bootstrap_custody = _verify_live_bootstrap_custody()
    bootstrap_descriptor = cast(int, bootstrap_custody["bootstrap_fd"])
    runtime_seal_descriptor = cast(int, bootstrap_custody["runtime_seal_fd"])
    bootstrap_path = _descriptor_path(bootstrap_descriptor)
    environment = dict(os.environ)
    stdout_path = output_directory / f"{rows.replicate_id}-restore.stdout"
    stderr_path = output_directory / f"{rows.replicate_id}-restore.stderr"
    with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    bootstrap_path,
                    "--bootstrap-fd",
                    str(bootstrap_descriptor),
                    "--runtime-seal-fd",
                    str(runtime_seal_descriptor),
                    "--restore-eval-entry",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--spec",
                    str(spec_path),
                    "--output",
                    str(restored_path),
                    "--runtime-identity-output",
                    str(restore_runtime_path),
                ],
                cwd=Path.cwd().resolve(strict=True),
                env=environment,
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
                pass_fds=(bootstrap_descriptor, runtime_seal_descriptor),
                timeout=_RESTART_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("fresh-process restore exceeded its sealed timeout") from error
        stdout_stream.flush()
        stderr_stream.flush()
        os.fsync(stdout_stream.fileno())
        os.fsync(stderr_stream.fileno())
    if stdout_path.stat().st_size > _RESTART_LOG_LIMIT_BYTES or stderr_path.stat().st_size > _RESTART_LOG_LIMIT_BYTES:
        raise RuntimeError("fresh-process restore exceeded its sealed log limit")
    _captured_bootstrap_custody()
    if completed.returncode != 0:
        stderr_excerpt = stderr_path.read_bytes()[:4096].decode("utf-8", errors="replace")
        raise RuntimeError(f"fresh-process restore failed with exit {completed.returncode}: {stderr_excerpt}")
    restored = load_evaluation(restored_path)
    parity = compare_parity(original, restored)
    restore_runtime_payload = restore_runtime_path.read_bytes()
    restore_runtime = json.loads(restore_runtime_payload)
    from .binding import (
        FORMAL_PROCESS_ENVIRONMENT_KEYS,
        package_roots,
        python_flag_identity,
    )

    runtime_seal = cast(dict[str, object], bootstrap_custody["runtime_seal"])
    closure_block = (
        cast(dict[str, object], runtime_seal.get("dependencies", {}))
        if runtime_seal.get("schema") == "prospect.world-model-lifecycle.formal-binding.v10"
        else runtime_seal
    )
    package_rows = closure_block.get("package_roots")
    package_ownership = closure_block.get("package_ownership")
    standard_library = closure_block.get("standard_library")
    if (
        not isinstance(package_rows, list)
        or len(package_rows) != 1
        or not isinstance(package_rows[0], dict)
        or not isinstance(package_ownership, dict)
        or not isinstance(standard_library, dict)
    ):
        raise RuntimeError("captured runtime seal has malformed root inventories")
    expected_restore_runtime = {
        "schema": "prospect.wm001.restart-runtime.v2",
        "python_executable": sys.executable,
        "python_executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "python_flags": python_flag_identity(),
        "process_environment": {
            key: value for key, value in sorted(os.environ.items()) if key in FORMAL_PROCESS_ENVIRONMENT_KEYS
        },
        "package_root": str(package_roots()[0]),
        "package_root_inventory": package_rows[0],
        "package_ownership": package_ownership,
        "standard_library": standard_library,
        "runtime_seal_sha256": bootstrap_custody["runtime_seal_sha256"],
        "runtime_seal_descriptor_custody": True,
        "bootstrap_source_sha256": bootstrap_custody["bootstrap_sha256"],
        "bootstrap_descriptor_custody": True,
        "deterministic_algorithms": True,
    }
    if (
        not isinstance(restore_runtime, dict)
        or restore_runtime != expected_restore_runtime
        or restore_runtime_payload != canonical_json_bytes(expected_restore_runtime) + b"\n"
    ):
        raise RuntimeError("fresh-process restore runtime differs from the parent closure")
    parity["restore_runtime"] = {
        **restore_runtime,
        "bytes": len(restore_runtime_payload),
        "sha256": hashlib.sha256(restore_runtime_payload).hexdigest(),
        "filename": restore_runtime_path.name,
    }

    def evaluation_reference(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "media_type": "application/vnd.prospect.wm001.restart-evaluation+json",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": path.name,
        }

    parity["live_evaluation"] = evaluation_reference(original_path)
    parity["restored_evaluation"] = evaluation_reference(restored_path)
    return cast(dict[str, object], parity)


def run_replicate(
    config: ExperimentConfig,
    *,
    master_seed: int,
    output_directory: Path,
) -> dict[str, object]:
    """Run one independent master-seed replicate from cold start through K7."""

    replicate_id = f"wm001-{config.lane}-{master_seed}"
    rows = ReplicateRows.create(replicate_id, master_seed)
    configure_determinism(rows.seed("torch_runtime"), device=config.device)
    print(f"[{utc_now()}] {replicate_id}: initialize cold model", flush=True)
    runtime = WorldModelRuntime.initialize(
        initialization_seed=rows.seed("model_initialization"),
        device=config.device,
    )
    cold = runtime.fork(device=config.device)
    frozen = runtime.fork(device=config.device)
    corrupted = runtime.fork(device=config.device)
    irrelevant = runtime.fork(device=config.device)
    cold_parameter_sha256 = runtime.digest
    cold_live_state_sha256 = runtime.live_state_digest
    preserve_evaluated_checkpoint(
        rows,
        condition="cold",
        runtime=cold,
        output_directory=output_directory,
    )
    preserve_evaluated_checkpoint(
        rows,
        condition="frozen",
        runtime=frozen,
        output_directory=output_directory,
    )

    learner_a = TransactionalWorldModelLearner(
        phase="train_a",
        bootstrap_seeds=[rows.seed("ensemble_bootstrap_a", index) for index in range(5)],
        minibatch_order_seed=rows.seed("minibatch_order_a"),
        optimizer_steps=config.optimizer_steps,
        device=config.device,
    )
    corrupted_learner = TransactionalWorldModelLearner(
        phase="train_a_corrupted",
        bootstrap_seeds=[rows.seed("ensemble_bootstrap_a", index) for index in range(5)],
        minibatch_order_seed=rows.seed("minibatch_order_a"),
        optimizer_steps=config.optimizer_steps,
        training_mode="joint_target_permuted",
        target_permutation_seed=rows.seed("corrupted_target_permutation"),
        device=config.device,
    )
    irrelevant_learner = TransactionalWorldModelLearner(
        phase="train_a_irrelevant",
        bootstrap_seeds=[rows.seed("ensemble_bootstrap_a", index) for index in range(5)],
        minibatch_order_seed=rows.seed("minibatch_order_a"),
        optimizer_steps=config.optimizer_steps,
        device=config.device,
    )
    custody = RuntimeCustody.create(f"{replicate_id}:canonical")
    collection_action_a_seed = rows.seed("collection_action", 0)
    collection_controller_a = UniformRandomController(collection_action_a_seed)
    collection_rng_a_start = collection_controller_a.rng_digest
    collection_actions_a_before = collection_controller_a.actions_emitted
    phase_started = utc_now()
    collection_a = collect_episodes(
        run_id=f"{replicate_id}:collect-a",
        task_id=TASK_A,
        episode_seeds=[rows.seed("collect_a_episode", index) for index in range(config.collection_episodes)],
        controller_factory=lambda _: collection_controller_a,
        backend=runtime,
        custody=custody,
        model_owner=runtime.owner,
        learner=learner_a,
    )
    phase_completed = utc_now()
    append_collection_rows(
        rows,
        collection_a,
        split="collect_a",
        condition="collection_random",
        checkpoint_id="cold",
        learning_allowed=True,
        replay_writes_allowed=True,
        started_at=phase_started,
        completed_at=phase_completed,
    )
    append_collection_policy_run(
        rows,
        collection_a,
        split="collect_a",
        condition="collection_random",
        checkpoint_id="cold",
        controller=collection_controller_a,
        seed_namespace="collection_action",
        seed_index=0,
        seed=collection_action_a_seed,
        rng_start_sha256=collection_rng_a_start,
        actions_before=collection_actions_a_before,
    )
    irrelevant_custody = RuntimeCustody.create(f"{replicate_id}:control:irrelevant")
    irrelevant_action_seed = rows.seed("irrelevant_collection_action")
    irrelevant_controller = UniformRandomController(irrelevant_action_seed)
    irrelevant_rng_start = irrelevant_controller.rng_digest
    irrelevant_actions_before = irrelevant_controller.actions_emitted
    irrelevant_started = utc_now()
    collection_irrelevant = collect_episodes(
        run_id=f"{replicate_id}:collect-irrelevant",
        task_id=INDEPENDENT_OSCILLATOR_TASK,
        episode_seeds=[rows.seed("collect_irrelevant_episode", index) for index in range(config.collection_episodes)],
        controller_factory=lambda _: irrelevant_controller,
        backend=irrelevant,
        custody=irrelevant_custody,
        model_owner=irrelevant.owner,
        learner=irrelevant_learner,
    )
    irrelevant_completed = utc_now()
    append_collection_rows(
        rows,
        collection_irrelevant,
        split="collect_irrelevant",
        condition="collection_random",
        checkpoint_id="cold",
        learning_allowed=True,
        replay_writes_allowed=True,
        started_at=irrelevant_started,
        completed_at=irrelevant_completed,
    )
    append_collection_policy_run(
        rows,
        collection_irrelevant,
        split="collect_irrelevant",
        condition="collection_random",
        checkpoint_id="cold",
        controller=irrelevant_controller,
        seed_namespace="irrelevant_collection_action",
        seed_index=0,
        seed=irrelevant_action_seed,
        rng_start_sha256=irrelevant_rng_start,
        actions_before=irrelevant_actions_before,
    )
    print(
        (
            f"[{utc_now()}] {replicate_id}: collected "
            f"A={len(collection_a.transitions)}, "
            f"irrelevant={len(collection_irrelevant.transitions)}; "
            "train candidates"
        ),
        flush=True,
    )
    replay_a = custody.replay.require_transitions(collection_a.transitions)
    corrupted_custody = RuntimeCustody.branch(
        custody,
        f"{replicate_id}:control:corrupted",
        replay_a,
    )
    corrupted_agent = make_branch_learning_agent(
        source_agent=collection_a.final_agent,
        at=TimePoint(collection_a.next_tick),
        model_owner=corrupted.owner,
        learner=corrupted_learner,
        custody=corrupted_custody,
    )
    receipt_a = collection_a.final_agent.learn(
        replay_a,
        at=TimePoint(collection_a.next_tick),
    )
    replay_irrelevant = collection_irrelevant.custody.replay.require_transitions(collection_irrelevant.transitions)
    irrelevant_receipt = collection_irrelevant.final_agent.learn(
        replay_irrelevant,
        at=TimePoint(collection_irrelevant.next_tick),
    )
    corrupted_receipt = corrupted_agent.learn(
        corrupted_custody.replay.require_transitions(replay_a),
        at=TimePoint(collection_a.next_tick),
    )
    if (
        learner_a.last_evidence is None
        or corrupted_learner.last_evidence is None
        or irrelevant_learner.last_evidence is None
    ):
        raise RuntimeError("transactional learners did not retain their preparation evidence")
    append_update_row(
        rows,
        receipt_a,
        learner_a.last_evidence,
        phase="train_a",
        eligible_splits=("collect_a",),
        committed_parameter_sha256=runtime.digest,
        committed_live_state_sha256=runtime.live_state_digest,
        manifest_directory=output_directory,
    )
    preserve_evaluated_checkpoint(
        rows,
        condition="corrupted",
        runtime=corrupted,
        output_directory=output_directory,
    )
    preserve_evaluated_checkpoint(
        rows,
        condition="after_a",
        runtime=runtime,
        output_directory=output_directory,
    )
    preserve_evaluated_checkpoint(
        rows,
        condition="irrelevant",
        runtime=irrelevant,
        output_directory=output_directory,
    )
    append_update_row(
        rows,
        irrelevant_receipt,
        irrelevant_learner.last_evidence,
        phase="train_a_irrelevant",
        eligible_splits=("collect_irrelevant",),
        committed_parameter_sha256=irrelevant.digest,
        committed_live_state_sha256=irrelevant.live_state_digest,
        manifest_directory=output_directory,
    )
    append_update_row(
        rows,
        corrupted_receipt,
        corrupted_learner.last_evidence,
        phase="train_a_corrupted",
        eligible_splits=("collect_a",),
        committed_parameter_sha256=corrupted.digest,
        committed_live_state_sha256=corrupted.live_state_digest,
        manifest_directory=output_directory,
    )

    validation_irrelevant_action_seed = rows.seed(
        "predictive_validation_irrelevant_action",
    )
    validation_irrelevant_controller = UniformRandomController(
        validation_irrelevant_action_seed,
    )
    validation_irrelevant_rng_start = validation_irrelevant_controller.rng_digest
    validation_irrelevant_actions_before = validation_irrelevant_controller.actions_emitted
    validation_irrelevant_backend = load_evaluated_checkpoint(
        rows,
        condition="irrelevant",
        output_directory=output_directory,
        device=config.device,
    )
    validation_irrelevant_parameter_sha256 = validation_irrelevant_backend.digest
    validation_irrelevant_live_state_sha256 = validation_irrelevant_backend.live_state_digest
    validation_irrelevant_started = utc_now()
    validation_irrelevant = collect_episodes(
        run_id=f"{replicate_id}:predictive-validation-irrelevant",
        task_id=INDEPENDENT_OSCILLATOR_TASK,
        episode_seeds=[
            rows.seed("predictive_validation_irrelevant_episode", index) for index in range(config.validation_episodes)
        ],
        controller_factory=lambda _: validation_irrelevant_controller,
        backend=validation_irrelevant_backend,
        custody=RuntimeCustody.create(f"{replicate_id}:validation-irrelevant"),
    )
    validation_irrelevant_completed = utc_now()
    if validation_irrelevant.custody.replay.event_count != 0:
        raise RuntimeError("irrelevant validation unexpectedly wrote to replay")
    if (
        validation_irrelevant_backend.digest != validation_irrelevant_parameter_sha256
        or validation_irrelevant_backend.live_state_digest != validation_irrelevant_live_state_sha256
    ):
        raise RuntimeError("irrelevant validation mutated its evaluated checkpoint")
    append_collection_rows(
        rows,
        validation_irrelevant,
        split="predictive_validation_irrelevant",
        condition="validation_random",
        checkpoint_id="irrelevant",
        learning_allowed=False,
        replay_writes_allowed=False,
        started_at=validation_irrelevant_started,
        completed_at=validation_irrelevant_completed,
    )
    append_collection_policy_run(
        rows,
        validation_irrelevant,
        split="predictive_validation_irrelevant",
        condition="validation_random",
        checkpoint_id="irrelevant",
        controller=validation_irrelevant_controller,
        seed_namespace="predictive_validation_irrelevant_action",
        seed_index=0,
        seed=validation_irrelevant_action_seed,
        rng_start_sha256=validation_irrelevant_rng_start,
        actions_before=validation_irrelevant_actions_before,
    )
    validation_irrelevant_batch = to_transition_batch(
        validation_irrelevant.transitions,
    )
    evaluation_irrelevant_backends = {
        "cold": load_evaluated_checkpoint(
            rows,
            condition="cold",
            output_directory=output_directory,
            device=config.device,
        ),
        "irrelevant": validation_irrelevant_backend,
    }
    for condition, backend in evaluation_irrelevant_backends.items():
        metrics = evaluate_mixture(
            backend.model,
            validation_irrelevant_batch,
            device=config.device,
        )
        append_predictive_metric(
            rows,
            metrics,
            runtime=backend,
            task_id=INDEPENDENT_OSCILLATOR_TASK,
            condition=condition,
            checkpoint_id=condition,
            split="predictive_validation_irrelevant",
            evidence_directory=output_directory,
        )

    probe = RejectedUpdateProbeLearner()
    probe_at = TimePoint(receipt_a.completed_at.tick + 1)
    probe_custody = RuntimeCustody.branch(
        custody,
        f"{replicate_id}:control:rejected-probe",
        replay_a,
    )
    probe_agent = make_branch_learning_agent(
        source_agent=collection_a.final_agent,
        at=probe_at,
        model_owner=runtime.owner,
        learner=probe,
        custody=probe_custody,
    )
    probe_before_snapshot = probe_agent.snapshot(probe_at)
    probe_before = capture_rejected_probe_state(
        runtime=runtime,
        source_custody=custody,
        probe_custody=probe_custody,
        probe_snapshot=probe_before_snapshot,
        at=probe_at,
        collection_controller=collection_controller_a,
        retained_learning_evidence=learner_a.last_evidence,
    )
    try:
        probe_agent.learn(probe_custody.replay.require_transitions(replay_a), at=probe_at)
    except Exception as error:
        if "different model digest" not in str(error):
            raise
    else:
        raise RuntimeError("predeclared rejected update probe unexpectedly committed")
    probe_after_snapshot = probe_agent.snapshot(probe_at)
    probe_after = capture_rejected_probe_state(
        runtime=runtime,
        source_custody=custody,
        probe_custody=probe_custody,
        probe_snapshot=probe_after_snapshot,
        at=probe_at,
        collection_controller=collection_controller_a,
        retained_learning_evidence=learner_a.last_evidence,
    )
    if probe_before != probe_after:
        raise RuntimeError("rejected update probe changed a live component")
    append_rejected_probe_row(
        rows,
        receipt_id=f"wm001:rejected_update_probe:receipt:{runtime.live_state_digest[:24]}",
        parameter_sha256=runtime.digest,
        model_version=runtime.version,
        live_state_sha256=runtime.live_state_digest,
        state_before=probe_before,
        state_after=probe_after,
        evidence_directory=output_directory,
    )

    validation_action_a_seed = rows.seed("predictive_validation_action", 0)
    validation_controller_a = UniformRandomController(validation_action_a_seed)
    validation_rng_a_start = validation_controller_a.rng_digest
    validation_actions_a_before = validation_controller_a.actions_emitted
    validation_a_started = utc_now()
    validation_a = collect_episodes(
        run_id=f"{replicate_id}:predictive-validation-a",
        task_id=TASK_A,
        episode_seeds=[
            rows.seed("predictive_validation_a_episode", index) for index in range(config.validation_episodes)
        ],
        controller_factory=lambda _: validation_controller_a,
        backend=runtime,
        custody=RuntimeCustody.create(f"{replicate_id}:validation-a"),
    )
    validation_a_completed = utc_now()
    append_collection_rows(
        rows,
        validation_a,
        split="predictive_validation_a",
        condition="validation_random",
        checkpoint_id="after_a",
        learning_allowed=False,
        replay_writes_allowed=False,
        started_at=validation_a_started,
        completed_at=validation_a_completed,
    )
    append_collection_policy_run(
        rows,
        validation_a,
        split="predictive_validation_a",
        condition="validation_random",
        checkpoint_id="after_a",
        controller=validation_controller_a,
        seed_namespace="predictive_validation_action",
        seed_index=0,
        seed=validation_action_a_seed,
        rng_start_sha256=validation_rng_a_start,
        actions_before=validation_actions_a_before,
    )
    validation_a_batch = to_transition_batch(validation_a.transitions)
    evaluation_a_backends = {
        condition: load_evaluated_checkpoint(
            rows,
            condition=condition,
            output_directory=output_directory,
            device=config.device,
        )
        for condition in (
            "cold",
            "frozen",
            "corrupted",
            "irrelevant",
            "after_a",
        )
    }
    for condition, backend in evaluation_a_backends.items():
        metrics = evaluate_mixture(
            backend.model,
            validation_a_batch,
            device=config.device,
        )
        append_predictive_metric(
            rows,
            metrics,
            runtime=backend,
            task_id=TASK_A,
            condition=condition,
            checkpoint_id=condition,
            split="predictive_validation_a",
            evidence_directory=output_directory,
        )

    behavior_a_seeds = [rows.seed("behavior_evaluation_a_episode", index) for index in range(config.behavior_episodes)]
    planner_seed = rows.seed("planner")
    print(f"[{utc_now()}] {replicate_id}: evaluate A controls", flush=True)
    for condition, backend in evaluation_a_backends.items():
        run_batched_cem_condition(
            rows,
            replicate_id=replicate_id,
            task_id=TASK_A,
            reset_seeds=behavior_a_seeds,
            condition=condition,
            checkpoint_id=condition,
            backend=backend,
            planner=CEMController(
                make_learned_model_env(backend.model, device=config.device),
                seed=planner_seed,
            ),
        )
    run_behavior_condition(
        rows,
        replicate_id=replicate_id,
        task_id=TASK_A,
        reset_seeds=behavior_a_seeds,
        condition="random",
        checkpoint_id="random",
        backend=evaluation_a_backends["cold"],
        controller=UniformRandomController(rows.seed("random_policy_action", 0)),
        seed_namespace="random_policy_action",
        seed_index=0,
    )
    oracle_backend = OraclePredictiveBackend()
    run_batched_cem_condition(
        rows,
        replicate_id=replicate_id,
        task_id=TASK_A,
        reset_seeds=behavior_a_seeds,
        condition="oracle",
        checkpoint_id="oracle",
        backend=oracle_backend,
        planner=CEMController(
            make_true_dynamics_env(device=config.device),
            seed=planner_seed,
        ),
    )

    reloaded_after_a = load_evaluated_checkpoint(
        rows,
        condition="after_a",
        output_directory=output_directory,
        device=config.device,
    )
    after_a_payload = reloaded_after_a.owner.snapshot_state().payload
    validation_action_b_seed = rows.seed("predictive_validation_action", 1)
    validation_controller_b = UniformRandomController(validation_action_b_seed)
    validation_rng_b_start = validation_controller_b.rng_digest
    validation_actions_b_before = validation_controller_b.actions_emitted
    validation_b_started = utc_now()
    validation_b = collect_episodes(
        run_id=f"{replicate_id}:predictive-validation-b",
        task_id=TASK_B,
        episode_seeds=[
            rows.seed("predictive_validation_b_episode", index) for index in range(config.validation_episodes)
        ],
        controller_factory=lambda _: validation_controller_b,
        backend=runtime,
        custody=RuntimeCustody.create(f"{replicate_id}:validation-b"),
    )
    validation_b_completed = utc_now()
    append_collection_rows(
        rows,
        validation_b,
        split="predictive_validation_b",
        condition="validation_random",
        checkpoint_id="after_a",
        learning_allowed=False,
        replay_writes_allowed=False,
        started_at=validation_b_started,
        completed_at=validation_b_completed,
    )
    append_collection_policy_run(
        rows,
        validation_b,
        split="predictive_validation_b",
        condition="validation_random",
        checkpoint_id="after_a",
        controller=validation_controller_b,
        seed_namespace="predictive_validation_action",
        seed_index=1,
        seed=validation_action_b_seed,
        rng_start_sha256=validation_rng_b_start,
        actions_before=validation_actions_b_before,
    )
    validation_b_batch = to_transition_batch(validation_b.transitions)
    before_b_metrics = evaluate_mixture(runtime.model, validation_b_batch, device=config.device)
    append_predictive_metric(
        rows,
        before_b_metrics,
        runtime=runtime,
        task_id=TASK_B,
        condition="after_a",
        checkpoint_id="after_a",
        split="predictive_validation_b",
        evidence_directory=output_directory,
    )

    learner_b_replay = TransactionalWorldModelLearner(
        phase="train_b_replay",
        bootstrap_seeds=[rows.seed("ensemble_bootstrap_b", index) for index in range(5)],
        minibatch_order_seed=rows.seed("minibatch_order_b"),
        optimizer_steps=config.optimizer_steps,
        balanced_tasks=True,
        device=config.device,
    )
    collection_action_b_seed = rows.seed("collection_action", 1)
    collection_controller_b = UniformRandomController(collection_action_b_seed)
    collection_rng_b_start = collection_controller_b.rng_digest
    collection_actions_b_before = collection_controller_b.actions_emitted
    b_started = utc_now()
    collection_b = collect_episodes(
        run_id=f"{replicate_id}:collect-b",
        task_id=TASK_B,
        episode_seeds=[rows.seed("collect_b_episode", index) for index in range(config.collection_episodes)],
        controller_factory=lambda _: collection_controller_b,
        backend=runtime,
        custody=custody,
        start_tick=receipt_a.completed_at.tick + 2,
        model_owner=runtime.owner,
        learner=learner_b_replay,
    )
    b_completed = utc_now()
    append_collection_rows(
        rows,
        collection_b,
        split="collect_b",
        condition="collection_random",
        checkpoint_id="after_a",
        learning_allowed=True,
        replay_writes_allowed=True,
        started_at=b_started,
        completed_at=b_completed,
    )
    append_collection_policy_run(
        rows,
        collection_b,
        split="collect_b",
        condition="collection_random",
        checkpoint_id="after_a",
        controller=collection_controller_b,
        seed_namespace="collection_action",
        seed_index=1,
        seed=collection_action_b_seed,
        rng_start_sha256=collection_rng_b_start,
        actions_before=collection_actions_b_before,
    )
    print(
        f"[{utc_now()}] {replicate_id}: collected B={len(collection_b.transitions)}; train replay/naive",
        flush=True,
    )
    naive = WorldModelRuntime.from_payload(after_a_payload, device=config.device)
    learner_b_naive = TransactionalWorldModelLearner(
        phase="train_b_naive",
        bootstrap_seeds=[rows.seed("ensemble_bootstrap_b", index) for index in range(5)],
        minibatch_order_seed=rows.seed("minibatch_order_b"),
        optimizer_steps=config.optimizer_steps,
        device=config.device,
    )
    replay_b = custody.replay.require_transitions(collection_b.transitions)
    naive_custody = RuntimeCustody.branch(
        custody,
        f"{replicate_id}:control:naive",
        (*replay_a, *replay_b),
    )
    naive_agent = make_branch_learning_agent(
        source_agent=collection_b.final_agent,
        at=TimePoint(collection_b.next_tick),
        model_owner=naive.owner,
        learner=learner_b_naive,
        custody=naive_custody,
    )
    replay_receipt = collection_b.final_agent.learn(
        custody.replay.require_transitions((*replay_a, *replay_b)),
        at=TimePoint(collection_b.next_tick),
    )
    naive_receipt = naive_agent.learn(
        naive_custody.replay.require_transitions(replay_b),
        at=TimePoint(collection_b.next_tick + 1),
    )
    if learner_b_naive.last_evidence is None or learner_b_replay.last_evidence is None:
        raise RuntimeError("B learners did not retain preparation evidence")
    append_update_row(
        rows,
        replay_receipt,
        learner_b_replay.last_evidence,
        phase="train_b_replay",
        eligible_splits=("collect_a", "collect_b"),
        committed_parameter_sha256=runtime.digest,
        committed_live_state_sha256=runtime.live_state_digest,
        manifest_directory=output_directory,
    )
    preserve_evaluated_checkpoint(
        rows,
        condition="after_b_replay",
        runtime=runtime,
        output_directory=output_directory,
    )
    preserve_evaluated_checkpoint(
        rows,
        condition="after_b_naive",
        runtime=naive,
        output_directory=output_directory,
    )
    append_update_row(
        rows,
        naive_receipt,
        learner_b_naive.last_evidence,
        phase="train_b_naive",
        eligible_splits=("collect_b",),
        committed_parameter_sha256=naive.digest,
        committed_live_state_sha256=naive.live_state_digest,
        manifest_directory=output_directory,
    )

    evaluation_b_backends = {
        condition: load_evaluated_checkpoint(
            rows,
            condition=condition,
            output_directory=output_directory,
            device=config.device,
        )
        for condition in ("after_b_replay", "after_b_naive")
    }
    for condition, backend in evaluation_b_backends.items():
        metrics = evaluate_mixture(
            backend.model,
            validation_b_batch,
            device=config.device,
        )
        append_predictive_metric(
            rows,
            metrics,
            runtime=backend,
            task_id=TASK_B,
            condition=condition,
            checkpoint_id=condition,
            split="predictive_validation_b",
            evidence_directory=output_directory,
        )
        a_metrics = evaluate_mixture(
            backend.model,
            validation_a_batch,
            device=config.device,
        )
        append_predictive_metric(
            rows,
            a_metrics,
            runtime=backend,
            task_id=TASK_A,
            condition=condition,
            checkpoint_id=condition,
            split="predictive_validation_a",
            evidence_directory=output_directory,
        )

    print(f"[{utc_now()}] {replicate_id}: evaluate B plasticity and A retention", flush=True)
    behavior_b_seeds = [rows.seed("behavior_evaluation_b_episode", index) for index in range(config.behavior_episodes)]
    for task_id, seeds, conditions in (
        (
            TASK_B,
            behavior_b_seeds,
            (
                (
                    "after_a",
                    load_evaluated_checkpoint(
                        rows,
                        condition="after_a",
                        output_directory=output_directory,
                        device=config.device,
                    ),
                ),
                ("after_b_replay", evaluation_b_backends["after_b_replay"]),
                ("after_b_naive", evaluation_b_backends["after_b_naive"]),
            ),
        ),
        (
            TASK_A,
            behavior_a_seeds,
            (
                ("after_b_replay", evaluation_b_backends["after_b_replay"]),
                ("after_b_naive", evaluation_b_backends["after_b_naive"]),
            ),
        ),
    ):
        for condition, backend in conditions:
            run_batched_cem_condition(
                rows,
                replicate_id=replicate_id,
                task_id=task_id,
                reset_seeds=seeds,
                condition=condition,
                checkpoint_id=condition,
                backend=backend,
                planner=CEMController(
                    make_learned_model_env(backend.model, device=config.device),
                    seed=planner_seed,
                ),
            )

    run_behavior_condition(
        rows,
        replicate_id=replicate_id,
        task_id=TASK_B,
        reset_seeds=behavior_b_seeds,
        condition="random",
        checkpoint_id="random",
        backend=load_evaluated_checkpoint(
            rows,
            condition="after_a",
            output_directory=output_directory,
            device=config.device,
        ),
        controller=UniformRandomController(rows.seed("random_policy_action", 1)),
        seed_namespace="random_policy_action",
        seed_index=1,
    )
    run_batched_cem_condition(
        rows,
        replicate_id=replicate_id,
        task_id=TASK_B,
        reset_seeds=behavior_b_seeds,
        condition="oracle",
        checkpoint_id="oracle",
        backend=oracle_backend,
        planner=CEMController(
            make_true_dynamics_env(device=config.device),
            seed=planner_seed,
        ),
    )

    checkpoint_path = output_directory / f"{replicate_id}-after-b-replay.checkpoint"
    restart_planner = CEMController(
        make_learned_model_env(runtime.model, device=config.device),
        seed=planner_seed,
    )
    manifest_sha256, component_rows, restart_tick = save_replicate_checkpoint(
        path=checkpoint_path,
        rows=rows,
        runtime=runtime,
        collection=collection_b,
        collection_a=collection_a.transitions,
        collection_b=collection_b.transitions,
        collection_rng_states={
            "task_a": collection_controller_a.state(),
            "task_b": collection_controller_b.state(),
        },
        planner=restart_planner,
        update_evidence=(
            learner_a.last_evidence,
            learner_b_replay.last_evidence,
        ),
        created_at=replay_receipt.completed_at,
    )
    rows.checkpoint_components.extend(component_rows)
    print(f"[{utc_now()}] {replicate_id}: fresh-process restore parity", flush=True)
    rows.restart_parity = run_restart_parity(
        checkpoint_path=checkpoint_path,
        output_directory=output_directory,
        rows=rows,
        runtime=runtime,
        custody=collection_b.custody,
        next_tick=restart_tick,
        planner=restart_planner,
        checkpoint_manifest_sha256=manifest_sha256,
        component_rows=component_rows,
        task_reset_seeds={
            TASK_A: rows.seed("behavior_evaluation_a_episode", 0),
            TASK_B: rows.seed("behavior_evaluation_b_episode", 0),
        },
        device=config.device,
        boundary_snapshot=collection_b.final_agent.snapshot(TimePoint(restart_tick - 1)),
    )
    if rows.restart_parity["checkpoint_manifest_sha256"] != manifest_sha256:
        raise RuntimeError("restart parity reported a different checkpoint manifest")

    if cold.digest != cold_parameter_sha256 or cold.live_state_digest != cold_live_state_sha256:
        raise RuntimeError("frozen cold checkpoint mutated during the replicate")
    result = rows.as_dict()
    partial_path = output_directory / f"{replicate_id}.json"
    atomic_write_exclusive(partial_path, canonical_json_bytes(result) + b"\n")
    print(f"[{utc_now()}] {replicate_id}: complete", flush=True)
    return result


def run_experiment(
    config: ExperimentConfig,
    *,
    output_directory: Path,
    formal_binding_path: Path | None = None,
    output_prepared: bool = False,
) -> tuple[dict[str, object], Path]:
    """Execute all configured replicates and write one schema-valid raw result."""

    from .analysis import analyze_result
    from .verify import verify_result

    config.validate()
    bootstrap_custody = _verify_live_bootstrap_custody()
    if config.lane == "formal" and not output_prepared:
        raise ValueError("formal execution requires an exclusive ProducerAttempt directory")
    if config.lane == "formal":
        if formal_binding_path is None:
            raise ValueError("formal execution requires its pre-run implementation binding")
        if (
            cast(dict[str, object], bootstrap_custody["runtime_seal"]).get("schema")
            != "prospect.world-model-lifecycle.formal-binding.v10"
            or formal_binding_path.read_bytes() != bootstrap_custody["runtime_seal_payload"]
        ):
            raise ValueError("formal binding differs from the pre-import runtime seal")
    elif formal_binding_path is not None:
        raise ValueError("development execution must not claim a formal binding")
    elif cast(dict[str, object], bootstrap_custody["runtime_seal"]).get("schema") != "prospect.wm001.runtime-seal.v1":
        raise ValueError("development execution requires a prospective runtime seal")
    if output_prepared:
        if not output_directory.is_dir():
            raise ValueError("prepared output directory does not exist")
        metadata_path = output_directory / "attempt-metadata.json"
        if not metadata_path.is_file():
            raise ValueError("prepared output directory has no custody metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != "prospect.wm001.producer-attempt.v1"
            or metadata.get("lane") != config.lane
        ):
            raise ValueError("prepared output directory custody metadata differs from the lane")
        if (output_directory / "producer-manifest.json").exists():
            raise ValueError("prepared output directory was already finalized")
    elif not output_prepared:
        output_directory.mkdir(parents=True, exist_ok=False)
    formal_launch_sha256: str | None = None
    if config.lane == "formal":
        assert formal_binding_path is not None
        from .artifact import claim_formal_launch_with_digest

        formal_preflight_reports = _run_formal_preflight(
            formal_binding_path,
            device=config.device,
        )
        conformance, oscillator_conformance = _validate_formal_preflight_reports(
            formal_preflight_reports,
            binding_path=formal_binding_path,
        )
        _recheck_live_bootstrap_custody(bootstrap_custody)
        _, formal_launch_sha256 = claim_formal_launch_with_digest(
            formal_binding_path,
            output_directory,
        )
    started_at = utc_now()
    monotonic_start = time.monotonic()
    if config.lane == "development":
        conformance = cast(
            dict[str, Any],
            run_pendulum_conformance(samples_per_task=128, seed=20260717),
        )
        if conformance.get("passed") is not True:
            raise RuntimeError("Pendulum semantic conformance preflight failed")
        oscillator_conformance = cast(
            dict[str, Any],
            run_independent_phase_oscillator_conformance(
                cases=32,
                seed=20260718,
            ),
        )
        if oscillator_conformance.get("passed") is not True:
            raise RuntimeError("independent oscillator conformance preflight failed")
        _recheck_live_bootstrap_custody(bootstrap_custody)
    replicate_results: list[dict[str, object]] = []
    for master_seed in config.master_seeds:
        replicate = run_replicate(
            config,
            master_seed=master_seed,
            output_directory=output_directory,
        )
        replicate_results.append(replicate)
        completed_master_seeds: list[int] = []
        for row in replicate_results:
            completed_seed = row.get("master_seed")
            if type(completed_seed) is not int:
                raise RuntimeError("replicate result has no integer master seed")
            completed_master_seeds.append(completed_seed)
        replicate_seed = replicate.get("master_seed")
        if type(replicate_seed) is not int:
            raise RuntimeError("replicate result has no integer master seed")
        progress = {
            "schema": "prospect.wm001.progress.v1",
            "lane": config.lane,
            "completed_master_seeds": completed_master_seeds,
            "updated_at_utc": utc_now(),
        }
        progress_path = output_directory / (f"progress-{len(replicate_results):02d}-{replicate_seed}.json")
        atomic_write_exclusive(progress_path, canonical_json_bytes(progress) + b"\n")

    git_commit = _git_value("rev-parse", "HEAD")
    git_tree = _git_value("rev-parse", "HEAD^{tree}")
    worktree_clean = not _git_value("status", "--short", "--untracked-files=all")
    from .binding import (
        FORMAL_PROCESS_ENVIRONMENT_KEYS,
        LOCKFILE,
        _cuda_driver_version,
        python_flag_identity,
    )

    lockfile = LOCKFILE
    binding_sha256 = (
        None if formal_binding_path is None else hashlib.sha256(formal_binding_path.read_bytes()).hexdigest()
    )
    wall_seconds = time.monotonic() - monotonic_start
    max_cuda_memory = int(torch.cuda.max_memory_allocated()) if config.device == "cuda" else 0
    bootstrap_custody = _verify_live_bootstrap_custody()
    runtime_seal = cast(dict[str, object], bootstrap_custody["runtime_seal"])
    closure_block = (
        cast(dict[str, object], runtime_seal.get("dependencies", {}))
        if runtime_seal.get("schema") == "prospect.world-model-lifecycle.formal-binding.v10"
        else runtime_seal
    )
    package_root_inventories = closure_block.get("package_roots")
    package_ownership_identity = closure_block.get("package_ownership")
    standard_library_inventory = closure_block.get("standard_library")
    if (
        not isinstance(package_root_inventories, list)
        or len(package_root_inventories) != 1
        or not isinstance(package_root_inventories[0], dict)
        or not isinstance(package_ownership_identity, dict)
        or not isinstance(standard_library_inventory, dict)
    ):
        raise RuntimeError("captured runtime seal has malformed closure inventories")
    result: dict[str, object] = {
        "schema": "prospect.world-model-lifecycle.raw-result.v9",
        "experiment_id": "WM-001",
        "protocol_version": "1.19.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "lane": config.lane,
        "claim_eligible": config.lane == "formal",
        "formal_binding_sha256": binding_sha256,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "execution": {
            "git_commit": git_commit,
            "git_tree": git_tree,
            "worktree_clean": worktree_clean,
            "dependency_lock_sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest(),
            "python_executable": sys.executable,
            "python_executable_sha256": hashlib.sha256(
                Path(sys.executable).resolve(strict=True).read_bytes()
            ).hexdigest(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": config.device,
            "python_flags": python_flag_identity(),
            "process_environment": {
                key: value for key, value in sorted(os.environ.items()) if key in FORMAL_PROCESS_ENVIRONMENT_KEYS
            },
            "accelerator": (torch.cuda.get_device_name(0) if config.device == "cuda" else None),
            "thread_count": torch.get_num_threads(),
            "interop_thread_count": torch.get_num_interop_threads(),
            "cuda_runtime": torch.version.cuda,
            "cuda_driver": (_cuda_driver_version() if config.device == "cuda" else None),
            "cublas_workspace_config": (os.environ.get("CUBLAS_WORKSPACE_CONFIG") if config.device == "cuda" else None),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "runtime_seal_sha256": bootstrap_custody["runtime_seal_sha256"],
            "runtime_seal_descriptor_custody": True,
            "producer_bootstrap_sha256": bootstrap_custody["bootstrap_sha256"],
            "bootstrap_descriptor_custody": True,
            "package_roots": package_root_inventories,
            "package_ownership": package_ownership_identity,
            "standard_library": standard_library_inventory,
            "launcher_process_id": os.getpid(),
            "formal_launch_file": ("formal-launch.json" if config.lane == "formal" else None),
            "formal_launch_sha256": formal_launch_sha256,
            "resource_telemetry": {
                "wall_seconds": wall_seconds,
                "max_cuda_memory_bytes": max_cuda_memory,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "validation_conformance_cases": conformance["cases"],
                "validation_conformance_observation_max_abs_error": (conformance["max_observation_absolute_error"]),
                "validation_conformance_reward_max_abs_error": (conformance["max_reward_absolute_error"]),
                "validation_conformance_planner_dtype": conformance["planner_dtype"],
                "validation_conformance_planner_observation_max_abs_error": (
                    conformance["max_planner_observation_absolute_error"]
                ),
                "validation_conformance_planner_reward_max_abs_error": (
                    conformance["max_planner_reward_absolute_error"]
                ),
                "oscillator_conformance_cases": oscillator_conformance["cases"],
                "oscillator_conformance_trajectory_sha256": (oscillator_conformance["trajectory_sha256"]),
            },
        },
        "replicates": replicate_results,
        "aggregate_metrics": [],
        "gate_results": [],
        "limitations": [
            (
                "Development runs are diagnostic and cannot support the sealed capability claim."
                if config.lane == "development"
                else "WM-001 is limited to same-machine, same-dependency Pendulum-v1 evidence."
            ),
            "Uncertainty is evaluated predictively but is not used as an intrinsic planning reward in WM-001.",
            "Replay retention is demonstrated on two observed-context actuator regimes, not broad continual learning.",
        ],
    }
    analysis = analyze_result(result)
    result["aggregate_metrics"] = analysis["aggregate_metrics"]
    result["gate_results"] = analysis["gate_results"]
    findings = cast(list[dict[str, object]], analysis["audit_findings"])
    result["limitations"] = [
        *cast(list[str], result["limitations"]),
        *[
            f"Analysis finding [{finding.get('severity', 'unknown')}]: {finding.get('message', finding)}"
            for finding in findings
        ],
    ]
    result_path = output_directory / "result.json"
    atomic_write_exclusive(result_path, canonical_json_bytes(result) + b"\n")
    verify_result(result_path, formal_binding_path)
    return result, result_path


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=Path.cwd().resolve(strict=True),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


__all__ = (
    "ExperimentConfig",
    "IsolatedConformanceReports",
    "OraclePredictiveBackend",
    "ReplicateRows",
    "append_collection_rows",
    "append_episode_rows",
    "append_predictive_metric",
    "configure_determinism",
    "run_experiment",
    "run_replicate",
    "to_transition_batch",
)
