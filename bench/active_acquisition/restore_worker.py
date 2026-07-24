"""Fresh-process, fresh-component WM-002 Q1 checkpoint terminal replay."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import cast

from bench.active_acquisition.checkpoint import (
    QualificationBinding,
    load_q1_checkpoint,
)
from bench.active_acquisition.contracts import (
    Q1_PROTOCOL_VERSION,
    canonical_json_bytes,
    load_canonical_json,
)
from bench.active_acquisition.problem import TerminalAction
from bench.active_acquisition.q1 import (
    Q1ExecutionAuthority,
    Q1ExecutionError,
    _execution_id,
    _mapping,
    _number,
    _private_binary_reader,
    _private_binary_writer,
    _read_checkpoint_frame_stream,
    _read_jsonl,
    _require_local_worker_capability,
    _require_private_directory,
    _require_started_child_attempt,
    _terminal_candidate_digest,
    _terminal_candidate_rows,
    _validate_artifact_fast,
    validate_q1_child_execution_authority,
)
from bench.active_acquisition.runtime_lane import (
    AGENT_ID,
    ArmMode,
    HiddenActuatorTerminalEnvironment,
    compose_restored_terminal_runtime,
    decode_posterior_model,
    validate_posterior_model_state,
)
from bench.active_acquisition.seeding import (
    EPISODES_PER_MASTER,
    PrivateQ1SeedSchedule,
)
from bench.active_acquisition.worker_capability import (
    Q1WorkerCapability,
    consume_worker_capability_fd,
)
from prospect.domain import TimePoint
from prospect.runtime import InteractionContext


def restore_q1_lane(
    *,
    capability: Q1WorkerCapability,
    worker_capability_sha256: str,
) -> None:
    """Restore exactly 1,024 frames for one authenticated master/arm launch."""

    _require_local_worker_capability(
        capability,
        role="restore",
        worker_capability_sha256=worker_capability_sha256,
    )
    paths = dict(capability.paths)
    master = capability.master
    arm = ArmMode(cast(str, capability.arm))
    frame_path = Path(cast(str, paths["frame_path"]))
    index_path = Path(cast(str, paths["index_path"]))
    raw_trace_path = Path(cast(str, paths["raw_trace_path"]))
    output_path = Path(cast(str, paths["output_path"]))
    master_directory = Path(cast(str, paths["master_directory"]))
    master_directory_identity = _require_private_directory(
        master_directory,
        label="restore master directory",
    )
    if any(path.parent != master_directory for path in (frame_path, index_path, raw_trace_path, output_path)):
        raise Q1ExecutionError("restore lane path lies outside its bound master directory")
    attempt_registry_directory = Path(cast(str, paths["attempt_registry_directory"]))
    _require_started_child_attempt(
        attempt_registry_directory=attempt_registry_directory,
        expected_run_identity=capability.run_identity,
        expected_worker_capability_sha256=worker_capability_sha256,
    )
    authority = validate_q1_child_execution_authority(
        entry_report_path=Path(cast(str, paths["entry_report_path"])),
        q0_report_path=Path(cast(str, paths["q0_report_path"])),
        secret_salt_path=Path(cast(str, paths["secret_salt_path"])),
        prospective_review_path=Path(cast(str, paths["prospective_review_path"])),
        execution_root=Path(cast(str, paths["execution_root"])),
        attempt_registry_directory=attempt_registry_directory,
    )
    if authority.run_identity != capability.run_identity:
        raise Q1ExecutionError("child run/attempt identity differs from fresh derivation")
    schedule = PrivateQ1SeedSchedule(authority.secret_salt)
    indexes = _read_jsonl(index_path)
    raw_rows = (
        row for row in _read_jsonl(raw_trace_path) if row.get("master") == master and row.get("arm") == arm.value
    )
    episode_count = 0
    with _private_binary_writer(output_path) as output, _private_binary_reader(frame_path) as frame_stream:
        for expected_episode, (index, raw) in enumerate(zip(indexes, raw_rows, strict=True)):
            if (
                index.get("master") != master
                or index.get("arm") != arm.value
                or index.get("episode") != expected_episode
                or raw.get("episode") != expected_episode
            ):
                raise Q1ExecutionError("restore lane order or semantic identity mismatch")
            payload = _read_checkpoint_frame_stream(
                frame_stream,
                frame_offset=cast(int, index["frame_offset"]),
                frame_length=cast(int, index["frame_length"]),
            )
            restored = restore_one(
                authority=authority,
                schedule=schedule,
                master=master,
                arm=arm,
                episode=expected_episode,
                checkpoint_payload=payload,
                checkpoint_index=index,
                raw_trace=raw,
            )
            output.write(canonical_json_bytes(restored, newline=True))
            episode_count += 1
        if episode_count != EPISODES_PER_MASTER:
            raise Q1ExecutionError("restore lane does not contain exactly 1,024 episodes")
        if frame_stream.tell() != os.fstat(frame_stream.fileno()).st_size:
            raise Q1ExecutionError("restore lane checkpoint sidecar contains trailing bytes")
    _require_private_directory(
        master_directory,
        expected_identity=master_directory_identity,
        label="restore master directory",
    )


def restore_one(
    *,
    authority: Q1ExecutionAuthority,
    schedule: PrivateQ1SeedSchedule,
    master: int,
    arm: ArmMode,
    episode: int,
    checkpoint_payload: bytes,
    checkpoint_index: Mapping[str, object],
    raw_trace: Mapping[str, object],
) -> Mapping[str, object]:
    """Construct a fresh graph from one frame and replay its terminal action."""

    if schedule.salt_commitment_sha256 != authority.salt_commitment_sha256:
        raise Q1ExecutionError("restore schedule differs from its entry authority")
    theta = schedule.theta(master, episode)
    return _restore_one(
        master=master,
        arm=arm,
        episode=episode,
        checkpoint_payload=checkpoint_payload,
        checkpoint_index=checkpoint_index,
        raw_trace=raw_trace,
        binding=authority.checkpoint_binding,
        terminal_success=lambda decision: schedule.terminal_success(
            master,
            episode,
            decision,
            theta,
        ),
        privacy_schedule=schedule,
    )


def _restore_one(
    *,
    master: int,
    arm: ArmMode,
    episode: int,
    checkpoint_payload: bytes,
    checkpoint_index: Mapping[str, object],
    raw_trace: Mapping[str, object],
    binding: QualificationBinding,
    terminal_success: Callable[[TerminalAction], bool],
    privacy_schedule: PrivateQ1SeedSchedule | None,
) -> Mapping[str, object]:
    _validate_artifact_fast("checkpoint_frame", checkpoint_index)
    _validate_artifact_fast("raw_trace", raw_trace)
    expected_index: dict[str, object] = {
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "entry_qualification_sha256": binding.entry_qualification_sha256,
        "master": master,
        "arm": arm.value,
        "episode": episode,
    }
    for name, expected in expected_index.items():
        if checkpoint_index.get(name) != expected:
            raise Q1ExecutionError(f"checkpoint index binding mismatch: {name}")
    expected_raw: dict[str, object] = {
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "protocol_sha256": binding.protocol_sha256,
        "implementation_sha256": binding.implementation_sha256,
        "q0_report_sha256": binding.q0_report_sha256,
        "entry_qualification_sha256": binding.entry_qualification_sha256,
        "salt_commitment_sha256": binding.salt_commitment_sha256,
        "master": master,
        "arm": arm.value,
        "episode": episode,
    }
    for name, expected in expected_raw.items():
        if raw_trace.get(name) != expected:
            raise Q1ExecutionError(f"raw trace binding mismatch: {name}")
    producer_pid = raw_trace.get("producer_pid")
    if isinstance(producer_pid, bool) or not isinstance(producer_pid, int) or producer_pid <= 0:
        raise Q1ExecutionError("raw trace producer PID is invalid")
    if producer_pid == os.getpid():
        raise Q1ExecutionError("restore interpreter PID must differ from producer PID")

    restored = load_q1_checkpoint(
        checkpoint_payload,
        expected_agent_id=AGENT_ID,
        expected_aggregate_sha256=cast(str, checkpoint_index["checkpoint_sha256"]),
        expected_binding=binding,
        model_validator=validate_posterior_model_state,
    )
    episode_id = f"{binding.run_id}:m{master}:a{arm.value}:e{episode}"
    expected_checkpoint_id = f"{episode_id}:checkpoint:preterminal"
    if restored.report.checkpoint_id != expected_checkpoint_id:
        raise Q1ExecutionError("restored checkpoint ID differs from its semantic episode")
    checkpoint = _mapping(raw_trace["checkpoint"], "raw checkpoint trace")
    component_sha256 = _mapping(
        checkpoint_index["component_sha256"],
        "checkpoint component digests",
    )
    actual_component_sha256 = {row.component_id: row.sha256 for row in restored.report.component_digests}
    if dict(component_sha256) != actual_component_sha256:
        raise Q1ExecutionError("checkpoint index component digests differ from the restored bundle")
    if (
        checkpoint.get("sha256") != restored.report.aggregate_sha256
        or checkpoint.get("size_bytes") != restored.report.size_bytes
        or checkpoint.get("component_sha256") != actual_component_sha256
    ):
        raise Q1ExecutionError("raw checkpoint identity differs from the restored bundle")
    runtime = compose_restored_terminal_runtime(
        episode_id=episode_id,
        arm=arm,
        restored_at=restored.snapshot.captured_at,
        identities=restored.identity_source,
        state=restored.agent_state,
        store=restored.experience_store,
        ledger=restored.ledger,
        model_owner=restored.model_owner,
    )
    model_state = restored.model_owner.snapshot_state()
    posterior = decode_posterior_model(model_state.payload)
    posterior_string = (
        str(posterior.posterior_direct.numerator)
        if posterior.posterior_direct.denominator == 1
        else f"{posterior.posterior_direct.numerator}/{posterior.posterior_direct.denominator}"
    )
    acquisition = _mapping(raw_trace["acquisition"], "raw acquisition trace")
    if (
        acquisition.get("model_after_sha256") != model_state.digest
        or acquisition.get("model_after_version") != model_state.version
        or acquisition.get("posterior_after") != posterior_string
    ):
        raise Q1ExecutionError("raw acquisition model identity differs from the restored bundle")
    selected_terminal = (
        TerminalAction.DIRECT if posterior.posterior_direct >= Fraction(1, 2) else TerminalAction.REVERSED
    )
    environment = HiddenActuatorTerminalEnvironment(
        agent_id=runtime.agent_id,
        terminal_success=terminal_success(selected_terminal),
        identities=runtime.identities,
    )
    terminal = runtime.terminal_agent.interact(
        environment,
        runtime.terminal_goal,
        context=InteractionContext(
            run_id=binding.run_id,
            task_id=runtime.terminal_goal.task_id,
            episode_id=episode_id,
            step_index=1,
        ),
        decide_at=TimePoint(restored.receipt.completed_at.tick + 1),
    )
    terminal_outcome = _mapping(
        terminal.experience.outcome.evidence.payload,
        "restored terminal outcome",
    )
    if terminal_outcome.get("executed_terminal_action") != int(selected_terminal):
        raise Q1ExecutionError("restored terminal action differs from exact posterior")
    terminal_reward = _number(
        terminal_outcome["terminal_reward"],
        "restored terminal reward",
    )
    episode_return = (
        restored.accumulator.task_payoff
        - restored.accumulator.physical_action_cost
        - restored.accumulator.information_acquisition_cost
        + terminal_reward
    )
    terminal_candidate_rows = _terminal_candidate_rows(terminal)
    row: dict[str, object] = {
        "schema": "prospect.wm002.active-acquisition.q1-restored-trace.v1",
        "protocol_version": Q1_PROTOCOL_VERSION,
        "run_id": binding.run_id,
        "attempt_id": binding.attempt_id,
        "protocol_sha256": binding.protocol_sha256,
        "implementation_sha256": binding.implementation_sha256,
        "entry_qualification_sha256": binding.entry_qualification_sha256,
        "q0_report_sha256": binding.q0_report_sha256,
        "salt_commitment_sha256": binding.salt_commitment_sha256,
        "master": master,
        "arm": arm.value,
        "episode": episode,
        "producer_pid": producer_pid,
        "restorer_pid": os.getpid(),
        "checkpoint_sha256": checkpoint_index["checkpoint_sha256"],
        "component_sha256": component_sha256,
        "model_sha256": model_state.digest,
        "model_version": model_state.version,
        "configuration_version": restored.snapshot.configuration_version,
        "posterior_direct": posterior_string,
        "acquisition_experience_id": restored.experience.experience_id,
        "acquisition_transition_id": restored.transition.transition_id,
        "acquisition_receipt_id": restored.receipt.receipt_id,
        "identity_counter_sha256": component_sha256["identity_counter"],
        "terminal_candidate_rows": terminal_candidate_rows,
        "terminal_candidate_rows_sha256": _terminal_candidate_digest(terminal),
        "selected_terminal_action": int(selected_terminal),
        "terminal_success": terminal_outcome["success"],
        "episode_return": episode_return,
        "terminal_decision_id": terminal.decision.decision_id,
        "terminal_intention_id": terminal.decision.intended_action.intention_id,
        "terminal_execution_id": _execution_id(terminal),
        "terminal_observation_id": terminal.experience.observation.observation_id,
        "terminal_outcome_id": terminal.experience.outcome.outcome_id,
        "terminal_experience_id": terminal.experience.experience_id,
        "terminal_transition_id": terminal.transition.transition_id,
    }
    if (
        restored.experience.experience_id != acquisition.get("experience_id")
        or restored.transition.transition_id != acquisition.get("transition_id")
        or restored.receipt.receipt_id != acquisition.get("receipt_id")
    ):
        raise Q1ExecutionError("restored acquisition ancestry differs from raw trace")
    _validate_artifact_fast("restored_trace", row)
    if privacy_schedule is not None:
        private = privacy_schedule.private_audit_material(master, episode, arm.value)
        privacy_schedule.assert_public_artifact(
            row,
            private_hmac_digests=private.private_hmac_digests(),
        )
    return row


def run_development_restore(*, spec_path: Path, output_path: Path) -> None:
    """Replay one explicit synthetic frame without a Q1 schedule or private draw."""

    spec = _mapping(load_canonical_json(spec_path), "development restore spec")
    if spec.get("schema") != "prospect.wm002.q1-development-restore-input.v1":
        raise Q1ExecutionError("development restore spec schema mismatch")
    raw = _mapping(spec["raw_trace"], "development raw trace")
    index = _mapping(spec["checkpoint_index"], "development checkpoint index")
    binding = _decode_binding(_mapping(spec["binding"], "development checkpoint binding"))
    terminal_success = spec.get("synthetic_terminal_success")
    if type(terminal_success) is not bool:
        raise Q1ExecutionError("synthetic terminal success must be an exact boolean")
    frame_path = Path(cast(str, spec["frame_path"]))
    with _private_binary_reader(frame_path) as frame_stream:
        payload = _read_checkpoint_frame_stream(
            frame_stream,
            frame_offset=cast(int, spec["frame_offset"]),
            frame_length=cast(int, spec["frame_length"]),
        )
    row = _restore_one(
        master=cast(int, raw["master"]),
        arm=ArmMode(cast(str, raw["arm"])),
        episode=cast(int, raw["episode"]),
        checkpoint_payload=payload,
        checkpoint_index=index,
        raw_trace=raw,
        binding=binding,
        terminal_success=lambda _decision: cast(bool, terminal_success),
        privacy_schedule=None,
    )
    with _private_binary_writer(output_path) as output:
        output.write(canonical_json_bytes(row, newline=True))


def _decode_binding(value: Mapping[str, object]) -> QualificationBinding:
    expected = {
        "schema",
        "protocol_version",
        "run_id",
        "attempt_id",
        "protocol_sha256",
        "implementation_sha256",
        "q0_report_sha256",
        "entry_qualification_sha256",
        "salt_commitment_sha256",
    }
    if set(value) != expected:
        raise Q1ExecutionError("development checkpoint binding shape mismatch")
    return QualificationBinding(
        protocol_version=cast(str, value["protocol_version"]),
        protocol_sha256=cast(str, value["protocol_sha256"]),
        implementation_sha256=cast(str, value["implementation_sha256"]),
        q0_report_sha256=cast(str, value["q0_report_sha256"]),
        entry_qualification_sha256=cast(
            str,
            value["entry_qualification_sha256"],
        ),
        salt_commitment_sha256=cast(str, value["salt_commitment_sha256"]),
        run_id=cast(str, value["run_id"]),
        attempt_id=cast(str, value["attempt_id"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    q1 = subparsers.add_parser("q1")
    q1.add_argument("--capability-fd", type=int, required=True)
    development = subparsers.add_parser("development")
    development.add_argument("--spec-path", type=Path, required=True)
    development.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "development":
        run_development_restore(
            spec_path=args.spec_path,
            output_path=args.output_path,
        )
        return 0
    decoded = consume_worker_capability_fd(
        args.capability_fd,
        expected_parent_pid=os.getppid(),
        expected_child_pid=os.getpid(),
    )
    restore_q1_lane(
        capability=decoded.capability,
        worker_capability_sha256=decoded.worker_capability_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "restore_one",
    "restore_q1_lane",
    "run_development_restore",
)
