from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

import bench.active_acquisition.q1 as q1
from bench.active_acquisition.contracts import canonical_json_bytes, validate_artifact
from bench.active_acquisition.seeding import Q1ExecutionMode
from bench.active_acquisition.worker_capability import (
    DecodedWorkerCapability,
    consume_worker_capability_fd,
)

_WORKER_CAPABILITY_SECRET = b"w" * 32
_WORKER_CAPABILITY_SHA256 = hashlib.sha256(_WORKER_CAPABILITY_SECRET).hexdigest()


def _start_capability_consumer(
    descriptor: int,
    *,
    parent_pid: int,
    child_pid: int,
) -> tuple[threading.Thread, list[DecodedWorkerCapability], list[BaseException]]:
    inherited = os.dup(descriptor)
    decoded: list[DecodedWorkerCapability] = []
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            decoded.append(
                consume_worker_capability_fd(
                    inherited,
                    expected_parent_pid=parent_pid,
                    expected_child_pid=child_pid,
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    return thread, decoded, errors


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _enabled_protocol_bytes(**extra: object) -> bytes:
    value: dict[str, object] = {
        "schema": "prospect.wm002.active-acquisition.q1-protocol.v1",
        "experiment": {
            "protocol_version": q1.Q1_PROTOCOL_VERSION,
            "claim_eligible": False,
            "formal_authorized": False,
            "execution_authorized": True,
        },
    }
    value.update(extra)
    return canonical_json_bytes(value, newline=True)


def test_result_free_development_probe_covers_all_schemas_and_fresh_restore() -> None:
    probe = q1.run_development_qualification_probe(
        protocol_sha256=_digest("protocol"),
        implementation_sha256=_digest("implementation"),
        q0_report_sha256=_digest("q0"),
        salt_commitment_sha256=_digest("salt"),
    )

    assert probe.violations == ()
    assert probe.synthetic_interactions == 19
    assert set(probe.artifact_samples) == {
        "aggregate",
        "audit_output",
        "checkpoint_frame",
        "private_audit",
        "raw_trace",
        "restored_trace",
    }
    for name, sample in probe.artifact_samples.items():
        validate_artifact(name, sample)

    raw = probe.artifact_samples["raw_trace"]
    restored = probe.artifact_samples["restored_trace"]
    assert isinstance(raw, dict)
    assert isinstance(restored, dict)
    assert restored["producer_pid"] != restored["restorer_pid"]
    assert restored["checkpoint_sha256"] == raw["checkpoint"]["sha256"]
    assert restored["terminal_candidate_rows"] == raw["terminal"]["candidate_rows"]
    assert restored["terminal_candidate_rows_sha256"] == raw["terminal"]["candidate_rows_sha256"]
    assert restored["selected_terminal_action"] == raw["terminal"]["selected_action"]
    assert restored["terminal_success"] == raw["terminal"]["success"]
    assert restored["episode_return"] == raw["terminal"]["episode_return"]
    assert restored["terminal_decision_id"] == raw["terminal"]["decision_id"]
    assert restored["terminal_transition_id"] == raw["terminal"]["transition_id"]


def test_checkpoint_binary_frame_round_trip_and_prefix_validation(tmp_path: Path) -> None:
    path = tmp_path / "frames.bin"
    with q1._private_binary_writer(path) as stream:
        first = q1.append_checkpoint_frame(stream, b"first")
        second = q1.append_checkpoint_frame(stream, b"second-frame")

    assert first.frame_offset == 0
    assert first.frame_length == 5
    assert second.frame_offset == 8 + 5
    assert (
        q1.read_checkpoint_frame(
            path,
            frame_offset=first.frame_offset,
            frame_length=first.frame_length,
        )
        == b"first"
    )
    assert (
        q1.read_checkpoint_frame(
            path,
            frame_offset=second.frame_offset,
            frame_length=second.frame_length,
        )
        == b"second-frame"
    )

    with pytest.raises(q1.Q1ExecutionError, match="length prefix"):
        q1.read_checkpoint_frame(
            path,
            frame_offset=first.frame_offset,
            frame_length=first.frame_length + 1,
        )
    with pytest.raises(q1.Q1ExecutionError, match="header is truncated"):
        q1.read_checkpoint_frame(
            path,
            frame_offset=path.stat().st_size,
            frame_length=1,
        )


def test_streaming_readers_reject_oversized_rows_and_frames_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q1, "_MAX_JSONL_ROW_BYTES", 7)
    monkeypatch.setattr(q1, "_MAX_CHECKPOINT_FRAME_BYTES", 7)

    frames_path = tmp_path / "frames.bin"
    with q1._private_binary_writer(frames_path) as stream:
        with pytest.raises(q1.Q1ExecutionError, match="7-byte bound"):
            q1.append_checkpoint_frame(stream, b"12345678")

    frames_path.write_bytes(q1._FRAME_LENGTH.pack(8) + b"12345678")
    with pytest.raises(q1.Q1ExecutionError, match="7-byte bound"):
        q1.read_checkpoint_frame(
            frames_path,
            frame_offset=0,
            frame_length=8,
        )

    rows_path = tmp_path / "rows.jsonl"
    with q1._private_binary_writer(rows_path) as stream:
        stream.write(b'{"x":1}\n')
    with pytest.raises(q1.Q1ExecutionError, match="7-byte row bound"):
        list(q1._read_jsonl(rows_path))


def test_synthetic_restore_subprocess_has_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, artifact = _test_synthetic_artifact()
    observed: list[float] = []

    def expire(_command: list[str], **kwargs: object) -> object:
        timeout = kwargs["timeout"]
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise AssertionError("subprocess timeout must be numeric")
        observed.append(float(timeout))
        raise q1.subprocess.TimeoutExpired(cmd="synthetic", timeout=observed[-1])

    monkeypatch.setattr(q1.subprocess, "run", expire)
    with pytest.raises(q1.subprocess.TimeoutExpired):
        q1._fresh_process_synthetic_restore(
            artifact=artifact,
            binding=authority.checkpoint_binding,
            timeout_seconds=0.25,
        )
    assert observed == [0.25]


def test_private_runtime_creation_forces_exact_modes_under_restrictive_umask(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    path = directory / "private.jsonl"
    previous_umask = os.umask(0o777)
    try:
        directory_identity = q1._mkdir_private_exact(directory)
        temporary = Path(q1.tempfile.mkdtemp(prefix="forced-private-", dir=tmp_path))
        assert stat.S_IMODE(temporary.stat().st_mode) == 0o000
        temporary_identity = q1._force_private_directory_exact(
            temporary,
            label="temporary private directory",
        )
        with q1._private_binary_writer(path) as stream:
            assert not os.get_inheritable(stream.fileno())
            stream.write(canonical_json_bytes({"fixture": True}, newline=True))
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(temporary.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_nlink == 1
    q1._require_private_directory(directory, expected_identity=directory_identity)
    q1._require_private_directory(temporary, expected_identity=temporary_identity)


def test_private_runtime_reader_rejects_symlink_hardlink_and_wrong_mode(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    with q1._private_binary_writer(path) as stream:
        stream.write(b'{"fixture":true}\n')

    hardlink = tmp_path / "rows-hardlink.jsonl"
    os.link(path, hardlink)
    with pytest.raises(q1.Q1ExecutionError, match="exactly one hard link"):
        list(q1._read_jsonl(path))
    hardlink.unlink()

    path.chmod(0o640)
    with pytest.raises(q1.Q1ExecutionError, match="exactly 0600"):
        list(q1._read_jsonl(path))
    path.chmod(0o600)

    symlink = tmp_path / "rows-symlink.jsonl"
    symlink.symlink_to(path)
    with pytest.raises(q1.Q1ExecutionError, match="cannot be opened safely"):
        list(q1._read_jsonl(symlink))


def test_private_runtime_reader_detects_path_and_parent_identity_substitution(tmp_path: Path) -> None:
    direct = tmp_path / "direct.jsonl"
    with q1._private_binary_writer(direct) as stream:
        stream.write(b'{"fixture":true}\n')
    displaced = tmp_path / "direct-displaced.jsonl"
    with pytest.raises(q1.Q1ExecutionError, match="identity changed"):
        with q1._private_binary_reader(direct) as stream:
            assert stream.read() == b'{"fixture":true}\n'
            direct.rename(displaced)
            with q1._private_binary_writer(direct) as replacement:
                replacement.write(b'{"replacement":true}\n')

    parent = tmp_path / "bound-parent"
    q1._mkdir_private_exact(parent)
    nested = parent / "nested.jsonl"
    with q1._private_binary_writer(nested) as stream:
        stream.write(b'{"fixture":true}\n')
    displaced_parent = tmp_path / "bound-parent-displaced"
    with pytest.raises(q1.Q1ExecutionError, match="parent identity changed"):
        with q1._private_binary_reader(nested) as stream:
            assert stream.read() == b'{"fixture":true}\n'
            parent.rename(displaced_parent)
            q1._mkdir_private_exact(parent)


def _test_worker_launch(
    tmp_path: Path,
    *,
    role: str = "restore",
    master: int = 0,
    arm: str = q1.ARM_ORDER[0],
    parent_pid: int | None = None,
) -> q1.AuthenticatedWorkerLaunch:
    execution_root = tmp_path.resolve() / "execution"
    incomplete = execution_root / "result.incomplete"
    master_directory = incomplete / f"master-{master}"
    restore = role == "restore"
    paths: dict[str, str | None] = {
        "attempt_registry_directory": str(tmp_path.resolve() / "attempts"),
        "entry_report_path": str(tmp_path.resolve() / "entry.json"),
        "execution_root": str(execution_root),
        "frame_path": str(master_directory / f"{arm}.frames.bin") if restore else None,
        "incomplete_directory": str(incomplete),
        "index_path": str(master_directory / f"{arm}.index.jsonl") if restore else None,
        "master_directory": str(master_directory),
        "output_path": str(master_directory / f"{arm}.restored.jsonl") if restore else str(master_directory),
        "prospective_review_path": str(tmp_path.resolve() / "review.json"),
        "q0_report_path": str(tmp_path.resolve() / "q0.json"),
        "raw_trace_path": str(master_directory / "raw.jsonl") if restore else None,
        "secret_salt_path": str(tmp_path.resolve() / "salt.bin"),
    }
    return q1.AuthenticatedWorkerLaunch(
        role=role,  # type: ignore[arg-type]
        run_identity=_test_q1_identity(),
        parent_pid=os.getpid() if parent_pid is None else parent_pid,
        master=master,
        arm=arm if restore else None,
        paths=tuple(sorted(paths.items())),
    )


def test_spawn_authenticated_worker_uses_exact_argv_pass_fds_and_pid_bound_wire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _test_worker_launch(tmp_path, role="producer")
    observed: dict[str, Any] = {}

    class _Process:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            self.pid = os.getpid() + 40_000
            self.returncode: int | None = None
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
            self.capability_thread, self.decoded, self.capability_errors = _start_capability_consumer(
                pass_fds[0],
                parent_pid=os.getpid(),
                child_pid=self.pid,
            )
            observed.update(command=command, kwargs=kwargs, process=self)

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(q1.subprocess, "Popen", _Process)
    child = q1._spawn_authenticated_worker(launch, secret=_WORKER_CAPABILITY_SECRET)
    process = observed["process"]
    assert isinstance(process, _Process)
    process.capability_thread.join(timeout=1.0)
    assert not process.capability_thread.is_alive()
    assert process.capability_errors == []
    assert len(process.decoded) == 1
    decoded = process.decoded[0]

    command = observed["command"]
    kwargs = observed["kwargs"]
    assert isinstance(command, list)
    assert isinstance(kwargs, dict)
    capability_fd = kwargs["pass_fds"]
    assert isinstance(capability_fd, tuple) and len(capability_fd) == 1
    assert command == [*launch.base_command, "--capability-fd", str(capability_fd[0])]
    assert kwargs["pass_fds"] == capability_fd
    assert kwargs["stdin"] is q1.subprocess.DEVNULL
    assert kwargs["stdout"] != q1.subprocess.PIPE
    assert kwargs["stderr"] != q1.subprocess.PIPE
    assert decoded.capability == launch.capability(process.pid)
    assert decoded.worker_capability_sha256 == _WORKER_CAPABILITY_SHA256
    rendered_command = repr(command)
    assert _WORKER_CAPABILITY_SECRET.hex() not in rendered_command
    assert all(value is None or value not in rendered_command for _name, value in launch.paths)
    with pytest.raises(OSError):
        os.fstat(capability_fd[0])
    child.close_captures()


def test_capability_delivery_supports_descriptor_above_fd_setsize() -> None:
    read_descriptor, write_descriptor = os.pipe()
    try:
        high_descriptor = fcntl.fcntl(write_descriptor, fcntl.F_DUPFD_CLOEXEC, 2048)
    except OSError:
        os.close(write_descriptor)
        os.close(read_descriptor)
        pytest.skip("process file-descriptor limit cannot allocate above FD_SETSIZE")
    os.close(write_descriptor)

    class _LiveProcess:
        def poll(self) -> None:
            return None

    try:
        q1._write_all_descriptor(
            high_descriptor,
            b"high-fd",
            process=_LiveProcess(),  # type: ignore[arg-type]
        )
        assert os.read(read_descriptor, 7) == b"high-fd"
    finally:
        os.close(high_descriptor)
        os.close(read_descriptor)


def test_capability_delivery_timeout_quiesces_nonreading_child_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _test_worker_launch(tmp_path, role="producer")
    events: list[str] = []
    captures: list[Any] = []
    inherited_descriptors: list[int] = []

    class _NeverReadingProcess:
        def __init__(self, _command: list[str], **kwargs: object) -> None:
            self.pid = os.getpid() + 50_000
            self.returncode: int | None = None
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple)
            inherited_descriptors.append(pass_fds[0])
            self.capability_reader = os.dup(pass_fds[0])
            captures.extend((kwargs["stdout"], kwargs["stderr"]))

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            raise q1.subprocess.TimeoutExpired("never-reader", timeout if timeout is not None else 0.0)

        def kill(self) -> None:
            events.append("kill")
            self.returncode = -9

    monkeypatch.setattr(q1.subprocess, "Popen", _NeverReadingProcess)
    monkeypatch.setattr(q1, "WORKER_CAPABILITY_DELIVERY_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(q1.Q1ExecutionError, match="capability acknowledgement exceeded"):
        q1._spawn_authenticated_worker(launch, secret=_WORKER_CAPABILITY_SECRET)

    assert events == ["terminate"]
    assert captures
    for descriptor in captures:
        assert isinstance(descriptor, int)
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert len(inherited_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(inherited_descriptors[0])


def test_producer_checks_origin_and_started_commitment_before_private_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = _test_worker_launch(
        tmp_path,
        role="producer",
        parent_pid=os.getppid(),
    )
    capability = launch.capability(os.getpid())
    output_path = Path(dict(capability.paths)["output_path"])  # type: ignore[arg-type]
    output_path.mkdir(mode=0o700, parents=True)
    events: list[str] = []

    monkeypatch.setattr(q1, "_validate_selected_module_origins", lambda: events.append("origins"))
    monkeypatch.setattr(
        q1,
        "_require_started_child_attempt",
        lambda **_kwargs: events.append("marker"),
    )

    def stop_at_private_authority(**_kwargs: object) -> q1.Q1ExecutionAuthority:
        events.append("private-authority")
        raise q1.Q1ExecutionError("stop before private authority read")

    monkeypatch.setattr(q1, "validate_q1_child_execution_authority", stop_at_private_authority)
    with pytest.raises(q1.Q1ExecutionError, match="stop before private authority read"):
        q1._run_master_worker(
            capability=capability,
            worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
        )

    assert events == ["origins", "marker", "private-authority"]


def test_restore_command_launcher_never_exceeds_frozen_concurrency_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"active": 0, "maximum": 0, "started": 0}

    class _Process:
        def __init__(self, _command: list[str], **kwargs: object) -> None:
            self.pid = os.getpid() + 10_000 + state["started"]
            self.returncode: int | None = None
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
            self.capability_thread, self.decoded, self.capability_errors = _start_capability_consumer(
                pass_fds[0],
                parent_pid=os.getpid(),
                child_pid=self.pid,
            )
            state["active"] += 1
            state["started"] += 1
            state["maximum"] = max(state["maximum"], state["active"])

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            self.returncode = 0
            state["active"] -= 1
            return self.returncode

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            if self.returncode is None:
                self.returncode = -15
                state["active"] -= 1

        def kill(self) -> None:
            self.terminate()

    monkeypatch.setattr(q1.subprocess, "Popen", _Process)
    launches = tuple((str(index), _test_worker_launch(tmp_path, master=index % q1.MASTER_COUNT)) for index in range(11))

    q1._run_bounded_commands(
        launches,
        secret=_WORKER_CAPABILITY_SECRET,
        label="restore-test",
        max_concurrency=q1.MAX_RESTORE_CONCURRENCY,
    )

    assert state == {"active": 0, "maximum": 4, "started": 11}


def test_bounded_capture_monitors_avoid_sibling_pipe_backpressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[q1._AuthenticatedWorkerProcess] = []

    def spawn(
        launch: q1.AuthenticatedWorkerLaunch,
        *,
        secret: bytes,
    ) -> q1._AuthenticatedWorkerProcess:
        assert secret == _WORKER_CAPABILITY_SECRET
        stdout_capture = q1._BoundedWorkerCapture(label="stdout", max_bytes=0)
        stderr_capture = q1._BoundedWorkerCapture(
            label="stderr",
            max_bytes=q1.WORKER_STDERR_CAPTURE_MAX_BYTES,
        )
        process = q1.subprocess.Popen(
            [
                q1.sys.executable,
                "-c",
                "import sys; sys.stderr.buffer.write(bytes([120]) * (32 * 1024))",
            ],
            stdout=stdout_capture.write_descriptor,
            stderr=stderr_capture.write_descriptor,
        )
        stdout_capture.start(process)
        stderr_capture.start(process)
        child = q1._AuthenticatedWorkerProcess(
            launch=launch,
            process=process,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
            launched_at=q1.time.monotonic(),
        )
        created.append(child)
        return child

    monkeypatch.setattr(q1, "_spawn_authenticated_worker", spawn)
    launches = (
        ("first", _test_worker_launch(tmp_path, master=0)),
        ("second", _test_worker_launch(tmp_path, master=1)),
    )
    q1._run_bounded_commands(
        launches,
        secret=_WORKER_CAPABILITY_SECRET,
        label="backpressure-test",
        max_concurrency=2,
    )

    assert len(created) == 2
    assert all(child.process.returncode == 0 for child in created)
    assert all(child.stdout_capture.closed and child.stderr_capture.closed for child in created)


def test_bounded_stderr_capture_overflow_terminates_child(tmp_path: Path) -> None:
    launch = _test_worker_launch(tmp_path, role="producer")
    stdout_capture = q1._BoundedWorkerCapture(label="stdout", max_bytes=0)
    stderr_capture = q1._BoundedWorkerCapture(
        label="stderr",
        max_bytes=q1.WORKER_STDERR_CAPTURE_MAX_BYTES,
    )
    process = q1.subprocess.Popen(
        [
            q1.sys.executable,
            "-c",
            "import os; os.write(2, bytes([120]) * (2 * 1024 * 1024))",
        ],
        stdout=stdout_capture.write_descriptor,
        stderr=stderr_capture.write_descriptor,
    )
    stdout_capture.start(process)
    stderr_capture.start(process)
    child = q1._AuthenticatedWorkerProcess(
        launch=launch,
        process=process,
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
        launched_at=q1.time.monotonic(),
    )

    with pytest.raises(q1.Q1ExecutionError, match="stderr exceeded the frozen"):
        q1._wait_authenticated_worker(
            child,
            identity=launch.identity,
            label="stderr-overflow-test",
            timeout=5.0,
        )
    assert stdout_capture.closed and stderr_capture.closed


def test_bounded_stdout_capture_rejects_first_byte(tmp_path: Path) -> None:
    launch = _test_worker_launch(tmp_path, role="producer")
    stdout_capture = q1._BoundedWorkerCapture(label="stdout", max_bytes=0)
    stderr_capture = q1._BoundedWorkerCapture(
        label="stderr",
        max_bytes=q1.WORKER_STDERR_CAPTURE_MAX_BYTES,
    )
    process = q1.subprocess.Popen(
        [q1.sys.executable, "-c", "import os; os.write(1, bytes([120]))"],
        stdout=stdout_capture.write_descriptor,
        stderr=stderr_capture.write_descriptor,
    )
    stdout_capture.start(process)
    stderr_capture.start(process)
    child = q1._AuthenticatedWorkerProcess(
        launch=launch,
        process=process,
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
        launched_at=q1.time.monotonic(),
    )

    with pytest.raises(q1.Q1ExecutionError, match="stdout exceeded the frozen"):
        q1._wait_authenticated_worker(
            child,
            identity=launch.identity,
            label="stdout-test",
            timeout=5.0,
        )
    assert stdout_capture.closed and stderr_capture.closed


def test_publication_cleanup_leaves_exactly_six_canonical_artifacts(
    tmp_path: Path,
) -> None:
    names = (
        "raw-trace.jsonl",
        "private-audit.jsonl",
        "checkpoint-index.jsonl",
        "checkpoint-frames.bin",
        "restored-trace.jsonl",
        "aggregate.json",
    )
    canonical_paths = tuple(tmp_path / name for name in names)
    for path in canonical_paths:
        with q1._private_binary_writer(path) as stream:
            stream.write(b"fixture")
    for master in range(4):
        worker = tmp_path / f"master-{master}"
        q1._mkdir_private_exact(worker)
        (worker / "internal.tmp").write_bytes(b"worker")

    q1._remove_worker_trees_and_require_exact_artifacts(
        root=tmp_path,
        canonical_paths=canonical_paths,
    )

    assert {path.name for path in tmp_path.iterdir()} == set(names)
    assert all(path.is_file() for path in canonical_paths)


@pytest.mark.parametrize("mutation", ["hardlink", "mode"])
def test_publication_cleanup_rejects_hardlink_or_mode_mutation_before_worker_removal(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "incomplete"
    q1._mkdir_private_exact(root)
    canonical_paths = tuple(root / name for name in q1._CANONICAL_ARTIFACT_NAMES)
    for path in canonical_paths:
        with q1._private_binary_writer(path) as stream:
            stream.write(b"fixture")
    workers = tuple(root / f"master-{master}" for master in range(q1.MASTER_COUNT))
    for worker in workers:
        q1._mkdir_private_exact(worker)

    if mutation == "hardlink":
        os.link(canonical_paths[0], tmp_path / "undeclared-hardlink")
        expected = "exactly one hard link"
    else:
        canonical_paths[0].chmod(0o640)
        expected = "exactly 0600"

    with pytest.raises(q1.Q1ExecutionError, match=expected):
        q1._remove_worker_trees_and_require_exact_artifacts(
            root=root,
            canonical_paths=canonical_paths,
        )

    assert all(worker.is_dir() for worker in workers)


def test_post_publication_gate_rejects_mode_mutation_and_identity_replacement(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    q1._mkdir_private_exact(execution_root)
    incomplete = execution_root / "result.incomplete"
    output = execution_root / "result"
    q1._mkdir_private_exact(incomplete)
    source_paths = tuple(incomplete / name for name in q1._CANONICAL_ARTIFACT_NAMES)
    for path in source_paths:
        with q1._private_binary_writer(path) as stream:
            stream.write(path.name.encode("ascii"))

    identities = q1._fsync_publication_set(source_paths, directory=incomplete)
    q1._publish_directory_noreplace(incomplete, output)
    published_paths = tuple(output / name for name in q1._CANONICAL_ARTIFACT_NAMES)
    published_paths[0].chmod(0o640)
    with pytest.raises(q1.Q1ExecutionError, match="exactly 0600"):
        q1._require_exact_private_artifact_set(
            published_paths,
            directory=output,
            expected_identities=identities,
        )

    published_paths[0].chmod(0o600)
    displaced = tmp_path / "displaced-publication-artifact"
    published_paths[0].rename(displaced)
    with q1._private_binary_writer(published_paths[0]) as replacement:
        replacement.write(b"valid-mode-replacement")
    with pytest.raises(q1.Q1ExecutionError, match="publication binding"):
        q1._require_exact_private_artifact_set(
            published_paths,
            directory=output,
            expected_identities=identities,
        )


def test_execution_authority_requires_byte_exact_fresh_qualification_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q1, "_validate_selected_module_origins", lambda: None)
    protocol_path = tmp_path / "protocol.json"
    original_protocol = _enabled_protocol_bytes()
    protocol_path.write_bytes(original_protocol)
    protocol_digest = hashlib.sha256(original_protocol).hexdigest()
    monkeypatch.setattr(q1, "Q1_PROTOCOL_PATH", protocol_path)
    implementation_digest = _digest("fresh-qualified-implementation")
    recomputed_value: dict[str, object] = {
        "schema": "prospect.wm002.active-acquisition.q1-entry-qualification.v1",
        "protocol_version": q1.Q1_PROTOCOL_VERSION,
        "protocol_sha256": protocol_digest,
        "q0_report_sha256": q1.Q0_REPORT_SHA256,
        "implementation_sha256": implementation_digest,
        "passed": True,
        "checks": [{"name": "fresh", "passed": True, "violations": []}],
    }

    class _FreshReport:
        passed = True
        implementation_sha256 = implementation_digest

        def as_dict(self) -> dict[str, object]:
            return recomputed_value

    import bench.active_acquisition.q1_qualification as qualification

    def run_fresh(**_kwargs: object) -> _FreshReport:
        return _FreshReport()

    monkeypatch.setattr(
        qualification,
        "run_entry_qualification",
        run_fresh,
    )
    salt_path = tmp_path / "salt.bin"
    original_salt = b"explicit-test-secret-with-at-least-32-bytes"
    salt_path.write_bytes(original_salt)
    salt_path.chmod(0o600)
    salt_commitment = hashlib.sha256(original_salt).hexdigest()
    recomputed_value["salt_commitment_sha256"] = salt_commitment
    q0_report_path = tmp_path / "q0.json"
    q0_report_path.write_bytes(b"fresh-q0-input")
    review_path = tmp_path / "prospective-review.json"
    review_path.write_bytes(b"fresh-review-input")
    execution_root = tmp_path / "execution"
    execution_root.mkdir(mode=0o700)
    attempt_registry = tmp_path / "attempts"
    attempt_registry.mkdir(mode=0o700)
    execution_identity = q1._capture_q1_runtime_directory_identity(
        execution_root,
        label="test execution root",
    )
    registry_identity = q1._capture_q1_runtime_directory_identity(
        attempt_registry,
        label="test attempt registry",
    )
    recomputed_value["resource_preflight"] = {
        "attempt_registry_directory": registry_identity.as_dict(),
        "execution_root": execution_identity.as_dict(),
    }
    entry_path = tmp_path / "entry.json"
    entry_path.write_bytes(canonical_json_bytes(recomputed_value, newline=True))

    def validate() -> q1.Q1ExecutionAuthority:
        return q1.validate_q1_execution_authority(
            entry_report_path=entry_path,
            q0_report_path=q0_report_path,
            secret_salt_path=salt_path,
            prospective_review_path=review_path,
            protocol_path=protocol_path,
            execution_root=execution_root,
            attempt_registry_directory=attempt_registry,
        )

    authority = validate()

    assert authority.protocol_sha256 == protocol_digest
    assert authority.implementation_sha256 == implementation_digest
    assert authority.salt_commitment_sha256 == salt_commitment
    assert authority.execution_root_identity == execution_identity
    assert authority.attempt_registry_identity == registry_identity
    assert authority.execution_root == execution_root.resolve()
    assert authority.attempt_registry_directory == attempt_registry.resolve()
    assert "explicit-test-secret" not in repr(authority)

    replaced_execution_root = tmp_path / "replaced-execution"
    execution_root.rename(replaced_execution_root)
    execution_root.mkdir(mode=0o700)
    try:
        with pytest.raises(q1.Q1ExecutionError, match="execution root.*entry-bound"):
            validate()
    finally:
        execution_root.rmdir()
        replaced_execution_root.rename(execution_root)

    for field_name, forged_value in (
        ("protocol_sha256", _digest("concurrently-replaced-protocol")),
        ("salt_commitment_sha256", _digest("concurrently-replaced-salt")),
    ):
        original_value = recomputed_value[field_name]
        recomputed_value[field_name] = forged_value
        with pytest.raises(
            q1.Q1ExecutionError,
            match=f"parent binding: {field_name}",
        ):
            validate()
        recomputed_value[field_name] = original_value

    def mutate_protocol(**_kwargs: object) -> _FreshReport:
        protocol_path.write_bytes(_enabled_protocol_bytes(concurrent_mutation=True))
        return _FreshReport()

    monkeypatch.setattr(
        qualification,
        "run_entry_qualification",
        mutate_protocol,
    )
    with pytest.raises(q1.Q1ExecutionError, match="protocol changed across"):
        validate()
    protocol_path.write_bytes(original_protocol)

    def mutate_salt(**_kwargs: object) -> _FreshReport:
        salt_path.write_bytes(b"concurrently-mutated-secret-with-32-bytes")
        salt_path.chmod(0o600)
        return _FreshReport()

    monkeypatch.setattr(
        qualification,
        "run_entry_qualification",
        mutate_salt,
    )
    with pytest.raises(q1.Q1ExecutionError, match="salt changed across"):
        validate()
    salt_path.write_bytes(original_salt)
    salt_path.chmod(0o600)
    monkeypatch.setattr(
        qualification,
        "run_entry_qualification",
        run_fresh,
    )

    forged_minimal = dict(recomputed_value)
    forged_minimal["checks"] = [{"passed": True, "violations": []}]
    entry_path.write_bytes(canonical_json_bytes(forged_minimal, newline=True))
    with pytest.raises(q1.Q1ExecutionError, match="fresh exact qualification"):
        validate()


@pytest.mark.parametrize("salt_mode", [0o644, 0o400])
def test_execution_authority_rejects_non_exact_salt_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    salt_mode: int,
) -> None:
    monkeypatch.setattr(q1, "_validate_selected_module_origins", lambda: None)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_bytes(_enabled_protocol_bytes())
    monkeypatch.setattr(q1, "Q1_PROTOCOL_PATH", protocol_path)
    q0_path = tmp_path / "q0.json"
    q0_path.write_bytes(b"not reached before salt permission rejection")
    salt_path = tmp_path / "salt.bin"
    salt_path.write_bytes(os.urandom(32))
    salt_path.chmod(salt_mode)

    with pytest.raises(q1.Q1ExecutionError, match="permissions"):
        q1.validate_q1_execution_authority(
            entry_report_path=tmp_path / "missing-entry.json",
            q0_report_path=q0_path,
            secret_salt_path=salt_path,
            prospective_review_path=tmp_path / "review.json",
            protocol_path=protocol_path,
            execution_root=tmp_path,
            attempt_registry_directory=tmp_path,
        )


def _test_q1_identity() -> q1.Q1RunIdentity:
    return q1.derive_run_identity(
        protocol_version=q1.Q1_PROTOCOL_VERSION,
        protocol_sha256=_digest("protocol"),
        implementation_sha256=_digest("implementation"),
        q0_report_sha256=_digest("q0"),
        entry_qualification_sha256=_digest("entry"),
        salt_commitment_sha256=_digest("salt"),
    )


def _test_q1_authority(
    *,
    execution_root: Path | None = None,
    attempt_registry_directory: Path | None = None,
    execution_mode: Q1ExecutionMode = Q1ExecutionMode.PRODUCTION,
) -> q1.Q1ExecutionAuthority:
    identity = _test_q1_identity()
    if (execution_root is None) != (attempt_registry_directory is None):
        raise AssertionError("test runtime directories must be supplied together")
    if execution_root is None or attempt_registry_directory is None:
        execution_identity = q1.Q1RuntimeDirectoryIdentity(
            canonical_path="/synthetic/wm002-q1-execution",
            st_dev=1,
            st_ino=1,
            st_uid=os.geteuid(),
            st_gid=os.getegid(),
        )
        registry_identity = q1.Q1RuntimeDirectoryIdentity(
            canonical_path="/synthetic/wm002-q1-registry",
            st_dev=1,
            st_ino=2,
            st_uid=os.geteuid(),
            st_gid=os.getegid(),
        )
    else:
        execution_identity = q1._capture_q1_runtime_directory_identity(
            execution_root,
            label="test execution root",
        )
        registry_identity = q1._capture_q1_runtime_directory_identity(
            attempt_registry_directory,
            label="test attempt registry",
        )
    return q1.Q1ExecutionAuthority(
        protocol_sha256=identity.protocol_sha256,
        implementation_sha256=identity.implementation_sha256,
        q0_report_sha256=identity.q0_report_sha256,
        entry_qualification_sha256=identity.entry_qualification_sha256,
        salt_commitment_sha256=identity.salt_commitment_sha256,
        run_identity=identity,
        execution_root_identity=execution_identity,
        attempt_registry_identity=registry_identity,
        execution_mode=execution_mode,
        secret_salt=b"s" * 32,
    )


def test_runtime_directory_identity_parser_requires_exact_seven_field_object(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime-root"
    root.mkdir(mode=0o700)
    identity = q1._capture_q1_runtime_directory_identity(root, label="test runtime root")
    assert q1._parse_q1_runtime_directory_identity(
        identity.as_dict(),
        label="test identity",
    ) == identity

    mutations: list[dict[str, object]] = []
    extra = identity.as_dict()
    extra["extra"] = False
    mutations.append(extra)
    missing = identity.as_dict()
    del missing["st_gid"]
    mutations.append(missing)
    for field_name, forged in (
        ("canonical_path", "relative/path"),
        ("file_type", "file"),
        ("mode", "0755"),
        ("st_dev", True),
        ("st_gid", -1),
        ("st_ino", "1"),
        ("st_uid", 1.0),
    ):
        mutated = identity.as_dict()
        mutated[field_name] = forged
        mutations.append(mutated)

    for mutated in mutations:
        with pytest.raises(q1.Q1ExecutionError):
            q1._parse_q1_runtime_directory_identity(mutated, label="mutated test identity")


def test_runtime_directory_identity_rejects_same_path_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime-root"
    old_root = tmp_path / "old-runtime-root"
    root.mkdir(mode=0o700)
    expected = q1._capture_q1_runtime_directory_identity(root, label="test runtime root")
    root.rename(old_root)
    root.mkdir(mode=0o700)

    with pytest.raises(q1.Q1ExecutionError, match="entry-bound directory identity"):
        q1._require_q1_runtime_directory_identity(
            root,
            expected,
            label="test runtime root",
        )


def test_run_q1_revalidates_entry_bound_directories_immediately_before_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    old_registry = tmp_path / "old-registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    monkeypatch.setattr(q1, "validate_q1_execution_authority", lambda **_kwargs: authority)
    real_commitment = q1.worker_capability_commitment

    def replace_registry(secret: bytes) -> str:
        registry.rename(old_registry)
        registry.mkdir(mode=0o700)
        return real_commitment(secret)

    monkeypatch.setattr(q1, "worker_capability_commitment", replace_registry)

    with pytest.raises(q1.Q1ExecutionError, match="attempt registry.*entry-bound"):
        q1.run_q1(
            entry_report_path=tmp_path / "entry.json",
            q0_report_path=tmp_path / "q0.json",
            secret_salt_path=tmp_path / "salt.bin",
            prospective_review_path=tmp_path / "review.json",
            execution_root=execution_root,
            attempt_registry_directory=registry,
            output_directory=execution_root / "result",
        )

    assert not q1.attempt_marker_path(registry, authority.run_identity).exists()
    assert not q1.attempt_marker_path(old_registry, authority.run_identity).exists()


def test_failed_attempt_refuses_substituted_registry_and_leaves_marker_started(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    old_registry = tmp_path / "old-registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    marker_path = q1.claim_attempt(
        registry,
        authority.run_identity,
        worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )
    registry.rename(old_registry)
    registry.mkdir(mode=0o700)
    error = q1.Q1ExecutionError("synthetic post-claim failure")

    q1._finalize_failed_q1_attempt(
        error=error,
        marker_path=marker_path,
        authority=authority,
        output_directory=execution_root / "result",
        incomplete_directory=execution_root / "result.incomplete",
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )

    old_marker = q1.attempt_marker_path(old_registry, authority.run_identity)
    assert q1.load_attempt_marker(
        old_marker,
        expected_identity=authority.run_identity,
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    ).status == "started"
    assert not q1.attempt_marker_path(registry, authority.run_identity).exists()
    assert any("identity cannot be revalidated" in note for note in error.__notes__)


def _test_synthetic_artifact() -> tuple[q1.Q1ExecutionAuthority, q1.EpisodeArtifacts]:
    authority = _test_q1_authority()
    artifact = q1._run_synthetic_episode(
        arm=q1.ArmMode.PROSPECT,
        synthetic_ordinal=1,
        binding=authority.checkpoint_binding,
        protocol_sha256=authority.protocol_sha256,
        implementation_sha256=authority.implementation_sha256,
        q0_report_sha256=authority.q0_report_sha256,
        salt_commitment_sha256=authority.salt_commitment_sha256,
    )
    return authority, artifact


def _bounded_orchestration_artifact(
    *,
    authority: q1.Q1ExecutionAuthority,
    master: int,
    arm: q1.ArmMode,
    producer_pid: int,
) -> q1.EpisodeArtifacts:
    """Build one explicit development episode without a Q1 schedule or draw."""

    private = q1._synthetic_private_material(
        arm=arm,
        master=master,
        episode=0,
        salt_commitment_sha256=authority.salt_commitment_sha256,
        private_variant=f"bounded-orchestration-master-{master}",
    )
    acquisition_potential, terminal_potential = q1._synthetic_private_potential_outcomes(private)
    uniform = q1.derive_public_uniform_selection(master, 0) if arm is q1.ArmMode.UNIFORM_RANDOM else None
    expected_action = uniform.action_id if uniform is not None else q1._EXPECTED_ACTION_BY_ARM[arm]
    return q1._run_live_episode(
        master=master,
        episode=0,
        arm=arm,
        producer_pid=producer_pid,
        identity_next_counter=q1.public_identity_counter_initialization(master, 0, arm.value),
        acquisition_observed_symbol=acquisition_potential[expected_action],
        terminal_success=lambda decision: terminal_potential[int(decision)],
        uniform_ordinal=uniform.index if uniform is not None else None,
        uniform_selector_digest=uniform.semantic_key_sha256 if uniform is not None else None,
        semantic_key_sha256=_digest(f"bounded-orchestration:{master}:{arm.value}"),
        binding=authority.checkpoint_binding,
        protocol_sha256=authority.protocol_sha256,
        implementation_sha256=authority.implementation_sha256,
        q0_report_sha256=authority.q0_report_sha256,
        salt_commitment_sha256=authority.salt_commitment_sha256,
        entry_qualification_sha256=authority.entry_qualification_sha256,
        private_audit=private,
        expected_action=expected_action,
        public_privacy_schedule=None,
    )


def test_run_q1_bounded_authenticated_happy_path_publishes_and_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise orchestration mechanics only; this is not a valid full-budget Q1 result."""

    from jsonschema import Draft202012Validator

    execution_root = tmp_path / "execution"
    execution_root.mkdir(mode=0o700)
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    output = execution_root / "result"
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    external_inputs = {
        "entry_report_path": tmp_path / "missing-entry.json",
        "q0_report_path": tmp_path / "missing-q0.json",
        "secret_salt_path": tmp_path / "missing-salt.bin",
        "prospective_review_path": tmp_path / "missing-review.json",
    }

    monkeypatch.setattr(q1, "episodes_per_master", lambda _mode: 1)
    monkeypatch.setattr(q1, "validate_q1_execution_authority", lambda **_kwargs: authority)

    def forbidden_q1_private_path(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("bounded orchestration invoked a Q1 schedule or private episode path")

    monkeypatch.setattr(q1, "PrivateQ1SeedSchedule", forbidden_q1_private_path)
    monkeypatch.setattr(q1, "_run_q1_episode", forbidden_q1_private_path)

    real_run_live_episode = q1._run_live_episode
    live_episode_calls = 0

    def counted_live_episode(**kwargs: Any) -> q1.EpisodeArtifacts:
        nonlocal live_episode_calls
        assert kwargs["public_privacy_schedule"] is None
        live_episode_calls += 1
        return real_run_live_episode(**kwargs)

    monkeypatch.setattr(q1, "_run_live_episode", counted_live_episode)

    aggregate_schema = copy.deepcopy(q1.schema_documents()["aggregate"])
    assert isinstance(aggregate_schema, dict)
    aggregate_properties = aggregate_schema["properties"]
    assert isinstance(aggregate_properties, dict)
    counts_schema = aggregate_properties["counts"]
    assert isinstance(counts_schema, dict)
    count_properties = counts_schema["properties"]
    assert isinstance(count_properties, dict)
    for field, value in {
        "episodes": 28,
        "environment_steps": 56,
        "transitions": 56,
        "acquisition_updates": 28,
        "checkpoints": 28,
        "restores": 28,
    }.items():
        field_schema = count_properties[field]
        assert isinstance(field_schema, dict)
        field_schema["const"] = value
    arm_means_schema = aggregate_properties["arm_means"]
    assert isinstance(arm_means_schema, dict)
    arm_mean_item = arm_means_schema["items"]
    assert isinstance(arm_mean_item, dict)
    arm_mean_properties = arm_mean_item["properties"]
    assert isinstance(arm_mean_properties, dict)
    episode_count_schema = arm_mean_properties["episode_count"]
    assert isinstance(episode_count_schema, dict)
    episode_count_schema["const"] = 1
    bounded_aggregate_validator = Draft202012Validator(aggregate_schema)
    real_artifact_validator = q1._artifact_validator

    def artifact_validator(name: str) -> Any:
        if name == "aggregate":
            return bounded_aggregate_validator
        return real_artifact_validator(name)

    monkeypatch.setattr(q1, "_artifact_validator", artifact_validator)

    artifacts: dict[tuple[int, str], q1.EpisodeArtifacts] = {}
    launch_records: list[tuple[q1.AuthenticatedWorkerLaunch, Any]] = []
    capability_commitments: set[str] = set()
    active = {"producer": 0, "restore": 0}
    maximum = {"producer": 0, "restore": 0}

    def emit_worker(capability: Any, pid: int) -> None:
        paths = dict(capability.paths)
        master_dir_value = paths["master_directory"]
        assert isinstance(master_dir_value, str)
        master_dir = Path(master_dir_value)
        if capability.role == "producer":
            raw_path = master_dir / "raw.jsonl"
            private_path = master_dir / "private.jsonl"
            with (
                q1._private_binary_writer(raw_path) as raw_stream,
                q1._private_binary_writer(private_path) as private_stream,
            ):
                for arm_name in q1.ARM_ORDER:
                    arm = q1.ArmMode(arm_name)
                    artifact = _bounded_orchestration_artifact(
                        authority=authority,
                        master=capability.master,
                        arm=arm,
                        producer_pid=pid,
                    )
                    artifacts[(capability.master, arm_name)] = artifact
                    raw_stream.write(canonical_json_bytes(artifact.raw_trace, newline=True))
                    private_stream.write(canonical_json_bytes(artifact.private_audit, newline=True))
                    frame_path = master_dir / f"{arm_name}.frames.bin"
                    index_path = master_dir / f"{arm_name}.index.jsonl"
                    with (
                        q1._private_binary_writer(frame_path) as frame_stream,
                        q1._private_binary_writer(index_path) as index_stream,
                    ):
                        location = q1.append_checkpoint_frame(frame_stream, artifact.checkpoint_payload)
                        index = dict(artifact.checkpoint_index)
                        index["frame_offset"] = location.frame_offset
                        index["frame_length"] = location.frame_length
                        q1._validate_artifact_fast("checkpoint_frame", index)
                        index_stream.write(canonical_json_bytes(index, newline=True))
            return

        arm_name = capability.arm
        output_path_value = paths["output_path"]
        assert isinstance(arm_name, str)
        assert isinstance(output_path_value, str)
        restored = q1._restored_schema_sample(artifacts[(capability.master, arm_name)].raw_trace)
        restored["restorer_pid"] = pid
        q1._validate_artifact_fast("restored_trace", restored)
        with q1._private_binary_writer(Path(output_path_value)) as stream:
            stream.write(canonical_json_bytes(restored, newline=True))

    def fake_spawn(
        launch: q1.AuthenticatedWorkerLaunch,
        *,
        secret: bytes,
    ) -> q1._AuthenticatedWorkerProcess:
        pid = 50_000 + len(launch_records)
        capability = launch.capability(pid)
        launch_records.append((launch, capability))
        capability_commitments.add(q1.worker_capability_commitment(secret))
        active[launch.role] += 1
        maximum[launch.role] = max(maximum[launch.role], active[launch.role])

        class FakeProcess:
            def __init__(self) -> None:
                self.pid = pid
                self.returncode: int | None = None

            def wait(self, timeout: float | None = None) -> int:
                assert timeout is not None and timeout > 0.0
                if self.returncode is None:
                    emit_worker(capability, self.pid)
                    self.returncode = 0
                    active[launch.role] -= 1
                return self.returncode

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                if self.returncode is None:
                    self.returncode = -15
                    active[launch.role] -= 1

            def kill(self) -> None:
                if self.returncode is None:
                    self.returncode = -9
                    active[launch.role] -= 1

        process = FakeProcess()
        stdout_capture = q1._BoundedWorkerCapture(label="stdout", max_bytes=0)
        stderr_capture = q1._BoundedWorkerCapture(
            label="stderr",
            max_bytes=q1.WORKER_STDERR_CAPTURE_MAX_BYTES,
        )
        stdout_capture.start(process)  # type: ignore[arg-type]
        stderr_capture.start(process)  # type: ignore[arg-type]
        return q1._AuthenticatedWorkerProcess(
            launch=launch,
            process=process,  # type: ignore[arg-type]
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
            launched_at=q1.time.monotonic(),
        )

    monkeypatch.setattr(q1, "_spawn_authenticated_worker", fake_spawn)

    stage_calls: list[str] = []
    for stage_name in (
        "_revalidate_q1_runtime_directories",
        "_merge_worker_outputs",
        "_stream_validate_and_aggregate",
        "_remove_worker_trees_and_require_exact_artifacts",
        "_fsync_publication_set",
        "_publish_directory_noreplace",
    ):
        original = getattr(q1, stage_name)

        def tracked_stage(
            *args: object,
            _stage_name: str = stage_name,
            _original: Any = original,
            **kwargs: object,
        ) -> object:
            stage_calls.append(_stage_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(q1, stage_name, tracked_stage)

    previous_umask = os.umask(0o777)
    try:
        outputs = q1.run_q1(
            **external_inputs,
            execution_root=execution_root,
            attempt_registry_directory=registry,
            output_directory=output,
        )
    finally:
        os.umask(previous_umask)

    expected_launches = [
        *(("producer", master, None) for master in range(q1.MASTER_COUNT)),
        *(("restore", master, arm) for master in range(q1.MASTER_COUNT) for arm in q1.ARM_ORDER),
    ]
    assert [(launch.role, launch.master, launch.arm) for launch, _capability in launch_records] == expected_launches
    assert len(launch_records) == 32
    assert maximum == {"producer": 4, "restore": 4}
    assert active == {"producer": 0, "restore": 0}
    assert len(capability_commitments) == 1
    for launch, capability in launch_records:
        expected_base = (
            (q1.sys.executable, "-S", "-m", "bench.active_acquisition.q1", "_producer-master")
            if launch.role == "producer"
            else (q1.sys.executable, "-S", "-m", "bench.active_acquisition.restore_worker", "q1")
        )
        assert launch.base_command == expected_base
        assert len(launch.base_command) == 5
        assert capability.run_identity == authority.run_identity
        assert capability.parent_pid == os.getpid()
        assert capability.master == launch.master
        assert capability.arm == launch.arm

    assert stage_calls == [
        "_revalidate_q1_runtime_directories",
        "_revalidate_q1_runtime_directories",
        "_merge_worker_outputs",
        "_stream_validate_and_aggregate",
        "_remove_worker_trees_and_require_exact_artifacts",
        "_fsync_publication_set",
        "_revalidate_q1_runtime_directories",
        "_publish_directory_noreplace",
        "_revalidate_q1_runtime_directories",
    ]
    assert live_episode_calls == 28
    assert all(not path.exists() for path in external_inputs.values())
    assert outputs.output_directory == output
    assert not os.path.lexists(output.with_name("result.incomplete"))
    expected_names = {
        "raw-trace.jsonl",
        "private-audit.jsonl",
        "checkpoint-index.jsonl",
        "checkpoint-frames.bin",
        "restored-trace.jsonl",
        "aggregate.json",
    }
    assert {path.name for path in output.iterdir()} == expected_names
    assert all(path.is_file() and not path.is_symlink() for path in output.iterdir())
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    assert all(path.stat().st_nlink == 1 for path in output.iterdir())
    assert not any(path.name.startswith("master-") for path in output.iterdir())
    assert stat.S_IMODE(outputs.private_audit.stat().st_mode) == 0o600

    raw_rows = list(q1._read_jsonl(outputs.raw_trace))
    private_rows = list(q1._read_jsonl(outputs.private_audit))
    index_rows = list(q1._read_jsonl(outputs.checkpoint_index))
    restored_rows = list(q1._read_jsonl(outputs.restored_trace))
    expected_coordinates = [(master, arm, 0) for master in range(q1.MASTER_COUNT) for arm in q1.ARM_ORDER]
    assert [(row["master"], row["arm"], row["episode"]) for row in raw_rows] == expected_coordinates
    assert [(row["master"], row["arm_id"], row["episode"]) for row in private_rows] == expected_coordinates
    assert [(row["master"], row["arm"], row["episode"]) for row in index_rows] == expected_coordinates
    assert [(row["master"], row["arm"], row["episode"]) for row in restored_rows] == expected_coordinates

    next_frame_offset = 0
    for index in index_rows:
        frame_offset = index["frame_offset"]
        frame_length = index["frame_length"]
        checkpoint_sha256 = index["checkpoint_sha256"]
        assert type(frame_offset) is int
        assert type(frame_length) is int
        assert isinstance(checkpoint_sha256, str)
        assert frame_offset == next_frame_offset
        payload = q1.read_checkpoint_frame(
            outputs.checkpoint_frames,
            frame_offset=frame_offset,
            frame_length=frame_length,
        )
        assert hashlib.sha256(payload).hexdigest() == checkpoint_sha256
        next_frame_offset += q1._FRAME_LENGTH.size + frame_length
    assert next_frame_offset == outputs.checkpoint_frames.stat().st_size

    aggregate_payload = outputs.producer_aggregate.read_bytes()
    aggregate = json.loads(aggregate_payload)
    assert aggregate_payload == canonical_json_bytes(aggregate, newline=True)
    assert aggregate["counts"] == {
        "masters": 4,
        "arms": 7,
        "episodes": 28,
        "environment_steps": 56,
        "transitions": 56,
        "acquisition_updates": 28,
        "terminal_updates": 0,
        "checkpoints": 28,
        "restores": 28,
    }
    assert len(aggregate["arm_means"]) == 28
    assert all(row["episode_count"] == 1 for row in aggregate["arm_means"])
    assert len(aggregate["comparisons"]) == 5
    assert aggregate["run_id"] == authority.run_identity.run_id
    assert aggregate["attempt_id"] == authority.run_identity.attempt_id
    assert aggregate["claim_eligible"] is False
    assert aggregate["formal_authorized"] is False
    assert aggregate["producer_analysis_authoritative"] is False

    worker_capability_sha256 = next(iter(capability_commitments))
    marker_path = q1.attempt_marker_path(registry, authority.run_identity)
    assert marker_path.name == "wm002-q1.attempt.json"
    assert {path.name for path in registry.iterdir()} == {marker_path.name}
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o600
    assert marker_path.stat().st_nlink == 1
    marker = q1.load_attempt_marker(
        marker_path,
        expected_identity=authority.run_identity,
        expected_worker_capability_sha256=worker_capability_sha256,
    )
    assert marker.status == "completed"
    assert marker.worker_capability_sha256 == worker_capability_sha256
    published_by_marker_name = {
        "aggregate": outputs.producer_aggregate,
        "checkpoint_frames": outputs.checkpoint_frames,
        "checkpoint_index": outputs.checkpoint_index,
        "private_audit": outputs.private_audit,
        "raw_trace": outputs.raw_trace,
        "restored_trace": outputs.restored_trace,
    }
    assert dict(marker.artifact_sha256) == {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(published_by_marker_name.items())
    }


def test_consolidation_rejects_restored_model_identity_tampering() -> None:
    _authority, artifact = _test_synthetic_artifact()
    raw = dict(artifact.raw_trace)
    restored = q1._restored_schema_sample(raw)

    q1._validate_consolidated_binding(
        raw=raw,
        private=artifact.private_audit,
        index=artifact.checkpoint_index,
        restored=restored,
    )

    acquisition = dict(q1._mapping(raw["acquisition"], "test acquisition"))
    acquisition["model_after_sha256"] = _digest("tampered-restored-model")
    raw["acquisition"] = acquisition
    with pytest.raises(
        q1.Q1ExecutionError,
        match="restored model parity mismatch: model_sha256",
    ):
        q1._validate_consolidated_binding(
            raw=raw,
            private=artifact.private_audit,
            index=artifact.checkpoint_index,
            restored=restored,
        )


def test_restore_rejects_component_digest_map_tampering() -> None:
    import bench.active_acquisition.restore_worker as restore_worker

    authority, artifact = _test_synthetic_artifact()
    raw = dict(artifact.raw_trace)
    raw["producer_pid"] = os.getpid() + 1
    index = dict(artifact.checkpoint_index)
    component_sha256 = dict(q1._mapping(index["component_sha256"], "test component digests"))
    component_sha256["identity_counter"] = _digest("tampered-component")
    index["component_sha256"] = component_sha256
    master = raw["master"]
    arm = raw["arm"]
    episode = raw["episode"]
    if type(master) is not int or type(episode) is not int:
        raise AssertionError("synthetic episode coordinates must be exact integers")
    if not isinstance(arm, str):
        raise AssertionError("synthetic arm must be a string")

    with pytest.raises(
        q1.Q1ExecutionError,
        match="component digests differ from the restored bundle",
    ):
        restore_worker._restore_one(
            master=master,
            arm=q1.ArmMode(arm),
            episode=episode,
            checkpoint_payload=artifact.checkpoint_payload,
            checkpoint_index=index,
            raw_trace=raw,
            binding=authority.checkpoint_binding,
            terminal_success=lambda _decision: False,
            privacy_schedule=None,
        )


def test_execution_authority_rejects_optimized_interpreter() -> None:
    script = """\
from pathlib import Path
import bench.active_acquisition.q1 as q1
try:
    q1.validate_q1_execution_authority(
        entry_report_path=Path("missing-entry"),
        q0_report_path=Path("missing-q0"),
        secret_salt_path=Path("missing-salt"),
        prospective_review_path=Path("missing-review"),
        execution_root=Path("."),
        attempt_registry_directory=Path("."),
    )
except q1.Q1ExecutionError as error:
    print(error)
else:
    raise SystemExit("optimized interpreter was accepted")
"""
    completed = q1.subprocess.run(
        [q1.sys.executable, "-O", "-c", script],
        cwd=Path(q1.__file__).resolve().parents[2],
        env=q1._q1_child_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Q1 requires an unoptimized Python interpreter"


def test_publication_cleanup_rejects_symlink_before_removing_workers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "incomplete"
    q1._mkdir_private_exact(root)
    names = (
        "raw-trace.jsonl",
        "private-audit.jsonl",
        "checkpoint-index.jsonl",
        "checkpoint-frames.bin",
        "restored-trace.jsonl",
        "aggregate.json",
    )
    canonical_paths = tuple(root / name for name in names)
    target = tmp_path / "outside-aggregate"
    target.write_bytes(b"outside")
    for path in canonical_paths[:-1]:
        with q1._private_binary_writer(path) as stream:
            stream.write(b"fixture")
    canonical_paths[-1].symlink_to(target)
    workers = tuple(root / f"master-{master}" for master in range(q1.MASTER_COUNT))
    for worker in workers:
        q1._mkdir_private_exact(worker)
        (worker / "internal.tmp").write_bytes(b"worker")

    with pytest.raises(q1.Q1ExecutionError, match="cannot be opened safely"):
        q1._remove_worker_trees_and_require_exact_artifacts(
            root=root,
            canonical_paths=canonical_paths,
        )

    assert all(worker.is_dir() for worker in workers)


def test_atomic_noreplace_publication_probe_and_collision_preserve_state(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir(mode=0o700)
    before = tuple(execution_root.iterdir())

    q1.probe_atomic_noreplace_publication(execution_root)

    assert tuple(execution_root.iterdir()) == before
    source = execution_root / "source"
    destination = execution_root / "destination"
    q1._mkdir_private_exact(source)
    q1._mkdir_private_exact(destination)
    with q1._private_binary_writer(source / "new") as stream:
        stream.write(b"new")
    with q1._private_binary_writer(destination / "old") as stream:
        stream.write(b"old")

    with pytest.raises(q1.Q1ExecutionError, match="appeared before"):
        q1._publish_directory_noreplace(source, destination)

    assert (source / "new").read_bytes() == b"new"
    assert (destination / "old").read_bytes() == b"old"


def test_restore_child_timeout_is_measured_from_each_popen_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    second_timeouts: list[float] = []
    events: list[str] = []
    started = 0

    class _Process:
        def __init__(self, _command: list[str], **kwargs: object) -> None:
            nonlocal started
            self.identity = "first" if started == 0 else "second"
            self.pid = os.getpid() + 20_000 + started
            started += 1
            self.returncode: int | None = None
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple)
            self.capability_thread, self.decoded, self.capability_errors = _start_capability_consumer(
                pass_fds[0],
                parent_pid=os.getpid(),
                child_pid=self.pid,
            )

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            if self.identity == "first":
                clock["now"] = q1.RESTORE_CHILD_TIMEOUT_SECONDS - 1.0
                self.returncode = 0
                return self.returncode
            second_timeouts.append(timeout)
            raise q1.subprocess.TimeoutExpired(self.identity, timeout)

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            events.append(f"terminate:{self.identity}")
            self.returncode = -15

        def kill(self) -> None:
            events.append(f"kill:{self.identity}")
            self.returncode = -9

    monkeypatch.setattr(q1.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(q1.subprocess, "Popen", _Process)

    with pytest.raises(q1.Q1ExecutionError, match="frozen child or stage timeout"):
        q1._run_bounded_commands(
            (
                ("first", _test_worker_launch(tmp_path, master=0)),
                ("second", _test_worker_launch(tmp_path, master=1)),
            ),
            secret=_WORKER_CAPABILITY_SECRET,
            label="restore-test",
            max_concurrency=2,
        )

    assert second_timeouts == [pytest.approx(1.0)]
    assert events == ["terminate:second"]


def test_quiescence_terminates_kills_and_reaps_stubborn_child() -> None:
    events: list[str] = []

    class _StubbornProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            events.append("poll")
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: float | None = None) -> int:
            events.append("wait")
            assert timeout is not None
            raise q1.subprocess.TimeoutExpired("stubborn", timeout)

        def kill(self) -> None:
            events.append("kill")
            self.returncode = -9

    process = _StubbornProcess()
    q1._quiesce_processes((("stubborn", process),), label="test")  # type: ignore[arg-type]

    assert "terminate" in events
    assert "kill" in events
    assert process.returncode == -9
    assert events[-1] == "poll"


def test_unproven_child_quiescence_leaves_attempt_started_and_output_untouched(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    execution_root.mkdir(mode=0o700)
    registry.mkdir(mode=0o700)
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    identity = authority.run_identity
    marker_path = q1.claim_attempt(registry, identity, worker_capability_sha256=_WORKER_CAPABILITY_SHA256)
    incomplete = execution_root / "result.incomplete"
    incomplete.mkdir()
    sentinel = incomplete / "sentinel"
    sentinel.write_bytes(b"preserve")
    error = q1.Q1ChildQuiescenceError("cannot prove exit")

    q1._finalize_failed_q1_attempt(
        error=error,
        marker_path=marker_path,
        authority=authority,
        output_directory=execution_root / "result",
        incomplete_directory=incomplete,
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )

    assert (
        q1.load_attempt_marker(
            marker_path,
            expected_identity=identity,
            expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
        ).status
        == "started"
    )
    assert sentinel.read_bytes() == b"preserve"
    assert any("remains started" in note for note in error.__notes__)


def test_partial_producer_spawn_failure_quiesces_started_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    q1._mkdir_private_exact(execution_root)
    q1._mkdir_private_exact(registry)
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    events: list[str] = []
    popen_calls = 0

    class _Process:
        def __init__(self) -> None:
            self.pid = os.getpid() + 30_000
            self.returncode: int | None = None
            self.capability_thread: Any = None
            self.decoded: Any = None
            self.capability_errors: Any = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            events.append("kill")
            self.returncode = -9

    def _popen(_command: list[str], **kwargs: object) -> _Process:
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 1:
            pass_fds = kwargs["pass_fds"]
            assert isinstance(pass_fds, tuple)
            process = _Process()
            process.capability_thread, process.decoded, process.capability_errors = _start_capability_consumer(
                pass_fds[0],
                parent_pid=os.getpid(),
                child_pid=process.pid,
            )
            return process
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(q1.subprocess, "Popen", _popen)
    output = execution_root / "result"

    with pytest.raises(OSError, match="synthetic spawn failure"):
        q1._execute_claimed_q1(
            authority=authority,
            entry_report_path=tmp_path / "entry.json",
            q0_report_path=tmp_path / "q0.json",
            secret_salt_path=tmp_path / "salt.bin",
            prospective_review_path=tmp_path / "review.json",
            execution_root=execution_root,
            attempt_registry_directory=registry,
            output_directory=output,
            worker_capability_secret=_WORKER_CAPABILITY_SECRET,
        )

    assert events == ["terminate"]
    assert output.with_name("result.incomplete").is_dir()


def test_run_q1_rejects_hardlinked_final_artifact_before_completed_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    registry = tmp_path / "registry"
    q1._mkdir_private_exact(execution_root)
    q1._mkdir_private_exact(registry)
    output = execution_root / "result"
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    monkeypatch.setattr(q1, "validate_q1_execution_authority", lambda **_kwargs: authority)
    monkeypatch.setattr(
        q1.os,
        "urandom",
        lambda size: _WORKER_CAPABILITY_SECRET if size == 32 else b"x" * size,
    )

    def execute(**_kwargs: object) -> q1.Q1RunOutputs:
        q1._mkdir_private_exact(output)
        paths = {
            "raw_trace": output / "raw-trace.jsonl",
            "private_audit": output / "private-audit.jsonl",
            "checkpoint_index": output / "checkpoint-index.jsonl",
            "checkpoint_frames": output / "checkpoint-frames.bin",
            "restored_trace": output / "restored-trace.jsonl",
            "producer_aggregate": output / "aggregate.json",
        }
        for path in paths.values():
            with q1._private_binary_writer(path) as stream:
                stream.write(b"fixture")
        os.link(paths["raw_trace"], tmp_path / "undeclared-final-hardlink")
        return q1.Q1RunOutputs(output_directory=output, **paths)

    statuses: list[str] = []
    real_finalize = q1.finalize_attempt

    def track_finalize(*args: object, **kwargs: object) -> object:
        status = kwargs.get("status")
        assert isinstance(status, str)
        statuses.append(status)
        return real_finalize(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(q1, "_execute_claimed_q1", execute)
    monkeypatch.setattr(q1, "finalize_attempt", track_finalize)

    with pytest.raises(q1.Q1ExecutionError, match="exactly one hard link"):
        q1.run_q1(
            entry_report_path=tmp_path / "entry.json",
            q0_report_path=tmp_path / "q0.json",
            secret_salt_path=tmp_path / "salt.bin",
            prospective_review_path=tmp_path / "review.json",
            execution_root=execution_root,
            attempt_registry_directory=registry,
            output_directory=output,
        )

    assert statuses == ["failed"]
    marker = q1.load_attempt_marker(
        q1.attempt_marker_path(registry, authority.run_identity),
        expected_identity=authority.run_identity,
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )
    assert marker.status == "failed"


def test_completed_marker_durability_error_is_reraised_without_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    execution_root.mkdir(mode=0o700)
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    output = execution_root / "result"
    authority = _test_q1_authority(
        execution_root=execution_root,
        attempt_registry_directory=registry,
    )
    real_finalize = q1.finalize_attempt

    monkeypatch.setattr(
        q1.os,
        "urandom",
        lambda size: _WORKER_CAPABILITY_SECRET if size == 32 else b"x" * size,
    )

    monkeypatch.setattr(
        q1,
        "validate_q1_execution_authority",
        lambda **_kwargs: authority,
    )

    def _execute(**_kwargs: object) -> q1.Q1RunOutputs:
        q1._mkdir_private_exact(output)
        paths = {
            "raw_trace": output / "raw-trace.jsonl",
            "private_audit": output / "private-audit.jsonl",
            "checkpoint_index": output / "checkpoint-index.jsonl",
            "checkpoint_frames": output / "checkpoint-frames.bin",
            "restored_trace": output / "restored-trace.jsonl",
            "producer_aggregate": output / "aggregate.json",
        }
        for path in paths.values():
            with q1._private_binary_writer(path) as stream:
                stream.write(b"fixture")
        marker_path = q1.attempt_marker_path(registry, authority.run_identity)
        assert (
            q1.load_attempt_marker(
                marker_path,
                expected_identity=authority.run_identity,
                expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
            ).status
            == "started"
        )
        return q1.Q1RunOutputs(output_directory=output, **paths)

    def _finalize_then_raise(*args: object, **kwargs: object) -> object:
        real_finalize(*args, **kwargs)  # type: ignore[arg-type]
        raise OSError("synthetic marker directory fsync failure")

    monkeypatch.setattr(q1, "_execute_claimed_q1", _execute)
    monkeypatch.setattr(q1, "finalize_attempt", _finalize_then_raise)

    with pytest.raises(OSError, match="directory fsync failure"):
        q1.run_q1(
            entry_report_path=tmp_path / "entry.json",
            q0_report_path=tmp_path / "q0.json",
            secret_salt_path=tmp_path / "salt.bin",
            prospective_review_path=tmp_path / "review.json",
            execution_root=execution_root,
            attempt_registry_directory=registry,
            output_directory=output,
        )

    marker = q1.load_attempt_marker(
        q1.attempt_marker_path(registry, authority.run_identity),
        expected_identity=authority.run_identity,
        expected_worker_capability_sha256=_WORKER_CAPABILITY_SHA256,
    )
    assert marker.status == "completed"


def test_worker_cli_exposes_only_capability_fd_and_public_entry_validates_first(
    tmp_path: Path,
) -> None:
    import bench.active_acquisition.restore_worker as restore_worker

    producer = q1._parser().parse_args(["_producer-master", "--capability-fd", "7"])
    restorer = restore_worker._parser().parse_args(["q1", "--capability-fd", "9"])
    assert vars(producer) == {"capability_fd": 7, "command": "_producer-master"}
    assert vars(restorer) == {"capability_fd": 9, "command": "q1"}

    with pytest.raises(SystemExit):
        q1._parser().parse_args(["_producer-master", "--secret-salt", "/private/salt.bin"])
    with pytest.raises(SystemExit):
        restore_worker._parser().parse_args(["q1", "--entry-report", "/inputs/entry.json"])

    execution_root = tmp_path / "execution"
    execution_root.mkdir(mode=0o700)
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    missing = tmp_path / "missing"
    command = (
        q1.sys.executable,
        "-S",
        "-m",
        "bench.active_acquisition.q1",
        "run",
        "--entry-report",
        str(missing),
        "--q0-report",
        str(missing),
        "--prospective-review",
        str(missing),
        "--secret-salt",
        str(missing),
        "--execution-root",
        str(execution_root),
        "--attempt-registry",
        str(registry),
        "--output-directory",
        str(execution_root / "result"),
    )
    protocol = q1.json.loads(q1.Q1_PROTOCOL_PATH.read_bytes())
    expected_error = (
        "successor protocol execution boundary mismatch: execution_authorized"
        if protocol["experiment"]["execution_authorized"] is False
        else "secret salt cannot be opened safely"
    )
    completed = q1.subprocess.run(
        command,
        cwd=Path(q1.__file__).resolve().parents[2],
        env=q1._q1_child_environment(),
        check=False,
        capture_output=True,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode != 0
    assert "required runtime module" not in stderr
    assert "runtime package search path mismatch" not in stderr
    assert expected_error in stderr


def test_child_environment_replaces_hostile_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/hostile/shadow")
    monkeypatch.setenv("pythonpath", "/hostile/lowercase-shadow")
    monkeypatch.setenv("PYTHONHOME", "/hostile/home")
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    monkeypatch.setenv("PYTHONSAFEPATH", "0")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/hostile/cache")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setenv("LD_PRELOAD", "/hostile/inject.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/hostile/libs")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "credential-must-not-cross")
    monkeypatch.setenv("GITHUB_TOKEN", "credential-must-not-cross")

    environment = q1._q1_child_environment()
    repository_root = Path(q1.__file__).resolve().parents[2]
    import_roots = [repository_root, repository_root / "src"]
    for name in ("purelib", "platlib"):
        dependency = Path(q1.sysconfig.get_path(name)).resolve()
        if dependency not in import_roots:
            import_roots.append(dependency)
    expected_python_environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in import_roots),
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
    }

    assert {
        key: value for key, value in environment.items() if key.upper().startswith("PYTHON")
    } == expected_python_environment
    assert environment == {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "TZ": "UTC",
        **expected_python_environment,
    }
    assert "/hostile/shadow" not in environment["PYTHONPATH"]
    assert "/hostile/lowercase-shadow" not in environment.values()

    probe_script = """
import json
import sys
import jsonschema
import bench.active_acquisition.q1 as q1
print(json.dumps({
    \"no_site\": sys.flags.no_site,
    \"site_loaded\": \"site\" in sys.modules,
    \"sitecustomize_loaded\": \"sitecustomize\" in sys.modules,
    \"sys_path\": sys.path,
    \"q1_origin\": q1.__file__,
    \"jsonschema_origin\": jsonschema.__file__,
}))
"""
    completed = q1.subprocess.run(
        [q1.sys.executable, "-S", "-c", probe_script],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.returncode == 0, completed.stderr
    probe = json.loads(completed.stdout)
    assert probe["no_site"] == 1
    assert probe["site_loaded"] is False
    assert probe["sitecustomize_loaded"] is False
    assert probe["q1_origin"] == str((repository_root / "bench/active_acquisition/q1.py").resolve())
    assert probe["jsonschema_origin"] == str((import_roots[-1] / "jsonschema/__init__.py").resolve())
    assert all(str(path) in probe["sys_path"] for path in import_roots)
    assert all("structsplat" not in path for path in probe["sys_path"])
