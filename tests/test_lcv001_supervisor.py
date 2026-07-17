from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from bench.sealed_lineage_verifier import runtime_probe, supervisor


def _completed(
    command: str | Sequence[str],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _mock_host_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "manifest", lambda: {"schema_version": supervisor.SCHEMA_VERSION})
    monkeypatch.setattr(supervisor, "_validate_executables", lambda: {})


def test_manifest_binds_host_executables_and_frozen_policy() -> None:
    value = supervisor.manifest()
    assert value == {
        "capture_limit_bytes": 8 * 1024 * 1024,
        "cgroup_version": 2,
        "cleanup_receipt_schema": "lcv001-cgroup-cleanup-v1",
        "dynamic_operational_environment": ["LCV001_CGROUP_ROLE", "LCV001_CGROUP_UNIT"],
        "executables": {
            "/home/alex/miniconda3/bin/python3.12": {
                "bytes": 30_753_048,
                "mode": 0o775,
                "sha256": "d9b73f600a860cefe799d7b4d21bad64719231227c2cf326ca17ed94624841af",
            },
            "/usr/bin/env": {
                "bytes": 48_072,
                "mode": 0o755,
                "sha256": "0aefff8f912fb75716c5d4de3b6acde93edbe8fa280fc8ee895c1226d3e373ef",
            },
            "/usr/bin/systemctl": {
                "bytes": 1_501_304,
                "mode": 0o755,
                "sha256": "7ba82b5ba146759c710e1b80fadaa3fdbc0f9b85c8fb2c8c3196b7b1a0037ef8",
            },
            "/usr/bin/systemd-run": {
                "bytes": 68_392,
                "mode": 0o755,
                "sha256": "49f0bf95eb8a781b93853bf9fc981b4929dd0009f55a3e6db95534c0a2d11716",
            },
        },
        "kill_mode": "control-group",
        "kill_signal": "SIGKILL",
        "loader_environment_overrides": {
            "LD_AUDIT": "",
            "LD_LIBRARY_PATH": "",
            "LD_PRELOAD": "",
        },
        "runtime_max_seconds": 180,
        "schema_version": "lcv001-systemd-cgroup-supervisor-v1",
        "send_sigkill": True,
        "unit_prefix": "lcv001-custody",
    }


def test_mocked_launch_preserves_lexical_argv0_and_exact_unit_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host_boundary(monkeypatch)
    invocations: list[tuple[str, ...]] = []

    def fake_run(command: Any, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        argv = tuple(command)
        invocations.append(argv)
        if argv[0] == str(supervisor.SYSTEMD_RUN):
            return _completed(argv, stdout="child-output")
        if argv[2] == "is-active":
            return _completed(argv, returncode=3, stdout="inactive\n")
        return _completed(argv, returncode=1)

    monkeypatch.setattr(supervisor, "_run_cancellable_process", fake_run)
    command = (str(supervisor.LEXICAL_PYTHON), "-I", "-S", "-B", "probe.py")
    completed = supervisor.run(
        command,
        cwd=tmp_path.resolve(),
        environment={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout_seconds=20.0,
        role="test-launch",
    )
    assert completed.args == command
    assert completed.returncode == 0
    assert completed.stdout == "child-output"
    assert completed.cleanup_receipt["status"] == "inactive_and_cgroup_absent"
    assert completed.cleanup_receipt["cgroup_paths"] == []

    service = invocations[0]
    assert service[:7] == (
        "/usr/bin/systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--quiet",
        "--pipe",
        "--expand-environment=no",
    )
    assert "--service-type=exec" in service
    assert "--setenv=LD_AUDIT=" in service
    assert "--setenv=LD_LIBRARY_PATH=" in service
    assert "--setenv=LD_PRELOAD=" in service
    assert "--property=RuntimeMaxSec=180s" in service
    assert "--property=KillMode=control-group" in service
    assert "--property=KillSignal=SIGKILL" in service
    assert "--property=SendSIGKILL=yes" in service
    assert "--property=Restart=no" in service
    assert f"--property=WorkingDirectory={tmp_path.resolve()}" in service
    unit_argument = next(value for value in service if value.startswith("--unit="))
    unit = unit_argument.removeprefix("--unit=")
    assert unit.startswith("lcv001-custody-test-launch-") and unit.endswith(".service")

    env_index = service.index("/usr/bin/env")
    assert service[env_index + 1] == "-i"
    guard_index = service.index("-c", env_index)
    assert service[guard_index - 4 : guard_index] == (
        "/home/alex/miniconda3/bin/python3.12",
        "-I",
        "-S",
        "-B",
    )
    assert "os.execv(real,[lexical,*sys.argv[5:]])" in service[guard_index + 1]
    assert "('LD_AUDIT','LD_LIBRARY_PATH','LD_PRELOAD')" in service[guard_index + 1]
    assert service[guard_index + 2 : guard_index + 6] == (
        unit,
        "test-launch",
        str(supervisor.REAL_PYTHON),
        str(supervisor.LEXICAL_PYTHON),
    )
    assignments = service[env_index + 2 : guard_index - 4]
    assert "LCV001_CGROUP_ROLE=test-launch" in assignments
    assert f"LCV001_CGROUP_UNIT={unit}" in assignments
    assert "PATH=/usr/bin:/bin" in assignments
    assert "PYTHONDONTWRITEBYTECODE=1" in assignments

    assert [command[2] for command in invocations[1:]] == ["kill", "stop", "reset-failed", "is-active"]
    assert all(command[-1] == unit for command in invocations[1:])


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(("systemd-run",), 1.0),
        KeyboardInterrupt("fixture interrupt"),
        supervisor.SupervisorError("fixture failure"),
    ],
)
def test_timeout_interrupt_and_failure_always_cleanup_and_verify_inactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    _mock_host_boundary(monkeypatch)
    cleanup_actions: list[str] = []
    service_called = False

    def fake_run(command: Any, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal service_called
        argv = tuple(command)
        if argv[0] == str(supervisor.SYSTEMD_RUN):
            service_called = True
            raise failure
        cleanup_actions.append(argv[2])
        if argv[2] == "is-active":
            return _completed(argv, returncode=4, stdout="unknown\n")
        return _completed(argv, returncode=1)

    monkeypatch.setattr(supervisor, "_run_cancellable_process", fake_run)
    with pytest.raises(type(failure)):
        supervisor.run(
            (str(supervisor.LEXICAL_PYTHON), "-I", "-S", "-B", "probe.py"),
            cwd=tmp_path.resolve(),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=20.0,
            role="failure",
        )
    assert service_called
    assert cleanup_actions == ["kill", "stop", "reset-failed", "is-active"]


def test_active_unit_fails_closed_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host_boundary(monkeypatch)

    def fake_run(command: Any, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        argv = tuple(command)
        if argv[0] == str(supervisor.SYSTEMD_RUN):
            return _completed(argv)
        if argv[2] == "is-active":
            return _completed(argv, stdout="active\n")
        return _completed(argv)

    monkeypatch.setattr(supervisor, "_run_cancellable_process", fake_run)
    with pytest.raises(supervisor.SupervisorError, match="remained active"):
        supervisor.run(
            (str(supervisor.LEXICAL_PYTHON), "-I", "-S", "-B", "probe.py"),
            cwd=tmp_path.resolve(),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=20.0,
            role="active",
        )


def test_resolved_argv0_is_rejected_before_unit_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "manifest", lambda: pytest.fail("host launch must not begin"))
    with pytest.raises(supervisor.SupervisorError, match=r"pinned lexical argv\[0\]"):
        supervisor.run(
            (str(supervisor.REAL_PYTHON), "-I", "-S", "-B", "probe.py"),
            cwd=tmp_path.resolve(),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=20.0,
            role="negative",
        )


@pytest.mark.parametrize(
    "command",
    [
        (str(supervisor.LEXICAL_PYTHON), "-S", "-B", "probe.py"),
        (str(supervisor.LEXICAL_PYTHON), "-I", "-B", "-S", "probe.py"),
        (str(supervisor.LEXICAL_PYTHON), "-I", "-S", "probe.py"),
    ],
)
def test_missing_or_reordered_isolation_flags_are_rejected_before_unit_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
) -> None:
    monkeypatch.setattr(supervisor, "manifest", lambda: pytest.fail("host launch must not begin"))
    with pytest.raises(supervisor.SupervisorError, match="-I -S -B prefix"):
        supervisor.run(
            command,
            cwd=tmp_path.resolve(),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=20.0,
            role="negative",
        )


@pytest.mark.parametrize("name", ["LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD"])
def test_loader_environment_injection_is_rejected_before_unit_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(supervisor, "manifest", lambda: pytest.fail("host launch must not begin"))
    with pytest.raises(supervisor.SupervisorError, match="environment grammar"):
        supervisor.run(
            (str(supervisor.LEXICAL_PYTHON), "-I", "-S", "-B", "probe.py"),
            cwd=tmp_path.resolve(),
            environment={"PYTHONDONTWRITEBYTECODE": "1", name: "/tmp/lcv001-poison.so"},
            timeout_seconds=20.0,
            role="negative",
        )


@pytest.mark.parametrize(
    ("bus_mode", "runtime_uid", "bus_uid"),
    [(stat.S_IFREG | 0o600, 1000, 1000), (stat.S_IFSOCK | 0o600, 1001, 1000), (stat.S_IFSOCK | 0o600, 1000, 1001)],
)
def test_user_bus_must_be_a_current_uid_owned_socket(
    monkeypatch: pytest.MonkeyPatch,
    bus_mode: int,
    runtime_uid: int,
    bus_uid: int,
) -> None:
    uid = 1000
    runtime = Path(f"/run/user/{uid}")
    bus = runtime / "bus"
    monkeypatch.setattr(supervisor.os, "getuid", lambda: uid)

    def fake_lstat(path: Path) -> object:
        if path == runtime:
            return type("Metadata", (), {"st_mode": stat.S_IFDIR | 0o700, "st_uid": runtime_uid})()
        if path == bus:
            return type("Metadata", (), {"st_mode": bus_mode, "st_uid": bus_uid})()
        raise AssertionError(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(supervisor.SupervisorError, match="bus boundary differs"):
        supervisor._systemd_client_environment()  # noqa: SLF001


def test_service_timeout_retains_outer_local_reaping_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_host_boundary(monkeypatch)
    observed: list[tuple[float, float | None]] = []

    def fake_run(command: Any, **kwargs: object) -> subprocess.CompletedProcess[str]:
        argv = tuple(command)
        if argv[0] == str(supervisor.SYSTEMD_RUN):
            observed.append(
                (
                    float(cast(float, kwargs["deadline"])),
                    cast(float | None, kwargs.get("reap_deadline")),
                )
            )
            raise subprocess.TimeoutExpired(argv, 1.0)
        if argv[2] == "is-active":
            return _completed(argv, returncode=4, stdout="unknown\n")
        return _completed(argv, returncode=1)

    monkeypatch.setattr(supervisor, "_run_cancellable_process", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor.run(
            (str(supervisor.LEXICAL_PYTHON), "-I", "-S", "-B", "probe.py"),
            cwd=tmp_path.resolve(),
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=20.0,
            role="timeout",
        )
    assert len(observed) == 1
    service_deadline, reap_deadline = observed[0]
    assert reap_deadline is not None
    assert float(reap_deadline) - service_deadline == pytest.approx(6.0)


def test_cleanup_receipt_is_strict_and_role_bound() -> None:
    unit = "lcv001-custody-formal-123-456-789.service"
    receipt = {
        "actions": [
            {"action": action, "returncode": 1, "stderr": "", "stdout": ""}
            for action in ("kill", "stop", "reset-failed")
        ],
        "cgroup_paths": [],
        "is_active_returncode": 4,
        "schema_version": "lcv001-cgroup-cleanup-v1",
        "state": "unknown",
        "status": "inactive_and_cgroup_absent",
        "unit": unit,
    }
    assert supervisor.validate_cleanup_receipt(receipt, role="formal") == receipt
    mutated = {**receipt, "cgroup_paths": [f"/sys/fs/cgroup/{unit}"]}
    with pytest.raises(supervisor.SupervisorError, match="postcondition"):
        supervisor.validate_cleanup_receipt(mutated, role="formal")
    with pytest.raises(supervisor.SupervisorError, match="unit identity"):
        supervisor.validate_cleanup_receipt(receipt, role="semantic")


def test_cgroup_census_fails_closed_on_walk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_walk(*_args: object, **kwargs: object) -> list[object]:
        callback = cast(Callable[[OSError], None], kwargs["onerror"])
        callback(PermissionError("injected cgroup census denial"))
        return []

    monkeypatch.setattr(supervisor.os, "walk", failed_walk)
    with pytest.raises(supervisor.SupervisorError, match="cgroup-v2 tree"):
        supervisor._unit_cgroup_paths("lcv001-custody-formal-1-2-3.service")  # noqa: SLF001


def test_assert_current_cgroup_requires_exact_operational_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = "lcv001-custody-formal-123-456-789.service"
    monkeypatch.setenv("LCV001_CGROUP_UNIT", unit)
    monkeypatch.setenv("LCV001_CGROUP_ROLE", "formal")
    original = Path.read_text

    def fake_read_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == Path("/proc/self/cgroup"):
            return f"0::/user.slice/{unit}\n"
        return original(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert supervisor.assert_current_cgroup("formal") == unit
    monkeypatch.setenv("LCV001_CGROUP_ROLE", "semantic")
    with pytest.raises(supervisor.SupervisorError, match="role environment"):
        supervisor.assert_current_cgroup("formal")


def _real_systemd_available() -> bool:
    try:
        supervisor.manifest()
        completed = subprocess.run(
            (str(supervisor.SYSTEMCTL), "--user", "is-system-running"),
            check=False,
            capture_output=True,
            env=supervisor._systemd_client_environment(),  # noqa: SLF001
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError, supervisor.SupervisorError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "running"


def test_real_cgroup_runtime_receipt_and_canary_match_direct_execution(tmp_path: Path) -> None:
    """Normative host probe; it does not read or write any sealed experiment tree."""

    if not _real_systemd_available():
        pytest.skip("normative user-systemd cgroup boundary is unavailable")
    probe = tmp_path / "runtime-receipt-probe.py"
    probe.write_text(
        """
import json
import os
import sys
import types

sys.path[:] = [
    "/home/alex/Documents/prospect",
    "/home/alex/miniconda3/lib/python312.zip",
    "/home/alex/miniconda3/lib/python3.12",
    "/home/alex/miniconda3/lib/python3.12/lib-dynload",
    "/home/alex/Documents/prospect/.venv/lib/python3.12/site-packages",
]
bench = types.ModuleType("bench")
bench.__path__ = ["/home/alex/Documents/prospect/bench"]
sealed = types.ModuleType("bench.sealed_lineage_verifier")
sealed.__path__ = ["/home/alex/Documents/prospect/bench/sealed_lineage_verifier"]
sys.modules["bench"] = bench
sys.modules["bench.sealed_lineage_verifier"] = sealed
from bench.sealed_lineage_verifier import supervisor

role = os.environ.get("LCV001_CGROUP_ROLE")
if role is not None:
    supervisor.assert_current_cgroup(role)
    os.environ.pop("LCV001_CGROUP_ROLE")
    os.environ.pop("LCV001_CGROUP_UNIT")
from bench.sealed_lineage_verifier import runtime_probe

print(json.dumps(runtime_probe.observe_and_validate(), sort_keys=True, separators=(",", ":")))
""".lstrip(),
        encoding="utf-8",
    )
    command = (
        str(supervisor.LEXICAL_PYTHON),
        "-I",
        "-S",
        "-B",
        str(probe),
    )
    environment = runtime_probe.frozen_environment()
    direct = subprocess.run(
        command,
        cwd=supervisor.REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    assert direct.returncode == 0, direct.stderr[-2000:]
    assert direct.stderr == ""
    cgroup = supervisor.run(
        command,
        cwd=supervisor.REPO_ROOT,
        environment=environment,
        timeout_seconds=180.0,
        role="normative",
    )
    assert cgroup.returncode == 0, cgroup.stderr[-2000:]
    assert cgroup.stderr == ""
    direct_receipt = json.loads(direct.stdout)
    cgroup_receipt = json.loads(cgroup.stdout)
    assert cgroup_receipt == direct_receipt
    assert cgroup_receipt["executable"] == str(supervisor.LEXICAL_PYTHON)
    assert cgroup_receipt["canary"]["bundle_sha256"] == runtime_probe.CANARY_BUNDLE_SHA256
    assert cgroup_receipt["canary"]["u_sha256"] == runtime_probe.CANARY_U_SHA256
    assert cgroup_receipt["canary"]["s_sha256"] == runtime_probe.CANARY_S_SHA256
    assert cgroup_receipt["canary"]["vh_sha256"] == runtime_probe.CANARY_VH_SHA256


def test_real_supervisor_scrubs_caller_loader_poison_with_structural_unit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise caller scrubbing; manager state remains an explicit unit override."""

    if not _real_systemd_available():
        pytest.skip("normative user-systemd cgroup boundary is unavailable")
    monkeypatch.setenv("LD_AUDIT", "/tmp/lcv001-missing-audit.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/lcv001-missing-library-path")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/lcv001-missing-preload.so")
    receipt = supervisor.preflight({"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    assert receipt["executable"] == str(supervisor.LEXICAL_PYTHON)
    assert receipt["proc_exe"] == str(supervisor.REAL_PYTHON)


def test_noisy_child_hits_capture_bound_and_is_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[int] = []
    original_kill = supervisor._kill_process_group  # noqa: SLF001

    def recording_kill(process: subprocess.Popen[bytes], *, deadline: float) -> None:
        killed.append(process.pid)
        original_kill(process, deadline=deadline)

    monkeypatch.setattr(supervisor, "_kill_process_group", recording_kill)
    now = supervisor.time.monotonic()
    with pytest.raises(supervisor.SupervisorError, match="capture bound"):
        supervisor._run_cancellable_process(  # noqa: SLF001
            (
                str(supervisor.REAL_PYTHON),
                "-I",
                "-S",
                "-B",
                "-c",
                "import os,time;os.write(1,b'x'*(9*1024*1024));time.sleep(30)",
            ),
            cwd=tmp_path,
            environment={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            deadline=now + 5.0,
            reap_deadline=now + 6.0,
        )
    assert len(killed) == 1
    assert not supervisor._process_group_exists(killed[0])  # noqa: SLF001
