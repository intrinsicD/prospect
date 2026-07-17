from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.multimodal_causal_diagnostics import sandbox


def test_landlock_abi_meets_frozen_requirement() -> None:
    assert sandbox.landlock_abi() >= sandbox.LANDLOCK_ABI_REQUIRED


def test_child_denies_unlisted_files_and_every_socket_family(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    output = tmp_path / "output"
    denied = tmp_path / "denied.txt"
    allowed.mkdir()
    output.mkdir()
    (allowed / "input.txt").write_text("source", encoding="utf-8")
    denied.write_text("future", encoding="utf-8")
    script = r"""
import importlib.machinery
import os
import socket
import sys
from pathlib import Path
from bench.multimodal_causal_diagnostics.sandbox import enter_source_only_sandbox

allowed, output, denied = (Path(value) for value in sys.argv[1:])
loader = importlib.machinery.SourceFileLoader("denied_module", str(denied))
enter_source_only_sandbox((allowed,), output)
assert (allowed / "input.txt").read_text(encoding="utf-8") == "source"
(output / "prediction.txt").write_text("prediction", encoding="utf-8")
try:
    denied.read_bytes()
except PermissionError:
    pass
else:
    raise SystemExit("denied future was readable")
try:
    descriptor = os.open(denied, os.O_RDONLY)
except PermissionError:
    pass
else:
    os.close(descriptor)
    raise SystemExit("raw os.open escaped the filesystem policy")
try:
    loader.get_data(str(denied))
except PermissionError:
    pass
else:
    raise SystemExit("manual source loader escaped the filesystem policy")
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    try:
        socket.socket(family, socket.SOCK_STREAM)
    except PermissionError:
        pass
    else:
        raise SystemExit(f"socket family {family} was available")
for label, operation in (
    ("setsid", os.setsid),
    ("setpgid", lambda: os.setpgid(0, 0)),
    ("execve", lambda: os.execve("/bin/true", ("true",), {})),
):
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise SystemExit(f"{label} was available")
"""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path.cwd() / "src"))),
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(allowed), str(output), str(denied)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        close_fds=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "prediction.txt").read_text(encoding="utf-8") == "prediction"


def test_landlock_rejects_relative_duplicate_and_symlink_rules(tmp_path: Path) -> None:
    real = tmp_path / "real"
    output = tmp_path / "output"
    real.mkdir()
    output.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(sandbox.SandboxUnavailable, match="absolute"):
        sandbox.install_landlock((Path("relative"),), output)
    with pytest.raises(sandbox.SandboxUnavailable, match="unique"):
        sandbox.install_landlock((real, real), output)
    with pytest.raises(sandbox.SandboxUnavailable, match="symlink"):
        sandbox.install_landlock((link,), output)


def test_seccomp_policy_closes_process_group_and_exec_escape_syscalls() -> None:
    blocked = set(sandbox._BLOCKED_SYSCALLS)  # noqa: SLF001
    assert {"execve", "execveat", "setsid", "setpgid"} <= blocked
