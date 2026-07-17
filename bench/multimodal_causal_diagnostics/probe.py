"""Known-answer authority probe for the MM-009 Landlock/seccomp boundary."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import socket
import stat
import sys
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = "mm009-isolation-probe-v3"
PROBE_FILE: Final = "isolation-probe.json"
_SYS_PROCESS_VM_READV: Final = 310
_SOCKET_VARIANTS: Final = (
    (socket.AF_INET, socket.SOCK_STREAM, "AF_INET/SOCK_STREAM"),
    (socket.AF_INET, socket.SOCK_DGRAM, "AF_INET/SOCK_DGRAM"),
    (socket.AF_INET6, socket.SOCK_STREAM, "AF_INET6/SOCK_STREAM"),
    (socket.AF_INET6, socket.SOCK_DGRAM, "AF_INET6/SOCK_DGRAM"),
    (socket.AF_UNIX, socket.SOCK_STREAM, "AF_UNIX/SOCK_STREAM"),
    (socket.AF_UNIX, socket.SOCK_DGRAM, "AF_UNIX/SOCK_DGRAM"),
)
_DYNAMIC_LOADER_CANDIDATES: Final = (
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib64"),
    Path("/etc/ld.so.cache"),
)
_OPERATIONAL_READ_CANDIDATES: Final = (
    Path("/dev/null"),
    Path("/sys/devices/system/cpu"),
)
_ENVIRONMENT: Final = {
    "HOME": "/nonexistent",
    "LC_ALL": "C",
    "MALLOC_ARENA_MAX": "2",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}


class ProbeError(RuntimeError):
    """Raised when a supposedly denied authority remains available."""


def _path(value: str, label: str, *, directory: bool | None = None) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.exists():
        raise ProbeError(f"{label} must be an absolute existing non-symlink path")
    if directory is True and not candidate.is_dir():
        raise ProbeError(f"{label} must be a directory")
    if directory is False and not candidate.is_file():
        raise ProbeError(f"{label} must be a file")
    return candidate


def _validate_dependency_root(path: Path) -> None:
    if stat.S_IMODE(path.lstat().st_mode) != 0o555:
        raise ProbeError("NumPy dependency root must be mode 0555")
    if {entry.name for entry in os.scandir(path)} != {"numpy", "numpy.libs"}:
        raise ProbeError("NumPy dependency root membership differs")
    for name in ("numpy", "numpy.libs"):
        package = path / name
        if package.is_symlink() or not package.is_dir():
            raise ProbeError("NumPy dependency package differs")


def _validate_stdlib_root(path: Path) -> None:
    if stat.S_IMODE(path.lstat().st_mode) != 0o555:
        raise ProbeError("copied stdlib root must be mode 0555")
    if (
        (path / "os.py").is_symlink()
        or not (path / "os.py").is_file()
        or (path / "lib-dynload").is_symlink()
        or not (path / "lib-dynload").is_dir()
        or (path / "site-packages").exists()
    ):
        raise ProbeError("copied stdlib root membership differs")


def _deny_path(path: Path) -> None:
    try:
        if path.is_dir():
            os.listdir(path)
        else:
            with path.open("rb") as handle:
                handle.read(1)
    except PermissionError:
        return
    raise ProbeError(f"unlisted path remained readable: {path}")


def _deny_sockets() -> int:
    denied = 0
    for family, kind, label in _SOCKET_VARIANTS:
        try:
            descriptor = socket.socket(family, kind)
        except PermissionError:
            denied += 1
            continue
        descriptor.close()
        raise ProbeError(f"socket variant remained available: {label}")
    return denied


def _deny_cross_process_read() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = int(libc.syscall(_SYS_PROCESS_VM_READV, os.getppid(), 0, 0, 0, 0, 0))
    if result != -1 or ctypes.get_errno() != errno.EACCES:
        raise ProbeError("process_vm_readv was not rejected by seccomp with EACCES")


def _runtime_read_paths(
    runtime: Path,
    dependency_root: Path,
    stdlib_root: Path,
    allowed: Path,
    config: Path,
) -> tuple[Path, ...]:
    stdlib = _standard_library_import_roots(stdlib_root)
    candidates = (
        runtime,
        dependency_root,
        allowed,
        config,
        *stdlib,
        *_DYNAMIC_LOADER_CANDIDATES,
        *_OPERATIONAL_READ_CANDIDATES,
    )
    return tuple(
        path
        for index, path in enumerate(candidates)
        if path.exists() and not path.is_symlink() and path not in candidates[:index]
    )


def _standard_library_import_roots(stdlib_root: Path) -> tuple[Path, ...]:
    _validate_stdlib_root(stdlib_root)
    return (stdlib_root, stdlib_root / "lib-dynload")


def _rebind_loaded_stdlib_packages(stdlib_root: Path) -> None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    live_stdlib = Path(sys.base_prefix) / "lib" / version
    for module in tuple(sys.modules.values()):
        paths = getattr(module, "__path__", None)
        if paths is None:
            continue
        rebound: list[str] = []
        changed = False
        for value in paths:
            path = Path(value)
            try:
                relative = path.relative_to(live_stdlib)
            except ValueError:
                rebound.append(str(path))
                continue
            candidate = stdlib_root / relative
            if not candidate.is_dir() or candidate.is_symlink():
                raise ProbeError(f"startup stdlib package is absent from copied closure: {relative}")
            rebound.append(str(candidate))
            changed = True
        if changed:
            module.__path__ = rebound


def _sealed_sys_path(runtime: Path, dependency_root: Path, stdlib_root: Path) -> list[str]:
    candidates = (
        runtime,
        dependency_root,
        *_standard_library_import_roots(stdlib_root),
    )
    paths = [str(path) for path in candidates if path.exists() and not path.is_symlink()]
    if str(runtime) not in paths or str(dependency_root) not in paths:
        raise ProbeError("sealed Python import roots are unavailable")
    return paths


def run(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) < 8:
        raise ProbeError(
            "probe requires ALLOWED CONFIG OUTPUT RUNTIME DEPENDENCY_ROOT STDLIB_ROOT LIVE_ROOT_COUNT DENIED..."
        )
    allowed = _path(arguments[0], "allowed input")
    config = _path(arguments[1], "config", directory=False)
    output = _path(arguments[2], "output", directory=True)
    runtime = _path(arguments[3], "runtime", directory=True)
    dependency_root = _path(arguments[4], "NumPy dependency root", directory=True)
    stdlib_root = _path(arguments[5], "copied stdlib root", directory=True)
    _validate_dependency_root(dependency_root)
    _validate_stdlib_root(stdlib_root)
    try:
        live_root_count = int(arguments[6])
    except ValueError as error:
        raise ProbeError("live Python root count must be an integer") from error
    denied = tuple(_path(value, "denied sentinel") for value in arguments[7:])
    if live_root_count < 1 or live_root_count > len(denied):
        raise ProbeError("live Python root count differs")
    if any(output.iterdir()) or os.environ != _ENVIRONMENT:
        raise ProbeError("probe output/environment differs")
    sys.path[:] = _sealed_sys_path(runtime, dependency_root, stdlib_root)
    _rebind_loaded_stdlib_packages(stdlib_root)
    os.chdir(output)
    os.umask(0o077)
    from bench.multimodal_causal_diagnostics import sandbox

    readable = _runtime_read_paths(runtime, dependency_root, stdlib_root, allowed, config)
    abi = sandbox.enter_source_only_sandbox(readable, output)
    for path in denied:
        _deny_path(path)
    network_socket_variants_denied = _deny_sockets()
    _deny_cross_process_read()
    value = {
        "denied_path_count": len(denied),
        "landlock_abi": abi,
        "live_python_roots_denied": live_root_count,
        "network_families_denied": 3,
        "network_socket_variants_denied": network_socket_variants_denied,
        "process_vm_readv_denied": True,
        "requested_denied_path_count": len(denied) - live_root_count,
        "schema_version": SCHEMA_VERSION,
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(output / PROBE_FILE, flags, 0o444)
    try:
        os.write(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())


__all__ = ["PROBE_FILE", "ProbeError", "SCHEMA_VERSION", "run"]
