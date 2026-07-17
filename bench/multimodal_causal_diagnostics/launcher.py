"""Standard-library bootstrap for one Landlock/seccomp MM-009 worker."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

SCHEMA_VERSION: Final = "mm009-source-launcher-v1"
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

# These are host-provided, read-only trust roots.  The scientific Python dependency
# root is *not* one of them: formal MM-009 prepares a separate tree containing only
# ``numpy`` and ``numpy.libs`` and passes that tree as an explicit launcher argument.
# CPython startup and this standard-library-only bootstrap necessarily run from the
# live base stdlib. ``run`` then rebinds sys.path and startup-loaded package paths to
# the audited copied stdlib before scientific imports and before Landlock entry. The
# dynamic-loader roots are an operating-system ABI boundary, not a claim that their
# contents are sealed by MM-009.
_DYNAMIC_LOADER_CANDIDATES: Final = (
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/lib64"),
    Path("/etc/ld.so.cache"),
)
_OPERATIONAL_READ_CANDIDATES: Final = (
    Path("/dev/null"),
    Path("/sys/devices/system/cpu"),
)


class LauncherError(RuntimeError):
    """Raised before scientific imports when launcher authority differs."""


def frozen_environment() -> dict[str, str]:
    return dict(_ENVIRONMENT)


def _existing_non_symlinks(candidates: tuple[Path, ...]) -> tuple[Path, ...]:
    output: list[Path] = []
    for path in candidates:
        if path.exists() and not path.is_symlink() and path not in output:
            output.append(path)
    return tuple(output)


def _validate_stdlib_root(path: Path) -> None:
    if stat.S_IMODE(path.lstat().st_mode) != 0o555:
        raise LauncherError("copied stdlib root must be mode 0555")
    required = (path / "os.py", path / "lib-dynload")
    if any(item.is_symlink() or not item.exists() for item in required):
        raise LauncherError("copied stdlib root membership differs")
    if (path / "site-packages").exists():
        raise LauncherError("copied stdlib unexpectedly contains site-packages")


def standard_library_import_roots(stdlib_root: Path) -> tuple[Path, ...]:
    """Return the exact copied stdlib roots admitted after ``python -I -S``."""

    _validate_stdlib_root(stdlib_root)
    return (stdlib_root, stdlib_root / "lib-dynload")


def _rebind_loaded_stdlib_packages(stdlib_root: Path) -> None:
    """Move startup-loaded package search paths off the live base stdlib."""

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
                raise LauncherError(f"startup stdlib package is absent from copied closure: {relative}")
            rebound.append(str(candidate))
            changed = True
        if changed:
            module.__path__ = rebound


def host_runtime_trust_roots() -> Mapping[str, tuple[Path, ...]]:
    """Expose the explicit read-only host boundary used by the sandbox.

    ``dynamic_loader`` supplies system ABI libraries needed by CPython/NumPy
    extensions. ``operational`` contains the non-code kernel/device views NumPy may
    inspect. The copied stdlib is separately bound and is not a host trust root. No
    live site-packages or whole Python installation prefix is admitted.
    """

    return {
        "dynamic_loader": _existing_non_symlinks(_DYNAMIC_LOADER_CANDIDATES),
        "operational": _existing_non_symlinks(_OPERATIONAL_READ_CANDIDATES),
    }


def _absolute_existing(value: str, label: str, *, directory: bool | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.exists():
        raise LauncherError(f"{label} must be an absolute existing non-symlink path")
    if directory is True and not path.is_dir():
        raise LauncherError(f"{label} must be a directory")
    if directory is False and not path.is_file():
        raise LauncherError(f"{label} must be a regular file")
    return path


def _validate_dependency_root(path: Path) -> None:
    if stat.S_IMODE(path.lstat().st_mode) != 0o555:
        raise LauncherError("NumPy dependency root must be mode 0555")
    if {entry.name for entry in os.scandir(path)} != {"numpy", "numpy.libs"}:
        raise LauncherError("NumPy dependency root membership differs")
    for name in ("numpy", "numpy.libs"):
        package = path / name
        if package.is_symlink() or not package.is_dir():
            raise LauncherError("NumPy dependency package differs")


def _runtime_read_paths(
    runtime: Path,
    dependency_root: Path,
    stdlib_root: Path,
    source: Path,
    config: Path,
) -> tuple[Path, ...]:
    trust = host_runtime_trust_roots()
    return _existing_non_symlinks(
        (
            runtime,
            dependency_root,
            *standard_library_import_roots(stdlib_root),
            source,
            config,
            *trust["dynamic_loader"],
            *trust["operational"],
        )
    )


def _sealed_sys_path(runtime: Path, dependency_root: Path, stdlib_root: Path) -> list[str]:
    """Return the closed import path used after ``python -I -S`` startup."""

    candidates = (
        runtime,
        dependency_root,
        *standard_library_import_roots(stdlib_root),
    )
    paths = [str(path) for path in candidates if path.exists() and not path.is_symlink()]
    if str(runtime) not in paths or str(dependency_root) not in paths:
        raise LauncherError("sealed Python import roots are unavailable")
    return paths


def run(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 6:
        raise LauncherError("launcher requires SOURCE CONFIG OUTPUT RUNTIME DEPENDENCY_ROOT STDLIB_ROOT")
    source = _absolute_existing(arguments[0], "source row", directory=False)
    config = _absolute_existing(arguments[1], "worker config", directory=False)
    output = _absolute_existing(arguments[2], "worker output", directory=True)
    runtime = _absolute_existing(arguments[3], "sealed runtime", directory=True)
    dependency_root = _absolute_existing(arguments[4], "NumPy dependency root", directory=True)
    stdlib_root = _absolute_existing(arguments[5], "copied stdlib root", directory=True)
    _validate_dependency_root(dependency_root)
    _validate_stdlib_root(stdlib_root)
    if any(output.iterdir()):
        raise LauncherError("worker output directory must start empty")
    if Path(__file__).resolve().parent.parent.parent != runtime:
        raise LauncherError("launcher was not imported from the declared shadow runtime")
    if os.environ != _ENVIRONMENT:
        raise LauncherError("launcher environment differs from the frozen allowlist")

    sys.path[:] = _sealed_sys_path(runtime, dependency_root, stdlib_root)
    _rebind_loaded_stdlib_packages(stdlib_root)
    os.chdir(output)
    os.umask(0o077)
    from bench.multimodal_causal_diagnostics import sandbox

    sandbox.enter_source_only_sandbox(
        _runtime_read_paths(runtime, dependency_root, stdlib_root, source, config), output
    )
    from bench.multimodal_causal_diagnostics import worker

    worker.run_source_row(source, config, output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())


__all__ = [
    "LauncherError",
    "SCHEMA_VERSION",
    "frozen_environment",
    "host_runtime_trust_roots",
    "run",
    "standard_library_import_roots",
]
