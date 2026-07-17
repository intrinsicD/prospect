"""Minimal ``python -I -S`` bootstrap for the fitting-free MM-011 future gate."""

from __future__ import annotations

import importlib
import os
import resource
import stat
import sys
from pathlib import Path
from typing import cast


class FutureIsolationLauncherError(RuntimeError):
    pass


MAX_FUTURE_FILE_BYTES = 1_000_000


def _import_roots(runtime: Path, dependency_root: Path, stdlib_root: Path) -> list[str]:
    candidates = (
        runtime,
        dependency_root,
        stdlib_root,
        stdlib_root / "lib-dynload",
    )
    return [str(path) for path in candidates if path.exists() and not path.is_symlink()]


def _validate_dependency_root(path: Path) -> None:
    if stat.S_IMODE(path.lstat().st_mode) != 0o555:
        raise FutureIsolationLauncherError("NumPy dependency root must be mode 0555")
    if {entry.name for entry in os.scandir(path)} != {"numpy", "numpy.libs"}:
        raise FutureIsolationLauncherError("NumPy dependency root membership differs")
    for name in ("numpy", "numpy.libs"):
        package = path / name
        if package.is_symlink() or not package.is_dir():
            raise FutureIsolationLauncherError("NumPy dependency package differs")


def _validate_stdlib_root(path: Path) -> None:
    if stat.S_IMODE(path.lstat().st_mode) != 0o555:
        raise FutureIsolationLauncherError("copied stdlib root must be mode 0555")
    if (
        not (path / "os.py").is_file()
        or (path / "os.py").is_symlink()
        or not (path / "lib-dynload").is_dir()
        or (path / "lib-dynload").is_symlink()
        or (path / "site-packages").exists()
    ):
        raise FutureIsolationLauncherError("copied stdlib root membership differs")


def _absolute_existing(value: str, label: str, *, directory: bool) -> Path:
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FutureIsolationLauncherError(f"{label} must exist") from error
    if not path.is_absolute() or path != resolved or path.is_symlink():
        raise FutureIsolationLauncherError(f"{label} must be a canonical absolute non-symlink path")
    if directory and not path.is_dir():
        raise FutureIsolationLauncherError(f"{label} must be a directory")
    if not directory and not path.is_file():
        raise FutureIsolationLauncherError(f"{label} must be a regular file")
    return path


def _private_output(value: str) -> tuple[Path, Path]:
    output = Path(value)
    parent = output.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as error:
        raise FutureIsolationLauncherError("output parent must exist") from error
    if (
        not output.is_absolute()
        or parent != resolved_parent
        or parent.is_symlink()
        or not parent.is_dir()
        or output.exists()
        or output.is_symlink()
        or any(parent.iterdir())
    ):
        raise FutureIsolationLauncherError("future-isolation output must be absent in a fresh canonical directory")
    return output, parent


def _readable_paths(
    runtime: Path,
    dependency_root: Path,
    stdlib_root: Path,
    freeze: Path,
    target_root: Path,
    prediction_root: Path,
) -> tuple[Path, ...]:
    from bench.multimodal_causal_assay import launcher

    trust = launcher.host_runtime_trust_roots()
    candidates = (
        runtime,
        dependency_root,
        *launcher.standard_library_import_roots(stdlib_root),
        freeze,
        target_root,
        prediction_root,
        *trust["dynamic_loader"],
        *trust["operational"],
    )
    output: list[Path] = []
    for path in candidates:
        if path not in output:
            output.append(path)
    return tuple(output)


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
                raise FutureIsolationLauncherError(f"startup stdlib package is absent from copied closure: {relative}")
            rebound.append(str(candidate))
            changed = True
        if changed:
            module.__path__ = rebound


def _assert_unimportable(module_names: set[str]) -> None:
    for name in sorted(module_names):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name is None or not (error.name == name or name.startswith(f"{error.name}.")):
                raise FutureIsolationLauncherError(
                    f"excluded module began importing before a dependency failed: {name}"
                ) from error
            continue
        raise FutureIsolationLauncherError(f"excluded fitting module remained importable: {name}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 7:
        raise FutureIsolationLauncherError(
            "future-isolation launcher requires RUNTIME DEPENDENCY_ROOT STDLIB_ROOT "
            "FREEZE TARGET_ROOT PREDICTION_ROOT OUTPUT"
        )
    runtime = _absolute_existing(arguments[0], "custody runtime", directory=True)
    dependency_root = _absolute_existing(arguments[1], "NumPy dependency root", directory=True)
    stdlib_root = _absolute_existing(arguments[2], "copied stdlib root", directory=True)
    freeze = _absolute_existing(arguments[3], "prediction freeze", directory=False)
    target_root = _absolute_existing(arguments[4], "target root", directory=True)
    prediction_root = _absolute_existing(arguments[5], "prediction root", directory=True)
    output, output_directory = _private_output(arguments[6])
    if Path(__file__).resolve().parent.parent.parent != runtime:
        raise FutureIsolationLauncherError("launcher is outside the declared custody runtime")
    _validate_dependency_root(dependency_root)
    _validate_stdlib_root(stdlib_root)
    sys.path[:] = _import_roots(runtime, dependency_root, stdlib_root)
    _rebind_loaded_stdlib_packages(stdlib_root)
    from bench.multimodal_causal_assay import launcher, sandbox

    os.environ.clear()
    os.environ.update(launcher.frozen_environment())
    os.chdir(output_directory)
    os.umask(0o077)
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (MAX_FUTURE_FILE_BYTES, MAX_FUTURE_FILE_BYTES),
    )
    forbidden = {
        "bench.multimodal_causal_assay.predictor",
        "bench.multimodal_causal_assay.worker",
        "bench.multimodal_mechanism_diagnostics.fitting_v22",
        "bench.multimodal_mechanism_diagnostics.geometry_v22",
        "bench.multimodal_mechanism_diagnostics.global_v22",
        "bench.multimodal_mechanism_diagnostics.nongrid_v22",
    }
    if forbidden.intersection(sys.modules):
        raise FutureIsolationLauncherError("launcher started with a fitting module loaded")
    sandbox.enter_source_only_sandbox(
        _readable_paths(
            runtime,
            dependency_root,
            stdlib_root,
            freeze,
            target_root,
            prediction_root,
        ),
        output_directory,
    )
    _assert_unimportable(forbidden)
    from bench.multimodal_causal_assay import future_isolation

    if forbidden != set(future_isolation.FORBIDDEN_MODULES) or forbidden.intersection(sys.modules):
        raise FutureIsolationLauncherError("future-isolation import graph loaded a fitting module")
    result = future_isolation.main([str(freeze), str(target_root), str(prediction_root), str(output)])
    if forbidden.intersection(sys.modules):
        raise FutureIsolationLauncherError("future-isolation execution loaded a fitting module")
    if (
        set(output_directory.iterdir()) != {output}
        or stat.S_IMODE(output.lstat().st_mode) != 0o444
        or output.lstat().st_size > MAX_FUTURE_FILE_BYTES
    ):
        raise FutureIsolationLauncherError("future-isolation output census differs")
    return cast(int, result)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["FutureIsolationLauncherError", "main"]
