"""Sealed shadow-runtime construction and one-row process supervision for MM-009."""

from __future__ import annotations

import hashlib
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, Final, cast

from . import launcher, records

SCHEMA_VERSION: Final = "mm009-shadow-runtime-v1"
CUSTODY_SCHEMA_VERSION: Final = "mm009-custody-runtime-v1"
DEPENDENCY_SCHEMA_VERSION: Final = "mm009-numpy-dependency-closure-v1"
DEPENDENCY_SOURCE_SCHEMA_VERSION: Final = "mm009-numpy-dependency-source-v1"
STDLIB_SCHEMA_VERSION: Final = "mm009-stdlib-closure-v1"
STDLIB_SOURCE_SCHEMA_VERSION: Final = "mm009-stdlib-source-v1"
CUSTODY_SOURCE_SCHEMA_VERSION: Final = "mm009-custody-runtime-source-v1"
HOST_TRUST_SCHEMA_VERSION: Final = "mm009-host-runtime-trust-v1"
REPO_ROOT: Final = Path(__file__).resolve().parents[2]
# Source location only.  Formal children must never receive this broad tree as an
# import root; ``build_numpy_dependency_closure`` copies its two admitted package
# directories into a dedicated immutable tree first.
SITE_PACKAGES: Final = REPO_ROOT / ".venv/lib/python3.12/site-packages"
STDLIB_SOURCE: Final = Path(sys.base_prefix) / "lib" / (f"python{sys.version_info.major}.{sys.version_info.minor}")
DEPENDENCY_PACKAGES: Final = (Path("numpy"), Path("numpy.libs"))
STDLIB_EXCLUDED_DIRECTORY_NAMES: Final = ("__pycache__", "site-packages")
LAUNCHER_RELATIVE: Final = Path("bench/multimodal_causal_diagnostics/launcher.py")
PROBE_RELATIVE: Final = Path("bench/multimodal_causal_diagnostics/probe.py")
FUTURE_ISOLATION_LAUNCHER_RELATIVE: Final = Path("bench/multimodal_causal_diagnostics/future_isolation_launcher.py")
SCORE_LAUNCHER_RELATIVE: Final = Path("bench/multimodal_causal_diagnostics/score_launcher.py")

_COPIED_SOURCES: Final[Mapping[Path, Path]] = {
    Path("bench/multimodal_causal_diagnostics/launcher.py"): Path("bench/multimodal_causal_diagnostics/launcher.py"),
    Path("bench/multimodal_causal_diagnostics/predictor.py"): Path("bench/multimodal_causal_diagnostics/predictor.py"),
    Path("bench/multimodal_causal_diagnostics/preparation.py"): Path(
        "bench/multimodal_causal_diagnostics/preparation.py"
    ),
    Path("bench/multimodal_causal_diagnostics/probe.py"): Path("bench/multimodal_causal_diagnostics/probe.py"),
    Path("bench/multimodal_causal_diagnostics/records.py"): Path("bench/multimodal_causal_diagnostics/records.py"),
    Path("bench/multimodal_causal_diagnostics/sandbox.py"): Path("bench/multimodal_causal_diagnostics/sandbox.py"),
    Path("bench/multimodal_causal_diagnostics/worker.py"): Path("bench/multimodal_causal_diagnostics/worker.py"),
    Path("bench/multimodal_mechanism_diagnostics/fitting_v22.py"): Path(
        "bench/multimodal_mechanism_diagnostics/fitting_v22.py"
    ),
    Path("bench/multimodal_mechanism_diagnostics/geometry_v22.py"): Path(
        "bench/multimodal_mechanism_diagnostics/geometry_v22.py"
    ),
    Path("bench/multimodal_mechanism_diagnostics/global_v22.py"): Path(
        "bench/multimodal_mechanism_diagnostics/global_v22.py"
    ),
    Path("bench/multimodal_mechanism_diagnostics/nongrid_v22.py"): Path(
        "bench/multimodal_mechanism_diagnostics/nongrid_v22.py"
    ),
}
_EMPTY_INITIALIZERS: Final = (
    Path("bench/__init__.py"),
    Path("bench/multimodal_causal_diagnostics/__init__.py"),
    Path("bench/multimodal_mechanism_diagnostics/__init__.py"),
)
RUNTIME_FILES: Final = tuple(sorted((*_COPIED_SOURCES, *_EMPTY_INITIALIZERS), key=str))

_CUSTODY_COPIED_SOURCES: Final[Mapping[Path, Path]] = {
    Path("bench/multimodal_causal_diagnostics/future_isolation.py"): Path(
        "bench/multimodal_causal_diagnostics/future_isolation.py"
    ),
    FUTURE_ISOLATION_LAUNCHER_RELATIVE: FUTURE_ISOLATION_LAUNCHER_RELATIVE,
    Path("bench/multimodal_causal_diagnostics/launcher.py"): Path("bench/multimodal_causal_diagnostics/launcher.py"),
    Path("bench/multimodal_causal_diagnostics/preparation.py"): Path(
        "bench/multimodal_causal_diagnostics/preparation.py"
    ),
    Path("bench/multimodal_causal_diagnostics/records.py"): Path("bench/multimodal_causal_diagnostics/records.py"),
    Path("bench/multimodal_causal_diagnostics/sandbox.py"): Path("bench/multimodal_causal_diagnostics/sandbox.py"),
    SCORE_LAUNCHER_RELATIVE: SCORE_LAUNCHER_RELATIVE,
    Path("bench/multimodal_causal_diagnostics/score_worker.py"): Path(
        "bench/multimodal_causal_diagnostics/score_worker.py"
    ),
    Path("bench/multimodal_causal_diagnostics/scoring.py"): Path("bench/multimodal_causal_diagnostics/scoring.py"),
}
_CUSTODY_EMPTY_INITIALIZERS: Final = (
    Path("bench/__init__.py"),
    Path("bench/multimodal_causal_diagnostics/__init__.py"),
)
CUSTODY_RUNTIME_FILES: Final = tuple(sorted((*_CUSTODY_COPIED_SOURCES, *_CUSTODY_EMPTY_INITIALIZERS), key=str))


class RuntimeValidationError(ValueError):
    """Raised when the shadow runtime or predictor process violates its contract."""


class WorkerProcessRegistry:
    """Thread-safe, one-shot cancellation registry for a formal predictor batch."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    @property
    def active_pids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._processes))

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def _register(self, process: subprocess.Popen[bytes]) -> bool:
        with self._lock:
            if self._cancelled.is_set():
                return False
            if process.pid in self._processes:
                raise RuntimeValidationError("worker process PID was registered twice")
            self._processes[process.pid] = process
            return True

    def _unregister(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._processes.pop(process.pid, None)

    def cancel_all(self) -> None:
        """Permanently cancel the batch and promptly signal every process group."""

        self._cancelled.set()
        with self._lock:
            processes = tuple(self._processes.values())
        for process in processes:
            _signal_process_group(process.pid, signal.SIGTERM)


def _signal_process_group(process_group_id: int, action: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, action)
    except ProcessLookupError:
        return


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate, then kill, the entire fresh-session process group."""

    _signal_process_group(process.pid, signal.SIGTERM)
    grace_deadline = time.monotonic() + 0.25
    while _process_group_exists(process.pid) and time.monotonic() < grace_deadline:
        time.sleep(0.01)
    if _process_group_exists(process.pid):
        _signal_process_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as error:
            raise RuntimeValidationError("worker process group could not be reaped") from error


def _bounded_stream_tail(stream: BinaryIO, *, maximum_bytes: int = 4000) -> tuple[int, str]:
    descriptor = stream.fileno()
    size = os.fstat(descriptor).st_size
    stream.seek(max(0, size - maximum_bytes))
    payload = stream.read(maximum_bytes)
    if not isinstance(payload, bytes):
        raise RuntimeValidationError("worker capture stream is not binary")
    return size, payload.decode("utf-8", errors="replace")


def _run_registered_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: int,
    process_registry: WorkerProcessRegistry,
) -> tuple[int, int, str, int, str]:
    """Run one child with monotonic cancellation and bounded stream collection."""

    with tempfile.TemporaryFile(mode="w+b") as stdout_stream, tempfile.TemporaryFile(mode="w+b") as stderr_stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=launcher.frozen_environment(),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
        )
        if not process_registry._register(process):
            _terminate_process_group(process)
            raise RuntimeValidationError("source-only worker cancelled before registration")
        deadline = time.monotonic() + timeout_seconds
        try:
            while process.poll() is None:
                if process_registry.cancelled:
                    _terminate_process_group(process)
                    raise RuntimeValidationError("source-only worker cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process_registry.cancel_all()
                    _terminate_process_group(process)
                    raise RuntimeValidationError("source-only worker timed out")
                time.sleep(min(0.05, remaining))
            if process_registry.cancelled:
                _terminate_process_group(process)
                raise RuntimeValidationError("source-only worker cancelled")
            returncode = process.wait()
            if returncode != 0:
                process_registry.cancel_all()
                _terminate_process_group(process)
            elif _process_group_exists(process.pid):
                process_registry.cancel_all()
                _terminate_process_group(process)
                raise RuntimeValidationError("source-only worker left a live descendant process")
            stdout_bytes, stdout_tail = _bounded_stream_tail(stdout_stream)
            stderr_bytes, stderr_tail = _bounded_stream_tail(stderr_stream)
            return returncode, stdout_bytes, stdout_tail, stderr_bytes, stderr_tail
        finally:
            if process.poll() is None or _process_group_exists(process.pid):
                _terminate_process_group(process)
            process_registry._unregister(process)


def _regular_directory(path: str | Path, label: str, *, empty: bool = False) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeValidationError(f"{label} must be a real directory")
    if empty and any(candidate.iterdir()):
        raise RuntimeValidationError(f"{label} must be empty")
    return candidate


def _read_source_file(
    path: Path,
    *,
    retain_bytes: bool,
) -> tuple[dict[str, records.JsonValue], bytes | None]:
    """Read stable source bytes while admitting package-manager hard links."""

    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeValidationError(f"source closure file is missing: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeValidationError(f"source closure contains a symlink or special file: {path}")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        stat.S_IFMT(before.st_mode),
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            stat.S_IFMT(opened.st_mode),
            opened.st_nlink,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != before_identity:
            raise RuntimeValidationError(f"source closure file identity changed before open: {path}")
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeValidationError(f"source closure file ended before its bound size: {path}")
            digest.update(chunk)
            if retain_bytes:
                chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeValidationError(f"source closure file grew while being read: {path}")
        after_read = os.fstat(descriptor)
        try:
            after_path = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeValidationError(f"source closure file disappeared while being read: {path}") from error
        after_read_identity = (
            after_read.st_dev,
            after_read.st_ino,
            after_read.st_size,
            stat.S_IFMT(after_read.st_mode),
            after_read.st_nlink,
            after_read.st_mtime_ns,
            after_read.st_ctime_ns,
        )
        after_path_identity = (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            stat.S_IFMT(after_path.st_mode),
            after_path.st_nlink,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        if after_read_identity != opened_identity or after_path_identity != opened_identity:
            raise RuntimeValidationError(f"source closure file changed while being read: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks) if retain_bytes else None
    return {"bytes": opened.st_size, "sha256": digest.hexdigest()}, payload


def _source_file_record(path: Path) -> dict[str, records.JsonValue]:
    """Hash one source file while admitting stable package-manager hard links."""

    record, _ = _read_source_file(path, retain_bytes=False)
    return record


def _copy_source_file(source: Path, destination: Path) -> None:
    """Dereference an admitted source hard link into a unique immutable file."""

    _, payload = _read_source_file(source, retain_bytes=True)
    if payload is None:  # pragma: no cover - internal contract guard
        raise RuntimeValidationError("source-copy reader did not retain bytes")
    records.write_immutable_bytes_exclusive(destination, payload)


def _source_manifest(
    root: Path,
    files: tuple[Path, ...],
    directories: tuple[Path, ...],
    *,
    schema_version: str,
    source_hardlinks_dereferenced: bool,
    extra: Mapping[str, records.JsonValue] | None = None,
) -> dict[str, records.JsonValue]:
    artifacts: dict[str, records.JsonValue] = {}
    total_bytes = 0
    for relative in files:
        record = _source_file_record(root / relative)
        artifacts[str(relative)] = record
        total_bytes += cast(int, record["bytes"])
    body: dict[str, records.JsonValue] = {
        "artifacts": artifacts,
        "directories": [str(relative) for relative in directories],
        "directory_count": len(directories),
        "file_count": len(files),
        "schema_version": schema_version,
        "source_hardlinks_dereferenced": source_hardlinks_dereferenced,
        "total_bytes": total_bytes,
    }
    if extra is not None:
        body.update(extra)
    body["manifest_sha256"] = hashlib.sha256(records.canonical_json_bytes(body)).hexdigest()
    return body


def _require_expected_manifest(actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    try:
        checked = records.require_sha256(expected, label)
    except records.RecordValidationError as error:
        raise RuntimeValidationError(f"{label} differs") from error
    if actual != checked:
        raise RuntimeValidationError(f"{label} differs from the prebound audit value")


def _prepared_dependency_root(path: str | Path) -> Path:
    root = _regular_directory(path, "NumPy dependency closure")
    if stat.S_IMODE(root.lstat().st_mode) != 0o555:
        raise RuntimeValidationError("NumPy dependency root must be mode 0555")
    if {entry.name for entry in os.scandir(root)} != {str(item) for item in DEPENDENCY_PACKAGES}:
        raise RuntimeValidationError("NumPy dependency top-level membership differs")
    for package in DEPENDENCY_PACKAGES:
        candidate = root / package
        if candidate.is_symlink() or not candidate.is_dir():
            raise RuntimeValidationError(f"NumPy dependency package differs: {package}")
    return root


def _dependency_members(
    root: Path,
    *,
    require_unique_files: bool = True,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    files: list[Path] = []
    directories: list[Path] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            path = current_path / dirname
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeValidationError("NumPy dependency closure contains a non-directory or symlink")
            directories.append(path.relative_to(root))
        for filename in filenames:
            path = current_path / filename
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or (require_unique_files and metadata.st_nlink != 1)
            ):
                raise RuntimeValidationError("NumPy dependency closure contains a non-unique file or symlink")
            files.append(path.relative_to(root))
    return tuple(sorted(files, key=str)), tuple(sorted(directories, key=str))


def _source_dependency_census(source_root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    source = _regular_directory(source_root, "NumPy dependency source")
    for package in DEPENDENCY_PACKAGES:
        path = source / package
        if path.is_symlink() or not path.is_dir():
            raise RuntimeValidationError(f"required NumPy dependency package is missing: {package}")
    files: list[Path] = []
    directories: list[Path] = []
    for package in DEPENDENCY_PACKAGES:
        package_files, package_directories = _dependency_members(
            source / package,
            require_unique_files=False,
        )
        directories.append(package)
        directories.extend(package / relative for relative in package_directories)
        files.extend(package / relative for relative in package_files)
    return tuple(sorted(files, key=str)), tuple(sorted(directories, key=str))


def numpy_dependency_source_manifest(
    source_root: str | Path = SITE_PACKAGES,
) -> dict[str, records.JsonValue]:
    """Bind the live NumPy inputs by membership, byte size, and SHA-256."""

    source = _regular_directory(source_root, "NumPy dependency source")
    files, directories = _source_dependency_census(source)
    return _source_manifest(
        source,
        files,
        directories,
        schema_version=DEPENDENCY_SOURCE_SCHEMA_VERSION,
        source_hardlinks_dereferenced=True,
        extra={"packages": [str(package) for package in DEPENDENCY_PACKAGES]},
    )


def build_numpy_dependency_closure(
    destination: str | Path,
    *,
    source_root: str | Path = SITE_PACKAGES,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, records.JsonValue]:
    """Copy only ``numpy`` and ``numpy.libs`` into a new read-only import root.

    Every regular file is copied opaquely with mode 0444.  Every directory is made
    mode 0555 after construction.  Source symlinks, hard links, special files, and
    membership changes fail closed.  The returned manifest binds the complete tree
    and is intended to be stored once by the formal supervisor.
    """

    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise RuntimeValidationError("NumPy dependency destination must not exist")
    source = _regular_directory(source_root, "NumPy dependency source")
    source_manifest = numpy_dependency_source_manifest(source)
    _require_expected_manifest(
        cast(str, source_manifest["manifest_sha256"]),
        expected_source_manifest_sha256,
        "NumPy source-manifest SHA-256",
    )
    files, directories = _source_dependency_census(source)
    records.ensure_directory(root)
    for relative in directories:
        records.ensure_directory(root / relative)
    for relative in files:
        _copy_source_file(source / relative, root / relative)
    for relative in sorted(directories, key=lambda value: (-len(value.parts), str(value))):
        os.chmod(root / relative, 0o555, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)
    manifest = numpy_dependency_manifest(root)
    if manifest["source_manifest_sha256"] != source_manifest["manifest_sha256"]:
        raise RuntimeValidationError("copied NumPy closure differs from its prebound source")
    return manifest


def numpy_dependency_manifest(root: str | Path) -> dict[str, records.JsonValue]:
    """Validate and bind an exact prepared NumPy-only dependency root."""

    dependency = _prepared_dependency_root(root)
    files, directories = _dependency_members(dependency)
    if not files or not all(package in directories for package in DEPENDENCY_PACKAGES):
        raise RuntimeValidationError("NumPy dependency closure census differs")
    for relative in directories:
        if stat.S_IMODE((dependency / relative).lstat().st_mode) != 0o555:
            raise RuntimeValidationError(f"NumPy dependency directory is not mode 0555: {relative}")
    artifacts: dict[str, records.JsonValue] = {}
    total_bytes = 0
    for relative in files:
        record = records.file_record(dependency / relative)
        if record["mode"] != 0o444:
            raise RuntimeValidationError(f"NumPy dependency file is not mode 0444: {relative}")
        artifacts[str(relative)] = record
        total_bytes += cast(int, record["bytes"])
    body: dict[str, records.JsonValue] = {
        "artifacts": artifacts,
        "directories": [str(relative) for relative in directories],
        "directory_count": len(directories),
        "file_count": len(files),
        "packages": [str(package) for package in DEPENDENCY_PACKAGES],
        "schema_version": DEPENDENCY_SCHEMA_VERSION,
        "source_manifest_sha256": _source_manifest(
            dependency,
            files,
            directories,
            schema_version=DEPENDENCY_SOURCE_SCHEMA_VERSION,
            source_hardlinks_dereferenced=True,
            extra={"packages": [str(package) for package in DEPENDENCY_PACKAGES]},
        )["manifest_sha256"],
        "total_bytes": total_bytes,
    }
    body["manifest_sha256"] = hashlib.sha256(records.canonical_json_bytes(body)).hexdigest()
    return body


def _stdlib_source_census(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    source = _regular_directory(root, "stdlib source")
    files: list[Path] = []
    directories: list[Path] = []
    for current, dirnames, filenames in os.walk(source, followlinks=False):
        current_path = Path(current)
        admitted_directories: list[str] = []
        for dirname in sorted(dirnames):
            if dirname in STDLIB_EXCLUDED_DIRECTORY_NAMES:
                continue
            path = current_path / dirname
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeValidationError(f"stdlib source contains a link or special: {path}")
            admitted_directories.append(dirname)
            directories.append(path.relative_to(source))
        dirnames[:] = admitted_directories
        for filename in sorted(filenames):
            if filename.endswith((".pyc", ".pyo")):
                continue
            path = current_path / filename
            _source_file_record(path)
            files.append(path.relative_to(source))
    return tuple(sorted(files, key=str)), tuple(sorted(directories, key=str))


def stdlib_source_manifest(
    source_root: str | Path = STDLIB_SOURCE,
) -> dict[str, records.JsonValue]:
    """Bind the admitted live stdlib bytes while excluding package directories."""

    source = _regular_directory(source_root, "stdlib source")
    files, directories = _stdlib_source_census(source)
    return _source_manifest(
        source,
        files,
        directories,
        schema_version=STDLIB_SOURCE_SCHEMA_VERSION,
        source_hardlinks_dereferenced=True,
        extra={
            "excluded_directory_names": list(STDLIB_EXCLUDED_DIRECTORY_NAMES),
            "excluded_file_suffixes": [".pyc", ".pyo"],
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    )


def _prepared_stdlib_root(path: str | Path) -> Path:
    root = _regular_directory(path, "copied stdlib closure")
    if stat.S_IMODE(root.lstat().st_mode) != 0o555:
        raise RuntimeValidationError("copied stdlib root must be mode 0555")
    if (
        not (root / "os.py").is_file()
        or (root / "os.py").is_symlink()
        or not (root / "lib-dynload").is_dir()
        or (root / "lib-dynload").is_symlink()
        or (root / "site-packages").exists()
    ):
        raise RuntimeValidationError("copied stdlib root membership differs")
    return root


def build_stdlib_closure(
    destination: str | Path,
    *,
    source_root: str | Path = STDLIB_SOURCE,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, records.JsonValue]:
    """Copy the stdlib without site-packages, caches, bytecode, links, or specials."""

    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise RuntimeValidationError("stdlib closure destination must not exist")
    source = _regular_directory(source_root, "stdlib source")
    source_manifest = stdlib_source_manifest(source)
    _require_expected_manifest(
        cast(str, source_manifest["manifest_sha256"]),
        expected_source_manifest_sha256,
        "stdlib source-manifest SHA-256",
    )
    files, directories = _stdlib_source_census(source)
    records.ensure_directory(root)
    for relative in directories:
        records.ensure_directory(root / relative)
    for relative in files:
        _copy_source_file(source / relative, root / relative)
    for relative in sorted(directories, key=lambda value: (-len(value.parts), str(value))):
        os.chmod(root / relative, 0o555, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)
    manifest = stdlib_manifest(root)
    if manifest["source_manifest_sha256"] != source_manifest["manifest_sha256"]:
        raise RuntimeValidationError("copied stdlib closure differs from its prebound source")
    return manifest


def stdlib_manifest(root: str | Path) -> dict[str, records.JsonValue]:
    """Validate and bind the immutable copied standard-library closure."""

    stdlib = _prepared_stdlib_root(root)
    files, directories = _dependency_members(stdlib)
    if not files or not directories:
        raise RuntimeValidationError("copied stdlib census is empty")
    for relative in directories:
        if any(part in STDLIB_EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            raise RuntimeValidationError("copied stdlib contains an excluded directory")
        if stat.S_IMODE((stdlib / relative).lstat().st_mode) != 0o555:
            raise RuntimeValidationError(f"copied stdlib directory is not mode 0555: {relative}")
    artifacts: dict[str, records.JsonValue] = {}
    total_bytes = 0
    for relative in files:
        if relative.suffix in {".pyc", ".pyo"}:
            raise RuntimeValidationError("copied stdlib contains excluded bytecode")
        record = records.file_record(stdlib / relative)
        if record["mode"] != 0o444:
            raise RuntimeValidationError(f"copied stdlib file is not mode 0444: {relative}")
        artifacts[str(relative)] = record
        total_bytes += cast(int, record["bytes"])
    source_manifest = _source_manifest(
        stdlib,
        files,
        directories,
        schema_version=STDLIB_SOURCE_SCHEMA_VERSION,
        source_hardlinks_dereferenced=True,
        extra={
            "excluded_directory_names": list(STDLIB_EXCLUDED_DIRECTORY_NAMES),
            "excluded_file_suffixes": [".pyc", ".pyo"],
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    )
    body: dict[str, records.JsonValue] = {
        "artifacts": artifacts,
        "directories": [str(relative) for relative in directories],
        "directory_count": len(directories),
        "file_count": len(files),
        "schema_version": STDLIB_SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "total_bytes": total_bytes,
    }
    body["manifest_sha256"] = hashlib.sha256(records.canonical_json_bytes(body)).hexdigest()
    return body


def host_runtime_trust_manifest() -> dict[str, records.JsonValue]:
    """Describe the explicit host bootstrap/interpreter/ABI trust boundary."""

    roots = launcher.host_runtime_trust_roots()
    interpreter = Path(sys.executable).resolve()
    return {
        "authority_model": "trusted-live-bootstrap-then-bound-copied-import-roots",
        "bootstrap": {
            "cache_tag": sys.implementation.cache_tag,
            "live_stdlib_root": str(STDLIB_SOURCE),
            "phase_boundary": (
                "python -I -S -B startup and launcher bootstrap imports use the live base stdlib; "
                "the launcher then rebinds sys.path and loaded package paths to the copied stdlib "
                "before scientific imports and before predictor Landlock entry"
            ),
            "version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
        },
        "interpreter": {
            "path": str(interpreter),
            "record": records.file_record(interpreter),
            "version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
        },
        "roots": {name: [str(path) for path in paths] for name, paths in roots.items()},
        "schema_version": HOST_TRUST_SCHEMA_VERSION,
    }


def live_python_roots() -> tuple[Path, ...]:
    """Return known live package trees that the formal sandbox must deny."""

    candidates = (
        SITE_PACKAGES,
        STDLIB_SOURCE / "site-packages",
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
    )
    output: list[Path] = []
    for path in candidates:
        if path.exists() and not path.is_symlink() and path not in output:
            output.append(path)
    if len(output) < 2:
        raise RuntimeValidationError("fewer than two live Python roots are available to probe")
    return tuple(output)


def source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for source in sorted(_COPIED_SOURCES, key=str):
        path = REPO_ROOT / source
        output[str(source)] = records.file_sha256(path)
    return output


def custody_source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for source in sorted(_CUSTODY_COPIED_SOURCES, key=str):
        output[str(source)] = records.file_sha256(REPO_ROOT / source)
    return output


def custody_runtime_source_manifest() -> dict[str, records.JsonValue]:
    """Prebind the exact live/virtual inputs of the fitting-free custody runtime."""

    artifacts: dict[str, records.JsonValue] = {}
    total_bytes = 0
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    for relative in CUSTODY_RUNTIME_FILES:
        if relative in _CUSTODY_EMPTY_INITIALIZERS:
            record: dict[str, records.JsonValue] = {"bytes": 0, "sha256": empty_sha256}
        else:
            record = _source_file_record(REPO_ROOT / _CUSTODY_COPIED_SOURCES[relative])
        artifacts[str(relative)] = record
        total_bytes += cast(int, record["bytes"])
    directories = tuple(
        sorted(
            (item for item in _expected_directories(CUSTODY_RUNTIME_FILES) if item != Path(".")),
            key=str,
        )
    )
    body: dict[str, records.JsonValue] = {
        "artifacts": artifacts,
        "directories": [str(relative) for relative in directories],
        "directory_count": len(directories),
        "file_count": len(CUSTODY_RUNTIME_FILES),
        "schema_version": CUSTODY_SOURCE_SCHEMA_VERSION,
        "source_hardlinks_dereferenced": True,
        "total_bytes": total_bytes,
    }
    body["manifest_sha256"] = hashlib.sha256(records.canonical_json_bytes(body)).hexdigest()
    return body


def build_shadow_runtime(destination: str | Path) -> dict[str, records.JsonValue]:
    """Copy the exact source-only closure into a new immutable shadow tree."""

    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise RuntimeValidationError("shadow runtime destination must not exist")
    records.ensure_directory(root)
    for relative in _EMPTY_INITIALIZERS:
        records.write_immutable_bytes_exclusive(root / relative, b"")
    for relative, source in _COPIED_SOURCES.items():
        records.copy_opaque_immutable_exclusive(REPO_ROOT / source, root / relative)
    _, directories = _members(root)
    for relative in sorted(directories, key=lambda value: (-len(value.parts), str(value))):
        os.chmod(root / relative, 0o555, follow_symlinks=False)
    return runtime_manifest(root)


def build_custody_runtime(
    destination: str | Path,
    *,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, records.JsonValue]:
    """Build the immutable fitting-free runtime used by target-custody children."""

    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise RuntimeValidationError("custody runtime destination must not exist")
    source_manifest = custody_runtime_source_manifest()
    _require_expected_manifest(
        cast(str, source_manifest["manifest_sha256"]),
        expected_source_manifest_sha256,
        "custody-runtime source-manifest SHA-256",
    )
    records.ensure_directory(root)
    for relative in _CUSTODY_EMPTY_INITIALIZERS:
        records.write_immutable_bytes_exclusive(root / relative, b"")
    for relative, source in _CUSTODY_COPIED_SOURCES.items():
        _copy_source_file(REPO_ROOT / source, root / relative)
    _, directories = _members(root)
    for relative in sorted(directories, key=lambda value: (-len(value.parts), str(value))):
        os.chmod(root / relative, 0o555, follow_symlinks=False)
    manifest = custody_runtime_manifest(root)
    if manifest["source_manifest_sha256"] != source_manifest["manifest_sha256"]:
        raise RuntimeValidationError("custody runtime differs from its prebound source")
    return manifest


def _members(root: Path) -> tuple[set[Path], set[Path]]:
    files: set[Path] = set()
    directories: set[Path] = {Path(".")}
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            path = current_path / dirname
            if path.is_symlink() or not path.is_dir():
                raise RuntimeValidationError("shadow runtime contains a non-directory or symlink")
            directories.add(path.relative_to(root))
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeValidationError("shadow runtime contains a non-file or symlink")
            files.add(path.relative_to(root))
    return files, directories


def _expected_directories(files: tuple[Path, ...]) -> set[Path]:
    expected = {Path(".")}
    for relative in files:
        parent = relative.parent
        while parent != Path("."):
            expected.add(parent)
            parent = parent.parent
    return expected


def runtime_manifest(root: str | Path) -> dict[str, records.JsonValue]:
    runtime = _regular_directory(root, "shadow runtime")
    files, directories = _members(runtime)
    if files != set(RUNTIME_FILES):
        raise RuntimeValidationError("shadow runtime file membership differs")
    if directories != _expected_directories(RUNTIME_FILES):
        raise RuntimeValidationError("shadow runtime directory membership differs")
    for relative in directories:
        if stat.S_IMODE((runtime / relative).lstat().st_mode) != 0o555:
            raise RuntimeValidationError("shadow runtime directories must be mode 0555")
    artifacts: dict[str, records.JsonValue] = {}
    for relative in RUNTIME_FILES:
        item = records.file_record(runtime / relative)
        if item["mode"] != 0o444:
            raise RuntimeValidationError("shadow runtime files must be mode 0444")
        artifacts[str(relative)] = item
    for relative in _EMPTY_INITIALIZERS:
        if records.read_regular_bytes(runtime / relative, maximum_bytes=0) != b"":
            raise RuntimeValidationError("shadow namespace initializers must be empty")
    for relative, source in _COPIED_SOURCES.items():
        if records.file_sha256(runtime / relative) != records.file_sha256(REPO_ROOT / source):
            raise RuntimeValidationError("shadow runtime differs from its bound live source")
    source_records: dict[str, records.JsonValue] = {name: digest for name, digest in source_hashes().items()}
    return {
        "artifacts": artifacts,
        "schema_version": SCHEMA_VERSION,
        "source_hashes": source_records,
    }


def custody_runtime_manifest(root: str | Path) -> dict[str, records.JsonValue]:
    """Validate the exact fitting-free import authority of a custody runtime."""

    runtime = _regular_directory(root, "custody runtime")
    files, directories = _members(runtime)
    if files != set(CUSTODY_RUNTIME_FILES):
        raise RuntimeValidationError("custody runtime file membership differs")
    if directories != _expected_directories(CUSTODY_RUNTIME_FILES):
        raise RuntimeValidationError("custody runtime directory membership differs")
    for relative in directories:
        if stat.S_IMODE((runtime / relative).lstat().st_mode) != 0o555:
            raise RuntimeValidationError("custody runtime directories must be mode 0555")
    artifacts: dict[str, records.JsonValue] = {}
    for relative in CUSTODY_RUNTIME_FILES:
        item = records.file_record(runtime / relative)
        if item["mode"] != 0o444:
            raise RuntimeValidationError("custody runtime files must be mode 0444")
        artifacts[str(relative)] = item
    for relative in _CUSTODY_EMPTY_INITIALIZERS:
        if records.read_regular_bytes(runtime / relative, maximum_bytes=0) != b"":
            raise RuntimeValidationError("custody namespace initializers must be empty")
    for relative, source in _CUSTODY_COPIED_SOURCES.items():
        if records.file_sha256(runtime / relative) != records.file_sha256(REPO_ROOT / source):
            raise RuntimeValidationError("custody runtime differs from its bound live source")
    source_records: dict[str, records.JsonValue] = {name: digest for name, digest in custody_source_hashes().items()}
    source_directories = tuple(sorted((item for item in directories if item != Path(".")), key=str))
    normalized_source = _source_manifest(
        runtime,
        CUSTODY_RUNTIME_FILES,
        source_directories,
        schema_version=CUSTODY_SOURCE_SCHEMA_VERSION,
        source_hardlinks_dereferenced=True,
    )
    return {
        "artifacts": artifacts,
        "excluded_module_roots": [
            "bench.multimodal_causal_diagnostics.predictor",
            "bench.multimodal_causal_diagnostics.worker",
            "bench.multimodal_mechanism_diagnostics",
        ],
        "schema_version": CUSTODY_SCHEMA_VERSION,
        "source_manifest_sha256": normalized_source["manifest_sha256"],
        "source_hashes": source_records,
    }


def run_worker_process(
    runtime_root: str | Path,
    source_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
    *,
    dependency_root: str | Path,
    stdlib_root: str | Path,
    timeout_seconds: int = 900,
    process_registry: WorkerProcessRegistry | None = None,
) -> None:
    """Launch one fresh isolated worker and require a silent successful exit."""

    runtime = _regular_directory(runtime_root, "shadow runtime")
    runtime_manifest(runtime)
    dependency = _prepared_dependency_root(dependency_root)
    stdlib = _prepared_stdlib_root(stdlib_root)
    source = Path(source_path)
    config = Path(config_path)
    output = _regular_directory(output_directory, "worker output", empty=True)
    for path, label in ((source, "source row"), (config, "worker config")):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise RuntimeValidationError(f"{label} must be an absolute regular file")
    if (
        not output.is_absolute()
        or not runtime.is_absolute()
        or not dependency.is_absolute()
        or not stdlib.is_absolute()
    ):
        raise RuntimeValidationError("runtime/dependency/stdlib/output paths must be absolute")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise RuntimeValidationError("worker timeout must be an integer in [1,3600]")
    if process_registry is not None and not isinstance(process_registry, WorkerProcessRegistry):
        raise RuntimeValidationError("worker process registry differs")
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(runtime / LAUNCHER_RELATIVE),
        str(source),
        str(config),
        str(output),
        str(runtime),
        str(dependency),
        str(stdlib),
    )
    registry = WorkerProcessRegistry() if process_registry is None else process_registry
    returncode, stdout_bytes, stdout_tail, stderr_bytes, stderr_tail = _run_registered_process(
        command,
        cwd=output,
        timeout_seconds=timeout_seconds,
        process_registry=registry,
    )
    if returncode != 0:
        detail = stderr_tail if stderr_tail else stdout_tail
        raise RuntimeValidationError(f"source-only worker failed with {returncode}: {detail}")
    if stdout_bytes or stderr_bytes:
        registry.cancel_all()
        raise RuntimeValidationError("source-only worker emitted an unexpected stream")


def run_isolation_probe(
    runtime_root: str | Path,
    allowed_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
    denied_paths: tuple[str | Path, ...],
    *,
    dependency_root: str | Path,
    stdlib_root: str | Path,
    timeout_seconds: int = 60,
) -> dict[str, object]:
    """Run a fresh sandbox known-answer test against explicitly denied paths."""

    runtime = _regular_directory(runtime_root, "shadow runtime")
    runtime_manifest(runtime)
    dependency = _prepared_dependency_root(dependency_root)
    stdlib = _prepared_stdlib_root(stdlib_root)
    allowed = Path(allowed_path)
    config = Path(config_path)
    output = _regular_directory(output_directory, "probe output", empty=True)
    requested_denied = tuple(Path(path) for path in denied_paths)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
        raise RuntimeValidationError("probe timeout must be an integer in [1,60]")
    if not requested_denied:
        raise RuntimeValidationError("isolation probe requires at least one denied path")
    mandatory_denied = tuple(path for path in live_python_roots() if path not in requested_denied)
    denied = (*requested_denied, *mandatory_denied)
    for path in (allowed, config, *denied):
        if not path.is_absolute() or path.is_symlink() or not path.exists():
            raise RuntimeValidationError("probe paths must be absolute existing non-symlinks")
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(runtime / PROBE_RELATIVE),
        str(allowed),
        str(config),
        str(output),
        str(runtime),
        str(dependency),
        str(stdlib),
        str(len(mandatory_denied)),
        *(str(path) for path in denied),
    )
    registry = WorkerProcessRegistry()
    returncode, stdout_bytes, stdout_tail, stderr_bytes, stderr_tail = _run_registered_process(
        command,
        cwd=output,
        timeout_seconds=timeout_seconds,
        process_registry=registry,
    )
    if returncode != 0:
        detail = stderr_tail if stderr_tail else stdout_tail
        raise RuntimeValidationError(f"isolation probe failed with {returncode}: {detail}")
    if stdout_bytes or stderr_bytes:
        registry.cancel_all()
        raise RuntimeValidationError("isolation probe emitted an unexpected stream")
    probe_path = output / "isolation-probe.json"
    if probe_path.is_symlink() or not probe_path.is_file():
        raise RuntimeValidationError("isolation probe did not emit its record")
    import json

    try:
        probe_payload = records.read_regular_bytes(probe_path, maximum_bytes=4096)
        value = json.loads(probe_payload.decode("ascii"))
    except (records.RecordValidationError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeValidationError("isolation probe record is not stable ASCII JSON") from error
    if (
        type(value) is not dict
        or value.get("schema_version") != "mm009-isolation-probe-v3"
        or value.get("denied_path_count") != len(denied)
        or value.get("landlock_abi") != 6
        or value.get("live_python_roots_denied") != len(mandatory_denied)
        or value.get("network_families_denied") != 3
        or value.get("network_socket_variants_denied") != 6
        or value.get("process_vm_readv_denied") is not True
        or value.get("requested_denied_path_count") != len(requested_denied)
    ):
        raise RuntimeValidationError("isolation probe record differs")
    return value


__all__ = [
    "CUSTODY_RUNTIME_FILES",
    "CUSTODY_SCHEMA_VERSION",
    "CUSTODY_SOURCE_SCHEMA_VERSION",
    "DEPENDENCY_PACKAGES",
    "DEPENDENCY_SCHEMA_VERSION",
    "DEPENDENCY_SOURCE_SCHEMA_VERSION",
    "FUTURE_ISOLATION_LAUNCHER_RELATIVE",
    "HOST_TRUST_SCHEMA_VERSION",
    "LAUNCHER_RELATIVE",
    "PROBE_RELATIVE",
    "REPO_ROOT",
    "RUNTIME_FILES",
    "RuntimeValidationError",
    "SCORE_LAUNCHER_RELATIVE",
    "SCHEMA_VERSION",
    "SITE_PACKAGES",
    "STDLIB_EXCLUDED_DIRECTORY_NAMES",
    "STDLIB_SCHEMA_VERSION",
    "STDLIB_SOURCE",
    "STDLIB_SOURCE_SCHEMA_VERSION",
    "WorkerProcessRegistry",
    "build_custody_runtime",
    "build_numpy_dependency_closure",
    "build_shadow_runtime",
    "build_stdlib_closure",
    "custody_runtime_manifest",
    "custody_runtime_source_manifest",
    "custody_source_hashes",
    "host_runtime_trust_manifest",
    "live_python_roots",
    "numpy_dependency_manifest",
    "numpy_dependency_source_manifest",
    "run_isolation_probe",
    "run_worker_process",
    "runtime_manifest",
    "source_hashes",
    "stdlib_manifest",
    "stdlib_source_manifest",
]
