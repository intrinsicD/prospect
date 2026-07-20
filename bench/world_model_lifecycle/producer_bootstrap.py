"""Standard-library custody bootstrap for every WM-001 runtime process.

This file is executed with ``-I -S -B``.  Before any package-root import it
recomputes the sealed interpreter, standard-library, package-root, environment,
and bootstrap identities.  Only then does it insert the one authorized root
and dispatch to an installed Prospect entry point.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = "prospect.wm001.runtime-seal.v1"
_PROTOCOL_VERSION = "1.7.0"
_PACKAGE_DOMAIN = b"prospect.wm001.package-root.v2\0"
_STDLIB_DOMAIN = b"prospect.wm001.standard-library.v2\0"
_ENVIRONMENT_KEYS = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "LAZY_LEGACY_OP",
        "LC_ALL",
        "MKL_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYGAME_HIDE_SUPPORT_PROMPT",
        "ROCR_VISIBLE_DEVICES",
        "SDL_AUDIODRIVER",
        "TZ",
    }
)
_EXPECTED_FLAGS = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "safe_path": True,
}
_OUTER_RECEIPT_SCHEMA = "prospect.wm001.outer-terminal-receipt.v1"
_OUTER_TRUST_MODEL = "trusted-single-principal-cooperative-lock-v1"
_ASSURANCE: dict[str, object] = {
    "trust_model_id": "prospect.wm001.trust-model.v1",
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}


class BootstrapError(RuntimeError):
    """The process differs from its pre-import runtime seal."""


def _canonical_value_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_bytes(value: object) -> bytes:
    return _canonical_value_bytes(value) + b"\n"


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BootstrapError(f"runtime seal contains duplicate key {key!r}")
        value[key] = item
    return value


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


def _read_descriptor(
    descriptor: int,
    *,
    limit: int,
    label: str,
    expected_nlink: int = 1,
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != expected_nlink
        ):
            raise BootstrapError(
                f"{label} must be a regular file with exactly "
                f"{expected_nlink} link(s)"
            )
        if before.st_size > limit:
            raise BootstrapError(f"{label} exceeds its byte limit")
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError as error:
        raise BootstrapError(f"{label} descriptor cannot be read") from error
    if len(payload) != before.st_size or _stat_identity(before) != _stat_identity(after):
        raise BootstrapError(f"{label} changed while it was read")
    return payload, before


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise BootstrapError(f"{label} path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise BootstrapError(f"{label} path cannot be resolved") from error
        if stat.S_ISLNK(mode):
            raise BootstrapError(f"{label} path contains a symbolic-link component")


def _open_regular(path: Path, *, label: str) -> int:
    _reject_symlink_components(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(f"{label} cannot be opened") from error
    try:
        _read_descriptor(descriptor, limit=64 << 20, label=label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _load_canonical_payload(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                BootstrapError(f"runtime seal contains non-finite value {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("runtime seal is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or payload != _canonical_bytes(value):
        raise BootstrapError("runtime seal is not one canonical JSON object followed by LF")
    return value


def _descriptor_path(descriptor: int) -> str:
    for prefix in ("/proc/self/fd/", "/dev/fd/"):
        candidate = f"{prefix}{descriptor}"
        if os.path.exists(candidate):
            return candidate
    raise BootstrapError("platform has no inherited-descriptor execution path")


def _capture_bootstrap() -> tuple[int, bytes]:
    if sys.argv[1:2] != ["--bootstrap-fd"] or len(sys.argv) < 3:
        raise BootstrapError(
            "producer bootstrap must be executed from an inherited descriptor"
        )
    try:
        descriptor = int(sys.argv[2])
    except ValueError as error:
        raise BootstrapError("--bootstrap-fd value is invalid") from error
    execution_path = _descriptor_path(descriptor)
    try:
        same_execution_file = os.path.samefile(sys.argv[0], execution_path)
    except OSError as error:
        raise BootstrapError(
            "producer bootstrap execution path cannot be verified"
        ) from error
    if sys.argv[0] != execution_path or not same_execution_file:
        raise BootstrapError(
            "interpreter did not execute the supplied producer-bootstrap descriptor"
        )
    del sys.argv[1:3]
    payload, _ = _read_descriptor(
        descriptor,
        limit=64 << 20,
        label="producer bootstrap",
    )
    return descriptor, payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flags() -> dict[str, object]:
    return {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    }


def _sanitize_module_search_path() -> tuple[str, ...]:
    """Drop absent entries and reject every extant ambient import root."""

    stdlib_root = Path(sysconfig.get_path("stdlib"))
    _reject_symlink_components(stdlib_root, label="standard-library root")
    if (
        not stdlib_root.is_dir()
        or stdlib_root.resolve(strict=True) != stdlib_root
    ):
        raise BootstrapError(
            "standard-library root must be one canonical directory"
        )
    authorized: set[Path] = set()
    for raw in sys.path:
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise BootstrapError(
                "module search path contains a non-absolute entry"
            )
        candidate = Path(raw)
        if not os.path.lexists(candidate):
            continue
        _reject_symlink_components(
            candidate,
            label="module search path entry",
        )
        if (
            not candidate.is_dir()
            or candidate.resolve(strict=True) != candidate
        ):
            raise BootstrapError(
                "module search path contains a non-directory entry"
            )
        try:
            relative = candidate.relative_to(stdlib_root)
        except ValueError as error:
            raise BootstrapError(
                "module search path contains an ambient import root"
            ) from error
        if {"site-packages", "dist-packages"} & set(relative.parts):
            raise BootstrapError(
                "module search path contains an undeclared package root"
            )
        if candidate in authorized:
            raise BootstrapError(
                "module search path contains a duplicate entry"
            )
        authorized.add(candidate)
    if stdlib_root not in authorized:
        raise BootstrapError(
            "module search path omits the standard-library root"
        )
    ordered = (
        str(stdlib_root),
        *(
            str(path)
            for path in sorted(authorized)
            if path != stdlib_root
        ),
    )
    sys.path[:] = ordered
    return ordered


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    if (
        set(environment) - _ENVIRONMENT_KEYS
        or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or environment.get("LAZY_LEGACY_OP") != "False"
        or environment.get("LC_ALL") != "C.UTF-8"
        or environment.get("PATH") != "/usr/bin:/bin"
        or environment.get("PYGAME_HIDE_SUPPORT_PROMPT") != "hide"
        or environment.get("SDL_AUDIODRIVER") != "dsp"
        or environment.get("TZ") != "UTC"
        or any("\0" in key or "\0" in value for key, value in environment.items())
    ):
        raise BootstrapError(
            "WM-001 requires env -i with the exact safe runtime environment"
        )
    return dict(sorted(environment.items()))


def _virtualenv_package_root() -> Path:
    executable = Path(sys.executable)
    if not executable.is_absolute():
        raise BootstrapError("WM-001 executable is not absolute")
    virtualenv = executable.parent.parent
    configuration = virtualenv / "pyvenv.cfg"
    if not configuration.is_file() or configuration.is_symlink():
        raise BootstrapError("WM-001 requires a dedicated virtual environment")
    settings = {
        key.strip().lower(): value.strip().lower()
        for line in configuration.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    if settings.get("include-system-site-packages") != "false":
        raise BootstrapError("WM-001 virtualenv inherits system site-packages")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = {
        virtualenv / "lib" / version / "site-packages",
        virtualenv / "lib64" / version / "site-packages",
    }
    roots = [
        candidate
        for candidate in sorted(candidates)
        if candidate.is_dir()
        and not candidate.is_symlink()
        and candidate.resolve(strict=True) == candidate
    ]
    if len(roots) != 1:
        raise BootstrapError("WM-001 requires exactly one canonical package root")
    return roots[0]


def _inventory(
    root: Path,
    *,
    domain: bytes,
    standard_library: bool,
) -> dict[str, object]:
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise BootstrapError(f"runtime inventory root is absent or aliased: {root}")
    def identity(metadata: os.stat_result) -> tuple[int, ...]:
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

    def discover() -> tuple[
        list[Path],
        list[Path],
        tuple[tuple[str, str, tuple[int, ...]], ...],
    ]:
        files: list[Path] = []
        directories: list[Path] = []
        namespace: list[tuple[str, str, tuple[int, ...]]] = []
        for directory, directory_names, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            directory_names.sort()
            retained_directories: list[str] = []
            for name in directory_names:
                directory_path = Path(directory) / name
                relative = directory_path.relative_to(root).as_posix()
                try:
                    metadata = directory_path.lstat()
                except OSError as error:
                    raise BootstrapError(
                        "runtime inventory directory cannot be inspected: "
                        f"{directory_path}"
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise BootstrapError(
                        f"runtime inventory contains symlink: {directory_path}"
                    )
                if not stat.S_ISDIR(metadata.st_mode):
                    raise BootstrapError(
                        "runtime inventory contains special directory: "
                        f"{directory_path}"
                    )
                if not (standard_library and name == "site-packages"):
                    retained_directories.append(name)
                    directories.append(directory_path)
                    namespace.append(
                        ("directory", relative, identity(metadata))
                    )
            directory_names[:] = retained_directories
            for filename in filenames:
                path = Path(directory) / filename
                relative = path.relative_to(root).as_posix()
                try:
                    metadata = path.lstat()
                except OSError as error:
                    raise BootstrapError(
                        f"runtime inventory file cannot be inspected: {relative}"
                    ) from error
                if stat.S_ISLNK(metadata.st_mode):
                    raise BootstrapError(
                        f"runtime inventory contains symlink: {relative}"
                    )
                if not stat.S_ISREG(metadata.st_mode):
                    raise BootstrapError(
                        f"runtime inventory contains special file: {relative}"
                    )
                files.append(path)
                namespace.append(("file", relative, identity(metadata)))
        return (
            files,
            directories,
            tuple(sorted(namespace, key=lambda row: (row[1], row[0]))),
        )

    digest = hashlib.sha256(domain)
    file_count = 0
    directory_count = 0
    total_bytes = 0
    inventory_files, inventory_directories, initial_namespace = discover()
    entries = [
        *(("directory", path) for path in inventory_directories),
        *(("file", path) for path in inventory_files),
    ]
    for kind, path in sorted(
        entries,
        key=lambda item: item[1].relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if kind == "directory":
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0directory\0")
            directory_count += 1
            continue
        before = path.lstat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(before.st_size.to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        after = path.lstat()
        if identity(before) != identity(after):
            raise BootstrapError(
                f"runtime inventory entry changed while read: {relative}"
            )
        digest.update(b"\0")
        file_count += 1
        total_bytes += before.st_size
    if file_count == 0:
        raise BootstrapError("runtime inventory root is empty")
    _, _, final_namespace = discover()
    if initial_namespace != final_namespace:
        raise BootstrapError(
            "runtime inventory namespace changed while it was captured"
        )
    return {
        "semantics_id": (
            "prospect.wm001.standard-library.v2"
            if standard_library
            else "prospect.wm001.package-root.v2"
        ),
        "path": str(root),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def _record_hash_identity(value: object) -> tuple[str, str] | None:
    """Read ``importlib.metadata.FileHash`` through its supported fields."""

    if value is None:
        return None
    mode = getattr(value, "mode", None)
    encoded = getattr(value, "value", None)
    if not isinstance(mode, str) or not mode or not isinstance(encoded, str) or not encoded:
        raise BootstrapError("installed distribution has malformed RECORD hash")
    return mode, encoded


def _record_sha256_hex(identity: tuple[str, str]) -> str:
    algorithm, encoded = identity
    if algorithm != "sha256":
        raise BootstrapError("shared package file has non-SHA256 RECORD")
    try:
        decoded = base64.b64decode(
            encoded.encode("ascii") + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise BootstrapError("shared package file has malformed RECORD hash") from error
    if len(decoded) != hashlib.sha256().digest_size:
        raise BootstrapError("shared package file has malformed SHA256 RECORD")
    return decoded.hex()


def _package_ownership(root: Path) -> dict[str, object]:
    owners: dict[str, list[tuple[str, tuple[str, str] | None]]] = {}
    for distribution in importlib.metadata.distributions(path=[str(root)]):
        raw_name = distribution.metadata["Name"]
        if not isinstance(raw_name, str) or not raw_name:
            raise BootstrapError("installed distribution has no Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        declared = tuple(distribution.files or ())
        if not declared:
            raise BootstrapError(f"installed distribution has no RECORD files: {name}")
        for entry in declared:
            located = Path(
                os.path.abspath(str(distribution.locate_file(entry)))
            )
            if not located.is_relative_to(root):
                continue
            relative = located.relative_to(root).as_posix()
            owners.setdefault(relative, []).append(
                (name, _record_hash_identity(entry.hash))
            )
    actual_files: dict[str, Path] = {}
    actual_directories: set[str] = set()
    for directory, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        directory_names.sort()
        filenames.sort()
        for name in directory_names:
            path = current / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise BootstrapError(
                    f"package ownership found non-directory: {relative}"
                )
            if name == "__pycache__":
                raise BootstrapError(
                    f"runtime package root contains bytecode cache: {relative}"
                )
            actual_directories.add(relative)
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise BootstrapError(
                    f"package ownership found non-regular file: {relative}"
                )
            if path.suffix == ".pyc":
                raise BootstrapError(
                    f"runtime package root contains bytecode: {relative}"
                )
            actual_files[relative] = path
    if set(owners) != set(actual_files):
        raise BootstrapError("package-root RECORD ownership is not exact")
    implied_directories = {
        parent.as_posix()
        for relative in actual_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_directories != implied_directories:
        raise BootstrapError("package-root directory ownership is not exact")
    rows: list[dict[str, object]] = []
    shared_file_count = 0
    for relative in sorted(actual_files):
        file_owners = sorted(owners[relative])
        owner_names = [name for name, _ in file_owners]
        if len(owner_names) != len(set(owner_names)):
            raise BootstrapError(
                f"package file has duplicate RECORD owner: {relative}"
            )
        if len(file_owners) > 1:
            hashes = {value for _, value in file_owners}
            if None in hashes or len(hashes) != 1:
                raise BootstrapError(
                    f"package file has conflicting owners: {relative}"
                )
            selected = next(iter(hashes))
            assert selected is not None
            expected = _record_sha256_hex(selected)
            if _sha256_file(actual_files[relative]) != expected:
                raise BootstrapError(
                    f"shared package file differs from RECORD: {relative}"
                )
            shared_file_count += 1
        rows.append(
            {
                "path": relative,
                "owners": owner_names,
            }
        )
    ownership_identity = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "files": rows,
        "directories": sorted(actual_directories),
    }
    return {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "file_count": len(actual_files),
        "directory_count": len(actual_directories),
        "shared_file_count": shared_file_count,
        "identity_sha256": hashlib.sha256(
            _canonical_value_bytes(ownership_identity)
        ).hexdigest(),
    }


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=Path.cwd(),
        env=_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise BootstrapError("runtime seal Git identity command failed")
    return completed.stdout.strip()


def _current_runtime_seal(*, bootstrap_sha256: str) -> dict[str, object]:
    if _flags() != _EXPECTED_FLAGS:
        raise BootstrapError("WM-001 bootstrap requires exact CPython flags -I -S -B")
    package_root = _virtualenv_package_root()
    stdlib_root = Path(sysconfig.get_path("stdlib"))
    return {
        "schema": _SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": _PROTOCOL_VERSION,
        "assurance": dict(_ASSURANCE),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "worktree_clean": not _git_value(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "python": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve(strict=True)),
            "sha256": _sha256_file(Path(sys.executable).resolve(strict=True)),
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
        },
        "required_flags": _EXPECTED_FLAGS,
        "process_environment": _environment(),
        "bootstrap_source_sha256": bootstrap_sha256,
        "standard_library": _inventory(
            stdlib_root,
            domain=_STDLIB_DOMAIN,
            standard_library=True,
        ),
        "package_roots": [
            _inventory(
                package_root,
                domain=_PACKAGE_DOMAIN,
                standard_library=False,
            )
        ],
        "package_ownership": _package_ownership(package_root),
    }


def _atomic_write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_from_binding(binding: dict[str, Any]) -> dict[str, object]:
    source = binding.get("source")
    dependencies = binding.get("dependencies")
    runtime = binding.get("runtime")
    if (
        binding.get("schema") != "prospect.world-model-lifecycle.formal-binding.v7"
        or binding.get("experiment_id") != "WM-001"
        or not isinstance(source, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(runtime, dict)
    ):
        raise BootstrapError("formal binding cannot serve as a runtime seal")
    execution_sources = source.get("execution_source_sha256")
    if not isinstance(execution_sources, dict):
        raise BootstrapError("formal binding has no execution-source identities")
    return {
        "schema": _SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": _PROTOCOL_VERSION,
        "assurance": binding.get("assurance"),
        "git_commit": source.get("git_commit"),
        "git_tree": source.get("git_tree"),
        "worktree_clean": source.get("worktree_clean"),
        "python": {
            "executable": dependencies.get("python_executable"),
            "resolved_executable": str(Path(sys.executable).resolve(strict=True)),
            "sha256": dependencies.get("python_executable_sha256"),
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
        },
        "required_flags": runtime.get("python_flags"),
        "process_environment": runtime.get("process_environment"),
        "bootstrap_source_sha256": execution_sources.get("producer_bootstrap.py"),
        "standard_library": dependencies.get("standard_library"),
        "package_roots": dependencies.get("package_roots"),
        "package_ownership": dependencies.get("package_ownership"),
    }


def _verify_runtime_seal(
    supplied: dict[str, Any],
    *,
    bootstrap_sha256: str,
) -> Path:
    expected = (
        _expected_from_binding(supplied)
        if supplied.get("schema")
        == "prospect.world-model-lifecycle.formal-binding.v7"
        else supplied
    )
    current = _current_runtime_seal(bootstrap_sha256=bootstrap_sha256)
    if expected != current:
        raise BootstrapError("live pre-import runtime differs from its seal")
    package_roots = current.get("package_roots")
    if (
        not isinstance(package_roots, list)
        or len(package_roots) != 1
        or not isinstance(package_roots[0], dict)
        or not isinstance(package_roots[0].get("path"), str)
    ):
        raise BootstrapError("live runtime seal has no single package root")
    return Path(package_roots[0]["path"])


def _verify_captured_descriptor(
    descriptor: int,
    *,
    expected_payload: bytes,
    expected_identity: tuple[int, ...],
    label: str,
    expected_nlink: int = 1,
) -> None:
    payload, metadata = _read_descriptor(
        descriptor,
        limit=64 << 20,
        label=label,
        expected_nlink=expected_nlink,
    )
    if payload != expected_payload or _stat_identity(metadata) != expected_identity:
        raise BootstrapError(f"{label} changed after pre-import verification")


def _runtime_custody_nlink(value: dict[str, Any]) -> int:
    schema = value.get("schema")
    if (
        value.get("experiment_id") != "WM-001"
        or value.get("assurance") != _ASSURANCE
        or schema
        not in {
            _SCHEMA,
            "prospect.world-model-lifecycle.formal-binding.v7",
        }
    ):
        raise BootstrapError(
            "runtime custody schema or assurance is invalid"
        )
    return 2 if schema == _SCHEMA else 1


def _select_runtime_seal() -> tuple[int, bytes, dict[str, Any], int]:
    if sys.argv[1:2] == ["--runtime-seal-fd"]:
        if len(sys.argv) < 3:
            raise BootstrapError("--runtime-seal-fd requires one descriptor")
        try:
            descriptor = int(sys.argv[2])
        except ValueError as error:
            raise BootstrapError("--runtime-seal-fd value is invalid") from error
        del sys.argv[1:3]
    else:
        raise BootstrapError(
            "WM-001 runtime entry requires one inherited runtime-seal descriptor"
        )
    try:
        observed_nlink = os.fstat(descriptor).st_nlink
    except OSError as error:
        raise BootstrapError(
            "runtime seal descriptor cannot be inspected"
        ) from error
    if observed_nlink not in {1, 2}:
        raise BootstrapError(
            "runtime seal descriptor has an invalid link count"
        )
    payload, _ = _read_descriptor(
        descriptor,
        limit=64 << 20,
        label="runtime seal",
        expected_nlink=observed_nlink,
    )
    value = _load_canonical_payload(payload)
    expected_nlink = _runtime_custody_nlink(value)
    if observed_nlink != expected_nlink:
        raise BootstrapError(
            "runtime custody link count differs from its schema"
        )
    return descriptor, payload, value, expected_nlink


def _select_outer_receipt_descriptor() -> int | None:
    if sys.argv[1:2] != ["--outer-receipt-fd"]:
        return None
    if len(sys.argv) < 3:
        raise BootstrapError("--outer-receipt-fd requires one descriptor")
    try:
        descriptor = int(sys.argv[2])
        metadata = os.fstat(descriptor)
    except (ValueError, OSError) as error:
        raise BootstrapError("--outer-receipt-fd is invalid") from error
    if not stat.S_ISFIFO(metadata.st_mode):
        raise BootstrapError("--outer-receipt-fd must be one inherited pipe")
    del sys.argv[1:3]
    return descriptor


def _reject_second_custody_source() -> None:
    forbidden = {
        "--bootstrap",
        "--bootstrap-fd",
        "--runtime-seal",
        "--runtime-seal-fd",
        "--formal-binding-seal",
        "--create-runtime-seal",
        "--outer-receipt-fd",
    }
    if any(
        argument == name or argument.startswith(f"{name}=")
        for argument in sys.argv[1:]
        for name in forbidden
    ):
        raise BootstrapError("runtime arguments contain a second custody source")


def register_outer_terminal(
    path: Path,
    *,
    logical_exit_code: int,
) -> None:
    """Register one final terminal for the outer launcher's logical commit."""

    registrar = getattr(
        sys,
        "_prospect_wm001_register_outer_terminal",
        None,
    )
    if not callable(registrar):
        raise BootstrapError(
            "terminal publication requires the authoritative outer launcher"
        )
    registrar(path, logical_exit_code=logical_exit_code)


def _capture_outer_terminal(
    path: Path,
    *,
    logical_exit_code: int,
) -> dict[str, object]:
    candidate = path
    results_root = (
        Path.cwd()
        / "bench"
        / "world_model_lifecycle"
        / "results"
    )
    if (
        type(logical_exit_code) is not int
        or logical_exit_code not in {0, 1, 2}
        or not candidate.is_absolute()
        or Path(os.path.abspath(candidate)) != candidate
        or candidate.resolve(strict=False) != candidate
        or not candidate.is_relative_to(results_root)
        or any(part.startswith(".") for part in candidate.relative_to(results_root).parts)
    ):
        raise BootstrapError("outer terminal registration is not canonical")
    _reject_symlink_components(candidate, label="outer terminal")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        payload, metadata = _read_descriptor(
            descriptor,
            limit=64 << 20,
            label="outer terminal",
        )
        if metadata.st_nlink != 1:
            raise BootstrapError(
                "outer terminal must be uncommitted with exactly one link"
            )
        return {
            "path": candidate,
            "descriptor": descriptor,
            "payload": payload,
            "identity": _stat_identity(metadata),
            "logical_exit_code": logical_exit_code,
        }
    except BaseException:
        os.close(descriptor)
        raise


def _recheck_outer_terminal(registration: dict[str, object]) -> None:
    path = registration["path"]
    descriptor = registration["descriptor"]
    payload = registration["payload"]
    identity = registration["identity"]
    assert isinstance(path, Path)
    assert isinstance(descriptor, int)
    assert isinstance(payload, bytes)
    assert isinstance(identity, tuple)
    _verify_captured_descriptor(
        descriptor,
        expected_payload=payload,
        expected_identity=identity,
        label="outer terminal",
    )
    try:
        same_file = os.path.samefile(path, _descriptor_path(descriptor))
    except OSError as error:
        raise BootstrapError("outer terminal path cannot be rechecked") from error
    if not same_file:
        raise BootstrapError("outer terminal path changed before receipt")


def _emit_outer_receipt(
    descriptor: int,
    registration: dict[str, object],
) -> None:
    _recheck_outer_terminal(registration)
    path = registration["path"]
    payload = registration["payload"]
    logical_exit_code = registration["logical_exit_code"]
    assert isinstance(path, Path)
    assert isinstance(payload, bytes)
    assert isinstance(logical_exit_code, int)
    receipt = {
        "schema": _OUTER_RECEIPT_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": _PROTOCOL_VERSION,
        "assurance": dict(_ASSURANCE),
        "trust_model": _OUTER_TRUST_MODEL,
        "terminal_path": str(path),
        "terminal_bytes": len(payload),
        "terminal_sha256": hashlib.sha256(payload).hexdigest(),
        "logical_exit_code": logical_exit_code,
    }
    receipt_payload = _canonical_bytes(receipt)
    try:
        written = os.write(descriptor, receipt_payload)
    except OSError as error:
        raise BootstrapError("outer terminal receipt could not be emitted") from error
    if written != len(receipt_payload):
        raise BootstrapError("outer terminal receipt was truncated")


def main() -> int:
    if _flags() != _EXPECTED_FLAGS:
        raise BootstrapError(
            "WM-001 bootstrap requires exact CPython flags -I -S -B"
        )
    _sanitize_module_search_path()
    bootstrap_descriptor, bootstrap_payload = _capture_bootstrap()
    bootstrap_sha256 = hashlib.sha256(bootstrap_payload).hexdigest()
    bootstrap_identity = _stat_identity(os.fstat(bootstrap_descriptor))
    outer_receipt_descriptor = _select_outer_receipt_descriptor()
    runtime_seal_descriptor: int | None = None
    runtime_seal_payload: bytes | None = None
    runtime_seal_identity: tuple[int, ...] | None = None
    runtime_seal: dict[str, Any] | None = None
    runtime_seal_expected_nlink: int | None = None
    dispatch_result: int | None = None
    registration: dict[str, object] | None = None
    read_only_entry = False

    def registrar(path: Path, *, logical_exit_code: int) -> None:
        nonlocal registration
        if registration is not None:
            raise BootstrapError("outer terminal was registered more than once")
        registration = _capture_outer_terminal(
            path,
            logical_exit_code=logical_exit_code,
        )

    sys._prospect_wm001_register_outer_terminal = registrar  # type: ignore[attr-defined]
    try:
        if sys.argv[1:2] == ["--create-runtime-seal"]:
            if len(sys.argv) != 3:
                raise BootstrapError(
                    "--create-runtime-seal requires exactly one output path"
                )
            if outer_receipt_descriptor is None:
                raise BootstrapError(
                    "runtime-seal publication requires the outer receipt"
                )
            seal = _current_runtime_seal(bootstrap_sha256=bootstrap_sha256)
            if seal["worktree_clean"] is not True:
                raise BootstrapError(
                    "runtime seal creation requires a clean worktree"
                )
            _verify_captured_descriptor(
                bootstrap_descriptor,
                expected_payload=bootstrap_payload,
                expected_identity=bootstrap_identity,
                label="producer bootstrap",
            )
            output = Path(sys.argv[2])
            _atomic_write_exclusive(output, _canonical_bytes(seal))
            registration = _capture_outer_terminal(
                output,
                logical_exit_code=0,
            )
            dispatch_result = 0
        else:
            (
                runtime_seal_descriptor,
                runtime_seal_payload,
                runtime_seal,
                runtime_seal_expected_nlink,
            ) = _select_runtime_seal()
            runtime_seal_identity = _stat_identity(
                os.fstat(runtime_seal_descriptor)
            )
            _reject_second_custody_source()
            package_root = _verify_runtime_seal(
                runtime_seal,
                bootstrap_sha256=bootstrap_sha256,
            )
            if any(
                "site-packages" in entry or "dist-packages" in entry
                for entry in sys.path
            ):
                raise BootstrapError(
                    "WM-001 inherited an undeclared package root"
                )
            sys.path.insert(0, str(package_root))
            sys._prospect_wm001_runtime_seal_fd = runtime_seal_descriptor  # type: ignore[attr-defined]
            sys._prospect_wm001_runtime_seal_sha256 = (  # type: ignore[attr-defined]
                hashlib.sha256(runtime_seal_payload).hexdigest()
            )
            sys._prospect_wm001_runtime_seal_payload = runtime_seal_payload  # type: ignore[attr-defined]
            sys._prospect_wm001_runtime_seal_identity = runtime_seal_identity  # type: ignore[attr-defined]
            sys._prospect_wm001_bootstrap_fd = bootstrap_descriptor  # type: ignore[attr-defined]
            sys._prospect_wm001_bootstrap_sha256 = bootstrap_sha256  # type: ignore[attr-defined]
            sys._prospect_wm001_bootstrap_payload = bootstrap_payload  # type: ignore[attr-defined]
            sys._prospect_wm001_bootstrap_identity = bootstrap_identity  # type: ignore[attr-defined]

            entry = sys.argv[1] if len(sys.argv) > 1 else None
            read_only_entry = entry in {
                "--restore-eval-entry",
                "preformal-runtime",
            }
            if not read_only_entry and outer_receipt_descriptor is None:
                raise BootstrapError(
                    "top-level publication requires the outer receipt"
                )
            if sys.argv[1:2] == ["--restore-eval-entry"]:
                del sys.argv[1]
                from bench.world_model_lifecycle.restore_eval import (
                    main as restore_main,
                )

                dispatch_result = int(restore_main())
            elif sys.argv[1:2] == ["--binding-entry"]:
                del sys.argv[1]
                from bench.world_model_lifecycle.operator import binding_main

                dispatch_result = int(binding_main())
            elif sys.argv[1:2] == ["--audit-entry"]:
                del sys.argv[1]
                from bench.world_model_lifecycle.operator import audit_main

                dispatch_result = int(audit_main())
            elif sys.argv[1:2] == ["--closure-entry"]:
                del sys.argv[1]
                from bench.world_model_lifecycle.operator import closure_main

                dispatch_result = int(closure_main())
            elif sys.argv[1:2] == ["--adjudication-entry"]:
                del sys.argv[1]
                adjudication_main = importlib.import_module(
                    "bench.world_model_lifecycle.adjudication"
                ).main
                dispatch_result = int(adjudication_main())
            elif sys.argv[1:2] == ["preformal-runtime"]:
                del sys.argv[1]
                from bench.world_model_lifecycle.preformal import runtime_main

                dispatch_result = int(runtime_main())
            else:
                from bench.world_model_lifecycle.run import main as run_main

                dispatch_result = int(run_main())
    finally:
        _verify_captured_descriptor(
            bootstrap_descriptor,
            expected_payload=bootstrap_payload,
            expected_identity=bootstrap_identity,
            label="producer bootstrap",
        )
        if (
            runtime_seal_descriptor is not None
            and runtime_seal_payload is not None
            and runtime_seal_identity is not None
            and runtime_seal is not None
            and runtime_seal_expected_nlink is not None
        ):
            _verify_captured_descriptor(
                runtime_seal_descriptor,
                expected_payload=runtime_seal_payload,
                expected_identity=runtime_seal_identity,
                label="runtime seal",
                expected_nlink=runtime_seal_expected_nlink,
            )
            _verify_runtime_seal(
                runtime_seal,
                bootstrap_sha256=bootstrap_sha256,
            )
    assert dispatch_result is not None
    if registration is not None:
        if registration.get("logical_exit_code") != dispatch_result:
            raise BootstrapError(
                "outer terminal exit status differs from entry result"
            )
        if outer_receipt_descriptor is None:
            raise BootstrapError("outer terminal has no receipt descriptor")
        try:
            _emit_outer_receipt(
                outer_receipt_descriptor,
                registration,
            )
        finally:
            terminal_descriptor = registration["descriptor"]
            assert isinstance(terminal_descriptor, int)
            os.close(terminal_descriptor)
            os.close(outer_receipt_descriptor)
        return 0
    if outer_receipt_descriptor is not None:
        os.close(outer_receipt_descriptor)
    if not read_only_entry:
        raise BootstrapError("top-level publisher emitted no terminal receipt")
    return dispatch_result


if __name__ == "__main__":
    raise SystemExit(main())
