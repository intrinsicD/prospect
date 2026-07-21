#!/usr/bin/env python3
"""Standard-library outer launcher for the descriptor-executed WM-001 producer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path

_MAX_CONTROL_BYTES = 64 << 20
_MAX_RECEIPT_BYTES = 4096
_OUTER_RECEIPT_SCHEMA = "prospect.wm001.outer-terminal-receipt.v1"
_OUTER_TRUST_MODEL = "trusted-single-principal-cooperative-lock-v1"
_RUNTIME_SEAL_SCHEMA = "prospect.wm001.runtime-seal.v1"
_FORMAL_BINDING_SCHEMA = "prospect.world-model-lifecycle.formal-binding.v10"
_OPERATOR_ATTEMPT_SCHEMA = "prospect.wm001.operator-attempt.v1"
_OPERATOR_TERMINAL = "operator-attempt.json"
_CLOSURE_REFERENCE_SCHEMA = "prospect.wm001.closure-reference.v1"
_PREFORMAL_REPORT_NAME = "preformal-test-report-v1.15.0.json"
_PREFORMAL_LOG_PREFIX = "preformal-v1.15.0-command-"
_SHA256_EMPTY = hashlib.sha256(b"").hexdigest()
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_DEVELOPMENT_MATRIX_CONTRACT_SHA256 = (
    "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
)
_PREFORMAL_COMMAND_NAMES = (
    "protocol-seal-continuity",
    "ruff",
    "mypy-core",
    "mypy-wm001",
    "pytest-epistemic",
    "pytest-wm001",
    "audit-runner-adversarial",
    "prospective-harness-review",
    "runtime-accepted-closure-evidence",
    "runtime-bootstrap-inventory-conformance",
)
_FORMAL_INPUT_PREFLIGHT_NAME = "formal-input-preflight.json"
_FORMAL_INPUT_PREFLIGHT_SCHEMA = (
    "prospect.wm001.formal-input-preflight.v1"
)
_RUNTIME_SEAL_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "assurance",
    "git_commit",
    "git_tree",
    "worktree_clean",
    "python",
    "required_flags",
    "process_environment",
    "bootstrap_source_sha256",
    "standard_library",
    "package_roots",
    "package_ownership",
}
_ASSURANCE: dict[str, object] = {
    "trust_model_id": "prospect.wm001.trust-model.v1",
    "tamper_resistant": False,
    "external_attestation": False,
    "exclusive_path_use_required": True,
}
_SAFE_ENVIRONMENT_KEYS = frozenset(
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


class LaunchError(RuntimeError):
    """The producer could not be entered with stable descriptor custody."""


def _close_quietly(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


class _StoreOnce(argparse.Action):
    """Reject repeated custody-bearing options instead of accepting the last."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string or self.dest} may be supplied exactly once")
        setattr(namespace, self.dest, values)


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.resolve(strict=False) != path:
        raise LaunchError(f"{label} must be one canonical absolute path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise LaunchError(f"{label} cannot be resolved") from error
        if stat.S_ISLNK(mode):
            raise LaunchError(f"{label} contains a symbolic-link component")


def _sanitize_module_search_path() -> tuple[str, ...]:
    """Retain only canonical, inventoried standard-library directories."""

    stdlib_root = Path(sysconfig.get_path("stdlib"))
    _reject_symlink_components(stdlib_root, label="standard-library root")
    if (
        not stdlib_root.is_dir()
        or stdlib_root.resolve(strict=True) != stdlib_root
    ):
        raise LaunchError("standard-library root must be one canonical directory")
    authorized: set[Path] = set()
    for raw in sys.path:
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise LaunchError("module search path contains a non-absolute entry")
        candidate = Path(raw)
        if not os.path.lexists(candidate):
            continue
        _reject_symlink_components(candidate, label="module search path entry")
        if (
            not candidate.is_dir()
            or candidate.resolve(strict=True) != candidate
        ):
            raise LaunchError(
                "module search path contains a non-directory entry"
            )
        try:
            relative = candidate.relative_to(stdlib_root)
        except ValueError as error:
            raise LaunchError(
                "module search path contains an ambient import root"
            ) from error
        if {"site-packages", "dist-packages"} & set(relative.parts):
            raise LaunchError(
                "module search path contains an undeclared package root"
            )
        if candidate in authorized:
            raise LaunchError("module search path contains a duplicate entry")
        authorized.add(candidate)
    if stdlib_root not in authorized:
        raise LaunchError("module search path omits the standard-library root")
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


def _read_descriptor(
    descriptor: int,
    *,
    label: str,
    expected_nlink: int = 1,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != expected_nlink:
            raise LaunchError(f"{label} must be a regular file with exactly {expected_nlink} link(s)")
        if before.st_size > _MAX_CONTROL_BYTES:
            raise LaunchError(f"{label} exceeds its byte limit")
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError as error:
        raise LaunchError(f"{label} descriptor cannot be read") from error
    if len(payload) != before.st_size or _identity(before) != _identity(after):
        raise LaunchError(f"{label} changed while read")
    return payload, _identity(before)


def _open_regular(
    path: Path,
    *,
    label: str,
    expected_nlink: int = 1,
) -> tuple[int, bytes, tuple[int, ...]]:
    _reject_symlink_components(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LaunchError(f"{label} cannot be opened") from error
    try:
        payload, identity = _read_descriptor(
            descriptor,
            label=label,
            expected_nlink=expected_nlink,
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, payload, identity


def _descriptor_path(descriptor: int) -> str:
    for prefix in ("/proc/self/fd/", "/dev/fd/"):
        candidate = f"{prefix}{descriptor}"
        if os.path.exists(candidate):
            return candidate
    raise LaunchError("platform has no inherited-descriptor execution path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--bootstrap",
        required=True,
        type=Path,
        action=_StoreOnce,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runtime-seal", type=Path, action=_StoreOnce)
    mode.add_argument("--create-runtime-seal", type=Path, action=_StoreOnce)
    parser.add_argument("producer_arguments", nargs=argparse.REMAINDER)
    return parser


def _pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in rows:
        if key in value:
            raise LaunchError(f"outer receipt repeats key {key!r}")
        value[key] = item
    return value


def _canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(LaunchError(f"{label} contains non-finite value {raw}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchError(f"{label} is not UTF-8 JSON") from error
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if not isinstance(value, dict) or payload != canonical:
        raise LaunchError(f"{label} is not canonical JSON followed by LF")
    return value


def _strict_json_equal(observed: object, expected: object) -> bool:
    """Compare decoded JSON without bool/int or int/float aliases."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        return set(observed) == set(expected) and all(
            _strict_json_equal(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(observed, list)
        return len(observed) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise LaunchError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + "+00:00"
        )
    except ValueError as error:
        raise LaunchError(f"{label} is not a real UTC timestamp") from error
    if parsed.tzinfo != UTC:
        raise LaunchError(f"{label} is not UTC")
    return parsed


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_receipt(descriptor: int) -> bytes:
    payload = bytearray()
    while len(payload) <= _MAX_RECEIPT_BYTES:
        chunk = os.read(
            descriptor,
            _MAX_RECEIPT_BYTES + 1 - len(payload),
        )
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > _MAX_RECEIPT_BYTES:
        raise LaunchError("outer terminal receipt exceeds its byte limit")
    return bytes(payload)


def _canonical_repository(bootstrap: Path) -> Path:
    _reject_symlink_components(bootstrap, label="producer bootstrap")
    expected_suffix = Path("bench") / "world_model_lifecycle" / "producer_bootstrap.py"
    try:
        repository = bootstrap.parents[2]
    except IndexError as error:
        raise LaunchError("producer bootstrap has no canonical repository root") from error
    if (
        bootstrap != repository / expected_suffix
        or repository.resolve(strict=True) != repository
        or not (repository / "bench" / "world_model_lifecycle" / "protocol.json").is_file()
    ):
        raise LaunchError("producer bootstrap is outside its canonical repository location")
    return repository


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    if (
        set(environment) - _SAFE_ENVIRONMENT_KEYS
        or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or environment.get("LAZY_LEGACY_OP") != "False"
        or environment.get("LC_ALL") != "C.UTF-8"
        or environment.get("PATH") != "/usr/bin:/bin"
        or environment.get("PYGAME_HIDE_SUPPORT_PROMPT") != "hide"
        or environment.get("SDL_AUDIODRIVER") != "dsp"
        or environment.get("TZ") != "UTC"
        or any("\0" in key or "\0" in value for key, value in environment.items())
    ):
        raise LaunchError("WM-001 outer launch requires env -i with the exact safe runtime environment")
    return dict(sorted(environment.items()))


def _acquire_runtime_lock(repository: Path) -> int:
    parent = repository / "bench" / "world_model_lifecycle" / "results"
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(parent, label="runtime lock directory")
    lock_path = parent / ".wm001-v1.15-runtime.lock"
    existed = os.path.lexists(lock_path)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not os.path.samefile(
                lock_path,
                _descriptor_path(descriptor),
            )
        ):
            raise LaunchError("runtime lock is not one singly linked regular file")
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise LaunchError("another WM-001 v1.15 outer invocation holds the runtime lock") from error
        if not existed:
            parent_descriptor = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        return descriptor
    except BaseException:
        if "descriptor" in locals():
            os.close(descriptor)
        raise


def _prepare_completion_root(repository: Path) -> Path:
    root = repository / "bench" / "world_model_lifecycle" / "results" / "outer-completions" / "v1.15"
    root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root, label="outer completion directory")
    if root.resolve(strict=True) != root:
        raise LaunchError("outer completion directory is aliased")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return root


def _verify_unchanged(
    descriptor: int,
    *,
    expected_payload: bytes,
    expected_identity: tuple[int, ...],
    label: str,
    expected_nlink: int = 1,
) -> None:
    payload, identity = _read_descriptor(
        descriptor,
        label=label,
        expected_nlink=expected_nlink,
    )
    if payload != expected_payload or identity != expected_identity:
        raise LaunchError(f"{label} changed during producer execution")


def _completion_marker(completion_root: Path, terminal: Path) -> Path:
    return completion_root / (hashlib.sha256(str(terminal).encode("utf-8")).hexdigest() + ".json")


def _prospective_runtime_seal(repository: Path) -> Path:
    return (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "runtime-seal-v1.15.0.json"
    )


def _verify_completion_inode(
    terminal: Path,
    *,
    payload: bytes,
    identity: tuple[int, ...],
    completion_root: Path,
    label: str,
) -> None:
    marker = _completion_marker(completion_root, terminal)
    marker_fd, marker_payload, marker_identity = _open_regular(
        marker,
        label=f"{label} outer completion marker",
        expected_nlink=2,
    )
    try:
        if marker_payload != payload or marker_identity != identity or not os.path.samefile(terminal, marker):
            raise LaunchError(f"{label} completion marker is not the terminal inode")
    finally:
        os.close(marker_fd)


def _sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _recorded_result_free_inventory(
    value: object,
    *,
    dependencies: dict[str, object],
) -> dict[str, object]:
    """Validate command-10 inventory using only sealed recorded objects."""

    if not isinstance(value, dict) or set(value) != {
        "packages",
        "package_roots",
        "standard_library",
        "package_ownership",
    }:
        raise LaunchError("formal binding runtime inventory has wrong fields")
    packages = value.get("packages")
    package_roots = value.get("package_roots")
    standard_library = value.get("standard_library")
    ownership = value.get("package_ownership")
    package_fields = {
        "name",
        "version",
        "distribution_sha256",
        "declared_file_count",
        "editable",
    }
    if not isinstance(packages, list) or not packages:
        raise LaunchError("formal binding runtime package inventory is absent")
    names: list[str] = []
    for row in packages:
        if (
            not isinstance(row, dict)
            or set(row) != package_fields
            or not isinstance(row.get("name"), str)
            or re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                str(row.get("name")),
            )
            is None
            or not isinstance(row.get("version"), str)
            or not row.get("version")
            or "\0" in str(row.get("version"))
            or not _sha256_string(row.get("distribution_sha256"))
            or type(row.get("declared_file_count")) is not int
            or row["declared_file_count"] <= 0
            or row.get("editable") is not False
        ):
            raise LaunchError("formal binding runtime package row is malformed")
        names.append(row["name"])
    if (
        names[0] != "python"
        or names[1:] != sorted(names[1:])
        or len(names) != len(set(names))
        or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", packages[0]["version"])
        is None
    ):
        raise LaunchError(
            "formal binding runtime packages are duplicated or unordered"
        )

    root_fields = {
        "semantics_id",
        "path",
        "file_count",
        "directory_count",
        "total_bytes",
        "inventory_sha256",
    }

    def valid_root(row: object, semantics_id: str) -> bool:
        if not isinstance(row, dict) or set(row) != root_fields:
            return False
        path = row.get("path")
        return bool(
            row.get("semantics_id") == semantics_id
            and isinstance(path, str)
            and "\0" not in path
            and Path(path).is_absolute()
            and os.path.abspath(path) == path
            and type(row.get("file_count")) is int
            and row["file_count"] > 0
            and type(row.get("directory_count")) is int
            and row["directory_count"] >= 0
            and type(row.get("total_bytes")) is int
            and row["total_bytes"] >= 0
            and _sha256_string(row.get("inventory_sha256"))
        )

    root = package_roots[0] if isinstance(package_roots, list) and package_roots else None
    if (
        not isinstance(package_roots, list)
        or len(package_roots) != 1
        or not valid_root(root, "prospect.wm001.package-root.v2")
        or not valid_root(
            standard_library,
            "prospect.wm001.standard-library.v2",
        )
        or not isinstance(root, dict)
        or not isinstance(ownership, dict)
        or set(ownership)
        != {
            "semantics_id",
            "root",
            "file_count",
            "directory_count",
            "shared_file_count",
            "identity_sha256",
        }
        or ownership.get("semantics_id")
        != "prospect.wm001.package-ownership.v1"
        or ownership.get("root") != root.get("path")
        or ownership.get("file_count") != root.get("file_count")
        or ownership.get("directory_count") != root.get("directory_count")
        or type(ownership.get("shared_file_count")) is not int
        or ownership["shared_file_count"] < 0
        or ownership["shared_file_count"] > ownership["file_count"]
        or not _sha256_string(ownership.get("identity_sha256"))
        or packages != dependencies.get("packages")
        or package_roots != dependencies.get("package_roots")
        or standard_library != dependencies.get("standard_library")
        or ownership != dependencies.get("package_ownership")
    ):
        raise LaunchError(
            "formal binding runtime inventory differs from bound dependencies"
        )
    return value


def _recorded_fresh_identity_conformance(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "mode",
        "challenge",
        "requesting_process_id",
        "verifier_process_id",
        "matrix_contract_sha256",
        "passed",
    }
    requesting = value.get("requesting_process_id") if isinstance(value, dict) else None
    verifier = value.get("verifier_process_id") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema")
        != "prospect.wm001.fresh-runtime-identity-conformance.v1"
        or value.get("experiment_id") != "WM-001"
        or value.get("protocol_version") != "1.15.0"
        or value.get("mode") != "fresh-identity-conformance"
        or not _sha256_string(value.get("challenge"))
        or type(requesting) is not int
        or requesting <= 0
        or type(verifier) is not int
        or verifier <= 0
        or requesting == verifier
        or value.get("matrix_contract_sha256")
        != _DEVELOPMENT_MATRIX_CONTRACT_SHA256
        or value.get("passed") is not True
    ):
        raise LaunchError(
            "formal binding fresh runtime identity conformance is malformed"
        )
    return value


def _recorded_accepted_closure_evidence(
    value: object,
    *,
    expected: dict[str, object] | None = None,
) -> dict[str, object]:
    fields = {
        "schema",
        "mode",
        "passed",
        "development_closure_sha256",
        "producer_manifest_sha256",
        "raw_result_sha256",
        "closure_attempt_manifest_sha256",
        "closure_outer_completion_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema")
        != "prospect.wm001.preformal-runtime-check.v1"
        or value.get("mode") != "accepted-closure-evidence"
        or value.get("passed") is not True
        or any(
            not _sha256_string(value.get(field))
            for field in (
                "development_closure_sha256",
                "producer_manifest_sha256",
                "raw_result_sha256",
                "closure_attempt_manifest_sha256",
                "closure_outer_completion_sha256",
            )
        )
        or (expected is not None and value != expected)
    ):
        raise LaunchError(
            "formal binding accepted-closure evidence is malformed or misbound"
        )
    return value


def _recorded_runtime_conformance(
    value: object,
    *,
    dependencies: dict[str, object],
    audit_execution: object,
    device: object,
) -> dict[str, object]:
    if not isinstance(value, dict) or not isinstance(audit_execution, dict):
        raise LaunchError("formal binding runtime conformance is malformed")
    inventory = _recorded_result_free_inventory(
        value.get("inventory"),
        dependencies=dependencies,
    )
    fresh = _recorded_fresh_identity_conformance(
        value.get("fresh_runtime_identity_conformance")
    )
    if (
        set(value)
        != {
            "schema",
            "mode",
            "device",
            "passed",
            "inventory",
            "inventory_sha256",
            "conformance_sha256",
            "fresh_runtime_identity_conformance",
            "fresh_runtime_identity_conformance_sha256",
            "restart_runtime_conformance_report_sha256",
            "restart_runtime_execution_receipt_sha256",
            "restart_runtime_support_files",
            "restart_runtime_repeat_count",
            "restart_runtime_path_descriptor_equal",
            "repeat_count",
            "path_descriptor_equal",
        }
        or value.get("schema")
        != "prospect.wm001.preformal-runtime-check.v1"
        or value.get("mode") != "bootstrap-inventory-conformance"
        or value.get("device") != device
        or value.get("passed") is not True
        or any(
            not _sha256_string(value.get(field))
            for field in (
                "inventory_sha256",
                "conformance_sha256",
                "fresh_runtime_identity_conformance_sha256",
                "restart_runtime_conformance_report_sha256",
                "restart_runtime_execution_receipt_sha256",
            )
        )
        or value.get("inventory_sha256") != _canonical_digest(inventory)
        or value.get("conformance_sha256")
        != _canonical_digest(audit_execution)
        or value.get("fresh_runtime_identity_conformance_sha256")
        != _canonical_digest(fresh)
        or value.get("restart_runtime_support_files")
        != [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ]
        or value.get("restart_runtime_support_files")
        != audit_execution.get("restart_runtime_support_files")
        or value.get("restart_runtime_repeat_count") != 3
        or value.get("restart_runtime_repeat_count")
        != audit_execution.get("restart_runtime_repeat_count")
        or value.get("restart_runtime_path_descriptor_equal") is not True
        or value.get("restart_runtime_path_descriptor_equal")
        is not audit_execution.get("restart_runtime_path_descriptor_equal")
        or value.get("repeat_count") != 3
        or value.get("repeat_count") != audit_execution.get("repeat_count")
        or value.get("path_descriptor_equal") is not True
        or value.get("path_descriptor_equal")
        is not audit_execution.get("path_descriptor_equal")
        or value.get("restart_runtime_conformance_report_sha256")
        != audit_execution.get("restart_runtime_conformance_report_sha256")
        or value.get("restart_runtime_execution_receipt_sha256")
        != audit_execution.get("restart_runtime_execution_receipt_sha256")
    ):
        raise LaunchError("formal binding runtime conformance is malformed")
    return value


def _regular_row(
    path: Path,
    *,
    label: str,
    expected_nlink: int = 1,
) -> tuple[dict[str, object], bytes, tuple[int, ...]]:
    descriptor, payload, identity = _open_regular(
        path,
        label=label,
        expected_nlink=expected_nlink,
    )
    try:
        return (
            {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            payload,
            identity,
        )
    finally:
        os.close(descriptor)


def _verify_development_producer(
    *,
    repository: Path,
    completion_root: Path,
) -> list[dict[str, object]]:
    """Reconstruct the two producer rows consumed by the development audit."""

    root = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "qualification-v1.15.0"
    )
    _reject_symlink_components(root, label="canonical development qualification")
    if not root.is_dir():
        raise LaunchError("canonical development qualification is absent")
    manifest_path = root / "producer-manifest.json"
    manifest_row, manifest_payload, manifest_identity = _regular_row(
        manifest_path,
        label="canonical development producer manifest",
        expected_nlink=2,
    )
    manifest = _canonical_object(
        manifest_payload,
        label="canonical development producer manifest",
    )
    expected_manifest_fields = {
        "schema",
        "experiment_id",
        "lane",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "error",
        "manifest_excludes",
        "file_count",
        "files",
    }
    rows = manifest.get("files")
    started_at = _parse_utc_timestamp(
        manifest.get("started_at_utc"),
        label="canonical development producer start time",
    )
    completed_at = _parse_utc_timestamp(
        manifest.get("completed_at_utc"),
        label="canonical development producer completion time",
    )
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema") != "prospect.wm001.producer-manifest.v1"
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("lane") != "development"
        or manifest.get("status") != "completed"
        or manifest.get("error") is not None
        or manifest.get("manifest_excludes") != ["producer-manifest.json"]
        or started_at > completed_at
        or not isinstance(rows, list)
        or type(manifest.get("file_count")) is not int
        or manifest.get("file_count") != len(rows)
    ):
        raise LaunchError("canonical development producer manifest is malformed")
    actual_rows: list[dict[str, object]] = []
    for directory, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        directory_names.sort()
        filenames.sort()
        for name in directory_names:
            candidate = current / name
            metadata = os.lstat(candidate)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise LaunchError("canonical development producer contains an unsafe directory")
        for name in filenames:
            path = current / name
            if path == manifest_path:
                continue
            row, _, _ = _regular_row(
                path,
                label=f"canonical development producer file {path.relative_to(root).as_posix()}",
            )
            actual_rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
            )
    actual_rows.sort(key=lambda row: str(row["path"]))
    if not _strict_json_equal(rows, actual_rows):
        raise LaunchError("canonical development producer files differ from its manifest")
    result_path = root / "result.json"
    result_row, result_payload, _ = _regular_row(
        result_path,
        label="canonical development raw result",
    )
    result = _canonical_object(
        result_payload,
        label="canonical development raw result",
    )
    if (
        result.get("schema") != "prospect.world-model-lifecycle.raw-result.v9"
        or result.get("experiment_id") != "WM-001"
        or result.get("protocol_version") != "1.15.0"
        or result.get("lane") != "development"
    ):
        raise LaunchError("canonical development raw result is malformed")
    _verify_completion_inode(
        manifest_path,
        payload=manifest_payload,
        identity=manifest_identity,
        completion_root=completion_root,
        label="canonical development producer",
    )
    return [manifest_row, result_row]


def _verify_development_audit(
    *,
    repository: Path,
    completion_root: Path,
    producer_rows: list[dict[str, object]],
) -> tuple[
    Path,
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    """Reconstruct the accepted canonical development-audit package."""

    attempt = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "operator-v1.15"
        / "audits"
        / "development-audit-v1.15.0"
    )
    terminal = attempt / _OPERATOR_TERMINAL
    terminal_row, terminal_payload, terminal_identity = _regular_row(
        terminal,
        label="canonical development audit terminal",
        expected_nlink=2,
    )
    manifest = _canonical_object(
        terminal_payload,
        label="canonical development audit terminal",
    )
    expected_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "assurance",
        "kind",
        "lane",
        "status",
        "inputs",
        "primary",
        "error",
        "files",
        "file_count",
        "manifest_excludes",
    }
    primary = manifest.get("primary")
    files = manifest.get("files")
    expected_primary_fields = {
        "producer_root",
        "audit_file",
        "executions",
        "execution_failures",
        "reproduction_file",
        "reproduction_runtime_file",
        "claim_file",
    }
    producer_root = (
        repository
        / "bench"
        / "world_model_lifecycle"
        / "results"
        / "development"
        / "qualification-v1.15.0"
    )
    if (
        set(manifest) != expected_fields
        or manifest.get("schema") != _OPERATOR_ATTEMPT_SCHEMA
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("protocol_version") != "1.15.0"
        or not _strict_json_equal(
            manifest.get("assurance"),
            _ASSURANCE,
        )
        or manifest.get("kind") != "audit"
        or manifest.get("lane") != "development"
        or manifest.get("status") != "accepted"
        or not _strict_json_equal(
            manifest.get("inputs"),
            producer_rows,
        )
        or manifest.get("error") is not None
        or manifest.get("manifest_excludes") != [_OPERATOR_TERMINAL]
        or not isinstance(primary, dict)
        or set(primary) != expected_primary_fields
        or primary.get("producer_root") != str(producer_root)
        or primary.get("audit_file") != "independent-audit.json"
        or primary.get("executions")
        != [
            "audit-execution-01.execution.json",
            "audit-execution-02.execution.json",
        ]
        or primary.get("execution_failures") != []
        or primary.get("reproduction_file") != "audit-reproduction.json"
        or primary.get("reproduction_runtime_file")
        != "audit-execution-02.runtime.json"
        or primary.get("claim_file") is not None
        or not isinstance(files, list)
        or type(manifest.get("file_count")) is not int
        or manifest.get("file_count") != len(files)
    ):
        raise LaunchError("canonical development audit terminal is malformed")
    actual_file_rows: list[dict[str, object]] = []
    closure_input_rows: list[dict[str, object]] = []
    for entry in sorted(os.scandir(attempt), key=lambda item: item.name):
        path = attempt / entry.name
        if (
            entry.name.startswith(".")
            or Path(entry.name).name != entry.name
            or not entry.is_file(follow_symlinks=False)
            or entry.is_symlink()
        ):
            raise LaunchError("canonical development audit namespace is unsafe")
        row, _, _ = _regular_row(
            path,
            label=f"canonical development audit file {entry.name}",
            expected_nlink=(2 if entry.name == _OPERATOR_TERMINAL else 1),
        )
        closure_input_rows.append(row)
        if entry.name != _OPERATOR_TERMINAL:
            actual_file_rows.append(
                {
                    "path": entry.name,
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
            )
    if not _strict_json_equal(files, actual_file_rows):
        raise LaunchError("canonical development audit files differ from its terminal")
    marker = _completion_marker(completion_root, terminal)
    marker_row, marker_payload, marker_identity = _regular_row(
        marker,
        label="canonical development audit outer completion",
        expected_nlink=2,
    )
    _verify_completion_inode(
        terminal,
        payload=terminal_payload,
        identity=terminal_identity,
        completion_root=completion_root,
        label="canonical development audit",
    )
    if marker_payload != terminal_payload or marker_identity != terminal_identity:
        raise LaunchError("canonical development audit completion identity changed")
    closure_input_rows.append(marker_row)
    return attempt, closure_input_rows, terminal_row, manifest


def _verify_closure_authorization(
    *,
    repository: Path,
    completion_root: Path,
    binding: dict[str, object],
) -> tuple[
    Path,
    dict[str, object],
    Path,
    bytes,
    tuple[int, ...],
    Path,
    bytes,
    tuple[int, ...],
    dict[str, object],
]:
    results = repository / "bench" / "world_model_lifecycle" / "results"
    closure_path = results / "development" / "development-closure-v1.15.0.json"
    closure_row, closure_payload, _ = _regular_row(
        closure_path,
        label="canonical development closure",
    )
    closure = _canonical_object(
        closure_payload,
        label="canonical development closure",
    )
    development = binding.get("development_qualification")
    if (
        not isinstance(development, dict)
        or type(development.get("closure_bytes")) is not int
        or development.get("closure_bytes") != closure_row["bytes"]
        or development.get("closure_sha256") != closure_row["sha256"]
        or closure.get("engineering_verified") is not True
        or closure.get("audit_reproduced") is not True
        or closure.get("performance_values_bound") is not False
    ):
        raise LaunchError("formal binding differs from the canonical development closure")

    attempt = results / "operator-v1.15" / "closures" / "development-closure-v1.15.0"
    terminal = attempt / _OPERATOR_TERMINAL
    terminal_row, terminal_payload, terminal_identity = _regular_row(
        terminal,
        label="canonical closure attempt terminal",
        expected_nlink=2,
    )
    terminal_manifest = _canonical_object(
        terminal_payload,
        label="canonical closure attempt terminal",
    )
    expected_terminal_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "assurance",
        "kind",
        "lane",
        "status",
        "inputs",
        "primary",
        "error",
        "files",
        "file_count",
        "manifest_excludes",
    }
    if (
        set(terminal_manifest) != expected_terminal_fields
        or terminal_manifest.get("schema") != _OPERATOR_ATTEMPT_SCHEMA
        or terminal_manifest.get("experiment_id") != "WM-001"
        or terminal_manifest.get("protocol_version") != "1.15.0"
        or not _strict_json_equal(
            terminal_manifest.get("assurance"),
            _ASSURANCE,
        )
        or terminal_manifest.get("kind") != "closure"
        or terminal_manifest.get("lane") != "development"
        or terminal_manifest.get("status") != "accepted"
        or not _strict_json_equal(
            terminal_manifest.get("primary"),
            {"closure_reference_file": "closure-reference.json"},
        )
        or terminal_manifest.get("error") is not None
        or terminal_manifest.get("manifest_excludes") != [_OPERATOR_TERMINAL]
    ):
        raise LaunchError("canonical closure attempt terminal is not accepted")
    rows = terminal_manifest.get("files")
    if (
        not isinstance(rows, list)
        or type(terminal_manifest.get("file_count")) is not int
        or terminal_manifest.get("file_count") != len(rows)
    ):
        raise LaunchError("canonical closure attempt file identities are malformed")
    actual_rows: list[dict[str, object]] = []
    for entry in sorted(os.scandir(attempt), key=lambda item: item.name):
        if entry.name == _OPERATOR_TERMINAL:
            continue
        path = attempt / entry.name
        if (
            entry.name.startswith(".")
            or Path(entry.name).name != entry.name
            or not entry.is_file(follow_symlinks=False)
            or entry.is_symlink()
        ):
            raise LaunchError("canonical closure attempt namespace is unsafe")
        descriptor, payload, _ = _open_regular(
            path,
            label=f"canonical closure attempt file {entry.name}",
        )
        try:
            actual_rows.append(
                {
                    "path": entry.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        finally:
            os.close(descriptor)
    if not _strict_json_equal(rows, actual_rows):
        raise LaunchError("canonical closure attempt files differ from its terminal")
    reference_path = attempt / "closure-reference.json"
    reference_descriptor, reference_payload, _ = _open_regular(
        reference_path,
        label="canonical closure reference",
    )
    try:
        reference = _canonical_object(
            reference_payload,
            label="canonical closure reference",
        )
    finally:
        os.close(reference_descriptor)
    expected_reference_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "closure_marker",
        "closure_sha256",
        "qualification_archive",
        "producer_root",
        "audit_attempt",
        "audit_attempt_manifest_sha256",
        "fresh_reopen_file",
        "fresh_reopen_sha256",
    }
    expected_audit = results / "operator-v1.15" / "audits" / "development-audit-v1.15.0"
    expected_producer = (
        results
        / "development"
        / "qualification-v1.15.0"
    )
    if (
        set(reference) != expected_reference_fields
        or reference.get("schema") != _CLOSURE_REFERENCE_SCHEMA
        or reference.get("experiment_id") != "WM-001"
        or reference.get("protocol_version") != "1.15.0"
        or reference.get("closure_marker") != str(closure_path)
        or reference.get("closure_sha256") != hashlib.sha256(closure_payload).hexdigest()
        or not _strict_json_equal(
            reference.get("qualification_archive"),
            closure.get("qualification_archive"),
        )
        or reference.get("producer_root") != str(expected_producer)
        or closure.get("producer_root") != str(expected_producer)
        or reference.get("audit_attempt") != str(expected_audit)
        or not _sha256_string(reference.get("audit_attempt_manifest_sha256"))
        or reference.get("fresh_reopen_file")
        != "fresh-runtime-reopen.json"
        or not _sha256_string(reference.get("fresh_reopen_sha256"))
    ):
        raise LaunchError("canonical closure reference differs from live evidence")
    fresh_reopen_path = attempt / "fresh-runtime-reopen.json"
    fresh_descriptor, fresh_payload, _ = _open_regular(
        fresh_reopen_path,
        label="canonical fresh closure-reopen report",
    )
    try:
        fresh_reopen = _canonical_object(
            fresh_payload,
            label="canonical fresh closure-reopen report",
        )
    finally:
        os.close(fresh_descriptor)
    archive = closure.get("qualification_archive")
    members = archive.get("members") if isinstance(archive, dict) else None
    member_digests = (
        {
            row.get("path"): row.get("sha256")
            for row in members
            if isinstance(row, dict)
        }
        if isinstance(members, list)
        else {}
    )
    requesting_process_id = fresh_reopen.get(
        "requesting_process_id"
    )
    verifier_process_id = fresh_reopen.get("verifier_process_id")
    if (
        set(fresh_reopen)
        != {
            "schema",
            "experiment_id",
            "protocol_version",
            "mode",
            "challenge",
            "requesting_process_id",
            "verifier_process_id",
            "matrix_contract_sha256",
            "development_closure_sha256",
            "producer_manifest_sha256",
            "raw_result_sha256",
            "passed",
        }
        or fresh_reopen.get("schema")
        != "prospect.wm001.development-closure-fresh-reopen.v1"
        or fresh_reopen.get("experiment_id") != "WM-001"
        or fresh_reopen.get("protocol_version") != "1.15.0"
        or fresh_reopen.get("mode") != "fresh-closure-reopen"
        or not _sha256_string(fresh_reopen.get("challenge"))
        or type(requesting_process_id) is not int
        or requesting_process_id <= 0
        or type(verifier_process_id) is not int
        or verifier_process_id <= 0
        or requesting_process_id == verifier_process_id
        or fresh_reopen.get("matrix_contract_sha256")
        != "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
        or fresh_reopen.get("development_closure_sha256")
        != closure_row["sha256"]
        or fresh_reopen.get("producer_manifest_sha256")
        != member_digests.get("producer/producer-manifest.json")
        or fresh_reopen.get("raw_result_sha256")
        != member_digests.get("producer/result.json")
        or fresh_reopen.get("passed") is not True
        or hashlib.sha256(fresh_payload).hexdigest()
        != reference.get("fresh_reopen_sha256")
    ):
        raise LaunchError(
            "canonical fresh closure-reopen report differs from live evidence"
        )
    producer_rows = _verify_development_producer(
        repository=repository,
        completion_root=completion_root,
    )
    (
        audit_attempt,
        audit_input_rows,
        audit_terminal_row,
        _,
    ) = _verify_development_audit(
        repository=repository,
        completion_root=completion_root,
        producer_rows=producer_rows,
    )
    if (
        audit_attempt != expected_audit
        or reference.get("audit_attempt_manifest_sha256")
        != audit_terminal_row["sha256"]
        or not _strict_json_equal(
            terminal_manifest.get("inputs"),
            [*producer_rows, *audit_input_rows],
        )
        or development.get("producer_manifest_sha256")
        != producer_rows[0]["sha256"]
        or development.get("raw_result_sha256")
        != producer_rows[1]["sha256"]
    ):
        raise LaunchError(
            "canonical closure authorization inputs differ from live producer/audit evidence"
        )
    marker = _completion_marker(completion_root, terminal)
    marker_row, marker_payload, marker_identity = _regular_row(
        marker,
        label="canonical closure outer completion",
        expected_nlink=2,
    )
    _verify_completion_inode(
        terminal,
        payload=terminal_payload,
        identity=terminal_identity,
        completion_root=completion_root,
        label="canonical closure attempt",
    )
    if (
        marker_payload != terminal_payload
        or marker_identity != terminal_identity
        or marker_row["sha256"] != terminal_row["sha256"]
    ):
        raise LaunchError("canonical closure completion identity changed")
    return (
        closure_path,
        closure_row,
        terminal,
        terminal_payload,
        terminal_identity,
        marker,
        marker_payload,
        marker_identity,
        {
            "schema": "prospect.wm001.preformal-runtime-check.v1",
            "mode": "accepted-closure-evidence",
            "passed": True,
            "development_closure_sha256": closure_row["sha256"],
            "producer_manifest_sha256": producer_rows[0]["sha256"],
            "raw_result_sha256": producer_rows[1]["sha256"],
            "closure_attempt_manifest_sha256": terminal_row["sha256"],
            "closure_outer_completion_sha256": marker_row["sha256"],
        },
    )


def _verify_binding_attempt_terminal(
    attempt: Path,
    *,
    binding_payload: bytes,
    repository: Path,
    completion_root: Path,
) -> None:
    terminal = attempt / _OPERATOR_TERMINAL
    terminal_fd, terminal_payload, terminal_identity = _open_regular(
        terminal,
        label="formal binding attempt terminal",
        expected_nlink=2,
    )
    try:
        manifest = _canonical_object(
            terminal_payload,
            label="formal binding attempt terminal",
        )
        expected_fields = {
            "schema",
            "experiment_id",
            "protocol_version",
            "assurance",
            "kind",
            "lane",
            "status",
            "inputs",
            "primary",
            "error",
            "files",
            "file_count",
            "manifest_excludes",
        }
        rows = manifest.get("files")
        inputs = manifest.get("inputs")
        if (
            set(manifest) != expected_fields
            or manifest.get("schema") != _OPERATOR_ATTEMPT_SCHEMA
            or manifest.get("experiment_id") != "WM-001"
            or manifest.get("protocol_version") != "1.15.0"
            or not _strict_json_equal(
                manifest.get("assurance"),
                _ASSURANCE,
            )
            or manifest.get("kind") != "binding"
            or manifest.get("lane") is not None
            or manifest.get("status") != "accepted"
            or not _strict_json_equal(
                manifest.get("primary"),
                {"binding_file": "formal-binding.json"},
            )
            or manifest.get("error") is not None
            or not isinstance(rows, list)
            or type(manifest.get("file_count")) is not int
            or manifest.get("file_count") != len(rows)
            or manifest.get("manifest_excludes") != [_OPERATOR_TERMINAL]
            or not isinstance(inputs, list)
        ):
            raise LaunchError("formal binding attempt terminal is malformed")
        input_paths: list[str] = []
        for row in inputs:
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row.get("path"), str)
                or not Path(row["path"]).is_absolute()
                or Path(os.path.abspath(row["path"])) != Path(row["path"])
                or type(row.get("bytes")) is not int
                or row["bytes"] < 0
                or not _sha256_string(row.get("sha256"))
            ):
                raise LaunchError("formal binding attempt input identity is malformed")
            input_paths.append(row["path"])
        if len(input_paths) != len(set(input_paths)):
            raise LaunchError("formal binding attempt input identities repeat")
        actual_rows: list[dict[str, object]] = []
        for entry in sorted(os.scandir(attempt), key=lambda item: item.name):
            if entry.name == _OPERATOR_TERMINAL:
                continue
            path = attempt / entry.name
            if (
                entry.name.startswith(".")
                or Path(entry.name).name != entry.name
                or not entry.is_file(follow_symlinks=False)
                or entry.is_symlink()
            ):
                raise LaunchError("formal binding attempt namespace is unsafe")
            descriptor, payload, _ = _open_regular(
                path,
                label=f"formal binding attempt file {entry.name}",
            )
            try:
                actual_rows.append(
                    {
                        "path": entry.name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            finally:
                os.close(descriptor)
        if not _strict_json_equal(rows, actual_rows) or not any(
            row
            == {
                "path": "formal-binding.json",
                "bytes": len(binding_payload),
                "sha256": hashlib.sha256(binding_payload).hexdigest(),
            }
            for row in actual_rows
        ):
            raise LaunchError("formal binding attempt files differ from its terminal")
        binding = _canonical_object(
            binding_payload,
            label="formal binding runtime seal",
        )
        source = binding.get("source")
        dependencies = binding.get("dependencies")
        runtime = binding.get("runtime")
        if (
            not isinstance(source, dict)
            or not isinstance(dependencies, dict)
            or not isinstance(runtime, dict)
            or not inputs
        ):
            raise LaunchError("formal binding attempt has no authorization inputs")
        preflight_path = attempt / _FORMAL_INPUT_PREFLIGHT_NAME
        preflight_fd, preflight_payload, _ = _open_regular(
            preflight_path,
            label="formal input preflight receipt",
        )
        try:
            preflight = _canonical_object(
                preflight_payload,
                label="formal input preflight receipt",
            )
        finally:
            os.close(preflight_fd)
        preflight_fields = {
            "schema",
            "experiment_id",
            "protocol_version",
            "binding_bytes",
            "binding_sha256",
            "preformal_report_sha256",
            "development_closure_sha256",
            "accepted_closure_evidence_sha256",
            "runtime_conformance_sha256",
            "auditor_source_sha256",
            "passed",
        }
        implementation = source.get("implementation_files")
        auditor_rows = (
            [
                row
                for row in implementation
                if isinstance(row, dict)
                and row.get("path")
                == "bench/world_model_lifecycle/artifact_audit.py"
            ]
            if isinstance(implementation, list)
            else []
        )
        if (
            set(preflight) != preflight_fields
            or preflight.get("schema")
            != _FORMAL_INPUT_PREFLIGHT_SCHEMA
            or preflight.get("experiment_id") != "WM-001"
            or preflight.get("protocol_version") != "1.15.0"
            or type(preflight.get("binding_bytes")) is not int
            or preflight.get("binding_bytes") != len(binding_payload)
            or preflight.get("binding_sha256")
            != hashlib.sha256(binding_payload).hexdigest()
            or preflight.get("passed") is not True
            or any(
                not _sha256_string(preflight.get(field))
                for field in (
                    "preformal_report_sha256",
                    "development_closure_sha256",
                    "accepted_closure_evidence_sha256",
                    "runtime_conformance_sha256",
                    "auditor_source_sha256",
                )
            )
            or len(auditor_rows) != 1
            or preflight.get("auditor_source_sha256")
            != auditor_rows[0].get("sha256")
        ):
            raise LaunchError(
                "formal input preflight receipt is malformed or misbound"
            )
        report_value = inputs[0].get("path")
        if not isinstance(report_value, str):
            raise LaunchError("formal binding preformal report input is malformed")
        report_path = Path(report_value)
        expected_report = (
            repository
            / "bench"
            / "world_model_lifecycle"
            / "results"
            / "development"
            / "v1.15.0"
            / "preformal"
            / _PREFORMAL_REPORT_NAME
        )
        if report_path != expected_report:
            raise LaunchError("formal binding does not use the canonical preformal report")
        report_row, report_payload, _ = _regular_row(
            report_path,
            label="formal binding preformal report",
        )
        report = _canonical_object(
            report_payload,
            label="formal binding preformal report",
        )
        commands = report.get("commands")
        if (
            report.get("schema")
            != "prospect.wm001.preformal-test-report.v2"
            or report.get("experiment_id") != "WM-001"
            or report.get("protocol_version") != "1.15.0"
            or report.get("device") not in {"cpu", "cuda"}
            or report.get("device") != runtime.get("device")
            or report.get("all_pass") is not True
            or not isinstance(commands, list)
            or tuple(
                row.get("name") if isinstance(row, dict) else None
                for row in commands
            )
            != _PREFORMAL_COMMAND_NAMES
        ):
            raise LaunchError("formal binding preformal report is malformed")
        source_logs = source.get("test_log_files")
        if not isinstance(source_logs, list):
            raise LaunchError("formal binding preformal logs are malformed")
        log_input_rows: list[dict[str, object]] = []
        log_payloads: dict[str, bytes] = {}
        for index, row in enumerate(source_logs):
            if (
                not isinstance(row, dict)
                or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row.get("path"), str)
                or Path(row["path"]).name != row["path"]
                or row["path"].startswith(".")
                or type(row.get("bytes")) is not int
                or row["bytes"] < 0
                or not _sha256_string(row.get("sha256"))
            ):
                raise LaunchError("formal binding preformal log identity is malformed")
            log_path = report_path.with_name(row["path"])
            live_row, live_payload, _ = _regular_row(
                log_path,
                label=f"formal binding preformal log {index}",
            )
            if live_row["bytes"] != row["bytes"] or live_row["sha256"] != row["sha256"]:
                raise LaunchError("formal binding preformal log bytes changed")
            log_payloads[row["path"]] = live_payload
            log_input_rows.append(live_row)
        runtime_receipts: dict[str, dict[str, object]] = {}
        for command_name, mode in (
            (
                "runtime-accepted-closure-evidence",
                "accepted-closure-evidence",
            ),
            (
                "runtime-bootstrap-inventory-conformance",
                "bootstrap-inventory-conformance",
            ),
        ):
            command = next(
                row
                for row in commands
                if isinstance(row, dict) and row.get("name") == command_name
            )
            stdout = command.get("stdout")
            stderr = command.get("stderr")
            ordinal = _PREFORMAL_COMMAND_NAMES.index(command_name) + 1
            if (
                type(command.get("exit_code")) is not int
                or command.get("exit_code") != 0
                or command.get("passed") is not True
                or not isinstance(stdout, dict)
                or set(stdout) != {"file", "bytes", "sha256"}
                or not isinstance(stdout.get("file"), str)
                or type(stdout.get("bytes")) is not int
                or stdout["bytes"] < 1
                or not _sha256_string(stdout.get("sha256"))
                or stdout.get("file")
                != (
                    f"{_PREFORMAL_LOG_PREFIX}{ordinal:02d}-{command_name}."
                    f"stdout.{stdout.get('sha256')}.log"
                )
                or stdout["file"] not in log_payloads
                or not isinstance(stderr, dict)
                or set(stderr) != {"file", "bytes", "sha256"}
                or type(stderr.get("bytes")) is not int
                or stderr.get("bytes") != 0
                or stderr.get("sha256") != _SHA256_EMPTY
                or stderr.get("file")
                != (
                    f"{_PREFORMAL_LOG_PREFIX}{ordinal:02d}-{command_name}."
                    f"stderr.{_SHA256_EMPTY}.log"
                )
                or stderr.get("file") not in log_payloads
                or log_payloads[str(stderr.get("file"))] != b""
            ):
                raise LaunchError(
                    "formal binding runtime preformal output is malformed"
                )
            output_payload = log_payloads[stdout["file"]]
            if (
                len(output_payload) != stdout["bytes"]
                or hashlib.sha256(output_payload).hexdigest()
                != stdout["sha256"]
            ):
                raise LaunchError(
                    "formal binding runtime preformal output identity changed"
                )
            output = _canonical_object(
                output_payload,
                label=f"formal binding {command_name} output",
            )
            if (
                output.get("schema")
                != "prospect.wm001.preformal-runtime-check.v1"
                or output.get("mode") != mode
                or output.get("passed") is not True
            ):
                raise LaunchError(
                    "formal binding runtime preformal output did not pass"
                )
            if command_name == "runtime-bootstrap-inventory-conformance":
                output = _recorded_runtime_conformance(
                    output,
                    dependencies=dependencies,
                    audit_execution=binding.get("audit_execution"),
                    device=report.get("device"),
                )
            else:
                output = _recorded_accepted_closure_evidence(output)
            runtime_receipts[command_name] = output
        (
            closure_path,
            closure_row,
            closure_terminal,
            closure_terminal_payload,
            _,
            closure_completion,
            closure_completion_payload,
            _,
            expected_accepted_closure,
        ) = _verify_closure_authorization(
            repository=repository,
            completion_root=completion_root,
            binding=binding,
        )
        closure_terminal_row = {
            "path": str(closure_terminal),
            "bytes": len(closure_terminal_payload),
            "sha256": hashlib.sha256(closure_terminal_payload).hexdigest(),
        }
        closure_completion_row = {
            "path": str(closure_completion),
            "bytes": len(closure_completion_payload),
            "sha256": hashlib.sha256(closure_completion_payload).hexdigest(),
        }
        _recorded_accepted_closure_evidence(
            runtime_receipts["runtime-accepted-closure-evidence"],
            expected=expected_accepted_closure,
        )
        expected_inputs = [
            report_row,
            *log_input_rows,
            closure_row,
            closure_terminal_row,
            closure_completion_row,
        ]
        if (
            source.get("test_report_file") != _PREFORMAL_REPORT_NAME
            or type(source.get("test_report_bytes")) is not int
            or source.get("test_report_bytes") != report_row["bytes"]
            or source.get("test_report_sha256") != report_row["sha256"]
            or not _strict_json_equal(inputs, expected_inputs)
            or preflight.get("preformal_report_sha256")
            != report_row["sha256"]
            or preflight.get("development_closure_sha256")
            != closure_row["sha256"]
            or preflight.get("accepted_closure_evidence_sha256")
            != _canonical_digest(
                runtime_receipts[
                    "runtime-accepted-closure-evidence"
                ]
            )
            or preflight.get("runtime_conformance_sha256")
            != _canonical_digest(
                runtime_receipts[
                    "runtime-bootstrap-inventory-conformance"
                ]
            )
            or closure_path
            != (
                repository
                / "bench"
                / "world_model_lifecycle"
                / "results"
                / "development"
                / "development-closure-v1.15.0.json"
            )
        ):
            raise LaunchError("formal binding authorization inputs differ from live evidence")
        _verify_completion_inode(
            terminal,
            payload=terminal_payload,
            identity=terminal_identity,
            completion_root=completion_root,
            label="formal binding attempt",
        )
    finally:
        os.close(terminal_fd)


def _open_typed_runtime_custody(
    path: Path,
    *,
    repository: Path,
    completion_root: Path,
) -> tuple[int, bytes, tuple[int, ...], int]:
    """Capture exactly one of the two protocol-authorized runtime seals."""

    results_root = repository / "bench" / "world_model_lifecycle" / "results"
    binding_attempt = results_root / "operator-v1.15" / "bindings" / "formal-binding-v1.15.0"
    formal_binding = binding_attempt / "formal-binding.json"
    prospective_runtime_seal = _prospective_runtime_seal(repository)
    if (
        not path.is_absolute()
        or Path(os.path.abspath(path)) != path
        or path.resolve(strict=False) != path
        or path not in {formal_binding, prospective_runtime_seal}
    ):
        raise LaunchError("runtime seal path is not one canonical protocol-1.15 seal")
    is_formal_binding = path == formal_binding
    expected_nlink = 1 if is_formal_binding else 2
    seal_fd, seal_payload, seal_identity = _open_regular(
        path,
        label=("formal binding runtime seal" if is_formal_binding else "prospective runtime seal"),
        expected_nlink=expected_nlink,
    )
    try:
        value = _canonical_object(
            seal_payload,
            label=("formal binding runtime seal" if is_formal_binding else "prospective runtime seal"),
        )
        expected_schema = _FORMAL_BINDING_SCHEMA if is_formal_binding else _RUNTIME_SEAL_SCHEMA
        if (
            value.get("schema") != expected_schema
            or value.get("experiment_id") != "WM-001"
            or not _strict_json_equal(
                value.get("assurance"),
                _ASSURANCE,
            )
            or (
                not is_formal_binding
                and (
                    set(value) != _RUNTIME_SEAL_FIELDS
                    or value.get("protocol_version") != "1.15.0"
                )
            )
        ):
            raise LaunchError("runtime custody schema or assurance is invalid")
        if is_formal_binding:
            _verify_binding_attempt_terminal(
                binding_attempt,
                binding_payload=seal_payload,
                repository=repository,
                completion_root=completion_root,
            )
        else:
            _verify_completion_inode(
                path,
                payload=seal_payload,
                identity=seal_identity,
                completion_root=completion_root,
                label="prospective runtime seal",
            )
        return seal_fd, seal_payload, seal_identity, expected_nlink
    except BaseException:
        os.close(seal_fd)
        raise


def _commit_outer_receipt(
    payload: bytes,
    *,
    repository: Path,
    completion_root: Path,
) -> int:
    receipt = _canonical_object(
        payload,
        label="outer terminal receipt",
    )
    expected = {
        "schema",
        "experiment_id",
        "protocol_version",
        "assurance",
        "trust_model",
        "terminal_path",
        "terminal_bytes",
        "terminal_sha256",
        "logical_exit_code",
    }
    terminal_raw = receipt.get("terminal_path")
    logical_exit_code = receipt.get("logical_exit_code")
    terminal_bytes = receipt.get("terminal_bytes")
    terminal_sha256 = receipt.get("terminal_sha256")
    results_root = repository / "bench" / "world_model_lifecycle" / "results"
    if (
        set(receipt) != expected
        or receipt.get("schema") != _OUTER_RECEIPT_SCHEMA
        or receipt.get("experiment_id") != "WM-001"
        or receipt.get("protocol_version") != "1.15.0"
        or not _strict_json_equal(
            receipt.get("assurance"),
            _ASSURANCE,
        )
        or receipt.get("trust_model") != _OUTER_TRUST_MODEL
        or not isinstance(terminal_raw, str)
        or type(terminal_bytes) is not int
        or terminal_bytes < 0
        or not isinstance(terminal_sha256, str)
        or len(terminal_sha256) != 64
        or any(character not in "0123456789abcdef" for character in terminal_sha256)
        or type(logical_exit_code) is not int
        or logical_exit_code not in {0, 1, 2}
    ):
        raise LaunchError("outer terminal receipt is malformed")
    terminal = Path(terminal_raw)
    if (
        not terminal.is_absolute()
        or Path(os.path.abspath(terminal)) != terminal
        or terminal.resolve(strict=False) != terminal
        or not terminal.is_relative_to(results_root)
        or terminal.is_relative_to(completion_root)
        or any(part.startswith(".") for part in terminal.relative_to(results_root).parts)
    ):
        raise LaunchError("outer receipt terminal path is outside results")
    terminal_fd, terminal_payload, _ = _open_regular(
        terminal,
        label="outer receipt terminal",
    )
    try:
        if len(terminal_payload) != terminal_bytes or hashlib.sha256(terminal_payload).hexdigest() != terminal_sha256:
            raise LaunchError("outer receipt terminal differs from its identity")
        marker = completion_root / (hashlib.sha256(str(terminal).encode("utf-8")).hexdigest() + ".json")
        try:
            os.link(terminal, marker, follow_symlinks=False)
        except FileExistsError as error:
            raise LaunchError("outer completion marker already exists") from error
        except OSError as error:
            raise LaunchError("outer completion hardlink failed") from error
        # Link success is the final logical commit.  Durability sync is
        # best-effort: a sync error cannot revoke an already visible commit.
        try:
            marker_parent = os.open(
                completion_root,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(marker_parent)
            finally:
                os.close(marker_parent)
        except OSError:
            pass
        return logical_exit_code
    finally:
        try:
            os.close(terminal_fd)
        except OSError:
            pass


def _run_locked(
    arguments: argparse.Namespace,
    *,
    environment: dict[str, str],
    repository: Path,
    completion_root: Path,
) -> int:
    producer_arguments = list(arguments.producer_arguments)
    if producer_arguments[:1] == ["--"]:
        del producer_arguments[0]
    if "--" in producer_arguments:
        raise LaunchError("producer arguments contain an ambiguous separator")
    forbidden = {
        "--bootstrap",
        "--bootstrap-fd",
        "--runtime-seal",
        "--runtime-seal-fd",
        "--formal-binding-seal",
        "--create-runtime-seal",
    }
    if any(item in forbidden or any(item.startswith(f"{name}=") for name in forbidden) for item in producer_arguments):
        raise LaunchError("producer arguments contain a second custody source")
    if (
        arguments.create_runtime_seal is not None
        and arguments.create_runtime_seal != _prospective_runtime_seal(repository)
    ):
        raise LaunchError(
            "runtime-seal creation requires the sole canonical protocol-1.15 prospective path"
        )
    bootstrap_fd, bootstrap_payload, bootstrap_identity = _open_regular(
        arguments.bootstrap,
        label="producer bootstrap",
    )
    seal_fd: int | None = None
    seal_payload: bytes | None = None
    seal_identity: tuple[int, ...] | None = None
    seal_expected_nlink: int | None = None
    receipt_read_fd, receipt_write_fd = os.pipe()
    try:
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            _descriptor_path(bootstrap_fd),
            "--bootstrap-fd",
            str(bootstrap_fd),
            "--outer-receipt-fd",
            str(receipt_write_fd),
        ]
        pass_fds = [bootstrap_fd, receipt_write_fd]
        if arguments.create_runtime_seal is not None:
            if producer_arguments:
                raise LaunchError("runtime-seal creation takes no producer arguments")
            command.extend(
                [
                    "--create-runtime-seal",
                    str(arguments.create_runtime_seal),
                ]
            )
        else:
            assert arguments.runtime_seal is not None
            if not producer_arguments:
                raise LaunchError("runtime entry requires producer arguments")
            (
                seal_fd,
                seal_payload,
                seal_identity,
                seal_expected_nlink,
            ) = _open_typed_runtime_custody(
                arguments.runtime_seal,
                repository=repository,
                completion_root=completion_root,
            )
            command.extend(["--runtime-seal-fd", str(seal_fd)])
            command.extend(producer_arguments)
            pass_fds.append(seal_fd)
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            check=False,
            pass_fds=tuple(pass_fds),
        )
        os.close(receipt_write_fd)
        receipt_write_fd = -1
        receipt_payload = _read_receipt(receipt_read_fd)
        _verify_unchanged(
            bootstrap_fd,
            expected_payload=bootstrap_payload,
            expected_identity=bootstrap_identity,
            label="producer bootstrap",
        )
        if (
            seal_fd is not None
            and seal_payload is not None
            and seal_identity is not None
            and seal_expected_nlink is not None
        ):
            _verify_unchanged(
                seal_fd,
                expected_payload=seal_payload,
                expected_identity=seal_identity,
                label="runtime seal",
                expected_nlink=seal_expected_nlink,
            )
        if completed.returncode != 0:
            return completed.returncode
        read_only = arguments.create_runtime_seal is None and producer_arguments[:1] in (
            ["preformal-runtime"],
            ["--restore-eval-entry"],
        )
        if read_only:
            if receipt_payload:
                raise LaunchError("read-only runtime emitted a terminal receipt")
            return 0
        if not receipt_payload:
            raise LaunchError("top-level publisher emitted no terminal receipt")
        return _commit_outer_receipt(
            receipt_payload,
            repository=repository,
            completion_root=completion_root,
        )
    finally:
        if receipt_write_fd >= 0:
            _close_quietly(receipt_write_fd)
        _close_quietly(receipt_read_fd)
        if seal_fd is not None:
            _close_quietly(seal_fd)
        _close_quietly(bootstrap_fd)


def main(argv: list[str] | None = None) -> int:
    if {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    } != {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }:
        raise LaunchError("outer launcher requires exact CPython flags -I -S -B")
    _sanitize_module_search_path()
    environment = _environment()
    arguments = _parser().parse_args(argv)
    repository = _canonical_repository(arguments.bootstrap)
    lock_descriptor = _acquire_runtime_lock(repository)
    try:
        completion_root = _prepare_completion_root(repository)
        return _run_locked(
            arguments,
            environment=environment,
            repository=repository,
            completion_root=completion_root,
        )
    finally:
        try:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        finally:
            _close_quietly(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
