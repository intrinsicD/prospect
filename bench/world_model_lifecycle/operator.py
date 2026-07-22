"""Atomic custody-preserving operator entry points for WM-001 protocol 1.18.

Every public entry point in this module is reached through
``producer_bootstrap.py``.  Outputs are complete attempt directories: work is
performed in a hidden sibling, a terminal manifest is written last, the whole
package is reopened, live inputs and inherited descriptors are rechecked, and
the directory is atomically renamed with no-replace semantics.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .assurance import ASSURANCE, assurance_record

_TERMINAL_MANIFEST = "operator-attempt.json"
_ATTEMPT_SCHEMA = "prospect.wm001.operator-attempt.v1"
_EXECUTION_SCHEMA = "prospect.wm001.captured-audit-execution.v1"
_EXECUTION_FAILURE_EVIDENCE_SCHEMA = "prospect.wm001.captured-audit-execution-failure.v1"
_FAILURE_SCHEMA = "prospect.wm001.operator-execution-failure.v2"
_CLOSURE_REFERENCE_SCHEMA = "prospect.wm001.closure-reference.v1"
_FORMAL_AUDIT_CLAIM_SCHEMA = "prospect.wm001.formal-audit-claim.v1"
_FORMAL_AUDIT_CLAIM_FILE = "formal-audit-claim.json"
_FORMAL_INPUT_PREFLIGHT_FILE = "formal-input-preflight.json"
_MAX_CONTROL_BYTES = 64 << 20
_ATTEMPT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_FAILURE_CODE = re.compile(r"^[a-z_][a-z0-9_]{0,127}$")
_FAILURE_PHASE = {
    "binding": "binding",
    "audit": "audit_execution",
    "closure": "development_closure",
}
_FAILURE_MESSAGE_CHUNK_CHARACTERS = 4096


def _repository_root() -> Path:
    completed = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=Path.cwd(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("WM-001 operator requires an explicit Git worktree")
    candidate = Path(completed.stdout.strip())
    if (
        not candidate.is_absolute()
        or candidate.resolve(strict=True) != candidate
        or not (candidate / ".git").exists()
        or not (candidate / "bench" / "world_model_lifecycle" / "protocol.json").is_file()
    ):
        raise RuntimeError("WM-001 operator Git worktree is absent or aliased")
    return candidate


REPO = _repository_root()
DEVELOPMENT_RESULTS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results" / "development"
DEVELOPMENT_QUALIFICATION_PATH = DEVELOPMENT_RESULTS_ROOT / "qualification-v1.18.0"
OPERATOR_RESULTS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results" / "operator-v1.18"
BINDING_ATTEMPTS_ROOT = OPERATOR_RESULTS_ROOT / "bindings"
AUDIT_ATTEMPTS_ROOT = OPERATOR_RESULTS_ROOT / "audits"
CLOSURE_ATTEMPTS_ROOT = OPERATOR_RESULTS_ROOT / "closures"
OUTER_COMPLETIONS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results" / "outer-completions" / "v1.18"
FORMAL_BINDING_ATTEMPT_PATH = BINDING_ATTEMPTS_ROOT / "formal-binding-v1.18.0"
DEVELOPMENT_AUDIT_ATTEMPT_PATH = AUDIT_ATTEMPTS_ROOT / "development-audit-v1.18.0"
FORMAL_AUDIT_ATTEMPT_PATH = AUDIT_ATTEMPTS_ROOT / "formal-audit-v1.18.0"
FORMAL_AUDIT_CLAIM_MARKER = (
    REPO
    / "bench"
    / "world_model_lifecycle"
    / "results"
    / "formal"
    / "formal-audit-v1.18.0.json"
)
CLOSURE_ATTEMPT_PATH = CLOSURE_ATTEMPTS_ROOT / "development-closure-v1.18.0"


class OperatorError(RuntimeError):
    """An operator request violates the fixed WM-001 custody contract."""


class _FormalAuditRetired(OperatorError):
    """The sole protocol-1.18 formal audit claim has already been consumed."""


class _StoreOnce(argparse.Action):
    """Reject repeated custody-bearing options instead of taking the last."""

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


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise OperatorError(f"canonical JSON repeats key {key!r}")
        value[key] = item
    return value


def _reject_constant(raw: str) -> object:
    raise OperatorError(f"canonical JSON contains non-finite value {raw}")


def _canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperatorError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise OperatorError(f"{label} is not one canonical JSON object followed by LF")
    return cast(dict[str, object], value)


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


def _require_lexical_absolute(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.name in {"", ".", ".."}:
        raise OperatorError(f"{label} must be one lexical canonical absolute path")
    return path


def _reject_symlink_components(
    path: Path,
    *,
    include_leaf: bool,
    label: str,
) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:] if include_leaf else path.parts[1:-1]
    for part in parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise OperatorError(f"{label} path cannot be resolved") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise OperatorError(f"{label} path contains a symbolic-link component")


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    candidate = _require_lexical_absolute(path, label=label)
    _reject_symlink_components(candidate, include_leaf=True, label=label)
    try:
        metadata = os.stat(candidate, follow_symlinks=False)
    except OSError as error:
        raise OperatorError(f"{label} is not an accessible directory") from error
    if not stat.S_ISDIR(metadata.st_mode) or candidate.resolve(strict=True) != candidate:
        raise OperatorError(f"{label} must be one canonical directory")
    return candidate


def _canonical_existing_file(path: Path, *, label: str) -> Path:
    capture = _FileCapture.open(path, label=label, retain_payload=False)
    try:
        return capture.path
    finally:
        capture.close()


def _ensure_attempt_root(root: Path) -> Path:
    if not root.is_absolute() or root == REPO or not root.is_relative_to(REPO) or root.resolve(strict=False) != root:
        raise OperatorError("operator attempt root is outside the canonical repository")
    current = _canonical_existing_directory(REPO, label="canonical repository")
    for part in root.relative_to(REPO).parts:
        candidate = current / part
        try:
            os.mkdir(candidate, 0o755)
        except FileExistsError:
            pass
        else:
            _fsync_directory(current)
        current = _canonical_existing_directory(
            candidate,
            label="operator attempt namespace",
        )
    return _canonical_existing_directory(root, label="operator attempt root")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attempt_output(path: Path, *, root: Path, label: str) -> Path:
    canonical_root = _ensure_attempt_root(root)
    candidate = _require_lexical_absolute(path, label=label)
    if (
        candidate.parent != canonical_root
        or not _ATTEMPT_NAME.fullmatch(candidate.name)
        or candidate.name.startswith(".")
        or candidate.resolve(strict=False) != candidate
    ):
        raise OperatorError(f"{label} must be one new named direct child of {canonical_root}")
    staging_claim = canonical_root / f".{candidate.name}.staging"
    if os.path.lexists(candidate) or os.path.lexists(staging_claim):
        raise FileExistsError(f"refusing to replace {label}: {candidate}")
    return candidate


def outer_completion_marker(terminal_path: Path) -> Path:
    """Return the deterministic outer-completion hardlink for a terminal."""

    terminal = _require_lexical_absolute(
        terminal_path,
        label="outer completion terminal",
    )
    digest = hashlib.sha256(str(terminal).encode("utf-8")).hexdigest()
    return OUTER_COMPLETIONS_ROOT / f"{digest}.json"


def _sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(
            descriptor,
            min(1 << 20, size - offset),
            offset,
        )
        if not chunk:
            raise OperatorError("captured input ended while it was hashed")
        digest.update(chunk)
        offset += len(chunk)
    if offset != size:
        raise OperatorError("captured input byte count changed")
    return digest.hexdigest()


@dataclass
class _FileCapture:
    path: Path
    descriptor: int
    identity: tuple[int, ...]
    size: int
    sha256: str
    payload: bytes | None
    label: str
    expected_nlink: int

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        retain_payload: bool,
        payload_limit: int = _MAX_CONTROL_BYTES,
        expected_nlink: int = 1,
    ) -> _FileCapture:
        candidate = _require_lexical_absolute(path, label=label)
        _reject_symlink_components(candidate, include_leaf=True, label=label)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(candidate, flags)
            before = os.fstat(descriptor)
        except OSError as error:
            if "descriptor" in locals():
                os.close(descriptor)
            raise OperatorError(f"{label} cannot be opened") from error
        try:
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != expected_nlink:
                raise OperatorError(f"{label} must be one regular file with exactly {expected_nlink} link(s)")
            digest = _sha256_descriptor(descriptor, before.st_size)
            payload = (
                os.pread(descriptor, before.st_size + 1, 0)
                if retain_payload and before.st_size <= payload_limit
                else None
            )
            if retain_payload and payload is None:
                raise OperatorError(f"{label} exceeds its retained-payload limit")
            after = os.fstat(descriptor)
            if (
                _stat_identity(before) != _stat_identity(after)
                or (payload is not None and len(payload) != before.st_size)
                or candidate.resolve(strict=True) != candidate
            ):
                raise OperatorError(f"{label} changed while captured")
            return cls(
                path=candidate,
                descriptor=descriptor,
                identity=_stat_identity(before),
                size=before.st_size,
                sha256=digest,
                payload=payload,
                label=label,
                expected_nlink=expected_nlink,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def row(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": self.size,
            "sha256": self.sha256,
        }

    def recheck(self) -> None:
        try:
            before = os.fstat(self.descriptor)
            current_digest = _sha256_descriptor(self.descriptor, before.st_size)
            after = os.fstat(self.descriptor)
            path_metadata = os.stat(self.path, follow_symlinks=False)
            descriptor_path = next(
                (
                    Path(f"{prefix}{self.descriptor}")
                    for prefix in ("/proc/self/fd/", "/dev/fd/")
                    if os.path.exists(f"{prefix}{self.descriptor}")
                ),
                None,
            )
            same_file = descriptor_path is not None and os.path.samefile(self.path, descriptor_path)
        except OSError as error:
            raise OperatorError(f"{self.label} cannot be reopened") from error
        if (
            not same_file
            or _stat_identity(before) != self.identity
            or _stat_identity(after) != self.identity
            or _stat_identity(path_metadata) != self.identity
            or current_digest != self.sha256
        ):
            raise OperatorError(f"{self.label} changed before attempt publication")

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass


class _Captures:
    def __init__(self) -> None:
        self.rows: list[_FileCapture] = []

    def add(
        self,
        path: Path,
        *,
        label: str,
        retain_payload: bool = False,
        expected_nlink: int = 1,
    ) -> _FileCapture:
        capture = _FileCapture.open(
            path,
            label=label,
            retain_payload=retain_payload,
            expected_nlink=expected_nlink,
        )
        self.rows.append(capture)
        return capture

    def recheck(self) -> None:
        for capture in self.rows:
            capture.recheck()

    def identities(self) -> list[dict[str, object]]:
        return [capture.row() for capture in self.rows]

    def close(self) -> None:
        for capture in reversed(self.rows):
            capture.close()


def _atomic_write(path: Path, payload: bytes) -> None:
    from .artifact import atomic_write_exclusive

    atomic_write_exclusive(path, payload)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_json_bytes(value))


def _regular_attempt_files(
    root: Path,
    *,
    terminal_nlink: int = 1,
) -> tuple[Path, ...]:
    files: list[Path] = []
    for entry in os.scandir(root):
        candidate = root / entry.name
        metadata = entry.stat(follow_symlinks=False)
        expected_nlink = (
            2 if entry.name == _FORMAL_AUDIT_CLAIM_FILE else terminal_nlink if entry.name == _TERMINAL_MANIFEST else 1
        )
        if (
            entry.name.startswith(".")
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != expected_nlink
            or candidate.resolve(strict=True) != candidate
        ):
            raise OperatorError(f"operator attempt contains unsafe entry {entry.name!r}")
        files.append(candidate)
    return tuple(sorted(files, key=lambda path: path.name))


def _file_rows(
    root: Path,
    *,
    exclude: set[str],
    terminal_nlink: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _regular_attempt_files(
        root,
        terminal_nlink=terminal_nlink,
    ):
        if path.name in exclude:
            continue
        capture = _FileCapture.open(
            path,
            label=f"operator output {path.name}",
            retain_payload=False,
            expected_nlink=(2 if path.name == _FORMAL_AUDIT_CLAIM_FILE else 1),
        )
        try:
            rows.append(
                {
                    "path": path.name,
                    "bytes": capture.size,
                    "sha256": capture.sha256,
                }
            )
        finally:
            capture.close()
    return rows


def _failure_code(error_type: str) -> str:
    """Derive the sole diagnostic code accepted for an exception class."""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", error_type).lower()


def _failure_message_identity(error: BaseException) -> tuple[int, str]:
    """Hash the rendered message without a message-sized encoded copy."""

    message = str(error)
    digest = hashlib.sha256()
    observed_bytes = 0
    for offset in range(
        0,
        len(message),
        _FAILURE_MESSAGE_CHUNK_CHARACTERS,
    ):
        payload = message[offset : offset + _FAILURE_MESSAGE_CHUNK_CHARACTERS].encode(
            "utf-8",
            errors="backslashreplace",
        )
        observed_bytes += len(payload)
        digest.update(payload)
    return observed_bytes, digest.hexdigest()


def _failure_record(
    *,
    kind: str,
    lane: str | None,
    phase: str,
    error: BaseException,
) -> dict[str, object]:
    error_type = type(error).__name__
    expected_phase = _FAILURE_PHASE.get(kind)
    if expected_phase is None or phase != expected_phase or _ERROR_TYPE.fullmatch(error_type) is None:
        raise OperatorError("operator failure diagnostic has no canonical kind, phase, or error type")
    code = _failure_code(error_type)
    if _FAILURE_CODE.fullmatch(code) is None:
        raise OperatorError("operator failure diagnostic code is not bounded")
    message_bytes, message_sha256 = _failure_message_identity(error)
    return {
        "schema": _FAILURE_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.18.0",
        "kind": kind,
        "lane": lane,
        "phase": phase,
        "error_type": error_type,
        "failure_code": code,
        "error_message_bytes": message_bytes,
        "error_message_sha256": message_sha256,
        "passed": False,
    }


def _write_execution(
    staging: Path,
    *,
    prefix: str,
    execution: object,
) -> dict[str, object]:
    from .audit_runner import AuditExecution

    if not isinstance(execution, AuditExecution):
        raise OperatorError("auditor returned no authenticated AuditExecution")
    report, passed = _canonical_report(execution.stdout)
    if (
        dict(execution.report) != report
        or execution.returncode != (0 if passed else 1)
        or execution.runtime_manifest_sha256 != hashlib.sha256(execution.runtime_manifest).hexdigest()
        or execution.invocation_manifest_sha256 != hashlib.sha256(execution.invocation_manifest).hexdigest()
    ):
        raise OperatorError("captured audit execution identities are inconsistent")
    filenames = {
        "stdout_file": f"{prefix}.stdout.json",
        "stderr_file": f"{prefix}.stderr.log",
        "runtime_manifest_file": f"{prefix}.runtime.json",
        "invocation_manifest_file": f"{prefix}.invocation.json",
    }
    _atomic_write(staging / filenames["stdout_file"], execution.stdout)
    _atomic_write(staging / filenames["stderr_file"], execution.stderr)
    _atomic_write(
        staging / filenames["runtime_manifest_file"],
        execution.runtime_manifest,
    )
    _atomic_write(
        staging / filenames["invocation_manifest_file"],
        execution.invocation_manifest,
    )
    receipt: dict[str, object] = {
        "schema": _EXECUTION_SCHEMA,
        "returncode": execution.returncode,
        "passed": passed,
        "source_mode": execution.source_mode,
        "command": list(execution.command),
        **filenames,
        "stdout_bytes": len(execution.stdout),
        "stdout_sha256": hashlib.sha256(execution.stdout).hexdigest(),
        "stderr_bytes": len(execution.stderr),
        "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        "runtime_manifest_bytes": len(execution.runtime_manifest),
        "runtime_manifest_sha256": execution.runtime_manifest_sha256,
        "invocation_manifest_bytes": len(execution.invocation_manifest),
        "invocation_manifest_sha256": execution.invocation_manifest_sha256,
        "bootstrap_sha256": execution.bootstrap_sha256,
        "auditor_source_sha256": execution.auditor_source_sha256,
        "support_files": [
            {
                "path": row.relative_path,
                "bytes": row.bytes,
                "sha256": row.sha256,
            }
            for row in execution.support_files
        ],
    }
    receipt_file = f"{prefix}.execution.json"
    _write_json(staging / receipt_file, receipt)
    return {
        "receipt_file": receipt_file,
        "passed": passed,
        "stdout_file": filenames["stdout_file"],
        "runtime_manifest_file": filenames["runtime_manifest_file"],
        "invocation_manifest_file": filenames["invocation_manifest_file"],
    }


def _write_execution_failure(
    staging: Path,
    *,
    prefix: str,
    error: BaseException,
) -> str:
    from .audit_runner import AuditExecutionFailure

    if not isinstance(error, AuditExecutionFailure):
        raise OperatorError("partial auditor evidence is not an AuditExecutionFailure")
    filenames = {
        "stdout_file": f"{prefix}.partial.stdout",
        "stderr_file": f"{prefix}.partial.stderr",
        "runtime_manifest_file": f"{prefix}.partial.runtime.json",
        "invocation_manifest_file": f"{prefix}.partial.invocation.json",
    }
    payloads = {
        "stdout": error.stdout,
        "stderr": error.stderr,
        "runtime_manifest": error.runtime_manifest,
        "invocation_manifest": error.invocation_manifest,
    }
    for field, payload in payloads.items():
        _atomic_write(
            staging / filenames[f"{field}_file"],
            payload,
        )
    receipt: dict[str, object] = {
        "schema": _EXECUTION_FAILURE_EVIDENCE_SCHEMA,
        "phase": error.phase,
        "returncode": error.returncode,
        "source_mode": error.source_mode,
        "command": list(error.command),
        **filenames,
        **{f"{field}_bytes": len(payload) for field, payload in payloads.items()},
        **{f"{field}_sha256": hashlib.sha256(payload).hexdigest() for field, payload in payloads.items()},
        "bootstrap_sha256": error.bootstrap_sha256,
        "auditor_source_sha256": error.auditor_source_sha256,
        "support_files": [
            {
                "path": row.relative_path,
                "bytes": row.bytes,
                "sha256": row.sha256,
            }
            for row in error.support_files
        ],
    }
    filename = f"{prefix}.failure.json"
    _write_json(staging / filename, receipt)
    return filename


def _canonical_report(payload: bytes) -> tuple[dict[str, object], bool]:
    report = _canonical_object(payload, label="independent audit report")
    if type(report.get("passed")) is not bool:
        raise OperatorError("independent audit report has no boolean passed field")
    return report, cast(bool, report["passed"])


def _sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _safe_attempt_filename(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(".")
        or Path(value).name != value
        or value == _TERMINAL_MANIFEST
    ):
        raise OperatorError(f"{label} filename is unsafe")
    return value


def _attempt_file_payload(root: Path, filename: object, *, label: str) -> bytes:
    name = _safe_attempt_filename(filename, label=label)
    capture = _FileCapture.open(
        root / name,
        label=label,
        retain_payload=True,
        expected_nlink=(2 if name == _FORMAL_AUDIT_CLAIM_FILE else 1),
    )
    try:
        assert capture.payload is not None
        return capture.payload
    finally:
        capture.close()


def _referenced_payload(
    root: Path,
    value: Mapping[str, object],
    *,
    prefix: str,
    label: str,
) -> bytes:
    payload = _attempt_file_payload(
        root,
        value.get(f"{prefix}_file"),
        label=label,
    )
    if (
        value.get(f"{prefix}_bytes") != len(payload)
        or value.get(f"{prefix}_sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise OperatorError(f"{label} differs from its captured identity")
    return payload


def _verify_execution_receipt(
    root: Path,
    filename: object,
) -> dict[str, object]:
    payload = _attempt_file_payload(
        root,
        filename,
        label="captured audit execution receipt",
    )
    receipt = _canonical_object(
        payload,
        label="captured audit execution receipt",
    )
    expected = {
        "schema",
        "returncode",
        "passed",
        "source_mode",
        "command",
        "stdout_file",
        "stderr_file",
        "runtime_manifest_file",
        "invocation_manifest_file",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "runtime_manifest_bytes",
        "runtime_manifest_sha256",
        "invocation_manifest_bytes",
        "invocation_manifest_sha256",
        "bootstrap_sha256",
        "auditor_source_sha256",
        "support_files",
    }
    passed = receipt.get("passed")
    returncode = receipt.get("returncode")
    command = receipt.get("command")
    support_files = receipt.get("support_files")
    if (
        set(receipt) != expected
        or receipt.get("schema") != _EXECUTION_SCHEMA
        or type(passed) is not bool
        or type(returncode) is not int
        or returncode != (0 if passed else 1)
        or receipt.get("source_mode") != "descriptor"
        or not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) for argument in command)
        or not _sha256_string(receipt.get("bootstrap_sha256"))
        or not _sha256_string(receipt.get("auditor_source_sha256"))
        or not isinstance(support_files, list)
    ):
        raise OperatorError("captured audit execution receipt is malformed")
    support_paths: list[str] = []
    for row in support_files:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or Path(cast(str, row["path"])).is_absolute()
            or ".." in Path(cast(str, row["path"])).parts
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _sha256_string(row.get("sha256"))
        ):
            raise OperatorError("captured audit support-file identity is malformed")
        support_paths.append(cast(str, row["path"]))
    if support_paths != sorted(support_paths) or len(support_paths) != len(set(support_paths)):
        raise OperatorError("captured audit support-file identities are unstable")
    stdout = _referenced_payload(
        root,
        receipt,
        prefix="stdout",
        label="captured audit stdout",
    )
    stderr = _referenced_payload(
        root,
        receipt,
        prefix="stderr",
        label="captured audit stderr",
    )
    runtime = _referenced_payload(
        root,
        receipt,
        prefix="runtime_manifest",
        label="captured audit runtime manifest",
    )
    invocation = _referenced_payload(
        root,
        receipt,
        prefix="invocation_manifest",
        label="captured audit invocation manifest",
    )
    _, report_passed = _canonical_report(stdout)
    _canonical_object(runtime, label="captured audit runtime manifest")
    _canonical_object(invocation, label="captured audit invocation manifest")
    if report_passed is not passed:
        raise OperatorError("captured audit receipt status differs from its stdout")
    return {
        "receipt": receipt,
        "stdout": stdout,
        "stderr": stderr,
        "runtime": runtime,
        "invocation": invocation,
        "passed": passed,
    }


def _verify_execution_failure_receipt(
    root: Path,
    filename: object,
) -> None:
    payload = _attempt_file_payload(
        root,
        filename,
        label="captured audit execution failure",
    )
    receipt = _canonical_object(
        payload,
        label="captured audit execution failure",
    )
    expected = {
        "schema",
        "phase",
        "returncode",
        "source_mode",
        "command",
        "stdout_file",
        "stderr_file",
        "runtime_manifest_file",
        "invocation_manifest_file",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "runtime_manifest_bytes",
        "runtime_manifest_sha256",
        "invocation_manifest_bytes",
        "invocation_manifest_sha256",
        "bootstrap_sha256",
        "auditor_source_sha256",
        "support_files",
    }
    command = receipt.get("command")
    support_files = receipt.get("support_files")
    if (
        set(receipt) != expected
        or receipt.get("schema") != _EXECUTION_FAILURE_EVIDENCE_SCHEMA
        or not isinstance(receipt.get("phase"), str)
        or (receipt.get("returncode") is not None and type(receipt.get("returncode")) is not int)
        or receipt.get("source_mode") != "descriptor"
        or not isinstance(command, list)
        or any(not isinstance(argument, str) for argument in command)
        or not _sha256_string(receipt.get("bootstrap_sha256"))
        or not _sha256_string(receipt.get("auditor_source_sha256"))
        or not isinstance(support_files, list)
    ):
        raise OperatorError("captured audit execution failure receipt is malformed")
    support_paths: list[str] = []
    for row in support_files:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or Path(cast(str, row["path"])).is_absolute()
            or ".." in Path(cast(str, row["path"])).parts
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _sha256_string(row.get("sha256"))
        ):
            raise OperatorError("captured audit failure support identity is malformed")
        support_paths.append(cast(str, row["path"]))
    if support_paths != sorted(support_paths) or len(support_paths) != len(set(support_paths)):
        raise OperatorError("captured audit failure support identities are unstable")
    _referenced_payload(
        root,
        receipt,
        prefix="stdout",
        label="captured failed-audit stdout",
    )
    _referenced_payload(
        root,
        receipt,
        prefix="stderr",
        label="captured failed-audit stderr",
    )
    runtime = _referenced_payload(
        root,
        receipt,
        prefix="runtime_manifest",
        label="captured failed-audit runtime manifest",
    )
    invocation = _referenced_payload(
        root,
        receipt,
        prefix="invocation_manifest",
        label="captured failed-audit invocation manifest",
    )
    _canonical_object(
        runtime,
        label="captured failed-audit runtime manifest",
    )
    _canonical_object(
        invocation,
        label="captured failed-audit invocation manifest",
    )


def _verify_reproduction_receipt(
    root: Path,
    filename: object,
    *,
    audit_payload: bytes,
    replay: Mapping[str, object],
) -> dict[str, object]:
    payload = _attempt_file_payload(
        root,
        filename,
        label="development audit reproduction",
    )
    receipt = _canonical_object(
        payload,
        label="development audit reproduction",
    )
    expected = {
        "schema",
        "experiment_id",
        "protocol_version",
        "supplied_audit_sha256",
        "reproduced_audit_sha256",
        "byte_identical",
        "returncode",
        "source_mode",
        "stdout_bytes",
        "stderr_file",
        "stderr_bytes",
        "stderr_sha256",
        "runtime_manifest_file",
        "runtime_manifest_bytes",
        "runtime_manifest_sha256",
        "invocation_manifest_file",
        "invocation_manifest_bytes",
        "invocation_manifest_sha256",
        "bootstrap_sha256",
        "runner_source_sha256",
        "auditor_source_sha256",
        "support_files",
        "passed",
    }
    audit_sha256 = hashlib.sha256(audit_payload).hexdigest()
    if (
        set(receipt) != expected
        or receipt.get("schema") != "prospect.wm001.audit-reproduction.v2"
        or receipt.get("experiment_id") != "WM-001"
        or receipt.get("protocol_version") != "1.18.0"
        or receipt.get("supplied_audit_sha256") != audit_sha256
        or receipt.get("reproduced_audit_sha256") != audit_sha256
        or receipt.get("byte_identical") is not True
        or receipt.get("returncode") != 0
        or receipt.get("source_mode") != "descriptor"
        or receipt.get("stdout_bytes") != len(audit_payload)
        or receipt.get("passed") is not True
        or not _sha256_string(receipt.get("bootstrap_sha256"))
        or not _sha256_string(receipt.get("runner_source_sha256"))
        or not _sha256_string(receipt.get("auditor_source_sha256"))
        or receipt.get("support_files") != cast(dict[str, object], replay["receipt"]).get("support_files")
    ):
        raise OperatorError("development audit reproduction is malformed")
    stderr = _referenced_payload(
        root,
        receipt,
        prefix="stderr",
        label="development reproduction stderr",
    )
    runtime = _referenced_payload(
        root,
        receipt,
        prefix="runtime_manifest",
        label="development reproduction runtime manifest",
    )
    invocation = _referenced_payload(
        root,
        receipt,
        prefix="invocation_manifest",
        label="development reproduction invocation manifest",
    )
    if (
        stderr != replay["stderr"]
        or runtime != replay["runtime"]
        or invocation != replay["invocation"]
        or receipt.get("bootstrap_sha256") != cast(dict[str, object], replay["receipt"]).get("bootstrap_sha256")
        or receipt.get("auditor_source_sha256")
        != cast(dict[str, object], replay["receipt"]).get("auditor_source_sha256")
    ):
        raise OperatorError("development audit reproduction differs from replay execution")
    return receipt


def _verify_audit_primary(
    root: Path,
    *,
    lane: object,
    status: str,
    primary: Mapping[str, object],
    inputs: Sequence[Mapping[str, object]],
) -> None:
    expected = {
        "producer_root",
        "audit_file",
        "executions",
        "execution_failures",
        "reproduction_file",
        "reproduction_runtime_file",
        "claim_file",
    }
    if status == "failure":
        expected.add("execution_failure_file")
    producer_root = primary.get("producer_root")
    execution_files = primary.get("executions")
    failure_files = primary.get("execution_failures")
    audit_file = primary.get("audit_file")
    if (
        set(primary) != expected
        or not isinstance(producer_root, str)
        or not Path(producer_root).is_absolute()
        or Path(os.path.abspath(producer_root)) != Path(producer_root)
        or not isinstance(execution_files, list)
        or not isinstance(failure_files, list)
    ):
        raise OperatorError("terminal audit primary evidence is malformed")
    if lane == "development" and Path(producer_root) != DEVELOPMENT_QUALIFICATION_PATH:
        raise OperatorError("development audit does not reference the canonical development qualification")
    maximum = 2 if lane == "development" else 1
    required = maximum if status in {"accepted", "rejected"} else None
    if (
        len(execution_files) > maximum
        or len(execution_files) + len(failure_files) > maximum
        or (required is not None and len(execution_files) != required)
        or execution_files
        != [f"audit-execution-{index:02d}.execution.json" for index in range(1, len(execution_files) + 1)]
    ):
        raise OperatorError("terminal audit execution set is incomplete")
    expected_failure_files = [
        f"audit-execution-{index:02d}.failure.json"
        for index in range(
            len(execution_files) + 1,
            len(execution_files) + len(failure_files) + 1,
        )
    ]
    if failure_files != expected_failure_files or (status != "failure" and failure_files):
        raise OperatorError("terminal audit partial-failure set is malformed")
    for failure_file in failure_files:
        _verify_execution_failure_receipt(root, failure_file)
    executions = [_verify_execution_receipt(root, filename) for filename in execution_files]
    if audit_file is None:
        if executions or status != "failure":
            raise OperatorError("terminal audit omits completed stdout evidence")
        audit_payload: bytes | None = None
    else:
        if audit_file != "independent-audit.json" or not executions:
            raise OperatorError("terminal audit report reference is malformed")
        audit_payload = _attempt_file_payload(
            root,
            audit_file,
            label="independent audit report",
        )
        if audit_payload != executions[0]["stdout"]:
            raise OperatorError("independent audit differs from first captured execution")
    if status in {"accepted", "rejected"}:
        assert executions
        if (status == "accepted") is not executions[0]["passed"]:
            raise OperatorError("terminal audit status differs from its report")
        if lane == "development" and (
            executions[1]["stdout"] != executions[0]["stdout"]
            or executions[1]["runtime"] != executions[0]["runtime"]
            or executions[1]["invocation"] != executions[0]["invocation"]
            or executions[1]["passed"] is not executions[0]["passed"]
        ):
            raise OperatorError("development audit replay is not byte-identical")
    reproduction_file = primary.get("reproduction_file")
    reproduction_runtime = primary.get("reproduction_runtime_file")
    if status == "accepted" and lane == "development":
        assert audit_payload is not None
        receipt = _verify_reproduction_receipt(
            root,
            reproduction_file,
            audit_payload=audit_payload,
            replay=executions[1],
        )
        if receipt.get("runtime_manifest_file") != reproduction_runtime:
            raise OperatorError("development reproduction runtime reference changed")
    elif reproduction_file is not None or reproduction_runtime is not None:
        if status != "failure" or audit_payload is None or len(executions) != 2:
            raise OperatorError("nonqualifying audit has reproduction evidence")
        receipt = _verify_reproduction_receipt(
            root,
            reproduction_file,
            audit_payload=audit_payload,
            replay=executions[1],
        )
        if receipt.get("runtime_manifest_file") != reproduction_runtime:
            raise OperatorError("failure reproduction runtime reference changed")
    if lane == "formal":
        _verify_formal_audit_claim(
            root,
            primary=primary,
            inputs=inputs,
        )
    elif primary.get("claim_file") is not None:
        raise OperatorError("development audit contains a formal claim")


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Use Linux renameat2 so attempt publication can never replace a path."""

    if source.parent != destination.parent:
        raise OperatorError("attempt staging and destination are on different parents")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OperatorError("platform has no atomic no-replace directory rename")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"refusing to replace operator attempt: {destination}")
        raise OperatorError(f"atomic no-replace attempt publication failed with errno {error_number}")
    parent_descriptor = os.open(
        destination.parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _allowed_statuses(kind: str) -> set[str]:
    return {
        "binding": {"accepted", "failure"},
        "audit": {"accepted", "rejected", "failure"},
        "closure": {"accepted", "failure"},
    }[kind]


def verify_binding_authorization_inputs(
    binding_path: Path,
    inputs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Reopen the exact live evidence authorized to create one binding."""

    from . import binding as binding_module
    from .preformal import (
        PREFORMAL_REPORT_NAME,
        PREFORMAL_REPORT_PATH,
    )
    from .verify import (
        _recorded_development_closure_identity,
        verify_binding,
    )

    binding = verify_binding(binding_path)
    if not inputs:
        raise OperatorError("accepted binding has no preformal authorization inputs")
    report_value = inputs[0].get("path")
    if not isinstance(report_value, str):
        raise OperatorError("binding preformal report input is malformed")
    report_path = _canonical_existing_file(
        Path(report_value),
        label="binding preformal report",
    )
    if report_path != PREFORMAL_REPORT_PATH:
        raise OperatorError("binding does not reference the canonical preformal report")
    report = binding_module.verify_canonical_machine_test_report(report_path)
    log_rows = binding_module.preformal_log_rows(report_path, report)
    log_paths: list[Path] = []
    for row in log_rows:
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name or name.startswith("."):
            raise OperatorError("binding preformal log reference is malformed")
        log_paths.append(report_path.with_name(name))

    closure_path = binding_module.DEVELOPMENT_CLOSURE_PATH
    closure, recorded_development, _ = (
        _recorded_development_closure_identity(closure_path)
    )
    closure_attempt = _verify_published_operator_attempt(
        CLOSURE_ATTEMPT_PATH,
        verify_live_closure=False,
    )
    closure_primary = closure_attempt.get("primary")
    if (
        closure_attempt.get("kind") != "closure"
        or closure_attempt.get("lane") != "development"
        or closure_attempt.get("status") != "accepted"
        or not isinstance(closure_primary, Mapping)
        or closure_primary.get("closure_reference_file") != "closure-reference.json"
    ):
        raise OperatorError("binding does not consume the accepted canonical closure attempt")
    closure_terminal = CLOSURE_ATTEMPT_PATH / _TERMINAL_MANIFEST
    closure_completion = outer_completion_marker(closure_terminal)
    completion_identity = verify_outer_completion(closure_terminal)

    captures = _Captures()
    try:
        report_capture = captures.add(
            report_path,
            label="binding preformal report",
        )
        log_captures = [
            captures.add(
                path,
                label=f"binding preformal log {index}",
            )
            for index, path in enumerate(log_paths)
        ]
        closure_capture = captures.add(
            closure_path,
            label="binding development closure",
            retain_payload=True,
        )
        terminal_capture = captures.add(
            closure_terminal,
            label="binding closure attempt terminal",
            retain_payload=True,
            expected_nlink=2,
        )
        completion_capture = captures.add(
            closure_completion,
            label="binding closure outer completion",
            retain_payload=True,
            expected_nlink=2,
        )
        try:
            same_completion = os.path.samefile(
                closure_terminal,
                closure_completion,
            )
        except OSError as error:
            raise OperatorError("binding closure completion cannot be compared") from error
        source = binding.get("source")
        development = binding.get("development_qualification")
        expected_recorded_development = dict(recorded_development)
        if isinstance(development, Mapping) and "closure_file" in development:
            expected_recorded_development["closure_file"] = development.get(
                "closure_file"
            )
        expected_log_rows = [
            {
                "path": path.name,
                "bytes": capture.size,
                "sha256": capture.sha256,
            }
            for path, capture in zip(log_paths, log_captures, strict=True)
        ]
        if (
            list(inputs) != captures.identities()
            or not isinstance(source, Mapping)
            or source.get("test_report_file") != PREFORMAL_REPORT_NAME
            or source.get("test_report_bytes") != report_capture.size
            or source.get("test_report_sha256") != report_capture.sha256
            or source.get("test_log_files") != expected_log_rows
            or log_rows != expected_log_rows
            or not isinstance(development, Mapping)
            or development.get("closure_bytes") != closure_capture.size
            or development.get("closure_sha256") != closure_capture.sha256
            or dict(development) != expected_recorded_development
            or not isinstance(closure, Mapping)
            or closure.get("engineering_verified") is not True
            or closure.get("audit_reproduced") is not True
            or closure.get("performance_values_bound") is not False
            or terminal_capture.payload != completion_capture.payload
            or not same_completion
            or completion_identity.get("terminal_sha256") != terminal_capture.sha256
            or completion_identity.get("marker_path") != str(closure_completion)
        ):
            raise OperatorError("binding authorization inputs differ from exact live evidence")
        captures.recheck()
        return captures.identities()
    finally:
        captures.close()


def verify_closure_authorization_inputs(
    inputs: Sequence[Mapping[str, object]],
    *,
    reference: Mapping[str, object],
) -> list[dict[str, object]]:
    """Reconstruct the exact producer and canonical audit closure inputs."""

    producer_value = reference.get("producer_root")
    if not isinstance(producer_value, str):
        raise OperatorError("closure authorization has no producer root")
    if Path(producer_value) != DEVELOPMENT_QUALIFICATION_PATH:
        raise OperatorError("closure authorization does not reference the canonical development qualification")
    producer, _ = _ProducerCustody.capture(
        Path(producer_value),
        lane="development",
    )
    audit_captures = _Captures()
    try:
        audit_manifest = verify_operator_attempt(DEVELOPMENT_AUDIT_ATTEMPT_PATH)
        audit_primary = audit_manifest.get("primary")
        if (
            audit_manifest.get("kind") != "audit"
            or audit_manifest.get("lane") != "development"
            or audit_manifest.get("status") != "accepted"
            or not isinstance(audit_primary, Mapping)
            or audit_primary.get("producer_root") != str(producer.root)
            or audit_primary.get("audit_file") != "independent-audit.json"
            or audit_primary.get("reproduction_file") != "audit-reproduction.json"
        ):
            raise OperatorError("closure does not consume the accepted canonical development audit")
        for path in _regular_attempt_files(
            DEVELOPMENT_AUDIT_ATTEMPT_PATH,
            terminal_nlink=2,
        ):
            audit_captures.add(
                path,
                label=f"closure development audit {path.name}",
                expected_nlink=(2 if path.name == _TERMINAL_MANIFEST else 1),
            )
        audit_terminal = DEVELOPMENT_AUDIT_ATTEMPT_PATH / _TERMINAL_MANIFEST
        audit_completion = outer_completion_marker(audit_terminal)
        audit_captures.add(
            audit_completion,
            label="closure development audit outer completion",
            expected_nlink=2,
        )
        completion_identity = verify_outer_completion(audit_terminal)
        terminal_row = next(row for row in audit_captures.identities() if row["path"] == str(audit_terminal))
        expected = [
            *producer.input_rows(),
            *audit_captures.identities(),
        ]
        if (
            list(inputs) != expected
            or reference.get("audit_attempt") != str(DEVELOPMENT_AUDIT_ATTEMPT_PATH)
            or reference.get("audit_attempt_manifest_sha256") != terminal_row["sha256"]
            or completion_identity.get("terminal_sha256") != terminal_row["sha256"]
            or completion_identity.get("marker_path") != str(audit_completion)
        ):
            raise OperatorError("closure authorization inputs differ from exact live producer/audit evidence")
        producer.recheck()
        audit_captures.recheck()
        return expected
    finally:
        audit_captures.close()
        producer.close()


def _verify_attempt_directory(
    path: Path,
    *,
    terminal_nlink: int = 1,
    verify_live_closure: bool = True,
) -> dict[str, object]:
    root = _canonical_existing_directory(path, label="operator attempt")
    manifest_path = root / _TERMINAL_MANIFEST
    manifest_capture = _FileCapture.open(
        manifest_path,
        label="operator terminal manifest",
        retain_payload=True,
        expected_nlink=terminal_nlink,
    )
    try:
        assert manifest_capture.payload is not None
        manifest = _canonical_object(
            manifest_capture.payload,
            label="operator terminal manifest",
        )
    finally:
        manifest_capture.close()
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
    kind = manifest.get("kind")
    lane = manifest.get("lane")
    status = manifest.get("status")
    inputs = manifest.get("inputs")
    rows = manifest.get("files")
    if (
        set(manifest) != expected_fields
        or manifest.get("schema") != _ATTEMPT_SCHEMA
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("protocol_version") != "1.18.0"
        or manifest.get("assurance") != ASSURANCE
        or kind not in {"binding", "audit", "closure"}
        or (
            (kind == "binding" and lane is not None)
            or (kind == "audit" and lane not in {"development", "formal"})
            or (kind == "closure" and lane != "development")
        )
        or not isinstance(status, str)
        or status not in _allowed_statuses(kind)
        or not isinstance(inputs, list)
        or not isinstance(manifest.get("primary"), dict)
        or not isinstance(rows, list)
        or manifest.get("file_count") != len(rows)
        or manifest.get("manifest_excludes") != [_TERMINAL_MANIFEST]
        or any(
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or Path(cast(str, row["path"])).name != row["path"]
            or cast(str, row["path"]).startswith(".")
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _sha256_string(row.get("sha256"))
            for row in rows
        )
    ):
        raise OperatorError("operator terminal manifest is malformed")
    if rows != sorted(
        rows,
        key=lambda row: cast(str, cast(dict[str, object], row)["path"]),
    ) or len({cast(str, row["path"]) for row in rows}) != len(rows):
        raise OperatorError("operator terminal file identities are not canonical")
    row_names = {cast(str, row["path"]) for row in rows}
    if (_FORMAL_AUDIT_CLAIM_FILE in row_names) is not (kind == "audit" and lane == "formal"):
        raise OperatorError("formal audit claim appears in the wrong namespace")
    input_paths: list[str] = []
    for row in inputs:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not Path(cast(str, row["path"])).is_absolute()
            or Path(os.path.abspath(cast(str, row["path"]))) != Path(cast(str, row["path"]))
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _sha256_string(row.get("sha256"))
        ):
            raise OperatorError("operator input identity is malformed")
        input_paths.append(cast(str, row["path"]))
    if len(input_paths) != len(set(input_paths)):
        raise OperatorError("operator input identities contain duplicates")
    error_value = manifest.get("error")
    if (status == "failure" and not isinstance(error_value, dict)) or (status != "failure" and error_value is not None):
        raise OperatorError("operator terminal error status is inconsistent")
    actual_rows = _file_rows(
        root,
        exclude={_TERMINAL_MANIFEST},
        terminal_nlink=terminal_nlink,
    )
    if rows != actual_rows:
        raise OperatorError("operator attempt files differ from the terminal manifest")
    primary = cast(dict[str, object], manifest["primary"])
    if status == "failure":
        failure_file = primary.get("execution_failure_file")
        if failure_file != "execution-failure.json" or not (root / "execution-failure.json").is_file():
            raise OperatorError("failed operator attempt has no canonical failure record")
        failure = _canonical_object(
            _attempt_file_payload(
                root,
                "execution-failure.json",
                label="operator execution failure",
            ),
            label="operator execution failure",
        )
        if (
            set(failure)
            != {
                "schema",
                "experiment_id",
                "protocol_version",
                "kind",
                "lane",
                "phase",
                "error_type",
                "failure_code",
                "error_message_bytes",
                "error_message_sha256",
                "passed",
            }
            or failure.get("schema") != _FAILURE_SCHEMA
            or failure.get("experiment_id") != "WM-001"
            or failure.get("protocol_version") != "1.18.0"
            or failure.get("kind") != kind
            or failure.get("lane") != lane
            or failure.get("phase") != _FAILURE_PHASE[cast(str, kind)]
            or not isinstance(failure.get("error_type"), str)
            or _ERROR_TYPE.fullmatch(cast(str, failure["error_type"])) is None
            or not isinstance(failure.get("failure_code"), str)
            or _FAILURE_CODE.fullmatch(cast(str, failure["failure_code"])) is None
            or failure.get("failure_code") != _failure_code(cast(str, failure["error_type"]))
            or type(failure.get("error_message_bytes")) is not int
            or cast(int, failure["error_message_bytes"]) < 0
            or not _sha256_string(failure.get("error_message_sha256"))
            or failure.get("passed") is not False
            or error_value
            != {
                "failure_code": failure["failure_code"],
                "error_type": failure["error_type"],
            }
        ):
            raise OperatorError("operator execution failure record is malformed")
        if kind == "audit":
            _verify_audit_primary(
                root,
                lane=lane,
                status=status,
                primary=primary,
                inputs=cast(list[dict[str, object]], inputs),
            )
        elif set(primary) != {"execution_failure_file"}:
            raise OperatorError("failed operator attempt has unexpected primary evidence")
    elif kind == "binding":
        if set(primary) != {"binding_file"}:
            raise OperatorError("accepted binding primary evidence is malformed")
        binding_file = primary.get("binding_file")
        if binding_file != "formal-binding.json":
            raise OperatorError("accepted binding attempt has no formal binding")
        verify_binding_authorization_inputs(
            root / "formal-binding.json",
            cast(list[dict[str, object]], inputs),
        )
        from .artifact_audit import preflight_formal_input_package

        expected_preflight = preflight_formal_input_package(
            root / "formal-binding.json"
        )
        observed_preflight = _canonical_object(
            _attempt_file_payload(
                root,
                _FORMAL_INPUT_PREFLIGHT_FILE,
                label="formal input preflight receipt",
            ),
            label="formal input preflight receipt",
        )
        if observed_preflight != expected_preflight:
            raise OperatorError(
                "accepted binding attempt has no exact independent "
                "formal-input preflight"
            )
    elif kind == "audit":
        _verify_audit_primary(
            root,
            lane=lane,
            status=status,
            primary=primary,
            inputs=cast(list[dict[str, object]], inputs),
        )
    elif kind == "closure":
        if set(primary) != {"closure_reference_file"}:
            raise OperatorError("accepted closure primary evidence is malformed")
        reference_file = primary.get("closure_reference_file")
        if reference_file != "closure-reference.json":
            raise OperatorError("accepted closure attempt has no closure reference")
        reference = _canonical_object(
            _attempt_file_payload(
                root,
                "closure-reference.json",
                label="closure reference",
            ),
            label="closure reference",
        )
        if (
            set(reference)
            != {
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
            or reference.get("schema") != _CLOSURE_REFERENCE_SCHEMA
            or reference.get("experiment_id") != "WM-001"
            or reference.get("protocol_version") != "1.18.0"
            or not _sha256_string(reference.get("closure_sha256"))
            or not _sha256_string(reference.get("audit_attempt_manifest_sha256"))
            or reference.get("fresh_reopen_file")
            != "fresh-runtime-reopen.json"
            or not _sha256_string(reference.get("fresh_reopen_sha256"))
        ):
            raise OperatorError("closure reference is malformed")
        if verify_live_closure:
            verify_closure_authorization_inputs(
                cast(list[dict[str, object]], inputs),
                reference=reference,
            )
        from .binding import DEVELOPMENT_CLOSURE_PATH

        closure_marker = reference.get("closure_marker")
        if not isinstance(closure_marker, str) or Path(closure_marker) != DEVELOPMENT_CLOSURE_PATH:
            raise OperatorError("closure reference marker path is malformed")
        closure_path = Path(closure_marker)
        if verify_live_closure:
            from .binding import verify_development_closure

            closure = verify_development_closure(closure_path)
        else:
            from .verify import _recorded_development_closure_identity

            closure, _, _ = _recorded_development_closure_identity(
                closure_path,
            )
        fresh_reopen_payload = _attempt_file_payload(
            root,
            "fresh-runtime-reopen.json",
            label="fresh runtime closure-reopen report",
        )
        fresh_reopen = _canonical_object(
            fresh_reopen_payload,
            label="fresh runtime closure-reopen report",
        )
        from .preformal import validate_fresh_closure_reopen_report

        validate_fresh_closure_reopen_report(
            fresh_reopen,
            development_closure=closure_path,
        )
        marker_payload = _FileCapture.open(
            closure_path,
            label="canonical development closure",
            retain_payload=True,
        )
        try:
            assert marker_payload.payload is not None
            closure_sha256 = hashlib.sha256(marker_payload.payload).hexdigest()
        finally:
            marker_payload.close()
        audit_attempt_value = reference.get("audit_attempt")
        if not isinstance(audit_attempt_value, str):
            raise OperatorError("closure reference audit path is malformed")
        audit_attempt_path = Path(audit_attempt_value)
        if audit_attempt_path != DEVELOPMENT_AUDIT_ATTEMPT_PATH:
            raise OperatorError("closure reference does not bind the canonical development audit")
        audit_manifest = verify_operator_attempt(audit_attempt_path)
        audit_terminal_capture = _FileCapture.open(
            audit_attempt_path / _TERMINAL_MANIFEST,
            label="referenced development audit terminal manifest",
            retain_payload=True,
            expected_nlink=2,
        )
        try:
            assert audit_terminal_capture.payload is not None
            audit_manifest_payload = audit_terminal_capture.payload
        finally:
            audit_terminal_capture.close()
        if (
            closure_sha256 != reference.get("closure_sha256")
            or hashlib.sha256(fresh_reopen_payload).hexdigest()
            != reference.get("fresh_reopen_sha256")
            or closure.get("qualification_archive") != reference.get("qualification_archive")
            or closure.get("producer_root") != reference.get("producer_root")
            or audit_manifest.get("lane") != "development"
            or audit_manifest.get("status") != "accepted"
            or cast(dict[str, object], audit_manifest["primary"]).get("producer_root") != reference.get("producer_root")
            or hashlib.sha256(audit_manifest_payload).hexdigest() != reference.get("audit_attempt_manifest_sha256")
        ):
            raise OperatorError("closure reference differs from canonical evidence")
    return manifest


def verify_outer_completion(terminal_path: Path) -> dict[str, object]:
    """Verify the outer launcher's same-inode logical commit marker."""

    terminal = _require_lexical_absolute(
        terminal_path,
        label="outer-completed terminal",
    )
    marker = outer_completion_marker(terminal)
    terminal_capture = _FileCapture.open(
        terminal,
        label="outer-completed terminal",
        retain_payload=True,
        expected_nlink=2,
    )
    marker_capture = _FileCapture.open(
        marker,
        label="outer completion marker",
        retain_payload=True,
        expected_nlink=2,
    )
    try:
        assert terminal_capture.payload is not None
        if marker_capture.payload != terminal_capture.payload or not os.path.samefile(terminal, marker):
            raise OperatorError("outer completion marker is not the terminal inode")
        return {
            "terminal_path": str(terminal),
            "terminal_bytes": terminal_capture.size,
            "terminal_sha256": terminal_capture.sha256,
            "marker_path": str(marker),
        }
    finally:
        marker_capture.close()
        terminal_capture.close()


def _verify_published_operator_attempt(
    path: Path,
    *,
    verify_live_closure: bool,
) -> dict[str, object]:
    """Reopen one canonical attempt under an explicit closure evidence role."""

    root = _canonical_existing_directory(path, label="published operator attempt")
    manifest = _verify_attempt_directory(
        root,
        terminal_nlink=2,
        verify_live_closure=verify_live_closure,
    )
    expected_root = {
        "binding": BINDING_ATTEMPTS_ROOT,
        "audit": AUDIT_ATTEMPTS_ROOT,
        "closure": CLOSURE_ATTEMPTS_ROOT,
    }[cast(str, manifest["kind"])]
    if root.parent != _ensure_attempt_root(expected_root):
        raise OperatorError("operator attempt is outside its kind-specific namespace")
    expected_attempt = {
        "binding": FORMAL_BINDING_ATTEMPT_PATH,
        "closure": CLOSURE_ATTEMPT_PATH,
        "audit": (DEVELOPMENT_AUDIT_ATTEMPT_PATH if manifest["lane"] == "development" else FORMAL_AUDIT_ATTEMPT_PATH),
    }[cast(str, manifest["kind"])]
    if root != expected_attempt:
        raise OperatorError("published operator attempt is not its canonical protocol-1.18 path")
    verify_outer_completion(root / _TERMINAL_MANIFEST)
    return manifest


def verify_operator_attempt(path: Path) -> dict[str, object]:
    """Strictly reopen one published attempt, including live closure evidence."""

    return _verify_published_operator_attempt(
        path,
        verify_live_closure=True,
    )


def inspect_unfinalized_operator_attempt(path: Path) -> dict[str, object]:
    """Authenticate a published precommit attempt for rejection forensics.

    This path is never accepted/citeable evidence.  It exists only so a
    downstream adjudicator can preserve a child-published attempt when the
    outer launcher never committed its same-inode completion marker.
    """

    root = _canonical_existing_directory(
        path,
        label="unfinalized operator attempt",
    )
    manifest = _verify_attempt_directory(root, terminal_nlink=1)
    expected_root = {
        "binding": BINDING_ATTEMPTS_ROOT,
        "audit": AUDIT_ATTEMPTS_ROOT,
        "closure": CLOSURE_ATTEMPTS_ROOT,
    }[cast(str, manifest["kind"])]
    if root.parent != _ensure_attempt_root(expected_root):
        raise OperatorError("unfinalized operator attempt is outside its namespace")
    expected_attempt = {
        "binding": FORMAL_BINDING_ATTEMPT_PATH,
        "closure": CLOSURE_ATTEMPT_PATH,
        "audit": (DEVELOPMENT_AUDIT_ATTEMPT_PATH if manifest["lane"] == "development" else FORMAL_AUDIT_ATTEMPT_PATH),
    }[cast(str, manifest["kind"])]
    if root != expected_attempt:
        raise OperatorError("unfinalized operator attempt is not its canonical protocol-1.18 path")
    terminal = root / _TERMINAL_MANIFEST
    marker = outer_completion_marker(terminal)
    if os.path.lexists(marker):
        raise OperatorError("operator attempt has a completion marker and is not unfinalized")
    terminal_capture = _FileCapture.open(
        terminal,
        label="unfinalized operator terminal manifest",
        retain_payload=False,
    )
    try:
        return {
            "manifest": manifest,
            "outer_finalized": False,
            "terminal": terminal_capture.row(),
            "expected_completion_marker": str(marker),
        }
    finally:
        terminal_capture.close()


class _Attempt:
    def __init__(
        self,
        *,
        final: Path,
        kind: str,
        lane: str | None,
        inputs: list[dict[str, object]],
    ) -> None:
        self.final = final
        self.kind = kind
        self.lane = lane
        self.inputs = inputs
        self.staging = final.parent / f".{final.name}.staging"
        try:
            os.mkdir(self.staging, 0o700)
        except FileExistsError:
            raise FileExistsError(
                f"operator one-shot staging claim already exists: {self.staging}"
            ) from None
        _fsync_directory(final.parent)
        self.published = False

    def finish(
        self,
        *,
        status: str,
        primary: dict[str, object],
        error: dict[str, object] | None,
        final_check: Callable[[], None],
    ) -> Path:
        if status not in _allowed_statuses(self.kind):
            raise OperatorError("operator attempt terminal status is invalid")
        rows = _file_rows(self.staging, exclude={_TERMINAL_MANIFEST})
        manifest = {
            "schema": _ATTEMPT_SCHEMA,
            "experiment_id": "WM-001",
            "protocol_version": "1.18.0",
            "assurance": assurance_record(),
            "kind": self.kind,
            "lane": self.lane,
            "status": status,
            "inputs": self.inputs,
            "primary": primary,
            "error": error,
            "files": rows,
            "file_count": len(rows),
            "manifest_excludes": [_TERMINAL_MANIFEST],
        }
        terminal = self.staging / _TERMINAL_MANIFEST
        output_captures = _Captures()
        try:
            # The manifest is the last staged file.  Keep every staged byte
            # open while the strict semantic verifier and final live-input
            # check run, then recheck both the descriptors and namespace
            # immediately before the no-replace rename.
            _write_json(terminal, manifest)
            directory_identity = _stat_identity(os.stat(self.staging, follow_symlinks=False))
            for path in _regular_attempt_files(self.staging):
                output_captures.add(
                    path,
                    label=f"staged operator output {path.name}",
                    expected_nlink=(2 if path.name == _FORMAL_AUDIT_CLAIM_FILE else 1),
                )
            _verify_attempt_directory(self.staging)
            final_check()
            output_captures.recheck()
            if _stat_identity(os.stat(self.staging, follow_symlinks=False)) != directory_identity:
                raise OperatorError("operator staging namespace changed before publication")
            directory_descriptor = os.open(
                self.staging,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            _rename_noreplace(self.staging, self.final)
            self.published = True
        finally:
            output_captures.close()
            if not self.published:
                terminal.unlink(missing_ok=True)
        _verify_attempt_directory(self.final)
        from .producer_bootstrap import register_outer_terminal

        logical_exit_code = {
            "accepted": 0,
            "rejected": 1,
            "failure": 2,
        }[status]
        register_outer_terminal(
            self.final / _TERMINAL_MANIFEST,
            logical_exit_code=logical_exit_code,
        )
        return self.final

    def cleanup(self) -> None:
        if self.published or not self.staging.exists():
            return
        for path in self.staging.iterdir():
            if path.is_file() and not path.is_symlink():
                path.unlink()
        self.staging.rmdir()


def _require_sealed_entry() -> None:
    from .experiment import _verify_live_bootstrap_custody

    try:
        _verify_live_bootstrap_custody()
    except (OSError, RuntimeError, ValueError) as error:
        raise OperatorError("operator must be entered through the sealed producer bootstrap") from error


def _binding_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one atomic WM-001 v1.18 binding attempt",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True, type=Path, action=_StoreOnce)
    parser.add_argument(
        "--test-report",
        required=True,
        type=Path,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--development-closure",
        required=True,
        type=Path,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--closure-attempt",
        required=True,
        type=Path,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=("cpu", "cuda"),
        action=_StoreOnce,
    )
    return parser


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one atomic captured-audit attempt",
        allow_abbrev=False,
    )
    parser.add_argument("lane", choices=("development", "formal"))
    parser.add_argument("--producer", required=True, type=Path, action=_StoreOnce)
    parser.add_argument("--output", required=True, type=Path, action=_StoreOnce)
    return parser


def _closure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transition a completed development audit into closure",
        allow_abbrev=False,
    )
    parser.add_argument("--producer", required=True, type=Path, action=_StoreOnce)
    parser.add_argument(
        "--audit-attempt",
        required=True,
        type=Path,
        action=_StoreOnce,
    )
    parser.add_argument("--output", required=True, type=Path, action=_StoreOnce)
    return parser


def _verify_finalized_producer(
    producer: Path,
    *,
    lane: str,
) -> tuple[dict[str, object], dict[str, object]]:
    from .artifact import verify_producer_manifest
    from .verify import verify_result

    manifest = cast(dict[str, object], verify_producer_manifest(producer))
    if manifest.get("status") != "completed" or manifest.get("lane") != lane or manifest.get("error") is not None:
        raise OperatorError(f"{lane} operation requires one completed producer")
    binding_path = producer / "formal-binding.json" if lane == "formal" else None
    result = cast(
        dict[str, object],
        verify_result(producer / "result.json", binding_path),
    )
    if result.get("lane") != lane:
        raise OperatorError("producer result lane differs from requested lane")
    return manifest, result


@dataclass
class _ProducerCustody:
    root: Path
    lane: str
    directory_identity: tuple[int, ...]
    manifest: dict[str, object]
    captures: _Captures

    @classmethod
    def capture(cls, root: Path, *, lane: str) -> tuple[_ProducerCustody, dict[str, object]]:
        producer = _canonical_existing_directory(root, label=f"{lane} producer")
        directory_identity = _stat_identity(os.stat(producer, follow_symlinks=False))
        manifest, result = _verify_finalized_producer(producer, lane=lane)
        captures = _Captures()
        try:
            captures.add(
                producer / "producer-manifest.json",
                label="producer manifest",
                expected_nlink=2,
            )
            captures.add(producer / "result.json", label="producer raw result")
            if lane == "formal":
                captures.add(
                    producer / "formal-binding.json",
                    label="copied formal binding",
                )
        except BaseException:
            captures.close()
            raise
        return (
            cls(
                root=producer,
                lane=lane,
                directory_identity=directory_identity,
                manifest=manifest,
                captures=captures,
            ),
            result,
        )

    def input_rows(self) -> list[dict[str, object]]:
        return self.captures.identities()

    def recheck(self) -> None:
        current_manifest, _ = _verify_finalized_producer(
            self.root,
            lane=self.lane,
        )
        self.captures.recheck()
        if (
            current_manifest != self.manifest
            or _stat_identity(os.stat(self.root, follow_symlinks=False)) != self.directory_identity
        ):
            raise OperatorError("producer changed before operator publication")

    def close(self) -> None:
        self.captures.close()


@dataclass
class _FormalAuditClaim:
    marker: _FileCapture
    launch_bootstrap: _FileCapture

    def recheck(self, claim_path: Path) -> None:
        self.marker.recheck()
        self.launch_bootstrap.recheck()
        try:
            same_inode = os.path.samefile(self.marker.path, claim_path)
        except OSError as error:
            raise OperatorError("formal audit marker/claim inode cannot be verified") from error
        if not same_inode:
            raise OperatorError("formal audit marker differs from its attempt claim")

    def close(self) -> None:
        self.marker.close()
        self.launch_bootstrap.close()


def _formal_claim_value(
    *,
    producer: _ProducerCustody,
    output: Path,
    launch_bootstrap_sha256: str,
) -> dict[str, object]:
    digests = {Path(cast(str, row["path"])).name: row["sha256"] for row in producer.input_rows()}
    if set(digests) != {
        "producer-manifest.json",
        "result.json",
        "formal-binding.json",
    }:
        raise OperatorError("formal producer custody cannot form the sole audit claim")
    from .verify import verify_binding

    formal_binding = verify_binding(producer.root / "formal-binding.json")
    source = formal_binding.get("source")
    execution_sources = source.get("execution_source_sha256") if isinstance(source, dict) else None
    if (
        not isinstance(execution_sources, dict)
        or execution_sources.get("launch_bootstrap.py") != launch_bootstrap_sha256
    ):
        raise OperatorError("formal binding does not bind the live launch bootstrap")
    return {
        "schema": _FORMAL_AUDIT_CLAIM_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.18.0",
        "claim_status": "consumed",
        "attempt_path": str(output),
        "marker_path": str(FORMAL_AUDIT_CLAIM_MARKER),
        "producer_root": str(producer.root),
        "producer_manifest_sha256": digests["producer-manifest.json"],
        "raw_result_sha256": digests["result.json"],
        "formal_binding_sha256": digests["formal-binding.json"],
        "launch_bootstrap_sha256": launch_bootstrap_sha256,
    }


def _publish_formal_audit_claim(
    *,
    attempt: _Attempt,
    producer: _ProducerCustody,
    on_irreversible: Callable[[], None],
) -> _FormalAuditClaim:
    marker = FORMAL_AUDIT_CLAIM_MARKER
    if not marker.is_absolute() or marker.resolve(strict=False) != marker or not marker.is_relative_to(REPO):
        raise OperatorError("formal audit claim marker path is not canonical")
    marker.parent.mkdir(parents=True, exist_ok=True)
    _canonical_existing_directory(
        marker.parent,
        label="formal audit claim marker directory",
    )
    if os.path.lexists(marker):
        raise _FormalAuditRetired("WM-001 protocol 1.18 formal audit claim is already consumed")
    launch_capture = _FileCapture.open(
        Path(__file__).with_name("launch_bootstrap.py"),
        label="formal audit launch bootstrap",
        retain_payload=False,
    )
    claim_path = attempt.staging / _FORMAL_AUDIT_CLAIM_FILE
    try:
        value = _formal_claim_value(
            producer=producer,
            output=attempt.final,
            launch_bootstrap_sha256=launch_capture.sha256,
        )
        _write_json(claim_path, value)
        try:
            os.link(claim_path, marker, follow_symlinks=False)
        except FileExistsError as error:
            raise _FormalAuditRetired("WM-001 protocol 1.18 formal audit claim is already consumed") from error
        except OSError as error:
            raise OperatorError("formal audit claim marker could not be published") from error
        on_irreversible()
        _after_formal_audit_claim_link()
        marker_parent_descriptor = os.open(
            marker.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(marker_parent_descriptor)
        finally:
            os.close(marker_parent_descriptor)
        staging_descriptor = os.open(
            attempt.staging,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        marker_capture = _FileCapture.open(
            marker,
            label="formal audit claim marker",
            retain_payload=True,
            expected_nlink=2,
        )
        try:
            assert marker_capture.payload is not None
            if marker_capture.payload != _canonical_json_bytes(value) or not os.path.samefile(marker, claim_path):
                raise OperatorError("formal audit marker is not the staged claim inode")
            return _FormalAuditClaim(
                marker=marker_capture,
                launch_bootstrap=launch_capture,
            )
        except BaseException:
            marker_capture.close()
            raise
    except BaseException:
        launch_capture.close()
        raise


def _after_formal_audit_claim_link() -> None:
    """Test hook after the formal claim link becomes irreversible."""


def _recover_formal_audit_claim(
    *,
    attempt: _Attempt,
    producer: _ProducerCustody,
) -> _FormalAuditClaim:
    """Reopen claim custody after a post-link publication fault."""

    launch_capture = _FileCapture.open(
        Path(__file__).with_name("launch_bootstrap.py"),
        label="formal audit launch bootstrap",
        retain_payload=False,
    )
    try:
        expected = _canonical_json_bytes(
            _formal_claim_value(
                producer=producer,
                output=attempt.final,
                launch_bootstrap_sha256=launch_capture.sha256,
            )
        )
        marker_capture = _FileCapture.open(
            FORMAL_AUDIT_CLAIM_MARKER,
            label="formal audit claim marker",
            retain_payload=True,
            expected_nlink=2,
        )
        try:
            claim_path = attempt.staging / _FORMAL_AUDIT_CLAIM_FILE
            if (
                marker_capture.payload != expected
                or claim_path.read_bytes() != expected
                or not os.path.samefile(
                    FORMAL_AUDIT_CLAIM_MARKER,
                    claim_path,
                )
            ):
                raise OperatorError("irreversible formal audit claim cannot be recovered")
            return _FormalAuditClaim(
                marker=marker_capture,
                launch_bootstrap=launch_capture,
            )
        except BaseException:
            marker_capture.close()
            raise
    except BaseException:
        launch_capture.close()
        raise


def _verify_formal_audit_claim(
    root: Path,
    *,
    primary: Mapping[str, object],
    inputs: Sequence[Mapping[str, object]],
) -> None:
    if primary.get("claim_file") != _FORMAL_AUDIT_CLAIM_FILE:
        raise OperatorError("formal audit attempt omits its sole claim")
    payload = _attempt_file_payload(
        root,
        _FORMAL_AUDIT_CLAIM_FILE,
        label="formal audit attempt claim",
    )
    claim = _canonical_object(payload, label="formal audit attempt claim")
    expected = {
        "schema",
        "experiment_id",
        "protocol_version",
        "claim_status",
        "attempt_path",
        "marker_path",
        "producer_root",
        "producer_manifest_sha256",
        "raw_result_sha256",
        "formal_binding_sha256",
        "launch_bootstrap_sha256",
    }
    if (
        set(claim) != expected
        or claim.get("schema") != _FORMAL_AUDIT_CLAIM_SCHEMA
        or claim.get("experiment_id") != "WM-001"
        or claim.get("protocol_version") != "1.18.0"
        or claim.get("claim_status") != "consumed"
        or claim.get("attempt_path") != str(FORMAL_AUDIT_ATTEMPT_PATH)
        or claim.get("marker_path") != str(FORMAL_AUDIT_CLAIM_MARKER)
        or claim.get("producer_root") != primary.get("producer_root")
        or any(
            not _sha256_string(claim.get(field))
            for field in (
                "producer_manifest_sha256",
                "raw_result_sha256",
                "formal_binding_sha256",
                "launch_bootstrap_sha256",
            )
        )
    ):
        raise OperatorError("formal audit claim is malformed")
    producer_root = Path(cast(str, primary["producer_root"]))
    producer, _ = _ProducerCustody.capture(
        producer_root,
        lane="formal",
    )
    try:
        live_inputs = producer.input_rows()
        if list(inputs) != live_inputs:
            raise OperatorError("formal audit claim inputs differ from the strict producer")
        input_digests = {Path(cast(str, row["path"])).name: row["sha256"] for row in live_inputs}
        if (
            claim.get("producer_manifest_sha256") != input_digests.get("producer-manifest.json")
            or claim.get("raw_result_sha256") != input_digests.get("result.json")
            or claim.get("formal_binding_sha256") != input_digests.get("formal-binding.json")
        ):
            raise OperatorError("formal audit claim differs from producer input identities")
        from .verify import verify_binding

        formal_binding = verify_binding(producer_root / "formal-binding.json")
        source = formal_binding.get("source")
        execution_sources = source.get("execution_source_sha256") if isinstance(source, dict) else None
        bound_launch_sha256 = (
            execution_sources.get("launch_bootstrap.py") if isinstance(execution_sources, dict) else None
        )
        if not _sha256_string(bound_launch_sha256) or claim.get("launch_bootstrap_sha256") != bound_launch_sha256:
            raise OperatorError("formal audit launch identity differs from its binding")
    finally:
        producer.close()
    marker_capture = _FileCapture.open(
        FORMAL_AUDIT_CLAIM_MARKER,
        label="formal audit claim marker",
        retain_payload=True,
        expected_nlink=2,
    )
    try:
        assert marker_capture.payload is not None
        if marker_capture.payload != payload or not os.path.samefile(
            FORMAL_AUDIT_CLAIM_MARKER,
            root / _FORMAL_AUDIT_CLAIM_FILE,
        ):
            raise OperatorError("formal audit marker is not the published attempt claim inode")
    finally:
        marker_capture.close()


def _publish_failure_attempt(
    attempt: _Attempt,
    *,
    phase: str,
    error: BaseException,
    final_check: Callable[[], None],
    preserved_primary: Mapping[str, object] | None = None,
) -> int:
    record = _failure_record(
        kind=attempt.kind,
        lane=attempt.lane,
        phase=phase,
        error=error,
    )
    _write_json(attempt.staging / "execution-failure.json", record)
    primary = dict(preserved_primary or {})
    primary["execution_failure_file"] = "execution-failure.json"
    attempt.finish(
        status="failure",
        primary=primary,
        error={
            "failure_code": record["failure_code"],
            "error_type": record["error_type"],
        },
        final_check=final_check,
    )
    return 2


def binding_main(argv: Sequence[str] | None = None) -> int:
    """Create, verify, and atomically publish a formal-binding attempt."""

    arguments = _binding_parser().parse_args(argv)
    if arguments.output != FORMAL_BINDING_ATTEMPT_PATH:
        raise OperatorError(
            f"formal binding output must be the canonical protocol-1.18 attempt path {FORMAL_BINDING_ATTEMPT_PATH}"
        )
    output = _attempt_output(
        arguments.output,
        root=BINDING_ATTEMPTS_ROOT,
        label="binding attempt",
    )
    test_report = _canonical_existing_file(
        arguments.test_report,
        label="preformal test report",
    )
    development_closure = _canonical_existing_file(
        arguments.development_closure,
        label="development closure",
    )
    closure_attempt = _canonical_existing_directory(
        arguments.closure_attempt,
        label="development closure attempt",
    )
    from . import binding as binding_module
    from .preformal import PREFORMAL_REPORT_PATH
    from .verify import verify_binding

    if test_report != PREFORMAL_REPORT_PATH:
        raise OperatorError("binding requires the canonical preformal report")
    if development_closure != binding_module.DEVELOPMENT_CLOSURE_PATH:
        raise OperatorError("binding requires the canonical development closure")
    if closure_attempt != CLOSURE_ATTEMPT_PATH:
        raise OperatorError("binding requires the canonical development closure attempt")
    closure_attempt_manifest = verify_operator_attempt(closure_attempt)
    closure_primary = cast(
        dict[str, object],
        closure_attempt_manifest["primary"],
    )
    if (
        closure_attempt_manifest.get("kind") != "closure"
        or closure_attempt_manifest.get("lane") != "development"
        or closure_attempt_manifest.get("status") != "accepted"
        or closure_primary.get("closure_reference_file") != "closure-reference.json"
    ):
        raise OperatorError("binding requires one accepted outer-finalized closure attempt")
    report = binding_module.verify_canonical_machine_test_report(test_report)
    log_rows = binding_module.preformal_log_rows(test_report, report)
    binding_module.verify_preclaim_log_schema_compatibility(log_rows)
    binding_module.verify_development_closure(development_closure)
    captures = _Captures()
    attempt: _Attempt | None = None
    try:
        captures.add(
            test_report,
            label="preformal test report",
            retain_payload=True,
        )
        for index, row in enumerate(log_rows):
            captures.add(
                test_report.with_name(cast(str, row["path"])),
                label=f"preformal log {index}",
            )
        captures.add(
            development_closure,
            label="development closure",
            retain_payload=True,
        )
        captures.add(
            closure_attempt / _TERMINAL_MANIFEST,
            label="development closure attempt terminal manifest",
            retain_payload=True,
            expected_nlink=2,
        )
        captures.add(
            outer_completion_marker(closure_attempt / _TERMINAL_MANIFEST),
            label="development closure outer completion",
            expected_nlink=2,
        )
        attempt = _Attempt(
            final=output,
            kind="binding",
            lane=None,
            inputs=captures.identities(),
        )
        _require_sealed_entry()
        created = binding_module.create_formal_binding(
            output_path=attempt.staging / "formal-binding.json",
            test_report_path=test_report,
            development_closure_path=development_closure,
            device=arguments.device,
        )
        verified = verify_binding(attempt.staging / "formal-binding.json")
        if verified != created:
            raise OperatorError("created binding differs from its strict verifier")
        from .artifact_audit import preflight_formal_input_package

        preflight = preflight_formal_input_package(
            attempt.staging / "formal-binding.json"
        )
        _atomic_write(
            attempt.staging / _FORMAL_INPUT_PREFLIGHT_FILE,
            _canonical_json_bytes(preflight),
        )

        def final_check() -> None:
            captures.recheck()
            binding_module.verify_canonical_machine_test_report(test_report)
            binding_module.verify_development_closure(development_closure)
            current_closure_attempt = verify_operator_attempt(closure_attempt)
            if current_closure_attempt != closure_attempt_manifest:
                raise OperatorError("development closure attempt changed before binding publication")
            _require_sealed_entry()

        attempt.finish(
            status="accepted",
            primary={"binding_file": "formal-binding.json"},
            error=None,
            final_check=final_check,
        )
        return 0
    except Exception as error:
        if attempt is None:
            raise
        if attempt.published:
            raise

        def failure_final_check() -> None:
            captures.recheck()
            _require_sealed_entry()

        return _publish_failure_attempt(
            attempt,
            phase="binding",
            error=error,
            final_check=failure_final_check,
        )
    finally:
        captures.close()
        if attempt is not None and attempt.published:
            attempt.cleanup()


def _development_audit_arguments(
    producer: _ProducerCustody,
    result: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    from . import binding as binding_module
    from .verify import PROTOCOL_PATH, RESULT_SCHEMA_PATH

    execution_identity = binding_module._validate_execution_identity(
        result.get("execution"),
        require_live_identity=True,
    )
    process_environment = execution_identity.get("process_environment")
    if not isinstance(process_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in process_environment.items()
    ):
        raise OperatorError("development producer environment is malformed")
    arguments: dict[str, object] = {
        "auditor_arguments": binding_module._development_audit_argv(producer.root),
        "support_files": {
            "producer_bootstrap.py": Path(__file__).with_name("producer_bootstrap.py"),
            "protocol.json": PROTOCOL_PATH,
            "schemas/raw-result.schema.json": RESULT_SCHEMA_PATH,
        },
        "closure_import_roots": binding_module.package_roots(),
        "source_mode": "descriptor",
        "working_directory": binding_module.REPO,
        "environment": binding_module._audit_environment(cast(dict[str, str], process_environment)),
    }
    source = Path(__file__).with_name("artifact_audit.py")
    return source, arguments


def audit_main(argv: Sequence[str] | None = None) -> int:
    """Run and atomically preserve every execution of one independent audit."""

    from .audit_runner import AuditExecution

    arguments = _audit_parser().parse_args(argv)
    lane = cast(str, arguments.lane)
    if lane == "development":
        if arguments.output != DEVELOPMENT_AUDIT_ATTEMPT_PATH:
            raise OperatorError(
                "development audit output must be the sole canonical "
                f"protocol-1.18 attempt path {DEVELOPMENT_AUDIT_ATTEMPT_PATH}"
            )
        if arguments.producer != DEVELOPMENT_QUALIFICATION_PATH:
            raise OperatorError(
                f"development audit requires the sole canonical qualification producer {DEVELOPMENT_QUALIFICATION_PATH}"
            )
        from . import binding as binding_module

        if os.path.lexists(binding_module.DEVELOPMENT_CLOSURE_PATH):
            raise OperatorError("WM-001 protocol 1.18 development audit is retired after development closure")
    else:
        if arguments.output != FORMAL_AUDIT_ATTEMPT_PATH:
            raise OperatorError(
                f"formal audit output must be the sole canonical protocol-1.18 attempt path {FORMAL_AUDIT_ATTEMPT_PATH}"
            )
        if os.path.lexists(FORMAL_AUDIT_CLAIM_MARKER):
            raise _FormalAuditRetired("WM-001 protocol 1.18 formal audit claim is already consumed")
    output = _attempt_output(
        arguments.output,
        root=AUDIT_ATTEMPTS_ROOT,
        label="audit attempt",
    )
    producer, result = _ProducerCustody.capture(arguments.producer, lane=lane)
    attempt = _Attempt(
        final=output,
        kind="audit",
        lane=lane,
        inputs=producer.input_rows(),
    )
    execution_receipts: list[str] = []
    execution_failures: list[str] = []
    audit_file: str | None = None
    reproduction_file: str | None = None
    reproduction_runtime_file: str | None = None
    formal_claim: _FormalAuditClaim | None = None
    formal_claim_consumed = False
    try:
        if lane == "development":
            from .audit_runner import run_captured_auditor

            source, common = _development_audit_arguments(producer, result)
            _require_sealed_entry()
            first = run_captured_auditor(
                source,
                auditor_arguments=cast(
                    Sequence[str],
                    common["auditor_arguments"],
                ),
                support_files=cast(
                    Mapping[str, Path],
                    common["support_files"],
                ),
                closure_import_roots=cast(
                    Sequence[Path],
                    common["closure_import_roots"],
                ),
                source_mode="descriptor",
                working_directory=cast(Path, common["working_directory"]),
                environment=cast(
                    Mapping[str, str],
                    common["environment"],
                ),
            )
        else:
            from .binding import run_bound_outcome_audit

            # Everything that can be prepared before consuming the one formal
            # audit claim is completed first.  The no-replace hardlink is the
            # final action immediately before entering the bound runner.
            producer.recheck()
            _require_sealed_entry()

            def mark_claim_irreversible() -> None:
                nonlocal formal_claim_consumed
                formal_claim_consumed = True

            formal_claim = _publish_formal_audit_claim(
                attempt=attempt,
                producer=producer,
                on_irreversible=mark_claim_irreversible,
            )
            first = cast(
                AuditExecution,
                run_bound_outcome_audit(producer.root),
            )
        first_row = _write_execution(
            attempt.staging,
            prefix="audit-execution-01",
            execution=first,
        )
        first_stdout = cast(Any, first).stdout
        _atomic_write(
            attempt.staging / "independent-audit.json",
            first_stdout,
        )
        audit_file = "independent-audit.json"
        _, first_passed = _canonical_report(first_stdout)
        execution_receipts.append(cast(str, first_row["receipt_file"]))
        if lane == "development":
            replay = run_captured_auditor(
                source,
                auditor_arguments=cast(
                    Sequence[str],
                    common["auditor_arguments"],
                ),
                support_files=cast(
                    Mapping[str, Path],
                    common["support_files"],
                ),
                closure_import_roots=cast(
                    Sequence[Path],
                    common["closure_import_roots"],
                ),
                source_mode="descriptor",
                working_directory=cast(Path, common["working_directory"]),
                environment=cast(
                    Mapping[str, str],
                    common["environment"],
                ),
                runtime_manifest=first.runtime_manifest,
                invocation_manifest=first.invocation_manifest,
            )
            replay_row = _write_execution(
                attempt.staging,
                prefix="audit-execution-02",
                execution=replay,
            )
            execution_receipts.append(cast(str, replay_row["receipt_file"]))
            _, replay_passed = _canonical_report(cast(Any, replay).stdout)
            if (
                replay_passed is not first_passed
                or cast(Any, replay).stdout != first_stdout
                or cast(Any, replay).runtime_manifest != cast(Any, first).runtime_manifest
                or cast(Any, replay).invocation_manifest != cast(Any, first).invocation_manifest
            ):
                raise OperatorError("development audit replay differs from its first execution")
            if first_passed:
                from . import binding as binding_module

                reproduction_file = "audit-reproduction.json"
                receipt = binding_module.create_audit_reproduction_receipt(
                    supplied_audit_path=attempt.staging / "independent-audit.json",
                    execution=replay,
                    output_path=attempt.staging / reproduction_file,
                )
                reproduction_runtime_file = cast(
                    str,
                    receipt["runtime_manifest_file"],
                )
        status = "accepted" if first_passed else "rejected"

        def final_check() -> None:
            producer.recheck()
            if formal_claim is not None:
                formal_claim.recheck(attempt.staging / _FORMAL_AUDIT_CLAIM_FILE)
            _require_sealed_entry()

        attempt.finish(
            status=status,
            primary={
                "producer_root": str(producer.root),
                "audit_file": audit_file,
                "executions": execution_receipts,
                "execution_failures": execution_failures,
                "reproduction_file": reproduction_file,
                "reproduction_runtime_file": reproduction_runtime_file,
                "claim_file": (_FORMAL_AUDIT_CLAIM_FILE if formal_claim is not None else None),
            },
            error=None,
            final_check=final_check,
        )
        return 0 if first_passed else 1
    except _FormalAuditRetired:
        raise
    except Exception as error:
        if attempt.published:
            raise
        if lane == "formal" and not formal_claim_consumed:
            raise
        if lane == "formal" and formal_claim_consumed and formal_claim is None:
            formal_claim = _recover_formal_audit_claim(
                attempt=attempt,
                producer=producer,
            )
        from .audit_runner import AuditExecutionFailure

        if isinstance(error, AuditExecutionFailure):
            ordinal = len(execution_receipts) + len(execution_failures) + 1
            execution_failures.append(
                _write_execution_failure(
                    attempt.staging,
                    prefix=f"audit-execution-{ordinal:02d}",
                    error=error,
                )
            )

        def failure_final_check() -> None:
            producer.recheck()
            if formal_claim is not None:
                formal_claim.recheck(attempt.staging / _FORMAL_AUDIT_CLAIM_FILE)
            _require_sealed_entry()

        return _publish_failure_attempt(
            attempt,
            phase="audit_execution",
            error=error,
            final_check=failure_final_check,
            preserved_primary={
                "producer_root": str(producer.root),
                "audit_file": audit_file,
                "executions": execution_receipts,
                "execution_failures": execution_failures,
                "reproduction_file": reproduction_file,
                "reproduction_runtime_file": reproduction_runtime_file,
                "claim_file": (_FORMAL_AUDIT_CLAIM_FILE if formal_claim_consumed else None),
            },
        )
    finally:
        if formal_claim is not None:
            formal_claim.close()
        producer.close()
        if attempt.published:
            attempt.cleanup()


def _closure_matches_inputs(
    closure: Mapping[str, object],
    *,
    producer: Path,
    audit_payload: bytes,
    reproduction_payload: bytes,
) -> bool:
    archive = closure.get("qualification_archive")
    if not isinstance(archive, Mapping):
        return False
    members = archive.get("members")
    if not isinstance(members, list):
        return False
    identities = {row.get("path"): row.get("sha256") for row in members if isinstance(row, Mapping)}
    return (
        closure.get("producer_root") == str(producer)
        and identities.get("evidence/independent-audit.json") == hashlib.sha256(audit_payload).hexdigest()
        and identities.get("evidence/audit-reproduction.json") == hashlib.sha256(reproduction_payload).hexdigest()
    )


def closure_main(argv: Sequence[str] | None = None) -> int:
    """Transition one accepted development-audit package into canonical closure."""

    arguments = _closure_parser().parse_args(argv)
    if arguments.output != CLOSURE_ATTEMPT_PATH:
        raise OperatorError(
            f"development closure output must be the canonical protocol-1.18 attempt path {CLOSURE_ATTEMPT_PATH}"
        )
    if arguments.producer != DEVELOPMENT_QUALIFICATION_PATH:
        raise OperatorError(
            f"development closure requires the sole canonical qualification producer {DEVELOPMENT_QUALIFICATION_PATH}"
        )
    from . import binding as binding_module

    marker = binding_module.DEVELOPMENT_CLOSURE_PATH
    if os.path.lexists(marker):
        raise OperatorError(
            "WM-001 protocol 1.18 development closure marker already "
            "consumes the one-shot closure"
        )
    output = _attempt_output(
        arguments.output,
        root=CLOSURE_ATTEMPTS_ROOT,
        label="closure attempt",
    )
    if arguments.audit_attempt != DEVELOPMENT_AUDIT_ATTEMPT_PATH:
        raise OperatorError(
            f"development closure requires the canonical protocol-1.18 audit attempt {DEVELOPMENT_AUDIT_ATTEMPT_PATH}"
        )
    audit_attempt = _canonical_existing_directory(
        arguments.audit_attempt,
        label="development audit attempt",
    )
    audit_manifest = verify_operator_attempt(audit_attempt)
    if (
        audit_manifest.get("kind") != "audit"
        or audit_manifest.get("lane") != "development"
        or audit_manifest.get("status") != "accepted"
    ):
        raise OperatorError("closure requires one accepted development audit attempt")
    audit_captures = _Captures()
    closure_outputs = _Captures()
    producer: _ProducerCustody | None = None
    attempt: _Attempt | None = None
    try:
        captured_audit_files: dict[str, _FileCapture] = {}
        for path in _regular_attempt_files(
            audit_attempt,
            terminal_nlink=2,
        ):
            captured_audit_files[path.name] = audit_captures.add(
                path,
                label=f"development audit package {path.name}",
                retain_payload=True,
                expected_nlink=(2 if path.name == _TERMINAL_MANIFEST else 1),
            )
        audit_completion = outer_completion_marker(audit_attempt / _TERMINAL_MANIFEST)
        audit_captures.add(
            audit_completion,
            label="development audit outer completion",
            expected_nlink=2,
        )
        # Establish that the exact bytes now held by descriptors are the same
        # package accepted by the strict public verifier.
        audit_manifest = verify_operator_attempt(audit_attempt)
        audit_captures.recheck()
        primary = cast(dict[str, object], audit_manifest["primary"])
        producer, _ = _ProducerCustody.capture(
            arguments.producer,
            lane="development",
        )
        if primary.get("producer_root") != str(producer.root):
            raise OperatorError("audit attempt targets a different producer")
        audit_name = cast(str, primary["audit_file"])
        reproduction_name = cast(str, primary["reproduction_file"])
        audit_capture = captured_audit_files[audit_name]
        reproduction_capture = captured_audit_files[reproduction_name]
        assert audit_capture.payload is not None
        assert reproduction_capture.payload is not None
        audit_payload = audit_capture.payload
        reproduction_payload = reproduction_capture.payload
        reproduction = _canonical_object(
            reproduction_payload,
            label="development audit reproduction",
        )
        runtime_name = cast(str, reproduction["runtime_manifest_file"])
        runtime_file = audit_attempt / runtime_name
        if runtime_name not in captured_audit_files:
            raise OperatorError("development audit package omits its runtime sidecar")
        audit_file = audit_attempt / audit_name
        reproduction_file = audit_attempt / reproduction_name
        inputs = [
            *producer.input_rows(),
            *audit_captures.identities(),
        ]
        attempt = _Attempt(
            final=output,
            kind="closure",
            lane="development",
            inputs=inputs,
        )
        _require_sealed_entry()
        binding_module.create_development_closure(
            producer_root=producer.root,
            audit_path=audit_file,
            audit_reproduction_path=reproduction_file,
            runtime_manifest_path=runtime_file,
        )
        closure = binding_module.verify_development_closure(marker)
        if not _closure_matches_inputs(
            closure,
            producer=producer.root,
            audit_payload=audit_payload,
            reproduction_payload=reproduction_payload,
        ):
            raise OperatorError("created development closure input identity mismatch")
        marker_capture = closure_outputs.add(
            marker,
            label="canonical development closure marker",
            retain_payload=True,
        )
        archive = closure.get("qualification_archive")
        if not isinstance(archive, dict) or not isinstance(archive.get("canonical_path"), str):
            raise OperatorError("development closure archive identity is malformed")
        archive_path = REPO / cast(str, archive["canonical_path"])
        closure_outputs.add(
            archive_path,
            label="canonical development qualification archive",
        )
        assert marker_capture.payload is not None
        marker_payload = marker_capture.payload
        terminal_capture = captured_audit_files[_TERMINAL_MANIFEST]
        assert terminal_capture.payload is not None
        from .preformal import (
            fresh_runtime_development_closure_reopen,
        )

        fresh_reopen = fresh_runtime_development_closure_reopen(marker)
        fresh_reopen_payload = _canonical_json_bytes(fresh_reopen)
        _atomic_write(
            attempt.staging / "fresh-runtime-reopen.json",
            fresh_reopen_payload,
        )
        reference = {
            "schema": _CLOSURE_REFERENCE_SCHEMA,
            "experiment_id": "WM-001",
            "protocol_version": "1.18.0",
            "closure_marker": str(marker),
            "closure_sha256": hashlib.sha256(marker_payload).hexdigest(),
            "qualification_archive": closure["qualification_archive"],
            "producer_root": str(producer.root),
            "audit_attempt": str(audit_attempt),
            "audit_attempt_manifest_sha256": hashlib.sha256(terminal_capture.payload).hexdigest(),
            "fresh_reopen_file": "fresh-runtime-reopen.json",
            "fresh_reopen_sha256": hashlib.sha256(
                fresh_reopen_payload
            ).hexdigest(),
        }
        _write_json(attempt.staging / "closure-reference.json", reference)

        def final_check() -> None:
            producer.recheck()
            audit_captures.recheck()
            closure_outputs.recheck()
            current = binding_module.verify_development_closure(marker)
            if not _closure_matches_inputs(
                current,
                producer=producer.root,
                audit_payload=audit_payload,
                reproduction_payload=reproduction_payload,
            ):
                raise OperatorError("closure inputs changed before publication")
            _require_sealed_entry()

        attempt.finish(
            status="accepted",
            primary={"closure_reference_file": "closure-reference.json"},
            error=None,
            final_check=final_check,
        )
        return 0
    except Exception as error:
        if attempt is None or producer is None:
            raise
        if attempt.published:
            raise

        def failure_final_check() -> None:
            producer.recheck()
            audit_captures.recheck()
            closure_outputs.recheck()
            _require_sealed_entry()

        return _publish_failure_attempt(
            attempt,
            phase="development_closure",
            error=error,
            final_check=failure_final_check,
        )
    finally:
        closure_outputs.close()
        audit_captures.close()
        if producer is not None:
            producer.close()
        if attempt is not None and attempt.published:
            attempt.cleanup()


__all__ = [
    "AUDIT_ATTEMPTS_ROOT",
    "BINDING_ATTEMPTS_ROOT",
    "CLOSURE_ATTEMPTS_ROOT",
    "CLOSURE_ATTEMPT_PATH",
    "DEVELOPMENT_AUDIT_ATTEMPT_PATH",
    "FORMAL_AUDIT_ATTEMPT_PATH",
    "FORMAL_AUDIT_CLAIM_MARKER",
    "FORMAL_BINDING_ATTEMPT_PATH",
    "OUTER_COMPLETIONS_ROOT",
    "OperatorError",
    "audit_main",
    "binding_main",
    "closure_main",
    "inspect_unfinalized_operator_attempt",
    "outer_completion_marker",
    "verify_binding_authorization_inputs",
    "verify_outer_completion",
    "verify_operator_attempt",
]
