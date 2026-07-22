"""Independent verifier for the WM-001 accepted-binding rehearsal.

The outer launcher owns rehearsal publication.  This module deliberately does
not import the launcher: it reopens the resulting package, its single-use
claim, and its outer completion through a separate implementation before the
package can authorize a formal launch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, cast

from .assurance import ASSURANCE


def _repository_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        candidate = Path(completed.stdout.strip())
        if (
            candidate.is_absolute()
            and candidate.resolve(strict=True) == candidate
            and (candidate / ".git").exists()
            and (
                candidate / "bench" / "world_model_lifecycle" / "protocol.json"
            ).is_file()
        ):
            return candidate
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / ".git").exists():
        return source_candidate
    raise RuntimeError(
        "WM-001 rehearsal custody requires a canonical Prospect Git worktree"
    )


PROTOCOL_VERSION = "1.18.0"
REPO = _repository_root()
RESULTS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results"
OPERATOR_RESULTS_ROOT = RESULTS_ROOT / "operator-v1.18"
REHEARSAL_ATTEMPTS_ROOT = OPERATOR_RESULTS_ROOT / "rehearsals"
REHEARSAL_ATTEMPT_PATH = (
    REHEARSAL_ATTEMPTS_ROOT / "accepted-binding-rehearsal-v1.18.0"
)
REHEARSAL_CLAIM_ROOT = RESULTS_ROOT / "rehearsals" / "v1.18"
OUTER_COMPLETIONS_ROOT = RESULTS_ROOT / "outer-completions" / "v1.18"
FORMAL_BINDING_PATH = (
    OPERATOR_RESULTS_ROOT
    / "bindings"
    / "formal-binding-v1.18.0"
    / "formal-binding.json"
)
LAUNCH_BOOTSTRAP_PATH = (
    REPO / "bench" / "world_model_lifecycle" / "launch_bootstrap.py"
)
PRODUCER_BOOTSTRAP_PATH = (
    REPO / "bench" / "world_model_lifecycle" / "producer_bootstrap.py"
)

CLAIM_NAME = "rehearsal-claim.json"
TERMINAL_NAME = "rehearsal-terminal.json"
STDOUT_NAME = "rehearsal.stdout.json"
STDERR_NAME = "rehearsal.stderr.log"
OUTER_RECEIPT_NAME = "rehearsal.outer-receipt.json"

CLAIM_PATH = REHEARSAL_ATTEMPT_PATH / CLAIM_NAME
TERMINAL_PATH = REHEARSAL_ATTEMPT_PATH / TERMINAL_NAME
STDOUT_PATH = REHEARSAL_ATTEMPT_PATH / STDOUT_NAME
STDERR_PATH = REHEARSAL_ATTEMPT_PATH / STDERR_NAME
OUTER_RECEIPT_PATH = REHEARSAL_ATTEMPT_PATH / OUTER_RECEIPT_NAME

CLAIM_SCHEMA = "prospect.wm001.accepted-binding-rehearsal-claim.v1"
TERMINAL_SCHEMA = "prospect.wm001.accepted-binding-rehearsal-terminal.v1"

_MAX_CONTROL_BYTES = 64 << 20
_MAX_STDOUT_BYTES = 1 << 20
_MAX_STDERR_BYTES = 1 << 20
_MAX_OUTER_RECEIPT_BYTES = 4096
_SHA256_EMPTY = hashlib.sha256(b"").hexdigest()
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_FAILURE_PHASES = {
    "pre_dispatch",
    "child_execution",
    "output_validation",
    "post_dispatch",
    "recovery",
}

_CLAIM_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "assurance",
    "status",
    "binding_path",
    "binding_bytes",
    "binding_sha256",
    "attempt_path",
    "marker_path",
    "launch_bootstrap_path",
    "launch_bootstrap_bytes",
    "launch_bootstrap_sha256",
    "producer_bootstrap_path",
    "producer_bootstrap_bytes",
    "producer_bootstrap_sha256",
    "command",
}

_TERMINAL_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "assurance",
    "status",
    "claim_file",
    "claim_marker",
    "claim_bytes",
    "claim_sha256",
    "binding_path",
    "binding_bytes",
    "binding_sha256",
    "child_started",
    "returncode",
    "stdout_file",
    "stdout_bytes",
    "stdout_sha256",
    "stderr_file",
    "stderr_bytes",
    "stderr_sha256",
    "outer_receipt_file",
    "outer_receipt_bytes",
    "outer_receipt_sha256",
    "formal_paths_absent_before",
    "formal_paths_absent_after",
    "phase",
    "error_code",
    "files",
    "file_count",
    "manifest_excludes",
}

_STDOUT_FIELDS = {
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


class RehearsalEvidenceError(RuntimeError):
    """The accepted-binding rehearsal evidence violates its sealed contract."""


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


def _canonical_bytes(value: object) -> bytes:
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


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise RehearsalEvidenceError(
                f"canonical rehearsal JSON repeats key {key!r}"
            )
        value[key] = item
    return value


def _reject_constant(raw: str) -> object:
    raise RehearsalEvidenceError(
        f"canonical rehearsal JSON contains non-finite value {raw}"
    )


def _canonical_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RehearsalEvidenceError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict) or payload != _canonical_bytes(value):
        raise RehearsalEvidenceError(
            f"{label} is not one canonical JSON object followed by LF"
        )
    return cast(dict[str, object], value)


def _strict_json_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(observed, dict):
        if not isinstance(expected, dict) or set(observed) != set(expected):
            return False
        return all(
            _strict_json_equal(observed[key], expected[key])
            for key in observed
        )
    if isinstance(observed, list):
        return isinstance(expected, list) and len(observed) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return bool(observed == expected)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise RehearsalEvidenceError(f"{label} is not a lexical absolute path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise RehearsalEvidenceError(f"{label} cannot be resolved") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RehearsalEvidenceError(
                f"{label} contains a symbolic-link component"
            )


def _canonical_directory(path: Path, *, expected: Path, label: str) -> tuple[int, ...]:
    if path != expected or not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise RehearsalEvidenceError(f"{label} is not its canonical path")
    _reject_symlink_components(path, label=label)
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RehearsalEvidenceError(f"{label} is absent") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.resolve(strict=True) != path:
        raise RehearsalEvidenceError(f"{label} is not a canonical directory")
    return _identity(metadata)


@dataclass
class _Capture:
    path: Path
    descriptor: int
    payload: bytes
    identity: tuple[int, ...]
    expected_nlink: int
    label: str

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        expected_nlink: int,
        maximum_bytes: int = _MAX_CONTROL_BYTES,
    ) -> _Capture:
        if not path.is_absolute() or Path(os.path.abspath(path)) != path:
            raise RehearsalEvidenceError(f"{label} is not a lexical absolute path")
        _reject_symlink_components(path, label=label)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            before = os.fstat(descriptor)
        except OSError as error:
            if "descriptor" in locals():
                os.close(descriptor)
            raise RehearsalEvidenceError(f"{label} cannot be opened") from error
        try:
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != expected_nlink
                or before.st_size < 0
                or before.st_size > maximum_bytes
            ):
                raise RehearsalEvidenceError(
                    f"{label} is not one bounded {expected_nlink}-link regular file"
                )
            payload = os.pread(descriptor, before.st_size + 1, 0)
            after = os.fstat(descriptor)
            current = os.stat(path, follow_symlinks=False)
            descriptor_path = next(
                (
                    Path(f"{prefix}{descriptor}")
                    for prefix in ("/proc/self/fd/", "/dev/fd/")
                    if os.path.exists(f"{prefix}{descriptor}")
                ),
                None,
            )
            if (
                len(payload) != before.st_size
                or _identity(before) != _identity(after)
                or _identity(before) != _identity(current)
                or descriptor_path is None
                or not os.path.samefile(path, descriptor_path)
                or path.resolve(strict=True) != path
            ):
                raise RehearsalEvidenceError(f"{label} changed while captured")
            return cls(
                path=path,
                descriptor=descriptor,
                payload=payload,
                identity=_identity(before),
                expected_nlink=expected_nlink,
                label=label,
            )
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def absolute_row(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "bytes": len(self.payload),
            "sha256": self.sha256,
        }

    def relative_row(self) -> dict[str, object]:
        return {
            "path": self.path.name,
            "bytes": len(self.payload),
            "sha256": self.sha256,
        }

    def recheck(self) -> None:
        _reject_symlink_components(self.path, label=self.label)
        try:
            before = os.fstat(self.descriptor)
            payload = os.pread(self.descriptor, before.st_size + 1, 0)
            after = os.fstat(self.descriptor)
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise RehearsalEvidenceError(
                f"{self.label} cannot be rechecked"
            ) from error
        if (
            _identity(before) != self.identity
            or _identity(after) != self.identity
            or _identity(current) != self.identity
            or payload != self.payload
            or self.path.resolve(strict=True) != self.path
        ):
            raise RehearsalEvidenceError(
                f"{self.label} changed during rehearsal verification"
            )

    def close(self) -> None:
        descriptor = self.descriptor
        if descriptor < 0:
            return
        self.descriptor = -1
        try:
            os.close(descriptor)
        except OSError:
            pass


@dataclass(frozen=True)
class _DirectoryCapture:
    path: Path
    identity: tuple[int, ...]
    names: frozenset[str]
    label: str

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        expected_names: set[str] | None = None,
    ) -> _DirectoryCapture:
        identity = _canonical_directory(path, expected=path, label=label)
        try:
            names = frozenset(entry.name for entry in os.scandir(path))
        except OSError as error:
            raise RehearsalEvidenceError(f"{label} cannot be enumerated") from error
        if expected_names is not None and names != frozenset(expected_names):
            raise RehearsalEvidenceError(f"{label} is not exact")
        return cls(path=path, identity=identity, names=names, label=label)

    def recheck(self) -> None:
        identity = _canonical_directory(
            self.path,
            expected=self.path,
            label=self.label,
        )
        try:
            names = frozenset(entry.name for entry in os.scandir(self.path))
        except OSError as error:
            raise RehearsalEvidenceError(
                f"{self.label} cannot be re-enumerated"
            ) from error
        if identity != self.identity or names != self.names:
            raise RehearsalEvidenceError(
                f"{self.label} changed during rehearsal verification"
            )


def rehearsal_claim_marker(binding_sha256: str) -> Path:
    """Return the sole deterministic non-formal claim marker for a binding."""

    if not _is_sha256(binding_sha256):
        raise RehearsalEvidenceError("rehearsal binding digest is malformed")
    return REHEARSAL_CLAIM_ROOT / f"accepted-binding-{binding_sha256}.json"


def rehearsal_outer_completion(terminal_path: Path | None = None) -> Path:
    """Return the deterministic outer-completion marker for a terminal."""

    if terminal_path is None:
        terminal_path = TERMINAL_PATH
    digest = hashlib.sha256(str(terminal_path).encode("utf-8")).hexdigest()
    return OUTER_COMPLETIONS_ROOT / f"{digest}.json"


def _verified_binding(path: Path) -> dict[str, object]:
    from .verify import verify_binding

    return cast(dict[str, object], verify_binding(path))


def _sealed_matrix_contract_sha256() -> str:
    """Return the matrix identity from the independently verified protocol."""

    from .verify import verify_protocol

    protocol = verify_protocol()
    bindings = protocol.get("bindings")
    development = (
        bindings.get("development_qualification")
        if isinstance(bindings, dict)
        else None
    )
    matrix_digest = (
        development.get("matrix_contract_sha256")
        if isinstance(development, dict)
        else None
    )
    if not _is_sha256(matrix_digest):
        raise RehearsalEvidenceError(
            "sealed protocol has no verified development matrix identity"
        )
    return cast(str, matrix_digest)


def _validate_stdout(
    payload: bytes,
    *,
    binding: dict[str, object],
) -> dict[str, object]:
    value = _canonical_object(payload, label="accepted-binding rehearsal stdout")
    dependencies = binding.get("dependencies")
    audit_execution = binding.get("audit_execution")
    runtime = binding.get("runtime")
    if (
        not isinstance(dependencies, dict)
        or not isinstance(audit_execution, dict)
        or not isinstance(runtime, dict)
    ):
        raise RehearsalEvidenceError(
            "formal binding omits recorded rehearsal conformance inputs"
        )
    inventory = value.get("inventory")
    expected_inventory = {
        "packages": dependencies.get("packages"),
        "package_roots": dependencies.get("package_roots"),
        "standard_library": dependencies.get("standard_library"),
        "package_ownership": dependencies.get("package_ownership"),
    }
    fresh = value.get("fresh_runtime_identity_conformance")
    requesting = fresh.get("requesting_process_id") if isinstance(fresh, dict) else None
    verifier = fresh.get("verifier_process_id") if isinstance(fresh, dict) else None
    matrix_digest = _sealed_matrix_contract_sha256()
    if (
        not isinstance(fresh, dict)
        or set(fresh)
        != {
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
        or fresh.get("schema")
        != "prospect.wm001.fresh-runtime-identity-conformance.v1"
        or fresh.get("experiment_id") != "WM-001"
        or fresh.get("protocol_version") != PROTOCOL_VERSION
        or fresh.get("mode") != "fresh-identity-conformance"
        or not _is_sha256(fresh.get("challenge"))
        or type(requesting) is not int
        or requesting <= 0
        or type(verifier) is not int
        or verifier <= 0
        or requesting == verifier
        or fresh.get("matrix_contract_sha256") != matrix_digest
        or fresh.get("passed") is not True
    ):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal fresh runtime identity is malformed"
        )
    fixed_support = [
        "producer_bootstrap.py",
        "protocol.json",
        "schemas/raw-result.schema.json",
    ]
    if (
        set(value) != _STDOUT_FIELDS
        or value.get("schema")
        != "prospect.wm001.preformal-runtime-check.v1"
        or value.get("mode") != "bootstrap-inventory-conformance"
        or value.get("device") != runtime.get("device")
        or value.get("passed") is not True
        or not _strict_json_equal(inventory, expected_inventory)
        or value.get("inventory_sha256") != _canonical_digest(expected_inventory)
        or value.get("conformance_sha256") != _canonical_digest(audit_execution)
        or value.get("fresh_runtime_identity_conformance_sha256")
        != _canonical_digest(fresh)
        or value.get("restart_runtime_conformance_report_sha256")
        != audit_execution.get("restart_runtime_conformance_report_sha256")
        or value.get("restart_runtime_execution_receipt_sha256")
        != audit_execution.get("restart_runtime_execution_receipt_sha256")
        or value.get("restart_runtime_support_files") != fixed_support
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
    ):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal stdout differs from its binding"
        )
    return value


def _verify_package(
    binding_path: Path,
    captures: dict[str, _Capture],
    namespaces: dict[str, _DirectoryCapture],
) -> tuple[dict[str, object], dict[str, _Capture]]:
    if binding_path != FORMAL_BINDING_PATH:
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal uses a noncanonical binding path"
        )
    binding_capture = _Capture.open(
        binding_path,
        label="accepted-binding rehearsal formal binding",
        expected_nlink=1,
    )
    captures["binding"] = binding_capture
    binding = _canonical_object(
        binding_capture.payload,
        label="accepted-binding rehearsal formal binding",
    )
    verified = _verified_binding(binding_path)
    if not _strict_json_equal(binding, verified):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal binding differs from its strict verifier"
        )
    binding_sha256 = binding_capture.sha256
    marker_path = rehearsal_claim_marker(binding_sha256)
    completion_path = rehearsal_outer_completion()
    expected_names = {
        CLAIM_NAME,
        STDOUT_NAME,
        STDERR_NAME,
        OUTER_RECEIPT_NAME,
        TERMINAL_NAME,
    }
    namespaces["attempts_root"] = _DirectoryCapture.open(
        REHEARSAL_ATTEMPTS_ROOT,
        label="accepted-binding rehearsal attempts namespace",
        expected_names={REHEARSAL_ATTEMPT_PATH.name},
    )
    namespaces["attempt"] = _DirectoryCapture.open(
        REHEARSAL_ATTEMPT_PATH,
        label="accepted-binding rehearsal attempt",
        expected_names=expected_names,
    )
    namespaces["claim_root"] = _DirectoryCapture.open(
        REHEARSAL_CLAIM_ROOT,
        label="accepted-binding rehearsal claim namespace",
        expected_names={marker_path.name},
    )
    namespaces["completion_root"] = _DirectoryCapture.open(
        OUTER_COMPLETIONS_ROOT,
        label="accepted-binding rehearsal outer-completion namespace",
    )
    if completion_path.name not in namespaces["completion_root"].names:
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal outer completion is absent from its namespace"
        )

    capture_specs = {
        "claim": (CLAIM_PATH, CLAIM_NAME, 2, _MAX_CONTROL_BYTES),
        "claim_marker": (marker_path, "claim marker", 2, _MAX_CONTROL_BYTES),
        "stdout": (
            STDOUT_PATH,
            "stdout",
            1,
            _MAX_STDOUT_BYTES,
        ),
        "stderr": (
            STDERR_PATH,
            "stderr",
            1,
            _MAX_STDERR_BYTES,
        ),
        "outer_receipt": (
            OUTER_RECEIPT_PATH,
            "child outer receipt",
            1,
            _MAX_OUTER_RECEIPT_BYTES,
        ),
        "terminal": (TERMINAL_PATH, TERMINAL_NAME, 2, _MAX_CONTROL_BYTES),
        "completion": (
            completion_path,
            "outer completion",
            2,
            _MAX_CONTROL_BYTES,
        ),
        "launch_bootstrap": (
            LAUNCH_BOOTSTRAP_PATH,
            "launch bootstrap source",
            1,
            _MAX_CONTROL_BYTES,
        ),
        "producer_bootstrap": (
            PRODUCER_BOOTSTRAP_PATH,
            "producer bootstrap source",
            1,
            _MAX_CONTROL_BYTES,
        ),
    }
    for role, (path, label, links, maximum) in capture_specs.items():
        captures[role] = _Capture.open(
            path,
            label=f"accepted-binding rehearsal {label}",
            expected_nlink=links,
            maximum_bytes=maximum,
        )
    if (
        not os.path.samefile(CLAIM_PATH, marker_path)
        or captures["claim"].payload != captures["claim_marker"].payload
    ):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal claim marker is not the claim inode"
        )
    if (
        not os.path.samefile(TERMINAL_PATH, completion_path)
        or captures["terminal"].payload != captures["completion"].payload
    ):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal outer completion is not the terminal inode"
        )

    claim = _canonical_object(
        captures["claim"].payload,
        label="accepted-binding rehearsal claim",
    )
    runtime = binding.get("runtime")
    device = runtime.get("device") if isinstance(runtime, dict) else None
    expected_command = [
        "preformal-runtime",
        "bootstrap-inventory-conformance",
        "--device",
        device,
    ]
    if (
        set(claim) != _CLAIM_FIELDS
        or claim.get("schema") != CLAIM_SCHEMA
        or claim.get("experiment_id") != "WM-001"
        or claim.get("protocol_version") != PROTOCOL_VERSION
        or not _strict_json_equal(claim.get("assurance"), ASSURANCE)
        or claim.get("status") != "consumed"
        or claim.get("binding_path") != str(binding_path)
        or type(claim.get("binding_bytes")) is not int
        or claim.get("binding_bytes") != len(binding_capture.payload)
        or claim.get("binding_sha256") != binding_sha256
        or claim.get("attempt_path") != str(REHEARSAL_ATTEMPT_PATH)
        or claim.get("marker_path") != str(marker_path)
        or claim.get("launch_bootstrap_path")
        != str(captures["launch_bootstrap"].path)
        or type(claim.get("launch_bootstrap_bytes")) is not int
        or claim.get("launch_bootstrap_bytes")
        != len(captures["launch_bootstrap"].payload)
        or claim.get("launch_bootstrap_sha256")
        != captures["launch_bootstrap"].sha256
        or claim.get("producer_bootstrap_path")
        != str(captures["producer_bootstrap"].path)
        or type(claim.get("producer_bootstrap_bytes")) is not int
        or claim.get("producer_bootstrap_bytes")
        != len(captures["producer_bootstrap"].payload)
        or claim.get("producer_bootstrap_sha256")
        != captures["producer_bootstrap"].sha256
        or device not in {"cpu", "cuda"}
        or not _strict_json_equal(claim.get("command"), expected_command)
    ):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal claim is malformed or misbound"
        )

    terminal = _canonical_object(
        captures["terminal"].payload,
        label="accepted-binding rehearsal terminal",
    )
    expected_rows = [
        captures[role].relative_row()
        for role in ("claim", "outer_receipt", "stderr", "stdout")
    ]
    expected_rows.sort(key=lambda row: cast(str, row["path"]))
    status = terminal.get("status")
    if (
        set(terminal) != _TERMINAL_FIELDS
        or terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("experiment_id") != "WM-001"
        or terminal.get("protocol_version") != PROTOCOL_VERSION
        or not _strict_json_equal(terminal.get("assurance"), ASSURANCE)
        or not isinstance(status, str)
        or status not in {"accepted", "failed"}
        or terminal.get("claim_file") != CLAIM_NAME
        or terminal.get("claim_marker") != str(marker_path)
        or type(terminal.get("claim_bytes")) is not int
        or terminal.get("claim_bytes") != len(captures["claim"].payload)
        or terminal.get("claim_sha256") != captures["claim"].sha256
        or terminal.get("binding_path") != str(binding_path)
        or type(terminal.get("binding_bytes")) is not int
        or terminal.get("binding_bytes") != len(binding_capture.payload)
        or terminal.get("binding_sha256") != binding_sha256
        or type(terminal.get("child_started")) is not bool
        or terminal.get("stdout_file") != STDOUT_NAME
        or type(terminal.get("stdout_bytes")) is not int
        or terminal.get("stdout_bytes") != len(captures["stdout"].payload)
        or terminal.get("stdout_sha256") != captures["stdout"].sha256
        or terminal.get("stderr_file") != STDERR_NAME
        or type(terminal.get("stderr_bytes")) is not int
        or terminal.get("stderr_bytes") != len(captures["stderr"].payload)
        or terminal.get("stderr_sha256") != captures["stderr"].sha256
        or terminal.get("outer_receipt_file") != OUTER_RECEIPT_NAME
        or type(terminal.get("outer_receipt_bytes")) is not int
        or terminal.get("outer_receipt_bytes")
        != len(captures["outer_receipt"].payload)
        or terminal.get("outer_receipt_sha256")
        != captures["outer_receipt"].sha256
        or type(terminal.get("formal_paths_absent_before")) is not bool
        or type(terminal.get("formal_paths_absent_after")) is not bool
        or not _strict_json_equal(terminal.get("files"), expected_rows)
        or type(terminal.get("file_count")) is not int
        or terminal.get("file_count") != len(expected_rows)
        or terminal.get("manifest_excludes") != [TERMINAL_NAME]
    ):
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal terminal is malformed or misbound"
        )
    child_started = terminal["child_started"]
    returncode = terminal.get("returncode")
    if status == "accepted":
        if (
            child_started is not True
            or type(returncode) is not int
            or returncode != 0
            or captures["stderr"].payload
            or captures["outer_receipt"].payload
            or terminal.get("stderr_sha256") != _SHA256_EMPTY
            or terminal.get("outer_receipt_sha256") != _SHA256_EMPTY
            or terminal.get("formal_paths_absent_before") is not True
            or terminal.get("formal_paths_absent_after") is not True
            or terminal.get("phase") != "complete"
            or terminal.get("error_code") is not None
        ):
            raise RehearsalEvidenceError(
                "accepted-binding rehearsal terminal does not record one clean child"
            )
        _validate_stdout(captures["stdout"].payload, binding=binding)
    else:
        error_code = terminal.get("error_code")
        phase = terminal.get("phase")
        if (
            not isinstance(phase, str)
            or phase not in _FAILURE_PHASES
            or not isinstance(error_code, str)
            or _ERROR_CODE.fullmatch(error_code) is None
            or (
                child_started is False
                and returncode is not None
            )
            or (
                child_started is True
                and type(returncode) is not int
            )
        ):
            raise RehearsalEvidenceError(
                "failed accepted-binding rehearsal terminal is malformed"
            )

    for capture in captures.values():
        capture.recheck()
    for namespace in namespaces.values():
        namespace.recheck()
    return terminal, captures


def verify_accepted_binding_rehearsal(
    binding_path: Path,
) -> dict[str, object]:
    """Strictly reopen one terminal accepted-binding rehearsal package.

    Both accepted and failed terminals are authenticatable.  Call
    :func:`accepted_binding_rehearsal_identity_rows` when evidence must
    authorize a formal launch; that helper rejects every failed terminal.
    """

    captures: dict[str, _Capture] = {}
    namespaces: dict[str, _DirectoryCapture] = {}
    try:
        terminal, captures = _verify_package(
            binding_path,
            captures,
            namespaces,
        )
        return terminal
    except RehearsalEvidenceError:
        raise
    except Exception as error:
        raise RehearsalEvidenceError(
            "accepted-binding rehearsal verification failed"
        ) from error
    finally:
        for capture in reversed(tuple(captures.values())):
            capture.close()


class AcceptedBindingRehearsalCustody:
    """Hold accepted rehearsal evidence stable across authority publication."""

    def __init__(self, binding_path: Path) -> None:
        self._binding_path = binding_path
        self._captures: dict[str, _Capture] = {}
        self._namespaces: dict[str, _DirectoryCapture] = {}
        self._terminal: dict[str, object] | None = None
        self._state = "new"

    def _close(self) -> None:
        for capture in reversed(tuple(self._captures.values())):
            capture.close()
        self._state = "closed"

    def _require_active(self) -> None:
        if self._state != "active" or self._terminal is None:
            raise RehearsalEvidenceError(
                "accepted-binding rehearsal custody is not active"
            )

    def __enter__(self) -> AcceptedBindingRehearsalCustody:
        if self._state != "new":
            raise RehearsalEvidenceError(
                "accepted-binding rehearsal custody is single-use"
            )
        self._state = "opening"
        try:
            terminal, self._captures = _verify_package(
                self._binding_path,
                self._captures,
                self._namespaces,
            )
            if terminal.get("status") != "accepted":
                raise RehearsalEvidenceError(
                    "failed accepted-binding rehearsal cannot authorize formal launch"
                )
            self._terminal = terminal
            self._state = "active"
            return self
        except RehearsalEvidenceError:
            self._close()
            raise
        except Exception as error:
            self._close()
            raise RehearsalEvidenceError(
                "accepted-binding rehearsal custody cannot be established"
            ) from error

    @property
    def terminal(self) -> dict[str, object]:
        """Return the verified accepted terminal value."""

        self._require_active()
        assert self._terminal is not None
        return dict(self._terminal)

    def recheck(self) -> None:
        """Recheck every held descriptor, hard link, and namespace snapshot."""

        self._require_active()
        try:
            for capture in self._captures.values():
                capture.recheck()
            if (
                not os.path.samefile(
                    self._captures["claim"].path,
                    self._captures["claim_marker"].path,
                )
                or self._captures["claim"].payload
                != self._captures["claim_marker"].payload
                or not os.path.samefile(
                    self._captures["terminal"].path,
                    self._captures["completion"].path,
                )
                or self._captures["terminal"].payload
                != self._captures["completion"].payload
            ):
                raise RehearsalEvidenceError(
                    "accepted-binding rehearsal hard-link custody changed"
                )
            for namespace in self._namespaces.values():
                namespace.recheck()
        except RehearsalEvidenceError:
            raise
        except Exception as error:
            raise RehearsalEvidenceError(
                "accepted-binding rehearsal custody recheck failed"
            ) from error

    def identity_rows(self) -> dict[str, dict[str, object]]:
        """Return the four accepted identities bound by formal-launch v3."""

        self.recheck()
        return {
            "claim": self._captures["claim"].absolute_row(),
            "claim_marker": self._captures["claim_marker"].absolute_row(),
            "terminal": self._captures["terminal"].absolute_row(),
            "outer_completion": self._captures["completion"].absolute_row(),
        }

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_type, traceback
        evidence_error: RehearsalEvidenceError | None = None
        try:
            if self._state == "active":
                self.recheck()
        except RehearsalEvidenceError as error:
            evidence_error = error
        finally:
            self._close()
        if evidence_error is not None:
            if exc_value is not None:
                exc_value.add_note(
                    "accepted-binding rehearsal custody also failed on exit: "
                    f"{evidence_error}"
                )
            else:
                raise evidence_error
        return False


def hold_accepted_binding_rehearsal(
    binding_path: Path,
) -> AcceptedBindingRehearsalCustody:
    """Create a single-use held-custody context for one accepted rehearsal."""

    return AcceptedBindingRehearsalCustody(binding_path)


def accepted_binding_rehearsal_identity_rows(
    binding_path: Path,
) -> dict[str, dict[str, object]]:
    """Return accepted claim/terminal identities for formal-launch v3."""

    with hold_accepted_binding_rehearsal(binding_path) as custody:
        return custody.identity_rows()


accepted_binding_rehearsal_identity = accepted_binding_rehearsal_identity_rows
# Compatibility for the pre-seal runbook draft.  New consumers should use the
# fully qualified accepted-binding name above.
accepted_rehearsal_identity = accepted_binding_rehearsal_identity_rows


__all__ = [
    "CLAIM_NAME",
    "CLAIM_PATH",
    "CLAIM_SCHEMA",
    "FORMAL_BINDING_PATH",
    "LAUNCH_BOOTSTRAP_PATH",
    "OPERATOR_RESULTS_ROOT",
    "OUTER_COMPLETIONS_ROOT",
    "OUTER_RECEIPT_NAME",
    "OUTER_RECEIPT_PATH",
    "PROTOCOL_VERSION",
    "PRODUCER_BOOTSTRAP_PATH",
    "REHEARSAL_ATTEMPT_PATH",
    "REHEARSAL_ATTEMPTS_ROOT",
    "REHEARSAL_CLAIM_ROOT",
    "RESULTS_ROOT",
    "STDERR_NAME",
    "STDERR_PATH",
    "STDOUT_NAME",
    "STDOUT_PATH",
    "TERMINAL_NAME",
    "TERMINAL_PATH",
    "TERMINAL_SCHEMA",
    "AcceptedBindingRehearsalCustody",
    "RehearsalEvidenceError",
    "accepted_binding_rehearsal_identity",
    "accepted_binding_rehearsal_identity_rows",
    "accepted_rehearsal_identity",
    "hold_accepted_binding_rehearsal",
    "rehearsal_claim_marker",
    "rehearsal_outer_completion",
    "verify_accepted_binding_rehearsal",
]
