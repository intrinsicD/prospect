"""One-shot terminal adjudication for the sole formal WM-001 v1.6 attempt.

The adjudicator consumes the canonical formal operator audit attempt.  A
finalized accepted/rejected audit attempt receives exactly one bound replay.
An authenticated operator failure, including a child-published attempt whose
outer completion is absent, receives no replay and is terminally rejected.

The version-scoped adjudication claim is a no-replace hardlink.  It is
published only after every pre-claim check and immediately before the sole
replay or failure-package action.  Once that claim exists, protocol 1.6 is
consumed even if the process crashes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from . import operator as operator_module
from .artifact import (
    FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
    FORMAL_BINDING_OUTER_COMPLETION_NAME,
    FORMAL_LAUNCH_MARKER_NAME,
    verify_producer_manifest,
)
from .artifact import (
    MANIFEST_NAME as PRODUCER_MANIFEST_NAME,
)
from .assurance import assurance_record, require_assurance

HERE = Path(__file__).resolve().parent
REPO = operator_module.REPO
AUDITOR_SOURCE_PATH = HERE / "artifact_audit.py"
RUNNER_SOURCE_PATH = HERE / "audit_runner.py"
ADJUDICATOR_SOURCE_PATH = HERE / "adjudication.py"
LAUNCH_BOOTSTRAP_SOURCE_PATH = HERE / "launch_bootstrap.py"

ADJUDICATION_RESULTS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results" / "adjudication-v1.6"
FORMAL_ADJUDICATION_PACKAGE_PATH = ADJUDICATION_RESULTS_ROOT / "formal-adjudication-v1.6.0"
FORMAL_ADJUDICATION_CLAIM_MARKER = (
    REPO / "bench" / "world_model_lifecycle" / "results" / "formal" / "formal-adjudication-v1.6.0.json"
)
FORMAL_AUDIT_ATTEMPT_PATH = operator_module.FORMAL_AUDIT_ATTEMPT_PATH
FORMAL_SEMANTIC_REVIEW_PATH = REPO / "artifacts" / "wm001-reviews" / "formal-v1.6.0.json"

ADJUDICATION_MANIFEST_NAME = "adjudication-manifest.json"
ADJUDICATION_CLAIM_NAME = "formal-adjudication-claim.json"
ADJUDICATION_REPLAY_STARTED_NAME = "adjudication-replay-started.json"
ADJUDICATION_RECOVERY_FAILURE_NAME = "adjudication-recovery-failure.json"
COPIED_AUDIT_NAME = "independent-audit-report.json"
REPRODUCED_AUDIT_NAME = "reproduced-audit-report.json"
COPIED_SEMANTIC_REVIEW_NAME = "semantic-review.json"
AUDIT_RUNTIME_NAME = "audit-runtime-manifest.json"
AUDIT_INVOCATION_NAME = "audit-invocation-manifest.json"
BOUND_AUDIT_RUNTIME_NAME = "bound-audit-runtime-manifest.json"
BOUND_AUDIT_INVOCATION_NAME = "bound-audit-invocation-manifest.json"
AUDIT_STDERR_NAME = "audit-stderr.log"
AUDIT_EXECUTION_NAME = "audit-execution.json"
AUDIT_FAILURE_NAME = "audit-replay-failure.json"
PARTIAL_REPLAY_STDOUT_NAME = "audit-replay-partial.stdout"
PARTIAL_REPLAY_STDERR_NAME = "audit-replay-partial.stderr"
PARTIAL_REPLAY_RUNTIME_NAME = "audit-replay-partial.runtime.json"
PARTIAL_REPLAY_INVOCATION_NAME = "audit-replay-partial.invocation.json"
INPUT_FAILURE_NAME = "formal-audit-input-failure.json"
COPIED_LAUNCH_NAME = "formal-launch.json"
COPIED_BOOTSTRAP_NAME = "audit-bootstrap.py"
COPIED_OPERATOR_PREFIX = "formal-audit-attempt--"
COPIED_SOURCE_PREFIX = "bound-source--"

_PACKAGE_SCHEMA = "prospect.wm001.adjudication-package.v8"
_CLAIM_SCHEMA = "prospect.wm001.formal-adjudication-claim.v2"
_EXECUTION_SCHEMA = "prospect.wm001.adjudication-audit-execution.v2"
_REPLAY_FAILURE_SCHEMA = "prospect.wm001.adjudication-replay-failure.v1"
_REPLAY_STARTED_SCHEMA = "prospect.wm001.adjudication-replay-started.v1"
_RECOVERY_FAILURE_SCHEMA = "prospect.wm001.adjudication-recovery-failure.v1"
_INPUT_FAILURE_SCHEMA = "prospect.wm001.adjudication-input-failure.v1"
_SEMANTIC_REVIEW_SCHEMA = "prospect.wm001.semantic-review.v2"
_MAX_CONTROL_BYTES = 64 << 20
_MAX_RESULT_BYTES = 4 << 30
_MAX_AUDIT_BYTES = 64 << 20
_MAX_REVIEW_BYTES = 16 << 20
_MAX_STDERR_BYTES = 16 << 20
_SHA256_LENGTH = 64
_GATES = tuple(f"K{index}" for index in range(8))
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FAILURE_CODES = frozenset(
    {
        "audit-timeout",
        "audit-output-limit",
        "audit-bootstrap-rejected",
        "audit-runtime-rejected",
        "audit-runner-rejected",
        "audit-io-failure",
        "audit-runner-contract-violation",
        "audit-execution-identity-mismatch",
    }
)
_RECOVERY_PHASES = frozenset(
    {
        "post_claim_link",
        "post_claim_hook",
        "replay_staging",
        "replay_execution",
        "terminal_staging",
        "pre_package_publish",
        "post_package_publish",
        "outer_registration",
        "interrupted_after_claim",
    }
)

Disposition = Literal["accepted", "rejected"]
OutcomeKind = Literal[
    "audit_report",
    "audit_replay_mismatch",
    "adjudication_replay_failure",
    "formal_audit_execution_failure",
    "adjudication_recovery_failure",
]
ReviewRole = Literal[
    "supplied_audit_review",
    "pre_replay_supplied_audit_review",
    "execution_failure_review",
]


class AdjudicationError(ValueError):
    """Evidence cannot enter the sole terminal WM-001 v1.6 adjudication."""


class _AdjudicationRetired(AdjudicationError):
    """The version-scoped adjudication claim is already consumed."""


class _StoreOnce(argparse.Action):
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _parse_canonical_json_object(
    payload: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise AdjudicationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(raw: str) -> object:
        raise AdjudicationError(f"{label} contains non-finite value {raw}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdjudicationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise AdjudicationError(f"{label} is not one canonical JSON object followed by LF")
    return cast(dict[str, Any], value)


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


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _require_lexical_absolute(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.name in {"", ".", ".."}:
        raise AdjudicationError(f"{label} must be one canonical absolute path")
    return path


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise AdjudicationError(f"{label} cannot be resolved safely") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AdjudicationError(f"{label} contains a symbolic-link component")


def _canonical_directory(path: Path, *, label: str) -> Path:
    candidate = _require_lexical_absolute(path, label=label)
    _reject_symlink_components(candidate, label=label)
    try:
        metadata = os.stat(candidate, follow_symlinks=False)
    except OSError as error:
        raise AdjudicationError(f"{label} is not accessible") from error
    if not stat.S_ISDIR(metadata.st_mode) or candidate.resolve(strict=True) != candidate:
        raise AdjudicationError(f"{label} is not one canonical directory")
    return candidate


def _descriptor_path(descriptor: int) -> Path:
    for prefix in ("/proc/self/fd", "/dev/fd"):
        candidate = Path(prefix) / str(descriptor)
        if candidate.exists():
            return candidate
    raise AdjudicationError("platform cannot recheck an open descriptor")


@dataclass
class _OpenFile:
    path: Path
    descriptor: int
    identity: tuple[int, ...]
    payload: bytes
    label: str

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        label: str,
        limit: int,
        expected_links: int,
    ) -> _OpenFile:
        candidate = _require_lexical_absolute(path, label=label)
        _reject_symlink_components(candidate, label=label)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError as error:
            raise AdjudicationError(f"{label} cannot be opened safely") from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != expected_links
                or before.st_size < 0
                or before.st_size > limit
            ):
                raise AdjudicationError(f"{label} has invalid file custody")
            payload = os.pread(descriptor, before.st_size + 1, 0)
            after = os.fstat(descriptor)
            if (
                len(payload) != before.st_size
                or _stat_identity(before) != _stat_identity(after)
                or candidate.resolve(strict=True) != candidate
            ):
                raise AdjudicationError(f"{label} changed while captured")
            return cls(
                path=candidate,
                descriptor=descriptor,
                identity=_stat_identity(before),
                payload=payload,
                label=label,
            )
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def sha256(self) -> str:
        return _sha256(self.payload)

    def row(self, *, path: str | None = None) -> dict[str, object]:
        return {
            "path": path if path is not None else str(self.path),
            "bytes": len(self.payload),
            "sha256": self.sha256,
        }

    def recheck(self, *, path: Path | None = None) -> None:
        candidate = self.path if path is None else path
        try:
            descriptor_before = os.fstat(self.descriptor)
            current = os.pread(
                self.descriptor,
                descriptor_before.st_size + 1,
                0,
            )
            descriptor_after = os.fstat(self.descriptor)
            path_metadata = os.stat(candidate, follow_symlinks=False)
            same_file = os.path.samefile(
                candidate,
                _descriptor_path(self.descriptor),
            )
        except OSError as error:
            raise AdjudicationError(f"{self.label} cannot be rechecked") from error
        if (
            _stat_identity(descriptor_before) != self.identity
            or _stat_identity(descriptor_after) != self.identity
            or _stat_identity(path_metadata) != self.identity
            or current != self.payload
            or not same_file
        ):
            raise AdjudicationError(f"{self.label} changed while adjudication was open")

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass


def _read_regular_file(
    path: Path,
    *,
    limit: int,
    label: str,
    expected_links: int = 1,
) -> bytes:
    capture = _OpenFile.open(
        path,
        label=label,
        limit=limit,
        expected_links=expected_links,
    )
    try:
        return capture.payload
    finally:
        capture.close()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"could not write adjudication evidence {path.name}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(staging: Path, destination: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as error:
        raise AdjudicationError("platform has no atomic no-replace directory publication") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(staging),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to replace adjudication package: {destination}")
    raise OSError(
        number,
        f"atomic adjudication publication failed: {os.strerror(number)}",
        destination,
    )


def _safe_package_filename(name: object, *, label: str) -> str:
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or Path(name).name != name or name.startswith("."):
        raise AdjudicationError(f"{label} is not a safe package filename")
    return name


@dataclass(frozen=True)
class _Upstream:
    root: Path
    producer_manifest: dict[str, Any]
    producer_manifest_payload: bytes
    result: dict[str, Any]
    result_payload: bytes
    binding: dict[str, Any]
    binding_payload: bytes
    launch: dict[str, Any]
    launch_payload: bytes
    audit_execution: dict[str, Any]
    outcome_runtime_payload: bytes
    outcome_invocation_payload: bytes
    source_payloads: dict[str, bytes]
    bootstrap_payload: bytes


def _manifested_digest(
    manifest: Mapping[str, object],
    *,
    filename: str,
    payload: bytes,
) -> None:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise AdjudicationError("producer manifest file rows are malformed")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("path") == filename]
    if len(matches) != 1:
        raise AdjudicationError(f"producer manifest does not contain exactly one {filename!r}")
    row = matches[0]
    if row.get("bytes") != len(payload) or row.get("sha256") != _sha256(payload):
        raise AdjudicationError(f"producer manifest does not bind exact {filename} bytes")


def _binding_evidence(
    root: Path,
    producer_manifest: Mapping[str, object],
    audit_execution: Mapping[str, object],
    *,
    prefix: str,
) -> bytes:
    filename = audit_execution.get(f"{prefix}_file")
    expected_bytes = audit_execution.get(f"{prefix}_bytes")
    expected_sha256 = audit_execution.get(f"{prefix}_sha256")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or type(expected_bytes) is not int
        or cast(int, expected_bytes) < 1
        or not _is_sha256(expected_sha256)
    ):
        raise AdjudicationError(f"bound {prefix} identity is malformed")
    payload = _read_regular_file(
        root / filename,
        limit=_MAX_CONTROL_BYTES,
        label=f"bound {prefix}",
    )
    if len(payload) != expected_bytes or _sha256(payload) != expected_sha256:
        raise AdjudicationError(f"bound {prefix} bytes changed")
    _manifested_digest(
        producer_manifest,
        filename=filename,
        payload=payload,
    )
    return payload


def _verify_formal_result(
    result_path: Path,
    binding_path: Path,
) -> dict[str, Any]:
    from .verify import verify_result

    return cast(dict[str, Any], verify_result(result_path, binding_path))


def _verify_formal_binding(binding_path: Path) -> dict[str, Any]:
    from .verify import verify_binding

    return cast(dict[str, Any], verify_binding(binding_path))


def _verify_launch(
    *,
    root: Path,
    producer_manifest: Mapping[str, object],
    result: Mapping[str, object],
    binding_payload: bytes,
) -> tuple[dict[str, Any], bytes]:
    launch_path = root / "formal-launch.json"
    launch_payload = _read_regular_file(
        launch_path,
        limit=_MAX_CONTROL_BYTES,
        label="formal launch record",
        expected_links=2,
    )
    launch = _parse_canonical_json_object(
        launch_payload,
        label="formal launch record",
    )
    expected_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "formal_binding_sha256",
        "formal_binding_attempt_path",
        "formal_binding_attempt_manifest_file",
        "formal_binding_attempt_manifest_sha256",
        "formal_binding_outer_completion_file",
        "formal_binding_outer_completion_marker",
        "formal_binding_outer_completion_sha256",
        "attempt_directory",
        "global_marker_file",
        "claimed_at_utc",
        "git_commit",
        "git_tree",
        "record_sha256",
    }
    body = dict(launch)
    record_sha256 = body.pop("record_sha256", None)
    execution = result.get("execution")
    binding_sha256 = _sha256(binding_payload)
    marker = root.parent.parent / FORMAL_LAUNCH_MARKER_NAME
    marker_payload = _read_regular_file(
        marker,
        limit=_MAX_CONTROL_BYTES,
        label="protocol-wide formal launch marker",
        expected_links=2,
    )
    try:
        same_inode = os.path.samefile(marker, launch_path)
    except OSError as error:
        raise AdjudicationError("formal launch and global marker cannot be compared") from error
    binding_attempt = operator_module.FORMAL_BINDING_ATTEMPT_PATH
    binding_attempt_terminal = binding_attempt / "operator-attempt.json"
    binding_attempt_completion = operator_module.outer_completion_marker(binding_attempt_terminal)
    try:
        attempt_manifest = operator_module.verify_operator_attempt(binding_attempt)
        completion_identity = operator_module.verify_outer_completion(binding_attempt_terminal)
        canonical_binding_payload = _read_regular_file(
            binding_attempt / "formal-binding.json",
            limit=_MAX_CONTROL_BYTES,
            label="canonical accepted formal binding",
        )
        binding_attempt_payload = _read_regular_file(
            binding_attempt_terminal,
            limit=_MAX_CONTROL_BYTES,
            label="canonical formal binding attempt manifest",
            expected_links=2,
        )
        binding_completion_payload = _read_regular_file(
            binding_attempt_completion,
            limit=_MAX_CONTROL_BYTES,
            label="canonical formal binding outer completion",
            expected_links=2,
        )
        binding_same_inode = os.path.samefile(
            binding_attempt_terminal,
            binding_attempt_completion,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise AdjudicationError("formal launch has no canonical accepted binding attempt") from error
    copied_attempt_payload = _read_regular_file(
        root / FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
        limit=_MAX_CONTROL_BYTES,
        label="copied formal binding attempt manifest",
    )
    copied_completion_payload = _read_regular_file(
        root / FORMAL_BINDING_OUTER_COMPLETION_NAME,
        limit=_MAX_CONTROL_BYTES,
        label="copied formal binding outer completion",
    )
    attempt_primary = attempt_manifest.get("primary")
    if (
        set(launch) != expected_fields
        or launch.get("schema") != "prospect.wm001.formal-launch.v2"
        or launch.get("experiment_id") != "WM-001"
        or launch.get("protocol_version") != "1.6.0"
        or launch.get("formal_binding_sha256") != binding_sha256
        or canonical_binding_payload != binding_payload
        or launch.get("formal_binding_attempt_path") != str(binding_attempt)
        or launch.get("formal_binding_attempt_manifest_file") != FORMAL_BINDING_ATTEMPT_MANIFEST_NAME
        or launch.get("formal_binding_attempt_manifest_sha256") != _sha256(copied_attempt_payload)
        or launch.get("formal_binding_outer_completion_file") != FORMAL_BINDING_OUTER_COMPLETION_NAME
        or launch.get("formal_binding_outer_completion_marker") != str(binding_attempt_completion)
        or launch.get("formal_binding_outer_completion_sha256") != _sha256(copied_completion_payload)
        or copied_attempt_payload != copied_completion_payload
        or copied_attempt_payload != binding_attempt_payload
        or binding_attempt_payload != binding_completion_payload
        or not binding_same_inode
        or attempt_manifest.get("kind") != "binding"
        or attempt_manifest.get("lane") is not None
        or attempt_manifest.get("status") != "accepted"
        or not isinstance(attempt_primary, Mapping)
        or attempt_primary.get("binding_file") != "formal-binding.json"
        or completion_identity.get("terminal_sha256") != _sha256(binding_attempt_payload)
        or completion_identity.get("marker_path") != str(binding_attempt_completion)
        or root.parent.name != binding_sha256
        or root.parent.parent.name != "formal"
        or launch.get("attempt_directory") != root.name
        or launch.get("global_marker_file") != FORMAL_LAUNCH_MARKER_NAME
        or not isinstance(launch.get("claimed_at_utc"), str)
        or not cast(str, launch["claimed_at_utc"])
        or not isinstance(execution, Mapping)
        or launch.get("git_commit") != execution.get("git_commit")
        or launch.get("git_tree") != execution.get("git_tree")
        or execution.get("formal_launch_file") != "formal-launch.json"
        or execution.get("formal_launch_sha256") != _sha256(launch_payload)
        or record_sha256 != _sha256(_canonical_json_bytes(body)[:-1])
        or marker_payload != launch_payload
        or not same_inode
    ):
        raise AdjudicationError("formal result does not bind the unique v1.6 launch record")
    _manifested_digest(
        producer_manifest,
        filename="formal-launch.json",
        payload=launch_payload,
    )
    _manifested_digest(
        producer_manifest,
        filename=FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
        payload=copied_attempt_payload,
    )
    _manifested_digest(
        producer_manifest,
        filename=FORMAL_BINDING_OUTER_COMPLETION_NAME,
        payload=copied_completion_payload,
    )
    return launch, launch_payload


def _source_snapshot(
    *,
    root: Path,
    producer_manifest: Mapping[str, object],
    filename: str,
    expected_sha256: object,
) -> bytes:
    safe_name = _safe_package_filename(
        filename,
        label="bound execution-source name",
    )
    if not safe_name.endswith(".py") or not _is_sha256(expected_sha256):
        raise AdjudicationError(f"bound {safe_name} execution-source identity is malformed")
    relative = f"source/bench/world_model_lifecycle/{safe_name}"
    payload = _read_regular_file(
        root / relative,
        limit=_MAX_CONTROL_BYTES,
        label=f"bound {safe_name} source snapshot",
    )
    if _sha256(payload) != expected_sha256:
        raise AdjudicationError(f"bound {safe_name} source snapshot changed")
    _manifested_digest(
        producer_manifest,
        filename=relative,
        payload=payload,
    )
    return payload


def _load_upstream(
    producer_root: Path,
    *,
    require_live_sources: bool,
) -> _Upstream:
    root = _canonical_directory(producer_root, label="formal producer root")
    verified_manifest = verify_producer_manifest(root)
    producer_manifest_payload = _read_regular_file(
        root / PRODUCER_MANIFEST_NAME,
        limit=_MAX_CONTROL_BYTES,
        label="producer manifest",
        expected_links=2,
    )
    producer_manifest = _parse_canonical_json_object(
        producer_manifest_payload,
        label="producer manifest",
    )
    if (
        producer_manifest != verified_manifest
        or producer_manifest.get("status") != "completed"
        or producer_manifest.get("lane") != "formal"
        or producer_manifest.get("error") is not None
    ):
        raise AdjudicationError("adjudication requires one completed formal producer")

    result_path = root / "result.json"
    binding_path = root / "formal-binding.json"
    result_payload = _read_regular_file(
        result_path,
        limit=_MAX_RESULT_BYTES,
        label="formal raw result",
    )
    result = _parse_canonical_json_object(
        result_payload,
        label="formal raw result",
    )
    binding_payload = _read_regular_file(
        binding_path,
        limit=_MAX_CONTROL_BYTES,
        label="formal binding",
    )
    binding = _parse_canonical_json_object(
        binding_payload,
        label="formal binding",
    )
    verified_binding = _verify_formal_binding(binding_path)
    verified_result = _verify_formal_result(result_path, binding_path)
    binding_sha256 = _sha256(binding_payload)
    _manifested_digest(
        producer_manifest,
        filename="result.json",
        payload=result_payload,
    )
    _manifested_digest(
        producer_manifest,
        filename="formal-binding.json",
        payload=binding_payload,
    )
    protocol = binding.get("protocol")
    if (
        binding != verified_binding
        or result != verified_result
        or result.get("schema") != "prospect.world-model-lifecycle.raw-result.v6"
        or result.get("experiment_id") != "WM-001"
        or result.get("protocol_version") != "1.6.0"
        or result.get("lane") != "formal"
        or result.get("claim_eligible") is not True
        or result.get("formal_binding_sha256") != binding_sha256
        or binding.get("schema") != "prospect.world-model-lifecycle.formal-binding.v6"
        or binding.get("experiment_id") != "WM-001"
        or not isinstance(protocol, Mapping)
        or protocol.get("version") != "1.6.0"
        or protocol.get("sha256") != result.get("protocol_sha256")
    ):
        raise AdjudicationError("formal result, binding, and protocol identities do not agree")
    launch, launch_payload = _verify_launch(
        root=root,
        producer_manifest=producer_manifest,
        result=result,
        binding_payload=binding_payload,
    )

    source = binding.get("source")
    execution_sources = source.get("execution_source_sha256") if isinstance(source, Mapping) else None
    audit_execution = binding.get("audit_execution")
    dependencies = binding.get("dependencies")
    required_sources = {
        "audit_runner.py",
        "artifact_audit.py",
        "adjudication.py",
        "binding.py",
        "launch_bootstrap.py",
        "operator.py",
        "preformal.py",
        "producer_bootstrap.py",
        "run.py",
        "verify.py",
    }
    if (
        not isinstance(execution_sources, Mapping)
        or set(execution_sources) != required_sources
        or not isinstance(audit_execution, Mapping)
        or not isinstance(dependencies, Mapping)
        or audit_execution.get("outcome_source_mode") != "descriptor"
        or audit_execution.get("outcome_argv_role") != ["<canonical-producer-root>"]
        or audit_execution.get("interpreter_flags") != ["-I", "-S", "-B"]
        or audit_execution.get("passed") is not True
    ):
        raise AdjudicationError("formal binding has no valid bound outcome-audit execution")

    source_payloads = {
        filename: _source_snapshot(
            root=root,
            producer_manifest=producer_manifest,
            filename=filename,
            expected_sha256=digest,
        )
        for filename, digest in sorted(execution_sources.items())
    }
    if (
        _sha256(source_payloads["audit_runner.py"]) != audit_execution.get("runner_source_sha256")
        or _sha256(source_payloads["artifact_audit.py"]) != audit_execution.get("auditor_source_sha256")
        or _sha256(source_payloads["adjudication.py"]) != audit_execution.get("adjudicator_source_sha256")
    ):
        raise AdjudicationError("bound runner, auditor, or adjudicator identity changed")
    if require_live_sources:
        for filename, expected in source_payloads.items():
            live = _read_regular_file(
                HERE / filename,
                limit=_MAX_CONTROL_BYTES,
                label=f"live {filename}",
            )
            if live != expected:
                raise AdjudicationError(f"live {filename} differs from the pre-outcome binding")

    bootstrap_payload = _binding_evidence(
        root,
        producer_manifest,
        audit_execution,
        prefix="bootstrap_source",
    )
    outcome_runtime_payload = _binding_evidence(
        root,
        producer_manifest,
        audit_execution,
        prefix="outcome_runtime_manifest",
    )
    runtime = _parse_canonical_json_object(
        outcome_runtime_payload,
        label="bound outcome runtime manifest",
    )
    runtime_source = runtime.get("source")
    runtime_limits = runtime.get("limits")
    if (
        _sha256(bootstrap_payload) != audit_execution.get("bootstrap_source_sha256")
        or not isinstance(runtime_source, Mapping)
        or runtime_source.get("mode") != "descriptor"
        or runtime_source.get("path") != "artifact_audit.py"
        or runtime_source.get("bytes") != len(source_payloads["artifact_audit.py"])
        or runtime_source.get("sha256") != _sha256(source_payloads["artifact_audit.py"])
        or runtime.get("bootstrap_sha256") != _sha256(bootstrap_payload)
        or not isinstance(runtime.get("support_files"), list)
        or not isinstance(runtime_limits, Mapping)
        or runtime_limits.get("timeout_seconds") != 600
        or runtime_limits.get("stdout_bytes") != _MAX_AUDIT_BYTES
        or runtime_limits.get("stderr_bytes") != _MAX_STDERR_BYTES
    ):
        raise AdjudicationError("bound outcome runtime manifest has invalid execution custody")
    from .audit_runner import build_invocation_manifest

    working_directory = audit_execution.get("outcome_working_directory")
    if not isinstance(working_directory, str) or not Path(working_directory).is_absolute():
        raise AdjudicationError("bound outcome audit has no canonical working directory")
    outcome_invocation_payload = build_invocation_manifest(
        runtime_manifest=outcome_runtime_payload,
        auditor_arguments=(str(root),),
        working_directory=Path(working_directory),
    )
    if not isinstance(dependencies.get("python_executable"), str) or not _is_sha256(
        dependencies.get("python_executable_sha256")
    ):
        raise AdjudicationError("bound Python executable identity is malformed")
    return _Upstream(
        root=root,
        producer_manifest=producer_manifest,
        producer_manifest_payload=producer_manifest_payload,
        result=result,
        result_payload=result_payload,
        binding=binding,
        binding_payload=binding_payload,
        launch=launch,
        launch_payload=launch_payload,
        audit_execution=dict(audit_execution),
        outcome_runtime_payload=outcome_runtime_payload,
        outcome_invocation_payload=outcome_invocation_payload,
        source_payloads=source_payloads,
        bootstrap_payload=bootstrap_payload,
    )


def _upstream_identity(upstream: _Upstream) -> tuple[bytes, ...]:
    return (
        upstream.producer_manifest_payload,
        upstream.result_payload,
        upstream.binding_payload,
        upstream.launch_payload,
        upstream.outcome_runtime_payload,
        upstream.outcome_invocation_payload,
        upstream.bootstrap_payload,
        *(upstream.source_payloads[name] for name in sorted(upstream.source_payloads)),
    )


@dataclass
class _AuditAttempt:
    root: Path
    directory_descriptor: int
    directory_identity: tuple[int, ...]
    manifest: dict[str, object]
    outer_finalized: bool
    captures: dict[str, _OpenFile]

    @property
    def terminal(self) -> _OpenFile:
        return self.captures["operator-attempt.json"]

    @property
    def formal_claim(self) -> _OpenFile:
        return self.captures["formal-audit-claim.json"]

    def payload(self, name: str) -> bytes:
        try:
            return self.captures[name].payload
        except KeyError as error:
            raise AdjudicationError(f"formal audit attempt omits authenticated {name!r}") from error

    def file_rows(self) -> list[dict[str, object]]:
        return [capture.row(path=name) for name, capture in sorted(self.captures.items())]

    def recheck(self) -> None:
        try:
            descriptor_metadata = os.fstat(self.directory_descriptor)
            path_metadata = os.stat(self.root, follow_symlinks=False)
            names = sorted(entry.name for entry in os.scandir(self.root))
            same_directory = os.path.samefile(
                self.root,
                _descriptor_path(self.directory_descriptor),
            )
        except OSError as error:
            raise AdjudicationError("formal audit attempt directory cannot be rechecked") from error
        if (
            _directory_identity(descriptor_metadata) != self.directory_identity
            or _directory_identity(path_metadata) != self.directory_identity
            or names != sorted(self.captures)
            or not same_directory
        ):
            raise AdjudicationError("formal audit attempt directory changed during adjudication")
        for capture in self.captures.values():
            capture.recheck()
        terminal = self.root / "operator-attempt.json"
        marker = operator_module.outer_completion_marker(terminal)
        if self.outer_finalized:
            try:
                operator_module.verify_outer_completion(terminal)
            except (OSError, RuntimeError, ValueError) as error:
                raise AdjudicationError("formal audit attempt lost outer completion") from error
        elif os.path.lexists(marker):
            raise AdjudicationError("formal audit attempt outer-completion state changed")

    def close(self) -> None:
        for capture in reversed(tuple(self.captures.values())):
            capture.close()
        try:
            os.close(self.directory_descriptor)
        except OSError:
            pass


def _load_audit_attempt(path: Path) -> _AuditAttempt:
    candidate = _require_lexical_absolute(path, label="formal audit attempt")
    if candidate != FORMAL_AUDIT_ATTEMPT_PATH:
        raise AdjudicationError(
            f"adjudication requires the sole canonical formal audit attempt {FORMAL_AUDIT_ATTEMPT_PATH}"
        )
    root = _canonical_directory(candidate, label="formal audit attempt")
    terminal = root / "operator-attempt.json"
    marker = operator_module.outer_completion_marker(terminal)
    outer_finalized = os.path.lexists(marker)
    try:
        if outer_finalized:
            manifest = operator_module.verify_operator_attempt(root)
        else:
            inspected = operator_module.inspect_unfinalized_operator_attempt(root)
            raw_manifest = inspected.get("manifest")
            if not isinstance(raw_manifest, dict):
                raise AdjudicationError("unfinalized operator inspection returned no manifest")
            manifest = raw_manifest
    except (OSError, RuntimeError, ValueError) as error:
        raise AdjudicationError("formal audit attempt failed operator authentication") from error
    primary = manifest.get("primary")
    rows = manifest.get("files")
    try:
        require_assurance(
            manifest.get("assurance"),
            label="formal audit attempt",
        )
    except ValueError as error:
        raise AdjudicationError(str(error)) from error
    if (
        manifest.get("schema") != "prospect.wm001.operator-attempt.v1"
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("protocol_version") != "1.6.0"
        or manifest.get("kind") != "audit"
        or manifest.get("lane") != "formal"
        or manifest.get("status") not in {"accepted", "rejected", "failure"}
        or not isinstance(primary, dict)
        or not isinstance(rows, list)
    ):
        raise AdjudicationError("operator evidence is not the formal v1.6 audit attempt")
    names: list[str] = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not _SAFE_NAME.fullmatch(cast(str, row["path"]))
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _is_sha256(row.get("sha256"))
        ):
            raise AdjudicationError("formal audit attempt file identity is malformed")
        names.append(cast(str, row["path"]))
    if names != sorted(set(names)) or "formal-audit-claim.json" not in names or "operator-attempt.json" in names:
        raise AdjudicationError("formal audit attempt file identities are not canonical")
    try:
        directory_descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise AdjudicationError("formal audit attempt directory cannot be held open") from error
    directory_identity = _directory_identity(os.fstat(directory_descriptor))
    captures: dict[str, _OpenFile] = {}
    try:
        for name in [*names, "operator-attempt.json"]:
            expected_links = (
                2
                if name == "formal-audit-claim.json"
                else 2
                if name == "operator-attempt.json" and outer_finalized
                else 1
            )
            captures[name] = _OpenFile.open(
                root / name,
                label=f"formal audit attempt {name}",
                limit=_MAX_AUDIT_BYTES,
                expected_links=expected_links,
            )
        parsed_manifest = _parse_canonical_json_object(
            captures["operator-attempt.json"].payload,
            label="formal audit attempt terminal manifest",
        )
        actual_rows = [captures[name].row(path=name) for name in sorted(names)]
        actual_names = sorted(entry.name for entry in os.scandir(root))
        if parsed_manifest != manifest or rows != actual_rows or actual_names != sorted(captures):
            raise AdjudicationError("formal audit attempt changed after operator authentication")
        attempt = _AuditAttempt(
            root=root,
            directory_descriptor=directory_descriptor,
            directory_identity=directory_identity,
            manifest=cast(dict[str, object], manifest),
            outer_finalized=outer_finalized,
            captures=captures,
        )
        attempt.recheck()
        return attempt
    except BaseException:
        for capture in captures.values():
            capture.close()
        os.close(directory_descriptor)
        raise


def _verify_acceptance_gates(result: Mapping[str, object]) -> None:
    rows = result.get("gate_results")
    if (
        not isinstance(rows, list)
        or len(rows) != len(_GATES)
        or any(not isinstance(row, Mapping) for row in rows)
        or [
            (
                cast(Mapping[str, object], row).get("gate"),
                cast(Mapping[str, object], row).get("passed"),
            )
            for row in rows
        ]
        != [(gate, True) for gate in _GATES]
    ):
        raise AdjudicationError("accepted adjudication requires K0 through K7 in order and passing")


def _verify_audit_identity(
    audit: Mapping[str, object],
    *,
    upstream: _Upstream,
) -> bool:
    if (
        audit.get("schema") != "prospect.world-model-lifecycle.artifact-audit.v2"
        or audit.get("artifact_root") != str(upstream.root)
        or audit.get("result_file") != "result.json"
        or audit.get("result_sha256") != _sha256(upstream.result_payload)
        or audit.get("lane") != "formal"
    ):
        raise AdjudicationError("independent audit identifies different formal evidence")
    counts = audit.get("check_counts")
    findings = audit.get("findings")
    gaps = audit.get("coverage_gaps")
    if (
        type(audit.get("integrity_passed")) is not bool
        or type(audit.get("engineering_complete")) is not bool
        or type(audit.get("complete_for_claim")) is not bool
        or type(audit.get("passed")) is not bool
        or not isinstance(counts, Mapping)
        or type(counts.get("passed")) is not int
        or cast(int, counts.get("passed")) < 1
        or type(counts.get("failed")) is not int
        or cast(int, counts.get("failed")) < 0
        or type(counts.get("coverage_gaps")) is not int
        or cast(int, counts.get("coverage_gaps")) < 0
        or not isinstance(findings, list)
        or not isinstance(gaps, list)
        or any(not isinstance(row, Mapping) for row in findings)
        or any(not isinstance(row, Mapping) for row in gaps)
    ):
        raise AdjudicationError("independent audit status block is malformed")
    failed = cast(int, counts["failed"])
    coverage_gaps = cast(int, counts["coverage_gaps"])
    integrity = cast(bool, audit["integrity_passed"])
    engineering = cast(bool, audit["engineering_complete"])
    complete = cast(bool, audit["complete_for_claim"])
    passed = cast(bool, audit["passed"])
    if (
        failed != len(findings)
        or coverage_gaps != len(gaps)
        or integrity != (failed == 0)
        or engineering != (coverage_gaps == 0)
        or complete != engineering
        or passed != (integrity and engineering)
    ):
        raise AdjudicationError("independent audit status block is internally inconsistent")
    custody = audit.get("custody")
    if (
        not isinstance(custody, Mapping)
        or custody.get("producer_manifest_checked") is not True
        or custody.get("producer_manifest_status") != "completed"
        or custody.get("producer_manifest_sha256") != _sha256(upstream.producer_manifest_payload)
    ):
        raise AdjudicationError("independent audit does not bind producer custody")
    implementation = audit.get("audit_implementation")
    coverage = upstream.binding.get("coverage_arithmetic")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(coverage, Mapping)
        or implementation.get("auditor_source_sha256") != _sha256(upstream.source_payloads["artifact_audit.py"])
        or implementation.get("bound_auditor_source_sha256") != upstream.audit_execution.get("auditor_source_sha256")
        or implementation.get("formal_test_report_sha256") != coverage.get("formal_test_report_sha256")
        or implementation.get("coverage_conformance_report_sha256") != coverage.get("conformance_report_sha256")
        or implementation.get("auditor_source_matches_binding") is not True
        or type(implementation.get("coverage_conformance_verified")) is not bool
        or type(implementation.get("audit_execution_conformance_verified")) is not bool
    ):
        raise AdjudicationError("independent audit implementation identity is invalid")
    return (
        passed
        and complete
        and cast(bool, implementation["coverage_conformance_verified"])
        and cast(bool, implementation["audit_execution_conformance_verified"])
    )


def _formal_claim_value(
    attempt: _AuditAttempt,
    *,
    upstream: _Upstream,
) -> dict[str, Any]:
    claim = _parse_canonical_json_object(
        attempt.formal_claim.payload,
        label="formal audit claim",
    )
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
        or claim.get("schema") != "prospect.wm001.formal-audit-claim.v1"
        or claim.get("experiment_id") != "WM-001"
        or claim.get("protocol_version") != "1.6.0"
        or claim.get("claim_status") != "consumed"
        or claim.get("attempt_path") != str(FORMAL_AUDIT_ATTEMPT_PATH)
        or claim.get("marker_path") != str(operator_module.FORMAL_AUDIT_CLAIM_MARKER)
        or claim.get("producer_root") != str(upstream.root)
        or claim.get("producer_manifest_sha256") != _sha256(upstream.producer_manifest_payload)
        or claim.get("raw_result_sha256") != _sha256(upstream.result_payload)
        or claim.get("formal_binding_sha256") != _sha256(upstream.binding_payload)
        or claim.get("launch_bootstrap_sha256") != _sha256(upstream.source_payloads["launch_bootstrap.py"])
    ):
        raise AdjudicationError("formal audit claim differs from its authenticated producer")
    try:
        same_inode = os.path.samefile(
            attempt.root / "formal-audit-claim.json",
            operator_module.FORMAL_AUDIT_CLAIM_MARKER,
        )
    except OSError as error:
        raise AdjudicationError("formal audit claim marker cannot be compared") from error
    if not same_inode:
        raise AdjudicationError("formal audit claim is not its version-scoped marker inode")
    return claim


def _validate_attempt_failure_execution(
    attempt: _AuditAttempt,
    *,
    filename: str,
    upstream: _Upstream,
) -> None:
    receipt = _parse_canonical_json_object(
        attempt.payload(filename),
        label="formal audit partial-failure receipt",
    )
    expected_fields = {
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
    names = {
        prefix: receipt.get(f"{prefix}_file")
        for prefix in (
            "stdout",
            "stderr",
            "runtime_manifest",
            "invocation_manifest",
        )
    }
    if any(not isinstance(value, str) for value in names.values()):
        raise AdjudicationError("formal audit partial-failure sidecar reference is malformed")
    payloads = {prefix: attempt.payload(cast(str, name)) for prefix, name in names.items()}
    dependencies = cast(Mapping[str, object], upstream.binding["dependencies"])
    expected_support = _parse_canonical_json_object(
        upstream.outcome_runtime_payload,
        label="bound outcome runtime manifest",
    ).get("support_files")
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != "prospect.wm001.captured-audit-execution-failure.v1"
        or not isinstance(receipt.get("phase"), str)
        or (receipt.get("returncode") is not None and type(receipt.get("returncode")) is not int)
        or receipt.get("source_mode") != "descriptor"
        or not isinstance(command, list)
        or len(command) != 5
        or any(not isinstance(argument, str) for argument in command)
        or command[:4]
        != [
            dependencies["python_executable"],
            "-I",
            "-S",
            "-B",
        ]
        or not (cast(str, command[4]).startswith("/proc/self/fd/") or cast(str, command[4]).startswith("/dev/fd/"))
        or payloads["runtime_manifest"] != upstream.outcome_runtime_payload
        or payloads["invocation_manifest"] != upstream.outcome_invocation_payload
        or len(payloads["stdout"]) > _MAX_AUDIT_BYTES
        or len(payloads["stderr"]) > _MAX_STDERR_BYTES
        or any(
            receipt.get(f"{prefix}_bytes") != len(payload) or receipt.get(f"{prefix}_sha256") != _sha256(payload)
            for prefix, payload in payloads.items()
        )
        or receipt.get("bootstrap_sha256") != _sha256(upstream.bootstrap_payload)
        or receipt.get("auditor_source_sha256") != _sha256(upstream.source_payloads["artifact_audit.py"])
        or receipt.get("support_files") != expected_support
    ):
        raise AdjudicationError("formal audit partial failure differs from the bound runtime identity")


def _validate_attempt_execution(
    attempt: _AuditAttempt,
    *,
    upstream: _Upstream,
) -> None:
    primary = cast(Mapping[str, object], attempt.manifest["primary"])
    executions = primary.get("executions")
    execution_failures = primary.get("execution_failures")
    if not isinstance(executions, list) or not isinstance(execution_failures, list):
        raise AdjudicationError("formal audit executions are malformed")
    status = attempt.manifest.get("status")
    expected_count = 1 if status in {"accepted", "rejected"} else None
    if (
        len(executions) > 1
        or len(executions) + len(execution_failures) > 1
        or (expected_count is not None and len(executions) != expected_count)
        or (status != "failure" and execution_failures)
    ):
        raise AdjudicationError("formal audit attempt did not execute exactly its permitted audit")
    for failure_name in execution_failures:
        if not isinstance(failure_name, str):
            raise AdjudicationError("formal audit partial-failure reference is malformed")
        _validate_attempt_failure_execution(
            attempt,
            filename=failure_name,
            upstream=upstream,
        )
    if not executions:
        return
    receipt_name = executions[0]
    if not isinstance(receipt_name, str):
        raise AdjudicationError("formal audit execution receipt is malformed")
    receipt = _parse_canonical_json_object(
        attempt.payload(receipt_name),
        label="formal audit execution receipt",
    )
    expected_fields = {
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
    dependencies = cast(Mapping[str, object], upstream.binding["dependencies"])
    command = receipt.get("command")
    runtime_name = receipt.get("runtime_manifest_file")
    invocation_name = receipt.get("invocation_manifest_file")
    stdout_name = receipt.get("stdout_file")
    stderr_name = receipt.get("stderr_file")
    if not all(isinstance(value, str) for value in (runtime_name, invocation_name, stdout_name, stderr_name)):
        raise AdjudicationError("formal audit execution references malformed sidecars")
    runtime = attempt.payload(cast(str, runtime_name))
    invocation = attempt.payload(cast(str, invocation_name))
    stdout = attempt.payload(cast(str, stdout_name))
    stderr = attempt.payload(cast(str, stderr_name))
    expected_support = _parse_canonical_json_object(
        upstream.outcome_runtime_payload,
        label="bound outcome runtime manifest",
    ).get("support_files")
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != "prospect.wm001.captured-audit-execution.v1"
        or receipt.get("source_mode") != "descriptor"
        or runtime != upstream.outcome_runtime_payload
        or invocation != upstream.outcome_invocation_payload
        or receipt.get("runtime_manifest_bytes") != len(runtime)
        or receipt.get("runtime_manifest_sha256") != _sha256(runtime)
        or receipt.get("invocation_manifest_bytes") != len(invocation)
        or receipt.get("invocation_manifest_sha256") != _sha256(invocation)
        or receipt.get("stdout_bytes") != len(stdout)
        or receipt.get("stdout_sha256") != _sha256(stdout)
        or receipt.get("stderr_bytes") != len(stderr)
        or receipt.get("stderr_sha256") != _sha256(stderr)
        or len(stderr) > _MAX_STDERR_BYTES
        or receipt.get("bootstrap_sha256") != _sha256(upstream.bootstrap_payload)
        or receipt.get("auditor_source_sha256") != _sha256(upstream.source_payloads["artifact_audit.py"])
        or receipt.get("support_files") != expected_support
        or not isinstance(command, list)
        or len(command) != 5
        or any(not isinstance(argument, str) for argument in command)
        or command[:4]
        != [
            dependencies["python_executable"],
            "-I",
            "-S",
            "-B",
        ]
        or not (cast(str, command[4]).startswith("/proc/self/fd/") or cast(str, command[4]).startswith("/dev/fd/"))
    ):
        raise AdjudicationError("formal audit execution differs from the bound runtime identity")
    report = _parse_canonical_json_object(
        stdout,
        label="formal audit execution stdout",
    )
    if (
        type(report.get("passed")) is not bool
        or receipt.get("passed") is not report.get("passed")
        or receipt.get("returncode") != (0 if report.get("passed") is True else 1)
    ):
        raise AdjudicationError("formal audit execution status differs from its report")


def _input_failure_record(
    attempt: _AuditAttempt,
    *,
    upstream: _Upstream,
) -> dict[str, object]:
    status = cast(str, attempt.manifest["status"])
    if attempt.outer_finalized and status != "failure":
        raise AdjudicationError("completed formal audit report has no input failure")
    primary = cast(Mapping[str, object], attempt.manifest["primary"])
    operator_failure_file = primary.get("execution_failure_file")
    operator_failure_sha256: str | None = None
    if operator_failure_file is not None:
        if not isinstance(operator_failure_file, str):
            raise AdjudicationError("operator execution-failure reference is malformed")
        operator_failure_sha256 = _sha256(attempt.payload(operator_failure_file))
    executions = primary.get("executions")
    execution_failures = primary.get("execution_failures")
    if not isinstance(executions, list) or not isinstance(execution_failures, list):
        raise AdjudicationError("partial execution references are malformed")
    partial_execution_rows: list[dict[str, object]] = []
    for value in executions:
        if not isinstance(value, str):
            raise AdjudicationError("partial execution receipt reference is malformed")
        partial_execution_rows.append(
            {
                "file": value,
                "sha256": _sha256(attempt.payload(value)),
            }
        )
    partial_failure_rows: list[dict[str, object]] = []
    for value in execution_failures:
        if not isinstance(value, str):
            raise AdjudicationError("partial execution-failure receipt reference is malformed")
        partial_failure_rows.append(
            {
                "file": value,
                "sha256": _sha256(attempt.payload(value)),
            }
        )
    audit_file = primary.get("audit_file")
    audit_sha256 = _sha256(attempt.payload(cast(str, audit_file))) if isinstance(audit_file, str) else None
    failure_code = "outer_completion_absent" if not attempt.outer_finalized else "operator_execution_failure"
    return {
        "schema": _INPUT_FAILURE_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "failure_code": failure_code,
        "terminal": True,
        "replay_performed": False,
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "audit_attempt_outer_finalized": attempt.outer_finalized,
        "audit_attempt_status": status,
        "expected_outer_completion_marker": str(
            operator_module.outer_completion_marker(attempt.root / "operator-attempt.json")
        ),
        "producer_root": str(upstream.root),
        "producer_manifest_sha256": _sha256(upstream.producer_manifest_payload),
        "result_sha256": _sha256(upstream.result_payload),
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "formal_audit_claim_sha256": attempt.formal_claim.sha256,
        "operator_failure_file": operator_failure_file,
        "operator_failure_sha256": operator_failure_sha256,
        "partial_audit_file": audit_file,
        "partial_audit_sha256": audit_sha256,
        "partial_execution_receipts": partial_execution_rows,
        "partial_execution_failures": partial_failure_rows,
        "attempt_files": attempt.file_rows(),
    }


@dataclass(frozen=True)
class _AttemptEvidence:
    kind: Literal["report", "execution_failure"]
    report_payload: bytes | None
    report: dict[str, Any] | None
    report_clean: bool | None
    failure_record: dict[str, object] | None


def _classify_attempt_evidence(
    attempt: _AuditAttempt,
    *,
    upstream: _Upstream,
) -> _AttemptEvidence:
    primary = cast(Mapping[str, object], attempt.manifest["primary"])
    producer_root = primary.get("producer_root")
    if producer_root != str(upstream.root):
        raise AdjudicationError("formal audit attempt identifies a different producer")
    _formal_claim_value(attempt, upstream=upstream)
    _validate_attempt_execution(attempt, upstream=upstream)
    status = attempt.manifest.get("status")
    if attempt.outer_finalized and status in {"accepted", "rejected"}:
        audit_file = primary.get("audit_file")
        if audit_file != "independent-audit.json":
            raise AdjudicationError("completed formal audit attempt has no canonical report")
        payload = attempt.payload("independent-audit.json")
        report = _parse_canonical_json_object(
            payload,
            label="formal operator audit report",
        )
        clean = _verify_audit_identity(report, upstream=upstream)
        if (status == "accepted") is not (report.get("passed") is True):
            raise AdjudicationError("formal audit attempt status differs from its report")
        return _AttemptEvidence(
            kind="report",
            report_payload=payload,
            report=report,
            report_clean=clean,
            failure_record=None,
        )
    return _AttemptEvidence(
        kind="execution_failure",
        report_payload=None,
        report=None,
        report_clean=None,
        failure_record=_input_failure_record(attempt, upstream=upstream),
    )


def _review_binding(
    attempt: _AuditAttempt,
    *,
    upstream: _Upstream,
    evidence: _AttemptEvidence,
) -> dict[str, object]:
    failure_sha256 = (
        _sha256(_canonical_json_bytes(evidence.failure_record)) if evidence.failure_record is not None else None
    )
    partial_audit_sha256 = (
        cast(dict[str, object], evidence.failure_record).get("partial_audit_sha256")
        if evidence.failure_record is not None
        else None
    )
    return {
        "schema": _SEMANTIC_REVIEW_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "evidence_kind": evidence.kind,
        "artifact_root": str(upstream.root),
        "result_sha256": _sha256(upstream.result_payload),
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "formal_audit_claim_sha256": attempt.formal_claim.sha256,
        "independent_audit_sha256": (
            _sha256(cast(bytes, evidence.report_payload))
            if evidence.report_payload is not None
            else partial_audit_sha256
        ),
        "execution_failure_sha256": failure_sha256,
    }


def inspect_adjudication_evidence(
    audit_attempt: Path,
) -> dict[str, object]:
    """Return exact semantic-review bindings without consuming adjudication.

    This read-only helper is the supported way for the independent reviewer to
    bind semantic-review v2 to either a report or canonical execution failure.
    It does not authorize acceptance and it does not publish a claim.
    """

    attempt = _load_audit_attempt(audit_attempt)
    try:
        primary = cast(Mapping[str, object], attempt.manifest["primary"])
        producer_value = primary.get("producer_root")
        if not isinstance(producer_value, str):
            raise AdjudicationError("formal audit attempt has no producer root")
        upstream = _load_upstream(
            Path(producer_value),
            require_live_sources=False,
        )
        evidence = _classify_attempt_evidence(
            attempt,
            upstream=upstream,
        )
        binding = _review_binding(
            attempt,
            upstream=upstream,
            evidence=evidence,
        )
        binding["reviewed_gates"] = list(_GATES) if evidence.kind == "report" else []
        binding["required_verdict"] = "accepted_or_rejected" if evidence.kind == "report" else "rejected"
        binding["execution_failure_record"] = evidence.failure_record
        return binding
    finally:
        attempt.close()


def _capture_semantic_review(
    path: Path,
    *,
    attempt: _AuditAttempt,
    upstream: _Upstream,
) -> _OpenFile:
    candidate = _require_lexical_absolute(path, label="semantic review")
    if candidate != FORMAL_SEMANTIC_REVIEW_PATH:
        raise AdjudicationError(
            "adjudication requires the sole canonical semantic review "
            f"{FORMAL_SEMANTIC_REVIEW_PATH}"
        )
    forbidden_roots = (
        attempt.root,
        upstream.root,
        FORMAL_ADJUDICATION_PACKAGE_PATH,
    )
    if any(candidate == root or candidate.is_relative_to(root) for root in forbidden_roots) or candidate in {
        FORMAL_ADJUDICATION_CLAIM_MARKER,
        operator_module.FORMAL_AUDIT_CLAIM_MARKER,
    }:
        raise AdjudicationError("semantic review must be independent of producer, attempt, and package")
    return _OpenFile.open(
        candidate,
        label="semantic review",
        limit=_MAX_REVIEW_BYTES,
        expected_links=1,
    )


def _verify_semantic_review(
    review: Mapping[str, object],
    *,
    attempt: _AuditAttempt,
    upstream: _Upstream,
    evidence: _AttemptEvidence,
    disposition: Disposition,
) -> None:
    expected_fields = {
        "schema",
        "experiment_id",
        "protocol_version",
        "assurance",
        "evidence_kind",
        "artifact_root",
        "result_sha256",
        "audit_attempt_path",
        "audit_attempt_manifest_sha256",
        "formal_audit_claim_sha256",
        "independent_audit_sha256",
        "execution_failure_sha256",
        "reviewer",
        "reviewed_gates",
        "verdict",
        "fatal_findings",
        "conclusion",
    }
    binding = _review_binding(
        attempt,
        upstream=upstream,
        evidence=evidence,
    )
    fatal = review.get("fatal_findings")
    try:
        require_assurance(review.get("assurance"), label="semantic review")
    except ValueError as error:
        raise AdjudicationError(str(error)) from error
    if (
        set(review) != expected_fields
        or any(review.get(field) != value for field, value in binding.items())
        or not isinstance(review.get("reviewer"), str)
        or not cast(str, review.get("reviewer"))
        or not isinstance(fatal, list)
        or any(not isinstance(row, Mapping) for row in fatal)
        or not isinstance(review.get("conclusion"), str)
        or not cast(str, review.get("conclusion"))
    ):
        raise AdjudicationError("semantic review v2 identity or assurance is invalid")
    if evidence.kind == "report":
        if (
            review.get("reviewed_gates") != list(_GATES)
            or review.get("verdict") != disposition
            or (disposition == "accepted" and bool(fatal))
            or (disposition == "rejected" and not fatal)
        ):
            raise AdjudicationError("report semantic review gates or verdict are invalid")
        if disposition == "accepted":
            if evidence.report_clean is not True:
                raise AdjudicationError("accepted review requires a claim-clean audit report")
            _verify_acceptance_gates(upstream.result)
    elif (
        evidence.kind != "execution_failure"
        or disposition != "rejected"
        or review.get("reviewed_gates") != []
        or review.get("verdict") != "rejected"
        or not fatal
    ):
        raise AdjudicationError("execution-failure review must be terminally rejected with a fatal finding")


@dataclass(frozen=True)
class _Replay:
    returncode: int
    command: tuple[str, ...]
    stdout: bytes
    stderr: bytes
    report: dict[str, object]
    runtime_manifest: bytes
    invocation_manifest: bytes
    bootstrap_sha256: str
    auditor_source_sha256: str
    support_files: list[dict[str, object]]
    source_mode: str


@dataclass(frozen=True)
class _PartialReplay:
    phase: str
    command: tuple[str, ...]
    returncode: int | None
    stdout: bytes
    stderr: bytes
    runtime_manifest: bytes
    invocation_manifest: bytes
    bootstrap_sha256: str
    auditor_source_sha256: str
    support_files: list[dict[str, object]]
    source_mode: str


@dataclass(frozen=True)
class _ExecutionFailure:
    code: str
    error_type: str
    partial: _PartialReplay | None = None


def _classify_failure(error: Exception) -> _ExecutionFailure:
    from .audit_runner import AuditRunnerError

    if isinstance(error, AuditRunnerError):
        text = str(error).lower()
        if "timed out" in text or "timeout" in text:
            code = "audit-timeout"
        elif "byte limit" in text or "file size limit" in text or "too large" in text:
            code = "audit-output-limit"
        elif "bootstrap" in text:
            code = "audit-bootstrap-rejected"
        elif "runtime manifest" in text or "invocation manifest" in text:
            code = "audit-runtime-rejected"
        else:
            code = "audit-runner-rejected"
    elif isinstance(error, OSError):
        code = "audit-io-failure"
    elif isinstance(error, RuntimeError):
        code = "audit-runtime-rejected"
    else:
        code = "audit-runner-contract-violation"
    return _ExecutionFailure(code=code, error_type=type(error).__name__)


def _capture_partial_replay(
    execution: object,
    *,
    phase: str,
) -> _PartialReplay | None:
    from .audit_runner import AuditExecution, AuditExecutionFailure

    if not isinstance(execution, (AuditExecution, AuditExecutionFailure)):
        return None
    stdout = execution.stdout
    stderr = execution.stderr
    runtime = execution.runtime_manifest
    invocation = execution.invocation_manifest
    if (
        len(stdout) > _MAX_AUDIT_BYTES
        or len(stderr) > _MAX_STDERR_BYTES
        or len(runtime) > _MAX_CONTROL_BYTES
        or len(invocation) > _MAX_CONTROL_BYTES
    ):
        return None
    return _PartialReplay(
        phase=phase,
        command=tuple(execution.command),
        returncode=execution.returncode,
        stdout=stdout,
        stderr=stderr,
        runtime_manifest=runtime,
        invocation_manifest=invocation,
        bootstrap_sha256=execution.bootstrap_sha256,
        auditor_source_sha256=execution.auditor_source_sha256,
        support_files=[
            {
                "path": row.relative_path,
                "bytes": row.bytes,
                "sha256": row.sha256,
            }
            for row in execution.support_files
        ],
        source_mode=execution.source_mode,
    )


def _validate_replay_execution(
    execution: object,
    *,
    upstream: _Upstream,
) -> _Replay:
    from .audit_runner import AuditExecution

    if not isinstance(execution, AuditExecution):
        raise AdjudicationError("bound auditor returned no authenticated AuditExecution")
    report = _parse_canonical_json_object(
        execution.stdout,
        label="fresh reproduced audit",
    )
    runtime = _parse_canonical_json_object(
        execution.runtime_manifest,
        label="fresh audit runtime manifest",
    )
    expected_support = runtime.get("support_files")
    support_files = [
        {
            "path": row.relative_path,
            "bytes": row.bytes,
            "sha256": row.sha256,
        }
        for row in execution.support_files
    ]
    dependencies = cast(Mapping[str, object], upstream.binding["dependencies"])
    expected_python = dependencies.get("python_executable")
    command = execution.command
    if (
        execution.source_mode != "descriptor"
        or execution.runtime_manifest != upstream.outcome_runtime_payload
        or execution.invocation_manifest != upstream.outcome_invocation_payload
        or execution.runtime_manifest_sha256 != _sha256(upstream.outcome_runtime_payload)
        or execution.invocation_manifest_sha256 != _sha256(upstream.outcome_invocation_payload)
        or execution.bootstrap_sha256 != _sha256(upstream.bootstrap_payload)
        or execution.auditor_source_sha256 != _sha256(upstream.source_payloads["artifact_audit.py"])
        or support_files != expected_support
        or len(execution.stderr) > _MAX_STDERR_BYTES
        or dict(execution.report) != report
        or type(report.get("passed")) is not bool
        or execution.returncode != (0 if report["passed"] is True else 1)
        or len(command) != 5
        or command[:4] != (expected_python, "-I", "-S", "-B")
        or not (command[4].startswith("/proc/self/fd/") or command[4].startswith("/dev/fd/"))
    ):
        raise AdjudicationError("fresh audit execution differs from its bound descriptor identity")
    return _Replay(
        returncode=execution.returncode,
        command=execution.command,
        stdout=execution.stdout,
        stderr=execution.stderr,
        report=report,
        runtime_manifest=execution.runtime_manifest,
        invocation_manifest=execution.invocation_manifest,
        bootstrap_sha256=execution.bootstrap_sha256,
        auditor_source_sha256=execution.auditor_source_sha256,
        support_files=support_files,
        source_mode=execution.source_mode,
    )


def _run_bound_replay(
    upstream: _Upstream,
) -> tuple[_Replay | None, _ExecutionFailure | None]:
    from .binding import run_bound_outcome_audit

    try:
        execution = run_bound_outcome_audit(upstream.root)
    except Exception as error:
        failure = _classify_failure(error)
        return None, _ExecutionFailure(
            code=failure.code,
            error_type=failure.error_type,
            partial=_capture_partial_replay(
                error,
                phase=getattr(error, "phase", "runner_failure"),
            ),
        )
    try:
        return _validate_replay_execution(execution, upstream=upstream), None
    except (AdjudicationError, OSError, RuntimeError, ValueError) as error:
        return None, _ExecutionFailure(
            code="audit-execution-identity-mismatch",
            error_type=type(error).__name__,
            partial=_capture_partial_replay(
                execution,
                phase="execution_identity_validation",
            ),
        )


def _execution_receipt(
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    replay: _Replay,
    supplied_audit_payload: bytes,
) -> dict[str, object]:
    dependencies = cast(Mapping[str, object], upstream.binding["dependencies"])
    return {
        "schema": _EXECUTION_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "producer_root": str(upstream.root),
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "source_mode": "descriptor",
        "interpreter": {
            "executable": dependencies["python_executable"],
            "sha256": dependencies["python_executable_sha256"],
            "flags": ["-I", "-S", "-B"],
        },
        "returncode": replay.returncode,
        "report_passed": replay.report["passed"],
        "stdout": {
            "file": REPRODUCED_AUDIT_NAME,
            "bytes": len(replay.stdout),
            "sha256": _sha256(replay.stdout),
        },
        "stderr": {
            "file": AUDIT_STDERR_NAME,
            "bytes": len(replay.stderr),
            "sha256": _sha256(replay.stderr),
        },
        "runtime_manifest": {
            "file": AUDIT_RUNTIME_NAME,
            "bytes": len(replay.runtime_manifest),
            "sha256": _sha256(replay.runtime_manifest),
        },
        "invocation_manifest": {
            "file": AUDIT_INVOCATION_NAME,
            "bytes": len(replay.invocation_manifest),
            "sha256": _sha256(replay.invocation_manifest),
        },
        "runner_source_sha256": _sha256(upstream.source_payloads["audit_runner.py"]),
        "bootstrap_source_sha256": replay.bootstrap_sha256,
        "auditor_source_sha256": replay.auditor_source_sha256,
        "adjudicator_source_sha256": _sha256(upstream.source_payloads["adjudication.py"]),
        "support_files": replay.support_files,
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "supplied_audit_sha256": _sha256(supplied_audit_payload),
        "byte_identical": replay.stdout == supplied_audit_payload,
        "execution_completed": True,
        "same_version_replay_permitted": False,
    }


def _replay_failure_record(
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    failure: _ExecutionFailure,
    supplied_audit_payload: bytes,
    requested_disposition: Disposition,
) -> dict[str, object]:
    partial = failure.partial
    partial_execution = (
        {
            "phase": partial.phase,
            "returncode": partial.returncode,
            "source_mode": partial.source_mode,
            "command": list(partial.command),
            "stdout": {
                "file": PARTIAL_REPLAY_STDOUT_NAME,
                "bytes": len(partial.stdout),
                "sha256": _sha256(partial.stdout),
            },
            "stderr": {
                "file": PARTIAL_REPLAY_STDERR_NAME,
                "bytes": len(partial.stderr),
                "sha256": _sha256(partial.stderr),
            },
            "runtime_manifest": {
                "file": PARTIAL_REPLAY_RUNTIME_NAME,
                "bytes": len(partial.runtime_manifest),
                "sha256": _sha256(partial.runtime_manifest),
            },
            "invocation_manifest": {
                "file": PARTIAL_REPLAY_INVOCATION_NAME,
                "bytes": len(partial.invocation_manifest),
                "sha256": _sha256(partial.invocation_manifest),
            },
            "bootstrap_sha256": partial.bootstrap_sha256,
            "auditor_source_sha256": partial.auditor_source_sha256,
            "support_files": partial.support_files,
        }
        if partial is not None
        else None
    )
    return {
        "schema": _REPLAY_FAILURE_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "producer_root": str(upstream.root),
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "failure_code": failure.code,
        "error_type": failure.error_type,
        "requested_disposition": requested_disposition,
        "supplied_audit_sha256": _sha256(supplied_audit_payload),
        "partial_execution": partial_execution,
        "runner_source_sha256": _sha256(upstream.source_payloads["audit_runner.py"]),
        "bootstrap_source_sha256": _sha256(upstream.bootstrap_payload),
        "auditor_source_sha256": _sha256(upstream.source_payloads["artifact_audit.py"]),
        "adjudicator_source_sha256": _sha256(upstream.source_payloads["adjudication.py"]),
        "outcome_runtime_manifest_sha256": _sha256(upstream.outcome_runtime_payload),
        "outcome_invocation_manifest_sha256": _sha256(upstream.outcome_invocation_payload),
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "execution_completed": (partial is not None and partial.returncode is not None),
        "terminal": True,
        "same_version_replay_permitted": False,
    }


def _partial_from_validated_replay(
    replay: _Replay,
    *,
    phase: str,
) -> _PartialReplay:
    return _PartialReplay(
        phase=phase,
        command=replay.command,
        returncode=replay.returncode,
        stdout=replay.stdout,
        stderr=replay.stderr,
        runtime_manifest=replay.runtime_manifest,
        invocation_manifest=replay.invocation_manifest,
        bootstrap_sha256=replay.bootstrap_sha256,
        auditor_source_sha256=replay.auditor_source_sha256,
        support_files=replay.support_files,
        source_mode=replay.source_mode,
    )


def _partial_replay_payloads(
    partial: _PartialReplay,
) -> dict[str, bytes]:
    return {
        PARTIAL_REPLAY_STDOUT_NAME: partial.stdout,
        PARTIAL_REPLAY_STDERR_NAME: partial.stderr,
        PARTIAL_REPLAY_RUNTIME_NAME: partial.runtime_manifest,
        PARTIAL_REPLAY_INVOCATION_NAME: partial.invocation_manifest,
    }


def _write_partial_replay_evidence(
    *,
    staging: Path,
    payloads: dict[str, bytes],
    failure: _ExecutionFailure,
) -> None:
    if failure.partial is None:
        return
    for name, payload in _partial_replay_payloads(failure.partial).items():
        payloads[name] = payload
        _write_private_file(staging / name, payload)


def _base_payloads(
    upstream: _Upstream,
    attempt: _AuditAttempt,
    review_payload: bytes,
) -> tuple[dict[str, bytes], list[dict[str, object]], list[dict[str, object]]]:
    payloads = {
        COPIED_SEMANTIC_REVIEW_NAME: review_payload,
        COPIED_LAUNCH_NAME: upstream.launch_payload,
        COPIED_BOOTSTRAP_NAME: upstream.bootstrap_payload,
        BOUND_AUDIT_RUNTIME_NAME: upstream.outcome_runtime_payload,
        BOUND_AUDIT_INVOCATION_NAME: upstream.outcome_invocation_payload,
    }
    source_rows: list[dict[str, object]] = []
    for source_name, payload in sorted(upstream.source_payloads.items()):
        package_name = f"{COPIED_SOURCE_PREFIX}{source_name}"
        payloads[package_name] = payload
        source_rows.append(
            {
                "source_file": source_name,
                "package_file": package_name,
                "sha256": _sha256(payload),
            }
        )
    attempt_rows: list[dict[str, object]] = []
    for source_name, capture in sorted(attempt.captures.items()):
        package_name = f"{COPIED_OPERATOR_PREFIX}{source_name}"
        payloads[package_name] = capture.payload
        attempt_rows.append(
            {
                "source_file": source_name,
                "package_file": package_name,
                "bytes": len(capture.payload),
                "sha256": capture.sha256,
            }
        )
    return payloads, source_rows, attempt_rows


def _prepare_output_paths() -> Path:
    package = _require_lexical_absolute(
        FORMAL_ADJUDICATION_PACKAGE_PATH,
        label="formal adjudication package",
    )
    marker = _require_lexical_absolute(
        FORMAL_ADJUDICATION_CLAIM_MARKER,
        label="formal adjudication claim marker",
    )
    if package.name != "formal-adjudication-v1.6.0" or marker.name != "formal-adjudication-v1.6.0.json":
        raise AdjudicationError("formal adjudication paths are not version-scoped to v1.6.0")
    package.parent.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    _canonical_directory(
        package.parent,
        label="formal adjudication package parent",
    )
    _canonical_directory(
        marker.parent,
        label="formal adjudication claim-marker parent",
    )
    if os.path.lexists(marker):
        raise _AdjudicationRetired("WM-001 protocol 1.6 adjudication claim is already consumed")
    if os.path.lexists(package):
        raise FileExistsError(f"refusing to replace formal adjudication package: {package}")
    return package


def _adjudication_claim_value(
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    review_sha256: str,
    requested_disposition: Disposition,
    formal_audit_claim: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": _CLAIM_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "claim_status": "consumed",
        "producer_root": str(upstream.root),
        "producer_manifest_sha256": _sha256(upstream.producer_manifest_payload),
        "result_sha256": _sha256(upstream.result_payload),
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "launch_bootstrap_sha256": formal_audit_claim["launch_bootstrap_sha256"],
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "semantic_review_path": str(FORMAL_SEMANTIC_REVIEW_PATH),
        "semantic_review_sha256": review_sha256,
        "requested_disposition": requested_disposition,
        "output_path": str(FORMAL_ADJUDICATION_PACKAGE_PATH),
        "marker_path": str(FORMAL_ADJUDICATION_CLAIM_MARKER),
    }


@dataclass
class _ClaimCustody:
    marker: _OpenFile
    payload: bytes

    def recheck(self, claim_path: Path) -> None:
        self.marker.recheck()
        try:
            same_inode = os.path.samefile(self.marker.path, claim_path)
        except OSError as error:
            raise AdjudicationError("adjudication claim and marker cannot be compared") from error
        if not same_inode:
            raise AdjudicationError("adjudication claim is not its version-scoped marker inode")

    def close(self) -> None:
        self.marker.close()


def _publish_adjudication_claim(
    *,
    staging: Path,
    value: Mapping[str, object],
    on_irreversible: Callable[[], None],
) -> _ClaimCustody:
    claim_path = staging / ADJUDICATION_CLAIM_NAME
    marker = FORMAL_ADJUDICATION_CLAIM_MARKER
    payload = _canonical_json_bytes(value)
    if os.path.lexists(marker):
        raise _AdjudicationRetired("WM-001 protocol 1.6 adjudication claim is already consumed")
    _write_private_file(claim_path, payload)
    try:
        os.link(claim_path, marker, follow_symlinks=False)
    except FileExistsError as error:
        raise _AdjudicationRetired("WM-001 protocol 1.6 adjudication claim is already consumed") from error
    except OSError as error:
        raise AdjudicationError("formal adjudication claim marker could not be published") from error
    on_irreversible()
    _after_adjudication_claim_link()
    _fsync_directory(staging)
    _fsync_directory(marker.parent)
    marker_capture = _OpenFile.open(
        marker,
        label="formal adjudication claim marker",
        limit=_MAX_CONTROL_BYTES,
        expected_links=2,
    )
    try:
        if marker_capture.payload != payload or not os.path.samefile(marker, claim_path):
            raise AdjudicationError("formal adjudication marker is not the staged claim inode")
        return _ClaimCustody(marker=marker_capture, payload=payload)
    except BaseException:
        marker_capture.close()
        raise


def _after_adjudication_claim() -> None:
    """Test hook at the irreversible claim boundary."""


def _after_adjudication_claim_link() -> None:
    """Test hook after link success but before claim publication returns."""


def _before_adjudication_replay(_staging: Path) -> None:
    """Test hook after durable replay-start evidence and before the runner."""


def _after_adjudication_replay(_staging: Path) -> None:
    """Test hook after the sole runner invocation returns."""


def _before_adjudication_claim() -> None:
    """Test hook after preclaim staging but before final input rechecks."""


def _before_package_publish(_staging: Path) -> None:
    """Test hook after staged descriptors are held and before rename."""


def _after_package_publish(_package: Path) -> None:
    """Test hook after the no-replace rename and before outer registration."""


def _before_outer_registration(_package: Path) -> None:
    """Test hook after exact final-byte verification and before registration."""


def _after_outer_registration(_package: Path) -> None:
    """Test hook after outer registration returns."""


def _file_rows(payloads: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
        for name, payload in sorted(payloads.items())
    ]


def _replay_started_record(
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    claim_payload: bytes,
) -> dict[str, object]:
    return {
        "schema": _REPLAY_STARTED_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "producer_root": str(upstream.root),
        "producer_manifest_sha256": _sha256(upstream.producer_manifest_payload),
        "result_sha256": _sha256(upstream.result_payload),
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "formal_audit_claim_sha256": attempt.formal_claim.sha256,
        "adjudication_claim_sha256": _sha256(claim_payload),
        "replay_status": "started",
        "same_version_replay_permitted": False,
    }


def _verify_replay_started(
    payload: bytes,
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    claim_payload: bytes,
) -> None:
    value = _parse_canonical_json_object(
        payload,
        label="adjudication replay-started receipt",
    )
    try:
        require_assurance(
            value.get("assurance"),
            label="adjudication replay-started receipt",
        )
    except ValueError as error:
        raise AdjudicationError(str(error)) from error
    if value != _replay_started_record(
        upstream=upstream,
        attempt=attempt,
        claim_payload=claim_payload,
    ):
        raise AdjudicationError("adjudication replay-started receipt differs from exact inputs")


def _recovery_failure_record(
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    claim_payload: bytes,
    requested_disposition: Disposition,
    recovery_mode: Literal["automatic", "explicit"],
    failure_phase: str,
    failure_type: str | None,
    prior_replay_state: Literal[
        "not_applicable",
        "not_observed",
        "started",
        "untrusted_or_partial",
    ],
    replay_started_payload: bytes | None,
) -> dict[str, object]:
    return {
        "schema": _RECOVERY_FAILURE_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "terminal": True,
        "disposition": "rejected",
        "producer_root": str(upstream.root),
        "producer_manifest_sha256": _sha256(upstream.producer_manifest_payload),
        "result_sha256": _sha256(upstream.result_payload),
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "formal_audit_claim_sha256": attempt.formal_claim.sha256,
        "adjudication_claim_sha256": _sha256(claim_payload),
        "requested_disposition": requested_disposition,
        "recovery_mode": recovery_mode,
        "failure_phase": failure_phase,
        "failure_type": failure_type,
        "prior_replay_state": prior_replay_state,
        "replay_started_sha256": (
            _sha256(replay_started_payload)
            if replay_started_payload is not None
            else None
        ),
        "recovery_replay_performed": False,
        "same_version_replay_permitted": False,
    }


def _manifest(
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    evidence: _AttemptEvidence,
    payloads: Mapping[str, bytes],
    source_rows: list[dict[str, object]],
    attempt_rows: list[dict[str, object]],
    claim_payload: bytes,
    requested_disposition: Disposition,
    disposition: Disposition,
    outcome_kind: OutcomeKind,
    review_role: ReviewRole,
    replay: _Replay | None,
) -> dict[str, object]:
    audit_payload = payloads.get(COPIED_AUDIT_NAME)
    reproduced_payload = payloads.get(REPRODUCED_AUDIT_NAME)
    execution_payload = payloads.get(AUDIT_EXECUTION_NAME)
    replay_failure_payload = payloads.get(AUDIT_FAILURE_NAME)
    input_failure_payload = payloads.get(INPUT_FAILURE_NAME)
    replay_started_payload = payloads.get(ADJUDICATION_REPLAY_STARTED_NAME)
    recovery_failure_payload = payloads.get(ADJUDICATION_RECOVERY_FAILURE_NAME)
    review_payload = payloads[COPIED_SEMANTIC_REVIEW_NAME]
    rows = _file_rows(payloads)
    return {
        "schema": _PACKAGE_SCHEMA,
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": assurance_record(),
        "lane": "formal",
        "requested_disposition": requested_disposition,
        "disposition": disposition,
        "outcome_kind": outcome_kind,
        "semantic_review_role": review_role,
        "terminal": True,
        "same_version_replay_permitted": False,
        "producer_root": str(upstream.root),
        "producer_manifest_sha256": _sha256(upstream.producer_manifest_payload),
        "result_sha256": _sha256(upstream.result_payload),
        "formal_binding_sha256": _sha256(upstream.binding_payload),
        "formal_launch_sha256": _sha256(upstream.launch_payload),
        "audit_attempt_path": str(attempt.root),
        "audit_attempt_manifest_sha256": attempt.terminal.sha256,
        "audit_attempt_outer_finalized": attempt.outer_finalized,
        "audit_attempt_status": attempt.manifest["status"],
        "audit_attempt_files": attempt_rows,
        "formal_audit_claim_sha256": attempt.formal_claim.sha256,
        "adjudication_claim_file": ADJUDICATION_CLAIM_NAME,
        "adjudication_claim_sha256": _sha256(claim_payload),
        "adjudication_claim_marker": str(FORMAL_ADJUDICATION_CLAIM_MARKER),
        "bound_source_files": source_rows,
        "bound_runtime_manifest_sha256": _sha256(upstream.outcome_runtime_payload),
        "bound_invocation_manifest_sha256": _sha256(upstream.outcome_invocation_payload),
        "audit_file": COPIED_AUDIT_NAME if audit_payload is not None else None,
        "audit_sha256": (_sha256(audit_payload) if audit_payload is not None else None),
        "supplied_audit_clean_for_claim": evidence.report_clean,
        "reproduced_audit_file": (REPRODUCED_AUDIT_NAME if reproduced_payload is not None else None),
        "reproduced_audit_sha256": (_sha256(reproduced_payload) if reproduced_payload is not None else None),
        "audit_execution_file": (AUDIT_EXECUTION_NAME if execution_payload is not None else None),
        "audit_execution_sha256": (_sha256(execution_payload) if execution_payload is not None else None),
        "audit_replay_failure_file": (AUDIT_FAILURE_NAME if replay_failure_payload is not None else None),
        "audit_replay_failure_sha256": (
            _sha256(replay_failure_payload) if replay_failure_payload is not None else None
        ),
        "input_failure_file": (INPUT_FAILURE_NAME if input_failure_payload is not None else None),
        "input_failure_sha256": (_sha256(input_failure_payload) if input_failure_payload is not None else None),
        "replay_started_file": (
            ADJUDICATION_REPLAY_STARTED_NAME
            if replay_started_payload is not None
            else None
        ),
        "replay_started_sha256": (
            _sha256(replay_started_payload)
            if replay_started_payload is not None
            else None
        ),
        "recovery_failure_file": (
            ADJUDICATION_RECOVERY_FAILURE_NAME
            if recovery_failure_payload is not None
            else None
        ),
        "recovery_failure_sha256": (
            _sha256(recovery_failure_payload)
            if recovery_failure_payload is not None
            else None
        ),
        "audit_execution_completed": (
            replay is not None
            or (
                replay_failure_payload is not None
                and _parse_canonical_json_object(
                    replay_failure_payload,
                    label="adjudication replay failure",
                ).get("execution_completed")
                is True
            )
        ),
        "audit_byte_identical": (
            replay.stdout == audit_payload if replay is not None and audit_payload is not None else None
        ),
        "semantic_review_file": COPIED_SEMANTIC_REVIEW_NAME,
        "semantic_review_source_path": str(FORMAL_SEMANTIC_REVIEW_PATH),
        "semantic_review_sha256": _sha256(review_payload),
        "files": rows,
        "file_count": len(rows),
        "manifest_excludes": [ADJUDICATION_MANIFEST_NAME],
    }


@dataclass
class _PackageCustody:
    root: Path
    directory_descriptor: int
    directory_identity: tuple[int, ...]
    captures: dict[str, _OpenFile]

    @classmethod
    def capture(
        cls,
        root: Path,
        *,
        terminal_links: int,
    ) -> _PackageCustody:
        package = _canonical_directory(root, label="staged adjudication package")
        try:
            directory_descriptor = os.open(
                package,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise AdjudicationError("adjudication package directory cannot be held open") from error
        captures: dict[str, _OpenFile] = {}
        try:
            terminal = _OpenFile.open(
                package / ADJUDICATION_MANIFEST_NAME,
                label="adjudication terminal manifest",
                limit=_MAX_CONTROL_BYTES,
                expected_links=terminal_links,
            )
            captures[ADJUDICATION_MANIFEST_NAME] = terminal
            manifest = _parse_canonical_json_object(
                terminal.payload,
                label="adjudication terminal manifest",
            )
            rows = manifest.get("files")
            if not isinstance(rows, list):
                raise AdjudicationError("adjudication terminal file rows are malformed")
            names: list[str] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise AdjudicationError("adjudication terminal file row is malformed")
                names.append(
                    _safe_package_filename(
                        row.get("path"),
                        label="adjudication file row",
                    )
                )
            if names != sorted(set(names)):
                raise AdjudicationError("adjudication terminal file rows are not canonical")
            for name in names:
                captures[name] = _OpenFile.open(
                    package / name,
                    label=f"adjudication package {name}",
                    limit=_MAX_RESULT_BYTES,
                    expected_links=(2 if name == ADJUDICATION_CLAIM_NAME else 1),
                )
            actual_names = sorted(entry.name for entry in os.scandir(package))
            if actual_names != sorted(captures):
                raise AdjudicationError("adjudication package contains unmanifested entries")
            return cls(
                root=package,
                directory_descriptor=directory_descriptor,
                directory_identity=_directory_identity(os.fstat(directory_descriptor)),
                captures=captures,
            )
        except BaseException:
            for capture in captures.values():
                capture.close()
            os.close(directory_descriptor)
            raise

    def recheck(self, *, root: Path | None = None) -> None:
        current_root = self.root if root is None else root
        try:
            descriptor_metadata = os.fstat(self.directory_descriptor)
            path_metadata = os.stat(current_root, follow_symlinks=False)
            names = sorted(entry.name for entry in os.scandir(current_root))
            same_directory = os.path.samefile(
                current_root,
                _descriptor_path(self.directory_descriptor),
            )
        except OSError as error:
            raise AdjudicationError("adjudication package directory cannot be rechecked") from error
        if (
            _directory_identity(descriptor_metadata) != self.directory_identity
            or _directory_identity(path_metadata) != self.directory_identity
            or names != sorted(self.captures)
            or not same_directory
        ):
            raise AdjudicationError("adjudication package directory changed before commit")
        for name, capture in self.captures.items():
            capture.recheck(path=current_root / name)

    def close(self) -> None:
        for capture in reversed(tuple(self.captures.values())):
            capture.close()
        try:
            os.close(self.directory_descriptor)
        except OSError:
            pass


_MANIFEST_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "assurance",
    "lane",
    "requested_disposition",
    "disposition",
    "outcome_kind",
    "semantic_review_role",
    "terminal",
    "same_version_replay_permitted",
    "producer_root",
    "producer_manifest_sha256",
    "result_sha256",
    "formal_binding_sha256",
    "formal_launch_sha256",
    "audit_attempt_path",
    "audit_attempt_manifest_sha256",
    "audit_attempt_outer_finalized",
    "audit_attempt_status",
    "audit_attempt_files",
    "formal_audit_claim_sha256",
    "adjudication_claim_file",
    "adjudication_claim_sha256",
    "adjudication_claim_marker",
    "bound_source_files",
    "bound_runtime_manifest_sha256",
    "bound_invocation_manifest_sha256",
    "audit_file",
    "audit_sha256",
    "supplied_audit_clean_for_claim",
    "reproduced_audit_file",
    "reproduced_audit_sha256",
    "audit_execution_file",
    "audit_execution_sha256",
    "audit_replay_failure_file",
    "audit_replay_failure_sha256",
    "input_failure_file",
    "input_failure_sha256",
    "replay_started_file",
    "replay_started_sha256",
    "recovery_failure_file",
    "recovery_failure_sha256",
    "audit_execution_completed",
    "audit_byte_identical",
    "semantic_review_file",
    "semantic_review_source_path",
    "semantic_review_sha256",
    "files",
    "file_count",
    "manifest_excludes",
}


def _payload_row(
    value: object,
    *,
    filename: str,
    payload: bytes,
    label: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"file", "bytes", "sha256"}
        or value.get("file") != filename
        or value.get("bytes") != len(payload)
        or value.get("sha256") != _sha256(payload)
    ):
        raise AdjudicationError(f"{label} identity is invalid")


def _optional_payload_reference(
    manifest: Mapping[str, object],
    payloads: Mapping[str, bytes],
    *,
    file_field: str,
    sha256_field: str,
    filename: str,
) -> bytes | None:
    payload = payloads.get(filename)
    expected_file = filename if payload is not None else None
    expected_sha256 = _sha256(payload) if payload is not None else None
    if manifest.get(file_field) != expected_file or manifest.get(sha256_field) != expected_sha256:
        raise AdjudicationError(f"adjudication {file_field} reference differs from package bytes")
    return payload


def _verify_packaged_execution(
    receipt_payload: bytes,
    *,
    payloads: Mapping[str, bytes],
    upstream: _Upstream,
    attempt: _AuditAttempt,
    supplied_audit: bytes,
) -> bool:
    receipt = _parse_canonical_json_object(
        receipt_payload,
        label="packaged adjudication replay receipt",
    )
    try:
        require_assurance(
            receipt.get("assurance"),
            label="adjudication replay receipt",
        )
    except ValueError as error:
        raise AdjudicationError(str(error)) from error
    stdout = payloads[REPRODUCED_AUDIT_NAME]
    stderr = payloads[AUDIT_STDERR_NAME]
    runtime = payloads[AUDIT_RUNTIME_NAME]
    invocation = payloads[AUDIT_INVOCATION_NAME]
    _payload_row(
        receipt.get("stdout"),
        filename=REPRODUCED_AUDIT_NAME,
        payload=stdout,
        label="adjudication replay stdout",
    )
    _payload_row(
        receipt.get("stderr"),
        filename=AUDIT_STDERR_NAME,
        payload=stderr,
        label="adjudication replay stderr",
    )
    _payload_row(
        receipt.get("runtime_manifest"),
        filename=AUDIT_RUNTIME_NAME,
        payload=runtime,
        label="adjudication replay runtime",
    )
    _payload_row(
        receipt.get("invocation_manifest"),
        filename=AUDIT_INVOCATION_NAME,
        payload=invocation,
        label="adjudication replay invocation",
    )
    dependencies = cast(Mapping[str, object], upstream.binding["dependencies"])
    report = _parse_canonical_json_object(
        stdout,
        label="packaged reproduced audit",
    )
    expected_support = _parse_canonical_json_object(
        runtime,
        label="packaged replay runtime",
    ).get("support_files")
    expected_fields = set(
        _execution_receipt(
            upstream=upstream,
            attempt=attempt,
            replay=_Replay(
                returncode=0,
                command=(),
                stdout=b"",
                stderr=b"",
                report={"passed": True},
                runtime_manifest=b"",
                invocation_manifest=b"",
                bootstrap_sha256="",
                auditor_source_sha256="",
                support_files=[],
                source_mode="descriptor",
            ),
            supplied_audit_payload=b"",
        )
    )
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != _EXECUTION_SCHEMA
        or receipt.get("experiment_id") != "WM-001"
        or receipt.get("protocol_version") != "1.6.0"
        or receipt.get("producer_root") != str(upstream.root)
        or receipt.get("audit_attempt_path") != str(attempt.root)
        or receipt.get("audit_attempt_manifest_sha256") != attempt.terminal.sha256
        or receipt.get("source_mode") != "descriptor"
        or receipt.get("interpreter")
        != {
            "executable": dependencies["python_executable"],
            "sha256": dependencies["python_executable_sha256"],
            "flags": ["-I", "-S", "-B"],
        }
        or type(report.get("passed")) is not bool
        or receipt.get("returncode") != (0 if report["passed"] is True else 1)
        or receipt.get("report_passed") is not report["passed"]
        or runtime != upstream.outcome_runtime_payload
        or invocation != upstream.outcome_invocation_payload
        or receipt.get("runner_source_sha256") != _sha256(upstream.source_payloads["audit_runner.py"])
        or receipt.get("bootstrap_source_sha256") != _sha256(upstream.bootstrap_payload)
        or receipt.get("auditor_source_sha256") != _sha256(upstream.source_payloads["artifact_audit.py"])
        or receipt.get("adjudicator_source_sha256") != _sha256(upstream.source_payloads["adjudication.py"])
        or receipt.get("support_files") != expected_support
        or receipt.get("formal_binding_sha256") != _sha256(upstream.binding_payload)
        or receipt.get("formal_launch_sha256") != _sha256(upstream.launch_payload)
        or receipt.get("supplied_audit_sha256") != _sha256(supplied_audit)
        or receipt.get("byte_identical") is not (stdout == supplied_audit)
        or receipt.get("execution_completed") is not True
        or receipt.get("same_version_replay_permitted") is not False
    ):
        raise AdjudicationError("packaged adjudication replay receipt is inconsistent")
    return stdout == supplied_audit


def _verify_replay_failure(
    payload: bytes,
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    supplied_audit: bytes,
    requested_disposition: object,
    execution_completed: bool,
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    failure = _parse_canonical_json_object(
        payload,
        label="packaged adjudication replay failure",
    )
    try:
        require_assurance(
            failure.get("assurance"),
            label="adjudication replay failure",
        )
    except ValueError as error:
        raise AdjudicationError(str(error)) from error
    if (
        set(failure)
        != {
            "schema",
            "experiment_id",
            "protocol_version",
            "assurance",
            "producer_root",
            "audit_attempt_path",
            "audit_attempt_manifest_sha256",
            "failure_code",
            "error_type",
            "requested_disposition",
            "supplied_audit_sha256",
            "partial_execution",
            "runner_source_sha256",
            "bootstrap_source_sha256",
            "auditor_source_sha256",
            "adjudicator_source_sha256",
            "outcome_runtime_manifest_sha256",
            "outcome_invocation_manifest_sha256",
            "formal_binding_sha256",
            "formal_launch_sha256",
            "execution_completed",
            "terminal",
            "same_version_replay_permitted",
        }
        or failure.get("schema") != _REPLAY_FAILURE_SCHEMA
        or failure.get("experiment_id") != "WM-001"
        or failure.get("protocol_version") != "1.6.0"
        or failure.get("producer_root") != str(upstream.root)
        or failure.get("audit_attempt_path") != str(attempt.root)
        or failure.get("audit_attempt_manifest_sha256") != attempt.terminal.sha256
        or failure.get("failure_code") not in _FAILURE_CODES
        or not isinstance(failure.get("error_type"), str)
        or not cast(str, failure["error_type"])
        or failure.get("requested_disposition") != requested_disposition
        or failure.get("supplied_audit_sha256") != _sha256(supplied_audit)
        or failure.get("runner_source_sha256") != _sha256(upstream.source_payloads["audit_runner.py"])
        or failure.get("bootstrap_source_sha256") != _sha256(upstream.bootstrap_payload)
        or failure.get("auditor_source_sha256") != _sha256(upstream.source_payloads["artifact_audit.py"])
        or failure.get("adjudicator_source_sha256") != _sha256(upstream.source_payloads["adjudication.py"])
        or failure.get("outcome_runtime_manifest_sha256") != _sha256(upstream.outcome_runtime_payload)
        or failure.get("outcome_invocation_manifest_sha256") != _sha256(upstream.outcome_invocation_payload)
        or failure.get("formal_binding_sha256") != _sha256(upstream.binding_payload)
        or failure.get("formal_launch_sha256") != _sha256(upstream.launch_payload)
        or failure.get("execution_completed") is not execution_completed
        or failure.get("terminal") is not True
        or failure.get("same_version_replay_permitted") is not False
    ):
        raise AdjudicationError("packaged adjudication replay failure is inconsistent")
    partial = failure.get("partial_execution")
    partial_names = {
        "stdout": PARTIAL_REPLAY_STDOUT_NAME,
        "stderr": PARTIAL_REPLAY_STDERR_NAME,
        "runtime_manifest": PARTIAL_REPLAY_RUNTIME_NAME,
        "invocation_manifest": PARTIAL_REPLAY_INVOCATION_NAME,
    }
    if partial is None:
        if any(name in payloads for name in partial_names.values()):
            raise AdjudicationError("replay failure has unbound partial execution files")
    elif isinstance(partial, Mapping):
        command = partial.get("command")
        support_files = partial.get("support_files")
        if (
            set(partial)
            != {
                "phase",
                "returncode",
                "source_mode",
                "command",
                "stdout",
                "stderr",
                "runtime_manifest",
                "invocation_manifest",
                "bootstrap_sha256",
                "auditor_source_sha256",
                "support_files",
            }
            or not isinstance(partial.get("phase"), str)
            or (partial.get("returncode") is not None and type(partial.get("returncode")) is not int)
            or partial.get("source_mode") != "descriptor"
            or not isinstance(command, list)
            or any(not isinstance(argument, str) for argument in command)
            or not _is_sha256(partial.get("bootstrap_sha256"))
            or not _is_sha256(partial.get("auditor_source_sha256"))
            or not isinstance(support_files, list)
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row.get("path"), str)
                or type(row.get("bytes")) is not int
                or cast(int, row["bytes"]) < 0
                or not _is_sha256(row.get("sha256"))
                for row in cast(list[object], support_files)
            )
            or (partial.get("returncode") is not None) is not execution_completed
        ):
            raise AdjudicationError("replay failure partial execution identity is malformed")
        for prefix, name in partial_names.items():
            try:
                partial_payload = payloads[name]
            except KeyError as error:
                raise AdjudicationError("replay failure omits bounded partial execution bytes") from error
            _payload_row(
                partial.get(prefix),
                filename=name,
                payload=partial_payload,
                label=f"replay failure partial {prefix}",
            )
        if (
            len(payloads[PARTIAL_REPLAY_STDOUT_NAME]) > _MAX_AUDIT_BYTES
            or len(payloads[PARTIAL_REPLAY_STDERR_NAME]) > _MAX_STDERR_BYTES
            or len(payloads[PARTIAL_REPLAY_RUNTIME_NAME]) > _MAX_CONTROL_BYTES
            or len(payloads[PARTIAL_REPLAY_INVOCATION_NAME]) > _MAX_CONTROL_BYTES
        ):
            raise AdjudicationError("replay failure partial evidence exceeds its bound")
    else:
        raise AdjudicationError("replay failure partial execution identity is malformed")
    return failure


def _verify_recovery_failure(
    payload: bytes,
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    claim_payload: bytes,
    requested_disposition: Disposition,
    evidence: _AttemptEvidence,
    replay_started_payload: bytes | None,
) -> dict[str, Any]:
    failure = _parse_canonical_json_object(
        payload,
        label="packaged adjudication recovery failure",
    )
    try:
        require_assurance(
            failure.get("assurance"),
            label="adjudication recovery failure",
        )
    except ValueError as error:
        raise AdjudicationError(str(error)) from error
    prior_replay_state = failure.get("prior_replay_state")
    expected_prior_states = (
        {"not_applicable"}
        if evidence.kind == "execution_failure"
        else (
            {"started"}
            if replay_started_payload is not None
            else {"not_observed", "untrusted_or_partial"}
        )
    )
    if (
        set(failure)
        != set(
            _recovery_failure_record(
                upstream=upstream,
                attempt=attempt,
                claim_payload=claim_payload,
                requested_disposition=requested_disposition,
                recovery_mode="explicit",
                failure_phase="interrupted_after_claim",
                failure_type=None,
                prior_replay_state=(
                    "not_applicable"
                    if evidence.kind == "execution_failure"
                    else "not_observed"
                ),
                replay_started_payload=None,
            )
        )
        or failure.get("schema") != _RECOVERY_FAILURE_SCHEMA
        or failure.get("experiment_id") != "WM-001"
        or failure.get("protocol_version") != "1.6.0"
        or failure.get("terminal") is not True
        or failure.get("disposition") != "rejected"
        or failure.get("producer_root") != str(upstream.root)
        or failure.get("producer_manifest_sha256") != _sha256(upstream.producer_manifest_payload)
        or failure.get("result_sha256") != _sha256(upstream.result_payload)
        or failure.get("formal_binding_sha256") != _sha256(upstream.binding_payload)
        or failure.get("formal_launch_sha256") != _sha256(upstream.launch_payload)
        or failure.get("audit_attempt_path") != str(attempt.root)
        or failure.get("audit_attempt_manifest_sha256") != attempt.terminal.sha256
        or failure.get("formal_audit_claim_sha256") != attempt.formal_claim.sha256
        or failure.get("adjudication_claim_sha256") != _sha256(claim_payload)
        or failure.get("requested_disposition") != requested_disposition
        or failure.get("recovery_mode") not in {"automatic", "explicit"}
        or failure.get("failure_phase") not in _RECOVERY_PHASES
        or (
            failure.get("failure_type") is not None
            and (
                not isinstance(failure.get("failure_type"), str)
                or not cast(str, failure["failure_type"])
            )
        )
        or (
            failure.get("recovery_mode") == "automatic"
            and (
                failure.get("failure_phase") == "interrupted_after_claim"
                or failure.get("failure_type") is None
            )
        )
        or (
            failure.get("recovery_mode") == "explicit"
            and (
                failure.get("failure_phase") != "interrupted_after_claim"
                or failure.get("failure_type") is not None
            )
        )
        or prior_replay_state not in expected_prior_states
        or failure.get("replay_started_sha256")
        != (
            _sha256(replay_started_payload)
            if replay_started_payload is not None
            else None
        )
        or failure.get("recovery_replay_performed") is not False
        or failure.get("same_version_replay_permitted") is not False
    ):
        raise AdjudicationError("packaged adjudication recovery failure is inconsistent")
    return failure


def _verify_adjudication_package(
    path: Path,
    *,
    require_outer: bool,
    allow_staging: bool,
) -> dict[str, object]:
    package = _canonical_directory(path, label="adjudication package")
    if not allow_staging and package != FORMAL_ADJUDICATION_PACKAGE_PATH:
        raise AdjudicationError("public adjudication verification requires the canonical v1.6 package")
    if require_outer:
        try:
            operator_module.verify_outer_completion(package / ADJUDICATION_MANIFEST_NAME)
        except (OSError, RuntimeError, ValueError) as error:
            raise AdjudicationError("adjudication package has no authoritative outer completion") from error
    custody = _PackageCustody.capture(
        package,
        terminal_links=2 if require_outer else 1,
    )
    attempt: _AuditAttempt | None = None
    try:
        terminal_payload = custody.captures[ADJUDICATION_MANIFEST_NAME].payload
        manifest = _parse_canonical_json_object(
            terminal_payload,
            label="adjudication terminal manifest",
        )
        try:
            require_assurance(
                manifest.get("assurance"),
                label="adjudication package",
            )
        except ValueError as error:
            raise AdjudicationError(str(error)) from error
        rows = manifest.get("files")
        if (
            set(manifest) != _MANIFEST_FIELDS
            or manifest.get("schema") != _PACKAGE_SCHEMA
            or manifest.get("experiment_id") != "WM-001"
            or manifest.get("protocol_version") != "1.6.0"
            or manifest.get("lane") != "formal"
            or manifest.get("requested_disposition") not in {"accepted", "rejected"}
            or manifest.get("disposition") not in {"accepted", "rejected"}
            or manifest.get("outcome_kind")
            not in {
                "audit_report",
                "audit_replay_mismatch",
                "adjudication_replay_failure",
                "formal_audit_execution_failure",
                "adjudication_recovery_failure",
            }
            or manifest.get("semantic_review_role")
            not in {
                "supplied_audit_review",
                "pre_replay_supplied_audit_review",
                "execution_failure_review",
            }
            or manifest.get("terminal") is not True
            or manifest.get("same_version_replay_permitted") is not False
            or manifest.get("adjudication_claim_file") != ADJUDICATION_CLAIM_NAME
            or manifest.get("adjudication_claim_marker") != str(FORMAL_ADJUDICATION_CLAIM_MARKER)
            or manifest.get("semantic_review_file") != COPIED_SEMANTIC_REVIEW_NAME
            or manifest.get("semantic_review_source_path") != str(FORMAL_SEMANTIC_REVIEW_PATH)
            or manifest.get("manifest_excludes") != [ADJUDICATION_MANIFEST_NAME]
            or not isinstance(rows, list)
            or manifest.get("file_count") != len(rows)
        ):
            raise AdjudicationError("adjudication manifest has invalid terminal identity")
        payloads = {
            name: capture.payload for name, capture in custody.captures.items() if name != ADJUDICATION_MANIFEST_NAME
        }
        actual_rows = _file_rows(payloads)
        if rows != actual_rows:
            raise AdjudicationError("adjudication package bytes differ from its manifest")
        _optional_payload_reference(
            manifest,
            payloads,
            file_field="audit_file",
            sha256_field="audit_sha256",
            filename=COPIED_AUDIT_NAME,
        )
        _optional_payload_reference(
            manifest,
            payloads,
            file_field="reproduced_audit_file",
            sha256_field="reproduced_audit_sha256",
            filename=REPRODUCED_AUDIT_NAME,
        )
        _optional_payload_reference(
            manifest,
            payloads,
            file_field="audit_execution_file",
            sha256_field="audit_execution_sha256",
            filename=AUDIT_EXECUTION_NAME,
        )
        _optional_payload_reference(
            manifest,
            payloads,
            file_field="audit_replay_failure_file",
            sha256_field="audit_replay_failure_sha256",
            filename=AUDIT_FAILURE_NAME,
        )
        _optional_payload_reference(
            manifest,
            payloads,
            file_field="input_failure_file",
            sha256_field="input_failure_sha256",
            filename=INPUT_FAILURE_NAME,
        )
        replay_started = _optional_payload_reference(
            manifest,
            payloads,
            file_field="replay_started_file",
            sha256_field="replay_started_sha256",
            filename=ADJUDICATION_REPLAY_STARTED_NAME,
        )
        recovery_failure = _optional_payload_reference(
            manifest,
            payloads,
            file_field="recovery_failure_file",
            sha256_field="recovery_failure_sha256",
            filename=ADJUDICATION_RECOVERY_FAILURE_NAME,
        )
        claim_payload = payloads.get(ADJUDICATION_CLAIM_NAME)
        if claim_payload is None or manifest.get("adjudication_claim_sha256") != _sha256(claim_payload):
            raise AdjudicationError("adjudication package omits its exact claim")
        try:
            same_claim = os.path.samefile(
                package / ADJUDICATION_CLAIM_NAME,
                FORMAL_ADJUDICATION_CLAIM_MARKER,
            )
        except OSError as error:
            raise AdjudicationError("adjudication claim marker cannot be compared") from error
        if not same_claim:
            raise AdjudicationError("packaged adjudication claim is not the marker inode")

        audit_attempt_value = manifest.get("audit_attempt_path")
        producer_value = manifest.get("producer_root")
        if not isinstance(audit_attempt_value, str) or not isinstance(producer_value, str):
            raise AdjudicationError("adjudication upstream paths are malformed")
        attempt = _load_audit_attempt(Path(audit_attempt_value))
        upstream = _load_upstream(
            Path(producer_value),
            require_live_sources=False,
        )
        evidence = _classify_attempt_evidence(
            attempt,
            upstream=upstream,
        )
        formal_audit_claim = _formal_claim_value(
            attempt,
            upstream=upstream,
        )
        review_payload = payloads.get(COPIED_SEMANTIC_REVIEW_NAME)
        if review_payload is None or manifest.get("semantic_review_sha256") != _sha256(review_payload):
            raise AdjudicationError("adjudication package omits its semantic review")
        live_review_payload = _read_regular_file(
            FORMAL_SEMANTIC_REVIEW_PATH,
            limit=_MAX_REVIEW_BYTES,
            label="canonical live semantic review",
        )
        if live_review_payload != review_payload:
            raise AdjudicationError(
                "packaged semantic review differs from the canonical live review"
            )
        review = _parse_canonical_json_object(
            review_payload,
            label="packaged semantic review",
        )
        requested = cast(Disposition, manifest["requested_disposition"])
        _verify_semantic_review(
            review,
            attempt=attempt,
            upstream=upstream,
            evidence=evidence,
            disposition=requested,
        )
        expected_claim = _adjudication_claim_value(
            upstream=upstream,
            attempt=attempt,
            review_sha256=_sha256(review_payload),
            requested_disposition=requested,
            formal_audit_claim=formal_audit_claim,
        )
        claim = _parse_canonical_json_object(
            claim_payload,
            label="packaged adjudication claim",
        )
        try:
            require_assurance(
                claim.get("assurance"),
                label="adjudication claim",
            )
        except ValueError as error:
            raise AdjudicationError(str(error)) from error
        if claim != expected_claim:
            raise AdjudicationError("packaged adjudication claim differs from exact inputs")
        if replay_started is not None:
            _verify_replay_started(
                replay_started,
                upstream=upstream,
                attempt=attempt,
                claim_payload=claim_payload,
            )

        _, expected_source_rows, expected_attempt_rows = _base_payloads(
            upstream,
            attempt,
            review_payload,
        )
        expected_identities = {
            "producer_manifest_sha256": _sha256(upstream.producer_manifest_payload),
            "result_sha256": _sha256(upstream.result_payload),
            "formal_binding_sha256": _sha256(upstream.binding_payload),
            "formal_launch_sha256": _sha256(upstream.launch_payload),
            "audit_attempt_manifest_sha256": attempt.terminal.sha256,
            "audit_attempt_outer_finalized": attempt.outer_finalized,
            "audit_attempt_status": attempt.manifest["status"],
            "audit_attempt_files": expected_attempt_rows,
            "formal_audit_claim_sha256": attempt.formal_claim.sha256,
            "bound_source_files": expected_source_rows,
            "bound_runtime_manifest_sha256": _sha256(upstream.outcome_runtime_payload),
            "bound_invocation_manifest_sha256": _sha256(upstream.outcome_invocation_payload),
        }
        if any(manifest.get(field) != value for field, value in expected_identities.items()):
            raise AdjudicationError("adjudication package upstream identities changed")
        if (
            payloads.get(COPIED_LAUNCH_NAME) != upstream.launch_payload
            or payloads.get(COPIED_BOOTSTRAP_NAME) != upstream.bootstrap_payload
            or payloads.get(BOUND_AUDIT_RUNTIME_NAME) != upstream.outcome_runtime_payload
            or payloads.get(BOUND_AUDIT_INVOCATION_NAME) != upstream.outcome_invocation_payload
            or any(
                payloads.get(f"{COPIED_SOURCE_PREFIX}{name}") != source_payload
                for name, source_payload in upstream.source_payloads.items()
            )
            or any(
                payloads.get(f"{COPIED_OPERATOR_PREFIX}{name}") != capture.payload
                for name, capture in attempt.captures.items()
            )
        ):
            raise AdjudicationError("packaged bound source or operator evidence changed")

        outcome = manifest["outcome_kind"]
        disposition = manifest["disposition"]
        supplied = payloads.get(COPIED_AUDIT_NAME)
        reproduced = payloads.get(REPRODUCED_AUDIT_NAME)
        execution = payloads.get(AUDIT_EXECUTION_NAME)
        replay_failure = payloads.get(AUDIT_FAILURE_NAME)
        input_failure = payloads.get(INPUT_FAILURE_NAME)
        partial_replay_names = (
            PARTIAL_REPLAY_STDOUT_NAME,
            PARTIAL_REPLAY_STDERR_NAME,
            PARTIAL_REPLAY_RUNTIME_NAME,
            PARTIAL_REPLAY_INVOCATION_NAME,
        )
        if outcome == "adjudication_recovery_failure":
            expected_failure = (
                _canonical_json_bytes(evidence.failure_record)
                if evidence.kind == "execution_failure"
                else None
            )
            expected_role = (
                "execution_failure_review"
                if evidence.kind == "execution_failure"
                else "pre_replay_supplied_audit_review"
            )
            if (
                disposition != "rejected"
                or manifest.get("semantic_review_role") != expected_role
                or recovery_failure is None
                or reproduced is not None
                or execution is not None
                or replay_failure is not None
                or manifest.get("audit_execution_completed") is not False
                or manifest.get("audit_byte_identical") is not None
                or any(name in payloads for name in partial_replay_names)
                or (
                    evidence.kind == "execution_failure"
                    and (
                        supplied is not None
                        or input_failure != expected_failure
                        or replay_started is not None
                        or manifest.get("supplied_audit_clean_for_claim") is not None
                    )
                )
                or (
                    evidence.kind == "report"
                    and (
                        supplied != evidence.report_payload
                        or input_failure is not None
                        or manifest.get("supplied_audit_clean_for_claim")
                        is not evidence.report_clean
                    )
                )
            ):
                raise AdjudicationError("adjudication recovery-failure package is inconsistent")
            _verify_recovery_failure(
                recovery_failure,
                upstream=upstream,
                attempt=attempt,
                claim_payload=claim_payload,
                requested_disposition=requested,
                evidence=evidence,
                replay_started_payload=replay_started,
            )
        elif evidence.kind == "execution_failure":
            expected_failure = _canonical_json_bytes(evidence.failure_record)
            if (
                outcome != "formal_audit_execution_failure"
                or requested != "rejected"
                or disposition != "rejected"
                or manifest.get("semantic_review_role") != "execution_failure_review"
                or supplied is not None
                or reproduced is not None
                or execution is not None
                or replay_failure is not None
                or input_failure != expected_failure
                or manifest.get("input_failure_file") != INPUT_FAILURE_NAME
                or manifest.get("input_failure_sha256") != _sha256(expected_failure)
                or manifest.get("audit_file") is not None
                or manifest.get("audit_sha256") is not None
                or manifest.get("supplied_audit_clean_for_claim") is not None
                or manifest.get("audit_execution_completed") is not False
                or manifest.get("audit_byte_identical") is not None
                or replay_started is not None
                or recovery_failure is not None
                or any(name in payloads for name in partial_replay_names)
            ):
                raise AdjudicationError("formal audit execution-failure package is inconsistent")
        else:
            assert evidence.report_payload is not None
            if (
                supplied != evidence.report_payload
                or manifest.get("audit_file") != COPIED_AUDIT_NAME
                or manifest.get("audit_sha256") != _sha256(supplied)
                or manifest.get("supplied_audit_clean_for_claim") is not evidence.report_clean
                or input_failure is not None
                or manifest.get("input_failure_file") is not None
                or manifest.get("input_failure_sha256") is not None
                or replay_started is None
                or recovery_failure is not None
            ):
                raise AdjudicationError("packaged supplied audit differs from the operator attempt")
            if outcome == "audit_report":
                if (
                    disposition != requested
                    or manifest.get("semantic_review_role") != "supplied_audit_review"
                    or reproduced is None
                    or execution is None
                    or replay_failure is not None
                    or reproduced != supplied
                    or manifest.get("audit_execution_completed") is not True
                    or manifest.get("audit_byte_identical") is not True
                ):
                    raise AdjudicationError("ordinary adjudication report package is inconsistent")
                _verify_packaged_execution(
                    execution,
                    payloads=payloads,
                    upstream=upstream,
                    attempt=attempt,
                    supplied_audit=supplied,
                )
                if disposition == "accepted":
                    if evidence.report_clean is not True:
                        raise AdjudicationError("accepted package has a non-clean audit")
                    _verify_acceptance_gates(upstream.result)
            elif outcome == "audit_replay_mismatch":
                if (
                    disposition != "rejected"
                    or manifest.get("semantic_review_role") != "pre_replay_supplied_audit_review"
                    or reproduced is None
                    or execution is None
                    or replay_failure is not None
                    or reproduced == supplied
                    or manifest.get("audit_execution_completed") is not True
                    or manifest.get("audit_byte_identical") is not False
                ):
                    raise AdjudicationError("audit replay-mismatch package is inconsistent")
                _verify_packaged_execution(
                    execution,
                    payloads=payloads,
                    upstream=upstream,
                    attempt=attempt,
                    supplied_audit=supplied,
                )
                reproduced_report = _parse_canonical_json_object(
                    reproduced,
                    label="packaged byte-distinct audit",
                )
                _verify_audit_identity(
                    reproduced_report,
                    upstream=upstream,
                )
            elif outcome == "adjudication_replay_failure":
                execution_completed = manifest.get("audit_execution_completed")
                if (
                    disposition != "rejected"
                    or manifest.get("semantic_review_role") != "pre_replay_supplied_audit_review"
                    or replay_failure is None
                    or type(execution_completed) is not bool
                    or (execution is not None and execution_completed is not True)
                ):
                    raise AdjudicationError("adjudication replay-failure package is inconsistent")
                if execution is not None:
                    _verify_packaged_execution(
                        execution,
                        payloads=payloads,
                        upstream=upstream,
                        attempt=attempt,
                        supplied_audit=supplied,
                    )
                failure_value = _verify_replay_failure(
                    replay_failure,
                    upstream=upstream,
                    attempt=attempt,
                    supplied_audit=supplied,
                    requested_disposition=requested,
                    execution_completed=cast(bool, execution_completed),
                    payloads=payloads,
                )
                if execution is not None:
                    if failure_value.get("failure_code") != "audit-execution-identity-mismatch" or reproduced is None:
                        raise AdjudicationError("completed replay failure has no identity-mismatch evidence")
                    reproduced_report = _parse_canonical_json_object(
                        reproduced,
                        label="failed reproduced audit",
                    )
                    try:
                        _verify_audit_identity(
                            reproduced_report,
                            upstream=upstream,
                        )
                    except AdjudicationError:
                        pass
                    else:
                        raise AdjudicationError("completed replay failure contains a valid audit identity")
            else:
                raise AdjudicationError("report evidence has an execution-failure outcome")
            if outcome != "adjudication_replay_failure" and any(name in payloads for name in partial_replay_names):
                raise AdjudicationError("non-failure adjudication contains partial replay evidence")
        custody.recheck()
        return cast(dict[str, object], manifest)
    finally:
        if attempt is not None:
            attempt.close()
        custody.close()


def verify_adjudication_package(path: Path) -> dict[str, object]:
    """Verify only an outer-finalized canonical v1.6 adjudication package."""

    return _verify_adjudication_package(
        path,
        require_outer=True,
        allow_staging=False,
    )


def _require_sealed_entry() -> None:
    from .experiment import _verify_live_bootstrap_custody

    try:
        _verify_live_bootstrap_custody()
    except (OSError, RuntimeError, ValueError) as error:
        raise AdjudicationError("adjudication must be entered through the sealed producer bootstrap") from error


def _open_adjudication_claim_marker() -> _OpenFile:
    marker = _require_lexical_absolute(
        FORMAL_ADJUDICATION_CLAIM_MARKER,
        label="formal adjudication claim marker",
    )
    try:
        metadata = os.lstat(marker)
    except OSError as error:
        raise AdjudicationError("formal adjudication claim has not been consumed") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink not in {1, 2}
    ):
        raise AdjudicationError("formal adjudication claim marker has invalid recovery custody")
    return _OpenFile.open(
        marker,
        label="formal adjudication claim marker",
        limit=_MAX_CONTROL_BYTES,
        expected_links=metadata.st_nlink,
    )


def _claim_holder(
    *,
    package: Path,
    marker: Path,
    expected_links: int,
) -> Path | None:
    holders: list[Path] = []
    package_claim = package / ADJUDICATION_CLAIM_NAME
    candidates: list[Path] = []
    if os.path.lexists(package_claim):
        candidates.append(package_claim)
    prefix = f".{package.name}.staging-"
    try:
        entries = tuple(os.scandir(package.parent))
    except OSError as error:
        raise AdjudicationError("adjudication recovery cannot scan its package parent") from error
    for entry in entries:
        if (
            not entry.name.startswith(prefix)
            or entry.is_symlink()
            or not entry.is_dir(follow_symlinks=False)
        ):
            continue
        candidate = Path(entry.path) / ADJUDICATION_CLAIM_NAME
        if os.path.lexists(candidate):
            candidates.append(candidate)
    for candidate in candidates:
        try:
            metadata = os.lstat(candidate)
            if (
                stat.S_ISREG(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
                and os.path.samefile(candidate, marker)
            ):
                holders.append(candidate)
        except OSError as error:
            raise AdjudicationError("adjudication claim holder cannot be compared") from error
    if expected_links == 1:
        if holders:
            raise AdjudicationError("marker-only adjudication claim has an unexpected package link")
        return None
    if len(holders) != 1:
        raise AdjudicationError("consumed adjudication claim has no unique recoverable package link")
    return holders[0]


def _validated_staged_replay_started(
    holder: Path | None,
    *,
    upstream: _Upstream,
    attempt: _AuditAttempt,
    claim_payload: bytes,
) -> tuple[Literal["not_observed", "started", "untrusted_or_partial"], bytes | None]:
    if holder is None:
        return "not_observed", None
    path = holder.parent / ADJUDICATION_REPLAY_STARTED_NAME
    if not os.path.lexists(path):
        return "not_observed", None
    try:
        payload = _read_regular_file(
            path,
            limit=_MAX_CONTROL_BYTES,
            label="staged adjudication replay-started receipt",
        )
        _verify_replay_started(
            payload,
            upstream=upstream,
            attempt=attempt,
            claim_payload=claim_payload,
        )
    except (AdjudicationError, OSError):
        # A partial write precedes runner entry because the complete receipt is
        # fsynced before the call.  It is therefore safe to treat as
        # unobserved; recovery never invokes the runner.
        return "untrusted_or_partial", None
    return "started", payload


def _publish_terminal_package(
    *,
    staging: Path,
    package: Path,
    manifest: dict[str, object],
    attempt: _AuditAttempt,
    upstream: _Upstream,
    review_capture: _OpenFile,
    claim_custody: _ClaimCustody,
    invoke_fault_hooks: bool = True,
    on_package_published: Callable[[], None] | None = None,
    on_outer_registered: Callable[[], None] | None = None,
) -> None:
    terminal_path = staging / ADJUDICATION_MANIFEST_NAME
    _write_private_file(
        terminal_path,
        _canonical_json_bytes(manifest),
    )
    _fsync_directory(staging)
    package_custody = _PackageCustody.capture(
        staging,
        terminal_links=1,
    )
    try:
        staged = _verify_adjudication_package(
            staging,
            require_outer=False,
            allow_staging=True,
        )
        if staged != manifest:
            raise AdjudicationError("staged adjudication differs from its terminal manifest")
        if invoke_fault_hooks:
            _before_package_publish(staging)
        package_custody.recheck()
        attempt.recheck()
        review_capture.recheck()
        claim_custody.recheck(staging / ADJUDICATION_CLAIM_NAME)
        reopened = _load_upstream(
            upstream.root,
            require_live_sources=True,
        )
        if _upstream_identity(reopened) != _upstream_identity(upstream):
            raise AdjudicationError("formal producer changed before adjudication publication")
        _require_sealed_entry()
        if os.path.lexists(package):
            raise FileExistsError(f"refusing to replace formal adjudication package: {package}")
        _publish_directory_no_replace(staging, package)
        if on_package_published is not None:
            on_package_published()
        _fsync_directory(package.parent)
        if invoke_fault_hooks:
            _after_package_publish(package)
        package_custody.recheck(root=package)
        claim_custody.recheck(package / ADJUDICATION_CLAIM_NAME)
        final = _verify_adjudication_package(
            package,
            require_outer=False,
            allow_staging=False,
        )
        if final != manifest:
            raise AdjudicationError("published adjudication differs from staged identity")
        from .producer_bootstrap import register_outer_terminal

        if invoke_fault_hooks:
            _before_outer_registration(package)
        register_outer_terminal(
            package / ADJUDICATION_MANIFEST_NAME,
            logical_exit_code=(0 if manifest["disposition"] == "accepted" else 1),
        )
        if on_outer_registered is not None:
            on_outer_registered()
        if invoke_fault_hooks:
            _after_outer_registration(package)
    finally:
        package_custody.close()


def _recover_adjudication_package(
    *,
    recovery_mode: Literal["automatic", "explicit"],
    failure_phase: str,
    failure_type: str | None,
    allow_completed: bool,
) -> dict[str, object]:
    """Recover a consumed claim without invoking the audit runner."""

    if failure_phase not in _RECOVERY_PHASES:
        raise AdjudicationError("adjudication recovery phase is invalid")
    if failure_type is not None and (not failure_type or len(failure_type) > 256):
        raise AdjudicationError("adjudication recovery failure type is invalid")
    _require_sealed_entry()
    package = _require_lexical_absolute(
        FORMAL_ADJUDICATION_PACKAGE_PATH,
        label="formal adjudication package",
    )
    marker_path = _require_lexical_absolute(
        FORMAL_ADJUDICATION_CLAIM_MARKER,
        label="formal adjudication claim marker",
    )
    if (
        package.name != "formal-adjudication-v1.6.0"
        or marker_path.name != "formal-adjudication-v1.6.0.json"
    ):
        raise AdjudicationError("formal adjudication recovery paths are not version-scoped")
    package.parent.mkdir(parents=True, exist_ok=True)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    _canonical_directory(package.parent, label="formal adjudication package parent")
    _canonical_directory(marker_path.parent, label="formal adjudication claim-marker parent")

    marker_capture = _open_adjudication_claim_marker()
    attempt: _AuditAttempt | None = None
    review_capture: _OpenFile | None = None
    claim_custody: _ClaimCustody | None = None
    staging: Path | None = None
    claim_relocated = False
    published = False
    try:
        claim_payload = marker_capture.payload
        claim = _parse_canonical_json_object(
            claim_payload,
            label="formal adjudication claim marker",
        )
        holder = _claim_holder(
            package=package,
            marker=marker_path,
            expected_links=marker_capture.identity[3],
        )
        if os.path.lexists(package):
            if holder != package / ADJUDICATION_CLAIM_NAME:
                raise AdjudicationError("canonical adjudication package does not hold its claim marker")
            try:
                completed = verify_adjudication_package(package)
            except AdjudicationError:
                completed = None
            if completed is not None:
                if not allow_completed:
                    raise _AdjudicationRetired(
                        "WM-001 protocol 1.6 adjudication is already outer-finalized"
                    )
                return completed
            exact = _verify_adjudication_package(
                package,
                require_outer=False,
                allow_staging=False,
            )
            marker_capture.recheck(path=package / ADJUDICATION_CLAIM_NAME)
            _require_sealed_entry()
            from .producer_bootstrap import register_outer_terminal

            register_outer_terminal(
                package / ADJUDICATION_MANIFEST_NAME,
                logical_exit_code=(0 if exact["disposition"] == "accepted" else 1),
            )
            return exact

        producer_value = claim.get("producer_root")
        audit_attempt_value = claim.get("audit_attempt_path")
        requested_value = claim.get("requested_disposition")
        if (
            not isinstance(producer_value, str)
            or not isinstance(audit_attempt_value, str)
            or requested_value not in {"accepted", "rejected"}
        ):
            raise AdjudicationError("formal adjudication claim has invalid recovery inputs")
        requested_disposition = cast(Disposition, requested_value)
        attempt = _load_audit_attempt(Path(audit_attempt_value))
        upstream = _load_upstream(
            Path(producer_value),
            require_live_sources=True,
        )
        evidence = _classify_attempt_evidence(
            attempt,
            upstream=upstream,
        )
        review_capture = _capture_semantic_review(
            FORMAL_SEMANTIC_REVIEW_PATH,
            attempt=attempt,
            upstream=upstream,
        )
        if review_capture.sha256 != claim.get("semantic_review_sha256"):
            raise AdjudicationError("canonical semantic review changed after adjudication claim")
        review = _parse_canonical_json_object(
            review_capture.payload,
            label="canonical semantic review",
        )
        _verify_semantic_review(
            review,
            attempt=attempt,
            upstream=upstream,
            evidence=evidence,
            disposition=requested_disposition,
        )
        formal_audit_claim = _formal_claim_value(
            attempt,
            upstream=upstream,
        )
        expected_claim = _adjudication_claim_value(
            upstream=upstream,
            attempt=attempt,
            review_sha256=review_capture.sha256,
            requested_disposition=requested_disposition,
            formal_audit_claim=formal_audit_claim,
        )
        if claim != expected_claim:
            raise AdjudicationError("formal adjudication claim differs from exact recovery inputs")
        replay_observation, replay_started_payload = (
            _validated_staged_replay_started(
                holder,
                upstream=upstream,
                attempt=attempt,
                claim_payload=claim_payload,
            )
            if evidence.kind == "report"
            else ("not_observed", None)
        )
        prior_replay_state: Literal[
            "not_applicable",
            "not_observed",
            "started",
            "untrusted_or_partial",
        ] = (
            "not_applicable"
            if evidence.kind == "execution_failure"
            else replay_observation
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{package.name}.staging-",
                dir=package.parent,
            )
        )
        if (
            staging.parent != package.parent
            or not staging.name.startswith(f".{package.name}.staging-")
            or staging.resolve(strict=True) != staging
        ):
            raise AdjudicationError("recovery staging directory is not a hidden sibling")
        os.chmod(staging, 0o700)
        payloads, source_rows, attempt_rows = _base_payloads(
            upstream,
            attempt,
            review_capture.payload,
        )
        if evidence.kind == "report":
            assert evidence.report_payload is not None
            payloads[COPIED_AUDIT_NAME] = evidence.report_payload
        else:
            assert evidence.failure_record is not None
            payloads[INPUT_FAILURE_NAME] = _canonical_json_bytes(evidence.failure_record)
        if replay_started_payload is not None:
            payloads[ADJUDICATION_REPLAY_STARTED_NAME] = replay_started_payload
        recovery_failure = _recovery_failure_record(
            upstream=upstream,
            attempt=attempt,
            claim_payload=claim_payload,
            requested_disposition=requested_disposition,
            recovery_mode=recovery_mode,
            failure_phase=failure_phase,
            failure_type=failure_type,
            prior_replay_state=prior_replay_state,
            replay_started_payload=replay_started_payload,
        )
        payloads[ADJUDICATION_RECOVERY_FAILURE_NAME] = _canonical_json_bytes(
            recovery_failure
        )
        for name, payload in sorted(payloads.items()):
            _write_private_file(staging / name, payload)
        _fsync_directory(staging)

        target_claim = staging / ADJUDICATION_CLAIM_NAME
        marker_capture.close()
        if holder is None:
            try:
                os.link(marker_path, target_claim, follow_symlinks=False)
            except OSError as error:
                raise AdjudicationError("marker-only adjudication claim could not be recovered") from error
        else:
            try:
                os.rename(holder, target_claim)
            except OSError as error:
                raise AdjudicationError("staged adjudication claim could not be relocated") from error
        claim_relocated = True
        _fsync_directory(staging)
        _fsync_directory(marker_path.parent)
        if holder is not None:
            _fsync_directory(holder.parent)
        claim_custody = _ClaimCustody(
            marker=_OpenFile.open(
                marker_path,
                label="recovered adjudication claim marker",
                limit=_MAX_CONTROL_BYTES,
                expected_links=2,
            ),
            payload=claim_payload,
        )
        claim_custody.recheck(target_claim)
        payloads[ADJUDICATION_CLAIM_NAME] = claim_payload
        if holder is not None and holder.parent != staging:
            shutil.rmtree(holder.parent)

        manifest = _manifest(
            upstream=upstream,
            attempt=attempt,
            evidence=evidence,
            payloads=payloads,
            source_rows=source_rows,
            attempt_rows=attempt_rows,
            claim_payload=claim_payload,
            requested_disposition=requested_disposition,
            disposition="rejected",
            outcome_kind="adjudication_recovery_failure",
            review_role=(
                "execution_failure_review"
                if evidence.kind == "execution_failure"
                else "pre_replay_supplied_audit_review"
            ),
            replay=None,
        )
        attempt.recheck()
        review_capture.recheck()
        _require_sealed_entry()
        _publish_terminal_package(
            staging=staging,
            package=package,
            manifest=manifest,
            attempt=attempt,
            upstream=upstream,
            review_capture=review_capture,
            claim_custody=claim_custody,
            invoke_fault_hooks=False,
        )
        published = True
        return manifest
    finally:
        marker_capture.close()
        if claim_custody is not None:
            claim_custody.close()
        if review_capture is not None:
            review_capture.close()
        if attempt is not None:
            attempt.close()
        if (
            staging is not None
            and staging.exists()
            and not published
            and not claim_relocated
        ):
            shutil.rmtree(staging)


def recover_adjudication_package() -> dict[str, object]:
    """Finalize a consumed v1.6 claim without another audit replay."""

    return _recover_adjudication_package(
        recovery_mode="explicit",
        failure_phase="interrupted_after_claim",
        failure_type=None,
        allow_completed=False,
    )


def create_adjudication_package(
    *,
    audit_attempt: Path,
    disposition: Disposition,
    semantic_review: Path,
) -> dict[str, object]:
    """Consume v1.6 and publish one accepted/rejected terminal package."""

    if disposition not in {"accepted", "rejected"}:
        raise AdjudicationError("adjudication is terminal; disposition must be accepted or rejected")
    _require_sealed_entry()
    package = _prepare_output_paths()
    attempt = _load_audit_attempt(audit_attempt)
    review_capture: _OpenFile | None = None
    claim_custody: _ClaimCustody | None = None
    staging: Path | None = None
    claim_published = False
    published = False
    package_published = False
    outer_registered = False
    postclaim_phase = "post_claim_link"
    try:
        primary = cast(Mapping[str, object], attempt.manifest["primary"])
        producer_value = primary.get("producer_root")
        if not isinstance(producer_value, str):
            raise AdjudicationError("formal audit attempt has no producer root")
        upstream = _load_upstream(
            Path(producer_value),
            require_live_sources=True,
        )
        evidence = _classify_attempt_evidence(
            attempt,
            upstream=upstream,
        )
        review_capture = _capture_semantic_review(
            semantic_review,
            attempt=attempt,
            upstream=upstream,
        )
        review = _parse_canonical_json_object(
            review_capture.payload,
            label="semantic review",
        )
        _verify_semantic_review(
            review,
            attempt=attempt,
            upstream=upstream,
            evidence=evidence,
            disposition=disposition,
        )
        formal_audit_claim = _formal_claim_value(
            attempt,
            upstream=upstream,
        )
        attempt.recheck()
        review_capture.recheck()
        _require_sealed_entry()

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{package.name}.staging-",
                dir=package.parent,
            )
        )
        if (
            staging.parent != package.parent
            or not staging.name.startswith(f".{package.name}.staging-")
            or staging.resolve(strict=True) != staging
        ):
            raise AdjudicationError("adjudication staging directory is not a hidden sibling")
        os.chmod(staging, 0o700)
        payloads, source_rows, attempt_rows = _base_payloads(
            upstream,
            attempt,
            review_capture.payload,
        )
        if evidence.kind == "report":
            assert evidence.report_payload is not None
            payloads[COPIED_AUDIT_NAME] = evidence.report_payload
        else:
            assert evidence.failure_record is not None
            payloads[INPUT_FAILURE_NAME] = _canonical_json_bytes(evidence.failure_record)
        for name, payload in sorted(payloads.items()):
            _write_private_file(staging / name, payload)
        _fsync_directory(staging)

        claim_value = _adjudication_claim_value(
            upstream=upstream,
            attempt=attempt,
            review_sha256=review_capture.sha256,
            requested_disposition=disposition,
            formal_audit_claim=formal_audit_claim,
        )
        _before_adjudication_claim()
        attempt.recheck()
        review_capture.recheck()
        reopened = _load_upstream(
            upstream.root,
            require_live_sources=True,
        )
        if _upstream_identity(reopened) != _upstream_identity(upstream):
            raise AdjudicationError("formal producer changed before adjudication claim")
        _require_sealed_entry()

        def mark_claim_irreversible() -> None:
            nonlocal claim_published
            claim_published = True

        claim_custody = _publish_adjudication_claim(
            staging=staging,
            value=claim_value,
            on_irreversible=mark_claim_irreversible,
        )
        payloads[ADJUDICATION_CLAIM_NAME] = claim_custody.payload
        postclaim_phase = "post_claim_hook"
        _after_adjudication_claim()

        replay: _Replay | None = None
        effective_disposition: Disposition
        outcome_kind: OutcomeKind
        review_role: ReviewRole
        if evidence.kind == "execution_failure":
            # The formal audit claim was consumed upstream, but no citeable
            # outer-finalized report exists.  Adjudication consumes its own
            # version claim and packages rejection without invoking a runner.
            effective_disposition = "rejected"
            outcome_kind = "formal_audit_execution_failure"
            review_role = "execution_failure_review"
        else:
            assert evidence.report_payload is not None
            postclaim_phase = "replay_staging"
            replay_started_payload = _canonical_json_bytes(
                _replay_started_record(
                    upstream=upstream,
                    attempt=attempt,
                    claim_payload=claim_custody.payload,
                )
            )
            _write_private_file(
                staging / ADJUDICATION_REPLAY_STARTED_NAME,
                replay_started_payload,
            )
            _fsync_directory(staging)
            payloads[ADJUDICATION_REPLAY_STARTED_NAME] = replay_started_payload
            _before_adjudication_replay(staging)
            postclaim_phase = "replay_execution"
            replay, failure = _run_bound_replay(upstream)
            _after_adjudication_replay(staging)
            postclaim_phase = "terminal_staging"
            if replay is None:
                assert failure is not None
                _write_partial_replay_evidence(
                    staging=staging,
                    payloads=payloads,
                    failure=failure,
                )
                failure_value = _replay_failure_record(
                    upstream=upstream,
                    attempt=attempt,
                    failure=failure,
                    supplied_audit_payload=evidence.report_payload,
                    requested_disposition=disposition,
                )
                failure_payload = _canonical_json_bytes(failure_value)
                payloads[AUDIT_FAILURE_NAME] = failure_payload
                _write_private_file(
                    staging / AUDIT_FAILURE_NAME,
                    failure_payload,
                )
                effective_disposition = "rejected"
                outcome_kind = "adjudication_replay_failure"
                review_role = "pre_replay_supplied_audit_review"
            else:
                payloads[REPRODUCED_AUDIT_NAME] = replay.stdout
                payloads[AUDIT_STDERR_NAME] = replay.stderr
                payloads[AUDIT_RUNTIME_NAME] = replay.runtime_manifest
                payloads[AUDIT_INVOCATION_NAME] = replay.invocation_manifest
                receipt = _execution_receipt(
                    upstream=upstream,
                    attempt=attempt,
                    replay=replay,
                    supplied_audit_payload=evidence.report_payload,
                )
                receipt_payload = _canonical_json_bytes(receipt)
                payloads[AUDIT_EXECUTION_NAME] = receipt_payload
                for name in (
                    REPRODUCED_AUDIT_NAME,
                    AUDIT_STDERR_NAME,
                    AUDIT_RUNTIME_NAME,
                    AUDIT_INVOCATION_NAME,
                    AUDIT_EXECUTION_NAME,
                ):
                    _write_private_file(staging / name, payloads[name])
                if replay.stdout == evidence.report_payload:
                    effective_disposition = disposition
                    outcome_kind = "audit_report"
                    review_role = "supplied_audit_review"
                else:
                    try:
                        reproduced = _parse_canonical_json_object(
                            replay.stdout,
                            label="byte-distinct reproduced audit",
                        )
                        _verify_audit_identity(
                            reproduced,
                            upstream=upstream,
                        )
                    except (AdjudicationError, OSError, RuntimeError, ValueError) as error:
                        failure = _ExecutionFailure(
                            code="audit-execution-identity-mismatch",
                            error_type=type(error).__name__,
                            partial=_partial_from_validated_replay(
                                replay,
                                phase="audit_identity_validation",
                            ),
                        )
                        _write_partial_replay_evidence(
                            staging=staging,
                            payloads=payloads,
                            failure=failure,
                        )
                        failure_value = _replay_failure_record(
                            upstream=upstream,
                            attempt=attempt,
                            failure=failure,
                            supplied_audit_payload=evidence.report_payload,
                            requested_disposition=disposition,
                        )
                        failure_payload = _canonical_json_bytes(failure_value)
                        payloads[AUDIT_FAILURE_NAME] = failure_payload
                        _write_private_file(
                            staging / AUDIT_FAILURE_NAME,
                            failure_payload,
                        )
                        effective_disposition = "rejected"
                        outcome_kind = "adjudication_replay_failure"
                        review_role = "pre_replay_supplied_audit_review"
                    else:
                        effective_disposition = "rejected"
                        outcome_kind = "audit_replay_mismatch"
                        review_role = "pre_replay_supplied_audit_review"

        manifest = _manifest(
            upstream=upstream,
            attempt=attempt,
            evidence=evidence,
            payloads=payloads,
            source_rows=source_rows,
            attempt_rows=attempt_rows,
            claim_payload=claim_custody.payload,
            requested_disposition=disposition,
            disposition=effective_disposition,
            outcome_kind=outcome_kind,
            review_role=review_role,
            replay=replay,
        )
        postclaim_phase = "pre_package_publish"

        def mark_package_published() -> None:
            nonlocal package_published, postclaim_phase
            package_published = True
            postclaim_phase = "post_package_publish"

        def mark_outer_registered() -> None:
            nonlocal outer_registered, postclaim_phase
            outer_registered = True
            postclaim_phase = "outer_registration"

        _publish_terminal_package(
            staging=staging,
            package=package,
            manifest=manifest,
            attempt=attempt,
            upstream=upstream,
            review_capture=review_capture,
            claim_custody=claim_custody,
            on_package_published=mark_package_published,
            on_outer_registered=mark_outer_registered,
        )
        published = True
        return manifest
    except Exception as error:
        if claim_published and not outer_registered:
            if claim_custody is not None:
                claim_custody.close()
                claim_custody = None
            try:
                return _recover_adjudication_package(
                    recovery_mode="automatic",
                    failure_phase=postclaim_phase,
                    failure_type=type(error).__name__,
                    allow_completed=True,
                )
            except Exception as recovery_error:
                location = (
                    "an exact unfinalized package"
                    if package_published
                    else "the consumed claim"
                )
                raise AdjudicationError(
                    "post-claim adjudication fault left "
                    f"{location}; sealed explicit recovery is required"
                ) from recovery_error
        raise
    finally:
        if claim_custody is not None:
            claim_custody.close()
        if review_capture is not None:
            review_capture.close()
        attempt.close()
        if staging is not None and staging.exists() and not published and not claim_published:
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--audit-attempt",
        type=Path,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--semantic-review",
        type=Path,
        action=_StoreOnce,
    )
    parser.add_argument(
        "--disposition",
        choices=("accepted", "rejected"),
        action=_StoreOnce,
    )
    parser.add_argument(
        "--recover",
        action="count",
        default=0,
        help="recover the already-consumed canonical claim without replay",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.recover:
        if (
            arguments.recover != 1
            or arguments.audit_attempt is not None
            or arguments.semantic_review is not None
            or arguments.disposition is not None
        ):
            parser.error("--recover must be supplied exactly once and without adjudication inputs")
    elif (
        arguments.audit_attempt is None
        or arguments.semantic_review is None
        or arguments.disposition is None
    ):
        parser.error(
            "ordinary adjudication requires --audit-attempt, "
            "--semantic-review, and --disposition"
        )
    try:
        manifest = (
            recover_adjudication_package()
            if arguments.recover
            else create_adjudication_package(
                audit_attempt=cast(Path, arguments.audit_attempt),
                disposition=cast(Disposition, arguments.disposition),
                semantic_review=cast(Path, arguments.semantic_review),
            )
        )
    except (AdjudicationError, OSError) as error:
        print(f"adjudication package refused: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_json_bytes(manifest))
    return 0 if manifest["disposition"] == "accepted" else 1


__all__ = (
    "ADJUDICATION_CLAIM_NAME",
    "ADJUDICATION_MANIFEST_NAME",
    "ADJUDICATION_RECOVERY_FAILURE_NAME",
    "ADJUDICATION_REPLAY_STARTED_NAME",
    "ADJUDICATION_RESULTS_ROOT",
    "AUDIT_EXECUTION_NAME",
    "AUDIT_FAILURE_NAME",
    "COPIED_AUDIT_NAME",
    "COPIED_SEMANTIC_REVIEW_NAME",
    "FORMAL_ADJUDICATION_CLAIM_MARKER",
    "FORMAL_ADJUDICATION_PACKAGE_PATH",
    "FORMAL_SEMANTIC_REVIEW_PATH",
    "INPUT_FAILURE_NAME",
    "REPRODUCED_AUDIT_NAME",
    "AdjudicationError",
    "Disposition",
    "create_adjudication_package",
    "inspect_adjudication_evidence",
    "main",
    "recover_adjudication_package",
    "verify_adjudication_package",
)


if __name__ == "__main__":
    raise SystemExit(main())
