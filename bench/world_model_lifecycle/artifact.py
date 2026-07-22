"""Immutable producer-side custody for a WM-001 execution attempt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import traceback
import uuid
from contextlib import AbstractContextManager, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, TextIO, cast

from .verify import (
    BINDING_SCHEMA_PATH,
    PROTOCOL_PATH,
    RESULT_SCHEMA_PATH,
    SEAL_PATH,
)


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
            and (candidate / "bench" / "world_model_lifecycle" / "protocol.json").is_file()
        ):
            return candidate
    source_candidate = Path(__file__).resolve().parents[2]
    if (source_candidate / ".git").exists():
        return source_candidate
    raise RuntimeError("WM-001 artifact custody requires a canonical Prospect Git worktree")


REPO = _repository_root()
HERE = Path(__file__).resolve().parent
LOCKFILE_PATH = REPO / "requirements-wm001.lock"
MANIFEST_NAME = "producer-manifest.json"
FORMAL_LAUNCH_NAME = "formal-launch.json"
FORMAL_LAUNCH_MARKER_NAME = "formal-launch-v1.20.0.json"
FORMAL_CONFIRMATION_NAME = "confirmation-v1.20.0"
FORMAL_BINDING_ATTEMPT_MANIFEST_NAME = "formal-binding-operator-attempt.json"
FORMAL_BINDING_OUTER_COMPLETION_NAME = "formal-binding-outer-completion.json"
FORMAL_INPUT_PREFLIGHT_NAME = "formal-input-preflight.json"
DEVELOPMENT_RESULT_QUALIFICATION_NAME = (
    "development-result-qualification.json"
)
FORMAL_RESULTS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results" / "formal"
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_PRODUCER_MANIFEST_FIELDS = {
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


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_absolute_path(
    path: Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    """Reject relative paths and every lexical or filesystem alias."""

    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError(f"{label} must be a canonical absolute path")
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as error:
        raise ValueError(f"{label} cannot be resolved") from error
    if resolved != path or path.is_symlink():
        raise ValueError(f"{label} must not contain a path alias")
    return path


def _stable_regular_file(path: Path, *, label: str) -> tuple[int, str]:
    """Hash one non-symbolic regular file through a stable descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(row: os.stat_result) -> tuple[int, ...]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_uid,
            row.st_gid,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )

    if (
        total != before.st_size
        or identity(before) != identity(after)
        or path.is_symlink()
        or path.stat().st_ino != before.st_ino
        or path.stat().st_dev != before.st_dev
    ):
        raise ValueError(f"{label} changed while read")
    return total, digest.hexdigest()


def _stable_regular_payload(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, os.stat_result]:
    """Read one small regular file through a stable non-following descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} has an invalid type or size")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        total != before.st_size
        or before.st_size != after.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mode != after.st_mode
        or before.st_nlink != after.st_nlink
        or before.st_uid != after.st_uid
        or before.st_gid != after.st_gid
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or path.is_symlink()
        or path.stat().st_dev != before.st_dev
        or path.stat().st_ino != before.st_ino
    ):
        raise ValueError(f"{label} changed while read")
    return b"".join(chunks), before


def _canonical_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """Decode one canonical JSON object while rejecting duplicate keys."""

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(token: str) -> object:
        raise ValueError(f"{label} contains non-finite value {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    if payload != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON plus one newline")
    return value


def _parse_utc_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"producer manifest {field} is not a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"producer manifest {field} is invalid") from error
    if parsed.tzinfo != UTC:
        raise ValueError(f"producer manifest {field} is not UTC")
    return parsed


def _producer_namespace_snapshot(
    root: Path,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Capture ordered path and inode metadata for the complete producer tree."""

    snapshot: list[tuple[str, tuple[int, ...]]] = []
    for path in _regular_producer_files(root):
        metadata = path.lstat()
        snapshot.append(
            (
                path.relative_to(root).as_posix(),
                (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                ),
            )
        )
    return tuple(snapshot)


def atomic_write_exclusive(path: Path, payload: bytes) -> None:
    """Publish bytes atomically while refusing to replace an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to replace immutable evidence: {path}") from None
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def copy_file_exclusive(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required evidence file is missing: {source}")
    atomic_write_exclusive(destination, source.read_bytes())


def formal_launch_marker_path(
    binding_path: Path,
    output_directory: Path,
    *,
    formal_results_root: Path = FORMAL_RESULTS_ROOT,
) -> tuple[Path, str]:
    """Validate the canonical output namespace without creating any path."""

    binding_lexical = binding_path if binding_path.is_absolute() else Path.cwd() / binding_path
    binding_absolute = Path(os.path.abspath(binding_path))
    if (
        binding_lexical != binding_absolute
        or binding_path.is_symlink()
        or not binding_path.is_file()
        or binding_absolute.resolve(strict=True) != binding_absolute
    ):
        raise ValueError("formal binding must be a regular non-symbolic-link file")
    binding_payload = binding_absolute.read_bytes()
    try:
        binding = json.loads(binding_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal binding is not valid JSON") from error
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != "prospect.world-model-lifecycle.formal-binding.v10"
        or binding.get("experiment_id") != "WM-001"
        or not isinstance(binding.get("protocol"), dict)
        or binding["protocol"].get("version") != "1.20.0"
    ):
        raise ValueError("formal binding is not a WM-001 protocol-1.20 binding")
    binding_sha256 = hashlib.sha256(binding_payload).hexdigest()
    results_root_lexical = (
        formal_results_root if formal_results_root.is_absolute() else Path.cwd() / formal_results_root
    )
    results_root = Path(os.path.abspath(formal_results_root))
    if (
        results_root_lexical != results_root
        or formal_results_root.is_symlink()
        or results_root.resolve(strict=False) != results_root
    ):
        raise ValueError("formal results root must not be aliased")
    binding_root = results_root / binding_sha256
    output_lexical = output_directory if output_directory.is_absolute() else Path.cwd() / output_directory
    output = Path(os.path.abspath(output_directory))
    if output_lexical != output or output.resolve(strict=False) != output:
        raise ValueError("formal output path must not be aliased")
    expected_output = binding_root / FORMAL_CONFIRMATION_NAME
    if output != expected_output:
        raise ValueError(
            "formal output must be the exact "
            "results/formal/<binding-sha256>/confirmation-v1.20.0 path"
        )
    return results_root / FORMAL_LAUNCH_MARKER_NAME, binding_sha256


def _claim_formal_launch(
    binding_path: Path,
    output_directory: Path,
    *,
    formal_results_root: Path = FORMAL_RESULTS_ROOT,
) -> tuple[Path, str]:
    """Publish the durable producer record as protocol 1.20's sole formal claim."""

    try:
        from .rehearsal import (
            RehearsalEvidenceError,
            hold_accepted_binding_rehearsal,
        )
    except ImportError as error:
        raise RuntimeError(
            "formal launch rehearsal verifier cannot be imported"
        ) from error

    binding_payload = binding_path.read_bytes()
    (
        binding_attempt_path,
        binding_attempt_manifest,
        binding_attempt_manifest_sha256,
        binding_outer_completion,
        binding_outer_completion_sha256,
    ) = _formal_binding_attempt_evidence(binding_payload)
    runtime_seal_descriptor = getattr(
        sys,
        "_prospect_wm001_runtime_seal_fd",
        None,
    )
    runtime_seal_identity = getattr(
        sys,
        "_prospect_wm001_runtime_seal_identity",
        None,
    )
    bootstrap_descriptor = getattr(
        sys,
        "_prospect_wm001_bootstrap_fd",
        None,
    )
    if (
        type(runtime_seal_descriptor) is not int
        or type(bootstrap_descriptor) is not int
        or not isinstance(runtime_seal_identity, tuple)
        or len(runtime_seal_identity) != 9
    ):
        raise RuntimeError("formal launch requires captured bootstrap authorization")
    try:
        before = os.fstat(runtime_seal_descriptor)
        captured_binding = os.pread(
            runtime_seal_descriptor,
            before.st_size + 1,
            0,
        )
        after = os.fstat(runtime_seal_descriptor)
        bootstrap_metadata = os.fstat(bootstrap_descriptor)
    except OSError as error:
        raise RuntimeError("formal launch bootstrap authorization cannot be reopened") from error
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not stat.S_ISREG(bootstrap_metadata.st_mode)
        or bootstrap_metadata.st_nlink != 1
        or identity != after_identity
        or identity != runtime_seal_identity
        or captured_binding != binding_payload
    ):
        raise RuntimeError("formal launch binding differs from captured bootstrap authorization")
    marker_path, binding_sha256 = formal_launch_marker_path(
        binding_path,
        output_directory,
        formal_results_root=formal_results_root,
    )
    output = Path(os.path.abspath(output_directory))
    if not output.is_dir() or output.is_symlink():
        raise ValueError("formal launch requires an existing non-aliased producer directory")
    if os.path.lexists(marker_path):
        raise RuntimeError("protocol 1.20 already has a formal launch claim; same-version resume or retry is forbidden")
    binding = _load_json(binding_path)
    source = binding.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("formal binding source block is invalid")
    try:
        with hold_accepted_binding_rehearsal(
            binding_attempt_path / "formal-binding.json"
        ) as rehearsal_custody:
            record: dict[str, object] = {
                "schema": "prospect.wm001.formal-launch.v3",
                "experiment_id": "WM-001",
                "protocol_version": "1.20.0",
                "formal_binding_sha256": binding_sha256,
                "formal_binding_attempt_path": str(binding_attempt_path),
                "formal_binding_attempt_manifest_file": FORMAL_BINDING_ATTEMPT_MANIFEST_NAME,
                "formal_binding_attempt_manifest_sha256": binding_attempt_manifest_sha256,
                "formal_binding_outer_completion_file": FORMAL_BINDING_OUTER_COMPLETION_NAME,
                "formal_binding_outer_completion_marker": str(binding_outer_completion),
                "formal_binding_outer_completion_sha256": binding_outer_completion_sha256,
                "accepted_binding_rehearsal": rehearsal_custody.identity_rows(),
                "attempt_directory": output.name,
                "global_marker_file": FORMAL_LAUNCH_MARKER_NAME,
                "claimed_at_utc": utc_now(),
                "git_commit": source.get("git_commit"),
                "git_tree": source.get("git_tree"),
            }
            record["record_sha256"] = hashlib.sha256(
                _canonical_json_bytes(record)
            ).hexdigest()
            producer_record = output / FORMAL_LAUNCH_NAME
            if (
                output / FORMAL_BINDING_ATTEMPT_MANIFEST_NAME
            ).read_bytes() != binding_attempt_manifest or (
                output / FORMAL_BINDING_OUTER_COMPLETION_NAME
            ).read_bytes() != binding_attempt_manifest:
                raise RuntimeError(
                    "preserved formal binding attempt evidence differs from its canonical source"
                )
            atomic_write_exclusive(
                producer_record,
                _canonical_json_bytes(record) + b"\n",
            )
            record_sha256 = sha256_file(producer_record)
            rehearsal_custody.recheck()
            try:
                os.link(producer_record, marker_path)
            except FileExistsError:
                raise RuntimeError(
                    "protocol 1.20 already has a formal launch claim; same-version resume or retry is forbidden"
                ) from None
            for directory in (output, marker_path.parent):
                directory_descriptor = os.open(
                    directory,
                    os.O_RDONLY | os.O_DIRECTORY,
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            if (
                producer_record.is_symlink()
                or marker_path.is_symlink()
                or not producer_record.is_file()
                or not marker_path.is_file()
                or not os.path.samefile(producer_record, marker_path)
                or producer_record.read_bytes() != marker_path.read_bytes()
                or sha256_file(marker_path) != record_sha256
            ):
                raise RuntimeError(
                    "protocol 1.20 launch publication did not preserve one exact inode"
                )
            return marker_path, record_sha256
    except RehearsalEvidenceError as error:
        raise RuntimeError(
            "formal launch requires stable accepted outer-finalized binding rehearsal custody"
        ) from error


def _formal_binding_attempt_evidence(
    binding_payload: bytes,
) -> tuple[Path, bytes, str, Path, str]:
    """Authenticate the sole binding attempt and return its terminal evidence."""

    from .operator import (
        FORMAL_BINDING_ATTEMPT_PATH,
        outer_completion_marker,
        verify_operator_attempt,
        verify_outer_completion,
    )

    attempt = FORMAL_BINDING_ATTEMPT_PATH
    manifest = verify_operator_attempt(attempt)
    primary = manifest.get("primary")
    if (
        manifest.get("kind") != "binding"
        or manifest.get("lane") is not None
        or manifest.get("status") != "accepted"
        or not isinstance(primary, dict)
        or primary.get("binding_file") != "formal-binding.json"
    ):
        raise ValueError("formal launch requires the accepted canonical binding attempt")
    canonical_binding = attempt / "formal-binding.json"
    if canonical_binding.read_bytes() != binding_payload:
        raise ValueError("formal binding differs from the accepted canonical binding attempt")
    terminal = attempt / "operator-attempt.json"
    completion = verify_outer_completion(terminal)
    marker = outer_completion_marker(terminal)
    manifest_payload = terminal.read_bytes()
    marker_payload = marker.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()
    if (
        manifest_payload != marker_payload
        or manifest_sha256 != marker_sha256
        or completion.get("terminal_sha256") != manifest_sha256
    ):
        raise RuntimeError("formal binding attempt outer completion identity is inconsistent")
    return attempt, manifest_payload, manifest_sha256, marker, marker_sha256


def claim_formal_launch(
    binding_path: Path,
    output_directory: Path,
    *,
    formal_results_root: Path = FORMAL_RESULTS_ROOT,
) -> Path:
    """Compatibility wrapper returning the version-scoped marker path."""

    marker, _ = _claim_formal_launch(
        binding_path,
        output_directory,
        formal_results_root=formal_results_root,
    )
    return marker


def claim_formal_launch_with_digest(
    binding_path: Path,
    output_directory: Path,
    *,
    formal_results_root: Path = FORMAL_RESULTS_ROOT,
) -> tuple[Path, str]:
    """Publish the sole marker and return its already-verified record digest."""

    return _claim_formal_launch(
        binding_path,
        output_directory,
        formal_results_root=formal_results_root,
    )


def _verify_producer_manifest(
    output_directory: Path,
    *,
    manifest_nlink: int,
) -> dict[str, Any]:
    """Reopen every producer file with an explicit terminal-link expectation."""

    root = _canonical_absolute_path(
        output_directory,
        label="producer root",
        must_exist=True,
    )
    if not root.is_dir():
        raise ValueError("producer root is not a directory")
    initial_namespace = _producer_namespace_snapshot(root)
    manifest_path = root / MANIFEST_NAME
    manifest_payload, manifest_metadata = _stable_regular_payload(
        manifest_path,
        label="producer manifest",
        maximum_bytes=16 << 20,
    )
    if manifest_metadata.st_nlink != manifest_nlink:
        raise ValueError("producer manifest has the wrong outer-completion link count")
    manifest = _canonical_json_object(
        manifest_payload,
        label="producer manifest",
    )
    if set(manifest) != _PRODUCER_MANIFEST_FIELDS:
        raise ValueError("producer manifest has the wrong top-level field set")
    lane = manifest.get("lane")
    status = manifest.get("status")
    error = manifest.get("error")
    if (
        manifest.get("schema") != "prospect.wm001.producer-manifest.v1"
        or manifest.get("experiment_id") != "WM-001"
        or lane not in {"development", "formal"}
        or status not in {"completed", "failed"}
        or manifest.get("manifest_excludes") != [MANIFEST_NAME]
    ):
        raise ValueError("producer manifest semantic identity is invalid")
    if status == "completed" and error is not None:
        raise ValueError("completed producer manifest must not contain an error")
    if status == "failed" and (
        not isinstance(error, dict)
        or set(error) != {"type", "message"}
        or not isinstance(error.get("type"), str)
        or not error.get("type")
        or not isinstance(error.get("message"), str)
    ):
        raise ValueError("failed producer manifest has an invalid error block")
    started_at = _parse_utc_timestamp(
        manifest.get("started_at_utc"),
        field="started_at_utc",
    )
    completed_at = _parse_utc_timestamp(
        manifest.get("completed_at_utc"),
        field="completed_at_utc",
    )
    if completed_at < started_at:
        raise ValueError("producer manifest completed before it started")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("producer manifest files must be an array")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError(f"producer manifest files[{index}] is not an object")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"producer manifest files[{index}].path is invalid")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
            raise ValueError(f"unsafe producer manifest path: {relative}")
        if relative in seen:
            raise ValueError(f"duplicate producer manifest path: {relative}")
        seen.add(relative)
        actual = root / candidate
        if actual.is_symlink() or not actual.is_file():
            raise ValueError(f"manifested producer file is missing: {relative}")
        metadata = actual.lstat()
        expected_links = 2 if lane == "formal" and relative == FORMAL_LAUNCH_NAME else 1
        if metadata.st_nlink != expected_links:
            raise ValueError(f"manifested producer file has invalid hard-link custody: {relative}")
        observed_bytes, observed_sha256 = _stable_regular_file(
            actual,
            label=f"manifested producer file {relative}",
        )
        if type(row.get("bytes")) is not int or row.get("bytes") != observed_bytes:
            raise ValueError(f"manifested producer file size changed: {relative}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None or digest != observed_sha256:
            raise ValueError(f"manifested producer file digest changed: {relative}")
    actual_files = tuple(
        candidate.relative_to(root).as_posix()
        for candidate in _regular_producer_files(root)
        if candidate != manifest_path
    )
    if tuple(row["path"] for row in rows) != actual_files:
        actual_set = set(actual_files)
        missing = sorted(actual_set - seen)
        stale = sorted(seen - actual_set)
        raise ValueError(f"producer manifest file set changed; unmanifested={missing}, missing={stale}")
    if type(manifest.get("file_count")) is not int or manifest.get("file_count") != len(rows):
        raise ValueError("producer manifest file_count changed")
    final_namespace = _producer_namespace_snapshot(root)
    if initial_namespace != final_namespace:
        raise ValueError("producer file namespace changed during manifest verification")
    return manifest


def _verify_producer_manifest_precommit(
    output_directory: Path,
) -> dict[str, Any]:
    """Verify a terminal manifest before the outer launcher commits it."""

    return _verify_producer_manifest(
        output_directory,
        manifest_nlink=1,
    )


def verify_producer_manifest(output_directory: Path) -> dict[str, Any]:
    """Verify one publicly citeable, outer-finalized producer attempt."""

    manifest = _verify_producer_manifest(
        output_directory,
        manifest_nlink=2,
    )
    from .operator import verify_outer_completion

    verify_outer_completion(
        _canonical_absolute_path(
            output_directory,
            label="producer root",
            must_exist=True,
        )
        / MANIFEST_NAME
    )
    return manifest


def _regular_producer_files(root: Path) -> tuple[Path, ...]:
    """Return every regular entry and reject aliases, specials, or empty dirs."""

    _canonical_absolute_path(root, label="producer root", must_exist=True)
    if not root.is_dir():
        raise ValueError("producer root is not a directory")
    files: list[Path] = []
    directories: list[Path] = []
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
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"producer root contains a symbolic-link directory: {candidate.relative_to(root)}")
            if not stat.S_ISDIR(mode):
                raise ValueError(f"producer root contains a special directory entry: {candidate.relative_to(root)}")
            directories.append(candidate)
        for name in filenames:
            candidate = current / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"producer root contains a symbolic link: {candidate.relative_to(root)}")
            if not stat.S_ISREG(mode):
                raise ValueError(f"producer root contains a special file: {candidate.relative_to(root)}")
            files.append(candidate)
    for producer_directory in directories:
        if not any(path.is_relative_to(producer_directory) for path in files):
            raise ValueError(f"producer root contains an empty directory: {producer_directory.relative_to(root)}")
    return tuple(
        sorted(
            files,
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _safe_sibling(path: Path, filename: object, *, field: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{field} is missing")
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename:
        raise ValueError(f"{field} must be a safe sibling filename")
    return path.parent / relative


def _bound_implementation_copies(source: dict[str, object]) -> tuple[tuple[Path, Path], ...]:
    """Validate bound source rows and map them into the durable source snapshot."""

    manifest = source.get("implementation_files")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("source.implementation_files must be a non-empty array")
    copies: list[tuple[Path, Path]] = []
    relative_paths: list[str] = []
    for index, row in enumerate(manifest):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError(f"source.implementation_files[{index}] is invalid")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"source.implementation_files[{index}].path is invalid")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
            raise ValueError(f"unsafe bound implementation path: {relative}")
        origin = REPO / candidate
        if origin.is_symlink() or not origin.is_file():
            raise FileNotFoundError(f"bound implementation file is missing: {relative}")
        if row.get("bytes") != origin.stat().st_size:
            raise ValueError(f"bound implementation byte size changed: {relative}")
        if row.get("sha256") != sha256_file(origin):
            raise ValueError(f"bound implementation digest changed: {relative}")
        relative_paths.append(relative)
        copies.append((origin, Path("source") / candidate))
    if relative_paths != sorted(set(relative_paths)):
        raise ValueError("source.implementation_files must be unique and ordered by path")
    return tuple(copies)


class _Tee:
    """Minimal synchronous tee used for durable process logs."""

    def __init__(self, terminal: TextIO, evidence: TextIO) -> None:
        self._terminal = terminal
        self._evidence = evidence

    @property
    def encoding(self) -> str:
        return self._terminal.encoding or "utf-8"

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        written = self._terminal.write(value)
        self._evidence.write(value)
        return written

    def flush(self) -> None:
        self._terminal.flush()
        self._evidence.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()

    def fileno(self) -> int:
        return self._terminal.fileno()


class ProducerAttempt(AbstractContextManager["ProducerAttempt"]):
    """Own one append-only attempt directory and its producer manifest."""

    def __init__(self, output_directory: Path, *, lane: str) -> None:
        self.output_directory = _canonical_absolute_path(
            output_directory,
            label="producer output",
            must_exist=False,
        )
        if lane not in {"development", "formal"}:
            raise ValueError("producer lane must be development or formal")
        self.lane = lane
        self.started_at_utc = utc_now()
        self._stdout_file: IO[str] | None = None
        self._stderr_file: IO[str] | None = None
        self._stdout_redirect: redirect_stdout[TextIO] | None = None
        self._stderr_redirect: redirect_stderr[TextIO] | None = None
        self._entered = False

    def __enter__(self) -> ProducerAttempt:
        _canonical_absolute_path(
            self.output_directory.parent,
            label="producer output parent",
            must_exist=True,
        )
        self.output_directory.mkdir(parents=True, exist_ok=False)
        _canonical_absolute_path(
            self.output_directory,
            label="producer output",
            must_exist=True,
        )
        metadata = {
            "schema": "prospect.wm001.producer-attempt.v1",
            "experiment_id": "WM-001",
            "lane": self.lane,
            "started_at_utc": self.started_at_utc,
            "process_id": os.getpid(),
        }
        atomic_write_exclusive(
            self.output_directory / "attempt-metadata.json",
            _canonical_json_bytes(metadata) + b"\n",
        )
        self._stdout_file = (self.output_directory / "main.stdout.log").open(
            "x",
            encoding="utf-8",
            buffering=1,
        )
        self._stderr_file = (self.output_directory / "main.stderr.log").open(
            "x",
            encoding="utf-8",
            buffering=1,
        )
        self._stdout_redirect = redirect_stdout(cast(TextIO, _Tee(sys.stdout, self._stdout_file)))
        self._stderr_redirect = redirect_stderr(cast(TextIO, _Tee(sys.stderr, self._stderr_file)))
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        self._entered = True
        return self

    def preserve_formal_inputs(self, binding_path: Path) -> Path:
        """Copy the full pre-outcome evidence package into this attempt."""

        if not self._entered:
            raise RuntimeError("attempt custody must be entered before preserving inputs")
        from .operator import FORMAL_BINDING_ATTEMPT_PATH, outer_completion_marker

        binding_path = _canonical_absolute_path(
            binding_path,
            label="formal binding",
            must_exist=True,
        )
        if binding_path != FORMAL_BINDING_ATTEMPT_PATH / "formal-binding.json":
            raise ValueError("formal inputs require the canonical accepted binding attempt")
        (
            binding_attempt,
            binding_attempt_manifest_payload,
            _,
            binding_completion,
            _,
        ) = _formal_binding_attempt_evidence(binding_path.read_bytes())
        binding_attempt_manifest = binding_attempt / "operator-attempt.json"
        formal_input_preflight = _canonical_absolute_path(
            binding_attempt / FORMAL_INPUT_PREFLIGHT_NAME,
            label="formal input preflight",
            must_exist=True,
        )
        development_result_qualification = _canonical_absolute_path(
            binding_attempt / DEVELOPMENT_RESULT_QUALIFICATION_NAME,
            label="development result qualification",
            must_exist=True,
        )
        if (
            formal_input_preflight.is_symlink()
            or formal_input_preflight.stat().st_nlink != 1
            or development_result_qualification.is_symlink()
            or development_result_qualification.stat().st_nlink != 1
        ):
            raise RuntimeError(
                "formal input sidecar lacks single-link custody"
            )
        if binding_completion != outer_completion_marker(binding_attempt_manifest):
            raise RuntimeError("formal binding outer-completion path changed")
        if (
            binding_attempt_manifest.read_bytes() != binding_attempt_manifest_payload
            or binding_completion.read_bytes() != binding_attempt_manifest_payload
        ):
            raise RuntimeError("formal binding attempt evidence changed before preservation")
        binding = _load_json(binding_path)
        source = binding.get("source", {})
        environment = binding.get("environment", {})
        irrelevant_control = binding.get("irrelevant_control", {})
        coverage_arithmetic = binding.get("coverage_arithmetic", {})
        development_qualification = binding.get("development_qualification", {})
        audit_execution = binding.get("audit_execution", {})
        if (
            not isinstance(source, dict)
            or not isinstance(environment, dict)
            or not isinstance(irrelevant_control, dict)
            or not isinstance(coverage_arithmetic, dict)
            or not isinstance(development_qualification, dict)
            or not isinstance(audit_execution, dict)
        ):
            raise ValueError("formal binding source/environment/control/coverage/development/audit blocks are invalid")
        result_qualification_sha256 = development_qualification.get(
            "result_qualification_sha256"
        )
        (
            result_qualification_payload,
            result_qualification_metadata,
        ) = _stable_regular_payload(
            development_result_qualification,
            label="development result qualification",
            maximum_bytes=64 << 20,
        )
        if (
            not isinstance(result_qualification_sha256, str)
            or re.fullmatch(
                r"[0-9a-f]{64}", result_qualification_sha256
            )
            is None
            or result_qualification_metadata.st_nlink != 1
            or hashlib.sha256(result_qualification_payload).hexdigest()
            != result_qualification_sha256
        ):
            raise ValueError(
                "development result qualification differs from the "
                "formal binding"
            )
        test_report = _safe_sibling(
            binding_path,
            source.get("test_report_file"),
            field="source.test_report_file",
        )
        conformance_report = _safe_sibling(
            binding_path,
            environment.get("conformance_report_file"),
            field="environment.conformance_report_file",
        )
        oscillator_conformance_report = _safe_sibling(
            binding_path,
            irrelevant_control.get("conformance_report_file"),
            field="irrelevant_control.conformance_report_file",
        )
        coverage_conformance_report = _safe_sibling(
            binding_path,
            coverage_arithmetic.get("conformance_report_file"),
            field="coverage_arithmetic.conformance_report_file",
        )
        test_log_rows = source.get("test_log_files")
        if (
            not isinstance(test_log_rows, list)
            or len(test_log_rows) != 20
            or any(not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} for row in test_log_rows)
        ):
            raise ValueError("formal binding source.test_log_files is invalid")
        test_logs = tuple(
            _safe_sibling(
                binding_path,
                row["path"],
                field=f"source.test_log_files[{index}].path",
            )
            for index, row in enumerate(test_log_rows)
        )
        development_closure = _safe_sibling(
            binding_path,
            development_qualification.get("closure_file"),
            field="development_qualification.closure_file",
        )
        audit_evidence = tuple(
            _safe_sibling(
                binding_path,
                audit_execution.get(f"{prefix}_file"),
                field=f"audit_execution.{prefix}_file",
            )
            for prefix in (
                "bootstrap_source",
                "prebinding_request",
                "prebinding_path_runtime_manifest",
                "prebinding_descriptor_runtime_manifest",
                "prebinding_path_invocation_manifest",
                "prebinding_descriptor_invocation_manifest",
                "prebinding_conformance_report",
                "prebinding_execution_receipt",
                "outcome_runtime_manifest",
                "restart_runtime_conformance_report",
                "restart_runtime_execution_receipt",
            )
        )
        copies = (
            (binding_path, Path("formal-binding.json")),
            (
                binding_attempt_manifest,
                Path(FORMAL_BINDING_ATTEMPT_MANIFEST_NAME),
            ),
            (
                binding_completion,
                Path(FORMAL_BINDING_OUTER_COMPLETION_NAME),
            ),
            (
                formal_input_preflight,
                Path(FORMAL_INPUT_PREFLIGHT_NAME),
            ),
            (PROTOCOL_PATH, Path("protocol.json")),
            (SEAL_PATH, Path("SEALED_PROTOCOL.sha256")),
            (
                BINDING_SCHEMA_PATH,
                Path("schemas") / "formal-binding.schema.json",
            ),
            (
                RESULT_SCHEMA_PATH,
                Path("schemas") / "raw-result.schema.json",
            ),
            (LOCKFILE_PATH, Path("requirements-wm001.lock")),
            (test_report, Path(test_report.name)),
            *((test_log, Path(test_log.name)) for test_log in test_logs),
            (development_closure, Path(development_closure.name)),
            *((audit_evidence_file, Path(audit_evidence_file.name)) for audit_evidence_file in audit_evidence),
            (conformance_report, Path(conformance_report.name)),
            (
                oscillator_conformance_report,
                Path(oscillator_conformance_report.name),
            ),
            (
                coverage_conformance_report,
                Path(coverage_conformance_report.name),
            ),
            *_bound_implementation_copies(source),
        )
        for origin, relative in copies:
            expected_bytes = origin.stat().st_size
            expected_sha256 = sha256_file(origin)
            destination = self.output_directory / relative
            copy_file_exclusive(origin, destination)
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != expected_bytes
                or sha256_file(destination) != expected_sha256
            ):
                raise RuntimeError(f"preserved formal input failed byte verification: {relative}")
        result_qualification_destination = (
            self.output_directory
            / DEVELOPMENT_RESULT_QUALIFICATION_NAME
        )
        atomic_write_exclusive(
            result_qualification_destination,
            result_qualification_payload,
        )
        if (
            result_qualification_destination.is_symlink()
            or not result_qualification_destination.is_file()
            or result_qualification_destination.stat().st_nlink != 1
            or result_qualification_destination.stat().st_size
            != len(result_qualification_payload)
            or sha256_file(result_qualification_destination)
            != result_qualification_sha256
        ):
            raise RuntimeError(
                "preserved development result qualification failed byte "
                "verification"
            )
        return self.output_directory / "formal-binding.json"

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> Literal[False]:
        status = "completed" if exception is None else "failed"
        if exception is not None:
            traceback.print_exception(exception_type, exception, traceback_value)
        self._close_logs()
        self._write_manifest(
            status=status,
            exception_type=None if exception is None else type(exception).__name__,
            exception_message=None if exception is None else str(exception),
        )
        return False

    def _close_logs(self) -> None:
        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(None, None, None)
        if self._stdout_redirect is not None:
            self._stdout_redirect.__exit__(None, None, None)
        for stream in (self._stdout_file, self._stderr_file):
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()

    def _write_manifest(
        self,
        *,
        status: str,
        exception_type: str | None,
        exception_message: str | None,
    ) -> None:
        initial_files = tuple(
            candidate
            for candidate in _regular_producer_files(self.output_directory)
            if candidate != self.output_directory / MANIFEST_NAME
        )
        files: list[dict[str, object]] = []
        for candidate in initial_files:
            relative = candidate.relative_to(self.output_directory).as_posix()
            expected_links = 2 if self.lane == "formal" and relative == FORMAL_LAUNCH_NAME else 1
            if candidate.lstat().st_nlink != expected_links:
                raise ValueError(f"producer file has invalid hard-link custody: {relative}")
            observed_bytes, observed_sha256 = _stable_regular_file(
                candidate,
                label=f"producer file {relative}",
            )
            files.append(
                {
                    "path": relative,
                    "bytes": observed_bytes,
                    "sha256": observed_sha256,
                }
            )
        final_files = tuple(
            candidate
            for candidate in _regular_producer_files(self.output_directory)
            if candidate != self.output_directory / MANIFEST_NAME
        )
        if initial_files != final_files:
            raise ValueError("producer namespace changed while manifest was written")
        manifest = {
            "schema": "prospect.wm001.producer-manifest.v1",
            "experiment_id": "WM-001",
            "lane": self.lane,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": utc_now(),
            "error": (
                None
                if exception_type is None
                else {
                    "type": exception_type,
                    "message": exception_message,
                }
            ),
            "manifest_excludes": [MANIFEST_NAME],
            "file_count": len(files),
            "files": files,
        }
        atomic_write_exclusive(
            self.output_directory / MANIFEST_NAME,
            _canonical_json_bytes(manifest) + b"\n",
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = (
    "FORMAL_BINDING_ATTEMPT_MANIFEST_NAME",
    "FORMAL_BINDING_OUTER_COMPLETION_NAME",
    "FORMAL_CONFIRMATION_NAME",
    "FORMAL_LAUNCH_MARKER_NAME",
    "FORMAL_LAUNCH_NAME",
    "MANIFEST_NAME",
    "ProducerAttempt",
    "atomic_write_exclusive",
    "claim_formal_launch",
    "claim_formal_launch_with_digest",
    "copy_file_exclusive",
    "formal_launch_marker_path",
    "sha256_file",
    "verify_producer_manifest",
)
