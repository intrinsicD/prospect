"""Immutable producer-side custody for a WM-001 execution attempt."""

from __future__ import annotations

import hashlib
import json
import os
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

REPO = Path(__file__).resolve().parents[2]
LOCKFILE_PATH = REPO / "requirements-wm001.lock"
MANIFEST_NAME = "producer-manifest.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def verify_producer_manifest(output_directory: Path) -> dict[str, Any]:
    """Reopen every producer file and reject omission, mutation, or aliasing."""

    root = output_directory.resolve()
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError("producer manifest must not be a symbolic link")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "prospect.wm001.producer-manifest.v1":
        raise ValueError("wrong producer manifest schema")
    if manifest.get("manifest_excludes") != [MANIFEST_NAME]:
        raise ValueError("producer manifest exclusion contract changed")
    if manifest.get("status") not in {"completed", "failed"}:
        raise ValueError("producer manifest status is invalid")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("producer manifest files must be an array")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
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
        if row.get("bytes") != actual.stat().st_size:
            raise ValueError(f"manifested producer file size changed: {relative}")
        if row.get("sha256") != sha256_file(actual):
            raise ValueError(f"manifested producer file digest changed: {relative}")
    actual_files = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != manifest_path
    }
    if seen != actual_files:
        missing = sorted(actual_files - seen)
        stale = sorted(seen - actual_files)
        raise ValueError(f"producer manifest file set changed; unmanifested={missing}, missing={stale}")
    if manifest.get("file_count") != len(rows):
        raise ValueError("producer manifest file_count changed")
    return manifest


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
        self.output_directory = output_directory.resolve()
        self.lane = lane
        self.started_at_utc = utc_now()
        self._stdout_file: IO[str] | None = None
        self._stderr_file: IO[str] | None = None
        self._stdout_redirect: redirect_stdout[TextIO] | None = None
        self._stderr_redirect: redirect_stderr[TextIO] | None = None
        self._entered = False

    def __enter__(self) -> ProducerAttempt:
        self.output_directory.mkdir(parents=True, exist_ok=False)
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
        binding_path = binding_path.resolve()
        binding = _load_json(binding_path)
        source = binding.get("source", {})
        environment = binding.get("environment", {})
        if not isinstance(source, dict) or not isinstance(environment, dict):
            raise ValueError("formal binding source/environment blocks are invalid")
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
        copies = (
            (binding_path, Path("formal-binding.json")),
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
            (conformance_report, Path(conformance_report.name)),
            *_bound_implementation_copies(source),
        )
        for origin, relative in copies:
            copy_file_exclusive(origin, self.output_directory / relative)
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
        files = [
            {
                "path": candidate.relative_to(self.output_directory).as_posix(),
                "bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
            for candidate in sorted(self.output_directory.rglob("*"))
            if candidate.is_file() and not candidate.is_symlink() and candidate != self.output_directory / MANIFEST_NAME
        ]
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
    "MANIFEST_NAME",
    "ProducerAttempt",
    "atomic_write_exclusive",
    "copy_file_exclusive",
    "sha256_file",
    "verify_producer_manifest",
)
