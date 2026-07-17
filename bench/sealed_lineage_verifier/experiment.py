"""Sealed lifecycle for the non-scientific LCV-001 lineage/runtime gate.

No historical experiment module is imported here.  Preparation treats MM-007 as
opaque pinned bytes; formal code parses only the copied top-level receipts and
recomputes the small consumer-semantic closure required by a successor assay.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import io
import json
import os
import secrets
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TextIO, cast

import numpy as np

from . import custody, runtime_probe, supervisor

SCHEMA_VERSION: Final = "lcv001-formal-v1"
EXPERIMENT_ID: Final = "LCV-001"
REPO_ROOT: Final = Path("/home/alex/Documents/prospect")
DEFAULT_OUTPUT: Final = Path("bench/sealed_lineage_verifier/results/LCV-001")
EXPECTED_OUTPUT: Final = REPO_ROOT / DEFAULT_OUTPUT
MM007_ROOT: Final = REPO_ROOT / "bench/multimodal_resolution_diagnostics/results/MM-007"
PROTOCOL_DOC: Final = REPO_ROOT / "docs/research/2026-07-16-lcv001-sealed-lineage-runtime-closure-protocol.md"
CONFIG_DOC: Final = REPO_ROOT / "docs/research/2026-07-16-lcv001-config.json"
PRE_REAL_AUDIT_DOC: Final = REPO_ROOT / "docs/research/2026-07-16-lcv001-pre-real-audit.json"
PREPARED_ROOT: Final = Path("prepared")
OUTCOMES_ROOT: Final = Path("outcomes")
PROTOCOL_COPY: Final = PREPARED_ROOT / "LCV-001-protocol.md"
CONFIG_COPY: Final = PREPARED_ROOT / "config.json"
AUDIT_COPY: Final = PREPARED_ROOT / "pre-real-audit.json"
FREEZE_RECORD: Final = PREPARED_ROOT / "freeze-record.json"
INPUT_MANIFEST: Final = PREPARED_ROOT / "input-manifest.json"
PARENT_COPY_ROOT: Final = PREPARED_ROOT / "inputs/MM-007"
PHASE_ANCHOR: Final = OUTCOMES_ROOT / "prepared-phase-anchor.json"
FORMAL_START: Final = OUTCOMES_ROOT / "formal-start.json"
RUNTIME_RECEIPT: Final = OUTCOMES_ROOT / "runtime-receipt.json"
PARENT_CLOSURE: Final = OUTCOMES_ROOT / "parent-closure.json"
MUTATION_CONTROLS: Final = OUTCOMES_ROOT / "mutation-controls.json"
PROVISIONAL_RESULT: Final = OUTCOMES_ROOT / "provisional-result.json"
CLEANUP_RECEIPT: Final = OUTCOMES_ROOT / "cleanup-receipt.json"
RESULTS_FILE: Final = OUTCOMES_ROOT / "LCV-001-results.json"
REPORT_FILE: Final = OUTCOMES_ROOT / "LCV-001-report.md"
ARTIFACT_MANIFEST: Final = OUTCOMES_ROOT / "artifact-manifest.json"
TERMINAL_FAILURE: Final = OUTCOMES_ROOT / "terminal-failure.json"

# sha256, bytes, original sealed mode.  Copies become read-only 0444 while the
# original mode remains authenticated in MM-007's own artifact manifest.
MM007_PINS: Final[dict[Path, tuple[str, int, int]]] = {
    Path("MM-007-evidence.json"): ("13dfa89e541e6122263ea9814d42fb328da303dcc74556cdaaa5d5860d99abaf", 273804, 0o644),
    Path("MM-007-frames-64x64.npz"): (
        "fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f",
        2525160,
        0o644,
    ),
    Path("MM-007-protocol.md"): ("24bbac1855cc2b51d2a65012b9c63037637c53555b86bbad7c66a6249108a73c", 15238, 0o644),
    Path("MM-007-report.md"): ("b18760128941ab2eff893b8c0afc469b92f71077d489e060d56519407990b8a2", 1086, 0o644),
    Path("MM-007-results.json"): ("3c92729e1e5c18c14461e36602bdb86acd31750d9f5a85f535cd33a43fb9c47b", 221177, 0o644),
    Path("artifact-manifest.json"): ("db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb", 2678, 0o644),
    Path("formal-start.json"): ("ea5c7bda870d71ead3172c1fc6e504d6a6b02d2ba785e9fd2fc75a91c667eee3", 2090, 0o444),
    Path("input-manifest.json"): ("1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f", 43618, 0o644),
    Path("inputs/MM-004/input-manifest.json"): (
        "597a8bfc9f6ae1f6ff1f0d3be456f57d768f1866a1fac59cd981dd260076dc90",
        31328,
        0o644,
    ),
    Path("inputs/MM-006/MM-006-evidence.json"): (
        "5c5ffa514ab0f0c06c8588e69b54d6c4f2f6be3a4471a0fe7a31aa1e1dd3dac2",
        3350469,
        0o644,
    ),
    Path("inputs/MM-006/MM-006-results.json"): (
        "c5e0737acf6030315a77b497f5d5ea78693eb8a5879399e37bdfb702e2b9f648",
        208329,
        0o644,
    ),
    Path("inputs/MM-006/artifact-manifest.json"): (
        "9727eefc6c5665b5eb8cc65ae9cfab57bb4c8b3e353b747363bec5e3c2f573b0",
        2381,
        0o644,
    ),
    Path("inputs/MM-006/input-manifest.json"): (
        "badd7676f1e4a60c56b59af12a1d7f82ef134e797febc86a5adcb4c33dda5cd1",
        24723,
        0o644,
    ),
    Path("inputs/MM-006/inputs/MM-005/inputs/MM-004/MM-004-pixel-grids.npz"): (
        "cca261a941e68a7ddc510eee3a3af958d33b6abaf958cb5562ed6b66c22f47c8",
        409427,
        0o644,
    ),
}
MM007_ARTIFACT_SHA256: Final = MM007_PINS[Path("artifact-manifest.json")][0]
MM007_INPUT_SHA256: Final = MM007_PINS[Path("input-manifest.json")][0]
FRAME_PACKAGE_SHA256: Final = MM007_PINS[Path("MM-007-frames-64x64.npz")][0]
FRAME_ARRAY_SHA256: Final = {
    "frames_uint8": "46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9",
    "timestamps": "128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55",
    "video_ids": "06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50",
}
MM007_CLASSIFICATION: Final = "physically_matched_resolution_failure_supported"
MM006_CLASSIFICATION: Final = "tested_pixel_warp_ceiling_failure_supported"
MATCHED_IDENTITY_SHA256: Final = "d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b"
R8_CURRENT_SHA256: Final = "587d28455a0bd0226f24c94a60ce6bd6ea9bee6bf05ec2a315089e6e10ffd787"
VIDEO_IDS: Final = (
    "video_10993",
    "video_1580",
    "video_2564",
    "video_3501",
    "video_6860",
    "video_8241",
    "video_874",
    "video_9253",
)
RAW_COUNTS: Final = {
    "video_10993": 63,
    "video_1580": 64,
    "video_2564": 59,
    "video_3501": 65,
    "video_6860": 65,
    "video_8241": 48,
    "video_874": 66,
    "video_9253": 47,
}
MATCHED_COUNTS: Final = {name: count - 3 for name, count in RAW_COUNTS.items()}
FOLDS: Final = tuple(
    {
        "index": index,
        "test": VIDEO_IDS[2 * index : 2 * index + 2],
        "train": tuple(name for name in VIDEO_IDS if name not in VIDEO_IDS[2 * index : 2 * index + 2]),
    }
    for index in range(4)
)

SOURCE_FILES: Final = (
    Path("bench/sealed_lineage_verifier/__init__.py"),
    Path("bench/sealed_lineage_verifier/__main__.py"),
    Path("bench/sealed_lineage_verifier/bootstrap.py"),
    Path("bench/sealed_lineage_verifier/canary_probe.py"),
    Path("bench/sealed_lineage_verifier/custody.py"),
    Path("bench/sealed_lineage_verifier/experiment.py"),
    Path("bench/sealed_lineage_verifier/runtime_probe.py"),
    Path("bench/sealed_lineage_verifier/supervisor.py"),
    Path("tests/test_lcv001_custody.py"),
    Path("tests/test_lcv001_experiment.py"),
    Path("tests/test_lcv001_runtime.py"),
    Path("tests/test_lcv001_supervisor.py"),
)
RUNTIME_SOURCE_ROOT: Final = PREPARED_ROOT / "runtime/source"
RUNTIME_SOURCE_FILES: Final = tuple(RUNTIME_SOURCE_ROOT / path for path in SOURCE_FILES)
RUNTIME_EMPTY_PARENT: Final = RUNTIME_SOURCE_ROOT / "bench/__init__.py"
RUNTIME_FILES: Final = (RUNTIME_EMPTY_PARENT, *RUNTIME_SOURCE_FILES)
PARENT_FILES: Final = tuple(sorted(MM007_PINS, key=str))
PARENT_DIRECTORIES: Final = custody.expected_directories(MM007_PINS)
PARENT_SOURCE_DIRECTORY_MODES: Final = {Path("."): 0o775, **{path: 0o775 for path in PARENT_DIRECTORIES}}
PREPARED_FILES: Final = (
    PROTOCOL_COPY,
    CONFIG_COPY,
    AUDIT_COPY,
    INPUT_MANIFEST,
    FREEZE_RECORD,
    PHASE_ANCHOR,
    *(PARENT_COPY_ROOT / path for path in PARENT_FILES),
    *RUNTIME_FILES,
)
OUTCOME_FILES: Final = (
    FORMAL_START,
    RUNTIME_RECEIPT,
    PARENT_CLOSURE,
    MUTATION_CONTROLS,
    PROVISIONAL_RESULT,
    CLEANUP_RECEIPT,
    RESULTS_FILE,
    REPORT_FILE,
)
ARTIFACT_FILES: Final = (*PREPARED_FILES, *OUTCOME_FILES)
COMPLETED_FILES: Final = (*ARTIFACT_FILES, ARTIFACT_MANIFEST)
PROVISIONAL_FILES: Final = (
    *PREPARED_FILES,
    FORMAL_START,
    RUNTIME_RECEIPT,
    PARENT_CLOSURE,
    MUTATION_CONTROLS,
    PROVISIONAL_RESULT,
)
READ_ONLY_MODE: Final = 0o444
NORMALIZER_STRIDES: Final = (768, 256, 32, 4)
WRONG_NORMALIZER_STRIDES: Final = (768, 4, 96, 12)
WRONG_NORMALIZER_FINGERPRINTS: Final = (
    "5dd95ccf5f6c7223ed38e1521d5c6641d445e72f48f1156bbb9a3e991c787c43",
    "a7574f2f3eb44335a6ecc94c24c4237a89c699cae69ece0ab2d1f3b964706326",
    "4a3e35c605bf42641668754856e78e514fae3ddc6c659720832a2e5bb032e77e",
    "b4ba8c0a9c8dcc3f54be1f87cc3458d4f57a9ad989a7f12d6e0853180307e2fb",
)
WRONG_NORMALIZER_MAX_ABS: Final = (
    1.6486811915683575e-14,
    9.492406860545088e-15,
    4.9404924595819466e-15,
    1.1157741397482823e-14,
)
WRONG_NORMALIZER_MAX_ULP: Final = (297, 171, 89, 201)
EXPECTED_CONTROL_COUNT: Final = 25
CLASSIFIED_CHILD_EXIT: Final = 2
RUNTIME_TRANSPORT_EXIT: Final = 70
RUNTIME_TRANSPORT_RETURN_CODES: Final = frozenset((-2, RUNTIME_TRANSPORT_EXIT, 120, 130))


class InvalidLCV001Artifact(ValueError):
    """The sealed parent or LCV package is not authentic and closed."""

    classification = "invalid_LCV001_artifact"


class InvalidLCV001Runtime(ValueError):
    """The formal/semantic child differs from the pinned runtime."""

    classification = "invalid_LCV001_runtime"


class InvalidLCV001Verifier(ValueError):
    """A negative control exposed a defect in the LCV-owned verifier."""

    classification = "invalid_LCV001_verifier"


def _safe_exception_text(error: BaseException, limit: int = 1000) -> str:
    try:
        rendered = str(error)
    except BaseException:
        rendered = "<exception text unavailable>"
    return rendered[-limit:]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _cli_json_line(value: object, *, compact: bool) -> str:
    separators = (",", ":") if compact else None
    return json.dumps(value, sort_keys=True, separators=separators, allow_nan=False) + "\n"


def _write_cli_line(payload: str, stream: TextIO) -> None:
    if stream.write(payload) != len(payload):
        raise OSError("CLI stream made a short write")
    stream.flush()


def _write_classified_cli_failure(error: BaseException, *, transport: bool) -> int:
    if not isinstance(error, (InvalidLCV001Artifact, InvalidLCV001Runtime, InvalidLCV001Verifier)):
        raise InvalidLCV001Verifier("CLI failure was not classified before emission")
    payload = _cli_json_line(
        {"classification": error.classification, "error": _safe_exception_text(error)},
        compact=False,
    )
    try:
        _write_cli_line(payload, sys.stderr)
    except (OSError, ValueError, KeyboardInterrupt):
        # The parent recognizes this dedicated transport code even if no structured
        # diagnostic survives a second stdout/stderr or interruption failure.
        return RUNTIME_TRANSPORT_EXIT
    return RUNTIME_TRANSPORT_EXIT if transport else CLASSIFIED_CHILD_EXIT


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise InvalidLCV001Artifact(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _json_from_bytes(payload: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise InvalidLCV001Artifact(f"non-finite JSON constant in {label}: {value}")

    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidLCV001Artifact(f"invalid JSON: {label}") from error


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_read(path: Path) -> tuple[bytes, dict[str, object]]:
    """Bind a path, descriptor, exact length, and post-read path identity."""

    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise InvalidLCV001Artifact(f"required file is missing: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise InvalidLCV001Artifact(f"required path is not a regular non-symlink file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise InvalidLCV001Artifact(f"file identity changed before open: {path}")
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise InvalidLCV001Artifact(f"file ended before its bound size: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise InvalidLCV001Artifact(f"file grew while read: {path}")
        after_descriptor = os.fstat(descriptor)
        try:
            after_path = path.lstat()
        except FileNotFoundError as error:
            raise InvalidLCV001Artifact(f"file disappeared while read: {path}") from error
        if _identity(after_descriptor) != _identity(opened) or _identity(after_path) != _identity(opened):
            raise InvalidLCV001Artifact(f"file mutated or was replaced while read: {path}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return payload, {
        "bytes": len(payload),
        "mode": stat.S_IMODE(before.st_mode),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tree_census(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    try:
        metadata = root.lstat()
    except FileNotFoundError as error:
        raise InvalidLCV001Artifact(f"tree root is missing: {root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise InvalidLCV001Artifact(f"tree root is not a real directory: {root}")
    files: list[Path] = []
    directories: list[Path] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        names.sort()
        filenames.sort()
        for name in names:
            child = base / name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISDIR(child_metadata.st_mode):
                raise InvalidLCV001Artifact(f"tree contains a symlink or special directory: {child}")
            directories.append(child.relative_to(root))
        for name in filenames:
            child = base / name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISREG(child_metadata.st_mode):
                raise InvalidLCV001Artifact(f"tree contains a symlink or special file: {child}")
            files.append(child.relative_to(root))
    return tuple(sorted(files, key=str)), tuple(sorted(directories, key=str))


def _expected_directories(files: Sequence[Path]) -> tuple[Path, ...]:
    directories: set[Path] = set()
    for path in files:
        parent = path.parent
        while parent != Path("."):
            directories.add(parent)
            parent = parent.parent
    return tuple(sorted(directories, key=str))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdirs(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise InvalidLCV001Artifact(f"directory ancestor is not a real directory: {cursor}")
    for item in reversed(missing):
        item.mkdir(mode=0o700)
        os.chmod(item, 0o755, follow_symlinks=False)
        _fsync_directory(item.parent)


def _workspace_allocation_checkpoint(name: str, path: Path) -> None:
    """Fault-injection boundary after exclusive directory ownership."""

    del name, path


def _private_workspace_path(parent: Path, prefix: str) -> Path:
    for _attempt in range(128):
        try:
            token = secrets.token_hex(16)
        except OSError as error:
            raise InvalidLCV001Runtime(f"LCV-001 private workspace randomness failed: {error}") from error
        candidate = parent / f"{prefix}{token}"
        try:
            candidate.lstat()
        except FileNotFoundError:
            return candidate
        except OSError as error:
            raise InvalidLCV001Runtime(f"LCV-001 private workspace lookup failed: {error}") from error
    raise InvalidLCV001Runtime("LCV-001 could not allocate a unique private workspace name")


def _create_owned_workspace(
    path: Path,
    *,
    mode: int,
    ownership: list[bool],
    checkpoint: str,
) -> None:
    if ownership != [False]:
        raise InvalidLCV001Verifier("LCV-001 private workspace ownership state differs before creation")
    try:
        os.mkdir(path, mode)
        ownership[0] = True
        _workspace_allocation_checkpoint(checkpoint, path)
        os.chmod(path, mode, follow_symlinks=False)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        if ownership[0]:
            try:
                custody.remove_created_tree(path)
                ownership[0] = False
            except custody.CustodyError as cleanup_error:
                raise InvalidLCV001Runtime(
                    f"LCV-001 private workspace cleanup failed after allocation: {cleanup_error}"
                ) from cleanup_error
            raise InvalidLCV001Runtime(
                "LCV-001 private workspace failed after exclusive creation"
            ) from error
        raise InvalidLCV001Runtime("LCV-001 private workspace name appeared before creation") from error
    except BaseException as error:
        try:
            if ownership[0] or (path.exists() and not path.is_symlink()):
                custody.remove_created_tree(path)
                ownership[0] = False
        except custody.CustodyError as cleanup_error:
            raise InvalidLCV001Runtime(
                f"LCV-001 private workspace cleanup failed after allocation: {cleanup_error}"
            ) from cleanup_error
        if isinstance(error, OSError):
            raise InvalidLCV001Runtime(f"LCV-001 private workspace creation failed: {error}") from error
        raise


def _write_exclusive(path: Path, payload: bytes, mode: int = READ_ONLY_MODE) -> None:
    _mkdirs(path.parent)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _file_record(path: Path) -> dict[str, object]:
    return _stable_read(path)[1]


def _payload_record(payload: bytes, mode: int = READ_ONLY_MODE) -> custody.ExpectedFile:
    return custody.ExpectedFile(hashlib.sha256(payload).hexdigest(), len(payload), mode)


def _record_dict(record: custody.ExpectedFile) -> dict[str, object]:
    return {"bytes": record.bytes, "mode": record.mode, "sha256": record.sha256}


def _snapshot_record(snapshot: custody.TreeSnapshot, relative: Path) -> dict[str, object]:
    try:
        return _record_dict(snapshot.records[relative])
    except KeyError as error:
        raise InvalidLCV001Artifact(f"sealed snapshot omits required file: {relative}") from error


def _snapshot_json(snapshot: custody.TreeSnapshot, relative: Path) -> dict[str, Any]:
    try:
        payload = snapshot.payloads[relative]
    except KeyError as error:
        raise InvalidLCV001Artifact(f"sealed snapshot omits required JSON: {relative}") from error
    value = _json_from_bytes(payload, str(relative))
    if not isinstance(value, dict):
        raise InvalidLCV001Artifact(f"LCV-001 JSON root is not an object: {relative}")
    return cast(dict[str, Any], value)


def _copy_expected_record(value: object, label: str) -> custody.ExpectedFile:
    record = _require_mapping(value, label)
    try:
        return custody.ExpectedFile(cast(str, record["sha256"]), cast(int, record["bytes"]), READ_ONLY_MODE)
    except KeyError as error:
        raise InvalidLCV001Artifact(f"{label} omits its byte/hash record") from error


def _source_snapshot() -> tuple[dict[Path, bytes], dict[str, dict[str, object]]]:
    payloads: dict[Path, bytes] = {}
    records: dict[str, dict[str, object]] = {}
    for path in SOURCE_FILES:
        payload, record = _stable_read(REPO_ROOT / path)
        payloads[path] = payload
        records[str(path)] = record
    return payloads, records


def _runtime_source_expectation(source: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    output = {
        str(RUNTIME_SOURCE_ROOT / Path(path)): {
            "bytes": record["bytes"],
            "mode": READ_ONLY_MODE,
            "sha256": record["sha256"],
        }
        for path, record in source.items()
    }
    output[str(RUNTIME_EMPTY_PARENT)] = {
        "bytes": 0,
        "mode": READ_ONLY_MODE,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    return output


def _runtime_source_records(output: Path) -> dict[str, dict[str, object]]:
    return {str(path): _file_record(output / path) for path in RUNTIME_FILES}


def _source_sha256(records: Mapping[str, object]) -> str:
    return _canonical_sha256(records)


def _validate_live_source(manifest: Mapping[str, object]) -> None:
    """Bind live parent-side orchestration to the source frozen at prepare time."""

    _payloads, observed = _source_snapshot()
    expected = _require_mapping(manifest.get("source"), "LCV-001 source manifest")
    if observed != expected or manifest.get("source_sha256") != _source_sha256(observed):
        raise InvalidLCV001Artifact("LCV-001 live orchestration source differs from the frozen source")


def _runtime_expectation() -> dict[str, object]:
    return {
        "base_executable": str(runtime_probe.BASE_EXECUTABLE),
        "base_prefix": str(runtime_probe.BASE_PREFIX),
        "canary_bundle_sha256": runtime_probe.CANARY_BUNDLE_SHA256,
        "executable": str(runtime_probe.LEXICAL_PYTHON),
        "numpy": runtime_probe.NUMPY_VERSION,
        "openblas_sha256": runtime_probe.OPENBLAS_SHA256,
        "prefix": str(runtime_probe.BASE_PREFIX),
        "venv_anchor": str(runtime_probe.VENV_PREFIX),
        "python": runtime_probe.PYTHON_VERSION,
        "python_terminal_sha256": runtime_probe.PYTHON_TERMINAL_SHA256,
        "supervisor": supervisor.manifest(),
        "thread_environment": dict(sorted(runtime_probe.THREAD_ENVIRONMENT.items())),
    }


def _config_payload() -> dict[str, object]:
    return {
        "classifications": {
            "artifact": InvalidLCV001Artifact.classification,
            "pass": "PASS",
            "runtime": InvalidLCV001Runtime.classification,
            "verifier": InvalidLCV001Verifier.classification,
        },
        "experiment_id": EXPERIMENT_ID,
        "lifecycle": {
            "canonical_output": str(DEFAULT_OUTPUT),
            "classified_child_return_code": CLASSIFIED_CHILD_EXIT,
            "completion_commit": "sealed_verified_sibling_RENAME_EXCHANGE",
            "completed_files": [str(path) for path in COMPLETED_FILES],
            "completed_mode": "files_0444_directories_0555",
            "postcommit_receipt_loss": "no_terminal_write_manual_inspection_no_PASS",
            "prepared_files": [str(path) for path in PREPARED_FILES],
            "provisional_files": [str(path) for path in PROVISIONAL_FILES],
            "required_preparation_receipt": {
                "parent_directory_fsync": True,
                "publication_warnings": [],
                "status": "prepared_durable",
            },
            "required_commit_receipt": {
                "atomic_exchange": True,
                "parent_directory_fsync": True,
                "retired_provisional_removed": True,
                "warnings": [],
            },
            "runtime_transport_return_codes": sorted(RUNTIME_TRANSPORT_RETURN_CODES),
            "shadow_flags": ["-I", "-S", "-B"],
            "supervisor_return_contract": "exact_type_args_int_rc_str_channels_valid_cleanup",
            "terminal_membership": "subset_of_provisional_files_plus_terminal_failure",
            "workspace_allocation": "token_hex_128bit_absent_precheck_exclusive_mkdir_owned_cleanup",
        },
        "parent": {
            "directory_modes": {str(path): mode for path, mode in PARENT_SOURCE_DIRECTORY_MODES.items()},
            "files": {
                str(path): {"bytes": size, "mode": mode, "sha256": digest}
                for path, (digest, size, mode) in MM007_PINS.items()
            },
            "root": str(MM007_ROOT.relative_to(REPO_ROOT)),
        },
        "runtime": {
            "base_executable": str(runtime_probe.BASE_EXECUTABLE),
            "base_prefix_under_S": str(runtime_probe.BASE_PREFIX),
            "canary": {
                "bundle_sha256": runtime_probe.CANARY_BUNDLE_SHA256,
                "input_sha256": runtime_probe.CANARY_INPUT_SHA256,
                "minimum_singular_gap": runtime_probe.CANARY_MIN_GAP,
                "relative_reconstruction_error": runtime_probe.CANARY_RELATIVE_RECONSTRUCTION,
                "s_sha256": runtime_probe.CANARY_S_SHA256,
                "shape": [512, 256],
                "u_sha256": runtime_probe.CANARY_U_SHA256,
                "vh_sha256": runtime_probe.CANARY_VH_SHA256,
            },
            "dependency_closures": {
                "numpy": {
                    "bytes": runtime_probe.NUMPY_CLOSURE_BYTES,
                    "count": runtime_probe.NUMPY_CLOSURE_COUNT,
                    "manifest_sha256": runtime_probe.NUMPY_CLOSURE_SHA256,
                },
                "stdlib": {
                    "bytes": runtime_probe.STDLIB_CLOSURE_BYTES,
                    "count": runtime_probe.STDLIB_CLOSURE_COUNT,
                    "manifest_sha256": runtime_probe.STDLIB_CLOSURE_SHA256,
                },
            },
            "dependency_hashes": {
                "numpy_init": runtime_probe.NUMPY_INIT_SHA256,
                "numpy_multiarray": runtime_probe.MULTIARRAY_SHA256,
                "numpy_umath_linalg": runtime_probe.UMATH_LINALG_SHA256,
                "openblas": runtime_probe.OPENBLAS_SHA256,
                "pyvenv_cfg": runtime_probe.PYVENV_CFG_SHA256,
                "python_terminal": runtime_probe.PYTHON_TERMINAL_SHA256,
            },
            "environment": runtime_probe.frozen_environment(),
            "lexical_executable": str(runtime_probe.LEXICAL_PYTHON),
            "numpy_version": runtime_probe.NUMPY_VERSION,
            "openblas_config": runtime_probe.OPENBLAS_CONFIG,
            "openblas_core": runtime_probe.OPENBLAS_CORE,
            "platform_receipt": "Linux-6.14.0-37-generic-x86_64-with-glibc2.39",
            "python_version": runtime_probe.PYTHON_VERSION,
            "sensitivity_bundles": {
                "base_2p1p3_threads1": "c0b4b0dde34f85cb8441446f104d1cc4b33fb55c1fe51f7afb709eeb7cf47334",
                "base_2p1p3_threads16": "ae753e31d8a34268ae356c3216e3647036fff832786c31f59006e9575e129b3f",
                "venv_2p4p6_threads16": "f6b21b168c3bd999df24b9a940f3d01a178c3429feb7978e5d429445c1dc7865",
            },
            "symlink_chain": [list(record) for record in runtime_probe.SYMLINK_CHAIN],
            "supervisor": supervisor.manifest(),
        },
        "schema_version": SCHEMA_VERSION,
        "semantics": {
            "matched_identity_sha256": MATCHED_IDENTITY_SHA256,
            "matched_rows": 453,
            "normalizer_order": "select_fold_training_current_indices_before_pool",
            "normalizer_strides": list(NORMALIZER_STRIDES),
            "raw_rows": 477,
            "scope": "frame_and_source_current_normalizer_consistency_only",
        },
    }


def _audit_binding(*, config_sha256: str, protocol_sha256: str, source_sha256: str) -> dict[str, object]:
    return {
        "config_sha256": config_sha256,
        "parent_artifact_manifest_sha256": MM007_ARTIFACT_SHA256,
        "protocol_sha256": protocol_sha256,
        "source_sha256": source_sha256,
    }


def _validate_pre_real_audit(value: object, binding: Mapping[str, object]) -> None:
    audit = _require_mapping(value, "LCV-001 pre-real audit")
    if set(audit) != {
        "binding",
        "checks",
        "decision",
        "experiment_id",
        "independent",
        "reviewer",
        "schema_version",
    }:
        raise InvalidLCV001Artifact("LCV-001 pre-real audit schema differs")
    checks = audit.get("checks")
    if (
        audit.get("experiment_id") != EXPERIMENT_ID
        or audit.get("schema_version") != "lcv001-pre-real-audit-v1"
        or audit.get("decision") != "GO"
        or audit.get("independent") is not True
        or not isinstance(audit.get("reviewer"), str)
        or not cast(str, audit["reviewer"]).strip()
        or audit.get("binding") != binding
        or not isinstance(checks, dict)
        or set(checks)
        != {
            "canonical_output_absent",
            "claim_boundary_accepted",
            "parent_pins_independently_checked",
            "runtime_closure_independently_checked",
            "source_tests_audited",
        }
        or not all(value is True for value in checks.values())
    ):
        raise InvalidLCV001Artifact("LCV-001 independent pre-real audit is not an exact GO")


def _prepared_manifest(
    protocol_record: Mapping[str, object],
    config_record: Mapping[str, object],
    audit_record: Mapping[str, object],
    source: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    parent = {
        str(path): {"bytes": values[1], "mode": values[2], "sha256": values[0]} for path, values in MM007_PINS.items()
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "expected_completed_files": [str(path) for path in COMPLETED_FILES],
        "expected_prepared_files": [str(path) for path in PREPARED_FILES],
        "parent": {
            "artifact_manifest_sha256": MM007_ARTIFACT_SHA256,
            "copy_root": str(PARENT_COPY_ROOT),
            "file_count": len(MM007_PINS),
            "files": parent,
            "live_root": str(MM007_ROOT.relative_to(REPO_ROOT)),
        },
        "audit": {"copy": str(AUDIT_COPY), **dict(audit_record)},
        "config": {"copy": str(CONFIG_COPY), **dict(config_record)},
        "protocol": {"copy": str(PROTOCOL_COPY), **dict(protocol_record)},
        "runtime": _runtime_expectation(),
        "runtime_source": _runtime_source_expectation(source),
        "schema_version": SCHEMA_VERSION,
        "source": dict(source),
        "source_sha256": _source_sha256(source),
        "status": "prepared_before_formal_verification",
    }


def _read_parent_snapshot(root: Path, *, copied: bool) -> dict[Path, bytes]:
    try:
        snapshot = (
            custody.verify_sealed_tree(root, MM007_PINS)
            if copied
            else custody.snapshot_exact_tree(
                root,
                MM007_PINS,
                directory_modes=PARENT_SOURCE_DIRECTORY_MODES,
            )
        )
    except custody.CustodyError as error:
        raise InvalidLCV001Artifact(str(error)) from error
    return dict(snapshot.payloads)


def _freeze_payload_from_records(
    manifest: Mapping[str, object], records: Mapping[Path, custody.ExpectedFile]
) -> dict[str, object]:
    runtime_records = {str(path): _record_dict(records[path]) for path in RUNTIME_FILES}
    return {
        "audit_sha256": records[AUDIT_COPY].sha256,
        "config_sha256": records[CONFIG_COPY].sha256,
        "experiment_id": EXPERIMENT_ID,
        "input_manifest_sha256": records[INPUT_MANIFEST].sha256,
        "parent_tree_sha256": _canonical_sha256({str(path): MM007_PINS[path][0] for path in PARENT_FILES}),
        "protocol_sha256": records[PROTOCOL_COPY].sha256,
        "runtime_source_sha256": _canonical_sha256(runtime_records),
        "schema_version": SCHEMA_VERSION,
        "source_sha256": manifest["source_sha256"],
        "status": "implementation_and_inputs_frozen_before_formal_marker",
    }


def _prepared_expectations(
    manifest: Mapping[str, Any], input_manifest_payload: bytes
) -> tuple[dict[Path, custody.ExpectedFile], dict[str, object]]:
    records: dict[Path, custody.ExpectedFile] = {
        PROTOCOL_COPY: _copy_expected_record(manifest.get("protocol"), "LCV-001 protocol record"),
        CONFIG_COPY: _copy_expected_record(manifest.get("config"), "LCV-001 config record"),
        AUDIT_COPY: _copy_expected_record(manifest.get("audit"), "LCV-001 audit record"),
        INPUT_MANIFEST: _payload_record(input_manifest_payload),
    }
    records.update(
        {
            PARENT_COPY_ROOT / path: custody.ExpectedFile(digest, size, READ_ONLY_MODE)
            for path, (digest, size, _source_mode) in MM007_PINS.items()
        }
    )
    runtime_source = _require_mapping(manifest.get("runtime_source"), "LCV-001 runtime source manifest")
    if set(runtime_source) != {str(path) for path in RUNTIME_FILES}:
        raise InvalidLCV001Artifact("LCV-001 runtime source membership differs")
    for path in RUNTIME_FILES:
        records[path] = _copy_expected_record(runtime_source[str(path)], f"runtime source {path}")
    freeze = _freeze_payload_from_records(manifest, records)
    records[FREEZE_RECORD] = _payload_record(_json_bytes(freeze))
    records[PHASE_ANCHOR] = _payload_record(_json_bytes(_phase_anchor_payload()))
    if set(records) != set(PREPARED_FILES):
        raise InvalidLCV001Artifact("LCV-001 prepared expectation membership differs")
    return records, freeze


def _prepared_directory_modes() -> dict[Path, int]:
    return {
        Path("."): 0o555,
        **{
            path: 0o755 if path == OUTCOMES_ROOT else 0o555
            for path in custody.expected_directories({item: ("0" * 64, 0, 0o444) for item in PREPARED_FILES})
        },
    }


def _phase_anchor_payload() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_root_sealed_outcomes_directory_open",
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one sibling staging directory without replacement."""

    if source.parent != destination.parent:
        raise InvalidLCV001Runtime("LCV-001 preparation publication paths are not siblings")
    parent = -1
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        parent = os.open(source.parent, flags)
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent,
            os.fsencode(source.name),
            parent,
            os.fsencode(destination.name),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise InvalidLCV001Artifact("LCV-001 output appeared before exclusive publish")
            raise InvalidLCV001Runtime(f"LCV-001 atomic publish failed: errno={error_number}")
    except OSError as error:
        raise InvalidLCV001Runtime(f"LCV-001 atomic publish failed: {error}") from error
    finally:
        if parent >= 0:
            os.close(parent)


def _rename_exchange(source: Path, destination: Path) -> None:
    """Atomically exchange two existing directories through retained parent fds."""

    source_parent = -1
    destination_parent = -1
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        source_parent = os.open(source.parent, flags)
        destination_parent = os.open(destination.parent, flags)
        if os.fstat(source_parent).st_dev != os.fstat(destination_parent).st_dev:
            raise InvalidLCV001Runtime("LCV-001 completion exchange paths are on different filesystems")
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_parent,
            os.fsencode(source.name),
            destination_parent,
            os.fsencode(destination.name),
            2,
        )
        if result != 0:
            raise InvalidLCV001Runtime(f"LCV-001 atomic completion exchange failed: errno={ctypes.get_errno()}")
    except OSError as error:
        raise InvalidLCV001Runtime(f"LCV-001 atomic completion exchange failed: {error}") from error
    finally:
        if destination_parent >= 0:
            os.close(destination_parent)
        if source_parent >= 0:
            os.close(source_parent)


def _seal_prepared_subtree(output: Path, records: Mapping[Path, custody.ExpectedFile]) -> None:
    relative_records = {
        path.relative_to(PREPARED_ROOT): record
        for path, record in records.items()
        if path == PREPARED_ROOT or PREPARED_ROOT in path.parents
    }
    try:
        custody.seal_copied_tree(output / PREPARED_ROOT, relative_records)
    except custody.CustodyError as error:
        raise InvalidLCV001Artifact(str(error)) from error
    os.chmod(output, 0o555, follow_symlinks=False)
    _fsync_directory(output)


def _prepare_at(
    output: Path = DEFAULT_OUTPUT,
    parent: Path = MM007_ROOT,
    audit: Path = PRE_REAL_AUDIT_DOC,
) -> dict[str, object]:
    """Copy opaque sealed parent bytes; never parse or execute them."""

    destination = output if output.is_absolute() else REPO_ROOT / output
    parent_root = parent if parent.is_absolute() else REPO_ROOT / parent
    audit_path = audit if audit.is_absolute() else REPO_ROOT / audit
    if destination.exists() or destination.is_symlink():
        raise InvalidLCV001Artifact("LCV-001 output must be absent before preparation")
    # Acquire and authenticate every parent byte before creating output.
    try:
        parent_snapshot = custody.snapshot_exact_tree(
            parent_root,
            MM007_PINS,
            directory_modes=PARENT_SOURCE_DIRECTORY_MODES,
        )
    except custody.CustodyError as error:
        raise InvalidLCV001Artifact(str(error)) from error
    protocol_payload, protocol_record = _stable_read(PROTOCOL_DOC)
    config_payload, config_record = _stable_read(CONFIG_DOC)
    if _json_from_bytes(config_payload, str(CONFIG_DOC)) != _config_payload():
        raise InvalidLCV001Artifact("LCV-001 external config differs from the frozen code config")
    source_payloads, source = _source_snapshot()
    audit_payload, audit_record = _stable_read(audit_path)
    audit_binding = _audit_binding(
        config_sha256=cast(str, config_record["sha256"]),
        protocol_sha256=cast(str, protocol_record["sha256"]),
        source_sha256=_source_sha256(source),
    )
    _validate_pre_real_audit(_json_from_bytes(audit_payload, str(audit_path)), audit_binding)
    manifest = _prepared_manifest(protocol_record, config_record, audit_record, source)
    _mkdirs(destination.parent)
    staging = _private_workspace_path(destination.parent, f".{destination.name}.prepare-")
    staging_owned = [False]
    published = False
    parent_directory_fsync = False
    staging_identity: tuple[int, int] | None = None
    publication_warnings: list[dict[str, str]] = []
    try:
        _create_owned_workspace(
            staging,
            mode=0o755,
            ownership=staging_owned,
            checkpoint="preparation_after_mkdir",
        )
        _write_exclusive(staging / PROTOCOL_COPY, protocol_payload)
        _write_exclusive(staging / CONFIG_COPY, config_payload)
        _write_exclusive(staging / AUDIT_COPY, audit_payload)
        manifest_payload = _json_bytes(manifest)
        _write_exclusive(staging / INPUT_MANIFEST, manifest_payload)
        _mkdirs((staging / PARENT_COPY_ROOT).parent)
        custody.write_snapshot_exclusive(staging / PARENT_COPY_ROOT, parent_snapshot)
        _write_exclusive(staging / RUNTIME_EMPTY_PARENT, b"")
        for relative in SOURCE_FILES:
            _write_exclusive(staging / RUNTIME_SOURCE_ROOT / relative, source_payloads[relative])
        records, freeze = _prepared_expectations(manifest, manifest_payload)
        _write_exclusive(staging / FREEZE_RECORD, _json_bytes(freeze))
        _write_exclusive(staging / PHASE_ANCHOR, _json_bytes(_phase_anchor_payload()))
        _seal_prepared_subtree(staging, records)
        _validate_prepared(staging)
        staging_metadata = staging.lstat()
        staging_identity = staging_metadata.st_dev, staging_metadata.st_ino
        _rename_noreplace(staging, destination)
        published = True
        try:
            _fsync_directory(destination.parent)
            parent_directory_fsync = True
        except BaseException as error:
            # The namespace publication already committed.  Reporting an ordinary
            # preparation failure here would falsely imply that the visible,
            # independently verifiable destination was absent.
            publication_warnings.append(
                {"error_type": type(error).__name__, "phase": "postrename_parent_fsync"}
            )
    except BaseException as error:
        if published:
            publication_warnings.append(
                {"error_type": type(error).__name__, "phase": "publication_return_boundary"}
            )
        elif staging_identity is not None:
            try:
                destination_metadata = destination.lstat()
            except OSError:
                destination_metadata = None
            if (
                destination_metadata is not None
                and stat.S_ISDIR(destination_metadata.st_mode)
                and not stat.S_ISLNK(destination_metadata.st_mode)
                and (destination_metadata.st_dev, destination_metadata.st_ino) == staging_identity
                and not staging.exists()
                and not staging.is_symlink()
            ):
                published = True
                publication_warnings.append(
                    {"error_type": type(error).__name__, "phase": "publication_return_boundary"}
                )
        if not published and staging_owned[0] and staging.exists() and not staging.is_symlink():
            try:
                custody.remove_created_tree(staging)
                staging_owned[0] = False
            except custody.CustodyError as cleanup_error:
                raise InvalidLCV001Artifact(
                    f"LCV-001 staging cleanup failed after preparation error: {cleanup_error}"
                ) from cleanup_error
        if not published:
            raise
    return {
        "experiment_id": EXPERIMENT_ID,
        "file_count": len(PREPARED_FILES),
        "output": str(destination),
        "parent_artifact_manifest_sha256": MM007_ARTIFACT_SHA256,
        "parent_directory_fsync": parent_directory_fsync,
        "publication_warnings": publication_warnings,
        "status": (
            "prepared_durable"
            if parent_directory_fsync
            else "prepared_namespace_committed_durability_unconfirmed"
        ),
    }


def prepare(
    output: Path = DEFAULT_OUTPUT,
    parent: Path = MM007_ROOT,
    audit: Path = PRE_REAL_AUDIT_DOC,
) -> dict[str, object]:
    """Prepare only the one canonical LCV-001 identity."""

    destination = output if output.is_absolute() else REPO_ROOT / output
    parent_root = parent if parent.is_absolute() else REPO_ROOT / parent
    audit_path = audit if audit.is_absolute() else REPO_ROOT / audit
    if destination != EXPECTED_OUTPUT or parent_root != MM007_ROOT or audit_path != PRE_REAL_AUDIT_DOC:
        raise InvalidLCV001Artifact("public preparation is restricted to canonical LCV-001 paths")
    return _prepare_at(destination, parent_root, audit_path)


def _validated_prepared_snapshot(output: Path) -> tuple[dict[str, object], custody.TreeSnapshot]:
    manifest_payload, manifest_record = _stable_read(output / INPUT_MANIFEST)
    if manifest_record["mode"] != READ_ONLY_MODE:
        raise InvalidLCV001Artifact("LCV-001 input manifest is not immutable 0444")
    manifest = _json_from_bytes(manifest_payload, str(INPUT_MANIFEST))
    if not isinstance(manifest, dict):
        raise InvalidLCV001Artifact("LCV-001 input manifest is not an object")
    records, freeze = _prepared_expectations(manifest, manifest_payload)
    try:
        snapshot = custody.snapshot_exact_tree(
            output,
            records,
            directory_modes=_prepared_directory_modes(),
        )
    except custody.CustodyError as error:
        raise InvalidLCV001Artifact(str(error)) from error
    _validate_input_manifest(manifest, snapshot, freeze)
    return cast(dict[str, object], manifest), snapshot


def _validate_prepared(output: Path) -> dict[str, object]:
    return _validated_prepared_snapshot(output)[0]


def _validate_input_manifest(
    manifest: Mapping[str, Any],
    snapshot: custody.TreeSnapshot,
    freeze: Mapping[str, object],
) -> None:
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "prepared_before_formal_verification"
        or manifest.get("expected_prepared_files") != [str(path) for path in PREPARED_FILES]
        or manifest.get("expected_completed_files") != [str(path) for path in COMPLETED_FILES]
        or manifest.get("runtime") != _runtime_expectation()
    ):
        raise InvalidLCV001Artifact("LCV-001 input manifest identity/config differs")
    source = _require_mapping(manifest.get("source"), "LCV-001 source manifest")
    if set(source) != {str(path) for path in SOURCE_FILES}:
        raise InvalidLCV001Artifact("LCV-001 source membership differs")
    for path in SOURCE_FILES:
        record = _require_mapping(source[str(path)], f"LCV-001 source record {path}")
        if (
            set(record) != {"bytes", "mode", "sha256"}
            or type(record.get("bytes")) is not int
            or cast(int, record["bytes"]) < 0
            or type(record.get("mode")) is not int
            or not isinstance(record.get("sha256"), str)
            or len(cast(str, record["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in cast(str, record["sha256"]))
        ):
            raise InvalidLCV001Artifact(f"LCV-001 source record schema differs: {path}")
    if manifest.get("source_sha256") != _source_sha256(source):
        raise InvalidLCV001Artifact("LCV-001 source aggregate differs")
    if manifest.get("runtime_source") != _runtime_source_expectation(cast(Mapping[str, Mapping[str, object]], source)):
        raise InvalidLCV001Artifact("LCV-001 source/runtime-source cross-link differs")
    runtime_source_records = {str(path): _snapshot_record(snapshot, path) for path in RUNTIME_FILES}
    if manifest.get("runtime_source") != runtime_source_records:
        raise InvalidLCV001Artifact("LCV-001 copied shadow source differs")
    if _snapshot_json(snapshot, CONFIG_COPY) != _config_payload():
        raise InvalidLCV001Artifact("LCV-001 copied config differs")
    config = _require_mapping(manifest.get("config"), "LCV-001 config record")
    audit = _require_mapping(manifest.get("audit"), "LCV-001 audit record")
    config_record = _snapshot_record(snapshot, CONFIG_COPY)
    audit_record = _snapshot_record(snapshot, AUDIT_COPY)
    if (
        config.get("copy") != str(CONFIG_COPY)
        or config.get("bytes") != config_record["bytes"]
        or config.get("sha256") != config_record["sha256"]
        or audit.get("copy") != str(AUDIT_COPY)
        or audit.get("bytes") != audit_record["bytes"]
        or audit.get("sha256") != audit_record["sha256"]
    ):
        raise InvalidLCV001Artifact("LCV-001 copied config/audit record differs")
    binding = _audit_binding(
        config_sha256=cast(str, config_record["sha256"]),
        protocol_sha256=cast(str, _snapshot_record(snapshot, PROTOCOL_COPY)["sha256"]),
        source_sha256=cast(str, manifest["source_sha256"]),
    )
    _validate_pre_real_audit(_snapshot_json(snapshot, AUDIT_COPY), binding)
    protocol = _require_mapping(manifest.get("protocol"), "LCV-001 protocol record")
    copied_protocol = _snapshot_record(snapshot, PROTOCOL_COPY)
    if (
        protocol.get("copy") != str(PROTOCOL_COPY)
        or protocol.get("bytes") != copied_protocol["bytes"]
        or protocol.get("sha256") != copied_protocol["sha256"]
    ):
        raise InvalidLCV001Artifact("LCV-001 copied protocol binding differs")
    if _snapshot_json(snapshot, FREEZE_RECORD) != freeze:
        raise InvalidLCV001Artifact("LCV-001 freeze record differs")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mm007_float_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256(b"mm007-array-v1")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _parse_parent_json(snapshot: Mapping[Path, bytes], path: str) -> dict[str, Any]:
    relative = Path(path)
    value = _json_from_bytes(snapshot[relative], path)
    if not isinstance(value, dict):
        raise InvalidLCV001Artifact(f"MM-007 JSON root is not an object: {path}")
    return cast(dict[str, Any], value)


def _validate_mm007_artifact_manifest(value: Mapping[str, Any]) -> None:
    if set(value) != {"artifacts", "experiment_id", "frame_array_sha256", "frame_package_sha256", "schema_version"}:
        raise InvalidLCV001Artifact("MM-007 artifact-manifest schema differs")
    if (
        value["experiment_id"] != "MM-007"
        or value["schema_version"] != "mm007-formal-v1"
        or value["frame_array_sha256"] != FRAME_ARRAY_SHA256
        or value["frame_package_sha256"] != FRAME_PACKAGE_SHA256
    ):
        raise InvalidLCV001Artifact("MM-007 artifact-manifest anchors differ")
    records = value["artifacts"]
    expected_paths = set(PARENT_FILES) - {Path("artifact-manifest.json")}
    if not isinstance(records, dict) or set(records) != {str(path) for path in expected_paths}:
        raise InvalidLCV001Artifact("MM-007 artifact-manifest does not bind exactly 13 files")
    for relative in expected_paths:
        digest, size, mode = MM007_PINS[relative]
        if records[str(relative)] != {"bytes": size, "mode": mode, "sha256": digest}:
            raise InvalidLCV001Artifact(f"MM-007 artifact record differs: {relative}")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InvalidLCV001Artifact(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _validate_mm007_crosslinks(
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    marker: Mapping[str, Any],
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, object]:
    _validate_mm007_artifact_manifest(artifact)
    if set(manifest) != {
        "config",
        "config_sha256",
        "dependencies",
        "expected_completed_files",
        "expected_prepared_files",
        "experiment_id",
        "input_validation",
        "media_lineage",
        "parent",
        "parent_replay",
        "prepared_membership_sha256",
        "protocol",
        "schema_version",
        "source",
        "source_count",
        "status",
    }:
        raise InvalidLCV001Artifact("MM-007 input-manifest schema differs")
    if (
        manifest.get("experiment_id") != "MM-007"
        or manifest.get("schema_version") != "mm007-formal-v1"
        or manifest.get("status") != "prepared_before_formal_execution"
    ):
        raise InvalidLCV001Artifact("MM-007 input-manifest identity differs")
    completed = manifest.get("expected_completed_files")
    if (
        not isinstance(completed, list)
        or len(completed) != 14
        or set(completed) != {str(path) for path in PARENT_FILES}
    ):
        raise InvalidLCV001Artifact("MM-007 declared completed membership differs")
    validation = _require_mapping(manifest.get("input_validation"), "MM-007 input_validation")
    if set(validation) != {
        "frame_array_sha256",
        "frame_file",
        "frame_package_sha256",
        "frame_schema",
        "frames_uint8_sha256",
        "low_resolution_parity",
        "media_contract",
        "media_contract_sha256",
        "parent_alignment_sha256",
    }:
        raise InvalidLCV001Artifact("MM-007 input-validation schema differs")
    if (
        validation.get("frame_array_sha256") != FRAME_ARRAY_SHA256
        or validation.get("frame_package_sha256") != FRAME_PACKAGE_SHA256
        or validation.get("frames_uint8_sha256") != FRAME_ARRAY_SHA256["frames_uint8"]
    ):
        raise InvalidLCV001Artifact("MM-007 input frame anchors differ")
    frame_file = _require_mapping(validation.get("frame_file"), "MM-007 frame_file")
    if frame_file != {
        "bytes": MM007_PINS[Path("MM-007-frames-64x64.npz")][1],
        "mode": MM007_PINS[Path("MM-007-frames-64x64.npz")][2],
        "sha256": FRAME_PACKAGE_SHA256,
    }:
        raise InvalidLCV001Artifact("MM-007 frame-file record differs")
    if set(marker) != {
        "config_sha256",
        "experiment_id",
        "frame_array_sha256",
        "frame_file_sha256",
        "frame_package_sha256",
        "frame_schema_sha256",
        "frames_uint8_sha256",
        "input_manifest_sha256",
        "media_contract_sha256",
        "mm004_receipt_sha256",
        "mm006_receipt_sha256",
        "parent_alignment_sha256",
        "prepared_membership_sha256",
        "protocol_sha256",
        "schema_version",
        "source_sha256",
        "status",
    } or (
        marker.get("experiment_id") != "MM-007"
        or marker.get("schema_version") != "mm007-formal-v1"
        or marker.get("status") != "formal_execution_started"
        or marker.get("input_manifest_sha256") != MM007_INPUT_SHA256
        or marker.get("frame_array_sha256") != FRAME_ARRAY_SHA256
        or marker.get("frame_file_sha256") != FRAME_PACKAGE_SHA256
        or marker.get("frame_package_sha256") != FRAME_PACKAGE_SHA256
        or marker.get("frames_uint8_sha256") != FRAME_ARRAY_SHA256["frames_uint8"]
        or marker.get("frame_schema_sha256") != _canonical_sha256(validation["frame_schema"])
        or marker.get("parent_alignment_sha256") != validation["parent_alignment_sha256"]
    ):
        raise InvalidLCV001Artifact("MM-007 formal marker cross-links differ")
    if set(evidence) != {
        "alignment",
        "normalizer_rows",
        "parent_classification",
        "real_metric_rows",
        "schema_version",
        "synthetic_expectations",
        "synthetic_rows",
        "synthetic_seed_map",
    } or (
        evidence.get("schema_version") != "mm007-method-v1"
        or evidence.get("parent_classification") != MM006_CLASSIFICATION
    ):
        raise InvalidLCV001Artifact("MM-007 evidence identity differs")
    if set(result) != {
        "epistemic_role",
        "evidence_sha256",
        "experiment_id",
        "formal_start",
        "frame_array_sha256",
        "frame_file_sha256",
        "frame_package_sha256",
        "frames_uint8_sha256",
        "parent_alignment_sha256",
        "parent_classification",
        "schema_version",
        "status",
        "summary",
    } or (
        result.get("experiment_id") != "MM-007"
        or result.get("schema_version") != "mm007-formal-v1"
        or result.get("status") != "completed"
        or result.get("formal_start") != marker
        or result.get("evidence_sha256") != _canonical_sha256(evidence)
        or result.get("frame_array_sha256") != FRAME_ARRAY_SHA256
        or result.get("frame_file_sha256") != FRAME_PACKAGE_SHA256
        or result.get("frame_package_sha256") != FRAME_PACKAGE_SHA256
        or result.get("frames_uint8_sha256") != FRAME_ARRAY_SHA256["frames_uint8"]
        or result.get("parent_alignment_sha256") != validation["parent_alignment_sha256"]
    ):
        raise InvalidLCV001Artifact("MM-007 result cross-links differ")
    summary = _require_mapping(result.get("summary"), "MM-007 summary")
    if set(summary) != {
        "alignment",
        "claim_boundary",
        "decision",
        "experiment_id",
        "families",
        "parent_classification",
        "relative_to_r8",
        "schema_version",
        "synthetic_control",
        "synthetic_expectations",
        "synthetic_seed_map",
    }:
        raise InvalidLCV001Artifact("MM-007 summary schema differs")
    decision = _require_mapping(summary.get("decision"), "MM-007 decision")
    if set(decision) != {"classification", "mechanism_labels", "onset_resolution", "recommended_next_step"}:
        raise InvalidLCV001Artifact("MM-007 decision schema differs")
    if (
        decision.get("classification") != MM007_CLASSIFICATION
        or summary.get("parent_classification") != MM006_CLASSIFICATION
        or result.get("parent_classification") != MM006_CLASSIFICATION
        or summary.get("alignment") != evidence.get("alignment")
    ):
        raise InvalidLCV001Artifact("MM-007 classification or summary/evidence cross-link differs")
    return {
        "artifact_manifest_canonical_sha256": _canonical_sha256(artifact),
        "artifact_manifest_file_sha256": MM007_ARTIFACT_SHA256,
        "classification": MM007_CLASSIFICATION,
        "evidence_canonical_sha256": _canonical_sha256(evidence),
        "evidence_file_sha256": MM007_PINS[Path("MM-007-evidence.json")][0],
        "input_manifest_file_sha256": MM007_INPUT_SHA256,
        "result_file_sha256": MM007_PINS[Path("MM-007-results.json")][0],
    }


def _load_frame_arrays(payload: bytes) -> dict[str, np.ndarray]:
    expected_members = (
        ("frames_uint8.npy", 5_861_504, 2_524_251, 0x423AA397, np.dtype("u1"), (477, 64, 64, 3)),
        ("timestamps.npy", 3_944, 262, 0x9978D848, np.dtype("<f8"), (477,)),
        ("video_ids.npy", 21_116, 251, 0x34E49F60, np.dtype("<U11"), (477,)),
    )
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as zipped:
            records = zipped.infolist()
            if [record.filename for record in records] != [record[0] for record in expected_members]:
                raise InvalidLCV001Artifact("MM-007 NPZ member order/membership differs")
            if len({record.filename for record in records}) != len(records):
                raise InvalidLCV001Artifact("MM-007 NPZ contains duplicate members")
            for record, expected in zip(records, expected_members, strict=True):
                name, size, compressed, crc, dtype, shape = expected
                if (
                    record.filename != name
                    or Path(name).name != name
                    or record.file_size != size
                    or record.compress_size != compressed
                    or record.CRC != crc
                    or record.compress_type != zipfile.ZIP_DEFLATED
                ):
                    raise InvalidLCV001Artifact(f"MM-007 NPZ member metadata differs: {name}")
                member = io.BytesIO(zipped.read(record))
                version = np.lib.format.read_magic(member)
                if version != (1, 0):
                    raise InvalidLCV001Artifact(f"MM-007 NPY version differs: {name}")
                observed_shape, fortran, observed_dtype = np.lib.format.read_array_header_1_0(member)
                if (
                    observed_shape != shape
                    or fortran is not False
                    or observed_dtype != dtype
                    or observed_dtype.hasobject
                ):
                    raise InvalidLCV001Artifact(f"MM-007 NPY header differs: {name}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"video_ids", "timestamps", "frames_uint8"}:
                raise InvalidLCV001Artifact("MM-007 frame archive membership differs")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise InvalidLCV001Artifact("MM-007 frame archive cannot be loaded safely") from error
    return arrays


def _validate_frame_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "video_ids": (np.dtype("<U11"), (477,)),
        "timestamps": (np.dtype("<f8"), (477,)),
        "frames_uint8": (np.dtype("u1"), (477, 64, 64, 3)),
    }
    if set(arrays) != set(expected):
        raise InvalidLCV001Artifact("MM-007 frame array membership differs")
    for name, (dtype, shape) in expected.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise InvalidLCV001Artifact(f"MM-007 frame array schema/layout differs: {name}")
        if name == "timestamps" and not np.all(np.isfinite(value)):
            raise InvalidLCV001Artifact("MM-007 timestamps are non-finite")
        if _array_sha256(value) != FRAME_ARRAY_SHA256[name]:
            raise InvalidLCV001Artifact(f"MM-007 frame array digest differs: {name}")


def _causal_indices(video_ids: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_ids = np.asarray([name for name in VIDEO_IDS for _ in range(RAW_COUNTS[name])], dtype="<U11")
    expected_times = np.asarray(
        [1.0 + 0.5 * index for name in VIDEO_IDS for index in range(RAW_COUNTS[name])], dtype="<f8"
    )
    if not np.array_equal(video_ids, expected_ids) or not np.array_equal(timestamps, expected_times):
        raise InvalidLCV001Artifact("MM-007 raw video/timestamp identities differ")
    previous: list[int] = []
    current: list[int] = []
    future: list[int] = []
    offset = 0
    for name in VIDEO_IDS:
        for position in range(1, RAW_COUNTS[name] - 2):
            previous.append(offset + position - 1)
            current.append(offset + position)
            future.append(offset + position + 1)
        offset += RAW_COUNTS[name]
    return (
        np.asarray(previous, dtype=np.int64),
        np.asarray(current, dtype=np.int64),
        np.asarray(future, dtype=np.int64),
    )


def _validate_r8_layout(value: np.ndarray, rows: int) -> None:
    if (
        value.dtype != np.dtype("<f4")
        or value.shape != (rows, 3, 8, 8)
        or not value.flags.c_contiguous
        or value.strides != NORMALIZER_STRIDES
    ):
        raise InvalidLCV001Artifact("R8 select-before-pool layout/strides differ")


def _pool_selected(frames: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.dtype != np.dtype("<i8") or indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise InvalidLCV001Artifact("selected frame indices are not one unique int64 vector")
    if len(indices) == 0 or int(np.min(indices)) < 0 or int(np.max(indices)) >= len(frames):
        raise InvalidLCV001Artifact("selected frame indices are outside the copied archive")
    selected = np.ascontiguousarray(frames[indices])
    values = selected.astype(np.float32) / np.float32(255.0)
    values = values.reshape(len(values), 8, 8, 8, 8, 3)
    pooled = np.asarray(np.mean(values, axis=(2, 4), dtype=np.float64), dtype=np.float32)
    output = np.ascontiguousarray(np.transpose(pooled, (0, 3, 1, 2)), dtype="<f4")
    _validate_r8_layout(output, len(indices))
    return output


def _validate_pooling_order(value: str) -> None:
    if value != "select_fold_training_current_indices_before_pool":
        raise InvalidLCV001Artifact("normalizer pooling/data-selection order differs")


def _normalizer_fingerprint(mean: np.ndarray, scale: np.ndarray) -> str:
    joined = np.ascontiguousarray(np.concatenate((mean.reshape(-1), scale.reshape(-1))), dtype="<f8")
    return _mm007_float_array_sha256(joined)


def _expected_train_current_indices(
    fold: Mapping[str, object], current: np.ndarray, matched_ids: np.ndarray
) -> np.ndarray:
    train_ids = cast(tuple[str, ...], fold["train"])
    mask = np.asarray([str(name) in train_ids for name in matched_ids], dtype=bool)
    return np.ascontiguousarray(current[mask], dtype="<i8")


def _validate_train_current_indices(
    candidate: np.ndarray,
    fold: Mapping[str, object],
    current: np.ndarray,
    matched_ids: np.ndarray,
) -> None:
    expected = _expected_train_current_indices(fold, current, matched_ids)
    if (
        candidate.dtype != np.dtype("<i8")
        or not candidate.flags.c_contiguous
        or not np.array_equal(candidate, expected)
    ):
        raise InvalidLCV001Artifact("fold training-current index vector differs")


def _recompute_normalizers(
    frames: np.ndarray, current: np.ndarray, matched_ids: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output: list[dict[str, object]] = []
    index_receipts: list[dict[str, object]] = []
    for fold in FOLDS:
        _validate_pooling_order("select_fold_training_current_indices_before_pool")
        train_ids = cast(tuple[str, ...], fold["train"])
        test_ids = cast(tuple[str, ...], fold["test"])
        train_indices = _expected_train_current_indices(fold, current, matched_ids)
        _validate_train_current_indices(train_indices, fold, current, matched_ids)
        train_r8 = _pool_selected(frames, train_indices)
        train = np.asarray(train_r8, dtype=np.float64)
        mean = np.mean(train, axis=(0, 2, 3), keepdims=True)
        scale = np.maximum(np.std(train, axis=(0, 2, 3), keepdims=True), 1e-6)
        output.append(
            {
                "fingerprint": _normalizer_fingerprint(mean, scale),
                "fold": fold["index"],
                "mean": mean.reshape(-1).tolist(),
                "scale": scale.reshape(-1).tolist(),
                "test_video_ids": list(test_ids),
                "train_rows": len(train_indices),
                "train_video_ids": list(train_ids),
                "uses_target": False,
            }
        )
        index_receipts.append(
            {
                "fold": fold["index"],
                "index_sha256": _array_sha256(train_indices),
                "rows": len(train_indices),
                "r8_sha256": _array_sha256(train_r8),
                "r8_strides": list(train_r8.strides),
            }
        )
    return output, index_receipts


def _validate_normalizer_rows(value: object, recomputed: Sequence[Mapping[str, object]]) -> None:
    if not isinstance(value, list) or len(value) != 16:
        raise InvalidLCV001Artifact("MM-007 stored normalizer membership differs")
    expected_keys = {
        "fingerprint",
        "fold",
        "mean",
        "resolution",
        "scale",
        "test_video_ids",
        "train_rows",
        "train_video_ids",
        "uses_target",
    }
    seen: set[tuple[int, int]] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise InvalidLCV001Artifact("MM-007 stored normalizer schema differs")
        resolution = item["resolution"]
        fold_index = item["fold"]
        if type(resolution) is not int or resolution not in (8, 16, 32, 64):
            raise InvalidLCV001Artifact("MM-007 normalizer resolution differs")
        if type(fold_index) is not int or fold_index not in range(4) or (resolution, fold_index) in seen:
            raise InvalidLCV001Artifact("MM-007 normalizer fold scope differs")
        if (
            item["uses_target"] is not False
            or type(item["train_rows"]) is not int
            or not isinstance(item["fingerprint"], str)
            or not isinstance(item["mean"], list)
            or not isinstance(item["scale"], list)
            or len(item["mean"]) != 3
            or len(item["scale"]) != 3
            or any(type(number) is not float for number in [*item["mean"], *item["scale"]])
            or not isinstance(item["train_video_ids"], list)
            or not isinstance(item["test_video_ids"], list)
            or any(not isinstance(name, str) for name in [*item["train_video_ids"], *item["test_video_ids"]])
        ):
            raise InvalidLCV001Artifact("MM-007 normalizer field types differ")
        seen.add((resolution, fold_index))
        reference = dict(recomputed[fold_index])
        observed = dict(item)
        observed.pop("resolution")
        if observed != reference:
            raise InvalidLCV001Artifact("MM-007 source-only normalizer does not replay bit-exactly")
        mean = np.asarray(item["mean"], dtype="<f8")
        scale = np.asarray(item["scale"], dtype="<f8")
        if item["fingerprint"] != _normalizer_fingerprint(mean, scale):
            raise InvalidLCV001Artifact("MM-007 stored normalizer fingerprint differs")
    if seen != {(resolution, fold) for resolution in (8, 16, 32, 64) for fold in range(4)}:
        raise InvalidLCV001Artifact("MM-007 normalizer scopes are incomplete")


def _wrong_select_after_pool_rows(
    frames: np.ndarray,
    current: np.ndarray,
    matched_ids: np.ndarray,
    correct: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = np.ascontiguousarray(frames[current])
    values = selected.astype(np.float32) / np.float32(255.0)
    values = values.reshape(len(values), 8, 8, 8, 8, 3)
    pooled = np.asarray(np.mean(values, axis=(2, 4), dtype=np.float64), dtype=np.float32)
    all_current_wrong = np.transpose(pooled, (0, 3, 1, 2))
    if all_current_wrong.flags.c_contiguous or all_current_wrong.strides != WRONG_NORMALIZER_STRIDES:
        raise InvalidLCV001Verifier("wrong-layout control no longer reaches its preregistered implementation")
    wrong: list[dict[str, object]] = []
    max_abs: list[float] = []
    max_ulp: list[int] = []
    for fold in FOLDS:
        train_ids = cast(tuple[str, ...], fold["train"])
        test_ids = cast(tuple[str, ...], fold["test"])
        mask = np.asarray([str(name) in train_ids for name in matched_ids], dtype=bool)
        train_wrong = all_current_wrong[mask]
        if train_wrong.strides != WRONG_NORMALIZER_STRIDES:
            raise InvalidLCV001Verifier("wrong-layout fold selection no longer preserves the endpoint strides")
        train = np.asarray(train_wrong, dtype=np.float64)
        mean = np.mean(train, axis=(0, 2, 3), keepdims=True)
        scale = np.maximum(np.std(train, axis=(0, 2, 3), keepdims=True), 1e-6)
        row = {
            "fingerprint": _normalizer_fingerprint(mean, scale),
            "fold": fold["index"],
            "mean": mean.reshape(-1).tolist(),
            "scale": scale.reshape(-1).tolist(),
            "test_video_ids": list(test_ids),
            "train_rows": int(np.sum(mask)),
            "train_video_ids": list(train_ids),
            "uses_target": False,
        }
        wrong.append(row)
        correct_values = np.asarray(
            [
                *cast(list[float], correct[cast(int, fold["index"])]["mean"]),
                *cast(list[float], correct[cast(int, fold["index"])]["scale"]),
            ],
            dtype="<f8",
        )
        wrong_values = np.asarray([*row["mean"], *row["scale"]], dtype="<f8")
        max_abs.append(float(np.max(np.abs(wrong_values - correct_values))))
        wrong_bits = wrong_values.view("<u8")
        correct_bits = correct_values.view("<u8")
        distances = np.where(wrong_bits >= correct_bits, wrong_bits - correct_bits, correct_bits - wrong_bits)
        max_ulp.append(int(np.max(distances)))
    fingerprints = tuple(cast(str, row["fingerprint"]) for row in wrong)
    if (
        fingerprints != WRONG_NORMALIZER_FINGERPRINTS
        or fingerprints == tuple(cast(str, row["fingerprint"]) for row in correct)
        or tuple(max_abs) != WRONG_NORMALIZER_MAX_ABS
        or tuple(max_ulp) != WRONG_NORMALIZER_MAX_ULP
    ):
        raise InvalidLCV001Verifier("wrong-layout normalizer endpoints differ from preregistration")
    stored_rows = [{**row, "resolution": resolution} for resolution in (8, 16, 32, 64) for row in wrong]
    return stored_rows, {
        "max_abs_difference": max_abs,
        "max_ulp_difference": max_ulp,
        "strides": list(WRONG_NORMALIZER_STRIDES),
        "wrong_fingerprints": list(fingerprints),
    }


def _consumer_semantics(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    _validate_frame_arrays(arrays)
    video_ids = arrays["video_ids"]
    timestamps = arrays["timestamps"]
    frames = arrays["frames_uint8"]
    previous, current, future = _causal_indices(video_ids, timestamps)
    matched_ids = np.ascontiguousarray(video_ids[current])
    matched_times = np.ascontiguousarray(timestamps[current])
    counts = dict(Counter(str(value) for value in matched_ids.tolist()))
    identity = _canonical_sha256(list(zip(matched_ids.tolist(), matched_times.tolist(), strict=True)))
    if len(current) != 453 or counts != MATCHED_COUNTS or identity != MATCHED_IDENTITY_SHA256:
        raise InvalidLCV001Artifact("MM-007 exact 453 current identities do not replay")
    for left, middle, right in zip(previous, current, future, strict=True):
        if (
            video_ids[left] != video_ids[middle]
            or video_ids[middle] != video_ids[right]
            or timestamps[left] + 0.5 != timestamps[middle]
            or timestamps[middle] + 0.5 != timestamps[right]
        ):
            raise InvalidLCV001Artifact("MM-007 previous/current/future identity is not one causal half-second triple")
    alignment = _require_mapping(evidence.get("alignment"), "MM-007 alignment")
    if set(alignment) != {"counts", "identity_sha256", "resolutions", "rows"}:
        raise InvalidLCV001Artifact("MM-007 alignment schema differs")
    if (
        alignment.get("rows") != 453
        or alignment.get("counts") != MATCHED_COUNTS
        or alignment.get("identity_sha256") != identity
    ):
        raise InvalidLCV001Artifact("MM-007 evidence alignment differs from copied frames")
    input_validation = _require_mapping(manifest.get("input_validation"), "MM-007 input validation")
    low_resolution = _require_mapping(input_validation.get("low_resolution_parity"), "MM-007 low-resolution parity")
    matched_panel = _require_mapping(low_resolution.get("matched_panel"), "MM-007 matched panel")
    if (
        matched_panel.get("rows") != 453
        or matched_panel.get("counts") != MATCHED_COUNTS
        or matched_panel.get("identity_sha256") != identity
    ):
        raise InvalidLCV001Artifact("MM-007 input-manifest matched panel differs from copied frames")
    normalizers, training_indices = _recompute_normalizers(frames, current, matched_ids)
    _validate_normalizer_rows(evidence.get("normalizer_rows"), normalizers)
    all_current_r8 = _pool_selected(frames, current)
    resolutions = _require_mapping(alignment.get("resolutions"), "MM-007 alignment resolutions")
    r8_record = _require_mapping(resolutions.get("8"), "MM-007 R8 alignment record")
    if (
        set(resolutions) != {"8", "16", "32", "64"}
        or set(r8_record) != {"current_sha256", "shape", "target_sha256"}
        or r8_record.get("current_sha256") != R8_CURRENT_SHA256
        or r8_record.get("shape") != [3, 8, 8]
        or _mm007_float_array_sha256(all_current_r8) != R8_CURRENT_SHA256
    ):
        raise InvalidLCV001Artifact("MM-007 full current R8 endpoint differs")
    triplets = [
        [
            str(video_ids[middle]),
            float(timestamps[left]),
            float(timestamps[middle]),
            float(timestamps[right]),
        ]
        for left, middle, right in zip(previous, current, future, strict=True)
    ]
    panel = {
        "counts": counts,
        "current_frames_uint8_sha256": _array_sha256(np.ascontiguousarray(frames[current])),
        "current_identity_sha256": identity,
        "future_frames_uint8_sha256": _array_sha256(np.ascontiguousarray(frames[future])),
        "previous_frames_uint8_sha256": _array_sha256(np.ascontiguousarray(frames[previous])),
        "rows": 453,
        "triplet_identity_sha256": _canonical_sha256(triplets),
    }
    frame_receipt = {
        name: {
            "dtype": arrays[name].dtype.str,
            "sha256": _array_sha256(arrays[name]),
            "shape": list(arrays[name].shape),
        }
        for name in ("frames_uint8", "timestamps", "video_ids")
    }
    closure: dict[str, object] = {
        "frame_archive": {
            "arrays": frame_receipt,
            "package_sha256": FRAME_PACKAGE_SHA256,
            "raw_counts": RAW_COUNTS,
            "rows": 477,
        },
        "normalizers": {
            "exact_replay": True,
            "folds": normalizers,
            "training_current_indices": training_indices,
            "intermediate_dtype": "<f4",
            "intermediate_strides": list(NORMALIZER_STRIDES),
            "full_current_r8_sha256": R8_CURRENT_SHA256,
            "scope_count": 16,
            "select_before_pool": True,
            "shared_resolutions": [8, 16, 32, 64],
        },
        "successor_panel": panel,
    }
    state: dict[str, object] = {
        "arrays": arrays,
        "current": current,
        "evidence": evidence,
        "matched_ids": matched_ids,
        "normalizers": normalizers,
        "r8": all_current_r8,
        "result": result,
    }
    return closure, state


def _validate_parent_closure(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    snapshot = _read_parent_snapshot(root, copied=True)
    artifact = _parse_parent_json(snapshot, "artifact-manifest.json")
    manifest = _parse_parent_json(snapshot, "input-manifest.json")
    marker = _parse_parent_json(snapshot, "formal-start.json")
    evidence = _parse_parent_json(snapshot, "MM-007-evidence.json")
    result = _parse_parent_json(snapshot, "MM-007-results.json")
    crosslinks = _validate_mm007_crosslinks(artifact, manifest, marker, evidence, result)
    arrays = _load_frame_arrays(snapshot[Path("MM-007-frames-64x64.npz")])
    semantics, state = _consumer_semantics(arrays, manifest, evidence, result)
    parent_tree = {str(path): MM007_PINS[path][0] for path in PARENT_FILES}
    closure: dict[str, object] = {
        "classification": "PASS",
        "crosslinks": crosslinks,
        "experiment_id": EXPERIMENT_ID,
        "parent": {
            "artifact_manifest_sha256": MM007_ARTIFACT_SHA256,
            "classification": MM007_CLASSIFICATION,
            "file_count": 14,
            "tree_sha256": _canonical_sha256(parent_tree),
        },
        "schema_version": SCHEMA_VERSION,
        "semantics": semantics,
        "status": "sealed_parent_consumer_semantics_verified",
    }
    state.update({"artifact": artifact, "manifest": manifest, "marker": marker})
    return closure, state


def _validate_runtime_receipt(value: object) -> Mapping[str, Any]:
    receipt = _require_mapping(value, "LCV-001 runtime receipt")
    if set(receipt) != {
        "base_exec_prefix",
        "base_executable",
        "base_prefix",
        "canary",
        "dependency_closures",
        "dependency_hashes",
        "dependency_manifests",
        "environment",
        "exec_prefix",
        "executable",
        "loaded_openblas",
        "numpy",
        "numpy_file",
        "openblas",
        "platform",
        "prefix",
        "python",
        "schema_version",
        "status",
        "supervisor",
        "symlink_chain",
        "thread_environment",
    }:
        raise InvalidLCV001Runtime("LCV-001 runtime receipt schema differs")
    canary = _require_mapping(receipt.get("canary"), "LCV-001 runtime canary")
    openblas = _require_mapping(receipt.get("openblas"), "LCV-001 OpenBLAS receipt")
    supervised = _require_mapping(receipt.get("supervisor"), "LCV-001 supervisor receipt")
    dependency_hashes = _require_mapping(receipt.get("dependency_hashes"), "LCV-001 dependency hashes")
    try:
        dependency_aggregates = runtime_probe.validate_dependency_manifests(receipt.get("dependency_manifests"))
    except runtime_probe.RuntimeClosureError as error:
        raise InvalidLCV001Runtime(str(error)) from error
    expected_canary = {
        "bundle_sha256": runtime_probe.CANARY_BUNDLE_SHA256,
        "input_sha256": runtime_probe.CANARY_INPUT_SHA256,
        "minimum_singular_gap": runtime_probe.CANARY_MIN_GAP,
        "relative_reconstruction_error": runtime_probe.CANARY_RELATIVE_RECONSTRUCTION,
        "s_sha256": runtime_probe.CANARY_S_SHA256,
        "schema_version": "lcv001-svd-canary-v1",
        "shape": [512, 256],
        "u_sha256": runtime_probe.CANARY_U_SHA256,
        "vh_sha256": runtime_probe.CANARY_VH_SHA256,
    }
    expected_dependency_hashes = {
        "numpy_init": runtime_probe.NUMPY_INIT_SHA256,
        "numpy_multiarray": runtime_probe.MULTIARRAY_SHA256,
        "numpy_umath_linalg": runtime_probe.UMATH_LINALG_SHA256,
        "openblas": runtime_probe.OPENBLAS_SHA256,
        "pyvenv_cfg": runtime_probe.PYVENV_CFG_SHA256,
        "python_terminal": runtime_probe.PYTHON_TERMINAL_SHA256,
    }
    expected_chain = [{"path": path, "target": target} for path, target in runtime_probe.SYMLINK_CHAIN]
    if (
        receipt.get("schema_version") != runtime_probe.SCHEMA_VERSION
        or receipt.get("status") != "runtime_closure_verified"
        or receipt.get("executable") != str(runtime_probe.LEXICAL_PYTHON)
        or receipt.get("prefix") != str(runtime_probe.BASE_PREFIX)
        or receipt.get("base_prefix") != str(runtime_probe.BASE_PREFIX)
        or receipt.get("base_executable") != str(runtime_probe.BASE_EXECUTABLE)
        or receipt.get("exec_prefix") != str(runtime_probe.BASE_PREFIX)
        or receipt.get("base_exec_prefix") != str(runtime_probe.BASE_PREFIX)
        or receipt.get("python") != runtime_probe.PYTHON_VERSION
        or receipt.get("numpy") != runtime_probe.NUMPY_VERSION
        or receipt.get("numpy_file") != str(runtime_probe.NUMPY_INIT)
        or receipt.get("loaded_openblas") != [str(runtime_probe.OPENBLAS)]
        or receipt.get("platform") != "Linux-6.14.0-37-generic-x86_64-with-glibc2.39"
        or receipt.get("environment") != runtime_probe.frozen_environment()
        or receipt.get("thread_environment") != dict(sorted(runtime_probe.THREAD_ENVIRONMENT.items()))
        or receipt.get("dependency_closures") != dependency_aggregates
        or dependency_hashes != expected_dependency_hashes
        or canary != expected_canary
        or openblas
        != {
            "config": runtime_probe.OPENBLAS_CONFIG,
            "core": runtime_probe.OPENBLAS_CORE,
            "parallel": 1,
            "threads": 1,
        }
        or receipt.get("symlink_chain") != expected_chain
        or set(supervised) != {"manifest", "role", "unit"}
        or supervised.get("manifest") != supervisor.manifest()
        or supervised.get("role") != "formal"
        or not isinstance(supervised.get("unit"), str)
        or not cast(str, supervised["unit"]).startswith("lcv001-custody-formal-")
        or not cast(str, supervised["unit"]).endswith(".service")
    ):
        raise InvalidLCV001Runtime("LCV-001 runtime receipt differs from the frozen closure")
    return receipt


def _expect_rejection(name: str, function: Any, classification: str) -> dict[str, object]:
    try:
        function()
    except (InvalidLCV001Artifact, InvalidLCV001Runtime) as error:
        if getattr(error, "classification", None) != classification:
            raise InvalidLCV001Verifier(f"negative control {name} produced the wrong classification") from error
        return {"classification": classification, "name": name, "passed": True}
    except Exception as error:
        raise InvalidLCV001Verifier(f"negative control {name} crashed outside the verifier boundary") from error
    raise InvalidLCV001Verifier(f"negative control {name} was accepted")


def _canary_child(output: Path, executable: Path, site_packages: Path, threads: int) -> dict[str, Any]:
    environment = runtime_probe.frozen_environment()
    for name in runtime_probe.THREAD_ENVIRONMENT:
        environment[name] = str(threads)
    probe = output / RUNTIME_SOURCE_ROOT / "bench/sealed_lineage_verifier/canary_probe.py"
    command = (str(executable), "-I", "-S", "-B", str(probe), str(site_packages))
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        raise InvalidLCV001Runtime("runtime sensitivity canary child failed to execute") from error
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        if completed.returncode == CLASSIFIED_CHILD_EXIT:
            raise InvalidLCV001Verifier(
                "runtime sensitivity canary used an unsupported classified-failure exit"
            )
        failure = _classified_child_failure(
            stderr,
            stdout=stdout,
            returncode=completed.returncode,
            default_runtime=False,
        )
        if isinstance(failure, InvalidLCV001Runtime):
            raise failure
        raise InvalidLCV001Verifier("runtime sensitivity canary child transport or logic failed") from failure
    return _validated_child_success(stdout, stderr, "runtime sensitivity canary stdout")


def _runtime_child_controls(output: Path, runtime_receipt: Mapping[str, Any]) -> list[dict[str, object]]:
    venv_site = runtime_probe.VENV_PREFIX / "lib/python3.12/site-packages"
    base_site = runtime_probe.BASE_PREFIX / "lib/python3.12/site-packages"
    direct = _canary_child(output, runtime_probe.LEXICAL_PYTHON, venv_site, 1)
    venv_threads16 = _canary_child(output, runtime_probe.LEXICAL_PYTHON, venv_site, 16)
    base_threads1 = _canary_child(output, runtime_probe.BASE_EXECUTABLE, base_site, 1)
    base_threads16 = _canary_child(output, runtime_probe.BASE_EXECUTABLE, base_site, 16)
    observed = {
        "direct_cgroup_parity": (
            direct["bundle_sha256"],
            cast(Mapping[str, Any], runtime_receipt["canary"])["bundle_sha256"],
            runtime_probe.CANARY_BUNDLE_SHA256,
        ),
        "venv_threads16_sensitivity": (
            venv_threads16["bundle_sha256"],
            "f6b21b168c3bd999df24b9a940f3d01a178c3429feb7978e5d429445c1dc7865",
        ),
        "base_threads1_sensitivity": (
            base_threads1["bundle_sha256"],
            "c0b4b0dde34f85cb8441446f104d1cc4b33fb55c1fe51f7afb709eeb7cf47334",
        ),
        "base_threads16_sensitivity": (
            base_threads16["bundle_sha256"],
            "ae753e31d8a34268ae356c3216e3647036fff832786c31f59006e9575e129b3f",
        ),
    }
    if any(len(set(values)) != 1 for values in observed.values()):
        raise InvalidLCV001Verifier("runtime child canary/control bundle differs from its preregistered endpoint")
    return [
        {
            "bundle_sha256": values[0],
            "classification": "runtime_control_endpoint_verified",
            "name": name,
            "passed": True,
        }
        for name, values in observed.items()
    ]


def _mutation_controls(
    output: Path, state: Mapping[str, object], runtime_receipt: Mapping[str, Any]
) -> dict[str, object]:
    artifact = copy.deepcopy(cast(dict[str, Any], state["artifact"]))
    manifest = copy.deepcopy(cast(dict[str, Any], state["manifest"]))
    marker = copy.deepcopy(cast(dict[str, Any], state["marker"]))
    evidence = copy.deepcopy(cast(dict[str, Any], state["evidence"]))
    result = copy.deepcopy(cast(dict[str, Any], state["result"]))
    arrays = cast(Mapping[str, np.ndarray], state["arrays"])
    normalizers = cast(Sequence[Mapping[str, object]], state["normalizers"])
    r8 = cast(np.ndarray, state["r8"])
    current = cast(np.ndarray, state["current"])
    matched_ids = cast(np.ndarray, state["matched_ids"])
    controls: list[dict[str, object]] = []

    bad_artifact = copy.deepcopy(artifact)
    bad_artifact["artifacts"]["MM-007-evidence.json"]["sha256"] = "0" * 64
    controls.append(
        _expect_rejection(
            "artifact_manifest_digest_mutation",
            lambda: _validate_mm007_artifact_manifest(bad_artifact),
            "invalid_LCV001_artifact",
        )
    )

    fold = FOLDS[0]
    expected_indices = _expected_train_current_indices(fold, current, matched_ids)
    heldout = expected_indices.copy()
    heldout_test_ids = cast(tuple[str, ...], fold["test"])
    heldout[0] = int(current[np.flatnonzero(np.isin(matched_ids, heldout_test_ids))[0]])
    future = np.ascontiguousarray(expected_indices + 1, dtype="<i8")
    previous = np.ascontiguousarray(expected_indices - 1, dtype="<i8")
    reordered = np.ascontiguousarray(expected_indices[::-1], dtype="<i8")
    for name, candidate in (
        ("heldout_current_index_injection", heldout),
        ("future_index_injection", future),
        ("off_by_one_previous_index_injection", previous),
        ("training_current_index_reordering", reordered),
    ):
        controls.append(
            _expect_rejection(
                name,
                lambda value=candidate: _validate_train_current_indices(value, fold, current, matched_ids),
                "invalid_LCV001_artifact",
            )
        )

    wrong_rows, wrong_endpoint = _wrong_select_after_pool_rows(
        cast(np.ndarray, arrays["frames_uint8"]), current, matched_ids, normalizers
    )
    wrong_control = _expect_rejection(
        "select_after_pool_noncontiguous_normalizer_implementation",
        lambda: _validate_normalizer_rows(wrong_rows, normalizers),
        "invalid_LCV001_artifact",
    )
    controls.append({**wrong_control, "endpoint": wrong_endpoint})

    bad_manifest_schema = copy.deepcopy(manifest)
    bad_manifest_schema["unexpected"] = True
    controls.append(
        _expect_rejection(
            "input_manifest_extra_key_mutation",
            lambda: _validate_mm007_crosslinks(artifact, bad_manifest_schema, marker, evidence, result),
            "invalid_LCV001_artifact",
        )
    )

    for name, field in (
        ("marker_frames_uint8_anchor_mutation", "frames_uint8_sha256"),
        ("marker_frame_schema_anchor_mutation", "frame_schema_sha256"),
        ("marker_parent_alignment_anchor_mutation", "parent_alignment_sha256"),
    ):
        bad_marker = copy.deepcopy(marker)
        bad_marker[field] = "0" * 64
        controls.append(
            _expect_rejection(
                name,
                lambda value=bad_marker: _validate_mm007_crosslinks(artifact, manifest, value, evidence, result),
                "invalid_LCV001_artifact",
            )
        )

    for name, field in (
        ("result_frame_file_anchor_mutation", "frame_file_sha256"),
        ("result_frames_uint8_anchor_mutation", "frames_uint8_sha256"),
        ("result_parent_alignment_anchor_mutation", "parent_alignment_sha256"),
    ):
        bad_anchor_result = copy.deepcopy(result)
        bad_anchor_result[field] = "0" * 64
        controls.append(
            _expect_rejection(
                name,
                lambda value=bad_anchor_result: _validate_mm007_crosslinks(artifact, manifest, marker, evidence, value),
                "invalid_LCV001_artifact",
            )
        )

    bad_evidence_schema = copy.deepcopy(evidence)
    bad_evidence_schema["unexpected"] = True
    controls.append(
        _expect_rejection(
            "evidence_extra_key_mutation",
            lambda: _validate_mm007_crosslinks(artifact, manifest, marker, bad_evidence_schema, result),
            "invalid_LCV001_artifact",
        )
    )

    bad_uses_target = copy.deepcopy(evidence)
    bad_uses_target["normalizer_rows"][0]["uses_target"] = 0
    controls.append(
        _expect_rejection(
            "normalizer_boolean_type_mutation",
            lambda: _validate_normalizer_rows(bad_uses_target["normalizer_rows"], normalizers),
            "invalid_LCV001_artifact",
        )
    )

    bad_r8_endpoint = copy.deepcopy(evidence)
    bad_r8_endpoint["alignment"]["resolutions"]["8"]["current_sha256"] = "0" * 64
    controls.append(
        _expect_rejection(
            "full_current_r8_endpoint_mutation",
            lambda: _consumer_semantics(arrays, manifest, bad_r8_endpoint, result),
            "invalid_LCV001_artifact",
        )
    )

    bad_result = copy.deepcopy(result)
    bad_result["evidence_sha256"] = "0" * 64
    controls.append(
        _expect_rejection(
            "result_evidence_crosslink_mutation",
            lambda: _validate_mm007_crosslinks(artifact, manifest, marker, evidence, bad_result),
            "invalid_LCV001_artifact",
        )
    )

    bad_arrays = {name: value.copy() for name, value in arrays.items()}
    bad_arrays["frames_uint8"][0, 0, 0, 0] ^= np.uint8(1)
    controls.append(
        _expect_rejection("frame_byte_mutation", lambda: _validate_frame_arrays(bad_arrays), "invalid_LCV001_artifact")
    )

    bad_evidence = copy.deepcopy(evidence)
    bad_evidence["normalizer_rows"][0]["fingerprint"] = "0" * 64
    controls.append(
        _expect_rejection(
            "stored_normalizer_fingerprint_mutation",
            lambda: _validate_normalizer_rows(bad_evidence["normalizer_rows"], normalizers),
            "invalid_LCV001_artifact",
        )
    )

    noncontiguous = np.asfortranarray(r8)
    controls.append(
        _expect_rejection(
            "reordered_noncontiguous_r8_implementation",
            lambda: _validate_r8_layout(noncontiguous, 453),
            "invalid_LCV001_artifact",
        )
    )

    bad_runtime = copy.deepcopy(dict(runtime_receipt))
    bad_runtime["openblas"]["threads"] = 2
    controls.append(
        _expect_rejection(
            "runtime_thread_receipt_mutation",
            lambda: _validate_runtime_receipt(bad_runtime),
            "invalid_LCV001_runtime",
        )
    )
    controls.extend(_runtime_child_controls(output, runtime_receipt))
    if len(controls) != EXPECTED_CONTROL_COUNT:
        raise InvalidLCV001Verifier("LCV-001 mutation control membership differs")
    return {
        "control_count": len(controls),
        "controls": controls,
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "all_mutation_controls_rejected",
    }


def _formal_marker(
    input_manifest: Mapping[str, object],
    runtime_receipt: Mapping[str, object],
    prepared_records: Mapping[Path, custody.ExpectedFile],
) -> dict[str, object]:
    serialized_records = {str(path): _record_dict(prepared_records[path]) for path in PREPARED_FILES}
    return {
        "audit_sha256": prepared_records[AUDIT_COPY].sha256,
        "config_sha256": prepared_records[CONFIG_COPY].sha256,
        "experiment_id": EXPERIMENT_ID,
        "freeze_record_sha256": prepared_records[FREEZE_RECORD].sha256,
        "input_manifest_sha256": prepared_records[INPUT_MANIFEST].sha256,
        "parent_artifact_manifest_sha256": MM007_ARTIFACT_SHA256,
        "prepared_membership_sha256": _canonical_sha256(serialized_records),
        "protocol_sha256": prepared_records[PROTOCOL_COPY].sha256,
        "runtime_receipt_sha256": _canonical_sha256(runtime_receipt),
        "schema_version": SCHEMA_VERSION,
        "source_sha256": input_manifest["source_sha256"],
        "status": "formal_verification_started",
    }


def _report(result: Mapping[str, Any], closure: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    panel = cast(Mapping[str, Any], cast(Mapping[str, Any], closure["semantics"])["successor_panel"])
    return "\n".join(
        (
            "# LCV-001 sealed-lineage/runtime-closure report",
            "",
            "- Classification: `PASS`",
            "- Role: non-scientific infrastructure gate; no MM-001--MM-009 outcome changed.",
            f"- Parent: exact 14-file MM-007 tree `{MM007_ARTIFACT_SHA256}`.",
            f"- Runtime: lexical `{runtime['executable']}`, NumPy `{runtime['numpy']}`, OpenBLAS threads `1`.",
            "- Runtime scope: pinned Python/stdlib/NumPy aggregate commitments and canary on a trusted live host OS.",
            f"- SVD canary: `{cast(Mapping[str, Any], runtime['canary'])['bundle_sha256']}`.",
            f"- Consumer panel: `{panel['rows']}` exact previous/current/future rows.",
            "- Four source-current-only R8 fold normalizers replay bit-for-bit and are shared across resolutions.",
            "- Historical PCA/model/flow/scoring verifiers were not executed.",
            "- Authorization: prepare a newly named causal assay only after it revalidates the host commitments "
            "and copies this sealed receipt.",
            "",
        )
    )


def _artifact_manifest_payload(records: Mapping[Path, custody.ExpectedFile]) -> dict[str, object]:
    return {
        "artifacts": {str(path): _record_dict(records[path]) for path in ARTIFACT_FILES},
        "classification": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "parent_artifact_manifest_sha256": MM007_ARTIFACT_SHA256,
        "schema_version": SCHEMA_VERSION,
    }


def _current_records(output: Path, files: Sequence[Path]) -> dict[Path, custody.ExpectedFile]:
    records: dict[Path, custody.ExpectedFile] = {}
    for relative in files:
        record = _file_record(output / relative)
        records[relative] = custody.ExpectedFile(
            cast(str, record["sha256"]),
            cast(int, record["bytes"]),
            cast(int, record["mode"]),
        )
    return records


def _unsealed_phase_directory_modes(files: Sequence[Path]) -> dict[Path, int]:
    records = {path: ("0" * 64, 0, 0o444) for path in files}
    return {
        Path("."): 0o555,
        **{
            path: 0o555 if path == PREPARED_ROOT or PREPARED_ROOT in path.parents else 0o755
            for path in custody.expected_directories(records)
        },
    }


def _snapshot_current_phase(
    output: Path,
    files: Sequence[Path],
    *,
    sealed: bool,
) -> custody.TreeSnapshot:
    records = _current_records(output, files)
    try:
        return custody.snapshot_exact_tree(
            output,
            records,
            directory_mode=0o555 if sealed else None,
            directory_modes=None if sealed else _unsealed_phase_directory_modes(files),
        )
    except custody.CustodyError as error:
        raise InvalidLCV001Artifact(str(error)) from error


def _validate_provisional_snapshot(
    snapshot: custody.TreeSnapshot,
    child_value: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _snapshot_json(snapshot, INPUT_MANIFEST)
    prepared_expectations, freeze = _prepared_expectations(manifest, snapshot.payloads[INPUT_MANIFEST])
    for path, record in prepared_expectations.items():
        if snapshot.records.get(path) != record:
            raise InvalidLCV001Artifact(f"prepared record changed before provisional completion: {path}")
    _validate_input_manifest(manifest, snapshot, freeze)
    marker = _snapshot_json(snapshot, FORMAL_START)
    runtime = _snapshot_json(snapshot, RUNTIME_RECEIPT)
    closure = _snapshot_json(snapshot, PARENT_CLOSURE)
    controls = _snapshot_json(snapshot, MUTATION_CONTROLS)
    provisional = _snapshot_json(snapshot, PROVISIONAL_RESULT)
    _validate_runtime_receipt(runtime)
    if marker != _formal_marker(manifest, runtime, prepared_expectations):
        raise InvalidLCV001Artifact("LCV-001 formal marker cross-links differ")
    if set(provisional) != {
        "classification",
        "experiment_id",
        "formal_start",
        "mutation_controls_sha256",
        "parent_closure_sha256",
        "runtime_receipt_sha256",
        "schema_version",
        "statement",
        "status",
    } or provisional != {
        "classification": "PENDING_CGROUP_CLEANUP",
        "experiment_id": EXPERIMENT_ID,
        "formal_start": marker,
        "mutation_controls_sha256": _canonical_sha256(controls),
        "parent_closure_sha256": _canonical_sha256(closure),
        "runtime_receipt_sha256": _canonical_sha256(runtime),
        "schema_version": SCHEMA_VERSION,
        "statement": "provisional host-bound lineage/runtime verification; external cgroup cleanup pending",
        "status": "provisional_completed_inside_cgroup",
    }:
        raise InvalidLCV001Artifact("LCV-001 provisional result cross-links differ")
    if child_value is not None and child_value != provisional:
        raise InvalidLCV001Verifier("formal child stdout differs from its provisional sealed payload")
    if (
        closure.get("classification") != "PASS"
        or closure.get("parent", {}).get("artifact_manifest_sha256") != MM007_ARTIFACT_SHA256
        or controls.get("status") != "all_mutation_controls_rejected"
        or controls.get("control_count") != EXPECTED_CONTROL_COUNT
        or not all(item.get("passed") is True for item in controls.get("controls", []))
    ):
        raise InvalidLCV001Artifact("LCV-001 provisional closure/control receipt differs")
    return manifest, marker, runtime, closure, controls


def _completion_workspace_pattern(output: Path) -> str:
    return f".{output.name}.completion-*"


def _directory_object_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise InvalidLCV001Runtime(f"LCV-001 completion namespace identity is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise InvalidLCV001Runtime(f"LCV-001 completion namespace member is not a real directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _finalization_checkpoint(name: str) -> None:
    """Fault-injection boundary; production execution is intentionally a no-op."""

    del name


def _copy_provisional_candidate(candidate: Path, snapshot: custody.TreeSnapshot) -> None:
    try:
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise InvalidLCV001Runtime("LCV-001 private completion candidate is not a real directory")
        if tuple(candidate.iterdir()):
            raise InvalidLCV001Runtime("LCV-001 private completion candidate was not created empty")
        for item in snapshot.files:
            _write_exclusive(candidate / item.path, item.payload)
        modes = _unsealed_phase_directory_modes(PROVISIONAL_FILES)
        for relative, mode in sorted(
            modes.items(),
            key=lambda item: (len(item[0].parts), item[0].as_posix()),
            reverse=True,
        ):
            directory = candidate if relative == Path(".") else candidate / relative
            os.chmod(directory, mode, follow_symlinks=False)
            _fsync_directory(directory)
    except OSError as error:
        raise InvalidLCV001Runtime(f"LCV-001 private completion candidate creation failed: {error}") from error


def _remove_completion_workspace(workspace: Path, ownership: list[bool]) -> None:
    if not workspace.exists() and not workspace.is_symlink():
        ownership[0] = False
        return
    try:
        custody.remove_created_tree(workspace)
        ownership[0] = False
    except custody.CustodyError as error:
        raise InvalidLCV001Runtime(f"LCV-001 private completion workspace cleanup failed: {error}") from error


def _candidate_check(label: str, action: Any) -> Any:
    """Reclassify a defect in parent-generated candidate bytes as verifier-owned."""

    try:
        return action()
    except InvalidLCV001Artifact as error:
        raise InvalidLCV001Verifier(f"LCV-001 private completion candidate {label} failed: {error}") from error


def _postcommit_action(
    name: str,
    action: Any,
    warnings: list[dict[str, str]],
) -> bool:
    """Run cleanup after the atomic PASS commit without reclassifying it."""

    try:
        _finalization_checkpoint(name)
        action()
    except BaseException as error:
        warnings.append({"error_type": type(error).__name__, "phase": name})
        return False
    return True


def _finalize_after_cleanup(
    output: Path,
    child_value: Mapping[str, object],
    cleanup_value: object,
) -> dict[str, object]:
    provisional_snapshot = _snapshot_current_phase(output, PROVISIONAL_FILES, sealed=False)
    manifest, marker, runtime, closure, controls = _validate_provisional_snapshot(provisional_snapshot, child_value)
    _validate_live_source(manifest)
    try:
        cleanup = supervisor.validate_cleanup_receipt(cleanup_value, role="formal")
    except supervisor.SupervisorError as error:
        raise InvalidLCV001Runtime(str(error)) from error
    runtime_unit = cast(Mapping[str, Any], runtime["supervisor"]).get("unit")
    if cleanup.get("unit") != runtime_unit:
        raise InvalidLCV001Runtime("LCV-001 cleanup receipt names a different formal cgroup")
    _finalization_checkpoint("after_cleanup_validation")
    workspace = _private_workspace_path(output.parent, f".{output.name}.completion-")
    workspace_owned = [False]
    candidate = workspace
    committed = False
    verified: dict[str, object] | None = None
    exchange_warnings: list[dict[str, str]] = []
    candidate_identity: tuple[int, int] | None = None
    provisional_identity: tuple[int, int] | None = None
    try:
        _create_owned_workspace(
            workspace,
            mode=0o700,
            ownership=workspace_owned,
            checkpoint="completion_after_mkdir",
        )
        _copy_provisional_candidate(candidate, provisional_snapshot)
        _finalization_checkpoint("after_candidate_copy")
        copied_provisional = _candidate_check(
            "provisional snapshot",
            lambda: _snapshot_current_phase(candidate, PROVISIONAL_FILES, sealed=False),
        )
        _candidate_check(
            "provisional cross-links",
            lambda: _validate_provisional_snapshot(copied_provisional, child_value),
        )
        _write_exclusive(candidate / CLEANUP_RECEIPT, _json_bytes(cleanup))
        _finalization_checkpoint("after_cleanup_receipt_write")
        result: dict[str, object] = {
            "classification": "PASS",
            "cleanup_receipt_sha256": _canonical_sha256(cleanup),
            "experiment_id": EXPERIMENT_ID,
            "formal_start": marker,
            "mutation_controls_sha256": _canonical_sha256(controls),
            "parent_closure_sha256": _canonical_sha256(closure),
            "runtime_receipt_sha256": _canonical_sha256(runtime),
            "schema_version": SCHEMA_VERSION,
            "status": "completed_after_cgroup_cleanup",
            "statement": (
                "host-bound sealed lineage/runtime verified after formal cgroup cleanup; no scientific outcome"
            ),
        }
        _write_exclusive(candidate / RESULTS_FILE, _json_bytes(result))
        _finalization_checkpoint("after_result_write")
        _write_exclusive(candidate / REPORT_FILE, _report(result, closure, runtime).encode("utf-8"))
        _finalization_checkpoint("after_report_write")
        artifact_snapshot = _candidate_check(
            "artifact snapshot",
            lambda: _snapshot_current_phase(candidate, ARTIFACT_FILES, sealed=False),
        )
        _write_exclusive(
            candidate / ARTIFACT_MANIFEST,
            _json_bytes(_artifact_manifest_payload(artifact_snapshot.records)),
        )
        _finalization_checkpoint("after_artifact_manifest_write")
        completed_records = _candidate_check(
            "completed records",
            lambda: _current_records(candidate, COMPLETED_FILES),
        )
        try:
            custody.snapshot_exact_tree(
                candidate,
                completed_records,
                directory_modes=_unsealed_phase_directory_modes(COMPLETED_FILES),
            )
            _finalization_checkpoint("before_candidate_seal")
            custody.seal_copied_tree(candidate, completed_records)
        except custody.CustodyError as error:
            raise InvalidLCV001Verifier(f"LCV-001 completion candidate sealing failed: {error}") from error
        _finalization_checkpoint("after_candidate_seal")
        verified = _candidate_check("sealed verification", lambda: _verify_in_process(candidate))
        _finalization_checkpoint("after_candidate_verify")

        # The live canonical side of the exchange must still be the exact
        # provisional package whose authenticated bytes produced the candidate.
        current = _snapshot_current_phase(output, PROVISIONAL_FILES, sealed=False)
        current_manifest, *_rest = _validate_provisional_snapshot(current, child_value)
        if current.records != provisional_snapshot.records:
            raise InvalidLCV001Artifact("LCV-001 canonical provisional records changed before commit")
        _validate_live_source(current_manifest)
        _finalization_checkpoint("before_atomic_exchange")
        candidate_identity = _directory_object_identity(candidate)
        provisional_identity = _directory_object_identity(output)
        _rename_exchange(candidate, output)
        committed = True
    except BaseException as error:
        if committed:
            exchange_warnings.append({"error_type": type(error).__name__, "phase": "exchange_return_boundary"})
        elif candidate_identity is not None and provisional_identity is not None:
            output_after = _directory_object_identity(output)
            candidate_after = _directory_object_identity(candidate)
            if output_after == candidate_identity and candidate_after == provisional_identity:
                committed = True
                exchange_warnings.append(
                    {"error_type": type(error).__name__, "phase": "exchange_return_boundary"}
                )
            elif output_after == provisional_identity and candidate_after == candidate_identity:
                if workspace_owned[0]:
                    _remove_completion_workspace(workspace, workspace_owned)
                raise
            else:
                raise InvalidLCV001Runtime(
                    "LCV-001 atomic exchange left an indeterminate namespace state"
                ) from error
        else:
            if workspace_owned[0]:
                _remove_completion_workspace(workspace, workspace_owned)
            raise

    assert verified is not None
    warnings = exchange_warnings
    parent_fsync = _postcommit_action("after_atomic_exchange", lambda: _fsync_directory(output.parent), warnings)
    retired_removed = _postcommit_action(
        "retired_provisional_cleanup",
        lambda: _remove_completion_workspace(workspace, workspace_owned),
        warnings,
    )
    _finalization_checkpoint("before_commit_receipt_return")
    return {
        **verified,
        "commit": {
            "atomic_exchange": True,
            "parent_directory_fsync": parent_fsync,
            "retired_provisional_removed": retired_removed,
            "warnings": warnings,
        },
    }


def _terminal_failure_payload(error: BaseException, cleanup: object | None) -> dict[str, object]:
    message = _safe_exception_text(error, 1024).encode("utf-8", errors="replace")
    if not isinstance(error, (InvalidLCV001Artifact, InvalidLCV001Runtime, InvalidLCV001Verifier)):
        raise InvalidLCV001Verifier("terminal failure was not classified before custody")
    classification = error.classification
    cleanup_summary: dict[str, object] | None = None
    if isinstance(cleanup, dict):
        try:
            validated_cleanup = supervisor.validate_cleanup_receipt(cleanup, role="formal")
            cleanup_summary = {
                "receipt_sha256": _canonical_sha256(validated_cleanup),
                "schema_version": validated_cleanup["schema_version"],
                "status": validated_cleanup["status"],
                "unit": validated_cleanup["unit"],
            }
        except BaseException:
            cleanup_summary = {
                "receipt_sha256": None,
                "schema_version": None,
                "status": "invalid_untrusted_cleanup_receipt",
                "unit": None,
            }
    return {
        "classification": classification,
        "cleanup": cleanup_summary,
        "error_sha256": hashlib.sha256(message).hexdigest(),
        "error_type": type(error).__name__,
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "terminal_incomplete_after_formal_start",
    }


def _child_failure_diagnostic(stderr: str) -> str | None:
    if not stderr.endswith("\n") or stderr.count("\n") != 1:
        return None
    try:
        payload = _json_from_bytes(stderr.encode("utf-8"), "lexical child failure diagnostic")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"classification", "error"}
            or payload.get("classification")
            not in {
                InvalidLCV001Artifact.classification,
                InvalidLCV001Runtime.classification,
                InvalidLCV001Verifier.classification,
            }
            or not isinstance(payload.get("error"), str)
            or stderr != _cli_json_line(payload, compact=False)
        ):
            return None
    except Exception:
        # Captured child bytes are untrusted transport. No bounded parser failure
        # may escape the parent classification/terminal-custody boundary.
        return None
    return cast(str, payload["classification"])


def _child_classification_claim(line: str) -> str | None:
    if not line.endswith("\n") or line.count("\n") != 1:
        return None
    try:
        payload = _json_from_bytes(line.encode("utf-8"), "lexical child classification claim")
        if (
            not isinstance(payload, dict)
            or "classification" not in payload
            or line not in {_cli_json_line(payload, compact=False), _cli_json_line(payload, compact=True)}
        ):
            return None
    except Exception:
        return None
    classification = payload["classification"]
    return classification if isinstance(classification, str) else "<invalid-non-string-classification>"


def _child_classification_claims(stderr: str) -> tuple[str, ...]:
    return tuple(
        claim
        for line in stderr.splitlines(keepends=True)
        if (claim := _child_classification_claim(line)) is not None
    )


def _classified_child_failure(
    stderr: str,
    *,
    stdout: str,
    returncode: int,
    default_runtime: bool,
) -> BaseException:
    diagnostic = _child_failure_diagnostic(stderr)
    claims = _child_classification_claims(stderr)
    if returncode == CLASSIFIED_CHILD_EXIT:
        classification = (
            diagnostic
            if stdout == "" and diagnostic is not None
            else InvalidLCV001Verifier.classification
        )
    elif returncode in RUNTIME_TRANSPORT_RETURN_CODES:
        classification = (
            InvalidLCV001Runtime.classification
            if all(value == InvalidLCV001Runtime.classification for value in claims)
            else InvalidLCV001Verifier.classification
        )
    else:
        classification = (
            InvalidLCV001Runtime.classification
            if default_runtime and not claims and stdout == ""
            else InvalidLCV001Verifier.classification
        )
    message = (
        f"lexical LCV-001 child failed (returncode={returncode}, {classification}): "
        f"stdout={stdout[-500:]!r}; stderr={stderr[-500:]!r}"
    )
    if classification == InvalidLCV001Runtime.classification:
        return InvalidLCV001Runtime(message)
    if classification == InvalidLCV001Verifier.classification:
        return InvalidLCV001Verifier(message)
    return InvalidLCV001Artifact(message)


def _validated_child_success(stdout: str, stderr: str, label: str) -> dict[str, Any]:
    if stderr != "":
        raise InvalidLCV001Verifier(f"{label} emitted stderr with a successful return code")
    if not stdout.endswith("\n") or stdout.count("\n") != 1:
        raise InvalidLCV001Verifier(f"{label} did not emit exactly one JSON line")
    try:
        value = _json_from_bytes(stdout.encode("utf-8"), label)
        if not isinstance(value, dict) or stdout != _cli_json_line(value, compact=True):
            raise InvalidLCV001Verifier(f"{label} did not emit one canonical JSON object")
    except InvalidLCV001Verifier:
        raise
    except Exception as error:
        raise InvalidLCV001Verifier(f"{label} stdout could not be parsed safely") from error
    return cast(dict[str, Any], value)


def _classify_post_marker_failure(error: BaseException, context: str) -> BaseException:
    if isinstance(error, (InvalidLCV001Artifact, InvalidLCV001Runtime, InvalidLCV001Verifier)):
        return error
    message = f"{context}: {type(error).__name__}: {_safe_exception_text(error)}"
    if isinstance(
        error,
        (OSError, subprocess.SubprocessError, supervisor.SupervisorError, KeyboardInterrupt),
    ):
        return InvalidLCV001Runtime(message)
    return InvalidLCV001Verifier(message)


def _classify_supervisor_failure(error: BaseException) -> BaseException:
    if isinstance(error, (InvalidLCV001Artifact, InvalidLCV001Runtime, InvalidLCV001Verifier)):
        return error
    return InvalidLCV001Runtime(
        f"LCV-001 cgroup supervisor failed: {type(error).__name__}: {_safe_exception_text(error)}"
    )


def _seal_terminal_failure(output: Path, error: BaseException, cleanup: object | None) -> None:
    if not (output / TERMINAL_FAILURE).exists():
        _write_exclusive(output / TERMINAL_FAILURE, _json_bytes(_terminal_failure_payload(error, cleanup)))
    files, _directories = _tree_census(output)
    allowed = {*PROVISIONAL_FILES, TERMINAL_FAILURE}
    if not set(files) <= allowed or FORMAL_START not in files or TERMINAL_FAILURE not in files:
        raise InvalidLCV001Artifact("terminal LCV-001 tree contains unexpected or missing custody files")
    records = _current_records(output, files)
    try:
        custody.snapshot_exact_tree(output, records)
        custody.seal_copied_tree(output, records)
    except custody.CustodyError as custody_error:
        raise InvalidLCV001Artifact(f"terminal LCV-001 tree could not be sealed: {custody_error}") from custody_error
    _fsync_directory(output.parent)


def _reconcile_formal_canonical_phase(output: Path) -> str:
    """Identify whether terminal writes remain safe after a parent-side failure."""

    try:
        files, directories = _tree_census(output)
    except BaseException:
        return "indeterminate"
    file_set = set(files)
    if TERMINAL_FAILURE in file_set:
        return "terminal"
    if file_set == set(COMPLETED_FILES):
        try:
            _verify_in_process(output)
        except BaseException:
            return "indeterminate"
        return "completed"
    if FORMAL_START in file_set and file_set <= set(PROVISIONAL_FILES):
        try:
            if directories != _expected_directories(files):
                return "indeterminate"
            records = _current_records(output, files)
            custody.snapshot_exact_tree(
                output,
                records,
                directory_modes=_unsealed_phase_directory_modes(files),
            )
        except BaseException:
            return "indeterminate"
        return "provisional"
    return "indeterminate"


def _formal_run_in_process(output: Path) -> dict[str, object]:
    if os.environ.get("LCV001_FORMAL_CHILD") != "1":
        raise InvalidLCV001Runtime("formal implementation may run only in the lexical child")
    try:
        unit = supervisor.assert_current_cgroup("formal")
    except supervisor.SupervisorError as error:
        raise InvalidLCV001Runtime(str(error)) from error
    os.environ.pop("LCV001_CGROUP_UNIT", None)
    os.environ.pop("LCV001_CGROUP_ROLE", None)
    input_manifest, prepared_snapshot = _validated_prepared_snapshot(output)
    _validate_live_source(input_manifest)
    try:
        runtime_receipt = runtime_probe.observe_and_validate()
    except runtime_probe.RuntimeClosureError as error:
        raise InvalidLCV001Runtime(str(error)) from error
    runtime_receipt["supervisor"] = {"manifest": supervisor.manifest(), "role": "formal", "unit": unit}
    # Reacquire the exact prepared tree immediately before the irreversible marker.
    input_manifest, prepared_snapshot = _validated_prepared_snapshot(output)
    _validate_live_source(input_manifest)
    marker = _formal_marker(input_manifest, runtime_receipt, prepared_snapshot.records)
    _write_exclusive(output / FORMAL_START, _json_bytes(marker))
    _write_exclusive(output / RUNTIME_RECEIPT, _json_bytes(runtime_receipt))
    closure, state = _validate_parent_closure(output / PARENT_COPY_ROOT)
    controls = _mutation_controls(output, state, runtime_receipt)
    _write_exclusive(output / PARENT_CLOSURE, _json_bytes(closure))
    _write_exclusive(output / MUTATION_CONTROLS, _json_bytes(controls))
    provisional: dict[str, object] = {
        "classification": "PENDING_CGROUP_CLEANUP",
        "experiment_id": EXPERIMENT_ID,
        "formal_start": marker,
        "mutation_controls_sha256": _canonical_sha256(controls),
        "parent_closure_sha256": _canonical_sha256(closure),
        "runtime_receipt_sha256": _canonical_sha256(runtime_receipt),
        "schema_version": SCHEMA_VERSION,
        "status": "provisional_completed_inside_cgroup",
        "statement": "provisional host-bound lineage/runtime verification; external cgroup cleanup pending",
    }
    _write_exclusive(output / PROVISIONAL_RESULT, _json_bytes(provisional))
    _snapshot_current_phase(output, PROVISIONAL_FILES, sealed=False)
    return provisional


def _post_supervisor_return_checkpoint(action: str, output: Path) -> None:
    """Fault-injection boundary still covered by parent terminal custody."""

    del action, output


def _validated_supervised_result(
    value: object,
    command: tuple[str, ...],
    role: str,
) -> supervisor.SupervisedCompletedProcess:
    if type(value) is not supervisor.SupervisedCompletedProcess:
        raise InvalidLCV001Verifier("LCV-001 supervisor returned a malformed child contract")
    completed = cast(supervisor.SupervisedCompletedProcess, value)
    if (
        completed.args != command
        or type(completed.returncode) is not int
        or type(completed.stdout) is not str
        or type(completed.stderr) is not str
    ):
        raise InvalidLCV001Verifier("LCV-001 supervisor returned a malformed child contract")
    try:
        supervisor.validate_cleanup_receipt(completed.cleanup_receipt, role=role)
    except BaseException as error:
        raise InvalidLCV001Runtime(
            "LCV-001 supervisor returned an invalid cleanup contract: "
            f"{type(error).__name__}: {_safe_exception_text(error)}"
        ) from error
    return completed


def _safe_cleanup_source(completed: object | None, error: BaseException) -> object | None:
    source = completed if completed is not None else error
    try:
        return getattr(source, "cleanup_receipt" if completed is not None else "lcv001_cleanup_receipt", None)
    except BaseException:
        return None


def _launch_lexical(action: str, output: Path) -> dict[str, object]:
    destination = output if output.is_absolute() else REPO_ROOT / output
    environment = runtime_probe.frozen_environment()
    if action == "_formal-run":
        environment["LCV001_FORMAL_CHILD"] = "1"
    bootstrap = destination / RUNTIME_SOURCE_ROOT / "bench/sealed_lineage_verifier/bootstrap.py"
    command = (
        str(runtime_probe.LEXICAL_PYTHON),
        "-I",
        "-S",
        "-B",
        str(bootstrap),
        action,
        "--output",
        str(destination),
    )
    role = "formal" if action == "_formal-run" else "semantic"
    completed: supervisor.SupervisedCompletedProcess | None = None
    try:
        completed = supervisor.run(
            command,
            cwd=REPO_ROOT,
            environment=environment,
            timeout_seconds=180.0,
            role=role,
        )
        _post_supervisor_return_checkpoint(action, destination)
        completed = _validated_supervised_result(completed, command, role)
        if completed.returncode != 0:
            raise _classified_child_failure(
                completed.stderr,
                stdout=completed.stdout,
                returncode=completed.returncode,
                default_runtime=action == "_formal-run" and not (destination / FORMAL_START).exists(),
            )
        value = _validated_child_success(completed.stdout, completed.stderr, f"{action} stdout")
        if action == "_formal-run":
            return _finalize_after_cleanup(destination, cast(dict[str, object], value), completed.cleanup_receipt)
        return cast(dict[str, object], value)
    except BaseException as error:
        cleanup = _safe_cleanup_source(completed, error)
        classified = (
            _classify_post_marker_failure(error, "LCV-001 child transport or parent finalization failed")
            if completed is not None
            else _classify_supervisor_failure(error)
        )
        if action == "_formal-run" and (destination / FORMAL_START).exists():
            phase = _reconcile_formal_canonical_phase(destination)
            if phase in {"completed", "indeterminate"}:
                raise InvalidLCV001Runtime(
                    "LCV-001 canonical phase is "
                    f"{phase} after a supervised parent failure; the exact commit receipt is unavailable, "
                    "canonical custody was not modified, manual inspection is required, and PASS is not authorized"
                ) from error
            if phase == "provisional":
                _seal_terminal_failure(destination, classified, cleanup)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise classified from error


def _run_at(output: Path) -> dict[str, object]:
    """Launch formal work through the lexical venv path without resolving it."""

    destination = output if output.is_absolute() else REPO_ROOT / output
    manifest = _validate_prepared(destination)
    _validate_live_source(manifest)
    try:
        _fsync_directory(destination.parent)
    except OSError as error:
        raise InvalidLCV001Runtime("LCV-001 prepared namespace durability could not be confirmed") from error
    return _launch_lexical("_formal-run", destination)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    destination = output if output.is_absolute() else REPO_ROOT / output
    if destination != EXPECTED_OUTPUT:
        raise InvalidLCV001Artifact("public run is restricted to canonical LCV-001 output")
    return _run_at(destination)


def _read_output_json(output: Path, relative: Path) -> dict[str, Any]:
    value = _json_from_bytes(_stable_read(output / relative)[0], str(relative))
    if not isinstance(value, dict):
        raise InvalidLCV001Artifact(f"LCV-001 JSON root is not an object: {relative}")
    return cast(dict[str, Any], value)


def _completed_snapshot(output: Path) -> custody.TreeSnapshot:
    artifact_payload, artifact_record = _stable_read(output / ARTIFACT_MANIFEST)
    artifact_value = _json_from_bytes(artifact_payload, str(ARTIFACT_MANIFEST))
    if not isinstance(artifact_value, dict) or set(artifact_value) != {
        "artifacts",
        "classification",
        "experiment_id",
        "parent_artifact_manifest_sha256",
        "schema_version",
    }:
        raise InvalidLCV001Artifact("LCV-001 artifact manifest schema differs")
    artifacts = _require_mapping(artifact_value.get("artifacts"), "LCV-001 artifact records")
    if set(artifacts) != {str(path) for path in ARTIFACT_FILES}:
        raise InvalidLCV001Artifact("LCV-001 artifact manifest membership differs")
    records: dict[Path, custody.ExpectedFile] = {}
    for path in ARTIFACT_FILES:
        record = _require_mapping(artifacts[str(path)], f"LCV-001 artifact record {path}")
        if set(record) != {"bytes", "mode", "sha256"} or record.get("mode") != READ_ONLY_MODE:
            raise InvalidLCV001Artifact(f"LCV-001 artifact record schema/mode differs: {path}")
        records[path] = custody.ExpectedFile(
            cast(str, record["sha256"]), cast(int, record["bytes"]), cast(int, record["mode"])
        )
    if artifact_record["mode"] != READ_ONLY_MODE:
        raise InvalidLCV001Artifact("LCV-001 artifact manifest is not immutable 0444")
    records[ARTIFACT_MANIFEST] = _payload_record(artifact_payload)
    try:
        return custody.snapshot_exact_tree(output, records, directory_mode=0o555)
    except custody.CustodyError as error:
        raise InvalidLCV001Artifact(str(error)) from error


def _verify_in_process(output: Path) -> dict[str, object]:
    snapshot = _completed_snapshot(output)
    input_manifest = _snapshot_json(snapshot, INPUT_MANIFEST)
    prepared_expectations, freeze = _prepared_expectations(input_manifest, snapshot.payloads[INPUT_MANIFEST])
    for path, record in prepared_expectations.items():
        if snapshot.records.get(path) != record:
            raise InvalidLCV001Artifact(f"completed prepared record differs: {path}")
    _validate_input_manifest(input_manifest, snapshot, freeze)
    artifact = _snapshot_json(snapshot, ARTIFACT_MANIFEST)
    if artifact != _artifact_manifest_payload(snapshot.records):
        raise InvalidLCV001Artifact("LCV-001 artifact manifest differs")
    _manifest, marker, runtime, closure, controls = _validate_provisional_snapshot(snapshot)
    cleanup = _snapshot_json(snapshot, CLEANUP_RECEIPT)
    result = _snapshot_json(snapshot, RESULTS_FILE)
    _validate_runtime_receipt(runtime)
    try:
        supervisor.validate_cleanup_receipt(cleanup, role="formal")
    except supervisor.SupervisorError as error:
        raise InvalidLCV001Artifact(str(error)) from error
    if cleanup.get("unit") != cast(Mapping[str, Any], runtime["supervisor"]).get("unit"):
        raise InvalidLCV001Artifact("LCV-001 cleanup/runtime unit cross-link differs")
    expected_result = {
        "classification": "PASS",
        "cleanup_receipt_sha256": _canonical_sha256(cleanup),
        "experiment_id": EXPERIMENT_ID,
        "formal_start": marker,
        "mutation_controls_sha256": _canonical_sha256(controls),
        "parent_closure_sha256": _canonical_sha256(closure),
        "runtime_receipt_sha256": _canonical_sha256(runtime),
        "schema_version": SCHEMA_VERSION,
        "status": "completed_after_cgroup_cleanup",
        "statement": "host-bound sealed lineage/runtime verified after formal cgroup cleanup; no scientific outcome",
    }
    if result != expected_result:
        raise InvalidLCV001Artifact("LCV-001 result cross-links differ")
    report = snapshot.payloads[REPORT_FILE].decode("utf-8")
    if report != _report(result, closure, runtime):
        raise InvalidLCV001Artifact("LCV-001 report differs from sealed result")
    return {
        "classification": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "outcomes": "verified_results",
        "status": "verified",
    }


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    destination = output if output.is_absolute() else REPO_ROOT / output
    return _verify_in_process(destination)


def _verify_semantic_in_process(output: Path) -> dict[str, object]:
    try:
        supervisor.assert_current_cgroup("semantic")
    except supervisor.SupervisorError as error:
        raise InvalidLCV001Runtime(str(error)) from error
    os.environ.pop("LCV001_CGROUP_UNIT", None)
    os.environ.pop("LCV001_CGROUP_ROLE", None)
    try:
        runtime_probe.observe_and_validate()
    except runtime_probe.RuntimeClosureError as error:
        raise InvalidLCV001Runtime(str(error)) from error
    result = _verify_in_process(output)
    expected = _read_output_json(output, PARENT_CLOSURE)
    observed, _ = _validate_parent_closure(output / PARENT_COPY_ROOT)
    if observed != expected:
        raise InvalidLCV001Artifact("LCV-001 parent consumer semantics do not regenerate")
    return {**result, "outcomes": "verified_semantic_results"}


def verify_semantic(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    return _launch_lexical("_semantic-verify", output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LCV-001 sealed lineage/runtime closure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--parent", type=Path, default=MM007_ROOT)
    prepare_parser.add_argument("--audit", type=Path, default=PRE_REAL_AUDIT_DOC)
    for name in ("run", "verify", "verify-semantic", "_formal-run", "_semantic-verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            value = prepare(arguments.output, arguments.parent, arguments.audit)
        elif arguments.command == "run":
            value = run(arguments.output)
        elif arguments.command == "verify":
            value = verify(arguments.output)
        elif arguments.command == "verify-semantic":
            value = verify_semantic(arguments.output)
        elif arguments.command == "_formal-run":
            path = arguments.output if arguments.output.is_absolute() else REPO_ROOT / arguments.output
            value = _formal_run_in_process(path)
        else:
            path = arguments.output if arguments.output.is_absolute() else REPO_ROOT / arguments.output
            value = _verify_semantic_in_process(path)
        success_payload = _cli_json_line(value, compact=True)
        try:
            _write_cli_line(success_payload, sys.stdout)
        except (OSError, ValueError) as output_error:
            classified_output_error = InvalidLCV001Runtime(
                f"LCV-001 child output failed: {type(output_error).__name__}: "
                f"{_safe_exception_text(output_error)}"
            )
            return _write_classified_cli_failure(classified_output_error, transport=True)
    except (InvalidLCV001Artifact, InvalidLCV001Runtime, InvalidLCV001Verifier) as error:
        return _write_classified_cli_failure(error, transport=False)
    except OSError as host_error:
        classified_host_error = InvalidLCV001Runtime(
            f"LCV-001 child host I/O failed: {type(host_error).__name__}: {_safe_exception_text(host_error)}"
        )
        return _write_classified_cli_failure(classified_host_error, transport=True)
    except KeyboardInterrupt as interrupt:
        if arguments.command not in {"_formal-run", "_semantic-verify"}:
            raise
        classified_interrupt = InvalidLCV001Runtime(
            f"LCV-001 child interrupted: {type(interrupt).__name__}: {_safe_exception_text(interrupt)}"
        )
        return _write_classified_cli_failure(classified_interrupt, transport=True)
    return 0


__all__ = [
    "DEFAULT_OUTPUT",
    "EXPECTED_OUTPUT",
    "InvalidLCV001Artifact",
    "InvalidLCV001Runtime",
    "InvalidLCV001Verifier",
    "prepare",
    "run",
    "verify",
    "verify_semantic",
]
