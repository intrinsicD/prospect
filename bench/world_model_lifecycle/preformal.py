"""Trusted, immutable preformal test evidence for WM-001 protocol 1.8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .assurance import ASSURANCE

SCHEMA = "prospect.wm001.preformal-test-report.v2"
EXPERIMENT_ID = "WM-001"
PROTOCOL_VERSION = "1.8.0"
REPORT_NAME = "preformal-test-report-v1.8.0.json"
PREFORMAL_REPORT_NAME = REPORT_NAME
LOG_PREFIX = "preformal-v1.8.0-command-"
_EVIDENCE_PREFIX = "preformal-"
SOURCE_RELATIVE_PATH = "bench/world_model_lifecycle/preformal.py"
REVIEW_RELATIVE_PATH = "docs/wm001-v180-prospective-harness-review.json"
REVIEW_SCHEMA = "prospect.wm001.prospective-harness-review.v1"


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
        raise RuntimeError("preformal evidence requires an explicit Prospect Git worktree")
    candidate = Path(completed.stdout.strip())
    if (
        not candidate.is_absolute()
        or candidate.resolve(strict=True) != candidate
        or not (candidate / ".git").exists()
        or not (candidate / SOURCE_RELATIVE_PATH).is_file()
    ):
        raise RuntimeError("preformal evidence Git worktree is absent or aliased")
    return candidate


REPO = _repository_root()
REVIEW_PATH = REPO / REVIEW_RELATIVE_PATH
DEVELOPMENT_RESULTS_ROOT = (
    REPO / "bench" / "world_model_lifecycle" / "results" / "development"
)
DEVELOPMENT_CLOSURE_PATH = (
    DEVELOPMENT_RESULTS_ROOT / "development-closure-v1.8.0.json"
)
PREFORMAL_REPORT_PATH = DEVELOPMENT_RESULTS_ROOT / REPORT_NAME
LAUNCH_BOOTSTRAP_PATH = REPO / "bench/world_model_lifecycle/launch_bootstrap.py"
PRODUCER_BOOTSTRAP_PATH = REPO / "bench/world_model_lifecycle/producer_bootstrap.py"
_OPTIONAL_ENVIRONMENT_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "ROCR_VISIBLE_DEVICES",
)
_FIXED_ENVIRONMENT = {
    "COLUMNS": "120",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LAZY_LEGACY_OP": "False",
    "NO_COLOR": "1",
    "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "SDL_AUDIODRIVER": "dsp",
    "TERM": "dumb",
    "TZ": "UTC",
}
_RUNTIME_ENVIRONMENT_KEYS = frozenset(
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
_RUNTIME_FLAGS = {
    "dont_write_bytecode": 1,
    "ignore_environment": 1,
    "isolated": 1,
    "no_site": 1,
    "no_user_site": 1,
    "safe_path": True,
}
_COMMAND_NAMES = (
    "protocol-seal-continuity",
    "ruff",
    "mypy-core",
    "mypy-wm001",
    "pytest-epistemic",
    "pytest-wm001",
    "audit-runner-adversarial",
    "prospective-harness-review",
    "runtime-development-evidence",
    "runtime-bootstrap-inventory-conformance",
)
_WM001_MYPY_FILES = (
    "bench/world_model_lifecycle/audit_runner.py",
    "bench/world_model_lifecycle/artifact.py",
    "bench/world_model_lifecycle/artifact_audit.py",
    "bench/world_model_lifecycle/adjudication.py",
    "bench/world_model_lifecycle/binding.py",
    "bench/world_model_lifecycle/experiment.py",
    "bench/world_model_lifecycle/launch_bootstrap.py",
    "bench/world_model_lifecycle/operator.py",
    "bench/world_model_lifecycle/preformal.py",
    "bench/world_model_lifecycle/producer_bootstrap.py",
    "bench/world_model_lifecycle/restore_eval.py",
    "bench/world_model_lifecycle/run.py",
)
_REVIEW_FIELDS = {
    "schema",
    "experiment_id",
    "protocol_version",
    "implementation_files",
    "implementation_manifest_sha256",
    "reviewer",
    "disposition",
    "unresolved_blockers",
    "findings",
}
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
_SINGLE_LINK_CUSTODY = 1
_OUTER_FINALIZED_CUSTODY = 2
_PREFORMAL_INPUT_NLINKS = {
    "development_closure": _SINGLE_LINK_CUSTODY,
    "launch_bootstrap": _SINGLE_LINK_CUSTODY,
    "producer_bootstrap": _SINGLE_LINK_CUSTODY,
    "prospective_review": _SINGLE_LINK_CUSTODY,
    "runtime_seal": _OUTER_FINALIZED_CUSTODY,
}


class PreformalEvidenceError(RuntimeError):
    """Preformal evidence is incomplete, mutable, or outside its fixed contract."""


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One exact required subprocess invocation."""

    name: str
    role: str
    argv: tuple[str, ...]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PreformalEvidenceError(f"canonical JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise PreformalEvidenceError(f"canonical JSON contains non-finite value {value}")


def _load_canonical_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreformalEvidenceError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PreformalEvidenceError(f"{label} must contain one JSON object")
    if payload != _canonical_json_bytes(value) + b"\n":
        raise PreformalEvidenceError(f"{label} is not canonical JSON followed by LF")
    return value


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PreformalEvidenceError(f"{label} does not exist") from error
    if resolved != absolute or not absolute.is_dir() or absolute.is_symlink():
        raise PreformalEvidenceError(f"{label} must be one canonical non-aliased directory")
    return absolute


def _canonical_existing_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PreformalEvidenceError(f"{label} does not exist") from error
    if resolved != absolute or not absolute.is_file() or absolute.is_symlink():
        raise PreformalEvidenceError(f"{label} must be one canonical non-aliased file")
    return absolute


def _read_regular(
    path: Path,
    *,
    label: str,
    expected_nlink: int = 1,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PreformalEvidenceError(f"{label} cannot be opened as a regular file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != expected_nlink
        ):
            raise PreformalEvidenceError(f"{label} is aliased or not a regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if identity_before != identity_after or len(payload) != before.st_size:
        raise PreformalEvidenceError(f"{label} changed while it was read")
    return payload


def _atomic_write_exclusive(path: Path, payload: bytes) -> None:
    """Publish one immutable file through an fsynced no-replace hard link."""

    parent = _canonical_existing_directory(path.parent, label="evidence directory")
    if path.parent != parent:
        raise PreformalEvidenceError("evidence path has a noncanonical parent")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to replace immutable preformal evidence: {path}") from None
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _file_identity(
    path: Path,
    *,
    label: str,
    expected_nlink: int,
) -> dict[str, object]:
    canonical = _canonical_existing_file(path, label=label)
    payload = _read_regular(
        canonical,
        label=label,
        expected_nlink=expected_nlink,
    )
    return {
        "path": str(canonical),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _executable_identity(
    path: str | Path = sys.executable,
    *,
    implementation: str | None = None,
    version: str | None = None,
) -> dict[str, object]:
    invocation = Path(path)
    if not invocation.is_absolute() or Path(os.path.abspath(invocation)) != invocation or not invocation.exists():
        raise PreformalEvidenceError("Python executable is not one existing absolute path")
    link_target = os.readlink(invocation) if invocation.is_symlink() else None
    resolved = invocation.resolve(strict=True)
    payload = _read_regular(resolved, label="resolved Python executable")
    return {
        "invocation_path": str(invocation),
        "invocation_symlink_target": link_target,
        "resolved_path": str(resolved),
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "implementation": implementation or platform.python_implementation(),
        "version": version or platform.python_version(),
    }


def _source_identity() -> dict[str, object]:
    source = REPO / SOURCE_RELATIVE_PATH
    if source.resolve(strict=True) != source:
        raise PreformalEvidenceError("preformal generator source is missing or aliased")
    payload = _read_regular(source, label="preformal generator source")
    return {
        "path": SOURCE_RELATIVE_PATH,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _sanitized_environment() -> dict[str, str]:
    path = os.environ.get("PATH")
    if not path or "\x00" in path:
        raise PreformalEvidenceError("PATH is required for the preformal command environment")
    environment = {
        **_FIXED_ENVIRONMENT,
        "PATH": path,
    }
    for name in _OPTIONAL_ENVIRONMENT_KEYS:
        value = os.environ.get(name)
        if value is not None:
            if "\x00" in value:
                raise PreformalEvidenceError(f"environment variable {name} contains NUL")
            environment[name] = value
    return dict(sorted(environment.items()))


def _environment_identity(environment: dict[str, str]) -> dict[str, object]:
    variables = [{"name": name, "value": value} for name, value in sorted(environment.items())]
    return {
        "variables": variables,
        "sha256": _sha256(_canonical_json_bytes(variables)),
    }


def _environment_from_identity(
    identity: object,
    *,
    role: str,
) -> dict[str, str]:
    if not isinstance(identity, dict) or set(identity) != {"variables", "sha256"}:
        raise PreformalEvidenceError(f"{role} environment identity has wrong fields")
    variables = identity.get("variables")
    if (
        not isinstance(variables, list)
        or not variables
        or any(
            not isinstance(row, dict)
            or set(row) != {"name", "value"}
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("value"), str)
            for row in variables
        )
    ):
        raise PreformalEvidenceError(f"{role} environment variables are invalid")
    pairs = [(str(row["name"]), str(row["value"])) for row in variables]
    if pairs != sorted(pairs) or len({name for name, _ in pairs}) != len(pairs):
        raise PreformalEvidenceError(f"{role} environment variables are duplicated or unordered")
    environment = dict(pairs)
    if any("\x00" in name or "\x00" in value for name, value in pairs):
        raise PreformalEvidenceError(f"{role} environment contains NUL")
    if role == "qa":
        allowed = {*_FIXED_ENVIRONMENT, "PATH", *_OPTIONAL_ENVIRONMENT_KEYS}
        valid = (
            not set(environment) - allowed
            and environment.get("PATH") not in {None, ""}
            and all(environment.get(name) == value for name, value in _FIXED_ENVIRONMENT.items())
        )
    elif role == "runtime":
        valid = (
            not set(environment) - _RUNTIME_ENVIRONMENT_KEYS
            and environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and environment.get("LAZY_LEGACY_OP") == "False"
            and environment.get("LC_ALL") == "C.UTF-8"
            and environment.get("PATH") == "/usr/bin:/bin"
            and environment.get("PYGAME_HIDE_SUPPORT_PROMPT") == "hide"
            and environment.get("SDL_AUDIODRIVER") == "dsp"
            and environment.get("TZ") == "UTC"
        )
    else:
        raise PreformalEvidenceError(f"unknown environment role {role!r}")
    if not valid or identity.get("sha256") != _environment_identity(environment)["sha256"]:
        raise PreformalEvidenceError(f"{role} environment differs from its fixed contract")
    return environment


def _canonical_distribution_name(value: str) -> str:
    canonical = re.sub(r"[-_.]+", "-", value).lower()
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", canonical) is None:
        raise PreformalEvidenceError("QA distribution has no canonical name")
    return canonical


def _distribution_is_editable(distribution: importlib.metadata.Distribution) -> bool:
    text = distribution.read_text("direct_url.json")
    if text is None:
        return False
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise PreformalEvidenceError("QA distribution direct_url.json is malformed") from error
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("dir_info"), dict)
        and value["dir_info"].get("editable") is True
    )


def _distribution_identity(
    name: str,
    distribution: importlib.metadata.Distribution,
) -> dict[str, object]:
    files = tuple(sorted(distribution.files or (), key=str))
    if not files:
        raise PreformalEvidenceError(f"QA distribution {name!r} has no declared files")
    digest = hashlib.sha256(b"prospect.wm001.qa-distribution.v1\0")
    total_bytes = 0
    seen: set[str] = set()
    for entry in files:
        relative = str(entry)
        if not relative or "\x00" in relative or relative in seen:
            raise PreformalEvidenceError(f"QA distribution {name!r} has unsafe declared files")
        seen.add(relative)
        located = Path(str(distribution.locate_file(entry)))
        absolute = Path(os.path.abspath(located))
        if absolute.resolve(strict=True) != absolute or absolute.is_symlink() or not absolute.is_file():
            raise PreformalEvidenceError(f"QA distribution {name!r} has missing or aliased file {relative!r}")
        payload = _read_regular(absolute, label=f"QA distribution {name} file")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        total_bytes += len(payload)
    identity = {
        "name": name,
        "version": distribution.version,
        "editable": _distribution_is_editable(distribution),
        "declared_file_count": len(files),
        "total_bytes": total_bytes,
        "distribution_sha256": digest.hexdigest(),
    }
    if identity["editable"] is True:
        raise PreformalEvidenceError(f"editable QA distribution is forbidden: {name}")
    return identity


def _installed_preformal_identity() -> dict[str, object]:
    """Bind the isolated QA module bytes to the live reviewed source bytes."""

    try:
        distribution = importlib.metadata.distribution("prospect")
    except importlib.metadata.PackageNotFoundError as error:
        raise PreformalEvidenceError(
            "isolated QA requires one installed Prospect distribution"
        ) from error
    if _distribution_is_editable(distribution):
        raise PreformalEvidenceError(
            "isolated QA requires a non-editable Prospect distribution"
        )
    matches = [
        entry
        for entry in (distribution.files or ())
        if str(entry).replace(os.sep, "/") == SOURCE_RELATIVE_PATH
    ]
    if len(matches) != 1:
        raise PreformalEvidenceError(
            "installed Prospect distribution does not declare exactly one preformal module"
        )
    installed = Path(str(distribution.locate_file(matches[0])))
    canonical_installed = Path(os.path.abspath(installed))
    if (
        canonical_installed.resolve(strict=True) != canonical_installed
        or canonical_installed.is_symlink()
        or not canonical_installed.is_file()
    ):
        raise PreformalEvidenceError(
            "installed Prospect preformal module is missing or aliased"
        )
    installed_payload = _read_regular(
        canonical_installed,
        label="installed Prospect preformal module",
    )
    live = _canonical_existing_file(
        REPO / SOURCE_RELATIVE_PATH,
        label="reviewed Prospect preformal source",
    )
    live_payload = _read_regular(
        live,
        label="reviewed Prospect preformal source",
    )
    if installed_payload != live_payload:
        raise PreformalEvidenceError(
            "installed Prospect preformal module differs from the live reviewed source"
        )
    return {
        "path": str(canonical_installed),
        "bytes": len(installed_payload),
        "sha256": _sha256(installed_payload),
    }


def _qa_closure() -> dict[str, object]:
    paths = list(sys.path)
    if not paths or any(not isinstance(path, str) or "\x00" in path for path in paths) or len(paths) != len(set(paths)):
        raise PreformalEvidenceError("QA sys.path is absent, duplicated, or malformed")
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not isinstance(raw_name, str):
            raise PreformalEvidenceError("QA distribution has no Name metadata")
        name = _canonical_distribution_name(raw_name)
        if name in distributions:
            raise PreformalEvidenceError(f"duplicate QA distribution identity: {name}")
        distributions[name] = distribution
    if not distributions:
        raise PreformalEvidenceError("QA environment has no installed distributions")
    rows = [_distribution_identity(name, distribution) for name, distribution in sorted(distributions.items())]
    closure: dict[str, object] = {
        "schema": "prospect.wm001.qa-closure.v1",
        "sys_path": paths,
        "distributions": rows,
    }
    closure["inventory_sha256"] = _sha256(_canonical_json_bytes(closure))
    return closure


def _validate_qa_closure(closure: object) -> dict[str, object]:
    if not isinstance(closure, dict) or set(closure) != {
        "schema",
        "sys_path",
        "distributions",
        "inventory_sha256",
    }:
        raise PreformalEvidenceError("QA closure has wrong fields")
    paths = closure.get("sys_path")
    rows = closure.get("distributions")
    if (
        closure.get("schema") != "prospect.wm001.qa-closure.v1"
        or not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or "\x00" in path for path in paths)
        or len(paths) != len(set(paths))
        or not isinstance(rows, list)
        or not rows
    ):
        raise PreformalEvidenceError("QA closure identity is malformed")
    names: list[str] = []
    row_fields = {
        "name",
        "version",
        "editable",
        "declared_file_count",
        "total_bytes",
        "distribution_sha256",
    }
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != row_fields
            or not isinstance(row.get("name"), str)
            or row.get("name") != _canonical_distribution_name(cast(str, row["name"]))
            or not isinstance(row.get("version"), str)
            or not row.get("version")
            or row.get("editable") is not False
            or type(row.get("declared_file_count")) is not int
            or cast(int, row["declared_file_count"]) <= 0
            or type(row.get("total_bytes")) is not int
            or cast(int, row["total_bytes"]) < 0
            or not _is_sha256(row.get("distribution_sha256"))
        ):
            raise PreformalEvidenceError("QA closure distribution row is malformed")
        names.append(cast(str, row["name"]))
    if names != sorted(names) or len(names) != len(set(names)):
        raise PreformalEvidenceError("QA closure distributions are duplicated or unordered")
    if "prospect" not in names:
        raise PreformalEvidenceError(
            "QA closure omits the non-editable Prospect distribution required by isolated QA commands"
        )
    unsigned = {key: value for key, value in closure.items() if key != "inventory_sha256"}
    if closure.get("inventory_sha256") != _sha256(_canonical_json_bytes(unsigned)):
        raise PreformalEvidenceError("QA closure digest is invalid")
    return closure


def _capture_qa_closure(
    *,
    executable: str,
    environment: dict[str, str],
) -> dict[str, object]:
    completed = subprocess.run(
        (
            executable,
            "-I",
            "-B",
            "-m",
            "bench.world_model_lifecycle.preformal",
            "emit-qa-closure",
        ),
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        diagnostic = completed.stderr[:4096].decode("utf-8", errors="replace")
        raise PreformalEvidenceError(f"QA closure subprocess failed or wrote stderr: {diagnostic}")
    return _validate_qa_closure(_load_canonical_object(completed.stdout, label="QA closure subprocess"))


def _test_files(pattern: str, *, label: str) -> tuple[str, ...]:
    matches = tuple(
        sorted(
            path.relative_to(REPO).as_posix()
            for path in (REPO / "tests").glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    )
    if not matches:
        raise PreformalEvidenceError(f"required {label} test set is empty")
    return matches


def _implementation_files(*, environment: dict[str, str] | None = None) -> list[dict[str, object]]:
    selected_environment = _sanitized_environment() if environment is None else environment
    completed = subprocess.run(
        ("git", "ls-files", "--", "src/prospect", "bench", "tests"),
        cwd=REPO,
        env=selected_environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise PreformalEvidenceError("prospective review cannot enumerate tracked implementation files")
    tracked_python = [REPO / relative for relative in completed.stdout.splitlines() if relative.endswith(".py")]
    candidates = [
        *tracked_python,
        REPO / "Makefile",
        REPO / "pyproject.toml",
        REPO / "requirements-wm001.lock",
        REPO / "bench/world_model_lifecycle/SEALED_PROTOCOL.sha256",
        REPO / "bench/world_model_lifecycle/assurance.py",
        REPO / "bench/world_model_lifecycle/protocol.json",
        REPO / "bench/world_model_lifecycle/schemas/raw-result.schema.json",
        REPO / "bench/world_model_lifecycle/schemas/formal-binding.schema.json",
    ]
    rows: list[dict[str, object]] = []
    for path in sorted(set(candidates)):
        canonical = _canonical_existing_file(path, label="prospective implementation source")
        payload = _read_regular(canonical, label="prospective implementation source")
        rows.append(
            {
                "path": canonical.relative_to(REPO).as_posix(),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return rows


def verify_prospective_review(review_path: Path) -> dict[str, Any]:
    """Verify an independently authored review against the exact live source manifest."""

    _installed_preformal_identity()
    canonical = _canonical_existing_file(review_path, label="prospective harness review")
    payload = _read_regular(canonical, label="prospective harness review")
    review = _load_canonical_object(payload, label="prospective harness review")
    rows = review.get("implementation_files")
    expected_rows = _implementation_files()
    reviewer = review.get("reviewer")
    findings = review.get("findings")
    if (
        set(review) != _REVIEW_FIELDS
        or review.get("schema") != REVIEW_SCHEMA
        or review.get("experiment_id") != EXPERIMENT_ID
        or review.get("protocol_version") != PROTOCOL_VERSION
        or rows != expected_rows
        or review.get("implementation_manifest_sha256") != _sha256(_canonical_json_bytes(expected_rows))
        or not isinstance(reviewer, dict)
        or set(reviewer) != {"kind", "identifier"}
        or reviewer.get("kind") != "independent-adversarial-referee"
        or not isinstance(reviewer.get("identifier"), str)
        or not reviewer.get("identifier")
        or cast(str, reviewer["identifier"]).strip() != reviewer["identifier"]
        or review.get("disposition") != "accepted"
        or review.get("unresolved_blockers") != []
        or not isinstance(findings, list)
    ):
        raise PreformalEvidenceError("prospective harness review is not an accepted exact-source review")
    finding_fields = {"id", "severity", "summary", "resolution"}
    finding_ids: list[str] = []
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or set(finding) != finding_fields
            or not isinstance(finding.get("id"), str)
            or not finding.get("id")
            or finding.get("severity") not in {"blocker", "major", "minor", "note"}
            or not isinstance(finding.get("summary"), str)
            or not finding.get("summary")
            or finding.get("resolution") not in {"resolved", "informational"}
        ):
            raise PreformalEvidenceError("prospective harness review finding is malformed")
        finding_ids.append(cast(str, finding["id"]))
    if finding_ids != sorted(finding_ids) or len(finding_ids) != len(set(finding_ids)):
        raise PreformalEvidenceError("prospective harness review findings are duplicated or unordered")
    return review


def _validated_runtime_seal(
    path: Path,
    *,
    runtime_executable: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    from .operator import verify_outer_completion

    canonical = _canonical_existing_file(path, label="runtime seal")
    payload = _read_regular(
        canonical,
        label="runtime seal",
        expected_nlink=_OUTER_FINALIZED_CUSTODY,
    )
    try:
        verify_outer_completion(canonical)
    except (OSError, RuntimeError, ValueError) as error:
        raise PreformalEvidenceError(
            "runtime seal lacks its deterministic outer completion"
        ) from error
    seal = _load_canonical_object(payload, label="runtime seal")
    python = seal.get("python")
    environment = seal.get("process_environment")
    if (
        set(seal) != _RUNTIME_SEAL_FIELDS
        or seal.get("schema") != "prospect.wm001.runtime-seal.v1"
        or seal.get("experiment_id") != EXPERIMENT_ID
        or seal.get("protocol_version") != PROTOCOL_VERSION
        or seal.get("assurance") != ASSURANCE
        or seal.get("worktree_clean") is not True
        or seal.get("required_flags") != _RUNTIME_FLAGS
        or not isinstance(python, dict)
        or set(python) != {"executable", "resolved_executable", "sha256", "version"}
        or python.get("executable") != str(runtime_executable)
        or not isinstance(python.get("version"), list)
        or len(cast(list[object], python["version"])) != 3
        or any(type(item) is not int or item < 0 for item in cast(list[object], python["version"]))
        or not _is_sha256(python.get("sha256"))
        or not isinstance(environment, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items())
        or not isinstance(seal.get("package_roots"), list)
        or len(cast(list[object], seal["package_roots"])) != 1
        or not isinstance(seal.get("standard_library"), dict)
        or not isinstance(seal.get("package_ownership"), dict)
        or not _is_sha256(seal.get("bootstrap_source_sha256"))
    ):
        raise PreformalEvidenceError("runtime seal identity is malformed")
    runtime_environment = cast(dict[str, str], environment)
    _environment_from_identity(_environment_identity(runtime_environment), role="runtime")
    version = ".".join(str(value) for value in cast(list[int], python["version"]))
    executable = _executable_identity(
        runtime_executable,
        implementation="CPython",
        version=version,
    )
    if python.get("resolved_executable") != executable["resolved_path"] or python.get("sha256") != executable["sha256"]:
        raise PreformalEvidenceError("runtime seal interpreter differs from its executable")
    return seal, runtime_environment


def required_commands(
    qa_executable_path: str | None = None,
    *,
    runtime_executable_path: str | None = None,
    runtime_seal_path: Path | None = None,
    development_closure_path: Path = DEVELOPMENT_CLOSURE_PATH,
    prospective_review_path: Path = REVIEW_PATH,
    device: str = "cpu",
) -> tuple[CommandSpec, ...]:
    """Return the fixed, ordered v1.8 preformal command contract."""

    if device not in {"cpu", "cuda"}:
        raise PreformalEvidenceError("preformal device must be cpu or cuda")
    qa_executable = str(
        _executable_identity(sys.executable if qa_executable_path is None else qa_executable_path)["invocation_path"]
    )
    runtime_executable = str(
        _executable_identity(qa_executable if runtime_executable_path is None else runtime_executable_path)[
            "invocation_path"
        ]
    )
    runtime_seal = (
        REPO
        / "bench/world_model_lifecycle/results/development/runtime-seal-v1.8.0.json"
        if runtime_seal_path is None
        else _canonical_existing_file(runtime_seal_path, label="runtime seal")
    )
    development_closure = (
        development_closure_path if development_closure_path.is_absolute() else Path.cwd() / development_closure_path
    )
    review = prospective_review_path if prospective_review_path.is_absolute() else Path.cwd() / prospective_review_path
    epistemic_tests = _test_files("test_epistemic_*.py", label="epistemic")
    wm001_tests = _test_files("test_world_model_*.py", label="WM-001")
    launch = str(LAUNCH_BOOTSTRAP_PATH)
    bootstrap = str(PRODUCER_BOOTSTRAP_PATH)
    runtime_prefix = (
        runtime_executable,
        "-I",
        "-S",
        "-B",
        launch,
        "--bootstrap",
        bootstrap,
        "--runtime-seal",
        str(runtime_seal),
        "preformal-runtime",
    )
    specifications = (
        CommandSpec(
            "protocol-seal-continuity",
            "qa",
            (qa_executable, "-m", "bench.world_model_lifecycle.verify", "protocol"),
        ),
        CommandSpec(
            "ruff",
            "qa",
            (qa_executable, "-m", "ruff", "check", "src/prospect", "bench", "tests"),
        ),
        CommandSpec("mypy-core", "qa", (qa_executable, "-m", "mypy")),
        CommandSpec(
            "mypy-wm001",
            "qa",
            (
                qa_executable,
                "-m",
                "mypy",
                "--follow-imports=skip",
                *_WM001_MYPY_FILES,
            ),
        ),
        CommandSpec(
            "pytest-epistemic",
            "qa",
            (qa_executable, "-m", "pytest", "-q", *epistemic_tests),
        ),
        CommandSpec(
            "pytest-wm001",
            "qa",
            (qa_executable, "-m", "pytest", "-q", *wm001_tests),
        ),
        CommandSpec(
            "audit-runner-adversarial",
            "qa",
            (
                qa_executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_world_model_audit_runner.py",
                "tests/test_world_model_prebinding_audit.py",
            ),
        ),
        CommandSpec(
            "prospective-harness-review",
            "qa",
            (
                qa_executable,
                "-I",
                "-B",
                "-m",
                "bench.world_model_lifecycle.preformal",
                "verify-prospective-review",
                "--review",
                str(review),
            ),
        ),
        CommandSpec(
            "runtime-development-evidence",
            "runtime",
            (
                *runtime_prefix,
                "development-evidence",
                "--development-closure",
                str(development_closure),
            ),
        ),
        CommandSpec(
            "runtime-bootstrap-inventory-conformance",
            "runtime",
            (
                *runtime_prefix,
                "bootstrap-inventory-conformance",
                "--device",
                device,
            ),
        ),
    )
    if tuple(specification.name for specification in specifications) != _COMMAND_NAMES:
        raise PreformalEvidenceError("internal preformal command order changed")
    return specifications


def _git_output(arguments: tuple[str, ...], *, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr[:4096].decode("utf-8", errors="replace")
        raise PreformalEvidenceError(f"Git identity command failed: {diagnostic}")
    return completed.stdout.decode("utf-8").strip()


def _git_identity(*, environment: dict[str, str]) -> dict[str, object]:
    identity = {
        "commit": _git_output(("rev-parse", "HEAD"), environment=environment),
        "tree": _git_output(("rev-parse", "HEAD^{tree}"), environment=environment),
        "worktree_clean": not _git_output(
            ("status", "--short", "--untracked-files=all"),
            environment=environment,
        ),
    }
    _validate_git_identity(identity)
    return identity


def _validate_git_identity(identity: object) -> None:
    if not isinstance(identity, dict) or set(identity) != {
        "commit",
        "tree",
        "worktree_clean",
    }:
        raise PreformalEvidenceError("Git identity has an unexpected field set")
    for field in ("commit", "tree"):
        value = identity.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PreformalEvidenceError(f"Git identity {field} is not a SHA-1")
    if type(identity.get("worktree_clean")) is not bool:
        raise PreformalEvidenceError("Git worktree cleanliness is not Boolean")


def _run_command(
    specification: CommandSpec,
    *,
    environment: dict[str, str],
) -> tuple[int, bytes, bytes]:
    completed = subprocess.run(
        specification.argv,
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _log_filename(ordinal: int, name: str, stream: str, digest: str) -> str:
    return f"{LOG_PREFIX}{ordinal:02d}-{name}.{stream}.{digest}.log"


def _is_preformal_evidence_name(name: str) -> bool:
    return name.startswith(_EVIDENCE_PREFIX) or name.startswith(f".{_EVIDENCE_PREFIX}")


def _write_log(
    directory: Path,
    *,
    ordinal: int,
    name: str,
    stream: str,
    payload: bytes,
) -> dict[str, object]:
    digest = _sha256(payload)
    filename = _log_filename(ordinal, name, stream, digest)
    _atomic_write_exclusive(directory / filename, payload)
    return {"file": filename, "bytes": len(payload), "sha256": digest}


def _input_identities(
    *,
    runtime_seal_path: Path,
    development_closure_path: Path,
    prospective_review_path: Path,
) -> dict[str, dict[str, object]]:
    inputs = {
        "development_closure": (
            development_closure_path,
            "development closure",
        ),
        "launch_bootstrap": (
            LAUNCH_BOOTSTRAP_PATH,
            "launch bootstrap",
        ),
        "producer_bootstrap": (
            PRODUCER_BOOTSTRAP_PATH,
            "producer bootstrap",
        ),
        "prospective_review": (
            prospective_review_path,
            "prospective harness review",
        ),
        "runtime_seal": (
            runtime_seal_path,
            "runtime seal",
        ),
    }
    return {
        name: _file_identity(
            path,
            label=label,
            expected_nlink=_PREFORMAL_INPUT_NLINKS[name],
        )
        for name, (path, label) in inputs.items()
    }


def generate_preformal_report(
    output_path: Path,
    *,
    runtime_executable: Path,
    runtime_seal: Path,
    development_closure: Path = DEVELOPMENT_CLOSURE_PATH,
    prospective_review: Path = REVIEW_PATH,
    device: str = "cpu",
) -> Path:
    """Run the closed command suite and atomically publish its immutable report."""

    lexical = output_path if output_path.is_absolute() else Path.cwd() / output_path
    output = Path(os.path.abspath(output_path))
    if lexical != output:
        raise PreformalEvidenceError("preformal report path must not contain aliases")
    if output != PREFORMAL_REPORT_PATH:
        raise PreformalEvidenceError(
            "preformal report must use the sole canonical protocol-1.8 "
            f"path {PREFORMAL_REPORT_PATH}"
        )
    directory = _canonical_existing_directory(output.parent, label="evidence directory")
    if output.parent != directory:
        raise PreformalEvidenceError("preformal report path is aliased")
    if os.path.lexists(output) or any(_is_preformal_evidence_name(candidate.name) for candidate in directory.iterdir()):
        raise FileExistsError("preformal evidence directory already contains this protocol's evidence")
    runtime_executable_path = Path(str(_executable_identity(runtime_executable)["invocation_path"]))
    runtime_seal_path = _canonical_existing_file(runtime_seal, label="runtime seal")
    development_closure_path = _canonical_existing_file(
        development_closure,
        label="development closure",
    )
    prospective_review_path = _canonical_existing_file(
        prospective_review,
        label="prospective harness review",
    )
    if prospective_review_path != REVIEW_PATH:
        raise PreformalEvidenceError(f"prospective harness review must be the canonical {REVIEW_RELATIVE_PATH}")
    qa_environment = _sanitized_environment()
    qa_environment_identity = _environment_identity(qa_environment)
    runtime_seal_value, runtime_environment = _validated_runtime_seal(
        runtime_seal_path,
        runtime_executable=runtime_executable_path,
    )
    runtime_environment_identity = _environment_identity(runtime_environment)
    repository = _canonical_existing_directory(REPO, label="repository")
    git_before = _git_identity(environment=qa_environment)
    if git_before.get("worktree_clean") is not True:
        raise PreformalEvidenceError("preformal checks require a clean worktree before execution")
    if (
        runtime_seal_value.get("git_commit") != git_before["commit"]
        or runtime_seal_value.get("git_tree") != git_before["tree"]
        or runtime_seal_value.get("bootstrap_source_sha256")
        != _file_identity(
            PRODUCER_BOOTSTRAP_PATH,
            label="producer bootstrap",
            expected_nlink=_SINGLE_LINK_CUSTODY,
        )["sha256"]
    ):
        raise PreformalEvidenceError("runtime seal differs from the clean source under test")
    prospective_review_value = verify_prospective_review(prospective_review_path)
    qa_executable_before = _executable_identity()
    runtime_version = ".".join(str(value) for value in cast(dict[str, Any], runtime_seal_value["python"])["version"])
    runtime_executable_before = _executable_identity(
        runtime_executable_path,
        implementation="CPython",
        version=runtime_version,
    )
    qa_closure_before = _capture_qa_closure(
        executable=cast(str, qa_executable_before["invocation_path"]),
        environment=qa_environment,
    )
    source_before = _source_identity()
    inputs_before = _input_identities(
        runtime_seal_path=runtime_seal_path,
        development_closure_path=development_closure_path,
        prospective_review_path=prospective_review_path,
    )
    specifications = required_commands(
        str(qa_executable_before["invocation_path"]),
        runtime_executable_path=str(runtime_executable_path),
        runtime_seal_path=runtime_seal_path,
        development_closure_path=development_closure_path,
        prospective_review_path=prospective_review_path,
        device=device,
    )
    rows: list[dict[str, object]] = []
    for ordinal, specification in enumerate(specifications, start=1):
        selected_environment = qa_environment if specification.role == "qa" else runtime_environment
        selected_identity = qa_environment_identity if specification.role == "qa" else runtime_environment_identity
        exit_code, stdout, stderr = _run_command(
            specification,
            environment=selected_environment,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "name": specification.name,
                "role": specification.role,
                "argv": list(specification.argv),
                "cwd": str(repository),
                "environment_sha256": selected_identity["sha256"],
                "exit_code": exit_code,
                "passed": exit_code == 0,
                "stdout": _write_log(
                    directory,
                    ordinal=ordinal,
                    name=specification.name,
                    stream="stdout",
                    payload=stdout,
                ),
                "stderr": _write_log(
                    directory,
                    ordinal=ordinal,
                    name=specification.name,
                    stream="stderr",
                    payload=stderr,
                ),
            }
        )
    git_after = _git_identity(environment=qa_environment)
    qa_executable_after = _executable_identity()
    runtime_executable_after = _executable_identity(
        runtime_executable_path,
        implementation="CPython",
        version=runtime_version,
    )
    qa_closure_after = _capture_qa_closure(
        executable=cast(str, qa_executable_before["invocation_path"]),
        environment=qa_environment,
    )
    source_after = _source_identity()
    inputs_after = _input_identities(
        runtime_seal_path=runtime_seal_path,
        development_closure_path=development_closure_path,
        prospective_review_path=prospective_review_path,
    )
    runtime_seal_after, runtime_environment_after = _validated_runtime_seal(
        runtime_seal_path,
        runtime_executable=runtime_executable_path,
    )
    prospective_review_after = verify_prospective_review(prospective_review_path)
    identities_stable = (
        git_before == git_after
        and qa_executable_before == qa_executable_after
        and runtime_executable_before == runtime_executable_after
        and qa_closure_before == qa_closure_after
        and source_before == source_after
        and inputs_before == inputs_after
        and runtime_seal_value == runtime_seal_after
        and runtime_environment == runtime_environment_after
        and prospective_review_value == prospective_review_after
    )
    all_pass = (
        all(row["passed"] is True for row in rows) and git_after.get("worktree_clean") is True and identities_stable
    )
    report = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "repository_cwd": str(repository),
        "device": device,
        "qa_environment": qa_environment_identity,
        "runtime_environment": runtime_environment_identity,
        "git_before": git_before,
        "git_after": git_after,
        "qa_executable_before": qa_executable_before,
        "qa_executable_after": qa_executable_after,
        "runtime_executable_before": runtime_executable_before,
        "runtime_executable_after": runtime_executable_after,
        "qa_closure_before": qa_closure_before,
        "qa_closure_after": qa_closure_after,
        "runtime_seal": runtime_seal_value,
        "prospective_review": prospective_review_value,
        "input_files_before": inputs_before,
        "input_files_after": inputs_after,
        "generator_source_before": source_before,
        "generator_source_after": source_after,
        "identities_stable": identities_stable,
        "commands": rows,
        "all_pass": all_pass,
    }
    _atomic_write_exclusive(output, _canonical_json_bytes(report) + b"\n")
    return output


def _safe_sibling(directory: Path, filename: object, *, label: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise PreformalEvidenceError(f"{label} filename is missing")
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != filename or filename in {".", ".."}:
        raise PreformalEvidenceError(f"{label} filename escapes the evidence directory")
    return directory / relative


def _verify_log(
    directory: Path,
    reference: object,
    *,
    ordinal: int,
    command_name: str,
    stream: str,
) -> str:
    if not isinstance(reference, dict) or set(reference) != {"file", "bytes", "sha256"}:
        raise PreformalEvidenceError(f"{command_name} {stream} reference has wrong fields")
    digest = reference.get("sha256")
    size = reference.get("bytes")
    if not _is_sha256(digest) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise PreformalEvidenceError(f"{command_name} {stream} identity is invalid")
    expected_filename = _log_filename(ordinal, command_name, stream, cast(str, digest))
    if reference.get("file") != expected_filename:
        raise PreformalEvidenceError(f"{command_name} {stream} filename is not content-addressed")
    path = _safe_sibling(directory, reference.get("file"), label=f"{command_name} {stream}")
    payload = _read_regular(path, label=f"{command_name} {stream} log")
    if len(payload) != size or _sha256(payload) != digest:
        raise PreformalEvidenceError(f"{command_name} {stream} log identity changed")
    return expected_filename


def _validate_file_identity(identity: object, *, label: str) -> dict[str, object]:
    if (
        not isinstance(identity, dict)
        or set(identity) != {"path", "bytes", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not Path(cast(str, identity["path"])).is_absolute()
        or os.path.abspath(cast(str, identity["path"])) != identity["path"]
        or type(identity.get("bytes")) is not int
        or cast(int, identity["bytes"]) < 0
        or not _is_sha256(identity.get("sha256"))
    ):
        raise PreformalEvidenceError(f"{label} identity is malformed")
    return identity


def _runtime_bootstrap_conformance_from_report(
    directory: Path,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Reopen the sealed command-10 result-free conformance identity."""

    commands = report.get("commands")
    matches = (
        [
            row
            for row in commands
            if isinstance(row, Mapping)
            and row.get("name")
            == "runtime-bootstrap-inventory-conformance"
        ]
        if isinstance(commands, list)
        else []
    )
    if len(matches) != 1:
        raise PreformalEvidenceError(
            "preformal report lacks one bootstrap conformance command"
        )
    reference = matches[0].get("stdout")
    if (
        not isinstance(reference, Mapping)
        or set(reference) != {"file", "bytes", "sha256"}
        or type(reference.get("bytes")) is not int
        or cast(int, reference.get("bytes")) < 1
        or not _is_sha256(reference.get("sha256"))
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance stdout reference is malformed"
        )
    path = _safe_sibling(
        directory,
        reference.get("file"),
        label="bootstrap conformance stdout",
    )
    payload = _read_regular(
        path,
        label="bootstrap conformance stdout",
    )
    if (
        len(payload) != reference.get("bytes")
        or _sha256(payload) != reference.get("sha256")
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance stdout identity changed"
        )
    value = _load_canonical_object(
        payload,
        label="bootstrap conformance stdout",
    )
    if (
        set(value)
        != {
            "schema",
            "mode",
            "device",
            "passed",
            "inventory_sha256",
            "conformance_sha256",
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
        or value.get("mode")
        != "bootstrap-inventory-conformance"
        or value.get("device") != report.get("device")
        or value.get("passed") is not True
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "inventory_sha256",
                "conformance_sha256",
                "restart_runtime_conformance_report_sha256",
                "restart_runtime_execution_receipt_sha256",
            )
        )
        or value.get("restart_runtime_support_files")
        != [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ]
        or value.get("restart_runtime_repeat_count") != 3
        or value.get("restart_runtime_path_descriptor_equal")
        is not True
        or value.get("repeat_count") != 3
        or value.get("path_descriptor_equal") is not True
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance stdout is not one complete result-free pass"
        )
    return value


def verify_preformal_report(report_path: Path) -> dict[str, Any]:
    """Strictly reopen and independently validate a passing v1.8 report."""

    lexical = report_path if report_path.is_absolute() else Path.cwd() / report_path
    absolute = Path(os.path.abspath(report_path))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise PreformalEvidenceError("preformal report is missing") from error
    if (
        lexical != absolute
        or absolute != PREFORMAL_REPORT_PATH
        or resolved != absolute
    ):
        raise PreformalEvidenceError("preformal report path is missing or aliased")
    directory = _canonical_existing_directory(absolute.parent, label="evidence directory")
    payload = _read_regular(absolute, label="preformal report")
    report = _load_canonical_object(payload, label="preformal report")
    expected_top_level = {
        "schema",
        "experiment_id",
        "protocol_version",
        "repository_cwd",
        "device",
        "qa_environment",
        "runtime_environment",
        "git_before",
        "git_after",
        "qa_executable_before",
        "qa_executable_after",
        "runtime_executable_before",
        "runtime_executable_after",
        "qa_closure_before",
        "qa_closure_after",
        "runtime_seal",
        "prospective_review",
        "input_files_before",
        "input_files_after",
        "generator_source_before",
        "generator_source_after",
        "identities_stable",
        "commands",
        "all_pass",
    }
    if set(report) != expected_top_level:
        raise PreformalEvidenceError("preformal report has an unexpected top-level field set")
    repository = _canonical_existing_directory(REPO, label="repository")
    qa_environment = _environment_from_identity(report.get("qa_environment"), role="qa")
    runtime_environment = _environment_from_identity(
        report.get("runtime_environment"),
        role="runtime",
    )
    git = _git_identity(environment=qa_environment)
    qa_executable = _executable_identity()
    runtime_before = report.get("runtime_executable_before")
    runtime_after = report.get("runtime_executable_after")
    if (
        not isinstance(runtime_before, dict)
        or runtime_before != runtime_after
        or not isinstance(runtime_before.get("invocation_path"), str)
        or not isinstance(runtime_before.get("version"), str)
    ):
        raise PreformalEvidenceError("preformal runtime executable identity is invalid")
    runtime_executable = _executable_identity(
        cast(str, runtime_before["invocation_path"]),
        implementation="CPython",
        version=cast(str, runtime_before["version"]),
    )
    qa_closure = _validate_qa_closure(report.get("qa_closure_before"))
    live_qa_closure = _capture_qa_closure(
        executable=cast(str, qa_executable["invocation_path"]),
        environment=qa_environment,
    )
    source = _source_identity()
    inputs_before = report.get("input_files_before")
    inputs_after = report.get("input_files_after")
    if (
        not isinstance(inputs_before, dict)
        or set(inputs_before)
        != {
            "development_closure",
            "launch_bootstrap",
            "producer_bootstrap",
            "prospective_review",
            "runtime_seal",
        }
        or inputs_before != inputs_after
    ):
        raise PreformalEvidenceError("preformal input file identities are incomplete or changed")
    for label, identity in inputs_before.items():
        recorded = _validate_file_identity(identity, label=label)
        live = _file_identity(
            Path(cast(str, recorded["path"])),
            label=label,
            expected_nlink=_PREFORMAL_INPUT_NLINKS[label],
        )
        if live != recorded:
            raise PreformalEvidenceError(f"preformal {label} input changed")
    runtime_seal_path = Path(cast(str, inputs_before["runtime_seal"]["path"]))
    development_closure_path = Path(cast(str, inputs_before["development_closure"]["path"]))
    review_path = Path(cast(str, inputs_before["prospective_review"]["path"]))
    if review_path != REVIEW_PATH:
        raise PreformalEvidenceError(f"prospective harness review must be the canonical {REVIEW_RELATIVE_PATH}")
    runtime_seal, live_runtime_environment = _validated_runtime_seal(
        runtime_seal_path,
        runtime_executable=Path(cast(str, runtime_before["invocation_path"])),
    )
    prospective_review = verify_prospective_review(review_path)
    if (
        report.get("schema") != SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("repository_cwd") != str(repository)
        or report.get("device") not in {"cpu", "cuda"}
        or report.get("git_before") != git
        or report.get("git_after") != git
        or report.get("qa_executable_before") != qa_executable
        or report.get("qa_executable_after") != qa_executable
        or runtime_before != runtime_executable
        or report.get("qa_closure_after") != qa_closure
        or qa_closure != live_qa_closure
        or report.get("runtime_seal") != runtime_seal
        or report.get("runtime_environment") != _environment_identity(live_runtime_environment)
        or runtime_environment != live_runtime_environment
        or report.get("prospective_review") != prospective_review
        or report.get("generator_source_before") != source
        or report.get("generator_source_after") != source
        or report.get("identities_stable") is not True
        or report.get("all_pass") is not True
        or git.get("worktree_clean") is not True
    ):
        raise PreformalEvidenceError("preformal report runtime, source, or Git identity differs")
    rows = report.get("commands")
    specifications = required_commands(
        cast(str, qa_executable["invocation_path"]),
        runtime_executable_path=cast(str, runtime_before["invocation_path"]),
        runtime_seal_path=runtime_seal_path,
        development_closure_path=development_closure_path,
        prospective_review_path=review_path,
        device=cast(str, report["device"]),
    )
    if not isinstance(rows, list) or len(rows) != len(specifications):
        raise PreformalEvidenceError("preformal report command set is incomplete")
    expected_logs: set[str] = set()
    seen_names: set[str] = set()
    expected_row_fields = {
        "ordinal",
        "name",
        "role",
        "argv",
        "cwd",
        "environment_sha256",
        "exit_code",
        "passed",
        "stdout",
        "stderr",
    }
    environment_digests = {
        "qa": cast(str, cast(dict[str, object], report["qa_environment"])["sha256"]),
        "runtime": cast(str, cast(dict[str, object], report["runtime_environment"])["sha256"]),
    }
    for ordinal, (row, specification) in enumerate(zip(rows, specifications, strict=True), start=1):
        if not isinstance(row, dict) or set(row) != expected_row_fields:
            raise PreformalEvidenceError(f"preformal command {ordinal} has wrong fields")
        name = row.get("name")
        if (
            name != specification.name
            or name in seen_names
            or row.get("role") != specification.role
            or type(row.get("ordinal")) is not int
            or row.get("ordinal") != ordinal
            or row.get("argv") != list(specification.argv)
            or row.get("cwd") != str(repository)
            or row.get("environment_sha256") != environment_digests[specification.role]
            or type(row.get("exit_code")) is not int
            or row.get("exit_code") != 0
            or row.get("passed") is not True
        ):
            raise PreformalEvidenceError(f"preformal command {ordinal} differs from its fixed contract")
        seen_names.add(specification.name)
        for stream in ("stdout", "stderr"):
            expected_logs.add(
                _verify_log(
                    directory,
                    row.get(stream),
                    ordinal=ordinal,
                    command_name=specification.name,
                    stream=stream,
                )
            )
    if tuple(row["name"] for row in rows) != _COMMAND_NAMES:
        raise PreformalEvidenceError("preformal command names are not unique and ordered")
    _runtime_bootstrap_conformance_from_report(
        directory,
        report,
    )
    actual_evidence = {
        candidate.name for candidate in directory.iterdir() if _is_preformal_evidence_name(candidate.name)
    }
    if actual_evidence != {*expected_logs, REPORT_NAME}:
        raise PreformalEvidenceError("preformal evidence file set has missing or extra members")
    return report


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _captured_payload(prefix: str) -> bytes:
    descriptor = getattr(sys, f"_prospect_wm001_{prefix}_fd", None)
    expected_payload = getattr(sys, f"_prospect_wm001_{prefix}_payload", None)
    expected_identity = getattr(sys, f"_prospect_wm001_{prefix}_identity", None)
    expected_sha256 = getattr(sys, f"_prospect_wm001_{prefix}_sha256", None)
    expected_nlink = {
        "bootstrap": 1,
        "runtime_seal": 2,
    }.get(prefix)
    if (
        expected_nlink is None
        or type(descriptor) is not int
        or not isinstance(expected_payload, bytes)
        or not isinstance(expected_identity, tuple)
        or len(expected_identity) != 9
        or not _is_sha256(expected_sha256)
    ):
        raise PreformalEvidenceError(f"{prefix} descriptor custody is absent")
    try:
        before = os.fstat(descriptor)
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    except OSError as error:
        raise PreformalEvidenceError(f"{prefix} descriptor custody cannot be reopened") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != expected_nlink
        or _descriptor_identity(before) != _descriptor_identity(after)
        or _descriptor_identity(before) != expected_identity
        or payload != expected_payload
        or _sha256(payload) != expected_sha256
    ):
        raise PreformalEvidenceError(f"{prefix} descriptor custody changed")
    return payload


def _verify_live_bootstrap_custody() -> dict[str, Any]:
    from .experiment import _verify_live_bootstrap_custody as verify_live_closure

    runtime_payload = _captured_payload("runtime_seal")
    _captured_payload("bootstrap")
    seal = _load_canonical_object(runtime_payload, label="captured runtime seal")
    if (
        set(seal) != _RUNTIME_SEAL_FIELDS
        or seal.get("schema") != "prospect.wm001.runtime-seal.v1"
        or seal.get("experiment_id") != EXPERIMENT_ID
        or seal.get("protocol_version") != PROTOCOL_VERSION
        or seal.get("assurance") != ASSURANCE
    ):
        raise PreformalEvidenceError("captured runtime seal is malformed")
    try:
        live_custody = verify_live_closure()
    except RuntimeError as error:
        raise PreformalEvidenceError(
            "live runtime closure differs from its pre-import bootstrap seal"
        ) from error
    if live_custody.get("runtime_seal") != seal:
        raise PreformalEvidenceError(
            "recomputed runtime closure returned a different captured seal"
        )
    return seal


def _runtime_development_evidence(path: Path) -> dict[str, object]:
    from .artifact import verify_producer_manifest
    from .binding import verify_development_closure
    from .verify import verify_result

    before = _verify_live_bootstrap_custody()
    closure_path = _canonical_existing_file(path, label="development closure")
    closure = verify_development_closure(closure_path)
    producer_root_raw = closure.get("producer_root")
    if not isinstance(producer_root_raw, str):
        raise PreformalEvidenceError("development closure has no producer root")
    producer_root = _canonical_existing_directory(
        Path(producer_root_raw),
        label="development producer root",
    )
    manifest = verify_producer_manifest(producer_root)
    result = verify_result(producer_root / "result.json", None)
    if (
        manifest.get("lane") != "development"
        or manifest.get("status") != "completed"
        or result.get("lane") != "development"
        or result.get("claim_eligible") is not False
        or closure.get("engineering_verified") is not True
        or closure.get("audit_reproduced") is not True
        or closure.get("performance_values_bound") is not False
    ):
        raise PreformalEvidenceError("development evidence is not a completed non-claim qualification")
    after = _verify_live_bootstrap_custody()
    if before != after:
        raise PreformalEvidenceError("runtime custody changed during development evidence verification")
    closure_identity = _file_identity(
        closure_path,
        label="development closure",
        expected_nlink=_SINGLE_LINK_CUSTODY,
    )
    manifest_identity = _file_identity(
        producer_root / "producer-manifest.json",
        label="producer manifest",
        expected_nlink=_OUTER_FINALIZED_CUSTODY,
    )
    result_identity = _file_identity(
        producer_root / "result.json",
        label="development result",
        expected_nlink=_SINGLE_LINK_CUSTODY,
    )
    return {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "development-evidence",
        "passed": True,
        "development_closure_sha256": closure_identity["sha256"],
        "producer_manifest_sha256": manifest_identity["sha256"],
        "raw_result_sha256": result_identity["sha256"],
    }


def _runtime_bootstrap_inventory_conformance(device: str) -> dict[str, object]:
    before = _verify_live_bootstrap_custody()
    try:
        import gymnasium as gym

        pendulum = gym.make("Pendulum-v1")
        try:
            if getattr(pendulum.spec, "id", None) != "Pendulum-v1":
                raise PreformalEvidenceError(
                    "result-free rehearsal instantiated the wrong Gymnasium environment"
                )
        finally:
            pendulum.close()
    except PreformalEvidenceError:
        raise
    except Exception as error:
        raise PreformalEvidenceError(
            "result-free Gymnasium import/instantiation rehearsal failed"
        ) from error
    after_pendulum = _verify_live_bootstrap_custody()
    if before != after_pendulum:
        raise PreformalEvidenceError(
            "runtime custody changed during result-free Gymnasium rehearsal"
        )

    from . import binding

    environment = binding.require_formal_process_environment()
    binding.verify_installed_source_snapshot()
    roots = binding.package_roots()
    packages = binding.installed_package_rows()
    binding.verify_lockfile_rows(packages)
    root_inventories = [binding.package_root_inventory(root) for root in roots]
    standard_library = binding.standard_library_inventory()
    ownership = binding.package_root_ownership()
    execution, payloads = binding.build_bound_audit_execution(
        device=device,
        packages=packages,
        roots=roots,
        standard_library=standard_library,
        producer_environment=environment,
    )
    restart_report_name = execution.get(
        "restart_runtime_conformance_report_file"
    )
    restart_receipt_name = execution.get(
        "restart_runtime_execution_receipt_file"
    )
    restart_report_payload = (
        payloads.get(cast(str, restart_report_name))
        if isinstance(restart_report_name, str)
        else None
    )
    restart_receipt_payload = (
        payloads.get(cast(str, restart_receipt_name))
        if isinstance(restart_receipt_name, str)
        else None
    )
    if (
        execution.get("repeat_count") != 3
        or execution.get("path_descriptor_equal") is not True
        or execution.get("restart_runtime_repeat_count") != 3
        or execution.get("restart_runtime_path_descriptor_equal")
        is not True
        or execution.get("restart_runtime_support_files")
        != [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ]
        or execution.get("passed") is not True
        or len(payloads) != 11
        or not isinstance(restart_report_payload, bytes)
        or not isinstance(restart_receipt_payload, bytes)
        or _sha256(restart_report_payload)
        != execution.get(
            "restart_runtime_conformance_report_sha256"
        )
        or _sha256(restart_receipt_payload)
        != execution.get(
            "restart_runtime_execution_receipt_sha256"
        )
    ):
        raise PreformalEvidenceError("bootstrap inventory conformance is incomplete")
    after = _verify_live_bootstrap_custody()
    if before != after:
        raise PreformalEvidenceError("runtime custody changed during bootstrap conformance")
    inventory = {
        "packages": packages,
        "package_roots": root_inventories,
        "standard_library": standard_library,
        "package_ownership": ownership,
    }
    return {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "bootstrap-inventory-conformance",
        "device": device,
        "passed": True,
        "inventory_sha256": _sha256(_canonical_json_bytes(inventory)),
        "conformance_sha256": _sha256(_canonical_json_bytes(execution)),
        "restart_runtime_conformance_report_sha256": _sha256(
            restart_report_payload
        ),
        "restart_runtime_execution_receipt_sha256": _sha256(
            restart_receipt_payload
        ),
        "restart_runtime_support_files": [
            "producer_bootstrap.py",
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "restart_runtime_repeat_count": 3,
        "restart_runtime_path_descriptor_equal": True,
        "repeat_count": 3,
        "path_descriptor_equal": True,
    }


def _runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sealed, result-free WM-001 preformal check")
    modes = parser.add_subparsers(dest="mode", required=True)
    development = modes.add_parser("development-evidence", allow_abbrev=False)
    development.add_argument("--development-closure", type=Path, required=True)
    conformance = modes.add_parser("bootstrap-inventory-conformance", allow_abbrev=False)
    conformance.add_argument("--device", choices=("cpu", "cuda"), required=True)
    return parser


def runtime_main(argv: list[str] | None = None) -> int:
    """Entry point reached only through launch-bootstrap descriptor custody."""

    arguments = _runtime_parser().parse_args(argv)
    if arguments.mode == "development-evidence":
        report = _runtime_development_evidence(arguments.development_closure)
    else:
        report = _runtime_bootstrap_inventory_conformance(arguments.device)
    sys.stdout.buffer.write(_canonical_json_bytes(report) + b"\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_subparsers(dest="mode", required=True)
    generate = modes.add_parser("generate-report", allow_abbrev=False)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--runtime-executable", type=Path, required=True)
    generate.add_argument("--runtime-seal", type=Path, required=True)
    generate.add_argument(
        "--development-closure",
        type=Path,
        default=DEVELOPMENT_CLOSURE_PATH,
    )
    generate.add_argument(
        "--prospective-review",
        type=Path,
        default=REVIEW_PATH,
    )
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    review = modes.add_parser("verify-prospective-review", allow_abbrev=False)
    review.add_argument("--review", type=Path, required=True)
    report = modes.add_parser("verify-report", allow_abbrev=False)
    report.add_argument("--report", type=Path, required=True)
    modes.add_parser("emit-qa-closure", allow_abbrev=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "generate-report":
        path = generate_preformal_report(
            arguments.output,
            runtime_executable=arguments.runtime_executable,
            runtime_seal=arguments.runtime_seal,
            development_closure=arguments.development_closure,
            prospective_review=arguments.prospective_review,
            device=arguments.device,
        )
        output = {
            "schema": "prospect.wm001.preformal-test-report-generation.v1",
            "report": str(path),
            "report_sha256": _file_identity(
                path,
                label="preformal report",
                expected_nlink=_SINGLE_LINK_CUSTODY,
            )["sha256"],
            "passed": True,
        }
    elif arguments.mode == "verify-prospective-review":
        review = verify_prospective_review(arguments.review)
        output = {
            "schema": "prospect.wm001.prospective-harness-review-verification.v1",
            "review_sha256": _file_identity(
                arguments.review,
                label="prospective harness review",
                expected_nlink=_SINGLE_LINK_CUSTODY,
            )["sha256"],
            "implementation_manifest_sha256": review["implementation_manifest_sha256"],
            "passed": True,
        }
    elif arguments.mode == "verify-report":
        report = verify_preformal_report(arguments.report)
        output = {
            "schema": "prospect.wm001.preformal-test-report-verification.v1",
            "report_sha256": _file_identity(
                arguments.report,
                label="preformal report",
                expected_nlink=_SINGLE_LINK_CUSTODY,
            )["sha256"],
            "command_count": len(cast(list[object], report["commands"])),
            "passed": True,
        }
    else:
        output = _qa_closure()
    sys.stdout.buffer.write(_canonical_json_bytes(output) + b"\n")
    return 0


__all__ = (
    "CommandSpec",
    "EXPERIMENT_ID",
    "LOG_PREFIX",
    "PREFORMAL_REPORT_NAME",
    "PREFORMAL_REPORT_PATH",
    "PROTOCOL_VERSION",
    "PreformalEvidenceError",
    "REPORT_NAME",
    "REVIEW_PATH",
    "SCHEMA",
    "generate_preformal_report",
    "required_commands",
    "runtime_main",
    "verify_preformal_report",
    "verify_prospective_review",
)


if __name__ == "__main__":
    raise SystemExit(main())
