"""Trusted, immutable preformal test evidence for WM-001 protocol 1.19."""

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
PROTOCOL_VERSION = "1.19.0"
REPORT_NAME = "preformal-test-report-v1.19.0.json"
PREFORMAL_REPORT_NAME = REPORT_NAME
LOG_PREFIX = "preformal-v1.19.0-command-"
_EVIDENCE_PREFIX = "preformal-"
SOURCE_RELATIVE_PATH = "bench/world_model_lifecycle/preformal.py"
REVIEW_RELATIVE_PATH = "docs/wm001-v1190-prospective-harness-review.json"
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
    DEVELOPMENT_RESULTS_ROOT / "development-closure-v1.19.0.json"
)
RUNTIME_SEAL_PATH = (
    DEVELOPMENT_RESULTS_ROOT / "runtime-seal-v1.19.0.json"
)
PREFORMAL_BUNDLE_PATH = (
    DEVELOPMENT_RESULTS_ROOT / "v1.19.0" / "preformal"
)
CLOSURE_ATTEMPT_PATH = (
    REPO
    / "bench"
    / "world_model_lifecycle"
    / "results"
    / "operator-v1.19"
    / "closures"
    / "development-closure-v1.19.0"
)
PREFORMAL_REPORT_PATH = PREFORMAL_BUNDLE_PATH / REPORT_NAME
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
    "runtime-accepted-closure-evidence",
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
    "bench/world_model_lifecycle/rehearsal.py",
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
_FRESH_CLOSURE_REOPEN_SCHEMA = (
    "prospect.wm001.development-closure-fresh-reopen.v1"
)
_FRESH_IDENTITY_CONFORMANCE_SCHEMA = (
    "prospect.wm001.fresh-runtime-identity-conformance.v1"
)
_DEVELOPMENT_MATRIX_CONTRACT_SHA256 = (
    "09a232a4a58c2690665cbef928936b49fbb28d7134405c8eb696a63371591b84"
)
_FRESH_CLOSURE_REOPEN_TIMEOUT_SECONDS = 3_600
_FRESH_CLOSURE_REOPEN_MAX_OUTPUT_BYTES = 64 << 10
_PREFORMAL_INPUT_NLINKS = {
    "closure_attempt_terminal": _OUTER_FINALIZED_CUSTODY,
    "closure_outer_completion": _OUTER_FINALIZED_CUSTODY,
    "development_closure": _SINGLE_LINK_CUSTODY,
    "launch_bootstrap": _SINGLE_LINK_CUSTODY,
    "producer_bootstrap": _SINGLE_LINK_CUSTODY,
    "prospective_review": _SINGLE_LINK_CUSTODY,
    "runtime_seal": _OUTER_FINALIZED_CUSTODY,
}
_PREFORMAL_INPUT_FIELDS = frozenset(_PREFORMAL_INPUT_NLINKS)
_DEVELOPMENT_CLOSURE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "protocol_version",
        "source",
        "producer_root",
        "producer_manifest_member",
        "raw_result_member",
        "result_qualification_member",
        "independent_audit_member",
        "audit_reproduction_member",
        "audit_runtime_manifest_member",
        "audit_invocation_manifest_member",
        "audit_stderr_member",
        "producer_execution",
        "producer_custody",
        "audit_execution",
        "qualification_archive",
        "engineering_verified",
        "audit_reproduced",
        "performance_values_bound",
    }
)
_DEVELOPMENT_EXECUTION_FIELDS = frozenset(
    {
        "git_commit",
        "git_tree",
        "worktree_clean",
        "dependency_lock_sha256",
        "python_executable",
        "python_executable_sha256",
        "python_version",
        "platform",
        "machine",
        "device",
        "python_flags",
        "process_environment",
        "accelerator",
        "thread_count",
        "interop_thread_count",
        "cuda_runtime",
        "cuda_driver",
        "cublas_workspace_config",
        "deterministic_algorithms",
        "runtime_seal_sha256",
        "runtime_seal_descriptor_custody",
        "producer_bootstrap_sha256",
        "bootstrap_descriptor_custody",
        "package_roots",
        "standard_library",
    }
)
_DEVELOPMENT_FIXED_ROLE_MEMBERS = {
    "producer_manifest_member": "producer/producer-manifest.json",
    "raw_result_member": "producer/result.json",
    "result_qualification_member": (
        "evidence/development-result-qualification.json"
    ),
    "independent_audit_member": "evidence/independent-audit.json",
    "audit_reproduction_member": "evidence/audit-reproduction.json",
}
_DEVELOPMENT_PRODUCER_CUSTODY_MEMBERS = {
    "runtime_seal_member": "evidence/producer-runtime-seal.json",
    "producer_bootstrap_member": "evidence/producer-bootstrap.py",
    "launch_bootstrap_member": "evidence/launch-bootstrap.py",
}
_MAX_QUALIFICATION_MEMBERS = 100_000
_MAX_QUALIFICATION_MEMBER_BYTES = 4 << 30
_MAX_QUALIFICATION_TOTAL_MEMBER_BYTES = 32 << 30
_MAX_QUALIFICATION_ARCHIVE_BYTES = 40 << 30


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


def _strict_json_equal(observed: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int/float aliases."""

    try:
        return _canonical_json_bytes(observed) == _canonical_json_bytes(
            expected
        )
    except (TypeError, ValueError):
        return False


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_preformal_staging(final_directory: Path) -> Path:
    """Irreversibly claim one bundle in a hidden sibling staging directory."""

    parent = final_directory.parent
    if not os.path.lexists(parent):
        grandparent = _canonical_existing_directory(
            parent.parent,
            label="preformal version parent",
        )
        if parent.parent != grandparent:
            raise PreformalEvidenceError(
                "preformal version path has a noncanonical parent"
            )
        os.mkdir(parent, 0o700)
        _fsync_directory(grandparent)
    canonical_parent = _canonical_existing_directory(
        parent,
        label="preformal version directory",
    )
    if parent != canonical_parent:
        raise PreformalEvidenceError(
            "preformal version directory is aliased"
        )
    staging = canonical_parent / f".{final_directory.name}.staging"
    with os.scandir(canonical_parent) as entries:
        existing_members = sorted(entry.name for entry in entries)
    if existing_members:
        raise FileExistsError(
            "preformal version namespace already contains a bundle, hidden "
            "one-shot claim, or unbound member"
        )
    try:
        os.mkdir(staging, 0o700)
    except FileExistsError:
        raise FileExistsError(
            "preformal hidden one-shot claim already exists"
        ) from None
    _fsync_directory(canonical_parent)
    return _canonical_existing_directory(
        staging,
        label="preformal hidden staging claim",
    )


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


def _tracked_implementation_paths(
    *,
    environment: dict[str, str] | None = None,
) -> tuple[str, ...]:
    selected_environment = (
        _sanitized_environment()
        if environment is None
        else environment
    )
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
        raise PreformalEvidenceError(
            "prospective review cannot enumerate tracked implementation files"
        )
    paths = tuple(completed.stdout.splitlines())
    if (
        not paths
        or len(paths) != len(set(paths))
        or paths != tuple(sorted(paths))
        or any(
            not path
            or Path(path).is_absolute()
            or Path(path).as_posix() != path
            or "." in Path(path).parts
            or ".." in Path(path).parts
            for path in paths
        )
    ):
        raise PreformalEvidenceError(
            "tracked implementation manifest is empty, aliased, or unordered"
        )
    return paths


def _test_files(pattern: str, *, label: str) -> tuple[str, ...]:
    tracked = tuple(
        path
        for path in _tracked_implementation_paths()
        if Path(path).parent.as_posix() == "tests"
        and Path(path).match(f"tests/{pattern}")
    )
    discovered = tuple(
        sorted(
            path.relative_to(REPO).as_posix()
            for path in (REPO / "tests").glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    )
    if not tracked:
        raise PreformalEvidenceError(f"required {label} test set is empty")
    if discovered != tracked:
        raise PreformalEvidenceError(
            f"required {label} test set contains untracked, ignored, "
            "missing, or aliased members"
        )
    return tracked


def _implementation_files(*, environment: dict[str, str] | None = None) -> list[dict[str, object]]:
    selected_environment = _sanitized_environment() if environment is None else environment
    tracked_python = [
        REPO / relative
        for relative in _tracked_implementation_paths(
            environment=selected_environment,
        )
        if relative.endswith(".py")
    ]
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
        REPO / "docs/wm001-v1190-confirmation-plan.md",
        REPO / "docs/wm001-v1190-operator-runbook.md",
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
    closure_attempt_path: Path = CLOSURE_ATTEMPT_PATH,
    prospective_review_path: Path = REVIEW_PATH,
    device: str = "cpu",
) -> tuple[CommandSpec, ...]:
    """Return the fixed, ordered v1.19 preformal command contract."""

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
        RUNTIME_SEAL_PATH
        if runtime_seal_path is None
        else _canonical_existing_file(runtime_seal_path, label="runtime seal")
    )
    development_closure = (
        development_closure_path if development_closure_path.is_absolute() else Path.cwd() / development_closure_path
    )
    closure_attempt = (
        closure_attempt_path
        if closure_attempt_path.is_absolute()
        else Path.cwd() / closure_attempt_path
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
            "runtime-accepted-closure-evidence",
            "runtime",
            (
                *runtime_prefix,
                "accepted-closure-evidence",
                "--development-closure",
                str(development_closure),
                "--closure-attempt",
                str(closure_attempt),
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
    closure_attempt_path: Path,
    prospective_review_path: Path,
) -> dict[str, dict[str, object]]:
    from .operator import outer_completion_marker

    closure_terminal = closure_attempt_path / "operator-attempt.json"
    inputs = {
        "closure_attempt_terminal": (
            closure_terminal,
            "development closure attempt terminal",
        ),
        "closure_outer_completion": (
            outer_completion_marker(closure_terminal),
            "development closure outer completion",
        ),
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
    closure_attempt: Path = CLOSURE_ATTEMPT_PATH,
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
            "preformal report must use the sole canonical protocol-1.19 "
            f"path {PREFORMAL_REPORT_PATH}"
        )
    final_directory = output.parent
    runtime_executable_path = Path(str(_executable_identity(runtime_executable)["invocation_path"]))
    runtime_seal_path = _canonical_existing_file(runtime_seal, label="runtime seal")
    development_closure_path = _canonical_existing_file(
        development_closure,
        label="development closure",
    )
    closure_attempt_path = _canonical_existing_directory(
        closure_attempt,
        label="development closure attempt",
    )
    if runtime_seal_path != RUNTIME_SEAL_PATH:
        raise PreformalEvidenceError(
            "runtime seal must use its canonical v1.19 path"
        )
    if development_closure_path != DEVELOPMENT_CLOSURE_PATH:
        raise PreformalEvidenceError(
            "development closure must use its canonical v1.19 path"
        )
    if closure_attempt_path != CLOSURE_ATTEMPT_PATH:
        raise PreformalEvidenceError(
            "development closure attempt must use its canonical path"
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
        closure_attempt_path=closure_attempt_path,
        prospective_review_path=prospective_review_path,
    )
    specifications = required_commands(
        str(qa_executable_before["invocation_path"]),
        runtime_executable_path=str(runtime_executable_path),
        runtime_seal_path=runtime_seal_path,
        development_closure_path=development_closure_path,
        closure_attempt_path=closure_attempt_path,
        prospective_review_path=prospective_review_path,
        device=device,
    )
    directory = _claim_preformal_staging(final_directory)
    executions: list[tuple[int, CommandSpec, int, bytes, bytes]] = []
    for ordinal, specification in enumerate(specifications, start=1):
        selected_environment = qa_environment if specification.role == "qa" else runtime_environment
        exit_code, stdout, stderr = _run_command(
            specification,
            environment=selected_environment,
        )
        executions.append(
            (ordinal, specification, exit_code, stdout, stderr)
        )
    rows: list[dict[str, object]] = []
    for ordinal, specification, exit_code, stdout, stderr in executions:
        selected_identity = (
            qa_environment_identity
            if specification.role == "qa"
            else runtime_environment_identity
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
        closure_attempt_path=closure_attempt_path,
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
    semantic_failures = _semantic_failure_checks(
        directory,
        {
            "commands": rows,
            "device": device,
            "input_files_before": inputs_before,
        },
    )
    all_pass = (
        all(row["passed"] is True for row in rows)
        and git_after.get("worktree_clean") is True
        and identities_stable
        and not semantic_failures
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
    _atomic_write_exclusive(
        directory / REPORT_NAME,
        _canonical_json_bytes(report) + b"\n",
    )
    from .operator import _rename_noreplace

    try:
        _rename_noreplace(directory, final_directory)
    except Exception as error:
        raise PreformalEvidenceError(
            "atomic preformal bundle publication failed"
        ) from error
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
    if (
        stream == "stderr"
        and (
            size != 0
            or digest != _sha256(b"")
            or payload != b""
        )
    ):
        raise PreformalEvidenceError(
            f"{command_name} stderr is not exactly empty"
        )
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


def _is_git_object_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_safe_archive_member(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        return False
    path = Path(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and len(path.parts) >= 2
        and "." not in path.parts
        and ".." not in path.parts
        and path.parts[0] in {"evidence", "producer"}
    )


def _validate_support_rows(value: object) -> bool:
    if not isinstance(value, list):
        return False
    paths: list[str] = []
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or not isinstance(row.get("path"), str)
            or not cast(str, row["path"])
            or "\x00" in cast(str, row["path"])
            or "\\" in cast(str, row["path"])
            or Path(cast(str, row["path"])).is_absolute()
            or Path(cast(str, row["path"])).as_posix() != row["path"]
            or "." in Path(cast(str, row["path"])).parts
            or ".." in Path(cast(str, row["path"])).parts
            or type(row.get("bytes")) is not int
            or cast(int, row["bytes"]) < 0
            or not _is_sha256(row.get("sha256"))
        ):
            return False
        paths.append(cast(str, row["path"]))
    return paths == sorted(paths) and len(paths) == len(set(paths))


def _verified_closure_member_digests(
    development_closure: Path,
) -> tuple[dict[str, object], str, str]:
    """Parse one sealed closure without re-entering runtime verification.

    Command 9 already performed the expensive runtime/live-inventory verification.
    This QA-side check independently parses the exact immutable closure bytes and
    validates every structural and archive-member identity needed to bind that
    runtime receipt.  In particular, it never compares the runtime package
    inventory to the intentionally different QA environment.
    """

    closure_path = _canonical_existing_file(
        development_closure,
        label="development closure with archived member identities",
    )
    closure = _load_canonical_object(
        _read_regular(
            closure_path,
            label="development closure with archived member identities",
            expected_nlink=_SINGLE_LINK_CUSTODY,
        ),
        label="development closure with archived member identities",
    )
    if (
        set(closure) != _DEVELOPMENT_CLOSURE_FIELDS
        or closure.get("schema")
        != "prospect.wm001.development-closure.v2"
        or closure.get("experiment_id") != EXPERIMENT_ID
        or closure.get("protocol_version") != PROTOCOL_VERSION
        or closure.get("engineering_verified") is not True
        or closure.get("audit_reproduced") is not True
        or closure.get("performance_values_bound") is not False
    ):
        raise PreformalEvidenceError(
            "development closure identity or status is invalid"
        )

    source = closure.get("source")
    execution = closure.get("producer_execution")
    producer_custody = closure.get("producer_custody")
    audit_execution = closure.get("audit_execution")
    archive = closure.get("qualification_archive")
    producer_root_raw = closure.get("producer_root")
    source_sha256_fields = {
        "dependency_lock_sha256",
        "producer_bootstrap_sha256",
        "launch_bootstrap_sha256",
        "runner_source_sha256",
        "auditor_source_sha256",
    }
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "git_commit",
            "git_tree",
            "worktree_clean",
            *source_sha256_fields,
        }
        or not _is_git_object_id(source.get("git_commit"))
        or not _is_git_object_id(source.get("git_tree"))
        or source.get("worktree_clean") is not True
        or any(not _is_sha256(source.get(field)) for field in source_sha256_fields)
        or not isinstance(execution, dict)
        or set(execution) != _DEVELOPMENT_EXECUTION_FIELDS
        or not isinstance(producer_custody, dict)
        or set(producer_custody)
        != {
            "runtime_seal_member",
            "runtime_seal_sha256",
            "producer_bootstrap_member",
            "producer_bootstrap_sha256",
            "launch_bootstrap_member",
            "launch_bootstrap_sha256",
            "package_ownership",
        }
        or not isinstance(producer_custody.get("package_ownership"), dict)
        or not isinstance(audit_execution, dict)
        or set(audit_execution)
        != {
            "receipt_sha256",
            "runtime_manifest_sha256",
            "invocation_manifest_sha256",
            "stderr_sha256",
            "bootstrap_sha256",
            "runner_source_sha256",
            "auditor_source_sha256",
            "support_files",
            "source_mode",
        }
        or audit_execution.get("source_mode") != "descriptor"
        or not _validate_support_rows(audit_execution.get("support_files"))
        or not isinstance(archive, dict)
        or set(archive)
        != {
            "format",
            "file",
            "canonical_path",
            "bytes",
            "sha256",
            "members",
        }
        or not isinstance(producer_root_raw, str)
        or not producer_root_raw
        or "\x00" in producer_root_raw
    ):
        raise PreformalEvidenceError(
            "development closure structural identity is malformed"
        )

    producer_root = Path(producer_root_raw)
    if (
        not producer_root.is_absolute()
        or Path(os.path.abspath(producer_root)) != producer_root
        or producer_root.resolve(strict=False) != producer_root
    ):
        raise PreformalEvidenceError(
            "development closure producer root is unsafe"
        )

    execution_sha256_fields = {
        "dependency_lock_sha256",
        "python_executable_sha256",
        "runtime_seal_sha256",
        "producer_bootstrap_sha256",
    }
    python_executable = execution.get("python_executable")
    process_environment = execution.get("process_environment")
    package_roots = execution.get("package_roots")
    if (
        not _is_git_object_id(execution.get("git_commit"))
        or not _is_git_object_id(execution.get("git_tree"))
        or execution.get("worktree_clean") is not True
        or any(
            not _is_sha256(execution.get(field))
            for field in execution_sha256_fields
        )
        or not isinstance(python_executable, str)
        or not python_executable
        or "\x00" in python_executable
        or not Path(python_executable).is_absolute()
        or Path(os.path.abspath(python_executable)) != Path(python_executable)
        or not isinstance(execution.get("python_version"), str)
        or not execution.get("python_version")
        or not isinstance(execution.get("platform"), str)
        or not execution.get("platform")
        or not isinstance(execution.get("machine"), str)
        or not execution.get("machine")
        or execution.get("device") not in {"cpu", "cuda"}
        or execution.get("python_flags") != _RUNTIME_FLAGS
        or not isinstance(process_environment, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in process_environment.items()
        )
        or execution.get("deterministic_algorithms") is not True
        or execution.get("runtime_seal_descriptor_custody") is not True
        or execution.get("bootstrap_descriptor_custody") is not True
        or type(execution.get("thread_count")) is not int
        or cast(int, execution["thread_count"]) <= 0
        or type(execution.get("interop_thread_count")) is not int
        or cast(int, execution["interop_thread_count"]) <= 0
        or not isinstance(package_roots, list)
        or len(package_roots) != 1
        or not isinstance(package_roots[0], dict)
        or not isinstance(execution.get("standard_library"), dict)
    ):
        raise PreformalEvidenceError(
            "development closure producer execution is malformed"
        )
    _environment_from_identity(
        _environment_identity(cast(dict[str, str], process_environment)),
        role="runtime",
    )
    device = cast(str, execution["device"])
    accelerator = execution.get("accelerator")
    cuda_runtime = execution.get("cuda_runtime")
    cuda_driver = execution.get("cuda_driver")
    workspace = execution.get("cublas_workspace_config")
    if (
        (
            device == "cpu"
            and (
                accelerator is not None
                or not (
                    cuda_runtime is None
                    or isinstance(cuda_runtime, str)
                )
                or cuda_driver is not None
                or workspace is not None
            )
        )
        or (
            device == "cuda"
            and (
                not isinstance(accelerator, str)
                or not accelerator
                or not isinstance(cuda_runtime, str)
                or not cuda_runtime
                or not isinstance(cuda_driver, str)
                or not cuda_driver
                or workspace != ":4096:8"
            )
        )
    ):
        raise PreformalEvidenceError(
            "development closure accelerator identity is malformed"
        )

    if any(
        closure.get(field) != expected
        for field, expected in _DEVELOPMENT_FIXED_ROLE_MEMBERS.items()
    ):
        raise PreformalEvidenceError(
            "development closure fixed archive roles changed"
        )
    sidecar_fields = (
        "audit_runtime_manifest_member",
        "audit_invocation_manifest_member",
        "audit_stderr_member",
    )
    sidecar_members = [closure.get(field) for field in sidecar_fields]
    if (
        any(
            not isinstance(member, str)
            or not member.startswith("evidence/")
            or not _is_safe_archive_member(member)
            or Path(member).name != member.removeprefix("evidence/")
            for member in sidecar_members
        )
        or len(set(cast(list[str], sidecar_members))) != len(sidecar_members)
    ):
        raise PreformalEvidenceError(
            "development closure audit sidecar roles are unsafe"
        )
    if any(
        producer_custody.get(field) != expected
        for field, expected in _DEVELOPMENT_PRODUCER_CUSTODY_MEMBERS.items()
    ):
        raise PreformalEvidenceError(
            "development closure producer-custody roles changed"
        )
    producer_custody_sha256_fields = {
        "runtime_seal_sha256",
        "producer_bootstrap_sha256",
        "launch_bootstrap_sha256",
    }
    audit_sha256_fields = {
        "receipt_sha256",
        "runtime_manifest_sha256",
        "invocation_manifest_sha256",
        "stderr_sha256",
        "bootstrap_sha256",
        "runner_source_sha256",
        "auditor_source_sha256",
    }
    if (
        any(
            not _is_sha256(producer_custody.get(field))
            for field in producer_custody_sha256_fields
        )
        or any(
            not _is_sha256(audit_execution.get(field))
            for field in audit_sha256_fields
        )
    ):
        raise PreformalEvidenceError(
            "development closure custody digests are malformed"
        )

    archive_sha256 = archive.get("sha256")
    archive_file = archive.get("file")
    archive_relative = archive.get("canonical_path")
    members = archive.get("members")
    expected_archive_prefix = (
        DEVELOPMENT_RESULTS_ROOT.relative_to(REPO).as_posix()
    )
    if (
        archive.get("format") != "ustar-uncompressed-v1"
        or not _is_sha256(archive_sha256)
        or not isinstance(archive_file, str)
        or archive_file
        != f"development-qualification-{cast(str, archive_sha256)[:16]}.tar"
        or Path(archive_file).name != archive_file
        or not isinstance(archive_relative, str)
        or archive_relative != f"{expected_archive_prefix}/{archive_file}"
        or Path(archive_relative).is_absolute()
        or Path(archive_relative).as_posix() != archive_relative
        or "." in Path(archive_relative).parts
        or ".." in Path(archive_relative).parts
        or type(archive.get("bytes")) is not int
        or not 0 <= cast(int, archive["bytes"]) <= _MAX_QUALIFICATION_ARCHIVE_BYTES
        or not isinstance(members, list)
        or not 1 <= len(members) <= _MAX_QUALIFICATION_MEMBERS
    ):
        raise PreformalEvidenceError(
            "development qualification archive identity is malformed"
        )

    member_paths: list[str] = []
    member_digests: dict[str, str] = {}
    total_member_bytes = 0
    for member in members:
        if (
            not isinstance(member, dict)
            or set(member) != {"path", "bytes", "sha256"}
            or not _is_safe_archive_member(member.get("path"))
            or type(member.get("bytes")) is not int
            or not 0
            <= cast(int, member["bytes"])
            <= _MAX_QUALIFICATION_MEMBER_BYTES
            or not _is_sha256(member.get("sha256"))
        ):
            raise PreformalEvidenceError(
                "development qualification archive member is malformed"
            )
        member_path = cast(str, member["path"])
        member_paths.append(member_path)
        member_digests[member_path] = cast(str, member["sha256"])
        total_member_bytes += cast(int, member["bytes"])
    if (
        member_paths != sorted(member_paths)
        or len(member_paths) != len(set(member_paths))
        or total_member_bytes > _MAX_QUALIFICATION_TOTAL_MEMBER_BYTES
    ):
        raise PreformalEvidenceError(
            "development qualification archive members are unordered or duplicated"
        )

    role_members = {
        **{
            field: cast(str, closure[field])
            for field in _DEVELOPMENT_FIXED_ROLE_MEMBERS
        },
        **{
            field: cast(str, closure[field])
            for field in sidecar_fields
        },
        **{
            field: cast(str, producer_custody[field])
            for field in _DEVELOPMENT_PRODUCER_CUSTODY_MEMBERS
        },
    }
    if (
        len(set(role_members.values())) != len(role_members)
        or any(
            member_paths.count(member) != 1
            for member in role_members.values()
        )
    ):
        raise PreformalEvidenceError(
            "development closure role has no unique archive member"
        )

    custody_member_digest_fields = {
        "runtime_seal_member": "runtime_seal_sha256",
        "producer_bootstrap_member": "producer_bootstrap_sha256",
        "launch_bootstrap_member": "launch_bootstrap_sha256",
    }
    audit_member_digest_fields = {
        "audit_reproduction_member": "receipt_sha256",
        "audit_runtime_manifest_member": "runtime_manifest_sha256",
        "audit_invocation_manifest_member": "invocation_manifest_sha256",
        "audit_stderr_member": "stderr_sha256",
    }
    if any(
        member_digests[cast(str, producer_custody[member_field])]
        != producer_custody[digest_field]
        for member_field, digest_field in custody_member_digest_fields.items()
    ) or any(
        member_digests[cast(str, closure[member_field])]
        != audit_execution[digest_field]
        for member_field, digest_field in audit_member_digest_fields.items()
    ):
        raise PreformalEvidenceError(
            "development closure role digest differs from its archive member"
        )

    return (
        closure,
        member_digests[cast(str, closure["producer_manifest_member"])],
        member_digests[cast(str, closure["raw_result_member"])],
    )


def _accepted_closure_evidence_from_report(
    directory: Path,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Reopen the sealed post-finalization closure authorization result."""

    commands = report.get("commands")
    matches = (
        [
            row
            for row in commands
            if isinstance(row, Mapping)
            and row.get("name")
            == "runtime-accepted-closure-evidence"
        ]
        if isinstance(commands, list)
        else []
    )
    if len(matches) != 1:
        raise PreformalEvidenceError(
            "preformal report lacks one accepted-closure command"
        )
    row = matches[0]
    stdout_reference = row.get("stdout")
    stderr_reference = row.get("stderr")
    if (
        not isinstance(stdout_reference, Mapping)
        or set(stdout_reference) != {"file", "bytes", "sha256"}
        or type(stdout_reference.get("bytes")) is not int
        or cast(int, stdout_reference["bytes"]) < 1
        or not _is_sha256(stdout_reference.get("sha256"))
        or not isinstance(stderr_reference, Mapping)
        or set(stderr_reference) != {"file", "bytes", "sha256"}
        or type(stderr_reference.get("bytes")) is not int
        or stderr_reference.get("bytes") != 0
        or stderr_reference.get("sha256") != _sha256(b"")
    ):
        raise PreformalEvidenceError(
            "accepted-closure command stream identities are malformed"
        )
    path = _safe_sibling(
        directory,
        stdout_reference.get("file"),
        label="accepted-closure stdout",
    )
    payload = _read_regular(
        path,
        label="accepted-closure stdout",
    )
    if (
        len(payload) != stdout_reference.get("bytes")
        or _sha256(payload) != stdout_reference.get("sha256")
    ):
        raise PreformalEvidenceError(
            "accepted-closure stdout identity changed"
        )
    value = _load_canonical_object(
        payload,
        label="accepted-closure stdout",
    )
    inputs = report.get("input_files_before")
    if not isinstance(inputs, Mapping):
        raise PreformalEvidenceError(
            "accepted-closure report has no input identities"
        )
    closure_input = inputs.get("development_closure")
    if (
        not isinstance(closure_input, Mapping)
        or not isinstance(closure_input.get("path"), str)
    ):
        raise PreformalEvidenceError(
            "accepted-closure report has no closure path"
        )
    _, producer_manifest_sha256, raw_result_sha256 = (
        _verified_closure_member_digests(
            Path(cast(str, closure_input["path"])),
        )
    )
    expected_fields = {
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
        set(value) != expected_fields
        or value.get("schema")
        != "prospect.wm001.preformal-runtime-check.v1"
        or value.get("mode") != "accepted-closure-evidence"
        or value.get("passed") is not True
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "development_closure_sha256",
                "producer_manifest_sha256",
                "raw_result_sha256",
                "closure_attempt_manifest_sha256",
                "closure_outer_completion_sha256",
            )
        )
        or value.get("development_closure_sha256")
        != closure_input.get("sha256")
        or value.get("producer_manifest_sha256")
        != producer_manifest_sha256
        or value.get("raw_result_sha256") != raw_result_sha256
        or value.get("closure_attempt_manifest_sha256")
        != cast(
            Mapping[str, object],
            inputs["closure_attempt_terminal"],
        ).get("sha256")
        or value.get("closure_outer_completion_sha256")
        != cast(
            Mapping[str, object],
            inputs["closure_outer_completion"],
        ).get("sha256")
    ):
        raise PreformalEvidenceError(
            "accepted-closure stdout is not one complete sealed pass"
        )
    return value


def _validate_recorded_result_free_inventory(
    value: object,
) -> dict[str, object]:
    """Validate a recorded runtime inventory without reading the QA runtime."""

    if not isinstance(value, dict) or set(value) != {
        "packages",
        "package_roots",
        "standard_library",
        "package_ownership",
    }:
        raise PreformalEvidenceError(
            "bootstrap conformance inventory has wrong fields"
        )
    packages = value.get("packages")
    package_roots = value.get("package_roots")
    standard_library = value.get("standard_library")
    package_ownership = value.get("package_ownership")
    package_fields = {
        "name",
        "version",
        "distribution_sha256",
        "declared_file_count",
        "editable",
    }
    if not isinstance(packages, list) or not packages:
        raise PreformalEvidenceError(
            "bootstrap conformance package inventory is absent"
        )
    names: list[str] = []
    for row in packages:
        if (
            not isinstance(row, dict)
            or set(row) != package_fields
            or not isinstance(row.get("name"), str)
            or re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                cast(str, row.get("name")),
            )
            is None
            or not isinstance(row.get("version"), str)
            or not row.get("version")
            or "\0" in cast(str, row.get("version"))
            or not _is_sha256(row.get("distribution_sha256"))
            or type(row.get("declared_file_count")) is not int
            or cast(int, row.get("declared_file_count")) <= 0
            or row.get("editable") is not False
        ):
            raise PreformalEvidenceError(
                "bootstrap conformance package row is malformed"
            )
        names.append(cast(str, row["name"]))
    if (
        names[0] != "python"
        or names[1:] != sorted(names[1:])
        or len(names) != len(set(names))
        or re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+",
            cast(str, packages[0]["version"]),
        )
        is None
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance packages are duplicated or unordered"
        )

    root_fields = {
        "semantics_id",
        "path",
        "file_count",
        "directory_count",
        "total_bytes",
        "inventory_sha256",
    }

    def valid_root(row: object, *, semantics_id: str) -> bool:
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
            and cast(int, row["file_count"]) > 0
            and type(row.get("directory_count")) is int
            and cast(int, row["directory_count"]) >= 0
            and type(row.get("total_bytes")) is int
            and cast(int, row["total_bytes"]) >= 0
            and _is_sha256(row.get("inventory_sha256"))
        )

    if (
        not isinstance(package_roots, list)
        or len(package_roots) != 1
        or not valid_root(
            package_roots[0],
            semantics_id="prospect.wm001.package-root.v2",
        )
        or not valid_root(
            standard_library,
            semantics_id="prospect.wm001.standard-library.v2",
        )
        or not isinstance(package_ownership, dict)
        or set(package_ownership)
        != {
            "semantics_id",
            "root",
            "file_count",
            "directory_count",
            "shared_file_count",
            "identity_sha256",
        }
        or package_ownership.get("semantics_id")
        != "prospect.wm001.package-ownership.v1"
        or package_ownership.get("root")
        != cast(dict[str, object], package_roots[0]).get("path")
        or package_ownership.get("file_count")
        != cast(dict[str, object], package_roots[0]).get("file_count")
        or package_ownership.get("directory_count")
        != cast(dict[str, object], package_roots[0]).get(
            "directory_count"
        )
        or type(package_ownership.get("shared_file_count")) is not int
        or cast(int, package_ownership["shared_file_count"]) < 0
        or cast(int, package_ownership["shared_file_count"])
        > cast(int, package_ownership["file_count"])
        or not _is_sha256(package_ownership.get("identity_sha256"))
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance root inventory is malformed"
        )
    return value


def _validate_recorded_fresh_identity_conformance(
    value: object,
) -> dict[str, object]:
    """Validate a retained fresh-child receipt without starting a child."""

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
    requesting_process_id = (
        value.get("requesting_process_id")
        if isinstance(value, dict)
        else None
    )
    verifier_process_id = (
        value.get("verifier_process_id")
        if isinstance(value, dict)
        else None
    )
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema") != _FRESH_IDENTITY_CONFORMANCE_SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("protocol_version") != PROTOCOL_VERSION
        or value.get("mode") != "fresh-identity-conformance"
        or not _is_sha256(value.get("challenge"))
        or type(requesting_process_id) is not int
        or cast(int, requesting_process_id) <= 0
        or type(verifier_process_id) is not int
        or cast(int, verifier_process_id) <= 0
        or requesting_process_id == verifier_process_id
        or value.get("matrix_contract_sha256")
        != _DEVELOPMENT_MATRIX_CONTRACT_SHA256
        or value.get("passed") is not True
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance fresh identity report is malformed"
        )
    return value


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
    row = matches[0]
    reference = row.get("stdout")
    stderr_reference = row.get("stderr")
    if (
        row.get("exit_code") != 0
        or row.get("passed") is not True
        or not isinstance(reference, Mapping)
        or set(reference) != {"file", "bytes", "sha256"}
        or type(reference.get("bytes")) is not int
        or cast(int, reference.get("bytes")) < 1
        or not _is_sha256(reference.get("sha256"))
        or reference.get("file")
        != _log_filename(
            10,
            "runtime-bootstrap-inventory-conformance",
            "stdout",
            cast(str, reference.get("sha256")),
        )
        or not isinstance(stderr_reference, Mapping)
        or set(stderr_reference) != {"file", "bytes", "sha256"}
        or type(stderr_reference.get("bytes")) is not int
        or stderr_reference.get("bytes") != 0
        or stderr_reference.get("sha256") != _sha256(b"")
        or stderr_reference.get("file")
        != _log_filename(
            10,
            "runtime-bootstrap-inventory-conformance",
            "stderr",
            _sha256(b""),
        )
    ):
        raise PreformalEvidenceError(
            "bootstrap conformance stream references are malformed"
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
    stderr_path = _safe_sibling(
        directory,
        stderr_reference.get("file"),
        label="bootstrap conformance stderr",
    )
    stderr_payload = _read_regular(
        stderr_path,
        label="bootstrap conformance stderr",
    )
    if stderr_payload:
        raise PreformalEvidenceError(
            "bootstrap conformance stderr is not empty"
        )
    value = _load_canonical_object(
        payload,
        label="bootstrap conformance stdout",
    )
    inventory = _validate_recorded_result_free_inventory(
        value.get("inventory")
    )
    fresh_identity = _validate_recorded_fresh_identity_conformance(
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
        or value.get("mode")
        != "bootstrap-inventory-conformance"
        or value.get("device") != report.get("device")
        or value.get("passed") is not True
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "inventory_sha256",
                "conformance_sha256",
                "fresh_runtime_identity_conformance_sha256",
                "restart_runtime_conformance_report_sha256",
                "restart_runtime_execution_receipt_sha256",
            )
        )
        or value.get("inventory_sha256")
        != _sha256(_canonical_json_bytes(inventory))
        or value.get("fresh_runtime_identity_conformance_sha256")
        != _sha256(_canonical_json_bytes(fresh_identity))
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


def _command_stderr_failure_checks(
    directory: Path,
    commands: object,
) -> list[str]:
    """Return stable identifiers for command stderr authorization failures."""

    if not isinstance(commands, list):
        return []
    failures: list[str] = []
    for ordinal, row in enumerate(commands, start=1):
        if not isinstance(row, Mapping) or not isinstance(
            row.get("name"),
            str,
        ):
            continue
        try:
            _verify_log(
                directory,
                row.get("stderr"),
                ordinal=ordinal,
                command_name=cast(str, row["name"]),
                stream="stderr",
            )
        except PreformalEvidenceError:
            failures.append(f"command_{ordinal:02d}_stderr_not_empty")
    return failures


def _semantic_failure_checks(
    directory: Path,
    report: Mapping[str, object],
) -> list[str]:
    """Return exact semantic failures beyond command exit statuses."""

    commands = report.get("commands")
    checks = (
        (
            "runtime-accepted-closure-evidence",
            "accepted_closure_semantics_failed",
            _accepted_closure_evidence_from_report,
        ),
        (
            "runtime-bootstrap-inventory-conformance",
            "runtime_conformance_semantics_failed",
            _runtime_bootstrap_conformance_from_report,
        ),
    )
    failures = _command_stderr_failure_checks(directory, commands)
    for command_name, failure_name, verifier in checks:
        matches = (
            [
                row
                for row in commands
                if isinstance(row, Mapping)
                and row.get("name") == command_name
            ]
            if isinstance(commands, list)
            else []
        )
        if len(matches) != 1:
            failures.append(failure_name)
            continue
        if matches[0].get("passed") is not True:
            continue
        try:
            verifier(directory, report)
        except Exception:
            failures.append(failure_name)
    return failures


def _verify_preformal_report(
    report_path: Path,
    *,
    require_live_qa_identity: bool,
) -> dict[str, Any]:
    """Validate a report, optionally reopening its intentionally larger QA closure."""

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
    qa_before = report.get("qa_executable_before")
    qa_after = report.get("qa_executable_after")
    if (
        not isinstance(qa_before, dict)
        or not isinstance(qa_after, dict)
        or not _strict_json_equal(qa_before, qa_after)
        or not isinstance(qa_before.get("invocation_path"), str)
        or not isinstance(qa_before.get("implementation"), str)
        or not isinstance(qa_before.get("version"), str)
    ):
        raise PreformalEvidenceError(
            "preformal QA executable identity is invalid"
        )
    qa_executable = _executable_identity(
        cast(str, qa_before["invocation_path"]),
        implementation=cast(str, qa_before["implementation"]),
        version=cast(str, qa_before["version"]),
    )
    live_qa_executable = (
        _executable_identity() if require_live_qa_identity else None
    )
    runtime_before = report.get("runtime_executable_before")
    runtime_after = report.get("runtime_executable_after")
    if (
        not isinstance(runtime_before, dict)
        or not isinstance(runtime_after, dict)
        or not _strict_json_equal(runtime_before, runtime_after)
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
        or set(inputs_before) != _PREFORMAL_INPUT_FIELDS
        or not isinstance(inputs_after, dict)
        or set(inputs_after) != _PREFORMAL_INPUT_FIELDS
    ):
        raise PreformalEvidenceError("preformal input file identities are incomplete or changed")
    for label, identity in inputs_before.items():
        recorded = _validate_file_identity(identity, label=label)
        recorded_after = _validate_file_identity(
            inputs_after[label],
            label=f"{label} after",
        )
        if not _strict_json_equal(recorded, recorded_after):
            raise PreformalEvidenceError(
                "preformal input file identities are incomplete or changed"
            )
        live = _file_identity(
            Path(cast(str, recorded["path"])),
            label=label,
            expected_nlink=_PREFORMAL_INPUT_NLINKS[label],
        )
        if live != recorded:
            raise PreformalEvidenceError(f"preformal {label} input changed")
    runtime_seal_path = Path(cast(str, inputs_before["runtime_seal"]["path"]))
    development_closure_path = Path(cast(str, inputs_before["development_closure"]["path"]))
    closure_attempt_terminal_path = Path(
        cast(str, inputs_before["closure_attempt_terminal"]["path"])
    )
    closure_outer_completion_path = Path(
        cast(str, inputs_before["closure_outer_completion"]["path"])
    )
    closure_attempt_path = closure_attempt_terminal_path.parent
    from .operator import outer_completion_marker

    runtime_seal_completion_path = outer_completion_marker(
        runtime_seal_path
    )
    runtime_seal_completion_identity = _file_identity(
        runtime_seal_completion_path,
        label="runtime seal outer completion",
        expected_nlink=_OUTER_FINALIZED_CUSTODY,
    )
    launch_bootstrap_path = Path(
        cast(str, inputs_before["launch_bootstrap"]["path"])
    )
    producer_bootstrap_path = Path(
        cast(str, inputs_before["producer_bootstrap"]["path"])
    )
    try:
        runtime_seal_same_inode = os.path.samefile(
            runtime_seal_path,
            runtime_seal_completion_path,
        )
        closure_attempt_same_inode = os.path.samefile(
            closure_attempt_terminal_path,
            closure_outer_completion_path,
        )
    except OSError as error:
        raise PreformalEvidenceError(
            "preformal outer-completion inputs cannot be compared"
        ) from error
    if (
        runtime_seal_path != RUNTIME_SEAL_PATH
        or development_closure_path != DEVELOPMENT_CLOSURE_PATH
        or launch_bootstrap_path != LAUNCH_BOOTSTRAP_PATH
        or producer_bootstrap_path != PRODUCER_BOOTSTRAP_PATH
        or closure_attempt_terminal_path
        != CLOSURE_ATTEMPT_PATH / "operator-attempt.json"
        or closure_outer_completion_path
        != outer_completion_marker(closure_attempt_terminal_path)
        or not runtime_seal_same_inode
        or not closure_attempt_same_inode
        or runtime_seal_completion_identity["bytes"]
        != inputs_before["runtime_seal"]["bytes"]
        or runtime_seal_completion_identity["sha256"]
        != inputs_before["runtime_seal"]["sha256"]
    ):
        raise PreformalEvidenceError(
            "preformal runtime-seal or closure inputs are not canonical"
        )
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
        or not _strict_json_equal(report.get("git_before"), git)
        or not _strict_json_equal(report.get("git_after"), git)
        or not _strict_json_equal(
            report.get("qa_executable_before"),
            qa_executable,
        )
        or not _strict_json_equal(
            report.get("qa_executable_after"),
            qa_executable,
        )
        or (
            require_live_qa_identity
            and not _strict_json_equal(
                live_qa_executable,
                qa_executable,
            )
        )
        or not _strict_json_equal(runtime_before, runtime_executable)
        or not _strict_json_equal(
            report.get("qa_closure_after"),
            qa_closure,
        )
        or not _strict_json_equal(qa_closure, live_qa_closure)
        or not _strict_json_equal(report.get("runtime_seal"), runtime_seal)
        or not _strict_json_equal(
            report.get("runtime_environment"),
            _environment_identity(live_runtime_environment),
        )
        or not _strict_json_equal(
            runtime_environment,
            live_runtime_environment,
        )
        or not _strict_json_equal(
            report.get("prospective_review"),
            prospective_review,
        )
        or not _strict_json_equal(
            report.get("generator_source_before"),
            source,
        )
        or not _strict_json_equal(
            report.get("generator_source_after"),
            source,
        )
        or report.get("identities_stable") is not True
        or type(report.get("all_pass")) is not bool
        or git.get("worktree_clean") is not True
    ):
        raise PreformalEvidenceError("preformal report runtime, source, or Git identity differs")
    rows = report.get("commands")
    specifications = required_commands(
        cast(str, qa_executable["invocation_path"]),
        runtime_executable_path=cast(str, runtime_before["invocation_path"]),
        runtime_seal_path=runtime_seal_path,
        development_closure_path=development_closure_path,
        closure_attempt_path=closure_attempt_path,
        prospective_review_path=review_path,
        device=cast(str, report["device"]),
    )
    if not isinstance(rows, list) or len(rows) != len(specifications):
        raise PreformalEvidenceError("preformal report command set is incomplete")
    expected_logs: set[str] = set()
    seen_names: set[str] = set()
    failed_commands: list[tuple[int, str, int]] = []
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
            or type(row.get("passed")) is not bool
            or row.get("passed") is not (row.get("exit_code") == 0)
        ):
            raise PreformalEvidenceError(f"preformal command {ordinal} differs from its fixed contract")
        if row["passed"] is False:
            failed_commands.append(
                (
                    ordinal,
                    specification.name,
                    cast(int, row["exit_code"]),
                )
            )
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
    semantic_failures = _semantic_failure_checks(directory, report)
    if semantic_failures:
        raise PreformalEvidenceError(
            "preformal report contains semantic failures: "
            + ", ".join(semantic_failures)
        )
    expected_all_pass = (
        not failed_commands
        and report.get("identities_stable") is True
        and git.get("worktree_clean") is True
    )
    if report.get("all_pass") is not expected_all_pass:
        raise PreformalEvidenceError(
            "preformal report all_pass differs from its command outcomes"
        )
    if failed_commands:
        detail = ", ".join(
            f"{ordinal}:{name}(exit={exit_code})"
            for ordinal, name, exit_code in failed_commands
        )
        raise PreformalEvidenceError(
            f"preformal report contains failed commands: {detail}"
        )
    actual_evidence = {candidate.name for candidate in directory.iterdir()}
    if actual_evidence != {*expected_logs, REPORT_NAME}:
        raise PreformalEvidenceError("preformal evidence file set has missing or extra members")
    return report


def verify_recorded_preformal_report(
    report_path: Path,
) -> dict[str, Any]:
    """Validate a report against its explicit QA path, not the caller's interpreter."""

    return _verify_preformal_report(
        report_path,
        require_live_qa_identity=False,
    )


def verify_preformal_report(report_path: Path) -> dict[str, Any]:
    """Validate a report and prove the caller is its recorded QA environment."""

    return _verify_preformal_report(
        report_path,
        require_live_qa_identity=True,
    )


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


def _runtime_custody_value(payload: bytes) -> tuple[dict[str, Any], int]:
    """Return one protocol-authorized runtime custody object and link count."""

    value = _load_canonical_object(
        payload,
        label="captured runtime custody",
    )
    schema = value.get("schema")
    if schema == "prospect.wm001.runtime-seal.v1":
        if (
            set(value) != _RUNTIME_SEAL_FIELDS
            or value.get("experiment_id") != EXPERIMENT_ID
            or value.get("protocol_version") != PROTOCOL_VERSION
            or not _strict_json_equal(value.get("assurance"), ASSURANCE)
        ):
            raise PreformalEvidenceError(
                "captured runtime seal is malformed"
            )
        return value, _OUTER_FINALIZED_CUSTODY
    if schema == "prospect.world-model-lifecycle.formal-binding.v10":
        protocol = value.get("protocol")
        if (
            value.get("experiment_id") != EXPERIMENT_ID
            or not _strict_json_equal(value.get("assurance"), ASSURANCE)
            or not isinstance(protocol, dict)
            or set(protocol)
            != {
                "version",
                "sha256",
                "raw_result_schema_sha256",
                "binding_schema_sha256",
            }
            or protocol.get("version") != PROTOCOL_VERSION
            or any(
                not _is_sha256(protocol.get(field))
                for field in (
                    "sha256",
                    "raw_result_schema_sha256",
                    "binding_schema_sha256",
                )
            )
        ):
            raise PreformalEvidenceError(
                "captured formal binding custody is malformed"
            )
        return value, _SINGLE_LINK_CUSTODY
    raise PreformalEvidenceError(
        "captured runtime custody has an unsupported schema"
    )


def _captured_payload(prefix: str) -> bytes:
    descriptor = getattr(sys, f"_prospect_wm001_{prefix}_fd", None)
    expected_payload = getattr(sys, f"_prospect_wm001_{prefix}_payload", None)
    expected_identity = getattr(sys, f"_prospect_wm001_{prefix}_identity", None)
    expected_sha256 = getattr(sys, f"_prospect_wm001_{prefix}_sha256", None)
    if (
        prefix not in {"bootstrap", "runtime_seal"}
        or type(descriptor) is not int
        or not isinstance(expected_payload, bytes)
        or not isinstance(expected_identity, tuple)
        or len(expected_identity) != 9
        or not _is_sha256(expected_sha256)
    ):
        raise PreformalEvidenceError(f"{prefix} descriptor custody is absent")
    expected_nlink = _SINGLE_LINK_CUSTODY
    if prefix == "runtime_seal":
        _, expected_nlink = _runtime_custody_value(expected_payload)
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
    seal, _ = _runtime_custody_value(runtime_payload)
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


def _runtime_accepted_closure_evidence(
    development_closure: Path,
    closure_attempt: Path,
) -> dict[str, object]:
    """Verify the closure marker and its accepted outer-finalized attempt."""

    from .operator import (
        CLOSURE_ATTEMPT_PATH as OPERATOR_CLOSURE_ATTEMPT_PATH,
    )
    from .operator import (
        outer_completion_marker,
        verify_operator_attempt,
        verify_outer_completion,
    )

    before = _verify_live_bootstrap_custody()
    closure_path = _canonical_existing_file(
        development_closure,
        label="accepted development closure",
    )
    attempt_path = _canonical_existing_directory(
        closure_attempt,
        label="accepted development closure attempt",
    )
    if (
        closure_path != DEVELOPMENT_CLOSURE_PATH
        or attempt_path != CLOSURE_ATTEMPT_PATH
        or attempt_path != OPERATOR_CLOSURE_ATTEMPT_PATH
    ):
        raise PreformalEvidenceError(
            "accepted closure evidence is outside its canonical namespace"
        )
    evidence = _runtime_development_evidence(closure_path)
    attempt = verify_operator_attempt(attempt_path)
    primary = attempt.get("primary")
    if (
        attempt.get("kind") != "closure"
        or attempt.get("lane") != "development"
        or attempt.get("status") != "accepted"
        or not isinstance(primary, dict)
        or primary.get("closure_reference_file")
        != "closure-reference.json"
    ):
        raise PreformalEvidenceError(
            "development closure attempt is not accepted"
        )
    terminal = attempt_path / "operator-attempt.json"
    completion = outer_completion_marker(terminal)
    outer = verify_outer_completion(terminal)
    terminal_identity = _file_identity(
        terminal,
        label="accepted closure attempt terminal",
        expected_nlink=_OUTER_FINALIZED_CUSTODY,
    )
    completion_identity = _file_identity(
        completion,
        label="accepted closure outer completion",
        expected_nlink=_OUTER_FINALIZED_CUSTODY,
    )
    if (
        outer.get("terminal_sha256") != terminal_identity["sha256"]
        or terminal_identity["sha256"]
        != completion_identity["sha256"]
    ):
        raise PreformalEvidenceError(
            "accepted closure completion identity changed"
        )
    after = _verify_live_bootstrap_custody()
    if before != after:
        raise PreformalEvidenceError(
            "runtime custody changed during accepted closure verification"
        )
    return {
        "schema": "prospect.wm001.preformal-runtime-check.v1",
        "mode": "accepted-closure-evidence",
        "passed": True,
        "development_closure_sha256": evidence[
            "development_closure_sha256"
        ],
        "producer_manifest_sha256": evidence[
            "producer_manifest_sha256"
        ],
        "raw_result_sha256": evidence["raw_result_sha256"],
        "closure_attempt_manifest_sha256": terminal_identity["sha256"],
        "closure_outer_completion_sha256": completion_identity["sha256"],
    }


def _runtime_fresh_closure_reopen(
    path: Path,
    *,
    challenge: str,
    requesting_process_id: int,
) -> dict[str, object]:
    """Reopen closure evidence after an exec into a fresh sealed interpreter."""

    from .binding import _development_matrix_contract_sha256

    if (
        not _is_sha256(challenge)
        or type(requesting_process_id) is not int
        or requesting_process_id <= 0
        or requesting_process_id == os.getpid()
    ):
        raise PreformalEvidenceError(
            "fresh closure-reopen challenge or process identity is malformed"
        )
    evidence = _runtime_development_evidence(path)
    return {
        "schema": _FRESH_CLOSURE_REOPEN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "mode": "fresh-closure-reopen",
        "challenge": challenge,
        "requesting_process_id": requesting_process_id,
        "verifier_process_id": os.getpid(),
        "matrix_contract_sha256": _development_matrix_contract_sha256(),
        "development_closure_sha256": evidence[
            "development_closure_sha256"
        ],
        "producer_manifest_sha256": evidence[
            "producer_manifest_sha256"
        ],
        "raw_result_sha256": evidence["raw_result_sha256"],
        "passed": True,
    }


def validate_fresh_closure_reopen_report(
    value: object,
    *,
    development_closure: Path,
) -> dict[str, object]:
    """Validate a retained fresh-process closure-reopen receipt."""

    from .binding import _development_matrix_contract_sha256

    expected_fields = {
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
    if not isinstance(value, dict):
        raise PreformalEvidenceError(
            "fresh closure-reopen report is not an object"
        )
    requesting_process_id = value.get("requesting_process_id")
    verifier_process_id = value.get("verifier_process_id")
    closure_path = _canonical_existing_file(
        development_closure,
        label="freshly reopened development closure",
    )
    (
        _,
        producer_manifest_sha256,
        raw_result_sha256,
    ) = _verified_closure_member_digests(closure_path)
    if (
        set(value) != expected_fields
        or value.get("schema") != _FRESH_CLOSURE_REOPEN_SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("protocol_version") != PROTOCOL_VERSION
        or value.get("mode") != "fresh-closure-reopen"
        or not _is_sha256(value.get("challenge"))
        or type(requesting_process_id) is not int
        or requesting_process_id <= 0
        or type(verifier_process_id) is not int
        or verifier_process_id <= 0
        or requesting_process_id == verifier_process_id
        or value.get("matrix_contract_sha256")
        != _development_matrix_contract_sha256()
        or value.get("development_closure_sha256")
        != _file_identity(
            closure_path,
            label="freshly reopened development closure",
            expected_nlink=_SINGLE_LINK_CUSTODY,
        )["sha256"]
        or value.get("producer_manifest_sha256")
        != producer_manifest_sha256
        or value.get("raw_result_sha256")
        != raw_result_sha256
        or value.get("passed") is not True
    ):
        raise PreformalEvidenceError(
            "fresh closure-reopen report differs from live sealed evidence"
        )
    return cast(dict[str, object], value)


def fresh_runtime_development_closure_reopen(
    development_closure: Path,
) -> dict[str, object]:
    """Exec a fresh child on inherited seal descriptors and retain its proof."""

    closure_path = _canonical_existing_file(
        development_closure,
        label="development closure for fresh reopen",
    )
    report = _fresh_runtime_child(
        (
            "fresh-closure-reopen",
            "--development-closure",
            str(closure_path),
        ),
        label="closure-reopen",
    )
    return validate_fresh_closure_reopen_report(
        report,
        development_closure=closure_path,
    )


def _fresh_runtime_child(
    mode_arguments: tuple[str, ...],
    *,
    label: str,
) -> dict[str, object]:
    """Execute one nested sealed-runtime child without reacquiring the lock."""

    before = _verify_live_bootstrap_custody()
    bootstrap_descriptor = getattr(
        sys,
        "_prospect_wm001_bootstrap_fd",
        None,
    )
    runtime_seal_descriptor = getattr(
        sys,
        "_prospect_wm001_runtime_seal_fd",
        None,
    )
    if (
        type(bootstrap_descriptor) is not int
        or bootstrap_descriptor < 0
        or type(runtime_seal_descriptor) is not int
        or runtime_seal_descriptor < 0
        or bootstrap_descriptor == runtime_seal_descriptor
    ):
        raise PreformalEvidenceError(
            f"fresh {label} has no inherited seal descriptors"
        )
    challenge = os.urandom(32).hex()
    requesting_process_id = os.getpid()
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        f"/proc/self/fd/{bootstrap_descriptor}",
        "--bootstrap-fd",
        str(bootstrap_descriptor),
        "--runtime-seal-fd",
        str(runtime_seal_descriptor),
        "preformal-runtime",
        *mode_arguments,
        "--challenge",
        challenge,
        "--requesting-process-id",
        str(requesting_process_id),
    )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            pass_fds=(
                bootstrap_descriptor,
                runtime_seal_descriptor,
            ),
            timeout=_FRESH_CLOSURE_REOPEN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreformalEvidenceError(
            f"fresh sealed {label} interpreter could not start"
        ) from error
    diagnostic = completed.stderr[:4096].decode(
        "utf-8",
        errors="replace",
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout)
        > _FRESH_CLOSURE_REOPEN_MAX_OUTPUT_BYTES
        or len(completed.stderr)
        > _FRESH_CLOSURE_REOPEN_MAX_OUTPUT_BYTES
    ):
        raise PreformalEvidenceError(
            f"fresh sealed {label} interpreter failed or wrote stderr: "
            f"{diagnostic}"
        )
    report = _load_canonical_object(
        completed.stdout,
        label=f"fresh sealed {label} report",
    )
    if (
        report.get("challenge") != challenge
        or report.get("requesting_process_id") != requesting_process_id
        or type(report.get("verifier_process_id")) is not int
        or cast(int, report["verifier_process_id"]) <= 0
        or report.get("verifier_process_id") == requesting_process_id
    ):
        raise PreformalEvidenceError(
            f"fresh sealed {label} response does not answer its challenge"
        )
    after = _verify_live_bootstrap_custody()
    if before != after:
        raise PreformalEvidenceError(
            f"runtime custody changed during fresh {label}"
        )
    return report


def _runtime_fresh_identity_conformance(
    *,
    challenge: str,
    requesting_process_id: int,
) -> dict[str, object]:
    """Prove a nested fresh interpreter can reopen custody and the golden."""

    from .binding import _development_matrix_contract_sha256

    before = _verify_live_bootstrap_custody()
    if (
        not _is_sha256(challenge)
        or type(requesting_process_id) is not int
        or requesting_process_id <= 0
        or requesting_process_id == os.getpid()
    ):
        raise PreformalEvidenceError(
            "fresh identity-conformance challenge is malformed"
        )
    report = {
        "schema": _FRESH_IDENTITY_CONFORMANCE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "mode": "fresh-identity-conformance",
        "challenge": challenge,
        "requesting_process_id": requesting_process_id,
        "verifier_process_id": os.getpid(),
        "matrix_contract_sha256": (
            _development_matrix_contract_sha256()
        ),
        "passed": True,
    }
    after = _verify_live_bootstrap_custody()
    if before != after:
        raise PreformalEvidenceError(
            "runtime custody changed during fresh identity conformance"
        )
    return report


def fresh_runtime_identity_conformance() -> dict[str, object]:
    """Exercise the exact nested descriptor path before outcome production."""

    from .binding import _development_matrix_contract_sha256

    report = _fresh_runtime_child(
        ("fresh-identity-conformance",),
        label="identity-conformance",
    )
    if (
        set(report)
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
        or report.get("schema")
        != _FRESH_IDENTITY_CONFORMANCE_SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("protocol_version") != PROTOCOL_VERSION
        or report.get("mode") != "fresh-identity-conformance"
        or report.get("matrix_contract_sha256")
        != _development_matrix_contract_sha256()
        or report.get("passed") is not True
    ):
        raise PreformalEvidenceError(
            "fresh sealed identity-conformance report is invalid"
        )
    return report


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
    fresh_identity = fresh_runtime_identity_conformance()

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
        "inventory": inventory,
        "inventory_sha256": _sha256(_canonical_json_bytes(inventory)),
        "conformance_sha256": _sha256(_canonical_json_bytes(execution)),
        "fresh_runtime_identity_conformance": fresh_identity,
        "fresh_runtime_identity_conformance_sha256": _sha256(
            _canonical_json_bytes(fresh_identity)
        ),
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
    accepted = modes.add_parser(
        "accepted-closure-evidence",
        allow_abbrev=False,
    )
    accepted.add_argument(
        "--development-closure",
        type=Path,
        required=True,
    )
    accepted.add_argument(
        "--closure-attempt",
        type=Path,
        required=True,
    )
    reopen = modes.add_parser("fresh-closure-reopen", allow_abbrev=False)
    reopen.add_argument(
        "--development-closure",
        type=Path,
        required=True,
    )
    reopen.add_argument("--challenge", required=True)
    reopen.add_argument(
        "--requesting-process-id",
        type=int,
        required=True,
    )
    identity = modes.add_parser(
        "fresh-identity-conformance",
        allow_abbrev=False,
    )
    identity.add_argument("--challenge", required=True)
    identity.add_argument(
        "--requesting-process-id",
        type=int,
        required=True,
    )
    conformance = modes.add_parser("bootstrap-inventory-conformance", allow_abbrev=False)
    conformance.add_argument("--device", choices=("cpu", "cuda"), required=True)
    return parser


def runtime_main(argv: list[str] | None = None) -> int:
    """Entry point reached only through launch-bootstrap descriptor custody."""

    arguments = _runtime_parser().parse_args(argv)
    if arguments.mode == "accepted-closure-evidence":
        report = _runtime_accepted_closure_evidence(
            arguments.development_closure,
            arguments.closure_attempt,
        )
    elif arguments.mode == "fresh-closure-reopen":
        report = _runtime_fresh_closure_reopen(
            arguments.development_closure,
            challenge=arguments.challenge,
            requesting_process_id=arguments.requesting_process_id,
        )
    elif arguments.mode == "fresh-identity-conformance":
        report = _runtime_fresh_identity_conformance(
            challenge=arguments.challenge,
            requesting_process_id=arguments.requesting_process_id,
        )
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
        "--closure-attempt",
        type=Path,
        default=CLOSURE_ATTEMPT_PATH,
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
    exit_code = 0
    if arguments.mode == "generate-report":
        path = generate_preformal_report(
            arguments.output,
            runtime_executable=arguments.runtime_executable,
            runtime_seal=arguments.runtime_seal,
            development_closure=arguments.development_closure,
            closure_attempt=arguments.closure_attempt,
            prospective_review=arguments.prospective_review,
            device=arguments.device,
        )
        generated = _load_canonical_object(
            _read_regular(path, label="generated preformal report"),
            label="generated preformal report",
        )
        commands = generated.get("commands")
        if not isinstance(commands, list):
            raise PreformalEvidenceError(
                "generated preformal report has no command rows"
            )
        failed_commands = [
            {
                "ordinal": row.get("ordinal"),
                "name": row.get("name"),
                "exit_code": row.get("exit_code"),
            }
            for row in commands
            if isinstance(row, Mapping) and row.get("passed") is False
        ]
        failed_checks: list[str] = []
        git_after = generated.get("git_after")
        if (
            not isinstance(git_after, Mapping)
            or git_after.get("worktree_clean") is not True
        ):
            failed_checks.append("post_run_worktree_not_clean")
        if generated.get("identities_stable") is not True:
            failed_checks.append("pre_post_identity_drift")
        failed_checks.extend(
            _semantic_failure_checks(path.parent, generated)
        )
        passed = (
            generated.get("all_pass") is True
            and not failed_checks
        )
        if passed:
            verify_preformal_report(path)
        elif not failed_commands and not failed_checks:
            raise PreformalEvidenceError(
                "generated preformal report failed without an identified "
                "command, identity, or semantic check"
            )
        output = {
            "schema": "prospect.wm001.preformal-test-report-generation.v2",
            "report": str(path),
            "report_sha256": _file_identity(
                path,
                label="preformal report",
                expected_nlink=_SINGLE_LINK_CUSTODY,
            )["sha256"],
            "failed_commands": failed_commands,
            "failed_checks": failed_checks,
            "passed": passed,
        }
        exit_code = 0 if passed else 1
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
    return exit_code


__all__ = (
    "CommandSpec",
    "EXPERIMENT_ID",
    "LOG_PREFIX",
    "PREFORMAL_REPORT_NAME",
    "PREFORMAL_REPORT_PATH",
    "PREFORMAL_BUNDLE_PATH",
    "PROTOCOL_VERSION",
    "PreformalEvidenceError",
    "REPORT_NAME",
    "REVIEW_PATH",
    "SCHEMA",
    "fresh_runtime_identity_conformance",
    "generate_preformal_report",
    "fresh_runtime_development_closure_reopen",
    "required_commands",
    "runtime_main",
    "validate_fresh_closure_reopen_report",
    "verify_preformal_report",
    "verify_recorded_preformal_report",
    "verify_prospective_review",
)


if __name__ == "__main__":
    raise SystemExit(main())
