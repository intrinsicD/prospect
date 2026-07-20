"""Create the second-stage implementation binding required before formal WM-001."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import stat
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .artifact import atomic_write_exclusive
from .assurance import ASSURANCE, assurance_record
from .checkpoint import canonical_json_bytes, manifest_schema_sha256
from .model import (
    COVERAGE_SEMANTICS,
    binary64_pit_is_in_inclusive_90_percent_interval,
    canonical_binary64_mixture_pit,
)
from .planning import run_pendulum_conformance
from .runtime_lane import (
    INDEPENDENT_OSCILLATOR_SOURCE,
    run_independent_phase_oscillator_conformance,
)
from .verify import (
    BINDING_SCHEMA_PATH,
    FORMAL_SEEDS,
    PROTOCOL_PATH,
    RESULT_SCHEMA_PATH,
)


def _repository_root() -> Path:
    """Locate the explicit Git worktree even when this module comes from a wheel."""

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
    raise RuntimeError("WM-001 requires an explicit canonical Prospect Git worktree")


REPO = _repository_root()
LOCKFILE = REPO / "requirements-wm001.lock"
DEVELOPMENT_RESULTS_ROOT = REPO / "bench" / "world_model_lifecycle" / "results" / "development"
DEVELOPMENT_CLOSURE_PATH = DEVELOPMENT_RESULTS_ROOT / "development-closure-v1.6.0.json"
ROOT_DISTRIBUTIONS = (
    "gymnasium",
    "jsonschema",
    "numpy",
    "torch",
    "torchrl",
    "tensordict",
    "prospect",
)
EXECUTION_SOURCE_FILES = (
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
)
FORMAL_PROCESS_ENVIRONMENT_KEYS = frozenset(
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
CHECKPOINT_IMPLEMENTATION_SOURCES = (
    "checkpoint.py",
    "domain_graph.py",
    "experiment.py",
    "parity.py",
)
FORMAL_CONFORMANCE_CASES = 1024
FORMAL_CONFORMANCE_SAMPLES_PER_TASK = 512
FORMAL_CONFORMANCE_SEED = 20260717
FORMAL_OSCILLATOR_CONFORMANCE_CASES = 512
FORMAL_OSCILLATOR_CONFORMANCE_SEED = 20260718
FORMAL_CONFORMANCE_TOLERANCES = {
    "observation_atol": 2e-6,
    "reward_atol": 1e-9,
    "planner_observation_atol": 2e-6,
    "planner_reward_atol": 2e-5,
}
FORMAL_CONFORMANCE_PARAMETERS = {
    "g": 10.0,
    "m": 1.0,
    "l": 1.0,
    "dt": 0.05,
    "max_speed": 8.0,
    "max_torque": 2.0,
}
_V130_BOUNDARY_TARGET_F32_HEX = "ac3cdebd"
_V130_BOUNDARY_MEANS_F32_HEX = (
    "8cd85cbb",
    "f032d7bb",
    "d0d5aebc",
    "fcaa09bc",
    "0086a53a",
)
_V130_BOUNDARY_LOG_VARIANCES_F32_HEX = (
    "66b8b3c0",
    "cb11b5c0",
    "d611b2c0",
    "86dcb2c0",
    "9390b2c0",
)
_V130_BOUNDARY_EXPECTED_PIT_HEX = "0x1.999998b3745adp-5"


def _load_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = _stable_regular_payload(path, label=label)
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not readable canonical JSON") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value) + b"\n":
        raise RuntimeError(f"{label} is not one canonical JSON object followed by LF")
    return value


def verify_machine_test_report(path: Path) -> dict[str, object]:
    """Validate the prospective, machine-readable preformal check report."""

    from .preformal import PreformalEvidenceError, verify_preformal_report

    try:
        return cast(dict[str, object], verify_preformal_report(path))
    except PreformalEvidenceError as error:
        raise RuntimeError("formal test report does not prove the complete fixed preformal check set") from error


def preformal_log_rows(
    report_path: Path,
    report: dict[str, object],
) -> list[dict[str, object]]:
    """Return the exact ordered stdout/stderr custody rows from report v2."""

    commands = report.get("commands")
    if not isinstance(commands, list):
        raise RuntimeError("preformal report command block is invalid")
    rows: list[dict[str, object]] = []
    for command in commands:
        if not isinstance(command, dict):
            raise RuntimeError("preformal report command row is invalid")
        for stream in ("stdout", "stderr"):
            reference = command.get(stream)
            if (
                not isinstance(reference, dict)
                or set(reference) != {"file", "bytes", "sha256"}
                or not isinstance(reference.get("file"), str)
            ):
                raise RuntimeError("preformal report log reference is invalid")
            filename = str(reference["file"])
            path = report_path.parent / filename
            payload = _stable_regular_payload(
                path,
                label=f"preformal report {stream} log",
            )
            if (
                path.parent != report_path.parent
                or len(payload) != reference.get("bytes")
                or hashlib.sha256(payload).hexdigest()
                != reference.get("sha256")
            ):
                raise RuntimeError("preformal report log bytes differ from their reference")
            rows.append(
                {
                    "path": filename,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise RuntimeError("preformal report contains duplicate log filenames")
    return rows


def _content_addressed_filename(
    stem: str,
    payload: bytes,
    suffix: str,
) -> str:
    return f"{stem}-{hashlib.sha256(payload).hexdigest()[:16]}{suffix}"


def _audit_environment(
    producer_environment: dict[str, str],
) -> dict[str, str]:
    from .audit_runner import SAFE_RUNTIME_ENVIRONMENT_KEYS

    return {key: value for key, value in producer_environment.items() if key in SAFE_RUNTIME_ENVIRONMENT_KEYS}


def build_bound_audit_execution(
    *,
    device: str,
    packages: list[dict[str, object]],
    roots: tuple[Path, ...],
    standard_library: dict[str, object],
    producer_environment: dict[str, str],
    repeat_count: int = 3,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Prove and materialize both prebinding and future outcome-audit roles."""

    from .artifact_audit import (
        build_prebinding_conformance_request,
        canonical_prebinding_request_bytes,
    )
    from .audit_runner import (
        bootstrap_source_bytes,
        build_runtime_manifest,
        captured_support_argument,
        conformance_receipt_bytes,
        run_source_mode_conformance,
    )

    if repeat_count < 3:
        raise ValueError("audit execution requires at least three source-mode repeats")
    auditor_source = Path(__file__).with_name("artifact_audit.py")
    support_root = REPO / "bench" / "world_model_lifecycle"
    protocol_path = support_root / "protocol.json"
    result_schema_path = support_root / "schemas" / "raw-result.schema.json"
    scientific_sources = {
        name: support_root / name
        for name in (
            "learning.py",
            "model.py",
            "planning.py",
            "runtime_lane.py",
        )
    }
    root_paths = {
        **{f"package-root-{index:02d}": root for index, root in enumerate(roots)},
        "standard-library": Path(str(standard_library["path"])),
    }
    request = build_prebinding_conformance_request(
        protocol_path,
        scientific_source_paths=scientific_sources,
        root_paths=root_paths,
        device=device,
        support_locator_root=support_root,
        package_rows=packages,
    )
    request_bytes = canonical_prebinding_request_bytes(request)
    safe_environment = _audit_environment(producer_environment)
    support_sources = {
        "protocol.json": protocol_path,
        **scientific_sources,
    }
    with tempfile.TemporaryDirectory(
        prefix="prospect-wm001-prebinding-",
    ) as temporary:
        request_path = Path(temporary).resolve(strict=True) / "prebinding-request.json"
        request_path.write_bytes(request_bytes)
        conformance = run_source_mode_conformance(
            auditor_source,
            auditor_arguments=(
                "--prebinding-conformance",
                captured_support_argument("prebinding-request.json"),
            ),
            support_files={
                "prebinding-request.json": request_path,
                **support_sources,
            },
            closure_import_roots=roots,
            working_directory=REPO,
            environment=safe_environment,
            repeat_count=repeat_count,
        )
    report = dict(conformance.path_execution.report)
    if report.get("passed") is not True or conformance.path_execution.stdout != conformance.descriptor_execution.stdout:
        raise RuntimeError("prebinding audit-execution conformance did not pass")
    path_runtime = conformance.path_execution.runtime_manifest
    descriptor_runtime = conformance.descriptor_execution.runtime_manifest
    path_invocation = conformance.path_execution.invocation_manifest
    descriptor_invocation = conformance.descriptor_execution.invocation_manifest
    conformance_report = conformance.path_execution.stdout
    conformance_receipt = conformance_receipt_bytes(conformance)
    outcome_support_files = {
        "protocol.json": protocol_path,
        "schemas/raw-result.schema.json": result_schema_path,
    }
    outcome_runtime = build_runtime_manifest(
        auditor_source,
        support_files=outcome_support_files,
        closure_import_roots=roots,
        source_mode="descriptor",
        environment=safe_environment,
    )
    bootstrap = bootstrap_source_bytes()
    payloads = {
        _content_addressed_filename("audit-bootstrap", bootstrap, ".py"): bootstrap,
        _content_addressed_filename(
            "audit-prebinding-request",
            request_bytes,
            ".json",
        ): request_bytes,
        _content_addressed_filename(
            "audit-prebinding-path-runtime",
            path_runtime,
            ".json",
        ): path_runtime,
        _content_addressed_filename(
            "audit-prebinding-descriptor-runtime",
            descriptor_runtime,
            ".json",
        ): descriptor_runtime,
        _content_addressed_filename(
            "audit-prebinding-path-invocation",
            path_invocation,
            ".json",
        ): path_invocation,
        _content_addressed_filename(
            "audit-prebinding-descriptor-invocation",
            descriptor_invocation,
            ".json",
        ): descriptor_invocation,
        _content_addressed_filename(
            "audit-prebinding-conformance",
            conformance_report,
            ".json",
        ): conformance_report,
        _content_addressed_filename(
            "audit-prebinding-execution-receipt",
            conformance_receipt,
            ".json",
        ): conformance_receipt,
        _content_addressed_filename(
            "audit-outcome-runtime",
            outcome_runtime,
            ".json",
        ): outcome_runtime,
    }
    if len(payloads) != 9:
        raise RuntimeError("audit-execution evidence filenames collided")

    def locate(prefix: str) -> tuple[str, bytes]:
        matches = [(filename, payload) for filename, payload in payloads.items() if filename.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"audit-execution evidence prefix is ambiguous: {prefix}")
        return matches[0]

    bootstrap_name, _ = locate("audit-bootstrap-")
    request_name, _ = locate("audit-prebinding-request-")
    path_runtime_name, _ = locate("audit-prebinding-path-runtime-")
    descriptor_runtime_name, _ = locate("audit-prebinding-descriptor-runtime-")
    path_invocation_name, _ = locate("audit-prebinding-path-invocation-")
    descriptor_invocation_name, _ = locate("audit-prebinding-descriptor-invocation-")
    conformance_name, _ = locate("audit-prebinding-conformance-")
    conformance_receipt_name, _ = locate("audit-prebinding-execution-receipt-")
    outcome_runtime_name, _ = locate("audit-outcome-runtime-")
    request_identity = report.get("request_sha256")
    if not isinstance(request_identity, str) or len(request_identity) != 64:
        raise RuntimeError("prebinding report has no semantic request identity")
    block: dict[str, object] = {
        "runner_source_sha256": sha256_file(Path(__file__).with_name("audit_runner.py")),
        "auditor_source_sha256": sha256_file(auditor_source),
        "adjudicator_source_sha256": sha256_file(Path(__file__).with_name("adjudication.py")),
        "bootstrap_source_file": bootstrap_name,
        "bootstrap_source_bytes": len(bootstrap),
        "bootstrap_source_sha256": hashlib.sha256(bootstrap).hexdigest(),
        "prebinding_request_file": request_name,
        "prebinding_request_bytes": len(request_bytes),
        "prebinding_request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "prebinding_request_identity_sha256": request_identity,
        "prebinding_path_runtime_manifest_file": path_runtime_name,
        "prebinding_path_runtime_manifest_bytes": len(path_runtime),
        "prebinding_path_runtime_manifest_sha256": hashlib.sha256(path_runtime).hexdigest(),
        "prebinding_descriptor_runtime_manifest_file": descriptor_runtime_name,
        "prebinding_descriptor_runtime_manifest_bytes": len(descriptor_runtime),
        "prebinding_descriptor_runtime_manifest_sha256": hashlib.sha256(descriptor_runtime).hexdigest(),
        "prebinding_path_invocation_manifest_file": path_invocation_name,
        "prebinding_path_invocation_manifest_bytes": len(path_invocation),
        "prebinding_path_invocation_manifest_sha256": hashlib.sha256(path_invocation).hexdigest(),
        "prebinding_descriptor_invocation_manifest_file": (descriptor_invocation_name),
        "prebinding_descriptor_invocation_manifest_bytes": len(descriptor_invocation),
        "prebinding_descriptor_invocation_manifest_sha256": hashlib.sha256(descriptor_invocation).hexdigest(),
        "prebinding_conformance_report_file": conformance_name,
        "prebinding_conformance_report_bytes": len(conformance_report),
        "prebinding_conformance_report_sha256": hashlib.sha256(conformance_report).hexdigest(),
        "prebinding_execution_receipt_file": conformance_receipt_name,
        "prebinding_execution_receipt_bytes": len(conformance_receipt),
        "prebinding_execution_receipt_sha256": hashlib.sha256(conformance_receipt).hexdigest(),
        "outcome_runtime_manifest_file": outcome_runtime_name,
        "outcome_runtime_manifest_bytes": len(outcome_runtime),
        "outcome_runtime_manifest_sha256": hashlib.sha256(outcome_runtime).hexdigest(),
        "outcome_source_mode": "descriptor",
        "outcome_support_files": [
            "protocol.json",
            "schemas/raw-result.schema.json",
        ],
        "outcome_argv_role": ["<canonical-producer-root>"],
        "outcome_working_directory": str(REPO),
        "interpreter_flags": ["-I", "-S", "-B"],
        "repeat_count": repeat_count,
        "path_descriptor_equal": True,
        "passed": True,
    }
    return block, payloads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_from_little_endian_hex(value: str) -> float:
    return float(struct.unpack("<f", bytes.fromhex(value))[0])


def run_coverage_conformance() -> dict[str, object]:
    """Exercise the producer reference on fixed endpoint and regression cases."""

    direct_cases = (
        ("lower-binary64", 0.05, True),
        ("lower-predecessor", math.nextafter(0.05, -math.inf), False),
        ("lower-successor", math.nextafter(0.05, math.inf), True),
        ("upper-binary64", 0.95, True),
        ("upper-predecessor", math.nextafter(0.95, -math.inf), True),
        ("upper-successor", math.nextafter(0.95, math.inf), False),
        ("central", 0.5, True),
        ("zero-tail", 0.0, False),
        ("one-tail", 1.0, False),
    )
    cases: list[dict[str, object]] = []
    passed = True
    for case_id, pit, expected in direct_cases:
        observed = binary64_pit_is_in_inclusive_90_percent_interval(pit)
        case_passed = observed is expected
        cases.append(
            {
                "case_id": case_id,
                "kind": "binary64_pit",
                "pit_hex": pit.hex(),
                "expected_covered": expected,
                "observed_covered": observed,
                "passed": case_passed,
            }
        )
        passed = passed and case_passed

    target = _float32_from_little_endian_hex(_V130_BOUNDARY_TARGET_F32_HEX)
    means = tuple(_float32_from_little_endian_hex(value) for value in _V130_BOUNDARY_MEANS_F32_HEX)
    log_variances = tuple(_float32_from_little_endian_hex(value) for value in _V130_BOUNDARY_LOG_VARIANCES_F32_HEX)
    pit = canonical_binary64_mixture_pit(target, means, log_variances)
    observed = binary64_pit_is_in_inclusive_90_percent_interval(pit)
    regression_passed = pit.hex() == _V130_BOUNDARY_EXPECTED_PIT_HEX and observed is False
    cases.append(
        {
            "case_id": "v130-disclosed-boundary-coordinate",
            "kind": "float32_mixture_inputs",
            "provenance": (
                "v1.3 formal seed 3332986400, task-A corrupted, sidecar row 207, target dimension 0; diagnostic only"
            ),
            "target_little_endian_f32_hex": _V130_BOUNDARY_TARGET_F32_HEX,
            "member_means_little_endian_f32_hex": list(_V130_BOUNDARY_MEANS_F32_HEX),
            "member_log_variances_little_endian_f32_hex": list(_V130_BOUNDARY_LOG_VARIANCES_F32_HEX),
            "expected_pit_hex": _V130_BOUNDARY_EXPECTED_PIT_HEX,
            "observed_pit_hex": pit.hex(),
            "expected_covered": False,
            "observed_covered": observed,
            "passed": regression_passed,
        }
    )
    passed = passed and regression_passed
    corpus = {"semantics_id": COVERAGE_SEMANTICS, "cases": cases}
    report: dict[str, object] = {
        "schema": "prospect.wm001.coverage-conformance.v1",
        "semantics_id": COVERAGE_SEMANTICS,
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "corpus_sha256": hashlib.sha256(canonical_json_bytes(corpus)).hexdigest(),
        "cases": cases,
        "passed": passed,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_is_clean() -> bool:
    return not git_output("status", "--short", "--untracked-files=all")


def implementation_files() -> list[dict[str, object]]:
    tracked_python = [
        REPO / relative
        for relative in git_output(
            "ls-files",
            "--",
            "src/prospect",
            "bench",
            "tests",
        ).splitlines()
        if relative.endswith(".py")
    ]
    candidates = [
        *tracked_python,
        REPO / "Makefile",
        REPO / "pyproject.toml",
        LOCKFILE,
        REPO / "bench" / "world_model_lifecycle" / "SEALED_PROTOCOL.sha256",
        REPO / "bench" / "world_model_lifecycle" / "assurance.py",
        REPO / "bench" / "world_model_lifecycle" / "protocol.json",
        REPO / "bench" / "world_model_lifecycle" / "schemas" / "raw-result.schema.json",
        REPO / "bench" / "world_model_lifecycle" / "schemas" / "formal-binding.schema.json",
        REPO / "docs" / "wm001-v160-prospective-harness-review.json",
    ]
    unique = sorted(set(candidates))
    return [
        {
            "path": path.relative_to(REPO).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in unique
    ]


def verify_installed_source_snapshot() -> None:
    """Require the non-editable wheel's executable source bytes to equal Git source."""

    roots = package_roots()
    distributions = _distribution_map(roots)
    try:
        prospect_distribution = distributions["prospect"]
    except KeyError as error:
        raise RuntimeError("isolated environment has no installed Prospect wheel") from error
    if _distribution_is_editable(prospect_distribution):
        raise RuntimeError("formal Prospect installation must not be editable")
    for row in implementation_files():
        relative = str(row["path"])
        if relative.startswith("src/prospect/"):
            installed_relative = relative.removeprefix("src/")
        elif relative.startswith("bench/"):
            installed_relative = relative
        else:
            continue
        installed = Path(str(prospect_distribution.locate_file(installed_relative)))
        if installed.is_symlink() or not installed.is_file():
            raise RuntimeError(f"installed Prospect wheel omits executable source: {relative}")
        if installed.stat().st_size != row["bytes"] or sha256_file(installed) != row["sha256"]:
            raise RuntimeError(f"installed Prospect source differs from committed source: {relative}")


def package_roots() -> tuple[Path, ...]:
    """Return the explicit import root used by the ``-S`` producer bootstrap."""

    executable = Path(sys.executable)
    if not executable.is_absolute():
        raise RuntimeError("WM-001 binding executable is not absolute")
    virtualenv = executable.parent.parent
    configuration = virtualenv / "pyvenv.cfg"
    if not configuration.is_file():
        raise RuntimeError("WM-001 virtual environment has no pyvenv.cfg")
    rows = {
        key.strip().lower(): value.strip().lower()
        for line in configuration.read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    if rows.get("include-system-site-packages") != "false":
        raise RuntimeError("WM-001 virtual environment inherits system site-packages")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = {
        virtualenv / "lib" / version / "site-packages",
        virtualenv / "lib64" / version / "site-packages",
    }
    roots: list[Path] = []
    aliases: list[tuple[Path, Path]] = []
    for candidate in sorted(candidates):
        if not candidate.exists():
            continue
        resolved = candidate.resolve(strict=True)
        if not candidate.is_absolute() or not candidate.is_dir():
            raise RuntimeError(f"WM-001 package root is absent or aliased: {candidate}")
        if resolved == candidate and not candidate.is_symlink():
            roots.append(candidate)
        else:
            aliases.append((candidate, resolved))
    if len(roots) != 1:
        raise RuntimeError("WM-001 virtual environment must have exactly one package root")
    if any(resolved != roots[0] for _, resolved in aliases):
        raise RuntimeError("WM-001 virtual environment has a foreign package-root alias")
    return tuple(roots)


def _distribution_map(
    roots: tuple[Path, ...] | None = None,
) -> dict[str, importlib.metadata.Distribution]:
    selected_roots = package_roots() if roots is None else roots
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions(path=[str(root) for root in selected_roots]):
        raw_name = distribution.metadata["Name"]
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("installed distribution has no canonicalizable Name metadata")
        name = canonicalize_name(raw_name)
        if name in distributions:
            raise RuntimeError(f"duplicate installed distribution identity: {name}")
        distributions[name] = distribution
    return distributions


def _stable_declared_files(
    distribution: importlib.metadata.Distribution,
) -> tuple[importlib.metadata.PackagePath, ...]:
    declared = tuple(distribution.files or ())
    if not declared:
        raise RuntimeError(f"installed distribution {distribution.metadata['Name']!r} has no stable declared files")
    return tuple(sorted(declared, key=str))


def _record_hash_identity(value: object) -> tuple[str, str] | None:
    """Read the supported ``FileHash`` API without its unstable repr text."""

    if value is None:
        return None
    mode = getattr(value, "mode", None)
    encoded = getattr(value, "value", None)
    if not isinstance(mode, str) or not mode or not isinstance(encoded, str) or not encoded:
        raise RuntimeError("installed distribution has a malformed RECORD hash")
    return mode, encoded


def _record_sha256_hex(identity: tuple[str, str]) -> str:
    algorithm, encoded = identity
    if algorithm != "sha256":
        raise RuntimeError("shared package file uses a non-SHA256 RECORD")
    try:
        decoded = base64.b64decode(
            encoded.encode("ascii") + b"=" * (-len(encoded) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise RuntimeError("shared package file has a malformed RECORD hash") from error
    if len(decoded) != hashlib.sha256().digest_size:
        raise RuntimeError("shared package file has a malformed SHA256 RECORD")
    return decoded.hex()


def package_root_ownership() -> dict[str, object]:
    """Prove every import-root file and directory is owned by installed RECORDs."""

    roots = package_roots()
    if len(roots) != 1:
        raise RuntimeError("WM-001 ownership requires exactly one package root")
    root = roots[0]
    distributions = _distribution_map(roots)
    owners: dict[str, list[tuple[str, tuple[str, str] | None]]] = {}
    for distribution_name, distribution in sorted(distributions.items()):
        for entry in _stable_declared_files(distribution):
            located = Path(os.path.abspath(str(distribution.locate_file(entry))))
            if not located.is_relative_to(root):
                continue
            relative = located.relative_to(root).as_posix()
            declared_hash = _record_hash_identity(entry.hash)
            owners.setdefault(relative, []).append((distribution_name, declared_hash))

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
                raise RuntimeError(f"package ownership found a non-directory entry: {relative}")
            if name == "__pycache__":
                raise RuntimeError(f"runtime package root contains forbidden bytecode cache: {relative}")
            actual_directories.add(relative)
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(f"package ownership found a non-regular file: {relative}")
            if path.suffix == ".pyc":
                raise RuntimeError(f"runtime package root contains forbidden bytecode: {relative}")
            actual_files[relative] = path
    owner_paths = set(owners)
    actual_paths = set(actual_files)
    if owner_paths != actual_paths:
        raise RuntimeError(
            "package root RECORD ownership is not exact; "
            f"unowned={sorted(actual_paths - owner_paths)[:20]}, "
            f"missing={sorted(owner_paths - actual_paths)[:20]}"
        )
    implied_directories = {
        parent.as_posix() for relative in actual_paths for parent in Path(relative).parents if parent != Path(".")
    }
    if actual_directories != implied_directories:
        raise RuntimeError(
            "package root directory set is not exactly implied by owned files; "
            f"extra={sorted(actual_directories - implied_directories)[:20]}, "
            f"missing={sorted(implied_directories - actual_directories)[:20]}"
        )
    ownership_rows: list[dict[str, object]] = []
    shared_file_count = 0
    for relative in sorted(actual_paths):
        file_owners = sorted(owners[relative])
        owner_names = [name for name, _ in file_owners]
        if len(owner_names) != len(set(owner_names)):
            raise RuntimeError(f"package file has a duplicate RECORD owner: {relative}")
        if len(file_owners) > 1:
            declared_hashes = {value for _, value in file_owners}
            if None in declared_hashes or len(declared_hashes) != 1:
                raise RuntimeError(f"package file has conflicting RECORD owners: {relative}")
            record_identity = cast(
                tuple[str, str],
                next(iter(declared_hashes)),
            )
            expected = _record_sha256_hex(record_identity)
            if sha256_file(actual_files[relative]) != expected:
                raise RuntimeError(f"shared package file differs from its RECORD: {relative}")
            shared_file_count += 1
        ownership_rows.append(
            {
                "path": relative,
                "owners": owner_names,
            }
        )
    identity = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "files": ownership_rows,
        "directories": sorted(actual_directories),
    }
    return {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": str(root),
        "file_count": len(actual_files),
        "directory_count": len(actual_directories),
        "shared_file_count": shared_file_count,
        "identity_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
    }


def _distribution_is_editable(distribution: importlib.metadata.Distribution) -> bool:
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return False
    try:
        value = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise RuntimeError("installed distribution has malformed direct_url.json") from error
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("dir_info"), dict)
        and value["dir_info"].get("editable") is True
    )


def distribution_sha256(
    name: str,
    *,
    roots: tuple[Path, ...] | None = None,
) -> str:
    """Hash every stable RECORD-declared installed file, including unhashed rows."""

    distributions = _distribution_map(roots)
    canonical_name = canonicalize_name(name)
    try:
        distribution = distributions[canonical_name]
    except KeyError as error:
        raise RuntimeError(f"installed distribution is missing: {canonical_name}") from error
    digest = hashlib.sha256()
    digest.update(b"prospect.wm001.distribution.v2\0")
    digest.update(canonical_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(distribution.version.encode("utf-8"))
    for entry in _stable_declared_files(distribution):
        path = Path(str(distribution.locate_file(entry)))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"installed distribution file is missing or aliased: {entry}")
        digest.update(b"\0")
        digest.update(str(entry).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def dependency_closure() -> tuple[str, ...]:
    """Return the marker-resolved installed closure of the execution roots."""

    roots = package_roots()
    distributions = _distribution_map(roots)
    environment: dict[str, str] = {**default_environment(), "extra": ""}
    pending = [canonicalize_name(name) for name in ROOT_DISTRIBUTIONS]
    discovered: set[str] = set()
    while pending:
        requested = pending.pop()
        if requested in discovered:
            continue
        try:
            distribution = distributions[requested]
        except KeyError as error:
            raise RuntimeError(f"required dependency is absent from the isolated root: {requested}") from error
        discovered.add(requested)
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            marker = requirement.marker
            if marker is None or marker.evaluate(environment):
                dependency = canonicalize_name(requirement.name)
                try:
                    installed = distributions[dependency]
                except KeyError as error:
                    raise RuntimeError(
                        f"marker-selected dependency is absent from the isolated root: {dependency}"
                    ) from error
                if requirement.specifier and not requirement.specifier.contains(
                    installed.version,
                    prereleases=True,
                ):
                    raise RuntimeError(f"installed {dependency}=={installed.version} violates {requirement.specifier}")
                pending.append(dependency)
    return tuple(sorted(discovered))


def package_root_inventory(root: Path) -> dict[str, object]:
    """Hash every regular file in one authorized package root."""

    if root.resolve(strict=True) != root or not root.is_dir():
        raise RuntimeError(f"package root is absent or aliased: {root}")
    digest = hashlib.sha256()
    digest.update(b"prospect.wm001.package-root.v2\0")
    file_count = 0
    directory_count = 0
    total_bytes = 0
    inventory_files: list[Path] = []
    inventory_directories: list[Path] = []
    for directory, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        for name in directory_names:
            directory_path = Path(directory) / name
            mode = directory_path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(
                    "authorized package root contains a symbolic-link directory: "
                    f"{directory_path.relative_to(root).as_posix()}"
                )
            if not stat.S_ISDIR(mode):
                raise RuntimeError(
                    "authorized package root contains a special directory: "
                    f"{directory_path.relative_to(root).as_posix()}"
                )
            inventory_directories.append(directory_path)
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"authorized package root contains a symbolic link: {relative}")
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"authorized package root contains a special file: {relative}")
            inventory_files.append(path)
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
        before = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(before.st_size.to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"authorized package-root entry changed while read: {relative}")
        digest.update(b"\0")
        file_count += 1
        total_bytes += before.st_size
    if file_count == 0:
        raise RuntimeError("authorized package root is empty")
    return {
        "semantics_id": "prospect.wm001.package-root.v2",
        "path": str(root),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def standard_library_inventory() -> dict[str, object]:
    """Bind every stdlib byte reachable under ``-S``, including cached bytecode."""

    root = Path(sysconfig.get_path("stdlib"))
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise RuntimeError("standard-library root is absent or aliased")
    digest = hashlib.sha256()
    digest.update(b"prospect.wm001.standard-library.v2\0")
    file_count = 0
    directory_count = 0
    total_bytes = 0
    inventory_files: list[Path] = []
    inventory_directories: list[Path] = []
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
            mode = directory_path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"standard-library root contains a symbolic-link directory: {relative}")
            if not stat.S_ISDIR(mode):
                raise RuntimeError(f"standard-library root contains a special directory: {relative}")
            if name != "site-packages":
                retained_directories.append(name)
                inventory_directories.append(directory_path)
        directory_names[:] = retained_directories
        for filename in filenames:
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"standard-library root contains a symbolic link: {relative}")
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"standard-library root contains a special file: {relative}")
            inventory_files.append(path)
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
        before = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0file\0")
        digest.update(before.st_size.to_bytes(8, "big", signed=False))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"standard-library entry changed while read: {relative}")
        digest.update(b"\0")
        file_count += 1
        total_bytes += before.st_size
    if file_count == 0:
        raise RuntimeError("standard-library root has no stable files")
    return {
        "semantics_id": "prospect.wm001.standard-library.v2",
        "path": str(root),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def python_flag_identity() -> dict[str, object]:
    return {
        "dont_write_bytecode": sys.flags.dont_write_bytecode,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "no_user_site": sys.flags.no_user_site,
        "safe_path": sys.flags.safe_path,
    }


def require_formal_python_flags() -> dict[str, object]:
    flags = python_flag_identity()
    expected = {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    if flags != expected:
        raise RuntimeError("formal producer and binding require exact CPython flags -I -S -B")
    return flags


def require_formal_process_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if (
        set(environment) - FORMAL_PROCESS_ENVIRONMENT_KEYS
        or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or environment.get("LAZY_LEGACY_OP") != "False"
        or environment.get("LC_ALL") != "C.UTF-8"
        or environment.get("PATH") != "/usr/bin:/bin"
        or environment.get("PYGAME_HIDE_SUPPORT_PROMPT") != "hide"
        or environment.get("SDL_AUDIODRIVER") != "dsp"
        or environment.get("TZ") != "UTC"
        or any("\0" in key or "\0" in value for key, value in environment.items())
    ):
        raise RuntimeError("formal runtime requires an env -i process with the exact safe WM-001 environment")
    return dict(sorted(environment.items()))


def installed_package_rows() -> list[dict[str, object]]:
    roots = package_roots()
    distributions = _distribution_map(roots)
    rows: list[dict[str, object]] = [
        {
            "name": "python",
            "version": platform.python_version(),
            "distribution_sha256": sha256_file(Path(sys.executable)),
            "declared_file_count": 1,
            "editable": False,
        }
    ]
    for name, distribution in sorted(distributions.items()):
        editable = _distribution_is_editable(distribution)
        if editable:
            raise RuntimeError(f"editable installed distribution is forbidden: {name}")
        rows.append(
            {
                "name": name,
                "version": distribution.version,
                "distribution_sha256": distribution_sha256(name, roots=roots),
                "declared_file_count": len(_stable_declared_files(distribution)),
                "editable": False,
            }
        )
    closure = set(dependency_closure())
    installed = {str(row["name"]) for row in rows if row["name"] != "python"}
    if closure != installed:
        raise RuntimeError(
            "isolated installed inventory must equal the exact runtime dependency closure; "
            f"extra={sorted(installed - closure)}, missing={sorted(closure - installed)}"
        )
    return rows


def verify_lockfile_rows(packages: list[dict[str, object]]) -> None:
    expected = [
        (
            str(row["name"]),
            str(row["version"]),
            str(row["distribution_sha256"]),
        )
        for row in packages
    ]
    parsed: list[tuple[str, str, str]] = []
    for line in LOCKFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            identity, digest = stripped.split(" --distribution-sha256=", 1)
            name, version = identity.split("==", 1)
        except ValueError as error:
            raise RuntimeError(f"malformed WM-001 lock row: {stripped!r}") from error
        parsed.append((name, version, digest))
    if parsed != expected or len(parsed) != len(set(parsed)):
        raise RuntimeError("WM-001 lockfile differs from the live installed dependency closure")


def create_audit_reproduction_receipt(
    *,
    supplied_audit_path: Path,
    execution: object,
    output_path: Path,
) -> dict[str, object]:
    """Preserve one actual descriptor-mode replay, never caller-supplied identities."""

    from .audit_runner import (
        INVOCATION_MANIFEST_SCHEMA,
        RUNTIME_MANIFEST_SCHEMA,
        AuditExecution,
        bootstrap_source_sha256,
    )

    if not isinstance(execution, AuditExecution):
        raise TypeError("audit reproduction requires one captured AuditExecution")
    supplied = _stable_regular_payload(
        supplied_audit_path,
        label="supplied independent audit",
    )
    if supplied != execution.stdout:
        raise RuntimeError("isolated audit reproduction differs from the supplied audit bytes")
    audit = _canonical_json_object(
        supplied,
        label="supplied independent audit",
    )
    runtime = _canonical_json_object(
        execution.runtime_manifest,
        label="captured audit runtime manifest",
    )
    invocation = _canonical_json_object(
        execution.invocation_manifest,
        label="captured audit invocation manifest",
    )
    auditor_source = Path(__file__).with_name("artifact_audit.py")
    runner_source = Path(__file__).with_name("audit_runner.py")
    expected_support = _expected_development_audit_support_rows()
    observed_support = [
        {
            "path": row.relative_path,
            "bytes": row.bytes,
            "sha256": row.sha256,
        }
        for row in execution.support_files
    ]
    if (
        execution.command[:4] != (sys.executable, "-I", "-S", "-B")
        or audit.get("passed") is not True
        or execution.report.get("passed") is not True
        or dict(execution.report) != audit
        or execution.returncode != 0
        or execution.source_mode != "descriptor"
        or execution.runtime_manifest_sha256 != hashlib.sha256(execution.runtime_manifest).hexdigest()
        or execution.invocation_manifest_sha256 != hashlib.sha256(execution.invocation_manifest).hexdigest()
        or execution.bootstrap_sha256 != bootstrap_source_sha256()
        or execution.auditor_source_sha256 != sha256_file(auditor_source)
        or observed_support != expected_support
        or runtime.get("schema") != RUNTIME_MANIFEST_SCHEMA
        or runtime.get("bootstrap_sha256") != execution.bootstrap_sha256
        or runtime.get("source")
        != {
            "mode": "descriptor",
            "path": "artifact_audit.py",
            "bytes": auditor_source.stat().st_size,
            "sha256": execution.auditor_source_sha256,
        }
        or runtime.get("support_files") != expected_support
        or invocation.get("schema") != INVOCATION_MANIFEST_SCHEMA
        or invocation.get("runtime_manifest_sha256") != execution.runtime_manifest_sha256
    ):
        raise RuntimeError("development qualification requires one authentic passing descriptor audit")
    if output_path.exists():
        raise FileExistsError(f"refusing to replace audit reproduction receipt: {output_path}")
    runtime_filename = _content_addressed_filename(
        "development-audit-runtime",
        execution.runtime_manifest,
        ".json",
    )
    invocation_filename = _content_addressed_filename(
        "development-audit-invocation",
        execution.invocation_manifest,
        ".json",
    )
    stderr_filename = _content_addressed_filename(
        "development-audit-stderr",
        execution.stderr,
        ".log",
    )
    sidecars = {
        runtime_filename: execution.runtime_manifest,
        invocation_filename: execution.invocation_manifest,
        stderr_filename: execution.stderr,
    }
    if any(output_path.with_name(filename).exists() for filename in sidecars):
        raise FileExistsError("refusing to replace audit reproduction sidecar")
    receipt = {
        "schema": "prospect.wm001.audit-reproduction.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "supplied_audit_sha256": hashlib.sha256(supplied).hexdigest(),
        "reproduced_audit_sha256": hashlib.sha256(execution.stdout).hexdigest(),
        "byte_identical": True,
        "returncode": execution.returncode,
        "source_mode": execution.source_mode,
        "stdout_bytes": len(execution.stdout),
        "stderr_file": stderr_filename,
        "stderr_bytes": len(execution.stderr),
        "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        "runtime_manifest_file": runtime_filename,
        "runtime_manifest_bytes": len(execution.runtime_manifest),
        "runtime_manifest_sha256": execution.runtime_manifest_sha256,
        "invocation_manifest_file": invocation_filename,
        "invocation_manifest_bytes": len(execution.invocation_manifest),
        "invocation_manifest_sha256": execution.invocation_manifest_sha256,
        "bootstrap_sha256": execution.bootstrap_sha256,
        "runner_source_sha256": sha256_file(runner_source),
        "auditor_source_sha256": execution.auditor_source_sha256,
        "support_files": observed_support,
        "passed": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for filename, payload in sidecars.items():
        atomic_write_exclusive(output_path.with_name(filename), payload)
    atomic_write_exclusive(output_path, canonical_json_bytes(receipt) + b"\n")
    return receipt


_DEVELOPMENT_EXECUTION_FIELDS = (
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
)
_DEVELOPMENT_RUNTIME_SEAL_FIELDS = {
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

_AUDIT_RECEIPT_FIELDS = {
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

_DEVELOPMENT_CLOSURE_FIELDS = {
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

_MAX_QUALIFICATION_MEMBERS = 100_000
_MAX_QUALIFICATION_MEMBER_BYTES = 4 << 30
_MAX_QUALIFICATION_TOTAL_MEMBER_BYTES = 32 << 30
_MAX_QUALIFICATION_ARCHIVE_BYTES = 40 << 30
_MAX_RETAINED_QUALIFICATION_MEMBER_BYTES = 64 << 20
_MAX_RETAINED_QUALIFICATION_TOTAL_BYTES = 256 << 20


@dataclass(frozen=True, slots=True)
class _QualificationFileSource:
    """One live producer member and its role-specific custody contract."""

    path: Path
    expected_nlink: int
    role: str


def _development_producer_source(
    producer_root: Path,
    path: Path,
) -> _QualificationFileSource:
    """Type a development producer member before it crosses into the archive."""

    try:
        relative = path.relative_to(producer_root).as_posix()
    except ValueError as error:
        raise RuntimeError(
            "development qualification source is outside the producer"
        ) from error
    if relative == "producer-manifest.json":
        return _QualificationFileSource(
            path=path,
            expected_nlink=2,
            role="outer-finalized producer manifest",
        )
    return _QualificationFileSource(
        path=path,
        expected_nlink=1,
        role="ordinary producer member",
    )


def _canonical_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict) or payload != canonical_json_bytes(value) + b"\n":
        raise RuntimeError(f"{label} is not one canonical JSON object followed by LF")
    return cast(dict[str, object], value)


def _stable_stat_identity(row: os.stat_result) -> tuple[int, ...]:
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


def _stable_regular_payload(
    path: Path,
    *,
    label: str,
    limit: int = 64 << 20,
    expected_nlink: int = 1,
) -> bytes:
    """Read one canonical regular file while binding its inode metadata."""

    if expected_nlink not in {1, 2}:
        raise ValueError(f"{label} has an unsupported link-count contract")
    candidate = path if path.is_absolute() else Path.cwd() / path
    if candidate.is_symlink() or not candidate.is_file() or candidate.resolve(strict=True) != candidate:
        raise RuntimeError(f"{label} is missing, aliased, or non-canonical")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != expected_nlink:
            raise RuntimeError(
                f"{label} violates its {expected_nlink}-link custody contract"
            )
        if before.st_size > limit:
            raise RuntimeError(f"{label} exceeds its byte limit")
        payload = os.pread(descriptor, before.st_size + 1, 0)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or _stable_stat_identity(before) != _stable_stat_identity(after):
        raise RuntimeError(f"{label} changed while read")
    return payload


def _expected_development_audit_support_rows() -> list[dict[str, object]]:
    return [
        {
            "path": "protocol.json",
            "bytes": PROTOCOL_PATH.stat().st_size,
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        {
            "path": "schemas/raw-result.schema.json",
            "bytes": RESULT_SCHEMA_PATH.stat().st_size,
            "sha256": sha256_file(RESULT_SCHEMA_PATH),
        },
    ]


def _receipt_sidecar(
    receipt_path: Path,
    receipt: dict[str, object],
    *,
    prefix: str,
    label: str,
) -> tuple[Path, bytes]:
    filename = receipt.get(f"{prefix}_file")
    expected_bytes = receipt.get(f"{prefix}_bytes")
    expected_sha256 = receipt.get(f"{prefix}_sha256")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise RuntimeError(f"{label} reference is malformed")
    path = receipt_path.with_name(filename)
    payload = _stable_regular_payload(path, label=label)
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != expected_sha256
        or filename
        != _content_addressed_filename(
            {
                "runtime_manifest": "development-audit-runtime",
                "invocation_manifest": "development-audit-invocation",
                "stderr": "development-audit-stderr",
            }[prefix],
            payload,
            ".log" if prefix == "stderr" else ".json",
        )
    ):
        raise RuntimeError(f"{label} differs from its content-addressed identity")
    return path, payload


def _validate_execution_identity(
    execution: object,
    *,
    require_live_identity: bool,
) -> dict[str, object]:
    if not isinstance(execution, dict) or set(execution) < set(_DEVELOPMENT_EXECUTION_FIELDS):
        raise RuntimeError("development result execution identity is incomplete")
    selected = {field: execution[field] for field in _DEVELOPMENT_EXECUTION_FIELDS}
    commit = selected["git_commit"]
    tree = selected["git_tree"]
    roots = selected["package_roots"]
    standard_library = selected["standard_library"]
    environment = selected["process_environment"]
    flags = selected["python_flags"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(tree, str)
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in tree)
        or selected["worktree_clean"] is not True
        or selected["deterministic_algorithms"] is not True
        or selected["runtime_seal_descriptor_custody"] is not True
        or selected["bootstrap_descriptor_custody"] is not True
        or not isinstance(roots, list)
        or len(roots) != 1
        or not isinstance(roots[0], dict)
        or not isinstance(standard_library, dict)
        or not isinstance(environment, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items())
        or not isinstance(flags, dict)
        or any(
            not isinstance(selected[field], str) or len(cast(str, selected[field])) != 64
            for field in (
                "dependency_lock_sha256",
                "python_executable_sha256",
                "runtime_seal_sha256",
                "producer_bootstrap_sha256",
            )
        )
    ):
        raise RuntimeError("development result execution identity is malformed")
    if require_live_identity:
        executable = Path(sys.executable).resolve(strict=True)
        packages = installed_package_rows()
        verify_lockfile_rows(packages)
        verify_installed_source_snapshot()
        current_roots = [package_root_inventory(root) for root in package_roots()]
        expected = {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_tree": git_output("rev-parse", "HEAD^{tree}"),
            "worktree_clean": source_is_clean(),
            "dependency_lock_sha256": sha256_file(LOCKFILE),
            "python_executable": sys.executable,
            "python_executable_sha256": sha256_file(executable),
            "python_version": platform.python_version(),
            "python_flags": require_formal_python_flags(),
            "process_environment": require_formal_process_environment(),
            "runtime_seal_descriptor_custody": True,
            "producer_bootstrap_sha256": sha256_file(Path(__file__).with_name("producer_bootstrap.py")),
            "bootstrap_descriptor_custody": True,
            "package_roots": current_roots,
            "standard_library": standard_library_inventory(),
        }
        if any(selected.get(field) != value for field, value in expected.items()):
            raise RuntimeError("development result runtime/source differs from the live sealed closure")
    return selected


def _validate_producer_custody(
    *,
    execution: dict[str, object],
    runtime_seal_payload: bytes,
    bootstrap_payload: bytes,
    launch_bootstrap_payload: bytes,
) -> dict[str, object]:
    """Cross-check the producer's pre-import seal separately from audit runtime."""

    runtime_seal = _canonical_json_object(
        runtime_seal_payload,
        label="development producer runtime seal",
    )
    expected_python = {
        "executable": execution["python_executable"],
        "resolved_executable": str(Path(str(execution["python_executable"])).resolve(strict=True)),
        "sha256": execution["python_executable_sha256"],
        "version": [int(part) for part in str(execution["python_version"]).split(".")],
    }
    ownership = package_root_ownership()
    bootstrap_sha256 = hashlib.sha256(bootstrap_payload).hexdigest()
    launch_sha256 = hashlib.sha256(launch_bootstrap_payload).hexdigest()
    if (
        set(runtime_seal) != _DEVELOPMENT_RUNTIME_SEAL_FIELDS
        or runtime_seal.get("schema") != "prospect.wm001.runtime-seal.v1"
        or runtime_seal.get("experiment_id") != "WM-001"
        or runtime_seal.get("protocol_version") != "1.6.0"
        or runtime_seal.get("assurance") != ASSURANCE
        or runtime_seal.get("git_commit") != execution["git_commit"]
        or runtime_seal.get("git_tree") != execution["git_tree"]
        or runtime_seal.get("worktree_clean") is not True
        or runtime_seal.get("python") != expected_python
        or runtime_seal.get("required_flags") != execution["python_flags"]
        or runtime_seal.get("process_environment") != execution["process_environment"]
        or runtime_seal.get("bootstrap_source_sha256") != bootstrap_sha256
        or runtime_seal.get("package_roots") != execution["package_roots"]
        or runtime_seal.get("standard_library") != execution["standard_library"]
        or runtime_seal.get("package_ownership") != ownership
        or hashlib.sha256(runtime_seal_payload).hexdigest() != execution["runtime_seal_sha256"]
        or bootstrap_sha256 != execution["producer_bootstrap_sha256"]
        or bootstrap_payload != Path(__file__).with_name("producer_bootstrap.py").read_bytes()
        or launch_bootstrap_payload != Path(__file__).with_name("launch_bootstrap.py").read_bytes()
    ):
        raise RuntimeError("development producer runtime seal/bootstrap custody is invalid")
    return {
        "runtime_seal_sha256": hashlib.sha256(runtime_seal_payload).hexdigest(),
        "producer_bootstrap_sha256": bootstrap_sha256,
        "launch_bootstrap_sha256": launch_sha256,
        "package_ownership": ownership,
    }


def _validate_development_audit_evidence(
    *,
    producer_root: Path,
    result_sha256: str,
    execution: dict[str, object],
    audit_payload: bytes,
    receipt_payload: bytes,
    runtime_payload: bytes,
    invocation_payload: bytes,
    stderr_payload: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    from .audit_runner import (
        INVOCATION_MANIFEST_SCHEMA,
        RUNTIME_MANIFEST_SCHEMA,
        bootstrap_source_sha256,
    )

    audit = _canonical_json_object(audit_payload, label="development independent audit")
    receipt = _canonical_json_object(
        receipt_payload,
        label="development audit reproduction",
    )
    runtime = _canonical_json_object(
        runtime_payload,
        label="development audit runtime manifest",
    )
    invocation = _canonical_json_object(
        invocation_payload,
        label="development audit invocation manifest",
    )
    auditor_sha256 = sha256_file(Path(__file__).with_name("artifact_audit.py"))
    runner_sha256 = sha256_file(Path(__file__).with_name("audit_runner.py"))
    support_rows = _expected_development_audit_support_rows()
    safe_environment = _audit_environment(cast(dict[str, str], execution["process_environment"]))
    if (
        set(receipt) != _AUDIT_RECEIPT_FIELDS
        or receipt.get("schema") != "prospect.wm001.audit-reproduction.v2"
        or receipt.get("experiment_id") != "WM-001"
        or receipt.get("protocol_version") != "1.6.0"
        or receipt.get("supplied_audit_sha256") != hashlib.sha256(audit_payload).hexdigest()
        or receipt.get("reproduced_audit_sha256") != hashlib.sha256(audit_payload).hexdigest()
        or receipt.get("byte_identical") is not True
        or receipt.get("returncode") != 0
        or receipt.get("source_mode") != "descriptor"
        or receipt.get("stdout_bytes") != len(audit_payload)
        or receipt.get("stderr_bytes") != len(stderr_payload)
        or receipt.get("stderr_sha256") != hashlib.sha256(stderr_payload).hexdigest()
        or receipt.get("stderr_file")
        != _content_addressed_filename(
            "development-audit-stderr",
            stderr_payload,
            ".log",
        )
        or receipt.get("runtime_manifest_bytes") != len(runtime_payload)
        or receipt.get("runtime_manifest_sha256") != hashlib.sha256(runtime_payload).hexdigest()
        or receipt.get("runtime_manifest_file")
        != _content_addressed_filename(
            "development-audit-runtime",
            runtime_payload,
            ".json",
        )
        or receipt.get("invocation_manifest_bytes") != len(invocation_payload)
        or receipt.get("invocation_manifest_sha256") != hashlib.sha256(invocation_payload).hexdigest()
        or receipt.get("invocation_manifest_file")
        != _content_addressed_filename(
            "development-audit-invocation",
            invocation_payload,
            ".json",
        )
        or receipt.get("bootstrap_sha256") != bootstrap_source_sha256()
        or receipt.get("runner_source_sha256") != runner_sha256
        or receipt.get("auditor_source_sha256") != auditor_sha256
        or receipt.get("support_files") != support_rows
        or receipt.get("passed") is not True
    ):
        raise RuntimeError("development audit reproduction receipt is invalid")
    audit_implementation = audit.get("audit_implementation")
    if (
        audit.get("schema") != "prospect.world-model-lifecycle.artifact-audit.v2"
        or audit.get("artifact_root") != str(producer_root)
        or audit.get("result_file") != "result.json"
        or audit.get("result_sha256") != result_sha256
        or audit.get("lane") != "development"
        or audit.get("integrity_passed") is not True
        or audit.get("engineering_complete") is not True
        or audit.get("complete_for_claim") is not False
        or audit.get("passed") is not True
        or not isinstance(audit.get("check_counts"), dict)
        or cast(dict[str, object], audit["check_counts"]).get("failed") != 0
        or cast(dict[str, object], audit["check_counts"]).get("coverage_gaps") != 0
        or audit.get("coverage_gaps") != []
        or not isinstance(audit_implementation, dict)
        or audit_implementation.get("auditor_source_sha256") != auditor_sha256
    ):
        raise RuntimeError("development independent audit is not a complete passing audit")
    runtime_python = runtime.get("python")
    source = runtime.get("source")
    expected_flags = {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    version_parts = tuple(int(part) for part in str(execution["python_version"]).split("."))
    runtime_version = runtime_python.get("version") if isinstance(runtime_python, dict) else None
    if (
        runtime.get("schema") != RUNTIME_MANIFEST_SCHEMA
        or runtime.get("assurance") != ASSURANCE
        or runtime.get("bootstrap_sha256") != bootstrap_source_sha256()
        or not isinstance(runtime_python, dict)
        or runtime_python.get("executable") != execution["python_executable"]
        or runtime_python.get("resolved_executable")
        != str(Path(str(execution["python_executable"])).resolve(strict=True))
        or runtime_python.get("sha256") != execution["python_executable_sha256"]
        or not isinstance(runtime_version, list)
        or tuple(runtime_version) != version_parts
        or runtime.get("required_flags") != expected_flags
        or execution["python_flags"] != expected_flags
        or source
        != {
            "mode": "descriptor",
            "path": "artifact_audit.py",
            "bytes": Path(__file__).with_name("artifact_audit.py").stat().st_size,
            "sha256": auditor_sha256,
        }
        or runtime.get("support_files") != support_rows
        or runtime.get("closure_import_roots") != execution["package_roots"]
        or runtime.get("standard_library") != execution["standard_library"]
        or runtime.get("environment") != safe_environment
    ):
        raise RuntimeError("development audit runtime differs from the producer closure")
    if (
        invocation.get("schema") != INVOCATION_MANIFEST_SCHEMA
        or invocation.get("runtime_manifest_sha256") != hashlib.sha256(runtime_payload).hexdigest()
        or invocation.get("working_directory") != str(REPO)
        or invocation.get("auditor_argv") != [str(producer_root)]
    ):
        raise RuntimeError("development audit invocation does not target the producer")
    return audit, receipt


def _development_matrix_contract_sha256() -> str:
    """Identify the exact non-performance matrix contract checked before closure."""

    from .verify import (
        COMMITTED_PHASE_SPLITS,
        EPISODE_CONTRACTS,
        FORMAL_EPISODE_CONTRACT_COUNTS,
        PREDICTIVE_CONTRACTS,
    )

    value = {
        "episode_contract_counts": [
            [*contract, count] for contract, count in sorted(FORMAL_EPISODE_CONTRACT_COUNTS.items())
        ],
        "predictive_contracts": [list(row) for row in PREDICTIVE_CONTRACTS],
        "policy_contracts": [list(row) for row in EPISODE_CONTRACTS],
        "committed_phase_splits": [[phase, list(splits)] for phase, splits in sorted(COMMITTED_PHASE_SPLITS.items())],
        "episodes": 496,
        "transitions": 99_200,
        "predictive_metrics": 12,
        "policy_runs": 20,
        "updates": 6,
        "optimizer_batch_manifests": 5,
        "optimizer_steps_per_committed_phase": 2_000,
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _result_qualification_payload(
    result: dict[str, object],
    *,
    result_sha256: str,
    execution: dict[str, object],
) -> bytes:
    replicates = cast(list[dict[str, object]], result["replicates"])
    value = {
        "schema": "prospect.wm001.development-result-qualification.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "raw_result_sha256": result_sha256,
        "lane": "development",
        "claim_eligible": False,
        "replicates": [
            {
                "replicate_id": replicate.get("replicate_id"),
                "master_seed": replicate.get("master_seed"),
                "episodes": len(cast(list[object], replicate["episodes"])),
                "transitions": len(cast(list[object], replicate["transitions"])),
                "predictive_metrics": len(cast(list[object], replicate["predictive_metrics"])),
                "policy_runs": len(cast(list[object], replicate["policy_runs"])),
                "updates": len(cast(list[object], replicate["updates"])),
                "optimizer_batch_manifests": len(
                    cast(
                        list[object],
                        replicate["optimizer_batch_manifests"],
                    )
                ),
            }
            for replicate in replicates
        ],
        "matrix_contract_sha256": _development_matrix_contract_sha256(),
        "producer_execution": execution,
    }
    return cast(bytes, canonical_json_bytes(value)) + b"\n"


def _validate_result_qualification(
    payload: bytes,
    *,
    archived_result_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    from .verify import DEVELOPMENT_SEEDS

    value = _canonical_json_object(
        payload,
        label="archived development result qualification",
    )
    replicates = value.get("replicates")
    expected_replicate_counts = {
        "episodes": 496,
        "transitions": 99_200,
        "predictive_metrics": 12,
        "policy_runs": 20,
        "updates": 6,
        "optimizer_batch_manifests": 5,
    }
    if (
        set(value)
        != {
            "schema",
            "experiment_id",
            "protocol_version",
            "protocol_sha256",
            "raw_result_sha256",
            "lane",
            "claim_eligible",
            "replicates",
            "matrix_contract_sha256",
            "producer_execution",
        }
        or value.get("schema") != "prospect.wm001.development-result-qualification.v1"
        or value.get("experiment_id") != "WM-001"
        or value.get("protocol_version") != "1.6.0"
        or value.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
        or value.get("raw_result_sha256") != archived_result_sha256
        or value.get("lane") != "development"
        or value.get("claim_eligible") is not False
        or value.get("matrix_contract_sha256") != _development_matrix_contract_sha256()
        or not isinstance(replicates, list)
        or len(replicates) != 2
        or tuple(row.get("master_seed") if isinstance(row, dict) else None for row in replicates) != DEVELOPMENT_SEEDS
        or any(
            not isinstance(row, dict)
            or set(row)
            != {
                "replicate_id",
                "master_seed",
                *expected_replicate_counts,
            }
            or not isinstance(row.get("replicate_id"), str)
            or any(row.get(field) != expected for field, expected in expected_replicate_counts.items())
            for row in replicates
        )
    ):
        raise RuntimeError("development result qualification does not prove exact seeds/budgets/matrix")
    execution = _validate_execution_identity(
        value.get("producer_execution"),
        require_live_identity=True,
    )
    return value, execution


def _tar_member_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _write_qualification_archive(
    *,
    destination_directory: Path,
    producer_root: Path,
    evidence_payloads: dict[str, bytes],
) -> tuple[Path, dict[str, object]]:
    """Write one deterministic uncompressed tar and publish by content address."""

    from .artifact import _regular_producer_files

    sources: dict[str, _QualificationFileSource | bytes] = {
        f"producer/{path.relative_to(producer_root).as_posix()}": (
            _development_producer_source(producer_root, path)
        )
        for path in _regular_producer_files(producer_root)
    }
    sources.update(evidence_payloads)
    names = sorted(sources)
    if len(names) != len(set(names)) or not 1 <= len(names) <= _MAX_QUALIFICATION_MEMBERS:
        raise RuntimeError("development qualification archive member names collided")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".development-qualification-",
        suffix=".tar.tmp",
        dir=destination_directory,
    )
    temporary = Path(temporary_name)
    rows: list[dict[str, object]] = []
    total_member_bytes = 0
    published_archive: Path | None = None
    try:
        with os.fdopen(descriptor, "w+b") as raw:
            with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in names:
                    source = sources[name]
                    if isinstance(source, _QualificationFileSource):
                        flags = os.O_RDONLY
                        if hasattr(os, "O_NOFOLLOW"):
                            flags |= os.O_NOFOLLOW
                        source_descriptor = os.open(source.path, flags)
                        try:
                            before = os.fstat(source_descriptor)
                            if (
                                not stat.S_ISREG(before.st_mode)
                                or before.st_nlink != source.expected_nlink
                            ):
                                raise RuntimeError(
                                    "development qualification "
                                    f"{source.role} violates its "
                                    f"{source.expected_nlink}-link custody contract"
                                )
                            payload_bytes = before.st_size
                            if (
                                payload_bytes > _MAX_QUALIFICATION_MEMBER_BYTES
                                or total_member_bytes + payload_bytes > _MAX_QUALIFICATION_TOTAL_MEMBER_BYTES
                            ):
                                raise RuntimeError("development qualification archive member exceeds its limits")
                            digest = hashlib.sha256()
                            offset = 0
                            while offset < payload_bytes:
                                chunk = os.pread(
                                    source_descriptor,
                                    min(1 << 20, payload_bytes - offset),
                                    offset,
                                )
                                if not chunk:
                                    raise RuntimeError("development qualification source ended while hashed")
                                digest.update(chunk)
                                offset += len(chunk)
                            payload_sha256 = digest.hexdigest()
                            os.lseek(source_descriptor, 0, os.SEEK_SET)
                            source_stream = os.fdopen(
                                os.dup(source_descriptor),
                                "rb",
                            )
                            with source_stream:
                                archive.addfile(
                                    _tar_member_info(name, payload_bytes),
                                    source_stream,
                                )
                            after = os.fstat(source_descriptor)
                            if offset != payload_bytes or _stable_stat_identity(before) != _stable_stat_identity(after):
                                raise RuntimeError("development qualification source changed during archive creation")
                        finally:
                            os.close(source_descriptor)
                    else:
                        payload_bytes = len(source)
                        if (
                            payload_bytes > _MAX_QUALIFICATION_MEMBER_BYTES
                            or total_member_bytes + payload_bytes > _MAX_QUALIFICATION_TOTAL_MEMBER_BYTES
                        ):
                            raise RuntimeError("development qualification archive member exceeds its limits")
                        payload_sha256 = hashlib.sha256(source).hexdigest()
                        with tempfile.SpooledTemporaryFile(max_size=1 << 20) as payload_stream:
                            payload_stream.write(source)
                            payload_stream.seek(0)
                            archive.addfile(
                                _tar_member_info(name, payload_bytes),
                                payload_stream,
                            )
                    rows.append(
                        {
                            "path": name,
                            "bytes": payload_bytes,
                            "sha256": payload_sha256,
                        }
                    )
                    total_member_bytes += payload_bytes
            raw.flush()
            os.fsync(raw.fileno())
        archive_bytes = temporary.stat().st_size
        if archive_bytes > _MAX_QUALIFICATION_ARCHIVE_BYTES:
            raise RuntimeError("development qualification tar exceeds its byte limit")
        archive_sha256 = sha256_file(temporary)
        archive_name = f"development-qualification-{archive_sha256[:16]}.tar"
        archive_path = destination_directory / archive_name
        try:
            canonical_archive_path = archive_path.relative_to(REPO).as_posix()
        except ValueError as error:
            raise RuntimeError("development qualification archive is outside the repository") from error
        if archive_path.parent != DEVELOPMENT_RESULTS_ROOT:
            raise RuntimeError("development qualification archive is outside its canonical root")
        archive_identity = {
            "format": "ustar-uncompressed-v1",
            "file": archive_name,
            "canonical_path": canonical_archive_path,
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "members": rows,
        }
        if os.path.lexists(archive_path):
            _stream_qualification_archive(
                archive_path,
                archive_identity,
                retained_members=set(),
            )
            temporary.unlink()
            return archive_path, archive_identity
        os.link(temporary, archive_path)
        published_archive = archive_path
        temporary.unlink()
        if archive_path.is_symlink() or archive_path.stat().st_nlink != 1:
            raise RuntimeError("development qualification archive publication is aliased")
        return archive_path, archive_identity
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published_archive is not None:
            published_archive.unlink(missing_ok=True)
        raise


def _stream_qualification_archive(
    archive_path: Path,
    archive_identity: dict[str, object],
    *,
    retained_members: set[str],
) -> dict[str, bytes]:
    """Reopen and hash every member without extracting any pathname."""

    if archive_path.is_symlink() or not archive_path.is_file() or archive_path.resolve(strict=True) != archive_path:
        raise RuntimeError("development qualification archive is missing or aliased")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    archive_descriptor = os.open(archive_path, flags)
    before_archive = os.fstat(archive_descriptor)
    if (
        not stat.S_ISREG(before_archive.st_mode)
        or before_archive.st_nlink != 1
        or before_archive.st_size > _MAX_QUALIFICATION_ARCHIVE_BYTES
    ):
        os.close(archive_descriptor)
        raise RuntimeError("development qualification archive violates its file limits")
    archive_digest = hashlib.sha256()
    archive_offset = 0
    while archive_offset < before_archive.st_size:
        chunk = os.pread(
            archive_descriptor,
            min(1 << 20, before_archive.st_size - archive_offset),
            archive_offset,
        )
        if not chunk:
            os.close(archive_descriptor)
            raise RuntimeError("development qualification archive ended while hashed")
        archive_digest.update(chunk)
        archive_offset += len(chunk)
    after_digest = os.fstat(archive_descriptor)
    archive_bytes = before_archive.st_size
    archive_sha256 = archive_digest.hexdigest()
    rows = archive_identity.get("members")
    row_paths: list[str] = []
    total_member_bytes = 0
    rows_are_valid = isinstance(rows, list) and 1 <= len(rows) <= _MAX_QUALIFICATION_MEMBERS
    if rows_are_valid:
        assert isinstance(rows, list)
        for raw_row in rows:
            if (
                not isinstance(raw_row, dict)
                or set(raw_row) != {"path", "bytes", "sha256"}
                or not isinstance(raw_row.get("path"), str)
                or not isinstance(raw_row.get("bytes"), int)
                or not 0 <= raw_row["bytes"] <= _MAX_QUALIFICATION_MEMBER_BYTES
                or not isinstance(raw_row.get("sha256"), str)
                or len(raw_row["sha256"]) != 64
            ):
                rows_are_valid = False
                break
            row_paths.append(raw_row["path"])
            total_member_bytes += raw_row["bytes"]
    expected_archive_bytes = 0
    if rows_are_valid:
        assert isinstance(rows, list)
        expected_archive_bytes = 2 * tarfile.BLOCKSIZE
        for row in rows:
            assert isinstance(row, dict)
            member_bytes = row["bytes"]
            assert isinstance(member_bytes, int)
            expected_archive_bytes += tarfile.BLOCKSIZE + (
                (member_bytes + tarfile.BLOCKSIZE - 1)
                // tarfile.BLOCKSIZE
            ) * tarfile.BLOCKSIZE
        expected_archive_bytes = (
            (expected_archive_bytes + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE
        ) * tarfile.RECORDSIZE
    if (
        archive_identity.get("format") != "ustar-uncompressed-v1"
        or archive_identity.get("file") != archive_path.name
        or archive_identity.get("canonical_path") != archive_path.relative_to(REPO).as_posix()
        or archive_identity.get("bytes") != archive_bytes
        or archive_identity.get("sha256") != archive_sha256
        or archive_path.name != f"development-qualification-{archive_sha256[:16]}.tar"
        or archive_bytes != expected_archive_bytes
        or archive_bytes > _MAX_QUALIFICATION_ARCHIVE_BYTES
        or not rows_are_valid
        or row_paths != sorted(row_paths)
        or len(set(row_paths)) != len(row_paths)
        or total_member_bytes > _MAX_QUALIFICATION_TOTAL_MEMBER_BYTES
    ):
        os.close(archive_descriptor)
        raise RuntimeError("development qualification archive identity is malformed")
    assert isinstance(rows, list)
    retained: dict[str, bytes] = {}
    retained_total_bytes = 0
    try:
        if archive_offset != archive_bytes or _stable_stat_identity(before_archive) != _stable_stat_identity(
            after_digest
        ):
            raise RuntimeError("development qualification archive changed while hashed")
        os.lseek(archive_descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(archive_descriptor), "rb") as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                observed_count = 0
                for member in archive:
                    if observed_count >= len(rows):
                        raise RuntimeError("development qualification tar has unbound extra members")
                    expected_raw = rows[observed_count]
                    expected = cast(dict[str, object], expected_raw)
                    if (
                        member.name != expected.get("path")
                        or "\\" in member.name
                        or Path(member.name).is_absolute()
                        or "." in Path(member.name).parts
                        or ".." in Path(member.name).parts
                        or Path(member.name).as_posix() != member.name
                        or not member.isreg()
                        or member.type != tarfile.REGTYPE
                        or bool(member.pax_headers)
                        or bool(member.sparse)
                        or member.mode != 0o444
                        or member.uid != 0
                        or member.gid != 0
                        or member.uname != ""
                        or member.gname != ""
                        or member.mtime != 0
                        or member.size != expected.get("bytes")
                    ):
                        raise RuntimeError("development qualification tar metadata changed")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError("development qualification tar member is unreadable")
                    digest = hashlib.sha256()
                    captured = bytearray()
                    if member.name in retained_members and member.size > _MAX_RETAINED_QUALIFICATION_MEMBER_BYTES:
                        raise RuntimeError("retained qualification member exceeds its byte limit")
                    while True:
                        chunk = extracted.read(1 << 20)
                        if not chunk:
                            break
                        digest.update(chunk)
                        if member.name in retained_members:
                            captured.extend(chunk)
                    if digest.hexdigest() != expected.get("sha256"):
                        raise RuntimeError("development qualification tar member digest changed")
                    if member.name in retained_members:
                        retained[member.name] = bytes(captured)
                        retained_total_bytes += len(captured)
                        if retained_total_bytes > _MAX_RETAINED_QUALIFICATION_TOTAL_BYTES:
                            raise RuntimeError("retained qualification evidence exceeds its total limit")
                    observed_count += 1
                if observed_count != len(rows):
                    raise RuntimeError("development qualification tar member set changed")
        after_stream = os.fstat(archive_descriptor)
        if _stable_stat_identity(before_archive) != _stable_stat_identity(after_stream):
            raise RuntimeError("development qualification archive changed while streamed")
    finally:
        os.close(archive_descriptor)
    if set(retained) != retained_members:
        raise RuntimeError("development qualification tar omits required semantic evidence")
    return retained


def _validate_archived_producer(
    retained: dict[str, bytes],
    member_rows: list[dict[str, object]],
) -> dict[str, object]:
    manifest_member = "producer/producer-manifest.json"
    manifest = _canonical_json_object(
        retained[manifest_member],
        label="archived development producer manifest",
    )
    rows = manifest.get("files")
    if (
        manifest.get("schema") != "prospect.wm001.producer-manifest.v1"
        or manifest.get("experiment_id") != "WM-001"
        or manifest.get("lane") != "development"
        or manifest.get("status") != "completed"
        or manifest.get("error") is not None
        or manifest.get("manifest_excludes") != ["producer-manifest.json"]
        or not isinstance(rows, list)
        or manifest.get("file_count") != len(rows)
        or any(not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} for row in rows)
    ):
        raise RuntimeError("archived development producer manifest is incomplete")
    manifest_paths: list[str] = []
    for raw_row in rows:
        row = cast(dict[str, object], raw_row)
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or "." in Path(relative).parts
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or not isinstance(row.get("bytes"), int)
            or not 0 <= cast(int, row["bytes"]) <= _MAX_QUALIFICATION_MEMBER_BYTES
            or not isinstance(row.get("sha256"), str)
            or len(cast(str, row["sha256"])) != 64
        ):
            raise RuntimeError("archived development producer manifest row is malformed")
        manifest_paths.append(relative)
    if manifest_paths != sorted(set(manifest_paths)):
        raise RuntimeError("archived development producer manifest rows are not exact and ordered")
    expected_producer = {
        f"producer/{cast(dict[str, object], row)['path']}": {
            "bytes": cast(dict[str, object], row)["bytes"],
            "sha256": cast(dict[str, object], row)["sha256"],
        }
        for row in rows
    }
    expected_producer[manifest_member] = {
        "bytes": len(retained[manifest_member]),
        "sha256": hashlib.sha256(retained[manifest_member]).hexdigest(),
    }
    observed_producer = {
        str(row["path"]): {
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in member_rows
        if str(row["path"]).startswith("producer/")
    }
    if expected_producer != observed_producer:
        raise RuntimeError("archived producer member set differs from its manifest")
    return manifest


def _reverify_live_development_producer(
    producer_root: Path,
    *,
    expected_manifest: dict[str, object],
    expected_result_sha256: str,
) -> None:
    """Close the mutation window immediately before terminal marker publication."""

    from .artifact import _regular_producer_files, verify_producer_manifest

    current_manifest = cast(
        dict[str, object],
        verify_producer_manifest(producer_root),
    )
    files = _regular_producer_files(producer_root)
    wrong_custody: list[_QualificationFileSource] = []
    for path in files:
        source = _development_producer_source(producer_root, path)
        if source.path.stat().st_nlink != source.expected_nlink:
            wrong_custody.append(source)
    result_payload = _stable_regular_payload(
        producer_root / "result.json",
        label="development result",
    )
    if (
        current_manifest != expected_manifest
        or hashlib.sha256(result_payload).hexdigest() != expected_result_sha256
        or wrong_custody
    ):
        raise RuntimeError(
            "development producer changed or violates typed link custody before closure"
        )


def _closure_path_mode(path: Path, payload: bytes) -> str:
    candidate = path if path.is_absolute() else Path.cwd() / path
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve(strict=True) != candidate
        or candidate.stat().st_nlink != 1
    ):
        raise RuntimeError("development closure marker is missing, aliased, or hard-linked")
    if candidate == DEVELOPMENT_CLOSURE_PATH:
        return "canonical"
    copied_name = f"development-closure-{hashlib.sha256(payload).hexdigest()[:16]}.json"
    if candidate.name == copied_name:
        return "content-addressed-copy"
    raise RuntimeError("development closure is neither canonical nor a preserved copy")


def _repo_artifact_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{label} is missing")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} is not a safe repository-relative path")
    path = REPO / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or aliased")
    return path


def verify_development_closure(path: Path) -> dict[str, object]:
    """Stream and semantically reopen canonical or formally preserved evidence."""

    closure_path = path if path.is_absolute() else Path.cwd() / path
    closure_payload = _stable_regular_payload(
        closure_path,
        label="development closure marker",
    )
    _closure_path_mode(closure_path, closure_payload)
    closure = _canonical_json_object(
        closure_payload,
        label="development closure marker",
    )
    if (
        set(closure) != _DEVELOPMENT_CLOSURE_FIELDS
        or closure.get("schema") != "prospect.wm001.development-closure.v2"
        or closure.get("experiment_id") != "WM-001"
        or closure.get("protocol_version") != "1.6.0"
        or closure.get("engineering_verified") is not True
        or closure.get("audit_reproduced") is not True
        or closure.get("performance_values_bound") is not False
    ):
        raise RuntimeError("development closure identity or status is invalid")
    source = closure.get("source")
    producer_execution = closure.get("producer_execution")
    producer_custody = closure.get("producer_custody")
    audit_execution = closure.get("audit_execution")
    archive_identity = closure.get("qualification_archive")
    producer_root_raw = closure.get("producer_root")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "git_commit",
            "git_tree",
            "worktree_clean",
            "dependency_lock_sha256",
            "producer_bootstrap_sha256",
            "launch_bootstrap_sha256",
            "runner_source_sha256",
            "auditor_source_sha256",
        }
        or not isinstance(producer_execution, dict)
        or set(producer_execution) != set(_DEVELOPMENT_EXECUTION_FIELDS)
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
        or any(
            not isinstance(producer_custody.get(field), str)
            for field in (
                "runtime_seal_member",
                "runtime_seal_sha256",
                "producer_bootstrap_member",
                "producer_bootstrap_sha256",
                "launch_bootstrap_member",
                "launch_bootstrap_sha256",
            )
        )
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
        or not isinstance(archive_identity, dict)
        or set(archive_identity)
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
    ):
        raise RuntimeError("development closure structural identity is malformed")
    producer_root = Path(producer_root_raw)
    if not producer_root.is_absolute() or producer_root.resolve(strict=False) != producer_root:
        raise RuntimeError("development closure producer root is not canonical")
    role_fields = {
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "result_qualification_member": "evidence/development-result-qualification.json",
        "independent_audit_member": "evidence/independent-audit.json",
        "audit_reproduction_member": "evidence/audit-reproduction.json",
    }
    if any(closure.get(field) != expected for field, expected in role_fields.items()):
        raise RuntimeError("development closure fixed archive roles changed")
    sidecar_roles = {
        str(closure["audit_runtime_manifest_member"]),
        str(closure["audit_invocation_manifest_member"]),
        str(closure["audit_stderr_member"]),
    }
    if len(sidecar_roles) != 3 or any(
        not member.startswith("evidence/") or Path(member).name != member.removeprefix("evidence/")
        for member in sidecar_roles
    ):
        raise RuntimeError("development closure audit sidecar roles are unsafe")
    retained_members = {
        "producer/producer-manifest.json",
        "evidence/development-result-qualification.json",
        "evidence/independent-audit.json",
        "evidence/audit-reproduction.json",
        *sidecar_roles,
        str(producer_custody["runtime_seal_member"]),
        str(producer_custody["producer_bootstrap_member"]),
        str(producer_custody["launch_bootstrap_member"]),
    }
    if {
        producer_custody["runtime_seal_member"],
        producer_custody["producer_bootstrap_member"],
        producer_custody["launch_bootstrap_member"],
    } != {
        "evidence/producer-runtime-seal.json",
        "evidence/producer-bootstrap.py",
        "evidence/launch-bootstrap.py",
    }:
        raise RuntimeError("development producer-custody archive roles changed")
    archive_file = archive_identity.get("file")
    archive_relative = archive_identity.get("canonical_path")
    if (
        not isinstance(archive_file, str)
        or Path(archive_file).name != archive_file
        or not isinstance(archive_relative, str)
        or Path(archive_relative).is_absolute()
        or ".." in Path(archive_relative).parts
        or Path(archive_relative).as_posix() != archive_relative
    ):
        raise RuntimeError("development qualification archive filename is unsafe")
    archive_path = REPO / archive_relative
    if archive_path != DEVELOPMENT_RESULTS_ROOT / archive_file or archive_path.resolve(strict=False) != archive_path:
        raise RuntimeError("development qualification archive path is not canonical")
    retained = _stream_qualification_archive(
        archive_path,
        archive_identity,
        retained_members=retained_members,
    )
    member_rows = cast(list[dict[str, object]], archive_identity["members"])
    _validate_archived_producer(retained, member_rows)
    result_rows = [row for row in member_rows if row["path"] == "producer/result.json"]
    if len(result_rows) != 1:
        raise RuntimeError("development qualification archive omits raw result identity")
    result_sha256 = str(result_rows[0]["sha256"])
    qualification, execution = _validate_result_qualification(
        retained["evidence/development-result-qualification.json"],
        archived_result_sha256=result_sha256,
    )
    expected_producer_custody = {
        **_validate_producer_custody(
            execution=execution,
            runtime_seal_payload=retained[str(producer_custody["runtime_seal_member"])],
            bootstrap_payload=retained[str(producer_custody["producer_bootstrap_member"])],
            launch_bootstrap_payload=retained[str(producer_custody["launch_bootstrap_member"])],
        ),
        "runtime_seal_member": "evidence/producer-runtime-seal.json",
        "producer_bootstrap_member": "evidence/producer-bootstrap.py",
        "launch_bootstrap_member": "evidence/launch-bootstrap.py",
    }
    runtime_member = str(closure["audit_runtime_manifest_member"])
    invocation_member = str(closure["audit_invocation_manifest_member"])
    stderr_member = str(closure["audit_stderr_member"])
    _, receipt = _validate_development_audit_evidence(
        producer_root=producer_root,
        result_sha256=result_sha256,
        execution=execution,
        audit_payload=retained["evidence/independent-audit.json"],
        receipt_payload=retained["evidence/audit-reproduction.json"],
        runtime_payload=retained[runtime_member],
        invocation_payload=retained[invocation_member],
        stderr_payload=retained[stderr_member],
    )
    expected_source = {
        "git_commit": execution["git_commit"],
        "git_tree": execution["git_tree"],
        "worktree_clean": True,
        "dependency_lock_sha256": execution["dependency_lock_sha256"],
        "producer_bootstrap_sha256": execution["producer_bootstrap_sha256"],
        "launch_bootstrap_sha256": expected_producer_custody["launch_bootstrap_sha256"],
        "runner_source_sha256": sha256_file(Path(__file__).with_name("audit_runner.py")),
        "auditor_source_sha256": sha256_file(Path(__file__).with_name("artifact_audit.py")),
    }
    expected_audit_execution = {
        "receipt_sha256": hashlib.sha256(retained["evidence/audit-reproduction.json"]).hexdigest(),
        "runtime_manifest_sha256": receipt["runtime_manifest_sha256"],
        "invocation_manifest_sha256": receipt["invocation_manifest_sha256"],
        "stderr_sha256": receipt["stderr_sha256"],
        "bootstrap_sha256": receipt["bootstrap_sha256"],
        "runner_source_sha256": receipt["runner_source_sha256"],
        "auditor_source_sha256": receipt["auditor_source_sha256"],
        "support_files": receipt["support_files"],
        "source_mode": "descriptor",
    }
    if (
        source != expected_source
        or producer_execution != execution
        or qualification.get("producer_execution") != execution
        or producer_custody != expected_producer_custody
        or audit_execution != expected_audit_execution
    ):
        raise RuntimeError("development closure identity differs from archived evidence")
    return closure


def _formal_development_identity(
    closure: dict[str, object],
    *,
    closure_filename: str,
    closure_bytes: int,
    closure_sha256: str,
) -> dict[str, object]:
    """Project a verified v2 closure into a performance-free binding identity."""

    archive = cast(dict[str, object], closure["qualification_archive"])
    members = cast(list[dict[str, object]], archive["members"])
    member_digests = {str(row["path"]): str(row["sha256"]) for row in members}

    def member_digest(role: str) -> str:
        member = closure.get(role)
        if not isinstance(member, str) or member not in member_digests:
            raise RuntimeError(f"development closure role {role} is absent from the archive")
        return member_digests[member]

    source = cast(dict[str, object], closure["source"])
    producer_execution = cast(
        dict[str, object],
        closure["producer_execution"],
    )
    producer_custody = cast(
        dict[str, object],
        closure["producer_custody"],
    )
    audit_execution = cast(
        dict[str, object],
        closure["audit_execution"],
    )
    return {
        "closure_schema": closure["schema"],
        "closure_file": closure_filename,
        "closure_bytes": closure_bytes,
        "closure_sha256": closure_sha256,
        "qualification_archive_file": archive["file"],
        "qualification_archive_path": archive["canonical_path"],
        "qualification_archive_bytes": archive["bytes"],
        "qualification_archive_sha256": archive["sha256"],
        "qualification_archive_members_sha256": hashlib.sha256(canonical_json_bytes({"members": members})).hexdigest(),
        "producer_manifest_sha256": member_digest("producer_manifest_member"),
        "raw_result_sha256": member_digest("raw_result_member"),
        "result_qualification_sha256": member_digest("result_qualification_member"),
        "independent_audit_sha256": member_digest("independent_audit_member"),
        "audit_reproduction_sha256": member_digest("audit_reproduction_member"),
        "audit_runtime_manifest_sha256": member_digest("audit_runtime_manifest_member"),
        "audit_invocation_manifest_sha256": member_digest("audit_invocation_manifest_member"),
        "audit_stderr_sha256": member_digest("audit_stderr_member"),
        "source_identity_sha256": hashlib.sha256(canonical_json_bytes(source)).hexdigest(),
        "producer_execution_identity_sha256": hashlib.sha256(canonical_json_bytes(producer_execution)).hexdigest(),
        "producer_custody_identity_sha256": hashlib.sha256(canonical_json_bytes(producer_custody)).hexdigest(),
        "audit_execution_identity_sha256": hashlib.sha256(canonical_json_bytes(audit_execution)).hexdigest(),
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }


def create_development_closure(
    *,
    producer_root: Path,
    audit_path: Path,
    audit_reproduction_path: Path,
    runtime_manifest_path: Path,
    output_path: Path = DEVELOPMENT_CLOSURE_PATH,
) -> dict[str, object]:
    """Close the sole v1.6 qualification into one self-contained evidence archive."""

    from .artifact import verify_producer_manifest
    from .verify import DEVELOPMENT_SEEDS, _verify_formal_matrix, verify_result

    expected_output = DEVELOPMENT_CLOSURE_PATH
    if output_path != expected_output:
        raise RuntimeError("development closure can only be published at DEVELOPMENT_CLOSURE_PATH")
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    if (
        expected_output.parent.resolve(strict=True) != expected_output.parent
        or expected_output.resolve(strict=False) != expected_output
    ):
        raise RuntimeError("development closure destination is aliased")
    if os.path.lexists(expected_output):
        raise FileExistsError(f"refusing to replace development closure marker: {expected_output}")
    producer_root = producer_root if producer_root.is_absolute() else Path.cwd() / producer_root
    if producer_root.is_symlink() or not producer_root.is_dir() or producer_root.resolve(strict=True) != producer_root:
        raise RuntimeError("development producer root is missing or aliased")
    audit_path = audit_path if audit_path.is_absolute() else Path.cwd() / audit_path
    audit_reproduction_path = (
        audit_reproduction_path if audit_reproduction_path.is_absolute() else Path.cwd() / audit_reproduction_path
    )
    runtime_manifest_path = (
        runtime_manifest_path if runtime_manifest_path.is_absolute() else Path.cwd() / runtime_manifest_path
    )
    result_path = producer_root / "result.json"
    producer_manifest = verify_producer_manifest(producer_root)
    if (
        producer_manifest.get("experiment_id") != "WM-001"
        or producer_manifest.get("lane") != "development"
        or producer_manifest.get("status") != "completed"
        or producer_manifest.get("error") is not None
    ):
        raise RuntimeError("development qualification requires a completed producer manifest")
    result = verify_result(result_path, None)
    replicates = result.get("replicates")
    if (
        not isinstance(replicates, list)
        or tuple(row.get("master_seed") if isinstance(row, dict) else None for row in replicates) != DEVELOPMENT_SEEDS
    ):
        raise RuntimeError("development qualification requires exactly both fresh v1.6 seeds")
    for replicate in replicates:
        assert isinstance(replicate, dict)
        _verify_formal_matrix(
            replicate,
            replicate_id=str(replicate.get("replicate_id")),
        )
    audit = _load_canonical_json(audit_path, label="development independent audit")
    audit_payload = _stable_regular_payload(
        audit_path,
        label="development independent audit",
    )
    receipt_payload = _stable_regular_payload(
        audit_reproduction_path,
        label="development audit reproduction",
    )
    receipt = _canonical_json_object(
        receipt_payload,
        label="development audit reproduction",
    )
    if set(receipt) != _AUDIT_RECEIPT_FIELDS:
        raise RuntimeError("development audit reproduction receipt has the wrong schema")
    runtime_sidecar, runtime_manifest = _receipt_sidecar(
        audit_reproduction_path,
        receipt,
        prefix="runtime_manifest",
        label="development audit runtime manifest",
    )
    invocation_sidecar, invocation_manifest = _receipt_sidecar(
        audit_reproduction_path,
        receipt,
        prefix="invocation_manifest",
        label="development audit invocation manifest",
    )
    stderr_sidecar, audit_stderr = _receipt_sidecar(
        audit_reproduction_path,
        receipt,
        prefix="stderr",
        label="development audit stderr",
    )
    if runtime_manifest_path != runtime_sidecar or runtime_manifest_path.is_symlink():
        raise RuntimeError("development runtime manifest argument differs from the receipt sidecar")
    result_sha256 = sha256_file(result_path)
    execution = _validate_execution_identity(
        result.get("execution"),
        require_live_identity=True,
    )
    _validate_development_audit_evidence(
        producer_root=producer_root,
        result_sha256=result_sha256,
        execution=execution,
        audit_payload=audit_payload,
        receipt_payload=receipt_payload,
        runtime_payload=runtime_manifest,
        invocation_payload=invocation_manifest,
        stderr_payload=audit_stderr,
    )
    audit = _canonical_json_object(
        audit_payload,
        label="development independent audit",
    )
    if (
        result.get("lane") != "development"
        or audit.get("lane") != "development"
        or audit.get("result_sha256") != result_sha256
        or audit.get("integrity_passed") is not True
        or audit.get("engineering_complete") is not True
        or audit.get("complete_for_claim") is not False
        or audit.get("passed") is not True
        or receipt.get("supplied_audit_sha256") != hashlib.sha256(audit_payload).hexdigest()
        or receipt.get("byte_identical") is not True
        or receipt.get("passed") is not True
    ):
        raise RuntimeError("development evidence does not satisfy engineering qualification")
    if (
        execution["git_commit"] != git_output("rev-parse", "HEAD")
        or execution["git_tree"] != git_output("rev-parse", "HEAD^{tree}")
        or source_is_clean() is not True
    ):
        raise RuntimeError("development qualification source differs from the current clean commit")
    runtime_seal_payload = getattr(
        sys,
        "_prospect_wm001_runtime_seal_payload",
        None,
    )
    bootstrap_payload = getattr(
        sys,
        "_prospect_wm001_bootstrap_payload",
        None,
    )
    launch_bootstrap_payload = Path(__file__).with_name("launch_bootstrap.py").read_bytes()
    if not isinstance(runtime_seal_payload, bytes) or not isinstance(bootstrap_payload, bytes):
        raise RuntimeError("development closure requires captured producer seal/bootstrap bytes")
    producer_custody_identity = _validate_producer_custody(
        execution=execution,
        runtime_seal_payload=runtime_seal_payload,
        bootstrap_payload=bootstrap_payload,
        launch_bootstrap_payload=launch_bootstrap_payload,
    )
    result_qualification_payload = _result_qualification_payload(
        cast(dict[str, object], result),
        result_sha256=result_sha256,
        execution=execution,
    )
    evidence_payloads = {
        "evidence/independent-audit.json": audit_payload,
        "evidence/audit-reproduction.json": receipt_payload,
        "evidence/development-result-qualification.json": (result_qualification_payload),
        "evidence/producer-runtime-seal.json": runtime_seal_payload,
        "evidence/producer-bootstrap.py": bootstrap_payload,
        "evidence/launch-bootstrap.py": launch_bootstrap_payload,
        f"evidence/{runtime_sidecar.name}": runtime_manifest,
        f"evidence/{invocation_sidecar.name}": invocation_manifest,
        f"evidence/{stderr_sidecar.name}": audit_stderr,
    }
    archive_path, archive_identity = _write_qualification_archive(
        destination_directory=expected_output.parent,
        producer_root=producer_root,
        evidence_payloads=evidence_payloads,
    )
    closure = {
        "schema": "prospect.wm001.development-closure.v2",
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "source": {
            "git_commit": execution["git_commit"],
            "git_tree": execution["git_tree"],
            "worktree_clean": True,
            "dependency_lock_sha256": execution["dependency_lock_sha256"],
            "producer_bootstrap_sha256": execution["producer_bootstrap_sha256"],
            "launch_bootstrap_sha256": producer_custody_identity["launch_bootstrap_sha256"],
            "runner_source_sha256": receipt["runner_source_sha256"],
            "auditor_source_sha256": receipt["auditor_source_sha256"],
        },
        "producer_root": str(producer_root),
        "producer_manifest_member": "producer/producer-manifest.json",
        "raw_result_member": "producer/result.json",
        "result_qualification_member": ("evidence/development-result-qualification.json"),
        "independent_audit_member": "evidence/independent-audit.json",
        "audit_reproduction_member": "evidence/audit-reproduction.json",
        "audit_runtime_manifest_member": f"evidence/{runtime_sidecar.name}",
        "audit_invocation_manifest_member": f"evidence/{invocation_sidecar.name}",
        "audit_stderr_member": f"evidence/{stderr_sidecar.name}",
        "producer_execution": execution,
        "producer_custody": {
            "runtime_seal_member": "evidence/producer-runtime-seal.json",
            "runtime_seal_sha256": producer_custody_identity["runtime_seal_sha256"],
            "producer_bootstrap_member": "evidence/producer-bootstrap.py",
            "producer_bootstrap_sha256": producer_custody_identity["producer_bootstrap_sha256"],
            "launch_bootstrap_member": "evidence/launch-bootstrap.py",
            "launch_bootstrap_sha256": producer_custody_identity["launch_bootstrap_sha256"],
            "package_ownership": producer_custody_identity["package_ownership"],
        },
        "audit_execution": {
            "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "runtime_manifest_sha256": receipt["runtime_manifest_sha256"],
            "invocation_manifest_sha256": receipt["invocation_manifest_sha256"],
            "stderr_sha256": receipt["stderr_sha256"],
            "bootstrap_sha256": receipt["bootstrap_sha256"],
            "runner_source_sha256": receipt["runner_source_sha256"],
            "auditor_source_sha256": receipt["auditor_source_sha256"],
            "support_files": receipt["support_files"],
            "source_mode": "descriptor",
        },
        "qualification_archive": archive_identity,
        "engineering_verified": True,
        "audit_reproduced": True,
        "performance_values_bound": False,
    }
    closure_payload = canonical_json_bytes(closure) + b"\n"
    try:
        closure_digest = hashlib.sha256(closure_payload).hexdigest()
        with tempfile.TemporaryDirectory(
            prefix="prospect-wm001-development-closure-verify-",
        ) as temporary:
            prospective = Path(temporary) / (f"development-closure-{closure_digest[:16]}.json")
            atomic_write_exclusive(prospective, closure_payload)
            verify_development_closure(prospective)
        _reverify_live_development_producer(
            producer_root,
            expected_manifest=cast(
                dict[str, object],
                producer_manifest,
            ),
            expected_result_sha256=result_sha256,
        )
        atomic_write_exclusive(expected_output, closure_payload)
        if (
            expected_output.is_symlink()
            or expected_output.stat().st_nlink != 1
            or expected_output.resolve(strict=True) != expected_output
            or _stable_regular_payload(
                expected_output,
                label="published development closure",
            )
            != closure_payload
        ):
            raise RuntimeError("development closure publication is aliased")
    except BaseException:
        if not os.path.lexists(expected_output):
            archive_path.unlink(missing_ok=True)
        raise
    return closure


def combined_file_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_implementation_sha256() -> str:
    """Bind every source file that encodes, archives, or restores K7 state."""

    module_directory = Path(__file__).parent
    return combined_file_sha256([module_directory / filename for filename in CHECKPOINT_IMPLEMENTATION_SOURCES])


def _created_conformance_satisfies_formal_contract(
    report: dict[str, object],
) -> bool:
    expected_fields = {
        "seed": FORMAL_CONFORMANCE_SEED,
        "samples_per_task": FORMAL_CONFORMANCE_SAMPLES_PER_TASK,
        "cases": FORMAL_CONFORMANCE_CASES,
        "spec_horizon": 200,
        "terminated_or_truncated_cases": 0,
        "planner_dtype": "float32",
        **FORMAL_CONFORMANCE_TOLERANCES,
    }
    if (
        report.get("schema") != "prospect.wm001.pendulum-conformance.v1"
        or report.get("environment_id") != "Pendulum-v1"
        or report.get("passed") is not True
        or any(report.get(field) != expected for field, expected in expected_fields.items())
        or report.get("semantic_parameters") != FORMAL_CONFORMANCE_PARAMETERS
        or report.get("semantic_parameter_absolute_errors") != {name: 0.0 for name in FORMAL_CONFORMANCE_PARAMETERS}
    ):
        return False
    error_limits = {
        "max_observation_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["observation_atol"],
        "max_reward_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["reward_atol"],
        "max_planner_observation_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["planner_observation_atol"],
        "max_planner_reward_absolute_error": FORMAL_CONFORMANCE_TOLERANCES["planner_reward_atol"],
    }
    for field, limit in error_limits.items():
        value = report.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= limit
        ):
            return False
    body = dict(report)
    report_sha256 = body.pop("report_sha256", None)
    return isinstance(report_sha256, str) and report_sha256 == hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def create_formal_binding(
    *,
    output_path: Path,
    test_report_path: Path,
    development_closure_path: Path | None = None,
    conformance_cases: int = 1024,
    device: str,
) -> dict[str, object]:
    """Write a complete binding, refusing a dirty source tree."""

    python_flags = require_formal_python_flags()
    process_environment = require_formal_process_environment()
    from .verify import verify_protocol

    verify_protocol()
    if not source_is_clean():
        raise RuntimeError("formal implementation binding requires a clean committed worktree")
    if not LOCKFILE.is_file():
        raise RuntimeError("WM-001 lockfile is missing")
    if not test_report_path.is_file():
        raise RuntimeError("formal binding requires a preserved test report")
    if conformance_cases != FORMAL_CONFORMANCE_CASES:
        raise ValueError("formal Pendulum conformance is fixed at exactly 1,024 cases (512 per task)")
    if development_closure_path is None or not development_closure_path.is_file():
        raise RuntimeError("formal binding requires the immutable v1.6 development closure")
    if device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA formal binding requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if output_path.exists():
        raise FileExistsError(f"refusing to replace formal binding: {output_path}")
    conformance_value = run_pendulum_conformance(
        samples_per_task=FORMAL_CONFORMANCE_SAMPLES_PER_TASK,
        seed=FORMAL_CONFORMANCE_SEED,
        **FORMAL_CONFORMANCE_TOLERANCES,
    )
    if not _created_conformance_satisfies_formal_contract(conformance_value):
        raise RuntimeError("Pendulum semantic conformance did not satisfy the fixed formal contract")
    conformance_bytes = canonical_json_bytes(conformance_value) + b"\n"
    conformance_digest = hashlib.sha256(conformance_bytes).hexdigest()
    conformance_filename = f"pendulum-conformance-{conformance_digest[:16]}.json"
    conformance_path = output_path.with_name(conformance_filename)
    oscillator_conformance = run_independent_phase_oscillator_conformance(
        cases=FORMAL_OSCILLATOR_CONFORMANCE_CASES,
        seed=FORMAL_OSCILLATOR_CONFORMANCE_SEED,
    )
    if oscillator_conformance.get("passed") is not True:
        raise RuntimeError("independent oscillator conformance did not pass")
    oscillator_conformance_bytes = canonical_json_bytes(oscillator_conformance) + b"\n"
    oscillator_conformance_digest = hashlib.sha256(oscillator_conformance_bytes).hexdigest()
    oscillator_conformance_filename = f"oscillator-conformance-{oscillator_conformance_digest[:16]}.json"
    oscillator_conformance_path = output_path.with_name(oscillator_conformance_filename)
    coverage_conformance = run_coverage_conformance()
    if coverage_conformance.get("passed") is not True:
        raise RuntimeError("coverage arithmetic conformance did not pass")
    coverage_conformance_bytes = canonical_json_bytes(coverage_conformance) + b"\n"
    coverage_conformance_digest = hashlib.sha256(coverage_conformance_bytes).hexdigest()
    coverage_conformance_filename = f"coverage-conformance-{coverage_conformance_digest[:16]}.json"
    coverage_conformance_path = output_path.with_name(coverage_conformance_filename)
    test_report = verify_machine_test_report(test_report_path)
    test_log_rows = preformal_log_rows(test_report_path, test_report)
    development_closure = verify_development_closure(development_closure_path)
    test_report_bytes = _stable_regular_payload(
        test_report_path,
        label="formal binding test report",
    )
    if not test_report_bytes:
        raise RuntimeError("formal binding test report must not be empty")
    test_report_digest = hashlib.sha256(test_report_bytes).hexdigest()
    from .preformal import REPORT_NAME as PREFORMAL_REPORT_NAME

    test_report_filename = PREFORMAL_REPORT_NAME
    preserved_test_report_path = output_path.with_name(test_report_filename)
    development_closure_bytes = _stable_regular_payload(
        development_closure_path,
        label="formal binding development closure",
    )
    development_closure_digest = hashlib.sha256(development_closure_bytes).hexdigest()
    development_closure_filename = f"development-closure-{development_closure_digest[:16]}.json"
    preserved_development_closure_path = output_path.with_name(development_closure_filename)
    development_identity = _formal_development_identity(
        development_closure,
        closure_filename=development_closure_filename,
        closure_bytes=len(development_closure_bytes),
        closure_sha256=development_closure_digest,
    )
    for candidate in (
        conformance_path,
        oscillator_conformance_path,
        coverage_conformance_path,
        preserved_test_report_path,
        preserved_development_closure_path,
        *(output_path.with_name(str(row["path"])) for row in test_log_rows),
    ):
        if candidate.exists():
            raise FileExistsError(f"refusing to replace formal binding evidence: {candidate}")

    pendulum_module = importlib.import_module("gymnasium.envs.classic_control.pendulum")
    pendulum_source = Path(inspect.getsourcefile(pendulum_module) or "")
    wrapper_sources = [
        Path(__file__).with_name("runtime_lane.py"),
        Path(__file__).with_name("planning.py"),
        pendulum_source,
    ]
    torch.use_deterministic_algorithms(True)
    verify_installed_source_snapshot()
    roots = package_roots()
    packages = installed_package_rows()
    root_inventories = [package_root_inventory(root) for root in roots]
    stdlib_inventory = standard_library_inventory()
    ownership = package_root_ownership()
    verify_lockfile_rows(packages)
    audit_execution, audit_execution_payloads = build_bound_audit_execution(
        device=device,
        packages=packages,
        roots=roots,
        standard_library=stdlib_inventory,
        producer_environment=process_environment,
    )
    for filename in audit_execution_payloads:
        candidate = output_path.with_name(filename)
        if candidate.exists():
            raise FileExistsError(f"refusing to replace formal audit-execution evidence: {candidate}")
    accelerator = torch.cuda.get_device_name(0) if device == "cuda" else None
    binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v6",
        "experiment_id": "WM-001",
        "assurance": assurance_record(),
        "protocol": {
            "version": "1.6.0",
            "sha256": sha256_file(PROTOCOL_PATH),
            "raw_result_schema_sha256": sha256_file(RESULT_SCHEMA_PATH),
            "binding_schema_sha256": sha256_file(BINDING_SCHEMA_PATH),
        },
        "sealed_at_utc": __import__("datetime")
        .datetime.now(__import__("datetime").UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_tree": git_output("rev-parse", "HEAD^{tree}"),
            "worktree_clean": True,
            "implementation_files": implementation_files(),
            "execution_source_sha256": {
                filename: sha256_file(Path(__file__).with_name(filename)) for filename in EXECUTION_SOURCE_FILES
            },
            "test_report_file": test_report_filename,
            "test_report_bytes": len(test_report_bytes),
            "test_report_sha256": test_report_digest,
            "test_log_files": test_log_rows,
        },
        "dependencies": {
            "lockfile": LOCKFILE.relative_to(REPO).as_posix(),
            "lockfile_sha256": sha256_file(LOCKFILE),
            "python_executable": sys.executable,
            "python_executable_sha256": sha256_file(Path(sys.executable)),
            "standard_library": stdlib_inventory,
            "package_roots": root_inventories,
            "package_ownership": ownership,
            "packages": packages,
        },
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": device,
            "python_flags": python_flags,
            "process_environment": process_environment,
            "accelerator": accelerator,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thread_count": torch.get_num_threads(),
            "interop_thread_count": torch.get_num_interop_threads(),
            "cuda_runtime": torch.version.cuda,
            "cuda_driver": _cuda_driver_version() if device == "cuda" else None,
            "cublas_workspace_config": (os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None),
        },
        "environment": {
            "id": "Pendulum-v1",
            "wrapper_source_sha256": combined_file_sha256(wrapper_sources),
            "installed_distribution_sha256": distribution_sha256("gymnasium"),
            "conformance_report_file": conformance_filename,
            "conformance_report_bytes": len(conformance_bytes),
            "conformance_report_sha256": conformance_digest,
        },
        "irrelevant_control": {
            "id": "independent_phase_oscillator",
            "source_id": INDEPENDENT_OSCILLATOR_SOURCE,
            "source_sha256": sha256_file(Path(__file__).with_name("runtime_lane.py")),
            "conformance_report_file": oscillator_conformance_filename,
            "conformance_report_bytes": len(oscillator_conformance_bytes),
            "conformance_report_sha256": oscillator_conformance_digest,
        },
        "coverage_arithmetic": {
            "semantics_id": COVERAGE_SEMANTICS,
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "producer_source_sha256": sha256_file(Path(__file__).with_name("model.py")),
            "auditor_source_sha256": sha256_file(Path(__file__).with_name("artifact_audit.py")),
            "formal_test_report_sha256": test_report_digest,
            "conformance_report_file": coverage_conformance_filename,
            "conformance_report_bytes": len(coverage_conformance_bytes),
            "conformance_report_sha256": coverage_conformance_digest,
        },
        "development_qualification": development_identity,
        "audit_execution": audit_execution,
        "checkpoint_implementation": {
            "serializer_source_sha256": checkpoint_implementation_sha256(),
            "manifest_schema_sha256": manifest_schema_sha256(),
            "component_ids": [
                "world_model",
                "optimizer",
                "model_version_ledger",
                "experience_store",
                "replay_index",
                "replay_sampling_history",
                "update_receipts",
                "agent_runtime",
                "scaling_configuration",
                "python_rng",
                "numpy_rng",
                "torch_cpu_rng",
                "torch_accelerator_rng",
                "collection_rng",
                "planner_rng",
            ],
        },
        "formal_replicate_master_seeds": list(FORMAL_SEEDS),
    }
    if (
        not source_is_clean()
        or git_output("rev-parse", "HEAD") != binding["source"]["git_commit"]
        or git_output("rev-parse", "HEAD^{tree}") != binding["source"]["git_tree"]
    ):
        raise RuntimeError("source changed while the formal implementation binding was assembled")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_exclusive(preserved_test_report_path, test_report_bytes)
    for row in test_log_rows:
        origin = test_report_path.parent / str(row["path"])
        destination = output_path.with_name(str(row["path"]))
        atomic_write_exclusive(
            destination,
            _stable_regular_payload(
                origin,
                label="formal binding preformal log",
            ),
        )
    atomic_write_exclusive(
        preserved_development_closure_path,
        development_closure_bytes,
    )
    atomic_write_exclusive(conformance_path, conformance_bytes)
    atomic_write_exclusive(
        oscillator_conformance_path,
        oscillator_conformance_bytes,
    )
    atomic_write_exclusive(coverage_conformance_path, coverage_conformance_bytes)
    for filename, payload in audit_execution_payloads.items():
        atomic_write_exclusive(output_path.with_name(filename), payload)
    atomic_write_exclusive(output_path, canonical_json_bytes(binding) + b"\n")
    return binding


def _verified_formal_binding_file(path: Path) -> dict[str, object]:
    """Verify a formal binding whose role requires exactly one live name."""

    from .verify import verify_binding

    _stable_regular_payload(
        path,
        label="formal binding",
        expected_nlink=1,
    )
    return cast(dict[str, object], verify_binding(path))


def verify_live_binding(path: Path, *, device: str) -> dict[str, object]:
    """Reject launch unless the live source, packages, and runtime equal the seal."""

    binding = _verified_formal_binding_file(path)
    source = cast(dict[str, object], binding["source"])
    if source["implementation_files"] != implementation_files():
        raise RuntimeError("formal launch implementation manifest differs from the complete live manifest")
    expected_execution_sources = {
        filename: sha256_file(Path(__file__).with_name(filename)) for filename in EXECUTION_SOURCE_FILES
    }
    if source.get("execution_source_sha256") != expected_execution_sources:
        raise RuntimeError("formal launch execution-source identities differ from the binding")
    if not source_is_clean():
        raise RuntimeError("formal launch worktree is not clean")
    if git_output("rev-parse", "HEAD") != source["git_commit"]:
        raise RuntimeError("formal launch HEAD differs from its binding")
    if git_output("rev-parse", "HEAD^{tree}") != source["git_tree"]:
        raise RuntimeError("formal launch Git tree differs from its binding")
    require_formal_python_flags()
    require_formal_process_environment()
    verify_installed_source_snapshot()
    actual_packages = installed_package_rows()
    verify_lockfile_rows(actual_packages)
    dependencies = cast(dict[str, object], binding["dependencies"])
    if actual_packages != dependencies["packages"]:
        raise RuntimeError("formal launch installed package closure differs from its binding")
    actual_roots = [package_root_inventory(root) for root in package_roots()]
    if actual_roots != dependencies["package_roots"]:
        raise RuntimeError("formal launch complete package-root inventory differs from its binding")
    if package_root_ownership() != dependencies["package_ownership"]:
        raise RuntimeError("formal launch package ownership differs from its binding")
    if standard_library_inventory() != dependencies["standard_library"]:
        raise RuntimeError("formal launch standard-library inventory differs from its binding")
    if dependencies.get("python_executable") != sys.executable or dependencies.get(
        "python_executable_sha256"
    ) != sha256_file(Path(sys.executable)):
        raise RuntimeError("formal launch CPython executable differs from its binding")
    pendulum_module = importlib.import_module("gymnasium.envs.classic_control.pendulum")
    pendulum_source = Path(inspect.getsourcefile(pendulum_module) or "")
    wrapper_sources = [
        Path(__file__).with_name("runtime_lane.py"),
        Path(__file__).with_name("planning.py"),
        pendulum_source,
    ]
    environment = cast(dict[str, object], binding["environment"])
    if combined_file_sha256(wrapper_sources) != environment["wrapper_source_sha256"]:
        raise RuntimeError("formal launch environment wrapper sources differ from its binding")
    if distribution_sha256("gymnasium") != environment["installed_distribution_sha256"]:
        raise RuntimeError("formal launch Gymnasium distribution differs from its binding")
    checkpoint = cast(dict[str, object], binding["checkpoint_implementation"])
    if checkpoint_implementation_sha256() != checkpoint["serializer_source_sha256"]:
        raise RuntimeError("formal launch checkpoint implementation differs from its binding")
    runtime = cast(dict[str, object], binding["runtime"])
    actual_runtime = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "python_flags": python_flag_identity(),
        "process_environment": dict(sorted(os.environ.items())),
        "accelerator": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "thread_count": torch.get_num_threads(),
        "interop_thread_count": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _cuda_driver_version() if device == "cuda" else None,
        "cublas_workspace_config": (os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None),
    }
    if actual_runtime != runtime:
        raise RuntimeError("formal launch runtime differs from its binding")
    return binding


def _binding_evidence_bytes(
    binding_path: Path,
    block: dict[str, object],
    *,
    prefix: str,
) -> bytes:
    raw_filename = block.get(f"{prefix}_file")
    if not isinstance(raw_filename, str) or Path(raw_filename).name != raw_filename:
        raise RuntimeError(f"bound {prefix} filename is unsafe")
    evidence_path = binding_path.parent / raw_filename
    payload = _stable_regular_payload(
        evidence_path,
        label=f"bound {prefix}",
    )
    if (
        block.get(f"{prefix}_bytes") != len(payload)
        or block.get(f"{prefix}_sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise RuntimeError(f"bound {prefix} bytes changed")
    return payload


def run_bound_preflight_conformance(
    binding_path: Path,
    device: str,
) -> object:
    """Replay exactly one bound descriptor audit immediately before launch."""

    from .audit_runner import (
        bootstrap_source_bytes,
        captured_support_argument,
        run_captured_auditor,
    )
    from .experiment import IsolatedConformanceReports

    binding = _verified_formal_binding_file(binding_path)
    runtime = cast(dict[str, object], binding["runtime"])
    if runtime.get("device") != device:
        raise RuntimeError("preflight device differs from the bound runtime")
    audit = cast(dict[str, object], binding["audit_execution"])
    root = binding_path.parent
    source_root = root / "source" / "bench" / "world_model_lifecycle"
    auditor_source = source_root / "artifact_audit.py"
    request_path = root / str(audit["prebinding_request_file"])
    support_files = {
        "prebinding-request.json": request_path,
        "protocol.json": root / "protocol.json",
        **{
            name: source_root / name
            for name in (
                "learning.py",
                "model.py",
                "planning.py",
                "runtime_lane.py",
            )
        },
    }
    descriptor_runtime = _binding_evidence_bytes(
        binding_path,
        audit,
        prefix="prebinding_descriptor_runtime_manifest",
    )
    descriptor_invocation = _binding_evidence_bytes(
        binding_path,
        audit,
        prefix="prebinding_descriptor_invocation_manifest",
    )
    expected_report = _binding_evidence_bytes(
        binding_path,
        audit,
        prefix="prebinding_conformance_report",
    )
    expected_execution_receipt = _binding_evidence_bytes(
        binding_path,
        audit,
        prefix="prebinding_execution_receipt",
    )
    bound_bootstrap = _binding_evidence_bytes(
        binding_path,
        audit,
        prefix="bootstrap_source",
    )
    if bound_bootstrap != bootstrap_source_bytes():
        raise RuntimeError("preflight bootstrap differs from the bound source")
    producer_environment = cast(dict[str, str], runtime["process_environment"])
    execution = run_captured_auditor(
        auditor_source,
        auditor_arguments=(
            "--prebinding-conformance",
            captured_support_argument("prebinding-request.json"),
        ),
        support_files=support_files,
        closure_import_roots=package_roots(),
        source_mode="descriptor",
        working_directory=Path(str(audit["outcome_working_directory"])),
        environment=_audit_environment(producer_environment),
        runtime_manifest=descriptor_runtime,
        invocation_manifest=descriptor_invocation,
    )
    try:
        receipt = json.loads(expected_execution_receipt)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("bound prebinding execution receipt is unreadable") from error
    receipt_rows = receipt.get("executions") if isinstance(receipt, dict) else None
    repeat_count = audit.get("repeat_count")
    descriptor_rows = (
        receipt_rows[repeat_count:] if isinstance(receipt_rows, list) and type(repeat_count) is int else None
    )
    if (
        not isinstance(descriptor_rows, list)
        or not isinstance(repeat_count, int)
        or len(descriptor_rows) != repeat_count
        or not descriptor_rows
        or any(not isinstance(row, dict) for row in descriptor_rows)
    ):
        raise RuntimeError("bound prebinding receipt has no exact descriptor replay rows")
    expected_descriptor = dict(cast(dict[str, object], descriptor_rows[0]))
    expected_descriptor.pop("ordinal", None)
    if any(
        {key: value for key, value in cast(dict[str, object], row).items() if key != "ordinal"} != expected_descriptor
        for row in descriptor_rows
    ):
        raise RuntimeError("bound prebinding descriptor executions were not deterministic")
    observed_descriptor = {
        "source_mode": execution.source_mode,
        "returncode": execution.returncode,
        "stdout": {
            "bytes": len(execution.stdout),
            "sha256": hashlib.sha256(execution.stdout).hexdigest(),
        },
        "stderr": {
            "bytes": len(execution.stderr),
            "sha256": hashlib.sha256(execution.stderr).hexdigest(),
        },
        "runtime_manifest": {
            "bytes": len(execution.runtime_manifest),
            "sha256": execution.runtime_manifest_sha256,
        },
        "invocation_manifest": {
            "bytes": len(execution.invocation_manifest),
            "sha256": execution.invocation_manifest_sha256,
        },
        "bootstrap_sha256": execution.bootstrap_sha256,
        "auditor_source_sha256": execution.auditor_source_sha256,
        "support_files": [
            {
                "path": support.relative_path,
                "bytes": support.bytes,
                "sha256": support.sha256,
            }
            for support in execution.support_files
        ],
        "auditor_report_passed": execution.report.get("passed"),
    }
    if (
        execution.runtime_manifest != descriptor_runtime
        or execution.invocation_manifest != descriptor_invocation
        or execution.stdout != expected_report
        or observed_descriptor != expected_descriptor
    ):
        raise RuntimeError("launch-time descriptor replay differs from the binding")
    report = dict(execution.report)
    components = report.get("components")
    if not isinstance(components, dict):
        raise RuntimeError("launch-time conformance report has no component block")
    pendulum = components.get("pendulum")
    oscillator = components.get("oscillator")
    coverage = components.get("coverage")
    if not all(
        isinstance(component, dict) and component.get("passed") is True
        for component in (pendulum, oscillator, coverage)
    ):
        raise RuntimeError("launch-time semantic conformance component failed")
    return IsolatedConformanceReports(
        pendulum_conformance=cast(dict[str, Any], pendulum),
        oscillator_conformance=cast(dict[str, Any], oscillator),
        coverage_conformance=cast(dict[str, Any], coverage),
        runner_verification={
            "passed": True,
            "source_mode": "descriptor",
            "single_launch_replay": True,
            "matches_bound_prebinding_reports": True,
            "prebinding_repeat_count": repeat_count,
            "report_sha256": hashlib.sha256(execution.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(execution.stderr).hexdigest(),
        },
    )


def run_bound_outcome_audit(producer_root: Path) -> object:
    """Run the normal auditor with the exact pre-outcome descriptor runtime."""

    from .audit_runner import run_captured_auditor

    root = producer_root.resolve(strict=True)
    if root != producer_root or not root.is_dir() or root.is_symlink():
        raise RuntimeError("outcome audit producer root is absent or aliased")
    binding_path = root / "formal-binding.json"
    binding = _verified_formal_binding_file(binding_path)
    audit = cast(dict[str, object], binding["audit_execution"])
    runtime = cast(dict[str, object], binding["runtime"])
    runtime_manifest = _binding_evidence_bytes(
        binding_path,
        audit,
        prefix="outcome_runtime_manifest",
    )
    source = root / "source" / "bench" / "world_model_lifecycle" / "artifact_audit.py"
    return run_captured_auditor(
        source,
        auditor_arguments=(str(root),),
        support_files={
            "protocol.json": root / "protocol.json",
            "schemas/raw-result.schema.json": (root / "schemas" / "raw-result.schema.json"),
        },
        closure_import_roots=package_roots(),
        source_mode="descriptor",
        working_directory=Path(str(audit["outcome_working_directory"])),
        environment=_audit_environment(cast(dict[str, str], runtime["process_environment"])),
        runtime_manifest=runtime_manifest,
    )


def _cuda_driver_version() -> str | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    versions = sorted({row.strip() for row in completed.stdout.splitlines() if row.strip()})
    return ",".join(versions) if versions else None


__all__ = (
    "checkpoint_implementation_sha256",
    "create_formal_binding",
    "dependency_closure",
    "distribution_sha256",
    "installed_package_rows",
    "implementation_files",
    "preformal_log_rows",
    "run_bound_outcome_audit",
    "run_bound_preflight_conformance",
    "sha256_file",
    "source_is_clean",
    "verify_live_binding",
)
