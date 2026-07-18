"""Create the second-stage implementation binding required before formal WM-001."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import math
import os
import platform
import struct
import subprocess
import sys
from functools import cache
from pathlib import Path

import torch
from packaging.markers import default_environment
from packaging.requirements import Requirement

from .artifact import atomic_write_exclusive
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
    SEAL_PATH,
)

REPO = Path(__file__).resolve().parents[2]
LOCKFILE = REPO / "requirements-wm001.lock"
ROOT_DISTRIBUTIONS = (
    "gymnasium",
    "jsonschema",
    "numpy",
    "torch",
    "torchrl",
    "tensordict",
    "prospect",
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
    log_variances = tuple(
        _float32_from_little_endian_hex(value)
        for value in _V130_BOUNDARY_LOG_VARIANCES_F32_HEX
    )
    pit = canonical_binary64_mixture_pit(target, means, log_variances)
    observed = binary64_pit_is_in_inclusive_90_percent_interval(pit)
    regression_passed = pit.hex() == _V130_BOUNDARY_EXPECTED_PIT_HEX and observed is False
    cases.append(
        {
            "case_id": "v130-disclosed-boundary-coordinate",
            "kind": "float32_mixture_inputs",
            "provenance": (
                "v1.3 formal seed 3332986400, task-A corrupted, sidecar row 207, "
                "target dimension 0; diagnostic only"
            ),
            "target_little_endian_f32_hex": _V130_BOUNDARY_TARGET_F32_HEX,
            "member_means_little_endian_f32_hex": list(_V130_BOUNDARY_MEANS_F32_HEX),
            "member_log_variances_little_endian_f32_hex": list(
                _V130_BOUNDARY_LOG_VARIANCES_F32_HEX
            ),
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
        SEAL_PATH,
        PROTOCOL_PATH,
        RESULT_SCHEMA_PATH,
        BINDING_SCHEMA_PATH,
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


@cache
def distribution_sha256(name: str) -> str:
    """Hash every RECORD-addressed installed file for one distribution."""

    distribution = importlib.metadata.distribution(name)
    digest = hashlib.sha256()
    canonical_name = str(distribution.metadata["Name"]).lower().replace("_", "-")
    digest.update(canonical_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(distribution.version.encode("utf-8"))
    declared_files = list(distribution.files or ())
    addressed_files = [entry for entry in declared_files if entry.hash is not None]
    selected_files = addressed_files or [
        entry for entry in declared_files if "__pycache__" not in entry.parts and entry.suffix != ".pyc"
    ]
    if not selected_files:
        raise RuntimeError(f"installed distribution {name!r} has no stable declared files")
    digest.update(b"\0record-addressed\0" if addressed_files else b"\0all-stable-declared\0")
    for entry in sorted(selected_files, key=str):
        path = distribution.locate_file(entry)
        if not path.is_file():
            raise RuntimeError(f"installed distribution file is missing: {entry}")
        digest.update(b"\0")
        digest.update(str(entry).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def dependency_closure() -> tuple[str, ...]:
    """Return the marker-resolved installed closure of the execution roots."""

    environment = default_environment()
    pending = list(ROOT_DISTRIBUTIONS)
    discovered: dict[str, str] = {}
    while pending:
        requested = pending.pop()
        distribution = importlib.metadata.distribution(requested)
        canonical = str(distribution.metadata["Name"]).lower().replace("_", "-")
        if canonical in discovered:
            continue
        discovered[canonical] = distribution.version
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            marker = requirement.marker
            if marker is not None and "extra" in str(marker):
                continue
            if marker is None or marker.evaluate(environment):
                pending.append(requirement.name)
    return tuple(sorted(discovered))


def installed_package_rows() -> list[dict[str, object]]:
    return [
        {
            "name": "python",
            "version": platform.python_version(),
            "distribution_sha256": sha256_file(Path(sys.executable)),
        },
        *[
            {
                "name": name,
                "version": importlib.metadata.version(name),
                "distribution_sha256": distribution_sha256(name),
            }
            for name in dependency_closure()
        ],
    ]


def verify_lockfile_rows(packages: list[dict[str, object]]) -> None:
    expected = {
        (
            str(row["name"]),
            str(row["version"]),
            str(row["distribution_sha256"]),
        )
        for row in packages
    }
    parsed: set[tuple[str, str, str]] = set()
    for line in LOCKFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            identity, digest = stripped.split(" --distribution-sha256=", 1)
            name, version = identity.split("==", 1)
        except ValueError as error:
            raise RuntimeError(f"malformed WM-001 lock row: {stripped!r}") from error
        parsed.add((name, version, digest))
    if parsed != expected or len(parsed) != len(expected):
        raise RuntimeError("WM-001 lockfile differs from the live installed dependency closure")


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
    conformance_cases: int = 1024,
    device: str,
) -> dict[str, object]:
    """Write a complete binding, refusing a dirty source tree."""

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
    test_report_bytes = test_report_path.read_bytes()
    if not test_report_bytes:
        raise RuntimeError("formal binding test report must not be empty")
    test_report_digest = hashlib.sha256(test_report_bytes).hexdigest()
    test_report_suffix = test_report_path.suffix or ".txt"
    test_report_filename = f"formal-test-report-{test_report_digest[:16]}{test_report_suffix}"
    preserved_test_report_path = output_path.with_name(test_report_filename)
    for candidate in (
        conformance_path,
        oscillator_conformance_path,
        coverage_conformance_path,
        preserved_test_report_path,
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
    packages = installed_package_rows()
    verify_lockfile_rows(packages)
    accelerator = torch.cuda.get_device_name(0) if device == "cuda" else None
    binding = {
        "schema": "prospect.world-model-lifecycle.formal-binding.v4",
        "experiment_id": "WM-001",
        "protocol": {
            "version": "1.4.0",
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
            "test_report_file": test_report_filename,
            "test_report_bytes": len(test_report_bytes),
            "test_report_sha256": test_report_digest,
        },
        "dependencies": {
            "lockfile": LOCKFILE.relative_to(REPO).as_posix(),
            "lockfile_sha256": sha256_file(LOCKFILE),
            "packages": packages,
        },
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": device,
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
    atomic_write_exclusive(conformance_path, conformance_bytes)
    atomic_write_exclusive(
        oscillator_conformance_path,
        oscillator_conformance_bytes,
    )
    atomic_write_exclusive(coverage_conformance_path, coverage_conformance_bytes)
    atomic_write_exclusive(output_path, canonical_json_bytes(binding) + b"\n")
    return binding


def verify_live_binding(path: Path, *, device: str) -> dict[str, object]:
    """Reject launch unless the live source, packages, and runtime equal the seal."""

    from .verify import verify_binding

    binding = verify_binding(path)
    source = binding["source"]
    if source["implementation_files"] != implementation_files():
        raise RuntimeError("formal launch implementation manifest differs from the complete live manifest")
    if not source_is_clean():
        raise RuntimeError("formal launch worktree is not clean")
    if git_output("rev-parse", "HEAD") != source["git_commit"]:
        raise RuntimeError("formal launch HEAD differs from its binding")
    if git_output("rev-parse", "HEAD^{tree}") != source["git_tree"]:
        raise RuntimeError("formal launch Git tree differs from its binding")
    actual_packages = installed_package_rows()
    verify_lockfile_rows(actual_packages)
    if actual_packages != binding["dependencies"]["packages"]:
        raise RuntimeError("formal launch installed package closure differs from its binding")
    pendulum_module = importlib.import_module("gymnasium.envs.classic_control.pendulum")
    pendulum_source = Path(inspect.getsourcefile(pendulum_module) or "")
    wrapper_sources = [
        Path(__file__).with_name("runtime_lane.py"),
        Path(__file__).with_name("planning.py"),
        pendulum_source,
    ]
    environment = binding["environment"]
    if combined_file_sha256(wrapper_sources) != environment["wrapper_source_sha256"]:
        raise RuntimeError("formal launch environment wrapper sources differ from its binding")
    if distribution_sha256("gymnasium") != environment["installed_distribution_sha256"]:
        raise RuntimeError("formal launch Gymnasium distribution differs from its binding")
    if checkpoint_implementation_sha256() != binding["checkpoint_implementation"]["serializer_source_sha256"]:
        raise RuntimeError("formal launch checkpoint implementation differs from its binding")
    runtime = binding["runtime"]
    actual_runtime = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
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
    "sha256_file",
    "source_is_clean",
    "verify_live_binding",
)
