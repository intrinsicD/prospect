#!/usr/bin/env python3
"""CLI for the sealed WM-001 development and formal lanes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple


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
        raise RuntimeError("WM-001 runner requires an explicit Git worktree")
    candidate = Path(completed.stdout.strip())
    if (
        not candidate.is_absolute()
        or candidate.resolve(strict=True) != candidate
        or not (candidate / ".git").exists()
        or not (
            candidate
            / "bench"
            / "world_model_lifecycle"
            / "protocol.json"
        ).is_file()
    ):
        raise RuntimeError("WM-001 runner Git worktree is absent or aliased")
    return candidate


REPO = _repository_root()
DEVELOPMENT_RESULTS_ROOT = (
    REPO / "bench" / "world_model_lifecycle" / "results" / "development"
)
DEVELOPMENT_QUALIFICATION_PATH = (
    DEVELOPMENT_RESULTS_ROOT / "qualification-v1.19.0"
)
DEVELOPMENT_CLOSURE_PATH = (
    DEVELOPMENT_RESULTS_ROOT / "development-closure-v1.19.0.json"
)
DEVELOPMENT_DIAGNOSTICS_ROOT = (
    DEVELOPMENT_RESULTS_ROOT / "diagnostics-v1.19.0"
)


class _DevelopmentLifecyclePaths(NamedTuple):
    results_root: Path
    qualification: Path
    closure: Path
    diagnostics: Path


def _development_lifecycle_paths() -> _DevelopmentLifecyclePaths:
    """Derive every coupled development path from one injectable root."""

    root = DEVELOPMENT_RESULTS_ROOT
    return _DevelopmentLifecyclePaths(
        results_root=root,
        qualification=root / "qualification-v1.19.0",
        closure=root / "development-closure-v1.19.0.json",
        diagnostics=root / "diagnostics-v1.19.0",
    )


def _ensure_deterministic_cuda_environment() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("WM-001 requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")


def _development_output(
    supplied_output: Path | None,
    *,
    seed_override: bool,
    diagnostic_stamp: str,
) -> Path:
    paths = _development_lifecycle_paths()
    if not seed_override:
        if (
            supplied_output is not None
            and supplied_output != paths.qualification
        ):
            raise ValueError(
                "the complete no-override development run requires the sole "
                f"qualification path {paths.qualification}"
            )
        return paths.qualification
    if supplied_output is not None:
        raise ValueError(
            "a seed-override diagnostic cannot select an output path; "
            "diagnostics are confined to their version-owned namespace"
        )
    return (
        paths.diagnostics / f"diagnostic-{diagnostic_stamp}"
    )


def main() -> int:
    _ensure_deterministic_cuda_environment()
    import torch

    from .artifact import (
        FORMAL_RESULTS_ROOT,
        MANIFEST_NAME,
        ProducerAttempt,
        _verify_producer_manifest_precommit,
        formal_launch_marker_path,
    )
    from .experiment import (
        ExperimentConfig,
        _verify_live_bootstrap_custody,
        run_experiment,
    )
    from .operator import (
        FORMAL_BINDING_ATTEMPT_PATH,
        verify_operator_attempt,
    )
    from .producer_bootstrap import register_outer_terminal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("development", "formal"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--master-seed",
        type=int,
        action="append",
        help="development-only subset of the two declared diagnostic seeds",
    )
    arguments = parser.parse_args()
    existing_launch_marker: Path | None = None
    if arguments.lane == "formal":
        if arguments.master_seed:
            parser.error("formal lane cannot override the eight sealed master seeds")
        if arguments.binding is None:
            parser.error("formal lane requires --binding")
        if arguments.output is None:
            parser.error(
                "formal lane requires --output at the exact "
                "results/formal/<binding-sha256>/confirmation-v1.19.0 path"
            )
        expected_binding = FORMAL_BINDING_ATTEMPT_PATH / "formal-binding.json"
        if (
            not arguments.binding.is_absolute()
            or Path(os.path.abspath(arguments.binding)) != arguments.binding
            or arguments.binding.resolve(strict=False) != arguments.binding
            or arguments.binding != expected_binding
        ):
            parser.error(f"formal lane requires the canonical accepted binding-attempt file {expected_binding}")
        binding_attempt = verify_operator_attempt(FORMAL_BINDING_ATTEMPT_PATH)
        binding_primary = binding_attempt.get("primary")
        if (
            binding_attempt.get("kind") != "binding"
            or binding_attempt.get("lane") is not None
            or binding_attempt.get("status") != "accepted"
            or not isinstance(binding_primary, dict)
            or binding_primary.get("binding_file") != "formal-binding.json"
        ):
            parser.error("formal lane requires one accepted outer-finalized binding attempt")
        config = ExperimentConfig.formal(device=arguments.device)
        output = arguments.output
        try:
            existing_launch_marker, _ = formal_launch_marker_path(
                arguments.binding,
                output,
                formal_results_root=FORMAL_RESULTS_ROOT,
            )
        except ValueError as error:
            parser.error(str(error))
        if os.path.lexists(existing_launch_marker):
            print(
                "WM-001 protocol 1.19 formal launch already consumed; same-version retry is forbidden",
                file=sys.stderr,
            )
            return 1
    else:
        development_paths = _development_lifecycle_paths()
        if os.path.lexists(development_paths.qualification):
            print(
                "WM-001 protocol 1.19 development qualification already consumed; "
                "resume and sibling attempts are forbidden",
                file=sys.stderr,
            )
            return 1
        if os.path.lexists(development_paths.closure):
            print(
                "WM-001 protocol 1.19 development is closed; additional same-version rehearsals are forbidden",
                file=sys.stderr,
            )
            return 1
        seeds = arguments.master_seed or None
        config = (
            ExperimentConfig.development(device=arguments.device)
            if seeds is None
            else ExperimentConfig.development(
                master_seeds=seeds,
                device=arguments.device,
            )
        )
        stamp = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
        try:
            output = _development_output(
                arguments.output,
                seed_override=seeds is not None,
                diagnostic_stamp=stamp,
            )
        except ValueError as error:
            parser.error(str(error))
    config.validate()
    try:
        _verify_live_bootstrap_custody()
    except Exception as error:
        print(
            f"WM-001 launch refused before producer-root creation: {error}",
            file=sys.stderr,
        )
        return 1
    if (
        arguments.lane == "development"
        and arguments.master_seed
    ):
        diagnostic_root = _development_lifecycle_paths().diagnostics
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        if diagnostic_root.resolve(strict=True) != diagnostic_root:
            raise RuntimeError(
                "WM-001 diagnostic namespace is aliased"
            )
    attempt = ProducerAttempt(output, lane=arguments.lane)
    exit_code = 0
    try:
        with attempt:
            binding_for_run: Path | None = None
            if arguments.lane == "formal":
                if arguments.binding is None:
                    raise RuntimeError("formal binding path disappeared after argument validation")
                binding_for_run = attempt.preserve_formal_inputs(arguments.binding)
                torch.use_deterministic_algorithms(True)
                metadata_path = attempt.output_directory / "attempt-metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(metadata, dict)
                    or metadata.get("schema") != "prospect.wm001.producer-attempt.v1"
                    or metadata.get("lane") != "formal"
                    or (attempt.output_directory / "producer-manifest.json").exists()
                ):
                    raise RuntimeError("formal producer directory failed final preclaim custody validation")
            _, result_path = run_experiment(
                config,
                output_directory=attempt.output_directory,
                formal_binding_path=binding_for_run,
                output_prepared=True,
            )
            _verify_live_bootstrap_custody()
            print(result_path, flush=True)
    except KeyboardInterrupt:
        exit_code = 1
    except Exception as error:
        if not (attempt.output_directory / "producer-manifest.json").is_file():
            print(f"WM-001 launch failed before attempt custody: {error}", file=sys.stderr)
        exit_code = 1
    terminal = attempt.output_directory / MANIFEST_NAME
    if terminal.is_file():
        _verify_producer_manifest_precommit(attempt.output_directory)
        register_outer_terminal(
            terminal,
            logical_exit_code=exit_code,
        )
    elif exit_code == 0:
        raise RuntimeError("successful WM-001 invocation emitted no producer manifest")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
