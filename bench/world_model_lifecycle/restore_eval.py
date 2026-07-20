#!/usr/bin/env python3
"""Fresh-interpreter entry point for WM-001 restart parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch

from .artifact import atomic_write_exclusive
from .binding import FORMAL_PROCESS_ENVIRONMENT_KEYS, python_flag_identity
from .checkpoint import canonical_json_bytes
from .parity import evaluate_checkpoint, save_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-identity-output", required=True)
    arguments = parser.parse_args()
    specification = json.loads(Path(arguments.spec).read_text(encoding="utf-8"))
    if (
        not isinstance(specification, dict)
        or set(specification) != {"schema", "task_reset_seeds", "device"}
        or specification.get("schema") != "prospect.wm001.restart-spec.v1"
    ):
        raise ValueError("unsupported restart specification")
    torch.use_deterministic_algorithms(True)
    runtime_seal_payload = getattr(
        sys,
        "_prospect_wm001_runtime_seal_payload",
        None,
    )
    runtime_seal_sha256 = getattr(
        sys,
        "_prospect_wm001_runtime_seal_sha256",
        None,
    )
    bootstrap_sha256 = getattr(
        sys,
        "_prospect_wm001_bootstrap_sha256",
        None,
    )
    if (
        not isinstance(runtime_seal_payload, bytes)
        or not isinstance(runtime_seal_sha256, str)
        or not isinstance(bootstrap_sha256, str)
    ):
        raise RuntimeError("restore evaluator lacks pre-import bootstrap custody")
    runtime_seal = json.loads(runtime_seal_payload)
    if not isinstance(runtime_seal, dict):
        raise RuntimeError("restore evaluator received a malformed runtime seal")
    closure_block = (
        runtime_seal.get("dependencies", {})
        if runtime_seal.get("schema")
        == "prospect.world-model-lifecycle.formal-binding.v7"
        else runtime_seal
    )
    if not isinstance(closure_block, dict):
        raise RuntimeError("restore evaluator received a malformed closure block")
    package_roots = closure_block.get("package_roots")
    package_ownership = closure_block.get("package_ownership")
    standard_library = closure_block.get("standard_library")
    if (
        not isinstance(package_roots, list)
        or len(package_roots) != 1
        or not isinstance(package_roots[0], dict)
        or not isinstance(package_ownership, dict)
        or not isinstance(standard_library, dict)
    ):
        raise RuntimeError("restore evaluator received a malformed runtime seal")
    runtime_identity = {
        "schema": "prospect.wm001.restart-runtime.v2",
        "python_executable": sys.executable,
        "python_executable_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "python_flags": python_flag_identity(),
        "process_environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key in FORMAL_PROCESS_ENVIRONMENT_KEYS
        },
        "package_root": sys.path[0],
        "package_root_inventory": package_roots[0],
        "package_ownership": package_ownership,
        "standard_library": standard_library,
        "runtime_seal_sha256": runtime_seal_sha256,
        "runtime_seal_descriptor_custody": True,
        "bootstrap_source_sha256": bootstrap_sha256,
        "bootstrap_descriptor_custody": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    atomic_write_exclusive(
        Path(arguments.runtime_identity_output),
        canonical_json_bytes(runtime_identity) + b"\n",
    )
    result = evaluate_checkpoint(
        arguments.checkpoint,
        task_reset_seeds=specification["task_reset_seeds"],
        device=str(specification["device"]),
    )
    save_evaluation(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
