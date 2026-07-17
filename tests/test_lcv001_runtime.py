"""Focused runtime closure tests for LCV-001."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from bench.sealed_lineage_verifier import experiment, runtime_probe


def test_full_numpy_and_stdlib_closure_manifests_match_frozen_aggregates() -> None:
    assert runtime_probe.dependency_closures() == {
        "numpy": {
            "bytes": 56_351_948,
            "count": 893,
            "manifest_sha256": "e6aa4a903960766e1b227e6717957d0a62cd8022b7953a38db17a4950b242422",
        },
        "stdlib": {
            "bytes": 34_805_888,
            "count": 916,
            "manifest_sha256": "6275b55cbe4b2a7453542bd2d2f91662bb355c48b4792925d7daeaf0c2ef0f59",
        },
    }


def test_per_file_runtime_manifest_mutation_is_rejected() -> None:
    manifests = runtime_probe.dependency_manifests()
    mutated = cast(dict[str, Any], copy.deepcopy(manifests))
    first = next(iter(mutated["numpy"]["records"]))
    mutated["numpy"]["records"][first]["sha256"] = "0" * 64
    with pytest.raises(runtime_probe.RuntimeClosureError):
        runtime_probe.validate_dependency_manifests(mutated)


def test_frozen_environment_is_exact_and_has_single_thread_policy() -> None:
    environment = runtime_probe.frozen_environment()
    assert environment["HOME"] == "/nonexistent"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert all(environment[name] == "1" for name in runtime_probe.THREAD_ENVIRONMENT)


def test_external_config_is_canonical_code_config() -> None:
    payload, _ = experiment._stable_read(experiment.CONFIG_DOC)
    assert json.loads(payload) == experiment._config_payload()


def test_svd_canary_input_is_thread_independent() -> None:
    # The current process need not have the formal thread policy, but integer
    # generation/centering/RMS normalization is frozen before LAPACK executes.
    value = runtime_probe.svd_canary()
    assert value["input_sha256"] == runtime_probe.CANARY_INPUT_SHA256
    assert value["shape"] == [512, 256]
