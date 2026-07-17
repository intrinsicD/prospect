"""VP-001: held-out validation of SS-001's frozen rank-panel winners.

VP-001 trains only untouched validator restarts.  It scores byte-pinned candidate
unions from SS-001, reconstructs the already-selected B0/C/X identities, and never
generates or selects a candidate.  Cross-model decisions use within-validator ranks.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

from bench.bridge_control.experiment import _train_model as train_balanced_model
from bench.bridge_control.fixture import EVAL_STARTS, load_dataset, semantic_hash
from bench.candidate_landscape import experiment as landscape
from bench.oracle_ladder import experiment as oracle
from bench.oracle_ladder.audit import average_tie_ranks, exact_discounted_scores
from bench.proposal_injection import experiment as injection
from bench.scorer_swap import experiment as scorer

SCHEMA_VERSION = "validator-panel-v1"
EXPERIMENT_ID = "VP-001"
VALIDATOR_SEEDS = tuple(range(44, 56))
PARENT_SOURCE_SEEDS = tuple(range(8, 20))
FRESH_SOURCE_SEEDS = tuple(range(32, 44))
DEVELOPMENT_SEED = 97
IDENTITY_NAMES = ("B0", "C", "X")
CALL_VOTE_FLOOR = 9
DIRECTION_SEED_FLOOR = 5
SENSITIVITY_SEED_FLOOR = 9
WITHIN_SEED_FRACTION = 0.75
EXPECTED_UNIQUE_CANDIDATES = 186

TARGET_CALLS = ((8, 0), (10, 0), (11, 2))
DIRECTION_CALLS = (
    (8, 0),
    (8, 1),
    (9, 0),
    (10, 0),
    (10, 3),
    (11, 0),
    (11, 2),
    (12, 2),
    (12, 3),
    (18, 0),
    (18, 3),
)
SENSITIVITY_CALLS = (
    (32, 1),
    (32, 2),
    (32, 3),
    (33, 2),
    (33, 3),
    (35, 0),
    (35, 2),
    (35, 3),
    (36, 2),
    (36, 3),
    (37, 0),
    (38, 1),
    (38, 3),
    (39, 1),
    (39, 3),
    (40, 0),
    (40, 1),
    (40, 2),
    (40, 3),
    (41, 0),
    (41, 2),
    (42, 1),
    (42, 2),
    (42, 3),
    (43, 2),
)
IDENTITY_SENTINEL_CALLS = ((9, 0), (12, 2), (12, 3), (18, 0), (18, 3))
SUBMATERIAL_CALLS = ((8, 1), (10, 3))
POSITIVE_X_DESCRIPTIVE_CALL = (11, 0)
POSITIVE_X_DESCRIPTIVE_EXACT_DELTA = 29.460094544989907
POSITIVE_X_DESCRIPTIVE_EXACT_RANK_GAIN = 131.0

TARGET_IDENTITIES: tuple[dict[str, object], ...] = (
    {
        "seed": 8,
        "episode_index": 0,
        "b0_union_index": 57,
        "c_union_index": 126,
        "x_union_index": 150,
        "b0_sequence_sha256": "b1d2403da558b395a433fb8eb84fed510057fc23b2aec528721ca281f414baab",
        "c_sequence_sha256": "ad79689698e9ea6212ba4fa61db28d589a1911c116eae2637482812169be95b5",
        "x_sequence_sha256": "a14d461c94421331d9f150630ad1fd4fcca25efe09baeaadbaaf62ab24e7bee5",
        "x_exact_delta_from_c": -9.224063279703334,
        "c_exact_rank": 4.0,
        "x_exact_rank": 118.0,
    },
    {
        "seed": 10,
        "episode_index": 0,
        "b0_union_index": 57,
        "c_union_index": 168,
        "x_union_index": 131,
        "b0_sequence_sha256": "d10a568dd2d48497b48ec071d2908e14dd9c460ec145efa6d552a3b3e55e1668",
        "c_sequence_sha256": "50db854c122933b1b94c56fad20f11d092eaf19777b7626c730d8fabc50eb5bf",
        "x_sequence_sha256": "80e4ddb4907ffca295ff7aac330471102cf89db0df97123e92d125a5ed90428e",
        "x_exact_delta_from_c": -0.515440530968359,
        "c_exact_rank": 2.0,
        "x_exact_rank": 45.0,
    },
    {
        "seed": 11,
        "episode_index": 2,
        "b0_union_index": 57,
        "c_union_index": 150,
        "x_union_index": 140,
        "b0_sequence_sha256": "4677d3185f6141462affb092e2f7bdc58a31c73ba1107065cf9e15c83476c962",
        "c_sequence_sha256": "e10fbfbb6eb33e556f7bcaceb30615464224a33394cd0fd4b3964340ac24c895",
        "x_sequence_sha256": "43acb785c92b80ef1dfe8a2a29cc5b8e9e76dc7c72a1416278055a760eef81a5",
        "x_exact_delta_from_c": -0.4131149418508384,
        "c_exact_rank": 2.0,
        "x_exact_rank": 47.0,
    },
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/validator_panel/results/VP-001")
PROTOCOL_DOC = Path("docs/research/2026-07-15-validator-panel-vp001-protocol.md")
PARENT_OUTPUT = Path("bench/scorer_swap/results/SS-001")
PARENT_COPY = Path("inputs/SS-001")
PARENT_TENSOR_COPY = PARENT_COPY / scorer.TENSOR_FILE
PARENT_CL_TENSOR_COPY = PARENT_COPY / scorer.PARENT_TENSOR_COPY
DATASET_COPY = PARENT_COPY / scorer.DATASET_COPY
TENSOR_FILE = Path(f"{EXPERIMENT_ID}-validators.npz")
RESULT_FILE = Path(f"{EXPERIMENT_ID}-results.json")
CSV_FILE = Path(f"{EXPERIMENT_ID}-calls.csv")
REPORT_FILE = Path(f"{EXPERIMENT_ID}-report.md")
STARTED_FILE = Path("formal-start.json")

PARENT_HASHES = {
    "SS-001-audit.npz": "4b6ff015aa0ef881031f430f7382002b423eb24ba4b1f492aa41642caadf954e",
    "SS-001-calls.csv": "bc155667142d5a604e256a938b5efb7b719f979343c098ec5169f49d6951cbe5",
    "SS-001-report.md": "c42375e644d2bbd5c1210dcd4c577d903663f9356f162a54a96fb147611e6837",
    "SS-001-results.json": "a9e4356b3a5b055f2fb6d85958125e2a8d0c600b5ca5f85286078e64f4e3a93c",
    "artifact-manifest.json": "89358a83821161f504d0238426bcca27891093cd9395099ea0230a8593c60c43",
    "formal-start.json": "d15669000f68fc1d6562408b06e6317c72f0b26838736cbdf44f19ec120b84ad",
    "input-manifest.json": "9731fb5e98f1d56130b9884eeb7f986fc766efdca1afc3d671f44e0c17649ef8",
    "inputs/CL-001/CL-001-calls.csv": "dbba501faf2b7ab7f1bfc21a324ca6c9c3c1a75c0fbdf6e4d0a0f0f49bed1333",
    "inputs/CL-001/CL-001-candidates.npz": "3ba92ad46f7ea084e6a019374d6fa98cd96c2fd6d0cd0dabdc7a7671ed3f0c48",
    "inputs/CL-001/CL-001-report.md": "b92f35139179593de6c8d0a6f4c5f79b81035c11372af28e40d193062e558783",
    "inputs/CL-001/CL-001-results.json": "97c26b1bc1125fd7fb9046e2382f972c5e2efe9cb7e6763cc37a52e6b1664746",
    "inputs/CL-001/artifact-manifest.json": "115eda81dd4411dbfb54ed630decc0cac2cf7bde77ca568eb7ffd0019593795d",
    "inputs/CL-001/formal-start.json": "ff7b0f6bbfa20ffd6c21c3e91577e7df22fb4ebc57cd293d617f25f876e3937c",
    "inputs/CL-001/input-manifest.json": "1d98d5c541d50b5c91fb10471f79b3e48bad401483d96b951df9f86e0fb3902c",
    "inputs/CL-001/inputs/BC-001-b1_r1_d8.npz": "9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948",
    "inputs/CL-001/protocol.json": "f9b92042341119ac99043d19cae67c7630d05b4be8d95ca8ccfe4dd7d0d03817",
    "protocol.json": "4f5f5ee4828e70bde4bf1b576293633e742f5b875c1b95755b56ab0c15007d3b",
}
PARENT_PROTOCOL_CANONICAL_SHA256 = "a2763eab15206c5491d9942c0faeec8a3110928c07d12a87651816b75959f3bc"
PARENT_TENSOR_SEMANTIC_SHA256 = "af62038891864dc03fd887fb4c7f4d7f9aa3145750b45b2cdf0481cafd8f7ccb"
PARENT_CLASSIFICATION = "auditor_direction_control_failed"

PROTECTED_OUTPUTS = (
    PARENT_OUTPUT,
    scorer.PARENT_OUTPUT,
    Path("bench/proposal_injection/results/PI-001"),
    Path("bench/proposal_injection_v2/results/PI-002"),
    Path("bench/proposal_injection_v3/results/PI-003"),
    Path("bench/oracle_ladder_v2/results/OL-002"),
    Path("bench/bridge_control/results/BC-001"),
)

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *scorer.SOURCE_FILES,
            Path("bench/validator_panel/__init__.py"),
            Path("bench/validator_panel/__main__.py"),
            Path("bench/validator_panel/experiment.py"),
            Path("tests/test_validator_panel_experiment.py"),
            PROTOCOL_DOC,
        )
    )
)

ARRAY_NAMES = (
    "validator_seeds",
    "parent_source_seeds",
    "fresh_source_seeds",
    "parent_validator_scores",
    "fresh_validator_scores",
    "parent_identity_union_indices",
    "fresh_identity_union_indices",
    "parent_identity_flat_indices",
    "fresh_identity_flat_indices",
    "parent_identity_sequences",
    "fresh_identity_sequences",
    "parent_identity_exact_scores",
    "fresh_identity_exact_scores",
    "parent_raw_states",
    "fresh_raw_states",
)

OUTCOME_PATHS = (
    STARTED_FILE,
    TENSOR_FILE,
    RESULT_FILE,
    CSV_FILE,
    REPORT_FILE,
    Path("artifact-manifest.json"),
)
PREPARED_PATHS = (
    Path("protocol.json"),
    Path("input-manifest.json"),
    *(PARENT_COPY / Path(name) for name in PARENT_HASHES),
)
ARTIFACT_PATHS = (
    *PREPARED_PATHS,
    STARTED_FILE,
    TENSOR_FILE,
    RESULT_FILE,
    CSV_FILE,
    REPORT_FILE,
)

RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "status",
        "interpretation_scope",
        "protocol_sha256",
        "input_manifest_sha256",
        "formal_start",
        "formal_start_sha256",
        "protocol",
        "input_manifest",
        "tensor_package",
        "execution",
        "evaluation_accounting",
        "repository",
        "versions",
        "parent_call_rows",
        "fresh_call_rows",
        "direction_seed_summaries",
        "sensitivity_seed_summaries",
        "decision",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_value(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _canonical_json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return cast(
        dict[str, Any],
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant),
    )


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("VP-001 evidence-package root cannot be a symlink")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"VP-001 evidence package contains a forbidden symlink: {candidate.relative_to(root)}")


def _assert_exact_tree(root: Path, expected_files: set[Path], *, label: str) -> None:
    """Close an evidence tree over files, directories, symlinks, and special entries."""

    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a real directory")
    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if candidate.is_symlink():
            raise ValueError(f"{label} contains a forbidden symlink: {relative}")
        if candidate.is_file():
            actual_files.add(relative)
        elif candidate.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"{label} contains a forbidden special entry: {relative}")
    expected_directories = {parent for path in expected_files for parent in path.parents if parent != Path(".")}
    if actual_files != expected_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise ValueError(f"{label} file set drifted; missing={missing}, extra={extra}")
    if actual_directories != expected_directories:
        missing = sorted(str(path) for path in expected_directories - actual_directories)
        extra = sorted(str(path) for path in actual_directories - expected_directories)
        raise ValueError(f"{label} directory set drifted; missing={missing}, extra={extra}")


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {str(path): _file_hash(REPO_ROOT / path) for path in SOURCE_FILES}


def _array_header(array: np.ndarray) -> dict[str, object]:
    return {"dtype": np.asarray(array).dtype.str, "shape": list(np.asarray(array).shape)}


def _array_bytes(array: np.ndarray) -> bytes:
    value = np.asarray(array)
    if value.dtype.kind == "f":
        canonical = np.asarray(value, dtype="<f8", order="C")
    elif value.dtype.kind in ("i", "u"):
        canonical = np.asarray(value, dtype="<i8", order="C")
    elif value.dtype.kind == "b":
        canonical = np.asarray(value, dtype=np.bool_, order="C")
    else:
        raise TypeError(f"unsupported tensor dtype {value.dtype}")
    return canonical.tobytes(order="C")


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = sha256()
    digest.update(b"VP-001 canonical tensor package\0")
    for name in sorted(arrays):
        encoded_name = name.encode("utf-8")
        header = _canonical_json_bytes(_array_header(arrays[name]))
        payload = _array_bytes(arrays[name])
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _arrays_metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "semantic_sha256": _array_digest(arrays),
        "arrays": {name: _array_header(arrays[name]) for name in sorted(arrays)},
    }


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cast(Any, np.savez_compressed)(path, **{name: arrays[name] for name in ARRAY_NAMES})


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        if len(package.files) != len(ARRAY_NAMES) or set(package.files) != set(ARRAY_NAMES):
            raise ValueError("VP-001 tensor package has unexpected array names")
        return {name: np.asarray(package[name]).copy() for name in ARRAY_NAMES}


def _expected_parent_manifest() -> dict[str, object]:
    return {
        "schema_version": scorer.SCHEMA_VERSION,
        "experiment_id": scorer.EXPERIMENT_ID,
        "artifacts": {name: digest for name, digest in PARENT_HASHES.items() if name != "artifact-manifest.json"},
    }


def _parent_snapshot(path: Path) -> dict[str, object]:
    _assert_exact_tree(
        path,
        {Path(name) for name in PARENT_HASHES},
        label="SS-001 parent package",
    )
    verification = scorer.verify(path, require_results=True)
    if verification["outcomes"] != "verified_results":
        raise ValueError("VP-001 requires verified SS-001 results")
    for name, expected in PARENT_HASHES.items():
        if _file_hash(path / name) != expected:
            raise ValueError(f"SS-001 hash drifted for {name}")
    if _read_json(path / "artifact-manifest.json") != _expected_parent_manifest():
        raise ValueError("SS-001 artifact manifest entry map drifted")
    protocol = _read_json(path / "protocol.json")
    if _canonical_json_sha256(protocol) != PARENT_PROTOCOL_CANONICAL_SHA256:
        raise ValueError("SS-001 canonical protocol hash drifted")
    results = _read_json(path / scorer.RESULT_FILE)
    tensor = cast(dict[str, Any], results["tensor_package"])
    decision = cast(dict[str, Any], results["decision"])
    if (
        results.get("schema_version") != scorer.SCHEMA_VERSION
        or results.get("experiment_id") != scorer.EXPERIMENT_ID
        or results.get("status") != "completed_cross_seed_scorer_swap"
        or decision.get("classification") != PARENT_CLASSIFICATION
        or cast(dict[str, Any], decision["recurrence_gate"])
        != {
            "passes": True,
            "required_seeds": 10,
            "supporting_seeds": 11,
        }
        or cast(dict[str, Any], decision["calibration_gate"])
        != {
            "eligible_seeds": 6,
            "passes": False,
            "required_seeds": 5,
            "supporting_seeds": 3,
        }
        or decision.get("mechanism") != "not_interpretable"
        or decision.get("rescue") != "not_interpretable"
        or tensor.get("semantic_sha256") != PARENT_TENSOR_SEMANTIC_SHA256
        or tensor.get("file_sha256") != PARENT_HASHES[str(scorer.TENSOR_FILE)]
    ):
        raise ValueError("SS-001 result identity, decision, or tensor metadata drifted")
    return {
        "experiment_id": scorer.EXPERIMENT_ID,
        "verification": verification["outcomes"],
        "hashes": dict(PARENT_HASHES),
        "protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
        "tensor_semantic_sha256": PARENT_TENSOR_SEMANTIC_SHA256,
        "classification": PARENT_CLASSIFICATION,
        "epistemic_role": "outcome-visible fixed-case diagnostic input; never replication evidence",
    }


def protocol_record() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_formal_execution",
        "protocol_document": str(PROTOCOL_DOC),
        "protocol_document_sha256": _file_hash(REPO_ROOT / PROTOCOL_DOC),
        "source_sha256": _source_hashes(),
        "runtime_constraints": {"python": sys.version.split()[0], "numpy": np.__version__},
        "scientific_scope": (
            "prospective validator outputs on three outcome-visible fixed-case units; same-data restart diagnostic only"
        ),
        "seeds": {
            "validators": list(VALIDATOR_SEEDS),
            "development_excluded": DEVELOPMENT_SEED,
            "outcome_visible_parent_generators": list(PARENT_SOURCE_SEEDS),
            "outcome_visible_fresh_generators": list(FRESH_SOURCE_SEEDS),
        },
        "experimental_unit": (
            "three fixed target generator calls measured by one correlated 12-validator panel; no pooled-vote inference"
        ),
        "training": {"steps": 1_800, "batch_size": 64, "dataset": "BC-001/b1_r1_d8"},
        "scoring": {
            "union_candidates": EXPECTED_UNIQUE_CANDIDATES,
            "identity_order": list(IDENTITY_NAMES),
            "horizon": injection.HORIZON,
            "discount": 0.99,
            "uncertainty_penalty": 0.0,
            "epistemic_horizon_bound": None,
            "own_encoder_per_validator": True,
            "ranking": "descending average-tie within-validator union rank",
            "normalized_c_over_x_rank_gap": "(rank(X)-rank(C))/(186-1); positive favors C",
            "quantiles": "numpy linear method",
            "new_candidate_generation": False,
            "new_panel_selection": False,
            "raw_cross_model_score_aggregation_forbidden": True,
        },
        "fixed_calls": {
            "targets": [list(call) for call in TARGET_CALLS],
            "target_identities": [dict(item) for item in TARGET_IDENTITIES],
            "direction_controls": [list(call) for call in DIRECTION_CALLS],
            "fixed_x_sensitivity_controls": [list(call) for call in SENSITIVITY_CALLS],
            "identity_sentinels": [list(call) for call in IDENTITY_SENTINEL_CALLS],
            "submaterial_descriptive": [list(call) for call in SUBMATERIAL_CALLS],
            "positive_x_descriptive": {
                "call": list(POSITIVE_X_DESCRIPTIVE_CALL),
                "x_exact_delta_from_c": POSITIVE_X_DESCRIPTIVE_EXACT_DELTA,
                "x_exact_rank_gain_from_c": POSITIVE_X_DESCRIPTIVE_EXACT_RANK_GAIN,
                "x_vs_c_role": "descriptive_non_gating",
            },
            "other_visible_calls": "provenance_and_exploratory_only; ineligible for gates",
        },
        "thresholds": {
            "call_vote_floor": CALL_VOTE_FLOOR,
            "within_seed_fraction": WITHIN_SEED_FRACTION,
            "direction_seed_floor": DIRECTION_SEED_FLOOR,
            "direction_seed_count": 6,
            "sensitivity_seed_floor": SENSITIVITY_SEED_FLOOR,
            "sensitivity_seed_count": 11,
            "statistical_role": "descriptive robustness thresholds; never binomial tests",
        },
        "classification_order": [
            "invalid_on_integrity_or_identity_sentinel_failure",
            "validator_direction_control_failed",
            "target_direction_confounded",
            "validator_fixed_X_sensitivity_control_failed",
            "target_branch",
        ],
        "target_branches": [
            "finite_panel_winners_curse_supported",
            "same_data_shared_blind_spot_supported",
            "heterogeneous_target_failure",
            "target_panel_inconclusive",
        ],
        "stop_rules": [
            "write the atomic formal-start marker before training seed 44",
            "run all 12 validators exactly once without retry, selection, or early stopping",
            "never generate candidates, compute a new panel winner, or change frozen B0/C/X",
            "never inspect outcomes between validators or change gates after start",
            "a post-start execution or verifier defect is terminal for VP-001",
            "do not modify production, tasks, ADRs, gates, SS-001, or CL-001",
        ],
        "interpretation_limits": [
            "VP-001 cannot reclassify SS-001",
            "same-data restart stability is not independent-data or architecture generalization",
            "fixed cases do not estimate prevalence",
            "offline ranks do not establish episode-return or control-policy effects",
        ],
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    dataset = load_dataset(output / DATASET_COPY)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": _canonical_json_sha256(protocol_record()),
        "original_parent": _parent_snapshot(REPO_ROOT / PARENT_OUTPUT),
        "copied_parent": _parent_snapshot(output / PARENT_COPY),
        "copied_dataset": {
            "path": str(DATASET_COPY),
            "file_sha256": _file_hash(output / DATASET_COPY),
            "semantic_sha256": semantic_hash(dataset),
            "name": dataset.name,
        },
        "prior_parent_semantic_verification": {
            "status": "verified_semantic_results_before_VP-001_design",
            "scope": "receipt only; child fast verification independently pins all parent bytes",
        },
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_safe_output(output: Path) -> None:
    resolved = output.resolve()
    repo_resolved = REPO_ROOT.resolve()
    if resolved == repo_resolved or resolved in repo_resolved.parents:
        raise ValueError("VP-001 output cannot be the repository root or an ancestor")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("VP-001 output must be a real directory path")
    for protected in PROTECTED_OUTPUTS:
        protected_resolved = (REPO_ROOT / protected).resolve()
        if _paths_overlap(resolved, protected_resolved):
            raise ValueError(f"VP-001 output overlaps protected evidence package {protected}")
    owned_results = (REPO_ROOT / "bench/validator_panel/results").resolve()
    if repo_resolved in resolved.parents and owned_results not in resolved.parents:
        raise ValueError("in-repository VP-001 output must be below bench/validator_panel/results")
    if resolved == owned_results:
        raise ValueError("VP-001 output must be below, not equal to, its results root")
    if (output / RESULT_FILE).exists():
        raise FileExistsError("VP-001 is already formal; preserve it and use a new id")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Freeze VP-001 sources, complete SS-001 copy, and protocol without outcomes."""

    _parent_snapshot(REPO_ROOT / PARENT_OUTPUT)
    _assert_safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("VP-001 output must be absent or an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / PARENT_OUTPUT, output / PARENT_COPY)
    _write_json(output / "protocol.json", protocol_record())
    _write_json(output / "input-manifest.json", _expected_input_manifest(output))
    result = verify(output)
    return {**result, "status": "prepared_only"}


def _formal_start_record(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "protocol_sha256": _canonical_json_sha256(_read_json(output / "protocol.json")),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
        "parent_artifact_manifest_sha256": PARENT_HASHES["artifact-manifest.json"],
        "parent_results_sha256": PARENT_HASHES[str(scorer.RESULT_FILE)],
        "parent_tensor_semantic_sha256": PARENT_TENSOR_SEMANTIC_SHA256,
        "parent_classification": PARENT_CLASSIFICATION,
    }


def _mark_formal_started(output: Path) -> dict[str, object]:
    """Durably consume VP-001 before any formal validator is trained."""

    _assert_safe_output(output)
    record = _formal_start_record(output)
    payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = output / STARTED_FILE
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return record


def _empty_arrays() -> dict[str, np.ndarray]:
    sources = (len(PARENT_SOURCE_SEEDS), len(EVAL_STARTS))
    pools = sources + (injection.NATIVE_ITERATIONS, injection.NATIVE_CANDIDATES)
    validators = (len(VALIDATOR_SEEDS),) + pools
    identities = sources + (len(IDENTITY_NAMES),)
    return {
        "validator_seeds": np.asarray(VALIDATOR_SEEDS, dtype=np.int64),
        "parent_source_seeds": np.asarray(PARENT_SOURCE_SEEDS, dtype=np.int64),
        "fresh_source_seeds": np.asarray(FRESH_SOURCE_SEEDS, dtype=np.int64),
        "parent_validator_scores": np.empty(validators, dtype=np.float64),
        "fresh_validator_scores": np.empty(validators, dtype=np.float64),
        "parent_identity_union_indices": np.empty(identities, dtype=np.int64),
        "fresh_identity_union_indices": np.empty(identities, dtype=np.int64),
        "parent_identity_flat_indices": np.empty(identities, dtype=np.int64),
        "fresh_identity_flat_indices": np.empty(identities, dtype=np.int64),
        "parent_identity_sequences": np.empty(
            identities + (injection.HORIZON, 2),
            dtype=np.float64,
        ),
        "fresh_identity_sequences": np.empty(
            identities + (injection.HORIZON, 2),
            dtype=np.float64,
        ),
        "parent_identity_exact_scores": np.empty(identities, dtype=np.float64),
        "fresh_identity_exact_scores": np.empty(identities, dtype=np.float64),
        "parent_raw_states": np.empty(sources + (3,), dtype=np.float64),
        "fresh_raw_states": np.empty(sources + (3,), dtype=np.float64),
    }


def _validate_array_shapes(arrays: dict[str, np.ndarray]) -> None:
    expected = _empty_arrays()
    if set(arrays) != set(expected):
        raise ValueError("VP-001 tensor arrays are incomplete")
    for name, template in expected.items():
        value = arrays[name]
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"VP-001 tensor {name} has {value.shape}/{value.dtype}, expected {template.shape}/{template.dtype}"
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"VP-001 tensor {name} contains non-finite values")
    for name, expected_seeds in (
        ("validator_seeds", VALIDATOR_SEEDS),
        ("parent_source_seeds", PARENT_SOURCE_SEEDS),
        ("fresh_source_seeds", FRESH_SOURCE_SEEDS),
    ):
        if not np.array_equal(arrays[name], np.asarray(expected_seeds, dtype=np.int64)):
            raise ValueError(f"VP-001 {name} axis differs from the frozen protocol")


def _parent_context(parent_root: Path) -> dict[str, Any]:
    ss_arrays = scorer._load_arrays(parent_root / scorer.TENSOR_FILE)
    cl_arrays = landscape._load_arrays(parent_root / scorer.PARENT_TENSOR_COPY)
    parent_step0 = scorer._parent_step0_arrays(cl_arrays)
    results = _read_json(parent_root / scorer.RESULT_FILE)
    parent_rows = {
        (int(row["seed"]), int(row["episode_index"])): row
        for row in cast(list[dict[str, Any]], results["parent_calibration_call_rows"])
    }
    fresh_rows = {
        (int(row["seed"]), int(row["episode_index"])): row
        for row in cast(list[dict[str, Any]], results["fresh_call_rows"])
    }
    expected_parent = {(seed, episode) for seed in PARENT_SOURCE_SEEDS for episode in range(len(EVAL_STARTS))}
    expected_fresh = {(seed, episode) for seed in FRESH_SOURCE_SEEDS for episode in range(len(EVAL_STARTS))}
    if set(parent_rows) != expected_parent or set(fresh_rows) != expected_fresh:
        raise ValueError("VP-001 parent call-row identity is incomplete")
    return {
        "ss_arrays": ss_arrays,
        "parent_step0": parent_step0,
        "parent_rows": parent_rows,
        "fresh_rows": fresh_rows,
    }


def _call_data(
    context: dict[str, Any],
    source_kind: str,
    seed_axis: int,
    episode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if source_kind == "parent_visible":
        source = cast(dict[str, np.ndarray], context["parent_step0"])
        seed = PARENT_SOURCE_SEEDS[seed_axis]
        rows = cast(dict[tuple[int, int], dict[str, Any]], context["parent_rows"])
        return (
            np.asarray(source["sequences"][seed_axis, episode], dtype=np.float64),
            np.asarray(source["exact_scores"][seed_axis, episode], dtype=np.float64),
            np.asarray(source["raw_states"][seed_axis, episode], dtype=np.float64),
            rows[(seed, episode)],
        )
    if source_kind == "fresh_visible":
        source = cast(dict[str, np.ndarray], context["ss_arrays"])
        seed = FRESH_SOURCE_SEEDS[seed_axis]
        rows = cast(dict[tuple[int, int], dict[str, Any]], context["fresh_rows"])
        return (
            np.asarray(source["fresh_sequences"][seed_axis, episode], dtype=np.float64),
            np.asarray(source["fresh_exact_scores"][seed_axis, episode], dtype=np.float64),
            np.asarray(source["fresh_raw_states"][seed_axis, episode], dtype=np.float64),
            rows[(seed, episode)],
        )
    raise ValueError("VP-001 source kind is invalid")


def _identity_payload(
    sequences: np.ndarray,
    exact_scores: np.ndarray,
    raw_state: np.ndarray,
    saved_row: dict[str, Any],
) -> dict[str, np.ndarray]:
    flat_sequences = np.asarray(sequences, dtype=np.float64).reshape(
        -1,
        injection.HORIZON,
        2,
    )
    flat_exact = np.asarray(exact_scores, dtype=np.float64).reshape(-1)
    unique_flat = scorer._first_unique_indices(flat_sequences)
    if len(unique_flat) != EXPECTED_UNIQUE_CANDIDATES:
        raise ValueError("VP-001 parent union cardinality drifted")
    union_indices = np.asarray(
        [
            int(saved_row["b0_union_index"]),
            int(saved_row["c_union_index"]),
            int(saved_row["x_union_index"]),
        ],
        dtype=np.int64,
    )
    if np.any(union_indices < 0) or np.any(union_indices >= EXPECTED_UNIQUE_CANDIDATES):
        raise ValueError("VP-001 frozen identity union index is out of bounds")
    flat_indices = unique_flat[union_indices]
    identity_sequences = flat_sequences[flat_indices]
    identity_exact = flat_exact[flat_indices]
    expected_hashes = (
        str(saved_row["b0_sequence_sha256"]),
        str(saved_row["c_sequence_sha256"]),
        str(saved_row["x_sequence_sha256"]),
    )
    actual_hashes = tuple(scorer._sequence_sha256(sequence) for sequence in identity_sequences)
    if actual_hashes != expected_hashes:
        raise ValueError("VP-001 frozen identity sequence hash drifted")
    expected_exact = np.asarray(
        [
            float(saved_row["b0_exact_score"]),
            float(saved_row["c_exact_score"]),
            float(saved_row["x_exact_score"]),
        ],
        dtype=np.float64,
    )
    if not np.array_equal(identity_exact, expected_exact):
        raise ValueError("VP-001 frozen identity exact score drifted")
    recomputed = exact_discounted_scores(raw_state, identity_sequences)
    if not np.array_equal(recomputed, identity_exact):
        raise ValueError("VP-001 identity exact scores do not recompute")
    return {
        "union_indices": union_indices,
        "flat_indices": flat_indices,
        "sequences": identity_sequences,
        "exact_scores": identity_exact,
    }


def _verify_fixed_sets(context: dict[str, Any]) -> None:
    parent_rows = cast(dict[tuple[int, int], dict[str, Any]], context["parent_rows"])
    fresh_rows = cast(dict[tuple[int, int], dict[str, Any]], context["fresh_rows"])
    derived_direction = {key for key, row in parent_rows.items() if bool(row["exact_improving_control"])}
    derived_sensitivity = {key for key, row in fresh_rows.items() if bool(row["exact_rescue"])}
    derived_targets = {
        key
        for key, row in parent_rows.items()
        if bool(row["exact_improving_control"]) and bool(row["x_materially_degrades_c"])
    }
    derived_sentinels = {
        key for key in DIRECTION_CALLS if parent_rows[key]["c_sequence_sha256"] == parent_rows[key]["x_sequence_sha256"]
    }
    if derived_direction != set(DIRECTION_CALLS):
        raise ValueError("VP-001 direction control call set drifted")
    if derived_sensitivity != set(SENSITIVITY_CALLS):
        raise ValueError("VP-001 fixed-X sensitivity call set drifted")
    if derived_targets != set(TARGET_CALLS):
        raise ValueError("VP-001 target call set drifted")
    if derived_sentinels != set(IDENTITY_SENTINEL_CALLS):
        raise ValueError("VP-001 identity sentinel call set drifted")
    positive_x = parent_rows[POSITIVE_X_DESCRIPTIVE_CALL]
    if (
        positive_x["x_exact_delta_from_c"] != POSITIVE_X_DESCRIPTIVE_EXACT_DELTA
        or positive_x["x_exact_rank_gain_from_c"] != POSITIVE_X_DESCRIPTIVE_EXACT_RANK_GAIN
        or POSITIVE_X_DESCRIPTIVE_CALL not in DIRECTION_CALLS
    ):
        raise ValueError("VP-001 descriptive positive-X diagnostic drifted")
    for frozen in TARGET_IDENTITIES:
        key = (int(cast(int, frozen["seed"])), int(cast(int, frozen["episode_index"])))
        row = parent_rows[key]
        for field in (
            "b0_union_index",
            "c_union_index",
            "x_union_index",
            "b0_sequence_sha256",
            "c_sequence_sha256",
            "x_sequence_sha256",
            "x_exact_delta_from_c",
            "c_exact_rank",
            "x_exact_rank",
        ):
            if row[field] != frozen[field]:
                raise ValueError(f"VP-001 target identity drifted for {key}/{field}")


def _fill_provenance(arrays: dict[str, np.ndarray], context: dict[str, Any]) -> None:
    for prefix, source_kind in (("parent", "parent_visible"), ("fresh", "fresh_visible")):
        for seed_axis in range(len(PARENT_SOURCE_SEEDS)):
            for episode in range(len(EVAL_STARTS)):
                sequences, exact, raw, row = _call_data(
                    context,
                    source_kind,
                    seed_axis,
                    episode,
                )
                payload = _identity_payload(sequences, exact, raw, row)
                arrays[f"{prefix}_identity_union_indices"][seed_axis, episode] = payload["union_indices"]
                arrays[f"{prefix}_identity_flat_indices"][seed_axis, episode] = payload["flat_indices"]
                arrays[f"{prefix}_identity_sequences"][seed_axis, episode] = payload["sequences"]
                arrays[f"{prefix}_identity_exact_scores"][seed_axis, episode] = payload["exact_scores"]
                arrays[f"{prefix}_raw_states"][seed_axis, episode] = raw


def _execute(output: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    dataset = load_dataset(output / DATASET_COPY)
    if dataset.name != "b1_r1_d8":
        raise ValueError("VP-001 requires the frozen BC-001 b1_r1_d8 dataset")
    context = _parent_context(output / PARENT_COPY)
    _verify_fixed_sets(context)
    arrays = _empty_arrays()
    _fill_provenance(arrays, context)
    records: list[dict[str, object]] = []
    for validator_axis, seed in enumerate(VALIDATOR_SEEDS):
        model = train_balanced_model(dataset, seed)
        fingerprint = oracle._model_fingerprint(model)
        for seed_axis in range(len(PARENT_SOURCE_SEEDS)):
            for episode in range(len(EVAL_STARTS)):
                parent_sequences, _, parent_raw, _ = _call_data(
                    context,
                    "parent_visible",
                    seed_axis,
                    episode,
                )
                fresh_sequences, _, fresh_raw, _ = _call_data(
                    context,
                    "fresh_visible",
                    seed_axis,
                    episode,
                )
                arrays["parent_validator_scores"][validator_axis, seed_axis, episode] = scorer._score_blocks(
                    model, parent_raw, parent_sequences
                )
                arrays["fresh_validator_scores"][validator_axis, seed_axis, episode] = scorer._score_blocks(
                    model, fresh_raw, fresh_sequences
                )
        records.append({"seed": seed, "model_sha256": fingerprint})
    _validate_array_shapes(arrays)
    return arrays, {"validator_records": records}


def _verify_provenance(
    arrays: dict[str, np.ndarray],
    context: dict[str, Any],
) -> None:
    expected = _empty_arrays()
    _fill_provenance(expected, context)
    for name in ARRAY_NAMES:
        if name.endswith("validator_scores"):
            continue
        if not np.array_equal(arrays[name], expected[name]):
            raise ValueError(f"VP-001 frozen provenance differs in {name}")
    for prefix, source_kind in (("parent", "parent_visible"), ("fresh", "fresh_visible")):
        for seed_axis in range(len(PARENT_SOURCE_SEEDS)):
            for episode in range(len(EVAL_STARTS)):
                sequences, _, _, row = _call_data(context, source_kind, seed_axis, episode)
                scores = arrays[f"{prefix}_validator_scores"][:, seed_axis, episode]
                scorer._verify_duplicate_scores(sequences, scores)
                identity_indices = arrays[f"{prefix}_identity_flat_indices"][seed_axis, episode]
                identity_scores = scores.reshape(len(VALIDATOR_SEEDS), -1)[:, identity_indices]
                identity_sequences = arrays[f"{prefix}_identity_sequences"][seed_axis, episode]
                if row["c_sequence_sha256"] == row["x_sequence_sha256"]:
                    if not np.array_equal(identity_sequences[1], identity_sequences[2]):
                        raise ValueError("VP-001 identity sentinel sequences differ")
                    if not np.array_equal(identity_scores[:, 1], identity_scores[:, 2]):
                        raise ValueError("VP-001 identity sentinel scores differ")


def _derive_call(
    *,
    source_kind: str,
    seed: int,
    episode_index: int,
    sequences: np.ndarray,
    validator_scores: np.ndarray,
    identity_flat_indices: np.ndarray,
    identity_exact_scores: np.ndarray,
) -> dict[str, object]:
    """Derive rank-only preferences for one frozen B0/C/X triplet."""

    if source_kind not in ("parent_visible", "fresh_visible"):
        raise ValueError("VP-001 call source kind is invalid")
    if source_kind == "parent_visible" and seed not in PARENT_SOURCE_SEEDS:
        raise ValueError("VP-001 parent source seed is outside the frozen axis")
    if source_kind == "fresh_visible" and seed not in FRESH_SOURCE_SEEDS:
        raise ValueError("VP-001 fresh source seed is outside the frozen axis")
    if episode_index not in range(len(EVAL_STARTS)):
        raise ValueError("VP-001 episode index is outside the frozen axis")

    block_shape = (injection.NATIVE_ITERATIONS, injection.NATIVE_CANDIDATES)
    if np.asarray(sequences).shape != block_shape + (injection.HORIZON, 2):
        raise ValueError("VP-001 call sequences have the wrong shape")
    if np.asarray(validator_scores).shape != (len(VALIDATOR_SEEDS),) + block_shape:
        raise ValueError("VP-001 call validator scores have the wrong shape")
    identity_indices = np.asarray(identity_flat_indices)
    identity_exact = np.asarray(identity_exact_scores)
    if identity_indices.shape != (len(IDENTITY_NAMES),) or identity_indices.dtype.kind not in (
        "i",
        "u",
    ):
        raise ValueError("VP-001 identity flat indices have the wrong schema")
    if identity_exact.shape != (len(IDENTITY_NAMES),):
        raise ValueError("VP-001 identity exact scores have the wrong shape")

    flat_sequences = np.asarray(sequences, dtype=np.float64).reshape(
        -1,
        injection.HORIZON,
        2,
    )
    flat_validators = np.asarray(validator_scores, dtype=np.float64).reshape(
        len(VALIDATOR_SEEDS),
        -1,
    )
    if not np.all(np.isfinite(flat_sequences)) or not np.all(np.isfinite(flat_validators)):
        raise ValueError("VP-001 call tensors must be finite")
    if not np.all(np.isfinite(identity_exact)):
        raise ValueError("VP-001 call exact identities must be finite")
    if np.any(identity_indices < 0) or np.any(identity_indices >= len(flat_sequences)):
        raise ValueError("VP-001 identity flat index is out of bounds")

    unique_flat = scorer._first_unique_indices(flat_sequences)
    if len(unique_flat) != EXPECTED_UNIQUE_CANDIDATES:
        raise ValueError("VP-001 call union cardinality drifted")
    scorer._verify_duplicate_scores(sequences, validator_scores)
    flat_to_union = {int(flat_index): union_index for union_index, flat_index in enumerate(unique_flat)}
    try:
        identity_union = np.asarray(
            [flat_to_union[int(flat_index)] for flat_index in identity_indices],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("VP-001 identity is not a canonical first occurrence") from error

    unique_scores = flat_validators[:, unique_flat]
    ranks = np.stack(
        [average_tie_ranks(score_row, descending=True) for score_row in unique_scores],
        axis=0,
    )
    identity_ranks = ranks[:, identity_union]
    b0_ranks, c_ranks, x_ranks = (
        identity_ranks[:, 0],
        identity_ranks[:, 1],
        identity_ranks[:, 2],
    )

    c_over_b0 = int(np.sum(c_ranks < b0_ranks))
    b0_over_c = int(np.sum(b0_ranks < c_ranks))
    b0_c_ties = len(VALIDATOR_SEEDS) - c_over_b0 - b0_over_c
    c_over_x = int(np.sum(c_ranks < x_ranks))
    x_over_c = int(np.sum(x_ranks < c_ranks))
    c_x_ties = len(VALIDATOR_SEEDS) - c_over_x - x_over_c
    x_over_b0 = int(np.sum(x_ranks < b0_ranks))
    b0_over_x = int(np.sum(b0_ranks < x_ranks))
    b0_x_ties = len(VALIDATOR_SEEDS) - x_over_b0 - b0_over_x

    key = (seed, episode_index)
    is_target = source_kind == "parent_visible" and key in TARGET_CALLS
    is_direction = source_kind == "parent_visible" and key in DIRECTION_CALLS
    is_sensitivity = source_kind == "fresh_visible" and key in SENSITIVITY_CALLS
    is_sentinel = source_kind == "parent_visible" and key in IDENTITY_SENTINEL_CALLS
    is_submaterial = source_kind == "parent_visible" and key in SUBMATERIAL_CALLS
    if is_direction and not identity_exact[1] > identity_exact[0]:
        raise ValueError("VP-001 direction control no longer has exact C > B0")
    if is_sensitivity and not identity_exact[2] > identity_exact[1]:
        raise ValueError("VP-001 sensitivity control no longer has exact X > C")
    if is_target and not identity_exact[1] > identity_exact[2]:
        raise ValueError("VP-001 target no longer has exact C > X")
    if is_sentinel:
        if not np.array_equal(flat_sequences[identity_indices[1]], flat_sequences[identity_indices[2]]):
            raise ValueError("VP-001 identity sentinel sequences are not identical")
        if not np.array_equal(c_ranks, x_ranks) or c_x_ties != len(VALIDATOR_SEEDS):
            raise ValueError("VP-001 identity sentinel ranks are not tied")

    normalized_gap = (x_ranks - c_ranks) / float(EXPECTED_UNIQUE_CANDIDATES - 1)
    hashes = [scorer._sequence_sha256(flat_sequences[index]) for index in identity_indices]
    return {
        "source_kind": source_kind,
        "seed": seed,
        "episode_index": episode_index,
        "unique_candidates": len(unique_flat),
        "is_target": is_target,
        "is_direction_control": is_direction,
        "is_sensitivity_control": is_sensitivity,
        "is_identity_sentinel": is_sentinel,
        "is_submaterial": is_submaterial,
        "b0_flat_index": int(identity_indices[0]),
        "c_flat_index": int(identity_indices[1]),
        "x_flat_index": int(identity_indices[2]),
        "b0_union_index": int(identity_union[0]),
        "c_union_index": int(identity_union[1]),
        "x_union_index": int(identity_union[2]),
        "b0_sequence_sha256": hashes[0],
        "c_sequence_sha256": hashes[1],
        "x_sequence_sha256": hashes[2],
        "b0_exact_score": float(identity_exact[0]),
        "c_exact_score": float(identity_exact[1]),
        "x_exact_score": float(identity_exact[2]),
        "c_exact_delta_from_b0": float(identity_exact[1] - identity_exact[0]),
        "x_exact_delta_from_c": float(identity_exact[2] - identity_exact[1]),
        "b0_validator_ranks": [float(value) for value in b0_ranks],
        "c_validator_ranks": [float(value) for value in c_ranks],
        "x_validator_ranks": [float(value) for value in x_ranks],
        "c_over_b0_votes": c_over_b0,
        "b0_over_c_votes": b0_over_c,
        "b0_c_tie_votes": b0_c_ties,
        "c_over_x_votes": c_over_x,
        "x_over_c_votes": x_over_c,
        "c_x_tie_votes": c_x_ties,
        "x_over_b0_votes": x_over_b0,
        "b0_over_x_votes": b0_over_x,
        "b0_x_tie_votes": b0_x_ties,
        "normalized_c_over_x_rank_gap_median": float(np.quantile(normalized_gap, 0.5, method="linear")),
        "normalized_c_over_x_rank_gap_q25": float(np.quantile(normalized_gap, 0.25, method="linear")),
        "normalized_c_over_x_rank_gap_q75": float(np.quantile(normalized_gap, 0.75, method="linear")),
        "direction_call_pass": bool(is_direction and c_over_b0 >= CALL_VOTE_FLOOR),
        "fixed_x_sensitivity_call_pass": bool(is_sensitivity and x_over_c >= CALL_VOTE_FLOOR),
        "heldout_reject": bool(is_target and c_over_x >= CALL_VOTE_FLOOR),
        "heldout_transfer": bool(is_target and x_over_c >= CALL_VOTE_FLOOR),
    }


def _seed_summaries(
    parent_rows: list[dict[str, object]],
    fresh_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Aggregate only the frozen eligible controls within source generator."""

    direction_rows = [row for row in parent_rows if bool(row["is_direction_control"])]
    sensitivity_rows = [row for row in fresh_rows if bool(row["is_sensitivity_control"])]
    if {(int(cast(int, row["seed"])), int(cast(int, row["episode_index"]))) for row in direction_rows} != set(
        DIRECTION_CALLS
    ):
        raise ValueError("VP-001 derived direction control call set drifted")
    if {(int(cast(int, row["seed"])), int(cast(int, row["episode_index"]))) for row in sensitivity_rows} != set(
        SENSITIVITY_CALLS
    ):
        raise ValueError("VP-001 derived sensitivity control call set drifted")

    direction: list[dict[str, object]] = []
    for seed in sorted({item[0] for item in DIRECTION_CALLS}):
        rows = [row for row in direction_rows if int(cast(int, row["seed"])) == seed]
        required = int(math.ceil(WITHIN_SEED_FRACTION * len(rows)))
        passing = sum(bool(row["direction_call_pass"]) for row in rows)
        direction.append(
            {
                "seed": seed,
                "eligible_calls": len(rows),
                "required_calls": required,
                "passing_calls": passing,
                "supports": passing >= required,
            }
        )

    sensitivity: list[dict[str, object]] = []
    for seed in sorted({item[0] for item in SENSITIVITY_CALLS}):
        rows = [row for row in sensitivity_rows if int(cast(int, row["seed"])) == seed]
        required = int(math.ceil(WITHIN_SEED_FRACTION * len(rows)))
        passing = sum(bool(row["fixed_x_sensitivity_call_pass"]) for row in rows)
        sensitivity.append(
            {
                "seed": seed,
                "eligible_calls": len(rows),
                "required_calls": required,
                "passing_calls": passing,
                "supports": passing >= required,
            }
        )
    if len(direction) != 6 or sum(int(cast(int, row["eligible_calls"])) for row in direction) != 11:
        raise ValueError("VP-001 direction control seed accounting drifted")
    if len(sensitivity) != 11 or sum(int(cast(int, row["eligible_calls"])) for row in sensitivity) != 25:
        raise ValueError("VP-001 sensitivity control seed accounting drifted")
    return {"direction": direction, "sensitivity": sensitivity}


def _decision(
    summaries: dict[str, list[dict[str, object]]],
    parent_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the frozen control-first decision tree to the three fixed targets."""

    direction = summaries["direction"]
    sensitivity = summaries["sensitivity"]
    if [int(cast(int, row["seed"])) for row in direction] != sorted({item[0] for item in DIRECTION_CALLS}):
        raise ValueError("VP-001 direction summary seed order drifted")
    if [int(cast(int, row["seed"])) for row in sensitivity] != sorted({item[0] for item in SENSITIVITY_CALLS}):
        raise ValueError("VP-001 sensitivity summary seed order drifted")

    target_by_key = {
        (int(cast(int, row["seed"])), int(cast(int, row["episode_index"]))): row
        for row in parent_rows
        if bool(row["is_target"])
    }
    if set(target_by_key) != set(TARGET_CALLS):
        raise ValueError("VP-001 target rows are incomplete")
    targets = [target_by_key[key] for key in TARGET_CALLS]
    for row in targets:
        if bool(row["heldout_reject"]) and bool(row["heldout_transfer"]):
            raise ValueError("VP-001 mutually exclusive target votes both passed")

    direction_count = sum(bool(row["supports"]) for row in direction)
    direction_passes = direction_count >= DIRECTION_SEED_FLOOR
    target_direction_count = sum(int(cast(int, row["c_over_b0_votes"])) >= CALL_VOTE_FLOOR for row in targets)
    target_direction_passes = target_direction_count == len(TARGET_CALLS)
    sensitivity_count = sum(bool(row["supports"]) for row in sensitivity)
    sensitivity_passes = sensitivity_count >= SENSITIVITY_SEED_FLOOR
    reject_count = sum(bool(row["heldout_reject"]) for row in targets)
    transfer_count = sum(bool(row["heldout_transfer"]) for row in targets)

    if not direction_passes:
        classification = "validator_direction_control_failed"
    elif not target_direction_passes:
        classification = "target_direction_confounded"
    elif not sensitivity_passes:
        classification = "validator_fixed_X_sensitivity_control_failed"
    elif reject_count == len(TARGET_CALLS):
        classification = "finite_panel_winners_curse_supported"
    elif transfer_count == len(TARGET_CALLS):
        classification = "same_data_shared_blind_spot_supported"
    elif reject_count >= 1 and transfer_count >= 1:
        classification = "heterogeneous_target_failure"
    else:
        classification = "target_panel_inconclusive"

    target_records: list[dict[str, object]] = []
    for row in targets:
        if bool(row["heldout_reject"]):
            outcome = "heldout_reject"
        elif bool(row["heldout_transfer"]):
            outcome = "heldout_transfer"
        else:
            outcome = "inconclusive"
        target_records.append(
            {
                "seed": row["seed"],
                "episode_index": row["episode_index"],
                "c_over_b0_votes": row["c_over_b0_votes"],
                "c_over_x_votes": row["c_over_x_votes"],
                "x_over_c_votes": row["x_over_c_votes"],
                "c_x_tie_votes": row["c_x_tie_votes"],
                "normalized_c_over_x_rank_gap_median": row["normalized_c_over_x_rank_gap_median"],
                "normalized_c_over_x_rank_gap_q25": row["normalized_c_over_x_rank_gap_q25"],
                "normalized_c_over_x_rank_gap_q75": row["normalized_c_over_x_rank_gap_q75"],
                "outcome": outcome,
            }
        )

    gates_interpretable = direction_passes and target_direction_passes and sensitivity_passes
    return {
        "classification": classification,
        "direction_control_gate": {
            "supporting_seeds": direction_count,
            "required_seeds": DIRECTION_SEED_FLOOR,
            "eligible_seeds": len(direction),
            "passes": direction_passes,
        },
        "target_direction_gate": {
            "passing_calls": target_direction_count,
            "required_calls": len(TARGET_CALLS),
            "passes": target_direction_passes,
        },
        "fixed_x_sensitivity_gate": {
            "supporting_seeds": sensitivity_count,
            "required_seeds": SENSITIVITY_SEED_FLOOR,
            "eligible_seeds": len(sensitivity),
            "passes": sensitivity_passes,
        },
        "target_panel": {
            "rejecting_calls": reject_count,
            "transferring_calls": transfer_count,
            "inconclusive_calls": len(TARGET_CALLS) - reject_count - transfer_count,
            "interpretable": gates_interpretable,
            "calls": target_records,
        },
        "ss001_reclassification": "forbidden",
        "prevalence_claim": "forbidden",
    }


def _derive(arrays: dict[str, np.ndarray], context: dict[str, Any]) -> dict[str, object]:
    parent_rows: list[dict[str, object]] = []
    fresh_rows: list[dict[str, object]] = []
    for prefix, source_kind, seeds, destination in (
        ("parent", "parent_visible", PARENT_SOURCE_SEEDS, parent_rows),
        ("fresh", "fresh_visible", FRESH_SOURCE_SEEDS, fresh_rows),
    ):
        for seed_axis, seed in enumerate(seeds):
            for episode in range(len(EVAL_STARTS)):
                sequences, _, _, _ = _call_data(context, source_kind, seed_axis, episode)
                destination.append(
                    _derive_call(
                        source_kind=source_kind,
                        seed=seed,
                        episode_index=episode,
                        sequences=sequences,
                        validator_scores=arrays[f"{prefix}_validator_scores"][:, seed_axis, episode],
                        identity_flat_indices=arrays[f"{prefix}_identity_flat_indices"][seed_axis, episode],
                        identity_exact_scores=arrays[f"{prefix}_identity_exact_scores"][seed_axis, episode],
                    )
                )
    summaries = _seed_summaries(parent_rows, fresh_rows)
    return {
        "parent_call_rows": parent_rows,
        "fresh_call_rows": fresh_rows,
        "direction_seed_summaries": summaries["direction"],
        "sensitivity_seed_summaries": summaries["sensitivity"],
        "decision": _decision(summaries, parent_rows),
    }


def _verify_execution_records(execution: dict[str, Any]) -> None:
    if set(execution) != {"validator_records"}:
        raise ValueError("VP-001 execution record map schema drifted")
    records = cast(list[dict[str, Any]], execution.get("validator_records"))
    if [int(row["seed"]) for row in records] != list(VALIDATOR_SEEDS):
        raise ValueError("VP-001 validator execution records are incomplete")
    for record in records:
        if set(record) != {"seed", "model_sha256"}:
            raise ValueError("VP-001 validator execution record schema drifted")
        fingerprint = str(record["model_sha256"])
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError("VP-001 model fingerprint is not a SHA-256 hex digest")


def _evaluation_accounting() -> dict[str, object]:
    calls = len(VALIDATOR_SEEDS) * (len(PARENT_SOURCE_SEEDS) + len(FRESH_SOURCE_SEEDS)) * len(EVAL_STARTS)
    full_sequences = calls * injection.NATIVE_ITERATIONS * injection.NATIVE_CANDIDATES
    identity_sequences = (len(PARENT_SOURCE_SEEDS) + len(FRESH_SOURCE_SEEDS)) * len(EVAL_STARTS) * len(IDENTITY_NAMES)
    return {
        "trained_models": {
            "validators": len(VALIDATOR_SEEDS),
            "optimizer_updates": len(VALIDATOR_SEEDS) * 1_800,
        },
        "validator_scoring": {
            "full_pool_calls": calls,
            "learned_sequences": full_sequences,
            "learned_transitions": full_sequences * injection.HORIZON,
            "unique_candidates_per_call": EXPECTED_UNIQUE_CANDIDATES,
            "stored_candidates_per_call": injection.NATIVE_ITERATIONS * injection.NATIVE_CANDIDATES,
        },
        "fast_verifier_exact_provenance": {
            "identity_sequences": identity_sequences,
            "identity_transitions": identity_sequences * injection.HORIZON,
        },
    }


def _csv_text(rows: list[dict[str, object]]) -> str:
    fields = (
        "source_kind",
        "seed",
        "episode_index",
        "is_target",
        "is_direction_control",
        "is_sensitivity_control",
        "is_identity_sentinel",
        "is_submaterial",
        "c_exact_delta_from_b0",
        "x_exact_delta_from_c",
        "c_over_b0_votes",
        "b0_over_c_votes",
        "b0_c_tie_votes",
        "c_over_x_votes",
        "x_over_c_votes",
        "c_x_tie_votes",
        "normalized_c_over_x_rank_gap_median",
        "normalized_c_over_x_rank_gap_q25",
        "normalized_c_over_x_rank_gap_q75",
        "direction_call_pass",
        "fixed_x_sensitivity_call_pass",
        "heldout_reject",
        "heldout_transfer",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue()


def _report_text(results: dict[str, Any]) -> str:
    decision = cast(dict[str, Any], results["decision"])
    direction = cast(list[dict[str, Any]], results["direction_seed_summaries"])
    sensitivity = cast(list[dict[str, Any]], results["sensitivity_seed_summaries"])
    panel = cast(dict[str, Any], decision["target_panel"])
    parent_rows = cast(list[dict[str, Any]], results["parent_call_rows"])
    fresh_rows = cast(list[dict[str, Any]], results["fresh_call_rows"])
    positive_x = next(
        row for row in parent_rows if (int(row["seed"]), int(row["episode_index"])) == POSITIVE_X_DESCRIPTIVE_CALL
    )
    noneligible = [
        row
        for row in (*parent_rows, *fresh_rows)
        if not (bool(row["is_target"]) or bool(row["is_direction_control"]) or bool(row["is_sensitivity_control"]))
    ]
    lines = [
        "# VP-001 held-out validator-panel report",
        "",
        f"**Classification:** `{decision['classification']}`",
        "",
        "VP-001 used untouched validator seeds 44..55 to score the sealed SS-001",
        "candidate unions. B0, C, and X were fixed before formal execution; no new",
        "candidate was generated or selected, and raw scores were never pooled across models.",
        "",
        "## Frozen gates",
        "",
        (
            f"- Direction calibration: {decision['direction_control_gate']['supporting_seeds']}/6 "
            f"visible source seeds (required "
            f"{decision['direction_control_gate']['required_seeds']})."
        ),
        f"- Target-local direction: {decision['target_direction_gate']['passing_calls']}/3 calls.",
        (
            f"- Fixed-X sensitivity: "
            f"{decision['fixed_x_sensitivity_gate']['supporting_seeds']}/11 "
            f"visible source seeds (required "
            f"{decision['fixed_x_sensitivity_gate']['required_seeds']})."
        ),
        "",
        "## Three fixed targets",
        "",
    ]
    for row in cast(list[dict[str, Any]], panel["calls"]):
        lines.append(
            f"- ({row['seed']},{row['episode_index']}): C>X {row['c_over_x_votes']}/12, "
            f"X>C {row['x_over_c_votes']}/12, ties {row['c_x_tie_votes']}/12; "
            f"median normalized C-over-X rank gap "
            f"{row['normalized_c_over_x_rank_gap_median']:.6f} "
            f"(IQR {row['normalized_c_over_x_rank_gap_q25']:.6f} to "
            f"{row['normalized_c_over_x_rank_gap_q75']:.6f}); `{row['outcome']}`."
        )
    lines.extend(
        [
            "",
            "## Control detail",
            "",
            (
                f"- Descriptive direction call total: "
                f"{sum(int(row['passing_calls']) for row in direction)}/11; "
                "the gate aggregates within source seed first."
            ),
            (
                f"- Descriptive sensitivity call total: "
                f"{sum(int(row['passing_calls']) for row in sensitivity)}/25; "
                "the gate aggregates within source seed first."
            ),
            (
                f"- (11,0) descriptive non-gating X/C contrast: X>C "
                f"{positive_x['x_over_c_votes']}/12, C>X "
                f"{positive_x['c_over_x_votes']}/12, ties "
                f"{positive_x['c_x_tie_votes']}/12."
            ),
            "- All five byte-identical C/X sentinels tied under every validator.",
            (f"- The remaining {len(noneligible)} non-T/D/G rows are provenance/exploratory only and enter no gate."),
            "",
            "## Interpretation boundary",
            "",
            "This is a prospective same-data, same-architecture validator-restart diagnostic",
            "on three outcome-visible fixed cases. It cannot reclassify SS-001, estimate failure",
            "prevalence, establish new-data or architecture generalization, or establish episode-",
            "return effects. Thresholds are descriptive robustness rules, not p-values.",
            "",
            "## Integrity",
            "",
            "The full 17-file SS-001 tree, frozen protocol and sources, full-union score tensors,",
            "identity provenance, exact recomputation, model fingerprints, deterministic derivations,",
            "and rendered artifacts are hash-bound. Use `verify-semantic` to retrain all 12 formal",
            "validators and regenerate the raw tensors in memory.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifest(output: Path) -> None:
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "artifacts": {str(path): _file_hash(output / path) for path in ARTIFACT_PATHS},
        },
    )


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Execute the 12 frozen held-out validators exactly once."""

    _assert_safe_output(output)
    verification = verify(output)
    if verification["outcomes"] != "prepared_only":
        raise FileExistsError("VP-001 formal execution is one-shot")
    formal_start = _mark_formal_started(output)
    arrays, execution = _execute(output)
    context = _parent_context(output / PARENT_COPY)
    _validate_array_shapes(arrays)
    _verify_provenance(arrays, context)
    _verify_execution_records(cast(dict[str, Any], execution))
    derived = _derive(arrays, context)
    _write_arrays(output / TENSOR_FILE, arrays)
    protocol = _read_json(output / "protocol.json")
    input_manifest = _read_json(output / "input-manifest.json")
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_heldout_validator_panel",
        "interpretation_scope": (
            "prospective same-data validator-restart diagnostic on outcome-visible fixed cases; "
            "no SS-001 reclassification"
        ),
        "protocol_sha256": verification["protocol_sha256"],
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
        "formal_start": formal_start,
        "formal_start_sha256": _file_hash(output / STARTED_FILE),
        "protocol": protocol,
        "input_manifest": input_manifest,
        "tensor_package": {
            "path": str(TENSOR_FILE),
            "file_sha256": _file_hash(output / TENSOR_FILE),
            **_arrays_metadata(arrays),
        },
        "execution": execution,
        "evaluation_accounting": _evaluation_accounting(),
        "repository": {
            "head": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--short")),
            "source_sha256": protocol["source_sha256"],
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        **derived,
    }
    _write_json(output / RESULT_FILE, results)
    all_rows = [
        *cast(list[dict[str, object]], results["parent_call_rows"]),
        *cast(list[dict[str, object]], results["fresh_call_rows"]),
    ]
    (output / CSV_FILE).write_text(_csv_text(all_rows), encoding="utf-8")
    (output / REPORT_FILE).write_text(_report_text(results), encoding="utf-8")
    _write_artifact_manifest(output)
    verify(output, require_results=True)
    return results


def _verify_result_schema(results: dict[str, Any]) -> None:
    """Close the non-derived result envelope before interpreting any saved claim."""

    if set(results) != RESULT_FIELDS:
        raise ValueError("VP-001 result top-level schema drifted")
    repository = results.get("repository")
    if not isinstance(repository, dict) or set(repository) != {
        "head",
        "dirty",
        "source_sha256",
    }:
        raise ValueError("VP-001 repository result schema drifted")
    head = repository.get("head")
    if (
        not isinstance(head, str)
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
        or not isinstance(repository.get("dirty"), bool)
        or not isinstance(repository.get("source_sha256"), dict)
    ):
        raise ValueError("VP-001 repository result values are malformed")
    versions = results.get("versions")
    if not isinstance(versions, dict) or set(versions) != {"python", "numpy", "platform"}:
        raise ValueError("VP-001 version result schema drifted")
    if not all(isinstance(versions[field], str) and versions[field] for field in versions):
        raise ValueError("VP-001 version result values are malformed")
    expected_container_types: dict[str, type[object]] = {
        "formal_start": dict,
        "protocol": dict,
        "input_manifest": dict,
        "tensor_package": dict,
        "execution": dict,
        "evaluation_accounting": dict,
        "parent_call_rows": list,
        "fresh_call_rows": list,
        "direction_seed_summaries": list,
        "sensitivity_seed_summaries": list,
        "decision": dict,
    }
    for field, expected_type in expected_container_types.items():
        if not isinstance(results.get(field), expected_type):
            raise ValueError(f"VP-001 result field {field} has the wrong container type")


def _verify_outcomes(
    output: Path,
    protocol: dict[str, Any],
    input_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _assert_exact_tree(
        output,
        {*ARTIFACT_PATHS, Path("artifact-manifest.json")},
        label="VP-001 completed package",
    )
    results = _read_json(output / RESULT_FILE)
    _verify_result_schema(results)
    if results.get("schema_version") != SCHEMA_VERSION or results.get("experiment_id") != (EXPERIMENT_ID):
        raise ValueError("VP-001 result identity does not match")
    if results.get("status") != "completed_heldout_validator_panel":
        raise ValueError("VP-001 result status does not match")
    if results.get("interpretation_scope") != (
        "prospective same-data validator-restart diagnostic on outcome-visible fixed cases; no SS-001 reclassification"
    ):
        raise ValueError("VP-001 interpretation scope drifted")
    protocol_sha = _canonical_json_sha256(protocol)
    if results.get("protocol") != protocol or results.get("protocol_sha256") != protocol_sha:
        raise ValueError("VP-001 results are not bound to the frozen protocol")
    if results.get("input_manifest") != input_manifest or results.get("input_manifest_sha256") != _file_hash(
        output / "input-manifest.json"
    ):
        raise ValueError("VP-001 results are not bound to the input manifest")
    formal_start = _read_json(output / STARTED_FILE)
    if (
        formal_start != _formal_start_record(output)
        or results.get("formal_start") != formal_start
        or results.get("formal_start_sha256") != _file_hash(output / STARTED_FILE)
    ):
        raise ValueError("VP-001 results are not bound to the formal-start record")
    repository = cast(dict[str, Any], results["repository"])
    if repository.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("VP-001 result source hashes do not match")

    arrays = _load_arrays(output / TENSOR_FILE)
    _validate_array_shapes(arrays)
    tensor_record = cast(dict[str, Any], results["tensor_package"])
    expected_tensor = {
        "path": str(TENSOR_FILE),
        "file_sha256": _file_hash(output / TENSOR_FILE),
        **_arrays_metadata(arrays),
    }
    if tensor_record != expected_tensor:
        raise ValueError("VP-001 tensor package metadata does not match")
    context = _parent_context(output / PARENT_COPY)
    _verify_fixed_sets(context)
    _verify_provenance(arrays, context)
    execution = cast(dict[str, Any], results["execution"])
    _verify_execution_records(execution)
    derived = _derive(arrays, context)
    for field in (
        "parent_call_rows",
        "fresh_call_rows",
        "direction_seed_summaries",
        "sensitivity_seed_summaries",
        "decision",
    ):
        if _canonical_json_value(results[field]) != _canonical_json_value(derived[field]):
            raise ValueError(f"VP-001 saved {field} does not recompute")
    if results.get("evaluation_accounting") != _evaluation_accounting():
        raise ValueError("VP-001 evaluation accounting does not recompute")

    all_rows = [
        *cast(list[dict[str, object]], results["parent_call_rows"]),
        *cast(list[dict[str, object]], results["fresh_call_rows"]),
    ]
    if (output / CSV_FILE).read_text(encoding="utf-8") != _csv_text(all_rows):
        raise ValueError("VP-001 CSV is not canonical")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != _report_text(results):
        raise ValueError("VP-001 report is not canonical")
    artifact_manifest = _read_json(output / "artifact-manifest.json")
    expected_artifacts = {str(path): _file_hash(output / path) for path in ARTIFACT_PATHS}
    if artifact_manifest != {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": expected_artifacts,
    }:
        raise ValueError("VP-001 artifact manifest does not match")
    return results, arrays


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
    semantic: bool = False,
) -> dict[str, object]:
    """Verify frozen inputs and outputs, optionally retraining formal validators."""

    result_path = output / RESULT_FILE
    started_path = output / STARTED_FILE
    has_any_outcome = any((output / path).exists() for path in OUTCOME_PATHS)
    if result_path.exists():
        _assert_exact_tree(
            output,
            {*ARTIFACT_PATHS, Path("artifact-manifest.json")},
            label="VP-001 completed package",
        )
    elif not has_any_outcome:
        _assert_exact_tree(
            output,
            set(PREPARED_PATHS),
            label="VP-001 prepared package",
        )
    else:
        _assert_no_symlinks(output)

    protocol = _read_json(output / "protocol.json")
    if protocol != protocol_record():
        raise ValueError("saved VP-001 protocol does not match current sources")
    input_manifest = _read_json(output / "input-manifest.json")
    if input_manifest != _expected_input_manifest(output):
        raise ValueError("saved VP-001 input manifest or SS-001 snapshot has drifted")
    if result_path.exists():
        if not all((output / path).exists() for path in OUTCOME_PATHS):
            raise ValueError("VP-001 outcome package is partial")
        if _read_json(started_path) != _formal_start_record(output):
            raise ValueError("VP-001 formal-start record does not match")
        saved, arrays = _verify_outcomes(output, protocol, input_manifest)
        outcomes = "verified_results"
        if semantic:
            regenerated_arrays, regenerated_execution = _execute(output)
            for name in ARRAY_NAMES:
                if not np.array_equal(arrays[name], regenerated_arrays[name]):
                    raise ValueError(f"VP-001 semantic regeneration differs in {name}")
            if _canonical_json_value(saved["execution"]) != _canonical_json_value(regenerated_execution):
                raise ValueError("VP-001 semantic regeneration differs in execution records")
            context = _parent_context(output / PARENT_COPY)
            regenerated = _derive(regenerated_arrays, context)
            for field in (
                "parent_call_rows",
                "fresh_call_rows",
                "direction_seed_summaries",
                "sensitivity_seed_summaries",
                "decision",
            ):
                if _canonical_json_value(saved[field]) != _canonical_json_value(regenerated[field]):
                    raise ValueError(f"VP-001 semantic regeneration differs in {field}")
            outcomes = "verified_semantic_results"
    elif started_path.exists():
        raise ValueError("VP-001 formal execution started but is incomplete; identifier is terminal")
    elif any((output / path).exists() for path in OUTCOME_PATHS):
        raise ValueError("VP-001 has partial outcomes without a formal-start record")
    else:
        outcomes = "prepared_only"
    if require_results and outcomes == "prepared_only":
        raise ValueError("complete VP-001 results are required")
    return {
        "status": "verified",
        "outcomes": outcomes,
        "protocol_sha256": _canonical_json_sha256(protocol),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
    }


def development_audit(output: Path = REPO_ROOT / PARENT_OUTPUT) -> dict[str, object]:
    """Exercise frozen identities and full-union scoring with excluded seed 97."""

    _parent_snapshot(output)
    dataset = load_dataset(output / scorer.DATASET_COPY)
    context = _parent_context(output)
    _verify_fixed_sets(context)
    model = train_balanced_model(dataset, DEVELOPMENT_SEED)
    checked: list[dict[str, object]] = []
    for source_kind, seed, episode in (
        ("parent_visible", 8, 0),
        ("parent_visible", 9, 0),
        ("fresh_visible", 32, 1),
    ):
        seeds = PARENT_SOURCE_SEEDS if source_kind == "parent_visible" else FRESH_SOURCE_SEEDS
        seed_axis = seeds.index(seed)
        sequences, exact, raw, saved_row = _call_data(
            context,
            source_kind,
            seed_axis,
            episode,
        )
        payload = _identity_payload(sequences, exact, raw, saved_row)
        first = scorer._score_blocks(model, raw, sequences)
        repeated = scorer._score_blocks(model, raw, sequences)
        if not np.array_equal(first, repeated):
            raise ValueError("VP-001 excluded development scorer is not deterministic")
        scorer._verify_duplicate_scores(sequences, first)
        unique_flat = scorer._first_unique_indices(sequences.reshape(-1, injection.HORIZON, 2))
        ranks = average_tie_ranks(first.reshape(-1)[unique_flat], descending=True)
        identity_ranks = ranks[cast(np.ndarray, payload["union_indices"])]
        if saved_row["c_sequence_sha256"] == saved_row["x_sequence_sha256"] and (
            identity_ranks[1] != identity_ranks[2]
        ):
            raise ValueError("VP-001 development identity sentinel failed")
        checked.append(
            {
                "source_kind": source_kind,
                "seed": seed,
                "episode_index": episode,
                "unique_candidates": len(unique_flat),
                "identity_ranks": [float(value) for value in identity_ranks],
            }
        )
    return {
        "seed": DEVELOPMENT_SEED,
        "excluded": True,
        "model_sha256": oracle._model_fingerprint(model),
        "scores_finite": True,
        "repeat_exact": True,
        "checked_calls": checked,
    }


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify(output, require_results=True)
    return _read_json(output / RESULT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="VP-001 held-out validator-panel experiment")
    parser.add_argument(
        "command",
        choices=("development", "prepare", "run", "verify", "verify-semantic", "analyze"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "development":
        result = development_audit(REPO_ROOT / PARENT_OUTPUT)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "prepare":
        result = prepare(args.output)
        print(f"prepare: {result['status']}")
    elif args.command == "run":
        result = run(args.output)
        decision = cast(dict[str, object], result["decision"])
        print(f"run: {decision['classification']}")
    elif args.command == "analyze":
        result = analyze(args.output)
        decision = cast(dict[str, object], result["decision"])
        print(f"analyze: {decision['classification']}")
    else:
        result = verify(
            args.output,
            require_results=True,
            semantic=args.command == "verify-semantic",
        )
        print(f"verify: {result['outcomes']}")


if __name__ == "__main__":
    main()
