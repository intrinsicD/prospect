"""SS-001: prospective cross-seed scorer swap on fresh candidate generators.

The experiment is bench-only.  It consumes a byte-pinned CL-001 package, uses the
outcome-visible CL calls only for a frozen direction-control panel, and evaluates
the primary mechanism on untouched generator seeds with disjoint untouched auditor
seeds.  Learned-score magnitudes never cross model boundaries; only within-model
ranks and pairwise preferences do.
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
from bench.candidate_landscape import experiment as parent
from bench.oracle_ladder import experiment as oracle
from bench.oracle_ladder.audit import average_tie_ranks, exact_discounted_scores
from bench.proposal_injection import experiment as injection
from bench.proposal_injection.providers import ExactReferenceProvider, provider_summary
from prospect.planning import FlatPlanner
from prospect.world_model import FlatWorldModel

SCHEMA_VERSION = "scorer-swap-v1"
EXPERIMENT_ID = "SS-001"
AUDITOR_SEEDS = tuple(range(20, 32))
GENERATOR_SEEDS = tuple(range(32, 44))
PARENT_CALIBRATION_SEEDS = tuple(range(8, 20))
CALIBRATION_POSITIVE_SEEDS = (8, 9, 10, 11, 12, 18)
DEVELOPMENT_SEED = 97
SCORE_ATOL = 1e-12
RANK_DAMAGE_FLOOR = 8.0
CALL_VOTE_FLOOR = 9
SEED_SUPPORT_FLOOR = 10
RECURRENCE_START_FLOOR = 2
WITHIN_SEED_FRACTION = 0.75
CALIBRATION_SEED_FLOOR = 5
EXPECTED_UNIQUE_CANDIDATES = 186

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/scorer_swap/results/SS-001")
PROTOCOL_DOC = Path("docs/research/2026-07-15-scorer-swap-ss001-protocol.md")
PARENT_OUTPUT = Path("bench/candidate_landscape/results/CL-001")
PARENT_COPY = Path("inputs/CL-001")
PARENT_TENSOR_COPY = PARENT_COPY / parent.TENSOR_FILE
DATASET_COPY = PARENT_COPY / parent.INPUT_COPY
TENSOR_FILE = Path(f"{EXPERIMENT_ID}-audit.npz")
RESULT_FILE = Path(f"{EXPERIMENT_ID}-results.json")
CSV_FILE = Path(f"{EXPERIMENT_ID}-calls.csv")
REPORT_FILE = Path(f"{EXPERIMENT_ID}-report.md")
STARTED_FILE = Path("formal-start.json")

PARENT_HASHES = {
    "CL-001-calls.csv": "dbba501faf2b7ab7f1bfc21a324ca6c9c3c1a75c0fbdf6e4d0a0f0f49bed1333",
    "CL-001-candidates.npz": "3ba92ad46f7ea084e6a019374d6fa98cd96c2fd6d0cd0dabdc7a7671ed3f0c48",
    "CL-001-report.md": "b92f35139179593de6c8d0a6f4c5f79b81035c11372af28e40d193062e558783",
    "CL-001-results.json": "97c26b1bc1125fd7fb9046e2382f972c5e2efe9cb7e6763cc37a52e6b1664746",
    "artifact-manifest.json": "115eda81dd4411dbfb54ed630decc0cac2cf7bde77ca568eb7ffd0019593795d",
    "formal-start.json": "ff7b0f6bbfa20ffd6c21c3e91577e7df22fb4ebc57cd293d617f25f876e3937c",
    "input-manifest.json": "1d98d5c541d50b5c91fb10471f79b3e48bad401483d96b951df9f86e0fb3902c",
    "inputs/BC-001-b1_r1_d8.npz": "9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948",
    "protocol.json": "f9b92042341119ac99043d19cae67c7630d05b4be8d95ca8ccfe4dd7d0d03817",
}
PARENT_PROTOCOL_CANONICAL_SHA256 = "8755884d2b443c46b4e01d881ee2ea8d6731ea7e4471bccf3e796680bae9fee1"
PARENT_TENSOR_SEMANTIC_SHA256 = "ea68157d93d076523f57e1d5bba5c28931cd6f51c028bd5278ba4d3a6d40937d"
PARENT_CLASSIFICATION = "neither_mechanism_supported"

PROTECTED_OUTPUTS = (
    PARENT_OUTPUT,
    Path("bench/proposal_injection/results/PI-001"),
    Path("bench/proposal_injection_v2/results/PI-002"),
    Path("bench/proposal_injection_v3/results/PI-003"),
    Path("bench/oracle_ladder_v2/results/OL-002"),
    Path("bench/bridge_control/results/BC-001"),
)

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *parent.SOURCE_FILES,
            Path("bench/scorer_swap/__init__.py"),
            Path("bench/scorer_swap/__main__.py"),
            Path("bench/scorer_swap/experiment.py"),
            Path("tests/test_scorer_swap_experiment.py"),
            PROTOCOL_DOC,
        )
    )
)

ARRAY_NAMES = (
    "generator_seeds",
    "auditor_seeds",
    "fresh_sequences",
    "fresh_generator_scores",
    "fresh_exact_scores",
    "fresh_injected",
    "fresh_raw_states",
    "fresh_auditor_scores",
    "parent_auditor_scores",
)

OUTCOME_PATHS = (
    STARTED_FILE,
    TENSOR_FILE,
    RESULT_FILE,
    CSV_FILE,
    REPORT_FILE,
    Path("artifact-manifest.json"),
)
ARTIFACT_PATHS = (
    Path("protocol.json"),
    Path("input-manifest.json"),
    *(PARENT_COPY / Path(name) for name in PARENT_HASHES),
    STARTED_FILE,
    TENSOR_FILE,
    RESULT_FILE,
    CSV_FILE,
    REPORT_FILE,
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
    digest.update(b"SS-001 canonical tensor package\0")
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
        if set(package.files) != set(ARRAY_NAMES):
            raise ValueError("SS-001 tensor package has unexpected array names")
        return {name: np.asarray(package[name]).copy() for name in ARRAY_NAMES}


def _expected_parent_manifest() -> dict[str, object]:
    return {
        "schema_version": parent.SCHEMA_VERSION,
        "experiment_id": parent.EXPERIMENT_ID,
        "artifacts": {name: digest for name, digest in PARENT_HASHES.items() if name != "artifact-manifest.json"},
    }


def _parent_snapshot(path: Path) -> dict[str, object]:
    verification = parent.verify(path, require_results=True)
    if verification["outcomes"] != "verified_results":
        raise ValueError("SS-001 requires verified CL-001 results")
    actual_files = {str(candidate.relative_to(path)) for candidate in path.rglob("*") if candidate.is_file()}
    if actual_files != set(PARENT_HASHES):
        raise ValueError("CL-001 package file set drifted")
    for name, expected in PARENT_HASHES.items():
        if _file_hash(path / name) != expected:
            raise ValueError(f"CL-001 hash drifted for {name}")
    if _read_json(path / "artifact-manifest.json") != _expected_parent_manifest():
        raise ValueError("CL-001 artifact manifest entry map drifted")
    protocol = _read_json(path / "protocol.json")
    if _canonical_json_sha256(protocol) != PARENT_PROTOCOL_CANONICAL_SHA256:
        raise ValueError("CL-001 canonical protocol hash drifted")
    results = _read_json(path / parent.RESULT_FILE)
    tensor = cast(dict[str, Any], results["tensor_package"])
    decision = cast(dict[str, Any], results["decision"])
    if (
        results.get("schema_version") != parent.SCHEMA_VERSION
        or results.get("experiment_id") != parent.EXPERIMENT_ID
        or results.get("status") != "completed_candidate_landscape"
        or decision.get("classification") != PARENT_CLASSIFICATION
        or tensor.get("semantic_sha256") != PARENT_TENSOR_SEMANTIC_SHA256
        or tensor.get("file_sha256") != PARENT_HASHES[str(parent.TENSOR_FILE)]
    ):
        raise ValueError("CL-001 result identity, decision, or tensor metadata drifted")
    return {
        "experiment_id": parent.EXPERIMENT_ID,
        "verification": verification["outcomes"],
        "hashes": dict(PARENT_HASHES),
        "protocol_canonical_sha256": PARENT_PROTOCOL_CANONICAL_SHA256,
        "tensor_semantic_sha256": PARENT_TENSOR_SEMANTIC_SHA256,
        "classification": PARENT_CLASSIFICATION,
        "epistemic_role": "outcome-visible calibration and transport evidence only",
    }


def protocol_record() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_formal_execution",
        "protocol_document": str(PROTOCOL_DOC),
        "protocol_document_sha256": _file_hash(REPO_ROOT / PROTOCOL_DOC),
        "source_sha256": _source_hashes(),
        "runtime_constraints": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "scientific_scope": (
            "prospective generator-restart diagnostic with a same-data, same-architecture auditor panel"
        ),
        "terminology": ("cross-seed scorer swap; not classical data cross-fitting or independent-data validation"),
        "seeds": {
            "auditors": list(AUDITOR_SEEDS),
            "fresh_generators": list(GENERATOR_SEEDS),
            "parent_calibration": list(PARENT_CALIBRATION_SEEDS),
            "development_excluded": DEVELOPMENT_SEED,
        },
        "experimental_unit": (
            "fresh generator seed; four starts are repeated measures and auditors form one fixed panel"
        ),
        "training": {"steps": 1_800, "batch_size": 64, "dataset": "BC-001/b1_r1_d8"},
        "generation": {
            "planner_arm": "privileged_injection",
            "episode_steps": injection.EVAL_STEPS,
            "retained_step": 0,
            "rounds": injection.NATIVE_ITERATIONS,
            "candidates_per_round": injection.NATIVE_CANDIDATES,
            "expected_unique_candidates": EXPECTED_UNIQUE_CANDIDATES,
            "horizon": injection.HORIZON,
            "uncertainty_penalty": 0.0,
            "ordinary_instrumented_parity_required": True,
        },
        "auditor_scoring": {
            "all_to_all": True,
            "own_encoder_per_auditor": True,
            "discount": 0.99,
            "uncertainty_penalty": 0.0,
            "epistemic_horizon_bound": None,
            "aggregation": "mean normalized within-auditor average-tie rank",
            "raw_cross_model_score_aggregation_forbidden": True,
            "tie_break": "first round-major candidate occurrence",
        },
        "thresholds": {
            "score_atol": SCORE_ATOL,
            "rank_damage_floor": RANK_DAMAGE_FLOOR,
            "call_vote_floor": CALL_VOTE_FLOOR,
            "seed_support_floor": SEED_SUPPORT_FLOOR,
            "recurrence_start_floor": RECURRENCE_START_FLOOR,
            "within_seed_fraction": WITHIN_SEED_FRACTION,
            "calibration_seed_floor": CALIBRATION_SEED_FLOOR,
            "calibration_seed_count": len(CALIBRATION_POSITIVE_SEEDS),
            "statistical_role": "descriptive robustness thresholds; never binomial tests",
        },
        "classification_order": [
            "invalid_on_integrity_or_parity_failure",
            "parent_signature_not_reproduced",
            "auditor_direction_control_failed",
            "mechanism_branch_plus_rescue_branch",
        ],
        "mechanism_branches": [
            "restart_specific_exploitation",
            "same_data_shared_bias",
            "heterogeneous_cross_model_transfer",
        ],
        "rescue_branches": [
            "cross_seed_rank_rescue",
            "no_robust_cross_seed_rank_rescue",
        ],
        "stop_rules": [
            "write the atomic formal-start marker before training any seed 20..43",
            "run every frozen seed without outcome-dependent stopping or retry",
            "never change roles, inputs, metrics, thresholds, or branches after start",
            "a post-start failure is terminal for SS-001",
            "do not modify production code, tasks, ADRs, gates, or sealed evidence packages",
        ],
        "interpretation_limits": [
            "same-data restart agreement cannot localize dataset, encoder, dynamics, or reward-head bias",
            "parent rows are visible controls and never enter fresh primary counts",
            "offline exact candidate rescue is not episode-return or production-control evidence",
        ],
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    copied_parent = output / PARENT_COPY
    dataset = load_dataset(output / DATASET_COPY)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": _canonical_json_sha256(protocol_record()),
        "original_parent": _parent_snapshot(REPO_ROOT / PARENT_OUTPUT),
        "copied_parent": _parent_snapshot(copied_parent),
        "copied_dataset": {
            "path": str(DATASET_COPY),
            "file_sha256": _file_hash(output / DATASET_COPY),
            "semantic_sha256": semantic_hash(dataset),
            "name": dataset.name,
        },
        "prior_parent_semantic_verification": {
            "status": "completed_before_SS-001_design",
            "receipt_scope": "prerequisite record; child fast verification independently pins raw parent bytes",
        },
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_safe_output(output: Path) -> None:
    resolved = output.resolve()
    repo_resolved = REPO_ROOT.resolve()
    if resolved == repo_resolved or resolved in repo_resolved.parents:
        raise ValueError("SS-001 output cannot be the repository root or an ancestor")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("SS-001 output must be a real directory path")
    for protected in PROTECTED_OUTPUTS:
        protected_resolved = (REPO_ROOT / protected).resolve()
        if _paths_overlap(resolved, protected_resolved):
            raise ValueError(f"SS-001 output overlaps protected evidence package {protected}")
    owned_results = (REPO_ROOT / "bench/scorer_swap/results").resolve()
    if repo_resolved in resolved.parents and owned_results not in resolved.parents:
        raise ValueError("in-repository SS-001 output must be below bench/scorer_swap/results")
    if resolved == owned_results:
        raise ValueError("SS-001 output must be below, not equal to, its results root")
    if (output / RESULT_FILE).exists():
        raise FileExistsError("SS-001 is already formal; preserve it and use a new id")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Freeze SS-001 sources, complete parent copy, and protocol without outcomes."""

    _parent_snapshot(REPO_ROOT / PARENT_OUTPUT)
    _assert_safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("SS-001 output must be absent or an empty directory")
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
        "parent_manifest_sha256": PARENT_HASHES["artifact-manifest.json"],
        "parent_tensor_semantic_sha256": PARENT_TENSOR_SEMANTIC_SHA256,
    }


def _mark_formal_started(output: Path) -> dict[str, object]:
    """Durably consume the identifier before any formal model training."""

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
    generators = len(GENERATOR_SEEDS)
    auditors = len(AUDITOR_SEEDS)
    calls = (generators, len(EVAL_STARTS))
    pools = calls + (injection.NATIVE_ITERATIONS, injection.NATIVE_CANDIDATES)
    auditor_pools = (auditors,) + pools
    return {
        "generator_seeds": np.asarray(GENERATOR_SEEDS, dtype=np.int64),
        "auditor_seeds": np.asarray(AUDITOR_SEEDS, dtype=np.int64),
        "fresh_sequences": np.empty(
            pools + (injection.HORIZON, 2),
            dtype=np.float64,
        ),
        "fresh_generator_scores": np.empty(pools, dtype=np.float64),
        "fresh_exact_scores": np.empty(pools, dtype=np.float64),
        "fresh_injected": np.empty(pools, dtype=bool),
        "fresh_raw_states": np.empty(calls + (3,), dtype=np.float64),
        "fresh_auditor_scores": np.empty(auditor_pools, dtype=np.float64),
        "parent_auditor_scores": np.empty(auditor_pools, dtype=np.float64),
    }


def _validate_array_shapes(arrays: dict[str, np.ndarray]) -> None:
    expected = _empty_arrays()
    if set(arrays) != set(expected):
        raise ValueError("SS-001 tensor arrays are incomplete")
    for name, template in expected.items():
        value = arrays[name]
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"SS-001 tensor {name} has {value.shape}/{value.dtype}, expected {template.shape}/{template.dtype}"
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"SS-001 tensor {name} contains non-finite values")
    if not np.array_equal(arrays["generator_seeds"], np.asarray(GENERATOR_SEEDS, dtype=np.int64)):
        raise ValueError("SS-001 generator seed axis differs from the frozen protocol")
    if not np.array_equal(arrays["auditor_seeds"], np.asarray(AUDITOR_SEEDS, dtype=np.int64)):
        raise ValueError("SS-001 auditor seed axis differs from the frozen protocol")


def _parent_step0_arrays(parent_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    seed_to_axis = {int(seed): axis for axis, seed in enumerate(parent_arrays["seeds"].tolist())}
    if not set(PARENT_CALIBRATION_SEEDS).issubset(seed_to_axis):
        raise ValueError("CL-001 tensor lacks frozen calibration seeds")
    axes = np.asarray([seed_to_axis[seed] for seed in PARENT_CALIBRATION_SEEDS], dtype=np.int64)
    return {
        "sequences": np.asarray(parent_arrays["sequences"][axes, :, 0], dtype=np.float64),
        "generator_scores": np.asarray(
            parent_arrays["learned_scores"][axes, :, 0],
            dtype=np.float64,
        ),
        "exact_scores": np.asarray(parent_arrays["exact_scores"][axes, :, 0], dtype=np.float64),
        "injected": np.asarray(parent_arrays["injected"][axes, :, 0], dtype=bool),
        "raw_states": np.asarray(parent_arrays["raw_states"][axes, :, 0], dtype=np.float64),
    }


def _scoring_planner(model: FlatWorldModel) -> FlatPlanner:
    return FlatPlanner(
        model,
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=injection.HORIZON,
        candidates=injection.NATIVE_CANDIDATES,
        elites=injection.NATIVE_ELITES,
        iterations=injection.NATIVE_ITERATIONS,
        discount=0.99,
        uncertainty_penalty=0.0,
        seed=0,
        epistemic_horizon_bound=None,
    )


def _score_blocks(
    model: FlatWorldModel,
    raw_state: np.ndarray,
    sequences: np.ndarray,
) -> np.ndarray:
    blocks = np.asarray(sequences, dtype=np.float64)
    expected = (
        injection.NATIVE_ITERATIONS,
        injection.NATIVE_CANDIDATES,
        injection.HORIZON,
        2,
    )
    if blocks.shape != expected:
        raise ValueError(f"SS-001 scorer expected sequence blocks {expected}, got {blocks.shape}")
    flat = blocks.reshape(-1, injection.HORIZON, 2)
    latent = model.encode(np.asarray(raw_state, dtype=np.float64))
    scores = _scoring_planner(model)._imagined_returns(latent, flat)
    shaped = np.asarray(scores, dtype=np.float64).reshape(
        injection.NATIVE_ITERATIONS,
        injection.NATIVE_CANDIDATES,
    )
    if not np.all(np.isfinite(shaped)):
        raise ValueError("SS-001 auditor scorer produced non-finite values")
    return shaped


def _fresh_generator(
    dataset_path: Path,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    dataset = load_dataset(dataset_path)
    model = train_balanced_model(dataset, seed)
    fingerprint = oracle._model_fingerprint(model)

    ordinary_provider = ExactReferenceProvider(seed, "privileged")
    ordinary_planner = injection._injection_planner(model, seed, ordinary_provider)
    ordinary_evaluation = injection._evaluate(
        ordinary_planner,
        injection._sidecar_encoder(model),
    )
    ordinary_row = parent._evaluation_row(
        seed,
        fingerprint,
        ordinary_evaluation,
        ordinary_provider,
    )

    audited_provider = ExactReferenceProvider(seed, "privileged")
    audited_planner = parent._audited_planner(model, seed, audited_provider)
    audited_evaluation = injection._evaluate(
        audited_planner,
        injection._sidecar_encoder(model),
    )
    audited_row = parent._evaluation_row(
        seed,
        fingerprint,
        audited_evaluation,
        audited_provider,
    )
    difference = injection._assert_parity(
        f"SS-001 seed {seed} ordinary versus instrumented",
        ordinary_evaluation,
        audited_evaluation,
    )
    if _canonical_json_value(ordinary_row) != _canonical_json_value(audited_row):
        raise ValueError(f"SS-001 seed {seed} instrumented row differs from ordinary planner")
    if _canonical_json_value(provider_summary(ordinary_provider)) != _canonical_json_value(
        provider_summary(audited_provider)
    ):
        raise ValueError(f"SS-001 seed {seed} provider summaries differ across parity runs")

    shape = (
        len(EVAL_STARTS),
        injection.NATIVE_ITERATIONS,
        injection.NATIVE_CANDIDATES,
    )
    sequences = np.empty(shape + (injection.HORIZON, 2), dtype=np.float64)
    learned = np.empty(shape, dtype=np.float64)
    exact = np.empty(shape, dtype=np.float64)
    injected = np.empty(shape, dtype=bool)
    raw_states = np.empty((len(EVAL_STARTS), 3), dtype=np.float64)
    grouped = parent._pool_groups(audited_planner.pool_audits)
    for episode in range(len(EVAL_STARTS)):
        records = grouped[(episode, 0)]
        parent._validate_pool_lineage(records)
        first = records[0]
        if first.call_index != episode * injection.EVAL_STEPS:
            raise ValueError("SS-001 fresh step-0 call index is not provider aligned")
        if not np.array_equal(first.raw_state, np.asarray(EVAL_STARTS[episode], dtype=np.float64)):
            raise ValueError("SS-001 fresh step-0 state differs from the frozen start")
        raw_states[episode] = first.raw_state
        for iteration, record in enumerate(records):
            sequences[episode, iteration] = record.sequences
            learned[episode, iteration] = record.learned_scores
            exact[episode, iteration] = record.exact_scores
            injected[episode, iteration] = record.injected
        rescored = _score_blocks(model, first.raw_state, sequences[episode])
        if not np.array_equal(rescored, learned[episode]):
            raise ValueError("SS-001 scorer-swap path differs from generator planner scores")

    return (
        {
            "sequences": sequences,
            "generator_scores": learned,
            "exact_scores": exact,
            "injected": injected,
            "raw_states": raw_states,
        },
        {
            "seed": seed,
            "model_sha256": fingerprint,
            "ordinary_instrumented_difference": difference,
            "ordinary_row_sha256": _canonical_json_sha256(ordinary_row),
            "instrumented_row_sha256": _canonical_json_sha256(audited_row),
            "provider_summary_sha256": _canonical_json_sha256(provider_summary(audited_provider)),
            "provider_summaries_canonical_equal": True,
        },
    )


def _execute(output: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    dataset_path = output / DATASET_COPY
    dataset = load_dataset(dataset_path)
    if dataset.name != "b1_r1_d8":
        raise ValueError("SS-001 requires the frozen BC-001 b1_r1_d8 dataset")
    parent_arrays = parent._load_arrays(output / PARENT_TENSOR_COPY)
    parent_step0 = _parent_step0_arrays(parent_arrays)
    arrays = _empty_arrays()

    generator_records: list[dict[str, object]] = []
    for generator_axis, seed in enumerate(GENERATOR_SEEDS):
        generated, record = _fresh_generator(dataset_path, seed)
        arrays["fresh_sequences"][generator_axis] = generated["sequences"]
        arrays["fresh_generator_scores"][generator_axis] = generated["generator_scores"]
        arrays["fresh_exact_scores"][generator_axis] = generated["exact_scores"]
        arrays["fresh_injected"][generator_axis] = generated["injected"]
        arrays["fresh_raw_states"][generator_axis] = generated["raw_states"]
        generator_records.append(record)

    auditor_records: list[dict[str, object]] = []
    for auditor_axis, seed in enumerate(AUDITOR_SEEDS):
        model = train_balanced_model(dataset, seed)
        fingerprint = oracle._model_fingerprint(model)
        for generator_axis in range(len(GENERATOR_SEEDS)):
            for episode in range(len(EVAL_STARTS)):
                arrays["fresh_auditor_scores"][auditor_axis, generator_axis, episode] = _score_blocks(
                    model,
                    arrays["fresh_raw_states"][generator_axis, episode],
                    arrays["fresh_sequences"][generator_axis, episode],
                )
                arrays["parent_auditor_scores"][auditor_axis, generator_axis, episode] = _score_blocks(
                    model,
                    parent_step0["raw_states"][generator_axis, episode],
                    parent_step0["sequences"][generator_axis, episode],
                )
        auditor_records.append({"seed": seed, "model_sha256": fingerprint})

    _validate_array_shapes(arrays)
    return arrays, {
        "generator_records": generator_records,
        "auditor_records": auditor_records,
    }


def _sequence_bytes(sequence: np.ndarray) -> bytes:
    return np.asarray(sequence, dtype="<f8", order="C").tobytes(order="C")


def _sequence_sha256(sequence: np.ndarray) -> str:
    return sha256(_sequence_bytes(sequence)).hexdigest()


def _first_unique_indices(sequences: np.ndarray) -> np.ndarray:
    flat = np.asarray(sequences, dtype=np.float64)
    if flat.ndim != 3 or flat.shape[1:] != (injection.HORIZON, 2):
        raise ValueError("SS-001 unique sequence input has the wrong shape")
    seen: set[bytes] = set()
    indices: list[int] = []
    for index, sequence in enumerate(flat):
        identity = _sequence_bytes(sequence)
        if identity not in seen:
            seen.add(identity)
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _harmful_signature(
    *,
    b0_injected: bool,
    c_iteration: int,
    learned_gain: float,
    exact_delta: float,
    exact_rank_damage: float,
) -> bool:
    return bool(
        b0_injected
        and c_iteration >= 1
        and learned_gain > SCORE_ATOL
        and exact_delta < -SCORE_ATOL
        and exact_rank_damage >= RANK_DAMAGE_FLOOR
    )


def _derive_call(
    *,
    source_kind: str,
    seed: int,
    episode_index: int,
    sequences: np.ndarray,
    generator_scores: np.ndarray,
    exact_scores: np.ndarray,
    injected: np.ndarray,
    auditor_scores: np.ndarray,
) -> dict[str, object]:
    if source_kind not in ("fresh_primary", "parent_visible_control"):
        raise ValueError("SS-001 call source kind is invalid")
    block_shape = (injection.NATIVE_ITERATIONS, injection.NATIVE_CANDIDATES)
    sequence_shape = block_shape + (injection.HORIZON, 2)
    if np.asarray(sequences).shape != sequence_shape:
        raise ValueError("SS-001 call sequences have the wrong shape")
    for name, value in (
        ("generator_scores", generator_scores),
        ("exact_scores", exact_scores),
        ("injected", injected),
    ):
        if np.asarray(value).shape != block_shape:
            raise ValueError(f"SS-001 call {name} has the wrong shape")
    expected_auditor_shape = (len(AUDITOR_SEEDS),) + block_shape
    if np.asarray(auditor_scores).shape != expected_auditor_shape:
        raise ValueError("SS-001 call auditor scores have the wrong shape")

    flat_sequences = np.asarray(sequences, dtype=np.float64).reshape(
        -1,
        injection.HORIZON,
        2,
    )
    flat_generator = np.asarray(generator_scores, dtype=np.float64).reshape(-1)
    flat_exact = np.asarray(exact_scores, dtype=np.float64).reshape(-1)
    flat_injected = np.asarray(injected, dtype=bool).reshape(-1)
    flat_auditors = np.asarray(auditor_scores, dtype=np.float64).reshape(
        len(AUDITOR_SEEDS),
        -1,
    )
    if not all(np.all(np.isfinite(value)) for value in (flat_sequences, flat_generator, flat_exact, flat_auditors)):
        raise ValueError("SS-001 call tensors must be finite")

    unique_flat_indices = _first_unique_indices(flat_sequences)
    if len(unique_flat_indices) != EXPECTED_UNIQUE_CANDIDATES:
        raise ValueError(
            f"SS-001 call has {len(unique_flat_indices)} unique candidates, expected {EXPECTED_UNIQUE_CANDIDATES}"
        )
    unique_sequences = flat_sequences[unique_flat_indices]
    unique_exact = flat_exact[unique_flat_indices]
    unique_auditors = flat_auditors[:, unique_flat_indices]
    identity_to_union = {_sequence_bytes(sequence): index for index, sequence in enumerate(unique_sequences)}

    b0_flat = int(np.argmax(flat_generator[: injection.NATIVE_CANDIDATES]))
    c_flat = int(np.argmax(flat_generator))
    b0_union = identity_to_union[_sequence_bytes(flat_sequences[b0_flat])]
    c_union = identity_to_union[_sequence_bytes(flat_sequences[c_flat])]
    c_iteration, c_candidate = divmod(c_flat, injection.NATIVE_CANDIDATES)
    b0_score = float(flat_generator[b0_flat])
    c_score = float(flat_generator[c_flat])
    b0_exact = float(flat_exact[b0_flat])
    c_exact = float(flat_exact[c_flat])
    exact_ranks = average_tie_ranks(unique_exact, descending=True)
    b0_exact_rank = float(exact_ranks[b0_union])
    c_exact_rank = float(exact_ranks[c_union])
    learned_gain = c_score - b0_score
    exact_delta = c_exact - b0_exact
    exact_rank_damage = c_exact_rank - b0_exact_rank
    harmful = _harmful_signature(
        b0_injected=bool(flat_injected[b0_flat]),
        c_iteration=c_iteration,
        learned_gain=learned_gain,
        exact_delta=exact_delta,
        exact_rank_damage=exact_rank_damage,
    )

    auditor_ranks = np.stack([average_tie_ranks(scores, descending=True) for scores in unique_auditors])
    b0_auditor_ranks = auditor_ranks[:, b0_union]
    c_auditor_ranks = auditor_ranks[:, c_union]
    reject_votes = int(np.sum(b0_auditor_ranks < c_auditor_ranks))
    transfer_votes = int(np.sum(c_auditor_ranks < b0_auditor_ranks))
    tie_votes = len(AUDITOR_SEEDS) - reject_votes - transfer_votes
    normalized = (auditor_ranks - 1.0) / float(EXPECTED_UNIQUE_CANDIDATES - 1)
    aggregate = np.mean(normalized, axis=0)
    x_union = int(np.argmin(aggregate))
    x_exact = float(unique_exact[x_union])
    x_exact_rank = float(exact_ranks[x_union])
    x_exact_delta_from_c = x_exact - c_exact
    x_exact_rank_gain_from_c = c_exact_rank - x_exact_rank
    exact_rescue = bool(harmful and x_exact_delta_from_c > SCORE_ATOL and x_exact_rank_gain_from_c >= RANK_DAMAGE_FLOOR)
    material_degradation = bool(x_exact_delta_from_c < -SCORE_ATOL and x_exact_rank - c_exact_rank >= RANK_DAMAGE_FLOOR)
    exact_improving_control = bool(exact_delta > SCORE_ATOL)
    calibration_call_pass = bool(
        exact_improving_control and transfer_votes >= CALL_VOTE_FLOOR and not material_degradation
    )

    return {
        "source_kind": source_kind,
        "seed": seed,
        "episode_index": episode_index,
        "unique_candidates": len(unique_flat_indices),
        "b0_flat_index": b0_flat,
        "b0_union_index": b0_union,
        "b0_sequence_sha256": _sequence_sha256(unique_sequences[b0_union]),
        "b0_injected": bool(flat_injected[b0_flat]),
        "b0_generator_score": b0_score,
        "b0_exact_score": b0_exact,
        "b0_exact_rank": b0_exact_rank,
        "c_flat_index": c_flat,
        "c_iteration": c_iteration,
        "c_candidate": c_candidate,
        "c_union_index": c_union,
        "c_sequence_sha256": _sequence_sha256(unique_sequences[c_union]),
        "c_generator_score": c_score,
        "c_exact_score": c_exact,
        "c_exact_rank": c_exact_rank,
        "generator_learned_gain": learned_gain,
        "c_exact_delta_from_b0": exact_delta,
        "c_exact_rank_damage_from_b0": exact_rank_damage,
        "harmful_signature": harmful,
        "exact_improving_control": exact_improving_control,
        "auditor_reject_votes": reject_votes,
        "auditor_transfer_votes": transfer_votes,
        "auditor_tie_votes": tie_votes,
        "restart_rejected": reject_votes >= CALL_VOTE_FLOOR,
        "shared_transfer": transfer_votes >= CALL_VOTE_FLOOR,
        "x_union_index": x_union,
        "x_sequence_sha256": _sequence_sha256(unique_sequences[x_union]),
        "x_mean_normalized_rank": float(aggregate[x_union]),
        "x_exact_score": x_exact,
        "x_exact_rank": x_exact_rank,
        "x_exact_delta_from_c": x_exact_delta_from_c,
        "x_exact_rank_gain_from_c": x_exact_rank_gain_from_c,
        "exact_rescue": exact_rescue,
        "x_materially_degrades_c": material_degradation,
        "calibration_call_pass": calibration_call_pass,
    }


def _seed_summaries(
    fresh_rows: list[dict[str, object]],
    calibration_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    fresh_summaries: list[dict[str, object]] = []
    for seed in GENERATOR_SEEDS:
        rows = [row for row in fresh_rows if int(cast(int, row["seed"])) == seed]
        if len(rows) != len(EVAL_STARTS):
            raise ValueError(f"SS-001 fresh generator {seed} does not have four starts")
        harmful = [row for row in rows if bool(row["harmful_signature"])]
        harmful_count = len(harmful)
        required = int(math.ceil(WITHIN_SEED_FRACTION * harmful_count))
        recurrence = harmful_count >= RECURRENCE_START_FLOOR
        rejection_count = sum(bool(row["restart_rejected"]) for row in harmful)
        transfer_count = sum(bool(row["shared_transfer"]) for row in harmful)
        rescue_count = sum(bool(row["exact_rescue"]) for row in harmful)
        fresh_summaries.append(
            {
                "seed": seed,
                "harmful_calls": harmful_count,
                "recurrence_required_calls": RECURRENCE_START_FLOOR,
                "recurrence_support": recurrence,
                "within_seed_required_calls": required,
                "restart_rejected_calls": rejection_count,
                "shared_transfer_calls": transfer_count,
                "exact_rescue_calls": rescue_count,
                "restart_rejection_support": recurrence and rejection_count >= required,
                "shared_transfer_support": recurrence and transfer_count >= required,
                "exact_rescue_support": recurrence and rescue_count >= required,
            }
        )

    calibration_summaries: list[dict[str, object]] = []
    positive_seed_set: set[int] = set()
    for seed in PARENT_CALIBRATION_SEEDS:
        rows = [
            row
            for row in calibration_rows
            if int(cast(int, row["seed"])) == seed and bool(row["exact_improving_control"])
        ]
        if not rows:
            continue
        positive_seed_set.add(seed)
        required = int(math.ceil(WITHIN_SEED_FRACTION * len(rows)))
        passing = sum(bool(row["calibration_call_pass"]) for row in rows)
        calibration_summaries.append(
            {
                "seed": seed,
                "positive_calls": len(rows),
                "required_calls": required,
                "passing_calls": passing,
                "supports": passing >= required,
            }
        )
    if positive_seed_set != set(CALIBRATION_POSITIVE_SEEDS):
        raise ValueError("SS-001 parent exact-improving calibration seed identity drifted")
    if sum(int(cast(int, row["positive_calls"])) for row in calibration_summaries) != 11:
        raise ValueError("SS-001 parent exact-improving calibration call count drifted")
    return {"fresh": fresh_summaries, "calibration": calibration_summaries}


def _decision(
    fresh_summaries: list[dict[str, object]],
    calibration_summaries: list[dict[str, object]],
) -> dict[str, object]:
    if [int(cast(int, row["seed"])) for row in fresh_summaries] != list(GENERATOR_SEEDS):
        raise ValueError("SS-001 fresh summary seed order is incomplete")
    if [int(cast(int, row["seed"])) for row in calibration_summaries] != list(CALIBRATION_POSITIVE_SEEDS):
        raise ValueError("SS-001 calibration summary seed order is incomplete")
    recurrence_count = sum(bool(row["recurrence_support"]) for row in fresh_summaries)
    rejection_count = sum(bool(row["restart_rejection_support"]) for row in fresh_summaries)
    transfer_count = sum(bool(row["shared_transfer_support"]) for row in fresh_summaries)
    rescue_count = sum(bool(row["exact_rescue_support"]) for row in fresh_summaries)
    calibration_count = sum(bool(row["supports"]) for row in calibration_summaries)
    recurrence_passes = recurrence_count >= SEED_SUPPORT_FLOOR
    calibration_passes = calibration_count >= CALIBRATION_SEED_FLOOR
    if not recurrence_passes:
        classification = "parent_signature_not_reproduced"
        mechanism = "not_interpretable"
        rescue = "not_interpretable"
    elif not calibration_passes:
        classification = "auditor_direction_control_failed"
        mechanism = "not_interpretable"
        rescue = "not_interpretable"
    else:
        rejection_passes = rejection_count >= SEED_SUPPORT_FLOOR
        transfer_passes = transfer_count >= SEED_SUPPORT_FLOOR
        if rejection_passes and transfer_passes:
            raise ValueError("SS-001 mutually exclusive mechanism branches both passed")
        if rejection_passes:
            mechanism = "restart_specific_exploitation"
        elif transfer_passes:
            mechanism = "same_data_shared_bias"
        else:
            mechanism = "heterogeneous_cross_model_transfer"
        rescue = "cross_seed_rank_rescue" if rescue_count >= SEED_SUPPORT_FLOOR else "no_robust_cross_seed_rank_rescue"
        classification = f"{mechanism}+{rescue}"
    return {
        "classification": classification,
        "mechanism": mechanism,
        "rescue": rescue,
        "fresh_generator_seeds": list(GENERATOR_SEEDS),
        "parent_calibration_seeds_excluded_from_primary": list(PARENT_CALIBRATION_SEEDS),
        "recurrence_gate": {
            "supporting_seeds": recurrence_count,
            "required_seeds": SEED_SUPPORT_FLOOR,
            "passes": recurrence_passes,
        },
        "calibration_gate": {
            "supporting_seeds": calibration_count,
            "required_seeds": CALIBRATION_SEED_FLOOR,
            "eligible_seeds": len(CALIBRATION_POSITIVE_SEEDS),
            "passes": calibration_passes,
        },
        "restart_rejection": {
            "supporting_seeds": rejection_count,
            "required_seeds": SEED_SUPPORT_FLOOR,
            "interpretable": recurrence_passes and calibration_passes,
        },
        "shared_transfer": {
            "supporting_seeds": transfer_count,
            "required_seeds": SEED_SUPPORT_FLOOR,
            "interpretable": recurrence_passes and calibration_passes,
        },
        "exact_rescue": {
            "supporting_seeds": rescue_count,
            "required_seeds": SEED_SUPPORT_FLOOR,
            "interpretable": recurrence_passes and calibration_passes,
        },
    }


def _derive(
    arrays: dict[str, np.ndarray],
    parent_arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    parent_step0 = _parent_step0_arrays(parent_arrays)
    fresh_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for seed_axis, seed in enumerate(GENERATOR_SEEDS):
        for episode in range(len(EVAL_STARTS)):
            fresh_rows.append(
                _derive_call(
                    source_kind="fresh_primary",
                    seed=seed,
                    episode_index=episode,
                    sequences=arrays["fresh_sequences"][seed_axis, episode],
                    generator_scores=arrays["fresh_generator_scores"][seed_axis, episode],
                    exact_scores=arrays["fresh_exact_scores"][seed_axis, episode],
                    injected=arrays["fresh_injected"][seed_axis, episode],
                    auditor_scores=arrays["fresh_auditor_scores"][:, seed_axis, episode],
                )
            )
    for seed_axis, seed in enumerate(PARENT_CALIBRATION_SEEDS):
        for episode in range(len(EVAL_STARTS)):
            calibration_rows.append(
                _derive_call(
                    source_kind="parent_visible_control",
                    seed=seed,
                    episode_index=episode,
                    sequences=parent_step0["sequences"][seed_axis, episode],
                    generator_scores=parent_step0["generator_scores"][seed_axis, episode],
                    exact_scores=parent_step0["exact_scores"][seed_axis, episode],
                    injected=parent_step0["injected"][seed_axis, episode],
                    auditor_scores=arrays["parent_auditor_scores"][:, seed_axis, episode],
                )
            )
    summaries = _seed_summaries(fresh_rows, calibration_rows)
    return {
        "fresh_call_rows": fresh_rows,
        "parent_calibration_call_rows": calibration_rows,
        "fresh_seed_summaries": summaries["fresh"],
        "calibration_seed_summaries": summaries["calibration"],
        "decision": _decision(summaries["fresh"], summaries["calibration"]),
    }


def _verify_duplicate_scores(sequences: np.ndarray, score_rows: np.ndarray) -> None:
    flat_sequences = np.asarray(sequences, dtype=np.float64).reshape(
        -1,
        injection.HORIZON,
        2,
    )
    scores = np.asarray(score_rows, dtype=np.float64).reshape((-1, len(flat_sequences)))
    first_by_identity: dict[bytes, int] = {}
    for index, sequence in enumerate(flat_sequences):
        identity = _sequence_bytes(sequence)
        first = first_by_identity.setdefault(identity, index)
        if not np.array_equal(scores[:, index], scores[:, first]):
            raise ValueError("SS-001 duplicate sequence scores are inconsistent")


def _verify_fresh_lineage(
    arrays: dict[str, np.ndarray],
    parent_arrays: dict[str, np.ndarray],
) -> None:
    keep_count = min(
        injection.NATIVE_ELITES,
        int(np.ceil(0.3 * injection.NATIVE_ELITES)),
    )
    expected_first_mask = np.zeros(injection.NATIVE_CANDIDATES, dtype=bool)
    expected_first_mask[-injection.INJECTION_COUNT :] = True
    for seed_axis in range(len(GENERATOR_SEEDS)):
        for episode in range(len(EVAL_STARTS)):
            raw = arrays["fresh_raw_states"][seed_axis, episode]
            sequences = arrays["fresh_sequences"][seed_axis, episode]
            learned = arrays["fresh_generator_scores"][seed_axis, episode]
            exact = arrays["fresh_exact_scores"][seed_axis, episode]
            masks = arrays["fresh_injected"][seed_axis, episode]
            if not np.array_equal(raw, np.asarray(EVAL_STARTS[episode], dtype=np.float64)):
                raise ValueError("SS-001 fresh raw state differs from the frozen start")
            if np.max(np.abs(sequences)) > 1.0:
                raise ValueError("SS-001 fresh candidate action exceeds frozen bounds")
            if not np.array_equal(masks[0], expected_first_mask):
                raise ValueError("SS-001 fresh round-0 injection mask drifted")
            expected_exact = exact_discounted_scores(
                raw,
                sequences.reshape(-1, injection.HORIZON, 2),
            ).reshape(injection.NATIVE_ITERATIONS, injection.NATIVE_CANDIDATES)
            if not np.array_equal(expected_exact, exact):
                raise ValueError("SS-001 fresh exact scores do not recompute")
            for iteration in range(1, injection.NATIVE_ITERATIONS):
                elite_indices = np.argsort(learned[iteration - 1])[-injection.NATIVE_ELITES :][::-1]
                carried = elite_indices[:keep_count]
                if not np.array_equal(
                    sequences[iteration, -keep_count:],
                    sequences[iteration - 1, carried],
                ):
                    raise ValueError("SS-001 fresh within-call candidate carry drifted")
                expected_mask = np.zeros(injection.NATIVE_CANDIDATES, dtype=bool)
                expected_mask[-keep_count:] = masks[iteration - 1, carried]
                if not np.array_equal(masks[iteration], expected_mask):
                    raise ValueError("SS-001 fresh injection lineage drifted")
            if len(_first_unique_indices(sequences.reshape(-1, injection.HORIZON, 2))) != (EXPECTED_UNIQUE_CANDIDATES):
                raise ValueError("SS-001 fresh union cardinality drifted")
            _verify_duplicate_scores(
                sequences,
                np.concatenate(
                    (
                        learned.reshape(1, -1),
                        exact.reshape(1, -1),
                        arrays["fresh_auditor_scores"][:, seed_axis, episode].reshape(
                            len(AUDITOR_SEEDS),
                            -1,
                        ),
                    ),
                    axis=0,
                ),
            )

    parent_step0 = _parent_step0_arrays(parent_arrays)
    for seed_axis in range(len(PARENT_CALIBRATION_SEEDS)):
        for episode in range(len(EVAL_STARTS)):
            sequences = parent_step0["sequences"][seed_axis, episode]
            if not np.array_equal(
                parent_step0["raw_states"][seed_axis, episode],
                np.asarray(EVAL_STARTS[episode], dtype=np.float64),
            ):
                raise ValueError("SS-001 parent calibration raw state drifted")
            if len(_first_unique_indices(sequences.reshape(-1, injection.HORIZON, 2))) != (EXPECTED_UNIQUE_CANDIDATES):
                raise ValueError("SS-001 parent calibration union cardinality drifted")
            _verify_duplicate_scores(
                sequences,
                arrays["parent_auditor_scores"][:, seed_axis, episode],
            )


def _verify_execution_records(execution: dict[str, Any]) -> None:
    generator_records = cast(list[dict[str, Any]], execution.get("generator_records"))
    auditor_records = cast(list[dict[str, Any]], execution.get("auditor_records"))
    if [int(row["seed"]) for row in generator_records] != list(GENERATOR_SEEDS):
        raise ValueError("SS-001 generator execution records are incomplete")
    if [int(row["seed"]) for row in auditor_records] != list(AUDITOR_SEEDS):
        raise ValueError("SS-001 auditor execution records are incomplete")
    for record in generator_records:
        if set(record) != {
            "seed",
            "model_sha256",
            "ordinary_instrumented_difference",
            "ordinary_row_sha256",
            "instrumented_row_sha256",
            "provider_summary_sha256",
            "provider_summaries_canonical_equal",
        }:
            raise ValueError("SS-001 generator execution record schema drifted")
        if record["ordinary_row_sha256"] != record["instrumented_row_sha256"]:
            raise ValueError("SS-001 generator ordinary/instrumented row hashes differ")
        if not bool(record["provider_summaries_canonical_equal"]):
            raise ValueError("SS-001 generator provider parity flag failed")
        difference = cast(dict[str, Any], record["ordinary_instrumented_difference"])
        if difference != {
            "max_action_abs": 0.0,
            "max_final_state_abs": 0.0,
            "max_return_abs": 0.0,
            "successes_equal": True,
        }:
            raise ValueError("SS-001 generator trajectory parity is not exact")
    for record in (*generator_records, *auditor_records):
        fingerprint = str(record["model_sha256"])
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError("SS-001 model fingerprint is not a SHA-256 hex digest")


def _evaluation_accounting() -> dict[str, object]:
    calls_per_evaluation = len(EVAL_STARTS) * injection.EVAL_STEPS
    planner_sequences = calls_per_evaluation * injection.NATIVE_ITERATIONS * injection.NATIVE_CANDIDATES
    retained_sequences_per_generator = len(EVAL_STARTS) * injection.NATIVE_ITERATIONS * injection.NATIVE_CANDIDATES
    auditor_sequences = (
        len(AUDITOR_SEEDS) * (len(GENERATOR_SEEDS) + len(PARENT_CALIBRATION_SEEDS)) * retained_sequences_per_generator
    )
    return {
        "trained_models": {
            "fresh_generators": len(GENERATOR_SEEDS),
            "auditors": len(AUDITOR_SEEDS),
            "total": len(GENERATOR_SEEDS) + len(AUDITOR_SEEDS),
            "optimizer_updates": (len(GENERATOR_SEEDS) + len(AUDITOR_SEEDS)) * 1_800,
        },
        "fresh_generation": {
            "ordinary_and_instrumented_learned_sequences": (len(GENERATOR_SEEDS) * 2 * planner_sequences),
            "retained_step0_sequences": len(GENERATOR_SEEDS) * retained_sequences_per_generator,
        },
        "auditor_scoring": {
            "learned_sequences": auditor_sequences,
            "learned_transitions": auditor_sequences * injection.HORIZON,
            "fresh_and_parent_calls": (
                len(AUDITOR_SEEDS) * (len(GENERATOR_SEEDS) + len(PARENT_CALIBRATION_SEEDS)) * len(EVAL_STARTS)
            ),
        },
        "fast_verifier_fresh_exact": {
            "sequences": len(GENERATOR_SEEDS) * retained_sequences_per_generator,
            "transitions": (len(GENERATOR_SEEDS) * retained_sequences_per_generator * injection.HORIZON),
        },
    }


def _csv_text(rows: list[dict[str, object]]) -> str:
    fields = (
        "source_kind",
        "seed",
        "episode_index",
        "harmful_signature",
        "exact_improving_control",
        "b0_injected",
        "c_iteration",
        "generator_learned_gain",
        "c_exact_delta_from_b0",
        "c_exact_rank_damage_from_b0",
        "auditor_reject_votes",
        "auditor_transfer_votes",
        "auditor_tie_votes",
        "restart_rejected",
        "shared_transfer",
        "x_exact_delta_from_c",
        "x_exact_rank_gain_from_c",
        "exact_rescue",
        "x_materially_degrades_c",
        "calibration_call_pass",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue()


def _report_text(results: dict[str, Any]) -> str:
    decision = cast(dict[str, Any], results["decision"])
    fresh = cast(list[dict[str, Any]], results["fresh_call_rows"])
    calibration = cast(list[dict[str, Any]], results["parent_calibration_call_rows"])
    harmful = [row for row in fresh if row["harmful_signature"]]
    positive = [row for row in calibration if row["exact_improving_control"]]
    lines = [
        "# SS-001 cross-seed scorer-swap report",
        "",
        f"**Classification:** `{decision['classification']}`",
        "",
        "SS-001 used fresh generator seeds 32..43 as its primary units and a disjoint",
        "all-to-all auditor panel at seeds 20..31. CL-001 seeds 8..19 were visible",
        "calibration/transport inputs only and were excluded from primary counts.",
        "",
        "## Frozen gates",
        "",
        f"- Harm recurrence: {decision['recurrence_gate']['supporting_seeds']}/12 generator seeds "
        f"(required {decision['recurrence_gate']['required_seeds']}).",
        f"- Calibration direction: {decision['calibration_gate']['supporting_seeds']}/6 visible control seeds "
        f"(required {decision['calibration_gate']['required_seeds']}).",
        f"- Restart rejection: {decision['restart_rejection']['supporting_seeds']}/12 generator seeds.",
        f"- Shared transfer: {decision['shared_transfer']['supporting_seeds']}/12 generator seeds.",
        f"- Exact rank rescue: {decision['exact_rescue']['supporting_seeds']}/12 generator seeds.",
        "",
        "## Call-level audit",
        "",
        f"- Fresh inherited harmful calls: {len(harmful)}/48.",
        f"- Harmful calls rejected by at least 9/12 auditors: "
        f"{sum(bool(row['restart_rejected']) for row in harmful)}/{len(harmful)}.",
        f"- Harmful calls transferred by at least 9/12 auditors: "
        f"{sum(bool(row['shared_transfer']) for row in harmful)}/{len(harmful)}.",
        f"- Harmful calls exactly rescued by the rank selector: "
        f"{sum(bool(row['exact_rescue']) for row in harmful)}/{len(harmful)}.",
        f"- Visible exact-improving calibration calls: {len(positive)} (frozen expectation: 11).",
        f"- Passing calibration calls: {sum(bool(row['calibration_call_pass']) for row in positive)}/{len(positive)}.",
        "",
        "## Interpretation boundary",
        "",
        "This is cross-seed scorer swapping, not data cross-fitting. All models used",
        "the same dataset and architecture. Auditor agreement can distinguish restart-",
        "specific from restart-stable ranking behavior on these candidate pools, but it",
        "cannot localize shared bias or establish new-data/environment generalization.",
        "The exact selector result is offline evidence, not episode-return or production",
        "control evidence. Thresholds are descriptive robustness rules, not p-values.",
        "",
        "## Integrity",
        "",
        "The complete CL-001 package, protocol/source hashes, fresh raw pools, raw auditor",
        "scores, model fingerprints, trajectory parity, deterministic derivations, and",
        "rendered artifacts are bound into the SS-001 package. Use `verify-semantic` to",
        "retrain all 24 formal models and regenerate every raw tensor in memory.",
        "",
    ]
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
    """Execute all frozen fresh generators and auditors exactly once."""

    _assert_safe_output(output)
    verification = verify(output)
    if verification["outcomes"] != "prepared_only":
        raise FileExistsError("SS-001 formal execution is one-shot")
    formal_start = _mark_formal_started(output)
    arrays, execution = _execute(output)
    parent_arrays = parent._load_arrays(output / PARENT_TENSOR_COPY)
    _validate_array_shapes(arrays)
    _verify_fresh_lineage(arrays, parent_arrays)
    _verify_execution_records(cast(dict[str, Any], execution))
    derived = _derive(arrays, parent_arrays)
    _write_arrays(output / TENSOR_FILE, arrays)
    protocol = _read_json(output / "protocol.json")
    input_manifest = _read_json(output / "input-manifest.json")
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_cross_seed_scorer_swap",
        "interpretation_scope": (
            "prospective same-data generator-restart mechanism diagnostic; parent controls are non-independent"
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
        *cast(list[dict[str, object]], results["fresh_call_rows"]),
        *cast(list[dict[str, object]], results["parent_calibration_call_rows"]),
    ]
    (output / CSV_FILE).write_text(_csv_text(all_rows), encoding="utf-8")
    (output / REPORT_FILE).write_text(_report_text(results), encoding="utf-8")
    _write_artifact_manifest(output)
    verify(output, require_results=True)
    return results


def _verify_outcomes(
    output: Path,
    protocol: dict[str, Any],
    input_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    results = _read_json(output / RESULT_FILE)
    if results.get("schema_version") != SCHEMA_VERSION or results.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("SS-001 result identity does not match")
    if results.get("status") != "completed_cross_seed_scorer_swap":
        raise ValueError("SS-001 result status does not match")
    if results.get("interpretation_scope") != (
        "prospective same-data generator-restart mechanism diagnostic; parent controls are non-independent"
    ):
        raise ValueError("SS-001 interpretation scope drifted")
    protocol_sha = _canonical_json_sha256(protocol)
    if results.get("protocol") != protocol or results.get("protocol_sha256") != protocol_sha:
        raise ValueError("SS-001 results are not bound to the frozen protocol")
    if results.get("input_manifest") != input_manifest or results.get("input_manifest_sha256") != (
        _file_hash(output / "input-manifest.json")
    ):
        raise ValueError("SS-001 results are not bound to the input manifest")
    formal_start = _read_json(output / STARTED_FILE)
    if (
        formal_start != _formal_start_record(output)
        or results.get("formal_start") != formal_start
        or results.get("formal_start_sha256") != _file_hash(output / STARTED_FILE)
    ):
        raise ValueError("SS-001 results are not bound to the formal-start record")
    repository = cast(dict[str, Any], results["repository"])
    if repository.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("SS-001 result source hashes do not match")

    arrays = _load_arrays(output / TENSOR_FILE)
    _validate_array_shapes(arrays)
    tensor_record = cast(dict[str, Any], results["tensor_package"])
    expected_tensor = {
        "path": str(TENSOR_FILE),
        "file_sha256": _file_hash(output / TENSOR_FILE),
        **_arrays_metadata(arrays),
    }
    if tensor_record != expected_tensor:
        raise ValueError("SS-001 tensor package metadata does not match")
    parent_arrays = parent._load_arrays(output / PARENT_TENSOR_COPY)
    _verify_fresh_lineage(arrays, parent_arrays)
    execution = cast(dict[str, Any], results["execution"])
    _verify_execution_records(execution)
    derived = _derive(arrays, parent_arrays)
    for field in (
        "fresh_call_rows",
        "parent_calibration_call_rows",
        "fresh_seed_summaries",
        "calibration_seed_summaries",
        "decision",
    ):
        if _canonical_json_value(results[field]) != _canonical_json_value(derived[field]):
            raise ValueError(f"SS-001 saved {field} does not recompute")
    if results.get("evaluation_accounting") != _evaluation_accounting():
        raise ValueError("SS-001 evaluation accounting does not recompute")

    all_rows = [
        *cast(list[dict[str, object]], results["fresh_call_rows"]),
        *cast(list[dict[str, object]], results["parent_calibration_call_rows"]),
    ]
    if (output / CSV_FILE).read_text(encoding="utf-8") != _csv_text(all_rows):
        raise ValueError("SS-001 CSV is not canonical")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != _report_text(results):
        raise ValueError("SS-001 report is not canonical")
    artifact_manifest = _read_json(output / "artifact-manifest.json")
    expected_artifacts = {str(path): _file_hash(output / path) for path in ARTIFACT_PATHS}
    if artifact_manifest != {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": expected_artifacts,
    }:
        raise ValueError("SS-001 artifact manifest does not match")
    return results, arrays


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
    semantic: bool = False,
) -> dict[str, object]:
    """Verify deterministic outputs, optionally retraining all formal models."""

    protocol = _read_json(output / "protocol.json")
    if protocol != protocol_record():
        raise ValueError("saved SS-001 protocol does not match current sources")
    input_manifest = _read_json(output / "input-manifest.json")
    if input_manifest != _expected_input_manifest(output):
        raise ValueError("saved SS-001 input manifest or CL-001 snapshot has drifted")
    result_path = output / RESULT_FILE
    started_path = output / STARTED_FILE
    if result_path.exists():
        if not all((output / path).exists() for path in OUTCOME_PATHS):
            raise ValueError("SS-001 outcome package is partial")
        if _read_json(started_path) != _formal_start_record(output):
            raise ValueError("SS-001 formal-start record does not match")
        saved, arrays = _verify_outcomes(output, protocol, input_manifest)
        outcomes = "verified_results"
        if semantic:
            regenerated_arrays, regenerated_execution = _execute(output)
            for name in ARRAY_NAMES:
                if not np.array_equal(arrays[name], regenerated_arrays[name]):
                    raise ValueError(f"SS-001 semantic regeneration differs in {name}")
            if _canonical_json_value(saved["execution"]) != _canonical_json_value(regenerated_execution):
                raise ValueError("SS-001 semantic regeneration differs in execution records")
            parent_arrays = parent._load_arrays(output / PARENT_TENSOR_COPY)
            regenerated = _derive(regenerated_arrays, parent_arrays)
            for field in (
                "fresh_call_rows",
                "parent_calibration_call_rows",
                "fresh_seed_summaries",
                "calibration_seed_summaries",
                "decision",
            ):
                if _canonical_json_value(saved[field]) != _canonical_json_value(regenerated[field]):
                    raise ValueError(f"SS-001 semantic regeneration differs in {field}")
            outcomes = "verified_semantic_results"
    elif started_path.exists():
        raise ValueError("SS-001 formal execution started but is incomplete; the identifier is terminal")
    elif any((output / path).exists() for path in OUTCOME_PATHS):
        raise ValueError("SS-001 has partial outcomes without a formal-start record")
    else:
        outcomes = "prepared_only"
    if require_results and outcomes == "prepared_only":
        raise ValueError("complete SS-001 results are required")
    return {
        "status": "verified",
        "outcomes": outcomes,
        "protocol_sha256": _canonical_json_sha256(protocol),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
    }


def development_audit(output: Path = REPO_ROOT / PARENT_OUTPUT) -> dict[str, object]:
    """Exercise the scorer path with excluded seed 97 and no persisted outcomes."""

    _parent_snapshot(output)
    dataset = load_dataset(output / parent.INPUT_COPY)
    arrays = parent._load_arrays(output / parent.TENSOR_FILE)
    step0 = _parent_step0_arrays(arrays)
    model = train_balanced_model(dataset, DEVELOPMENT_SEED)
    first = _score_blocks(model, step0["raw_states"][0, 0], step0["sequences"][0, 0])
    repeated = _score_blocks(model, step0["raw_states"][0, 0], step0["sequences"][0, 0])
    if not np.array_equal(first, repeated):
        raise ValueError("SS-001 excluded development scorer is not deterministic")
    _verify_duplicate_scores(step0["sequences"][0, 0], first)
    return {
        "seed": DEVELOPMENT_SEED,
        "excluded": True,
        "model_sha256": oracle._model_fingerprint(model),
        "scores_finite": bool(np.all(np.isfinite(first))),
        "repeat_exact": True,
        "unique_candidates": len(_first_unique_indices(step0["sequences"][0, 0].reshape(-1, injection.HORIZON, 2))),
    }


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify(output, require_results=True)
    return _read_json(output / RESULT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="SS-001 cross-seed scorer-swap experiment")
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
