"""CL-001: confirmatory learned-versus-exact iCEM landscape audit."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np

from bench.bridge_control.experiment import _train_model as train_balanced_model
from bench.bridge_control.fixture import EVAL_STARTS, load_dataset, semantic_hash
from bench.oracle_ladder import experiment as oracle
from bench.oracle_ladder.audit import (
    REFERENCE_ELITE_COUNT,
    REFERENCE_SEARCH,
    STANDARD_PROPOSAL_COUNT,
    average_tie_ranks,
    exact_discounted_scores,
    pearson_correlation,
    spearman_correlation,
)
from bench.proposal_injection import experiment as injection
from bench.proposal_injection.providers import (
    PROVIDER_SEED_SALT,
    ExactReferenceProvider,
    provider_summary,
)
from bench.proposal_injection_v3 import experiment as parent
from prospect.types import Action, LatentState
from prospect.world_model import FlatWorldModel

from .planner import CandidateLandscapePlanner, CandidatePoolAudit

SCHEMA_VERSION = "candidate-landscape-v1"
EXPERIMENT_ID = "CL-001"
REPLAY_SEEDS = tuple(range(8))
CONFIRMATORY_SEEDS = tuple(range(8, 20))
ALL_SEEDS = REPLAY_SEEDS + CONFIRMATORY_SEEDS
DEVELOPMENT_SEED = 97
AUDIT_STEPS = (0, 1, 2)
SCORE_ATOL = 1e-12
RANK_DAMAGE_FLOOR = 8.0
CONFIRMATORY_SUPPORT_FLOOR = 10
COMMON_NATIVE_COUNT = 40
COMMON_BANK_SEED_SALT = 0x434C0001

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("bench/candidate_landscape/results/CL-001")
PROTOCOL_DOC = Path("docs/research/2026-07-15-candidate-landscape-cl001-protocol.md")
PARENT_OUTPUT = Path("bench/proposal_injection_v3/results/PI-003")
PARENT_RESULTS = PARENT_OUTPUT / "PI-003-results.json"
PARENT_POSTHOC = Path("bench/proposal_injection_v3/results/PI-003-posthoc-step-audit.json")
PARENT_DATASET = PARENT_OUTPUT / injection.INPUT_COPY
INPUT_COPY = Path("inputs/BC-001-b1_r1_d8.npz")
TENSOR_FILE = Path(f"{EXPERIMENT_ID}-candidates.npz")
RESULT_FILE = Path(f"{EXPERIMENT_ID}-results.json")
CSV_FILE = Path(f"{EXPERIMENT_ID}-calls.csv")
REPORT_FILE = Path(f"{EXPERIMENT_ID}-report.md")
STARTED_FILE = Path("formal-start.json")

PROTECTED_OUTPUTS = (
    PARENT_OUTPUT,
    Path("bench/proposal_injection/results/PI-001"),
    Path("bench/proposal_injection_v2/results/PI-002"),
    Path("bench/oracle_ladder_v2/results/OL-002"),
    Path("bench/bridge_control/results/BC-001"),
)

PARENT_HASHES = {
    "PI-003-results.json": "bea3a1ad850099b97628e313d1b1a2a889d54912aee7ada0bfab82343f743251",
    "PI-003-runs.csv": "bb003b16ce91b918561381982ebd4e0c6eedae5c05ef3de59318a7bf8fd63c5e",
    "PI-003-report.md": "c5a3b8aaa6e8fe4a02ca0cef79075fa8c5b3358571e8bb30273b733d393af08b",
    "protocol.json": "5630c0578f8c871d8a99b5d472c10665702fdf18f5cb6b03e7ce64b15263538d",
    "input-manifest.json": "bbbc94d96690f240dd3a6b45377d968b0e5761739d09fd6ef5427fea293c0e0c",
    "artifact-manifest.json": "8184327d9bfe737137c221ac62546213f344a1e46fb0ef4a01a528bc2b198ebb",
    str(injection.INPUT_COPY): "9182143e6aee081da68c1fb9d521fc87c3fad90e0bb0d8adbda095db09b22948",
}
PARENT_POSTHOC_SHA256 = "3ea84b6ac74725231c7733870d6a6de21991eae27822db7ef0f9d3df6f987b4f"

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *parent.SOURCE_FILES,
            Path("bench/candidate_landscape/__init__.py"),
            Path("bench/candidate_landscape/__main__.py"),
            Path("bench/candidate_landscape/planner.py"),
            Path("bench/candidate_landscape/experiment.py"),
            Path("tests/test_candidate_landscape_planner.py"),
            Path("tests/test_candidate_landscape_experiment.py"),
            PROTOCOL_DOC,
        )
    )
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
    INPUT_COPY,
    STARTED_FILE,
    TENSOR_FILE,
    RESULT_FILE,
    CSV_FILE,
    REPORT_FILE,
)

ARRAY_NAMES = (
    "seeds",
    "sequences",
    "learned_scores",
    "exact_scores",
    "injected",
    "raw_states",
    "latent_states",
    "latent_ood",
    "latent_ood_present",
    "cold_sequences",
    "cold_learned_scores",
    "cold_exact_scores",
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
    return {
        "dtype": np.asarray(array).dtype.str,
        "shape": list(np.asarray(array).shape),
    }


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
    digest.update(b"CL-001 canonical tensor package\0")
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
            raise ValueError("CL-001 tensor package has unexpected array names")
        return {name: np.asarray(package[name]).copy() for name in ARRAY_NAMES}


def _parent_snapshot() -> dict[str, object]:
    verification = parent.verify(REPO_ROOT / PARENT_OUTPUT, require_results=True)
    if verification["outcomes"] != "verified_results":
        raise ValueError("CL-001 requires a verified PI-003 parent")
    for name, expected in PARENT_HASHES.items():
        if _file_hash(REPO_ROOT / PARENT_OUTPUT / name) != expected:
            raise ValueError(f"PI-003 parent hash drifted for {name}")
    if _file_hash(REPO_ROOT / PARENT_POSTHOC) != PARENT_POSTHOC_SHA256:
        raise ValueError("PI-003 post-hoc audit hash drifted")
    results = _read_json(REPO_ROOT / PARENT_RESULTS)
    decision = cast(dict[str, Any], results["decision"])
    if decision["classification"] != "no_privileged_rescue:trigger_not_statewise":
        raise ValueError("PI-003 parent decision is not the required trigger")
    return {
        "experiment_id": "PI-003",
        "verification": verification["outcomes"],
        "hashes": dict(PARENT_HASHES),
        "posthoc_sha256": PARENT_POSTHOC_SHA256,
        "classification": decision["classification"],
        "scientific_relationship": (
            "seeds 0..7 are replay-only hypothesis-generating evidence; CL-001 confirmation uses untouched seeds 8..19"
        ),
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
        "replay_seeds": list(REPLAY_SEEDS),
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seed_excluded": DEVELOPMENT_SEED,
        "experimental_unit": "model seed; four starts are repeated measures",
        "training": {"steps": 1_800, "batch_size": 64, "dataset": "BC-001/b1_r1_d8"},
        "evaluation": {
            "starts": EVAL_STARTS.tolist(),
            "episode_steps": injection.EVAL_STEPS,
            "retained_steps": list(AUDIT_STEPS),
        },
        "planner": {
            "arm": "privileged_injection",
            "horizon": injection.HORIZON,
            "candidates": injection.NATIVE_CANDIDATES,
            "elites": injection.NATIVE_ELITES,
            "iterations": injection.NATIVE_ITERATIONS,
            "injection_count": injection.INJECTION_COUNT,
            "uncertainty_penalty": 0.0,
        },
        "cold_bank": {
            "native_sequences": COMMON_NATIVE_COUNT,
            "reference_sequences_per_retained_state": injection.INJECTION_COUNT,
            "reference_steps": list(AUDIT_STEPS),
            "identical_sequences_reused_across_steps": True,
            "seed_salt": COMMON_BANK_SEED_SALT,
        },
        "co_primary": {
            "within_call_exploitation": {
                "step": 0,
                "round0_best_must_be_injected": True,
                "selected_iteration_floor": 1,
                "learned_gain_floor": SCORE_ATOL,
                "exact_delta_ceiling": -SCORE_ATOL,
                "exact_rank_damage_floor": RANK_DAMAGE_FLOOR,
                "supporting_start_floor": 3,
            },
            "visited_state_scorer_shift": {
                "steps": [0, 2],
                "reference_exact_rank_ceiling": injection.NATIVE_ELITES,
                "rank_residual_deterioration_floor": RANK_DAMAGE_FLOOR,
                "supporting_start_floor": 3,
            },
        },
        "decision": {
            "support_floor": CONFIRMATORY_SUPPORT_FLOOR,
            "confirmatory_seed_count": len(CONFIRMATORY_SEEDS),
            "statistical_role": ("descriptive model-seed robustness; not a binomial p-value"),
            "classifications": [
                "within_call_exploitation_and_statewise_scorer_shift",
                "within_call_exploitation_only",
                "statewise_scorer_shift_only",
                "neither_mechanism_supported",
            ],
        },
        "stop_rules": [
            "invalid on parent, replay, fingerprint, trajectory, provider, or capture parity failure",
            "invalid on non-finite, shape, score-recomputation, provenance, selection, or artifact failure",
            "execute all replay and confirmatory seeds without outcome-dependent stopping",
            "do not change seeds, endpoints, thresholds, branches, or retries after formal execution begins",
            "do not modify production, tasks, ADRs, gates, or sealed parent packages",
        ],
    }


def _expected_input_manifest(output: Path) -> dict[str, object]:
    copied = output / INPUT_COPY
    dataset = load_dataset(copied)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256(_canonical_json_bytes(protocol_record())).hexdigest(),
        "parent": _parent_snapshot(),
        "copied_dataset": {
            "path": str(INPUT_COPY),
            "file_sha256": _file_hash(copied),
            "semantic_sha256": semantic_hash(dataset),
            "name": dataset.name,
        },
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_safe_output(output: Path) -> None:
    resolved = output.resolve()
    repo_resolved = REPO_ROOT.resolve()
    if resolved == repo_resolved or resolved in repo_resolved.parents:
        raise ValueError("CL-001 output cannot be the repository root or an ancestor")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("CL-001 output must be a real directory path")
    for protected in PROTECTED_OUTPUTS:
        protected_resolved = (REPO_ROOT / protected).resolve()
        if _paths_overlap(resolved, protected_resolved):
            raise ValueError(f"CL-001 output overlaps protected evidence package {protected}")
    owned_results = (REPO_ROOT / "bench/candidate_landscape/results").resolve()
    if repo_resolved in resolved.parents and owned_results not in resolved.parents:
        raise ValueError("in-repository CL-001 output must be below bench/candidate_landscape/results")
    if (output / RESULT_FILE).exists():
        raise FileExistsError("CL-001 is already formal; preserve it and use a new id")


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Freeze CL-001 sources, inputs, protocol, and parent hashes without outcomes."""

    _parent_snapshot()
    _assert_safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("CL-001 output must be absent or an empty directory")
    (output / INPUT_COPY.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / PARENT_DATASET, output / INPUT_COPY)
    _write_json(output / "protocol.json", protocol_record())
    _write_json(output / "input-manifest.json", _expected_input_manifest(output))
    result = verify(output)
    return {**result, "status": "prepared_only"}


def _formal_start_record(output: Path) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "formal_execution_started",
        "protocol_sha256": sha256(_canonical_json_bytes(_read_json(output / "protocol.json"))).hexdigest(),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
    }


def _mark_formal_started(output: Path) -> dict[str, object]:
    """Durably consume the CL-001 identifier before any formal model is trained."""

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


def _parent_privileged_rows() -> dict[int, dict[str, Any]]:
    results = _read_json(REPO_ROOT / PARENT_RESULTS)
    rows = {
        int(row["seed"]): row
        for row in cast(list[dict[str, Any]], results["rows"])
        if row["arm"] == "privileged_injection"
    }
    if set(rows) != set(REPLAY_SEEDS):
        raise ValueError("PI-003 privileged replay rows are incomplete")
    return rows


def _audited_planner(
    model: FlatWorldModel,
    seed: int,
    provider: ExactReferenceProvider,
) -> CandidateLandscapePlanner:
    return CandidateLandscapePlanner(
        model,
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        horizon=injection.HORIZON,
        candidates=injection.NATIVE_CANDIDATES,
        elites=injection.NATIVE_ELITES,
        iterations=injection.NATIVE_ITERATIONS,
        uncertainty_penalty=0.0,
        seed=seed,
        injection_count=injection.INJECTION_COUNT,
        sequence_provider=provider,
        episode_steps=injection.EVAL_STEPS,
        audit_steps=AUDIT_STEPS,
    )


def _common_native_sequences(seed: int, episode_index: int) -> np.ndarray:
    sequence = np.random.SeedSequence([COMMON_BANK_SEED_SALT, seed, episode_index])
    rng = np.random.default_rng(sequence)
    white = rng.normal(size=(COMMON_NATIVE_COUNT, injection.HORIZON, 2))
    spectrum = np.fft.rfft(white, axis=1)
    frequencies = np.fft.rfftfreq(injection.HORIZON)
    frequencies[0] = frequencies[1]
    scale = frequencies ** (-1.0)
    kernel = np.fft.irfft(scale, n=injection.HORIZON)
    normalizer = float(np.sqrt(np.sum(kernel**2)))
    spectrum *= scale[None, :, None]
    colored = np.fft.irfft(spectrum, n=injection.HORIZON, axis=1) / normalizer
    # FlatPlanner's initial standard deviation is 0.5 on [-1, 1].
    return np.asarray(np.clip(0.5 * colored, -1.0, 1.0), dtype=np.float64)


def _empty_arrays() -> dict[str, np.ndarray]:
    seed_count = len(ALL_SEEDS)
    prefix = (seed_count, len(EVAL_STARTS), len(AUDIT_STEPS))
    pools = prefix + (
        injection.NATIVE_ITERATIONS,
        injection.NATIVE_CANDIDATES,
    )
    return {
        "seeds": np.asarray(ALL_SEEDS, dtype=np.int64),
        "sequences": np.empty(pools + (injection.HORIZON, 2), dtype=np.float64),
        "learned_scores": np.empty(pools, dtype=np.float64),
        "exact_scores": np.empty(pools, dtype=np.float64),
        "injected": np.empty(pools, dtype=bool),
        "raw_states": np.empty(prefix + (3,), dtype=np.float64),
        "latent_states": np.empty(prefix + (6,), dtype=np.float64),
        "latent_ood": np.empty(prefix, dtype=np.float64),
        "latent_ood_present": np.empty(prefix, dtype=bool),
        "cold_sequences": np.empty(
            prefix + (injection.NATIVE_CANDIDATES, injection.HORIZON, 2),
            dtype=np.float64,
        ),
        "cold_learned_scores": np.empty(prefix + (injection.NATIVE_CANDIDATES,), dtype=np.float64),
        "cold_exact_scores": np.empty(prefix + (injection.NATIVE_CANDIDATES,), dtype=np.float64),
    }


def _pool_groups(
    pools: Iterable[CandidatePoolAudit],
) -> dict[tuple[int, int], list[CandidatePoolAudit]]:
    grouped: dict[tuple[int, int], list[CandidatePoolAudit]] = {}
    for pool in pools:
        grouped.setdefault((pool.episode_index, pool.step), []).append(pool)
    for records in grouped.values():
        records.sort(key=lambda item: item.iteration)
    return grouped


def _validate_pool_lineage(records: list[CandidatePoolAudit]) -> None:
    if [record.iteration for record in records] != list(range(injection.NATIVE_ITERATIONS)):
        raise ValueError("CL-001 call does not contain exactly three ordered pools")
    if not all(np.array_equal(record.raw_state, records[0].raw_state) for record in records):
        raise ValueError("CL-001 raw state changed within one planning call")
    if not all(np.array_equal(record.latent_state, records[0].latent_state) for record in records):
        raise ValueError("CL-001 latent state changed within one planning call")
    keep_count = min(
        injection.NATIVE_ELITES,
        int(np.ceil(0.3 * injection.NATIVE_ELITES)),
    )
    for previous, current in zip(records[:-1], records[1:], strict=True):
        elite_indices = np.argsort(previous.learned_scores)[-injection.NATIVE_ELITES :][::-1]
        expected_sequences = previous.sequences[elite_indices[:keep_count]]
        expected_injected = previous.injected[elite_indices[:keep_count]]
        if not np.array_equal(current.sequences[-keep_count:], expected_sequences):
            raise ValueError("CL-001 carried candidate tail does not match learned elites")
        if not np.array_equal(current.injected[-keep_count:], expected_injected):
            raise ValueError("CL-001 carried injection provenance does not match")
        if np.any(current.injected[:-keep_count]):
            raise ValueError("CL-001 later fresh rows cannot be current-call injected")


def _store_seed_arrays(
    arrays: dict[str, np.ndarray],
    seed_axis: int,
    seed: int,
    model: FlatWorldModel,
    planner: CandidateLandscapePlanner,
    provider: ExactReferenceProvider,
    evaluation: injection.Evaluation,
) -> None:
    pools = planner.pool_audits
    if len(pools) != len(EVAL_STARTS) * len(AUDIT_STEPS) * injection.NATIVE_ITERATIONS:
        raise ValueError("CL-001 captured the wrong number of candidate pools")
    grouped = _pool_groups(pools)
    expected_keys = {(episode, step) for episode in range(len(EVAL_STARTS)) for step in AUDIT_STEPS}
    if set(grouped) != expected_keys:
        raise ValueError("CL-001 captured the wrong episode/step calls")

    for episode in range(len(EVAL_STARTS)):
        common_native = _common_native_sequences(seed, episode)
        references_by_step: list[np.ndarray] = []
        first_records: list[CandidatePoolAudit] = []
        for step_axis, step in enumerate(AUDIT_STEPS):
            records = grouped[(episode, step)]
            _validate_pool_lineage(records)
            first = records[0]
            first_records.append(first)
            expected_call = episode * injection.EVAL_STEPS + step
            if first.call_index != expected_call:
                raise ValueError("CL-001 call index is not provider-aligned")
            expected_mask = np.zeros(injection.NATIVE_CANDIDATES, dtype=bool)
            carried = 0 if step == 0 else 3
            fresh = injection.NATIVE_CANDIDATES - carried
            expected_mask[fresh - injection.INJECTION_COUNT : fresh] = True
            if not np.array_equal(first.injected, expected_mask):
                raise ValueError("CL-001 first-round injection positions differ from PI-003")

            provider_call = provider.calls[expected_call]
            if provider_call.raw_sha256 != ExactReferenceProvider._raw_hash(first.raw_state):
                raise ValueError("CL-001 captured raw state differs from provider input")
            if not np.array_equal(
                first.exact_scores[first.injected],
                np.asarray(provider_call.output_exact_scores, dtype=np.float64),
            ):
                raise ValueError("CL-001 injected exact scores differ from provider records")

            for record in records:
                index = (seed_axis, episode, step_axis, record.iteration)
                arrays["sequences"][index] = record.sequences
                arrays["learned_scores"][index] = record.learned_scores
                arrays["exact_scores"][index] = record.exact_scores
                arrays["injected"][index] = record.injected
            call_index = (seed_axis, episode, step_axis)
            arrays["raw_states"][call_index] = first.raw_state
            arrays["latent_states"][call_index] = first.latent_state
            arrays["latent_ood_present"][call_index] = first.latent_ood is not None
            arrays["latent_ood"][call_index] = float(first.latent_ood) if first.latent_ood is not None else 0.0

            selected_flat = int(np.argmax(np.stack([record.learned_scores for record in records]).reshape(-1)))
            selected_iteration, selected_candidate = divmod(selected_flat, injection.NATIVE_CANDIDATES)
            selected_action = records[selected_iteration].sequences[selected_candidate, 0]
            saved_action = np.asarray(evaluation.action_traces[episode][step], dtype=np.float64)
            if not np.array_equal(selected_action, saved_action):
                raise ValueError("CL-001 selected action does not reconstruct from pools")
            references_by_step.append(first.sequences[first.injected].copy())

        cold = np.concatenate([common_native, *references_by_step], axis=0)
        if cold.shape != (
            injection.NATIVE_CANDIDATES,
            injection.HORIZON,
            2,
        ):
            raise ValueError("CL-001 common cold bank has the wrong shape")
        for step_axis, first in enumerate(first_records):
            call_index = (seed_axis, episode, step_axis)
            latent = LatentState(
                z=first.latent_state.copy(),
                ood=first.latent_ood,
            )
            learned = planner._imagined_returns(latent, cold)
            exact = exact_discounted_scores(first.raw_state, cold)
            arrays["cold_sequences"][call_index] = cold
            arrays["cold_learned_scores"][call_index] = learned
            arrays["cold_exact_scores"][call_index] = exact


def _evaluation_row(
    seed: int,
    fingerprint: str,
    evaluation: injection.Evaluation,
    provider: ExactReferenceProvider,
) -> dict[str, object]:
    return injection._row(
        "privileged_injection",
        seed,
        fingerprint,
        evaluation,
        provider=provider,
    )


def _execute(dataset_path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    dataset = load_dataset(dataset_path)
    if dataset.name != "b1_r1_d8":
        raise ValueError("CL-001 requires the frozen BC-001 b1_r1_d8 dataset")
    parent_rows = _parent_privileged_rows()
    arrays = _empty_arrays()
    trajectories: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []

    for seed_axis, seed in enumerate(ALL_SEEDS):
        model = train_balanced_model(dataset, seed)
        fingerprint = oracle._model_fingerprint(model)

        ordinary_row: dict[str, object] | None = None
        ordinary_provider: ExactReferenceProvider | None = None
        ordinary_evaluation: injection.Evaluation | None = None
        if seed in CONFIRMATORY_SEEDS:
            ordinary_provider = ExactReferenceProvider(seed, "privileged")
            ordinary_planner = injection._injection_planner(model, seed, ordinary_provider)
            ordinary_evaluation = injection._evaluate(
                ordinary_planner,
                injection._sidecar_encoder(model),
            )
            ordinary_row = _evaluation_row(seed, fingerprint, ordinary_evaluation, ordinary_provider)

        audited_provider = ExactReferenceProvider(seed, "privileged")
        audited_planner = _audited_planner(model, seed, audited_provider)
        audited_evaluation = injection._evaluate(
            audited_planner,
            injection._sidecar_encoder(model),
        )
        audited_row = _evaluation_row(seed, fingerprint, audited_evaluation, audited_provider)

        if seed in REPLAY_SEEDS:
            saved = parent_rows[seed]
            if fingerprint != saved["model_sha256"]:
                raise ValueError(f"seed {seed} model fingerprint differs from PI-003")
            if _canonical_json_value(audited_row) != _canonical_json_value(saved):
                raise ValueError(f"seed {seed} instrumented replay differs from PI-003")
            parity_record: dict[str, object] = {
                "seed": seed,
                "phase": "replay",
                "model_sha256": fingerprint,
                "parent_row_canonical_equal": True,
                "parent_row_sha256": _canonical_json_sha256(saved),
                "instrumented_row_sha256": _canonical_json_sha256(audited_row),
                "provider_summary_sha256": _canonical_json_sha256(provider_summary(audited_provider)),
            }
        else:
            assert ordinary_row is not None
            assert ordinary_provider is not None
            assert ordinary_evaluation is not None
            difference = injection._assert_parity(
                f"seed {seed} ordinary versus instrumented",
                ordinary_evaluation,
                audited_evaluation,
            )
            if _canonical_json_value(ordinary_row) != _canonical_json_value(audited_row):
                raise ValueError(f"seed {seed} instrumented row differs from ordinary planner")
            parity_record = {
                "seed": seed,
                "phase": "confirmatory",
                "model_sha256": fingerprint,
                "ordinary_instrumented_difference": difference,
                "provider_summaries_canonical_equal": True,
                "ordinary_row_sha256": _canonical_json_sha256(ordinary_row),
                "instrumented_row_sha256": _canonical_json_sha256(audited_row),
                "provider_summary_sha256": _canonical_json_sha256(provider_summary(audited_provider)),
            }

        _store_seed_arrays(
            arrays,
            seed_axis,
            seed,
            model,
            audited_planner,
            audited_provider,
            audited_evaluation,
        )
        trajectories.append(audited_row)
        parity.append(parity_record)

    derived = _derive(arrays, trajectories)
    return {
        "trajectories": trajectories,
        "parity": parity,
        **derived,
    }, arrays


def _first_unique_indices(sequences: np.ndarray) -> np.ndarray:
    seen: set[bytes] = set()
    indices: list[int] = []
    for index, sequence in enumerate(sequences):
        key = np.asarray(sequence, dtype="<f8", order="C").tobytes()
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _sequence_bytes(sequence: np.ndarray) -> bytes:
    value = np.asarray(sequence, dtype="<f8", order="C")
    if value.shape != (injection.HORIZON, 2):
        raise ValueError("CL-001 candidate sequence has the wrong shape")
    return value.tobytes(order="C")


def _sequence_sha256(sequence: np.ndarray) -> str:
    return sha256(_sequence_bytes(sequence)).hexdigest()


def _sequence_collection_sha256(sequences: np.ndarray) -> str:
    value = np.asarray(sequences, dtype=np.float64)
    digest = sha256()
    digest.update(b"CL-001 ordered sequence collection\0")
    digest.update(_canonical_json_bytes(_array_header(value)))
    digest.update(_array_bytes(value))
    return digest.hexdigest()


def _identity_trajectory(
    label: str,
    target: np.ndarray,
    sequences: np.ndarray,
    learned: np.ndarray,
    exact: np.ndarray,
    injected: np.ndarray,
) -> dict[str, object]:
    target_bytes = _sequence_bytes(target)
    rounds: list[dict[str, object]] = []
    for iteration in range(injection.NATIVE_ITERATIONS):
        matches = [
            index for index, sequence in enumerate(sequences[iteration]) if _sequence_bytes(sequence) == target_bytes
        ]
        if not matches:
            rounds.append({"iteration": iteration, "present": False})
            continue
        learned_values = learned[iteration, matches]
        exact_values = exact[iteration, matches]
        if not np.all(learned_values == learned_values[0]) or not np.all(exact_values == exact_values[0]):
            raise ValueError("duplicate CL-001 sequence has inconsistent scores")
        first = matches[0]
        learned_ranks = average_tie_ranks(learned[iteration])
        exact_ranks = average_tie_ranks(exact[iteration])
        elite_indices = set(np.argsort(learned[iteration])[-injection.NATIVE_ELITES :][::-1].tolist())
        rounds.append(
            {
                "iteration": iteration,
                "present": True,
                "candidate_indices": matches,
                "first_candidate_index": first,
                "learned_score": float(learned_values[0]),
                "exact_score": float(exact_values[0]),
                "learned_rank": float(learned_ranks[first]),
                "exact_rank": float(exact_ranks[first]),
                "injected": bool(injected[iteration, first]),
                "learned_elite": first in elite_indices,
            }
        )
    return {
        "label": label,
        "sequence_sha256": _sequence_sha256(target),
        "rounds": rounds,
    }


def _union_identity_values(
    flat_index: int,
    flat_sequences: np.ndarray,
    flat_learned: np.ndarray,
    flat_exact: np.ndarray,
    first_flat_by_bytes: dict[bytes, int],
    unique_position: dict[bytes, int],
    unique_ranks: np.ndarray,
) -> tuple[int, float, float, float]:
    key = _sequence_bytes(flat_sequences[flat_index])
    first_flat = first_flat_by_bytes[key]
    return (
        first_flat,
        float(flat_learned[first_flat]),
        float(flat_exact[first_flat]),
        float(unique_ranks[unique_position[key]]),
    )


def _top_k_overlap(left: np.ndarray, right: np.ndarray, k: int) -> int:
    left_top = set(np.argsort(left)[-k:].tolist())
    right_top = set(np.argsort(right)[-k:].tolist())
    return len(left_top & right_top)


def _within_call_support(
    *,
    round0_best_injected: bool,
    selected_first_iteration: int,
    learned_gain: float,
    exact_delta: float,
    exact_rank_damage: float,
) -> bool:
    return bool(
        round0_best_injected
        and selected_first_iteration > 0
        and learned_gain > SCORE_ATOL
        and exact_delta < -SCORE_ATOL
        and exact_rank_damage >= RANK_DAMAGE_FLOOR
    )


def _state_shift_support(
    *,
    step0_exact_rank: float,
    step2_exact_rank: float,
    rank_residual_deterioration: float,
) -> bool:
    return bool(
        step0_exact_rank <= injection.NATIVE_ELITES
        and step2_exact_rank <= injection.NATIVE_ELITES
        and rank_residual_deterioration >= RANK_DAMAGE_FLOOR
    )


def _pool_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_axis, seed in enumerate(arrays["seeds"].tolist()):
        phase = "replay" if int(seed) in REPLAY_SEEDS else "confirmatory"
        for episode in range(len(EVAL_STARTS)):
            for step_axis, step in enumerate(AUDIT_STEPS):
                for iteration in range(injection.NATIVE_ITERATIONS):
                    index = (seed_axis, episode, step_axis, iteration)
                    sequences = arrays["sequences"][index]
                    learned = arrays["learned_scores"][index]
                    exact = arrays["exact_scores"][index]
                    learned_best = int(np.argmax(learned))
                    exact_best = int(np.argmax(exact))
                    learned_elites = np.argsort(learned)[-injection.NATIVE_ELITES :][::-1]
                    exact_ranks = average_tie_ranks(exact)
                    rows.append(
                        {
                            "seed": int(seed),
                            "phase": phase,
                            "episode_index": episode,
                            "step": step,
                            "iteration": iteration,
                            "candidate_count": len(learned),
                            "injected_count": int(np.count_nonzero(arrays["injected"][index])),
                            "pool_sha256": _sequence_collection_sha256(sequences),
                            "candidate_sha256": [_sequence_sha256(sequence) for sequence in sequences],
                            "learned_elite_indices": learned_elites.tolist(),
                            "learned_best_candidate": learned_best,
                            "learned_best_sha256": _sequence_sha256(sequences[learned_best]),
                            "exact_best_candidate": exact_best,
                            "exact_best_sha256": _sequence_sha256(sequences[exact_best]),
                            "learned_exact_pearson": pearson_correlation(learned, exact),
                            "learned_exact_spearman": spearman_correlation(learned, exact),
                            "top_eight_overlap": _top_k_overlap(learned, exact, injection.NATIVE_ELITES),
                            "learned_best_exact_score": float(exact[learned_best]),
                            "learned_best_exact_rank": float(exact_ranks[learned_best]),
                            "learned_best_exact_regret": float(np.max(exact) - exact[learned_best]),
                        }
                    )
    return rows


def _call_rows(
    arrays: dict[str, np.ndarray],
    trajectories: list[dict[str, object]],
) -> list[dict[str, object]]:
    trajectory_by_seed = {int(cast(int, row["seed"])): row for row in trajectories}
    rows: list[dict[str, object]] = []
    for seed_axis, seed_value in enumerate(arrays["seeds"].tolist()):
        seed = int(seed_value)
        phase = "replay" if seed in REPLAY_SEEDS else "confirmatory"
        actions = cast(list[list[list[float]]], trajectory_by_seed[seed]["action_traces"])
        for episode in range(len(EVAL_STARTS)):
            for step_axis, step in enumerate(AUDIT_STEPS):
                prefix = (seed_axis, episode, step_axis)
                sequences = arrays["sequences"][prefix]
                learned = arrays["learned_scores"][prefix]
                exact = arrays["exact_scores"][prefix]
                injected = arrays["injected"][prefix]
                flat_sequences = sequences.reshape(-1, injection.HORIZON, 2)
                flat_learned = learned.reshape(-1)
                flat_exact = exact.reshape(-1)

                reference_candidates = np.flatnonzero(injected[0])
                if len(reference_candidates) != injection.INJECTION_COUNT:
                    raise ValueError("CL-001 round-0 reference count is not eight")
                reference_candidate = int(reference_candidates[np.argmax(exact[0, reference_candidates])])
                reference_flat = reference_candidate
                round0_best_candidate = int(np.argmax(learned[0]))
                round0_best_flat = round0_best_candidate
                selected_flat = int(np.argmax(flat_learned))
                selected_iteration, selected_candidate = divmod(selected_flat, injection.NATIVE_CANDIDATES)
                if not np.array_equal(
                    flat_sequences[selected_flat, 0],
                    np.asarray(actions[episode][step], dtype=np.float64),
                ):
                    raise ValueError("CL-001 derived selection differs from trajectory")

                unique = _first_unique_indices(flat_sequences)
                first_flat_by_bytes = {
                    _sequence_bytes(flat_sequences[int(flat_index)]): int(flat_index) for flat_index in unique
                }
                for first_flat in unique:
                    key = _sequence_bytes(flat_sequences[int(first_flat)])
                    matches = [
                        index for index, sequence in enumerate(flat_sequences) if _sequence_bytes(sequence) == key
                    ]
                    if not np.all(flat_learned[matches] == flat_learned[matches[0]]) or not np.all(
                        flat_exact[matches] == flat_exact[matches[0]]
                    ):
                        raise ValueError("duplicate CL-001 sequence has inconsistent union scores")
                unique_ranks = average_tie_ranks(flat_exact[unique])
                unique_position = {
                    _sequence_bytes(flat_sequences[int(value)]): index for index, value in enumerate(unique)
                }

                reference_first, reference_learned, reference_exact, reference_rank = _union_identity_values(
                    reference_flat,
                    flat_sequences,
                    flat_learned,
                    flat_exact,
                    first_flat_by_bytes,
                    unique_position,
                    unique_ranks,
                )
                round0_first, round0_learned, round0_exact, round0_rank = _union_identity_values(
                    round0_best_flat,
                    flat_sequences,
                    flat_learned,
                    flat_exact,
                    first_flat_by_bytes,
                    unique_position,
                    unique_ranks,
                )
                selected_first, selected_learned, selected_exact, selected_rank = _union_identity_values(
                    selected_flat,
                    flat_sequences,
                    flat_learned,
                    flat_exact,
                    first_flat_by_bytes,
                    unique_position,
                    unique_ranks,
                )
                selected_first_iteration = selected_first // injection.NATIVE_CANDIDATES
                if selected_first != selected_flat:
                    raise ValueError("CL-001 planner selection is not the first identical occurrence")
                refinement_learned_gain = selected_learned - round0_learned
                refinement_exact_delta = selected_exact - round0_exact
                refinement_exact_rank_damage = selected_rank - round0_rank
                round0_best_injected = bool(injected[0, round0_best_candidate])
                supports_within_call = _within_call_support(
                    round0_best_injected=round0_best_injected,
                    selected_first_iteration=selected_first_iteration,
                    learned_gain=refinement_learned_gain,
                    exact_delta=refinement_exact_delta,
                    exact_rank_damage=refinement_exact_rank_damage,
                )

                cold_learned = arrays["cold_learned_scores"][prefix]
                cold_exact = arrays["cold_exact_scores"][prefix]
                reference_group_start = COMMON_NATIVE_COUNT + step_axis * injection.INJECTION_COUNT
                cold_references = np.arange(
                    reference_group_start,
                    reference_group_start + injection.INJECTION_COUNT,
                    dtype=np.int64,
                )
                cold_reference = int(cold_references[np.argmax(cold_exact[cold_references])])
                cold_exact_ranks = average_tie_ranks(cold_exact)
                cold_learned_ranks = average_tie_ranks(cold_learned)
                cold_reference_exact_rank = float(cold_exact_ranks[cold_reference])
                cold_reference_learned_rank = float(cold_learned_ranks[cold_reference])
                cold_reference_rank_residual = cold_reference_learned_rank - cold_reference_exact_rank

                rows.append(
                    {
                        "seed": seed,
                        "phase": phase,
                        "episode_index": episode,
                        "step": step,
                        "reference_candidate": reference_candidate,
                        "reference_first_flat_index": reference_first,
                        "reference_sha256": _sequence_sha256(flat_sequences[reference_flat]),
                        "round0_best_candidate": round0_best_candidate,
                        "round0_best_first_flat_index": round0_first,
                        "round0_best_sha256": _sequence_sha256(flat_sequences[round0_best_flat]),
                        "round0_best_injected": round0_best_injected,
                        "selected_iteration": selected_iteration,
                        "selected_candidate": selected_candidate,
                        "selected_first_iteration": selected_first_iteration,
                        "selected_first_flat_index": selected_first,
                        "selected_sha256": _sequence_sha256(flat_sequences[selected_flat]),
                        "unique_candidate_count": len(unique),
                        "reference_learned_score": reference_learned,
                        "round0_best_learned_score": round0_learned,
                        "selected_learned_score": selected_learned,
                        "reference_to_selected_learned_gain": float(selected_learned - reference_learned),
                        "refinement_learned_gain": refinement_learned_gain,
                        "reference_exact_score": reference_exact,
                        "round0_best_exact_score": round0_exact,
                        "selected_exact_score": selected_exact,
                        "reference_to_selected_exact_delta": float(selected_exact - reference_exact),
                        "refinement_exact_delta": refinement_exact_delta,
                        "reference_exact_rank": reference_rank,
                        "round0_best_exact_rank": round0_rank,
                        "selected_exact_rank": selected_rank,
                        "reference_to_selected_exact_rank_damage": (selected_rank - reference_rank),
                        "refinement_exact_rank_damage": (refinement_exact_rank_damage),
                        "supports_within_call_exploitation": supports_within_call,
                        "identity_trajectories": {
                            "R": _identity_trajectory(
                                "R",
                                flat_sequences[reference_flat],
                                sequences,
                                learned,
                                exact,
                                injected,
                            ),
                            "B0": _identity_trajectory(
                                "B0",
                                flat_sequences[round0_best_flat],
                                sequences,
                                learned,
                                exact,
                                injected,
                            ),
                            "C": _identity_trajectory(
                                "C",
                                flat_sequences[selected_flat],
                                sequences,
                                learned,
                                exact,
                                injected,
                            ),
                        },
                        "cold_bank_sha256": _sequence_collection_sha256(arrays["cold_sequences"][prefix]),
                        "cold_reference_group_start": reference_group_start,
                        "cold_reference_candidate": cold_reference,
                        "cold_reference_sha256": _sequence_sha256(arrays["cold_sequences"][prefix][cold_reference]),
                        "cold_reference_exact_rank": cold_reference_exact_rank,
                        "cold_reference_learned_rank": cold_reference_learned_rank,
                        "cold_reference_rank_residual": (cold_reference_rank_residual),
                        "cold_learned_exact_spearman": spearman_correlation(cold_learned, cold_exact),
                    }
                )
    return rows


def _median(rows: list[dict[str, object]], field: str) -> float:
    return float(np.median([float(cast(float, row[field])) for row in rows]))


def _seed_summaries(call_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for seed in ALL_SEEDS:
        seed_rows = [row for row in call_rows if row["seed"] == seed]
        by_step = {step: [row for row in seed_rows if row["step"] == step] for step in AUDIT_STEPS}
        if any(len(rows) != len(EVAL_STARTS) for rows in by_step.values()):
            raise ValueError(f"seed {seed} does not have four calls per retained step")
        step0 = by_step[0]
        exploitation_starts = [
            {
                "episode_index": int(cast(int, row["episode_index"])),
                "supports": bool(row["supports_within_call_exploitation"]),
            }
            for row in step0
        ]
        exploitation_start_count = sum(bool(row["supports"]) for row in exploitation_starts)
        exploitation = {
            "median_refinement_learned_gain": _median(step0, "refinement_learned_gain"),
            "median_refinement_exact_delta": _median(step0, "refinement_exact_delta"),
            "median_refinement_exact_rank_damage": _median(step0, "refinement_exact_rank_damage"),
            "supporting_starts": exploitation_start_count,
            "required_starts": 3,
            "start_support": exploitation_starts,
        }
        exploitation_support = exploitation_start_count >= 3
        by_step_episode = {
            step: {int(cast(int, row["episode_index"])): row for row in by_step[step]} for step in AUDIT_STEPS
        }
        paired_rank_residual_changes: dict[int, list[float]] = {}
        paired_learned_rank_changes: dict[int, list[float]] = {}
        for step in (1, 2):
            paired_rank_residual_changes[step] = [
                float(
                    cast(
                        float,
                        by_step_episode[step][episode]["cold_reference_rank_residual"],
                    )
                )
                - float(
                    cast(
                        float,
                        by_step_episode[0][episode]["cold_reference_rank_residual"],
                    )
                )
                for episode in range(len(EVAL_STARTS))
            ]
            paired_learned_rank_changes[step] = [
                float(
                    cast(
                        float,
                        by_step_episode[step][episode]["cold_reference_learned_rank"],
                    )
                )
                - float(
                    cast(
                        float,
                        by_step_episode[0][episode]["cold_reference_learned_rank"],
                    )
                )
                for episode in range(len(EVAL_STARTS))
            ]
        shift_starts: list[dict[str, object]] = []
        for episode, residual_change in enumerate(paired_rank_residual_changes[2]):
            step0_row = by_step_episode[0][episode]
            step2_row = by_step_episode[2][episode]
            exact_rank0 = float(cast(float, step0_row["cold_reference_exact_rank"]))
            exact_rank2 = float(cast(float, step2_row["cold_reference_exact_rank"]))
            supports = _state_shift_support(
                step0_exact_rank=exact_rank0,
                step2_exact_rank=exact_rank2,
                rank_residual_deterioration=residual_change,
            )
            shift_starts.append(
                {
                    "episode_index": episode,
                    "step0_exact_rank": exact_rank0,
                    "step2_exact_rank": exact_rank2,
                    "rank_residual_deterioration": residual_change,
                    "supports": supports,
                }
            )
        shift_start_count = sum(bool(row["supports"]) for row in shift_starts)
        shift = {
            "median_step0_exact_rank": _median(by_step[0], "cold_reference_exact_rank"),
            "median_step2_exact_rank": _median(by_step[2], "cold_reference_exact_rank"),
            "median_step1_learned_rank_change": float(np.median(paired_learned_rank_changes[1])),
            "median_step2_learned_rank_change": float(np.median(paired_learned_rank_changes[2])),
            "median_step1_rank_residual_deterioration": float(np.median(paired_rank_residual_changes[1])),
            "median_step2_rank_residual_deterioration": float(np.median(paired_rank_residual_changes[2])),
            "supporting_starts": shift_start_count,
            "required_starts": 3,
            "paired_start_support": shift_starts,
        }
        shift_support = shift_start_count >= 3
        summaries.append(
            {
                "seed": seed,
                "phase": "replay" if seed in REPLAY_SEEDS else "confirmatory",
                "within_call_exploitation": {
                    **exploitation,
                    "supports": exploitation_support,
                },
                "visited_state_scorer_shift": {
                    **shift,
                    "supports": shift_support,
                },
            }
        )
    return summaries


def _decision(seed_summaries: list[dict[str, object]]) -> dict[str, object]:
    confirmatory = [row for row in seed_summaries if row["phase"] == "confirmatory"]
    if [int(cast(int, row["seed"])) for row in confirmatory] != list(CONFIRMATORY_SEEDS):
        raise ValueError("CL-001 confirmatory seed summaries are incomplete or unordered")
    exploitation_count = sum(
        bool(cast(dict[str, object], row["within_call_exploitation"])["supports"]) for row in confirmatory
    )
    shift_count = sum(
        bool(cast(dict[str, object], row["visited_state_scorer_shift"])["supports"]) for row in confirmatory
    )
    exploitation_passes = exploitation_count >= CONFIRMATORY_SUPPORT_FLOOR
    shift_passes = shift_count >= CONFIRMATORY_SUPPORT_FLOOR
    if exploitation_passes and shift_passes:
        classification = "within_call_exploitation_and_statewise_scorer_shift"
    elif exploitation_passes:
        classification = "within_call_exploitation_only"
    elif shift_passes:
        classification = "statewise_scorer_shift_only"
    else:
        classification = "neither_mechanism_supported"
    return {
        "classification": classification,
        "within_call_exploitation": {
            "supporting_seeds": exploitation_count,
            "required_seeds": CONFIRMATORY_SUPPORT_FLOOR,
            "passes": exploitation_passes,
        },
        "visited_state_scorer_shift": {
            "supporting_seeds": shift_count,
            "required_seeds": CONFIRMATORY_SUPPORT_FLOOR,
            "passes": shift_passes,
        },
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "replay_seeds_excluded": list(REPLAY_SEEDS),
    }


def _evaluation_count(sequences: int) -> dict[str, int]:
    return {
        "sequences": sequences,
        "transitions": sequences * injection.HORIZON,
    }


def _sum_evaluation_counts(*counts: dict[str, int]) -> dict[str, int]:
    return {
        "sequences": sum(count["sequences"] for count in counts),
        "transitions": sum(count["transitions"] for count in counts),
    }


def _evaluation_accounting(
    trajectories: list[dict[str, object]],
) -> dict[str, object]:
    expected_online_per_seed = (
        len(EVAL_STARTS) * injection.EVAL_STEPS * injection.NATIVE_CANDIDATES * injection.NATIVE_ITERATIONS
    )
    provider_sequences = 0
    online_sequences = 0
    for expected_seed, row in zip(ALL_SEEDS, trajectories, strict=True):
        if int(cast(int, row["seed"])) != expected_seed:
            raise ValueError("CL-001 accounting trajectories are unordered")
        row_online = int(cast(int, row["learned_sequence_evaluations"]))
        row_transitions = int(cast(int, row["learned_transition_evaluations"]))
        if row_online != expected_online_per_seed or row_transitions != (row_online * injection.HORIZON):
            raise ValueError("CL-001 learned planner accounting is inconsistent")
        provider = cast(dict[str, object], row["provider"])
        provider_row_sequences = int(cast(int, provider["oracle_sequence_evaluations"]))
        provider_row_transitions = int(cast(int, provider["oracle_transition_evaluations"]))
        if provider_row_transitions != provider_row_sequences * injection.HORIZON:
            raise ValueError("CL-001 provider accounting is inconsistent")
        online_sequences += row_online
        provider_sequences += provider_row_sequences

    cold_per_seed = len(EVAL_STARTS) * len(AUDIT_STEPS) * injection.NATIVE_CANDIDATES
    live_pool_per_seed = cold_per_seed * injection.NATIVE_ITERATIONS
    audited_online = _evaluation_count(online_sequences)
    audited_cold_learned = _evaluation_count(len(ALL_SEEDS) * cold_per_seed)
    audited_provider = _evaluation_count(provider_sequences)
    audited_live_exact = _evaluation_count(len(ALL_SEEDS) * live_pool_per_seed)
    audited_cold_exact = _evaluation_count(len(ALL_SEEDS) * cold_per_seed)
    audited_learned_total = _sum_evaluation_counts(audited_online, audited_cold_learned)
    audited_oracle_total = _sum_evaluation_counts(audited_provider, audited_live_exact, audited_cold_exact)

    parity_online = _evaluation_count(len(CONFIRMATORY_SEEDS) * expected_online_per_seed)
    provider_per_seed = provider_sequences // len(ALL_SEEDS)
    if provider_per_seed * len(ALL_SEEDS) != provider_sequences:
        raise ValueError("CL-001 provider counts differ across audited seeds")
    parity_provider = _evaluation_count(len(CONFIRMATORY_SEEDS) * provider_per_seed)
    return {
        "scope": ("one formal _execute call; excludes model-training updates and subsequent verifier reruns"),
        "audited_path": {
            "learned": {
                "online_planner": audited_online,
                "cold_bank": audited_cold_learned,
                "total": audited_learned_total,
            },
            "oracle": {
                "reference_provider": audited_provider,
                "live_pool_audit": audited_live_exact,
                "cold_bank": audited_cold_exact,
                "total": audited_oracle_total,
            },
        },
        "confirmatory_parity_overhead": {
            "learned": {"online_planner": parity_online, "total": parity_online},
            "oracle": {"reference_provider": parity_provider, "total": parity_provider},
        },
        "actual_execute_total": {
            "learned": _sum_evaluation_counts(audited_learned_total, parity_online),
            "oracle": _sum_evaluation_counts(audited_oracle_total, parity_provider),
        },
        "fast_verifier_exact_recomputation": _sum_evaluation_counts(audited_live_exact, audited_cold_exact),
    }


def _derive(
    arrays: dict[str, np.ndarray],
    trajectories: list[dict[str, object]],
) -> dict[str, object]:
    pools = _pool_rows(arrays)
    calls = _call_rows(arrays, trajectories)
    seeds = _seed_summaries(calls)
    return {
        "pool_rows": pools,
        "call_rows": calls,
        "seed_summaries": seeds,
        "decision": _decision(seeds),
        "evaluation_accounting": _evaluation_accounting(trajectories),
    }


def _validate_array_shapes(arrays: dict[str, np.ndarray]) -> None:
    expected = _empty_arrays()
    if set(arrays) != set(expected):
        raise ValueError("CL-001 tensor arrays are incomplete")
    for name, template in expected.items():
        value = arrays[name]
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"CL-001 tensor {name} has {value.shape}/{value.dtype}, expected {template.shape}/{template.dtype}"
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"CL-001 tensor {name} contains non-finite values")
    if not np.array_equal(arrays["seeds"], np.asarray(ALL_SEEDS, dtype=np.int64)):
        raise ValueError("CL-001 tensor seed axis differs from the frozen protocol")


def _verify_exact_scores(arrays: dict[str, np.ndarray]) -> None:
    for seed_axis in range(len(ALL_SEEDS)):
        for episode in range(len(EVAL_STARTS)):
            for step_axis in range(len(AUDIT_STEPS)):
                prefix = (seed_axis, episode, step_axis)
                raw = arrays["raw_states"][prefix]
                for iteration in range(injection.NATIVE_ITERATIONS):
                    expected = exact_discounted_scores(
                        raw,
                        arrays["sequences"][prefix + (iteration,)],
                    )
                    if not np.array_equal(
                        expected,
                        arrays["exact_scores"][prefix + (iteration,)],
                    ):
                        raise ValueError("CL-001 stored pool exact scores do not recompute")
                cold_expected = exact_discounted_scores(raw, arrays["cold_sequences"][prefix])
                if not np.array_equal(cold_expected, arrays["cold_exact_scores"][prefix]):
                    raise ValueError("CL-001 stored cold-bank exact scores do not recompute")


def _verify_tensor_lineage(arrays: dict[str, np.ndarray]) -> None:
    if np.max(np.abs(arrays["sequences"])) > 1.0 or np.max(np.abs(arrays["cold_sequences"])) > 1.0:
        raise ValueError("CL-001 action tensor exceeds frozen bounds")
    absent_ood = ~arrays["latent_ood_present"]
    if np.any(arrays["latent_ood"][absent_ood] != 0.0):
        raise ValueError("CL-001 absent latent OOD values must use the zero sentinel")

    keep_count = min(
        injection.NATIVE_ELITES,
        int(np.ceil(0.3 * injection.NATIVE_ELITES)),
    )
    for seed_axis, seed_value in enumerate(arrays["seeds"].tolist()):
        seed = int(seed_value)
        for episode in range(len(EVAL_STARTS)):
            cold = arrays["cold_sequences"][seed_axis, episode]
            if not all(np.array_equal(cold[0], item) for item in cold[1:]):
                raise ValueError("CL-001 cold-bank sequence set changed across states")
            if not np.array_equal(
                cold[0, :COMMON_NATIVE_COUNT],
                _common_native_sequences(seed, episode),
            ):
                raise ValueError("CL-001 cold-bank native prefix does not regenerate")

            for step_axis, step in enumerate(AUDIT_STEPS):
                prefix = (seed_axis, episode, step_axis)
                sequences = arrays["sequences"][prefix]
                learned = arrays["learned_scores"][prefix]
                masks = arrays["injected"][prefix]
                expected_mask = np.zeros(injection.NATIVE_CANDIDATES, dtype=bool)
                carried = 0 if step == 0 else keep_count
                fresh = injection.NATIVE_CANDIDATES - carried
                expected_mask[fresh - injection.INJECTION_COUNT : fresh] = True
                if not np.array_equal(masks[0], expected_mask):
                    raise ValueError("CL-001 persisted first-round injection mask differs")

                for iteration in range(1, injection.NATIVE_ITERATIONS):
                    elite_indices = np.argsort(learned[iteration - 1])[-injection.NATIVE_ELITES :][::-1]
                    carried_indices = elite_indices[:keep_count]
                    if not np.array_equal(
                        sequences[iteration, -keep_count:],
                        sequences[iteration - 1, carried_indices],
                    ):
                        raise ValueError("CL-001 persisted within-call elite carry differs")
                    expected_carried_mask = np.zeros(injection.NATIVE_CANDIDATES, dtype=bool)
                    expected_carried_mask[-keep_count:] = masks[iteration - 1, carried_indices]
                    if not np.array_equal(masks[iteration], expected_carried_mask):
                        raise ValueError("CL-001 persisted carried provenance differs")

                reference_start = COMMON_NATIVE_COUNT + step_axis * injection.INJECTION_COUNT
                reference_stop = reference_start + injection.INJECTION_COUNT
                references = sequences[0, masks[0]]
                if not np.array_equal(cold[step_axis, reference_start:reference_stop], references):
                    raise ValueError("CL-001 cold-bank reference block differs from pool")
                if not np.array_equal(
                    arrays["cold_exact_scores"][prefix][reference_start:reference_stop],
                    arrays["exact_scores"][prefix][0, masks[0]],
                ):
                    raise ValueError("CL-001 cold/pool reference exact scores differ")

                if step_axis > 0:
                    previous_prefix = (seed_axis, episode, step_axis - 1)
                    previous_sequences = arrays["sequences"][previous_prefix]
                    previous_learned = arrays["learned_scores"][previous_prefix]
                    previous_elites = np.argsort(previous_learned[-1])[-injection.NATIVE_ELITES :][::-1][:keep_count]
                    expected_warm = CandidateLandscapePlanner._shift_sequences(previous_sequences[-1, previous_elites])
                    if not np.array_equal(sequences[0, -keep_count:], expected_warm):
                        raise ValueError("CL-001 persisted cross-step warm carry differs")


def _verify_trajectory_records(
    arrays: dict[str, np.ndarray],
    trajectories: list[dict[str, object]],
) -> None:
    expected_keys = {
        "arm",
        "seed",
        "model_sha256",
        "mean_eval_return",
        "success_rate",
        "episode_returns",
        "episode_successes",
        "final_states",
        "action_traces",
        "plan_diagnostics",
        "learned_sequence_evaluations",
        "learned_transition_evaluations",
        "provider",
    }
    expected_online = (
        len(EVAL_STARTS) * injection.EVAL_STEPS * injection.NATIVE_CANDIDATES * injection.NATIVE_ITERATIONS
    )
    expected_provider_sequences = (
        REFERENCE_SEARCH.evaluated_sequences
        + STANDARD_PROPOSAL_COUNT
        + REFERENCE_ELITE_COUNT
        + 2 * injection.INJECTION_COUNT
    )
    retained_axis = {step: axis for axis, step in enumerate(AUDIT_STEPS)}

    for seed_axis, (seed, row) in enumerate(zip(ALL_SEEDS, trajectories, strict=True)):
        if set(row) != expected_keys:
            raise ValueError("CL-001 trajectory row schema differs")
        if row["arm"] != "privileged_injection" or int(cast(int, row["seed"])) != seed:
            raise ValueError("CL-001 trajectory identity differs")
        if int(cast(int, row["learned_sequence_evaluations"])) != expected_online:
            raise ValueError("CL-001 trajectory learned sequence count differs")
        if int(cast(int, row["learned_transition_evaluations"])) != (expected_online * injection.HORIZON):
            raise ValueError("CL-001 trajectory learned transition count differs")

        actions = np.asarray(row["action_traces"], dtype=np.float64)
        returns = np.asarray(row["episode_returns"], dtype=np.float64)
        successes = [bool(value) for value in cast(list[object], row["episode_successes"])]
        finals = np.asarray(row["final_states"], dtype=np.float64)
        if actions.shape != (len(EVAL_STARTS), injection.EVAL_STEPS, 2):
            raise ValueError("CL-001 trajectory action shape differs")
        if returns.shape != (len(EVAL_STARTS),) or finals.shape != (
            len(EVAL_STARTS),
            3,
        ):
            raise ValueError("CL-001 trajectory outcome shape differs")
        if len(successes) != len(EVAL_STARTS) or not np.all(np.isfinite(actions)):
            raise ValueError("CL-001 trajectory values are incomplete")
        if float(cast(float, row["mean_eval_return"])) != float(np.mean(returns)):
            raise ValueError("CL-001 mean return does not recompute")
        if float(cast(float, row["success_rate"])) != float(np.mean(successes)):
            raise ValueError("CL-001 success rate does not recompute")

        raw_by_call: list[np.ndarray] = []
        recomputed_returns: list[float] = []
        recomputed_finals: list[np.ndarray] = []
        recomputed_successes: list[bool] = []
        for episode, start in enumerate(EVAL_STARTS):
            env = injection.BridgeControlEnv()
            observation = env.set_state(start)
            total = 0.0
            for step in range(injection.EVAL_STEPS):
                raw = np.asarray(observation.data, dtype=np.float64)
                raw_by_call.append(raw.copy())
                if step in retained_axis and not np.array_equal(
                    arrays["raw_states"][seed_axis, episode, retained_axis[step]],
                    raw,
                ):
                    raise ValueError("CL-001 retained raw state differs from trajectory")
                observation, reward, _ = env.step(Action(data=actions[episode, step]))
                total += reward
            final = np.asarray(observation.data, dtype=np.float64)
            recomputed_returns.append(float(total))
            recomputed_finals.append(final)
            recomputed_successes.append(injection._success(final))
        if not np.array_equal(returns, np.asarray(recomputed_returns)):
            raise ValueError("CL-001 trajectory returns do not replay exactly")
        if not np.array_equal(finals, np.asarray(recomputed_finals)):
            raise ValueError("CL-001 trajectory final states do not replay exactly")
        if successes != recomputed_successes:
            raise ValueError("CL-001 trajectory successes do not replay exactly")

        provider = cast(dict[str, object], row["provider"])
        calls = cast(list[dict[str, object]], provider["calls"])
        if (
            provider["mode"] != "privileged"
            or int(cast(int, provider["call_count"])) != len(raw_by_call)
            or len(calls) != len(raw_by_call)
        ):
            raise ValueError("CL-001 provider summary identity differs")
        all_reference_scores: list[float] = []
        all_output_scores: list[float] = []
        for call_index, (raw, call) in enumerate(zip(raw_by_call, calls, strict=True)):
            expected_bank_seed = int(
                np.random.SeedSequence([PROVIDER_SEED_SALT, seed, call_index]).generate_state(1, dtype=np.uint32)[0]
            )
            reference_scores = np.asarray(call["reference_exact_scores"], dtype=np.float64)
            output_scores = np.asarray(call["output_exact_scores"], dtype=np.float64)
            if (
                int(cast(int, call["call_index"])) != call_index
                or int(cast(int, call["bank_seed"])) != expected_bank_seed
                or call["raw_sha256"] != ExactReferenceProvider._raw_hash(raw)
                or call["mode"] != "privileged"
                or int(cast(int, call["time_shift"])) != 0
                or reference_scores.shape != (injection.INJECTION_COUNT,)
                or not np.array_equal(reference_scores, output_scores)
                or not np.all(np.isfinite(reference_scores))
                or int(cast(int, call["oracle_sequence_evaluations"])) != expected_provider_sequences
                or int(cast(int, call["oracle_transition_evaluations"]))
                != expected_provider_sequences * injection.HORIZON
            ):
                raise ValueError("CL-001 provider call does not recompute")
            if not isinstance(call["bank_sha256"], str) or len(cast(str, call["bank_sha256"])) != 64:
                raise ValueError("CL-001 provider bank hash is malformed")
            all_reference_scores.extend(reference_scores.tolist())
            all_output_scores.extend(output_scores.tolist())

            episode, step = divmod(call_index, injection.EVAL_STEPS)
            if step in retained_axis:
                step_axis = retained_axis[step]
                mask = arrays["injected"][seed_axis, episode, step_axis, 0]
                if not np.array_equal(
                    arrays["exact_scores"][seed_axis, episode, step_axis, 0][mask],
                    output_scores,
                ):
                    raise ValueError("CL-001 provider scores differ from retained pool")

        provider_total = expected_provider_sequences * len(calls)
        if int(cast(int, provider["oracle_sequence_evaluations"])) != provider_total:
            raise ValueError("CL-001 provider sequence total differs")
        if int(cast(int, provider["oracle_transition_evaluations"])) != (provider_total * injection.HORIZON):
            raise ValueError("CL-001 provider transition total differs")
        if float(cast(float, provider["mean_reference_exact_score"])) != float(np.mean(all_reference_scores)) or float(
            cast(float, provider["mean_output_exact_score"])
        ) != float(np.mean(all_output_scores)):
            raise ValueError("CL-001 provider score means differ")
        if float(cast(float, provider["mean_exact_score_delta"])) != 0.0:
            raise ValueError("CL-001 privileged provider delta must be zero")

        diagnostics = cast(list[dict[str, object]], row["plan_diagnostics"])
        if len(diagnostics) != len(raw_by_call):
            raise ValueError("CL-001 plan diagnostics are incomplete")
        for call_index, diagnostic in enumerate(diagnostics):
            episode, step = divmod(call_index, injection.EVAL_STEPS)
            if (
                int(cast(int, diagnostic["episode_index"])) != episode
                or int(cast(int, diagnostic["step"])) != step
                or int(cast(int, diagnostic["injected_count"])) != injection.INJECTION_COUNT
                or int(cast(int, diagnostic["candidate_eval_count"]))
                != injection.NATIVE_CANDIDATES * injection.NATIVE_ITERATIONS
                or int(cast(int, diagnostic["candidate_transition_eval_count"]))
                != injection.NATIVE_CANDIDATES * injection.NATIVE_ITERATIONS * injection.HORIZON
                or bool(diagnostic["episode_success"]) != successes[episode]
            ):
                raise ValueError("CL-001 plan diagnostic accounting differs")
            if step not in retained_axis:
                continue
            step_axis = retained_axis[step]
            learned = arrays["learned_scores"][seed_axis, episode, step_axis]
            injected = arrays["injected"][seed_axis, episode, step_axis]
            first_best = int(np.argmax(learned[0]))
            first_elites = np.argsort(learned[0])[-injection.NATIVE_ELITES :][::-1]
            selected_flat = int(np.argmax(learned.reshape(-1)))
            selected_iteration, selected_candidate = divmod(selected_flat, injection.NATIVE_CANDIDATES)
            selected_injected = bool(injected[selected_iteration, selected_candidate])
            if (
                bool(diagnostic["first_round_best_injected"]) != bool(injected[0, first_best])
                or int(cast(int, diagnostic["injected_top_elite_count"]))
                != int(np.count_nonzero(injected[0, first_elites]))
                or bool(diagnostic["best_sequence_injected"]) != selected_injected
                or diagnostic["selected_first_action_source"] != ("injected" if selected_injected else "native")
            ):
                raise ValueError("CL-001 retained plan diagnostics differ from pools")


def _verify_parity_records(
    trajectories: list[dict[str, object]],
    parity: list[dict[str, object]],
) -> None:
    if len(parity) != len(ALL_SEEDS):
        raise ValueError("CL-001 parity evidence is incomplete")
    parent_rows = _parent_privileged_rows()
    expected_zero_difference = {
        "max_return_abs": 0.0,
        "max_final_state_abs": 0.0,
        "max_action_abs": 0.0,
        "successes_equal": True,
    }
    for seed, trajectory, record in zip(ALL_SEEDS, trajectories, parity, strict=True):
        phase = "replay" if seed in REPLAY_SEEDS else "confirmatory"
        trajectory_sha = _canonical_json_sha256(trajectory)
        provider_sha = _canonical_json_sha256(trajectory["provider"])
        if (
            int(cast(int, record["seed"])) != seed
            or record["phase"] != phase
            or record["model_sha256"] != trajectory["model_sha256"]
            or record["instrumented_row_sha256"] != trajectory_sha
            or record["provider_summary_sha256"] != provider_sha
        ):
            raise ValueError("CL-001 parity record identity differs")
        if phase == "replay":
            parent_row = parent_rows[seed]
            if (
                not bool(record["parent_row_canonical_equal"])
                or _canonical_json_value(trajectory) != _canonical_json_value(parent_row)
                or record["parent_row_sha256"] != _canonical_json_sha256(parent_row)
            ):
                raise ValueError("CL-001 replay parity does not bind to PI-003")
        elif (
            not bool(record["provider_summaries_canonical_equal"])
            or record["ordinary_row_sha256"] != trajectory_sha
            or _canonical_json_value(record["ordinary_instrumented_difference"]) != expected_zero_difference
        ):
            raise ValueError("CL-001 confirmatory parity record differs")


def _csv_text(call_rows: list[dict[str, object]]) -> str:
    fields = (
        "seed",
        "phase",
        "episode_index",
        "step",
        "round0_best_injected",
        "selected_first_iteration",
        "refinement_learned_gain",
        "refinement_exact_delta",
        "refinement_exact_rank_damage",
        "supports_within_call_exploitation",
        "cold_reference_exact_rank",
        "cold_reference_learned_rank",
        "cold_reference_rank_residual",
        "cold_learned_exact_spearman",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in call_rows:
        writer.writerow({field: row[field] for field in fields})
    return handle.getvalue()


def _report_text(results: dict[str, Any]) -> str:
    decision = cast(dict[str, Any], results["decision"])
    exploitation = cast(dict[str, Any], decision["within_call_exploitation"])
    shift = cast(dict[str, Any], decision["visited_state_scorer_shift"])
    summaries = cast(list[dict[str, Any]], results["seed_summaries"])
    confirmatory = [row for row in summaries if row["phase"] == "confirmatory"]
    accounting = cast(dict[str, Any], results["evaluation_accounting"])
    execute_total = cast(dict[str, Any], accounting["actual_execute_total"])
    learned_total = cast(dict[str, Any], execute_total["learned"])
    oracle_total = cast(dict[str, Any], execute_total["oracle"])
    lines = [
        "# CL-001 candidate-landscape result",
        "",
        f"**Status:** {results['status']}  ",
        "**Scope:** non-gated BridgeControl mechanism evidence",
        "",
        "## Outcome",
        "",
        f"Classification: **{decision['classification']}**.",
        "",
        f"- Within-call exploitation: {exploitation['supporting_seeds']}/12 seeds "
        f"(required 10) — {'supported' if exploitation['passes'] else 'not supported'}.",
        f"- Visited-state scorer shift: {shift['supporting_seeds']}/12 seeds "
        f"(required 10) — {'supported' if shift['passes'] else 'not supported'}.",
        "",
        "Seeds 0..7 are replay-only and excluded from these decisions.",
        "",
        "## Confirmatory seed summaries",
        "",
        "| Seed | supporting starts | learned gain | exact delta | rank damage | "
        "step-2 residual shift | supporting pairs | Exploit | Shift |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in confirmatory:
        exploit = cast(dict[str, Any], row["within_call_exploitation"])
        state_shift = cast(dict[str, Any], row["visited_state_scorer_shift"])
        lines.append(
            f"| {row['seed']} | {exploit['supporting_starts']}/4 | "
            f"{exploit['median_refinement_learned_gain']:.6f} | "
            f"{exploit['median_refinement_exact_delta']:.6f} | "
            f"{exploit['median_refinement_exact_rank_damage']:.2f} | "
            f"{state_shift['median_step2_rank_residual_deterioration']:.2f} | "
            f"{state_shift['supporting_starts']}/4 | "
            f"{'yes' if exploit['supports'] else 'no'} | "
            f"{'yes' if state_shift['supports'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "CL-001 uses one authored fixture and deterministic model-seed blocks. "
            "Exact simulation is diagnostic oracle compute, and no production "
            "planner behavior or benchmark gate changed.",
            "",
            "The 10/12 rule is descriptive model-seed robustness, not a binomial "
            "p-value or an environment-level generality claim.",
            "",
            "One formal execution used "
            f"{learned_total['sequences']} learned-scored sequences "
            f"({learned_total['transitions']} transitions) and "
            f"{oracle_total['sequences']} oracle-scored sequences "
            f"({oracle_total['transitions']} transitions), including the frozen "
            "confirmatory parity run.",
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
    """Execute every frozen replay and confirmatory seed exactly once."""

    _assert_safe_output(output)
    verification = verify(output)
    if verification["outcomes"] != "prepared_only":
        raise FileExistsError("CL-001 formal execution is one-shot")
    formal_start = _mark_formal_started(output)
    execution, arrays = _execute(output / INPUT_COPY)
    _write_arrays(output / TENSOR_FILE, arrays)
    protocol = _read_json(output / "protocol.json")
    input_manifest = _read_json(output / "input-manifest.json")
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_candidate_landscape",
        "interpretation_scope": "non-gated BridgeControl mechanism evidence",
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
        **execution,
    }
    _write_json(output / RESULT_FILE, results)
    (output / CSV_FILE).write_text(
        _csv_text(cast(list[dict[str, object]], results["call_rows"])),
        encoding="utf-8",
    )
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
        raise ValueError("CL-001 result identity does not match")
    if results.get("status") != "completed_candidate_landscape":
        raise ValueError("CL-001 result status does not match")
    protocol_sha = sha256(_canonical_json_bytes(protocol)).hexdigest()
    if results.get("protocol") != protocol or results.get("protocol_sha256") != protocol_sha:
        raise ValueError("CL-001 results are not bound to the frozen protocol")
    if results.get("input_manifest") != input_manifest or results.get("input_manifest_sha256") != _file_hash(
        output / "input-manifest.json"
    ):
        raise ValueError("CL-001 results are not bound to the input manifest")
    formal_start = _read_json(output / STARTED_FILE)
    if (
        formal_start != _formal_start_record(output)
        or results.get("formal_start") != formal_start
        or results.get("formal_start_sha256") != _file_hash(output / STARTED_FILE)
    ):
        raise ValueError("CL-001 results are not bound to the formal-start record")
    if cast(dict[str, Any], results["repository"])["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("CL-001 result source hashes do not match")

    arrays = _load_arrays(output / TENSOR_FILE)
    _validate_array_shapes(arrays)
    tensor_record = cast(dict[str, Any], results["tensor_package"])
    expected_tensor = {
        "path": str(TENSOR_FILE),
        "file_sha256": _file_hash(output / TENSOR_FILE),
        **_arrays_metadata(arrays),
    }
    if tensor_record != expected_tensor:
        raise ValueError("CL-001 tensor package metadata does not match")
    _verify_exact_scores(arrays)
    _verify_tensor_lineage(arrays)

    trajectories = cast(list[dict[str, object]], results["trajectories"])
    if [int(cast(int, row["seed"])) for row in trajectories] != list(ALL_SEEDS):
        raise ValueError("CL-001 trajectories are incomplete or unordered")
    _verify_trajectory_records(arrays, trajectories)
    derived = _derive(arrays, trajectories)
    for field in (
        "pool_rows",
        "call_rows",
        "seed_summaries",
        "decision",
        "evaluation_accounting",
    ):
        if _canonical_json_value(results[field]) != _canonical_json_value(derived[field]):
            raise ValueError(f"CL-001 saved {field} does not recompute")
    _verify_parity_records(trajectories, cast(list[dict[str, object]], results["parity"]))
    if (output / CSV_FILE).read_text(encoding="utf-8") != _csv_text(
        cast(list[dict[str, object]], results["call_rows"])
    ):
        raise ValueError("CL-001 CSV is not canonical")
    if (output / REPORT_FILE).read_text(encoding="utf-8") != _report_text(results):
        raise ValueError("CL-001 report is not canonical")
    artifact_manifest = _read_json(output / "artifact-manifest.json")
    expected_artifacts = {str(path): _file_hash(output / path) for path in ARTIFACT_PATHS}
    if artifact_manifest != {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifacts": expected_artifacts,
    }:
        raise ValueError("CL-001 artifact manifest does not match")
    return results, arrays


def verify(
    output: Path = DEFAULT_OUTPUT,
    *,
    require_results: bool = False,
    semantic: bool = False,
) -> dict[str, object]:
    """Verify persisted derivations, optionally retraining and regenerating all seeds."""

    protocol = _read_json(output / "protocol.json")
    if protocol != protocol_record():
        raise ValueError("saved CL-001 protocol does not match current sources")
    input_manifest = _read_json(output / "input-manifest.json")
    if input_manifest != _expected_input_manifest(output):
        raise ValueError("saved CL-001 input manifest or parent snapshot has drifted")
    result_path = output / RESULT_FILE
    started_path = output / STARTED_FILE
    if result_path.exists():
        if not all((output / path).exists() for path in OUTCOME_PATHS):
            raise ValueError("CL-001 outcome package is partial")
        if _read_json(started_path) != _formal_start_record(output):
            raise ValueError("CL-001 formal-start record does not match")
        saved, arrays = _verify_outcomes(output, protocol, input_manifest)
        outcomes = "verified_results"
        if semantic:
            regenerated, regenerated_arrays = _execute(output / INPUT_COPY)
            for name in ARRAY_NAMES:
                if not np.array_equal(arrays[name], regenerated_arrays[name]):
                    raise ValueError(f"CL-001 semantic regeneration differs in {name}")
            for field in (
                "trajectories",
                "parity",
                "pool_rows",
                "call_rows",
                "seed_summaries",
                "decision",
                "evaluation_accounting",
            ):
                if _canonical_json_value(saved[field]) != _canonical_json_value(regenerated[field]):
                    raise ValueError(f"CL-001 semantic regeneration differs in {field}")
            outcomes = "verified_semantic_results"
    elif started_path.exists():
        raise ValueError("CL-001 formal execution started but is incomplete; the identifier is terminal")
    elif any((output / path).exists() for path in OUTCOME_PATHS):
        raise ValueError("CL-001 has partial outcomes without a formal-start record")
    else:
        outcomes = "prepared_only"
    if require_results and outcomes == "prepared_only":
        raise ValueError("complete CL-001 results are required")
    return {
        "status": "verified",
        "outcomes": outcomes,
        "protocol_sha256": sha256(_canonical_json_bytes(protocol)).hexdigest(),
        "input_manifest_sha256": _file_hash(output / "input-manifest.json"),
    }


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify(output, require_results=True)
    return _read_json(output / RESULT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-001 candidate-landscape experiment")
    parser.add_argument(
        "command",
        choices=("prepare", "run", "verify", "verify-semantic", "analyze"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
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
