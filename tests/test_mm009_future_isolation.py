from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bench.multimodal_causal_diagnostics import future_isolation, preparation, records, runtime

PROTOCOL_SHA256 = records.PROTOCOL_SHA256
CONFIG_SHA256 = "2" * 64


@pytest.fixture(scope="module")
def isolation_roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("mm009-isolation-roots").resolve()
    dependency = root / "dependency"
    stdlib = root / "stdlib"
    custody = root / "custody"
    runtime.build_numpy_dependency_closure(dependency)
    runtime.build_stdlib_closure(stdlib)
    runtime.build_custody_runtime(custody)
    return {"custody": custody, "dependency": dependency, "stdlib": stdlib}


def _full(value: float) -> np.ndarray:
    return np.full((3, 64, 64), value, dtype="<f8")


def _target_arrays(ordinal: int) -> dict[str, np.ndarray]:
    rows = preparation.canonical_row_index()
    mapping = preparation.half_cycle_derangement(rows)
    row = rows[ordinal]
    return {
        "current_timestamp": np.asarray([row.current_timestamp], dtype="<f8"),
        "deranged_future": _full(float(ordinal) + 0.75),
        "derangement_row_ordinal": np.asarray([int(mapping[ordinal])], dtype="<i8"),
        "future": _full(float(ordinal) + 0.25),
        "future_timestamp": np.asarray([row.future_timestamp], dtype="<f8"),
        "row_ordinal": np.asarray([ordinal], dtype="<i8"),
        "video_id": np.asarray([row.video_id], dtype="<U11"),
    }


def _build_formal_census(root: Path, *, include_targets: bool = True) -> tuple[Path, Path, Path]:
    target_root = root / "rows" / "target"
    prediction_root = root / "predictions"
    target_root.mkdir(parents=True)
    prediction_root.mkdir(parents=True)
    chain: list[records.JsonValue] = []
    last_commit_sha256 = ""
    for ordinal in range(preparation.MATCHED_ROWS):
        row_name = f"{ordinal:06d}"
        target_directory = target_root / row_name
        prediction_directory = prediction_root / row_name
        if include_targets:
            target_directory.mkdir()
        prediction_directory.mkdir(parents=True)
        if include_targets:
            preparation.write_target_row_npz(
                target_directory / future_isolation.TARGET_FILE,
                _target_arrays(ordinal),
            )
        prediction = prediction_directory / future_isolation.PREDICTION_FILE
        evidence = prediction_directory / future_isolation.WORKER_EVIDENCE_FILE
        commit = prediction_directory / future_isolation.COMMIT_FILE
        records.write_immutable_bytes_exclusive(prediction, f"prediction-{ordinal}\n".encode("ascii"))
        records.write_immutable_json_exclusive(
            evidence,
            {
                "ordinal": ordinal,
                "schema_version": "test-worker-evidence-v1",
            },
        )
        records.write_immutable_json_exclusive(
            commit,
            {
                "ordinal": ordinal,
                "schema_version": "test-supervisor-commit-v1",
            },
        )
        last_commit_sha256 = records.file_sha256(commit)
        chain.append(
            {
                "commit_sha256": last_commit_sha256,
                "ordinal": ordinal,
                "prediction_sha256": records.file_sha256(prediction),
                "worker_evidence_sha256": records.file_sha256(evidence),
            }
        )
    freeze_path = root / "prediction-freeze.json"
    records.write_immutable_json_exclusive(
        freeze_path,
        {
            "chain": chain,
            "config_sha256": CONFIG_SHA256,
            "genesis_sha256": hashlib.sha256(b"genesis").hexdigest(),
            "last_commit_sha256": last_commit_sha256,
            "protocol_sha256": PROTOCOL_SHA256,
            "row_count": preparation.MATCHED_ROWS,
            "schema_version": "mm009-prediction-freeze-v1",
            "status": "all_predictions_frozen_before_target_scoring",
        },
    )
    return freeze_path, target_root, prediction_root


def test_exact_mutation_primitives_are_deterministic_finite_and_bit_local() -> None:
    first = future_isolation.random_finite_replacement((3, 64, 64), 17)
    second = future_isolation.random_finite_replacement((3, 64, 64), 17)
    other = future_isolation.random_finite_replacement((3, 64, 64), 18)
    assert first.dtype == np.dtype("<f8")
    assert first.flags.c_contiguous
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)

    value: np.ndarray = np.arange(3 * 64 * 64, dtype="<f8").reshape(3, 64, 64)
    reversed_value = future_isolation.spatial_reverse(value)
    assert np.array_equal(reversed_value, value[:, ::-1, ::-1])
    assert reversed_value.flags.c_contiguous

    flipped = future_isolation.flip_first_central_lsb(value)
    before_bits: np.ndarray = value.view("<u8")
    after_bits: np.ndarray = flipped.view("<u8")
    changed = np.argwhere(before_bits != after_bits)
    assert changed.tolist() == [[0, 8, 8]]
    assert int(before_bits[0, 8, 8]) ^ int(after_bits[0, 8, 8]) == 1
    assert np.isfinite(flipped[0, 8, 8])
    assert flipped[0, 8, 8] != value[0, 8, 8]


@pytest.mark.parametrize(
    "module_name",
    [
        "bench.multimodal_causal_diagnostics.predictor",
        "bench.multimodal_causal_diagnostics.worker",
        "bench.multimodal_mechanism_diagnostics.fitting_v22",
    ],
)
def test_fresh_custody_process_cannot_import_fitting_authority(
    isolation_roots: dict[str, Path],
    module_name: str,
) -> None:
    code = """
import importlib
import sys
sys.path[:] = sys.argv[1:5]
name = sys.argv[5]
try:
    importlib.import_module(name)
except ModuleNotFoundError as error:
    if error.name is None or not (error.name == name or name.startswith(error.name + ".")):
        raise
else:
    raise SystemExit("excluded module imported")
"""
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(isolation_roots["custody"]),
            str(isolation_roots["dependency"]),
            str(isolation_roots["stdlib"]),
            str(isolation_roots["stdlib"] / "lib-dynload"),
            module_name,
        ),
        env={"PATH": os.environ.get("PATH", "")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_fresh_process_sweeps_all_targets_without_loading_fitters(
    tmp_path: Path,
    isolation_roots: dict[str, Path],
) -> None:
    freeze, target_root, prediction_root = _build_formal_census(tmp_path)
    output_root = tmp_path / "future-output"
    output_root.mkdir()
    output = output_root / "future-isolation.json"
    launcher = isolation_roots["custody"] / runtime.FUTURE_ISOLATION_LAUNCHER_RELATIVE
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(launcher),
            str(isolation_roots["custody"]),
            str(isolation_roots["dependency"]),
            str(isolation_roots["stdlib"]),
            str(freeze),
            str(target_root),
            str(prediction_root),
            str(output),
        ),
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert records.file_record(output)["mode"] == 0o444
    payload = output.read_bytes()
    value = json.loads(payload)
    assert payload == records.json_file_bytes(value)
    assert value["schema_version"] == future_isolation.SCHEMA_VERSION
    assert value["status"] == "passed_before_target_scoring"
    assert value["import_guard"] == {
        "all_absent": True,
        "forbidden_modules": list(future_isolation.FORBIDDEN_MODULES),
    }
    assert value["mutation_counts"] == {
        "byte_lsb_finite_valid": preparation.MATCHED_ROWS,
        "deranged_future_valid": preparation.MATCHED_ROWS,
        "nan_rejected": preparation.MATCHED_ROWS,
        "random_finite_valid": preparation.MATCHED_ROWS,
        "rows": preparation.MATCHED_ROWS,
        "spatial_reverse_valid": preparation.MATCHED_ROWS,
    }
    assert value["source_side_manifest"]["before_after_bit_exact"] is True
    assert value["source_side_manifest"]["file_count"] == 1 + 3 * preparation.MATCHED_ROWS
    assert value["source_side_aggregate_manifest_sha256"] == value["source_side_manifest"]["sha256"]
    assert value["target_input_manifest"] == {
        "before_after_bit_exact": True,
        "file_count": preparation.MATCHED_ROWS,
        "sha256": value["target_input_manifest"]["sha256"],
    }
    assert len(value["finite_mutation_manifest_sha256"]) == 64
    assert len(value["source_side_manifest"]["sha256"]) == 64
    future_isolation.validate_evidence(
        value,
        prediction_freeze_record=records.file_record(freeze),
        protocol_sha256=PROTOCOL_SHA256,
    )

    mutations: list[tuple[str, object]] = []
    extra = deepcopy(value)
    extra["extra"] = True
    mutations.append(("schema", extra))
    import_guard = deepcopy(value)
    import_guard["import_guard"]["all_absent"] = False
    mutations.append(("import guard", import_guard))
    target_manifest = deepcopy(value)
    target_manifest["target_input_manifest"]["before_after_bit_exact"] = False
    mutations.append(("target manifest", target_manifest))
    grammar = deepcopy(value)
    grammar["mutation_grammar"]["byte_lsb"]["operation"] = "no-op"
    mutations.append(("mutation grammar", grammar))
    malformed_digest = deepcopy(value)
    malformed_digest["finite_mutation_manifest_sha256"] = "not-a-digest"
    mutations.append(("digest grammar", malformed_digest))
    source_digest = deepcopy(value)
    source_digest["source_side_manifest"]["sha256"] = "e" * 64
    mutations.append(("source-side manifest", source_digest))
    for message, mutated in mutations:
        with pytest.raises(future_isolation.FutureIsolationError, match=message):
            future_isolation.validate_evidence(
                mutated,
                prediction_freeze_record=records.file_record(freeze),
                protocol_sha256=PROTOCOL_SHA256,
            )


def test_freeze_chain_mutation_fails_before_output(
    tmp_path: Path,
    isolation_roots: dict[str, Path],
) -> None:
    freeze, target_root, prediction_root = _build_formal_census(tmp_path, include_targets=False)
    prediction = prediction_root / "000000" / future_isolation.PREDICTION_FILE
    os.chmod(prediction, 0o644)
    output_root = tmp_path / "future-output"
    output_root.mkdir()
    output = output_root / "future-isolation.json"
    launcher = isolation_roots["custody"] / runtime.FUTURE_ISOLATION_LAUNCHER_RELATIVE
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(launcher),
            str(isolation_roots["custody"]),
            str(isolation_roots["dependency"]),
            str(isolation_roots["stdlib"]),
            str(freeze),
            str(target_root),
            str(prediction_root),
            str(output),
        ),
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", "")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "immutable mode 0444" in result.stderr
    assert not output.exists()
