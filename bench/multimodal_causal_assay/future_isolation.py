"""Fitting-free future-mutation isolation gate for frozen MM-011 predictions.

This module is intended to run in a fresh process after ``prediction-freeze.json``
exists and before target scoring starts.  It imports only target-custody primitives,
never imports a predictor or fitter, and never writes a mutated target.  Every
mutation is made in memory and passed through the exact detached-target validator.
The complete prediction-side file census is hashed before and after the sweep.
"""

from __future__ import annotations

import json
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

import numpy as np

from bench.multimodal_causal_assay import preparation, records

SCHEMA_VERSION: Final = "mm011-future-isolation-v1"
RANDOM_SEED_BASE: Final = 991_000
TARGET_FILE: Final = "target.npz"
PREDICTION_FILE: Final = "predictions.npy"
WORKER_EVIDENCE_FILE: Final = "worker-evidence.json"
COMMIT_FILE: Final = "commit.json"
SOURCE_SIDE_FILES_PER_ROW: Final = 3

# The launcher checks this list both before importing this module and after the gate
# finishes.  Keeping the same check here makes direct CLI use fail closed as well.
FORBIDDEN_MODULES: Final = (
    "bench.multimodal_causal_assay.predictor",
    "bench.multimodal_causal_assay.worker",
    "bench.multimodal_mechanism_diagnostics.fitting_v22",
    "bench.multimodal_mechanism_diagnostics.geometry_v22",
    "bench.multimodal_mechanism_diagnostics.global_v22",
    "bench.multimodal_mechanism_diagnostics.nongrid_v22",
)


class FutureIsolationError(ValueError):
    """Raised when target isolation or immutable source custody fails closed."""


def assert_fitting_modules_absent() -> None:
    loaded = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    if loaded:
        raise FutureIsolationError(f"future-isolation process loaded forbidden modules: {loaded}")


def _real_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise FutureIsolationError(f"{label} directory is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FutureIsolationError(f"{label} must be a real non-symlink directory")
    return candidate


def _require_census(directory: Path, expected: set[str], label: str) -> None:
    actual = {entry.name for entry in directory.iterdir()}
    if actual != expected:
        raise FutureIsolationError(f"{label} membership differs: expected={sorted(expected)}, actual={sorted(actual)}")


def _immutable_file_record(path: Path, label: str) -> dict[str, records.JsonValue]:
    try:
        record = records.file_record(path)
    except (OSError, records.RecordValidationError) as error:
        raise FutureIsolationError(f"{label} must be a unique regular file") from error
    if record["mode"] != 0o444:
        raise FutureIsolationError(f"{label} must have immutable mode 0444")
    return record


def _load_canonical_json(path: Path, label: str) -> dict[str, object]:
    _immutable_file_record(path, label)

    def reject_constant(value: str) -> None:
        raise FutureIsolationError(f"{label} contains nonfinite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise FutureIsolationError(f"{label} contains duplicate JSON key {key}")
            output[key] = value
        return output

    try:
        payload = records.read_regular_bytes(path, maximum_bytes=100_000_000)
        value = json.loads(
            payload.decode("ascii"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FutureIsolationError(f"{label} is not strict ASCII JSON") from error
    if type(value) is not dict:
        raise FutureIsolationError(f"{label} must be a JSON object")
    checked = cast(dict[str, object], value)
    try:
        if payload != records.json_file_bytes(cast(records.JsonValue, checked)):
            raise FutureIsolationError(f"{label} is not canonical JSON bytes")
    except records.RecordValidationError as error:
        raise FutureIsolationError(f"{label} is outside canonical JSON") from error
    return checked


def _validate_freeze(path: Path) -> tuple[dict[str, object], str]:
    freeze = _load_canonical_json(path, "prediction freeze")
    expected = {
        "chain",
        "config_sha256",
        "genesis_sha256",
        "last_commit_sha256",
        "protocol_sha256",
        "row_count",
        "schema_version",
        "status",
    }
    if (
        set(freeze) != expected
        or freeze.get("schema_version") != "mm011-prediction-freeze-v1"
        or freeze.get("status") != "all_predictions_frozen_before_target_scoring"
        or freeze.get("row_count") != preparation.MATCHED_ROWS
    ):
        raise FutureIsolationError("prediction freeze schema or lifecycle state differs")
    try:
        protocol_sha256 = records.require_sha256(freeze.get("protocol_sha256"), "protocol SHA-256")
        records.require_sha256(freeze.get("config_sha256"), "config SHA-256")
        records.require_sha256(freeze.get("genesis_sha256"), "prediction genesis SHA-256")
        records.require_sha256(freeze.get("last_commit_sha256"), "last prediction commit SHA-256")
    except records.RecordValidationError as error:
        raise FutureIsolationError("prediction freeze hash binding differs") from error
    if protocol_sha256 != records.PROTOCOL_SHA256 or protocol_sha256 != preparation.PROTOCOL_SHA256:
        raise FutureIsolationError("prediction freeze protocol binding differs")
    chain = freeze.get("chain")
    if type(chain) is not list or len(chain) != preparation.MATCHED_ROWS:
        raise FutureIsolationError("prediction freeze chain census differs")
    return freeze, protocol_sha256


def _source_side_snapshot(
    freeze_path: Path,
    prediction_root: Path,
    freeze: Mapping[str, object],
) -> dict[str, records.JsonValue]:
    expected_rows = {f"{ordinal:06d}" for ordinal in range(preparation.MATCHED_ROWS)}
    _require_census(prediction_root, expected_rows, "prediction root")
    chain = cast(list[object], freeze["chain"])
    entries: list[records.JsonValue] = [
        {
            "path": "prediction-freeze.json",
            "record": _immutable_file_record(freeze_path, "prediction freeze"),
        }
    ]
    for ordinal in range(preparation.MATCHED_ROWS):
        row_name = f"{ordinal:06d}"
        row_directory = _real_directory(prediction_root / row_name, f"prediction row {ordinal}")
        _require_census(
            row_directory,
            {PREDICTION_FILE, WORKER_EVIDENCE_FILE, COMMIT_FILE},
            f"prediction row {ordinal}",
        )
        row_records = {
            name: _immutable_file_record(row_directory / name, f"prediction row {ordinal}/{name}")
            for name in (PREDICTION_FILE, WORKER_EVIDENCE_FILE, COMMIT_FILE)
        }
        chain_item = chain[ordinal]
        if (
            type(chain_item) is not dict
            or set(chain_item)
            != {
                "commit_sha256",
                "ordinal",
                "prediction_sha256",
                "worker_evidence_sha256",
            }
            or chain_item.get("ordinal") != ordinal
            or chain_item.get("commit_sha256") != row_records[COMMIT_FILE]["sha256"]
            or chain_item.get("prediction_sha256") != row_records[PREDICTION_FILE]["sha256"]
            or chain_item.get("worker_evidence_sha256") != row_records[WORKER_EVIDENCE_FILE]["sha256"]
        ):
            raise FutureIsolationError(f"prediction freeze chain differs at row {ordinal}")
        for name in (PREDICTION_FILE, WORKER_EVIDENCE_FILE, COMMIT_FILE):
            entries.append(
                {
                    "path": f"predictions/{row_name}/{name}",
                    "record": row_records[name],
                }
            )
    if cast(dict[str, object], chain[-1]).get("commit_sha256") != freeze["last_commit_sha256"]:
        raise FutureIsolationError("prediction freeze last-commit binding differs")
    return {
        "entries": entries,
        "file_count": 1 + SOURCE_SIDE_FILES_PER_ROW * preparation.MATCHED_ROWS,
        "schema_version": "mm011-future-isolation-source-snapshot-v1",
    }


def _target_snapshot(target_root: Path) -> dict[str, records.JsonValue]:
    expected_rows = {f"{ordinal:06d}" for ordinal in range(preparation.MATCHED_ROWS)}
    _require_census(target_root, expected_rows, "target root")
    entries: list[records.JsonValue] = []
    for ordinal in range(preparation.MATCHED_ROWS):
        row_name = f"{ordinal:06d}"
        row_directory = _real_directory(target_root / row_name, f"target row {ordinal}")
        _require_census(row_directory, {TARGET_FILE}, f"target row {ordinal}")
        entries.append(
            {
                "path": f"targets/{row_name}/{TARGET_FILE}",
                "record": _immutable_file_record(row_directory / TARGET_FILE, f"target row {ordinal}"),
            }
        )
    return {
        "entries": entries,
        "file_count": preparation.MATCHED_ROWS,
        "schema_version": "mm011-future-isolation-target-snapshot-v1",
    }


def random_finite_replacement(shape: tuple[int, ...], ordinal: int) -> np.ndarray:
    if shape != (preparation.CHANNELS, preparation.NATIVE_SIZE, preparation.NATIVE_SIZE):
        raise FutureIsolationError("random replacement shape differs from the target grammar")
    if type(ordinal) is not int or ordinal not in range(preparation.MATCHED_ROWS):
        raise FutureIsolationError("random replacement ordinal is outside the frozen panel")
    generator = np.random.Generator(np.random.PCG64(RANDOM_SEED_BASE + ordinal))
    value = np.ascontiguousarray(generator.standard_normal(shape, dtype=np.float64), dtype="<f8")
    if not np.all(np.isfinite(value)):
        raise FutureIsolationError("PCG64 random replacement produced a nonfinite value")
    return cast(np.ndarray, value)


def spatial_reverse(future: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.ascontiguousarray(future[:, ::-1, ::-1], dtype="<f8"))


def flip_first_central_lsb(future: np.ndarray) -> np.ndarray:
    value = np.array(future, dtype="<f8", order="C", copy=True)
    bits = value.view("<u8")
    original_bits = int(bits[0, 8, 8])
    bits[0, 8, 8] = np.uint64(original_bits ^ 1)
    if (
        int(bits[0, 8, 8]) == original_bits
        or not np.isfinite(value[0, 8, 8])
        or bool(value[0, 8, 8] == future[0, 8, 8])
    ):
        raise FutureIsolationError("uint64 LSB mutation was not a finite value change")
    return cast(np.ndarray, value)


def _replace_future(target: Mapping[str, np.ndarray], replacement: np.ndarray) -> Mapping[str, np.ndarray]:
    candidate = {name: np.ascontiguousarray(value.copy()) for name, value in target.items()}
    candidate["future"] = np.ascontiguousarray(replacement, dtype="<f8")
    return cast(Mapping[str, np.ndarray], preparation.validate_target_row_arrays(candidate))


def _mutation_sweep(target_root: Path, *, protocol_sha256: str) -> tuple[dict[str, int], str]:
    counts = {
        "byte_lsb_finite_valid": 0,
        "deranged_future_valid": 0,
        "nan_rejected": 0,
        "random_finite_valid": 0,
        "rows": 0,
        "spatial_reverse_valid": 0,
    }
    finite_manifest: list[records.JsonValue] = []
    for ordinal in range(preparation.MATCHED_ROWS):
        path = target_root / f"{ordinal:06d}" / TARGET_FILE
        try:
            target = preparation.load_target_row_npz(path)
        except (OSError, records.RecordValidationError, preparation.PreparationValidationError) as error:
            raise FutureIsolationError(f"detached target validation failed at row {ordinal}") from error
        if records.read_regular_bytes(path, maximum_bytes=100_000_000) != preparation.target_row_npz_bytes(target):
            raise FutureIsolationError(f"detached target is not canonical NPZ bytes at row {ordinal}")
        if int(target["row_ordinal"][0]) != ordinal:
            raise FutureIsolationError(f"detached target ordinal differs at row {ordinal}")
        future = target["future"]
        replacements = (
            (
                "random_finite",
                "random_finite_valid",
                random_finite_replacement(future.shape, ordinal),
            ),
            ("spatial_reverse", "spatial_reverse_valid", spatial_reverse(future)),
            (
                "deranged_future",
                "deranged_future_valid",
                np.ascontiguousarray(target["deranged_future"].copy(), dtype="<f8"),
            ),
            ("byte_lsb", "byte_lsb_finite_valid", flip_first_central_lsb(future)),
        )
        for mutation, count_key, replacement in replacements:
            _replace_future(target, replacement)
            finite_manifest.append(
                {
                    "mutation": mutation,
                    "row_ordinal": ordinal,
                    "sha256": records.scientific_array_sha256(
                        f"future-isolation:{mutation}:{ordinal}",
                        replacement,
                        protocol_sha256=protocol_sha256,
                    ),
                }
            )
            counts[count_key] += 1

        nan_future = np.array(future, dtype="<f8", order="C", copy=True)
        nan_future[0, 8, 8] = np.nan
        try:
            _replace_future(target, nan_future)
        except (records.RecordValidationError, preparation.PreparationValidationError):
            counts["nan_rejected"] += 1
        else:
            raise FutureIsolationError(f"NaN target mutation was accepted at row {ordinal}")
        counts["rows"] += 1
    expected = preparation.MATCHED_ROWS
    if counts != {
        "byte_lsb_finite_valid": expected,
        "deranged_future_valid": expected,
        "nan_rejected": expected,
        "random_finite_valid": expected,
        "rows": expected,
        "spatial_reverse_valid": expected,
    }:
        raise FutureIsolationError("future mutation census differs")
    digest = records.canonical_json_sha256(finite_manifest, protocol_sha256=protocol_sha256)
    return counts, digest


def run_future_isolation(
    prediction_freeze_path: str | Path,
    target_root: str | Path,
    prediction_root: str | Path,
    output_path: str | Path,
) -> dict[str, records.JsonValue]:
    """Run and seal the exact 453-row post-freeze future-mutation gate."""

    assert_fitting_modules_absent()
    freeze_path = Path(prediction_freeze_path)
    targets = _real_directory(target_root, "target root")
    predictions = _real_directory(prediction_root, "prediction root")
    freeze, protocol_sha256 = _validate_freeze(freeze_path)

    source_before = _source_side_snapshot(freeze_path, predictions, freeze)
    target_before = _target_snapshot(targets)
    source_before_sha256 = records.canonical_json_sha256(source_before, protocol_sha256=protocol_sha256)
    target_before_sha256 = records.canonical_json_sha256(target_before, protocol_sha256=protocol_sha256)
    counts, finite_manifest_sha256 = _mutation_sweep(targets, protocol_sha256=protocol_sha256)
    source_after = _source_side_snapshot(freeze_path, predictions, freeze)
    target_after = _target_snapshot(targets)
    if source_after != source_before:
        raise FutureIsolationError("source-side prediction files changed during future mutations")
    if target_after != target_before:
        raise FutureIsolationError("detached target files changed during in-memory future mutations")
    if records.canonical_json_sha256(source_after, protocol_sha256=protocol_sha256) != source_before_sha256:
        raise FutureIsolationError("source-side aggregate manifest hash changed")
    if records.canonical_json_sha256(target_after, protocol_sha256=protocol_sha256) != target_before_sha256:
        raise FutureIsolationError("target aggregate manifest hash changed")
    assert_fitting_modules_absent()
    count_record: dict[str, records.JsonValue] = {name: value for name, value in counts.items()}

    output: dict[str, records.JsonValue] = {
        "finite_mutation_manifest_sha256": finite_manifest_sha256,
        "import_guard": {
            "all_absent": True,
            "forbidden_modules": list(FORBIDDEN_MODULES),
        },
        "mutation_counts": count_record,
        "mutation_grammar": {
            "byte_lsb": {
                "array_view": "little_endian_uint64",
                "index_chw": [0, 8, 8],
                "operation": "xor_1",
                "requires_finite_and_numeric_change": True,
            },
            "deranged_future": {
                "operation": "replace_future_with_existing_deranged_future",
            },
            "nan_sentinel": {
                "index_chw": [0, 8, 8],
                "operation": "replace_with_quiet_NaN",
                "required_result": "target_validation_rejection",
            },
            "random_finite": {
                "bit_generator": "numpy.random.PCG64",
                "distribution": "standard_normal_float64",
                "seed_formula": "991000 + row_ordinal",
            },
            "spatial_reverse": {
                "axes_chw": [1, 2],
                "operation": "reverse_both_height_and_width",
            },
        },
        "prediction_freeze": _immutable_file_record(freeze_path, "prediction freeze"),
        "protocol_sha256": protocol_sha256,
        "schema_version": SCHEMA_VERSION,
        "source_side_aggregate_manifest_sha256": source_before_sha256,
        "source_side_manifest": {
            "before_after_bit_exact": True,
            "file_count": 1 + SOURCE_SIDE_FILES_PER_ROW * preparation.MATCHED_ROWS,
            "sha256": source_before_sha256,
        },
        "status": "passed_before_target_scoring",
        "target_input_manifest": {
            "before_after_bit_exact": True,
            "file_count": preparation.MATCHED_ROWS,
            "sha256": target_before_sha256,
        },
    }
    records.write_immutable_json_exclusive(output_path, output)
    return output


def validate_evidence(
    value: object,
    *,
    prediction_freeze_record: Mapping[str, records.JsonValue],
    protocol_sha256: str,
) -> dict[str, records.JsonValue]:
    """Validate every stored future-isolation claim before scoring is licensed."""

    expected_keys = {
        "finite_mutation_manifest_sha256",
        "import_guard",
        "mutation_counts",
        "mutation_grammar",
        "prediction_freeze",
        "protocol_sha256",
        "schema_version",
        "source_side_aggregate_manifest_sha256",
        "source_side_manifest",
        "status",
        "target_input_manifest",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise FutureIsolationError("future-isolation evidence schema differs")
    evidence = cast(dict[str, object], value)
    if (
        evidence["schema_version"] != SCHEMA_VERSION
        or evidence["status"] != "passed_before_target_scoring"
        or evidence["protocol_sha256"] != protocol_sha256
        or evidence["prediction_freeze"] != prediction_freeze_record
    ):
        raise FutureIsolationError("future-isolation evidence lifecycle binding differs")
    try:
        records.require_sha256(protocol_sha256, "protocol SHA-256")
        records.require_sha256(evidence["finite_mutation_manifest_sha256"], "finite mutation manifest SHA-256")
        source_digest = records.require_sha256(
            evidence["source_side_aggregate_manifest_sha256"],
            "source-side aggregate manifest SHA-256",
        )
    except records.RecordValidationError as error:
        raise FutureIsolationError("future-isolation evidence digest grammar differs") from error

    expected_count = preparation.MATCHED_ROWS
    if evidence["mutation_counts"] != {
        "byte_lsb_finite_valid": expected_count,
        "deranged_future_valid": expected_count,
        "nan_rejected": expected_count,
        "random_finite_valid": expected_count,
        "rows": expected_count,
        "spatial_reverse_valid": expected_count,
    }:
        raise FutureIsolationError("future-isolation mutation census differs")
    if evidence["import_guard"] != {
        "all_absent": True,
        "forbidden_modules": list(FORBIDDEN_MODULES),
    }:
        raise FutureIsolationError("future-isolation import guard differs")
    if evidence["mutation_grammar"] != {
        "byte_lsb": {
            "array_view": "little_endian_uint64",
            "index_chw": [0, 8, 8],
            "operation": "xor_1",
            "requires_finite_and_numeric_change": True,
        },
        "deranged_future": {
            "operation": "replace_future_with_existing_deranged_future",
        },
        "nan_sentinel": {
            "index_chw": [0, 8, 8],
            "operation": "replace_with_quiet_NaN",
            "required_result": "target_validation_rejection",
        },
        "random_finite": {
            "bit_generator": "numpy.random.PCG64",
            "distribution": "standard_normal_float64",
            "seed_formula": "991000 + row_ordinal",
        },
        "spatial_reverse": {
            "axes_chw": [1, 2],
            "operation": "reverse_both_height_and_width",
        },
    }:
        raise FutureIsolationError("future-isolation mutation grammar differs")
    expected_source_manifest = {
        "before_after_bit_exact": True,
        "file_count": 1 + SOURCE_SIDE_FILES_PER_ROW * expected_count,
        "sha256": source_digest,
    }
    if evidence["source_side_manifest"] != expected_source_manifest:
        raise FutureIsolationError("future-isolation source-side manifest differs")
    target_manifest = evidence["target_input_manifest"]
    if type(target_manifest) is not dict or set(target_manifest) != {
        "before_after_bit_exact",
        "file_count",
        "sha256",
    }:
        raise FutureIsolationError("future-isolation target manifest schema differs")
    target = cast(dict[str, object], target_manifest)
    try:
        target_digest = records.require_sha256(target["sha256"], "target aggregate manifest SHA-256")
    except records.RecordValidationError as error:
        raise FutureIsolationError("future-isolation target digest grammar differs") from error
    if target != {
        "before_after_bit_exact": True,
        "file_count": expected_count,
        "sha256": target_digest,
    }:
        raise FutureIsolationError("future-isolation target manifest differs")
    return cast(dict[str, records.JsonValue], evidence)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_freeze")
    parser.add_argument("target_root")
    parser.add_argument("prediction_root")
    parser.add_argument("output")
    arguments = parser.parse_args(argv)
    run_future_isolation(
        arguments.prediction_freeze,
        arguments.target_root,
        arguments.prediction_root,
        arguments.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FORBIDDEN_MODULES",
    "FutureIsolationError",
    "RANDOM_SEED_BASE",
    "SCHEMA_VERSION",
    "assert_fitting_modules_absent",
    "flip_first_central_lsb",
    "main",
    "random_finite_replacement",
    "run_future_isolation",
    "spatial_reverse",
    "validate_evidence",
]
