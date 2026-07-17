from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_causal_diagnostics import predictor, preparation, records, worker


def _replace_commit(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        os.chmod(path, 0o644)
        path.unlink()
    records.write_immutable_json_exclusive(path, cast(records.JsonValue, value))


def _rehash_commit(value: dict[str, Any], protocol_sha256: str) -> None:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = records.canonical_json_sha256(
        cast(records.JsonValue, value),
        protocol_sha256=protocol_sha256,
    )


def _source_row(tmp_path: Path) -> Path:
    ids, times = preparation.expected_raw_identities()
    rows = preparation.build_row_index(ids, times)
    values = np.arange(
        preparation.RAW_ROWS * preparation.NATIVE_SIZE * preparation.NATIVE_SIZE * preparation.CHANNELS,
        dtype=np.uint32,
    )
    frames = np.ascontiguousarray(
        ((37 * values + values // 97) % 251)
        .astype(np.uint8)
        .reshape(
            preparation.RAW_ROWS,
            preparation.NATIVE_SIZE,
            preparation.NATIVE_SIZE,
            preparation.CHANNELS,
        )
    )
    normalizers = preparation.fit_fold_normalizers(frames, rows)
    derangement = preparation.half_cycle_derangement(rows)
    source = preparation.construct_source_row(frames, 0, rows, normalizers, derangement)
    path = tmp_path / "source.npz"
    preparation.write_source_row_npz(path, source)
    return path


def _config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    value: dict[str, object] = {
        "config_sha256": "2" * 64,
        "prediction_roles": list(worker.PREDICTION_ROLES),
        "protocol_sha256": records.PROTOCOL_SHA256,
        "schema_version": "mm009-worker-config-v1",
    }
    path = tmp_path / "config.json"
    records.write_immutable_json_exclusive(path, value)  # type: ignore[arg-type]
    return path, value


def test_worker_emits_compact_future_blind_bundle_and_detects_mutation(tmp_path: Path) -> None:
    source_path = _source_row(tmp_path)
    config_path, config = _config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    commit = worker.run_source_row(source_path, config_path, output)

    assert "future" not in repr(commit).lower()
    assert commit["row_ordinal"] == 0
    predictions = worker.load_prediction_array(output / worker.PREDICTION_FILE)
    assert predictions.shape == worker.PREDICTION_SHAPE
    assert np.all(np.isfinite(predictions))
    source = preparation.load_source_row_npz(source_path)
    worker.validate_worker_output(output, config, source, source_path=source_path)
    role_index = {role: index for index, role in enumerate(worker.PREDICTION_ROLES)}
    reverse = commit["reverse"]
    assert isinstance(reverse, dict)
    control_application = reverse["control_application"]
    assert isinstance(control_application, dict)
    assert control_application["semantics"] == ("fit_current_to_previous_then_apply_frozen_fit_to_actual_current")
    for arm in ("affine", "appearance", "combined"):
        arm_record = reverse[arm]
        assert isinstance(arm_record, dict)
        expected = predictor.apply_operator_once(
            source["current"],
            np.asarray(arm_record["parameters"], dtype="<f8"),
            np.asarray(arm_record["gains"], dtype="<f8"),
            np.asarray(arm_record["biases"], dtype="<f8"),
        )
        assert np.array_equal(predictions[role_index[f"forecast_reverse_{arm}"]], expected)

    commit_path = output / worker.COMMIT_FILE

    # The outer digest detects arbitrary nested corruption before semantic parsing.
    corrupted = cast(dict[str, Any], copy.deepcopy(commit))
    corrupted["ordered"]["affine"]["history_fit"]["certificate"]["scalar_replay_bit_exact"] = False
    _replace_commit(commit_path, corrupted)
    with pytest.raises(worker.WorkerError, match="evidence digest differs"):
        worker.validate_worker_output(output, config, source, source_path=source_path)

    # Recompute the outer digest to exercise every strict nested validator rather
    # than allowing the aggregate mutation seal to mask a weak inner contract.
    mutations: list[tuple[str, Any]] = [
        (
            "nested selected parameter",
            lambda value: value["ordered"]["affine"]["history_fit"]["parameters"].__setitem__(0, 1.0),
        ),
        (
            "top-level selected gain consistency",
            lambda value: value["ordered"]["appearance"]["gains"].__setitem__(0, 0.125),
        ),
        (
            "certificate boolean",
            lambda value: value["ordered"]["combined"]["history_fit"]["certificate"].__setitem__(
                "scalar_replay_bit_exact", False
            ),
        ),
        (
            "global context metadata",
            lambda value: value["ordered"]["affine"]["history_fit"].__setitem__("context_key", "wrong"),
        ),
        (
            "checker parity context",
            lambda value: value["history"]["arms"]["combined"]["output_parity_fits"][0].__setitem__(
                "context_key", "mm009/source/history/checkerboard/output-1/combined"
            ),
        ),
        (
            "checker appearance digest",
            lambda value: value["history"]["arms"]["appearance"]["output_parity_fits"][1]["hashes"].__setitem__(
                "appearance_gains", "0" * 64
            ),
        ),
        (
            "checker bias parity value",
            lambda value: value["history"]["bias_only"]["output_parity_fits"][0]["first_biases"].__setitem__(
                0,
                value["history"]["bias_only"]["output_parity_fits"][0]["first_biases"][0] + 0.125,
            ),
        ),
        (
            "full bias-only value",
            lambda value: value["bias_only"]["history_fit"]["biases"].__setitem__(
                0, value["bias_only"]["history_fit"]["biases"][0] + 0.125
            ),
        ),
        (
            "nested extra key",
            lambda value: value["ordered"]["appearance"]["history_fit"].__setitem__("unexpected", True),
        ),
        (
            "nested missing key",
            lambda value: value["ordered"]["combined"]["history_fit"]["certificate"].pop("geometry_sha256"),
        ),
        (
            "bounded claim",
            lambda value: value["bounded"].__setitem__("combined", not value["bounded"]["combined"]),
        ),
        (
            "reverse hash membership",
            lambda value: value["reverse"]["control_application"]["forecast_sha256"].__setitem__(
                "unexpected", "0" * 64
            ),
        ),
    ]
    for label, mutate in mutations:
        mutated = copy.deepcopy(commit)
        mutate(mutated)
        _rehash_commit(mutated, cast(str, config["protocol_sha256"]))
        _replace_commit(commit_path, mutated)
        try:
            worker.validate_worker_output(output, config, source, source_path=source_path)
        except worker.WorkerError:
            continue
        pytest.fail(f"nested evidence mutation unexpectedly passed validation: {label}")

    _replace_commit(commit_path, cast(dict[str, Any], commit))
    worker.validate_worker_output(output, config, source, source_path=source_path)

    prediction_path = output / worker.PREDICTION_FILE
    os.chmod(prediction_path, 0o644)
    with prediction_path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(worker.WorkerError, match="hash differs"):
        worker.validate_worker_output(output, config, source, source_path=source_path)
