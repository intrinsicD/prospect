from __future__ import annotations

import hashlib
import io
import os
import signal
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bench.multimodal_causal_diagnostics import (
    experiment,
    future_isolation,
    future_isolation_launcher,
    preparation,
    records,
    score_launcher,
    synthetic_controls,
    worker,
)


def _immutable_npy(path: Path, value: np.ndarray) -> None:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    records.write_immutable_bytes_exclusive(path, stream.getvalue())


def _json_object(value: records.JsonValue) -> dict[str, records.JsonValue]:
    assert type(value) is dict
    return cast(dict[str, records.JsonValue], value)


def _json_list(value: records.JsonValue) -> list[records.JsonValue]:
    assert type(value) is list
    return cast(list[records.JsonValue], value)


def _install_interrupting_executor(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, bool]]:
    shutdown_calls: list[tuple[bool, bool]] = []

    class RecordingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == experiment.MAX_WORKERS

        def submit(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("empty interruption fixture unexpectedly submitted work")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def interrupt_collection(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(experiment.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(experiment.concurrent.futures, "as_completed", interrupt_collection)
    return shutdown_calls


def _wait_for_path(path: Path, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return path.exists()


def _kill_escaped_fixture(pid_file: Path, token: Path) -> None:
    """Best-effort cleanup without risking a recycled, unrelated PID."""

    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="ascii"))
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return
    if str(token).encode("utf-8") not in command:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _control_evidence_fixture(config_sha256: str, protocol_sha256: str) -> dict[str, records.JsonValue]:
    arms = ("affine", "appearance", "combined")
    positive_arms = {
        "translation": "affine",
        "affine": "affine",
        "appearance": "appearance",
        "combined": "combined",
    }
    positive: list[records.JsonValue] = []
    for scenario, arm in positive_arms.items():
        for seed in synthetic_controls.SEEDS:
            errors: dict[str, records.JsonValue] = {
                "i": 100.0,
                "a": 1.0,
                "q": 100.0,
                "u": 100.0,
                "p": 100.0,
                "c": 1.0,
                "h": 100.0,
                "r": 100.0,
                "z": 100.0,
                "d": 100.0,
                "pd": 100.0,
                "b": 100.0,
                "bd": 100.0,
            }
            positive.append(
                {
                    "arm": arm,
                    "complete_predicates": {
                        "directional": True,
                        "future": True,
                        "history": True,
                        "joint": True,
                    },
                    "errors": errors,
                    "forecast_sse": 1.0,
                    "operator": {
                        "biases": [0.0, 0.0, 0.0],
                        "forecast_sha256": "a" * 64,
                        "gains": [1.0, 1.0, 1.0],
                        "history_sha256": "b" * 64,
                        "parameters": [0.0] * 6,
                    },
                    "persistence_sse": 100.0,
                    "scenario": scenario,
                    "seed": seed,
                }
            )
    reversal: list[records.JsonValue] = [
        {
            "arm": positive_arms[scenario],
            "persistence_sse": 1.0,
            "repeated_sse": 2.0,
            "scenario": f"{scenario}_reversal",
            "seed": seed,
        }
        for scenario in ("affine", "appearance")
        for seed in synthetic_controls.SEEDS
    ]
    stationary: list[records.JsonValue] = [
        {
            "arm_sse": {arm: 0.0 for arm in arms},
            "eligible": False,
            "persistence_sse": 0.0,
            "scenario": "stationary",
            "seed": seed,
        }
        for seed in synthetic_controls.SEEDS
    ]
    independent: list[records.JsonValue] = [
        {
            "distinct_future_fixture": True,
            "forecast_sse": {arm: 10.0 for arm in arms},
            "future_bias_sse": 10.0,
            "future_seed": future_seed,
            "history_bias_sse": 10.0,
            "history_identity_sse": 10.0,
            "history_xfit_sse": {arm: 10.0 for arm in arms},
            "persistence_sse": 10.0,
            "ratios_to_persistence": {arm: 1.0 for arm in arms},
            "scenario": "independent",
            "seed": seed,
        }
        for seed, future_seed in zip(
            synthetic_controls.SEEDS,
            synthetic_controls.INDEPENDENT_FUTURE_SEEDS,
            strict=True,
        )
    ]
    body: dict[str, records.JsonValue] = {
        "boundary": [
            {
                "bounded": {"affine": True, "combined": True},
                "scenario": "coupled_boundary",
                "seed": seed,
            }
            for seed in synthetic_controls.SEEDS
        ],
        "channel_permutation_exact": True,
        "config_sha256": config_sha256,
        "constant_target": [
            {
                "bias_match_sse": {arm: 0.0 for arm in arms},
                "scenario": "constant_target",
                "seed": seed,
            }
            for seed in synthetic_controls.SEEDS
        ],
        "independent": independent,
        "independent_aggregate_branches": {
            arm: {
                "expected_branch": "tested_family_identifiability_failure",
                "future_source_credit": False,
                "historical_source_credit": False,
                "joint_support": False,
            }
            for arm in arms
        },
        "positive": positive,
        "protocol_sha256": protocol_sha256,
        "reserved_seed_overlap": False,
        "reversal": reversal,
        "schema_version": synthetic_controls.SCHEMA_VERSION,
        "seeds": list(synthetic_controls.ALL_SEEDS),
        "stationary": stationary,
    }
    return {
        **body,
        "evidence_sha256": records.canonical_json_sha256(body, protocol_sha256=protocol_sha256),
    }


def _future_evidence_fixture(
    *,
    prediction_freeze_record: dict[str, records.JsonValue],
    protocol_sha256: str,
    source_manifest_sha256: str,
    target_manifest_sha256: str,
    finite_manifest_sha256: str,
) -> dict[str, records.JsonValue]:
    return {
        "finite_mutation_manifest_sha256": finite_manifest_sha256,
        "import_guard": {
            "all_absent": True,
            "forbidden_modules": list(future_isolation.FORBIDDEN_MODULES),
        },
        "mutation_counts": {
            "byte_lsb_finite_valid": preparation.MATCHED_ROWS,
            "deranged_future_valid": preparation.MATCHED_ROWS,
            "nan_rejected": preparation.MATCHED_ROWS,
            "random_finite_valid": preparation.MATCHED_ROWS,
            "rows": preparation.MATCHED_ROWS,
            "spatial_reverse_valid": preparation.MATCHED_ROWS,
        },
        "mutation_grammar": {
            "byte_lsb": {
                "array_view": "little_endian_uint64",
                "index_chw": [0, 8, 8],
                "operation": "xor_1",
                "requires_finite_and_numeric_change": True,
            },
            "deranged_future": {"operation": "replace_future_with_existing_deranged_future"},
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
        "prediction_freeze": prediction_freeze_record,
        "protocol_sha256": protocol_sha256,
        "schema_version": future_isolation.SCHEMA_VERSION,
        "source_side_aggregate_manifest_sha256": source_manifest_sha256,
        "source_side_manifest": {
            "before_after_bit_exact": True,
            "file_count": 1 + 3 * preparation.MATCHED_ROWS,
            "sha256": source_manifest_sha256,
        },
        "status": "passed_before_target_scoring",
        "target_input_manifest": {
            "before_after_bit_exact": True,
            "file_count": preparation.MATCHED_ROWS,
            "sha256": target_manifest_sha256,
        },
    }


def test_scientific_source_census_and_audit_template_are_self_bound() -> None:
    paths = experiment._scientific_source_paths()  # noqa: SLF001 - lifecycle contract test
    assert paths == tuple(sorted(set(paths), key=str))
    assert Path("bench/multimodal_causal_diagnostics/experiment.py") in paths
    assert Path("bench/multimodal_causal_diagnostics/score_worker.py") in paths
    assert Path("bench/multimodal_causal_diagnostics/future_isolation.py") in paths
    assert Path("bench/multimodal_mechanism_diagnostics/synthetic_v22.py") in paths
    assert Path("bench/multimodal_mechanism_diagnostics/calibration_v22.py") in paths
    assert set(experiment.MM007_VERIFIER_SOURCES).issubset(paths)
    assert experiment.PROTOCOL_DOC.relative_to(experiment.REPO_ROOT) in paths

    source_hashes = experiment._source_hashes()  # noqa: SLF001 - lifecycle contract test
    assert set(source_hashes) == {str(path) for path in paths}
    assert all(len(value) == 64 for value in source_hashes.values())
    assert {name: source_hashes[name] for name in experiment.V22_SOURCE_PINS} == experiment.V22_SOURCE_PINS
    assert {
        name: source_hashes[name] for name in experiment.V22_SYNTHETIC_SOURCE_PINS
    } == experiment.V22_SYNTHETIC_SOURCE_PINS

    audit = experiment.audit_template()
    assert set(audit) == {
        "claim_boundary",
        "config_sha256",
        "decision",
        "experiment_id",
        "findings_closed",
        "protocol_sha256",
        "reviewed_source_hashes",
        "reviewed_source_hashes_sha256",
        "reviewed_runtime_source_manifest_sha256",
        "reviewer",
        "schema_version",
    }
    assert audit["decision"] == "GO"
    assert audit["findings_closed"] == []
    assert audit["reviewer"] == "prospect-results-audit"
    assert audit["reviewed_source_hashes_sha256"] == records.canonical_json_sha256(
        audit["reviewed_source_hashes"],  # type: ignore[arg-type]
        protocol_sha256=str(audit["protocol_sha256"]),
    )
    runtime_sources = audit["reviewed_runtime_source_manifest_sha256"]
    assert isinstance(runtime_sources, dict)
    assert set(runtime_sources) == {"custody_runtime", "numpy_dependency", "python_stdlib"}
    assert all(isinstance(value, str) and len(value) == 64 for value in runtime_sources.values())


def test_parent_lineage_exclusion_is_exactly_bound_in_scientific_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_source_hashes", lambda: {})
    monkeypatch.setattr(
        experiment,
        "_runtime_source_manifest_sha256",
        lambda: {"custody_runtime": "1" * 64, "numpy_dependency": "2" * 64, "python_stdlib": "3" * 64},
    )
    monkeypatch.setattr(experiment, "_systemd_supervisor_manifest", lambda: {})
    monkeypatch.setattr(experiment.runtime, "host_runtime_trust_manifest", lambda: {})

    config = experiment._scientific_config("4" * 64)  # noqa: SLF001 - frozen config contract
    assert config["parent_excluded_entries"] == [
        {
            "authority": "excluded_non_authoritative_lineage_only",
            "copied": False,
            "name": "inputs",
            "required_type": "non_symlink_directory",
            "traversed": False,
        }
    ]


def test_opaque_parent_copy_normalizes_only_modes_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = tmp_path / "live"
    copied = tmp_path / "copied"
    live.mkdir()
    copied.mkdir()
    (live / "inputs").mkdir()
    (live / "inputs" / "lineage-only-sentinel").symlink_to(tmp_path / "absent")
    payloads = {"a.bin": b"opaque-a", "b.bin": b"opaque-b-with-a-different-size"}
    live_modes = {"a.bin": 0o644, "b.bin": 0o444}
    pins: dict[str, records.ParentPin] = {}
    for name, payload in payloads.items():
        (live / name).write_bytes(payload)
        os.chmod(live / name, live_modes[name])
        pins[name] = records.ParentPin(hashlib.sha256(payload).hexdigest(), live_modes[name])
        records.copy_opaque_immutable_exclusive(live / name, copied / name)
    monkeypatch.setattr(records, "PARENT_PINS", pins)

    validated = experiment._validate_opaque_parent_copy(copied, live_root=live)  # noqa: SLF001
    assert set(validated) == set(payloads)
    assert not (copied / "inputs").exists()
    assert all(record["mode"] == 0o444 for record in validated.values())
    assert {name: record["bytes"] for name, record in validated.items()} == {
        name: len(payload) for name, payload in payloads.items()
    }

    os.chmod(copied / "a.bin", 0o644)
    with pytest.raises(experiment.InvalidMM009Package, match="opaque immutable parent copy differs"):
        experiment._validate_opaque_parent_copy(copied, live_root=live)  # noqa: SLF001
    os.chmod(copied / "a.bin", 0o444)

    os.chmod(copied / "b.bin", 0o644)
    (copied / "b.bin").write_bytes(b"opaque-b-mutated")
    os.chmod(copied / "b.bin", 0o444)
    with pytest.raises(experiment.InvalidMM009Package, match="opaque immutable parent copy differs"):
        experiment._validate_opaque_parent_copy(copied, live_root=live)  # noqa: SLF001
    os.chmod(copied / "b.bin", 0o644)
    (copied / "b.bin").write_bytes(payloads["b.bin"])
    os.chmod(copied / "b.bin", 0o444)

    records.write_immutable_bytes_exclusive(copied / "extra.bin", b"extra")
    with pytest.raises(experiment.InvalidMM009Package, match="membership differs"):
        experiment._validate_opaque_parent_copy(copied, live_root=live)  # noqa: SLF001


def test_prediction_chain_replays_canonical_worker_bundle_and_rejects_extra_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "MM-009-fixture"
    row = preparation.canonical_row_index()[0]
    rows = (row,)
    protocol_sha256 = "1" * 64
    config_sha256 = "2" * 64
    worker_config: records.JsonValue = {
        "config_sha256": config_sha256,
        "prediction_roles": list(worker.PREDICTION_ROLES),
        "protocol_sha256": protocol_sha256,
        "schema_version": "mm009-worker-config-v1",
    }
    records.write_immutable_json_exclusive(output / experiment.WORKER_CONFIG_FILE, worker_config)
    for relative in (experiment.START_FILE, experiment.CONTROL_FILE, experiment.POST_ISOLATION_FILE):
        records.write_immutable_json_exclusive(output / relative, {"fixture": str(relative)})
    source_path = output / experiment.SOURCE_ROOT / "000000/source.npz"
    records.write_immutable_bytes_exclusive(source_path, b"synthetic-source-custody")
    attempt = experiment._prediction_attempt_record(  # noqa: SLF001
        output,
        rows,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    records.write_immutable_json_exclusive(output / experiment.PREDICTION_ATTEMPT_FILE, attempt)

    prediction_directory = output / experiment.PREDICTION_ROOT / "000000"
    predictions = np.zeros(worker.PREDICTION_SHAPE, dtype="<f8")
    _immutable_npy(prediction_directory / "predictions.npy", predictions)
    evidence: records.JsonValue = {
        "prediction_hashes": {
            role: {
                "mm009_sha256": records.scientific_array_sha256(
                    f"prediction:{role}", predictions[index], protocol_sha256=protocol_sha256
                )
            }
            for index, role in enumerate(worker.PREDICTION_ROLES)
        }
    }
    records.write_immutable_json_exclusive(prediction_directory / "worker-evidence.json", evidence)
    predecessor = records.file_sha256(output / experiment.PREDICTION_ATTEMPT_FILE)
    commit = experiment._supervisor_commit(  # noqa: SLF001
        output,
        row,
        prediction_directory,
        predecessor,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
    )
    records.write_immutable_json_exclusive(prediction_directory / "commit.json", commit)
    commit_sha256 = records.file_sha256(prediction_directory / "commit.json")
    freeze: records.JsonValue = {
        "chain": [
            {
                "commit_sha256": commit_sha256,
                "ordinal": row.ordinal,
                "prediction_sha256": records.file_sha256(prediction_directory / "predictions.npy"),
                "worker_evidence_sha256": records.file_sha256(prediction_directory / "worker-evidence.json"),
            }
        ],
        "config_sha256": config_sha256,
        "genesis_sha256": predecessor,
        "last_commit_sha256": commit_sha256,
        "protocol_sha256": protocol_sha256,
        "row_count": 1,
        "schema_version": "mm009-prediction-freeze-v1",
        "status": "all_predictions_frozen_before_target_scoring",
    }
    records.write_immutable_json_exclusive(output / experiment.PREDICTION_FREEZE_FILE, freeze)

    replayed: list[Path] = []

    def fake_load_source(path: str | Path) -> dict[str, np.ndarray]:
        assert Path(path) == source_path
        return {"synthetic": np.asarray([1], dtype="<i8")}

    def fake_validate_worker(
        directory: str | Path,
        config: object,
        source: object,
        *,
        source_path: str | Path,
    ) -> None:
        child = Path(directory)
        assert {path.name for path in child.iterdir()} == {worker.PREDICTION_FILE, worker.COMMIT_FILE}
        assert all(records.file_record(path)["mode"] == 0o444 for path in child.iterdir())
        assert config == worker_config
        assert isinstance(source, dict)
        assert Path(source_path) == output / experiment.SOURCE_ROOT / "000000/source.npz"
        replayed.append(child)

    monkeypatch.setattr(preparation, "load_source_row_npz", fake_load_source)
    monkeypatch.setattr(worker, "validate_worker_output", fake_validate_worker)
    (prediction_directory / "commit.json").unlink()
    experiment._validate_canonical_worker_custody(  # noqa: SLF001
        output,
        row,
        prediction_directory,
        worker_config,  # type: ignore[arg-type]
        supervisor_commit_required=False,
    )
    records.write_immutable_json_exclusive(prediction_directory / "commit.json", commit)
    assert experiment._validate_prediction_chain(output, rows) == freeze  # noqa: SLF001
    assert len(replayed) == 2

    records.write_immutable_bytes_exclusive(prediction_directory / "unexpected.bin", b"not in custody schema")
    with pytest.raises(experiment.InvalidMM009Package, match="membership differs"):
        experiment._validate_prediction_chain(output, rows)  # noqa: SLF001


def test_freeze_record_binds_source_and_complete_copied_runtime_manifests(tmp_path: Path) -> None:
    protocol_sha256 = "9" * 64
    records.write_immutable_json_exclusive(tmp_path / experiment.AUDIT_FILE, {"fixture": "audit"})
    records.write_immutable_json_exclusive(tmp_path / experiment.INPUT_MANIFEST_FILE, {"fixture": "input"})
    closures: dict[str, records.JsonValue] = {
        "custody_runtime": {"artifacts": {"a.py": {"sha256": "a" * 64}}},
        "numpy_dependency": {"artifacts": {"numpy/a.py": {"sha256": "b" * 64}}},
        "python_stdlib": {"artifacts": {"os.py": {"sha256": "c" * 64}}},
    }
    source_manifests: dict[str, records.JsonValue] = {
        "custody_runtime": "d" * 64,
        "numpy_dependency": "e" * 64,
        "python_stdlib": "f" * 64,
    }
    manifest: dict[str, records.JsonValue] = {
        "config_sha256": "8" * 64,
        "protocol_sha256": protocol_sha256,
        "runtime_closures": closures,
        "runtime_source_manifest_sha256": source_manifests,
        "source_hashes": {"fixture.py": "7" * 64},
    }
    freeze = experiment._freeze_record(tmp_path, manifest)  # noqa: SLF001
    assert freeze["runtime_source_manifest_sha256"] == source_manifests
    assert freeze["runtime_closure_manifest_sha256"] == {
        name: records.canonical_json_sha256(closure, protocol_sha256=protocol_sha256)
        for name, closure in closures.items()
    }


def test_formal_start_precedes_controls_and_controls_precede_parent_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_sha256 = "3" * 64
    config_sha256 = "4" * 64
    manifest: dict[str, records.JsonValue] = {
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
    }

    class OrderedStop(RuntimeError):
        pass

    monkeypatch.setattr(experiment, "_assert_output", lambda _output: tmp_path)
    monkeypatch.setattr(experiment, "_assert_current_cgroup_role", lambda _role: None)
    monkeypatch.setattr(experiment, "_validate_frozen", lambda _output: (manifest, {}))
    monkeypatch.setattr(
        experiment,
        "_formal_start",
        lambda _output, _manifest, _freeze: {"schema_version": "synthetic-formal-start"},
    )

    def controls(**kwargs: object) -> records.JsonValue:
        assert kwargs == {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256}
        assert (tmp_path / experiment.START_FILE).is_file()
        assert not (tmp_path / experiment.CONTROL_FILE).exists()
        return {"schema_version": "synthetic-controls"}

    def parent_access(_output: Path) -> None:
        assert (tmp_path / experiment.START_FILE).is_file()
        assert (tmp_path / experiment.CONTROL_FILE).is_file()
        assert control_validations == [(config_sha256, protocol_sha256)]
        raise OrderedStop

    control_validations: list[tuple[str, str]] = []

    def validate_controls(
        _output: Path,
        *,
        config_sha256: str,
        protocol_sha256: str,
    ) -> dict[str, records.JsonValue]:
        assert (tmp_path / experiment.CONTROL_FILE).is_file()
        control_validations.append((config_sha256, protocol_sha256))
        return {"schema_version": "synthetic-controls"}

    monkeypatch.setattr(synthetic_controls, "run_controls", controls)
    monkeypatch.setattr(experiment, "_validate_controls", validate_controls)
    monkeypatch.setattr(experiment, "_post_marker_inputs", parent_access)
    with pytest.raises(OrderedStop):
        experiment._run_formal_in_process(tmp_path)  # noqa: SLF001


def test_formal_rejects_validator_valid_control_replacement_before_parent_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_sha256 = "3" * 64
    config_sha256 = "4" * 64
    controls = _control_evidence_fixture(config_sha256, protocol_sha256)
    replacement = deepcopy(controls)
    first_positive = _json_object(_json_list(replacement["positive"])[0])
    _json_object(first_positive["operator"])["forecast_sha256"] = "c" * 64
    replacement_body = {
        name: value for name, value in replacement.items() if name != "evidence_sha256"
    }
    replacement["evidence_sha256"] = records.canonical_json_sha256(
        replacement_body,
        protocol_sha256=protocol_sha256,
    )
    assert (
        synthetic_controls.validate_control_evidence(
            replacement,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        == replacement
    )
    assert records.json_file_bytes(replacement) != records.json_file_bytes(controls)

    manifest: dict[str, records.JsonValue] = {
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
    }
    monkeypatch.setattr(experiment, "_assert_output", lambda _output: tmp_path)
    monkeypatch.setattr(experiment, "_assert_current_cgroup_role", lambda _role: None)
    monkeypatch.setattr(experiment, "_validate_frozen", lambda _output: (manifest, {}))
    monkeypatch.setattr(
        experiment,
        "_formal_start",
        lambda _output, _manifest, _freeze: {"schema_version": "synthetic-formal-start"},
    )
    monkeypatch.setattr(synthetic_controls, "run_controls", lambda **_kwargs: controls)

    original_write = records.write_immutable_json_exclusive

    def replacing_write(path: str | Path, value: records.JsonValue) -> dict[str, records.JsonValue]:
        candidate = Path(path)
        if candidate == tmp_path / experiment.CONTROL_FILE:
            assert value == controls
            return original_write(candidate, replacement)
        return original_write(candidate, value)

    parent_access_reached = False

    def parent_access(_output: Path) -> None:
        nonlocal parent_access_reached
        parent_access_reached = True
        pytest.fail("parent access must not follow a replaced synthetic-control commit")

    monkeypatch.setattr(records, "write_immutable_json_exclusive", replacing_write)
    monkeypatch.setattr(experiment, "_post_marker_inputs", parent_access)
    with pytest.raises(
        experiment.InvalidMM009Package,
        match="differs from the trusted in-memory result",
    ):
        experiment._run_formal_in_process(tmp_path)  # noqa: SLF001
    assert parent_access_reached is False


def test_prepare_rejects_any_preexisting_output_before_other_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "_assert_output", lambda _output: tmp_path)
    monkeypatch.setattr(
        experiment,
        "_validate_audit",
        lambda _path: pytest.fail("audit must not be opened after output exclusivity fails"),
    )
    with pytest.raises(FileExistsError, match="wholly absent"):
        experiment.prepare(tmp_path)


def test_output_path_is_exact_and_rejects_a_symlink_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "canonical"
    expected.mkdir()
    monkeypatch.setattr(experiment, "EXPECTED_OUTPUT", expected)
    assert experiment._assert_output(expected) == expected  # noqa: SLF001
    with pytest.raises(experiment.InvalidMM009Package, match="output must be"):
        experiment._assert_output(tmp_path / "different")  # noqa: SLF001

    alias = tmp_path / "alias"
    alias.symlink_to(expected, target_is_directory=True)
    with pytest.raises(experiment.InvalidMM009Package, match="contains a symlink"):
        experiment._assert_output(alias)  # noqa: SLF001


def test_total_wall_deadline_is_monotonic_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert experiment.TOTAL_WALL_SECONDS == 14_400
    monkeypatch.setattr(experiment.time, "monotonic", lambda: 100.0)
    experiment._check_deadline(100.1, "fixture")  # noqa: SLF001
    assert experiment._deadline_timeout(101.0, 30.0, "fixture") == 1.0  # noqa: SLF001
    with pytest.raises(experiment.InvalidMM009Package, match="total-wall deadline exceeded: fixture"):
        experiment._check_deadline(100.0, "fixture")  # noqa: SLF001


def test_post_detachment_probe_uses_remaining_formal_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    def probe(
        _runtime: Path,
        _allowed: Path,
        _config: Path,
        output: Path,
        _denied: tuple[Path, ...],
        *,
        dependency_root: Path,
        stdlib_root: Path,
        timeout_seconds: int,
    ) -> dict[str, object]:
        assert dependency_root == (tmp_path / experiment.DEPENDENCY_ROOT).resolve()
        assert stdlib_root == (tmp_path / experiment.STDLIB_ROOT).resolve()
        observed.append(timeout_seconds)
        records.write_immutable_json_exclusive(
            output / "isolation-probe.json",
            {"schema_version": "fixture"},
        )
        return {"schema_version": "fixture"}

    monkeypatch.setattr(experiment.runtime, "run_isolation_probe", probe)
    experiment._post_detachment_probe(  # noqa: SLF001
        tmp_path,
        (),
        deadline=time.monotonic() + 0.2,
    )
    assert observed == [1]


def test_in_process_lifecycle_bodies_reject_cgroup_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "_assert_output",
        lambda _output: pytest.fail("output access must follow cgroup authorization"),
    )
    with pytest.raises(experiment.InvalidMM009Package, match="outside its frozen cgroup"):
        experiment._run_formal_in_process(tmp_path)  # noqa: SLF001
    with pytest.raises(experiment.InvalidMM009Package, match="outside its frozen cgroup"):
        experiment._verify_semantic_in_process(tmp_path)  # noqa: SLF001


def test_local_process_supervision_cancels_before_waiting_for_child_timeout(tmp_path: Path) -> None:
    completed = experiment._run_cancellable_process(  # noqa: SLF001
        (experiment.sys.executable, "-c", "pass"),
        cwd=tmp_path,
        timeout_seconds=5.0,
    )
    assert completed.returncode == 0
    assert completed.stdout == completed.stderr == ""

    cancel_event = experiment.threading.Event()
    cancel_event.set()
    with pytest.raises(experiment.InvalidMM009Package, match="cancelled after sibling failure"):
        experiment._run_cancellable_process(  # noqa: SLF001
            (experiment.sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            timeout_seconds=30.0,
            cancel_event=cancel_event,
        )


def test_cgroup_supervisor_contains_nested_session_after_normal_root_exit(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "normal-child.pid"
    sentinel = tmp_path / "normal-child-escaped.txt"
    nested = (
        "import os,sys,time;from pathlib import Path;"
        "Path(sys.argv[1]).write_text(str(os.getpid()),encoding='ascii');"
        "time.sleep(0.8);"
        "Path(sys.argv[2]).write_text('escaped',encoding='ascii');"
        "time.sleep(60)"
    )
    root = (
        "import subprocess,sys,time;from pathlib import Path;"
        "subprocess.Popen((sys.executable,'-I','-S','-B','-c',"
        "sys.argv[1],sys.argv[2],sys.argv[3]),start_new_session=True);"
        "deadline=time.monotonic()+2.0;pid_file=Path(sys.argv[2]);"
        "\nwhile not pid_file.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "assert pid_file.exists()"
    )
    try:
        completed = experiment._run_cgroup_supervised_process(  # noqa: SLF001
            (
                experiment.sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                root,
                nested,
                str(pid_file),
                str(sentinel),
            ),
            cwd=tmp_path,
            timeout_seconds=3.0,
            role="test-normal-exit",
        )
        assert completed.returncode == 0
        assert completed.stdout == completed.stderr == ""
        assert _wait_for_path(pid_file, timeout_seconds=1.0)
        time.sleep(1.0)
        assert not sentinel.exists()
    finally:
        _kill_escaped_fixture(pid_file, tmp_path)


def test_cgroup_supervisor_timeout_contains_double_forked_orphan(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "orphan.pid"
    sentinel = tmp_path / "orphan-escaped.txt"
    script = """
import os
import sys
import time
from pathlib import Path

child = os.fork()
if child == 0:
    os.setsid()
    grandchild = os.fork()
    if grandchild != 0:
        os._exit(0)
    Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
    time.sleep(1.5)
    Path(sys.argv[2]).write_text("escaped", encoding="ascii")
    time.sleep(60)
os.waitpid(child, 0)
deadline = time.monotonic() + 2.0
while not Path(sys.argv[1]).exists() and time.monotonic() < deadline:
    time.sleep(0.01)
assert Path(sys.argv[1]).exists()
time.sleep(60)
"""
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            experiment._run_cgroup_supervised_process(  # noqa: SLF001
                (
                    experiment.sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    script,
                    str(pid_file),
                    str(sentinel),
                ),
                cwd=tmp_path,
                timeout_seconds=0.75,
                role="test-timeout",
            )
        assert _wait_for_path(pid_file, timeout_seconds=1.0)
        time.sleep(1.0)
        assert not sentinel.exists()
    finally:
        _kill_escaped_fixture(pid_file, tmp_path)


def test_cgroup_supervisor_interrupt_cleans_live_unit_and_nested_session(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "interrupt-child.pid"
    ready = tmp_path / "interrupt-ready.txt"
    sentinel = tmp_path / "interrupt-child-escaped.txt"
    nested = (
        "import os,sys,time;from pathlib import Path;"
        "Path(sys.argv[1]).write_text(str(os.getpid()),encoding='ascii');"
        "time.sleep(1.5);"
        "Path(sys.argv[2]).write_text('escaped',encoding='ascii');"
        "time.sleep(60)"
    )
    root = (
        "import subprocess,sys,time;from pathlib import Path;"
        "subprocess.Popen((sys.executable,'-I','-S','-B','-c',"
        "sys.argv[1],sys.argv[2],sys.argv[4]),start_new_session=True);"
        "deadline=time.monotonic()+2.0;pid_file=Path(sys.argv[2]);"
        "\nwhile not pid_file.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "assert pid_file.exists();Path(sys.argv[3]).write_text('ready',encoding='ascii');"
        "time.sleep(60)"
    )
    stop = threading.Event()
    parent_pid = os.getpid()

    def interrupt_after_service_start() -> None:
        deadline = time.monotonic() + 5.0
        while not stop.is_set() and time.monotonic() < deadline:
            if ready.exists():
                os.kill(parent_pid, signal.SIGINT)
                return
            time.sleep(0.01)

    interrupter = threading.Thread(target=interrupt_after_service_start, daemon=True)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            experiment._run_cgroup_supervised_process(  # noqa: SLF001
                (
                    experiment.sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    root,
                    nested,
                    str(pid_file),
                    str(ready),
                    str(sentinel),
                ),
                cwd=tmp_path,
                timeout_seconds=10.0,
                role="test-interrupt",
            )
        assert _wait_for_path(pid_file, timeout_seconds=1.0)
        time.sleep(1.7)
        assert not sentinel.exists()
    finally:
        stop.set()
        interrupter.join(timeout=2.0)
        _kill_escaped_fixture(pid_file, tmp_path)


def test_prediction_batch_keyboard_interrupt_cancels_registry_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in (experiment.CONTROL_FILE, experiment.START_FILE, experiment.POST_ISOLATION_FILE):
        records.write_immutable_bytes_exclusive(tmp_path / relative, b"fixture\n")

    class RecordingRegistry:
        def __init__(self) -> None:
            self.cancel_count = 0

        def cancel_all(self) -> None:
            self.cancel_count += 1

    registry = RecordingRegistry()
    monkeypatch.setattr(experiment.runtime, "WorkerProcessRegistry", lambda: registry)
    shutdown_calls = _install_interrupting_executor(monkeypatch)

    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        experiment._predict_all(  # noqa: SLF001
            tmp_path,
            (),
            config_sha256="6" * 64,
            deadline=experiment.time.monotonic() + 30.0,
            protocol_sha256="5" * 64,
        )
    assert registry.cancel_count == 1
    assert shutdown_calls == [(True, True)]


def test_score_batch_keyboard_interrupt_sets_shared_cancel_and_shuts_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in (experiment.FUTURE_ISOLATION_FILE, experiment.PREDICTION_FREEZE_FILE):
        records.write_immutable_bytes_exclusive(tmp_path / relative, b"fixture\n")
    monkeypatch.setattr(
        experiment,
        "_validate_pre_score_budget",
        lambda _output, *, protocol_sha256: {},
    )
    monkeypatch.setattr(experiment, "_validate_prediction_chain", lambda _output, _rows: {})
    monkeypatch.setattr(
        experiment,
        "_score_attempt_record",
        lambda _output, _rows, **_kwargs: {},
    )

    class RecordingEvent:
        def __init__(self) -> None:
            self.set_count = 0

        def set(self) -> None:
            self.set_count += 1

    cancel_event = RecordingEvent()

    class ThreadingFixture:
        @staticmethod
        def Event() -> RecordingEvent:  # noqa: N802 - mirrors threading.Event
            return cancel_event

    monkeypatch.setattr(experiment, "threading", ThreadingFixture)
    shutdown_calls = _install_interrupting_executor(monkeypatch)

    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        experiment._run_scores(  # noqa: SLF001
            tmp_path,
            (),
            config_sha256="6" * 64,
            deadline=experiment.time.monotonic() + 30.0,
            protocol_sha256="5" * 64,
        )
    assert cancel_event.set_count == 1
    assert shutdown_calls == [(True, True)]


def test_semantic_batch_keyboard_interrupt_cancels_registry_and_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / experiment.FUTURE_ISOLATION_FILE
    records.write_immutable_bytes_exclusive(canonical, b"synthetic-future-isolation\n")
    protocol_sha256 = "5" * 64
    config_sha256 = "6" * 64
    controls = _control_evidence_fixture(config_sha256, protocol_sha256)
    records.write_immutable_json_exclusive(tmp_path / experiment.CONTROL_FILE, controls)
    monkeypatch.setattr(experiment, "_assert_current_cgroup_role", lambda _role: None)
    monkeypatch.setattr(experiment, "_assert_output", lambda _output: tmp_path)
    monkeypatch.setattr(
        experiment,
        "verify",
        lambda _output: {
            "artifact_count": 1,
            "branch": "fixture",
            "classification": "fixture",
            "go": False,
            "outcomes": "verified_results",
            "result_sha256": "8" * 64,
            "status": "verified",
        },
    )
    monkeypatch.setattr(
        experiment,
        "_load_worker_config",
        lambda _path: {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256},
    )
    monkeypatch.setattr(synthetic_controls, "run_controls", lambda **_kwargs: controls)
    monkeypatch.setattr(preparation, "canonical_row_index", lambda: ())

    def replay_future(
        _output: Path,
        *,
        evidence_path: Path | None = None,
        deadline: float | None = None,
    ) -> dict[str, records.JsonValue]:
        assert evidence_path is not None
        assert deadline is not None
        records.write_immutable_bytes_exclusive(evidence_path, b"synthetic-future-isolation\n")
        return {}

    monkeypatch.setattr(experiment, "_run_future_isolation", replay_future)

    class RecordingRegistry:
        active_pids: tuple[int, ...] = ()
        cancelled = False

        def __init__(self) -> None:
            self.cancel_count = 0

        def cancel_all(self) -> None:
            self.cancel_count += 1
            self.cancelled = True

    registry = RecordingRegistry()
    monkeypatch.setattr(experiment.runtime, "WorkerProcessRegistry", lambda: registry)
    shutdown_calls = _install_interrupting_executor(monkeypatch)

    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        experiment._verify_semantic_in_process(tmp_path)  # noqa: SLF001
    assert registry.cancel_count >= 2
    assert shutdown_calls == [(True, True)]


def test_semantic_verifier_reruns_and_bit_compares_future_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / experiment.FUTURE_ISOLATION_FILE
    records.write_immutable_bytes_exclusive(canonical, b"synthetic-future-isolation\n")
    protocol_sha256 = "5" * 64
    config_sha256 = "6" * 64
    controls = _control_evidence_fixture(config_sha256, protocol_sha256)
    records.write_immutable_json_exclusive(tmp_path / experiment.CONTROL_FILE, controls)
    monkeypatch.setattr(experiment, "_assert_current_cgroup_role", lambda _role: None)
    monkeypatch.setattr(experiment, "_assert_output", lambda _output: tmp_path)
    monkeypatch.setattr(
        experiment,
        "verify",
        lambda _output: {
            "artifact_count": 1,
            "branch": "fixture",
            "classification": "fixture",
            "go": False,
            "outcomes": "verified_results",
            "result_sha256": "8" * 64,
            "status": "verified",
        },
    )
    monkeypatch.setattr(
        experiment,
        "_load_worker_config",
        lambda _path: {
            "config_sha256": config_sha256,
            "protocol_sha256": protocol_sha256,
        },
    )
    monkeypatch.setattr(
        synthetic_controls,
        "run_controls",
        lambda **kwargs: (
            controls
            if kwargs == {"config_sha256": config_sha256, "protocol_sha256": protocol_sha256}
            else pytest.fail("semantic control replay bindings differ")
        ),
    )
    monkeypatch.setattr(preparation, "canonical_row_index", lambda: ())
    replay_paths: list[Path] = []

    def replay_future(
        _output: Path,
        *,
        evidence_path: Path | None = None,
        deadline: float | None = None,
    ) -> dict[str, records.JsonValue]:
        assert evidence_path is not None
        assert deadline is not None
        records.write_immutable_bytes_exclusive(evidence_path, canonical.read_bytes())
        replay_paths.append(evidence_path)
        return {}

    monkeypatch.setattr(experiment, "_run_future_isolation", replay_future)
    result = experiment._verify_semantic_in_process(tmp_path)  # noqa: SLF001
    assert result["semantic_controls_replayed"] is True
    assert result["semantic_future_isolation_replayed"] is True
    assert result["semantic_prediction_rows_replayed"] == 0
    assert result["semantic_total_wall_limit_seconds"] == 14_400
    assert isinstance(result["semantic_total_wall_elapsed_seconds"], float)
    assert len(replay_paths) == 1


def test_control_validation_requires_complete_semantics_and_rejects_self_digest_forgery(tmp_path: Path) -> None:
    protocol_sha256 = "5" * 64
    config_sha256 = "6" * 64
    controls = _control_evidence_fixture(config_sha256, protocol_sha256)
    records.write_immutable_json_exclusive(tmp_path / experiment.CONTROL_FILE, controls)
    assert (
        experiment._validate_controls(  # noqa: SLF001
            tmp_path,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )
        == controls
    )
    with pytest.raises(experiment.InvalidMM009Package, match="semantics differ"):
        experiment._validate_controls(  # noqa: SLF001
            tmp_path,
            config_sha256="7" * 64,
            protocol_sha256=protocol_sha256,
        )

    forged_root = tmp_path / "forged"
    forged = deepcopy(controls)
    first_positive = _json_object(_json_list(forged["positive"])[0])
    _json_object(first_positive["complete_predicates"])["joint"] = False
    forged_body = {name: value for name, value in forged.items() if name != "evidence_sha256"}
    forged["evidence_sha256"] = records.canonical_json_sha256(
        forged_body,  # type: ignore[arg-type]
        protocol_sha256=protocol_sha256,
    )
    records.write_immutable_json_exclusive(forged_root / experiment.CONTROL_FILE, forged)
    with pytest.raises(experiment.InvalidMM009Package, match="declared predicates differ"):
        experiment._validate_controls(  # noqa: SLF001
            forged_root,
            config_sha256=config_sha256,
            protocol_sha256=protocol_sha256,
        )


def test_pre_score_projection_is_exact_at_ceiling_and_fails_one_byte_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_sha256 = "5" * 64
    records.write_immutable_bytes_exclusive(tmp_path / "frozen-input.bin", b"frozen")
    initial = experiment._pre_score_budget_record(  # noqa: SLF001
        tmp_path,
        protocol_sha256=protocol_sha256,
    )
    projected = cast(int, initial["projected_max_artifact_bytes"])
    remaining = _json_object(initial["remaining_upper_bounds"])
    assert remaining == {
        "artifact_manifest": experiment.MAX_ARTIFACT_MANIFEST_FILE_BYTES,
        "budget_record": experiment.MAX_BUDGET_FILE_BYTES,
        "evidence": experiment.MAX_EVIDENCE_FILE_BYTES,
        "future_isolation": experiment.MAX_FUTURE_FILE_BYTES,
        "report": experiment.MAX_REPORT_FILE_BYTES,
        "result": experiment.MAX_RESULT_FILE_BYTES,
        "row_scores": preparation.MATCHED_ROWS * experiment.MAX_SCORE_FILE_BYTES,
        "score_attempt": experiment.MAX_SCORE_ATTEMPT_BYTES,
    }
    monkeypatch.setattr(experiment, "MAX_ARTIFACT_BYTES", projected)
    exact = experiment._pre_score_budget_record(  # noqa: SLF001
        tmp_path,
        protocol_sha256=protocol_sha256,
    )
    assert exact["status"] == "within_frozen_ceiling"
    monkeypatch.setattr(experiment, "MAX_ARTIFACT_BYTES", projected - 1)
    exceeded = experiment._pre_score_budget_record(  # noqa: SLF001
        tmp_path,
        protocol_sha256=protocol_sha256,
    )
    assert exceeded["status"] == "projected_ceiling_exceeded"


def test_custody_launcher_file_limits_match_supervisor_projection() -> None:
    assert future_isolation_launcher.MAX_FUTURE_FILE_BYTES == experiment.MAX_FUTURE_FILE_BYTES
    assert score_launcher.MAX_SCORE_FILE_BYTES == experiment.MAX_SCORE_FILE_BYTES


@pytest.mark.parametrize("relative", [experiment.SCORE_ROOT, experiment.EVIDENCE_FILE])
def test_pre_score_projection_rejects_preexisting_later_outputs(
    tmp_path: Path,
    relative: Path,
) -> None:
    if relative == experiment.SCORE_ROOT:
        (tmp_path / relative).mkdir(parents=True)
    else:
        records.write_immutable_bytes_exclusive(tmp_path / relative, b"premature")
    with pytest.raises(experiment.InvalidMM009Package, match="wholly absent"):
        experiment._require_pre_score_output_exclusivity(tmp_path)  # noqa: SLF001


def test_saved_pre_score_budget_rejects_snapshot_mutation(tmp_path: Path) -> None:
    protocol_sha256 = "5" * 64
    records.write_immutable_bytes_exclusive(tmp_path / "frozen-input.bin", b"frozen")
    budget = experiment._pre_score_budget_record(  # noqa: SLF001
        tmp_path,
        protocol_sha256=protocol_sha256,
    )
    records.write_immutable_json_exclusive(
        tmp_path / experiment.PRE_SCORE_BUDGET_FILE,
        budget,
    )
    assert (
        experiment._validate_pre_score_budget(  # noqa: SLF001
            tmp_path,
            protocol_sha256=protocol_sha256,
        )
        == budget
    )
    records.write_immutable_bytes_exclusive(tmp_path / "unregistered.bin", b"mutation")
    with pytest.raises(experiment.InvalidMM009Package, match="budget differs"):
        experiment._validate_pre_score_budget(  # noqa: SLF001
            tmp_path,
            protocol_sha256=protocol_sha256,
        )


def test_bounded_canonical_validator_rejects_oversize_and_padded_score_json(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.json"
    records.write_immutable_json_exclusive(oversized, {})
    with pytest.raises(experiment.InvalidMM009Package, match="file bound"):
        experiment._validate_bounded_canonical_file(  # noqa: SLF001
            oversized,
            maximum_bytes=2,
            label="row score",
        )

    padded = tmp_path / "padded.json"
    records.write_immutable_bytes_exclusive(padded, b"{ }\n")
    with pytest.raises(experiment.InvalidMM009Package, match="canonical JSON"):
        experiment._validate_bounded_canonical_file(  # noqa: SLF001
            padded,
            maximum_bytes=100,
            label="row score",
        )


def test_future_isolation_validation_recomputes_all_manifests_and_exact_grammar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_sha256 = "1" * 64
    freeze_path = tmp_path / experiment.PREDICTION_FREEZE_FILE
    records.write_immutable_json_exclusive(freeze_path, {"fixture": "prediction freeze"})
    freeze_record = records.file_record(freeze_path)
    freeze: dict[str, object] = {"fixture": "validated freeze"}
    source_snapshot: dict[str, records.JsonValue] = {
        "entries": [],
        "file_count": 1 + 3 * preparation.MATCHED_ROWS,
        "schema_version": "mm009-future-isolation-source-snapshot-v1",
    }
    target_snapshot: dict[str, records.JsonValue] = {
        "entries": [],
        "file_count": preparation.MATCHED_ROWS,
        "schema_version": "mm009-future-isolation-target-snapshot-v1",
    }
    source_sha256 = records.canonical_json_sha256(source_snapshot, protocol_sha256=protocol_sha256)
    target_sha256 = records.canonical_json_sha256(target_snapshot, protocol_sha256=protocol_sha256)
    finite_sha256 = "f" * 64
    counts = {
        "byte_lsb_finite_valid": preparation.MATCHED_ROWS,
        "deranged_future_valid": preparation.MATCHED_ROWS,
        "nan_rejected": preparation.MATCHED_ROWS,
        "random_finite_valid": preparation.MATCHED_ROWS,
        "rows": preparation.MATCHED_ROWS,
        "spatial_reverse_valid": preparation.MATCHED_ROWS,
    }
    evidence = _future_evidence_fixture(
        prediction_freeze_record=freeze_record,
        protocol_sha256=protocol_sha256,
        source_manifest_sha256=source_sha256,
        target_manifest_sha256=target_sha256,
        finite_manifest_sha256=finite_sha256,
    )

    monkeypatch.setattr(experiment, "_protocol_sha256", lambda: protocol_sha256)
    monkeypatch.setattr(future_isolation, "_validate_freeze", lambda _path: (freeze, protocol_sha256))
    monkeypatch.setattr(
        future_isolation,
        "_source_side_snapshot",
        lambda _path, _predictions, _freeze: source_snapshot,
    )
    monkeypatch.setattr(future_isolation, "_target_snapshot", lambda _targets: target_snapshot)
    monkeypatch.setattr(
        future_isolation,
        "_mutation_sweep",
        lambda _targets, *, protocol_sha256: (counts, finite_sha256),
    )

    valid_path = tmp_path / "valid-future-isolation.json"
    records.write_immutable_json_exclusive(valid_path, evidence)
    assert experiment._validate_future_isolation(tmp_path, evidence_path=valid_path) == evidence  # noqa: SLF001

    mutations: list[dict[str, records.JsonValue]] = []
    finite = deepcopy(evidence)
    finite["finite_mutation_manifest_sha256"] = "0" * 64
    mutations.append(finite)
    source = deepcopy(evidence)
    source["source_side_aggregate_manifest_sha256"] = "2" * 64
    _json_object(source["source_side_manifest"])["sha256"] = "2" * 64
    mutations.append(source)
    target = deepcopy(evidence)
    _json_object(target["target_input_manifest"])["sha256"] = "3" * 64
    mutations.append(target)
    grammar = deepcopy(evidence)
    mutation_grammar = _json_object(grammar["mutation_grammar"])
    _json_object(mutation_grammar["random_finite"])["seed_formula"] = "991001 + row_ordinal"
    mutations.append(grammar)
    import_guard = deepcopy(evidence)
    _json_object(import_guard["import_guard"])["forbidden_modules"] = []
    mutations.append(import_guard)
    extra = deepcopy(evidence)
    extra["unregistered_claim"] = True
    mutations.append(extra)
    for index, mutation in enumerate(mutations):
        path = tmp_path / f"mutated-future-isolation-{index}.json"
        records.write_immutable_json_exclusive(path, mutation)
        with pytest.raises(experiment.InvalidMM009Package, match="future-isolation"):
            experiment._validate_future_isolation(tmp_path, evidence_path=path)  # noqa: SLF001


@pytest.mark.parametrize(
    ("go", "decision_label", "expected"),
    (
        (True, "MM009_GO", "preregister_new_TAESD_MM001_successor_then_audit_before_execution"),
        (
            False,
            "tested_causal_operator_failure_supported",
            "preregister_MM010_source_only_analog_coverage_then_audit_before_execution",
        ),
        (False, "MM009_inconclusive", "neither_branch_diagnose_and_preregister_new_assay"),
        (False, "invalid_MM009", "neither_branch_diagnose_and_preregister_new_assay"),
    ),
)
def test_branch_mapping_is_fail_closed(go: bool, decision_label: str, expected: str) -> None:
    assert experiment._branch_for_decision(go=go, decision_label=decision_label) == expected  # noqa: SLF001


def test_artifact_manifest_is_self_excluding_stable_and_rejects_nonregular_entries(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir(mode=0o711)
    records.write_immutable_bytes_exclusive(tmp_path / "payload.bin", b"payload")
    before = experiment._artifact_manifest(tmp_path)  # noqa: SLF001
    assert experiment.ARTIFACT_MANIFEST_FILE.as_posix() not in before["artifacts"]  # type: ignore[operator]
    assert before["directory_modes"] == {
        ".": os.stat(tmp_path).st_mode & 0o7777,
        "nested": 0o711,
    }
    records.write_immutable_json_exclusive(tmp_path / experiment.ARTIFACT_MANIFEST_FILE, before)
    assert experiment._artifact_manifest(tmp_path) == before  # noqa: SLF001

    os.mkfifo(tmp_path / "unmanifested-fifo")
    with pytest.raises(experiment.InvalidMM009Package, match="non-regular entry"):
        experiment._artifact_manifest(tmp_path)  # noqa: SLF001
