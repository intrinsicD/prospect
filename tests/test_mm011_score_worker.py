from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bench.multimodal_causal_assay import preparation, records, runtime, score_worker, scoring

PROTOCOL_SHA256 = records.PROTOCOL_SHA256
CONFIG_SHA256 = "2" * 64


def _full(value: float) -> np.ndarray:
    return np.full((3, 64, 64), value, dtype="<f8")


def _write_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    records.write_immutable_bytes_exclusive(path, buffer.getvalue())


def _source_arrays(ordinal: int = 0) -> dict[str, np.ndarray]:
    rows = preparation.canonical_row_index()
    mapping = preparation.half_cycle_derangement(rows)
    row = rows[ordinal]
    mean: np.ndarray = np.zeros(3, dtype="<f8")
    scale: np.ndarray = np.ones(3, dtype="<f8")
    return {
        "current": _full(1.0),
        "current_timestamp": np.asarray([row.current_timestamp], dtype="<f8"),
        "fold_index": np.asarray([row.fold_index], dtype="<i8"),
        "normalizer_fingerprint": np.asarray([preparation.mm007_normalizer_fingerprint(mean, scale)], dtype="<U64"),
        "normalizer_mean": mean,
        "normalizer_scale": scale,
        "previous": _full(0.0),
        "previous_timestamp": np.asarray([row.previous_timestamp], dtype="<f8"),
        "row_ordinal": np.asarray([ordinal], dtype="<i8"),
        "shuffle_row_ordinal": np.asarray([int(mapping[ordinal])], dtype="<i8"),
        "shuffled_previous": _full(0.25),
        "video_id": np.asarray([row.video_id], dtype="<U11"),
    }


def _target_arrays(ordinal: int = 0) -> dict[str, np.ndarray]:
    rows = preparation.canonical_row_index()
    mapping = preparation.half_cycle_derangement(rows)
    row = rows[ordinal]
    return {
        "current_timestamp": np.asarray([row.current_timestamp], dtype="<f8"),
        "deranged_future": _full(4.0),
        "derangement_row_ordinal": np.asarray([int(mapping[ordinal])], dtype="<i8"),
        "future": _full(2.0),
        "future_timestamp": np.asarray([row.future_timestamp], dtype="<f8"),
        "row_ordinal": np.asarray([ordinal], dtype="<i8"),
        "video_id": np.asarray([row.video_id], dtype="<U11"),
    }


def _prediction_array() -> np.ndarray:
    values = {
        "history_identity": 0.0,
        "history_xfit_affine": 1.0,
        "history_xfit_appearance": 1.0,
        "history_xfit_combined": 1.0,
        "history_shuffle_xfit_affine": 2.0,
        "history_shuffle_xfit_appearance": 2.0,
        "history_shuffle_xfit_combined": 2.0,
        "history_bias": 0.5,
        "persistence": 1.0,
        "velocity": 4.0,
        "forecast_affine": 2.0,
        "forecast_appearance": 2.0,
        "forecast_combined": 2.0,
        "forecast_shuffle_affine": 0.0,
        "forecast_shuffle_appearance": 0.0,
        "forecast_shuffle_combined": 0.0,
        "forecast_reverse_affine": 3.0,
        "forecast_reverse_appearance": 3.0,
        "forecast_reverse_combined": 3.0,
        "forecast_bias": 1.5,
    }
    return cast(
        np.ndarray,
        np.ascontiguousarray(
            np.stack(
                [np.full(scoring.CENTRAL_SHAPE, values[role], dtype="<f8") for role in score_worker.PREDICTION_ROLES]
            ),
            dtype="<f8",
        ),
    )


def _worker_evidence(
    source: Path,
    predictions: Path,
    prediction_array: np.ndarray,
    *,
    roles: list[str] | None = None,
    corrupted_hash_role: str | None = None,
) -> dict[str, records.JsonValue]:
    source_arrays = preparation.load_source_row_npz(source)
    bound_roles: records.JsonValue = list(score_worker.PREDICTION_ROLES) if roles is None else list(roles)
    hashes: dict[str, records.JsonValue] = {}
    for index, role in enumerate(score_worker.PREDICTION_ROLES):
        mm011 = records.scientific_array_sha256(
            f"prediction:{role}", prediction_array[index], protocol_sha256=PROTOCOL_SHA256
        )
        if role == corrupted_hash_role:
            mm011 = "f" * 64
        hashes[role] = {
            "mm011_sha256": mm011,
            "predictor_sha256": hashlib.sha256(role.encode("ascii")).hexdigest(),
        }
    evidence: dict[str, records.JsonValue] = {
        "bias_only": {},
        "bounded": {},
        "config_sha256": CONFIG_SHA256,
        "fold_index": int(source_arrays["fold_index"][0]),
        "history": {},
        "normalizer_fingerprint": str(source_arrays["normalizer_fingerprint"][0]),
        "ordered": {},
        "prediction_file": records.file_record(predictions),
        "prediction_hashes": hashes,
        "prediction_roles": bound_roles,
        "prediction_shape": list(score_worker.PREDICTION_SHAPE),
        "protocol_sha256": PROTOCOL_SHA256,
        "reverse": {},
        "row_ordinal": int(source_arrays["row_ordinal"][0]),
        "schema_version": "mm011-source-worker-v1",
        "shuffle_row_ordinal": int(source_arrays["shuffle_row_ordinal"][0]),
        "shuffled": {},
        "shuffled_history": {},
        "source_file_sha256": records.file_sha256(source),
        "video_id": str(source_arrays["video_id"][0]),
    }
    evidence["evidence_sha256"] = records.canonical_json_sha256(
        evidence,
        protocol_sha256=PROTOCOL_SHA256,
    )
    return evidence


def _rewrite_json(path: Path, value: dict[str, Any]) -> None:
    os.chmod(path, 0o644)
    path.unlink()
    records.write_immutable_json_exclusive(path, value)  # type: ignore[arg-type]


def _build_fixture(
    tmp_path: Path,
    *,
    evidence_roles: list[str] | None = None,
    corrupted_hash_role: str | None = None,
) -> dict[str, Any]:
    source = tmp_path / "source.npz"
    target = tmp_path / "target.npz"
    preparation.write_source_row_npz(source, _source_arrays())
    preparation.write_target_row_npz(target, _target_arrays())

    config: dict[str, records.JsonValue] = {
        "config_sha256": CONFIG_SHA256,
        "prediction_roles": list(score_worker.PREDICTION_ROLES),
        "protocol_sha256": PROTOCOL_SHA256,
        "schema_version": "mm011-worker-config-v1",
    }
    config_path = tmp_path / "worker-config.json"
    records.write_immutable_json_exclusive(config_path, config)

    row_directory = tmp_path / "prediction-row"
    row_directory.mkdir()
    prediction_path = row_directory / score_worker.PREDICTION_FILE
    prediction_array = _prediction_array()
    _write_npy(prediction_path, prediction_array)
    evidence_path = row_directory / score_worker.WORKER_EVIDENCE_FILE
    evidence = _worker_evidence(
        source,
        prediction_path,
        prediction_array,
        roles=evidence_roles,
        corrupted_hash_role=corrupted_hash_role,
    )
    records.write_immutable_json_exclusive(evidence_path, evidence)

    row = preparation.canonical_row_index()[0]
    commit: dict[str, records.JsonValue] = {
        "config_sha256": CONFIG_SHA256,
        "fold_index": row.fold_index,
        "predecessor_sha256": "9" * 64,
        "protocol_sha256": PROTOCOL_SHA256,
        "row_ordinal": row.ordinal,
        "schema_version": "mm011-supervisor-prediction-commit-v1",
        "source_file": records.file_record(source),
        "video_id": row.video_id,
        "video_row": row.video_row,
        "worker_evidence_file": records.file_record(evidence_path),
        "worker_prediction_file": records.file_record(prediction_path),
    }
    commit_path = row_directory / score_worker.COMMIT_FILE
    records.write_immutable_json_exclusive(commit_path, commit)
    output = tmp_path / "score-output"
    output.mkdir()
    return {
        "commit": commit,
        "commit_path": commit_path,
        "config": config,
        "config_path": config_path,
        "evidence": evidence,
        "evidence_path": evidence_path,
        "output": output,
        "prediction_array": prediction_array,
        "prediction_path": prediction_path,
        "source": source,
        "target": target,
    }


def test_score_worker_scores_all_families_and_binds_exact_custody(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    result = score_worker.run_score_row(
        fixture["source"],
        fixture["target"],
        fixture["prediction_path"],
        fixture["commit_path"],
        fixture["config_path"],
        fixture["output"],
    )

    assert set(fixture["output"].iterdir()) == {fixture["output"] / score_worker.SCORE_FILE}
    score_path = fixture["output"] / score_worker.SCORE_FILE
    assert oct(score_path.stat().st_mode & 0o777) == "0o444"
    assert json.loads(score_path.read_text(encoding="ascii")) == result
    assert result["row_ordinal"] == 0
    assert result["video_row"] == 0
    assert result["supervisor_commit_sha256"] == records.file_sha256(fixture["commit_path"])
    assert result["target_file_sha256"] == records.file_sha256(fixture["target"])

    unit = scoring.ELEMENTS_PER_ROW
    expected = {
        "i": 1.0,
        "a": 0.0,
        "q": 1.0,
        "p": 1.0,
        "c": 0.0,
        "h": 4.0,
        "r": 1.0,
        "z": 4.0,
        "d": 4.0,
        "pd": 9.0,
        "u": 0.25,
        "b": 0.25,
        "bd": 6.25,
    }
    families = result["families"]
    assert isinstance(families, dict)
    for family in scoring.FAMILIES:
        metrics = families[family]
        assert isinstance(metrics, dict)
        for name, mse in expected.items():
            assert metrics[name] == {"count": unit, "sse": mse * unit}


def test_score_launcher_uses_only_prepared_custody_import_roots(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    dependency_root = (tmp_path / "numpy-dependency").resolve()
    stdlib_root = (tmp_path / "stdlib").resolve()
    custody_root = (tmp_path / "custody").resolve()
    runtime.build_numpy_dependency_closure(dependency_root)
    runtime.build_stdlib_closure(stdlib_root)
    runtime.build_custody_runtime(custody_root)
    launcher = custody_root / runtime.SCORE_LAUNCHER_RELATIVE
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(launcher),
            str(custody_root),
            str(dependency_root),
            str(stdlib_root),
            str(fixture["source"]),
            str(fixture["target"]),
            str(fixture["prediction_path"]),
            str(fixture["commit_path"]),
            str(fixture["config_path"]),
            str(fixture["output"]),
        ),
        cwd=fixture["output"],
        env={"PATH": os.environ.get("PATH", "")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert (fixture["output"] / score_worker.SCORE_FILE).is_file()


def test_score_launcher_sandbox_denies_sibling_live_code_network_and_process_escape(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    sibling_target = tmp_path / "sibling-target.npz"
    preparation.write_target_row_npz(sibling_target, _target_arrays(1))
    dependency_root = (tmp_path / "numpy-dependency").resolve()
    stdlib_root = (tmp_path / "stdlib").resolve()
    custody_root = (tmp_path / "custody").resolve()
    runtime.build_numpy_dependency_closure(dependency_root)
    runtime.build_stdlib_closure(stdlib_root)
    runtime.build_custody_runtime(custody_root)

    live_predictor = Path(__file__).resolve().parents[1] / "bench/multimodal_causal_assay/predictor.py"
    malicious_worker = custody_root / "bench/multimodal_causal_assay/score_worker.py"
    malicious_source = f"""\
import importlib.machinery
import os
import socket
from pathlib import Path


def _must_be_denied(label, operation):
    try:
        operation()
    except PermissionError:
        return
    raise RuntimeError(label + " escaped custody")


def main(arguments=None):
    values = arguments
    target = Path(values[1])
    sibling = target.with_name("sibling-target.npz")
    live = Path({str(live_predictor)!r})
    loader = importlib.machinery.SourceFileLoader("live_predictor", str(live))
    _must_be_denied("sibling target", sibling.read_bytes)
    _must_be_denied("live predictor raw open", live.read_bytes)
    _must_be_denied("live predictor manual loader", lambda: loader.get_data(str(live)))
    _must_be_denied("network", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    _must_be_denied("setsid", os.setsid)
    _must_be_denied("setpgid", lambda: os.setpgid(0, 0))
    _must_be_denied("execve", lambda: os.execve("/bin/true", ("true",), {{}}))
    score = Path(values[-1]) / "score.json"
    score.write_text("{{}}\\n", encoding="ascii")
    score.chmod(0o444)
    return 0
"""
    os.chmod(malicious_worker, 0o644)
    malicious_worker.write_text(malicious_source, encoding="ascii")
    os.chmod(malicious_worker, 0o444)

    launcher = custody_root / runtime.SCORE_LAUNCHER_RELATIVE
    result = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(launcher),
            str(custody_root),
            str(dependency_root),
            str(stdlib_root),
            str(fixture["source"]),
            str(fixture["target"]),
            str(fixture["prediction_path"]),
            str(fixture["commit_path"]),
            str(fixture["config_path"]),
            str(fixture["output"]),
        ),
        cwd=fixture["output"],
        env={"PATH": os.environ.get("PATH", "")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert set(fixture["output"].iterdir()) == {fixture["output"] / score_worker.SCORE_FILE}


@pytest.mark.parametrize("mutation", ["source", "prediction", "evidence", "config", "commit", "target"])
def test_score_worker_input_mutations_fail_before_output(tmp_path: Path, mutation: str) -> None:
    fixture = _build_fixture(tmp_path)
    if mutation in {"source", "prediction", "evidence"}:
        path = fixture[{"source": "source", "prediction": "prediction_path", "evidence": "evidence_path"}[mutation]]
        os.chmod(path, 0o644)
        with path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([byte[0] ^ 1]))
        os.chmod(path, 0o444)
    elif mutation == "config":
        changed = dict(fixture["config"])
        changed["config_sha256"] = "3" * 64
        _rewrite_json(fixture["config_path"], changed)
    elif mutation == "commit":
        changed = dict(fixture["commit"])
        changed["video_row"] = 1
        _rewrite_json(fixture["commit_path"], changed)
    else:
        os.chmod(fixture["target"], 0o644)
        fixture["target"].unlink()
        preparation.write_target_row_npz(fixture["target"], _target_arrays(1))

    with pytest.raises(score_worker.ScoreWorkerError):
        score_worker.run_score_row(
            fixture["source"],
            fixture["target"],
            fixture["prediction_path"],
            fixture["commit_path"],
            fixture["config_path"],
            fixture["output"],
        )
    assert not any(fixture["output"].iterdir())


@pytest.mark.parametrize("defect", ["role_order", "role_hash"])
def test_score_worker_replays_role_order_and_per_role_hashes(tmp_path: Path, defect: str) -> None:
    roles = list(score_worker.PREDICTION_ROLES)
    if defect == "role_order":
        roles[0], roles[1] = roles[1], roles[0]
    fixture = _build_fixture(
        tmp_path,
        evidence_roles=roles,
        corrupted_hash_role=None if defect == "role_order" else "forecast_affine",
    )
    with pytest.raises(score_worker.ScoreWorkerError, match="role|hash"):
        score_worker.run_score_row(
            fixture["source"],
            fixture["target"],
            fixture["prediction_path"],
            fixture["commit_path"],
            fixture["config_path"],
            fixture["output"],
        )


def test_score_worker_is_one_shot_and_never_overwrites_score(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    arguments = (
        fixture["source"],
        fixture["target"],
        fixture["prediction_path"],
        fixture["commit_path"],
        fixture["config_path"],
        fixture["output"],
    )
    score_worker.run_score_row(*arguments)
    original = (fixture["output"] / score_worker.SCORE_FILE).read_bytes()
    with pytest.raises(score_worker.ScoreWorkerError, match="empty"):
        score_worker.run_score_row(*arguments)
    assert (fixture["output"] / score_worker.SCORE_FILE).read_bytes() == original


def test_score_worker_import_graph_and_fresh_process_exclude_fitting_modules(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    forbidden = (
        "bench.multimodal_causal_assay.worker",
        "bench.multimodal_causal_assay.predictor",
        "bench.multimodal_mechanism_diagnostics.fitting_v22",
        "bench.multimodal_mechanism_diagnostics.global_v22",
        "bench.multimodal_mechanism_diagnostics.nongrid_v22",
    )
    code = (
        "import sys\n"
        "from bench.multimodal_causal_assay import score_worker\n"
        f"forbidden={forbidden!r}\n"
        "loaded=[name for name in forbidden if name in sys.modules]\n"
        "raise SystemExit('forbidden imports: '+repr(loaded) if loaded else 0)\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "src"),
            )
        ),
    }
    import_check = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_check.returncode == 0, import_check.stderr

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.multimodal_causal_assay.score_worker",
            str(fixture["source"]),
            str(fixture["target"]),
            str(fixture["prediction_path"]),
            str(fixture["commit_path"]),
            str(fixture["config_path"]),
            str(fixture["output"]),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert (fixture["output"] / score_worker.SCORE_FILE).is_file()
