from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from bench.multimodal_causal_assay import preparation, records, runtime, worker


def _fake_worker_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
) -> tuple[Path, Path, Path, Path, Path]:
    shadow = (tmp_path / "fake-shadow").resolve()
    shadow.mkdir()
    launcher_path = shadow / "fake-worker.py"
    launcher_path.write_text(script, encoding="utf-8")
    source = (tmp_path / "source.bin").resolve()
    source.write_bytes(b"source")
    config = (tmp_path / "config.json").resolve()
    config.write_bytes(b"{}\n")
    dependency = (tmp_path / "dependency").resolve()
    dependency.mkdir()
    stdlib = (tmp_path / "stdlib").resolve()
    stdlib.mkdir()
    monkeypatch.setattr(runtime, "LAUNCHER_RELATIVE", Path("fake-worker.py"))
    monkeypatch.setattr(runtime, "runtime_manifest", lambda _root: {})
    monkeypatch.setattr(runtime, "_prepared_dependency_root", lambda root: Path(root))
    monkeypatch.setattr(runtime, "_prepared_stdlib_root", lambda root: Path(root))
    return shadow, source, config, dependency, stdlib


def _formal_like_source(path: Path) -> Path:
    ids, times = preparation.expected_raw_identities()
    rows = preparation.build_row_index(ids, times)
    values = np.arange(
        preparation.RAW_ROWS * preparation.NATIVE_SIZE * preparation.NATIVE_SIZE * preparation.CHANNELS,
        dtype=np.uint32,
    )
    frames = np.ascontiguousarray(
        ((17 * values + values // 89) % 253)
        .astype(np.uint8)
        .reshape(
            preparation.RAW_ROWS,
            preparation.NATIVE_SIZE,
            preparation.NATIVE_SIZE,
            preparation.CHANNELS,
        )
    )
    normalizers = preparation.fit_fold_normalizers(frames, rows)
    mapping = preparation.half_cycle_derangement(rows)
    source = preparation.construct_source_row(frames, 0, rows, normalizers, mapping)
    preparation.write_source_row_npz(path, source)
    return path


def test_shadow_runtime_runs_one_silent_landlocked_worker(tmp_path: Path) -> None:
    shadow = (tmp_path / "shadow").resolve()
    manifest = runtime.build_shadow_runtime(shadow)
    assert set(manifest) == {"artifacts", "schema_version", "source_hashes"}
    assert oct(shadow.stat().st_mode & 0o777) == "0o555"

    numpy_source_manifest = runtime.numpy_dependency_source_manifest()
    dependency_root = (tmp_path / "numpy-dependency").resolve()
    dependency_manifest = runtime.build_numpy_dependency_closure(
        dependency_root,
        expected_source_manifest_sha256=str(numpy_source_manifest["manifest_sha256"]),
    )
    assert dependency_manifest["packages"] == ["numpy", "numpy.libs"]
    dependency_artifacts = cast(dict[str, records.JsonValue], dependency_manifest["artifacts"])
    dependency_file_count = cast(int, dependency_manifest["file_count"])
    assert dependency_file_count == len(dependency_artifacts)
    assert dependency_file_count > 0
    assert set(entry.name for entry in dependency_root.iterdir()) == {"numpy", "numpy.libs"}
    assert oct(dependency_root.stat().st_mode & 0o777) == "0o555"
    assert runtime.numpy_dependency_manifest(dependency_root) == dependency_manifest
    assert dependency_manifest["source_manifest_sha256"] == numpy_source_manifest["manifest_sha256"]

    stdlib_source_manifest = runtime.stdlib_source_manifest()
    stdlib_root = (tmp_path / "stdlib").resolve()
    stdlib_manifest = runtime.build_stdlib_closure(
        stdlib_root,
        expected_source_manifest_sha256=str(stdlib_source_manifest["manifest_sha256"]),
    )
    assert stdlib_manifest["source_manifest_sha256"] == stdlib_source_manifest["manifest_sha256"]
    assert runtime.stdlib_manifest(stdlib_root) == stdlib_manifest
    assert not (stdlib_root / "site-packages").exists()
    assert not any(stdlib_root.rglob("*.pyc"))

    custody_source_manifest = runtime.custody_runtime_source_manifest()
    custody_root = (tmp_path / "custody").resolve()
    custody_manifest = runtime.build_custody_runtime(
        custody_root,
        expected_source_manifest_sha256=str(custody_source_manifest["manifest_sha256"]),
    )
    assert custody_manifest["source_manifest_sha256"] == custody_source_manifest["manifest_sha256"]
    assert runtime.custody_runtime_manifest(custody_root) == custody_manifest

    host_trust = runtime.host_runtime_trust_manifest()
    assert host_trust["authority_model"] == "trusted-live-bootstrap-then-bound-copied-import-roots"
    bootstrap = host_trust["bootstrap"]
    assert isinstance(bootstrap, dict)
    assert bootstrap["live_stdlib_root"] == str(runtime.STDLIB_SOURCE)
    phase_boundary = bootstrap["phase_boundary"]
    assert isinstance(phase_boundary, str)
    assert "before scientific imports" in phase_boundary
    roots = cast(dict[str, records.JsonValue], host_trust["roots"])
    assert isinstance(roots, dict)
    flat_roots = {path for values in roots.values() for path in cast(list[str], values)}
    assert str(runtime.SITE_PACKAGES) not in flat_roots
    assert str(Path(sys.base_prefix)) not in flat_roots
    assert "/usr/lib" not in flat_roots

    source = _formal_like_source((tmp_path / "source.npz").resolve())
    config_value: records.JsonValue = {
        "config_sha256": "2" * 64,
        "prediction_roles": list(worker.PREDICTION_ROLES),
        "protocol_sha256": records.PROTOCOL_SHA256,
        "schema_version": "mm011-worker-config-v1",
    }
    config = (tmp_path / "config.json").resolve()
    records.write_immutable_json_exclusive(config, config_value)
    denied_file = (tmp_path / "future.bin").resolve()
    denied_file.write_bytes(b"future")
    denied_directory = (tmp_path / "targets").resolve()
    denied_directory.mkdir()
    (denied_directory / "000000.npz").write_bytes(b"future")
    probe_output = (tmp_path / "probe-output").resolve()
    probe_output.mkdir()
    probe = runtime.run_isolation_probe(
        shadow,
        source,
        config,
        probe_output,
        (denied_file, denied_directory),
        dependency_root=dependency_root,
        stdlib_root=stdlib_root,
    )
    assert probe["requested_denied_path_count"] == 2
    assert probe["live_python_roots_denied"] == len(runtime.live_python_roots())
    assert probe["denied_path_count"] == 2 + len(runtime.live_python_roots())
    assert probe["network_families_denied"] == 3
    assert probe["network_socket_variants_denied"] == 6

    output = (tmp_path / "output").resolve()
    output.mkdir()

    runtime.run_worker_process(
        shadow,
        source,
        config,
        output,
        dependency_root=dependency_root,
        stdlib_root=stdlib_root,
    )

    arrays = preparation.load_source_row_npz(source)
    worker.validate_worker_output(output, config_value, arrays, source_path=source)  # type: ignore[arg-type]


def test_broad_or_mutable_dependency_roots_fail_closed(tmp_path: Path) -> None:
    with np.testing.assert_raises(runtime.RuntimeValidationError):
        runtime.numpy_dependency_manifest(runtime.SITE_PACKAGES)
    with np.testing.assert_raises(runtime.RuntimeValidationError):
        runtime.stdlib_manifest(runtime.STDLIB_SOURCE)

    dependency_root = (tmp_path / "numpy-dependency").resolve()
    runtime.build_numpy_dependency_closure(dependency_root)
    os.chmod(dependency_root / "numpy/__init__.py", 0o644)
    with np.testing.assert_raises(runtime.RuntimeValidationError):
        runtime.numpy_dependency_manifest(dependency_root)


def test_shared_registry_cancels_worker_and_descendant_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = """
import subprocess
import sys
import time
from pathlib import Path
sentinel = sys.argv[3] + "/descendant-survived.txt"
code = (
    "import pathlib,signal,sys,time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "time.sleep(1); pathlib.Path(sys.argv[1]).write_text('alive')"
)
subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", code, sentinel])
Path(sys.argv[3], "descendant-started.txt").write_text("started")
time.sleep(60)
"""
    shadow, source, config, dependency, stdlib = _fake_worker_inputs(
        tmp_path,
        monkeypatch,
        script,
    )
    output = (tmp_path / "cancel-output").resolve()
    output.mkdir()
    registry = runtime.WorkerProcessRegistry()
    failures: list[BaseException] = []

    def launch() -> None:
        try:
            runtime.run_worker_process(
                shadow,
                source,
                config,
                output,
                dependency_root=dependency,
                stdlib_root=stdlib,
                timeout_seconds=30,
                process_registry=registry,
            )
        except BaseException as error:  # noqa: BLE001 - captured from a test thread
            failures.append(error)

    thread = threading.Thread(target=launch)
    thread.start()
    deadline = time.monotonic() + 3
    while (
        not registry.active_pids or not (output / "descendant-started.txt").exists()
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert registry.active_pids
    assert (output / "descendant-started.txt").exists()
    registry.cancel_all()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert registry.cancelled is True
    assert registry.active_pids == ()
    assert len(failures) == 1
    assert isinstance(failures[0], runtime.RuntimeValidationError)
    assert "cancelled" in str(failures[0])
    time.sleep(1.1)
    assert not (output / "descendant-survived.txt").exists()


def test_worker_timeout_cancels_registry_and_reaps_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow, source, config, dependency, stdlib = _fake_worker_inputs(
        tmp_path,
        monkeypatch,
        "import time\ntime.sleep(60)\n",
    )
    output = (tmp_path / "timeout-output").resolve()
    output.mkdir()
    registry = runtime.WorkerProcessRegistry()
    started = time.monotonic()
    with pytest.raises(runtime.RuntimeValidationError, match="timed out"):
        runtime.run_worker_process(
            shadow,
            source,
            config,
            output,
            dependency_root=dependency,
            stdlib_root=stdlib,
            timeout_seconds=1,
            process_registry=registry,
        )
    assert time.monotonic() - started < 3
    assert registry.cancelled is True
    assert registry.active_pids == ()


def test_isolation_probe_timeout_kills_stubborn_descendant_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = """
import subprocess
import sys
import time
from pathlib import Path
sentinel = str(Path(sys.argv[3]).parent / "probe-descendant-survived.txt")
code = (
    "import pathlib,signal,sys,time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "time.sleep(1.5); pathlib.Path(sys.argv[1]).write_text('alive')"
)
subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", code, sentinel])
time.sleep(60)
"""
    shadow, allowed, config, dependency, stdlib = _fake_worker_inputs(
        tmp_path,
        monkeypatch,
        script,
    )
    monkeypatch.setattr(runtime, "PROBE_RELATIVE", Path("fake-worker.py"))
    denied = (tmp_path / "denied.bin").resolve()
    denied.write_bytes(b"denied")
    monkeypatch.setattr(runtime, "live_python_roots", lambda: (denied,))
    output = (tmp_path / "probe-timeout-output").resolve()
    output.mkdir()
    with pytest.raises(runtime.RuntimeValidationError, match="timed out"):
        runtime.run_isolation_probe(
            shadow,
            allowed,
            config,
            output,
            (denied,),
            dependency_root=dependency,
            stdlib_root=stdlib,
            timeout_seconds=1,
        )
    time.sleep(1.6)
    assert not (tmp_path / "probe-descendant-survived.txt").exists()
