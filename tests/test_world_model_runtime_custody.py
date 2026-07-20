from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from bench.world_model_lifecycle import experiment
from bench.world_model_lifecycle.assurance import ASSURANCE
from bench.world_model_lifecycle.checkpoint import canonical_json_bytes


def _payload(schema: str) -> bytes:
    return (
        canonical_json_bytes(
            {
                "schema": schema,
                "experiment_id": "WM-001",
                "assurance": dict(ASSURANCE),
            }
        )
        + b"\n"
    )


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _captured_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema: str,
    runtime_nlink: int,
) -> Iterator[tuple[Path, int]]:
    runtime = tmp_path / "runtime-custody.json"
    runtime.write_bytes(_payload(schema))
    for index in range(1, runtime_nlink):
        os.link(runtime, tmp_path / f"runtime-link-{index}.json")
    bootstrap = tmp_path / "producer-bootstrap.py"
    bootstrap.write_bytes(b"# bootstrap\n")
    runtime_descriptor = os.open(runtime, os.O_RDONLY)
    bootstrap_descriptor = os.open(bootstrap, os.O_RDONLY)
    try:
        for prefix, descriptor, payload in (
            ("runtime_seal", runtime_descriptor, runtime.read_bytes()),
            ("bootstrap", bootstrap_descriptor, bootstrap.read_bytes()),
        ):
            metadata = os.fstat(descriptor)
            monkeypatch.setattr(
                sys,
                f"_prospect_wm001_{prefix}_fd",
                descriptor,
                raising=False,
            )
            monkeypatch.setattr(
                sys,
                f"_prospect_wm001_{prefix}_payload",
                payload,
                raising=False,
            )
            monkeypatch.setattr(
                sys,
                f"_prospect_wm001_{prefix}_identity",
                _identity(metadata),
                raising=False,
            )
            monkeypatch.setattr(
                sys,
                f"_prospect_wm001_{prefix}_sha256",
                hashlib.sha256(payload).hexdigest(),
                raising=False,
            )
        yield runtime, runtime_descriptor
    finally:
        os.close(bootstrap_descriptor)
        os.close(runtime_descriptor)


@pytest.mark.parametrize(
    ("schema", "runtime_nlink"),
    [
        ("prospect.wm001.runtime-seal.v1", 2),
        ("prospect.world-model-lifecycle.formal-binding.v9", 1),
    ],
)
def test_experiment_accepts_schema_typed_runtime_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    runtime_nlink: int,
) -> None:
    with _captured_descriptors(
        tmp_path,
        monkeypatch,
        schema=schema,
        runtime_nlink=runtime_nlink,
    ):
        custody = experiment._captured_bootstrap_custody()
        assert custody["runtime_seal"]["schema"] == schema
        assert custody["runtime_seal_payload"] == _payload(schema)


@pytest.mark.parametrize(
    ("schema", "runtime_nlink"),
    [
        ("prospect.wm001.runtime-seal.v1", 1),
        ("prospect.world-model-lifecycle.formal-binding.v9", 2),
    ],
)
def test_experiment_rejects_link_count_for_other_custody_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    runtime_nlink: int,
) -> None:
    with _captured_descriptors(
        tmp_path,
        monkeypatch,
        schema=schema,
        runtime_nlink=runtime_nlink,
    ):
        with pytest.raises(RuntimeError, match="typed link count"):
            experiment._captured_bootstrap_custody()


def test_experiment_recheck_detects_new_runtime_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _captured_descriptors(
        tmp_path,
        monkeypatch,
        schema="prospect.wm001.runtime-seal.v1",
        runtime_nlink=2,
    ) as (runtime, _):
        experiment._captured_bootstrap_custody()
        os.link(runtime, tmp_path / "unexpected-third-link.json")
        with pytest.raises(RuntimeError, match="typed link count"):
            experiment._captured_bootstrap_custody()


@pytest.mark.parametrize(
    ("schema", "runtime_nlink"),
    [
        ("prospect.wm001.runtime-seal.v1", 2),
        ("prospect.world-model-lifecycle.formal-binding.v9", 1),
    ],
)
def test_real_subprocess_reopens_typed_runtime_descriptor(
    tmp_path: Path,
    schema: str,
    runtime_nlink: int,
) -> None:
    runtime = tmp_path / "runtime-custody.json"
    runtime.write_bytes(_payload(schema))
    for index in range(1, runtime_nlink):
        os.link(runtime, tmp_path / f"runtime-link-{index}.json")
    bootstrap = tmp_path / "producer-bootstrap.py"
    bootstrap.write_bytes(b"# bootstrap\n")
    runtime_descriptor = os.open(runtime, os.O_RDONLY)
    bootstrap_descriptor = os.open(bootstrap, os.O_RDONLY)
    script = """
import hashlib
import os
import sys
from bench.world_model_lifecycle import experiment

for prefix, descriptor in (
    ("runtime_seal", int(sys.argv[1])),
    ("bootstrap", int(sys.argv[2])),
):
    metadata = os.fstat(descriptor)
    payload = os.pread(descriptor, metadata.st_size + 1, 0)
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    setattr(sys, f"_prospect_wm001_{prefix}_fd", descriptor)
    setattr(sys, f"_prospect_wm001_{prefix}_payload", payload)
    setattr(sys, f"_prospect_wm001_{prefix}_identity", identity)
    setattr(
        sys,
        f"_prospect_wm001_{prefix}_sha256",
        hashlib.sha256(payload).hexdigest(),
    )

custody = experiment._captured_bootstrap_custody()
print(custody["runtime_seal"]["schema"])
"""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                str(runtime_descriptor),
                str(bootstrap_descriptor),
            ],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(runtime_descriptor, bootstrap_descriptor),
            timeout=30,
        )
    finally:
        os.close(bootstrap_descriptor)
        os.close(runtime_descriptor)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{schema}\n"
    assert completed.stderr == ""


def test_post_conformance_custody_recheck_requires_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"runtime_seal_sha256": "a" * 64}
    monkeypatch.setattr(
        experiment,
        "_verify_live_bootstrap_custody",
        lambda: dict(expected),
    )
    experiment._recheck_live_bootstrap_custody(expected)

    monkeypatch.setattr(
        experiment,
        "_verify_live_bootstrap_custody",
        lambda: {"runtime_seal_sha256": "b" * 64},
    )
    with pytest.raises(RuntimeError, match="changed after conformance"):
        experiment._recheck_live_bootstrap_custody(expected)
