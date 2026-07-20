from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bench.world_model_lifecycle import run


def test_fresh_runtime_imports_preserve_sealed_environment() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
    script = """
import os
import sys

before = dict(os.environ)
sys.path.insert(0, sys.argv[1])
import torch
from bench.world_model_lifecycle.artifact import ProducerAttempt
from bench.world_model_lifecycle.binding import DEVELOPMENT_CLOSURE_PATH
from bench.world_model_lifecycle.experiment import ExperimentConfig
from bench.world_model_lifecycle.operator import FORMAL_BINDING_ATTEMPT_PATH
from bench.world_model_lifecycle.producer_bootstrap import register_outer_terminal

assert torch is not None
assert ProducerAttempt is not None
assert DEVELOPMENT_CLOSURE_PATH is not None
assert ExperimentConfig is not None
assert FORMAL_BINDING_ATTEMPT_PATH is not None
assert register_outer_terminal is not None
assert os.environ == before
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(repository),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def _checked(arguments: list[str], *, cwd: Path) -> None:
    subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_installed_runner_derives_qualification_from_canonical_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    lifecycle = repository / "bench" / "world_model_lifecycle"
    lifecycle.mkdir(parents=True)
    (lifecycle / "protocol.json").write_text("{}\n", encoding="utf-8")
    _checked(["git", "init", "--quiet"], cwd=repository)

    installed = (
        tmp_path
        / "site-packages"
        / "bench"
        / "world_model_lifecycle"
        / "run.py"
    )
    installed.parent.mkdir(parents=True)
    source = (
        Path(__file__).resolve().parents[1]
        / "bench"
        / "world_model_lifecycle"
        / "run.py"
    )
    shutil.copyfile(source, installed)
    monkeypatch.chdir(repository)

    specification = importlib.util.spec_from_file_location(
        "installed_wm001_run",
        installed,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert module.REPO == repository
    assert module.DEVELOPMENT_QUALIFICATION_PATH == (
        lifecycle
        / "results"
        / "development"
        / "qualification-v1.5.0-attempt-2"
    )


def test_only_no_override_development_run_can_occupy_qualification_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "development"
    qualification = root / "qualification-v1.5.0-attempt-2"
    monkeypatch.setattr(run, "DEVELOPMENT_RESULTS_ROOT", root)
    monkeypatch.setattr(run, "DEVELOPMENT_QUALIFICATION_PATH", qualification)

    assert (
        run._development_output(
            None,
            seed_override=False,
            diagnostic_stamp="stamp",
        )
        == qualification
    )
    assert (
        run._development_output(
            qualification,
            seed_override=False,
            diagnostic_stamp="stamp",
        )
        == qualification
    )
    with pytest.raises(ValueError, match="sole qualification path"):
        run._development_output(
            root / "same-budget-sibling",
            seed_override=False,
            diagnostic_stamp="stamp",
        )
    with pytest.raises(ValueError, match="cannot occupy"):
        run._development_output(
            qualification,
            seed_override=True,
            diagnostic_stamp="stamp",
        )
    assert run._development_output(
        None,
        seed_override=True,
        diagnostic_stamp="stamp",
    ) == (root / "diagnostic-stamp")
