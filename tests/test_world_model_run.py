from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bench.world_model_lifecycle import artifact as artifact_module
from bench.world_model_lifecycle import experiment as experiment_module
from bench.world_model_lifecycle import run


def test_fresh_runtime_imports_preserve_sealed_environment() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    script = """
import os
import sys

before = dict(os.environ)
sys.path.insert(0, sys.argv[1])
import torch
import gymnasium as gym
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
pendulum = gym.make("Pendulum-v1")
pendulum.close()
config = ExperimentConfig.development(device="cuda")
config.validate()
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
        / "qualification-v1.18.0"
    )
    assert module.DEVELOPMENT_CLOSURE_PATH == (
        lifecycle
        / "results"
        / "development"
        / "development-closure-v1.18.0.json"
    )
    assert module.DEVELOPMENT_DIAGNOSTICS_ROOT == (
        lifecycle
        / "results"
        / "development"
        / "diagnostics-v1.18.0"
    )


def test_only_no_override_development_run_can_occupy_qualification_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "development"
    qualification = root / "qualification-v1.18.0"
    monkeypatch.setattr(run, "DEVELOPMENT_RESULTS_ROOT", root)

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
    for reserved in (
        qualification,
        root / "development-closure-v1.18.0.json",
        root / "v1.18.0" / "preformal",
        tmp_path / "operator-v1.18" / "bindings" / "formal-binding-v1.18.0",
    ):
        with pytest.raises(
            ValueError,
            match="cannot select an output path",
        ):
            run._development_output(
                reserved,
                seed_override=True,
                diagnostic_stamp="stamp",
            )
    assert run._development_output(
        None,
        seed_override=True,
        diagnostic_stamp="stamp",
    ) == (root / "diagnostics-v1.18.0" / "diagnostic-stamp")


def test_existing_qualification_consumes_all_development_entrypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "development"
    qualification = root / "qualification-v1.18.0"
    qualification.mkdir(parents=True)
    monkeypatch.setattr(run, "DEVELOPMENT_RESULTS_ROOT", root)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(sys, "argv", ["wm001", "development", "--master-seed", "1"])

    assert run.main() == 1
    assert "qualification already consumed" in capsys.readouterr().err


def test_runtime_custody_refusal_precedes_producer_root_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "development"
    qualification = root / "qualification-v1.18.0"
    monkeypatch.setattr(run, "DEVELOPMENT_RESULTS_ROOT", root)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: (_ for _ in ()).throw(RuntimeError("no sealed custody")),
    )
    monkeypatch.setattr(
        artifact_module,
        "ProducerAttempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("producer root must not be constructed")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm001",
            "development",
            "--device",
            "cpu",
            "--output",
            str(qualification),
        ],
    )

    assert run.main() == 1
    assert not qualification.exists()
    assert "refused before producer-root creation" in capsys.readouterr().err


def test_existing_closure_consumes_development_before_runtime_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "development"
    qualification = root / "qualification-v1.18.0"
    closure = root / "development-closure-v1.18.0.json"
    root.mkdir(parents=True)
    closure.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(run, "DEVELOPMENT_RESULTS_ROOT", root)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        experiment_module,
        "_verify_live_bootstrap_custody",
        lambda: (_ for _ in ()).throw(
            AssertionError("closed development must not reach runtime custody")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wm001",
            "development",
            "--device",
            "cpu",
            "--output",
            str(qualification),
        ],
    )

    assert run.main() == 1
    assert not qualification.exists()
    assert "development is closed" in capsys.readouterr().err
