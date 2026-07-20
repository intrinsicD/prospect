from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from bench.world_model_lifecycle import run


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
        / "qualification-v1.5.0"
    )


def test_only_no_override_development_run_can_occupy_qualification_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "development"
    qualification = root / "qualification-v1.5.0"
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
