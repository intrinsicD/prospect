from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle import verify as verify_module
from bench.world_model_lifecycle.planning import run_pendulum_conformance


def _rehash_conformance(report: dict[str, object]) -> None:
    body = dict(report)
    body.pop("report_sha256", None)
    payload = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(payload).hexdigest()


def test_exact_implementation_manifest_accepts_only_complete_ordered_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [
        {"path": "a.py", "bytes": 1, "sha256": "1" * 64},
        {"path": "b.py", "bytes": 2, "sha256": "2" * 64},
    ]
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: copy.deepcopy(expected),
    )

    verify_module._verify_implementation_manifest(copy.deepcopy(expected))

    adversarial = {
        "omitted": expected[1:],
        "extra": [
            *expected,
            {"path": "c.py", "bytes": 3, "sha256": "3" * 64},
        ],
        "reordered": list(reversed(expected)),
        "digest": [
            {**expected[0], "sha256": "f" * 64},
            expected[1],
        ],
    }
    for manifest in adversarial.values():
        with pytest.raises(
            verify_module.Violation,
            match="exact complete ordered",
        ):
            verify_module._verify_implementation_manifest(copy.deepcopy(manifest))


def test_fixed_formal_conformance_report_is_independently_accepted() -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )

    verify_module._verify_pendulum_conformance_report(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cases", 2),
        ("samples_per_task", 1),
        ("seed", 7),
        ("spec_horizon", 199),
        ("terminated_or_truncated_cases", 1),
        ("reward_atol", 1.0),
    ],
)
def test_fixed_formal_conformance_rejects_rehashed_contract_changes(
    field: str,
    value: object,
) -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )
    report[field] = value
    _rehash_conformance(report)

    with pytest.raises(verify_module.Violation):
        verify_module._verify_pendulum_conformance_report(report)


def test_fixed_formal_conformance_rejects_invalid_self_hash() -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )
    report["report_sha256"] = "0" * 64

    with pytest.raises(verify_module.Violation, match="self-hash"):
        verify_module._verify_pendulum_conformance_report(report)


def test_create_binding_refuses_nonformal_conformance_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_report = tmp_path / "tests.txt"
    test_report.write_text("passed\n", encoding="utf-8")
    monkeypatch.setattr(verify_module, "verify_protocol", lambda: {})
    monkeypatch.setattr(binding_module, "source_is_clean", lambda: True)

    with pytest.raises(ValueError, match="exactly 1,024"):
        binding_module.create_formal_binding(
            output_path=tmp_path / "binding.json",
            test_report_path=test_report,
            conformance_cases=2,
            device="cpu",
        )


def test_live_binding_rechecks_complete_implementation_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {
        "source": {
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
            "implementation_files": [{"path": "a.py", "bytes": 1, "sha256": "3" * 64}],
        }
    }
    monkeypatch.setattr(verify_module, "verify_binding", lambda path: binding)
    monkeypatch.setattr(
        binding_module,
        "implementation_files",
        lambda: [{"path": "b.py", "bytes": 1, "sha256": "4" * 64}],
    )

    with pytest.raises(RuntimeError, match="complete live manifest"):
        binding_module.verify_live_binding(
            tmp_path / "formal-binding.json",
            device="cpu",
        )
