"""Focused structural and exposed-seed tests for the pure v2.2 bank layer.

The real 48-row scorer is intentionally not executed here.  Structural tests use
``RowScore`` shells only behind patched deep validators; production exposes no such
injection seam.  Synthetic generation uses only the already exposed seed 12345.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bench.multimodal_mechanism_diagnostics import bank_v22 as bank
from bench.multimodal_mechanism_diagnostics import calibration_v22 as calibration
from bench.multimodal_mechanism_diagnostics import scoring_v22 as scoring
from bench.multimodal_mechanism_diagnostics import synthetic_v22 as synthetic

EXPOSED_SEED = 12_345
CONFIG_SHA256 = "4" * 64


@pytest.fixture(scope="module")
def cases() -> tuple[synthetic.SyntheticCase, ...]:
    return tuple(
        synthetic.generate_case(scenario, seed=EXPOSED_SEED)
        for scenario in synthetic.SCENARIOS
    )


def _authority() -> dict[str, int]:
    return dict(bank.uniform_seed_authority(EXPOSED_SEED).entries)


def _fake_row(
    scenario: synthetic.Scenario,
    row: int,
    *,
    seed: int = EXPOSED_SEED,
    config_sha256: str = CONFIG_SHA256,
    input_orientation: scoring.InputOrientation = "native",
    expectation_passed: bool = True,
) -> scoring.RowScore:
    """Build a structural shell that is usable only with a patched deep validator."""

    result = object.__new__(scoring.RowScore)
    object.__setattr__(result, "scenario", scenario)
    object.__setattr__(result, "seed", seed)
    object.__setattr__(result, "row", row)
    object.__setattr__(result, "config_sha256", config_sha256)
    object.__setattr__(result, "input_orientation", input_orientation)
    object.__setattr__(result, "persistence_estimate", None)
    object.__setattr__(result, "persistence", None)
    context_names = tuple(
        SimpleNamespace(plan=SimpleNamespace(name=name)) for name in scoring.CONTEXT_ORDER
    )
    arms = tuple(
        SimpleNamespace(arm=arm, contexts=context_names)
        for arm in scoring.scenario_arms(scenario)
    )
    object.__setattr__(result, "arms", arms)
    object.__setattr__(result, "bias", SimpleNamespace(contexts=context_names))
    grid_arms = tuple(arm for arm in scoring.GRID_ARMS if arm in scoring.scenario_arms(scenario))
    streams = tuple(
        SimpleNamespace(
            stream=stream,
            grid_arms=grid_arms,
            consumer_keys=tuple(
                f"{scenario}/{row}/{stream}/{index}"
                for index in range(len(grid_arms) * (1 if stream == "full" else 3))
            ),
        )
        for stream in ("full", "p0", "p1")
    )
    object.__setattr__(result, "grid_streams", streams)
    object.__setattr__(result, "dominance", ())
    object.__setattr__(
        result,
        "expectations",
        (scoring.ExpectationCheck(f"{scenario}:required", expectation_passed),),
    )
    return result


def _fake_rows() -> tuple[scoring.RowScore, ...]:
    return tuple(
        _fake_row(scenario, row)
        for scenario in synthetic.SCENARIOS
        for row in range(calibration.SYNTHETIC_ROWS)
    )


def test_seed_authority_is_exact_copied_and_never_reads_reserved_map() -> None:
    supplied = _authority()
    authority = bank.SeedAuthority.from_mapping(supplied)
    supplied["translation"] = 999
    assert authority == bank.exposed_seed_authority()
    assert authority.seed_for("translation") == EXPOSED_SEED
    assert tuple(authority.as_mapping()) == synthetic.SCENARIOS
    with pytest.raises(TypeError):
        authority.as_mapping()["translation"] = 1  # type: ignore[index]

    for invalid in (
        {key: value for key, value in _authority().items() if key != "translation"},
        {**_authority(), "extra": EXPOSED_SEED},
        {**_authority(), "translation": True},
    ):
        with pytest.raises(bank.BankV22Error):
            bank.SeedAuthority.from_mapping(invalid)


def test_unordered_native_collection_has_exact_ledger_conjunctions_and_boundaries(
    cases: tuple[synthetic.SyntheticCase, ...],
) -> None:
    calls: list[tuple[synthetic.Scenario, int]] = []

    def accept(score: scoring.RowScore, case: synthetic.SyntheticCase) -> None:
        assert score.scenario == case.scenario
        calls.append((score.scenario, score.row))

    with patch.object(bank.scoring, "validate_row_score", accept):
        evidence = bank.assemble_bank(
            tuple(reversed(_fake_rows())),
            tuple(reversed(cases)),
            expected_seed_by_scenario=_authority(),
            config_sha256=CONFIG_SHA256,
        )
        bank.validate_bank(
            evidence,
            cases,
            expected_seed_by_scenario=_authority(),
            config_sha256=CONFIG_SHA256,
        )

    expected_keys = tuple(
        (scenario, row)
        for scenario in synthetic.SCENARIOS
        for row in range(calibration.SYNTHETIC_ROWS)
    )
    assert tuple((row.scenario, row.row) for row in evidence.rows) == expected_keys
    assert calls == [*expected_keys, *expected_keys]
    assert evidence.primitive_ledger == bank.EXPECTED_PRIMITIVE_LEDGER
    assert evidence.primitive_ledger == bank.PrimitiveLedger(34, 204, 1_428, 630, 336, 48)
    assert tuple(summary.scenario for summary in evidence.scenario_expectations) == synthetic.SCENARIOS
    assert all(summary.passed and summary.row_passes == (True,) * 6 for summary in evidence.scenario_expectations)
    assert evidence.expectations_pass
    assert evidence.expectation_failures == ()

    by_scenario = {record.scenario: record for record in evidence.boundary_occupancy}
    assert tuple(by_scenario) == (
        "translation",
        "affine",
        "appearance",
        "combined",
        "coupled_boundary",
    )
    for scenario in ("translation", "affine", "appearance", "combined"):
        record = by_scenario[scenario]
        assert record.clip_denominator == 6
        assert record.clip_occupancy == 0.0
        assert not record.site_flow_boundary
        assert not record.gradient_boundary
        assert record.expectation_passes
    coupled = by_scenario["coupled_boundary"]
    assert coupled.clip_denominator == 6
    assert coupled.clip_occupancy == 0.0
    assert coupled.site_flow_boundary
    assert coupled.gradient_boundary
    assert coupled.site_flow_boundary_count == 48
    assert coupled.gradient_boundary_count == 1
    assert coupled.max_abs_flow == 8.0
    assert coupled.max_abs_gradient == 4.0
    assert coupled.expectation_passes

    with pytest.raises(FrozenInstanceError):
        evidence.config_sha256 = "5" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evidence.primitive_ledger.row_arms = 1  # type: ignore[misc]


def test_six_row_conjunction_preserves_a_negative_row(
    cases: tuple[synthetic.SyntheticCase, ...],
) -> None:
    rows = list(_fake_rows())
    failure_index = synthetic.SCENARIOS.index("affine") * calibration.SYNTHETIC_ROWS + 4
    rows[failure_index] = _fake_row("affine", 4, expectation_passed=False)
    with patch.object(bank.scoring, "validate_row_score", return_value=None):
        evidence = bank.assemble_bank(
            tuple(rows),
            cases,
            expected_seed_by_scenario=_authority(),
            config_sha256=CONFIG_SHA256,
        )
    affine = next(
        summary for summary in evidence.scenario_expectations if summary.scenario == "affine"
    )
    assert affine.row_passes == (True, True, True, True, False, True)
    assert not affine.passed
    assert not evidence.expectations_pass
    assert evidence.expectation_failures == ("affine/row-4/affine:required",)


@pytest.mark.parametrize("defect", ("duplicate", "missing", "config", "seed", "provenance"))
def test_collection_rejects_duplicate_missing_and_authority_defects(
    defect: str,
    cases: tuple[synthetic.SyntheticCase, ...],
) -> None:
    rows = list(_fake_rows())
    if defect == "duplicate":
        rows[-1] = rows[0]
    elif defect == "missing":
        rows.pop()
    elif defect == "config":
        rows[0] = _fake_row("translation", 0, config_sha256="5" * 64)
    elif defect == "seed":
        rows[0] = _fake_row("translation", 0, seed=EXPOSED_SEED + 1)
    else:
        rows[0] = _fake_row("translation", 0, input_orientation="transposed")

    with patch.object(bank.scoring, "validate_row_score", return_value=None):
        with pytest.raises(bank.BankV22Error):
            bank.assemble_bank(
                tuple(rows),
                cases,
                expected_seed_by_scenario=_authority(),
                config_sha256=CONFIG_SHA256,
            )


def test_collect_bank_calls_exact_48_native_rows_without_a_scoring_seam(
    cases: tuple[synthetic.SyntheticCase, ...],
) -> None:
    expected = tuple(
        (scenario, row)
        for scenario in synthetic.SCENARIOS
        for row in range(calibration.SYNTHETIC_ROWS)
    )
    completed = {(row.scenario, row.row): row for row in _fake_rows()}
    calls: list[tuple[synthetic.Scenario, int, str]] = []

    def fake_score(
        case: synthetic.SyntheticCase, row: int, *, config_sha256: str
    ) -> scoring.RowScore:
        calls.append((case.scenario, row, config_sha256))
        return completed[(case.scenario, row)]

    with patch.object(bank.scoring, "score_row", fake_score):
        evidence = bank.collect_bank(
            tuple(reversed(cases)),
            expected_seed_by_scenario=_authority(),
            config_sha256=CONFIG_SHA256,
        )
    assert tuple((scenario, row) for scenario, row, _ in calls) == expected
    assert all(config == CONFIG_SHA256 for _, _, config in calls)
    assert evidence.primitive_ledger == bank.EXPECTED_PRIMITIVE_LEDGER


def test_deep_validation_rejects_case_authority_and_boundary_forgery(
    cases: tuple[synthetic.SyntheticCase, ...],
) -> None:
    with patch.object(bank.scoring, "validate_row_score", return_value=None):
        evidence = bank.assemble_bank(
            _fake_rows(),
            cases,
            expected_seed_by_scenario=_authority(),
            config_sha256=CONFIG_SHA256,
        )

    wrong_authority = _authority()
    wrong_authority["combined"] += 1
    with patch.object(bank.scoring, "validate_row_score", return_value=None):
        with pytest.raises(bank.BankV22Error):
            bank.validate_bank(
                evidence,
                cases,
                expected_seed_by_scenario=wrong_authority,
                config_sha256=CONFIG_SHA256,
            )
        with pytest.raises(bank.BankV22Error):
            bank.validate_bank(
                evidence,
                cases,
                expected_seed_by_scenario=_authority(),
                config_sha256="5" * 64,
            )

    coupled = evidence.boundary_occupancy[-1]
    forged_boundary = replace(
        coupled,
        site_flow_boundary_count=0,
        site_flow_boundary_fraction=0.0,
        gradient_boundary_count=0,
        gradient_boundary_fraction=0.0,
        site_flow_boundary=False,
        gradient_boundary=False,
    )
    forged = replace(
        evidence,
        boundary_occupancy=(*evidence.boundary_occupancy[:-1], forged_boundary),
    )
    assert not forged.expectations_pass
    with patch.object(bank.scoring, "validate_row_score", return_value=None):
        with pytest.raises(bank.BankV22Error, match="derived evidence"):
            bank.validate_bank(
                forged,
                cases,
                expected_seed_by_scenario=_authority(),
                config_sha256=CONFIG_SHA256,
            )
