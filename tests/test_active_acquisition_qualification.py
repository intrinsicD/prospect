from __future__ import annotations

import json
import math
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest

import bench.active_acquisition.qualification as qualification
from bench.active_acquisition.oracle import FractionOracle
from bench.active_acquisition.problem import HiddenActuatorProblem
from bench.active_acquisition.qualification import (
    PROTOCOL_PATH,
    TOLERANCE,
    QualificationReport,
    _exact_semantic_matrix,
    _semantic_matrix_violations,
    forbidden_public_field_paths,
    run_qualification,
)
from bench.active_acquisition.run import main

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_IMPLEMENTATION_PATHS = (
    "bench/active_acquisition/protocol.json",
    "bench/active_acquisition/__init__.py",
    "bench/active_acquisition/problem.py",
    "bench/active_acquisition/policies.py",
    "bench/active_acquisition/oracle.py",
    "bench/active_acquisition/qualification.py",
    "bench/active_acquisition/run.py",
    "src/prospect/domain/__init__.py",
    "src/prospect/domain/protocols.py",
    "src/prospect/domain/records.py",
    "src/prospect/epistemics/information.py",
)


def test_q0_report_passes_all_result_free_semantic_gates() -> None:
    report = run_qualification()

    assert report.passed
    assert report.protocol_version == "0.2.0-q"
    assert report.protocol_sha256 == sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    assert report.claim_eligible is False
    assert report.formal_authorized is False
    assert report.environment_interactions == 0
    assert report.maximum_absolute_error <= TOLERANCE
    assert all(check.passed for check in report.checks)
    assert {check.name for check in report.checks} >= {
        "candidate_identity_and_order",
        "exact_float_matrix",
        "exact_semantic_matrix",
        "independent_transcendental_matrix",
        "implementation_binding",
        "nuisance_negative_control",
        "oracle_independence",
        "protocol_parity",
        "public_schema_isolation",
        "selector_identity",
        "sign_label_invariance",
        "uniform_control_determinism",
        "uniform_protocol_vectors",
        "zero_environment_interactions",
    }


def test_q0_covers_five_exact_rows_and_required_selectors() -> None:
    report = run_qualification()

    assert tuple(row.action_id for row in report.action_rows) == (
        "skip",
        "weak",
        "strong",
        "overpowered",
        "nuisance",
    )
    assert tuple(row.exact_hand_total for row in report.action_rows) == (
        "1/2",
        "63/100",
        "37/50",
        "9/20",
        "49/100",
    )
    for row in report.action_rows:
        comparison_names = {comparison.name for comparison in row.rational_comparisons}
        assert {
            "expected_immediate_payoff",
            "prior_terminal_value",
            "expected_terminal_value_after_observation",
            "expected_decision_value",
            "action_cost",
            "acquisition_cost",
            "net_incremental_value",
            "expected_episode_value",
        } <= comparison_names
        assert row.cost_charged_once
        assert row.passed
        assert row.maximum_absolute_error <= TOLERANCE
        assert all(comparison.passed for comparison in row.rational_comparisons)
        assert row.raw_observation_entropy.passed
        assert row.information_gain.passed

    assert {row.policy: row.selected_action for row in report.selector_rows} == {
        "prospect_expected_return": "strong",
        "independent_fraction_oracle": "strong",
        "goal_only": "skip",
        "raw_observation_entropy": "nuisance",
        "eig_only": "overpowered",
        "shuffled_information": "weak",
    }
    assert all(row.passed for row in report.selector_rows)


def test_q0_nuisance_control_is_maximally_random_but_irrelevant() -> None:
    report = run_qualification()
    nuisance = next(row for row in report.action_rows if row.action_id == "nuisance")

    assert nuisance.raw_observation_entropy.independent_value == pytest.approx(math.log(4.0))
    assert nuisance.information_gain.independent_value == 0.0
    assert nuisance.information_gain.prospect_value == 0.0
    assert nuisance.raw_observation_entropy.independent_value > max(
        row.raw_observation_entropy.independent_value for row in report.action_rows if row.action_id != "nuisance"
    )


def test_q0_implementation_manifest_binds_every_declared_selected_source() -> None:
    report = run_qualification()

    assert report.implementation_binding_violations == ()
    assert tuple(row.relative_path for row in report.implementation_manifest) == (_REQUIRED_IMPLEMENTATION_PATHS)
    for row in report.implementation_manifest:
        assert row.passed
        assert row.sha256 == sha256((_REPOSITORY_ROOT / row.relative_path).read_bytes()).hexdigest()
    manifest_payload = "".join(
        f"{row.relative_path}\0{row.sha256}\0{int(row.passed)}\n" for row in report.implementation_manifest
    ).encode("utf-8")
    assert report.implementation_sha256 == sha256(manifest_payload).hexdigest()
    assert report.oracle_sha256 == next(
        row.sha256
        for row in report.implementation_manifest
        if row.relative_path == "bench/active_acquisition/oracle.py"
    )


def test_q0_checks_real_record_schemas_and_emits_no_forbidden_fields() -> None:
    report = run_qualification()
    covered_types = {row.record_type for row in report.schema_coverage_rows}

    assert "prospect.domain.records.ExperienceEvent" in covered_types
    assert "prospect.domain.records.EpistemicTransition" in covered_types
    assert "prospect.domain.records.UpdateReceipt" in covered_types
    assert "bench.active_acquisition.problem.AcquisitionResult" in covered_types
    assert "bench.active_acquisition.qualification.QualificationReport" in covered_types
    assert report.uncovered_schema_types == ()
    assert all(row.passed and not row.forbidden_fields for row in report.schema_coverage_rows)
    assert forbidden_public_field_paths(report.as_dict()) == ()


def test_q0_oracle_has_no_forbidden_import_or_call_dependency() -> None:
    report = run_qualification()

    assert report.oracle_independence_violations == ()
    assert len(report.oracle_sha256) == 64


def test_q0_is_deterministic_and_json_is_canonical() -> None:
    first = run_qualification().to_json()
    second = run_qualification().to_json()

    assert first == second
    parsed = json.loads(first)
    assert first == json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def test_q0_never_calls_environment_realization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_interaction(*args: object, **kwargs: object) -> object:
        raise AssertionError("Q0 attempted an environment interaction")

    monkeypatch.setattr(
        HiddenActuatorProblem,
        "realize_acquisition",
        forbidden_interaction,
    )
    monkeypatch.setattr(
        HiddenActuatorProblem,
        "realize_terminal",
        forbidden_interaction,
    )

    report = run_qualification()
    assert report.passed
    assert report.environment_interactions == 0


def test_q0_cli_prints_one_canonical_report_and_matches_exit_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = run_qualification()

    assert main() == 0
    assert capsys.readouterr().out == f"{expected.to_json()}\n"


def test_q0_rejects_a_protocol_that_claims_formal_authority(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["experiment"]["formal_authorized"] = True
    mutated_path = tmp_path / "protocol.json"
    mutated_path.write_text(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    report = run_qualification(mutated_path)
    assert not report.passed
    assert report.claim_eligible is False
    assert report.formal_authorized is False
    assert main(mutated_path) == 1
    assert capsys.readouterr().out == f"{report.to_json()}\n"


def test_q0_exact_semantic_matrix_has_all_88_cells_and_exact_aggregates() -> None:
    report = run_qualification()
    cells = _exact_semantic_matrix()

    assert report.semantic_matrix_cell_count == len(cells) == 88
    assert report.semantic_matrix_sha256 == "0a29e4a48ca9187e7825d2a8823f251699aa8df255fe0cf824ffeefcc5510e8e"
    assert report.semantic_matrix_action_totals == (
        "skip=1/2",
        "weak=63/100",
        "strong=37/50",
        "overpowered=9/20",
        "nuisance=49/100",
    )
    assert report.semantic_matrix_violations == ()
    assert (
        len(
            {
                (
                    cell.state,
                    cell.action_id,
                    cell.observation,
                    cell.terminal_decision,
                    cell.terminal_success,
                )
                for cell in cells
            }
        )
        == 88
    )
    assert all(cell.path_probability >= 0 for cell in cells)
    strong_success = next(
        cell
        for cell in cells
        if cell.state == 1
        and cell.action_id == "strong"
        and cell.observation == 1
        and cell.terminal_decision == 1
        and cell.terminal_success
    )
    assert strong_success.path_probability == Fraction(81, 200)
    assert strong_success.posterior_direct == Fraction(9, 10)
    assert strong_success.immediate_payoff == 1
    assert strong_success.action_cost == Fraction(29, 50)
    assert strong_success.acquisition_cost == 0
    assert strong_success.terminal_outcome_probability == Fraction(9, 10)
    assert strong_success.realized_return == Fraction(71, 50)


def test_q0_semantic_matrix_detects_omitted_realized_cost_component() -> None:
    exact = FractionOracle()
    cells = list(_exact_semantic_matrix(exact))
    index = next(
        index
        for index, cell in enumerate(cells)
        if cell.action_id == "strong"
        and cell.terminal_decision == (1 if cell.posterior_direct >= Fraction(1, 2) else -1)
        and cell.action_cost
        and cell.path_probability
    )
    cell = cells[index]
    cells[index] = replace(
        cell,
        realized_return=cell.realized_return + cell.action_cost,
    )

    violations, _ = _semantic_matrix_violations(
        tuple(cells),
        exact,
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
    )
    assert any(item.startswith("aggregation:strong:") for item in violations)


def test_q0_semantic_matrix_detects_omitted_realized_acquisition_cost() -> None:
    exact = FractionOracle()
    cells = list(_exact_semantic_matrix(exact))
    index = next(
        index
        for index, cell in enumerate(cells)
        if cell.action_id == "nuisance"
        and cell.terminal_decision == 1
        and cell.acquisition_cost
        and cell.path_probability
    )
    cell = cells[index]
    assert cell.realized_return == (
        cell.immediate_payoff + Fraction(int(cell.terminal_success)) - cell.action_cost - cell.acquisition_cost
    )
    cells[index] = replace(
        cell,
        realized_return=cell.realized_return + cell.acquisition_cost,
    )

    violations, _ = _semantic_matrix_violations(
        tuple(cells),
        exact,
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")),
    )
    assert any("cell_semantics:" in item and "realized_return=" in item for item in violations)
    assert any(item.startswith("aggregation:nuisance:") for item in violations)


def _mutated_protocol_report(
    tmp_path: Path,
    mutate: object,
) -> QualificationReport:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(protocol)
    path = tmp_path / "mutated-protocol.json"
    path.write_text(json.dumps(protocol, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return run_qualification(path)


@pytest.mark.parametrize(
    ("category", "mutate"),
    (
        ("fixture_identity", lambda p: p["fixture"].update(id="mutated")),
        ("fixture_identity", lambda p: p["fixture"]["latent_state"].update(name="mutated")),
        ("latent_fixture", lambda p: p["fixture"]["latent_state"].update(support=[-1, 0, 1])),
        ("latent_fixture", lambda p: p["fixture"]["latent_state"].update(prior=["1/3", "2/3"])),
        ("fixture_identity", lambda p: p["fixture"]["latent_state"].update(lifetime="mutated")),
        ("fixture_identity", lambda p: p["fixture"]["latent_state"].update(visibility="mutated")),
        ("action_supports", lambda p: p["fixture"]["actions"][1].update(outcomes=[1, -1])),
        ("action_likelihoods", lambda p: p["fixture"]["actions"][1].update(likelihood="mutated")),
        (
            "action_immediate_payoffs",
            lambda p: p["fixture"]["actions"][1].update(immediate_task_payoff="0"),
        ),
        ("action_reliabilities", lambda p: p["fixture"]["actions"][1].update(reliability_q="3/4")),
        ("action_reliabilities", lambda p: p["fixture"]["actions"][0].update(reliability_q=None)),
        ("action_costs", lambda p: p["fixture"]["actions"][2].update(action_cost="57/100")),
        (
            "action_totals",
            lambda p: p["fixture"]["actions"][2].update(expected_net_return_at_prior="73/100"),
        ),
        (
            "terminal_semantics",
            lambda p: p["fixture"]["terminal_decision"].update(match_success_probability="4/5"),
        ),
        (
            "terminal_declarations",
            lambda p: p["fixture"]["terminal_decision"].update(success_likelihood="mutated"),
        ),
        (
            "terminal_declarations",
            lambda p: p["fixture"]["terminal_decision"].update(payoff="mutated"),
        ),
        ("return_semantics", lambda p: p["fixture"].update(net_return="mutated")),
        ("return_semantics", lambda p: p["fixture"].update(cost_rule="mutated")),
        (
            "noncandidate_boundary",
            lambda p: p["fixture"].update(q0_only_non_candidate_fixtures=[]),
        ),
        ("noncandidate_boundary", lambda p: p["fixture"].update(non_candidate_rule="mutated")),
        (
            "candidate_order",
            lambda p: p["decision_semantics"].update(
                candidate_order=["weak", "skip", "strong", "overpowered", "nuisance"]
            ),
        ),
        (
            "required_prior_values",
            lambda p: p["decision_semantics"]["required_prior_values"].update(strong="73/100"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(prospect_total_value="mutated"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(expected_decision_value="mutated"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(information_gain="mutated"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(raw_observation_entropy="mutated"),
        ),
        ("decision_declarations", lambda p: p["decision_semantics"].update(tie_break="mutated")),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(candidate_total_value_unit="mutated"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(fraction_exact_boundary="mutated"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(floating_log_boundary="mutated"),
        ),
        (
            "decision_declarations",
            lambda p: p["decision_semantics"].update(cost_accounting="mutated"),
        ),
        ("arms", lambda p: p["arms"][0].update(selection_kind="mutated")),
        ("arm_roles", lambda p: p["arms"][0].update(role="mutated")),
        ("shuffled_mapping", lambda p: p["arms"][5]["information_source_by_action"].update(weak="weak")),
        ("report_contract", lambda p: p["q0"].update(absolute_tolerance="1e-9")),
        ("q0_contract", lambda p: p["q0"].update(name="mutated")),
        ("q0_contract", lambda p: p["q0"].update(requirements=[])),
        ("q0_contract", lambda p: p["q0"].update(on_failure="mutated")),
        ("scope_boundary", lambda p: p["scope"].update(does_not_establish=[])),
        ("scope_boundary", lambda p: p["scope"].update(exact_oracle_boundary="mutated")),
        ("authority_boundary", lambda p: p["q1"].update(execution_authorized=True)),
        ("authority_boundary", lambda p: p["seed_schedule"].update(status="mutated")),
        (
            "authority_boundary",
            lambda p: p["seed_schedule"].update(formal_master_indices=[0]),
        ),
        ("authority_boundary", lambda p: p["formal_boundary"].update(promotion_rule="mutated")),
    ),
)
def test_q0_protocol_parity_categories_fail_closed(
    tmp_path: Path,
    category: str,
    mutate: object,
) -> None:
    report = _mutated_protocol_report(tmp_path, mutate)

    assert not report.passed
    assert any(item.startswith(f"{category}:") for item in report.protocol_parity_violations)
    assert not next(check for check in report.checks if check.name == "protocol_parity").passed


def test_q0_uniform_vectors_verify_digest_index_and_action(tmp_path: Path) -> None:
    report = run_qualification()

    assert report.uniform_vector_count == 5
    assert report.uniform_vectors_sha256 == "a3f4be077e222a9c3e1aa674763cdc8f86b09dff7bac9528e77a475eb1d30879"
    assert report.uniform_vector_violations == ()

    mutated = _mutated_protocol_report(
        tmp_path,
        lambda protocol: protocol["uniform_selector"]["fixed_vectors"][0].update(
            digest="0" * 64,
            index=0,
            action="skip",
        ),
    )
    assert not mutated.passed
    assert {violation.split(":", 1)[0] for violation in mutated.uniform_vector_violations} == {"uniform_vectors"}


def test_q0_public_uniform_seed_is_allowed_but_private_seed_is_forbidden() -> None:
    assert forbidden_public_field_paths({"uniform_seed": 7}) == ()
    assert forbidden_public_field_paths({"private_seed": 7}) == ("private_seed",)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p["uniform_selector"].update(accepted_input="mutated"),
        lambda p: p["uniform_selector"]["fixed_vectors"].pop(),
        lambda p: p["uniform_selector"]["fixed_vectors"].append(
            {"seed": 99, "digest": "0" * 64, "index": 0, "action": "skip"}
        ),
        lambda p: p["uniform_selector"]["fixed_vectors"].reverse(),
    ),
)
def test_q0_uniform_contract_mutations_fail_named_vector_gate(
    tmp_path: Path,
    mutate: object,
) -> None:
    report = _mutated_protocol_report(tmp_path, mutate)

    assert not report.passed
    assert report.uniform_vector_violations
    assert all(item.startswith("uniform_vectors:") for item in report.uniform_vector_violations)
    assert not next(check for check in report.checks if check.name == "uniform_protocol_vectors").passed


def test_q0_exact_semantic_cells_are_auditor_private_and_never_serialized() -> None:
    report = run_qualification()

    assert not hasattr(qualification, "ExactSemanticCell")
    assert "ExactSemanticCell" not in qualification.__all__
    assert "exact_semantic_matrix" not in qualification.__all__
    assert all("SemanticCell" not in row.record_type for row in report.schema_coverage_rows)
    assert "state" not in report.as_dict()
    assert forbidden_public_field_paths(report.as_dict()) == ()
