from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from bench.epistemic.lifecycle import (
    IRRELEVANT_PROBE,
    NOISE_PROBE,
    RELEVANT_PROBE,
    BinaryRuleTask,
    ExactRuleAgent,
    ProbeAction,
    diagnose_probe,
    empirically_exact_training_signals,
    evaluate_frozen,
    evaluation_metric,
    make_balanced_task_suite,
    select_probe_by_raw_entropy,
    select_probe_by_voi,
)
from bench.epistemic.maturity import (
    EvidenceDisposition,
    MaturityRun,
    TrainingCondition,
    assert_matched_resources,
    gate_by_id,
    metric_by_name,
    report_is_deterministic,
    run_maturity_benchmark,
    train_condition,
)
from bench.epistemic.run_maturity import main as run_maturity_main
from prospect.domain import EvidenceOrigin, ExperienceKind, UpdateStatus


@pytest.fixture(scope="module")
def maturity_run() -> MaturityRun:
    return run_maturity_benchmark()


def test_exact_policy_separates_entropy_information_and_value() -> None:
    relevant = diagnose_probe(0.5, RELEVANT_PROBE)
    irrelevant = diagnose_probe(0.5, IRRELEVANT_PROBE)
    noise = diagnose_probe(0.5, NOISE_PROBE)

    assert relevant.observation_entropy_nats == pytest.approx(noise.observation_entropy_nats)
    assert irrelevant.observation_entropy_nats == pytest.approx(noise.observation_entropy_nats)
    assert relevant.expected_information_gain_nats > 0.0
    assert relevant.expected_decision_value > 0.0
    assert relevant.net_value > 0.0
    assert irrelevant.expected_information_gain_nats == pytest.approx(0.0, abs=1e-12)
    assert irrelevant.expected_decision_value == pytest.approx(0.0, abs=1e-12)
    assert noise.expected_information_gain_nats == pytest.approx(0.0, abs=1e-12)
    assert noise.expected_decision_value == pytest.approx(0.0, abs=1e-12)

    assert select_probe_by_voi(0.5) is ProbeAction.RELEVANT
    assert select_probe_by_voi(0.8) is ProbeAction.RELEVANT
    assert select_probe_by_voi(16.0 / 17.0) is ProbeAction.EXPLOIT
    assert select_probe_by_raw_entropy(0.5) is ProbeAction.NOISE
    assert select_probe_by_raw_entropy(0.8) is ProbeAction.NOISE


def test_real_interaction_builds_linked_experience_transition_and_receipt() -> None:
    task = BinaryRuleTask("linkage-000", 0)
    agent = ExactRuleAgent("linkage-agent")
    step = agent.interact(
        task,
        ProbeAction.RELEVANT,
        0,
        experience_id="linkage-experience",
        update=True,
        require_optimal=True,
    )

    event = step.experience
    assert event.kind is ExperienceKind.INTERACTION
    assert event.task_id == task.task_id
    assert event.run_id == "run-linkage-agent"
    assert event.decision is not None
    assert event.execution is not None
    assert event.closed_at.tick > event.decision.decided_at.tick
    raw_payload = event.observation.evidence.payload
    assert isinstance(raw_payload, dict)
    payload = cast(Mapping[str, object], raw_payload)
    assert "rule" not in payload
    assert event.observation.evidence.lineage.origin is EvidenceOrigin.OBSERVED

    assessment = event.decision.selected_assessment
    assert assessment.action.action_kind == "probe:relevant"
    assert assessment.information_value.expected_reduction > 0.0
    assert assessment.total_value == pytest.approx(
        assessment.utility.expected_value
        + assessment.information_value.expected_reduction
        - assessment.information_value.expected_cost
    )

    assert step.transition is not None
    assert step.receipt is not None
    assert step.transition.experience.experience_id == event.experience_id
    assert step.transition.belief_update.prior.belief_id == event.decision.belief.belief_id
    assert step.transition.proper_scores[0].prediction_id == assessment.prediction.prediction_id
    assert step.receipt.status is UpdateStatus.APPLIED
    assert step.receipt.transitions == (step.transition,)
    assert step.receipt.previous_configuration_version != step.receipt.new_configuration_version


def test_only_relevant_evidence_changes_the_exact_rule_belief() -> None:
    task = BinaryRuleTask("belief-000", 0)
    relevant = ExactRuleAgent("relevant-agent")
    irrelevant = ExactRuleAgent("irrelevant-agent")

    relevant.interact(
        task,
        ProbeAction.RELEVANT,
        0,
        experience_id="relevant-0",
        update=True,
        require_optimal=True,
    )
    irrelevant.interact(
        task,
        ProbeAction.IRRELEVANT,
        0,
        experience_id="irrelevant-0",
        update=True,
    )

    assert relevant.posterior(task.task_id) == pytest.approx(0.2)
    assert irrelevant.posterior(task.task_id) == pytest.approx(0.5)
    assert len(relevant.receipts) == len(irrelevant.receipts) == 1


def test_experience_identity_is_single_consume() -> None:
    task = BinaryRuleTask("identity-000", 0)
    agent = ExactRuleAgent()
    agent.interact(
        task,
        ProbeAction.RELEVANT,
        0,
        experience_id="one-experience",
        update=True,
    )
    with pytest.raises(ValueError, match="unique"):
        agent.interact(
            task,
            ProbeAction.RELEVANT,
            0,
            experience_id="one-experience",
            update=True,
        )


def test_frozen_evaluation_has_zero_state_mutation() -> None:
    task = BinaryRuleTask("frozen-eval-000", 0)
    agent = ExactRuleAgent()
    agent.interact(
        task,
        ProbeAction.RELEVANT,
        0,
        experience_id="frozen-eval-experience",
        update=True,
    )
    digest_before = agent.state_digest()
    experience_count = len(agent.experiences)
    receipt_count = len(agent.receipts)

    evaluation = evaluate_frozen(agent, (task,), evaluation_id="frozen-evaluation")

    assert agent.state_digest() == digest_before
    assert len(agent.experiences) == experience_count
    assert len(agent.receipts) == receipt_count
    assert not evaluation.training_updates_allowed
    assert evaluation.update_receipts == ()
    assert evaluation.transition_ids == ()


def test_checkpoint_roundtrip_restores_behavior_memory_and_resources(tmp_path: Path) -> None:
    task = BinaryRuleTask("checkpoint-000", 1)
    agent = ExactRuleAgent("checkpoint-agent")
    for round_index in (0, 1):
        agent.interact(
            task,
            ProbeAction.RELEVANT,
            1,
            experience_id=f"checkpoint-{round_index}",
            update=True,
            require_optimal=True,
        )
    checkpoint_path = tmp_path / "reference-checkpoint.json"
    agent.save_checkpoint(checkpoint_path)
    restarted = ExactRuleAgent.load_checkpoint(checkpoint_path)

    assert restarted.checkpoint_json() == agent.checkpoint_json()
    assert restarted.state_digest() == agent.state_digest()
    assert restarted.posterior(task.task_id) == pytest.approx(agent.posterior(task.task_id))
    assert restarted.total_probe_cost == pytest.approx(agent.total_probe_cost)
    assert restarted.total_environment_steps == agent.total_environment_steps
    original_eval = evaluate_frozen(agent, (task,), evaluation_id="original")
    restart_eval = evaluate_frozen(restarted, (task,), evaluation_id="restarted")
    assert evaluation_metric(restart_eval, "mean_log_score") == pytest.approx(
        evaluation_metric(original_eval, "mean_log_score")
    )
    assert evaluation_metric(restart_eval, "mean_external_utility") == pytest.approx(
        evaluation_metric(original_eval, "mean_external_utility")
    )


def test_checkpoint_rejects_noncanonical_or_corrupt_state() -> None:
    checkpoint = ExactRuleAgent().checkpoint_json()
    decoded = json.loads(checkpoint)
    assert isinstance(decoded, dict)
    decoded["consumed_ids"] = ["missing-experience"]
    corrupt = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="consumed ids"):
        ExactRuleAgent.from_checkpoint_json(corrupt)


def test_training_schedule_exactly_matches_two_probe_channel_distribution() -> None:
    tasks = make_balanced_task_suite("schedule", tasks_per_rule=25)
    patterns: dict[tuple[bool, bool], int] = {}
    for task in tasks[::2]:
        pattern = tuple(empirically_exact_training_signals(task, round_index) == task.rule for round_index in (0, 1))
        assert len(pattern) == 2
        typed_pattern = (pattern[0], pattern[1])
        patterns[typed_pattern] = patterns.get(typed_pattern, 0) + 1

    assert patterns[(True, True)] == 16
    assert patterns[(True, False)] + patterns[(False, True)] == 8
    assert patterns[(False, False)] == 1


def test_e2_collection_uses_real_records_and_matched_budgets(maturity_run: MaturityRun) -> None:
    exact = maturity_run.exact_collection
    raw = maturity_run.raw_entropy_collection
    random = maturity_run.random_collection
    gate = gate_by_id(maturity_run.report, "E2")

    assert gate.diagnostic_checks_passed
    assert not gate.claim_supported
    assert gate.disposition is EvidenceDisposition.REFERENCE_ONLY
    assert exact.relevant_count == exact.environment_steps == 30
    assert raw.noise_count == raw.environment_steps == 30
    assert random.relevant_count == random.irrelevant_count == random.noise_count == 10
    assert exact.total_cost == pytest.approx(raw.total_cost)
    assert exact.total_cost == pytest.approx(random.total_cost)
    assert all(event.observation.evidence.lineage.origin is EvidenceOrigin.OBSERVED for event in exact.experiences)
    assert all(event.behavior_policy_version == "exact-voi-v1" for event in exact.experiences)
    assert all(event.behavior_policy_version == "raw-observation-entropy-v1" for event in raw.experiences)
    assert metric_by_name(gate, "exact_eig_per_cost") > metric_by_name(gate, "random_eig_per_cost")


def test_e2_integrity_lane_uses_authoritative_runtime_store_and_ledger(
    maturity_run: MaturityRun,
) -> None:
    result = maturity_run.runtime_integrity
    interaction = result.interaction

    assert result.passed
    assert result.store.get(interaction.experience.experience_id) is interaction.experience
    assert result.ledger.get_transition(interaction.transition.transition_id) is interaction.transition
    assert result.ledger.get_update(result.receipt.receipt_id) is result.receipt
    assert result.receipt.transitions == (interaction.transition,)
    assert len(interaction.decision.alternatives) == 3
    assert interaction.decision.selected_assessment.action.action_kind == "probe:relevant"
    assert result.final_snapshot.belief is interaction.transition.belief_update.posterior
    assert result.final_snapshot.configuration_version == "runtime-config-v1"
    assert (
        metric_by_name(
            gate_by_id(maturity_run.report, "E2"),
            "authoritative_runtime_integrity",
        )
        == 1.0
    )


def test_e3_learning_gain_requires_correct_evidence_linkage(maturity_run: MaturityRun) -> None:
    gate = gate_by_id(maturity_run.report, "E3")
    relevant = maturity_run.relevant_training
    frozen = maturity_run.frozen_training
    shuffled = maturity_run.shuffled_training
    irrelevant = maturity_run.irrelevant_training

    assert gate.diagnostic_checks_passed
    assert not gate.claim_supported
    assert gate.disposition is EvidenceDisposition.REFERENCE_ONLY
    assert evaluation_metric(relevant.evaluation, "mean_log_score") < evaluation_metric(
        frozen.evaluation, "mean_log_score"
    )
    assert evaluation_metric(irrelevant.evaluation, "mean_log_score") == pytest.approx(
        evaluation_metric(frozen.evaluation, "mean_log_score")
    )
    assert evaluation_metric(shuffled.evaluation, "mean_log_score") > evaluation_metric(
        frozen.evaluation, "mean_log_score"
    )
    assert len(relevant.agent.receipts) == 100
    assert len(shuffled.agent.receipts) == 100
    assert len(irrelevant.agent.receipts) == 100
    assert len(frozen.agent.receipts) == 0
    assert_matched_resources(
        relevant.agent,
        frozen.agent,
        shuffled.agent,
        irrelevant.agent,
    )


def test_shuffled_label_control_is_marked_as_derived(maturity_run: MaturityRun) -> None:
    events = maturity_run.shuffled_training.agent.experiences
    assert events
    assert all(event.observation.evidence.lineage.origin is EvidenceOrigin.DERIVED for event in events)
    assert all(event.observation.evidence.lineage.producer_version == "task-label-permutation-v1" for event in events)
    assert all(event.observation.evidence.lineage.provenance.trust.value == 0 for event in events)


def test_e4_proper_score_gain_changes_external_behavior(maturity_run: MaturityRun) -> None:
    gate = gate_by_id(maturity_run.report, "E4")

    assert gate.diagnostic_checks_passed
    assert not gate.claim_supported
    assert gate.disposition is EvidenceDisposition.BLOCKED
    assert metric_by_name(gate, "learned_external_utility") == pytest.approx(0.8832)
    assert metric_by_name(gate, "frozen_external_utility") == pytest.approx(0.76)
    assert metric_by_name(gate, "external_utility_gain") == pytest.approx(0.1232)
    assert metric_by_name(gate, "external_regret_reduction") == pytest.approx(0.1232)
    assert metric_by_name(gate, "frozen_evaluation_policy") == 1.0


def test_e5_a_b_restart_a_retains_both_improvements(maturity_run: MaturityRun) -> None:
    gate = gate_by_id(maturity_run.report, "E5")
    trace = maturity_run.retention

    assert gate.diagnostic_checks_passed
    assert not gate.claim_supported
    assert gate.disposition is EvidenceDisposition.BLOCKED
    assert trace.sequence == (
        "learn:A",
        "evaluate:A",
        "learn:B",
        "evaluate:B",
        "evaluate:A",
        "save",
        "restart",
        "evaluate:A",
        "evaluate:B",
    )
    assert metric_by_name(gate, "a_interference_drift") == pytest.approx(0.0)
    assert metric_by_name(gate, "a_restart_utility_drift") == pytest.approx(0.0)
    assert metric_by_name(gate, "a_restart_log_score_drift") == pytest.approx(0.0)
    assert metric_by_name(gate, "b_utility_after_b") > metric_by_name(gate, "b_utility_before_b")
    assert trace.checkpoint == trace.restarted_agent.checkpoint_json()


def test_machine_report_is_complete_and_deterministic(maturity_run: MaturityRun) -> None:
    second = run_maturity_benchmark()
    decoded = json.loads(maturity_run.report.to_json())

    assert not maturity_run.report.passed
    assert [gate.gate_id for gate in maturity_run.report.gates] == ["E2", "E3", "E4", "E5"]
    assert isinstance(decoded, dict)
    assert decoded["schema"] == "prospect.epistemic.reference-diagnostics.e2-e5.v2"
    assert decoded["passed"] is False
    assert len(decoded["gates"]) == 4
    decoded_gates = cast(list[dict[str, object]], decoded["gates"])
    assert all(gate["diagnostic_checks_passed"] is True for gate in decoded_gates)
    assert all(gate["claim_supported"] is False for gate in decoded_gates)
    assert report_is_deterministic(maturity_run.report, second.report)


def test_reference_rows_do_not_form_a_single_agent_causal_chain(
    maturity_run: MaturityRun,
) -> None:
    collection_ids = {event.experience_id for event in maturity_run.exact_collection.experiences}
    learned_ids = {
        transition.experience.experience_id
        for receipt in maturity_run.relevant_training.agent.receipts
        for transition in receipt.transitions
    }

    assert collection_ids
    assert learned_ids
    assert collection_ids.isdisjoint(learned_ids)
    assert not maturity_run.report.passed


def test_cli_separates_reference_diagnostics_from_capability_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_maturity_main(["--diagnostics"]) == 0
    diagnostic_report = json.loads(capsys.readouterr().out)
    assert diagnostic_report["passed"] is False

    assert run_maturity_main([]) == 1
    capability_report = json.loads(capsys.readouterr().out)
    assert capability_report == diagnostic_report


def test_frozen_condition_collects_equal_experience_without_learning() -> None:
    tasks = make_balanced_task_suite("small-frozen", tasks_per_rule=2)
    trace = train_condition(TrainingCondition.FROZEN, tasks)

    assert len(trace.agent.experiences) == 8
    assert len(trace.agent.transitions) == 0
    assert len(trace.agent.receipts) == 0
    assert all(trace.agent.posterior(task.task_id) == pytest.approx(0.5) for task in tasks)


def test_task_rule_generalizes_to_both_heldout_cues() -> None:
    identity = BinaryRuleTask("identity-task", 0)
    flip = BinaryRuleTask("flip-task", 1)

    assert (identity.label(0), identity.label(1)) == (0, 1)
    assert (flip.label(0), flip.label(1)) == (1, 0)
