from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from bench.active_acquisition import q1_audit
from bench.active_acquisition.contracts import (
    Q0_REPORT_SHA256,
    canonical_json_bytes,
    canonical_sha256,
    validate_artifact,
)
from bench.active_acquisition.runtime_lane import MODEL_VERSION_PREFIX
from bench.active_acquisition.seeding import PrivateQ1SeedSchedule, derive_public_uniform_selection

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64
_SALT = bytes.fromhex("fbffef00112233445566778899aabbccddeeff102132435465768798a9bacbdc")
_RUN_ID = f"wm002-q1-{'0' * 64}"
_ATTEMPT_ID = f"{_RUN_ID}-attempt-0001"


def _prospective_review(
    *,
    protocol_sha256: str = _SHA_A,
    implementation_sha256: str = _SHA_B,
    reviewed_source_count: int = 47,
) -> dict[str, object]:
    return {
        "assurance_boundary": q1_audit._PROSPECTIVE_REVIEW_ASSURANCE_BOUNDARY,
        "blocking_findings": [],
        "claim_eligible": False,
        "formal_authorized": False,
        "implementation_sha256": implementation_sha256,
        "nonblocking_findings": ["Result-free local procedural review only."],
        "passed": True,
        "protocol_sha256": protocol_sha256,
        "protocol_version": q1_audit.Q1_PROTOCOL_VERSION,
        "q1_environment_interactions": 0,
        "q1_private_draws": 0,
        "review_method": q1_audit._PROSPECTIVE_REVIEW_METHOD,
        "review_scope": list(q1_audit._PROSPECTIVE_REVIEW_SCOPE),
        "reviewed_source_count": reviewed_source_count,
        "reviewer": "independent-adversarial-referee",
        "schema": q1_audit._PROSPECTIVE_REVIEW_SCHEMA,
        "statement": "No Q1 result or private draw was produced or inspected.",
    }


def _candidate_rows(arm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ordinal, action in enumerate(q1_audit.ACTION_ORDER):
        decision_value = q1_audit._EXPECTED_DECISION_VALUE[action]
        rows.append(
            {
                "arm_selection_score": q1_audit._arm_selection_score(arm, action),
                "assessment_id": f"assessment:{ordinal}",
                "expected_decision_value": float(decision_value),
                "expected_episode_value": float(q1_audit._EXPECTED_EPISODE_VALUE[action]),
                "expected_immediate_payoff": float(q1_audit._EXPECTED_IMMEDIATE[action]),
                "expected_information_gain_nats": q1_audit._binary_information_gain(action),
                "expected_terminal_value": float(Fraction(1, 2) + decision_value),
                "information_acquisition_cost": float(q1_audit._INFORMATION_COST[action]),
                "ordinal": ordinal,
                "physical_action_cost": float(q1_audit._ACTION_COST[action]),
                "raw_observation_entropy_nats": q1_audit._raw_observation_entropy(action),
                "selection_unit": "return",
                "semantic_action": action,
            }
        )
    return rows


def _raw_and_private(
    *,
    salt: bytes = _SALT,
) -> tuple[dict[str, object], dict[str, object]]:
    schedule = q1_audit._IndependentSeedSchedule(salt)
    master = 0
    episode = 0
    arm = "prospect_expected_return"
    action = "strong"
    theta = schedule.theta(master, episode)
    symbol = schedule.pulse_observation(master, episode, action, theta)
    posterior = q1_audit._posterior_direct(action, symbol)
    terminal_action = 1 if posterior >= Fraction(1, 2) else -1
    success = schedule.terminal_success(master, episode, terminal_action, theta)
    task_payoff = float(symbol == 1)
    action_cost = float(q1_audit._ACTION_COST[action])
    information_cost = float(q1_audit._INFORMATION_COST[action])
    episode_return = task_payoff - action_cost - information_cost + float(success)
    candidates = _candidate_rows(arm)
    terminal_candidates = [
        {
            "assessment_id": "terminal:assessment:+1",
            "expected_success": float(Fraction(1, 10) + Fraction(4, 5) * posterior),
            "ordinal": 0,
            "prediction_id": "terminal:prediction:+1",
            "terminal_action": 1,
            "unit": "return",
        },
        {
            "assessment_id": "terminal:assessment:-1",
            "expected_success": float(Fraction(9, 10) - Fraction(4, 5) * posterior),
            "ordinal": 1,
            "prediction_id": "terminal:prediction:-1",
            "terminal_action": -1,
            "unit": "return",
        },
    ]
    raw: dict[str, object] = {
        "attempt_id": _ATTEMPT_ID,
        "run_id": _RUN_ID,
        "acquisition": {
            "decision_id": "acquisition:decision",
            "evidence_count_after": 1,
            "evidence_count_before": 0,
            "execution_id": "acquisition:execution",
            "experience_id": "acquisition:experience",
            "information_acquisition_cost": information_cost,
            "intention_id": "acquisition:intention",
            "model_after_sha256": _SHA_B,
            "model_after_version": f"{MODEL_VERSION_PREFIX}{_SHA_B}",
            "model_before_sha256": q1_audit._INITIAL_MODEL_SHA256,
            "model_before_version": q1_audit._INITIAL_MODEL_VERSION,
            "net_reward": task_payoff - action_cost - information_cost,
            "observation_id": "acquisition:observation",
            "observed_symbol": symbol,
            "outcome_id": "acquisition:outcome",
            "physical_action_cost": action_cost,
            "posterior_after": str(posterior),
            "posterior_before": "1/2",
            "receipt_id": "acquisition:receipt",
            "selected_action": action,
            "task_payoff": task_payoff,
            "transition_id": "acquisition:transition",
            "uniform_selector_sha256": None,
        },
        "arm": arm,
        "candidate_rows": candidates,
        "candidate_rows_sha256": canonical_sha256(candidates),
        "checkpoint": {
            "component_sha256": {
                "domain_custody": _SHA_A,
                "episode_accumulator": _SHA_B,
                "identity_counter": _SHA_C,
                "posterior_model": _SHA_D,
                "qualification_binding": _SHA_E,
            },
            "sha256": _SHA_F,
            "size_bytes": 123,
        },
        "counts": {
            "acquisition_updates": 1,
            "decisions": 2,
            "executions": 2,
            "experiences": 2,
            "terminal_updates": 0,
            "transitions": 2,
        },
        "entry_qualification_sha256": _SHA_E,
        "episode": episode,
        "implementation_sha256": _SHA_B,
        "master": master,
        "producer_pid": 100,
        "protocol_sha256": _SHA_A,
        "protocol_version": q1_audit.Q1_PROTOCOL_VERSION,
        "q0_report_sha256": Q0_REPORT_SHA256,
        "salt_commitment_sha256": schedule.salt_commitment_sha256,
        "schema": "prospect.wm002.active-acquisition.q1-raw-trace.v1",
        "semantic_key_sha256": schedule.theta_semantic_sha256(master, episode),
        "terminal": {
            "candidate_rows": terminal_candidates,
            "candidate_rows_sha256": canonical_sha256(terminal_candidates),
            "decision_id": "terminal:decision",
            "episode_return": episode_return,
            "execution_id": "terminal:execution",
            "experience_id": "terminal:experience",
            "intention_id": "terminal:intention",
            "observation_id": "terminal:observation",
            "outcome_id": "terminal:outcome",
            "selected_action": terminal_action,
            "success": success,
            "terminal_reward": float(success),
            "transition_id": "terminal:transition",
        },
    }
    private = schedule.private_row(master, episode, arm)
    private["run_id"] = _RUN_ID
    private["attempt_id"] = _ATTEMPT_ID
    return raw, private


def _restored_from_raw(raw: dict[str, object]) -> dict[str, object]:
    acquisition = raw["acquisition"]
    terminal = raw["terminal"]
    checkpoint = raw["checkpoint"]
    assert isinstance(acquisition, dict)
    assert isinstance(terminal, dict)
    assert isinstance(checkpoint, dict)
    component_sha256 = checkpoint["component_sha256"]
    assert isinstance(component_sha256, dict)
    return {
        "acquisition_experience_id": acquisition["experience_id"],
        "acquisition_receipt_id": acquisition["receipt_id"],
        "acquisition_transition_id": acquisition["transition_id"],
        "arm": raw["arm"],
        "checkpoint_sha256": checkpoint["sha256"],
        "component_sha256": component_sha256,
        "configuration_version": f"wm002-config-sha256:{acquisition['model_after_sha256']}",
        "entry_qualification_sha256": raw["entry_qualification_sha256"],
        "episode": raw["episode"],
        "episode_return": terminal["episode_return"],
        "identity_counter_sha256": component_sha256["identity_counter"],
        "implementation_sha256": raw["implementation_sha256"],
        "master": raw["master"],
        "attempt_id": raw["attempt_id"],
        "run_id": raw["run_id"],
        "model_sha256": acquisition["model_after_sha256"],
        "model_version": acquisition["model_after_version"],
        "posterior_direct": acquisition["posterior_after"],
        "producer_pid": raw["producer_pid"],
        "protocol_sha256": raw["protocol_sha256"],
        "protocol_version": raw["protocol_version"],
        "q0_report_sha256": raw["q0_report_sha256"],
        "restorer_pid": 101,
        "salt_commitment_sha256": raw["salt_commitment_sha256"],
        "schema": "prospect.wm002.active-acquisition.q1-restored-trace.v1",
        "selected_terminal_action": terminal["selected_action"],
        "terminal_candidate_rows_sha256": terminal["candidate_rows_sha256"],
        "terminal_candidate_rows": terminal["candidate_rows"],
        "terminal_decision_id": terminal["decision_id"],
        "terminal_execution_id": terminal["execution_id"],
        "terminal_experience_id": terminal["experience_id"],
        "terminal_intention_id": terminal["intention_id"],
        "terminal_observation_id": terminal["observation_id"],
        "terminal_outcome_id": terminal["outcome_id"],
        "terminal_success": terminal["success"],
        "terminal_transition_id": terminal["transition_id"],
    }


def test_auditor_cli_requires_python_no_site_mode() -> None:
    if q1_audit.sys.flags.no_site == 1:
        pytest.skip("test requires a site-enabled parent interpreter")
    with pytest.raises(SystemExit, match="Q1 independent audit requires invocation with Python -S"):
        q1_audit._require_no_site_cli()


def test_auditor_has_no_producer_or_analysis_import() -> None:
    source = Path(q1_audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    forbidden = {
        "bench.active_acquisition.attempt",
        "bench.active_acquisition.oracle",
        "bench.active_acquisition.policies",
        "bench.active_acquisition.q1",
        "bench.active_acquisition.q1_qualification",
        "bench.active_acquisition.runtime_lane",
        "bench.active_acquisition.seeding",
    }
    assert forbidden.isdisjoint(imported_modules)


def test_independent_seed_formulas_match_producer_contract() -> None:
    independent = q1_audit._IndependentSeedSchedule(_SALT)
    producer = PrivateQ1SeedSchedule(_SALT)
    for master in range(q1_audit.MASTER_COUNT):
        for episode in (0, 1, 511, 512, 1023):
            theta = int(producer.theta(master, episode))
            assert independent.theta(master, episode) == theta
            assert (
                independent.theta_semantic_sha256(master, episode)
                == producer.theta_reference(master, episode).semantic_key_sha256
            )
            assert independent.nuisance_observation(master, episode) == producer.nuisance_observation(master, episode)
            for action in ("weak", "strong", "overpowered"):
                assert independent.pulse_observation(master, episode, action, theta) == producer.pulse_observation(
                    master, episode, action, theta
                )
            for decision in (1, -1):
                assert independent.terminal_success(master, episode, decision, theta) == producer.terminal_success(
                    master, episode, decision, theta
                )
            producer_uniform = derive_public_uniform_selection(master, episode)
            assert independent.uniform_selection(master, episode) == (
                producer_uniform.action_id,
                producer_uniform.semantic_key_sha256,
            )


def test_result_free_audit_sample_matches_strict_schema() -> None:
    sample = q1_audit.development_audit_artifact_sample()
    validate_artifact("audit_output", sample)
    assert sample["claim_eligible"] is False
    assert sample["formal_authorized"] is False
    assert sample["passed"] is False
    assert sample["prospective_review_sha256"] == "0" * 64
    assert sample["worker_capability_sha256"] == "0" * 64

    missing_commitment = dict(sample)
    del missing_commitment["worker_capability_sha256"]
    with pytest.raises(ValueError, match="worker_capability_sha256.*required property"):
        validate_artifact("audit_output", missing_commitment)


def test_prospective_review_decoder_rejects_duplicate_keys_without_echo(tmp_path: Path) -> None:
    private_key = "recognizable-private-duplicate-review-key"
    path = tmp_path / "duplicate-review.json"
    path.write_text(
        f'{{"{private_key}":1,"{private_key}":1}}\n',
        encoding="utf-8",
    )

    with pytest.raises(q1_audit.Q1AuditError, match="JSON decoding failed") as captured:
        q1_audit._load_validated_prospective_review(
            path,
            protocol_sha256=_SHA_A,
            implementation_sha256=_SHA_B,
            reviewed_source_count=47,
        )
    assert private_key not in str(captured.value)


def test_prospective_review_is_reopened_canonical_schema_valid_and_exact(tmp_path: Path) -> None:
    review = _prospective_review()
    review_path = tmp_path / "prospective-review.json"
    payload = canonical_json_bytes(review, newline=True)
    review_path.write_bytes(payload)

    digest = q1_audit._load_validated_prospective_review(
        review_path,
        protocol_sha256=_SHA_A,
        implementation_sha256=_SHA_B,
        reviewed_source_count=47,
    )
    assert digest == hashlib.sha256(payload).hexdigest()

    noncanonical = tmp_path / "noncanonical-review.json"
    noncanonical.write_text(json.dumps(review, indent=2), encoding="utf-8")
    with pytest.raises(q1_audit.Q1AuditError, match="canonical JSON"):
        q1_audit._load_validated_prospective_review(
            noncanonical,
            protocol_sha256=_SHA_A,
            implementation_sha256=_SHA_B,
            reviewed_source_count=47,
        )

    symlink = tmp_path / "linked-review.json"
    symlink.symlink_to(review_path)
    with pytest.raises(q1_audit.Q1AuditError, match="cannot open Q1 prospective review"):
        q1_audit._load_validated_prospective_review(
            symlink,
            protocol_sha256=_SHA_A,
            implementation_sha256=_SHA_B,
            reviewed_source_count=47,
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    (
        ("protocol_sha256", _SHA_C),
        ("implementation_sha256", _SHA_C),
        ("reviewed_source_count", 46),
        ("review_scope", list(reversed(q1_audit._PROSPECTIVE_REVIEW_SCOPE))),
        ("review_method", "different-method"),
        ("assurance_boundary", "different-boundary"),
        ("q1_environment_interactions", 1),
        ("q1_private_draws", 1),
        ("claim_eligible", True),
        ("formal_authorized", True),
        ("passed", False),
        ("blocking_findings", ["unresolved blocker"]),
        ("nonblocking_findings", [1]),
        ("nonblocking_findings", [""]),
    ),
)
def test_prospective_review_rejects_every_required_semantic_mutation(
    tmp_path: Path,
    field: str,
    mutation: object,
) -> None:
    review = _prospective_review()
    review[field] = mutation
    path = tmp_path / f"review-{field}.json"
    path.write_bytes(canonical_json_bytes(review, newline=True))

    with pytest.raises(q1_audit.Q1AuditError):
        q1_audit._load_validated_prospective_review(
            path,
            protocol_sha256=_SHA_A,
            implementation_sha256=_SHA_B,
            reviewed_source_count=47,
        )


def test_prospective_review_schema_diagnostic_never_echoes_private_value(tmp_path: Path) -> None:
    private_text = _SALT[:16].hex()
    review = _prospective_review()
    review[private_text] = private_text
    path = tmp_path / "private-valued-review.json"
    path.write_bytes(canonical_json_bytes(review, newline=True))

    with pytest.raises(q1_audit.Q1AuditError) as captured:
        q1_audit._load_validated_prospective_review(
            path,
            protocol_sha256=_SHA_A,
            implementation_sha256=_SHA_B,
            reviewed_source_count=47,
        )
    assert private_text not in str(captured.value)


def test_entry_must_bind_exact_reopened_prospective_review_digest() -> None:
    digest = hashlib.sha256(b"review").hexdigest()
    q1_audit._require_entry_prospective_review_binding(
        {"prospective_review_sha256": digest},
        digest,
    )
    with pytest.raises(q1_audit.Q1AuditError, match="differs from reopened review bytes"):
        q1_audit._require_entry_prospective_review_binding(
            {"prospective_review_sha256": hashlib.sha256(b"other").hexdigest()},
            digest,
        )


def test_completed_audit_private_scan_refuses_publication_without_echo() -> None:
    private_text = _SALT[:16].hex()
    artifact = q1_audit.development_audit_artifact_sample()
    artifact["scope_limitations"] = [f"malformed-{private_text}-diagnostic"]
    scanner = q1_audit.PrivatePrefixScanner.from_private_values((_SALT,))

    with pytest.raises(q1_audit.Q1AuditError) as captured:
        q1_audit._require_completed_audit_private_clean(artifact, scanner)
    assert "private-prefix material" in str(captured.value)
    assert private_text not in str(captured.value)


def test_auditor_cli_requires_prospective_review_path() -> None:
    arguments = [
        "artifact-directory",
        "--secret-salt",
        "salt.bin",
        "--attempt-marker",
        q1_audit._ATTEMPT_MARKER_FILENAME,
        "--q0-report",
        "q0.json",
        "--entry-report",
        "entry.json",
        "--output",
        "audit.json",
    ]
    with pytest.raises(SystemExit):
        q1_audit._build_parser().parse_args(arguments)
    parsed = q1_audit._build_parser().parse_args([*arguments, "--prospective-review", "review.json"])
    assert parsed.prospective_review == Path("review.json")


def test_audit_count_schema_rejects_missing_and_undeclared_fields() -> None:
    undeclared = q1_audit.development_audit_artifact_sample()
    counts = undeclared["counts"]
    assert isinstance(counts, dict)
    counts["undeclared_count"] = 0
    with pytest.raises(ValueError, match="Additional properties are not allowed"):
        validate_artifact("audit_output", undeclared)

    missing = q1_audit.development_audit_artifact_sample()
    counts = missing["counts"]
    assert isinstance(counts, dict)
    del counts["masters"]
    with pytest.raises(ValueError, match=".masters. is a required property"):
        validate_artifact("audit_output", missing)


def test_episode_semantics_recompute_without_producer_aggregate() -> None:
    raw, private = _raw_and_private()
    violations, episode_return = q1_audit._episode_semantic_violations(
        raw,
        private,
        q1_audit._IndependentSeedSchedule(_SALT),
    )
    assert violations == []
    terminal = raw["terminal"]
    assert isinstance(terminal, dict)
    assert episode_return == terminal["episode_return"]


def test_mutation_tampered_primitive_return_is_rejected() -> None:
    raw, private = _raw_and_private()
    terminal = raw["terminal"]
    assert isinstance(terminal, dict)
    terminal["episode_return"] = float(terminal["episode_return"]) + 1.0
    violations, _ = q1_audit._episode_semantic_violations(
        raw,
        private,
        q1_audit._IndependentSeedSchedule(_SALT),
    )
    assert ("Q1-K1", "episode return differs from primitive executed reconstruction") in violations


def test_mutation_hidden_field_is_rejected_by_frozen_schema() -> None:
    raw, _ = _raw_and_private()
    validate_artifact("raw_trace", raw)
    raw["theta"] = 1
    with pytest.raises(Exception, match="schema"):
        validate_artifact("raw_trace", raw)


def test_mutation_posterior_is_rejected() -> None:
    raw, private = _raw_and_private()
    acquisition = raw["acquisition"]
    assert isinstance(acquisition, dict)
    acquisition["posterior_after"] = "1/2"
    violations, _ = q1_audit._episode_semantic_violations(
        raw,
        private,
        q1_audit._IndependentSeedSchedule(_SALT),
    )
    assert ("Q1-K3", "posterior_after differs from true executed-action likelihood") in violations


def test_mutation_restored_row_and_terminal_identity_are_rejected() -> None:
    raw, _ = _raw_and_private()
    restored = _restored_from_raw(raw)
    violations = q1_audit._Violations()
    q1_audit._audit_restored_parity(
        key=(0, "prospect_expected_return", 0),
        raw=raw,
        restored=restored,
        violations=violations,
    )
    assert violations.rows("Q1-K5") == ()

    original_return = restored["episode_return"]
    assert isinstance(original_return, (int, float))
    restored["episode_return"] = float(original_return) + 1.0
    restored["terminal_transition_id"] = "tampered"
    q1_audit._audit_restored_parity(
        key=(0, "prospect_expected_return", 0),
        raw=raw,
        restored=restored,
        violations=violations,
    )
    assert any("restored/live parity mismatch" in row for row in violations.rows("Q1-K5"))


def test_mutation_secret_salt_breaks_private_recomputation() -> None:
    raw, private = _raw_and_private()
    violations, _ = q1_audit._episode_semantic_violations(
        raw,
        private,
        q1_audit._IndependentSeedSchedule(b"another-independent-audit-salt00"),
    )
    assert any(gate == "Q1-K1" and "reconstructed" in message for gate, message in violations)


def test_df3_interval_is_recomputed_from_four_master_means() -> None:
    returns: dict[tuple[int, str], list[float]] = {}
    for master in range(4):
        for arm in q1_audit.ARM_ORDER:
            value = 0.7 if arm in {"prospect_expected_return", "independent_fraction_oracle"} else 0.5
            returns[(master, arm)] = [value] * 1024
    violations = q1_audit._Violations()
    comparisons = q1_audit._recompute_comparisons(returns, violations)
    assert violations.rows("Q1-K4") == ()
    assert [row["control_arm"] for row in comparisons] == list(q1_audit.CONTROL_ARMS)
    assert all(row["passed"] is True for row in comparisons)
    assert all(row["master_differences"] == pytest.approx([0.2] * 4) for row in comparisons)


def test_schema_valid_producer_aggregate_needs_no_passed_boolean() -> None:
    returns: dict[tuple[int, str], list[float]] = {}
    for master in range(q1_audit.MASTER_COUNT):
        for arm in q1_audit.ARM_ORDER:
            value = 0.7 if arm in {"prospect_expected_return", "independent_fraction_oracle"} else 0.5
            returns[(master, arm)] = [value] * q1_audit.EPISODES_PER_MASTER

    setup_violations = q1_audit._Violations()
    arm_means, means = q1_audit._recompute_arm_means(returns, setup_violations)
    comparisons = q1_audit._comparison_rows(means)
    assert setup_violations.rows("Q1-K4") == ()
    producer_comparisons = [
        {
            key: row[key]
            for key in (
                "control_arm",
                "master_differences",
                "mean_difference",
                "ci95_lower",
                "ci95_upper",
            )
        }
        for row in comparisons
    ]
    assert all("passed" not in row for row in producer_comparisons)
    aggregate = {
        "schema": "prospect.wm002.active-acquisition.q1-aggregate.v1",
        "protocol_version": q1_audit.Q1_PROTOCOL_VERSION,
        "run_id": _RUN_ID,
        "attempt_id": _ATTEMPT_ID,
        "protocol_sha256": _SHA_A,
        "implementation_sha256": _SHA_B,
        "entry_qualification_sha256": _SHA_C,
        "q0_report_sha256": Q0_REPORT_SHA256,
        "salt_commitment_sha256": _SHA_D,
        "claim_eligible": False,
        "formal_authorized": False,
        "producer_analysis_authoritative": False,
        "counts": {
            "masters": q1_audit.MASTER_COUNT,
            "arms": len(q1_audit.ARM_ORDER),
            "episodes": q1_audit.EXPECTED_EPISODES,
            "environment_steps": q1_audit.EXPECTED_EPISODES * 2,
            "transitions": q1_audit.EXPECTED_EPISODES * 2,
            "acquisition_updates": q1_audit.EXPECTED_EPISODES,
            "terminal_updates": 0,
            "checkpoints": q1_audit.EXPECTED_EPISODES,
            "restores": q1_audit.EXPECTED_EPISODES,
        },
        "arm_means": arm_means,
        "comparisons": producer_comparisons,
    }
    validate_artifact("aggregate", aggregate)

    audit_counts = {
        "acquisition_updates": q1_audit.EXPECTED_EPISODES,
        "arms": len(q1_audit.ARM_ORDER),
        "checkpoint_frames": q1_audit.EXPECTED_EPISODES,
        "environment_steps": q1_audit.EXPECTED_EPISODES * 2,
        "episodes": q1_audit.EXPECTED_EPISODES,
        "masters": q1_audit.MASTER_COUNT,
        "private_rows": q1_audit.EXPECTED_EPISODES,
        "raw_rows": q1_audit.EXPECTED_EPISODES,
        "restored_rows": q1_audit.EXPECTED_EPISODES,
        "terminal_updates": 0,
        "transitions": q1_audit.EXPECTED_EPISODES * 2,
    }
    violations = q1_audit._Violations()
    q1_audit._validate_producer_aggregate(
        aggregate,
        counts=audit_counts,
        arm_means=arm_means,
        comparisons=comparisons,
        violations=violations,
    )
    assert violations.rows("Q1-K0") == ()
    assert violations.rows("Q1-K4") == ()


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("producer_stage_timeout_seconds", 3599.0),
        ("restore_child_timeout_seconds", 900),
        ("restore_stage_timeout_seconds", 7199.0),
        ("process_terminate_grace_seconds", 9.0),
        ("rule", "mutated watchdog rule"),
    ],
)
def test_external_bindings_reject_process_watchdog_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    mutation: float | int | str,
) -> None:
    protocol = json.loads(q1_audit.Q1_PROTOCOL_PATH.read_text(encoding="utf-8"))
    experiment = protocol["experiment"]
    assert isinstance(experiment, dict)
    experiment["execution_authorized"] = True
    runtime = protocol["runtime"]
    assert isinstance(runtime, dict)
    process_watchdogs = runtime["process_watchdogs"]
    assert isinstance(process_watchdogs, dict)
    process_watchdogs[field] = mutation
    protocol_path = tmp_path / "q1_protocol.json"
    protocol_path.write_bytes(canonical_json_bytes(protocol, newline=True))
    monkeypatch.setattr(q1_audit, "Q1_PROTOCOL_PATH", protocol_path)

    violations = q1_audit._Violations()
    q1_audit._resolve_external_bindings(
        protocol_path=protocol_path,
        q0_report_path=tmp_path / "missing-q0.json",
        entry_report_path=tmp_path / "missing-entry.json",
        prospective_review_path=tmp_path / "missing-review.json",
        salt_commitment=_SHA_D,
        violations=violations,
    )
    assert any(
        f"canonical protocol process watchdog mismatch:{field}" in message for message in violations.rows("Q1-K0")
    )


def test_missing_directory_fails_closed_with_schema_valid_output(tmp_path: Path) -> None:
    salt_path = tmp_path / "salt.bin"
    salt_path.write_bytes(_SALT)
    salt_path.chmod(0o600)
    artifact = q1_audit.audit_q1_directory(
        tmp_path,
        attempt_marker_path=tmp_path / "missing-attempt.json",
        secret_salt_path=salt_path,
        q0_report_path=tmp_path / "missing-q0.json",
        entry_report_path=tmp_path / "missing-entry.json",
        prospective_review_path=tmp_path / "missing-review.json",
    )
    validate_artifact("audit_output", artifact)
    assert artifact["passed"] is False
    assert artifact["accepted_prefix"] == []


def test_selected_source_closure_exactly_matches_entry_qualification() -> None:
    from bench.active_acquisition.q1_qualification import Q1_IMPLEMENTATION_PATHS

    assert q1_audit.Q1_AUDIT_IMPLEMENTATION_PATHS == Q1_IMPLEMENTATION_PATHS
    manifest, digest = q1_audit.implementation_manifest(q1_audit.Q1_AUDIT_IMPLEMENTATION_PATHS)
    assert [row.relative_path for row in manifest] == sorted(q1_audit.Q1_AUDIT_IMPLEMENTATION_PATHS)
    assert len(digest) == 64


def test_loaded_module_origins_are_bound_to_selected_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    q1_audit._validate_loaded_source_origins()
    prospect_module = q1_audit.sys.modules["prospect"]
    monkeypatch.setattr(prospect_module, "__file__", str(tmp_path / "alternate-prospect.py"))
    with pytest.raises(q1_audit.Q1AuditError, match="origin differs from hashed source"):
        q1_audit._validate_loaded_source_origins()


def test_auditor_rejects_optimized_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q1_audit.sys, "flags", SimpleNamespace(optimize=1))
    with pytest.raises(q1_audit.Q1AuditError, match="unoptimized Python interpreter"):
        q1_audit._validate_loaded_source_origins()


def _make_exact_publication(directory: Path) -> None:
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    for name in q1_audit._ARTIFACT_FILENAMES:
        path = directory / name
        path.write_bytes(b"\n")
        path.chmod(0o600 if name == q1_audit.PRIVATE_AUDIT_FILENAME else 0o644)


def test_publication_set_rejects_extra_and_symlink_entries(tmp_path: Path) -> None:
    exact = tmp_path / "exact"
    _make_exact_publication(exact)
    q1_audit._validate_artifact_directory(exact)

    extra = tmp_path / "extra"
    _make_exact_publication(extra)
    (extra / "seventh.txt").write_text("unbound", encoding="utf-8")
    with pytest.raises(q1_audit.Q1AuditError, match="exact six"):
        q1_audit._validate_artifact_directory(extra)

    linked = tmp_path / "linked"
    _make_exact_publication(linked)
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    replaced = linked / q1_audit.RAW_TRACE_FILENAME
    replaced.unlink()
    replaced.symlink_to(target)
    with pytest.raises(q1_audit.Q1AuditError, match="non-symlink regular"):
        q1_audit._validate_artifact_directory(linked)


def test_auditor_regular_readers_reject_hard_links_and_rename_away(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_text = "recognizable-private-auditor-reader-value"
    path = tmp_path / "document.json"
    backup = tmp_path / "document-backup.json"
    replacement = tmp_path / "document-replacement.json"
    path.write_bytes(canonical_json_bytes({"private": private_text}, newline=True))
    replacement.write_bytes(canonical_json_bytes({"safe": True}, newline=True))

    hard_link = tmp_path / "document-hard-link.json"
    os.link(path, hard_link)
    with pytest.raises(q1_audit.Q1AuditError, match="exactly one hard link"):
        q1_audit._read_regular_file(path, label="hard-linked document")
    hard_link.unlink()

    real_fstat = q1_audit.os.fstat
    calls = 0

    def swap_after_final_descriptor_stat(descriptor: int) -> os.stat_result:
        nonlocal calls
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 3:
            path.rename(backup)
            replacement.rename(path)
            assert real_fstat(descriptor).st_nlink == 1
        return metadata

    monkeypatch.setattr(q1_audit.os, "fstat", swap_after_final_descriptor_stat)
    with pytest.raises(q1_audit.Q1AuditError, match="path differs from its opened descriptor") as captured:
        q1_audit._read_regular_file(path, label="rename-away document")
    assert private_text not in str(captured.value)


def test_auditor_stream_hash_rejects_rename_away(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.bin"
    backup = tmp_path / "artifact-backup.bin"
    replacement = tmp_path / "artifact-replacement.bin"
    path.write_bytes(b"first-artifact")
    replacement.write_bytes(b"second-artifact")
    real_fstat = q1_audit.os.fstat
    calls = 0

    def swap_after_final_descriptor_stat(descriptor: int) -> os.stat_result:
        nonlocal calls
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 3:
            path.rename(backup)
            replacement.rename(path)
        return metadata

    monkeypatch.setattr(q1_audit.os, "fstat", swap_after_final_descriptor_stat)
    with pytest.raises(q1_audit.Q1AuditError, match="path differs from its opened descriptor"):
        q1_audit._stream_file_sha256(path, label="rename-away artifact")


def test_publication_set_rejects_hard_linked_entries(tmp_path: Path) -> None:
    directory = tmp_path / "hard-linked-publication"
    _make_exact_publication(directory)
    outside = tmp_path / "outside-hard-link.bin"
    os.link(directory / q1_audit.RAW_TRACE_FILENAME, outside)

    with pytest.raises(q1_audit.Q1AuditError, match="exactly one hard link"):
        q1_audit._validate_artifact_directory(directory)


def test_streamed_checkpoint_frames_are_sequential_and_trailing_bytes_are_visible() -> None:
    first = b"first-frame"
    second = b"second-frame"
    first_frame = len(first).to_bytes(8, "big") + first
    second_frame = len(second).to_bytes(8, "big") + second
    stream = io.BytesIO(first_frame + second_frame + b"!")
    digest = hashlib.sha256()
    violations = q1_audit._Violations()
    first_index = {"frame_offset": 0, "frame_header_bytes": 8, "frame_length": len(first)}
    payload, offset = q1_audit._read_streamed_checkpoint(
        stream=stream,
        digest=digest,
        index=first_index,
        expected_offset=0,
        key=(0, q1_audit.ARM_ORDER[0], 0),
        violations=violations,
    )
    assert payload == first
    second_index = {"frame_offset": offset, "frame_header_bytes": 8, "frame_length": len(second)}
    payload, offset = q1_audit._read_streamed_checkpoint(
        stream=stream,
        digest=digest,
        index=second_index,
        expected_offset=offset,
        key=(0, q1_audit.ARM_ORDER[0], 1),
        violations=violations,
    )
    assert payload == second
    assert offset == len(first_frame) + len(second_frame)
    trailing_bytes, trailing_rows = q1_audit._consume_hashed_remainder(stream, digest)
    assert (trailing_bytes, trailing_rows) == (1, 1)
    assert violations.rows("Q1-K0") == ()

    mutated = io.BytesIO(first_frame)
    mutated_violations = q1_audit._Violations()
    q1_audit._read_streamed_checkpoint(
        stream=mutated,
        digest=hashlib.sha256(),
        index={"frame_offset": 0, "frame_header_bytes": 8, "frame_length": len(first) + 1},
        expected_offset=0,
        key=(0, q1_audit.ARM_ORDER[0], 0),
        violations=mutated_violations,
    )
    assert any("header differs" in row for row in mutated_violations.rows("Q1-K0"))


def test_streaming_auditor_has_no_full_sidecar_materializer() -> None:
    source = Path(q1_audit.__file__).read_text(encoding="utf-8")
    assert "class _ArtifactInputs" not in source
    assert "def _load_inputs" not in source
    assert "checkpoint_frames: bytes" not in source
    assert ".read_bytes()" not in source


def test_global_private_scanner_rejects_cross_episode_transplant() -> None:
    schedule = q1_audit._IndependentSeedSchedule(_SALT)
    scanner = q1_audit.PrivatePrefixScanner.from_private_values(q1_audit._global_private_values(schedule, _SALT))
    transplanted = schedule.private_hmac_digests(0, 0, q1_audit.ARM_ORDER[0])[0].hex()
    leaks = scanner.scan({"episode": 1, "allowed_identifier": f"prefix-{transplanted}-suffix"})
    assert leaks


def test_restorer_pid_reuse_across_lanes_is_legal() -> None:
    violations = q1_audit._Violations()
    q1_audit._validate_stream_process_partition(
        producer_by_master={master: {100 + master} for master in range(q1_audit.MASTER_COUNT)},
        restorer_by_lane={
            (master, arm): {999} for master in range(q1_audit.MASTER_COUNT) for arm in q1_audit.ARM_ORDER
        },
        violations=violations,
    )
    assert violations.rows("Q1-K0") == ()
    assert violations.rows("Q1-K5") == ()


def _updated_model_payload() -> bytes:
    return canonical_json_bytes(
        {
            "evidence_count": 1,
            "last_experience_id": "experience-1",
            "last_transition_id": "transition-1",
            "likelihood_version": q1_audit._LIKELIHOOD_VERSION,
            "posterior_direct": {"denominator": 10, "numerator": 9},
            "schema": q1_audit._POSTERIOR_MODEL_SCHEMA,
        }
    )


def test_local_posterior_decoder_is_strict_and_version_bound() -> None:
    decoded = q1_audit._decode_posterior_model(_updated_model_payload())
    assert decoded.evidence_count == 1
    assert decoded.posterior_direct == Fraction(9, 10)

    with pytest.raises(ValueError, match="canonical"):
        q1_audit._decode_posterior_model(b" " + _updated_model_payload())
    unreduced = canonical_json_bytes(
        {
            "evidence_count": 1,
            "last_experience_id": "experience-1",
            "last_transition_id": "transition-1",
            "likelihood_version": q1_audit._LIKELIHOOD_VERSION,
            "posterior_direct": {"denominator": 4, "numerator": 2},
            "schema": q1_audit._POSTERIOR_MODEL_SCHEMA,
        }
    )
    with pytest.raises(ValueError, match="reduced"):
        q1_audit._decode_posterior_model(unreduced)
    missing_lineage = canonical_json_bytes(
        {
            "evidence_count": 1,
            "last_experience_id": None,
            "last_transition_id": None,
            "likelihood_version": q1_audit._LIKELIHOOD_VERSION,
            "posterior_direct": {"denominator": 2, "numerator": 1},
            "schema": q1_audit._POSTERIOR_MODEL_SCHEMA,
        }
    )
    with pytest.raises(ValueError, match="lineage"):
        q1_audit._decode_posterior_model(missing_lineage)

    payload = _updated_model_payload()
    with pytest.raises(ValueError, match="model version"):
        q1_audit._auditor_model_validator(q1_audit.ModelState(version="wrong-version", payload=payload))


def _passing_audit_artifact() -> dict[str, object]:
    artifact = copy.deepcopy(q1_audit.development_audit_artifact_sample())
    artifact["comparisons"] = [
        {
            "ci95_lower": 1.0,
            "ci95_upper": 1.0,
            "control_arm": control,
            "master_differences": [1.0, 1.0, 1.0, 1.0],
            "mean_difference": 1.0,
            "passed": True,
        }
        for control in q1_audit.CONTROL_ARMS
    ]
    artifact["gates"] = [{"gate": gate, "passed": True, "violations": []} for gate in q1_audit.GATE_ORDER]
    artifact["accepted_prefix"] = list(q1_audit.GATE_ORDER)
    artifact["passed"] = True
    artifact["counts"] = {
        "acquisition_updates": q1_audit.EXPECTED_EPISODES,
        "arms": len(q1_audit.ARM_ORDER),
        "checkpoint_frames": q1_audit.EXPECTED_EPISODES,
        "environment_steps": q1_audit.EXPECTED_EPISODES * 2,
        "episodes": q1_audit.EXPECTED_EPISODES,
        "masters": q1_audit.MASTER_COUNT,
        "private_rows": q1_audit.EXPECTED_EPISODES,
        "raw_rows": q1_audit.EXPECTED_EPISODES,
        "restored_rows": q1_audit.EXPECTED_EPISODES,
        "terminal_updates": 0,
        "transitions": q1_audit.EXPECTED_EPISODES * 2,
    }
    artifact["worker_capability_sha256"] = _SHA_F
    artifact["verdict"] = "PASS: synthetic semantic validator fixture."
    return artifact


def test_audit_output_semantics_reject_schema_valid_contradictions() -> None:
    artifact = _passing_audit_artifact()
    validate_artifact("audit_output", artifact)
    q1_audit._validate_audit_semantics(artifact)

    zero_worker_commitment = copy.deepcopy(artifact)
    zero_worker_commitment["worker_capability_sha256"] = "0" * 64
    validate_artifact("audit_output", zero_worker_commitment)
    with pytest.raises(q1_audit.Q1AuditError, match="zero worker capability commitment"):
        q1_audit._validate_audit_semantics(zero_worker_commitment)

    missing_review_binding = copy.deepcopy(artifact)
    del missing_review_binding["prospective_review_sha256"]
    with pytest.raises(ValueError, match="prospective_review_sha256.*required property"):
        validate_artifact("audit_output", missing_review_binding)
    with pytest.raises(q1_audit.Q1AuditError, match="prospective-review digest"):
        q1_audit._validate_audit_semantics(missing_review_binding)

    contradictory_gate = copy.deepcopy(artifact)
    contradictory_gate["gates"][0]["violations"] = ["failure"]  # type: ignore[index]
    validate_artifact("audit_output", contradictory_gate)
    with pytest.raises(q1_audit.Q1AuditError, match="pass/violation"):
        q1_audit._validate_audit_semantics(contradictory_gate)

    duplicate_control = copy.deepcopy(artifact)
    duplicate_control["comparisons"][1]["control_arm"] = duplicate_control["comparisons"][0]["control_arm"]  # type: ignore[index]
    validate_artifact("audit_output", duplicate_control)
    with pytest.raises(q1_audit.Q1AuditError, match="control-arm order"):
        q1_audit._validate_audit_semantics(duplicate_control)

    failed_comparison = copy.deepcopy(artifact)
    failed_comparison["comparisons"][0].update(  # type: ignore[index]
        {
            "ci95_lower": -1.0,
            "ci95_upper": -1.0,
            "master_differences": [-1.0, -1.0, -1.0, -1.0],
            "mean_difference": -1.0,
            "passed": False,
        }
    )
    validate_artifact("audit_output", failed_comparison)
    with pytest.raises(q1_audit.Q1AuditError, match="passing Q1-K4"):
        q1_audit._validate_audit_semantics(failed_comparison)

    wrong_count = copy.deepcopy(artifact)
    wrong_count["counts"]["raw_rows"] = q1_audit.EXPECTED_EPISODES - 1  # type: ignore[index]
    validate_artifact("audit_output", wrong_count)
    with pytest.raises(q1_audit.Q1AuditError, match="frozen budget"):
        q1_audit._validate_audit_semantics(wrong_count)


def test_audit_writer_is_exclusive_canonical_and_durable(tmp_path: Path) -> None:
    artifact = q1_audit.development_audit_artifact_sample()
    audited_directory = tmp_path / "audited-artifacts"
    salt_path = tmp_path / "salt.bin"
    salt_path.write_bytes(_SALT)
    salt_path.chmod(0o600)
    path = tmp_path / "audit.json"
    previous_umask = os.umask(0o777)
    try:
        q1_audit.write_audit_artifact(
            path,
            artifact,
            secret_salt_path=salt_path,
            audited_directory=audited_directory,
        )
    finally:
        os.umask(previous_umask)
    assert path.read_bytes() == canonical_json_bytes(artifact, newline=True)
    assert (path.stat().st_mode & 0o777) == 0o644
    assert path.stat().st_nlink == 1
    with pytest.raises(q1_audit.Q1AuditError, match="already exists"):
        q1_audit.write_audit_artifact(
            path,
            artifact,
            secret_salt_path=salt_path,
            audited_directory=audited_directory,
        )

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target")
    with pytest.raises(q1_audit.Q1AuditError, match="already exists"):
        q1_audit.write_audit_artifact(
            dangling,
            artifact,
            secret_salt_path=salt_path,
            audited_directory=audited_directory,
        )

    private_text = _SALT[:16].hex()
    leaking = copy.deepcopy(artifact)
    leaking["scope_limitations"] = [f"malformed-{private_text}-diagnostic"]
    rejected_path = tmp_path / "private-leaking-audit.json"
    with pytest.raises(q1_audit.Q1AuditError, match="private-prefix material"):
        q1_audit.write_audit_artifact(
            rejected_path,
            leaking,
            secret_salt_path=salt_path,
            audited_directory=audited_directory,
        )
    assert not rejected_path.exists()


def test_audit_writer_rejects_output_inside_or_aliasing_evidence_and_inputs(tmp_path: Path) -> None:
    artifact = q1_audit.development_audit_artifact_sample()
    audited_directory = tmp_path / "artifacts"
    audited_directory.mkdir(mode=0o700)
    for filename in q1_audit._ARTIFACT_FILENAMES:
        (audited_directory / filename).write_bytes(b"evidence\n")

    salt_path = tmp_path / "salt.bin"
    salt_path.write_bytes(_SALT)
    salt_path.chmod(0o600)
    alias_directory = tmp_path / "artifact-alias"
    alias_directory.symlink_to(audited_directory, target_is_directory=True)

    for output_path in (audited_directory / "audit.json", alias_directory / "audit.json"):
        with pytest.raises(q1_audit.Q1AuditError, match="outside the audited artifact directory"):
            q1_audit.write_audit_artifact(
                output_path,
                artifact,
                secret_salt_path=salt_path,
                audited_directory=audited_directory,
            )
        assert not output_path.exists()

    outside_directory = tmp_path / "outside"
    outside_directory.mkdir(mode=0o700)
    salt_alias = outside_directory / "salt-alias.bin"
    os.link(salt_path, salt_alias)
    with pytest.raises(q1_audit.Q1AuditError, match="aliases a protected audit input"):
        q1_audit.write_audit_artifact(
            salt_alias,
            artifact,
            secret_salt_path=salt_path,
            audited_directory=audited_directory,
        )


def test_audit_writer_rejects_parent_replacement_after_identity_binding(tmp_path: Path) -> None:
    output_directory = tmp_path / "outside"
    output_directory.mkdir(mode=0o700)
    output_directory.chmod(0o700)
    target = output_directory / "audit.json"
    audited_directory = tmp_path / "artifacts"
    audited_directory.mkdir(mode=0o700)
    expected_identity = q1_audit._require_disjoint_audit_output_path(
        target,
        audited_directory=audited_directory,
        protected_input_paths=(),
    )

    moved_directory = tmp_path / "outside-original"
    output_directory.rename(moved_directory)
    output_directory.mkdir(mode=0o700)
    output_directory.chmod(0o700)

    with pytest.raises(q1_audit.Q1AuditError, match="identity changed"):
        q1_audit._write_exclusive_durable(
            target,
            b"{}\n",
            expected_parent_identity=expected_identity,
        )
    assert not target.exists()


def test_auditor_independently_parses_exact_directory_identity_objects(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    captured = q1_audit._capture_audit_directory_identity(
        directory,
        label="test private directory",
    )

    assert (
        q1_audit._parse_audit_directory_identity(
            captured.as_dict(),
            label="test identity",
        )
        == captured
    )

    for mutation in ("extra", "missing", "bool-stat", "wrong-type", "wrong-mode"):
        value = captured.as_dict()
        if mutation == "extra":
            value["extra"] = True
        elif mutation == "missing":
            del value["st_gid"]
        elif mutation == "bool-stat":
            value["st_dev"] = True
        elif mutation == "wrong-type":
            value["file_type"] = "regular"
        else:
            value["mode"] = "0755"
        with pytest.raises(q1_audit.Q1AuditError):
            q1_audit._parse_audit_directory_identity(value, label="test identity")

    noncanonical = captured.as_dict()
    noncanonical["canonical_path"] = f"{captured.canonical_path}/../{directory.name}"
    with pytest.raises(q1_audit.Q1AuditError, match="not canonical"):
        q1_audit._parse_audit_directory_identity(noncanonical, label="test identity")


def test_auditor_rejects_entry_bound_directory_replacement_at_same_path(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    declared = q1_audit._capture_audit_directory_identity(
        directory,
        label="test private directory",
    )
    displaced = tmp_path / "displaced"
    directory.rename(displaced)
    directory.mkdir(mode=0o700)

    with pytest.raises(q1_audit.Q1AuditError, match="differs from its entry-bound"):
        q1_audit._require_entry_bound_directory_identity(
            declared,
            label="test private directory",
        )


def test_attempt_marker_exactly_binds_six_reopened_artifacts(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    execution = tmp_path / "execution"
    registry.mkdir(mode=0o700)
    execution.mkdir(mode=0o700)
    registry.chmod(0o700)
    execution.chmod(0o700)
    run_sha256 = "6" * 64
    run_id = f"wm002-q1-{run_sha256}"
    bindings = q1_audit._AuditBindings(
        protocol_sha256=_SHA_A,
        implementation_sha256=_SHA_B,
        q0_report_sha256=Q0_REPORT_SHA256,
        entry_qualification_sha256=_SHA_C,
        prospective_review_sha256=_SHA_E,
        salt_commitment_sha256=_SHA_D,
        run_sha256=run_sha256,
        run_id=run_id,
        attempt_id=f"{run_id}-attempt-0001",
        execution_root=q1_audit._capture_audit_directory_identity(
            execution,
            label="test execution root",
        ),
        attempt_registry_directory=q1_audit._capture_audit_directory_identity(
            registry,
            label="test attempt registry",
        ),
    )
    digests = {
        "producer_aggregate": _SHA_A,
        "checkpoint_frames": _SHA_B,
        "checkpoint_index": _SHA_C,
        "private_audit": _SHA_D,
        "raw_trace": _SHA_E,
        "restored_trace": _SHA_F,
    }
    marker = {
        "artifact_sha256": {
            "aggregate": digests["producer_aggregate"],
            "checkpoint_frames": digests["checkpoint_frames"],
            "checkpoint_index": digests["checkpoint_index"],
            "private_audit": digests["private_audit"],
            "raw_trace": digests["raw_trace"],
            "restored_trace": digests["restored_trace"],
        },
        "attempt_id": bindings.attempt_id,
        "entry_qualification_sha256": bindings.entry_qualification_sha256,
        "expected_counts": q1_audit._ATTEMPT_EXPECTED_COUNTS,
        "implementation_sha256": bindings.implementation_sha256,
        "protocol_sha256": bindings.protocol_sha256,
        "protocol_version": q1_audit.Q1_PROTOCOL_VERSION,
        "q0_report_sha256": bindings.q0_report_sha256,
        "run_id": bindings.run_id,
        "run_sha256": bindings.run_sha256,
        "salt_commitment_sha256": bindings.salt_commitment_sha256,
        "worker_capability_sha256": _SHA_F,
        "schema": q1_audit._ATTEMPT_MARKER_SCHEMA,
        "status": "completed",
    }
    marker_path = registry / q1_audit._ATTEMPT_MARKER_FILENAME
    marker_path.write_bytes(canonical_json_bytes(marker, newline=True))
    marker_path.chmod(0o600)
    violations = q1_audit._Violations()
    marker_sha256, worker_capability_sha256 = q1_audit._validate_attempt_marker(
        marker_path, bindings, digests, violations
    )
    assert marker_sha256 == hashlib.sha256(marker_path.read_bytes()).hexdigest()
    assert worker_capability_sha256 == _SHA_F
    assert violations.rows("Q1-K0") == ()

    missing_commitment = dict(marker)
    del missing_commitment["worker_capability_sha256"]
    marker_path.write_bytes(canonical_json_bytes(missing_commitment, newline=True))
    marker_path.chmod(0o600)
    missing_commitment_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(marker_path, bindings, digests, missing_commitment_rejected)
    assert any("fields differ" in row for row in missing_commitment_rejected.rows("Q1-K0"))

    zero_commitment = dict(marker)
    zero_commitment["worker_capability_sha256"] = "0" * 64
    marker_path.write_bytes(canonical_json_bytes(zero_commitment, newline=True))
    marker_path.chmod(0o600)
    zero_commitment_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(marker_path, bindings, digests, zero_commitment_rejected)
    assert any("worker capability commitment is invalid" in row for row in zero_commitment_rejected.rows("Q1-K0"))

    marker_path.write_bytes(canonical_json_bytes(marker, newline=True))
    marker_path.chmod(0o600)

    legacy_name = registry / f"{run_id}.attempt.json"
    legacy_name.write_bytes(marker_path.read_bytes())
    legacy_name.chmod(0o600)
    legacy_name_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(legacy_name, bindings, digests, legacy_name_rejected)
    assert any("filename differs" in row for row in legacy_name_rejected.rows("Q1-K0"))

    marker["status"] = "running"
    marker_path.write_bytes(canonical_json_bytes(marker, newline=True))
    marker_path.chmod(0o600)
    rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(marker_path, bindings, digests, rejected)
    assert any("not durably completed" in row for row in rejected.rows("Q1-K0"))

    marker["status"] = "completed"
    artifact_sha256 = marker["artifact_sha256"]
    assert isinstance(artifact_sha256, dict)
    artifact_sha256["raw_trace"] = _SHA_A
    marker_path.write_bytes(canonical_json_bytes(marker, newline=True))
    marker_path.chmod(0o600)
    digest_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(marker_path, bindings, digests, digest_rejected)
    assert any("artifact digests differ" in row for row in digest_rejected.rows("Q1-K0"))

    artifact_sha256["raw_trace"] = digests["raw_trace"]
    marker_path.write_bytes(canonical_json_bytes(marker, newline=True))
    marker_path.chmod(0o644)
    permission_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(marker_path, bindings, digests, permission_rejected)
    assert any("exact private mode 0600" in row for row in permission_rejected.rows("Q1-K0"))

    wrong_name = registry / "wrong-name.attempt.json"
    wrong_name.write_bytes(canonical_json_bytes(marker, newline=True))
    wrong_name.chmod(0o600)
    filename_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(wrong_name, bindings, digests, filename_rejected)
    assert any("filename differs" in row for row in filename_rejected.rows("Q1-K0"))

    marker_path.write_bytes(canonical_json_bytes(marker, newline=True))
    marker_path.chmod(0o600)
    displaced_registry = tmp_path / "registry-displaced"
    registry.rename(displaced_registry)
    registry.mkdir(mode=0o700)
    replacement_marker = registry / q1_audit._ATTEMPT_MARKER_FILENAME
    replacement_marker.write_bytes(canonical_json_bytes(marker, newline=True))
    replacement_marker.chmod(0o600)
    substituted_registry_rejected = q1_audit._Violations()
    q1_audit._validate_attempt_marker(
        replacement_marker,
        bindings,
        digests,
        substituted_registry_rejected,
    )
    assert any("differs from its entry-bound" in row for row in substituted_registry_rejected.rows("Q1-K0"))


@pytest.mark.parametrize(
    "next_counter",
    (
        pytest.param(42, id="below-final"),
        pytest.param(44, id="above-final"),
        pytest.param(10**1000, id="enormous"),
    ),
)
def test_final_identity_counter_is_exact_before_collection_allocation(
    next_counter: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prospect.decision import CounterIdentitySource

    raw, _ = _raw_and_private()
    namespace = f"{_RUN_ID}:m0:a{q1_audit.ARM_ORDER[0]}:e0"
    source = CounterIdentitySource(namespace, next_counter=next_counter)
    payload = source.checkpoint_bytes()
    assert len(payload) < 2048
    assert json.loads(payload) == {
        "namespace": namespace,
        "next_counter": next_counter,
        "schema_version": 1,
    }
    restored = cast(
        q1_audit.RestoredQ1Checkpoint,
        SimpleNamespace(identity_source=source),
    )

    def fail_collection_allocation(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        raise AssertionError("invalid final counter reached range/set construction")

    monkeypatch.setattr(q1_audit, "range", fail_collection_allocation, raising=False)
    monkeypatch.setattr(q1_audit, "set", fail_collection_allocation, raising=False)

    assert q1_audit._identity_counter_violations(restored, raw) == [
        (
            "Q1-K1",
            "checkpoint identity next_counter must equal the exact preterminal value 43",
        )
    ]


def test_exact_terminal_ids_continue_from_checkpoint_counter() -> None:
    from prospect.decision import CounterIdentitySource

    raw, _ = _raw_and_private()
    namespace = f"{_RUN_ID}:m0:a{q1_audit.ARM_ORDER[0]}:e0"
    next_counter = 43
    source = CounterIdentitySource(namespace, next_counter=next_counter)
    checkpoint = raw["checkpoint"]
    terminal = raw["terminal"]
    assert isinstance(checkpoint, dict)
    assert isinstance(checkpoint["component_sha256"], dict)
    assert isinstance(terminal, dict)
    rows = terminal["candidate_rows"]
    assert isinstance(rows, list)
    rows[0]["prediction_id"] = f"{namespace}:prediction-terminal-direct:{next_counter}"
    rows[0]["assessment_id"] = f"{namespace}:assessment-terminal-direct:{next_counter + 4}"
    rows[1]["prediction_id"] = f"{namespace}:prediction-terminal-reversed:{next_counter + 5}"
    rows[1]["assessment_id"] = f"{namespace}:assessment-terminal-reversed:{next_counter + 9}"
    terminal.update(
        {
            "intention_id": f"{namespace}:intention:{next_counter + 10}",
            "decision_id": f"{namespace}:decision:{next_counter + 11}",
            "execution_id": f"{namespace}:execution-terminal:{next_counter + 12}",
            "observation_id": f"{namespace}:observation-terminal:{next_counter + 13}",
            "outcome_id": f"{namespace}:outcome-terminal:{next_counter + 15}",
            "experience_id": f"{namespace}:experience:{next_counter + 16}",
            "transition_id": f"{namespace}:transition:{next_counter + 24}",
        }
    )
    checkpoint["component_sha256"]["identity_counter"] = hashlib.sha256(source.checkpoint_bytes()).hexdigest()
    restored = SimpleNamespace(
        identity_source=source,
        snapshot=tuple(f"{namespace}:record:{counter}" for counter in range(next_counter)),
        experience=(),
        transition=(),
        receipt=(),
    )
    restored_checkpoint = cast(q1_audit.RestoredQ1Checkpoint, restored)
    assert q1_audit._identity_counter_violations(restored_checkpoint, raw) == []
    terminal["transition_id"] = f"{namespace}:transition:{next_counter + 23}"
    assert any(
        "terminal.transition_id" in message
        for _, message in q1_audit._identity_counter_violations(restored_checkpoint, raw)
    )


def test_result_free_real_checkpoint_passes_deep_independent_audit() -> None:
    from bench.active_acquisition import q1
    from bench.active_acquisition.attempt import derive_run_identity
    from bench.active_acquisition.checkpoint import QualificationBinding
    from bench.active_acquisition.runtime_lane import ArmMode

    protocol_sha256 = hashlib.sha256(b"deep-probe-protocol").hexdigest()
    implementation_sha256 = hashlib.sha256(b"deep-probe-implementation").hexdigest()
    q0_report_sha256 = hashlib.sha256(b"deep-probe-q0").hexdigest()
    entry_sha256 = hashlib.sha256(b"deep-probe-entry").hexdigest()
    salt_commitment_sha256 = hashlib.sha256(b"deep-probe-salt").hexdigest()
    identity = derive_run_identity(
        protocol_version=q1.Q1_PROTOCOL_VERSION,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        entry_qualification_sha256=entry_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    binding = QualificationBinding(
        protocol_version=q1.Q1_PROTOCOL_VERSION,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        entry_qualification_sha256=entry_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )
    artifact = q1._run_synthetic_episode(
        arm=ArmMode.PROSPECT,
        synthetic_ordinal=1,
        binding=binding,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        q0_report_sha256=q0_report_sha256,
        salt_commitment_sha256=salt_commitment_sha256,
    )
    schedule = q1_audit._IndependentSeedSchedule(_SALT)
    privacy_scanner = q1_audit.PrivatePrefixScanner.from_private_values(
        q1_audit._global_private_values(schedule, _SALT)
    )
    violations = q1_audit._Violations()
    q1_audit._audit_checkpoint_payload(
        key=(q1.MASTER_COUNT - 1, ArmMode.PROSPECT.value, q1.EPISODES_PER_MASTER - 1),
        raw=artifact.raw_trace,
        index=artifact.checkpoint_index,
        payload=artifact.checkpoint_payload,
        expected_binding=binding,
        privacy_scanner=privacy_scanner,
        violations=violations,
    )
    assert {gate: violations.rows(gate) for gate in q1_audit.GATE_ORDER if violations.rows(gate)} == {}
