"""Result-free Q0 semantic qualification for WM-002.

Q0 performs no environment interaction.  It compares Prospect's floating
epistemic path with an independently enumerated ``Fraction`` oracle, verifies
the declared controls, and binds the exact prospective protocol bytes.  A
passing report is deliberately claim-ineligible and grants no formal authority.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Final

from bench.active_acquisition.oracle import (
    ExactAcquisitionEvaluation,
    ExactOracleDecision,
    FractionOracle,
)
from bench.active_acquisition.policies import (
    SHUFFLED_INFORMATION_SOURCE_BY_ACTION,
    AcquisitionPolicyDecision,
    ActionScore,
    eig_only,
    goal_only,
    prospect_voi,
    raw_entropy,
    shuffled_information,
    uniform_random,
    uniform_selector_vector,
)
from bench.active_acquisition.policies import (
    oracle as oracle_policy,
)
from bench.active_acquisition.problem import (
    ACQUISITION_ACTIONS,
    NUISANCE_SCAN,
    OVERPOWERED_NEGATIVE,
    OVERPOWERED_POSITIVE,
    SKIP,
    STRONG_NEGATIVE,
    STRONG_POSITIVE,
    WEAK_NEGATIVE,
    WEAK_POSITIVE,
    AcquisitionAction,
    AcquisitionDiagnostics,
    AcquisitionResult,
    HiddenActuatorProblem,
    TerminalAction,
    TerminalResult,
)
from prospect import domain as prospect_domain
from prospect.domain import (
    Action,
    AgentSnapshot,
    Belief,
    BeliefUpdate,
    CandidateAssessment,
    DecisionRecord,
    Distribution,
    EpistemicEffect,
    EpistemicTarget,
    EpistemicTransition,
    EvaluationMetric,
    EvaluationRecord,
    Evidence,
    EvidenceLineage,
    ExecutedAction,
    ExperienceEvent,
    Goal,
    InformationSet,
    InformationValue,
    IntendedAction,
    Observation,
    Outcome,
    Prediction,
    ProperScore,
    Provenance,
    ResourceLedger,
    ResourceUse,
    TimePoint,
    UncertaintyEstimate,
    UpdateReceipt,
    Utility,
)

REPORT_SCHEMA: Final = "prospect.wm002.active-acquisition.q0-qualification.v1"
PROTOCOL_PATH: Final = Path(__file__).with_name("protocol.json")
TOLERANCE: Final = 1e-12
_HALF: Final = Fraction(1, 2)
_FORBIDDEN_PUBLIC_FIELD_FRAGMENTS: Final = ("theta", "regime", "latent")
_IMPLEMENTATION_BINDING_SCOPE: Final = (
    "Selected-source Q0 manifest; it identifies the executable benchmark and Prospect "
    "sources explicitly reviewed here, not a complete import, dependency, environment, "
    "or process-isolation closure."
)


@dataclass(frozen=True, slots=True)
class ScalarComparison:
    """One floating value compared with an exact rational reference."""

    name: str
    prospect_value: float
    oracle_fraction: str
    oracle_value: float
    absolute_error: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "absolute_error": self.absolute_error,
            "name": self.name,
            "oracle_fraction": self.oracle_fraction,
            "oracle_value": self.oracle_value,
            "passed": self.passed,
            "prospect_value": self.prospect_value,
        }


@dataclass(frozen=True, slots=True)
class FloatComparison:
    """One independently recomputed transcendental quantity."""

    name: str
    prospect_value: float
    independent_value: float
    absolute_error: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "absolute_error": self.absolute_error,
            "independent_value": self.independent_value,
            "name": self.name,
            "passed": self.passed,
            "prospect_value": self.prospect_value,
        }


@dataclass(frozen=True, slots=True)
class ActionQualificationRow:
    """Complete Q0 comparison for one of the five protocol actions."""

    action_id: str
    rational_comparisons: tuple[ScalarComparison, ...]
    raw_observation_entropy: FloatComparison
    information_gain: FloatComparison
    exact_hand_total: str
    cost_charged_once: bool
    passed: bool

    @property
    def maximum_absolute_error(self) -> float:
        return max(
            (
                *(row.absolute_error for row in self.rational_comparisons),
                self.raw_observation_entropy.absolute_error,
                self.information_gain.absolute_error,
            ),
            default=0.0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "cost_charged_once": self.cost_charged_once,
            "exact_hand_total": self.exact_hand_total,
            "information_gain": self.information_gain.as_dict(),
            "maximum_absolute_error": self.maximum_absolute_error,
            "passed": self.passed,
            "rational_comparisons": [row.as_dict() for row in self.rational_comparisons],
            "raw_observation_entropy": self.raw_observation_entropy.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class SelectorQualificationRow:
    """Required action identity for one deterministic selector."""

    policy: str
    expected_action: str
    selected_action: str
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_action": self.expected_action,
            "passed": self.passed,
            "policy": self.policy,
            "selected_action": self.selected_action,
        }


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    """One result-free Q0 predicate."""

    name: str
    passed: bool
    summary: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class SchemaCoverageRow:
    """Field-name isolation coverage for one real public record type."""

    record_type: str
    checked_fields: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "checked_fields": list(self.checked_fields),
            "forbidden_fields": list(self.forbidden_fields),
            "passed": self.passed,
            "record_type": self.record_type,
        }


@dataclass(frozen=True, slots=True)
class ImplementationBindingRow:
    """Digest of one executable input covered by this qualification."""

    relative_path: str
    sha256: str
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _ExactSemanticCell:
    """Auditor-private exact cell; never serialized in the public Q0 report.

    Q0 publishes only the matrix count, canonical digest, exact action totals,
    and any semantic violations derived from these rows.
    """

    state: int
    action_id: str
    observation: int
    terminal_decision: int
    terminal_success: bool
    path_probability: Fraction
    posterior_direct: Fraction
    immediate_payoff: Fraction
    action_cost: Fraction
    acquisition_cost: Fraction
    terminal_outcome_probability: Fraction
    realized_return: Fraction

    def canonical_dict(self) -> dict[str, object]:
        return {
            "acquisition_cost": str(self.acquisition_cost),
            "action_cost": str(self.action_cost),
            "action_id": self.action_id,
            "immediate_payoff": str(self.immediate_payoff),
            "observation": self.observation,
            "path_probability": str(self.path_probability),
            "posterior_direct": str(self.posterior_direct),
            "realized_return": str(self.realized_return),
            "state": self.state,
            "terminal_decision": self.terminal_decision,
            "terminal_outcome_probability": str(self.terminal_outcome_probability),
            "terminal_success": self.terminal_success,
        }


@dataclass(frozen=True, slots=True)
class QualificationReport:
    """Deterministic canonical Q0 report."""

    schema: str
    protocol_sha256: str
    oracle_sha256: str
    implementation_sha256: str
    implementation_manifest: tuple[ImplementationBindingRow, ...]
    implementation_binding_violations: tuple[str, ...]
    implementation_binding_scope: str
    interpreter_identity: str
    protocol_version: str
    tolerance: float
    claim_eligible: bool
    formal_authorized: bool
    environment_interactions: int
    action_rows: tuple[ActionQualificationRow, ...]
    selector_rows: tuple[SelectorQualificationRow, ...]
    schema_coverage_rows: tuple[SchemaCoverageRow, ...]
    uncovered_schema_types: tuple[str, ...]
    oracle_independence_violations: tuple[str, ...]
    semantic_matrix_cell_count: int
    semantic_matrix_sha256: str
    semantic_matrix_action_totals: tuple[str, ...]
    semantic_matrix_violations: tuple[str, ...]
    protocol_parity_violations: tuple[str, ...]
    uniform_vector_count: int
    uniform_vectors_sha256: str
    uniform_vector_violations: tuple[str, ...]
    checks: tuple[QualificationCheck, ...]
    maximum_absolute_error: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "action_rows": [row.as_dict() for row in self.action_rows],
            "checks": [check.as_dict() for check in self.checks],
            "claim_eligible": self.claim_eligible,
            "environment_interactions": self.environment_interactions,
            "formal_authorized": self.formal_authorized,
            "implementation_binding_scope": self.implementation_binding_scope,
            "implementation_binding_violations": list(self.implementation_binding_violations),
            "implementation_manifest": [row.as_dict() for row in self.implementation_manifest],
            "implementation_sha256": self.implementation_sha256,
            "interpreter_identity": self.interpreter_identity,
            "maximum_absolute_error": self.maximum_absolute_error,
            "oracle_independence_violations": list(self.oracle_independence_violations),
            "oracle_sha256": self.oracle_sha256,
            "passed": self.passed,
            "protocol_parity_violations": list(self.protocol_parity_violations),
            "protocol_sha256": self.protocol_sha256,
            "protocol_version": self.protocol_version,
            "schema": self.schema,
            "schema_coverage_rows": [row.as_dict() for row in self.schema_coverage_rows],
            "semantic_matrix_action_totals": list(self.semantic_matrix_action_totals),
            "semantic_matrix_cell_count": self.semantic_matrix_cell_count,
            "semantic_matrix_sha256": self.semantic_matrix_sha256,
            "semantic_matrix_violations": list(self.semantic_matrix_violations),
            "selector_rows": [row.as_dict() for row in self.selector_rows],
            "tolerance": self.tolerance,
            "uncovered_schema_types": list(self.uncovered_schema_types),
            "uniform_vector_count": self.uniform_vector_count,
            "uniform_vector_violations": list(self.uniform_vector_violations),
            "uniform_vectors_sha256": self.uniform_vectors_sha256,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


_HAND_TOTALS: Final = {
    SKIP: Fraction(1, 2),
    WEAK_POSITIVE: Fraction(63, 100),
    STRONG_POSITIVE: Fraction(37, 50),
    OVERPOWERED_POSITIVE: Fraction(9, 20),
    NUISANCE_SCAN: Fraction(49, 100),
}


def run_qualification(protocol_path: Path = PROTOCOL_PATH) -> QualificationReport:
    """Run Q0 without constructing or stepping an environment."""

    protocol_bytes = protocol_path.read_bytes()
    protocol_digest = sha256(protocol_bytes).hexdigest()
    oracle_digest, oracle_independence_violations = _oracle_independence()
    (
        implementation_manifest,
        implementation_digest,
        implementation_binding_violations,
    ) = _implementation_binding()
    protocol, protocol_parse_passed = _parse_protocol(protocol_bytes)
    protocol_version = _nested_text(protocol, "experiment", "protocol_version")
    protocol_identity_passed = (
        protocol_parse_passed
        and protocol.get("schema") == "prospect.wm002.active-acquisition.qualification-protocol.v0"
        and _nested_text(protocol, "experiment", "id") == "WM-002"
        and protocol_version == "0.2.0-q"
    )
    prospective_boundary_passed = (
        protocol_parse_passed
        and _nested_value(protocol, "q0", "claim_eligible") is False
        and _nested_value(protocol, "q0", "environment_interactions") == 0
        and _nested_value(protocol, "budgets", "q0_environment_steps") == 0
        and _nested_value(protocol, "experiment", "formal_authorized") is False
        and _nested_value(protocol, "formal_boundary", "authorized") is False
        and _nested_value(protocol, "formal_boundary", "formal_seed_set") == []
    )
    candidate_ids = tuple(action.action_id for action in ACQUISITION_ACTIONS)
    candidate_identity_passed = (
        candidate_ids == ("skip", "weak", "strong", "overpowered", "nuisance")
        and _protocol_candidate_ids(protocol) == candidate_ids
    )

    problem = HiddenActuatorProblem()
    exact = FractionOracle()
    semantic_cells = _exact_semantic_matrix(exact)
    semantic_digest = _semantic_matrix_digest(semantic_cells)
    semantic_matrix_violations, semantic_action_totals = _semantic_matrix_violations(semantic_cells, exact, protocol)
    protocol_parity_violations = _protocol_parity_violations(protocol, problem, exact)
    uniform_vector_count, uniform_vectors_digest, uniform_vector_violations = _uniform_vector_checks(protocol, problem)
    action_rows = tuple(_qualify_action(problem, exact, action) for action in ACQUISITION_ACTIONS)
    exact_float_matrix_passed = len(action_rows) == 5 and all(row.passed for row in action_rows)
    maximum_error = max((row.maximum_absolute_error for row in action_rows), default=0.0)

    selector_rows = _selector_rows(problem)
    selector_identity_passed = all(row.passed for row in selector_rows)
    nuisance = next(row for row in action_rows if row.action_id == "nuisance")
    nuisance_entropy = nuisance.raw_observation_entropy.independent_value
    nuisance_negative_control_passed = (
        nuisance.information_gain.independent_value == 0.0
        and exact.evaluate(NUISANCE_SCAN).expected_decision_value == 0
        and nuisance_entropy == max(row.raw_observation_entropy.independent_value for row in action_rows)
        and nuisance_entropy
        > max(row.raw_observation_entropy.independent_value for row in action_rows if row.action_id != "nuisance")
    )
    sign_label_invariance_passed = _sign_label_invariance(problem, exact)
    hand_totals_and_cost_once_passed = all(
        exact.evaluate(action).expected_episode_value == expected
        and next(row for row in action_rows if row.action_id == _public_action_id(action)).cost_charged_once
        for action, expected in _HAND_TOTALS.items()
    )
    uniform_control_determinism_passed = _uniform_control_is_deterministic(problem)
    schema_coverage_rows = _schema_coverage_rows()
    uncovered_schema_types = _uncovered_schema_types(schema_coverage_rows)
    public_schema_isolation_passed = (
        bool(schema_coverage_rows) and all(row.passed for row in schema_coverage_rows) and not uncovered_schema_types
    )
    oracle_independence_passed = not oracle_independence_violations
    implementation_binding_passed = (
        not implementation_binding_violations
        and all(row.passed for row in implementation_manifest)
        and _manifest_digest(implementation_manifest) == implementation_digest
        and _manifest_sha(implementation_manifest, "bench/active_acquisition/protocol.json") == protocol_digest
        and _manifest_sha(implementation_manifest, "bench/active_acquisition/oracle.py") == oracle_digest
    )
    direct_environment_call_violations = _direct_environment_call_violations()
    zero_interactions_passed = not direct_environment_call_violations

    checks = (
        QualificationCheck(
            "protocol_identity",
            protocol_identity_passed,
            "The exact prospective protocol bytes and version are identified.",
        ),
        QualificationCheck(
            "prospective_boundary",
            prospective_boundary_passed,
            "Qualification and formal authority remain disabled for claims.",
        ),
        QualificationCheck(
            "candidate_identity_and_order",
            candidate_identity_passed,
            "The protocol and implementation bind the exact five selectable action IDs.",
        ),
        QualificationCheck(
            "protocol_parity",
            not protocol_parity_violations,
            "Structured fixture, selector, report, and authority declarations match executable Q0 semantics.",
        ),
        QualificationCheck(
            "exact_semantic_matrix",
            not semantic_matrix_violations,
            "All 88 exact realization cells are unique, normalized, and aggregate to the declared action totals.",
        ),
        QualificationCheck(
            "exact_float_matrix",
            exact_float_matrix_passed,
            "All five floating rows match the independent exact rows within tolerance.",
        ),
        QualificationCheck(
            "independent_transcendental_matrix",
            all(row.raw_observation_entropy.passed and row.information_gain.passed for row in action_rows),
            "Raw entropy and information gain were recomputed from exact likelihoods.",
        ),
        QualificationCheck(
            "selector_identity",
            selector_identity_passed,
            "Every deterministic policy selects its protocol-required action.",
        ),
        QualificationCheck(
            "nuisance_negative_control",
            nuisance_negative_control_passed,
            "The nuisance channel is maximally unpredictable and decision-irrelevant.",
        ),
        QualificationCheck(
            "sign_label_invariance",
            sign_label_invariance_passed,
            "Pulse values are invariant to a simultaneous sign-label permutation.",
        ),
        QualificationCheck(
            "hand_totals_and_cost_once",
            hand_totals_and_cost_once_passed,
            "Exact hand totals hold and each selected cost appears once.",
        ),
        QualificationCheck(
            "uniform_control_determinism",
            uniform_control_determinism_passed,
            "The declared random control is reproducible and stays in support.",
        ),
        QualificationCheck(
            "uniform_protocol_vectors",
            not uniform_vector_violations,
            "Every declared public vector matches its full SHA-256 digest, modulo index, and action.",
        ),
        QualificationCheck(
            "oracle_independence",
            oracle_independence_passed,
            (
                "The exact oracle has no forbidden direct Prospect epistemics or selector call "
                "dependency; transitive imports are not claimed isolated."
            ),
        ),
        QualificationCheck(
            "implementation_binding",
            implementation_binding_passed,
            (
                "The report binds a selected-source Q0 manifest by SHA-256 without claiming "
                "complete import, transitive dependency, or environment closure."
            ),
        ),
        QualificationCheck(
            "public_schema_isolation",
            public_schema_isolation_passed,
            "Private-control identifiers do not occur in public field names.",
        ),
        QualificationCheck(
            "zero_environment_interactions",
            zero_interactions_passed,
            (
                "A source scan finds no direct environment realization call; tests also replace "
                "both realization gateways with fail-closed sentinels."
            ),
        ),
    )
    passed = all(check.passed for check in checks)
    return QualificationReport(
        schema=REPORT_SCHEMA,
        protocol_sha256=protocol_digest,
        oracle_sha256=oracle_digest,
        implementation_sha256=implementation_digest,
        implementation_manifest=implementation_manifest,
        implementation_binding_violations=implementation_binding_violations,
        implementation_binding_scope=_IMPLEMENTATION_BINDING_SCOPE,
        interpreter_identity=_interpreter_identity(),
        protocol_version=protocol_version,
        tolerance=TOLERANCE,
        claim_eligible=False,
        formal_authorized=False,
        environment_interactions=0,
        action_rows=action_rows,
        selector_rows=selector_rows,
        schema_coverage_rows=schema_coverage_rows,
        uncovered_schema_types=uncovered_schema_types,
        oracle_independence_violations=oracle_independence_violations,
        semantic_matrix_cell_count=len(semantic_cells),
        semantic_matrix_sha256=semantic_digest,
        semantic_matrix_action_totals=semantic_action_totals,
        semantic_matrix_violations=semantic_matrix_violations,
        protocol_parity_violations=protocol_parity_violations,
        uniform_vector_count=uniform_vector_count,
        uniform_vectors_sha256=uniform_vectors_digest,
        uniform_vector_violations=uniform_vector_violations,
        checks=checks,
        maximum_absolute_error=maximum_error,
        passed=passed,
    )


def forbidden_public_field_paths(value: object) -> tuple[str, ...]:
    """Return forbidden key paths in a prospective public payload."""

    paths: list[str] = []

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_text = str(key)
                next_path = (*path, key_text)
                normalized = key_text.casefold()
                if _is_forbidden_public_field(normalized):
                    paths.append(".".join(next_path))
                walk(child, next_path)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, child in enumerate(node):
                walk(child, (*path, str(index)))

    walk(value, ())
    return tuple(paths)


def _qualify_action(
    problem: HiddenActuatorProblem,
    exact: FractionOracle,
    action: AcquisitionAction,
) -> ActionQualificationRow:
    prospect = problem.diagnose(action)
    oracle = exact.evaluate(action)
    comparisons = _rational_comparisons(problem, exact, action, prospect, oracle)
    independent_entropy, independent_gain = _independent_entropy_and_gain(exact, action)
    entropy_comparison = _float_comparison(
        "raw_observation_entropy_nats",
        prospect.observation_entropy_nats,
        independent_entropy,
    )
    gain_comparison = _float_comparison(
        "information_gain_nats",
        prospect.expected_information_gain_nats,
        independent_gain,
    )
    hand_total = _HAND_TOTALS[action]
    cost_once = (
        oracle.expected_immediate_payoff
        + oracle.expected_terminal_value_after_observation
        - oracle.action_cost
        - oracle.acquisition_cost
        == oracle.expected_episode_value
        == hand_total
    )
    passed = (
        all(row.passed for row in comparisons) and entropy_comparison.passed and gain_comparison.passed and cost_once
    )
    return ActionQualificationRow(
        action_id=_public_action_id(action),
        rational_comparisons=comparisons,
        raw_observation_entropy=entropy_comparison,
        information_gain=gain_comparison,
        exact_hand_total=str(hand_total),
        cost_charged_once=cost_once,
        passed=passed,
    )


def _rational_comparisons(
    problem: HiddenActuatorProblem,
    exact: FractionOracle,
    action: AcquisitionAction,
    prospect: AcquisitionDiagnostics,
    oracle: ExactAcquisitionEvaluation,
) -> tuple[ScalarComparison, ...]:
    comparisons = [
        _scalar_comparison(
            f"outcome_probability[{index}]",
            prospect_probability,
            oracle_probability,
        )
        for index, (prospect_probability, oracle_probability) in enumerate(
            zip(prospect.outcome_probabilities, oracle.outcome_probabilities, strict=True)
        )
    ]
    for outcome in exact.outcomes(action):
        comparisons.append(
            _scalar_comparison(
                f"posterior_direct[outcome={outcome}]",
                problem.posterior_direct(action, outcome),
                exact.posterior_direct(action, outcome),
            )
        )
    comparisons.extend(
        (
            _scalar_comparison(
                "expected_immediate_payoff",
                prospect.expected_immediate_payoff,
                oracle.expected_immediate_payoff,
            ),
            _scalar_comparison(
                "prior_terminal_value",
                prospect.prior_terminal_value,
                oracle.prior_terminal_value,
            ),
            _scalar_comparison(
                "expected_terminal_value_after_observation",
                prospect.prior_terminal_value + prospect.expected_decision_value,
                oracle.expected_terminal_value_after_observation,
            ),
            _scalar_comparison(
                "expected_decision_value",
                prospect.expected_decision_value,
                oracle.expected_decision_value,
            ),
            _scalar_comparison(
                "action_cost",
                prospect.expected_action_cost,
                oracle.action_cost,
            ),
            _scalar_comparison(
                "acquisition_cost",
                prospect.expected_acquisition_cost,
                oracle.acquisition_cost,
            ),
            _scalar_comparison(
                "net_incremental_value",
                prospect.net_incremental_value,
                oracle.net_incremental_value,
            ),
            _scalar_comparison(
                "expected_episode_value",
                prospect.expected_episode_value,
                oracle.expected_episode_value,
            ),
        )
    )
    return tuple(comparisons)


def _scalar_comparison(name: str, prospect_value: float, oracle_value: Fraction) -> ScalarComparison:
    exact_float = float(oracle_value)
    error = abs(prospect_value - exact_float)
    return ScalarComparison(
        name=name,
        prospect_value=prospect_value,
        oracle_fraction=str(oracle_value),
        oracle_value=exact_float,
        absolute_error=error,
        passed=error <= TOLERANCE,
    )


def _float_comparison(name: str, prospect_value: float, independent_value: float) -> FloatComparison:
    error = abs(prospect_value - independent_value)
    return FloatComparison(
        name=name,
        prospect_value=prospect_value,
        independent_value=independent_value,
        absolute_error=error,
        passed=error <= TOLERANCE,
    )


def _independent_entropy_and_gain(
    exact: FractionOracle,
    action: AcquisitionAction,
) -> tuple[float, float]:
    reversed_row, direct_row = exact.likelihoods(action)
    predictive = tuple(
        _HALF * reversed_probability + _HALF * direct_probability
        for reversed_probability, direct_probability in zip(reversed_row, direct_row, strict=True)
    )
    raw_entropy = _fraction_entropy(predictive)
    expected_posterior_entropy = 0.0
    for index, probability in enumerate(predictive):
        if probability == 0:
            continue
        posterior_direct = (_HALF * direct_row[index]) / probability
        expected_posterior_entropy += float(probability) * _fraction_entropy(
            (Fraction(1) - posterior_direct, posterior_direct)
        )
    gain = _fraction_entropy((_HALF, _HALF)) - expected_posterior_entropy
    if abs(gain) <= 1e-15:
        gain = 0.0
    return raw_entropy, gain


def _fraction_entropy(probabilities: Sequence[Fraction]) -> float:
    return -sum(float(value) * math.log(float(value)) for value in probabilities if value)


def _selector_rows(problem: HiddenActuatorProblem) -> tuple[SelectorQualificationRow, ...]:
    declarations: tuple[tuple[str, AcquisitionPolicyDecision, AcquisitionAction], ...] = (
        ("prospect_expected_return", prospect_voi(problem), STRONG_POSITIVE),
        ("independent_fraction_oracle", oracle_policy(problem), STRONG_POSITIVE),
        ("goal_only", goal_only(problem), SKIP),
        ("raw_observation_entropy", raw_entropy(problem), NUISANCE_SCAN),
        ("eig_only", eig_only(problem), OVERPOWERED_POSITIVE),
        ("shuffled_information", shuffled_information(problem), WEAK_POSITIVE),
    )
    return tuple(
        SelectorQualificationRow(
            policy=policy,
            expected_action=_public_action_id(expected),
            selected_action=_public_action_id(decision.selected_action),
            passed=decision.selected_action == expected,
        )
        for policy, decision, expected in declarations
    )


def _uniform_control_is_deterministic(problem: HiddenActuatorProblem) -> bool:
    expected = (
        "nuisance",
        "strong",
        "overpowered",
        "strong",
        "strong",
        "nuisance",
        "strong",
        "skip",
        "overpowered",
        "nuisance",
    )
    observed = tuple(uniform_random(problem, seed=index).selected_action.action_id for index in range(len(expected)))
    repeated = tuple(uniform_random(problem, seed=index).selected_action.action_id for index in range(len(expected)))
    return observed == repeated == expected


def _sign_label_invariance(problem: HiddenActuatorProblem, exact: FractionOracle) -> bool:
    pairs = (
        (WEAK_POSITIVE, WEAK_NEGATIVE),
        (STRONG_POSITIVE, STRONG_NEGATIVE),
        (OVERPOWERED_POSITIVE, OVERPOWERED_NEGATIVE),
    )
    for positive, negative in pairs:
        positive_float = problem.diagnose(positive)
        negative_float = problem.diagnose(negative)
        positive_exact = exact.evaluate(positive)
        negative_exact = exact.evaluate(negative)
        if positive_exact.expected_episode_value != negative_exact.expected_episode_value:
            return False
        if abs(positive_float.expected_episode_value - negative_float.expected_episode_value) > TOLERANCE:
            return False
        if any(
            positive_probability != negative_probability
            for positive_row, negative_row in zip(
                exact.likelihoods(positive),
                exact.likelihoods(negative),
                strict=True,
            )
            for positive_probability, negative_probability in zip(
                positive_row,
                reversed(negative_row),
                strict=True,
            )
        ):
            return False
        for outcome in exact.outcomes(positive):
            if exact.posterior_direct(positive, outcome) != exact.posterior_direct(negative, -outcome):
                return False
            if (
                abs(problem.posterior_direct(positive, outcome) - problem.posterior_direct(negative, -outcome))
                > TOLERANCE
            ):
                return False
    return True


def _exact_semantic_matrix(exact: FractionOracle | None = None) -> tuple[_ExactSemanticCell, ...]:
    """Enumerate the 88 exact equation cells without realizing an environment."""

    oracle = FractionOracle() if exact is None else exact
    cells: list[_ExactSemanticCell] = []
    prior = Fraction(1, 2)
    for state_index, state in enumerate((-1, 1)):
        for action in ACQUISITION_ACTIONS:
            likelihood_row = oracle.likelihoods(action)[state_index]
            for observation_index, observation in enumerate(oracle.outcomes(action)):
                observation_probability = likelihood_row[observation_index]
                posterior = oracle.posterior_direct(action, observation)
                immediate = Fraction(int(action.is_pulse and observation == 1))
                action_cost = oracle.action_cost(action)
                acquisition_cost = oracle.acquisition_cost(action)
                for terminal_decision in (1, -1):
                    success_probability = Fraction(9, 10) if terminal_decision == state else Fraction(1, 10)
                    for terminal_success in (False, True):
                        terminal_probability = (
                            success_probability if terminal_success else Fraction(1) - success_probability
                        )
                        cells.append(
                            _ExactSemanticCell(
                                state=state,
                                action_id=action.action_id,
                                observation=observation,
                                terminal_decision=terminal_decision,
                                terminal_success=terminal_success,
                                path_probability=prior * observation_probability * terminal_probability,
                                posterior_direct=posterior,
                                immediate_payoff=immediate,
                                action_cost=action_cost,
                                acquisition_cost=acquisition_cost,
                                terminal_outcome_probability=terminal_probability,
                                realized_return=immediate
                                + Fraction(int(terminal_success))
                                - action_cost
                                - acquisition_cost,
                            )
                        )
    return tuple(cells)


def _semantic_matrix_digest(cells: Sequence[_ExactSemanticCell]) -> str:
    payload = json.dumps(
        [cell.canonical_dict() for cell in cells],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(payload).hexdigest()


def _semantic_matrix_violations(
    cells: Sequence[_ExactSemanticCell],
    exact: FractionOracle,
    protocol: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    violations: list[str] = []
    declared_count = _nested_value(protocol, "q0", "semantic_matrix_cell_count")
    if declared_count != 88:
        violations.append("coverage:protocol cell count is not 88")
    if len(cells) != 88:
        violations.append(f"coverage:observed {len(cells)} cells instead of 88")
    keys = tuple(
        (
            cell.state,
            cell.action_id,
            cell.observation,
            cell.terminal_decision,
            cell.terminal_success,
        )
        for cell in cells
    )
    if len(set(keys)) != len(keys):
        violations.append("coverage:duplicate semantic cell key")

    action_by_id = {action.action_id: action for action in ACQUISITION_ACTIONS}
    for cell in cells:
        key = (
            f"state={cell.state},action={cell.action_id},observation={cell.observation},"
            f"decision={cell.terminal_decision},success={int(cell.terminal_success)}"
        )
        action = action_by_id.get(cell.action_id)
        if action is None or cell.state not in (-1, 1) or cell.terminal_decision not in (-1, 1):
            violations.append(f"cell_semantics:{key}:unsupported key")
            continue
        outcomes = exact.outcomes(action)
        if cell.observation not in outcomes:
            violations.append(f"cell_semantics:{key}:unsupported observation")
            continue
        observation_index = outcomes.index(cell.observation)
        state_index = 0 if cell.state == -1 else 1
        observation_probability = exact.likelihoods(action)[state_index][observation_index]
        terminal_success_probability = Fraction(9, 10) if cell.terminal_decision == cell.state else Fraction(1, 10)
        terminal_outcome_probability = (
            terminal_success_probability if cell.terminal_success else Fraction(1) - terminal_success_probability
        )
        immediate = Fraction(int(action.is_pulse and cell.observation == 1))
        action_cost = exact.action_cost(action)
        acquisition_cost = exact.acquisition_cost(action)
        expected_fields = {
            "path_probability": Fraction(1, 2) * observation_probability * terminal_outcome_probability,
            "posterior_direct": exact.posterior_direct(action, cell.observation),
            "immediate_payoff": immediate,
            "action_cost": action_cost,
            "acquisition_cost": acquisition_cost,
            "terminal_outcome_probability": terminal_outcome_probability,
            "realized_return": immediate + Fraction(int(cell.terminal_success)) - action_cost - acquisition_cost,
        }
        for field_name, expected_value in expected_fields.items():
            if getattr(cell, field_name) != expected_value:
                violations.append(f"cell_semantics:{key}:{field_name}={getattr(cell, field_name)}!={expected_value}")

    for action in ACQUISITION_ACTIONS:
        for terminal_decision in (1, -1):
            total_probability = sum(
                (
                    cell.path_probability
                    for cell in cells
                    if cell.action_id == action.action_id and cell.terminal_decision == terminal_decision
                ),
                Fraction(0),
            )
            if total_probability != 1:
                violations.append(f"normalization:{action.action_id}:{terminal_decision}={total_probability}")

    totals: list[str] = []
    for action in ACQUISITION_ACTIONS:
        expected_return = sum(
            (
                cell.path_probability * cell.realized_return
                for cell in cells
                if cell.action_id == action.action_id
                and cell.terminal_decision
                == int(
                    exact.terminal_value(prior_direct=cell.posterior_direct)
                    == exact.expected_terminal_payoff(
                        TerminalAction.DIRECT,
                        prior_direct=cell.posterior_direct,
                    )
                )
                * 2
                - 1
            ),
            Fraction(0),
        )
        expected = exact.evaluate(action).expected_episode_value
        totals.append(f"{action.action_id}={expected_return}")
        if expected_return != expected:
            violations.append(f"aggregation:{action.action_id}:{expected_return}!={expected}")
    if any(cell.path_probability < 0 for cell in cells):
        violations.append("probability:negative path probability")
    return tuple(violations), tuple(totals)


def _q0_normative_protocol_projection(protocol: Mapping[str, object]) -> dict[str, object]:
    """Select every claim-bearing Q0 declaration; exclude prospective Q1 mechanics."""

    fixture_value = protocol.get("fixture")
    fixture = fixture_value if isinstance(fixture_value, dict) else {}
    latent_value = fixture.get("latent_state")
    latent = latent_value if isinstance(latent_value, dict) else {}
    action_value = fixture.get("actions")
    actions = action_value if isinstance(action_value, list) else []
    terminal_value = fixture.get("terminal_decision")
    terminal = terminal_value if isinstance(terminal_value, dict) else {}
    decision_value = protocol.get("decision_semantics")
    decision = decision_value if isinstance(decision_value, dict) else {}
    arm_value = protocol.get("arms")
    arms = arm_value if isinstance(arm_value, list) else []
    uniform_value = protocol.get("uniform_selector")
    uniform = uniform_value if isinstance(uniform_value, dict) else {}
    vector_value = uniform.get("fixed_vectors")
    vectors = vector_value if isinstance(vector_value, list) else []
    q0_value = protocol.get("q0")
    q0 = q0_value if isinstance(q0_value, dict) else {}
    scope_value = protocol.get("scope")
    scope = scope_value if isinstance(scope_value, dict) else {}
    seed_value = protocol.get("seed_schedule")
    seed = seed_value if isinstance(seed_value, dict) else {}
    formal_value = protocol.get("formal_boundary")
    formal = formal_value if isinstance(formal_value, dict) else {}
    return {
        "fixture_identity": {
            "id": fixture.get("id"),
            "latent_name": latent.get("name"),
            "latent_support": latent.get("support"),
            "latent_prior": latent.get("prior"),
            "latent_lifetime": latent.get("lifetime"),
            "latent_visibility": latent.get("visibility"),
        },
        "action_declarations": [
            {
                "id": row.get("id"),
                "likelihood": row.get("likelihood"),
                "immediate_task_payoff": row.get("immediate_task_payoff"),
            }
            for row in actions
            if isinstance(row, dict)
        ],
        "terminal_declarations": {
            "success_likelihood": terminal.get("success_likelihood"),
            "payoff": terminal.get("payoff"),
        },
        "return_semantics": {
            "net_return": fixture.get("net_return"),
            "cost_rule": fixture.get("cost_rule"),
        },
        "noncandidate_boundary": {
            "fixtures": fixture.get("q0_only_non_candidate_fixtures"),
            "rule": fixture.get("non_candidate_rule"),
        },
        "decision_declarations": {
            name: decision.get(name)
            for name in (
                "prospect_total_value",
                "expected_decision_value",
                "information_gain",
                "raw_observation_entropy",
                "tie_break",
                "candidate_total_value_unit",
                "fraction_exact_boundary",
                "floating_log_boundary",
                "cost_accounting",
                "required_prior_values",
            )
        },
        "arm_roles": [{"id": row.get("id"), "role": row.get("role")} for row in arms if isinstance(row, dict)],
        "uniform_contract": {
            "accepted_input": uniform.get("accepted_input"),
            "fixed_vector_seeds": [row.get("seed") if isinstance(row, dict) else None for row in vectors],
        },
        "q0_contract": {
            "name": q0.get("name"),
            "requirements": q0.get("requirements"),
            "on_failure": q0.get("on_failure"),
        },
        "scope_boundary": {
            "does_not_establish": scope.get("does_not_establish"),
            "exact_oracle_boundary": scope.get("exact_oracle_boundary"),
        },
        "authority_boundary": {
            "seed_status": seed.get("status"),
            "formal_master_indices": seed.get("formal_master_indices"),
            "promotion_rule": formal.get("promotion_rule"),
        },
    }


def _expected_q0_normative_protocol() -> dict[str, object]:
    """Literal Q0 declaration projection independently frozen in executable code."""

    return {
        "fixture_identity": {
            "id": "wm002-hidden-actuator-v0",
            "latent_name": "theta",
            "latent_support": [-1, 1],
            "latent_prior": ["1/2", "1/2"],
            "latent_lifetime": "fixed for one episode",
            "latent_visibility": (
                "harness-private; theta, private seeds, the environment salt, and keyed "
                "private potential-outcome material must not occur in an agent-visible "
                "observation, outcome, decision, experience, transition, update receipt, "
                "checkpoint component, or public result row"
            ),
        },
        "action_declarations": [
            {
                "id": "skip",
                "likelihood": "P(null | theta, skip) = 1",
                "immediate_task_payoff": "0",
            },
            {
                "id": "weak",
                "likelihood": "P(y = theta | theta, weak) = 7/10",
                "immediate_task_payoff": "1[y = +1]",
            },
            {
                "id": "strong",
                "likelihood": "P(y = theta | theta, strong) = 9/10",
                "immediate_task_payoff": "1[y = +1]",
            },
            {
                "id": "overpowered",
                "likelihood": "P(y = theta | theta, overpowered) = 1",
                "immediate_task_payoff": "1[y = +1]",
            },
            {
                "id": "nuisance",
                "likelihood": "P(z = k | theta, nuisance) = 1/4 for every k and theta",
                "immediate_task_payoff": "0",
            },
        ],
        "terminal_declarations": {
            "success_likelihood": ("P(success | d = theta) = 9/10 and P(success | d != theta) = 1/10"),
            "payoff": "1[success]",
        },
        "return_semantics": {
            "net_return": ("immediate acquisition task payoff + terminal success - action_cost - acquisition_cost"),
            "cost_rule": (
                "action_cost is physical pulse/resource cost; acquisition_cost is "
                "information-acquisition cost. Each selected component is subtracted exactly "
                "once. Unselected candidate costs are forecasts only and are never charged."
            ),
        },
        "noncandidate_boundary": {
            "fixtures": [
                "weak:sign_inverted_non_candidate",
                "strong:sign_inverted_non_candidate",
                "overpowered:sign_inverted_non_candidate",
            ],
            "rule": (
                "The sign-inverted pulse variants exist only for Q0 sign-label invariance "
                "checks. They are never selectable and never appear in Q1 candidate lists."
            ),
        },
        "decision_declarations": {
            "prospect_total_value": (
                "expected immediate acquisition task payoff + prior-optimal terminal value + "
                "expected decision value of the acquisition observation - action_cost - "
                "acquisition_cost"
            ),
            "expected_decision_value": (
                "expected posterior-optimal terminal success probability minus prior-optimal "
                "terminal success probability"
            ),
            "information_gain": ("prior entropy over theta minus expected posterior entropy over theta"),
            "raw_observation_entropy": (
                "entropy of the acquisition observation marginal, without conditioning on "
                "relevance to theta or terminal utility"
            ),
            "tie_break": (
                "candidate ties are resolved by declared list order: skip, weak, strong, "
                "overpowered, nuisance; terminal value ties choose d=+1"
            ),
            "candidate_total_value_unit": "executed episode return",
            "fraction_exact_boundary": (
                "Likelihoods, probabilities, reachable posteriors, expected task payoff, prior "
                "and posterior-optimal terminal values, EVSI, both cost components, and "
                "expected episode net returns are rational and must be checked exactly with "
                "fractions.Fraction."
            ),
            "floating_log_boundary": (
                "Entropy and EIG contain logarithms and are not Fraction-exact. They must be "
                "independently recomputed from explicit log formulas and compared to Prospect "
                "floating values within absolute tolerance 1e-12."
            ),
            "cost_accounting": (
                "action_cost and acquisition_cost are distinct return-unit components and each "
                "is subtracted exactly once in forecasts and realized net return."
            ),
            "required_prior_values": {
                "skip": "1/2",
                "weak": "63/100",
                "strong": "74/100",
                "overpowered": "45/100",
                "nuisance": "49/100",
            },
        },
        "arm_roles": [
            {"id": "prospect_expected_return", "role": "primary"},
            {
                "id": "independent_fraction_oracle",
                "role": "ceiling_and_semantic_oracle",
            },
            {"id": "goal_only", "role": "non_oracle_control"},
            {"id": "raw_observation_entropy", "role": "non_oracle_control"},
            {"id": "eig_only", "role": "non_oracle_control"},
            {
                "id": "shuffled_information",
                "role": "non_oracle_marginal_preserving_control",
            },
            {"id": "uniform_random", "role": "non_oracle_control"},
        ],
        "uniform_contract": {
            "accepted_input": (
                "nonnegative integer seed represented in canonical decimal with no sign, "
                "leading zero, or whitespace except the integer zero itself"
            ),
            "fixed_vector_seeds": [0, 1, 2, 7, 13],
        },
        "q0_contract": {
            "name": "exact_semantic_matrix",
            "requirements": [
                (
                    "enumerate both theta values, every acquisition action, every supported "
                    "observation, both terminal decisions, and both terminal outcomes"
                ),
                ("compute the full oracle using fractions.Fraction without importing the Prospect selector"),
                ("compute the Prospect path through the generic floating Bayes, EIG, and EVSI functions"),
                (
                    "match every reachable posterior, expected decision value, information "
                    "gain, raw observation entropy, action total, and selector within absolute "
                    "tolerance 1e-12"
                ),
                (
                    "match the five hand-derived expected net returns exactly in the Fraction "
                    "oracle and within 1e-12 in the floating path"
                ),
                "require Prospect and the independent oracle to select strong",
                (
                    "require goal-only to select skip, raw entropy to select nuisance, EIG-only "
                    "to select overpowered, and shuffled information to select weak"
                ),
                "verify sign-label permutation invariance",
                (
                    "verify nuisance outcomes have zero information gain and zero expected "
                    "decision value despite maximal raw entropy"
                ),
                "verify action_cost and acquisition_cost are each charged exactly once",
                (
                    "verify declared public record field sets and every emitted Q0 payload "
                    "contain no theta, hidden-regime, private-seed, or latent-state field; Q1 "
                    "must separately test actual serialized runtime values"
                ),
            ],
            "on_failure": ("Stop WM-002. Fix semantics or retire the formulation before any Q1 run."),
        },
        "scope_boundary": {
            "does_not_establish": [
                "a learned information-value estimator",
                "calibrated learned uncertainty",
                "general active learning or general dual control",
                "neural world-model improvement",
                "transfer to unseen dynamics or task families",
                "continual-learning scale, multimodality, or long-horizon autonomy",
                "superiority to published baselines or state of the art",
                "formal WM-002 evidence",
            ],
            "exact_oracle_boundary": (
                "Prospect and the independent oracle are intentionally equal in this "
                "qualification. Their agreement is a semantic and integration check, not "
                "evidence that uncertainty or VOI was learned."
            ),
        },
        "authority_boundary": {
            "seed_status": "prospective design; unavailable until Q1 authorization",
            "formal_master_indices": [],
            "promotion_rule": (
                "Only after Q0 and Q1 pass, their independent audit passes, and the exact "
                "formulation survives a prospective review may a new separately versioned "
                "formal protocol be proposed. That protocol must allocate fresh seeds and "
                "freeze its own claim, thresholds, schemas, implementation binding, and audit "
                "contract before any formal interaction."
            ),
        },
    }


def _q0_normative_projection_violations(
    protocol: Mapping[str, object],
) -> tuple[str, ...]:
    actual = _q0_normative_protocol_projection(protocol)
    expected = _expected_q0_normative_protocol()
    return tuple(
        f"{category}:normative projection mismatch:{actual.get(category)!r}!={expected_value!r}"
        for category, expected_value in expected.items()
        if actual.get(category) != expected_value
    )


def _protocol_parity_violations(
    protocol: Mapping[str, object],
    problem: HiddenActuatorProblem,
    exact: FractionOracle,
) -> tuple[str, ...]:
    violations = list(_q0_normative_projection_violations(protocol))

    def require(category: str, label: str, observed: object, expected: object) -> None:
        if observed != expected:
            violations.append(f"{category}:{label}:{observed!r}!={expected!r}")

    fixture = protocol.get("fixture")
    fixture_map = fixture if isinstance(fixture, dict) else {}
    require("fixture_identity", "id", fixture_map.get("id"), "wm002-hidden-actuator-v0")
    latent = fixture_map.get("latent_state")
    latent_map = latent if isinstance(latent, dict) else {}
    require("fixture_identity", "latent.name", latent_map.get("name"), "theta")
    require("latent_fixture", "support", latent_map.get("support"), [-1, 1])
    require("latent_fixture", "prior", latent_map.get("prior"), ["1/2", "1/2"])
    raw_actions = fixture_map.get("actions")
    action_rows = raw_actions if isinstance(raw_actions, list) else []
    by_id = {row.get("id"): row for row in action_rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
    expected_supports = {
        "skip": ("acquisition", ["null"]),
        "weak": ("pulse", [-1, 1]),
        "strong": ("pulse", [-1, 1]),
        "overpowered": ("pulse", [-1, 1]),
        "nuisance": ("nuisance_scan", [0, 1, 2, 3]),
    }
    require("action_supports", "ordered_ids", tuple(by_id), tuple(expected_supports))
    for action_id, (kind, outcomes) in expected_supports.items():
        row = by_id.get(action_id, {})
        require("action_supports", f"{action_id}.kind", row.get("kind"), kind)
        require("action_supports", f"{action_id}.outcomes", row.get("outcomes"), outcomes)

    expected_likelihoods = {
        "skip": "P(null | theta, skip) = 1",
        "weak": "P(y = theta | theta, weak) = 7/10",
        "strong": "P(y = theta | theta, strong) = 9/10",
        "overpowered": "P(y = theta | theta, overpowered) = 1",
        "nuisance": "P(z = k | theta, nuisance) = 1/4 for every k and theta",
    }
    expected_immediate_payoffs = {
        "skip": "0",
        "weak": "1[y = +1]",
        "strong": "1[y = +1]",
        "overpowered": "1[y = +1]",
        "nuisance": "0",
    }
    for action_id in expected_supports:
        row = by_id.get(action_id, {})
        require(
            "action_likelihoods",
            action_id,
            row.get("likelihood"),
            expected_likelihoods[action_id],
        )
        require(
            "action_immediate_payoffs",
            action_id,
            row.get("immediate_task_payoff"),
            expected_immediate_payoffs[action_id],
        )

    expected_reliability = {
        "skip": None,
        "weak": Fraction(7, 10),
        "strong": Fraction(9, 10),
        "overpowered": Fraction(1),
        "nuisance": None,
    }
    for action_id, expected in expected_reliability.items():
        row = by_id.get(action_id, {})
        field_present = "reliability_q" in row
        require("action_reliabilities", f"{action_id}.field_present", field_present, expected is not None)
        observed = _optional_protocol_fraction(row.get("reliability_q"))
        require("action_reliabilities", action_id, observed, expected)

    for action in ACQUISITION_ACTIONS:
        row = by_id.get(action.action_id, {})
        require(
            "action_costs",
            f"{action.action_id}.action_cost",
            _optional_protocol_fraction(row.get("action_cost")),
            exact.action_cost(action),
        )
        require(
            "action_costs",
            f"{action.action_id}.acquisition_cost",
            _optional_protocol_fraction(row.get("acquisition_cost")),
            exact.acquisition_cost(action),
        )
        require(
            "action_totals",
            action.action_id,
            _optional_protocol_fraction(row.get("expected_net_return_at_prior")),
            exact.evaluate(action).expected_episode_value,
        )

    require(
        "return_semantics",
        "net_return",
        fixture_map.get("net_return"),
        "immediate acquisition task payoff + terminal success - action_cost - acquisition_cost",
    )
    require(
        "return_semantics",
        "cost_rule",
        fixture_map.get("cost_rule"),
        (
            "action_cost is physical pulse/resource cost; acquisition_cost is "
            "information-acquisition cost. Each selected component is subtracted exactly once. "
            "Unselected candidate costs are forecasts only and are never charged."
        ),
    )

    terminal = fixture_map.get("terminal_decision")
    terminal_map = terminal if isinstance(terminal, dict) else {}
    require("terminal_semantics", "actions", terminal_map.get("actions"), [1, -1])
    require(
        "terminal_semantics",
        "match_probability",
        _optional_protocol_fraction(terminal_map.get("match_success_probability")),
        Fraction(9, 10),
    )
    require(
        "terminal_semantics",
        "mismatch_probability",
        _optional_protocol_fraction(terminal_map.get("mismatch_success_probability")),
        Fraction(1, 10),
    )
    require("terminal_semantics", "outcomes", terminal_map.get("outcomes"), ["failure", "success"])
    require(
        "terminal_semantics",
        "tie_break",
        terminal_map.get("tie_break"),
        "When posterior terminal values tie, choose d=+1.",
    )
    require("terminal_semantics", "learning_allowed", terminal_map.get("learning_allowed"), False)

    decision_semantics = protocol.get("decision_semantics")
    decision_map = decision_semantics if isinstance(decision_semantics, dict) else {}
    candidate_ids = [action.action_id for action in problem.acquisition_actions]
    require("candidate_order", "decision_semantics", decision_map.get("candidate_order"), candidate_ids)
    require(
        "required_prior_values",
        "decision_semantics",
        decision_map.get("required_prior_values"),
        {
            "skip": "1/2",
            "weak": "63/100",
            "strong": "74/100",
            "overpowered": "45/100",
            "nuisance": "49/100",
        },
    )

    decision_by_id = {
        decision.policy: decision
        for decision in (
            prospect_voi(problem),
            oracle_policy(problem),
            goal_only(problem),
            raw_entropy(problem),
            eig_only(problem),
            shuffled_information(problem),
            uniform_random(problem, seed=0),
        )
    }
    raw_arms = protocol.get("arms")
    arms = raw_arms if isinstance(raw_arms, list) else []
    arm_ids = tuple(row.get("id") for row in arms if isinstance(row, dict))
    require("arms", "ordered_ids", arm_ids, tuple(decision_by_id))
    expected_required = {
        "prospect_expected_return": "strong",
        "independent_fraction_oracle": "strong",
        "goal_only": "skip",
        "raw_observation_entropy": "nuisance",
        "eig_only": "overpowered",
        "shuffled_information": "weak",
        "uniform_random": "seed_dependent",
    }
    expected_roles = {
        "prospect_expected_return": "primary",
        "independent_fraction_oracle": "ceiling_and_semantic_oracle",
        "goal_only": "non_oracle_control",
        "raw_observation_entropy": "non_oracle_control",
        "eig_only": "non_oracle_control",
        "shuffled_information": "non_oracle_marginal_preserving_control",
        "uniform_random": "non_oracle_control",
    }
    for row in arms:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            violations.append("arms:invalid row")
            continue
        arm_id = row["id"]
        decision = decision_by_id.get(arm_id)
        if decision is None:
            violations.append(f"arms:unknown implementation policy:{arm_id}")
            continue
        kinds = {score.score_kind for score in decision.scores}
        units = {score.unit for score in decision.scores}
        require("arms", f"{arm_id}.selection_kind", kinds, {row.get("selection_kind")})
        require("arms", f"{arm_id}.selection_unit", units, {row.get("selection_unit")})
        require(
            "arms",
            f"{arm_id}.required_action",
            row.get("required_action_at_prior"),
            expected_required[arm_id],
        )
        require("arm_roles", arm_id, row.get("role"), expected_roles[arm_id])
        if arm_id == "uniform_random":
            require("arms", "uniform.null_scores", {score.value for score in decision.scores}, {None})
        else:
            require(
                "arms",
                f"{arm_id}.selected_action",
                decision.selected_action.action_id,
                expected_required[arm_id],
            )

    shuffled_row = next(
        (row for row in arms if isinstance(row, dict) and row.get("id") == "shuffled_information"),
        {},
    )
    require(
        "shuffled_mapping",
        "information_source_by_action",
        shuffled_row.get("information_source_by_action"),
        SHUFFLED_INFORMATION_SOURCE_BY_ACTION,
    )
    require("shuffled_mapping", "selection_only", shuffled_row.get("selection_only_perturbation"), True)

    q0 = protocol.get("q0")
    q0_map = q0 if isinstance(q0, dict) else {}
    require("report_contract", "schema", q0_map.get("report_schema"), REPORT_SCHEMA)
    require("report_contract", "tolerance", q0_map.get("absolute_tolerance"), "1e-12")
    require("report_contract", "cell_count", q0_map.get("semantic_matrix_cell_count"), 88)

    experiment = protocol.get("experiment")
    experiment_map = experiment if isinstance(experiment, dict) else {}
    q1 = protocol.get("q1")
    q1_map = q1 if isinstance(q1, dict) else {}
    budgets = protocol.get("budgets")
    budget_map = budgets if isinstance(budgets, dict) else {}
    formal = protocol.get("formal_boundary")
    formal_map = formal if isinstance(formal, dict) else {}
    seed_schedule = protocol.get("seed_schedule")
    seed_schedule_map = seed_schedule if isinstance(seed_schedule, dict) else {}
    scope = protocol.get("scope")
    scope_map = scope if isinstance(scope, dict) else {}
    authority_expected = {
        "experiment.status": "prospective_claim_ineligible_qualification_only",
        "experiment.statement": (
            "This document authorizes only claim-ineligible Q0 semantic qualification. It "
            "describes Q1 prospectively, but Q1 is blocked and execution_authorized is false "
            "until every declared entry requirement closes in a revised protocol. It cannot "
            "authorize a formal attempt or a capability claim."
        ),
        "experiment.implemented": False,
        "experiment.q0_implemented": True,
        "experiment.q1_implemented": False,
        "experiment.sealed": False,
        "experiment.formal_authorized": False,
        "experiment.formal_result_schema_exists": False,
        "q0.claim_eligible": False,
        "q0.environment_interactions": 0,
        "q1.claim_eligible": False,
        "q1.execution_authorized": False,
        "budgets.q0_environment_steps": 0,
        "formal.authorized": False,
        "formal.formal_protocol_version": None,
        "formal.formal_seed_set": [],
        "seed_schedule.formal_master_indices": [],
        "formal.formal_thresholds": None,
        "formal.formal_binding": None,
        "formal.formal_result_schema": None,
        "formal.q0_q1_outcome_use": (
            "Q0 and Q1 remain permanently claim-ineligible. They may kill or redesign the "
            "formulation and qualify the harness, but cannot be relabeled as confirmation or "
            "pooled with future formal evidence."
        ),
        "scope.q0_establishes": [
            "the exact known-model VOI selector and independent oracle agree on this finite fixture",
            (
                "the 88-cell exact semantic matrix is complete, normalized, and aggregates "
                "to all five declared expected returns"
            ),
            (
                "the structured fixture, policy, public-vector, report, and authority "
                "declarations match the Q0 implementation"
            ),
            "Q0 used no environment realization and remains claim-ineligible with Q1 and formal execution unauthorized",
        ],
        "scope.later_q1_must_establish": [
            "one active Prospect arm forms a continuous acquisition-to-update-to-executed-behavior chain",
            (
                "the exact selected acquisition transition is the only transition eligible "
                "for the pre-terminal persistent update"
            ),
            "the pre-terminal state and selected terminal decision survive a fresh process",
        ],
    }
    authority_observed = {
        "experiment.status": experiment_map.get("status"),
        "experiment.statement": experiment_map.get("statement"),
        "experiment.implemented": experiment_map.get("implemented"),
        "experiment.q0_implemented": experiment_map.get("q0_implemented"),
        "experiment.q1_implemented": experiment_map.get("q1_implemented"),
        "experiment.sealed": experiment_map.get("sealed"),
        "experiment.formal_authorized": experiment_map.get("formal_authorized"),
        "experiment.formal_result_schema_exists": experiment_map.get("formal_result_schema_exists"),
        "q0.claim_eligible": q0_map.get("claim_eligible"),
        "q0.environment_interactions": q0_map.get("environment_interactions"),
        "q1.claim_eligible": q1_map.get("claim_eligible"),
        "q1.execution_authorized": q1_map.get("execution_authorized"),
        "budgets.q0_environment_steps": budget_map.get("q0_environment_steps"),
        "formal.authorized": formal_map.get("authorized"),
        "formal.formal_protocol_version": formal_map.get("formal_protocol_version"),
        "formal.formal_seed_set": formal_map.get("formal_seed_set"),
        "seed_schedule.formal_master_indices": seed_schedule_map.get("formal_master_indices"),
        "formal.formal_thresholds": formal_map.get("formal_thresholds"),
        "formal.formal_binding": formal_map.get("formal_binding"),
        "formal.formal_result_schema": formal_map.get("formal_result_schema"),
        "formal.q0_q1_outcome_use": formal_map.get("q0_q1_outcome_use"),
        "scope.q0_establishes": scope_map.get("q0_establishes_if_qualification_passes"),
        "scope.later_q1_must_establish": scope_map.get("later_q1_must_establish"),
    }
    for label, authority_value in authority_expected.items():
        require("authority_boundary", label, authority_observed[label], authority_value)
    return tuple(violations)


def _uniform_vector_checks(
    protocol: Mapping[str, object],
    problem: HiddenActuatorProblem,
) -> tuple[int, str, tuple[str, ...]]:
    violations: list[str] = []
    raw = protocol.get("uniform_selector")
    selector = raw if isinstance(raw, dict) else {}
    expected_order = [action.action_id for action in problem.acquisition_actions]
    declarations = {
        "accepted_input": (
            "nonnegative integer seed represented in canonical decimal with no sign, leading "
            "zero, or whitespace except the integer zero itself"
        ),
        "payload_utf8": "WM-002|0.2.0-q|uniform|<canonical decimal seed>",
        "digest": "SHA256(payload_utf8)",
        "index": "unsigned big-endian integer represented by the full 32-byte digest modulo 5",
        "candidate_order": expected_order,
    }
    for name, expected in declarations.items():
        if selector.get(name) != expected:
            violations.append(f"uniform_vectors:{name}:{selector.get(name)!r}!={expected!r}")
    raw_vectors = selector.get("fixed_vectors")
    vectors = raw_vectors if isinstance(raw_vectors, list) else []
    expected_seeds = (0, 1, 2, 7, 13)
    declared_seeds = tuple(row.get("seed") if isinstance(row, dict) else None for row in vectors)
    if declared_seeds != expected_seeds:
        violations.append(f"uniform_vectors:fixed_vector_seeds:{declared_seeds!r}!={expected_seeds!r}")
    observed_rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for row in vectors:
        if not isinstance(row, dict) or isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
            violations.append("uniform_vectors:invalid vector row")
            continue
        seed = row["seed"]
        if seed < 0 or seed in seen:
            violations.append(f"uniform_vectors:invalid or duplicate seed:{seed}")
        seen.add(seed)
        digest, index, action = uniform_selector_vector(problem, seed=seed)
        observed = {"action": action.action_id, "digest": digest, "index": index, "seed": seed}
        observed_rows.append(observed)
        for name in ("digest", "index", "action"):
            if row.get(name) != observed[name]:
                violations.append(f"uniform_vectors:seed={seed}.{name}:{row.get(name)!r}!={observed[name]!r}")
    payload = json.dumps(
        observed_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if not vectors:
        violations.append("uniform_vectors:no fixed vectors")
    return len(observed_rows), sha256(payload).hexdigest(), tuple(violations)


def _optional_protocol_fraction(value: object) -> Fraction | None:
    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _is_forbidden_public_field(normalized: str) -> bool:
    return any(fragment in normalized for fragment in _FORBIDDEN_PUBLIC_FIELD_FRAGMENTS) or (
        "seed" in normalized
        and any(private in normalized for private in ("private", "secret", "environment", "master"))
    )


def _interpreter_identity() -> str:
    implementation = sys.implementation
    cache_tag = implementation.cache_tag or "none"
    version = sys.version_info
    return f"{implementation.name}|{version.major}.{version.minor}.{version.micro}|{cache_tag}"


def _schema_coverage_rows() -> tuple[SchemaCoverageRow, ...]:
    record_types = (
        Action,
        AgentSnapshot,
        Belief,
        BeliefUpdate,
        CandidateAssessment,
        DecisionRecord,
        Distribution,
        EpistemicEffect,
        EpistemicTarget,
        EpistemicTransition,
        EvaluationMetric,
        EvaluationRecord,
        Evidence,
        EvidenceLineage,
        ExecutedAction,
        ExperienceEvent,
        Goal,
        InformationSet,
        InformationValue,
        IntendedAction,
        Observation,
        Outcome,
        Prediction,
        ProperScore,
        Provenance,
        ResourceLedger,
        ResourceUse,
        TimePoint,
        UncertaintyEstimate,
        UpdateReceipt,
        Utility,
        AcquisitionAction,
        AcquisitionDiagnostics,
        AcquisitionResult,
        TerminalResult,
        ExactAcquisitionEvaluation,
        ExactOracleDecision,
        ActionScore,
        AcquisitionPolicyDecision,
        ScalarComparison,
        FloatComparison,
        ActionQualificationRow,
        SelectorQualificationRow,
        QualificationCheck,
        SchemaCoverageRow,
        ImplementationBindingRow,
        QualificationReport,
    )
    return tuple(
        SchemaCoverageRow(
            record_type=f"{record_type.__module__}.{record_type.__qualname__}",
            checked_fields=tuple(field.name for field in fields(record_type)),
            forbidden_fields=tuple(
                field.name for field in fields(record_type) if _is_forbidden_public_field(field.name.casefold())
            ),
            passed=not any(_is_forbidden_public_field(field.name.casefold()) for field in fields(record_type)),
        )
        for record_type in record_types
    )


def _uncovered_schema_types(
    rows: Sequence[SchemaCoverageRow],
) -> tuple[str, ...]:
    covered = {row.record_type for row in rows}
    public_domain_records = []
    for name in prospect_domain.__all__:
        value = getattr(prospect_domain, name)
        if isinstance(value, type) and is_dataclass(value):
            public_domain_records.append(f"{value.__module__}.{value.__qualname__}")
    return tuple(sorted(set(public_domain_records) - covered))


def _direct_environment_call_violations() -> tuple[str, ...]:
    """Statically reject direct use of either environment realization gateway."""

    path = Path(__file__)
    tree = ast.parse(path.read_bytes(), filename=str(path))
    forbidden = {"realize_acquisition", "realize_terminal"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name in forbidden:
            violations.append(f"direct environment call:{name}:line={node.lineno}")
    return tuple(sorted(violations))


def _oracle_independence() -> tuple[str, tuple[str, ...]]:
    path = Path(__file__).with_name("oracle.py")
    payload = path.read_bytes()
    tree = ast.parse(payload, filename=str(path))
    violations: list[str] = []
    forbidden_modules = (
        "prospect",
        "bench.active_acquisition.policies",
        "bench.active_acquisition.qualification",
    )
    forbidden_calls = {
        "bayes_posterior",
        "diagnose",
        "eig_only",
        "entropy",
        "expected_information_gain",
        "expected_value_of_sample_information",
        "goal_only",
        "oracle",
        "predictive_distribution",
        "prospect_voi",
        "raw_entropy",
        "shuffled_information",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(forbidden_modules):
                    violations.append(f"forbidden import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(forbidden_modules):
                violations.append(f"forbidden import:{module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            else:
                continue
            if call_name in forbidden_calls:
                violations.append(f"forbidden call:{call_name}")
    return sha256(payload).hexdigest(), tuple(sorted(set(violations)))


def _implementation_binding() -> tuple[tuple[ImplementationBindingRow, ...], str, tuple[str, ...]]:
    repository_root = Path(__file__).resolve().parents[2]
    required_paths = (
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
    rows: list[ImplementationBindingRow] = []
    violations: list[str] = []
    for relative_path in required_paths:
        path = repository_root / relative_path
        if not path.is_file():
            rows.append(ImplementationBindingRow(relative_path, "", False))
            violations.append(f"missing:{relative_path}")
            continue
        rows.append(
            ImplementationBindingRow(
                relative_path=relative_path,
                sha256=sha256(path.read_bytes()).hexdigest(),
                passed=True,
            )
        )
    manifest = tuple(rows)
    return manifest, _manifest_digest(manifest), tuple(violations)


def _manifest_digest(manifest: Sequence[ImplementationBindingRow]) -> str:
    payload = "".join(f"{row.relative_path}\0{row.sha256}\0{int(row.passed)}\n" for row in manifest).encode("utf-8")
    return sha256(payload).hexdigest()


def _manifest_sha(
    manifest: Sequence[ImplementationBindingRow],
    relative_path: str,
) -> str:
    return next(
        (row.sha256 for row in manifest if row.relative_path == relative_path),
        "",
    )


def _public_action_id(action: AcquisitionAction) -> str:
    return action.action_id


def _parse_protocol(payload: bytes) -> tuple[dict[str, object], bool]:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, False
    if not isinstance(parsed, dict):
        return {}, False
    return parsed, True


def _protocol_candidate_ids(protocol: Mapping[str, object]) -> tuple[str, ...]:
    fixture = protocol.get("fixture")
    if not isinstance(fixture, dict):
        return ()
    actions = fixture.get("actions")
    if not isinstance(actions, list):
        return ()
    identifiers: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            return ()
        identifier = action.get("id")
        if not isinstance(identifier, str):
            return ()
        identifiers.append(identifier)
    return tuple(identifiers)


def _nested_value(source: Mapping[str, object], first: str, second: str) -> object:
    nested = source.get(first)
    if not isinstance(nested, dict):
        return None
    return nested.get(second)


def _nested_text(source: Mapping[str, object], first: str, second: str) -> str:
    value = _nested_value(source, first, second)
    return value if isinstance(value, str) else "invalid"


__all__ = (
    "PROTOCOL_PATH",
    "REPORT_SCHEMA",
    "TOLERANCE",
    "ActionQualificationRow",
    "FloatComparison",
    "ImplementationBindingRow",
    "QualificationCheck",
    "QualificationReport",
    "ScalarComparison",
    "SchemaCoverageRow",
    "SelectorQualificationRow",
    "forbidden_public_field_paths",
    "run_qualification",
)
