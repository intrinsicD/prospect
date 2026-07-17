"""E2--E5 development diagnostics for the exact epistemic reference agent.

These diagnostics were initially written as if they formed one linked argument:

* E2: a target-aware policy acquires diagnostic rather than merely surprising data;
* E3: only correctly linked diagnostic experience improves held-out proper score;
* E4: that epistemic improvement changes externally scored behavior;
* E5: improvements for task A survive learning B and a serialized restart.

An independent results audit showed that this implementation does *not* establish
that argument.  The rows use different agents and task suites, E3 is task-local
belief assimilation rather than model training, E4 is an analytic expectation
rather than an executed held-out policy, and E5 excludes interference by using
task-keyed posterior slots and an incomplete benchmark-only checkpoint.

The calculations and controls remain useful exact-reference diagnostics.  Their
machine report therefore separates a passing numeric predicate from a supported
capability claim and deliberately exits as an unsupported lifecycle result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from typing import Final, Literal

from bench.epistemic.lifecycle import (
    IRRELEVANT_PROBE,
    NOISE_PROBE,
    RELEVANT_PROBE,
    BinaryRuleTask,
    CollectionPolicy,
    ExactRuleAgent,
    ProbeAction,
    ProbeSpecification,
    diagnose_probe,
    empirically_exact_training_signals,
    evaluate_frozen,
    evaluation_metric,
    make_balanced_task_suite,
    select_probe_by_raw_entropy,
)
from bench.epistemic.runtime_lane import (
    RuntimeIntegrityResult,
    run_runtime_integrity_lane,
)
from prospect.domain import (
    EvaluationRecord,
    EvidenceOrigin,
    ExperienceEvent,
    TrustLevel,
)

_TOL: Final = 1e-12


class TrainingCondition(StrEnum):
    """Causal learning and negative-control conditions."""

    RELEVANT = "relevant"
    FROZEN = "frozen"
    SHUFFLED_LABEL = "shuffled_label"
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"


class EvidenceDisposition(StrEnum):
    """What the diagnostic evidence is allowed to establish."""

    SUPPORTED = "supported"
    REFERENCE_ONLY = "reference_only"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class GateMetric:
    """One machine-readable numeric gate metric."""

    name: str
    value: float
    unit: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One explicit threshold applied to a scalar observation."""

    name: str
    observed: float
    comparator: Literal["ge", "le"]
    threshold: float

    @property
    def passed(self) -> bool:
        if self.comparator == "ge":
            return self.observed + _TOL >= self.threshold
        return self.observed <= self.threshold + _TOL

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "observed": self.observed,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    """Numeric diagnostic plus an independently audited claim disposition."""

    gate_id: str
    claim: str
    metrics: tuple[GateMetric, ...]
    checks: tuple[GateCheck, ...]
    controls: tuple[str, ...]
    disposition: EvidenceDisposition
    audit_reason: str

    @property
    def diagnostic_checks_passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def claim_supported(self) -> bool:
        return self.disposition is EvidenceDisposition.SUPPORTED and self.diagnostic_checks_passed

    @property
    def passed(self) -> bool:
        """Compatibility alias for the audited capability disposition."""

        return self.claim_supported

    def as_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "claim": self.claim,
            "disposition": self.disposition.value,
            "audit_reason": self.audit_reason,
            "diagnostic_checks_passed": self.diagnostic_checks_passed,
            "claim_supported": self.claim_supported,
            "metrics": [metric.as_dict() for metric in self.metrics],
            "checks": [check.as_dict() for check in self.checks],
            "controls": list(self.controls),
        }


@dataclass(frozen=True, slots=True)
class MaturityReport:
    """Serializable audited report; ``passed`` means the full claim is supported."""

    schema: str
    gates: tuple[GateResult, ...]
    limitations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(gate.claim_supported for gate in self.gates)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "gates": [gate.as_dict() for gate in self.gates],
            "limitations": list(self.limitations),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CollectionTrace:
    """Real interaction records acquired by one collection policy."""

    policy: CollectionPolicy
    experiences: tuple[ExperienceEvent, ...]
    relevant_count: int
    irrelevant_count: int
    noise_count: int
    total_expected_information_gain_nats: float
    total_cost: float

    @property
    def environment_steps(self) -> int:
        return len(self.experiences)

    @property
    def relevant_fraction(self) -> float:
        return self.relevant_count / self.environment_steps

    @property
    def information_gain_per_cost(self) -> float:
        return self.total_expected_information_gain_nats / self.total_cost


@dataclass(frozen=True, slots=True)
class TrainingTrace:
    """Agent, real records, updates, and frozen evaluation for one condition."""

    condition: TrainingCondition
    agent: ExactRuleAgent
    evaluation: EvaluationRecord


@dataclass(frozen=True, slots=True)
class RetentionTrace:
    """The required A -> B -> restart -> A sequence and frozen evaluations."""

    sequence: tuple[str, ...]
    agent_before_restart: ExactRuleAgent
    restarted_agent: ExactRuleAgent
    a_after_a: EvaluationRecord
    b_before_b: EvaluationRecord
    b_after_b: EvaluationRecord
    a_after_b: EvaluationRecord
    a_after_restart: EvaluationRecord
    b_after_restart: EvaluationRecord
    checkpoint: str


@dataclass(frozen=True, slots=True)
class MaturityRun:
    """Report plus inspectable evidence objects used to derive it."""

    report: MaturityReport
    runtime_integrity: RuntimeIntegrityResult
    exact_collection: CollectionTrace
    raw_entropy_collection: CollectionTrace
    random_collection: CollectionTrace
    relevant_training: TrainingTrace
    frozen_training: TrainingTrace
    shuffled_training: TrainingTrace
    irrelevant_training: TrainingTrace
    retention: RetentionTrace


def run_maturity_benchmark() -> MaturityRun:
    """Run the complete deterministic E2--E5 evidence program."""

    runtime_integrity = run_runtime_integrity_lane()
    collection_tasks = make_balanced_task_suite("collection", tasks_per_rule=15)
    exact_collection = collect_experience(CollectionPolicy.EXACT_VOI, collection_tasks)
    raw_collection = collect_experience(CollectionPolicy.RAW_ENTROPY, collection_tasks)
    random_collection = collect_experience(CollectionPolicy.RANDOM, collection_tasks)

    training_tasks = make_balanced_task_suite("learn", tasks_per_rule=25)
    relevant = train_condition(TrainingCondition.RELEVANT, training_tasks)
    frozen = train_condition(TrainingCondition.FROZEN, training_tasks)
    shuffled = train_condition(TrainingCondition.SHUFFLED_LABEL, training_tasks)
    irrelevant = train_condition(TrainingCondition.IRRELEVANT_EVIDENCE, training_tasks)
    retention = run_retention_sequence()

    e2 = _collection_gate(
        exact_collection,
        raw_collection,
        random_collection,
        runtime_integrity,
    )
    e3 = _learning_gate(relevant, frozen, shuffled, irrelevant)
    e4 = _behavior_gate(relevant, frozen, shuffled, irrelevant)
    e5 = _retention_gate(retention)
    report = MaturityReport(
        schema="prospect.epistemic.reference-diagnostics.e2-e5.v2",
        gates=(e2, e3, e4, e5),
        limitations=(
            "The E2 collection records are not the records consumed by E3; the "
            "rows use different agent identities and do not form one causal chain.",
            "The exact finite oracle validates semantics and causal custody, "
            "not approximation quality in a neural model.",
            "The E2 integrity lane uses the authoritative runtime; multi-task E3-E5 "
            "still use the benchmark reference learner, so a production learner "
            "adapter remains a separate gate.",
            "E3 changes task-local posterior state and a configuration counter, "
            "not predictive-model, representation, or policy parameters.",
            "E3 evaluates task identities seen during assimilation rather than a disjoint held-out task split.",
            "E4 computes analytic expected utility and does not execute held-out "
            "actions and outcomes in an environment.",
            "Task-scoped posterior slots eliminate parameter interference by "
            "construction; E5 is not yet a hard continual-learning test.",
            "The E5 reference checkpoint omits canonical experiences, transitions, "
            "and update receipts and does not use the production coordinator.",
            "The shuffled-label arm is intentionally derived corrupted evidence, "
            "while the primary learning arm uses observed environment records.",
            "The finite signal schedule matches the two-probe channel distribution "
            "exactly but does not test sensitivity across stochastic seeds.",
            "Checkpointing here is a benchmark-contained reference format, "
            "not the production runtime checkpoint coordinator.",
        ),
    )
    return MaturityRun(
        report=report,
        runtime_integrity=runtime_integrity,
        exact_collection=exact_collection,
        raw_entropy_collection=raw_collection,
        random_collection=random_collection,
        relevant_training=relevant,
        frozen_training=frozen,
        shuffled_training=shuffled,
        irrelevant_training=irrelevant,
        retention=retention,
    )


def collect_experience(
    policy: CollectionPolicy,
    tasks: tuple[BinaryRuleTask, ...],
) -> CollectionTrace:
    """Collect one matched-cost interaction per previously unseen task."""

    agent = ExactRuleAgent(agent_id=f"collector-{policy.value}")
    relevant_count = 0
    irrelevant_count = 0
    noise_count = 0
    total_eig = 0.0

    for index, task in enumerate(tasks):
        if policy is CollectionPolicy.EXACT_VOI:
            action = agent.select_action(task.task_id)
        elif policy is CollectionPolicy.RAW_ENTROPY:
            action = select_probe_by_raw_entropy(agent.posterior(task.task_id))
        else:
            action = (
                ProbeAction.RELEVANT,
                ProbeAction.IRRELEVANT,
                ProbeAction.NOISE,
            )[index % 3]
        if action is ProbeAction.EXPLOIT:
            raise RuntimeError("fresh tasks must have a positive-value relevant probe")
        probe = _probe_for_action(action)
        signal = _collection_signal(task, action, index)
        agent.interact(
            task,
            action,
            signal,
            experience_id=f"{policy.value}-{task.task_id}",
            update=False,
            require_optimal=policy is CollectionPolicy.EXACT_VOI,
            behavior_policy_version={
                CollectionPolicy.EXACT_VOI: "exact-voi-v1",
                CollectionPolicy.RAW_ENTROPY: "raw-observation-entropy-v1",
                CollectionPolicy.RANDOM: "deterministic-random-v1",
            }[policy],
        )
        total_eig += diagnose_probe(0.5, probe).expected_information_gain_nats
        relevant_count += int(action is ProbeAction.RELEVANT)
        irrelevant_count += int(action is ProbeAction.IRRELEVANT)
        noise_count += int(action is ProbeAction.NOISE)

    return CollectionTrace(
        policy=policy,
        experiences=agent.experiences,
        relevant_count=relevant_count,
        irrelevant_count=irrelevant_count,
        noise_count=noise_count,
        total_expected_information_gain_nats=total_eig,
        total_cost=agent.total_probe_cost,
    )


def train_condition(
    condition: TrainingCondition,
    tasks: tuple[BinaryRuleTask, ...],
) -> TrainingTrace:
    """Train or freeze one causal-control arm under the same probe budget."""

    if len(tasks) % 2:
        raise ValueError("training tasks must be paired by opposite rule")
    agent = ExactRuleAgent(agent_id=f"learner-{condition.value}")
    source_by_task = {task.task_id: tasks[index ^ 1] for index, task in enumerate(tasks)}
    for round_index in (0, 1):
        for index, task in enumerate(tasks):
            source_task = source_by_task[task.task_id]
            if condition is TrainingCondition.SHUFFLED_LABEL:
                signal = empirically_exact_training_signals(source_task, round_index)
                action = ProbeAction.RELEVANT
                origin = EvidenceOrigin.DERIVED
                source_id = "shuffled-label-negative-control"
                source_kind = "negative_control"
                trust = TrustLevel.UNTRUSTED
                producer_version = "task-label-permutation-v1"
            elif condition is TrainingCondition.IRRELEVANT_EVIDENCE:
                signal = ((index // 2) + round_index) % 2
                action = ProbeAction.IRRELEVANT
                origin = EvidenceOrigin.OBSERVED
                source_id = "independent-nuisance-sensor"
                source_kind = "irrelevant_sensor"
                trust = TrustLevel.HIGH
                producer_version = None
            else:
                signal = empirically_exact_training_signals(task, round_index)
                action = ProbeAction.RELEVANT
                origin = EvidenceOrigin.OBSERVED
                source_id = "binary-rule-environment"
                source_kind = "diagnostic_sensor"
                trust = TrustLevel.HIGH
                producer_version = None
            agent.interact(
                task,
                action,
                signal,
                experience_id=f"{condition.value}-{task.task_id}-round-{round_index}",
                update=condition is not TrainingCondition.FROZEN,
                require_optimal=condition
                in {
                    TrainingCondition.RELEVANT,
                    TrainingCondition.FROZEN,
                    TrainingCondition.SHUFFLED_LABEL,
                },
                origin=origin,
                source_id=source_id,
                source_kind=source_kind,
                trust=trust,
                producer_version=producer_version,
                behavior_policy_version=(
                    "irrelevant-evidence-control-v1" if condition is TrainingCondition.IRRELEVANT_EVIDENCE else None
                ),
            )
    evaluation = evaluate_frozen(
        agent,
        tasks,
        evaluation_id=f"heldout-{condition.value}",
    )
    return TrainingTrace(condition, agent, evaluation)


def run_retention_sequence() -> RetentionTrace:
    """Execute A -> B -> serialized restart -> A with frozen checks throughout."""

    task_a = BinaryRuleTask("retention-a-000", 0)
    task_b = BinaryRuleTask("retention-b-001", 1)
    agent = ExactRuleAgent(agent_id="retention-agent")

    for round_index in (0, 1):
        agent.interact(
            task_a,
            ProbeAction.RELEVANT,
            task_a.rule,
            experience_id=f"retention-a-round-{round_index}",
            update=True,
            require_optimal=True,
        )
    a_after_a = evaluate_frozen(agent, (task_a,), evaluation_id="retention-a-after-a")
    b_before_b = evaluate_frozen(agent, (task_b,), evaluation_id="retention-b-before-b")

    for round_index in (0, 1):
        agent.interact(
            task_b,
            ProbeAction.RELEVANT,
            task_b.rule,
            experience_id=f"retention-b-round-{round_index}",
            update=True,
            require_optimal=True,
        )
    b_after_b = evaluate_frozen(agent, (task_b,), evaluation_id="retention-b-after-b")
    a_after_b = evaluate_frozen(agent, (task_a,), evaluation_id="retention-a-after-b")
    checkpoint = agent.checkpoint_json()
    restarted = ExactRuleAgent.from_checkpoint_json(checkpoint)
    a_after_restart = evaluate_frozen(
        restarted,
        (task_a,),
        evaluation_id="retention-a-after-restart",
    )
    b_after_restart = evaluate_frozen(
        restarted,
        (task_b,),
        evaluation_id="retention-b-after-restart",
    )
    return RetentionTrace(
        sequence=(
            "learn:A",
            "evaluate:A",
            "learn:B",
            "evaluate:B",
            "evaluate:A",
            "save",
            "restart",
            "evaluate:A",
            "evaluate:B",
        ),
        agent_before_restart=agent,
        restarted_agent=restarted,
        a_after_a=a_after_a,
        b_before_b=b_before_b,
        b_after_b=b_after_b,
        a_after_b=a_after_b,
        a_after_restart=a_after_restart,
        b_after_restart=b_after_restart,
        checkpoint=checkpoint,
    )


def _collection_gate(
    exact: CollectionTrace,
    raw: CollectionTrace,
    random: CollectionTrace,
    runtime_integrity: RuntimeIntegrityResult,
) -> GateResult:
    costs = (exact.total_cost, raw.total_cost, random.total_cost)
    steps = (exact.environment_steps, raw.environment_steps, random.environment_steps)
    best_baseline_rate = max(raw.information_gain_per_cost, random.information_gain_per_cost)
    advantage = exact.information_gain_per_cost - best_baseline_rate
    metrics = (
        GateMetric("exact_relevant_fraction", exact.relevant_fraction, "fraction"),
        GateMetric("raw_entropy_relevant_fraction", raw.relevant_fraction, "fraction"),
        GateMetric("random_relevant_fraction", random.relevant_fraction, "fraction"),
        GateMetric("exact_eig_per_cost", exact.information_gain_per_cost, "nats/utility"),
        GateMetric("raw_entropy_eig_per_cost", raw.information_gain_per_cost, "nats/utility"),
        GateMetric("random_eig_per_cost", random.information_gain_per_cost, "nats/utility"),
        GateMetric("matched_cost_spread", max(costs) - min(costs), "utility"),
        GateMetric("matched_step_spread", float(max(steps) - min(steps)), "step"),
        GateMetric(
            "authoritative_runtime_integrity",
            float(runtime_integrity.passed),
            "boolean",
        ),
    )
    checks = (
        GateCheck("diagnostic_selection_fraction", exact.relevant_fraction, "ge", 0.99),
        GateCheck("target_information_advantage_per_cost", advantage, "ge", 3.0),
        GateCheck("matched_probe_cost", max(costs) - min(costs), "le", 1e-12),
        GateCheck("matched_environment_steps", float(max(steps) - min(steps)), "le", 0.0),
        GateCheck(
            "authoritative_decide_step_observe_learn_path",
            float(runtime_integrity.passed),
            "ge",
            1.0,
        ),
    )
    return GateResult(
        gate_id="E2",
        claim=(
            "an exact known-model VOI reference selects the predefined diagnostic probe under an equal synthetic budget"
        ),
        metrics=metrics,
        checks=checks,
        controls=(
            "raw_observation_entropy",
            "deterministic_random_policy",
            "canonical_runtime_store_and_ledger",
        ),
        disposition=EvidenceDisposition.REFERENCE_ONLY,
        audit_reason=(
            "The arithmetic and one-step runtime custody are valid, but this "
            "collector is not the agent whose experience is consumed by E3."
        ),
    )


def _learning_gate(
    relevant: TrainingTrace,
    frozen: TrainingTrace,
    shuffled: TrainingTrace,
    irrelevant: TrainingTrace,
) -> GateResult:
    learned_loss = evaluation_metric(relevant.evaluation, "mean_log_score")
    frozen_loss = evaluation_metric(frozen.evaluation, "mean_log_score")
    shuffled_loss = evaluation_metric(shuffled.evaluation, "mean_log_score")
    irrelevant_loss = evaluation_metric(irrelevant.evaluation, "mean_log_score")
    agents = (
        relevant.agent,
        frozen.agent,
        shuffled.agent,
        irrelevant.agent,
    )
    costs = tuple(agent.total_probe_cost for agent in agents)
    steps = tuple(agent.total_environment_steps for agent in agents)
    metrics = (
        GateMetric("learned_mean_log_score", learned_loss, "nats"),
        GateMetric("frozen_mean_log_score", frozen_loss, "nats"),
        GateMetric("shuffled_mean_log_score", shuffled_loss, "nats"),
        GateMetric("irrelevant_mean_log_score", irrelevant_loss, "nats"),
        GateMetric("proper_score_improvement_vs_frozen", frozen_loss - learned_loss, "nats"),
        GateMetric("matched_training_cost_spread", max(costs) - min(costs), "utility"),
        GateMetric("matched_training_step_spread", float(max(steps) - min(steps)), "step"),
    )
    checks = (
        GateCheck("proper_score_improves", frozen_loss - learned_loss, "ge", 0.25),
        GateCheck("irrelevant_evidence_does_not_explain_gain", irrelevant_loss - learned_loss, "ge", 0.25),
        GateCheck("shuffled_labels_destroy_gain", shuffled_loss - learned_loss, "ge", 1.0),
        GateCheck("matched_training_cost", max(costs) - min(costs), "le", 1e-12),
        GateCheck("matched_training_steps", float(max(steps) - min(steps)), "le", 0.0),
    )
    return GateResult(
        gate_id="E3",
        claim=("exact Bayesian assimilation improves same-task posterior score under the reference channel model"),
        metrics=metrics,
        checks=checks,
        controls=("frozen_learner", "task_label_permutation", "irrelevant_evidence"),
        disposition=EvidenceDisposition.REFERENCE_ONLY,
        audit_reason=(
            "Only task-local posterior/configuration state changes, and evaluation "
            "reuses training task identities; this is not predictive-model learning "
            "on a disjoint held-out split."
        ),
    )


def _behavior_gate(
    relevant: TrainingTrace,
    frozen: TrainingTrace,
    shuffled: TrainingTrace,
    irrelevant: TrainingTrace,
) -> GateResult:
    learned_utility = evaluation_metric(relevant.evaluation, "mean_external_utility")
    frozen_utility = evaluation_metric(frozen.evaluation, "mean_external_utility")
    shuffled_utility = evaluation_metric(shuffled.evaluation, "mean_external_utility")
    irrelevant_utility = evaluation_metric(irrelevant.evaluation, "mean_external_utility")
    learned_regret = evaluation_metric(relevant.evaluation, "mean_regret")
    frozen_regret = evaluation_metric(frozen.evaluation, "mean_regret")
    agents = (
        relevant.agent,
        frozen.agent,
        shuffled.agent,
        irrelevant.agent,
    )
    immutable_evaluation = all(
        not trace.evaluation.training_updates_allowed and not trace.evaluation.update_receipts
        for trace in (relevant, frozen, shuffled, irrelevant)
    )
    step_spread = max(agent.total_environment_steps for agent in agents) - min(
        agent.total_environment_steps for agent in agents
    )
    metrics = (
        GateMetric("learned_external_utility", learned_utility, "utility"),
        GateMetric("frozen_external_utility", frozen_utility, "utility"),
        GateMetric("shuffled_external_utility", shuffled_utility, "utility"),
        GateMetric("irrelevant_external_utility", irrelevant_utility, "utility"),
        GateMetric("external_utility_gain", learned_utility - frozen_utility, "utility"),
        GateMetric("external_regret_reduction", frozen_regret - learned_regret, "utility"),
        GateMetric("frozen_evaluation_policy", float(immutable_evaluation), "boolean"),
        GateMetric(
            "training_budget_step_spread",
            float(step_spread),
            "step",
        ),
    )
    checks = (
        GateCheck("external_behavior_improves", learned_utility - frozen_utility, "ge", 0.10),
        GateCheck("external_regret_falls", frozen_regret - learned_regret, "ge", 0.10),
        GateCheck("shuffled_control_underperforms", frozen_utility - shuffled_utility, "ge", 0.40),
        GateCheck("evaluation_disables_training", float(immutable_evaluation), "ge", 1.0),
    )
    return GateResult(
        gate_id="E4",
        claim="the learned agent improves executed held-out external behavior",
        metrics=metrics,
        checks=checks,
        controls=("frozen_learner", "task_label_permutation", "irrelevant_evidence", "zero_update_evaluator"),
        disposition=EvidenceDisposition.BLOCKED,
        audit_reason=(
            "The endpoint is analytically computed from the hidden task rule; no "
            "held-out action/outcome interaction is executed."
        ),
    )


def _retention_gate(trace: RetentionTrace) -> GateResult:
    a_initial_utility = evaluation_metric(trace.a_after_a, "mean_external_utility")
    a_after_b_utility = evaluation_metric(trace.a_after_b, "mean_external_utility")
    a_restart_utility = evaluation_metric(trace.a_after_restart, "mean_external_utility")
    b_before_utility = evaluation_metric(trace.b_before_b, "mean_external_utility")
    b_after_utility = evaluation_metric(trace.b_after_b, "mean_external_utility")
    b_restart_utility = evaluation_metric(trace.b_after_restart, "mean_external_utility")
    a_initial_score = evaluation_metric(trace.a_after_a, "mean_log_score")
    a_restart_score = evaluation_metric(trace.a_after_restart, "mean_log_score")
    checkpoint_equal = (
        trace.checkpoint == trace.restarted_agent.checkpoint_json()
        and trace.agent_before_restart.state_digest() == trace.restarted_agent.state_digest()
    )
    metrics = (
        GateMetric("a_utility_after_a", a_initial_utility, "utility"),
        GateMetric("a_utility_after_b", a_after_b_utility, "utility"),
        GateMetric("a_utility_after_restart", a_restart_utility, "utility"),
        GateMetric("b_utility_before_b", b_before_utility, "utility"),
        GateMetric("b_utility_after_b", b_after_utility, "utility"),
        GateMetric("b_utility_after_restart", b_restart_utility, "utility"),
        GateMetric("a_interference_drift", abs(a_after_b_utility - a_initial_utility), "utility"),
        GateMetric("a_restart_utility_drift", abs(a_restart_utility - a_initial_utility), "utility"),
        GateMetric("a_restart_log_score_drift", abs(a_restart_score - a_initial_score), "nats"),
        GateMetric("canonical_checkpoint_parity", float(checkpoint_equal), "boolean"),
    )
    checks = (
        GateCheck("b_learns_after_a", b_after_utility - b_before_utility, "ge", 0.20),
        GateCheck("a_survives_learning_b", abs(a_after_b_utility - a_initial_utility), "le", 1e-12),
        GateCheck("a_utility_survives_restart", abs(a_restart_utility - a_initial_utility), "le", 1e-12),
        GateCheck("a_score_survives_restart", abs(a_restart_score - a_initial_score), "le", 1e-12),
        GateCheck("b_utility_survives_restart", abs(b_restart_utility - b_after_utility), "le", 1e-12),
        GateCheck("checkpoint_roundtrip_is_canonical", float(checkpoint_equal), "ge", 1.0),
    )
    return GateResult(
        gate_id="E5",
        claim=(
            "the same behavioral gain survives shared-state interference and a production checkpoint/process restart"
        ),
        metrics=metrics,
        checks=checks,
        controls=("pre_b_baseline", "post_b_a_recheck", "canonical_save_restart_roundtrip"),
        disposition=EvidenceDisposition.BLOCKED,
        audit_reason=(
            "Independent task slots prevent interference, there is no A "
            "pre-learning baseline, and the benchmark checkpoint omits canonical "
            "experience/transition/update custody."
        ),
    )


def _collection_signal(task: BinaryRuleTask, action: ProbeAction, index: int) -> int:
    if action is ProbeAction.RELEVANT:
        return empirically_exact_training_signals(task, 0)
    return (index // 2) % 2


def _probe_for_action(action: ProbeAction) -> ProbeSpecification:
    if action is ProbeAction.RELEVANT:
        return RELEVANT_PROBE
    if action is ProbeAction.IRRELEVANT:
        return IRRELEVANT_PROBE
    if action is ProbeAction.NOISE:
        return NOISE_PROBE
    raise ValueError("exploit has no probe specification")


def metric_by_name(gate: GateResult, name: str) -> float:
    """Extract a named report metric."""

    for metric in gate.metrics:
        if metric.name == name:
            return metric.value
    raise KeyError(name)


def gate_by_id(report: MaturityReport, gate_id: str) -> GateResult:
    """Extract one gate result by stable id."""

    for gate in report.gates:
        if gate.gate_id == gate_id:
            return gate
    raise KeyError(gate_id)


def assert_matched_resources(*agents: ExactRuleAgent) -> None:
    """Raise if agents do not have exactly matched step and probe-cost budgets."""

    if not agents:
        raise ValueError("at least one agent is required")
    steps = {agent.total_environment_steps for agent in agents}
    costs = {agent.total_probe_cost for agent in agents}
    if len(steps) != 1 or len(costs) != 1:
        raise AssertionError("agent resource budgets are not matched")


def report_is_deterministic(first: MaturityReport, second: MaturityReport) -> bool:
    """Compare canonical reports without relying on object identity."""

    return isclose(float(first.passed), float(second.passed)) and first.to_json() == second.to_json()


__all__ = (
    "CollectionTrace",
    "EvidenceDisposition",
    "GateCheck",
    "GateMetric",
    "GateResult",
    "MaturityReport",
    "MaturityRun",
    "RetentionTrace",
    "TrainingCondition",
    "TrainingTrace",
    "assert_matched_resources",
    "collect_experience",
    "gate_by_id",
    "metric_by_name",
    "report_is_deterministic",
    "run_maturity_benchmark",
    "run_retention_sequence",
    "train_condition",
)
