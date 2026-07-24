"""Acquisition policies and causal controls for the hidden-actuator fixture."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import Final

from bench.active_acquisition.oracle import FractionOracle
from bench.active_acquisition.problem import (
    NUISANCE_SCAN,
    OVERPOWERED_POSITIVE,
    SKIP,
    STRONG_POSITIVE,
    WEAK_POSITIVE,
    AcquisitionAction,
    AcquisitionKind,
    HiddenActuatorProblem,
)


@dataclass(frozen=True, slots=True)
class ActionScore:
    """One policy-specific scalar used only for selecting an action."""

    action: AcquisitionAction
    value: float | None
    score_kind: str
    unit: str | None


@dataclass(frozen=True, slots=True)
class AcquisitionPolicyDecision:
    """Selected action plus the complete pre-action score table."""

    policy: str
    selected_action: AcquisitionAction
    scores: tuple[ActionScore, ...]

    @property
    def selected_score(self) -> float | None:
        return next(row.value for row in self.scores if row.action == self.selected_action)


_REPRESENTATIVE: Final = {
    AcquisitionKind.SKIP: SKIP,
    AcquisitionKind.WEAK_PULSE: WEAK_POSITIVE,
    AcquisitionKind.STRONG_PULSE: STRONG_POSITIVE,
    AcquisitionKind.OVERPOWERED_PULSE: OVERPOWERED_POSITIVE,
    AcquisitionKind.NUISANCE_SCAN: NUISANCE_SCAN,
}

# Each action receives another action's decision-value estimate while retaining
# its own utility and costs. This structured map is the sole implementation
# source of truth and is compared directly with the protocol.
SHUFFLED_INFORMATION_SOURCE_BY_ACTION: Final = {
    "skip": "skip",
    "weak": "overpowered",
    "strong": "weak",
    "overpowered": "strong",
    "nuisance": "nuisance",
}
_ACTION_KIND_BY_ID: Final = {representative.action_id: kind for kind, representative in _REPRESENTATIVE.items()}


def prospect_voi(
    problem: HiddenActuatorProblem,
    *,
    prior_direct: float | None = None,
) -> AcquisitionPolicyDecision:
    """Select by external utility plus net decision-relevant information value."""

    scores = tuple(
        ActionScore(
            action,
            problem.diagnose(action, prior_direct=prior_direct).expected_episode_value,
            "expected_episode_net_return",
            "return",
        )
        for action in problem.acquisition_actions
    )
    return _best("prospect_expected_return", scores)


def goal_only(
    problem: HiddenActuatorProblem,
    *,
    prior_direct: float | None = None,
) -> AcquisitionPolicyDecision:
    """Ignore future decision improvement while retaining every declared cost."""

    scores = tuple(
        ActionScore(
            action,
            (
                problem.diagnose(action, prior_direct=prior_direct).prior_terminal_value
                + problem.diagnose(action, prior_direct=prior_direct).expected_immediate_payoff
                - action.action_cost
                - action.acquisition_cost
            ),
            "goal_only_expected_return",
            "return",
        )
        for action in problem.acquisition_actions
    )
    return _best("goal_only", scores)


def raw_entropy(
    problem: HiddenActuatorProblem,
    *,
    prior_direct: float | None = None,
) -> AcquisitionPolicyDecision:
    """Select the most unpredictable observation, irrespective of relevance."""

    scores = tuple(
        ActionScore(
            action,
            problem.diagnose(action, prior_direct=prior_direct).observation_entropy_nats,
            "raw_observation_entropy",
            "nats",
        )
        for action in problem.acquisition_actions
    )
    return _best("raw_observation_entropy", scores)


def eig_only(
    problem: HiddenActuatorProblem,
    *,
    prior_direct: float | None = None,
) -> AcquisitionPolicyDecision:
    """Select mutual information about the regime while ignoring value and cost."""

    scores = tuple(
        ActionScore(
            action,
            problem.diagnose(action, prior_direct=prior_direct).expected_information_gain_nats,
            "expected_information_gain",
            "nats",
        )
        for action in problem.acquisition_actions
    )
    return _best("eig_only", scores)


def uniform_selector_vector(
    problem: HiddenActuatorProblem,
    *,
    seed: int,
) -> tuple[str, int, AcquisitionAction]:
    """Return the declared full digest, modulo index, and selected action."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("uniform random seed must be a nonnegative integer")
    payload = f"WM-002|0.2.0-q|uniform|{seed}".encode()
    digest = sha256(payload).hexdigest()
    index = int(digest, 16) % len(problem.acquisition_actions)
    return digest, index, problem.acquisition_actions[index]


def uniform_random(
    problem: HiddenActuatorProblem,
    *,
    seed: int,
) -> AcquisitionPolicyDecision:
    """Select uniformly from the exact declared action list."""

    _, _, selected = uniform_selector_vector(problem, seed=seed)
    scores = tuple(ActionScore(action, None, "uniform_random", None) for action in problem.acquisition_actions)
    return AcquisitionPolicyDecision(
        policy="uniform_random",
        selected_action=selected,
        scores=scores,
    )


def shuffled_information(
    problem: HiddenActuatorProblem,
    *,
    prior_direct: float | None = None,
) -> AcquisitionPolicyDecision:
    """Break the action/information link while preserving kind-level scores."""

    source_values = {
        kind: problem.diagnose(
            representative,
            prior_direct=prior_direct,
        ).expected_decision_value
        for kind, representative in _REPRESENTATIVE.items()
    }
    scores = []
    for action in problem.acquisition_actions:
        diagnostics = problem.diagnose(action, prior_direct=prior_direct)
        shuffled_source = _ACTION_KIND_BY_ID[SHUFFLED_INFORMATION_SOURCE_BY_ACTION[action.action_id]]
        score = (
            diagnostics.prior_terminal_value
            + diagnostics.expected_immediate_payoff
            + source_values[shuffled_source]
            - diagnostics.expected_action_cost
            - diagnostics.expected_acquisition_cost
        )
        scores.append(
            ActionScore(
                action,
                score,
                "shuffled_information_expected_episode_net_return",
                "return",
            )
        )
    return _best("shuffled_information", tuple(scores))


def oracle(
    problem: HiddenActuatorProblem,
    *,
    prior_direct: float | None = None,
) -> AcquisitionPolicyDecision:
    """Select with the independent exact finite Bellman oracle."""

    prior = problem.prior_direct if prior_direct is None else prior_direct
    exact_prior = Fraction(str(prior))
    exact = FractionOracle().decide(prior_direct=exact_prior)
    scores = tuple(
        ActionScore(
            row.action,
            float(row.expected_episode_value),
            "independent_expected_episode_net_return",
            "return",
        )
        for row in exact.evaluations
    )
    return AcquisitionPolicyDecision(
        policy="independent_fraction_oracle",
        selected_action=exact.selected_action,
        scores=scores,
    )


def _best(policy: str, scores: tuple[ActionScore, ...]) -> AcquisitionPolicyDecision:
    if not scores:
        raise ValueError("an acquisition policy requires at least one action")

    # ``max`` returns the first maximum, preserving the declared candidate
    # order carried by ``scores``.
    def numeric_value(row: ActionScore) -> float:
        if row.value is None:
            raise ValueError("ranked acquisition scores must be numeric")
        return row.value

    selected = max(scores, key=numeric_value)
    return AcquisitionPolicyDecision(
        policy=policy,
        selected_action=selected.action,
        scores=scores,
    )


__all__ = (
    "ActionScore",
    "AcquisitionPolicyDecision",
    "eig_only",
    "SHUFFLED_INFORMATION_SOURCE_BY_ACTION",
    "goal_only",
    "oracle",
    "prospect_voi",
    "raw_entropy",
    "shuffled_information",
    "uniform_random",
    "uniform_selector_vector",
)
