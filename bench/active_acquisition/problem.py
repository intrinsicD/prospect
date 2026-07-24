"""Finite hidden-actuator problem for decision-relevant active acquisition.

The fixture has one acquisition/control move followed by one irrevocable
terminal command.  A hidden actuator regime ``theta`` is either reversed
(``-1``) or direct (``+1``).  A signed pulse both moves the plant and reveals
evidence about that regime.  This makes the first move dual-purpose while
remaining small enough for an independent exact oracle.

All floating-point epistemic quantities in this module are deliberately
computed through :mod:`prospect.epistemics.information`.  The independent
``Fraction`` implementation lives in :mod:`bench.active_acquisition.oracle`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final

from prospect.epistemics.information import (
    bayes_posterior,
    entropy,
    expected_value_of_sample_information,
    predictive_distribution,
)


class ActuatorRegime(IntEnum):
    """Environment-hidden sign of the actuator."""

    REVERSED = -1
    DIRECT = 1


class AcquisitionKind(StrEnum):
    """Available first-stage interventions."""

    SKIP = "skip"
    WEAK_PULSE = "weak_pulse"
    STRONG_PULSE = "strong_pulse"
    OVERPOWERED_PULSE = "overpowered_pulse"
    NUISANCE_SCAN = "nuisance_scan"


class TerminalAction(IntEnum):
    """Irrevocable terminal command."""

    REVERSED = -1
    DIRECT = 1


_PULSE_ACCURACY: Final = {
    AcquisitionKind.WEAK_PULSE: 0.70,
    AcquisitionKind.STRONG_PULSE: 0.90,
    AcquisitionKind.OVERPOWERED_PULSE: 1.00,
}
_PULSE_ACTION_COST: Final = {
    AcquisitionKind.WEAK_PULSE: 0.53,
    AcquisitionKind.STRONG_PULSE: 0.58,
    AcquisitionKind.OVERPOWERED_PULSE: 0.95,
}
_NUISANCE_ACQUISITION_COST: Final = 0.01
_EXPLOIT_RELIABILITY: Final = 0.90


@dataclass(frozen=True, slots=True)
class AcquisitionAction:
    """One sealed acquisition action and, for pulses, its signed command."""

    kind: AcquisitionKind
    command: int = 0

    def __post_init__(self) -> None:
        if self.is_pulse:
            if self.command not in (-1, 1):
                raise ValueError("pulse command must be -1 or +1")
        elif self.command != 0:
            raise ValueError("skip and nuisance actions require command 0")

    @property
    def is_pulse(self) -> bool:
        return self.kind in _PULSE_ACCURACY

    @property
    def accuracy(self) -> float | None:
        return _PULSE_ACCURACY.get(self.kind)

    @property
    def action_cost(self) -> float:
        """Physical execution cost, separate from information acquisition cost."""

        return _PULSE_ACTION_COST.get(self.kind, 0.0)

    @property
    def acquisition_cost(self) -> float:
        """Cost of obtaining information rather than moving the plant."""

        return _NUISANCE_ACQUISITION_COST if self.kind is AcquisitionKind.NUISANCE_SCAN else 0.0

    @property
    def action_id(self) -> str:
        canonical = {
            AcquisitionKind.SKIP: "skip",
            AcquisitionKind.WEAK_PULSE: "weak",
            AcquisitionKind.STRONG_PULSE: "strong",
            AcquisitionKind.OVERPOWERED_PULSE: "overpowered",
            AcquisitionKind.NUISANCE_SCAN: "nuisance",
        }[self.kind]
        if self.is_pulse and self.command == -1:
            return f"{canonical}:sign_inverted_non_candidate"
        return canonical


SKIP: Final = AcquisitionAction(AcquisitionKind.SKIP)
WEAK_POSITIVE: Final = AcquisitionAction(AcquisitionKind.WEAK_PULSE, 1)
WEAK_NEGATIVE: Final = AcquisitionAction(AcquisitionKind.WEAK_PULSE, -1)
STRONG_POSITIVE: Final = AcquisitionAction(AcquisitionKind.STRONG_PULSE, 1)
STRONG_NEGATIVE: Final = AcquisitionAction(AcquisitionKind.STRONG_PULSE, -1)
OVERPOWERED_POSITIVE: Final = AcquisitionAction(AcquisitionKind.OVERPOWERED_PULSE, 1)
OVERPOWERED_NEGATIVE: Final = AcquisitionAction(AcquisitionKind.OVERPOWERED_PULSE, -1)
NUISANCE_SCAN: Final = AcquisitionAction(AcquisitionKind.NUISANCE_SCAN)

ACQUISITION_ACTIONS: Final = (
    SKIP,
    WEAK_POSITIVE,
    STRONG_POSITIVE,
    OVERPOWERED_POSITIVE,
    NUISANCE_SCAN,
)
NONCANDIDATE_SIGN_INVARIANCE_ACTIONS: Final = (
    WEAK_NEGATIVE,
    STRONG_NEGATIVE,
    OVERPOWERED_NEGATIVE,
)
TERMINAL_ACTIONS: Final = (TerminalAction.DIRECT, TerminalAction.REVERSED)


@dataclass(frozen=True, slots=True)
class AcquisitionDiagnostics:
    """Prospective quantities for one acquisition action."""

    action: AcquisitionAction
    outcome_probabilities: tuple[float, ...]
    observation_entropy_nats: float
    expected_information_gain_nats: float
    expected_decision_value: float
    expected_immediate_payoff: float
    expected_action_cost: float
    expected_acquisition_cost: float
    prior_terminal_value: float
    net_incremental_value: float
    expected_episode_value: float


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """One realized first-stage response without exposing the hidden regime."""

    action: AcquisitionAction
    outcome: int
    task_payoff: float
    action_cost: float
    acquisition_cost: float
    net_reward: float


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """One realized terminal response without exposing the hidden regime."""

    action: TerminalAction
    success: bool
    net_reward: float


@dataclass(frozen=True, slots=True)
class HiddenActuatorProblem:
    """Two-stage hidden-sign actuator with a decision-relevant probe."""

    prior_direct: float = 0.5

    def __post_init__(self) -> None:
        _require_probability(self.prior_direct, "prior_direct")

    @property
    def acquisition_actions(self) -> tuple[AcquisitionAction, ...]:
        return ACQUISITION_ACTIONS

    @property
    def terminal_actions(self) -> tuple[TerminalAction, ...]:
        return TERMINAL_ACTIONS

    @property
    def exploit_reliability(self) -> float:
        return _EXPLOIT_RELIABILITY

    def outcomes(self, action: AcquisitionAction) -> tuple[int, ...]:
        if action.is_pulse:
            return (-1, 1)
        if action.kind is AcquisitionKind.NUISANCE_SCAN:
            return (0, 1, 2, 3)
        return (0,)

    def likelihoods(self, action: AcquisitionAction) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Rows are ``P(outcome | theta=-1)`` and ``P(outcome | theta=+1)``."""

        if action.kind is AcquisitionKind.SKIP:
            return ((1.0,), (1.0,))
        if action.kind is AcquisitionKind.NUISANCE_SCAN:
            return ((0.25, 0.25, 0.25, 0.25), (0.25, 0.25, 0.25, 0.25))
        accuracy = action.accuracy
        if accuracy is None:
            raise ValueError("pulse likelihood requires a declared accuracy")
        if action.command == 1:
            return (
                (accuracy, 1.0 - accuracy),
                (1.0 - accuracy, accuracy),
            )
        return (
            (1.0 - accuracy, accuracy),
            (accuracy, 1.0 - accuracy),
        )

    def posterior_direct(
        self,
        action: AcquisitionAction,
        outcome: int,
        *,
        prior_direct: float | None = None,
    ) -> float:
        outcomes = self.outcomes(action)
        try:
            outcome_index = outcomes.index(outcome)
        except ValueError as error:
            raise ValueError(f"{outcome!r} is not an outcome of {action.action_id}") from error
        prior = self._prior(prior_direct)
        return bayes_posterior(
            (1.0 - prior, prior),
            self.likelihoods(action),
            outcome_index,
        )[1]

    def terminal_utilities(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Rows follow ``TERMINAL_ACTIONS``; columns are reversed/direct regimes."""

        reliability = self.exploit_reliability
        return (
            (1.0 - reliability, reliability),
            (reliability, 1.0 - reliability),
        )

    def expected_terminal_payoff(
        self,
        action: TerminalAction,
        *,
        prior_direct: float | None = None,
    ) -> float:
        prior = self._prior(prior_direct)
        if action is TerminalAction.DIRECT:
            return prior * self.exploit_reliability + (1.0 - prior) * (1.0 - self.exploit_reliability)
        return (1.0 - prior) * self.exploit_reliability + prior * (1.0 - self.exploit_reliability)

    def best_terminal_action(self, *, prior_direct: float | None = None) -> TerminalAction:
        prior = self._prior(prior_direct)
        return max(
            self.terminal_actions,
            key=lambda action: self.expected_terminal_payoff(action, prior_direct=prior),
        )

    def terminal_value(self, *, prior_direct: float | None = None) -> float:
        prior = self._prior(prior_direct)
        return self.expected_terminal_payoff(
            self.best_terminal_action(prior_direct=prior),
            prior_direct=prior,
        )

    def expected_immediate_payoff(
        self,
        action: AcquisitionAction,
        *,
        prior_direct: float | None = None,
    ) -> float:
        if not action.is_pulse:
            return 0.0
        prior = self._prior(prior_direct)
        predictive = predictive_distribution(
            (1.0 - prior, prior),
            self.likelihoods(action),
        )
        return predictive[self.outcomes(action).index(1)]

    def diagnose(
        self,
        action: AcquisitionAction,
        *,
        prior_direct: float | None = None,
    ) -> AcquisitionDiagnostics:
        """Score an acquisition action with Prospect's existing exact functions."""

        prior = self._prior(prior_direct)
        belief = (1.0 - prior, prior)
        likelihoods = self.likelihoods(action)
        predictive = predictive_distribution(belief, likelihoods)
        information = expected_value_of_sample_information(
            belief,
            likelihoods,
            self.terminal_utilities(),
            acquisition_cost=action.acquisition_cost,
        )
        immediate = self.expected_immediate_payoff(action, prior_direct=prior)
        prior_value = self.terminal_value(prior_direct=prior)
        net_increment = immediate + information.net_value - action.action_cost
        return AcquisitionDiagnostics(
            action=action,
            outcome_probabilities=predictive,
            observation_entropy_nats=entropy(predictive),
            expected_information_gain_nats=information.expected_information_gain_nats,
            expected_decision_value=information.expected_decision_value,
            expected_immediate_payoff=immediate,
            expected_action_cost=action.action_cost,
            expected_acquisition_cost=action.acquisition_cost,
            prior_terminal_value=prior_value,
            net_incremental_value=net_increment,
            expected_episode_value=prior_value + net_increment,
        )

    def realize_acquisition(
        self,
        action: AcquisitionAction,
        regime: ActuatorRegime,
        *,
        unit_interval: float,
    ) -> AcquisitionResult:
        """Map a predeclared unit-interval draw to one acquisition outcome."""

        draw = _unit_interval(unit_interval)
        if action.kind is AcquisitionKind.SKIP:
            outcome = 0
        elif action.kind is AcquisitionKind.NUISANCE_SCAN:
            outcome = min(int(draw * 4.0), 3)
        else:
            accuracy = action.accuracy
            if accuracy is None:
                raise ValueError("pulse realization requires a declared accuracy")
            nominal = int(regime) * action.command
            outcome = nominal if draw < accuracy else -nominal
        task_payoff = 1.0 if action.is_pulse and outcome == 1 else 0.0
        net_reward = task_payoff - action.action_cost - action.acquisition_cost
        return AcquisitionResult(
            action=action,
            outcome=outcome,
            task_payoff=task_payoff,
            action_cost=action.action_cost,
            acquisition_cost=action.acquisition_cost,
            net_reward=net_reward,
        )

    def realize_terminal(
        self,
        action: TerminalAction,
        regime: ActuatorRegime,
        *,
        unit_interval: float,
    ) -> TerminalResult:
        """Map a predeclared unit-interval draw to the terminal success bit."""

        draw = _unit_interval(unit_interval)
        success_probability = self.exploit_reliability if int(action) == int(regime) else 1.0 - self.exploit_reliability
        success = draw < success_probability
        return TerminalResult(
            action=action,
            success=success,
            net_reward=float(success),
        )

    def _prior(self, prior_direct: float | None) -> float:
        prior = self.prior_direct if prior_direct is None else prior_direct
        _require_probability(prior, "prior_direct")
        return prior


def _require_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability")


def _unit_interval(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("unit_interval must be finite and in [0, 1)")
    return value


__all__ = (
    "ACQUISITION_ACTIONS",
    "NONCANDIDATE_SIGN_INVARIANCE_ACTIONS",
    "NUISANCE_SCAN",
    "OVERPOWERED_NEGATIVE",
    "OVERPOWERED_POSITIVE",
    "SKIP",
    "STRONG_NEGATIVE",
    "STRONG_POSITIVE",
    "TERMINAL_ACTIONS",
    "WEAK_NEGATIVE",
    "WEAK_POSITIVE",
    "AcquisitionAction",
    "AcquisitionDiagnostics",
    "AcquisitionKind",
    "AcquisitionResult",
    "ActuatorRegime",
    "HiddenActuatorProblem",
    "TerminalAction",
    "TerminalResult",
)
