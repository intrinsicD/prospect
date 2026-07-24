"""Behaviorally independent exact arithmetic for the hidden-actuator fixture.

This module intentionally does not call Prospect's Bayesian or information
functions. Every probability and value calculation is separately enumerated
with :class:`fractions.Fraction`.

The module shares immutable action and terminal identifiers with the floating
fixture by importing :mod:`bench.active_acquisition.problem`; that module has
transitive Prospect imports. Q0 therefore establishes behavioral agreement
and no forbidden direct call dependency, not process or transitive isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from bench.active_acquisition.problem import (
    ACQUISITION_ACTIONS,
    AcquisitionAction,
    AcquisitionKind,
    TerminalAction,
)

_ZERO: Final = Fraction(0)
_ONE: Final = Fraction(1)
_HALF: Final = Fraction(1, 2)
_EXPLOIT_RELIABILITY: Final = Fraction(9, 10)


@dataclass(frozen=True, slots=True)
class ExactAcquisitionEvaluation:
    """One independently enumerated action value."""

    action: AcquisitionAction
    outcome_probabilities: tuple[Fraction, ...]
    expected_immediate_payoff: Fraction
    prior_terminal_value: Fraction
    expected_terminal_value_after_observation: Fraction
    expected_decision_value: Fraction
    action_cost: Fraction
    acquisition_cost: Fraction
    net_incremental_value: Fraction
    expected_episode_value: Fraction


@dataclass(frozen=True, slots=True)
class ExactOracleDecision:
    """Exact action choice and all values used to make it."""

    selected_action: AcquisitionAction
    evaluations: tuple[ExactAcquisitionEvaluation, ...]

    @property
    def selected_evaluation(self) -> ExactAcquisitionEvaluation:
        return next(row for row in self.evaluations if row.action == self.selected_action)


class FractionOracle:
    """Exhaustive exact arithmetic for the declared two-stage equations."""

    def likelihoods(
        self,
        action: AcquisitionAction,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        if action.kind is AcquisitionKind.SKIP:
            return ((_ONE,), (_ONE,))
        if action.kind is AcquisitionKind.NUISANCE_SCAN:
            quarter = Fraction(1, 4)
            row = (quarter, quarter, quarter, quarter)
            return (row, row)
        accuracy = {
            AcquisitionKind.WEAK_PULSE: Fraction(7, 10),
            AcquisitionKind.STRONG_PULSE: Fraction(9, 10),
            AcquisitionKind.OVERPOWERED_PULSE: _ONE,
        }[action.kind]
        error = _ONE - accuracy
        if action.command == 1:
            return ((accuracy, error), (error, accuracy))
        return ((error, accuracy), (accuracy, error))

    def outcomes(self, action: AcquisitionAction) -> tuple[int, ...]:
        if action.kind is AcquisitionKind.SKIP:
            return (0,)
        if action.kind is AcquisitionKind.NUISANCE_SCAN:
            return (0, 1, 2, 3)
        return (-1, 1)

    def action_cost(self, action: AcquisitionAction) -> Fraction:
        return {
            AcquisitionKind.SKIP: _ZERO,
            AcquisitionKind.WEAK_PULSE: Fraction(53, 100),
            AcquisitionKind.STRONG_PULSE: Fraction(29, 50),
            AcquisitionKind.OVERPOWERED_PULSE: Fraction(19, 20),
            AcquisitionKind.NUISANCE_SCAN: _ZERO,
        }[action.kind]

    def acquisition_cost(self, action: AcquisitionAction) -> Fraction:
        return Fraction(1, 100) if action.kind is AcquisitionKind.NUISANCE_SCAN else _ZERO

    def posterior_direct(
        self,
        action: AcquisitionAction,
        outcome: int,
        *,
        prior_direct: Fraction = _HALF,
    ) -> Fraction:
        self._require_prior(prior_direct)
        outcomes = self.outcomes(action)
        try:
            index = outcomes.index(outcome)
        except ValueError as error:
            raise ValueError(f"{outcome!r} is not an outcome of {action.action_id}") from error
        reversed_likelihood, direct_likelihood = self.likelihoods(action)
        direct_joint = prior_direct * direct_likelihood[index]
        evidence_probability = direct_joint + (_ONE - prior_direct) * reversed_likelihood[index]
        if evidence_probability == 0:
            raise ValueError("cannot condition on a zero-probability outcome")
        return direct_joint / evidence_probability

    def expected_terminal_payoff(
        self,
        action: TerminalAction,
        *,
        prior_direct: Fraction,
    ) -> Fraction:
        self._require_prior(prior_direct)
        if action is TerminalAction.DIRECT:
            return prior_direct * _EXPLOIT_RELIABILITY + (_ONE - prior_direct) * (_ONE - _EXPLOIT_RELIABILITY)
        return (_ONE - prior_direct) * _EXPLOIT_RELIABILITY + prior_direct * (_ONE - _EXPLOIT_RELIABILITY)

    def terminal_value(self, *, prior_direct: Fraction) -> Fraction:
        return max(
            self.expected_terminal_payoff(TerminalAction.DIRECT, prior_direct=prior_direct),
            self.expected_terminal_payoff(TerminalAction.REVERSED, prior_direct=prior_direct),
        )

    def evaluate(
        self,
        action: AcquisitionAction,
        *,
        prior_direct: Fraction = _HALF,
    ) -> ExactAcquisitionEvaluation:
        self._require_prior(prior_direct)
        reversed_likelihood, direct_likelihood = self.likelihoods(action)
        outcomes = self.outcomes(action)
        outcome_probabilities = tuple(
            (_ONE - prior_direct) * reversed_probability + prior_direct * direct_probability
            for reversed_probability, direct_probability in zip(
                reversed_likelihood,
                direct_likelihood,
                strict=True,
            )
        )
        expected_after = _ZERO
        expected_immediate = _ZERO
        for index, (outcome, probability) in enumerate(zip(outcomes, outcome_probabilities, strict=True)):
            if probability == 0:
                continue
            direct_joint = prior_direct * direct_likelihood[index]
            posterior_direct = direct_joint / probability
            expected_after += probability * self.terminal_value(prior_direct=posterior_direct)
            if (
                action.kind
                in {
                    AcquisitionKind.WEAK_PULSE,
                    AcquisitionKind.STRONG_PULSE,
                    AcquisitionKind.OVERPOWERED_PULSE,
                }
                and outcome == 1
            ):
                expected_immediate += probability
        prior_value = self.terminal_value(prior_direct=prior_direct)
        decision_value = expected_after - prior_value
        action_cost = self.action_cost(action)
        acquisition_cost = self.acquisition_cost(action)
        net_increment = expected_immediate + decision_value - action_cost - acquisition_cost
        return ExactAcquisitionEvaluation(
            action=action,
            outcome_probabilities=outcome_probabilities,
            expected_immediate_payoff=expected_immediate,
            prior_terminal_value=prior_value,
            expected_terminal_value_after_observation=expected_after,
            expected_decision_value=decision_value,
            action_cost=action_cost,
            acquisition_cost=acquisition_cost,
            net_incremental_value=net_increment,
            expected_episode_value=prior_value + net_increment,
        )

    def decide(
        self,
        *,
        prior_direct: Fraction = _HALF,
    ) -> ExactOracleDecision:
        evaluations = tuple(self.evaluate(action, prior_direct=prior_direct) for action in ACQUISITION_ACTIONS)
        # ``max`` returns the first maximum, preserving the declared action
        # order independently of action-label spelling.
        selected = max(evaluations, key=lambda row: row.expected_episode_value)
        return ExactOracleDecision(
            selected_action=selected.action,
            evaluations=evaluations,
        )

    @staticmethod
    def _require_prior(prior_direct: Fraction) -> None:
        if not _ZERO <= prior_direct <= _ONE:
            raise ValueError("prior_direct must be in [0, 1]")


__all__ = (
    "ExactAcquisitionEvaluation",
    "ExactOracleDecision",
    "FractionOracle",
)
