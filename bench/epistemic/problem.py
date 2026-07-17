"""An exact two-hypothesis diagnostic decision problem.

The benchmark keeps three quantities separate:

* observation entropy: how unpredictable a probe's output is;
* information gain: how much the output identifies the original hypothesis;
* value of information: how much the output can improve the downstream decision.

That separation supplies analytic negative controls for noisy observations,
irrelevant observations, and interventions that manufacture physical certainty.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from math import isfinite, log2

_NUMERICAL_ZERO = 1e-12


class Hypothesis(Enum):
    """The two mutually exclusive latent hypotheses."""

    LEFT = "left"
    RIGHT = "right"


class ProbeKind(Enum):
    """Available diagnostic interventions."""

    INFORMATIVE = "informative"
    IRRELEVANT_BIT = "irrelevant_bit"
    NOISY_SIGNAL = "noisy_signal"
    DESTRUCTIVE_CERTAINTY = "destructive_certainty"


class ProbeOutcome(Enum):
    """Possible observations returned by a probe."""

    ZERO = 0
    ONE = 1
    DESTROYED = "destroyed"


class ExploitAction(Enum):
    """Terminal decisions whose payoffs define decision relevance."""

    GUESS_LEFT = "guess_left"
    GUESS_RIGHT = "guess_right"
    KNOWN_SAFE = "known_safe"


class FutureEvidenceError(ValueError):
    """Raised when a decision attempts to consume evidence from the future."""


@dataclass(frozen=True)
class Probe:
    """A diagnostic intervention and its explicit sampling cost."""

    kind: ProbeKind
    cost: float = 0.0
    accuracy: float = 0.5

    def __post_init__(self) -> None:
        _require_finite(self.cost, "cost")
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")
        _require_probability(self.accuracy, "accuracy")
        if self.kind is ProbeKind.INFORMATIVE:
            if self.accuracy <= 0.5:
                raise ValueError("an informative probe must have accuracy greater than 0.5")
        elif self.accuracy != 0.5:
            raise ValueError("accuracy is fixed at 0.5 for non-informative probe kinds")

    @classmethod
    def informative(cls, *, cost: float = 0.0, accuracy: float = 0.9) -> Probe:
        """Construct a symmetric binary diagnostic channel."""

        return cls(ProbeKind.INFORMATIVE, cost=cost, accuracy=accuracy)

    @classmethod
    def irrelevant_bit(cls, *, cost: float = 0.0) -> Probe:
        """Construct a fair nuisance bit independent of the hypothesis."""

        return cls(ProbeKind.IRRELEVANT_BIT, cost=cost)

    @classmethod
    def noisy_signal(cls, *, cost: float = 0.0) -> Probe:
        """Construct an irreducibly random pseudo-signal."""

        return cls(ProbeKind.NOISY_SIGNAL, cost=cost)

    @classmethod
    def destructive_certainty(cls, *, cost: float = 0.0) -> Probe:
        """Construct an intervention that erases, rather than diagnoses, the state."""

        return cls(ProbeKind.DESTRUCTIVE_CERTAINTY, cost=cost)

    @property
    def is_admissible_epistemic_action(self) -> bool:
        """Whether the intervention can count as an information-seeking action."""

        return self.kind is not ProbeKind.DESTRUCTIVE_CERTAINTY


@dataclass(frozen=True)
class Evidence:
    """A time-stamped observation produced by a probe."""

    probe: Probe
    outcome: ProbeOutcome
    observed_at: int

    def __post_init__(self) -> None:
        if self.observed_at < 0:
            raise ValueError("observed_at must be non-negative")


@dataclass(frozen=True)
class ExploitChoice:
    """The best terminal decision under a particular belief."""

    action: ExploitAction
    expected_value: float


@dataclass(frozen=True)
class ProbeEvaluation:
    """Exact pre-action diagnostics for a candidate probe."""

    observation_entropy_bits: float
    expected_information_gain_bits: float
    expected_value_of_sample_information: float
    net_value_of_information: float
    physical_state_entropy_reduction_bits: float
    is_admissible_epistemic_action: bool


@dataclass(frozen=True)
class DiagnosticDecisionProblem:
    """A two-hypothesis diagnostic problem with an analytic Bayes solution."""

    prior_right: float = 0.5
    correct_payoff: float = 1.0
    incorrect_payoff: float = 0.0
    known_safe_payoff: float = 0.65

    def __post_init__(self) -> None:
        _require_probability(self.prior_right, "prior_right")
        _require_finite(self.correct_payoff, "correct_payoff")
        _require_finite(self.incorrect_payoff, "incorrect_payoff")
        _require_finite(self.known_safe_payoff, "known_safe_payoff")
        if self.correct_payoff < self.incorrect_payoff:
            raise ValueError("correct_payoff must not be lower than incorrect_payoff")

    def outcomes(self, probe: Probe) -> tuple[ProbeOutcome, ...]:
        """Return the complete outcome support for ``probe``."""

        if probe.kind is ProbeKind.DESTRUCTIVE_CERTAINTY:
            return (ProbeOutcome.DESTROYED,)
        return (ProbeOutcome.ZERO, ProbeOutcome.ONE)

    def likelihood(
        self,
        probe: Probe,
        outcome: ProbeOutcome,
        hypothesis: Hypothesis,
    ) -> float:
        """Return ``P(outcome | hypothesis, probe)``."""

        if outcome not in self.outcomes(probe):
            raise ValueError(f"{outcome.value!r} is not an outcome of {probe.kind.value!r}")

        if probe.kind is ProbeKind.DESTRUCTIVE_CERTAINTY:
            return 1.0
        if probe.kind in (ProbeKind.IRRELEVANT_BIT, ProbeKind.NOISY_SIGNAL):
            return 0.5

        reports_right = outcome is ProbeOutcome.ONE
        hypothesis_is_right = hypothesis is Hypothesis.RIGHT
        return probe.accuracy if reports_right == hypothesis_is_right else 1.0 - probe.accuracy

    def outcome_probability(
        self,
        probe: Probe,
        outcome: ProbeOutcome,
        *,
        prior_right: float | None = None,
    ) -> float:
        """Return the marginal probability of an observation."""

        belief = self._belief(prior_right)
        right_term = belief * self.likelihood(probe, outcome, Hypothesis.RIGHT)
        left_term = (1.0 - belief) * self.likelihood(probe, outcome, Hypothesis.LEFT)
        return right_term + left_term

    def posterior_right(
        self,
        probe: Probe,
        outcome: ProbeOutcome,
        *,
        prior_right: float | None = None,
    ) -> float:
        """Return ``P(original hypothesis = RIGHT | outcome, probe)``."""

        belief = self._belief(prior_right)
        evidence_probability = self.outcome_probability(probe, outcome, prior_right=belief)
        if evidence_probability == 0.0:
            raise ValueError("cannot condition on an impossible observation")
        right_joint = belief * self.likelihood(probe, outcome, Hypothesis.RIGHT)
        return right_joint / evidence_probability

    def posterior_from_evidence(
        self,
        evidence: Evidence,
        *,
        as_of: int,
        prior_right: float | None = None,
    ) -> float:
        """Condition on evidence only after it has actually been observed."""

        if as_of < 0:
            raise ValueError("as_of must be non-negative")
        if evidence.observed_at > as_of:
            raise FutureEvidenceError(f"evidence observed at t={evidence.observed_at} is unavailable at t={as_of}")
        return self.posterior_right(evidence.probe, evidence.outcome, prior_right=prior_right)

    def observation_entropy_bits(
        self,
        probe: Probe,
        *,
        prior_right: float | None = None,
    ) -> float:
        """Return entropy of the observation itself, not its relevance."""

        belief = self._belief(prior_right)
        probabilities = (
            self.outcome_probability(probe, outcome, prior_right=belief) for outcome in self.outcomes(probe)
        )
        return _entropy(probabilities)

    def expected_information_gain_bits(
        self,
        probe: Probe,
        *,
        prior_right: float | None = None,
    ) -> float:
        """Return expected reduction in uncertainty about the original hypothesis."""

        belief = self._belief(prior_right)
        expected_posterior_entropy = 0.0
        for outcome in self.outcomes(probe):
            probability = self.outcome_probability(probe, outcome, prior_right=belief)
            if probability == 0.0:
                continue
            posterior = self.posterior_right(probe, outcome, prior_right=belief)
            expected_posterior_entropy += probability * binary_entropy(posterior)
        gain = binary_entropy(belief) - expected_posterior_entropy
        return _clamp_numerical_zero(gain)

    def expected_exploit_value(
        self,
        action: ExploitAction,
        *,
        prior_right: float | None = None,
    ) -> float:
        """Return the expected terminal payoff of one exploit action."""

        belief = self._belief(prior_right)
        if action is ExploitAction.KNOWN_SAFE:
            return self.known_safe_payoff
        if action is ExploitAction.GUESS_RIGHT:
            return belief * self.correct_payoff + (1.0 - belief) * self.incorrect_payoff
        return (1.0 - belief) * self.correct_payoff + belief * self.incorrect_payoff

    def best_exploit(self, *, prior_right: float | None = None) -> ExploitChoice:
        """Return the Bayes-optimal terminal decision."""

        belief = self._belief(prior_right)
        actions = (
            ExploitAction.KNOWN_SAFE,
            ExploitAction.GUESS_LEFT,
            ExploitAction.GUESS_RIGHT,
        )
        action = max(actions, key=lambda candidate: self.expected_exploit_value(candidate, prior_right=belief))
        return ExploitChoice(action, self.expected_exploit_value(action, prior_right=belief))

    def expected_value_of_sample_information(
        self,
        probe: Probe,
        *,
        prior_right: float | None = None,
    ) -> float:
        """Return gross EVSI: improvement from an optional post-sample decision."""

        belief = self._belief(prior_right)
        value_after_sampling = 0.0
        for outcome in self.outcomes(probe):
            probability = self.outcome_probability(probe, outcome, prior_right=belief)
            if probability == 0.0:
                continue
            posterior = self.posterior_right(probe, outcome, prior_right=belief)
            value_after_sampling += probability * self.best_exploit(prior_right=posterior).expected_value
        baseline = self.best_exploit(prior_right=belief).expected_value
        return _clamp_numerical_zero(value_after_sampling - baseline)

    def evaluate_probe(
        self,
        probe: Probe,
        *,
        prior_right: float | None = None,
    ) -> ProbeEvaluation:
        """Return all benchmark metrics for one candidate probe."""

        belief = self._belief(prior_right)
        gross_voi = self.expected_value_of_sample_information(probe, prior_right=belief)
        physical_reduction = binary_entropy(belief) if probe.kind is ProbeKind.DESTRUCTIVE_CERTAINTY else 0.0
        return ProbeEvaluation(
            observation_entropy_bits=self.observation_entropy_bits(probe, prior_right=belief),
            expected_information_gain_bits=self.expected_information_gain_bits(probe, prior_right=belief),
            expected_value_of_sample_information=gross_voi,
            net_value_of_information=gross_voi - probe.cost,
            physical_state_entropy_reduction_bits=physical_reduction,
            is_admissible_epistemic_action=probe.is_admissible_epistemic_action,
        )

    def select_probe(
        self,
        candidates: Iterable[Probe],
        *,
        prior_right: float | None = None,
    ) -> Probe | None:
        """Choose the admissible probe with strictly positive net VOI, if any."""

        belief = self._belief(prior_right)
        selected: Probe | None = None
        selected_net_value = 0.0
        for probe in candidates:
            evaluation = self.evaluate_probe(probe, prior_right=belief)
            if not evaluation.is_admissible_epistemic_action:
                continue
            if evaluation.net_value_of_information > selected_net_value + _NUMERICAL_ZERO:
                selected = probe
                selected_net_value = evaluation.net_value_of_information
        return selected

    def _belief(self, prior_right: float | None) -> float:
        belief = self.prior_right if prior_right is None else prior_right
        _require_probability(belief, "prior_right")
        return belief


def binary_entropy(probability: float) -> float:
    """Return binary Shannon entropy in bits."""

    _require_probability(probability, "probability")
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * log2(probability) - (1.0 - probability) * log2(1.0 - probability)


def _entropy(probabilities: Iterable[float]) -> float:
    entropy = 0.0
    for probability in probabilities:
        _require_probability(probability, "outcome probability")
        if probability > 0.0:
            entropy -= probability * log2(probability)
    return entropy


def _require_probability(value: float, name: str) -> None:
    _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _clamp_numerical_zero(value: float) -> float:
    return 0.0 if abs(value) <= _NUMERICAL_ZERO else value
