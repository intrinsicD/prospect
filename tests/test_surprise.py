"""Unit tests for the decomposed VoE seam (P0-002, ADR-0002).

The surprise signal is a `Surprise` (total NLL + epistemic/aleatoric attribution),
never a bare float, and a `Transition` can name the skill it was collected under.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from prospect import interfaces
from prospect.types import Action, LatentState, Option, Surprise, Transition
from prospect.voe import SurpriseCompetenceMonitor


def test_surprise_is_decomposed_and_frozen() -> None:
    s = Surprise(total=3.0, epistemic=2.0, aleatoric=1.0)
    assert s.total == 3.0
    assert s.epistemic == 2.0
    assert s.aleatoric == 1.0
    with pytest.raises(FrozenInstanceError):
        s.total = 0.0  # type: ignore[misc]


def _transition(**kwargs: object) -> Transition:
    return Transition(
        state=LatentState(z=[0.0]),
        action=Action(data=[0.0]),
        next_state=LatentState(z=[1.0]),
        reward=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_transition_option_defaults_to_none() -> None:
    assert _transition().option is None


def test_transition_attributes_its_skill() -> None:
    opt = Option(name="reach")
    assert _transition(option=opt).option is opt


def test_monitor_still_satisfies_protocol() -> None:
    assert isinstance(SurpriseCompetenceMonitor(), interfaces.CompetenceMonitor)
