"""Unit tests for the external knowledge tier (P10-001): nearest-key content lookup and
configurable trust. The content is an `Observation` (to be ingested through the codec),
not a pre-digested latent — the distinction from the internal `SemanticStore`."""
from __future__ import annotations

import numpy as np

from prospect import interfaces
from prospect.knowledge import ExternalKnowledgeSource
from prospect.types import KnowledgeItem, Modality, Observation, Provenance, Trust


def _fact(key: list[float], next_obs: list[float], trust: Trust = Trust.HIGH) -> KnowledgeItem:
    content = Observation(modality=Modality.STATE, data=np.array(next_obs, dtype=float))
    return KnowledgeItem(content=(np.array(key, dtype=float), content),
                         provenance=Provenance(source="unit", trust=trust))


def test_external_source_returns_nearest_content() -> None:  # P10-001
    store = ExternalKnowledgeSource(trust=Trust.HIGH)
    assert store.query(np.array([0.0, 0.0])) == []  # empty store answers nothing
    store.write(_fact([0.0, 0.0], [1.0, 2.0, 3.0]))
    store.write(_fact([5.0, 5.0], [9.0, 9.0, 9.0]))
    near = store.query(np.array([0.1, -0.1]))[0]
    assert isinstance(near.content[1], Observation)  # content is an observation, not a latent
    assert list(near.content[1].data) == [1.0, 2.0, 3.0]  # nearest key [0,0]
    assert list(store.query(np.array([4.9, 5.2]))[0].content[1].data) == [9.0, 9.0, 9.0]
    assert len(store) == 2


def test_external_source_is_a_knowledge_source_with_configurable_trust() -> None:  # P10-001
    assert isinstance(ExternalKnowledgeSource(), interfaces.KnowledgeSource)
    assert ExternalKnowledgeSource().trust == Trust.UNTRUSTED  # external defaults to untrusted
    assert ExternalKnowledgeSource(trust=Trust.HIGH).trust == Trust.HIGH  # a vetted source
