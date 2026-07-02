"""Universal codec (R6): any input -> shared latent, latent -> any output.
Also the entry point for knowledge-as-tokens (ADR-0004). See ADR-0001. Task: P6-001.
"""
from __future__ import annotations

from .types import LatentState, Observation


class UniversalCodec:
    """Perceiver-IO-style encoder/decoder skeleton. Introduced LAST (P6): swap the
    single-modality codec for this once the core loop is proven.

    Contract: interfaces.Codec.
    """

    def encode(self, obs: Observation) -> LatentState:
        raise NotImplementedError("P6-001")

    def decode(self, state: LatentState, query: object) -> object:
        raise NotImplementedError("P6-001")
