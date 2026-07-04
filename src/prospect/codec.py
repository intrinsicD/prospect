"""Universal codec (R6): any input -> shared latent, latent -> any output.
Also the entry point for knowledge-as-tokens (ADR-0004). See ADR-0001. Task: P6-001.

The P6 swap is a REPRESENTATION change, not a mere interface swap (P0-011): the
dynamics model, option model, competence statistics and stored replay latents all
couple to the incumbent latent's *distribution* (the latent is a contract,
ADR-0001). So the codec is DISTILLED into that incumbent latent space — its
`encode` is trained to reproduce the frozen P1 encoder on the shared modality —
before it replaces it, and downstream stays valid within tolerance. Any-to-any is
then real: a second modality carrying the same situation distils to the SAME
latent, so the frozen core loop reasons over it identically.
"""
from __future__ import annotations

import numpy as np

from .types import LatentState, Modality, Observation
from .world_model import _MLP  # core-internal reuse (the small numpy MLP + Adam)


class UniversalCodec:
    """Perceiver-IO-style multi-modality codec (P6-001), numpy backend.

    Encode: each modality has an input adapter (`data -> token`); a shared trunk
    (`token -> latent`) is the minimal Perceiver bottleneck — it lands ANY
    modality in the one fixed latent (true multi-head cross-attention is earned by
    a later gate if a modality needs it). Decode: a per-modality readout head
    (`latent -> data`) produces any output (R6), trained on the FROZEN latent so it
    never pressures the dynamics latent (reconstruction stays off the dynamics
    path, ADR-0006).

    Distillation is the migration (P0-011): `distill_encode` fits adapters+trunk to
    match given target latents (the incumbent encoder's outputs); `fit_decode` fits
    a readout to reconstruct a modality from the latent. Codec training is
    supervised latent-matching, not transition replay — so it uses these dedicated
    methods rather than force-fitting the `Learner` seam.

    Contract: interfaces.Codec.
    """

    def __init__(
        self,
        modality_dims: dict[Modality, int] | None = None,
        latent_dim: int = 8,
        token_dim: int = 32,
        hidden: int = 64,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        self._modality_dims = dict(modality_dims or {Modality.STATE: 3})
        self.latent_dim = latent_dim
        self._adapters: dict[Modality, _MLP] = {}
        self._decoders: dict[Modality, _MLP] = {}
        self._stats: dict[Modality, tuple[np.ndarray, np.ndarray]] = {}
        for i, (modality, dim) in enumerate(self._modality_dims.items()):
            self._adapters[modality] = _MLP(
                [dim, hidden, token_dim], np.random.default_rng(seed * 131 + 2 * i + 1), lr)
            self._decoders[modality] = _MLP(
                [latent_dim, hidden, dim], np.random.default_rng(seed * 577 + 2 * i + 3), lr)
        self._trunk = _MLP([token_dim, hidden, latent_dim], np.random.default_rng(seed + 7), lr)

    # ------------------------------------------------------------- modality I/O
    def _adapter(self, modality: Modality) -> _MLP:
        adapter = self._adapters.get(modality)
        if adapter is None:
            known = ", ".join(str(m) for m in self._modality_dims)
            raise KeyError(f"no adapter for modality {modality!r}; registered: {known}")
        return adapter

    def _standardize(self, modality: Modality, x: np.ndarray) -> np.ndarray:
        stats = self._stats.get(modality)
        if stats is None:  # not yet distilled for this modality — pass through
            return x
        mean, std = stats
        return np.asarray((x - mean) / std, dtype=float)

    def encode(self, obs: Observation) -> LatentState:
        """Any modality -> the shared latent (routes by `obs.modality`)."""
        adapter = self._adapter(obs.modality)
        x = self._standardize(obs.modality, np.asarray(obs.data, dtype=float).reshape(1, -1))
        token, _ = adapter.forward(x)
        latent, _ = self._trunk.forward(token)
        return LatentState(z=latent[0])

    def decode(self, state: LatentState, query: object) -> Observation:
        """Latent -> the queried modality's data (R6's 'produce any output')."""
        modality = query if isinstance(query, Modality) else Modality(str(query))
        decoder = self._decoders.get(modality)
        if decoder is None:
            known = ", ".join(str(m) for m in self._modality_dims)
            raise KeyError(f"no decoder for modality {modality!r}; registered: {known}")
        out, _ = decoder.forward(np.asarray(state.z, dtype=float).reshape(1, -1))
        return Observation(modality=modality, data=out[0])

    # ---------------------------------------------------------- distillation
    def distill_encode(
        self, data: np.ndarray, modality: Modality, target_latents: np.ndarray
    ) -> float:
        """One gradient step fitting `encode(modality)` to `target_latents` — the
        migration into the incumbent latent (P0-011). Freezes input standardization
        from the first batch seen for this modality."""
        adapter = self._adapter(modality)
        if modality not in self._stats:
            self._stats[modality] = (data.mean(axis=0), data.std(axis=0) + 1e-6)
        x = self._standardize(modality, data)
        adapter.zero_grad()
        self._trunk.zero_grad()
        token, cache_a = adapter.forward(x)
        latent, cache_t = self._trunk.forward(token)
        diff = latent - target_latents
        loss = float(np.mean(np.sum(diff**2, axis=1)))
        d_latent = 2.0 * diff / data.shape[0]
        d_token = self._trunk.backward(d_latent, cache_t)
        adapter.backward(d_token, cache_a)
        self._trunk.step()
        adapter.step()
        return loss

    def fit_decode(self, latents: np.ndarray, data: np.ndarray, modality: Modality) -> float:
        """One gradient step fitting the modality's readout to reconstruct `data`
        from the FROZEN latent (standardized target; never touches the dynamics)."""
        decoder = self._decoders.get(modality)
        if decoder is None:
            raise KeyError(f"no decoder for modality {modality!r}")
        target = self._standardize(modality, data)
        decoder.zero_grad()
        out, cache = decoder.forward(latents)
        diff = out - target
        loss = float(np.mean(np.sum(diff**2, axis=1)))
        decoder.backward(2.0 * diff / latents.shape[0], cache)
        decoder.step()
        return loss
