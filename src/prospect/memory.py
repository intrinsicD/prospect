"""Memory tiers (R7, R8). Episodic replay + *generative* replay (rehearsal), a
semantic store, and an uncertainty-gated router over the tiers. See ADR-0004.
ReplayBuffer implemented in P3-003; SemanticStore + UncertaintyMemoryRouter in P8-001.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .interfaces import KnowledgeSource, WorldModel
from .types import KnowledgeItem, LatentState, Option, Transition


class ReplayBuffer:
    """Real experience for learning + generative replay for rehearsal (P3-003).

    Storage: a ring buffer of REAL transitions carrying raw observations
    (P0-011: experience stays re-encodable under a future codec). Dreamed
    transitions are NEVER stored — dream-of-dreams is structurally impossible,
    not merely checked (ADR-0006 lineage rule).

    `generative_replay(n)` returns a mixed rehearsal batch, anti-collapse by
    construction (ADR-0006):
    - a fixed fraction of every batch is real experience (`real_fraction` is a
      floor — gated-out dreams are replaced by extra real samples);
    - dreams start from real stored states, encoded via the model's `encode`
      (duck-typed; identity fallback for latent-native models), and roll
      `model.predict` forward at most `max_dream_depth` steps with actions
      bootstrapped from stored real actions;
    - each dreamed step is quality-gated: its epistemic must not exceed
      `epistemic_multiplier` x the median depth-1 epistemic of this call
      (self-calibrating; rehearse only in-distribution dreams).

    Dreamed transitions live in LATENT space and are marked
    `Option("__dream__", metadata={"depth": k})`, carrying the dreaming
    `prediction`; the P7 consolidation trainer consumes them.

    Contract: interfaces.EpisodicMemory.
    """

    DREAM_SKILL = "__dream__"

    def __init__(
        self,
        world_model: WorldModel | None = None,
        capacity: int = 50_000,
        real_fraction: float = 0.5,
        max_dream_depth: int = 3,
        epistemic_multiplier: float = 4.0,
        seed: int = 0,
    ) -> None:
        self._model = world_model
        self.capacity = capacity
        self.real_fraction = real_fraction
        self.max_dream_depth = max_dream_depth
        self.epistemic_multiplier = epistemic_multiplier
        self._rng = np.random.default_rng(seed)
        self._data: list[Transition] = []
        self._cursor = 0  # ring-buffer write position once full

    def __len__(self) -> int:
        return len(self._data)

    def add(self, transition: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:  # FIFO eviction
            self._data[self._cursor] = transition
            self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, n: int) -> list[Transition]:
        if not self._data:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = self._rng.integers(0, len(self._data), size=n)
        return [self._data[i] for i in indices]

    def generative_replay(self, n: int) -> list[Transition]:
        if self._model is None:
            raise ValueError("generative replay needs a world model to dream with")
        n_real = max(1, round(self.real_fraction * n))
        batch = self.sample(n_real)
        dreams = self._dream(n - n_real)
        batch.extend(dreams)
        if len(batch) < n:  # gated-out dreams are replaced by real anchors
            batch.extend(self.sample(n - len(batch)))
        return batch

    def _encode(self, raw: object) -> np.ndarray:
        encode = getattr(self._model, "encode", None)
        if encode is None:  # latent-native model: states are already latents
            return np.asarray(raw, dtype=float)
        return np.asarray(encode(raw).z, dtype=float)

    def _dream(self, n: int) -> list[Transition]:
        """Roll the model forward from real starts; gate on epistemic; cap depth."""
        assert self._model is not None
        if n <= 0:
            return []
        starts = self.sample(n)
        first_actions = [t.action for t in self.sample(n)]  # bootstrapped real actions
        deeper_actions = iter(t.action for t in self.sample(n * self.max_dream_depth))
        first_steps = []
        for start, action in zip(starts, first_actions, strict=True):
            latent = LatentState(z=self._encode(start.state.z))
            first_steps.append((latent, action, self._model.predict(latent, action)))
        # Self-calibrating gate: depth-1 dreams from real states define "in-distribution".
        gate = self.epistemic_multiplier * float(
            np.median([p.epistemic for _, _, p in first_steps])
        )
        dreams: list[Transition] = []
        for latent, action, prediction in first_steps:
            for depth in range(1, self.max_dream_depth + 1):
                if prediction.epistemic > gate:
                    break  # off-distribution dream: do not rehearse it
                dreams.append(
                    Transition(
                        state=latent,
                        action=action,
                        next_state=LatentState(z=prediction.mean),
                        reward=prediction.reward,
                        prediction=prediction,
                        option=Option(name=self.DREAM_SKILL, metadata={"depth": depth}),
                    )
                )
                if depth == self.max_dream_depth:
                    break
                latent = LatentState(z=prediction.mean)
                action = next(deeper_actions)
                prediction = self._model.predict(latent, action)
        return dreams[:n]


class SemanticStore:
    """Distilled facts as a queryable `KnowledgeSource` (P8-001). Its read side is
    a `KnowledgeSource` (one query verb into every tier, P0-008); `write` is the
    consolidation surface.

    A fact's `content` is a `(key, answer)` pair — the query key it answers and the
    answer it holds (knowledge-as-tokens, ADR-0004: the answer is a next-latent in
    the model's own space, a drop-in for the model's prediction). `query(key)`
    returns the single nearest fact by key distance (or `[]` when empty). Every
    item carries `Provenance` (P0-008).

    Contract: interfaces.SemanticMemory.
    """

    name = "semantic"

    def __init__(self) -> None:
        self._items: list[KnowledgeItem] = []
        self._keys: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._items)

    def write(self, item: KnowledgeItem) -> None:
        key, _ = item.content
        self._items.append(item)
        self._keys.append(np.asarray(key, dtype=float))

    def query(self, query: object) -> list[KnowledgeItem]:
        if not self._items:
            return []
        q = np.asarray(query, dtype=float)
        keys = np.stack(self._keys)
        nearest = int(np.argmin(np.sum((keys - q) ** 2, axis=1)))
        return [self._items[nearest]]


class UncertaintyMemoryRouter:
    """Route a query to a tier by current epistemic uncertainty (P8-001): answer
    from the model when confident — `route()` returns `None`, the parametric tier
    (P0-008) — and retrieve when uncertain (retrieval-as-action, ADR-0004). The
    threshold is calibrated to the model's epistemic scale by the caller.

    Minimal tier selection: below-threshold ⇒ parametric (`None`); above ⇒ the
    first source. Trust-ordered selection among external sources is P8-002.

    Contract: interfaces.MemoryRouter.
    """

    def __init__(self, sources: Sequence[KnowledgeSource] = (), threshold: float = 0.0) -> None:
        self._sources = list(sources)
        self.threshold = threshold

    def route(self, query: object, epistemic: float) -> KnowledgeSource | None:
        if epistemic <= self.threshold or not self._sources:
            return None  # confident (or nothing to retrieve from): answer parametrically
        return self._sources[0]
