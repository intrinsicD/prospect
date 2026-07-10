"""Memory tiers (R7, R8). Episodic replay + *generative* replay (rehearsal), a
semantic store, and an uncertainty-gated, provenance-respecting router over the
tiers. See ADR-0004. ReplayBuffer implemented in P3-003; SemanticStore +
UncertaintyMemoryRouter in P8-001; trust-ordered routing in P8-002.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .interfaces import KnowledgeSource, WorldModel
from .types import (
    Action,
    KnowledgeItem,
    LatentState,
    MemberRollout,
    Option,
    Prediction,
    Transition,
    Trust,
)


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

    `trust` is the store's source-level provenance floor (P8-002): `HIGH` by default
    (the internal distilled store), but a test/adversarial store can be built at a
    lower level so the router treats it as untrusted.

    Contract: interfaces.SemanticMemory.
    """

    name = "semantic"

    def __init__(self, trust: Trust = Trust.HIGH) -> None:
        self.trust = trust
        self._items: list[KnowledgeItem] = []
        self._keys: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # cached key stack (read-heavy query path)

    def __len__(self) -> int:
        return len(self._items)

    def write(self, item: KnowledgeItem) -> None:
        key, _ = item.content
        self._items.append(item)
        self._keys.append(np.asarray(key, dtype=float))
        self._matrix = None  # invalidate; rebuilt lazily on the next query

    def _key_matrix(self) -> np.ndarray:
        # Stack once and reuse: a store is written once then queried many times (in
        # planning, hundreds of thousands of times), so re-stacking per query is the
        # difference between a fast gate and a hung one.
        if self._matrix is None:
            self._matrix = np.stack(self._keys)
        return self._matrix

    def query(self, query: object) -> list[KnowledgeItem]:
        if not self._items:
            return []
        q = np.asarray(query, dtype=float)
        nearest = int(np.argmin(np.sum((self._key_matrix() - q) ** 2, axis=1)))
        return [self._items[nearest]]


class UncertaintyMemoryRouter:
    """Route a query to a tier by epistemic uncertainty *and* provenance (P8-001,
    P8-002): answer from the model when confident — `route()` returns `None`, the
    parametric tier (P0-008) — and retrieve when uncertain (retrieval-as-action,
    ADR-0004). The epistemic `threshold` is calibrated to the model's scale by the
    caller.

    Selection is **trust-ordered** (P8-002). A source is eligible only if its declared
    `trust` clears `min_trust`; among eligible sources the highest-trust one wins. If
    nothing clears the floor, `route()` returns `None` — an untrusted source is data,
    never instruction (ADR-0004): it must never override the agent's own prediction,
    so the router falls back to the parametric tier. `min_trust=LOW` (the default)
    excludes only `UNTRUSTED`.

    Contract: interfaces.MemoryRouter.
    """

    def __init__(
        self,
        sources: Sequence[KnowledgeSource] = (),
        threshold: float = 0.0,
        min_trust: Trust = Trust.LOW,
    ) -> None:
        self._sources = list(sources)
        self.threshold = threshold
        self.min_trust = min_trust

    def route(self, query: object, epistemic: float) -> KnowledgeSource | None:
        if epistemic <= self.threshold:
            return None  # confident: answer parametrically
        eligible = [s for s in self._sources if s.trust >= self.min_trust]
        if not eligible:
            return None  # nothing trusted enough to override the model's own prediction
        return max(eligible, key=lambda s: s.trust)  # trust-ordered; ties keep first


class RetrievalAugmentedWorldModel:
    """A `WorldModel` that patches its predictions with retrieved facts where it is
    epistemically uncertain (P9-001) — retrieval-as-action (ADR-0004) applied to the
    prediction the *planner* consumes, so retrieval extends the agent's competence
    into regions its parametric model has not learned.

    Wraps a base model + an `UncertaintyMemoryRouter`. For each (state, action): if the
    base prediction's epistemic clears the router's gate, retrieve the nearest fact —
    a next-latent in the base model's own space, keyed by `concat(latent, action)` — and
    substitute it; otherwise return the base prediction unchanged. The planner (which
    duck-types `predict_batch`) plans over this transparently.

    **Distance-gated substitution (P9-007).** A retrieved fact is trusted only when it is
    *close* to the query. Inside a planner's imagination rollout the queries are imagined
    latents that, at depth, wander far from any real stored transition; substituting the
    nearest (far, misaligned) fact there corrupts multi-step control — the P9-002 finding.
    So when `reliability_radius` is set, a would-be retrieval whose nearest-fact key
    distance exceeds the radius is *skipped* (the model's own prediction stands), and an
    accepted retrieval carries **honest epistemic scaled by distance** (`epi ×
    min(1, dist/radius)`: an exact hit is trusted, a boundary hit keeps the model's
    uncertainty) instead of the certainty (`epi = 0`) that let CEM exploit the retrieval
    seam. Reliability = closeness (the P9-006 distance-as-reliability insight, now gating
    *whether* to retrieve). The radius is calibrated to the store's coverage by the caller
    (as the router's epistemic `threshold` is). `reliability_radius=None` keeps the legacy
    substitute-and-zero behaviour (the 1-step P8 role, where queries are real states).

    Contract: interfaces.WorldModel. `retrievals`/`calls` count the gated retrievals
    (the run-level evidence that retrieval fired where the model was uncertain)."""

    def __init__(
        self, base: WorldModel, router: UncertaintyMemoryRouter,
        reliability_radius: float | None = None,
    ) -> None:
        self._base = base
        self._router = router
        self._reliability_radius = reliability_radius
        self.retrievals = 0
        self.calls = 0

    def predict(self, state: LatentState, action: Action) -> Prediction:
        mean, var, epi, ale, rew = self._rows(
            np.asarray(state.z, dtype=float).reshape(1, -1),
            np.asarray(action.data, dtype=float).reshape(1, -1),
            initial_ood=state.ood,
        )
        return Prediction(mean=mean[0], var=var[0], epistemic=float(epi[0]),
                          aleatoric=float(ale[0]), reward=float(rew[0]))

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        if getattr(self._base, "predict_member_batch", None) is not None:
            trajectory_predictions: list[Prediction] = []
            member_states = np.asarray(state.z, dtype=float).reshape(1, -1)
            accumulated_variance: np.ndarray | None = None
            for step, action in enumerate(actions):
                rollout = self.predict_member_batch(
                    member_states,
                    np.asarray(action.data, dtype=float).reshape(1, -1),
                    initial_ood=state.ood if step == 0 else None,
                )
                member_states = np.asarray(rollout.states, dtype=float)
                step_variance = np.asarray(rollout.variances, dtype=float)
                if accumulated_variance is None:
                    accumulated_variance = np.zeros_like(step_variance)
                accumulated_variance += step_variance
                epistemic_variance = member_states.var(axis=0)[0]
                aleatoric_variance = accumulated_variance.mean(axis=0)[0]
                trajectory_predictions.append(
                    Prediction(
                        mean=member_states.mean(axis=0)[0],
                        var=aleatoric_variance + epistemic_variance,
                        epistemic=float(np.asarray(rollout.epistemic, dtype=float)[0]),
                        aleatoric=float(aleatoric_variance.mean()),
                        reward=float(np.asarray(rollout.rewards, dtype=float).mean(axis=0)[0]),
                    )
                )
            return trajectory_predictions

        # A narrow protocol-only base has no member states to propagate.
        predictions: list[Prediction] = []
        current = state
        for action in actions:
            prediction = self.predict(current, action)
            predictions.append(prediction)
            current = LatentState(z=prediction.mean)
        return predictions

    def predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._rows(np.asarray(latents, dtype=float), np.asarray(actions, dtype=float))

    def predict_member_batch(
        self,
        member_latents: np.ndarray,
        actions: np.ndarray,
        initial_ood: float | None = None,
    ) -> MemberRollout:
        """Propagate TS∞ particles without bypassing retrieval (U-001/P9-007).

        When the base offers member trajectories, each candidate makes one lookup
        at its mean member state.  A distance-gated fact is accepted only when it
        covers *every* member query, then corrects all particles; this preserves
        TS∞ honesty without multiplying the exact-NN hot path by the ensemble size.
        The returned candidate epistemic remains model-owned. A protocol-only base
        falls back to one mean particle while preserving ``_rows``' effective signal.
        """
        states = np.asarray(member_latents, dtype=float)
        act = np.asarray(actions, dtype=float)
        base_member_batch = getattr(self._base, "predict_member_batch", None)
        if base_member_batch is None:
            if states.ndim == 3:
                if states.shape[0] != 1:
                    raise ValueError("a protocol-only base can propagate only one mean trajectory")
                states = states[0]
            mean, var, epi, _, rew = self._rows(states, act, initial_ood=initial_ood)
            return MemberRollout(
                states=mean[None, :, :],
                variances=var[None, :, :],
                rewards=rew[None, :],
                epistemic=epi,
            )

        base_rollout: MemberRollout = base_member_batch(
            states, act, initial_ood=initial_ood
        )
        next_states = np.asarray(base_rollout.states, dtype=float).copy()
        variances = np.asarray(base_rollout.variances, dtype=float)
        rewards = np.asarray(base_rollout.rewards, dtype=float)
        epistemic = np.asarray(base_rollout.epistemic, dtype=float).copy()
        member_count, candidates, _ = next_states.shape
        if states.ndim == 2:
            query_states = np.repeat(states[None, :, :], member_count, axis=0)
        elif states.ndim == 3 and states.shape[:2] == (member_count, candidates):
            query_states = states
        else:
            raise ValueError("member_latents shape does not match the base member rollout")

        self.calls += candidates
        source = self._router.route(None, float(np.max(epistemic)) if candidates else 0.0)
        if source is None:
            return MemberRollout(next_states, variances, rewards, epistemic)

        for candidate in np.nonzero(epistemic > self._router.threshold)[0]:
            base_epistemic = float(epistemic[candidate])
            mean_key = np.concatenate([query_states[:, candidate].mean(axis=0), act[candidate]])
            items = source.query(mean_key)
            if not items:
                continue
            fact_key, answer = items[0].content
            residual_epistemic = 0.0
            if self._reliability_radius is not None:
                member_keys = np.concatenate(
                    [
                        query_states[:, candidate],
                        np.repeat(act[candidate][None, :], member_count, axis=0),
                    ],
                    axis=1,
                )
                distances = np.sum(
                    (member_keys - np.asarray(fact_key, dtype=float)) ** 2, axis=1
                )
                max_distance = float(np.max(distances))
                if max_distance > self._reliability_radius:
                    continue
                residual_epistemic = base_epistemic * min(
                    1.0, max_distance / self._reliability_radius
                )
            next_states[:, candidate] = np.asarray(answer, dtype=float)
            self.retrievals += 1
            epistemic[candidate] = residual_epistemic
        return MemberRollout(next_states, variances, rewards, epistemic)

    def _rows(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        initial_ood: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = getattr(self._base, "predict_batch", None)
        if batch is not None and initial_ood is None:
            mean, var, epi, ale, rew = batch(latents, actions)
            mean, epi = np.array(mean, dtype=float), np.array(epi, dtype=float)
        else:  # protocol-only or OOD root: preserve the full LatentState signal
            preds = [self._base.predict(LatentState(z=z, ood=initial_ood), Action(data=a))
                     for z, a in zip(latents, actions, strict=True)]
            mean = np.stack([np.asarray(p.mean, dtype=float) for p in preds])
            var = np.stack([np.asarray(p.var, dtype=float) for p in preds])
            epi = np.array([p.epistemic for p in preds], dtype=float)
            ale = np.array([p.aleatoric for p in preds], dtype=float)
            rew = np.array([p.reward for p in preds], dtype=float)
        self.calls += len(latents)
        # Which rows the router would retrieve for: gating is monotone in epistemic, so
        # one route() probe fixes the selected source and the threshold — then the mask
        # is vectorized and only the uncertain rows touch the store (the hot path).
        source = self._router.route(None, float(np.max(epi)) if len(epi) else 0.0)
        if source is None:
            return mean, var, epi, ale, rew  # nothing uncertain enough / nothing trusted
        for i in np.nonzero(epi > self._router.threshold)[0]:
            key = np.concatenate([latents[i], actions[i]])
            items = source.query(key)
            if not items:
                continue
            fact_key, answer = items[0].content
            if self._reliability_radius is not None:  # distance-gated (P9-007)
                dist = float(np.sum((key - np.asarray(fact_key, dtype=float)) ** 2))
                if dist > self._reliability_radius:
                    continue  # far fact = fiction at rollout depth; keep the model
                epi[i] *= min(1.0, dist / self._reliability_radius)  # honest: reliability=closeness
            else:
                epi[i] = 0.0  # legacy 1-step role: the fact stands in as a confident guess
            mean[i] = np.asarray(answer, dtype=float)
            self.retrievals += 1
        return mean, var, epi, ale, rew
