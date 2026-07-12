"""Memory tiers (R7, R8). Episodic replay + *generative* replay (rehearsal), a
semantic store, and an uncertainty-gated, provenance-respecting router over the
tiers. See ADR-0004. ReplayBuffer implemented in P3-003; SemanticStore +
UncertaintyMemoryRouter in P8-001; trust-ordered routing in P8-002.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

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

RETRIEVAL_NEIGHBORS = 3


def _nearest_indices(distances: np.ndarray, k: int = RETRIEVAL_NEIGHBORS) -> np.ndarray:
    """Return up to ``k`` nearest indices, sorting only the small selected set."""
    count = min(k, len(distances))
    if count == 0:
        return np.empty(0, dtype=int)
    if count == len(distances):
        selected = np.arange(count)
    else:
        selected = np.argpartition(distances, count - 1)[:count]
    return selected[np.argsort(distances[selected], kind="stable")]


def blend_retrieved_items(
    query: np.ndarray,
    fallback: np.ndarray,
    items: Sequence[KnowledgeItem],
    temperature: float,
    reliability_radius: float | None = None,
    *,
    answer_transform: Callable[[object], np.ndarray] | None = None,
    coverage_queries: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Distance-kernel aggregate ``items`` and blend it with ``fallback`` (U-005).

    The returned tuple is ``(prediction, reliability, weights)``. Kernel weights are
    a stable ``softmax(-squared_distance / temperature)``. With a reliability radius,
    only facts covering every supplied query are eligible and reliability decreases
    linearly from one at an exact hit to zero at the boundary. Without a hard radius,
    the kernel scale supplies a soft reliability ``exp(-nearest / temperature)``.
    """
    base = np.asarray(fallback, dtype=float)
    if not items:
        return base.copy(), 0.0, np.empty(0, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("retrieval kernel temperature must be finite and positive")
    if reliability_radius is not None and reliability_radius <= 0.0:
        raise ValueError("retrieval reliability radius must be positive")

    q = np.asarray(query, dtype=float)
    keys = np.stack([np.asarray(item.content[0], dtype=float) for item in items])
    kernel_distances = np.sum((keys - q) ** 2, axis=1)
    eligible = np.ones(len(items), dtype=bool)
    coverage_distances = kernel_distances
    if reliability_radius is not None:
        coverage = q.reshape(1, -1) if coverage_queries is None else np.asarray(
            coverage_queries, dtype=float
        ).reshape(-1, q.size)
        coverage_distances = np.max(
            np.sum((coverage[:, None, :] - keys[None, :, :]) ** 2, axis=2), axis=0
        )
        eligible = coverage_distances <= reliability_radius
    if not np.any(eligible):
        return base.copy(), 0.0, np.empty(0, dtype=float)

    selected_items = [item for item, keep in zip(items, eligible, strict=True) if keep]
    selected_distances = kernel_distances[eligible]
    logits = -(selected_distances - float(np.min(selected_distances))) / temperature
    weights = np.exp(logits)
    weights /= float(np.sum(weights))
    transform = answer_transform or (lambda answer: np.asarray(answer, dtype=float))
    answers = np.stack(
        [np.asarray(transform(item.content[1]), dtype=float) for item in selected_items]
    )
    retrieved = np.tensordot(weights, answers, axes=(0, 0))

    nearest = float(np.min(coverage_distances[eligible]))
    if reliability_radius is None:
        reliability = float(np.exp(-nearest / temperature))
    else:
        reliability = max(0.0, 1.0 - nearest / reliability_radius)
    blended = (1.0 - reliability) * base + reliability * retrieved
    return np.asarray(blended, dtype=float), reliability, weights


class ReplayBuffer:
    """Real experience for learning + generative replay for rehearsal (P3-003).

    Storage: a fixed-budget hybrid of recent REAL transitions in a FIFO ring and
    older REAL transitions in an Algorithm-R reservoir (U-004). Each transition
    lives in one segment: an item aging out of FIFO becomes a reservoir candidate,
    so the reservoir is uniform over the lifetime history older than the recent
    window without double-weighting entries. Raw observations are retained
    (P0-011: experience stays re-encodable under a future codec). Dreamed transitions
    are NEVER stored — dream-of-dreams is structurally impossible, not merely
    checked (ADR-0006 lineage rule).

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
    RESERVOIR_FRACTION = 0.4

    def __init__(
        self,
        world_model: WorldModel | None = None,
        capacity: int = 50_000,
        real_fraction: float = 0.5,
        max_dream_depth: int = 3,
        epistemic_multiplier: float = 4.0,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self._model = world_model
        self.capacity = capacity
        self.reservoir_capacity = (
            max(1, min(capacity - 1, round(capacity * self.RESERVOIR_FRACTION)))
            if capacity > 1
            else 0
        )
        self.fifo_capacity = capacity - self.reservoir_capacity
        self.real_fraction = real_fraction
        self.max_dream_depth = max_dream_depth
        self.epistemic_multiplier = epistemic_multiplier
        self._rng = np.random.default_rng(seed)
        self._reservoir_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self._fifo: list[Transition] = []
        self._reservoir: list[Transition] = []
        self._fifo_cursor = 0  # ring-buffer write position once the recent window is full
        self._reservoir_seen = 0  # lifetime count of FIFO evictees considered by Algorithm R

    def __len__(self) -> int:
        return len(self._fifo) + len(self._reservoir)

    def add(self, transition: Transition) -> None:
        if len(self._fifo) < self.fifo_capacity:
            self._fifo.append(transition)
            return

        evicted = self._fifo[self._fifo_cursor]
        self._fifo[self._fifo_cursor] = transition
        self._fifo_cursor = (self._fifo_cursor + 1) % self.fifo_capacity
        if self.reservoir_capacity == 0:
            return
        self._reservoir_seen += 1
        if len(self._reservoir) < self.reservoir_capacity:
            self._reservoir.append(evicted)
            return
        slot = int(self._reservoir_rng.integers(0, self._reservoir_seen))
        if slot < self.reservoir_capacity:
            self._reservoir[slot] = evicted

    def sample(self, n: int) -> list[Transition]:
        if len(self) == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        fifo_size = len(self._fifo)
        indices = self._rng.integers(0, len(self), size=n)
        return [
            self._fifo[i] if i < fifo_size else self._reservoir[i - fifo_size]
            for i in indices
        ]

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
    returns up to three nearest facts ranked by key distance (or `[]` when empty).
    Every item carries `Provenance` (P0-008).

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
        distances = np.sum((self._key_matrix() - q) ** 2, axis=1)
        return [self._items[i] for i in _nearest_indices(distances)]


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
    base prediction's epistemic clears the router's gate, retrieve up to three nearby
    facts — next-latents keyed by `concat(latent, action)` — aggregate them with a
    distance kernel, and blend that answer with the model prediction (U-005). Otherwise
    return the base prediction unchanged. The planner (which duck-types `predict_batch`)
    plans over this transparently.

    **Distance-gated blending (P9-007/U-005).** A retrieved fact is trusted only when it is
    *close* to the query. Inside a planner's imagination rollout the queries are imagined
    latents that, at depth, wander far from any real stored transition; substituting the
    nearest (far, misaligned) fact there corrupts multi-step control — the P9-002 finding.
    So when `reliability_radius` is set, a would-be retrieval whose nearest-fact key
    distance exceeds the radius is *skipped* (the model's own prediction stands), and an
    accepted retrieval blends toward the aggregated fact by the same closeness and carries
    **honest residual epistemic** (`epi × (1 - reliability)`: an exact hit trusts the
    aggregate, a boundary hit keeps the model and its uncertainty). Reliability =
    closeness (the P9-006 insight). The radius and distance-kernel temperature are
    calibrated to store coverage by the caller. Without a hard radius, kernel distance
    supplies a soft reliability rather than reverting to hard substitution.

    Contract: interfaces.WorldModel. Instrumentation separates `calls` (candidate
    predictions scored), `gate_hits` (epistemic threshold exceedances), and `retrievals`
    (distance-covered aggregates actually blended)."""

    def __init__(
        self, base: WorldModel, router: UncertaintyMemoryRouter,
        reliability_radius: float | None = None,
        kernel_temperature: float = 1.0,
    ) -> None:
        if not np.isfinite(kernel_temperature) or kernel_temperature <= 0.0:
            raise ValueError("retrieval kernel temperature must be finite and positive")
        self._base = base
        self._router = router
        self._reliability_radius = reliability_radius
        self._kernel_temperature = kernel_temperature
        self.retrievals = 0
        self.calls = 0
        self.gate_hits = 0

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
        gate_mask = epistemic > self._router.threshold
        self.gate_hits += int(np.sum(gate_mask))
        source = self._router.route(None, float(np.max(epistemic)) if candidates else 0.0)
        if source is None:
            return MemberRollout(next_states, variances, rewards, epistemic)

        for candidate in np.nonzero(gate_mask)[0]:
            mean_key = np.concatenate([query_states[:, candidate].mean(axis=0), act[candidate]])
            items = source.query(mean_key)
            if not items:
                continue
            member_keys = np.concatenate(
                [
                    query_states[:, candidate],
                    np.repeat(act[candidate][None, :], member_count, axis=0),
                ],
                axis=1,
            )
            blended, reliability, _ = blend_retrieved_items(
                mean_key,
                next_states[:, candidate],
                items,
                self._kernel_temperature,
                self._reliability_radius,
                coverage_queries=member_keys,
            )
            if reliability <= 0.0:
                continue
            next_states[:, candidate] = blended
            self.retrievals += 1
            epistemic[candidate] *= 1.0 - reliability
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
        gate_mask = epi > self._router.threshold
        self.gate_hits += int(np.sum(gate_mask))
        # Which rows the router would retrieve for: gating is monotone in epistemic, so
        # one route() probe fixes the selected source and the threshold — then the mask
        # is vectorized and only the uncertain rows touch the store (the hot path).
        source = self._router.route(None, float(np.max(epi)) if len(epi) else 0.0)
        if source is None:
            return mean, var, epi, ale, rew  # nothing uncertain enough / nothing trusted
        for i in np.nonzero(gate_mask)[0]:
            key = np.concatenate([latents[i], actions[i]])
            items = source.query(key)
            if not items:
                continue
            blended, reliability, _ = blend_retrieved_items(
                key,
                mean[i],
                items,
                self._kernel_temperature,
                self._reliability_radius,
            )
            if reliability <= 0.0:
                continue
            mean[i] = blended
            epi[i] *= 1.0 - reliability
            self.retrievals += 1
        return mean, var, epi, ale, rew
