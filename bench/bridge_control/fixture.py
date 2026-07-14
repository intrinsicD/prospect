"""Deterministic BridgeControl fixture and fixed factorial datasets.

The true environment never changes.  The intervention is entirely in the fixed
training evidence: whether the observed state-action pairing contains the directed
bridge, whether actions around that bridge span both local control directions, and
whether each controllable state-action microcell has one or eight distinct samples.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from prospect.types import Action, LatentState, Modality, Observation, Prediction, Transition

SCHEMA_VERSION = "bridge-control-v1"
REGION_CENTERS = np.array([-0.90, -0.65, -0.40, -0.11, 0.11, 0.40, 0.65, 0.90])
BRIDGE_SOURCE_REGION = 3
BRIDGE_TARGET_REGION = 4
POST_STABILIZATION_REGIONS = (5, 6)
DOOR_LANE = 0
DECOY_LANE = 1
ACTION_CORNERS = np.array(
    [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ],
    dtype=float,
)
DOOR_Y_LEVELS = np.array([-0.30, -0.10, 0.10, 0.30])
DECOY_Y_LEVELS = np.array([0.56, 0.62, 0.68, 0.74])
JITTER = np.array([-0.014, -0.010, -0.006, -0.002, 0.002, 0.006, 0.010, 0.014])
NUISANCE_LEVELS = np.array([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875])
REPLICATES = 8
EVAL_STARTS = np.array(
    [
        [-0.90, -0.28, -0.75],
        [-0.90, -0.20, 0.50],
        [-0.90, 0.20, -0.25],
        [-0.90, 0.28, 0.75],
    ],
    dtype=float,
)
GOAL_Y = 0.25


@dataclass(frozen=True)
class FactorCell:
    """One cell of the bridge x local-rank x controllable-density factorial."""

    bridge: bool
    full_rank: bool
    density: int

    def __post_init__(self) -> None:
        if self.density not in (1, 8):
            raise ValueError("density must be 1 or 8")

    @property
    def name(self) -> str:
        return f"b{int(self.bridge)}_r{int(self.full_rank)}_d{self.density}"

    def as_dict(self) -> dict[str, bool | int | str]:
        return {
            "name": self.name,
            "bridge": self.bridge,
            "full_rank": self.full_rank,
            "density": self.density,
        }


@dataclass(frozen=True)
class BridgeDataset:
    """Canonical numeric evidence for one factorial cell or named control."""

    name: str
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    region_ids: np.ndarray
    next_region_ids: np.ndarray
    lane_ids: np.ndarray
    slot_ids: np.ndarray
    replicate_ids: np.ndarray
    cell: FactorCell | None = None
    control: str | None = None

    def __post_init__(self) -> None:
        n = len(self.states)
        lengths = {
            len(self.actions),
            len(self.next_states),
            len(self.rewards),
            len(self.region_ids),
            len(self.next_region_ids),
            len(self.lane_ids),
            len(self.slot_ids),
            len(self.replicate_ids),
        }
        if lengths != {n}:
            raise ValueError(f"dataset arrays have inconsistent lengths: {lengths | {n}}")
        if self.states.shape != (n, 3) or self.next_states.shape != (n, 3):
            raise ValueError("states and next_states must have shape (n, 3)")
        if self.actions.shape != (n, 2):
            raise ValueError("actions must have shape (n, 2)")

    def transitions(self) -> list[Transition]:
        return [
            Transition(
                state=LatentState(z=state.copy()),
                action=Action(data=action.copy()),
                next_state=LatentState(z=next_state.copy()),
                reward=float(reward),
            )
            for state, action, next_state, reward in zip(
                self.states,
                self.actions,
                self.next_states,
                self.rewards,
                strict=True,
            )
        ]


def factor_cells() -> list[FactorCell]:
    return [
        FactorCell(bridge=bridge, full_rank=full_rank, density=density)
        for bridge in (False, True)
        for full_rank in (False, True)
        for density in (1, 8)
    ]


def region_id(x: float) -> int:
    return int(np.argmin(np.abs(REGION_CENTERS - x)))


def transition_dynamics(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
    """One deterministic transition in the fixed causal environment.

    Longitudinal motion is globally simple.  Lateral action semantics reverse and
    couple locally around the bridge, making two independent action directions
    necessary to identify corrections there.  Crossing x=0 succeeds only through
    the narrow lateral opening.
    """

    s = np.asarray(state, dtype=float).reshape(3)
    a = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
    x, y, nuisance = (float(v) for v in s)
    if 0.30 <= x <= 0.75 and abs(y) < 0.50:
        # Sum/difference coordinates in the post-bridge strip make longitudinal
        # progress and lateral goal stabilization independent local directions.
        # A diagonal action design observes only the former; the bridge transition
        # itself remains a separate, ordinary-action manipulation.
        candidate_x = x + 0.16 * (a[0] + a[1])
        candidate_y = y + 0.16 * (a[0] - a[1])
    elif abs(y) >= 0.55:
        # Off-lane decoy states are deliberately unreachable from the evaluation
        # corridor and laterally absorbing.  They provide a nuisance/transfer
        # negative-control region without changing goal-side control evidence.
        candidate_x = x + 0.22 * a[0]
        candidate_y = y
    else:
        candidate_x = x + 0.22 * a[0]
        candidate_y = y + 0.12 * a[1]
    candidate_x = float(np.clip(candidate_x, -0.95, 0.95))
    candidate_y = float(np.clip(candidate_y, -0.75, 0.75))
    bridge_width = 0.15
    if x < 0.0 <= candidate_x and abs(candidate_y) > bridge_width:
        candidate_x = x
    elif x > 0.0 >= candidate_x and abs(candidate_y) > bridge_width:
        candidate_x = x
    # The nuisance is deliberately high-coverage but trivial to predict.  A
    # nuisance-only control can therefore detect representation distraction
    # without making nuisance dynamics the experiment's hidden challenge.
    next_nuisance = nuisance
    next_state = np.array([candidate_x, candidate_y, next_nuisance], dtype=float)
    reward = (
        1.25 * candidate_x
        - 2.0 * abs(candidate_y - GOAL_Y)
        - 0.01 * float(a @ a)
    )
    return next_state, float(reward)


class BridgeControlEnv:
    """Continuous two-action environment with an exact-place harness surface."""

    obs_dim = 3
    action_dim = 2
    action_low = -1.0
    action_high = 1.0

    def __init__(self) -> None:
        self._state = EVAL_STARTS[0].copy()

    def reset(self, seed: int | None = None) -> Observation:
        index = 0 if seed is None else int(seed) % len(EVAL_STARTS)
        self._state = EVAL_STARTS[index].copy()
        return self._obs()

    def set_state(self, state: Sequence[float]) -> Observation:
        values = np.asarray(state, dtype=float).reshape(3)
        self._state = values.copy()
        return self._obs()

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        self._state, reward = transition_dynamics(self._state, np.asarray(action.data, dtype=float))
        return self._obs(), reward, False

    def _obs(self) -> Observation:
        return Observation(modality=Modality.STATE, data=self._state.copy())


class ExactBridgeModel:
    """Harness-only exact world model used to validate the unchanged planner."""

    def predict(self, state: LatentState, action: Action) -> Prediction:
        next_state, reward = transition_dynamics(
            np.asarray(state.z, dtype=float), np.asarray(action.data, dtype=float)
        )
        return Prediction(
            mean=next_state,
            var=np.full(3, 1e-6),
            epistemic=0.0,
            aleatoric=1e-6,
            reward=reward,
        )

    def imagine(self, state: LatentState, actions: Sequence[Action]) -> list[Prediction]:
        current = LatentState(z=np.asarray(state.z, dtype=float).copy())
        predictions: list[Prediction] = []
        for action in actions:
            prediction = self.predict(current, action)
            predictions.append(prediction)
            current = LatentState(z=np.asarray(prediction.mean, dtype=float))
        return predictions

    def predict_batch(
        self, latents: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outputs = [
            transition_dynamics(state, action)
            for state, action in zip(latents, actions, strict=True)
        ]
        means = np.stack([output[0] for output in outputs])
        rewards = np.array([output[1] for output in outputs], dtype=float)
        n = len(means)
        return (
            means,
            np.full_like(means, 1e-6),
            np.zeros(n),
            np.full(n, 1e-6),
            rewards,
        )


def _post_actions(full_rank: bool, lane: int) -> np.ndarray:
    if lane == DECOY_LANE:
        return ACTION_CORNERS.copy()
    values = np.array([-0.9, -0.3, 0.3, 0.9])
    if full_rank:
        # Same coordinate-wise action marginals as the rank-deficient design,
        # but an orthogonal pairing with zero cross-covariance.
        return np.column_stack([values, np.array([-0.3, 0.9, -0.9, 0.3])])
    return np.column_stack([values, values])


def _bridge_y_indices(bridge: bool) -> np.ndarray:
    """Pair the same four y values with actions to expose or hide the bridge."""

    # Slot actions are (--), (-+), (+-), (++).  Present pairs the two
    # positive-longitudinal actions with the central y states; absent pairs them
    # with off-door states.  State and action marginals are identical.
    return np.array([0, 3, 2, 1] if bridge else [1, 2, 0, 3])


def generate_dataset(cell: FactorCell) -> BridgeDataset:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rewards: list[float] = []
    region_ids: list[int] = []
    next_region_ids: list[int] = []
    lane_ids: list[int] = []
    slot_ids: list[int] = []
    replicate_ids: list[int] = []

    for region, center in enumerate(REGION_CENTERS):
        for lane, y_levels in ((DOOR_LANE, DOOR_Y_LEVELS), (DECOY_LANE, DECOY_Y_LEVELS)):
            if region in POST_STABILIZATION_REGIONS:
                # Cross every base state with four action slots.  Rank changes the
                # local action pairing at identical door-lane states while each
                # action coordinate retains exactly the same marginal distribution.
                specifications = [
                    (base_index * 4 + action_slot, base_index, action)
                    for base_index in range(4)
                    for action_slot, action in enumerate(_post_actions(cell.full_rank, lane))
                ]
            else:
                y_indices = (
                    _bridge_y_indices(cell.bridge)
                    if region == BRIDGE_SOURCE_REGION and lane == DOOR_LANE
                    else np.arange(4)
                )
                specifications = [
                    (slot, int(y_index), action)
                    for slot, (action, y_index) in enumerate(
                        zip(ACTION_CORNERS, y_indices, strict=True)
                    )
                ]
            for slot, y_index, action in specifications:
                for replicate in range(REPLICATES):
                    varied_microstate = (
                        region in POST_STABILIZATION_REGIONS
                        and lane == DOOR_LANE
                        and cell.density == 8
                    )
                    x_offset = float(JITTER[replicate]) if varied_microstate else 0.0
                    y_offset = (
                        float(JITTER[(replicate + 2 * y_index) % REPLICATES])
                        if varied_microstate
                        else 0.0
                    )
                    nuisance_index = (replicate + region + lane + 2 * y_index) % REPLICATES
                    state = np.array(
                        [
                            float(center + x_offset),
                            float(y_levels[y_index] + y_offset),
                            float(NUISANCE_LEVELS[nuisance_index]),
                        ],
                        dtype=float,
                    )
                    next_state, reward = transition_dynamics(state, action)
                    states.append(state)
                    actions.append(action.copy())
                    next_states.append(next_state)
                    rewards.append(reward)
                    region_ids.append(region)
                    next_region_ids.append(region_id(float(next_state[0])))
                    lane_ids.append(lane)
                    slot_ids.append(slot)
                    replicate_ids.append(replicate)
    return BridgeDataset(
        name=cell.name,
        states=np.stack(states),
        actions=np.stack(actions),
        next_states=np.stack(next_states),
        rewards=np.array(rewards, dtype=float),
        region_ids=np.array(region_ids, dtype=np.int64),
        next_region_ids=np.array(next_region_ids, dtype=np.int64),
        lane_ids=np.array(lane_ids, dtype=np.int64),
        slot_ids=np.array(slot_ids, dtype=np.int64),
        replicate_ids=np.array(replicate_ids, dtype=np.int64),
        cell=cell,
    )


def permuted_action_control(dataset: BridgeDataset) -> BridgeDataset:
    """Corrupt action labels while retaining every state/next-state histogram."""

    if dataset.cell != FactorCell(bridge=True, full_rank=True, density=8):
        raise ValueError("action permutation is defined from the balanced b1_r1_d8 dataset")
    actions = np.empty_like(dataset.actions)
    for region in range(len(REGION_CENTERS)):
        for lane in (DOOR_LANE, DECOY_LANE):
            mask = (dataset.region_ids == region) & (dataset.lane_ids == lane)
            rows = np.flatnonzero(mask).reshape(-1, REPLICATES)
            actions[rows] = dataset.actions[np.roll(rows, shift=1, axis=0)]
    return replace(
        dataset,
        name="control_action_permuted",
        actions=actions,
        cell=None,
        control="action_permuted",
    )


def constant_nuisance_control(dataset: BridgeDataset) -> BridgeDataset:
    """Remove nuisance diversity from the bridge/full-rank/density-1 dataset."""

    if dataset.cell != FactorCell(bridge=True, full_rank=True, density=1):
        raise ValueError("constant nuisance control is defined from b1_r1_d1")
    states = dataset.states.copy()
    states[:, 2] = 0.0
    outputs = [transition_dynamics(state, action) for state, action in zip(states, dataset.actions, strict=True)]
    next_states = np.stack([output[0] for output in outputs])
    rewards = np.array([output[1] for output in outputs], dtype=float)
    return replace(
        dataset,
        name="control_nuisance_constant",
        states=states,
        next_states=next_states,
        rewards=rewards,
        next_region_ids=np.array([region_id(float(s[0])) for s in next_states], dtype=np.int64),
        cell=None,
        control="nuisance_constant",
    )


def _canonical_array(name: str, value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype="<f8", order="C")
    elif np.issubdtype(array.dtype, np.integer):
        array = np.asarray(array, dtype="<i8", order="C")
    else:
        raise TypeError(f"unsupported array dtype for {name}: {array.dtype}")
    header = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return len(header).to_bytes(8, "big") + header + array.tobytes(order="C")


def semantic_hash(dataset: BridgeDataset) -> str:
    digest = sha256()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "name": dataset.name,
        "cell": None if dataset.cell is None else dataset.cell.as_dict(),
        "control": dataset.control,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    arrays = {
        "actions": dataset.actions,
        "next_region_ids": dataset.next_region_ids,
        "next_states": dataset.next_states,
        "lane_ids": dataset.lane_ids,
        "region_ids": dataset.region_ids,
        "replicate_ids": dataset.replicate_ids,
        "rewards": dataset.rewards,
        "slot_ids": dataset.slot_ids,
        "states": dataset.states,
    }
    for name in sorted(arrays):
        digest.update(_canonical_array(name, arrays[name]))
    return digest.hexdigest()


def dataset_diagnostics(dataset: BridgeDataset) -> dict[str, object]:
    source_mask = (
        np.isin(dataset.region_ids, POST_STABILIZATION_REGIONS)
        & (dataset.lane_ids == DOOR_LANE)
    )
    local_actions = dataset.actions[source_mask]
    centered = local_actions - local_actions.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered / np.sqrt(len(centered)), compute_uv=False)
    region_counts = np.bincount(dataset.region_ids, minlength=len(REGION_CENTERS))
    probabilities = region_counts / region_counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    corner_action_counts = {
        f"{int(action[0]):+d},{int(action[1]):+d}": int(
            np.sum(np.all(dataset.actions == action, axis=1))
        )
        for action in ACTION_CORNERS
    }
    coordinate_histograms: dict[str, dict[str, int]] = {}
    for dimension in range(2):
        values, counts = np.unique(np.round(dataset.actions[:, dimension], 12), return_counts=True)
        coordinate_histograms[f"a{dimension}"] = {
            f"{float(value):+.3f}": int(count)
            for value, count in zip(values, counts, strict=True)
        }
    nuisance_values, nuisance_counts = np.unique(
        np.round(dataset.states[:, 2], 12), return_counts=True
    )
    nuisance_histogram = {
        f"{float(value):+.3f}": int(count)
        for value, count in zip(nuisance_values, nuisance_counts, strict=True)
    }
    state_node_counts = np.zeros((len(REGION_CENTERS), 2), dtype=np.int64)
    for region in range(len(REGION_CENTERS)):
        for lane in (DOOR_LANE, DECOY_LANE):
            state_node_counts[region, lane] = int(
                np.sum((dataset.region_ids == region) & (dataset.lane_ids == lane))
            )
    unique_per_cell: list[int] = []
    for region in POST_STABILIZATION_REGIONS:
        for slot in range(16):
            rows = (
                (dataset.region_ids == region)
                & (dataset.lane_ids == DOOR_LANE)
                & (dataset.slot_ids == slot)
            )
            controllable_states = dataset.states[rows, :2]
            unique_per_cell.append(len(np.unique(np.round(controllable_states, 12), axis=0)))
    bridge_edges = int(
        np.sum(
            (dataset.region_ids == BRIDGE_SOURCE_REGION)
            & (dataset.lane_ids == DOOR_LANE)
            & (dataset.next_region_ids == BRIDGE_TARGET_REGION)
        )
    )
    return {
        "semantic_hash": semantic_hash(dataset),
        "rows": len(dataset.states),
        "node_coverage": int(np.sum(state_node_counts > 0)),
        "state_region_counts": region_counts.tolist(),
        "state_region_lane_counts": state_node_counts.tolist(),
        "state_region_entropy": entropy,
        "global_corner_action_counts": corner_action_counts,
        "global_action_coordinate_histograms": coordinate_histograms,
        "nuisance_levels": sorted(float(v) for v in np.unique(dataset.states[:, 2])),
        "nuisance_histogram": nuisance_histogram,
        "bridge_edge_count": bridge_edges,
        "directed_start_goal_reachable": bridge_edges > 0,
        "minimum_forward_cut_support": bridge_edges,
        "local_action_min_singular": float(singular[-1]),
        "controllable_unique_per_cell_min": int(min(unique_per_cell)),
        "controllable_unique_per_cell_max": int(max(unique_per_cell)),
    }


def save_dataset(dataset: BridgeDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array(dataset.name),
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        rewards=dataset.rewards,
        region_ids=dataset.region_ids,
        next_region_ids=dataset.next_region_ids,
        lane_ids=dataset.lane_ids,
        slot_ids=dataset.slot_ids,
        replicate_ids=dataset.replicate_ids,
        bridge=np.array(-1 if dataset.cell is None else int(dataset.cell.bridge)),
        full_rank=np.array(-1 if dataset.cell is None else int(dataset.cell.full_rank)),
        density=np.array(-1 if dataset.cell is None else dataset.cell.density),
        control=np.array("" if dataset.control is None else dataset.control),
    )


def load_dataset(path: Path) -> BridgeDataset:
    with np.load(path, allow_pickle=False) as data:
        bridge, full_rank, density = int(data["bridge"]), int(data["full_rank"]), int(data["density"])
        cell = None if bridge < 0 else FactorCell(bool(bridge), bool(full_rank), density)
        control_raw = str(data["control"])
        return BridgeDataset(
            name=str(data["name"]),
            states=np.asarray(data["states"], dtype=float),
            actions=np.asarray(data["actions"], dtype=float),
            next_states=np.asarray(data["next_states"], dtype=float),
            rewards=np.asarray(data["rewards"], dtype=float),
            region_ids=np.asarray(data["region_ids"], dtype=np.int64),
            next_region_ids=np.asarray(data["next_region_ids"], dtype=np.int64),
            lane_ids=np.asarray(data["lane_ids"], dtype=np.int64),
            slot_ids=np.asarray(data["slot_ids"], dtype=np.int64),
            replicate_ids=np.asarray(data["replicate_ids"], dtype=np.int64),
            cell=cell,
            control=None if not control_raw else control_raw,
        )
