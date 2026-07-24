"""Private deterministic Q1 schedule and public seed/privacy contract.

The hidden-actuator harness needs paired potential outcomes without one
order-consumed pseudo-random stream.  This module derives every private value
from a canonical semantic key under an environment-only HMAC-SHA256 salt.
Only the salt commitment and a non-secret SHA-256 digest of the semantic key
are suitable for public artifacts.

The Q1 protocol remains permanently claim-ineligible. Runtime execution is
permitted only through the exact entry and single-attempt authority. These
utilities implement its ``q1v3`` randomness contract; importing or testing
them does not authorize an environment interaction.
"""

from __future__ import annotations

import base64
import hmac
import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from hashlib import sha256
from typing import Final, NoReturn

from bench.active_acquisition.problem import (
    ACQUISITION_ACTIONS,
    ActuatorRegime,
    TerminalAction,
)

EXPERIMENT_ID: Final = "WM-002"
PROTOCOL_VERSION: Final = "0.3.0-q1"
Q1_KEY_VERSION: Final = "q1v3"

MASTER_COUNT: Final = 4
EPISODES_PER_MASTER: Final = 1024
REGIMES_PER_MASTER: Final = EPISODES_PER_MASTER // 2
REHEARSAL_EPISODES_PER_MASTER: Final = 2
_DIGEST_DENOMINATOR: Final = 1 << 256
_MINIMUM_SALT_BYTES: Final = 32

THETA_BALANCED_ORDER_NAMESPACE: Final = "q1v3-theta-balanced-order"
PULSE_OUTCOME_NAMESPACE: Final = "q1v3-pulse-outcome"
NUISANCE_OUTCOME_NAMESPACE: Final = "q1v3-nuisance-outcome"
TERMINAL_OUTCOME_NAMESPACE: Final = "q1v3-terminal-outcome"
PUBLIC_UNIFORM_POLICY_NAMESPACE: Final = "q1v3-uniform"

PRIVATE_NAMESPACES: Final = (
    THETA_BALANCED_ORDER_NAMESPACE,
    PULSE_OUTCOME_NAMESPACE,
    NUISANCE_OUTCOME_NAMESPACE,
    TERMINAL_OUTCOME_NAMESPACE,
)

SEMANTIC_ACTION_IDS: Final = tuple(action.action_id for action in ACQUISITION_ACTIONS)
PULSE_ACTION_IDS: Final = ("weak", "strong", "overpowered")
Q1_ARM_IDS: Final = (
    "prospect_expected_return",
    "independent_fraction_oracle",
    "goal_only",
    "raw_observation_entropy",
    "eig_only",
    "shuffled_information",
    "uniform_random",
)

_PULSE_RELIABILITIES: Final = {
    "weak": Fraction(7, 10),
    "strong": Fraction(9, 10),
    "overpowered": Fraction(1, 1),
}
_CANONICAL_PREFIX: Final = f"{EXPERIMENT_ID}|{PROTOCOL_VERSION}|{Q1_KEY_VERSION}|"


class SeedContractError(ValueError):
    """A value cannot identify one canonical Q1 random variable."""


class Q1ExecutionMode(StrEnum):
    """Mutually exclusive orchestration modes selected by protocol bytes.

    ``PRODUCTION`` is the sole full-budget Q1 attempt and requires
    ``execution_authorized: true``. ``REHEARSAL`` exercises the identical
    orchestration shape at a deliberately tiny episode budget and requires
    ``execution_authorized: false``, so exactly one mode is reachable for any
    protocol document. A rehearsal is mechanics coverage, never Q1 evidence:
    its schedule is unbalanced at this budget, its artifacts carry a distinct
    aggregate schema, and its episode counts fail every frozen Q1 contract.
    """

    PRODUCTION = "production"
    REHEARSAL = "rehearsal"


def episodes_per_master(mode: Q1ExecutionMode) -> int:
    """Return the exact per-master/arm episode budget for one execution mode."""

    if mode is Q1ExecutionMode.PRODUCTION:
        return EPISODES_PER_MASTER
    if mode is Q1ExecutionMode.REHEARSAL:
        return REHEARSAL_EPISODES_PER_MASTER
    raise SeedContractError(f"unknown Q1 execution mode {mode!r}")


class PrivateMaterialLeakError(ValueError):
    """A public artifact contains or cannot exclude private material."""


@dataclass(frozen=True, slots=True)
class PublicSemanticKey:
    """Public membership reference with no salt or HMAC output.

    ``semantic_key_sha256`` hashes the non-secret canonical semantic key.  It
    lets producer and auditor bind the same random-variable identity without
    publishing the private HMAC digest used to realize that variable.
    """

    namespace: str
    semantic_key_sha256: str
    salt_commitment_sha256: str
    schema: str = "prospect.wm002.q1-public-semantic-key.v1"

    def as_dict(self) -> dict[str, str]:
        """Return the exact JSON-safe public representation."""

        return {
            "namespace": self.namespace,
            "salt_commitment_sha256": self.salt_commitment_sha256,
            "schema": self.schema,
            "semantic_key_sha256": self.semantic_key_sha256,
        }


@dataclass(frozen=True, slots=True)
class PublicUniformSelection:
    """Salt-independent Q1 selection from the successor protocol public rule."""

    action_id: str
    index: int
    semantic_key_sha256: str

    namespace: str = PUBLIC_UNIFORM_POLICY_NAMESPACE
    schema: str = "prospect.wm002.q1-public-uniform-selection.v1"

    def as_dict(self) -> dict[str, str | int]:
        """Return the exact JSON-safe public representation."""

        return {
            "action_id": self.action_id,
            "index": self.index,
            "namespace": self.namespace,
            "schema": self.schema,
            "semantic_key_sha256": self.semantic_key_sha256,
        }


@dataclass(frozen=True, slots=True, repr=False)
class PrivateEpisodeSeedMaterial:
    """Permissioned auditor reconstruction material for one episode and arm."""

    arm_id: str
    episode: int
    master: int
    nuisance_hmac_sha256: str
    pulse_hmac_sha256: tuple[tuple[str, str], ...]
    salt_commitment_sha256: str
    terminal_hmac_sha256: tuple[tuple[str, str], ...]
    theta: int
    theta_order_hmac_sha256: str
    schema: str = "prospect.wm002.q1-private-seed-material.v1"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(master={self.master}, episode={self.episode}, "
            f"arm_id={self.arm_id!r}, private_material=<redacted>)"
        )

    def __reduce__(self) -> NoReturn:
        raise pickle.PicklingError("PrivateEpisodeSeedMaterial cannot be pickled")

    def as_private_dict(self) -> dict[str, object]:
        """Return the explicit permissioned JSON shape; never a public row."""

        return {
            "arm_id": self.arm_id,
            "episode": self.episode,
            "hmac_sha256": {
                "nuisance": self.nuisance_hmac_sha256,
                "pulse": dict(self.pulse_hmac_sha256),
                "terminal": dict(self.terminal_hmac_sha256),
                "theta_order": self.theta_order_hmac_sha256,
            },
            "master": self.master,
            "salt_commitment_sha256": self.salt_commitment_sha256,
            "schema": self.schema,
            "theta": self.theta,
        }

    def private_hmac_digests(self) -> tuple[bytes, ...]:
        """Return all raw digests for recursive public-leak scanning."""

        hex_digests = (
            self.theta_order_hmac_sha256,
            *(digest for _, digest in self.pulse_hmac_sha256),
            self.nuisance_hmac_sha256,
            *(digest for _, digest in self.terminal_hmac_sha256),
        )
        return tuple(bytes.fromhex(digest) for digest in hex_digests)


class PrivateQ1SeedSchedule:
    """Environment-private HMAC schedule with deliberately redacted surfaces."""

    __slots__ = ("_salt", "_salt_commitment", "_theta_cache")

    def __init__(self, secret_salt: bytes) -> None:
        _validate_secret_salt(secret_salt)
        self._salt = bytes(secret_salt)
        self._salt_commitment = sha256(secret_salt).hexdigest()
        self._theta_cache: dict[int, tuple[ActuatorRegime, ...]] = {}

    @property
    def salt_commitment_sha256(self) -> str:
        """Return the only public value derived directly from the salt."""

        return self._salt_commitment

    def __repr__(self) -> str:
        return f"{type(self).__name__}(salt_commitment_sha256={self._salt_commitment!r}, secret_salt=<redacted>)"

    def __getstate__(self) -> NoReturn:
        """Block accidental generic serialization of the environment secret."""

        raise TypeError("PrivateQ1SeedSchedule cannot be serialized")

    def __reduce__(self) -> NoReturn:
        """Block pickle even if it bypasses ``__getstate__``."""

        raise pickle.PicklingError("PrivateQ1SeedSchedule cannot be pickled")

    def public_binding(self) -> dict[str, str]:
        """Return the minimal JSON-safe public salt binding."""

        return {
            "algorithm": "HMAC-SHA256",
            "key_version": Q1_KEY_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "salt_commitment_sha256": self._salt_commitment,
            "schema": "prospect.wm002.q1-seed-binding.v1",
        }

    def theta_schedule(self, master: int) -> tuple[ActuatorRegime, ...]:
        """Return the private, exactly balanced regime schedule for one master."""

        canonical_master = _require_master(master)
        cached = self._theta_cache.get(canonical_master)
        if cached is not None:
            return cached

        ranked = sorted(
            (
                self._private_digest(
                    _theta_semantic_key(canonical_master, episode),
                ),
                episode,
            )
            for episode in range(EPISODES_PER_MASTER)
        )
        reversed_episodes = {episode for _, episode in ranked[:REGIMES_PER_MASTER]}
        schedule = tuple(
            ActuatorRegime.REVERSED if episode in reversed_episodes else ActuatorRegime.DIRECT
            for episode in range(EPISODES_PER_MASTER)
        )
        self._theta_cache[canonical_master] = schedule
        return schedule

    def theta(self, master: int, episode: int) -> ActuatorRegime:
        """Return the private regime assigned to one canonical episode."""

        canonical_master = _require_master(master)
        canonical_episode = _require_episode(episode)
        return self.theta_schedule(canonical_master)[canonical_episode]

    def theta_reference(self, master: int, episode: int) -> PublicSemanticKey:
        """Return a public reference to the hidden schedule variable, not its value."""

        return self._public_reference(_theta_semantic_key(_require_master(master), _require_episode(episode)))

    def acquisition_observation(
        self,
        master: int,
        episode: int,
        semantic_action: str,
    ) -> int:
        """Realize the selected acquisition symbol with exact 256-bit arithmetic."""

        canonical_master = _require_master(master)
        canonical_episode = _require_episode(episode)
        action_id = _require_semantic_action(semantic_action)
        if action_id == "skip":
            return 0
        if action_id == "nuisance":
            return self.nuisance_observation(canonical_master, canonical_episode)
        return self.pulse_observation(
            canonical_master,
            canonical_episode,
            action_id,
            self.theta(canonical_master, canonical_episode),
        )

    def pulse_observation(
        self,
        master: int,
        episode: int,
        semantic_action: str,
        theta: ActuatorRegime | int,
    ) -> int:
        """Realize one pulse observation from its action-keyed potential draw."""

        canonical_master = _require_master(master)
        canonical_episode = _require_episode(episode)
        action_id = _require_pulse_action(semantic_action)
        regime = _require_regime(theta)
        return pulse_observation_from_digest(
            self._private_digest(_pulse_semantic_key(canonical_master, canonical_episode, action_id)),
            action_id,
            regime,
        )

    def pulse_reference(self, master: int, episode: int, semantic_action: str) -> PublicSemanticKey:
        """Return the public semantic identity of one private pulse draw."""

        return self._public_reference(
            _pulse_semantic_key(
                _require_master(master),
                _require_episode(episode),
                _require_pulse_action(semantic_action),
            )
        )

    def nuisance_observation(self, master: int, episode: int) -> int:
        """Realize the four-way nuisance outcome without an order-consumed RNG."""

        return nuisance_observation_from_digest(
            self._private_digest(_nuisance_semantic_key(_require_master(master), _require_episode(episode)))
        )

    def nuisance_reference(self, master: int, episode: int) -> PublicSemanticKey:
        """Return the public semantic identity of one private nuisance draw."""

        return self._public_reference(_nuisance_semantic_key(_require_master(master), _require_episode(episode)))

    def terminal_success(
        self,
        master: int,
        episode: int,
        terminal_decision: TerminalAction | int,
        theta: ActuatorRegime | int,
    ) -> bool:
        """Realize terminal success keyed by decision, deliberately not by arm."""

        canonical_master = _require_master(master)
        canonical_episode = _require_episode(episode)
        decision = _require_terminal_decision(terminal_decision)
        regime = _require_regime(theta)
        return terminal_success_from_digest(
            self._private_digest(_terminal_semantic_key(canonical_master, canonical_episode, decision)),
            decision,
            regime,
        )

    def terminal_reference(
        self,
        master: int,
        episode: int,
        terminal_decision: TerminalAction | int,
    ) -> PublicSemanticKey:
        """Return a decision-keyed reference containing no arm identity."""

        return self._public_reference(
            _terminal_semantic_key(
                _require_master(master),
                _require_episode(episode),
                _require_terminal_decision(terminal_decision),
            )
        )

    def private_audit_material(
        self,
        master: int,
        episode: int,
        arm_id: str,
    ) -> PrivateEpisodeSeedMaterial:
        """Return the permissioned per-arm reconstruction sidecar value."""

        canonical_master = _require_master(master)
        canonical_episode = _require_episode(episode)
        canonical_arm = _require_arm_id(arm_id)
        theta_key = _theta_semantic_key(canonical_master, canonical_episode)
        nuisance_key = _nuisance_semantic_key(canonical_master, canonical_episode)
        return PrivateEpisodeSeedMaterial(
            arm_id=canonical_arm,
            episode=canonical_episode,
            master=canonical_master,
            nuisance_hmac_sha256=self._private_digest(nuisance_key).hex(),
            pulse_hmac_sha256=tuple(
                (
                    action_id,
                    self._private_digest(_pulse_semantic_key(canonical_master, canonical_episode, action_id)).hex(),
                )
                for action_id in PULSE_ACTION_IDS
            ),
            salt_commitment_sha256=self._salt_commitment,
            terminal_hmac_sha256=tuple(
                (
                    _signed_integer(decision),
                    self._private_digest(_terminal_semantic_key(canonical_master, canonical_episode, decision)).hex(),
                )
                for decision in (1, -1)
            ),
            theta=int(self.theta(canonical_master, canonical_episode)),
            theta_order_hmac_sha256=self._private_digest(theta_key).hex(),
        )

    def assert_public_artifact(
        self,
        value: object,
        *,
        theta_sentinels: Sequence[object] = (),
        private_hmac_digests: Sequence[bytes] = (),
        known_private_draw_values: Sequence[object] = (),
    ) -> None:
        """Reject nested public data containing supplied private sentinels.

        The environment salt is always scanned in raw, UTF-8, hexadecimal, and
        base64 forms.  Callers add recognizable hidden-regime sentinels, any
        HMAC digests held by the private harness/auditor, and deliberately
        distinctive counterfactual draw values used by isolation tests.
        """

        private_bytes = (self._salt, *private_hmac_digests)
        violations = private_material_paths(
            value,
            private_bytes=private_bytes,
            private_values=(*theta_sentinels, *known_private_draw_values),
        )
        if violations:
            raise PrivateMaterialLeakError(
                "public artifact contains or cannot exclude private material at " + ", ".join(violations)
            )

    def _private_digest(self, semantic_key: bytes) -> bytes:
        return hmac.digest(self._salt, semantic_key, "sha256")

    def _public_reference(self, semantic_key: bytes) -> PublicSemanticKey:
        namespace = semantic_key.decode("ascii").split("|", 4)[3]
        return PublicSemanticKey(
            namespace=namespace,
            semantic_key_sha256=sha256(semantic_key).hexdigest(),
            salt_commitment_sha256=self._salt_commitment,
        )


def pulse_observation_from_digest(
    digest: bytes,
    semantic_action: str,
    theta: ActuatorRegime | int,
) -> int:
    """Map one full HMAC digest to a pulse symbol without binary64."""

    draw = _private_digest_integer(digest)
    action_id = _require_pulse_action(semantic_action)
    regime = _require_regime(theta)
    reliability = _PULSE_RELIABILITIES[action_id]
    agrees = draw * reliability.denominator < reliability.numerator * _DIGEST_DENOMINATOR
    return int(regime if agrees else -regime)


def nuisance_observation_from_digest(digest: bytes) -> int:
    """Map one full HMAC digest to one of four exact equal-width bins."""

    return _private_digest_integer(digest) * 4 // _DIGEST_DENOMINATOR


def terminal_success_from_digest(
    digest: bytes,
    terminal_decision: TerminalAction | int,
    theta: ActuatorRegime | int,
) -> bool:
    """Map one full HMAC digest to terminal success without binary64."""

    draw = _private_digest_integer(digest)
    decision = _require_terminal_decision(terminal_decision)
    regime = _require_regime(theta)
    probability = Fraction(9, 10) if decision == regime else Fraction(1, 10)
    return draw * probability.denominator < probability.numerator * _DIGEST_DENOMINATOR


def derive_public_uniform_selection(master: int, episode: int) -> PublicUniformSelection:
    """Derive the Q1 uniform arm selection without the environment secret.

    The successor protocol freezes the exact payload
    ``WM-002|0.3.0-q1|q1v3-uniform|<master>|<episode>``. Its full SHA-256
    digest is interpreted as an unsigned big-endian integer modulo the five
    canonical acquisition actions.
    """

    semantic_key = _uniform_semantic_key(_require_master(master), _require_episode(episode))
    semantic_digest = sha256(semantic_key).hexdigest()
    index = int(semantic_digest, 16) % len(ACQUISITION_ACTIONS)
    action = ACQUISITION_ACTIONS[index]
    return PublicUniformSelection(
        action_id=action.action_id,
        index=index,
        semantic_key_sha256=semantic_digest,
    )


def public_identity_counter_initialization(
    master: int,
    episode: int,
    arm_id: str,
) -> int:
    """Return the public counter start for a uniquely namespaced episode.

    Record identity namespaces already contain the master, arm, and episode.
    Starting every fresh namespace at zero is deterministic, collision-free
    across lanes, and—unlike an HMAC-derived start—cannot disclose private
    scheduling material through public record identifiers or checkpoints.
    """

    _require_master(master)
    _require_episode(episode)
    _require_arm_id(arm_id)
    return 0


def private_material_paths(
    value: object,
    *,
    private_bytes: Sequence[bytes] = (),
    private_values: Sequence[object] = (),
) -> tuple[str, ...]:
    """Return recursive paths that leak sentinels or violate JSON shape.

    Text representations of private bytes are scanned because actual public
    artifacts are JSON and therefore commonly encode binary material as
    hexadecimal or base64.  Non-JSON types, non-string keys, non-finite
    numbers, and cycles are rejected rather than assumed safe.
    """

    byte_patterns = _private_byte_patterns(private_bytes)
    violations: list[str] = []
    active: set[int] = set()

    def visit(node: object, path: str) -> None:
        if _matches_private_value(node, private_values):
            violations.append(path)
            return
        if isinstance(node, str):
            if any(pattern in node for pattern in byte_patterns):
                violations.append(path)
            return
        if node is None or isinstance(node, bool) or isinstance(node, int):
            return
        if isinstance(node, float):
            if not math.isfinite(node):
                violations.append(f"{path}:non_finite_float")
            return
        if isinstance(node, bytes):
            violations.append(f"{path}:bytes_not_public_json")
            return

        identity = id(node)
        if identity in active:
            violations.append(f"{path}:cycle")
            return
        active.add(identity)
        try:
            if isinstance(node, Mapping):
                for key, item in node.items():
                    if not isinstance(key, str):
                        violations.append(f"{path}:non_string_key")
                        continue
                    key_path = f"{path}.{key}"
                    if _matches_private_value(key, private_values) or any(pattern in key for pattern in byte_patterns):
                        violations.append(f"{key_path}:private_key")
                    visit(item, key_path)
                return
            if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                for index, item in enumerate(node):
                    visit(item, f"{path}[{index}]")
                return
            violations.append(f"{path}:non_json_type:{type(node).__name__}")
        finally:
            active.remove(identity)

    visit(value, "$")
    return tuple(dict.fromkeys(violations))


def canonical_public_json(value: object) -> bytes:
    """Serialize a sentinel-checked public value as canonical UTF-8 JSON."""

    violations = private_material_paths(value)
    if violations:
        raise PrivateMaterialLeakError("value is not a canonical public JSON shape at " + ", ".join(violations))
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _theta_semantic_key(master: int, episode: int) -> bytes:
    return _semantic_key(
        THETA_BALANCED_ORDER_NAMESPACE,
        ("master", master),
        ("episode", episode),
        ("role", "order"),
    )


def _pulse_semantic_key(master: int, episode: int, action_id: str) -> bytes:
    return _semantic_key(
        PULSE_OUTCOME_NAMESPACE,
        ("master", master),
        ("episode", episode),
        ("action", action_id),
        ("role", "observation"),
    )


def _nuisance_semantic_key(master: int, episode: int) -> bytes:
    return _semantic_key(
        NUISANCE_OUTCOME_NAMESPACE,
        ("master", master),
        ("episode", episode),
        ("action", "nuisance"),
        ("role", "observation"),
    )


def _terminal_semantic_key(master: int, episode: int, decision: int) -> bytes:
    return _semantic_key(
        TERMINAL_OUTCOME_NAMESPACE,
        ("master", master),
        ("episode", episode),
        ("decision", _signed_integer(decision)),
        ("role", "success"),
    )


def _uniform_semantic_key(master: int, episode: int) -> bytes:
    return (f"{EXPERIMENT_ID}|{PROTOCOL_VERSION}|{PUBLIC_UNIFORM_POLICY_NAMESPACE}|{master}|{episode}").encode("ascii")


def _semantic_key(namespace: str, *fields: tuple[str, int | str]) -> bytes:
    if namespace not in (*PRIVATE_NAMESPACES, PUBLIC_UNIFORM_POLICY_NAMESPACE):
        raise SeedContractError("undeclared q1v3 namespace")
    segments = [namespace]
    for name, value in fields:
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz_" for character in name):
            raise SeedContractError("semantic field names must be canonical lowercase identifiers")
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise SeedContractError("semantic field values must be canonical integers or strings")
        rendered = str(value)
        if not rendered or any(character in rendered for character in "|=\r\n\t "):
            raise SeedContractError("semantic field value is not canonically delimited")
        segments.append(f"{name}={rendered}")
    return (_CANONICAL_PREFIX + "|".join(segments)).encode("ascii")


def _require_master(value: object) -> int:
    if type(value) is not int or not 0 <= value < MASTER_COUNT:
        raise SeedContractError(f"master must be a canonical integer in [0,{MASTER_COUNT - 1}]")
    return value


def _require_episode(value: object) -> int:
    if type(value) is not int or not 0 <= value < EPISODES_PER_MASTER:
        raise SeedContractError(f"episode must be a canonical integer in [0,{EPISODES_PER_MASTER - 1}]")
    return value


def _require_semantic_action(value: object) -> str:
    if type(value) is not str or value not in SEMANTIC_ACTION_IDS:
        raise SeedContractError(f"semantic action must be one of {SEMANTIC_ACTION_IDS!r}")
    return value


def _require_pulse_action(value: object) -> str:
    if type(value) is not str or value not in PULSE_ACTION_IDS:
        raise SeedContractError(f"pulse action must be one of {PULSE_ACTION_IDS!r}")
    return value


def _require_arm_id(value: object) -> str:
    if type(value) is not str or value not in Q1_ARM_IDS:
        raise SeedContractError("arm_id is not one of the seven declared Q1 arms")
    return value


def _require_regime(value: ActuatorRegime | int) -> ActuatorRegime:
    if isinstance(value, bool):
        raise SeedContractError("theta must be exactly -1 or +1")
    try:
        regime = ActuatorRegime(value)
    except (TypeError, ValueError) as error:
        raise SeedContractError("theta must be exactly -1 or +1") from error
    if not isinstance(value, (int, ActuatorRegime)):
        raise SeedContractError("theta must be a canonical integer or ActuatorRegime")
    return regime


def _require_terminal_decision(value: TerminalAction | int) -> int:
    if isinstance(value, bool):
        raise SeedContractError("terminal decision must be exactly -1 or +1")
    try:
        decision = TerminalAction(value)
    except (TypeError, ValueError) as error:
        raise SeedContractError("terminal decision must be exactly -1 or +1") from error
    if not isinstance(value, (int, TerminalAction)):
        raise SeedContractError("terminal decision must be a canonical integer or TerminalAction")
    return int(decision)


def _signed_integer(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _private_digest_integer(digest: bytes) -> int:
    if type(digest) is not bytes or len(digest) != 32:
        raise SeedContractError("private realization digest must be exactly 32 bytes")
    return int.from_bytes(digest, "big")


def _validate_secret_salt(value: object) -> None:
    if type(value) is not bytes:
        raise SeedContractError("secret_salt must be immutable bytes")
    if len(value) < _MINIMUM_SALT_BYTES:
        raise SeedContractError(f"secret_salt must contain at least {_MINIMUM_SALT_BYTES} bytes")


def _private_byte_patterns(values: Sequence[bytes]) -> tuple[str, ...]:
    patterns: list[str] = []
    for value in values:
        if type(value) is not bytes or not value:
            raise SeedContractError("private byte sentinels must be nonempty immutable bytes")
        minimum_prefix_bytes = min(16, len(value))
        for prefix_length in range(minimum_prefix_bytes, len(value) + 1):
            prefix = value[:prefix_length]
            patterns.extend(
                (
                    prefix.hex(),
                    prefix.hex().upper(),
                    base64.b64encode(prefix).decode("ascii").rstrip("="),
                    base64.b64encode(prefix).decode("ascii"),
                    base64.urlsafe_b64encode(prefix).decode("ascii"),
                    base64.urlsafe_b64encode(prefix).decode("ascii").rstrip("="),
                )
            )
            try:
                decoded = prefix.decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                if decoded:
                    patterns.append(decoded)
    return tuple(dict.fromkeys(patterns))


def _matches_private_value(value: object, private_values: Sequence[object]) -> bool:
    for private in private_values:
        if type(value) is type(private) and value == private:
            return True
    return False


__all__ = (
    "EPISODES_PER_MASTER",
    "EXPERIMENT_ID",
    "MASTER_COUNT",
    "NUISANCE_OUTCOME_NAMESPACE",
    "PRIVATE_NAMESPACES",
    "PROTOCOL_VERSION",
    "PULSE_OUTCOME_NAMESPACE",
    "PrivateEpisodeSeedMaterial",
    "PrivateMaterialLeakError",
    "PrivateQ1SeedSchedule",
    "PublicSemanticKey",
    "PublicUniformSelection",
    "Q1_ARM_IDS",
    "Q1_KEY_VERSION",
    "Q1ExecutionMode",
    "REGIMES_PER_MASTER",
    "REHEARSAL_EPISODES_PER_MASTER",
    "SEMANTIC_ACTION_IDS",
    "SeedContractError",
    "TERMINAL_OUTCOME_NAMESPACE",
    "THETA_BALANCED_ORDER_NAMESPACE",
    "canonical_public_json",
    "derive_public_uniform_selection",
    "episodes_per_master",
    "nuisance_observation_from_digest",
    "private_material_paths",
    "pulse_observation_from_digest",
    "public_identity_counter_initialization",
    "terminal_success_from_digest",
)
