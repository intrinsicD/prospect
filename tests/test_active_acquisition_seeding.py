from __future__ import annotations

import base64
import hmac
import json
import pickle
from collections.abc import Callable
from hashlib import sha256

import pytest

from bench.active_acquisition.problem import (
    ACQUISITION_ACTIONS,
    ActuatorRegime,
    TerminalAction,
)
from bench.active_acquisition.seeding import (
    EPISODES_PER_MASTER,
    MASTER_COUNT,
    PRIVATE_NAMESPACES,
    PULSE_OUTCOME_NAMESPACE,
    REGIMES_PER_MASTER,
    TERMINAL_OUTCOME_NAMESPACE,
    THETA_BALANCED_ORDER_NAMESPACE,
    PrivateMaterialLeakError,
    PrivateQ1SeedSchedule,
    SeedContractError,
    canonical_public_json,
    derive_public_uniform_selection,
    nuisance_observation_from_digest,
    private_material_paths,
    public_identity_counter_initialization,
    pulse_observation_from_digest,
    terminal_success_from_digest,
)

_SALT = b"wm002-private-environment-salt!" * 2
_OTHER_SALT = b"wm002-independent-auditor-salt" * 2


def _digest_bytes(value: int) -> bytes:
    return value.to_bytes(32, "big")


def test_private_schedule_uses_a_stable_commitment_and_redacted_surfaces() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)

    assert schedule.salt_commitment_sha256 == sha256(_SALT).hexdigest()
    assert schedule.public_binding() == {
        "algorithm": "HMAC-SHA256",
        "key_version": "q1v3",
        "protocol_version": "0.3.0-q1",
        "salt_commitment_sha256": sha256(_SALT).hexdigest(),
        "schema": "prospect.wm002.q1-seed-binding.v1",
    }
    rendered = repr(schedule)
    assert _SALT.decode() not in rendered
    assert _SALT.hex() not in rendered
    assert "<redacted>" in rendered
    with pytest.raises((TypeError, pickle.PicklingError), match="pickl|serial"):
        pickle.dumps(schedule)
    with pytest.raises(TypeError):
        json.dumps(schedule)


def test_every_master_has_exactly_512_regimes_in_private_order() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)

    assert MASTER_COUNT == 4
    assert EPISODES_PER_MASTER == 1024
    assert REGIMES_PER_MASTER == 512
    for master in range(MASTER_COUNT):
        regimes = schedule.theta_schedule(master)
        assert len(regimes) == EPISODES_PER_MASTER
        assert regimes.count(ActuatorRegime.REVERSED) == REGIMES_PER_MASTER
        assert regimes.count(ActuatorRegime.DIRECT) == REGIMES_PER_MASTER
        assert regimes == schedule.theta_schedule(master)
        assert all(schedule.theta(master, episode) == regimes[episode] for episode in range(1024))


def test_private_salt_changes_order_without_changing_balance() -> None:
    first = PrivateQ1SeedSchedule(_SALT)
    second = PrivateQ1SeedSchedule(_OTHER_SALT)

    assert first.theta_schedule(0) != second.theta_schedule(0)
    assert first.theta_schedule(0).count(ActuatorRegime.REVERSED) == REGIMES_PER_MASTER
    assert second.theta_schedule(0).count(ActuatorRegime.REVERSED) == REGIMES_PER_MASTER


def test_semantic_potential_outcomes_do_not_depend_on_iteration_order() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    keys = [
        (master, episode, action)
        for master in range(MASTER_COUNT)
        for episode in (0, 1, 17, 511, 512, 1023)
        for action in ("weak", "strong", "overpowered")
    ]

    forward = {key: schedule.pulse_observation(*key, schedule.theta(key[0], key[1])) for key in keys}
    reverse = {key: schedule.pulse_observation(*key, schedule.theta(key[0], key[1])) for key in reversed(keys)}
    nuisance_forward = {
        (master, episode): schedule.nuisance_observation(master, episode) for master, episode, _ in keys
    }
    nuisance_reverse = {
        (master, episode): schedule.nuisance_observation(master, episode) for master, episode, _ in reversed(keys)
    }

    assert forward == reverse
    assert nuisance_forward == nuisance_reverse
    assert set(forward.values()) <= {-1, 1}
    assert set(nuisance_forward.values()) <= {0, 1, 2, 3}


def test_terminal_draw_is_keyed_by_decision_and_contains_no_arm_dimension() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    theta = schedule.theta(2, 73)

    direct = tuple(
        schedule.terminal_success(2, 73, TerminalAction.DIRECT, theta)
        for _arm in (
            "prospect_expected_return",
            "goal_only",
            "uniform_random",
        )
    )
    reversed_action = schedule.terminal_success(2, 73, TerminalAction.REVERSED, theta)

    assert len(set(direct)) == 1
    assert schedule.terminal_reference(2, 73, 1).namespace == TERMINAL_OUTCOME_NAMESPACE
    assert (
        schedule.terminal_reference(2, 73, 1).semantic_key_sha256
        != schedule.terminal_reference(2, 73, -1).semantic_key_sha256
    )
    assert isinstance(reversed_action, bool)
    with pytest.raises(TypeError, match="unexpected keyword"):
        schedule.terminal_success(2, 73, 1, theta, arm_id="goal_only")  # type: ignore[call-arg]


def test_all_private_namespaces_are_distinct_and_references_bind_semantic_keys() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    references = (
        schedule.theta_reference(1, 9),
        schedule.pulse_reference(1, 9, "strong"),
        schedule.nuisance_reference(1, 9),
        schedule.terminal_reference(1, 9, TerminalAction.DIRECT),
    )

    assert len(PRIVATE_NAMESPACES) == 4
    assert len(set(PRIVATE_NAMESPACES)) == len(PRIVATE_NAMESPACES)
    assert {reference.namespace for reference in references} == set(PRIVATE_NAMESPACES)
    assert len({reference.semantic_key_sha256 for reference in references}) == len(references)
    assert {reference.salt_commitment_sha256 for reference in references} == {schedule.salt_commitment_sha256}
    assert references[0].namespace == THETA_BALANCED_ORDER_NAMESPACE
    assert references[1].namespace == PULSE_OUTCOME_NAMESPACE
    assert references[-1].namespace == TERMINAL_OUTCOME_NAMESPACE


def test_public_semantic_reference_has_no_private_preimage_or_hmac_output() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    reference = schedule.pulse_reference(3, 101, "weak")
    public = reference.as_dict()
    canonical_preimage = b"WM-002|0.3.0-q1|q1v3|q1v3-pulse-outcome|master=3|episode=101|action=weak|role=observation"
    private_hmac = hmac.digest(_SALT, canonical_preimage, "sha256")

    assert public["semantic_key_sha256"] == sha256(canonical_preimage).hexdigest()
    assert public["salt_commitment_sha256"] == sha256(_SALT).hexdigest()
    assert canonical_preimage.decode() not in json.dumps(public, sort_keys=True)
    assert private_hmac.hex() not in json.dumps(public, sort_keys=True)
    schedule.assert_public_artifact(public, private_hmac_digests=(private_hmac,))
    assert canonical_public_json(public).endswith(b"\n")


def test_public_uniform_derivation_is_salt_independent_and_uses_successor_rule() -> None:
    first = derive_public_uniform_selection(2, 37)
    repeated = derive_public_uniform_selection(2, 37)
    other = derive_public_uniform_selection(2, 38)

    payload = b"WM-002|0.3.0-q1|q1v3-uniform|2|37"
    digest = sha256(payload).hexdigest()
    index = int(digest, 16) % len(ACQUISITION_ACTIONS)
    assert first == repeated
    assert first != other
    assert (first.semantic_key_sha256, first.index, first.action_id) == (
        digest,
        index,
        ACQUISITION_ACTIONS[index].action_id,
    )
    assert first.namespace == "q1v3-uniform"
    assert PrivateQ1SeedSchedule(_SALT).salt_commitment_sha256 not in json.dumps(first.as_dict())
    assert PrivateQ1SeedSchedule(_OTHER_SALT).salt_commitment_sha256 not in json.dumps(first.as_dict())


def test_identity_counter_start_is_public_zero_and_salt_independent() -> None:
    first = public_identity_counter_initialization(2, 37, "prospect_expected_return")
    second = public_identity_counter_initialization(2, 37, "prospect_expected_return")

    assert first == second == 0
    assert not hasattr(PrivateQ1SeedSchedule(_SALT), "identity_counter_initialization")
    assert not hasattr(PrivateQ1SeedSchedule(_OTHER_SALT), "identity_counter_reference")
    with pytest.raises(SeedContractError):
        public_identity_counter_initialization(2, 37, "unknown_arm")


def test_exact_schedule_realizes_every_fixture_outcome_without_binary64() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)

    assert not hasattr(schedule, "acquisition_unit_interval")
    assert not hasattr(schedule, "terminal_unit_interval")

    for master in range(MASTER_COUNT):
        for episode in range(EPISODES_PER_MASTER):
            theta = schedule.theta(master, episode)
            for action in ACQUISITION_ACTIONS:
                action_id = action.action_id
                observed = schedule.acquisition_observation(master, episode, action_id)
                if action_id == "skip":
                    assert observed == 0
                elif action_id == "nuisance":
                    assert observed == schedule.nuisance_observation(master, episode)
                else:
                    assert observed == schedule.pulse_observation(
                        master,
                        episode,
                        action_id,
                        theta,
                    )
            for decision in (TerminalAction.DIRECT, TerminalAction.REVERSED):
                success = schedule.terminal_success(master, episode, decision, theta)
                assert type(success) is bool


@pytest.mark.parametrize(
    ("action_id", "numerator", "denominator"),
    (
        ("weak", 7, 10),
        ("strong", 9, 10),
    ),
)
def test_pulse_realization_is_exact_immediately_around_rational_thresholds(
    action_id: str,
    numerator: int,
    denominator: int,
) -> None:
    digest_denominator = 1 << 256
    first_failure = (numerator * digest_denominator + denominator - 1) // denominator

    assert pulse_observation_from_digest(_digest_bytes(first_failure - 1), action_id, ActuatorRegime.DIRECT) == 1
    assert pulse_observation_from_digest(_digest_bytes(first_failure), action_id, ActuatorRegime.DIRECT) == -1


def test_nuisance_realization_is_exact_immediately_around_bin_boundaries() -> None:
    digest_denominator = 1 << 256
    for bin_index in (1, 2, 3):
        boundary = bin_index * digest_denominator // 4
        assert nuisance_observation_from_digest(_digest_bytes(boundary - 1)) == bin_index - 1
        assert nuisance_observation_from_digest(_digest_bytes(boundary)) == bin_index


@pytest.mark.parametrize(
    ("decision", "regime", "numerator", "denominator"),
    (
        (TerminalAction.DIRECT, ActuatorRegime.DIRECT, 9, 10),
        (TerminalAction.DIRECT, ActuatorRegime.REVERSED, 1, 10),
    ),
)
def test_terminal_realization_is_exact_immediately_around_rational_thresholds(
    decision: TerminalAction,
    regime: ActuatorRegime,
    numerator: int,
    denominator: int,
) -> None:
    digest_denominator = 1 << 256
    first_failure = (numerator * digest_denominator + denominator - 1) // denominator

    assert not terminal_success_from_digest(_digest_bytes(first_failure), decision, regime)
    assert terminal_success_from_digest(_digest_bytes(first_failure - 1), decision, regime)


@pytest.mark.parametrize(
    "digest",
    (
        b"x" * 31,
        bytearray(b"x" * 32),
    ),
)
def test_exact_realization_helpers_require_one_full_sha256_digest(digest: object) -> None:
    with pytest.raises(SeedContractError, match="exactly 32 bytes"):
        pulse_observation_from_digest(digest, "strong", 1)  # type: ignore[arg-type]


def test_permissioned_sidecar_freezes_per_arm_private_reconstruction_schema() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    prospect = schedule.private_audit_material(2, 73, "prospect_expected_return")
    control = schedule.private_audit_material(2, 73, "goal_only")
    private = prospect.as_private_dict()
    hmac_rows = private["hmac_sha256"]
    assert isinstance(hmac_rows, dict)
    pulse_rows = hmac_rows["pulse"]
    terminal_rows = hmac_rows["terminal"]
    assert isinstance(pulse_rows, dict)
    assert isinstance(terminal_rows, dict)

    assert set(private) == {
        "arm_id",
        "episode",
        "hmac_sha256",
        "master",
        "salt_commitment_sha256",
        "schema",
        "theta",
    }
    assert set(hmac_rows) == {
        "nuisance",
        "pulse",
        "terminal",
        "theta_order",
    }
    assert tuple(pulse_rows) == ("weak", "strong", "overpowered")
    assert tuple(terminal_rows) == ("+1", "-1")
    assert len(prospect.private_hmac_digests()) == 7
    assert all(len(digest) == 32 for digest in prospect.private_hmac_digests())
    assert prospect.theta == int(schedule.theta(2, 73))
    assert prospect.theta_order_hmac_sha256 == control.theta_order_hmac_sha256
    assert prospect.pulse_hmac_sha256 == control.pulse_hmac_sha256
    assert prospect.nuisance_hmac_sha256 == control.nuisance_hmac_sha256
    assert prospect.terminal_hmac_sha256 == control.terminal_hmac_sha256
    serialized = json.dumps(private, sort_keys=True)
    assert _SALT.decode() not in serialized
    assert _SALT.hex() not in serialized
    assert "private_material=<redacted>" in repr(prospect)
    assert prospect.theta_order_hmac_sha256 not in repr(prospect)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(prospect)


def test_full_and_truncated_hmac_values_are_rejected_from_public_artifacts() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    material = schedule.private_audit_material(0, 5, "uniform_random")
    full_hmac = material.private_hmac_digests()[0]
    truncated_hmac = full_hmac[:16]
    leaking_artifacts = (
        {"full_hex": full_hmac.hex()},
        {"truncated_hex": truncated_hmac.hex()},
        {"full_base64": base64.b64encode(full_hmac).decode("ascii")},
        {"truncated_base64": base64.b64encode(truncated_hmac).decode("ascii")},
        {"truncated_base64_unpadded": base64.b64encode(truncated_hmac).decode("ascii").rstrip("=")},
    )

    for artifact in leaking_artifacts:
        with pytest.raises(PrivateMaterialLeakError, match="private material"):
            schedule.assert_public_artifact(
                artifact,
                private_hmac_digests=material.private_hmac_digests(),
            )


@pytest.mark.parametrize(
    ("method", "args"),
    (
        ("theta", (True, 0)),
        ("theta", (-1, 0)),
        ("theta", (4, 0)),
        ("theta", (0, True)),
        ("theta", (0, -1)),
        ("theta", (0, 1024)),
        ("acquisition_observation", (0, 0, "weak:sign_inverted_non_candidate")),
        ("acquisition_observation", (0, 0, "unknown")),
        ("pulse_observation", (0, 0, "weak:sign_inverted_non_candidate", 1)),
        ("pulse_observation", (0, 0, "skip", 1)),
        ("pulse_observation", (0, 0, "strong", 0)),
        ("terminal_success", (0, 0, True, 1)),
        ("terminal_success", (0, 0, 0, 1)),
        ("terminal_success", (0, 0, 1, 0)),
    ),
)
def test_semantic_inputs_reject_noncanonical_or_out_of_domain_values(
    method: str,
    args: tuple[object, ...],
) -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    operation: Callable[..., object] = getattr(schedule, method)

    with pytest.raises(SeedContractError):
        operation(*args)


@pytest.mark.parametrize(
    "salt",
    (
        "not-bytes",
        bytearray(b"x" * 32),
        b"",
        b"x" * 31,
    ),
)
def test_secret_salt_requires_at_least_256_bits_of_immutable_key_material(salt: object) -> None:
    with pytest.raises(SeedContractError, match="secret_salt"):
        PrivateQ1SeedSchedule(salt)  # type: ignore[arg-type]


def test_recursive_privacy_sentinel_finds_every_declared_private_form() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    theta_sentinel = "PRIVATE_THETA_SENTINEL_7f3a"
    raw_hmac = bytes.fromhex("12" * 32)
    known_private_draw = 918273645546372819

    leaking_artifacts = (
        {"nested": [{"theta": theta_sentinel}]},
        {"raw_salt": _SALT},
        {"salt_hex": _SALT.hex()},
        {"salt_base64": base64.b64encode(_SALT).decode()},
        {"raw_hmac": raw_hmac},
        {"hmac_hex": raw_hmac.hex()},
        {"private_draw": known_private_draw},
        {theta_sentinel: "sentinel hidden in a mapping key"},
    )
    for artifact in leaking_artifacts:
        with pytest.raises(PrivateMaterialLeakError, match="private material"):
            schedule.assert_public_artifact(
                artifact,
                theta_sentinels=(theta_sentinel,),
                private_hmac_digests=(raw_hmac,),
                known_private_draw_values=(known_private_draw,),
            )


def test_recursive_privacy_sentinel_accepts_commitments_and_public_references() -> None:
    schedule = PrivateQ1SeedSchedule(_SALT)
    artifact = {
        "binding": schedule.public_binding(),
        "keys": [
            schedule.theta_reference(0, 0).as_dict(),
            schedule.pulse_reference(0, 0, "strong").as_dict(),
            schedule.terminal_reference(0, 0, 1).as_dict(),
        ],
        "uniform": derive_public_uniform_selection(0, 0).as_dict(),
    }

    schedule.assert_public_artifact(
        artifact,
        theta_sentinels=("PRIVATE_THETA_SENTINEL",),
        private_hmac_digests=(bytes.fromhex("ab" * 32),),
        known_private_draw_values=(123456789987654321,),
    )
    payload = canonical_public_json(artifact)
    assert json.loads(payload) == artifact


def test_recursive_public_shape_validation_rejects_ambiguous_values() -> None:
    cycle: list[object] = []
    cycle.append(cycle)

    assert private_material_paths({1: "non-string key"})
    assert private_material_paths({"nan": float("nan")})
    assert private_material_paths({"raw": b"bytes"})
    assert private_material_paths({"cycle": cycle})
    assert private_material_paths({"set": {1, 2}})
    with pytest.raises(PrivateMaterialLeakError, match="canonical public JSON"):
        canonical_public_json({"raw": b"bytes"})


def test_private_derivations_have_stable_golden_vectors() -> None:
    schedule = PrivateQ1SeedSchedule(bytes(range(32)))

    assert schedule.salt_commitment_sha256 == ("630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd")
    assert schedule.theta_reference(0, 0).semantic_key_sha256 == (
        "7d110bba311bfb73007410584a73e0ef463de0c088ac3e754a80ba4b6f77a3fa"
    )
    assert schedule.pulse_observation(0, 0, "weak", ActuatorRegime.DIRECT) == -1
    assert schedule.nuisance_observation(0, 0) == 3
    assert schedule.terminal_success(0, 0, TerminalAction.DIRECT, ActuatorRegime.DIRECT)
    assert derive_public_uniform_selection(0, 0).action_id == "strong"
