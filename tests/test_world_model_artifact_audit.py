from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import bench.world_model_lifecycle.artifact_audit as artifact_audit_module
from bench.world_model_lifecycle import binding as binding_module
from bench.world_model_lifecycle.artifact_audit import (
    _GRAPH_RECORD_FIELDS,
    ArtifactAuditError,
    PredictionRecomputation,
    _Audit,
    _audit_analytic_transition_dynamics,
    _audit_bound_irrelevant_control,
    _audit_bound_source_snapshot,
    _audit_formal_runtime_binding,
    _audit_formal_schedule,
    _audit_irrelevant_prediction_manipulation,
    _audit_prediction_coverage,
    _audit_predictions,
    _audit_recomputed_analysis,
    _audit_rejected_probe_full_state,
    _audit_report,
    _audit_restart_parity_evidence,
    _audit_retained_replay_components,
    _binary64_mixture_pit_is_covered,
    _decode_sealed_model,
    _derive_seed,
    _EvaluatedCheckpoint,
    _expected_formal_oscillator_conformance,
    _expected_prediction_target_f32,
    _expected_prediction_targets_f32,
    _independent_coverage_count_gate_checks,
    _independent_oscillator_reset,
    _independent_oscillator_step,
    _independent_recompute_aggregate_metrics,
    _independent_recompute_gate_results,
    _read_stable_regular_file,
    _reconstruct_optimizer_sampling,
    _replay_cem_action_trace,
    _validate_domain_graph_structure,
    _validate_formal_conformance_report,
    _validate_formal_coverage_conformance_report,
    _verify_producer_manifest_locally,
    audit_artifact,
    decode_sampling_manifest,
    recompute_prediction_evidence,
)
from bench.world_model_lifecycle.checkpoint import (
    CANONICAL_COMPONENT_IDS,
    ComponentPayload,
    save_checkpoint,
)
from bench.world_model_lifecycle.learning import WorldModelRuntime
from bench.world_model_lifecycle.model import (
    FixedScaling,
    ProbabilisticEnsemble,
    TransitionBatch,
    WorldModelConfig,
    encode_prediction_evidence,
    evaluate_mixture,
    prepare_candidate,
)
from bench.world_model_lifecycle.planning import (
    CEMController,
    make_learned_model_env,
    make_true_dynamics_env,
    run_pendulum_conformance,
)
from prospect.domain import TimePoint

MAGIC = b"PROSPECT-WM001\0"
TASK_A = "pendulum_normal_torque"
TASK_IRRELEVANT = "independent_phase_oscillator"


@pytest.fixture(autouse=True)
def _isolated_outer_completion_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.parent / f".outer-completions-{tmp_path.name}"
    root.mkdir()
    monkeypatch.setattr(
        artifact_audit_module,
        "_OUTER_COMPLETIONS_ROOT",
        root,
    )


def test_complete_development_audit_passes_without_becoming_claim_complete(
    tmp_path: Path,
) -> None:
    audit = _Audit()
    audit.passed_checks = 1

    report = _audit_report(
        audit,
        root=tmp_path,
        result_path=tmp_path / "result.json",
        result_sha256="0" * 64,
        lane="development",
        require_claim_completeness=True,
    )

    assert report["integrity_passed"] is True
    assert report["engineering_complete"] is True
    assert report["complete_for_claim"] is False
    assert report["passed"] is True


def test_preformal_runtime_seal_requires_exact_negative_assurance() -> None:
    source = {
        "git_commit": "a" * 40,
        "git_tree": "b" * 40,
        "execution_source_sha256": {
            "producer_bootstrap.py": "c" * 64,
        },
    }
    dependencies = {
        "python_executable": "/isolated/bin/python",
        "python_executable_sha256": "d" * 64,
        "standard_library": {"inventory_sha256": "e" * 64},
        "package_roots": [{"inventory_sha256": "f" * 64}],
        "package_ownership": {"inventory_sha256": "0" * 64},
    }
    runtime = {
        "python_flags": {
            "isolated": 1,
            "no_site": 1,
            "no_user_site": 1,
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "safe_path": True,
        },
        "process_environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LAZY_LEGACY_OP": "False",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
            "SDL_AUDIODRIVER": "dsp",
            "TZ": "UTC",
        },
    }
    seal = {
        "schema": "prospect.wm001.runtime-seal.v1",
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "assurance": dict(artifact_audit_module._ASSURANCE),
        "git_commit": source["git_commit"],
        "git_tree": source["git_tree"],
        "worktree_clean": True,
        "python": {
            "executable": dependencies["python_executable"],
            "resolved_executable": "/isolated/bin/python3.12",
            "sha256": dependencies["python_executable_sha256"],
            "version": [
                artifact_audit_module.sys.version_info.major,
                artifact_audit_module.sys.version_info.minor,
                artifact_audit_module.sys.version_info.micro,
            ],
        },
        "required_flags": runtime["python_flags"],
        "process_environment": runtime["process_environment"],
        "bootstrap_source_sha256": source["execution_source_sha256"][
            "producer_bootstrap.py"
        ],
        "standard_library": dependencies["standard_library"],
        "package_roots": dependencies["package_roots"],
        "package_ownership": dependencies["package_ownership"],
    }

    assert (
        artifact_audit_module._preformal_runtime_seal(
            seal,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )
        == seal
    )

    missing = dict(seal)
    del missing["assurance"]
    with pytest.raises(ArtifactAuditError, match="runtime seal differs"):
        artifact_audit_module._preformal_runtime_seal(
            missing,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )

    overstated = dict(seal)
    overstated["assurance"] = {
        **artifact_audit_module._ASSURANCE,
        "tamper_resistant": True,
    }
    with pytest.raises(ArtifactAuditError, match="runtime seal differs"):
        artifact_audit_module._preformal_runtime_seal(
            overstated,
            source=source,
            dependencies=dependencies,
            runtime=runtime,
        )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sampling_payload(indices: np.ndarray, transition_ids: list[str]) -> bytes:
    raw_indices = indices.astype("<u4", copy=False).tobytes(order="C")
    identity_payload = json.dumps(
        transition_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    header = _canonical(
        {
            "dtype": "uint32-le",
            "format": "prospect.wm001.bootstrap-manifest.v1",
            "payload_sha256": hashlib.sha256(raw_indices).hexdigest(),
            "shape": list(indices.shape),
            "transition_ids_sha256": hashlib.sha256(identity_payload).hexdigest(),
        }
    )
    return MAGIC + struct.pack(">Q", len(header)) + header + raw_indices


def test_independent_prediction_decoder_recomputes_simple_gaussian_metrics() -> None:
    transition_ids = ("t0", "t1", "t2")
    targets = torch.zeros((3, 4), dtype=torch.float32)
    means = torch.zeros((5, 3, 4), dtype=torch.float32)
    log_variances = torch.zeros_like(means)
    payload = encode_prediction_evidence(
        transition_ids,
        targets,
        means,
        log_variances,
    )

    recomputed = recompute_prediction_evidence(payload)

    assert recomputed.transition_ids == transition_ids
    assert recomputed.mixture_nll_nats_per_target_dimension == pytest.approx(0.5 * math.log(2.0 * math.pi))
    assert recomputed.normalized_rmse == 0.0
    assert recomputed.interval_90_coverage == 1.0
    assert recomputed.covered_target_count == 12
    assert recomputed.coverage_target_count == 12

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ArtifactAuditError, match="SHA-256"):
        recompute_prediction_evidence(bytes(corrupted))


def test_independent_oscillator_replays_reset_dynamics_and_ignored_action() -> None:
    seed = 123_456
    source = "prospect:IndependentPhaseOscillator-v1"
    digest = hashlib.sha256(f"{source}:{seed}".encode("ascii")).digest()
    phase = (2.0 * int.from_bytes(digest[:8], "big") / float(1 << 64) - 1.0) * math.pi
    velocity = 0.5 + int.from_bytes(digest[8:16], "big") / float(1 << 64)
    expected_reset = np.asarray(
        [math.cos(phase), math.sin(phase), velocity],
        dtype=np.float64,
    )
    reset = _independent_oscillator_reset(seed)
    assert np.array_equal(reset, expected_reset)

    next_phase = math.remainder(phase + 0.05 * velocity, 2.0 * math.pi)
    expected_next = np.asarray(
        [math.cos(next_phase), math.sin(next_phase), velocity],
        dtype=np.float64,
    )
    actual_next, reward, applied = _independent_oscillator_step(reset.tolist())
    assert np.allclose(actual_next, expected_next, rtol=0.0, atol=1e-15)
    assert reward == pytest.approx(math.cos(next_phase), abs=1e-15)
    assert applied == 0.0

    row = {
        "task_id": TASK_IRRELEVANT,
        "task_context": 2.0,
        "pre_observation": reset.tolist(),
        "next_observation": actual_next.tolist(),
        "intended_action": 1.75,
        "applied_action": 0.0,
        "reward": reward,
    }
    audit = _Audit()
    _audit_analytic_transition_dynamics(
        audit,
        row,
        replicate_id="oscillator-test",
        transition_id="transition:0",
    )
    assert audit.failed_checks == 0

    row["applied_action"] = 1.75
    tampered = _Audit()
    _audit_analytic_transition_dynamics(
        tampered,
        row,
        replicate_id="oscillator-test",
        transition_id="transition:0",
    )
    assert "transition_dynamics_applied_action_mismatch" in {finding["code"] for finding in tampered.findings}


def test_v140_oscillator_manipulation_crossbinds_pair_targets_and_isolation() -> None:
    transition_id = "transition:oscillator-heldout"
    before = _independent_oscillator_reset(123_456)
    after, reward, _ = _independent_oscillator_step(before.tolist())
    target = np.asarray(
        [
            [
                (after[0] - before[0]) / 2.0,
                (after[1] - before[1]) / 2.0,
                (after[2] - before[2]) / 16.0,
                reward / 16.2736044,
            ]
        ],
        dtype=np.float32,
    )

    def prediction(normalized_targets: np.ndarray) -> PredictionRecomputation:
        means = np.zeros((5, 1, 4), dtype=np.float32)
        return PredictionRecomputation(
            transition_ids=(transition_id,),
            normalized_targets=normalized_targets,
            member_means=means,
            member_log_variances=means,
            mixture_nll_nats_per_target_dimension=1.0,
            normalized_rmse=1.0,
            interval_90_coverage=0.9,
            covered_target_count=4,
            coverage_target_count=4,
        )

    split = "predictive_validation_irrelevant"
    cold_key = (TASK_IRRELEVANT, split, "cold")
    irrelevant_key = (TASK_IRRELEVANT, split, "irrelevant")
    replicate: dict[str, object] = {
        "episodes": [
            {
                "task_id": TASK_IRRELEVANT,
                "split": split,
                "transition_ids": [transition_id],
            }
        ],
        "predictive_metrics": [
            {
                "task_id": TASK_IRRELEVANT,
                "split": split,
                "condition": "cold",
                "checkpoint_id": "cold",
            },
            {
                "task_id": TASK_IRRELEVANT,
                "split": split,
                "condition": "irrelevant",
                "checkpoint_id": "irrelevant",
            },
        ],
        "updates": [
            {
                "phase": "train_a_irrelevant",
                "eligible_transition_ids": ["transition:oscillator-training"],
            }
        ],
    }
    transitions = {
        transition_id: {
            "transition_id": transition_id,
            "task_id": TASK_IRRELEVANT,
            "split": split,
            "pre_observation": before.tolist(),
        }
    }
    verified = {
        cold_key: prediction(target.copy()),
        irrelevant_key: prediction(target.copy()),
    }

    valid = _Audit()
    _audit_irrelevant_prediction_manipulation(
        valid,
        replicate,
        replicate_id="v140-test",
        transitions_by_id=transitions,
        verified=verified,
    )
    assert valid.failed_checks == 0

    altered = target.copy()
    altered[0, 0] += np.float32(0.25)
    target_tampered = _Audit()
    _audit_irrelevant_prediction_manipulation(
        target_tampered,
        replicate,
        replicate_id="v140-test",
        transitions_by_id=transitions,
        verified={
            cold_key: prediction(target.copy()),
            irrelevant_key: prediction(altered),
        },
    )
    assert {
        "irrelevant_prediction_pair_binding_mismatch",
        "irrelevant_prediction_analytic_target_mismatch",
    }.issubset({finding["code"] for finding in target_tampered.findings})

    contaminated = json.loads(json.dumps(replicate))
    contaminated["updates"][0]["eligible_transition_ids"].append(transition_id)
    isolation_tampered = _Audit()
    _audit_irrelevant_prediction_manipulation(
        isolation_tampered,
        contaminated,
        replicate_id="v140-test",
        transitions_by_id=transitions,
        verified=verified,
    )
    assert "irrelevant_validation_training_contamination" in {
        finding["code"] for finding in isolation_tampered.findings
    }


def test_v140_prediction_audit_reopens_both_oscillator_sidecars(
    tmp_path: Path,
) -> None:
    runtime = WorldModelRuntime.initialize(initialization_seed=7)
    transition_id = "transition:oscillator-heldout"
    before = _independent_oscillator_reset(123_456)
    after, reward, _ = _independent_oscillator_step(before.tolist())
    batch = TransitionBatch.from_arrays(
        transition_ids=[transition_id],
        observations=np.asarray([before], dtype=np.float32),
        contexts=[2.0],
        actions=[1.25],
        next_observations=np.asarray([after], dtype=np.float32),
        rewards=[reward],
    )
    produced = evaluate_mixture(runtime.model, batch)
    decoded = recompute_prediction_evidence(produced.prediction_payload)
    split = "predictive_validation_irrelevant"
    transition = {
        "transition_id": transition_id,
        "task_id": TASK_IRRELEVANT,
        "task_context": 2.0,
        "split": split,
        "pre_observation": before.tolist(),
        "intended_action": 1.25,
        "next_observation": after.tolist(),
        "reward": reward,
        "scaled_target": decoded.normalized_targets[0].tolist(),
    }
    checkpoint = _EvaluatedCheckpoint(
        condition="cold",
        model_version=runtime.version,
        parameter_sha256=runtime.digest,
        live_state_sha256=runtime.live_state_digest,
        model_tensors=_decode_sealed_model(runtime.model_bytes),
    )
    checkpoints = {
        "cold": checkpoint,
        "irrelevant": _EvaluatedCheckpoint(
            condition="irrelevant",
            model_version=checkpoint.model_version,
            parameter_sha256=checkpoint.parameter_sha256,
            live_state_sha256=checkpoint.live_state_sha256,
            model_tensors=checkpoint.model_tensors,
        ),
    }

    def metric_row(
        condition: str,
        payload: bytes,
    ) -> dict[str, object]:
        evidence = recompute_prediction_evidence(payload)
        path = tmp_path / f"{condition}-oscillator-predictions.bin"
        path.write_bytes(payload)
        return {
            "task_id": TASK_IRRELEVANT,
            "condition": condition,
            "checkpoint_id": condition,
            "model_version": checkpoint.model_version,
            "parameter_sha256": checkpoint.parameter_sha256,
            "live_state_sha256": checkpoint.live_state_sha256,
            "split": split,
            "transition_count": 1,
            "mixture_nll_nats_per_target_dimension": (evidence.mixture_nll_nats_per_target_dimension),
            "normalized_rmse": evidence.normalized_rmse,
            "coverage_semantics": "wm001-mixture-pit-binary64-count-v1",
            "interval_90_covered_target_count": evidence.covered_target_count,
            "coverage_target_count": evidence.coverage_target_count,
            "interval_90_coverage": evidence.interval_90_coverage,
            "prediction_rows_sha256": hashlib.sha256(payload).hexdigest(),
            "prediction_evidence_file": path.name,
            "prediction_evidence_bytes": len(payload),
        }

    replicate: dict[str, object] = {
        "episodes": [
            {
                "task_id": TASK_IRRELEVANT,
                "split": split,
                "transition_ids": [transition_id],
            }
        ],
        "predictive_metrics": [
            metric_row("cold", produced.prediction_payload),
            metric_row("irrelevant", produced.prediction_payload),
        ],
        "updates": [
            {
                "phase": "train_a_irrelevant",
                "eligible_transition_ids": ["transition:oscillator-training"],
            }
        ],
    }
    valid = _Audit()
    _audit_predictions(
        valid,
        tmp_path,
        replicate,
        replicate_id="v140-test",
        device="cpu",
        transitions_by_id={transition_id: transition},
        evaluated_checkpoints=checkpoints,
    )
    assert valid.failed_checks == 0

    altered_targets = decoded.normalized_targets.copy()
    altered_targets[0, 0] = np.nextafter(
        altered_targets[0, 0],
        np.float32(math.inf),
        dtype=np.float32,
    )
    one_ulp_tamper = encode_prediction_evidence(
        decoded.transition_ids,
        torch.from_numpy(altered_targets),
        torch.from_numpy(decoded.member_means),
        torch.from_numpy(decoded.member_log_variances),
    )
    predictive_rows = replicate["predictive_metrics"]
    assert isinstance(predictive_rows, list)
    predictive_rows[1] = metric_row("irrelevant", one_ulp_tamper)
    one_ulp = _Audit()
    _audit_predictions(
        one_ulp,
        tmp_path,
        replicate,
        replicate_id="v140-test",
        device="cpu",
        transitions_by_id={transition_id: transition},
        evaluated_checkpoints=checkpoints,
    )
    assert "prediction_target_binding_mismatch" in {finding["code"] for finding in one_ulp.findings}

    altered_targets = decoded.normalized_targets.copy()
    altered_targets[0, 0] += np.float32(0.25)
    rehashed_tamper = encode_prediction_evidence(
        decoded.transition_ids,
        torch.from_numpy(altered_targets),
        torch.from_numpy(decoded.member_means),
        torch.from_numpy(decoded.member_log_variances),
    )
    predictive_rows[1] = metric_row("irrelevant", rehashed_tamper)
    tampered = _Audit()
    _audit_predictions(
        tampered,
        tmp_path,
        replicate,
        replicate_id="v140-test",
        device="cpu",
        transitions_by_id={transition_id: transition},
        evaluated_checkpoints=checkpoints,
    )
    assert {
        "prediction_target_binding_mismatch",
        "irrelevant_prediction_pair_binding_mismatch",
        "irrelevant_prediction_analytic_target_mismatch",
    }.issubset({finding["code"] for finding in tampered.findings})


def test_formal_irrelevant_control_audit_recomputes_source_and_report(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source" / "bench" / "world_model_lifecycle" / "runtime_lane.py"
    source_path.parent.mkdir(parents=True)
    source_payload = b"independently bound oscillator source\n"
    source_path.write_bytes(source_payload)
    report = _expected_formal_oscillator_conformance()
    report_payload = _canonical(report) + b"\n"
    report_path = tmp_path / "oscillator-conformance.json"
    report_path.write_bytes(report_payload)
    block: dict[str, object] = {
        "id": TASK_IRRELEVANT,
        "source_id": "prospect:IndependentPhaseOscillator-v1",
        "source_sha256": hashlib.sha256(source_payload).hexdigest(),
        "conformance_report_file": report_path.name,
        "conformance_report_bytes": len(report_payload),
        "conformance_report_sha256": hashlib.sha256(report_payload).hexdigest(),
    }

    valid = _Audit()
    _audit_bound_irrelevant_control(valid, tmp_path, block)
    assert valid.failed_checks == 0

    source_tampered = _Audit()
    _audit_bound_irrelevant_control(
        source_tampered,
        tmp_path,
        {**block, "source_sha256": "0" * 64},
    )
    assert "formal_irrelevant_control_source_mismatch" in {finding["code"] for finding in source_tampered.findings}

    semantically_tampered = dict(report)
    semantically_tampered["trajectory_sha256"] = "0" * 64
    body = dict(semantically_tampered)
    body.pop("report_sha256")
    semantically_tampered["report_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    tampered_payload = _canonical(semantically_tampered) + b"\n"
    report_path.write_bytes(tampered_payload)
    report_tampered = _Audit()
    _audit_bound_irrelevant_control(
        report_tampered,
        tmp_path,
        {
            **block,
            "conformance_report_bytes": len(tampered_payload),
            "conformance_report_sha256": hashlib.sha256(tampered_payload).hexdigest(),
        },
    )
    assert "formal_irrelevant_control_verification_failed" in {finding["code"] for finding in report_tampered.findings}


def test_checkpoint_replay_audit_rejects_irrelevant_ids_and_manifests(
    tmp_path: Path,
) -> None:
    collect_a = {
        "transition_id": "transition:a",
        "run_id": "run:a",
        "task_id": TASK_A,
        "episode_id": "episode:a",
        "step_index": 0,
        "split": "collect_a",
        "pre_observation": [1.0, 0.0, 0.0],
        "task_context": 0.0,
        "intended_action": 0.25,
        "next_observation": [0.99, 0.1, 0.05],
        "reward": -0.1,
    }
    collect_b = {
        "transition_id": "transition:b",
        "run_id": "run:b",
        "task_id": "pendulum_reversed_torque",
        "episode_id": "episode:b",
        "step_index": 0,
        "split": "collect_b",
        "pre_observation": [1.0, 0.0, 0.0],
        "task_context": 1.0,
        "intended_action": -0.25,
        "next_observation": [0.99, -0.1, -0.05],
        "reward": -0.2,
    }

    manifest_payloads = {
        "train_a": b"train-a-sampling",
        "train_b_replay": b"train-b-replay-sampling",
    }
    manifest_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    retained_rows: list[dict[str, object]] = []
    for phase, payload in manifest_payloads.items():
        path = tmp_path / f"{phase}.bin"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        manifest_rows.append(
            {
                "phase": phase,
                "media_type": "application/octet-stream",
                "bytes": len(payload),
                "sha256": digest,
                "filename": path.name,
            }
        )
        update_rows.append(
            {
                "phase": phase,
                "status": "committed",
                "sampling_manifest_sha256": digest,
            }
        )
        retained_rows.append(
            {
                "phase": phase,
                "sha256": digest,
                "bytes": len(payload),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
            }
        )

    def dataset(row: dict[str, object]) -> dict[str, object]:
        return {
            "transition_ids": [row["transition_id"]],
            "observations": [row["pre_observation"]],
            "contexts": [row["task_context"]],
            "actions": [row["intended_action"]],
            "next_observations": [row["next_observation"]],
            "rewards": [row["reward"]],
        }

    replay_index: dict[str, object] = {
        "schema": "prospect.wm001.replay-index.v1",
        "canonical_experience_rows": [
            {
                "experience_id": "experience:a",
                "run_id": "run:a",
                "task_id": TASK_A,
                "episode_id": "episode:a",
                "step_index": 0,
                "closed_at": ["wm001", 1],
            },
            {
                "experience_id": "experience:b",
                "run_id": "run:b",
                "task_id": "pendulum_reversed_torque",
                "episode_id": "episode:b",
                "step_index": 0,
                "closed_at": ["wm001", 2],
            },
        ],
        "collect_a": dataset(collect_a),
        "collect_b": dataset(collect_b),
    }
    replay_sampling = {
        "schema": "prospect.wm001.replay-sampling-history.v1",
        "manifests": retained_rows,
    }
    replicate = {
        "transitions": [collect_a, collect_b],
        "updates": update_rows,
        "optimizer_batch_manifests": manifest_rows,
    }

    valid = _Audit()
    _audit_retained_replay_components(
        valid,
        tmp_path,
        replicate,
        replicate_id="replicate",
        replay_index=replay_index,
        replay_sampling_history=replay_sampling,
    )
    assert valid.failed_checks == 0

    contaminated_index = json.loads(json.dumps(replay_index))
    contaminated_index["canonical_experience_rows"].append(
        {
            "experience_id": "experience:irrelevant",
            "run_id": "run:irrelevant",
            "task_id": TASK_IRRELEVANT,
            "episode_id": "episode:irrelevant",
            "step_index": 0,
            "closed_at": ["wm001", 3],
        }
    )
    index_audit = _Audit()
    _audit_retained_replay_components(
        index_audit,
        tmp_path,
        replicate,
        replicate_id="replicate",
        replay_index=contaminated_index,
        replay_sampling_history=replay_sampling,
    )
    assert "checkpoint_replay_index_isolation_mismatch" in {finding["code"] for finding in index_audit.findings}

    contaminated_sampling = json.loads(json.dumps(replay_sampling))
    contaminated_sampling["manifests"].append(
        {
            "phase": "train_a_irrelevant",
            "sha256": "0" * 64,
            "bytes": 1,
            "payload_base64": "AA==",
        }
    )
    sampling_audit = _Audit()
    _audit_retained_replay_components(
        sampling_audit,
        tmp_path,
        replicate,
        replicate_id="replicate",
        replay_index=replay_index,
        replay_sampling_history=contaminated_sampling,
    )
    assert "checkpoint_replay_sampling_isolation_mismatch" in {finding["code"] for finding in sampling_audit.findings}

    heldout = {
        **collect_a,
        "transition_id": "transition:heldout",
        "run_id": "run:heldout",
        "episode_id": "episode:heldout",
        "split": "predictive_validation_a",
    }
    heldout_replicate = {
        **replicate,
        "transitions": [collect_a, collect_b, heldout],
    }
    contaminated_heldout_index = json.loads(json.dumps(replay_index))
    contaminated_heldout_index["collect_a"]["transition_ids"].append(heldout["transition_id"])
    heldout_audit = _Audit()
    _audit_retained_replay_components(
        heldout_audit,
        tmp_path,
        heldout_replicate,
        replicate_id="replicate",
        replay_index=contaminated_heldout_index,
        replay_sampling_history=replay_sampling,
    )
    assert "checkpoint_replay_heldout_contamination" in {finding["code"] for finding in heldout_audit.findings}


def test_auditor_formal_seed_constant_matches_sealed_protocol() -> None:
    protocol = json.loads((artifact_audit_module.HERE / "protocol.json").read_text(encoding="utf-8"))
    protocol_seeds = tuple(protocol["seed_schedule"]["formal_replicate_master_seeds"])

    assert artifact_audit_module._FORMAL_SEEDS == protocol_seeds


def test_formal_runtime_binding_selects_only_bound_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_runtime: dict[str, object] = {
        "platform": "bound-platform",
        "machine": "bound-machine",
        "device": "cuda",
        "accelerator": "bound-gpu",
        "deterministic_algorithms": True,
        "thread_count": 4,
        "interop_thread_count": 4,
        "cuda_runtime": "12.8",
        "cuda_driver": "570.00",
        "cublas_workspace_config": ":4096:8",
    }
    process_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "LAZY_LEGACY_OP": "False",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        "SDL_AUDIODRIVER": "dsp",
        "TZ": "UTC",
    }
    for name, value in process_environment.items():
        if name != "PATH":
            monkeypatch.setenv(name, value)
    runtime = {
        **shared_runtime,
        "python_flags": dict(artifact_audit_module._PREBINDING_PRODUCER_FLAGS),
        "process_environment": process_environment,
    }
    packages = [
        {
            "name": "python",
            "version": "3.13.5",
            "distribution_sha256": "a" * 64,
        }
    ]
    package_root = {
        "path": "/bound/site-packages",
        "semantics_id": "prospect.wm001.package-root.v2",
        "file_count": 7,
        "directory_count": 3,
        "total_bytes": 70,
        "inventory_sha256": "1" * 64,
    }
    standard_library = {
        "path": "/bound/stdlib",
        "semantics_id": "prospect.wm001.standard-library.v2",
        "file_count": 5,
        "directory_count": 2,
        "total_bytes": 50,
        "inventory_sha256": "2" * 64,
    }
    ownership = {
        "semantics_id": "prospect.wm001.package-ownership.v1",
        "root": package_root["path"],
        "file_count": 7,
        "directory_count": 3,
        "shared_file_count": 0,
        "identity_sha256": "3" * 64,
    }
    dependencies = {
        "packages": packages,
        "python_executable": artifact_audit_module.sys.executable,
        "python_executable_sha256": artifact_audit_module._sha256_file(
            Path(artifact_audit_module.sys.executable).resolve()
        ),
        "package_roots": [package_root],
        "package_ownership": ownership,
        "standard_library": standard_library,
    }
    monkeypatch.setattr(
        artifact_audit_module,
        "_live_runtime_identity",
        lambda device: dict(shared_runtime),
    )
    monkeypatch.setattr(
        artifact_audit_module,
        "_live_bound_package_rows",
        lambda rows: rows,
    )
    monkeypatch.setattr(
        artifact_audit_module,
        "_prebinding_live_python_flags",
        lambda: dict(artifact_audit_module._PREBINDING_AUDITOR_FLAGS),
    )

    def inventory(
        identifier: str,
        _raw_path: object,
        *,
        kind: str,
    ) -> dict[str, object]:
        expected = standard_library if kind == "standard_library" else package_root
        return {
            "id": identifier,
            "kind": kind,
            **{key: value for key, value in expected.items() if key != "path"},
        }

    monkeypatch.setattr(
        artifact_audit_module,
        "_prebinding_root_inventory",
        inventory,
    )
    monkeypatch.setattr(
        artifact_audit_module,
        "_live_package_ownership",
        lambda _root: dict(ownership),
    )
    valid = _Audit()
    selected = _audit_formal_runtime_binding(
        valid,
        runtime=runtime,
        dependencies=dependencies,
        execution=dict(runtime),
    )
    assert selected == "cuda"
    assert valid.failed_checks == 0

    mismatched = _Audit()
    selected = _audit_formal_runtime_binding(
        mismatched,
        runtime=runtime,
        dependencies=dependencies,
        execution={
            **runtime,
            "device": "cpu",
        },
    )
    assert selected == "cuda"
    assert "formal_result_runtime_binding_mismatch" in {finding["code"] for finding in mismatched.findings}

    monkeypatch.setattr(
        artifact_audit_module,
        "_live_runtime_identity",
        lambda device: {
            **shared_runtime,
            "cuda_driver": "different-driver",
        },
    )
    live_runtime_mismatch = _Audit()
    _audit_formal_runtime_binding(
        live_runtime_mismatch,
        runtime=runtime,
        dependencies=dependencies,
        execution=dict(runtime),
    )
    assert "formal_auditor_runtime_binding_mismatch" in {finding["code"] for finding in live_runtime_mismatch.findings}

    monkeypatch.setattr(
        artifact_audit_module,
        "_live_runtime_identity",
        lambda device: dict(shared_runtime),
    )
    monkeypatch.setattr(
        artifact_audit_module,
        "_live_bound_package_rows",
        lambda rows: [{**packages[0], "distribution_sha256": "b" * 64}],
    )
    live_dependency_mismatch = _Audit()
    _audit_formal_runtime_binding(
        live_dependency_mismatch,
        runtime=runtime,
        dependencies=dependencies,
        execution=dict(runtime),
    )
    assert "formal_auditor_dependency_binding_mismatch" in {
        finding["code"] for finding in live_dependency_mismatch.findings
    }


def test_formal_schedule_binds_exact_v140_seed_order_and_replicate_sidecars(
    tmp_path: Path,
) -> None:
    protocol = json.loads((artifact_audit_module.HERE / "protocol.json").read_text(encoding="utf-8"))
    seeds = tuple(protocol["seed_schedule"]["formal_replicate_master_seeds"])
    replicates = [
        {
            "replicate_id": f"wm001-formal-{seed}",
            "master_seed": seed,
            "episodes": [],
            "transitions": [],
            "predictive_metrics": [],
            "updates": [
                {
                    "phase": phase,
                    "status": "committed",
                    "optimizer_steps": 2_000,
                }
                for phase in (
                    "train_a",
                    "train_a_irrelevant",
                    "train_a_corrupted",
                    "train_b_replay",
                    "train_b_naive",
                )
            ]
            + [
                {
                    "phase": "rejected_update_probe",
                    "status": "rejected",
                    "optimizer_steps": 0,
                }
            ],
        }
        for seed in seeds
    ]
    for replicate in replicates:
        (tmp_path / f"{replicate['replicate_id']}.json").write_bytes(_canonical(replicate) + b"\n")
    result = {"lane": "formal", "replicates": replicates}
    audit = _Audit()
    _audit_formal_schedule(audit, tmp_path, result)
    codes = {finding["code"] for finding in audit.findings}
    assert "formal_replicate_schedule_mismatch" not in codes
    assert "formal_replicate_sidecar_mismatch" not in codes
    assert "formal_update_budget_mismatch" not in codes

    first = replicates[0]
    (tmp_path / f"{first['replicate_id']}.json").write_bytes(b"{}\n")
    tampered_sidecar = _Audit()
    _audit_formal_schedule(tampered_sidecar, tmp_path, result)
    assert "formal_replicate_sidecar_mismatch" in {finding["code"] for finding in tampered_sidecar.findings}

    reordered = {"lane": "formal", "replicates": list(reversed(replicates))}
    tampered_order = _Audit()
    _audit_formal_schedule(tampered_order, tmp_path, reordered)
    assert "formal_replicate_schedule_mismatch" in {finding["code"] for finding in tampered_order.findings}

    wrong_update_order = json.loads(json.dumps(result))
    first_updates = wrong_update_order["replicates"][0]["updates"]
    first_updates[1], first_updates[2] = first_updates[2], first_updates[1]
    update_order_audit = _Audit()
    _audit_formal_schedule(update_order_audit, tmp_path, wrong_update_order)
    assert "formal_update_budget_mismatch" in {finding["code"] for finding in update_order_audit.findings}


def test_binary64_coverage_endpoint_classifier_is_exact_and_inclusive() -> None:
    lower = 0.05
    upper = 0.95

    assert _binary64_mixture_pit_is_covered(lower)
    assert not _binary64_mixture_pit_is_covered(math.nextafter(lower, -math.inf))
    assert _binary64_mixture_pit_is_covered(upper)
    assert not _binary64_mixture_pit_is_covered(math.nextafter(upper, math.inf))
    with pytest.raises(ArtifactAuditError, match="non-finite"):
        _binary64_mixture_pit_is_covered(math.nan)


def test_prediction_target_binding_rejects_boundary_flipping_one_ulp() -> None:
    target = np.frombuffer(bytes.fromhex("ac3cdebd"), dtype="<f4")[0]
    transition = {
        "pre_observation": [0.0, 0.0, 0.0],
        "next_observation": [
            float(np.multiply(target, np.float32(2.0), dtype=np.float32)),
            0.0,
            0.0,
        ],
        "reward": 0.0,
    }
    expected = _expected_prediction_target_f32(transition)
    batch = TransitionBatch.from_arrays(
        transition_ids=["boundary"],
        observations=[[0.0, 0.0, 0.0]],
        contexts=[0.0],
        actions=[0.0],
        next_observations=[transition["next_observation"]],
        rewards=[0.0],
    )
    _, producer_targets = batch.encoded(FixedScaling())
    assert producer_targets.numpy().astype("<f4", copy=False).tobytes() == expected.tobytes()
    assert expected[0].tobytes().hex() == "ac3cdebd"

    altered = expected.copy()
    altered[0] = np.nextafter(
        altered[0],
        np.float32(math.inf),
        dtype=np.float32,
    )
    assert np.allclose(altered, expected, rtol=2e-6, atol=2e-7)
    assert altered.tobytes() != expected.tobytes()

    means = [
        struct.unpack("<f", bytes.fromhex(value))[0]
        for value in ("8cd85cbb", "f032d7bb", "d0d5aebc", "fcaa09bc", "0086a53a")
    ]
    log_variances = [
        struct.unpack("<f", bytes.fromhex(value))[0]
        for value in ("66b8b3c0", "cb11b5c0", "d611b2c0", "86dcb2c0", "9390b2c0")
    ]

    def pit(value: float) -> float:
        return (
            math.fsum(
                0.5 * (1.0 + math.erf((value - mean) * math.exp(-0.5 * log_variance) / math.sqrt(2.0)))
                for mean, log_variance in zip(
                    means,
                    log_variances,
                    strict=True,
                )
            )
            / 5
        )

    assert not _binary64_mixture_pit_is_covered(pit(float(expected[0])))
    assert _binary64_mixture_pit_is_covered(pit(float(altered[0])))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="formal CUDA target arithmetic requires a CUDA runtime",
)
def test_cuda_prediction_targets_match_bound_device_reconstruction() -> None:
    rng = np.random.default_rng(20260718)
    row_count = 4_096
    before = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, row_count),
            rng.uniform(-1.0, 1.0, row_count),
            rng.uniform(-8.0, 8.0, row_count),
        )
    )
    after = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, row_count),
            rng.uniform(-1.0, 1.0, row_count),
            rng.uniform(-8.0, 8.0, row_count),
        )
    )
    rewards = rng.uniform(-16.2736044, 0.0, row_count)
    boundary = np.frombuffer(bytes.fromhex("ac3cdebd"), dtype="<f4")[0]
    before[0] = (0.0, 0.0, 0.0)
    after[0] = (
        float(np.multiply(boundary, np.float32(2.0), dtype=np.float32)),
        0.0,
        0.0,
    )
    rewards[0] = 0.0
    delta = np.subtract(after, before, dtype=np.float64)
    reconstructed_after = np.add(before, delta, dtype=np.float64)
    batch = TransitionBatch.from_arrays(
        transition_ids=[f"cuda-target:{index}" for index in range(row_count)],
        observations=before,
        contexts=np.zeros(row_count),
        actions=np.zeros(row_count),
        next_observations=reconstructed_after,
        rewards=rewards,
    )
    _, cuda_targets = batch.encoded(FixedScaling(), device="cuda")
    actual = cuda_targets.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    transition_rows = [
        {
            "pre_observation": before[index].tolist(),
            "next_observation": after[index].tolist(),
            "reward": float(rewards[index]),
        }
        for index in range(row_count)
    ]
    expected = (
        _expected_prediction_targets_f32(
            transition_rows,
            device="cuda",
        )
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )
    cpu_bytes = (
        _expected_prediction_targets_f32(
            transition_rows,
            device="cpu",
        )
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )

    assert actual == expected
    assert actual != cpu_bytes


def test_independent_auditor_accepts_bound_coverage_conformance_corpus() -> None:
    report = binding_module.run_coverage_conformance()

    _validate_formal_coverage_conformance_report(report)

    tampered = json.loads(json.dumps(report))
    tampered["cases"][-1]["observed_covered"] = True
    body = dict(tampered)
    body.pop("report_sha256")
    tampered["report_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ArtifactAuditError, match="boundary regression"):
        _validate_formal_coverage_conformance_report(tampered)


def test_independent_coverage_gate_uses_exact_integer_cross_products() -> None:
    rows = [
        {
            "_a_after_a_interval_90_covered_target_count": 4_480.0,
            "_a_after_a_coverage_target_count": 6_400.0,
        }
        for _ in range(8)
    ]
    lower, upper = _independent_coverage_count_gate_checks(rows)
    assert lower["passed"] is True
    assert upper["passed"] is True

    rows[0]["_a_after_a_interval_90_covered_target_count"] -= 1.0
    lower, _ = _independent_coverage_count_gate_checks(rows)
    assert lower["passed"] is False


def test_prediction_coverage_requires_exact_independent_count() -> None:
    target_count = 6_400
    recomputed_count = 5_837
    recomputed = PredictionRecomputation(
        transition_ids=tuple(f"transition:{index}" for index in range(target_count // 4)),
        normalized_targets=np.empty((target_count // 4, 4), dtype=np.float32),
        member_means=np.empty((5, target_count // 4, 4), dtype=np.float32),
        member_log_variances=np.empty((5, target_count // 4, 4), dtype=np.float32),
        mixture_nll_nats_per_target_dimension=0.0,
        normalized_rmse=0.0,
        interval_90_coverage=recomputed_count / target_count,
        covered_target_count=recomputed_count,
        coverage_target_count=target_count,
    )

    exact_row: dict[str, object] = {
        "coverage_semantics": "wm001-mixture-pit-binary64-count-v1",
        "transition_count": target_count // 4,
        "interval_90_covered_target_count": recomputed_count,
        "coverage_target_count": target_count,
        "interval_90_coverage": recomputed_count / target_count,
    }
    exact = _Audit()
    _audit_prediction_coverage(
        exact,
        recomputed,
        exact_row,
        label="exact",
        replicate_id="replicate",
    )
    assert exact.failed_checks == 0

    one_target_row = {
        **exact_row,
        "interval_90_covered_target_count": recomputed_count + 1,
        "interval_90_coverage": (recomputed_count + 1) / target_count,
    }
    one_target = _Audit()
    _audit_prediction_coverage(
        one_target,
        recomputed,
        one_target_row,
        label="one-target",
        replicate_id="replicate",
    )
    assert {finding["code"] for finding in one_target.findings} == {"prediction_coverage_mismatch"}
    evidence = one_target.findings[0]["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["recomputed_covered_target_count"] == recomputed_count
    assert evidence["stored_covered_target_count"] == recomputed_count + 1
    assert evidence["covered_target_count_difference"] == 1


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        (
            {"coverage_semantics": "wrong"},
            "prediction_coverage_semantics_mismatch",
        ),
        (
            {"coverage_target_count": 6_399},
            "prediction_coverage_count_contract_mismatch",
        ),
        (
            {"interval_90_covered_target_count": True},
            "prediction_coverage_count_contract_mismatch",
        ),
        (
            {"interval_90_coverage": math.nextafter(5_837 / 6_400, 1.0)},
            "prediction_coverage_fraction_mismatch",
        ),
    ],
)
def test_prediction_coverage_rejects_inconsistent_contract_fields(
    updates: dict[str, object],
    expected_code: str,
) -> None:
    target_count = 6_400
    recomputed_count = 5_837
    recomputed = PredictionRecomputation(
        transition_ids=("unused",),
        normalized_targets=np.empty((1, 4), dtype=np.float32),
        member_means=np.empty((5, 1, 4), dtype=np.float32),
        member_log_variances=np.empty((5, 1, 4), dtype=np.float32),
        mixture_nll_nats_per_target_dimension=0.0,
        normalized_rmse=0.0,
        interval_90_coverage=recomputed_count / target_count,
        covered_target_count=recomputed_count,
        coverage_target_count=target_count,
    )

    row: dict[str, object] = {
        "coverage_semantics": "wm001-mixture-pit-binary64-count-v1",
        "transition_count": target_count // 4,
        "interval_90_covered_target_count": recomputed_count,
        "coverage_target_count": target_count,
        "interval_90_coverage": recomputed_count / target_count,
    }
    row.update(updates)
    audit = _Audit()
    _audit_prediction_coverage(
        audit,
        recomputed,
        row,
        label="adversarial",
        replicate_id="replicate",
    )

    assert expected_code in {finding["code"] for finding in audit.findings}


def test_independent_sampling_decoder_checks_shape_and_payload_digest() -> None:
    indices = np.arange(5 * 256, dtype="<u4").reshape(1, 5, 256) % 3
    payload = _sampling_payload(indices, ["a", "b", "c"])

    decoded = decode_sampling_manifest(payload)

    assert np.array_equal(decoded.indices, indices)
    assert decoded.payload_sha256 == hashlib.sha256(indices.tobytes(order="C")).hexdigest()

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ArtifactAuditError, match="SHA-256"):
        decode_sampling_manifest(bytes(corrupted))


def test_independent_sampling_replay_matches_balanced_producer_bytes() -> None:
    master_seed = 1_905_245_264
    transition_ids = ("a0", "a1", "b0", "b1")
    transitions = TransitionBatch.from_arrays(
        transition_ids=transition_ids,
        observations=np.tile([1.0, 0.0, 0.0], (4, 1)),
        contexts=[0.0, 0.0, 1.0, 1.0],
        actions=[0.0, 0.0, 0.0, 0.0],
        next_observations=np.tile([1.0, 0.0, 0.0], (4, 1)),
        rewards=[0.0, 0.0, 0.0, 0.0],
    )
    model = ProbabilisticEnsemble(
        WorldModelConfig(hidden_dimensions=(4, 4)),
        initialization_seed=1,
    )
    producer = prepare_candidate(
        model,
        transitions,
        optimizer_steps=2,
        bootstrap_seeds=[_derive_seed("ensemble_bootstrap_b", master_seed, member) for member in range(5)],
        minibatch_order_seed=_derive_seed("minibatch_order_b", master_seed),
        balanced_tasks=True,
    )
    transition_rows = {
        identity: {"task_id": TASK_A if identity.startswith("a") else "pendulum_reversed_torque"}
        for identity in transition_ids
    }

    _, reconstructed = _reconstruct_optimizer_sampling(
        phase="train_b_replay",
        master_seed=master_seed,
        optimizer_steps=2,
        eligible_ids=transition_ids,
        transitions_by_id=transition_rows,
    )

    assert reconstructed == producer.sampling_manifest


def _transition(
    transition_id: str,
    episode_id: str,
    *,
    split: str,
    step: int,
    action: float,
    parameter_sha256: str,
    model_version: str,
    before: list[float],
) -> dict[str, object]:
    theta = math.atan2(before[1], before[0])
    previous_velocity = before[2]
    reward = -(theta * theta + 0.1 * previous_velocity * previous_velocity + 0.001 * action * action)
    angular_velocity = max(
        -8.0,
        min(8.0, previous_velocity + (15.0 * math.sin(theta) + 3.0 * action) * 0.05),
    )
    angle = theta + angular_velocity * 0.05
    after = [math.cos(angle), math.sin(angle), angular_velocity]
    scaled_target = [
        (after[0] - before[0]) / 2.0,
        (after[1] - before[1]) / 2.0,
        (after[2] - before[2]) / 16.0,
        reward / 16.2736044,
    ]
    return {
        "transition_id": transition_id,
        "run_id": f"run:{split}",
        "episode_id": episode_id,
        "task_id": TASK_A,
        "task_context": 0.0,
        "split": split,
        "step_index": step,
        "real_or_imagined": "real",
        "pre_observation_id": f"{transition_id}:pre",
        "decision_id": f"{transition_id}:decision",
        "executed_action_id": f"{transition_id}:action",
        "next_observation_id": f"{transition_id}:next",
        "model_version_at_action": model_version,
        "parameter_sha256_at_action": parameter_sha256,
        "pre_observation": before,
        "intended_action": action,
        "applied_action": action,
        "next_observation": after,
        "reward": reward,
        "terminated": False,
        "truncated": step == 199,
        "scaled_target": scaled_target,
        "target_sha256": hashlib.sha256(struct.pack("<4d", *scaled_target)).hexdigest(),
    }


def _episode(
    episode_id: str,
    *,
    split: str,
    transition_ids: list[str],
    actions: list[float],
    rewards: list[float],
    parameter_sha256: str,
    model_version: str,
    reset_seed: int,
    task_id: str = TASK_A,
    applied_actions: list[float] | None = None,
) -> dict[str, object]:
    run_id = f"run:{split}"
    realized = actions if applied_actions is None else applied_actions
    return {
        "episode_id": episode_id,
        "run_id": run_id,
        "task_id": task_id,
        "split": split,
        "condition": "collection_random" if split.startswith("collect_") else "validation_random",
        "checkpoint_id": "cold",
        "reset_seed": reset_seed,
        "process_id": 1,
        "model_version": model_version,
        "parameter_sha256": parameter_sha256,
        "learning_allowed": split in {"collect_a", "collect_irrelevant"},
        "replay_writes_allowed": split in {"collect_a", "collect_irrelevant"},
        "environment_steps": len(transition_ids),
        "return": math.fsum(rewards),
        "started_at_utc": "2026-07-17T00:00:00Z",
        "completed_at_utc": "2026-07-17T00:01:00Z",
        "action_trace_sha256": hashlib.sha256(_canonical({"intended": actions, "applied": realized})).hexdigest(),
        "transition_ids": transition_ids,
    }


def _write_minimal_auditable_artifact(root: Path) -> Path:
    runtime = WorldModelRuntime.initialize(initialization_seed=7)
    owned_state = runtime.owner.snapshot_state()
    parameter_sha256 = runtime.digest
    model_version = runtime.version
    master_seed = 2999896578

    def seed(namespace: str, index: int = 0) -> int:
        return int.from_bytes(
            hashlib.sha256(f"WM-001|1.6.0|{namespace}|{master_seed}|{index}".encode()).digest()[:4],
            "big",
        )

    collect_ids = [f"collect:{index}" for index in range(200)]
    irrelevant_ids = [f"irrelevant:{index}" for index in range(200)]
    validation_ids = [f"validation:{index}" for index in range(200)]
    collect_seed = seed("collection_action")
    irrelevant_seed = seed("irrelevant_collection_action")
    validation_seed = seed("predictive_validation_action")
    collect_reset_seed = seed("collect_a_episode")
    irrelevant_reset_seed = seed("collect_irrelevant_episode")
    validation_reset_seed = seed("predictive_validation_a_episode")
    collect_rng = np.random.default_rng(collect_seed)
    irrelevant_rng = np.random.default_rng(irrelevant_seed)
    validation_rng = np.random.default_rng(validation_seed)
    collect_actions = [float(collect_rng.uniform(-2.0, 2.0)) for _ in collect_ids]
    irrelevant_actions = [float(irrelevant_rng.uniform(-2.0, 2.0)) for _ in irrelevant_ids]
    validation_actions = [float(validation_rng.uniform(-2.0, 2.0)) for _ in validation_ids]

    def chained_transitions(
        identities: list[str],
        episode_id: str,
        split: str,
        actions: list[float],
        reset_seed: int,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        reset_rng = np.random.default_rng(reset_seed)
        theta, angular_velocity = reset_rng.uniform(
            low=np.asarray([-math.pi, -1.0]),
            high=np.asarray([math.pi, 1.0]),
        )
        before = np.asarray(
            [math.cos(theta), math.sin(theta), angular_velocity],
            dtype=np.float32,
        ).tolist()
        for index, (identity, action) in enumerate(zip(identities, actions, strict=True)):
            transition = _transition(
                identity,
                episode_id,
                split=split,
                step=index,
                action=action,
                parameter_sha256=parameter_sha256,
                model_version=model_version,
                before=before,
            )
            result.append(transition)
            before = list(transition["next_observation"])
        return result

    collect_transitions = chained_transitions(
        collect_ids,
        "episode:collect",
        "collect_a",
        collect_actions,
        collect_reset_seed,
    )
    oscillator_source = "prospect:IndependentPhaseOscillator-v1"
    oscillator_digest = hashlib.sha256(f"{oscillator_source}:{irrelevant_reset_seed}".encode("ascii")).digest()
    phase_unit = int.from_bytes(oscillator_digest[:8], "big") / float(1 << 64)
    velocity_unit = int.from_bytes(oscillator_digest[8:16], "big") / float(1 << 64)
    oscillator_phase = (2.0 * phase_unit - 1.0) * math.pi
    oscillator_velocity = 0.5 + velocity_unit
    oscillator_before = [
        math.cos(oscillator_phase),
        math.sin(oscillator_phase),
        oscillator_velocity,
    ]
    irrelevant_transitions: list[dict[str, object]] = []
    for index, (identity, action) in enumerate(zip(irrelevant_ids, irrelevant_actions, strict=True)):
        oscillator_phase = math.remainder(
            oscillator_phase + 0.05 * oscillator_velocity,
            2.0 * math.pi,
        )
        oscillator_after = [
            math.cos(oscillator_phase),
            math.sin(oscillator_phase),
            oscillator_velocity,
        ]
        reward = math.cos(oscillator_phase)
        scaled_target = [
            (oscillator_after[0] - oscillator_before[0]) / 2.0,
            (oscillator_after[1] - oscillator_before[1]) / 2.0,
            0.0,
            reward / 16.2736044,
        ]
        irrelevant_transitions.append(
            {
                "transition_id": identity,
                "run_id": "run:collect_irrelevant",
                "episode_id": "episode:irrelevant",
                "task_id": TASK_IRRELEVANT,
                "task_context": 2.0,
                "split": "collect_irrelevant",
                "step_index": index,
                "real_or_imagined": "real",
                "pre_observation_id": f"{identity}:pre",
                "decision_id": f"{identity}:decision",
                "executed_action_id": f"{identity}:action",
                "next_observation_id": f"{identity}:next",
                "model_version_at_action": model_version,
                "parameter_sha256_at_action": parameter_sha256,
                "pre_observation": oscillator_before,
                "intended_action": action,
                "applied_action": 0.0,
                "next_observation": oscillator_after,
                "reward": reward,
                "terminated": False,
                "truncated": index == 199,
                "scaled_target": scaled_target,
                "target_sha256": hashlib.sha256(struct.pack("<4d", *scaled_target)).hexdigest(),
            }
        )
        oscillator_before = oscillator_after
    validation_transitions = chained_transitions(
        validation_ids,
        "episode:validation",
        "predictive_validation_a",
        validation_actions,
        validation_reset_seed,
    )
    transitions = [*collect_transitions, *irrelevant_transitions, *validation_transitions]
    episodes = [
        _episode(
            "episode:collect",
            split="collect_a",
            transition_ids=collect_ids,
            actions=collect_actions,
            rewards=[float(row["reward"]) for row in collect_transitions],
            parameter_sha256=parameter_sha256,
            model_version=model_version,
            reset_seed=collect_reset_seed,
        ),
        _episode(
            "episode:irrelevant",
            split="collect_irrelevant",
            transition_ids=irrelevant_ids,
            actions=irrelevant_actions,
            applied_actions=[0.0] * len(irrelevant_actions),
            rewards=[float(row["reward"]) for row in irrelevant_transitions],
            parameter_sha256=parameter_sha256,
            model_version=model_version,
            reset_seed=irrelevant_reset_seed,
            task_id=TASK_IRRELEVANT,
        ),
        _episode(
            "episode:validation",
            split="predictive_validation_a",
            transition_ids=validation_ids,
            actions=validation_actions,
            rewards=[float(row["reward"]) for row in validation_transitions],
            parameter_sha256=parameter_sha256,
            model_version=model_version,
            reset_seed=validation_reset_seed,
        ),
    ]

    validation_observations = np.asarray(
        [row["pre_observation"] for row in validation_transitions],
        dtype=np.float32,
    )
    validation_next = np.asarray(
        [row["next_observation"] for row in validation_transitions],
        dtype=np.float32,
    )
    validation_rewards = np.asarray(
        [row["reward"] for row in validation_transitions],
        dtype=np.float32,
    )
    validation_batch = TransitionBatch.from_arrays(
        transition_ids=validation_ids,
        observations=validation_observations,
        contexts=np.zeros((200, 1), dtype=np.float32),
        actions=np.asarray(validation_actions, dtype=np.float32),
        next_observations=validation_next,
        rewards=validation_rewards,
    )
    generated_prediction = evaluate_mixture(runtime.model, validation_batch)
    prediction_payload = generated_prediction.prediction_payload
    prediction_file = root / "predictions.bin"
    prediction_file.write_bytes(prediction_payload)
    prediction = recompute_prediction_evidence(prediction_payload)

    evaluated_checkpoints: list[dict[str, object]] = []
    for condition in (
        "cold",
        "frozen",
        "corrupted",
        "irrelevant",
        "after_a",
        "after_b_replay",
        "after_b_naive",
    ):
        filename = f"{condition}-model-state.bin"
        (root / filename).write_bytes(owned_state.payload)
        evaluated_checkpoints.append(
            {
                "condition": condition,
                "model_version": model_version,
                "parameter_sha256": parameter_sha256,
                "live_state_sha256": owned_state.digest,
                "media_type": "application/vnd.prospect.wm001.owned-model-state",
                "bytes": len(owned_state.payload),
                "sha256": owned_state.digest,
                "filename": filename,
            }
        )

    indices = np.empty((1, 5, 256), dtype="<u4")
    for member in range(5):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed("ensemble_bootstrap_a", member))
        indices[0, member] = (
            torch.randint(
                len(collect_ids),
                (256,),
                generator=generator,
                dtype=torch.long,
            )
            .numpy()
            .astype("<u4", copy=False)
        )
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(seed("minibatch_order_a"))
    order = torch.randperm(1, generator=order_generator).numpy()
    indices = indices[order].copy()
    manifest_payload = _sampling_payload(indices, collect_ids)
    manifest_file = root / "train-a.bin"
    manifest_file.write_bytes(manifest_payload)
    corrupted_manifest_file = root / "train-a-corrupted.bin"
    corrupted_manifest_file.write_bytes(manifest_payload)
    irrelevant_manifest_payload = _sampling_payload(indices, irrelevant_ids)
    irrelevant_manifest_file = root / "train-a-irrelevant.bin"
    irrelevant_manifest_file.write_bytes(irrelevant_manifest_payload)
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(seed("corrupted_target_permutation"))
    permutation_payload = (
        torch.randperm(len(collect_ids), generator=permutation_generator)
        .numpy()
        .astype("<u4", copy=False)
        .tobytes(order="C")
    )
    permutation_file = root / "train-a-corrupted-permutation.bin"
    permutation_file.write_bytes(permutation_payload)
    consumed = hashlib.sha256()
    encoded_ids = [identity.encode() + b"\n" for identity in collect_ids]
    for index in indices.reshape(-1):
        consumed.update(encoded_ids[int(index)])

    checkpoint_path = root / "checkpoint.zip"
    components = {
        component_id: ComponentPayload(
            component_id=component_id,
            logical_version=f"version:{component_id}",
            payload=f"payload:{component_id}".encode(),
        )
        for component_id in CANONICAL_COMPONENT_IDS
    }
    checkpoint = save_checkpoint(
        checkpoint_path,
        checkpoint_id="after_b_replay",
        agent_id="prospect-wm001-agent",
        created_at=TimePoint(1),
        components=components,
    )
    checkpoint_payload = checkpoint_path.read_bytes()
    update = {
        "receipt_id": "receipt:train-a",
        "phase": "train_a",
        "status": "committed",
        "predecessor_parameter_sha256": parameter_sha256,
        "candidate_parameter_sha256": parameter_sha256,
        "committed_parameter_sha256": parameter_sha256,
        "predecessor_model_version": model_version,
        "committed_model_version": model_version,
        "eligible_splits": ["collect_a"],
        "eligible_transition_count": len(collect_ids),
        "eligible_transition_ids": collect_ids,
        "consumed_sample_count": int(indices.size),
        "consumed_multiset_sha256": consumed.hexdigest(),
        "sampling_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "target_permutation_sha256": None,
        "target_permutation_file": None,
        "optimizer_steps": 1,
        "live_state_before_sha256": owned_state.digest,
        "live_state_after_sha256": owned_state.digest,
    }
    corrupted_update = {
        **update,
        "receipt_id": "receipt:train-a-corrupted",
        "phase": "train_a_corrupted",
        "target_permutation_sha256": hashlib.sha256(permutation_payload).hexdigest(),
        "target_permutation_file": {
            "media_type": "application/octet-stream",
            "bytes": len(permutation_payload),
            "sha256": hashlib.sha256(permutation_payload).hexdigest(),
            "filename": permutation_file.name,
        },
    }
    irrelevant_consumed = hashlib.sha256()
    encoded_irrelevant_ids = [identity.encode() + b"\n" for identity in irrelevant_ids]
    for index in indices.reshape(-1):
        irrelevant_consumed.update(encoded_irrelevant_ids[int(index)])
    irrelevant_update = {
        **update,
        "receipt_id": "receipt:train-a-irrelevant",
        "phase": "train_a_irrelevant",
        "eligible_splits": ["collect_irrelevant"],
        "eligible_transition_count": len(irrelevant_ids),
        "eligible_transition_ids": irrelevant_ids,
        "consumed_multiset_sha256": irrelevant_consumed.hexdigest(),
        "sampling_manifest_sha256": hashlib.sha256(irrelevant_manifest_payload).hexdigest(),
    }

    def policy_run(
        *,
        namespace: str,
        controller_seed: int,
        split: str,
        episode_id: str,
        actions: list[float],
        reset_seed: int,
        task_id: str = TASK_A,
        applied_actions: list[float] | None = None,
    ) -> dict[str, object]:
        rng = np.random.default_rng(controller_seed)
        start = hashlib.sha256(_canonical(rng.bit_generator.state)).hexdigest()
        reproduced = [float(rng.uniform(-2.0, 2.0)) for _ in actions]
        assert reproduced == actions
        end = hashlib.sha256(_canonical(rng.bit_generator.state)).hexdigest()
        realized = actions if applied_actions is None else applied_actions
        trace = {
            "episode_ids": [episode_id],
            "intended": actions,
            "applied": realized,
        }
        return {
            "run_id": f"run:{split}",
            "task_id": task_id,
            "split": split,
            "condition": "collection_random" if split.startswith("collect_") else "validation_random",
            "checkpoint_id": "cold",
            "controller_kind": "uniform_random",
            "controller_version": "wm001-uniform-random-v1",
            "seed_namespace": namespace,
            "seed_index": 0,
            "seed": controller_seed,
            "reset_seeds": [reset_seed],
            "episode_ids": [episode_id],
            "rng_start_sha256": start,
            "rng_end_sha256": end,
            "action_count": len(actions),
            "action_trace_sha256": hashlib.sha256(_canonical(trace)).hexdigest(),
            "planner_budget": None,
        }

    seed_counts = {
        "model_initialization": 1,
        "torch_runtime": 1,
        "collection_action": 2,
        "irrelevant_collection_action": 1,
        "predictive_validation_irrelevant_action": 1,
        "predictive_validation_action": 2,
        "random_policy_action": 2,
        "ensemble_bootstrap_a": 5,
        "ensemble_bootstrap_b": 5,
        "minibatch_order_a": 1,
        "minibatch_order_b": 1,
        "corrupted_target_permutation": 1,
        "collect_a_episode": 8,
        "collect_irrelevant_episode": 8,
        "predictive_validation_irrelevant_episode": 8,
        "predictive_validation_a_episode": 8,
        "behavior_evaluation_a_episode": 32,
        "collect_b_episode": 8,
        "predictive_validation_b_episode": 8,
        "behavior_evaluation_b_episode": 32,
        "planner": 1,
    }
    replicate = {
        "replicate_id": f"wm001-development-{master_seed}",
        "master_seed": master_seed,
        "derived_seeds": [
            {
                "namespace": namespace,
                "values": [seed(namespace, index) for index in range(count)],
            }
            for namespace, count in seed_counts.items()
        ],
        "episodes": episodes,
        "transitions": transitions,
        "updates": [update, corrupted_update, irrelevant_update],
        "optimizer_batch_manifests": [
            {
                "phase": "train_a",
                "media_type": "application/octet-stream",
                "bytes": len(manifest_payload),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "filename": manifest_file.name,
            },
            {
                "phase": "train_a_corrupted",
                "media_type": "application/octet-stream",
                "bytes": len(manifest_payload),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "filename": corrupted_manifest_file.name,
            },
            {
                "phase": "train_a_irrelevant",
                "media_type": "application/octet-stream",
                "bytes": len(irrelevant_manifest_payload),
                "sha256": hashlib.sha256(irrelevant_manifest_payload).hexdigest(),
                "filename": irrelevant_manifest_file.name,
            },
        ],
        "predictive_metrics": [
            {
                "task_id": TASK_A,
                "condition": "cold",
                "checkpoint_id": "cold",
                "model_version": model_version,
                "parameter_sha256": parameter_sha256,
                "live_state_sha256": owned_state.digest,
                "split": "predictive_validation_a",
                "transition_count": len(validation_ids),
                "mixture_nll_nats_per_target_dimension": (prediction.mixture_nll_nats_per_target_dimension),
                "normalized_rmse": prediction.normalized_rmse,
                "coverage_semantics": "wm001-mixture-pit-binary64-count-v1",
                "interval_90_covered_target_count": (prediction.covered_target_count),
                "coverage_target_count": prediction.coverage_target_count,
                "interval_90_coverage": prediction.interval_90_coverage,
                "prediction_rows_sha256": hashlib.sha256(prediction_payload).hexdigest(),
                "prediction_evidence_file": prediction_file.name,
                "prediction_evidence_bytes": len(prediction_payload),
            }
        ],
        "policy_runs": [
            policy_run(
                namespace="collection_action",
                controller_seed=collect_seed,
                split="collect_a",
                episode_id="episode:collect",
                actions=collect_actions,
                reset_seed=collect_reset_seed,
            ),
            policy_run(
                namespace="irrelevant_collection_action",
                controller_seed=irrelevant_seed,
                split="collect_irrelevant",
                episode_id="episode:irrelevant",
                actions=irrelevant_actions,
                applied_actions=[0.0] * len(irrelevant_actions),
                reset_seed=irrelevant_reset_seed,
                task_id=TASK_IRRELEVANT,
            ),
            policy_run(
                namespace="predictive_validation_action",
                controller_seed=validation_seed,
                split="predictive_validation_a",
                episode_id="episode:validation",
                actions=validation_actions,
                reset_seed=validation_reset_seed,
            ),
        ],
        "evaluated_checkpoints": evaluated_checkpoints,
        "checkpoint_components": list(checkpoint.component_rows()),
        "checkpoint_archive": {
            "media_type": "application/zip",
            "bytes": len(checkpoint_payload),
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "filename": checkpoint_path.name,
        },
        "restart_parity": {
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
        },
    }
    result: dict[str, Any] = {
        "schema": "prospect.world-model-lifecycle.raw-result.v6",
        "experiment_id": "WM-001",
        "protocol_version": "1.6.0",
        "protocol_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "bench" / "world_model_lifecycle" / "protocol.json").read_bytes()
        ).hexdigest(),
        "lane": "development",
        "execution": {"device": "cpu"},
        "replicates": [replicate],
    }
    result_path = root / "result.json"
    result_path.write_bytes(_canonical(result) + b"\n")
    return result_path


def test_artifact_audit_recomputes_current_evidence_and_detects_metric_tampering(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is True
    assert report["lane"] == "development"
    assert report["engineering_complete"] is False
    assert report["complete_for_claim"] is False
    assert report["passed"] is True
    assert {gap["code"] for gap in report["coverage_gaps"]} == {
        "cem_action_trace_replay_absent",
        "checkpoint_domain_graph_semantics_unverified",
        "checkpoint_replay_semantics_unverified",
        "irrelevant_prediction_manipulation_absent",
        "producer_custody_not_verified",
        "rejected_probe_full_state_unavailable",
        "restart_original_and_restored_trace_unavailable",
    }

    result = json.loads(result_path.read_text())
    result["replicates"][0]["predictive_metrics"][0]["mixture_nll_nats_per_target_dimension"] += 0.5
    result_path.write_bytes(_canonical(result) + b"\n")

    tampered = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert tampered["integrity_passed"] is False
    assert "prediction_nll_mismatch" in {finding["code"] for finding in tampered["findings"]}


def test_restart_audit_reopens_both_traces_and_rejects_derived_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifact_audit_module,
        "_validate_restart_restore_runtime",
        lambda **_kwargs: None,
    )
    master_seed = 70_359_369
    checkpoint_manifest_sha256 = hashlib.sha256(b"checkpoint").hexdigest()
    component_hashes = {
        component_id: hashlib.sha256(component_id.encode()).hexdigest() for component_id in CANONICAL_COMPONENT_IDS
    }
    parameter_sha256 = component_hashes["world_model"]
    model_version = f"wm001-state-sha256:{hashlib.sha256(b'live-state').hexdigest()}"
    receipt_id = "receipt:train-b-replay"
    custody = {
        "experiences": 3200,
        "transitions": 3200,
        "updates": 2,
        "replay_events": 3200,
    }
    post_custody = {
        "experiences": 3600,
        "transitions": 3600,
        "updates": 2,
        "replay_events": 3200,
    }
    prediction = {
        "member_means": [[0.0, 0.0, 0.0] for _ in range(5)],
        "member_variances": [[1.0, 1.0, 1.0] for _ in range(5)],
    }

    def evaluation(process_id: int) -> dict[str, object]:
        tasks = []
        for task_id, namespace in (
            ("pendulum_normal_torque", "behavior_evaluation_a_episode"),
            ("pendulum_reversed_torque", "behavior_evaluation_b_episode"),
        ):
            tasks.append(
                {
                    "task_id": task_id,
                    "reset_seed": _derive_seed(namespace, master_seed, 0),
                    "return": -10.0,
                    "actions": [0.0] * 200,
                    "predictions": [prediction] * 200,
                    "identities": [
                        [
                            f"decision:{task_id}:{index}",
                            f"prediction:{task_id}:{index}",
                            f"experience:{task_id}:{index}",
                            f"transition:{task_id}:{index}",
                        ]
                        for index in range(200)
                    ],
                }
            )
        return {
            "schema": "prospect.wm001.restart-evaluation.v1",
            "process_id": process_id,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "component_hashes": component_hashes,
            "model_version": model_version,
            "parameter_sha256": parameter_sha256,
            "boundary_state": {
                "snapshot_id": "snapshot:retained",
                "agent_id": "prospect-wm001-agent",
                "captured_at": ["interaction", 9604],
                "belief_id": "belief:retained",
                "latest_update_id": receipt_id,
                "configuration_version": "configuration:retained",
                "memory_version": "memory:retained",
                "knowledge_version": "knowledge:retained",
                "model_version": model_version,
                "representation_version": "representation:physical",
                "policy_version": "policy:cem",
                "custody": custody,
            },
            "post_evaluation_custody": post_custody,
            "tasks": tasks,
        }

    def write_evaluation(name: str, value: object) -> dict[str, object]:
        payload = _canonical(value) + b"\n"
        path = tmp_path / name
        path.write_bytes(payload)
        return {
            "media_type": "application/vnd.prospect.wm001.restart-evaluation+json",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": name,
        }

    live_reference = write_evaluation("live.json", evaluation(101))
    restored_reference = write_evaluation("restored.json", evaluation(202))
    parity = {
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "original_process_id": 101,
        "restored_process_id": 202,
        "fresh_process": True,
        "component_hash_mismatches": [],
        "identity_or_lineage_mismatches": 0,
        "prediction_max_abs_difference": 0.0,
        "action_max_abs_difference": 0.0,
        "episode_return_max_abs_difference": 0.0,
        "live_evaluation": live_reference,
        "restored_evaluation": restored_reference,
    }
    replicate = {
        "master_seed": master_seed,
        "updates": [
            {
                "phase": "train_b_replay",
                "status": "committed",
                "receipt_id": receipt_id,
                "committed_parameter_sha256": parameter_sha256,
                "committed_model_version": model_version,
            }
        ],
        "checkpoint_components": [
            {
                "component_id": component_id,
                "sha256": component_hashes[component_id],
            }
            for component_id in CANONICAL_COMPONENT_IDS
        ],
        "restart_parity": parity,
    }
    audit = _Audit()
    _audit_restart_parity_evidence(
        audit,
        tmp_path,
        replicate,
        replicate_id=f"wm001-formal-{master_seed}",
        launcher_process_id=101,
        execution={},
        binding_runtime=None,
        dependencies=None,
        source=None,
    )
    assert audit.failed_checks == 0
    assert audit.coverage_gaps == []

    parity["action_max_abs_difference"] = 0.5
    tampered = _Audit()
    _audit_restart_parity_evidence(
        tampered,
        tmp_path,
        replicate,
        replicate_id=f"wm001-formal-{master_seed}",
        launcher_process_id=101,
        execution={},
        binding_runtime=None,
        dependencies=None,
        source=None,
    )
    assert "restart_parity_recomputation_mismatch" in {finding["code"] for finding in tampered.findings}


def _add_independent_analysis_rows(result_path: Path) -> None:
    result = json.loads(result_path.read_text())
    metrics, replicate_values = _independent_recompute_aggregate_metrics(result)
    result["aggregate_metrics"] = metrics
    result["gate_results"] = _independent_recompute_gate_results(
        metrics,
        replicate_values,
    )
    result_path.write_bytes(_canonical(result) + b"\n")


def _two_replicate_analysis_result() -> dict[str, Any]:
    replicate_returns = (217.45840023805044, 521.5666528679101)
    replicates: list[dict[str, Any]] = []
    for index, after_a_return in enumerate(replicate_returns):
        behavior_returns = {
            "cold": 0.0,
            "after_a": after_a_return,
            "frozen": 0.0,
            "corrupted": 0.0,
            "irrelevant": 0.0,
            "after_b_replay": after_a_return,
            "after_b_naive": 0.0,
            "random": 0.0,
            "oracle": 1000.0,
        }
        replicates.append(
            {
                "replicate_id": f"synthetic-{index}",
                "episodes": [
                    {
                        "task_id": TASK_A,
                        "split": "behavior_evaluation_a",
                        "condition": condition,
                        "return": value,
                    }
                    for condition, value in behavior_returns.items()
                ],
                "predictive_metrics": [
                    {
                        "task_id": TASK_A,
                        "split": "predictive_validation_a",
                        "condition": condition,
                        "checkpoint_id": condition,
                        "mixture_nll_nats_per_target_dimension": nll,
                        "coverage_semantics": ("wm001-mixture-pit-binary64-count-v1"),
                        "interval_90_covered_target_count": 36,
                        "coverage_target_count": 40,
                        "interval_90_coverage": 0.9,
                    }
                    for condition, nll in (
                        ("after_a", 0.0),
                        ("frozen", 1.0),
                        ("corrupted", 1.0),
                        ("irrelevant", 1.0),
                    )
                ]
                + [
                    {
                        "task_id": TASK_IRRELEVANT,
                        "split": "predictive_validation_irrelevant",
                        "condition": condition,
                        "checkpoint_id": condition,
                        "mixture_nll_nats_per_target_dimension": nll,
                        "coverage_semantics": ("wm001-mixture-pit-binary64-count-v1"),
                        "interval_90_covered_target_count": 36,
                        "coverage_target_count": 40,
                        "interval_90_coverage": 0.9,
                    }
                    for condition, nll in (
                        ("cold", 1.0),
                        ("irrelevant", 0.0),
                    )
                ],
                "updates": [],
                "checkpoint_components": [],
            }
        )
    return {"replicates": replicates}


def test_two_replicate_ci_preserves_sealed_operation_order() -> None:
    result = _two_replicate_analysis_result()
    metrics, _ = _independent_recompute_aggregate_metrics(result)
    metric = next(row for row in metrics if row["name"] == "a_return_improvement_after_a_vs_cold")
    values = [217.45840023805044, 521.5666528679101]
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(2)
    margin = 12.706204736 * standard_error

    assert metric["replicate_values"] == values
    assert metric["mean"] == mean
    assert metric["ci_95_lower"] == mean - margin
    assert metric["ci_95_upper"] == mean + margin
    assert metric["ci_95_lower"] == -1562.5183333581233


def test_v140_analysis_recomputes_oscillator_manipulation_metric_and_k3() -> None:
    result = _two_replicate_analysis_result()
    metrics, replicate_values = _independent_recompute_aggregate_metrics(result)
    manipulation = next(
        row for row in metrics if row["name"] == "irrelevant_source_nll_improvement_after_irrelevant_vs_cold"
    )
    assert manipulation["replicate_values"] == [1.0, 1.0]
    assert manipulation["mean"] == 1.0
    assert manipulation["ci_95_lower"] == 1.0

    gates = _independent_recompute_gate_results(metrics, replicate_values)
    k3 = next(row for row in gates if row["gate"] == "K3")
    assert [check["name"] for check in k3["checks"][:2]] == [
        "irrelevant_source_vs_cold_mean_nll_improvement",
        "irrelevant_source_vs_cold_nll_improvement_ci_lower",
    ]
    assert all(check["passed"] for check in k3["checks"][:2])


def test_two_replicate_rehashed_aggregate_and_gate_tampering_is_rejected() -> None:
    result = _two_replicate_analysis_result()
    metrics, replicate_values = _independent_recompute_aggregate_metrics(result)
    result["aggregate_metrics"] = metrics
    result["gate_results"] = _independent_recompute_gate_results(
        metrics,
        replicate_values,
    )
    baseline = _Audit()
    _audit_recomputed_analysis(baseline, result)
    assert baseline.failed_checks == 0
    assert baseline.passed_checks == 2

    tampered = json.loads(json.dumps(result))
    metric = next(row for row in tampered["aggregate_metrics"] if row["name"] == "a_return_improvement_after_a_vs_cold")
    metric["ci_95_lower"] = math.nextafter(
        metric["ci_95_lower"],
        math.inf,
    )
    fabricated_evidence_sha256 = hashlib.sha256(_canonical(metric)).hexdigest()
    k4 = next(row for row in tampered["gate_results"] if row["gate"] == "K4")
    for check in k4["checks"][:2]:
        check["raw_evidence_sha256"] = fabricated_evidence_sha256
    k4["checks"][1]["observed"] = metric["ci_95_lower"]

    audit = _Audit()
    _audit_recomputed_analysis(audit, tampered)

    assert audit.failed_checks == 2
    assert {finding["code"] for finding in audit.findings} == {
        "aggregate_metrics_recomputation_mismatch",
        "gate_results_recomputation_mismatch",
    }


def test_artifact_audit_rejects_fabricated_aggregate_values(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    _add_independent_analysis_rows(result_path)
    baseline = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert baseline["integrity_passed"] is True

    result = json.loads(result_path.read_text())
    result["aggregate_metrics"][0]["mean"] += 1.0
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert report["integrity_passed"] is False
    assert "aggregate_metrics_recomputation_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_fabricated_gate_values(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    _add_independent_analysis_rows(result_path)
    baseline = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert baseline["integrity_passed"] is True

    result = json.loads(result_path.read_text())
    result["gate_results"][0]["checks"][0]["observed"] = 1
    result["gate_results"][0]["checks"][0]["raw_evidence_sha256"] = hashlib.sha256(
        _canonical(["fabricated"])
    ).hexdigest()
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )
    assert report["integrity_passed"] is False
    assert "gate_results_recomputation_mismatch" in {finding["code"] for finding in report["findings"]}


def _write_test_producer_manifest(root: Path) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "producer-manifest.json"
    ]
    manifest: dict[str, object] = {
        "schema": "prospect.wm001.producer-manifest.v1",
        "experiment_id": "WM-001",
        "lane": "development",
        "status": "completed",
        "started_at_utc": "2026-07-17T00:00:00Z",
        "completed_at_utc": "2026-07-17T01:00:00Z",
        "error": None,
        "manifest_excludes": ["producer-manifest.json"],
        "file_count": len(files),
        "files": files,
    }
    manifest_path = root / "producer-manifest.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    completion = artifact_audit_module._OUTER_COMPLETIONS_ROOT / (
        hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest() + ".json"
    )
    completion.hardlink_to(manifest_path)
    return manifest


def test_artifact_audit_reopens_finalized_producer_manifest(tmp_path: Path) -> None:
    _write_minimal_auditable_artifact(tmp_path)
    _write_test_producer_manifest(tmp_path)

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
    )

    assert report["integrity_passed"] is True
    assert report["custody"]["producer_manifest_checked"] is True
    assert report["custody"]["producer_manifest_status"] == "completed"


def test_local_producer_manifest_rejects_duplicate_keys_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    _write_minimal_auditable_artifact(tmp_path)
    manifest = _write_test_producer_manifest(tmp_path)
    manifest_path = tmp_path / "producer-manifest.json"
    canonical = _canonical(manifest)
    duplicate = canonical.replace(
        b'"experiment_id":"WM-001",',
        b'"experiment_id":"WM-001","experiment_id":"WM-001",',
        1,
    )
    manifest_path.write_bytes(duplicate + b"\n")
    with pytest.raises(ArtifactAuditError, match="duplicate object key"):
        _verify_producer_manifest_locally(tmp_path)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ArtifactAuditError, match="canonical JSON"):
        _verify_producer_manifest_locally(tmp_path)

    manifest_path.write_bytes(canonical)
    with pytest.raises(ArtifactAuditError, match="trailing newline"):
        _verify_producer_manifest_locally(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "prospect.wm001.producer-manifest.v0", "schema identity"),
        ("experiment_id", "WM-999", "experiment identity"),
        ("lane", "unknown", "lane identity"),
        ("status", "running", "status"),
        ("file_count", -1, "file_count"),
    ],
)
def test_local_producer_manifest_validates_top_level_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _write_minimal_auditable_artifact(tmp_path)
    manifest = _write_test_producer_manifest(tmp_path)
    manifest[field] = value
    (tmp_path / "producer-manifest.json").write_bytes(_canonical(manifest) + b"\n")

    with pytest.raises(ArtifactAuditError, match=message):
        _verify_producer_manifest_locally(tmp_path)


def test_local_producer_manifest_validates_file_identity_and_exact_set(
    tmp_path: Path,
) -> None:
    _write_minimal_auditable_artifact(tmp_path)
    manifest = _write_test_producer_manifest(tmp_path)
    manifest_path = tmp_path / "producer-manifest.json"
    rows = manifest["files"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    original_bytes = first["bytes"]
    first["bytes"] = int(original_bytes) + 1
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    with pytest.raises(ArtifactAuditError, match="size changed"):
        _verify_producer_manifest_locally(tmp_path)

    first["bytes"] = original_bytes
    original_digest = first["sha256"]
    first["sha256"] = "0" * 64
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    with pytest.raises(ArtifactAuditError, match="digest changed"):
        _verify_producer_manifest_locally(tmp_path)

    first["sha256"] = original_digest
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    (tmp_path / "unmanifested.txt").write_text("not in custody\n")
    with pytest.raises(ArtifactAuditError, match="file set changed"):
        _verify_producer_manifest_locally(tmp_path)


def test_local_producer_manifest_rejects_escapes_and_symlinks(tmp_path: Path) -> None:
    _write_minimal_auditable_artifact(tmp_path)
    manifest = _write_test_producer_manifest(tmp_path)
    manifest_path = tmp_path / "producer-manifest.json"
    rows = manifest["files"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    original_path = first["path"]
    first["path"] = "../outside"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    with pytest.raises(ArtifactAuditError, match="unsafe producer manifest path"):
        _verify_producer_manifest_locally(tmp_path)

    first["path"] = original_path
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    (tmp_path / "alias").symlink_to(tmp_path / str(original_path))
    with pytest.raises(ArtifactAuditError, match="symbolic link"):
        _verify_producer_manifest_locally(tmp_path)


def test_stable_regular_read_is_bounded_and_detects_in_read_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"abc")
    with pytest.raises(ArtifactAuditError, match="byte audit limit"):
        _read_stable_regular_file(
            target,
            2,
            label="test evidence",
            capture_payload=False,
        )

    original_read = artifact_audit_module.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, length: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, length)
        if chunk and not mutated:
            mutated = True
            target.write_bytes(b"xyz")
        return chunk

    monkeypatch.setattr(artifact_audit_module.os, "read", mutate_after_first_read)
    with pytest.raises(ArtifactAuditError, match="changed while it was being read"):
        _read_stable_regular_file(
            target,
            16,
            label="test evidence",
            capture_payload=False,
        )


def test_artifact_audit_rejects_reset_state_not_generated_by_episode_seed(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    first = result["replicates"][0]["transitions"][0]
    first["pre_observation"][0] = float(first["pre_observation"][0]) + 0.01
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "episode_reset_observation_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_reordered_episode_seed_schedule(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    replicate = result["replicates"][0]
    collect_seeds = next(row["values"] for row in replicate["derived_seeds"] if row["namespace"] == "collect_a_episode")
    collect_episode = next(row for row in replicate["episodes"] if row["split"] == "collect_a")
    collect_policy = next(row for row in replicate["policy_runs"] if row["split"] == "collect_a")
    collect_episode["reset_seed"] = collect_seeds[1]
    collect_policy["reset_seeds"] = [collect_seeds[1]]
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "policy_reset_seed_schedule_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_rehashed_but_wrong_optimizer_rng_trace(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    replicate = result["replicates"][0]
    update = next(row for row in replicate["updates"] if row["phase"] == "train_a")
    manifest = next(row for row in replicate["optimizer_batch_manifests"] if row["phase"] == "train_a")
    path = tmp_path / manifest["filename"]
    decoded = decode_sampling_manifest(path.read_bytes())
    altered = decoded.indices.copy()
    altered[0, 0, 0] = (int(altered[0, 0, 0]) + 1) % int(update["eligible_transition_count"])
    payload = _sampling_payload(altered, update["eligible_transition_ids"])
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest["bytes"] = len(payload)
    manifest["sha256"] = digest
    update["sampling_manifest_sha256"] = digest
    consumed = hashlib.sha256()
    encoded_ids = [identity.encode() + b"\n" for identity in update["eligible_transition_ids"]]
    for sample_index in altered.reshape(-1):
        consumed.update(encoded_ids[int(sample_index)])
    update["consumed_multiset_sha256"] = consumed.hexdigest()
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "optimizer_sampling_seed_replay_mismatch" in {finding["code"] for finding in report["findings"]}


def test_artifact_audit_rejects_rehashed_but_wrong_corruption_permutation(
    tmp_path: Path,
) -> None:
    result_path = _write_minimal_auditable_artifact(tmp_path)
    result = json.loads(result_path.read_text())
    update = next(row for row in result["replicates"][0]["updates"] if row["phase"] == "train_a_corrupted")
    reference = update["target_permutation_file"]
    path = tmp_path / reference["filename"]
    permutation = np.frombuffer(path.read_bytes(), dtype="<u4").copy()
    permutation[[0, 1]] = permutation[[1, 0]]
    payload = permutation.astype("<u4", copy=False).tobytes(order="C")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    reference["bytes"] = len(payload)
    reference["sha256"] = digest
    update["target_permutation_sha256"] = digest
    result_path.write_bytes(_canonical(result) + b"\n")

    report = audit_artifact(
        tmp_path,
        validate_schema=False,
        require_claim_completeness=False,
        verify_custody=False,
    )

    assert report["integrity_passed"] is False
    assert "target_permutation_seed_replay_mismatch" in {finding["code"] for finding in report["findings"]}


@pytest.mark.parametrize("oracle", [False, True], ids=["learned", "oracle"])
def test_standalone_cem_replay_matches_producer_actions_exactly(
    oracle: bool,
) -> None:
    runtime = WorldModelRuntime.initialize(initialization_seed=7)
    states = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.9799, -0.3]],
            [[0.99, 0.1, 0.2], [0.3, 0.9539, 0.4]],
        ],
        dtype=np.float32,
    )
    seed = 99173
    planner = CEMController(
        make_true_dynamics_env() if oracle else make_learned_model_env(runtime.model),
        seed=seed,
    )
    start_digest = planner.rng_digest
    expected: list[np.ndarray] = []
    for step_states in states:
        contextual = np.concatenate(
            (step_states, np.zeros((len(step_states), 1), dtype=np.float32)),
            axis=1,
        )
        expected.append(planner.act(contextual).detach().cpu().numpy())
    expected_actions = np.stack(expected)

    replayed, replay_start, replay_end = _replay_cem_action_trace(
        observed_states=states,
        context=0.0,
        seed=seed,
        device="cpu",
        model_tensors=(None if oracle else _decode_sealed_model(runtime.model_bytes)),
    )

    assert np.array_equal(replayed, expected_actions)
    assert replay_start == start_digest
    assert replay_end == planner.rng_digest
    tampered = expected_actions.copy()
    tampered[0, 0, 0] = np.nextafter(
        tampered[0, 0, 0],
        np.float32(math.inf),
    )
    assert not np.array_equal(replayed, tampered)


def _rejected_probe_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    runtime = WorldModelRuntime.initialize(initialization_seed=17)

    def node(index: int, type_name: str) -> dict[str, object]:
        return {
            "ref": f"n{index:08d}",
            "type": type_name,
            "fields": {field: None for field in _GRAPH_RECORD_FIELDS[type_name]},
        }

    graph = {
        "schema": "prospect.wm001.domain-graph.v1",
        "roots": {
            "agent_snapshot": {"$ref": "n00000000"},
            "source_events": {"$tuple": [{"$ref": "n00000001"}]},
            "source_transitions": {"$tuple": [{"$ref": "n00000002"}]},
            "source_updates": {"$tuple": [{"$ref": "n00000003"}]},
            "probe_transitions": {"$tuple": []},
            "probe_updates": {"$tuple": []},
        },
        "nodes": [
            node(0, "AgentSnapshot"),
            node(1, "ExperienceEvent"),
            node(2, "EpistemicTransition"),
            node(3, "UpdateReceipt"),
        ],
        "observation_sequences": [],
    }
    sampling = _sampling_payload(
        np.zeros((1, 5, 256), dtype="<u4"),
        ["t-1"],
    )
    empty_digest = hashlib.sha256(b"fixture").hexdigest()
    train_a_update = {
        "eligible_transition_ids": ["t-1"],
        "consumed_multiset_sha256": empty_digest,
        "predecessor_parameter_sha256": empty_digest,
        "committed_parameter_sha256": runtime.digest,
        "live_state_before_sha256": empty_digest,
        "live_state_after_sha256": runtime.live_state_digest,
        "optimizer_steps": 1,
        "sampling_manifest_sha256": hashlib.sha256(sampling).hexdigest(),
    }
    encoded_marker = base64.b64encode(b"x").decode()
    state = {
        "schema": "prospect.wm001.rejected-probe-full-state.v1",
        "captured_at": ["interaction", 9],
        "model_state": {
            "version": runtime.version,
            "digest": runtime.live_state_digest,
            "payload_base64": base64.b64encode(runtime.owner.snapshot_state().payload).decode(),
        },
        "domain_graph": graph,
        "source_replay_rows": [],
        "probe_replay_rows": [],
        "source_identity_base64": encoded_marker,
        "probe_identity_base64": encoded_marker,
        "collection_rng_state": {"state": 1},
        "process_rng": {
            "python_base64": encoded_marker,
            "numpy_base64": encoded_marker,
            "torch_cpu_base64": encoded_marker,
            "torch_accelerator_base64": encoded_marker,
        },
        "retained_learning_evidence": {
            "phase": "train_a",
            "consumed_transition_ids": ["t-1"],
            "consumed_multiset_sha256": empty_digest,
            "predecessor_parameter_sha256": empty_digest,
            "candidate_parameter_sha256": runtime.digest,
            "predecessor_live_state_sha256": empty_digest,
            "candidate_live_state_sha256": runtime.live_state_digest,
            "optimizer_steps": 1,
            "sampling_manifest_base64": base64.b64encode(sampling).decode(),
            "sampling_manifest_sha256": hashlib.sha256(sampling).hexdigest(),
            "sampled_id_counts": [["t-1", 1280]],
            "target_permutation_sha256": None,
            "target_permutation_base64": None,
            "loss_history": [0.5],
        },
    }
    payload = _canonical(state)
    before_path = root / "probe-before.json"
    after_path = root / "probe-after.json"
    before_path.write_bytes(payload)
    after_path.write_bytes(payload)

    def reference(path: Path) -> dict[str, object]:
        return {
            "media_type": ("application/vnd.prospect.wm001.rejected-probe-state+json"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": path.name,
        }

    rejected_update = {
        "live_state_before_sha256": runtime.live_state_digest,
        "predecessor_model_version": runtime.version,
        "predecessor_parameter_sha256": runtime.digest,
        "full_state_before_sha256": hashlib.sha256(payload).hexdigest(),
        "full_state_after_sha256": hashlib.sha256(payload).hexdigest(),
        "full_state_before_file": reference(before_path),
        "full_state_after_file": reference(after_path),
    }
    return rejected_update, train_a_update


def test_rejected_probe_audit_reopens_and_compares_complete_state(
    tmp_path: Path,
) -> None:
    rejected_update, train_a_update = _rejected_probe_fixture(tmp_path)
    audit = _Audit()

    _audit_rejected_probe_full_state(
        audit,
        tmp_path,
        rejected_update,
        replicate_id="dev-101",
        train_a_update=train_a_update,
    )

    assert audit.failed_checks == 0
    assert audit.coverage_gaps == []

    after_reference = rejected_update["full_state_after_file"]
    after_path = tmp_path / after_reference["filename"]
    after_payload = after_path.read_bytes() + b" "
    after_path.write_bytes(after_payload)
    after_digest = hashlib.sha256(after_payload).hexdigest()
    after_reference["bytes"] = len(after_payload)
    after_reference["sha256"] = after_digest
    rejected_update["full_state_after_sha256"] = after_digest
    tampered = _Audit()

    _audit_rejected_probe_full_state(
        tampered,
        tmp_path,
        rejected_update,
        replicate_id="dev-101",
        train_a_update=train_a_update,
    )

    assert tampered.failed_checks > 0
    assert {finding["code"] for finding in tampered.findings} >= {
        "rejected_probe_full_state_changed",
        "rejected_probe_full_state_invalid",
    }


def test_checkpoint_domain_graph_audit_rejects_unknown_tags_and_external_refs() -> None:
    receipt_fields: dict[str, object] = {field: None for field in _GRAPH_RECORD_FIELDS["UpdateReceipt"]}
    receipt_fields["transitions"] = {
        "$tuple": [
            {"$external": "transition:t-1"},
        ]
    }
    graph = {
        "schema": "prospect.wm001.domain-graph.v1",
        "roots": {
            "receipts": {
                "$tuple": [
                    {"$ref": "n00000000"},
                ]
            }
        },
        "nodes": [
            {
                "ref": "n00000000",
                "type": "UpdateReceipt",
                "fields": receipt_fields,
            }
        ],
        "observation_sequences": [],
    }

    _validate_domain_graph_structure(graph, component_id="update_receipts")

    unknown_tag = json.loads(json.dumps(graph))
    unknown_tag["nodes"][0]["fields"]["transitions"] = {"$pickle": "payload"}
    with pytest.raises(ArtifactAuditError, match="unknown encoded tag"):
        _validate_domain_graph_structure(
            unknown_tag,
            component_id="update_receipts",
        )

    wrong_namespace = json.loads(json.dumps(graph))
    wrong_namespace["nodes"][0]["fields"]["transitions"]["$tuple"][0] = {"$external": "belief:b-1"}
    with pytest.raises(ArtifactAuditError, match="external ref"):
        _validate_domain_graph_structure(
            wrong_namespace,
            component_id="update_receipts",
        )


def _source_row(relative: str, payload: bytes) -> dict[str, object]:
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_formal_source_snapshot_audit_requires_exact_bound_file_set(
    tmp_path: Path,
) -> None:
    makefile_payload = b"all:\n\t@true\n"
    module_payload = b'"""fixture"""\n'
    (tmp_path / "source" / "Makefile").parent.mkdir(parents=True)
    (tmp_path / "source" / "Makefile").write_bytes(makefile_payload)
    module = tmp_path / "source" / "bench" / "fixture.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(module_payload)
    source = {
        "implementation_files": [
            _source_row("Makefile", makefile_payload),
            _source_row("bench/fixture.py", module_payload),
        ]
    }
    audit = _Audit()

    _audit_bound_source_snapshot(audit, tmp_path, source)

    assert audit.passed_checks == 1
    assert audit.failed_checks == 0

    (tmp_path / "source" / "unexpected.py").write_bytes(b"pass\n")
    with pytest.raises(ArtifactAuditError, match="file set differs"):
        _audit_bound_source_snapshot(_Audit(), tmp_path, source)


def test_formal_source_snapshot_audit_rejects_tamper_and_manifest_omission(
    tmp_path: Path,
) -> None:
    makefile_payload = b"all:\n\t@true\n"
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "Makefile").write_bytes(makefile_payload)
    source = {
        "implementation_files": [
            _source_row("Makefile", makefile_payload),
        ]
    }
    (source_root / "Makefile").write_bytes(b"tampered\n")
    with pytest.raises(ArtifactAuditError, match="size/digest changed"):
        _audit_bound_source_snapshot(_Audit(), tmp_path, source)

    (source_root / "Makefile").write_bytes(makefile_payload)
    source["implementation_files"] = []
    with pytest.raises(ArtifactAuditError, match="no implementation manifest"):
        _audit_bound_source_snapshot(_Audit(), tmp_path, source)


def test_artifact_auditor_independently_rejects_rehashed_conformance_bypass() -> None:
    report = run_pendulum_conformance(
        samples_per_task=512,
        seed=20260717,
        observation_atol=2e-6,
        reward_atol=1e-9,
        planner_observation_atol=2e-6,
        planner_reward_atol=2e-5,
    )

    _validate_formal_conformance_report(report)

    report["cases"] = 2
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(ArtifactAuditError, match="exactly 512 cases"):
        _validate_formal_conformance_report(report)
