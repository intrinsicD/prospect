from __future__ import annotations

import ast
import struct
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from bench.multimodal_mechanism_diagnostics import oracle_v22 as oracle

CONFIG_SHA256 = "0" * 64


def _integer_source() -> np.ndarray:
    index = np.arange(3 * 64 * 64, dtype=np.int64).reshape(3, 64, 64)
    return np.ascontiguousarray((((index * 17 + 13) % 257) - 128).astype(np.float64) / 32.0)


def _assert_same_float(left: float, right: float) -> None:
    assert struct.pack("<d", left) == struct.pack("<d", right)


def _assert_same_array(left: np.ndarray | None, right: np.ndarray | None) -> None:
    if left is None or right is None:
        assert left is right
        return
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert left.tobytes(order="C") == right.tobytes(order="C")


def test_oracle_is_standalone_and_reconstructs_frozen_primitives() -> None:
    module_path = Path(oracle.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    allowed_imports = {
        "__future__",
        "dataclasses",
        "hashlib",
        "itertools",
        "math",
        "numpy",
        "re",
        "struct",
        "types",
        "typing",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imported.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "__import__"
    assert imported <= allowed_imports

    oracle.validate_frozen_reconstruction()
    assert oracle.CANDIDATE_ORDER_SHA256 == (
        "dac8a2fcfa35d333f9338f54cd54648ecf0a5a62f96d6d345817b9e2e23d6e79"
    )
    assert oracle.ADMISSIBLE_LIST_SHA256 == (
        "6c7dfa679e7a10f52bcbedbb2bbdbaabd397157d7333350930d193e14876711d"
    )
    assert oracle.INVALID_BITMAP_SHA256 == (
        "cc478d3eba041f34e5153199f9cccf43fd7891672ff26db96e55d15f5e721132"
    )
    assert oracle.GEOMETRY_SHA256 == (
        "759f3f8b0a76984dafd7f93a00fbf755f8d86de0c9f327efaab7a71ea43574d5"
    )
    assert len(oracle.CANONICAL_STATES) == oracle.STATE_COUNT == 15_625
    assert len(oracle.ADMISSIBLE_INDICES) == oracle.ADMISSIBLE_COUNT == 2_809
    assert oracle.state_index((0.0,) * 6) == 0
    assert oracle.state_index((4.0, 0.0, 4.0, 0.0, 0.0, 0.0)) == 679
    assert oracle.state_index((8.0, 8.0, 4.0, 4.0, 4.0, 4.0)) not in frozenset(
        oracle.ADMISSIBLE_INDICES
    )
    assert not oracle.CANONICAL_GRID.flags.writeable
    assert not oracle.FULL_MASK.flags.writeable
    with pytest.raises(ValueError):
        oracle.CANONICAL_GRID.setflags(write=True)


def test_scalar_bilinear_sampler_has_a_hand_derived_two_axis_kat() -> None:
    yy, xx = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    source = np.ascontiguousarray(
        np.stack(
            (
                yy * yy + 2.0 * xx * xx + 3.0 * yy * xx,
                2.0 * yy * yy + 3.0 * xx * xx + 4.0 * yy * xx + 1.0,
                3.0 * yy * yy + 4.0 * xx * xx + 5.0 * yy * xx + 2.0,
            )
        )
    )
    sampled = oracle.sample_scalar(
        source, (0.0, 0.0, 2.0, 0.0, 0.0, 2.0), oracle.FULL_MASK
    )
    site = (31 - 8) * 48 + (32 - 8)
    expected = np.asarray(
        (
            float.fromhex("0x1.75e70fe07a611p+12"),
            float.fromhex("0x1.1803db60ffe26p+13"),
            float.fromhex("0x1.75142ed1c2942p+13"),
        ),
        dtype=np.float64,
    )
    assert sampled[:, site].tobytes(order="C") == expected.tobytes(order="C")


def test_complete_scalar_affine_oracle_matches_production_bit_exactly() -> None:
    # Production imports are confined to the comparison side of this test.  The
    # oracle module above has already been statically shown to have no such edge.
    from bench.multimodal_mechanism_diagnostics import global_v22 as production

    source = _integer_source()
    fit_target = np.ascontiguousarray(source[:, 8:56, 8:56].reshape(3, -1))
    independent = oracle.fit_scalar_oracle(
        source,
        fit_target,
        oracle.FULL_MASK,
        oracle.FULL_MASK,
        "affine",
        context_key="kat/integer/full",
        config_sha256=CONFIG_SHA256,
    )
    actual = production.fit_global(
        source,
        fit_target,
        oracle.FULL_MASK,
        oracle.FULL_MASK,
        "affine",
        context_key="kat/integer/full",
        config_sha256=CONFIG_SHA256,
    )

    assert independent.maximum_candidate_tensor_size == 1
    assert independent.context_key == actual.context_key
    assert independent.arm == actual.arm
    assert (
        independent.source_grid.scope_sha256
        == actual.source_grid.scope_sha256
        == "624947178dcde61c1a8577bf61f40abb498b4f335a0d2780cf93988599b25fd3"
    )
    assert (
        independent.source_grid.partition_sha256
        == actual.source_grid.partition_sha256
        == "f036f69ce14d7a43e3fe856fd1586c20bb0b3a9d44e49233d82c4ae188f55d52"
    )
    assert (
        independent.source_grid.sample_stream_sha256
        == actual.source_grid.sample_stream_sha256
        == "6f1cabd6061be3fccf14d94f7f3e9e85a4a426e0475e062a460a5cb9c6b9ad57"
    )
    assert (
        independent.source_grid.content_sha256
        == actual.source_grid.content_sha256
        == "96897d1e2f3384e8e1126e3717a3d6e81f3adcf374de7d23c2b256c7e5e502ef"
    )
    assert len(independent.source_grid.batch_records) == len(actual.source_grid.batch_records) == 22
    for independent_batch, actual_batch in zip(
        independent.source_grid.batch_records, actual.source_grid.batch_records, strict=True
    ):
        for name in (
            "ordinal",
            "indices",
            "shape",
            "dtype",
            "sample_sha256",
            "batch_sha256",
        ):
            assert getattr(independent_batch, name) == getattr(actual_batch, name)
    assert independent.source_grid.batch_records[-1].shape[0] == 121

    assert (
        independent.objective_cache.scope_sha256
        == actual.objective_cache.scope_sha256
        == "a2ce34406d46677fc5eac6a310a91d93c212d50cefafa5d867016926bc19deef"
    )
    assert (
        independent.objective_cache.content_sha256
        == actual.objective_cache.content_sha256
        == "fdc3feb6a9e3ce9d1dded9e23b6f10a9760fc64f9860f40dbea3be5beabdc6b3"
    )
    _assert_same_array(independent.objective_cache.objectives, actual.objective_cache.objectives)
    _assert_same_array(independent.objective_cache.gains, actual.objective_cache.gains)
    _assert_same_array(independent.objective_cache.biases, actual.objective_cache.biases)
    assert independent.objective_cache.retained_macro_ids == actual.objective_cache.retained_macro_ids

    assert independent.selected.state_index == actual.selected.state_index == 0
    assert independent.selected.admissible_rank == actual.selected.admissible_rank == 0
    _assert_same_array(independent.selected.parameters, actual.selected.parameters)
    _assert_same_float(independent.selected.objective, actual.selected.objective)
    _assert_same_array(independent.selected.gains, actual.selected.gains)
    _assert_same_array(independent.selected.biases, actual.selected.biases)
    assert independent.selected.retained_macro_ids == actual.selected.retained_macro_ids
    _assert_same_array(independent.selected.fit_prediction, actual.selected.fit_prediction)
    assert (
        independent.selected.evaluation_sha256
        == actual.selected.evaluation_sha256
        == "fb5010e749a2b32bbd11de81071da27643b4d3c7dbf72e30263f38a0eaeae230"
    )
    _assert_same_array(independent.prediction, actual.prediction)
    assert (
        independent.prediction_sha256
        == actual.prediction_sha256
        == "9a73f17beb6be4ffb7f889a9141240f533838fe954291a82b897a1134e611ac6"
    )
    for field in fields(actual.certificate):
        independent_value = getattr(independent.certificate, field.name)
        actual_value = getattr(actual.certificate, field.name)
        if isinstance(actual_value, float):
            _assert_same_float(independent_value, actual_value)
        else:
            assert independent_value == actual_value

    for array in (
        independent.objective_cache.objectives,
        independent.selected.parameters,
        independent.selected.fit_prediction,
        independent.prediction,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)

    changed_objectives = independent.objective_cache.objectives.copy()
    changed_objectives[0] += 1.0
    changed_prediction = independent.prediction.copy()
    changed_prediction[0, 0] += 1.0
    with pytest.raises(oracle.OracleV22Error):
        replace(independent.source_grid.batch_records[0], ordinal=1)
    with pytest.raises(oracle.OracleV22Error):
        replace(independent.source_grid, content_sha256="0" * 64)
    with pytest.raises(oracle.OracleV22Error):
        replace(independent.objective_cache, objectives=changed_objectives)
    with pytest.raises(oracle.OracleV22Error):
        replace(independent.selected, state_index=1)
    with pytest.raises(oracle.OracleV22Error):
        replace(independent.certificate, candidate_count=-1)
    with pytest.raises(oracle.OracleV22Error):
        replace(independent.certificate, scalar_replay_bit_exact=False)
    with pytest.raises(oracle.OracleV22Error):
        replace(independent, arm="combined")
    with pytest.raises(oracle.OracleV22Error):
        replace(independent, prediction=changed_prediction)

    oracle.validate_oracle_result(
        independent,
        source,
        fit_target,
        oracle.FULL_MASK,
        oracle.FULL_MASK,
        config_sha256=CONFIG_SHA256,
    )
    lying_certificate = replace(independent.certificate, config_sha256="1" * 64)
    lying_result = replace(independent, certificate=lying_certificate)
    with pytest.raises(oracle.OracleV22Error, match="complete scalar rebuild"):
        oracle.validate_oracle_result(
            lying_result,
            source,
            fit_target,
            oracle.FULL_MASK,
            oracle.FULL_MASK,
            config_sha256=CONFIG_SHA256,
        )


def test_scalar_oracle_uses_full_canonical_key_for_exact_objective_ties() -> None:
    source = np.zeros((3, 64, 64), dtype=np.float64)
    fit_target = np.zeros((3, oracle.SITE_COUNT), dtype=np.float64)
    result = oracle.fit_scalar_oracle(
        source,
        fit_target,
        oracle.FULL_MASK,
        oracle.FULL_MASK,
        "affine",
        context_key="kat/all-tied/full",
        config_sha256=CONFIG_SHA256,
    )

    assert result.selected.state_index == 0
    assert result.selected.admissible_rank == 0
    assert result.selected.objective == 0.0
    assert result.certificate.exact_tie_multiplicity == oracle.ADMISSIBLE_COUNT
    assert result.certificate.second_best_objective_gap == 0.0
    assert result.certificate.second_best_nonflow_gap == 0.0
    assert result.certificate.scalar_replay_bit_exact


def test_complete_scalar_combined_oracle_matches_all_production_payloads() -> None:
    from bench.multimodal_mechanism_diagnostics import global_v22 as production

    source = _integer_source()
    central = np.ascontiguousarray(source[:, 8:56, 8:56].reshape(3, -1))
    fit_target = np.ascontiguousarray(
        np.asarray((1.25, 0.75, 1.5), dtype=np.float64)[:, None] * central
        + np.asarray((0.375, -0.25, 0.125), dtype=np.float64)[:, None]
    )
    independent = oracle.fit_scalar_oracle(
        source,
        fit_target,
        oracle.FULL_MASK,
        oracle.FULL_MASK,
        "combined",
        context_key="kat/combined/full",
        config_sha256=CONFIG_SHA256,
    )
    actual = production.fit_global(
        source,
        fit_target,
        oracle.FULL_MASK,
        oracle.FULL_MASK,
        "combined",
        context_key="kat/combined/full",
        config_sha256=CONFIG_SHA256,
    )

    assert independent.source_grid.content_sha256 == actual.source_grid.content_sha256
    assert independent.objective_cache.scope_sha256 == actual.objective_cache.scope_sha256
    assert (
        independent.objective_cache.content_sha256
        == actual.objective_cache.content_sha256
        == "98cb1ef598b61aba883676c57ed2fae2a955787f5ea43a5a9ce1f7aab2e1b8df"
    )
    _assert_same_array(independent.objective_cache.objectives, actual.objective_cache.objectives)
    _assert_same_array(independent.objective_cache.gains, actual.objective_cache.gains)
    _assert_same_array(independent.objective_cache.biases, actual.objective_cache.biases)
    assert independent.objective_cache.retained_macro_ids == actual.objective_cache.retained_macro_ids
    assert independent.selected.state_index == actual.selected.state_index == 0
    _assert_same_float(independent.selected.objective, actual.selected.objective)
    _assert_same_array(independent.selected.gains, actual.selected.gains)
    _assert_same_array(independent.selected.biases, actual.selected.biases)
    assert independent.selected.retained_macro_ids == actual.selected.retained_macro_ids
    _assert_same_array(independent.selected.fit_prediction, actual.selected.fit_prediction)
    assert independent.selected.evaluation_sha256 == actual.selected.evaluation_sha256
    _assert_same_array(independent.prediction, actual.prediction)
    assert (
        independent.prediction_sha256
        == actual.prediction_sha256
        == "8a17d93fbe424076568e4b393b7d6985bae2353fc0342123dbdfd6ee635def0f"
    )
    for field in fields(actual.certificate):
        independent_value = getattr(independent.certificate, field.name)
        actual_value = getattr(actual.certificate, field.name)
        if isinstance(actual_value, float):
            _assert_same_float(independent_value, actual_value)
        else:
            assert independent_value == actual_value


def test_combined_ols_uses_floor_and_raw_bias_before_independent_clipping() -> None:
    pixel = np.arange(oracle.SITE_COUNT)
    local = (((pixel // 48) % 8) * 8 + (pixel % 8) + 1).astype(np.float64) / 64.0
    sampled = np.ascontiguousarray(np.stack((local, np.full_like(local, 0.5), local)))
    fit_target = np.ascontiguousarray(
        np.stack((10.0 * local + 2.0, np.full_like(local, 9.0), -5.0 * local - 6.0))
    )
    result = oracle.evaluate_candidate("combined", sampled, fit_target, oracle.FULL_MASK)

    assert result.gains is not None and result.biases is not None
    assert result.gains.tobytes(order="C") == np.asarray(
        (4.0, 0.0, -2.0), dtype=np.float64
    ).tobytes(order="C")
    assert result.biases.tobytes(order="C") == np.asarray(
        (2.0, 4.0, -4.0), dtype=np.float64
    ).tobytes(order="C")
    assert result.retained_macro_ids == tuple(range(27))
    _assert_same_float(result.objective, float.fromhex("0x1.0d0d2aaaaaaabp+4"))


@pytest.mark.parametrize(
    ("fit_mask", "expected_objective", "expected_retained_count"),
    (
        (oracle.FULL_MASK, float.fromhex("0x1.28b3dfe57daedp-20"), 27),
        (oracle.PARITY_MASKS[0], float.fromhex("0x1.44a1bf9ffa99ap-20"), 14),
    ),
)
def test_direct_combined_math_matches_production_kat(
    fit_mask: np.ndarray, expected_objective: float, expected_retained_count: int
) -> None:
    from bench.multimodal_mechanism_diagnostics import fitting_v22 as production

    count = int(np.count_nonzero(fit_mask))
    pixel = np.arange(count, dtype=np.float64)
    sampled = np.ascontiguousarray(
        np.stack(
            (
                ((pixel % 97.0) - 48.0) / 16.0,
                (((pixel * 3.0) % 89.0) - 44.0) / 8.0,
                (((pixel * 7.0) % 101.0) - 50.0) / 32.0,
            )
        )
    )
    macro = oracle.MACRO_IDS[fit_mask].astype(np.float64)
    perturbation = np.ascontiguousarray(
        np.stack(
            (
                ((macro % 7.0) - 3.0) / 1024.0,
                ((macro % 5.0) - 2.0) / -2048.0,
                ((macro % 11.0) - 5.0) / 4096.0,
            )
        )
    )
    fit_target = np.ascontiguousarray(
        np.asarray((1.25, 0.75, 1.5), dtype=np.float64)[:, None] * sampled
        + np.asarray((0.375, -0.25, 0.125), dtype=np.float64)[:, None]
        + perturbation
    )

    independent = oracle.evaluate_candidate("combined", sampled, fit_target, fit_mask)
    actual = production.reduce_candidate("combined", sampled[None, :, :], fit_target, fit_mask)
    assert isinstance(actual, production.CombinedCandidateFit)
    _assert_same_float(independent.objective, expected_objective)
    _assert_same_float(independent.objective, actual.objective)
    _assert_same_array(independent.gains, actual.gains)
    _assert_same_array(independent.biases, actual.biases)
    _assert_same_array(independent.prediction, actual.prediction)
    assert independent.retained_macro_ids == actual.retained_macro_ids
    assert len(independent.retained_macro_ids) == expected_retained_count
    assert not independent.prediction.flags.writeable
    assert independent.gains is not None and not independent.gains.flags.writeable
    assert independent.biases is not None and not independent.biases.flags.writeable
