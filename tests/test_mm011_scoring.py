from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from bench.multimodal_causal_assay import scoring


def _central(value: float) -> np.ndarray:
    return np.full(scoring.CENTRAL_SHAPE, value, dtype="<f8")


def _inputs(*, family: scoring.Family = "combined", row_index: int = 0) -> scoring.RowScoreInputs:
    return scoring.RowScoreInputs(
        video_id="synthetic_video",
        fold=0,
        row_index=row_index,
        family=family,
        current_target=_central(1.0),
        future_target=_central(2.0),
        deranged_future_target=_central(4.0),
        history_identity=_central(0.0),
        history_xfit=_central(1.0),
        history_shuffle_xfit=_central(2.0),
        persistence=_central(1.0),
        forecast=_central(2.0),
        forecast_shuffle=_central(0.0),
        forecast_reverse=_central(3.0),
        velocity=_central(4.0),
        history_bias=_central(0.5),
        forecast_bias=_central(1.5),
    )


def test_score_row_uses_untrimmed_named_targets() -> None:
    row = scoring.score_row(_inputs())
    unit = scoring.ELEMENTS_PER_ROW
    assert row.i == scoring.ErrorPrimitive(float(unit), unit)
    assert row.a == scoring.ErrorPrimitive(0.0, unit)
    assert row.q == scoring.ErrorPrimitive(float(unit), unit)
    assert row.p == scoring.ErrorPrimitive(float(unit), unit)
    assert row.c == scoring.ErrorPrimitive(0.0, unit)
    assert row.h == scoring.ErrorPrimitive(4.0 * unit, unit)
    assert row.r == scoring.ErrorPrimitive(float(unit), unit)
    assert row.z == scoring.ErrorPrimitive(4.0 * unit, unit)
    assert row.d == scoring.ErrorPrimitive(4.0 * unit, unit)
    assert row.pd == scoring.ErrorPrimitive(9.0 * unit, unit)
    assert row.u == scoring.ErrorPrimitive(0.25 * unit, unit)
    assert row.b == scoring.ErrorPrimitive(0.25 * unit, unit)
    assert row.bd == scoring.ErrorPrimitive(6.25 * unit, unit)
    scoring.validate_row_scores(_inputs(), row)


@pytest.mark.parametrize("family", scoring.FAMILIES)
def test_every_family_requires_and_scores_bias_controls(family: scoring.Family) -> None:
    row = scoring.score_row(_inputs(family=family))
    assert row.u == scoring.ErrorPrimitive(0.25 * scoring.ELEMENTS_PER_ROW, scoring.ELEMENTS_PER_ROW)
    assert row.b == scoring.ErrorPrimitive(0.25 * scoring.ELEMENTS_PER_ROW, scoring.ELEMENTS_PER_ROW)
    with pytest.raises(scoring.ScoringError, match="exact NumPy array"):
        replace(_inputs(family=family), history_bias=cast(np.ndarray, None))


@pytest.mark.parametrize("mutation", ["dtype", "shape", "order", "nonfinite"])
def test_scoring_arrays_fail_closed(mutation: str) -> None:
    value = _central(0.0)
    if mutation == "dtype":
        bad = value.astype(np.float32)
    elif mutation == "shape":
        bad = value[:, :-1].copy()
    elif mutation == "order":
        bad = value[:, ::-1]
    else:
        bad = value.copy()
        bad[0, 0] = np.nan
    with pytest.raises(scoring.ScoringError):
        replace(_inputs(), forecast=bad)


def test_scoring_inputs_are_detached_and_hard_readonly() -> None:
    original = _central(2.0)
    inputs = replace(_inputs(), forecast=original)
    original.fill(100.0)
    assert np.all(inputs.forecast == 2.0)
    assert not inputs.forecast.flags.writeable
    with pytest.raises(ValueError):
        inputs.forecast.reshape(-1)[0] = 3.0


def test_scoring_rejects_array_subclasses_and_unrepresentable_sse() -> None:
    class CentralSubclass(np.ndarray):
        pass

    subclass: np.ndarray = _central(0.0).view(CentralSubclass)
    with pytest.raises(scoring.ScoringError, match="exact NumPy array"):
        replace(_inputs(), forecast=subclass)
    with pytest.raises(scoring.ScoringError, match="finite and nonnegative"):
        scoring.ErrorPrimitive(10**10000, 1)


def test_video_aggregation_sums_row_sse_then_count() -> None:
    first = scoring.score_row(_inputs(row_index=0))
    second = scoring.score_row(_inputs(row_index=1))
    video = scoring.aggregate_video((first, second))
    assert video.row_count == 2
    assert video.i.sse == first.i.sse + second.i.sse
    assert video.i.count == 2 * scoring.ELEMENTS_PER_ROW
    assert video.bd.sse == 2.0 * first.bd.sse
    scoring.validate_video_scores((first, second), video)


def test_video_aggregation_rejects_order_and_tampering() -> None:
    first = scoring.score_row(_inputs(row_index=0))
    second = scoring.score_row(_inputs(row_index=1))
    with pytest.raises(scoring.ScoringError, match="canonical order"):
        scoring.aggregate_video((second, first))
    video = scoring.aggregate_video((first, second))
    tampered = replace(video, c=scoring.ErrorPrimitive(video.c.sse + 1.0, video.c.count))
    with pytest.raises(scoring.ScoringError, match="canonical row aggregation"):
        scoring.validate_video_scores((first, second), tampered)
    tampered_row = replace(first, c=scoring.ErrorPrimitive(1.0, scoring.ELEMENTS_PER_ROW))
    with pytest.raises(scoring.ScoringError, match="pure target recomputation"):
        scoring.validate_row_scores(_inputs(), tampered_row)


def test_sse_overflow_is_rejected() -> None:
    with pytest.raises(scoring.ScoringError, match="overflowed"):
        scoring.squared_error(_central(1e308), _central(-1e308))
