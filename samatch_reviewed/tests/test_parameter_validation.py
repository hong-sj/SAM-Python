"""
Numeric parameter validation.

Before these checks existed, a nonsensical value did not raise: it changed the
meaning of the parameter. `top_m=-1` reached a `[:m]` slice and kept every
candidate except the farthest one, which is the opposite of "keep the nearest
one".
"""

import numpy as np
import pandas as pd
import pytest

import samatch


@pytest.fixture(scope="module")
def three_group():
    data = samatch.load_sample_3group()
    covariates = [
        column
        for column in data.columns
        if column not in ("synthetic_id", "treatment", "mortality_28d")
    ]
    anchor = data["treatment"].astype(str).unique()[0]

    fit = samatch.estimate_gps_multinom(
        data, X_vars=covariates, treatment_var="treatment", anchor_level=anchor
    )
    search = samatch.gps_candidate_search(
        data,
        fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
        top_m=10,
    )
    return data, anchor, fit, search


@pytest.mark.parametrize("top_m", [0, -1, 2.5, "10", None])
def test_invalid_top_m_raises(three_group, top_m):
    data, anchor, fit, _ = three_group

    with pytest.raises(ValueError, match="top_m must be a positive integer"):
        samatch.gps_candidate_search(
            data,
            fit["gps"],
            treatment_var="treatment",
            anchor_level=anchor,
            top_m=top_m,
        )


@pytest.mark.parametrize("top_n", [0, -1, 2.5, "10"])
def test_invalid_top_n_raises(three_group, top_n):
    data, _, fit, search = three_group

    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        samatch.match_3way(
            data,
            search,
            fit["gps"],
            treatment_var="treatment",
            top_n=top_n,
        )


def test_valid_top_m_keeps_exactly_that_many(three_group):
    data, anchor, fit, _ = three_group

    search = samatch.gps_candidate_search(
        data,
        fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
        top_m=3,
    )

    for group in search["groups"]:
        assert len(search["candidates"][0][group]) == 3


def test_numpy_integers_are_accepted(three_group):
    data, anchor, fit, _ = three_group

    search = samatch.gps_candidate_search(
        data,
        fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
        top_m=np.int64(4),
    )

    for group in search["groups"]:
        assert len(search["candidates"][0][group]) == 4


def test_booleans_are_rejected():
    """`True` is an int in Python, but it is not a candidate count."""
    frame = pd.DataFrame({"T": ["A", "B", "B"]})
    gps = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3})

    with pytest.raises(ValueError, match="top_m must be a positive integer"):
        samatch.gps_candidate_search(
            frame, gps, treatment_var="T", anchor_level="A", top_m=True
        )
