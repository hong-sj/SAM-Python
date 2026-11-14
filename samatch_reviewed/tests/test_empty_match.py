"""
A match that formed no sets stays usable downstream.

`match_3way` with a tight caliper legitimately returns zero matched sets. The
empty frame was built without dtypes, so every column came back as `object`,
and the first diagnostic to use those columns as row indices failed with a bare
`IndexError` instead of reporting an empty match.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import samatch
from samatch._match_common import (
    build_matched_frame,
    matched_frame_columns,
    matched_frame_dtypes,
)


@pytest.fixture(scope="module")
def empty_three_group():
    """A real three-way run whose caliper admits nothing."""
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matched = samatch.match_3way(
            data,
            search,
            fit["gps"],
            treatment_var="treatment",
            caliper=1e-9,
        )

    return data, covariates, search, fit, matched


def test_the_empty_branch_is_actually_reached(empty_three_group):
    """Guards the premise: without this the rest proves nothing."""
    _, _, _, _, matched = empty_three_group

    assert len(matched["matched"]) == 0
    assert matched["matching_rate"] == 0.0


def test_empty_frame_dtypes_match_the_populated_frame(empty_three_group):
    _, _, _, _, matched = empty_three_group
    frame = matched["matched"]

    assert list(frame.columns) == matched_frame_columns(
        ["cefepime", "meropenem"], extra_columns=("rassen_perimeter",)
    )

    for column, dtype in matched_frame_dtypes(
        ["cefepime", "meropenem"], extra_columns=("rassen_perimeter",)
    ).items():
        assert frame[column].dtype == np.dtype(dtype), column


def test_row_columns_of_an_empty_frame_can_index_data(empty_three_group):
    """The failure was an object dtype reaching `DataFrame.iloc`."""
    data, _, _, _, matched = empty_three_group

    assert len(data.iloc[matched["matched"]["anchor"].to_numpy()]) == 0


def test_sam_evaluate_reports_an_empty_match(empty_three_group):
    data, covariates, search, fit, matched = empty_three_group

    report = samatch.sam_evaluate(
        data,
        search,
        matched,
        fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
    )

    assert len(report["smd_balance"]["by_covariate"]) == 0
    assert len(report["smd_balance"]["summary"]) == 0
    assert len(report["treatment_discrimination_auc"]["pairwise"]) == 0
    assert np.isnan(report["treatment_discrimination_auc"]["mean_auc"])
    assert np.isnan(report["loss_distribution"]["mean"].iloc[0])
    assert report["matching_rate"] == 0.0


def test_an_empty_match_is_not_reported_as_unassessable_covariates(
    empty_three_group,
):
    """
    "Nothing matched" and "this covariate has no variance" are different
    findings and must not arrive through the same channel.
    """
    data, covariates, _, _, matched = empty_three_group

    balance = samatch.compute_smd_balance(
        data, matched["matched"], covariates, ["cefepime", "meropenem"]
    )

    # Before the fix this returned one row per comparator group reporting
    # n_undefined == len(covariates), which reads as a covariate problem.
    assert len(balance["summary"]) == 0
    assert balance["summary"]["n_undefined"].sum() == 0


def test_extract_matched_data_still_refuses_an_empty_match(empty_three_group):
    data, _, search, _, matched = empty_three_group

    with pytest.raises(ValueError, match="no matched sets"):
        samatch.extract_matched_data(
            data, search, matched, treatment_var="treatment"
        )


def test_empty_and_populated_frames_agree_on_schema():
    groups = ["B", "C"]

    populated = build_matched_frame(
        [
            {
                "matched_set_id": 1,
                "anchor": 0,
                "B": 1,
                "dist_B": 0.5,
                "C": 2,
                "dist_C": 1.5,
                "loss": 2.0,
            }
        ],
        groups,
    )
    empty = build_matched_frame([], groups)

    assert list(populated.columns) == list(empty.columns)
    assert populated.dtypes.to_dict() == empty.dtypes.to_dict()


def test_smd_balance_on_a_single_matched_set_is_not_a_crash():
    """
    A one-set match cannot estimate a variance, so no SMD is defined. It must
    still come back as a report rather than an exception.
    """
    rng = np.random.default_rng(5)
    data = pd.DataFrame(rng.normal(size=(60, 2)), columns=["x", "y"])
    matched = build_matched_frame(
        [
            {
                "matched_set_id": 1,
                "anchor": 0,
                "B": 1,
                "dist_B": 0.5,
                "C": 2,
                "dist_C": 1.5,
                "loss": 2.0,
            }
        ],
        ["B", "C"],
    )

    balance = samatch.compute_smd_balance(data, matched, ["x", "y"], ["B", "C"])

    assert not balance["by_covariate"]["smd_defined"].any()
    assert (balance["summary"]["n_undefined"] == 2).all()
