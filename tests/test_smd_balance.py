"""
Standardized mean differences, including covariates that cannot be assessed.

A covariate with no variance in either arm divides by zero. The resulting NaN
was then skipped by the summary, so the balance report looked complete while a
covariate had in fact never been checked.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import samatch


@pytest.fixture(scope="module")
def matched_sample():
    data = samatch.load_sample_4group()
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
        gps_space="logit",
    )
    matched = samatch.sam_match(
        data, search, X_vars=covariates, treatment_var="treatment"
    )
    return data, covariates, search, matched


def test_zero_variance_covariate_does_not_warn_or_produce_nan(matched_sample):
    data, covariates, search, matched = matched_sample

    frame = data.copy()
    frame["const"] = 1.0

    with warnings.catch_warnings():
        # A raw "invalid value encountered in scalar divide" would fail here.
        warnings.simplefilter("error")
        balance = samatch.compute_smd_balance(
            frame,
            matched["matched"],
            covariates + ["const"],
            search["groups"],
        )

    by_covariate = balance["by_covariate"]
    constant = by_covariate[by_covariate["covariate"] == "const"]

    assert len(constant) == len(search["groups"])
    assert constant["smd"].notna().all()
    assert not constant["smd_defined"].any()


def test_undefined_covariates_are_counted_in_the_summary(matched_sample):
    data, covariates, search, matched = matched_sample

    frame = data.copy()
    frame["const_a"] = 1.0
    frame["const_b"] = 7.0

    balance = samatch.compute_smd_balance(
        frame,
        matched["matched"],
        covariates + ["const_a", "const_b"],
        search["groups"],
    )

    summary = balance["summary"]

    assert (summary["n_undefined"] == 2).all()
    assert summary["max_abs_smd"].notna().all()
    assert summary["mean_abs_smd"].notna().all()


def test_undefined_covariates_do_not_change_the_summary_statistics(
    matched_sample,
):
    """Adding an unassessable covariate must not move the reported balance."""
    data, covariates, search, matched = matched_sample

    without = samatch.compute_smd_balance(
        data, matched["matched"], covariates, search["groups"]
    )["summary"]

    frame = data.copy()
    frame["const"] = 1.0

    with_constant = samatch.compute_smd_balance(
        frame, matched["matched"], covariates + ["const"], search["groups"]
    )["summary"]

    np.testing.assert_allclose(
        without["mean_abs_smd"], with_constant["mean_abs_smd"]
    )
    np.testing.assert_allclose(
        without["max_abs_smd"], with_constant["max_abs_smd"]
    )
    assert (without["n_undefined"] == 0).all()
    assert (with_constant["n_undefined"] == 1).all()


def test_ordinary_covariates_stay_flagged_as_defined(matched_sample):
    data, covariates, search, matched = matched_sample

    balance = samatch.compute_smd_balance(
        data, matched["matched"], covariates, search["groups"]
    )

    assert balance["by_covariate"]["smd_defined"].all()
    assert (balance["summary"]["n_undefined"] == 0).all()


def test_smd_matches_the_hand_computed_value():
    """
    Anchor x = {0, 2}: mean 1, var 2. Comparator x = {3, 5}: mean 4, var 2.

        SMD = (1 - 4) / sqrt((2 + 2) / 2) = -3 / sqrt(2)
    """
    frame = pd.DataFrame(
        {"x": [0.0, 2.0, 3.0, 5.0], "T": ["A", "A", "B", "B"]}
    )
    matched = pd.DataFrame(
        [
            {"matched_set_id": 1, "anchor": 0, "B": 2, "loss": 0.0},
            {"matched_set_id": 2, "anchor": 1, "B": 3, "loss": 0.0},
        ]
    )

    balance = samatch.compute_smd_balance(frame, matched, ["x"], ["B"])

    np.testing.assert_allclose(
        balance["by_covariate"]["smd"], [-3.0 / np.sqrt(2.0)]
    )
    assert balance["by_covariate"]["smd_defined"].all()


def test_identical_arms_give_zero_smd():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 1.0, 2.0], "T": ["A", "A", "B", "B"]}
    )
    matched = pd.DataFrame(
        [
            {"matched_set_id": 1, "anchor": 0, "B": 2, "loss": 0.0},
            {"matched_set_id": 2, "anchor": 1, "B": 3, "loss": 0.0},
        ]
    )

    balance = samatch.compute_smd_balance(frame, matched, ["x"], ["B"])

    np.testing.assert_allclose(balance["by_covariate"]["smd"], [0.0])
    assert balance["by_covariate"]["smd_defined"].all()
