"""
Tests for Shared Anchor Matching, three-way matching, diagnostics,
outcome analysis, and multi-arm weighting.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import samatch
from samatch._match_3way import (
    calc_caliper_3way,
    kdtree_build,
    kdtree_nearest,
    kdtree_range,
)


@pytest.fixture(scope="module")
def four_group_data():
    """Load the four-treatment-group example dataset."""
    return samatch.load_sample_4group().reset_index(drop=True)


@pytest.fixture(scope="module")
def three_group_data():
    """Load the three-treatment-group example dataset."""
    return samatch.load_sample_3group().reset_index(drop=True)


def get_covariates(data):
    """Return covariate columns from an example dataset."""
    excluded = {"synthetic_id", "treatment", "mortality_28d"}
    return [column for column in data.columns if column not in excluded]


def fit_and_search(data, top_m=10, gps_space="logit"):
    """Estimate GPS and perform candidate search."""
    covariates = get_covariates(data)
    anchor = data["treatment"].astype(str).unique()[0]

    fit = samatch.estimate_gps_multinom(
        data,
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    search = samatch.gps_candidate_search(
        data,
        fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
        top_m=top_m,
        gps_space=gps_space,
    )

    return covariates, anchor, fit, search


# Mahalanobis utilities --------------------------------------------------------


def test_mahalanobis_distance_matrix():
    rng = np.random.default_rng(1)

    x1 = rng.normal(size=(10, 2))
    x2 = rng.normal(size=(15, 2))
    s_inv = np.eye(2)

    distances = samatch.mahalanobis_distance_matrix(x1, x2, s_inv)

    assert distances.shape == (10, 15)
    assert (distances >= 0).all()
    assert np.allclose(distances, cdist(x1, x2))


def test_get_pooled_covariance(four_group_data):
    covariates = get_covariates(four_group_data)

    pooled = samatch.get_pooled_covariance(
        four_group_data,
        covariates,
        "treatment",
    )

    p = len(covariates)

    assert pooled["S"].shape == (p, p)
    assert pooled["S_inv"].shape == (p, p)
    assert np.allclose(pooled["S"], pooled["S"].T)


# GPS estimation and candidate search -----------------------------------------


def test_estimate_gps_multinom(four_group_data):
    covariates, anchor, fit, _ = fit_and_search(four_group_data)
    gps = fit["gps"]

    n_groups = four_group_data["treatment"].nunique()

    assert gps.shape == (len(four_group_data), n_groups)
    assert gps.columns[0] == anchor
    assert np.allclose(gps.sum(axis=1), 1.0)
    assert (gps.to_numpy() >= 0).all()
    assert (gps.to_numpy() <= 1).all()
    assert len(covariates) > 0


def test_gps_candidate_search(four_group_data):
    _, anchor, _, search = fit_and_search(
        four_group_data,
        top_m=5,
    )

    n_anchor = (
        four_group_data["treatment"].astype(str) == anchor
    ).sum()

    assert len(search["anchor_rows"]) == n_anchor
    assert len(search["groups"]) == 3

    for candidates in search["candidates"]:
        for group in search["groups"]:
            assert len(candidates[group]) <= 5


# Shared Anchor Matching -------------------------------------------------------


def test_sam_match(four_group_data):
    covariates, anchor, _, search = fit_and_search(four_group_data)

    result = samatch.sam_match(
        four_group_data,
        search,
        X_vars=covariates,
        treatment_var="treatment",
    )

    matched = result["matched"]

    assert 0 <= result["matching_rate"] <= 1
    assert len(matched) == (
        len(search["anchor_rows"])
        - len(result["unmatched_anchor_rows"])
    )

    if len(matched) == 0:
        return

    assert (matched["loss"] >= 0).all()
    assert matched["anchor"].is_unique

    anchor_treatment = four_group_data.iloc[
        matched["anchor"].to_numpy()
    ]["treatment"].astype(str)

    assert (anchor_treatment == anchor).all()

    for group in search["groups"]:
        assert matched[group].is_unique

        group_treatment = four_group_data.iloc[
            matched[group].to_numpy()
        ]["treatment"].astype(str)

        assert (group_treatment == group).all()


def test_sam_evaluate(four_group_data):
    covariates, _, fit, search = fit_and_search(four_group_data)

    result = samatch.sam_match(
        four_group_data,
        search,
        X_vars=covariates,
        treatment_var="treatment",
    )

    report = samatch.sam_evaluate(
        four_group_data,
        search,
        result,
        fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
    )

    assert report["matching_rate"] == result["matching_rate"]

    if len(result["matched"]) > 0:
        assert np.isfinite(
            report["loss_distribution"]["mean"].iloc[0]
        )
        assert np.isfinite(
            report["dispersion_distribution"]["mean"].iloc[0]
        )

        mean_auc = report[
            "treatment_discrimination_auc"
        ]["mean_auc"]

        assert 0 <= mean_auc <= 1


# Matched cohort and outcome analysis -----------------------------------------


def test_extract_matched_data(four_group_data):
    covariates, anchor, _, search = fit_and_search(four_group_data)

    result = samatch.sam_match(
        four_group_data,
        search,
        X_vars=covariates,
        treatment_var="treatment",
    )

    matched_data = samatch.extract_matched_data(
        four_group_data,
        search,
        result,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    n_groups = four_group_data["treatment"].nunique()

    assert "matched_set_id" in matched_data.columns
    assert "matched_role" in matched_data.columns
    assert "original_row" in matched_data.columns
    assert len(matched_data) == len(result["matched"]) * n_groups


def test_sam_estimate_effects(four_group_data):
    covariates, anchor, _, search = fit_and_search(four_group_data)

    result = samatch.sam_match(
        four_group_data,
        search,
        X_vars=covariates,
        treatment_var="treatment",
    )

    matched_data = samatch.extract_matched_data(
        four_group_data,
        search,
        result,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    effects = samatch.sam_estimate_effects(
        matched_data,
        outcome_var="mortality_28d",
        treatment_var="treatment",
        anchor_level=anchor,
    )

    assert "analysis_summary" in effects
    assert "group_risk" in effects
    assert "contrasts" in effects
    assert "vcov_cluster" in effects

    assert len(effects["group_risk"]) == four_group_data[
        "treatment"
    ].nunique()


# Multi-arm weighting ----------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["iptw", "overlap", "matching"],
)
def test_compute_balancing_weights(four_group_data, method):
    covariates, anchor, fit, _ = fit_and_search(four_group_data)

    result = samatch.compute_balancing_weights(
        four_group_data,
        method=method,
        gps=fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    assert len(result["weights"]) == len(four_group_data)
    assert np.isfinite(result["weights"]).all()
    assert (result["weights"] > 0).all()

    if method in ("overlap", "matching"):
        assert (result["weights"] <= 1.0 + 1e-9).all()


def test_evaluate_comparator_weighting(four_group_data):
    covariates, anchor, fit, _ = fit_and_search(four_group_data)

    result = samatch.evaluate_comparator_weighting(
        four_group_data,
        method="overlap",
        gps=fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    assert "balance" in result
    assert "ess" in result

    assert (result["ess"]["ess"] > 0).all()
    assert (result["ess"]["ess"] <= result["ess"]["n"] + 1e-9).all()


# Three-way matching -----------------------------------------------------------


def test_calc_caliper_3way():
    rng = np.random.default_rng(2)

    ps_used = rng.normal(size=(60, 2))
    treatment = np.repeat(["A", "B", "C"], 20)

    caliper = calc_caliper_3way(ps_used, treatment)

    assert np.isfinite(caliper)
    assert caliper > 0


def test_kdtree_nearest():
    rng = np.random.default_rng(3)

    coords = rng.normal(size=(50, 2))
    tree = kdtree_build(coords)
    active = np.ones(50, dtype=bool)

    active[rng.choice(50, size=10, replace=False)] = False

    for _ in range(20):
        query = rng.normal(size=2)

        observed = kdtree_nearest(
            tree,
            coords,
            query,
            active,
        )

        distance_sq = np.sum(
            (coords - query) ** 2,
            axis=1,
        )
        distance_sq[~active] = np.inf

        expected = int(np.argmin(distance_sq))

        assert observed == expected


def test_kdtree_range():
    rng = np.random.default_rng(4)

    coords = rng.normal(size=(50, 2))
    tree = kdtree_build(coords)
    active = np.ones(50, dtype=bool)

    active[rng.choice(50, size=10, replace=False)] = False

    for _ in range(20):
        query = rng.normal(size=2)
        radius_sq = float(rng.uniform(0.1, 2.0))

        observed = set(
            kdtree_range(
                tree,
                coords,
                query,
                radius_sq,
                active,
            )
        )

        distance_sq = np.sum(
            (coords - query) ** 2,
            axis=1,
        )

        expected = set(
            np.flatnonzero(
                (distance_sq < radius_sq) & active
            )
        )

        assert observed == expected


@pytest.mark.parametrize(
    "gps_space",
    ["raw", "logit"],
)
def test_match_3way(three_group_data, gps_space):
    _, _, fit, search = fit_and_search(three_group_data)

    result = samatch.match_3way(
        three_group_data,
        search,
        fit["gps"],
        treatment_var="treatment",
        gps_space=gps_space,
    )

    matched = result["matched"]

    assert 0 <= result["matching_rate"] <= 1
    assert result["caliper"] > 0
    assert result["red_level"] in fit["gps"].columns
    assert result["reference_level"] in fit["gps"].columns

    if len(matched) == 0:
        return

    assert (matched["loss"] >= 0).all()
    assert (matched["rassen_perimeter"] >= 0).all()
    assert matched["anchor"].is_unique

    for group in search["groups"]:
        assert matched[group].is_unique

    assert (
        matched["rassen_perimeter"]
        >= matched["loss"] - 1e-9
    ).all()


def test_match_3way_integrates_with_sam_evaluate(three_group_data):
    covariates, _, fit, search = fit_and_search(three_group_data)

    result = samatch.match_3way(
        three_group_data,
        search,
        fit["gps"],
        treatment_var="treatment",
    )

    report = samatch.sam_evaluate(
        three_group_data,
        search,
        result,
        fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
    )

    assert report["matching_rate"] == result["matching_rate"]

    if len(result["matched"]) > 0:
        assert np.isfinite(
            report["loss_distribution"]["mean"].iloc[0]
        )
        assert np.isfinite(
            report["dispersion_distribution"]["mean"].iloc[0]
        )


def test_match_3way_caliper_monotonicity(three_group_data):
    covariates, _, fit, search = fit_and_search(three_group_data)

    tight = samatch.match_3way(
        three_group_data,
        search,
        fit["gps"],
        treatment_var="treatment",
        caliper=0.05,
    )

    loose = samatch.match_3way(
        three_group_data,
        search,
        fit["gps"],
        treatment_var="treatment",
        caliper=999.0,
    )

    assert loose["matching_rate"] >= tight["matching_rate"]