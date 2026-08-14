"""
Public API surface: naming, defaults, deprecations, and reported metrics.
"""

import numpy as np
import pandas as pd
import pytest

import samatch
from samatch._match_common import build_matched_frame, groups_from_matched


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
    return data, covariates, fit, search


def test_all_exported_names_exist():
    for name in samatch.__all__:
        assert hasattr(samatch, name), name


# gps_space / ps_space -------------------------------------------------------


def test_ps_space_is_deprecated_but_still_works(three_group):
    data, _, fit, search = three_group

    with pytest.warns(DeprecationWarning, match="ps_space"):
        old = samatch.match_3way(
            data, search, fit["gps"], treatment_var="treatment", ps_space="logit"
        )

    new = samatch.match_3way(
        data, search, fit["gps"], treatment_var="treatment", gps_space="logit"
    )

    pd.testing.assert_frame_equal(old["matched"], new["matched"])


def test_invalid_gps_space_raises(three_group):
    data, _, fit, search = three_group

    with pytest.raises(ValueError, match="raw.*logit"):
        samatch.match_3way(
            data, search, fit["gps"], treatment_var="treatment", gps_space="nope"
        )

    with pytest.raises(ValueError, match="raw.*logit"):
        samatch.gps_candidate_search(
            data,
            fit["gps"],
            treatment_var="treatment",
            anchor_level="piperacillin_tazobactam",
            gps_space="nope",
        )


def test_match_3way_warns_that_it_ignores_covariates(three_group):
    data, covariates, fit, search = three_group

    with pytest.warns(UserWarning, match="ignores X_vars"):
        samatch.match_3way(
            data,
            search,
            fit["gps"],
            X_vars=covariates,
            treatment_var="treatment",
        )


# Inferred defaults ----------------------------------------------------------


def test_groups_are_inferred_from_the_matched_frame(three_group):
    data, covariates, fit, search = three_group

    matched = samatch.match_3way(
        data, search, fit["gps"], treatment_var="treatment"
    )["matched"]

    assert groups_from_matched(matched) == list(search["groups"])

    explicit = samatch.compute_smd_balance(
        data, matched, covariates, search["groups"]
    )
    inferred = samatch.compute_smd_balance(data, matched, covariates)

    pd.testing.assert_frame_equal(
        explicit["by_covariate"], inferred["by_covariate"]
    )


def test_pairwise_auc_requires_an_anchor_level(three_group):
    data, _, fit, search = three_group

    matched = samatch.match_3way(
        data, search, fit["gps"], treatment_var="treatment"
    )["matched"]

    with pytest.raises(ValueError, match="anchor_level"):
        samatch.compute_pairwise_treatment_auc(fit["gps"], matched)


# Reported metrics -----------------------------------------------------------


def test_matching_rate_is_bounded_by_max_possible_rate():
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

    counts = data["treatment"].astype(str).value_counts()
    smallest = counts.drop(anchor).min()

    assert matched["max_possible_rate"] == pytest.approx(
        smallest / counts[anchor]
    )
    assert matched["matching_rate"] <= matched["max_possible_rate"]

    # On this dataset SAM forms every set the group sizes permit.
    assert matched["matching_rate"] == pytest.approx(
        matched["max_possible_rate"]
    )


# Empty results --------------------------------------------------------------


def test_empty_match_keeps_the_column_schema(three_group):
    """A caliper that admits nothing must still return a well-formed frame."""
    data, _, fit, search = three_group

    result = samatch.match_3way(
        data,
        search,
        fit["gps"],
        treatment_var="treatment",
        caliper=1e-12,
    )

    matched = result["matched"]

    assert len(matched) == 0
    assert result["matching_rate"] == 0.0
    assert len(result["unmatched_anchor_rows"]) == len(search["anchor_rows"])

    expected = [
        "matched_set_id",
        "anchor",
        *search["groups"],
        *[f"dist_{group}" for group in search["groups"]],
        "loss",
        "rassen_perimeter",
    ]
    assert list(matched.columns) == expected


def test_build_matched_frame_schema_matches_the_populated_case():
    groups = ["B", "C"]

    populated = build_matched_frame(
        [
            {
                "matched_set_id": 1,
                "anchor": 0,
                "B": 1,
                "dist_B": 0.5,
                # Deliberately use the interleaved insertion order emitted by
                # the matching engines. The assembly helper owns the schema.
                "C": 2,
                "dist_C": 0.25,
                "loss": 0.75,
            }
        ],
        groups,
    )
    empty = build_matched_frame([], groups)

    assert list(populated.columns) == list(empty.columns)


def test_extract_matched_data_rejects_an_empty_match(three_group):
    data, _, fit, search = three_group

    result = samatch.match_3way(
        data, search, fit["gps"], treatment_var="treatment", caliper=1e-12
    )

    with pytest.raises(ValueError, match="no matched sets"):
        samatch.extract_matched_data(
            data, search, result, treatment_var="treatment"
        )


# match_3way input validation --------------------------------------------------


def test_match_3way_requires_exactly_three_groups():
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
        top_m=5,
    )

    with pytest.raises(ValueError, match="exactly three treatment groups"):
        samatch.match_3way(
            data, search, fit["gps"], treatment_var="treatment"
        )


@pytest.mark.parametrize("caliper", [0.0, -1.0, np.nan, np.inf])
def test_non_positive_caliper_raises(three_group, caliper):
    data, _, fit, search = three_group

    with pytest.raises(ValueError, match="finite and greater than zero"):
        samatch.match_3way(
            data,
            search,
            fit["gps"],
            treatment_var="treatment",
            caliper=caliper,
        )


def test_auto_caliper_rejects_a_singleton_treatment_group():
    data = pd.DataFrame({"T": ["A", "B", "C"]})
    gps = pd.DataFrame(
        {
            "A": [0.8, 0.1, 0.1],
            "B": [0.1, 0.8, 0.1],
            "C": [0.1, 0.1, 0.8],
        },
        index=data.index,
    )
    search = samatch.gps_candidate_search(data, gps, anchor_level="A", top_m=1)

    with pytest.raises(ValueError, match="at least two subjects"):
        samatch.match_3way(data, search, gps, caliper="auto")

    # An explicitly chosen finite caliper remains valid for this small cohort.
    result = samatch.match_3way(data, search, gps, caliper=10.0)
    assert len(result["matched"]) == 1


def test_invalid_caliper_string_raises(three_group):
    data, _, fit, search = three_group

    with pytest.raises(ValueError, match='"auto"'):
        samatch.match_3way(
            data,
            search,
            fit["gps"],
            treatment_var="treatment",
            caliper="tight",
        )


def test_gps_row_count_must_match_data(three_group):
    data, _, fit, search = three_group

    with pytest.raises(ValueError, match="same number of rows"):
        samatch.gps_candidate_search(
            data,
            fit["gps"].iloc[:-1],
            treatment_var="treatment",
            anchor_level="piperacillin_tazobactam",
        )


def test_missing_covariate_column_is_named(three_group):
    data, _, _, search = three_group

    with pytest.raises(ValueError, match="not found in data.*nonexistent"):
        samatch.sam_match(
            data,
            search,
            X_vars=["age", "nonexistent"],
            treatment_var="treatment",
        )


def test_missing_treatment_column_raises(three_group):
    data, covariates, _, search = three_group

    with pytest.raises(ValueError, match="not found in data"):
        samatch.sam_match(
            data, search, X_vars=covariates, treatment_var="absent"
        )


# Small public helpers ---------------------------------------------------------


def test_logit_and_expit_round_trip():
    values = np.array([0.01, 0.25, 0.5, 0.75, 0.99])

    np.testing.assert_allclose(
        samatch.expit(samatch.logit(values)), values, rtol=1e-12
    )


def test_auc_mannwhitney_matches_hand_computed_value():
    # Perfect separation, then a deliberate single swap.
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert samatch.auc_mannwhitney(scores, labels) == pytest.approx(1.0)

    labels = np.array([0.0, 1.0, 0.0, 1.0])
    assert samatch.auc_mannwhitney(scores, labels) == pytest.approx(0.75)


def test_build_group_distance_matrices_shapes_and_values(three_group):
    data, covariates, _, search = three_group

    built = samatch.build_group_distance_matrices(
        data,
        covariates,
        "treatment",
        search["anchor_rows"],
        search["groups"],
    )

    X = data[covariates].to_numpy(dtype=float)
    treatment = data["treatment"].astype(str).to_numpy()

    for group in search["groups"]:
        rows = np.flatnonzero(treatment == group)

        assert built["D"][group].shape == (len(search["anchor_rows"]), len(rows))
        np.testing.assert_array_equal(built["group_rows"][group], rows)

        expected = samatch.mahalanobis_distance_matrix(
            X[search["anchor_rows"]], X[rows], built["S_inv"]
        )
        np.testing.assert_allclose(built["D"][group], expected)


def test_extract_matched_data_expands_sets_without_reordering():
    """
    The expansion is a reshape of the matched-set block rather than a per-row
    loop. Role order within a set, and set order overall, must be unaffected.
    """
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
    cohort = samatch.extract_matched_data(
        data, search, matched, treatment_var="treatment"
    )

    expected_order = [anchor, *search["groups"]]
    k = len(expected_order)

    assert len(cohort) == k * len(matched["matched"])
    assert cohort["matched_role"].tolist()[:k] == expected_order
    assert (cohort["matched_role"] == cohort["treatment"].astype(str)).all()

    # Every subject-level row must point back at the row the matched set named.
    # The anchor is held in the "anchor" column, comparators under their group.
    first_set = matched["matched"].iloc[0]
    np.testing.assert_array_equal(
        cohort["original_row"].to_numpy()[:k],
        np.array(
            [first_set["anchor"], *[first_set[g] for g in search["groups"]]],
            dtype=int,
        ),
    )
