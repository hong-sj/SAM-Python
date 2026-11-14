"""
Treatment labels are compared consistently across the pipeline.

`estimate_gps_multinom` casts the treatment column to `str`, so every GPS
column label and group name downstream is a string. When the other modules
compared those against the raw dtype, a numeric treatment column matched
nothing and the pipeline returned an empty result without raising.
"""

import numpy as np
import pandas as pd
import pytest

import samatch


def _numeric_treatment_frame(seed=7, n=300):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    frame["T"] = rng.choice([0, 1, 2], n)
    return frame


def test_numeric_treatment_column_matches_all_anchors():
    frame = _numeric_treatment_frame()
    n_anchor = int((frame["T"] == 0).sum())

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level=0
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level=0, top_m=5
    )

    assert len(search["anchor_rows"]) == n_anchor
    assert len(search["groups"]) == 2

    matched = samatch.sam_match(
        frame, search, X_vars=list("abc"), treatment_var="T"
    )

    assert len(matched["matched"]) > 0
    assert np.isfinite(matched["matching_rate"])


@pytest.mark.parametrize("anchor_level", [0, "0"])
def test_anchor_level_accepts_either_representation(anchor_level):
    frame = _numeric_treatment_frame()

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level=anchor_level
    )

    assert list(fit["gps"].columns) == ["0", "1", "2"]


def test_numeric_and_string_anchor_level_give_identical_results():
    frame = _numeric_treatment_frame(seed=3, n=200)

    numeric = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level=0
    )
    string = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level="0"
    )

    pd.testing.assert_frame_equal(numeric["gps"], string["gps"])


def test_absent_anchor_level_still_raises():
    frame = _numeric_treatment_frame()

    with pytest.raises(ValueError, match="not found in treatment variable"):
        samatch.estimate_gps_multinom(
            frame, X_vars=list("abc"), treatment_var="T", anchor_level=99
        )


def test_a_group_with_no_rows_raises_rather_than_matching_nothing():
    """A level present in the GPS but absent from `data` must not pass."""
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "T": ["A", "A", "B", "B"]}
    )
    # "C" is a GPS column but no subject carries it.
    gps = pd.DataFrame(
        {
            "A": [0.4, 0.4, 0.4, 0.4],
            "B": [0.4, 0.4, 0.4, 0.4],
            "C": [0.2, 0.2, 0.2, 0.2],
        }
    )

    with pytest.raises(ValueError, match="no rows found with T == 'C'"):
        samatch.gps_candidate_search(
            frame, gps, treatment_var="T", anchor_level="A", top_m=2
        )


def test_missing_treatment_column_is_reported():
    frame = _numeric_treatment_frame()

    with pytest.raises(ValueError, match="'absent' not found in data"):
        samatch.estimate_gps_multinom(
            frame, X_vars=list("abc"), treatment_var="absent", anchor_level=0
        )


def test_string_treatment_column_is_unaffected():
    """The existing string path must keep behaving exactly as before."""
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

    assert len(search["anchor_rows"]) == int(
        (data["treatment"] == anchor).sum()
    )


def test_build_group_distance_matrices_accepts_the_callers_own_labels():
    """
    The one public function that still compared raw labels. With a numeric
    treatment column it reported a level as absent while quoting it in the very
    form that is present, which is the symptom this module exists to prevent.
    """
    frame = _numeric_treatment_frame()

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level=0
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level=0, top_m=5
    )

    from_data = samatch.build_group_distance_matrices(
        frame, list("abc"), "T", search["anchor_rows"], [1, 2]
    )
    from_search = samatch.build_group_distance_matrices(
        frame, list("abc"), "T", search["anchor_rows"], search["groups"]
    )

    assert [matrix.shape for matrix in from_data["D"].values()] == [
        matrix.shape for matrix in from_search["D"].values()
    ]

    for numeric_level, string_level in ((1, "1"), (2, "2")):
        np.testing.assert_array_equal(
            from_data["D"][numeric_level], from_search["D"][string_level]
        )


def test_an_absent_level_is_reported_distinguishably():
    """
    The message has to show whether the level it looked for was a string, since
    that is exactly what the reader needs to check.
    """
    frame = _numeric_treatment_frame()

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level=0
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level=0, top_m=5
    )

    with pytest.raises(ValueError, match=r"no rows found with T == 9"):
        samatch.build_group_distance_matrices(
            frame, list("abc"), "T", search["anchor_rows"], [9]
        )
