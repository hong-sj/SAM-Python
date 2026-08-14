"""
Covariates are validated before they reach the linear algebra.

`numpy.linalg.inv` does not raise on `NaN` input -- it returns an all-`NaN`
matrix -- so the singular-matrix fallback never fired for missing data and a
single missing cell silently destroyed every Mahalanobis distance.
"""

import numpy as np
import pandas as pd
import pytest

import samatch


@pytest.fixture
def fitted():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    frame["T"] = rng.choice(list("ABC"), 300)

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=5
    )
    return frame, fit, search


def test_numpy_inv_does_not_raise_on_nan():
    """The premise: this is why catching LinAlgError was not enough."""
    result = np.linalg.inv(np.array([[1.0, np.nan], [np.nan, 2.0]]))
    assert np.isnan(result).all()


def test_one_missing_value_raises_instead_of_emptying_the_match(fitted):
    frame, _, search = fitted

    corrupted = frame.copy()
    corrupted.loc[5, "c"] = np.nan

    with pytest.raises(ValueError, match=r"non-finite values: c \(1 rows\)"):
        samatch.sam_match(
            corrupted, search, X_vars=list("abc"), treatment_var="T"
        )


def test_infinite_values_are_rejected(fitted):
    frame, _, _ = fitted

    corrupted = frame.copy()
    corrupted.loc[7, "a"] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        samatch.get_pooled_covariance(corrupted, list("abc"), "T")


def test_every_offending_column_is_named(fitted):
    frame, _, _ = fitted

    corrupted = frame.copy()
    corrupted.loc[1, "a"] = np.nan
    corrupted.loc[2, "b"] = np.nan
    corrupted.loc[3, "b"] = np.nan

    with pytest.raises(ValueError) as excinfo:
        samatch.get_pooled_covariance(corrupted, list("abc"), "T")

    message = str(excinfo.value)
    assert "a (1 rows)" in message
    assert "b (2 rows)" in message
    assert "c" not in message.split("values:")[1].split(".")[0]


def test_categorical_covariates_are_rejected_with_guidance(fitted):
    frame, _, _ = fitted

    frame = frame.copy()
    frame["sex"] = "M"

    with pytest.raises(ValueError, match="not numeric: sex"):
        samatch.estimate_gps_multinom(
            frame,
            X_vars=list("abc") + ["sex"],
            treatment_var="T",
            anchor_level="A",
        )


def test_missing_covariate_column_is_named(fitted):
    frame, _, _ = fitted

    with pytest.raises(ValueError, match="not found in data: nope"):
        samatch.estimate_gps_multinom(
            frame,
            X_vars=list("abc") + ["nope"],
            treatment_var="T",
            anchor_level="A",
        )


def test_duplicated_covariates_are_rejected_explicitly(fitted):
    frame, _, _ = fitted

    with pytest.raises(
        ValueError,
        match=r"Duplicated covariate column\(s\).*a",
    ):
        samatch.get_pooled_covariance(
            frame,
            X_vars=["a", "b", "a"],
            treatment_var="T",
        )


def test_singular_but_finite_covariance_still_uses_pinv(fitted):
    """The fallback must remain reachable for genuine rank deficiency."""
    frame, _, _ = fitted

    frame = frame.copy()
    frame["duplicate"] = frame["a"]

    with pytest.warns(RuntimeWarning, match="singular"):
        pooled = samatch.get_pooled_covariance(
            frame, list("abc") + ["duplicate"], "T"
        )

    assert np.isfinite(pooled["S_inv"]).all()


def test_insufficient_degrees_of_freedom_raises():
    """One subject per group leaves no residual variance to pool."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "T": ["A", "B"]})

    with pytest.raises(ValueError, match="degrees of freedom"):
        samatch.get_pooled_covariance(frame, ["a", "b"], "T")


def test_clean_covariates_are_unaffected(fitted):
    frame, _, search = fitted

    matched = samatch.sam_match(
        frame, search, X_vars=list("abc"), treatment_var="T"
    )

    assert len(matched["matched"]) > 0
    assert np.isfinite(matched["matched"]["loss"].to_numpy()).all()
