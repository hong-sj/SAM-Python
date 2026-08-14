"""GPS inputs remain aligned with subjects throughout the pipeline."""

import warnings

import numpy as np
import pandas as pd
import pytest

import samatch
from samatch._gps import _fit_unregularized_multinomial_logit


def _small_data_and_gps():
    data = pd.DataFrame({"T": ["A", "B", "B"]}, index=[10, 11, 12])
    gps = pd.DataFrame(
        {"A": [0.9, 0.8, 0.1], "B": [0.1, 0.2, 0.9]},
        index=data.index,
    )
    return data, gps


@pytest.mark.parametrize("function", ["candidate_search", "weighting"])
def test_reordered_gps_rows_are_rejected(function):
    data, gps = _small_data_and_gps()
    reordered = gps.iloc[[0, 2, 1]]

    with pytest.raises(ValueError, match="identical indices in the same order"):
        if function == "candidate_search":
            samatch.gps_candidate_search(data, reordered, anchor_level="A")
        else:
            samatch.compute_balancing_weights(data, gps=reordered)


@pytest.mark.parametrize(
    "values, message",
    [
        ([[0.9, 0.1], [np.nan, np.nan], [0.1, 0.9]], "finite"),
        ([[0.9, 0.1], [-0.1, 1.1], [0.1, 0.9]], r"\[0, 1\]"),
        ([[0.9, 0.1], [0.4, 0.4], [0.1, 0.9]], "sum to 1"),
    ],
)
def test_invalid_gps_values_are_rejected(values, message):
    data, gps = _small_data_and_gps()
    invalid = pd.DataFrame(values, columns=gps.columns, index=gps.index)

    with pytest.raises(ValueError, match=message):
        samatch.gps_candidate_search(data, invalid, anchor_level="A")


def test_missing_treatment_group_in_gps_is_rejected():
    data = pd.DataFrame({"T": ["A", "B", "C"]})
    gps = pd.DataFrame(
        {"A": [0.8, 0.2, 0.5], "B": [0.2, 0.8, 0.5]},
        index=data.index,
    )

    with pytest.raises(ValueError, match="not found in gps: C"):
        samatch.gps_candidate_search(data, gps, anchor_level="A")


def test_missing_treatment_labels_are_rejected_before_string_conversion():
    data, gps = _small_data_and_gps()
    data = data.copy()
    data.loc[11, "T"] = np.nan

    with pytest.raises(ValueError, match="missing treatment labels"):
        samatch.gps_candidate_search(data, gps, anchor_level="A")


def test_evaluation_rejects_a_different_gps_with_the_same_index():
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "x": rng.normal(size=30),
            "T": np.repeat(["A", "B", "C"], 10),
        }
    )
    fit = samatch.estimate_gps_multinom(data, X_vars=["x"], anchor_level="A")
    search = samatch.gps_candidate_search(data, fit["gps"], anchor_level="A")
    matched = samatch.sam_match(data, search, X_vars=["x"])

    changed = fit["gps"].copy()
    changed.iloc[[0, 1]] = changed.iloc[[1, 0]].to_numpy()

    with pytest.raises(ValueError, match="does not match the gps"):
        samatch.sam_evaluate(
            data,
            search,
            matched,
            changed,
            X_vars=["x"],
        )


def test_regularized_fallback_warning_is_not_swallowed(monkeypatch):
    class LegacyLogisticRegression:
        def __init__(self, **kwargs):
            if "penalty" in kwargs:
                raise TypeError("unexpected keyword argument 'penalty'")
            self.n_iter_ = np.array([1])

        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "samatch._gps.LogisticRegression", LegacyLogisticRegression
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["A", "A", "B", "B"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit_unregularized_multinomial_logit(X, y)

    assert any("falling back" in str(warning.message) for warning in caught)
