"""
Correctness tests for the KD-tree candidate search.

The search is validated against an explicit brute-force reference rather than
against invariants, since the KD-tree rewrite has to reproduce the previous
full-distance-matrix behaviour exactly, including how equidistant candidates
are ordered.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

import samatch


def _brute_force_candidates(gps_used, treatment, anchor_level, groups, top_m):
    """Full-distance-matrix reference: nearest `top_m`, ties by row order."""
    anchor_rows = np.flatnonzero(treatment == anchor_level)
    x_anchor = gps_used[anchor_rows]

    reference = {}

    for group in groups:
        group_rows = np.flatnonzero(treatment == group)
        x_group = gps_used[group_rows]

        squared = ((x_anchor[:, None, :] - x_group[None, :, :]) ** 2).sum(-1)
        squared[squared < 0] = 0.0
        distances = np.sqrt(squared)

        m = min(top_m, len(group_rows))

        reference[group] = [
            group_rows[np.argsort(distances[i], kind="stable")[:m]]
            for i in range(len(anchor_rows))
        ]

    return reference


def _make_data(seed, duplicate_rows=0):
    rng = np.random.default_rng(seed)
    n = 300

    X = rng.normal(size=(n, 4))
    linear = X @ rng.normal(scale=0.4, size=(4, 3))
    probabilities = np.exp(linear)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    frame = pd.DataFrame(X, columns=list("abcd"))
    frame["T"] = ["ABC"[rng.choice(3, p=row)] for row in probabilities]

    if duplicate_rows:
        # Duplicated subjects give exactly equidistant candidates, which is
        # the only case where tie handling is observable.
        frame = pd.concat(
            [frame, frame.iloc[:duplicate_rows]], ignore_index=True
        )

    return frame


@pytest.mark.parametrize("duplicate_rows", [0, 120])
@pytest.mark.parametrize("gps_space", ["raw", "logit"])
@pytest.mark.parametrize("top_m", [1, 3, 10, 25])
def test_matches_brute_force_exactly(duplicate_rows, gps_space, top_m):
    frame = _make_data(seed=3, duplicate_rows=duplicate_rows)

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abcd"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame,
        fit["gps"],
        treatment_var="T",
        anchor_level="A",
        top_m=top_m,
        gps_space=gps_space,
    )

    gps_values = fit["gps"].to_numpy(dtype=float)
    gps_used = (
        logit(np.clip(gps_values, 1e-6, 1 - 1e-6))
        if gps_space == "logit"
        else gps_values
    )

    reference = _brute_force_candidates(
        gps_used,
        frame["T"].astype(str).to_numpy(),
        "A",
        search["groups"],
        top_m,
    )

    for group in search["groups"]:
        for i in range(len(search["anchor_rows"])):
            np.testing.assert_array_equal(
                search["candidates"][i][group], reference[group][i]
            )


def test_ties_are_broken_by_row_order():
    """With more tied candidates than slots, the lowest rows must win."""
    # Six identical comparator subjects, so every candidate is equidistant.
    frame = pd.DataFrame(
        {
            "a": [0.0] + [1.0] * 6,
            "b": [0.0] + [1.0] * 6,
            "T": ["A"] + ["B"] * 6,
        }
    )

    gps = pd.DataFrame(
        {"A": [0.5] * 7, "B": [0.5] * 7},
        index=frame.index,
    )

    search = samatch.gps_candidate_search(
        frame, gps, treatment_var="T", anchor_level="A", top_m=3
    )

    np.testing.assert_array_equal(search["candidates"][0]["B"], [1, 2, 3])


def test_top_m_larger_than_group_is_clamped():
    frame = _make_data(seed=1)

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abcd"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=10_000
    )

    treatment = frame["T"].astype(str).to_numpy()

    for group in search["groups"]:
        expected = int((treatment == group).sum())
        assert len(search["candidates"][0][group]) == expected


def test_candidates_are_within_the_requested_group():
    frame = _make_data(seed=2)

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abcd"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=5
    )

    treatment = frame["T"].astype(str).to_numpy()

    for group in search["groups"]:
        for candidate_rows in (entry[group] for entry in search["candidates"]):
            assert set(treatment[candidate_rows]) == {group}
