"""
Correctness tests for the `sam_match` greedy engine.

The production implementation is a lazy-deletion heap with a reverse index of
which anchors are relying on which comparator subject. That machinery exists
for speed, so it is checked against a direct, deliberately slow restatement of
what the algorithm is defined to do: repeatedly take the globally cheapest
still-available matched set.
"""

import numpy as np
import pandas as pd
import pytest

import samatch
from samatch._mahalanobis import (
    get_pooled_covariance,
    mahalanobis_distance_matrix,
)


def reference_greedy(data, search, X_vars, treatment_var):
    """Direct restatement of the matching rule. O(n^2), for testing only."""
    anchor_rows = np.asarray(search["anchor_rows"], dtype=int)
    groups = list(search["groups"])

    s_inv = get_pooled_covariance(data, X_vars, treatment_var)["S_inv"]
    X = data[X_vars].to_numpy(dtype=float)

    ranked = {}

    for i, anchor_row in enumerate(anchor_rows):
        for group in groups:
            rows = np.asarray(search["candidates"][i][group], dtype=int)
            distances = mahalanobis_distance_matrix(
                X[anchor_row : anchor_row + 1], X[rows], s_inv
            )[0]
            order = np.argsort(distances, kind="stable")
            ranked[i, group] = (rows[order], distances[order])

    used = {group: set() for group in groups}
    alive = set(range(len(anchor_rows)))
    matched_rows = []

    while alive:
        best = None

        for i in sorted(alive):
            total = 0.0
            pick = {}

            for group in groups:
                rows, distances = ranked[i, group]
                choice = next(
                    (
                        (int(row), float(distance))
                        for row, distance in zip(rows, distances)
                        if row not in used[group]
                    ),
                    None,
                )

                if choice is None:
                    total = None
                    break

                pick[group] = choice
                total += choice[1]

            if total is None:
                alive.discard(i)
                continue

            if best is None or total < best[0]:
                best = (total, i, pick)

        if best is None:
            break

        total, i, pick = best

        row = {
            "matched_set_id": len(matched_rows) + 1,
            "anchor": int(anchor_rows[i]),
        }

        for group in groups:
            row[group] = pick[group][0]
            row[f"dist_{group}"] = pick[group][1]

        row["loss"] = total
        matched_rows.append(row)

        alive.discard(i)

        for group in groups:
            used[group].add(pick[group][0])

    return pd.DataFrame(matched_rows)


def _make_data(seed, n=180):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n, 4))
    linear = X @ rng.normal(scale=0.5, size=(4, 3))
    probabilities = np.exp(linear)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    frame = pd.DataFrame(X, columns=list("abcd"))
    frame["T"] = ["ABC"[rng.choice(3, p=row)] for row in probabilities]
    return frame


def _run(frame, top_m=6):
    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abcd"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=top_m
    )
    matched = samatch.sam_match(
        frame, search, X_vars=list("abcd"), treatment_var="T"
    )
    return search, matched


@pytest.mark.parametrize("seed", range(5))
def test_matches_direct_greedy_reference(seed):
    frame = _make_data(seed)
    search, matched = _run(frame)

    expected = reference_greedy(frame, search, list("abcd"), "T")
    actual = matched["matched"]

    assert len(actual) == len(expected)

    for column in ["anchor", *search["groups"]]:
        np.testing.assert_array_equal(
            actual[column].to_numpy(), expected[column].to_numpy()
        )

    np.testing.assert_allclose(
        actual["loss"].to_numpy(), expected["loss"].to_numpy()
    )


@pytest.mark.parametrize("seed", range(3))
def test_no_comparator_subject_is_reused(seed):
    frame = _make_data(seed)
    search, matched = _run(frame)

    for column in ["anchor", *search["groups"]]:
        assert matched["matched"][column].is_unique


def test_losses_are_selected_in_ascending_order():
    """Matched sets are chosen globally cheapest first."""
    frame = _make_data(seed=0)
    _, matched = _run(frame)

    assert matched["matched"]["loss"].is_monotonic_increasing


def test_repeated_runs_are_identical():
    frame = _make_data(seed=0)

    _, first = _run(frame)
    _, second = _run(frame)

    pd.testing.assert_frame_equal(first["matched"], second["matched"])


def test_unmatched_anchors_account_for_the_remainder():
    frame = _make_data(seed=1)
    search, matched = _run(frame)

    n_anchor = len(search["anchor_rows"])

    assert len(matched["matched"]) + len(
        matched["unmatched_anchor_rows"]
    ) == n_anchor
    assert matched["matching_rate"] == pytest.approx(
        len(matched["matched"]) / n_anchor
    )
