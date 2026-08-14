"""
Ground-truth tests on examples small enough to work out by hand.

Every other matching test checks structure (shapes, uniqueness, bounds) or
compares against another implementation. These pin the actual numbers, so a
change in the distance metric, the pooled covariance, or the greedy selection
rule has to be deliberate.
"""

import numpy as np
import pandas as pd
import pytest

import samatch

# Six subjects on a single covariate, two per treatment group:
#
#   row:   0     1     2     3     4     5
#   x:     0     3     1     4     2     5
#   T:     A     A     B     B     C     C
#
# Every group has the same spread, so the pooled within-group covariance is
# easy to write down:
#
#   each group contributes 2 * 1.5^2 = 4.5
#   S_within = 3 * 4.5 = 13.5,  df = 6 - 3 = 3,  S = 4.5
#
# With one covariate the Mahalanobis distance is therefore |dx| / sqrt(4.5).
UNIT = 1.0 / np.sqrt(4.5)


@pytest.fixture
def tiny():
    frame = pd.DataFrame(
        {
            "x": [0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            "T": ["A", "A", "B", "B", "C", "C"],
        }
    )

    # Flat GPS: the candidate search must not filter anything out, so the
    # matching result depends only on the Mahalanobis distances.
    gps = pd.DataFrame(
        {"A": [0.34] * 6, "B": [0.33] * 6, "C": [0.33] * 6}
    )

    search = samatch.gps_candidate_search(
        frame, gps, treatment_var="T", anchor_level="A", top_m=2
    )
    return frame, search


def test_pooled_covariance_is_the_hand_computed_value(tiny):
    frame, _ = tiny

    pooled = samatch.get_pooled_covariance(frame, ["x"], "T")

    np.testing.assert_allclose(pooled["S"], [[4.5]])
    np.testing.assert_allclose(pooled["S_inv"], [[1.0 / 4.5]])


def test_mahalanobis_distance_is_scaled_absolute_difference(tiny):
    frame, _ = tiny

    s_inv = samatch.get_pooled_covariance(frame, ["x"], "T")["S_inv"]

    distances = samatch.mahalanobis_distance_matrix([[0.0]], [[1.0], [5.0]], s_inv)

    np.testing.assert_allclose(distances, [[UNIT, 5 * UNIT]])


def test_matched_sets_and_distances_are_exactly_as_derived(tiny):
    """
    Anchor x=3 is the cheaper of the two anchors and is matched first.

        anchor x=3 -> B x=4 (1 unit), C x=2 (1 unit)   loss 2 units
        anchor x=0 -> B x=1 (1 unit), C x=5 (5 units)  loss 6 units

    Anchor x=0 would have preferred C x=2, which is one unit away rather than
    five, but the cheaper set consumed it. That is the point of selecting
    matched sets globally cheapest first rather than anchor by anchor.
    """
    frame, search = tiny

    matched = samatch.sam_match(frame, search, X_vars=["x"], treatment_var="T")
    result = matched["matched"]

    assert len(result) == 2

    np.testing.assert_array_equal(result["matched_set_id"], [1, 2])
    np.testing.assert_array_equal(result["anchor"], [1, 0])
    np.testing.assert_array_equal(result["B"], [3, 2])
    np.testing.assert_array_equal(result["C"], [4, 5])

    np.testing.assert_allclose(result["dist_B"], [UNIT, UNIT])
    np.testing.assert_allclose(result["dist_C"], [UNIT, 5 * UNIT])
    np.testing.assert_allclose(result["loss"], [2 * UNIT, 6 * UNIT])

    assert matched["matching_rate"] == 1.0
    assert matched["max_possible_rate"] == 1.0
    assert len(matched["unmatched_anchor_rows"]) == 0


def test_candidate_search_orders_by_gps_distance():
    """With one GPS coordinate the ordering is readable by inspection."""
    frame = pd.DataFrame({"T": ["A", "B", "B", "B"]})

    # Anchor at 0.10; comparators at 0.50, 0.20, 0.90 in the "B" coordinate.
    gps = pd.DataFrame(
        {
            "A": [0.90, 0.50, 0.80, 0.10],
            "B": [0.10, 0.50, 0.20, 0.90],
        }
    )

    search = samatch.gps_candidate_search(
        frame, gps, treatment_var="T", anchor_level="A", top_m=3
    )

    # Distance from the anchor row runs 0.20 < 0.50 < 0.90, i.e. rows 2, 1, 3.
    np.testing.assert_array_equal(search["candidates"][0]["B"], [2, 1, 3])


def test_top_m_keeps_only_the_nearest_candidates():
    frame = pd.DataFrame({"T": ["A", "B", "B", "B"]})
    gps = pd.DataFrame(
        {
            "A": [0.90, 0.50, 0.80, 0.10],
            "B": [0.10, 0.50, 0.20, 0.90],
        }
    )

    search = samatch.gps_candidate_search(
        frame, gps, treatment_var="T", anchor_level="A", top_m=1
    )

    np.testing.assert_array_equal(search["candidates"][0]["B"], [2])


def test_smd_of_a_perfectly_balanced_match_is_zero():
    """Identical covariate values in both arms give an SMD of exactly zero."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "T": ["A", "A", "B", "B", "C", "C"],
        }
    )
    matched = pd.DataFrame(
        [
            {"matched_set_id": 1, "anchor": 0, "B": 2, "C": 4, "loss": 0.0},
            {"matched_set_id": 2, "anchor": 1, "B": 3, "C": 5, "loss": 0.0},
        ]
    )

    balance = samatch.compute_smd_balance(frame, matched, ["x"], ["B", "C"])

    np.testing.assert_allclose(balance["by_covariate"]["smd"], [0.0, 0.0])
    assert balance["by_covariate"]["smd_defined"].all()


def test_smd_matches_the_hand_computed_value():
    """
    Anchor x = {0, 2}: mean 1, var 2. Comparator x = {3, 5}: mean 4, var 2.

        SMD = (1 - 4) / sqrt((2 + 2) / 2) = -3 / sqrt(2)
    """
    frame = pd.DataFrame(
        {
            "x": [0.0, 2.0, 3.0, 5.0],
            "T": ["A", "A", "B", "B"],
        }
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


def test_effective_sample_size_matches_the_hand_computed_value():
    """weights (1, 1, 2) -> (1+1+2)^2 / (1+1+4) = 16 / 6."""
    assert samatch.compute_effective_sample_size(
        np.array([1.0, 1.0, 2.0])
    ) == pytest.approx(16.0 / 6.0)


def test_single_covariate_matching_runs():
    """A 1x1 covariance is a real edge case for the matrix inverse."""
    frame = pd.DataFrame(
        {"x": [0.0, 3.0, 1.0, 4.0], "T": ["A", "A", "B", "B"]}
    )
    gps = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4})

    search = samatch.gps_candidate_search(
        frame, gps, treatment_var="T", anchor_level="A", top_m=2
    )
    matched = samatch.sam_match(
        frame, search, X_vars=["x"], treatment_var="T"
    )

    assert len(matched["matched"]) == 2
    assert matched["matching_rate"] == 1.0
