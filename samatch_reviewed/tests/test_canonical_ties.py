"""
Tie-breaking that does not depend on rounding noise or index internals.

The fixtures under ``golden/ties_*`` are shared verbatim with the R package's
test suite: both implementations must reproduce ``ties_expected.csv`` exactly.
The data has only 72 distinct covariate profiles, so mathematically equal
distances, losses and perimeters are everywhere -- before the canonical
tie-break, R and Python (and two releases of either) could pick different
matches from the same input.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from samatch import gps_candidate_search, match_3way, sam_match
from samatch._match_3way import _ActiveGroup
from samatch._match_common import order_key

GOLDEN = pathlib.Path(__file__).parent / "golden"
X_VARS = [f"x{j}" for j in range(5)]


@pytest.fixture(scope="module")
def ties():
    data = pd.read_csv(GOLDEN / "ties_data.csv")
    gps = pd.read_csv(GOLDEN / "ties_gps.csv")
    expected = pd.read_csv(GOLDEN / "ties_expected.csv")
    return data, gps, expected


@pytest.mark.parametrize("gps_space", ["raw", "logit"])
@pytest.mark.parametrize("method", ["sam", "3way"])
def test_tied_data_matches_the_shared_fixture(ties, method, gps_space):
    data, gps, expected = ties
    search = gps_candidate_search(
        data, gps, treatment_var="T", anchor_level="A", top_m=5,
        gps_space=gps_space,
    )

    if method == "sam":
        result = sam_match(data, search, X_vars=X_VARS, treatment_var="T")
    else:
        result = match_3way(
            data, search, gps, treatment_var="T", gps_space=gps_space,
            top_n=5,
        )

    want = expected[
        (expected["method"] == method) & (expected["gps_space"] == gps_space)
    ]
    got = result["matched"][["anchor", "B", "C"]].to_numpy()

    np.testing.assert_array_equal(got, want[["anchor", "B", "C"]].to_numpy())


def test_order_key_absorbs_rounding_noise_but_not_real_differences():
    assert order_key(1.0) == order_key(1.0 + 4e-16)
    assert order_key(0.1 + 0.2) == order_key(0.3)
    assert order_key(1.0) < order_key(1.0 + 1e-9)
    # Round-half-even, as R's round() with no digits.
    assert order_key(0.25e-10) == 0.0
    assert order_key(0.75e-10) == 1.0


def test_sam_match_is_unaffected_by_a_large_covariate_offset():
    # Adding a constant to a covariate changes no Mahalanobis distance. The
    # expanded form |a|^2 + |b|^2 - 2ab loses about eps * offset^2 / var of
    # accuracy to cancellation -- here around 1e-4 -- while differencing
    # whitened rows is exact to rounding.
    rng = np.random.default_rng(3)
    n = 900
    data = pd.DataFrame(
        {f"x{j}": rng.normal(size=n) for j in range(3)}
    )
    data["T"] = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    gps = pd.DataFrame(
        rng.dirichlet([5, 5, 5], size=n), columns=["A", "B", "C"]
    )
    x_vars = ["x0", "x1", "x2"]

    search = gps_candidate_search(data, gps, treatment_var="T", anchor_level="A")
    base = sam_match(data, search, X_vars=x_vars, treatment_var="T")["matched"]

    shifted_data = data.copy()
    shifted_data["x0"] = shifted_data["x0"] + 1e6
    shifted_search = gps_candidate_search(
        shifted_data, gps, treatment_var="T", anchor_level="A"
    )
    shifted = sam_match(
        shifted_data, shifted_search, X_vars=x_vars, treatment_var="T"
    )["matched"]

    pd.testing.assert_frame_equal(
        base[["anchor", "B", "C"]], shifted[["anchor", "B", "C"]]
    )
    np.testing.assert_allclose(shifted["loss"], base["loss"], atol=1e-8)


def test_nearest_prefers_the_lowest_row_among_equidistant_subjects():
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    group = _ActiveGroup(coords)

    assert group.nearest(np.array([1.0, 1.0])) == 1
    # Equidistant from (1.5, 1.5): rows 1, 2 and 3. The lowest wins.
    assert group.nearest(np.array([1.5, 1.5])) == 1
    np.testing.assert_array_equal(group.nearest_many(coords), [0, 1, 1, 3])

    group.deactivate(1)
    assert group.nearest(np.array([1.0, 1.0])) == 2

    group.deactivate(2)
    assert group.nearest(np.array([1.5, 1.5])) == 3


def test_within_is_strict_and_sorted_after_consumption():
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.5]])
    group = _ActiveGroup(coords)
    query = np.array([0.0, 0.0])

    # The boundary point at distance exactly 1 is excluded.
    np.testing.assert_array_equal(group.within(query, 1.0), [0, 1, 3])

    for point in (0, 3):
        group.deactivate(point)  # the second triggers a rebuild

    np.testing.assert_array_equal(group.within(query, 1.0), [1])
    np.testing.assert_array_equal(group.within(query, 1.0 + 1e-12), [1, 2])
