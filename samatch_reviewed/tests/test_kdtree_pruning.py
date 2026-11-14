"""
Three-way matching prunes consumed subjects out of its KD-trees.

`match_3way` walked a tree built once over every subject, so once most of a
comparator group had been matched away each query still descended the consumed
neighbourhood to find the few survivors. Two things fix that: a per-node count
of what is still available, and rebuilding over the survivors when half of them
are gone. Neither may change a matching decision, which is what these check.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import samatch
from samatch._match_3way import (
    kdtree_build,
    kdtree_deactivate,
    kdtree_index,
    kdtree_nearest,
    kdtree_range,
)


def _three_group_frame(n, seed=5):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 6))
    frame = pd.DataFrame(values, columns=[f"X{i}" for i in range(1, 7)])

    linear = values[:, 0] * 0.5
    odds = np.exp(np.column_stack([linear * 0, linear, -linear]))
    odds /= odds.sum(axis=1, keepdims=True)

    frame["T"] = [
        "ABC"[np.searchsorted(np.cumsum(row), draw)]
        for row, draw in zip(odds, rng.random(n))
    ]
    return frame


def _run(frame, **kwargs):
    fit = samatch.estimate_gps_multinom(
        frame,
        X_vars=[f"X{i}" for i in range(1, 7)],
        treatment_var="T",
        anchor_level="A",
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=10
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return samatch.match_3way(
            frame, search, fit["gps"], treatment_var="T", **kwargs
        )


# The tree helpers ------------------------------------------------------------


def test_kdtree_build_rejects_input_it_cannot_index():
    with pytest.raises(ValueError, match="exactly two columns"):
        kdtree_build(np.zeros((5, 3)))

    with pytest.raises(ValueError, match="at least one point"):
        kdtree_build(np.zeros((0, 2)))


def test_node_counts_start_at_the_subtree_size():
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    tree = kdtree_build(coords)

    assert tree["n_active"] == 4
    assert tree["left"]["n_active"] + tree["right"]["n_active"] == 4


def test_deactivating_decrements_every_ancestor():
    coords = np.array([[float(i), 0.0] for i in range(8)])
    tree = kdtree_build(coords)
    leaf_of = kdtree_index(tree, len(coords))

    kdtree_deactivate(leaf_of[3])

    assert tree["n_active"] == 7
    assert leaf_of[3]["n_active"] == 0

    for point in (0, 1, 2, 4, 5, 6, 7):
        kdtree_deactivate(leaf_of[point])

    assert tree["n_active"] == 0


def test_index_reaches_every_point():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(37, 2))
    tree = kdtree_build(coords)
    leaf_of = kdtree_index(tree, len(coords))

    assert all(leaf is not None for leaf in leaf_of)
    assert sorted(leaf["point"] for leaf in leaf_of) == list(range(37))


def test_queries_agree_with_brute_force_after_deactivation():
    """
    Pruning may only remove points that could not have been returned. Integer
    coordinates are used so that exact ties occur.
    """
    rng = np.random.default_rng(0)

    for _ in range(150):
        n = int(rng.integers(1, 60))
        coords = rng.integers(0, 6, size=(n, 2)).astype(float)
        tree = kdtree_build(coords)
        leaf_of = kdtree_index(tree, n)
        active = np.ones(n, dtype=bool)

        for point in np.flatnonzero(rng.random(n) < 0.5):
            active[point] = False
            kdtree_deactivate(leaf_of[point])

        for _ in range(4):
            query = rng.integers(0, 6, size=2).astype(float)
            radius_sq = float(rng.random() * 12)

            distance_sq = ((coords - query) ** 2).sum(axis=1)

            assert sorted(
                kdtree_range(tree, coords, query, radius_sq, active)
            ) == sorted(
                np.flatnonzero(active & (distance_sq < radius_sq)).tolist()
            )

            nearest = kdtree_nearest(tree, coords, query, active)

            if active.any():
                assert distance_sq[nearest] == distance_sq[active].min()
            else:
                assert nearest is None


def test_pruning_does_not_depend_on_the_caller_reporting_consumption():
    """
    A caller that only flips `active` forfeits the speedup but must still get
    the right answer, since the counts then never drop.
    """
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(40, 2))
    tree = kdtree_build(coords)

    active = np.ones(40, dtype=bool)
    active[rng.random(40) < 0.6] = False

    query = np.array([0.1, -0.2])
    distance_sq = ((coords - query) ** 2).sum(axis=1)
    nearest = kdtree_nearest(tree, coords, query, active)

    assert distance_sq[nearest] == distance_sq[active].min()


# The engine ------------------------------------------------------------------


def test_rebuilding_preserves_the_matched_sets():
    """
    A rebuild reindexes the survivors, so nothing about which trio is chosen may
    depend on how many rebuilds happened. Running the same data twice would not
    catch that; a size large enough to trigger several rebuilds is the point.
    """
    frame = _three_group_frame(1800)

    first = _run(frame, top_n=10)["matched"]
    second = _run(frame, top_n=10)["matched"]

    pd.testing.assert_frame_equal(first, second)
    assert len(first) > 300, "too small to exercise a rebuild"


def test_every_subject_is_used_at_most_once():
    """Matching without replacement, across enough data to force rebuilds."""
    frame = _three_group_frame(1800)
    result = _run(frame, top_n=10)
    matched = result["matched"]

    for column in ("anchor", "B", "C"):
        values = matched[column].to_numpy()
        assert len(np.unique(values)) == len(values), column

    # No subject may appear in two different roles either.
    everything = np.concatenate(
        [matched[column].to_numpy() for column in ("anchor", "B", "C")]
    )
    assert len(np.unique(everything)) == len(everything)


def test_selected_trios_respect_the_caliper():
    frame = _three_group_frame(1800)
    result = _run(frame, top_n=10)

    assert (
        result["matched"]["rassen_perimeter"].to_numpy()
        <= result["caliper"] + 1e-12
    ).all()
