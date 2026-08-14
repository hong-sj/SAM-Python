"""
Three-way nearest-neighbor propensity score matching.

Implements the three-treatment-group matching approach of Rassen et al.
using a two-dimensional propensity score space, KD-tree candidate search,
a perimeter-based caliper, and global greedy selection.
"""

import heapq
import math
import warnings

import numpy as np
import pandas as pd

from ._match_common import (
    build_matched_frame,
    max_possible_rate,
    summarize_matching,
    transform_ps,
)
from ._validate import (
    check_data_fingerprint,
    check_gps_fingerprint,
    require_positive_int,
    treatment_labels,
    treatment_level,
    validate_gps,
)


def calc_caliper_3way(ps_used, treatment_var_values):
    """
    Compute the data-driven caliper for three-way matching.

    Parameters
    ----------
    ps_used : array-like of shape (n, 2)
        Two propensity score coordinates used for matching.
    treatment_var_values : array-like of length n
        Treatment-group labels.

    Returns
    -------
    float
        Perimeter-scale caliper.
    """
    ps_used = np.asarray(ps_used, dtype=float)
    treatment_var_values = np.asarray(treatment_var_values)

    if ps_used.ndim != 2 or ps_used.shape[1] != 2:
        raise ValueError("`ps_used` must have exactly two columns.")

    if ps_used.shape[0] != len(treatment_var_values):
        raise ValueError(
            "`ps_used` and `treatment_var_values` must have the same "
            "number of rows."
        )

    groups = np.unique(treatment_var_values)

    if len(groups) != 3:
        raise ValueError(
            "`treatment_var_values` must contain exactly three treatment groups."
        )

    if not np.isfinite(ps_used).all():
        raise ValueError("`ps_used` must contain only finite values.")

    too_small = [
        str(group)
        for group in groups
        if np.sum(treatment_var_values == group) < 2
    ]

    if too_small:
        raise ValueError(
            "The automatic three-way caliper requires at least two subjects "
            "in every treatment group; too few in: " + ", ".join(too_small)
        )

    var_by_group = np.array(
        [
            [
                ps_used[treatment_var_values == group, j].var(ddof=1)
                for j in range(2)
            ]
            for group in groups
        ]
    )

    return float(0.6 * np.sqrt(var_by_group.mean(axis=1).sum() / 3))


# KD-tree utilities -------------------------------------------------------------


def _distance_sq_2d(point_1, point_2):
    """Return a squared 2D distance without allocating temporary arrays."""
    delta_0 = point_1[0] - point_2[0]
    delta_1 = point_1[1] - point_2[1]
    return float(delta_0 * delta_0 + delta_1 * delta_1)


def kdtree_build(coords, idx=None, depth=0):
    """
    Build a static two-dimensional KD-tree.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Two-dimensional coordinates.
    idx : array-like of int, optional
        Row indices included in the current subtree.
    depth : int, default=0
        Current recursion depth.

    Returns
    -------
    dict
        KD-tree node. Each node carries ``n_active``, the number of points in
        its subtree that are still available; see `kdtree_deactivate()`.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must have exactly two columns.")

    if idx is None:
        idx = np.arange(coords.shape[0])

    if len(idx) == 0:
        raise ValueError("a KD-tree needs at least one point.")

    # Coercion and validation happen once here instead of at each of the ~2n
    # recursive calls, and the split keys are read from plain lists. On subsets
    # this small the per-node `np.asarray`, fancy index and `argsort` were
    # almost entirely dispatch overhead.
    columns = (coords[:, 0].tolist(), coords[:, 1].tolist())

    return _kdtree_build_node(columns, np.asarray(idx).tolist(), depth)


def _kdtree_build_node(columns, idx, depth):
    """Recursive half of `kdtree_build()`, over pre-validated lists."""
    n = len(idx)

    if n == 1:
        return {
            "leaf": True,
            "point": idx[0],
            "n_active": 1,
        }

    dim = depth % 2
    key = columns[dim]

    # `sorted` is stable, as `argsort(kind="stable")` was, and `idx` arrives in
    # the parent's order in both versions -- so tied coordinates keep the same
    # relative order and the tree is identical.
    ordered_idx = sorted(idx, key=key.__getitem__)

    mid = n // 2
    left_idx = ordered_idx[:mid]
    right_idx = ordered_idx[mid:]

    return {
        "leaf": False,
        "split_dim": dim,
        "split_val": key[right_idx[0]],
        "n_active": n,
        "left": _kdtree_build_node(columns, left_idx, depth + 1),
        "right": _kdtree_build_node(columns, right_idx, depth + 1),
    }


def kdtree_index(node, n_points):
    """
    Link each point to its leaf so that consumption can be propagated upward.

    Parameters
    ----------
    node : dict
        Root of a tree from `kdtree_build()`.
    n_points : int
        Number of points the tree was built over.

    Returns
    -------
    list
        ``leaf_of[point]`` is the leaf node holding that point.

    Notes
    -----
    Descending to a point by comparing against ``split_val`` is not reliable:
    the build splits a stable ordering at its midpoint, so a point whose
    coordinate equals the split value may sit on either side. Recording the
    leaves once, together with a parent link, gives an unambiguous path.
    """
    leaf_of = [None] * n_points
    stack = [(node, None)]

    while stack:
        current, parent = stack.pop()
        current["parent"] = parent

        if current["leaf"]:
            leaf_of[current["point"]] = current
        else:
            stack.append((current["left"], current))
            stack.append((current["right"], current))

    return leaf_of


def kdtree_deactivate(leaf):
    """
    Mark one point as consumed, so its exhausted ancestors can be skipped.

    Parameters
    ----------
    leaf : dict
        Leaf node for the consumed point, from `kdtree_index()`.

    Notes
    -----
    Must be called exactly once per point, paired with setting that point's
    entry in the caller's ``active`` array to False. `kdtree_nearest()` and
    `kdtree_range()` still consult ``active`` at the leaves, so a caller that
    skips this only forfeits the pruning -- results are unaffected either way.
    Without it, a query late in the matching loop walks a tree that is mostly
    consumed, which is what made three-way matching grow superlinearly.
    """
    node = leaf

    while node is not None:
        node["n_active"] -= 1
        node = node["parent"]


def kdtree_nearest(node, coords, query, active):
    """
    Find the nearest active point in a KD-tree.

    Parameters
    ----------
    node : dict
        KD-tree node.
    coords : numpy.ndarray
        Coordinate matrix used to construct the tree.
    query : array-like of shape (2,)
        Query coordinate.
    active : array-like of bool
        Availability indicator for each point.

    Returns
    -------
    int or None
        Index of the nearest active point, or None if none is available.
    """
    # A subtree with nothing left in it cannot hold the answer. This is only
    # informative for callers that report consumption via kdtree_deactivate();
    # for the rest the count stays at its initial value and the walk proceeds
    # exactly as before.
    if node["n_active"] == 0:
        return None

    if node["leaf"]:
        point = node["point"]
        return point if active[point] else None

    dim = node["split_dim"]
    gap = query[dim] - node["split_val"]

    if gap <= 0:
        primary = node["left"]
        other = node["right"]
    else:
        primary = node["right"]
        other = node["left"]

    best = kdtree_nearest(primary, coords, query, active)

    if best is None:
        best_distance_sq = float("inf")
    else:
        best_distance_sq = _distance_sq_2d(coords[best], query)

    if gap * gap < best_distance_sq:
        candidate = kdtree_nearest(other, coords, query, active)

        if candidate is not None:
            candidate_distance_sq = _distance_sq_2d(
                coords[candidate], query
            )

            if candidate_distance_sq < best_distance_sq:
                best = candidate

    return best


def kdtree_range(node, coords, query, radius2, active):
    """
    Find active points strictly within a KD-tree search radius.

    Parameters
    ----------
    node : dict
        KD-tree node.
    coords : numpy.ndarray
        Coordinate matrix used to construct the tree.
    query : array-like of shape (2,)
        Query coordinate.
    radius2 : float
        Squared search radius.
    active : array-like of bool
        Availability indicator for each point.

    Returns
    -------
    list of int
        Indices of active points within the search radius.

    Notes
    -----
    A subtree with nothing available in it is skipped. That only removes points
    which could not have been returned, so the result -- and its order -- is
    the same as an exhaustive scan.

    Pruning on a per-node bounding box was tried as well and removed: it cut
    only 8% of visited nodes, because the subtree the query descends into always
    contains the query and so is never outside the radius, while the test cost
    applies to every node visited. Reducing the node count further needs a
    C-level tree, not a tighter bound here.
    """
    # Points are appended to one accumulator rather than each level returning a
    # fresh list for its parent to concatenate. Descent order is unchanged --
    # primary subtree, then the other -- so the order points arrive in, which
    # the perimeter greedy tie-breaks on, is exactly as before.
    found = []
    _kdtree_range_into(node, coords, query, radius2, active, found)
    return found


def _kdtree_range_into(node, coords, query, radius2, active, found):
    """Append the active in-radius points of one subtree to `found`."""
    if node["n_active"] == 0:
        return

    if node["leaf"]:
        point = node["point"]

        if not active[point]:
            return

        distance_sq = _distance_sq_2d(coords[point], query)

        # Boundary points are excluded by design.
        if distance_sq < radius2:
            found.append(point)

        return

    dim = node["split_dim"]
    gap = query[dim] - node["split_val"]

    if gap <= 0:
        primary = node["left"]
        other = node["right"]
    else:
        primary = node["right"]
        other = node["left"]

    _kdtree_range_into(primary, coords, query, radius2, active, found)

    if gap * gap < radius2:
        _kdtree_range_into(other, coords, query, radius2, active, found)


# Three-way matching -----------------------------------------------------------


def match_3way(
    data,
    search,
    gps,
    X_vars=None,
    treatment_var="T",
    caliper="auto",
    gps_space="raw",
    top_n=10,
    reference_level=None,
    ps_space=None,
):
    """
    Perform three-way nearest-neighbor propensity score matching.

    The smallest treatment group is used as the search base. Each subject
    in that group is matched to one subject from each of the other two
    groups using a two-dimensional propensity score space. Candidate
    trios are ranked by triangle perimeter and selected globally without
    replacement.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment variable.
    search : dict
        Output from `gps_candidate_search()`. The anchor rows and comparator
        group labels are used to define the three treatment groups.
    gps : pandas.DataFrame
        Generalized propensity score matrix.
    X_vars : list of str, optional
        Included for interface compatibility with `sam_match()`. Not used by
        the three-way propensity score matching algorithm; passing it emits a
        `UserWarning`.
    treatment_var : str, default="T"
        Name of the treatment variable.
    caliper : {"auto"} or float, default="auto"
        Perimeter-scale caliper. ``"auto"`` uses `calc_caliper_3way()`.
    gps_space : {"raw", "logit"}, default="raw"
        Propensity score scale used for matching. Named to match
        `gps_candidate_search()`.
    top_n : int, default=10
        Maximum number of candidate trios retained per search-base subject
        during each candidate-generation step.
    reference_level : str, optional
        Treatment group whose GPS column is omitted when constructing the
        two-dimensional propensity score space. Defaults to the last group.
    ps_space : {"raw", "logit"}, optional
        Deprecated alias for `gps_space`.

    Returns
    -------
    dict
        Dictionary containing:

        - ``matched``: matched trios and distance information.
        - ``unmatched_anchor_rows``: unmatched anchor row indices.
        - ``matching_rate``: proportion of anchor subjects matched.
        - ``max_possible_rate``: the highest rate the group sizes allow.
        - ``caliper``: caliper used for matching.
        - ``red_level``: smallest treatment group used as the search base.
        - ``reference_level``: treatment group omitted from PS coordinates.
    """
    # X_vars is retained for interface compatibility with sam_match(). This
    # algorithm matches in propensity score space only, so passing covariates
    # here has no effect and is worth saying out loud.
    if X_vars is not None:
        warnings.warn(
            "match_3way() ignores X_vars: three-way matching operates in "
            "propensity score space only. Covariates influence the result "
            "through estimate_gps_multinom() instead.",
            UserWarning,
            stacklevel=2,
        )

    if ps_space is not None:
        warnings.warn(
            "`ps_space` is deprecated; use `gps_space` instead, which is the "
            "name gps_candidate_search() already uses for the same option.",
            DeprecationWarning,
            stacklevel=2,
        )
        gps_space = ps_space

    if gps_space not in ("raw", "logit"):
        raise ValueError('gps_space must be "raw" or "logit"')

    if treatment_var not in data.columns:
        raise ValueError(f"'{treatment_var}' not found in data")

    check_data_fingerprint(search, data, treatment_var, "match_3way")
    validate_gps(data, gps, treatment_var, "match_3way")
    check_gps_fingerprint(search, gps, "match_3way")

    top_n = require_positive_int(top_n, "top_n")

    groups = list(search["groups"])

    if len(groups) != 2:
        raise ValueError(
            "match_3way() requires exactly three treatment groups: "
            "one anchor group and two comparator groups."
        )

    anchor_rows = np.asarray(search["anchor_rows"], dtype=int)
    treatment = treatment_labels(data, treatment_var)

    anchor_levels = np.unique(treatment[anchor_rows])

    if len(anchor_levels) != 1:
        raise ValueError(
            "`search['anchor_rows']` must contain subjects from "
            "a single treatment group."
        )

    anchor_level = str(anchor_levels[0])
    all_levels = [anchor_level, *groups]

    missing_gps_levels = [
        level for level in all_levels if level not in gps.columns
    ]

    if missing_gps_levels:
        raise ValueError(
            "Treatment group(s) not found in gps: "
            + ", ".join(missing_gps_levels)
        )

    if reference_level is None:
        reference_level = all_levels[-1]
    else:
        reference_level = treatment_level(reference_level)

    if reference_level not in all_levels:
        raise ValueError(
            "`reference_level` must be one of the three treatment groups."
        )

    ps_levels = [
        level for level in all_levels if level != reference_level
    ]

    ps_used = transform_ps(gps[ps_levels].to_numpy(dtype=float), gps_space)

    if isinstance(caliper, str):
        if caliper != "auto":
            raise ValueError(
                '`caliper` must be "auto" or a positive numeric value.'
            )

        caliper = calc_caliper_3way(ps_used, treatment)

    caliper = float(caliper)

    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError("`caliper` must be finite and greater than zero.")

    # Use the smallest treatment group as the search base.
    rows_by_level = {
        level: np.flatnonzero(treatment == level)
        for level in all_levels
    }

    sizes = [
        len(rows_by_level[level])
        for level in all_levels
    ]

    # min() preserves the first group when group sizes are tied.
    red_idx = min(
        range(len(all_levels)),
        key=lambda index: sizes[index],
    )

    red_level = all_levels[red_idx]
    other_levels = [
        level for level in all_levels if level != red_level
    ]
    o1, o2 = other_levels

    n_red = sizes[red_idx]

    coords_red = ps_used[rows_by_level[red_level]]
    coords_o1 = ps_used[rows_by_level[o1]]
    coords_o2 = ps_used[rows_by_level[o2]]

    # The tree walks read these one point and one flag at a time, hundreds of
    # thousands of times, and a numpy row view or a `np.bool_` costs several
    # times what a plain list lookup does. `tolist()` yields the identical
    # doubles -- both are IEEE binary64 -- so `_distance_sq_2d()` evaluates the
    # same two products and the same sum, and every branch it feeds is
    # unchanged. The numpy arrays are kept for the vectorised perimeter block.
    coords_red_list = coords_red.tolist()
    coords_o1_list = coords_o1.tolist()
    coords_o2_list = coords_o2.tolist()

    active_o1 = [True] * len(coords_o1)
    active_o2 = [True] * len(coords_o2)

    # A KD-tree built once over every subject keeps describing subjects that
    # have since been matched away: its bounding boxes stay wide and its
    # subtrees stay walkable while holding almost nothing available, so late
    # queries walk the whole consumed neighbourhood to find a few survivors.
    # Rebuilding over the survivors whenever half of them are gone keeps every
    # query proportional to what is actually left, at an amortised cost of
    # O(n log n) per halving.
    #
    # `kdtree_build()` takes the surviving positions directly and keeps
    # reporting original ones, so a rebuild is transparent to everything else.
    trees = {}

    def rebuild(side, coords, active):
        """Index the surviving subjects of one comparator group."""
        survivors = np.flatnonzero(np.asarray(active, dtype=bool))

        if len(survivors) == 0:
            trees[side] = (None, None, 0)
            return

        root = kdtree_build(coords, survivors)

        trees[side] = (
            root,
            kdtree_index(root, len(coords)),
            len(survivors),
        )

    def consume(side, coords, active, point):
        """Retire one subject, rebuilding the index once half of them are gone."""
        root, leaf_of, at_last_rebuild = trees[side]

        active[point] = False
        kdtree_deactivate(leaf_of[point])

        if root["n_active"] * 2 <= at_last_rebuild:
            rebuild(side, coords, active)

    rebuild("o1", coords_o1, active_o1)
    rebuild("o2", coords_o2, active_o2)

    matched_flag = np.zeros(n_red, dtype=bool)
    exhausted = np.zeros(n_red, dtype=bool)
    counters = np.zeros(n_red, dtype=int)

    candidate_heap = []
    next_candidate_order = 0

    def push_candidates(i):
        """Generate candidate trios for one search-base subject."""
        nonlocal next_candidate_order
        point_red = coords_red[i]
        point_red_list = coords_red_list[i]

        tree_o1 = trees["o1"][0]
        tree_o2 = trees["o2"][0]

        if tree_o1 is None or tree_o2 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        nearest_o1 = kdtree_nearest(
            tree_o1,
            coords_o1_list,
            point_red_list,
            active_o1,
        )

        if nearest_o1 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        nearest_o2 = kdtree_nearest(
            tree_o2,
            coords_o2_list,
            coords_o1_list[nearest_o1],
            active_o2,
        )

        if nearest_o2 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        # `np.sum((a - b) ** 2)` over two elements is the same two squares and
        # the same single addition that `_distance_sq_2d()` performs, and
        # `math.sqrt` and `np.sqrt` are both the correctly-rounded IEEE root, so
        # these three seed distances are bit-for-bit what they were.
        point_o1 = coords_o1_list[nearest_o1]
        point_o2 = coords_o2_list[nearest_o2]

        dist_red_o1 = math.sqrt(_distance_sq_2d(point_red_list, point_o1))
        dist_red_o2 = math.sqrt(_distance_sq_2d(point_red_list, point_o2))
        dist_o1_o2 = math.sqrt(_distance_sq_2d(point_o1, point_o2))

        seed_perimeter = dist_red_o1 + dist_red_o2 + dist_o1_o2

        candidate_o1 = []
        candidate_o2 = []
        candidate_perimeter = []

        # Always consider the nearest-neighbor seed trio.
        if seed_perimeter <= caliper:
            candidate_o1.append(nearest_o1)
            candidate_o2.append(nearest_o2)
            candidate_perimeter.append(seed_perimeter)

        radius_sq = (seed_perimeter / 2) ** 2

        neighbors_o1 = np.asarray(
            kdtree_range(
                tree_o1,
                coords_o1_list,
                point_red_list,
                radius_sq,
                active_o1,
            ),
            dtype=int,
        )
        neighbors_o2 = np.asarray(
            kdtree_range(
                tree_o2,
                coords_o2_list,
                point_red_list,
                radius_sq,
                active_o2,
            ),
            dtype=int,
        )

        if len(neighbors_o1) > 0 and len(neighbors_o2) > 0:
            points_o1 = coords_o1[neighbors_o1]
            points_o2 = coords_o2[neighbors_o2]

            dist_red_to_o1 = np.sqrt(
                np.sum((points_o1 - point_red) ** 2, axis=1)
            )
            dist_red_to_o2 = np.sqrt(
                np.sum((points_o2 - point_red) ** 2, axis=1)
            )

            o1_sq = np.sum(points_o1**2, axis=1)
            o2_sq = np.sum(points_o2**2, axis=1)

            pair_distance_sq = (
                o1_sq[:, None]
                + o2_sq[None, :]
                - 2 * (points_o1 @ points_o2.T)
            )
            pair_distance_sq[pair_distance_sq < 0] = 0.0

            dist_o1_to_o2 = np.sqrt(pair_distance_sq)

            perimeter = (
                dist_red_to_o1[:, None]
                + dist_red_to_o2[None, :]
                + dist_o1_to_o2
            )

            mask = (
                (perimeter < seed_perimeter)
                & (perimeter <= caliper)
            )

            # Preserve column-major candidate scanning for deterministic ties.
            o2_idx, o1_idx = np.where(mask.T)

            if len(o1_idx) > 0:
                candidate_o1.extend(neighbors_o1[o1_idx].tolist())
                candidate_o2.extend(neighbors_o2[o2_idx].tolist())
                candidate_perimeter.extend(
                    perimeter[o1_idx, o2_idx].tolist()
                )

        if len(candidate_perimeter) == 0:
            exhausted[i] = True
            counters[i] = 0
            return

        candidate_perimeter = np.asarray(candidate_perimeter)

        # Stable sorting preserves candidate order when perimeters are tied.
        order = np.argsort(
            candidate_perimeter,
            kind="stable",
        )[: min(top_n, len(candidate_perimeter))]

        for candidate_idx in order:
            # The monotonic insertion order is the second heap key. It
            # preserves min(list, key=...)'s first-occurrence tie handling
            # without scanning and deleting from parallel Python lists.
            heapq.heappush(
                candidate_heap,
                (
                    float(candidate_perimeter[candidate_idx]),
                    next_candidate_order,
                    i,
                    candidate_o1[candidate_idx],
                    candidate_o2[candidate_idx],
                ),
            )
            next_candidate_order += 1

        counters[i] = len(order)

    for i in range(n_red):
        push_candidates(i)

    matched_rows = []

    # Global greedy selection without replacement.
    while candidate_heap:
        (
            perimeter_selected,
            _,
            red_idx_local,
            o1_idx_local,
            o2_idx_local,
        ) = heapq.heappop(candidate_heap)

        if matched_flag[red_idx_local]:
            continue

        if active_o1[o1_idx_local] and active_o2[o2_idx_local]:
            matched_flag[red_idx_local] = True

            consume("o1", coords_o1, active_o1, o1_idx_local)
            consume("o2", coords_o2, active_o2, o2_idx_local)

            row_red = int(
                rows_by_level[red_level][red_idx_local]
            )
            row_o1 = int(
                rows_by_level[o1][o1_idx_local]
            )
            row_o2 = int(
                rows_by_level[o2][o2_idx_local]
            )

            rows_this_trio = {
                red_level: row_red,
                o1: row_o1,
                o2: row_o2,
            }

            row_anchor = rows_this_trio[anchor_level]

            row = {
                "matched_set_id": len(matched_rows) + 1,
                "anchor": row_anchor,
            }

            for group in groups:
                row[group] = rows_this_trio[group]

            # Output distances always use the SAM anchor as the origin.
            for group in groups:
                row[f"dist_{group}"] = float(
                    np.sqrt(
                        np.sum(
                            (
                                ps_used[row_anchor]
                                - ps_used[row[group]]
                            )
                            ** 2
                        )
                    )
                )

            # Retained for output compatibility with sam_match().
            row["loss"] = float(
                sum(row[f"dist_{group}"] for group in groups)
            )
            row["rassen_perimeter"] = float(perimeter_selected)

            matched_rows.append(row)

        else:
            counters[red_idx_local] -= 1

            if counters[red_idx_local] <= 0 and not exhausted[red_idx_local]:
                push_candidates(red_idx_local)

    matched = build_matched_frame(
        matched_rows, groups, extra_columns=("rassen_perimeter",)
    )
    unmatched_anchor_rows, matching_rate = summarize_matching(
        matched, anchor_rows
    )

    return {
        "matched": matched,
        "unmatched_anchor_rows": unmatched_anchor_rows,
        "matching_rate": matching_rate,
        "max_possible_rate": max_possible_rate(
            {group: rows_by_level[group] for group in groups}, anchor_rows
        ),
        "caliper": caliper,
        "red_level": red_level,
        "reference_level": reference_level,
    }
