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
from scipy.spatial import cKDTree

from ._match_common import (
    build_matched_frame,
    max_possible_rate,
    order_key,
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


# Canonical candidate search ---------------------------------------------------

# Relative widening applied to spatial-index queries. The index computes
# distances its own way, so it is asked for slightly more than needed and the
# exact test below decides.
_INDEX_RTOL = 1e-9
_INDEX_ATOL = 1e-300


class _ActiveGroup:
    """
    Comparator subjects still available for three-way matching.

    A scipy `cKDTree` over the surviving subjects proposes points; whether a
    point qualifies, and in which order, is decided here on squared distances
    computed as ``dx * dx + dy * dy`` -- the same arithmetic the R
    implementation performs -- so results do not depend on the index.

    The tree is rebuilt over the survivors whenever half of the subjects it
    was built on have been consumed, which keeps queries proportional to what
    is left at an amortised O(n log n) per halving.
    """

    def __init__(self, coords):
        self.coords = coords
        self.x = np.ascontiguousarray(coords[:, 0])
        self.y = np.ascontiguousarray(coords[:, 1])
        self.active = np.ones(len(coords), dtype=bool)
        self.n_active = len(coords)
        self._build()

    def _build(self):
        self.members = np.flatnonzero(self.active)
        self.n_indexed = len(self.members)
        self.tree = (
            cKDTree(self.coords[self.members]) if self.n_indexed else None
        )

    def deactivate(self, point):
        """Consume one subject."""
        self.active[point] = False
        self.n_active -= 1

        if self.n_active * 2 <= self.n_indexed:
            self._build()

    def distance_sq(self, points, query):
        """Exact squared distances from `query` to `points`."""
        dx = self.x[points] - query[0]
        dy = self.y[points] - query[1]
        return dx * dx + dy * dy

    def nearest(self, query):
        """Lowest-row available subject at the minimum distance, or None."""
        if self.n_active == 0:
            return None

        k = min(8, self.n_indexed)

        while True:
            distance, position = self.tree.query(query, k=k)
            distance = np.atleast_1d(distance)
            points = self.members[np.atleast_1d(position)]
            points = points[self.active[points]]

            if len(points) > 0:
                distance_sq = self.distance_sq(points, query)
                best = distance_sq.min()

                # Everything the tree did not return lies at least
                # distance[-1] away. Once that clears the best distance by the
                # tolerance, no unseen subject can tie or win.
                if k >= self.n_indexed or distance[-1] > (
                    math.sqrt(best) * (1 + _INDEX_RTOL) + _INDEX_ATOL
                ):
                    return int(points[distance_sq == best].min())
            elif k >= self.n_indexed:
                return None

            k = min(4 * k, self.n_indexed)

    def nearest_many(self, queries):
        """`nearest()` for many queries while nothing has been consumed."""
        if self.n_active != len(self.coords):
            raise RuntimeError("nearest_many() requires an untouched group.")

        k = min(2, self.n_indexed)
        distance, position = self.tree.query(queries, k=k, workers=-1)

        if k == 1:
            return self.members[position]

        result = self.members[position[:, 0]]

        # A clearly farther runner-up means the nearest point is unique.
        # Anything closer is resolved exactly, which is rare outside data with
        # duplicated propensity scores.
        unclear = ~(
            distance[:, 1] > distance[:, 0] * (1 + _INDEX_RTOL) + _INDEX_ATOL
        )

        for j in np.flatnonzero(unclear):
            result[j] = self.nearest(queries[j])

        return result

    def within(self, query, radius_sq):
        """Available subjects strictly inside the radius, by ascending row."""
        if self.n_active == 0:
            return np.empty(0, dtype=np.intp)

        radius = math.sqrt(radius_sq) * (1 + _INDEX_RTOL) + _INDEX_ATOL
        points = self.members[
            np.asarray(self.tree.query_ball_point(query, radius), dtype=np.intp)
        ]
        points = points[self.active[points]]
        points = points[self.distance_sq(points, query) < radius_sq]
        points.sort()
        return points

    def within_many(self, queries, radius_sq):
        """`within()` for many queries while nothing has been consumed."""
        if self.n_active != len(self.coords):
            raise RuntimeError("within_many() requires an untouched group.")

        radius = np.sqrt(radius_sq) * (1 + _INDEX_RTOL) + _INDEX_ATOL
        proposals = self.tree.query_ball_point(queries, radius, workers=-1)
        result = []

        for j, proposed in enumerate(proposals):
            points = self.members[np.asarray(proposed, dtype=np.intp)]
            points = points[
                self.distance_sq(points, queries[j]) < radius_sq[j]
            ]
            points.sort()
            result.append(points)

        return result


def _point_distance(a, b):
    """Row-wise distances between two (n, 2) arrays, as sqrt(dx^2 + dy^2)."""
    dx = a[:, 0] - b[:, 0]
    dy = a[:, 1] - b[:, 1]
    return np.sqrt(dx * dx + dy * dy)


def _seed_perimeter(red, nearest_o1, nearest_o2):
    """Perimeter of each seed trio, summed in the canonical order."""
    return (
        _point_distance(nearest_o1, red) + _point_distance(nearest_o2, red)
    ) + _point_distance(nearest_o1, nearest_o2)


def _rank_trios(query, group_o1, group_o2, nearest_o1, nearest_o2,
                seed_perimeter, within_o1, within_o2, caliper, top_n):
    """
    Return one subject's best `top_n` candidate trios.

    Candidates are the seed trio (when within the caliper) and every pair of
    in-radius subjects whose perimeter is below the seed's and within the
    caliper, ranked by (perimeter key, o1 row, o2 row).
    """
    trio_o1 = []
    trio_o2 = []
    trio_perimeter = []

    if seed_perimeter <= caliper:
        trio_o1.append(np.array([nearest_o1], dtype=np.intp))
        trio_o2.append(np.array([nearest_o2], dtype=np.intp))
        trio_perimeter.append(np.array([seed_perimeter]))

    if len(within_o1) > 0 and len(within_o2) > 0:
        distance_red_o1 = np.sqrt(group_o1.distance_sq(within_o1, query))
        distance_red_o2 = np.sqrt(group_o2.distance_sq(within_o2, query))

        dx = group_o1.x[within_o1][:, None] - group_o2.x[within_o2][None, :]
        dy = group_o1.y[within_o1][:, None] - group_o2.y[within_o2][None, :]
        distance_o1_o2 = np.sqrt(dx * dx + dy * dy)

        perimeter = (
            distance_red_o1[:, None] + distance_red_o2[None, :]
        ) + distance_o1_o2

        keep_o1, keep_o2 = np.nonzero(
            (perimeter < seed_perimeter) & (perimeter <= caliper)
        )

        if len(keep_o1) > 0:
            trio_o1.append(within_o1[keep_o1])
            trio_o2.append(within_o2[keep_o2])
            trio_perimeter.append(perimeter[keep_o1, keep_o2])

    if not trio_perimeter:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, np.empty(0)

    trio_o1 = np.concatenate(trio_o1)
    trio_o2 = np.concatenate(trio_o2)
    trio_perimeter = np.concatenate(trio_perimeter)

    order = np.lexsort((trio_o2, trio_o1, order_key(trio_perimeter)))[:top_n]

    return trio_o1[order], trio_o2[order], trio_perimeter[order]


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

    # Candidate generation follows one canonical rule, written so that its
    # result depends only on the data and never on how a spatial index happens
    # to order equidistant points:
    #
    # * the seed partner is the nearest available subject, lowest row first
    #   among equidistant ones (`_ActiveGroup.nearest`);
    # * candidate trios are ranked by (perimeter key, o1 row, o2 row), the
    #   perimeter key being `order_key()` of the perimeter;
    # * every distance is computed as sqrt(dx * dx + dy * dy) and every
    #   perimeter as (d_red_o1 + d_red_o2) + d_o1_o2, exactly as the R
    #   implementation does, so both languages see bit-identical values.
    #
    # The KD-trees only propose points; membership is decided on those exact
    # values. This is what lets the search use a compiled, multithreaded index
    # while staying reproducible across implementations.
    group_o1 = _ActiveGroup(coords_o1)
    group_o2 = _ActiveGroup(coords_o2)

    matched_flag = np.zeros(n_red, dtype=bool)
    exhausted = np.zeros(n_red, dtype=bool)
    counters = np.zeros(n_red, dtype=int)

    candidate_heap = []
    next_candidate_order = 0

    # Search radius used whenever the caliper is tighter than a subject's seed
    # trio. Any trio kept needs perimeter < seed AND perimeter <= caliper, and
    # each side of a triangle is at most half its perimeter, so nothing farther
    # than min(seed, caliper) / 2 from the base subject can contribute. When
    # the caliper binds, the bound is inclusive, so it is widened by a slack
    # far above rounding error and far below any meaningful distance; the few
    # extra points gathered fail the perimeter test.
    caliper_half = caliper / 2 + 1e-6 * (1.0 + float(np.abs(ps_used).max()))

    def search_half_width(seed_perimeter):
        return np.where(
            seed_perimeter <= caliper, seed_perimeter / 2, caliper_half
        )

    def push_candidates(i, nearest_o1, nearest_o2, seed_perimeter, within_o1,
                        within_o2):
        """Rank one search-base subject's candidate trios onto the heap."""
        nonlocal next_candidate_order

        trio_o1, trio_o2, trio_perimeter = _rank_trios(
            coords_red[i],
            group_o1,
            group_o2,
            nearest_o1,
            nearest_o2,
            seed_perimeter,
            within_o1,
            within_o2,
            caliper,
            top_n,
        )

        if len(trio_perimeter) == 0:
            exhausted[i] = True
            counters[i] = 0
            return

        keys = order_key(trio_perimeter)

        for key, a, b, perimeter in zip(
            keys.tolist(),
            trio_o1.tolist(),
            trio_o2.tolist(),
            trio_perimeter.tolist(),
        ):
            # The monotonic insertion order is the second heap key, so equal
            # perimeters leave in the order they were ranked.
            heapq.heappush(
                candidate_heap,
                (key, next_candidate_order, i, a, b, perimeter),
            )
            next_candidate_order += 1

        counters[i] = len(trio_perimeter)

    def regenerate(i):
        """Search one subject again against the subjects still available."""
        query = coords_red[i]
        nearest_o1 = group_o1.nearest(query)

        if nearest_o1 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        nearest_o2 = group_o2.nearest(coords_o1[nearest_o1])

        if nearest_o2 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        seed = float(
            _seed_perimeter(
                coords_red[i : i + 1],
                coords_o1[[nearest_o1]],
                coords_o2[[nearest_o2]],
            )[0]
        )
        half = float(search_half_width(seed))
        radius_sq = half * half

        push_candidates(
            i,
            nearest_o1,
            nearest_o2,
            seed,
            group_o1.within(query, radius_sq),
            group_o2.within(query, radius_sq),
        )

    # Initial pass. Nothing has been consumed yet, so every subject's search
    # is independent of the others and is issued to the trees as one batch,
    # which scipy spreads over all cores. The heap is then filled in subject
    # order, exactly as a one-at-a-time loop would.
    if n_red > 0:
        nearest_o1_all = group_o1.nearest_many(coords_red)
        nearest_o2_all = group_o2.nearest_many(coords_o1[nearest_o1_all])
        seed_all = _seed_perimeter(
            coords_red,
            coords_o1[nearest_o1_all],
            coords_o2[nearest_o2_all],
        )
        half_all = search_half_width(seed_all)
        radius_sq_all = half_all * half_all
        within_o1_all = group_o1.within_many(coords_red, radius_sq_all)
        within_o2_all = group_o2.within_many(coords_red, radius_sq_all)

        for i in range(n_red):
            push_candidates(
                i,
                int(nearest_o1_all[i]),
                int(nearest_o2_all[i]),
                float(seed_all[i]),
                within_o1_all[i],
                within_o2_all[i],
            )

    matched_rows = []

    # Global greedy selection without replacement.
    while candidate_heap:
        (
            _,
            _,
            red_idx_local,
            o1_idx_local,
            o2_idx_local,
            perimeter_selected,
        ) = heapq.heappop(candidate_heap)

        if matched_flag[red_idx_local]:
            continue

        if group_o1.active[o1_idx_local] and group_o2.active[o2_idx_local]:
            matched_flag[red_idx_local] = True

            group_o1.deactivate(o1_idx_local)
            group_o2.deactivate(o2_idx_local)

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
                regenerate(red_idx_local)

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
