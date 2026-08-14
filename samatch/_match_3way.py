"""
Three-way nearest-neighbor propensity score matching.

Implements the three-treatment-group matching approach of Rassen et al.
using a two-dimensional propensity score space, KD-tree candidate search,
a perimeter-based caliper, and global greedy selection.
"""

import numpy as np
import pandas as pd
from scipy.special import logit as _qlogis

from ._validate import check_data_fingerprint, require_positive_int


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
        KD-tree node.
    """
    coords = np.asarray(coords, dtype=float)

    if idx is None:
        idx = np.arange(coords.shape[0])

    n = len(idx)

    if n == 1:
        return {
            "leaf": True,
            "point": int(idx[0]),
        }

    dim = depth % 2

    # Stable sorting preserves deterministic tie handling.
    order = np.argsort(coords[idx, dim], kind="stable")
    ordered_idx = idx[order]

    mid = n // 2
    left_idx = ordered_idx[:mid]
    right_idx = ordered_idx[mid:]
    split_val = float(coords[right_idx[0], dim])

    return {
        "leaf": False,
        "split_dim": dim,
        "split_val": split_val,
        "left": kdtree_build(coords, left_idx, depth + 1),
        "right": kdtree_build(coords, right_idx, depth + 1),
    }


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
        best_distance_sq = float(np.sum((coords[best] - query) ** 2))

    if gap * gap < best_distance_sq:
        candidate = kdtree_nearest(other, coords, query, active)

        if candidate is not None:
            candidate_distance_sq = float(
                np.sum((coords[candidate] - query) ** 2)
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
    """
    if node["leaf"]:
        point = node["point"]

        if not active[point]:
            return []

        distance_sq = float(np.sum((coords[point] - query) ** 2))

        # Boundary points are excluded by design.
        return [point] if distance_sq < radius2 else []

    dim = node["split_dim"]
    gap = query[dim] - node["split_val"]

    if gap <= 0:
        primary = node["left"]
        other = node["right"]
    else:
        primary = node["right"]
        other = node["left"]

    result = kdtree_range(primary, coords, query, radius2, active)

    if gap * gap < radius2:
        result += kdtree_range(other, coords, query, radius2, active)

    return result


# Three-way matching -----------------------------------------------------------


def match_3way(
    data,
    search,
    gps,
    X_vars=None,
    treatment_var="T",
    caliper="auto",
    ps_space="raw",
    top_n=10,
    reference_level=None,
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
        Included for interface compatibility with `sam_match()`. Not used
        by the three-way propensity score matching algorithm.
    treatment_var : str, default="T"
        Name of the treatment variable.
    caliper : {"auto"} or float, default="auto"
        Perimeter-scale caliper. ``"auto"`` uses `calc_caliper_3way()`.
    ps_space : {"raw", "logit"}, default="raw"
        Propensity score scale used for matching.
    top_n : int, default=10
        Maximum number of candidate trios retained per search-base subject
        during each candidate-generation step.
    reference_level : str, optional
        Treatment group whose GPS column is omitted when constructing the
        two-dimensional propensity score space. Defaults to the last group.

    Returns
    -------
    dict
        Dictionary containing:

        - ``matched``: matched trios and distance information.
        - ``unmatched_anchor_rows``: unmatched anchor row indices.
        - ``matching_rate``: proportion of anchor subjects matched.
        - ``caliper``: caliper used for matching.
        - ``red_level``: smallest treatment group used as the search base.
        - ``reference_level``: treatment group omitted from PS coordinates.
    """
    # X_vars is retained for interface compatibility with sam_match().
    _ = X_vars

    if ps_space not in ("raw", "logit"):
        raise ValueError('ps_space must be "raw" or "logit"')

    if treatment_var not in data.columns:
        raise ValueError(f"'{treatment_var}' not found in data")

    if len(gps) != len(data):
        raise ValueError("gps and data must contain the same number of rows")

    check_data_fingerprint(search, data, treatment_var, "match_3way")

    top_n = require_positive_int(top_n, "top_n")

    groups = list(search["groups"])

    if len(groups) != 2:
        raise ValueError(
            "match_3way() requires exactly three treatment groups: "
            "one anchor group and two comparator groups."
        )

    anchor_rows = np.asarray(search["anchor_rows"], dtype=int)
    treatment = data[treatment_var].astype(str).to_numpy()

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

    if reference_level not in all_levels:
        raise ValueError(
            "`reference_level` must be one of the three treatment groups."
        )

    ps_levels = [
        level for level in all_levels if level != reference_level
    ]

    ps_raw = gps[ps_levels].to_numpy(dtype=float)

    if ps_space == "logit":
        eps = 1e-6
        ps_used = _qlogis(np.clip(ps_raw, eps, 1 - eps))
    else:
        ps_used = ps_raw

    if isinstance(caliper, str):
        if caliper != "auto":
            raise ValueError(
                '`caliper` must be "auto" or a positive numeric value.'
            )

        caliper = calc_caliper_3way(ps_used, treatment)

    caliper = float(caliper)

    if caliper <= 0:
        raise ValueError("`caliper` must be greater than zero.")

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

    tree_o1 = kdtree_build(coords_o1)
    tree_o2 = kdtree_build(coords_o2)

    active_o1 = np.ones(len(coords_o1), dtype=bool)
    active_o2 = np.ones(len(coords_o2), dtype=bool)

    matched_flag = np.zeros(n_red, dtype=bool)
    exhausted = np.zeros(n_red, dtype=bool)
    counters = np.zeros(n_red, dtype=int)

    pool_red = []
    pool_o1 = []
    pool_o2 = []
    pool_perimeter = []
    pool_dist_o1 = []
    pool_dist_o2 = []

    def push_candidates(i):
        """Generate candidate trios for one search-base subject."""
        point_red = coords_red[i]

        nearest_o1 = kdtree_nearest(
            tree_o1,
            coords_o1,
            point_red,
            active_o1,
        )

        if nearest_o1 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        nearest_o2 = kdtree_nearest(
            tree_o2,
            coords_o2,
            coords_o1[nearest_o1],
            active_o2,
        )

        if nearest_o2 is None:
            exhausted[i] = True
            counters[i] = 0
            return

        dist_red_o1 = float(
            np.sqrt(
                np.sum((point_red - coords_o1[nearest_o1]) ** 2)
            )
        )
        dist_red_o2 = float(
            np.sqrt(
                np.sum((point_red - coords_o2[nearest_o2]) ** 2)
            )
        )
        dist_o1_o2 = float(
            np.sqrt(
                np.sum(
                    (coords_o1[nearest_o1] - coords_o2[nearest_o2]) ** 2
                )
            )
        )

        seed_perimeter = dist_red_o1 + dist_red_o2 + dist_o1_o2

        candidate_o1 = []
        candidate_o2 = []
        candidate_perimeter = []
        candidate_dist_o1 = []
        candidate_dist_o2 = []

        # Always consider the nearest-neighbor seed trio.
        if seed_perimeter <= caliper:
            candidate_o1.append(nearest_o1)
            candidate_o2.append(nearest_o2)
            candidate_perimeter.append(seed_perimeter)
            candidate_dist_o1.append(dist_red_o1)
            candidate_dist_o2.append(dist_red_o2)

        radius_sq = (seed_perimeter / 2) ** 2

        neighbors_o1 = np.asarray(
            kdtree_range(
                tree_o1,
                coords_o1,
                point_red,
                radius_sq,
                active_o1,
            ),
            dtype=int,
        )
        neighbors_o2 = np.asarray(
            kdtree_range(
                tree_o2,
                coords_o2,
                point_red,
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
                candidate_dist_o1.extend(
                    dist_red_to_o1[o1_idx].tolist()
                )
                candidate_dist_o2.extend(
                    dist_red_to_o2[o2_idx].tolist()
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
            pool_red.append(i)
            pool_o1.append(candidate_o1[candidate_idx])
            pool_o2.append(candidate_o2[candidate_idx])
            pool_perimeter.append(candidate_perimeter[candidate_idx])
            pool_dist_o1.append(candidate_dist_o1[candidate_idx])
            pool_dist_o2.append(candidate_dist_o2[candidate_idx])

        counters[i] = len(order)

    for i in range(n_red):
        push_candidates(i)

    matched_rows = []

    # Global greedy selection without replacement.
    while pool_perimeter:
        # min() selects the first occurrence when perimeters are tied.
        pop_idx = min(
            range(len(pool_perimeter)),
            key=lambda index: pool_perimeter[index],
        )

        red_idx_local = pool_red[pop_idx]
        o1_idx_local = pool_o1[pop_idx]
        o2_idx_local = pool_o2[pop_idx]
        perimeter_selected = pool_perimeter[pop_idx]

        del pool_red[pop_idx]
        del pool_o1[pop_idx]
        del pool_o2[pop_idx]
        del pool_perimeter[pop_idx]
        del pool_dist_o1[pop_idx]
        del pool_dist_o2[pop_idx]

        if matched_flag[red_idx_local]:
            continue

        if active_o1[o1_idx_local] and active_o2[o2_idx_local]:
            matched_flag[red_idx_local] = True
            active_o1[o1_idx_local] = False
            active_o2[o2_idx_local] = False

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

    if matched_rows:
        matched = pd.DataFrame(matched_rows)
    else:
        columns = [
            "matched_set_id",
            "anchor",
            *groups,
            *[f"dist_{group}" for group in groups],
            "loss",
            "rassen_perimeter",
        ]
        matched = pd.DataFrame(columns=columns)

    if len(matched) > 0:
        matched_anchor_rows = set(matched["anchor"].tolist())
    else:
        matched_anchor_rows = set()

    unmatched_anchor_rows = np.asarray(
        [
            row
            for row in anchor_rows
            if row not in matched_anchor_rows
        ],
        dtype=int,
    )

    if len(anchor_rows) > 0:
        matching_rate = len(matched) / len(anchor_rows)
    else:
        matching_rate = float("nan")

    return {
        "matched": matched,
        "unmatched_anchor_rows": unmatched_anchor_rows,
        "matching_rate": matching_rate,
        "caliper": caliper,
        "red_level": red_level,
        "reference_level": reference_level,
    }