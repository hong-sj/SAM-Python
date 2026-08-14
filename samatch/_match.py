"""
Core Shared Anchor Matching algorithm.
"""

import heapq

import numpy as np
import pandas as pd

from ._mahalanobis import get_pooled_covariance, mahalanobis_distance_matrix
from ._match_common import (
    build_matched_frame,
    max_possible_rate,
    summarize_matching,
)
from ._validate import (
    check_data_fingerprint,
    covariate_matrix,
    require_rows,
    treatment_labels,
    treatment_level,
)


def sam_match(
    data,
    search,
    X_vars=None,
    treatment_var="T",
):
    """
    Match multiple treatment groups to a shared anchor group.

    For each anchor subject, SAM identifies the nearest available candidate
    from every comparator group in Mahalanobis space. Matched sets are then
    selected globally in ascending total-distance loss without replacement.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the covariates and treatment variable.
    search : dict
        Output from `gps_candidate_search()`.
    X_vars : list of str, optional
        Covariate column names used for Mahalanobis distance.
        Defaults to X1 through X10.
    treatment_var : str, default="T"
        Name of the treatment variable.

    Returns
    -------
    dict
        Dictionary containing:

        - ``matched``: successfully matched sets, row indices,
          group-specific Mahalanobis distances, and total loss.
        - ``unmatched_anchor_rows``: row indices of unmatched anchors.
        - ``matching_rate``: proportion of anchor subjects successfully
          matched.
        - ``max_possible_rate``: the highest rate the group sizes allow.

    Notes
    -----
    Every matched set consumes one subject from each comparator group, so
    ``matching_rate`` cannot exceed the size of the smallest comparator group
    divided by the anchor count. Compare it against ``max_possible_rate``
    before reading a low rate as a poor match.
    """
    if X_vars is None:
        X_vars = [f"X{i}" for i in range(1, 11)]

    check_data_fingerprint(search, data, treatment_var, "sam_match")

    anchor_rows = np.asarray(search["anchor_rows"], dtype=int)
    groups = list(search["groups"])
    candidates = search["candidates"]
    n_anchor = len(anchor_rows)

    pooled = get_pooled_covariance(data, X_vars, treatment_var)
    s_inv = pooled["S_inv"]

    treatment = treatment_labels(data, treatment_var)
    group_rows = {
        group: require_rows(
            np.flatnonzero(treatment == treatment_level(group)),
            group,
            treatment_var,
        )
        for group in groups
    }

    # Materialise the covariates once and slice with numpy rather than
    # rebuilding an intermediate DataFrame per group.
    X = covariate_matrix(data, X_vars)
    x_anchor = X[anchor_rows]
    x_group = {group: X[group_rows[group]] for group in groups}

    # Convert global candidate rows to within-group positions and calculate
    # Mahalanobis distances only for GPS-screened candidates.
    local_position = {
        group: {
            row: position
            for position, row in enumerate(group_rows[group])
        }
        for group in groups
    }

    candidate_local = {group: [] for group in groups}
    candidate_distance = {group: [] for group in groups}

    for group in groups:
        for i in range(n_anchor):
            local_indices = np.asarray(
                [
                    local_position[group][row]
                    for row in candidates[i][group]
                    if row in local_position[group]
                ],
                dtype=int,
            )

            if len(local_indices) == 0:
                candidate_local[group].append(local_indices)
                candidate_distance[group].append(np.array([], dtype=float))
                continue

            distances = mahalanobis_distance_matrix(
                x_anchor[i : i + 1],
                x_group[group][local_indices],
                s_inv,
            )[0]

            # Stable sorting preserves deterministic tie handling.
            order = np.argsort(distances, kind="stable")

            candidate_local[group].append(local_indices[order])
            candidate_distance[group].append(distances[order])

    active = {
        group: np.ones(len(group_rows[group]), dtype=bool)
        for group in groups
    }
    pointer = {
        group: np.zeros(n_anchor, dtype=int)
        for group in groups
    }

    anchor_active = np.ones(n_anchor, dtype=bool)
    best_loss = np.full(n_anchor, np.inf)

    best_choice = {
        group: np.full(n_anchor, -1, dtype=int)
        for group in groups
    }
    best_distance = {
        group: np.full(n_anchor, np.nan)
        for group in groups
    }

    version = np.zeros(n_anchor, dtype=np.int64)
    heap = []

    # Reverse index: which anchors currently have their best match pointing at
    # a given comparator subject. Without it, consuming a subject would mean
    # rescanning every remaining anchor to find the few that were relying on
    # it, which is quadratic in the anchor count.
    claimants = {
        group: [set() for _ in range(len(group_rows[group]))]
        for group in groups
    }

    def release(i):
        """Drop anchor `i`'s outstanding claims on comparator subjects."""
        for group in groups:
            chosen = int(best_choice[group][i])

            if chosen >= 0:
                claimants[group][chosen].discard(i)
                best_choice[group][i] = -1

    def recompute(i):
        """Recompute the current best available match for one anchor."""
        release(i)

        total_loss = 0.0

        for group in groups:
            local_indices = candidate_local[group][i]
            distances = candidate_distance[group][i]
            position = int(pointer[group][i])

            while (
                position < len(local_indices)
                and not active[group][local_indices[position]]
            ):
                position += 1

            pointer[group][i] = position

            if position >= len(local_indices):
                release(i)
                best_loss[i] = np.inf

                for other_group in groups:
                    best_distance[other_group][i] = np.nan

                version[i] += 1
                return

            chosen = int(local_indices[position])

            best_choice[group][i] = chosen
            claimants[group][chosen].add(i)
            best_distance[group][i] = float(distances[position])
            total_loss += float(distances[position])

        best_loss[i] = total_loss
        version[i] += 1

        heapq.heappush(
            heap,
            (float(total_loss), int(i), int(version[i])),
        )

    for i in range(n_anchor):
        recompute(i)

    matched_rows = []
    n_active_anchor = n_anchor

    while n_active_anchor > 0 and heap:
        loss, anchor_idx, entry_version = heapq.heappop(heap)

        if not anchor_active[anchor_idx]:
            continue

        if entry_version != version[anchor_idx]:
            continue

        if not np.isfinite(best_loss[anchor_idx]):
            continue

        if loss != best_loss[anchor_idx]:
            continue

        choice = {
            group: int(best_choice[group][anchor_idx])
            for group in groups
        }

        # Recompute if a selected comparator subject has already been used.
        if any(
            choice[group] < 0 or not active[group][choice[group]]
            for group in groups
        ):
            recompute(anchor_idx)
            continue

        row = {
            "matched_set_id": len(matched_rows) + 1,
            "anchor": int(anchor_rows[anchor_idx]),
        }

        for group in groups:
            row[group] = int(group_rows[group][choice[group]])
            row[f"dist_{group}"] = float(best_distance[group][anchor_idx])

        row["loss"] = float(best_loss[anchor_idx])
        matched_rows.append(row)

        anchor_active[anchor_idx] = False
        n_active_anchor -= 1

        affected = set()

        for group in groups:
            active[group][choice[group]] = False
            affected |= claimants[group][choice[group]]

        release(anchor_idx)
        affected.discard(anchor_idx)

        if n_active_anchor == 0:
            break

        # Only anchors using one of the newly assigned comparator subjects
        # need to have their current best match recomputed. Sorting keeps the
        # order independent of set iteration order.
        for i in sorted(affected):
            if anchor_active[i]:
                recompute(i)

    matched = build_matched_frame(matched_rows, groups)
    unmatched_anchor_rows, matching_rate = summarize_matching(
        matched, anchor_rows
    )

    return {
        "matched": matched,
        "unmatched_anchor_rows": unmatched_anchor_rows,
        "matching_rate": matching_rate,
        "max_possible_rate": max_possible_rate(group_rows, anchor_rows),
    }