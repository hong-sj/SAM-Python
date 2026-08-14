"""
GPS-guided candidate search for Shared Anchor Matching.

For each anchor subject, candidate subjects are identified from each
comparator group using Euclidean distance in generalized propensity
score (GPS) space.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logit as _qlogis

from ._validate import (
    data_fingerprint,
    require_positive_int,
    require_rows,
    treatment_labels,
    treatment_level,
)


# Tolerances for recognising equidistant candidates. Distances computed by the
# KD-tree and by numpy need not agree to the last bit, so exact comparison
# would miss ties that the previous full-matrix implementation caught.
_TIE_RTOL = 1e-12
_TIE_ATOL = 1e-300


def _nearest_candidates(x_group, x_anchor, group_rows, top_m):
    """
    Return the `top_m` nearest comparator rows for each anchor.

    A KD-tree is used instead of a full anchor-by-comparator distance matrix:
    only `top_m` neighbours are ever kept, so materialising and sorting every
    pairwise distance is wasted work that also makes memory grow with the
    product of the two group sizes.

    Ties are resolved by row order. That requires care in two places. The
    KD-tree orders equidistant neighbours arbitrarily, so the returned
    neighbours are re-sorted by ``(distance, row)``. More subtly, when more
    candidates are tied at the cutoff distance than there are slots left, the
    tree returns an arbitrary subset of them; those anchors are re-resolved
    against every candidate within the cutoff radius. Both cases only arise
    with exactly equidistant candidates, which in GPS space effectively means
    duplicated subjects.

    Parameters
    ----------
    x_group : numpy.ndarray of shape (n_group, k)
        Comparator subjects in GPS space.
    x_anchor : numpy.ndarray of shape (n_anchor, k)
        Anchor subjects in GPS space.
    group_rows : numpy.ndarray of int
        Positional row indices of the comparator subjects, ascending.
    top_m : int
        Number of candidates to retain per anchor.

    Returns
    -------
    list of numpy.ndarray
        Candidate row indices for each anchor, nearest first.
    """
    n_group = len(group_rows)
    m = min(top_m, n_group)

    tree = cKDTree(x_group)

    # One extra neighbour reveals whether a tie straddles the cutoff.
    k = min(m + 1, n_group)
    distances, positions = tree.query(x_anchor, k=k, workers=-1)

    # query() drops the neighbour axis when k == 1.
    if k == 1:
        distances = distances[:, None]
        positions = positions[:, None]

    cutoff = distances[:, m - 1]

    if k > m:
        ambiguous = np.flatnonzero(
            np.isclose(distances[:, m - 1], distances[:, m], rtol=_TIE_RTOL, atol=0.0)
        )
    else:
        ambiguous = np.empty(0, dtype=int)

    neighbour_rows = group_rows[positions[:, :m]]
    order = np.lexsort((neighbour_rows, distances[:, :m]), axis=-1)
    neighbour_rows = np.take_along_axis(neighbour_rows, order, axis=-1)

    candidates = list(neighbour_rows)

    for i in ambiguous:
        # The radius is widened slightly because the tree's distances and the
        # recomputed ones below need not agree to the last bit. Over-gathering
        # is harmless: the extra candidates are farther away, so they sort
        # after the genuine ties and fall outside the top `m`.
        radius = float(cutoff[i]) * (1.0 + _TIE_RTOL) + _TIE_ATOL

        within = np.asarray(
            tree.query_ball_point(x_anchor[i], radius), dtype=int
        )
        tied_distances = np.linalg.norm(x_group[within] - x_anchor[i], axis=1)
        tied_rows = group_rows[within]

        candidates[i] = tied_rows[np.lexsort((tied_rows, tied_distances))[:m]]

    return candidates


def gps_candidate_search(
    data,
    gps,
    treatment_var="T",
    anchor_level="A",
    top_m=10,
    gps_space="raw",
):
    """
    Identify GPS-nearest candidates for each anchor subject.

    For each anchor subject, the function retains the `top_m` nearest
    subjects from each comparator group using Euclidean distance in GPS
    space. The resulting candidate pools are used by `sam_match()` for
    subsequent Mahalanobis-distance matching.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment variable.
    gps : pandas.DataFrame
        Generalized propensity score matrix with one row per subject and
        one column per treatment group.
    treatment_var : str, default="T"
        Name of the treatment variable.
    anchor_level : str, default="A"
        Anchor treatment group.
    top_m : int, default=10
        Number of candidates retained per anchor and comparator group.
    gps_space : {"raw", "logit"}, default="raw"
        GPS scale used to calculate Euclidean distance.

    Returns
    -------
    dict
        Dictionary containing:

        - ``anchor_rows``: positional row indices of anchor subjects.
        - ``groups``: comparator treatment groups.
        - ``candidates``: candidate row indices for each anchor and
          comparator group.
        - ``data_fingerprint``: identifies the frame these positional indices
          refer to, so later stages can reject a modified `data`.

    Notes
    -----
    The returned row indices are positional. The same DataFrame must be
    passed unmodified to `sam_match()`, `sam_evaluate()` and
    `extract_matched_data()`; re-sorting or filtering it in between would
    repoint those indices at different subjects.
    """
    if gps_space not in ("raw", "logit"):
        raise ValueError('gps_space must be "raw" or "logit"')

    if len(gps) != len(data):
        raise ValueError("gps and data must contain the same number of rows")

    top_m = require_positive_int(top_m, "top_m")
    anchor_level = treatment_level(anchor_level)

    gps_values = gps.to_numpy(dtype=float)

    # Transform GPS values to the logit scale if requested.
    if gps_space == "logit":
        eps = 1e-6
        gps_used = _qlogis(np.clip(gps_values, eps, 1 - eps))
    else:
        gps_used = gps_values

    groups = [
        group for group in gps.columns if treatment_level(group) != anchor_level
    ]
    treatment = treatment_labels(data, treatment_var)

    anchor_rows = require_rows(
        np.flatnonzero(treatment == anchor_level),
        anchor_level,
        treatment_var,
    )
    x_anchor = gps_used[anchor_rows]

    # Find the nearest comparator subjects for each anchor in GPS space.
    #
    # A KD-tree is used rather than a full anchor-by-comparator distance
    # matrix: only `top_m` neighbours are ever kept, so materialising and
    # sorting every pairwise distance is wasted work that also makes memory
    # grow with the product of the group sizes.
    candidates_by_group = {}

    for group in groups:
        group_rows = require_rows(
            np.flatnonzero(treatment == treatment_level(group)),
            group,
            treatment_var,
        )

        candidates_by_group[group] = _nearest_candidates(
            gps_used[group_rows],
            x_anchor,
            group_rows,
            top_m,
        )

    # Organize candidate lists by anchor.
    candidates = [
        {
            group: candidates_by_group[group][i]
            for group in groups
        }
        for i in range(len(anchor_rows))
    ]

    return {
        "anchor_rows": anchor_rows,
        "data_fingerprint": data_fingerprint(data, treatment_var),
        "groups": groups,
        "candidates": candidates,
    }