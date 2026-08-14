"""
GPS-guided candidate search for Shared Anchor Matching.

For each anchor subject, candidate subjects are identified from each
comparator group using Euclidean distance in generalized propensity
score (GPS) space.
"""

import numpy as np
from scipy.special import logit as _qlogis

from ._validate import (
    require_positive_int,
    require_rows,
    treatment_labels,
    treatment_level,
)


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

    # Compute Euclidean GPS distances for each comparator group.
    candidates_by_group = {}

    for group in groups:
        group_rows = require_rows(
            np.flatnonzero(treatment == treatment_level(group)),
            group,
            treatment_var,
        )
        x_group = gps_used[group_rows]

        x_sq = np.sum(x_anchor**2, axis=1)
        y_sq = np.sum(x_group**2, axis=1)
        cross = x_anchor @ x_group.T

        d2 = x_sq[:, None] + y_sq[None, :] - 2 * cross
        d2[d2 < 0] = 0.0
        dist_mat = np.sqrt(d2)

        m = min(top_m, len(group_rows))

        # Preserve row order when distances are tied.
        candidates_by_group[group] = [
            group_rows[np.argsort(dist_mat[i], kind="stable")[:m]]
            for i in range(len(anchor_rows))
        ]

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
        "groups": groups,
        "candidates": candidates,
    }