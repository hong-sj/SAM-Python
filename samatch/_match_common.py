"""
Result assembly shared by the matching algorithms.

`sam_match()` and `match_3way()` run genuinely different algorithms -- a
Mahalanobis heap-greedy in covariate space and a KD-tree perimeter greedy in
two-dimensional propensity score space -- but they return the same shape of
result. Only that shared plumbing lives here; the engines stay separate.
"""

import numpy as np
import pandas as pd
from scipy.special import logit as _qlogis

# Probabilities are clipped away from the boundary before the logit, which
# would otherwise map 0 and 1 to infinities.
_LOGIT_EPS = 1e-6


def transform_ps(values, space):
    """
    Return propensity scores on the requested scale.

    Parameters
    ----------
    values : numpy.ndarray
        Propensity score matrix.
    space : {"raw", "logit"}
        Scale to return.

    Returns
    -------
    numpy.ndarray
        Transformed scores.
    """
    if space not in ("raw", "logit"):
        raise ValueError('propensity score space must be "raw" or "logit"')

    if space == "raw":
        return values

    return _qlogis(np.clip(values, _LOGIT_EPS, 1 - _LOGIT_EPS))


def matched_frame_columns(groups, extra_columns=()):
    """Return the column order of a matched-set frame."""
    return [
        "matched_set_id",
        "anchor",
        *groups,
        *[f"dist_{group}" for group in groups],
        "loss",
        *extra_columns,
    ]


def build_matched_frame(matched_rows, groups, extra_columns=()):
    """
    Assemble the matched-set frame, preserving the column schema when empty.

    Parameters
    ----------
    matched_rows : list of dict
        One record per matched set.
    groups : list of str
        Comparator treatment groups.
    extra_columns : tuple of str, optional
        Additional columns appended after ``loss``.

    Returns
    -------
    pandas.DataFrame
        Matched sets, or an empty frame with the correct columns.
    """
    if matched_rows:
        return pd.DataFrame(matched_rows)

    return pd.DataFrame(columns=matched_frame_columns(groups, extra_columns))


def groups_from_matched(matched):
    """
    Recover the comparator group names from a matched-set frame.

    Lets the diagnostic functions be called with just the matched sets,
    instead of requiring the caller to carry `search["groups"]` alongside.
    """
    reserved = {"matched_set_id", "anchor", "loss", "rassen_perimeter"}

    return [
        column
        for column in matched.columns
        if column not in reserved and not column.startswith("dist_")
    ]


def summarize_matching(matched, anchor_rows):
    """
    Return the unmatched anchor rows and the matching rate.

    Parameters
    ----------
    matched : pandas.DataFrame
        Matched sets, with an ``anchor`` column.
    anchor_rows : array-like of int
        Positional row indices of every anchor subject.

    Returns
    -------
    tuple
        ``(unmatched_anchor_rows, matching_rate, max_possible_rate)``.

    Notes
    -----
    ``matching_rate`` is bounded above by the size of the smallest comparator
    group divided by the anchor count, since each matched set consumes one
    subject from every comparator group. ``max_possible_rate`` is reported
    alongside it so a rate that looks low can be recognised as saturated.
    """
    anchor_rows = np.asarray(anchor_rows, dtype=int)

    matched_anchor_rows = (
        set(matched["anchor"].tolist()) if len(matched) > 0 else set()
    )

    unmatched_anchor_rows = np.asarray(
        [row for row in anchor_rows if row not in matched_anchor_rows],
        dtype=int,
    )

    matching_rate = (
        len(matched) / len(anchor_rows)
        if len(anchor_rows) > 0
        else float("nan")
    )

    return unmatched_anchor_rows, matching_rate


def max_possible_rate(group_rows, anchor_rows):
    """
    Return the highest matching rate the group sizes allow.

    Parameters
    ----------
    group_rows : dict
        Positional row indices for each comparator group.
    anchor_rows : array-like of int
        Positional row indices of every anchor subject.

    Returns
    -------
    float
        ``min(comparator group sizes) / n_anchor``.
    """
    if len(anchor_rows) == 0:
        return float("nan")

    smallest = min(
        (len(rows) for rows in group_rows.values()),
        default=0,
    )

    return min(smallest, len(anchor_rows)) / len(anchor_rows)
