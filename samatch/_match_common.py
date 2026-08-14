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


def matched_frame_dtypes(groups, extra_columns=()):
    """
    Return the dtype of each column of a matched-set frame.

    Row references are integers and distances are floats. An empty frame has to
    say so too: `pandas` infers `object` from no rows, and an `object` column
    cannot be used to index `data`, so a zero-set match would otherwise fail
    with a bare `IndexError` the first time any diagnostic touched it.
    """
    dtypes = {
        "matched_set_id": "int64",
        "anchor": "int64",
        "loss": "float64",
    }

    for group in groups:
        dtypes[group] = "int64"
        dtypes[f"dist_{group}"] = "float64"

    for column in extra_columns:
        dtypes[column] = "float64"

    return dtypes


def build_matched_frame(matched_rows, groups, extra_columns=()):
    """
    Assemble the matched-set frame with a schema independent of the result.

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
        Matched sets, or an empty frame with the same columns and dtypes.
    """
    columns = matched_frame_columns(groups, extra_columns)

    # Supplying columns on both paths prevents dict insertion order in the
    # matching engines from making the populated schema differ from the empty
    # schema.
    matched = pd.DataFrame(matched_rows, columns=columns)

    if len(matched) == 0:
        matched = matched.astype(matched_frame_dtypes(groups, extra_columns))

    return matched


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
        ``(unmatched_anchor_rows, matching_rate)``.

    Notes
    -----
    ``matching_rate`` is bounded above by the size of the smallest comparator
    group divided by the anchor count, since each matched set consumes one
    subject from every comparator group. `max_possible_rate()` reports that
    ceiling, so a rate that looks low can be recognised as saturated.
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
