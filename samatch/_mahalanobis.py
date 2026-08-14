"""
Mahalanobis-distance utilities for Shared Anchor Matching.
"""

import warnings

import numpy as np

from ._validate import (
    covariate_matrix,
    require_rows,
    treatment_labels,
    treatment_level,
)


def get_pooled_covariance(data, X_vars=None, treatment_var="T"):
    """
    Compute the pooled within-group covariance matrix.

    Each treatment group's covariates are centered by their group-specific
    means. The within-group cross-products are then pooled across treatment
    groups and divided by the residual degrees of freedom.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the covariates and treatment variable.
    X_vars : list of str, optional
        Covariate column names. Defaults to X1 through X10.
    treatment_var : str, default="T"
        Name of the treatment variable.

    Returns
    -------
    dict
        Dictionary containing:

        - ``S``: pooled within-group covariance matrix.
        - ``S_inv``: inverse covariance matrix.
    """
    if X_vars is None:
        X_vars = [f"X{i}" for i in range(1, 11)]

    X = covariate_matrix(data, X_vars)
    treatment = treatment_labels(data, treatment_var)

    groups = np.unique(treatment)
    p = len(X_vars)

    df = len(data) - len(groups)

    if df <= 0:
        raise ValueError(
            "insufficient residual degrees of freedom to estimate the pooled "
            f"covariance matrix: {len(data)} rows across {len(groups)} "
            "treatment groups. Each group needs more than one subject."
        )

    S_within = np.zeros((p, p))

    for group in groups:
        X_group = X[treatment == group]

        X_centered = X_group - X_group.mean(
            axis=0,
            keepdims=True,
        )

        S_within += X_centered.T @ X_centered

    S = S_within / df

    # numpy.linalg.inv does not raise on every degenerate input -- notably it
    # returns an all-NaN matrix rather than raising -- so the result is
    # checked explicitly instead of relying solely on the exception.
    try:
        S_inv = np.linalg.inv(S)
        degenerate = not np.isfinite(S_inv).all()
    except np.linalg.LinAlgError:
        degenerate = True

    if degenerate:
        warnings.warn(
            "Pooled covariance matrix is numerically singular; "
            "falling back to numpy.linalg.pinv().",
            RuntimeWarning,
        )
        S_inv = np.linalg.pinv(S)

    return {
        "S": S,
        "S_inv": S_inv,
    }


def mahalanobis_distance_matrix(
    X_query,
    X_reference,
    S_inv,
):
    """
    Compute pairwise Mahalanobis distances between two sets of observations.

    Parameters
    ----------
    X_query : array-like of shape (n_query, p)
        Query observations.
    X_reference : array-like of shape (n_reference, p)
        Reference observations.
    S_inv : array-like of shape (p, p)
        Precision matrix used for Mahalanobis distance.

    Returns
    -------
    numpy.ndarray
        Matrix of non-squared Mahalanobis distances with shape
        ``(n_query, n_reference)``.
    """
    X_query = np.asarray(X_query, dtype=float)
    X_reference = np.asarray(X_reference, dtype=float)
    S_inv = np.asarray(S_inv, dtype=float)

    query_transformed = X_query @ S_inv
    reference_transformed = X_reference @ S_inv

    query_sq = np.sum(
        query_transformed * X_query,
        axis=1,
    )
    reference_sq = np.sum(
        reference_transformed * X_reference,
        axis=1,
    )
    cross = query_transformed @ X_reference.T

    distance_sq = (
        query_sq[:, None]
        + reference_sq[None, :]
        - 2 * cross
    )

    # Guard against small negative values from floating-point error.
    distance_sq[distance_sq < 0] = 0.0

    return np.sqrt(distance_sq)


def build_group_distance_matrices(
    data,
    X_vars,
    treatment_var,
    anchor_rows,
    groups,
):
    """
    Build anchor-to-comparator Mahalanobis distance matrices.

    The pooled within-group covariance matrix is used to compute the full
    pairwise Mahalanobis distance matrix between the anchor group and each
    comparator group.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the covariates and treatment variable.
    X_vars : list of str
        Covariate column names.
    treatment_var : str
        Name of the treatment variable.
    anchor_rows : array-like of int
        Positional row indices of anchor subjects.
    groups : list of str
        Comparator treatment groups.

    Returns
    -------
    dict
        Dictionary containing:

        - ``S_inv``: pooled precision matrix.
        - ``group_rows``: positional row indices for each comparator group.
        - ``D``: anchor-to-comparator Mahalanobis distance matrix for each
          comparator group.
    """
    pooled = get_pooled_covariance(
        data,
        X_vars,
        treatment_var,
    )

    treatment = treatment_labels(data, treatment_var)

    # Comparison goes through treatment_level() for the same reason every other
    # call site does: treatment labels are canonicalised to strings, so a raw
    # numeric group label would match no rows and report a level as absent that
    # is in fact present.
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
    X_anchor = X[np.asarray(anchor_rows, dtype=int)]

    distance_matrices = {}

    for group in groups:
        distance_matrices[group] = mahalanobis_distance_matrix(
            X_anchor,
            X[group_rows[group]],
            pooled["S_inv"],
        )

    return {
        "S_inv": pooled["S_inv"],
        "group_rows": group_rows,
        "D": distance_matrices,
    }