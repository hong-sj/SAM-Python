"""
Mahalanobis-distance utilities for Shared Anchor Matching.
"""

import warnings

import numpy as np


def get_pooled_covariance(data, X_vars, treatment_var):
    """
    Compute the pooled within-group covariance matrix.

    Each treatment group's covariates are centered by their group-specific
    means. The within-group cross-products are then pooled across treatment
    groups and divided by the residual degrees of freedom.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the covariates and treatment variable.
    X_vars : list of str
        Covariate column names.
    treatment_var : str
        Name of the treatment variable.

    Returns
    -------
    dict
        Dictionary containing:

        - ``S``: pooled within-group covariance matrix.
        - ``S_inv``: inverse covariance matrix.
    """
    groups = data[treatment_var].unique()
    p = len(X_vars)

    S_within = np.zeros((p, p))

    for group in groups:
        X_group = data.loc[
            data[treatment_var] == group,
            X_vars,
        ].to_numpy(dtype=float)

        X_centered = X_group - X_group.mean(
            axis=0,
            keepdims=True,
        )

        S_within += X_centered.T @ X_centered

    df = len(data) - len(groups)
    S = S_within / df

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
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

    treatment = data[treatment_var].to_numpy()

    group_rows = {
        group: np.flatnonzero(treatment == group)
        for group in groups
    }

    X_anchor = data.iloc[anchor_rows][
        X_vars
    ].to_numpy(dtype=float)

    distance_matrices = {}

    for group in groups:
        X_group = data.iloc[group_rows[group]][
            X_vars
        ].to_numpy(dtype=float)

        distance_matrices[group] = mahalanobis_distance_matrix(
            X_anchor,
            X_group,
            pooled["S_inv"],
        )

    return {
        "S_inv": pooled["S_inv"],
        "group_rows": group_rows,
        "D": distance_matrices,
    }