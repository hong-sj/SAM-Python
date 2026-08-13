"""
Multi-arm propensity score weighting utilities for Shared Anchor Matching.
"""

import numpy as np
import pandas as pd

from ._gps import estimate_gps_multinom


def compute_balancing_weights(
    data,
    method="iptw",
    gps=None,
    X_vars=None,
    treatment_var="T",
    anchor_level="A",
    stabilize=True,
):
    """
    Compute propensity score balancing weights.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment and covariate variables.
    method : {"iptw", "overlap", "matching"}, default="iptw"
        Weighting method.
    gps : pandas.DataFrame, optional
        Precomputed generalized propensity scores. If None, GPS values are
        estimated using `estimate_gps_multinom()`.
    X_vars : list of str, optional
        Covariate column names used for GPS estimation.
    treatment_var : str, default="T"
        Name of the treatment variable.
    anchor_level : str, default="A"
        Anchor treatment group used for GPS estimation.
    stabilize : bool, default=True
        Whether to stabilize IPTW using empirical treatment prevalence.
        Ignored for overlap and matching weights.

    Returns
    -------
    dict
        Dictionary containing weights, tilting function values, weighting
        method, and generalized propensity scores.
    """
    if method not in ("iptw", "overlap", "matching"):
        raise ValueError('method must be "iptw", "overlap", or "matching"')

    if treatment_var not in data.columns:
        raise ValueError(f"'{treatment_var}' not found in data")

    if gps is None:
        gps = estimate_gps_multinom(
            data,
            X_vars=X_vars,
            treatment_var=treatment_var,
            anchor_level=anchor_level,
        )["gps"]

    if len(gps) != len(data):
        raise ValueError("gps and data must contain the same number of rows")

    treatment = data[treatment_var].astype(str).to_numpy()

    missing_levels = [
        level for level in np.unique(treatment) if level not in gps.columns
    ]
    if missing_levels:
        raise ValueError(
            "Treatment group(s) not found in gps: " + ", ".join(missing_levels)
        )

    gps_values = gps.to_numpy(dtype=float)
    column_index = {column: i for i, column in enumerate(gps.columns)}

    own_gps = gps_values[
        np.arange(len(gps_values)),
        [column_index[level] for level in treatment],
    ]

    if method == "iptw":
        h = np.ones(len(gps_values))
    elif method == "overlap":
        h = 1.0 / np.sum(1.0 / gps_values, axis=1)
    else:
        h = gps_values.min(axis=1)

    weights = h / own_gps

    if method == "iptw" and stabilize:
        levels, counts = np.unique(treatment, return_counts=True)
        marginal_probability = dict(zip(levels, counts / counts.sum()))

        weights *= np.asarray(
            [marginal_probability[level] for level in treatment]
        )

    return {
        "weights": weights,
        "h": h,
        "method": method,
        "gps": gps,
    }


def compute_weighted_balance(
    data,
    weights,
    X_vars=None,
    treatment_var="T",
    anchor_level="A",
):
    """
    Compute weighted covariate balance.

    Standardized mean differences are calculated between the weighted anchor
    group and each weighted comparator group.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment and covariate variables.
    weights : array-like
        Subject-level balancing weights.
    X_vars : list of str, optional
        Covariate column names. Defaults to X1 through X10.
    treatment_var : str, default="T"
        Name of the treatment variable.
    anchor_level : str, default="A"
        Anchor treatment group.

    Returns
    -------
    dict
        Dictionary containing covariate-specific weighted SMDs and summary
        balance statistics.
    """
    if X_vars is None:
        X_vars = [f"X{i}" for i in range(1, 11)]

    weights = np.asarray(weights, dtype=float)

    if len(weights) != len(data):
        raise ValueError(
            "weights and data must contain the same number of observations"
        )

    treatment = data[treatment_var].astype(str).to_numpy()
    groups = [group for group in pd.unique(treatment) if group != anchor_level]

    def weighted_mean(x, w):
        return np.sum(x * w) / np.sum(w)

    def weighted_variance(x, w):
        mean = weighted_mean(x, w)
        return np.sum(w * (x - mean) ** 2) / np.sum(w)

    rows = []
    anchor_mask = treatment == anchor_level

    for group in groups:
        group_mask = treatment == group

        for covariate in X_vars:
            x = data[covariate].to_numpy(dtype=float)

            mean_anchor = weighted_mean(x[anchor_mask], weights[anchor_mask])
            mean_group = weighted_mean(x[group_mask], weights[group_mask])

            var_anchor = weighted_variance(x[anchor_mask], weights[anchor_mask])
            var_group = weighted_variance(x[group_mask], weights[group_mask])

            pooled_sd = np.sqrt((var_anchor + var_group) / 2)
            smd = (mean_anchor - mean_group) / pooled_sd if pooled_sd > 0 else 0.0

            rows.append(
                {
                    "group": group,
                    "covariate": covariate,
                    "smd": smd,
                }
            )

    by_covariate = pd.DataFrame(rows)
    summary_rows = []

    for group in groups:
        values = by_covariate.loc[
            by_covariate["group"] == group, "smd"
        ].abs()

        summary_rows.append(
            {
                "group": group,
                "mean_abs_smd": values.mean(),
                "max_abs_smd": values.max(),
            }
        )

    return {
        "by_covariate": by_covariate,
        "summary": pd.DataFrame(summary_rows),
    }


def compute_effective_sample_size(weights, treatment=None):
    """
    Compute effective sample size from balancing weights.

    Parameters
    ----------
    weights : array-like
        Subject-level balancing weights.
    treatment : array-like, optional
        Treatment-group labels. If supplied, ESS is calculated separately
        for each treatment group.

    Returns
    -------
    float or pandas.DataFrame
        Overall ESS or treatment-specific ESS values.
    """
    weights = np.asarray(weights, dtype=float)

    def ess(w):
        denominator = np.sum(w**2)

        if denominator == 0:
            return float("nan")

        return float(np.sum(w) ** 2 / denominator)

    if treatment is None:
        return ess(weights)

    treatment = np.asarray(treatment)

    if len(treatment) != len(weights):
        raise ValueError("treatment and weights must have the same length")

    rows = []

    for group in pd.unique(treatment):
        mask = treatment == group
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "ess": ess(weights[mask]),
            }
        )

    return pd.DataFrame(rows)


def evaluate_comparator_weighting(
    data,
    method="iptw",
    gps=None,
    X_vars=None,
    treatment_var="T",
    anchor_level="A",
    stabilize=True,
):
    """
    Evaluate a multi-arm propensity score weighting method.

    Computes balancing weights, weighted covariate balance, and effective
    sample size.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment and covariate variables.
    method : {"iptw", "overlap", "matching"}, default="iptw"
        Weighting method.
    gps : pandas.DataFrame, optional
        Precomputed generalized propensity scores.
    X_vars : list of str, optional
        Covariate column names.
    treatment_var : str, default="T"
        Name of the treatment variable.
    anchor_level : str, default="A"
        Anchor treatment group.
    stabilize : bool, default=True
        Whether to stabilize IPTW.

    Returns
    -------
    dict
        Dictionary containing weights, GPS values, weighted balance,
        and effective sample size.
    """
    result = compute_balancing_weights(
        data,
        method=method,
        gps=gps,
        X_vars=X_vars,
        treatment_var=treatment_var,
        anchor_level=anchor_level,
        stabilize=stabilize,
    )

    balance = compute_weighted_balance(
        data,
        result["weights"],
        X_vars=X_vars,
        treatment_var=treatment_var,
        anchor_level=anchor_level,
    )

    ess = compute_effective_sample_size(
        result["weights"],
        treatment=data[treatment_var].astype(str).to_numpy(),
    )

    return {
        "method": method,
        "weights": result["weights"],
        "gps": result["gps"],
        "balance": balance,
        "ess": ess,
    }