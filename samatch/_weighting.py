"""
Multi-arm propensity score weighting utilities for Shared Anchor Matching.
"""

import numpy as np
import pandas as pd

from ._gps import estimate_gps_multinom
from ._validate import (
    covariate_matrix,
    require_rows,
    treatment_labels,
    treatment_level,
    validate_gps,
)


def compute_balancing_weights(
    data,
    method="iptw",
    gps=None,
    X_vars=None,
    treatment_var="T",
    anchor_level="A",
    stabilize=True,
    trim=1e-3,
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
    trim : float, default=1e-3
        Lower bound applied to the generalized propensity scores before
        weights are formed. Set to 0 to disable.

    Returns
    -------
    dict
        Dictionary containing weights, tilting function values, weighting
        method, generalized propensity scores, and ``n_trimmed``.

    Notes
    -----
    Weights divide by each subject's own propensity score, which is unbounded
    as that score approaches zero. Because `estimate_gps_multinom()` fits an
    unregularized model, near-separation can push scores arbitrarily close to
    zero and a single subject can then dominate the weighted estimator, or
    produce infinities that propagate silently into the balance and effective
    sample size summaries. `trim` bounds this, and ``n_trimmed`` reports how
    many subjects were affected -- a nonzero count is a positivity warning
    worth investigating rather than a routine detail.
    """
    if method not in ("iptw", "overlap", "matching"):
        raise ValueError('method must be "iptw", "overlap", or "matching"')

    if gps is None:
        gps = estimate_gps_multinom(
            data,
            X_vars=X_vars,
            treatment_var=treatment_var,
            anchor_level=anchor_level,
        )["gps"]

    gps_values = validate_gps(
        data, gps, treatment_var, "compute_balancing_weights"
    )
    treatment = treatment_labels(data, treatment_var)

    # Hash rather than sort the treatment column; see the same check in
    # `validate_gps`. Sorting only the missing levels keeps the message stable.
    missing_levels = sorted(
        level for level in pd.unique(treatment) if level not in gps.columns
    )
    if missing_levels:
        raise ValueError(
            "Treatment group(s) not found in gps: " + ", ".join(missing_levels)
        )

    if (
        isinstance(trim, bool)
        or not isinstance(trim, (int, float, np.integer, np.floating))
        or not np.isfinite(trim)
        or trim < 0
        or trim >= 1
    ):
        raise ValueError("trim must be in [0, 1)")

    trim = float(trim)

    column_index = {column: i for i, column in enumerate(gps.columns)}

    # Bound the scores away from zero before dividing by them. Left unbounded,
    # a single near-zero score produces a weight large enough to dominate every
    # downstream summary, or an infinity that propagates without a warning.
    n_trimmed = int((gps_values < trim).any(axis=1).sum()) if trim > 0 else 0

    if n_trimmed:
        gps_values = np.clip(gps_values, trim, None)
        gps_values = gps_values / gps_values.sum(axis=1, keepdims=True)

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
        "n_trimmed": n_trimmed,
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

    if weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional array")

    if len(weights) != len(data):
        raise ValueError(
            "weights and data must contain the same number of observations"
        )

    if not np.isfinite(weights).all():
        raise ValueError("weights must all be finite")

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    treatment = treatment_labels(data, treatment_var)
    anchor_level = treatment_level(anchor_level)
    groups = [group for group in pd.unique(treatment) if group != anchor_level]

    anchor_rows = require_rows(
        np.flatnonzero(treatment == anchor_level),
        anchor_level,
        treatment_var,
    )
    group_rows = {
        group: require_rows(
            np.flatnonzero(treatment == group), group, treatment_var
        )
        for group in groups
    }

    X = covariate_matrix(data, X_vars)

    if weights[anchor_rows].sum() <= 0:
        raise ValueError("anchor-group weights must have a positive sum")

    zero_weight_groups = [
        group for group, rows in group_rows.items() if weights[rows].sum() <= 0
    ]

    if zero_weight_groups:
        raise ValueError(
            "Weights must have a positive sum in every treatment group; "
            "zero total weight in: " + ", ".join(zero_weight_groups)
        )

    # The weight total and the mean are passed in rather than recomputed: the
    # variance needs the mean its caller already has, and every covariate in an
    # arm shares one weight total.
    def weighted_mean(x, w, w_sum):
        return np.sum(x * w) / w_sum

    def weighted_variance(x, w, w_sum, mean):
        return np.sum(w * (x - mean) ** 2) / w_sum

    rows = []
    anchor_mask = treatment == anchor_level

    # Each arm's covariates and weights are gathered once, outside the covariate
    # loop. Re-running the boolean gather per covariate copied 20,000 elements
    # eight times over for each of the covariates, which at that size cost far
    # more than the arithmetic it fed.
    x_anchor_all = X[anchor_mask]
    w_anchor = weights[anchor_mask]
    w_anchor_sum = np.sum(w_anchor)

    for group in groups:
        group_mask = treatment == group

        x_group_all = X[group_mask]
        w_group = weights[group_mask]
        w_group_sum = np.sum(w_group)

        for position, covariate in enumerate(X_vars):
            x_anchor = x_anchor_all[:, position]
            x_group = x_group_all[:, position]

            mean_anchor = weighted_mean(x_anchor, w_anchor, w_anchor_sum)
            mean_group = weighted_mean(x_group, w_group, w_group_sum)

            var_anchor = weighted_variance(
                x_anchor, w_anchor, w_anchor_sum, mean_anchor
            )
            var_group = weighted_variance(
                x_group, w_group, w_group_sum, mean_group
            )

            pooled_sd = np.sqrt((var_anchor + var_group) / 2)
            smd = (mean_anchor - mean_group) / pooled_sd if pooled_sd > 0 else 0.0

            rows.append(
                {
                    "group": group,
                    "covariate": covariate,
                    "smd": smd,
                    "smd_defined": bool(pooled_sd > 0),
                }
            )

    by_covariate = pd.DataFrame(rows)
    summary_rows = []

    for group in groups:
        # Not `group_rows`: that name already holds this function's row indices
        # per treatment group, and shadowing it here would be a trap for the
        # next change to this loop.
        group_smd = by_covariate.loc[by_covariate["group"] == group]
        values = group_smd.loc[group_smd["smd_defined"], "smd"].abs()

        summary_rows.append(
            {
                "group": group,
                "mean_abs_smd": values.mean(),
                "max_abs_smd": values.max(),
                "n_undefined": int((~group_smd["smd_defined"]).sum()),
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
    trim=1e-3,
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
    trim : float, default=1e-3
        Lower bound applied to the generalized propensity scores. Passed
        through to `compute_balancing_weights()`.

    Returns
    -------
    dict
        Dictionary containing weights, GPS values, weighted balance,
        effective sample size, and ``n_trimmed``.
    """
    result = compute_balancing_weights(
        data,
        method=method,
        gps=gps,
        X_vars=X_vars,
        treatment_var=treatment_var,
        anchor_level=anchor_level,
        stabilize=stabilize,
        trim=trim,
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
        "n_trimmed": result["n_trimmed"],
    }
