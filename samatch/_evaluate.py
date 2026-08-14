"""
SAM matching evaluation and matched-cohort outcome analysis.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._diagnostics import (
    compute_pairwise_treatment_auc,
    compute_smd_balance,
)
from ._validate import check_data_fingerprint


def _infer_anchor_level(data, anchor_rows, treatment_var):
    """
    Determine the anchor treatment level from the anchor rows.

    Every anchor row must carry the same treatment label; reading only the
    first one would silently accept a mismatched `data`/`search` pair.
    """
    anchor_values = pd.unique(data.iloc[anchor_rows][treatment_var].astype(str))

    if len(anchor_values) != 1:
        raise ValueError(
            "Could not uniquely determine `anchor_level` from "
            "`search['anchor_rows']`: found "
            + ", ".join(map(repr, anchor_values[:5]))
            + ". This usually means `data` is not the frame the candidate "
            "search was built from."
        )

    return str(anchor_values[0])


def _loss_dist_of(x):
    """Summarize a matched-set distance metric."""
    if len(x) == 0:
        return pd.DataFrame(
            [{
                "mean": np.nan,
                "median": np.nan,
                "sd": np.nan,
                "p95": np.nan,
                "max": np.nan,
            }]
        )

    x = np.asarray(x, dtype=float)

    return pd.DataFrame(
        [{
            "mean": x.mean(),
            "median": np.median(x),
            "sd": x.std(ddof=1),
            "p95": np.quantile(x, 0.95),
            "max": x.max(),
        }]
    )


def sam_evaluate(
    data,
    search,
    match_result,
    gps,
    X_vars=None,
    treatment_var="T",
):
    """
    Evaluate a completed SAM match.

    Evaluates matched-set loss, Mahalanobis dispersion, covariate balance,
    pairwise treatment-discrimination AUC, and the matching rate.

    Parameters
    ----------
    data : pandas.DataFrame
        Original data used for matching.
    search : dict
        Output from `gps_candidate_search()`.
    match_result : dict
        Output from `sam_match()`.
    gps : pandas.DataFrame
        Generalized propensity score matrix.
    X_vars : list of str, optional
        Covariate column names. Defaults to X1 through X10.
    treatment_var : str, default="T"
        Name of the treatment variable.

    Returns
    -------
    dict
        Dictionary containing:

        - ``loss_distribution``: summary of matched-set losses.
        - ``dispersion_distribution``: summary of Mahalanobis dispersion.
        - ``smd_balance``: standardized mean difference diagnostics.
        - ``treatment_discrimination_auc``: pairwise treatment AUCs.
        - ``matching_rate``: proportion of anchor subjects matched.
    """
    if X_vars is None:
        X_vars = [f"X{i}" for i in range(1, 11)]

    check_data_fingerprint(search, data, treatment_var, "sam_evaluate")

    groups = search["groups"]
    matched = match_result["matched"]

    anchor_rows = np.asarray(search["anchor_rows"], dtype=int)
    anchor_level = _infer_anchor_level(data, anchor_rows, treatment_var)

    loss_values = matched["loss"].to_numpy() if len(matched) else []
    loss_distribution = _loss_dist_of(loss_values)

    if len(matched) > 0:
        dist_cols = [f"dist_{group}" for group in groups]
        dispersion_values = (
            matched[dist_cols].to_numpy(dtype=float) ** 2
        ).sum(axis=1)
    else:
        dispersion_values = np.array([])

    dispersion_distribution = _loss_dist_of(dispersion_values)

    smd_balance = compute_smd_balance(
        data,
        matched,
        X_vars,
        groups,
    )

    treatment_discrimination_auc = compute_pairwise_treatment_auc(
        gps,
        matched,
        groups,
        anchor_level,
    )

    return {
        "loss_distribution": loss_distribution,
        "dispersion_distribution": dispersion_distribution,
        "smd_balance": smd_balance,
        "treatment_discrimination_auc": treatment_discrimination_auc,
        "matching_rate": match_result["matching_rate"],
    }


# Matched-cohort reconstruction ------------------------------------------------


def extract_matched_data(
    data,
    search,
    match_result,
    treatment_var="T",
    anchor_level=None,
):
    """
    Reconstruct the subject-level matched cohort from a completed SAM match.

    Each retained subject is assigned its matched-set identifier, treatment
    role, and original positional row index.

    Parameters
    ----------
    data : pandas.DataFrame
        Original data used for matching.
    search : dict
        Output from `gps_candidate_search()`.
    match_result : dict
        Output from `sam_match()`.
    treatment_var : str, default="T"
        Name of the treatment variable.
    anchor_level : str, optional
        Anchor treatment group. If None, inferred from `search`.

    Returns
    -------
    pandas.DataFrame
        Subject-level matched cohort with ``matched_set_id``,
        ``matched_role``, and ``original_row`` columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas.DataFrame.")

    if treatment_var not in data.columns:
        raise ValueError(f"Treatment column not found: {treatment_var}")

    check_data_fingerprint(search, data, treatment_var, "extract_matched_data")

    matched = match_result["matched"]
    groups = list(search["groups"])

    if matched is None or len(matched) == 0:
        raise ValueError("`match_result` contains no matched sets.")

    required_match_cols = ["anchor", *groups]
    missing_cols = [
        col for col in required_match_cols
        if col not in matched.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing matched-set column(s): " + ", ".join(missing_cols)
        )

    # Determine the anchor treatment level.
    if anchor_level is None:
        anchor_level = _infer_anchor_level(
            data,
            np.asarray(search["anchor_rows"], dtype=int),
            treatment_var,
        )
    else:
        anchor_level = str(anchor_level)

    treatment_order = [anchor_level, *groups]
    k = len(treatment_order)

    # Support older match objects without matched_set_id.
    if "matched_set_id" in matched.columns:
        set_ids = matched["matched_set_id"].to_numpy()
    else:
        set_ids = np.arange(1, len(matched) + 1, dtype=int)

    # Expand matched sets into subject-level records.
    index_rows = []

    for i in range(len(matched)):
        subject_rows = [
            int(matched.iloc[i]["anchor"]),
            *[int(matched.iloc[i][group]) for group in groups],
        ]

        for role, subject_row in zip(treatment_order, subject_rows):
            index_rows.append(
                {
                    "matched_set_id": set_ids[i],
                    "matched_role": role,
                    "original_row": subject_row,
                }
            )

    matched_index = pd.DataFrame(index_rows)
    original_rows = matched_index["original_row"].to_numpy(dtype=int)

    if np.any(original_rows < 0) or np.any(original_rows >= len(data)):
        raise ValueError(
            "Invalid row index detected in `match_result['matched']`."
        )

    matched_data = data.iloc[original_rows].copy()
    matched_data["matched_set_id"] = matched_index[
        "matched_set_id"
    ].to_numpy()
    matched_data["matched_role"] = matched_index[
        "matched_role"
    ].to_numpy()
    matched_data["original_row"] = original_rows

    # Preserve matched-set and treatment ordering.
    role_order = {
        group: i for i, group in enumerate(treatment_order)
    }

    matched_data["_sam_role_order"] = (
        matched_data[treatment_var]
        .astype(str)
        .map(role_order)
    )

    matched_data = (
        matched_data
        .sort_values(["matched_set_id", "_sam_role_order"])
        .drop(columns="_sam_role_order")
        .reset_index(drop=True)
    )

    # Each matched set must contain exactly one subject per treatment group.
    set_sizes = matched_data.groupby(
        "matched_set_id",
        sort=False,
    ).size()

    if not np.all(set_sizes.to_numpy() == k):
        raise ValueError(
            f"At least one matched set does not contain exactly {k} subjects."
        )

    count_table = pd.crosstab(
        matched_data["matched_set_id"],
        matched_data[treatment_var].astype(str),
    )

    missing_groups = [
        group for group in treatment_order
        if group not in count_table.columns
    ]

    if missing_groups:
        raise ValueError(
            "Matched data are missing treatment group(s): "
            + ", ".join(missing_groups)
        )

    if not np.all(count_table[treatment_order].to_numpy() == 1):
        raise ValueError(
            "Each matched set must contain exactly one subject "
            "from every treatment group."
        )

    matched_data.attrs["anchor_level"] = anchor_level
    matched_data.attrs["groups"] = groups
    matched_data.attrs["K"] = k

    return matched_data


# Outcome-analysis utilities ---------------------------------------------------


def _fit_treatment_only_logistic(
    data,
    outcome_var,
    treatment_var,
    anchor_level,
):
    """Fit an unadjusted logistic model of outcome on treatment."""
    treatment = data[treatment_var].astype(str).to_numpy()

    treatment_levels = [
        anchor_level,
        *[
            group
            for group in pd.unique(treatment)
            if group != anchor_level
        ],
    ]
    comparator_levels = treatment_levels[1:]

    n = len(data)
    p = 1 + len(comparator_levels)

    # Intercept plus one indicator for each comparator group.
    X = np.zeros((n, p), dtype=float)
    X[:, 0] = 1.0

    coef_names = ["(Intercept)"]

    for j, group in enumerate(comparator_levels, start=1):
        X[:, j] = (treatment == group).astype(float)
        coef_names.append(f"{treatment_var}{group}")

    y = data[outcome_var].to_numpy(dtype=float)

    # Initialize coefficients from observed treatment-group risks.
    eps = 1e-8
    p_anchor = np.clip(
        y[treatment == anchor_level].mean(),
        eps,
        1 - eps,
    )

    beta = np.zeros(p, dtype=float)
    beta[0] = np.log(p_anchor / (1 - p_anchor))

    for j, group in enumerate(comparator_levels, start=1):
        p_group = np.clip(
            y[treatment == group].mean(),
            eps,
            1 - eps,
        )
        beta[j] = np.log(p_group / (1 - p_group)) - beta[0]

    converged = False
    max_iter = 100
    tol = 1e-10

    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = mu * (1 - mu)

        score = X.T @ (y - mu)
        info = X.T @ (X * weights[:, None])

        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info) @ score

        beta_new = beta + step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break

        beta = beta_new

    mu = 1.0 / (1.0 + np.exp(-(X @ beta)))

    return {
        "beta": beta,
        "X": X,
        "y": y,
        "mu": mu,
        "coef_names": coef_names,
        "treatment_levels": treatment_levels,
        "comparator_levels": comparator_levels,
        "converged": converged,
    }


def _cluster_robust_vcov(fit, cluster):
    """Compute a matched-set cluster-robust covariance matrix."""
    X = fit["X"]
    y = fit["y"]
    mu = fit["mu"]

    cluster = np.asarray(cluster)

    if len(cluster) != len(y):
        raise ValueError(
            "`cluster` length must equal the number of observations."
        )

    weights = mu * (1 - mu)
    xtwx = X.T @ (X * weights[:, None])

    try:
        bread = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(xtwx)

    score_i = X * (y - mu)[:, None]

    # Accumulate per-cluster scores with a scatter-add. Masking the full score
    # matrix once per cluster would be quadratic here, since SAM produces one
    # matched set per anchor and so the cluster count grows with the sample.
    codes, unique_clusters = pd.factorize(cluster)

    score_cluster = np.column_stack(
        [
            np.bincount(
                codes,
                weights=score_i[:, column],
                minlength=len(unique_clusters),
            )
            for column in range(X.shape[1])
        ]
    )

    meat = score_cluster.T @ score_cluster
    vcov = bread @ meat @ bread

    # CR1 finite-sample correction.
    g = len(unique_clusters)
    n, p = X.shape

    if g > 1 and n > p:
        correction = (g / (g - 1)) * ((n - 1) / (n - p))
        vcov = correction * vcov

    return vcov


def _numeric_gradient(fun, beta, rel_step=1e-6):
    """Compute a central-difference numerical gradient."""
    beta = np.asarray(beta, dtype=float)
    gradient = np.zeros_like(beta)

    for j in range(len(beta)):
        h = rel_step * max(1.0, abs(beta[j]))

        beta_hi = beta.copy()
        beta_lo = beta.copy()

        beta_hi[j] += h
        beta_lo[j] -= h

        gradient[j] = (
            fun(beta_hi) - fun(beta_lo)
        ) / (2.0 * h)

    return gradient


def sam_estimate_effects(
    matched_data,
    outcome_var,
    treatment_var="T",
    set_id_var="matched_set_id",
    anchor_level=None,
    conf_level=0.95,
):
    """
    Estimate treatment effects after SAM matching.

    Fits an unadjusted logistic regression of the binary outcome on treatment.
    Confidence intervals use a matched-set cluster-robust covariance matrix.

    Reports treatment-group risks and comparator-versus-anchor odds ratios
    (OR), risk ratios (RR), and risk differences (RD).

    Parameters
    ----------
    matched_data : pandas.DataFrame
        Subject-level matched data returned by `extract_matched_data()`.
    outcome_var : str
        Binary outcome column coded as 0/1.
    treatment_var : str, default="T"
        Name of the treatment variable.
    set_id_var : str, default="matched_set_id"
        Name of the matched-set identifier.
    anchor_level : str, optional
        Anchor treatment group. If None, obtained from `matched_data.attrs`.
    conf_level : float, default=0.95
        Confidence level.

    Returns
    -------
    dict
        Dictionary containing:

        - ``analysis_summary``: matched-cohort summary.
        - ``group_risk``: treatment-group risk estimates.
        - ``contrasts``: anchor-referenced OR, RR, and RD estimates.
        - ``model``: fitted logistic model components.
        - ``vcov_cluster``: matched-set cluster-robust covariance matrix.
    """
    if not isinstance(matched_data, pd.DataFrame):
        raise TypeError("`matched_data` must be a pandas.DataFrame.")

    required_cols = [outcome_var, treatment_var, set_id_var]
    missing_cols = [
        col for col in required_cols
        if col not in matched_data.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_cols)
        )

    if not 0 < conf_level < 1:
        raise ValueError("`conf_level` must be between 0 and 1.")

    # Remove entire matched sets containing missing analysis variables.
    row_complete = matched_data[required_cols].notna().all(axis=1)

    incomplete_set_ids = pd.unique(
        matched_data.loc[~row_complete, set_id_var]
    )

    analysis_data = matched_data.loc[
        ~matched_data[set_id_var].isin(incomplete_set_ids)
    ].copy()

    if len(analysis_data) == 0:
        raise ValueError(
            "No complete matched sets are available for outcome analysis."
        )

    k_expected = matched_data.attrs.get("K")

    if k_expected is None:
        k_expected = (
            matched_data[treatment_var]
            .astype(str)
            .nunique()
        )

    retained_set_sizes = analysis_data.groupby(
        set_id_var,
        sort=False,
    ).size()

    if not np.all(retained_set_sizes.to_numpy() == int(k_expected)):
        raise ValueError(
            "Outcome analysis contains incomplete matched sets."
        )

    # Validate the binary outcome before integer conversion.
    y = analysis_data[outcome_var]

    if y.dtype == bool:
        analysis_data[outcome_var] = y.astype(int)
    else:
        y_numeric = pd.to_numeric(y, errors="coerce")

        if y_numeric.isna().any():
            raise ValueError(
                "`outcome_var` must be numeric/logical and coded as 0/1."
            )

        if not set(y_numeric.unique()).issubset({0, 1}):
            raise ValueError(
                "`outcome_var` must be binary and coded as 0/1."
            )

        analysis_data[outcome_var] = y_numeric.astype(int)

    if anchor_level is None:
        anchor_level = matched_data.attrs.get("anchor_level")

    if anchor_level is None:
        raise ValueError(
            "`anchor_level` must be supplied or available from "
            "`extract_matched_data()`."
        )

    anchor_level = str(anchor_level)

    observed_levels = list(
        pd.unique(analysis_data[treatment_var].astype(str))
    )

    if anchor_level not in observed_levels:
        raise ValueError(
            "`anchor_level` is not present in the matched dataset."
        )

    comparator_levels = [
        group for group in observed_levels
        if group != anchor_level
    ]
    treatment_levels = [anchor_level, *comparator_levels]

    # Warn when a treatment group has complete outcome separation.
    for group in treatment_levels:
        y_group = analysis_data.loc[
            analysis_data[treatment_var].astype(str) == group,
            outcome_var,
        ]

        if y_group.sum() == 0 or y_group.sum() == len(y_group):
            warnings.warn(
                f"Treatment group '{group}' has "
                f"{int(y_group.sum())}/{len(y_group)} events. "
                "The treatment-only logistic model may exhibit complete "
                "separation and OR inference may be unstable.",
                RuntimeWarning,
            )

    fit = _fit_treatment_only_logistic(
        analysis_data,
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        anchor_level=anchor_level,
    )

    if not fit["converged"]:
        warnings.warn(
            "The treatment-only logistic model did not satisfy the "
            "Newton-Raphson convergence tolerance.",
            RuntimeWarning,
        )

    beta = fit["beta"]
    coef_index = {
        name: i
        for i, name in enumerate(fit["coef_names"])
    }

    vcov_cluster = _cluster_robust_vcov(
        fit,
        cluster=analysis_data[set_id_var].to_numpy(),
    )

    z_value = norm.ppf(1.0 - (1.0 - conf_level) / 2.0)

    # Marginal risk by treatment group.
    group_risk_rows = []

    for group in treatment_levels:
        x_group = np.zeros(len(beta), dtype=float)
        x_group[0] = 1.0

        if group != anchor_level:
            coef_name = f"{treatment_var}{group}"
            x_group[coef_index[coef_name]] = 1.0

        eta_group = float(x_group @ beta)
        se_eta_group = float(
            np.sqrt(x_group @ vcov_cluster @ x_group)
        )

        risk = 1.0 / (1.0 + np.exp(-eta_group))
        risk_ci_low = 1.0 / (
            1.0 + np.exp(-(eta_group - z_value * se_eta_group))
        )
        risk_ci_high = 1.0 / (
            1.0 + np.exp(-(eta_group + z_value * se_eta_group))
        )

        is_group = (
            analysis_data[treatment_var].astype(str) == group
        )
        y_group = analysis_data.loc[is_group, outcome_var]

        group_risk_rows.append(
            {
                "treatment": group,
                "n": int(is_group.sum()),
                "events": int(y_group.sum()),
                "risk": float(risk),
                "risk_ci_low": float(risk_ci_low),
                "risk_ci_high": float(risk_ci_high),
            }
        )

    group_risk = pd.DataFrame(group_risk_rows)

    # Anchor-referenced treatment contrasts.
    contrast_rows = []

    for group in comparator_levels:
        coef_name = f"{treatment_var}{group}"
        coef_j = coef_index[coef_name]

        def risk_anchor_fun(b):
            return 1.0 / (1.0 + np.exp(-b[0]))

        def risk_comparator_fun(b):
            eta = b[0] + b[coef_j]
            return 1.0 / (1.0 + np.exp(-eta))

        def log_or_fun(b):
            return float(b[coef_j])

        def log_rr_fun(b):
            p_anchor = risk_anchor_fun(b)
            p_group = risk_comparator_fun(b)
            return float(np.log(p_group / p_anchor))

        def rd_fun(b):
            return float(
                risk_comparator_fun(b) - risk_anchor_fun(b)
            )

        # Odds ratio.
        log_or = log_or_fun(beta)

        grad_log_or = np.zeros(len(beta), dtype=float)
        grad_log_or[coef_j] = 1.0

        se_log_or = float(
            np.sqrt(
                grad_log_or
                @ vcov_cluster
                @ grad_log_or
            )
        )

        or_value = float(np.exp(log_or))
        or_ci_low = float(
            np.exp(log_or - z_value * se_log_or)
        )
        or_ci_high = float(
            np.exp(log_or + z_value * se_log_or)
        )

        # Risk ratio.
        log_rr = log_rr_fun(beta)
        grad_log_rr = _numeric_gradient(log_rr_fun, beta)

        se_log_rr = float(
            np.sqrt(
                grad_log_rr
                @ vcov_cluster
                @ grad_log_rr
            )
        )

        rr_value = float(np.exp(log_rr))
        rr_ci_low = float(
            np.exp(log_rr - z_value * se_log_rr)
        )
        rr_ci_high = float(
            np.exp(log_rr + z_value * se_log_rr)
        )

        # Risk difference.
        rd_value = rd_fun(beta)
        grad_rd = _numeric_gradient(rd_fun, beta)

        se_rd = float(
            np.sqrt(
                grad_rd
                @ vcov_cluster
                @ grad_rd
            )
        )

        rd_ci_low = float(rd_value - z_value * se_rd)
        rd_ci_high = float(rd_value + z_value * se_rd)

        is_anchor = (
            analysis_data[treatment_var].astype(str)
            == anchor_level
        )
        is_group = (
            analysis_data[treatment_var].astype(str)
            == group
        )

        contrast_rows.append(
            {
                "anchor": anchor_level,
                "comparator": group,
                "n_anchor": int(is_anchor.sum()),
                "n_comparator": int(is_group.sum()),
                "events_anchor": int(
                    analysis_data.loc[
                        is_anchor,
                        outcome_var,
                    ].sum()
                ),
                "events_comparator": int(
                    analysis_data.loc[
                        is_group,
                        outcome_var,
                    ].sum()
                ),
                "risk_anchor": float(risk_anchor_fun(beta)),
                "risk_comparator": float(
                    risk_comparator_fun(beta)
                ),
                "log_or": float(log_or),
                "se_log_or": se_log_or,
                "OR": or_value,
                "OR_ci_low": or_ci_low,
                "OR_ci_high": or_ci_high,
                "log_rr": float(log_rr),
                "se_log_rr": se_log_rr,
                "RR": rr_value,
                "RR_ci_low": rr_ci_low,
                "RR_ci_high": rr_ci_high,
                "RD": float(rd_value),
                "se_RD": se_rd,
                "RD_ci_low": rd_ci_low,
                "RD_ci_high": rd_ci_high,
            }
        )

    contrasts = pd.DataFrame(contrast_rows)

    analysis_summary = pd.DataFrame(
        [{
            "anchor_level": anchor_level,
            "n_matched_sets": int(
                analysis_data[set_id_var].nunique()
            ),
            "K": int(len(treatment_levels)),
            "n_subjects": int(len(analysis_data)),
            "confidence_level": float(conf_level),
        }]
    )

    return {
        "analysis_summary": analysis_summary,
        "group_risk": group_risk,
        "contrasts": contrasts,
        "model": fit,
        "vcov_cluster": vcov_cluster,
    }