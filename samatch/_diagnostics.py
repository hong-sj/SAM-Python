"""
Matching diagnostics for Shared Anchor Matching.
"""

import itertools

import numpy as np
import pandas as pd

from ._match_common import groups_from_matched
from ._utils_math import auc_mannwhitney


def compute_smd_balance(data, matched, X_vars=None, groups=None):
    """
    Compute standardized mean difference balance after matching.

    Standardized mean differences are calculated between the matched anchor
    group and each comparator group for all specified covariates.

    Parameters
    ----------
    data : pandas.DataFrame
        Original data containing the covariates.
    matched : pandas.DataFrame
        Matched-set data returned by `sam_match()` or `match_3way()`.
    X_vars : list of str, optional
        Covariate column names. Defaults to X1 through X10.
    groups : list of str, optional
        Comparator treatment groups. Inferred from `matched` if omitted.

    Returns
    -------
    dict
        Dictionary containing:

        - ``by_covariate``: SMD for each covariate and comparator group, with
          an ``smd_defined`` flag that is False when a covariate has no
          variance in either arm.
        - ``summary``: mean and maximum absolute SMD for each comparator
          group, plus ``n_undefined``, the number of covariates that could
          not be assessed. Undefined covariates are excluded from the mean
          and maximum rather than propagating a NaN.

    Notes
    -----
    A match that formed no sets has nothing to balance. It is reported as
    zero rows in ``by_covariate`` and an empty ``summary`` rather than as
    covariates that could not be assessed, which is a different finding.
    """
    if X_vars is None:
        X_vars = [f"X{i}" for i in range(1, 11)]

    if groups is None:
        groups = groups_from_matched(matched)

    rows = []

    if len(matched) == 0:
        return {
            "by_covariate": pd.DataFrame(
                columns=["group", "covariate", "smd", "smd_defined"]
            ).astype({"smd": "float64", "smd_defined": "bool"}),
            "summary": pd.DataFrame(
                columns=["group", "mean_abs_smd", "max_abs_smd", "n_undefined"]
            ).astype(
                {
                    "mean_abs_smd": "float64",
                    "max_abs_smd": "float64",
                    "n_undefined": "int64",
                }
            ),
        }

    x_anchor = data.iloc[matched["anchor"].to_numpy()][X_vars]

    for group in groups:
        x_group = data.iloc[matched[group].to_numpy()][X_vars]

        for covariate in X_vars:
            mean_anchor = x_anchor[covariate].mean()
            mean_group = x_group[covariate].mean()

            var_anchor = x_anchor[covariate].var(ddof=1)
            var_group = x_group[covariate].var(ddof=1)

            pooled_sd = np.sqrt((var_anchor + var_group) / 2)

            # A covariate with no variance in either arm has no defined SMD.
            # Reporting 0.0 keeps it out of the summary statistics without
            # emitting a NaN that would then be silently skipped.
            smd = (
                (mean_anchor - mean_group) / pooled_sd
                if pooled_sd > 0
                else 0.0
            )

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
        group_rows = by_covariate.loc[by_covariate["group"] == group]
        defined = group_rows.loc[group_rows["smd_defined"]]
        values = defined["smd"].abs()

        summary_rows.append(
            {
                "group": group,
                "mean_abs_smd": values.mean(),
                "max_abs_smd": values.max(),
                # Covariates with no variance in either arm cannot be
                # assessed; counting them keeps that visible rather than
                # letting the summary look complete.
                "n_undefined": int((~group_rows["smd_defined"]).sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return {
        "by_covariate": by_covariate,
        "summary": summary,
    }


def compute_pairwise_treatment_auc(gps, matched, groups=None, anchor_level=None):
    """
    Compute pairwise treatment-discrimination AUC after matching.

    For each pair of treatment groups, subjects are scored using the
    log-ratio of their generalized propensity scores. AUC values close to
    0.5 indicate limited discrimination between the matched treatment groups.

    Parameters
    ----------
    gps : pandas.DataFrame
        Generalized propensity score matrix.
    matched : pandas.DataFrame
        Matched-set data returned by `sam_match()` or `match_3way()`.
    groups : list of str, optional
        Comparator treatment groups. Inferred from `matched` if omitted.
    anchor_level : str
        Anchor treatment group.

    Returns
    -------
    dict
        Dictionary containing:

        - ``pairwise``: pairwise AUC for each treatment-group comparison.
        - ``mean_auc``: mean pairwise AUC.

    Notes
    -----
    Subjects are scored with the GPS model as fitted, on the same subjects it
    was fitted to. The reported AUC therefore measures residual separation
    after matching, not held-out discrimination.

    A match that formed no sets has no subjects to discriminate between, and is
    reported as zero pairwise rows with a NaN mean.
    """
    if groups is None:
        groups = groups_from_matched(matched)

    if anchor_level is None:
        raise ValueError("`anchor_level` is required")

    if len(matched) == 0:
        return {
            "pairwise": pd.DataFrame(
                columns=["group_1", "group_2", "auc"]
            ).astype({"auc": "float64"}),
            "mean_auc": float("nan"),
        }

    all_levels = [anchor_level] + list(groups)

    rows_by_level = {
        level: (
            matched["anchor"].to_numpy()
            if level == anchor_level
            else matched[level].to_numpy()
        )
        for level in all_levels
    }

    pairwise_rows = []

    for group_1, group_2 in itertools.combinations(all_levels, 2):
        score = np.log(gps[group_1].to_numpy()) - np.log(
            gps[group_2].to_numpy()
        )

        rows_group_1 = rows_by_level[group_1]
        rows_group_2 = rows_by_level[group_2]

        combined_rows = np.concatenate([rows_group_1, rows_group_2])

        labels = np.concatenate(
            [
                np.ones(len(rows_group_1)),
                np.zeros(len(rows_group_2)),
            ]
        )

        auc = auc_mannwhitney(
            score[combined_rows],
            labels,
        )

        pairwise_rows.append(
            {
                "group_1": group_1,
                "group_2": group_2,
                "auc": auc,
            }
        )

    pairwise = pd.DataFrame(pairwise_rows)

    return {
        "pairwise": pairwise,
        "mean_auc": pairwise["auc"].mean(),
    }