"""
Matching diagnostics for Shared Anchor Matching.
"""

import itertools

import numpy as np
import pandas as pd

from ._utils_math import auc_mannwhitney


def compute_smd_balance(data, matched, X_vars, groups):
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
    X_vars : list of str
        Covariate column names.
    groups : list of str
        Comparator treatment groups.

    Returns
    -------
    dict
        Dictionary containing:

        - ``by_covariate``: SMD for each covariate and comparator group.
        - ``summary``: mean and maximum absolute SMD for each comparator group.
    """
    rows = []

    x_anchor = data.iloc[matched["anchor"].to_numpy()][X_vars]

    for group in groups:
        x_group = data.iloc[matched[group].to_numpy()][X_vars]

        for covariate in X_vars:
            mean_anchor = x_anchor[covariate].mean()
            mean_group = x_group[covariate].mean()

            var_anchor = x_anchor[covariate].var(ddof=1)
            var_group = x_group[covariate].var(ddof=1)

            smd = (mean_anchor - mean_group) / np.sqrt(
                (var_anchor + var_group) / 2
            )

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
            by_covariate["group"] == group,
            "smd",
        ].abs()

        summary_rows.append(
            {
                "group": group,
                "mean_abs_smd": values.mean(),
                "max_abs_smd": values.max(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return {
        "by_covariate": by_covariate,
        "summary": summary,
    }


def compute_pairwise_treatment_auc(gps, matched, groups, anchor_level):
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
    groups : list of str
        Comparator treatment groups.
    anchor_level : str
        Anchor treatment group.

    Returns
    -------
    dict
        Dictionary containing:

        - ``pairwise``: pairwise AUC for each treatment-group comparison.
        - ``mean_auc``: mean pairwise AUC.
    """
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