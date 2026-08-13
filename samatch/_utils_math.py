"""
Mathematical utilities for Shared Anchor Matching.
"""

import numpy as np
from scipy.stats import rankdata


def expit(x):
    """
    Compute the inverse-logit transformation.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    numpy.ndarray
        Values transformed to the interval (0, 1).
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    """
    Compute the logit transformation.

    Parameters
    ----------
    p : array-like
        Probabilities in the interval (0, 1).

    Returns
    -------
    numpy.ndarray
        Log-odds values.
    """
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def auc_mannwhitney(score, label):
    """
    Compute AUC using the Mann-Whitney U statistic.

    Parameters
    ----------
    score : array-like
        Continuous discrimination scores.
    label : array-like
        Binary labels coded as 0 and 1.

    Returns
    -------
    float
        Area under the ROC curve, or NaN if either class is absent.
    """
    score = np.asarray(score, dtype=float)
    label = np.asarray(label)

    n_positive = int(np.sum(label == 1))
    n_negative = int(np.sum(label == 0))

    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = rankdata(score)
    rank_sum_positive = ranks[label == 1].sum()

    u_statistic = (
        rank_sum_positive
        - n_positive * (n_positive + 1) / 2.0
    )

    return u_statistic / (n_positive * n_negative)