"""
Generalized propensity score estimation for Shared Anchor Matching.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ._validate import covariate_matrix, treatment_labels, treatment_level


def _fit_unregularized_multinomial_logit(
    X,
    y,
    max_iter=5000,
    tol=1e-10,
):
    """Fit an unregularized multinomial logistic regression."""
    # Standardize covariates to improve numerical conditioning.
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    X_scaled = (X - mean) / sd_safe

    # Request an unregularized model across supported scikit-learn versions.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        try:
            model = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=max_iter,
                tol=tol,
            )
            model.fit(X_scaled, y)
        except TypeError as error:
            if "penalty" not in str(error):
                raise

            # Very old scikit-learn cannot express "no penalty" this way. The
            # fallback is regularized, which contradicts what this function
            # promises, so it must not pass unnoticed.
            warnings.warn(
                "This scikit-learn version does not support penalty=None; "
                "falling back to the default L2-regularized model. The "
                "estimated GPS will be shrunk toward zero. Upgrade "
                "scikit-learn to obtain unregularized estimates.",
                RuntimeWarning,
            )

            model = LogisticRegression(
                solver="lbfgs",
                max_iter=max_iter,
                tol=tol,
            )
            model.fit(X_scaled, y)

        convergence_warnings = [
            warning
            for warning in caught
            if "Convergence" in warning.category.__name__
        ]

    n_iter = getattr(model, "n_iter_", [None])[0]

    if convergence_warnings or (
        n_iter is not None and n_iter >= max_iter
    ):
        warnings.warn(
            "The multinomial GPS model did not converge within "
            f"max_iter={max_iter}. The estimated GPS may be numerically "
            "imprecise. Consider increasing max_iter or checking for "
            "near-separation in the treatment model.",
            RuntimeWarning,
        )

    model._sam_standardize_mean = mean
    model._sam_standardize_sd = sd_safe

    return model


def estimate_gps_multinom(
    data,
    X_vars=None,
    treatment_var="T",
    anchor_level="A",
):
    """
    Estimate generalized propensity scores using multinomial logistic regression.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment and covariate variables.
    X_vars : list of str, optional
        Covariate column names. Defaults to X1 through X10.
    treatment_var : str, default="T"
        Name of the treatment variable.
    anchor_level : str, default="A"
        Anchor treatment group. This group is placed first in the returned
        GPS matrix.

    Returns
    -------
    dict
        Dictionary containing:

        - ``model``: fitted multinomial logistic regression model.
        - ``gps``: predicted treatment probabilities with one column per
          treatment group and the anchor group listed first.
    """
    if X_vars is None:
        X_vars = [f"X{i}" for i in range(1, 11)]

    X = covariate_matrix(data, X_vars)
    y = treatment_labels(data, treatment_var)
    anchor_level = treatment_level(anchor_level)

    treatment_levels = np.unique(y)

    if anchor_level not in treatment_levels:
        raise ValueError(
            f"anchor_level '{anchor_level}' not found in treatment variable"
        )

    model = _fit_unregularized_multinomial_logit(
        X,
        y,
        max_iter=5000,
        tol=1e-10,
    )

    # Apply the same standardization used during model fitting.
    X_scaled = (
        X - model._sam_standardize_mean
    ) / model._sam_standardize_sd

    probabilities = model.predict_proba(X_scaled)

    other_levels = [
        level
        for level in model.classes_
        if level != anchor_level
    ]
    ordered_levels = [anchor_level, *other_levels]

    column_indices = [
        list(model.classes_).index(level)
        for level in ordered_levels
    ]

    gps = pd.DataFrame(
        probabilities[:, column_indices],
        columns=ordered_levels,
        index=data.index,
    )

    return {
        "model": model,
        "gps": gps,
    }