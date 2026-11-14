"""
Generalized propensity score estimation for Shared Anchor Matching.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ._validate import covariate_matrix, treatment_labels, treatment_level

# scikit-learn changed how an unregularized fit is requested. Up to 1.7 it is
# `penalty=None`; 1.8 deprecates that and directs callers to `C=np.inf`, and 1.10
# removes `penalty` altogether. The two produce bit-identical coefficients, so
# which one is used is purely a compatibility matter -- but getting it wrong
# would silently substitute a regularized model, which is the one thing this
# function promises not to be. The signature is the reliable signal: 1.8 marks
# the parameter's default with a deprecation sentinel.
_BENIGN_C_WARNING = "will ignore the C and l1_ratio"


def _unregularized_kwargs():
    """Return the keyword arguments that request an unregularized fit."""
    try:
        parameters = inspect.signature(LogisticRegression).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {"penalty": None}

    if "penalty" in parameters:
        default = parameters["penalty"].default

        if isinstance(default, str) and default == "deprecated":
            return {"C": np.inf}

        return {"penalty": None}

    if "C" in parameters:
        # A real signature that has dropped `penalty`: 1.10 and later.
        return {"C": np.inf}

    # Anything else -- notably a `**kwargs` signature -- says nothing either
    # way. Use the historical spelling and let the fallback below deal with a
    # version that rejects it.
    return {"penalty": None}


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
    regularized_fallback = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        try:
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=max_iter,
                tol=tol,
                **_unregularized_kwargs(),
            )
            model.fit(X_scaled, y)
        except TypeError as error:
            if "penalty" not in str(error):
                raise

            regularized_fallback = True

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

    # Anything else scikit-learn had to say is re-emitted rather than dropped.
    # Recording was only ever meant to intercept the convergence warning, whose
    # wording is replaced below with one that names the parameter to raise;
    # silencing the rest would hide, for example, a data-conversion warning.
    #
    # The one exception is scikit-learn 1.8 answering `C=np.inf` with a notice
    # that C is ignored because the penalty is None. That is precisely what was
    # asked for, so passing it on would be noise on every call.
    for warning in caught:
        if warning in convergence_warnings:
            continue

        if _BENIGN_C_WARNING in str(warning.message):
            continue

        warnings.warn_explicit(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
        )

    if regularized_fallback:
        # Emit this after leaving catch_warnings(record=True). Warning inside
        # that context would be captured in `caught` and never reach callers.
        warnings.warn(
            "This scikit-learn version does not support penalty=None; "
            "falling back to the default L2-regularized model. The "
            "estimated GPS will be shrunk toward zero. Upgrade "
            "scikit-learn to obtain unregularized estimates.",
            RuntimeWarning,
        )

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


def _absorb_standardization(model):
    """
    Rewrite the coefficients so the model applies to unstandardized covariates.

    Standardizing only helps the optimizer; a caller holding the returned model
    should not have to know it happened. Folding the shift and scale into the
    coefficients means `model.predict_proba()` can be applied to the covariates
    as they appear in `data`, and `model.coef_` is on their original scale.
    Without this the model silently returns probabilities for the wrong point,
    since nothing about it signals that an undone transformation is required.

        eta = b0 + sum_j c_j (x_j - m_j) / s_j
            = (b0 - sum_j (c_j / s_j) m_j) + sum_j (c_j / s_j) x_j
    """
    mean = model._sam_standardize_mean
    sd_safe = model._sam_standardize_sd

    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        # A stand-in used only to exercise the version fallback below.
        return model

    model.coef_ = model.coef_ / sd_safe
    model.intercept_ = model.intercept_ - model.coef_ @ mean

    del model._sam_standardize_mean
    del model._sam_standardize_sd

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

    # Predict on the same scale the model was fitted on, then fold the
    # standardization into the coefficients so the returned model needs no
    # such handling from the caller.
    X_scaled = (
        X - model._sam_standardize_mean
    ) / model._sam_standardize_sd

    probabilities = model.predict_proba(X_scaled)

    _absorb_standardization(model)

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
