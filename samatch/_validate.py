"""
Shared input validation and canonicalisation helpers.

Every SAM entry point routes its inputs through this module so that the same
input is interpreted the same way at every stage. Historically each module
made its own assumptions, which allowed a mismatch to pass through silently
and produce an empty or wrong result rather than an error.
"""

import numpy as np
import pandas as pd


def treatment_labels(data, treatment_var):
    """
    Return the treatment column as a canonical array of string labels.

    `estimate_gps_multinom()` casts the treatment column to `str` before
    fitting, so every GPS column label and every group name flowing through
    the pipeline is a string. Comparing those labels against a raw numeric
    treatment column never matches, which previously yielded an empty match
    with no error. Canonicalising here keeps every stage on the same
    representation.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the treatment variable.
    treatment_var : str
        Name of the treatment variable.

    Returns
    -------
    numpy.ndarray
        Treatment labels as strings.
    """
    if treatment_var not in data.columns:
        raise ValueError(f"'{treatment_var}' not found in data")

    return data[treatment_var].astype(str).to_numpy()


def treatment_level(anchor_level):
    """Return a treatment level in the canonical string representation."""
    return None if anchor_level is None else str(anchor_level)


def require_rows(rows, level, treatment_var):
    """
    Raise if a treatment level matched no rows.

    An empty selection means the requested level is not present under the
    canonical string representation, which otherwise propagates silently as
    an empty matched set.
    """
    if len(rows) == 0:
        raise ValueError(
            f"no rows found with {treatment_var} == '{level}'. "
            "Treatment levels are compared as strings; check that the level "
            "matches the values in the treatment column."
        )

    return rows


def require_covariates(data, X_vars):
    """Raise if any covariate column is absent from `data`."""
    missing = [covariate for covariate in X_vars if covariate not in data.columns]

    if missing:
        raise ValueError(
            "Covariate column(s) not found in data: " + ", ".join(missing)
        )


def covariate_matrix(data, X_vars, rows=None):
    """
    Return covariates as a float matrix, validating that they are usable.

    Non-numeric and non-finite covariates are rejected here with a message
    naming the offending columns. Reaching the linear algebra with `NaN`
    present is not detectable downstream: `numpy.linalg.inv` returns an
    all-`NaN` matrix without raising, so a single missing value silently
    destroys every Mahalanobis distance.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the covariates.
    X_vars : list of str
        Covariate column names.
    rows : array-like of int, optional
        Positional row indices to restrict the returned matrix to.
        Validation always covers the whole column, since the pooled
        covariance uses every row.

    Returns
    -------
    numpy.ndarray
        Covariate matrix of shape ``(n_rows, len(X_vars))``.
    """
    require_covariates(data, X_vars)

    frame = data[X_vars]

    non_numeric = [
        covariate
        for covariate in X_vars
        if not pd.api.types.is_numeric_dtype(frame[covariate])
    ]

    if non_numeric:
        raise ValueError(
            "Covariate column(s) are not numeric: "
            + ", ".join(non_numeric)
            + ". Categorical covariates must be encoded (for example with "
            "pandas.get_dummies) before being passed as X_vars."
        )

    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values)

    if not finite.all():
        offending = [
            f"{covariate} ({int((~finite[:, position]).sum())} rows)"
            for position, covariate in enumerate(X_vars)
            if not finite[:, position].all()
        ]
        raise ValueError(
            "Covariate column(s) contain missing or non-finite values: "
            + ", ".join(offending)
            + ". SAM does not impute; drop or impute these rows before "
            "matching."
        )

    return values if rows is None else values[np.asarray(rows, dtype=int)]


def require_positive_int(value, name):
    """Raise unless `value` is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")

    return int(value)
