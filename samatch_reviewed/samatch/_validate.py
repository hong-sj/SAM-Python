"""
Shared input validation and canonicalisation helpers.

Every SAM entry point routes its inputs through this module so that the same
input is interpreted the same way at every stage. Historically each module
made its own assumptions, which allowed a mismatch to pass through silently
and produce an empty or wrong result rather than an error.
"""

import hashlib

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

    if data[treatment_var].isna().any():
        raise ValueError(
            f"'{treatment_var}' contains missing treatment labels. "
            "Drop or impute these rows before matching."
        )

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
        # `!r` rather than explicit quotes: the whole point of this message is
        # that levels are compared as strings, which is impossible to see if a
        # numeric level is rendered as though it were already one.
        raise ValueError(
            f"no rows found with {treatment_var} == {level!r}. "
            "Treatment levels are compared as strings; check that the level "
            "matches the values in the treatment column."
        )

    return rows


def require_covariates(data, X_vars):
    """Raise if any covariate column is absent from `data`."""
    seen = set()
    duplicates = []

    for covariate in X_vars:
        if covariate in seen and covariate not in duplicates:
            duplicates.append(covariate)
        seen.add(covariate)

    if duplicates:
        raise ValueError(
            "Duplicated covariate column(s) in 'X_vars': "
            + ", ".join(map(str, duplicates))
        )

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


def _ordered_hash(values):
    """Return an order-sensitive digest of a 1-D sequence."""
    codes = pd.util.hash_pandas_object(
        pd.Series(np.asarray(values)),
        index=False,
    ).to_numpy()

    return hashlib.blake2b(codes.tobytes(), digest_size=16).hexdigest()


def _frame_hash(frame):
    """Return an order-sensitive digest of a DataFrame's row values."""
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame,
            index=False,
        ).to_numpy()
    except TypeError:
        # Unrelated metadata columns may legitimately contain lists, dicts, or
        # other unhashable Python objects. Keep them in the identity check by
        # falling back to a type-qualified representation instead of making
        # candidate search reject an otherwise valid analysis frame.
        normalized = pd.DataFrame(
            {
                position: frame.iloc[:, position].map(
                    lambda value: (
                        f"{type(value).__module__}."
                        f"{type(value).__qualname__}:{value!r}"
                    )
                )
                for position in range(frame.shape[1])
            },
            index=frame.index,
        )
        row_hashes = pd.util.hash_pandas_object(
            normalized,
            index=False,
        ).to_numpy()

    return hashlib.blake2b(
        row_hashes.tobytes(), digest_size=16
    ).hexdigest()


def data_fingerprint(data, treatment_var, columns=None):
    """
    Return a cheap fingerprint identifying the exact frame used for matching.

    Matched sets are stored as *positional* row indices, so every stage after
    `gps_candidate_search()` must be handed the same frame in the same order.
    Re-sorting or filtering in between silently repoints those indices at
    different subjects.

    The index, treatment column, and every column present when the fingerprint
    is created are hashed order-sensitively. The full row hash is necessary
    because swapping two subjects within the same treatment group and then
    resetting the index leaves both the index and treatment sequence unchanged.

    When ``columns`` is supplied, only those columns are hashed. This lets the
    validation step ignore columns added after candidate search while still
    verifying that every original row value remains attached to the same
    position.
    """
    if columns is None:
        columns = list(data.columns)
    else:
        columns = list(columns)

    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise ValueError(
            "Column(s) used to identify the original data are missing: "
            + ", ".join(map(str, missing))
        )

    return {
        "n_rows": int(len(data)),
        "index_hash": _ordered_hash(data.index),
        "treatment_hash": _ordered_hash(data[treatment_var].astype(str)),
        "data_columns": columns,
        "data_hash": _frame_hash(data.loc[:, columns]),
    }


def gps_fingerprint(gps):
    """Return an order-sensitive digest of a GPS DataFrame."""
    return {
        "n_rows": int(len(gps)),
        "index_hash": _ordered_hash(gps.index),
        "columns_hash": _ordered_hash(gps.columns),
        "values_hash": _frame_hash(gps),
    }


def validate_gps(data, gps, treatment_var, context):
    """Validate GPS shape, alignment, and probability values."""
    if not isinstance(gps, pd.DataFrame):
        raise TypeError(f"gps passed to {context}() must be a pandas.DataFrame")

    if len(gps) != len(data):
        raise ValueError("gps and data must contain the same number of rows")

    if not gps.index.equals(data.index):
        raise ValueError(
            f"gps and data passed to {context}() must have identical indices "
            "in the same order. GPS values are matched to subjects by row "
            "position."
        )

    if not gps.columns.is_unique:
        raise ValueError("gps column names must be unique")

    if gps.shape[1] == 0:
        raise ValueError("gps must contain at least one treatment column")

    try:
        values = gps.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("gps values must be numeric") from error

    if not np.isfinite(values).all():
        raise ValueError("gps values must all be finite")

    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("gps values must lie in the interval [0, 1]")

    row_sums = values.sum(axis=1)

    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("each gps row must sum to 1")

    # Validate the treatment column here as well, before callers derive group
    # masks from its canonical string representation. Every observed arm must
    # have a score column; otherwise candidate search would silently omit that
    # arm and form lower-dimensional matched sets.
    treatment = treatment_labels(data, treatment_var)
    # `pd.unique` hashes where `np.unique` sorts, and sorting the whole
    # treatment column of strings costs more than everything else in this
    # function. Only the handful of missing levels is sorted, which keeps the
    # message below in the same order it has always been reported in.
    missing_levels = sorted(
        level for level in pd.unique(treatment) if level not in gps.columns
    )

    if missing_levels:
        raise ValueError(
            "Treatment group(s) not found in gps: " + ", ".join(missing_levels)
        )

    return values


def check_data_fingerprint(search, data, treatment_var, context):
    """
    Verify `data` is the frame the candidate search was built from.

    Silently accepts a `search` dict without a fingerprint so that objects
    pickled by earlier versions still work.
    """
    expected = search.get("data_fingerprint") if hasattr(search, "get") else None

    if expected is None or treatment_var not in data.columns:
        return

    # Older search objects contain only these three fields. Compare that common
    # subset so they remain usable, while newer objects also get the stronger
    # full-row identity check below.
    actual = data_fingerprint(
        data,
        treatment_var,
        columns=expected.get("data_columns", list(data.columns)),
    )

    # Naming the part that moved matters: the values hash also trips on an edit
    # to a column matching never touched, and a caller told only that they
    # "re-sorted or filtered" would go looking for something they never did.
    checks = [
        (
            "n_rows",
            f"the row count changed ({expected.get('n_rows')} -> "
            f"{actual['n_rows']}); rows were added or dropped",
        ),
        ("index_hash", "the index changed; it was re-ordered or re-indexed"),
        (
            "treatment_hash",
            "the treatment column changed; rows were re-ordered or recoded",
        ),
        (
            "data_hash",
            "a column value changed; note that this covers every column "
            "present at search time, including ones matching does not use",
        ),
    ]

    reasons = [
        reason
        for key, reason in checks
        if key in expected and actual.get(key) != expected.get(key)
    ]

    if reasons:
        raise ValueError(
            f"the data passed to {context}() does not match the data used by "
            "gps_candidate_search(): "
            + "; ".join(reasons)
            + ". Matched sets reference positional row indices, so the same "
            "DataFrame must be passed unmodified through the whole pipeline."
        )


def check_gps_fingerprint(search, gps, context):
    """Verify that a later pipeline stage received the GPS used for search."""
    expected = search.get("gps_fingerprint") if hasattr(search, "get") else None

    if expected is None:
        return

    if gps_fingerprint(gps) != expected:
        raise ValueError(
            f"the gps passed to {context}() does not match the gps used by "
            "gps_candidate_search(). Use the same GPS DataFrame, with the same "
            "rows, columns, and values, throughout the pipeline."
        )
