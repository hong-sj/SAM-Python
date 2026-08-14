"""
Shared input validation and canonicalisation helpers.

Every SAM entry point routes its inputs through this module so that the same
input is interpreted the same way at every stage. Historically each module
made its own assumptions, which allowed a mismatch to pass through silently
and produce an empty or wrong result rather than an error.
"""

import numpy as np


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


def require_positive_int(value, name):
    """Raise unless `value` is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")

    return int(value)
