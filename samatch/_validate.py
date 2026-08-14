"""
Shared input validation and canonicalisation helpers.

Every SAM entry point routes its inputs through this module so that the same
input is interpreted the same way at every stage. Historically each module
made its own assumptions, which allowed a mismatch to pass through silently
and produce an empty or wrong result rather than an error.
"""

import numpy as np


def require_positive_int(value, name):
    """Raise unless `value` is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")

    return int(value)
