"""
Which dependency build the golden fixtures were produced by.

The generalized propensity score is an LBFGS fit and the outcome model a
Newton-Raphson solve. Both stop on a convergence tolerance rather than at
machine precision, so their last digits belong to a particular numerical
stack: a different scikit-learn reaches a slightly different optimum, and a
different BLAS accumulates differently. Comparing the fixtures to full
precision is therefore only meaningful on the stack that generated them.
"""

import json
import pathlib

import numpy
import pandas
import scipy
import sklearn

ENVIRONMENT_FILE = pathlib.Path(__file__).parent / "golden" / "ENVIRONMENT.json"


def current_environment():
    """Return the dependency versions that determine the fixture values."""
    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def reference_environment():
    """Return the environment the fixtures were generated on, if recorded."""
    if not ENVIRONMENT_FILE.exists():
        return None

    return json.loads(ENVIRONMENT_FILE.read_text())


def is_reference_environment():
    """True when the fixtures can be compared to full precision."""
    reference = reference_environment()
    return reference is not None and reference == current_environment()


def environment_summary():
    """A one-line description of how the current stack differs, for messages."""
    reference = reference_environment() or {}
    current = current_environment()

    differences = [
        f"{package} {reference.get(package, '?')} -> {version}"
        for package, version in sorted(current.items())
        if reference.get(package) != version
    ]

    return ", ".join(differences) if differences else "identical"
