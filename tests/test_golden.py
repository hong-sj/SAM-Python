"""
Golden regression tests.

These pin the numerical output of the full SAM pipeline on the bundled example
datasets, so that refactors intended to be behaviour-preserving can be *proven*
not to have moved any number.

How strictly they can be compared depends on the environment. The GPS is an
LBFGS fit and the outcome model a Newton-Raphson solve; both stop on a
convergence tolerance, so their last digits belong to a particular numerical
stack. On the stack the fixtures were generated on, everything is compared to
full precision and matched sets must be identical row for row. Elsewhere, a
different scikit-learn reaches a slightly different optimum -- enough, on the
three-group data, to change which subjects end up matched -- so the estimates
are compared loosely and the matched sets are not compared at all. The rest of
the suite carries correctness on those environments: it validates against
brute-force references and hand-computed values rather than pinned output.

If a strict comparison fails, either the change was not behaviour-preserving or
the fixtures need regenerating deliberately via `python tests/_make_golden.py`.
"""

import pathlib

import pandas as pd
import pytest

from _golden_environment import (
    environment_summary,
    is_reference_environment,
    reference_environment,
)
from _golden_pipeline import run_four_group, run_three_group

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"

# Tables whose values are estimates. Comparable, with a tolerance that depends
# on whether the solvers behind them are the ones that produced the fixtures.
ESTIMATE_TABLES = {
    "four": ["gps", "smd", "auc", "group_risk", "contrasts", "weighted_balance", "ess"],
    "three": ["gps"],
}

# Tables whose values are decisions -- which subject was matched to which.
# These are integers, so there is no useful loose comparison: once the GPS
# input shifts, a near-tie can resolve the other way and the rows describe
# different pairs entirely.
DECISION_TABLES = {"four": ["matched"], "three": ["matched"]}

STRICT = dict(rtol=1e-9, atol=1e-12)

# Off the reference environment the loose tier exists to catch a gross
# regression -- a sign flip, a shifted formula -- not solver drift. It needs an
# absolute floor as well as a relative one, because these tables hold
# standardized mean differences and probabilities, and a value near zero makes
# relative error meaningless: scikit-learn 1.6.1 vs 1.8.0 moves an SMD of
# -0.0165 by 4.8e-05, which is 0.3% relatively but 2,000x below the 0.1
# threshold anyone reads an SMD against.
LOOSE = dict(rtol=1e-3, atol=1e-3)


@pytest.fixture(scope="module")
def four_group_result():
    return run_four_group()


@pytest.fixture(scope="module")
def three_group_result():
    return run_three_group()


def _load(path):
    if not path.exists():
        pytest.skip(f"golden fixture missing: {path.name} (run tests/_make_golden.py)")

    return pd.read_csv(path)


def _assert_shape(actual, expected, name):
    assert list(actual.columns) == list(expected.columns), f"{name}: column set changed"
    assert len(actual) == len(expected), f"{name}: row count changed"


def _assert_values(actual, expected, name, tolerance):
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            pd.testing.assert_series_equal(
                actual[column].astype(float),
                expected[column].astype(float),
                check_names=False,
                **tolerance,
            )
        else:
            pd.testing.assert_series_equal(
                actual[column].astype(str),
                expected[column].astype(str),
                check_names=False,
            )


def _check(result, prefix, table, strict_only):
    path = GOLDEN_DIR / f"{prefix}_{table}.csv"
    expected = _load(path)
    actual = result[table].reset_index(drop=True)

    # Shape is stack-independent, so it is always checked.
    _assert_shape(actual, expected, path.name)

    if is_reference_environment():
        _assert_values(actual, expected, path.name, STRICT)
        return

    if strict_only:
        pytest.skip(
            f"matched sets are only comparable on the environment the fixtures "
            f"were generated on; here {environment_summary()}"
        )

    _assert_values(actual, expected, path.name, LOOSE)


def test_reference_environment_is_recorded():
    """Without this the tests cannot know how strictly to compare."""
    assert reference_environment() is not None, (
        "tests/golden/ENVIRONMENT.json is missing; run tests/_make_golden.py"
    )


@pytest.mark.parametrize("table", ESTIMATE_TABLES["four"])
def test_four_group_estimates_match_golden(four_group_result, table):
    _check(four_group_result, "four", table, strict_only=False)


@pytest.mark.parametrize("table", DECISION_TABLES["four"])
def test_four_group_matched_sets_match_golden(four_group_result, table):
    _check(four_group_result, "four", table, strict_only=True)


@pytest.mark.parametrize("table", ESTIMATE_TABLES["three"])
def test_three_group_estimates_match_golden(three_group_result, table):
    _check(three_group_result, "three", table, strict_only=False)


@pytest.mark.parametrize("table", DECISION_TABLES["three"])
def test_three_group_matched_sets_match_golden(three_group_result, table):
    _check(three_group_result, "three", table, strict_only=True)


def test_pipeline_is_deterministic():
    """Two runs on identical input must produce identical matched sets.

    Unlike the comparisons above this needs no fixture, so it holds on every
    environment. It guards against nondeterminism creeping in through dict or
    set iteration order during refactoring.
    """
    first = run_four_group()["matched"]
    second = run_four_group()["matched"]
    pd.testing.assert_frame_equal(first, second)
