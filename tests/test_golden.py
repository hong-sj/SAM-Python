"""
Golden regression tests.

These pin the exact numerical output of the full SAM pipeline on the bundled
example datasets. They exist so that refactors intended to be
behaviour-preserving (vectorisation, shared-helper extraction, KD-tree
candidate search) can be *proven* not to have moved any number.

If one of these fails, either the change was not behaviour-preserving or the
fixtures need regenerating deliberately via ``python tests/_make_golden.py``.
"""

import pathlib

import pandas as pd
import pytest

from _golden_pipeline import run_four_group, run_three_group

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"

FOUR_GROUP_TABLES = [
    "gps",
    "matched",
    "smd",
    "auc",
    "group_risk",
    "contrasts",
    "weighted_balance",
    "ess",
]

THREE_GROUP_TABLES = ["gps", "matched"]


@pytest.fixture(scope="module")
def four_group_result():
    return run_four_group()


@pytest.fixture(scope="module")
def three_group_result():
    return run_three_group()


def _assert_matches_golden(actual, path):
    if not path.exists():
        pytest.skip(f"golden fixture missing: {path.name} (run tests/_make_golden.py)")

    expected = pd.read_csv(path)
    actual = actual.reset_index(drop=True)

    assert list(actual.columns) == list(expected.columns), (
        f"{path.name}: column set changed"
    )
    assert len(actual) == len(expected), f"{path.name}: row count changed"

    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            pd.testing.assert_series_equal(
                actual[column].astype(float),
                expected[column].astype(float),
                check_names=False,
                rtol=1e-10,
                atol=1e-12,
            )
        else:
            pd.testing.assert_series_equal(
                actual[column].astype(str),
                expected[column].astype(str),
                check_names=False,
            )


@pytest.mark.parametrize("table", FOUR_GROUP_TABLES)
def test_four_group_pipeline_matches_golden(four_group_result, table):
    _assert_matches_golden(four_group_result[table], GOLDEN_DIR / f"four_{table}.csv")


@pytest.mark.parametrize("table", THREE_GROUP_TABLES)
def test_three_group_pipeline_matches_golden(three_group_result, table):
    _assert_matches_golden(three_group_result[table], GOLDEN_DIR / f"three_{table}.csv")


def test_pipeline_is_deterministic():
    """Two runs on identical input must produce byte-identical matched sets.

    Guards against accidental nondeterminism creeping in via dict/set
    iteration order during refactoring.
    """
    first = run_four_group()["matched"]
    second = run_four_group()["matched"]
    pd.testing.assert_frame_equal(first, second)
