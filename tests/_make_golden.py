"""
Regenerate the golden fixtures used by ``tests/test_golden.py``.

The fixtures pin the *current* numerical output of the full SAM pipeline so
that refactors which are supposed to be behaviour-preserving (vectorisation,
shared-helper extraction, KD-tree candidate search) can be proven not to have
changed any result.

Run from the repository root::

    python tests/_make_golden.py

Only run this when a change is *intended* to alter results, and review the
resulting diff carefully.
"""

import pathlib

import samatch

from _golden_pipeline import run_four_group, run_three_group

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"


def main():
    GOLDEN_DIR.mkdir(exist_ok=True)

    for name, frame in run_four_group().items():
        frame.to_csv(GOLDEN_DIR / f"four_{name}.csv", index=False)

    for name, frame in run_three_group().items():
        frame.to_csv(GOLDEN_DIR / f"three_{name}.csv", index=False)

    print(f"golden fixtures written to {GOLDEN_DIR} (samatch {samatch.__version__})")


if __name__ == "__main__":
    main()
