"""
The pipeline rejects a `data` frame that is not the one it matched on.

Matched sets store *positional* row indices. Re-sorting or filtering `data`
between matching and evaluation silently repointed them at other subjects, so
the balance report described a cohort that was never matched.
"""

import numpy as np
import pandas as pd
import pytest

import samatch


@pytest.fixture(scope="module")
def pipeline():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    frame["T"] = rng.choice(list("ABC"), 300)

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=5
    )
    matched = samatch.sam_match(
        frame, search, X_vars=list("abc"), treatment_var="T"
    )
    return frame, fit, search, matched


def test_search_records_a_fingerprint(pipeline):
    _, _, search, _ = pipeline

    assert set(search["data_fingerprint"]) == {
        "n_rows",
        "index_hash",
        "treatment_hash",
    }


MUTATIONS = [
    pytest.param(lambda d: d.sample(frac=1, random_state=0), id="reordered"),
    pytest.param(
        lambda d: d.sample(frac=1, random_state=0).reset_index(drop=True),
        id="reordered_then_reindexed",
    ),
    pytest.param(lambda d: d.iloc[:-1], id="row_dropped"),
    pytest.param(
        lambda d: d.sort_values("a"), id="sorted_by_covariate"
    ),
    pytest.param(
        lambda d: pd.concat([d, d.iloc[:1]], ignore_index=True), id="row_added"
    ),
]


@pytest.mark.parametrize("mutate", MUTATIONS)
def test_sam_evaluate_rejects_a_modified_frame(pipeline, mutate):
    frame, fit, search, matched = pipeline

    with pytest.raises(ValueError, match="does not match the data"):
        samatch.sam_evaluate(
            mutate(frame),
            search,
            matched,
            fit["gps"],
            X_vars=list("abc"),
            treatment_var="T",
        )


@pytest.mark.parametrize("mutate", MUTATIONS)
def test_extract_matched_data_rejects_a_modified_frame(pipeline, mutate):
    frame, _, search, matched = pipeline

    with pytest.raises(ValueError, match="does not match the data"):
        samatch.extract_matched_data(
            mutate(frame), search, matched, treatment_var="T"
        )


def test_sam_match_rejects_a_modified_frame(pipeline):
    frame, _, search, _ = pipeline

    with pytest.raises(ValueError, match="does not match the data"):
        samatch.sam_match(
            frame.sample(frac=1, random_state=0),
            search,
            X_vars=list("abc"),
            treatment_var="T",
        )


def test_reordering_within_a_group_is_detected(pipeline):
    """The treatment column alone would not change; the index order does."""
    frame, fit, search, matched = pipeline

    reordered = pd.concat(
        [frame[frame["T"] != "A"], frame[frame["T"] == "A"]]
    ).reset_index(drop=True)

    with pytest.raises(ValueError, match="does not match the data"):
        samatch.sam_evaluate(
            reordered,
            search,
            matched,
            fit["gps"],
            X_vars=list("abc"),
            treatment_var="T",
        )


def test_adding_a_column_between_stages_is_allowed(pipeline):
    """Only row order matters, so attaching an outcome must stay legal."""
    frame, fit, search, matched = pipeline

    report = samatch.sam_evaluate(
        frame.assign(outcome=1),
        search,
        matched,
        fit["gps"],
        X_vars=list("abc"),
        treatment_var="T",
    )

    assert report["matching_rate"] == matched["matching_rate"]


def test_the_unmodified_frame_still_works(pipeline):
    frame, fit, search, matched = pipeline

    report = samatch.sam_evaluate(
        frame,
        search,
        matched,
        fit["gps"],
        X_vars=list("abc"),
        treatment_var="T",
    )

    assert report["smd_balance"]["summary"]["max_abs_smd"].notna().all()


def test_a_search_without_a_fingerprint_is_still_accepted(pipeline):
    """Objects produced by earlier versions must keep working."""
    frame, fit, search, matched = pipeline

    legacy = {k: v for k, v in search.items() if k != "data_fingerprint"}

    report = samatch.sam_evaluate(
        frame,
        legacy,
        matched,
        fit["gps"],
        X_vars=list("abc"),
        treatment_var="T",
    )

    assert report["matching_rate"] == matched["matching_rate"]


def test_mixed_anchor_rows_are_reported(pipeline):
    """Anchor level is inferred from every anchor row, not just the first."""
    frame, _, search, matched = pipeline

    legacy = {k: v for k, v in search.items() if k != "data_fingerprint"}
    shuffled = frame.sample(frac=1, random_state=0)

    with pytest.raises(ValueError, match="Could not uniquely determine"):
        samatch.extract_matched_data(
            shuffled, legacy, matched, treatment_var="T"
        )
