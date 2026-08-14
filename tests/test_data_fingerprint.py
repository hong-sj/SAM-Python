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
        "data_columns",
        "data_hash",
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


def test_reordering_within_a_group_then_resetting_index_is_detected(pipeline):
    """Index and treatment hashes alone cannot detect this permutation."""
    frame, fit, search, matched = pipeline

    positions = np.flatnonzero(frame["T"].to_numpy() == frame.loc[0, "T"])
    order = np.arange(len(frame))
    order[positions[:2]] = order[positions[:2]][::-1]
    reordered = frame.iloc[order].reset_index(drop=True)

    assert reordered.index.equals(frame.index)
    np.testing.assert_array_equal(reordered["T"], frame["T"])

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


def test_unhashable_metadata_does_not_break_fingerprinting():
    frame = pd.DataFrame(
        {
            "T": ["A", "B", "B"],
            "metadata": [["anchor"], {"site": 1}, ["comparator"]],
        }
    )
    gps = pd.DataFrame(
        {"A": [0.8, 0.3, 0.2], "B": [0.2, 0.7, 0.8]},
        index=frame.index,
    )

    search = samatch.gps_candidate_search(frame, gps, anchor_level="A")
    assert "data_hash" in search["data_fingerprint"]


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


def test_a_search_with_the_legacy_fingerprint_is_still_accepted(pipeline):
    """Search objects from before full-row hashing remain usable."""
    frame, fit, search, matched = pipeline
    legacy = dict(search)
    legacy["data_fingerprint"] = {
        key: search["data_fingerprint"][key]
        for key in ("n_rows", "index_hash", "treatment_hash")
    }

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


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda f: f.iloc[:-1], "row count changed"),
        (lambda f: f.sort_values("a"), "index changed"),
        (lambda f: f.assign(T=f["T"].str.lower()), "treatment column changed"),
        (lambda f: f.assign(a=f["a"].round(3)), "column value changed"),
    ],
)
def test_the_rejection_names_what_moved(pipeline, mutate, expected):
    """
    The values hash also trips on an edit to a column matching never uses, so a
    message that only says "re-sorted or filtered" sends the reader looking for
    something they did not do. Each axis has to be reported on its own.
    """
    frame, fit, search, matched = pipeline

    with pytest.raises(ValueError, match=expected):
        samatch.sam_evaluate(
            mutate(frame),
            search,
            matched,
            fit["gps"],
            X_vars=list("abc"),
            treatment_var="T",
        )


def test_editing_an_unused_column_says_so_rather_than_blaming_the_row_order():
    """
    Recoding a column nothing in the pipeline reads is still rejected -- the
    full-row hash is what catches two same-treatment subjects being swapped --
    but it must be reported as a value change, not as a re-ordering.
    """
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    frame["T"] = rng.choice(list("ABC"), 120)
    frame["subject_id"] = np.arange(120)

    fit = samatch.estimate_gps_multinom(
        frame, X_vars=list("abc"), treatment_var="T", anchor_level="A"
    )
    search = samatch.gps_candidate_search(
        frame, fit["gps"], treatment_var="T", anchor_level="A", top_m=5
    )
    matched = samatch.sam_match(
        frame, search, X_vars=list("abc"), treatment_var="T"
    )

    recoded = frame.assign(subject_id=frame["subject_id"].astype(str))

    with pytest.raises(ValueError) as error:
        samatch.sam_evaluate(
            recoded,
            search,
            matched,
            fit["gps"],
            X_vars=list("abc"),
            treatment_var="T",
        )

    message = str(error.value)

    assert "column value changed" in message
    assert "including ones matching does not use" in message
    assert "row count changed" not in message
    assert "index changed" not in message
