"""
Pairwise treatment-discrimination AUC diagnostics.

Two failure modes are pinned here:

1. An unregularized GPS can be exactly 0 or 1 under complete separation.
   With three or more groups a subject can score 0 in *both* members of a
   pair, so the log-ratio is ``log(0) - log(0) = NaN``. scipy's rankdata
   then returns all-NaN ranks and the AUC silently becomes NaN -- no error,
   no warning. The scores must be clipped so every AUC stays finite.

2. A pair's AUC is directional: 0.6 for (A, B) and 0.4 for the same two
   groups scored the other way describe identical discrimination. Averaging
   raw AUCs lets opposite-direction pairs cancel toward 0.5 and understate
   residual separation, so ``mean_auc`` folds each AUC to
   ``max(auc, 1 - auc)``.
"""

import numpy as np
import pandas as pd
import pytest

import samatch


def _matched_frame(n_sets, groups):
    """
    A matched-set frame with one row per set.

    Role columns point at contiguous blocks of gps rows: the anchor block
    first, then one block per comparator group. Tests that need a different
    row layout (for example interleaved) override the role columns directly.
    """
    frame = {"matched_set_id": np.arange(1, n_sets + 1)}

    start = 0
    frame["anchor"] = np.arange(start, start + n_sets)
    start += n_sets

    for group in groups:
        frame[group] = np.arange(start, start + n_sets)
        start += n_sets

    return pd.DataFrame(frame)


def test_zero_gps_in_both_groups_of_a_pair_keeps_auc_finite():
    # Subject 0 is an anchor subject whose GPS is exactly 0 in both A and B
    # (possible with three groups: the mass sits entirely in C). Unclipped,
    # its log-ratio for the (A, B) pair is NaN.
    gps = pd.DataFrame(
        [
            [0.0, 0.0, 1.0],   # anchor subject, zero in A and B
            [0.2, 0.6, 0.2],   # comparator B
            [0.8, 0.1, 0.1],   # anchor subject
            [0.1, 0.6, 0.3],   # comparator B
        ],
        columns=["A", "B", "C"],
    )

    matched = _matched_frame(2, ["B"])  # anchor + B only
    matched["anchor"] = [0, 2]
    matched["B"] = [1, 3]

    result = samatch.compute_pairwise_treatment_auc(
        gps, matched, groups=["B"], anchor_level="A"
    )

    auc = result["pairwise"]["auc"].iloc[0]

    assert np.isfinite(auc)
    # The zero/zero subject ranks at the extreme (score 0 after clipping),
    # between the two B subjects and below the other anchor subject.
    assert auc == pytest.approx(1.0)


def test_mean_auc_does_not_cancel_opposite_directions():
    # A deliberately reversed model: anchor subjects carry a *low* A/B ratio,
    # so the (A, B) and (A, C) log-ratios point against the anchor while
    # (B, C) points with it. A fitted GPS produces sub-0.5 AUCs only through
    # noise around chance, but the summary must not cancel either kind.
    gps = pd.DataFrame(
        [
            [0.2, 0.7, 0.1],   # anchor: low A/B and moderate A/C ratios
            [0.2, 0.7, 0.1],   # anchor
            [0.7, 0.2, 0.1],   # B: high A/B and A/C ratios
            [0.7, 0.2, 0.1],   # B
            [0.7, 0.1, 0.2],   # C: high A/C ratio, moderate B/C
            [0.7, 0.1, 0.2],   # C
        ],
        columns=["A", "B", "C"],
    )

    matched = _matched_frame(2, ["B", "C"])

    result = samatch.compute_pairwise_treatment_auc(
        gps, matched, groups=["B", "C"], anchor_level="A"
    )

    aucs = result["pairwise"]["auc"].to_numpy()

    assert aucs[0] == pytest.approx(0.0)  # (A, B): B dominates
    assert aucs[1] == pytest.approx(0.0)  # (A, C): C dominates
    assert aucs[2] == pytest.approx(1.0)  # (B, C): B dominates

    raw_mean = aucs.mean()
    assert abs(raw_mean - 1.0 / 3.0) < 1e-9  # the cancellation the fix removes

    assert result["mean_auc"] == pytest.approx(1.0)


def test_mean_auc_is_unchanged_when_all_pairs_point_the_same_way():
    # Backward compatibility: when every pair discriminates in the same
    # direction, max(auc, 1 - auc) == auc and the summary equals the old
    # raw mean.
    gps = pd.DataFrame(
        [
            [0.8, 0.1, 0.1],   # anchor: high A/B and A/C ratios
            [0.8, 0.1, 0.1],   # anchor
            [0.2, 0.6, 0.2],   # B: low A/B ratio, moderate B/C
            [0.2, 0.6, 0.2],   # B
            [0.1, 0.3, 0.6],   # C: low A/C and B/C ratios
            [0.1, 0.3, 0.6],   # C
        ],
        columns=["A", "B", "C"],
    )

    matched = _matched_frame(2, ["B", "C"])

    result = samatch.compute_pairwise_treatment_auc(
        gps, matched, groups=["B", "C"], anchor_level="A"
    )

    aucs = result["pairwise"]["auc"].to_numpy()

    assert np.all(aucs > 0.5)
    assert result["mean_auc"] == pytest.approx(aucs.mean())


def test_pipeline_mean_auc_lies_between_half_and_one():
    # For any non-empty result, max(auc, 1 - auc) is in [0.5, 1], so the
    # summary can never dip below chance even when pairs point both ways.
    data = samatch.load_sample_4group()

    covariates = [
        column
        for column in data.columns
        if column not in ("synthetic_id", "treatment", "mortality_28d")
    ]

    anchor = data["treatment"].astype(str).unique()[0]

    fit = samatch.estimate_gps_multinom(
        data, X_vars=covariates, treatment_var="treatment", anchor_level=anchor
    )

    search = samatch.gps_candidate_search(
        data,
        fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
        top_m=10,
        gps_space="logit",
    )

    matched = samatch.sam_match(
        data, search, X_vars=covariates, treatment_var="treatment"
    )

    report = samatch.sam_evaluate(
        data, search, matched, fit["gps"], X_vars=covariates, treatment_var="treatment"
    )

    mean_auc = report["treatment_discrimination_auc"]["mean_auc"]

    assert 0.5 <= mean_auc <= 1.0
