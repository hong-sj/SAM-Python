"""
Weighting robustness and outcome-model diagnostics.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import samatch


@pytest.fixture(scope="module")
def four_group():
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
    return data, covariates, anchor, fit


# Positivity trimming ----------------------------------------------------------


def test_no_trimming_on_well_behaved_data(four_group):
    data, covariates, anchor, fit = four_group

    for method in ("iptw", "overlap", "matching"):
        result = samatch.compute_balancing_weights(
            data,
            method=method,
            gps=fit["gps"],
            treatment_var="treatment",
            anchor_level=anchor,
        )
        assert result["n_trimmed"] == 0
        assert np.isfinite(result["weights"]).all()


def test_extreme_propensity_scores_are_bounded():
    """A near-zero own-GPS must not produce an unbounded weight."""
    # The last subject is in arm A but has almost no probability of it, so its
    # own propensity score is what the weight divides by.
    frame = pd.DataFrame({"T": ["A", "A", "B", "A"]})

    gps = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.5, 1e-12],
            "B": [0.5, 0.5, 0.5, 1.0 - 1e-12],
        }
    )

    untrimmed = samatch.compute_balancing_weights(
        frame, method="iptw", gps=gps, treatment_var="T", anchor_level="A", trim=0
    )
    trimmed = samatch.compute_balancing_weights(
        frame, method="iptw", gps=gps, treatment_var="T", anchor_level="A"
    )

    assert untrimmed["n_trimmed"] == 0
    assert trimmed["n_trimmed"] == 1

    # Without trimming the fourth subject carries ~1e12 times its share.
    assert untrimmed["weights"].max() > 1e9
    assert trimmed["weights"].max() < 1e4
    assert np.isfinite(trimmed["weights"]).all()


def test_zero_propensity_score_produces_infinity_without_trimming():
    frame = pd.DataFrame({"T": ["A", "A", "B", "A"]})
    gps = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.0], "B": [0.5, 0.5, 0.5, 1.0]})

    with np.errstate(divide="ignore"):
        untrimmed = samatch.compute_balancing_weights(
            frame,
            method="iptw",
            gps=gps,
            treatment_var="T",
            anchor_level="A",
            trim=0,
        )

    assert not np.isfinite(untrimmed["weights"]).all()

    trimmed = samatch.compute_balancing_weights(
        frame, method="iptw", gps=gps, treatment_var="T", anchor_level="A"
    )

    assert np.isfinite(trimmed["weights"]).all()
    assert trimmed["n_trimmed"] == 1


@pytest.mark.parametrize("trim", [-0.1, 1.0, 2.0, np.nan, np.inf, True, "0.1"])
def test_invalid_trim_raises(four_group, trim):
    data, _, anchor, fit = four_group

    with pytest.raises(ValueError, match=r"trim must be in \[0, 1\)"):
        samatch.compute_balancing_weights(
            data,
            gps=fit["gps"],
            treatment_var="treatment",
            anchor_level=anchor,
            trim=trim,
        )


def test_weighted_balance_with_uniform_weights_equals_unweighted(four_group):
    """Uniform weights must reproduce the plain group means and variances."""
    data, covariates, anchor, _ = four_group

    balance = samatch.compute_weighted_balance(
        data,
        np.ones(len(data)),
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    treatment = data["treatment"].astype(str).to_numpy()
    anchor_mask = treatment == anchor

    for _, row in balance["by_covariate"].iterrows():
        x = data[row["covariate"]].to_numpy(dtype=float)
        group_mask = treatment == row["group"]

        # compute_weighted_balance uses the population variance (ddof=0).
        pooled_sd = np.sqrt(
            (x[anchor_mask].var(ddof=0) + x[group_mask].var(ddof=0)) / 2
        )
        expected = (x[anchor_mask].mean() - x[group_mask].mean()) / pooled_sd

        assert row["smd"] == pytest.approx(expected)


def test_evaluate_comparator_weighting_reports_trimming(four_group):
    data, covariates, anchor, fit = four_group

    result = samatch.evaluate_comparator_weighting(
        data,
        method="overlap",
        gps=fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    assert result["n_trimmed"] == 0
    assert (result["balance"]["summary"]["n_undefined"] == 0).all()


def test_numeric_anchor_level_is_canonicalized_for_weighted_balance():
    frame = pd.DataFrame(
        {"T": [0, 0, 1, 1], "x": [0.0, 1.0, 2.0, 3.0]}
    )

    numeric = samatch.compute_weighted_balance(
        frame,
        np.ones(len(frame)),
        X_vars=["x"],
        treatment_var="T",
        anchor_level=0,
    )
    string = samatch.compute_weighted_balance(
        frame,
        np.ones(len(frame)),
        X_vars=["x"],
        treatment_var="T",
        anchor_level="0",
    )

    pd.testing.assert_frame_equal(numeric["by_covariate"], string["by_covariate"])
    assert numeric["by_covariate"]["smd_defined"].all()


@pytest.mark.parametrize(
    "weights, message",
    [
        ([1.0, np.nan, 1.0, 1.0], "finite"),
        ([1.0, -1.0, 1.0, 1.0], "non-negative"),
        ([0.0, 0.0, 1.0, 1.0], "positive sum"),
    ],
)
def test_weighted_balance_rejects_invalid_weights(weights, message):
    frame = pd.DataFrame(
        {"T": ["A", "A", "B", "B"], "x": [0.0, 1.0, 2.0, 3.0]}
    )

    with pytest.raises(ValueError, match=message):
        samatch.compute_weighted_balance(
            frame,
            weights,
            X_vars=["x"],
            treatment_var="T",
            anchor_level="A",
        )


# Effective sample size --------------------------------------------------------


def test_effective_sample_size_matches_the_formula(four_group):
    data, _, anchor, fit = four_group

    weights = samatch.compute_balancing_weights(
        data,
        method="overlap",
        gps=fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
    )["weights"]

    treatment = data["treatment"].astype(str).to_numpy()
    ess = samatch.compute_effective_sample_size(weights, treatment=treatment)

    for _, row in ess.iterrows():
        w = weights[treatment == row["group"]]
        assert row["ess"] == pytest.approx(w.sum() ** 2 / (w**2).sum())
        assert row["ess"] <= row["n"] + 1e-9


def test_uniform_weights_give_ess_equal_to_n():
    weights = np.ones(50)
    assert samatch.compute_effective_sample_size(weights) == pytest.approx(50.0)


# Outcome model ----------------------------------------------------------------


def test_separation_is_flagged_on_the_contrasts_frame():
    """A zero-event arm must be marked, not only warned about."""
    n_sets = 30

    frame = pd.DataFrame(
        {
            "matched_set_id": np.repeat(np.arange(1, n_sets + 1), 2),
            "treatment": ["A", "B"] * n_sets,
            # Group B never experiences the outcome.
            "y": [1, 0] * (n_sets // 2) + [0, 0] * (n_sets - n_sets // 2),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = samatch.sam_estimate_effects(
            frame, outcome_var="y", treatment_var="treatment", anchor_level="A"
        )

    assert effects["contrasts"]["separation"].all()


def test_no_separation_flag_on_the_sample_data(four_group):
    data, covariates, anchor, fit = four_group

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
    matched_data = samatch.extract_matched_data(
        data, search, matched, treatment_var="treatment", anchor_level=anchor
    )
    effects = samatch.sam_estimate_effects(
        matched_data,
        outcome_var="mortality_28d",
        treatment_var="treatment",
        anchor_level=anchor,
    )

    assert not effects["contrasts"]["separation"].any()
    assert np.isfinite(effects["contrasts"]["se_log_or"].to_numpy()).all()


def test_non_binary_outcome_is_rejected():
    frame = pd.DataFrame(
        {
            "matched_set_id": [1, 1, 2, 2],
            "treatment": ["A", "B", "A", "B"],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError):
        samatch.sam_estimate_effects(
            frame, outcome_var="y", treatment_var="treatment", anchor_level="A"
        )


# Confidence-interval methods --------------------------------------------------


def _two_arm_frame(n_sets):
    """Matched sets with a 50/50 outcome in each arm, alternating by set.

    The treatment-only model is exactly identified at beta = (0, 0) with
    mu = 1/2 everywhere, so every cluster score is nonzero and the
    intervals below are non-degenerate.
    """
    rows = []

    for s in range(1, n_sets + 1):
        rows.append({"matched_set_id": s, "treatment": "A", "y": s % 2})
        rows.append({"matched_set_id": s, "treatment": "B", "y": 1 - (s % 2)})

    return pd.DataFrame(rows)


def _effects(frame, **kwargs):
    return samatch.sam_estimate_effects(
        frame, outcome_var="y", treatment_var="treatment", anchor_level="A", **kwargs
    )


def test_default_ci_is_the_normal_delta_method():
    frame = _two_arm_frame(20)

    effects = _effects(frame)

    assert effects["analysis_summary"]["ci_method"].iloc[0] == "normal"

    row = effects["contrasts"].iloc[0]
    z = 1.959963984540054  # norm.ppf(0.975)

    np.testing.assert_allclose(
        row["OR_ci_low"], np.exp(row["log_or"] - z * row["se_log_or"]), rtol=1e-9
    )
    np.testing.assert_allclose(
        row["OR_ci_high"], np.exp(row["log_or"] + z * row["se_log_or"]), rtol=1e-9
    )
    np.testing.assert_allclose(
        row["RD_ci_low"], row["RD"] - z * row["se_RD"], rtol=1e-9, atol=1e-12
    )


def test_t_ci_is_wider_than_normal_with_few_sets():
    """t on g-1 degrees of freedom exceeds the normal critical value."""
    frame = _two_arm_frame(20)

    normal = _effects(frame, ci_method="normal")
    t_based = _effects(frame, ci_method="t")

    n_row = normal["contrasts"].iloc[0]
    t_row = t_based["contrasts"].iloc[0]

    # Point estimates and standard errors are untouched by the method.
    assert t_row["OR"] == n_row["OR"]
    assert t_row["se_log_or"] == n_row["se_log_or"]

    # The wider critical value widens the interval on both sides.
    assert t_row["OR_ci_low"] < n_row["OR_ci_low"]
    assert t_row["OR_ci_high"] > n_row["OR_ci_high"]


def test_t_falls_back_to_normal_with_a_single_set():
    frame = _two_arm_frame(1)

    with pytest.warns(RuntimeWarning, match="no degrees of freedom"):
        t_based = _effects(frame, ci_method="t")

    normal = _effects(frame, ci_method="normal")

    pd.testing.assert_frame_equal(t_based["contrasts"], normal["contrasts"])
    assert np.isfinite(t_based["contrasts"][["OR_ci_low", "OR_ci_high"]]).all().all()


def test_wild_bootstrap_is_reproducible():
    frame = _two_arm_frame(20)

    first = _effects(frame, ci_method="wild_bootstrap", random_state=42)
    second = _effects(frame, ci_method="wild_bootstrap", random_state=42)

    pd.testing.assert_frame_equal(first["contrasts"], second["contrasts"])
    assert first["analysis_summary"]["ci_method"].iloc[0] == "wild_bootstrap"


def test_wild_bootstrap_ci_is_finite_and_ordered():
    frame = _two_arm_frame(20)

    effects = _effects(frame, ci_method="wild_bootstrap", random_state=42)
    row = effects["contrasts"].iloc[0]

    for low, high in (
        ("OR_ci_low", "OR_ci_high"),
        ("RR_ci_low", "RR_ci_high"),
        ("RD_ci_low", "RD_ci_high"),
    ):
        assert np.isfinite(row[low]) and np.isfinite(row[high])
        assert row[low] < row[high]

    risk = effects["group_risk"]
    assert np.isfinite(risk[["risk_ci_low", "risk_ci_high"]]).all().all()
    assert (risk["risk_ci_low"] < risk["risk_ci_high"]).all()


def test_wild_bootstrap_tracks_the_delta_method_on_sample_data(four_group):
    """With enough sets the percentile interval approximates the delta method."""
    data, covariates, anchor, fit = four_group

    search = samatch.gps_candidate_search(
        data, fit["gps"], treatment_var="treatment", anchor_level=anchor, top_m=10, gps_space="logit"
    )
    matched = samatch.sam_match(data, search, X_vars=covariates, treatment_var="treatment")
    matched_data = samatch.extract_matched_data(
        data, search, matched, treatment_var="treatment", anchor_level=anchor
    )

    normal = samatch.sam_estimate_effects(
        matched_data, outcome_var="mortality_28d", treatment_var="treatment"
    )
    boot = samatch.sam_estimate_effects(
        matched_data,
        outcome_var="mortality_28d",
        treatment_var="treatment",
        ci_method="wild_bootstrap",
        n_bootstrap=999,
        random_state=7,
    )

    for column in ("OR_ci_low", "OR_ci_high"):
        np.testing.assert_allclose(
            boot["contrasts"][column], normal["contrasts"][column], rtol=0.35, atol=0.1
        )


@pytest.mark.parametrize("ci_method", ["bayes", "NORMAL", "", None])
def test_invalid_ci_method_raises(ci_method):
    frame = _two_arm_frame(4)

    with pytest.raises(ValueError, match="ci_method"):
        _effects(frame, ci_method=ci_method)


@pytest.mark.parametrize("n_bootstrap", [0, -3, 2.5, "999"])
def test_invalid_n_bootstrap_raises(n_bootstrap):
    frame = _two_arm_frame(4)

    with pytest.raises(ValueError, match="n_bootstrap"):
        _effects(frame, ci_method="wild_bootstrap", n_bootstrap=n_bootstrap)


# Boundary-safe transforms -----------------------------------------------------


def test_logit_clips_only_when_asked():
    with np.errstate(divide="ignore"):
        assert np.isinf(samatch.logit([0.0, 1.0])).all()

    clipped = samatch.logit([0.0, 1.0], eps=1e-6)
    assert np.isfinite(clipped).all()


def test_expit_does_not_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = samatch.expit([-2000.0, 0.0, 2000.0])

    np.testing.assert_allclose(values, [0.0, 0.5, 1.0])
