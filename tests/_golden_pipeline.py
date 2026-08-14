"""
The exact pipeline runs whose output is pinned by the golden fixtures.

Kept separate from both the test module and the fixture generator so that the
two always execute *identical* code paths.
"""

import samatch


def _covariates(data):
    return [
        column
        for column in data.columns
        if column not in ("synthetic_id", "treatment", "mortality_28d")
    ]


def run_four_group():
    """Full four-group SAM pipeline, as documented in the README."""
    data = samatch.load_sample_4group()
    covariates = _covariates(data)
    anchor = data["treatment"].astype(str).unique()[0]

    fit = samatch.estimate_gps_multinom(
        data,
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
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
        data,
        search,
        X_vars=covariates,
        treatment_var="treatment",
    )
    report = samatch.sam_evaluate(
        data,
        search,
        matched,
        fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
    )
    matched_data = samatch.extract_matched_data(
        data,
        search,
        matched,
        treatment_var="treatment",
        anchor_level=anchor,
    )
    effects = samatch.sam_estimate_effects(
        matched_data,
        outcome_var="mortality_28d",
        treatment_var="treatment",
        anchor_level=anchor,
    )
    weighting = samatch.evaluate_comparator_weighting(
        data,
        method="overlap",
        gps=fit["gps"],
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )

    return {
        "gps": fit["gps"],
        "matched": matched["matched"],
        "smd": report["smd_balance"]["by_covariate"],
        "auc": report["treatment_discrimination_auc"]["pairwise"],
        "group_risk": effects["group_risk"],
        "contrasts": effects["contrasts"],
        "weighted_balance": weighting["balance"]["by_covariate"],
        "ess": weighting["ess"],
    }


def run_three_group():
    """Three-group ``match_3way`` pipeline, as documented in the README."""
    data = samatch.load_sample_3group()
    covariates = _covariates(data)
    anchor = data["treatment"].astype(str).unique()[0]

    fit = samatch.estimate_gps_multinom(
        data,
        X_vars=covariates,
        treatment_var="treatment",
        anchor_level=anchor,
    )
    search = samatch.gps_candidate_search(
        data,
        fit["gps"],
        treatment_var="treatment",
        anchor_level=anchor,
        top_m=10,
        gps_space="logit",
    )
    matched = samatch.match_3way(
        data,
        search,
        fit["gps"],
        treatment_var="treatment",
        caliper="auto",
        gps_space="raw",
        top_n=10,
    )

    return {
        "gps": fit["gps"],
        "matched": matched["matched"],
    }
