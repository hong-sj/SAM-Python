# SAM-Python

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21926925.svg)](https://doi.org/10.5281/zenodo.21926925)
[![CI](https://github.com/hong-sj/SAM-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/hong-sj/SAM-Python/actions/workflows/ci.yml)

## Shared Anchor Matching: A Scalable Matching Framework for Multiple Treatment Groups

**SAM** is a Python package for matching observational data with multiple treatment groups using a shared anchor group.

The framework combines generalized propensity score (GPS) estimation, GPS-guided candidate screening, and Mahalanobis-distance-based matching. Each subject in the anchor group is matched simultaneously to one subject from each comparator group, producing matched sets across multiple treatment groups.

The package also provides tools for covariate-balance assessment, treatment-discrimination diagnostics, matched-cohort outcome analysis, three-way propensity score matching, and multi-arm weighting.

---

## Installation

Install the latest release from PyPI:

```bash
pip install samatch
```

Install the package from GitHub:

```bash
pip install git+https://github.com/hong-sj/SAM-Python.git
```

Then import the package:

```python
import samatch
```

---

## Quick Start

Import SAMatch and load the included four-treatment-group example dataset:

```python
import samatch

data = samatch.load_sample_4group()

print(data.head())
print(data["treatment"].value_counts())
```

Define the covariates and anchor treatment:

```python
covariates = [
    column
    for column in data.columns
    if column not in ("synthetic_id", "treatment", "mortality_28d")
]

anchor = data["treatment"].astype(str).unique()[0]
```

### 1. Estimate generalized propensity scores

Estimate the GPS using multinomial logistic regression:

```python
fit = samatch.estimate_gps_multinom(
    data,
    X_vars=covariates,
    treatment_var="treatment",
    anchor_level=anchor,
)

print(fit["gps"].head())
```

`fit["gps"]` contains the estimated treatment-assignment probabilities for each subject.

### 2. Search for candidate matches

For each anchor subject, identify candidate subjects from each comparator group in GPS space:

```python
search = samatch.gps_candidate_search(
    data,
    fit["gps"],
    treatment_var="treatment",
    anchor_level=anchor,
    top_m=10,
    gps_space="logit",
)
```

By default, SAM retains the nearest candidate subjects from each comparator group for subsequent matching.

### 3. Perform Shared Anchor Matching

Match each anchor subject simultaneously to one subject from every comparator group:

```python
matched = samatch.sam_match(
    data,
    search,
    X_vars=covariates,
    treatment_var="treatment",
)

print(matched["matching_rate"])
print(matched["max_possible_rate"])
print(matched["matched"].head())
```

The resulting object contains the matched sets, group-specific Mahalanobis distances, total matching loss, unmatched anchor subjects, and overall matching rate.

Every matched set consumes one subject from each comparator group, so the matching rate is bounded above by the size of the smallest comparator group divided by the number of anchor subjects. On the bundled four-group dataset that ceiling is 59/448 = 0.132, which is also the rate SAM achieves — every possible set was formed. Read `matching_rate` against `max_possible_rate` rather than against 1.0.

### 4. Evaluate matching quality

Evaluate covariate balance and matching diagnostics:

```python
report = samatch.sam_evaluate(
    data,
    search,
    matched,
    fit["gps"],
    X_vars=covariates,
    treatment_var="treatment",
)

print(report["matching_rate"])
print(report["loss_distribution"])
print(report["smd_balance"]["summary"])
print(report["treatment_discrimination_auc"])
```

The diagnostic output includes matching loss, covariate balance based on standardized mean differences (SMDs), and pairwise treatment-discrimination AUCs.

---

## Matched Cohort and Outcome Analysis

The matched observations can be extracted into a subject-level dataset:

```python
matched_data = samatch.extract_matched_data(
    data,
    search,
    matched,
    treatment_var="treatment",
    anchor_level=anchor,
)

print(matched_data.head())
```

Treatment effects can then be estimated in the matched cohort using:

```python
effects = samatch.sam_estimate_effects(
    matched_data,
    outcome_var="mortality_28d",
    treatment_var="treatment",
    anchor_level=anchor,
)

print(effects)
```

---

## Three-Group Matching

The package also provides `match_3way()` for three-treatment-group propensity score matching.

The included `sample_3group` dataset can be used as an example:

```python
from samatch import load_sample_3group

data3 = load_sample_3group()

covariates3 = [
    column
    for column in data3.columns
    if column not in ("synthetic_id", "treatment", "mortality_28d")
]

anchor3 = data3["treatment"].astype(str).unique()[0]

fit3 = samatch.estimate_gps_multinom(
    data3,
    X_vars=covariates3,
    treatment_var="treatment",
    anchor_level=anchor3,
)

search3 = samatch.gps_candidate_search(
    data3,
    fit3["gps"],
    treatment_var="treatment",
    anchor_level=anchor3,
    top_m=10,
    gps_space="logit",
)

matched3 = samatch.match_3way(
    data3,
    search3,
    fit3["gps"],
    treatment_var="treatment",
    caliper="auto",
    gps_space="raw",
    top_n=10,
)

print(matched3["matching_rate"])
print(matched3["matched"].head())
```

`match_3way()` requires exactly three treatment groups. It matches in propensity score space only, so it takes no `X_vars`; covariates enter through `estimate_gps_multinom()`.

---

## Multi-Arm Weighting

SAM also includes functions for multi-arm propensity score weighting:

- `compute_balancing_weights()`
- `compute_weighted_balance()`
- `compute_effective_sample_size()`
- `evaluate_comparator_weighting()`

These functions provide an alternative weighting-based approach for evaluating covariate balance across multiple treatment groups.

Example:

```python
weighting = samatch.evaluate_comparator_weighting(
    data,
    method="overlap",
    gps=fit["gps"],
    X_vars=covariates,
    treatment_var="treatment",
    anchor_level=anchor,
)

print(weighting["balance"]["summary"])
print(weighting["ess"])
print(weighting["n_trimmed"])
```

Balancing weights divide by each subject's own propensity score, so a score near zero yields a weight large enough to dominate every summary that follows. Scores are therefore bounded below by `trim` (default `1e-3`, set `trim=0` to disable), and `n_trimmed` reports how many subjects were affected. A nonzero count is a positivity problem worth investigating rather than a routine detail.

---

## Main Functions

| Function | Description |
|---|---|
| `estimate_gps_multinom()` | Estimate generalized propensity scores using multinomial logistic regression |
| `gps_candidate_search()` | Identify candidate matches in GPS space |
| `sam_match()` | Perform Shared Anchor Matching |
| `sam_evaluate()` | Evaluate matching quality and covariate balance |
| `extract_matched_data()` | Extract the matched subject-level cohort |
| `sam_estimate_effects()` | Estimate treatment effects in the matched cohort |
| `match_3way()` | Perform three-group propensity score matching |
| `compute_balancing_weights()` | Compute multi-arm balancing weights |
| `compute_weighted_balance()` | Assess weighted covariate balance |
| `compute_effective_sample_size()` | Calculate effective sample size after weighting |
| `evaluate_comparator_weighting()` | Evaluate weighting, balance, and effective sample size |

The individual diagnostics used by `sam_evaluate()` are also public, and can be called directly on a matched-set frame:

| Function | Description |
|---|---|
| `compute_smd_balance()` | Standardized mean differences per covariate and comparator group |
| `compute_pairwise_treatment_auc()` | Pairwise treatment-discrimination AUC |
| `get_pooled_covariance()` | Pooled within-group covariance and precision matrix |
| `mahalanobis_distance_matrix()` | Pairwise Mahalanobis distances |
| `build_group_distance_matrices()` | Anchor-to-comparator distance matrix per group |

---

## Requirements on the Input Data

- **Covariates must be numeric and complete.** `X_vars` columns are used directly in linear algebra, so categorical covariates need encoding (for example with `pandas.get_dummies`) and missing values need dropping or imputing first. SAM does not impute.
- **Pass the same DataFrame through the whole pipeline.** Matched sets reference positional row indices, so `data` must reach `sam_match()`, `sam_evaluate()` and `extract_matched_data()` in the same row order that `gps_candidate_search()` saw. This is checked, and a modified frame raises rather than producing a report about the wrong subjects. Adding a column between stages is fine.
- **Treatment levels are compared as strings.** A numeric treatment column works, and `anchor_level` may be given either as `0` or as `"0"`.

---

## Example Data

Two example datasets are included:

- `sample_4group`: four-treatment-group example data for Shared Anchor Matching
- `sample_3group`: three-treatment-group example data for three-way matching

They can be loaded using:

```python
from samatch import load_sample_3group, load_sample_4group

data3 = load_sample_3group()
data4 = load_sample_4group()
```

---

## Citation

If you use SAM-Python in your research, please cite:

> Hong S, Hong S, Lee KH, Cha N. **SAM-Python: Shared Anchor Matching**. Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.21926925

A formal citation for the associated methodological paper will be added upon publication.

---

## License

This package is distributed under the MIT License.
