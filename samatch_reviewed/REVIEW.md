# SAM-Python Review Report

This directory contains a reviewed and improved copy of the `samatch` package
(version 0.2.0), prepared in support of the methodological paper. The original
package (v0.1.2) is untouched in the repository root; everything here is a
drop-in replacement that can be diffed against it.

## 1. Scope and method

- All 13 source modules (~4,300 lines) were read in full and reviewed for
  statistical correctness, edge-case behavior, API consistency, and
  performance.
- The full test suite (211 tests) was run on the original code: **all pass**.
- The package was exercised end-to-end beyond the test suite: every README
  pipeline, plus edge cases (numeric treatment codes, K=5 groups, `top_m=1`,
  zero-match results, complete separation in the outcome model, `trim=0`
  weighting with near-zero GPS, re-sorted data rejection, three-way matching
  with the search base different from the anchor).
- A performance check at n = 20,000 subjects: GPS fit 0.03 s, candidate
  search 0.03 s, `sam_match` 0.31 s, evaluation + cohort extraction + effects
  0.04 s total. The KD-tree and heap implementations scale as intended.
- A literature review grounded the algorithmic choices (Section 3).

## 2. What was verified to work as intended

| Area | Verdict |
|---|---|
| `estimate_gps_multinom` | Unregularized multinomial logit across scikit-learn versions; standardization is folded back into the coefficients so `predict_proba` works on raw covariates. Convergence and regularized-fallback warnings reach the caller. |
| `gps_candidate_search` | KD-tree nearest candidates with deterministic (distance, row) tie-breaking; ties straddling the cutoff are re-resolved by radius query. Fingerprinting rejects modified data/GPS at every later stage (verified: re-sorted data raises with a specific message). |
| `sam_match` | Heap-greedy global selection on total Mahalanobis loss with a claimant reverse index; no quadratic rescans. On the bundled data it reaches `max_possible_rate` exactly, as documented. |
| `match_3way` | Matches the published three-way method (Section 3): two-dimensional GPS space, triangle-perimeter ranking, perimeter caliper, global greedy without replacement. Works with the search base different from the anchor; zero-match results keep a usable schema downstream. |
| `sam_evaluate` / diagnostics | SMD with pooled SD and an explicit `smd_defined` flag; empty matches report zero rows rather than NaNs. |
| `extract_matched_data` / `sam_estimate_effects` | One-subject-per-group validation via integer tabulation; incomplete sets removed wholesale; complete separation is warned about and flagged (`separation` column) with finite estimates. |
| Weighting | IPTW (stabilized), overlap, and matching weights match the published multi-arm formulas; trimming is reported via `n_trimmed`; non-finite weights are rejected by the balance check. |
| Packaging | Golden fixtures pin the full pipeline output; on the reference stack they compare at rtol 1e-9. |

## 3. Literature review

**Three-way matching (`match_3way`).** The method is that of Rassen, Shelat,
Franklin, Glynn, Solomon & Schneeweiss, *Matching by propensity score in cohort
studies with three treatment groups*, Epidemiology 24(3):401-9 (2013):
generalized propensity scores from multinomial logistic regression, 1:1:1
matched sets without replacement, triplets ranked by the perimeter of the
triangle in the two-dimensional space of the (K-1 = 2) non-redundant GPS
coordinates. An independent neutral comparison study (Austin & Austin, *BMC
Medical Research Methodology* 26:111, 2026) describes the algorithm in the
same terms. The implementation matches that description, including the
KD-tree candidate search (cf. Hott, Brunelle, Myers, Rassen & Shelat,
*KD-Tree Algorithm for Propensity Score Matching With Three or More Treatment
Groups*, technical report, 2012).

Two items the authors should confirm against the primary source (the 2013
paper is paywalled; it could not be retrieved from open sources):

1. **The `0.6` constant in `calc_caliper_3way`.** The automatic caliper is
   `0.6 * sqrt(mean over groups of the mean within-group coordinate variance)`,
   i.e. a perimeter-scale analog of the classic `0.2 x SD(logit PS)` caliper
   (Austin 2011; Wang et al., *PLOS ONE* 8(6):e64731, 2013 for three-group
   matching). The constant and the exact variance-averaging should be checked
   against Rassen 2013 before publication. The formula is data-driven and
   documented; only the constant is unverified.
2. **The `>= 2 subjects per group` requirement** of the automatic caliper is a
   consequence of `var(ddof=1)` and is documented; users with a singleton
   group must supply a manual caliper. This is defensible but worth stating in
   the paper's methods section.

**Shared Anchor Matching (`sam_match`).** No prior use of the term "shared
anchor matching" was found in the literature; the closest relatives are Rassen
2013's three-way matching (three groups, PS space only) and the
common-referent approach. The SAM contribution — a *shared* anchor group
matched simultaneously to one subject from each of K-1 comparator groups in
Mahalanobis (covariate) space, with GPS-guided screening and global greedy
selection on total loss — appears novel for K > 3, which is the paper's
differentiator. The neutral comparison study (above) notes that three-way
matching "targets a population that is difficult to formally define"; the same
caution applies to SAM and should be addressed in the paper's estimand
discussion (the matched cohort is a selected sample; `matching_rate` vs
`max_possible_rate` reporting helps).

**Weighting.** Overlap weights `h = 1 / sum_k (1/e_k)` and matching weights
`h = min_k e_k` match Li & Li's framework (matching/overlap as special cases;
cf. Yoshida et al., *Epidemiology* 28(3):387-95, 2017) and Aylward et al.
(2019). IPTW stabilization by empirical prevalence is standard.

**Diagnostics.** Pooled-SD SMD follows Austin (2009); the cluster-robust
variance uses the CR1 finite-sample correction `g/(g-1) * (n-1)/(n-p)`
(Keele & Kenney, 2004); ESS is Kish's `(sum w)^2 / sum w^2`. All consistent
with the literature.

## 4. Changes in this copy (0.1.2 -> 0.2.0)

Two real defects were found and fixed, both in `samatch/_diagnostics.py`:

1. **Silent NaN AUC under complete separation.** The log-ratio score
   `log(p1) - log(p2)` is NaN when a subject's GPS is exactly 0 in *both*
   members of a pair (possible with K >= 3 under complete separation, since an
   unregularized model can return exact 0/1). scipy's `rankdata` then returns
   all-NaN ranks and the AUC silently becomes NaN — no error, no warning. The
   scores are now clipped to `[1e-6, 1 - 1e-6]` (the same boundary
   `transform_ps()` uses); a zero-scored subject ranks at the extreme, which
   is where maximal discrimination belongs.

2. **`mean_auc` cancelled opposite-direction discrimination.** A pair's AUC is
   directional: 0.6 for (A, B) and 0.4 for the same two groups scored the
   other way describe identical discrimination. Averaging raw AUCs lets
   opposite-direction pairs cancel toward 0.5, understating residual
   separation (observed on real data: bundled four-group mean moved 0.5897 ->
   0.5965 because one pair sits below 0.5). `mean_auc` now folds each value to
   `max(auc, 1 - auc)`. The directional values remain in the `pairwise` table.
   Note for the paper: with a well-fitted GPS every pair's AUC is structurally
   >= 0.5 (group_1 members carry their own probability high); sub-0.5 values
   arise from noise around chance, and the fold is convex, so `mean_auc` has a
   small upward bias under pure noise. If the paper prefers an unbiased
    summary, `mean |auc - 0.5|` is the alternative; both are one line away.

3. **Small-sample confidence intervals in `sam_estimate_effects`.** The
   normal-approximation interval (estimate ± 1.96 × cluster-robust SE) is
   anti-conservative when there are few matched sets. `sam_estimate_effects`
   now takes a `ci_method` argument:

   - `"normal"` (default, unchanged): the previous behavior.
   - `"t"`: same delta-method interval with a t critical value on g − 1
     degrees of freedom (g = number of matched sets) — the standard
     small-sample correction for cluster-robust inference (Cameron and Miller,
     *Journal of Economic Perspectives* 29(1):101-119, 2015). Falls back to
     `"normal"` with a warning when g < 2.
   - `"wild_bootstrap"`: percentile interval from a linear wild cluster
     bootstrap (Rademacher weights, `n_bootstrap` replicates, reproducible via
     `random_state`). Each replicate sign-flips the per-cluster score
     contributions and maps them through the sandwich bread, so no refitting —
     important because resampling clusters with replacement would frequently
     drop an entire arm and leave the saturated treatment model unidentified.

   Point estimates, standard errors, and all other output are identical across
   methods; only the interval bounds change. `analysis_summary` records the
   method used. The default stays `"normal"` so previously published numbers
   are reproducible; the paper should state which method it uses.

Supporting changes:

- `tests/test_auc_diagnostics.py` (new): four regression tests pinning both
  fixes, including the backward-compatibility case where all pairs point the
  same way.
- `tests/test_packaging.py`: the version test now checks the in-code fallback
  constant when running from a source checkout. The old form compared
  `samatch.__version__` against pyproject, which fails in any environment with
  a *stale* samatch installed elsewhere (importlib.metadata resolves to that
  installation, not the checkout). This also catches the stale fallback in the
  original repo (it said 0.1.1 while pyproject said 0.1.2).
- `samatch/__init__.py`: fallback version aligned to 0.2.0.
- `pyproject.toml` / `CITATION.cff`: version 0.2.0 (bump for the changed
  `mean_auc` semantics; adjust at release time).
- `README.md`: one sentence documenting the direction-free `mean_auc`, plus a
  note on the `ci_method` option.
- `tests/test_weighting_and_effects.py`: twelve new tests for the CI methods —
  default equals the normal delta method, t is strictly wider with few sets,
  single-set fallback warns and matches `"normal"`, wild bootstrap is
  reproducible / finite / ordered, tracks the delta method on the sample data,
  and invalid inputs raise.
- `benchmarks/` (new): timing harness (`bench.py`, `compare_bench.py`) and a
  README with the scaling findings.

**Behavior preservation.** On the reference stack (numpy 2.4.2, pandas 3.0.1,
scipy 1.17.1, scikit-learn 1.8.0 — the one that generated the golden
fixtures), all strict (rtol 1e-9) golden comparisons pass unchanged: GPS,
matched sets, SMDs, the pairwise AUC table, effects, and weighting outputs are
bit-identical to 0.1.2. The only user-visible numeric change is `mean_auc`
itself, which is not pinned in the fixtures.

## 4b. Second pass: cross-implementation reproducibility and speed

**Finding.** With the same GPS as input, R and Python agreed on the bundled
data but *not* on data with discrete covariates (2,000 subjects, 216
covariate profiles): different matched sets and a different total loss, for
both `sam_match` and `match_3way`. Two causes:

1. Mathematically equal distances (common when covariates are discrete --
   two subjects can differ from an anchor by the same amount in different
   ways) come out of floating-point arithmetic differing in the last bits.
   The order then depended on summation order (numpy vs R), not on the
   documented lowest-index tie-break; the greedy selection diverged from
   that step on. The existing R test suite already noted that "1e-14 is
   enough to flip an exact tie".
2. `match_3way` ties followed the KD-tree traversal order, which differed
   between the builds (Python rebuilt its tree, R did not).

**Fix (both languages, identical rule).**

- `order_key()` (`_match_common.py`): every ranking comparison uses
  `rint(value * 1e10)`, then the explicit tie-break. Raw values are still
  reported.
- `match_3way`: seed partner = nearest available subject, lowest row among
  equidistant; trios ranked by (perimeter key, o1 row, o2 row); all distances
  as `sqrt(dx*dx + dy*dy)` and perimeters as `(d_r1 + d_r2) + d_12`, which
  is bit-identical in R and numpy.
- `sam_match`: Mahalanobis distances from differences of whitened, centred
  rows instead of the expanded form `|a|^2 + |b|^2 - 2ab`. The expanded form
  lost up to 6e-4 in the loss when 1e6 was added to one covariate; now < 1e-8.

After the fix R and Python produce identical matched sets on every dataset
tried (bundled 3/4-group data, raw and logit, and the tie-heavy datasets at
n = 2,000 and 20,000). `tests/golden/ties_*` pins this; the same files are in
the R suite, and the pre-fix code of both languages fails them.

**Performance.** `match_3way` now uses scipy's `cKDTree` (compiled; the
initial pass is one batched, multi-threaded query with `workers=-1`) with
exact filtering in Python, and bounds the search radius by
`min(seed, caliper) / 2`, which is exact (each side of a triangle is at most
half its perimeter). 5x faster at n = 50,000 (10.5 s -> 2.1 s). On
continuous data every matched set is identical to before; perimeters move by
at most ~1e-11 (the old expanded-form pair distances carried that error).

`tests/test_canonical_ties.py` (new, 8 tests): shared fixture, `order_key`,
covariate-offset invariance, nearest/within tie and consumption behaviour.

## 5. Items for the authors (not changed here)

- **Confirm the `0.6` caliper constant** against Rassen 2013 (Section 3).
- **State the CI method in the paper.** The default remains the normal
  approximation for backward compatibility, but `ci_method="t"` (or
  `"wild_bootstrap"`) is now available and more defensible when the number of
  matched sets is small. If the paper's applications have few matched sets,
  use `"t"` and say so in the methods section; report `n_matched_sets` from
  `analysis_summary` alongside it.
- **`__version__` quirk:** when a different samatch is installed in the
  environment, running from a source checkout reports the *installed*
  distribution's version (importlib.metadata), not the checkout's. Standard
  behavior, but worth knowing when debugging; the improved packaging test
  handles it.
- **Estimand discussion** for both matching methods (Section 3): the matched
  cohort is a selected sample; state what population it represents.

## 6. Verification summary

- Original code: 211/211 tests pass; all README pipelines and edge cases
  behave as documented (Section 2).
- Reviewed copy: **229/229 tests pass** (211 + 4 AUC regression tests +
  12 CI-method tests), including strict golden comparisons on the reference
  stack.
- Every README example re-run end-to-end against this copy: matching rates,
  diagnostics, effects, and weighting all reproduce; the two fixed behaviors
  verified through the public API.
- Benchmarks (`benchmarks/`): original and reviewed builds are performance-
  equivalent (every stage within ±4% run-to-run noise); matching rates
  identical. `sam_match` scales near-linearly (0.09 s / 0.40 s / 1.13 s at
  n = 5k/20k/50k); `match_3way` is superlinear (2.2 s / 11.1 s at
  n = 20k/50k) and is the bottleneck for large three-group studies.
