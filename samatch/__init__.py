"""
SAM (Shared Anchor Matching): A Scalable Matching Framework for
Multiple Treatment Groups.

SAM provides tools for simultaneous matching of multiple treatment groups
to a shared anchor group. The framework combines generalized propensity
score estimation, GPS-guided candidate screening, and
Mahalanobis-distance-based matching.

The package also provides matching diagnostics, matched-cohort outcome
analysis, three-way propensity score matching, and multi-arm weighting.
"""

from ._candidate_search import gps_candidate_search
from ._diagnostics import (
    compute_pairwise_treatment_auc,
    compute_smd_balance,
)
from ._evaluate import (
    extract_matched_data,
    sam_estimate_effects,
    sam_evaluate,
)
from ._gps import estimate_gps_multinom
from ._mahalanobis import (
    build_group_distance_matrices,
    get_pooled_covariance,
    mahalanobis_distance_matrix,
)
from ._match import sam_match
from ._match_3way import calc_caliper_3way, match_3way
from ._utils_math import auc_mannwhitney, expit, logit
from ._weighting import (
    compute_balancing_weights,
    compute_effective_sample_size,
    compute_weighted_balance,
    evaluate_comparator_weighting,
)
from .datasets import load_sample_3group, load_sample_4group


__all__ = [
    # Generalized propensity scores and matching
    "estimate_gps_multinom",
    "gps_candidate_search",
    "sam_match",
    "sam_evaluate",
    "extract_matched_data",
    "sam_estimate_effects",
    # Matching diagnostics
    "compute_smd_balance",
    "compute_pairwise_treatment_auc",
    # Mahalanobis utilities
    "get_pooled_covariance",
    "mahalanobis_distance_matrix",
    "build_group_distance_matrices",
    # Three-way matching
    "calc_caliper_3way",
    "match_3way",
    # Multi-arm weighting
    "compute_balancing_weights",
    "compute_weighted_balance",
    "compute_effective_sample_size",
    "evaluate_comparator_weighting",
    # Mathematical utilities
    "expit",
    "logit",
    "auc_mannwhitney",
    # Example datasets
    "load_sample_3group",
    "load_sample_4group",
]