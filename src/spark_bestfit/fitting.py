"""Distribution fitting using Pandas UDFs for efficient parallel processing.

This module provides the public API for distribution fitting. Implementation
is split across submodules:

- spark_bestfit.estimation: Parameter estimation (MLE, MSE, bootstrap)
- spark_bestfit.metrics: Goodness-of-fit metrics (KS, AD, AIC, BIC)

All symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

# Re-export everything from estimation module
from spark_bestfit.estimation import (
    FIT_RESULT_SCHEMA,
    FITTING_SAMPLE_SIZE,
    HEAVY_TAIL_DISTRIBUTIONS,
    EstimationMethod,
    _failed_fit_result,
    _filter_bootstrap_outliers,
    bootstrap_confidence_intervals,
    compute_data_stats,
    compute_pdf_range,
    create_fitting_udf,
    create_sample_data,
    detect_heavy_tail,
    evaluate_pdf,
    extract_distribution_params,
    fit_mse,
    fit_single_distribution,
    get_continuous_param_names,
)

# Re-export everything from metrics module
from spark_bestfit.metrics import (
    AD_PVALUE_DISTRIBUTIONS,
    compute_ad_pvalue,
    compute_ad_statistic,
    compute_ad_statistic_frozen,
    compute_information_criteria,
    compute_information_criteria_frozen,
    compute_ks_ad_metrics,
    compute_ks_statistic,
    compute_ks_statistic_frozen,
)

# Re-export truncated distribution helper for backward compatibility
from spark_bestfit.truncated import TruncatedFrozenDist

# Private alias for backward compatibility (used internally)
_create_truncated_dist = __import__("spark_bestfit.truncated", fromlist=["create_truncated_dist"]).create_truncated_dist

__all__ = [
    # Constants
    "FITTING_SAMPLE_SIZE",
    "FIT_RESULT_SCHEMA",
    "HEAVY_TAIL_DISTRIBUTIONS",
    "AD_PVALUE_DISTRIBUTIONS",
    "EstimationMethod",
    # Data analysis
    "compute_data_stats",
    "detect_heavy_tail",
    # Parameter estimation
    "fit_mse",
    "fit_single_distribution",
    "create_fitting_udf",
    "bootstrap_confidence_intervals",
    "create_sample_data",
    # Metrics
    "compute_information_criteria",
    "compute_information_criteria_frozen",
    "compute_ks_statistic",
    "compute_ks_statistic_frozen",
    "compute_ad_statistic",
    "compute_ad_statistic_frozen",
    "compute_ad_pvalue",
    "compute_ks_ad_metrics",
    # Helpers
    "_failed_fit_result",
    "_filter_bootstrap_outliers",
    "_create_truncated_dist",
    "evaluate_pdf",
    "get_continuous_param_names",
    "extract_distribution_params",
    "compute_pdf_range",
    # Classes
    "TruncatedFrozenDist",
]
