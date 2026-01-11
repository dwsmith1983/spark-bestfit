"""Results handling for fitted distributions.

This module provides backward-compatible access to distribution fitting results.
The implementation has been split into two modules for better organization:

- ``storage``: Data classes for individual results (DistributionFitResult, LazyMetricsContext)
- ``collection``: Collection classes for managing multiple results (BaseFitResults, etc.)

All public APIs are re-exported here for backward compatibility.

Classes:
    DistributionFitResult: Stores a single distribution's fitted parameters and metrics.
    LazyMetricsContext: Context for deferred KS/AD metric computation.
    BaseFitResults: Abstract base class for fit result collections.
    EagerFitResults: Results with all metrics pre-computed.
    LazyFitResults: Results with lazy KS/AD metric computation.

Functions:
    create_fit_results: Factory function for creating FitResults.

Type Aliases:
    MetricName: Valid metric names for sorting/filtering.
    ContinuousHistogram: Tuple type for continuous distribution histograms.
    DiscreteHistogram: Tuple type for discrete distribution histograms.
    HistogramBins: Array of bin edges (len = n_bins + 1).
    HistogramCounts: Array of counts/density per bin.
    HistogramResult: Tuple type for HistogramComputer results (counts, bins).
    FitResultsType: Union of EagerFitResults and LazyFitResults.

Constants:
    FITTING_SAMPLE_SIZE: Default sample size for fitting (10000).
    DEFAULT_PVALUE_THRESHOLD: Default p-value threshold (0.05).
    DEFAULT_KS_THRESHOLD: Default KS statistic threshold (0.10).
    DEFAULT_AD_THRESHOLD: Default AD statistic threshold (2.0).
    DEFAULT_BINS: Default number of histogram bins (50).
    DEFAULT_BOOTSTRAP_SAMPLES: Default bootstrap iterations (1000).
    DEFAULT_MAX_SAMPLES: Default maximum samples to collect (10000).
    DEFAULT_DPI: Default plot DPI (100).
    DEFAULT_SAMPLE_SIZE: Default sample size for sampling methods (1000).
"""

# Re-export everything from collection module
from spark_bestfit.collection import (  # Classes; Factory function; Type alias; Backward-compatible alias
    BaseFitResults,
    EagerFitResults,
    FitResults,
    FitResultsType,
    LazyFitResults,
    create_fit_results,
)

# Re-export everything from storage module
from spark_bestfit.storage import (  # Constants; Type aliases; Data classes
    DEFAULT_AD_THRESHOLD,
    DEFAULT_BINS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_DPI,
    DEFAULT_KS_THRESHOLD,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_PVALUE_THRESHOLD,
    DEFAULT_SAMPLE_SIZE,
    FITTING_SAMPLE_SIZE,
    ContinuousHistogram,
    DiscreteHistogram,
    DistributionFitResult,
    HistogramBins,
    HistogramCounts,
    HistogramResult,
    LazyMetricsContext,
    MetricName,
)

# Re-export TruncatedFrozenDist for backward compatibility (was imported in original results.py)
from spark_bestfit.truncated import TruncatedFrozenDist

# Define __all__ for explicit public API
__all__ = [
    # Constants
    "FITTING_SAMPLE_SIZE",
    "DEFAULT_PVALUE_THRESHOLD",
    "DEFAULT_KS_THRESHOLD",
    "DEFAULT_AD_THRESHOLD",
    "DEFAULT_BINS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_DPI",
    "DEFAULT_SAMPLE_SIZE",
    # Type aliases
    "MetricName",
    "ContinuousHistogram",
    "DiscreteHistogram",
    "HistogramBins",
    "HistogramCounts",
    "HistogramResult",
    "FitResultsType",
    # Data classes
    "DistributionFitResult",
    "LazyMetricsContext",
    # Collection classes
    "BaseFitResults",
    "EagerFitResults",
    "LazyFitResults",
    # Factory function and alias
    "create_fit_results",
    "FitResults",
    # TruncatedFrozenDist for backward compatibility
    "TruncatedFrozenDist",
]
