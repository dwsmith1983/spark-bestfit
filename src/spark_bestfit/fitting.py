"""Distribution fitting using Pandas UDFs for efficient parallel processing."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import rv_continuous

from spark_bestfit.truncated import TruncatedFrozenDist

# PySpark is optional - only import if available
try:
    from pyspark import Broadcast
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

    _PYSPARK_AVAILABLE = True
except ImportError:
    Broadcast = None  # type: ignore[assignment]
    pandas_udf = None  # type: ignore[assignment]
    ArrayType = None  # type: ignore[assignment]
    FloatType = None  # type: ignore[assignment]
    StringType = None  # type: ignore[assignment]
    StructField = None  # type: ignore[assignment]
    StructType = None  # type: ignore[assignment]
    _PYSPARK_AVAILABLE = False

# Constant for fitting sample size
FITTING_SAMPLE_SIZE: int = 10_000  # Most scipy distributions fit well with 10k samples

# Distributions that support Anderson-Darling p-value computation via scipy
# Maps our distribution names to scipy.anderson's dist parameter
AD_PVALUE_DISTRIBUTIONS: Dict[str, str] = {
    "norm": "norm",
    "expon": "expon",
    "logistic": "logistic",
    "gumbel_r": "gumbel",
    "gumbel_l": "gumbel_l",
}

# Define output schema for Pandas UDF (only if PySpark is available)
# Note: Pandas infers all columns as nullable, so we match that here
if _PYSPARK_AVAILABLE:
    FIT_RESULT_SCHEMA = StructType(
        [
            StructField("column_name", StringType(), True),  # Column being fitted
            StructField("distribution", StringType(), True),
            StructField("parameters", ArrayType(FloatType()), True),
            StructField("sse", FloatType(), True),
            StructField("aic", FloatType(), True),
            StructField("bic", FloatType(), True),
            StructField("ks_statistic", FloatType(), True),
            StructField("pvalue", FloatType(), True),
            StructField("ad_statistic", FloatType(), True),
            StructField("ad_pvalue", FloatType(), True),
            # Flat data summary columns for provenance (v2.0: replaced MapType for ~20% perf)
            StructField("data_min", FloatType(), True),
            StructField("data_max", FloatType(), True),
            StructField("data_mean", FloatType(), True),
            StructField("data_stddev", FloatType(), True),
            StructField("data_count", FloatType(), True),
            # Bounds for truncated distribution fitting (v1.4.0)
            StructField("lower_bound", FloatType(), True),
            StructField("upper_bound", FloatType(), True),
        ]
    )
else:
    FIT_RESULT_SCHEMA = None  # type: ignore[assignment]


def compute_data_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics of the original data.

    These statistics provide lightweight provenance information for debugging
    and understanding the context of fitted distributions.

    Args:
        data: Data array used for fitting

    Returns:
        Dictionary with keys: data_min, data_max, data_mean, data_stddev, data_count
    """
    return {
        "data_min": float(np.min(data)),
        "data_max": float(np.max(data)),
        "data_mean": float(np.mean(data)),
        "data_stddev": float(np.std(data)),
        "data_count": float(len(data)),
    }


def create_fitting_udf(
    histogram_broadcast: Broadcast[Tuple[np.ndarray, np.ndarray]],
    data_sample_broadcast: Broadcast[np.ndarray],
    column_name: Optional[str] = None,
    data_stats: Optional[Dict[str, float]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    lazy_metrics: bool = False,
) -> Callable[[pd.Series], pd.DataFrame]:
    """Factory function to create Pandas UDF with broadcasted data.

    This is the KEY optimization: The histogram and data sample are
    broadcasted once to all executors, then the Pandas UDF processes
    batches of distributions efficiently using vectorized operations.

    Args:
        histogram_broadcast: Broadcast variable containing (y_hist, bin_edges)
        data_sample_broadcast: Broadcast variable containing data sample
        column_name: Name of the column being fitted (for result tracking)
        data_stats: Pre-computed summary statistics (data_min, data_max, etc.)
        lower_bound: Lower bound for truncated distribution fitting (v1.4.0)
        upper_bound: Upper bound for truncated distribution fitting (v1.4.0)
        lazy_metrics: If True, skip expensive KS/AD computation during fitting.
            These metrics will be computed on-demand when accessed via
            FitResults.best() or DistributionFitResult properties. (v1.5.0)

    Returns:
        Pandas UDF function for fitting distributions

    Example:
        >>> # In DistributionFitter:
        >>> hist_bc = spark.sparkContext.broadcast((y_hist, bin_edges))
        >>> data_bc = spark.sparkContext.broadcast(data_sample)
        >>> stats = compute_data_stats(data_sample)
        >>> fitting_udf = create_fitting_udf(hist_bc, data_bc, column_name="value", data_stats=stats)
        >>> results = df.select(fitting_udf(col('distribution_name')))
    """

    @pandas_udf(FIT_RESULT_SCHEMA)
    def fit_distributions_batch(distribution_names: pd.Series) -> pd.DataFrame:
        """Vectorized UDF to fit multiple distributions in a batch.

        This function processes a batch of distribution names, fitting each
        against the broadcasted histogram and data sample. Uses Apache Arrow
        for efficient data transfer.

        Args:
            distribution_names: Series of scipy distribution names to fit

        Returns:
            DataFrame with fit result columns including data_min, data_max, etc.
        """
        # Get broadcasted data (no serialization overhead!)
        y_hist, bin_edges = histogram_broadcast.value
        data_sample = data_sample_broadcast.value

        # Fit each distribution in the batch
        results = []
        for dist_name in distribution_names:
            try:
                result = fit_single_distribution(
                    dist_name=dist_name,
                    data_sample=data_sample,
                    bin_edges=bin_edges,
                    y_hist=y_hist,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )
            except Exception:
                # Safety net: catch any unexpected exceptions to prevent job failure
                result = _failed_fit_result(dist_name, column_name, data_stats, lower_bound, upper_bound)
            results.append(result)

        # Create DataFrame with explicit schema compliance
        df = pd.DataFrame(results)
        # Ensure non-nullable columns have no None values
        df["distribution"] = df["distribution"].astype(str)
        df["sse"] = df["sse"].astype(float)
        return df

    return fit_distributions_batch


def fit_single_distribution(
    dist_name: str,
    data_sample: np.ndarray,
    bin_edges: np.ndarray,
    y_hist: np.ndarray,
    column_name: Optional[str] = None,
    data_stats: Optional[Dict[str, float]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    lazy_metrics: bool = False,
) -> Dict[str, Any]:
    """Fit a single distribution and compute goodness-of-fit metrics.

    Uses CDF-based density computation for accurate SSE calculation.
    Instead of evaluating PDF at bin centers (point approximation),
    we compute the exact average density over each bin using:
        expected_density = (CDF(hi) - CDF(lo)) / bin_width

    This is mathematically equivalent to integrating the PDF over each bin,
    and is both faster (~5%) and more accurate (2-10x better SSE for peaked
    distributions like Weibull, Pareto, Chi-squared).

    Args:
        dist_name: Name of scipy.stats distribution
        data_sample: Sample of raw data for parameter fitting
        bin_edges: Histogram bin edge values (len = n_bins + 1)
        y_hist: Histogram density values
        column_name: Name of the column being fitted (for multi-column support)
        data_stats: Pre-computed summary statistics (data_min, data_max, etc.)
        lower_bound: Lower bound for truncated distribution fitting (v1.4.0)
        upper_bound: Upper bound for truncated distribution fitting (v1.4.0)
        lazy_metrics: If True, skip expensive KS/AD computation. These metrics
            will be None in the result and computed on-demand later. (v1.5.0)

    Returns:
        Dictionary with fit result fields including data_min, data_max, etc.
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # Get distribution object
            dist = getattr(st, dist_name)

            # Fit distribution to data sample (always fit unbounded first)
            params = dist.fit(data_sample)

            # Check for NaN in parameters (convergence failure)
            if any(not np.isfinite(p) for p in params):
                return _failed_fit_result(dist_name, column_name, data_stats, lower_bound, upper_bound)

            # Create frozen distribution (possibly truncated) for metrics
            frozen_dist = dist(*params)
            if lower_bound is not None or upper_bound is not None:
                lb = lower_bound if lower_bound is not None else -np.inf
                ub = upper_bound if upper_bound is not None else np.inf
                frozen_dist = _create_truncated_dist(frozen_dist, lb, ub)

            # CDF-based density computation: exact average density over each bin
            # This is more accurate than point evaluation at bin centers, especially
            # for distributions with rapidly varying PDFs (Weibull, Pareto, etc.)
            bin_widths = np.diff(bin_edges)
            cdf_values = frozen_dist.cdf(bin_edges)
            bin_probs = np.diff(cdf_values)  # P(lo < X < hi) for each bin
            expected_density = bin_probs / bin_widths  # Convert to density
            expected_density = np.nan_to_num(expected_density, nan=0.0, posinf=0.0, neginf=0.0)

            # Compute Sum of Squared Errors
            sse = np.sum((y_hist - expected_density) ** 2.0)

            # Check for invalid SSE
            if not np.isfinite(sse):
                return _failed_fit_result(dist_name, column_name, data_stats, lower_bound, upper_bound)

            # Compute information criteria using frozen distribution (fast, always computed)
            aic, bic = compute_information_criteria_frozen(frozen_dist, len(params), data_sample)

            # Compute expensive metrics only if not lazy
            if lazy_metrics:
                # Skip KS/AD computation for performance - will be computed on-demand
                ks_stat, pvalue = None, None
                ad_stat, ad_pvalue = None, None
            else:
                # Compute Kolmogorov-Smirnov statistic and p-value using frozen distribution
                ks_stat, pvalue = compute_ks_statistic_frozen(frozen_dist, data_sample)

                # Compute Anderson-Darling statistic using frozen distribution
                ad_stat = compute_ad_statistic_frozen(frozen_dist, data_sample)

                # Compute Anderson-Darling p-value (only for supported distributions, unbounded)
                # Note: A-D p-value tables are for standard distributions, not truncated
                ad_pvalue = (
                    compute_ad_pvalue(dist_name, data_sample) if lower_bound is None and upper_bound is None else None
                )

            # Log any warnings that were caught (for debugging)
            for w in caught_warnings:
                if "convergence" in str(w.message).lower() or "nan" in str(w.message).lower():
                    # These indicate fitting issues - return failed result
                    return _failed_fit_result(dist_name, column_name, data_stats, lower_bound, upper_bound)

            return {
                "column_name": column_name,
                "distribution": dist_name,
                "parameters": [float(p) for p in params],
                "sse": float(sse),
                "aic": float(aic),
                "bic": float(bic),
                "ks_statistic": float(ks_stat) if ks_stat is not None else None,
                "pvalue": float(pvalue) if pvalue is not None else None,
                "ad_statistic": float(ad_stat) if ad_stat is not None else None,
                "ad_pvalue": ad_pvalue,
                **(data_stats or {}),  # Flat data stats: data_min, data_max, etc.
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

    except Exception:
        # Catch all exceptions to ensure fitting never crashes the Spark job
        # This matches behavior of LocalBackend and RayBackend which skip failed fits
        return _failed_fit_result(dist_name, column_name, data_stats, lower_bound, upper_bound)


def _create_truncated_dist(frozen_dist, lb: float, ub: float):
    """Create a truncated distribution wrapper.

    Args:
        frozen_dist: Frozen scipy.stats distribution
        lb: Lower bound
        ub: Upper bound

    Returns:
        Truncated distribution wrapper with pdf, logpdf, cdf methods

    Note:
        Uses raise_on_empty=False so fitting continues silently when
        truncation bounds contain no probability mass.
    """
    return TruncatedFrozenDist(frozen_dist, lb, ub, raise_on_empty=False)


def _failed_fit_result(
    dist_name: str,
    column_name: Optional[str] = None,
    data_stats: Optional[Dict[str, float]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Dict[str, Any]:
    """Return sentinel values for failed fits.

    Args:
        dist_name: Name of the distribution that failed
        column_name: Name of the column being fitted (for multi-column support)
        data_stats: Pre-computed summary statistics (data_min, data_max, etc.)
        lower_bound: Lower bound for truncated distribution fitting (v1.4.0)
        upper_bound: Upper bound for truncated distribution fitting (v1.4.0)

    Returns:
        Dictionary with sentinel values indicating fit failure
    """
    return {
        "column_name": column_name,
        "distribution": dist_name,
        "parameters": [float(np.nan)],
        "sse": float(np.inf),
        "aic": float(np.inf),
        "bic": float(np.inf),
        "ks_statistic": float(np.inf),
        "pvalue": 0.0,
        "ad_statistic": float(np.inf),
        "ad_pvalue": None,
        **(data_stats or {}),  # Flat data stats: data_min, data_max, etc.
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def evaluate_pdf(dist: rv_continuous, params: Tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Evaluate probability density function at given points.

    Args:
        dist: scipy.stats distribution object
        params: Distribution parameters (shape params, loc, scale)
        x: Points at which to evaluate PDF

    Returns:
        PDF values at x
    """
    # Extract shape, loc, scale from params
    arg = params[:-2]  # Shape parameters
    loc = params[-2]  # Location
    scale = params[-1]  # Scale

    # Evaluate PDF
    pdf = dist.pdf(x, *arg, loc=loc, scale=scale)

    # Handle potential numerical issues
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

    return pdf


def compute_information_criteria(
    dist: rv_continuous, params: Tuple[float, ...], data: np.ndarray
) -> Tuple[float, float]:
    """Compute AIC and BIC information criteria.

    These criteria help compare model complexity vs fit quality.
    Lower values indicate better models.

    Args:
        dist: scipy.stats distribution object
        params: Fitted distribution parameters
        data: Original data sample

    Returns:
        Tuple of (aic, bic)
    """
    try:
        n = len(data)
        k = len(params)  # Number of parameters

        # Compute log-likelihood
        log_likelihood = np.sum(dist.logpdf(data, *params))

        # Handle numerical issues
        if not np.isfinite(log_likelihood):
            return np.inf, np.inf

        # Akaike Information Criterion
        aic = 2 * k - 2 * log_likelihood

        # Bayesian Information Criterion
        bic = k * np.log(n) - 2 * log_likelihood

        return aic, bic

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, np.inf


def compute_ks_statistic(dist: rv_continuous, params: Tuple[float, ...], data: np.ndarray) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic and p-value.

    The KS statistic measures the maximum distance between the empirical
    distribution function of the sample and the CDF of the fitted distribution.
    Lower values indicate better fit.

    Note:
        When parameters are estimated from the same data being tested (as is
        the case here), the p-values are approximate and tend to be conservative
        (larger than they should be). The KS statistic itself remains valid for
        comparing fits, but p-values should be interpreted with caution.

    Args:
        dist: scipy.stats distribution object
        params: Fitted distribution parameters
        data: Original data sample

    Returns:
        Tuple of (ks_statistic, pvalue)
    """
    try:
        # Use scipy's kstest with the distribution name and fitted parameters
        result = st.kstest(data, dist.name, args=params)
        ks_stat = result.statistic
        pvalue = result.pvalue

        # Handle numerical issues
        if not np.isfinite(ks_stat):
            return np.inf, 0.0
        if not np.isfinite(pvalue):
            pvalue = 0.0

        return ks_stat, pvalue

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, 0.0


def compute_ad_statistic(dist: rv_continuous, params: Tuple[float, ...], data: np.ndarray) -> float:
    """Compute Anderson-Darling statistic.

    The A-D statistic measures how well a distribution fits data, with more
    weight on the tails than the K-S test. Lower values indicate better fit.

    Formula:
        A² = -n - (1/n) Σᵢ₌₁ⁿ (2i-1)[ln F(Xᵢ) + ln(1-F(Xₙ₊₁₋ᵢ))]

    where F is the CDF of the fitted distribution and Xᵢ are the sorted data.

    Args:
        dist: scipy.stats distribution object
        params: Fitted distribution parameters
        data: Original data sample

    Returns:
        Anderson-Darling statistic (A²)
    """
    try:
        n = len(data)
        if n < 2:
            return np.inf

        # Sort data
        sorted_data = np.sort(data)

        # Compute CDF values at sorted data points
        cdf_values = dist.cdf(sorted_data, *params)

        # Clamp CDF values to avoid log(0) or log(negative)
        cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)

        # Compute A-D statistic using the formula
        # A² = -n - (1/n) Σᵢ₌₁ⁿ (2i-1)[ln F(Xᵢ) + ln(1-F(Xₙ₊₁₋ᵢ))]
        i = np.arange(1, n + 1)
        term1 = np.log(cdf_values)
        term2 = np.log(1 - cdf_values[::-1])  # Reversed for X_{n+1-i}
        ad_stat = -n - (1 / n) * np.sum((2 * i - 1) * (term1 + term2))

        if not np.isfinite(ad_stat):
            return np.inf

        return float(ad_stat)

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf


def compute_ad_pvalue(dist_name: str, data: np.ndarray) -> float | None:
    """Compute Anderson-Darling p-value for supported distributions.

    P-values are only available for distributions where scipy has critical
    value tables: norm, expon, logistic, gumbel_r, gumbel_l.

    Note:
        scipy.stats.anderson uses standardized data (zero mean, unit variance)
        for the test, which is the standard approach for A-D testing.

    Args:
        dist_name: Name of scipy.stats distribution
        data: Original data sample

    Returns:
        P-value if the distribution is supported, None otherwise
    """
    if dist_name not in AD_PVALUE_DISTRIBUTIONS:
        return None

    try:
        scipy_dist_name = AD_PVALUE_DISTRIBUTIONS[dist_name]
        result = st.anderson(data, dist=scipy_dist_name)

        # scipy.anderson returns statistic and critical_values/significance_level
        # We need to interpolate to get an approximate p-value
        statistic = result.statistic
        critical_values = result.critical_values
        significance_levels = result.significance_level  # [15, 10, 5, 2.5, 1] percent

        # If statistic is smaller than smallest critical value, p > 0.15
        if statistic < critical_values[0]:
            return 0.25  # Conservative estimate

        # If statistic is larger than largest critical value, p < 0.01
        if statistic > critical_values[-1]:
            return 0.005  # Conservative estimate

        # Interpolate between critical values
        # significance_levels are in percent: [15, 10, 5, 2.5, 1]
        sig_levels_decimal = np.array(significance_levels) / 100.0

        # Find where our statistic falls and interpolate
        for i in range(len(critical_values) - 1):
            if critical_values[i] <= statistic <= critical_values[i + 1]:
                # Linear interpolation in log space for better accuracy
                frac = (statistic - critical_values[i]) / (critical_values[i + 1] - critical_values[i])
                pvalue = sig_levels_decimal[i] - frac * (sig_levels_decimal[i] - sig_levels_decimal[i + 1])
                return float(pvalue)

        return None

    except (ValueError, RuntimeError, FloatingPointError):
        return None


def compute_information_criteria_frozen(frozen_dist: Any, n_params: int, data: np.ndarray) -> Tuple[float, float]:
    """Compute AIC and BIC information criteria using a frozen distribution.

    This version works with frozen (and possibly truncated) distributions,
    unlike compute_information_criteria which requires separate dist and params.

    Args:
        frozen_dist: Frozen scipy.stats distribution (possibly truncated)
        n_params: Number of parameters in the original distribution
        data: Original data sample

    Returns:
        Tuple of (aic, bic)
    """
    try:
        n = len(data)
        k = n_params

        # Compute log-likelihood using frozen distribution
        log_likelihood = np.sum(frozen_dist.logpdf(data))

        # Handle numerical issues
        if not np.isfinite(log_likelihood):
            return np.inf, np.inf

        # Akaike Information Criterion
        aic = 2 * k - 2 * log_likelihood

        # Bayesian Information Criterion
        bic = k * np.log(n) - 2 * log_likelihood

        return aic, bic

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, np.inf


def compute_ks_statistic_frozen(frozen_dist: Any, data: np.ndarray) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic and p-value using a frozen distribution.

    This version works with frozen (and possibly truncated) distributions.

    Args:
        frozen_dist: Frozen scipy.stats distribution (possibly truncated)
        data: Original data sample

    Returns:
        Tuple of (ks_statistic, pvalue)
    """
    try:
        # Use scipy's kstest with the frozen distribution's CDF
        result = st.kstest(data, frozen_dist.cdf)
        ks_stat = result.statistic
        pvalue = result.pvalue

        # Handle numerical issues
        if not np.isfinite(ks_stat):
            return np.inf, 0.0
        if not np.isfinite(pvalue):
            pvalue = 0.0

        return ks_stat, pvalue

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, 0.0


def compute_ad_statistic_frozen(frozen_dist: Any, data: np.ndarray) -> float:
    """Compute Anderson-Darling statistic using a frozen distribution.

    This version works with frozen (and possibly truncated) distributions.

    Args:
        frozen_dist: Frozen scipy.stats distribution (possibly truncated)
        data: Original data sample

    Returns:
        Anderson-Darling statistic (A²)
    """
    try:
        n = len(data)
        if n < 2:
            return np.inf

        # Sort data
        sorted_data = np.sort(data)

        # Compute CDF values at sorted data points using frozen distribution
        cdf_values = frozen_dist.cdf(sorted_data)

        # Clamp CDF values to avoid log(0) or log(negative)
        cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)

        # Compute A-D statistic using the formula
        # A² = -n - (1/n) Σᵢ₌₁ⁿ (2i-1)[ln F(Xᵢ) + ln(1-F(Xₙ₊₁₋ᵢ))]
        i = np.arange(1, n + 1)
        term1 = np.log(cdf_values)
        term2 = np.log(1 - cdf_values[::-1])  # Reversed for X_{n+1-i}
        ad_stat = -n - (1 / n) * np.sum((2 * i - 1) * (term1 + term2))

        if not np.isfinite(ad_stat):
            return np.inf

        return float(ad_stat)

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf


def get_continuous_param_names(dist_name: str) -> List[str]:
    """Get parameter names for a continuous scipy distribution.

    Args:
        dist_name: Name of scipy.stats distribution

    Returns:
        List of parameter names in order: [shape_params..., "loc", "scale"]

    Example:
        >>> get_continuous_param_names("norm")
        ['loc', 'scale']
        >>> get_continuous_param_names("gamma")
        ['a', 'loc', 'scale']
        >>> get_continuous_param_names("beta")
        ['a', 'b', 'loc', 'scale']
    """
    dist = getattr(st, dist_name)
    shapes = dist.shapes
    if shapes:
        shape_names = [s.strip() for s in shapes.split(",")]
    else:
        shape_names = []
    return shape_names + ["loc", "scale"]


def bootstrap_confidence_intervals(
    dist_name: str,
    data: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap confidence intervals for distribution parameters.

    Uses the percentile bootstrap method: resample data with replacement,
    refit the distribution, and compute confidence intervals from the
    empirical distribution of fitted parameters.

    Args:
        dist_name: Name of scipy.stats distribution
        data: Data array used for fitting
        alpha: Significance level (default 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default 1000)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping parameter names to (lower, upper) bounds

    Example:
        >>> data = np.random.normal(loc=10, scale=2, size=1000)
        >>> ci = bootstrap_confidence_intervals("norm", data, alpha=0.05)
        >>> print(ci)
        {'loc': (9.85, 10.15), 'scale': (1.92, 2.08)}

    Note:
        Bootstrap fitting may fail for some resamples (e.g., if resampled
        data doesn't have enough variation). Failed fits are skipped.
    """
    rng = np.random.default_rng(random_seed)
    dist = getattr(st, dist_name)
    n = len(data)

    # Collect bootstrap parameter estimates
    bootstrap_params: List[Tuple[float, ...]] = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(data, size=n, replace=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(sample)
                # Skip if any parameter is non-finite
                if all(np.isfinite(p) for p in params):
                    bootstrap_params.append(params)
        except (ValueError, RuntimeError, FloatingPointError):
            continue  # Skip failed fits

    if len(bootstrap_params) < 10:
        raise ValueError(
            f"Too few successful bootstrap fits ({len(bootstrap_params)}/{n_bootstrap}). "
            "Data may be unsuitable for this distribution."
        )

    # Convert to array for percentile computation
    bootstrap_array = np.array(bootstrap_params)

    # Remove outlier bootstrap estimates using IQR filtering per parameter
    # This prevents extreme outliers from blowing up the CI bounds
    bootstrap_array = _filter_bootstrap_outliers(bootstrap_array)

    if len(bootstrap_array) < 10:
        raise ValueError(
            "Too few bootstrap samples remain after outlier filtering. " "Data may be unsuitable for this distribution."
        )

    # Compute percentile confidence intervals
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    # Get parameter names
    param_names = get_continuous_param_names(dist_name)

    result: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(param_names):
        lower = float(np.percentile(bootstrap_array[:, i], lower_pct))
        upper = float(np.percentile(bootstrap_array[:, i], upper_pct))
        result[name] = (lower, upper)

    return result


def _filter_bootstrap_outliers(bootstrap_array: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Filter bootstrap samples with outlier parameter values using IQR.

    For each parameter, identifies outliers as values beyond Q1 - k*IQR or
    Q3 + k*IQR. Removes entire bootstrap samples (rows) where ANY parameter
    is an outlier.

    Args:
        bootstrap_array: Array of shape (n_bootstrap, n_params)
        k: IQR multiplier for outlier detection (default 3.0 = far outliers)

    Returns:
        Filtered array with outlier rows removed
    """
    n_params = bootstrap_array.shape[1]
    mask = np.ones(len(bootstrap_array), dtype=bool)

    for i in range(n_params):
        col = bootstrap_array[:, i]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1

        # Avoid division by zero for constant parameters
        if iqr == 0:
            continue

        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        mask &= (col >= lower_bound) & (col <= upper_bound)

    return bootstrap_array[mask]


def create_sample_data(
    data_full: np.ndarray, sample_size: int = FITTING_SAMPLE_SIZE, random_seed: int = 42
) -> np.ndarray:
    """Create a sample of data for distribution fitting.

    Most scipy distributions can be fit accurately with ~10k samples,
    avoiding the need to pass entire large datasets to UDFs.

    Args:
        data_full: Full dataset
        sample_size: Target sample size
        random_seed: Random seed for reproducibility

    Returns:
        Sampled data (or full data if smaller than sample_size)
    """
    if len(data_full) <= sample_size:
        return data_full

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(data_full), size=sample_size, replace=False)
    return data_full[indices]


def compute_ks_ad_metrics(
    dist_name: str,
    params: List[float],
    data_sample: np.ndarray,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute KS and AD metrics for a fitted distribution.

    This is the core computation function used for lazy metric evaluation.
    It recreates the frozen distribution and computes all KS/AD metrics.

    Args:
        dist_name: Name of scipy.stats distribution
        params: Fitted distribution parameters
        data_sample: Data sample for metric computation
        lower_bound: Optional lower bound for truncated distributions
        upper_bound: Optional upper bound for truncated distributions

    Returns:
        Tuple of (ks_statistic, pvalue, ad_statistic, ad_pvalue)
        Returns (None, None, None, None) if computation fails
    """
    try:
        # Get distribution object and create frozen distribution
        dist = getattr(st, dist_name)
        frozen_dist = dist(*params)

        # Apply truncation if bounds are set
        if lower_bound is not None or upper_bound is not None:
            lb = lower_bound if lower_bound is not None else -np.inf
            ub = upper_bound if upper_bound is not None else np.inf
            frozen_dist = _create_truncated_dist(frozen_dist, lb, ub)

        # Compute Kolmogorov-Smirnov statistic and p-value
        ks_stat, pvalue = compute_ks_statistic_frozen(frozen_dist, data_sample)

        # Compute Anderson-Darling statistic
        ad_stat = compute_ad_statistic_frozen(frozen_dist, data_sample)

        # Compute Anderson-Darling p-value (only for supported distributions, unbounded)
        ad_pvalue = compute_ad_pvalue(dist_name, data_sample) if lower_bound is None and upper_bound is None else None

        return (
            float(ks_stat) if ks_stat is not None and np.isfinite(ks_stat) else None,
            float(pvalue) if pvalue is not None and np.isfinite(pvalue) else None,
            float(ad_stat) if ad_stat is not None and np.isfinite(ad_stat) else None,
            ad_pvalue,
        )

    except (ValueError, RuntimeError, FloatingPointError, AttributeError):
        return (None, None, None, None)


def extract_distribution_params(params: List[float]) -> Tuple[Tuple[float, ...], float, float]:
    """Extract shape, loc, scale from scipy distribution parameters.

    scipy.stats distributions return parameters as: (shape_params..., loc, scale)
    This function separates them into their components.

    Args:
        params: List of distribution parameters from scipy fit

    Returns:
        Tuple of (shape_params, loc, scale) where shape_params is a tuple
        that may be empty for 2-parameter distributions like normal.

    Example:
        >>> # Normal distribution (no shape params)
        >>> params = [50.0, 10.0]  # loc=50, scale=10
        >>> shape, loc, scale = extract_distribution_params(params)
        >>> # shape=(), loc=50.0, scale=10.0

        >>> # Gamma distribution (1 shape param)
        >>> params = [2.0, 0.0, 5.0]  # a=2, loc=0, scale=5
        >>> shape, loc, scale = extract_distribution_params(params)
        >>> # shape=(2.0,), loc=0.0, scale=5.0
    """
    if len(params) < 2:
        raise ValueError(f"Parameters must have at least 2 elements (loc, scale), got {len(params)}")

    shape = tuple(params[:-2]) if len(params) > 2 else ()
    loc = params[-2]
    scale = params[-1]
    return shape, loc, scale


def compute_pdf_range(
    dist: rv_continuous,
    params: List[float],
    x_hist: np.ndarray,
    percentile: float = 0.01,
) -> Tuple[float, float]:
    """Compute safe range for PDF plotting.

    Uses the distribution's ppf (percent point function) to find a reasonable
    range that covers most of the distribution's mass, with fallback to
    histogram bounds if ppf fails.

    Args:
        dist: scipy.stats distribution object
        params: Distribution parameters
        x_hist: Histogram bin centers (used as fallback)
        percentile: Lower percentile for range (upper = 1 - percentile)

    Returns:
        Tuple of (start, end) for PDF plotting range
    """
    shape, loc, scale = extract_distribution_params(params)

    try:
        start = dist.ppf(percentile, *shape, loc=loc, scale=scale)
        end = dist.ppf(1 - percentile, *shape, loc=loc, scale=scale)
    except (ValueError, RuntimeError, FloatingPointError):
        start = float(x_hist.min())
        end = float(x_hist.max())

    # Validate and fallback for non-finite values
    if not np.isfinite(start):
        start = float(x_hist.min())
    if not np.isfinite(end):
        end = float(x_hist.max())

    return start, end
