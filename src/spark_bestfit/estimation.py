"""Distribution parameter estimation using Pandas UDFs for efficient parallel processing.

This module provides functions for estimating distribution parameters including
Maximum Likelihood Estimation (MLE), Maximum Spacing Estimation (MSE), and
bootstrap confidence intervals.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
from scipy.stats import rv_continuous

from spark_bestfit.metrics import (
    compute_ad_pvalue,
    compute_ad_statistic_frozen,
    compute_information_criteria_frozen,
    compute_ks_statistic_frozen,
)
from spark_bestfit.truncated import create_truncated_dist

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
            # Heavy-tail detection stats (v2.3.0)
            StructField("data_kurtosis", FloatType(), True),
            StructField("data_skewness", FloatType(), True),
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
        Dictionary with keys: data_min, data_max, data_mean, data_stddev, data_count,
        data_kurtosis, data_skewness
    """
    return {
        "data_min": float(np.min(data)),
        "data_max": float(np.max(data)),
        "data_mean": float(np.mean(data)),
        "data_stddev": float(np.std(data)),
        "data_count": float(len(data)),
        "data_kurtosis": float(st.kurtosis(data, fisher=True)),  # Excess kurtosis (0 for normal)
        "data_skewness": float(st.skew(data)),
    }


# Heavy-tailed distributions have infinite variance or slow tail decay
HEAVY_TAIL_DISTRIBUTIONS = frozenset(
    {
        "cauchy",  # Undefined mean and variance
        "levy",  # Infinite variance
        "levy_l",  # Infinite variance
        "levy_stable",  # Can have infinite variance
        "pareto",  # Heavy tail, finite variance only for alpha > 2
        "powerlaw",  # Heavy tail
        "t",  # Heavy tail for low df
        "burr",  # Can be heavy-tailed
        "burr12",  # Can be heavy-tailed
        "fisk",  # Log-logistic, can be heavy-tailed
        "lomax",  # Pareto type II
        "invgauss",  # Can be heavy-tailed
        "genhyperbolic",  # Can be heavy-tailed
        "johnsonsu",  # Can be heavy-tailed for certain parameters
    }
)


def detect_heavy_tail(data: np.ndarray, kurtosis_threshold: float = 6.0) -> Dict[str, Any]:
    """Detect heavy-tail characteristics in data.

    Heavy-tailed distributions have slower tail decay than normal/exponential
    distributions. This function checks multiple indicators:
    1. Excess kurtosis > threshold (normal has excess kurtosis = 0)
    2. Extreme value ratio (max / 99th percentile)

    Args:
        data: Input data array
        kurtosis_threshold: Excess kurtosis threshold for heavy-tail warning.
            Default 6.0 (normal=0, t(5)~6, Cauchy=undefined/inf)

    Returns:
        Dictionary with:
            - is_heavy_tailed: bool, True if heavy-tail indicators present
            - kurtosis: float, excess kurtosis value
            - extreme_ratio: float, max / 99th percentile ratio
            - indicators: list of string descriptions of detected indicators
    """
    clean_data = data[np.isfinite(data)]
    if len(clean_data) < 10:
        return {"is_heavy_tailed": False, "kurtosis": 0.0, "extreme_ratio": 1.0, "indicators": []}

    kurtosis = float(st.kurtosis(clean_data, fisher=True))
    p99 = float(np.percentile(clean_data, 99))
    max_val = float(np.max(clean_data))

    # Avoid division by zero
    extreme_ratio = max_val / p99 if p99 != 0 else 1.0

    indicators = []

    # Check excess kurtosis (normal distribution has excess kurtosis = 0)
    if kurtosis > kurtosis_threshold:
        indicators.append(f"high kurtosis ({kurtosis:.1f} > {kurtosis_threshold})")

    # Check extreme value ratio (heavy tails have outliers far from 99th percentile)
    if extreme_ratio > 3.0:
        indicators.append(f"extreme values (max/p99 = {extreme_ratio:.1f})")

    return {
        "is_heavy_tailed": len(indicators) > 0,
        "kurtosis": kurtosis,
        "extreme_ratio": extreme_ratio,
        "indicators": indicators,
    }


# Type alias for estimation method
EstimationMethod = str  # Literal["mle", "mse", "auto"]


def fit_mse(
    dist: rv_continuous,
    data: np.ndarray,
    initial_params: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, ...]:
    """Fit distribution using Maximum Spacing Estimation (MSE).

    MSE estimates parameters by maximizing the geometric mean of spacings
    between consecutive order statistics of the CDF-transformed data.
    It is particularly robust for heavy-tailed distributions where MLE
    can fail or converge poorly.

    The MSE objective maximizes:
        S(theta) = (1/(n+1)) sum_i log(D_i)

    where D_i = F(x_(i); theta) - F(x_(i-1); theta) are the spacings, with
    x_(0) = -inf (so F(x_(0)) = 0) and x_(n+1) = +inf (so F(x_(n+1)) = 1).

    Args:
        dist: scipy.stats distribution object (unfrozen)
        data: Data array to fit
        initial_params: Initial parameter guess. If None, uses MLE estimate
            as starting point (warm start).

    Returns:
        Tuple of fitted parameters (same format as scipy.stats.fit)

    Raises:
        ValueError: If optimization fails to converge

    References:
        Ranneby, B. (1984). "The Maximum Spacing Method. An Estimation Method
        Related to the Maximum Likelihood Method." Scandinavian Journal of
        Statistics, 11(2), 93-112.

    Example:
        >>> from scipy import stats
        >>> data = stats.pareto.rvs(b=2.5, size=1000, random_state=42)
        >>> params = fit_mse(stats.pareto, data)
        >>> # params ~ (2.5, loc, scale)
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)

    if n < 2:
        raise ValueError("Need at least 2 data points for MSE fitting")

    def mse_objective(params: np.ndarray) -> float:
        """Negative log of geometric mean of spacings (to minimize)."""
        try:
            # Unpack parameters: shape params are all but last two (loc, scale)
            if len(params) > 2:
                shape = tuple(params[:-2])
                loc = params[-2]
                scale = params[-1]
            else:
                shape = ()
                loc = params[0]
                scale = params[1]

            # Ensure scale is positive
            if scale <= 0:
                return np.inf

            # Compute CDF values at sorted data points
            cdf_vals = dist.cdf(sorted_data, *shape, loc=loc, scale=scale)

            # Clamp to (epsilon, 1-epsilon) to avoid log(0)
            epsilon = 1e-10
            cdf_vals = np.clip(cdf_vals, epsilon, 1 - epsilon)

            # Add boundary values: F(x_(0)) = 0, F(x_(n+1)) = 1
            u = np.concatenate([[0.0], cdf_vals, [1.0]])

            # Compute spacings D_i = u_(i) - u_(i-1)
            spacings = np.diff(u)

            # Ensure no zero or negative spacings
            spacings = np.maximum(spacings, epsilon)

            # MSE objective: minimize negative mean log spacing
            # (equivalent to maximizing geometric mean of spacings)
            return -np.mean(np.log(spacings))

        except (ValueError, RuntimeError, FloatingPointError):
            return np.inf

    # Get initial parameter estimate using MLE if not provided
    if initial_params is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                initial_params = dist.fit(data)
        except Exception:
            # If MLE fails, try method of moments or default guess
            # Use data statistics for a rough initial guess
            mean, std = np.mean(data), np.std(data)
            if dist.shapes:
                # For distributions with shape parameters, start with 1.0
                n_shapes = len(dist.shapes.split(","))
                initial_params = tuple([1.0] * n_shapes + [mean, std])
            else:
                initial_params = (mean, std)

    # Optimize using Nelder-Mead (robust for non-smooth objectives)
    result = minimize(
        mse_objective,
        initial_params,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
    )

    if not result.success or result.fun == np.inf:
        # Try with L-BFGS-B as fallback (handles bounds better)
        n_params = len(initial_params)
        # Set bounds: shape params unbounded, loc unbounded, scale > 0
        bounds = [(None, None)] * (n_params - 1) + [(1e-10, None)]
        result = minimize(
            mse_objective,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000},
        )

    if not result.success or result.fun == np.inf:
        raise ValueError(f"MSE optimization failed: {result.message}")

    return tuple(result.x)


def create_fitting_udf(
    histogram_broadcast: Broadcast[Tuple[np.ndarray, np.ndarray]],
    data_sample_broadcast: Broadcast[np.ndarray],
    column_name: Optional[str] = None,
    data_stats: Optional[Dict[str, float]] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    lazy_metrics: bool = False,
    custom_distributions_broadcast: Optional[Broadcast[Dict[str, rv_continuous]]] = None,
    estimation_method: str = "mle",
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
        custom_distributions_broadcast: Broadcast variable containing dict of
            custom distributions. If provided, distribution names are looked up
            here first before falling back to scipy.stats. (v2.4.0)
        estimation_method: Parameter estimation method (v2.5.0):
            - "mle": Maximum Likelihood Estimation (default)
            - "mse": Maximum Spacing Estimation (robust for heavy-tailed data)

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
        custom_distributions = custom_distributions_broadcast.value if custom_distributions_broadcast else None

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
                    custom_distributions=custom_distributions,
                    estimation_method=estimation_method,
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
    custom_distributions: Optional[Dict[str, rv_continuous]] = None,
    estimation_method: str = "mle",
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
        dist_name: Name of scipy.stats distribution or custom distribution
        data_sample: Sample of raw data for parameter fitting
        bin_edges: Histogram bin edge values (len = n_bins + 1)
        y_hist: Histogram density values
        column_name: Name of the column being fitted (for multi-column support)
        data_stats: Pre-computed summary statistics (data_min, data_max, etc.)
        lower_bound: Lower bound for truncated distribution fitting (v1.4.0)
        upper_bound: Upper bound for truncated distribution fitting (v1.4.0)
        lazy_metrics: If True, skip expensive KS/AD computation. These metrics
            will be None in the result and computed on-demand later. (v1.5.0)
        custom_distributions: Dict mapping custom distribution names to
            rv_continuous objects. If provided, dist_name is looked up here
            first, falling back to scipy.stats if not found. (v2.4.0)
        estimation_method: Parameter estimation method (v2.5.0):
            - "mle": Maximum Likelihood Estimation (default, uses scipy.stats.fit)
            - "mse": Maximum Spacing Estimation (robust for heavy-tailed data)

    Returns:
        Dictionary with fit result fields including data_min, data_max, etc.
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # Get distribution object - check custom distributions first
            if custom_distributions and dist_name in custom_distributions:
                dist = custom_distributions[dist_name]
            else:
                dist = getattr(st, dist_name)

            # Fit distribution to data sample using specified estimation method
            if estimation_method == "mse":
                try:
                    params = fit_mse(dist, data_sample)
                except (ValueError, RuntimeError):
                    # Fall back to MLE if MSE fails
                    params = dist.fit(data_sample)
            else:
                # Default: MLE via scipy.stats.fit
                params = dist.fit(data_sample)

            # Check for NaN in parameters (convergence failure)
            if any(not np.isfinite(p) for p in params):
                return _failed_fit_result(dist_name, column_name, data_stats, lower_bound, upper_bound)

            # Create frozen distribution (possibly truncated) for metrics
            frozen_dist = dist(*params)
            if lower_bound is not None or upper_bound is not None:
                lb = lower_bound if lower_bound is not None else -np.inf
                ub = upper_bound if upper_bound is not None else np.inf
                frozen_dist = create_truncated_dist(frozen_dist, lb, ub)

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


__all__ = [
    # Constants
    "FITTING_SAMPLE_SIZE",
    "FIT_RESULT_SCHEMA",
    "HEAVY_TAIL_DISTRIBUTIONS",
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
    # Helpers
    "_failed_fit_result",
    "_filter_bootstrap_outliers",
    "evaluate_pdf",
    "get_continuous_param_names",
    "extract_distribution_params",
    "compute_pdf_range",
]
