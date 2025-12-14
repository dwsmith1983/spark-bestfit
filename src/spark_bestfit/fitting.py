"""Distribution fitting using Pandas UDFs for efficient parallel processing."""

import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from pyspark import Broadcast
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType
from scipy.stats import rv_continuous

# Constant for fitting sample size
FITTING_SAMPLE_SIZE: int = 10_000  # Most scipy distributions fit well with 10k samples

# Define output schema for Pandas UDF
# Note: Pandas infers all columns as nullable, so we match that here
FIT_RESULT_SCHEMA = StructType(
    [
        StructField("distribution", StringType(), True),
        StructField("parameters", ArrayType(FloatType()), True),
        StructField("sse", FloatType(), True),
        StructField("aic", FloatType(), True),
        StructField("bic", FloatType(), True),
    ]
)


def create_fitting_udf(
    histogram_broadcast: Broadcast[Tuple[np.ndarray, np.ndarray]],
    data_sample_broadcast: Broadcast[np.ndarray],
) -> Callable[[pd.Series], pd.DataFrame]:
    """Factory function to create Pandas UDF with broadcasted data.

    This is the KEY optimization: The histogram and data sample are
    broadcasted once to all executors, then the Pandas UDF processes
    batches of distributions efficiently using vectorized operations.

    Args:
        histogram_broadcast: Broadcast variable containing (y_hist, x_hist)
        data_sample_broadcast: Broadcast variable containing data sample

    Returns:
        Pandas UDF function for fitting distributions

    Example:
        >>> # In DistributionFitter:
        >>> hist_bc = spark.sparkContext.broadcast((y_hist, x_hist))
        >>> data_bc = spark.sparkContext.broadcast(data_sample)
        >>> fitting_udf = create_fitting_udf(hist_bc, data_bc)
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
            DataFrame with columns: distribution, parameters, sse, aic, bic
        """
        # Get broadcasted data (no serialization overhead!)
        y_hist, x_hist = histogram_broadcast.value
        data_sample = data_sample_broadcast.value

        # Fit each distribution in the batch
        results = []
        for dist_name in distribution_names:
            result = fit_single_distribution(
                dist_name=dist_name,
                data_sample=data_sample,
                x_hist=x_hist,
                y_hist=y_hist,
            )
            results.append(result)

        # Create DataFrame with explicit schema compliance
        df = pd.DataFrame(results)
        # Ensure non-nullable columns have no None values
        df["distribution"] = df["distribution"].astype(str)
        df["sse"] = df["sse"].astype(float)
        return df

    return fit_distributions_batch


def fit_single_distribution(
    dist_name: str, data_sample: np.ndarray, x_hist: np.ndarray, y_hist: np.ndarray
) -> Dict[str, Any]:
    """Fit a single distribution and compute goodness-of-fit metrics.

    Args:
        dist_name: Name of scipy.stats distribution
        data_sample: Sample of raw data for parameter fitting
        x_hist: Histogram bin centers
        y_hist: Histogram density values

    Returns:
        Dictionary with keys: distribution, parameters, sse, aic, bic
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # Get distribution object
            dist = getattr(st, dist_name)

            # Fit distribution to data sample
            params = dist.fit(data_sample)

            # Check for NaN in parameters (convergence failure)
            if any(not np.isfinite(p) for p in params):
                return _failed_fit_result(dist_name)

            # Evaluate PDF at histogram bin centers
            pdf_values = evaluate_pdf(dist, params, x_hist)

            # Compute Sum of Squared Errors
            sse = np.sum((y_hist - pdf_values) ** 2.0)

            # Check for invalid SSE
            if not np.isfinite(sse):
                return _failed_fit_result(dist_name)

            # Compute information criteria
            aic, bic = compute_information_criteria(dist, params, data_sample)

            # Log any warnings that were caught (for debugging)
            for w in caught_warnings:
                if "convergence" in str(w.message).lower() or "nan" in str(w.message).lower():
                    # These indicate fitting issues - return failed result
                    return _failed_fit_result(dist_name)

            return {
                "distribution": dist_name,
                "parameters": [float(p) for p in params],
                "sse": float(sse),
                "aic": float(aic),
                "bic": float(bic),
            }

    except (ValueError, RuntimeError, FloatingPointError, AttributeError):
        return _failed_fit_result(dist_name)


def _failed_fit_result(dist_name: str) -> Dict[str, Any]:
    """Return sentinel values for failed fits.

    Args:
        dist_name: Name of the distribution that failed

    Returns:
        Dictionary with sentinel values indicating fit failure
    """
    return {
        "distribution": dist_name,
        "parameters": [float(np.nan)],
        "sse": float(np.inf),
        "aic": float(np.inf),
        "bic": float(np.inf),
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
