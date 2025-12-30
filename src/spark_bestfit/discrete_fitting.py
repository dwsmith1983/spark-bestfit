"""Discrete distribution fitting using MLE optimization and Pandas UDFs."""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st
from pyspark import Broadcast
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType, MapType, StringType, StructField, StructType

from spark_bestfit.distributions import DiscreteDistributionRegistry

# Output schema for discrete fitting results
# Note: ad_statistic and ad_pvalue are included for schema compatibility with FitResults
# but are always None for discrete distributions (A-D test is for continuous distributions)
DISCRETE_FIT_RESULT_SCHEMA = StructType(
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
        # data_summary: summary statistics of the original data for provenance
        StructField("data_summary", MapType(StringType(), FloatType()), True),
    ]
)


def fit_discrete_mle(
    dist_name: str,
    data: np.ndarray,
    initial_params: List[float],
    bounds: List[Tuple[float, float]],
) -> Tuple[np.ndarray, float]:
    """Fit a discrete distribution using maximum likelihood estimation.

    Since scipy discrete distributions don't have a fit() method, we use
    scipy.optimize.minimize to find parameters that maximize the likelihood.

    Args:
        dist_name: Name of the scipy.stats discrete distribution
        data: Integer data to fit
        initial_params: Initial parameter guesses
        bounds: Parameter bounds as list of (min, max) tuples

    Returns:
        Tuple of (fitted_params, negative_log_likelihood)

    Raises:
        ValueError: If optimization fails to converge
    """
    dist = getattr(st, dist_name)

    def neg_log_likelihood(params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        try:
            # Ensure integer parameters where needed (e.g., n in binomial)
            int_param_dists = {"binom", "betabinom", "hypergeom", "nhypergeom", "boltzmann", "zipfian"}
            if dist_name in int_param_dists:
                # First parameter is typically the integer one
                params = list(params)
                params[0] = int(round(params[0]))
                params = tuple(params)

            ll = np.sum(dist.logpmf(data, *params))
            if not np.isfinite(ll):
                return np.inf
            return -ll
        except (ValueError, RuntimeError, ZeroDivisionError):
            return np.inf

    # Run optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opt.minimize(
            neg_log_likelihood,
            initial_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8},
        )

    if not result.success and result.fun == np.inf:
        raise ValueError(f"Optimization failed for {dist_name}: {result.message}")

    return result.x, result.fun


def compute_discrete_histogram(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram for discrete (integer) data.

    Unlike continuous histograms, discrete histograms use integer-aligned bins
    and compute empirical probability mass function (PMF).

    Args:
        data: Integer data array

    Returns:
        Tuple of (values, pmf) where:
            - values: unique integer values in data
            - pmf: empirical probability mass at each value
    """
    # Get unique values and counts
    values, counts = np.unique(data.astype(int), return_counts=True)

    # Convert to PMF (probability mass)
    pmf = counts / len(data)

    return values, pmf


def evaluate_pmf(
    dist: Any,
    params: Tuple[float, ...],
    x: np.ndarray,
    dist_name: str,
) -> np.ndarray:
    """Evaluate probability mass function at given integer points.

    Args:
        dist: scipy.stats discrete distribution object
        params: Distribution parameters
        x: Integer points at which to evaluate PMF
        dist_name: Name of distribution (for special handling)

    Returns:
        PMF values at x
    """
    # Handle distributions requiring integer parameters
    int_param_dists = {"binom", "betabinom", "hypergeom", "nhypergeom", "boltzmann", "zipfian"}
    if dist_name in int_param_dists:
        params_list = list(params)
        params_list[0] = int(round(params_list[0]))
        params = tuple(params_list)

    try:
        pmf = dist.pmf(x, *params)
        pmf = np.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0)
        return pmf
    except (ValueError, RuntimeError):
        return np.zeros_like(x, dtype=float)


def compute_discrete_sse(
    dist: Any,
    params: Tuple[float, ...],
    x_values: np.ndarray,
    empirical_pmf: np.ndarray,
    dist_name: str,
) -> float:
    """Compute sum of squared errors between empirical and fitted PMF.

    Args:
        dist: scipy.stats discrete distribution object
        params: Fitted distribution parameters
        x_values: Integer values where PMF is evaluated
        empirical_pmf: Empirical probability mass at each x value
        dist_name: Name of distribution

    Returns:
        Sum of squared errors
    """
    fitted_pmf = evaluate_pmf(dist, params, x_values, dist_name)
    sse = np.sum((empirical_pmf - fitted_pmf) ** 2)

    if not np.isfinite(sse):
        return np.inf

    return float(sse)


def compute_discrete_information_criteria(
    dist: Any,
    params: Tuple[float, ...],
    data: np.ndarray,
    dist_name: str,
) -> Tuple[float, float]:
    """Compute AIC and BIC for discrete distribution.

    Args:
        dist: scipy.stats discrete distribution object
        params: Fitted distribution parameters
        data: Original integer data
        dist_name: Name of distribution

    Returns:
        Tuple of (aic, bic)
    """
    try:
        # Handle integer parameter distributions
        int_param_dists = {"binom", "betabinom", "hypergeom", "nhypergeom", "boltzmann", "zipfian"}
        if dist_name in int_param_dists:
            params_list = list(params)
            params_list[0] = int(round(params_list[0]))
            params = tuple(params_list)

        n = len(data)
        k = len(params)

        # Compute log-likelihood using logpmf
        log_likelihood = np.sum(dist.logpmf(data, *params))

        if not np.isfinite(log_likelihood):
            return np.inf, np.inf

        # AIC and BIC
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return float(aic), float(bic)

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, np.inf


def compute_discrete_ks_statistic(
    dist: Any,
    params: Tuple[float, ...],
    data: np.ndarray,
    dist_name: str,
) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic for discrete distribution.

    Computes the two-sided KS statistic D_n = max(D+, D-) which measures
    the maximum distance between empirical and theoretical CDFs.

    Note:
        The standard KS test assumes continuous distributions.
        For discrete distributions, the KS statistic is valid for comparing
        fits, but p-values are conservative and should not be used for
        formal hypothesis testing. Use AIC/BIC for model selection instead.

    Args:
        dist: scipy.stats discrete distribution object
        params: Fitted distribution parameters
        data: Original integer data
        dist_name: Name of distribution

    Returns:
        Tuple of (ks_statistic, pvalue) where pvalue is approximate only
    """
    try:
        # Handle integer parameter distributions
        int_param_dists = {"binom", "betabinom", "hypergeom", "nhypergeom", "boltzmann", "zipfian"}
        if dist_name in int_param_dists:
            params_list = list(params)
            params_list[0] = int(round(params_list[0]))
            params = tuple(params_list)

        # Compute empirical CDF
        sorted_data = np.sort(data)
        n = len(data)

        # Compute theoretical CDF at sorted data points
        tcdf = dist.cdf(sorted_data, *params)

        # Two-sided KS statistic: D_n = max(D+, D-)
        # D+ = max_i(i/n - F(x_i)) - max deviation where empirical > theoretical
        # D- = max_i(F(x_i) - (i-1)/n) - max deviation where theoretical > empirical
        ecdf_upper = np.arange(1, n + 1) / n  # F_n(x_i) = i/n (value after jump)
        ecdf_lower = np.arange(0, n) / n  # F_n(x_i-) = (i-1)/n (value before jump)

        d_plus = np.max(ecdf_upper - tcdf)
        d_minus = np.max(tcdf - ecdf_lower)
        ks_stat = max(d_plus, d_minus)

        # Approximate p-value using asymptotic distribution
        # Note: This is approximate for discrete distributions
        # sqrt(n) * D_n converges to Kolmogorov distribution
        pvalue = float(st.kstwobign.sf(np.sqrt(n) * ks_stat))

        if not np.isfinite(ks_stat):
            return np.inf, 0.0
        if not np.isfinite(pvalue):
            pvalue = 0.0

        return float(ks_stat), pvalue

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf, 0.0


def fit_single_discrete_distribution(
    dist_name: str,
    data_sample: np.ndarray,
    x_values: np.ndarray,
    empirical_pmf: np.ndarray,
    registry: DiscreteDistributionRegistry,
    column_name: Optional[str] = None,
    data_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fit a single discrete distribution and compute goodness-of-fit metrics.

    Args:
        dist_name: Name of scipy.stats discrete distribution
        data_sample: Sample of integer data for parameter fitting
        x_values: Unique integer values in data
        empirical_pmf: Empirical PMF at each x value
        registry: DiscreteDistributionRegistry for parameter configs
        column_name: Name of the column being fitted (for multi-column support)
        data_summary: Pre-computed summary statistics of the original data

    Returns:
        Dictionary with keys: column_name, distribution, parameters, sse, aic, bic, ks_statistic, pvalue, data_summary
    """
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # Get distribution object
            dist = getattr(st, dist_name)

            # Get parameter configuration
            config = registry.get_param_config(dist_name)
            initial = config["initial"](data_sample)
            bounds = config["bounds"](data_sample)

            # Fit using MLE optimization
            params, neg_ll = fit_discrete_mle(dist_name, data_sample, initial, bounds)

            # Check for invalid parameters
            if any(not np.isfinite(p) for p in params):
                return _failed_discrete_fit_result(dist_name, column_name, data_summary)

            # Compute SSE using PMF
            sse = compute_discrete_sse(dist, tuple(params), x_values, empirical_pmf, dist_name)

            if not np.isfinite(sse):
                return _failed_discrete_fit_result(dist_name, column_name, data_summary)

            # Compute information criteria
            aic, bic = compute_discrete_information_criteria(dist, tuple(params), data_sample, dist_name)

            # Compute KS statistic
            ks_stat, pvalue = compute_discrete_ks_statistic(dist, tuple(params), data_sample, dist_name)

            # Check for convergence warnings
            for w in caught_warnings:
                if "convergence" in str(w.message).lower() or "nan" in str(w.message).lower():
                    return _failed_discrete_fit_result(dist_name, column_name, data_summary)

            return {
                "column_name": column_name,
                "distribution": dist_name,
                "parameters": [float(p) for p in params],
                "sse": float(sse),
                "aic": float(aic),
                "bic": float(bic),
                "ks_statistic": float(ks_stat),
                "pvalue": float(pvalue),
                "ad_statistic": None,  # A-D not computed for discrete distributions
                "ad_pvalue": None,
                "data_summary": data_summary,
            }

    except (ValueError, RuntimeError, FloatingPointError, AttributeError):
        return _failed_discrete_fit_result(dist_name, column_name, data_summary)


def _failed_discrete_fit_result(
    dist_name: str,
    column_name: Optional[str] = None,
    data_summary: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return sentinel values for failed discrete fits.

    Args:
        dist_name: Name of the distribution that failed
        column_name: Name of the column being fitted (for multi-column support)
        data_summary: Pre-computed summary statistics of the original data

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
        "ad_statistic": None,  # A-D not computed for discrete distributions
        "ad_pvalue": None,
        "data_summary": data_summary,
    }


def create_discrete_fitting_udf(
    histogram_broadcast: Broadcast[Tuple[np.ndarray, np.ndarray]],
    data_sample_broadcast: Broadcast[np.ndarray],
    column_name: Optional[str] = None,
    data_summary: Optional[Dict[str, float]] = None,
) -> Callable[[pd.Series], pd.DataFrame]:
    """Factory function to create Pandas UDF for discrete distribution fitting.

    Args:
        histogram_broadcast: Broadcast variable containing (x_values, empirical_pmf)
        data_sample_broadcast: Broadcast variable containing integer data sample
        column_name: Name of the column being fitted (for result tracking)
        data_summary: Pre-computed summary statistics of the original data

    Returns:
        Pandas UDF function for fitting discrete distributions
    """
    # Create registry once - will be serialized to workers
    registry = DiscreteDistributionRegistry()

    @pandas_udf(DISCRETE_FIT_RESULT_SCHEMA)
    def fit_discrete_distributions_batch(distribution_names: pd.Series) -> pd.DataFrame:
        """Vectorized UDF to fit multiple discrete distributions in a batch.

        Args:
            distribution_names: Series of scipy discrete distribution names

        Returns:
            DataFrame with columns: column_name, distribution, parameters, sse, aic, bic, ks_statistic, pvalue, data_summary
        """
        # Get broadcasted data
        x_values, empirical_pmf = histogram_broadcast.value
        data_sample = data_sample_broadcast.value

        # Fit each distribution in the batch
        results = []
        for dist_name in distribution_names:
            result = fit_single_discrete_distribution(
                dist_name=dist_name,
                data_sample=data_sample,
                x_values=x_values,
                empirical_pmf=empirical_pmf,
                registry=registry,
                column_name=column_name,
                data_summary=data_summary,
            )
            results.append(result)

        # Create DataFrame
        df = pd.DataFrame(results)
        df["distribution"] = df["distribution"].astype(str)
        df["sse"] = df["sse"].astype(float)
        return df

    return fit_discrete_distributions_batch


def create_discrete_sample_data(
    data_full: np.ndarray,
    sample_size: int = 10_000,
    random_seed: int = 42,
) -> np.ndarray:
    """Create a sample of discrete data for distribution fitting.

    Args:
        data_full: Full integer dataset
        sample_size: Target sample size
        random_seed: Random seed for reproducibility

    Returns:
        Sampled integer data
    """
    if len(data_full) <= sample_size:
        return data_full.astype(int)

    rng = np.random.RandomState(random_seed)
    indices = rng.choice(len(data_full), size=sample_size, replace=False)
    return data_full[indices].astype(int)


def get_discrete_param_names(dist_name: str) -> List[str]:
    """Get parameter names for a discrete scipy distribution.

    Args:
        dist_name: Name of scipy.stats discrete distribution

    Returns:
        List of parameter names

    Example:
        >>> get_discrete_param_names("poisson")
        ['mu']
        >>> get_discrete_param_names("binom")
        ['n', 'p']
        >>> get_discrete_param_names("nbinom")
        ['n', 'p']
    """
    registry = DiscreteDistributionRegistry()
    config = registry.get_param_config(dist_name)
    return config["param_names"]


def bootstrap_discrete_confidence_intervals(
    dist_name: str,
    data: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap confidence intervals for discrete distribution parameters.

    Uses the percentile bootstrap method: resample data with replacement,
    refit the distribution using MLE, and compute confidence intervals from
    the empirical distribution of fitted parameters.

    Args:
        dist_name: Name of scipy.stats discrete distribution
        data: Integer data array used for fitting
        alpha: Significance level (default 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default 1000)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping parameter names to (lower, upper) bounds

    Example:
        >>> data = np.random.poisson(lam=7, size=1000)
        >>> ci = bootstrap_discrete_confidence_intervals("poisson", data, alpha=0.05)
        >>> print(ci)
        {'mu': (6.75, 7.25)}

    Note:
        Bootstrap fitting may fail for some resamples. Failed fits are skipped.
    """
    rng = np.random.default_rng(random_seed)
    data = data.astype(int)
    n = len(data)

    # Get parameter configuration
    registry = DiscreteDistributionRegistry()
    config = registry.get_param_config(dist_name)
    param_names = config["param_names"]

    # Collect bootstrap parameter estimates
    bootstrap_params: List[Tuple[float, ...]] = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(data, size=n, replace=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                initial = config["initial"](sample)
                bounds = config["bounds"](sample)
                params, _ = fit_discrete_mle(dist_name, sample, initial, bounds)
                # Skip if any parameter is non-finite
                if all(np.isfinite(p) for p in params):
                    bootstrap_params.append(tuple(params))
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
