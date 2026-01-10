"""Goodness-of-fit metrics for distribution fitting.

This module provides functions for computing various goodness-of-fit statistics
including information criteria (AIC, BIC), Kolmogorov-Smirnov tests, and
Anderson-Darling tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as st
from scipy.stats import rv_continuous

from spark_bestfit.truncated import create_truncated_dist

# Distributions that support Anderson-Darling p-value computation via scipy
# Maps our distribution names to scipy.anderson's dist parameter
AD_PVALUE_DISTRIBUTIONS: Dict[str, str] = {
    "norm": "norm",
    "expon": "expon",
    "logistic": "logistic",
    "gumbel_r": "gumbel",
    "gumbel_l": "gumbel_l",
}


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


def compute_ad_statistic(dist: rv_continuous, params: Tuple[float, ...], data: np.ndarray) -> float:
    """Compute Anderson-Darling statistic.

    The A-D statistic measures how well a distribution fits data, with more
    weight on the tails than the K-S test. Lower values indicate better fit.

    Formula:
        A^2 = -n - (1/n) sum_i (2i-1)[ln F(X_i) + ln(1-F(X_{n+1-i}))]

    where F is the CDF of the fitted distribution and X_i are the sorted data.

    Args:
        dist: scipy.stats distribution object
        params: Fitted distribution parameters
        data: Original data sample

    Returns:
        Anderson-Darling statistic (A^2)
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
        # A^2 = -n - (1/n) sum_i (2i-1)[ln F(X_i) + ln(1-F(X_{n+1-i}))]
        i = np.arange(1, n + 1)
        term1 = np.log(cdf_values)
        term2 = np.log(1 - cdf_values[::-1])  # Reversed for X_{n+1-i}
        ad_stat = -n - (1 / n) * np.sum((2 * i - 1) * (term1 + term2))

        if not np.isfinite(ad_stat):
            return np.inf

        return float(ad_stat)

    except (ValueError, RuntimeError, FloatingPointError):
        return np.inf


def compute_ad_statistic_frozen(frozen_dist: Any, data: np.ndarray) -> float:
    """Compute Anderson-Darling statistic using a frozen distribution.

    This version works with frozen (and possibly truncated) distributions.

    Args:
        frozen_dist: Frozen scipy.stats distribution (possibly truncated)
        data: Original data sample

    Returns:
        Anderson-Darling statistic (A^2)
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
        # A^2 = -n - (1/n) sum_i (2i-1)[ln F(X_i) + ln(1-F(X_{n+1-i}))]
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


def compute_ks_ad_metrics(
    dist_name: str,
    params: List[float],
    data_sample: np.ndarray,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    custom_distributions: Optional[Dict[str, rv_continuous]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute KS and AD metrics for a fitted distribution.

    This is the core computation function used for lazy metric evaluation.
    It recreates the frozen distribution and computes all KS/AD metrics.

    Args:
        dist_name: Name of scipy.stats distribution or custom distribution
        params: Fitted distribution parameters
        data_sample: Data sample for metric computation
        lower_bound: Optional lower bound for truncated distributions
        upper_bound: Optional upper bound for truncated distributions
        custom_distributions: Dict mapping custom distribution names to
            rv_continuous objects. Used for lazy metrics on custom dists. (v2.4.0)

    Returns:
        Tuple of (ks_statistic, pvalue, ad_statistic, ad_pvalue)
        Returns (None, None, None, None) if computation fails
    """
    try:
        # Get distribution object - check custom distributions first
        if custom_distributions and dist_name in custom_distributions:
            dist = custom_distributions[dist_name]
        else:
            dist = getattr(st, dist_name)
        frozen_dist = dist(*params)

        # Apply truncation if bounds are set
        if lower_bound is not None or upper_bound is not None:
            lb = lower_bound if lower_bound is not None else -np.inf
            ub = upper_bound if upper_bound is not None else np.inf
            frozen_dist = create_truncated_dist(frozen_dist, lb, ub)

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


__all__ = [
    "AD_PVALUE_DISTRIBUTIONS",
    "compute_information_criteria",
    "compute_information_criteria_frozen",
    "compute_ks_statistic",
    "compute_ks_statistic_frozen",
    "compute_ad_statistic",
    "compute_ad_statistic_frozen",
    "compute_ad_pvalue",
    "compute_ks_ad_metrics",
]
