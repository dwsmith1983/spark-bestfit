"""Fast percent point function (PPF/inverse CDF) implementations.

This module provides optimized PPF computations for common distributions,
bypassing scipy.stats overhead by calling scipy.special functions directly.

The standard scipy.stats.rv_continuous.ppf() uses iterative root-finding,
which adds ~28x overhead compared to direct formulas. This module implements
closed-form PPFs for distributions where they exist.

Supported distributions with fast PPF:
    - norm: Normal/Gaussian
    - expon: Exponential
    - uniform: Uniform
    - lognorm: Log-normal
    - weibull_min: Weibull (minimum)
    - gamma: Gamma
    - beta: Beta

Example:
    >>> from spark_bestfit.fast_ppf import fast_ppf
    >>> import numpy as np
    >>>
    >>> # Fast PPF for normal distribution (loc=0, scale=1)
    >>> q = np.array([0.1, 0.5, 0.9])
    >>> values = fast_ppf("norm", (0, 1), q)
    >>>
    >>> # Falls back to scipy for unsupported distributions
    >>> values = fast_ppf("pareto", (2.0, 0, 1), q)
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import special
from scipy import stats as st

# Type alias for PPF function signature
PPFFunc = Callable[[np.ndarray, Tuple], np.ndarray]


def _ppf_norm(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for normal distribution.

    ppf(q) = loc + scale * ndtri(q)

    where ndtri is the inverse of the standard normal CDF (erfinv-based).
    """
    if len(params) >= 2:
        loc, scale = params[-2], params[-1]
    elif len(params) == 1:
        loc, scale = params[0], 1.0
    else:
        loc, scale = 0.0, 1.0
    return loc + scale * special.ndtri(q)


def _ppf_expon(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for exponential distribution.

    ppf(q) = loc - scale * log(1 - q)

    Using log1p(-q) for numerical stability when q is close to 0.
    """
    if len(params) >= 2:
        loc, scale = params[-2], params[-1]
    elif len(params) == 1:
        loc, scale = params[0], 1.0
    else:
        loc, scale = 0.0, 1.0
    # Use -log1p(-q) which equals -log(1-q) but is more stable for small q
    return loc - scale * np.log1p(-q)


def _ppf_uniform(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for uniform distribution.

    ppf(q) = loc + scale * q

    Scipy uniform is on [loc, loc+scale].
    """
    if len(params) >= 2:
        loc, scale = params[-2], params[-1]
    elif len(params) == 1:
        loc, scale = params[0], 1.0
    else:
        loc, scale = 0.0, 1.0
    return loc + scale * q


def _ppf_lognorm(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for log-normal distribution.

    Scipy lognorm parameterization: X = exp(s*Z)*scale + loc where Z ~ N(0,1)
    ppf(q) = exp(s * ndtri(q)) * scale + loc

    params = (s, loc, scale) where s is the shape parameter (sigma of log).
    """
    if len(params) >= 3:
        s, loc, scale = params[0], params[-2], params[-1]
    elif len(params) == 2:
        s, loc, scale = params[0], 0.0, params[1]
    else:
        s, loc, scale = params[0], 0.0, 1.0
    return np.exp(s * special.ndtri(q)) * scale + loc


def _ppf_weibull_min(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for Weibull (minimum) distribution.

    ppf(q) = loc + scale * (-log(1 - q))^(1/c)

    params = (c, loc, scale) where c is the shape parameter.
    """
    if len(params) >= 3:
        c, loc, scale = params[0], params[-2], params[-1]
    elif len(params) == 2:
        c, loc, scale = params[0], 0.0, params[1]
    else:
        c, loc, scale = params[0], 0.0, 1.0
    # Use -log1p(-q) for stability
    return loc + scale * np.power(-np.log1p(-q), 1.0 / c)


def _ppf_gamma(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for gamma distribution.

    ppf(q) = loc + scale * gammaincinv(a, q)

    params = (a, loc, scale) where a is the shape parameter.
    """
    if len(params) >= 3:
        a, loc, scale = params[0], params[-2], params[-1]
    elif len(params) == 2:
        a, loc, scale = params[0], 0.0, params[1]
    else:
        a, loc, scale = params[0], 0.0, 1.0
    return loc + scale * special.gammaincinv(a, q)


def _ppf_beta(q: np.ndarray, params: Tuple) -> np.ndarray:
    """Fast PPF for beta distribution.

    ppf(q) = loc + scale * betaincinv(a, b, q)

    params = (a, b, loc, scale) where a, b are the shape parameters.
    """
    if len(params) >= 4:
        a, b, loc, scale = params[0], params[1], params[-2], params[-1]
    elif len(params) == 3:
        a, b, loc, scale = params[0], params[1], 0.0, params[2]
    elif len(params) == 2:
        a, b, loc, scale = params[0], params[1], 0.0, 1.0
    else:
        raise ValueError("Beta distribution requires at least 2 shape parameters (a, b)")
    return loc + scale * special.betaincinv(a, b, q)


# Registry of fast PPF implementations
_FAST_PPF_REGISTRY: Dict[str, PPFFunc] = {
    "norm": _ppf_norm,
    "expon": _ppf_expon,
    "uniform": _ppf_uniform,
    "lognorm": _ppf_lognorm,
    "weibull_min": _ppf_weibull_min,
    "gamma": _ppf_gamma,
    "beta": _ppf_beta,
}


def has_fast_ppf(distribution: str) -> bool:
    """Check if a distribution has a fast PPF implementation.

    Args:
        distribution: Name of the scipy.stats distribution

    Returns:
        True if a fast implementation is available
    """
    return distribution in _FAST_PPF_REGISTRY


def fast_ppf(
    distribution: str,
    params: Tuple,
    q: np.ndarray,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
) -> np.ndarray:
    """Compute PPF using optimized implementations where available.

    This function provides significant speedups for common distributions by:
    1. Using direct scipy.special function calls instead of scipy.stats
    2. Avoiding scipy's generic PPF machinery (root-finding, validation, etc.)
    3. Supporting truncation efficiently

    Args:
        distribution: Name of the scipy.stats distribution
        params: Distribution parameters (as returned by scipy.stats.fit)
        q: Quantiles in [0, 1] as numpy array
        lb: Lower truncation bound (None for no truncation)
        ub: Upper truncation bound (None for no truncation)

    Returns:
        PPF values as numpy array

    Example:
        >>> import numpy as np
        >>> from spark_bestfit.fast_ppf import fast_ppf
        >>>
        >>> q = np.array([0.25, 0.5, 0.75])
        >>> # Normal distribution with mean=0, std=1
        >>> fast_ppf("norm", (0, 1), q)
        array([-0.67448975,  0.        ,  0.67448975])
        >>>
        >>> # Truncated normal [0, inf)
        >>> fast_ppf("norm", (0, 1), q, lb=0.0)
        array([0.31863936, 0.67448975, 1.15034938])
    """
    q = np.asarray(q, dtype=np.float64)

    # Handle truncation by mapping quantiles
    if lb is not None or ub is not None:
        q = _map_truncated_quantiles(distribution, params, q, lb, ub)

    # Use fast implementation if available
    if distribution in _FAST_PPF_REGISTRY:
        return _FAST_PPF_REGISTRY[distribution](q, params)

    # Fall back to scipy.stats
    dist = getattr(st, distribution)
    return dist.ppf(q, *params)


def _map_truncated_quantiles(
    distribution: str,
    params: Tuple,
    q: np.ndarray,
    lb: Optional[float],
    ub: Optional[float],
) -> np.ndarray:
    """Map quantiles to the truncated distribution range.

    For a truncated distribution [lb, ub], we need to map q from [0, 1] to
    [CDF(lb), CDF(ub)] before applying PPF.

    q_mapped = CDF(lb) + q * (CDF(ub) - CDF(lb))
    """
    dist = getattr(st, distribution)

    cdf_lb = dist.cdf(lb, *params) if lb is not None and np.isfinite(lb) else 0.0
    cdf_ub = dist.cdf(ub, *params) if ub is not None and np.isfinite(ub) else 1.0

    norm = cdf_ub - cdf_lb
    if norm <= 0:
        # Degenerate case - return lower bound
        return np.full_like(q, cdf_lb)

    return cdf_lb + q * norm


def fast_ppf_batch(
    distributions: list,
    params_list: list,
    q_arrays: list,
    lb_list: Optional[list] = None,
    ub_list: Optional[list] = None,
) -> list:
    """Compute PPF for multiple columns/distributions in batch.

    This is optimized for the copula use case where we need to apply PPF
    to multiple columns simultaneously.

    Args:
        distributions: List of distribution names
        params_list: List of parameter tuples, one per distribution
        q_arrays: List of quantile arrays, one per distribution
        lb_list: List of lower bounds (None entries mean no truncation)
        ub_list: List of upper bounds (None entries mean no truncation)

    Returns:
        List of PPF result arrays
    """
    n = len(distributions)
    results = []

    lb_list = lb_list or [None] * n
    ub_list = ub_list or [None] * n

    for i in range(n):
        result = fast_ppf(
            distributions[i],
            params_list[i],
            q_arrays[i],
            lb=lb_list[i],
            ub=ub_list[i],
        )
        results.append(result)

    return results


__all__ = [
    "fast_ppf",
    "fast_ppf_batch",
    "has_fast_ppf",
]
