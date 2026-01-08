"""Truncated distribution wrapper for scipy frozen distributions.

This module is intentionally dependency-free (only numpy) to avoid
circular imports when used by core.py, results.py, and fitting.py.
"""

import numpy as np


class TruncatedFrozenDist:
    """Wrapper for frozen scipy distributions with truncation bounds.

    Implements truncation for arbitrary scipy.stats frozen distributions
    using CDF inversion for sampling and proper normalization for PDF/CDF.

    This is needed because scipy.stats.truncate() only works with the new
    distribution infrastructure (scipy 1.14+), not with traditional rv_frozen objects.

    Args:
        frozen_dist: Frozen scipy.stats distribution
        lb: Lower bound (-np.inf for no lower bound)
        ub: Upper bound (np.inf for no upper bound)
        raise_on_empty: If True, raise ValueError when truncation has no probability mass.
            If False, methods return zeros/empty results silently. Default True.

    Example:
        >>> from scipy import stats
        >>> from spark_bestfit import TruncatedFrozenDist
        >>> # Create a normal distribution truncated to [0, inf)
        >>> frozen = stats.norm(loc=0, scale=1)
        >>> truncated = TruncatedFrozenDist(frozen, lb=0, ub=np.inf)
        >>> truncated.pdf(0.5)  # Evaluate PDF at x=0.5
    """

    def __init__(self, frozen_dist, lb: float, ub: float, *, raise_on_empty: bool = True):
        """Initialize truncated distribution."""
        self._dist = frozen_dist
        self._lb = lb
        self._ub = ub
        self._raise_on_empty = raise_on_empty

        # Pre-compute normalization constant
        self._cdf_lb = frozen_dist.cdf(lb) if np.isfinite(lb) else 0.0
        self._cdf_ub = frozen_dist.cdf(ub) if np.isfinite(ub) else 1.0
        self._norm = self._cdf_ub - self._cdf_lb

        if self._norm <= 0 and raise_on_empty:
            raise ValueError(f"Invalid truncation: no probability mass in [{lb}, {ub}]")

    @property
    def bounds(self) -> tuple:
        """Return (lower_bound, upper_bound) tuple."""
        return (self._lb, self._ub)

    def pdf(self, x):
        """Evaluate probability density function.

        Returns 0 for values outside the truncation bounds.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        if self._norm <= 0:
            return result

        # Only compute PDF for values within bounds
        mask = (x >= self._lb) & (x <= self._ub)
        if np.any(mask):
            result[mask] = self._dist.pdf(x[mask]) / self._norm

        return result

    def logpdf(self, x):
        """Evaluate log probability density function.

        Returns -inf for values outside the truncation bounds.
        """
        x = np.asarray(x)
        result = np.full_like(x, -np.inf, dtype=float)

        if self._norm <= 0:
            return result

        # Only compute logPDF for values within bounds
        mask = (x >= self._lb) & (x <= self._ub)
        if np.any(mask):
            result[mask] = self._dist.logpdf(x[mask]) - np.log(self._norm)

        return result

    def cdf(self, x):
        """Evaluate cumulative distribution function.

        Returns 0 for x < lower_bound, 1 for x > upper_bound.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # Below lower bound: 0
        below = x < self._lb
        result[below] = 0.0

        # Above upper bound: 1
        above = x > self._ub
        result[above] = 1.0

        # Within bounds: scaled CDF
        between = ~below & ~above
        if np.any(between) and self._norm > 0:
            result[between] = (self._dist.cdf(x[between]) - self._cdf_lb) / self._norm

        return result

    def ppf(self, q):
        """Evaluate percent point function (inverse CDF).

        Args:
            q: Quantile(s) in [0, 1]

        Returns:
            Value(s) at the given quantile(s) within the truncated distribution
        """
        q = np.asarray(q)

        # Map quantile to the truncated range
        q_mapped = self._cdf_lb + q * self._norm

        return self._dist.ppf(q_mapped)

    def rvs(self, size=1, random_state=None):
        """Generate random samples using inverse CDF method.

        Args:
            size: Number of samples to generate
            random_state: Random seed or numpy Generator for reproducibility

        Returns:
            Array of random samples from the truncated distribution
        """
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0, 1, size=size)
        return self.ppf(u)

    def mean(self):
        """Approximate mean of truncated distribution via sampling."""
        samples = self.rvs(size=10000, random_state=42)
        return np.mean(samples)

    def std(self):
        """Approximate standard deviation of truncated distribution via sampling."""
        samples = self.rvs(size=10000, random_state=42)
        return np.std(samples)


__all__ = ["TruncatedFrozenDist"]
