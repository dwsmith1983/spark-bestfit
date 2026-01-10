"""Truncated distribution wrapper for scipy frozen distributions.

This module provides analytical truncated moments for common distributions
(norm, expon, uniform) and falls back to Monte Carlo for others.
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

    def _get_dist_name(self) -> str:
        """Get the distribution name from the frozen distribution."""
        return self._dist.dist.name

    def _get_dist_params(self) -> tuple:
        """Extract loc, scale, and shape params from frozen distribution."""
        args = self._dist.args
        kwds = self._dist.kwds
        loc = kwds.get("loc", 0.0)
        scale = kwds.get("scale", 1.0)
        return args, loc, scale

    def mean(self):
        """Compute mean of truncated distribution.

        Uses analytical formulas for norm, expon, and uniform distributions.
        Falls back to Monte Carlo for other distributions.
        """
        dist_name = self._get_dist_name()
        args, loc, scale = self._get_dist_params()
        lb, ub = self._lb, self._ub

        if dist_name == "norm":
            return self._mean_truncated_norm(loc, scale, lb, ub)
        elif dist_name == "expon":
            return self._mean_truncated_expon(loc, scale, lb, ub)
        elif dist_name == "uniform":
            return self._mean_truncated_uniform(loc, scale, lb, ub)
        else:
            # Fall back to Monte Carlo
            samples = self.rvs(size=10000, random_state=42)
            return np.mean(samples)

    def std(self):
        """Compute standard deviation of truncated distribution.

        Uses analytical formulas for norm, expon, and uniform distributions.
        Falls back to Monte Carlo for other distributions.
        """
        dist_name = self._get_dist_name()
        args, loc, scale = self._get_dist_params()
        lb, ub = self._lb, self._ub

        if dist_name == "norm":
            return self._std_truncated_norm(loc, scale, lb, ub)
        elif dist_name == "expon":
            return self._std_truncated_expon(loc, scale, lb, ub)
        elif dist_name == "uniform":
            return self._std_truncated_uniform(loc, scale, lb, ub)
        else:
            # Fall back to Monte Carlo
            samples = self.rvs(size=10000, random_state=42)
            return np.std(samples)

    def _mean_truncated_norm(self, mu: float, sigma: float, lb: float, ub: float) -> float:
        """Analytical mean for truncated normal distribution.

        E[X] = mu + sigma * (phi(alpha) - phi(beta)) / Z
        where alpha = (lb - mu) / sigma, beta = (ub - mu) / sigma
        phi = standard normal PDF, Z = Phi(beta) - Phi(alpha)
        """
        alpha = (lb - mu) / sigma if np.isfinite(lb) else -np.inf
        beta = (ub - mu) / sigma if np.isfinite(ub) else np.inf

        # Standard normal PDF at alpha and beta
        phi_alpha = np.exp(-0.5 * alpha**2) / np.sqrt(2 * np.pi) if np.isfinite(alpha) else 0.0
        phi_beta = np.exp(-0.5 * beta**2) / np.sqrt(2 * np.pi) if np.isfinite(beta) else 0.0

        # Z is already computed as self._norm (CDF(ub) - CDF(lb))
        Z = self._norm
        if Z <= 0:
            return mu

        return mu + sigma * (phi_alpha - phi_beta) / Z

    def _std_truncated_norm(self, mu: float, sigma: float, lb: float, ub: float) -> float:
        """Analytical standard deviation for truncated normal distribution.

        Var[X] = sigma^2 * [1 + (alpha*phi(alpha) - beta*phi(beta))/Z - ((phi(alpha) - phi(beta))/Z)^2]
        """
        alpha = (lb - mu) / sigma if np.isfinite(lb) else -np.inf
        beta = (ub - mu) / sigma if np.isfinite(ub) else np.inf

        # Standard normal PDF at alpha and beta
        phi_alpha = np.exp(-0.5 * alpha**2) / np.sqrt(2 * np.pi) if np.isfinite(alpha) else 0.0
        phi_beta = np.exp(-0.5 * beta**2) / np.sqrt(2 * np.pi) if np.isfinite(beta) else 0.0

        # Handle infinite bounds for alpha*phi(alpha) and beta*phi(beta)
        alpha_phi_alpha = alpha * phi_alpha if np.isfinite(alpha) else 0.0
        beta_phi_beta = beta * phi_beta if np.isfinite(beta) else 0.0

        Z = self._norm
        if Z <= 0:
            return sigma

        term1 = (alpha_phi_alpha - beta_phi_beta) / Z
        term2 = ((phi_alpha - phi_beta) / Z) ** 2

        variance = sigma**2 * (1 + term1 - term2)
        return np.sqrt(max(0, variance))

    def _mean_truncated_expon(self, loc: float, scale: float, lb: float, ub: float) -> float:
        """Analytical mean for truncated exponential distribution.

        For exponential with rate lambda = 1/scale, truncated to [a, b]:
        E[X] = loc + scale * [1 - (b-a)/scale * exp(-(b-a)/scale) / (1 - exp(-(b-a)/scale))]
        when lb >= loc.
        """
        # Effective bounds relative to loc
        a = max(lb - loc, 0) if np.isfinite(lb) else 0.0
        b = ub - loc if np.isfinite(ub) else np.inf

        if not np.isfinite(b):
            # One-sided truncation from below
            # E[X|X > a] = loc + a + scale
            return loc + a + scale

        # Two-sided truncation
        lam = 1.0 / scale
        exp_term = np.exp(-lam * (b - a))

        if exp_term >= 1.0:
            # Degenerate case
            return loc + (a + b) / 2

        denom = 1 - exp_term
        if denom <= 0:
            return loc + (a + b) / 2

        # Mean of truncated exponential
        mean_shifted = (1 / lam) - (b - a) * exp_term / denom
        return loc + a + mean_shifted

    def _std_truncated_expon(self, loc: float, scale: float, lb: float, ub: float) -> float:
        """Analytical standard deviation for truncated exponential distribution."""
        # Effective bounds relative to loc
        a = max(lb - loc, 0) if np.isfinite(lb) else 0.0
        b = ub - loc if np.isfinite(ub) else np.inf

        if not np.isfinite(b):
            # One-sided truncation: Std = scale (memoryless property)
            return scale

        # Two-sided truncation: use variance formula
        lam = 1.0 / scale
        exp_term = np.exp(-lam * (b - a))
        denom = 1 - exp_term

        if denom <= 0:
            return 0.0

        # E[X] and E[X^2] for variance calculation
        delta = b - a
        mean_rel = (1 / lam) - delta * exp_term / denom

        # E[X^2] for truncated exponential
        e_x2 = (2 / lam**2) - (delta**2 + 2 * delta / lam) * exp_term / denom

        variance = e_x2 - mean_rel**2
        return np.sqrt(max(0, variance))

    def _mean_truncated_uniform(self, loc: float, scale: float, lb: float, ub: float) -> float:
        """Analytical mean for truncated uniform distribution.

        For uniform on [loc, loc+scale], truncated to [lb, ub]:
        Mean = (effective_lb + effective_ub) / 2
        """
        # Original support is [loc, loc + scale]
        effective_lb = max(lb, loc) if np.isfinite(lb) else loc
        effective_ub = min(ub, loc + scale) if np.isfinite(ub) else loc + scale

        return (effective_lb + effective_ub) / 2

    def _std_truncated_uniform(self, loc: float, scale: float, lb: float, ub: float) -> float:
        """Analytical standard deviation for truncated uniform distribution.

        Std = (effective_ub - effective_lb) / sqrt(12)
        """
        # Original support is [loc, loc + scale]
        effective_lb = max(lb, loc) if np.isfinite(lb) else loc
        effective_ub = min(ub, loc + scale) if np.isfinite(ub) else loc + scale

        width = effective_ub - effective_lb
        return width / np.sqrt(12)


def create_truncated_dist(frozen_dist, lb: float, ub: float) -> TruncatedFrozenDist:
    """Create a truncated distribution wrapper.

    Convenience function that creates a TruncatedFrozenDist with raise_on_empty=False
    so fitting continues silently when truncation bounds contain no probability mass.

    Args:
        frozen_dist: Frozen scipy.stats distribution
        lb: Lower bound
        ub: Upper bound

    Returns:
        Truncated distribution wrapper with pdf, logpdf, cdf methods
    """
    return TruncatedFrozenDist(frozen_dist, lb, ub, raise_on_empty=False)


__all__ = ["TruncatedFrozenDist", "create_truncated_dist"]
