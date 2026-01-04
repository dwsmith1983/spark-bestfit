"""Distribution registry and management for scipy.stats distributions."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.stats as st


class DistributionRegistry:
    """Registry for managing scipy.stats continuous distributions.

    Handles filtering of distributions based on exclusions and support constraints.
    All scipy.stats continuous distributions are available by default, with
    sensible exclusions for slow-computing distributions.

    Example:
        >>> registry = DistributionRegistry()
        >>> distributions = registry.get_distributions()
        >>> len(distributions)
        ~100

        >>> # Only non-negative distributions
        >>> pos_distributions = registry.get_distributions(support_at_zero=True)

        >>> # Add custom exclusions
        >>> distributions = registry.get_distributions(
        ...     additional_exclusions=["ncf", "ncx2"]
        ... )
    """

    # Default exclusions: distributions that are very slow or numerically unstable
    DEFAULT_EXCLUSIONS = {
        "levy_stable",  # Extremely slow - MLE doesn't always converge
        "kappa4",  # Extremely slow
        "ncx2",  # Slow - non-central chi-squared
        "ksone",  # Slow - Kolmogorov-Smirnov one-sided
        "ncf",  # Slow - non-central F
        "wald",  # Sometimes numerically unstable
        "mielke",  # Slow
        "exponpow",  # Slow - exponential power
        "studentized_range",  # Very slow - scipy docs recommend approximation
        "gausshyper",  # Very slow - Gauss hypergeometric
        "geninvgauss",  # Can hang - generalized inverse Gaussian
        "genhyperbolic",  # Slow - generalized hyperbolic
        "kstwo",  # Slow - Kolmogorov-Smirnov two-sided
        "kstwobign",  # Slow - KS limit distribution
        "recipinvgauss",  # Can be slow
        "vonmises",  # Can be slow on fitting
        "vonmises_line",  # Can be slow on fitting
        "tukeylambda",  # Extremely slow (~7s) - ill-conditioned optimization
        "nct",  # Very slow (~1.4s) - non-central t distribution
        "dpareto_lognorm",  # Slow (~0.5s) - double Pareto-lognormal
    }

    # Distributions that are included but noticeably slower (~3-5x average).
    # Used for partition weighting when fitting with all distributions.
    SLOW_DISTRIBUTIONS: Set[str] = {
        "powerlognorm",  # ~160ms
        "norminvgauss",  # ~150ms
        "t",  # ~144ms
        "pearson3",  # ~141ms
        "exponweib",  # ~136ms
        "johnsonsb",  # ~133ms
        "jf_skew_t",  # ~125ms
        "fisk",  # ~120ms
        "gengamma",  # ~120ms
        "johnsonsu",  # ~106ms
        "burr",  # ~105ms
        "burr12",  # ~105ms
        "truncweibull_min",  # ~104ms
        "invweibull",  # ~92ms
        "rice",  # ~91ms
        "genexpon",  # ~89ms
    }

    # All scipy continuous distributions
    ALL_DISTRIBUTIONS = [name for name in dir(st) if isinstance(getattr(st, name), st.rv_continuous)]

    def __init__(self, custom_exclusions: Optional[Set[str]] = None):
        """Initialize the distribution registry.

        Args:
            custom_exclusions: Optional set of distribution names to exclude
                             (replaces default exclusions if provided)
        """
        self._excluded = custom_exclusions if custom_exclusions is not None else self.DEFAULT_EXCLUSIONS.copy()

    def get_distributions(
        self,
        support_at_zero: bool = False,
        additional_exclusions: Optional[List[str]] = None,
    ) -> List[str]:
        """Get filtered list of distributions based on criteria.

        Args:
            support_at_zero: If True, only include distributions with support at zero
                           (non-negative distributions)
            additional_exclusions: Additional distribution names to exclude

        Returns:
            List of distribution names meeting the criteria

        Example:
            >>> registry = DistributionRegistry()
            >>> # Get all non-excluded distributions
            >>> dists = registry.get_distributions()

            >>> # Get only non-negative distributions
            >>> pos_dists = registry.get_distributions(support_at_zero=True)

            >>> # Exclude more distributions
            >>> filtered = registry.get_distributions(
            ...     additional_exclusions=["norm", "expon"]
            ... )
        """
        # Start with excluded set
        excluded = self._excluded.copy()

        # Add any additional exclusions
        if additional_exclusions:
            excluded.update(additional_exclusions)

        # Filter out excluded distributions
        distributions = [d for d in self.ALL_DISTRIBUTIONS if d not in excluded]

        # Filter by support if requested
        if support_at_zero:
            distributions = [d for d in distributions if self._has_support_at_zero(d)]

        return distributions

    @staticmethod
    def _has_support_at_zero(dist_name: str) -> bool:
        """Check if a distribution has support at zero (non-negative).

        Args:
            dist_name: Name of the scipy distribution

        Returns:
            True if distribution support starts at 0 or greater
        """
        try:
            dist = getattr(st, dist_name)
            return dist.a >= 0
        except (AttributeError, TypeError):
            # If we can't determine, exclude it to be safe
            return False

    def add_exclusion(self, dist_name: str) -> None:
        """Add a distribution to the exclusion list.

        Args:
            dist_name: Name of the distribution to exclude
        """
        self._excluded.add(dist_name)

    def remove_exclusion(self, dist_name: str) -> None:
        """Remove a distribution from the exclusion list.

        Args:
            dist_name: Name of the distribution to include
        """
        self._excluded.discard(dist_name)

    def get_exclusions(self) -> Set[str]:
        """Get current set of excluded distributions.

        Returns:
            Set of excluded distribution names
        """
        return self._excluded.copy()

    def reset_exclusions(self) -> None:
        """Reset exclusions to default set."""
        self._excluded = self.DEFAULT_EXCLUSIONS.copy()


class DiscreteDistributionRegistry:
    """Registry for managing scipy.stats discrete distributions.

    Unlike continuous distributions, discrete distributions in scipy do not have
    a built-in fit() method. This registry provides parameter configuration
    (initial values, bounds, estimation functions) needed for MLE fitting via
    optimization.

    Example:
        >>> registry = DiscreteDistributionRegistry()
        >>> distributions = registry.get_distributions()
        >>> len(distributions)
        ~15

        >>> # Get parameter config for fitting
        >>> config = registry.get_param_config("poisson")
        >>> initial = config["initial"](data)
        >>> bounds = config["bounds"](data)
    """

    # Default exclusions: distributions that are slow, require special handling,
    # or have complex parameter constraints
    DEFAULT_EXCLUSIONS = {
        "nchypergeom_fisher",  # Very slow - non-central hypergeometric
        "nchypergeom_wallenius",  # Very slow - non-central hypergeometric
        "randint",  # Discrete uniform - trivial, not useful for fitting
        "bernoulli",  # Special case of binomial with n=1
        "poisson_binom",  # Poisson binomial - complex, requires list of probabilities
    }

    # All scipy discrete distributions
    ALL_DISTRIBUTIONS = [name for name in dir(st) if isinstance(getattr(st, name), st.rv_discrete)]

    def __init__(self, custom_exclusions: Optional[Set[str]] = None):
        """Initialize the discrete distribution registry.

        Args:
            custom_exclusions: Optional set of distribution names to exclude
                             (replaces default exclusions if provided)
        """
        self._excluded = custom_exclusions if custom_exclusions is not None else self.DEFAULT_EXCLUSIONS.copy()
        self._param_configs = self._build_param_configs()

    @staticmethod
    def _build_param_configs() -> Dict[str, Dict[str, Any]]:
        """Build parameter configurations for each discrete distribution.

        Each config contains:
            - initial: Callable[[np.ndarray], List[float]] - initial parameter estimates
            - bounds: Callable[[np.ndarray], List[Tuple[float, float]]] - parameter bounds
            - param_names: List[str] - names of parameters

        Returns:
            Dictionary mapping distribution names to their parameter configs
        """
        configs: Dict[str, Dict[str, Any]] = {}

        # Poisson: λ (mu) - rate parameter
        # MLE: λ = mean(data)
        configs["poisson"] = {
            "param_names": ["mu"],
            "initial": lambda data: [max(np.mean(data), 0.1)],
            "bounds": lambda data: [(1e-6, max(np.mean(data) * 10, 100))],
        }

        # Geometric: p - probability of success
        # MLE: p = 1 / mean(data) for scipy's parameterization (starting from 1)
        configs["geom"] = {
            "param_names": ["p"],
            "initial": lambda data: [min(1 / max(np.mean(data), 1), 0.99)],
            "bounds": lambda data: [(1e-6, 1 - 1e-6)],
        }

        # Binomial: n (trials), p (probability)
        # Estimation: n ≈ max(data), p ≈ mean(data) / n
        configs["binom"] = {
            "param_names": ["n", "p"],
            "initial": lambda data: [
                max(int(np.max(data)) + 5, int(np.mean(data) + 3 * np.std(data))),
                np.clip(np.mean(data) / max(np.max(data), 1), 0.01, 0.99),
            ],
            "bounds": lambda data: [
                (max(int(np.max(data)), 1), max(int(np.max(data)) * 3, 100)),
                (1e-6, 1 - 1e-6),
            ],
        }

        # Negative Binomial: n (successes), p (probability)
        # Method of moments: p = mean / (mean + var/mean), n = mean * p / (1-p)
        def nbinom_initial(data: np.ndarray) -> List[float]:
            mean_val = max(np.mean(data), 0.1)
            var_val = max(np.var(data), mean_val + 0.1)  # Ensure overdispersion
            p = np.clip(mean_val / var_val, 0.01, 0.99)
            n = max(mean_val * p / (1 - p), 0.1)
            return [n, p]

        configs["nbinom"] = {
            "param_names": ["n", "p"],
            "initial": nbinom_initial,
            "bounds": lambda data: [(1e-2, 1000), (1e-6, 1 - 1e-6)],
        }

        # Zipf: a - shape parameter (a > 1)
        # Initial estimate based on log-log regression slope
        configs["zipf"] = {
            "param_names": ["a"],
            "initial": lambda data: [2.0],  # Common default
            "bounds": lambda data: [(1.0 + 1e-6, 10.0)],
        }

        # Zipfian: a, n - generalized Zipf
        configs["zipfian"] = {
            "param_names": ["a", "n"],
            "initial": lambda data: [1.5, int(np.max(data)) + 1],
            "bounds": lambda data: [(0.0, 10.0), (int(np.max(data)), int(np.max(data)) * 2 + 10)],
        }

        # Hypergeometric: M (population), n (success states), N (draws)
        # Complex constraints: n <= M, N <= M, max(data) <= min(n, N)
        def hypergeom_initial(data: np.ndarray) -> List[float]:
            max_val = int(np.max(data))
            mean_val = np.mean(data)
            # Rough estimates
            N = max(max_val + 5, int(mean_val * 2))  # draws
            n = max(max_val + 10, N)  # success states
            M = max(n + N, int(n * 2))  # population
            return [M, n, N]

        def hypergeom_bounds(data: np.ndarray) -> List[Tuple[float, float]]:
            max_val = int(np.max(data))
            return [
                (max_val + 10, 10000),  # M (population)
                (max_val + 1, 5000),  # n (success states)
                (max_val + 1, 5000),  # N (draws)
            ]

        configs["hypergeom"] = {
            "param_names": ["M", "n", "N"],
            "initial": hypergeom_initial,
            "bounds": hypergeom_bounds,
        }

        # Beta-Binomial: n, a, b
        configs["betabinom"] = {
            "param_names": ["n", "a", "b"],
            "initial": lambda data: [int(np.max(data)) + 5, 1.0, 1.0],
            "bounds": lambda data: [
                (int(np.max(data)), int(np.max(data)) * 3 + 10),
                (1e-2, 100),
                (1e-2, 100),
            ],
        }

        # Beta-Negative Binomial: n, a, b
        configs["betanbinom"] = {
            "param_names": ["n", "a", "b"],
            "initial": lambda data: [max(np.mean(data), 1), 1.0, 1.0],
            "bounds": lambda data: [(1e-2, 1000), (1e-2, 100), (1e-2, 100)],
        }

        # Boltzmann: lambda, N
        configs["boltzmann"] = {
            "param_names": ["lambda", "N"],
            "initial": lambda data: [1.0, int(np.max(data)) + 1],
            "bounds": lambda data: [(1e-6, 100), (int(np.max(data)) + 1, int(np.max(data)) * 2 + 10)],
        }

        # Discrete Laplacian: a
        configs["dlaplace"] = {
            "param_names": ["a"],
            "initial": lambda data: [0.5],
            "bounds": lambda data: [(1e-6, 10.0)],
        }

        # Logarithmic (Log-Series): p
        configs["logser"] = {
            "param_names": ["p"],
            "initial": lambda data: [0.5],
            "bounds": lambda data: [(1e-6, 1 - 1e-6)],
        }

        # Planck: lambda
        configs["planck"] = {
            "param_names": ["lambda"],
            "initial": lambda data: [1.0],
            "bounds": lambda data: [(1e-6, 100)],
        }

        # Skellam: mu1, mu2 (difference of two Poissons)
        def skellam_initial(data: np.ndarray) -> List[float]:
            mean_val = np.mean(data)
            var_val = np.var(data)
            # mu1 + mu2 = var, mu1 - mu2 = mean
            mu1 = max((var_val + mean_val) / 2, 0.1)
            mu2 = max((var_val - mean_val) / 2, 0.1)
            return [mu1, mu2]

        configs["skellam"] = {
            "param_names": ["mu1", "mu2"],
            "initial": skellam_initial,
            "bounds": lambda data: [(1e-6, 1000), (1e-6, 1000)],
        }

        # Yule-Simon: alpha
        configs["yulesimon"] = {
            "param_names": ["alpha"],
            "initial": lambda data: [2.0],
            "bounds": lambda data: [(1e-6, 20.0)],
        }

        # Non-central hypergeometric (nhypergeom): M, n, N, odds (4 params - complex)
        configs["nhypergeom"] = {
            "param_names": ["M", "n", "r"],
            "initial": lambda data: [int(np.max(data)) * 2 + 20, int(np.max(data)) + 10, int(np.max(data)) + 5],
            "bounds": lambda data: [
                (int(np.max(data)) + 10, 10000),
                (int(np.max(data)) + 1, 5000),
                (1, 5000),
            ],
        }

        return configs

    def get_distributions(
        self,
        additional_exclusions: Optional[List[str]] = None,
    ) -> List[str]:
        """Get filtered list of discrete distributions.

        Only returns distributions that have parameter configurations defined.

        Args:
            additional_exclusions: Additional distribution names to exclude

        Returns:
            List of distribution names that can be fitted
        """
        excluded = self._excluded.copy()
        if additional_exclusions:
            excluded.update(additional_exclusions)

        # Only include distributions we have configs for
        return [d for d in self.ALL_DISTRIBUTIONS if d not in excluded and d in self._param_configs]

    def get_param_config(self, dist_name: str) -> Dict[str, Any]:
        """Get parameter configuration for a distribution.

        Args:
            dist_name: Name of the scipy discrete distribution

        Returns:
            Dictionary with 'param_names', 'initial', and 'bounds' keys

        Raises:
            ValueError: If distribution is not supported
        """
        if dist_name not in self._param_configs:
            raise ValueError(
                f"Distribution '{dist_name}' is not supported. " f"Supported: {list(self._param_configs.keys())}"
            )
        return self._param_configs[dist_name]

    def add_exclusion(self, dist_name: str) -> None:
        """Add a distribution to the exclusion list."""
        self._excluded.add(dist_name)

    def remove_exclusion(self, dist_name: str) -> None:
        """Remove a distribution from the exclusion list."""
        self._excluded.discard(dist_name)

    def get_exclusions(self) -> Set[str]:
        """Get current set of excluded distributions."""
        return self._excluded.copy()

    def reset_exclusions(self) -> None:
        """Reset exclusions to default set."""
        self._excluded = self.DEFAULT_EXCLUSIONS.copy()
