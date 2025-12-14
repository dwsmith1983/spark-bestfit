"""Distribution registry and management for scipy.stats distributions."""

from typing import List, Optional, Set

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
        "exonpow",  # Slow - exponential power
        "studentized_range",  # Very slow - scipy docs recommend approximation
        "gausshyper",  # Very slow - Gauss hypergeometric
        "geninvgauss",  # Can hang - generalized inverse Gaussian
        "genhyperbolic",  # Slow - generalized hyperbolic
        "kstwo",  # Slow - Kolmogorov-Smirnov two-sided
        "kstwobign",  # Slow - KS limit distribution
        "recipinvgauss",  # Can be slow
        "vonmises",  # Can be slow on fitting
        "vonmises_line",  # Can be slow on fitting
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
