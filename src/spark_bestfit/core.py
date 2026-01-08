"""Core distribution fitting engine for Spark - re-exports for backward compatibility.

This module provides backward-compatible imports for:
- DistributionFitter (continuous distributions)
- DiscreteDistributionFitter (discrete/count data)
- TruncatedFrozenDist (truncated distribution wrapper)
- Default exclusion constants

The actual implementations are in:
- spark_bestfit.continuous_fitter (DistributionFitter)
- spark_bestfit.discrete_fitter (DiscreteDistributionFitter)
- spark_bestfit.truncated (TruncatedFrozenDist)
"""

from typing import List

# Re-export fitter classes for backward compatibility
# (from spark_bestfit.core import DistributionFitter)
from spark_bestfit.continuous_fitter import DistributionFitter
from spark_bestfit.discrete_fitter import DiscreteDistributionFitter
from spark_bestfit.distributions import DiscreteDistributionRegistry, DistributionRegistry

# Re-export TruncatedFrozenDist from truncated module
from spark_bestfit.truncated import TruncatedFrozenDist

# Re-export constants
DEFAULT_EXCLUDED_DISTRIBUTIONS = tuple(DistributionRegistry.DEFAULT_EXCLUSIONS)
DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS = tuple(DiscreteDistributionRegistry.DEFAULT_EXCLUSIONS)


def _interleave_distributions(distributions: List[str]) -> List[str]:
    """Interleave slow distributions among fast ones for better partition balance.

    When Spark repartitions the distribution DataFrame, this ordering ensures
    slow distributions are spread across partitions rather than clustered,
    reducing straggler effects.

    Args:
        distributions: List of distribution names to fit

    Returns:
        Reordered list with slow distributions spread evenly among fast ones

    Example:
        >>> dists = ["norm", "expon", "gamma", "burr", "t", "johnsonsb"]
        >>> # burr, t, johnsonsb are slow - they get spread out:
        >>> _interleave_distributions(dists)
        ['burr', 'norm', 'expon', 't', 'gamma', 'johnsonsb']
    """
    slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
    slow = [d for d in distributions if d in slow_set]
    fast = [d for d in distributions if d not in slow_set]

    if not slow or not fast:
        return distributions

    # Insert slow distributions at even intervals among fast ones
    result: List[str] = []
    slow_interval = max(1, len(fast) // len(slow))
    slow_idx = 0
    for i, d in enumerate(fast):
        if slow_idx < len(slow) and i % slow_interval == 0:
            result.append(slow[slow_idx])
            slow_idx += 1
        result.append(d)
    result.extend(slow[slow_idx:])  # Remaining slow ones at end
    return result


__all__ = [
    "DistributionFitter",
    "DiscreteDistributionFitter",
    "TruncatedFrozenDist",
    "DEFAULT_EXCLUDED_DISTRIBUTIONS",
    "DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS",
    "_interleave_distributions",
]
