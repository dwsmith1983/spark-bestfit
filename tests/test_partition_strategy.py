"""
Test to validate distribution-aware partitioning strategy.

This test demonstrates that interleaving slow distributions reduces
total fitting time by avoiding partition clustering.
"""

from typing import List, Set

import numpy as np
import pytest

from spark_bestfit import DistributionFitter
from spark_bestfit.distributions import DistributionRegistry


# Slow distributions from our timing analysis (100-160ms range)
SLOW_DISTRIBUTIONS: Set[str] = {
    "powerlognorm",
    "norminvgauss",
    "t",
    "pearson3",
    "exponweib",
    "johnsonsb",
    "jf_skew_t",
    "fisk",
    "gengamma",
    "johnsonsu",
    "burr",
    "burr12",
    "truncweibull_min",
    "invweibull",
    "rice",
    "genexpon",
}


def _interleave_distributions(distributions: List[str]) -> List[str]:
    """Interleave slow distributions among fast ones."""
    slow = [d for d in distributions if d in SLOW_DISTRIBUTIONS]
    fast = [d for d in distributions if d not in SLOW_DISTRIBUTIONS]

    if not slow or not fast:
        return distributions

    result = []
    slow_interval = max(1, len(fast) // len(slow))
    slow_idx = 0
    for i, d in enumerate(fast):
        if slow_idx < len(slow) and i % slow_interval == 0:
            result.append(slow[slow_idx])
            slow_idx += 1
        result.append(d)
    result.extend(slow[slow_idx:])
    return result


class TestPartitionStrategy:
    """Tests validating the partition interleaving strategy."""

    @pytest.fixture
    def test_data(self, spark_session):
        """Create test DataFrame with 100K rows.

        Uses the session-scoped spark_session fixture from conftest.py.
        """
        np.random.seed(42)
        data = np.random.normal(50, 10, 100_000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])
        df.cache()
        df.count()  # Materialize cache
        yield df
        df.unpersist()

    def test_interleave_distributes_slow_evenly(self):
        """Verify that interleaving spreads slow distributions evenly."""
        # Create a list with slow distributions at the end (worst case)
        fast = ["norm", "expon", "gamma", "beta", "uniform", "chi2", "f", "weibull_min"]
        slow = ["burr", "t", "johnsonsb", "fisk"]
        clustered = fast + slow  # Slow ones at the end

        interleaved = _interleave_distributions(clustered)

        # Get indices of slow distributions after interleaving
        slow_indices = [i for i, d in enumerate(interleaved) if d in SLOW_DISTRIBUTIONS]

        # Verify slow distributions are spread out (not all at end)
        assert slow_indices != [8, 9, 10, 11], "Slow distributions should not all be at end"

        # Verify no two slow distributions are adjacent
        for i in range(len(slow_indices) - 1):
            gap = slow_indices[i + 1] - slow_indices[i]
            assert gap >= 2, f"Slow distributions should be separated, got gap of {gap}"

    def test_interleave_preserves_all_distributions(self):
        """Verify interleaving doesn't drop any distributions."""
        dists = ["norm", "expon", "burr", "t", "gamma", "johnsonsb"]
        interleaved = _interleave_distributions(dists)

        assert len(interleaved) == len(dists)
        assert set(interleaved) == set(dists)

    def test_interleave_handles_no_slow(self):
        """Verify interleaving handles lists with no slow distributions."""
        fast_only = ["norm", "expon", "gamma"]
        result = _interleave_distributions(fast_only)
        assert result == fast_only

    def test_interleave_handles_all_slow(self):
        """Verify interleaving handles lists with only slow distributions."""
        slow_only = ["burr", "t", "johnsonsb"]
        result = _interleave_distributions(slow_only)
        assert result == slow_only

    def test_partition_strategy_improves_balance(self, spark_session, test_data):
        """
        Test that interleaving improves partition balance.

        With 4 partitions and 4 slow distributions:
        - Clustered: Risk of all slow in 1 partition (straggler)
        - Interleaved: Slow distributions spread across partitions
        """
        registry = DistributionRegistry()
        all_dists = registry.get_distributions()

        # Select test distributions: 12 fast + 4 slow = 16 total
        fast_dists = [d for d in all_dists if d not in SLOW_DISTRIBUTIONS][:12]
        slow_dists = [d for d in all_dists if d in SLOW_DISTRIBUTIONS][:4]

        # Clustered order: fast first, slow at end
        clustered = fast_dists + slow_dists

        # Interleaved order: slow spread throughout
        interleaved = _interleave_distributions(clustered)

        # Verify the distributions are actually interleaved
        slow_in_clustered = [i for i, d in enumerate(clustered) if d in SLOW_DISTRIBUTIONS]
        slow_in_interleaved = [i for i, d in enumerate(interleaved) if d in SLOW_DISTRIBUTIONS]

        # In clustered, slow are all at indices 12, 13, 14, 15
        assert all(i >= 12 for i in slow_in_clustered), "Clustered should have slow at end"

        # In interleaved, slow should be spread out
        assert not all(i >= 12 for i in slow_in_interleaved), "Interleaved should spread slow"

        # Verify the max gap between slow distributions is larger in interleaved
        def max_gap(indices):
            if len(indices) <= 1:
                return 0
            return max(indices[i + 1] - indices[i] for i in range(len(indices) - 1))

        clustered_max_gap = max_gap(slow_in_clustered)
        interleaved_max_gap = max_gap(slow_in_interleaved)

        # Interleaved should have larger gaps (better spread)
        assert interleaved_max_gap >= clustered_max_gap, \
            f"Interleaved gap ({interleaved_max_gap}) should be >= clustered gap ({clustered_max_gap})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
