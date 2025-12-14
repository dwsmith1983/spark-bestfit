"""Tests for distributions module."""

import scipy.stats as st

from spark_bestfit.distributions import DistributionRegistry


class TestDistributionRegistry:
    """Tests for DistributionRegistry class."""

    def test_initialization_default(self):
        """Test initialization with default exclusions."""
        registry = DistributionRegistry()

        assert "levy_stable" in registry.get_exclusions()
        assert "kappa4" in registry.get_exclusions()
        assert "studentized_range" in registry.get_exclusions()
        assert "gausshyper" in registry.get_exclusions()
        assert len(registry.get_exclusions()) == 17

    def test_initialization_custom_exclusions(self):
        """Test initialization with custom exclusions."""
        custom = {"norm", "expon"}
        registry = DistributionRegistry(custom_exclusions=custom)

        assert registry.get_exclusions() == custom
        assert "levy_stable" not in registry.get_exclusions()

    def test_all_distributions_class_constant(self):
        """Test ALL_DISTRIBUTIONS class constant contains scipy distributions."""
        all_dists = DistributionRegistry.ALL_DISTRIBUTIONS

        # Should have ~100 distributions
        assert len(all_dists) > 80
        assert "norm" in all_dists
        assert "expon" in all_dists
        assert "gamma" in all_dists

    def test_get_distributions_no_filtering(self):
        """Test getting distributions with default exclusions."""
        registry = DistributionRegistry()
        dists = registry.get_distributions()

        # Should exclude default slow distributions
        assert "levy_stable" not in dists
        assert "kappa4" not in dists

        # Should include common distributions
        assert "norm" in dists
        assert "expon" in dists
        assert "gamma" in dists

    def test_get_distributions_support_at_zero(self):
        """Test filtering for non-negative distributions."""
        registry = DistributionRegistry()
        pos_dists = registry.get_distributions(support_at_zero=True)

        # Should include non-negative distributions
        assert "expon" in pos_dists
        assert "gamma" in pos_dists
        assert "chi2" in pos_dists

        # Should exclude distributions with negative support
        assert "norm" not in pos_dists  # norm has support (-inf, inf)

        # Verify support check
        for dist_name in pos_dists:
            dist = getattr(st, dist_name)
            assert dist.a >= 0, f"{dist_name} should have support at 0 or greater"

    def test_get_distributions_additional_exclusions(self):
        """Test adding additional exclusions."""
        registry = DistributionRegistry()
        dists = registry.get_distributions(additional_exclusions=["norm", "expon"])

        # Should exclude additionally specified distributions
        assert "norm" not in dists
        assert "expon" not in dists

        # Should still exclude default exclusions
        assert "levy_stable" not in dists

        # Should include others
        assert "gamma" in dists

    def test_get_distributions_combined_filtering(self):
        """Test combining support_at_zero and additional_exclusions."""
        registry = DistributionRegistry()
        dists = registry.get_distributions(support_at_zero=True, additional_exclusions=["expon", "gamma"])

        # Should exclude specified distributions
        assert "expon" not in dists
        assert "gamma" not in dists

        # Should only include non-negative distributions
        assert "norm" not in dists

        # Should include other non-negative distributions
        assert "chi2" in dists

    def test_add_exclusion(self):
        """Test adding an exclusion dynamically."""
        registry = DistributionRegistry()
        initial_count = len(registry.get_exclusions())

        registry.add_exclusion("norm")

        assert "norm" in registry.get_exclusions()
        assert len(registry.get_exclusions()) == initial_count + 1

        # Should affect get_distributions
        dists = registry.get_distributions()
        assert "norm" not in dists

    def test_remove_exclusion(self):
        """Test removing an exclusion."""
        registry = DistributionRegistry()
        initial_exclusions = registry.get_exclusions().copy()

        registry.remove_exclusion("levy_stable")

        assert "levy_stable" not in registry.get_exclusions()
        assert len(registry.get_exclusions()) == len(initial_exclusions) - 1

        # Should affect get_distributions
        dists = registry.get_distributions()
        assert "levy_stable" in dists

    def test_remove_nonexistent_exclusion(self):
        """Test removing an exclusion that doesn't exist (no error)."""
        registry = DistributionRegistry()
        initial_exclusions = registry.get_exclusions().copy()

        registry.remove_exclusion("nonexistent_distribution")

        # Should not change exclusions
        assert registry.get_exclusions() == initial_exclusions

    def test_reset_exclusions(self):
        """Test resetting exclusions to defaults."""
        registry = DistributionRegistry()

        # Modify exclusions
        registry.add_exclusion("norm")
        registry.remove_exclusion("levy_stable")

        # Reset
        registry.reset_exclusions()

        # Should have default exclusions
        assert "levy_stable" in registry.get_exclusions()
        assert "norm" not in registry.get_exclusions()
        assert len(registry.get_exclusions()) == 17

    def test_has_support_at_zero_positive(self):
        """Test _has_support_at_zero for positive distributions."""
        registry = DistributionRegistry()

        # Test known non-negative distributions
        assert registry._has_support_at_zero("expon") is True
        assert registry._has_support_at_zero("gamma") is True
        assert registry._has_support_at_zero("chi2") is True

    def test_has_support_at_zero_negative(self):
        """Test _has_support_at_zero for distributions with negative support."""
        registry = DistributionRegistry()

        # Test distributions with negative support
        assert registry._has_support_at_zero("norm") is False
        assert registry._has_support_at_zero("cauchy") is False

    def test_has_support_at_zero_invalid_distribution(self):
        """Test _has_support_at_zero with invalid distribution name."""
        registry = DistributionRegistry()

        # Should return False for invalid distribution
        assert registry._has_support_at_zero("invalid_dist_name") is False

    def test_immutability_of_exclusions(self):
        """Test that get_exclusions returns a copy (immutable)."""
        registry = DistributionRegistry()
        exclusions1 = registry.get_exclusions()
        exclusions2 = registry.get_exclusions()

        # Modify one copy
        exclusions1.add("new_exclusion")

        # Should not affect the other copy or registry
        assert "new_exclusion" not in exclusions2
        assert "new_exclusion" not in registry.get_exclusions()
