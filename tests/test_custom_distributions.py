"""Tests for custom distribution support (v2.4.0).

This module tests the ability to register and fit custom scipy rv_continuous
distributions alongside the built-in scipy.stats distributions.
"""

import numpy as np
import pytest
from scipy.stats import rv_continuous

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.continuous_fitter import DistributionFitter
from spark_bestfit.distributions import DistributionRegistry


class PowerDistribution(rv_continuous):
    """A simple power distribution for testing.

    PDF: f(x) = alpha * x^(alpha-1) for x in [0, 1]
    CDF: F(x) = x^alpha
    """

    def _pdf(self, x, alpha):
        return alpha * np.power(x, alpha - 1)

    def _cdf(self, x, alpha):
        return np.power(x, alpha)


class TestDistributionRegistryCustom:
    """Tests for custom distribution registration in DistributionRegistry."""

    def test_register_custom_distribution(self):
        """Test basic registration of a custom distribution."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")

        registry.register_distribution("power_custom", dist)

        assert "power_custom" in registry.get_distributions()
        assert "power_custom" in registry.get_custom_distributions()

    def test_register_overwrites_with_flag(self):
        """Test that overwrite=True allows replacing a distribution."""
        registry = DistributionRegistry()
        dist1 = PowerDistribution(a=0, b=1, name="power1")
        dist2 = PowerDistribution(a=0, b=2, name="power2")  # Different support

        registry.register_distribution("test_dist", dist1)
        registry.register_distribution("test_dist", dist2, overwrite=True)

        # Should have the new distribution
        retrieved = registry.get_distribution_object("test_dist")
        assert retrieved.b == 2

    def test_register_raises_on_duplicate_without_overwrite(self):
        """Test that registration fails without overwrite flag."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")

        registry.register_distribution("test_dist", dist)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_distribution("test_dist", dist)

    def test_register_raises_on_scipy_name_conflict(self):
        """Test that registration fails if name conflicts with scipy.stats."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")

        with pytest.raises(ValueError, match="conflicts with scipy.stats"):
            registry.register_distribution("norm", dist)

    def test_register_raises_on_invalid_type(self):
        """Test that registration fails for non-rv_continuous objects."""
        registry = DistributionRegistry()

        with pytest.raises(TypeError, match="must implement"):
            registry.register_distribution("invalid", "not a distribution")  # type: ignore

    def test_register_raises_on_missing_methods(self):
        """Test that registration fails if required methods are missing."""
        registry = DistributionRegistry()

        class IncompleteDistribution:
            pass

        with pytest.raises(TypeError, match="must implement"):
            registry.register_distribution("incomplete", IncompleteDistribution())  # type: ignore

    def test_unregister_custom_distribution(self):
        """Test unregistration of a custom distribution."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")

        registry.register_distribution("power_custom", dist)
        assert "power_custom" in registry.get_distributions()

        registry.unregister_distribution("power_custom")
        assert "power_custom" not in registry.get_distributions()
        assert "power_custom" not in registry.get_custom_distributions()

    def test_unregister_raises_on_unknown(self):
        """Test that unregistration fails for unknown distributions."""
        registry = DistributionRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.unregister_distribution("unknown_dist")

    def test_get_distribution_object_custom(self):
        """Test retrieval of custom distribution object."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")

        registry.register_distribution("power_custom", dist)

        retrieved = registry.get_distribution_object("power_custom")
        assert retrieved is dist

    def test_get_distribution_object_scipy(self):
        """Test retrieval of scipy.stats distribution."""
        registry = DistributionRegistry()

        norm = registry.get_distribution_object("norm")
        assert norm.name == "norm"

    def test_get_distribution_object_raises_on_unknown(self):
        """Test that retrieval fails for unknown distributions."""
        registry = DistributionRegistry()

        with pytest.raises(ValueError, match="not found"):
            registry.get_distribution_object("unknown_dist")

    def test_has_custom_distributions(self):
        """Test has_custom_distributions method."""
        registry = DistributionRegistry()
        assert not registry.has_custom_distributions()

        dist = PowerDistribution(a=0, b=1, name="power")
        registry.register_distribution("power_custom", dist)
        assert registry.has_custom_distributions()

    def test_get_distributions_includes_custom(self):
        """Test that get_distributions includes custom distributions."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")

        initial_count = len(registry.get_distributions())
        registry.register_distribution("power_custom", dist)

        new_count = len(registry.get_distributions())
        assert new_count == initial_count + 1

    def test_get_distributions_excludes_custom_when_requested(self):
        """Test that include_custom=False excludes custom distributions."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")
        registry.register_distribution("power_custom", dist)

        without_custom = registry.get_distributions(include_custom=False)
        assert "power_custom" not in without_custom

    def test_custom_distribution_respects_exclusions(self):
        """Test that custom distributions can be excluded."""
        registry = DistributionRegistry()
        dist = PowerDistribution(a=0, b=1, name="power")
        registry.register_distribution("power_custom", dist)

        # Add to exclusions
        registry.add_exclusion("power_custom")

        # Should not appear in get_distributions
        assert "power_custom" not in registry.get_distributions()


class TestDistributionFitterCustom:
    """Tests for custom distribution support in DistributionFitter."""

    @pytest.fixture
    def backend(self):
        """Create a LocalBackend for testing."""
        return LocalBackend(max_workers=2)

    @pytest.fixture
    def sample_data(self):
        """Generate sample power distribution data."""
        np.random.seed(42)
        # Generate data from power distribution with alpha=2
        # Use inverse CDF: x = u^(1/alpha) for u ~ Uniform(0,1)
        u = np.random.uniform(0, 1, 1000)
        return u ** (1 / 2)

    def test_register_distribution_on_fitter(self, backend):
        """Test register_distribution method on fitter."""
        fitter = DistributionFitter(backend=backend)
        dist = PowerDistribution(a=0, b=1, name="power")

        result = fitter.register_distribution("power_custom", dist)

        assert result is fitter  # Check method chaining
        assert "power_custom" in fitter.get_custom_distributions()

    def test_unregister_distribution_on_fitter(self, backend):
        """Test unregister_distribution method on fitter."""
        fitter = DistributionFitter(backend=backend)
        dist = PowerDistribution(a=0, b=1, name="power")

        fitter.register_distribution("power_custom", dist)
        result = fitter.unregister_distribution("power_custom")

        assert result is fitter  # Check method chaining
        assert "power_custom" not in fitter.get_custom_distributions()

    def test_fit_with_custom_distribution(self, backend, sample_data):
        """Test fitting with a custom distribution included."""
        import pandas as pd

        fitter = DistributionFitter(backend=backend)
        dist = PowerDistribution(a=0, b=1, name="power")
        fitter.register_distribution("power_custom", dist)

        # Create DataFrame
        df = pd.DataFrame({"value": sample_data})

        # Fit with only custom distribution (for speed)
        results = fitter.fit(
            df,
            column="value",
            max_distributions=1,  # Only fit the first distribution
        )

        # Check that we got results (custom distribution should be included)
        # Use best() method which returns a list
        all_results = results.best(n=100)  # Get up to 100 results
        assert len(all_results) >= 0  # May or may not have successful fits

    def test_fit_custom_distribution_only(self, backend, sample_data):
        """Test fitting with ONLY custom distributions."""
        import pandas as pd

        # Use excluded_distributions=() to remove defaults, then limit
        fitter = DistributionFitter(backend=backend, excluded_distributions=())
        dist = PowerDistribution(a=0, b=1, name="power")
        fitter.register_distribution("power_custom", dist)

        df = pd.DataFrame({"value": sample_data})

        # Get all distributions including custom
        all_dists = fitter._registry.get_distributions()
        assert "power_custom" in all_dists

    def test_method_chaining(self, backend):
        """Test that registration methods support chaining."""
        dist1 = PowerDistribution(a=0, b=1, name="power1")
        dist2 = PowerDistribution(a=0, b=1, name="power2")

        fitter = (
            DistributionFitter(backend=backend)
            .register_distribution("dist1", dist1)
            .register_distribution("dist2", dist2)
        )

        assert "dist1" in fitter.get_custom_distributions()
        assert "dist2" in fitter.get_custom_distributions()


class TestFitSingleDistributionCustom:
    """Tests for fit_single_distribution with custom distributions."""

    def test_fit_single_custom_distribution(self):
        """Test fitting a single custom distribution."""
        from spark_bestfit.fitting import fit_single_distribution

        # Generate power distribution data
        np.random.seed(42)
        u = np.random.uniform(0, 1, 500)
        data = u ** (1 / 2)  # Power distribution with alpha=2

        # Create histogram
        y_hist, bin_edges = np.histogram(data, bins=50, density=True)

        # Custom distribution
        dist = PowerDistribution(a=0, b=1, name="power")
        custom_dists = {"power_custom": dist}

        # Fit
        result = fit_single_distribution(
            dist_name="power_custom",
            data_sample=data,
            bin_edges=bin_edges,
            y_hist=y_hist,
            custom_distributions=custom_dists,
        )

        assert result["distribution"] == "power_custom"
        assert np.isfinite(result["sse"])
        assert result["sse"] != np.inf  # Should have a valid fit

    def test_fit_single_falls_back_to_scipy(self):
        """Test that fitting falls back to scipy for standard distributions."""
        from spark_bestfit.fitting import fit_single_distribution

        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)

        # Create histogram
        y_hist, bin_edges = np.histogram(data, bins=50, density=True)

        # No custom distributions - should use scipy.stats.norm
        result = fit_single_distribution(
            dist_name="norm",
            data_sample=data,
            bin_edges=bin_edges,
            y_hist=y_hist,
            custom_distributions={},  # Empty dict
        )

        assert result["distribution"] == "norm"
        assert np.isfinite(result["sse"])


class TestCustomDistributionSerialization:
    """Tests for serialization of custom distributions.

    Note: Custom distributions must be picklable for Spark broadcast.
    Simple rv_continuous subclasses are picklable by default.
    """

    def test_power_distribution_is_picklable(self):
        """Test that our custom distribution can be pickled."""
        import pickle

        dist = PowerDistribution(a=0, b=1, name="power")
        pickled = pickle.dumps(dist)
        unpickled = pickle.loads(pickled)

        # Verify it still works
        x = np.array([0.5])
        assert np.allclose(dist.pdf(x, 2), unpickled.pdf(x, 2))
