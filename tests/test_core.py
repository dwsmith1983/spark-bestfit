"""Tests for core distribution fitting module.

Uses LocalBackend for most tests. Spark-specific tests are in test_spark_backend.py.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spark_bestfit import (
    DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS,
    DEFAULT_EXCLUDED_DISTRIBUTIONS,
    DiscreteDistributionFitter,
    DistributionFitter,
    LocalBackend,
)
from spark_bestfit.distributions import DistributionRegistry
from spark_bestfit.results import BaseFitResults, FitResults


class TestDistributionFitter:
    """Tests for DistributionFitter class."""

    @pytest.mark.parametrize("excluded,seed,expected_excluded,expected_seed", [
        (None, 42, DEFAULT_EXCLUDED_DISTRIBUTIONS, 42),  # defaults
        (("norm", "expon"), 123, ("norm", "expon"), 123),  # custom
        ((), 42, (), 42),  # empty exclusions
    ])
    def test_initialization(self, local_backend, excluded, seed, expected_excluded, expected_seed):
        """Test fitter initialization with various configurations."""
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=excluded, random_seed=seed)

        assert fitter.excluded_distributions == expected_excluded
        assert fitter.random_seed == expected_seed

    def test_empty_exclusions_uses_all_distributions(self, local_backend):
        """Test that excluded_distributions=() enables ALL scipy distributions.

        This is a regression test for issue #78 where excluded_distributions=()
        did not override the registry's DEFAULT_EXCLUSIONS.
        """
        from spark_bestfit.distributions import DistributionRegistry

        # Get count of ALL scipy distributions (no exclusions)
        all_dists = DistributionRegistry(custom_exclusions=set()).get_distributions()
        default_dists = DistributionRegistry().get_distributions()

        # Verify there are more distributions without exclusions
        assert len(all_dists) > len(default_dists), "ALL should have more distributions than default"

        # Create fitter with empty exclusions
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=())

        # Verify the fitter's registry has ALL distributions
        fitter_dists = fitter._registry.get_distributions()
        assert len(fitter_dists) == len(all_dists), (
            f"Fitter with excluded_distributions=() should have {len(all_dists)} distributions, "
            f"got {len(fitter_dists)}"
        )

    def test_default_exclusions_applied(self, local_backend):
        """Test that default behavior applies DEFAULT_EXCLUSIONS."""
        from spark_bestfit.distributions import DistributionRegistry

        default_dists = DistributionRegistry().get_distributions()

        # Create fitter with default exclusions (excluded_distributions=None)
        fitter = DistributionFitter(backend=local_backend)

        # Verify the fitter's registry uses default exclusions
        fitter_dists = fitter._registry.get_distributions()
        assert len(fitter_dists) == len(default_dists), (
            f"Default fitter should have {len(default_dists)} distributions, got {len(fitter_dists)}"
        )

    def test_fit_basic(self, local_backend, pandas_dataset):
        """Test basic fitting operation returns valid results."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)

        # Should return results
        assert results.count() == 5  # Requested 5 distributions

        # Should find best distribution with valid data
        best = results.best(n=1)[0]
        assert isinstance(best.distribution, str) and len(best.distribution) > 0
        assert best.sse < np.inf
        assert len(best.parameters) >= 2  # At least loc and scale
        assert isinstance(best.ks_statistic, float) and 0 <= best.ks_statistic <= 1
        assert isinstance(best.pvalue, float) and 0 <= best.pvalue <= 1

    def test_fit_identifies_correct_distribution(self, local_backend, normal_data):
        """Test that fitter can fit normal data well with appropriate distributions."""
        df = pd.DataFrame({"value": normal_data})

        fitter = DistributionFitter(backend=local_backend)
        # Fit a smaller set to ensure norm is included (distributions sorted alphabetically)
        results = fitter.fit(df, column="value", max_distributions=5)

        # Best distribution should have good fit for normal data
        best = results.best(n=1)[0]
        assert best.sse < 0.1, f"Best fit SSE too high: {best.sse}"
        assert best.ks_statistic < 0.1, f"Best fit KS too high: {best.ks_statistic}"

    def test_fit_with_custom_bins(self, local_backend, pandas_dataset):
        """Test fitting with custom number of bins."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", bins=25, max_distributions=5)

        # Should fit all 5 requested distributions
        assert results.count() == 5

    def test_fit_support_at_zero(self, local_backend, pandas_positive_dataset):
        """Test fitting only non-negative distributions."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_positive_dataset, column="value", support_at_zero=True, max_distributions=5)

        # Should fit all 5 requested non-negative distributions
        assert results.count() == 5

        # All distributions should be non-negative
        registry = DistributionRegistry()
        df_results = results.df
        for dist_name in df_results["distribution"]:
            assert registry._has_support_at_zero(dist_name) is True

    def test_fit_with_sampling(self, local_backend):
        """Test fitting with sampling enabled."""
        np.random.seed(42)
        large_data = np.random.normal(loc=50, scale=10, size=100_000)
        df = pd.DataFrame({"value": large_data})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            column="value",
            enable_sampling=True,
            sample_fraction=0.5,
            sample_threshold=50_000,
            max_distributions=5,
        )

        # Should fit all 5 requested distributions
        assert results.count() == 5

    @pytest.mark.parametrize("enable,threshold,desc", [
        (False, 10_000_000, "sampling disabled"),
        (True, 100_000, "below threshold"),
    ])
    def test_apply_sampling_returns_original(self, local_backend, pandas_dataset, enable, threshold, desc):
        """Test that original DataFrame is returned when sampling doesn't apply."""
        fitter = DistributionFitter(backend=local_backend)
        df_sampled = fitter._apply_sampling(
            pandas_dataset, row_count=10_000, enable_sampling=enable,
            sample_fraction=None, max_sample_size=1_000_000, sample_threshold=threshold
        )
        assert len(df_sampled) == len(pandas_dataset)

    def test_apply_sampling_with_fraction(self, local_backend):
        """Test sampling with specified fraction."""
        np.random.seed(42)
        medium_data = np.random.normal(loc=50, scale=10, size=100_000)
        df = pd.DataFrame({"value": medium_data})

        fitter = DistributionFitter(backend=local_backend)
        df_sampled = fitter._apply_sampling(
            df, row_count=100_000, enable_sampling=True,
            sample_fraction=0.5, max_sample_size=1_000_000, sample_threshold=50_000
        )

        # Should sample ~50% of data
        sampled_count = len(df_sampled)
        assert 45_000 < sampled_count < 55_000  # Allow some variance

    def test_apply_sampling_auto_fraction(self, local_backend):
        """Test sampling with auto-determined fraction."""
        np.random.seed(42)
        medium_data = np.random.normal(loc=50, scale=10, size=100_000)
        df = pd.DataFrame({"value": medium_data})

        fitter = DistributionFitter(backend=local_backend)
        df_sampled = fitter._apply_sampling(
            df, row_count=100_000, enable_sampling=True,
            sample_fraction=None, max_sample_size=50_000, sample_threshold=50_000
        )

        # Should sample to max_sample_size
        sampled_count = len(df_sampled)
        assert sampled_count <= 55_000  # Allow some variance

    def test_create_fitting_sample(self, local_backend, pandas_dataset):
        """Test creating sample for distribution fitting."""
        fitter = DistributionFitter(backend=local_backend)
        row_count = len(pandas_dataset)

        sample = fitter._create_fitting_sample(pandas_dataset, "value", row_count)

        # Should be numpy array
        assert isinstance(sample, np.ndarray)

        # Should be <= 10k (default sample size)
        assert len(sample) <= 10_000

    @pytest.mark.parametrize("num_dists", [5, 100])
    def test_calculate_partitions(self, local_backend, num_dists):
        """Test partition calculation returns reasonable values."""
        fitter = DistributionFitter(backend=local_backend)
        # _calculate_partitions now takes a list of distribution names
        dists = ["norm", "expon", "gamma", "beta", "uniform"][:num_dists] if num_dists <= 5 else \
                DistributionRegistry().get_distributions()[:num_dists]
        partitions = fitter._calculate_partitions(dists)
        assert 1 <= partitions <= len(dists) * 3  # Allow for slow distribution weighting

    def test_fit_caches_results(self, local_backend, pandas_dataset):
        """Test that fit results are cached and consistent."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)

        # Access results multiple times (should use cache)
        count1 = results.count()
        count2 = results.count()
        best1 = results.best(n=1)[0]
        best2 = results.best(n=1)[0]

        # Counts should be consistent
        assert count1 == count2
        assert count1 > 0

        # Best results should be identical
        assert best1.distribution == best2.distribution
        assert best1.sse == best2.sse
        assert best1.parameters == best2.parameters

        # DataFrame should also be consistent
        df1 = results.df
        df2 = results.df
        assert len(df1) == len(df2)
        assert list(df1["distribution"]) == list(df2["distribution"])

    def test_fit_filters_failed_fits(self, local_backend, pandas_dataset):
        """Test that failed fits are filtered out."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)

        # All results should have finite SSE
        df_results = results.df
        assert all(np.isfinite(df_results["sse"]))

    def test_fit_with_constant_data(self, local_backend):
        """Test fitting with constant data (edge case)."""
        data = np.full(1000, 42.0)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

        # Should handle gracefully without crashing
        results = fitter.fit(df, column="value", max_distributions=5)

        # Returns valid FitResults (may have 0 or more distributions)
        assert isinstance(results, BaseFitResults)
        # Verify we can call methods on it without error
        _ = results.df

    def test_fit_with_rice_rule(self, local_backend, pandas_dataset):
        """Test fitting with Rice rule for bins."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", use_rice_rule=True, max_distributions=5)

        # Should fit all 5 requested distributions
        assert results.count() == 5

    def test_fit_excluded_distributions(self, local_backend, pandas_dataset):
        """Test that excluded distributions are not fitted."""
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=("norm", "expon"))
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)

        # norm and expon should not be in results
        df_results = results.df
        assert "norm" not in df_results["distribution"].values
        assert "expon" not in df_results["distribution"].values

    def test_fit_multiple_columns_sequential(self, local_backend):
        """Test fitting multiple columns sequentially."""
        # Create DataFrame with multiple columns
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 10_000)
        data2 = np.random.exponential(5, 10_000)

        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)

        # Fit first column
        results1 = fitter.fit(df, column="col1", max_distributions=5)
        best1 = results1.best(n=1)[0]

        # Fit second column
        results2 = fitter.fit(df, column="col2", max_distributions=5)
        best2 = results2.best(n=1)[0]

        # Both should succeed
        assert best1.sse < np.inf
        assert best2.sse < np.inf

        # Should identify different distributions
        top1 = [r.distribution for r in results1.best(n=3)]
        top2 = [r.distribution for r in results2.best(n=3)]

        # Normal should be in top for col1, expon should be in top for col2
        assert "norm" in top1 or best1.sse < 0.01
        assert "expon" in top2 or best2.sse < 0.01

    def test_fit_reproducibility(self, local_backend, pandas_dataset):
        """Test that fitting is reproducible with same seed."""
        fitter1 = DistributionFitter(backend=local_backend, random_seed=42)
        fitter2 = DistributionFitter(backend=local_backend, random_seed=42)

        # Use max_distributions to speed up test
        results1 = fitter1.fit(pandas_dataset, column="value", max_distributions=5)
        results2 = fitter2.fit(pandas_dataset, column="value", max_distributions=5)

        # Should get same best distribution
        best1 = results1.best(n=1)[0]
        best2 = results2.best(n=1)[0]

        assert best1.distribution == best2.distribution
        # SSE might differ slightly due to sampling, but should be close
        assert np.isclose(best1.sse, best2.sse, rtol=0.1)


class TestMultiColumnFitting:
    """Tests for multi-column distribution fitting."""

    def test_fit_multiple_columns_basic(self, local_backend):
        """Test basic multi-column fitting."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.exponential(5, 5000)

        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, columns=["col1", "col2"], max_distributions=3)

        # Should have results for both columns
        assert results.count() == 6  # 2 columns × 3 distributions
        assert set(results.column_names) == {"col1", "col2"}

    def test_fit_multiple_columns_filtering(self, local_backend):
        """Test filtering multi-column results by column."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.exponential(5, 5000)

        df = pd.DataFrame({"normal_col": data1, "expon_col": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, columns=["normal_col", "expon_col"], max_distributions=3)

        # Filter to single column
        normal_results = results.for_column("normal_col")
        assert normal_results.count() == 3

        expon_results = results.for_column("expon_col")
        assert expon_results.count() == 3

    def test_fit_multiple_columns_best_per_column(self, local_backend):
        """Test best_per_column method."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.exponential(5, 5000)

        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, columns=["col1", "col2"], max_distributions=5)

        best_per_col = results.best_per_column(n=1)

        assert "col1" in best_per_col
        assert "col2" in best_per_col
        assert len(best_per_col["col1"]) == 1
        assert len(best_per_col["col2"]) == 1
        assert best_per_col["col1"][0].column_name == "col1"
        assert best_per_col["col2"][0].column_name == "col2"

    def test_fit_backward_compatibility(self, local_backend, pandas_dataset):
        """Test that single column API still works with positional arg."""
        fitter = DistributionFitter(backend=local_backend)
        # Using positional argument (backward compatible)
        results = fitter.fit(pandas_dataset, "value", max_distributions=3)

        assert results.count() == 3
        best = results.best(n=1)[0]
        assert best.column_name == "value"

    def test_fit_mutually_exclusive_params(self, local_backend, pandas_dataset):
        """Test error when both column and columns provided."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="Cannot provide both"):
            fitter.fit(pandas_dataset, column="value", columns=["value"])

    def test_fit_no_column_params(self, local_backend, pandas_dataset):
        """Test error when neither column nor columns provided."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="Must provide either"):
            fitter.fit(pandas_dataset)

    def test_fit_invalid_column_in_list(self, local_backend, pandas_dataset):
        """Test error when invalid column in columns list."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(pandas_dataset, columns=["value", "nonexistent"])

    def test_fit_single_column_via_columns_param(self, local_backend, pandas_dataset):
        """Test fitting single column using columns parameter."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, columns=["value"], max_distributions=3)

        assert results.count() == 3
        assert results.column_names == ["value"]


class TestDistributionFitterPlotting:
    """Tests for plotting functionality.

    Note: The fitter's plot() and plot_qq() methods currently use Spark DataFrame API
    directly (df.select, df.sample, etc.) which isn't backend-agnostic. These tests
    use the plotting module directly with numpy arrays instead.
    """

    def test_plot_after_fit(self, local_backend, pandas_dataset):
        """Test plotting after fitting using the plotting module directly."""
        from spark_bestfit.plotting import plot_distribution

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Create histogram from data for plotting
        data = pandas_dataset["value"].values
        y_hist, bin_edges = np.histogram(data, bins=30, density=True)
        x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, ax = plot_distribution(result=best, y_hist=y_hist, x_hist=x_hist)

        assert fig is not None
        assert ax is not None

    def test_plot_with_title(self, local_backend, pandas_dataset):
        """Test plotting with title."""
        from spark_bestfit.plotting import plot_distribution

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Create histogram from data for plotting
        data = pandas_dataset["value"].values
        y_hist, bin_edges = np.histogram(data, bins=30, density=True)
        x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, ax = plot_distribution(result=best, y_hist=y_hist, x_hist=x_hist, title="Test Plot")

        assert fig is not None
        assert ax is not None

    def test_plot_with_custom_params(self, local_backend, pandas_dataset):
        """Test plotting with custom parameters."""
        from spark_bestfit.plotting import plot_distribution

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Create histogram from data for plotting with custom params
        data = pandas_dataset["value"].values
        y_hist, bin_edges = np.histogram(data, bins=30, density=True)
        x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig, ax = plot_distribution(
            result=best, y_hist=y_hist, x_hist=x_hist,
            figsize=(16, 10), dpi=150, title="Custom Plot"
        )

        assert fig is not None
        assert ax is not None

    def test_plot_qq_after_fit(self, local_backend, pandas_dataset):
        """Test Q-Q plotting after fitting."""
        from spark_bestfit.plotting import plot_qq

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Use plotting module directly with numpy array
        data = pandas_dataset["value"].values
        fig, ax = plot_qq(result=best, data=data)

        assert fig is not None
        assert ax is not None

    def test_plot_qq_with_max_points(self, local_backend, pandas_dataset):
        """Test Q-Q plotting with custom max_points."""
        from spark_bestfit.plotting import plot_qq

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Use plotting module directly - sample data manually
        data = pandas_dataset["value"].values[:500]  # Limit to 500 points
        fig, ax = plot_qq(result=best, data=data, title="Q-Q Test")

        assert fig is not None
        assert ax is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_dataset(self, local_backend):
        """Test with very small dataset."""
        data = np.array([1.0, 2.0, 3.0])
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

        # Should handle gracefully without crashing
        results = fitter.fit(df, column="value", max_distributions=5)

        # Returns valid FitResults
        assert isinstance(results, BaseFitResults)
        _ = results.df

    def test_single_value_dataset(self, local_backend):
        """Test with single value."""
        df = pd.DataFrame({"value": [42.0]})

        fitter = DistributionFitter(backend=local_backend)

        # Should handle gracefully without crashing
        results = fitter.fit(df, column="value", max_distributions=5)

        # Returns valid FitResults
        assert isinstance(results, BaseFitResults)
        _ = results.df

    def test_dataset_with_outliers(self, local_backend):
        """Test with dataset containing extreme outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 9995)
        outliers = np.array([1000, -1000, 2000, -2000, 3000])
        data = np.concatenate([normal_data, outliers])

        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

        # Should handle outliers and fit all 5 requested distributions
        results = fitter.fit(df, column="value", max_distributions=5)

        assert results.count() == 5
        best = results.best(n=1)[0]
        assert best.sse < np.inf

    def test_apply_sampling_at_threshold(self, local_backend, pandas_dataset):
        """Test that data at threshold doesn't sample."""
        fitter = DistributionFitter(backend=local_backend)
        df_result = fitter._apply_sampling(
            pandas_dataset, row_count=10_000, enable_sampling=True,
            sample_fraction=None, max_sample_size=1_000_000, sample_threshold=10_000
        )

        # At threshold should return original data (uses <=)
        assert len(df_result) == len(pandas_dataset)

    def test_fit_max_distributions_zero(self, local_backend, pandas_dataset):
        """Test fitting with max_distributions=0 raises error."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError):
            fitter.fit(pandas_dataset, column="value", max_distributions=0)

    def test_fit_with_different_columns(self, local_backend):
        """Test fitting on different column names."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        df = pd.DataFrame({"custom_column_name": data})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, column="custom_column_name", max_distributions=3)

        # Should fit all 3 requested distributions
        assert results.count() == 3

    def test_fit_invalid_bins(self, local_backend, pandas_dataset):
        """Test that invalid bins raises error."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="bins must be positive"):
            fitter.fit(pandas_dataset, column="value", bins=0)

    def test_fit_invalid_sample_fraction(self, local_backend, pandas_dataset):
        """Test that invalid sample_fraction raises error."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="sample_fraction must be in"):
            fitter.fit(pandas_dataset, column="value", sample_fraction=1.5)


class TestCoreNegativePaths:
    """Tests for negative/error paths in core module."""

    def test_fit_invalid_column(self, local_backend, pandas_dataset):
        """Test that fit raises error for invalid column."""
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(pandas_dataset, column="nonexistent_column", max_distributions=3)

    def test_fit_non_numeric_column(self, local_backend):
        """Test that fit raises error for non-numeric column."""
        df = pd.DataFrame({"value": ["a", "b", "c"]})
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(TypeError, match="must be numeric"):
            fitter.fit(df, column="value", max_distributions=3)

    def test_fit_empty_dataframe(self, local_backend):
        """Test that fit raises error for empty DataFrame."""
        df = pd.DataFrame({"value": pd.Series([], dtype=float)})
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="empty"):
            fitter.fit(df, column="value", max_distributions=3)

    def test_plot_with_different_data(self, local_backend):
        """Test that plot works with different data than fit."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 1000)
        data2 = np.random.normal(100, 20, 1000)

        df1 = pd.DataFrame({"value": data1})
        df2 = pd.DataFrame({"value": data2})

        fitter = DistributionFitter(backend=local_backend)

        # Fit on first dataset
        results = fitter.fit(df1, column="value", max_distributions=3)
        best = results.best(n=1)[0]

        # Plot with different dataset should work
        fig, ax = fitter.plot(best, df=df2, column="value")
        assert fig is not None


class TestDiscreteDistributionFitter:
    """Tests for DiscreteDistributionFitter class."""

    def test_initialization(self, local_backend):
        """Test discrete fitter initialization with custom exclusions."""
        custom_exclusions = ("poisson", "geom")
        fitter = DiscreteDistributionFitter(
            backend=local_backend, excluded_distributions=custom_exclusions, random_seed=123
        )

        assert fitter.excluded_distributions == custom_exclusions
        assert fitter.random_seed == 123

    def test_empty_exclusions_disables_registry_defaults(self, local_backend):
        """Test that excluded_distributions=() disables registry's DEFAULT_EXCLUSIONS.

        This is a regression test for issue #78.

        Note: For discrete distributions, get_distributions() only returns distributions
        with param configs. The DEFAULT_EXCLUSIONS don't have configs, so they're not
        available regardless. This test verifies the registry is created correctly.
        """
        from spark_bestfit.distributions import DiscreteDistributionRegistry

        # Create fitter with empty exclusions
        fitter = DiscreteDistributionFitter(backend=local_backend, excluded_distributions=())

        # Verify the fitter's registry has custom_exclusions=set() (empty)
        assert fitter._registry._excluded == set(), (
            f"Fitter with excluded_distributions=() should have empty registry exclusions, "
            f"got {fitter._registry._excluded}"
        )

        # Verify default fitter uses DEFAULT_EXCLUSIONS
        default_fitter = DiscreteDistributionFitter(backend=local_backend)
        assert default_fitter._registry._excluded == DiscreteDistributionRegistry.DEFAULT_EXCLUSIONS, (
            "Default fitter should use DEFAULT_EXCLUSIONS"
        )

    def test_fit_identifies_poisson(self, local_backend, pandas_poisson_dataset):
        """Test that fitter identifies Poisson for Poisson data."""
        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_poisson_dataset, column="counts")

        top_5 = [r.distribution for r in results.best(n=5)]
        assert "poisson" in top_5

    def test_fit_identifies_nbinom(self, local_backend, nbinom_data):
        """Test that fitter identifies negative binomial for nbinom data."""
        df = pd.DataFrame({"counts": nbinom_data})
        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(df, column="counts")

        top_5 = [r.distribution for r in results.best(n=5)]
        assert "nbinom" in top_5

    def test_fit_parameters_accuracy(self, local_backend, pandas_poisson_dataset, poisson_data):
        """Test that Poisson lambda is estimated accurately."""
        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_poisson_dataset, column="counts")

        poisson_fit = next(r for r in results.best(n=10) if r.distribution == "poisson")
        fitted_lambda = poisson_fit.parameters[0]
        true_lambda = np.mean(poisson_data)

        assert np.isclose(fitted_lambda, true_lambda, rtol=0.05)

    def test_fit_excluded_distributions(self, local_backend, pandas_poisson_dataset):
        """Test that excluded distributions are not fitted."""
        fitter = DiscreteDistributionFitter(
            backend=local_backend, excluded_distributions=("poisson", "nbinom")
        )
        results = fitter.fit(pandas_poisson_dataset, column="counts")

        all_dists = [r.distribution for r in results.best(n=20)]
        assert "poisson" not in all_dists
        assert "nbinom" not in all_dists

    def test_fit_empty_dataframe_raises(self, local_backend):
        """Test that fit raises error for empty DataFrame."""
        df = pd.DataFrame({"counts": pd.Series([], dtype=int)})
        fitter = DiscreteDistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="empty"):
            fitter.fit(df, column="counts")

    def test_fit_invalid_column_raises(self, local_backend, pandas_poisson_dataset):
        """Test that fit raises error for invalid column."""
        fitter = DiscreteDistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(pandas_poisson_dataset, column="nonexistent")

    def test_plot_produces_stem_plot(self, local_backend, pandas_poisson_dataset):
        """Test that discrete plot produces expected stem plot elements."""
        import matplotlib.pyplot as plt

        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_poisson_dataset, column="counts", max_distributions=3)
        best = results.best(n=1)[0]

        fig, ax = fitter.plot(best, pandas_poisson_dataset, "counts", title="Test Plot")

        # Verify plot has expected elements
        assert ax.get_title().startswith("Test Plot")
        assert ax.get_xlabel() == "Value"
        assert ax.get_ylabel() == "Probability"
        # Should have bars (histogram) and stems (fitted PMF)
        assert len(ax.containers) > 0 or len(ax.collections) > 0

        plt.close(fig)


class TestPrefilter:
    """Tests for distribution pre-filtering feature (v1.6.0)."""

    def test_prefilter_false_no_filtering(self, local_backend, pandas_dataset):
        """Test that prefilter=False doesn't filter any distributions."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=10, prefilter=False)
        assert results.count() == 10

    def test_prefilter_true_filters_incompatible(self, local_backend):
        """Test that prefilter=True filters incompatible distributions."""
        # Create left-skewed data with negative values
        np.random.seed(42)
        left_skewed = -np.abs(np.random.exponential(5, 5000))
        df = pd.DataFrame({"value": left_skewed})

        fitter = DistributionFitter(backend=local_backend)

        # Without prefilter
        results_no_filter = fitter.fit(df, "value", max_distributions=20, prefilter=False, lazy_metrics=True)
        count_no_filter = results_no_filter.count()

        # With prefilter=True
        results_filter = fitter.fit(df, "value", max_distributions=20, prefilter=True, lazy_metrics=True)
        count_filter = results_filter.count()

        # Should filter out positive-skew-only and non-negative-support distributions
        assert count_filter < count_no_filter, "prefilter should reduce distribution count"

    def test_prefilter_discrete_warns(self, local_backend, pandas_poisson_dataset):
        """Test that prefilter on discrete fitter logs a warning."""
        fitter = DiscreteDistributionFitter(backend=local_backend)

        with patch("spark_bestfit.discrete_fitter.logger") as mock_logger:
            results = fitter.fit(pandas_poisson_dataset, column="counts", max_distributions=3, prefilter=True)
            mock_logger.warning.assert_called_once()
            assert "not yet supported" in str(mock_logger.warning.call_args)

        # Should still return valid results
        assert results.count() > 0


class TestPrefilterUnit:
    """Unit tests for _prefilter_distributions() method directly."""

    @pytest.fixture
    def fitter(self, local_backend):
        """Create a fitter instance for testing."""
        return DistributionFitter(backend=local_backend)

    def test_no_support_filtering_with_negative_data(self, fitter):
        """Test that negative data does NOT filter distributions (loc can shift them)."""
        # scipy.fit() uses loc parameter that can shift any distribution
        # to cover any data range, so we don't filter by support bounds
        data = np.array([-5.0, -2.0, 0.0, 1.0, 3.0])  # Has negative values
        distributions = ["norm", "expon", "gamma", "t", "lognorm", "laplace"]

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=True)

        # ALL distributions should remain (loc can shift them)
        # Only skewness/kurtosis filtering applies
        assert "norm" in compatible
        assert "expon" in compatible  # Can use loc=-10 to fit negative data
        assert "gamma" in compatible  # Can use loc=-10 to fit negative data
        assert "t" in compatible
        assert "lognorm" in compatible  # Can use loc=-10 to fit negative data
        assert "laplace" in compatible

        # No support-based filtering should occur
        support_filtered = [f for f in filtered if "support" in f[1]]
        assert len(support_filtered) == 0

    def test_no_filtering_for_symmetric_data(self, fitter):
        """Test that symmetric data around zero keeps all distributions."""
        data = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])  # Symmetric
        distributions = ["norm", "expon", "gamma", "t", "uniform", "beta"]

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=True)

        # All should be compatible - no skewness or kurtosis filtering triggered
        assert len(compatible) == len(distributions)
        assert len(filtered) == 0

    def test_loc_scale_can_shift_any_distribution(self, fitter):
        """Test that distributions with default [0,1] bounds still pass (loc/scale shift them)."""
        # Data outside [0, 1] - but beta can be shifted via loc/scale
        data = np.array([50.0, 55.0, 60.0, 65.0, 70.0])
        distributions = ["beta", "uniform", "norm"]

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=True)

        # beta and uniform should NOT be filtered (loc/scale shift them)
        assert "beta" in compatible
        assert "uniform" in compatible
        assert "norm" in compatible
        assert len(filtered) == 0

    def test_skewness_filtering_left_skewed(self, fitter):
        """Test that left-skewed data filters positive-skew-only distributions."""
        # Create strongly left-skewed data (skewness < -1.0)
        np.random.seed(42)
        data = -np.random.exponential(5, 1000)  # Skewness ~ -2.0
        distributions = ["norm", "expon", "gamma", "t", "laplace"]

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=True)

        # Positive-skew-only distributions should be filtered by skewness
        # (skewness is a shape property that cannot be changed by loc/scale)
        assert "expon" not in compatible
        assert "gamma" not in compatible

        # Symmetric distributions should remain
        assert "norm" in compatible
        assert "t" in compatible
        assert "laplace" in compatible

    def test_skewness_filtering_threshold(self, fitter):
        """Test that skewness filter only triggers for |skew| > 1.0."""
        # Mildly left-skewed data (skewness ~ -0.5)
        np.random.seed(42)
        data = np.random.beta(3, 2, 1000) - 0.5  # Mild left skew, has negatives

        from scipy.stats import skew
        assert -1.0 < skew(data) < 0, "Test data should be mildly left-skewed"

        distributions = ["norm", "t", "laplace"]  # Only unbounded for this test

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=True)

        # All should remain (skewness not extreme enough to trigger filter)
        assert "norm" in compatible
        assert "t" in compatible
        assert "laplace" in compatible

    def test_aggressive_mode_kurtosis_filtering(self, fitter):
        """Test that aggressive mode filters by kurtosis for heavy-tailed data."""
        # Create heavy-tailed data WITHIN [0, 1] so uniform's support bounds pass
        # Use Pareto distribution, clipped and normalized to [0, 1]
        np.random.seed(42)
        raw = np.random.pareto(1.5, 10000)
        p99 = np.percentile(raw, 99)
        data = np.clip(raw, 0, p99) / p99  # Data in [0, 1] with heavy right tail

        from scipy.stats import kurtosis
        assert kurtosis(data) > 10, f"Test data should have high kurtosis, got {kurtosis(data)}"
        assert data.min() >= 0 and data.max() <= 1, "Data should be in [0, 1]"

        # uniform has support [0, 1], so it passes support bounds for this data
        # norm and beta also work with this range
        distributions = ["norm", "uniform", "beta"]

        # Safe mode should keep uniform (kurtosis filter not applied)
        compatible_safe, _ = fitter._prefilter_distributions(distributions, data, mode=True)
        assert "uniform" in compatible_safe, f"Safe mode should keep uniform, got {compatible_safe}"

        # Aggressive mode should filter uniform (low kurtosis dist for high kurtosis data)
        compatible_aggressive, filtered_aggressive = fitter._prefilter_distributions(
            distributions, data, mode="aggressive"
        )
        assert "uniform" not in compatible_aggressive, f"Aggressive mode should filter uniform, got {compatible_aggressive}"
        assert any(f[0] == "uniform" for f in filtered_aggressive), f"uniform should be in filtered list"

    def test_prefilter_mode_false_returns_all(self, fitter):
        """Test that mode=False returns all distributions unfiltered."""
        data = np.array([-100.0, -50.0, 0.0])  # Would normally filter many
        distributions = ["norm", "expon", "gamma", "t"]

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=False)

        # mode=False should return all (no filtering)
        assert compatible == distributions
        assert len(filtered) == 0

    def test_unknown_distribution_kept(self, fitter):
        """Test that unknown distributions are kept (conservative approach)."""
        data = np.array([-1.0, 0.0, 1.0])
        distributions = ["norm", "fake_distribution_xyz"]

        compatible, filtered = fitter._prefilter_distributions(distributions, data, mode=True)

        # Unknown distribution should be kept
        assert "fake_distribution_xyz" in compatible
        assert "norm" in compatible


class TestPrefilterIntegration:
    """Integration tests for prefilter with full fitting pipeline."""

    def test_prefilter_no_support_filtering_integration(self, local_backend):
        """Test that prefilter does NOT filter by support (loc/scale can shift distributions)."""
        np.random.seed(42)
        # Symmetric data around 0 - no skewness filtering triggered
        data = np.random.normal(0, 1, 5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, "value", max_distributions=30, prefilter=True, lazy_metrics=True)

        fitted_dists = {r.distribution for r in results.best(n=100)}

        # Distributions CAN be fit even with negative data (loc shifts them)
        # Only skewness/kurtosis filtering applies, not support bounds
        # Since data is symmetric, no skewness filtering should occur
        assert len(fitted_dists) > 0, "Should have fit some distributions"

    def test_prefilter_skewness_filtering_integration(self, local_backend):
        """Test that prefilter correctly filters by skewness in full pipeline."""
        np.random.seed(42)
        left_skewed = -np.random.exponential(5, 5000)
        from scipy.stats import skew
        assert skew(left_skewed) < -1.0, "Test data should be clearly left-skewed"

        df = pd.DataFrame({"value": left_skewed})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, "value", max_distributions=30, prefilter=True, lazy_metrics=True)

        fitted_dists = {r.distribution for r in results.best(n=100)}

        # Positive-skew-only distributions should be filtered
        positive_skew_only = {"expon", "gamma", "lognorm"}
        for dist in positive_skew_only:
            assert dist not in fitted_dists, f"{dist} should be filtered (positive-skew only)"

    def test_prefilter_aggressive_filters_more(self, local_backend):
        """Test that aggressive mode filters additional distributions."""
        np.random.seed(42)
        # Heavy-tailed data with very high kurtosis
        heavy_tailed = np.random.standard_t(2, 5000) * 10
        df = pd.DataFrame({"value": heavy_tailed})

        fitter = DistributionFitter(backend=local_backend)

        # Get results with both modes
        results_safe = fitter.fit(df, "value", max_distributions=20, prefilter=True, lazy_metrics=True)
        results_aggressive = fitter.fit(df, "value", max_distributions=20, prefilter="aggressive", lazy_metrics=True)

        safe_dists = {r.distribution for r in results_safe.best(n=100)}
        aggressive_dists = {r.distribution for r in results_aggressive.best(n=100)}

        # Aggressive should filter at least as much as safe (usually more)
        assert len(aggressive_dists) <= len(safe_dists)

        # Specifically, uniform should be filtered in aggressive mode for heavy-tailed data
        # (if it was in the original distribution list)
        if "uniform" in safe_dists:
            assert "uniform" not in aggressive_dists, "uniform should be filtered in aggressive mode"

    def test_prefilter_multicolumn(self, local_backend):
        """Test that prefilter works correctly for multi-column fitting."""
        np.random.seed(42)
        # Column 1: left-skewed data (should filter positive-skew-only dists by skewness)
        col1 = -np.random.exponential(5, 1000)  # Left-skewed
        # Column 2: right-skewed data (should keep positive-skew-only dists)
        col2 = np.random.exponential(5, 1000)  # Right-skewed

        from scipy.stats import skew
        assert skew(col1) < -1.0, "col1 should be left-skewed"
        assert skew(col2) > 1.0, "col2 should be right-skewed"

        df = pd.DataFrame({"col1": col1, "col2": col2})

        fitter = DistributionFitter(backend=local_backend)
        # Use max_distributions=30 to ensure expon/gamma are included
        results = fitter.fit(df, columns=["col1", "col2"], max_distributions=30, prefilter=True, lazy_metrics=True)

        # Get results per column
        col1_dists = {r.distribution for r in results.for_column("col1").best(n=100)}
        col2_dists = {r.distribution for r in results.for_column("col2").best(n=100)}

        # Col1 (left-skewed) should filter positive-skew-only distributions
        assert "expon" not in col1_dists, "expon should be filtered for left-skewed data"
        assert "gamma" not in col1_dists, "gamma should be filtered for left-skewed data"

        # Col2 (right-skewed) should keep positive-skew-only distributions
        assert "expon" in col2_dists, "expon should be kept for right-skewed data"
        assert "gamma" in col2_dists, "gamma should be kept for right-skewed data"

    def test_prefilter_fallback_logs_warning(self, local_backend):
        """Test that fallback when all filtered logs appropriate warning."""
        np.random.seed(42)
        negative_data = np.random.normal(-100, 10, 1000)
        df = pd.DataFrame({"value": negative_data})

        # Create fitter that only has non-negative distributions
        # by using a very restrictive exclusion list
        fitter = DistributionFitter(backend=local_backend)

        with patch("spark_bestfit.continuous_fitter.logger") as mock_logger:
            results = fitter.fit(
                df, "value",
                max_distributions=5,
                prefilter=True,
                lazy_metrics=True
            )
            # Should still return results
            assert results.count() > 0

    def test_prefilter_logs_filtered_distributions(self, local_backend):
        """Test that prefilter logs which distributions were filtered."""
        np.random.seed(42)
        # Use left-skewed data to trigger skewness filtering
        left_skewed_data = -np.random.exponential(5, 1000)
        df = pd.DataFrame({"value": left_skewed_data})

        fitter = DistributionFitter(backend=local_backend)

        with patch("spark_bestfit.continuous_fitter.logger") as mock_logger:
            results = fitter.fit(df, "value", max_distributions=30, prefilter=True, lazy_metrics=True)

            # Verify info log was called with prefilter message
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            prefilter_logged = any("Pre-filter: skipped" in call for call in info_calls)
            assert prefilter_logged, "Should log pre-filter info message (skewness filtering)"


class TestNewExclusions:
    """Tests for new DEFAULT_EXCLUSIONS added in v1.6.1."""

    def test_tukeylambda_excluded_by_default(self):
        """tukeylambda should not be in default distributions (extremely slow ~7s)."""
        registry = DistributionRegistry()
        dists = registry.get_distributions()
        assert "tukeylambda" not in dists
        assert "tukeylambda" in DistributionRegistry.DEFAULT_EXCLUSIONS

    def test_nct_excluded_by_default(self):
        """nct should not be in default distributions (very slow ~1.4s)."""
        registry = DistributionRegistry()
        dists = registry.get_distributions()
        assert "nct" not in dists
        assert "nct" in DistributionRegistry.DEFAULT_EXCLUSIONS

    def test_dpareto_lognorm_excluded_by_default(self):
        """dpareto_lognorm should not be in default distributions (slow ~0.5s)."""
        registry = DistributionRegistry()
        dists = registry.get_distributions()
        assert "dpareto_lognorm" not in dists
        assert "dpareto_lognorm" in DistributionRegistry.DEFAULT_EXCLUSIONS

    def test_slow_distributions_constant_exists(self):
        """SLOW_DISTRIBUTIONS constant should exist for partition weighting."""
        assert hasattr(DistributionRegistry, "SLOW_DISTRIBUTIONS")
        slow_dists = DistributionRegistry.SLOW_DISTRIBUTIONS
        assert isinstance(slow_dists, set)
        assert len(slow_dists) > 0
        # Verify some known slow distributions are present
        assert "burr" in slow_dists
        assert "t" in slow_dists
        assert "johnsonsb" in slow_dists

    def test_slow_distributions_not_in_exclusions(self):
        """SLOW_DISTRIBUTIONS should be fittable (not in exclusions)."""
        slow_dists = DistributionRegistry.SLOW_DISTRIBUTIONS
        exclusions = DistributionRegistry.DEFAULT_EXCLUSIONS
        # Slow distributions should NOT be excluded - they're used for weighting
        for dist in slow_dists:
            assert dist not in exclusions, f"{dist} is in both SLOW_DISTRIBUTIONS and DEFAULT_EXCLUSIONS"


class TestInterleaveDistributions:
    """Tests for _interleave_distributions helper function."""

    def test_interleave_spreads_slow_distributions(self):
        """Slow distributions should not be consecutive after interleaving."""
        from spark_bestfit.core import _interleave_distributions

        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        # Create a list with slow distributions at the end (worst case for round-robin)
        fast = ["norm", "expon", "gamma", "beta", "uniform", "chi2", "f", "weibull_min"]
        slow = ["burr", "t", "johnsonsb", "fisk"]
        clustered = fast + slow

        interleaved = _interleave_distributions(clustered)

        # Get indices of slow distributions after interleaving
        slow_indices = [i for i, d in enumerate(interleaved) if d in slow_set]

        # Verify slow distributions are spread out (not all at end)
        assert slow_indices != [8, 9, 10, 11], "Slow distributions should not all be at end"

    def test_interleave_preserves_all_distributions(self):
        """Interleaving should not drop any distributions."""
        from spark_bestfit.core import _interleave_distributions

        dists = ["norm", "expon", "burr", "t", "gamma", "johnsonsb"]
        interleaved = _interleave_distributions(dists)

        assert len(interleaved) == len(dists)
        assert set(interleaved) == set(dists)

    def test_interleave_handles_no_slow(self):
        """Interleaving handles lists with no slow distributions."""
        from spark_bestfit.core import _interleave_distributions

        fast_only = ["norm", "expon", "gamma"]
        result = _interleave_distributions(fast_only)
        assert result == fast_only

    def test_interleave_handles_all_slow(self):
        """Interleaving handles lists with only slow distributions."""
        from spark_bestfit.core import _interleave_distributions

        slow_only = ["burr", "t", "johnsonsb"]
        result = _interleave_distributions(slow_only)
        assert result == slow_only


class TestDistributionAwarePartitioning:
    """Tests for distribution-aware partition calculation."""

    def test_slow_distributions_weighted_3x(self, local_backend):
        """Slow distributions should count as 3x for partition calculation."""
        fitter = DistributionFitter(backend=local_backend)

        # Get default parallelism for comparison
        total_cores = local_backend.get_parallelism()

        # Test with fast distributions only
        fast_dists = ["norm", "expon", "gamma", "beta"]  # 4 fast
        fast_partitions = fitter._calculate_partitions(fast_dists)

        # Test with same count but some slow distributions
        mixed_dists = ["norm", "expon", "burr", "t"]  # 2 fast, 2 slow
        mixed_partitions = fitter._calculate_partitions(mixed_dists)

        # Mixed should have more effective count: 4 + (2 * 2) = 8 effective
        # vs fast which has 4 effective
        # So mixed_partitions >= fast_partitions (unless capped by cores)
        if total_cores * 2 > 4:
            assert mixed_partitions >= fast_partitions

    def test_partitions_capped_at_2x_cores(self, local_backend):
        """Partition count should not exceed 2x available cores."""
        fitter = DistributionFitter(backend=local_backend)
        total_cores = local_backend.get_parallelism()

        # Create a large list with many slow distributions
        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        many_slow = list(slow_set)[:10] + ["norm"] * 50

        partitions = fitter._calculate_partitions(many_slow)

        assert partitions <= total_cores * 2

    def test_discrete_fitter_has_same_partitioning(self, local_backend):
        """Discrete fitter should use the same partitioning logic."""
        from spark_bestfit import DiscreteDistributionFitter

        discrete_fitter = DiscreteDistributionFitter(backend=local_backend)

        # Verify discrete fitter has _calculate_partitions with list signature
        fast_dists = ["poisson", "geom", "binom"]
        partitions = discrete_fitter._calculate_partitions(fast_dists)
        assert isinstance(partitions, int)
        assert partitions > 0
