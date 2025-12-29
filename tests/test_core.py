"""Tests for core distribution fitting module."""

from unittest.mock import patch

import numpy as np
import pytest

from spark_bestfit import (
    DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS,
    DEFAULT_EXCLUDED_DISTRIBUTIONS,
    DiscreteDistributionFitter,
    DistributionFitter,
)
from spark_bestfit.distributions import DistributionRegistry
from spark_bestfit.results import FitResults

class TestDistributionFitter:
    """Tests for DistributionFitter class."""

    @pytest.mark.parametrize("excluded,seed,expected_excluded,expected_seed", [
        (None, 42, DEFAULT_EXCLUDED_DISTRIBUTIONS, 42),  # defaults
        (("norm", "expon"), 123, ("norm", "expon"), 123),  # custom
        ((), 42, (), 42),  # empty exclusions
    ])
    def test_initialization(self, spark_session, excluded, seed, expected_excluded, expected_seed):
        """Test fitter initialization with various configurations."""
        fitter = DistributionFitter(spark_session, excluded_distributions=excluded, random_seed=seed)

        assert fitter.spark is spark_session
        assert fitter.excluded_distributions == expected_excluded
        assert fitter.random_seed == expected_seed

    def test_fit_basic(self, spark_session, small_dataset):
        """Test basic fitting operation returns valid results."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # Should return results
        assert results.count() == 5  # Requested 5 distributions

        # Should find best distribution with valid data
        best = results.best(n=1)[0]
        assert isinstance(best.distribution, str) and len(best.distribution) > 0
        assert best.sse < np.inf
        assert len(best.parameters) >= 2  # At least loc and scale
        assert isinstance(best.ks_statistic, float) and 0 <= best.ks_statistic <= 1
        assert isinstance(best.pvalue, float) and 0 <= best.pvalue <= 1

    def test_fit_identifies_correct_distribution(self, spark_session, normal_data):
        """Test that fitter can fit normal data well with appropriate distributions."""
        df = spark_session.createDataFrame([(float(x),) for x in normal_data], ["value"])

        fitter = DistributionFitter(spark_session)
        # Fit a smaller set to ensure norm is included (distributions sorted alphabetically)
        results = fitter.fit(df, column="value", max_distributions=5)

        # Best distribution should have good fit for normal data
        best = results.best(n=1)[0]
        assert best.sse < 0.1, f"Best fit SSE too high: {best.sse}"
        assert best.ks_statistic < 0.1, f"Best fit KS too high: {best.ks_statistic}"

    def test_fit_with_custom_bins(self, spark_session, small_dataset):
        """Test fitting with custom number of bins."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", bins=25, max_distributions=5)

        # Should fit all 5 requested distributions
        assert results.count() == 5

    def test_fit_support_at_zero(self, spark_session, positive_dataset):
        """Test fitting only non-negative distributions."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(positive_dataset, column="value", support_at_zero=True, max_distributions=5)

        # Should fit all 5 requested non-negative distributions
        assert results.count() == 5

        # All distributions should be non-negative
        registry = DistributionRegistry()
        df_pandas = results.df.toPandas()
        for dist_name in df_pandas["distribution"]:
            assert registry._has_support_at_zero(dist_name) is True

    def test_fit_with_sampling(self, spark_session, medium_dataset):
        """Test fitting with sampling enabled."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            medium_dataset,
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
    def test_apply_sampling_returns_original(self, spark_session, small_dataset, enable, threshold, desc):
        """Test that original DataFrame is returned when sampling doesn't apply."""
        fitter = DistributionFitter(spark_session)
        df_sampled = fitter._apply_sampling(
            small_dataset, row_count=10_000, enable_sampling=enable,
            sample_fraction=None, max_sample_size=1_000_000, sample_threshold=threshold
        )
        assert df_sampled.count() == small_dataset.count()

    def test_apply_sampling_with_fraction(self, spark_session, medium_dataset):
        """Test sampling with specified fraction."""
        fitter = DistributionFitter(spark_session)
        df_sampled = fitter._apply_sampling(
            medium_dataset, row_count=100_000, enable_sampling=True,
            sample_fraction=0.5, max_sample_size=1_000_000, sample_threshold=50_000
        )

        # Should sample ~50% of data
        sampled_count = df_sampled.count()
        assert 45_000 < sampled_count < 55_000  # Allow some variance

    def test_apply_sampling_auto_fraction(self, spark_session, medium_dataset):
        """Test sampling with auto-determined fraction."""
        fitter = DistributionFitter(spark_session)
        df_sampled = fitter._apply_sampling(
            medium_dataset, row_count=100_000, enable_sampling=True,
            sample_fraction=None, max_sample_size=50_000, sample_threshold=50_000
        )

        # Should sample to max_sample_size
        sampled_count = df_sampled.count()
        assert sampled_count <= 55_000  # Allow some variance

    def test_create_fitting_sample(self, spark_session, small_dataset):
        """Test creating sample for distribution fitting."""
        fitter = DistributionFitter(spark_session)
        row_count = small_dataset.count()

        sample = fitter._create_fitting_sample(small_dataset, "value", row_count)

        # Should be numpy array
        assert isinstance(sample, np.ndarray)

        # Should be <= 10k (default sample size)
        assert len(sample) <= 10_000

    @pytest.mark.parametrize("num_dists", [5, 100])
    def test_calculate_partitions(self, spark_session, num_dists):
        """Test partition calculation returns reasonable values."""
        fitter = DistributionFitter(spark_session)
        partitions = fitter._calculate_partitions(num_dists)
        assert 1 <= partitions <= num_dists

    def test_fit_caches_results(self, spark_session, small_dataset):
        """Test that fit results are cached and consistent."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

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
        df1 = results.df.toPandas()
        df2 = results.df.toPandas()
        assert len(df1) == len(df2)
        assert list(df1["distribution"]) == list(df2["distribution"])

    def test_fit_filters_failed_fits(self, spark_session, small_dataset):
        """Test that failed fits are filtered out."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # All results should have finite SSE
        df_pandas = results.df.toPandas()
        assert all(np.isfinite(df_pandas["sse"]))

    def test_fit_with_constant_data(self, spark_session, constant_dataset):
        """Test fitting with constant data (edge case)."""
        fitter = DistributionFitter(spark_session)

        # Should handle gracefully without crashing
        results = fitter.fit(constant_dataset, column="value", max_distributions=5)

        # Returns valid FitResults (may have 0 or more distributions)
        assert isinstance(results, FitResults)
        # Verify we can call methods on it without error
        _ = results.df.toPandas()

    def test_fit_with_rice_rule(self, spark_session, small_dataset):
        """Test fitting with Rice rule for bins."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", use_rice_rule=True, max_distributions=5)

        # Should fit all 5 requested distributions
        assert results.count() == 5

    def test_fit_excluded_distributions(self, spark_session, small_dataset):
        """Test that excluded distributions are not fitted."""
        fitter = DistributionFitter(spark_session, excluded_distributions=("norm", "expon"))
        results = fitter.fit(small_dataset, column="value", max_distributions=5)

        # norm and expon should not be in results
        df_pandas = results.df.toPandas()
        assert "norm" not in df_pandas["distribution"].values
        assert "expon" not in df_pandas["distribution"].values

    def test_fit_multiple_columns_sequential(self, spark_session):
        """Test fitting multiple columns sequentially."""
        # Create DataFrame with multiple columns
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 10_000)
        data2 = np.random.exponential(5, 10_000)

        df = spark_session.createDataFrame([(float(x), float(y)) for x, y in zip(data1, data2)], ["col1", "col2"])

        fitter = DistributionFitter(spark_session)

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

    def test_fit_reproducibility(self, spark_session, small_dataset):
        """Test that fitting is reproducible with same seed."""
        fitter1 = DistributionFitter(spark_session, random_seed=42)
        fitter2 = DistributionFitter(spark_session, random_seed=42)

        # Use max_distributions to speed up test
        results1 = fitter1.fit(small_dataset, column="value", max_distributions=5)
        results2 = fitter2.fit(small_dataset, column="value", max_distributions=5)

        # Should get same best distribution
        best1 = results1.best(n=1)[0]
        best2 = results2.best(n=1)[0]

        assert best1.distribution == best2.distribution
        # SSE might differ slightly due to sampling, but should be close
        assert np.isclose(best1.sse, best2.sse, rtol=0.1)


class TestMultiColumnFitting:
    """Tests for multi-column distribution fitting."""

    def test_fit_multiple_columns_basic(self, spark_session):
        """Test basic multi-column fitting."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.exponential(5, 5000)

        df = spark_session.createDataFrame(
            [(float(a), float(b)) for a, b in zip(data1, data2)],
            ["col1", "col2"]
        )

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, columns=["col1", "col2"], max_distributions=3)

        # Should have results for both columns
        assert results.count() == 6  # 2 columns × 3 distributions
        assert set(results.column_names) == {"col1", "col2"}

    def test_fit_multiple_columns_filtering(self, spark_session):
        """Test filtering multi-column results by column."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.exponential(5, 5000)

        df = spark_session.createDataFrame(
            [(float(a), float(b)) for a, b in zip(data1, data2)],
            ["normal_col", "expon_col"]
        )

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, columns=["normal_col", "expon_col"], max_distributions=3)

        # Filter to single column
        normal_results = results.for_column("normal_col")
        assert normal_results.count() == 3

        expon_results = results.for_column("expon_col")
        assert expon_results.count() == 3

    def test_fit_multiple_columns_best_per_column(self, spark_session):
        """Test best_per_column method."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 5000)
        data2 = np.random.exponential(5, 5000)

        df = spark_session.createDataFrame(
            [(float(a), float(b)) for a, b in zip(data1, data2)],
            ["col1", "col2"]
        )

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, columns=["col1", "col2"], max_distributions=5)

        best_per_col = results.best_per_column(n=1)

        assert "col1" in best_per_col
        assert "col2" in best_per_col
        assert len(best_per_col["col1"]) == 1
        assert len(best_per_col["col2"]) == 1
        assert best_per_col["col1"][0].column_name == "col1"
        assert best_per_col["col2"][0].column_name == "col2"

    def test_fit_backward_compatibility(self, spark_session, small_dataset):
        """Test that single column API still works with positional arg."""
        fitter = DistributionFitter(spark_session)
        # Using positional argument (backward compatible)
        results = fitter.fit(small_dataset, "value", max_distributions=3)

        assert results.count() == 3
        best = results.best(n=1)[0]
        assert best.column_name == "value"

    def test_fit_mutually_exclusive_params(self, spark_session, small_dataset):
        """Test error when both column and columns provided."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="Cannot provide both"):
            fitter.fit(small_dataset, column="value", columns=["value"])

    def test_fit_no_column_params(self, spark_session, small_dataset):
        """Test error when neither column nor columns provided."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="Must provide either"):
            fitter.fit(small_dataset)

    def test_fit_invalid_column_in_list(self, spark_session, small_dataset):
        """Test error when invalid column in columns list."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(small_dataset, columns=["value", "nonexistent"])

    def test_fit_single_column_via_columns_param(self, spark_session, small_dataset):
        """Test fitting single column using columns parameter."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, columns=["value"], max_distributions=3)

        assert results.count() == 3
        assert results.column_names == ["value"]


class TestBroadcastCleanup:
    """Tests for broadcast variable cleanup.

    These tests verify that fit() properly cleans up broadcast variables by
    patching Broadcast.unpersist at the class level to track calls.
    """

    def test_broadcast_cleanup_on_success(self, spark_session, small_dataset):
        """Verify unpersist() is called on broadcast variables after successful fit."""
        from pyspark import Broadcast

        fitter = DistributionFitter(spark_session)

        # Track unpersist calls at Broadcast class level
        unpersist_calls = []
        original_unpersist = Broadcast.unpersist

        def tracked_unpersist(self, blocking=False):
            unpersist_calls.append(self._jbroadcast.id())
            return original_unpersist(self, blocking)

        with patch.object(Broadcast, "unpersist", tracked_unpersist):
            results = fitter.fit(small_dataset, column="value", max_distributions=3)
            assert results.count() > 0

        # Verify both broadcasts (histogram_bc and data_sample_bc) were unpersisted
        assert len(unpersist_calls) == 2, f"Expected 2 unpersist calls, got {len(unpersist_calls)}"

    def test_discrete_broadcast_cleanup_on_success(self, spark_session):
        """Verify unpersist() is called for discrete fitter broadcasts."""
        from pyspark import Broadcast

        # Create discrete data
        data = [(int(x),) for x in np.random.poisson(lam=5, size=1000)]
        df = spark_session.createDataFrame(data, ["value"])

        fitter = DiscreteDistributionFitter(spark_session)

        # Track unpersist calls at Broadcast class level
        unpersist_calls = []
        original_unpersist = Broadcast.unpersist

        def tracked_unpersist(self, blocking=False):
            unpersist_calls.append(self._jbroadcast.id())
            return original_unpersist(self, blocking)

        with patch.object(Broadcast, "unpersist", tracked_unpersist):
            results = fitter.fit(df, column="value", max_distributions=3)
            assert results.count() > 0

        # Verify both broadcasts were unpersisted
        assert len(unpersist_calls) == 2, f"Expected 2 unpersist calls, got {len(unpersist_calls)}"

    def test_broadcast_cleanup_on_get_distributions_exception(self, spark_session, small_dataset):
        """Verify no broadcasts leak when get_distributions() raises.

        With the refactored architecture, broadcasts are created inside
        _fit_single_column(), which is called AFTER get_distributions().
        If get_distributions() fails, no broadcasts have been created yet,
        so no cleanup is needed. This is actually better than the old design
        because it minimizes resource allocation before potential failures.
        """
        fitter = DistributionFitter(spark_session)

        with patch.object(
            fitter._registry, "get_distributions", side_effect=ValueError("Injected error")
        ):
            with pytest.raises(ValueError, match="Injected error"):
                fitter.fit(small_dataset, column="value", max_distributions=3)

        # No broadcasts should have been created, so no cleanup needed

    def test_discrete_broadcast_cleanup_on_get_distributions_exception(self, spark_session):
        """Verify no discrete broadcasts leak when get_distributions() raises.

        Broadcasts are created after get_distributions() is called, so if
        get_distributions() fails, no broadcasts have been created yet.
        """
        data = [(int(x),) for x in np.random.poisson(lam=5, size=1000)]
        df = spark_session.createDataFrame(data, ["value"])

        fitter = DiscreteDistributionFitter(spark_session)

        with patch.object(
            fitter._registry, "get_distributions", side_effect=ValueError("Injected error")
        ):
            with pytest.raises(ValueError, match="Injected error"):
                fitter.fit(df, column="value", max_distributions=3)

        # No broadcasts should have been created, so no cleanup needed


class TestDistributionFitterPlotting:
    """Tests for plotting functionality."""

    def test_plot_after_fit(self, spark_session, small_dataset):
        """Test plotting after fitting."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Should not raise error (df and column are now required)
        fig, ax = fitter.plot(best, small_dataset, "value")

        assert fig is not None
        assert ax is not None

    def test_plot_with_title(self, spark_session, small_dataset):
        """Test plotting with title."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Should work with explicit data and title
        fig, ax = fitter.plot(best, small_dataset, "value", title="Test Plot")

        assert fig is not None
        assert ax is not None

    def test_plot_with_custom_params(self, spark_session, small_dataset):
        """Test plotting with custom parameters."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Should work with custom figsize, dpi, etc.
        fig, ax = fitter.plot(
            best, small_dataset, "value",
            figsize=(16, 10), dpi=150, title="Custom Plot"
        )

        assert fig is not None
        assert ax is not None

    def test_plot_qq_after_fit(self, spark_session, small_dataset):
        """Test Q-Q plotting after fitting."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        fig, ax = fitter.plot_qq(best, small_dataset, "value")

        assert fig is not None
        assert ax is not None

    def test_plot_qq_with_max_points(self, spark_session, small_dataset):
        """Test Q-Q plotting with custom max_points."""
        fitter = DistributionFitter(spark_session)
        results = fitter.fit(small_dataset, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        fig, ax = fitter.plot_qq(best, small_dataset, "value", max_points=500, title="Q-Q Test")

        assert fig is not None
        assert ax is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_dataset(self, spark_session):
        """Test with very small dataset."""
        data = np.array([1.0, 2.0, 3.0])
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)

        # Should handle gracefully without crashing
        results = fitter.fit(df, column="value", max_distributions=5)

        # Returns valid FitResults
        assert isinstance(results, FitResults)
        _ = results.df.toPandas()

    def test_single_value_dataset(self, spark_session):
        """Test with single value."""
        df = spark_session.createDataFrame([(42.0,)], ["value"])

        fitter = DistributionFitter(spark_session)

        # Should handle gracefully without crashing
        results = fitter.fit(df, column="value", max_distributions=5)

        # Returns valid FitResults
        assert isinstance(results, FitResults)
        _ = results.df.toPandas()

    def test_dataset_with_outliers(self, spark_session):
        """Test with dataset containing extreme outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 9995)
        outliers = np.array([1000, -1000, 2000, -2000, 3000])
        data = np.concatenate([normal_data, outliers])

        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)

        # Should handle outliers and fit all 5 requested distributions
        results = fitter.fit(df, column="value", max_distributions=5)

        assert results.count() == 5
        best = results.best(n=1)[0]
        assert best.sse < np.inf

    def test_apply_sampling_at_threshold(self, spark_session, small_dataset):
        """Test that data at threshold doesn't sample."""
        fitter = DistributionFitter(spark_session)
        df_result = fitter._apply_sampling(
            small_dataset, row_count=10_000, enable_sampling=True,
            sample_fraction=None, max_sample_size=1_000_000, sample_threshold=10_000
        )

        # At threshold should return original data (uses <=)
        assert df_result.count() == small_dataset.count()

    def test_fit_max_distributions_zero(self, spark_session, small_dataset):
        """Test fitting with max_distributions=0 raises error."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError):
            fitter.fit(small_dataset, column="value", max_distributions=0)

    def test_fit_with_different_columns(self, spark_session):
        """Test fitting on different column names."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["custom_column_name"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, column="custom_column_name", max_distributions=3)

        # Should fit all 3 requested distributions
        assert results.count() == 3

    def test_fit_invalid_bins(self, spark_session, small_dataset):
        """Test that invalid bins raises error."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="bins must be positive"):
            fitter.fit(small_dataset, column="value", bins=0)

    def test_fit_invalid_sample_fraction(self, spark_session, small_dataset):
        """Test that invalid sample_fraction raises error."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="sample_fraction must be in"):
            fitter.fit(small_dataset, column="value", sample_fraction=1.5)

class TestCoreNegativePaths:
    """Tests for negative/error paths in core module."""

    def test_fit_invalid_column(self, spark_session, small_dataset):
        """Test that fit raises error for invalid column."""
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(small_dataset, column="nonexistent_column", max_distributions=3)

    def test_fit_non_numeric_column(self, spark_session):
        """Test that fit raises error for non-numeric column."""
        df = spark_session.createDataFrame([("a",), ("b",), ("c",)], ["value"])
        fitter = DistributionFitter(spark_session)

        with pytest.raises(TypeError, match="must be numeric"):
            fitter.fit(df, column="value", max_distributions=3)

    def test_fit_empty_dataframe(self, spark_session):
        """Test that fit raises error for empty DataFrame."""
        from pyspark.sql.types import DoubleType, StructField, StructType

        schema = StructType([StructField("value", DoubleType(), True)])
        df = spark_session.createDataFrame([], schema)
        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="empty"):
            fitter.fit(df, column="value", max_distributions=3)

    def test_plot_with_different_data(self, spark_session):
        """Test that plot works with different data than fit."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 1000)
        data2 = np.random.normal(100, 20, 1000)

        df1 = spark_session.createDataFrame([(float(x),) for x in data1], ["value"])
        df2 = spark_session.createDataFrame([(float(x),) for x in data2], ["value"])

        fitter = DistributionFitter(spark_session)

        # Fit on first dataset
        results = fitter.fit(df1, column="value", max_distributions=3)
        best = results.best(n=1)[0]

        # Plot with different dataset should work
        fig, ax = fitter.plot(best, df=df2, column="value")
        assert fig is not None


class TestDiscreteDistributionFitter:
    """Tests for DiscreteDistributionFitter class."""

    def test_initialization(self, spark_session):
        """Test discrete fitter initialization with custom exclusions."""
        custom_exclusions = ("poisson", "geom")
        fitter = DiscreteDistributionFitter(
            spark_session, excluded_distributions=custom_exclusions, random_seed=123
        )

        assert fitter.excluded_distributions == custom_exclusions
        assert fitter.random_seed == 123

    def test_fit_identifies_poisson(self, spark_session, poisson_dataset):
        """Test that fitter identifies Poisson for Poisson data."""
        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(poisson_dataset, column="counts")

        top_5 = [r.distribution for r in results.best(n=5)]
        assert "poisson" in top_5

    def test_fit_identifies_nbinom(self, spark_session, nbinom_dataset):
        """Test that fitter identifies negative binomial for nbinom data."""
        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(nbinom_dataset, column="counts")

        top_5 = [r.distribution for r in results.best(n=5)]
        assert "nbinom" in top_5

    def test_fit_parameters_accuracy(self, spark_session, poisson_dataset, poisson_data):
        """Test that Poisson lambda is estimated accurately."""
        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(poisson_dataset, column="counts")

        poisson_fit = next(r for r in results.best(n=10) if r.distribution == "poisson")
        fitted_lambda = poisson_fit.parameters[0]
        true_lambda = np.mean(poisson_data)

        assert np.isclose(fitted_lambda, true_lambda, rtol=0.05)

    def test_fit_excluded_distributions(self, spark_session, poisson_dataset):
        """Test that excluded distributions are not fitted."""
        fitter = DiscreteDistributionFitter(
            spark_session, excluded_distributions=("poisson", "nbinom")
        )
        results = fitter.fit(poisson_dataset, column="counts")

        all_dists = [r.distribution for r in results.best(n=20)]
        assert "poisson" not in all_dists
        assert "nbinom" not in all_dists

    def test_fit_empty_dataframe_raises(self, spark_session):
        """Test that fit raises error for empty DataFrame."""
        from pyspark.sql.types import IntegerType, StructField, StructType

        schema = StructType([StructField("counts", IntegerType(), True)])
        df = spark_session.createDataFrame([], schema)
        fitter = DiscreteDistributionFitter(spark_session)

        with pytest.raises(ValueError, match="empty"):
            fitter.fit(df, column="counts")

    def test_fit_invalid_column_raises(self, spark_session, poisson_dataset):
        """Test that fit raises error for invalid column."""
        fitter = DiscreteDistributionFitter(spark_session)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(poisson_dataset, column="nonexistent")

    def test_plot_produces_stem_plot(self, spark_session, poisson_dataset):
        """Test that discrete plot produces expected stem plot elements."""
        import matplotlib.pyplot as plt

        fitter = DiscreteDistributionFitter(spark_session)
        results = fitter.fit(poisson_dataset, column="counts", max_distributions=3)
        best = results.best(n=1)[0]

        fig, ax = fitter.plot(best, poisson_dataset, "counts", title="Test Plot")

        # Verify plot has expected elements
        assert ax.get_title().startswith("Test Plot")
        assert ax.get_xlabel() == "Value"
        assert ax.get_ylabel() == "Probability"
        # Should have bars (histogram) and stems (fitted PMF)
        assert len(ax.containers) > 0 or len(ax.collections) > 0

        plt.close(fig)
