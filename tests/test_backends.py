"""Tests for execution backend implementations."""

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.backends.spark import SparkBackend
from spark_bestfit.fitting import compute_data_stats, fit_single_distribution


@pytest.fixture(scope="module")
def spark():
    """Create SparkSession for testing."""
    return (
        SparkSession.builder.master("local[2]")
        .appName("test_backends")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@pytest.fixture
def normal_data():
    """Generate normal distribution test data."""
    np.random.seed(42)
    return np.random.normal(loc=50, scale=10, size=1000)


@pytest.fixture
def histogram(normal_data):
    """Create histogram from normal data."""
    y_hist, bin_edges = np.histogram(normal_data, bins=30, density=True)
    return (y_hist, bin_edges)


class TestSparkBackend:
    """Tests for SparkBackend implementation."""

    def test_init_with_spark_session(self, spark):
        """SparkBackend initializes with provided SparkSession."""
        backend = SparkBackend(spark)
        assert backend.spark is spark

    def test_broadcast_and_destroy(self, spark, normal_data):
        """SparkBackend broadcasts and cleans up data."""
        backend = SparkBackend(spark)

        # Broadcast data
        handle = backend.broadcast(normal_data)
        assert handle is not None
        assert hasattr(handle, "value")
        np.testing.assert_array_equal(handle.value, normal_data)

        # Cleanup (should not raise)
        backend.destroy_broadcast(handle)

    def test_get_parallelism(self, spark):
        """SparkBackend reports parallelism from Spark."""
        backend = SparkBackend(spark)
        parallelism = backend.get_parallelism()
        assert parallelism >= 1
        assert parallelism == spark.sparkContext.defaultParallelism

    def test_create_dataframe(self, spark):
        """SparkBackend creates Spark DataFrames."""
        backend = SparkBackend(spark)
        data = [("a",), ("b",), ("c",)]
        df = backend.create_dataframe(data, ["name"])

        assert df.count() == 3
        assert df.columns == ["name"]
        assert [row["name"] for row in df.collect()] == ["a", "b", "c"]

    def test_collect_column(self, spark):
        """SparkBackend collects column as numpy array."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["value"])

        result = backend.collect_column(spark_df, "value")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_column_stats(self, spark):
        """SparkBackend computes column statistics."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)], ["value"])

        stats = backend.get_column_stats(spark_df, "value")
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5

    def test_sample_column(self, spark, normal_data):
        """SparkBackend samples column data."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame([(float(x),) for x in normal_data], ["value"])

        sample = backend.sample_column(spark_df, "value", fraction=0.1, seed=42)
        assert len(sample) > 0
        assert len(sample) < len(normal_data)

    def test_parallel_fit_continuous(self, spark, normal_data, histogram):
        """SparkBackend fits distributions in parallel."""
        backend = SparkBackend(spark)

        results = backend.parallel_fit(
            distributions=["norm", "uniform"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test_col",
            data_stats=compute_data_stats(normal_data),
            num_partitions=2,
            is_discrete=False,
        )

        # Should have results for both distributions
        assert len(results) == 2
        dist_names = {r["distribution"] for r in results}
        assert dist_names == {"norm", "uniform"}

        # Verify result structure and value validity
        for result in results:
            assert result["column_name"] == "test_col"
            assert isinstance(result["parameters"], list)
            assert len(result["parameters"]) >= 2  # At least loc, scale
            assert np.isfinite(result["sse"])
            assert np.isfinite(result["aic"])
            assert np.isfinite(result["bic"])
            # Data stats should be present
            assert result["data_min"] is not None
            assert result["data_max"] is not None

        # Normal should fit better than uniform for normal data
        norm_result = next(r for r in results if r["distribution"] == "norm")
        unif_result = next(r for r in results if r["distribution"] == "uniform")
        assert norm_result["sse"] < unif_result["sse"]

    def test_parallel_fit_with_lazy_metrics(self, spark, normal_data, histogram):
        """SparkBackend respects lazy_metrics flag."""
        backend = SparkBackend(spark)

        results = backend.parallel_fit(
            distributions=["norm"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            lazy_metrics=True,
            is_discrete=False,
        )

        assert len(results) == 1
        # With lazy_metrics=True, KS/AD should be None
        assert results[0]["ks_statistic"] is None
        assert results[0]["pvalue"] is None
        assert results[0]["ad_statistic"] is None
        # But SSE, AIC, BIC should still be computed
        assert results[0]["sse"] is not None
        assert np.isfinite(results[0]["sse"])

    def test_parallel_fit_with_bounds(self, spark, normal_data, histogram):
        """SparkBackend handles bounded fitting."""
        backend = SparkBackend(spark)

        results = backend.parallel_fit(
            distributions=["norm"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            lower_bound=30.0,
            upper_bound=70.0,
            is_discrete=False,
        )

        assert len(results) == 1
        # Bounds should be recorded in result
        assert results[0]["lower_bound"] == 30.0
        assert results[0]["upper_bound"] == 70.0

    def test_parallel_fit_invalid_distribution(self, spark, normal_data, histogram):
        """SparkBackend gracefully handles invalid distributions."""
        backend = SparkBackend(spark)

        # Mix valid and invalid distributions
        results = backend.parallel_fit(
            distributions=["norm", "not_a_real_distribution"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        # Should still get valid result for norm, invalid filtered out
        assert len(results) >= 1
        valid_dists = {r["distribution"] for r in results}
        assert "norm" in valid_dists

    def test_parallel_fit_verifies_parameters(self, spark, normal_data, histogram):
        """SparkBackend produces accurate fitted parameters for known data."""
        backend = SparkBackend(spark)

        # Data generated with loc=50, scale=10
        results = backend.parallel_fit(
            distributions=["norm"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        assert len(results) == 1
        params = results[0]["parameters"]
        fitted_loc, fitted_scale = params[0], params[1]

        # Parameters should be close to true values (loc=50, scale=10)
        assert abs(fitted_loc - 50) < 2.0  # Within 2 units
        assert abs(fitted_scale - 10) < 2.0  # Within 2 units


class TestLocalBackend:
    """Tests for LocalBackend implementation."""

    def test_init_default_workers(self):
        """LocalBackend initializes with default workers."""
        backend = LocalBackend()
        assert backend.max_workers >= 1

    def test_init_custom_workers(self):
        """LocalBackend respects custom worker count."""
        backend = LocalBackend(max_workers=4)
        assert backend.max_workers == 4

    def test_broadcast_no_op(self, normal_data):
        """LocalBackend broadcast is no-op."""
        backend = LocalBackend()
        handle = backend.broadcast(normal_data)
        # For local backend, broadcast returns data as-is
        np.testing.assert_array_equal(handle, normal_data)

    def test_destroy_broadcast_no_op(self, normal_data):
        """LocalBackend destroy_broadcast is no-op."""
        backend = LocalBackend()
        handle = backend.broadcast(normal_data)
        # Should not raise
        backend.destroy_broadcast(handle)

    def test_get_parallelism(self):
        """LocalBackend reports worker count as parallelism."""
        backend = LocalBackend(max_workers=8)
        assert backend.get_parallelism() == 8

    def test_create_dataframe(self):
        """LocalBackend creates pandas DataFrames."""
        backend = LocalBackend()
        data = [("a",), ("b",), ("c",)]
        df = backend.create_dataframe(data, ["name"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["name"]
        assert list(df["name"]) == ["a", "b", "c"]

    def test_collect_column(self):
        """LocalBackend extracts column as numpy array."""
        backend = LocalBackend()
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        result = backend.collect_column(df, "value")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_column_stats(self):
        """LocalBackend computes column statistics."""
        backend = LocalBackend()
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        stats = backend.get_column_stats(df, "value")
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5

    def test_sample_column(self, normal_data):
        """LocalBackend samples column data."""
        backend = LocalBackend()
        df = pd.DataFrame({"value": normal_data})

        sample = backend.sample_column(df, "value", fraction=0.1, seed=42)
        assert len(sample) > 0
        assert len(sample) < len(normal_data)

    def test_parallel_fit_continuous(self, normal_data, histogram):
        """LocalBackend fits distributions in parallel using threads."""
        backend = LocalBackend(max_workers=2)

        results = backend.parallel_fit(
            distributions=["norm", "uniform"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test_col",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        # Should have results for both distributions
        assert len(results) == 2
        dist_names = {r["distribution"] for r in results}
        assert dist_names == {"norm", "uniform"}

        # Verify result structure and value validity
        for result in results:
            assert result["column_name"] == "test_col"
            assert isinstance(result["parameters"], list)
            assert np.isfinite(result["sse"])
            assert np.isfinite(result["aic"])

        # Normal should fit better than uniform for normal data
        norm_result = next(r for r in results if r["distribution"] == "norm")
        unif_result = next(r for r in results if r["distribution"] == "uniform")
        assert norm_result["sse"] < unif_result["sse"]

    def test_parallel_fit_with_lazy_metrics(self, normal_data, histogram):
        """LocalBackend respects lazy_metrics flag."""
        backend = LocalBackend(max_workers=2)

        results = backend.parallel_fit(
            distributions=["norm"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            lazy_metrics=True,
            is_discrete=False,
        )

        assert len(results) == 1
        # With lazy_metrics=True, KS/AD should be None
        assert results[0]["ks_statistic"] is None
        assert results[0]["pvalue"] is None

    def test_parallel_fit_verifies_parameters(self, normal_data, histogram):
        """LocalBackend produces accurate fitted parameters for known data."""
        backend = LocalBackend(max_workers=2)

        results = backend.parallel_fit(
            distributions=["norm"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        assert len(results) == 1
        params = results[0]["parameters"]
        fitted_loc, fitted_scale = params[0], params[1]

        # Parameters should be close to true values (loc=50, scale=10)
        assert abs(fitted_loc - 50) < 2.0
        assert abs(fitted_scale - 10) < 2.0

    def test_parallel_fit_discrete(self):
        """LocalBackend fits discrete distributions correctly."""
        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        backend = LocalBackend(max_workers=2)

        # Create Poisson data with known lambda=7
        np.random.seed(42)
        data_sample = np.random.poisson(lam=7, size=1000).astype(int)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        results = backend.parallel_fit(
            distributions=["poisson"],
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="counts",
            data_stats=data_stats,
            is_discrete=True,
        )

        assert len(results) == 1
        # Fitted lambda should be close to true value (7)
        fitted_lambda = results[0]["parameters"][0]
        assert abs(fitted_lambda - 7) < 0.5


class TestBackendEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_distributions_list(self, spark, normal_data, histogram):
        """Backends handle empty distribution list gracefully."""
        spark_backend = SparkBackend(spark)
        local_backend = LocalBackend()

        spark_results = spark_backend.parallel_fit(
            distributions=[],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        local_results = local_backend.parallel_fit(
            distributions=[],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        assert spark_results == []
        assert local_results == []

    def test_small_data_sample(self, spark):
        """Backends handle very small data samples."""
        backend = SparkBackend(spark)

        # Very small sample (10 points)
        np.random.seed(42)
        small_data = np.random.normal(loc=0, scale=1, size=10)
        y_hist, bin_edges = np.histogram(small_data, bins=5, density=True)

        results = backend.parallel_fit(
            distributions=["norm"],
            histogram=(y_hist, bin_edges),
            data_sample=small_data,
            fit_func=fit_single_distribution,
            column_name="small",
            data_stats=compute_data_stats(small_data),
            is_discrete=False,
        )

        # Should still produce a result (fitting may be less accurate)
        assert len(results) == 1
        assert np.isfinite(results[0]["sse"])

    def test_sample_column_reproducibility(self, spark, normal_data):
        """Sample column produces reproducible results with same seed."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame([(float(x),) for x in normal_data], ["value"])

        sample1 = backend.sample_column(spark_df, "value", fraction=0.1, seed=42)
        sample2 = backend.sample_column(spark_df, "value", fraction=0.1, seed=42)

        # Same seed should produce same sample
        np.testing.assert_array_equal(sample1, sample2)

    def test_sample_column_different_seeds(self, spark, normal_data):
        """Sample column produces different results with different seeds."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame([(float(x),) for x in normal_data], ["value"])

        sample1 = backend.sample_column(spark_df, "value", fraction=0.5, seed=42)
        sample2 = backend.sample_column(spark_df, "value", fraction=0.5, seed=123)

        # Different seeds should produce different samples (with high probability)
        assert not np.array_equal(sample1, sample2)

    def test_many_distributions_parallel(self, spark, normal_data, histogram):
        """Backends efficiently handle many distributions in parallel."""
        backend = SparkBackend(spark)

        # Fit many distributions
        distributions = ["norm", "expon", "gamma", "lognorm", "weibull_min", "beta"]

        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        # Should get results for multiple distributions (some may fail)
        assert len(results) >= 3
        # All results should have valid SSE
        for r in results:
            assert np.isfinite(r["sse"])


class TestBackendCompatibility:
    """Tests verifying backends produce compatible results."""

    def test_continuous_fit_results_compatible(self, spark, normal_data, histogram):
        """SparkBackend and LocalBackend produce compatible continuous results."""
        spark_backend = SparkBackend(spark)
        local_backend = LocalBackend(max_workers=2)

        # Fit same distributions with both backends
        distributions = ["norm", "expon"]
        data_stats = compute_data_stats(normal_data)

        spark_results = spark_backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=data_stats,
            is_discrete=False,
        )

        local_results = local_backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=data_stats,
            is_discrete=False,
        )

        # Both should produce results
        assert len(spark_results) > 0
        assert len(local_results) > 0

        # Results should have same structure
        spark_keys = set(spark_results[0].keys())
        local_keys = set(local_results[0].keys())
        assert spark_keys == local_keys

    def test_discrete_fit_results_compatible(self, spark):
        """SparkBackend and LocalBackend produce compatible discrete results."""
        # Create discrete test data (Poisson-like)
        np.random.seed(42)
        data_sample = np.random.poisson(lam=5, size=500).astype(int)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        spark_backend = SparkBackend(spark)
        local_backend = LocalBackend(max_workers=2)

        # Fit with both backends
        distributions = ["poisson"]

        # Import discrete fitting function
        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        spark_results = spark_backend.parallel_fit(
            distributions=distributions,
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="test",
            data_stats=data_stats,
            is_discrete=True,
        )

        local_results = local_backend.parallel_fit(
            distributions=distributions,
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="test",
            data_stats=data_stats,
            is_discrete=True,
        )

        # Both should produce results
        assert len(spark_results) > 0
        assert len(local_results) > 0

        # SSE values should be similar (both fitting same data)
        spark_sse = spark_results[0]["sse"]
        local_sse = local_results[0]["sse"]
        # Allow some tolerance due to potential numerical differences
        assert abs(spark_sse - local_sse) < 0.01


class TestBackendWithFitter:
    """Tests verifying backends work correctly with fitter classes."""

    def test_continuous_fitter_with_spark_backend(self, spark, normal_data):
        """DistributionFitter works with explicit SparkBackend."""
        from spark_bestfit import DistributionFitter, SparkBackend

        # Create fitter with explicit backend
        backend = SparkBackend(spark)
        fitter = DistributionFitter(backend=backend)

        # Create test DataFrame
        spark_df = spark.createDataFrame([(float(x),) for x in normal_data], ["value"])

        # Fit should work
        results = fitter.fit(spark_df, column="value", max_distributions=3)
        assert len(results.best(n=1)) == 1

    def test_continuous_fitter_backward_compatible(self, spark, normal_data):
        """DistributionFitter works without explicit backend (backward compat)."""
        from spark_bestfit import DistributionFitter

        # Old way: just pass spark
        fitter = DistributionFitter(spark)

        # Fitter should have created SparkBackend internally
        assert hasattr(fitter, "_backend")

        # Create test DataFrame and fit
        spark_df = spark.createDataFrame([(float(x),) for x in normal_data], ["value"])
        results = fitter.fit(spark_df, column="value", max_distributions=3)
        assert len(results.best(n=1)) == 1

    def test_discrete_fitter_with_spark_backend(self, spark):
        """DiscreteDistributionFitter works with explicit SparkBackend."""
        from spark_bestfit import DiscreteDistributionFitter, SparkBackend

        # Create test data
        np.random.seed(42)
        poisson_data = np.random.poisson(lam=7, size=500)

        # Create fitter with explicit backend
        backend = SparkBackend(spark)
        fitter = DiscreteDistributionFitter(backend=backend)

        # Create test DataFrame
        spark_df = spark.createDataFrame([(int(x),) for x in poisson_data], ["counts"])

        # Fit should work
        results = fitter.fit(spark_df, column="counts", max_distributions=3)
        assert len(results.best(n=1, metric="aic")) == 1
