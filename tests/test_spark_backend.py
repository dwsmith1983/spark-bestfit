"""Tests for Spark backend implementation.

This module contains Spark-specific tests. All tests are skipped if PySpark
is not installed, following the same pattern as test_ray_backend.py.
"""

import numpy as np
import pandas as pd
import pytest

# Skip all tests if pyspark not installed
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.backends.spark import SparkBackend
from spark_bestfit.fitting import compute_data_stats, fit_single_distribution


@pytest.fixture(scope="module")
def spark():
    """Create SparkSession for testing."""
    return (
        SparkSession.builder.master("local[2]")
        .appName("test_spark_backend")
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


class TestSparkBackendInit:
    """Tests for SparkBackend initialization."""

    def test_init_with_spark_session(self, spark):
        """SparkBackend initializes with provided SparkSession."""
        backend = SparkBackend(spark)
        assert backend.spark is spark

    def test_get_parallelism(self, spark):
        """SparkBackend reports parallelism from Spark."""
        backend = SparkBackend(spark)
        parallelism = backend.get_parallelism()
        assert parallelism >= 1
        assert parallelism == spark.sparkContext.defaultParallelism


class TestSparkBackendBroadcast:
    """Tests for SparkBackend broadcast operations."""

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


class TestSparkBackendDataFrame:
    """Tests for SparkBackend DataFrame operations."""

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


class TestSparkBackendParallelFit:
    """Tests for SparkBackend parallel fitting."""

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


class TestSparkBackendVsLocalBackend:
    """Tests verifying SparkBackend and LocalBackend produce compatible results."""

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
        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

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


class TestSparkBackendWithFitter:
    """Tests verifying fitter classes work correctly with SparkBackend."""

    def test_continuous_fitter_with_spark_backend(self, spark, normal_data):
        """DistributionFitter works with explicit SparkBackend."""
        from spark_bestfit import DistributionFitter

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
        from spark_bestfit import DiscreteDistributionFitter

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


class TestSparkBackendCorrelation:
    """Tests for SparkBackend.compute_correlation method."""

    @pytest.fixture
    def correlated_data(self):
        """Generate correlated data for testing."""
        np.random.seed(42)
        n = 500
        # Generate correlated data: x and y have high positive correlation
        x = np.random.normal(0, 1, n)
        y = 0.8 * x + 0.2 * np.random.normal(0, 1, n)  # Correlated with x
        z = np.random.normal(0, 1, n)  # Uncorrelated
        return pd.DataFrame({"x": x, "y": y, "z": z})

    def test_compute_correlation_shape(self, spark, correlated_data):
        """SparkBackend.compute_correlation returns correct shape."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame(correlated_data)

        corr = backend.compute_correlation(spark_df, ["x", "y", "z"], method="spearman")

        assert corr.shape == (3, 3)
        # Diagonal should be 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0], decimal=5)

    def test_compute_correlation_values(self, spark, correlated_data):
        """SparkBackend.compute_correlation captures correlation structure."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame(correlated_data)

        corr = backend.compute_correlation(spark_df, ["x", "y", "z"], method="spearman")

        # x and y should have high positive correlation (> 0.7)
        assert corr[0, 1] > 0.7
        # x and z should have low correlation (< 0.2)
        assert abs(corr[0, 2]) < 0.2

    def test_compute_correlation_pearson(self, spark, correlated_data):
        """SparkBackend.compute_correlation supports pearson method."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame(correlated_data)

        corr = backend.compute_correlation(spark_df, ["x", "y"], method="pearson")

        assert corr.shape == (2, 2)
        # Should still capture the correlation
        assert corr[0, 1] > 0.7

    def test_correlation_backends_compatible(self, spark, correlated_data):
        """SparkBackend and LocalBackend produce similar correlation matrices."""
        spark_backend = SparkBackend(spark)
        local_backend = LocalBackend()
        spark_df = spark.createDataFrame(correlated_data)

        spark_corr = spark_backend.compute_correlation(spark_df, ["x", "y"], method="spearman")
        local_corr = local_backend.compute_correlation(correlated_data, ["x", "y"], method="spearman")

        # Correlation values should be very close
        np.testing.assert_array_almost_equal(spark_corr, local_corr, decimal=2)


class TestSparkBackendHistogram:
    """Tests for SparkBackend.compute_histogram method."""

    @pytest.fixture
    def histogram_data(self):
        """Generate data for histogram testing."""
        np.random.seed(42)
        return pd.DataFrame({"value": np.random.normal(50, 10, 1000)})

    def test_compute_histogram_shape(self, spark, histogram_data):
        """SparkBackend.compute_histogram returns correct shape."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame(histogram_data)

        bin_edges = np.linspace(0, 100, 21)  # 20 bins
        bin_counts, total = backend.compute_histogram(spark_df, "value", bin_edges)

        assert len(bin_counts) == 20
        assert total == len(histogram_data)

    def test_compute_histogram_sum(self, spark, histogram_data):
        """SparkBackend.compute_histogram bin counts sum to total."""
        backend = SparkBackend(spark)
        spark_df = spark.createDataFrame(histogram_data)

        # Use bin edges that cover the full data range
        bin_edges = np.linspace(0, 100, 11)  # 10 bins
        bin_counts, total = backend.compute_histogram(spark_df, "value", bin_edges)

        # Bins should capture all data
        assert sum(bin_counts) == total
        assert total == len(histogram_data)

    def test_histogram_backends_compatible(self, spark, histogram_data):
        """SparkBackend and LocalBackend produce identical histogram counts."""
        spark_backend = SparkBackend(spark)
        local_backend = LocalBackend()
        spark_df = spark.createDataFrame(histogram_data)

        # Use bin edges that cover full data range
        bin_edges = np.linspace(0, 100, 11)

        spark_counts, spark_total = spark_backend.compute_histogram(spark_df, "value", bin_edges)
        local_counts, local_total = local_backend.compute_histogram(histogram_data, "value", bin_edges)

        # Totals should be equal
        assert spark_total == local_total
        # Bin counts should be equal
        np.testing.assert_array_equal(spark_counts, local_counts)


class TestSparkBackendGenerateSamples:
    """Tests for SparkBackend.generate_samples method."""

    def test_generate_samples_shape(self, spark):
        """SparkBackend.generate_samples returns correct number of samples."""
        backend = SparkBackend(spark)

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"col1": rng.normal(0, 1, n_samples), "col2": rng.normal(0, 1, n_samples)}

        result = backend.generate_samples(
            n=100,
            generator_func=generator,
            column_names=["col1", "col2"],
            num_partitions=2,
            random_seed=42,
        )

        # Collect and verify
        pdf = result.toPandas()
        assert len(pdf) == 100
        assert list(pdf.columns) == ["col1", "col2"]

    def test_generate_samples_reproducibility(self, spark):
        """SparkBackend.generate_samples is reproducible with same seed."""
        backend = SparkBackend(spark)

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(0, 1, n_samples)}

        result1 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"], num_partitions=1, random_seed=42
        )
        result2 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"], num_partitions=1, random_seed=42
        )

        pdf1 = result1.toPandas()
        pdf2 = result2.toPandas()
        np.testing.assert_array_almost_equal(pdf1["value"].values, pdf2["value"].values)

    def test_generate_samples_statistical_properties(self, spark):
        """SparkBackend.generate_samples produces samples with correct statistics."""
        backend = SparkBackend(spark)

        # Generate samples from known distribution
        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(loc=100, scale=15, size=n_samples)}

        result = backend.generate_samples(
            n=1000,
            generator_func=generator,
            column_names=["value"],
            num_partitions=4,
            random_seed=42,
        )

        pdf = result.toPandas()

        # Mean should be close to 100
        assert abs(pdf["value"].mean() - 100) < 3.0
        # Std should be close to 15
        assert abs(pdf["value"].std() - 15) < 3.0


class TestSparkBackendSampling:
    """Tests for sampling module with SparkBackend."""

    def test_sample_distributed_shape(self, spark):
        """sample_distributed with SparkBackend returns correct shape."""
        from spark_bestfit.sampling import sample_distributed

        backend = SparkBackend(spark)
        result = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=100,
            backend=backend,
            num_partitions=2,
            random_seed=42,
        )

        pdf = result.toPandas()
        assert len(pdf) == 100
        assert "sample" in pdf.columns

    def test_sample_distributed_statistics(self, spark):
        """sample_distributed with SparkBackend produces correct statistics."""
        from spark_bestfit.sampling import sample_distributed

        backend = SparkBackend(spark)
        result = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=1000,
            backend=backend,
            num_partitions=4,
            random_seed=42,
        )

        pdf = result.toPandas()
        # Mean should be close to 50
        assert abs(pdf["sample"].mean() - 50) < 5.0
        # Std should be close to 10
        assert abs(pdf["sample"].std() - 10) < 3.0

    def test_sample_spark_backward_compatible(self, spark):
        """sample_spark (backward compat) still works."""
        from spark_bestfit.sampling import sample_spark

        result = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=50,
            spark=spark,
            random_seed=42,
        )

        pdf = result.toPandas()
        assert len(pdf) == 50
        assert "sample" in pdf.columns


class TestSparkBackendProgressCallback:
    """Tests for progress callback with SparkBackend."""

    def test_progress_callback(self, spark, normal_data, histogram):
        """SparkBackend invokes progress callback via StatusTracker."""
        backend = SparkBackend(spark)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma", "uniform"]
        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            num_partitions=2,
            progress_callback=on_progress,
        )

        # Fitting should complete successfully
        assert len(results) > 0

    def test_progress_callback_error_handling(self, spark, normal_data, histogram):
        """SparkBackend handles callback errors gracefully."""
        backend = SparkBackend(spark)

        def failing_callback(completed, total, percent):
            raise ValueError("Intentional callback error")

        distributions = ["norm", "expon"]
        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            num_partitions=2,
            progress_callback=failing_callback,
        )

        # Fitting should still complete successfully
        assert len(results) > 0

    def test_no_callback(self, spark, normal_data, histogram):
        """SparkBackend works fine without progress callback."""
        backend = SparkBackend(spark)

        distributions = ["norm", "expon"]
        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            num_partitions=2,
            progress_callback=None,
        )

        assert len(results) > 0

    def test_fitter_progress_callback(self, spark, normal_data):
        """DistributionFitter progress callback works with SparkBackend."""
        from spark_bestfit import DistributionFitter

        fitter = DistributionFitter(spark)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        df = spark.createDataFrame([(float(x),) for x in normal_data], ["value"])
        results = fitter.fit(
            df,
            column="value",
            max_distributions=5,
            progress_callback=on_progress,
        )

        # Should have results
        assert len(results.best(n=1)) == 1


class TestSparkBackendEdgeCases:
    """Edge case tests for SparkBackend."""

    def test_empty_distributions_list(self, spark, normal_data, histogram):
        """SparkBackend handles empty distribution list gracefully."""
        backend = SparkBackend(spark)

        results = backend.parallel_fit(
            distributions=[],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        assert results == []

    def test_small_data_sample(self, spark):
        """SparkBackend handles very small data samples."""
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
        """SparkBackend efficiently handles many distributions in parallel."""
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

    def test_correlation_matrix_symmetry(self, spark):
        """Correlation matrix should be symmetric."""
        backend = SparkBackend(spark)

        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.normal(0, 1, 100),
        })
        spark_df = spark.createDataFrame(df)

        corr = backend.compute_correlation(spark_df, ["a", "b", "c"], method="spearman")

        # Verify symmetry: corr[i,j] == corr[j,i]
        np.testing.assert_array_almost_equal(corr, corr.T, decimal=10)

    def test_correlation_detects_negative_correlation(self, spark):
        """Correlation should detect negative relationships."""
        backend = SparkBackend(spark)

        np.random.seed(42)
        n = 300
        x = np.random.normal(0, 1, n)
        y = -0.9 * x + 0.1 * np.random.normal(0, 1, n)  # Strong negative correlation
        df = pd.DataFrame({"x": x, "y": y})
        spark_df = spark.createDataFrame(df)

        corr = backend.compute_correlation(spark_df, ["x", "y"], method="spearman")

        # Should detect negative correlation
        assert corr[0, 1] < -0.7

    def test_generate_samples_different_seeds_differ(self, spark):
        """Different seeds should produce different samples."""
        backend = SparkBackend(spark)

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(0, 1, n_samples)}

        result1 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"],
            num_partitions=1, random_seed=42
        )
        result2 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"],
            num_partitions=1, random_seed=999
        )

        pdf1 = result1.toPandas()
        pdf2 = result2.toPandas()

        # Values should be different (not identical)
        assert not np.allclose(pdf1["value"].values, pdf2["value"].values)


class TestHistogramComputerWithSparkBackend:
    """Tests for HistogramComputer with SparkBackend."""

    def test_histogram_computer_with_spark_backend(self, spark):
        """HistogramComputer works with SparkBackend."""
        from spark_bestfit.histogram import HistogramComputer

        backend = SparkBackend(spark)
        computer = HistogramComputer(backend)

        # Create test data
        np.random.seed(42)
        data = np.random.normal(50, 10, 500)
        spark_df = spark.createDataFrame([(float(x),) for x in data], ["value"])

        y_hist, bin_edges = computer.compute_histogram(spark_df, "value", bins=20)

        assert len(y_hist) == 20
        assert len(bin_edges) == 21

    def test_histogram_computer_backward_compatible(self, spark):
        """HistogramComputer works without explicit backend (backward compat)."""
        from spark_bestfit.histogram import HistogramComputer

        # Create without backend (should auto-create SparkBackend)
        computer = HistogramComputer()

        # Create test data
        np.random.seed(42)
        data = np.random.normal(50, 10, 500)
        spark_df = spark.createDataFrame([(float(x),) for x in data], ["value"])

        y_hist, bin_edges = computer.compute_histogram(spark_df, "value", bins=20)

        assert len(y_hist) == 20


class TestCopulaWithSparkBackend:
    """Tests for GaussianCopula with SparkBackend."""

    @pytest.fixture
    def copula_marginals(self):
        """Create marginals for copula testing."""
        from spark_bestfit.results import DistributionFitResult

        return {
            "col1": DistributionFitResult(
                distribution="norm",
                parameters=[50.0, 10.0],
                sse=0.01,
                column_name="col1",
            ),
            "col2": DistributionFitResult(
                distribution="norm",
                parameters=[100.0, 20.0],
                sse=0.01,
                column_name="col2",
            ),
        }

    def test_copula_sample_spark_backward_compatible(self, spark, copula_marginals):
        """GaussianCopula.sample_spark works (backward compat)."""
        from spark_bestfit.copula import GaussianCopula

        # Create copula directly with known correlation
        corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        # Use sample_spark (backward compat method)
        samples_df = copula.sample_spark(n=50, spark=spark, random_seed=42)

        # Should return Spark DataFrame
        pdf = samples_df.toPandas()
        assert len(pdf) == 50
        assert list(pdf.columns) == ["col1", "col2"]
