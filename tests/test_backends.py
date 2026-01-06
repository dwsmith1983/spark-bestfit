"""Tests for LocalBackend implementation.

Spark backend tests are in test_spark_backend.py (following Ray pattern).
"""

import numpy as np
import pandas as pd
import pytest

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.fitting import compute_data_stats, fit_single_distribution


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

    def test_parallel_fit_with_bounds(self, normal_data, histogram):
        """LocalBackend handles bounded fitting."""
        backend = LocalBackend(max_workers=2)

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

    def test_parallel_fit_invalid_distribution(self, normal_data, histogram):
        """LocalBackend gracefully handles invalid distributions."""
        backend = LocalBackend(max_workers=2)

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


class TestLocalBackendEdgeCases:
    """Edge case tests for LocalBackend."""

    def test_empty_distributions_list(self, normal_data, histogram):
        """LocalBackend handles empty distribution list gracefully."""
        backend = LocalBackend()

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

    def test_small_data_sample(self):
        """LocalBackend handles very small data samples."""
        backend = LocalBackend()

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

    def test_sample_column_reproducibility(self, normal_data):
        """Sample column produces reproducible results with same seed."""
        backend = LocalBackend()
        df = pd.DataFrame({"value": normal_data})

        sample1 = backend.sample_column(df, "value", fraction=0.1, seed=42)
        sample2 = backend.sample_column(df, "value", fraction=0.1, seed=42)

        # Same seed should produce same sample
        np.testing.assert_array_equal(sample1, sample2)

    def test_sample_column_different_seeds(self, normal_data):
        """Sample column produces different results with different seeds."""
        backend = LocalBackend()
        df = pd.DataFrame({"value": normal_data})

        sample1 = backend.sample_column(df, "value", fraction=0.5, seed=42)
        sample2 = backend.sample_column(df, "value", fraction=0.5, seed=123)

        # Different seeds should produce different samples (with high probability)
        assert not np.array_equal(sample1, sample2)

    def test_many_distributions_parallel(self, normal_data, histogram):
        """LocalBackend efficiently handles many distributions in parallel."""
        backend = LocalBackend(max_workers=4)

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


class TestLocalBackendWithFitter:
    """Tests verifying LocalBackend works correctly with fitter classes."""

    def test_continuous_fitter_with_local_backend(self, normal_data):
        """DistributionFitter works with LocalBackend."""
        from spark_bestfit import DistributionFitter, LocalBackend

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)

        df = pd.DataFrame({"value": normal_data})
        results = fitter.fit(df, column="value", max_distributions=3)
        assert len(results.best(n=1)) == 1

    def test_discrete_fitter_with_local_backend(self):
        """DiscreteDistributionFitter works with LocalBackend."""
        from spark_bestfit import DiscreteDistributionFitter, LocalBackend

        np.random.seed(42)
        poisson_data = np.random.poisson(lam=7, size=500)

        backend = LocalBackend(max_workers=2)
        fitter = DiscreteDistributionFitter(backend=backend)

        df = pd.DataFrame({"counts": poisson_data})
        results = fitter.fit(df, column="counts", max_distributions=3)
        assert len(results.best(n=1, metric="aic")) == 1


class TestLocalBackendCorrelation:
    """Tests for LocalBackend correlation computation."""

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

    def test_compute_correlation_shape(self, correlated_data):
        """LocalBackend.compute_correlation returns correct shape."""
        backend = LocalBackend()

        corr = backend.compute_correlation(correlated_data, ["x", "y", "z"], method="spearman")

        assert corr.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0], decimal=5)

    def test_compute_correlation_values(self, correlated_data):
        """LocalBackend.compute_correlation captures correlation structure."""
        backend = LocalBackend()

        corr = backend.compute_correlation(correlated_data, ["x", "y", "z"], method="spearman")

        # x and y should have high positive correlation (> 0.7)
        assert corr[0, 1] > 0.7
        # x and z should have low correlation (< 0.2)
        assert abs(corr[0, 2]) < 0.2

    def test_compute_correlation_pearson(self, correlated_data):
        """LocalBackend.compute_correlation supports pearson method."""
        backend = LocalBackend()

        corr = backend.compute_correlation(correlated_data, ["x", "y"], method="pearson")

        assert corr.shape == (2, 2)
        # Should still capture the correlation
        assert corr[0, 1] > 0.7

    def test_correlation_matrix_symmetry(self):
        """Correlation matrix should be symmetric."""
        backend = LocalBackend()

        np.random.seed(42)
        df = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.normal(0, 1, 100),
        })

        corr = backend.compute_correlation(df, ["a", "b", "c"], method="spearman")

        # Verify symmetry: corr[i,j] == corr[j,i]
        np.testing.assert_array_almost_equal(corr, corr.T, decimal=10)

    def test_correlation_detects_negative_correlation(self):
        """Correlation should detect negative relationships."""
        backend = LocalBackend()

        np.random.seed(42)
        n = 300
        x = np.random.normal(0, 1, n)
        y = -0.9 * x + 0.1 * np.random.normal(0, 1, n)  # Strong negative correlation
        df = pd.DataFrame({"x": x, "y": y})

        corr = backend.compute_correlation(df, ["x", "y"], method="spearman")

        # Should detect negative correlation
        assert corr[0, 1] < -0.7

    def test_correlation_handles_missing_values(self):
        """LocalBackend correlation handles NaN values correctly."""
        backend = LocalBackend()

        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, np.nan, 5.0],
            "y": [2.0, 4.0, 6.0, 8.0, np.nan],
        })

        # Should not crash, and should compute on available pairs
        corr = backend.compute_correlation(df, ["x", "y"], method="pearson")

        assert corr.shape == (2, 2)
        # With perfect linear data (ignoring NaN), correlation should be high
        assert corr[0, 1] > 0.9


class TestLocalBackendHistogram:
    """Tests for LocalBackend histogram computation."""

    @pytest.fixture
    def histogram_data(self):
        """Generate data for histogram testing."""
        np.random.seed(42)
        return pd.DataFrame({"value": np.random.normal(50, 10, 1000)})

    def test_compute_histogram_shape(self, histogram_data):
        """LocalBackend.compute_histogram returns correct shape."""
        backend = LocalBackend()

        bin_edges = np.linspace(0, 100, 21)
        bin_counts, total = backend.compute_histogram(histogram_data, "value", bin_edges)

        assert len(bin_counts) == 20
        assert total == len(histogram_data)

    def test_compute_histogram_captures_distribution(self, histogram_data):
        """LocalBackend.compute_histogram captures distribution shape."""
        backend = LocalBackend()

        # Use bins covering full range
        bin_edges = np.linspace(0, 100, 13)  # 12 bins
        bin_counts, total = backend.compute_histogram(histogram_data, "value", bin_edges)

        # Center bins (around 50) should have more counts than edge bins (normal distribution)
        center_bins = bin_counts[5:7]  # Bins around 50
        edge_bins = np.concatenate([bin_counts[:2], bin_counts[-2:]])

        assert sum(center_bins) > sum(edge_bins)

    def test_histogram_handles_data_at_boundaries(self):
        """Histogram correctly handles data exactly at bin boundaries."""
        backend = LocalBackend()

        # Data with values exactly at bin edges
        df = pd.DataFrame({"value": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
        bin_edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])

        bin_counts, total = backend.compute_histogram(df, "value", bin_edges)

        # All values should be counted
        assert total == 6
        assert sum(bin_counts) == 6


class TestLocalBackendGenerateSamples:
    """Tests for LocalBackend sample generation."""

    def test_generate_samples_shape(self):
        """LocalBackend.generate_samples returns correct number of samples."""
        backend = LocalBackend()

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"col1": rng.normal(0, 1, n_samples), "col2": rng.normal(0, 1, n_samples)}

        result = backend.generate_samples(
            n=100,
            generator_func=generator,
            column_names=["col1", "col2"],
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert list(result.columns) == ["col1", "col2"]

    def test_generate_samples_reproducibility(self):
        """LocalBackend.generate_samples is reproducible with same seed."""
        backend = LocalBackend()

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(0, 1, n_samples)}

        result1 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"], random_seed=42
        )
        result2 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"], random_seed=42
        )

        np.testing.assert_array_almost_equal(result1["value"].values, result2["value"].values)

    def test_generate_samples_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        backend = LocalBackend()

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(0, 1, n_samples)}

        result1 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"], random_seed=42
        )
        result2 = backend.generate_samples(
            n=50, generator_func=generator, column_names=["value"], random_seed=999
        )

        # Values should be different (not identical)
        assert not np.allclose(result1["value"].values, result2["value"].values)


class TestHistogramComputerWithLocalBackend:
    """Tests for HistogramComputer with LocalBackend."""

    def test_histogram_computer_with_local_backend(self):
        """HistogramComputer works with LocalBackend."""
        from spark_bestfit.histogram import HistogramComputer

        backend = LocalBackend()
        computer = HistogramComputer(backend)

        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.normal(50, 10, 500)})

        y_hist, bin_edges = computer.compute_histogram(df, "value", bins=20)

        assert len(y_hist) == 20
        assert len(bin_edges) == 21
        # Histogram should be normalized (area approximately 1)
        bin_widths = np.diff(bin_edges)
        area = np.sum(y_hist * bin_widths)
        assert abs(area - 1.0) < 0.01


class TestCopulaWithLocalBackend:
    """Tests for GaussianCopula with LocalBackend."""

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

    def test_copula_correlation_with_local_backend(self, copula_marginals):
        """GaussianCopula can use LocalBackend for correlation computation."""
        from spark_bestfit.copula import GaussianCopula

        backend = LocalBackend()

        # Create correlated pandas data
        np.random.seed(42)
        n = 200
        x = np.random.normal(50, 10, n)
        y = 0.6 * x + 0.8 * np.random.normal(100, 20, n)
        df = pd.DataFrame({"col1": x, "col2": y})

        # Compute correlation using backend
        corr_matrix = backend.compute_correlation(df, ["col1", "col2"], method="spearman")

        # Create copula directly
        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        assert copula.column_names == ["col1", "col2"]
        assert copula.correlation_matrix.shape == (2, 2)
        # Correlation should capture positive relationship
        assert copula.correlation_matrix[0, 1] > 0.3

    def test_copula_sample_distributed_local(self, copula_marginals):
        """GaussianCopula.sample_distributed works with LocalBackend."""
        from spark_bestfit.copula import GaussianCopula

        backend = LocalBackend()

        # Create correlation matrix
        np.random.seed(42)
        corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        # Generate samples
        samples_df = copula.sample_distributed(n=100, backend=backend, random_seed=42)

        assert isinstance(samples_df, pd.DataFrame)
        assert len(samples_df) == 100
        assert list(samples_df.columns) == ["col1", "col2"]

    def test_copula_sample_statistics(self, copula_marginals):
        """GaussianCopula samples have correct marginal statistics."""
        from spark_bestfit.copula import GaussianCopula

        backend = LocalBackend()

        # Create copula with known parameters
        corr_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        # Generate many samples
        samples_df = copula.sample_distributed(n=1000, backend=backend, random_seed=42)

        # col1 marginal is norm(50, 10), col2 is norm(100, 20)
        assert abs(samples_df["col1"].mean() - 50) < 3.0
        assert abs(samples_df["col1"].std() - 10) < 3.0
        assert abs(samples_df["col2"].mean() - 100) < 5.0
        assert abs(samples_df["col2"].std() - 20) < 5.0

    def test_copula_sample_preserves_correlation(self, copula_marginals):
        """GaussianCopula samples preserve correlation structure."""
        from spark_bestfit.copula import GaussianCopula

        backend = LocalBackend()

        # Create copula with known high correlation
        corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        # Generate samples
        samples_df = copula.sample_distributed(n=1000, backend=backend, random_seed=42)

        # Verify samples have similar correlation to the input
        sample_corr = samples_df[["col1", "col2"]].corr(method="spearman").values[0, 1]
        assert abs(sample_corr - 0.8) < 0.15  # Should be close to 0.8


class TestSamplingWithLocalBackend:
    """Tests for sampling module with LocalBackend."""

    def test_sample_distributed_local_shape(self):
        """sample_distributed with LocalBackend returns correct shape."""
        from spark_bestfit.sampling import sample_distributed

        backend = LocalBackend()
        result = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=100,
            backend=backend,
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert "sample" in result.columns

    def test_sample_distributed_local_custom_column(self):
        """sample_distributed respects custom column name."""
        from spark_bestfit.sampling import sample_distributed

        backend = LocalBackend()
        result = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=50,
            backend=backend,
            column_name="my_samples",
            random_seed=42,
        )

        assert "my_samples" in result.columns
        assert "sample" not in result.columns

    def test_sample_distributed_local_statistics(self):
        """sample_distributed produces samples with correct statistics."""
        from spark_bestfit.sampling import sample_distributed

        backend = LocalBackend()
        result = sample_distributed(
            distribution="norm",
            parameters=[100.0, 15.0],  # mean=100, std=15
            n=1000,
            backend=backend,
            random_seed=42,
        )

        # Mean should be close to 100
        assert abs(result["sample"].mean() - 100) < 5.0
        # Std should be close to 15
        assert abs(result["sample"].std() - 15) < 3.0

    def test_sample_distributed_local_reproducibility(self):
        """sample_distributed is reproducible with same seed."""
        from spark_bestfit.sampling import sample_distributed

        backend = LocalBackend()

        result1 = sample_distributed(
            distribution="expon",
            parameters=[0.0, 2.0],
            n=50,
            backend=backend,
            random_seed=42,
        )
        result2 = sample_distributed(
            distribution="expon",
            parameters=[0.0, 2.0],
            n=50,
            backend=backend,
            random_seed=42,
        )

        np.testing.assert_array_almost_equal(
            result1["sample"].values, result2["sample"].values
        )

    def test_sample_distributed_different_distributions(self):
        """sample_distributed works with various scipy distributions."""
        from spark_bestfit.sampling import sample_distributed

        backend = LocalBackend()

        # Test exponential
        result_exp = sample_distributed(
            distribution="expon",
            parameters=[0.0, 1.0],
            n=100,
            backend=backend,
            random_seed=42,
        )
        assert len(result_exp) == 100
        assert result_exp["sample"].min() >= 0  # Exponential is non-negative

        # Test uniform
        result_unif = sample_distributed(
            distribution="uniform",
            parameters=[0.0, 10.0],  # loc=0, scale=10 -> U(0,10)
            n=100,
            backend=backend,
            random_seed=42,
        )
        assert len(result_unif) == 100
        assert result_unif["sample"].min() >= 0
        assert result_unif["sample"].max() <= 10


class TestProgressCallbackLocal:
    """Tests for progress_callback with LocalBackend."""

    def test_progress_callback(self, normal_data, histogram):
        """LocalBackend invokes progress callback with correct values."""
        backend = LocalBackend(max_workers=2)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma"]
        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=on_progress,
        )

        # Should have exactly len(distributions) callback calls
        assert len(progress_calls) == len(distributions)

        # Verify progress values are correct
        for i, (completed, total, percent) in enumerate(progress_calls):
            assert total == len(distributions)
            # Can't guarantee order of completion due to parallelism,
            # but all should be in valid range
            assert 1 <= completed <= len(distributions)
            assert 0 < percent <= 100

        # Last callback should show 100%
        last_completed = max(call[0] for call in progress_calls)
        assert last_completed == len(distributions)

    def test_progress_callback_error_handling(self, normal_data, histogram):
        """LocalBackend handles callback errors gracefully."""
        backend = LocalBackend(max_workers=2)

        def failing_callback(completed, total, percent):
            raise ValueError("Intentional callback error")

        # Should not raise despite callback errors
        distributions = ["norm", "expon"]
        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=failing_callback,
        )

        # Fitting should still complete successfully
        assert len(results) > 0

    def test_no_callback(self, normal_data, histogram):
        """LocalBackend works fine without progress callback."""
        backend = LocalBackend(max_workers=2)

        distributions = ["norm", "expon"]
        results = backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=None,
        )

        assert len(results) > 0

    def test_fitter_progress_callback_local(self, normal_data):
        """DistributionFitter progress callback works with LocalBackend."""
        from spark_bestfit import DistributionFitter, LocalBackend

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        df = pd.DataFrame({"value": normal_data})
        results = fitter.fit(
            df,
            column="value",
            max_distributions=5,
            progress_callback=on_progress,
        )

        # Should have received progress calls
        assert len(progress_calls) > 0
        # Should have results
        assert len(results.best(n=1)) == 1

    def test_progress_callback_empty_distributions(self, normal_data, histogram):
        """Progress callback handles empty distribution list gracefully."""
        backend = LocalBackend(max_workers=2)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        results = backend.parallel_fit(
            distributions=[],  # Empty list
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=on_progress,
        )

        # Should return empty results, no callback invocations
        assert len(results) == 0
        assert len(progress_calls) == 0

    def test_progress_callback_single_distribution(self, normal_data, histogram):
        """Progress callback works with single distribution."""
        backend = LocalBackend(max_workers=2)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        results = backend.parallel_fit(
            distributions=["norm"],  # Single distribution
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=on_progress,
        )

        # Should have exactly one callback call with 100%
        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1, 100.0)
        assert len(results) == 1

    def test_progress_strictly_increasing(self, normal_data, histogram):
        """LocalBackend progress shows strictly increasing completed count."""
        backend = LocalBackend(max_workers=2)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma", "uniform", "beta"]
        backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=on_progress,
        )

        # Verify completed count is strictly increasing
        completed_values = [call[0] for call in progress_calls]
        for i in range(1, len(completed_values)):
            assert completed_values[i] == completed_values[i - 1] + 1

        # Verify invariants on all calls
        for completed, total, percent in progress_calls:
            assert 1 <= completed <= total
            assert 0 < percent <= 100
            assert total == len(distributions)
            assert abs(percent - (completed / total * 100)) < 0.01

    def test_progress_callback_bounds_invariants(self, normal_data, histogram):
        """Progress callback values are always within valid bounds."""
        backend = LocalBackend(max_workers=2)

        progress_calls = []

        def on_progress(completed, total, percent):
            # Validate invariants inside callback
            assert isinstance(completed, int)
            assert isinstance(total, int)
            assert isinstance(percent, float)
            assert completed >= 1
            assert completed <= total
            assert percent > 0
            assert percent <= 100
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma"]
        backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=on_progress,
        )

        # If we get here, all invariants held
        assert len(progress_calls) == len(distributions)

    def test_discrete_fitter_progress_callback_local(self):
        """DiscreteDistributionFitter progress callback works with LocalBackend."""
        from spark_bestfit import DiscreteDistributionFitter, LocalBackend

        np.random.seed(42)
        count_data = np.random.poisson(lam=10, size=500)

        backend = LocalBackend(max_workers=2)
        fitter = DiscreteDistributionFitter(backend=backend)

        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        df = pd.DataFrame({"counts": count_data})
        results = fitter.fit(
            df,
            column="counts",
            max_distributions=3,
            progress_callback=on_progress,
        )

        # Should have received progress calls
        assert len(progress_calls) > 0
        # Should have results
        assert len(results.best(n=1, metric="aic")) == 1
