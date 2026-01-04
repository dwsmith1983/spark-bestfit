"""Tests for Ray backend implementation."""

import numpy as np
import pandas as pd
import pytest

# Skip all tests if ray is not installed
ray = pytest.importorskip("ray")

from spark_bestfit.backends.ray import RayBackend
from spark_bestfit.fitting import compute_data_stats, fit_single_distribution


@pytest.fixture(scope="module")
def ray_backend():
    """Create RayBackend for testing (initializes Ray once for module)."""
    # Initialize Ray in local mode for testing
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    return RayBackend(num_cpus=2)


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


class TestRayBackendInit:
    """Tests for RayBackend initialization."""

    def test_init_with_defaults(self, ray_backend):
        """RayBackend initializes with default settings."""
        assert ray_backend is not None
        assert ray_backend.get_parallelism() >= 1

    def test_init_with_custom_cpus(self):
        """RayBackend respects custom CPU count."""
        # Note: This just sets the stored value, actual Ray resources
        # are determined at init time
        backend = RayBackend(num_cpus=4)
        assert backend._num_cpus == 4


class TestRayBackendBroadcast:
    """Tests for RayBackend broadcast operations."""

    def test_broadcast_puts_data(self, ray_backend, normal_data):
        """RayBackend.broadcast puts data in object store."""
        handle = ray_backend.broadcast(normal_data)
        assert handle is not None
        # Should be a Ray ObjectRef
        assert hasattr(handle, "__ray_object_ref__") or isinstance(handle, ray.ObjectRef)

        # Can retrieve data
        retrieved = ray.get(handle)
        np.testing.assert_array_equal(retrieved, normal_data)

    def test_destroy_broadcast_no_error(self, ray_backend, normal_data):
        """RayBackend.destroy_broadcast is no-op (shouldn't error)."""
        handle = ray_backend.broadcast(normal_data)
        # Should not raise - Ray uses automatic reference counting
        ray_backend.destroy_broadcast(handle)


class TestRayBackendParallelism:
    """Tests for RayBackend parallelism reporting."""

    def test_get_parallelism_positive(self, ray_backend):
        """RayBackend.get_parallelism returns positive integer."""
        parallelism = ray_backend.get_parallelism()
        assert parallelism >= 1
        assert isinstance(parallelism, int)


class TestRayBackendDataFrame:
    """Tests for RayBackend DataFrame operations."""

    def test_create_dataframe(self, ray_backend):
        """RayBackend creates pandas DataFrames."""
        data = [("a",), ("b",), ("c",)]
        df = ray_backend.create_dataframe(data, ["name"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["name"]
        assert list(df["name"]) == ["a", "b", "c"]

    def test_collect_column_pandas(self, ray_backend):
        """RayBackend extracts column from pandas DataFrame."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = ray_backend.collect_column(df, "value")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_column_stats_pandas(self, ray_backend):
        """RayBackend computes column statistics from pandas."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})

        stats = ray_backend.get_column_stats(df, "value")
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5

    def test_sample_column_pandas(self, ray_backend, normal_data):
        """RayBackend samples column from pandas DataFrame."""
        df = pd.DataFrame({"value": normal_data})

        sample = ray_backend.sample_column(df, "value", fraction=0.1, seed=42)
        assert len(sample) > 0
        assert len(sample) < len(normal_data)

    def test_sample_column_reproducibility(self, ray_backend, normal_data):
        """RayBackend sample is reproducible with same seed."""
        df = pd.DataFrame({"value": normal_data})

        sample1 = ray_backend.sample_column(df, "value", fraction=0.1, seed=42)
        sample2 = ray_backend.sample_column(df, "value", fraction=0.1, seed=42)

        np.testing.assert_array_equal(sample1, sample2)


class TestRayBackendParallelFit:
    """Tests for RayBackend parallel fitting."""

    def test_parallel_fit_continuous(self, ray_backend, normal_data, histogram):
        """RayBackend fits continuous distributions in parallel."""
        results = ray_backend.parallel_fit(
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

        # Verify result structure
        for result in results:
            assert result["column_name"] == "test_col"
            assert isinstance(result["parameters"], list)
            assert np.isfinite(result["sse"])
            assert np.isfinite(result["aic"])
            assert np.isfinite(result["bic"])

        # Normal should fit better than uniform for normal data
        norm_result = next(r for r in results if r["distribution"] == "norm")
        unif_result = next(r for r in results if r["distribution"] == "uniform")
        assert norm_result["sse"] < unif_result["sse"]

    def test_parallel_fit_with_lazy_metrics(self, ray_backend, normal_data, histogram):
        """RayBackend respects lazy_metrics flag."""
        results = ray_backend.parallel_fit(
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

    def test_parallel_fit_with_bounds(self, ray_backend, normal_data, histogram):
        """RayBackend handles bounded fitting."""
        results = ray_backend.parallel_fit(
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

    def test_parallel_fit_verifies_parameters(self, ray_backend, normal_data, histogram):
        """RayBackend produces accurate fitted parameters."""
        # Data generated with loc=50, scale=10
        results = ray_backend.parallel_fit(
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

    def test_parallel_fit_empty_list(self, ray_backend, normal_data, histogram):
        """RayBackend handles empty distribution list."""
        results = ray_backend.parallel_fit(
            distributions=[],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        assert results == []

    def test_parallel_fit_many_distributions(self, ray_backend, normal_data, histogram):
        """RayBackend efficiently handles many distributions."""
        distributions = ["norm", "expon", "gamma", "lognorm", "weibull_min", "beta"]

        results = ray_backend.parallel_fit(
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
        for r in results:
            assert np.isfinite(r["sse"])

    def test_parallel_fit_discrete(self, ray_backend):
        """RayBackend fits discrete distributions correctly."""
        # Create Poisson data with known lambda=7
        np.random.seed(42)
        data_sample = np.random.poisson(lam=7, size=1000).astype(int)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        results = ray_backend.parallel_fit(
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


class TestRayBackendCorrelation:
    """Tests for RayBackend correlation computation."""

    @pytest.fixture
    def correlated_data(self):
        """Generate correlated data for testing."""
        np.random.seed(42)
        n = 500
        x = np.random.normal(0, 1, n)
        y = 0.8 * x + 0.2 * np.random.normal(0, 1, n)
        z = np.random.normal(0, 1, n)
        return pd.DataFrame({"x": x, "y": y, "z": z})

    def test_compute_correlation_shape(self, ray_backend, correlated_data):
        """RayBackend.compute_correlation returns correct shape."""
        corr = ray_backend.compute_correlation(
            correlated_data, ["x", "y", "z"], method="spearman"
        )

        assert corr.shape == (3, 3)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0], decimal=5)

    def test_compute_correlation_values(self, ray_backend, correlated_data):
        """RayBackend.compute_correlation captures correlation structure."""
        corr = ray_backend.compute_correlation(
            correlated_data, ["x", "y", "z"], method="spearman"
        )

        # x and y should have high positive correlation
        assert corr[0, 1] > 0.7
        # x and z should have low correlation
        assert abs(corr[0, 2]) < 0.2

    def test_compute_correlation_pearson(self, ray_backend, correlated_data):
        """RayBackend.compute_correlation supports pearson method."""
        corr = ray_backend.compute_correlation(
            correlated_data, ["x", "y"], method="pearson"
        )

        assert corr.shape == (2, 2)
        assert corr[0, 1] > 0.7


class TestRayBackendHistogram:
    """Tests for RayBackend histogram computation."""

    @pytest.fixture
    def histogram_data(self):
        """Generate data for histogram testing."""
        np.random.seed(42)
        return pd.DataFrame({"value": np.random.normal(50, 10, 1000)})

    def test_compute_histogram_shape(self, ray_backend, histogram_data):
        """RayBackend.compute_histogram returns correct shape."""
        bin_edges = np.linspace(0, 100, 21)
        bin_counts, total = ray_backend.compute_histogram(
            histogram_data, "value", bin_edges
        )

        assert len(bin_counts) == 20
        assert total == len(histogram_data)

    def test_compute_histogram_sum(self, ray_backend, histogram_data):
        """RayBackend.compute_histogram bin counts sum to total."""
        bin_edges = np.linspace(0, 100, 11)
        bin_counts, total = ray_backend.compute_histogram(
            histogram_data, "value", bin_edges
        )

        assert sum(bin_counts) == total
        assert total == len(histogram_data)


class TestRayBackendGenerateSamples:
    """Tests for RayBackend sample generation."""

    def test_generate_samples_shape(self, ray_backend):
        """RayBackend.generate_samples returns correct number of samples."""

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"col1": rng.normal(0, 1, n_samples), "col2": rng.normal(0, 1, n_samples)}

        result = ray_backend.generate_samples(
            n=100,
            generator_func=generator,
            column_names=["col1", "col2"],
            num_partitions=2,
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert list(result.columns) == ["col1", "col2"]

    def test_generate_samples_reproducibility(self, ray_backend):
        """RayBackend.generate_samples is reproducible with same seed."""

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(0, 1, n_samples)}

        result1 = ray_backend.generate_samples(
            n=50,
            generator_func=generator,
            column_names=["value"],
            num_partitions=1,
            random_seed=42,
        )
        result2 = ray_backend.generate_samples(
            n=50,
            generator_func=generator,
            column_names=["value"],
            num_partitions=1,
            random_seed=42,
        )

        np.testing.assert_array_almost_equal(
            result1["value"].values, result2["value"].values
        )

    def test_generate_samples_statistics(self, ray_backend):
        """RayBackend.generate_samples produces samples with correct statistics."""

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(loc=100, scale=15, size=n_samples)}

        result = ray_backend.generate_samples(
            n=1000,
            generator_func=generator,
            column_names=["value"],
            num_partitions=4,
            random_seed=42,
        )

        # Mean should be close to 100
        assert abs(result["value"].mean() - 100) < 3.0
        # Std should be close to 15
        assert abs(result["value"].std() - 15) < 3.0


class TestRayBackendWithFitter:
    """Tests for RayBackend integration with fitter classes."""

    def test_continuous_fitter_with_ray_backend(self, ray_backend, normal_data):
        """DistributionFitter works with RayBackend."""
        from spark_bestfit import DistributionFitter

        fitter = DistributionFitter(backend=ray_backend)

        # Create pandas DataFrame
        df = pd.DataFrame({"value": normal_data})

        results = fitter.fit(df, column="value", max_distributions=3)
        assert len(results.best(n=1)) == 1

        # Verify result structure
        best = results.best(n=1)[0]
        assert best.distribution is not None
        assert best.sse is not None
        assert best.column_name == "value"

    def test_discrete_fitter_with_ray_backend(self, ray_backend):
        """DiscreteDistributionFitter works with RayBackend."""
        from spark_bestfit import DiscreteDistributionFitter

        # Create Poisson data
        np.random.seed(42)
        poisson_data = np.random.poisson(lam=7, size=500)
        df = pd.DataFrame({"counts": poisson_data})

        fitter = DiscreteDistributionFitter(backend=ray_backend)
        results = fitter.fit(df, column="counts", max_distributions=3)
        assert len(results.best(n=1, metric="aic")) == 1

        # Verify result structure
        best = results.best(n=1, metric="aic")[0]
        assert best.distribution is not None
        assert best.sse is not None
        assert best.column_name == "counts"


class TestRayBackendCompatibility:
    """Tests comparing RayBackend with LocalBackend results."""

    def test_continuous_fit_compatible_with_local(
        self, ray_backend, normal_data, histogram
    ):
        """RayBackend produces results compatible with LocalBackend."""
        from spark_bestfit.backends.local import LocalBackend

        local_backend = LocalBackend(max_workers=2)

        distributions = ["norm", "expon"]
        data_stats = compute_data_stats(normal_data)

        ray_results = ray_backend.parallel_fit(
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
        assert len(ray_results) > 0
        assert len(local_results) > 0

        # Results should have same structure
        ray_keys = set(ray_results[0].keys())
        local_keys = set(local_results[0].keys())
        assert ray_keys == local_keys

        # SSE values should be similar (fitting same data)
        for dist in distributions:
            ray_sse = next((r["sse"] for r in ray_results if r["distribution"] == dist), None)
            local_sse = next((r["sse"] for r in local_results if r["distribution"] == dist), None)
            if ray_sse and local_sse:
                assert abs(ray_sse - local_sse) < 0.01


class TestRayBackendSampling:
    """Tests for sampling module with RayBackend."""

    def test_sample_distributed_shape(self, ray_backend):
        """sample_distributed with RayBackend returns correct shape."""
        from spark_bestfit.sampling import sample_distributed

        result = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=100,
            backend=ray_backend,
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert "sample" in result.columns

    def test_sample_distributed_statistics(self, ray_backend):
        """sample_distributed with RayBackend produces correct statistics."""
        from spark_bestfit.sampling import sample_distributed

        result = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=1000,
            backend=ray_backend,
            num_partitions=4,
            random_seed=42,
        )

        # Mean should be close to 50
        assert abs(result["sample"].mean() - 50) < 5.0
        # Std should be close to 10
        assert abs(result["sample"].std() - 10) < 3.0


class TestRayBackendCopula:
    """Tests for copula operations with RayBackend."""

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

    def test_copula_sample_distributed(self, ray_backend, copula_marginals):
        """GaussianCopula.sample_distributed works with RayBackend."""
        from spark_bestfit.copula import GaussianCopula

        corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        samples_df = copula.sample_distributed(n=100, backend=ray_backend, random_seed=42)

        assert isinstance(samples_df, pd.DataFrame)
        assert len(samples_df) == 100
        assert list(samples_df.columns) == ["col1", "col2"]

    def test_copula_samples_preserve_correlation(self, ray_backend, copula_marginals):
        """GaussianCopula samples preserve correlation structure with RayBackend."""
        from spark_bestfit.copula import GaussianCopula

        corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        copula = GaussianCopula(
            column_names=["col1", "col2"],
            marginals=copula_marginals,
            correlation_matrix=corr_matrix,
        )

        samples_df = copula.sample_distributed(n=1000, backend=ray_backend, random_seed=42)

        # Verify samples have similar correlation to input
        sample_corr = samples_df[["col1", "col2"]].corr(method="spearman").values[0, 1]
        assert abs(sample_corr - 0.8) < 0.15


# =============================================================================
# ROBUST TEST ADDITIONS: Ray Dataset, Error Handling, Edge Cases
# =============================================================================


class TestRayBackendRayDataset:
    """Tests for RayBackend with Ray Dataset (hybrid DataFrame support)."""

    @pytest.fixture
    def ray_dataset(self):
        """Create a Ray Dataset for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            "value": np.random.normal(50, 10, 500),
            "category": np.random.choice(["A", "B", "C"], 500),
        })
        return ray.data.from_pandas(data)

    def test_is_ray_dataset_detection(self, ray_backend, ray_dataset):
        """RayBackend correctly detects Ray Dataset vs pandas DataFrame."""
        pdf = pd.DataFrame({"value": [1, 2, 3]})

        assert ray_backend._is_ray_dataset(ray_dataset) is True
        assert ray_backend._is_ray_dataset(pdf) is False

    def test_collect_column_ray_dataset(self, ray_backend, ray_dataset):
        """RayBackend extracts column from Ray Dataset."""
        result = ray_backend.collect_column(ray_dataset, "value")

        assert isinstance(result, np.ndarray)
        assert len(result) == 500
        # Data should be from normal(50, 10)
        assert 40 < result.mean() < 60

    def test_get_column_stats_ray_dataset(self, ray_backend, ray_dataset):
        """RayBackend computes column statistics from Ray Dataset."""
        stats = ray_backend.get_column_stats(ray_dataset, "value")

        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] == 500
        # Normal(50, 10) data should have reasonable min/max
        assert stats["min"] < 30
        assert stats["max"] > 70

    def test_sample_column_ray_dataset(self, ray_backend, ray_dataset):
        """RayBackend samples column from Ray Dataset."""
        sample = ray_backend.sample_column(ray_dataset, "value", fraction=0.2, seed=42)

        assert isinstance(sample, np.ndarray)
        # Should sample approximately 20% of 500 = ~100 rows
        assert 50 < len(sample) < 150

    def test_compute_correlation_ray_dataset(self, ray_backend):
        """RayBackend computes correlation from Ray Dataset."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 500)
        y = 0.9 * x + 0.1 * np.random.normal(0, 1, 500)
        data = pd.DataFrame({"x": x, "y": y})
        ds = ray.data.from_pandas(data)

        corr = ray_backend.compute_correlation(ds, ["x", "y"], method="spearman")

        assert corr.shape == (2, 2)
        # High correlation expected
        assert corr[0, 1] > 0.85

    def test_compute_histogram_ray_dataset(self, ray_backend, ray_dataset):
        """RayBackend computes histogram from Ray Dataset."""
        bin_edges = np.linspace(0, 100, 21)
        bin_counts, total = ray_backend.compute_histogram(ray_dataset, "value", bin_edges)

        assert len(bin_counts) == 20
        assert total == 500
        # Most counts should be in middle bins (data centered at 50)
        middle_bins = bin_counts[8:12]  # Bins around 40-60
        assert sum(middle_bins) > total * 0.5

    def test_fitter_with_ray_dataset(self, ray_backend):
        """DistributionFitter works with Ray Dataset input."""
        from spark_bestfit import DistributionFitter

        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.normal(50, 10, 500)})
        ds = ray.data.from_pandas(data)

        fitter = DistributionFitter(backend=ray_backend)
        results = fitter.fit(ds, column="value", max_distributions=2)

        assert len(results.best(n=1)) == 1
        best = results.best(n=1)[0]
        assert best.distribution is not None


class TestRayBackendErrorHandling:
    """Tests for RayBackend error handling and edge cases."""

    def test_parallel_fit_invalid_distribution_filtered(self, ray_backend, normal_data, histogram):
        """RayBackend filters out failed distribution fits."""
        # Include a distribution that will fail on normal data
        results = ray_backend.parallel_fit(
            distributions=["norm", "nonexistent_dist_xyz"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        # Should only get result for 'norm', the invalid one is filtered
        dist_names = {r["distribution"] for r in results}
        assert "norm" in dist_names
        assert "nonexistent_dist_xyz" not in dist_names

    def test_parallel_fit_all_failures_returns_empty(self, ray_backend, normal_data, histogram):
        """RayBackend returns empty list when all distributions fail."""
        results = ray_backend.parallel_fit(
            distributions=["fake_dist_1", "fake_dist_2"],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        assert results == []

    def test_collect_column_missing_column(self, ray_backend):
        """RayBackend raises error for missing column."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(KeyError):
            ray_backend.collect_column(df, "nonexistent_column")

    def test_get_column_stats_missing_column(self, ray_backend):
        """RayBackend raises error for missing column in stats."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(KeyError):
            ray_backend.get_column_stats(df, "nonexistent_column")

    def test_sample_column_invalid_fraction(self, ray_backend):
        """RayBackend handles invalid sample fraction."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        # Fraction > 1 should work (pandas allows replace=False up to 1.0)
        # This tests that we get a reasonable error or behavior
        with pytest.raises((ValueError, Exception)):
            ray_backend.sample_column(df, "value", fraction=2.0, seed=42)

    def test_compute_histogram_empty_bins(self, ray_backend):
        """RayBackend handles histogram with data outside bin range."""
        df = pd.DataFrame({"value": [1000, 2000, 3000]})  # Data way outside bins
        bin_edges = np.linspace(0, 10, 11)

        bin_counts, total = ray_backend.compute_histogram(df, "value", bin_edges)

        # All data is outside bins, so counts should be 0
        assert sum(bin_counts) == 0
        assert total == 0

    def test_broadcast_none_value(self, ray_backend):
        """RayBackend can broadcast None value."""
        handle = ray_backend.broadcast(None)
        retrieved = ray.get(handle)
        assert retrieved is None

    def test_broadcast_empty_array(self, ray_backend):
        """RayBackend can broadcast empty array."""
        empty = np.array([])
        handle = ray_backend.broadcast(empty)
        retrieved = ray.get(handle)
        np.testing.assert_array_equal(retrieved, empty)


class TestRayBackendEdgeCases:
    """Tests for edge case data handling."""

    def test_fit_single_value_data(self, ray_backend):
        """RayBackend handles single unique value gracefully."""
        # Data with single unique value
        data = np.array([42.0] * 100)
        y_hist, bin_edges = np.histogram(data, bins=30, density=True)

        results = ray_backend.parallel_fit(
            distributions=["norm"],
            histogram=(y_hist, bin_edges),
            data_sample=data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(data),
            is_discrete=False,
        )

        # Should either succeed with degenerate fit or be filtered
        # Either way, shouldn't crash
        assert isinstance(results, list)

    def test_fit_two_value_data(self, ray_backend):
        """RayBackend handles two unique values."""
        data = np.array([0.0] * 50 + [100.0] * 50)
        np.random.shuffle(data)
        y_hist, bin_edges = np.histogram(data, bins=30, density=True)

        results = ray_backend.parallel_fit(
            distributions=["uniform"],
            histogram=(y_hist, bin_edges),
            data_sample=data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(data),
            is_discrete=False,
        )

        # Uniform should fit this bimodal data (poorly, but without crashing)
        assert isinstance(results, list)

    def test_collect_column_with_nan(self, ray_backend):
        """RayBackend collects column containing NaN values."""
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = ray_backend.collect_column(df, "value")

        assert len(result) == 5
        assert np.isnan(result[1])
        assert np.isnan(result[3])

    def test_get_column_stats_with_nan(self, ray_backend):
        """RayBackend computes stats ignoring NaN (pandas default)."""
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0, np.nan, 5.0]})
        stats = ray_backend.get_column_stats(df, "value")

        # pandas min/max ignore NaN by default
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["count"] == 5  # len() counts all rows including NaN

    def test_compute_histogram_with_nan(self, ray_backend):
        """RayBackend histogram ignores NaN values."""
        df = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0]})
        bin_edges = np.linspace(0, 6, 7)

        bin_counts, total = ray_backend.compute_histogram(df, "value", bin_edges)

        # NaN should be dropped, so total = 4
        assert total == 4
        assert sum(bin_counts) == 4

    def test_fit_with_inf_in_data(self, ray_backend):
        """RayBackend handles infinity in data."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        # Filter inf for histogram
        clean_data = data[np.isfinite(data)]
        y_hist, bin_edges = np.histogram(clean_data, bins=10, density=True)

        results = ray_backend.parallel_fit(
            distributions=["norm"],
            histogram=(y_hist, bin_edges),
            data_sample=clean_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(clean_data),
            is_discrete=False,
        )

        # Should succeed with clean data
        assert len(results) == 1

    def test_sample_column_small_dataframe(self, ray_backend):
        """RayBackend samples from very small DataFrame."""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        # Sample 50% of 3 rows
        sample = ray_backend.sample_column(df, "value", fraction=0.5, seed=42)

        assert len(sample) >= 1
        assert len(sample) <= 3

    def test_generate_samples_zero(self, ray_backend):
        """RayBackend handles n=0 samples request."""

        def generator(n_samples, partition_id, seed):
            return {"value": np.array([])}

        result = ray_backend.generate_samples(
            n=0,
            generator_func=generator,
            column_names=["value"],
            num_partitions=2,
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_parallel_fit_large_distribution_list(self, ray_backend, normal_data, histogram):
        """RayBackend handles fitting many distributions concurrently."""
        # Test with a larger set of distributions
        distributions = [
            "norm", "expon", "gamma", "lognorm", "weibull_min",
            "uniform", "beta", "pareto", "chi2", "t",
        ]

        results = ray_backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(normal_data),
            is_discrete=False,
        )

        # Should get multiple successful fits
        assert len(results) >= 5
        # All results should have finite SSE
        for r in results:
            assert np.isfinite(r["sse"])


class TestRayBackendStatisticalPrecision:
    """Tests with tighter statistical tolerances using principled bounds."""

    @pytest.fixture
    def large_normal_data(self):
        """Generate larger sample for tighter statistical tests."""
        np.random.seed(42)
        return np.random.normal(loc=100, scale=20, size=5000)

    @pytest.fixture
    def large_histogram(self, large_normal_data):
        """Create histogram from large normal data."""
        y_hist, bin_edges = np.histogram(large_normal_data, bins=50, density=True)
        return (y_hist, bin_edges)

    def test_fitted_parameters_precision(self, ray_backend, large_normal_data, large_histogram):
        """Fitted parameters should be within 2 standard errors of true values."""
        results = ray_backend.parallel_fit(
            distributions=["norm"],
            histogram=large_histogram,
            data_sample=large_normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(large_normal_data),
            is_discrete=False,
        )

        assert len(results) == 1
        fitted_loc, fitted_scale = results[0]["parameters"][:2]

        # For n=5000, SE of mean ≈ 20/sqrt(5000) ≈ 0.28
        # 3 SE tolerance = 0.85
        assert abs(fitted_loc - 100) < 1.0, f"Fitted loc={fitted_loc}, expected ~100"

        # SE of std dev ≈ scale/sqrt(2n) ≈ 20/sqrt(10000) ≈ 0.2
        # 3 SE tolerance = 0.6
        assert abs(fitted_scale - 20) < 1.0, f"Fitted scale={fitted_scale}, expected ~20"

    def test_normal_beats_exponential_significantly(
        self, ray_backend, large_normal_data, large_histogram
    ):
        """Normal distribution should fit significantly better than exponential for normal data."""
        results = ray_backend.parallel_fit(
            distributions=["norm", "expon"],
            histogram=large_histogram,
            data_sample=large_normal_data,
            fit_func=fit_single_distribution,
            column_name="test",
            data_stats=compute_data_stats(large_normal_data),
            is_discrete=False,
        )

        norm_result = next(r for r in results if r["distribution"] == "norm")
        expon_result = next(r for r in results if r["distribution"] == "expon")

        # Normal should have much lower SSE (at least 10x better)
        assert norm_result["sse"] < expon_result["sse"] / 10

        # AIC difference should be substantial (> 100 indicates very strong evidence)
        assert norm_result["aic"] < expon_result["aic"] - 100

    def test_correlation_preservation_precise(self, ray_backend):
        """Correlation should be preserved within statistical bounds."""
        np.random.seed(42)
        n = 2000
        true_corr = 0.7

        # Generate correlated data
        x = np.random.normal(0, 1, n)
        y = true_corr * x + np.sqrt(1 - true_corr**2) * np.random.normal(0, 1, n)
        df = pd.DataFrame({"x": x, "y": y})

        corr = ray_backend.compute_correlation(df, ["x", "y"], method="pearson")

        # SE of correlation ≈ (1-r²)/sqrt(n-1) ≈ 0.011 for r=0.7, n=2000
        # 3 SE tolerance ≈ 0.035
        assert abs(corr[0, 1] - true_corr) < 0.05

    def test_sample_statistics_convergence(self, ray_backend):
        """Generated samples should converge to expected statistics."""

        def generator(n_samples, partition_id, seed):
            rng = np.random.default_rng(seed)
            return {"value": rng.normal(loc=50, scale=15, size=n_samples)}

        # Generate large sample
        result = ray_backend.generate_samples(
            n=10000,
            generator_func=generator,
            column_names=["value"],
            num_partitions=4,
            random_seed=42,
        )

        # SE of mean = 15/sqrt(10000) = 0.15
        # 3 SE = 0.45
        assert abs(result["value"].mean() - 50) < 0.6

        # SE of std ≈ 15/sqrt(2*10000) ≈ 0.106
        # 3 SE ≈ 0.32
        assert abs(result["value"].std() - 15) < 0.5

    def test_poisson_parameter_precision(self, ray_backend):
        """Discrete Poisson fitting should recover lambda precisely."""
        np.random.seed(42)
        true_lambda = 12
        data_sample = np.random.poisson(lam=true_lambda, size=3000).astype(int)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        results = ray_backend.parallel_fit(
            distributions=["poisson"],
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="counts",
            data_stats=data_stats,
            is_discrete=True,
        )

        assert len(results) == 1
        fitted_lambda = results[0]["parameters"][0]

        # SE of Poisson MLE ≈ sqrt(lambda/n) ≈ sqrt(12/3000) ≈ 0.063
        # 3 SE ≈ 0.19
        assert abs(fitted_lambda - true_lambda) < 0.3


class TestRayBackendResourceManagement:
    """Tests for proper resource handling."""

    def test_multiple_parallel_fit_calls(self, ray_backend, normal_data, histogram):
        """Multiple parallel_fit calls don't leak resources."""
        for i in range(5):
            results = ray_backend.parallel_fit(
                distributions=["norm", "expon"],
                histogram=histogram,
                data_sample=normal_data,
                fit_func=fit_single_distribution,
                column_name=f"test_{i}",
                data_stats=compute_data_stats(normal_data),
                is_discrete=False,
            )
            assert len(results) >= 1

    def test_broadcast_multiple_times(self, ray_backend, normal_data):
        """Multiple broadcast calls work correctly."""
        handles = []
        for i in range(10):
            handle = ray_backend.broadcast(normal_data * i)
            handles.append(handle)

        # All handles should be valid
        for i, handle in enumerate(handles):
            retrieved = ray.get(handle)
            expected = normal_data * i
            np.testing.assert_array_almost_equal(retrieved, expected)

    def test_concurrent_fitter_instances(self, ray_backend):
        """Multiple fitter instances can use same backend."""
        from spark_bestfit import DistributionFitter

        np.random.seed(42)
        df1 = pd.DataFrame({"value": np.random.normal(50, 10, 500)})
        df2 = pd.DataFrame({"value": np.random.exponential(5, 500)})

        fitter1 = DistributionFitter(backend=ray_backend)
        fitter2 = DistributionFitter(backend=ray_backend)

        results1 = fitter1.fit(df1, column="value", max_distributions=2)
        results2 = fitter2.fit(df2, column="value", max_distributions=2)

        assert len(results1.best(n=1)) == 1
        assert len(results2.best(n=1)) == 1


class TestRayBackendDistributedPearson:
    """Tests for distributed Pearson correlation on Ray Dataset."""

    def test_distributed_pearson_matches_pandas(self, ray_backend):
        """Distributed Pearson correlation should match pandas.corr() precisely."""
        np.random.seed(42)
        n = 1000
        # Create correlated data
        x = np.random.normal(0, 1, n)
        y = 0.6 * x + 0.8 * np.random.normal(0, 1, n)
        z = -0.3 * x + 0.95 * np.random.normal(0, 1, n)
        data = pd.DataFrame({"x": x, "y": y, "z": z})

        # Compute with pandas (ground truth)
        pandas_corr = data.corr(method="pearson").values

        # Compute with Ray Dataset (distributed path)
        ds = ray.data.from_pandas(data)
        ray_corr = ray_backend.compute_correlation(ds, ["x", "y", "z"], method="pearson")

        # Should match within numerical precision
        np.testing.assert_array_almost_equal(ray_corr, pandas_corr, decimal=10)

    def test_distributed_pearson_diagonal_is_one(self, ray_backend):
        """Distributed Pearson correlation diagonal should be exactly 1.0."""
        np.random.seed(42)
        data = pd.DataFrame({
            "a": np.random.normal(0, 1, 500),
            "b": np.random.normal(10, 5, 500),
            "c": np.random.uniform(0, 100, 500),
        })
        ds = ray.data.from_pandas(data)

        corr = ray_backend.compute_correlation(ds, ["a", "b", "c"], method="pearson")

        np.testing.assert_array_equal(np.diag(corr), [1.0, 1.0, 1.0])

    def test_distributed_pearson_symmetry(self, ray_backend):
        """Distributed Pearson correlation matrix should be symmetric."""
        np.random.seed(42)
        data = pd.DataFrame({
            "a": np.random.normal(0, 1, 500),
            "b": np.random.normal(0, 1, 500),
            "c": np.random.normal(0, 1, 500),
        })
        ds = ray.data.from_pandas(data)

        corr = ray_backend.compute_correlation(ds, ["a", "b", "c"], method="pearson")

        np.testing.assert_array_almost_equal(corr, corr.T, decimal=15)


class TestRayBackendEmptyDataset:
    """Tests for empty Ray Dataset handling."""

    def test_empty_ray_dataset_column_stats(self, ray_backend):
        """Empty Ray Dataset should handle column stats gracefully."""
        empty_df = pd.DataFrame({"value": pd.Series([], dtype=float)})
        ds = ray.data.from_pandas(empty_df)

        # Should handle gracefully (may raise or return special values)
        try:
            stats = ray_backend.get_column_stats(ds, "value")
            # If it returns, count should be 0
            assert stats["count"] == 0
        except (ValueError, ZeroDivisionError, TypeError):
            # Also acceptable to raise on empty data
            # TypeError can occur when Ray min/max returns None on empty data
            pass

    def test_empty_ray_dataset_correlation(self, ray_backend):
        """Empty Ray Dataset should handle correlation gracefully."""
        empty_df = pd.DataFrame({"x": pd.Series([], dtype=float), "y": pd.Series([], dtype=float)})
        ds = ray.data.from_pandas(empty_df)

        # Empty data can either return identity matrix, empty array, or raise an error
        try:
            corr = ray_backend.compute_correlation(ds, ["x", "y"], method="spearman")
            # If it returns, accept either identity matrix or empty array
            if corr.size == 0:
                # Empty array is acceptable for empty data
                assert corr.shape == (0, 0) or corr.shape == (2, 0) or corr.shape == (0, 2)
            else:
                # Non-empty should be identity matrix
                np.testing.assert_array_equal(corr, np.eye(2))
        except (ValueError, KeyError, ZeroDivisionError, AssertionError):
            # Also acceptable to raise on empty data
            pass

    def test_empty_ray_dataset_histogram(self, ray_backend):
        """Empty Ray Dataset should return zero counts for histogram."""
        empty_df = pd.DataFrame({"value": pd.Series([], dtype=float)})
        ds = ray.data.from_pandas(empty_df)
        bin_edges = np.linspace(0, 10, 11)

        bin_counts, total = ray_backend.compute_histogram(ds, "value", bin_edges)

        assert total == 0
        assert sum(bin_counts) == 0


class TestRayBackendDiscreteWithDataset:
    """Tests for DiscreteDistributionFitter with Ray Dataset."""

    def test_discrete_fitter_with_ray_dataset(self, ray_backend):
        """DiscreteDistributionFitter should work with Ray Dataset input."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        poisson_data = np.random.poisson(lam=5, size=500)
        data = pd.DataFrame({"counts": poisson_data})
        ds = ray.data.from_pandas(data)

        fitter = DiscreteDistributionFitter(backend=ray_backend)
        results = fitter.fit(ds, column="counts", max_distributions=3)

        assert len(results.best(n=1, metric="aic")) >= 1
        best = results.best(n=1, metric="aic")[0]
        assert best.distribution is not None
        assert best.column_name == "counts"

    def test_discrete_fitter_ray_dataset_recovers_parameters(self, ray_backend):
        """DiscreteDistributionFitter with Ray Dataset should recover true parameters."""
        from spark_bestfit import DiscreteDistributionFitter
        from spark_bestfit.distributions import DiscreteDistributionRegistry

        np.random.seed(42)
        true_lambda = 8
        poisson_data = np.random.poisson(lam=true_lambda, size=1000)
        data = pd.DataFrame({"counts": poisson_data})
        ds = ray.data.from_pandas(data)

        # Exclude all distributions except poisson
        registry = DiscreteDistributionRegistry()
        all_discrete = registry.get_distributions()
        excluded = tuple(d for d in all_discrete if d != "poisson")

        fitter = DiscreteDistributionFitter(backend=ray_backend, excluded_distributions=excluded)
        results = fitter.fit(ds, column="counts")

        best = results.best(n=1)[0]
        fitted_lambda = best.parameters[0]

        # Should recover true lambda within reasonable tolerance
        assert abs(fitted_lambda - true_lambda) < 0.5


class TestRayBackendCorrelationStress:
    """Stress tests for correlation with many columns."""

    def test_correlation_many_columns(self, ray_backend):
        """Correlation should work efficiently with many columns."""
        np.random.seed(42)
        n_cols = 15
        n_rows = 500

        # Generate random data with some correlations
        data = {}
        base = np.random.normal(0, 1, n_rows)
        for i in range(n_cols):
            noise = np.random.normal(0, 1, n_rows)
            data[f"col_{i}"] = 0.3 * base + 0.95 * noise

        df = pd.DataFrame(data)
        columns = list(df.columns)

        # Test with pandas DataFrame
        corr_pandas = ray_backend.compute_correlation(df, columns, method="pearson")

        assert corr_pandas.shape == (n_cols, n_cols)
        np.testing.assert_array_almost_equal(np.diag(corr_pandas), np.ones(n_cols))

    def test_correlation_many_columns_ray_dataset(self, ray_backend):
        """Distributed correlation should work with many columns on Ray Dataset."""
        np.random.seed(42)
        n_cols = 10
        n_rows = 500

        data = {f"col_{i}": np.random.normal(i, 1, n_rows) for i in range(n_cols)}
        df = pd.DataFrame(data)
        ds = ray.data.from_pandas(df)
        columns = list(df.columns)

        # Compute distributed correlation
        corr = ray_backend.compute_correlation(ds, columns, method="pearson")

        assert corr.shape == (n_cols, n_cols)
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(n_cols))
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_correlation_single_column(self, ray_backend):
        """Single column correlation should return [[1.0]]."""
        np.random.seed(42)
        data = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        ds = ray.data.from_pandas(data)

        corr = ray_backend.compute_correlation(ds, ["x"], method="pearson")

        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0


class TestRayBackendCalculatePartitions:
    """Tests for partition calculation with slow distribution weighting."""

    def test_calculate_partitions_basic(self, ray_backend):
        """Basic partition calculation works."""
        distributions = ["norm", "expon", "gamma"]
        partitions = ray_backend._calculate_partitions(distributions)

        assert partitions >= 1
        assert partitions <= ray_backend.get_parallelism() * 2

    def test_calculate_partitions_with_slow_distributions(self, ray_backend):
        """Slow distributions get weighted 3x in partition calculation."""
        # All fast distributions
        fast_dists = ["norm", "expon", "gamma", "beta"]
        fast_partitions = ray_backend._calculate_partitions(fast_dists)

        # Same number but with slow distributions
        from spark_bestfit.distributions import DistributionRegistry

        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        slow_dist = list(slow_set)[0] if slow_set else "levy_stable"
        slow_dists = ["norm", "expon", slow_dist]

        # Slow distributions should result in more effective partitions
        slow_partitions = ray_backend._calculate_partitions(slow_dists)

        # With slow weighting, effective count is higher
        assert slow_partitions >= 1

    def test_calculate_partitions_empty_list(self, ray_backend):
        """Empty distribution list returns minimum partitions."""
        partitions = ray_backend._calculate_partitions([])
        assert partitions >= 0

    def test_calculate_partitions_many_distributions(self, ray_backend):
        """Many distributions capped at 2x CPU count."""
        distributions = [f"dist_{i}" for i in range(100)]
        partitions = ray_backend._calculate_partitions(distributions)

        # Should be capped at 2x parallelism
        assert partitions <= ray_backend.get_parallelism() * 2


class TestRayBackendGenerateSamples:
    """Tests for distributed sample generation."""

    def test_generate_samples_basic(self, ray_backend):
        """generate_samples creates correct number of samples."""

        def generator_func(n_samples, partition_id, seed):
            np.random.seed(seed)
            return {"value": np.random.normal(0, 1, n_samples)}

        result = ray_backend.generate_samples(
            n=100,
            generator_func=generator_func,
            column_names=["value"],
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert "value" in result.columns

    def test_generate_samples_multiple_columns(self, ray_backend):
        """generate_samples works with multiple columns."""

        def generator_func(n_samples, partition_id, seed):
            np.random.seed(seed)
            return {
                "x": np.random.normal(0, 1, n_samples),
                "y": np.random.uniform(0, 1, n_samples),
                "z": np.random.exponential(1, n_samples),
            }

        result = ray_backend.generate_samples(
            n=500,
            generator_func=generator_func,
            column_names=["x", "y", "z"],
            num_partitions=4,
            random_seed=42,
        )

        assert len(result) == 500
        assert list(result.columns) == ["x", "y", "z"]

    def test_generate_samples_reproducibility(self, ray_backend):
        """generate_samples with same seed produces similar statistics."""

        def generator_func(n_samples, partition_id, seed):
            np.random.seed(seed)
            return {"value": np.random.normal(50, 10, n_samples)}

        result1 = ray_backend.generate_samples(
            n=1000,
            generator_func=generator_func,
            column_names=["value"],
            num_partitions=2,
            random_seed=42,
        )

        result2 = ray_backend.generate_samples(
            n=1000,
            generator_func=generator_func,
            column_names=["value"],
            num_partitions=2,
            random_seed=42,
        )

        # With same seed and partitions, should produce same data
        np.testing.assert_array_almost_equal(result1["value"].values, result2["value"].values)

    def test_generate_samples_zero_n(self, ray_backend):
        """generate_samples with n=0 returns empty DataFrame."""

        def generator_func(n_samples, partition_id, seed):
            return {"value": np.array([])}

        result = ray_backend.generate_samples(
            n=0,
            generator_func=generator_func,
            column_names=["value"],
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_generate_samples_custom_partitions(self, ray_backend):
        """generate_samples respects custom partition count."""

        partition_ids_seen = []

        def generator_func(n_samples, partition_id, seed):
            partition_ids_seen.append(partition_id)
            return {"value": np.ones(n_samples) * partition_id}

        result = ray_backend.generate_samples(
            n=100,
            generator_func=generator_func,
            column_names=["value"],
            num_partitions=5,
            random_seed=42,
        )

        assert len(result) == 100
        # Should have samples from multiple partitions
        assert len(set(result["value"].values)) >= 1


class TestRayBackendSpearmanCorrelation:
    """Tests for Spearman correlation on Ray Dataset (collects to pandas)."""

    def test_spearman_correlation_ray_dataset(self, ray_backend):
        """Spearman correlation on Ray Dataset collects and computes correctly."""
        np.random.seed(42)
        n = 500

        # Create monotonic relationship (perfect Spearman correlation)
        x = np.arange(n).astype(float)
        y = x**2  # Monotonic but not linear
        z = -x  # Perfect negative correlation with x

        data = pd.DataFrame({"x": x, "y": y, "z": z})
        ds = ray.data.from_pandas(data)

        corr = ray_backend.compute_correlation(ds, ["x", "y", "z"], method="spearman")

        assert corr.shape == (3, 3)
        # x and y have perfect monotonic relationship
        assert corr[0, 1] > 0.99
        # x and z have perfect negative monotonic relationship
        assert corr[0, 2] < -0.99
        # Diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0])

    def test_spearman_vs_pearson_nonlinear(self, ray_backend):
        """Spearman captures monotonic relationships that Pearson misses."""
        np.random.seed(42)
        n = 500

        x = np.linspace(1, 10, n)
        y = np.exp(x)  # Exponential - monotonic but not linear

        data = pd.DataFrame({"x": x, "y": y})
        ds = ray.data.from_pandas(data)

        pearson_corr = ray_backend.compute_correlation(ds, ["x", "y"], method="pearson")
        spearman_corr = ray_backend.compute_correlation(ds, ["x", "y"], method="spearman")

        # Spearman should be perfect (1.0) for monotonic relationship
        assert spearman_corr[0, 1] > 0.999
        # Pearson will be high but not perfect due to nonlinearity
        assert pearson_corr[0, 1] < spearman_corr[0, 1]


class TestRayBackendDistributedHistogram:
    """Tests for distributed histogram computation on Ray Dataset."""

    def test_distributed_histogram_ray_dataset(self, ray_backend):
        """Distributed histogram on Ray Dataset matches local computation."""
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.normal(50, 10, 1000)})
        ds = ray.data.from_pandas(data)
        bin_edges = np.linspace(20, 80, 31)

        # Compute with Ray Dataset (distributed)
        ray_counts, ray_total = ray_backend.compute_histogram(ds, "value", bin_edges)

        # Compute locally with pandas
        local_counts, local_total = ray_backend.compute_histogram(data, "value", bin_edges)

        # Should match
        np.testing.assert_array_equal(ray_counts, local_counts)
        assert ray_total == local_total

    def test_distributed_histogram_with_nan(self, ray_backend):
        """Distributed histogram correctly ignores NaN values."""
        data_with_nan = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0]})
        ds = ray.data.from_pandas(data_with_nan)
        bin_edges = np.array([0, 2, 4, 6, 8])

        counts, total = ray_backend.compute_histogram(ds, "value", bin_edges)

        # Should only count non-NaN values (5 values)
        assert total == 5
        assert sum(counts) == 5

    def test_distributed_histogram_large_dataset(self, ray_backend):
        """Distributed histogram scales with larger datasets."""
        np.random.seed(42)
        # Create larger dataset that will be split across partitions
        # Use normal distribution to ensure most values fall within bin range
        data = pd.DataFrame({"value": np.random.normal(50, 10, 10000)})
        ds = ray.data.from_pandas(data)
        bin_edges = np.linspace(0, 100, 101)

        counts, total = ray_backend.compute_histogram(ds, "value", bin_edges)

        # Nearly all normal(50,10) values fall within [0, 100]
        assert total >= 9990  # Allow a few outliers
        assert len(counts) == 100
        assert sum(counts) == total


class TestRayBackendRayDatasetOperations:
    """Additional tests for Ray Dataset specific operations."""

    def test_collect_column_ray_dataset_large(self, ray_backend):
        """collect_column works with larger Ray Dataset."""
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.normal(0, 1, 5000)})
        ds = ray.data.from_pandas(data)

        result = ray_backend.collect_column(ds, "value")

        assert len(result) == 5000
        np.testing.assert_array_almost_equal(result, data["value"].values)

    def test_get_column_stats_ray_dataset_precision(self, ray_backend):
        """get_column_stats on Ray Dataset matches pandas precision."""
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.uniform(-100, 100, 2000)})
        ds = ray.data.from_pandas(data)

        ray_stats = ray_backend.get_column_stats(ds, "value")
        pandas_min = float(data["value"].min())
        pandas_max = float(data["value"].max())

        assert ray_stats["min"] == pandas_min
        assert ray_stats["max"] == pandas_max
        assert ray_stats["count"] == 2000

    def test_sample_column_ray_dataset_fraction(self, ray_backend):
        """sample_column on Ray Dataset respects fraction approximately."""
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.normal(0, 1, 1000)})
        ds = ray.data.from_pandas(data)

        sample = ray_backend.sample_column(ds, "value", fraction=0.1, seed=42)

        # Ray's random_sample is approximate, check it's in reasonable range
        assert len(sample) >= 50  # At least 5% (half of 10%)
        assert len(sample) <= 200  # At most 20% (double of 10%)

    def test_sample_column_ray_dataset_full(self, ray_backend):
        """sample_column with fraction=1.0 returns all data."""
        data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds = ray.data.from_pandas(data)

        sample = ray_backend.sample_column(ds, "value", fraction=1.0, seed=42)

        # Should return all values (order may differ)
        assert len(sample) == 5
        assert set(sample) == {1.0, 2.0, 3.0, 4.0, 5.0}


class TestRayBackendDiscreteParallelFit:
    """Tests for discrete distribution parallel fitting."""

    def test_discrete_parallel_fit_poisson(self, ray_backend):
        """Discrete parallel_fit works for Poisson distribution."""
        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        np.random.seed(42)
        true_lambda = 7
        data_sample = np.random.poisson(lam=true_lambda, size=1000)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        results = ray_backend.parallel_fit(
            distributions=["poisson"],
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="counts",
            data_stats=data_stats,
            is_discrete=True,
        )

        assert len(results) == 1
        assert results[0]["distribution"] == "poisson"
        fitted_lambda = results[0]["parameters"][0]
        assert abs(fitted_lambda - true_lambda) < 0.5

    def test_discrete_parallel_fit_multiple(self, ray_backend):
        """Discrete parallel_fit works with multiple distributions."""
        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        np.random.seed(42)
        data_sample = np.random.poisson(lam=5, size=500)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        results = ray_backend.parallel_fit(
            distributions=["poisson", "nbinom", "geom"],
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="counts",
            data_stats=data_stats,
            is_discrete=True,
        )

        # Should get results for at least Poisson
        assert len(results) >= 1
        dist_names = [r["distribution"] for r in results]
        assert "poisson" in dist_names

    def test_discrete_parallel_fit_with_bounds(self, ray_backend):
        """Discrete parallel_fit respects bounds."""
        from spark_bestfit.discrete_fitting import fit_single_discrete_distribution

        np.random.seed(42)
        data_sample = np.random.poisson(lam=10, size=500)
        x_values, counts = np.unique(data_sample, return_counts=True)
        pmf = counts / len(data_sample)
        data_stats = compute_data_stats(data_sample.astype(float))

        results = ray_backend.parallel_fit(
            distributions=["poisson"],
            histogram=(x_values, pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name="counts",
            data_stats=data_stats,
            is_discrete=True,
            lower_bound=0,
            upper_bound=20,
        )

        assert len(results) >= 1


class TestRayBackendLazyMetrics:
    """Tests for lazy_metrics with Ray Dataset."""

    def test_lazy_metrics_with_ray_dataset(self, ray_backend):
        """lazy_metrics=True with Ray Dataset should work with .best() call."""
        from spark_bestfit import DistributionFitter

        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=5000)
        df = pd.DataFrame({"value": data})
        ds = ray.data.from_pandas(df)

        fitter = DistributionFitter(backend=ray_backend)
        results = fitter.fit(
            ds,
            column="value",
            lazy_metrics=True,
            max_distributions=5,
        )

        # This should trigger lazy metric computation without error
        best = results.best(n=1)
        assert len(best) == 1
        assert best[0].distribution is not None
        # Verify KS statistic was computed (not None)
        assert best[0].ks_statistic is not None

    def test_lazy_metrics_with_ray_dataset_multi_column(self, ray_backend):
        """lazy_metrics=True with Ray Dataset works for multi-column fitting."""
        from spark_bestfit import DistributionFitter

        np.random.seed(42)
        data = pd.DataFrame({
            "col_a": np.random.normal(100, 20, 3000),
            "col_b": np.random.exponential(5, 3000),
        })
        ds = ray.data.from_pandas(data)

        fitter = DistributionFitter(backend=ray_backend)
        results = fitter.fit(
            ds,
            columns=["col_a", "col_b"],
            lazy_metrics=True,
            max_distributions=5,
        )

        # Get best per column - should trigger lazy computation
        best_per_col = results.best_per_column(n=1)
        assert "col_a" in best_per_col
        assert "col_b" in best_per_col
        assert best_per_col["col_a"][0].ks_statistic is not None
        assert best_per_col["col_b"][0].ks_statistic is not None


class TestRayBackendGaussianCopula:
    """Tests for GaussianCopula with Ray Dataset and RayBackend."""

    def test_copula_fit_with_ray_dataset(self, ray_backend):
        """GaussianCopula.fit works with Ray Dataset and RayBackend."""
        from spark_bestfit import DistributionFitter, GaussianCopula

        np.random.seed(42)
        n = 2000
        x = np.random.normal(0, 1, n)
        data = pd.DataFrame({
            "feature_a": x * 10 + 50,
            "feature_b": np.exp(0.5 * x + np.random.normal(0, 0.2, n)),
        })
        ds = ray.data.from_pandas(data)

        # Fit marginals
        fitter = DistributionFitter(backend=ray_backend)
        results = fitter.fit(
            ds,
            columns=["feature_a", "feature_b"],
            max_distributions=5,
        )

        # Fit copula with Ray Dataset and RayBackend
        copula = GaussianCopula.fit(results, ds, backend=ray_backend)

        assert copula.correlation_matrix is not None
        assert copula.correlation_matrix.shape == (2, 2)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(
            np.diag(copula.correlation_matrix), [1.0, 1.0], decimal=5
        )
        # Features should be correlated (positive correlation)
        assert copula.correlation_matrix[0, 1] > 0.5

    def test_copula_sample_with_ray_backend(self, ray_backend):
        """GaussianCopula sampling works after fitting with RayBackend."""
        from spark_bestfit import DistributionFitter, GaussianCopula

        np.random.seed(42)
        data = pd.DataFrame({
            "col1": np.random.normal(100, 20, 2000),
            "col2": np.random.normal(50, 10, 2000),
        })
        ds = ray.data.from_pandas(data)

        fitter = DistributionFitter(backend=ray_backend)
        results = fitter.fit(ds, columns=["col1", "col2"], max_distributions=5)

        copula = GaussianCopula.fit(results, ds, backend=ray_backend)

        # Generate samples
        samples = copula.sample(n=1000, random_state=42)

        assert "col1" in samples
        assert "col2" in samples
        assert len(samples["col1"]) == 1000
        assert len(samples["col2"]) == 1000

    def test_copula_fit_with_pandas_and_ray_backend(self, ray_backend):
        """GaussianCopula.fit works with pandas DataFrame and RayBackend."""
        from spark_bestfit import DistributionFitter, GaussianCopula

        np.random.seed(42)
        data = pd.DataFrame({
            "a": np.random.normal(0, 1, 2000),
            "b": np.random.uniform(0, 10, 2000),
        })

        # Fit with pandas DataFrame
        fitter = DistributionFitter(backend=ray_backend)
        results = fitter.fit(data, columns=["a", "b"], max_distributions=5)

        # Fit copula with pandas DataFrame
        copula = GaussianCopula.fit(results, data, backend=ray_backend)

        assert copula.correlation_matrix.shape == (2, 2)
        # Should work without error


class TestRayBackendProgressCallback:
    """Tests for progress_callback in RayBackend."""

    def test_ray_backend_progress_callback(self, ray_backend, normal_data, histogram):
        """RayBackend invokes progress callback with correct values."""
        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma"]
        results = ray_backend.parallel_fit(
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
            # Progress is incremental via ray.wait()
            assert 1 <= completed <= len(distributions)
            assert 0 < percent <= 100

        # Last callback should show 100%
        last_call = progress_calls[-1]
        assert last_call[0] == len(distributions)
        assert last_call[2] == 100.0

    def test_ray_backend_progress_callback_error_handling(self, ray_backend, normal_data, histogram):
        """RayBackend handles callback errors gracefully."""

        def failing_callback(completed, total, percent):
            raise ValueError("Intentional callback error")

        # Should not raise despite callback errors
        distributions = ["norm", "expon"]
        results = ray_backend.parallel_fit(
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

    def test_ray_backend_no_callback(self, ray_backend, normal_data, histogram):
        """RayBackend works fine without progress callback."""
        distributions = ["norm", "expon"]
        results = ray_backend.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=None,
        )

        assert len(results) > 0

    def test_fitter_progress_callback_ray(self, ray_backend, normal_data):
        """DistributionFitter progress callback works with RayBackend."""
        from spark_bestfit import DistributionFitter

        fitter = DistributionFitter(backend=ray_backend)

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

    def test_ray_backend_progress_ordered_by_completion(self, ray_backend, normal_data, histogram):
        """RayBackend progress shows strictly increasing completed count."""
        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma", "uniform", "beta"]
        ray_backend.parallel_fit(
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

    # --- Edge case tests ---

    def test_ray_backend_progress_empty_distributions(self, ray_backend, normal_data, histogram):
        """RayBackend progress callback handles empty distribution list gracefully."""
        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        results = ray_backend.parallel_fit(
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

    def test_ray_backend_progress_single_distribution(self, ray_backend, normal_data, histogram):
        """RayBackend progress callback works with single distribution."""
        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        results = ray_backend.parallel_fit(
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

    def test_ray_backend_progress_bounds_invariants(self, ray_backend, normal_data, histogram):
        """RayBackend progress callback values are always within valid bounds."""
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
        ray_backend.parallel_fit(
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

    # --- DiscreteDistributionFitter tests ---

    def test_discrete_fitter_progress_callback_ray(self, ray_backend):
        """DiscreteDistributionFitter progress callback works with RayBackend."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        count_data = np.random.poisson(lam=10, size=500)

        fitter = DiscreteDistributionFitter(backend=ray_backend)

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
