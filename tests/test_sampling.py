"""Tests for distributed sampling module.

Uses LocalBackend for most tests. Spark-specific tests are in a separate class.
"""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.results import DistributionFitResult
from spark_bestfit.sampling import sample_distributed


class TestSampleDistributed:
    """Tests for sample_distributed function with LocalBackend."""

    def test_sample_distributed_basic(self, local_backend):
        """Test basic distributed sampling with LocalBackend."""
        df = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=1000,
            backend=local_backend,
            random_seed=42,
        )

        # Check structure (pandas DataFrame for LocalBackend)
        assert "sample" in df.columns
        assert len(df) == 1000

    def test_sample_distributed_custom_column_name(self, local_backend):
        """Test sampling with custom column name."""
        df = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=local_backend,
            column_name="my_samples",
        )

        assert "my_samples" in df.columns

    def test_sample_distributed_reproducibility(self, local_backend):
        """Test that sampling is reproducible with seed."""
        df1 = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=local_backend,
            num_partitions=2,
            random_seed=42,
        )

        df2 = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=local_backend,
            num_partitions=2,
            random_seed=42,
        )

        samples1 = sorted(df1["sample"].tolist())
        samples2 = sorted(df2["sample"].tolist())

        assert np.allclose(samples1, samples2)

    def test_sample_distributed_different_seeds_different_samples(self, local_backend):
        """Test that different seeds produce different samples."""
        df1 = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=local_backend,
            random_seed=42,
        )

        df2 = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=local_backend,
            random_seed=123,
        )

        samples1 = df1["sample"].tolist()
        samples2 = df2["sample"].tolist()

        assert not np.allclose(sorted(samples1), sorted(samples2))

    def test_sample_distributed_statistical_properties(self, local_backend):
        """Test that samples have expected statistical properties."""
        n = 10000
        loc, scale = 100.0, 15.0

        df = sample_distributed(
            distribution="norm",
            parameters=[loc, scale],
            n=n,
            backend=local_backend,
            random_seed=42,
        )

        samples = df["sample"].values

        # Check approximate statistical properties
        assert abs(samples.mean() - loc) < 1.0  # Within 1 unit of expected mean
        assert abs(samples.std() - scale) < 1.0  # Within 1 unit of expected std

    def test_sample_distributed_various_distributions(self, local_backend):
        """Test sampling from various distributions."""
        distributions = [
            ("norm", [0.0, 1.0]),
            ("expon", [0.0, 1.0]),
            ("gamma", [2.0, 0.0, 1.0]),
            ("beta", [2.0, 5.0, 0.0, 1.0]),
        ]

        for dist_name, params in distributions:
            df = sample_distributed(
                distribution=dist_name,
                parameters=params,
                n=100,
                backend=local_backend,
                random_seed=42,
            )

            assert len(df) == 100, f"Failed for {dist_name}"

    def test_sample_distributed_explicit_partitions(self, local_backend):
        """Test sampling with explicit partition count."""
        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=1000,
            backend=local_backend,
            num_partitions=4,
            random_seed=42,
        )

        assert len(df) == 1000

    def test_sample_distributed_large_n(self, local_backend):
        """Test sampling with large n."""
        n = 100_000

        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=n,
            backend=local_backend,
            random_seed=42,
        )

        assert len(df) == n


class TestDistributionFitResultSample:
    """Tests for sample method on DistributionFitResult (local sampling)."""

    def test_sample_method(self):
        """Test local sample method on DistributionFitResult."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        samples = result.sample(size=500, random_state=42)

        assert len(samples) == 500
        assert isinstance(samples, np.ndarray)

    def test_sample_reproducibility(self):
        """Test reproducibility of sample method."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.01,
        )

        samples1 = result.sample(size=100, random_state=42)
        samples2 = result.sample(size=100, random_state=42)

        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_sample_uses_fitted_parameters(self):
        """Test that sample uses the fitted distribution parameters."""
        loc, scale = 100.0, 5.0
        result = DistributionFitResult(
            distribution="norm",
            parameters=[loc, scale],
            sse=0.01,
        )

        samples = result.sample(size=5000, random_state=42)

        # Samples should be approximately normal(loc, scale)
        assert abs(samples.mean() - loc) < 1.0
        assert abs(samples.std() - scale) < 0.5


class TestSamplingEdgeCases:
    """Edge case tests for robust coverage."""

    def test_sample_distributed_n_equals_one(self, local_backend):
        """Test sampling exactly 1 sample."""
        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=1,
            backend=local_backend,
            random_seed=42,
        )
        assert len(df) == 1

    def test_sample_distributed_uneven_partition_distribution(self, local_backend):
        """Test when n doesn't divide evenly by partitions (remainder handling)."""
        # 103 samples across 4 partitions: 26, 26, 26, 25
        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=103,
            backend=local_backend,
            num_partitions=4,
            random_seed=42,
        )
        assert len(df) == 103

    def test_sample_distributed_more_partitions_than_samples(self, local_backend):
        """Test when partitions > n (some partitions get 0 samples)."""
        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=3,
            backend=local_backend,
            num_partitions=10,
            random_seed=42,
        )
        assert len(df) == 3

    def test_sample_distributed_discrete_distribution(self, local_backend):
        """Test sampling from discrete distribution (Poisson)."""
        df = sample_distributed(
            distribution="poisson",
            parameters=[5.0],  # mu=5
            n=1000,
            backend=local_backend,
            random_seed=42,
        )
        assert len(df) == 1000

        # Verify samples are integer-like (Poisson produces integers)
        samples = df["sample"].values
        assert np.allclose(samples, np.round(samples))
        # Mean should be approximately mu=5
        assert abs(samples.mean() - 5.0) < 0.5

    def test_sample_distributed_binomial_distribution(self, local_backend):
        """Test sampling from binomial distribution."""
        df = sample_distributed(
            distribution="binom",
            parameters=[10, 0.3],  # n=10, p=0.3
            n=1000,
            backend=local_backend,
            random_seed=42,
        )
        assert len(df) == 1000

        samples = df["sample"].values
        # Mean should be approximately n*p = 3
        assert abs(samples.mean() - 3.0) < 0.5

    def test_sample_distributed_without_seed(self, local_backend):
        """Test sampling without random seed still produces valid samples."""
        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=100,
            backend=local_backend,
            random_seed=None,
        )

        # Should have correct count and valid samples
        assert len(df) == 100
        samples = df["sample"].values
        assert np.all(np.isfinite(samples))
        # Mean should be approximately 0 for standard normal
        assert abs(samples.mean()) < 0.5

    def test_sample_distributed_ks_test_validation(self, local_backend):
        """Validate samples against theoretical distribution using K-S test."""
        loc, scale = 50.0, 10.0
        n = 5000

        df = sample_distributed(
            distribution="norm",
            parameters=[loc, scale],
            n=n,
            backend=local_backend,
            random_seed=42,
        )
        samples = df["sample"].values

        # K-S test: samples should come from norm(loc, scale)
        ks_stat, p_value = st.kstest(samples, "norm", args=(loc, scale))

        # With 5000 samples, p-value should be > 0.01 if from correct distribution
        assert p_value > 0.01, f"K-S test failed: stat={ks_stat}, p={p_value}"

    def test_sample_distributed_extreme_parameters(self, local_backend):
        """Test sampling with extreme but valid parameters."""
        # Very small scale
        df = sample_distributed(
            distribution="norm",
            parameters=[0.0, 0.001],
            n=100,
            backend=local_backend,
            random_seed=42,
        )
        samples = df["sample"].values
        assert samples.std() < 0.01

        # Very large location
        df = sample_distributed(
            distribution="norm",
            parameters=[1e6, 1.0],
            n=100,
            backend=local_backend,
            random_seed=42,
        )
        samples = df["sample"].values
        assert abs(samples.mean() - 1e6) < 1.0

    def test_sample_distributed_weibull_shape_parameter(self, local_backend):
        """Test distribution with shape parameter (Weibull)."""
        # Weibull: c=1.5 (shape), loc=0, scale=2
        df = sample_distributed(
            distribution="weibull_min",
            parameters=[1.5, 0.0, 2.0],
            n=1000,
            backend=local_backend,
            random_seed=42,
        )
        assert len(df) == 1000

        samples = df["sample"].values
        # All samples should be positive for Weibull with loc=0
        assert np.all(samples >= 0)


class TestSamplingIntegration:
    """Integration tests for end-to-end workflows."""

    def test_fit_then_sample(self, local_backend):
        """Test complete workflow: fit distribution then sample via backend."""
        import pandas as pd
        from spark_bestfit import DistributionFitter

        # Generate original data
        np.random.seed(42)
        original_data = np.random.exponential(scale=5.0, size=5000)
        df = pd.DataFrame({"value": original_data})

        # Fit
        fitter = DistributionFitter(backend=local_backend, random_seed=42)
        results = fitter.fit(df, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Sample using backend
        samples_df = sample_distributed(
            distribution=best.distribution,
            parameters=best.parameters,
            n=5000,
            backend=local_backend,
            random_seed=42,
        )
        samples = samples_df["sample"].values

        # Original and generated should have similar statistics
        assert abs(original_data.mean() - samples.mean()) < 1.0
        assert abs(original_data.std() - samples.std()) < 1.0


# ============================================================================
# Spark-specific tests (skip if PySpark not installed)
# ============================================================================

try:
    from pyspark.sql import SparkSession
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestSampleDistributed:
    """Tests for sample_distributed function with SparkBackend.

    Uses the shared spark_session fixture from conftest.py.
    """

    def test_sample_distributed_basic(self, spark_session):
        """Test basic distributed sampling with Spark."""
        from spark_bestfit.backends import BackendFactory
        from spark_bestfit.sampling import sample_distributed

        backend = BackendFactory.create("spark", spark_session=spark_session)
        df = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=1000,
            backend=backend,
            random_seed=42,
        )

        # Check structure
        assert df.columns == ["sample"]
        assert df.count() == 1000

    def test_sample_distributed_reproducibility(self, spark_session):
        """Test that distributed sampling is reproducible with seed."""
        from spark_bestfit.backends import BackendFactory
        from spark_bestfit.sampling import sample_distributed

        backend = BackendFactory.create("spark", spark_session=spark_session)

        df1 = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=backend,
            num_partitions=2,
            random_seed=42,
        )

        df2 = sample_distributed(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            backend=backend,
            num_partitions=2,
            random_seed=42,
        )

        samples1 = sorted(df1.toPandas()["sample"].tolist())
        samples2 = sorted(df2.toPandas()["sample"].tolist())

        assert np.allclose(samples1, samples2)

    def test_sample_distributed_statistical_properties(self, spark_session):
        """Test that distributed samples have expected statistical properties."""
        from spark_bestfit.backends import BackendFactory
        from spark_bestfit.sampling import sample_distributed

        backend = BackendFactory.create("spark", spark_session=spark_session)
        n = 10000
        loc, scale = 100.0, 15.0

        df = sample_distributed(
            distribution="norm",
            parameters=[loc, scale],
            n=n,
            backend=backend,
            random_seed=42,
        )

        samples = df.toPandas()["sample"].values

        # Check approximate statistical properties
        assert abs(samples.mean() - loc) < 1.0
        assert abs(samples.std() - scale) < 1.0

    def test_sample_distributed_via_result(self, spark_session):
        """Test sample_distributed with DistributionFitResult parameters."""
        from spark_bestfit.backends import BackendFactory
        from spark_bestfit.sampling import sample_distributed

        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        backend = BackendFactory.create("spark", spark_session=spark_session)
        df = sample_distributed(
            distribution=result.distribution,
            parameters=result.parameters,
            n=500,
            backend=backend,
            random_seed=42,
        )

        assert df.count() == 500
        assert df.columns == ["sample"]

    def test_fit_then_sample_distributed(self, spark_session):
        """Test complete workflow: fit distribution then sample via backend."""
        from spark_bestfit import DistributionFitter
        from spark_bestfit.backends import BackendFactory
        from spark_bestfit.sampling import sample_distributed

        # Generate original data
        np.random.seed(42)
        original_data = np.random.exponential(scale=5.0, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in original_data], ["value"])

        # Fit
        fitter = DistributionFitter(spark_session, random_seed=42)
        results = fitter.fit(df, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Sample using sample_distributed
        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = sample_distributed(
            distribution=best.distribution,
            parameters=best.parameters,
            n=5000,
            backend=backend,
            random_seed=42,
        )
        samples = samples_df.toPandas()["sample"].values

        # Original and generated should have similar statistics
        assert abs(original_data.mean() - samples.mean()) < 1.0
        assert abs(original_data.std() - samples.std()) < 1.0
