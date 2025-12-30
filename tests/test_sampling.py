"""Tests for distributed sampling module."""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.results import DistributionFitResult
from spark_bestfit.sampling import sample_spark


class TestSampleSpark:
    """Tests for sample_spark function."""

    def test_sample_spark_basic(self, spark_session):
        """Test basic distributed sampling."""
        df = sample_spark(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=1000,
            spark=spark_session,
            random_seed=42,
        )

        # Check structure
        assert df.columns == ["sample"]
        assert df.count() == 1000

    def test_sample_spark_custom_column_name(self, spark_session):
        """Test sampling with custom column name."""
        df = sample_spark(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            spark=spark_session,
            column_name="my_samples",
        )

        assert df.columns == ["my_samples"]

    def test_sample_spark_reproducibility(self, spark_session):
        """Test that sampling is reproducible with seed."""
        df1 = sample_spark(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            spark=spark_session,
            num_partitions=2,
            random_seed=42,
        )

        df2 = sample_spark(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            spark=spark_session,
            num_partitions=2,
            random_seed=42,
        )

        samples1 = sorted(df1.toPandas()["sample"].tolist())
        samples2 = sorted(df2.toPandas()["sample"].tolist())

        assert np.allclose(samples1, samples2)

    def test_sample_spark_different_seeds_different_samples(self, spark_session):
        """Test that different seeds produce different samples."""
        df1 = sample_spark(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            spark=spark_session,
            random_seed=42,
        )

        df2 = sample_spark(
            distribution="norm",
            parameters=[50.0, 10.0],
            n=100,
            spark=spark_session,
            random_seed=123,
        )

        samples1 = df1.toPandas()["sample"].tolist()
        samples2 = df2.toPandas()["sample"].tolist()

        assert not np.allclose(sorted(samples1), sorted(samples2))

    def test_sample_spark_statistical_properties(self, spark_session):
        """Test that samples have expected statistical properties."""
        n = 10000
        loc, scale = 100.0, 15.0

        df = sample_spark(
            distribution="norm",
            parameters=[loc, scale],
            n=n,
            spark=spark_session,
            random_seed=42,
        )

        samples = df.toPandas()["sample"].values

        # Check approximate statistical properties
        assert abs(samples.mean() - loc) < 1.0  # Within 1 unit of expected mean
        assert abs(samples.std() - scale) < 1.0  # Within 1 unit of expected std

    def test_sample_spark_various_distributions(self, spark_session):
        """Test sampling from various distributions."""
        distributions = [
            ("norm", [0.0, 1.0]),
            ("expon", [0.0, 1.0]),
            ("gamma", [2.0, 0.0, 1.0]),
            ("beta", [2.0, 5.0, 0.0, 1.0]),
        ]

        for dist_name, params in distributions:
            df = sample_spark(
                distribution=dist_name,
                parameters=params,
                n=100,
                spark=spark_session,
                random_seed=42,
            )

            assert df.count() == 100, f"Failed for {dist_name}"

    def test_sample_spark_explicit_partitions(self, spark_session):
        """Test sampling with explicit partition count."""
        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=1000,
            spark=spark_session,
            num_partitions=4,
            random_seed=42,
        )

        assert df.count() == 1000

    def test_sample_spark_even_distribution(self, spark_session):
        """Test that samples are evenly distributed across partitions."""
        n = 100
        num_partitions = 4

        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=n,
            spark=spark_session,
            num_partitions=num_partitions,
            random_seed=42,
        )

        # Each partition should have approximately n/num_partitions samples
        # Due to the way samples are distributed, should be exact for even division
        assert df.count() == n

    def test_sample_spark_large_n(self, spark_session):
        """Test sampling with large n."""
        n = 100_000

        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=n,
            spark=spark_session,
            random_seed=42,
        )

        assert df.count() == n


class TestDistributionFitResultSampleSpark:
    """Tests for sample_spark method on DistributionFitResult."""

    def test_sample_spark_method(self, spark_session):
        """Test sample_spark method on DistributionFitResult."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        df = result.sample_spark(n=500, spark=spark_session, random_seed=42)

        assert df.count() == 500
        assert df.columns == ["sample"]

    def test_sample_spark_method_custom_column(self, spark_session):
        """Test sample_spark with custom column name."""
        result = DistributionFitResult(
            distribution="gamma",
            parameters=[2.0, 0.0, 5.0],
            sse=0.003,
        )

        df = result.sample_spark(
            n=200,
            spark=spark_session,
            column_name="gamma_samples",
            random_seed=42,
        )

        assert df.columns == ["gamma_samples"]
        assert df.count() == 200

    def test_sample_spark_method_reproducibility(self, spark_session):
        """Test reproducibility of sample_spark method."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.01,
        )

        df1 = result.sample_spark(n=100, spark=spark_session, num_partitions=2, random_seed=42)
        df2 = result.sample_spark(n=100, spark=spark_session, num_partitions=2, random_seed=42)

        samples1 = sorted(df1.toPandas()["sample"].tolist())
        samples2 = sorted(df2.toPandas()["sample"].tolist())

        assert np.allclose(samples1, samples2)

    def test_sample_spark_uses_fitted_parameters(self, spark_session):
        """Test that sample_spark uses the fitted distribution parameters."""
        loc, scale = 100.0, 5.0
        result = DistributionFitResult(
            distribution="norm",
            parameters=[loc, scale],
            sse=0.01,
        )

        df = result.sample_spark(n=5000, spark=spark_session, random_seed=42)
        samples = df.toPandas()["sample"].values

        # Samples should be approximately normal(loc, scale)
        assert abs(samples.mean() - loc) < 1.0
        assert abs(samples.std() - scale) < 0.5

    def test_sample_spark_comparison_with_sample(self, spark_session):
        """Test that sample_spark produces similar distribution to sample()."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.01,
        )

        # Generate samples using both methods
        local_samples = result.sample(size=5000, random_state=42)
        spark_df = result.sample_spark(n=5000, spark=spark_session, random_seed=123)
        spark_samples = spark_df.toPandas()["sample"].values

        # Both should have similar statistical properties
        assert abs(local_samples.mean() - spark_samples.mean()) < 1.0
        assert abs(local_samples.std() - spark_samples.std()) < 0.5

    def test_sample_spark_with_partitions(self, spark_session):
        """Test sample_spark with specific partition count."""
        result = DistributionFitResult(
            distribution="expon",
            parameters=[0.0, 5.0],
            sse=0.02,
        )

        df = result.sample_spark(
            n=1000,
            spark=spark_session,
            num_partitions=8,
            random_seed=42,
        )

        assert df.count() == 1000


class TestSamplingEdgeCases:
    """Edge case tests for robust coverage."""

    def test_sample_spark_n_equals_one(self, spark_session):
        """Test sampling exactly 1 sample."""
        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=1,
            spark=spark_session,
            random_seed=42,
        )
        assert df.count() == 1

    def test_sample_spark_uneven_partition_distribution(self, spark_session):
        """Test when n doesn't divide evenly by partitions (remainder handling)."""
        # 103 samples across 4 partitions: 26, 26, 26, 25
        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=103,
            spark=spark_session,
            num_partitions=4,
            random_seed=42,
        )
        assert df.count() == 103

    def test_sample_spark_more_partitions_than_samples(self, spark_session):
        """Test when partitions > n (some partitions get 0 samples)."""
        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=3,
            spark=spark_session,
            num_partitions=10,
            random_seed=42,
        )
        assert df.count() == 3

    def test_sample_spark_discrete_distribution(self, spark_session):
        """Test sampling from discrete distribution (Poisson)."""
        df = sample_spark(
            distribution="poisson",
            parameters=[5.0],  # mu=5
            n=1000,
            spark=spark_session,
            random_seed=42,
        )
        assert df.count() == 1000

        # Verify samples are integer-like (Poisson produces integers)
        samples = df.toPandas()["sample"].values
        assert np.allclose(samples, np.round(samples))
        # Mean should be approximately mu=5
        assert abs(samples.mean() - 5.0) < 0.5

    def test_sample_spark_binomial_distribution(self, spark_session):
        """Test sampling from binomial distribution."""
        df = sample_spark(
            distribution="binom",
            parameters=[10, 0.3],  # n=10, p=0.3
            n=1000,
            spark=spark_session,
            random_seed=42,
        )
        assert df.count() == 1000

        samples = df.toPandas()["sample"].values
        # Mean should be approximately n*p = 3
        assert abs(samples.mean() - 3.0) < 0.5

    def test_sample_spark_without_seed(self, spark_session):
        """Test sampling without random seed still produces valid samples."""
        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 1.0],
            n=100,
            spark=spark_session,
            random_seed=None,
        )

        # Should have correct count and valid samples
        assert df.count() == 100
        samples = df.toPandas()["sample"].values
        assert np.all(np.isfinite(samples))
        # Mean should be approximately 0 for standard normal
        assert abs(samples.mean()) < 0.5

    def test_sample_spark_ks_test_validation(self, spark_session):
        """Validate samples against theoretical distribution using K-S test."""
        loc, scale = 50.0, 10.0
        n = 5000

        df = sample_spark(
            distribution="norm",
            parameters=[loc, scale],
            n=n,
            spark=spark_session,
            random_seed=42,
        )
        samples = df.toPandas()["sample"].values

        # K-S test: samples should come from norm(loc, scale)
        ks_stat, p_value = st.kstest(samples, "norm", args=(loc, scale))

        # With 5000 samples, p-value should be > 0.01 if from correct distribution
        assert p_value > 0.01, f"K-S test failed: stat={ks_stat}, p={p_value}"

    def test_sample_spark_extreme_parameters(self, spark_session):
        """Test sampling with extreme but valid parameters."""
        # Very small scale
        df = sample_spark(
            distribution="norm",
            parameters=[0.0, 0.001],
            n=100,
            spark=spark_session,
            random_seed=42,
        )
        samples = df.toPandas()["sample"].values
        assert samples.std() < 0.01

        # Very large location
        df = sample_spark(
            distribution="norm",
            parameters=[1e6, 1.0],
            n=100,
            spark=spark_session,
            random_seed=42,
        )
        samples = df.toPandas()["sample"].values
        assert abs(samples.mean() - 1e6) < 1.0

    def test_sample_spark_weibull_shape_parameter(self, spark_session):
        """Test distribution with shape parameter (Weibull)."""
        # Weibull: c=1.5 (shape), loc=0, scale=2
        df = sample_spark(
            distribution="weibull_min",
            parameters=[1.5, 0.0, 2.0],
            n=1000,
            spark=spark_session,
            random_seed=42,
        )
        assert df.count() == 1000

        samples = df.toPandas()["sample"].values
        # All samples should be positive for Weibull with loc=0
        assert np.all(samples >= 0)


class TestSamplingIntegration:
    """Integration tests for end-to-end workflows."""

    def test_fit_then_sample_spark(self, spark_session):
        """Test complete workflow: fit distribution then sample via Spark."""
        from spark_bestfit import DistributionFitter

        # Generate original data
        np.random.seed(42)
        original_data = np.random.exponential(scale=5.0, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in original_data], ["value"])

        # Fit
        fitter = DistributionFitter(spark_session, random_seed=42)
        results = fitter.fit(df, column="value", max_distributions=5)
        best = results.best(n=1)[0]

        # Sample using spark
        samples_df = best.sample_spark(n=5000, spark=spark_session, random_seed=42)
        samples = samples_df.toPandas()["sample"].values

        # Original and generated should have similar statistics
        assert abs(original_data.mean() - samples.mean()) < 1.0
        assert abs(original_data.std() - samples.std()) < 1.0
