"""Benchmarks for discrete distribution fitting.

Compares discrete fitting (MLE-based) vs continuous fitting (scipy.fit).

Run with: make benchmark
"""

import pytest

from spark_bestfit import DiscreteDistributionFitter, DistributionFitter


class TestDiscreteVsContinuous:
    """Compare discrete and continuous fitting performance."""

    def test_continuous_fit_10k(self, benchmark, spark_session, df_10k):
        """Benchmark continuous fitting on 10K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_continuous():
            # Use max_distributions=10 for fair comparison with discrete
            results = fitter.fit(df_10k, "value", max_distributions=10, num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_continuous)
        assert result.count() > 0  # Verify fit completed

    def test_discrete_fit_10k(self, benchmark, spark_session, discrete_df_10k):
        """Benchmark discrete fitting on 10K rows."""
        fitter = DiscreteDistributionFitter(spark_session)

        def fit_discrete():
            # Use max_distributions=10 for fair comparison with continuous
            results = fitter.fit(discrete_df_10k, "counts", max_distributions=10, num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_discrete)
        assert result.count() > 0


class TestDiscreteFitterScaling:
    """Benchmark discrete fitter scaling."""

    def test_discrete_all_distributions(self, benchmark, spark_session, discrete_df_10k):
        """Benchmark fitting all discrete distributions."""
        fitter = DiscreteDistributionFitter(spark_session)

        def fit_all_discrete():
            results = fitter.fit(discrete_df_10k, "counts", num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_all_discrete)
        assert result.count() > 0  # Verify fit completed
