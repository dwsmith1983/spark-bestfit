"""Benchmarks for discrete distribution fitting.

Compares discrete fitting (MLE-based) vs continuous fitting (scipy.fit).

Run with: make benchmark

Note: These benchmarks use distribution-aware partitioning (v1.6.1+) which
automatically calculates optimal partition count based on distribution mix.
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
            results = fitter.fit(df_10k, "value", max_distributions=10)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_continuous)
        assert result.count() > 0  # Verify fit completed

    def test_discrete_fit_10k(self, benchmark, spark_session, discrete_df_10k):
        """Benchmark discrete fitting on 10K rows."""
        fitter = DiscreteDistributionFitter(spark_session)

        def fit_discrete():
            # Use max_distributions=10 for fair comparison with continuous
            results = fitter.fit(discrete_df_10k, "counts", max_distributions=10)
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
            # Let fitter use distribution-aware partitioning
            results = fitter.fit(discrete_df_10k, "counts")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_all_discrete)
        assert result.count() > 0  # Verify fit completed


class TestDiscreteMultiColumnEfficiency:
    """Benchmark discrete multi-column fitting efficiency."""

    def test_discrete_3_columns_separately(
        self, benchmark, spark_session, discrete_df_multi_3col_10k
    ):
        """Benchmark fitting 3 discrete columns in 3 separate calls."""
        fitter = DiscreteDistributionFitter(spark_session)
        columns = ["col_poisson", "col_nbinom", "col_geom"]

        def fit_separately():
            all_results = []
            for col in columns:
                results = fitter.fit(discrete_df_multi_3col_10k, column=col)
                _ = results.best(n=1, metric="aic")
                all_results.append(results)
            return all_results

        results = benchmark(fit_separately)
        assert len(results) == 3
        assert all(r.count() > 0 for r in results)

    def test_discrete_3_columns_together(
        self, benchmark, spark_session, discrete_df_multi_3col_10k
    ):
        """Benchmark fitting 3 discrete columns in 1 multi-column call."""
        fitter = DiscreteDistributionFitter(spark_session)
        columns = ["col_poisson", "col_nbinom", "col_geom"]

        def fit_together():
            results = fitter.fit(discrete_df_multi_3col_10k, columns=columns)
            _ = results.best_per_column(n=1, metric="aic")
            return results

        result = benchmark(fit_together)
        assert result.count() > 0
        assert len(result.column_names) == 3
