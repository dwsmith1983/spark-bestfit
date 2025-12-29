"""Benchmarks for measuring scaling characteristics.

These benchmarks measure how fit time scales with:
- Data size (25K, 100K, 500K, 1M rows)
- Number of distributions (5, 20, 50, all ~100)

Run with: make benchmark
"""

import pytest

from spark_bestfit import DistributionFitter


class TestDataSizeScaling:
    """Benchmark fit time vs data size.

    Tests are ordered from largest to smallest to minimize warmup effects.
    The 1M test runs first to absorb any remaining JIT compilation overhead.
    """

    def test_fit_1m_rows(self, benchmark, spark_session, df_1m):
        """Benchmark fitting 1M rows."""
        fitter = DistributionFitter(spark_session)

        def fit_1m():
            results = fitter.fit(df_1m, "value", num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_1m)
        assert result.count() > 0

    def test_fit_500k_rows(self, benchmark, spark_session, df_500k):
        """Benchmark fitting 500K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_500k():
            results = fitter.fit(df_500k, "value", num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_500k)
        assert result.count() > 0

    def test_fit_100k_rows(self, benchmark, spark_session, df_100k):
        """Benchmark fitting 100K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_100k():
            results = fitter.fit(df_100k, "value", num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_100k)
        assert result.count() > 0

    def test_fit_25k_rows(self, benchmark, spark_session, df_25k):
        """Benchmark fitting 25K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_25k():
            results = fitter.fit(df_25k, "value", num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_25k)
        assert result.count() > 0


class TestDistributionCountScaling:
    """Benchmark fit time vs number of distributions.

    Tests are ordered from most to fewest distributions.
    """

    def test_fit_all_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting all ~100 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_all_dists():
            results = fitter.fit(df_10k, "value", num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_all_dists)
        assert result.count() > 0

    def test_fit_50_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting 50 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_50_dists():
            results = fitter.fit(df_10k, "value", max_distributions=50, num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_50_dists)
        assert result.count() > 0

    def test_fit_20_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting 20 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_20_dists():
            results = fitter.fit(df_10k, "value", max_distributions=20, num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_20_dists)
        assert result.count() > 0

    def test_fit_5_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting only 5 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_5_dists():
            results = fitter.fit(df_10k, "value", max_distributions=5, num_partitions=8)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_5_dists)
        assert result.count() > 0


class TestMultiColumnEfficiency:
    """Benchmark multi-column fitting efficiency.

    Compares fitting 3 columns separately vs together to demonstrate
    the efficiency gains from shared Spark overhead.
    """

    def test_fit_3_columns_separately(self, benchmark, spark_session, df_multi_3col_10k):
        """Benchmark fitting 3 columns in 3 separate calls (baseline)."""
        fitter = DistributionFitter(spark_session)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_separately():
            all_results = []
            for col in columns:
                results = fitter.fit(
                    df_multi_3col_10k, column=col, max_distributions=20, num_partitions=8
                )
                _ = results.best(n=1)
                all_results.append(results)
            return all_results

        results = benchmark(fit_separately)
        assert len(results) == 3
        assert all(r.count() > 0 for r in results)

    def test_fit_3_columns_together(self, benchmark, spark_session, df_multi_3col_10k):
        """Benchmark fitting 3 columns in 1 multi-column call."""
        fitter = DistributionFitter(spark_session)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_together():
            results = fitter.fit(
                df_multi_3col_10k, columns=columns, max_distributions=20, num_partitions=8
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together)
        assert result.count() > 0
        assert len(result.column_names) == 3

    def test_fit_3_columns_together_100k(self, benchmark, spark_session, df_multi_3col_100k):
        """Benchmark multi-column fitting on larger dataset (100K rows)."""
        fitter = DistributionFitter(spark_session)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_together_100k():
            results = fitter.fit(
                df_multi_3col_100k, columns=columns, max_distributions=20, num_partitions=8
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together_100k)
        assert result.count() > 0
        assert len(result.column_names) == 3
