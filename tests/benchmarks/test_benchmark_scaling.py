"""Benchmarks for measuring scaling characteristics.

These benchmarks measure how fit time scales with:
- Data size (25K, 100K, 500K, 1M rows)
- Number of distributions (5, 20, 50, ~90 with default exclusions)

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
            results = fitter.fit(df_1m, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_1m)
        assert result.count() > 0

    def test_fit_500k_rows(self, benchmark, spark_session, df_500k):
        """Benchmark fitting 500K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_500k():
            results = fitter.fit(df_500k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_500k)
        assert result.count() > 0

    def test_fit_100k_rows(self, benchmark, spark_session, df_100k):
        """Benchmark fitting 100K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_100k():
            results = fitter.fit(df_100k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_100k)
        assert result.count() > 0

    def test_fit_25k_rows(self, benchmark, spark_session, df_25k):
        """Benchmark fitting 25K rows."""
        fitter = DistributionFitter(spark_session)

        def fit_25k():
            results = fitter.fit(df_25k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_25k)
        assert result.count() > 0


class TestDistributionCountScaling:
    """Benchmark fit time vs number of distributions.

    Tests are ordered from most to fewest distributions.

    Note: These tests use distribution-aware partitioning (v1.6.1+) which
    automatically calculates optimal partition count based on distribution mix.
    Slow distributions are weighted 3x and interleaved for better load balance.
    """

    def test_fit_default_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting ~90 distributions (with default exclusions).

        Uses distribution-aware partitioning to handle slow distributions
        like burr, t, johnsonsb, etc. (100-160ms each vs ~32ms median).

        Note: This uses default exclusions which remove 20 slow distributions
        (levy_stable, tukeylambda, nct, etc.). For truly ALL 110 distributions,
        see TestSlowDistributionOptimizations.test_fit_all_distributions.
        """
        fitter = DistributionFitter(spark_session)

        def fit_default_dists():
            # Let fitter use distribution-aware partitioning
            results = fitter.fit(df_10k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_default_dists)
        assert result.count() > 0

    def test_fit_50_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting 50 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_50_dists():
            # Let fitter calculate partitions based on distribution mix
            results = fitter.fit(df_10k, "value", max_distributions=50)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_50_dists)
        assert result.count() > 0

    def test_fit_20_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting 20 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_20_dists():
            # Let fitter calculate partitions based on distribution mix
            results = fitter.fit(df_10k, "value", max_distributions=20)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_20_dists)
        assert result.count() > 0

    def test_fit_5_distributions(self, benchmark, spark_session, df_10k):
        """Benchmark fitting only 5 distributions."""
        fitter = DistributionFitter(spark_session)

        def fit_5_dists():
            # Let fitter calculate partitions based on distribution mix
            results = fitter.fit(df_10k, "value", max_distributions=5)
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
                    df_multi_3col_10k, column=col, max_distributions=20
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
                df_multi_3col_10k, columns=columns, max_distributions=20
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
                df_multi_3col_100k, columns=columns, max_distributions=20
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together_100k)
        assert result.count() > 0
        assert len(result.column_names) == 3


class TestLazyMetricsPerformance:
    """Benchmark lazy metrics feature (v1.5.0+).

    Compares eager vs lazy fitting to measure the speedup from
    skipping KS/AD computation.

    Note: Uses ALL distributions (no max_distributions) to capture the
    true benefit - the biggest savings come from slow distributions like
    levy_stable and studentized_range where KS/AD computation is expensive.
    """

    def test_fit_eager_all_metrics(self, benchmark, spark_session, df_100k):
        """Baseline: eager fitting with all metrics computed."""
        fitter = DistributionFitter(spark_session)

        def fit_eager():
            results = fitter.fit(
                df_100k, "value", lazy_metrics=False
            )
            _ = results.best(n=1, metric="aic")
            return results

        result = benchmark(fit_eager)
        best = result.best(n=1)[0]
        assert best.ks_statistic is not None
        assert best.ad_statistic is not None

    def test_fit_lazy_aic_only(self, benchmark, spark_session, df_100k):
        """Lazy fitting with only AIC accessed (no KS/AD computation)."""
        fitter = DistributionFitter(spark_session)

        def fit_lazy_aic():
            results = fitter.fit(
                df_100k, "value", lazy_metrics=True
            )
            _ = results.best(n=1, metric="aic")
            return results

        result = benchmark(fit_lazy_aic)
        best = result.best(n=1, metric="aic")[0]
        assert best.ks_statistic is None  # Not computed
        assert best.aic is not None

    def test_fit_lazy_with_ks_on_demand(self, benchmark, spark_session, df_100k):
        """Lazy fitting with on-demand KS computation for top candidates."""
        fitter = DistributionFitter(spark_session)

        def fit_lazy_with_ks():
            results = fitter.fit(
                df_100k, "value", lazy_metrics=True
            )
            _ = results.best(n=1, metric="ks_statistic")  # Triggers on-demand
            return results

        result = benchmark(fit_lazy_with_ks)
        best = result.best(n=1, metric="ks_statistic")[0]
        assert best.ks_statistic is not None  # Computed on-demand

    def test_fit_lazy_materialize(self, benchmark, spark_session, df_100k):
        """Lazy fitting with full materialization."""
        fitter = DistributionFitter(spark_session)

        def fit_lazy_materialize():
            results = fitter.fit(
                df_100k, "value", lazy_metrics=True
            )
            materialized = results.materialize()
            _ = materialized.best(n=1, metric="ks_statistic")
            return materialized

        result = benchmark(fit_lazy_materialize)
        assert not result.is_lazy
        best = result.best(n=1)[0]
        assert best.ks_statistic is not None


class TestSlowDistributionOptimizations:
    """Benchmark v1.6.1 slow distribution optimizations.

    Compares:
    - Default exclusions (with 3 new slow distributions excluded)
    - All distributions (with partition optimization)

    This demonstrates the ~75% speedup from excluding extremely slow
    distributions (tukeylambda, nct, dpareto_lognorm).
    """

    def test_fit_default_exclusions(self, benchmark, spark_session, df_100k):
        """Benchmark with default exclusions (v1.6.1: 20 exclusions).

        This is the typical use case - excludes extremely slow distributions
        like tukeylambda (~7s), nct (~1.4s), and dpareto_lognorm (~0.5s).
        """
        fitter = DistributionFitter(spark_session)

        def fit_default():
            results = fitter.fit(df_100k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_default)
        assert result.count() > 0

    def test_fit_all_distributions(self, benchmark, spark_session, df_100k):
        """Benchmark with nearly all scipy distributions (107 of 110).

        Uses excluded_distributions to include all but the 3 extremely slow
        distributions that can cause benchmarks to hang:
        - tukeylambda: ~7s+ per fit (ill-conditioned optimization)
        - nct: ~1.4s per fit (non-central t)
        - dpareto_lognorm: ~0.5s per fit (double Pareto-lognormal)

        This demonstrates the fix for excluded_distributions parameter while
        keeping benchmark runtime reasonable. Compared to default (90 dists),
        this adds 17 more distributions that were previously excluded.
        """
        # Exclude only the 3 extremely slow distributions that can hang
        extremely_slow = ("tukeylambda", "nct", "dpareto_lognorm")
        fitter = DistributionFitter(spark_session, excluded_distributions=extremely_slow)

        def fit_all():
            # Let the fitter use distribution-aware partitioning
            results = fitter.fit(df_100k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_all)
        assert result.count() > 0

    def test_interleaving_effectiveness(self, benchmark, spark_session, df_100k):
        """Benchmark to verify interleaving spreads slow distributions.

        Uses a controlled set of distributions with known timing characteristics
        to measure partition balance improvement.
        """
        from spark_bestfit.distributions import DistributionRegistry

        # Get all distributions and verify we have both slow and fast
        registry = DistributionRegistry()
        all_dists = registry.get_distributions()
        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS

        # Count slow distributions in the list
        slow_count = sum(1 for d in all_dists if d in slow_set)

        fitter = DistributionFitter(spark_session)

        def fit_with_interleaving():
            results = fitter.fit(df_100k, "value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_with_interleaving)
        assert result.count() > 0
        # Verify we had slow distributions in the mix
        assert slow_count > 0, "Expected some slow distributions for meaningful test"
