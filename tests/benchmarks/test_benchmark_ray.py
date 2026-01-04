"""Benchmarks for measuring Ray backend scaling characteristics.

These benchmarks measure how Ray fit time scales with:
- Data size (25K, 100K, 500K, 1M rows)
- Number of distributions (5, 20, 50, ~90 with default exclusions)

Run with: make benchmark-ray

Comparison with Spark:
- Ray has lower startup overhead than Spark
- Both scale with available CPUs
- Ray uses plasma object store; Spark uses JVM heap + off-heap
"""

import pytest

ray = pytest.importorskip("ray")

from spark_bestfit import DistributionFitter


class TestRayDataSizeScaling:
    """Benchmark Ray fit time vs data size.

    Tests are ordered from largest to smallest to minimize warmup effects.
    """

    def test_fit_1m_rows(self, benchmark, ray_backend, ray_df_1m):
        """Benchmark fitting 1M rows with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_1m():
            results = fitter.fit(ray_df_1m, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_1m)
        assert result.count() > 0

    def test_fit_500k_rows(self, benchmark, ray_backend, ray_df_500k):
        """Benchmark fitting 500K rows with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_500k():
            results = fitter.fit(ray_df_500k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_500k)
        assert result.count() > 0

    def test_fit_100k_rows(self, benchmark, ray_backend, ray_df_100k):
        """Benchmark fitting 100K rows with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_100k():
            results = fitter.fit(ray_df_100k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_100k)
        assert result.count() > 0

    def test_fit_25k_rows(self, benchmark, ray_backend, ray_df_25k):
        """Benchmark fitting 25K rows with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_25k():
            results = fitter.fit(ray_df_25k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_25k)
        assert result.count() > 0


class TestRayDistributionCountScaling:
    """Benchmark Ray fit time vs number of distributions.

    Tests are ordered from most to fewest distributions.
    """

    def test_fit_default_distributions(self, benchmark, ray_backend, ray_df_10k):
        """Benchmark fitting ~90 distributions (with default exclusions) with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_default_dists():
            results = fitter.fit(ray_df_10k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_default_dists)
        assert result.count() > 0

    def test_fit_50_distributions(self, benchmark, ray_backend, ray_df_10k):
        """Benchmark fitting 50 distributions with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_50_dists():
            results = fitter.fit(ray_df_10k, column="value", max_distributions=50)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_50_dists)
        assert result.count() > 0

    def test_fit_20_distributions(self, benchmark, ray_backend, ray_df_10k):
        """Benchmark fitting 20 distributions with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_20_dists():
            results = fitter.fit(ray_df_10k, column="value", max_distributions=20)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_20_dists)
        assert result.count() > 0

    def test_fit_5_distributions(self, benchmark, ray_backend, ray_df_10k):
        """Benchmark fitting only 5 distributions with Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_5_dists():
            results = fitter.fit(ray_df_10k, column="value", max_distributions=5)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_5_dists)
        assert result.count() > 0


class TestRayMultiColumnEfficiency:
    """Benchmark Ray multi-column fitting efficiency.

    Compares fitting 3 columns separately vs together.
    """

    def test_fit_3_columns_separately(self, benchmark, ray_backend, ray_df_multi_3col_10k):
        """Benchmark fitting 3 columns in 3 separate calls (baseline) with Ray."""
        fitter = DistributionFitter(backend=ray_backend)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_separately():
            all_results = []
            for col in columns:
                results = fitter.fit(
                    ray_df_multi_3col_10k, column=col, max_distributions=20
                )
                _ = results.best(n=1)
                all_results.append(results)
            return all_results

        results = benchmark(fit_separately)
        assert len(results) == 3
        assert all(r.count() > 0 for r in results)

    def test_fit_3_columns_together(self, benchmark, ray_backend, ray_df_multi_3col_10k):
        """Benchmark fitting 3 columns in 1 multi-column call with Ray."""
        fitter = DistributionFitter(backend=ray_backend)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_together():
            results = fitter.fit(
                ray_df_multi_3col_10k, columns=columns, max_distributions=20
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together)
        assert result.count() > 0
        assert len(result.column_names) == 3

    def test_fit_3_columns_together_100k(self, benchmark, ray_backend, ray_df_multi_3col_100k):
        """Benchmark multi-column fitting on larger dataset (100K rows) with Ray."""
        fitter = DistributionFitter(backend=ray_backend)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_together_100k():
            results = fitter.fit(
                ray_df_multi_3col_100k, columns=columns, max_distributions=20
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together_100k)
        assert result.count() > 0
        assert len(result.column_names) == 3


class TestRayLazyMetricsPerformance:
    """Benchmark Ray lazy metrics feature.

    Compares eager vs lazy fitting to measure the speedup from
    skipping KS/AD computation.
    """

    def test_fit_eager_all_metrics(self, benchmark, ray_backend, ray_df_100k):
        """Baseline: eager fitting with all metrics computed using Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_eager():
            results = fitter.fit(
                ray_df_100k, column="value", lazy_metrics=False
            )
            _ = results.best(n=1, metric="aic")
            return results

        result = benchmark(fit_eager)
        best = result.best(n=1)[0]
        assert best.ks_statistic is not None
        assert best.ad_statistic is not None

    def test_fit_lazy_aic_only(self, benchmark, ray_backend, ray_df_100k):
        """Lazy fitting with only AIC accessed (no KS/AD computation) using Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_lazy_aic():
            results = fitter.fit(
                ray_df_100k, column="value", lazy_metrics=True
            )
            _ = results.best(n=1, metric="aic")
            return results

        result = benchmark(fit_lazy_aic)
        best = result.best(n=1, metric="aic")[0]
        assert best.ks_statistic is None  # Not computed
        assert best.aic is not None

    def test_fit_lazy_with_ks_on_demand(self, benchmark, ray_backend, ray_df_100k):
        """Lazy fitting with on-demand KS computation using Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_lazy_with_ks():
            results = fitter.fit(
                ray_df_100k, column="value", lazy_metrics=True
            )
            _ = results.best(n=1, metric="ks_statistic")  # Triggers on-demand
            return results

        result = benchmark(fit_lazy_with_ks)
        best = result.best(n=1, metric="ks_statistic")[0]
        assert best.ks_statistic is not None  # Computed on-demand


class TestRaySlowDistributionOptimizations:
    """Benchmark Ray slow distribution optimizations.

    Compares:
    - Default exclusions (with 3 new slow distributions excluded)
    - All distributions (with partition optimization)
    """

    def test_fit_default_exclusions(self, benchmark, ray_backend, ray_df_100k):
        """Benchmark with default exclusions using Ray."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_default():
            results = fitter.fit(ray_df_100k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_default)
        assert result.count() > 0

    def test_fit_all_distributions(self, benchmark, ray_backend, ray_df_100k):
        """Benchmark with nearly all scipy distributions using Ray.

        Excludes only the 3 extremely slow distributions that can cause hangs.
        """
        extremely_slow = ("tukeylambda", "nct", "dpareto_lognorm")
        fitter = DistributionFitter(backend=ray_backend, excluded_distributions=extremely_slow)

        def fit_all():
            results = fitter.fit(ray_df_100k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_all)
        assert result.count() > 0


# =============================================================================
# Ray Dataset Benchmarks (distributed comparison with Spark)
# =============================================================================


class TestRayDatasetDataSizeScaling:
    """Benchmark Ray Dataset fit time vs data size.

    Uses Ray Datasets (distributed) instead of pandas DataFrames for
    a fair comparison with Spark's distributed DataFrames.
    """

    def test_dataset_fit_1m_rows(self, benchmark, ray_backend, ray_dataset_1m):
        """Benchmark fitting 1M rows with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_1m():
            results = fitter.fit(ray_dataset_1m, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_1m)
        assert result.count() > 0

    def test_dataset_fit_500k_rows(self, benchmark, ray_backend, ray_dataset_500k):
        """Benchmark fitting 500K rows with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_500k():
            results = fitter.fit(ray_dataset_500k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_500k)
        assert result.count() > 0

    def test_dataset_fit_100k_rows(self, benchmark, ray_backend, ray_dataset_100k):
        """Benchmark fitting 100K rows with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_100k():
            results = fitter.fit(ray_dataset_100k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_100k)
        assert result.count() > 0

    def test_dataset_fit_25k_rows(self, benchmark, ray_backend, ray_dataset_25k):
        """Benchmark fitting 25K rows with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_25k():
            results = fitter.fit(ray_dataset_25k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_25k)
        assert result.count() > 0


class TestRayDatasetDistributionCountScaling:
    """Benchmark Ray Dataset fit time vs number of distributions."""

    def test_dataset_fit_default_distributions(self, benchmark, ray_backend, ray_dataset_10k):
        """Benchmark fitting ~90 distributions with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_default_dists():
            results = fitter.fit(ray_dataset_10k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_default_dists)
        assert result.count() > 0

    def test_dataset_fit_50_distributions(self, benchmark, ray_backend, ray_dataset_10k):
        """Benchmark fitting 50 distributions with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_50_dists():
            results = fitter.fit(ray_dataset_10k, column="value", max_distributions=50)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_50_dists)
        assert result.count() > 0

    def test_dataset_fit_20_distributions(self, benchmark, ray_backend, ray_dataset_10k):
        """Benchmark fitting 20 distributions with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_20_dists():
            results = fitter.fit(ray_dataset_10k, column="value", max_distributions=20)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_20_dists)
        assert result.count() > 0

    def test_dataset_fit_5_distributions(self, benchmark, ray_backend, ray_dataset_10k):
        """Benchmark fitting only 5 distributions with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)

        def fit_5_dists():
            results = fitter.fit(ray_dataset_10k, column="value", max_distributions=5)
            _ = results.best(n=1)
            return results

        result = benchmark(fit_5_dists)
        assert result.count() > 0


class TestRayDatasetMultiColumnEfficiency:
    """Benchmark Ray Dataset multi-column fitting efficiency."""

    def test_dataset_fit_3_columns_together(self, benchmark, ray_backend, ray_dataset_multi_3col_10k):
        """Benchmark fitting 3 columns with Ray Dataset."""
        fitter = DistributionFitter(backend=ray_backend)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_together():
            results = fitter.fit(
                ray_dataset_multi_3col_10k, columns=columns, max_distributions=20
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together)
        assert result.count() > 0
        assert len(result.column_names) == 3

    def test_dataset_fit_3_columns_together_100k(self, benchmark, ray_backend, ray_dataset_multi_3col_100k):
        """Benchmark multi-column fitting on larger Ray Dataset (100K rows)."""
        fitter = DistributionFitter(backend=ray_backend)
        columns = ["col_normal", "col_exp", "col_gamma"]

        def fit_together_100k():
            results = fitter.fit(
                ray_dataset_multi_3col_100k, columns=columns, max_distributions=20
            )
            _ = results.best_per_column(n=1)
            return results

        result = benchmark(fit_together_100k)
        assert result.count() > 0
        assert len(result.column_names) == 3


class TestRayDatasetSlowDistributionOptimizations:
    """Benchmark Ray Dataset slow distribution optimizations.

    Compares default exclusions vs all distributions for Ray Dataset.
    """

    def test_dataset_fit_all_distributions(self, benchmark, ray_backend, ray_dataset_100k):
        """Benchmark with nearly all scipy distributions using Ray Dataset.

        Excludes only the 3 extremely slow distributions that can cause hangs.
        """
        extremely_slow = ("tukeylambda", "nct", "dpareto_lognorm")
        fitter = DistributionFitter(backend=ray_backend, excluded_distributions=extremely_slow)

        def fit_all():
            results = fitter.fit(ray_dataset_100k, column="value")
            _ = results.best(n=1)
            return results

        result = benchmark(fit_all)
        assert result.count() > 0
