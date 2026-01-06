"""Tests for lazy metrics feature (v1.5.0)."""

import warnings

import numpy as np
import pytest

# Skip all tests if pyspark not installed
pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from spark_bestfit import DiscreteDistributionFitter, DistributionFitter

# Mark all tests in this module as requiring Spark
pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    """Create or get SparkSession for testing."""
    return SparkSession.builder.master("local[2]").appName("test_lazy_metrics").getOrCreate()


@pytest.fixture(scope="module")
def sample_df(spark):
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=10, size=1000)
    return spark.createDataFrame([(float(x),) for x in data], ["value"])


@pytest.fixture(scope="module")
def discrete_df(spark):
    """Create sample DataFrame with count data for testing."""
    np.random.seed(42)
    data = np.random.poisson(lam=5, size=500)
    return spark.createDataFrame([(int(x),) for x in data], ["counts"])


class TestLazyMetricsContinuous:
    """Tests for lazy metrics with continuous distributions."""

    def test_lazy_metrics_default_false(self, spark, sample_df):
        """Test that lazy_metrics defaults to False (metrics computed)."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3)

        best = results.best(n=1)[0]
        assert best.ks_statistic is not None
        assert best.ad_statistic is not None

    def test_lazy_metrics_true_skips_computation(self, spark, sample_df):
        """Test that lazy_metrics=True skips KS/AD computation."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Use AIC metric to get best (since KS is None)
        best = results.best(n=1, metric="aic")[0]
        assert best.ks_statistic is None
        assert best.ad_statistic is None
        assert best.pvalue is None
        assert best.ad_pvalue is None

    def test_lazy_metrics_aic_bic_sse_still_computed(self, spark, sample_df):
        """Test that AIC/BIC/SSE are still computed with lazy_metrics=True."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        best = results.best(n=1, metric="aic")[0]
        assert best.aic is not None
        assert best.bic is not None
        assert best.sse is not None
        assert np.isfinite(best.aic)
        assert np.isfinite(best.bic)
        assert np.isfinite(best.sse)

    def test_lazy_metrics_best_computes_ks_on_demand(self, spark, sample_df):
        """Test that best() computes KS on-demand when lazy_metrics=True."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Request best by KS - should trigger on-demand computation
        best = results.best(n=1, metric="ks_statistic")[0]

        # KS should now be computed (not None)
        assert best.ks_statistic is not None
        assert best.pvalue is not None
        assert best.ad_statistic is not None
        assert np.isfinite(best.ks_statistic)
        assert np.isfinite(best.pvalue)

    def test_lazy_metrics_best_no_warn_for_aic(self, spark, sample_df):
        """Test that best() does not warn when sorting by AIC with lazy metrics."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.best(n=1, metric="aic")
            # Should not have the lazy metrics warning
            lazy_warnings = [warning for warning in w if "lazy_metrics" in str(warning.message)]
            assert len(lazy_warnings) == 0

    def test_lazy_metrics_filter_warns_for_ks_threshold(self, spark, sample_df):
        """Test that filter() warns when filtering by KS with lazy metrics."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.filter(ks_threshold=0.1)
            # Find our specific warning
            lazy_warnings = [warning for warning in w if "lazy_metrics=True" in str(warning.message)]
            assert len(lazy_warnings) >= 1

    def test_lazy_metrics_filter_no_warn_for_aic(self, spark, sample_df):
        """Test that filter() does not warn when filtering by AIC with lazy metrics."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.filter(aic_threshold=1000)
            lazy_warnings = [warning for warning in w if "lazy_metrics" in str(warning.message)]
            assert len(lazy_warnings) == 0


class TestLazyMetricsDiscrete:
    """Tests for lazy metrics with discrete distributions."""

    def test_discrete_lazy_metrics_skips_ks(self, spark, discrete_df):
        """Test that lazy_metrics=True skips KS computation for discrete."""
        fitter = DiscreteDistributionFitter(spark)
        results = fitter.fit(discrete_df, column="counts", max_distributions=3, lazy_metrics=True)

        best = results.best(n=1, metric="aic")[0]
        assert best.ks_statistic is None
        assert best.pvalue is None
        # AD is always None for discrete
        assert best.ad_statistic is None

    def test_discrete_lazy_metrics_aic_bic_computed(self, spark, discrete_df):
        """Test that AIC/BIC are computed for discrete with lazy_metrics=True."""
        fitter = DiscreteDistributionFitter(spark)
        results = fitter.fit(discrete_df, column="counts", max_distributions=3, lazy_metrics=True)

        best = results.best(n=1, metric="aic")[0]
        assert best.aic is not None
        assert best.bic is not None
        assert np.isfinite(best.aic)
        assert np.isfinite(best.bic)


class TestLazyMetricsMultiColumn:
    """Tests for lazy metrics with multi-column fitting."""

    def test_lazy_metrics_multi_column(self, spark):
        """Test lazy_metrics with multi-column fitting."""
        np.random.seed(42)
        data = [(float(np.random.normal(50, 10)), float(np.random.exponential(5))) for _ in range(500)]
        df = spark.createDataFrame(data, ["col_a", "col_b"])

        fitter = DistributionFitter(spark)
        results = fitter.fit(df, columns=["col_a", "col_b"], max_distributions=3, lazy_metrics=True)

        # Check both columns have lazy metrics
        for col in ["col_a", "col_b"]:
            col_results = results.for_column(col)
            best = col_results.best(n=1, metric="aic")[0]
            assert best.ks_statistic is None
            assert best.aic is not None


class TestLazyMetricsBounded:
    """Tests for lazy metrics combined with bounded fitting."""

    def test_lazy_metrics_with_bounded(self, spark, sample_df):
        """Test lazy_metrics combined with bounded fitting."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(
            sample_df,
            column="value",
            max_distributions=3,
            bounded=True,
            lower_bound=0,
            upper_bound=100,
            lazy_metrics=True,
        )

        best = results.best(n=1, metric="aic")[0]
        # Lazy metrics
        assert best.ks_statistic is None
        assert best.ad_statistic is None
        # Bounds preserved
        assert best.lower_bound == 0.0
        assert best.upper_bound == 100.0
        # AIC/BIC computed
        assert best.aic is not None


class TestTrueLazyBehavior:
    """Tests for true on-demand lazy metric computation (v1.5.0)."""

    def test_is_lazy_property(self, spark, sample_df):
        """Test is_lazy property reflects lazy context presence."""
        fitter = DistributionFitter(spark)

        # Non-lazy results
        results_eager = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=False)
        assert not results_eager.is_lazy

        # Lazy results
        results_lazy = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)
        assert results_lazy.is_lazy

    def test_materialize_computes_all_metrics(self, spark, sample_df):
        """Test materialize() computes all lazy metrics."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Before materialization, KS is None
        best_before = results.best(n=1, metric="aic")[0]
        assert best_before.ks_statistic is None

        # Materialize
        materialized = results.materialize()

        # After materialization, is_lazy should be False
        assert not materialized.is_lazy

        # All metrics should be computed
        best_after = materialized.best(n=1, metric="ks_statistic")[0]
        assert best_after.ks_statistic is not None
        assert best_after.ad_statistic is not None
        assert np.isfinite(best_after.ks_statistic)

    def test_for_column_preserves_lazy_context(self, spark):
        """Test for_column() preserves lazy context for filtered results."""
        np.random.seed(42)
        data = [(float(np.random.normal(50, 10)), float(np.random.exponential(5))) for _ in range(500)]
        df = spark.createDataFrame(data, ["col_a", "col_b"])

        fitter = DistributionFitter(spark)
        results = fitter.fit(df, columns=["col_a", "col_b"], max_distributions=3, lazy_metrics=True)

        # Get results for col_a
        col_a_results = results.for_column("col_a")
        assert col_a_results.is_lazy

        # Request KS should trigger on-demand computation
        best_ks = col_a_results.best(n=1, metric="ks_statistic")[0]
        assert best_ks.ks_statistic is not None
        assert best_ks.column_name == "col_a"

    def test_best_ad_statistic_computed_on_demand(self, spark, sample_df):
        """Test that best() computes AD on-demand when lazy_metrics=True."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Request best by AD - should trigger on-demand computation
        best = results.best(n=1, metric="ad_statistic")[0]

        # AD should now be computed (not None)
        assert best.ad_statistic is not None
        assert np.isfinite(best.ad_statistic)

    def test_filter_preserves_lazy_context(self, spark, sample_df):
        """Test filter() preserves lazy context for filtered results."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=5, lazy_metrics=True)

        # Filter by AIC
        filtered = results.filter(aic_threshold=100000)
        assert filtered.is_lazy

        # Request KS should still work
        best_ks = filtered.best(n=1, metric="ks_statistic")[0]
        assert best_ks.ks_statistic is not None


class TestDiscreteTrueLazy:
    """Tests for true lazy behavior with discrete distributions."""

    def test_discrete_computes_ks_on_demand(self, spark, discrete_df):
        """Test discrete distributions compute KS on-demand."""
        fitter = DiscreteDistributionFitter(spark)
        results = fitter.fit(discrete_df, column="counts", max_distributions=3, lazy_metrics=True)

        # Request best by KS
        best = results.best(n=1, metric="ks_statistic")[0]

        # KS should be computed
        assert best.ks_statistic is not None
        assert best.pvalue is not None

        # AD is always None for discrete
        assert best.ad_statistic is None

    def test_discrete_materialize(self, spark, discrete_df):
        """Test materialize() for discrete distributions."""
        fitter = DiscreteDistributionFitter(spark)
        results = fitter.fit(discrete_df, column="counts", max_distributions=3, lazy_metrics=True)

        # Materialize
        materialized = results.materialize()
        assert not materialized.is_lazy

        # KS should be computed
        best = materialized.best(n=1, metric="ks_statistic")[0]
        assert best.ks_statistic is not None


class TestLazyMetricsEdgeCases:
    """Edge case tests for lazy metrics robustness."""

    def test_reproducibility_same_seed_same_metrics(self, spark, sample_df):
        """Test that same seed produces same metrics on repeated calls."""
        fitter = DistributionFitter(spark, random_seed=42)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Get best by KS twice - should produce same result
        best1 = results.best(n=1, metric="ks_statistic")[0]
        best2 = results.best(n=1, metric="ks_statistic")[0]

        assert best1.distribution == best2.distribution
        assert best1.ks_statistic == best2.ks_statistic
        assert best1.pvalue == best2.pvalue

    def test_materialize_idempotent(self, spark, sample_df):
        """Test that calling materialize() twice is safe."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Materialize twice
        materialized1 = results.materialize()
        materialized2 = materialized1.materialize()

        # Both should work and be non-lazy
        assert not materialized1.is_lazy
        assert not materialized2.is_lazy

        # Results should be identical
        best1 = materialized1.best(n=1, metric="ks_statistic")[0]
        best2 = materialized2.best(n=1, metric="ks_statistic")[0]
        assert best1.ks_statistic == best2.ks_statistic

    def test_non_lazy_materialize_passthrough(self, spark, sample_df):
        """Test that materialize() on non-lazy results is a no-op."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=False)

        assert not results.is_lazy

        # Materialize should just return the same result
        materialized = results.materialize()
        assert not materialized.is_lazy

    def test_bounded_multi_column_lazy(self, spark):
        """Test lazy metrics with bounded + multi-column fitting."""
        np.random.seed(42)
        # Create data with different ranges per column
        data = [
            (float(np.random.uniform(0, 100)), float(np.random.uniform(10, 50)))
            for _ in range(500)
        ]
        df = spark.createDataFrame(data, ["percent", "age"])

        fitter = DistributionFitter(spark)
        results = fitter.fit(
            df,
            columns=["percent", "age"],
            max_distributions=3,
            bounded=True,
            lower_bound={"percent": 0.0, "age": 10.0},
            upper_bound={"percent": 100.0, "age": 50.0},
            lazy_metrics=True,
        )

        assert results.is_lazy

        # Test on-demand computation for each column
        for col in ["percent", "age"]:
            col_results = results.for_column(col)
            assert col_results.is_lazy

            best_ks = col_results.best(n=1, metric="ks_statistic")[0]
            assert best_ks.ks_statistic is not None
            assert best_ks.column_name == col

    def test_lazy_with_empty_results_after_filter(self, spark, sample_df):
        """Test lazy metrics behavior when filter returns empty results."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Apply a very strict filter that should return empty results
        filtered = results.filter(aic_threshold=-1000000)  # Impossible threshold

        # Should still be lazy
        assert filtered.is_lazy

        # Best should return empty list, not error
        best = filtered.best(n=1, metric="aic")
        assert best == []

    def test_best_n_greater_than_results(self, spark, sample_df):
        """Test best(n=100) when fewer distributions fitted."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=True)

        # Request more than available
        best = results.best(n=100, metric="ks_statistic")

        # Should return all available (3), not error
        assert len(best) == 3
        for result in best:
            assert result.ks_statistic is not None

    def test_materialize_all_rows_have_metrics(self, spark, sample_df):
        """Test materialize() computes metrics for ALL rows, not just accessed ones.

        This is a regression test to catch bugs in DataFrame row processing
        where metrics might not be computed due to grouping/keying issues.
        """
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=5, lazy_metrics=True)

        # Before materialization, verify KS is None in raw DataFrame
        raw_rows_before = results.df.collect()
        assert all(row["ks_statistic"] is None for row in raw_rows_before), \
            "KS should be None before materialization"

        # Materialize
        materialized = results.materialize()

        # After materialization, check EVERY row in the raw DataFrame
        # This bypasses best() which has its own lazy computation logic
        raw_rows_after = materialized.df.collect()

        # All rows must have non-null KS statistic
        for row in raw_rows_after:
            assert row["ks_statistic"] is not None, \
                f"Distribution '{row['distribution']}' has ks_statistic=None after materialize()"
            assert row["ad_statistic"] is not None, \
                f"Distribution '{row['distribution']}' has ad_statistic=None after materialize()"
            assert row["pvalue"] is not None, \
                f"Distribution '{row['distribution']}' has pvalue=None after materialize()"

    def test_materialize_preserves_column_name(self, spark):
        """Test materialize() preserves column_name field correctly.

        Regression test for bugs in column name handling during materialization.
        """
        np.random.seed(42)
        data = [(float(np.random.normal(50, 10)),) for _ in range(500)]
        df = spark.createDataFrame(data, ["my_custom_column"])

        fitter = DistributionFitter(spark)
        results = fitter.fit(df, column="my_custom_column", max_distributions=3, lazy_metrics=True)

        # Verify column name is set before materialization
        raw_before = results.df.collect()
        assert all(row["column_name"] == "my_custom_column" for row in raw_before)

        # Materialize
        materialized = results.materialize()

        # Verify column name is preserved after materialization
        raw_after = materialized.df.collect()
        assert all(row["column_name"] == "my_custom_column" for row in raw_after), \
            "column_name should be preserved after materialization"

        # And metrics should be computed
        assert all(row["ks_statistic"] is not None for row in raw_after)

    def test_best_skips_null_ks_values(self, spark, sample_df):
        """Test best() returns valid distributions when some have NULL ks_statistic.

        Regression test for NULLS FIRST issue - when some distributions produce
        non-finite KS statistics (resulting in NULL), best() should skip them
        and return distributions with valid metrics.
        """
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=10, lazy_metrics=False)

        # Get best by ks_statistic
        best = results.best(n=3, metric="ks_statistic")

        # All returned results should have non-None ks_statistic
        # (NULL values should be at the end, not included in top N)
        for result in best:
            if result.ks_statistic is None:
                # If we get a None, it means all non-None values were exhausted
                # This is acceptable - we just want to ensure NULLs don't come first
                break
            assert np.isfinite(result.ks_statistic), \
                f"ks_statistic should be finite, got {result.ks_statistic}"

    def test_best_validates_n_parameter(self, spark, sample_df):
        """Test best() validates n parameter is positive.

        Regression test for missing input validation - best(n=-1) should
        raise ValueError, not leak ugly Spark AnalysisException.
        """
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=3, lazy_metrics=False)

        # n=0 should raise ValueError
        with pytest.raises(ValueError, match="n must be a positive integer"):
            results.best(n=0)

        # n=-1 should raise ValueError
        with pytest.raises(ValueError, match="n must be a positive integer"):
            results.best(n=-1)


class TestFitResultsUnpersist:
    """Tests for FitResults.unpersist() method."""

    def test_unpersist_returns_self(self, spark, sample_df):
        """Test that unpersist() returns self for method chaining."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2)

        # unpersist() should return self
        returned = results.unpersist()
        assert returned is results

    def test_unpersist_blocking(self, spark, sample_df):
        """Test that unpersist(blocking=True) works without error."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2)

        # Should not raise any exceptions
        results.unpersist(blocking=True)

    def test_unpersist_after_materialize(self, spark, sample_df):
        """Test unpersist after materialize for lazy metrics workflow."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2, lazy_metrics=True)

        # Typical workflow: materialize metrics, then unpersist
        results.materialize()
        returned = results.unpersist()

        # Should return self for chaining
        assert returned is results

    def test_unpersist_method_chaining(self, spark, sample_df):
        """Test method chaining with unpersist()."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2, lazy_metrics=True)

        # Chain materialize() and unpersist()
        results.materialize().unpersist()


class TestLazyMetricsLifecycle:
    """Tests for LazyMetrics DataFrame lifecycle error handling.

    When lazy_metrics=True, the FitResults holds a reference to the source
    DataFrame. If that DataFrame becomes unavailable (unpersisted, deleted,
    corrupted), accessing lazy metrics should raise a clear RuntimeError.
    """

    def test_invalid_source_df_raises_runtime_error(self, spark, sample_df):
        """Accessing lazy metrics with invalid source_df raises RuntimeError."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2, lazy_metrics=True)

        # Verify lazy context exists
        assert results.is_lazy

        # Corrupt the source_df reference in the lazy context
        # This simulates what happens when the DataFrame is unavailable
        for context in results._lazy_contexts.values():
            context.source_df = None  # Invalidate the reference

        # Attempting to compute lazy metrics should raise RuntimeError
        with pytest.raises(RuntimeError, match="Failed to recreate sample"):
            results.best(n=1, metric="ks_statistic")

    def test_invalid_source_df_error_message_helpful(self, spark, sample_df):
        """RuntimeError message includes guidance about materialize()."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2, lazy_metrics=True)

        # Corrupt the source_df reference
        for context in results._lazy_contexts.values():
            context.source_df = None

        # Error message should mention materialize() as the solution
        with pytest.raises(RuntimeError, match="materialize.*before unpersisting"):
            results.best(n=1, metric="ks_statistic")

    def test_materialize_before_invalidation_succeeds(self, spark, sample_df):
        """Materializing before source_df invalidation preserves metrics."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2, lazy_metrics=True)

        # Materialize while source_df is still valid
        materialized = results.materialize()

        # Now corrupt the original results' source_df
        for context in results._lazy_contexts.values():
            context.source_df = None

        # Materialized results should still work (no lazy context)
        assert not materialized.is_lazy
        best = materialized.best(n=1, metric="ks_statistic")
        assert len(best) == 1
        assert best[0].ks_statistic is not None

    def test_non_ks_metrics_work_without_source_df(self, spark, sample_df):
        """Non-lazy metrics (SSE, AIC, BIC) work even with invalid source_df."""
        fitter = DistributionFitter(spark)
        results = fitter.fit(sample_df, column="value", max_distributions=2, lazy_metrics=True)

        # Corrupt the source_df reference
        for context in results._lazy_contexts.values():
            context.source_df = None

        # SSE, AIC, BIC are computed during fit, not lazily
        # These should still work
        best_sse = results.best(n=1, metric="sse")
        assert len(best_sse) == 1
        assert best_sse[0].sse is not None

        best_aic = results.best(n=1, metric="aic")
        assert len(best_aic) == 1
        assert best_aic[0].aic is not None
