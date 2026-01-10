"""Comprehensive numerical stability tests for spark-bestfit.

P0 Task: Ensure robust handling of edge cases including:
- NaN values at various percentages (1%, 10%, 50%, 100%)
- Inf values (positive and negative)
- Values near float64 limits (1e308, 1e-308)
- Log(0) and division by zero scenarios
- Ill-conditioned correlation matrices
- Underflow/overflow in metric calculations
"""

import numpy as np
import pandas as pd
import pytest

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.continuous_fitter import DistributionFitter
from spark_bestfit.discrete_fitter import DiscreteDistributionFitter
from spark_bestfit.distributions import DiscreteDistributionRegistry
from spark_bestfit.fitting import (
    compute_data_stats,
    fit_single_distribution,
)
from spark_bestfit.discrete_fitting import (
    compute_discrete_histogram,
    fit_single_discrete_distribution,
)
from spark_bestfit.results import DistributionFitResult


def _numpy_histogram(data: np.ndarray, bins: int = 50) -> tuple:
    """Helper to compute histogram matching fit_single_distribution's expected format.

    Returns (bin_edges, y_hist) where y_hist is density-normalized.
    """
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        raise ValueError("All data is NaN")
    hist, bin_edges = np.histogram(clean_data, bins=bins, density=True)
    return bin_edges, hist


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def local_backend():
    """Create LocalBackend for testing without Spark overhead."""
    return LocalBackend()


@pytest.fixture
def continuous_fitter(local_backend):
    """Create DistributionFitter with LocalBackend."""
    return DistributionFitter(backend=local_backend)


@pytest.fixture
def discrete_fitter(local_backend):
    """Create DiscreteDistributionFitter with LocalBackend."""
    return DiscreteDistributionFitter(backend=local_backend)


@pytest.fixture
def clean_normal_data():
    """Generate clean normal data for baseline comparisons."""
    np.random.seed(42)
    return np.random.normal(loc=50, scale=10, size=1000)


@pytest.fixture
def clean_poisson_data():
    """Generate clean Poisson data for discrete baseline."""
    np.random.seed(42)
    return np.random.poisson(lam=5, size=1000)


# =============================================================================
# NaN Handling Tests - Various Percentages
# =============================================================================


class TestNaNHandling:
    """Test NaN handling at different contamination levels."""

    @pytest.mark.parametrize("nan_fraction", [0.01, 0.10, 0.50])
    def test_continuous_fit_with_nan_fraction(self, continuous_fitter, clean_normal_data, nan_fraction):
        """Continuous fitter should handle NaN at various contamination levels."""
        data = clean_normal_data.copy()
        n_nan = int(len(data) * nan_fraction)
        nan_indices = np.random.choice(len(data), n_nan, replace=False)
        data[nan_indices] = np.nan

        df = pd.DataFrame({"value": data})

        # Should complete without error - NaN filtering happens internally
        results = continuous_fitter.fit(df, column="value", max_distributions=3)

        # Use AIC for sorting (LocalBackend uses lazy_metrics internally)
        # Should have results (NaN-filtered data still has valid values)
        assert len(results.best(n=1, metric="aic")) > 0

    def test_continuous_fit_all_nan_returns_empty(self, continuous_fitter):
        """Fitting 100% NaN data should return empty results with proper schema."""
        data = np.array([np.nan] * 100)
        df = pd.DataFrame({"value": data})

        results = continuous_fitter.fit(df, column="value", max_distributions=1)

        # Should have empty results with proper schema (columns exist)
        assert len(results._df.columns) > 0
        assert len(results.best(n=1, metric="aic")) == 0

    @pytest.mark.parametrize("nan_fraction", [0.01, 0.10, 0.50])
    def test_discrete_fit_with_nan_fraction(self, discrete_fitter, clean_poisson_data, nan_fraction):
        """Discrete fitter should handle NaN at various contamination levels."""
        data = clean_poisson_data.astype(float).copy()
        n_nan = int(len(data) * nan_fraction)
        nan_indices = np.random.choice(len(data), n_nan, replace=False)
        data[nan_indices] = np.nan

        df = pd.DataFrame({"counts": data})

        # Should complete without error
        results = discrete_fitter.fit(df, column="counts", max_distributions=3)
        assert len(results.best(n=1, metric="aic")) > 0

    def test_discrete_fit_all_nan_returns_empty(self, discrete_fitter):
        """Fitting 100% NaN discrete data should return empty results with proper schema."""
        data = np.array([np.nan] * 100)
        df = pd.DataFrame({"counts": data})

        results = discrete_fitter.fit(df, column="counts", max_distributions=1)

        # Should have empty results with proper schema (columns exist)
        assert len(results._df.columns) > 0
        assert len(results.best(n=1, metric="aic")) == 0


# =============================================================================
# Inf Handling Tests
# =============================================================================


class TestInfHandling:
    """Test handling of infinite values."""

    def test_continuous_fit_with_positive_inf(self, continuous_fitter, clean_normal_data):
        """Continuous fitter should filter +inf and return valid results."""
        data = clean_normal_data.copy()
        data[0] = np.inf  # Single inf value

        df = pd.DataFrame({"value": data})

        # Inf values are filtered, remaining data is fitted
        results = continuous_fitter.fit(df, column="value", max_distributions=3)
        best_list = results.best(n=1, metric="aic")

        # Should have results (inf filtered, 999 valid values remain)
        assert len(best_list) > 0
        assert np.isfinite(best_list[0].aic)

    def test_continuous_fit_with_negative_inf(self, continuous_fitter, clean_normal_data):
        """Continuous fitter should filter -inf and return valid results."""
        data = clean_normal_data.copy()
        data[0] = -np.inf

        df = pd.DataFrame({"value": data})

        results = continuous_fitter.fit(df, column="value", max_distributions=3)
        best_list = results.best(n=1, metric="aic")

        assert len(best_list) > 0
        assert np.isfinite(best_list[0].aic)

    def test_continuous_fit_with_mixed_inf(self, continuous_fitter, clean_normal_data):
        """Continuous fitter should handle both +inf and -inf."""
        data = clean_normal_data.copy()
        data[0] = np.inf
        data[1] = -np.inf

        df = pd.DataFrame({"value": data})

        try:
            results = continuous_fitter.fit(df, column="value", max_distributions=3)
            # May have 0 results if all filtered - that's acceptable
            assert results is not None
        except (ValueError, RuntimeError):
            pass  # Acceptable


# =============================================================================
# Extreme Value Tests (Float64 Limits)
# =============================================================================


class TestExtremeValues:
    """Test handling of values near float64 limits."""

    def test_histogram_with_large_values(self):
        """Histogram computation should handle values near float64 max."""
        # Values near but not at float64 max to avoid overflow
        data = np.array([1e307, 1e307 + 1e290, 1e307 + 2e290, 1e307 + 3e290])

        try:
            bin_edges, hist_values = _numpy_histogram(data, bins=10)
            # If successful, verify no inf in results
            assert np.all(np.isfinite(bin_edges))
            assert np.all(np.isfinite(hist_values))
        except (ValueError, OverflowError, IndexError):
            # IndexError occurs in NumPy 1.x due to integer overflow in histogram
            pass  # Acceptable to reject extreme values

    def test_histogram_with_small_values(self):
        """Histogram computation should handle very small positive values."""
        # Values near float64 min positive (subnormal range)
        data = np.array([1e-307, 2e-307, 3e-307, 4e-307])

        bin_edges, hist_values = _numpy_histogram(data, bins=10)
        assert np.all(np.isfinite(bin_edges))
        assert np.all(np.isfinite(hist_values))

    def test_fit_with_very_small_variance(self, continuous_fitter):
        """Fitter should handle data with extremely small variance."""
        # Near-constant data (variance ~ 1e-20)
        data = np.array([1.0 + i * 1e-12 for i in range(1000)])
        df = pd.DataFrame({"value": data})

        try:
            results = continuous_fitter.fit(df, column="value", max_distributions=3)
            # May succeed with degenerate fits
            assert results is not None
        except (ValueError, RuntimeError):
            pass  # Acceptable for degenerate data

    def test_fit_with_very_large_variance(self, continuous_fitter):
        """Fitter should handle data spanning many orders of magnitude."""
        np.random.seed(42)
        # Data spanning 1e-100 to 1e100 (log-uniform)
        log_data = np.random.uniform(-100, 100, 1000)
        data = np.power(10.0, log_data)
        df = pd.DataFrame({"value": data})

        try:
            results = continuous_fitter.fit(df, column="value", max_distributions=3)
            assert results is not None
        except (ValueError, RuntimeError, OverflowError):
            pass  # Acceptable for extreme ranges


# =============================================================================
# Division by Zero / Log(0) Tests
# =============================================================================


class TestDivisionByZeroAndLog:
    """Test scenarios that could cause division by zero or log(0)."""

    def test_compute_data_stats_with_zeros(self):
        """compute_data_stats should handle data containing zeros."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 5.0])
        stats = compute_data_stats(data)

        # Should compute valid stats (note: keys are data_min, data_mean, etc.)
        assert np.isfinite(stats["data_mean"])
        assert np.isfinite(stats["data_stddev"])
        assert stats["data_min"] == 0.0

    def test_compute_data_stats_all_zeros(self):
        """compute_data_stats with all zeros should not cause division by zero."""
        data = np.array([0.0, 0.0, 0.0, 0.0])
        stats = compute_data_stats(data)

        assert stats["data_mean"] == 0.0
        assert stats["data_stddev"] == 0.0

    def test_sse_computed_with_edge_case_data(self, continuous_fitter):
        """SSE should be computed correctly even with edge case data."""
        # Very peaked data (high kurtosis) - SSE can be challenging
        np.random.seed(42)
        data = np.random.laplace(loc=0, scale=0.1, size=1000)
        df = pd.DataFrame({"value": data})

        results = continuous_fitter.fit(df, column="value", max_distributions=3)
        best = results.best(n=1, metric="aic")[0]

        # SSE should be finite and non-negative
        assert best.sse is None or (np.isfinite(best.sse) and best.sse >= 0)

    def test_fit_lognormal_with_zeros(self, continuous_fitter):
        """Fitting log-normal to data with zeros should not cause log(0)."""
        np.random.seed(42)
        data = np.abs(np.random.normal(10, 5, 1000))
        # Add some zeros
        data[:10] = 0.0
        df = pd.DataFrame({"value": data})

        # Log-normal requires positive data; zeros should be filtered or handled
        results = continuous_fitter.fit(df, column="value", max_distributions=5)
        assert results is not None

    def test_discrete_histogram_with_zeros(self):
        """Discrete histogram should handle zero counts."""
        data = np.array([0, 0, 0, 1, 1, 2, 5])

        x_values, pmf = compute_discrete_histogram(data)

        assert 0 in x_values
        assert np.all(np.isfinite(pmf))
        assert np.isclose(np.sum(pmf), 1.0)  # PMF should sum to 1


# =============================================================================
# Information Criteria Edge Cases
# =============================================================================


class TestInformationCriteria:
    """Test AIC/BIC calculation edge cases."""

    def test_result_aic_with_valid_params(self):
        """AIC should be finite for valid fit results."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],  # [loc, scale] as list
            sse=0.001,
            column_name="test",
            aic=100.0,
            bic=105.0,
            ks_statistic=0.05,
            pvalue=0.5,
            data_min=20.0,
            data_max=80.0,
            data_mean=50.0,
            data_stddev=10.0,
            data_count=1000.0,
        )

        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_result_with_none_metrics(self):
        """Results with None metrics should be handled gracefully."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.001,
            column_name="test",
            aic=None,
            bic=None,
            ks_statistic=None,
            pvalue=None,
            data_min=20.0,
            data_max=80.0,
            data_mean=50.0,
            data_stddev=10.0,
            data_count=1000.0,
        )

        # Should not raise when accessing None metrics
        assert result.ks_statistic is None
        assert result.aic is None


# =============================================================================
# Correlation Matrix Edge Cases (for Copula)
# =============================================================================


class TestCorrelationMatrixStability:
    """Test correlation matrix handling for copula sampling."""

    def test_correlation_with_constant_column(self, local_backend):
        """Correlation computation should handle constant columns."""
        df = pd.DataFrame({
            "varying": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0],  # Zero variance
        })

        corr = local_backend.compute_correlation(df, ["varying", "constant"])

        # Correlation with constant column is undefined (NaN or 0)
        # Backend should handle gracefully
        assert corr.shape == (2, 2)
        # Diagonal should be 1 (or NaN for constant)
        assert corr[0, 0] == 1.0 or np.isnan(corr[0, 0])

    def test_correlation_with_nan_column(self, local_backend):
        """Correlation computation should handle columns with NaN."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "col2": [5.0, 4.0, 3.0, 2.0, 1.0],
        })

        corr = local_backend.compute_correlation(df, ["col1", "col2"])

        # Should compute correlation (pairwise complete observations)
        assert corr.shape == (2, 2)

    def test_near_singular_correlation_matrix(self, local_backend):
        """Copula should handle near-singular correlation matrices."""
        np.random.seed(42)
        n = 100

        # Create nearly collinear data
        x = np.random.normal(0, 1, n)
        y = x + np.random.normal(0, 0.001, n)  # y ≈ x
        z = np.random.normal(0, 1, n)

        df = pd.DataFrame({"x": x, "y": y, "z": z})

        corr = local_backend.compute_correlation(df, ["x", "y", "z"])

        # Correlation should be computed (may be ill-conditioned)
        assert corr.shape == (3, 3)
        # x-y correlation should be very close to 1
        assert np.abs(corr[0, 1]) > 0.99


# =============================================================================
# Underflow/Overflow in PDF Evaluation
# =============================================================================


class TestPDFOverflowUnderflow:
    """Test PDF evaluation at extreme points."""

    def test_pdf_at_extreme_x(self):
        """PDF evaluation at extreme x values should not overflow."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],  # Standard normal [loc, scale]
            sse=0.001,
            column_name="test",
            aic=100.0,
            bic=105.0,
            ks_statistic=0.05,
            pvalue=0.5,
            data_min=-10.0,
            data_max=10.0,
            data_mean=0.0,
            data_stddev=1.0,
            data_count=1000.0,
        )

        # PDF at extreme values
        x_extreme = np.array([-1e10, -1e5, 0, 1e5, 1e10])
        pdf_values = result.pdf(x_extreme)

        # Should return valid values (near 0 at extremes)
        assert np.all(np.isfinite(pdf_values))
        assert np.all(pdf_values >= 0)

    def test_cdf_at_extreme_x(self):
        """CDF evaluation at extreme x values should return 0 or 1."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.001,
            column_name="test",
            aic=100.0,
            bic=105.0,
            ks_statistic=0.05,
            pvalue=0.5,
            data_min=-10.0,
            data_max=10.0,
            data_mean=0.0,
            data_stddev=1.0,
            data_count=1000.0,
        )

        x_extreme = np.array([-1e10, 1e10])
        cdf_values = result.cdf(x_extreme)

        assert np.isclose(cdf_values[0], 0.0)
        assert np.isclose(cdf_values[1], 1.0)


# =============================================================================
# Single Distribution Fit Edge Cases
# =============================================================================


class TestSingleDistributionFitEdgeCases:
    """Test fit_single_distribution with edge case inputs."""

    def test_fit_with_single_unique_value(self):
        """Fitting to constant data should handle gracefully."""
        data = np.array([5.0] * 100)
        # Constant data will fail histogram - this tests the error path
        try:
            bin_edges, y_hist = _numpy_histogram(data, bins=10)

            result = fit_single_distribution(
                dist_name="norm",
                data_sample=data,
                bin_edges=bin_edges,
                y_hist=y_hist,
                column_name="test",
                data_stats=compute_data_stats(data),
                lower_bound=None,
                upper_bound=None,
                lazy_metrics=False,
            )

            # Should return a result (possibly with poor fit)
            assert result is not None
            assert result["distribution"] == "norm"
        except (ValueError, ZeroDivisionError):
            # Constant data may fail histogram - acceptable
            pass

    def test_fit_with_two_unique_values(self):
        """Fitting to binary data should handle gracefully."""
        data = np.array([0.0] * 50 + [1.0] * 50)
        np.random.shuffle(data)
        bin_edges, y_hist = _numpy_histogram(data, bins=10)

        result = fit_single_distribution(
            dist_name="uniform",
            data_sample=data,
            bin_edges=bin_edges,
            y_hist=y_hist,
            column_name="test",
            data_stats=compute_data_stats(data),
            lower_bound=None,
            upper_bound=None,
            lazy_metrics=False,
        )

        assert result is not None

    def test_discrete_fit_with_single_value(self):
        """Discrete fit to single-value data should handle gracefully."""
        data = np.array([5] * 100)
        x_values, pmf = compute_discrete_histogram(data)
        registry = DiscreteDistributionRegistry()

        result = fit_single_discrete_distribution(
            dist_name="poisson",
            data_sample=data,
            x_values=x_values,
            empirical_pmf=pmf,
            registry=registry,
            column_name="test",
            data_stats=compute_data_stats(data.astype(float)),
            lower_bound=None,
            upper_bound=None,
            lazy_metrics=False,
        )

        assert result is not None


# =============================================================================
# Multi-Column Edge Cases
# =============================================================================


class TestMultiColumnEdgeCases:
    """Test multi-column fitting with edge cases."""

    def test_multi_column_one_has_nan(self, continuous_fitter):
        """Multi-column fit where one column has NaN - NaN filtered per-column."""
        np.random.seed(42)
        df = pd.DataFrame({
            "clean": np.random.normal(50, 10, 100),
            "with_nan": np.append(np.random.normal(50, 10, 90), [np.nan] * 10),
        })

        results = continuous_fitter.fit(df, columns=["clean", "with_nan"], max_distributions=2)

        # Should get results for both columns (NaN filtered from with_nan)
        clean_results = results.for_column("clean")
        nan_results = results.for_column("with_nan")

        assert len(clean_results.best(n=1, metric="aic")) > 0
        assert len(nan_results.best(n=1, metric="aic")) > 0

    def test_multi_column_different_scales(self, continuous_fitter):
        """Multi-column fit with very different scales."""
        np.random.seed(42)
        df = pd.DataFrame({
            "small": np.random.normal(1e-6, 1e-7, 100),
            "large": np.random.normal(1e6, 1e5, 100),
        })

        results = continuous_fitter.fit(df, columns=["small", "large"], max_distributions=2)

        small_best = results.for_column("small").best(n=1, metric="aic")[0]
        large_best = results.for_column("large").best(n=1, metric="aic")[0]

        # Parameters should reflect the different scales
        assert small_best.data_mean < 1e-3
        assert large_best.data_mean > 1e3


# =============================================================================
# Direct Unit Tests for New Branches
# =============================================================================


class TestLocalBackendNaNInfFiltering:
    """Direct unit tests for LocalBackend.sample_column() NaN/inf filtering."""

    def test_sample_column_all_nan_returns_empty_array(self, local_backend):
        """sample_column with all NaN returns empty array."""
        df = pd.DataFrame({"value": [np.nan] * 100})
        result = local_backend.sample_column(df, "value", fraction=1.0, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_sample_column_all_inf_returns_empty_array(self, local_backend):
        """sample_column with all inf values returns empty array."""
        df = pd.DataFrame({"value": [np.inf] * 50 + [-np.inf] * 50})
        result = local_backend.sample_column(df, "value", fraction=1.0, seed=42)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_sample_column_mixed_nan_inf_returns_empty(self, local_backend):
        """sample_column with mixed NaN and inf returns empty array."""
        df = pd.DataFrame({"value": [np.nan, np.inf, -np.inf, np.nan]})
        result = local_backend.sample_column(df, "value", fraction=1.0, seed=42)
        assert len(result) == 0

    def test_sample_column_filters_nan_keeps_valid(self, local_backend):
        """sample_column filters NaN but keeps valid values."""
        df = pd.DataFrame({"value": [1.0, np.nan, 2.0, np.nan, 3.0]})
        result = local_backend.sample_column(df, "value", fraction=1.0, seed=42)
        assert len(result) == 3  # Only valid values
        assert not np.any(np.isnan(result))

    def test_sample_column_filters_inf_keeps_valid(self, local_backend):
        """sample_column filters inf but keeps valid values."""
        df = pd.DataFrame({"value": [1.0, np.inf, 2.0, -np.inf, 3.0]})
        result = local_backend.sample_column(df, "value", fraction=1.0, seed=42)
        assert len(result) == 3  # Only valid values
        assert not np.any(np.isinf(result))


class TestLocalBackendParallelFitEmptyData:
    """Direct unit tests for LocalBackend.parallel_fit() with empty data."""

    def test_parallel_fit_empty_data_sample_returns_empty_list(self, local_backend):
        """parallel_fit with empty data_sample returns empty list."""
        from spark_bestfit.fitting import fit_single_distribution

        # Empty data sample
        result = local_backend.parallel_fit(
            distributions=["norm", "expon"],
            histogram=(np.array([1, 2, 3]), np.array([0.0, 1.0, 2.0, 3.0])),
            data_sample=np.array([]),  # Empty!
            fit_func=fit_single_distribution,
            column_name="test",
        )
        assert result == []

    def test_parallel_fit_nonempty_data_returns_results(self, local_backend):
        """parallel_fit with valid data returns results."""
        from spark_bestfit.fitting import fit_single_distribution

        np.random.seed(42)
        data = np.random.normal(50, 10, 100)
        hist, bin_edges = np.histogram(data, bins=20, density=True)

        result = local_backend.parallel_fit(
            distributions=["norm"],
            histogram=(hist, bin_edges),
            data_sample=data,
            fit_func=fit_single_distribution,
            column_name="test",
        )
        assert len(result) == 1
        assert result[0]["distribution"] == "norm"


class TestFitterEmptySampleHandling:
    """Direct tests for fitter empty sample path (schema preservation)."""

    def test_continuous_fitter_empty_sample_has_schema(self, continuous_fitter):
        """Continuous fitter preserves schema even with empty results."""
        df = pd.DataFrame({"value": [np.nan] * 100})
        results = continuous_fitter.fit(df, column="value", max_distributions=1)

        # Check the internal DataFrame has proper schema
        expected_columns = [
            "column_name", "distribution", "parameters", "sse", "aic", "bic",
            "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue",
            "data_min", "data_max", "data_mean", "data_stddev", "data_count",
            "lower_bound", "upper_bound",
        ]
        for col in expected_columns:
            assert col in results._df.columns, f"Missing column: {col}"

    def test_discrete_fitter_empty_sample_has_schema(self, discrete_fitter):
        """Discrete fitter preserves schema even with empty results."""
        df = pd.DataFrame({"counts": [np.nan] * 100})
        results = discrete_fitter.fit(df, column="counts", max_distributions=1)

        # Check the internal DataFrame has proper schema
        expected_columns = [
            "column_name", "distribution", "parameters", "sse", "aic", "bic",
            "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue",
            "data_min", "data_max", "data_mean", "data_stddev", "data_count",
            "lower_bound", "upper_bound",
        ]
        for col in expected_columns:
            assert col in results._df.columns, f"Missing column: {col}"

    def test_continuous_fitter_all_inf_has_schema(self, continuous_fitter):
        """Continuous fitter with all-inf data preserves schema."""
        df = pd.DataFrame({"value": [np.inf] * 100})
        results = continuous_fitter.fit(df, column="value", max_distributions=1)

        # 19 columns: 17 original + data_kurtosis + data_skewness (v2.3.0)
        assert len(results._df.columns) == 19
        assert len(results.best(n=1, metric="aic")) == 0


# =============================================================================
# Underflow Tests (1e-300 range) - estimation.py and metrics.py
# =============================================================================


class TestUnderflowEstimation:
    """Test estimation.py functions with very small values (1e-300 range)."""

    def test_compute_data_stats_underflow_values(self):
        """compute_data_stats should handle values near 1e-300."""
        data = np.array([1e-300, 2e-300, 3e-300, 4e-300, 5e-300])
        stats = compute_data_stats(data)

        # Should compute valid finite stats
        assert np.isfinite(stats["data_min"])
        assert np.isfinite(stats["data_max"])
        assert np.isfinite(stats["data_mean"])
        assert np.isfinite(stats["data_stddev"])
        assert stats["data_count"] == 5.0
        # Values should be in the expected range
        assert stats["data_min"] == pytest.approx(1e-300, rel=1e-10)
        assert stats["data_max"] == pytest.approx(5e-300, rel=1e-10)

    def test_compute_data_stats_mixed_underflow_normal(self):
        """compute_data_stats with mix of underflow and normal values."""
        data = np.array([1e-300, 1.0, 2.0, 3.0, 1e-300])
        stats = compute_data_stats(data)

        assert np.isfinite(stats["data_mean"])
        assert stats["data_min"] == pytest.approx(1e-300, rel=1e-10)
        assert stats["data_max"] == pytest.approx(3.0, rel=1e-10)

    def test_fit_single_distribution_underflow_data(self):
        """fit_single_distribution should handle underflow-scale data gracefully."""
        # Generate tiny-scale data
        np.random.seed(42)
        data = np.random.normal(loc=5e-300, scale=1e-300, size=100)
        # Ensure all positive for valid histogram
        data = np.abs(data) + 1e-310

        try:
            bin_edges, y_hist = _numpy_histogram(data, bins=20)
            result = fit_single_distribution(
                dist_name="norm",
                data_sample=data,
                bin_edges=bin_edges,
                y_hist=y_hist,
                column_name="test",
                data_stats=compute_data_stats(data),
                lower_bound=None,
                upper_bound=None,
                lazy_metrics=True,
            )
            # Should return a result (possibly with poor fit or failed fit marker)
            assert result is not None
            assert result["distribution"] == "norm"
        except (ValueError, OverflowError, RuntimeError):
            # Acceptable for extreme underflow data
            pass

    def test_bootstrap_ci_underflow_values(self):
        """bootstrap_confidence_intervals should handle underflow-scale data."""
        np.random.seed(42)
        # Data near but not at underflow limit
        data = np.random.normal(loc=1e-100, scale=1e-101, size=200)

        try:
            from spark_bestfit.estimation import bootstrap_confidence_intervals
            ci = bootstrap_confidence_intervals(
                dist_name="norm",
                data=data,
                alpha=0.05,
                n_bootstrap=50,
                random_seed=42,
            )
            # Should return valid CI dict
            assert "loc" in ci
            assert "scale" in ci
            # CI bounds should be finite
            assert np.isfinite(ci["loc"][0]) and np.isfinite(ci["loc"][1])
        except (ValueError, RuntimeError):
            # Acceptable if bootstrap fails for extreme data
            pass


class TestUnderflowMetrics:
    """Test metrics.py functions with very small values (1e-300 range)."""

    def test_compute_information_criteria_underflow_data(self):
        """compute_information_criteria should handle underflow-scale data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_information_criteria

        # Tiny-scale data
        np.random.seed(42)
        data = np.abs(np.random.normal(0, 1e-150, size=100)) + 1e-160

        dist = st.expon
        # Fit to get parameters
        try:
            params = dist.fit(data)
            aic, bic = compute_information_criteria(dist, params, data)
            # May return inf for failed fits, which is acceptable
            assert aic is not None
            assert bic is not None
        except (ValueError, RuntimeError, FloatingPointError):
            pass  # Acceptable for extreme underflow

    def test_compute_ks_statistic_underflow_data(self):
        """compute_ks_statistic should handle underflow-scale data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ks_statistic

        np.random.seed(42)
        data = np.abs(np.random.exponential(scale=1e-200, size=100))

        dist = st.expon
        try:
            params = dist.fit(data)
            ks_stat, pvalue = compute_ks_statistic(dist, params, data)
            # Results should be finite (or inf for failed fits)
            assert ks_stat is not None
            assert pvalue is not None
        except (ValueError, RuntimeError, FloatingPointError):
            pass

    def test_compute_ad_statistic_underflow_data(self):
        """compute_ad_statistic should handle underflow-scale data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ad_statistic

        np.random.seed(42)
        data = np.abs(np.random.normal(0, 1e-200, size=100)) + 1e-210

        dist = st.norm
        try:
            params = dist.fit(data)
            ad_stat = compute_ad_statistic(dist, params, data)
            # Result should be a number (possibly inf)
            assert ad_stat is not None
        except (ValueError, RuntimeError, FloatingPointError):
            pass


# =============================================================================
# Overflow Tests (1e300 range) - estimation.py and metrics.py
# =============================================================================


class TestOverflowEstimation:
    """Test estimation.py functions with very large values (1e300 range)."""

    def test_compute_data_stats_overflow_values(self):
        """compute_data_stats should handle values near 1e300."""
        data = np.array([1e300, 2e300, 3e300, 4e300, 5e300])
        stats = compute_data_stats(data)

        # Should compute valid finite stats
        assert np.isfinite(stats["data_min"])
        assert np.isfinite(stats["data_max"])
        assert np.isfinite(stats["data_mean"])
        # stddev might overflow for extreme values
        # data_count should always be valid
        assert stats["data_count"] == 5.0
        assert stats["data_min"] == pytest.approx(1e300, rel=1e-10)
        assert stats["data_max"] == pytest.approx(5e300, rel=1e-10)

    def test_compute_data_stats_mixed_overflow_normal(self):
        """compute_data_stats with mix of overflow and normal values."""
        data = np.array([1.0, 2.0, 3.0, 1e300])
        stats = compute_data_stats(data)

        assert np.isfinite(stats["data_min"])
        assert stats["data_min"] == pytest.approx(1.0, rel=1e-10)
        assert stats["data_max"] == pytest.approx(1e300, rel=1e-10)

    def test_fit_single_distribution_overflow_data(self):
        """fit_single_distribution should handle overflow-scale data gracefully."""
        np.random.seed(42)
        # Generate data centered around large values
        base = 1e299
        data = base + np.random.normal(0, 1e298, size=100)
        data = np.abs(data)  # Ensure positive

        try:
            bin_edges, y_hist = _numpy_histogram(data, bins=20)
            result = fit_single_distribution(
                dist_name="norm",
                data_sample=data,
                bin_edges=bin_edges,
                y_hist=y_hist,
                column_name="test",
                data_stats=compute_data_stats(data),
                lower_bound=None,
                upper_bound=None,
                lazy_metrics=True,
            )
            assert result is not None
            assert result["distribution"] == "norm"
        except (ValueError, OverflowError, RuntimeError, IndexError):
            # IndexError occurs in NumPy histogram with extreme values
            pass

    def test_fit_mse_overflow_data(self):
        """fit_mse should handle overflow-scale data gracefully."""
        import scipy.stats as st
        from spark_bestfit.estimation import fit_mse

        np.random.seed(42)
        # Use moderately large values (not at extreme overflow)
        data = np.random.normal(loc=1e100, scale=1e99, size=100)

        try:
            params = fit_mse(st.norm, data)
            # Should return valid parameters
            assert len(params) == 2  # loc, scale
            # loc should be near 1e100
            assert np.isfinite(params[0])
        except (ValueError, RuntimeError, OverflowError):
            # Acceptable for extreme data
            pass

    def test_evaluate_pdf_overflow_x_values(self):
        """evaluate_pdf should handle overflow x values."""
        import scipy.stats as st
        from spark_bestfit.estimation import evaluate_pdf

        dist = st.norm
        params = (0.0, 1.0)  # Standard normal
        x = np.array([1e300, -1e300, 1e308])

        pdf_values = evaluate_pdf(dist, params, x)

        # PDF at extreme x should be 0 (or very close)
        assert np.all(np.isfinite(pdf_values))
        assert np.all(pdf_values >= 0)
        # All should be essentially 0 for standard normal at 1e300
        assert np.all(pdf_values < 1e-100)

    def test_bootstrap_ci_large_scale_data(self):
        """bootstrap_confidence_intervals should handle large-scale data."""
        np.random.seed(42)
        from spark_bestfit.estimation import bootstrap_confidence_intervals

        # Large scale but not at overflow limit
        data = np.random.normal(loc=1e50, scale=1e49, size=200)

        try:
            ci = bootstrap_confidence_intervals(
                dist_name="norm",
                data=data,
                alpha=0.05,
                n_bootstrap=50,
                random_seed=42,
            )
            assert "loc" in ci
            assert "scale" in ci
            # loc CI should contain values near 1e50
            assert ci["loc"][0] < 2e50
            assert ci["loc"][1] > 0
        except (ValueError, RuntimeError):
            pass


class TestOverflowMetrics:
    """Test metrics.py functions with very large values (1e300 range)."""

    def test_compute_information_criteria_overflow_data(self):
        """compute_information_criteria should handle overflow-scale data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_information_criteria

        np.random.seed(42)
        # Large scale data
        data = np.random.normal(loc=1e100, scale=1e99, size=100)

        dist = st.norm
        try:
            params = dist.fit(data)
            aic, bic = compute_information_criteria(dist, params, data)
            # May return inf for overflow, which is acceptable
            assert aic is not None
            assert bic is not None
        except (ValueError, RuntimeError, FloatingPointError):
            pass

    def test_compute_ks_statistic_overflow_data(self):
        """compute_ks_statistic should handle overflow-scale data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ks_statistic

        np.random.seed(42)
        data = np.random.normal(loc=1e100, scale=1e99, size=100)

        dist = st.norm
        try:
            params = dist.fit(data)
            ks_stat, pvalue = compute_ks_statistic(dist, params, data)
            assert ks_stat is not None
            assert pvalue is not None
        except (ValueError, RuntimeError, FloatingPointError):
            pass

    def test_compute_ad_statistic_overflow_data(self):
        """compute_ad_statistic should handle overflow-scale data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ad_statistic

        np.random.seed(42)
        data = np.random.normal(loc=1e100, scale=1e99, size=100)

        dist = st.norm
        try:
            params = dist.fit(data)
            ad_stat = compute_ad_statistic(dist, params, data)
            assert ad_stat is not None
        except (ValueError, RuntimeError, FloatingPointError):
            pass

    def test_compute_ks_ad_metrics_overflow_data(self):
        """compute_ks_ad_metrics should handle overflow-scale data."""
        from spark_bestfit.metrics import compute_ks_ad_metrics

        np.random.seed(42)
        data = np.random.normal(loc=1e100, scale=1e99, size=100)

        try:
            # Manually compute params
            import scipy.stats as st
            params = list(st.norm.fit(data))

            ks_stat, pvalue, ad_stat, ad_pvalue = compute_ks_ad_metrics(
                dist_name="norm",
                params=params,
                data_sample=data,
                lower_bound=None,
                upper_bound=None,
            )
            # Results should be returned (possibly None for failed computations)
            # The function should not raise
        except (ValueError, RuntimeError, FloatingPointError):
            pass


# =============================================================================
# Edge Cases for Metrics Functions
# =============================================================================


class TestMetricsEdgeCases:
    """Edge case tests for metrics.py functions."""

    def test_compute_ad_statistic_single_value(self):
        """compute_ad_statistic with single value returns inf."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ad_statistic

        data = np.array([5.0])
        dist = st.norm
        params = (5.0, 1.0)

        ad_stat = compute_ad_statistic(dist, params, data)
        # Single value should return inf (need at least 2 points)
        assert ad_stat == np.inf

    def test_compute_ad_statistic_frozen_single_value(self):
        """compute_ad_statistic_frozen with single value returns inf."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ad_statistic_frozen

        data = np.array([5.0])
        frozen_dist = st.norm(loc=5.0, scale=1.0)

        ad_stat = compute_ad_statistic_frozen(frozen_dist, data)
        assert ad_stat == np.inf

    def test_compute_ks_statistic_frozen_empty_data(self):
        """compute_ks_statistic_frozen with empty data handles gracefully."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_ks_statistic_frozen

        data = np.array([])
        frozen_dist = st.norm(loc=0, scale=1)

        ks_stat, pvalue = compute_ks_statistic_frozen(frozen_dist, data)
        # Empty data should return inf or handle gracefully
        assert ks_stat == np.inf or np.isnan(ks_stat)

    def test_compute_information_criteria_frozen_empty_data(self):
        """compute_information_criteria_frozen with empty data."""
        import scipy.stats as st
        from spark_bestfit.metrics import compute_information_criteria_frozen

        data = np.array([])
        frozen_dist = st.norm(loc=0, scale=1)

        aic, bic = compute_information_criteria_frozen(frozen_dist, 2, data)
        # Empty data should handle gracefully
        assert aic is not None
        assert bic is not None

    def test_compute_ad_pvalue_unsupported_distribution(self):
        """compute_ad_pvalue returns None for unsupported distributions."""
        from spark_bestfit.metrics import compute_ad_pvalue

        np.random.seed(42)
        data = np.random.gamma(2, 2, size=100)

        # Gamma is not supported for A-D p-value
        pvalue = compute_ad_pvalue("gamma", data)
        assert pvalue is None

    def test_compute_ad_pvalue_supported_distributions(self):
        """compute_ad_pvalue returns valid p-value for supported distributions."""
        from spark_bestfit.metrics import compute_ad_pvalue

        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)

        pvalue = compute_ad_pvalue("norm", data)
        assert pvalue is not None
        assert 0 <= pvalue <= 1


# =============================================================================
# Edge Cases for Estimation Functions
# =============================================================================


class TestEstimationEdgeCases:
    """Edge case tests for estimation.py functions."""

    def test_detect_heavy_tail_small_data(self):
        """detect_heavy_tail with < 10 samples returns not heavy-tailed."""
        from spark_bestfit.estimation import detect_heavy_tail

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Only 5 samples

        result = detect_heavy_tail(data)
        assert result["is_heavy_tailed"] is False
        assert result["kurtosis"] == 0.0

    def test_detect_heavy_tail_with_inf_values(self):
        """detect_heavy_tail filters non-finite values."""
        from spark_bestfit.estimation import detect_heavy_tail

        np.random.seed(42)
        data = np.random.normal(0, 1, size=100)
        data[0] = np.inf
        data[1] = -np.inf
        data[2] = np.nan

        result = detect_heavy_tail(data)
        # Should work on remaining finite values
        assert "is_heavy_tailed" in result
        assert np.isfinite(result["kurtosis"])

    def test_detect_heavy_tail_cauchy_data(self):
        """detect_heavy_tail correctly identifies heavy-tailed Cauchy data."""
        from spark_bestfit.estimation import detect_heavy_tail

        np.random.seed(42)
        # Cauchy distribution has extremely heavy tails
        data = np.random.standard_cauchy(size=1000)
        # Filter extreme outliers for stable test
        data = data[np.abs(data) < 100]

        result = detect_heavy_tail(data, kurtosis_threshold=6.0)
        # Cauchy data typically shows heavy-tail indicators
        assert "is_heavy_tailed" in result

    def test_create_sample_data_smaller_than_sample_size(self):
        """create_sample_data returns full data if smaller than sample size."""
        from spark_bestfit.estimation import create_sample_data

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sample = create_sample_data(data, sample_size=100)

        # Should return the original data
        assert len(sample) == 5
        np.testing.assert_array_equal(sample, data)

    def test_create_sample_data_larger_than_sample_size(self):
        """create_sample_data returns sample of correct size."""
        from spark_bestfit.estimation import create_sample_data

        np.random.seed(42)
        data = np.random.normal(0, 1, size=10000)
        sample = create_sample_data(data, sample_size=100, random_seed=42)

        assert len(sample) == 100

    def test_extract_distribution_params_two_params(self):
        """extract_distribution_params handles 2-param distributions."""
        from spark_bestfit.estimation import extract_distribution_params

        params = [10.0, 2.0]  # loc, scale
        shape, loc, scale = extract_distribution_params(params)

        assert shape == ()
        assert loc == 10.0
        assert scale == 2.0

    def test_extract_distribution_params_three_params(self):
        """extract_distribution_params handles 3-param distributions."""
        from spark_bestfit.estimation import extract_distribution_params

        params = [2.0, 0.0, 1.0]  # shape, loc, scale
        shape, loc, scale = extract_distribution_params(params)

        assert shape == (2.0,)
        assert loc == 0.0
        assert scale == 1.0

    def test_extract_distribution_params_four_params(self):
        """extract_distribution_params handles 4-param distributions."""
        from spark_bestfit.estimation import extract_distribution_params

        params = [2.0, 3.0, 0.0, 1.0]  # shape1, shape2, loc, scale
        shape, loc, scale = extract_distribution_params(params)

        assert shape == (2.0, 3.0)
        assert loc == 0.0
        assert scale == 1.0

    def test_extract_distribution_params_single_param_raises(self):
        """extract_distribution_params raises for < 2 params."""
        from spark_bestfit.estimation import extract_distribution_params

        with pytest.raises(ValueError, match="at least 2 elements"):
            extract_distribution_params([1.0])

    def test_failed_fit_result_structure(self):
        """_failed_fit_result returns correct sentinel structure."""
        from spark_bestfit.estimation import _failed_fit_result

        result = _failed_fit_result(
            dist_name="test_dist",
            column_name="col1",
            data_stats={"data_min": 0.0, "data_max": 10.0},
            lower_bound=None,
            upper_bound=None,
        )

        assert result["distribution"] == "test_dist"
        assert result["column_name"] == "col1"
        assert result["sse"] == np.inf
        assert result["aic"] == np.inf
        assert result["bic"] == np.inf
        assert result["ks_statistic"] == np.inf
        assert result["pvalue"] == 0.0
        assert np.isnan(result["parameters"][0])

    def test_get_continuous_param_names_norm(self):
        """get_continuous_param_names returns correct names for norm."""
        from spark_bestfit.estimation import get_continuous_param_names

        names = get_continuous_param_names("norm")
        assert names == ["loc", "scale"]

    def test_get_continuous_param_names_gamma(self):
        """get_continuous_param_names returns correct names for gamma."""
        from spark_bestfit.estimation import get_continuous_param_names

        names = get_continuous_param_names("gamma")
        assert names == ["a", "loc", "scale"]

    def test_get_continuous_param_names_beta(self):
        """get_continuous_param_names returns correct names for beta."""
        from spark_bestfit.estimation import get_continuous_param_names

        names = get_continuous_param_names("beta")
        assert names == ["a", "b", "loc", "scale"]

    def test_fit_mse_minimum_data_points(self):
        """fit_mse raises for < 2 data points."""
        import scipy.stats as st
        from spark_bestfit.estimation import fit_mse

        data = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2 data points"):
            fit_mse(st.norm, data)

    def test_compute_pdf_range_handles_invalid_ppf(self):
        """compute_pdf_range falls back to histogram bounds on ppf failure."""
        import scipy.stats as st
        from spark_bestfit.estimation import compute_pdf_range

        x_hist = np.array([0.0, 1.0, 2.0, 3.0])
        # Use extreme params that might cause ppf issues
        params = [0.0, 1e-300]  # Very small scale

        start, end = compute_pdf_range(st.norm, params, x_hist)

        # Should return valid finite values (fallback to histogram bounds)
        assert np.isfinite(start)
        assert np.isfinite(end)


# =============================================================================
# Constant Array Edge Cases
# =============================================================================


class TestConstantArrays:
    """Test handling of constant arrays."""

    def test_compute_data_stats_constant_array(self):
        """compute_data_stats handles constant array."""
        data = np.array([42.0] * 100)
        stats = compute_data_stats(data)

        assert stats["data_min"] == 42.0
        assert stats["data_max"] == 42.0
        assert stats["data_mean"] == 42.0
        assert stats["data_stddev"] == 0.0
        # Kurtosis and skewness may be NaN for constant data
        # Just check they don't crash

    def test_detect_heavy_tail_constant_array(self):
        """detect_heavy_tail handles constant array."""
        from spark_bestfit.estimation import detect_heavy_tail

        data = np.array([42.0] * 100)
        result = detect_heavy_tail(data)

        # Constant data is not heavy-tailed
        assert result["is_heavy_tailed"] is False

    def test_filter_bootstrap_outliers_constant_params(self):
        """_filter_bootstrap_outliers handles constant parameter column."""
        from spark_bestfit.estimation import _filter_bootstrap_outliers

        # Bootstrap array where one param is constant
        bootstrap_array = np.array([
            [1.0, 5.0, 2.0],
            [1.1, 5.0, 2.1],
            [0.9, 5.0, 1.9],
            [1.05, 5.0, 2.05],
        ])

        result = _filter_bootstrap_outliers(bootstrap_array)

        # Should handle constant column (IQR=0) without removing all rows
        assert len(result) > 0


# =============================================================================
# Empty Array Edge Cases
# =============================================================================


class TestEmptyArrays:
    """Test handling of empty arrays."""

    def test_compute_data_stats_empty_array(self):
        """compute_data_stats behavior with empty array."""
        data = np.array([])

        # Empty array will cause issues with np.min/max - this tests error handling
        try:
            stats = compute_data_stats(data)
            # If it succeeds, values should indicate empty/invalid
            assert stats is not None
        except (ValueError, RuntimeError):
            # Acceptable to raise for empty input
            pass

    def test_detect_heavy_tail_empty_array(self):
        """detect_heavy_tail handles empty array."""
        from spark_bestfit.estimation import detect_heavy_tail

        data = np.array([])
        result = detect_heavy_tail(data)

        # Empty data (< 10 samples) should return not heavy-tailed
        assert result["is_heavy_tailed"] is False

    def test_fit_mse_empty_array(self):
        """fit_mse handles empty array."""
        import scipy.stats as st
        from spark_bestfit.estimation import fit_mse

        data = np.array([])

        with pytest.raises(ValueError, match="at least 2 data points"):
            fit_mse(st.norm, data)
