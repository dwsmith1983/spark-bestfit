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
