"""Tests for histogram computation module.

Uses LocalBackend for most tests. Spark-specific tests are in test_spark_backend.py.
"""

import numpy as np
import pandas as pd
import pytest

from spark_bestfit.histogram import HistogramComputer


class TestHistogramComputer:
    """Tests for HistogramComputer class."""

    def test_compute_histogram_basic(self, local_backend, pandas_dataset):
        """Test basic histogram computation."""
        computer = HistogramComputer(local_backend)
        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=50)

        # Should return arrays of correct size (bin_edges has n+1 elements)
        assert len(y_hist) == 50
        assert len(bin_edges) == 51  # n_bins + 1

        # Histogram should be normalized (density sums to ~1 when multiplied by bin widths)
        bin_widths = np.diff(bin_edges)
        total_area = np.sum(y_hist * bin_widths)
        assert np.isclose(total_area, 1.0, atol=0.01)

        # All values should be non-negative
        assert np.all(y_hist >= 0)

    def test_compute_histogram_custom_bins(self, local_backend, pandas_dataset):
        """Test histogram with custom number of bins."""
        computer = HistogramComputer(local_backend)

        for n_bins in [10, 25, 100]:
            y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=n_bins)

            assert len(y_hist) == n_bins
            assert len(bin_edges) == n_bins + 1  # edges have n+1 elements

    def test_compute_histogram_rice_rule(self, local_backend, pandas_dataset):
        """Test histogram with Rice rule for bin calculation."""
        computer = HistogramComputer(local_backend)
        row_count = len(pandas_dataset)

        y_hist, bin_edges = computer.compute_histogram(
            pandas_dataset, "value", bins=50, use_rice_rule=True, approx_count=row_count
        )

        # Rice rule: bins = 2 * n^(1/3)
        expected_bins = int(np.ceil(row_count ** (1 / 3)) * 2)

        assert len(y_hist) == expected_bins
        assert len(bin_edges) == expected_bins + 1

    def test_compute_histogram_constant_values(self, local_backend):
        """Test histogram with constant values (edge case)."""
        constant_df = pd.DataFrame({"value": np.full(1000, 42.0)})
        computer = HistogramComputer(local_backend)
        y_hist, bin_edges = computer.compute_histogram(constant_df, "value", bins=50)

        # Should handle min == max case (returns single bin)
        assert len(y_hist) == 1
        assert len(bin_edges) == 1  # Special case: single value returns single edge

        # Single bin at the constant value
        assert np.isclose(bin_edges[0], 42.0)
        assert np.isclose(y_hist[0], 1.0)

    def test_compute_histogram_positive_data(self, local_backend, pandas_positive_dataset):
        """Test histogram with only positive values."""
        computer = HistogramComputer(local_backend)
        y_hist, bin_edges = computer.compute_histogram(pandas_positive_dataset, "value", bins=50)

        # All bin edges should be positive (or at least non-negative)
        assert np.all(bin_edges >= 0)

        # Should have correct size
        assert len(y_hist) == 50
        assert len(bin_edges) == 51

    def test_compute_histogram_bin_edges_array(self, local_backend, pandas_dataset):
        """Test histogram with custom bin edges as array."""
        computer = HistogramComputer(local_backend)
        custom_bins = np.array([0, 20, 40, 60, 80, 100])

        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=custom_bins)

        # Should have len(bins) - 1 bins
        assert len(y_hist) == len(custom_bins) - 1
        assert len(bin_edges) == len(custom_bins)  # Returns the custom edges

    def test_compute_histogram_via_local_backend(self, local_backend, pandas_dataset):
        """Test that histogram is computed via LocalBackend abstraction."""
        computer = HistogramComputer(local_backend)

        # Compute histogram via backend
        bin_edges = np.linspace(0, 100, 51)
        bin_counts, total = local_backend.compute_histogram(pandas_dataset, "value", bin_edges)

        # Result should be numpy arrays with correct shape
        assert len(bin_counts) == 50  # 51 edges = 50 bins
        assert total > 0  # Should have counted some data
        assert np.sum(bin_counts) == total

    def test_compute_statistics(self, local_backend, pandas_dataset):
        """Test computing basic statistics via backend."""
        computer = HistogramComputer(local_backend)
        stats = computer.compute_statistics(pandas_dataset, "value")

        # Should have basic statistics from backend
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats

        # Values should be reasonable for normal(50, 10) data
        assert stats["min"] is not None
        assert stats["max"] is not None
        assert stats["count"] == len(pandas_dataset)

    def test_compute_statistics_types(self, local_backend, pandas_dataset):
        """Test that statistics are returned as numeric types."""
        computer = HistogramComputer(local_backend)
        stats = computer.compute_statistics(pandas_dataset, "value")

        # All should be numeric (float or int) or None
        for key, value in stats.items():
            if value is not None:
                assert isinstance(value, (float, int, np.integer, np.floating))

    def test_histogram_no_data_loss(self, local_backend, pandas_dataset):
        """Test that histogram captures all data (no bins with zero when they shouldn't be)."""
        computer = HistogramComputer(local_backend)
        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=50)

        # For normal distribution, most bins should have some data
        non_zero_bins = np.sum(y_hist > 0)
        assert non_zero_bins > 40  # At least 80% of bins should have data

    def test_histogram_with_outliers(self, local_backend):
        """Test histogram computation with outliers."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 9900)
        outliers = np.array([0, 200, -50, 250])  # Extreme outliers
        data = np.concatenate([normal_data, outliers])

        df = pd.DataFrame({"value": data})

        computer = HistogramComputer(local_backend)
        y_hist, bin_edges = computer.compute_histogram(df, "value", bins=50)

        # Should handle outliers gracefully
        assert len(y_hist) == 50
        assert len(bin_edges) == 51

        # Min and max edges should capture outliers
        assert bin_edges.min() < 0
        assert bin_edges.max() > 200

    def test_medium_dataset_performance(self, local_backend):
        """Test histogram computation on medium dataset (100K rows)."""
        np.random.seed(42)
        medium_data = np.random.normal(loc=50, scale=10, size=100_000)
        medium_df = pd.DataFrame({"value": medium_data})

        computer = HistogramComputer(local_backend)

        # Should complete without errors
        y_hist, bin_edges = computer.compute_histogram(medium_df, "value", bins=100)

        assert len(y_hist) == 100
        assert len(bin_edges) == 101

        # Should still be normalized
        bin_widths = np.diff(bin_edges)
        total_area = np.sum(y_hist * bin_widths)
        assert np.isclose(total_area, 1.0, atol=0.01)


class TestHistogramErrorHandling:
    """Error handling tests for HistogramComputer."""

    def test_invalid_column_name(self, local_backend, pandas_dataset):
        """Test that invalid column name raises appropriate error."""
        computer = HistogramComputer(local_backend)

        with pytest.raises(Exception):  # Will raise KeyError for pandas
            computer.compute_histogram(pandas_dataset, "nonexistent_column", bins=50)

    def test_empty_dataframe(self, local_backend):
        """Test histogram computation with empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame({"value": pd.Series([], dtype=float)})
        computer = HistogramComputer(local_backend)

        # Empty DataFrame should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="no valid"):
            computer.compute_histogram(empty_df, "value", bins=50)

    def test_single_row_dataframe(self, local_backend):
        """Test histogram computation with single row."""
        single_row_df = pd.DataFrame({"value": [42.0]})
        computer = HistogramComputer(local_backend)

        y_hist, bin_edges = computer.compute_histogram(single_row_df, "value", bins=50)

        # Should handle single value case (like constant dataset)
        assert len(y_hist) >= 1
        assert len(bin_edges) >= 1

    def test_two_distinct_values(self, local_backend):
        """Test histogram with only two distinct values."""
        df = pd.DataFrame({"value": [1.0, 100.0]})
        computer = HistogramComputer(local_backend)

        y_hist, bin_edges = computer.compute_histogram(df, "value", bins=50)

        # Should create proper histogram with two values
        assert len(y_hist) == 50
        assert len(bin_edges) == 51

    def test_all_null_values(self, local_backend):
        """Test histogram computation with all null values.

        Note: LocalBackend handles this differently than Spark - it returns
        arrays with NaN values rather than raising an error. This test verifies
        the LocalBackend behavior is consistent.
        """
        # Use np.nan to ensure pandas treats them as actual null values
        null_df = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})
        computer = HistogramComputer(local_backend)

        # LocalBackend returns NaN arrays for all-null data
        y_hist, bin_edges = computer.compute_histogram(null_df, "value", bins=50)

        # All values should be 0 or NaN (no valid data to histogram)
        assert np.all((y_hist == 0) | np.isnan(y_hist))

    def test_mixed_null_values(self, local_backend):
        """Test histogram with some null values mixed in filters them out."""
        data = list(range(100)) + [None] * 10
        df = pd.DataFrame({"value": data})
        computer = HistogramComputer(local_backend)

        # Null values should be filtered out, histogram computed on valid values
        y_hist, bin_edges = computer.compute_histogram(df, "value", bins=10)

        assert len(y_hist) == 10
        assert len(bin_edges) == 11
        assert np.all(y_hist >= 0)

    def test_very_large_bin_count(self, local_backend, pandas_dataset):
        """Test histogram with very large number of bins."""
        computer = HistogramComputer(local_backend)

        # Many bins (more than data points would have in many bins)
        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=1000)

        assert len(y_hist) == 1000
        assert len(bin_edges) == 1001

    def test_single_bin(self, local_backend, pandas_dataset):
        """Test histogram with bins=1 is automatically upgraded to bins=2.

        The code automatically upgrades bins=1 to bins=2 for robustness.
        """
        computer = HistogramComputer(local_backend)

        # bins=1 is upgraded to bins=2
        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=1)

        assert len(y_hist) == 2
        assert len(bin_edges) == 3

    def test_rice_rule_small_dataset(self, local_backend):
        """Test Rice rule with very small dataset."""
        small_df = pd.DataFrame({"value": list(range(5))})
        computer = HistogramComputer(local_backend)

        y_hist, bin_edges = computer.compute_histogram(
            small_df, "value", bins=50, use_rice_rule=True, approx_count=5
        )

        # Rice rule with n=5: bins = 2 * 5^(1/3) ≈ 3.4 → 4 bins
        expected_bins = int(np.ceil(5 ** (1 / 3)) * 2)
        assert len(y_hist) == expected_bins

    def test_compute_statistics_invalid_column(self, local_backend, pandas_dataset):
        """Test compute_statistics with invalid column."""
        computer = HistogramComputer(local_backend)

        with pytest.raises(Exception):
            computer.compute_statistics(pandas_dataset, "nonexistent_column")

    def test_compute_statistics_empty_dataframe(self, local_backend):
        """Test compute_statistics with empty DataFrame."""
        empty_df = pd.DataFrame({"value": pd.Series([], dtype=float)})
        computer = HistogramComputer(local_backend)

        stats = computer.compute_statistics(empty_df, "value")

        # Should return stats with None values or zeros
        assert "count" in stats
        assert stats["count"] == 0 or stats["count"] is None

    def test_histogram_returns_numpy_arrays(self, local_backend, pandas_dataset):
        """Test that histogram returns numpy arrays."""
        computer = HistogramComputer(local_backend)
        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=50)

        assert isinstance(y_hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)

    def test_histogram_preserves_data_range(self, local_backend, pandas_dataset):
        """Test that histogram bin edges cover the data range."""
        computer = HistogramComputer(local_backend)
        stats = computer.compute_statistics(pandas_dataset, "value")
        y_hist, bin_edges = computer.compute_histogram(pandas_dataset, "value", bins=50)

        # Bin edges should cover the data range (with small epsilon tolerance)
        bin_width = bin_edges[1] - bin_edges[0]
        assert bin_edges.min() <= stats["min"] + bin_width
        assert bin_edges.max() >= stats["max"] - bin_width
