"""Tests for histogram computation module."""

import numpy as np
import pytest

from spark_bestfit.histogram import HistogramComputer

class TestHistogramComputer:
    """Tests for HistogramComputer class."""

    def test_initialization(self):
        """Test histogram computer initialization."""
        computer = HistogramComputer()

        # HistogramComputer is a simple class with no required initialization
        assert computer is not None

    def test_compute_histogram_basic(self, spark_session, small_dataset):
        """Test basic histogram computation."""
        computer = HistogramComputer()
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        # Should return arrays of correct size
        assert len(y_hist) == 50
        assert len(x_hist) == 50

        # Histogram should be normalized (density sums to ~1 when multiplied by bin widths)
        bin_width = x_hist[1] - x_hist[0]
        total_area = np.sum(y_hist * bin_width)
        assert np.isclose(total_area, 1.0, atol=0.01)

        # All values should be non-negative
        assert np.all(y_hist >= 0)

    def test_compute_histogram_custom_bins(self, spark_session, small_dataset):
        """Test histogram with custom number of bins."""
        computer = HistogramComputer()

        for n_bins in [10, 25, 100]:
            y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=n_bins)

            assert len(y_hist) == n_bins
            assert len(x_hist) == n_bins

    def test_compute_histogram_rice_rule(self, spark_session, small_dataset):
        """Test histogram with Rice rule for bin calculation."""
        computer = HistogramComputer()
        row_count = small_dataset.count()

        y_hist, x_hist = computer.compute_histogram(
            small_dataset, "value", bins=50, use_rice_rule=True, approx_count=row_count
        )

        # Rice rule: bins = 2 * n^(1/3)
        expected_bins = int(np.ceil(row_count ** (1 / 3)) * 2)

        assert len(y_hist) == expected_bins
        assert len(x_hist) == expected_bins

    def test_compute_histogram_constant_values(self, spark_session, constant_dataset):
        """Test histogram with constant values (edge case)."""
        computer = HistogramComputer()
        y_hist, x_hist = computer.compute_histogram(constant_dataset, "value", bins=50)

        # Should handle min == max case
        assert len(y_hist) == 1
        assert len(x_hist) == 1

        # Single bin centered at the constant value
        assert np.isclose(x_hist[0], 42.0)
        assert np.isclose(y_hist[0], 1.0)

    def test_compute_histogram_positive_data(self, spark_session, positive_dataset):
        """Test histogram with only positive values."""
        computer = HistogramComputer()
        y_hist, x_hist = computer.compute_histogram(positive_dataset, "value", bins=50)

        # All bin centers should be positive
        assert np.all(x_hist >= 0)

        # Should have correct size
        assert len(y_hist) == 50
        assert len(x_hist) == 50

    def test_compute_histogram_bin_edges_array(self, spark_session, small_dataset):
        """Test histogram with custom bin edges as array."""
        computer = HistogramComputer()
        custom_bins = np.array([0, 20, 40, 60, 80, 100])

        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=custom_bins)

        # Should have len(bins) - 1 bins
        assert len(y_hist) == len(custom_bins) - 1
        assert len(x_hist) == len(custom_bins) - 1

    def test_compute_histogram_distributed_no_collect(self, spark_session, small_dataset):
        """Test that histogram stays distributed (doesn't collect raw data)."""
        computer = HistogramComputer()

        # This should NOT collect raw data, only aggregated histogram
        bin_edges = np.linspace(0, 100, 51)
        histogram_df = computer._compute_histogram_distributed(small_dataset, "value", bin_edges)

        # Result should be a DataFrame with (bin_id, count)
        assert "bin_id" in histogram_df.columns
        assert "count" in histogram_df.columns

        # Should have at most len(bin_edges) - 1 rows (some bins may be empty)
        assert histogram_df.count() <= len(bin_edges) - 1

    def test_compute_statistics(self, spark_session, normal_data, small_dataset):
        """Test computing basic statistics."""
        computer = HistogramComputer()
        stats = computer.compute_statistics(small_dataset, "value")

        # Should have all statistics
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "stddev" in stats
        assert "count" in stats

        # Values should be reasonable for normal(50, 10) data
        assert stats["mean"] is not None
        assert 45 < stats["mean"] < 55  # Close to 50

        assert stats["stddev"] is not None
        assert 8 < stats["stddev"] < 12  # Close to 10

        assert stats["count"] == len(normal_data)

    def test_compute_statistics_types(self, spark_session, small_dataset):
        """Test that statistics are returned as floats."""
        computer = HistogramComputer()
        stats = computer.compute_statistics(small_dataset, "value")

        # All should be floats or None
        for key, value in stats.items():
            if value is not None:
                assert isinstance(value, float)

    def test_histogram_no_data_loss(self, spark_session, small_dataset):
        """Test that histogram captures all data (no bins with zero when they shouldn't be)."""
        computer = HistogramComputer()
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        # For normal distribution, most bins should have some data
        non_zero_bins = np.sum(y_hist > 0)
        assert non_zero_bins > 40  # At least 80% of bins should have data

    def test_histogram_with_outliers(self, spark_session):
        """Test histogram computation with outliers."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 9900)
        outliers = np.array([0, 200, -50, 250])  # Extreme outliers
        data = np.concatenate([normal_data, outliers])

        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        computer = HistogramComputer()
        y_hist, x_hist = computer.compute_histogram(df, "value", bins=50)

        # Should handle outliers gracefully
        assert len(y_hist) == 50
        assert len(x_hist) == 50

        # Min and max should capture outliers
        assert x_hist.min() < 0
        assert x_hist.max() > 200

    def test_medium_dataset_performance(self, spark_session, medium_dataset):
        """Test histogram computation on medium dataset (100K rows)."""
        computer = HistogramComputer()

        # Should complete without errors
        y_hist, x_hist = computer.compute_histogram(medium_dataset, "value", bins=100)

        assert len(y_hist) == 100
        assert len(x_hist) == 100

        # Should still be normalized
        bin_width = x_hist[1] - x_hist[0]
        total_area = np.sum(y_hist * bin_width)
        assert np.isclose(total_area, 1.0, atol=0.01)

class TestHistogramErrorHandling:
    """Error handling tests for HistogramComputer."""

    def test_invalid_column_name(self, spark_session, small_dataset):
        """Test that invalid column name raises appropriate error."""
        computer = HistogramComputer()

        with pytest.raises(Exception):  # Spark will raise AnalysisException
            computer.compute_histogram(small_dataset, "nonexistent_column", bins=50)

    def test_empty_dataframe(self, spark_session):
        """Test histogram computation with empty DataFrame raises ValueError."""
        empty_df = spark_session.createDataFrame([], "value: double")
        computer = HistogramComputer()

        # Empty DataFrame should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="no valid"):
            computer.compute_histogram(empty_df, "value", bins=50)

    def test_single_row_dataframe(self, spark_session):
        """Test histogram computation with single row."""
        single_row_df = spark_session.createDataFrame([(42.0,)], ["value"])
        computer = HistogramComputer()

        y_hist, x_hist = computer.compute_histogram(single_row_df, "value", bins=50)

        # Should handle single value case (like constant dataset)
        assert len(y_hist) >= 1
        assert len(x_hist) >= 1

    def test_two_distinct_values(self, spark_session):
        """Test histogram with only two distinct values."""
        df = spark_session.createDataFrame([(1.0,), (100.0,)], ["value"])
        computer = HistogramComputer()

        y_hist, x_hist = computer.compute_histogram(df, "value", bins=50)

        # Should create proper histogram with two values
        assert len(y_hist) == 50
        assert len(x_hist) == 50

    def test_all_null_values(self, spark_session):
        """Test histogram computation with all null values raises ValueError."""
        null_df = spark_session.createDataFrame([(None,), (None,), (None,)], "value: double")
        computer = HistogramComputer()

        # All null values should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="no valid"):
            computer.compute_histogram(null_df, "value", bins=50)

    def test_mixed_null_values(self, spark_session):
        """Test histogram with some null values mixed in filters them out."""
        data = [(float(i),) for i in range(100)] + [(None,) for _ in range(10)]
        df = spark_session.createDataFrame(data, ["value"])
        computer = HistogramComputer()

        # Null values should be filtered out, histogram computed on valid values
        y_hist, x_hist = computer.compute_histogram(df, "value", bins=10)

        assert len(y_hist) == 10
        assert len(x_hist) == 10
        assert np.all(y_hist >= 0)

    def test_very_large_bin_count(self, spark_session, small_dataset):
        """Test histogram with very large number of bins."""
        computer = HistogramComputer()

        # Many bins (more than data points would have in many bins)
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=1000)

        assert len(y_hist) == 1000
        assert len(x_hist) == 1000

    def test_single_bin(self, spark_session, small_dataset):
        """Test histogram with bins=1 is automatically upgraded to bins=2.

        Spark's Bucketizer requires at least 3 splits (2 bins minimum).
        The code automatically upgrades bins=1 to bins=2.
        """
        computer = HistogramComputer()

        # bins=1 is upgraded to bins=2
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=1)

        assert len(y_hist) == 2
        assert len(x_hist) == 2

    def test_rice_rule_small_dataset(self, spark_session):
        """Test Rice rule with very small dataset."""
        small_df = spark_session.createDataFrame([(float(i),) for i in range(5)], ["value"])
        computer = HistogramComputer()

        y_hist, x_hist = computer.compute_histogram(
            small_df, "value", bins=50, use_rice_rule=True, approx_count=5
        )

        # Rice rule with n=5: bins = 2 * 5^(1/3) ≈ 3.4 → 4 bins
        expected_bins = int(np.ceil(5 ** (1 / 3)) * 2)
        assert len(y_hist) == expected_bins

    def test_compute_statistics_invalid_column(self, spark_session, small_dataset):
        """Test compute_statistics with invalid column."""
        computer = HistogramComputer()

        with pytest.raises(Exception):
            computer.compute_statistics(small_dataset, "nonexistent_column")

    def test_compute_statistics_empty_dataframe(self, spark_session):
        """Test compute_statistics with empty DataFrame."""
        empty_df = spark_session.createDataFrame([], "value: double")
        computer = HistogramComputer()

        stats = computer.compute_statistics(empty_df, "value")

        # Should return stats with None values or zeros
        assert "count" in stats
        assert stats["count"] == 0 or stats["count"] is None

    def test_histogram_returns_numpy_arrays(self, spark_session, small_dataset):
        """Test that histogram returns numpy arrays."""
        computer = HistogramComputer()
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        assert isinstance(y_hist, np.ndarray)
        assert isinstance(x_hist, np.ndarray)

    def test_histogram_preserves_data_range(self, spark_session, small_dataset):
        """Test that histogram x values cover the data range."""
        computer = HistogramComputer()
        stats = computer.compute_statistics(small_dataset, "value")
        y_hist, x_hist = computer.compute_histogram(small_dataset, "value", bins=50)

        # Bin centers should be within or close to data range
        assert x_hist.min() >= stats["min"] - (x_hist[1] - x_hist[0])
        assert x_hist.max() <= stats["max"] + (x_hist[1] - x_hist[0])
