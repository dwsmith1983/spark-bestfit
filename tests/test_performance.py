"""Performance regression tests.

These tests ensure that key operations complete within expected time bounds.
They are marked with pytest.mark.slow and can be skipped in quick test runs.
"""

import time

import numpy as np
import pytest

from spark_bestfit import DistributionFitter
from spark_bestfit.fitting import (
    FITTING_SAMPLE_SIZE,
    create_sample_data,
    fit_single_distribution,
)
from spark_bestfit.histogram import HistogramComputer


class TestFittingPerformance:
    """Performance tests for distribution fitting operations."""

    @pytest.mark.slow
    def test_fit_single_distribution_time(self):
        """Test that fitting a single distribution completes in reasonable time."""
        # Generate test data
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=10_000)
        hist, bin_edges = np.histogram(data, bins=50, density=True)


        # Time the fit
        start = time.perf_counter()
        result = fit_single_distribution("norm", data, bin_edges, hist)
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second for a simple distribution
        assert elapsed < 1.0, f"fit_single_distribution took {elapsed:.2f}s, expected < 1s"
        assert result["distribution"] == "norm"
        assert result["sse"] < float("inf")

    @pytest.mark.slow
    def test_fit_multiple_distributions_time(self):
        """Test that fitting multiple distributions scales reasonably."""
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=10_000)
        hist, bin_edges = np.histogram(data, bins=50, density=True)


        distributions = ["norm", "expon", "gamma", "lognorm", "weibull_min"]

        start = time.perf_counter()
        results = []
        for dist_name in distributions:
            result = fit_single_distribution(dist_name, data, bin_edges, hist)
            results.append(result)
        elapsed = time.perf_counter() - start

        # 5 distributions should complete in under 5 seconds
        assert elapsed < 5.0, f"Fitting 5 distributions took {elapsed:.2f}s, expected < 5s"
        assert len(results) == 5
        # At least some should succeed
        successful = [r for r in results if r["sse"] < float("inf")]
        assert len(successful) >= 3

    @pytest.mark.slow
    def test_create_sample_data_performance(self):
        """Test that data sampling is fast."""
        np.random.seed(42)
        large_data = np.random.normal(size=1_000_000)

        start = time.perf_counter()
        sample = create_sample_data(large_data, sample_size=FITTING_SAMPLE_SIZE)
        elapsed = time.perf_counter() - start

        # Sampling 10k from 1M should be nearly instant
        assert elapsed < 0.5, f"create_sample_data took {elapsed:.2f}s, expected < 0.5s"
        assert len(sample) == FITTING_SAMPLE_SIZE


class TestHistogramPerformance:
    """Performance tests for histogram computation."""

    @pytest.mark.slow
    def test_histogram_computation_time(self, spark_session):
        """Test that histogram computation is fast."""
        # Create test DataFrame
        np.random.seed(42)
        data = [(float(x),) for x in np.random.normal(size=100_000)]
        df = spark_session.createDataFrame(data, ["value"])

        computer = HistogramComputer()

        start = time.perf_counter()
        y_hist, bin_edges = computer.compute_histogram(df, "value", bins=100)
        elapsed = time.perf_counter() - start

        # Histogram of 100k rows should complete in under 10 seconds
        assert elapsed < 10.0, f"compute_histogram took {elapsed:.2f}s, expected < 10s"
        assert len(bin_edges) == 101  # n_bins + 1
        assert len(y_hist) == 100


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    @pytest.mark.slow
    def test_full_fit_small_dataset(self, spark_session):
        """Test full fitting pipeline on small dataset."""
        np.random.seed(42)
        data = [(float(x),) for x in np.random.normal(loc=100, scale=15, size=10_000)]
        df = spark_session.createDataFrame(data, ["value"])

        fitter = DistributionFitter(spark_session)

        start = time.perf_counter()
        # Fit only 5 distributions for speed
        results = fitter.fit(df, "value", bins=50, use_rice_rule=False, max_distributions=5)
        elapsed = time.perf_counter() - start

        # Full pipeline with 5 distributions should complete in under 30 seconds
        assert elapsed < 30.0, f"Full fit took {elapsed:.2f}s, expected < 30s"
        assert results.count() == 5  # Should fit exactly 5 distributions as requested

    @pytest.mark.slow
    def test_full_fit_medium_dataset(self, spark_session):
        """Test full fitting pipeline on medium dataset."""
        np.random.seed(42)
        data = [(float(x),) for x in np.random.gamma(shape=2, scale=5, size=50_000)]
        df = spark_session.createDataFrame(data, ["value"])

        fitter = DistributionFitter(spark_session)

        start = time.perf_counter()
        # Fit only 3 distributions for speed
        results = fitter.fit(df, "value", bins=75, use_rice_rule=False, max_distributions=3)
        elapsed = time.perf_counter() - start

        # Medium dataset with 3 distributions should complete in under 45 seconds
        assert elapsed < 45.0, f"Full fit took {elapsed:.2f}s, expected < 45s"
        assert results.count() > 0


class TestMemoryEfficiency:
    """Tests for memory-efficient operations."""

    def test_sample_data_correct_size(self):
        """Test that sampled data has correct size."""
        np.random.seed(42)
        large_data = np.random.normal(size=100_000)

        result = create_sample_data(large_data, sample_size=5000)

        assert len(result) == 5000
        # Should be a subset, not the original
        assert result is not large_data
