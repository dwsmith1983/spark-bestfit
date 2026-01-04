"""Shared test utilities for backend testing.

This module provides common test functions that can be used across different
backend test files (test_backends.py, test_ray_backend.py) to reduce code
duplication while maintaining backend-specific test isolation.

Usage:
    from tests.shared_backend_tests import run_progress_callback_tests

    class TestProgressCallback:
        def test_backend_progress_callback(self, backend, normal_data, histogram):
            run_progress_callback_tests(backend, normal_data, histogram)
"""

import numpy as np

from spark_bestfit.fitting import compute_data_stats, fit_single_distribution


def run_progress_callback_basic_test(backend, normal_data, histogram, distributions=None):
    """Test that backend invokes progress callback with correct values.

    Args:
        backend: ExecutionBackend instance (LocalBackend, RayBackend, SparkBackend)
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)
        distributions: List of distribution names (default: ["norm", "expon", "gamma"])

    Returns:
        List of progress calls as (completed, total, percent) tuples
    """
    if distributions is None:
        distributions = ["norm", "expon", "gamma"]

    progress_calls = []

    def on_progress(completed, total, percent):
        progress_calls.append((completed, total, percent))

    results = backend.parallel_fit(
        distributions=distributions,
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=on_progress,
    )

    # Verify we got results
    assert len(results) > 0, "Should have at least one successful fit"

    # Should have exactly len(distributions) callback calls
    assert len(progress_calls) == len(distributions), (
        f"Expected {len(distributions)} callbacks, got {len(progress_calls)}"
    )

    # Verify progress values are correct
    for completed, total, percent in progress_calls:
        assert total == len(distributions), f"Total should be {len(distributions)}"
        assert 1 <= completed <= len(distributions), "Completed should be in valid range"
        assert 0 < percent <= 100, "Percent should be in (0, 100]"

    # Last callback should show 100%
    last_completed = max(call[0] for call in progress_calls)
    assert last_completed == len(distributions), "Final completed count should equal total"

    return progress_calls


def run_progress_callback_error_test(backend, normal_data, histogram):
    """Test that backend handles callback errors gracefully.

    Args:
        backend: ExecutionBackend instance
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)
    """
    def failing_callback(completed, total, percent):
        raise ValueError("Intentional callback error")

    distributions = ["norm", "expon"]
    results = backend.parallel_fit(
        distributions=distributions,
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=failing_callback,
    )

    # Fitting should still complete successfully despite callback errors
    assert len(results) > 0, "Should complete fitting despite callback errors"


def run_progress_callback_none_test(backend, normal_data, histogram):
    """Test that backend works fine without progress callback.

    Args:
        backend: ExecutionBackend instance
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)
    """
    distributions = ["norm", "expon"]
    results = backend.parallel_fit(
        distributions=distributions,
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=None,
    )

    assert len(results) > 0, "Should have successful fits with no callback"


def run_progress_callback_empty_distributions_test(backend, normal_data, histogram):
    """Test that backend handles empty distribution list correctly.

    Args:
        backend: ExecutionBackend instance
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)
    """
    progress_calls = []

    def on_progress(completed, total, percent):
        progress_calls.append((completed, total, percent))

    results = backend.parallel_fit(
        distributions=[],
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=on_progress,
    )

    assert len(results) == 0, "Empty distributions should yield no results"
    assert len(progress_calls) == 0, "No callbacks for empty distributions"


def run_progress_callback_single_distribution_test(backend, normal_data, histogram):
    """Test progress callback with single distribution.

    Args:
        backend: ExecutionBackend instance
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)
    """
    progress_calls = []

    def on_progress(completed, total, percent):
        progress_calls.append((completed, total, percent))

    results = backend.parallel_fit(
        distributions=["norm"],
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=on_progress,
    )

    assert len(results) > 0, "Should fit single distribution"
    assert len(progress_calls) == 1, "Single distribution = single callback"
    assert progress_calls[0] == (1, 1, 100.0), "Should report 1/1 = 100%"


def run_progress_strictly_increasing_test(backend, normal_data, histogram):
    """Test that completed count is strictly increasing.

    Args:
        backend: ExecutionBackend instance
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)

    Note:
        This test verifies thread safety of the completed counter.
    """
    progress_calls = []

    def on_progress(completed, total, percent):
        progress_calls.append((completed, total, percent))

    distributions = ["norm", "expon", "gamma", "uniform", "lognorm"]
    backend.parallel_fit(
        distributions=distributions,
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=on_progress,
    )

    # Extract completed values and sort by call order
    completed_values = [call[0] for call in progress_calls]

    # Verify strictly increasing (each new completed > previous)
    for i in range(1, len(completed_values)):
        assert completed_values[i] == completed_values[i - 1] + 1, (
            f"Completed count should increment by 1: "
            f"got {completed_values[i - 1]} -> {completed_values[i]}"
        )


def run_progress_bounds_invariants_test(backend, normal_data, histogram):
    """Test progress callback bounds and invariants.

    Args:
        backend: ExecutionBackend instance
        normal_data: Numpy array of normal distribution data
        histogram: Tuple of (y_hist, bin_edges)
    """
    progress_calls = []

    def on_progress(completed, total, percent):
        progress_calls.append((completed, total, percent))

    distributions = ["norm", "expon", "gamma", "uniform"]
    backend.parallel_fit(
        distributions=distributions,
        histogram=histogram,
        data_sample=normal_data,
        fit_func=fit_single_distribution,
        column_name="value",
        data_stats=compute_data_stats(normal_data),
        progress_callback=on_progress,
    )

    for completed, total, percent in progress_calls:
        # Invariant: completed <= total
        assert completed <= total, f"completed ({completed}) should be <= total ({total})"

        # Invariant: percent = completed / total * 100
        expected_percent = (completed / total) * 100.0
        assert abs(percent - expected_percent) < 0.01, (
            f"percent ({percent}) should equal completed/total*100 ({expected_percent})"
        )

        # Invariant: 0 < percent <= 100
        assert 0 < percent <= 100, f"percent ({percent}) should be in (0, 100]"
