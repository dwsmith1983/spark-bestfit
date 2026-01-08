"""Distributed histogram computation without collecting raw data.

This module provides the HistogramComputer class that uses the backend
abstraction for distributed histogram computation.
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend


class HistogramComputer:
    """Computes histograms efficiently using distributed aggregations.

    This implementation avoids collecting raw data to the driver by using
    the backend's distributed aggregation capabilities. Only the final histogram
    (typically ~100 values) is collected, not the raw dataset.

    Supports multiple backends:
    - SparkBackend: Uses Spark ML Bucketizer + groupBy (scales to billions of rows)
    - LocalBackend: Uses numpy histogram (for testing and small datasets)

    Example:
        >>> from spark_bestfit.backends.spark import SparkBackend
        >>> backend = SparkBackend(spark)
        >>> computer = HistogramComputer(backend)
        >>> y_hist, bin_edges = computer.compute_histogram(
        ...     df, column='value', bins=50
        ... )
        >>> # y_hist has 50 values, bin_edges has 51 values
        >>> x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Compute centers if needed

    Auto-detection example (detects backend from DataFrame type):
        >>> computer = HistogramComputer()
        >>> y_hist, bin_edges = computer.compute_histogram(pandas_df, column='value')  # Uses LocalBackend
    """

    def __init__(self, backend: Optional["ExecutionBackend"] = None):
        """Initialize HistogramComputer.

        Args:
            backend: Execution backend. If None, auto-detects from DataFrame type
                when compute_histogram is called (LocalBackend for pandas,
                RayBackend for Ray, SparkBackend for Spark).
        """
        self._backend = backend

    def compute_histogram(
        self,
        df: Any,
        column: str,
        bins: Union[int, np.ndarray] = 50,
        use_rice_rule: bool = False,
        approx_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram using distributed aggregations.

        This method computes the histogram WITHOUT collecting the raw data.
        It uses the backend's distributed aggregation to compute bin counts,
        then collects only the aggregated histogram.

        Args:
            df: DataFrame containing data (Spark DataFrame or pandas DataFrame)
            column: Column name to compute histogram for
            bins: Number of bins (int) or array of bin edges
            use_rice_rule: Use Rice rule to automatically determine bin count
            approx_count: Approximate row count (avoids full count if provided)

        Returns:
            Tuple of (y_hist, bin_edges) where:
                - y_hist: Normalized frequency density for each bin
                - bin_edges: Array of bin edge values (len = n_bins + 1)

        Example:
            >>> computer = HistogramComputer(backend)
            >>> y, x = computer.compute_histogram(df, 'value', bins=100)
            >>> # y and x are small numpy arrays (~100 elements)
        """
        # Get or create backend (auto-detect from DataFrame type if not configured)
        backend = self._get_backend(df)

        # Determine number of bins if using Rice rule
        if use_rice_rule:
            if approx_count is None:
                stats = backend.get_column_stats(df, column)
                approx_count = int(stats["count"])
            bins = int(np.ceil(approx_count ** (1 / 3)) * 2)

        # Ensure minimum of 2 bins (Bucketizer requires at least 3 splits)
        if isinstance(bins, int) and bins < 2:
            bins = 2

        # Get min and max values via backend
        stats = backend.get_column_stats(df, column)

        # Handle empty DataFrame or all-null values
        if stats["min"] is None or stats["max"] is None or stats["count"] == 0:
            raise ValueError(f"Cannot compute histogram: column '{column}' contains no valid (non-null) values")

        min_val, max_val = float(stats["min"]), float(stats["max"])

        # Handle edge case where min == max
        if min_val == max_val:
            # Return single bin centered at the value
            return np.array([1.0]), np.array([min_val])

        # Create bin edges
        if isinstance(bins, int):
            # Add small epsilon to max to ensure max value falls in last bin
            epsilon = (max_val - min_val) * 1e-10
            bin_edges = np.linspace(min_val, max_val + epsilon, bins + 1)
        else:
            bin_edges = np.asarray(bins)

        # Compute histogram using backend's distributed method
        bin_counts, total_count = backend.compute_histogram(df, column, bin_edges)

        # Normalize to density (area under curve = 1)
        bin_widths = np.diff(bin_edges)

        if total_count > 0:
            y_density = bin_counts / (total_count * bin_widths)
        else:
            # Edge case: no data
            y_density = bin_counts

        # Return bin edges for CDF-based fitting (more accurate than center-point evaluation)
        # Callers can compute centers as: (edges[:-1] + edges[1:]) / 2.0
        return y_density, bin_edges

    def _get_backend(self, df: Any = None) -> "ExecutionBackend":
        """Get or create the execution backend.

        Args:
            df: DataFrame (optional). If provided and no backend was configured,
                auto-detects backend from DataFrame type.

        Returns:
            The configured backend, or auto-detected backend based on DataFrame type.
        """
        if self._backend is None:
            if df is not None:
                from spark_bestfit.backends.factory import BackendFactory

                self._backend = BackendFactory.for_dataframe(df)
            else:
                # Default to SparkBackend for backward compatibility
                from spark_bestfit.backends.spark import SparkBackend

                self._backend = SparkBackend()
        return self._backend

    def compute_statistics(self, df: Any, column: str) -> dict:
        """Compute basic statistics for a column (useful for validation).

        Args:
            df: DataFrame (Spark DataFrame or pandas DataFrame)
            column: Column name

        Returns:
            Dictionary with min, max, count (and optionally mean, stddev)
        """
        backend = self._get_backend(df)
        return backend.get_column_stats(df, column)
