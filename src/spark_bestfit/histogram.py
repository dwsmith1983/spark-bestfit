"""Distributed histogram computation using Spark without collecting raw data."""

from typing import Optional, Tuple, Union

import numpy as np
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame


class HistogramComputer:
    """Computes histograms efficiently using Spark aggregations.

    This implementation avoids collecting raw data to the driver by using
    Spark's distributed aggregation capabilities. Only the final histogram
    (typically ~100 values) is collected, not the raw dataset.

    Example:
        >>> computer = HistogramComputer()
        >>> y_hist, x_hist = computer.compute_histogram(
        ...     df, column='value', bins=50
        ... )
        >>> # y_hist and x_hist are numpy arrays (~100 values total)
    """

    def compute_histogram(
        self,
        df: DataFrame,
        column: str,
        bins: Union[int, np.ndarray] = 50,
        use_rice_rule: bool = False,
        approx_count: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram using distributed Spark aggregations.

        This method computes the histogram WITHOUT collecting the raw data.
        It uses Spark's Bucketizer and groupBy to compute bin counts in a
        distributed manner, then collects only the aggregated histogram.

        Args:
            df: Spark DataFrame containing data
            column: Column name to compute histogram for
            bins: Number of bins (int) or array of bin edges
            use_rice_rule: Use Rice rule to automatically determine bin count
            approx_count: Approximate row count (avoids full count if provided)

        Returns:
            Tuple of (y_hist, x_centers) where:
                - y_hist: Normalized frequency density for each bin
                - x_centers: Center point of each bin

        Example:
            >>> computer = HistogramComputer()
            >>> y, x = computer.compute_histogram(df, 'value', bins=100)
            >>> # y and x are small numpy arrays (~100 elements)
        """
        # Determine number of bins if using Rice rule
        if use_rice_rule:
            if approx_count is None:
                approx_count = df.count()
            bins = int(np.ceil(approx_count ** (1 / 3)) * 2)

        # Ensure minimum of 2 bins (Bucketizer requires at least 3 splits)
        if isinstance(bins, int) and bins < 2:
            bins = 2

        # Filter out null values before computing statistics
        df_filtered = df.filter(F.col(column).isNotNull())

        # Get min and max values (small aggregation, not a full collect)
        stats = df_filtered.agg(F.min(column).alias("min_val"), F.max(column).alias("max_val")).collect()[0]

        # Handle empty DataFrame or all-null values
        if stats["min_val"] is None or stats["max_val"] is None:
            raise ValueError(f"Cannot compute histogram: column '{column}' contains no valid (non-null) values")

        min_val, max_val = float(stats["min_val"]), float(stats["max_val"])

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

        # Compute histogram using distributed Bucketizer + groupBy
        # Use filtered DataFrame to avoid null bin_id issues
        histogram_df = self._compute_histogram_distributed(df_filtered, column, bin_edges)

        # Collect ONLY the aggregated histogram (small data)
        hist_data = histogram_df.orderBy("bin_id").collect()

        # Extract counts (fill missing bins with zeros)
        bin_counts = np.zeros(len(bin_edges) - 1)
        for row in hist_data:
            bin_id = int(row["bin_id"])
            if 0 <= bin_id < len(bin_counts):
                bin_counts[bin_id] = row["count"]

        # Compute bin centers
        x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Normalize to density (area under curve = 1)
        bin_widths = np.diff(bin_edges)
        total_count = bin_counts.sum()

        if total_count > 0:
            y_density = bin_counts / (total_count * bin_widths)
        else:
            # Edge case: no data
            y_density = bin_counts

        return y_density, x_centers

    @staticmethod
    def _compute_histogram_distributed(df: DataFrame, column: str, bin_edges: np.ndarray) -> DataFrame:
        """Compute histogram using Bucketizer and groupBy (stays distributed).

        This is the key optimization: we use Spark ML's Bucketizer to assign
        each row to a bin, then use groupBy to count rows per bin. This all
        happens in the cluster without collecting data to the driver.

        Args:
            df: Spark DataFrame
            column: Column to histogram
            bin_edges: Array of bin edge values

        Returns:
            DataFrame with columns (bin_id, count)
        """
        # Create temp column name to avoid conflicts
        temp_col = f"__{column}_bin_temp__"

        # Use Bucketizer to assign bin IDs
        bucketizer = Bucketizer(
            splits=bin_edges.tolist(),
            inputCol=column,
            outputCol=temp_col,
            handleInvalid="keep",  # Keep invalid values in a special bin
        )

        # Transform and aggregate
        bucketed = bucketizer.transform(df)
        histogram = bucketed.groupBy(temp_col).count().withColumnRenamed(temp_col, "bin_id")

        return histogram

    @staticmethod
    def compute_statistics(df: DataFrame, column: str) -> dict:
        """Compute basic statistics for a column (useful for validation).

        Args:
            df: Spark DataFrame
            column: Column name

        Returns:
            Dictionary with min, max, mean, stddev, count
        """
        stats = (
            df.agg(
                F.min(column).alias("min"),
                F.max(column).alias("max"),
                F.mean(column).alias("mean"),
                F.stddev(column).alias("stddev"),
                F.count(column).alias("count"),
            )
            .collect()[0]
            .asDict()
        )

        return {k: float(v) if v is not None else None for k, v in stats.items()}
