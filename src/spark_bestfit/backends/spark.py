"""Apache Spark backend for distributed distribution fitting.

This module provides the SparkBackend class that implements the ExecutionBackend
protocol using Apache Spark's Pandas UDFs for parallel processing.

Example:
    >>> from pyspark.sql import SparkSession
    >>> from spark_bestfit.backends.spark import SparkBackend
    >>> from spark_bestfit import DistributionFitter
    >>>
    >>> spark = SparkSession.builder.getOrCreate()
    >>> backend = SparkBackend(spark)
    >>> fitter = DistributionFitter(backend=backend)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from spark_bestfit.utils import get_spark_session


class SparkBackend:
    """Apache Spark backend using Pandas UDFs for parallel distribution fitting.

    This is the default backend for spark-bestfit. It uses Spark's broadcast
    variables for efficient data sharing and Pandas UDFs for vectorized
    distribution fitting across the cluster.

    Attributes:
        spark: The SparkSession instance used for distributed operations
    """

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize SparkBackend.

        Args:
            spark: SparkSession instance. If None, attempts to get the active
                session or create a new one.

        Raises:
            RuntimeError: If no SparkSession provided and no active session exists
        """
        self.spark = get_spark_session(spark)

    def broadcast(self, data: Any) -> Any:
        """Broadcast data to all Spark executors.

        Creates a read-only variable cached on each worker node. This is
        essential for sharing histogram and sample data efficiently without
        sending copies with each task.

        Args:
            data: Data to broadcast (numpy arrays, tuples, etc.)

        Returns:
            Spark Broadcast object wrapping the data
        """
        return self.spark.sparkContext.broadcast(data)

    @staticmethod
    def destroy_broadcast(handle: Any) -> None:
        """Release broadcast variable from executor memory.

        Uses unpersist() rather than destroy() because Spark's lazy evaluation
        may still reference the broadcast in pending operations.

        Args:
            handle: Broadcast variable returned by broadcast()
        """
        if handle is not None:
            handle.unpersist()

    def parallel_fit(
        self,
        distributions: List[str],
        histogram: Tuple[np.ndarray, np.ndarray],
        data_sample: np.ndarray,
        fit_func: Callable[..., Dict[str, Any]],
        column_name: str,
        data_stats: Optional[Dict[str, float]] = None,
        num_partitions: Optional[int] = None,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        lazy_metrics: bool = False,
        is_discrete: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute distribution fitting in parallel using Pandas UDFs.

        This method encapsulates all Spark-specific operations for fitting:
        1. Broadcasts histogram and sample data to executors
        2. Creates a DataFrame of distribution names
        3. Applies the fitting UDF to compute results in parallel
        4. Collects and returns results

        Args:
            distributions: List of scipy distribution names to fit
            histogram: Tuple of (y_hist, bin_edges) for continuous or
                (x_values, pmf) for discrete distributions
            data_sample: Sample data array for MLE fitting
            fit_func: Pure Python fitting function (not used directly here;
                we use the Pandas UDF factories instead)
            column_name: Name of the source column
            data_stats: Optional dict with data_min, data_max, etc.
            num_partitions: Number of partitions (None = auto)
            lower_bound: Lower bound for truncated fitting
            upper_bound: Upper bound for truncated fitting
            lazy_metrics: If True, skip expensive KS/AD computation
            is_discrete: If True, use discrete distribution fitting

        Returns:
            List of fit result dicts
        """
        # Handle empty distribution list
        if not distributions:
            return []

        # Broadcast data to executors
        histogram_bc = self.broadcast(histogram)
        data_sample_bc = self.broadcast(data_sample)

        try:
            # Create DataFrame of distributions
            dist_df = self.create_dataframe(
                data=[(d,) for d in distributions],
                columns=["distribution_name"],
            )

            # Repartition for optimal parallelism
            n_partitions = num_partitions or self._calculate_partitions(distributions)
            dist_df = dist_df.repartition(n_partitions)

            # Create and apply appropriate fitting UDF
            if is_discrete:
                from spark_bestfit.discrete_fitting import create_discrete_fitting_udf

                fitting_udf = create_discrete_fitting_udf(
                    histogram_bc,
                    data_sample_bc,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )
            else:
                from spark_bestfit.fitting import create_fitting_udf

                fitting_udf = create_fitting_udf(
                    histogram_bc,
                    data_sample_bc,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )

            # Apply UDF and expand struct
            results_df = dist_df.select(fitting_udf(F.col("distribution_name")).alias("result")).select("result.*")

            # Filter failed fits (SSE = infinity)
            results_df = results_df.filter(F.col("sse") < float(np.inf))

            # Collect results to driver
            return [row.asDict() for row in results_df.collect()]

        finally:
            # Always clean up broadcast variables
            self.destroy_broadcast(histogram_bc)
            self.destroy_broadcast(data_sample_bc)

    def get_parallelism(self) -> int:
        """Get the default parallelism from Spark configuration.

        Returns the total number of cores available across the cluster,
        which is used to determine optimal partition counts.

        Returns:
            Number of available parallel execution slots
        """
        return self.spark.sparkContext.defaultParallelism

    @staticmethod
    def collect_column(df: DataFrame, column: str) -> np.ndarray:
        """Collect a single column from Spark DataFrame as numpy array.

        Warning: This collects data to the driver node. Use sparingly
        for large datasets.

        Args:
            df: Spark DataFrame
            column: Column name to collect

        Returns:
            Numpy array of column values
        """
        return df.select(column).toPandas()[column].values

    @staticmethod
    def get_column_stats(df: DataFrame, column: str) -> Dict[str, float]:
        """Compute min, max, and count for a column in a single pass.

        Uses Spark aggregations to compute statistics efficiently without
        collecting all data to the driver.

        Args:
            df: Spark DataFrame
            column: Column name

        Returns:
            Dict with keys: 'min', 'max', 'count'
        """
        stats = df.agg(
            F.min(column).alias("min"),
            F.max(column).alias("max"),
            F.count(column).alias("count"),
        ).first()

        return {
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "count": int(stats["count"]),
        }

    @staticmethod
    def sample_column(
        df: DataFrame,
        column: str,
        fraction: float,
        seed: int,
    ) -> np.ndarray:
        """Sample a column and collect as numpy array.

        Performs distributed sampling before collection, reducing the amount
        of data transferred to the driver.

        Args:
            df: Spark DataFrame
            column: Column name
            fraction: Fraction to sample (0 < fraction <= 1)
            seed: Random seed for reproducibility

        Returns:
            Numpy array of sampled values
        """
        sample_df = df.select(column).sample(fraction=fraction, seed=seed)
        return sample_df.toPandas()[column].values

    def create_dataframe(
        self,
        data: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> DataFrame:
        """Create a Spark DataFrame from local data.

        Used internally to create the distribution name DataFrame for
        parallel fitting.

        Args:
            data: List of row tuples
            columns: Column names

        Returns:
            Spark DataFrame
        """
        return self.spark.createDataFrame(data, columns)

    def _calculate_partitions(self, distributions: List[str]) -> int:
        """Calculate optimal partition count for distribution fitting.

        Uses distribution-aware weighting where slow distributions count
        as 3x for partition calculation to reduce straggler effects.

        Args:
            distributions: List of distribution names to fit

        Returns:
            Optimal partition count
        """
        from spark_bestfit.distributions import DistributionRegistry

        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        slow_count = sum(1 for d in distributions if d in slow_set)
        # Slow distributions count 3x (1 base + 2 extra)
        effective_count = len(distributions) + slow_count * 2
        total_cores = self.get_parallelism()
        return min(effective_count, total_cores * 2)
