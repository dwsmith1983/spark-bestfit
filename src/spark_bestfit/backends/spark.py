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
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
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
            progress_callback: Optional callback for progress updates.
                Called with (completed_tasks, total_tasks, percent) at the
                Spark task level via StatusTracker polling.

        Returns:
            List of fit result dicts
        """
        # Handle empty distribution list
        if not distributions:
            return []

        # Start progress tracking if callback provided
        tracker = None
        if progress_callback is not None:
            from spark_bestfit.progress import ProgressTracker

            tracker = ProgressTracker(self.spark, progress_callback)
            tracker.start()

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
            # Stop progress tracking
            if tracker is not None:
                tracker.stop()

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
            Dict with keys: 'min', 'max', 'count'. Values are NaN for
            empty DataFrames or columns with all null values, ensuring
            consistent return type with LocalBackend and RayBackend.
        """
        stats = df.agg(
            F.min(column).alias("min"),
            F.max(column).alias("max"),
            F.count(column).alias("count"),
        ).first()

        return {
            "min": float(stats["min"]) if stats["min"] is not None else float("nan"),
            "max": float(stats["max"]) if stats["max"] is not None else float("nan"),
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

    # =========================================================================
    # Copula and Histogram Methods (v2.0)
    # =========================================================================

    @staticmethod
    def compute_correlation(
        df: DataFrame,
        columns: List[str],
        method: str = "spearman",
    ) -> np.ndarray:
        """Compute correlation matrix using Spark ML.

        Uses distributed computation via Spark ML's Correlation, enabling
        correlation computation on DataFrames with billions of rows without
        collecting data to the driver.

        Args:
            df: Spark DataFrame
            columns: List of column names to compute correlation for
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            Correlation matrix as numpy array of shape (n_columns, n_columns)
        """
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.stat import Correlation

        # Assemble columns into a vector
        assembler = VectorAssembler(
            inputCols=columns,
            outputCol="_corr_features",
            handleInvalid="skip",  # Skip rows with nulls
        )
        vector_df = assembler.transform(df).select("_corr_features")

        # Compute correlation using Spark ML
        corr_result = Correlation.corr(vector_df, "_corr_features", method=method)

        # Extract correlation matrix from result
        corr_matrix = corr_result.head()[0].toArray()

        return corr_matrix

    @staticmethod
    def compute_histogram(
        df: DataFrame,
        column: str,
        bin_edges: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Compute histogram using distributed Bucketizer and groupBy.

        This is the key optimization: uses Spark ML's Bucketizer to assign
        each row to a bin, then uses groupBy to count rows per bin. All
        computation happens in the cluster without collecting data.

        Args:
            df: Spark DataFrame
            column: Column to histogram
            bin_edges: Array of bin edge values (n_bins + 1 values)

        Returns:
            Tuple of (bin_counts, total_count) where bin_counts is an array
            of counts for each bin
        """
        from pyspark.ml.feature import Bucketizer

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

        # Collect ONLY the aggregated histogram (small data)
        hist_data = histogram.orderBy("bin_id").collect()

        # Extract counts (fill missing bins with zeros)
        bin_counts = np.zeros(len(bin_edges) - 1)
        total_count = 0
        for row in hist_data:
            bin_id = row["bin_id"]
            count = row["count"]
            # Skip None bin_id (can occur with handleInvalid="keep" for out-of-range values)
            if bin_id is not None:
                bin_id = int(bin_id)
                if 0 <= bin_id < len(bin_counts):
                    bin_counts[bin_id] = count
                    total_count += count

        return bin_counts, total_count

    def generate_samples(
        self,
        n: int,
        generator_func: Callable[[int, int, Optional[int]], Dict[str, np.ndarray]],
        column_names: List[str],
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> DataFrame:
        """Generate samples distributed across Spark partitions.

        Uses mapInPandas to generate samples in each partition, enabling
        generation of millions of samples distributed across the cluster.

        Args:
            n: Total number of samples to generate
            generator_func: Function(n_samples, partition_id, seed) -> Dict[col, array]
                that generates samples for one partition
            column_names: Names of columns in output
            num_partitions: Number of partitions (None = default parallelism)
            random_seed: Base random seed (partition_id added for uniqueness)

        Returns:
            Spark DataFrame with generated samples
        """
        import pandas as pd
        from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

        if num_partitions is None:
            num_partitions = self.get_parallelism()

        # Calculate samples per partition
        base_samples = n // num_partitions
        remainder = n % num_partitions

        # Create partition info DataFrame
        partition_data = []
        for i in range(num_partitions):
            samples_for_partition = base_samples + (1 if i < remainder else 0)
            if samples_for_partition > 0:
                partition_data.append((i, samples_for_partition))

        partition_df = self.spark.createDataFrame(
            partition_data,
            StructType(
                [
                    StructField("partition_id", IntegerType(), False),
                    StructField("n_samples", IntegerType(), False),
                ]
            ),
        )

        # Repartition to ensure parallelism
        partition_df = partition_df.repartition(len(partition_data))

        # Define output schema
        output_fields = [StructField(col, DoubleType(), False) for col in column_names]
        output_schema = StructType(output_fields)

        # Create the mapInPandas function
        def generate_partition_samples(iterator):
            """Generate samples for each partition."""
            for pdf in iterator:
                if len(pdf) == 0:
                    continue

                for idx in range(len(pdf)):
                    n_samples = int(pdf.iloc[idx]["n_samples"])
                    partition_id = int(pdf.iloc[idx]["partition_id"])

                    # Compute seed for this partition
                    seed = None
                    if random_seed is not None:
                        seed = random_seed + partition_id

                    # Generate samples using the provided function
                    samples = generator_func(n_samples, partition_id, seed)

                    yield pd.DataFrame(samples)

        # Apply the generator
        result_df = partition_df.mapInPandas(
            generate_partition_samples,
            schema=output_schema,
        )

        return result_df
