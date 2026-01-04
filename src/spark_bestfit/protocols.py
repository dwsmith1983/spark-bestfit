"""Protocol definitions for execution backend abstraction.

This module defines the ExecutionBackend protocol that enables spark-bestfit
to work with multiple distributed computing backends (Spark, Ray, local).

The protocol uses Python's structural subtyping (PEP 544), so any class that
implements the required methods is compatible without explicit inheritance.

Example:
    >>> from spark_bestfit.backends.spark import SparkBackend
    >>> from spark_bestfit import DistributionFitter
    >>>
    >>> backend = SparkBackend(spark)
    >>> fitter = DistributionFitter(backend=backend)
    >>> results = fitter.fit(df, column='value')
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


class ExecutionBackend(Protocol):
    """Protocol for distributed execution backends.

    This protocol defines the interface that backends must implement to support
    parallel distribution fitting. Backends handle:
    - Broadcasting data to workers
    - Parallel execution of fitting tasks
    - Data collection and sampling
    - Resource information (parallelism)

    Implementations:
        - SparkBackend: Apache Spark via Pandas UDFs (default)
        - LocalBackend: concurrent.futures for testing/development
        - RayBackend: Ray distributed computing (v2.0)
    """

    def broadcast(self, data: Any) -> Any:
        """Broadcast data to workers.

        Creates a read-only variable cached on each worker node, avoiding
        repeated serialization of large objects.

        Args:
            data: Data to broadcast (typically numpy arrays or tuples)

        Returns:
            Backend-specific broadcast handle. For Spark, this is a Broadcast
            object. For local backend, returns the data directly.
        """
        ...

    def destroy_broadcast(self, handle: Any) -> None:
        """Clean up broadcast variable resources.

        Releases memory held by broadcast variables on worker nodes.
        Should be called in a finally block after parallel operations complete.

        Args:
            handle: Broadcast handle returned by broadcast()
        """
        ...

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
        """Execute distribution fitting in parallel.

        Distributes fitting tasks across workers and collects results.
        Each distribution is fitted independently, making this embarrassingly
        parallel.

        Args:
            distributions: List of scipy distribution names to fit
            histogram: Tuple of (y_values, bin_edges) for continuous or
                (x_values, pmf) for discrete distributions
            data_sample: Sample data array for MLE fitting
            fit_func: Pure Python fitting function to apply. Should be
                fit_single_distribution or fit_discrete_distribution.
            column_name: Name of the source column (for result metadata)
            data_stats: Optional dict with data_min, data_max, data_mean,
                data_stddev, data_count for provenance
            num_partitions: Optional parallelism hint
            lower_bound: Optional lower bound for truncated fitting
            upper_bound: Optional upper bound for truncated fitting
            lazy_metrics: If True, skip expensive KS/AD computation
            is_discrete: If True, use discrete distribution fitting
            progress_callback: Optional callback for progress updates.
                Called with (completed, total, percent) after each distribution
                completes fitting. Callback is invoked from worker thread for
                LocalBackend/RayBackend, or via StatusTracker for SparkBackend.

        Returns:
            List of fit result dicts. Each dict contains:
            - distribution: str
            - column: str
            - params: tuple
            - sse, aic, bic: float
            - ks_statistic, ks_pvalue: Optional[float]
            - data_min, data_max, etc.: Optional[float]
        """
        ...

    def get_parallelism(self) -> int:
        """Get available parallelism (cores/executors).

        Returns the number of parallel tasks that can run concurrently.
        Used to optimize partition counts and work distribution.

        Returns:
            Number of available parallel execution slots
        """
        ...

    def collect_column(self, df: Any, column: str) -> np.ndarray:
        """Collect single column from DataFrame as numpy array.

        Gathers distributed data to the driver. Use sparingly for large
        datasets—prefer sampling when possible.

        Args:
            df: Backend-specific DataFrame (Spark DataFrame, Ray Dataset, etc.)
            column: Column name to collect

        Returns:
            Numpy array of column values
        """
        ...

    def get_column_stats(self, df: Any, column: str) -> Dict[str, float]:
        """Get column statistics (min, max, count).

        Computes basic statistics in a single distributed pass.
        More efficient than multiple separate aggregations.

        Args:
            df: Backend-specific DataFrame
            column: Column name

        Returns:
            Dict with keys: 'min', 'max', 'count'
        """
        ...

    def sample_column(
        self,
        df: Any,
        column: str,
        fraction: float,
        seed: int,
    ) -> np.ndarray:
        """Sample column data and collect as numpy array.

        Performs distributed sampling before collection, reducing data
        transfer for large datasets.

        Args:
            df: Backend-specific DataFrame
            column: Column name
            fraction: Fraction of data to sample (0 < fraction <= 1)
            seed: Random seed for reproducibility

        Returns:
            Numpy array of sampled values
        """
        ...

    def create_dataframe(
        self,
        data: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> Any:
        """Create a backend-specific DataFrame.

        Used to create the distribution name DataFrame for parallel fitting.

        Args:
            data: List of row tuples
            columns: Column names

        Returns:
            Backend-specific DataFrame
        """
        ...

    # =========================================================================
    # Copula and Histogram Methods (v2.0)
    # =========================================================================

    def compute_correlation(
        self,
        df: Any,
        columns: List[str],
        method: str = "spearman",
    ) -> np.ndarray:
        """Compute correlation matrix for specified columns.

        Used by GaussianCopula to compute the correlation structure.
        For Spark, this uses distributed Spark ML computation (no .toPandas()).
        For local, this uses pandas correlation.

        Args:
            df: Backend-specific DataFrame
            columns: List of column names to compute correlation for
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            Correlation matrix as numpy array of shape (n_columns, n_columns)
        """
        ...

    def compute_histogram(
        self,
        df: Any,
        column: str,
        bin_edges: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Compute histogram bin counts using distributed aggregation.

        Used by HistogramComputer for efficient distributed histogram
        computation without collecting raw data.

        Args:
            df: Backend-specific DataFrame
            column: Column name to compute histogram for
            bin_edges: Array of bin edge values (n_bins + 1 values)

        Returns:
            Tuple of (bin_counts, total_count) where bin_counts is an array
            of counts for each bin
        """
        ...

    def generate_samples(
        self,
        n: int,
        generator_func: Callable[[int, int, Optional[int]], Dict[str, np.ndarray]],
        column_names: List[str],
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Any:
        """Generate samples distributed across partitions.

        Used by GaussianCopula.sample_spark() for distributed sample
        generation. Each partition generates a subset of samples.

        Args:
            n: Total number of samples to generate
            generator_func: Function(n_samples, partition_id, seed) -> Dict[col, array]
                that generates samples for one partition
            column_names: Names of columns in output
            num_partitions: Number of partitions (None = backend default)
            random_seed: Base random seed (partition_id added for uniqueness)

        Returns:
            Backend-specific DataFrame with generated samples
        """
        ...
