"""Local backend for testing and development without Spark.

This module provides the LocalBackend class that implements the ExecutionBackend
protocol using Python's concurrent.futures for parallel processing.

This backend is useful for:
- Unit testing without Spark dependency
- Development and debugging on small datasets
- Environments where Spark is not available

Example:
    >>> import pandas as pd
    >>> from spark_bestfit.backends.local import LocalBackend
    >>> from spark_bestfit import DistributionFitter
    >>>
    >>> backend = LocalBackend(max_workers=4)
    >>> fitter = DistributionFitter(backend=backend)
    >>> # Note: DataFrames are pandas DataFrames with LocalBackend
    >>> df = pd.DataFrame({'value': [1.0, 2.0, 3.0, ...]})
    >>> results = fitter.fit(df, column='value')
"""

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class LocalBackend:
    """Local backend using ThreadPoolExecutor for parallel distribution fitting.

    This backend runs distribution fitting locally using Python threads.
    It's primarily useful for testing and development without requiring
    a Spark cluster.

    Attributes:
        max_workers: Number of worker threads for parallel execution
    """

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize LocalBackend.

        Args:
            max_workers: Maximum number of worker threads. If None, uses
                the number of CPU cores.
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()

    @staticmethod
    def broadcast(data: Any) -> Any:
        """No-op broadcast for local execution.

        In local mode, data is already accessible to all threads, so we
        simply return the data as-is.

        Args:
            data: Data to "broadcast"

        Returns:
            The same data (no transformation needed)
        """
        return data

    def destroy_broadcast(self, handle: Any) -> None:
        """No-op cleanup for local execution.

        Args:
            handle: Data reference (ignored)
        """
        pass  # Nothing to clean up in local mode

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
        """Execute distribution fitting in parallel using threads.

        Uses ThreadPoolExecutor to fit distributions concurrently. Each
        distribution is fitted independently using the provided fit_func.

        Args:
            distributions: List of scipy distribution names to fit
            histogram: Tuple of (y_hist, bin_edges) for continuous or
                (x_values, pmf) for discrete distributions
            data_sample: Sample data array for MLE fitting
            fit_func: Pure Python fitting function to apply. For continuous
                distributions, this is fit_single_distribution. For discrete,
                use fit_single_discrete_distribution.
            column_name: Name of the source column
            data_stats: Optional dict with data_min, data_max, etc.
            num_partitions: Ignored (uses max_workers instead)
            lower_bound: Lower bound for truncated fitting
            upper_bound: Upper bound for truncated fitting
            lazy_metrics: If True, skip expensive KS/AD computation
            is_discrete: If True, use discrete distribution fitting
            progress_callback: Optional callback for progress updates.
                Called with (completed, total, percent) after each distribution.

        Returns:
            List of fit result dicts (only successful fits, SSE < inf)
        """
        if not distributions:
            return []

        # Check for empty or invalid data sample
        if len(data_sample) == 0:
            return []

        # Unpack histogram based on distribution type
        if is_discrete:
            x_values, empirical_pmf = histogram
            y_hist = None
            bin_edges = None
        else:
            y_hist, bin_edges = histogram
            x_values = None
            empirical_pmf = None

        def fit_one_distribution(dist_name: str) -> Dict[str, Any]:
            """Fit a single distribution (runs in thread pool)."""
            if is_discrete:
                # Import inside function to avoid circular imports
                from spark_bestfit.discrete_fitting import fit_single_discrete_distribution
                from spark_bestfit.distributions import DiscreteDistributionRegistry

                registry = DiscreteDistributionRegistry()
                return fit_single_discrete_distribution(
                    dist_name=dist_name,
                    data_sample=data_sample.astype(int),
                    x_values=x_values,
                    empirical_pmf=empirical_pmf,
                    registry=registry,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )
            else:
                from spark_bestfit.fitting import fit_single_distribution

                return fit_single_distribution(
                    dist_name=dist_name,
                    data_sample=data_sample,
                    bin_edges=bin_edges,
                    y_hist=y_hist,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )

        # Execute in parallel using ThreadPoolExecutor
        results = []
        total = len(distributions)
        completed = 0
        lock = threading.Lock()  # Thread safety for shared state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fit_one_distribution, d): d for d in distributions}

            # Use as_completed for progress tracking
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Filter failed fits (SSE = infinity)
                    if result["sse"] < float(np.inf):
                        results.append(result)
                except Exception:
                    # Skip distributions that fail completely
                    pass

                # Update progress with thread safety
                with lock:
                    completed += 1
                    current_completed = completed
                if progress_callback is not None:
                    percent = (current_completed / total) * 100.0
                    try:
                        progress_callback(current_completed, total, percent)
                    except Exception:
                        pass  # Don't let callback errors break fitting

        return results

    def get_parallelism(self) -> int:
        """Get the number of worker threads.

        Returns:
            Number of parallel execution slots (max_workers)
        """
        return self.max_workers

    @staticmethod
    def collect_column(df: pd.DataFrame, column: str) -> np.ndarray:
        """Extract a column from pandas DataFrame as numpy array.

        Args:
            df: Pandas DataFrame
            column: Column name to extract

        Returns:
            Numpy array of column values
        """
        return df[column].values

    @staticmethod
    def get_column_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
        """Compute min, max, and count for a column.

        Args:
            df: Pandas DataFrame
            column: Column name

        Returns:
            Dict with keys: 'min', 'max', 'count'
        """
        return {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "count": len(df[column]),
        }

    @staticmethod
    def sample_column(
        df: pd.DataFrame,
        column: str,
        fraction: float,
        seed: int,
    ) -> np.ndarray:
        """Sample a column and return as numpy array.

        Filters out NaN and infinite values before sampling to ensure
        clean data for distribution fitting.

        Args:
            df: Pandas DataFrame
            column: Column name
            fraction: Fraction to sample (0 < fraction <= 1)
            seed: Random seed for reproducibility

        Returns:
            Numpy array of sampled values (NaN/inf filtered)
        """
        # Filter out NaN and inf values before sampling
        clean_df = df[[column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_df) == 0:
            return np.array([])
        sample_df = clean_df.sample(frac=fraction, random_state=seed)
        return sample_df[column].values

    @staticmethod
    def create_dataframe(
        data: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from local data.

        Args:
            data: List of row tuples
            columns: Column names

        Returns:
            Pandas DataFrame
        """
        return pd.DataFrame(data, columns=columns)

    # =========================================================================
    # Copula and Histogram Methods (v2.0)
    # =========================================================================

    @staticmethod
    def compute_correlation(
        df: pd.DataFrame,
        columns: List[str],
        method: str = "spearman",
    ) -> np.ndarray:
        """Compute correlation matrix using pandas.

        Args:
            df: Pandas DataFrame
            columns: List of column names to compute correlation for
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            Correlation matrix as numpy array of shape (n_columns, n_columns)
        """
        return df[columns].corr(method=method).values

    @staticmethod
    def compute_histogram(
        df: pd.DataFrame,
        column: str,
        bin_edges: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Compute histogram bin counts using numpy.

        Args:
            df: Pandas DataFrame
            column: Column to histogram
            bin_edges: Array of bin edge values (n_bins + 1 values)

        Returns:
            Tuple of (bin_counts, total_count) where bin_counts is an array
            of counts for each bin
        """
        data = df[column].dropna().values
        bin_counts, _ = np.histogram(data, bins=bin_edges)
        total_count = int(bin_counts.sum())
        return bin_counts.astype(float), total_count

    def generate_samples(
        self,
        n: int,
        generator_func: Callable[[int, int, Optional[int]], Dict[str, np.ndarray]],
        column_names: List[str],
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate samples locally.

        Unlike SparkBackend, this generates all samples in a single call
        since there's no distributed cluster to leverage.

        Args:
            n: Total number of samples to generate
            generator_func: Function(n_samples, partition_id, seed) -> Dict[col, array]
                that generates samples for one partition
            column_names: Names of columns in output (for interface compatibility)
            num_partitions: Ignored (no partitioning in local mode)
            random_seed: Random seed for reproducibility

        Returns:
            Pandas DataFrame with generated samples
        """
        # Generate all samples in one call (partition_id=0)
        samples = generator_func(n, 0, random_seed)
        return pd.DataFrame(samples)
