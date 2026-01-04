"""Ray backend for distributed distribution fitting.

This module provides the RayBackend class that implements the ExecutionBackend
protocol using Ray for parallel processing.

This backend is useful for:
- Ray clusters and distributed computing environments
- Integration with Ray-based ML pipelines
- Environments where Spark is not available but Ray is

Example:
    >>> import pandas as pd
    >>> from spark_bestfit.backends.ray import RayBackend
    >>> from spark_bestfit import DistributionFitter
    >>>
    >>> backend = RayBackend()  # Auto-initializes Ray if not already running
    >>> fitter = DistributionFitter(backend=backend)
    >>> df = pd.DataFrame({'value': [1.0, 2.0, 3.0, ...]})
    >>> results = fitter.fit(df, column='value')
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Conditional Ray import
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore[assignment]


def _fit_continuous_distribution(
    dist_name: str,
    data_sample: np.ndarray,
    bin_edges: np.ndarray,
    y_hist: np.ndarray,
    column_name: str,
    data_stats: Optional[Dict[str, float]],
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    lazy_metrics: bool,
) -> Dict[str, Any]:
    """Fit a single continuous distribution (runs as Ray task).

    This function is defined at module level so Ray can properly serialize it.
    """
    # Import here to ensure module is loaded in worker process
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


def _fit_discrete_distribution(
    dist_name: str,
    data_sample: np.ndarray,
    x_values: np.ndarray,
    empirical_pmf: np.ndarray,
    column_name: str,
    data_stats: Optional[Dict[str, float]],
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    lazy_metrics: bool,
) -> Dict[str, Any]:
    """Fit a single discrete distribution (runs as Ray task).

    This function is defined at module level so Ray can properly serialize it.
    """
    # Import here to ensure module is loaded in worker process
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


# Cached remote function references (created lazily)
_fit_continuous_remote: Optional[Any] = None
_fit_discrete_remote: Optional[Any] = None
_remote_functions_lock = threading.Lock()


def _get_remote_functions() -> Tuple[Any, Any]:
    """Get cached remote function references, creating them if needed.

    Thread-safe initialization of Ray remote function wrappers.

    Returns:
        Tuple of (fit_continuous_remote, fit_discrete_remote)
    """
    global _fit_continuous_remote, _fit_discrete_remote

    with _remote_functions_lock:
        if _fit_continuous_remote is None:
            _fit_continuous_remote = ray.remote(_fit_continuous_distribution)
        if _fit_discrete_remote is None:
            _fit_discrete_remote = ray.remote(_fit_discrete_distribution)

    return _fit_continuous_remote, _fit_discrete_remote


class RayBackend:
    """Ray backend using @ray.remote for parallel distribution fitting.

    This backend runs distribution fitting using Ray's task-based parallelism.
    It supports both Ray Datasets (for big data operations) and pandas DataFrames.

    Attributes:
        num_cpus: Number of CPUs available for parallel execution
    """

    def __init__(
        self,
        address: Optional[str] = None,
        num_cpus: Optional[int] = None,
    ):
        """Initialize RayBackend.

        Args:
            address: Ray cluster address. If None, starts local Ray.
                Use "auto" to connect to an existing cluster.
            num_cpus: Number of CPUs to use. If None, uses all available.

        Raises:
            ImportError: If ray is not installed.
        """
        if not RAY_AVAILABLE:
            raise ImportError("ray is required for RayBackend. " "Install with: pip install spark-bestfit[ray]")

        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(address=address, num_cpus=num_cpus, ignore_reinit_error=True)

        # Store available resources
        self._num_cpus = num_cpus or int(ray.available_resources().get("CPU", 1))

    def broadcast(self, data: Any) -> Any:
        """Put data in Ray's object store for sharing across tasks.

        Args:
            data: Data to broadcast (typically numpy arrays or tuples)

        Returns:
            Ray ObjectRef that can be passed to remote tasks
        """
        return ray.put(data)

    def destroy_broadcast(self, handle: Any) -> None:
        """No-op for Ray - uses automatic reference counting.

        Ray automatically garbage collects objects from the object store
        when no references remain. No explicit cleanup needed.

        Args:
            handle: ObjectRef (ignored - Ray handles cleanup automatically)
        """
        pass  # Ray uses automatic reference counting

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
        """Execute distribution fitting in parallel using Ray tasks.

        Uses @ray.remote tasks to fit distributions concurrently across
        the Ray cluster. Data is placed in the object store once and
        shared across all tasks efficiently.

        Args:
            distributions: List of scipy distribution names to fit
            histogram: Tuple of (y_hist, bin_edges) for continuous or
                (x_values, pmf) for discrete distributions
            data_sample: Sample data array for MLE fitting
            fit_func: Pure Python fitting function (unused - we call fitting directly)
            column_name: Name of the source column
            data_stats: Optional dict with data_min, data_max, etc.
            num_partitions: Ignored (Ray handles task scheduling)
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

        total = len(distributions)

        # Get cached remote function references
        fit_continuous_remote, fit_discrete_remote = _get_remote_functions()

        # Launch all fitting tasks in parallel
        if is_discrete:
            x_values, empirical_pmf = histogram
            futures = [
                fit_discrete_remote.remote(
                    dist_name=d,
                    data_sample=data_sample,
                    x_values=x_values,
                    empirical_pmf=empirical_pmf,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )
                for d in distributions
            ]
        else:
            y_hist, bin_edges = histogram
            futures = [
                fit_continuous_remote.remote(
                    dist_name=d,
                    data_sample=data_sample,
                    bin_edges=bin_edges,
                    y_hist=y_hist,
                    column_name=column_name,
                    data_stats=data_stats,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lazy_metrics=lazy_metrics,
                )
                for d in distributions
            ]

        # Collect results with progress tracking using ray.wait()
        results = []
        completed = 0
        pending = list(futures)

        while pending:
            # Wait for at least one task to complete
            done, pending = ray.wait(pending, num_returns=1)

            for ref in done:
                try:
                    result = ray.get(ref)
                    # Filter failed fits (SSE = infinity)
                    if result["sse"] < float(np.inf):
                        results.append(result)
                except Exception:
                    # Skip distributions that fail completely
                    pass

                # Update progress
                completed += 1
                if progress_callback is not None:
                    percent = (completed / total) * 100.0
                    try:
                        progress_callback(completed, total, percent)
                    except Exception:
                        pass  # Don't let callback errors break fitting

        return results

    def get_parallelism(self) -> int:
        """Get the number of available CPUs.

        Returns:
            Number of CPUs available for parallel execution
        """
        return self._num_cpus

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

    def collect_column(self, df: Any, column: str) -> np.ndarray:
        """Extract a column as numpy array.

        Supports both Ray Datasets and pandas DataFrames.

        Args:
            df: Ray Dataset or pandas DataFrame
            column: Column name to extract

        Returns:
            Numpy array of column values
        """
        if self._is_ray_dataset(df):
            return df.select_columns([column]).to_pandas()[column].values
        else:
            return df[column].values

    def get_column_stats(self, df: Any, column: str) -> Dict[str, float]:
        """Compute min, max, and count for a column.

        Supports both Ray Datasets and pandas DataFrames.
        For Ray Datasets, uses distributed aggregation without collecting data.

        Args:
            df: Ray Dataset or pandas DataFrame
            column: Column name

        Returns:
            Dict with keys: 'min', 'max', 'count'
        """
        if self._is_ray_dataset(df):
            # Use Ray's distributed aggregation (no .to_pandas() needed)
            ds = df.select_columns([column])
            return {
                "min": float(ds.min(column)),
                "max": float(ds.max(column)),
                "count": ds.count(),
            }
        else:
            return {
                "min": float(df[column].min()),
                "max": float(df[column].max()),
                "count": len(df),
            }

    def sample_column(
        self,
        df: Any,
        column: str,
        fraction: float,
        seed: int,
    ) -> np.ndarray:
        """Sample a column and return as numpy array.

        Supports both Ray Datasets and pandas DataFrames.

        Args:
            df: Ray Dataset or pandas DataFrame
            column: Column name
            fraction: Fraction to sample (0 < fraction <= 1)
            seed: Random seed for reproducibility

        Returns:
            Numpy array of sampled values
        """
        if self._is_ray_dataset(df):
            sampled = df.random_sample(fraction, seed=seed)
            return sampled.select_columns([column]).to_pandas()[column].values
        else:
            sample_df = df[[column]].sample(frac=fraction, random_state=seed)
            return sample_df[column].values

    def create_dataframe(
        self,
        data: List[Tuple[Any, ...]],
        columns: List[str],
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from local data.

        Returns pandas DataFrame (not Ray Dataset) since the distribution
        name list is small and doesn't benefit from distribution.

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

    def compute_correlation(
        self,
        df: Any,
        columns: List[str],
        method: str = "spearman",
    ) -> np.ndarray:
        """Compute correlation matrix using distributed statistics.

        Supports both Ray Datasets and pandas DataFrames.
        For Ray Datasets with Pearson correlation, uses distributed computation
        of sufficient statistics (sums, products) to compute correlation without
        collecting raw data. Spearman correlation requires ranking and thus
        collects data (like Spark ML's approach).

        Args:
            df: Ray Dataset or pandas DataFrame
            columns: List of column names to compute correlation for
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            Correlation matrix as numpy array of shape (n_columns, n_columns)
        """
        if self._is_ray_dataset(df):
            if method == "pearson":
                # Distributed Pearson using sufficient statistics
                return self._compute_pearson_distributed(df, columns)
            else:
                # Spearman requires ranking - collect to pandas
                # (This matches Spark ML behavior which also collects for Spearman)
                pandas_df = df.select_columns(columns).to_pandas()
                return pandas_df.corr(method=method).values
        else:
            return df[columns].corr(method=method).values

    def _compute_pearson_distributed(
        self,
        df: Any,
        columns: List[str],
    ) -> np.ndarray:
        """Compute Pearson correlation using distributed sufficient statistics.

        Uses map_batches to compute partial sums, sums of squares, and
        cross-products, then combines to compute correlation matrix.
        Final computation is fully vectorized using numpy operations.

        Args:
            df: Ray Dataset
            columns: List of column names

        Returns:
            Pearson correlation matrix as numpy array
        """
        n_cols = len(columns)
        col_names = columns  # Capture for closure

        def compute_partial_stats(batch: pd.DataFrame) -> pd.DataFrame:
            """Compute sufficient statistics for one batch (vectorized)."""
            # Get data matrix, drop rows with any NaN
            data = batch[col_names].dropna().values
            n = len(data)

            if n == 0:
                # Return zeros for empty batch
                return pd.DataFrame(
                    {
                        "count": [0],
                        "sums": [np.zeros(n_cols).tobytes()],
                        "sum_sqs": [np.zeros(n_cols).tobytes()],
                        "cross_prods": [np.zeros(n_cols * (n_cols - 1) // 2).tobytes()],
                    }
                )

            # Vectorized computation of sums and sums of squares
            sums = data.sum(axis=0)
            sum_sqs = (data**2).sum(axis=0)

            # Vectorized cross-products: compute full matrix then extract upper triangle
            cross_prod_matrix = data.T @ data  # (n_cols, n_cols)
            cross_prods = cross_prod_matrix[np.triu_indices(n_cols, k=1)]

            return pd.DataFrame(
                {
                    "count": [n],
                    "sums": [sums.tobytes()],
                    "sum_sqs": [sum_sqs.tobytes()],
                    "cross_prods": [cross_prods.tobytes()],
                }
            )

        # Apply to all batches
        partial_stats = df.select_columns(columns).map_batches(
            compute_partial_stats,
            batch_format="pandas",
        )

        # Collect and aggregate (small data - one row per batch)
        stats_df = partial_stats.to_pandas()
        total_count = stats_df["count"].sum()

        if total_count == 0:
            return np.eye(n_cols)

        # Vectorized aggregation: stack binary arrays and sum
        valid_rows = stats_df[stats_df["count"] > 0]

        if len(valid_rows) == 0:
            return np.eye(n_cols)

        # Convert binary columns to numpy arrays (vectorized)
        sums_matrix = np.vstack([np.frombuffer(b, dtype=np.float64) for b in valid_rows["sums"]])
        sum_sqs_matrix = np.vstack([np.frombuffer(b, dtype=np.float64) for b in valid_rows["sum_sqs"]])
        cross_prods_matrix = np.vstack([np.frombuffer(b, dtype=np.float64) for b in valid_rows["cross_prods"]])

        # Sum across batches (fully vectorized)
        sums_total = sums_matrix.sum(axis=0)
        sum_sqs_total = sum_sqs_matrix.sum(axis=0)
        cross_prods_total = cross_prods_matrix.sum(axis=0)

        # Compute means and variances (vectorized)
        means = sums_total / total_count
        variances = (sum_sqs_total / total_count) - (means**2)
        stds = np.sqrt(np.maximum(variances, 0))  # Avoid negative due to numerical error

        # Build E[XY] matrix from cross-products (vectorized)
        E_XY = np.zeros((n_cols, n_cols))
        E_XY[np.triu_indices(n_cols, k=1)] = cross_prods_total / total_count
        E_XY += E_XY.T  # Make symmetric
        np.fill_diagonal(E_XY, sum_sqs_total / total_count)  # E[X²] on diagonal

        # E[X]E[Y] matrix (outer product of means)
        E_X_E_Y = np.outer(means, means)

        # Covariance matrix
        cov_matrix = E_XY - E_X_E_Y

        # Correlation matrix (vectorized with safe division)
        std_outer = np.outer(stds, stds)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.where(std_outer > 0, cov_matrix / std_outer, 0.0)

        # Set diagonal to 1 and clamp for numerical stability
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        return corr_matrix

    def compute_histogram(
        self,
        df: Any,
        column: str,
        bin_edges: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Compute histogram bin counts using distributed aggregation.

        Supports both Ray Datasets and pandas DataFrames.
        For Ray Datasets, uses map_batches to compute partial histograms
        in each partition, then aggregates results without collecting raw data.

        Args:
            df: Ray Dataset or pandas DataFrame
            column: Column to histogram
            bin_edges: Array of bin edge values (n_bins + 1 values)

        Returns:
            Tuple of (bin_counts, total_count)
        """
        if self._is_ray_dataset(df):
            # Use map_batches for distributed histogram computation
            n_bins = len(bin_edges) - 1
            edges = bin_edges  # Capture for closure

            def compute_partial_histogram(batch: pd.DataFrame) -> pd.DataFrame:
                """Compute histogram for one batch, return as single row."""
                data = batch[column].dropna().values
                if len(data) == 0:
                    counts = np.zeros(n_bins)
                else:
                    counts, _ = np.histogram(data, bins=edges)
                # Return counts as columns (bin_0, bin_1, ..., bin_n-1)
                return pd.DataFrame([counts], columns=[f"bin_{i}" for i in range(n_bins)])

            # Apply to all batches and sum results
            partial_hists = df.select_columns([column]).map_batches(
                compute_partial_histogram,
                batch_format="pandas",
            )
            # Collect partial histograms (small data - one row per batch)
            partial_df = partial_hists.to_pandas()
            bin_counts = partial_df.sum().values.astype(float)
            total_count = int(bin_counts.sum())
            return bin_counts, total_count
        else:
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
        """Generate samples using Ray tasks.

        Uses Ray remote tasks to generate samples in parallel across
        multiple partitions, then concatenates results.

        Args:
            n: Total number of samples to generate
            generator_func: Function(n_samples, partition_id, seed) -> Dict[col, array]
            column_names: Names of columns in output
            num_partitions: Number of partitions (None = use CPU count)
            random_seed: Base random seed (partition_id added for uniqueness)

        Returns:
            Pandas DataFrame with generated samples
        """
        partitions = num_partitions or self._num_cpus

        # Calculate samples per partition
        base_samples = n // partitions
        remainder = n % partitions

        @ray.remote
        def generate_partition(
            partition_id: int,
            n_samples: int,
            gen_func: Callable,
            seed: Optional[int],
        ) -> pd.DataFrame:
            """Generate samples for one partition (Ray task)."""
            partition_seed = seed + partition_id if seed is not None else None
            samples = gen_func(n_samples, partition_id, partition_seed)
            return pd.DataFrame(samples)

        # Launch generation tasks
        futures = []
        for i in range(partitions):
            # First 'remainder' partitions get one extra sample
            partition_n = base_samples + (1 if i < remainder else 0)
            if partition_n > 0:
                futures.append(generate_partition.remote(i, partition_n, generator_func, random_seed))

        # Collect and concatenate results
        partition_dfs = ray.get(futures)
        if partition_dfs:
            return pd.concat(partition_dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=column_names)

    @staticmethod
    def _is_ray_dataset(df: Any) -> bool:
        """Check if df is a Ray Dataset.

        Args:
            df: Object to check

        Returns:
            True if df is a Ray Dataset, False otherwise
        """
        # Check for Ray Dataset by looking for characteristic methods
        return hasattr(df, "select_columns") and hasattr(df, "to_pandas")
