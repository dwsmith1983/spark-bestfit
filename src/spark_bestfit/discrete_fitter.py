"""Discrete distribution fitting engine for Spark."""

import logging
from functools import reduce
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import NumericType

from spark_bestfit.discrete_fitting import (
    DISCRETE_FIT_RESULT_SCHEMA,
    compute_discrete_histogram,
    create_discrete_sample_data,
    fit_single_discrete_distribution,
)
from spark_bestfit.distributions import DiscreteDistributionRegistry
from spark_bestfit.fitting import FITTING_SAMPLE_SIZE, compute_data_stats
from spark_bestfit.results import DistributionFitResult, FitResults, LazyMetricsContext
from spark_bestfit.utils import get_spark_session

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend

logger = logging.getLogger(__name__)

# Re-export for convenience
DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS: Tuple[str, ...] = tuple(DiscreteDistributionRegistry.DEFAULT_EXCLUSIONS)


class DiscreteDistributionFitter:
    """Spark distribution fitting engine for discrete (count) data.

    Efficiently fits scipy.stats discrete distributions to integer data using
    Spark's parallel processing capabilities. Uses MLE optimization since
    scipy discrete distributions don't have a built-in fit() method.

    Metric Selection:
        For discrete distributions, **AIC is recommended** for model selection:
        - ``aic``: Proper model selection criterion with complexity penalty
        - ``bic``: Similar to AIC but stronger penalty for complex models
        - ``ks_statistic``: Valid for ranking, but p-values are not reliable
        - ``sse``: Simple comparison metric

        The K-S test assumes continuous distributions. For discrete data,
        the K-S statistic can rank fits, but p-values are conservative and
        should not be used for hypothesis testing.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from spark_bestfit import DiscreteDistributionFitter
        >>>
        >>> spark = SparkSession.builder.appName("my-app").getOrCreate()
        >>> df = spark.createDataFrame([(x,) for x in count_data], ['counts'])
        >>>
        >>> fitter = DiscreteDistributionFitter(spark)
        >>> results = fitter.fit(df, column='counts')
        >>>
        >>> # Use AIC for model selection (recommended)
        >>> best = results.best(n=1, metric='aic')[0]
        >>> print(f"Best: {best.distribution} (AIC={best.aic:.2f})")
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        excluded_distributions: Optional[Tuple[str, ...]] = None,
        random_seed: int = 42,
        backend: Optional["ExecutionBackend"] = None,
    ):
        """Initialize DiscreteDistributionFitter.

        Args:
            spark: SparkSession. If None, uses the active session.
                Ignored if ``backend`` is provided.
            excluded_distributions: Distributions to exclude from fitting.
                Defaults to DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS.
                Pass an empty tuple ``()`` to include ALL scipy discrete distributions.
            random_seed: Random seed for reproducible sampling.
            backend: Optional execution backend (v2.0). If None, creates a
                SparkBackend from the spark session. Allows plugging in
                alternative backends like LocalBackend for testing.

        Raises:
            RuntimeError: If no SparkSession provided and no active session exists
        """
        # Initialize backend (lazy import to avoid circular dependency)
        if backend is not None:
            self._backend = backend
            # Extract SparkSession from SparkBackend if available
            if hasattr(backend, "spark"):
                self.spark = backend.spark
            else:
                # For non-Spark backends (LocalBackend, RayBackend), no SparkSession needed
                self.spark = None
        else:
            self.spark = get_spark_session(spark)
            # Lazy import to avoid circular dependency
            from spark_bestfit.backends.spark import SparkBackend

            self._backend = SparkBackend(self.spark)

        self.excluded_distributions = (
            excluded_distributions if excluded_distributions is not None else DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS
        )
        self.random_seed = random_seed
        # When excluded_distributions=() is explicitly passed, disable registry's
        # default exclusions so ALL scipy discrete distributions are available
        if excluded_distributions == ():
            self._registry = DiscreteDistributionRegistry(custom_exclusions=set())
        else:
            self._registry = DiscreteDistributionRegistry()

    def fit(
        self,
        df: DataFrame,
        column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        max_distributions: Optional[int] = None,
        enable_sampling: bool = True,
        sample_fraction: Optional[float] = None,
        max_sample_size: int = 1_000_000,
        sample_threshold: int = 10_000_000,
        num_partitions: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        bounded: bool = False,
        lower_bound: Optional[Union[float, Dict[str, float]]] = None,
        upper_bound: Optional[Union[float, Dict[str, float]]] = None,
        lazy_metrics: bool = False,
        prefilter: Union[bool, str] = False,
    ) -> FitResults:
        """Fit discrete distributions to integer data column(s).

        Args:
            df: Spark DataFrame containing integer count data
            column: Name of single column to fit distributions to
            columns: List of column names for multi-column fitting
            max_distributions: Limit number of distributions (for testing)
            enable_sampling: Enable sampling for large datasets
            sample_fraction: Fraction to sample (None = auto-determine)
            max_sample_size: Maximum rows to sample when auto-determining
            sample_threshold: Row count above which sampling is applied
            num_partitions: Spark partitions (None = auto-determine)
            progress_callback: Optional callback for progress updates.
                Called with (completed_tasks, total_tasks, percent_complete).
                Callback is invoked from background thread - ensure thread-safety.
            bounded: Enable bounded distribution fitting. When True, bounds
                are auto-detected from data or use explicit lower_bound/upper_bound.
            lower_bound: Lower bound for truncated distribution fitting.
                Can be a float (applied to all columns) or a dict mapping
                column names to bounds (v1.5.0). If None and bounded=True,
                auto-detects from each column's minimum.
            upper_bound: Upper bound for truncated distribution fitting.
                Can be a float (applied to all columns) or a dict mapping
                column names to bounds (v1.5.0). If None and bounded=True,
                auto-detects from each column's maximum.
            lazy_metrics: If True, defer computation of expensive KS metrics
                until accessed (v1.5.0). Improves fitting performance when only
                using AIC/BIC/SSE for model selection. Default False for
                backward compatibility.
            prefilter: Pre-filter distributions (v1.6.0). Currently only supported
                for continuous distributions. For discrete, this parameter is
                accepted but ignored (logs a warning if enabled).

        Returns:
            FitResults object with fitted distributions

        Raises:
            ValueError: If column not found, DataFrame empty, or invalid params
            TypeError: If column is not numeric

        Example:
            >>> # Single column
            >>> results = fitter.fit(df, column='counts')
            >>> best = results.best(n=1, metric='aic')
            >>>
            >>> # Multi-column
            >>> results = fitter.fit(df, columns=['counts1', 'counts2'])
            >>> best_per_col = results.best_per_column(n=1, metric='aic')
            >>>
            >>> # Bounded fitting
            >>> results = fitter.fit(df, column='counts', bounded=True, lower_bound=0, upper_bound=100)
            >>>
            >>> # Per-column bounds (v1.5.0)
            >>> results = fitter.fit(
            ...     df, columns=['counts1', 'counts2'],
            ...     bounded=True,
            ...     lower_bound={'counts1': 0, 'counts2': 5},
            ...     upper_bound={'counts1': 100, 'counts2': 200}
            ... )
            >>>
            >>> # Lazy metrics for faster fitting when only using AIC/BIC (v1.5.0)
            >>> results = fitter.fit(df, 'counts', lazy_metrics=True)
            >>> best_aic = results.best(n=1, metric='aic')[0]  # Fast, no KS computed
        """
        # Validate column/columns parameters
        if column is None and columns is None:
            raise ValueError("Must provide either 'column' or 'columns' parameter")
        if column is not None and columns is not None:
            raise ValueError("Cannot provide both 'column' and 'columns' - use one or the other")

        # Normalize to list of columns
        target_columns = [column] if column is not None else columns

        # Input validation for all columns
        for col in target_columns:
            self._validate_inputs(df, col, max_distributions, sample_fraction)

        # Warn if prefilter is enabled (not yet supported for discrete)
        if prefilter:
            logger.warning("prefilter is not yet supported for discrete distributions; ignoring")

        # Validate bounds - handle both scalar and dict forms
        self._validate_bounds(lower_bound, upper_bound, target_columns)

        # Get row count (single operation for all columns)
        # Handle Spark DataFrame, Ray Dataset, and pandas DataFrame
        if hasattr(df, "sparkSession"):
            row_count = df.count()
        elif hasattr(df, "select_columns") and hasattr(df, "count"):
            # Ray Dataset - use count() method
            row_count = df.count()
        else:
            row_count = len(df)
        if row_count == 0:
            raise ValueError("DataFrame is empty")
        logger.info(f"Row count: {row_count}")

        # Build per-column bounds dict: {col: (lower, upper)}
        column_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if bounded:
            column_bounds = self._resolve_bounds(df, target_columns, lower_bound, upper_bound)

        # Sample if needed (single operation for all columns)
        df_sample = self._apply_sampling(
            df, row_count, enable_sampling, sample_fraction, max_sample_size, sample_threshold
        )

        # Get distributions to fit (same for all columns)
        distributions = self._registry.get_distributions(
            additional_exclusions=list(self.excluded_distributions),
        )
        if max_distributions is not None and max_distributions > 0:
            distributions = distributions[:max_distributions]

        # Start progress tracking if callback provided (Spark only)
        tracker = None
        if progress_callback is not None and self.spark is not None:
            from spark_bestfit.progress import ProgressTracker

            tracker = ProgressTracker(self.spark, progress_callback)
            tracker.start()

        try:
            # Fit each column and collect results
            all_results_dfs = []
            lazy_contexts: Dict[str, LazyMetricsContext] = {}

            for col in target_columns:
                # Get per-column bounds (empty dict if not bounded)
                col_lower, col_upper = column_bounds.get(col, (None, None))
                logger.info(f"Fitting discrete column '{col}'...")
                results_df = self._fit_single_column(
                    df_sample=df_sample,
                    column=col,
                    row_count=row_count,
                    distributions=distributions,
                    num_partitions=num_partitions,
                    lower_bound=col_lower,
                    upper_bound=col_upper,
                    lazy_metrics=lazy_metrics,
                )
                all_results_dfs.append(results_df)

                # Build lazy context for on-demand metric computation
                if lazy_metrics:
                    lazy_contexts[col] = LazyMetricsContext(
                        source_df=df_sample,
                        column=col,
                        random_seed=self.random_seed,
                        row_count=row_count,
                        lower_bound=col_lower,
                        upper_bound=col_upper,
                        is_discrete=True,  # Discrete distributions
                    )

            # Union all results - handle both Spark and pandas DataFrames
            if self.spark is not None:
                # Spark: union DataFrames
                combined_df = reduce(DataFrame.union, all_results_dfs)
                combined_df = combined_df.cache()
                total_results = combined_df.count()
            else:
                # Non-Spark backend: concatenate pandas DataFrames
                import pandas as pd

                combined_df = pd.concat(all_results_dfs, ignore_index=True)
                total_results = len(combined_df)

            logger.info(
                f"Total results: {total_results} ({len(target_columns)} columns × ~{len(distributions)} distributions)"
            )

            # Pass lazy contexts to FitResults for on-demand metric computation
            return FitResults(combined_df, lazy_contexts=lazy_contexts if lazy_metrics else None)
        finally:
            if tracker is not None:
                tracker.stop()

    def _fit_single_column(
        self,
        df_sample: DataFrame,
        column: str,
        row_count: int,
        distributions: List[str],
        num_partitions: Optional[int],
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        lazy_metrics: bool = False,
    ) -> DataFrame:
        """Fit discrete distributions to a single column (internal method).

        Args:
            df_sample: Sampled DataFrame
            column: Column name
            row_count: Original row count
            distributions: List of distribution names to fit
            num_partitions: Number of Spark partitions
            lower_bound: Optional lower bound for truncated distribution
            upper_bound: Optional upper bound for truncated distribution
            lazy_metrics: If True, skip KS computation for performance (v1.5.0)

        Returns:
            Spark DataFrame with fit results for this column
        """
        # Create integer data sample for fitting
        sample_size = min(FITTING_SAMPLE_SIZE, row_count)
        fraction = min(sample_size / row_count, 1.0)
        # Use backend's sample_column which handles both Spark and pandas
        data_sample = self._backend.sample_column(df_sample, column, fraction=fraction, seed=self.random_seed).astype(
            int
        )
        data_sample = create_discrete_sample_data(data_sample, sample_size=FITTING_SAMPLE_SIZE)
        logger.info(f"  Data sample for '{column}': {len(data_sample)} values")

        # Compute discrete histogram (PMF)
        x_values, empirical_pmf = compute_discrete_histogram(data_sample)
        logger.info(f"  PMF for '{column}': {len(x_values)} unique values (range: {x_values.min()}-{x_values.max()})")

        # Compute data stats for provenance (once per column)
        data_stats = compute_data_stats(data_sample.astype(float))

        # Interleave slow distributions for better partition balance
        # (Currently no slow discrete distributions, but maintains consistency)
        # Lazy import to avoid circular dependency with core.py
        from spark_bestfit.core import _interleave_distributions

        distributions = _interleave_distributions(distributions)

        # Execute parallel fitting via backend (v2.0 abstraction)
        # Backend handles: broadcast, partitioning, UDF application, collection
        results = self._backend.parallel_fit(
            distributions=distributions,
            histogram=(x_values, empirical_pmf),
            data_sample=data_sample,
            fit_func=fit_single_discrete_distribution,
            column_name=column,
            data_stats=data_stats,
            num_partitions=num_partitions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lazy_metrics=lazy_metrics,
            is_discrete=True,
        )

        # Convert results to DataFrame
        if self.spark is not None:
            # Spark backend
            if results:
                results_df = self.spark.createDataFrame(results, schema=DISCRETE_FIT_RESULT_SCHEMA)
            else:
                results_df = self.spark.createDataFrame([], schema=DISCRETE_FIT_RESULT_SCHEMA)
        else:
            # Non-Spark backend: use pandas DataFrame
            import pandas as pd

            results_df = pd.DataFrame(results) if results else pd.DataFrame()

        num_results = len(results)
        logger.info(f"  Fit {num_results}/{len(distributions)} distributions for '{column}'")

        return results_df

    @staticmethod
    def _validate_inputs(
        df: DataFrame,
        column: str,
        max_distributions: Optional[int],
        sample_fraction: Optional[float],
    ) -> None:
        """Validate input parameters for discrete distribution fitting.

        Args:
            df: Spark DataFrame containing data
            column: Column name to validate
            max_distributions: Maximum distributions to fit (0 is invalid)
            sample_fraction: Sampling fraction (must be in (0, 1] if provided)

        Raises:
            ValueError: If max_distributions is 0, column not found,
                or sample_fraction out of range
            TypeError: If column is not numeric
        """
        if max_distributions == 0:
            raise ValueError("max_distributions cannot be 0")

        # Get columns list - handle Spark, pandas, and Ray Dataset
        if hasattr(df, "select_columns") and hasattr(df, "schema"):
            # Ray Dataset - use schema() method to get column names
            columns_list = df.schema().names
        elif hasattr(df, "schema") and hasattr(df.schema, "__iter__"):
            # Spark DataFrame
            columns_list = df.columns
        else:
            # pandas DataFrame (or similar)
            columns_list = list(df.columns)

        if column not in columns_list:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {columns_list}")

        # Handle type checking for Spark, pandas, and Ray Dataset
        if hasattr(df, "select_columns") and hasattr(df, "schema"):
            # Ray Dataset - check schema for numeric type
            schema = df.schema()
            col_idx = schema.names.index(column)
            col_type = schema.types[col_idx]
            # Ray DataType has string repr like 'double', 'int64', etc.
            type_str = str(col_type).lower()
            numeric_types = ("int", "float", "double", "decimal")
            if not any(t in type_str for t in numeric_types):
                raise TypeError(f"Column '{column}' must be numeric, got {col_type}")
        elif hasattr(df, "schema") and hasattr(df.schema, "__getitem__"):
            # Spark DataFrame
            col_type = df.schema[column].dataType
            if not isinstance(col_type, NumericType):
                raise TypeError(f"Column '{column}' must be numeric, got {col_type}")
        else:
            # pandas DataFrame (or similar)
            import pandas as pd

            if hasattr(df, "dtypes"):
                col_dtype = df[column].dtype
                if not pd.api.types.is_numeric_dtype(col_dtype):
                    raise TypeError(f"Column '{column}' must be numeric, got {col_dtype}")

        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    @staticmethod
    def _validate_bounds(
        lower_bound: Optional[Union[float, Dict[str, float]]],
        upper_bound: Optional[Union[float, Dict[str, float]]],
        target_columns: List[str],
    ) -> None:
        """Validate bounds parameters.

        Args:
            lower_bound: Scalar or dict of lower bounds
            upper_bound: Scalar or dict of upper bounds
            target_columns: List of columns being fitted

        Raises:
            ValueError: If bounds are invalid (lower >= upper, unknown columns in dict)
        """
        # Validate scalar bounds
        if isinstance(lower_bound, (int, float)) and isinstance(upper_bound, (int, float)):
            if lower_bound >= upper_bound:
                raise ValueError(f"lower_bound ({lower_bound}) must be less than upper_bound ({upper_bound})")
            return

        # Validate dict bounds - check for unknown columns
        if isinstance(lower_bound, dict):
            unknown = set(lower_bound.keys()) - set(target_columns)
            if unknown:
                raise ValueError(f"lower_bound contains unknown columns: {unknown}. Valid columns: {target_columns}")

        if isinstance(upper_bound, dict):
            unknown = set(upper_bound.keys()) - set(target_columns)
            if unknown:
                raise ValueError(f"upper_bound contains unknown columns: {unknown}. Valid columns: {target_columns}")

        # Validate that lower < upper for each column where both are specified
        lower_dict = lower_bound if isinstance(lower_bound, dict) else {}
        upper_dict = upper_bound if isinstance(upper_bound, dict) else {}

        for col in target_columns:
            col_lower = lower_dict.get(col) if isinstance(lower_bound, dict) else lower_bound
            col_upper = upper_dict.get(col) if isinstance(upper_bound, dict) else upper_bound
            if col_lower is not None and col_upper is not None:
                if col_lower >= col_upper:
                    raise ValueError(
                        f"lower_bound ({col_lower}) must be less than upper_bound ({col_upper}) for column '{col}'"
                    )

    @staticmethod
    def _resolve_bounds(
        df: DataFrame,
        target_columns: List[str],
        lower_bound: Optional[Union[float, Dict[str, float]]],
        upper_bound: Optional[Union[float, Dict[str, float]]],
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Resolve bounds to per-column dict, auto-detecting from data where needed.

        Args:
            df: DataFrame containing data
            target_columns: List of columns being fitted
            lower_bound: Scalar, dict, or None
            upper_bound: Scalar, dict, or None

        Returns:
            Dict mapping column name to (lower, upper) tuple
        """
        # Determine which columns need auto-detection
        lower_dict = lower_bound if isinstance(lower_bound, dict) else {}
        upper_dict = upper_bound if isinstance(upper_bound, dict) else {}

        cols_need_lower = [
            col for col in target_columns if not isinstance(lower_bound, (int, float)) and col not in lower_dict
        ]
        cols_need_upper = [
            col for col in target_columns if not isinstance(upper_bound, (int, float)) and col not in upper_dict
        ]

        # Build aggregation expressions for auto-detection
        agg_exprs = []
        for col in cols_need_lower:
            agg_exprs.append(F.min(col).alias(f"min_{col}"))
        for col in cols_need_upper:
            agg_exprs.append(F.max(col).alias(f"max_{col}"))

        # Execute single aggregation for all needed bounds
        auto_bounds: Dict[str, float] = {}
        if agg_exprs:
            bounds_row = df.agg(*agg_exprs).first()
            for col in cols_need_lower:
                auto_bounds[f"min_{col}"] = float(bounds_row[f"min_{col}"])
            for col in cols_need_upper:
                auto_bounds[f"max_{col}"] = float(bounds_row[f"max_{col}"])

        # Build final per-column bounds dict
        result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for col in target_columns:
            # Determine lower bound for this column
            if isinstance(lower_bound, (int, float)):
                col_lower = float(lower_bound)
            elif isinstance(lower_bound, dict) and col in lower_bound:
                col_lower = float(lower_bound[col])
            else:
                col_lower = auto_bounds.get(f"min_{col}")

            # Determine upper bound for this column
            if isinstance(upper_bound, (int, float)):
                col_upper = float(upper_bound)
            elif isinstance(upper_bound, dict) and col in upper_bound:
                col_upper = float(upper_bound[col])
            else:
                col_upper = auto_bounds.get(f"max_{col}")

            result[col] = (col_lower, col_upper)
            logger.info(f"Bounded fitting for '{col}': bounds=[{col_lower}, {col_upper}]")

        return result

    def _apply_sampling(
        self,
        df: DataFrame,
        row_count: int,
        enable_sampling: bool,
        sample_fraction: Optional[float],
        max_sample_size: int,
        sample_threshold: int,
    ) -> DataFrame:
        """Apply sampling to DataFrame if dataset exceeds threshold.

        Args:
            df: Spark DataFrame to sample
            row_count: Total row count of DataFrame
            enable_sampling: Whether sampling is enabled
            sample_fraction: Explicit fraction to sample (None = auto-determine)
            max_sample_size: Maximum rows when auto-determining fraction
            sample_threshold: Row count above which sampling is applied

        Returns:
            Original DataFrame if no sampling needed, otherwise sampled DataFrame
        """
        if not enable_sampling or row_count <= sample_threshold:
            return df

        if sample_fraction is not None:
            fraction = sample_fraction
        else:
            fraction = min(max_sample_size / row_count, 0.35)

        logger.info(f"Sampling {fraction * 100:.1f}% of data ({int(row_count * fraction)} rows)")
        # Handle both Spark and pandas DataFrames
        if hasattr(df, "sparkSession"):
            return df.sample(fraction=fraction, seed=self.random_seed)
        else:
            # pandas DataFrame
            return df.sample(frac=fraction, random_state=self.random_seed)

    def _calculate_partitions(self, distributions: List[str]) -> int:
        """Calculate optimal Spark partition count for distribution fitting.

        Uses distribution-aware weighting: slow distributions count 3x to ensure
        adequate parallelism when fitting computationally expensive distributions.

        Args:
            distributions: List of distribution names to fit

        Returns:
            Optimal partition count for the fitting operation
        """
        from spark_bestfit.distributions import DistributionRegistry

        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        slow_count = sum(1 for d in distributions if d in slow_set)
        # Slow distributions count 3x (1 base + 2 extra)
        effective_count = len(distributions) + slow_count * 2
        total_cores = self._backend.get_parallelism()
        return min(effective_count, total_cores * 2)

    def plot(
        self,
        result: DistributionFitResult,
        df: DataFrame,
        column: str,
        title: str = "",
        xlabel: str = "Value",
        ylabel: str = "Probability",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        show_histogram: bool = True,
        histogram_alpha: float = 0.7,
        pmf_linewidth: int = 2,
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        legend_fontsize: int = 10,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        save_format: str = "png",
    ):
        """Plot fitted discrete distribution against data histogram.

        Args:
            result: DistributionFitResult to plot
            df: DataFrame with data
            column: Column name
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            dpi: Dots per inch for saved figures
            show_histogram: Show data histogram
            histogram_alpha: Histogram transparency (0-1)
            pmf_linewidth: Line width for PMF curve
            title_fontsize: Title font size
            label_fontsize: Axis label font size
            legend_fontsize: Legend font size
            grid_alpha: Grid transparency (0-1)
            save_path: Path to save figure (optional)
            save_format: Save format (png, pdf, svg)

        Returns:
            Tuple of (figure, axis) from matplotlib
        """
        from spark_bestfit.plotting import plot_discrete_distribution

        # Get data sample
        # Handle Spark DataFrame, Ray Dataset, and pandas DataFrame
        if hasattr(df, "sparkSession"):
            row_count = df.count()
        elif hasattr(df, "select_columns") and hasattr(df, "count"):
            # Ray Dataset - use count() method
            row_count = df.count()
        else:
            row_count = len(df)
        fraction = min(10000 / row_count, 1.0)
        data = self._backend.sample_column(df, column, fraction=fraction, seed=self.random_seed).astype(int)

        return plot_discrete_distribution(
            result=result,
            data=data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            show_histogram=show_histogram,
            histogram_alpha=histogram_alpha,
            pmf_linewidth=pmf_linewidth,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            grid_alpha=grid_alpha,
            save_path=save_path,
            save_format=save_format,
        )
