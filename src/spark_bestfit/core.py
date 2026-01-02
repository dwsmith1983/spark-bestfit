"""Core distribution fitting engine for Spark."""

import logging
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import NumericType

from spark_bestfit.discrete_fitting import (
    compute_discrete_histogram,
    create_discrete_fitting_udf,
    create_discrete_sample_data,
)
from spark_bestfit.distributions import DiscreteDistributionRegistry, DistributionRegistry
from spark_bestfit.fitting import FITTING_SAMPLE_SIZE, compute_data_summary, create_fitting_udf
from spark_bestfit.histogram import HistogramComputer
from spark_bestfit.results import DistributionFitResult, FitResults, LazyMetricsContext
from spark_bestfit.utils import get_spark_session

logger = logging.getLogger(__name__)

# Re-export for convenience
DEFAULT_EXCLUDED_DISTRIBUTIONS: Tuple[str, ...] = tuple(DistributionRegistry.DEFAULT_EXCLUSIONS)


def _interleave_distributions(distributions: List[str]) -> List[str]:
    """Interleave slow distributions among fast ones for better partition balance.

    When Spark repartitions the distribution DataFrame, this ordering ensures
    slow distributions are spread across partitions rather than clustered,
    reducing straggler effects.

    Args:
        distributions: List of distribution names to fit

    Returns:
        Reordered list with slow distributions spread evenly among fast ones

    Example:
        >>> dists = ["norm", "expon", "gamma", "burr", "t", "johnsonsb"]
        >>> # burr, t, johnsonsb are slow - they get spread out:
        >>> _interleave_distributions(dists)
        ['burr', 'norm', 'expon', 't', 'gamma', 'johnsonsb']
    """
    from spark_bestfit.distributions import DistributionRegistry

    slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
    slow = [d for d in distributions if d in slow_set]
    fast = [d for d in distributions if d not in slow_set]

    if not slow or not fast:
        return distributions

    # Insert slow distributions at even intervals among fast ones
    result: List[str] = []
    slow_interval = max(1, len(fast) // len(slow))
    slow_idx = 0
    for i, d in enumerate(fast):
        if slow_idx < len(slow) and i % slow_interval == 0:
            result.append(slow[slow_idx])
            slow_idx += 1
        result.append(d)
    result.extend(slow[slow_idx:])  # Remaining slow ones at end
    return result


class DistributionFitter:
    """Modern Spark distribution fitting engine.

    Efficiently fits ~100 scipy.stats distributions to data using Spark's
    parallel processing capabilities. Uses broadcast variables and Pandas UDFs
    to avoid data collection and minimize serialization overhead.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from spark_bestfit import DistributionFitter
        >>>
        >>> # Create your own SparkSession
        >>> spark = SparkSession.builder.appName("my-app").getOrCreate()
        >>> df = spark.createDataFrame([(float(x),) for x in data], ['value'])
        >>>
        >>> # Simple usage
        >>> fitter = DistributionFitter(spark)
        >>> results = fitter.fit(df, column='value')
        >>> best = results.best(n=1)[0]
        >>> print(f"Best: {best.distribution} with SSE={best.sse}")
        >>>
        >>> # With custom parameters
        >>> fitter = DistributionFitter(spark, random_seed=123)
        >>> results = fitter.fit(df, 'value', bins=100, support_at_zero=True)
        >>>
        >>> # Plot the best fit
        >>> fitter.plot(best, df, 'value', title='Best Fit')
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        excluded_distributions: Optional[Tuple[str, ...]] = None,
        random_seed: int = 42,
    ):
        """Initialize DistributionFitter.

        Args:
            spark: SparkSession. If None, uses the active session.
            excluded_distributions: Distributions to exclude from fitting.
                Defaults to DEFAULT_EXCLUDED_DISTRIBUTIONS (slow distributions).
            random_seed: Random seed for reproducible sampling.

        Raises:
            RuntimeError: If no SparkSession provided and no active session exists
        """
        self.spark: SparkSession = get_spark_session(spark)
        self.excluded_distributions = (
            excluded_distributions if excluded_distributions is not None else DEFAULT_EXCLUDED_DISTRIBUTIONS
        )
        self.random_seed = random_seed
        self._registry = DistributionRegistry()
        self._histogram_computer = HistogramComputer()

    def fit(
        self,
        df: DataFrame,
        column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        bins: Union[int, Tuple[float, ...]] = 50,
        use_rice_rule: bool = True,
        support_at_zero: bool = False,
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
        """Fit distributions to data column(s).

        Args:
            df: Spark DataFrame containing data
            column: Name of single column to fit distributions to
            columns: List of column names for multi-column fitting
            bins: Number of histogram bins or tuple of bin edges
            use_rice_rule: Use Rice rule to auto-determine bin count
            support_at_zero: Only fit non-negative distributions
            max_distributions: Limit number of distributions (for testing)
            enable_sampling: Enable sampling for large datasets
            sample_fraction: Fraction to sample (None = auto-determine)
            max_sample_size: Maximum rows to sample when auto-determining
            sample_threshold: Row count above which sampling is applied
            num_partitions: Spark partitions (None = auto-determine)
            progress_callback: Optional callback for progress updates.
                Called with (completed_tasks, total_tasks, percent_complete).
                Callback is invoked from background thread - ensure thread-safety.
            bounded: If True, fit truncated distributions (v1.4.0).
                When enabled, distributions are truncated to [lower_bound, upper_bound]
                using scipy.stats.truncate(). Requires scipy >= 1.14.0.
            lower_bound: Lower bound for truncated distribution fitting.
                Can be a float (applied to all columns) or a dict mapping
                column names to bounds (v1.5.0). If None and bounded=True,
                auto-detects from each column's minimum.
            upper_bound: Upper bound for truncated distribution fitting.
                Can be a float (applied to all columns) or a dict mapping
                column names to bounds (v1.5.0). If None and bounded=True,
                auto-detects from each column's maximum.
            lazy_metrics: If True, defer computation of expensive KS/AD metrics
                until accessed (v1.5.0). Improves fitting performance when only
                using AIC/BIC/SSE for model selection. Default False for
                backward compatibility.
            prefilter: Pre-filter distributions based on data characteristics (v1.6.0).
                Skips distributions that are mathematically incompatible with the data,
                reducing fitting time by 30-70% for non-normal data.
                - False (default): No pre-filtering, fit all distributions
                - True: Safe mode - filters by support bounds and skewness sign
                - 'aggressive': Also filters by kurtosis (may skip valid distributions)
                Pre-filtering uses scipy's distribution support bounds (dist.a, dist.b)
                and sample moments. Filtered distributions are logged for transparency.

        Returns:
            FitResults object with fitted distributions

        Raises:
            ValueError: If column not found, DataFrame empty, or invalid params
            TypeError: If column is not numeric

        Example:
            >>> # Single column
            >>> results = fitter.fit(df, column='value')
            >>> results = fitter.fit(df, 'value', bins=100, support_at_zero=True)
            >>>
            >>> # Multi-column
            >>> results = fitter.fit(df, columns=['col1', 'col2', 'col3'])
            >>> best_col1 = results.for_column('col1').best(n=1)[0]
            >>> best_per_col = results.best_per_column(n=1)
            >>>
            >>> # Bounded fitting (v1.4.0)
            >>> results = fitter.fit(df, 'value', bounded=True)  # Auto-detect bounds
            >>> results = fitter.fit(df, 'value', bounded=True, lower_bound=0, upper_bound=100)
            >>>
            >>> # Per-column bounds (v1.5.0)
            >>> results = fitter.fit(
            ...     df, columns=['col1', 'col2'],
            ...     bounded=True,
            ...     lower_bound={'col1': 0, 'col2': -10},
            ...     upper_bound={'col1': 100, 'col2': 50}
            ... )
            >>>
            >>> # Lazy metrics for faster fitting when only using AIC/BIC (v1.5.0)
            >>> results = fitter.fit(df, 'value', lazy_metrics=True)
            >>> best_aic = results.best(n=1, metric='aic')[0]  # Fast, no KS/AD computed
            >>>
            >>> # Pre-filter distributions for faster fitting (v1.6.0)
            >>> results = fitter.fit(df, 'value', prefilter=True)  # Safe mode
            >>> results = fitter.fit(df, 'value', prefilter='aggressive')  # Maximum speed
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
            self._validate_inputs(df, col, max_distributions, bins, sample_fraction)

        # Validate bounds - handle both scalar and dict forms
        self._validate_bounds(lower_bound, upper_bound, target_columns)

        # Get row count (single operation for all columns)
        row_count = df.count()
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
            support_at_zero=support_at_zero,
            additional_exclusions=list(self.excluded_distributions),
        )
        if max_distributions is not None and max_distributions > 0:
            distributions = distributions[:max_distributions]

        # Start progress tracking if callback provided
        tracker = None
        if progress_callback is not None:
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
                logger.info(f"Fitting column '{col}'...")
                results_df = self._fit_single_column(
                    df_sample=df_sample,
                    column=col,
                    row_count=row_count,
                    bins=bins,
                    use_rice_rule=use_rice_rule,
                    distributions=distributions,
                    num_partitions=num_partitions,
                    lower_bound=col_lower,
                    upper_bound=col_upper,
                    lazy_metrics=lazy_metrics,
                    prefilter=prefilter,
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
                        is_discrete=False,
                    )

            # Union all results using reduce for cleaner query plan
            combined_df = reduce(DataFrame.union, all_results_dfs)

            combined_df = combined_df.cache()
            total_results = combined_df.count()
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
        bins: Union[int, Tuple[float, ...]],
        use_rice_rule: bool,
        distributions: List[str],
        num_partitions: Optional[int],
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        lazy_metrics: bool = False,
        prefilter: Union[bool, str] = False,
    ) -> DataFrame:
        """Fit distributions to a single column (internal method).

        Args:
            df_sample: Sampled DataFrame
            column: Column name
            row_count: Original row count (for histogram computation)
            bins: Number of histogram bins
            use_rice_rule: Use Rice rule for bin count
            distributions: List of distribution names to fit
            num_partitions: Number of Spark partitions
            lower_bound: Lower bound for truncated distribution fitting (v1.4.0)
            upper_bound: Upper bound for truncated distribution fitting (v1.4.0)
            lazy_metrics: If True, skip KS/AD computation for performance (v1.5.0)
            prefilter: Pre-filter mode (False, True, or 'aggressive') (v1.6.0)

        Returns:
            Spark DataFrame with fit results for this column
        """
        # Compute histogram (returns bin edges for CDF-based fitting)
        y_hist, bin_edges = self._histogram_computer.compute_histogram(
            df_sample, column, bins=bins, use_rice_rule=use_rice_rule, approx_count=row_count
        )
        logger.info(f"  Histogram for '{column}': {len(bin_edges) - 1} bins")

        # Broadcast histogram (y_hist densities + bin_edges for CDF computation)
        histogram_bc = self.spark.sparkContext.broadcast((y_hist, bin_edges))

        # Create fitting sample
        data_sample = self._create_fitting_sample(df_sample, column, row_count)

        # Apply pre-filtering if enabled (v1.6.0)
        original_distributions = distributions
        original_count = len(distributions)
        if prefilter:
            distributions, filtered = self._prefilter_distributions(distributions, data_sample, prefilter)
            if filtered:
                filtered_names = [f[0] for f in filtered]
                logger.info(
                    f"  Pre-filter: skipped {len(filtered)}/{original_count} distributions "
                    f"({', '.join(filtered_names[:5])}{'...' if len(filtered_names) > 5 else ''})"
                )
            # Safeguard: if all distributions filtered, fall back to fitting all
            if not distributions:
                logger.warning(
                    f"  Pre-filter removed all {original_count} distributions; "
                    f"falling back to fitting all distributions"
                )
                distributions = original_distributions

        data_sample_bc = self.spark.sparkContext.broadcast(data_sample)

        # Compute data summary for provenance (once per column)
        data_summary = compute_data_summary(data_sample)

        try:
            # Interleave slow distributions for better partition balance
            distributions = _interleave_distributions(distributions)

            # Create DataFrame of distributions
            dist_df = self.spark.createDataFrame([(dist,) for dist in distributions], ["distribution_name"])

            # Determine partitioning (weighted for slow distributions)
            n_partitions = num_partitions or self._calculate_partitions(distributions)
            dist_df = dist_df.repartition(n_partitions)

            # Apply fitting UDF
            fitting_udf = create_fitting_udf(
                histogram_bc,
                data_sample_bc,
                column_name=column,
                data_summary=data_summary,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lazy_metrics=lazy_metrics,
            )
            results_df = dist_df.select(fitting_udf(F.col("distribution_name")).alias("result")).select("result.*")

            # Filter failed fits
            results_df = results_df.filter(F.col("sse") < float(np.inf))

            num_results = results_df.count()
            logger.info(f"  Fit {num_results}/{len(distributions)} distributions for '{column}'")

            return results_df

        finally:
            # Release broadcast variables from executor memory
            # Note: Using unpersist() rather than destroy() because Spark's lazy
            # evaluation means the broadcast may still be referenced by pending
            # DataFrame operations. unpersist() allows completion before cleanup.
            histogram_bc.unpersist()
            data_sample_bc.unpersist()

    def plot(
        self,
        result: DistributionFitResult,
        df: DataFrame,
        column: str,
        bins: Union[int, Tuple[float, ...]] = 50,
        use_rice_rule: bool = True,
        title: str = "",
        xlabel: str = "Value",
        ylabel: str = "Density",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        show_histogram: bool = True,
        histogram_alpha: float = 0.5,
        pdf_linewidth: int = 2,
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        legend_fontsize: int = 10,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        save_format: str = "png",
    ):
        """Plot fitted distribution against data histogram.

        Args:
            result: DistributionFitResult to plot
            df: DataFrame with data
            column: Column name
            bins: Number of histogram bins
            use_rice_rule: Use Rice rule for bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            dpi: Dots per inch for saved figures
            show_histogram: Show data histogram
            histogram_alpha: Histogram transparency (0-1)
            pdf_linewidth: Line width for PDF curve
            title_fontsize: Title font size
            label_fontsize: Axis label font size
            legend_fontsize: Legend font size
            grid_alpha: Grid transparency (0-1)
            save_path: Path to save figure (optional)
            save_format: Save format (png, pdf, svg)

        Returns:
            Tuple of (figure, axis) from matplotlib

        Example:
            >>> fitter.plot(best, df, 'value', title='Best Fit')
            >>> fitter.plot(best, df, 'value', figsize=(16, 10), dpi=300)
        """
        from spark_bestfit.plotting import plot_distribution

        # Compute histogram for plotting (bin edges -> centers for display)
        y_hist, bin_edges = self._histogram_computer.compute_histogram(
            df, column, bins=bins, use_rice_rule=use_rice_rule
        )
        x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        return plot_distribution(
            result=result,
            y_hist=y_hist,
            x_hist=x_centers,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            show_histogram=show_histogram,
            histogram_alpha=histogram_alpha,
            pdf_linewidth=pdf_linewidth,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            grid_alpha=grid_alpha,
            save_path=save_path,
            save_format=save_format,
        )

    def plot_comparison(
        self,
        results: List[DistributionFitResult],
        df: DataFrame,
        column: str,
        bins: Union[int, Tuple[float, ...]] = 50,
        use_rice_rule: bool = True,
        title: str = "Distribution Comparison",
        xlabel: str = "Value",
        ylabel: str = "Density",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        show_histogram: bool = True,
        histogram_alpha: float = 0.5,
        pdf_linewidth: int = 2,
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        legend_fontsize: int = 10,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        save_format: str = "png",
    ):
        """Plot multiple distributions for comparison.

        Args:
            results: List of DistributionFitResult objects
            df: DataFrame with data
            column: Column name
            bins: Number of histogram bins
            use_rice_rule: Use Rice rule for bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            dpi: Dots per inch
            show_histogram: Show data histogram
            histogram_alpha: Histogram transparency
            pdf_linewidth: PDF line width
            title_fontsize: Title font size
            label_fontsize: Label font size
            legend_fontsize: Legend font size
            grid_alpha: Grid transparency
            save_path: Path to save figure
            save_format: Save format

        Returns:
            Tuple of (figure, axis)

        Example:
            >>> top_3 = results.best(n=3)
            >>> fitter.plot_comparison(top_3, df, 'value')
        """
        from spark_bestfit.plotting import plot_comparison

        # Compute histogram for plotting (bin edges -> centers for display)
        y_hist, bin_edges = self._histogram_computer.compute_histogram(
            df, column, bins=bins, use_rice_rule=use_rice_rule
        )
        x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        return plot_comparison(
            results=results,
            y_hist=y_hist,
            x_hist=x_centers,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            show_histogram=show_histogram,
            histogram_alpha=histogram_alpha,
            pdf_linewidth=pdf_linewidth,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            legend_fontsize=legend_fontsize,
            grid_alpha=grid_alpha,
            save_path=save_path,
            save_format=save_format,
        )

    @staticmethod
    def _validate_inputs(
        df: DataFrame,
        column: str,
        max_distributions: Optional[int],
        bins: Union[int, Tuple[float, ...]],
        sample_fraction: Optional[float],
    ) -> None:
        """Validate input parameters for distribution fitting.

        Args:
            df: Spark DataFrame containing data
            column: Column name to validate
            max_distributions: Maximum distributions to fit (0 is invalid)
            bins: Number of histogram bins (must be positive if int)
            sample_fraction: Sampling fraction (must be in (0, 1] if provided)

        Raises:
            ValueError: If max_distributions is 0, column not found, bins invalid,
                or sample_fraction out of range
            TypeError: If column is not numeric
        """
        if max_distributions == 0:
            raise ValueError("max_distributions cannot be 0")

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {df.columns}")

        col_type = df.schema[column].dataType
        if not isinstance(col_type, NumericType):
            raise TypeError(f"Column '{column}' must be numeric, got {col_type}")

        if isinstance(bins, int) and bins <= 0:
            raise ValueError(f"bins must be positive, got {bins}")

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
        return df.sample(fraction=fraction, seed=self.random_seed)

    def _create_fitting_sample(self, df: DataFrame, column: str, row_count: int) -> np.ndarray:
        """Create numpy sample array for scipy distribution fitting.

        Samples up to FITTING_SAMPLE_SIZE rows from the DataFrame for use in
        scipy's distribution fitting functions.

        Args:
            df: Spark DataFrame containing data
            column: Column name to sample
            row_count: Total row count (used to calculate sampling fraction)

        Returns:
            Numpy array of sampled values for distribution fitting
        """
        sample_size = min(FITTING_SAMPLE_SIZE, row_count)
        fraction = min(sample_size / row_count, 1.0)
        sample_df = df.select(column).sample(fraction=fraction, seed=self.random_seed)
        return sample_df.toPandas()[column].values

    def _calculate_partitions(self, distributions: List[str]) -> int:
        """Calculate optimal Spark partition count for distribution fitting.

        Uses distribution-aware weighting: slow distributions count as 3x
        for partition calculation to reduce straggler effects when slow
        distributions cluster in the same partition.

        Args:
            distributions: List of distribution names to fit

        Returns:
            Optimal partition count for the fitting operation
        """
        from spark_bestfit.distributions import DistributionRegistry

        slow_set = DistributionRegistry.SLOW_DISTRIBUTIONS
        slow_count = sum(1 for d in distributions if d in slow_set)
        # Slow distributions count as 3x (add 2 extra for each slow one)
        effective_count = len(distributions) + slow_count * 2
        total_cores = self.spark.sparkContext.defaultParallelism
        return min(effective_count, total_cores * 2)

    def _prefilter_distributions(
        self,
        distributions: List[str],
        data_sample: np.ndarray,
        mode: Union[bool, str],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Pre-filter distributions based on data characteristics.

        Uses a layered approach based on SHAPE properties (not location/scale):
        1. Skewness sign (~95% reliable): Skip positive-skew-only distributions
           for clearly left-skewed data (skewness < -1.0)
        2. Kurtosis (aggressive mode only, ~80% reliable): Skip low-kurtosis
           distributions for very heavy-tailed data

        Note: We do NOT filter by support bounds (dist.a/dist.b) because scipy's
        fitting process uses loc/scale parameters that can shift any distribution
        to cover any data range. Shape properties (skewness, kurtosis) are
        intrinsic and cannot be changed by loc/scale.

        Args:
            distributions: List of distribution names to filter
            data_sample: Numpy array of sample data
            mode: True for safe mode, 'aggressive' for additional kurtosis filter

        Returns:
            Tuple of (compatible_distributions, filtered_with_reasons)
        """
        # Early return if filtering is disabled
        if not mode:
            return distributions.copy(), []

        from scipy.stats import kurtosis, skew

        data_skew = float(skew(data_sample))
        data_kurt = float(kurtosis(data_sample))  # Excess kurtosis

        compatible = []
        filtered = []

        # Distributions that can only have positive skewness (mathematical constraint)
        positive_skew_only = {
            "expon",
            "gamma",
            "lognorm",
            "chi2",
            "weibull_min",
            "pareto",
            "rayleigh",
            "invgamma",
            "exponweib",
            "genpareto",
            "invweibull",
            "fisk",
            "burr",
            "burr12",
            "loggamma",
            "invgauss",
            "genextreme",  # When shape > 0
            "gompertz",
            "halfnorm",
            "halfcauchy",
            "halflogistic",
            "halfgennorm",
            "rice",
            "nakagami",
            "wald",
            "gengamma",
            "powerlognorm",
        }

        for dist_name in distributions:
            try:
                # We intentionally do NOT check support bounds (dist.a/dist.b)
                # because scipy.fit() uses loc/scale parameters that can shift
                # any distribution to cover any data range.

                # Layer 1: Skewness sign check (~95% reliable)
                # Only filter if data is CLEARLY left-skewed (threshold = -1.0)
                # These distributions are intrinsically right-skewed regardless of loc/scale
                if data_skew < -1.0 and dist_name in positive_skew_only:
                    filtered.append((dist_name, "positive-skew only"))
                    continue

                # Layer 2: Kurtosis check (aggressive mode only, ~80% reliable)
                if mode == "aggressive" and data_kurt > 10:
                    # Very heavy-tailed data - skip uniform which has kurtosis = -1.2
                    # Uniform's kurtosis is intrinsic and cannot be changed by loc/scale
                    if dist_name == "uniform":
                        filtered.append((dist_name, "low kurtosis distribution"))
                        continue

                compatible.append(dist_name)

            except AttributeError:
                # Unknown distribution - keep it (conservative)
                compatible.append(dist_name)

        return compatible, filtered

    def plot_qq(
        self,
        result: DistributionFitResult,
        df: DataFrame,
        column: str,
        max_points: int = 1000,
        title: str = "",
        xlabel: str = "Theoretical Quantiles",
        ylabel: str = "Sample Quantiles",
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
        marker: str = "o",
        marker_size: int = 30,
        marker_alpha: float = 0.6,
        marker_color: str = "steelblue",
        line_color: str = "red",
        line_style: str = "--",
        line_width: float = 1.5,
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        save_format: str = "png",
    ):
        """Create a Q-Q plot to assess goodness-of-fit.

        A Q-Q (quantile-quantile) plot compares sample quantiles against
        theoretical quantiles from the fitted distribution. Points falling
        close to the reference line indicate a good fit.

        Args:
            result: DistributionFitResult to plot
            df: DataFrame with data
            column: Column name
            max_points: Maximum data points to sample for plotting
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            dpi: Dots per inch for saved figures
            marker: Marker style for data points
            marker_size: Size of markers
            marker_alpha: Marker transparency (0-1)
            marker_color: Color of markers
            line_color: Color of reference line
            line_style: Style of reference line
            line_width: Width of reference line
            title_fontsize: Title font size
            label_fontsize: Axis label font size
            grid_alpha: Grid transparency (0-1)
            save_path: Path to save figure (optional)
            save_format: Save format (png, pdf, svg)

        Returns:
            Tuple of (figure, axis) from matplotlib

        Example:
            >>> best = results.best(n=1)[0]
            >>> fitter.plot_qq(best, df, 'value', title='Q-Q Plot')
        """
        from spark_bestfit.plotting import plot_qq

        # Sample data for Q-Q plot using sample() instead of orderBy(rand())
        # sample() operates per-partition without shuffle, much faster for large datasets
        row_count = df.count()
        fraction = min(max_points * 3 / row_count, 1.0) if row_count > 0 else 1.0
        sample_df = df.select(column).sample(fraction=fraction, seed=self.random_seed).limit(max_points)
        data = sample_df.toPandas()[column].values

        return plot_qq(
            result=result,
            data=data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            marker=marker,
            marker_size=marker_size,
            marker_alpha=marker_alpha,
            marker_color=marker_color,
            line_color=line_color,
            line_style=line_style,
            line_width=line_width,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            grid_alpha=grid_alpha,
            save_path=save_path,
            save_format=save_format,
        )

    def plot_pp(
        self,
        result: DistributionFitResult,
        df: DataFrame,
        column: str,
        max_points: int = 1000,
        title: str = "",
        xlabel: str = "Theoretical Probabilities",
        ylabel: str = "Sample Probabilities",
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
        marker: str = "o",
        marker_size: int = 30,
        marker_alpha: float = 0.6,
        marker_color: str = "steelblue",
        line_color: str = "red",
        line_style: str = "--",
        line_width: float = 1.5,
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        save_format: str = "png",
    ):
        """
        Create a P-P plot to assess goodness-of-fit.

        A P-P (probability-probability) plot compares the empirical CDF of the
        sample data against the theoretical CDF of the fitted distribution.
        Points falling close to the reference line indicate a good fit,
        particularly in the center of the distribution.

        Args:
            result: DistributionFitResult to plot
            df: DataFrame with data
            column: Column name
            max_points: Maximum data points to sample for plotting
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
            dpi: Dots per inch for saved figures
            marker: Marker style for data points
            marker_size: Size of markers
            marker_alpha: Marker transparency (0-1)
            marker_color: Color of markers
            line_color: Color of reference line
            line_style: Style of reference line
            line_width: Width of reference line
            title_fontsize: Title font size
            label_fontsize: Axis label font size
            grid_alpha: Grid transparency (0-1)
            save_path: Path to save figure (optional)
            save_format: Save format (png, pdf, svg)

        Returns:
            Tuple of (figure, axis) from matplotlib

        Example:
            >>> best = results.best(n=1)[0]
            >>> fitter.plot_pp(best, df, 'value', title='P-P Plot')
        """
        from spark_bestfit.plotting import plot_pp

        # Sample data for P-P plot using sample() instead of orderBy(rand())
        # sample() operates per-partition without shuffle, much faster for large datasets
        row_count = df.count()
        fraction = min(max_points * 3 / row_count, 1.0) if row_count > 0 else 1.0
        sample_df = df.select(column).sample(fraction=fraction, seed=self.random_seed).limit(max_points)
        data = sample_df.toPandas()[column].values

        return plot_pp(
            result=result,
            data=data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            marker=marker,
            marker_size=marker_size,
            marker_alpha=marker_alpha,
            marker_color=marker_color,
            line_color=line_color,
            line_style=line_style,
            line_width=line_width,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            grid_alpha=grid_alpha,
            save_path=save_path,
            save_format=save_format,
        )


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
    ):
        """Initialize DiscreteDistributionFitter.

        Args:
            spark: SparkSession. If None, uses the active session.
            excluded_distributions: Distributions to exclude from fitting.
                Defaults to DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS.
            random_seed: Random seed for reproducible sampling.

        Raises:
            RuntimeError: If no SparkSession provided and no active session exists
        """
        self.spark: SparkSession = get_spark_session(spark)
        self.excluded_distributions = (
            excluded_distributions if excluded_distributions is not None else DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS
        )
        self.random_seed = random_seed
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
        row_count = df.count()
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

        # Start progress tracking if callback provided
        tracker = None
        if progress_callback is not None:
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

            # Union all results using reduce for cleaner query plan
            combined_df = reduce(DataFrame.union, all_results_dfs)

            combined_df = combined_df.cache()
            total_results = combined_df.count()
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
        sample_df = df_sample.select(column).sample(fraction=fraction, seed=self.random_seed)
        data_sample = sample_df.toPandas()[column].values.astype(int)
        data_sample = create_discrete_sample_data(data_sample, sample_size=FITTING_SAMPLE_SIZE)
        logger.info(f"  Data sample for '{column}': {len(data_sample)} values")

        # Compute discrete histogram (PMF)
        x_values, empirical_pmf = compute_discrete_histogram(data_sample)
        logger.info(f"  PMF for '{column}': {len(x_values)} unique values (range: {x_values.min()}-{x_values.max()})")

        # Broadcast histogram and data
        histogram_bc = self.spark.sparkContext.broadcast((x_values, empirical_pmf))
        data_sample_bc = self.spark.sparkContext.broadcast(data_sample)

        # Compute data summary for provenance (once per column)
        data_summary = compute_data_summary(data_sample.astype(float))

        try:
            # Interleave slow distributions for better partition balance
            # (Currently no slow discrete distributions, but maintains consistency)
            distributions = _interleave_distributions(distributions)

            # Create DataFrame of distributions
            dist_df = self.spark.createDataFrame([(dist,) for dist in distributions], ["distribution_name"])

            # Determine partitioning
            n_partitions = num_partitions or self._calculate_partitions(distributions)
            dist_df = dist_df.repartition(n_partitions)

            # Apply discrete fitting UDF
            fitting_udf = create_discrete_fitting_udf(
                histogram_bc,
                data_sample_bc,
                column_name=column,
                data_summary=data_summary,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lazy_metrics=lazy_metrics,
            )
            results_df = dist_df.select(fitting_udf(F.col("distribution_name")).alias("result")).select("result.*")

            # Filter failed fits
            results_df = results_df.filter(F.col("sse") < float(np.inf))

            num_results = results_df.count()
            logger.info(f"  Fit {num_results}/{len(distributions)} distributions for '{column}'")

            return results_df

        finally:
            # Release broadcast variables (see note in _fit_column for why unpersist)
            histogram_bc.unpersist()
            data_sample_bc.unpersist()

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

        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {df.columns}")

        col_type = df.schema[column].dataType
        if not isinstance(col_type, NumericType):
            raise TypeError(f"Column '{column}' must be numeric, got {col_type}")

        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    def _validate_bounds(
        self,
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

    def _resolve_bounds(
        self,
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
        return df.sample(fraction=fraction, seed=self.random_seed)

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
        total_cores = self.spark.sparkContext.defaultParallelism
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
        sample_df = df.select(column).sample(fraction=min(10000 / df.count(), 1.0), seed=self.random_seed)
        data = sample_df.toPandas()[column].values.astype(int)

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
