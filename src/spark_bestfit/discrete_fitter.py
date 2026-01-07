"""Discrete distribution fitting engine for Spark."""

import logging
from functools import reduce
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

# PySpark is optional - only import if available
try:
    from pyspark.sql import DataFrame, SparkSession

    _PYSPARK_AVAILABLE = True
except ImportError:
    DataFrame = None  # type: ignore[assignment,misc]
    SparkSession = None  # type: ignore[assignment,misc]
    _PYSPARK_AVAILABLE = False

from spark_bestfit.base_fitter import BaseFitter
from spark_bestfit.config import FitterConfig
from spark_bestfit.discrete_fitting import (
    DISCRETE_FIT_RESULT_SCHEMA,
    compute_discrete_histogram,
    create_discrete_sample_data,
    fit_single_discrete_distribution,
)
from spark_bestfit.distributions import DiscreteDistributionRegistry
from spark_bestfit.fitting import FITTING_SAMPLE_SIZE, compute_data_stats
from spark_bestfit.results import DistributionFitResult, FitResults, FitResultsType, LazyMetricsContext

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend

logger = logging.getLogger(__name__)

# Re-export for convenience
DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS: Tuple[str, ...] = tuple(DiscreteDistributionRegistry.DEFAULT_EXCLUSIONS)


class DiscreteDistributionFitter(BaseFitter):
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

    # Class attributes for BaseFitter
    _registry_class = DiscreteDistributionRegistry
    _default_exclusions = DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS

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
        super().__init__(
            spark=spark,
            excluded_distributions=excluded_distributions,
            random_seed=random_seed,
            backend=backend,
        )

    def fit(
        self,
        df: DataFrame,
        column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        config: Optional[FitterConfig] = None,
        *,
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
    ) -> FitResultsType:
        """Fit discrete distributions to integer data column(s).

        Args:
            df: Spark DataFrame containing integer count data
            column: Name of single column to fit distributions to
            columns: List of column names for multi-column fitting
            config: FitterConfig object (v2.2.0). Provides a cleaner way to
                configure fitting with many parameters. If provided, individual
                parameters below are ignored (except progress_callback which
                can override the config's callback). Note: bins, use_rice_rule,
                support_at_zero, and prefilter in config are ignored for
                discrete fitting.
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
            >>> # Using FitterConfig (v2.2.0)
            >>> from spark_bestfit import FitterConfigBuilder
            >>> config = (FitterConfigBuilder()
            ...     .with_bounds(lower=0, upper=100)
            ...     .with_sampling(fraction=0.1)
            ...     .build())
            >>> results = fitter.fit(df, column='counts', config=config)
            >>>
            >>> # Single column (backward compatible)
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
            >>> # Lazy metrics for faster fitting when only using AIC/BIC (v1.5.0)
            >>> results = fitter.fit(df, 'counts', lazy_metrics=True)
            >>> best_aic = results.best(n=1, metric='aic')[0]  # Fast, no KS computed
        """
        # Resolve config: explicit config takes precedence over individual parameters
        if config is not None:
            # Use config values, but allow progress_callback override
            cfg = config
            if progress_callback is not None:
                cfg = config.with_progress_callback(progress_callback)
        else:
            # Create config from individual parameters (backward compatibility)
            cfg = FitterConfig(
                max_distributions=max_distributions,
                prefilter=prefilter,
                enable_sampling=enable_sampling,
                sample_fraction=sample_fraction,
                max_sample_size=max_sample_size,
                sample_threshold=sample_threshold,
                bounded=bounded,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                num_partitions=num_partitions,
                lazy_metrics=lazy_metrics,
                progress_callback=progress_callback,
            )

        # Normalize column/columns to list
        target_columns = self._normalize_columns(column, columns)

        # Input validation for all columns
        for col in target_columns:
            self._validate_inputs(df, col, cfg.max_distributions, cfg.sample_fraction)

        # Warn if prefilter is enabled (not yet supported for discrete)
        if cfg.prefilter:
            logger.warning("prefilter is not yet supported for discrete distributions; ignoring")

        # Validate bounds - handle both scalar and dict forms
        self._validate_bounds(cfg.lower_bound, cfg.upper_bound, target_columns)

        # Get row count (single operation for all columns)
        row_count = self._get_row_count(df)
        if row_count == 0:
            raise ValueError("DataFrame is empty")
        logger.info(f"Row count: {row_count}")

        # Build per-column bounds dict: {col: (lower, upper)}
        column_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if cfg.bounded:
            column_bounds = self._resolve_bounds(df, target_columns, cfg.lower_bound, cfg.upper_bound)

        # Sample if needed (single operation for all columns)
        df_sample = self._apply_sampling(
            df,
            row_count,
            cfg.enable_sampling,
            cfg.sample_fraction,
            cfg.max_sample_size,
            cfg.sample_threshold,
        )

        # Get distributions to fit (same for all columns)
        distributions = self._registry.get_distributions(
            additional_exclusions=list(self.excluded_distributions),
        )
        if cfg.max_distributions is not None and cfg.max_distributions > 0:
            distributions = distributions[: cfg.max_distributions]

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
                num_partitions=cfg.num_partitions,
                lower_bound=col_lower,
                upper_bound=col_upper,
                lazy_metrics=cfg.lazy_metrics,
                progress_callback=cfg.progress_callback,
            )
            all_results_dfs.append(results_df)

            # Build lazy context for on-demand metric computation
            if cfg.lazy_metrics:
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
        return FitResults(combined_df, lazy_contexts=lazy_contexts if cfg.lazy_metrics else None)

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
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
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
            progress_callback: Optional callback for progress updates (v2.0.0)

        Returns:
            Spark DataFrame with fit results for this column
        """
        # Create integer data sample for fitting
        sample_size = min(FITTING_SAMPLE_SIZE, row_count)
        fraction = min(sample_size / row_count, 1.0)
        # Use backend's sample_column which handles both Spark and pandas
        raw_sample = self._backend.sample_column(df_sample, column, fraction=fraction, seed=self.random_seed)

        # Handle empty sample (all NaN/inf data filtered out)
        if len(raw_sample) == 0:
            logger.warning(f"  No valid data for '{column}' after filtering NaN/inf values")
            import pandas as pd

            if self.spark is not None:
                return self.spark.createDataFrame([], schema=DISCRETE_FIT_RESULT_SCHEMA)
            else:
                return pd.DataFrame(
                    columns=[
                        "column_name",
                        "distribution",
                        "parameters",
                        "sse",
                        "aic",
                        "bic",
                        "ks_statistic",
                        "pvalue",
                        "ad_statistic",
                        "ad_pvalue",
                        "data_min",
                        "data_max",
                        "data_mean",
                        "data_stddev",
                        "data_count",
                        "lower_bound",
                        "upper_bound",
                    ]
                )

        data_sample = raw_sample.astype(int)
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
            progress_callback=progress_callback,
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

            if results:
                results_df = pd.DataFrame(results)
            else:
                # Create empty DataFrame with proper schema to preserve API contract
                results_df = pd.DataFrame(
                    columns=[
                        "column_name",
                        "distribution",
                        "parameters",
                        "sse",
                        "aic",
                        "bic",
                        "ks_statistic",
                        "pvalue",
                        "ad_statistic",
                        "ad_pvalue",
                        "data_min",
                        "data_max",
                        "data_mean",
                        "data_stddev",
                        "data_count",
                        "lower_bound",
                        "upper_bound",
                    ]
                )

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
        # Use base class validation methods (no bins for discrete)
        BaseFitter._validate_max_distributions(max_distributions)
        BaseFitter._validate_column_exists(df, column)
        BaseFitter._validate_column_numeric(df, column)
        BaseFitter._validate_sample_fraction(sample_fraction)

    # _validate_bounds inherited from BaseFitter
    # _resolve_bounds inherited from BaseFitter
    # _apply_sampling inherited from BaseFitter
    # _calculate_partitions inherited from BaseFitter

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
