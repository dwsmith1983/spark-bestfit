"""Continuous distribution fitting engine for Spark."""

import logging
from functools import reduce
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

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
from spark_bestfit.distributions import DistributionRegistry
from spark_bestfit.fitting import FIT_RESULT_SCHEMA, FITTING_SAMPLE_SIZE, compute_data_stats, fit_single_distribution
from spark_bestfit.histogram import HistogramComputer
from spark_bestfit.results import DistributionFitResult, FitResults, FitResultsType, LazyMetricsContext

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend

logger = logging.getLogger(__name__)

# Re-export for convenience
DEFAULT_EXCLUDED_DISTRIBUTIONS: Tuple[str, ...] = tuple(DistributionRegistry.DEFAULT_EXCLUSIONS)


class DistributionFitter(BaseFitter):
    """Modern Spark distribution fitting engine.

    Efficiently fits ~90 scipy.stats distributions to data using Spark's
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

    # Class attributes for BaseFitter
    _registry_class = DistributionRegistry
    _default_exclusions = DEFAULT_EXCLUDED_DISTRIBUTIONS

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        excluded_distributions: Optional[Tuple[str, ...]] = None,
        random_seed: int = 42,
        backend: Optional["ExecutionBackend"] = None,
    ):
        """Initialize DistributionFitter.

        Args:
            spark: SparkSession. If None, uses the active session.
                Ignored if ``backend`` is provided.
            excluded_distributions: Distributions to exclude from fitting.
                Defaults to DEFAULT_EXCLUDED_DISTRIBUTIONS (slow distributions).
                Pass an empty tuple ``()`` to include ALL scipy distributions.
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
        self._histogram_computer = HistogramComputer(backend=self._backend)

    def fit(
        self,
        df: DataFrame,
        column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        config: Optional[FitterConfig] = None,
        *,
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
    ) -> FitResultsType:
        """Fit distributions to data column(s).

        Args:
            df: Spark DataFrame containing data
            column: Name of single column to fit distributions to
            columns: List of column names for multi-column fitting
            config: FitterConfig object (v2.2.0). Provides a cleaner way to
                configure fitting with many parameters. If provided, individual
                parameters below are ignored (except progress_callback which
                can override the config's callback). Use FitterConfigBuilder
                for fluent configuration.
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
            >>> # Using FitterConfig (recommended for complex configs, v2.2.0)
            >>> from spark_bestfit import FitterConfigBuilder
            >>> config = (FitterConfigBuilder()
            ...     .with_bins(100)
            ...     .with_bounds(lower=0, upper=100)
            ...     .with_sampling(fraction=0.1)
            ...     .build())
            >>> results = fitter.fit(df, column='value', config=config)
            >>>
            >>> # Single column (backward compatible)
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
            >>> # Lazy metrics for faster fitting when only using AIC/BIC (v1.5.0)
            >>> results = fitter.fit(df, 'value', lazy_metrics=True)
            >>> best_aic = results.best(n=1, metric='aic')[0]  # Fast, no KS/AD computed
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
                bins=bins,
                use_rice_rule=use_rice_rule,
                support_at_zero=support_at_zero,
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
            self._validate_inputs(df, col, cfg.max_distributions, cfg.bins, cfg.sample_fraction)

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
            support_at_zero=cfg.support_at_zero,
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
            logger.info(f"Fitting column '{col}'...")
            results_df = self._fit_single_column(
                df_sample=df_sample,
                column=col,
                row_count=row_count,
                bins=cfg.bins,
                use_rice_rule=cfg.use_rice_rule,
                distributions=distributions,
                num_partitions=cfg.num_partitions,
                lower_bound=col_lower,
                upper_bound=col_upper,
                lazy_metrics=cfg.lazy_metrics,
                prefilter=cfg.prefilter,
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
                    is_discrete=False,
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
        bins: Union[int, Tuple[float, ...]],
        use_rice_rule: bool,
        distributions: List[str],
        num_partitions: Optional[int],
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        lazy_metrics: bool = False,
        prefilter: Union[bool, str] = False,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
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
            progress_callback: Optional callback for progress updates (v2.0.0)

        Returns:
            Spark DataFrame with fit results for this column
        """
        # Compute histogram (returns bin edges for CDF-based fitting)
        y_hist, bin_edges = self._histogram_computer.compute_histogram(
            df_sample, column, bins=bins, use_rice_rule=use_rice_rule, approx_count=row_count
        )
        logger.info(f"  Histogram for '{column}': {len(bin_edges) - 1} bins")

        # Create fitting sample
        data_sample = self._create_fitting_sample(df_sample, column, row_count)

        # Handle empty sample (all NaN/inf data filtered out)
        if len(data_sample) == 0:
            logger.warning(f"  No valid data for '{column}' after filtering NaN/inf values")
            import pandas as pd

            if self.spark is not None:
                return self.spark.createDataFrame([], schema=FIT_RESULT_SCHEMA)
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

        # Compute data stats for provenance (once per column)
        data_stats = compute_data_stats(data_sample)

        # Interleave slow distributions for better partition balance
        # Lazy import to avoid circular dependency with core.py
        from spark_bestfit.core import _interleave_distributions

        distributions = _interleave_distributions(distributions)

        # Execute parallel fitting via backend (v2.0 abstraction)
        # Backend handles: broadcast, partitioning, UDF application, collection
        results = self._backend.parallel_fit(
            distributions=distributions,
            histogram=(y_hist, bin_edges),
            data_sample=data_sample,
            fit_func=fit_single_distribution,
            column_name=column,
            data_stats=data_stats,
            num_partitions=num_partitions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lazy_metrics=lazy_metrics,
            is_discrete=False,
            progress_callback=progress_callback,
        )

        # Convert results to DataFrame
        if self.spark is not None:
            # Spark backend
            if results:
                results_df = self.spark.createDataFrame(results, schema=FIT_RESULT_SCHEMA)
            else:
                results_df = self.spark.createDataFrame([], schema=FIT_RESULT_SCHEMA)
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
        # Use base class validation methods
        BaseFitter._validate_max_distributions(max_distributions)
        BaseFitter._validate_column_exists(df, column)
        BaseFitter._validate_column_numeric(df, column)
        BaseFitter._validate_sample_fraction(sample_fraction)

        # Continuous-specific: validate bins parameter
        if isinstance(bins, int) and bins <= 0:
            raise ValueError(f"bins must be positive, got {bins}")

    # _validate_bounds inherited from BaseFitter
    # _resolve_bounds inherited from BaseFitter
    # _apply_sampling inherited from BaseFitter

    def _create_fitting_sample(self, df: DataFrame, column: str, row_count: int) -> np.ndarray:
        """Create numpy sample array for scipy distribution fitting.

        Samples up to FITTING_SAMPLE_SIZE rows from the DataFrame for use in
        scipy's distribution fitting functions.

        Args:
            df: Spark DataFrame or pandas DataFrame containing data
            column: Column name to sample
            row_count: Total row count (used to calculate sampling fraction)

        Returns:
            Numpy array of sampled values for distribution fitting
        """
        sample_size = min(FITTING_SAMPLE_SIZE, row_count)
        fraction = min(sample_size / row_count, 1.0)
        # Use backend's sample_column which handles both Spark and pandas
        return self._backend.sample_column(df, column, fraction=fraction, seed=self.random_seed)

    # _calculate_partitions inherited from BaseFitter

    @staticmethod
    def _prefilter_distributions(
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
