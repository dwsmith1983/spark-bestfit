"""Core distribution fitting engine for Spark."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import NumericType

from spark_bestfit.distributions import DistributionRegistry
from spark_bestfit.fitting import FITTING_SAMPLE_SIZE, create_fitting_udf
from spark_bestfit.histogram import HistogramComputer
from spark_bestfit.results import DistributionFitResult, FitResults
from spark_bestfit.utils import get_spark_session

logger = logging.getLogger(__name__)

# Re-export for convenience
DEFAULT_EXCLUDED_DISTRIBUTIONS: Tuple[str, ...] = tuple(DistributionRegistry.DEFAULT_EXCLUSIONS)


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
        column: str,
        bins: Union[int, Tuple[float, ...]] = 50,
        use_rice_rule: bool = True,
        support_at_zero: bool = False,
        max_distributions: Optional[int] = None,
        enable_sampling: bool = True,
        sample_fraction: Optional[float] = None,
        max_sample_size: int = 1_000_000,
        sample_threshold: int = 10_000_000,
        num_partitions: Optional[int] = None,
    ) -> FitResults:
        """Fit distributions to data column.

        Args:
            df: Spark DataFrame containing data
            column: Name of column to fit distributions to
            bins: Number of histogram bins or tuple of bin edges
            use_rice_rule: Use Rice rule to auto-determine bin count
            support_at_zero: Only fit non-negative distributions
            max_distributions: Limit number of distributions (for testing)
            enable_sampling: Enable sampling for large datasets
            sample_fraction: Fraction to sample (None = auto-determine)
            max_sample_size: Maximum rows to sample when auto-determining
            sample_threshold: Row count above which sampling is applied
            num_partitions: Spark partitions (None = auto-determine)

        Returns:
            FitResults object with fitted distributions

        Raises:
            ValueError: If column not found, DataFrame empty, or invalid params
            TypeError: If column is not numeric

        Example:
            >>> results = fitter.fit(df, 'value')
            >>> results = fitter.fit(df, 'value', bins=100, support_at_zero=True)
        """
        # Input validation
        self._validate_inputs(df, column, max_distributions, bins, sample_fraction)

        # Get row count
        row_count = df.count()
        if row_count == 0:
            raise ValueError("DataFrame is empty")
        logger.info(f"Row count: {row_count}")

        # Sample if needed
        df_sample = self._apply_sampling(
            df, row_count, enable_sampling, sample_fraction, max_sample_size, sample_threshold
        )

        # Compute histogram
        logger.info("Computing histogram...")
        y_hist, x_hist = self._histogram_computer.compute_histogram(
            df_sample, column, bins=bins, use_rice_rule=use_rice_rule, approx_count=row_count
        )
        logger.info(f"Histogram computed: {len(x_hist)} bins")

        # Broadcast histogram
        histogram_bc = self.spark.sparkContext.broadcast((y_hist, x_hist))

        # Create fitting sample
        logger.info("Creating data sample for parameter fitting...")
        data_sample = self._create_fitting_sample(df_sample, column, row_count)
        data_sample_bc = self.spark.sparkContext.broadcast(data_sample)
        logger.info(f"Data sample size: {len(data_sample)}")

        # Get distributions to fit
        distributions = self._registry.get_distributions(
            support_at_zero=support_at_zero,
            additional_exclusions=list(self.excluded_distributions),
        )

        if max_distributions is not None and max_distributions > 0:
            distributions = distributions[:max_distributions]

        logger.info(f"Fitting {len(distributions)} distributions...")

        try:
            # Create DataFrame of distributions
            dist_df = self.spark.createDataFrame([(dist,) for dist in distributions], ["distribution_name"])

            # Determine partitioning
            n_partitions = num_partitions or self._calculate_partitions(len(distributions))
            dist_df = dist_df.repartition(n_partitions)

            # Apply fitting UDF
            fitting_udf = create_fitting_udf(histogram_bc, data_sample_bc)
            results_df = dist_df.select(fitting_udf(F.col("distribution_name")).alias("result")).select("result.*")

            # Filter failed fits and cache
            results_df = results_df.filter(F.col("sse") < float(np.inf))
            results_df = results_df.cache()

            num_results = results_df.count()
            logger.info(f"Successfully fit {num_results}/{len(distributions)} distributions")

            return FitResults(results_df)

        finally:
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

        # Compute histogram for plotting
        y_hist, x_hist = self._histogram_computer.compute_histogram(df, column, bins=bins, use_rice_rule=use_rice_rule)

        return plot_distribution(
            result=result,
            y_hist=y_hist,
            x_hist=x_hist,
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

        y_hist, x_hist = self._histogram_computer.compute_histogram(df, column, bins=bins, use_rice_rule=use_rice_rule)

        return plot_comparison(
            results=results,
            y_hist=y_hist,
            x_hist=x_hist,
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
        """Validate inputs."""
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

    def _apply_sampling(
        self,
        df: DataFrame,
        row_count: int,
        enable_sampling: bool,
        sample_fraction: Optional[float],
        max_sample_size: int,
        sample_threshold: int,
    ) -> DataFrame:
        """Apply sampling if needed."""
        if not enable_sampling or row_count <= sample_threshold:
            return df

        if sample_fraction is not None:
            fraction = sample_fraction
        else:
            fraction = min(max_sample_size / row_count, 0.35)

        logger.info(f"Sampling {fraction * 100:.1f}% of data ({int(row_count * fraction)} rows)")
        return df.sample(fraction=fraction, seed=self.random_seed)

    def _create_fitting_sample(self, df: DataFrame, column: str, row_count: int) -> np.ndarray:
        """Create sample for scipy distribution fitting."""
        sample_size = min(FITTING_SAMPLE_SIZE, row_count)
        fraction = min(sample_size / row_count, 1.0)
        sample_df = df.select(column).sample(fraction=fraction, seed=self.random_seed)
        return sample_df.toPandas()[column].values

    def _calculate_partitions(self, num_distributions: int) -> int:
        """Calculate optimal partition count."""
        total_cores = self.spark.sparkContext.defaultParallelism
        return min(num_distributions, total_cores * 2)
