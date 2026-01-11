"""Data storage classes for distribution fitting results.

This module contains the core data classes for storing individual distribution
fit results and related context objects. These are the fundamental building
blocks used throughout spark_bestfit.

Classes:
    DistributionFitResult: Stores a single distribution's fitted parameters and metrics.
    LazyMetricsContext: Context for deferred KS/AD metric computation.

Type Aliases:
    MetricName: Valid metric names for sorting/filtering.
    ContinuousHistogram: Tuple type for continuous distribution histograms.
    DiscreteHistogram: Tuple type for discrete distribution histograms.
    HistogramBins: Array of bin edges (len = n_bins + 1).
    HistogramCounts: Array of counts/density per bin.
    HistogramResult: Tuple type for HistogramComputer results (counts, bins).

Constants:
    FITTING_SAMPLE_SIZE: Default sample size for fitting (10000).
    DEFAULT_PVALUE_THRESHOLD: Default p-value threshold (0.05).
    DEFAULT_KS_THRESHOLD: Default KS statistic threshold (0.10).
    DEFAULT_AD_THRESHOLD: Default AD statistic threshold (2.0).
    DEFAULT_BINS: Default number of histogram bins (50).
    DEFAULT_BOOTSTRAP_SAMPLES: Default bootstrap iterations (1000).
    DEFAULT_MAX_SAMPLES: Default maximum samples to collect (10000).
    DEFAULT_DPI: Default plot DPI (100).
    DEFAULT_SAMPLE_SIZE: Default sample size for sampling methods (1000).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeAlias, Union

import numpy as np
import scipy.stats as st

from spark_bestfit.truncated import TruncatedFrozenDist

# PySpark is optional - only import if available
try:
    from pyspark.sql import DataFrame

    _PYSPARK_AVAILABLE = True
except ImportError:
    DataFrame = None  # type: ignore[assignment,misc]
    _PYSPARK_AVAILABLE = False

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


# =============================================================================
# Constants
# =============================================================================

# Default sample size for fitting operations
FITTING_SAMPLE_SIZE = 10000

# Default threshold values for quality assessment
DEFAULT_PVALUE_THRESHOLD = 0.05
DEFAULT_KS_THRESHOLD = 0.10
DEFAULT_AD_THRESHOLD = 2.0

# Default histogram and plotting parameters
DEFAULT_BINS = 50
DEFAULT_DPI = 100

# Default bootstrap and sampling parameters
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_MAX_SAMPLES = 10000
DEFAULT_SAMPLE_SIZE = 1000


# =============================================================================
# Type Aliases
# =============================================================================

# Valid metric names for sorting/filtering (for IDE autocomplete and type checking)
MetricName = Literal["sse", "aic", "bic", "ks_statistic", "ad_statistic"]

# Histogram type aliases - distinguish continuous from discrete semantically
# ContinuousHistogram: (density_values, bin_edges) where len(edges) = len(density) + 1
ContinuousHistogram = Tuple[np.ndarray, np.ndarray]

# DiscreteHistogram: (x_values, empirical_pmf) where both arrays have same length
DiscreteHistogram = Tuple[np.ndarray, np.ndarray]

# HistogramBins: Array of bin edges (len = n_bins + 1)
HistogramBins: TypeAlias = np.ndarray

# HistogramCounts: Array of counts/density per bin
HistogramCounts: TypeAlias = np.ndarray

# HistogramResult: Complete histogram from HistogramComputer.compute_histogram()
# Order is (counts, bins) where counts has n_bins elements and bins has n_bins + 1 edges
HistogramResult: TypeAlias = Tuple[np.ndarray, np.ndarray]


# =============================================================================
# DataFrame Utilities (Multi-Backend Support)
# =============================================================================


def _is_spark_dataframe(df) -> bool:
    """Check if df is a Spark DataFrame.

    Uses duck typing to detect Spark DataFrames without requiring pyspark import.

    Args:
        df: DataFrame to check

    Returns:
        True if df is a Spark DataFrame, False otherwise
    """
    return hasattr(df, "toPandas") and hasattr(df, "select")


def _is_ray_dataset(df) -> bool:
    """Check if df is a Ray Dataset (not a pandas DataFrame).

    Uses duck typing to detect Ray Datasets. Note that pandas DataFrames
    don't have select_columns(), so this won't match them.

    Args:
        df: DataFrame to check

    Returns:
        True if df is a Ray Dataset, False otherwise
    """
    return hasattr(df, "select_columns") and hasattr(df, "to_pandas")


def _get_dataframe_row_count(df) -> int:
    """Get total row count from any supported DataFrame type.

    Supports Spark DataFrames, Ray Datasets, and pandas DataFrames.

    Args:
        df: DataFrame (Spark, Ray Dataset, or pandas)

    Returns:
        Total number of rows in the DataFrame
    """
    if _is_spark_dataframe(df):
        return df.count()
    elif _is_ray_dataset(df):
        return df.count()
    else:  # pandas DataFrame
        return len(df)


def _collect_dataframe_column(df, column: str) -> np.ndarray:
    """Extract a column from any supported DataFrame type as numpy array.

    Supports Spark DataFrames, Ray Datasets, and pandas DataFrames.

    Args:
        df: DataFrame (Spark, Ray Dataset, or pandas)
        column: Column name to extract

    Returns:
        Numpy array of column values
    """
    if _is_spark_dataframe(df):
        return np.array(df.select(column).toPandas()[column].values)
    elif _is_ray_dataset(df):
        return df.select_columns([column]).to_pandas()[column].values
    else:  # pandas DataFrame
        return df[column].values


def _sample_dataframe_column(df, column: str, fraction: float, seed: Optional[int]) -> np.ndarray:
    """Sample a column from any supported DataFrame type and return as numpy array.

    Supports Spark DataFrames, Ray Datasets, and pandas DataFrames.

    Args:
        df: DataFrame (Spark, Ray Dataset, or pandas)
        column: Column name to sample
        fraction: Fraction of rows to sample (0 < fraction <= 1)
        seed: Random seed for reproducibility (can be None)

    Returns:
        Numpy array of sampled column values
    """
    if _is_spark_dataframe(df):
        if seed is not None:
            sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed)
        else:
            sampled = df.sample(withReplacement=False, fraction=fraction)
        return np.array(sampled.select(column).toPandas()[column].values)
    elif _is_ray_dataset(df):
        sampled = df.random_sample(fraction, seed=seed)
        return sampled.select_columns([column]).to_pandas()[column].values
    else:  # pandas DataFrame
        sample_df = df[[column]].sample(frac=fraction, random_state=seed)
        return sample_df[column].values


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(slots=True)
class LazyMetricsContext:
    """Context for deferred KS/AD metric computation.

    When lazy_metrics=True during fitting, this context stores everything
    needed to compute KS/AD metrics on-demand later. The key insight is that
    with the same (DataFrame, column, seed), we can recreate the exact sample.

    Attributes:
        source_df: Reference to the source DataFrame for sampling
        column: Column name to sample from
        random_seed: Seed used for reproducible sampling
        row_count: Total row count for calculating sample fraction
        lower_bound: Optional lower bound for truncated distributions
        upper_bound: Optional upper bound for truncated distributions
        is_discrete: Whether this is discrete distribution fitting

    Note:
        The source_df reference must remain valid (not unpersisted) for lazy
        metric computation to work. Call materialize() before unpersisting
        if you need the metrics.
    """

    source_df: DataFrame
    column: str
    random_seed: int
    row_count: int
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    is_discrete: bool = False


@dataclass(slots=True)
class DistributionFitResult:
    """Result from fitting a single distribution.

    Attributes:
        distribution: Name of the scipy.stats distribution
        parameters: Fitted parameters (shape params + loc + scale)
        sse: Sum of Squared Errors
        column_name: Name of the column that was fitted (for multi-column support)
        aic: Akaike Information Criterion (lower is better)
        bic: Bayesian Information Criterion (lower is better)
        ks_statistic: Kolmogorov-Smirnov statistic (lower is better)
        pvalue: P-value from KS test (higher indicates better fit)
        ad_statistic: Anderson-Darling statistic (lower is better)
        ad_pvalue: P-value from A-D test (only for norm, expon, logistic, gumbel_r, gumbel_l)
        data_min: Minimum value in the data used for fitting
        data_max: Maximum value in the data used for fitting
        data_mean: Mean of the data used for fitting
        data_stddev: Standard deviation of the data used for fitting
        data_count: Number of samples in the data used for fitting
        lower_bound: Lower bound for truncated distribution fitting (v1.4.0).
            When set, the distribution is truncated at this lower limit.
        upper_bound: Upper bound for truncated distribution fitting (v1.4.0).
            When set, the distribution is truncated at this upper limit.

    Note:
        The p-value from the KS test is approximate when parameters are
        estimated from the same data being tested. It tends to be conservative
        (larger than it should be). Use it for rough guidance, not strict
        hypothesis testing. The ks_statistic is valid for ranking fits.

        The ad_pvalue is only available for 5 distributions (norm, expon,
        logistic, gumbel_r, gumbel_l) where scipy has critical value tables.
        For other distributions, ad_pvalue will be None but ad_statistic
        is still valid for ranking fits.

        When bounds are set (lower_bound and/or upper_bound), methods like
        sample(), pdf(), cdf(), and ppf() automatically use scipy.stats.truncate()
        to return values respecting the bounded domain.
    """

    distribution: str
    parameters: List[float]
    sse: float
    column_name: Optional[str] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    ks_statistic: Optional[float] = None
    pvalue: Optional[float] = None
    ad_statistic: Optional[float] = None
    ad_pvalue: Optional[float] = None
    # Flat data stats (v2.0: replaced data_summary MapType for ~20% perf)
    data_min: Optional[float] = None
    data_max: Optional[float] = None
    data_mean: Optional[float] = None
    data_stddev: Optional[float] = None
    data_count: Optional[float] = None
    # Bounds for truncated distribution fitting (v1.4.0)
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "column_name": self.column_name,
            "distribution": self.distribution,
            "parameters": self.parameters,
            "sse": self.sse,
            "aic": self.aic,
            "bic": self.bic,
            "ks_statistic": self.ks_statistic,
            "pvalue": self.pvalue,
            "ad_statistic": self.ad_statistic,
            "ad_pvalue": self.ad_pvalue,
            "data_min": self.data_min,
            "data_max": self.data_max,
            "data_mean": self.data_mean,
            "data_stddev": self.data_stddev,
            "data_count": self.data_count,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    def get_scipy_dist(self, frozen: bool = True):
        """Get scipy distribution object.

        Args:
            frozen: If True (default), return a frozen distribution with parameters applied.
                If False, return the unfrozen distribution class.

        Returns:
            scipy.stats distribution object. If bounds are set and frozen=True,
            returns a TruncatedFrozenDist wrapper that handles truncation.

        Note:
            When bounds are set (lower_bound and/or upper_bound), the returned
            distribution is truncated. This ensures that sampling and PDF/CDF
            evaluation respect the bounds.
        """
        dist_class = getattr(st, self.distribution)

        if not frozen:
            return dist_class

        # Create frozen distribution with parameters
        frozen_dist = dist_class(*self.parameters)

        # Apply truncation if bounds are set
        if self.lower_bound is not None or self.upper_bound is not None:
            lb = self.lower_bound if self.lower_bound is not None else -np.inf
            ub = self.upper_bound if self.upper_bound is not None else np.inf
            return TruncatedFrozenDist(frozen_dist, lb, ub)

        return frozen_dist

    def sample(self, size: int = DEFAULT_SAMPLE_SIZE, random_state: Optional[int] = None) -> np.ndarray:
        """Generate random samples from the fitted distribution.

        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of random samples. If bounds are set, samples are
            guaranteed to be within [lower_bound, upper_bound].

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> samples = result.sample(size=10000, random_state=42)
        """
        # get_scipy_dist() returns a frozen distribution, optionally truncated
        frozen_dist = self.get_scipy_dist()
        return frozen_dist.rvs(size=size, random_state=random_state)

    def sample_spark(
        self,
        n: int,
        spark: Optional["SparkSession"] = None,
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
        column_name: str = "sample",
    ) -> DataFrame:
        """Generate distributed samples from the fitted distribution using Spark.

        .. deprecated:: 2.0.0
            Will be removed in v3.0.0. Use :func:`spark_bestfit.sampling.sample_distributed`
            with ``SparkBackend`` instead.

        Uses Spark's parallelism to generate samples across the cluster,
        enabling efficient generation of millions of samples.

        Args:
            n: Total number of samples to generate
            spark: SparkSession. If None, uses the active session.
            num_partitions: Number of partitions to use. Defaults to spark default parallelism.
            random_seed: Random seed for reproducibility. Each partition uses seed + partition_id.
            column_name: Name for the output column (default: "sample")

        Returns:
            Spark DataFrame with single column containing samples

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> samples_df = result.sample_spark(n=1_000_000, spark=spark)
            >>> samples_df.show(5)
            +-------------------+
            |             sample|
            +-------------------+
            | 0.4691122931291924|
            |-0.2828633018445851|
            | 1.0093545783546243|
            +-------------------+
        """
        import warnings

        warnings.warn(
            "sample_spark() is deprecated and will be removed in v3.0.0. "
            "Use sample_distributed(backend=SparkBackend()) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from spark_bestfit.backends.spark import SparkBackend
        from spark_bestfit.sampling import sample_distributed

        backend = SparkBackend(spark)
        return sample_distributed(
            distribution=self.distribution,
            parameters=self.parameters,
            n=n,
            backend=backend,
            num_partitions=num_partitions,
            random_seed=random_seed,
            column_name=column_name,
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate probability density function at given points.

        Args:
            x: Points at which to evaluate PDF

        Returns:
            PDF values at x. If bounds are set, the PDF is normalized
            to integrate to 1 over the bounded domain.

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> x = np.linspace(0, 10, 100)
            >>> y = result.pdf(x)
        """
        # get_scipy_dist() returns a frozen distribution, optionally truncated
        frozen_dist = self.get_scipy_dist()
        return frozen_dist.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate cumulative distribution function at given points.

        Args:
            x: Points at which to evaluate CDF

        Returns:
            CDF values at x. If bounds are set, the CDF is adjusted
            for the truncated domain (0 at lower_bound, 1 at upper_bound).
        """
        # get_scipy_dist() returns a frozen distribution, optionally truncated
        frozen_dist = self.get_scipy_dist()
        return frozen_dist.cdf(x)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Evaluate percent point function (inverse CDF) at given quantiles.

        Args:
            q: Quantiles at which to evaluate PPF (0 to 1)

        Returns:
            PPF values at q. If bounds are set, values are guaranteed
            to be within [lower_bound, upper_bound].
        """
        # get_scipy_dist() returns a frozen distribution, optionally truncated
        frozen_dist = self.get_scipy_dist()
        return frozen_dist.ppf(q)

    def save(
        self,
        path: Union[str, Path],
        format: Optional[Literal["json", "pickle"]] = None,
        indent: Optional[int] = 2,
    ) -> None:
        """Save fitted distribution to file.

        Serializes the distribution parameters and metrics to JSON or pickle format.
        JSON is recommended for human-readable, version-safe output. Pickle is
        available for faster serialization when human-readability is not required.

        Args:
            path: File path. Format is detected from extension if not specified.
            format: Output format - 'json' (human-readable) or 'pickle'.
                If None, detected from file extension (.json, .pkl, .pickle).
            indent: JSON indentation level (default 2). Use None for compact output.
                Ignored for pickle format.

        Raises:
            SerializationError: If format cannot be determined or write fails.

        Example:
            >>> best = results.best(n=1)[0]
            >>> best.save("model.json")
            >>> best.save("model.pkl", format="pickle")
            >>> best.save("compact.json", indent=None)
        """
        from spark_bestfit.serialization import detect_format, save_json, save_pickle, serialize_to_dict

        path = Path(path)
        file_format = format or detect_format(path)

        if file_format == "json":
            data = serialize_to_dict(self)
            save_json(data, path, indent)
        else:  # pickle
            save_pickle(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DistributionFitResult":
        """Load fitted distribution from file.

        Reconstructs a DistributionFitResult from a previously saved file.
        The loaded result can be used for sampling, PDF/CDF evaluation, etc.

        Args:
            path: File path. Format is detected from extension (.json, .pkl, .pickle).

        Returns:
            Reconstructed DistributionFitResult

        Raises:
            SerializationError: If file format is invalid or distribution is unknown.
            FileNotFoundError: If file does not exist.

        Example:
            >>> loaded = DistributionFitResult.load("model.json")
            >>> samples = loaded.sample(n=1000)
            >>> pdf_values = loaded.pdf(np.linspace(0, 100, 100))

        Warning:
            Only load pickle files from trusted sources.
        """
        from spark_bestfit.serialization import deserialize_from_dict, detect_format, load_json, load_pickle

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_format = detect_format(path)

        if file_format == "json":
            data = load_json(path)
            return deserialize_from_dict(data)
        else:  # pickle
            return load_pickle(path)

    def get_param_names(self) -> List[str]:
        """Get parameter names for this distribution.

        Returns:
            List of parameter names in order matching self.parameters

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> print(result.distribution)
            'gamma'
            >>> print(result.get_param_names())
            ['a', 'loc', 'scale']
            >>> print(dict(zip(result.get_param_names(), result.parameters)))
            {'a': 2.5, 'loc': 0.0, 'scale': 3.2}
        """
        from spark_bestfit.distributions import DiscreteDistributionRegistry
        from spark_bestfit.fitting import get_continuous_param_names

        # Check if this is a discrete distribution
        registry = DiscreteDistributionRegistry()
        if self.distribution in registry.get_distributions():
            config = registry.get_param_config(self.distribution)
            return config["param_names"]
        else:
            # Continuous distribution
            return get_continuous_param_names(self.distribution)

    def confidence_intervals(
        self,
        df,
        column: str,
        alpha: float = DEFAULT_PVALUE_THRESHOLD,
        n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for fitted parameters.

        Uses the percentile bootstrap method: resample data with replacement,
        refit the distribution, and compute confidence intervals from the
        empirical distribution of fitted parameters.

        Args:
            df: DataFrame containing the data (Spark DataFrame, pandas DataFrame,
                or Ray Dataset)
            column: Column name containing the data
            alpha: Significance level (default 0.05 for 95% CI)
            n_bootstrap: Number of bootstrap samples (default 1000)
            max_samples: Maximum rows to collect from DataFrame (default 10000)
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary mapping parameter names to (lower, upper) bounds

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> ci = result.confidence_intervals(df, 'value', alpha=0.05, random_seed=42)
            >>> print(result.distribution)
            'gamma'
            >>> for param, (lower, upper) in ci.items():
            ...     print(f"  {param}: [{lower:.4f}, {upper:.4f}]")
            a: [2.35, 2.65]
            loc: [-0.12, 0.08]
            scale: [3.05, 3.35]

        Note:
            Bootstrap computation can be slow for large n_bootstrap values.
            The default 1000 iterations provides reasonable precision.
        """
        from spark_bestfit.discrete_fitting import bootstrap_discrete_confidence_intervals
        from spark_bestfit.distributions import DiscreteDistributionRegistry
        from spark_bestfit.fitting import bootstrap_confidence_intervals

        # Sample data from DataFrame (supports Spark, pandas, and Ray backends)
        total_rows = _get_dataframe_row_count(df)
        if total_rows <= max_samples:
            # Collect all rows
            data = _collect_dataframe_column(df, column)
        else:
            # Sample rows
            fraction = max_samples / total_rows
            data = _sample_dataframe_column(df, column, fraction, random_seed)

        # Check if this is a discrete distribution
        registry = DiscreteDistributionRegistry()
        if self.distribution in registry.get_distributions():
            return bootstrap_discrete_confidence_intervals(
                dist_name=self.distribution,
                data=data.astype(int),
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )
        else:
            return bootstrap_confidence_intervals(
                dist_name=self.distribution,
                data=data,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                random_seed=random_seed,
            )

    def diagnostics(
        self,
        data: np.ndarray,
        y_hist: Optional[np.ndarray] = None,
        x_hist: Optional[np.ndarray] = None,
        bins: int = DEFAULT_BINS,
        title: str = "",
        figsize: Tuple[int, int] = (14, 12),
        dpi: int = DEFAULT_DPI,
        title_fontsize: int = 16,
        subplot_title_fontsize: int = 12,
        label_fontsize: int = 10,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        save_format: str = "png",
    ):
        """Create a 2x2 diagnostic plot panel for assessing distribution fit quality.

        Generates four diagnostic plots:
        - Q-Q Plot (top-left): Compares sample quantiles vs theoretical quantiles
        - P-P Plot (top-right): Compares empirical vs theoretical probabilities
        - Residual Histogram (bottom-left): Distribution of fit residuals
        - CDF Comparison (bottom-right): Empirical vs theoretical CDF overlay

        Args:
            data: Sample data array (1D numpy array)
            y_hist: Optional pre-computed histogram density values. If None,
                computed from data using specified bins.
            x_hist: Optional pre-computed histogram bin centers. If None,
                computed from data using specified bins.
            bins: Number of histogram bins (used if y_hist/x_hist not provided)
            title: Overall figure title
            figsize: Figure size (width, height)
            dpi: Dots per inch for saved figures
            title_fontsize: Main title font size
            subplot_title_fontsize: Subplot title font size
            label_fontsize: Axis label font size
            grid_alpha: Grid transparency (0-1)
            save_path: Optional path to save figure
            save_format: Save format (png, pdf, svg)

        Returns:
            Tuple of (figure, array of axes)

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> fig, axes = result.diagnostics(data, title='Fit Diagnostics')
            >>> plt.show()
        """
        from spark_bestfit.plotting import plot_diagnostics

        return plot_diagnostics(
            result=self,
            data=data,
            y_hist=y_hist,
            x_hist=x_hist,
            bins=bins,
            title=title,
            figsize=figsize,
            dpi=dpi,
            title_fontsize=title_fontsize,
            subplot_title_fontsize=subplot_title_fontsize,
            label_fontsize=label_fontsize,
            grid_alpha=grid_alpha,
            save_path=save_path,
            save_format=save_format,
        )

    def __repr__(self) -> str:
        """String representation of the result."""
        param_str = ", ".join([f"{p:.4f}" for p in self.parameters])
        aic_str = f"{self.aic:.2f}" if self.aic is not None else "None"
        bic_str = f"{self.bic:.2f}" if self.bic is not None else "None"
        ks_str = f"{self.ks_statistic:.6f}" if self.ks_statistic is not None else "None"
        pval_str = f"{self.pvalue:.4f}" if self.pvalue is not None else "None"
        ad_str = f"{self.ad_statistic:.6f}" if self.ad_statistic is not None else "None"
        ad_pval_str = f"{self.ad_pvalue:.4f}" if self.ad_pvalue is not None else "None"
        col_str = f"column_name='{self.column_name}', " if self.column_name else ""

        # Build bounds string if set
        bounds_parts = []
        if self.lower_bound is not None:
            bounds_parts.append(f"lower_bound={self.lower_bound:.4f}")
        if self.upper_bound is not None:
            bounds_parts.append(f"upper_bound={self.upper_bound:.4f}")
        bounds_str = ", ".join(bounds_parts)
        bounds_suffix = f", {bounds_str}" if bounds_str else ""

        return (
            f"DistributionFitResult({col_str}distribution='{self.distribution}', "
            f"sse={self.sse:.6f}, aic={aic_str}, bic={bic_str}, "
            f"ks_statistic={ks_str}, pvalue={pval_str}, "
            f"ad_statistic={ad_str}, ad_pvalue={ad_pval_str}, "
            f"parameters=[{param_str}]{bounds_suffix})"
        )
