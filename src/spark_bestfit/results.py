"""Results handling for fitted distributions."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as st

from spark_bestfit.truncated import TruncatedFrozenDist

# PySpark is optional - only import if available
try:
    import pyspark.sql.functions as F
    from pyspark.sql import DataFrame

    _PYSPARK_AVAILABLE = True
except ImportError:
    F = None  # type: ignore[assignment]
    DataFrame = None  # type: ignore[assignment,misc]
    _PYSPARK_AVAILABLE = False

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

# Type alias for valid metric names (for IDE autocomplete and type checking)
MetricName = Literal["sse", "aic", "bic", "ks_statistic", "ad_statistic"]

# Default sample size for fitting
FITTING_SAMPLE_SIZE = 10000


@dataclass
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


@dataclass
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

    def sample(self, size: int = 1000, random_state: Optional[int] = None) -> np.ndarray:
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
        df: "SparkDataFrame",
        column: str,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        max_samples: int = 10000,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for fitted parameters.

        Uses the percentile bootstrap method: resample data with replacement,
        refit the distribution, and compute confidence intervals from the
        empirical distribution of fitted parameters.

        Args:
            df: Spark DataFrame containing the data
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

        # Sample data from DataFrame
        total_rows = df.count()
        if total_rows <= max_samples:
            # Collect all rows
            data = np.array(df.select(column).toPandas()[column].values)
        else:
            # Sample rows
            fraction = max_samples / total_rows
            if random_seed is not None:
                sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=random_seed)
            else:
                sampled_df = df.sample(withReplacement=False, fraction=fraction)
            data = np.array(sampled_df.select(column).toPandas()[column].values)

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


class BaseFitResults(ABC):
    """Abstract base class for distribution fit results.

    Provides convenient methods for accessing, filtering, and analyzing
    fitted distributions. Wraps a Spark DataFrame but provides pandas-like
    interface for common operations.

    Subclasses:
        - EagerFitResults: All metrics pre-computed during fitting
        - LazyFitResults: KS/AD metrics computed on-demand

    Example:
        >>> results = fitter.fit(df, 'value')
        >>> # Get the best distribution
        >>> best = results.best(n=1)[0]
        >>> # Get top 5 by AIC
        >>> top_aic = results.best(n=5, metric='aic')
        >>> # Convert to pandas for analysis
        >>> df_pandas = results.df.toPandas()
        >>> # Filter by SSE threshold
        >>> good_fits = results.filter(sse_threshold=0.01)
    """

    def __init__(self, results_df: Union[DataFrame, pd.DataFrame]):
        """Initialize BaseFitResults.

        Args:
            results_df: Spark DataFrame or pandas DataFrame with fit results
        """
        self._df = results_df
        # Cache whether this is a Spark DataFrame for fast access
        self._is_spark = hasattr(results_df, "sparkSession")

    @property
    def is_spark_df(self) -> bool:
        """Check if the underlying DataFrame is a Spark DataFrame.

        Returns:
            True if Spark DataFrame, False if pandas DataFrame.
        """
        return self._is_spark

    @property
    @abstractmethod
    def is_lazy(self) -> bool:
        """Check if lazy metrics are available for on-demand computation.

        Returns:
            True if this is a LazyFitResults with lazy contexts,
            False if this is an EagerFitResults with all metrics computed.
        """
        pass

    @abstractmethod
    def materialize(self) -> "EagerFitResults":
        """Force computation of all lazy metrics.

        When lazy_metrics=True was used during fitting, this method computes
        KS and AD statistics for all distributions. Call this before unpersisting
        the source DataFrame if you need the metrics later.

        Returns:
            EagerFitResults with all metrics computed.

        Raises:
            RuntimeError: If the source DataFrame is no longer available (LazyFitResults only).

        Example:
            >>> results = fitter.fit(df, 'value', lazy_metrics=True)
            >>> # Fast: only AIC/BIC/SSE computed
            >>> best_aic = results.best(n=1, metric='aic')[0]
            >>>
            >>> # Before unpersisting, materialize all metrics
            >>> materialized = results.materialize()
            >>> df.unpersist()  # Safe now
            >>>
            >>> # Access KS on materialized results
            >>> best_ks = materialized.best(n=1, metric='ks_statistic')[0]
        """
        pass

    def unpersist(self, blocking: bool = False) -> "BaseFitResults":
        """Release the cached DataFrame from memory.

        Call this method when you no longer need the FitResults to free
        executor memory. This is especially useful in notebook sessions
        where multiple fits accumulate cached DataFrames.

        Note:
            If lazy_metrics=True was used during fitting and you haven't
            called materialize(), you should do so before unpersisting if
            you need KS/AD metrics later. After unpersisting, methods like
            best(), filter(), etc. may trigger recomputation from source.

        Args:
            blocking: If True, block until unpersist completes. Default False.

        Returns:
            Self for method chaining.

        Example:
            >>> results = fitter.fit(df, 'value')
            >>> best = results.best(n=3)  # Get what you need
            >>> results.unpersist()  # Release memory
            >>>
            >>> # With lazy metrics, materialize first
            >>> lazy_results = fitter.fit(df, 'value', lazy_metrics=True)
            >>> materialized = lazy_results.materialize()
            >>> lazy_results.unpersist()  # Release lazy version
        """
        if self._is_spark:
            self._df.unpersist(blocking)
        # For pandas DataFrames, no unpersist needed (garbage collected automatically)
        return self

    @staticmethod
    def _recreate_sample(context: LazyMetricsContext) -> np.ndarray:
        """Recreate the exact sample used during fitting.

        Uses the stored seed and row count to reproduce the same sample
        that was used during initial fitting.

        Supports Spark DataFrames, Ray Datasets, and pandas DataFrames.

        Args:
            context: LazyMetricsContext with source DataFrame and sampling params

        Returns:
            NumPy array with the recreated sample

        Raises:
            RuntimeError: If source DataFrame is no longer available
        """
        try:
            sample_size = min(FITTING_SAMPLE_SIZE, context.row_count)
            fraction = min(sample_size / context.row_count, 1.0)
            df = context.source_df
            column = context.column
            seed = context.random_seed

            # Detect DataFrame type and use appropriate sampling method
            if hasattr(df, "select_columns") and hasattr(df, "random_sample"):
                # Ray Dataset
                sampled = df.random_sample(fraction, seed=seed)
                data = sampled.select_columns([column]).to_pandas()[column].values
            elif hasattr(df, "sample") and hasattr(df, "iloc"):
                # pandas DataFrame
                sample_df = df[[column]].sample(frac=fraction, random_state=seed)
                data = sample_df[column].values
            else:
                # Spark DataFrame (default)
                sample_df = df.select(column).sample(
                    fraction=fraction,
                    seed=seed,
                )
                data = sample_df.toPandas()[column].values

            return data.astype(int) if context.is_discrete else data.astype(float)
        except Exception as e:
            raise RuntimeError(
                f"Failed to recreate sample from source DataFrame. "
                f"The DataFrame may have been unpersisted. "
                f"Call materialize() before unpersisting if you need lazy metrics. "
                f"Original error: {e}"
            ) from e

    def _compute_lazy_metrics_for_results(
        self,
        rows: list,
        context: LazyMetricsContext,
    ) -> List["DistributionFitResult"]:
        """Compute lazy metrics for a batch of result rows.

        Recreates the sample once and computes KS/AD for all distributions
        in the batch.

        Args:
            rows: List of Spark Row objects with distribution fit results
            context: LazyMetricsContext for the column

        Returns:
            List of DistributionFitResult with computed metrics
        """
        # Import appropriate metric computation function
        if context.is_discrete:
            from spark_bestfit.discrete_fitting import compute_ks_ad_metrics_discrete as compute_metrics
        else:
            from spark_bestfit.fitting import compute_ks_ad_metrics as compute_metrics

        # Recreate sample once for all distributions
        data_sample = self._recreate_sample(context)

        def _get_row_value(row, key, default=None):
            """Helper to get value from row (Spark Row or dict)."""
            if self._is_spark:
                return row[key] if key in row else default
            else:
                return row.get(key, default)

        results = []
        for row in rows:
            # Compute metrics for this distribution
            ks_stat, pvalue, ad_stat, ad_pvalue = compute_metrics(
                dist_name=_get_row_value(row, "distribution"),
                params=list(_get_row_value(row, "parameters", [])),
                data_sample=data_sample,
                lower_bound=context.lower_bound,
                upper_bound=context.upper_bound,
            )

            # Create result with computed metrics
            results.append(
                DistributionFitResult(
                    distribution=_get_row_value(row, "distribution"),
                    parameters=list(_get_row_value(row, "parameters", [])),
                    sse=_get_row_value(row, "sse"),
                    column_name=_get_row_value(row, "column_name"),
                    aic=_get_row_value(row, "aic"),
                    bic=_get_row_value(row, "bic"),
                    ks_statistic=ks_stat,
                    pvalue=pvalue,
                    ad_statistic=ad_stat,
                    ad_pvalue=ad_pvalue,
                    data_min=_get_row_value(row, "data_min"),
                    data_max=_get_row_value(row, "data_max"),
                    data_mean=_get_row_value(row, "data_mean"),
                    data_stddev=_get_row_value(row, "data_stddev"),
                    data_count=_get_row_value(row, "data_count"),
                    lower_bound=_get_row_value(row, "lower_bound"),
                    upper_bound=_get_row_value(row, "upper_bound"),
                )
            )

        return results

    @property
    def df(self) -> DataFrame:
        """Get underlying Spark DataFrame.

        Returns:
            Spark DataFrame with results
        """
        return self._df

    @abstractmethod
    def best(
        self,
        n: int = 1,
        metric: MetricName = "ks_statistic",
        warn_if_poor: bool = False,
        pvalue_threshold: float = 0.05,
    ) -> List[DistributionFitResult]:
        """Get top n distributions by specified metric.

        Args:
            n: Number of results to return
            metric: Metric to sort by ('ks_statistic', 'sse', 'aic', 'bic', or 'ad_statistic').
                Defaults to 'ks_statistic' (Kolmogorov-Smirnov statistic).
            warn_if_poor: If True, emit a warning when the best fit has a p-value
                below pvalue_threshold, indicating a potentially poor fit.
            pvalue_threshold: P-value threshold for poor fit warning (default 0.05).
                Only used when warn_if_poor=True.

        Returns:
            List of DistributionFitResult objects

        Example:
            >>> best = results.best(n=1)[0]
            >>> top_5 = results.best(n=5, metric='aic')
        """
        pass

    def _best_from_dataframe(
        self,
        n: int,
        metric: MetricName,
        warn_if_poor: bool,
        pvalue_threshold: float,
    ) -> List[DistributionFitResult]:
        """Shared helper to get best results from DataFrame.

        Used by both EagerFitResults and LazyFitResults for the common
        sort-and-return logic.
        """
        # Validate inputs
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")

        valid_metrics = {"sse", "aic", "bic", "ks_statistic", "ad_statistic"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        # Get top N results sorted by metric (ascending, nulls last)
        if self._is_spark:
            top_n = self._df.orderBy(F.col(metric).asc_nulls_last()).limit(n).collect()
        else:
            # pandas: sort by metric, NaN values go to end with na_position='last'
            sorted_df = self._df.sort_values(by=metric, ascending=True, na_position="last")
            top_n = sorted_df.head(n).to_dict("records")

        def _get_row_value(row, key, default=None):
            """Helper to get value from row (Spark Row or dict)."""
            if self._is_spark:
                return row[key] if key in row else default
            else:
                return row.get(key, default)

        results = [
            DistributionFitResult(
                distribution=_get_row_value(row, "distribution"),
                parameters=list(_get_row_value(row, "parameters", [])),
                sse=_get_row_value(row, "sse"),
                column_name=_get_row_value(row, "column_name"),
                aic=_get_row_value(row, "aic"),
                bic=_get_row_value(row, "bic"),
                ks_statistic=_get_row_value(row, "ks_statistic"),
                pvalue=_get_row_value(row, "pvalue"),
                ad_statistic=_get_row_value(row, "ad_statistic"),
                ad_pvalue=_get_row_value(row, "ad_pvalue"),
                data_min=_get_row_value(row, "data_min"),
                data_max=_get_row_value(row, "data_max"),
                data_mean=_get_row_value(row, "data_mean"),
                data_stddev=_get_row_value(row, "data_stddev"),
                data_count=_get_row_value(row, "data_count"),
                lower_bound=_get_row_value(row, "lower_bound"),
                upper_bound=_get_row_value(row, "upper_bound"),
            )
            for row in top_n
        ]

        # Emit warning if requested and best fit has poor p-value
        if warn_if_poor and results:
            best_result = results[0]
            if best_result.pvalue is not None and best_result.pvalue < pvalue_threshold:
                warnings.warn(
                    f"Best fit '{best_result.distribution}' has p-value {best_result.pvalue:.4f} "
                    f"< {pvalue_threshold}, indicating a potentially poor fit. "
                    f"Consider using quality_report() for detailed diagnostics.",
                    UserWarning,
                    stacklevel=2,
                )

        return results

    @abstractmethod
    def filter(
        self,
        sse_threshold: Optional[float] = None,
        aic_threshold: Optional[float] = None,
        bic_threshold: Optional[float] = None,
        ks_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
        ad_threshold: Optional[float] = None,
    ) -> "BaseFitResults":
        """Filter results by metric thresholds.

        Args:
            sse_threshold: Maximum SSE to include
            aic_threshold: Maximum AIC to include
            bic_threshold: Maximum BIC to include
            ks_threshold: Maximum K-S statistic to include
            pvalue_threshold: Minimum p-value to include (higher = better fit)
            ad_threshold: Maximum A-D statistic to include

        Returns:
            New FitResults with filtered data (same type as self)

        Example:
            >>> good_fits = results.filter(sse_threshold=0.01)
        """
        pass

    def _filter_dataframe(
        self,
        sse_threshold: Optional[float] = None,
        aic_threshold: Optional[float] = None,
        bic_threshold: Optional[float] = None,
        ks_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
        ad_threshold: Optional[float] = None,
    ) -> Union[DataFrame, pd.DataFrame]:
        """Shared helper to filter the DataFrame by thresholds.

        Returns the filtered DataFrame for subclasses to wrap appropriately.
        """
        filtered = self._df

        if self._is_spark:
            # Spark DataFrame filtering
            if sse_threshold is not None:
                filtered = filtered.filter(F.col("sse") < sse_threshold)
            if aic_threshold is not None:
                filtered = filtered.filter(F.col("aic") < aic_threshold)
            if bic_threshold is not None:
                filtered = filtered.filter(F.col("bic") < bic_threshold)
            if ks_threshold is not None:
                filtered = filtered.filter(F.col("ks_statistic") < ks_threshold)
            if pvalue_threshold is not None:
                filtered = filtered.filter(F.col("pvalue") > pvalue_threshold)
            if ad_threshold is not None:
                filtered = filtered.filter(F.col("ad_statistic") < ad_threshold)
        else:
            # pandas DataFrame filtering
            if sse_threshold is not None:
                filtered = filtered[filtered["sse"] < sse_threshold]
            if aic_threshold is not None:
                filtered = filtered[filtered["aic"] < aic_threshold]
            if bic_threshold is not None:
                filtered = filtered[filtered["bic"] < bic_threshold]
            if ks_threshold is not None:
                filtered = filtered[filtered["ks_statistic"] < ks_threshold]
            if pvalue_threshold is not None:
                filtered = filtered[filtered["pvalue"] > pvalue_threshold]
            if ad_threshold is not None:
                filtered = filtered[filtered["ad_statistic"] < ad_threshold]

        return filtered

    @abstractmethod
    def for_column(self, column_name: str) -> "BaseFitResults":
        """Filter results to a single column.

        Args:
            column_name: Column to filter for

        Returns:
            New FitResults containing only results for the specified column
            (same type as self).

        Example:
            >>> col1_results = results.for_column("col1")
        """
        pass

    def _filter_for_column(self, column_name: str) -> Union[DataFrame, pd.DataFrame]:
        """Shared helper to filter DataFrame to a single column.

        Returns the filtered DataFrame for subclasses to wrap appropriately.
        """
        if self._is_spark:
            return self._df.filter(F.col("column_name") == column_name)
        else:
            return self._df[self._df["column_name"] == column_name].copy()

    @property
    def column_names(self) -> List[str]:
        """Get list of unique column names in results.

        Returns:
            List of column names that have fit results

        Example:
            >>> results = fitter.fit(df, columns=["col1", "col2"])
            >>> print(results.column_names)
            ['col1', 'col2']
        """
        # Check if column_name column exists and has non-null values
        if "column_name" not in self._df.columns:
            return []

        if self._is_spark:
            rows = self._df.select("column_name").distinct().filter(F.col("column_name").isNotNull()).collect()
            return [row["column_name"] for row in rows]
        else:
            # pandas: get unique non-null values
            unique_cols = self._df["column_name"].dropna().unique()
            return list(unique_cols)

    def best_per_column(
        self, n: int = 1, metric: MetricName = "ks_statistic"
    ) -> Dict[str, List["DistributionFitResult"]]:
        """Get top n distributions for each column.

        Args:
            n: Number of results per column
            metric: Metric to sort by ('ks_statistic', 'sse', 'aic', 'bic', or 'ad_statistic')

        Returns:
            Dict mapping column_name -> List[DistributionFitResult]

        Example:
            >>> results = fitter.fit(df, columns=["col1", "col2", "col3"])
            >>> best_per_col = results.best_per_column(n=1)
            >>> for col, fits in best_per_col.items():
            ...     print(f"{col}: {fits[0].distribution}")
        """
        result: Dict[str, List[DistributionFitResult]] = {}
        for col in self.column_names:
            result[col] = self.for_column(col).best(n=n, metric=metric)
        return result

    def summary(self) -> pd.DataFrame:
        """Get summary statistics of fit quality.

        Returns:
            DataFrame with min, mean, max for each metric

        Example:
            >>> results.summary()
                   min_sse  mean_sse  max_sse  min_ks  mean_ks  max_ks  min_ad  mean_ad  max_ad  count
            0      0.001     0.15      2.34    0.02    0.08     0.25    0.10    0.50     2.0      95
        """
        if self._is_spark:
            summary = self._df.select(
                F.min("sse").alias("min_sse"),
                F.mean("sse").alias("mean_sse"),
                F.max("sse").alias("max_sse"),
                F.min("aic").alias("min_aic"),
                F.mean("aic").alias("mean_aic"),
                F.max("aic").alias("max_aic"),
                F.min("ks_statistic").alias("min_ks"),
                F.mean("ks_statistic").alias("mean_ks"),
                F.max("ks_statistic").alias("max_ks"),
                F.min("pvalue").alias("min_pvalue"),
                F.mean("pvalue").alias("mean_pvalue"),
                F.max("pvalue").alias("max_pvalue"),
                F.min("ad_statistic").alias("min_ad"),
                F.mean("ad_statistic").alias("mean_ad"),
                F.max("ad_statistic").alias("max_ad"),
                F.count("*").alias("total_distributions"),
            ).toPandas()
        else:
            # pandas DataFrame
            df = self._df
            summary = pd.DataFrame(
                {
                    "min_sse": [df["sse"].min()],
                    "mean_sse": [df["sse"].mean()],
                    "max_sse": [df["sse"].max()],
                    "min_aic": [df["aic"].min()],
                    "mean_aic": [df["aic"].mean()],
                    "max_aic": [df["aic"].max()],
                    "min_ks": [df["ks_statistic"].min()],
                    "mean_ks": [df["ks_statistic"].mean()],
                    "max_ks": [df["ks_statistic"].max()],
                    "min_pvalue": [df["pvalue"].min()],
                    "mean_pvalue": [df["pvalue"].mean()],
                    "max_pvalue": [df["pvalue"].max()],
                    "min_ad": [df["ad_statistic"].min()],
                    "mean_ad": [df["ad_statistic"].mean()],
                    "max_ad": [df["ad_statistic"].max()],
                    "total_distributions": [len(df)],
                }
            )

        return summary

    def count(self) -> int:
        """Get number of fitted distributions.

        Returns:
            Count of distributions
        """
        if self._is_spark:
            return self._df.count()
        else:
            return len(self._df)

    def __len__(self) -> int:
        """Get number of fitted distributions."""
        return self.count()

    def quality_report(
        self,
        n: int = 5,
        pvalue_threshold: float = 0.05,
        ks_threshold: float = 0.10,
        ad_threshold: float = 2.0,
    ) -> Dict[str, Union[List[DistributionFitResult], Dict[str, float], List[str]]]:
        """Generate a quality assessment report for the fitting results.

        Provides a comprehensive view of fit quality including the top fits,
        summary statistics, and any quality concerns.

        Args:
            n: Number of top distributions to include (default 5)
            pvalue_threshold: Minimum p-value for acceptable fit (default 0.05)
            ks_threshold: Maximum K-S statistic for acceptable fit (default 0.10)
            ad_threshold: Maximum A-D statistic for acceptable fit (default 2.0)

        Returns:
            Dictionary with:
                - 'top_fits': List of top n DistributionFitResult objects
                - 'summary': Dict with summary statistics (min/max/mean for key metrics)
                - 'warnings': List of warning messages about fit quality
                - 'n_acceptable': Number of distributions meeting all thresholds

        Example:
            >>> report = results.quality_report()
            >>> print(f"Top fit: {report['top_fits'][0].distribution}")
            >>> print(f"Warnings: {report['warnings']}")
            >>> if report['warnings']:
            ...     print("Consider reviewing fit quality")
        """
        top_fits = self.best(n=n)
        warnings_list: List[str] = []

        # Get summary stats
        summary_df = self.summary()
        summary_dict = {
            "min_ks": float(summary_df["min_ks"].iloc[0]) if summary_df["min_ks"].iloc[0] is not None else None,
            "max_ks": float(summary_df["max_ks"].iloc[0]) if summary_df["max_ks"].iloc[0] is not None else None,
            "mean_ks": float(summary_df["mean_ks"].iloc[0]) if summary_df["mean_ks"].iloc[0] is not None else None,
            "min_pvalue": (
                float(summary_df["min_pvalue"].iloc[0]) if summary_df["min_pvalue"].iloc[0] is not None else None
            ),
            "max_pvalue": (
                float(summary_df["max_pvalue"].iloc[0]) if summary_df["max_pvalue"].iloc[0] is not None else None
            ),
            "mean_pvalue": (
                float(summary_df["mean_pvalue"].iloc[0]) if summary_df["mean_pvalue"].iloc[0] is not None else None
            ),
            "min_ad": float(summary_df["min_ad"].iloc[0]) if summary_df["min_ad"].iloc[0] is not None else None,
            "max_ad": float(summary_df["max_ad"].iloc[0]) if summary_df["max_ad"].iloc[0] is not None else None,
            "total_distributions": int(summary_df["total_distributions"].iloc[0]),
        }

        # Count acceptable fits
        if self._is_spark:
            acceptable_filter = self._df
            acceptable_filter = acceptable_filter.filter(F.col("pvalue") >= pvalue_threshold)
            acceptable_filter = acceptable_filter.filter(F.col("ks_statistic") <= ks_threshold)
            # Only filter by A-D if values exist
            if summary_dict["min_ad"] is not None:
                acceptable_filter = acceptable_filter.filter(
                    (F.col("ad_statistic").isNull()) | (F.col("ad_statistic") <= ad_threshold)
                )
            n_acceptable = acceptable_filter.count()
        else:
            # pandas DataFrame
            acceptable = self._df[(self._df["pvalue"] >= pvalue_threshold) & (self._df["ks_statistic"] <= ks_threshold)]
            if summary_dict["min_ad"] is not None:
                acceptable = acceptable[
                    acceptable["ad_statistic"].isna() | (acceptable["ad_statistic"] <= ad_threshold)
                ]
            n_acceptable = len(acceptable)

        # Generate warnings
        if top_fits:
            best = top_fits[0]
            if best.pvalue is not None and best.pvalue < pvalue_threshold:
                warnings_list.append(
                    f"Best fit '{best.distribution}' has low p-value ({best.pvalue:.4f} < {pvalue_threshold})"
                )
            if best.ks_statistic is not None and best.ks_statistic > ks_threshold:
                warnings_list.append(
                    f"Best fit '{best.distribution}' has high K-S statistic ({best.ks_statistic:.4f} > {ks_threshold})"
                )
            if best.ad_statistic is not None and best.ad_statistic > ad_threshold:
                warnings_list.append(
                    f"Best fit '{best.distribution}' has high A-D statistic ({best.ad_statistic:.4f} > {ad_threshold})"
                )

        if n_acceptable == 0:
            warnings_list.append("No distributions meet all quality thresholds")
        elif n_acceptable < 3:
            warnings_list.append(f"Only {n_acceptable} distribution(s) meet quality thresholds")

        return {
            "top_fits": top_fits,
            "summary": summary_dict,
            "warnings": warnings_list,
            "n_acceptable": n_acceptable,
        }

    def __repr__(self) -> str:
        """String representation of results."""
        count = self.count()
        class_name = self.__class__.__name__
        if count > 0:
            best = self.best(n=1)[0]
            ks_str = f"{best.ks_statistic:.6f}" if best.ks_statistic is not None else "N/A"
            return f"{class_name}({count} distributions fitted, " f"best: {best.distribution} with KS={ks_str})"
        return f"{class_name}({count} distributions fitted)"


class EagerFitResults(BaseFitResults):
    """Fit results with all metrics pre-computed.

    This class represents distribution fit results where all metrics
    (SSE, AIC, BIC, KS, AD) have been computed during fitting.

    Example:
        >>> results = fitter.fit(df, 'value')  # Default: eager evaluation
        >>> best = results.best(n=1)[0]
        >>> print(f"KS: {best.ks_statistic:.4f}")
    """

    @property
    def is_lazy(self) -> Literal[False]:
        """Return False - eager results have all metrics computed."""
        return False

    def materialize(self) -> "EagerFitResults":
        """Return self - already materialized.

        For eager results, this is a no-op since all metrics are
        already computed.

        Returns:
            Self (no copy needed).
        """
        return self

    def best(
        self,
        n: int = 1,
        metric: MetricName = "ks_statistic",
        warn_if_poor: bool = False,
        pvalue_threshold: float = 0.05,
    ) -> List[DistributionFitResult]:
        """Get top n distributions by specified metric.

        Args:
            n: Number of results to return
            metric: Metric to sort by ('ks_statistic', 'sse', 'aic', 'bic', or 'ad_statistic')
            warn_if_poor: If True, warn when best fit has poor p-value
            pvalue_threshold: P-value threshold for poor fit warning

        Returns:
            List of DistributionFitResult objects
        """
        return self._best_from_dataframe(n, metric, warn_if_poor, pvalue_threshold)

    def filter(
        self,
        sse_threshold: Optional[float] = None,
        aic_threshold: Optional[float] = None,
        bic_threshold: Optional[float] = None,
        ks_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
        ad_threshold: Optional[float] = None,
    ) -> "EagerFitResults":
        """Filter results by metric thresholds.

        Args:
            sse_threshold: Maximum SSE to include
            aic_threshold: Maximum AIC to include
            bic_threshold: Maximum BIC to include
            ks_threshold: Maximum K-S statistic to include
            pvalue_threshold: Minimum p-value to include
            ad_threshold: Maximum A-D statistic to include

        Returns:
            New EagerFitResults with filtered data
        """
        filtered_df = self._filter_dataframe(
            sse_threshold, aic_threshold, bic_threshold, ks_threshold, pvalue_threshold, ad_threshold
        )
        return EagerFitResults(filtered_df)

    def for_column(self, column_name: str) -> "EagerFitResults":
        """Filter results to a single column.

        Args:
            column_name: Column to filter for

        Returns:
            New EagerFitResults for the specified column
        """
        filtered_df = self._filter_for_column(column_name)
        return EagerFitResults(filtered_df)


class LazyFitResults(BaseFitResults):
    """Fit results with lazy KS/AD metric computation.

    This class represents distribution fit results where only fast metrics
    (SSE, AIC, BIC) are pre-computed. KS and AD statistics are computed
    on-demand when first accessed via best() with those metrics.

    Important:
        The source DataFrame must remain valid (not unpersisted) for lazy
        metric computation to work. Call materialize() before unpersisting
        the source DataFrame if you need the metrics later.

    Example:
        >>> results = fitter.fit(df, 'value', lazy_metrics=True)
        >>> best_aic = results.best(n=1, metric='aic')[0]  # Fast
        >>> best_ks = results.best(n=1, metric='ks_statistic')[0]  # Computes on-demand
        >>>
        >>> # Before unpersisting source, materialize all metrics
        >>> materialized = results.materialize()
        >>> df.unpersist()  # Safe now
    """

    def __init__(
        self,
        results_df: Union[DataFrame, pd.DataFrame],
        lazy_contexts: Dict[str, LazyMetricsContext],
    ):
        """Initialize LazyFitResults.

        Args:
            results_df: Spark DataFrame or pandas DataFrame with fit results
            lazy_contexts: Dict mapping column names to LazyMetricsContext
                for on-demand KS/AD computation. Required (not optional).
        """
        super().__init__(results_df)
        self._lazy_contexts = lazy_contexts

    @property
    def is_lazy(self) -> Literal[True]:
        """Return True - lazy results have deferred metric computation."""
        return True

    @property
    def source_dataframes(self) -> Dict[str, DataFrame]:
        """Get source DataFrames for lifecycle visibility.

        Use this to understand what DataFrames the lazy computation depends on.

        Returns:
            Dict mapping column names to their source DataFrames.
        """
        return {k: v.source_df for k, v in self._lazy_contexts.items()}

    def is_source_available(self) -> bool:
        """Check if source DataFrames are still accessible.

        Use this to verify that lazy metric computation can still succeed.

        Returns:
            True if all source DataFrames can be accessed, False otherwise.
        """
        try:
            for context in self._lazy_contexts.values():
                # Attempt a lightweight operation to validate availability
                if hasattr(context.source_df, "schema"):
                    # Spark DataFrame - just access schema
                    _ = context.source_df.schema
                elif hasattr(context.source_df, "columns"):
                    # pandas DataFrame
                    _ = len(context.source_df.columns)
            return True
        except Exception:
            return False

    def materialize(self) -> EagerFitResults:
        """Force computation of all lazy metrics.

        Computes KS and AD statistics for all distributions, returning
        an EagerFitResults that no longer depends on the source DataFrame.

        Returns:
            EagerFitResults with all metrics computed.

        Raises:
            RuntimeError: If the source DataFrame is no longer available.
        """
        # Collect all rows - handle both Spark and pandas DataFrames
        if self._is_spark:
            all_rows = self._df.collect()
        else:
            all_rows = self._df.to_dict("records")

        column_names = self.column_names if self.column_names else [None]

        # Group rows by column
        rows_by_column: Dict[Optional[str], list] = {}
        for row in all_rows:
            if self._is_spark:
                col = row["column_name"] if hasattr(row, "column_name") else None
            else:
                col = row.get("column_name")
            if col not in rows_by_column:
                rows_by_column[col] = []
            rows_by_column[col].append(row)

        # Compute metrics for each column
        materialized_results: List[Dict] = []

        for col_name in column_names:
            context_key = col_name or "_single_column_"
            if context_key not in self._lazy_contexts:
                if self._lazy_contexts:
                    context_key = next(iter(self._lazy_contexts.keys()))
                else:
                    # No context, just pass through
                    for row in rows_by_column.get(col_name, []):
                        if self._is_spark:
                            materialized_results.append(dict(row.asDict()))
                        else:
                            materialized_results.append(dict(row))
                    continue

            context = self._lazy_contexts[context_key]
            data_sample = self._recreate_sample(context)

            # Select appropriate metric computation function
            if context.is_discrete:
                from spark_bestfit.discrete_fitting import compute_ks_ad_metrics_discrete as compute_metrics
            else:
                from spark_bestfit.fitting import compute_ks_ad_metrics as compute_metrics

            for row in rows_by_column.get(col_name, []):
                if self._is_spark:
                    row_dict = dict(row.asDict())
                else:
                    row_dict = dict(row)

                # Compute metrics if they're None
                if row_dict.get("ks_statistic") is None:
                    ks_stat, pvalue, ad_stat, ad_pvalue = compute_metrics(
                        dist_name=row_dict["distribution"],
                        params=list(row_dict["parameters"]),
                        data_sample=data_sample,
                        lower_bound=context.lower_bound,
                        upper_bound=context.upper_bound,
                    )
                    row_dict["ks_statistic"] = ks_stat
                    row_dict["pvalue"] = pvalue
                    row_dict["ad_statistic"] = ad_stat
                    row_dict["ad_pvalue"] = ad_pvalue

                materialized_results.append(row_dict)

        # Create new DataFrame from materialized results
        if self._is_spark:
            from spark_bestfit.fitting import FIT_RESULT_SCHEMA

            spark = self._df.sparkSession
            materialized_df = spark.createDataFrame(materialized_results, schema=FIT_RESULT_SCHEMA)
            return EagerFitResults(materialized_df.cache())
        else:
            materialized_df = pd.DataFrame(materialized_results)
            return EagerFitResults(materialized_df)

    def best(
        self,
        n: int = 1,
        metric: MetricName = "ks_statistic",
        warn_if_poor: bool = False,
        pvalue_threshold: float = 0.05,
    ) -> List[DistributionFitResult]:
        """Get top n distributions by specified metric.

        For KS and AD metrics, computation happens on-demand using the
        stored lazy context.

        Args:
            n: Number of results to return
            metric: Metric to sort by ('ks_statistic', 'sse', 'aic', 'bic', or 'ad_statistic')
            warn_if_poor: If True, warn when best fit has poor p-value
            pvalue_threshold: P-value threshold for poor fit warning

        Returns:
            List of DistributionFitResult objects
        """
        # Validate inputs
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")

        valid_metrics = {"sse", "aic", "bic", "ks_statistic", "ad_statistic"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        # For lazy metrics (KS/AD), compute on-demand
        lazy_metric_names = {"ks_statistic", "ad_statistic"}
        if metric in lazy_metric_names:
            # Check if the first row has the metric as None (lazy mode)
            if self._is_spark:
                sample_row = self._df.limit(1).collect()
                first_metric_value = sample_row[0][metric] if sample_row else None
            else:
                first_metric_value = self._df[metric].iloc[0] if len(self._df) > 0 else None

            if first_metric_value is None or (isinstance(first_metric_value, float) and np.isnan(first_metric_value)):
                return self._best_with_lazy_computation(n, metric, warn_if_poor, pvalue_threshold)

        # Fall back to standard DataFrame query
        return self._best_from_dataframe(n, metric, warn_if_poor, pvalue_threshold)

    def _best_with_lazy_computation(
        self,
        n: int,
        metric: MetricName,
        warn_if_poor: bool,
        pvalue_threshold: float,
    ) -> List[DistributionFitResult]:
        """Get best distributions with on-demand KS/AD computation.

        Computes metrics only for top N*3+5 candidates (sorted by AIC as proxy),
        then re-sorts by the actual requested metric.
        """
        total_count = self._df.count() if self._is_spark else len(self._df)
        candidate_count = min(n * 3 + 5, total_count)

        column_names = self.column_names if self.column_names else [None]
        all_results: List[DistributionFitResult] = []

        for col_name in column_names:
            context_key = col_name or "_single_column_"
            if context_key not in self._lazy_contexts:
                if self._lazy_contexts:
                    context_key = next(iter(self._lazy_contexts.keys()))
                else:
                    continue

            context = self._lazy_contexts[context_key]

            # Get candidate rows sorted by AIC (proxy for good fits)
            if self._is_spark:
                if col_name:
                    candidates_df = self._df.filter(F.col("column_name") == col_name)
                else:
                    candidates_df = self._df
                candidate_rows = candidates_df.orderBy(F.col("aic").asc_nulls_last()).limit(candidate_count).collect()
            else:
                if col_name:
                    candidates_df = self._df[self._df["column_name"] == col_name]
                else:
                    candidates_df = self._df
                sorted_df = candidates_df.sort_values(by="aic", ascending=True, na_position="last")
                candidate_rows = sorted_df.head(candidate_count).to_dict("records")

            # Compute lazy metrics for candidates
            computed_results = self._compute_lazy_metrics_for_results(candidate_rows, context)
            all_results.extend(computed_results)

        # Sort by the requested metric
        if metric == "ks_statistic":
            all_results.sort(key=lambda r: r.ks_statistic if r.ks_statistic is not None else float("inf"))
        else:
            all_results.sort(key=lambda r: r.ad_statistic if r.ad_statistic is not None else float("inf"))

        results = all_results[:n]

        # Emit warning if requested
        if warn_if_poor and results:
            best_result = results[0]
            if best_result.pvalue is not None and best_result.pvalue < pvalue_threshold:
                warnings.warn(
                    f"Best fit '{best_result.distribution}' has p-value {best_result.pvalue:.4f} "
                    f"< {pvalue_threshold}, indicating a potentially poor fit. "
                    f"Consider using quality_report() for detailed diagnostics.",
                    UserWarning,
                    stacklevel=2,
                )

        return results

    def filter(
        self,
        sse_threshold: Optional[float] = None,
        aic_threshold: Optional[float] = None,
        bic_threshold: Optional[float] = None,
        ks_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
        ad_threshold: Optional[float] = None,
    ) -> "LazyFitResults":
        """Filter results by metric thresholds.

        Note:
            Filtering by KS/AD thresholds with lazy metrics will exclude all
            results since those metrics are None. Use AIC/BIC/SSE thresholds
            or call materialize() first.

        Returns:
            New LazyFitResults with filtered data (preserves lazy contexts)
        """
        # Warn if filtering by lazy metrics
        lazy_filter_requested = ks_threshold is not None or pvalue_threshold is not None or ad_threshold is not None
        if lazy_filter_requested and self._is_spark:
            sample_row = self._df.limit(1).collect()
            if sample_row and sample_row[0]["ks_statistic"] is None:
                warnings.warn(
                    "Filtering by KS/AD metrics when lazy_metrics=True was used during fitting. "
                    "These metrics are None, so filtering will exclude all results. "
                    "Use aic/bic/sse thresholds instead, or call materialize() first.",
                    UserWarning,
                    stacklevel=2,
                )

        filtered_df = self._filter_dataframe(
            sse_threshold, aic_threshold, bic_threshold, ks_threshold, pvalue_threshold, ad_threshold
        )
        return LazyFitResults(filtered_df, lazy_contexts=self._lazy_contexts)

    def for_column(self, column_name: str) -> "LazyFitResults":
        """Filter results to a single column.

        Args:
            column_name: Column to filter for

        Returns:
            New LazyFitResults for the specified column (preserves lazy context)
        """
        filtered_df = self._filter_for_column(column_name)

        # Preserve only the relevant lazy context for this column
        filtered_contexts = {}
        if column_name in self._lazy_contexts:
            filtered_contexts[column_name] = self._lazy_contexts[column_name]

        return LazyFitResults(filtered_df, lazy_contexts=filtered_contexts)


# Type alias for annotations
FitResultsType = Union[EagerFitResults, LazyFitResults]


def create_fit_results(
    results_df: Union[DataFrame, pd.DataFrame],
    lazy_contexts: Optional[Dict[str, LazyMetricsContext]] = None,
) -> FitResultsType:
    """Factory function for creating FitResults.

    Creates the appropriate FitResults variant based on whether lazy contexts
    are provided.

    Args:
        results_df: Spark DataFrame or pandas DataFrame with fit results
        lazy_contexts: Optional dict mapping column names to LazyMetricsContext
            for on-demand KS/AD computation

    Returns:
        LazyFitResults if lazy_contexts provided, EagerFitResults otherwise

    Example:
        >>> # From fitter (automatic)
        >>> results = fitter.fit(df, 'value')  # Returns EagerFitResults
        >>> lazy = fitter.fit(df, 'value', lazy_metrics=True)  # Returns LazyFitResults
        >>>
        >>> # Direct construction (rare)
        >>> eager = create_fit_results(df)  # EagerFitResults
        >>> lazy = create_fit_results(df, lazy_contexts={...})  # LazyFitResults
    """
    if lazy_contexts:
        return LazyFitResults(results_df, lazy_contexts)
    return EagerFitResults(results_df)


# Backward-compatible alias (PascalCase to match original class name)
# This allows existing code `FitResults(df)` to continue working
FitResults = create_fit_results
