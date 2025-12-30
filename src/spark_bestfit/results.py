"""Results handling for fitted distributions."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import scipy.stats as st
from pyspark.sql import DataFrame

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

# Type alias for valid metric names (for IDE autocomplete and type checking)
MetricName = Literal["sse", "aic", "bic", "ks_statistic", "ad_statistic"]


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
        data_summary: Optional summary statistics of the original data (sample_size,
            min, max, mean, std). Captured during fitting to aid debugging and
            provenance tracking.

    Note:
        The p-value from the KS test is approximate when parameters are
        estimated from the same data being tested. It tends to be conservative
        (larger than it should be). Use it for rough guidance, not strict
        hypothesis testing. The ks_statistic is valid for ranking fits.

        The ad_pvalue is only available for 5 distributions (norm, expon,
        logistic, gumbel_r, gumbel_l) where scipy has critical value tables.
        For other distributions, ad_pvalue will be None but ad_statistic
        is still valid for ranking fits.
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
    data_summary: Optional[Dict[str, float]] = None

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
            "data_summary": self.data_summary,
        }

    def get_scipy_dist(self):
        """Get scipy distribution object.

        Returns:
            scipy.stats distribution object
        """
        return getattr(st, self.distribution)

    def sample(self, size: int = 1000, random_state: Optional[int] = None) -> np.ndarray:
        """Generate random samples from the fitted distribution.

        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of random samples

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> samples = result.sample(size=10000, random_state=42)
        """
        dist = self.get_scipy_dist()
        # Parameters are all positional: (shape params..., loc, scale)
        return dist.rvs(*self.parameters, size=size, random_state=random_state)

    def sample_spark(
        self,
        n: int,
        spark: Optional["SparkSession"] = None,
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
        column_name: str = "sample",
    ) -> DataFrame:
        """Generate distributed samples from the fitted distribution using Spark.

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
        from spark_bestfit.sampling import sample_spark

        return sample_spark(
            distribution=self.distribution,
            parameters=self.parameters,
            n=n,
            spark=spark,
            num_partitions=num_partitions,
            random_seed=random_seed,
            column_name=column_name,
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate probability density function at given points.

        Args:
            x: Points at which to evaluate PDF

        Returns:
            PDF values at x

        Example:
            >>> result = fitter.fit(df, 'value').best(n=1)[0]
            >>> x = np.linspace(0, 10, 100)
            >>> y = result.pdf(x)
        """
        dist = self.get_scipy_dist()
        # Parameters are all positional: (shape params..., loc, scale)
        return dist.pdf(x, *self.parameters)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate cumulative distribution function at given points.

        Args:
            x: Points at which to evaluate CDF

        Returns:
            CDF values at x
        """
        dist = self.get_scipy_dist()
        # Parameters are all positional: (shape params..., loc, scale)
        return dist.cdf(x, *self.parameters)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Evaluate percent point function (inverse CDF) at given quantiles.

        Args:
            q: Quantiles at which to evaluate PPF (0 to 1)

        Returns:
            PPF values at q
        """
        dist = self.get_scipy_dist()
        # Parameters are all positional: (shape params..., loc, scale)
        return dist.ppf(q, *self.parameters)

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
        return (
            f"DistributionFitResult({col_str}distribution='{self.distribution}', "
            f"sse={self.sse:.6f}, aic={aic_str}, bic={bic_str}, "
            f"ks_statistic={ks_str}, pvalue={pval_str}, "
            f"ad_statistic={ad_str}, ad_pvalue={ad_pval_str}, "
            f"parameters=[{param_str}])"
        )


class FitResults:
    """Container for multiple distribution fit results.

    Provides convenient methods for accessing, filtering, and analyzing
    fitted distributions. Wraps a Spark DataFrame but provides pandas-like
    interface for common operations.

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

    def __init__(self, results_df: DataFrame):
        """Initialize FitResults.

        Args:
            results_df: Spark DataFrame with fit results
        """
        self._df = results_df

    @property
    def df(self) -> DataFrame:
        """Get underlying Spark DataFrame.

        Returns:
            Spark DataFrame with results
        """
        return self._df

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
            >>> # Get best distribution (by K-S statistic, the default)
            >>> best = results.best(n=1)[0]
            >>> # Get top 5 by AIC
            >>> top_5 = results.best(n=5, metric='aic')
            >>> # Get best by SSE
            >>> best_sse = results.best(n=1, metric='sse')[0]
            >>> # Get best by Anderson-Darling statistic
            >>> best_ad = results.best(n=1, metric='ad_statistic')[0]
            >>> # Get best with warning if poor fit
            >>> best = results.best(n=1, warn_if_poor=True)[0]
        """
        valid_metrics = {"sse", "aic", "bic", "ks_statistic", "ad_statistic"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        top_n = self._df.orderBy(metric).limit(n).collect()

        results = [
            DistributionFitResult(
                distribution=row["distribution"],
                parameters=list(row["parameters"]),
                sse=row["sse"],
                column_name=row["column_name"] if "column_name" in row else None,
                aic=row["aic"],
                bic=row["bic"],
                ks_statistic=row["ks_statistic"],
                pvalue=row["pvalue"],
                ad_statistic=row["ad_statistic"],
                ad_pvalue=row["ad_pvalue"],
                data_summary=dict(row["data_summary"]) if "data_summary" in row and row["data_summary"] else None,
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

    def filter(
        self,
        sse_threshold: Optional[float] = None,
        aic_threshold: Optional[float] = None,
        bic_threshold: Optional[float] = None,
        ks_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
        ad_threshold: Optional[float] = None,
    ) -> "FitResults":
        """Filter results by metric thresholds.

        Args:
            sse_threshold: Maximum SSE to include
            aic_threshold: Maximum AIC to include
            bic_threshold: Maximum BIC to include
            ks_threshold: Maximum K-S statistic to include
            pvalue_threshold: Minimum p-value to include (higher = better fit)
            ad_threshold: Maximum A-D statistic to include

        Returns:
            New FitResults with filtered data

        Example:
            >>> # Get only good fits
            >>> good_fits = results.filter(sse_threshold=0.01)
            >>> # Get models with low AIC
            >>> low_aic = results.filter(aic_threshold=1000)
            >>> # Get fits with p-value > 0.05
            >>> significant = results.filter(pvalue_threshold=0.05)
            >>> # Get fits with A-D statistic < 1.0
            >>> good_ad = results.filter(ad_threshold=1.0)
        """
        filtered = self._df

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

        return FitResults(filtered.cache())

    def for_column(self, column_name: str) -> "FitResults":
        """Filter results to a single column.

        Args:
            column_name: Column to filter for

        Returns:
            New FitResults containing only results for the specified column

        Example:
            >>> results = fitter.fit(df, columns=["col1", "col2"])
            >>> col1_results = results.for_column("col1")
            >>> best = col1_results.best(n=1)[0]
        """
        filtered = self._df.filter(F.col("column_name") == column_name)
        return FitResults(filtered.cache())

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
        rows = self._df.select("column_name").distinct().filter(F.col("column_name").isNotNull()).collect()
        return [row["column_name"] for row in rows]

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

        return summary

    def count(self) -> int:
        """Get number of fitted distributions.

        Returns:
            Count of distributions
        """
        return self._df.count()

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
        acceptable_filter = self._df
        acceptable_filter = acceptable_filter.filter(F.col("pvalue") >= pvalue_threshold)
        acceptable_filter = acceptable_filter.filter(F.col("ks_statistic") <= ks_threshold)
        # Only filter by A-D if values exist
        if summary_dict["min_ad"] is not None:
            acceptable_filter = acceptable_filter.filter(
                (F.col("ad_statistic").isNull()) | (F.col("ad_statistic") <= ad_threshold)
            )
        n_acceptable = acceptable_filter.count()

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
        if count > 0:
            best = self.best(n=1)[0]
            return (
                f"FitResults({count} distributions fitted, "
                f"best: {best.distribution} with KS={best.ks_statistic:.6f})"
            )
        return f"FitResults({count} distributions fitted)"
