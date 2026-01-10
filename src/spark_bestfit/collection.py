"""Collection classes for managing multiple distribution fit results.

This module contains the classes for storing, filtering, and analyzing
collections of distribution fit results. These provide convenient methods
for accessing, filtering, and comparing fitted distributions.

Classes:
    BaseFitResults: Abstract base class for fit result collections.
    EagerFitResults: Results with all metrics pre-computed.
    LazyFitResults: Results with lazy KS/AD metric computation.

Functions:
    create_fit_results: Factory function for creating FitResults.

Type Aliases:
    FitResultsType: Union of EagerFitResults and LazyFitResults.
"""

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from spark_bestfit.storage import (
    DEFAULT_AD_THRESHOLD,
    DEFAULT_KS_THRESHOLD,
    DEFAULT_PVALUE_THRESHOLD,
    FITTING_SAMPLE_SIZE,
    DistributionFitResult,
    LazyMetricsContext,
    MetricName,
)

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
    pass


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
        pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
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
        pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
        ks_threshold: float = DEFAULT_KS_THRESHOLD,
        ad_threshold: float = DEFAULT_AD_THRESHOLD,
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
        pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
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
        pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
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


# =============================================================================
# Type Aliases and Factory Functions
# =============================================================================

# Type alias for type annotations
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
