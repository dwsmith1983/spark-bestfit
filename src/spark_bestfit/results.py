"""Results handling for fitted distributions."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import scipy.stats as st
from pyspark.sql import DataFrame


@dataclass
class DistributionFitResult:
    """Result from fitting a single distribution.

    Attributes:
        distribution: Name of the scipy.stats distribution
        parameters: Fitted parameters (shape params + loc + scale)
        sse: Sum of Squared Errors
        aic: Akaike Information Criterion (lower is better)
        bic: Bayesian Information Criterion (lower is better)
        ks_statistic: Kolmogorov-Smirnov statistic (lower is better)
        pvalue: P-value from KS test (higher indicates better fit)

    Note:
        The p-value from the KS test is approximate when parameters are
        estimated from the same data being tested. It tends to be conservative
        (larger than it should be). Use it for rough guidance, not strict
        hypothesis testing. The ks_statistic is valid for ranking fits.
    """

    distribution: str
    parameters: List[float]
    sse: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    ks_statistic: Optional[float] = None
    pvalue: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "distribution": self.distribution,
            "parameters": self.parameters,
            "sse": self.sse,
            "aic": self.aic,
            "bic": self.bic,
            "ks_statistic": self.ks_statistic,
            "pvalue": self.pvalue,
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

    def __repr__(self) -> str:
        """String representation of the result."""
        param_str = ", ".join([f"{p:.4f}" for p in self.parameters])
        aic_str = f"{self.aic:.2f}" if self.aic is not None else "None"
        bic_str = f"{self.bic:.2f}" if self.bic is not None else "None"
        ks_str = f"{self.ks_statistic:.6f}" if self.ks_statistic is not None else "None"
        pval_str = f"{self.pvalue:.4f}" if self.pvalue is not None else "None"
        return (
            f"DistributionFitResult(distribution='{self.distribution}', "
            f"sse={self.sse:.6f}, aic={aic_str}, bic={bic_str}, "
            f"ks_statistic={ks_str}, pvalue={pval_str}, "
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

    def best(self, n: int = 1, metric: str = "ks_statistic") -> List[DistributionFitResult]:
        """Get top n distributions by specified metric.

        Args:
            n: Number of results to return
            metric: Metric to sort by ('ks_statistic', 'sse', 'aic', or 'bic').
                Defaults to 'ks_statistic' (Kolmogorov-Smirnov statistic).

        Returns:
            List of DistributionFitResult objects

        Example:
            >>> # Get best distribution (by K-S statistic, the default)
            >>> best = results.best(n=1)[0]
            >>> # Get top 5 by AIC
            >>> top_5 = results.best(n=5, metric='aic')
            >>> # Get best by SSE
            >>> best_sse = results.best(n=1, metric='sse')[0]
        """
        valid_metrics = {"sse", "aic", "bic", "ks_statistic"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        top_n = self._df.orderBy(metric).limit(n).collect()

        return [
            DistributionFitResult(
                distribution=row["distribution"],
                parameters=list(row["parameters"]),
                sse=row["sse"],
                aic=row["aic"],
                bic=row["bic"],
                ks_statistic=row["ks_statistic"],
                pvalue=row["pvalue"],
            )
            for row in top_n
        ]

    def filter(
        self,
        sse_threshold: Optional[float] = None,
        aic_threshold: Optional[float] = None,
        bic_threshold: Optional[float] = None,
        ks_threshold: Optional[float] = None,
        pvalue_threshold: Optional[float] = None,
    ) -> "FitResults":
        """Filter results by metric thresholds.

        Args:
            sse_threshold: Maximum SSE to include
            aic_threshold: Maximum AIC to include
            bic_threshold: Maximum BIC to include
            ks_threshold: Maximum K-S statistic to include
            pvalue_threshold: Minimum p-value to include (higher = better fit)

        Returns:
            New FitResults with filtered data

        Example:
            >>> # Get only good fits
            >>> good_fits = results.filter(sse_threshold=0.01)
            >>> # Get models with low AIC
            >>> low_aic = results.filter(aic_threshold=1000)
            >>> # Get fits with p-value > 0.05
            >>> significant = results.filter(pvalue_threshold=0.05)
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

        return FitResults(filtered)

    def summary(self) -> pd.DataFrame:
        """Get summary statistics of fit quality.

        Returns:
            DataFrame with min, mean, max for each metric

        Example:
            >>> results.summary()
                   min_sse  mean_sse  max_sse  min_ks  mean_ks  max_ks  count
            0      0.001     0.15      2.34    0.02    0.08     0.25     95
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

    def __repr__(self) -> str:
        """String representation of results."""
        count = self.count()
        if count > 0:
            best = self.best(n=1)[0]
            return f"FitResults({count} distributions fitted, " f"best: {best.distribution} with SSE={best.sse:.6f})"
        return f"FitResults({count} distributions fitted)"
