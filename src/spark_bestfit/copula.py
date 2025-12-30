"""Gaussian Copula for correlated multi-column sampling.

This module provides scalable copula modeling that works on massive DataFrames:
- Correlation computed via Spark ML (no .toPandas() required)
- Distributed sampling via sample_spark() for millions of correlated samples

Example:
    >>> from spark_bestfit import DistributionFitter, GaussianCopula
    >>>
    >>> # Fit multiple columns
    >>> fitter = DistributionFitter(spark)
    >>> results = fitter.fit(df, columns=["price", "quantity", "revenue"])
    >>>
    >>> # Fit copula - correlation computed via Spark ML
    >>> copula = GaussianCopula.fit(results, df)
    >>>
    >>> # Generate correlated samples
    >>> samples = copula.sample(n=10000)  # Dict[str, np.ndarray]
    >>> samples_df = copula.sample_spark(n=1_000_000, spark=spark)
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as st
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from spark_bestfit._version import __version__
from spark_bestfit.results import DistributionFitResult, MetricName
from spark_bestfit.serialization import SCHEMA_VERSION, SerializationError, detect_format
from spark_bestfit.utils import get_spark_session

if TYPE_CHECKING:
    from spark_bestfit.results import FitResults


@dataclass
class GaussianCopula:
    """Gaussian copula for generating correlated multi-column samples.

    Preserves both the marginal distributions (from fitting) and the
    correlation structure (from the original data) when generating samples.

    This implementation is designed for big data:
    - Correlation is computed via Spark ML, not .toPandas()
    - sample_spark() generates distributed samples across the cluster

    Attributes:
        column_names: List of column names in order
        marginals: Dict mapping column name to DistributionFitResult
        correlation_matrix: Spearman correlation matrix as numpy array

    Example:
        >>> copula = GaussianCopula.fit(results, df)
        >>> samples = copula.sample(n=10000)
        >>> samples_df = copula.sample_spark(n=1_000_000, spark=spark)
    """

    column_names: List[str]
    marginals: Dict[str, DistributionFitResult]
    correlation_matrix: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        """Validate copula state after initialization."""
        if len(self.column_names) < 2:
            raise ValueError("GaussianCopula requires at least 2 columns")

        if set(self.column_names) != set(self.marginals.keys()):
            raise ValueError("column_names must match marginals keys")

        n = len(self.column_names)
        if self.correlation_matrix.shape != (n, n):
            raise ValueError(f"correlation_matrix shape {self.correlation_matrix.shape} " f"doesn't match {n} columns")

    @classmethod
    def fit(
        cls,
        results: "FitResults",
        df: DataFrame,
        columns: Optional[List[str]] = None,
        metric: MetricName = "ks_statistic",
    ) -> "GaussianCopula":
        """Fit a Gaussian copula from multi-column fit results.

        Computes the Spearman correlation matrix using Spark ML's distributed
        computation - no .toPandas() required, scales to billions of rows.

        Args:
            results: FitResults from DistributionFitter.fit() with multiple columns
            df: Original Spark DataFrame used for fitting (for correlation computation)
            columns: Columns to include. Defaults to all columns in results.
            metric: Metric to use for selecting best distribution per column

        Returns:
            Fitted GaussianCopula instance

        Raises:
            ValueError: If fewer than 2 columns or columns not in results

        Example:
            >>> results = fitter.fit(df, columns=["price", "quantity", "revenue"])
            >>> copula = GaussianCopula.fit(results, df)
        """
        # Determine columns to use
        if columns is None:
            columns = results.column_names
            if not columns:
                raise ValueError(
                    "No columns found in results. " "Use fitter.fit(df, columns=[...]) for multi-column fitting."
                )

        if len(columns) < 2:
            raise ValueError("GaussianCopula requires at least 2 columns")

        # Verify columns exist in results
        available_columns = set(results.column_names)
        missing = set(columns) - available_columns
        if missing:
            raise ValueError(f"Columns not found in results: {missing}")

        # Get best marginal distribution for each column
        marginals: Dict[str, DistributionFitResult] = {}
        for col in columns:
            col_results = results.for_column(col)
            best = col_results.best(n=1, metric=metric)
            if not best:
                raise ValueError(f"No fit results for column '{col}'")
            marginals[col] = best[0]

        # Compute Spearman correlation via Spark ML (scales to billions of rows)
        correlation_matrix = cls._compute_correlation_spark(df, columns)

        return cls(
            column_names=list(columns),
            marginals=marginals,
            correlation_matrix=correlation_matrix,
        )

    @staticmethod
    def _compute_correlation_spark(df: DataFrame, columns: List[str]) -> np.ndarray:
        """Compute Spearman correlation matrix using Spark ML.

        This method uses distributed computation and doesn't require .toPandas(),
        enabling correlation computation on DataFrames with billions of rows.

        Args:
            df: Spark DataFrame
            columns: Columns to compute correlation for

        Returns:
            Correlation matrix as numpy array
        """
        # Assemble columns into a vector
        assembler = VectorAssembler(
            inputCols=columns,
            outputCol="_copula_features",
            handleInvalid="skip",  # Skip rows with nulls
        )
        vector_df = assembler.transform(df).select("_copula_features")

        # Compute Spearman correlation using Spark ML
        corr_result = Correlation.corr(vector_df, "_copula_features", method="spearman")

        # Extract correlation matrix from result
        corr_matrix = corr_result.head()[0].toArray()

        return corr_matrix

    def _get_frozen_dist(self, col: str) -> st.rv_continuous:
        """Get a frozen (pre-parameterized) scipy distribution for a column.

        Frozen distributions are cached for performance - avoids recreating
        distribution objects and re-parsing parameters on each call.

        Args:
            col: Column name

        Returns:
            Frozen scipy distribution with parameters bound
        """
        if not hasattr(self, "_frozen_dists"):
            self._frozen_dists: Dict[str, st.rv_continuous] = {}
        if col not in self._frozen_dists:
            marginal = self.marginals[col]
            dist = marginal.get_scipy_dist()
            self._frozen_dists[col] = dist(*marginal.parameters)
        return self._frozen_dists[col]

    def sample(
        self,
        n: int,
        random_state: Optional[int] = None,
        return_uniform: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Generate correlated samples locally.

        Uses the Gaussian copula to generate samples that preserve both:
        - Marginal distributions (from the fitted distributions)
        - Correlation structure (from the original data)

        For large sample sizes (>10M), use sample_spark() instead.

        Args:
            n: Number of samples to generate
            random_state: Random seed for reproducibility
            return_uniform: If True, return uniform [0,1] samples without
                marginal transformation. This is faster and matches statsmodels
                behavior. Default False returns samples transformed to the
                fitted marginal distributions.

        Returns:
            Dict mapping column names to sample arrays

        Example:
            >>> samples = copula.sample(n=10000, random_state=42)
            >>> df = pd.DataFrame(samples)

            >>> # For raw copula samples (faster, no marginal transform)
            >>> uniform_samples = copula.sample(n=10000, return_uniform=True)
        """
        rng = np.random.default_rng(random_state)

        # Generate multivariate normal samples with the correlation structure
        # Mean is 0 since we'll transform through the marginals
        mvn_samples = rng.multivariate_normal(
            mean=np.zeros(len(self.column_names)),
            cov=self.correlation_matrix,
            size=n,
        )

        # Transform normal -> uniform via standard normal CDF (vectorized for all columns)
        uniform_samples = st.norm.cdf(mvn_samples)

        # If user wants raw uniform samples, return early (fast path)
        if return_uniform:
            return {col: uniform_samples[:, i] for i, col in enumerate(self.column_names)}

        # Transform uniform -> target marginal via inverse CDF (PPF)
        # Uses frozen (cached) distributions for better performance
        result: Dict[str, np.ndarray] = {}
        for i, col in enumerate(self.column_names):
            frozen_dist = self._get_frozen_dist(col)
            result[col] = frozen_dist.ppf(uniform_samples[:, i])

        return result

    def sample_spark(
        self,
        n: int,
        spark: Optional[SparkSession] = None,
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
        return_uniform: bool = False,
    ) -> DataFrame:
        """Generate correlated samples using Spark distributed computing.

        This is the key differentiator - generates millions of correlated samples
        across the cluster, leveraging Spark's parallelism.

        Args:
            n: Total number of samples to generate
            spark: SparkSession. If None, uses the active session.
            num_partitions: Number of partitions. Defaults to Spark default parallelism.
            random_seed: Random seed for reproducibility
            return_uniform: If True, return uniform [0,1] samples without
                marginal transformation. This is faster. Default False returns
                samples transformed to the fitted marginal distributions.

        Returns:
            Spark DataFrame with one column per marginal

        Example:
            >>> samples_df = copula.sample_spark(n=100_000_000, spark=spark)
            >>> samples_df.show(5)
        """
        spark = get_spark_session(spark)
        if num_partitions is None:
            num_partitions = spark.sparkContext.defaultParallelism

        # Calculate samples per partition
        base_samples = n // num_partitions
        remainder = n % num_partitions

        # Create partition info DataFrame
        partition_data = []
        for i in range(num_partitions):
            samples_for_partition = base_samples + (1 if i < remainder else 0)
            if samples_for_partition > 0:
                partition_data.append((i, samples_for_partition))

        partition_df = spark.createDataFrame(
            partition_data,
            StructType(
                [
                    StructField("partition_id", IntegerType(), False),
                    StructField("n_samples", IntegerType(), False),
                ]
            ),
        )

        # Repartition to ensure each row goes to its own partition
        partition_df = partition_df.repartition(len(partition_data))

        # Define output schema with one column per marginal
        output_fields = [StructField(col, DoubleType(), False) for col in self.column_names]
        output_schema = StructType(output_fields)

        # Prepare data for serialization to workers
        corr_matrix = self.correlation_matrix.tolist()
        marginal_data: Dict[str, Dict[str, Any]] = {
            col: {
                "distribution": m.distribution,
                "parameters": m.parameters,
            }
            for col, m in self.marginals.items()
        }
        column_names = self.column_names

        def generate_correlated_samples(
            iterator: Iterator[pd.DataFrame],
        ) -> Iterator[pd.DataFrame]:
            """Generate correlated samples for each partition."""
            # Pre-create frozen distributions once per worker (cached)
            # Only needed if we're doing marginal transforms
            frozen_dists: Dict[str, st.rv_continuous] = {}
            if not return_uniform:
                for col in column_names:
                    m_info = marginal_data[col]
                    dist = getattr(st, m_info["distribution"])
                    frozen_dists[col] = dist(*m_info["parameters"])

            # Pre-convert correlation matrix once
            corr_np = np.array(corr_matrix)

            for pdf in iterator:
                if len(pdf) == 0:
                    continue

                # Process all rows in the partition at once (vectorized, no iterrows)
                for idx in range(len(pdf)):
                    n_samples = int(pdf.iloc[idx]["n_samples"])
                    partition_id = int(pdf.iloc[idx]["partition_id"])

                    # Create unique seed for this partition
                    if random_seed is not None:
                        rng = np.random.default_rng(random_seed + partition_id)
                    else:
                        rng = np.random.default_rng()

                    # Generate multivariate normal samples
                    mvn_samples = rng.multivariate_normal(
                        mean=np.zeros(len(column_names)),
                        cov=corr_np,
                        size=n_samples,
                    )

                    # Transform normal -> uniform (vectorized for all columns at once)
                    uniform_samples = st.norm.cdf(mvn_samples)

                    # Fast path: return uniform samples without marginal transform
                    if return_uniform:
                        result_data = {col: uniform_samples[:, i] for i, col in enumerate(column_names)}
                    else:
                        # Transform through frozen distributions for each column
                        result_data = {}
                        for i, col in enumerate(column_names):
                            result_data[col] = frozen_dists[col].ppf(uniform_samples[:, i])

                    yield pd.DataFrame(result_data)

        # Apply the UDF
        result_df = partition_df.mapInPandas(
            generate_correlated_samples,
            schema=output_schema,
        )

        return result_df

    def save(
        self,
        path: Union[str, Path],
        format: Optional[Literal["json", "pickle"]] = None,
        indent: Optional[int] = 2,
    ) -> None:
        """Save the copula to a file.

        Args:
            path: File path (.json or .pkl/.pickle)
            format: File format. Auto-detected from extension if not specified.
            indent: JSON indentation level (None for compact)

        Example:
            >>> copula.save("copula.json")
            >>> copula.save("copula.pkl", format="pickle")
        """
        path = Path(path)

        if format is None:
            format = detect_format(path)

        if format == "json":
            data = self._to_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GaussianCopula":
        """Load a copula from a file.

        Args:
            path: File path (.json or .pkl/.pickle)

        Returns:
            Loaded GaussianCopula instance

        Example:
            >>> copula = GaussianCopula.load("copula.json")
        """
        path = Path(path)
        format = detect_format(path)

        if format == "json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise SerializationError(f"Invalid JSON: {e}") from e
            return cls._from_dict(data)
        else:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise SerializationError(f"Failed to load pickle: {e}") from e

    def _to_dict(self) -> Dict[str, Any]:
        """Convert copula to dictionary for JSON serialization."""
        return {
            "schema_version": SCHEMA_VERSION,
            "spark_bestfit_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "gaussian_copula",
            "column_names": self.column_names,
            "correlation_matrix": self.correlation_matrix.tolist(),
            "marginals": {
                col: {
                    "distribution": m.distribution,
                    "parameters": m.parameters,
                }
                for col, m in self.marginals.items()
            },
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "GaussianCopula":
        """Reconstruct copula from dictionary."""
        # Validate required fields
        required = ["column_names", "correlation_matrix", "marginals"]
        for field_name in required:
            if field_name not in data:
                raise SerializationError(f"Missing required field: '{field_name}'")

        # Reconstruct marginals
        marginals: Dict[str, DistributionFitResult] = {}
        for col, m_data in data["marginals"].items():
            if "distribution" not in m_data or "parameters" not in m_data:
                raise SerializationError(f"Invalid marginal data for column '{col}'")

            # Validate distribution exists
            if not hasattr(st, m_data["distribution"]):
                raise SerializationError(f"Unknown distribution: '{m_data['distribution']}'")

            marginals[col] = DistributionFitResult(
                distribution=m_data["distribution"],
                parameters=list(m_data["parameters"]),
                sse=float("inf"),  # Not stored in copula serialization
            )

        return cls(
            column_names=list(data["column_names"]),
            marginals=marginals,
            correlation_matrix=np.array(data["correlation_matrix"]),
        )
