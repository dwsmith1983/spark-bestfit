"""Gaussian Copula for correlated multi-column sampling.

This module provides scalable copula modeling that works on massive DataFrames:
- Correlation computed via backend (Spark ML, pandas, or other backends)
- Distributed sampling via sample_distributed() for millions of correlated samples

Example:
    >>> from spark_bestfit import DistributionFitter, GaussianCopula
    >>>
    >>> # Fit multiple columns
    >>> fitter = DistributionFitter(spark)
    >>> results = fitter.fit(df, columns=["price", "quantity", "revenue"])
    >>>
    >>> # Fit copula - auto-detects backend from DataFrame type
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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
import scipy.stats as st

from spark_bestfit._version import __version__
from spark_bestfit.results import DistributionFitResult, MetricName
from spark_bestfit.serialization import SCHEMA_VERSION, SerializationError, detect_format

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend
    from spark_bestfit.results import FitResultsType


@dataclass
class GaussianCopula:
    """Gaussian copula for generating correlated multi-column samples.

    Preserves both the marginal distributions (from fitting) and the
    correlation structure (from the original data) when generating samples.

    This implementation supports multiple backends:
    - SparkBackend: Correlation via Spark ML (no .toPandas()), distributed sampling
    - LocalBackend: Correlation via pandas, local parallel sampling

    Attributes:
        column_names: List of column names in order
        marginals: Dict mapping column name to DistributionFitResult
        correlation_matrix: Spearman correlation matrix as numpy array

    Example:
        >>> copula = GaussianCopula.fit(results, df, backend=fitter._backend)
        >>> samples = copula.sample(n=10000)
        >>> samples_df = copula.sample_distributed(n=1_000_000, backend=backend)
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
        results: "FitResultsType",
        df: Any,
        columns: Optional[List[str]] = None,
        metric: MetricName = "ks_statistic",
        backend: Optional["ExecutionBackend"] = None,
    ) -> "GaussianCopula":
        """Fit a Gaussian copula from multi-column fit results.

        Computes the Spearman correlation matrix using the provided backend.
        For SparkBackend, this uses Spark ML's distributed computation and
        scales to billions of rows without .toPandas().

        Args:
            results: FitResults from DistributionFitter.fit() with multiple columns
            df: DataFrame used for fitting (Spark DataFrame or pandas DataFrame)
            columns: Columns to include. Defaults to all columns in results.
            metric: Metric to use for selecting best distribution per column
            backend: Execution backend. If None, auto-detects from DataFrame type
                (LocalBackend for pandas, RayBackend for Ray, SparkBackend for Spark).

        Returns:
            Fitted GaussianCopula instance

        Raises:
            ValueError: If fewer than 2 columns or columns not in results

        Example:
            >>> results = fitter.fit(df, columns=["price", "quantity", "revenue"])
            >>> copula = GaussianCopula.fit(results, df)  # auto-detects backend
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

        # Auto-detect backend from DataFrame type if not provided
        if backend is None:
            from spark_bestfit.backends.factory import BackendFactory

            backend = BackendFactory.for_dataframe(df)

        # Compute Spearman correlation via backend
        correlation_matrix = backend.compute_correlation(df, columns, method="spearman")

        return cls(
            column_names=list(columns),
            marginals=marginals,
            correlation_matrix=correlation_matrix,
        )

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
            # get_scipy_dist() now returns frozen distribution by default
            self._frozen_dists[col] = marginal.get_scipy_dist()
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

    def sample_distributed(
        self,
        n: int,
        backend: "ExecutionBackend",
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
        return_uniform: bool = False,
    ) -> Any:
        """Generate correlated samples using distributed computing.

        This is the key differentiator - generates millions of correlated samples
        across the cluster, leveraging the backend's parallelism.

        Args:
            n: Total number of samples to generate
            backend: Execution backend (SparkBackend, LocalBackend, etc.)
            num_partitions: Number of partitions. Defaults to backend parallelism.
            random_seed: Random seed for reproducibility
            return_uniform: If True, return uniform [0,1] samples without
                marginal transformation. This is faster. Default False returns
                samples transformed to the fitted marginal distributions.

        Returns:
            Backend-specific DataFrame with one column per marginal
            (Spark DataFrame for SparkBackend, pandas DataFrame for LocalBackend)

        Example:
            >>> samples_df = copula.sample_distributed(n=100_000_000, backend=backend)
            >>> samples_df.show(5)  # For Spark
        """
        # Prepare data for serialization to workers
        corr_matrix = self.correlation_matrix
        marginal_data: Dict[str, Dict[str, Any]] = {
            col: {
                "distribution": m.distribution,
                "parameters": m.parameters,
            }
            for col, m in self.marginals.items()
        }
        column_names = self.column_names

        def generate_copula_samples(
            n_samples: int,
            partition_id: int,
            seed: Optional[int],
        ) -> Dict[str, np.ndarray]:
            """Generate correlated samples for a partition.

            This function is passed to the backend's generate_samples method.
            It's designed to be serializable and run on workers.
            """
            # Pre-create frozen distributions (only if doing marginal transforms)
            frozen_dists: Dict[str, st.rv_continuous] = {}
            if not return_uniform:
                for col in column_names:
                    m_info = marginal_data[col]
                    dist = getattr(st, m_info["distribution"])
                    frozen_dists[col] = dist(*m_info["parameters"])

            # Create RNG with partition-specific seed
            rng = np.random.default_rng(seed)

            # Generate multivariate normal samples
            mvn_samples = rng.multivariate_normal(
                mean=np.zeros(len(column_names)),
                cov=corr_matrix,
                size=n_samples,
            )

            # Transform normal -> uniform (vectorized for all columns at once)
            uniform_samples = st.norm.cdf(mvn_samples)

            # Fast path: return uniform samples without marginal transform
            if return_uniform:
                return {col: uniform_samples[:, i] for i, col in enumerate(column_names)}

            # Transform through frozen distributions for each column
            result_data: Dict[str, np.ndarray] = {}
            for i, col in enumerate(column_names):
                result_data[col] = frozen_dists[col].ppf(uniform_samples[:, i])

            return result_data

        # Use backend to generate samples distributed across partitions
        return backend.generate_samples(
            n=n,
            generator_func=generate_copula_samples,
            column_names=column_names,
            num_partitions=num_partitions,
            random_seed=random_seed,
        )

    def sample_spark(
        self,
        n: int,
        spark: Optional[Any] = None,
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
        return_uniform: bool = False,
    ) -> Any:
        """Generate correlated samples using Spark distributed computing.

        .. deprecated:: 2.0.0
            Will be removed in v3.0.0. Use :meth:`sample_distributed` with
            ``SparkBackend`` instead.

        This is a backward-compatible wrapper around sample_distributed().

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
        import warnings

        warnings.warn(
            "sample_spark() is deprecated and will be removed in v3.0.0. "
            "Use sample_distributed(backend=SparkBackend()) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from spark_bestfit.backends.spark import SparkBackend

        backend = SparkBackend(spark)
        return self.sample_distributed(
            n=n,
            backend=backend,
            num_partitions=num_partitions,
            random_seed=random_seed,
            return_uniform=return_uniform,
        )

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
