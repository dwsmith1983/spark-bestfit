"""Multivariate distribution fitting for correlated multi-column data.

This module provides direct multivariate distribution fitting as an alternative
to copula-based correlation modeling. Use multivariate fitting when:
- The joint distribution is known to be multivariate normal
- Interpretability of joint parameters (mean, covariance) is important
- Statistical tests require the MVN assumption

Example:
    >>> from spark_bestfit import MultivariateNormalFitter, LocalBackend
    >>>
    >>> # Fit multivariate normal to multiple columns
    >>> fitter = MultivariateNormalFitter(backend=LocalBackend())
    >>> result = fitter.fit(df, columns=["x", "y", "z"])
    >>>
    >>> # Access joint parameters
    >>> print(result.mean)  # [mu_x, mu_y, mu_z]
    >>> print(result.cov)   # 3x3 covariance matrix
    >>>
    >>> # Generate correlated samples
    >>> samples = result.sample(n=10000)

Note:
    This is positioned as an **optional alternative** to copulas, not a replacement.
    Copulas remain the primary recommendation because:
    - Copulas support all 100+ marginal distributions
    - Multivariate distributions limited to MVN (and future MVt, Dirichlet)
    - Copulas scale better to many columns
"""

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
import scipy.stats as st

from spark_bestfit._version import __version__
from spark_bestfit.serialization import SCHEMA_VERSION, SerializationError, detect_format

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend


@dataclass(slots=True)
class MultivariateNormalResult:
    """Result of multivariate normal distribution fitting.

    Contains the fitted parameters (mean vector and covariance matrix) and
    provides methods for sampling and evaluation.

    Attributes:
        column_names: List of column names in order
        mean: Mean vector of shape (n_columns,)
        cov: Covariance matrix of shape (n_columns, n_columns)
        n_samples: Number of samples used in fitting

    Example:
        >>> result = fitter.fit(df, columns=["x", "y", "z"])
        >>> print(result.mean)      # [mu_x, mu_y, mu_z]
        >>> print(result.cov)       # 3x3 covariance matrix
        >>> samples = result.sample(n=10000)
    """

    column_names: List[str]
    mean: np.ndarray = field(repr=False)
    cov: np.ndarray = field(repr=False)
    n_samples: int = 0
    _cholesky: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Validate result state after initialization."""
        if len(self.column_names) < 2:
            raise ValueError("MultivariateNormalResult requires at least 2 columns")

        n = len(self.column_names)
        if self.mean.shape != (n,):
            raise ValueError(f"mean shape {self.mean.shape} doesn't match {n} columns")

        if self.cov.shape != (n, n):
            raise ValueError(f"cov shape {self.cov.shape} doesn't match {n} columns")

        # Ensure covariance matrix is positive semi-definite
        # Add small regularization if needed for numerical stability
        try:
            self._cholesky = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            # Add small regularization for numerical stability
            regularized_cov = self.cov + np.eye(n) * 1e-10
            self._cholesky = np.linalg.cholesky(regularized_cov)

    @property
    def n_dimensions(self) -> int:
        """Return the number of dimensions (columns)."""
        return len(self.column_names)

    def sample(
        self,
        n: int,
        random_state: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate samples from the multivariate normal distribution.

        Uses the Cholesky decomposition for efficient sampling that preserves
        the correlation structure.

        Args:
            n: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Dict mapping column names to sample arrays

        Example:
            >>> samples = result.sample(n=10000, random_state=42)
            >>> df = pd.DataFrame(samples)
        """
        if n == 0:
            return {col: np.array([]) for col in self.column_names}

        rng = np.random.default_rng(random_state)

        # Generate multivariate normal samples using Cholesky decomposition
        # This is faster than rng.multivariate_normal() which recomputes Cholesky
        z = rng.standard_normal(size=(n, len(self.column_names)))
        samples = z @ self._cholesky.T + self.mean

        return {col: samples[:, i] for i, col in enumerate(self.column_names)}

    def sample_distributed(
        self,
        n: int,
        backend: "ExecutionBackend",
        num_partitions: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Any:
        """Generate samples using distributed computing.

        Generates millions of correlated samples across the cluster,
        leveraging the backend's parallelism.

        Args:
            n: Total number of samples to generate
            backend: Execution backend (SparkBackend, LocalBackend, etc.)
            num_partitions: Number of partitions. Defaults to backend parallelism.
            random_seed: Random seed for reproducibility

        Returns:
            Backend-specific DataFrame with one column per dimension
            (Spark DataFrame for SparkBackend, pandas DataFrame for LocalBackend)

        Example:
            >>> samples_df = result.sample_distributed(n=100_000_000, backend=backend)
            >>> samples_df.show(5)  # For Spark
        """
        # Prepare data for serialization to workers
        mean_vec = self.mean.copy()
        cholesky_matrix = self._cholesky.copy()
        column_names = self.column_names

        def generate_mvn_samples(
            n_samples: int,
            partition_id: int,
            seed: Optional[int],
        ) -> Dict[str, np.ndarray]:
            """Generate MVN samples for a partition."""
            rng = np.random.default_rng(seed)

            # Generate using pre-computed Cholesky
            z = rng.standard_normal(size=(n_samples, len(column_names)))
            samples = z @ cholesky_matrix.T + mean_vec

            return {col: samples[:, i] for i, col in enumerate(column_names)}

        return backend.generate_samples(
            n=n,
            generator_func=generate_mvn_samples,
            column_names=column_names,
            num_partitions=num_partitions,
            random_seed=random_seed,
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the probability density function at points x.

        Args:
            x: Array of shape (n_points, n_dimensions) or (n_dimensions,)

        Returns:
            Array of PDF values

        Example:
            >>> densities = result.pdf(np.array([[1, 2, 3], [4, 5, 6]]))
        """
        return st.multivariate_normal(mean=self.mean, cov=self.cov).pdf(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the log probability density function at points x.

        Args:
            x: Array of shape (n_points, n_dimensions) or (n_dimensions,)

        Returns:
            Array of log-PDF values

        Example:
            >>> log_densities = result.logpdf(np.array([[1, 2, 3]]))
        """
        return st.multivariate_normal(mean=self.mean, cov=self.cov).logpdf(x)

    def mahalanobis(self, x: np.ndarray) -> np.ndarray:
        """Compute the Mahalanobis distance from points to the distribution center.

        The Mahalanobis distance accounts for the covariance structure and is
        useful for outlier detection and hypothesis testing.

        Args:
            x: Array of shape (n_points, n_dimensions) or (n_dimensions,)

        Returns:
            Array of Mahalanobis distances

        Example:
            >>> distances = result.mahalanobis(test_data)
            >>> outliers = distances > 3.0  # ~99.7% threshold for MVN
        """
        x = np.atleast_2d(x)
        diff = x - self.mean
        # Mahalanobis distance: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
        # Using solve with Cholesky is more numerically stable than direct inverse
        inv_cov = np.linalg.inv(self.cov)
        mahal_sq = np.sum(diff @ inv_cov * diff, axis=1)
        return np.sqrt(mahal_sq)

    def correlation_matrix(self) -> np.ndarray:
        """Compute the correlation matrix from the covariance matrix.

        Returns:
            Correlation matrix of shape (n_columns, n_columns)

        Example:
            >>> corr = result.correlation_matrix()
            >>> print(corr)  # Values between -1 and 1
        """
        std_devs = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(std_devs, std_devs)

    def save(
        self,
        path: Union[str, Path],
        format: Optional[Literal["json", "pickle"]] = None,
        indent: Optional[int] = 2,
    ) -> None:
        """Save the result to a file.

        Args:
            path: File path (.json or .pkl/.pickle)
            format: File format. Auto-detected from extension if not specified.
            indent: JSON indentation level (None for compact)

        Example:
            >>> result.save("mvn_result.json")
            >>> result.save("mvn_result.pkl", format="pickle")
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
    def load(cls, path: Union[str, Path]) -> "MultivariateNormalResult":
        """Load a result from a file.

        Args:
            path: File path (.json or .pkl/.pickle)

        Returns:
            Loaded MultivariateNormalResult instance

        Example:
            >>> result = MultivariateNormalResult.load("mvn_result.json")
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
        """Convert result to dictionary for JSON serialization."""
        return {
            "schema_version": SCHEMA_VERSION,
            "spark_bestfit_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "multivariate_normal",
            "column_names": self.column_names,
            "mean": self.mean.tolist(),
            "cov": self.cov.tolist(),
            "n_samples": self.n_samples,
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "MultivariateNormalResult":
        """Reconstruct result from dictionary."""
        required = ["column_names", "mean", "cov"]
        for field_name in required:
            if field_name not in data:
                raise SerializationError(f"Missing required field: '{field_name}'")

        return cls(
            column_names=list(data["column_names"]),
            mean=np.array(data["mean"]),
            cov=np.array(data["cov"]),
            n_samples=data.get("n_samples", 0),
        )


class MultivariateNormalFitter:
    """Fitter for multivariate normal distributions.

    Fits a multivariate normal distribution to multi-column data using
    maximum likelihood estimation (sample mean and covariance).

    This provides a direct alternative to copula-based modeling when the
    joint distribution is assumed to be multivariate normal.

    Attributes:
        _backend: Execution backend for distributed operations

    Example:
        >>> from spark_bestfit import MultivariateNormalFitter, LocalBackend
        >>>
        >>> fitter = MultivariateNormalFitter(backend=LocalBackend())
        >>> result = fitter.fit(df, columns=["x", "y", "z"])
        >>>
        >>> # Access fitted parameters
        >>> print(result.mean)
        >>> print(result.cov)
        >>>
        >>> # Generate samples
        >>> samples = result.sample(n=10000)
    """

    def __init__(
        self,
        backend: Optional["ExecutionBackend"] = None,
    ) -> None:
        """Initialize the fitter.

        Args:
            backend: Execution backend. If None, uses LocalBackend.

        Example:
            >>> fitter = MultivariateNormalFitter()  # Uses LocalBackend
            >>> fitter = MultivariateNormalFitter(backend=SparkBackend(spark))
        """
        if backend is None:
            from spark_bestfit.backends.local import LocalBackend

            backend = LocalBackend()

        self._backend = backend

    def fit(
        self,
        df: Any,
        columns: List[str],
        bias: bool = False,
    ) -> MultivariateNormalResult:
        """Fit a multivariate normal distribution to data.

        Estimates the mean vector and covariance matrix using maximum
        likelihood estimation.

        Args:
            df: DataFrame (Spark, pandas, or Ray Dataset)
            columns: List of column names to fit (minimum 2)
            bias: If True, use biased covariance estimate (ddof=0, MLE).
                If False (default), use unbiased estimate (ddof=1).

        Returns:
            MultivariateNormalResult with fitted parameters

        Raises:
            ValueError: If fewer than 2 columns specified

        Example:
            >>> result = fitter.fit(df, columns=["x", "y", "z"])
            >>> print(f"Mean: {result.mean}")
            >>> print(f"Covariance:\\n{result.cov}")
        """
        if len(columns) < 2:
            raise ValueError("MultivariateNormalFitter requires at least 2 columns")

        # Validate columns exist
        self._validate_columns(df, columns)

        # Collect data for mean and covariance computation
        # For very large datasets, could add sampling option in future
        data = self._collect_columns(df, columns)

        # Compute MLE estimates
        mean = np.mean(data, axis=0)
        ddof = 0 if bias else 1
        cov = np.cov(data, rowvar=False, ddof=ddof)

        # Handle case where cov is scalar (2 columns, 1 sample edge case)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Check condition number for numerical stability warning
        cond_num = np.linalg.cond(cov)
        if cond_num > 1e10:
            warnings.warn(
                f"Covariance matrix has high condition number ({cond_num:.2e}). "
                "This may indicate near-collinear columns and could cause "
                "numerical instability. Consider removing highly correlated columns.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(f"Near-singular covariance matrix detected: condition number = {cond_num:.2e}")

        return MultivariateNormalResult(
            column_names=list(columns),
            mean=mean,
            cov=cov,
            n_samples=len(data),
        )

    def _validate_columns(self, df: Any, columns: List[str]) -> None:
        """Validate that columns exist in the DataFrame."""
        # Get columns list based on DataFrame type
        if hasattr(df, "select_columns") and hasattr(df, "schema"):
            # Ray Dataset
            df_columns = df.schema().names
        elif hasattr(df, "sparkSession"):
            # Spark DataFrame
            df_columns = df.columns
        else:
            # pandas DataFrame
            df_columns = list(df.columns)

        missing = set(columns) - set(df_columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}. " f"Available columns: {df_columns}")

    def _collect_columns(self, df: Any, columns: List[str]) -> np.ndarray:
        """Collect specified columns as numpy array."""
        if hasattr(df, "sparkSession"):
            # Spark DataFrame - select columns and collect
            pdf = df.select(columns).toPandas()
            return pdf[columns].values
        elif hasattr(df, "select_columns") and hasattr(df, "to_pandas"):
            # Ray Dataset
            pdf = df.select_columns(columns).to_pandas()
            return pdf[columns].values
        else:
            # pandas DataFrame
            return df[columns].values
