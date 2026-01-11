"""Gaussian Mixture Model fitting for multi-modal data.

This module provides Gaussian Mixture Model (GMM) fitting using the
Expectation-Maximization (EM) algorithm. Use mixture models when:
- Data appears to come from multiple populations
- A single distribution doesn't adequately describe the data
- You want to decompose data into component distributions

Example:
    >>> from spark_bestfit import GaussianMixtureFitter
    >>>
    >>> # Fit a 2-component mixture
    >>> fitter = GaussianMixtureFitter(n_components=2)
    >>> result = fitter.fit(data)
    >>>
    >>> # Access fitted parameters
    >>> print(result.weights_)     # [0.4, 0.6]
    >>> print(result.means_)       # [mu_1, mu_2]
    >>> print(result.covariances_) # [cov_1, cov_2]
    >>>
    >>> # Generate samples from the mixture
    >>> samples = result.sample(n=10000)
    >>>
    >>> # Model selection with BIC
    >>> print(f"BIC: {result.bic}")

Note:
    This is a univariate/multivariate Gaussian mixture model.
    For arbitrary distribution mixtures, consider separate fitting
    of each component using DistributionFitter.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from scipy.special import logsumexp

from spark_bestfit._version import __version__
from spark_bestfit.serialization import SCHEMA_VERSION, SerializationError, detect_format

logger = logging.getLogger(__name__)


def _compute_log_gaussian_prob(
    X: np.ndarray,
    means: np.ndarray,
    precisions_chol: np.ndarray,
    covariance_type: str = "full",
) -> np.ndarray:
    """Compute log probability of X under each Gaussian component.

    Args:
        X: Data array of shape (n_samples, n_features)
        means: Component means of shape (n_components, n_features)
        precisions_chol: Cholesky decomposition of precision matrices
        covariance_type: Type of covariance ('full' only for now)

    Returns:
        Log probabilities of shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]

    log_prob = np.empty((n_samples, n_components))

    for k in range(n_components):
        # Compute Mahalanobis distance using Cholesky of precision
        diff = X - means[k]
        y = np.dot(diff, precisions_chol[k])
        log_prob[:, k] = -0.5 * np.sum(y**2, axis=1)

    # Add normalization constant
    log_det = np.array([2 * np.sum(np.log(np.diag(precisions_chol[k]))) for k in range(n_components)])
    log_prob += 0.5 * (log_det - n_features * np.log(2 * np.pi))

    return log_prob


def _compute_precision_cholesky(covariances: np.ndarray, reg_covar: float = 1e-6) -> np.ndarray:
    """Compute Cholesky decomposition of precision matrices.

    Args:
        covariances: Covariance matrices of shape (n_components, n_features, n_features)
        reg_covar: Regularization added to diagonal

    Returns:
        Cholesky of precision matrices
    """
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))

    for k in range(n_components):
        cov = covariances[k] + np.eye(n_features) * reg_covar
        try:
            cov_chol = np.linalg.cholesky(cov)
            precisions_chol[k] = np.linalg.solve(cov_chol, np.eye(n_features)).T
        except np.linalg.LinAlgError:
            raise ValueError(
                f"Component {k} has singular covariance matrix. " "Try increasing reg_covar or reducing n_components."
            )

    return precisions_chol


@dataclass(slots=True)
class GaussianMixtureResult:
    """Result of Gaussian Mixture Model fitting.

    Contains the fitted parameters (weights, means, covariances) and
    provides methods for sampling, evaluation, and prediction.

    Attributes:
        n_components: Number of mixture components
        weights_: Mixing weights of shape (n_components,)
        means_: Component means of shape (n_components, n_features)
        covariances_: Component covariances of shape (n_components, n_features, n_features)
        converged_: Whether EM converged
        n_iter_: Number of iterations used
        n_samples_: Number of samples used in fitting
        log_likelihood_: Final log-likelihood
        responsibilities_: Soft assignments of shape (n_samples, n_components)

    Example:
        >>> result = fitter.fit(data)
        >>> print(result.weights_)        # [0.4, 0.6]
        >>> print(result.means_)          # Component means
        >>> print(result.bic)             # Bayesian Information Criterion
        >>> samples = result.sample(n=10000)
    """

    n_components: int
    weights_: np.ndarray = field(repr=False)
    means_: np.ndarray = field(repr=False)
    covariances_: np.ndarray = field(repr=False)
    converged_: bool = False
    n_iter_: int = 0
    n_samples_: int = 0
    log_likelihood_: float = field(default=float("-inf"), repr=False)
    responsibilities_: np.ndarray = field(default=None, repr=False)
    _precisions_chol: np.ndarray = field(init=False, repr=False, default=None)
    _reg_covar: float = field(default=1e-6, repr=False)

    def __post_init__(self) -> None:
        """Validate result state and precompute precision Cholesky."""
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")

        if self.weights_.shape != (self.n_components,):
            raise ValueError(f"weights_ shape {self.weights_.shape} doesn't match n_components={self.n_components}")

        if self.means_.shape[0] != self.n_components:
            raise ValueError(f"means_ has {self.means_.shape[0]} components, expected {self.n_components}")

        if self.covariances_.shape[0] != self.n_components:
            raise ValueError(f"covariances_ has {self.covariances_.shape[0]} components, expected {self.n_components}")

        # Validate weights sum to 1
        if not np.isclose(np.sum(self.weights_), 1.0, rtol=1e-5):
            raise ValueError(f"weights_ must sum to 1, got {np.sum(self.weights_)}")

        # Precompute precision Cholesky for efficient evaluation
        self._precisions_chol = _compute_precision_cholesky(self.covariances_, self._reg_covar)

    @property
    def n_features(self) -> int:
        """Return the number of features (dimensions)."""
        return self.means_.shape[1]

    @property
    def aic(self) -> float:
        """Akaike Information Criterion for model selection.

        Lower values indicate better fit with penalty for complexity.
        AIC = -2 * log_likelihood + 2 * n_parameters
        """
        return -2 * self.log_likelihood_ + 2 * self._n_parameters

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion for model selection.

        Lower values indicate better fit with stronger penalty for complexity.
        BIC = -2 * log_likelihood + n_parameters * log(n_samples)
        """
        return -2 * self.log_likelihood_ + self._n_parameters * np.log(self.n_samples_)

    @property
    def _n_parameters(self) -> int:
        """Count the number of free parameters in the model."""
        n_features = self.n_features
        # Weights: n_components - 1 (they sum to 1)
        # Means: n_components * n_features
        # Covariances (full): n_components * n_features * (n_features + 1) / 2
        cov_params = self.n_components * n_features * (n_features + 1) // 2
        return (self.n_components - 1) + self.n_components * n_features + cov_params

    def sample(
        self,
        n: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Generate samples from the Gaussian mixture distribution.

        Args:
            n: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of samples with shape (n, n_features)

        Example:
            >>> samples = result.sample(n=10000, random_state=42)
        """
        if n == 0:
            return np.empty((0, self.n_features))

        rng = np.random.default_rng(random_state)

        # Sample component indices according to weights
        component_indices = rng.choice(self.n_components, size=n, p=self.weights_)

        # Count samples per component
        samples = np.empty((n, self.n_features))

        for k in range(self.n_components):
            mask = component_indices == k
            n_k = np.sum(mask)
            if n_k > 0:
                samples[mask] = rng.multivariate_normal(
                    mean=self.means_[k],
                    cov=self.covariances_[k],
                    size=n_k,
                )

        return samples

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the probability density function at points x.

        Args:
            x: Array of shape (n_points, n_features) or (n_features,)

        Returns:
            Array of PDF values

        Example:
            >>> densities = result.pdf(np.array([[1, 2], [3, 4]]))
        """
        return np.exp(self.logpdf(x))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the log probability density function at points x.

        Args:
            x: Array of shape (n_points, n_features) or (n_features,)

        Returns:
            Array of log-PDF values

        Example:
            >>> log_densities = result.logpdf(np.array([[1, 2]]))
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")

        log_prob = _compute_log_gaussian_prob(x, self.means_, self._precisions_chol)
        # Add log weights and marginalize
        weighted_log_prob = log_prob + np.log(self.weights_)
        return logsumexp(weighted_log_prob, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict component labels for data points (hard assignment).

        Args:
            x: Array of shape (n_points, n_features) or (n_features,)

        Returns:
            Array of component labels (0 to n_components-1)

        Example:
            >>> labels = result.predict(test_data)
        """
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict component probabilities for data points (soft assignment).

        Args:
            x: Array of shape (n_points, n_features) or (n_features,)

        Returns:
            Array of probabilities with shape (n_points, n_components)

        Example:
            >>> probs = result.predict_proba(test_data)
            >>> print(probs[0])  # [0.3, 0.7] - 30% comp 0, 70% comp 1
        """
        x = np.atleast_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")

        log_prob = _compute_log_gaussian_prob(x, self.means_, self._precisions_chol)
        weighted_log_prob = log_prob + np.log(self.weights_)

        # Normalize to get responsibilities
        log_sum = logsumexp(weighted_log_prob, axis=1, keepdims=True)
        return np.exp(weighted_log_prob - log_sum)

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
            >>> result.save("gmm_result.json")
            >>> result.save("gmm_result.pkl", format="pickle")
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
    def load(cls, path: Union[str, Path]) -> "GaussianMixtureResult":
        """Load a result from a file.

        Args:
            path: File path (.json or .pkl/.pickle)

        Returns:
            Loaded GaussianMixtureResult instance

        Example:
            >>> result = GaussianMixtureResult.load("gmm_result.json")
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
        data = {
            "schema_version": SCHEMA_VERSION,
            "spark_bestfit_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "gaussian_mixture",
            "n_components": self.n_components,
            "weights": self.weights_.tolist(),
            "means": self.means_.tolist(),
            "covariances": self.covariances_.tolist(),
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "n_samples": self.n_samples_,
            "log_likelihood": self.log_likelihood_,
            "reg_covar": self._reg_covar,
        }
        # Only include responsibilities if not too large
        if self.responsibilities_ is not None and self.responsibilities_.size < 100000:
            data["responsibilities"] = self.responsibilities_.tolist()
        return data

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "GaussianMixtureResult":
        """Reconstruct result from dictionary."""
        required = ["n_components", "weights", "means", "covariances"]
        for field_name in required:
            if field_name not in data:
                raise SerializationError(f"Missing required field: '{field_name}'")

        responsibilities = None
        if "responsibilities" in data:
            responsibilities = np.array(data["responsibilities"])

        return cls(
            n_components=data["n_components"],
            weights_=np.array(data["weights"]),
            means_=np.array(data["means"]),
            covariances_=np.array(data["covariances"]),
            converged_=data.get("converged", False),
            n_iter_=data.get("n_iter", 0),
            n_samples_=data.get("n_samples", 0),
            log_likelihood_=data.get("log_likelihood", float("-inf")),
            responsibilities_=responsibilities,
            _reg_covar=data.get("reg_covar", 1e-6),
        )


class GaussianMixtureFitter:
    """Fitter for Gaussian Mixture Models using Expectation-Maximization.

    Fits a mixture of Gaussian distributions to data using the EM algorithm.
    This is useful for data that appears to come from multiple populations.

    Attributes:
        n_components: Number of mixture components
        max_iter: Maximum number of EM iterations
        tol: Convergence threshold for log-likelihood change
        n_init: Number of initializations to try (best result kept)
        init_method: Initialization method ('kmeans' or 'random')
        random_state: Random seed for reproducibility
        reg_covar: Regularization for covariance stability

    Example:
        >>> from spark_bestfit import GaussianMixtureFitter
        >>>
        >>> # Fit a 2-component mixture
        >>> fitter = GaussianMixtureFitter(n_components=2, random_state=42)
        >>> result = fitter.fit(data)
        >>>
        >>> # Check convergence and model selection
        >>> print(f"Converged: {result.converged_}")
        >>> print(f"BIC: {result.bic}")
        >>>
        >>> # Generate samples
        >>> samples = result.sample(n=10000)
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 1,
        init_method: str = "kmeans",
        random_state: Optional[int] = None,
        reg_covar: float = 1e-6,
    ) -> None:
        """Initialize the Gaussian Mixture fitter.

        Args:
            n_components: Number of mixture components (default: 2)
            max_iter: Maximum EM iterations (default: 100)
            tol: Convergence tolerance for log-likelihood change (default: 1e-4)
            n_init: Number of initializations to try (default: 1)
            init_method: 'kmeans' or 'random' initialization (default: 'kmeans')
            random_state: Random seed for reproducibility
            reg_covar: Regularization added to covariance diagonal (default: 1e-6)

        Example:
            >>> fitter = GaussianMixtureFitter(n_components=3, n_init=5)
        """
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        if init_method not in ("kmeans", "random"):
            raise ValueError("init_method must be 'kmeans' or 'random'")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_method = init_method
        self.random_state = random_state
        self.reg_covar = reg_covar

    def fit(
        self,
        data: np.ndarray,
    ) -> GaussianMixtureResult:
        """Fit Gaussian Mixture Model to data using EM algorithm.

        Args:
            data: Array of shape (n_samples,) for univariate or
                  (n_samples, n_features) for multivariate

        Returns:
            GaussianMixtureResult with fitted parameters

        Raises:
            ValueError: If data has fewer samples than components

        Example:
            >>> result = fitter.fit(data)
            >>> print(f"Means: {result.means_}")
            >>> print(f"Converged in {result.n_iter_} iterations")
        """
        # Convert to 2D array
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.shape[0] == 1:
            # Handle row vector case
            data = data.T

        n_samples, n_features = data.shape

        if n_samples < self.n_components:
            raise ValueError(f"n_samples ({n_samples}) must be >= n_components ({self.n_components})")

        # Try multiple initializations
        rng = np.random.default_rng(self.random_state)
        best_result = None
        best_log_likelihood = float("-inf")

        for init_idx in range(self.n_init):
            # Different seed for each initialization
            init_seed = rng.integers(0, 2**31) if self.random_state is not None else None

            try:
                result = self._fit_single(data, init_seed)
                if result.log_likelihood_ > best_log_likelihood:
                    best_log_likelihood = result.log_likelihood_
                    best_result = result
            except ValueError as e:
                logger.warning(f"Initialization {init_idx} failed: {e}")
                continue

        if best_result is None:
            raise ValueError("All initializations failed. Try reducing n_components or increasing reg_covar.")

        return best_result

    def _fit_single(self, data: np.ndarray, random_state: Optional[int]) -> GaussianMixtureResult:
        """Fit GMM with a single initialization."""
        n_samples, n_features = data.shape
        rng = np.random.default_rng(random_state)

        # Initialize parameters
        weights, means, covariances = self._initialize_parameters(data, rng)

        # EM iterations
        prev_log_likelihood = float("-inf")
        converged = False
        responsibilities = None

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            precisions_chol = _compute_precision_cholesky(covariances, self.reg_covar)
            log_prob = _compute_log_gaussian_prob(data, means, precisions_chol)
            weighted_log_prob = log_prob + np.log(weights)

            log_sum = logsumexp(weighted_log_prob, axis=1, keepdims=True)
            log_likelihood = np.mean(log_sum)
            responsibilities = np.exp(weighted_log_prob - log_sum)

            # Check convergence
            change = log_likelihood - prev_log_likelihood
            if change < self.tol and iteration > 0:
                converged = True
                break

            prev_log_likelihood = log_likelihood

            # M-step: update parameters
            weights, means, covariances = self._m_step(data, responsibilities)

            # Check for empty components
            empty_components = np.sum(responsibilities, axis=0) < 1.0
            if np.any(empty_components):
                logger.warning(
                    f"Component(s) {np.where(empty_components)[0]} have near-zero responsibility. "
                    "Consider reducing n_components."
                )

        # Final log-likelihood calculation
        final_log_likelihood = n_samples * prev_log_likelihood

        return GaussianMixtureResult(
            n_components=self.n_components,
            weights_=weights,
            means_=means,
            covariances_=covariances,
            converged_=converged,
            n_iter_=iteration + 1,
            n_samples_=n_samples,
            log_likelihood_=final_log_likelihood,
            responsibilities_=responsibilities,
            _reg_covar=self.reg_covar,
        )

    def _initialize_parameters(
        self,
        data: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple:
        """Initialize GMM parameters.

        Returns:
            Tuple of (weights, means, covariances)
        """
        n_samples, n_features = data.shape

        # Initialize weights uniformly
        weights = np.ones(self.n_components) / self.n_components

        if self.init_method == "kmeans":
            # K-means++ style initialization for means
            means = self._kmeans_init(data, rng)
        else:
            # Random: pick random data points as means
            indices = rng.choice(n_samples, size=self.n_components, replace=False)
            means = data[indices].copy()

        # Initialize covariances based on data assigned to each center
        covariances = np.empty((self.n_components, n_features, n_features))

        # Compute initial assignments
        distances = np.array([np.sum((data - means[k]) ** 2, axis=1) for k in range(self.n_components)])
        assignments = np.argmin(distances, axis=0)

        for k in range(self.n_components):
            mask = assignments == k
            if np.sum(mask) > 1:
                covariances[k] = np.cov(data[mask], rowvar=False)
                # Ensure positive definite
                covariances[k] += np.eye(n_features) * self.reg_covar
            else:
                # Fall back to global covariance
                covariances[k] = np.cov(data, rowvar=False) + np.eye(n_features) * self.reg_covar

        # Handle scalar covariance case (univariate)
        if covariances.ndim == 1 or n_features == 1:
            covariances = covariances.reshape(self.n_components, 1, 1)

        return weights, means, covariances

    def _kmeans_init(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-means++ style initialization for means."""
        n_samples, n_features = data.shape
        means = np.empty((self.n_components, n_features))

        # First center: random point
        idx = rng.integers(0, n_samples)
        means[0] = data[idx]

        for k in range(1, self.n_components):
            # Compute distances to nearest existing center
            distances = np.min(
                [np.sum((data - means[j]) ** 2, axis=1) for j in range(k)],
                axis=0,
            )
            # Sample proportional to distance squared
            probs = distances / distances.sum()
            idx = rng.choice(n_samples, p=probs)
            means[k] = data[idx]

        return means

    def _m_step(
        self,
        data: np.ndarray,
        responsibilities: np.ndarray,
    ) -> tuple:
        """M-step: update parameters given responsibilities.

        Returns:
            Tuple of (weights, means, covariances)
        """
        n_samples, n_features = data.shape

        # Effective number of points per component
        nk = np.sum(responsibilities, axis=0) + 1e-10  # Avoid division by zero

        # Update weights
        weights = nk / n_samples

        # Update means
        means = np.dot(responsibilities.T, data) / nk[:, np.newaxis]

        # Update covariances
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = data - means[k]
            covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / nk[k]
            # Add regularization
            covariances[k] += np.eye(n_features) * self.reg_covar

        return weights, means, covariances
