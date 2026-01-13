ADR-0007: Copula Implementation
================================

:Status: Accepted
:Date: 2025-12-30 (v1.3.0)

Context
-------

spark-bestfit fits marginal distributions independently for each column.
However, real-world data often has correlations between columns:

- Financial data: asset returns are correlated
- Sensor data: temperature and humidity co-vary
- Biological data: gene expressions have dependencies

Users need to generate synthetic samples that preserve both:

1. Marginal distributions (individual column distributions)
2. Dependency structure (correlations between columns)

Copulas provide a mathematical framework to separate marginal distributions
from dependency structure, enabling this.

Decision
--------

We implemented Gaussian Copula in ``copula.py``::

    class GaussianCopula:
        def __init__(
            self,
            marginals: Dict[str, FitResult],
            correlation_matrix: Optional[np.ndarray] = None,
        ):
            self.marginals = marginals
            self.correlation = correlation_matrix

        def fit(self, df: Any, columns: List[str]) -> "GaussianCopula":
            # Compute Spearman correlation matrix
            # Store with marginal distributions

        def sample(self, n: int) -> pd.DataFrame:
            # 1. Generate correlated normal samples via Cholesky
            # 2. Transform to uniform via normal CDF
            # 3. Apply inverse CDF of each marginal

**Algorithm:**

1. **Fitting**: Compute Spearman rank correlation (robust to non-normality)
2. **Sampling**:
   - Generate ``Z ~ N(0, Sigma)`` using Cholesky decomposition: ``L @ standard_normal``
   - Transform to uniform: ``U = Phi(Z)`` where Phi is standard normal CDF
   - Apply marginal inverse CDF: ``X_i = F_i^{-1}(U_i)``

**Optimizations** (v2.7.0, v2.8.0)::

    # Cached Cholesky decomposition
    self._cholesky = np.linalg.cholesky(self.correlation)

    # Fast PPF using scipy.special.ndtri instead of norm.ppf
    from scipy.special import ndtri
    uniform = ndtri(standard_normal)  # 10x faster than norm.cdf

**Distributed sampling** (v1.3.0)::

    def sample_distributed(
        self,
        n: int,
        backend: ExecutionBackend,
        num_partitions: Optional[int] = None,
    ) -> Any:
        # Each partition generates subset of samples
        # Returns backend-native DataFrame (Spark/Ray/pandas)

Consequences
------------

**Positive:**

- Preserves both marginal distributions and correlations
- Gaussian copula is computationally efficient
- Distributed sampling scales to billions of rows
- Backend-agnostic via ExecutionBackend protocol

**Negative:**

- Gaussian copula assumes elliptical dependency; tail dependencies
  (common in finance) are not captured
- Correlation matrix must be positive semi-definite
- Memory scales as O(columns^2) for correlation matrix

**Neutral:**

- Spearman correlation chosen over Pearson for robustness to non-linearity
- Future work could add t-copula for tail dependencies

References
----------

- `PR #50 <https://github.com/dwsmith1983/spark-bestfit/pull/50>`_: Gaussian Copula (v1.3.0)
- `Commit 8ea8f20 <https://github.com/dwsmith1983/spark-bestfit/commit/8ea8f20>`_: fast_ppf optimization (v2.7.0)
- `PR #125 <https://github.com/dwsmith1983/spark-bestfit/pull/125>`_: Cholesky+ndtr optimization (v2.8.0)
- Nelsen, R. B. (2006). "An Introduction to Copulas"
- Related: :doc:`ADR-0001-multi-backend-architecture` (distributed sampling)
