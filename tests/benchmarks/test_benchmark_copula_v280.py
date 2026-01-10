"""Benchmarks for v2.8.0 copula sampling optimizations.

Measures the performance improvement from:
1. Cached Cholesky decomposition (vs recomputing each sample() call)
2. scipy.special.ndtr (vs scipy.stats.norm.cdf)

These optimizations provide ~1.8x speedup for copula.sample().

Run with: pytest tests/benchmarks/test_benchmark_copula_v280.py -v --benchmark-min-rounds=20
"""

import numpy as np
import pytest
import scipy.stats as st
from scipy.special import ndtr

from spark_bestfit.copula import GaussianCopula
from spark_bestfit.results import DistributionFitResult


@pytest.fixture
def copula_2col():
    """Create a 2-column copula for benchmarking."""
    marginals = {
        "col_a": DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.0,
        ),
        "col_b": DistributionFitResult(
            distribution="gamma",
            parameters=[2.0, 0.0, 1.0],
            sse=0.0,
        ),
    }
    correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    return GaussianCopula(
        column_names=["col_a", "col_b"],
        marginals=marginals,
        correlation_matrix=correlation_matrix,
    )


@pytest.fixture
def copula_5col():
    """Create a 5-column copula for benchmarking (more realistic)."""
    marginals = {
        f"col_{i}": DistributionFitResult(
            distribution="norm",
            parameters=[i * 0.1, 1.0 + i * 0.1],
            sse=0.0,
        )
        for i in range(5)
    }
    # Create a valid positive-definite correlation matrix
    np.random.seed(42)
    A = np.random.randn(5, 5)
    cov = A @ A.T
    # Convert to correlation matrix
    d = np.sqrt(np.diag(cov))
    correlation_matrix = cov / np.outer(d, d)
    # Ensure perfect diagonal
    np.fill_diagonal(correlation_matrix, 1.0)

    return GaussianCopula(
        column_names=[f"col_{i}" for i in range(5)],
        marginals=marginals,
        correlation_matrix=correlation_matrix,
    )


# =============================================================================
# Baseline benchmarks: Raw scipy operations (OLD approach)
# =============================================================================


class TestPreOptimization:
    """Benchmark the OLD approach (before v2.8.0 optimizations).

    These tests simulate what the code did before caching Cholesky
    and using ndtr.
    """

    def test_old_multivariate_normal_1k(self, benchmark, copula_2col):
        """OLD: rng.multivariate_normal (recomputes Cholesky each call)."""
        rng = np.random.default_rng(42)
        corr = copula_2col.correlation_matrix

        def run():
            # OLD approach: multivariate_normal recomputes Cholesky internally
            return rng.multivariate_normal(
                mean=np.zeros(2),
                cov=corr,
                size=1_000,
            )

        result = benchmark(run)
        assert result.shape == (1_000, 2)

    def test_old_multivariate_normal_10k(self, benchmark, copula_2col):
        """OLD: rng.multivariate_normal with 10K samples."""
        rng = np.random.default_rng(42)
        corr = copula_2col.correlation_matrix

        def run():
            return rng.multivariate_normal(
                mean=np.zeros(2),
                cov=corr,
                size=10_000,
            )

        result = benchmark(run)
        assert result.shape == (10_000, 2)

    def test_old_multivariate_normal_100k(self, benchmark, copula_2col):
        """OLD: rng.multivariate_normal with 100K samples."""
        rng = np.random.default_rng(42)
        corr = copula_2col.correlation_matrix

        def run():
            return rng.multivariate_normal(
                mean=np.zeros(2),
                cov=corr,
                size=100_000,
            )

        result = benchmark(run)
        assert result.shape == (100_000, 2)

    def test_old_multivariate_normal_1m(self, benchmark, copula_2col):
        """OLD: rng.multivariate_normal with 1M samples."""
        rng = np.random.default_rng(42)
        corr = copula_2col.correlation_matrix

        def run():
            return rng.multivariate_normal(
                mean=np.zeros(2),
                cov=corr,
                size=1_000_000,
            )

        result = benchmark(run)
        assert result.shape == (1_000_000, 2)

    def test_old_norm_cdf_100k(self, benchmark):
        """OLD: scipy.stats.norm.cdf for CDF transformation."""
        np.random.seed(42)
        data = np.random.randn(100_000, 2)

        def run():
            return st.norm.cdf(data)

        result = benchmark(run)
        assert result.shape == (100_000, 2)

    def test_old_norm_cdf_1m(self, benchmark):
        """OLD: scipy.stats.norm.cdf for CDF transformation (1M)."""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 2)

        def run():
            return st.norm.cdf(data)

        result = benchmark(run)
        assert result.shape == (1_000_000, 2)


# =============================================================================
# NEW approach benchmarks (v2.8.0 optimizations)
# =============================================================================


class TestPostOptimization:
    """Benchmark the NEW approach (v2.8.0 optimizations).

    Uses cached Cholesky decomposition and scipy.special.ndtr.
    """

    def test_new_cholesky_1k(self, benchmark, copula_2col):
        """NEW: Pre-cached Cholesky with matrix multiply."""
        rng = np.random.default_rng(42)
        chol = copula_2col._cholesky  # Pre-computed in __post_init__

        def run():
            z = rng.standard_normal(size=(1_000, 2))
            return z @ chol.T

        result = benchmark(run)
        assert result.shape == (1_000, 2)

    def test_new_cholesky_10k(self, benchmark, copula_2col):
        """NEW: Pre-cached Cholesky with 10K samples."""
        rng = np.random.default_rng(42)
        chol = copula_2col._cholesky

        def run():
            z = rng.standard_normal(size=(10_000, 2))
            return z @ chol.T

        result = benchmark(run)
        assert result.shape == (10_000, 2)

    def test_new_cholesky_100k(self, benchmark, copula_2col):
        """NEW: Pre-cached Cholesky with 100K samples."""
        rng = np.random.default_rng(42)
        chol = copula_2col._cholesky

        def run():
            z = rng.standard_normal(size=(100_000, 2))
            return z @ chol.T

        result = benchmark(run)
        assert result.shape == (100_000, 2)

    def test_new_cholesky_1m(self, benchmark, copula_2col):
        """NEW: Pre-cached Cholesky with 1M samples."""
        rng = np.random.default_rng(42)
        chol = copula_2col._cholesky

        def run():
            z = rng.standard_normal(size=(1_000_000, 2))
            return z @ chol.T

        result = benchmark(run)
        assert result.shape == (1_000_000, 2)

    def test_new_ndtr_100k(self, benchmark):
        """NEW: scipy.special.ndtr for CDF transformation."""
        np.random.seed(42)
        data = np.random.randn(100_000, 2)

        def run():
            return ndtr(data)

        result = benchmark(run)
        assert result.shape == (100_000, 2)

    def test_new_ndtr_1m(self, benchmark):
        """NEW: scipy.special.ndtr for CDF transformation (1M)."""
        np.random.seed(42)
        data = np.random.randn(1_000_000, 2)

        def run():
            return ndtr(data)

        result = benchmark(run)
        assert result.shape == (1_000_000, 2)


# =============================================================================
# End-to-end copula.sample() benchmarks (CURRENT optimized implementation)
# =============================================================================


class TestCopulaSampleOptimized:
    """End-to-end benchmarks for the optimized copula.sample() method.

    These measure the full copula sampling pipeline with v2.8.0 optimizations.
    Compare with TestCopulaSamplingWithFastPPF in test_benchmark_fast_ppf.py
    for the fast_ppf impact (separate from Cholesky/ndtr optimizations).
    """

    def test_copula_sample_2col_1k(self, benchmark, copula_2col):
        """Full copula.sample() with 2 columns, 1K samples."""

        def run():
            return copula_2col.sample(n=1_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_a"]) == 1_000

    def test_copula_sample_2col_10k(self, benchmark, copula_2col):
        """Full copula.sample() with 2 columns, 10K samples."""

        def run():
            return copula_2col.sample(n=10_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_a"]) == 10_000

    def test_copula_sample_2col_100k(self, benchmark, copula_2col):
        """Full copula.sample() with 2 columns, 100K samples."""

        def run():
            return copula_2col.sample(n=100_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_a"]) == 100_000

    def test_copula_sample_2col_1m(self, benchmark, copula_2col):
        """Full copula.sample() with 2 columns, 1M samples."""

        def run():
            return copula_2col.sample(n=1_000_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_a"]) == 1_000_000

    def test_copula_sample_5col_100k(self, benchmark, copula_5col):
        """Full copula.sample() with 5 columns, 100K samples."""

        def run():
            return copula_5col.sample(n=100_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_0"]) == 100_000

    def test_copula_sample_5col_1m(self, benchmark, copula_5col):
        """Full copula.sample() with 5 columns, 1M samples."""

        def run():
            return copula_5col.sample(n=1_000_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_0"]) == 1_000_000

    def test_copula_sample_uniform_100k(self, benchmark, copula_2col):
        """Copula uniform samples only (skip marginal transform)."""

        def run():
            return copula_2col.sample(n=100_000, random_state=42, return_uniform=True)

        result = benchmark(run)
        assert len(result["col_a"]) == 100_000
        # Verify values are in [0, 1]
        assert np.all(result["col_a"] >= 0)
        assert np.all(result["col_a"] <= 1)

    def test_copula_sample_uniform_1m(self, benchmark, copula_2col):
        """Copula uniform samples only (1M, skip marginal transform)."""

        def run():
            return copula_2col.sample(n=1_000_000, random_state=42, return_uniform=True)

        result = benchmark(run)
        assert len(result["col_a"]) == 1_000_000
