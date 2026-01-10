"""Benchmarks for fast_ppf performance.

Measures speedup of fast_ppf over scipy.stats.ppf for:
- Each supported distribution
- Various array sizes (1K, 10K, 100K, 1M)
- Truncated vs non-truncated distributions
- Batch operations vs individual calls

Run with: make benchmark
"""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.fast_ppf import fast_ppf, fast_ppf_batch, has_fast_ppf


# Fixtures for quantile arrays of different sizes
@pytest.fixture
def q_1k():
    """Generate 1K random quantiles."""
    np.random.seed(42)
    return np.random.uniform(0.01, 0.99, 1_000)


@pytest.fixture
def q_10k():
    """Generate 10K random quantiles."""
    np.random.seed(42)
    return np.random.uniform(0.01, 0.99, 10_000)


@pytest.fixture
def q_100k():
    """Generate 100K random quantiles."""
    np.random.seed(42)
    return np.random.uniform(0.01, 0.99, 100_000)


@pytest.fixture
def q_1m():
    """Generate 1M random quantiles."""
    np.random.seed(42)
    return np.random.uniform(0.01, 0.99, 1_000_000)


class TestFastPPFvsScipyByDistribution:
    """Compare fast_ppf to scipy.stats.ppf for each supported distribution.

    Uses 100K elements to measure meaningful speedup while keeping
    benchmark runtime reasonable.
    """

    def test_norm_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for normal distribution."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_norm_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.norm.ppf for comparison."""
        params = (0.0, 1.0)

        def run():
            return st.norm.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_expon_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for exponential distribution."""
        params = (0.0, 2.0)

        def run():
            return fast_ppf("expon", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_expon_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.expon.ppf for comparison."""
        params = (0.0, 2.0)

        def run():
            return st.expon.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_uniform_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for uniform distribution."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("uniform", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_uniform_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.uniform.ppf for comparison."""
        params = (0.0, 1.0)

        def run():
            return st.uniform.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_lognorm_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for lognormal distribution."""
        params = (0.5, 0.0, 1.0)  # s, loc, scale

        def run():
            return fast_ppf("lognorm", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_lognorm_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.lognorm.ppf for comparison."""
        params = (0.5, 0.0, 1.0)

        def run():
            return st.lognorm.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_weibull_min_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for Weibull minimum distribution."""
        params = (2.0, 0.0, 1.0)  # c, loc, scale

        def run():
            return fast_ppf("weibull_min", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_weibull_min_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.weibull_min.ppf for comparison."""
        params = (2.0, 0.0, 1.0)

        def run():
            return st.weibull_min.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_gamma_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for gamma distribution."""
        params = (2.0, 0.0, 1.0)  # a, loc, scale

        def run():
            return fast_ppf("gamma", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_gamma_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.gamma.ppf for comparison."""
        params = (2.0, 0.0, 1.0)

        def run():
            return st.gamma.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_beta_fast_ppf(self, benchmark, q_100k):
        """Benchmark fast_ppf for beta distribution."""
        params = (2.0, 5.0, 0.0, 1.0)  # a, b, loc, scale

        def run():
            return fast_ppf("beta", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_beta_scipy(self, benchmark, q_100k):
        """Benchmark scipy.stats.beta.ppf for comparison."""
        params = (2.0, 5.0, 0.0, 1.0)

        def run():
            return st.beta.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000


class TestFastPPFScaling:
    """Benchmark how fast_ppf scales with array size.

    Uses normal distribution as the representative case.
    """

    def test_norm_1k(self, benchmark, q_1k):
        """Benchmark fast_ppf with 1K elements."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_1k)

        result = benchmark(run)
        assert len(result) == 1_000

    def test_norm_10k(self, benchmark, q_10k):
        """Benchmark fast_ppf with 10K elements."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_10k)

        result = benchmark(run)
        assert len(result) == 10_000

    def test_norm_100k(self, benchmark, q_100k):
        """Benchmark fast_ppf with 100K elements."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_norm_1m(self, benchmark, q_1m):
        """Benchmark fast_ppf with 1M elements."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_1m)

        result = benchmark(run)
        assert len(result) == 1_000_000


class TestFastPPFTruncation:
    """Benchmark truncation overhead.

    Measures the cost of applying truncation bounds to the PPF.
    """

    def test_norm_no_truncation(self, benchmark, q_100k):
        """Baseline: normal PPF without truncation."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_norm_lower_truncation(self, benchmark, q_100k):
        """Normal PPF with lower truncation bound."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_100k, lb=0.0)

        result = benchmark(run)
        assert len(result) == 100_000
        assert np.all(result >= 0.0)

    def test_norm_upper_truncation(self, benchmark, q_100k):
        """Normal PPF with upper truncation bound."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_100k, ub=0.0)

        result = benchmark(run)
        assert len(result) == 100_000
        assert np.all(result <= 0.0)

    def test_norm_both_truncation(self, benchmark, q_100k):
        """Normal PPF with both truncation bounds."""
        params = (0.0, 1.0)

        def run():
            return fast_ppf("norm", params, q_100k, lb=-1.0, ub=1.0)

        result = benchmark(run)
        assert len(result) == 100_000
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestFastPPFBatchEfficiency:
    """Benchmark batch processing efficiency.

    Measures whether batch processing provides any overhead reduction
    compared to individual calls.
    """

    @pytest.fixture
    def batch_params(self):
        """Parameters for batch benchmarks."""
        return {
            "distributions": ["norm", "expon", "gamma", "beta", "lognorm"],
            "params_list": [
                (0.0, 1.0),
                (0.0, 2.0),
                (2.0, 0.0, 1.0),
                (2.0, 5.0, 0.0, 1.0),
                (0.5, 0.0, 1.0),
            ],
        }

    def test_batch_5_distributions(self, benchmark, q_100k, batch_params):
        """Benchmark batch processing 5 distributions."""
        q_arrays = [q_100k] * 5

        def run():
            return fast_ppf_batch(
                batch_params["distributions"],
                batch_params["params_list"],
                q_arrays,
            )

        results = benchmark(run)
        assert len(results) == 5
        assert all(len(r) == 100_000 for r in results)

    def test_individual_5_distributions(self, benchmark, q_100k, batch_params):
        """Benchmark individual calls for 5 distributions (for comparison)."""

        def run():
            results = []
            for dist, params in zip(
                batch_params["distributions"], batch_params["params_list"]
            ):
                results.append(fast_ppf(dist, params, q_100k))
            return results

        results = benchmark(run)
        assert len(results) == 5
        assert all(len(r) == 100_000 for r in results)


class TestFastPPFFallbackPerformance:
    """Benchmark fallback to scipy for unsupported distributions.

    Verifies that fallback doesn't add significant overhead beyond
    the scipy call itself.
    """

    def test_fallback_pareto(self, benchmark, q_100k):
        """Benchmark fallback for Pareto distribution."""
        params = (2.0, 0.0, 1.0)  # shape, loc, scale

        # Verify this uses fallback
        assert not has_fast_ppf("pareto")

        def run():
            return fast_ppf("pareto", params, q_100k)

        result = benchmark(run)
        assert len(result) == 100_000

    def test_direct_scipy_pareto(self, benchmark, q_100k):
        """Benchmark direct scipy Pareto (for fallback comparison)."""
        params = (2.0, 0.0, 1.0)

        def run():
            return st.pareto.ppf(q_100k, *params)

        result = benchmark(run)
        assert len(result) == 100_000


# =============================================================================
# End-to-end Copula Sampling Benchmarks
# =============================================================================


class TestCopulaSamplingWithFastPPF:
    """End-to-end benchmarks for copula.sample() with fast_ppf optimization.

    These benchmarks measure the real-world speedup when using fast_ppf in
    copula sampling. The fast_ppf module claims ~28x speedup over scipy's
    generic PPF machinery; these tests validate that claim in context.

    The comparison is between:
    - Distributions with fast_ppf support (norm, gamma) - uses direct formulas
    - Distributions without fast_ppf support (pareto) - falls back to scipy
    """

    @pytest.fixture
    def copula_fast_ppf(self):
        """Create a 2-column copula using distributions with fast_ppf support."""
        from spark_bestfit.copula import GaussianCopula
        from spark_bestfit.results import DistributionFitResult

        # Use norm and gamma - both have fast_ppf implementations
        marginals = {
            "col_norm": DistributionFitResult(
                distribution="norm",
                parameters=[0.0, 1.0],  # loc, scale
                sse=0.0,
            ),
            "col_gamma": DistributionFitResult(
                distribution="gamma",
                parameters=[2.0, 0.0, 1.0],  # a, loc, scale
                sse=0.0,
            ),
        }

        # Simple identity-like correlation (slightly correlated)
        correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

        return GaussianCopula(
            column_names=["col_norm", "col_gamma"],
            marginals=marginals,
            correlation_matrix=correlation_matrix,
        )

    @pytest.fixture
    def copula_scipy_fallback(self):
        """Create a 2-column copula using distributions WITHOUT fast_ppf support."""
        from spark_bestfit.copula import GaussianCopula
        from spark_bestfit.results import DistributionFitResult

        # Use pareto and chi2 - neither have fast_ppf implementations
        marginals = {
            "col_pareto": DistributionFitResult(
                distribution="pareto",
                parameters=[2.0, 0.0, 1.0],  # b (shape), loc, scale
                sse=0.0,
            ),
            "col_chi2": DistributionFitResult(
                distribution="chi2",
                parameters=[5.0, 0.0, 1.0],  # df, loc, scale
                sse=0.0,
            ),
        }

        # Verify these don't have fast_ppf
        assert not has_fast_ppf("pareto")
        assert not has_fast_ppf("chi2")

        correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

        return GaussianCopula(
            column_names=["col_pareto", "col_chi2"],
            marginals=marginals,
            correlation_matrix=correlation_matrix,
        )

    # -------------------------------------------------------------------------
    # Benchmarks with fast_ppf enabled (norm, gamma)
    # -------------------------------------------------------------------------

    def test_copula_sample_1k_fast_ppf(self, benchmark, copula_fast_ppf):
        """Copula sample 1K with fast_ppf-supported distributions."""

        def run():
            return copula_fast_ppf.sample(n=1_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_norm"]) == 1_000
        assert len(result["col_gamma"]) == 1_000

    def test_copula_sample_10k_fast_ppf(self, benchmark, copula_fast_ppf):
        """Copula sample 10K with fast_ppf-supported distributions."""

        def run():
            return copula_fast_ppf.sample(n=10_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_norm"]) == 10_000

    def test_copula_sample_100k_fast_ppf(self, benchmark, copula_fast_ppf):
        """Copula sample 100K with fast_ppf-supported distributions."""

        def run():
            return copula_fast_ppf.sample(n=100_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_norm"]) == 100_000

    def test_copula_sample_1m_fast_ppf(self, benchmark, copula_fast_ppf):
        """Copula sample 1M with fast_ppf-supported distributions."""

        def run():
            return copula_fast_ppf.sample(n=1_000_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_norm"]) == 1_000_000

    # -------------------------------------------------------------------------
    # Benchmarks with scipy fallback (pareto, chi2)
    # -------------------------------------------------------------------------

    def test_copula_sample_1k_scipy_fallback(self, benchmark, copula_scipy_fallback):
        """Copula sample 1K with scipy fallback distributions."""

        def run():
            return copula_scipy_fallback.sample(n=1_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_pareto"]) == 1_000

    def test_copula_sample_10k_scipy_fallback(self, benchmark, copula_scipy_fallback):
        """Copula sample 10K with scipy fallback distributions."""

        def run():
            return copula_scipy_fallback.sample(n=10_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_pareto"]) == 10_000

    def test_copula_sample_100k_scipy_fallback(self, benchmark, copula_scipy_fallback):
        """Copula sample 100K with scipy fallback distributions."""

        def run():
            return copula_scipy_fallback.sample(n=100_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_pareto"]) == 100_000

    def test_copula_sample_1m_scipy_fallback(self, benchmark, copula_scipy_fallback):
        """Copula sample 1M with scipy fallback distributions."""

        def run():
            return copula_scipy_fallback.sample(n=1_000_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_pareto"]) == 1_000_000


class TestCopulaSamplingForcedFallback:
    """Benchmark copula sampling with fast_ppf forcibly disabled.

    These benchmarks use the SAME distributions (norm, gamma) but force
    scipy fallback by patching has_fast_ppf. This isolates the speedup
    attributable to fast_ppf vs scipy's generic PPF machinery.

    Compare these results with TestCopulaSamplingWithFastPPF to see the
    true impact of fast_ppf optimization.
    """

    @pytest.fixture
    def copula_norm_gamma(self):
        """Create a 2-column copula using norm and gamma distributions."""
        from spark_bestfit.copula import GaussianCopula
        from spark_bestfit.results import DistributionFitResult

        marginals = {
            "col_norm": DistributionFitResult(
                distribution="norm",
                parameters=[0.0, 1.0],
                sse=0.0,
            ),
            "col_gamma": DistributionFitResult(
                distribution="gamma",
                parameters=[2.0, 0.0, 1.0],
                sse=0.0,
            ),
        }

        correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

        return GaussianCopula(
            column_names=["col_norm", "col_gamma"],
            marginals=marginals,
            correlation_matrix=correlation_matrix,
        )

    def test_copula_sample_100k_forced_scipy(self, benchmark, copula_norm_gamma, monkeypatch):
        """Copula sample 100K with fast_ppf DISABLED (forced scipy fallback).

        This uses norm and gamma distributions but patches has_fast_ppf to
        return False, forcing scipy's generic PPF path.

        Compare with test_copula_sample_100k_fast_ppf to see the speedup.
        """
        # Patch has_fast_ppf to always return False
        import spark_bestfit.copula as copula_module

        monkeypatch.setattr(copula_module, "has_fast_ppf", lambda x: False)

        def run():
            return copula_norm_gamma.sample(n=100_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_norm"]) == 100_000

    def test_copula_sample_1m_forced_scipy(self, benchmark, copula_norm_gamma, monkeypatch):
        """Copula sample 1M with fast_ppf DISABLED (forced scipy fallback).

        Compare with test_copula_sample_1m_fast_ppf to see the speedup.
        """
        import spark_bestfit.copula as copula_module

        monkeypatch.setattr(copula_module, "has_fast_ppf", lambda x: False)

        def run():
            return copula_norm_gamma.sample(n=1_000_000, random_state=42)

        result = benchmark(run)
        assert len(result["col_norm"]) == 1_000_000
