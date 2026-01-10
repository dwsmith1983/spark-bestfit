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
