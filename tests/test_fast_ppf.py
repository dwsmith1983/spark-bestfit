"""Tests for fast PPF (percent point function) implementations.

These tests verify that the optimized PPF implementations produce results
that match scipy.stats.ppf within acceptable tolerance.
"""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.fast_ppf import fast_ppf, fast_ppf_batch, has_fast_ppf


class TestFastPPFRegistry:
    """Tests for the fast_ppf registry and has_fast_ppf function."""

    def test_has_fast_ppf_supported_distributions(self):
        """Verify has_fast_ppf returns True for supported distributions."""
        supported = ["norm", "expon", "uniform", "lognorm", "weibull_min", "gamma", "beta"]
        for dist in supported:
            assert has_fast_ppf(dist), f"{dist} should be supported"

    def test_has_fast_ppf_unsupported_distributions(self):
        """Verify has_fast_ppf returns False for unsupported distributions."""
        unsupported = ["pareto", "chi2", "t", "f", "laplace"]
        for dist in unsupported:
            assert not has_fast_ppf(dist), f"{dist} should not be supported"


class TestFastPPFAccuracy:
    """Tests verifying fast_ppf matches scipy.stats.ppf."""

    @pytest.fixture
    def quantiles(self):
        """Standard quantiles to test."""
        return np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])

    def test_norm_ppf(self, quantiles):
        """Test normal distribution PPF accuracy."""
        params = (0.0, 1.0)  # loc, scale
        expected = st.norm.ppf(quantiles, *params)
        result = fast_ppf("norm", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_norm_ppf_with_params(self, quantiles):
        """Test normal distribution PPF with non-default params."""
        params = (10.0, 2.5)  # loc=10, scale=2.5
        expected = st.norm.ppf(quantiles, *params)
        result = fast_ppf("norm", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_expon_ppf(self, quantiles):
        """Test exponential distribution PPF accuracy."""
        params = (0.0, 1.0)  # loc, scale
        expected = st.expon.ppf(quantiles, *params)
        result = fast_ppf("expon", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_expon_ppf_with_params(self, quantiles):
        """Test exponential distribution PPF with non-default params."""
        params = (5.0, 3.0)  # loc=5, scale=3
        expected = st.expon.ppf(quantiles, *params)
        result = fast_ppf("expon", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_uniform_ppf(self, quantiles):
        """Test uniform distribution PPF accuracy."""
        params = (0.0, 1.0)  # loc, scale
        expected = st.uniform.ppf(quantiles, *params)
        result = fast_ppf("uniform", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_uniform_ppf_with_params(self, quantiles):
        """Test uniform distribution PPF with non-default params."""
        params = (10.0, 50.0)  # [10, 60]
        expected = st.uniform.ppf(quantiles, *params)
        result = fast_ppf("uniform", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_lognorm_ppf(self, quantiles):
        """Test lognormal distribution PPF accuracy."""
        params = (0.5, 0.0, 1.0)  # s=0.5, loc=0, scale=1
        expected = st.lognorm.ppf(quantiles, *params)
        result = fast_ppf("lognorm", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_lognorm_ppf_with_params(self, quantiles):
        """Test lognormal distribution PPF with various params."""
        params = (1.2, 5.0, 2.0)  # s=1.2, loc=5, scale=2
        expected = st.lognorm.ppf(quantiles, *params)
        result = fast_ppf("lognorm", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_weibull_min_ppf(self, quantiles):
        """Test Weibull minimum distribution PPF accuracy."""
        params = (2.0, 0.0, 1.0)  # c=2, loc=0, scale=1
        expected = st.weibull_min.ppf(quantiles, *params)
        result = fast_ppf("weibull_min", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_weibull_min_ppf_with_params(self, quantiles):
        """Test Weibull minimum distribution PPF with various params."""
        params = (1.5, 2.0, 3.0)  # c=1.5, loc=2, scale=3
        expected = st.weibull_min.ppf(quantiles, *params)
        result = fast_ppf("weibull_min", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_ppf(self, quantiles):
        """Test gamma distribution PPF accuracy."""
        params = (2.0, 0.0, 1.0)  # a=2, loc=0, scale=1
        expected = st.gamma.ppf(quantiles, *params)
        result = fast_ppf("gamma", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_ppf_with_params(self, quantiles):
        """Test gamma distribution PPF with various params."""
        params = (5.0, 1.0, 2.0)  # a=5, loc=1, scale=2
        expected = st.gamma.ppf(quantiles, *params)
        result = fast_ppf("gamma", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_beta_ppf(self, quantiles):
        """Test beta distribution PPF accuracy."""
        params = (2.0, 5.0, 0.0, 1.0)  # a=2, b=5, loc=0, scale=1
        expected = st.beta.ppf(quantiles, *params)
        result = fast_ppf("beta", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_beta_ppf_with_params(self, quantiles):
        """Test beta distribution PPF with various params."""
        params = (0.5, 0.5, 10.0, 20.0)  # a=0.5, b=0.5, loc=10, scale=20
        expected = st.beta.ppf(quantiles, *params)
        result = fast_ppf("beta", params, quantiles)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestFastPPFFallback:
    """Tests for fallback behavior with unsupported distributions."""

    def test_fallback_to_scipy(self):
        """Verify fallback to scipy for unsupported distributions."""
        quantiles = np.array([0.1, 0.5, 0.9])
        params = (2.0, 0.0, 1.0)  # shape, loc, scale

        # Pareto is not in fast_ppf registry
        expected = st.pareto.ppf(quantiles, *params)
        result = fast_ppf("pareto", params, quantiles)

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestFastPPFTruncation:
    """Tests for truncated distribution support."""

    @pytest.fixture
    def quantiles(self):
        return np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    def test_truncated_norm_lower_bound(self, quantiles):
        """Test truncated normal with lower bound only."""
        params = (0.0, 1.0)  # loc=0, scale=1
        lb = 0.0  # truncate at 0

        # Compute expected using scipy
        frozen = st.norm(*params)
        cdf_lb = frozen.cdf(lb)
        q_mapped = cdf_lb + quantiles * (1.0 - cdf_lb)
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("norm", params, quantiles, lb=lb)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_norm_upper_bound(self, quantiles):
        """Test truncated normal with upper bound only."""
        params = (0.0, 1.0)
        ub = 0.0  # truncate at 0 (upper)

        frozen = st.norm(*params)
        cdf_ub = frozen.cdf(ub)
        q_mapped = quantiles * cdf_ub
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("norm", params, quantiles, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_norm_both_bounds(self, quantiles):
        """Test truncated normal with both bounds."""
        params = (0.0, 1.0)
        lb = -1.0
        ub = 1.0

        frozen = st.norm(*params)
        cdf_lb = frozen.cdf(lb)
        cdf_ub = frozen.cdf(ub)
        norm = cdf_ub - cdf_lb
        q_mapped = cdf_lb + quantiles * norm
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("norm", params, quantiles, lb=lb, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_expon_lower_bound(self, quantiles):
        """Test truncated exponential with lower bound."""
        params = (0.0, 2.0)  # loc=0, scale=2
        lb = 1.0

        frozen = st.expon(*params)
        cdf_lb = frozen.cdf(lb)
        q_mapped = cdf_lb + quantiles * (1.0 - cdf_lb)
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("expon", params, quantiles, lb=lb)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestFastPPFBatch:
    """Tests for batch PPF processing."""

    def test_batch_multiple_distributions(self):
        """Test batch processing with multiple distributions."""
        distributions = ["norm", "expon", "gamma"]
        params_list = [(0.0, 1.0), (0.0, 2.0), (2.0, 0.0, 1.0)]
        quantiles = np.array([0.1, 0.5, 0.9])
        q_arrays = [quantiles, quantiles, quantiles]

        results = fast_ppf_batch(distributions, params_list, q_arrays)

        assert len(results) == 3

        # Verify each result
        expected_norm = st.norm.ppf(quantiles, 0.0, 1.0)
        expected_expon = st.expon.ppf(quantiles, 0.0, 2.0)
        expected_gamma = st.gamma.ppf(quantiles, 2.0, 0.0, 1.0)

        np.testing.assert_allclose(results[0], expected_norm, rtol=1e-10)
        np.testing.assert_allclose(results[1], expected_expon, rtol=1e-10)
        np.testing.assert_allclose(results[2], expected_gamma, rtol=1e-10)

    def test_batch_with_truncation(self):
        """Test batch processing with truncation bounds."""
        distributions = ["norm", "norm"]
        params_list = [(0.0, 1.0), (0.0, 1.0)]
        quantiles = np.array([0.25, 0.5, 0.75])
        q_arrays = [quantiles, quantiles]
        lb_list = [0.0, None]
        ub_list = [None, 0.0]

        results = fast_ppf_batch(distributions, params_list, q_arrays, lb_list, ub_list)

        assert len(results) == 2
        # First result should be truncated at lower bound
        assert np.all(results[0] >= 0.0)
        # Second result should be truncated at upper bound
        assert np.all(results[1] <= 0.0)


class TestFastPPFEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_extreme_quantiles(self):
        """Test with extreme quantile values."""
        params = (0.0, 1.0)
        q = np.array([1e-10, 1 - 1e-10])

        result = fast_ppf("norm", params, q)
        expected = st.norm.ppf(q, *params)

        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_single_quantile(self):
        """Test with a single quantile value."""
        params = (0.0, 1.0)
        q = np.array([0.5])

        result = fast_ppf("norm", params, q)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_large_array(self):
        """Test with large quantile array for performance."""
        params = (0.0, 1.0)
        q = np.linspace(0.01, 0.99, 10000)

        result = fast_ppf("norm", params, q)
        expected = st.norm.ppf(q, *params)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_weibull_small_shape(self):
        """Test Weibull with small shape parameter (heavy tail)."""
        params = (0.5, 0.0, 1.0)  # c=0.5 (heavy tail)
        q = np.array([0.1, 0.5, 0.9])

        result = fast_ppf("weibull_min", params, q)
        expected = st.weibull_min.ppf(q, *params)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_small_shape(self):
        """Test gamma with small shape parameter."""
        params = (0.5, 0.0, 1.0)  # a=0.5
        q = np.array([0.1, 0.5, 0.9])

        result = fast_ppf("gamma", params, q)
        expected = st.gamma.ppf(q, *params)

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestFastPPFPerformance:
    """Performance sanity checks for fast_ppf.

    These are basic timing checks to catch performance regressions.
    For detailed benchmarks, run: pytest tests/benchmarks/test_benchmark_fast_ppf.py

    Expected speedups (100K elements):
        - uniform:     ~16x faster (linear transformation)
        - weibull_min: ~2.7x faster (closed-form)
        - expon:       ~2.3x faster (closed-form)
        - norm:        ~1.5x faster (direct ndtri)
        - lognorm:     ~1.3x faster (exp of ndtri)
        - gamma:       ~1.0x (same scipy.special function)
        - beta:        ~1.0x (same scipy.special function)
    """

    def test_uniform_significantly_faster(self):
        """Verify uniform PPF is significantly faster than scipy.

        Uniform has a trivial closed-form PPF and should be ~16x faster.
        We test for at least 5x to account for system variance.
        """
        import time

        params = (0.0, 1.0)
        q = np.random.uniform(0.01, 0.99, 100000)

        # Warm up
        fast_ppf("uniform", params, q)
        st.uniform.ppf(q, *params)

        # Time fast_ppf
        start = time.perf_counter()
        for _ in range(10):
            fast_ppf("uniform", params, q)
        fast_time = time.perf_counter() - start

        # Time scipy
        start = time.perf_counter()
        for _ in range(10):
            st.uniform.ppf(q, *params)
        scipy_time = time.perf_counter() - start

        speedup = scipy_time / fast_time
        # Expect at least 5x speedup (conservative to avoid flaky tests)
        assert speedup >= 5.0, (
            f"Expected uniform to be at least 5x faster, got {speedup:.1f}x "
            f"(fast: {fast_time:.3f}s, scipy: {scipy_time:.3f}s)"
        )

    def test_norm_faster_than_scipy(self):
        """Verify fast_ppf is faster than scipy for normal distribution.

        Normal uses scipy.special.ndtri and should be ~1.5x faster.
        We test for at least 1.1x to account for system variance.
        """
        import time

        params = (0.0, 1.0)
        q = np.random.uniform(0.01, 0.99, 100000)

        # Warm up
        fast_ppf("norm", params, q)
        st.norm.ppf(q, *params)

        # Time fast_ppf
        start = time.perf_counter()
        for _ in range(10):
            fast_ppf("norm", params, q)
        fast_time = time.perf_counter() - start

        # Time scipy
        start = time.perf_counter()
        for _ in range(10):
            st.norm.ppf(q, *params)
        scipy_time = time.perf_counter() - start

        speedup = scipy_time / fast_time
        # Expect at least 1.1x speedup (conservative to avoid flaky tests)
        assert speedup >= 1.1, (
            f"Expected norm to be at least 1.1x faster, got {speedup:.1f}x "
            f"(fast: {fast_time:.3f}s, scipy: {scipy_time:.3f}s)"
        )

    def test_gamma_not_slower_than_scipy(self):
        """Verify fast_ppf is not slower than scipy for gamma distribution.

        Gamma uses scipy.special.gammaincinv which is the same function
        scipy.stats uses internally, so no speedup is expected. We verify
        it's at least not significantly slower.
        """
        import time

        params = (2.0, 0.0, 1.0)
        q = np.random.uniform(0.01, 0.99, 100000)

        # Warm up
        fast_ppf("gamma", params, q)
        st.gamma.ppf(q, *params)

        # Time fast_ppf
        start = time.perf_counter()
        for _ in range(5):
            fast_ppf("gamma", params, q)
        fast_time = time.perf_counter() - start

        # Time scipy
        start = time.perf_counter()
        for _ in range(5):
            st.gamma.ppf(q, *params)
        scipy_time = time.perf_counter() - start

        # fast_ppf should not be more than 20% slower
        assert fast_time <= scipy_time * 1.2, (
            f"fast_ppf for gamma is too slow: {fast_time:.3f}s vs scipy {scipy_time:.3f}s"
        )
