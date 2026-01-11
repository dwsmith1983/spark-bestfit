"""Tests for fast PPF (percent point function) implementations.

These tests verify that the optimized PPF implementations produce results
that match scipy.stats.ppf within acceptable tolerance.
"""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.fast_ppf import (
    clear_ppf_cache,
    fast_ppf,
    fast_ppf_batch,
    has_fast_ppf,
)


class TestFastPPFRegistry:
    """Tests for the fast_ppf registry and has_fast_ppf function."""

    def test_has_fast_ppf_supported_distributions(self):
        """Verify has_fast_ppf returns True for supported distributions."""
        supported = [
            "norm",
            "expon",
            "uniform",
            "lognorm",
            "weibull_min",
            "gamma",
            "beta",
        ]
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


class TestFastPPFParameterEdgeCases:
    """Tests for edge cases with different parameter counts.

    Each distribution function has branches for handling:
    - Full parameters (all shape + loc + scale)
    - Minimal parameters (shape only)
    - Empty parameters (defaults)

    These tests ensure all branches are exercised.
    """

    @pytest.fixture
    def quantiles(self):
        return np.array([0.1, 0.5, 0.9])

    # Normal distribution parameter cases
    def test_norm_no_params(self, quantiles):
        """Test normal with empty params (defaults to loc=0, scale=1)."""
        params = ()
        result = fast_ppf("norm", params, quantiles)
        expected = st.norm.ppf(quantiles, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_norm_one_param(self, quantiles):
        """Test normal with one param (loc only, scale defaults to 1)."""
        params = (5.0,)
        result = fast_ppf("norm", params, quantiles)
        expected = st.norm.ppf(quantiles, 5.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # Exponential distribution parameter cases
    def test_expon_no_params(self, quantiles):
        """Test exponential with empty params (defaults to loc=0, scale=1)."""
        params = ()
        result = fast_ppf("expon", params, quantiles)
        expected = st.expon.ppf(quantiles, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_expon_one_param(self, quantiles):
        """Test exponential with one param (loc only)."""
        params = (2.0,)
        result = fast_ppf("expon", params, quantiles)
        expected = st.expon.ppf(quantiles, 2.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # Uniform distribution parameter cases
    def test_uniform_no_params(self, quantiles):
        """Test uniform with empty params (defaults to loc=0, scale=1)."""
        params = ()
        result = fast_ppf("uniform", params, quantiles)
        expected = st.uniform.ppf(quantiles, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_uniform_one_param(self, quantiles):
        """Test uniform with one param (loc only)."""
        params = (10.0,)
        result = fast_ppf("uniform", params, quantiles)
        expected = st.uniform.ppf(quantiles, 10.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    # Lognormal distribution parameter cases
    def test_lognorm_one_param(self, quantiles):
        """Test lognormal with shape only (s=0.5, loc=0, scale=1)."""
        params = (0.5,)
        result = fast_ppf("lognorm", params, quantiles)
        expected = st.lognorm.ppf(quantiles, 0.5, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_lognorm_two_params(self, quantiles):
        """Test lognormal with shape and scale (s, scale), loc=0."""
        params = (0.5, 2.0)  # s=0.5, scale=2.0
        result = fast_ppf("lognorm", params, quantiles)
        expected = st.lognorm.ppf(quantiles, 0.5, 0.0, 2.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # Weibull minimum distribution parameter cases
    def test_weibull_min_one_param(self, quantiles):
        """Test Weibull with shape only (c=2, loc=0, scale=1)."""
        params = (2.0,)
        result = fast_ppf("weibull_min", params, quantiles)
        expected = st.weibull_min.ppf(quantiles, 2.0, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_weibull_min_two_params(self, quantiles):
        """Test Weibull with shape and scale (c, scale), loc=0."""
        params = (1.5, 3.0)  # c=1.5, scale=3.0
        result = fast_ppf("weibull_min", params, quantiles)
        expected = st.weibull_min.ppf(quantiles, 1.5, 0.0, 3.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # Gamma distribution parameter cases
    def test_gamma_one_param(self, quantiles):
        """Test gamma with shape only (a=2, loc=0, scale=1)."""
        params = (2.0,)
        result = fast_ppf("gamma", params, quantiles)
        expected = st.gamma.ppf(quantiles, 2.0, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gamma_two_params(self, quantiles):
        """Test gamma with shape and scale (a, scale), loc=0."""
        params = (2.0, 3.0)  # a=2.0, scale=3.0
        result = fast_ppf("gamma", params, quantiles)
        expected = st.gamma.ppf(quantiles, 2.0, 0.0, 3.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # Beta distribution parameter cases
    def test_beta_two_params(self, quantiles):
        """Test beta with minimal params (a, b), defaults loc=0, scale=1."""
        params = (2.0, 5.0)
        result = fast_ppf("beta", params, quantiles)
        expected = st.beta.ppf(quantiles, 2.0, 5.0, 0.0, 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_beta_three_params(self, quantiles):
        """Test beta with (a, b, scale), loc=0."""
        params = (2.0, 5.0, 10.0)  # a=2, b=5, scale=10
        result = fast_ppf("beta", params, quantiles)
        expected = st.beta.ppf(quantiles, 2.0, 5.0, 0.0, 10.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_beta_insufficient_params_raises(self):
        """Test that beta with < 2 params raises ValueError."""
        quantiles = np.array([0.5])
        with pytest.raises(ValueError, match="at least 2 shape parameters"):
            fast_ppf("beta", (2.0,), quantiles)


class TestFastPPFCacheManagement:
    """Tests for cache clearing functionality."""

    def test_clear_ppf_cache(self):
        """Test that clear_ppf_cache runs without error."""
        # First populate the cache
        quantiles = np.array([0.5])
        fast_ppf("norm", (0.0, 1.0), quantiles, lb=0.0)
        fast_ppf("norm", (0.0, 1.0), quantiles, lb=0.0, ub=1.0)

        # Clear should not raise
        clear_ppf_cache()

        # Operations should still work after clearing
        result = fast_ppf("norm", (0.0, 1.0), quantiles, lb=0.0)
        assert len(result) == 1

    def test_cache_hit_after_repeated_calls(self):
        """Verify cache is being used for repeated calls."""
        from spark_bestfit.fast_ppf import _cached_truncation_bounds

        # Clear cache first
        clear_ppf_cache()

        # First call should be a miss
        params = (0.0, 1.0)
        _ = _cached_truncation_bounds("norm", params, 0.0, 1.0)
        info = _cached_truncation_bounds.cache_info()
        first_misses = info.misses

        # Second call with same args should hit cache
        _ = _cached_truncation_bounds("norm", params, 0.0, 1.0)
        info = _cached_truncation_bounds.cache_info()
        assert info.hits >= 1, "Cache should have at least one hit"
        assert info.misses == first_misses, "No new cache misses expected"


class TestFastPPFDegenerateTruncation:
    """Tests for degenerate truncation cases."""

    def test_truncation_zero_range(self):
        """Test when lower and upper bounds are equal (degenerate case)."""
        quantiles = np.array([0.1, 0.5, 0.9])
        params = (0.0, 1.0)
        lb = 0.5
        ub = 0.5  # Same as lb - zero range

        result = fast_ppf("norm", params, quantiles, lb=lb, ub=ub)

        # With zero range, all quantiles should map to cdf_lb
        # which means PPF should return values at that single CDF point
        assert len(result) == 3
        # All results should be identical since the range is degenerate
        np.testing.assert_allclose(result, result[0] * np.ones(3), rtol=1e-10)

    def test_truncation_inverted_bounds(self):
        """Test when upper < lower (impossible range)."""
        quantiles = np.array([0.5])
        params = (0.0, 1.0)
        lb = 1.0
        ub = -1.0  # Upper below lower - degenerate

        result = fast_ppf("norm", params, quantiles, lb=lb, ub=ub)

        # Should handle gracefully without crashing
        assert len(result) == 1

    def test_truncation_with_infinite_bounds(self):
        """Test truncation with explicit infinite bounds (should behave like no truncation)."""
        quantiles = np.array([0.1, 0.5, 0.9])
        params = (0.0, 1.0)

        # Explicit infinities
        result_inf = fast_ppf(
            "norm", params, quantiles, lb=float("-inf"), ub=float("inf")
        )
        # No truncation
        result_none = fast_ppf("norm", params, quantiles)

        np.testing.assert_allclose(result_inf, result_none, rtol=1e-10)


class TestFastPPFAdditionalTruncation:
    """Additional truncation edge cases for full coverage."""

    @pytest.fixture
    def quantiles(self):
        return np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    def test_truncated_gamma_both_bounds(self, quantiles):
        """Test truncated gamma with both bounds."""
        params = (2.0, 0.0, 1.0)  # a=2, loc=0, scale=1
        lb = 0.5
        ub = 5.0

        frozen = st.gamma(*params)
        cdf_lb = frozen.cdf(lb)
        cdf_ub = frozen.cdf(ub)
        norm = cdf_ub - cdf_lb
        q_mapped = cdf_lb + quantiles * norm
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("gamma", params, quantiles, lb=lb, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_beta_both_bounds(self, quantiles):
        """Test truncated beta with both bounds."""
        params = (2.0, 5.0, 0.0, 1.0)  # a=2, b=5, loc=0, scale=1
        lb = 0.1
        ub = 0.8

        frozen = st.beta(*params)
        cdf_lb = frozen.cdf(lb)
        cdf_ub = frozen.cdf(ub)
        norm = cdf_ub - cdf_lb
        q_mapped = cdf_lb + quantiles * norm
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("beta", params, quantiles, lb=lb, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_uniform_both_bounds(self, quantiles):
        """Test truncated uniform with both bounds."""
        params = (0.0, 10.0)  # loc=0, scale=10 -> [0, 10]
        lb = 2.0
        ub = 8.0

        frozen = st.uniform(*params)
        cdf_lb = frozen.cdf(lb)
        cdf_ub = frozen.cdf(ub)
        norm = cdf_ub - cdf_lb
        q_mapped = cdf_lb + quantiles * norm
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("uniform", params, quantiles, lb=lb, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_lognorm_lower_only(self, quantiles):
        """Test truncated lognormal with lower bound only."""
        params = (0.5, 0.0, 1.0)  # s=0.5, loc=0, scale=1
        lb = 0.5

        frozen = st.lognorm(*params)
        cdf_lb = frozen.cdf(lb)
        q_mapped = cdf_lb + quantiles * (1.0 - cdf_lb)
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("lognorm", params, quantiles, lb=lb)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_weibull_upper_only(self, quantiles):
        """Test truncated Weibull with upper bound only."""
        params = (2.0, 0.0, 1.0)  # c=2, loc=0, scale=1
        ub = 2.0

        frozen = st.weibull_min(*params)
        cdf_ub = frozen.cdf(ub)
        q_mapped = quantiles * cdf_ub
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("weibull_min", params, quantiles, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_truncated_fallback_distribution(self, quantiles):
        """Test truncation works with scipy fallback distributions."""
        # Pareto is not in fast_ppf registry, will use scipy fallback
        params = (2.0, 0.0, 1.0)  # shape, loc, scale
        lb = 1.5
        ub = 5.0

        frozen = st.pareto(*params)
        cdf_lb = frozen.cdf(lb)
        cdf_ub = frozen.cdf(ub)
        norm = cdf_ub - cdf_lb
        q_mapped = cdf_lb + quantiles * norm
        expected = frozen.ppf(q_mapped)

        result = fast_ppf("pareto", params, quantiles, lb=lb, ub=ub)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
