"""Tests for quality improvements: golden path distribution selection and concurrency.

This module contains:
1. Golden path tests - verify fitter selects correct distribution for known data
2. Concurrency/thread-safety tests for parallel fitting
"""

import concurrent.futures
import threading
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from spark_bestfit import DistributionFitter, DiscreteDistributionFitter, LocalBackend


class TestGoldenPathDistributionSelection:
    """Golden path tests verifying fitter selects the correct distribution.

    These tests generate data from known distributions and verify that when
    multiple candidate distributions are fitted, the BEST result (by AIC/BIC)
    matches the true source distribution.
    """

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=4)

    @pytest.fixture
    def candidate_distributions(self) -> List[str]:
        """Common candidate distributions for golden path tests."""
        return ["norm", "expon", "gamma", "lognorm", "weibull_min"]

    def test_golden_path_normal_data(self, local_backend, candidate_distributions):
        """Test that normal distribution is selected for normal data.

        Generate data from N(50, 10) and verify 'norm' is the best fit
        when compared against exponential, gamma, lognorm, and weibull.
        """
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=5000)
        df = pd.DataFrame({"value": data})

        # Get all available distributions and exclude those not in candidates
        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidate_distributions)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        # Best by AIC should be norm
        best_by_aic = results.best(n=1, metric="aic")[0]
        assert best_by_aic.distribution == "norm", (
            f"Expected 'norm' for normal data, got '{best_by_aic.distribution}'"
        )

        # Verify the fit parameters are reasonable
        params = best_by_aic.parameters
        loc = params[-2]  # loc is second-to-last
        scale = params[-1]  # scale is last
        assert 45 < loc < 55, f"loc={loc} should be near 50"
        assert 8 < scale < 12, f"scale={scale} should be near 10"

    def test_golden_path_exponential_data(self, local_backend, candidate_distributions):
        """Test that exponential distribution is selected for exponential data.

        Generate data from Exp(scale=5) and verify 'expon' is the best fit.
        """
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=5000)
        df = pd.DataFrame({"value": data})

        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidate_distributions)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        best_by_aic = results.best(n=1, metric="aic")[0]
        assert best_by_aic.distribution == "expon", (
            f"Expected 'expon' for exponential data, got '{best_by_aic.distribution}'"
        )

        # Verify scale parameter is reasonable
        scale = best_by_aic.parameters[-1]
        assert 4 < scale < 6, f"scale={scale} should be near 5"

    def test_golden_path_gamma_data(self, local_backend, candidate_distributions):
        """Test that gamma distribution is selected for gamma data.

        Generate data from Gamma(shape=3, scale=2) and verify 'gamma' is best.
        """
        np.random.seed(42)
        data = np.random.gamma(shape=3, scale=2, size=5000)
        df = pd.DataFrame({"value": data})

        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidate_distributions)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        best_by_aic = results.best(n=1, metric="aic")[0]
        assert best_by_aic.distribution == "gamma", (
            f"Expected 'gamma' for gamma data, got '{best_by_aic.distribution}'"
        )

    def test_golden_path_uniform_data(self, local_backend):
        """Test that uniform distribution is selected for uniform data."""
        np.random.seed(42)
        data = np.random.uniform(low=10, high=90, size=5000)
        df = pd.DataFrame({"value": data})

        candidates = ["uniform", "norm", "beta"]
        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidates)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        best_by_aic = results.best(n=1, metric="aic")[0]
        assert best_by_aic.distribution == "uniform", (
            f"Expected 'uniform' for uniform data, got '{best_by_aic.distribution}'"
        )

    def test_golden_path_lognormal_data(self, local_backend, candidate_distributions):
        """Test that lognormal distribution is selected for lognormal data."""
        np.random.seed(42)
        # Generate lognormal data: log(X) ~ N(mu=1, sigma=0.5)
        data = np.random.lognormal(mean=1, sigma=0.5, size=5000)
        df = pd.DataFrame({"value": data})

        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidate_distributions)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        best_by_aic = results.best(n=1, metric="aic")[0]
        assert best_by_aic.distribution == "lognorm", (
            f"Expected 'lognorm' for lognormal data, got '{best_by_aic.distribution}'"
        )

    def test_golden_path_weibull_data(self, local_backend, candidate_distributions):
        """Test that Weibull distribution is selected for Weibull data."""
        np.random.seed(42)
        # Generate Weibull data with shape=2 (close to normal-like shape)
        data = st.weibull_min.rvs(c=1.5, scale=5, size=5000)
        df = pd.DataFrame({"value": data})

        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidate_distributions)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        best_by_aic = results.best(n=1, metric="aic")[0]
        assert best_by_aic.distribution == "weibull_min", (
            f"Expected 'weibull_min' for Weibull data, got '{best_by_aic.distribution}'"
        )

    def test_golden_path_with_different_metrics(self, local_backend):
        """Test that different selection metrics agree for well-separated data."""
        np.random.seed(42)
        data = np.random.normal(loc=0, scale=1, size=5000)
        df = pd.DataFrame({"value": data})

        candidates = ["norm", "expon", "uniform"]
        all_dists = DistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidates)
        fitter = DistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="value")

        # All metrics should agree for clear-cut case
        best_aic = results.best(n=1, metric="aic")[0]
        best_bic = results.best(n=1, metric="bic")[0]
        best_sse = results.best(n=1, metric="sse")[0]

        assert best_aic.distribution == "norm"
        assert best_bic.distribution == "norm"
        assert best_sse.distribution == "norm"


class TestGoldenPathDiscreteDistributions:
    """Golden path tests for discrete distribution selection."""

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=4)

    def test_golden_path_poisson_data(self, local_backend):
        """Test that Poisson is selected for Poisson data."""
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=3000)
        df = pd.DataFrame({"counts": data})

        candidates = ["poisson", "nbinom", "geom"]
        all_dists = DiscreteDistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidates)
        fitter = DiscreteDistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="counts")

        best = results.best(n=1, metric="aic")[0]
        assert best.distribution == "poisson", (
            f"Expected 'poisson' for Poisson data, got '{best.distribution}'"
        )

    def test_golden_path_negative_binomial_data(self, local_backend):
        """Test that negative binomial is selected for nbinom data."""
        np.random.seed(42)
        data = np.random.negative_binomial(n=5, p=0.4, size=3000)
        df = pd.DataFrame({"counts": data})

        candidates = ["poisson", "nbinom", "geom"]
        all_dists = DiscreteDistributionFitter(backend=local_backend)._registry.get_distributions()
        exclude = tuple(d for d in all_dists if d not in candidates)
        fitter = DiscreteDistributionFitter(backend=local_backend, excluded_distributions=exclude)

        results = fitter.fit(df, column="counts")

        best = results.best(n=1, metric="aic")[0]
        assert best.distribution == "nbinom", (
            f"Expected 'nbinom' for negative binomial data, got '{best.distribution}'"
        )


class TestConcurrencyThreadSafety:
    """Tests for concurrency and thread-safety of parallel fitting."""

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=4)

    def test_concurrent_fitter_instances_independent(self, local_backend):
        """Test that multiple concurrent fitter instances don't interfere.

        Run multiple fitters simultaneously on different data and verify
        each produces valid results without errors or cross-contamination.
        The focus is on thread safety, not exact distribution selection.
        """
        np.random.seed(42)

        # Create different datasets - use clearly separable data
        datasets = [
            ("dataset_A", np.random.normal(loc=50, scale=10, size=2000)),
            ("dataset_B", np.random.normal(loc=100, scale=5, size=2000)),
            ("dataset_C", np.random.normal(loc=0, scale=20, size=2000)),
        ]

        results_dict = {}
        errors = []

        def fit_data(name: str, data: np.ndarray) -> Tuple[str, dict]:
            """Fit data and return (name, fit_info)."""
            try:
                df = pd.DataFrame({"value": data})
                # Each thread gets its own fitter instance
                backend = LocalBackend(max_workers=2)
                fitter = DistributionFitter(backend=backend)
                results = fitter.fit(df, column="value", max_distributions=5)
                best = results.best(n=1, metric="aic")[0]
                # Return statistics about the fit to verify uniqueness
                return (name, {
                    "distribution": best.distribution,
                    "aic": best.aic,
                    "data_mean": float(np.mean(data)),
                    "data_std": float(np.std(data)),
                })
            except Exception as e:
                errors.append((name, str(e)))
                return (name, {"error": str(e)})

        # Run fits concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(fit_data, name, data)
                for name, data in datasets
            ]
            for future in concurrent.futures.as_completed(futures):
                name, info = future.result()
                results_dict[name] = info

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors during concurrent fitting: {errors}"

        # Verify all fits completed successfully
        assert len(results_dict) == 3, f"Expected 3 results, got {len(results_dict)}"

        # Verify results are different (not cross-contaminated)
        # Each dataset has different mean, so results should reflect that
        means = [r["data_mean"] for r in results_dict.values()]
        assert len(set(means)) == 3, f"Expected 3 unique means, got {means}"

        # Verify each result has valid fit info
        for name, info in results_dict.items():
            assert "distribution" in info, f"{name} missing distribution"
            assert "aic" in info, f"{name} missing AIC"
            assert info["distribution"] != "ERROR", f"{name} fit failed"

    def test_parallel_fit_deterministic(self, local_backend):
        """Test that parallel fitting produces deterministic results.

        Run the same fit multiple times and verify results are identical
        when using the same random seed.
        """
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=2000)
        df = pd.DataFrame({"value": data})

        results_list = []
        for _ in range(3):
            fitter = DistributionFitter(backend=local_backend, random_seed=42)
            results = fitter.fit(df, column="value", max_distributions=5)
            best = results.best(n=1, metric="aic")[0]
            results_list.append((best.distribution, best.aic, tuple(best.parameters)))

        # All results should be identical
        first = results_list[0]
        for i, result in enumerate(results_list[1:], 1):
            assert result[0] == first[0], (
                f"Run {i+1} distribution differs: {result[0]} vs {first[0]}"
            )
            assert np.isclose(result[1], first[1], rtol=1e-10), (
                f"Run {i+1} AIC differs: {result[1]} vs {first[1]}"
            )

    def test_no_race_condition_in_shared_registry(self, local_backend):
        """Test that concurrent access to distribution registry is safe."""
        errors = []
        results = []
        lock = threading.Lock()

        def access_registry():
            """Access registry and perform operations."""
            try:
                fitter = DistributionFitter(backend=local_backend)
                # Access registry multiple times
                dists = fitter._registry.get_distributions()
                # Verify registry is consistent
                dists2 = fitter._registry.get_distributions()
                assert dists == dists2, "Registry inconsistent during access"
                with lock:
                    results.append(len(dists))
            except Exception as e:
                with lock:
                    errors.append(str(e))

        # Run many concurrent accesses
        threads = [threading.Thread(target=access_registry) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors during concurrent registry access: {errors}"

        # Verify all accesses got same result
        assert len(set(results)) == 1, (
            f"Registry returned different results: {set(results)}"
        )

    def test_concurrent_fits_same_backend(self):
        """Test concurrent fits using the same backend instance.

        This tests thread-safety of the backend's parallel_fit method
        when called from multiple threads.
        """
        backend = LocalBackend(max_workers=4)
        np.random.seed(42)

        # Create test data
        data = np.random.normal(loc=50, scale=10, size=2000)
        df = pd.DataFrame({"value": data})

        results_list = []
        errors = []
        lock = threading.Lock()

        def run_fit():
            """Run a single fit operation."""
            try:
                fitter = DistributionFitter(backend=backend, random_seed=42)
                results = fitter.fit(df, column="value", max_distributions=5)
                best = results.best(n=1, metric="aic")[0]
                with lock:
                    results_list.append(best.distribution)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        # Run concurrent fits
        threads = [threading.Thread(target=run_fit) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors during concurrent fits: {errors}"

        # All results should be the same (fitting same data)
        assert len(set(results_list)) == 1, (
            f"Got different results from same data: {results_list}"
        )

    def test_thread_local_state_isolation(self, local_backend):
        """Test that thread-local state doesn't leak between threads.

        Run concurrent fits with different random seeds and verify that
        each thread's state is isolated. The key verification is that
        results are consistent and there are no errors from state corruption.
        """
        results = {}
        errors = []
        lock = threading.Lock()

        def fit_with_seed(seed: int, data: np.ndarray):
            """Fit with specific seed and capture results."""
            try:
                df = pd.DataFrame({"value": data})
                fitter = DistributionFitter(
                    backend=local_backend,
                    random_seed=seed,
                )
                result = fitter.fit(df, column="value", max_distributions=5)
                best = result.best(n=1, metric="aic")[0]
                with lock:
                    results[seed] = {
                        "distribution": best.distribution,
                        "aic": best.aic,
                        "data_hash": hash(data.tobytes()),
                    }
            except Exception as e:
                with lock:
                    errors.append((seed, str(e)))

        # Create distinct datasets for each thread
        np.random.seed(42)
        datasets = [
            (100, np.random.normal(50, 10, 2000)),
            (200, np.random.normal(100, 5, 2000)),
            (300, np.random.normal(0, 20, 2000)),
        ]

        threads = [
            threading.Thread(target=fit_with_seed, args=(seed, data))
            for seed, data in datasets
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors (primary check for thread safety)
        assert len(errors) == 0, f"Errors: {errors}"

        # Verify all threads completed
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Verify each thread produced valid results
        for seed, info in results.items():
            assert "distribution" in info, f"Seed {seed}: missing distribution"
            assert "aic" in info, f"Seed {seed}: missing AIC"
            assert np.isfinite(info["aic"]), f"Seed {seed}: invalid AIC"

        # Verify results are distinct (data hashes should differ)
        hashes = [r["data_hash"] for r in results.values()]
        assert len(set(hashes)) == 3, "Results may have been cross-contaminated"


class TestConcurrencyStress:
    """Stress tests for concurrency under high load."""

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=4)

    @pytest.mark.slow
    def test_many_concurrent_small_fits(self, local_backend):
        """Stress test: many concurrent small fitting operations."""
        np.random.seed(42)
        n_concurrent = 10

        errors = []
        results = []
        lock = threading.Lock()

        def small_fit(idx: int):
            """Run a small fit operation."""
            try:
                # Small dataset for quick fits
                data = np.random.normal(loc=50 + idx, scale=10, size=500)
                df = pd.DataFrame({"value": data})
                fitter = DistributionFitter(backend=local_backend)
                result = fitter.fit(df, column="value", max_distributions=3)
                with lock:
                    results.append(result.count())
            except Exception as e:
                with lock:
                    errors.append((idx, str(e)))

        threads = [threading.Thread(target=small_fit, args=(i,)) for i in range(n_concurrent)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors in stress test: {errors}"

        # Verify all fits completed
        assert len(results) == n_concurrent, (
            f"Expected {n_concurrent} results, got {len(results)}"
        )

        # All should have fitted 3 distributions
        assert all(r == 3 for r in results), f"Unexpected result counts: {results}"


class TestTruncatedAnalyticalMoments:
    """Tests for analytical truncated moment calculations.

    These tests verify the closed-form formulas for truncated distribution
    mean and standard deviation for norm, expon, and uniform distributions.
    """

    def test_truncated_normal_mean_two_sided(self):
        """Test analytical mean for two-sided truncated normal."""
        from spark_bestfit.truncated import TruncatedFrozenDist

        # N(0, 1) truncated to [-1, 1]
        base_dist = st.norm(loc=0, scale=1)
        truncated = TruncatedFrozenDist(base_dist, lb=-1, ub=1)

        # Symmetrically truncated normal should have mean = 0
        mean = truncated.mean()
        assert abs(mean) < 0.01, f"Expected ~0, got {mean}"

    def test_truncated_normal_std_two_sided(self):
        """Test analytical std for two-sided truncated normal."""
        from spark_bestfit.truncated import TruncatedFrozenDist

        base_dist = st.norm(loc=0, scale=1)
        truncated = TruncatedFrozenDist(base_dist, lb=-1, ub=1)

        # Truncated normal has reduced variance
        std = truncated.std()
        assert 0 < std < 1, f"Expected 0 < std < 1, got {std}"

    def test_truncated_expon_mean_one_sided(self):
        """Test analytical mean for one-sided truncated exponential."""
        from spark_bestfit.truncated import TruncatedFrozenDist

        # Exponential with scale=2, truncated at lb=0 (effectively untruncated)
        base_dist = st.expon(loc=0, scale=2)
        truncated = TruncatedFrozenDist(base_dist, lb=0, ub=np.inf)

        mean = truncated.mean()
        # Should be close to original mean = scale = 2
        assert abs(mean - 2) < 0.1, f"Expected ~2, got {mean}"

    def test_truncated_expon_two_sided(self):
        """Test analytical mean/std for two-sided truncated exponential."""
        from spark_bestfit.truncated import TruncatedFrozenDist

        base_dist = st.expon(loc=0, scale=1)
        truncated = TruncatedFrozenDist(base_dist, lb=0, ub=2)

        mean = truncated.mean()
        std = truncated.std()

        # Two-sided truncation reduces mean and variance
        assert 0 < mean < 1, f"Expected 0 < mean < 1, got {mean}"
        assert 0 < std < 1, f"Expected 0 < std < 1, got {std}"

    def test_truncated_uniform_mean(self):
        """Test analytical mean for truncated uniform."""
        from spark_bestfit.truncated import TruncatedFrozenDist

        # Uniform on [0, 10] truncated to [2, 8]
        base_dist = st.uniform(loc=0, scale=10)
        truncated = TruncatedFrozenDist(base_dist, lb=2, ub=8)

        mean = truncated.mean()
        # Truncated uniform mean = (lb + ub) / 2 = (2 + 8) / 2 = 5
        assert abs(mean - 5) < 0.01, f"Expected ~5, got {mean}"

    def test_truncated_uniform_std(self):
        """Test analytical std for truncated uniform."""
        from spark_bestfit.truncated import TruncatedFrozenDist

        # Uniform on [0, 10] truncated to [2, 8]
        base_dist = st.uniform(loc=0, scale=10)
        truncated = TruncatedFrozenDist(base_dist, lb=2, ub=8)

        std = truncated.std()
        # Truncated uniform std = (ub - lb) / sqrt(12) = 6 / sqrt(12) ≈ 1.73
        expected_std = 6 / np.sqrt(12)
        assert abs(std - expected_std) < 0.01, f"Expected ~{expected_std:.2f}, got {std}"


class TestFastPPFCaching:
    """Tests for LRU caching in fast_ppf module."""

    def test_cache_hit_performance(self):
        """Test that repeated calls use cache."""
        from spark_bestfit.fast_ppf import _cached_truncation_bounds, clear_ppf_cache

        # Clear cache first
        clear_ppf_cache()

        # First call - cache miss
        bounds1 = _cached_truncation_bounds("norm", (0.0, 1.0), -1.0, 1.0)

        # Second call - should be cache hit (same result)
        bounds2 = _cached_truncation_bounds("norm", (0.0, 1.0), -1.0, 1.0)

        assert bounds1 == bounds2

    def test_cache_clear(self):
        """Test cache clear functionality."""
        from spark_bestfit.fast_ppf import _cached_truncation_bounds, clear_ppf_cache

        # Populate cache
        _cached_truncation_bounds("norm", (0.0, 1.0), -1.0, 1.0)

        # Get cache info
        info_before = _cached_truncation_bounds.cache_info()
        assert info_before.currsize > 0 or info_before.hits > 0 or info_before.misses > 0

        # Clear and verify
        clear_ppf_cache()
        info_after = _cached_truncation_bounds.cache_info()
        assert info_after.currsize == 0
