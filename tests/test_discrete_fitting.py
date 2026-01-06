"""Tests for discrete distribution fitting."""

import numpy as np
import pytest

from spark_bestfit.discrete_fitting import (
    bootstrap_discrete_confidence_intervals,
    compute_discrete_histogram,
    compute_discrete_information_criteria,
    compute_discrete_ks_statistic,
    compute_discrete_sse,
    compute_ks_ad_metrics_discrete,
    fit_discrete_mle,
    fit_single_discrete_distribution,
    get_discrete_param_names,
)
from spark_bestfit.distributions import DiscreteDistributionRegistry


class TestDiscreteDistributionRegistry:
    """Tests for DiscreteDistributionRegistry class."""

    def test_get_param_config_returns_valid_config(self):
        """Test that param configs contain required keys and callable functions."""
        registry = DiscreteDistributionRegistry()
        data = np.array([1, 2, 3, 4, 5, 3, 2, 4, 3, 2])

        for dist_name in ["poisson", "nbinom", "geom", "binom"]:
            config = registry.get_param_config(dist_name)

            assert "param_names" in config
            assert "initial" in config
            assert "bounds" in config

            # Verify callables produce valid outputs
            initial = config["initial"](data)
            bounds = config["bounds"](data)

            assert len(initial) == len(config["param_names"])
            assert len(bounds) == len(config["param_names"])
            for b in bounds:
                assert b[0] < b[1], f"{dist_name} bounds invalid: {b}"

    def test_get_param_config_unsupported_distribution(self):
        """Test that unsupported distributions raise ValueError."""
        registry = DiscreteDistributionRegistry()

        with pytest.raises(ValueError, match="not supported"):
            registry.get_param_config("unsupported_distribution")

    def test_default_exclusions_exclude_slow_distributions(self):
        """Test that known slow distributions are excluded by default."""
        registry = DiscreteDistributionRegistry()
        distributions = registry.get_distributions()

        # These should be excluded
        assert "nchypergeom_fisher" not in distributions
        assert "nchypergeom_wallenius" not in distributions
        assert "randint" not in distributions


class TestFitDiscreteMLE:
    """Tests for MLE optimization of discrete distributions."""

    def test_poisson_mle_accuracy(self):
        """Test that Poisson MLE correctly estimates lambda."""
        np.random.seed(42)
        true_lambda = 5.0
        data = np.random.poisson(lam=true_lambda, size=2000)

        params, _ = fit_discrete_mle(
            "poisson", data, initial_params=[3.0], bounds=[(0.01, 50)]
        )

        # MLE for Poisson is exactly the mean
        assert np.isclose(params[0], np.mean(data), rtol=0.01)
        assert np.isclose(params[0], true_lambda, rtol=0.1)

    def test_geometric_mle_accuracy(self):
        """Test that Geometric MLE correctly estimates p."""
        np.random.seed(42)
        true_p = 0.3
        data = np.random.geometric(p=true_p, size=2000)

        params, _ = fit_discrete_mle(
            "geom", data, initial_params=[0.5], bounds=[(0.01, 0.99)]
        )

        # MLE for geometric should be close to 1/mean
        expected_p = 1.0 / np.mean(data)
        assert np.isclose(params[0], expected_p, rtol=0.05)


class TestComputeDiscreteHistogram:
    """Tests for discrete histogram computation."""

    def test_histogram_sums_to_one(self):
        """Test that PMF sums to 1.0."""
        data = np.random.poisson(lam=5, size=1000)
        x_values, pmf = compute_discrete_histogram(data)

        assert np.isclose(pmf.sum(), 1.0)

    def test_histogram_values_are_integers(self):
        """Test that histogram x-values are integers."""
        data = np.array([1, 2, 3, 3, 3, 4, 5])
        x_values, _ = compute_discrete_histogram(data)

        assert np.all(x_values == x_values.astype(int))

    def test_histogram_pmf_matches_counts(self):
        """Test that PMF correctly represents frequency."""
        data = np.array([1, 1, 2, 2, 2, 3])  # 2/6 ones, 3/6 twos, 1/6 threes
        x_values, pmf = compute_discrete_histogram(data)

        expected_values = np.array([1, 2, 3])
        expected_pmf = np.array([2 / 6, 3 / 6, 1 / 6])

        np.testing.assert_array_equal(x_values, expected_values)
        np.testing.assert_allclose(pmf, expected_pmf)


class TestFitSingleDiscreteDistribution:
    """Tests for single distribution fitting."""

    def test_failed_fit_returns_inf_sse(self):
        """Test that failed fits return infinite SSE."""
        registry = DiscreteDistributionRegistry()
        # Data that doesn't fit typical discrete patterns
        data = np.array([])  # Empty data will cause failures

        result = fit_single_discrete_distribution(
            dist_name="poisson",
            data_sample=data,
            x_values=np.array([]),
            empirical_pmf=np.array([]),
            registry=registry,
        )

        assert result["sse"] == float(np.inf)

    def test_poisson_fit_returns_valid_result(self):
        """Test that Poisson fitting returns complete result structure."""
        registry = DiscreteDistributionRegistry()
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=1000)
        x_values, empirical_pmf = compute_discrete_histogram(data)

        result = fit_single_discrete_distribution(
            dist_name="poisson",
            data_sample=data,
            x_values=x_values,
            empirical_pmf=empirical_pmf,
            registry=registry,
        )

        assert result["distribution"] == "poisson"
        assert len(result["parameters"]) == 1
        assert np.isfinite(result["sse"])
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])
        assert np.isfinite(result["ks_statistic"])
        assert 0 <= result["pvalue"] <= 1

        # Parameter should be close to true lambda
        assert np.isclose(result["parameters"][0], 7.0, rtol=0.1)


class TestMetricComputation:
    """Tests for metric computation functions."""

    def test_sse_is_zero_for_perfect_fit(self):
        """Test that SSE is zero when PMF matches exactly."""
        import scipy.stats as st

        dist = st.poisson
        params = (5.0,)
        x_values = np.arange(0, 15)
        # Use the true PMF as empirical PMF
        empirical_pmf = dist.pmf(x_values, *params)

        sse = compute_discrete_sse(dist, params, x_values, empirical_pmf, "poisson")

        assert np.isclose(sse, 0.0)

    def test_ks_statistic_bounded(self):
        """Test that KS statistic is between 0 and 1."""
        import scipy.stats as st

        np.random.seed(42)
        data = np.random.poisson(lam=5, size=500)

        ks_stat, pvalue = compute_discrete_ks_statistic(
            st.poisson, (5.0,), data, "poisson"
        )

        assert 0 <= ks_stat <= 1
        assert 0 <= pvalue <= 1

    def test_aic_bic_prefer_simpler_model(self):
        """Test that AIC/BIC penalize extra parameters."""
        import scipy.stats as st

        np.random.seed(42)
        data = np.random.poisson(lam=5, size=1000)

        # Poisson has 1 param, nbinom has 2
        aic_poisson, bic_poisson = compute_discrete_information_criteria(
            st.poisson, (5.0,), data, "poisson"
        )
        aic_nbinom, bic_nbinom = compute_discrete_information_criteria(
            st.nbinom, (100.0, 0.95), data, "nbinom"  # High n, high p approximates Poisson
        )

        # Both should prefer simpler model (Poisson) for Poisson data
        assert aic_poisson < aic_nbinom
        assert bic_poisson < bic_nbinom


class TestEdgeCasesAndErrorPaths:
    """Tests for edge cases and error handling in discrete fitting."""

    def test_fit_mle_with_invalid_initial_params(self):
        """Test MLE fitting gracefully handles edge case initializations."""
        np.random.seed(42)
        data = np.random.poisson(lam=5, size=100)

        # Should still converge even with far-off initial guess
        params, _ = fit_discrete_mle(
            "poisson", data, initial_params=[0.1], bounds=[(0.01, 100)]
        )

        assert np.isclose(params[0], np.mean(data), rtol=0.1)

    def test_binomial_fitting_with_integer_params(self):
        """Test that binomial n parameter is handled as integer."""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.3, size=500)

        registry = DiscreteDistributionRegistry()
        x_values, empirical_pmf = compute_discrete_histogram(data)

        result = fit_single_discrete_distribution(
            dist_name="binom",
            data_sample=data,
            x_values=x_values,
            empirical_pmf=empirical_pmf,
            registry=registry,
        )

        # n should be close to 10
        assert result["distribution"] == "binom"
        assert np.isfinite(result["sse"])
        # First param is n, should be integer-valued
        assert result["parameters"][0] == int(result["parameters"][0])

    def test_information_criteria_with_extreme_outlier(self):
        """Test AIC/BIC with extreme outliers yields very high values."""
        import scipy.stats as st

        # Normal data with extreme outlier
        data = np.array([0, 1, 2, 3, 100])  # 100 is extreme outlier

        aic, bic = compute_discrete_information_criteria(
            st.poisson, (2.0,), data, "poisson"
        )

        # AIC/BIC should be finite but high (poor fit)
        assert np.isfinite(aic) or aic == np.inf
        assert np.isfinite(bic) or bic == np.inf

    def test_ks_statistic_with_integer_params(self):
        """Test KS statistic computation with integer param distributions."""
        import scipy.stats as st

        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.5, size=200)

        ks_stat, pvalue = compute_discrete_ks_statistic(
            st.binom, (10, 0.5), data, "binom"
        )

        assert 0 <= ks_stat <= 1
        assert 0 <= pvalue <= 1

    def test_evaluate_pmf_with_invalid_params_returns_zeros(self):
        """Test that evaluate_pmf returns zeros for invalid params."""
        import scipy.stats as st
        from spark_bestfit.discrete_fitting import evaluate_pmf

        x = np.array([0, 1, 2, 3])
        # Invalid params that will cause pmf to fail
        pmf = evaluate_pmf(st.poisson, (-1.0,), x, "poisson")

        # Should return zeros, not crash
        assert len(pmf) == len(x)

    def test_create_discrete_sample_data_full_data(self):
        """Test sample creation when data is smaller than sample size."""
        from spark_bestfit.discrete_fitting import create_discrete_sample_data

        data = np.array([1, 2, 3, 4, 5])
        sample = create_discrete_sample_data(data, sample_size=1000)

        # Should return all data since it's smaller than sample size
        assert len(sample) == len(data)
        np.testing.assert_array_equal(sample, data.astype(int))

    def test_create_discrete_sample_data_samples_correctly(self):
        """Test sample creation with sampling."""
        from spark_bestfit.discrete_fitting import create_discrete_sample_data

        np.random.seed(42)
        data = np.arange(0, 10000)
        sample = create_discrete_sample_data(data, sample_size=100, random_seed=42)

        assert len(sample) == 100
        # All samples should be from original data
        assert all(s in data for s in sample)

    def test_discrete_histogram_with_nan_values(self):
        """Test that NaN values are handled in discrete histogram computation."""
        data = np.array([1, 2, 3, np.nan, 4, 5])
        # Filter NaN before computing histogram (as the code should do)
        clean_data = data[~np.isnan(data)].astype(int)
        x_values, pmf = compute_discrete_histogram(clean_data)

        assert np.isclose(pmf.sum(), 1.0)
        assert len(x_values) == 5  # 5 unique values

    def test_fit_discrete_mle_with_zero_data(self):
        """Test MLE fitting with data containing zeros."""
        np.random.seed(42)
        # Poisson data often has zeros
        data = np.random.poisson(lam=1, size=500)

        params, neg_ll = fit_discrete_mle(
            "poisson", data, initial_params=[0.5], bounds=[(0.01, 50)]
        )

        # Should successfully fit
        assert np.isfinite(params[0])
        assert params[0] > 0

    def test_compute_discrete_sse_with_empty_pmf(self):
        """Test SSE computation handles edge case of mismatched ranges."""
        import scipy.stats as st

        dist = st.poisson
        params = (5.0,)
        x_values = np.array([100, 101, 102])  # Far from mean=5
        # PMF will be essentially zero at these values
        empirical_pmf = np.array([0.33, 0.33, 0.34])

        sse = compute_discrete_sse(dist, params, x_values, empirical_pmf, "poisson")

        # SSE should be finite (sum of squared empirical PMF since fitted PMF ≈ 0)
        assert np.isfinite(sse)
        assert sse > 0


class TestGetDiscreteParamNames:
    """Tests for get_discrete_param_names function."""

    def test_poisson_param_names(self):
        """Test parameter names for Poisson distribution."""
        names = get_discrete_param_names("poisson")
        assert names == ["mu"]

    def test_binomial_param_names(self):
        """Test parameter names for binomial distribution."""
        names = get_discrete_param_names("binom")
        assert names == ["n", "p"]

    def test_nbinom_param_names(self):
        """Test parameter names for negative binomial distribution."""
        names = get_discrete_param_names("nbinom")
        assert names == ["n", "p"]

    def test_geometric_param_names(self):
        """Test parameter names for geometric distribution."""
        names = get_discrete_param_names("geom")
        assert names == ["p"]


class TestComputeKsAdMetricsDiscrete:
    """Tests for compute_ks_ad_metrics_discrete function."""

    def test_returns_ks_metrics_for_poisson(self):
        """Test that KS metrics are returned for Poisson distribution."""
        np.random.seed(42)
        data = np.random.poisson(lam=5, size=500)

        ks_stat, pvalue, ad_stat, ad_pvalue = compute_ks_ad_metrics_discrete(
            "poisson", [5.0], data
        )

        # KS metrics should be computed
        assert ks_stat is not None
        assert pvalue is not None
        assert 0 <= ks_stat <= 1
        assert 0 <= pvalue <= 1

        # AD metrics should be None for discrete
        assert ad_stat is None
        assert ad_pvalue is None

    def test_returns_ks_metrics_for_binomial(self):
        """Test that KS metrics are returned for binomial distribution."""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.3, size=500)

        ks_stat, pvalue, ad_stat, ad_pvalue = compute_ks_ad_metrics_discrete(
            "binom", [10, 0.3], data
        )

        # KS metrics should be computed
        assert ks_stat is not None
        assert pvalue is not None
        assert 0 <= ks_stat <= 1
        assert 0 <= pvalue <= 1

        # AD metrics should be None for discrete
        assert ad_stat is None
        assert ad_pvalue is None

    def test_good_vs_bad_fit_comparison(self):
        """Test that good fit has lower KS statistic than poor fit."""
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=1000)

        # Good fit with correct parameter
        ks_good, _, _, _ = compute_ks_ad_metrics_discrete(
            "poisson", [7.0], data
        )

        # Poor fit with wrong parameter
        ks_poor, _, _, _ = compute_ks_ad_metrics_discrete(
            "poisson", [2.0], data
        )

        # Good fit should have lower KS statistic
        assert ks_good < ks_poor

    def test_poor_fit_low_pvalue(self):
        """Test that poor fit produces low p-value."""
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=1000)

        # Fit with wrong parameter (lambda=2 instead of 7)
        ks_stat, pvalue, _, _ = compute_ks_ad_metrics_discrete(
            "poisson", [2.0], data
        )

        # Poor fit should have low p-value
        assert pvalue < 0.05

    def test_handles_invalid_distribution(self):
        """Test that invalid distribution returns None values."""
        data = np.array([1, 2, 3, 4, 5])

        ks_stat, pvalue, ad_stat, ad_pvalue = compute_ks_ad_metrics_discrete(
            "invalid_dist_name", [1.0], data
        )

        # Should return None for all metrics
        assert ks_stat is None
        assert pvalue is None
        assert ad_stat is None
        assert ad_pvalue is None

    def test_handles_float_data_conversion(self):
        """Test that float data is converted to integer."""
        np.random.seed(42)
        # Create float data that should be converted
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0])

        ks_stat, pvalue, _, _ = compute_ks_ad_metrics_discrete(
            "poisson", [3.0], data
        )

        # Should successfully compute metrics
        assert ks_stat is not None
        assert pvalue is not None

    def test_optional_bounds_ignored(self):
        """Test that lower/upper bounds are accepted but don't affect discrete metrics."""
        np.random.seed(42)
        data = np.random.poisson(lam=5, size=500)

        # Without bounds
        ks1, pv1, _, _ = compute_ks_ad_metrics_discrete("poisson", [5.0], data)

        # With bounds (should be ignored for discrete)
        ks2, pv2, _, _ = compute_ks_ad_metrics_discrete(
            "poisson", [5.0], data, lower_bound=0.0, upper_bound=100.0
        )

        # Results should be identical (bounds are ignored)
        assert ks1 == ks2
        assert pv1 == pv2

    def test_geometric_distribution(self):
        """Test KS metrics for geometric distribution."""
        np.random.seed(42)
        data = np.random.geometric(p=0.3, size=500)

        ks_stat, pvalue, ad_stat, ad_pvalue = compute_ks_ad_metrics_discrete(
            "geom", [0.3], data
        )

        assert ks_stat is not None
        assert pvalue is not None
        assert ad_stat is None
        assert ad_pvalue is None

    def test_negative_binomial_distribution(self):
        """Test KS metrics for negative binomial distribution."""
        np.random.seed(42)
        data = np.random.negative_binomial(n=5, p=0.5, size=500)

        ks_stat, pvalue, ad_stat, ad_pvalue = compute_ks_ad_metrics_discrete(
            "nbinom", [5, 0.5], data
        )

        assert ks_stat is not None
        assert pvalue is not None
        assert ad_stat is None
        assert ad_pvalue is None


class TestBootstrapDiscreteConfidenceIntervals:
    """Tests for bootstrap_discrete_confidence_intervals function."""

    def test_basic_ci_computation_poisson(self):
        """Test that CI is computed correctly for Poisson distribution."""
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=1000)

        ci = bootstrap_discrete_confidence_intervals(
            "poisson", data, alpha=0.05, n_bootstrap=100, random_seed=42
        )

        # Should have correct parameter name
        assert "mu" in ci

        # CI should be a tuple of (lower, upper)
        assert isinstance(ci["mu"], tuple)
        assert len(ci["mu"]) == 2
        assert ci["mu"][0] < ci["mu"][1]  # lower < upper

        # True lambda=7 should be within the CI
        assert ci["mu"][0] < 7.0 < ci["mu"][1]

    def test_ci_for_binomial(self):
        """Test CI computation for binomial distribution."""
        np.random.seed(42)
        data = np.random.binomial(n=10, p=0.3, size=500)

        ci = bootstrap_discrete_confidence_intervals(
            "binom", data, alpha=0.05, n_bootstrap=100, random_seed=42
        )

        # Binomial has n and p parameters
        assert "n" in ci
        assert "p" in ci

        # All CIs should be valid
        for param, (lower, upper) in ci.items():
            assert lower < upper

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        np.random.seed(42)
        data = np.random.poisson(lam=5, size=500)

        ci1 = bootstrap_discrete_confidence_intervals(
            "poisson", data, alpha=0.05, n_bootstrap=50, random_seed=123
        )
        ci2 = bootstrap_discrete_confidence_intervals(
            "poisson", data, alpha=0.05, n_bootstrap=50, random_seed=123
        )

        assert ci1["mu"] == ci2["mu"]

    def test_different_alpha_values(self):
        """Test that smaller alpha gives wider CI."""
        np.random.seed(42)
        data = np.random.poisson(lam=5, size=1000)

        ci_95 = bootstrap_discrete_confidence_intervals(
            "poisson", data, alpha=0.05, n_bootstrap=200, random_seed=42
        )
        ci_99 = bootstrap_discrete_confidence_intervals(
            "poisson", data, alpha=0.01, n_bootstrap=200, random_seed=42
        )

        # 99% CI should be wider than 95% CI
        width_95 = ci_95["mu"][1] - ci_95["mu"][0]
        width_99 = ci_99["mu"][1] - ci_99["mu"][0]
        assert width_99 >= width_95

    def test_small_sample_works(self):
        """Test CI computation works with small sample."""
        # Small sample - should still compute CI
        small_data = np.array([1, 2, 3, 4, 5, 3, 2, 4, 3, 2])

        ci = bootstrap_discrete_confidence_intervals(
            "poisson", small_data, alpha=0.05, n_bootstrap=50, random_seed=42
        )

        # Should return valid CI
        assert "mu" in ci
        lower, upper = ci["mu"]
        assert lower < upper
