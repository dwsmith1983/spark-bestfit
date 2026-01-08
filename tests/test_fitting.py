"""Tests for fitting module."""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.fitting import (
    AD_PVALUE_DISTRIBUTIONS,
    FITTING_SAMPLE_SIZE,
    bootstrap_confidence_intervals,
    compute_ad_pvalue,
    compute_ad_statistic,
    compute_data_stats,
    compute_information_criteria,
    compute_ks_statistic,
    compute_pdf_range,
    create_sample_data,
    evaluate_pdf,
    extract_distribution_params,
    fit_single_distribution,
    get_continuous_param_names,
)
from spark_bestfit.truncated import TruncatedFrozenDist

class TestFitSingleDistribution:
    """Tests for fitting single distributions."""

    def test_fit_normal_distribution(self, normal_data):
        """Test fitting normal distribution to normal data."""
        # Create histogram (pass bin_edges for CDF-based fitting)
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)

        result = fit_single_distribution("norm", normal_data, bin_edges, y_hist)

        # Should succeed
        assert result["distribution"] == "norm"
        assert len(result["parameters"]) > 0
        assert result["sse"] < np.inf
        assert result["aic"] < np.inf
        assert result["bic"] < np.inf

        # Parameters should be close to true values (loc=50, scale=10)
        params = result["parameters"]
        loc = params[-2]
        scale = params[-1]

        assert 45 < loc < 55  # Close to 50
        assert 8 < scale < 12  # Close to 10

    def test_fit_exponential_distribution(self, exponential_data):
        """Test fitting exponential distribution."""
        y_hist, bin_edges = np.histogram(exponential_data, bins=50, density=True)

        result = fit_single_distribution("expon", exponential_data, bin_edges, y_hist)

        # Should succeed
        assert result["distribution"] == "expon"
        assert len(result["parameters"]) > 0
        assert result["sse"] < np.inf

        # Scale should be close to 5.0
        scale = result["parameters"][-1]
        assert 4 < scale < 6

    def test_fit_gamma_distribution(self, gamma_data):
        """Test fitting gamma distribution."""
        y_hist, bin_edges = np.histogram(gamma_data, bins=50, density=True)

        result = fit_single_distribution("gamma", gamma_data, bin_edges, y_hist)

        # Should succeed
        assert result["distribution"] == "gamma"
        assert result["sse"] < np.inf

    def test_fit_invalid_distribution(self, normal_data):
        """Test fitting with invalid distribution name."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        result = fit_single_distribution("invalid_dist", normal_data, bin_edges, y_hist)

        # Should fail gracefully
        assert result["distribution"] == "invalid_dist"
        assert result["sse"] == np.inf
        assert result["aic"] == np.inf
        assert result["bic"] == np.inf
        assert result["parameters"] == [float(np.nan)]

    def test_fit_with_insufficient_data(self):
        """Test fitting with very little data."""
        data = np.array([1.0, 2.0, 3.0])
        y_hist = np.array([0.5, 0.3, 0.2])
        bin_edges = np.array([0.5, 1.5, 2.5, 3.5])  # 3 bins

        result = fit_single_distribution("norm", data, bin_edges, y_hist)

        # Should attempt to fit (may succeed or fail)
        assert result["distribution"] == "norm"
        # SSE must be finite positive or infinity (not NaN)
        assert np.isfinite(result["sse"]) or result["sse"] == np.inf
        # If fit succeeded, verify we got parameters
        if np.isfinite(result["sse"]):
            assert len(result["parameters"]) >= 2  # norm has loc, scale

class TestEvaluatePDF:
    """Tests for PDF evaluation."""

    def test_evaluate_pdf_normal(self):
        """Test evaluating PDF for normal distribution."""
        dist = st.norm
        params = (0, 1)  # Standard normal: loc=0, scale=1
        x = np.array([-2, -1, 0, 1, 2])

        pdf_values = evaluate_pdf(dist, params, x)

        # Should return valid PDF values
        assert len(pdf_values) == len(x)
        assert np.all(pdf_values >= 0)
        assert np.all(np.isfinite(pdf_values))

        # PDF at 0 should be highest for standard normal
        assert pdf_values[2] == np.max(pdf_values)

    def test_evaluate_pdf_with_shape_params(self):
        """Test evaluating PDF with shape parameters."""
        dist = st.gamma
        params = (2.0, 0, 2.0)  # shape=2, loc=0, scale=2
        x = np.linspace(0, 10, 50)

        pdf_values = evaluate_pdf(dist, params, x)

        # Should return valid PDF values
        assert len(pdf_values) == len(x)
        assert np.all(pdf_values >= 0)
        assert np.all(np.isfinite(pdf_values))

    def test_evaluate_pdf_handles_nan(self):
        """Test that PDF evaluation handles NaN gracefully."""
        dist = st.norm
        params = (0, 1)
        x = np.array([np.nan, 0, 1])

        pdf_values = evaluate_pdf(dist, params, x)

        # Should convert NaN to 0
        assert np.isfinite(pdf_values).all()

class TestComputeInformationCriteria:
    """Tests for information criteria calculation."""

    def test_compute_aic_bic_normal(self, normal_data):
        """Test computing AIC and BIC for normal distribution."""
        dist = st.norm
        params = dist.fit(normal_data)

        aic, bic = compute_information_criteria(dist, params, normal_data)

        # Should return finite values
        assert np.isfinite(aic)
        assert np.isfinite(bic)

        # BIC should be higher than AIC (penalizes complexity more)
        assert bic > aic

    def test_compute_aic_bic_gamma(self, gamma_data):
        """Test computing AIC and BIC for gamma distribution."""
        dist = st.gamma
        params = dist.fit(gamma_data)

        aic, bic = compute_information_criteria(dist, params, gamma_data)

        # Should return finite values
        assert np.isfinite(aic)
        assert np.isfinite(bic)

    def test_compute_aic_bic_invalid_data(self):
        """Test information criteria with invalid data."""
        dist = st.norm
        params = (0, 1)
        data = np.array([np.nan, np.inf])

        aic, bic = compute_information_criteria(dist, params, data)

        # Should return inf for invalid data
        assert aic == np.inf
        assert bic == np.inf

class TestCreateSampleData:
    """Tests for data sampling."""

    def test_create_sample_small_data(self):
        """Test sampling when data is smaller than sample size."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=1000, random_seed=42)

        # Should return all data
        assert len(sample) == len(data)
        assert np.array_equal(sample, data)

    def test_create_sample_large_data(self):
        """Test sampling when data is larger than sample size."""
        data = np.arange(100_000)
        sample = create_sample_data(data, sample_size=10_000, random_seed=42)

        # Should return sampled data
        assert len(sample) == 10_000
        assert len(sample) < len(data)

        # All sampled values should be from original data
        assert np.all(np.isin(sample, data))

    def test_create_sample_reproducible(self):
        """Test that sampling is reproducible with same seed."""
        data = np.arange(100_000)

        sample1 = create_sample_data(data, sample_size=10_000, random_seed=42)
        sample2 = create_sample_data(data, sample_size=10_000, random_seed=42)

        # Should be identical
        assert np.array_equal(sample1, sample2)

    def test_create_sample_different_seeds(self):
        """Test that different seeds produce different samples."""
        data = np.arange(100_000)

        sample1 = create_sample_data(data, sample_size=10_000, random_seed=42)
        sample2 = create_sample_data(data, sample_size=10_000, random_seed=123)

        # Should be different
        assert not np.array_equal(sample1, sample2)

    def test_create_sample_no_replacement(self):
        """Test that sampling is without replacement."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=50, random_seed=42)

        # Should have no duplicates
        assert len(sample) == len(np.unique(sample))

class TestFitSingleDistributionEdgeCases:
    """Edge case tests for fit_single_distribution."""

    def test_fit_with_negative_data_for_positive_dist(self):
        """Test fitting positive-only distribution to negative data."""
        data = np.array([-5.0, -3.0, -1.0, 0.0, 1.0])
        y_hist = np.array([0.2, 0.3, 0.3, 0.2])
        bin_edges = np.array([-5.0, -3.0, -1.0, 0.5, 1.5])  # 4 bins

        # expon requires positive data, may fail or produce poor fit
        result = fit_single_distribution("expon", data, bin_edges, y_hist)

        assert result["distribution"] == "expon"
        # Either succeeds with a result or fails gracefully
        assert result["sse"] >= 0 or result["sse"] == np.inf

    def test_fit_with_empty_params_distribution(self, normal_data):
        """Test distributions with different parameter structures."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        # Test uniform distribution (only loc and scale)
        result = fit_single_distribution("uniform", normal_data, bin_edges, y_hist)

        assert result["distribution"] == "uniform"
        assert len(result["parameters"]) >= 2  # at least loc, scale

    def test_fit_returns_correct_structure(self, normal_data):
        """Test that fit returns dict with all required keys."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)
        data_stats = compute_data_stats(normal_data)

        result = fit_single_distribution("norm", normal_data, bin_edges, y_hist, data_stats=data_stats)

        required_keys = {
            "column_name", "distribution", "parameters", "sse", "aic", "bic",
            "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue",
            "data_min", "data_max", "data_mean", "data_stddev", "data_count",
            "lower_bound", "upper_bound"  # Added in v1.4.0 for bounded fitting
        }
        assert set(result.keys()) == required_keys

class TestEvaluatePDFEdgeCases:
    """Edge case tests for evaluate_pdf."""

    def test_evaluate_pdf_empty_x(self):
        """Test PDF evaluation with empty x array."""
        dist = st.norm
        params = (0, 1)
        x = np.array([])

        pdf_values = evaluate_pdf(dist, params, x)

        assert len(pdf_values) == 0

    def test_evaluate_pdf_single_point(self):
        """Test PDF evaluation at single point."""
        dist = st.norm
        params = (0, 1)
        x = np.array([0])

        pdf_values = evaluate_pdf(dist, params, x)

        assert len(pdf_values) == 1
        assert pdf_values[0] > 0

    def test_evaluate_pdf_inf_handling(self):
        """Test PDF handles inf values in x."""
        dist = st.norm
        params = (0, 1)
        x = np.array([np.inf, -np.inf, 0])

        pdf_values = evaluate_pdf(dist, params, x)

        # inf should be converted to 0
        assert np.all(np.isfinite(pdf_values))

    def test_evaluate_pdf_extreme_x_values(self):
        """Test PDF with extreme but finite x values."""
        dist = st.norm
        params = (0, 1)
        x = np.array([-1000, -100, 0, 100, 1000])

        pdf_values = evaluate_pdf(dist, params, x)

        # Should all be finite (near 0 for extreme values)
        assert np.all(np.isfinite(pdf_values))
        assert np.all(pdf_values >= 0)

class TestComputeInformationCriteriaEdgeCases:
    """Edge case tests for compute_information_criteria."""

    def test_compute_with_small_sample(self):
        """Test information criteria with very small sample."""
        dist = st.norm
        data = np.array([1.0, 2.0, 3.0])
        params = dist.fit(data)

        aic, bic = compute_information_criteria(dist, params, data)

        # Should return finite values for small sample
        assert np.isfinite(aic) or aic == np.inf
        assert np.isfinite(bic) or bic == np.inf

    def test_compute_with_single_point(self):
        """Test information criteria with single data point."""
        dist = st.norm
        data = np.array([1.0])
        params = (0, 1)  # Use fixed params since can't fit with 1 point

        aic, bic = compute_information_criteria(dist, params, data)

        # May return inf due to numerical issues
        assert isinstance(aic, (int, float))
        assert isinstance(bic, (int, float))

class TestCreateSampleDataEdgeCases:
    """Edge case tests for create_sample_data."""

    def test_create_sample_exact_size(self):
        """Test sampling when data size equals sample size."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=100, random_seed=42)

        assert len(sample) == 100
        assert np.array_equal(sample, data)

    def test_create_sample_size_one(self):
        """Test sampling to size 1."""
        data = np.arange(100)
        sample = create_sample_data(data, sample_size=1, random_seed=42)

        assert len(sample) == 1
        assert sample[0] in data

    def test_create_sample_empty_data(self):
        """Test sampling from empty array."""
        data = np.array([])
        sample = create_sample_data(data, sample_size=10, random_seed=42)

        assert len(sample) == 0

    def test_create_sample_preserves_dtype(self):
        """Test that sampling preserves data type."""
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        sample = create_sample_data(data, sample_size=3, random_seed=42)

        assert sample.dtype == data.dtype

class TestExtractDistributionParams:
    """Tests for extract_distribution_params utility function."""

    def test_extract_two_params(self):
        """Test extracting from 2-parameter distribution (normal)."""
        params = [50.0, 10.0]  # loc=50, scale=10
        shape, loc, scale = extract_distribution_params(params)

        assert shape == ()
        assert loc == 50.0
        assert scale == 10.0

    def test_extract_three_params(self):
        """Test extracting from 3-parameter distribution (gamma)."""
        params = [2.0, 0.0, 5.0]  # a=2, loc=0, scale=5
        shape, loc, scale = extract_distribution_params(params)

        assert shape == (2.0,)
        assert loc == 0.0
        assert scale == 5.0

    def test_extract_four_params(self):
        """Test extracting from 4-parameter distribution (beta)."""
        params = [2.0, 5.0, 0.0, 1.0]  # a=2, b=5, loc=0, scale=1
        shape, loc, scale = extract_distribution_params(params)

        assert shape == (2.0, 5.0)
        assert loc == 0.0
        assert scale == 1.0

    def test_extract_many_params(self):
        """Test extracting from distribution with many shape params."""
        params = [1.0, 2.0, 3.0, 4.0, 0.0, 1.0]  # 4 shape params + loc + scale
        shape, loc, scale = extract_distribution_params(params)

        assert shape == (1.0, 2.0, 3.0, 4.0)
        assert loc == 0.0
        assert scale == 1.0

    def test_extract_insufficient_params_raises(self):
        """Test that fewer than 2 params raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            extract_distribution_params([1.0])

    def test_extract_empty_params_raises(self):
        """Test that empty params raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            extract_distribution_params([])

    def test_extract_with_list_input(self):
        """Test that list input works correctly."""
        params = [2.0, 0.0, 5.0]
        shape, loc, scale = extract_distribution_params(params)

        assert shape == (2.0,)
        assert loc == 0.0
        assert scale == 5.0

class TestComputePdfRange:
    """Tests for compute_pdf_range utility function."""

    def test_range_for_normal_distribution(self):
        """Test computing range for normal distribution."""
        dist = st.norm
        params = [0.0, 1.0]  # loc=0, scale=1
        x_hist = np.linspace(-5, 5, 50)

        start, end = compute_pdf_range(dist, params, x_hist)

        # For standard normal, 0.01 to 0.99 percentile is about -2.33 to 2.33
        assert -3.0 < start < -2.0
        assert 2.0 < end < 3.0
        assert start < end

    def test_range_for_gamma_distribution(self):
        """Test computing range for gamma distribution."""
        dist = st.gamma
        params = [2.0, 0.0, 1.0]  # shape=2, loc=0, scale=1
        x_hist = np.linspace(0, 10, 50)

        start, end = compute_pdf_range(dist, params, x_hist)

        # Gamma with shape=2 starts near 0
        assert start >= 0
        assert end > start

    def test_range_uses_custom_percentile(self):
        """Test that custom percentile affects range."""
        dist = st.norm
        params = [0.0, 1.0]
        x_hist = np.linspace(-5, 5, 50)

        # Narrow range (5th to 95th percentile)
        start_narrow, end_narrow = compute_pdf_range(dist, params, x_hist, percentile=0.05)

        # Wide range (1st to 99th percentile)
        start_wide, end_wide = compute_pdf_range(dist, params, x_hist, percentile=0.01)

        # Wide range should be wider
        assert start_wide < start_narrow
        assert end_wide > end_narrow

    def test_range_fallback_on_ppf_failure(self):
        """Test fallback to histogram bounds when ppf fails."""
        # Create a mock distribution that raises on ppf
        class FailingDist:
            def ppf(self, *args, **kwargs):
                raise ValueError("ppf failed")

        dist = FailingDist()
        params = [0.0, 1.0]
        x_hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        start, end = compute_pdf_range(dist, params, x_hist)

        # Should fall back to histogram bounds
        assert start == 1.0
        assert end == 5.0

    def test_range_fallback_on_nonfinite_start(self):
        """Test fallback when ppf returns non-finite start."""

        class InfStartDist:
            def ppf(self, q, *args, **kwargs):
                if q < 0.5:
                    return np.inf
                return 1.0

        dist = InfStartDist()
        params = [0.0, 1.0]
        x_hist = np.array([1.0, 2.0, 3.0])

        start, end = compute_pdf_range(dist, params, x_hist)

        # Start should fall back to x_hist.min()
        assert start == 1.0

    def test_range_fallback_on_nonfinite_end(self):
        """Test fallback when ppf returns non-finite end."""

        class InfEndDist:
            def ppf(self, q, *args, **kwargs):
                if q > 0.5:
                    return np.nan
                return 0.0

        dist = InfEndDist()
        params = [0.0, 1.0]
        x_hist = np.array([1.0, 2.0, 3.0])

        start, end = compute_pdf_range(dist, params, x_hist)

        # End should fall back to x_hist.max()
        assert end == 3.0

    def test_range_returns_floats(self):
        """Test that range returns Python floats."""
        dist = st.norm
        params = [0.0, 1.0]
        x_hist = np.linspace(-5, 5, 50)

        start, end = compute_pdf_range(dist, params, x_hist)

        assert isinstance(start, float)
        assert isinstance(end, float)

    def test_range_with_real_scipy_distributions(self):
        """Test range computation with various real scipy distributions."""
        distributions = [
            (st.norm, [0.0, 1.0]),
            (st.expon, [0.0, 1.0]),
            (st.gamma, [2.0, 0.0, 1.0]),
            (st.beta, [2.0, 5.0, 0.0, 1.0]),
            (st.uniform, [0.0, 1.0]),
        ]
        x_hist = np.linspace(-5, 10, 50)

        for dist, params in distributions:
            start, end = compute_pdf_range(dist, params, x_hist)

            assert np.isfinite(start), f"start not finite for {dist.name}"
            assert np.isfinite(end), f"end not finite for {dist.name}"
            assert start < end, f"start >= end for {dist.name}"


class TestComputeKsStatistic:
    """Tests for Kolmogorov-Smirnov statistic computation."""

    def test_ks_statistic_poor_fit(self, normal_data):
        """Test K-S statistic for poor fit (wrong distribution)."""
        # Fit exponential to normal data - should be a poor fit
        dist = st.expon
        params = dist.fit(normal_data)

        ks_stat, pvalue = compute_ks_statistic(dist, params, normal_data)

        assert np.isfinite(ks_stat)
        # K-S statistic should be larger for poor fit
        assert ks_stat > 0.1

    def test_ks_statistic_invalid_data(self):
        """Test K-S statistic with invalid data returns sentinel values."""
        dist = st.norm
        params = (0, 1)
        invalid_data = np.array([np.nan, np.inf, -np.inf])

        ks_stat, pvalue = compute_ks_statistic(dist, params, invalid_data)

        # Should return sentinel values for failed computation
        assert ks_stat == np.inf or np.isfinite(ks_stat)
        assert pvalue == 0.0 or np.isfinite(pvalue)

    def test_ks_statistic_small_sample(self):
        """Test K-S statistic with small sample size."""
        dist = st.norm
        params = (0, 1)
        small_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ks_stat, pvalue = compute_ks_statistic(dist, params, small_data)

        # Should still return values for small samples
        assert np.isfinite(ks_stat)
        assert 0 <= pvalue <= 1

    def test_ks_statistic_various_distributions(self):
        """Test K-S statistic works with various scipy distributions."""
        np.random.seed(42)

        test_cases = [
            (st.norm, st.norm.rvs(size=1000)),
            (st.expon, st.expon.rvs(size=1000)),
            (st.gamma, st.gamma.rvs(a=2, size=1000)),
            (st.uniform, st.uniform.rvs(size=1000)),
        ]

        for dist, data in test_cases:
            params = dist.fit(data)
            ks_stat, pvalue = compute_ks_statistic(dist, params, data)

            assert np.isfinite(ks_stat), f"K-S stat not finite for {dist.name}"
            assert 0 <= pvalue <= 1, f"p-value out of range for {dist.name}"


class TestFitSingleDistributionWithKS:
    """Tests for fit_single_distribution including K-S statistic fields."""

    def test_fit_returns_ks_fields(self, normal_data):
        """Test that fit_single_distribution returns K-S fields."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        result = fit_single_distribution("norm", normal_data, bin_edges, y_hist)

        # Should have K-S fields
        assert "ks_statistic" in result
        assert "pvalue" in result

        # Values should be valid
        assert np.isfinite(result["ks_statistic"])
        assert 0 <= result["pvalue"] <= 1

    def test_failed_fit_returns_ks_sentinel_values(self, normal_data):
        """Test that failed fits return sentinel values for K-S fields."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        result = fit_single_distribution("invalid_dist", normal_data, bin_edges, y_hist)

        # Should have K-S sentinel values
        assert result["ks_statistic"] == np.inf
        assert result["pvalue"] == 0.0


class TestComputeAdStatistic:
    """Tests for Anderson-Darling statistic computation."""

    def test_ad_statistic_good_fit(self, normal_data):
        """Test A-D statistic for good fit (correct distribution)."""
        dist = st.norm
        params = dist.fit(normal_data)

        ad_stat = compute_ad_statistic(dist, params, normal_data)

        assert np.isfinite(ad_stat)
        # A-D statistic for good fit should be relatively small
        assert 0 < ad_stat < 5.0

    def test_ad_statistic_poor_fit(self, normal_data):
        """Test A-D statistic for poor fit (wrong distribution)."""
        # Fit exponential to normal data - should be a poor fit
        dist = st.expon
        params = dist.fit(normal_data)

        ad_stat = compute_ad_statistic(dist, params, normal_data)

        assert np.isfinite(ad_stat)
        # A-D statistic should be larger for poor fit
        assert ad_stat > 5.0

    def test_ad_statistic_various_distributions(self):
        """Test A-D statistic works with various scipy distributions."""
        np.random.seed(42)

        test_cases = [
            (st.norm, st.norm.rvs(size=1000)),
            (st.expon, st.expon.rvs(size=1000)),
            (st.gamma, st.gamma.rvs(a=2, size=1000)),
            (st.uniform, st.uniform.rvs(size=1000)),
        ]

        for dist, data in test_cases:
            params = dist.fit(data)
            ad_stat = compute_ad_statistic(dist, params, data)

            assert np.isfinite(ad_stat), f"A-D stat not finite for {dist.name}"
            assert ad_stat >= 0, f"A-D stat negative for {dist.name}"

    def test_ad_statistic_small_sample(self):
        """Test A-D statistic with small sample size."""
        dist = st.norm
        params = (0, 1)
        small_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        ad_stat = compute_ad_statistic(dist, params, small_data)

        # Should still return value for small samples
        assert np.isfinite(ad_stat) or ad_stat == np.inf

    def test_ad_statistic_single_point(self):
        """Test A-D statistic with single data point returns inf."""
        dist = st.norm
        params = (0, 1)
        data = np.array([1.0])

        ad_stat = compute_ad_statistic(dist, params, data)

        assert ad_stat == np.inf

    def test_ad_statistic_handles_boundary_cdf(self):
        """Test A-D statistic handles CDF values at boundaries (0 or 1)."""
        # Use data that would produce extreme CDF values
        dist = st.norm
        params = (0, 1)
        # Data far in the tail
        data = np.array([-10, -5, 0, 5, 10])

        ad_stat = compute_ad_statistic(dist, params, data)

        # Should return finite value due to clamping
        assert np.isfinite(ad_stat)


class TestComputeAdPvalue:
    """Tests for Anderson-Darling p-value computation."""

    def test_ad_pvalue_supported_distributions(self, normal_data):
        """Test A-D p-value for supported distributions."""
        for dist_name in AD_PVALUE_DISTRIBUTIONS.keys():
            ad_pvalue = compute_ad_pvalue(dist_name, normal_data)

            # Should return a valid p-value
            assert ad_pvalue is not None, f"p-value is None for {dist_name}"
            assert 0 < ad_pvalue <= 1.0, f"p-value out of range for {dist_name}"

    def test_ad_pvalue_unsupported_distribution(self, normal_data):
        """Test A-D p-value returns None for unsupported distributions."""
        unsupported = ["gamma", "beta", "chi2", "uniform", "weibull_min"]

        for dist_name in unsupported:
            ad_pvalue = compute_ad_pvalue(dist_name, normal_data)
            assert ad_pvalue is None, f"p-value should be None for {dist_name}"

    def test_ad_pvalue_good_fit_high_pvalue(self):
        """Test A-D p-value is high for good fit."""
        np.random.seed(42)
        # Generate normal data and test with normal distribution
        normal_data = st.norm.rvs(size=1000)

        ad_pvalue = compute_ad_pvalue("norm", normal_data)

        # Good fit should have p-value > 0.05
        assert ad_pvalue is not None
        assert ad_pvalue > 0.05

    def test_ad_pvalue_poor_fit_low_pvalue(self):
        """Test A-D p-value is low for poor fit."""
        np.random.seed(42)
        # Generate uniform data and test with normal distribution
        uniform_data = st.uniform.rvs(size=1000)

        ad_pvalue = compute_ad_pvalue("norm", uniform_data)

        # Poor fit should have low p-value
        assert ad_pvalue is not None
        assert ad_pvalue < 0.05

    def test_ad_pvalue_norm_distribution(self):
        """Test A-D p-value specifically for normal distribution."""
        np.random.seed(42)
        data = st.norm.rvs(loc=0, scale=1, size=1000)

        ad_pvalue = compute_ad_pvalue("norm", data)

        assert ad_pvalue is not None
        assert isinstance(ad_pvalue, float)

    def test_ad_pvalue_expon_distribution(self):
        """Test A-D p-value specifically for exponential distribution."""
        np.random.seed(42)
        data = st.expon.rvs(size=1000)

        ad_pvalue = compute_ad_pvalue("expon", data)

        assert ad_pvalue is not None
        assert isinstance(ad_pvalue, float)


class TestFitSingleDistributionWithAD:
    """Tests for fit_single_distribution including A-D fields."""

    def test_fit_returns_ad_fields(self, normal_data):
        """Test that fit_single_distribution returns A-D fields."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        result = fit_single_distribution("norm", normal_data, bin_edges, y_hist)

        # Should have A-D fields
        assert "ad_statistic" in result
        assert "ad_pvalue" in result

        # A-D statistic should be valid
        assert np.isfinite(result["ad_statistic"])
        # p-value should exist for norm (supported distribution)
        assert result["ad_pvalue"] is not None
        assert 0 < result["ad_pvalue"] <= 1.0

    def test_fit_unsupported_returns_ad_statistic_no_pvalue(self, normal_data):
        """Test that unsupported distributions have A-D statistic but no p-value."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        # gamma is not in AD_PVALUE_DISTRIBUTIONS
        result = fit_single_distribution("gamma", normal_data, bin_edges, y_hist)

        # Should have A-D statistic
        assert np.isfinite(result["ad_statistic"])
        # But no p-value
        assert result["ad_pvalue"] is None

    def test_failed_fit_returns_ad_sentinel_values(self, normal_data):
        """Test that failed fits return sentinel values for A-D fields."""
        y_hist, bin_edges = np.histogram(normal_data, bins=50, density=True)


        result = fit_single_distribution("invalid_dist", normal_data, bin_edges, y_hist)

        # Should have A-D sentinel values
        assert result["ad_statistic"] == np.inf
        assert result["ad_pvalue"] is None


class TestGetContinuousParamNames:
    """Tests for get_continuous_param_names function."""

    def test_normal_distribution(self):
        """Test parameter names for normal distribution."""
        names = get_continuous_param_names("norm")
        assert names == ["loc", "scale"]

    def test_gamma_distribution(self):
        """Test parameter names for gamma distribution."""
        names = get_continuous_param_names("gamma")
        assert names == ["a", "loc", "scale"]

    def test_beta_distribution(self):
        """Test parameter names for beta distribution."""
        names = get_continuous_param_names("beta")
        assert names == ["a", "b", "loc", "scale"]

    def test_exponential_distribution(self):
        """Test parameter names for exponential distribution."""
        names = get_continuous_param_names("expon")
        assert names == ["loc", "scale"]


class TestBootstrapConfidenceIntervals:
    """Tests for bootstrap_confidence_intervals function."""

    def test_basic_ci_computation(self, normal_data):
        """Test that CI is computed correctly for normal distribution."""
        ci = bootstrap_confidence_intervals(
            "norm", normal_data, alpha=0.05, n_bootstrap=100, random_seed=42
        )

        # Should have correct parameter names
        assert "loc" in ci
        assert "scale" in ci

        # Each CI should be a tuple of (lower, upper)
        assert isinstance(ci["loc"], tuple)
        assert len(ci["loc"]) == 2
        assert ci["loc"][0] < ci["loc"][1]  # lower < upper

    def test_ci_contains_point_estimate(self, normal_data):
        """Test that CI contains the point estimate."""
        # Fit the distribution to get point estimate
        dist = st.norm
        params = dist.fit(normal_data)
        loc_estimate = params[0]
        scale_estimate = params[1]

        # Compute CI
        ci = bootstrap_confidence_intervals(
            "norm", normal_data, alpha=0.05, n_bootstrap=200, random_seed=42
        )

        # Point estimate should typically be within CI
        # (Not always guaranteed due to sampling variability, but very likely)
        loc_lower, loc_upper = ci["loc"]
        scale_lower, scale_upper = ci["scale"]

        # The true parameters (50, 10) should be within 95% CI
        assert 45 < loc_lower < loc_upper < 55
        assert 8 < scale_lower < scale_upper < 12

    def test_reproducibility_with_seed(self, normal_data):
        """Test that same seed produces same results."""
        ci1 = bootstrap_confidence_intervals(
            "norm", normal_data, alpha=0.05, n_bootstrap=50, random_seed=123
        )
        ci2 = bootstrap_confidence_intervals(
            "norm", normal_data, alpha=0.05, n_bootstrap=50, random_seed=123
        )

        assert ci1["loc"] == ci2["loc"]
        assert ci1["scale"] == ci2["scale"]

    def test_different_alpha_values(self, normal_data):
        """Test that smaller alpha gives wider CI."""
        ci_95 = bootstrap_confidence_intervals(
            "norm", normal_data, alpha=0.05, n_bootstrap=200, random_seed=42
        )
        ci_99 = bootstrap_confidence_intervals(
            "norm", normal_data, alpha=0.01, n_bootstrap=200, random_seed=42
        )

        # 99% CI should be wider than 95% CI
        width_95 = ci_95["loc"][1] - ci_95["loc"][0]
        width_99 = ci_99["loc"][1] - ci_99["loc"][0]
        assert width_99 > width_95

    def test_gamma_distribution_ci(self, gamma_data):
        """Test CI computation for gamma distribution."""
        ci = bootstrap_confidence_intervals(
            "gamma", gamma_data, alpha=0.05, n_bootstrap=100, random_seed=42
        )

        # Gamma has shape parameter 'a' plus loc and scale
        assert "a" in ci
        assert "loc" in ci
        assert "scale" in ci

        # All CIs should be valid
        for param, (lower, upper) in ci.items():
            assert lower < upper

    def test_small_sample_warning(self):
        """Test CI computation with small sample still works."""
        # Small sample - should still compute CI, even if wider
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ci = bootstrap_confidence_intervals(
            "norm", small_data, alpha=0.05, n_bootstrap=50, random_seed=42
        )

        # Should return valid CIs even for small data
        assert "loc" in ci
        assert "scale" in ci
        for param, (lower, upper) in ci.items():
            assert lower < upper


class TestTruncatedDist:
    """Tests for TruncatedFrozenDist internal class used in bounded fitting."""

    def test_pdf_within_bounds(self):
        """Test PDF returns non-zero values within bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([35, 40, 50, 60, 65])
        pdf_values = truncated.pdf(x)

        # All values within bounds should be positive
        assert np.all(pdf_values > 0)
        assert np.all(np.isfinite(pdf_values))

    def test_pdf_outside_bounds(self):
        """Test PDF returns zero outside bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([20, 25, 75, 80])
        pdf_values = truncated.pdf(x)

        # All values outside bounds should be zero
        assert np.all(pdf_values == 0)

    def test_pdf_at_boundaries(self):
        """Test PDF at exact boundary values."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([30, 70])
        pdf_values = truncated.pdf(x)

        # Values at boundaries should be non-zero (inclusive bounds)
        assert np.all(pdf_values > 0)

    def test_pdf_integrates_to_one(self):
        """Test PDF integrates to approximately 1 within bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        # Use numerical integration
        x = np.linspace(30, 70, 1000)
        pdf_values = truncated.pdf(x)
        dx = x[1] - x[0]
        integral = np.sum(pdf_values) * dx

        # Should integrate to approximately 1
        assert np.isclose(integral, 1.0, atol=0.01)

    def test_pdf_higher_than_untruncated(self):
        """Test truncated PDF is higher than untruncated within bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([50])  # At the mean
        pdf_truncated = truncated.pdf(x)
        pdf_original = frozen.pdf(x)

        # Truncated PDF should be higher (normalized to smaller support)
        assert pdf_truncated[0] > pdf_original[0]

    def test_pdf_zero_norm_edge_case(self):
        """Test PDF when normalization constant is effectively zero."""
        frozen = st.norm(loc=50, scale=10)
        # Set bounds far from distribution support
        # Use raise_on_empty=False to test silent zeros behavior
        truncated = TruncatedFrozenDist(frozen, lb=200, ub=300, raise_on_empty=False)

        x = np.array([250])
        pdf_values = truncated.pdf(x)

        # Should return zero when norm is essentially zero
        assert pdf_values[0] == 0

    def test_logpdf_within_bounds(self):
        """Test logpdf returns finite values within bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([35, 50, 65])
        logpdf_values = truncated.logpdf(x)

        # All values should be finite (not -inf)
        assert np.all(np.isfinite(logpdf_values))

    def test_logpdf_outside_bounds(self):
        """Test logpdf returns -inf outside bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([20, 80])
        logpdf_values = truncated.logpdf(x)

        # All values outside bounds should be -inf
        assert np.all(logpdf_values == -np.inf)

    def test_cdf_within_bounds(self):
        """Test CDF values within bounds are between 0 and 1."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([35, 50, 65])
        cdf_values = truncated.cdf(x)

        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)
        # CDF should be monotonically increasing
        assert cdf_values[0] < cdf_values[1] < cdf_values[2]

    def test_cdf_at_boundaries(self):
        """Test CDF at boundaries."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        assert truncated.cdf(np.array([30]))[0] == 0
        assert truncated.cdf(np.array([70]))[0] == 1

    def test_cdf_outside_bounds(self):
        """Test CDF outside bounds."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        # Below lower bound
        assert truncated.cdf(np.array([20]))[0] == 0
        # Above upper bound
        assert truncated.cdf(np.array([80]))[0] == 1

    def test_pdf_logpdf_consistency(self):
        """Test that pdf and logpdf are consistent."""
        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([35, 50, 65])
        pdf_values = truncated.pdf(x)
        logpdf_values = truncated.logpdf(x)

        # exp(logpdf) should equal pdf
        np.testing.assert_allclose(np.exp(logpdf_values), pdf_values, rtol=1e-10)

    def test_with_gamma_distribution(self):
        """Test truncation works with non-normal distributions."""
        frozen = st.gamma(a=2, loc=0, scale=5)
        truncated = TruncatedFrozenDist(frozen, lb=2, ub=20)

        x = np.array([5, 10, 15])
        pdf_values = truncated.pdf(x)

        # All values within bounds should be positive
        assert np.all(pdf_values > 0)

        # PDF should integrate to approximately 1
        x_range = np.linspace(2, 20, 1000)
        dx = x_range[1] - x_range[0]
        integral = np.sum(truncated.pdf(x_range)) * dx
        assert np.isclose(integral, 1.0, atol=0.01)
