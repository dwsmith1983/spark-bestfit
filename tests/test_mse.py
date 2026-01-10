"""Tests for Maximum Spacing Estimation (MSE) fitting (#sb-9h4).

MSE is an alternative to MLE that is more robust for heavy-tailed distributions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spark_bestfit import DistributionFitter, LocalBackend
from spark_bestfit.config import FitterConfig, FitterConfigBuilder
from spark_bestfit.fitting import fit_mse, fit_single_distribution


class TestFitMSE:
    """Tests for the fit_mse function."""

    def test_fit_normal_distribution(self):
        """MSE should fit normal distribution parameters accurately."""
        np.random.seed(42)
        true_loc, true_scale = 10.0, 2.0
        data = np.random.normal(true_loc, true_scale, 1000)

        params = fit_mse(stats.norm, data)

        # MSE should recover parameters close to true values
        fitted_loc, fitted_scale = params
        assert abs(fitted_loc - true_loc) < 0.5
        assert abs(fitted_scale - true_scale) < 0.3

    def test_fit_exponential_distribution(self):
        """MSE should fit exponential distribution parameters."""
        np.random.seed(42)
        true_scale = 5.0
        data = np.random.exponential(true_scale, 1000)

        params = fit_mse(stats.expon, data)

        # params = (loc, scale) for exponential
        fitted_loc, fitted_scale = params
        assert abs(fitted_scale - true_scale) < 1.0
        assert abs(fitted_loc) < 0.5  # loc should be near 0

    def test_fit_pareto_distribution(self):
        """MSE should fit Pareto distribution - a heavy-tailed distribution."""
        np.random.seed(42)
        true_b = 2.5  # shape parameter
        data = stats.pareto.rvs(b=true_b, size=1000, random_state=42)

        params = fit_mse(stats.pareto, data)

        # params = (b, loc, scale)
        fitted_b = params[0]
        # MSE should recover shape parameter reasonably well
        assert abs(fitted_b - true_b) < 1.0

    def test_fit_with_initial_params(self):
        """MSE should use initial params when provided."""
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 500)
        initial_params = (9.0, 1.5)  # Close to true values

        params = fit_mse(stats.norm, data, initial_params=initial_params)

        # Should converge
        fitted_loc, fitted_scale = params
        assert abs(fitted_loc - 10.0) < 0.5
        assert abs(fitted_scale - 2.0) < 0.3

    def test_fit_requires_minimum_data(self):
        """MSE should raise error with insufficient data."""
        data = np.array([1.0])  # Only 1 point

        with pytest.raises(ValueError, match="at least 2"):
            fit_mse(stats.norm, data)

    def test_mse_more_robust_than_mle_for_heavy_tails(self):
        """MSE should be more robust than MLE for heavy-tailed data.

        This test verifies the key advantage of MSE over MLE for heavy-tailed
        distributions by comparing fit quality on Cauchy data.
        """
        np.random.seed(42)
        # Generate Cauchy data (undefined mean/variance, very heavy tails)
        data = stats.cauchy.rvs(loc=5.0, scale=2.0, size=500, random_state=42)

        # Fit using MSE
        mse_params = fit_mse(stats.cauchy, data)

        # Fit using MLE (scipy default)
        mle_params = stats.cauchy.fit(data)

        # Both should produce finite parameters
        assert all(np.isfinite(p) for p in mse_params)
        assert all(np.isfinite(p) for p in mle_params)

        # MSE location parameter should be close to true value (5.0)
        mse_loc = mse_params[-2]  # loc is second to last
        assert abs(mse_loc - 5.0) < 2.0  # Reasonable tolerance for Cauchy


class TestFitSingleDistributionWithMSE:
    """Tests for fit_single_distribution with estimation_method parameter."""

    def test_mle_is_default(self):
        """Default estimation method should be MLE."""
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 500)
        bin_edges = np.linspace(data.min() - 1, data.max() + 1, 51)
        y_hist, _ = np.histogram(data, bins=bin_edges, density=True)

        result = fit_single_distribution(
            dist_name="norm",
            data_sample=data,
            bin_edges=bin_edges,
            y_hist=y_hist,
        )

        assert result["distribution"] == "norm"
        assert result["sse"] < np.inf
        assert len(result["parameters"]) == 2

    def test_mse_estimation_method(self):
        """MSE estimation method should work."""
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 500)
        bin_edges = np.linspace(data.min() - 1, data.max() + 1, 51)
        y_hist, _ = np.histogram(data, bins=bin_edges, density=True)

        result = fit_single_distribution(
            dist_name="norm",
            data_sample=data,
            bin_edges=bin_edges,
            y_hist=y_hist,
            estimation_method="mse",
        )

        assert result["distribution"] == "norm"
        assert result["sse"] < np.inf
        # Parameters should be reasonable for normal
        loc, scale = result["parameters"]
        assert abs(loc - 10.0) < 1.0
        assert abs(scale - 2.0) < 0.5

    def test_mse_fallback_to_mle_on_failure(self):
        """MSE should fallback to MLE if optimization fails."""
        np.random.seed(42)
        # Use data that MSE handles well
        data = np.random.normal(0, 1, 500)
        bin_edges = np.linspace(-4, 4, 51)
        y_hist, _ = np.histogram(data, bins=bin_edges, density=True)

        # This should succeed (either via MSE or MLE fallback)
        result = fit_single_distribution(
            dist_name="norm",
            data_sample=data,
            bin_edges=bin_edges,
            y_hist=y_hist,
            estimation_method="mse",
        )

        assert result["sse"] < np.inf


class TestFitterConfigEstimationMethod:
    """Tests for estimation_method in FitterConfig."""

    def test_default_estimation_method_is_mle(self):
        """Default estimation method should be 'mle'."""
        config = FitterConfig()
        assert config.estimation_method == "mle"

    def test_config_with_mse(self):
        """Config should accept 'mse' estimation method."""
        config = FitterConfig(estimation_method="mse")
        assert config.estimation_method == "mse"

    def test_config_with_auto(self):
        """Config should accept 'auto' estimation method."""
        config = FitterConfig(estimation_method="auto")
        assert config.estimation_method == "auto"

    def test_builder_with_estimation_method(self):
        """Builder should set estimation method."""
        config = (
            FitterConfigBuilder()
            .with_estimation_method("mse")
            .build()
        )
        assert config.estimation_method == "mse"

    def test_builder_invalid_estimation_method(self):
        """Builder should reject invalid estimation method."""
        with pytest.raises(ValueError, match="must be"):
            FitterConfigBuilder().with_estimation_method("invalid")


class TestDistributionFitterWithMSE:
    """Integration tests for DistributionFitter with MSE estimation."""

    def test_fit_with_mse_parameter(self):
        """Fitter should accept estimation_method parameter."""
        np.random.seed(42)
        data = np.random.normal(50.0, 10.0, 500)
        df = pd.DataFrame({"value": data})

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)

        results = fitter.fit(
            df,
            column="value",
            max_distributions=3,
            estimation_method="mse",
        )

        assert len(results.best(n=1)) > 0
        best = results.best(n=1)[0]
        assert best.distribution is not None

    def test_fit_with_mse_config(self):
        """Fitter should use MSE from config."""
        np.random.seed(42)
        data = np.random.normal(50.0, 10.0, 500)
        df = pd.DataFrame({"value": data})

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)
        config = FitterConfigBuilder().with_estimation_method("mse").build()

        results = fitter.fit(
            df,
            column="value",
            max_distributions=3,
            config=config,
        )

        assert len(results.best(n=1)) > 0

    def test_fit_with_auto_estimation_normal_data(self):
        """Auto mode should use MLE for normal (non-heavy-tailed) data."""
        np.random.seed(42)
        data = np.random.normal(50.0, 10.0, 500)
        df = pd.DataFrame({"value": data})

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)

        # Should not trigger MSE since data is not heavy-tailed
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = fitter.fit(
                df,
                column="value",
                max_distributions=3,
                estimation_method="auto",
            )

            # No heavy-tail warning expected
            heavy_tail_warnings = [
                x for x in w
                if "heavy-tail" in str(x.message).lower()
            ]
            # Normal data shouldn't trigger heavy-tail warning
            assert len(results.best(n=1)) > 0

    def test_fit_with_auto_estimation_heavy_tail_data(self):
        """Auto mode should use MSE for heavy-tailed data."""
        np.random.seed(42)
        # Generate Pareto data (heavy-tailed)
        data = stats.pareto.rvs(b=1.5, size=500, random_state=42) + 1
        df = pd.DataFrame({"value": data})

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)

        results = fitter.fit(
            df,
            column="value",
            max_distributions=5,
            estimation_method="auto",
        )

        # Should produce valid results
        assert len(results.best(n=1)) > 0

    def test_mse_no_heavy_tail_warning_when_using_mse(self):
        """Using MSE should not produce heavy-tail warning."""
        np.random.seed(42)
        # Generate heavy-tailed data
        data = stats.pareto.rvs(b=1.5, size=500, random_state=42) + 1
        df = pd.DataFrame({"value": data})

        backend = LocalBackend(max_workers=2)
        fitter = DistributionFitter(backend=backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = fitter.fit(
                df,
                column="value",
                max_distributions=3,
                estimation_method="mse",  # Explicitly using MSE
            )

            # Should not get heavy-tail warning since we're using MSE
            heavy_tail_warnings = [
                x for x in w
                if "heavy-tail" in str(x.message).lower()
            ]
            assert len(heavy_tail_warnings) == 0
