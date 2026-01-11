"""Tests for right-censored data fitting (v2.9.0 feature).

These tests verify that:
1. fit_censored_mle() correctly estimates parameters with censored data
2. FitterConfig.censoring_column is passed through correctly
3. FitterConfigBuilder.with_censoring() creates proper config
4. KS/AD statistics are skipped for censored fits
5. Heavy censoring warning is raised
6. Censoring validation catches invalid columns
7. Integration with all three backends (LocalBackend, SparkBackend, RayBackend)

Uses LocalBackend for most tests. Spark-specific tests are in test_spark_backend.py.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from spark_bestfit import DistributionFitter, FitterConfigBuilder
from spark_bestfit.estimation import fit_censored_mle
from spark_bestfit.distributions import DistributionRegistry


@pytest.fixture
def survival_fitter(local_backend):
    """Create a DistributionFitter with only survival-analysis distributions.

    Censored MLE is expensive - limit to relevant distributions for fast tests.
    """
    # Get all distribution names and exclude all except survival-analysis ones
    all_dists = set(DistributionRegistry.ALL_DISTRIBUTIONS)
    keep_dists = {"expon", "weibull_min", "gamma", "lognorm"}
    exclude_dists = tuple(all_dists - keep_dists)

    return DistributionFitter(backend=local_backend, excluded_distributions=exclude_dists)


class TestFitCensoredMLE:
    """Unit tests for fit_censored_mle function."""

    def test_no_censoring_matches_standard_mle(self):
        """When all observations are observed, censored MLE matches standard MLE."""
        np.random.seed(42)
        data = st.expon.rvs(scale=10, size=500)
        censoring = np.ones(500, dtype=bool)  # All observed

        params_censored = fit_censored_mle(st.expon, data, censoring)
        params_standard = st.expon.fit(data)

        # loc and scale should be very similar
        assert params_censored[0] == pytest.approx(params_standard[0], rel=0.1)
        assert params_censored[1] == pytest.approx(params_standard[1], rel=0.1)

    def test_weibull_parameter_recovery(self):
        """Censored MLE recovers Weibull parameters with moderate censoring."""
        np.random.seed(42)
        true_shape = 2.0
        true_scale = 10.0

        # Simulate survival data with ~30% censoring
        n = 1000
        true_times = st.weibull_min.rvs(c=true_shape, scale=true_scale, size=n)
        censor_times = np.random.uniform(5, 20, size=n)
        observed_times = np.minimum(true_times, censor_times)
        event_occurred = true_times <= censor_times

        # About 30% should be censored
        censor_fraction = 1 - event_occurred.mean()
        assert 0.1 < censor_fraction < 0.5, f"Censoring fraction: {censor_fraction}"

        params = fit_censored_mle(st.weibull_min, observed_times, event_occurred)

        # Extract shape (c) and scale from params (c, loc, scale)
        fitted_shape = params[0]
        fitted_scale = params[2]

        # Should recover parameters within 15%
        assert fitted_shape == pytest.approx(true_shape, rel=0.15)
        assert fitted_scale == pytest.approx(true_scale, rel=0.15)

    def test_exponential_parameter_recovery(self):
        """Censored MLE recovers exponential scale with censoring."""
        np.random.seed(123)
        true_scale = 5.0

        # Simulate with ~25% censoring
        n = 800
        true_times = st.expon.rvs(scale=true_scale, size=n)
        censor_times = np.random.uniform(2, 15, size=n)
        observed_times = np.minimum(true_times, censor_times)
        event_occurred = true_times <= censor_times

        params = fit_censored_mle(st.expon, observed_times, event_occurred)

        # Scale is params[1] for exponential (loc, scale)
        fitted_scale = params[1]

        # Should recover within 15%
        assert fitted_scale == pytest.approx(true_scale, rel=0.15)

    def test_heavy_censoring_warning(self):
        """Warning is raised when >80% of observations are censored."""
        np.random.seed(42)
        data = st.expon.rvs(scale=10, size=100)
        # 90% censored (only 10 events)
        censoring = np.zeros(100, dtype=bool)
        censoring[:10] = True

        with pytest.warns(UserWarning, match="Heavy censoring detected"):
            fit_censored_mle(st.expon, data, censoring)

    def test_error_on_no_observed_events(self):
        """Error raised when all observations are censored."""
        np.random.seed(42)
        data = st.expon.rvs(scale=10, size=100)
        censoring = np.zeros(100, dtype=bool)  # All censored

        with pytest.raises(ValueError, match="no observed events"):
            fit_censored_mle(st.expon, data, censoring)

    def test_error_on_length_mismatch(self):
        """Error raised when data and censoring indicator lengths differ."""
        data = np.array([1, 2, 3, 4, 5])
        censoring = np.array([True, True, False])  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            fit_censored_mle(st.expon, data, censoring)


class TestCensoredFitterConfig:
    """Tests for FitterConfig and FitterConfigBuilder censoring support."""

    def test_config_builder_with_censoring(self):
        """FitterConfigBuilder.with_censoring() sets censoring_column."""
        config = FitterConfigBuilder().with_censoring("event_occurred").build()

        assert config.censoring_column == "event_occurred"

    def test_config_builder_chaining(self):
        """Censoring can be combined with other config options."""
        config = (
            FitterConfigBuilder()
            .with_censoring("event")
            .with_bins(100)
            .with_sampling(fraction=0.5)
            .build()
        )

        assert config.censoring_column == "event"
        assert config.bins == 100
        assert config.sample_fraction == 0.5


class TestCensoredFitting:
    """Integration tests for censored distribution fitting."""

    def test_censored_fitting_basic(self, survival_fitter):
        """Basic censored fitting works with LocalBackend."""
        np.random.seed(42)

        # Simulate survival data
        n = 500
        true_times = st.weibull_min.rvs(c=1.5, scale=10, size=n)
        censor_times = np.random.uniform(5, 25, size=n)
        observed_times = np.minimum(true_times, censor_times)
        event_occurred = true_times <= censor_times

        df = pd.DataFrame({
            "time": observed_times,
            "event": event_occurred,
        })

        config = FitterConfigBuilder().with_censoring("event").build()
        # survival_fitter only includes expon, weibull_min, gamma, lognorm
        results = survival_fitter.fit(df, column="time", config=config, max_distributions=2)

        # Should have results
        assert len(results.best(n=2)) > 0

    def test_censored_fitting_skips_ks_ad(self, survival_fitter):
        """KS and AD statistics are skipped for censored fits."""
        np.random.seed(42)

        n = 300
        true_times = st.expon.rvs(scale=10, size=n)
        censor_times = np.random.uniform(5, 20, size=n)
        observed_times = np.minimum(true_times, censor_times)
        event_occurred = true_times <= censor_times

        df = pd.DataFrame({
            "time": observed_times,
            "event": event_occurred,
        })

        config = FitterConfigBuilder().with_censoring("event").build()
        # survival_fitter only includes expon, weibull_min, gamma, lognorm
        results = survival_fitter.fit(df, column="time", config=config, max_distributions=1)

        best = results.best(n=1)[0]

        # KS and AD should be None for censored fits
        assert best.ks_statistic is None
        assert best.pvalue is None
        assert best.ad_statistic is None

        # AIC/BIC should still be computed
        assert best.aic is not None
        assert best.bic is not None
        assert np.isfinite(best.aic)
        assert np.isfinite(best.bic)

    def test_censored_vs_uncensored_bias(self, local_backend):
        """Censored MLE should be less biased than standard MLE on censored data."""
        # This test uses direct fit_censored_mle to avoid slow fitter
        np.random.seed(42)

        true_scale = 10.0
        n = 600

        # Generate heavily censored data
        true_times = st.expon.rvs(scale=true_scale, size=n)
        censor_times = np.random.uniform(2, 8, size=n)  # Early censoring
        observed_times = np.minimum(true_times, censor_times)
        event_occurred = true_times <= censor_times

        # Fit with censoring (correct approach)
        params_censored = fit_censored_mle(st.expon, observed_times, event_occurred)
        scale_censored = params_censored[1]

        # Fit without censoring (biased - ignores censoring)
        params_uncensored = st.expon.fit(observed_times)
        scale_uncensored = params_uncensored[1]

        # Censored MLE should be closer to true scale (10.0)
        error_censored = abs(scale_censored - true_scale)
        error_uncensored = abs(scale_uncensored - true_scale)

        # Uncensored MLE underestimates scale because it treats censored times as events
        assert scale_uncensored < true_scale, "Uncensored should underestimate scale"
        assert error_censored < error_uncensored, "Censored MLE should be less biased"


class TestCensoredValidation:
    """Tests for censoring column validation."""

    def test_invalid_censoring_column_not_found(self, local_backend):
        """Error raised when censoring column doesn't exist."""
        df = pd.DataFrame({
            "time": [1, 2, 3, 4, 5],
        })

        config = FitterConfigBuilder().with_censoring("nonexistent").build()
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="not found"):
            fitter.fit(df, column="time", config=config)

    def test_invalid_censoring_column_wrong_type(self, local_backend):
        """Error raised when censoring column is not boolean/integer."""
        df = pd.DataFrame({
            "time": [1, 2, 3, 4, 5],
            "event": ["yes", "yes", "no", "yes", "no"],  # String, not boolean
        })

        config = FitterConfigBuilder().with_censoring("event").build()
        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="boolean or integer"):
            fitter.fit(df, column="time", config=config)

    def test_valid_censoring_column_boolean(self, survival_fitter):
        """Boolean censoring column is accepted."""
        np.random.seed(42)
        df = pd.DataFrame({
            "time": np.random.exponential(10, 100),
            "event": np.random.choice([True, False], 100),
        })

        config = FitterConfigBuilder().with_censoring("event").build()
        # survival_fitter only includes expon, weibull_min, gamma, lognorm
        results = survival_fitter.fit(df, column="time", config=config, max_distributions=2)
        assert len(results.best(n=2)) > 0

    def test_valid_censoring_column_integer(self, survival_fitter):
        """Integer (0/1) censoring column is accepted."""
        np.random.seed(42)
        df = pd.DataFrame({
            "time": np.random.exponential(10, 100),
            "event": np.random.choice([0, 1], 100),
        })

        config = FitterConfigBuilder().with_censoring("event").build()
        # survival_fitter only includes expon, weibull_min, gamma, lognorm
        results = survival_fitter.fit(df, column="time", config=config, max_distributions=2)
        assert len(results.best(n=2)) > 0
