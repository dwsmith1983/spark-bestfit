"""Property-based tests for spark-bestfit using Hypothesis.

These tests verify invariants that must hold for ALL valid inputs,
not just specific test cases. This catches edge cases that example-based
tests might miss.

Property categories tested:
1. Serialization - round-trip invariants
2. Distribution functions - PDF, CDF, PPF mathematical properties
3. FitResults hierarchy - type invariants for lazy/eager classes
4. Fitting operations - result validity invariants
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings, HealthCheck

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.results import (
    BaseFitResults,
    DistributionFitResult,
    EagerFitResults,
    FitResults,
    LazyFitResults,
)

from .strategies import (
    builder_method_sequence,
    distribution_fit_result_data,
    distribution_with_params,
    finite_float_array,
    fitter_config,
    fitter_config_data,
    metric_name,
    probabilities,
    probability,
    scipy_continuous_distribution,
)


# =============================================================================
# Serialization Properties
# =============================================================================


class TestSerializationProperties:
    """Property-based tests for serialization round-trips."""

    @given(data=distribution_fit_result_data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_json_round_trip_preserves_distribution(self, data):
        """Property: JSON round-trip preserves distribution name exactly."""
        result = DistributionFitResult(**data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.json"
            result.save(path)
            loaded = DistributionFitResult.load(path)

            assert loaded.distribution == result.distribution

    @given(data=distribution_fit_result_data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_json_round_trip_preserves_parameters(self, data):
        """Property: JSON round-trip preserves parameters (within floating point precision)."""
        result = DistributionFitResult(**data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.json"
            result.save(path)
            loaded = DistributionFitResult.load(path)

            assert len(loaded.parameters) == len(result.parameters)
            for orig, loaded_val in zip(result.parameters, loaded.parameters):
                # Allow small floating point differences
                assert abs(orig - loaded_val) < 1e-10 or np.isclose(orig, loaded_val, rtol=1e-10)

    @given(data=distribution_fit_result_data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_pickle_round_trip_preserves_all_fields(self, data):
        """Property: Pickle round-trip preserves all fields exactly."""
        result = DistributionFitResult(**data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.pkl"
            result.save(path, format="pickle")
            loaded = DistributionFitResult.load(path)

            assert loaded.distribution == result.distribution
            assert loaded.parameters == result.parameters
            assert loaded.sse == result.sse
            assert loaded.aic == result.aic
            assert loaded.bic == result.bic

    @given(data=distribution_fit_result_data())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_round_trip_produces_functional_result(self, data):
        """Property: Loaded results can generate samples without error."""
        result = DistributionFitResult(**data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.json"
            result.save(path)
            loaded = DistributionFitResult.load(path)

            # Should be able to sample from loaded result
            samples = loaded.sample(size=10, random_state=42)
            assert len(samples) == 10
            assert np.all(np.isfinite(samples))


# =============================================================================
# Distribution Function Properties
# =============================================================================


class TestDistributionFunctionProperties:
    """Property-based tests for PDF, CDF, PPF mathematical invariants."""

    @given(dist_params=distribution_with_params(), x=finite_float_array(min_size=5, max_size=50))
    @settings(max_examples=50)
    def test_pdf_is_non_negative(self, dist_params, x):
        """Property: PDF(x) >= 0 for all x (where finite)."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        pdf_values = result.pdf(x)

        # PDF must be non-negative (where finite - some distributions have poles)
        finite_mask = np.isfinite(pdf_values)
        assert np.all(pdf_values[finite_mask] >= 0), f"PDF returned negative values for {dist_name}"
        # Infinite values should be positive (not negative infinity)
        inf_mask = np.isinf(pdf_values)
        if np.any(inf_mask):
            assert np.all(pdf_values[inf_mask] > 0), f"PDF returned -inf for {dist_name}"

    @given(dist_params=distribution_with_params(), x=finite_float_array(min_size=5, max_size=50))
    @settings(max_examples=50)
    def test_cdf_is_bounded_zero_one(self, dist_params, x):
        """Property: 0 <= CDF(x) <= 1 for all x."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        cdf_values = result.cdf(x)

        # CDF must be in [0, 1]
        assert np.all(cdf_values >= 0), f"CDF returned values < 0 for {dist_name}"
        assert np.all(cdf_values <= 1), f"CDF returned values > 1 for {dist_name}"
        # CDF must be finite
        assert np.all(np.isfinite(cdf_values)), f"CDF returned non-finite values for {dist_name}"

    @given(dist_params=distribution_with_params(), x=finite_float_array(min_size=10, max_size=50))
    @settings(max_examples=50)
    def test_cdf_is_monotonically_non_decreasing(self, dist_params, x):
        """Property: CDF is monotonically non-decreasing."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        # Sort x to test monotonicity
        x_sorted = np.sort(x)
        cdf_values = result.cdf(x_sorted)

        # CDF(x1) <= CDF(x2) when x1 <= x2
        diffs = np.diff(cdf_values)
        assert np.all(diffs >= -1e-10), f"CDF is not monotonic for {dist_name}"

    @given(dist_params=distribution_with_params(), p=probability())
    @settings(max_examples=50)
    def test_ppf_cdf_inverse_relationship(self, dist_params, p):
        """Property: CDF(PPF(p)) ≈ p for p in (0, 1)."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        # PPF is the inverse of CDF
        x = result.ppf(p)
        assume(np.isfinite(x))  # Skip if PPF returns inf (can happen at boundaries)

        p_recovered = result.cdf(np.array([x]))[0]

        # Should recover the original probability
        # Use rtol=1e-4 to account for numerical precision in scipy's PPF/CDF
        # implementations, especially for distributions with extreme shape parameters
        assert np.isclose(p_recovered, p, rtol=1e-4), f"CDF(PPF({p})) = {p_recovered} != {p} for {dist_name}"

    @given(dist_params=distribution_with_params(), probs=probabilities(min_size=5, max_size=20))
    @settings(max_examples=50)
    def test_ppf_is_monotonically_non_decreasing(self, dist_params, probs):
        """Property: PPF is monotonically non-decreasing."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        probs_sorted = np.sort(probs)
        ppf_values = np.array([result.ppf(p) for p in probs_sorted])

        # Filter out infinities that can occur at extreme probabilities
        finite_mask = np.isfinite(ppf_values)
        ppf_finite = ppf_values[finite_mask]

        if len(ppf_finite) > 1:
            diffs = np.diff(ppf_finite)
            assert np.all(diffs >= -1e-10), f"PPF is not monotonic for {dist_name}"


# =============================================================================
# FitResults Hierarchy Properties
# =============================================================================


class TestFitResultsHierarchyProperties:
    """Property-based tests for FitResults class hierarchy invariants."""

    @given(data=finite_float_array(min_size=50, max_size=200))
    @settings(max_examples=20, deadline=None)
    def test_eager_is_lazy_always_false(self, data):
        """Property: EagerFitResults.is_lazy is always False."""
        df = pd.DataFrame({"value": data})

        # Create minimal fit result for testing
        results_df = pd.DataFrame(
            {
                "column_name": ["value"],
                "distribution": ["norm"],
                "parameters": [[float(np.mean(data)), float(np.std(data))]],
                "sse": [0.01],
                "aic": [100.0],
                "bic": [105.0],
                "ks_statistic": [0.05],
                "pvalue": [0.5],
                "ad_statistic": [None],
                "ad_pvalue": [None],
                "data_min": [float(np.min(data))],
                "data_max": [float(np.max(data))],
                "data_mean": [float(np.mean(data))],
                "data_stddev": [float(np.std(data))],
                "data_count": [float(len(data))],
                "lower_bound": [None],
                "upper_bound": [None],
            }
        )
        results = FitResults(results_df)

        assert isinstance(results, EagerFitResults)
        assert results.is_lazy is False

    @given(data=finite_float_array(min_size=50, max_size=200))
    @settings(max_examples=20, deadline=None)
    def test_eager_materialize_returns_self(self, data):
        """Property: EagerFitResults.materialize() returns self."""
        results_df = pd.DataFrame(
            {
                "column_name": ["value"],
                "distribution": ["norm"],
                "parameters": [[float(np.mean(data)), float(np.std(data))]],
                "sse": [0.01],
                "aic": [100.0],
                "bic": [105.0],
                "ks_statistic": [0.05],
                "pvalue": [0.5],
                "ad_statistic": [None],
                "ad_pvalue": [None],
                "data_min": [float(np.min(data))],
                "data_max": [float(np.max(data))],
                "data_mean": [float(np.mean(data))],
                "data_stddev": [float(np.std(data))],
                "data_count": [float(len(data))],
                "lower_bound": [None],
                "upper_bound": [None],
            }
        )
        results = FitResults(results_df)

        materialized = results.materialize()

        assert materialized is results

    @given(data=finite_float_array(min_size=50, max_size=200))
    @settings(max_examples=20, deadline=None)
    def test_base_fit_results_isinstance(self, data):
        """Property: All FitResults instances are BaseFitResults."""
        results_df = pd.DataFrame(
            {
                "column_name": ["value"],
                "distribution": ["norm"],
                "parameters": [[float(np.mean(data)), float(np.std(data))]],
                "sse": [0.01],
                "aic": [100.0],
                "bic": [105.0],
                "ks_statistic": [0.05],
                "pvalue": [0.5],
                "ad_statistic": [None],
                "ad_pvalue": [None],
                "data_min": [float(np.min(data))],
                "data_max": [float(np.max(data))],
                "data_mean": [float(np.mean(data))],
                "data_stddev": [float(np.std(data))],
                "data_count": [float(len(data))],
                "lower_bound": [None],
                "upper_bound": [None],
            }
        )
        results = FitResults(results_df)

        assert isinstance(results, BaseFitResults)


# =============================================================================
# Fitting Operation Properties
# =============================================================================


class TestFittingOperationProperties:
    """Property-based tests for fitting operation invariants."""

    @given(data=finite_float_array(min_size=100, max_size=500, min_value=0.1, max_value=1000))
    @settings(max_examples=15, deadline=None)
    def test_fitting_returns_finite_metrics(self, data):
        """Property: Fitting valid data returns results with finite AIC/BIC."""
        # Use positive data to avoid issues with log-based distributions
        df = pd.DataFrame({"value": np.abs(data) + 0.1})

        backend = LocalBackend()
        from spark_bestfit.continuous_fitter import DistributionFitter

        fitter = DistributionFitter(backend=backend)

        results = fitter.fit(df, column="value", max_distributions=3)
        best = results.best(n=1, metric="aic")

        if best:  # May be empty for edge case data
            assert np.isfinite(best[0].aic), "AIC should be finite"
            assert np.isfinite(best[0].bic), "BIC should be finite"

    @given(metric=metric_name())
    @settings(max_examples=20, deadline=None)
    def test_best_with_valid_metric_does_not_raise(self, metric):
        """Property: best() with any valid metric name does not raise."""
        # Create a simple result set
        results_df = pd.DataFrame(
            {
                "column_name": ["value", "value"],
                "distribution": ["norm", "expon"],
                "parameters": [[50.0, 10.0], [0.0, 10.0]],
                "sse": [0.01, 0.02],
                "aic": [100.0, 110.0],
                "bic": [105.0, 115.0],
                "ks_statistic": [0.05, 0.06],
                "pvalue": [0.5, 0.4],
                "ad_statistic": [0.3, 0.4],
                "ad_pvalue": [0.2, 0.15],
                "data_min": [20.0, 20.0],
                "data_max": [80.0, 80.0],
                "data_mean": [50.0, 50.0],
                "data_stddev": [10.0, 10.0],
                "data_count": [1000.0, 1000.0],
                "lower_bound": [None, None],
                "upper_bound": [None, None],
            }
        )
        results = FitResults(results_df)

        # Should not raise
        best = results.best(n=1, metric=metric)
        assert len(best) <= 1

    @given(n=finite_float_array(min_size=100, max_size=300, min_value=1, max_value=100))
    @settings(max_examples=10, deadline=None)
    def test_filter_returns_subset(self, n):
        """Property: filter() returns result with count <= original count."""
        data = n  # Use the generated array
        df = pd.DataFrame({"value": data})

        backend = LocalBackend()
        from spark_bestfit.continuous_fitter import DistributionFitter

        fitter = DistributionFitter(backend=backend)

        results = fitter.fit(df, column="value", max_distributions=5)
        original_count = results.count()

        # Filter to distributions with low SSE (using sse_threshold parameter)
        filtered = results.filter(sse_threshold=0.5)
        filtered_count = filtered.count()

        assert filtered_count <= original_count


# =============================================================================
# Sample Generation Properties
# =============================================================================


class TestSampleGenerationProperties:
    """Property-based tests for sample generation invariants."""

    @given(dist_params=distribution_with_params())
    @settings(max_examples=30)
    def test_samples_are_finite(self, dist_params):
        """Property: Generated samples are finite (no inf/nan)."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        samples = result.sample(size=100, random_state=42)

        assert np.all(np.isfinite(samples)), f"Samples contain non-finite values for {dist_name}"

    @given(dist_params=distribution_with_params())
    @settings(max_examples=30)
    def test_sample_size_is_correct(self, dist_params):
        """Property: Sample size matches requested size."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        for size in [10, 100, 1000]:
            samples = result.sample(size=size, random_state=42)
            assert len(samples) == size, f"Sample size mismatch for {dist_name}"

    @given(dist_params=distribution_with_params())
    @settings(max_examples=20)
    def test_random_state_reproducibility(self, dist_params):
        """Property: Same random_state produces same samples."""
        dist_name, params = dist_params
        result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

        samples1 = result.sample(size=100, random_state=12345)
        samples2 = result.sample(size=100, random_state=12345)

        np.testing.assert_array_equal(
            samples1, samples2, err_msg=f"Samples not reproducible for {dist_name}"
        )


# =============================================================================
# FitterConfig Properties
# =============================================================================


class TestFitterConfigProperties:
    """Property-based tests for FitterConfig and FitterConfigBuilder."""

    @given(data=fitter_config_data())
    @settings(max_examples=50)
    def test_config_is_frozen(self, data):
        """Property: FitterConfig is immutable (frozen dataclass)."""
        from spark_bestfit.config import FitterConfig

        config = FitterConfig(**data)

        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.bins = 999

    @given(data=fitter_config_data())
    @settings(max_examples=50)
    def test_config_equality(self, data):
        """Property: Two configs with same data are equal."""
        from spark_bestfit.config import FitterConfig

        config1 = FitterConfig(**data)
        config2 = FitterConfig(**data)

        assert config1 == config2

    @given(data=fitter_config_data())
    @settings(max_examples=50)
    def test_config_hash_consistency(self, data):
        """Property: Equal configs have equal hashes."""
        from spark_bestfit.config import FitterConfig

        config1 = FitterConfig(**data)
        config2 = FitterConfig(**data)

        assert hash(config1) == hash(config2)

    @given(config=fitter_config())
    @settings(max_examples=30)
    def test_with_progress_callback_returns_new_instance(self, config):
        """Property: with_progress_callback returns a new config, not mutated original."""

        def dummy_callback(completed, total, percent):
            pass

        new_config = config.with_progress_callback(dummy_callback)

        # Should be a different object
        assert new_config is not config
        # New config should have the callback
        assert new_config.progress_callback is dummy_callback
        # Original should be unchanged
        assert config.progress_callback is None

    @given(config=fitter_config())
    @settings(max_examples=30)
    def test_config_all_fields_accessible(self, config):
        """Property: All FitterConfig fields are accessible without error."""
        # Access all fields - should not raise
        _ = config.bins
        _ = config.use_rice_rule
        _ = config.support_at_zero
        _ = config.max_distributions
        _ = config.prefilter
        _ = config.enable_sampling
        _ = config.sample_fraction
        _ = config.max_sample_size
        _ = config.sample_threshold
        _ = config.bounded
        _ = config.lower_bound
        _ = config.upper_bound
        _ = config.num_partitions
        _ = config.lazy_metrics
        _ = config.progress_callback


class TestFitterConfigBuilderProperties:
    """Property-based tests for FitterConfigBuilder."""

    @given(methods=builder_method_sequence())
    @settings(max_examples=50)
    def test_builder_methods_return_self(self, methods):
        """Property: All builder methods return self for chaining."""
        from spark_bestfit.config import FitterConfigBuilder

        builder = FitterConfigBuilder()

        for method_name, kwargs in methods:
            method = getattr(builder, method_name)
            result = method(**kwargs)
            assert result is builder, f"{method_name} should return self"

    @given(methods=builder_method_sequence())
    @settings(max_examples=50)
    def test_builder_build_returns_fitter_config(self, methods):
        """Property: build() always returns a valid FitterConfig."""
        from spark_bestfit.config import FitterConfig, FitterConfigBuilder

        builder = FitterConfigBuilder()

        for method_name, kwargs in methods:
            method = getattr(builder, method_name)
            method(**kwargs)

        config = builder.build()

        assert isinstance(config, FitterConfig)

    @given(methods=builder_method_sequence())
    @settings(max_examples=30)
    def test_builder_produces_frozen_config(self, methods):
        """Property: Builder always produces an immutable (frozen) config."""
        from spark_bestfit.config import FitterConfigBuilder

        builder = FitterConfigBuilder()

        for method_name, kwargs in methods:
            getattr(builder, method_name)(**kwargs)

        config = builder.build()

        # Should not be able to modify
        with pytest.raises(Exception):
            config.bins = 999

    def test_default_builder_matches_default_config(self):
        """Property: Default builder produces config with default values."""
        from spark_bestfit.config import FitterConfig, FitterConfigBuilder

        default_config = FitterConfig()
        builder_config = FitterConfigBuilder().build()

        # All fields should match (except progress_callback which is always None)
        assert builder_config.bins == default_config.bins
        assert builder_config.use_rice_rule == default_config.use_rice_rule
        assert builder_config.support_at_zero == default_config.support_at_zero
        assert builder_config.max_distributions == default_config.max_distributions
        assert builder_config.prefilter == default_config.prefilter
        assert builder_config.enable_sampling == default_config.enable_sampling
        assert builder_config.sample_fraction == default_config.sample_fraction
        assert builder_config.max_sample_size == default_config.max_sample_size
        assert builder_config.sample_threshold == default_config.sample_threshold
        assert builder_config.bounded == default_config.bounded
        assert builder_config.lower_bound == default_config.lower_bound
        assert builder_config.upper_bound == default_config.upper_bound
        assert builder_config.num_partitions == default_config.num_partitions
        assert builder_config.lazy_metrics == default_config.lazy_metrics


class TestFitterConfigIntegrationProperties:
    """Property-based tests for FitterConfig integration with fitters."""

    @given(config=fitter_config(), data=finite_float_array(min_size=100, max_size=300))
    @settings(max_examples=10, deadline=None)
    def test_fit_with_config_does_not_raise(self, config, data):
        """Property: fit() with any valid config does not raise."""
        from spark_bestfit.config import FitterConfig
        from spark_bestfit.continuous_fitter import DistributionFitter

        # Force sensible values for actual fitting
        config = FitterConfig(
            bins=min(config.bins if isinstance(config.bins, int) else 50, 100),
            max_distributions=3,  # Limit for speed
            enable_sampling=False,
            lazy_metrics=config.lazy_metrics,
            bounded=False,  # Avoid bound validation issues
        )

        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)
        df = pd.DataFrame({"value": data})

        # Should not raise
        results = fitter.fit(df, column="value", config=config)
        assert results.count() >= 0

    @given(data=finite_float_array(min_size=100, max_size=200))
    @settings(max_examples=10, deadline=None)
    def test_config_vs_params_equivalence(self, data):
        """Property: config= produces same results as equivalent direct params."""
        from spark_bestfit.config import FitterConfig
        from spark_bestfit.continuous_fitter import DistributionFitter

        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)
        df = pd.DataFrame({"value": data})

        # Fit with direct params
        results_params = fitter.fit(
            df,
            column="value",
            bins=30,
            max_distributions=3,
            lazy_metrics=True,
        )

        # Fit with equivalent config
        config = FitterConfig(
            bins=30,
            max_distributions=3,
            lazy_metrics=True,
        )
        results_config = fitter.fit(df, column="value", config=config)

        # Should have same distributions fitted
        params_dists = {r.distribution for r in results_params.best(n=10, metric="aic")}
        config_dists = {r.distribution for r in results_config.best(n=10, metric="aic")}

        assert params_dists == config_dists
