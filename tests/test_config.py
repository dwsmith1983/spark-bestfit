"""Tests for FitterConfig dataclass and FitterConfigBuilder fluent API."""

import dataclasses

import pytest

from spark_bestfit import FitterConfig, FitterConfigBuilder


class TestFitterConfig:
    """Test FitterConfig dataclass behavior."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FitterConfig()

        # Histogram defaults
        assert config.bins == 50
        assert config.use_rice_rule is True

        # Distribution selection defaults
        assert config.support_at_zero is False
        assert config.max_distributions is None
        assert config.prefilter is False

        # Sampling defaults
        assert config.enable_sampling is True
        assert config.sample_fraction is None
        assert config.max_sample_size == 1_000_000
        assert config.sample_threshold == 10_000_000

        # Bounds defaults
        assert config.bounded is False
        assert config.lower_bound is None
        assert config.upper_bound is None

        # Performance defaults
        assert config.num_partitions is None
        assert config.lazy_metrics is False

        # Callback defaults
        assert config.progress_callback is None

    def test_frozen_immutable(self):
        """Test that config is immutable (frozen dataclass)."""
        config = FitterConfig()

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.bins = 100  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.bounded = True  # type: ignore

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = FitterConfig(
            bins=100,
            use_rice_rule=False,
            support_at_zero=True,
            max_distributions=5,
            prefilter=True,
            enable_sampling=False,
            sample_fraction=0.1,
            max_sample_size=500_000,
            sample_threshold=5_000_000,
            bounded=True,
            lower_bound=0.0,
            upper_bound=100.0,
            num_partitions=4,
            lazy_metrics=True,
        )

        assert config.bins == 100
        assert config.use_rice_rule is False
        assert config.support_at_zero is True
        assert config.max_distributions == 5
        assert config.prefilter is True
        assert config.enable_sampling is False
        assert config.sample_fraction == 0.1
        assert config.max_sample_size == 500_000
        assert config.sample_threshold == 5_000_000
        assert config.bounded is True
        assert config.lower_bound == 0.0
        assert config.upper_bound == 100.0
        assert config.num_partitions == 4
        assert config.lazy_metrics is True

    def test_with_progress_callback(self):
        """Test with_progress_callback returns new config with callback."""
        config = FitterConfig()
        callback_calls = []

        def my_callback(completed: int, total: int, percent: float) -> None:
            callback_calls.append((completed, total, percent))

        # Returns new config (original unchanged)
        new_config = config.with_progress_callback(my_callback)

        assert config.progress_callback is None  # Original unchanged
        assert new_config.progress_callback is my_callback

        # Callback is callable
        new_config.progress_callback(1, 10, 0.1)
        assert callback_calls == [(1, 10, 0.1)]

    def test_with_progress_callback_override(self):
        """Test progress callback can be overridden on existing config."""
        callback1 = lambda c, t, p: None
        callback2 = lambda c, t, p: None

        config1 = FitterConfig(progress_callback=callback1)
        config2 = config1.with_progress_callback(callback2)

        assert config1.progress_callback is callback1
        assert config2.progress_callback is callback2

    def test_bounds_with_dict(self):
        """Test bounds can be per-column dicts."""
        config = FitterConfig(
            bounded=True,
            lower_bound={"col1": 0.0, "col2": -10.0},
            upper_bound={"col1": 100.0, "col2": 10.0},
        )

        assert config.lower_bound == {"col1": 0.0, "col2": -10.0}
        assert config.upper_bound == {"col1": 100.0, "col2": 10.0}

    def test_bins_with_tuple(self):
        """Test bins can be explicit bin edges."""
        bin_edges = (0.0, 10.0, 20.0, 50.0, 100.0)
        config = FitterConfig(bins=bin_edges)

        assert config.bins == bin_edges

    def test_prefilter_modes(self):
        """Test prefilter accepts bool and string modes."""
        config_bool = FitterConfig(prefilter=True)
        config_str = FitterConfig(prefilter="aggressive")

        assert config_bool.prefilter is True
        assert config_str.prefilter == "aggressive"


class TestFitterConfigBuilder:
    """Test FitterConfigBuilder fluent API."""

    def test_builder_chain(self):
        """Test method chaining returns builder for fluent API."""
        builder = FitterConfigBuilder()

        # Each method should return the builder
        result = builder.with_bins(100)
        assert result is builder

        result = builder.with_bounds(lower=0)
        assert result is builder

        result = builder.with_sampling(fraction=0.1)
        assert result is builder

        result = builder.with_lazy_metrics()
        assert result is builder

    def test_with_bins(self):
        """Test with_bins configures histogram binning."""
        config = FitterConfigBuilder().with_bins(100, use_rice_rule=False).build()

        assert config.bins == 100
        assert config.use_rice_rule is False

    def test_with_bins_tuple(self):
        """Test with_bins accepts bin edge tuple."""
        edges = (0.0, 10.0, 20.0, 100.0)
        config = FitterConfigBuilder().with_bins(edges).build()

        assert config.bins == edges

    def test_with_bounds_explicit(self):
        """Test with_bounds with explicit bounds."""
        config = FitterConfigBuilder().with_bounds(lower=0.0, upper=100.0).build()

        assert config.bounded is True
        assert config.lower_bound == 0.0
        assert config.upper_bound == 100.0

    def test_with_bounds_auto_detect(self):
        """Test with_bounds with auto_detect."""
        config = FitterConfigBuilder().with_bounds(auto_detect=True).build()

        assert config.bounded is True
        assert config.lower_bound is None
        assert config.upper_bound is None

    def test_with_bounds_partial(self):
        """Test with_bounds with only one bound."""
        config = FitterConfigBuilder().with_bounds(lower=0.0).build()

        assert config.bounded is True
        assert config.lower_bound == 0.0
        assert config.upper_bound is None

    def test_with_bounds_dict(self):
        """Test with_bounds with per-column dict."""
        bounds = {"col1": 0.0, "col2": -10.0}
        config = FitterConfigBuilder().with_bounds(lower=bounds).build()

        assert config.bounded is True
        assert config.lower_bound == bounds

    def test_with_sampling(self):
        """Test with_sampling configures sampling behavior."""
        config = (
            FitterConfigBuilder()
            .with_sampling(fraction=0.05, max_size=100_000, threshold=1_000_000, enabled=True)
            .build()
        )

        assert config.enable_sampling is True
        assert config.sample_fraction == 0.05
        assert config.max_sample_size == 100_000
        assert config.sample_threshold == 1_000_000

    def test_with_sampling_disabled(self):
        """Test with_sampling can disable sampling."""
        config = FitterConfigBuilder().with_sampling(enabled=False).build()

        assert config.enable_sampling is False

    def test_with_lazy_metrics(self):
        """Test with_lazy_metrics enables lazy computation."""
        config = FitterConfigBuilder().with_lazy_metrics().build()
        assert config.lazy_metrics is True

        config = FitterConfigBuilder().with_lazy_metrics(lazy=False).build()
        assert config.lazy_metrics is False

    def test_with_prefilter(self):
        """Test with_prefilter configures distribution filtering."""
        config = FitterConfigBuilder().with_prefilter().build()
        assert config.prefilter is True

        config = FitterConfigBuilder().with_prefilter(mode="aggressive").build()
        assert config.prefilter == "aggressive"

    def test_with_support_at_zero(self):
        """Test with_support_at_zero for non-negative distributions."""
        config = FitterConfigBuilder().with_support_at_zero().build()
        assert config.support_at_zero is True

        config = FitterConfigBuilder().with_support_at_zero(enabled=False).build()
        assert config.support_at_zero is False

    def test_with_max_distributions(self):
        """Test with_max_distributions limits fitted distributions."""
        config = FitterConfigBuilder().with_max_distributions(5).build()
        assert config.max_distributions == 5

        config = FitterConfigBuilder().with_max_distributions(None).build()
        assert config.max_distributions is None

    def test_with_partitions(self):
        """Test with_partitions configures parallel partitions."""
        config = FitterConfigBuilder().with_partitions(8).build()
        assert config.num_partitions == 8

    def test_build_returns_config(self):
        """Test build() returns FitterConfig instance."""
        builder = FitterConfigBuilder()
        config = builder.build()

        assert isinstance(config, FitterConfig)

    def test_complete_builder_example(self):
        """Test complete builder usage pattern from docstring."""
        config = (
            FitterConfigBuilder()
            .with_bins(100)
            .with_bounds(lower=0, upper=100)
            .with_sampling(fraction=0.1)
            .with_lazy_metrics()
            .build()
        )

        assert config.bins == 100
        assert config.bounded is True
        assert config.lower_bound == 0
        assert config.upper_bound == 100
        assert config.sample_fraction == 0.1
        assert config.lazy_metrics is True

    def test_builder_defaults(self):
        """Test builder with no customizations uses dataclass defaults."""
        config = FitterConfigBuilder().build()

        # Should match dataclass defaults
        default = FitterConfig()
        assert config.bins == default.bins
        assert config.use_rice_rule == default.use_rice_rule
        assert config.enable_sampling == default.enable_sampling
        assert config.bounded == default.bounded


class TestFitterWithConfig:
    """Test fitters work correctly with FitterConfig."""

    def test_fit_with_config_continuous(self):
        """Test continuous fitter accepts config parameter."""
        import numpy as np

        from spark_bestfit import DistributionFitter, LocalBackend

        # Create fitter with LocalBackend
        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)

        # Create config
        config = FitterConfigBuilder().with_max_distributions(2).with_lazy_metrics().build()

        # Generate sample data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        import pandas as pd

        df = pd.DataFrame({"value": data})

        # Fit with config
        results = fitter.fit(df, column="value", config=config)

        # Should have results (max_distributions=2 means at most 2 fitted)
        best = results.best(n=10, metric="aic")  # Request more than max
        assert len(best) <= 2

    def test_fit_with_config_discrete(self):
        """Test discrete fitter accepts config parameter."""
        import numpy as np

        from spark_bestfit import DiscreteDistributionFitter, LocalBackend

        # Create fitter with LocalBackend
        backend = LocalBackend()
        fitter = DiscreteDistributionFitter(backend=backend)

        # Create config
        config = FitterConfigBuilder().with_max_distributions(2).with_lazy_metrics().build()

        # Generate sample data
        np.random.seed(42)
        data = np.random.poisson(5, 100)
        import pandas as pd

        df = pd.DataFrame({"counts": data})

        # Fit with config
        results = fitter.fit(df, column="counts", config=config)

        # Should have results (max_distributions=2 means at most 2 fitted)
        best = results.best(n=10, metric="aic")  # Request more than max
        assert len(best) <= 2

    def test_fit_backward_compat(self):
        """Test backward compatibility - fit still works without config."""
        import numpy as np

        from spark_bestfit import DistributionFitter, LocalBackend

        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)

        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        import pandas as pd

        df = pd.DataFrame({"value": data})

        # Fit with individual parameters (old API)
        results = fitter.fit(df, column="value", max_distributions=2, lazy_metrics=True)

        best = results.best(n=10, metric="aic")
        assert len(best) <= 2

    def test_config_overrides_params(self):
        """Test that config takes precedence over individual params."""
        import numpy as np

        from spark_bestfit import DistributionFitter, LocalBackend

        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)

        # Config says 2 distributions
        config = FitterConfigBuilder().with_max_distributions(2).build()

        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        import pandas as pd

        df = pd.DataFrame({"value": data})

        # Pass config AND max_distributions param - config should win
        results = fitter.fit(df, column="value", config=config, max_distributions=10)

        # Config's max_distributions=2 should apply
        best = results.best(n=10, metric="aic")
        assert len(best) <= 2

    def test_progress_callback_override(self):
        """Test that progress_callback can override config's callback."""
        import numpy as np

        from spark_bestfit import DistributionFitter, LocalBackend

        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)

        config_calls = []
        param_calls = []

        def config_callback(c, t, p):
            config_calls.append(c)

        def param_callback(c, t, p):
            param_calls.append(c)

        # Config has its own callback
        config = (
            FitterConfigBuilder().with_max_distributions(2).build().with_progress_callback(config_callback)
        )

        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        import pandas as pd

        df = pd.DataFrame({"value": data})

        # Pass progress_callback param - should override config's callback
        fitter.fit(df, column="value", config=config, progress_callback=param_callback)

        # Only param_callback should have been called
        assert len(config_calls) == 0
        assert len(param_calls) > 0

    def test_config_reuse_across_fits(self):
        """Test that same config can be reused for multiple fits."""
        import numpy as np

        from spark_bestfit import DistributionFitter, LocalBackend

        backend = LocalBackend()
        fitter = DistributionFitter(backend=backend)

        # Create config once
        config = FitterConfigBuilder().with_max_distributions(2).with_lazy_metrics().build()

        np.random.seed(42)
        import pandas as pd

        # Fit multiple columns with same config
        df = pd.DataFrame(
            {
                "value1": np.random.normal(0, 1, 100),
                "value2": np.random.exponential(1, 100),
            }
        )

        results1 = fitter.fit(df, column="value1", config=config)
        results2 = fitter.fit(df, column="value2", config=config)

        # Both should work
        assert len(results1.best(n=10, metric="aic")) <= 2
        assert len(results2.best(n=10, metric="aic")) <= 2
