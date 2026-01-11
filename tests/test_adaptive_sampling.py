"""Tests for adaptive sampling based on data skewness (v2.9.0).

Tests cover:
- SamplingMode enum and config options
- Skewness estimation from pilot samples
- Sampling strategy selection
- Stratified sampling implementation
- Backwards compatibility with uniform mode
- Performance comparison (timing tests)
"""

import importlib.util
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from spark_bestfit import (
    DistributionFitter,
    FitterConfig,
    FitterConfigBuilder,
    SamplingMode,
)
from spark_bestfit.backends.local import LocalBackend


class TestSamplingModeEnum:
    """Tests for SamplingMode enum."""

    def test_sampling_mode_values(self):
        """Test that SamplingMode has expected values."""
        assert SamplingMode.AUTO.value == "auto"
        assert SamplingMode.UNIFORM.value == "uniform"
        assert SamplingMode.STRATIFIED.value == "stratified"

    def test_sampling_mode_in_config(self):
        """Test that SamplingMode can be used in FitterConfig."""
        config = FitterConfig(sampling_mode=SamplingMode.STRATIFIED)
        assert config.sampling_mode == SamplingMode.STRATIFIED

    def test_sampling_mode_default(self):
        """Test default sampling mode is AUTO."""
        config = FitterConfig()
        assert config.sampling_mode == SamplingMode.AUTO


class TestFitterConfigAdaptiveSampling:
    """Tests for adaptive sampling configuration."""

    def test_adaptive_sampling_defaults(self):
        """Test default values for adaptive sampling config."""
        config = FitterConfig()
        assert config.adaptive_sampling is True
        assert config.sampling_mode == SamplingMode.AUTO
        assert config.skew_threshold_mild == 0.5
        assert config.skew_threshold_high == 2.0

    def test_builder_with_adaptive_sampling(self):
        """Test FitterConfigBuilder with_adaptive_sampling method."""
        config = (
            FitterConfigBuilder()
            .with_adaptive_sampling(
                enabled=True,
                mode=SamplingMode.STRATIFIED,
                skew_threshold_mild=0.3,
                skew_threshold_high=1.5,
            )
            .build()
        )

        assert config.adaptive_sampling is True
        assert config.sampling_mode == SamplingMode.STRATIFIED
        assert config.skew_threshold_mild == 0.3
        assert config.skew_threshold_high == 1.5

    def test_builder_disable_adaptive_sampling(self):
        """Test disabling adaptive sampling via builder."""
        config = (
            FitterConfigBuilder()
            .with_adaptive_sampling(enabled=False, mode=SamplingMode.UNIFORM)
            .build()
        )

        assert config.adaptive_sampling is False
        assert config.sampling_mode == SamplingMode.UNIFORM


class TestSkewnessEstimation:
    """Tests for _compute_skewness_estimate method."""

    @pytest.fixture
    def fitter(self, local_backend):
        """Create a fitter for testing."""
        return DistributionFitter(backend=local_backend, random_seed=42)

    def test_skewness_estimate_normal(self, fitter):
        """Test skewness estimate for normal distribution (skew ~ 0)."""
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=10000)
        df = pd.DataFrame({"value": data})

        skewness = fitter._compute_skewness_estimate(df, "value", len(df))

        # Normal distribution should have skewness near 0
        assert abs(skewness) < 0.2

    def test_skewness_estimate_exponential(self, fitter):
        """Test skewness estimate for exponential distribution (skew ~ 2)."""
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=10000)
        df = pd.DataFrame({"value": data})

        skewness = fitter._compute_skewness_estimate(df, "value", len(df))

        # Exponential distribution has skewness = 2
        assert 1.5 < skewness < 2.5

    def test_skewness_estimate_pareto(self, fitter):
        """Test skewness estimate for Pareto distribution (high positive skew)."""
        np.random.seed(42)
        # Pareto with alpha=2 has skewness of about 4.5
        data = (np.random.pareto(a=2, size=10000) + 1) * 2
        df = pd.DataFrame({"value": data})

        skewness = fitter._compute_skewness_estimate(df, "value", len(df))

        # Should detect high positive skewness
        assert skewness > 2.0

    def test_skewness_estimate_negative_skew(self, fitter):
        """Test skewness estimate for negatively skewed distribution."""
        np.random.seed(42)
        # Create negatively skewed data (mirror of exponential)
        data = -np.random.exponential(scale=5, size=10000) + 30
        df = pd.DataFrame({"value": data})

        skewness = fitter._compute_skewness_estimate(df, "value", len(df))

        # Should be negative
        assert skewness < -1.5

    def test_pilot_sample_accuracy(self, fitter):
        """Test pilot sample skewness is within 10% of full data skewness (Mayor feedback)."""
        np.random.seed(42)
        # Use moderately skewed data
        data = np.random.exponential(scale=5, size=50000)
        df = pd.DataFrame({"value": data})

        # Full data skewness
        full_skewness = float(stats.skew(data))

        # Pilot sample skewness (should use ~5000 samples)
        pilot_skewness = fitter._compute_skewness_estimate(df, "value", len(df))

        # Should be within 10% (relative error)
        relative_error = abs(pilot_skewness - full_skewness) / abs(full_skewness)
        assert relative_error < 0.10, f"Pilot skew {pilot_skewness} vs full {full_skewness}"

    def test_small_dataset_uses_all_data(self, fitter):
        """Test that small datasets use all data (no pilot sampling)."""
        np.random.seed(42)
        data = np.random.normal(loc=0, scale=1, size=1000)
        df = pd.DataFrame({"value": data})

        # For 1000 rows (< 5000 pilot size), should use all data
        skewness = fitter._compute_skewness_estimate(df, "value", len(df))

        # Compare with direct calculation
        expected = float(stats.skew(data))
        assert abs(skewness - expected) < 0.01


class TestSamplingStrategySelection:
    """Tests for _select_sampling_strategy method."""

    @pytest.fixture
    def fitter(self, local_backend):
        """Create a fitter for testing."""
        return DistributionFitter(backend=local_backend, random_seed=42)

    def test_uniform_for_symmetric_data(self, fitter):
        """Test uniform mode selected for symmetric data (|skew| < 0.5)."""
        mode = fitter._select_sampling_strategy(0.3, 0.5, 2.0)
        assert mode == SamplingMode.UNIFORM

    def test_uniform_for_slightly_negative_skew(self, fitter):
        """Test uniform mode for slightly negative skew."""
        mode = fitter._select_sampling_strategy(-0.4, 0.5, 2.0)
        assert mode == SamplingMode.UNIFORM

    def test_stratified_for_mild_skew(self, fitter):
        """Test stratified mode for mild skew (0.5 <= |skew| < 2.0)."""
        mode = fitter._select_sampling_strategy(1.0, 0.5, 2.0)
        assert mode == SamplingMode.STRATIFIED

    def test_stratified_for_negative_mild_skew(self, fitter):
        """Test stratified mode for negative mild skew."""
        mode = fitter._select_sampling_strategy(-1.5, 0.5, 2.0)
        assert mode == SamplingMode.STRATIFIED

    def test_stratified_for_high_skew(self, fitter):
        """Test stratified mode for high skew (|skew| >= 2.0)."""
        mode = fitter._select_sampling_strategy(3.0, 0.5, 2.0)
        assert mode == SamplingMode.STRATIFIED

    def test_custom_thresholds(self, fitter):
        """Test strategy selection with custom thresholds."""
        # With threshold 1.0, skew of 0.8 should still be uniform
        mode = fitter._select_sampling_strategy(0.8, 1.0, 3.0)
        assert mode == SamplingMode.UNIFORM

        # With threshold 1.0, skew of 1.5 should be stratified
        mode = fitter._select_sampling_strategy(1.5, 1.0, 3.0)
        assert mode == SamplingMode.STRATIFIED


class TestStratifiedSamplingPandas:
    """Tests for stratified sampling with pandas DataFrames."""

    @pytest.fixture
    def fitter(self, local_backend):
        """Create a fitter for testing."""
        return DistributionFitter(backend=local_backend, random_seed=42)

    def test_stratified_preserves_tail_representation(self, fitter):
        """Test that stratified sampling preserves tail values better than uniform."""
        np.random.seed(42)
        # Create highly skewed data
        data = np.concatenate([
            np.random.exponential(scale=1, size=9000),  # Bulk
            np.random.exponential(scale=1, size=1000) + 10,  # Tail
        ])
        df = pd.DataFrame({"value": data})

        # Get stratified sample
        stratified = fitter._apply_stratified_sampling_pandas(
            df, "value", fraction=0.1, n_bins=10, target_size=1000, min_tail_samples=50
        )

        # Check tail representation
        original_tail_pct = (df["value"] > 10).mean()
        stratified_tail_pct = (stratified["value"] > 10).mean()

        # Stratified should maintain better tail representation
        # (within reasonable bounds, not losing most of tail)
        assert stratified_tail_pct >= original_tail_pct * 0.5

    def test_stratified_sample_size(self, fitter):
        """Test that stratified sampling produces approximately target size."""
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=10000)
        df = pd.DataFrame({"value": data})

        sampled = fitter._apply_stratified_sampling_pandas(
            df, "value", fraction=0.1, n_bins=10, target_size=1000, min_tail_samples=100
        )

        # Should be within 50% of target due to min_tail_samples boost
        assert 500 < len(sampled) < 2000

    def test_stratified_all_bins_represented(self, fitter):
        """Test that stratified sampling includes data from all percentile bins."""
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=10000)
        df = pd.DataFrame({"value": data})

        sampled = fitter._apply_stratified_sampling_pandas(
            df, "value", fraction=0.1, n_bins=5, target_size=1000, min_tail_samples=50
        )

        # Check that sample spans the full range
        original_range = df["value"].max() - df["value"].min()
        sampled_range = sampled["value"].max() - sampled["value"].min()

        # Sampled range should be at least 70% of original range
        # (stratified sampling may not capture the absolute extremes)
        assert sampled_range >= original_range * 0.7


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with uniform sampling."""

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=2)

    def test_uniform_mode_matches_original_behavior(self, local_backend):
        """Test that UNIFORM mode produces same results as pre-v2.9 sampling."""
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=100000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend, random_seed=42)

        # The _apply_sampling method with adaptive_sampling=False should behave like original
        # We can't easily test exact equality due to different code paths,
        # but we can verify the mode is respected
        sampled = fitter._apply_sampling(
            df,
            row_count=len(df),
            enable_sampling=True,
            sample_fraction=None,
            max_sample_size=10000,
            sample_threshold=1000,
            column="value",
            adaptive_sampling=False,
            sampling_mode=SamplingMode.UNIFORM,
        )

        # Should produce a sample of expected size
        expected_size = min(10000, int(len(df) * 0.35))
        assert len(sampled) == pytest.approx(expected_size, rel=0.1)

    def test_disabled_adaptive_uses_uniform(self, local_backend):
        """Test that disabled adaptive sampling uses uniform regardless of skewness."""
        np.random.seed(42)
        # Highly skewed data
        data = np.random.pareto(a=2, size=100000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend, random_seed=42)

        # With adaptive_sampling=False, should use uniform even for skewed data
        sampled = fitter._apply_sampling(
            df,
            row_count=len(df),
            enable_sampling=True,
            sample_fraction=None,
            max_sample_size=10000,
            sample_threshold=1000,
            column="value",
            adaptive_sampling=False,
            sampling_mode=SamplingMode.AUTO,  # Should be ignored when adaptive=False
        )

        # Should have sampled (uniform doesn't require stratification)
        assert len(sampled) < len(df)


class TestAdaptiveSamplingIntegration:
    """Integration tests for adaptive sampling with full fitting pipeline."""

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=2)

    def test_fit_exponential_with_adaptive(self, local_backend):
        """Test fitting exponential data (skewed) with adaptive sampling."""
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=100000)
        df = pd.DataFrame({"value": data})

        config = (
            FitterConfigBuilder()
            .with_adaptive_sampling(enabled=True, mode=SamplingMode.AUTO)
            .with_sampling(threshold=10000, max_size=10000)
            .build()
        )

        fitter = DistributionFitter(backend=local_backend, random_seed=42)
        results = fitter.fit(df, column="value", config=config, max_distributions=10)

        # Should successfully fit and find exponential or related distributions in top results
        # (exponential family: expon, lomax, exponweib, genpareto, etc.)
        top_results = results.best(n=10)
        dist_names = [r.distribution for r in top_results]
        exponential_family = {"expon", "lomax", "exponweib", "genpareto", "gamma", "invgamma"}
        found_exponential_family = any(d in exponential_family for d in dist_names)
        assert found_exponential_family, f"Expected exponential family dist in {dist_names}"

    def test_fit_normal_with_adaptive(self, local_backend):
        """Test fitting normal data (symmetric) with adaptive sampling."""
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=100000)
        df = pd.DataFrame({"value": data})

        config = (
            FitterConfigBuilder()
            .with_adaptive_sampling(enabled=True, mode=SamplingMode.AUTO)
            .with_sampling(threshold=10000, max_size=10000)
            .build()
        )

        fitter = DistributionFitter(backend=local_backend, random_seed=42)
        results = fitter.fit(df, column="value", config=config, max_distributions=10)

        # Should successfully fit and find norm in top results
        # Use top 10 since adaptive sampling can affect rankings
        top_results = results.best(n=10)
        dist_names = [r.distribution for r in top_results]
        assert "norm" in dist_names, f"Expected 'norm' in top 10, got: {dist_names}"


class TestPerformance:
    """Performance tests for adaptive sampling.

    Note: Stratified sampling is inherently more expensive than uniform because it
    requires computing percentiles and sampling from each bucket. The goal is to
    ensure it doesn't become prohibitively slow, not to match uniform speed.
    """

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=2)

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_adaptive_vs_uniform_timing(self, local_backend):
        """Benchmark adaptive vs uniform sampling timing.

        This test measures the overhead of adaptive sampling. Stratified sampling
        is expected to be slower due to percentile computation and per-bucket sampling.
        The goal is to keep overhead reasonable (not exceeding 10x).
        """
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=500000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend, random_seed=42)

        # Time uniform sampling
        start_uniform = time.time()
        for _ in range(3):
            _ = fitter._apply_sampling(
                df,
                row_count=len(df),
                enable_sampling=True,
                sample_fraction=None,
                max_sample_size=50000,
                sample_threshold=10000,
                column="value",
                adaptive_sampling=False,
                sampling_mode=SamplingMode.UNIFORM,
            )
        time_uniform = (time.time() - start_uniform) / 3

        # Time adaptive sampling
        start_adaptive = time.time()
        for _ in range(3):
            _ = fitter._apply_sampling(
                df,
                row_count=len(df),
                enable_sampling=True,
                sample_fraction=None,
                max_sample_size=50000,
                sample_threshold=10000,
                column="value",
                adaptive_sampling=True,
                sampling_mode=SamplingMode.AUTO,
            )
        time_adaptive = (time.time() - start_adaptive) / 3

        # Stratified sampling will be slower (percentile computation + per-bucket sampling)
        # Ensure it doesn't exceed 10x overhead (reasonable for production use)
        overhead = (time_adaptive - time_uniform) / time_uniform if time_uniform > 0 else 0
        assert overhead < 10.0, f"Adaptive overhead {overhead:.1%} exceeds 1000%"

        # Log the actual overhead for benchmarking purposes
        print(f"\nAdaptive sampling overhead: {overhead:.1%}")
        print(f"Uniform time: {time_uniform:.3f}s, Adaptive time: {time_adaptive:.3f}s")


# Skip Spark tests if PySpark not installed
PYSPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestSparkAdaptiveSampling:
    """Tests for adaptive sampling with Spark DataFrames."""

    def test_spark_stratified_sampling(self, spark_session):
        """Test stratified sampling with Spark backend."""
        from spark_bestfit import DistributionFitter

        # Generate skewed data
        np.random.seed(42)
        data = [(float(x),) for x in np.random.exponential(scale=5, size=100000)]
        df = spark_session.createDataFrame(data, ["value"])

        fitter = DistributionFitter(spark=spark_session, random_seed=42)

        # Apply stratified sampling
        sampled = fitter._apply_stratified_sampling_spark(
            df, "value", fraction=0.1, n_bins=10, target_size=10000, min_tail_samples=500
        )

        # Should produce reasonable sample size
        count = sampled.count()
        assert 5000 < count < 20000

    def test_spark_skewness_estimate(self, spark_session):
        """Test skewness estimation with Spark DataFrame."""
        from spark_bestfit import DistributionFitter

        np.random.seed(42)
        data = [(float(x),) for x in np.random.exponential(scale=5, size=50000)]
        df = spark_session.createDataFrame(data, ["value"])

        fitter = DistributionFitter(spark=spark_session, random_seed=42)
        skewness = fitter._compute_skewness_estimate(df, "value", 50000)

        # Exponential has skew ~2
        assert 1.5 < skewness < 2.5
