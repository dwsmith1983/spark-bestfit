"""Tests for heavy-tail distribution detection (#64)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from spark_bestfit import DistributionFitter, LocalBackend
from spark_bestfit.fitting import HEAVY_TAIL_DISTRIBUTIONS, compute_data_stats, detect_heavy_tail


class TestDetectHeavyTail:
    """Tests for detect_heavy_tail function."""

    def test_normal_data_not_heavy_tailed(self):
        """Normal data should not be detected as heavy-tailed."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = detect_heavy_tail(data)

        assert result["is_heavy_tailed"] is False
        assert abs(result["kurtosis"]) < 1.0  # Near 0 for normal
        assert len(result["indicators"]) == 0

    def test_cauchy_data_is_heavy_tailed(self):
        """Cauchy data should be detected as heavy-tailed."""
        np.random.seed(42)
        data = np.random.standard_cauchy(1000)

        result = detect_heavy_tail(data)

        assert result["is_heavy_tailed"] is True
        assert result["kurtosis"] > 6.0  # Very high kurtosis
        assert len(result["indicators"]) > 0
        assert any("kurtosis" in ind for ind in result["indicators"])

    def test_pareto_data_is_heavy_tailed(self):
        """Pareto data should be detected as heavy-tailed."""
        np.random.seed(42)
        data = (np.random.pareto(1.5, 1000) + 1) * 10

        result = detect_heavy_tail(data)

        assert result["is_heavy_tailed"] is True
        assert result["kurtosis"] > 6.0

    def test_uniform_data_not_heavy_tailed(self):
        """Uniform data should not be detected as heavy-tailed."""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 1000)

        result = detect_heavy_tail(data)

        assert result["is_heavy_tailed"] is False
        # Uniform has negative excess kurtosis (-1.2)
        assert result["kurtosis"] < 0

    def test_extreme_ratio_detection(self):
        """Test detection of extreme values."""
        np.random.seed(42)
        # Create data with extreme outliers
        data = np.random.normal(0, 1, 1000)
        # Add extreme outliers
        data = np.append(data, [100, 200, 300])

        result = detect_heavy_tail(data)

        assert result["is_heavy_tailed"] is True
        assert result["extreme_ratio"] > 3.0
        assert any("extreme" in ind for ind in result["indicators"])

    def test_custom_kurtosis_threshold(self):
        """Test custom kurtosis threshold."""
        np.random.seed(42)
        # t-distribution with 5 degrees of freedom has excess kurtosis ~6
        from scipy.stats import t

        data = t.rvs(df=5, size=1000, random_state=42)

        # Default threshold (6.0) - borderline
        result_default = detect_heavy_tail(data, kurtosis_threshold=6.0)

        # Lower threshold (3.0) - should detect
        result_low = detect_heavy_tail(data, kurtosis_threshold=3.0)

        assert result_low["is_heavy_tailed"] is True

    def test_small_sample_size(self):
        """Test with very small sample size."""
        data = np.array([1.0, 2.0, 3.0])  # Less than 10 samples

        result = detect_heavy_tail(data)

        # Should return safe defaults for small samples
        assert result["is_heavy_tailed"] is False
        assert len(result["indicators"]) == 0

    def test_handles_nan_and_inf(self):
        """Test that NaN and inf values are handled."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        data = np.append(data, [np.nan, np.inf, -np.inf])

        result = detect_heavy_tail(data)

        # Should not crash and should analyze finite values only
        assert "kurtosis" in result
        assert np.isfinite(result["kurtosis"])


class TestComputeDataStats:
    """Tests for compute_data_stats including kurtosis."""

    def test_includes_kurtosis_and_skewness(self):
        """Stats should include kurtosis and skewness."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        stats = compute_data_stats(data)

        assert "data_kurtosis" in stats
        assert "data_skewness" in stats
        assert isinstance(stats["data_kurtosis"], float)
        assert isinstance(stats["data_skewness"], float)

    def test_normal_data_stats(self):
        """Normal data should have near-zero excess kurtosis and skewness."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)

        stats = compute_data_stats(data)

        # Excess kurtosis ~0 for normal
        assert abs(stats["data_kurtosis"]) < 0.5
        # Skewness ~0 for normal
        assert abs(stats["data_skewness"]) < 0.2

    def test_exponential_data_stats(self):
        """Exponential data should have positive skewness and kurtosis."""
        np.random.seed(42)
        data = np.random.exponential(1, 10000)

        stats = compute_data_stats(data)

        # Exponential has excess kurtosis = 6
        assert stats["data_kurtosis"] > 3
        # Exponential has skewness = 2
        assert stats["data_skewness"] > 1


class TestHeavyTailDistributionsConstant:
    """Tests for HEAVY_TAIL_DISTRIBUTIONS constant."""

    def test_constant_exists_and_is_frozenset(self):
        """Constant should exist and be immutable."""
        assert isinstance(HEAVY_TAIL_DISTRIBUTIONS, frozenset)
        assert len(HEAVY_TAIL_DISTRIBUTIONS) > 0

    def test_known_heavy_tail_distributions_included(self):
        """Known heavy-tail distributions should be in the set."""
        expected = {"cauchy", "pareto", "t", "levy", "burr"}
        for dist in expected:
            assert dist in HEAVY_TAIL_DISTRIBUTIONS, f"{dist} should be in HEAVY_TAIL_DISTRIBUTIONS"

    def test_normal_not_in_heavy_tail(self):
        """Normal distribution should not be in heavy-tail set."""
        assert "norm" not in HEAVY_TAIL_DISTRIBUTIONS


class TestHeavyTailWarning:
    """Tests for heavy-tail warning during fitting."""

    @pytest.fixture
    def local_backend(self):
        """Create LocalBackend for testing."""
        return LocalBackend(max_workers=2)

    def test_warning_emitted_for_heavy_tailed_data(self, local_backend):
        """Fitting heavy-tailed data should emit a warning."""
        np.random.seed(42)
        # Create clearly heavy-tailed data
        data = np.random.standard_cauchy(500)
        # Clip extreme values to avoid fitting failures
        data = np.clip(data, -100, 100)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(df, column="value", max_distributions=3)

            # Filter for UserWarning about heavy-tail
            heavy_tail_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "heavy-tail" in str(x.message).lower()
            ]

            assert len(heavy_tail_warnings) >= 1
            assert "kurtosis" in str(heavy_tail_warnings[0].message).lower()

    def test_no_warning_for_normal_data(self, local_backend):
        """Fitting normal data should not emit heavy-tail warning."""
        np.random.seed(42)
        data = np.random.normal(50, 10, 500)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitter.fit(df, column="value", max_distributions=3)

            # Filter for heavy-tail warnings
            heavy_tail_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "heavy-tail" in str(x.message).lower()
            ]

            assert len(heavy_tail_warnings) == 0
