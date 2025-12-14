"""Tests for plotting module."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from spark_bestfit.plotting import plot_comparison, plot_distribution
from spark_bestfit.results import DistributionFitResult


@pytest.fixture
def sample_histogram():
    """Create sample histogram data."""
    np.random.seed(42)
    data = np.random.normal(50, 10, 10000)
    y_hist, x_edges = np.histogram(data, bins=50, density=True)
    x_hist = (x_edges[:-1] + x_edges[1:]) / 2
    return y_hist, x_hist


@pytest.fixture
def normal_result():
    """Create a sample normal distribution result."""
    return DistributionFitResult(
        distribution="norm",
        parameters=[50.0, 10.0],  # loc, scale
        sse=0.005,
        aic=1500.0,
        bic=1520.0,
    )


@pytest.fixture
def gamma_result():
    """Create a sample gamma distribution result."""
    return DistributionFitResult(
        distribution="gamma",
        parameters=[2.0, 0.0, 2.0],  # shape, loc, scale
        sse=0.003,
        aic=1400.0,
        bic=1430.0,
    )


@pytest.fixture
def expon_result():
    """Create a sample exponential distribution result."""
    return DistributionFitResult(
        distribution="expon",
        parameters=[0.0, 5.0],  # loc, scale
        sse=0.008,
        aic=1600.0,
        bic=1615.0,
    )


class TestPlotDistribution:
    """Tests for plot_distribution function."""

    def test_basic_plot(self, normal_result, sample_histogram):
        """Test basic distribution plotting creates valid figure with expected elements."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(normal_result, y_hist, x_hist)

        # Verify figure and axes are valid matplotlib objects
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Verify plot has expected elements
        assert len(ax.patches) > 0  # Histogram bars
        assert len(ax.lines) >= 1  # PDF line

        # Verify legend exists
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        # Legend should have at least one entry (histogram or PDF line)
        assert len(legend_texts) > 0

        plt.close(fig)

    def test_plot_with_title(self, normal_result, sample_histogram):
        """Test plotting with custom title."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(normal_result, y_hist, x_hist, title="Test Title")

        assert "Test Title" in ax.get_title()
        plt.close(fig)

    def test_plot_with_custom_labels(self, normal_result, sample_histogram):
        """Test plotting with custom axis labels."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(
            normal_result,
            y_hist,
            x_hist,
            xlabel="Custom X",
            ylabel="Custom Y",
        )

        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"
        plt.close(fig)

    def test_plot_without_histogram(self, normal_result, sample_histogram):
        """Test plotting without showing histogram."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(normal_result, y_hist, x_hist, show_histogram=False)

        # Should only have PDF line, no bars
        assert fig is not None
        plt.close(fig)

    def test_plot_gamma_distribution(self, gamma_result, sample_histogram):
        """Test plotting gamma distribution (has shape params)."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(gamma_result, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)

    def test_plot_without_aic_bic(self, sample_histogram):
        """Test plotting result without AIC/BIC."""
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005, aic=None, bic=None)
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(result, y_hist, x_hist)

        # Should not show AIC/BIC in title
        assert "AIC" not in ax.get_title()
        plt.close(fig)

    def test_plot_custom_parameters(self, normal_result, sample_histogram):
        """Test plotting with custom parameters."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(
            normal_result, y_hist, x_hist, figsize=(16, 10), dpi=300, histogram_alpha=0.7, pdf_linewidth=3
        )

        assert fig.get_figwidth() == 16
        assert fig.get_figheight() == 10
        plt.close(fig)

    def test_plot_handles_ppf_failure(self, sample_histogram):
        """Test that plotting handles ppf failure gracefully."""
        # Create result with parameters that might cause ppf issues
        result = DistributionFitResult(
            distribution="beta",
            parameters=[0.5, 0.5, 0.0, 1.0],  # shape params that work
            sse=0.01,
        )
        y_hist, x_hist = sample_histogram

        # Should not raise even if ppf has issues
        fig, ax = plot_distribution(result, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)

    def test_plot_single_bin_histogram(self, normal_result):
        """Test plotting with single bin histogram (edge case)."""
        y_hist = np.array([1.0])
        x_hist = np.array([50.0])

        fig, ax = plot_distribution(normal_result, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)


class TestPlotComparison:
    """Tests for plot_comparison function."""

    def test_comparison_multiple_distributions(self, normal_result, gamma_result, expon_result, sample_histogram):
        """Test comparing multiple distributions."""
        y_hist, x_hist = sample_histogram
        results = [normal_result, gamma_result, expon_result]

        fig, ax = plot_comparison(results, y_hist, x_hist)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_comparison_single_distribution(self, normal_result, sample_histogram):
        """Test comparison with single distribution."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_comparison([normal_result], y_hist, x_hist)

        assert fig is not None
        plt.close(fig)

    def test_comparison_empty_results_raises(self, sample_histogram):
        """Test that empty results raises ValueError."""
        y_hist, x_hist = sample_histogram

        with pytest.raises(ValueError, match="Must provide at least one result"):
            plot_comparison([], y_hist, x_hist)

    def test_comparison_with_title(self, normal_result, gamma_result, sample_histogram):
        """Test comparison with custom title."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_comparison(
            [normal_result, gamma_result],
            y_hist,
            x_hist,
            title="Custom Comparison",
        )

        assert "Custom Comparison" in ax.get_title()
        plt.close(fig)

    def test_comparison_without_histogram(self, normal_result, gamma_result, sample_histogram):
        """Test comparison without showing histogram."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_comparison([normal_result, gamma_result], y_hist, x_hist, show_histogram=False)

        assert fig is not None
        plt.close(fig)

    def test_comparison_legend_entries(self, normal_result, gamma_result, expon_result, sample_histogram):
        """Test that legend has entries for all distributions."""
        y_hist, x_hist = sample_histogram
        results = [normal_result, gamma_result, expon_result]

        fig, ax = plot_comparison(results, y_hist, x_hist)

        # Check legend has expected entries
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]

        assert any("norm" in text for text in legend_texts)
        assert any("gamma" in text for text in legend_texts)
        assert any("expon" in text for text in legend_texts)
        plt.close(fig)

    def test_comparison_handles_ppf_failure(self, sample_histogram):
        """Test comparison handles ppf failure gracefully."""
        # Create results with various distributions
        results = [
            DistributionFitResult("norm", [50.0, 10.0], 0.01),
            DistributionFitResult("beta", [0.5, 0.5, 0.0, 1.0], 0.02),
        ]
        y_hist, x_hist = sample_histogram

        fig, ax = plot_comparison(results, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)


class TestPlotSaving:
    """Tests for plot saving functionality."""

    def test_save_plot(self, normal_result, sample_histogram, tmp_path):
        """Test saving plot to file."""
        y_hist, x_hist = sample_histogram
        save_path = str(tmp_path / "test_plot.png")

        fig, ax = plot_distribution(normal_result, y_hist, x_hist, save_path=save_path)

        assert (tmp_path / "test_plot.png").exists()
        plt.close(fig)

    def test_save_comparison_plot(self, normal_result, gamma_result, sample_histogram, tmp_path):
        """Test saving comparison plot to file."""
        y_hist, x_hist = sample_histogram
        save_path = str(tmp_path / "comparison.png")

        fig, ax = plot_comparison(
            [normal_result, gamma_result],
            y_hist,
            x_hist,
            save_path=save_path,
        )

        assert (tmp_path / "comparison.png").exists()
        plt.close(fig)


class TestPlotEdgeCases:
    """Tests for edge cases in plotting."""

    def test_plot_with_inf_in_histogram(self, normal_result):
        """Test plotting handles inf values in histogram."""
        y_hist = np.array([0.1, 0.2, np.inf, 0.2, 0.1])
        x_hist = np.array([40, 45, 50, 55, 60])

        # Should handle gracefully (matplotlib will warn but not crash)
        fig, ax = plot_distribution(normal_result, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)

    def test_plot_with_nan_in_histogram(self, normal_result):
        """Test plotting handles NaN values in histogram."""
        y_hist = np.array([0.1, 0.2, np.nan, 0.2, 0.1])
        x_hist = np.array([40, 45, 50, 55, 60])

        # Should handle gracefully
        fig, ax = plot_distribution(normal_result, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)

    def test_plot_very_small_histogram(self, normal_result):
        """Test plotting with very small histogram."""
        y_hist = np.array([0.5, 0.5])
        x_hist = np.array([49, 51])

        fig, ax = plot_distribution(normal_result, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)

    def test_comparison_many_distributions(self, sample_histogram):
        """Test comparison with many distributions."""
        y_hist, x_hist = sample_histogram

        results = [DistributionFitResult("norm", [50.0 + i, 10.0], 0.01 + i * 0.001) for i in range(10)]

        fig, ax = plot_comparison(results, y_hist, x_hist)

        assert fig is not None
        plt.close(fig)
