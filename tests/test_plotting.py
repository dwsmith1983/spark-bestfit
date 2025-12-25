"""Tests for plotting module."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for tests
import matplotlib.pyplot as plt

from spark_bestfit.results import DistributionFitResult
from spark_bestfit.plotting import plot_comparison, plot_distribution, plot_qq, plot_pp

# Fixtures are now in conftest.py: normal_result, gamma_result, expon_result,
# result_with_ks, sample_histogram, sample_data


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

        # Should only have PDF line, no histogram bars
        assert fig is not None
        assert len(ax.patches) == 0, "Histogram bars should not be present when show_histogram=False"
        assert len(ax.lines) >= 1, "PDF line should be present"
        plt.close(fig)

    def test_plot_gamma_distribution(self, gamma_result, sample_histogram):
        """Test plotting gamma distribution (has shape params)."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_distribution(gamma_result, y_hist, x_hist)

        assert fig is not None
        # Verify title contains distribution name
        assert "gamma" in ax.get_title().lower()
        # Verify plot has both histogram and PDF line
        assert len(ax.patches) > 0, "Should have histogram bars"
        assert len(ax.lines) >= 1, "Should have PDF line"
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

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Should have histogram bars and 3 PDF lines (one per distribution)
        assert len(ax.patches) > 0, "Should have histogram bars"
        assert len(ax.lines) >= 3, "Should have one line per distribution"

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

        assert isinstance(fig, plt.Figure)
        assert len(ax.patches) == 0, "Should have no histogram bars when show_histogram=False"
        assert len(ax.lines) >= 2, "Should still have PDF lines"

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

    def test_comparison_custom_axis_labels(self, normal_result, gamma_result, sample_histogram):
        """Test comparison with custom axis labels."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_comparison(
            [normal_result, gamma_result],
            y_hist,
            x_hist,
            xlabel="Custom X Label",
            ylabel="Custom Y Label",
        )

        assert ax.get_xlabel() == "Custom X Label"
        assert ax.get_ylabel() == "Custom Y Label"
        plt.close(fig)

    def test_comparison_custom_figsize(self, normal_result, sample_histogram):
        """Test comparison with custom figure size."""
        y_hist, x_hist = sample_histogram

        fig, ax = plot_comparison(
            [normal_result],
            y_hist,
            x_hist,
            figsize=(16, 12),
        )

        width, height = fig.get_size_inches()
        assert width == 16
        assert height == 12
        plt.close(fig)

    def test_comparison_save_pdf_format(self, normal_result, gamma_result, sample_histogram, tmp_path):
        """Test saving comparison plot as PDF."""
        y_hist, x_hist = sample_histogram
        save_path = str(tmp_path / "comparison.pdf")

        fig, ax = plot_comparison(
            [normal_result, gamma_result],
            y_hist,
            x_hist,
            save_path=save_path,
            save_format="pdf",
        )

        assert (tmp_path / "comparison.pdf").exists()
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

    def test_save_qq_plot(self, result_with_ks, sample_data, tmp_path):
        """Test saving Q-Q plot to file."""
        save_path = str(tmp_path / "qq_plot.png")
        fig, ax = plot_qq(result_with_ks, sample_data, save_path=save_path)
        assert (tmp_path / "qq_plot.png").exists()
        plt.close(fig)

    def test_save_pp_plot(self, result_with_ks, sample_data, tmp_path):
        """Test saving P-P plot to file."""
        save_path = str(tmp_path / "pp_plot.png")
        fig, ax = plot_pp(result_with_ks, sample_data, save_path=save_path)
        assert (tmp_path / "pp_plot.png").exists()
        plt.close(fig)


class TestPlotQQ:
    """Tests for plot_qq function."""

    def test_basic_qq_plot(self, result_with_ks, sample_data):
        """Test basic Q-Q plot creates valid figure with expected elements."""
        fig, ax = plot_qq(result_with_ks, sample_data)

        # Verify figure and axes are valid matplotlib objects
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Verify plot has scatter points (collections) and reference line
        assert len(ax.collections) >= 1  # Scatter plot
        assert len(ax.lines) >= 1  # Reference line

        # Verify legend exists
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_qq_plot_with_title(self, result_with_ks, sample_data):
        """Test Q-Q plot with custom title."""
        fig, ax = plot_qq(result_with_ks, sample_data, title="Test Q-Q Plot")

        assert "Test Q-Q Plot" in ax.get_title()
        plt.close(fig)

    def test_qq_plot_with_custom_labels(self, result_with_ks, sample_data):
        """Test Q-Q plot with custom axis labels."""
        fig, ax = plot_qq(
            result_with_ks,
            sample_data,
            xlabel="Custom Theoretical",
            ylabel="Custom Sample",
        )

        assert ax.get_xlabel() == "Custom Theoretical"
        assert ax.get_ylabel() == "Custom Sample"
        plt.close(fig)

    def test_qq_plot_shows_ks_statistic(self, result_with_ks, sample_data):
        """Test Q-Q plot shows KS statistic when available."""
        fig, ax = plot_qq(result_with_ks, sample_data)

        title = ax.get_title()
        assert "KS=" in title
        assert "p=" in title
        plt.close(fig)

    def test_qq_plot_without_ks_shows_sse(self, normal_result, sample_data):
        """Test Q-Q plot shows SSE when KS statistic not available."""
        fig, ax = plot_qq(normal_result, sample_data)

        title = ax.get_title()
        assert "SSE=" in title
        plt.close(fig)

    def test_qq_plot_gamma_distribution(self, gamma_result, sample_data):
        """Test Q-Q plot with gamma distribution (has shape params)."""
        fig, ax = plot_qq(gamma_result, sample_data)

        assert fig is not None
        assert "gamma" in ax.get_title()
        plt.close(fig)

    def test_qq_plot_custom_markers(self, result_with_ks, sample_data):
        """Test Q-Q plot with custom marker settings."""
        fig, ax = plot_qq(
            result_with_ks,
            sample_data,
            marker="s",
            marker_size=50,
            marker_alpha=0.8,
            marker_color="red",
        )

        assert fig is not None
        plt.close(fig)

    def test_qq_plot_custom_line(self, result_with_ks, sample_data):
        """Test Q-Q plot with custom reference line settings."""
        fig, ax = plot_qq(
            result_with_ks,
            sample_data,
            line_color="blue",
            line_style="-",
            line_width=2.0,
        )

        assert fig is not None
        plt.close(fig)

    def test_qq_plot_small_data(self, result_with_ks):
        """Test Q-Q plot with small sample size."""
        small_data = np.array([45, 50, 55, 48, 52])

        fig, ax = plot_qq(result_with_ks, small_data)

        assert isinstance(fig, plt.Figure)
        assert len(ax.collections) >= 1, "Should have scatter plot even with small data"
        assert len(ax.lines) >= 1, "Should have reference line"
        # With 5 data points, scatter should have 5 points
        scatter_data = ax.collections[0].get_offsets()
        assert len(scatter_data) == 5, "Should plot all 5 data points"

        plt.close(fig)

    def test_qq_plot_equal_aspect(self, result_with_ks, sample_data):
        """Test Q-Q plot has equal aspect ratio."""
        fig, ax = plot_qq(result_with_ks, sample_data)

        # Check aspect is equal
        assert ax.get_aspect() == "equal" or ax.get_aspect() == 1.0
        plt.close(fig)

    def test_qq_plot_figsize(self, result_with_ks, sample_data):
        """Test Q-Q plot with custom figure size."""
        fig, ax = plot_qq(result_with_ks, sample_data, figsize=(8, 8))

        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 8
        plt.close(fig)


class TestPlotPP:
    """Tests for plot_pp function (Probability-Probability plots)."""

    def test_basic_pp_plot(self, result_with_ks, sample_data):
        """Test basic P-P plot creates valid figure with expected elements."""
        fig, ax = plot_pp(result_with_ks, sample_data)

        # Verify figure and axes are valid matplotlib objects
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Verify plot has scatter points and reference line
        assert len(ax.collections) >= 1, "Should have scatter plot"
        assert len(ax.lines) >= 1, "Should have reference line"

        # Verify P-P specific bounds (probabilities are always 0 to 1)
        assert ax.get_xlim() == (0, 1), "X-axis should be [0, 1] for probabilities"
        assert ax.get_ylim() == (0, 1), "Y-axis should be [0, 1] for probabilities"

        # Verify scatter data is within probability bounds
        scatter_data = ax.collections[0].get_offsets()
        assert np.all(scatter_data >= 0) and np.all(scatter_data <= 1), "All points should be in [0, 1]"

        # Verify legend exists
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_pp_plot_with_title(self, result_with_ks, sample_data):
        """Test P-P plot with custom title."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data,
            title="Custom P-P Plot"
        )

        assert "Custom P-P Plot" in ax.get_title()
        plt.close(fig)

    def test_pp_plot_with_custom_labels(self, result_with_ks, sample_data):
        """Test P-P plot with custom axis labels."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data,
            xlabel="Empirical",
            ylabel="Theoretical"
        )

        assert ax.get_xlabel() == "Empirical"
        assert ax.get_ylabel() == "Theoretical"
        plt.close(fig)

    def test_pp_plot_shows_ks_statistic(self, result_with_ks, sample_data):
        """Test P-P plot shows KS and p-value when available."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data
        )

        title = ax.get_title()
        assert "KS=" in title
        assert "p=" in title
        plt.close(fig)

    def test_pp_plot_without_ks_shows_sse(self, normal_result, sample_data):
        """Test P-P plot fallback to SSE when KS is missing."""
        fig, ax = plot_pp(
            normal_result,
            sample_data
        )

        assert "SSE=" in ax.get_title()
        plt.close(fig)

    def test_pp_plot_gamma_distribution(self, gamma_result, sample_data):
        """Test P-P plot handles shape parameters in title."""
        fig, ax = plot_pp(
            gamma_result,
            sample_data
        )

        assert "gamma" in ax.get_title().lower()
        plt.close(fig)

    def test_pp_plot_custom_markers(self, result_with_ks, sample_data):
        """Test P-P plot with custom marker settings."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data,
            marker="D",
            marker_color="green"
        )

        assert fig is not None
        plt.close(fig)

    def test_pp_plot_custom_line(self, result_with_ks, sample_data):
        """Test P-P plot with custom reference line."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data,
            line_color="orange",
            line_width=3.0
        )

        assert fig is not None
        plt.close(fig)

    def test_pp_plot_small_data(self, result_with_ks):
        """Test P-P plot edge case with very small data."""
        small_data = np.array([1.2, 2.3, 3.4, 4.5, 5.6])

        fig, ax = plot_pp(result_with_ks, small_data)

        assert isinstance(fig, plt.Figure)
        assert len(ax.collections) >= 1, "Should have scatter plot even with small data"
        assert len(ax.lines) >= 1, "Should have reference line"
        # With 5 data points, scatter should have 5 points
        scatter_data = ax.collections[0].get_offsets()
        assert len(scatter_data) == 5, "Should plot all 5 data points"
        # P-P bounds should still be [0, 1]
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)

        plt.close(fig)

    def test_pp_plot_equal_aspect(self, result_with_ks, sample_data):
        """Test P-P plot enforces equal aspect ratio."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data
        )

        assert ax.get_aspect() in ["equal", 1.0]
        plt.close(fig)

    def test_pp_plot_figsize(self, result_with_ks, sample_data):
        """Test P-P plot with custom figure size."""
        fig, ax = plot_pp(
            result_with_ks,
            sample_data,
            figsize=(12, 12)
        )

        assert fig.get_figwidth() == 12
        plt.close(fig)

    def test_pp_plot_data_with_ties(self, result_with_ks):
        """Test P-P plot handles duplicate values correctly."""
        # Data with ties (duplicate values)
        data_with_ties = np.array([50, 50, 50, 45, 55, 55, 60])

        fig, ax = plot_pp(result_with_ks, data_with_ties)

        assert isinstance(fig, plt.Figure)
        scatter_data = ax.collections[0].get_offsets()
        assert len(scatter_data) == 7, "Should plot all points including duplicates"
        # All probabilities should still be in [0, 1]
        assert np.all(scatter_data >= 0) and np.all(scatter_data <= 1)

        plt.close(fig)

    def test_pp_plot_single_point(self, result_with_ks):
        """Test P-P plot handles single data point edge case."""
        single_point = np.array([50.0])

        fig, ax = plot_pp(result_with_ks, single_point)

        assert isinstance(fig, plt.Figure)
        scatter_data = ax.collections[0].get_offsets()
        assert len(scatter_data) == 1, "Should plot the single point"
        assert ax.get_xlim() == (0, 1), "Bounds should still be [0, 1]"
        assert ax.get_ylim() == (0, 1)

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
