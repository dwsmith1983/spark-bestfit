"""Visualization utilities for fitted distributions."""

from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spark_bestfit.fitting import compute_pdf_range, extract_distribution_params

if TYPE_CHECKING:
    from spark_bestfit.results import DistributionFitResult


def plot_distribution(
    result: "DistributionFitResult",
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "Density",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    show_histogram: bool = True,
    histogram_alpha: float = 0.5,
    pdf_linewidth: int = 2,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """Plot fitted distribution against data histogram.

    Args:
        result: Fitted distribution result
        y_hist: Histogram density values
        x_hist: Histogram bin centers
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        show_histogram: Show data histogram
        histogram_alpha: Histogram transparency (0-1)
        pdf_linewidth: Line width for PDF curve
        title_fontsize: Title font size
        label_fontsize: Axis label font size
        legend_fontsize: Legend font size
        grid_alpha: Grid transparency (0-1)
        save_path: Optional path to save figure
        save_format: Save format (png, pdf, svg)

    Returns:
        Tuple of (figure, axis)

    Example:
        >>> from spark_bestfit import DistributionFitter
        >>> fitter = DistributionFitter(spark)
        >>> results = fitter.fit(df, 'value')
        >>> best = results.best(n=1)[0]
        >>> fitter.plot(best, df, 'value', title='Best Fit')
    """
    # Get scipy distribution and parameters
    dist = getattr(st, result.distribution)
    params = result.parameters

    # Extract shape, loc, scale using utility function
    shape, loc, scale = extract_distribution_params(params)

    # Compute PDF range using utility function
    start, end = compute_pdf_range(dist, params, x_hist)

    x_pdf = np.linspace(start, end, 1000)
    y_pdf = dist.pdf(x_pdf, *shape, loc=loc, scale=scale)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot PDF
    ax.plot(
        x_pdf,
        y_pdf,
        "r-",
        lw=pdf_linewidth,
        label="Fitted PDF",
        zorder=3,
    )

    # Plot histogram
    if show_histogram:
        # Convert histogram density to bar plot
        bin_width = x_hist[1] - x_hist[0] if len(x_hist) > 1 else 1.0
        ax.bar(
            x_hist,
            y_hist,
            width=bin_width * 0.9,
            alpha=histogram_alpha,
            label="Data Histogram",
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            zorder=2,
        )

    # Format parameter string
    param_names = (dist.shapes + ", loc, scale").split(", ") if dist.shapes else ["loc", "scale"]
    param_str = ", ".join([f"{k}={v:.4f}" for k, v in zip(param_names, params)])

    dist_title = f"{result.distribution}({param_str})"
    sse_str = f"SSE: {result.sse:.6f}"

    if result.aic is not None and result.bic is not None:
        metrics_str = f"{sse_str}, AIC: {result.aic:.2f}, BIC: {result.bic:.2f}"
    else:
        metrics_str = sse_str

    # Set title
    full_title = f"{title}\n{dist_title}\n{metrics_str}" if title else f"{dist_title}\n{metrics_str}"

    ax.set_title(full_title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Configure legend
    ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.9)

    # Configure grid
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=1)

    # Improve layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax


def plot_comparison(
    results: List["DistributionFitResult"],
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    title: str = "Distribution Comparison",
    xlabel: str = "Value",
    ylabel: str = "Density",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    show_histogram: bool = True,
    histogram_alpha: float = 0.5,
    pdf_linewidth: int = 2,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """Plot multiple fitted distributions for comparison.

    Args:
        results: List of DistributionFitResult objects
        y_hist: Histogram density values
        x_hist: Histogram bin centers
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch
        show_histogram: Show data histogram
        histogram_alpha: Histogram transparency
        pdf_linewidth: PDF line width
        title_fontsize: Title font size
        label_fontsize: Label font size
        legend_fontsize: Legend font size
        grid_alpha: Grid transparency
        save_path: Optional path to save figure
        save_format: Save format

    Returns:
        Tuple of (figure, axis)

    Example:
        >>> top_3 = results.best(n=3)
        >>> fitter.plot_comparison(top_3, df, 'value')
    """
    if not results:
        raise ValueError("Must provide at least one result to plot")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    if show_histogram:
        bin_width = x_hist[1] - x_hist[0] if len(x_hist) > 1 else 1.0
        ax.bar(
            x_hist,
            y_hist,
            width=bin_width * 0.9,
            alpha=histogram_alpha,
            label="Data Histogram",
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            zorder=1,
        )

    # Define colors for multiple distributions
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Plot each distribution
    for i, result in enumerate(results):
        dist = getattr(st, result.distribution)
        params = result.parameters

        # Extract parameters and compute range using utility functions
        shape, loc, scale = extract_distribution_params(params)
        start, end = compute_pdf_range(dist, params, x_hist)

        x_pdf = np.linspace(start, end, 1000)
        y_pdf = dist.pdf(x_pdf, *shape, loc=loc, scale=scale)

        # Plot with label
        label = f"{result.distribution} (SSE={result.sse:.4f})"
        ax.plot(
            x_pdf,
            y_pdf,
            lw=pdf_linewidth,
            label=label,
            color=colors[i],
            zorder=2 + i,
        )

    # Configure plot
    ax.set_title(title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.9)
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax
