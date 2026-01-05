"""Visualization utilities for fitted distributions."""

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import scipy.stats as st

# Optional matplotlib import - users can install with: pip install spark-bestfit[plotting]
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment]
    Axes = None  # type: ignore[assignment,misc]
    Figure = None  # type: ignore[assignment,misc]

from spark_bestfit.fitting import compute_pdf_range, extract_distribution_params


def _check_matplotlib() -> None:
    """Raise helpful error if matplotlib is not installed."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for plotting. Install with:\n"
            "  pip install spark-bestfit[plotting]\n\n"
            "Alternatively, use result.pdf(), result.cdf(), result.sample() "
            "to get data for your own plots with any visualization library."
        )


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
    _check_matplotlib()

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
    _check_matplotlib()

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


def plot_qq(
    result: "DistributionFitResult",
    data: np.ndarray,
    title: str = "",
    xlabel: str = "Theoretical Quantiles",
    ylabel: str = "Sample Quantiles",
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 100,
    marker: str = "o",
    marker_size: int = 30,
    marker_alpha: float = 0.6,
    marker_color: str = "steelblue",
    line_color: str = "red",
    line_style: str = "--",
    line_width: float = 1.5,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """Create a Q-Q (quantile-quantile) plot for goodness-of-fit assessment.

    A Q-Q plot compares the quantiles of the sample data against the theoretical
    quantiles of the fitted distribution. If the data follows the fitted
    distribution well, the points will fall approximately along the reference line.

    Args:
        result: Fitted distribution result
        data: Sample data array (1D numpy array)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        marker: Marker style for data points
        marker_size: Size of markers
        marker_alpha: Marker transparency (0-1)
        marker_color: Color of markers
        line_color: Color of reference line
        line_style: Style of reference line
        line_width: Width of reference line
        title_fontsize: Title font size
        label_fontsize: Axis label font size
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
        >>> fitter.plot_qq(best, df, 'value', title='Q-Q Plot')
    """
    _check_matplotlib()

    # Sort the data
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # Calculate plotting positions using Blom's formula
    # This is a standard approach that works well across distributions
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)

    # Calculate theoretical quantiles using the fitted distribution
    theoretical_quantiles = result.ppf(positions)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    ax.scatter(
        theoretical_quantiles,
        sorted_data,
        s=marker_size,
        alpha=marker_alpha,
        c=marker_color,
        marker=marker,
        edgecolors="white",
        linewidth=0.5,
        label="Data",
        zorder=2,
    )

    # Add reference line (y = x)
    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    margin = (max_val - min_val) * 0.05
    line_range = [min_val - margin, max_val + margin]

    ax.plot(
        line_range,
        line_range,
        color=line_color,
        linestyle=line_style,
        linewidth=line_width,
        label="Reference (y=x)",
        zorder=1,
    )

    # Set equal aspect ratio and limits
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect("equal", adjustable="box")

    # Format title with distribution info
    dist = getattr(st, result.distribution)
    param_names = (dist.shapes + ", loc, scale").split(", ") if dist.shapes else ["loc", "scale"]
    param_str = ", ".join([f"{k}={v:.4f}" for k, v in zip(param_names, result.parameters)])
    dist_title = f"{result.distribution}({param_str})"

    # Add K-S statistic if available
    if result.ks_statistic is not None:
        metrics_str = f"KS={result.ks_statistic:.6f}"
        if result.pvalue is not None:
            metrics_str += f", p={result.pvalue:.4f}"
    else:
        metrics_str = f"SSE={result.sse:.6f}"

    full_title = f"{title}\n{dist_title}\n{metrics_str}" if title else f"{dist_title}\n{metrics_str}"

    ax.set_title(full_title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Configure legend and grid
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax


def plot_pp(
    result: "DistributionFitResult",
    data: np.ndarray,
    title: str = "",
    xlabel: str = "Theoretical Probabilities",
    ylabel: str = "Sample Probabilities",
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 100,
    marker: str = "o",
    marker_size: int = 30,
    marker_alpha: float = 0.6,
    marker_color: str = "steelblue",
    line_color: str = "red",
    line_style: str = "--",
    line_width: float = 1.5,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """
    Create a P-P (probability-probability) plot for goodness-of-fit assessment.

    A P-P plot compares the empirical cumulative distribution function (CDF) of
    the sample data against the theoretical CDF of the fitted distribution.
    It is particularly useful for assessing fit in the center of the distribution.

    Args:
        result: Fitted distribution result
        data: Sample data array (1D numpy array)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        marker: Marker style for data points
        marker_size: Size of markers
        marker_alpha: Marker transparency (0-1)
        marker_color: Color of markers
        line_color: Color of reference line
        line_style: Style of reference line
        line_width: Width of reference line
        title_fontsize: Title font size
        label_fontsize: Axis label font size
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
        >>> fitter.plot_pp(best, df, 'value', title='P-P Plot')
    """
    _check_matplotlib()

    # 1. Sort the data
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # 2. Calculate empirical probabilities (y-axis) using Blom's formula
    empirical_probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)

    # 3. Calculate theoretical probabilities (x-axis) using the fitted CDF
    # This is the main difference from plot_qq!
    theoretical_probs = result.cdf(sorted_data)

    # 4. Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # 5. Plot data points
    ax.scatter(
        theoretical_probs,
        empirical_probs,
        s=marker_size,
        alpha=marker_alpha,
        c=marker_color,
        marker=marker,
        edgecolors="white",
        linewidth=0.5,
        label="Data",
        zorder=2,
    )

    # 6. Add reference line (y = x from 0 to 1)
    ax.plot(
        [0, 1], [0, 1], color=line_color, linestyle=line_style, linewidth=line_width, label="Reference (y=x)", zorder=1
    )

    # 7. Set aspect and limits (probabilities are always 0 to 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal", adjustable="box")

    # 8. Title and labels
    dist = getattr(st, result.distribution)
    param_names = (dist.shapes + ", loc, scale").split(", ") if dist.shapes else ["loc", "scale"]
    param_str = ", ".join([f"{k}={v:.4f}" for k, v in zip(param_names, result.parameters)])
    dist_title = f"{result.distribution}({param_str})"

    # Add K-S or SSE metric
    # Add K-S statistic if available
    if result.ks_statistic is not None:
        metrics_str = f"KS={result.ks_statistic:.6f}"
        if result.pvalue is not None:
            metrics_str += f", p={result.pvalue:.4f}"
    else:
        # Fallback to SSE if KS is not available
        metrics_str = f"SSE={result.sse:.6f}"

    # Combine title, distribution info, and metrics
    full_title = f"{title}\n{dist_title}\n{metrics_str}" if title else f"{dist_title}\n{metrics_str}"

    ax.set_title(full_title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax


def plot_discrete_distribution(
    result: "DistributionFitResult",
    data: np.ndarray,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "Probability",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    show_histogram: bool = True,
    histogram_alpha: float = 0.7,
    pmf_linewidth: int = 2,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """Plot fitted discrete distribution against data histogram.

    Args:
        result: Fitted discrete distribution result
        data: Integer data array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        show_histogram: Show data histogram
        histogram_alpha: Histogram transparency (0-1)
        pmf_linewidth: Line width for PMF markers
        title_fontsize: Title font size
        label_fontsize: Axis label font size
        legend_fontsize: Legend font size
        grid_alpha: Grid transparency (0-1)
        save_path: Optional path to save figure
        save_format: Save format (png, pdf, svg)

    Returns:
        Tuple of (figure, axis)
    """
    _check_matplotlib()

    # Get scipy distribution and parameters
    dist = getattr(st, result.distribution)
    params = list(result.parameters)

    # Handle integer parameters for certain distributions
    int_param_dists = {"binom", "betabinom", "hypergeom", "nhypergeom", "boltzmann", "zipfian"}
    if result.distribution in int_param_dists:
        params[0] = int(round(params[0]))

    # Compute empirical PMF from data
    data_int = data.astype(int)
    unique_vals, counts = np.unique(data_int, return_counts=True)
    empirical_pmf = counts / len(data_int)

    # Extend range slightly for theoretical PMF
    x_min = max(0, unique_vals.min() - 2)
    x_max = unique_vals.max() + 2
    x_range = np.arange(x_min, x_max + 1)

    # Compute theoretical PMF
    theoretical_pmf = dist.pmf(x_range, *params)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot empirical histogram as bars
    if show_histogram:
        ax.bar(
            unique_vals,
            empirical_pmf,
            width=0.8,
            alpha=histogram_alpha,
            label="Empirical PMF",
            color="skyblue",
            edgecolor="navy",
            linewidth=0.5,
            zorder=2,
        )

    # Plot theoretical PMF as stems/lollipops
    markerline, stemlines, baseline = ax.stem(
        x_range,
        theoretical_pmf,
        linefmt="r-",
        markerfmt="ro",
        basefmt=" ",
        label="Fitted PMF",
    )
    plt.setp(markerline, markersize=6, zorder=3)
    plt.setp(stemlines, linewidth=pmf_linewidth, zorder=3)

    # Format parameter string
    param_names = _get_discrete_param_names(result.distribution)
    param_str = ", ".join([f"{k}={v:.4f}" for k, v in zip(param_names, result.parameters)])

    dist_title = f"{result.distribution}({param_str})"

    # Build metrics string
    metrics_parts = []
    if result.sse is not None:
        metrics_parts.append(f"SSE: {result.sse:.6f}")
    if result.ks_statistic is not None:
        metrics_parts.append(f"KS: {result.ks_statistic:.4f}")
    metrics_str = ", ".join(metrics_parts)

    # Set title
    full_title = f"{title}\n{dist_title}\n{metrics_str}" if title else f"{dist_title}\n{metrics_str}"

    ax.set_title(full_title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Configure legend
    ax.legend(fontsize=legend_fontsize, loc="best", framealpha=0.9)

    # Configure grid
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=1)

    # Set x-axis to integers
    ax.set_xticks(x_range[:: max(1, len(x_range) // 20)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig, ax


def _get_discrete_param_names(dist_name: str) -> List[str]:
    """Get parameter names for discrete distributions."""
    param_map = {
        "poisson": ["mu"],
        "binom": ["n", "p"],
        "nbinom": ["n", "p"],
        "geom": ["p"],
        "hypergeom": ["M", "n", "N"],
        "betabinom": ["n", "a", "b"],
        "betanbinom": ["n", "a", "b"],
        "zipf": ["a"],
        "zipfian": ["a", "n"],
        "boltzmann": ["lambda", "N"],
        "dlaplace": ["a"],
        "logser": ["p"],
        "planck": ["lambda"],
        "skellam": ["mu1", "mu2"],
        "yulesimon": ["alpha"],
        "nhypergeom": ["M", "n", "r"],
    }
    return param_map.get(dist_name, ["param" + str(i) for i in range(10)])
