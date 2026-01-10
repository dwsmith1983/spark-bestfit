"""Visualization utilities for fitted distributions."""

import warnings
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


def _get_scipy_distribution(dist_name: str):
    """Safely get scipy distribution by name.

    Args:
        dist_name: Name of the scipy.stats distribution

    Returns:
        The scipy distribution class

    Raises:
        ValueError: If distribution name is not found in scipy.stats
    """
    try:
        return getattr(st, dist_name)
    except AttributeError:
        raise ValueError(f"Unknown distribution '{dist_name}'. " f"Must be a valid scipy.stats distribution name.")


def _format_distribution_params(
    result: "DistributionFitResult",
    precision: int = 4,
) -> Tuple[str, str]:
    """Format distribution name and parameters for display.

    Args:
        result: Fitted distribution result
        precision: Number of decimal places for parameter values

    Returns:
        Tuple of (distribution_title, param_string)
        e.g., ("norm(loc=50.0000, scale=10.0000)", "loc=50.0000, scale=10.0000")
    """
    param_names = result.get_param_names()
    param_str = ", ".join([f"{k}={v:.{precision}f}" for k, v in zip(param_names, result.parameters)])
    dist_title = f"{result.distribution}({param_str})"
    return dist_title, param_str


def _validate_histogram_inputs(
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    func_name: str = "plot",
) -> None:
    """Validate histogram input arrays.

    Args:
        y_hist: Histogram density values
        x_hist: Histogram bin centers
        func_name: Name of calling function for error messages

    Raises:
        ValueError: If inputs are invalid
    """
    if y_hist is None or x_hist is None:
        raise ValueError(f"{func_name} requires both y_hist and x_hist arrays")

    if len(y_hist) == 0 or len(x_hist) == 0:
        raise ValueError(f"{func_name} requires non-empty histogram arrays")

    if len(y_hist) != len(x_hist):
        raise ValueError(
            f"{func_name} requires y_hist and x_hist to have same length, " f"got {len(y_hist)} and {len(x_hist)}"
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
    _validate_histogram_inputs(y_hist, x_hist, "plot_distribution")

    # Get scipy distribution and parameters
    dist = _get_scipy_distribution(result.distribution)
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
    dist_title, _ = _format_distribution_params(result)
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
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

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
    _validate_histogram_inputs(y_hist, x_hist, "plot_comparison")

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
        dist = _get_scipy_distribution(result.distribution)
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
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

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
    dist_title, _ = _format_distribution_params(result)

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
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

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
    dist_title, _ = _format_distribution_params(result)

    # Add K-S or SSE metric
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
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

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
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

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


def plot_residual_histogram(
    result: "DistributionFitResult",
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    title: str = "",
    xlabel: str = "Residual (Observed - Expected)",
    ylabel: str = "Frequency",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    bins: int = 30,
    histogram_alpha: float = 0.7,
    histogram_color: str = "steelblue",
    show_zero_line: bool = True,
    zero_line_color: str = "red",
    zero_line_style: str = "--",
    zero_line_width: float = 1.5,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """Plot a histogram of residuals (observed - expected density).

    Residuals are computed as the difference between the empirical density
    (from histogram) and the theoretical density (from fitted distribution).
    A good fit should show residuals centered near zero.

    Args:
        result: Fitted distribution result
        y_hist: Histogram density values (empirical density)
        x_hist: Histogram bin centers
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        bins: Number of bins for the residual histogram
        histogram_alpha: Histogram transparency (0-1)
        histogram_color: Color of histogram bars
        show_zero_line: Whether to show a vertical line at zero
        zero_line_color: Color of the zero reference line
        zero_line_style: Style of the zero reference line
        zero_line_width: Width of the zero reference line
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
        >>> y_hist, x_edges = np.histogram(data, bins=50, density=True)
        >>> x_hist = (x_edges[:-1] + x_edges[1:]) / 2
        >>> plot_residual_histogram(best, y_hist, x_hist)
    """
    _check_matplotlib()
    _validate_histogram_inputs(y_hist, x_hist, "plot_residual_histogram")

    # Compute theoretical density at bin centers
    theoretical_density = result.pdf(x_hist)

    # Compute residuals
    residuals = y_hist - theoretical_density

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram of residuals
    ax.hist(
        residuals,
        bins=bins,
        alpha=histogram_alpha,
        color=histogram_color,
        edgecolor="white",
        linewidth=0.5,
        label="Residuals",
        zorder=2,
    )

    # Add zero reference line
    if show_zero_line:
        ax.axvline(
            x=0,
            color=zero_line_color,
            linestyle=zero_line_style,
            linewidth=zero_line_width,
            label="Zero",
            zorder=3,
        )

    # Format title with distribution info and residual statistics
    dist_title, _ = _format_distribution_params(result)

    # Compute residual statistics
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    stats_str = f"Mean={mean_resid:.6f}, Std={std_resid:.6f}"

    full_title = f"{title}\n{dist_title}\n{stats_str}" if title else f"{dist_title}\n{stats_str}"

    ax.set_title(full_title, fontsize=title_fontsize, pad=15)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Configure legend and grid
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

    return fig, ax


def plot_cdf_comparison(
    result: "DistributionFitResult",
    data: np.ndarray,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "Cumulative Probability",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    empirical_color: str = "steelblue",
    empirical_linewidth: float = 2.0,
    empirical_alpha: float = 0.8,
    theoretical_color: str = "red",
    theoretical_linewidth: float = 2.0,
    theoretical_linestyle: str = "--",
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    legend_fontsize: int = 10,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, Axes]:
    """Plot empirical CDF overlaid with theoretical CDF from the fitted distribution.

    The empirical CDF is computed from the sample data using the step function.
    The theoretical CDF is computed from the fitted distribution. A good fit
    shows close alignment between the two CDFs.

    Args:
        result: Fitted distribution result
        data: Sample data array (1D numpy array)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        empirical_color: Color of empirical CDF line
        empirical_linewidth: Line width for empirical CDF
        empirical_alpha: Transparency of empirical CDF line
        theoretical_color: Color of theoretical CDF line
        theoretical_linewidth: Line width for theoretical CDF
        theoretical_linestyle: Line style for theoretical CDF
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
        >>> plot_cdf_comparison(best, data, title='CDF Comparison')
    """
    _check_matplotlib()

    # Validate data array
    if data is None or len(data) == 0:
        raise ValueError("plot_cdf_comparison requires non-empty data array")

    # Sort data for empirical CDF
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # Compute empirical CDF (step function)
    empirical_cdf = np.arange(1, n + 1) / n

    # Compute theoretical CDF over the data range
    x_range = np.linspace(sorted_data.min(), sorted_data.max(), 1000)
    theoretical_cdf = result.cdf(x_range)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot empirical CDF as step function
    ax.step(
        sorted_data,
        empirical_cdf,
        where="post",
        color=empirical_color,
        linewidth=empirical_linewidth,
        alpha=empirical_alpha,
        label="Empirical CDF",
        zorder=2,
    )

    # Plot theoretical CDF as smooth curve
    ax.plot(
        x_range,
        theoretical_cdf,
        color=theoretical_color,
        linewidth=theoretical_linewidth,
        linestyle=theoretical_linestyle,
        label="Theoretical CDF",
        zorder=3,
    )

    # Format title with distribution info
    dist_title, _ = _format_distribution_params(result)

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

    # Set y-axis limits
    ax.set_ylim([0, 1.05])

    # Configure legend and grid
    ax.legend(fontsize=legend_fontsize, loc="lower right", framealpha=0.9)
    ax.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

    return fig, ax


def plot_diagnostics(
    result: "DistributionFitResult",
    data: np.ndarray,
    y_hist: Optional[np.ndarray] = None,
    x_hist: Optional[np.ndarray] = None,
    bins: int = 50,
    title: str = "",
    figsize: Tuple[int, int] = (14, 12),
    dpi: int = 100,
    title_fontsize: int = 16,
    subplot_title_fontsize: int = 12,
    label_fontsize: int = 10,
    grid_alpha: float = 0.3,
    save_path: Optional[str] = None,
    save_format: str = "png",
) -> Tuple[Figure, np.ndarray]:
    """Create a 2x2 diagnostic plot panel for assessing distribution fit quality.

    Generates four diagnostic plots:
    - Q-Q Plot (top-left): Compares sample quantiles vs theoretical quantiles
    - P-P Plot (top-right): Compares empirical vs theoretical probabilities
    - Residual Histogram (bottom-left): Distribution of fit residuals
    - CDF Comparison (bottom-right): Empirical vs theoretical CDF overlay

    Args:
        result: Fitted distribution result
        data: Sample data array (1D numpy array)
        y_hist: Optional pre-computed histogram density values. If None,
            computed from data using specified bins.
        x_hist: Optional pre-computed histogram bin centers. If None,
            computed from data using specified bins.
        bins: Number of histogram bins (used if y_hist/x_hist not provided)
        title: Overall figure title
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figures
        title_fontsize: Main title font size
        subplot_title_fontsize: Subplot title font size
        label_fontsize: Axis label font size
        grid_alpha: Grid transparency (0-1)
        save_path: Optional path to save figure
        save_format: Save format (png, pdf, svg)

    Returns:
        Tuple of (figure, array of axes)

    Example:
        >>> from spark_bestfit import DistributionFitter
        >>> fitter = DistributionFitter(spark)
        >>> results = fitter.fit(df, 'value')
        >>> best = results.best(n=1)[0]
        >>> fig, axes = plot_diagnostics(best, data, title='Fit Diagnostics')
    """
    _check_matplotlib()

    # Validate data array
    if data is None or len(data) == 0:
        raise ValueError("plot_diagnostics requires non-empty data array")

    # Compute histogram if not provided
    if y_hist is None or x_hist is None:
        y_hist_computed, x_edges = np.histogram(data, bins=bins, density=True)
        x_hist_computed = (x_edges[:-1] + x_edges[1:]) / 2
        y_hist = y_hist_computed
        x_hist = x_hist_computed

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Format distribution info for subplot titles
    dist_info, _ = _format_distribution_params(result, precision=2)

    # Q-Q Plot (top-left)
    ax_qq = axes[0, 0]
    sorted_data = np.sort(data)
    n = len(sorted_data)
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical_quantiles = result.ppf(positions)

    ax_qq.scatter(
        theoretical_quantiles,
        sorted_data,
        s=20,
        alpha=0.5,
        c="steelblue",
        edgecolors="white",
        linewidth=0.3,
        zorder=2,
    )
    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    margin = (max_val - min_val) * 0.05
    line_range = [min_val - margin, max_val + margin]
    ax_qq.plot(line_range, line_range, "r--", lw=1.5, label="y=x", zorder=1)
    ax_qq.set_xlim(line_range)
    ax_qq.set_ylim(line_range)
    ax_qq.set_aspect("equal", adjustable="box")
    ax_qq.set_title("Q-Q Plot", fontsize=subplot_title_fontsize)
    ax_qq.set_xlabel("Theoretical Quantiles", fontsize=label_fontsize)
    ax_qq.set_ylabel("Sample Quantiles", fontsize=label_fontsize)
    ax_qq.legend(fontsize=8, loc="upper left")
    ax_qq.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    # P-P Plot (top-right)
    ax_pp = axes[0, 1]
    empirical_probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical_probs = result.cdf(sorted_data)

    ax_pp.scatter(
        theoretical_probs,
        empirical_probs,
        s=20,
        alpha=0.5,
        c="steelblue",
        edgecolors="white",
        linewidth=0.3,
        zorder=2,
    )
    ax_pp.plot([0, 1], [0, 1], "r--", lw=1.5, label="y=x", zorder=1)
    ax_pp.set_xlim([0, 1])
    ax_pp.set_ylim([0, 1])
    ax_pp.set_aspect("equal", adjustable="box")
    ax_pp.set_title("P-P Plot", fontsize=subplot_title_fontsize)
    ax_pp.set_xlabel("Theoretical Probabilities", fontsize=label_fontsize)
    ax_pp.set_ylabel("Sample Probabilities", fontsize=label_fontsize)
    ax_pp.legend(fontsize=8, loc="upper left")
    ax_pp.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    # Residual Histogram (bottom-left)
    ax_resid = axes[1, 0]
    theoretical_density = result.pdf(x_hist)
    residuals = y_hist - theoretical_density
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)

    ax_resid.hist(
        residuals,
        bins=30,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax_resid.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero", zorder=3)
    ax_resid.set_title(
        f"Residual Histogram\nMean={mean_resid:.4f}, Std={std_resid:.4f}", fontsize=subplot_title_fontsize
    )
    ax_resid.set_xlabel("Residual (Observed - Expected)", fontsize=label_fontsize)
    ax_resid.set_ylabel("Frequency", fontsize=label_fontsize)
    ax_resid.legend(fontsize=8, loc="upper right")
    ax_resid.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    # CDF Comparison (bottom-right)
    ax_cdf = axes[1, 1]
    empirical_cdf = np.arange(1, n + 1) / n
    x_range = np.linspace(sorted_data.min(), sorted_data.max(), 1000)
    theoretical_cdf = result.cdf(x_range)

    ax_cdf.step(
        sorted_data,
        empirical_cdf,
        where="post",
        color="steelblue",
        linewidth=1.5,
        alpha=0.8,
        label="Empirical CDF",
        zorder=2,
    )
    ax_cdf.plot(
        x_range,
        theoretical_cdf,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label="Theoretical CDF",
        zorder=3,
    )
    ax_cdf.set_ylim([0, 1.05])
    ax_cdf.set_title("CDF Comparison", fontsize=subplot_title_fontsize)
    ax_cdf.set_xlabel("Value", fontsize=label_fontsize)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=label_fontsize)
    ax_cdf.legend(fontsize=8, loc="lower right")
    ax_cdf.grid(alpha=grid_alpha, linestyle="--", linewidth=0.5, zorder=0)

    # Set overall title
    if title:
        fig.suptitle(f"{title}\n{dist_info}", fontsize=title_fontsize, y=1.02)
    else:
        fig.suptitle(dist_info, fontsize=title_fontsize, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format=save_format, bbox_inches="tight")
        warnings.warn(f"Plot saved to: {save_path}", stacklevel=2)

    return fig, axes
