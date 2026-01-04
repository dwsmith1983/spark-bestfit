#!/usr/bin/env python3
"""Generate scaling charts from benchmark results.

This script reads pytest-benchmark JSON results and generates
publication-quality scaling charts for documentation.

Usage:
    python scripts/generate_scaling_charts.py [--results-dir PATH] [--output-dir PATH]

Example:
    # Run benchmarks first
    make benchmark       # Spark benchmarks
    make benchmark-ray   # Ray benchmarks

    # Generate charts from latest results
    python scripts/generate_scaling_charts.py
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Modern Minimal Chart Style (Stripe/Linear inspired)
# =============================================================================

# Color palette - high contrast, professional
COLORS = {
    "spark": "#f97316",  # Orange (warm, distinct from cool colors)
    "ray_pandas": "#8b5cf6",  # Violet
    "ray_dataset": "#06b6d4",  # Cyan
    "accent": "#ef4444",  # Red (for highlights)
    "muted": "#94a3b8",  # Slate gray
    "grid": "#e2e8f0",  # Light slate
    "text": "#334155",  # Dark slate
    "background": "#ffffff",  # White
    # Version history colors
    "v1_5": "#94a3b8",  # Slate (oldest)
    "v1_7": "#6366f1",  # Indigo (middle)
    "v2_0": "#22c55e",  # Green (newest/best)
}

# Markers
MARKERS = {
    "spark": "s",  # Square
    "ray_pandas": "o",  # Circle
    "ray_dataset": "^",  # Triangle
}


def apply_modern_style():
    """Apply modern minimal style to matplotlib."""
    plt.rcParams.update(
        {
            # Figure
            "figure.facecolor": COLORS["background"],
            "figure.edgecolor": COLORS["background"],
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            # Axes
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.labelcolor": COLORS["text"],
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlecolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            # Lines
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            # Ticks
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 10,
            # Font
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )


# =============================================================================
# Modern Chart Generation Functions (Minimal Set)
# =============================================================================


def generate_backend_comparison_chart(
    spark_data: dict,
    ray_data: dict,
    output_path: Path,
    chart_type: str = "data_size",
) -> None:
    """Generate a clean 3-way backend comparison chart.

    Args:
        spark_data: Spark benchmark data
        ray_data: Ray benchmark data (includes both pandas and Dataset)
        output_path: Where to save the chart
        chart_type: Either 'data_size' or 'dist_count'
    """
    apply_modern_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "data_size":
        # Data size scaling chart
        spark_x = np.array(spark_data["data_size"]["sizes"])
        spark_y = np.array(spark_data["data_size"]["times"])
        spark_err = np.array(spark_data["data_size"]["stddevs"])

        ray_x = np.array(ray_data["data_size"]["sizes"])
        ray_y = np.array(ray_data["data_size"]["times"])
        ray_err = np.array(ray_data["data_size"]["stddevs"])

        ds_data = ray_data.get("dataset_data_size", {})
        has_dataset = bool(ds_data.get("sizes"))

        # Sort data
        for arr_set in [(spark_x, spark_y, spark_err), (ray_x, ray_y, ray_err)]:
            idx = np.argsort(arr_set[0])
            arr_set[0][:], arr_set[1][:], arr_set[2][:] = arr_set[0][idx], arr_set[1][idx], arr_set[2][idx]

        # Plot lines
        ax.errorbar(
            spark_x,
            spark_y,
            yerr=spark_err,
            fmt=f"-{MARKERS['spark']}",
            color=COLORS["spark"],
            label="Spark",
            capsize=3,
            capthick=1.5,
        )
        ax.errorbar(
            ray_x,
            ray_y,
            yerr=ray_err,
            fmt=f"-{MARKERS['ray_pandas']}",
            color=COLORS["ray_pandas"],
            label="Ray + pandas",
            capsize=3,
            capthick=1.5,
        )

        if has_dataset:
            ds_x = np.array(ds_data["sizes"])
            ds_y = np.array(ds_data["times"])
            ds_err = np.array(ds_data["stddevs"])
            idx = np.argsort(ds_x)
            ds_x, ds_y, ds_err = ds_x[idx], ds_y[idx], ds_err[idx]
            ax.errorbar(
                ds_x,
                ds_y,
                yerr=ds_err,
                fmt=f"-{MARKERS['ray_dataset']}",
                color=COLORS["ray_dataset"],
                label="Ray Dataset",
                capsize=3,
                capthick=1.5,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Data Size (rows)")
        ax.set_ylabel("Fit Time (seconds)")
        ax.set_title("Backend Performance: Data Size Scaling")

        # Format x-axis with readable labels
        ax.set_xticks([25000, 100000, 500000, 1000000])
        ax.set_xticklabels(["25K", "100K", "500K", "1M"])

    else:  # dist_count
        # Distribution count scaling chart
        spark_x = np.array(spark_data["dist_count"]["counts"])
        spark_y = np.array(spark_data["dist_count"]["times"])
        spark_err = np.array(spark_data["dist_count"]["stddevs"])

        ray_x = np.array(ray_data["dist_count"]["counts"])
        ray_y = np.array(ray_data["dist_count"]["times"])
        ray_err = np.array(ray_data["dist_count"]["stddevs"])

        ds_data = ray_data.get("dataset_dist_count", {})
        has_dataset = bool(ds_data.get("counts"))

        # Sort data
        for arr_set in [(spark_x, spark_y, spark_err), (ray_x, ray_y, ray_err)]:
            idx = np.argsort(arr_set[0])
            arr_set[0][:], arr_set[1][:], arr_set[2][:] = arr_set[0][idx], arr_set[1][idx], arr_set[2][idx]

        # Plot lines
        ax.errorbar(
            spark_x,
            spark_y,
            yerr=spark_err,
            fmt=f"-{MARKERS['spark']}",
            color=COLORS["spark"],
            label="Spark",
            capsize=3,
            capthick=1.5,
        )
        ax.errorbar(
            ray_x,
            ray_y,
            yerr=ray_err,
            fmt=f"-{MARKERS['ray_pandas']}",
            color=COLORS["ray_pandas"],
            label="Ray + pandas",
            capsize=3,
            capthick=1.5,
        )

        if has_dataset:
            ds_x = np.array(ds_data["counts"])
            ds_y = np.array(ds_data["times"])
            ds_err = np.array(ds_data["stddevs"])
            idx = np.argsort(ds_x)
            ds_x, ds_y, ds_err = ds_x[idx], ds_y[idx], ds_err[idx]
            ax.errorbar(
                ds_x,
                ds_y,
                yerr=ds_err,
                fmt=f"-{MARKERS['ray_dataset']}",
                color=COLORS["ray_dataset"],
                label="Ray Dataset",
                capsize=3,
                capthick=1.5,
            )

        ax.set_xlabel("Number of Distributions")
        ax.set_ylabel("Fit Time (seconds)")
        ax.set_title("Backend Performance: Distribution Count Scaling")
        ax.set_xlim(0, 115)

    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_lazy_metrics_chart(data: dict, output_path: Path) -> None:
    """Generate a clean lazy metrics performance chart.

    Shows the speedup from using lazy metrics (AIC-only workflow).
    """
    apply_modern_style()

    labels = data.get("lazy_metrics", {}).get("labels", [])
    times = data.get("lazy_metrics", {}).get("times", [])
    stddevs = data.get("lazy_metrics", {}).get("stddevs", [])

    if len(labels) < 2:
        print("Not enough lazy metrics data for chart")
        return

    # Simplify labels
    clean_labels = []
    for label in labels:
        if "Eager" in label:
            clean_labels.append("Eager\n(all metrics)")
        elif "AIC only" in label.lower() or "aic" in label.lower():
            clean_labels.append("Lazy\n(AIC only)")
        elif "KS" in label:
            clean_labels.append("Lazy\n(+KS)")
        elif "materialize" in label.lower():
            clean_labels.append("Lazy\n(+materialize)")
        else:
            clean_labels.append(label)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(clean_labels))
    bars = ax.bar(x, times, yerr=stddevs, capsize=4, color=COLORS["muted"], edgecolor="none")

    # Highlight the fastest (lazy AIC only)
    if len(times) >= 2:
        min_idx = np.argmin(times)
        bars[min_idx].set_color(COLORS["ray_pandas"])

        # Calculate speedup
        baseline = times[0]  # Eager is baseline
        speedup = (baseline - times[min_idx]) / baseline * 100
        ax.annotate(
            f"{speedup:.0f}% faster",
            xy=(min_idx, times[min_idx]),
            xytext=(min_idx, times[min_idx] + 0.5),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=COLORS["ray_pandas"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels)
    ax.set_ylabel("Fit Time (seconds)")
    ax.set_title("Lazy Metrics: Skip Unnecessary Computations")
    ax.set_ylim(bottom=0)

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{time:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["text"],
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_version_history_chart(output_path: Path) -> None:
    """Generate a chart showing Spark performance improvements across versions.

    Data based on documented improvements in CHANGELOG and performance.rst.
    """
    apply_modern_style()

    # Performance data by version (100K rows, 90 distributions)
    # Based on documented benchmarks in performance.rst
    versions = ["v1.4", "v1.5", "v1.7", "v2.0"]

    # Fit time progression:
    # v1.4: ~20s - no lazy metrics, fewer exclusions, all metrics eager
    # v1.5: ~18s - lazy metrics introduced (optional), still mostly eager workflows
    # v1.7: ~10s - slow dist exclusions (tukeylambda, nct, dpareto_lognorm) added
    # v2.0: ~5s - flattened schema + broadcast lifecycle
    fit_times = [20.0, 18.0, 10.0, 5.0]

    # Key features per version (Spark-specific improvements only)
    features = [
        "All metrics\neager",
        "Lazy metrics\nintroduced",
        "Slow dist\nexclusions",
        "Flattened schema\n+ broadcast",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(versions))
    colors = [COLORS["muted"], COLORS["v1_5"], COLORS["v1_7"], COLORS["v2_0"]]

    bars = ax.bar(x, fit_times, color=colors, edgecolor="none", width=0.6)

    # Add value labels
    for bar, time in zip(bars, fit_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{time:.1f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
        )

    # Add feature labels below bars
    for i, (bar, feature) in enumerate(zip(bars, features)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, -1.2, feature, ha="center", va="top", fontsize=9, color=COLORS["text"]
        )

    # Add improvement annotations
    baseline = fit_times[0]
    for i in range(1, len(fit_times)):
        improvement = (baseline - fit_times[i]) / baseline * 100
        # Use white text on dark bars (indigo v1.7, green v2.0)
        text_color = "white" if i >= 2 else COLORS["text"]
        ax.annotate(
            f"-{improvement:.0f}%",
            xy=(i, fit_times[i] / 2),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=text_color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=12, fontweight="bold")
    ax.set_ylabel("Fit Time (seconds)")
    ax.set_title("Spark Performance: Version History")
    ax.set_ylim(bottom=0, top=24)

    # Remove x-axis line to make room for feature labels
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for feature labels
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_overhead_chart(spark_data: dict, ray_data: dict, output_path: Path) -> None:
    """Generate a clean backend overhead comparison chart.

    Shows why Ray is faster for small workloads (lower startup overhead).
    """
    apply_modern_style()

    # Get the smallest workload times (5 distributions)
    spark_dists = dict(zip(spark_data["dist_count"]["counts"], spark_data["dist_count"]["times"]))
    ray_dists = dict(zip(ray_data["dist_count"]["counts"], ray_data["dist_count"]["times"]))

    # Use 5 distributions as the "overhead-dominated" benchmark
    spark_5 = spark_dists.get(5, 0)
    ray_5 = ray_dists.get(5, 0)

    # Use 90 distributions as the "work-dominated" benchmark
    spark_90 = spark_dists.get(90, 0)
    ray_90 = ray_dists.get(90, 0)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.array([0, 1])
    width = 0.35

    # Spark bars
    spark_times = [spark_5, spark_90]
    ray_times = [ray_5, ray_90]

    bars1 = ax.bar(x - width / 2, spark_times, width, label="Spark", color=COLORS["spark"])
    bars2 = ax.bar(x + width / 2, ray_times, width, label="Ray + pandas", color=COLORS["ray_pandas"])

    ax.set_xticks(x)
    ax.set_xticklabels(["5 distributions\n(overhead-dominated)", "90 distributions\n(work-dominated)"])
    ax.set_ylabel("Fit Time (seconds)")
    ax.set_title("Why Ray is Faster: Lower Startup Overhead")
    ax.legend()
    ax.set_ylim(bottom=0)

    # Add speedup annotations
    for i, (s, r) in enumerate(zip(spark_times, ray_times)):
        if s > 0 and r > 0:
            speedup = s / r
            ax.annotate(
                f"{speedup:.1f}×",
                xy=(i, max(s, r) + 0.2),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=COLORS["accent"],
            )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["text"],
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def load_benchmark_results(results_dir: Path, backend: str = "spark") -> Optional[dict]:
    """Load the latest benchmark results JSON file for a specific backend.

    Args:
        results_dir: Directory containing benchmark results
        backend: Either 'spark' or 'ray' to load appropriate results

    Returns:
        Benchmark results dict or None if not found
    """
    # pytest-benchmark saves results in .benchmarks/ directory
    benchmark_files = list(results_dir.glob("**/*.json"))
    if not benchmark_files:
        print(f"No benchmark results found in {results_dir}")
        return None

    # Filter files by backend type based on content
    backend_files = []
    for f in benchmark_files:
        with open(f) as fp:
            try:
                data = json.load(fp)
                # Check if file contains Ray or Spark benchmarks
                benchmarks = data.get("benchmarks", [])
                if benchmarks:
                    fullname = benchmarks[0].get("fullname", "")
                    if backend == "ray" and "Ray" in fullname:
                        backend_files.append(f)
                    elif backend == "spark" and "Ray" not in fullname:
                        backend_files.append(f)
            except json.JSONDecodeError:
                continue

    if not backend_files:
        # Fall back to latest file if no backend-specific match
        if backend == "spark":
            # For Spark, use files without 'ray' in name
            backend_files = [f for f in benchmark_files if "ray" not in f.name.lower()]
        else:
            # For Ray, use files with 'ray' in name
            backend_files = [f for f in benchmark_files if "ray" in f.name.lower()]

    if not backend_files:
        print(f"No {backend} benchmark results found in {results_dir}")
        return None

    # Get the most recent file
    latest = max(backend_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading {backend} results from: {latest}")

    with open(latest) as file_handle:
        return json.load(file_handle)


def extract_scaling_data(results: dict, include_dataset: bool = False) -> dict:
    """Extract scaling data from benchmark results.

    Args:
        results: Benchmark results dict from pytest-benchmark JSON
        include_dataset: If True, also extract Ray Dataset benchmarks into separate keys

    Returns:
        Dict with scaling data. If include_dataset=True, includes 'dataset_data_size'
        and 'dataset_dist_count' keys for Ray Dataset results.
    """
    data: dict[str, dict[str, list]] = {
        "data_size": {"sizes": [], "times": [], "stddevs": []},
        "dist_count": {"counts": [], "times": [], "stddevs": []},
        "multi_column": {"labels": [], "times": [], "stddevs": []},
        "lazy_metrics": {"labels": [], "times": [], "stddevs": []},
        "slow_dist_opt": {"labels": [], "times": [], "stddevs": []},  # v1.7.0
    }

    if include_dataset:
        data["dataset_data_size"] = {"sizes": [], "times": [], "stddevs": []}
        data["dataset_dist_count"] = {"counts": [], "times": [], "stddevs": []}

    for benchmark in results.get("benchmarks", []):
        name = benchmark["name"]
        fullname = benchmark.get("fullname", name)  # For class-based matching
        mean_time = benchmark["stats"]["mean"]
        stddev = benchmark["stats"]["stddev"]

        # Check if this is a Ray Dataset benchmark (has "dataset_" prefix)
        is_dataset = "dataset_" in name

        # Parse data size benchmarks
        if "25k_rows" in name and not is_dataset:
            data["data_size"]["sizes"].append(25_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)
        elif "100k_rows" in name and "lazy" not in name and not is_dataset:
            data["data_size"]["sizes"].append(100_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)
        elif "500k_rows" in name and not is_dataset:
            data["data_size"]["sizes"].append(500_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)
        elif "1m_rows" in name and not is_dataset:
            data["data_size"]["sizes"].append(1_000_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)

        # Parse Ray Dataset data size benchmarks
        if include_dataset and is_dataset:
            if "25k_rows" in name:
                data["dataset_data_size"]["sizes"].append(25_000)
                data["dataset_data_size"]["times"].append(mean_time)
                data["dataset_data_size"]["stddevs"].append(stddev)
            elif "100k_rows" in name:
                data["dataset_data_size"]["sizes"].append(100_000)
                data["dataset_data_size"]["times"].append(mean_time)
                data["dataset_data_size"]["stddevs"].append(stddev)
            elif "500k_rows" in name:
                data["dataset_data_size"]["sizes"].append(500_000)
                data["dataset_data_size"]["times"].append(mean_time)
                data["dataset_data_size"]["stddevs"].append(stddev)
            elif "1m_rows" in name:
                data["dataset_data_size"]["sizes"].append(1_000_000)
                data["dataset_data_size"]["times"].append(mean_time)
                data["dataset_data_size"]["stddevs"].append(stddev)

        # Parse distribution count benchmarks
        if "5_distributions" in name and not is_dataset:
            data["dist_count"]["counts"].append(5)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "20_distributions" in name and not is_dataset:
            data["dist_count"]["counts"].append(20)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "50_distributions" in name and not is_dataset:
            data["dist_count"]["counts"].append(50)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "default_distributions" in name and "discrete" not in name and not is_dataset:
            # ~90 distributions with default exclusions (20 slow dists excluded)
            data["dist_count"]["counts"].append(90)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "SlowDistributionOptimizations" in fullname and "all_distributions" in name and not is_dataset:
            # 107 distributions (only 3 extremely slow excluded) - shows fix works
            data["dist_count"]["counts"].append(107)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)

        # Parse Ray Dataset distribution count benchmarks
        if include_dataset and is_dataset:
            if "5_distributions" in name:
                data["dataset_dist_count"]["counts"].append(5)
                data["dataset_dist_count"]["times"].append(mean_time)
                data["dataset_dist_count"]["stddevs"].append(stddev)
            elif "20_distributions" in name:
                data["dataset_dist_count"]["counts"].append(20)
                data["dataset_dist_count"]["times"].append(mean_time)
                data["dataset_dist_count"]["stddevs"].append(stddev)
            elif "50_distributions" in name:
                data["dataset_dist_count"]["counts"].append(50)
                data["dataset_dist_count"]["times"].append(mean_time)
                data["dataset_dist_count"]["stddevs"].append(stddev)
            elif "default_distributions" in name:
                data["dataset_dist_count"]["counts"].append(90)
                data["dataset_dist_count"]["times"].append(mean_time)
                data["dataset_dist_count"]["stddevs"].append(stddev)
            elif "SlowDistributionOptimizations" in fullname and "all_distributions" in name:
                # 107 distributions (only 3 extremely slow excluded)
                data["dataset_dist_count"]["counts"].append(107)
                data["dataset_dist_count"]["times"].append(mean_time)
                data["dataset_dist_count"]["stddevs"].append(stddev)

        # Parse multi-column efficiency benchmarks (exclude Ray Dataset tests)
        if "3_columns_separately" in name and "discrete" not in name and not is_dataset:
            data["multi_column"]["labels"].append("3 Separate Fits")
            data["multi_column"]["times"].append(mean_time)
            data["multi_column"]["stddevs"].append(stddev)
        elif "3_columns_together" in name and "100k" not in name and "discrete" not in name and not is_dataset:
            data["multi_column"]["labels"].append("1 Multi-Column Fit")
            data["multi_column"]["times"].append(mean_time)
            data["multi_column"]["stddevs"].append(stddev)

        # Parse lazy metrics benchmarks
        if "eager_all_metrics" in name:
            data["lazy_metrics"]["labels"].append("Eager\n(all metrics)")
            data["lazy_metrics"]["times"].append(mean_time)
            data["lazy_metrics"]["stddevs"].append(stddev)
        elif "lazy_aic_only" in name:
            data["lazy_metrics"]["labels"].append("Lazy\n(AIC only)")
            data["lazy_metrics"]["times"].append(mean_time)
            data["lazy_metrics"]["stddevs"].append(stddev)
        elif "lazy_with_ks_on_demand" in name:
            data["lazy_metrics"]["labels"].append("Lazy\n(+ KS on-demand)")
            data["lazy_metrics"]["times"].append(mean_time)
            data["lazy_metrics"]["stddevs"].append(stddev)
        elif "lazy_materialize" in name:
            data["lazy_metrics"]["labels"].append("Lazy\n(+ materialize)")
            data["lazy_metrics"]["times"].append(mean_time)
            data["lazy_metrics"]["stddevs"].append(stddev)

        # Parse slow distribution optimization benchmarks (v1.7.0)
        if "SlowDistributionOptimizations" in fullname:
            if "default_exclusions" in name:
                data["slow_dist_opt"]["labels"].append("Default Exclusions\n(20 excluded)")
                data["slow_dist_opt"]["times"].append(mean_time)
                data["slow_dist_opt"]["stddevs"].append(stddev)
            elif "all_distributions" in name:
                data["slow_dist_opt"]["labels"].append("All Distributions\n(0 excluded)")
                data["slow_dist_opt"]["times"].append(mean_time)
                data["slow_dist_opt"]["stddevs"].append(stddev)

    return data


def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: y = a * x^b."""
    return a * np.power(x, b)


def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear function: y = a * x + b."""
    return a * x + b


def generate_data_size_chart(data: dict, output_path: Path, backend_label: str = "Spark") -> None:
    """Generate fit time vs data size chart showing sub-linear scaling."""
    sizes = np.array(data["data_size"]["sizes"])
    times = np.array(data["data_size"]["times"])
    stddevs = np.array(data["data_size"]["stddevs"])

    if len(sizes) < 2:
        print("Not enough data points for data size chart")
        return

    # Sort by size
    sort_idx = np.argsort(sizes)
    sizes = sizes[sort_idx]
    times = times[sort_idx]
    stddevs = stddevs[sort_idx]

    # Calculate scaling factors
    data_increase = sizes.max() / sizes.min()
    time_increase = times[-1] / times[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot measured data with error bars and connecting line
    ax.errorbar(
        sizes,
        times,
        yerr=stddevs,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=10,
        linewidth=2,
        label="Measured",
        color="#2563eb",
    )

    # Plot what O(N) linear scaling would look like for comparison
    linear_times = times[0] * (sizes / sizes.min())
    ax.plot(
        sizes,
        linear_times,
        "--",
        linewidth=2,
        label="If O(N) linear",
        color="#dc2626",
        alpha=0.5,
    )

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Data Size (rows)", fontsize=12)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title(f"{backend_label}: Fit Time vs Data Size (Sub-linear Scaling)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(sizes.min() * 0.5, sizes.max() * 2)

    # Add insight annotation
    ax.text(
        0.97,
        0.15,
        f"{data_increase:.0f}× data increase\n{time_increase:.1f}× time increase\n= Sub-linear scaling!",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_distribution_count_chart(data: dict, output_path: Path, backend_label: str = "Spark") -> None:
    """Generate fit time vs distribution count chart."""
    counts = np.array(data["dist_count"]["counts"])
    times = np.array(data["dist_count"]["times"])
    stddevs = np.array(data["dist_count"]["stddevs"])

    if len(counts) < 2:
        print("Not enough data points for distribution count chart")
        return

    # Sort by count
    sort_idx = np.argsort(counts)
    counts = counts[sort_idx]
    times = times[sort_idx]
    stddevs = stddevs[sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot measured data with connecting lines
    ax.errorbar(
        counts,
        times,
        yerr=stddevs,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=10,
        linewidth=2,
        label="Measured",
        color="#2563eb",
    )

    # Calculate slope for fast distributions (first segment up to 50)
    fast_slope = None
    if len(counts) >= 2:
        fast_mask = counts <= 50
        if np.sum(fast_mask) >= 2:
            fast_counts = counts[fast_mask]
            fast_times = times[fast_mask]
            fast_slope = (fast_times[-1] - fast_times[0]) / (fast_counts[-1] - fast_counts[0])

            # Draw trend line for fast distributions
            intercept = fast_times[0] - fast_slope * fast_counts[0]
            ax.plot(
                [0, 60],
                [intercept, intercept + fast_slope * 60],
                "--",
                linewidth=2,
                label=f"Fast dists trend: ~{fast_slope*1000:.0f}ms/dist",
                color="#16a34a",
                alpha=0.7,
            )

    # Formatting
    ax.set_xlabel("Number of Distributions", fontsize=12)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title(f"{backend_label}: Fit Time vs Distribution Count", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 115)
    ax.set_ylim(bottom=0)

    # Add annotation explaining the jump at high distribution counts
    if len(counts) >= 4 and times[-1] > times[-2] * 1.5:
        ax.annotate(
            "Slow distributions\n(burr, t, johnsonsb, etc.)",
            xy=(counts[-1], times[-1]),
            xytext=(counts[-1] - 30, times[-1] * 0.65),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_multi_column_chart(data: dict, output_path: Path, backend_label: str = "Spark") -> None:
    """Generate multi-column efficiency comparison chart."""
    labels = data["multi_column"]["labels"]
    times = np.array(data["multi_column"]["times"])
    stddevs = np.array(data["multi_column"]["stddevs"])

    if len(labels) < 2:
        print("Not enough data points for multi-column chart")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar colors
    colors = ["#dc2626", "#16a34a"]  # Red for separate, green for together

    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(x, times, yerr=stddevs, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{time:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Calculate speedup
    if len(times) >= 2:
        speedup = times[0] / times[1]
        savings_pct = (1 - times[1] / times[0]) * 100

        ax.annotate(
            f"{speedup:.1f}× faster\n({savings_pct:.0f}% time saved)",
            xy=(0.5, max(times) * 0.5),
            fontsize=14,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title(
        f"{backend_label}: Multi-Column Fitting Efficiency\n(3 columns, 10K rows each, 20 distributions)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_slow_dist_opt_chart(data: dict, output_path: Path) -> None:
    """Generate slow distribution optimization comparison chart (v1.7.0)."""
    labels = data["slow_dist_opt"]["labels"]
    times = np.array(data["slow_dist_opt"]["times"])
    stddevs = np.array(data["slow_dist_opt"]["stddevs"])

    if len(labels) < 2:
        print("Not enough data points for slow distribution optimization chart")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bar colors: green for default (fast), red for all (slow)
    colors = ["#16a34a", "#dc2626"]

    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(x, times, yerr=stddevs, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{time:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Calculate speedup
    if len(times) >= 2:
        slowdown = times[1] / times[0]
        savings_pct = (1 - times[0] / times[1]) * 100

        ax.annotate(
            f"Default is {slowdown:.1f}× faster\n({savings_pct:.0f}% time saved)",
            xy=(0.5, max(times) * 0.5),
            fontsize=14,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title(
        "Slow Distribution Optimization (v1.7.0)\n(100K rows, distribution-aware partitioning)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    # Add explanation
    ax.text(
        0.98,
        0.95,
        "Excluded: tukeylambda (~7s),\nnct (~1.4s), dpareto_lognorm (~0.5s)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_spark_vs_ray_data_size_chart(
    spark_data: dict, ray_data: dict, output_path: Path, include_dataset: bool = False
) -> None:
    """Generate Spark vs Ray comparison chart for data size scaling.

    Args:
        spark_data: Spark benchmark data
        ray_data: Ray benchmark data (with optional dataset_data_size key)
        output_path: Path to save chart
        include_dataset: If True, include Ray Dataset line for 3-way comparison
    """
    spark_sizes = np.array(spark_data["data_size"]["sizes"])
    spark_times = np.array(spark_data["data_size"]["times"])
    spark_stddevs = np.array(spark_data["data_size"]["stddevs"])

    ray_sizes = np.array(ray_data["data_size"]["sizes"])
    ray_times = np.array(ray_data["data_size"]["times"])
    ray_stddevs = np.array(ray_data["data_size"]["stddevs"])

    if len(spark_sizes) < 2 or len(ray_sizes) < 2:
        print("Not enough data points for Spark vs Ray data size comparison")
        return

    # Sort by size
    spark_sort = np.argsort(spark_sizes)
    spark_sizes = spark_sizes[spark_sort]
    spark_times = spark_times[spark_sort]
    spark_stddevs = spark_stddevs[spark_sort]

    ray_sort = np.argsort(ray_sizes)
    ray_sizes = ray_sizes[ray_sort]
    ray_times = ray_times[ray_sort]
    ray_stddevs = ray_stddevs[ray_sort]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Spark data
    ax.errorbar(
        spark_sizes,
        spark_times,
        yerr=spark_stddevs,
        fmt="s-",
        capsize=5,
        capthick=2,
        markersize=10,
        linewidth=2,
        label="Spark DataFrame",
        color="#f97316",  # Orange
    )

    # Plot Ray pandas data
    ax.errorbar(
        ray_sizes,
        ray_times,
        yerr=ray_stddevs,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=10,
        linewidth=2,
        label="Ray + pandas",
        color="#8b5cf6",  # Purple
    )

    # Plot Ray Dataset data if available
    if include_dataset and "dataset_data_size" in ray_data:
        ds_sizes = np.array(ray_data["dataset_data_size"]["sizes"])
        ds_times = np.array(ray_data["dataset_data_size"]["times"])
        ds_stddevs = np.array(ray_data["dataset_data_size"]["stddevs"])

        if len(ds_sizes) >= 2:
            ds_sort = np.argsort(ds_sizes)
            ds_sizes = ds_sizes[ds_sort]
            ds_times = ds_times[ds_sort]
            ds_stddevs = ds_stddevs[ds_sort]

            ax.errorbar(
                ds_sizes,
                ds_times,
                yerr=ds_stddevs,
                fmt="^-",
                capsize=5,
                capthick=2,
                markersize=10,
                linewidth=2,
                label="Ray Dataset",
                color="#22c55e",  # Green
            )

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Data Size (rows)", fontsize=12)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    title = "Backend Comparison: Fit Time vs Data Size"
    if include_dataset:
        title += "\n(Distributed vs In-Memory)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_spark_vs_ray_dist_count_chart(
    spark_data: dict, ray_data: dict, output_path: Path, include_dataset: bool = False
) -> None:
    """Generate Spark vs Ray comparison chart for distribution count scaling.

    Args:
        spark_data: Spark benchmark data
        ray_data: Ray benchmark data (with optional dataset_dist_count key)
        output_path: Path to save chart
        include_dataset: If True, include Ray Dataset line for 3-way comparison
    """
    spark_counts = np.array(spark_data["dist_count"]["counts"])
    spark_times = np.array(spark_data["dist_count"]["times"])
    spark_stddevs = np.array(spark_data["dist_count"]["stddevs"])

    ray_counts = np.array(ray_data["dist_count"]["counts"])
    ray_times = np.array(ray_data["dist_count"]["times"])
    ray_stddevs = np.array(ray_data["dist_count"]["stddevs"])

    if len(spark_counts) < 2 or len(ray_counts) < 2:
        print("Not enough data points for Spark vs Ray dist count comparison")
        return

    # Sort by count
    spark_sort = np.argsort(spark_counts)
    spark_counts = spark_counts[spark_sort]
    spark_times = spark_times[spark_sort]
    spark_stddevs = spark_stddevs[spark_sort]

    ray_sort = np.argsort(ray_counts)
    ray_counts = ray_counts[ray_sort]
    ray_times = ray_times[ray_sort]
    ray_stddevs = ray_stddevs[ray_sort]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Spark data
    ax.errorbar(
        spark_counts,
        spark_times,
        yerr=spark_stddevs,
        fmt="s-",
        capsize=5,
        capthick=2,
        markersize=10,
        linewidth=2,
        label="Spark DataFrame",
        color="#f97316",  # Orange
    )

    # Plot Ray pandas data
    ax.errorbar(
        ray_counts,
        ray_times,
        yerr=ray_stddevs,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=10,
        linewidth=2,
        label="Ray + pandas",
        color="#8b5cf6",  # Purple
    )

    # Plot Ray Dataset data if available
    if include_dataset and "dataset_dist_count" in ray_data:
        ds_counts = np.array(ray_data["dataset_dist_count"]["counts"])
        ds_times = np.array(ray_data["dataset_dist_count"]["times"])
        ds_stddevs = np.array(ray_data["dataset_dist_count"]["stddevs"])

        if len(ds_counts) >= 2:
            ds_sort = np.argsort(ds_counts)
            ds_counts = ds_counts[ds_sort]
            ds_times = ds_times[ds_sort]
            ds_stddevs = ds_stddevs[ds_sort]

            ax.errorbar(
                ds_counts,
                ds_times,
                yerr=ds_stddevs,
                fmt="^-",
                capsize=5,
                capthick=2,
                markersize=10,
                linewidth=2,
                label="Ray Dataset",
                color="#22c55e",  # Green
            )

    # Formatting
    ax.set_xlabel("Number of Distributions", fontsize=12)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    title = "Backend Comparison: Fit Time vs Distribution Count"
    if include_dataset:
        title += "\n(Distributed vs In-Memory)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 115)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_backend_overhead_chart(spark_data: dict, ray_data: dict, output_path: Path) -> None:
    """Generate chart comparing startup/overhead between backends.

    Uses 5-distribution benchmark as proxy for startup overhead
    since fitting time is minimal and overhead dominates.
    """
    # Get 5-distribution times as proxy for startup overhead
    spark_5 = None
    ray_5 = None

    for count, time in zip(spark_data["dist_count"]["counts"], spark_data["dist_count"]["times"]):
        if count == 5:
            spark_5 = time
            break

    for count, time in zip(ray_data["dist_count"]["counts"], ray_data["dist_count"]["times"]):
        if count == 5:
            ray_5 = time
            break

    if spark_5 is None or ray_5 is None:
        print("Not enough data for overhead comparison chart")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["SparkBackend", "RayBackend"]
    times = [spark_5, ray_5]
    colors = ["#f97316", "#8b5cf6"]

    bars = ax.bar(labels, times, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{time:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Calculate difference
    if ray_5 < spark_5:
        speedup = spark_5 / ray_5
        ax.annotate(
            f"Ray has {speedup:.1f}× lower\nstartup overhead",
            xy=(0.5, max(times) * 0.5),
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )
    else:
        speedup = ray_5 / spark_5
        ax.annotate(
            f"Spark has {speedup:.1f}× lower\nstartup overhead",
            xy=(0.5, max(times) * 0.5),
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title(
        "Backend Startup Overhead\n(5 distributions, 10K rows - overhead dominates)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(data: dict, results: dict, output_path: Path, backend_label: str = "Spark") -> None:
    """Generate a markdown summary table of benchmark results."""
    # Extract machine info
    machine_info = results.get("machine_info", {})
    cpu_info = machine_info.get("cpu", {})
    cpu_brand = cpu_info.get("brand_raw", "Unknown CPU")
    cpu_count = cpu_info.get("count", "?")
    system = machine_info.get("system", "Unknown")
    python_version = machine_info.get("python_version", "?")

    # Backend-specific info
    if "Ray" in backend_label:
        backend_info = f"| Backend | {backend_label} |"
    else:
        backend_info = "| Spark | local[*] (single node) |"

    lines = [
        f"# {backend_label} Benchmark Results Summary",
        "",
        "## Test Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| CPU | {cpu_brand} |",
        f"| Cores | {cpu_count} |",
        f"| OS | {system} |",
        f"| Python | {python_version} |",
        backend_info,
        "",
        "> **Note:** These benchmarks were run on a local development machine.",
        "> Absolute times will vary based on hardware. The key insight is the",
        "> **scaling characteristics**: sub-linear for data size, O(D) for distribution count.",
        "",
        f"## {backend_label} Data Size Scaling",
        "",
        "| Data Size | Fit Time (mean) | Std Dev |",
        "|-----------|-----------------|---------|",
    ]

    sizes = data["data_size"]["sizes"]
    times = data["data_size"]["times"]
    stddevs = data["data_size"]["stddevs"]

    for size, time, std in sorted(zip(sizes, times, stddevs)):
        size_str = f"{size:,}"
        lines.append(f"| {size_str} | {time:.3f}s | ±{std:.3f}s |")

    lines.extend(
        [
            "",
            f"## {backend_label} Distribution Count Scaling",
            "",
            "| # Distributions | Fit Time (mean) | Std Dev |",
            "|-----------------|-----------------|---------|",
        ]
    )

    counts = data["dist_count"]["counts"]
    times = data["dist_count"]["times"]
    stddevs = data["dist_count"]["stddevs"]

    for count, time, std in sorted(zip(counts, times, stddevs)):
        lines.append(f"| {count} | {time:.3f}s | ±{std:.3f}s |")

    # Multi-column efficiency section
    if data["multi_column"]["labels"]:
        lines.extend(
            [
                "",
                f"## {backend_label} Multi-Column Efficiency",
                "",
                "| Approach | Fit Time (mean) | Std Dev |",
                "|----------|-----------------|---------|",
            ]
        )

        mc_labels = data["multi_column"]["labels"]
        mc_times = data["multi_column"]["times"]
        mc_stddevs = data["multi_column"]["stddevs"]

        for label, time, std in zip(mc_labels, mc_times, mc_stddevs):
            lines.append(f"| {label} | {time:.3f}s | ±{std:.3f}s |")

        if len(mc_times) >= 2:
            speedup = mc_times[0] / mc_times[1]
            savings_pct = (1 - mc_times[1] / mc_times[0]) * 100
            lines.extend(
                [
                    "",
                    f"**Speedup:** {speedup:.1f}× faster ({savings_pct:.0f}% time saved)",
                ]
            )

    # Lazy metrics section
    if data["lazy_metrics"]["labels"]:
        lines.extend(
            [
                "",
                "## Lazy Metrics Performance (v1.5.0+)",
                "",
                "| Mode | Fit Time (mean) | Std Dev | Speedup |",
                "|------|-----------------|---------|---------|",
            ]
        )

        lm_labels = data["lazy_metrics"]["labels"]
        lm_times = data["lazy_metrics"]["times"]
        lm_stddevs = data["lazy_metrics"]["stddevs"]

        eager_time = lm_times[0] if lm_times else 1.0
        for label, time, std in zip(lm_labels, lm_times, lm_stddevs):
            label_clean = label.replace("\n", " ")
            speedup_pct = (1 - time / eager_time) * 100 if eager_time > 0 else 0
            speedup_str = f"-{speedup_pct:.0f}%" if speedup_pct > 0 else "baseline"
            lines.append(f"| {label_clean} | {time:.3f}s | ±{std:.3f}s | {speedup_str} |")

        if len(lm_times) >= 2:
            aic_speedup = (1 - lm_times[1] / lm_times[0]) * 100
            lines.extend(
                [
                    "",
                    f"**AIC-only workflow:** ~{aic_speedup:.0f}% faster than eager fitting",
                ]
            )

    # New exclusions section (v1.7.0)
    lines.extend(
        [
            "",
            "## New Exclusions (v1.7.0)",
            "",
            "Three slow distributions added to `DEFAULT_EXCLUSIONS`:",
            "",
            "- `tukeylambda` (~7s) - ill-conditioned optimization",
            "- `nct` (~1.4s) - non-central t distribution",
            "- `dpareto_lognorm` (~0.5s) - double Pareto-lognormal",
        ]
    )

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def generate_rst_comparison_table(spark_data: dict, ray_data: dict, output_path: Path) -> None:
    """Generate RST-formatted comparison table for performance.rst.

    This generates actual benchmark data that can be included in documentation.
    """
    lines = [
        ".. This file is auto-generated by scripts/generate_scaling_charts.py",
        ".. Run 'make benchmark-charts' to update",
        "",
        "Benchmark Comparison (Auto-generated)",
        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        "",
        "The following benchmarks were run on a local development machine.",
        "",
        "**Data Size Scaling** (90 distributions)",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 20 20 20 20 20",
        "",
        "   * - Data Size",
        "     - Spark",
        "     - Ray + pandas",
        "     - Ray Dataset",
        "     - Fastest",
    ]

    # Get data size benchmarks
    spark_sizes = dict(zip(spark_data["data_size"]["sizes"], spark_data["data_size"]["times"]))
    ray_sizes = dict(zip(ray_data["data_size"]["sizes"], ray_data["data_size"]["times"]))
    ds_sizes = dict(
        zip(
            ray_data.get("dataset_data_size", {}).get("sizes", []),
            ray_data.get("dataset_data_size", {}).get("times", []),
        )
    )

    for size in sorted(set(spark_sizes.keys()) | set(ray_sizes.keys())):
        spark_t = spark_sizes.get(size)
        ray_t = ray_sizes.get(size)
        ds_t = ds_sizes.get(size)

        spark_str = f"{spark_t:.2f}s" if spark_t else "—"
        ray_str = f"{ray_t:.2f}s" if ray_t else "—"
        ds_str = f"{ds_t:.2f}s" if ds_t else "—"

        # Find fastest
        times = [(spark_t, "Spark"), (ray_t, "Ray+pandas"), (ds_t, "Ray Dataset")]
        valid_times = [(t, n) for t, n in times if t is not None]
        if valid_times:
            fastest = min(valid_times, key=lambda x: x[0])
            fastest_str = f"**{fastest[1]}**"
        else:
            fastest_str = "—"

        size_str = f"{size:,}"
        lines.append(f"   * - {size_str}")
        lines.append(f"     - {spark_str}")
        lines.append(f"     - {ray_str}")
        lines.append(f"     - {ds_str}")
        lines.append(f"     - {fastest_str}")

    lines.extend(
        [
            "",
            "**Distribution Count Scaling** (10K rows)",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 20 20 20 20 20",
            "",
            "   * - # Distributions",
            "     - Spark",
            "     - Ray + pandas",
            "     - Ray Dataset",
            "     - Fastest",
        ]
    )

    # Get distribution count benchmarks
    spark_dists = dict(zip(spark_data["dist_count"]["counts"], spark_data["dist_count"]["times"]))
    ray_dists = dict(zip(ray_data["dist_count"]["counts"], ray_data["dist_count"]["times"]))
    ds_dists = dict(
        zip(
            ray_data.get("dataset_dist_count", {}).get("counts", []),
            ray_data.get("dataset_dist_count", {}).get("times", []),
        )
    )

    for count in sorted(set(spark_dists.keys()) | set(ray_dists.keys()) | set(ds_dists.keys())):
        spark_t = spark_dists.get(count)
        ray_t = ray_dists.get(count)
        ds_t = ds_dists.get(count)

        spark_str = f"{spark_t:.2f}s" if spark_t else "—"
        ray_str = f"{ray_t:.2f}s" if ray_t else "—"
        ds_str = f"{ds_t:.2f}s" if ds_t else "—"

        # Find fastest
        times = [(spark_t, "Spark"), (ray_t, "Ray+pandas"), (ds_t, "Ray Dataset")]
        valid_times = [(t, n) for t, n in times if t is not None]
        if valid_times:
            fastest = min(valid_times, key=lambda x: x[0])
            fastest_str = f"**{fastest[1]}**"
        else:
            fastest_str = "—"

        lines.append(f"   * - {count}")
        lines.append(f"     - {spark_str}")
        lines.append(f"     - {ray_str}")
        lines.append(f"     - {ds_str}")
        lines.append(f"     - {fastest_str}")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scaling charts from benchmarks")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(".benchmarks"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/_static"),
        help="Output directory for charts",
    )
    parser.add_argument(
        "--backend",
        choices=["spark", "ray", "all"],
        default="all",
        help="Which backend(s) to generate charts for",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    spark_data = None
    ray_data = None
    spark_results = None
    ray_results = None

    # Load Spark results
    if args.backend in ("spark", "all"):
        spark_results = load_benchmark_results(args.results_dir, backend="spark")
        if spark_results:
            spark_data = extract_scaling_data(spark_results)
            # Generate markdown summary for reference
            generate_summary_table(
                spark_data, spark_results, args.output_dir / "benchmark_summary.md", backend_label="Spark"
            )
        elif args.backend == "spark":
            print("Run 'make benchmark' first to generate Spark results")
            return 1

    # Load Ray results
    if args.backend in ("ray", "all"):
        ray_results = load_benchmark_results(args.results_dir, backend="ray")
        if ray_results:
            # Extract with include_dataset=True to get Ray Dataset benchmarks
            ray_data = extract_scaling_data(ray_results, include_dataset=True)
            # Generate markdown summary for reference
            generate_summary_table(
                ray_data, ray_results, args.output_dir / "ray_benchmark_summary.md", backend_label="Ray + pandas"
            )
        elif args.backend == "ray":
            print("Run 'make benchmark-ray' first to generate Ray results")
            return 1

    # Generate minimal chart set (modern style)
    if spark_data and ray_data:
        print("\nGenerating minimal chart set (modern style)...")

        # 1. Backend comparison: Data size scaling
        generate_backend_comparison_chart(
            spark_data, ray_data, args.output_dir / "backend_data_size.png", chart_type="data_size"
        )

        # 2. Backend comparison: Distribution count scaling
        generate_backend_comparison_chart(
            spark_data, ray_data, args.output_dir / "backend_dist_count.png", chart_type="dist_count"
        )

        # 3. Backend overhead explanation
        generate_overhead_chart(spark_data, ray_data, args.output_dir / "backend_overhead.png")

        # 4. Lazy metrics performance (Spark data)
        generate_lazy_metrics_chart(spark_data, args.output_dir / "lazy_metrics.png")

        # 5. Auto-generated RST comparison table
        generate_rst_comparison_table(spark_data, ray_data, args.output_dir / "benchmark_comparison.rst")

        # 6. Version history chart (always generate - uses documented values)
        generate_version_history_chart(args.output_dir / "version_history.png")

    elif spark_data:
        print("\nGenerating Spark-only charts...")
        # Just lazy metrics for Spark-only
        generate_lazy_metrics_chart(spark_data, args.output_dir / "lazy_metrics.png")

    if spark_data or ray_data:
        print("\nCharts generated successfully!")
        return 0
    else:
        print("No benchmark results found. Run 'make benchmark' or 'make benchmark-ray' first.")
        return 1


if __name__ == "__main__":
    exit(main())
