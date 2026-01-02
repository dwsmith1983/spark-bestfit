#!/usr/bin/env python3
"""Generate scaling charts from benchmark results.

This script reads pytest-benchmark JSON results and generates
publication-quality scaling charts for documentation.

Usage:
    python scripts/generate_scaling_charts.py [--results-dir PATH] [--output-dir PATH]

Example:
    # Run benchmarks first
    make benchmark

    # Generate charts from latest results
    python scripts/generate_scaling_charts.py
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(results_dir: Path) -> Optional[dict]:
    """Load the latest benchmark results JSON file."""
    # pytest-benchmark saves results in .benchmarks/ directory
    benchmark_files = list(results_dir.glob("**/*.json"))
    if not benchmark_files:
        print(f"No benchmark results found in {results_dir}")
        return None

    # Get the most recent file
    latest = max(benchmark_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from: {latest}")

    with open(latest) as f:
        return json.load(f)


def extract_scaling_data(results: dict) -> dict:
    """Extract scaling data from benchmark results."""
    data: dict[str, dict[str, list]] = {
        "data_size": {"sizes": [], "times": [], "stddevs": []},
        "dist_count": {"counts": [], "times": [], "stddevs": []},
        "multi_column": {"labels": [], "times": [], "stddevs": []},
        "lazy_metrics": {"labels": [], "times": [], "stddevs": []},
        "slow_dist_opt": {"labels": [], "times": [], "stddevs": []},  # v1.7.0
    }

    for benchmark in results.get("benchmarks", []):
        name = benchmark["name"]
        fullname = benchmark.get("fullname", name)  # For class-based matching
        mean_time = benchmark["stats"]["mean"]
        stddev = benchmark["stats"]["stddev"]

        # Parse data size benchmarks
        if "25k_rows" in name:
            data["data_size"]["sizes"].append(25_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)
        elif "100k_rows" in name and "lazy" not in name:
            data["data_size"]["sizes"].append(100_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)
        elif "500k_rows" in name:
            data["data_size"]["sizes"].append(500_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)
        elif "1m_rows" in name:
            data["data_size"]["sizes"].append(1_000_000)
            data["data_size"]["times"].append(mean_time)
            data["data_size"]["stddevs"].append(stddev)

        # Parse distribution count benchmarks
        if "5_distributions" in name:
            data["dist_count"]["counts"].append(5)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "20_distributions" in name:
            data["dist_count"]["counts"].append(20)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "50_distributions" in name:
            data["dist_count"]["counts"].append(50)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)
        elif "all_distributions" in name and "discrete" not in name and "SlowDistributionOptimizations" not in fullname:
            data["dist_count"]["counts"].append(100)
            data["dist_count"]["times"].append(mean_time)
            data["dist_count"]["stddevs"].append(stddev)

        # Parse multi-column efficiency benchmarks
        if "3_columns_separately" in name and "discrete" not in name:
            data["multi_column"]["labels"].append("3 Separate Fits")
            data["multi_column"]["times"].append(mean_time)
            data["multi_column"]["stddevs"].append(stddev)
        elif "3_columns_together" in name and "100k" not in name and "discrete" not in name:
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


def generate_data_size_chart(data: dict, output_path: Path) -> None:
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
    ax.set_title("spark-bestfit: Fit Time vs Data Size (Sub-linear Scaling)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(sizes.min() * 0.5, sizes.max() * 2)

    # Add insight annotation
    ax.text(
        0.95,
        0.05,
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


def generate_distribution_count_chart(data: dict, output_path: Path) -> None:
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
    ax.set_title("spark-bestfit: Fit Time vs Distribution Count", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 110)
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


def generate_multi_column_chart(data: dict, output_path: Path) -> None:
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
        "Multi-Column Fitting Efficiency\n(3 columns, 10K rows each, 20 distributions)", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_lazy_metrics_chart(data: dict, output_path: Path) -> None:
    """Generate lazy metrics performance comparison chart."""
    labels = data["lazy_metrics"]["labels"]
    times = np.array(data["lazy_metrics"]["times"])
    stddevs = np.array(data["lazy_metrics"]["stddevs"])

    if len(labels) < 2:
        print("Not enough data points for lazy metrics chart")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar colors: eager=red, lazy variants=shades of green
    colors = ["#dc2626", "#22c55e", "#16a34a", "#15803d"]

    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(x, times, yerr=stddevs, capsize=8, color=colors[: len(labels)], edgecolor="black", linewidth=1.5)

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
            fontsize=11,
            fontweight="bold",
        )

    # Calculate and display speedups relative to eager
    if len(times) >= 2:
        eager_time = times[0]
        for i, (label, time) in enumerate(zip(labels[1:], times[1:]), 1):
            speedup_pct = (1 - time / eager_time) * 100
            if speedup_pct > 0:
                ax.annotate(
                    f"-{speedup_pct:.0f}%",
                    xy=(i, time / 2),
                    fontsize=10,
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title("Lazy Metrics Performance (v1.5.0+)\n(100K rows, 50 distributions)", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    # Add legend/explanation
    ax.text(
        0.98,
        0.95,
        "Green = lazy_metrics=True\nRed = lazy_metrics=False",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

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


def generate_summary_table(data: dict, results: dict, output_path: Path) -> None:
    """Generate a markdown summary table of benchmark results."""
    # Extract machine info
    machine_info = results.get("machine_info", {})
    cpu_info = machine_info.get("cpu", {})
    cpu_brand = cpu_info.get("brand_raw", "Unknown CPU")
    cpu_count = cpu_info.get("count", "?")
    system = machine_info.get("system", "Unknown")
    python_version = machine_info.get("python_version", "?")

    lines = [
        "# Benchmark Results Summary",
        "",
        "## Test Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| CPU | {cpu_brand} |",
        f"| Cores | {cpu_count} |",
        f"| OS | {system} |",
        f"| Python | {python_version} |",
        "| Spark | local[*] (single node) |",
        "",
        "> **Note:** These benchmarks were run on a local development machine.",
        "> Absolute times will vary based on hardware. The key insight is the",
        "> **scaling characteristics**: sub-linear for data size, O(D) for distribution count.",
        "",
        "## Data Size Scaling",
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
            "## Distribution Count Scaling",
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
                "## Multi-Column Efficiency",
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
    args = parser.parse_args()

    # Load results
    results = load_benchmark_results(args.results_dir)
    if results is None:
        print("Run 'make benchmark' first to generate results")
        return 1

    # Extract data
    data = extract_scaling_data(results)

    # Generate charts
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_data_size_chart(data, args.output_dir / "scaling_data_size.png")
    generate_distribution_count_chart(data, args.output_dir / "scaling_dist_count.png")
    generate_multi_column_chart(data, args.output_dir / "multi_column_efficiency.png")
    # Note: slow_dist_opt chart removed - benchmark doesn't show meaningful difference
    # because excluded_distributions=() doesn't override registry's DEFAULT_EXCLUSIONS.
    # Note: lazy_metrics chart removed - wall-clock comparison doesn't capture the
    # real benefit (skipping 95% of computations). See docs/performance.rst for details.
    generate_summary_table(data, results, args.output_dir / "benchmark_summary.md")

    print("\nCharts generated successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
