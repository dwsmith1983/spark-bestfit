#!/usr/bin/env python3
"""Generate copula sampling performance chart for v2.8.0.

Creates a chart showing:
- Cholesky caching speedup
- ndtr vs norm.cdf speedup
- Overall copula.sample() performance at different scales

Usage:
    python scripts/generate_copula_chart.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def apply_modern_style():
    """Apply modern minimal style to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#e2e8f0",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.labelcolor": "#334155",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )


def generate_copula_optimization_chart(output_path: Path):
    """Generate chart showing v2.8.0 copula optimizations."""
    apply_modern_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Data from benchmarks (in ms)
    sample_sizes = ["1K", "10K", "100K", "1M"]

    # Chart 1: Cholesky caching speedup
    ax1 = axes[0]
    old_cholesky = [0.028, 0.15, 1.2, 12.1]  # multivariate_normal
    new_cholesky = [0.010, 0.11, 0.97, 9.3]  # cached cholesky

    x = np.arange(len(sample_sizes))
    width = 0.35

    ax1.bar(x - width / 2, old_cholesky, width, label="Old (recompute)", color="#94a3b8")
    ax1.bar(x + width / 2, new_cholesky, width, label="New (cached)", color="#22c55e")

    ax1.set_xlabel("Sample Size")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Cholesky Decomposition: Caching Speedup")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_sizes)
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Add speedup labels
    for i, (old, new) in enumerate(zip(old_cholesky, new_cholesky)):
        speedup = old / new
        ax1.annotate(
            f"{speedup:.1f}×",
            xy=(i, max(old, new) + 0.5),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#22c55e",
        )

    # Chart 2: CDF transformation speedup
    ax2 = axes[1]
    sizes_cdf = ["100K", "1M"]
    old_cdf = [2.4, 24.4]  # norm.cdf
    new_cdf = [1.7, 16.9]  # ndtr

    x2 = np.arange(len(sizes_cdf))

    ax2.bar(x2 - width / 2, old_cdf, width, label="norm.cdf", color="#94a3b8")
    ax2.bar(x2 + width / 2, new_cdf, width, label="ndtr", color="#8b5cf6")

    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("CDF Transformation: ndtr vs norm.cdf")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sizes_cdf)
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # Add speedup labels
    for i, (old, new) in enumerate(zip(old_cdf, new_cdf)):
        speedup = old / new
        ax2.annotate(
            f"{speedup:.1f}×",
            xy=(i, max(old, new) + 1),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#8b5cf6",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def generate_copula_sample_chart(output_path: Path):
    """Generate chart showing copula.sample() performance with fast_ppf."""
    apply_modern_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes = ["1K", "10K", "100K", "1M"]
    x = np.arange(len(sample_sizes))
    width = 0.35

    # Data from benchmarks
    fast_ppf_times = [0.19, 1.9, 19.8, 199]  # ms
    scipy_fallback_times = [0.30, 2.8, 26.7, 269]  # ms

    bars1 = ax.bar(x - width / 2, fast_ppf_times, width, label="With fast_ppf", color="#22c55e")
    bars2 = ax.bar(x + width / 2, scipy_fallback_times, width, label="scipy fallback", color="#f97316")

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Copula Sampling: fast_ppf vs scipy Fallback")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_sizes)
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label = f"{height:.1f}" if height >= 1 else f"{height:.2f}"
            ax.annotate(
                f"{label}ms",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path("docs/_static")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_copula_optimization_chart(output_dir / "copula_v280_optimizations.png")
    generate_copula_sample_chart(output_dir / "copula_sampling_performance.png")

    print("\nCopula charts generated successfully!")


if __name__ == "__main__":
    main()
