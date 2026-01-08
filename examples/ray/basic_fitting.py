#!/usr/bin/env python3
"""Basic distribution fitting with RayBackend.

This example demonstrates how to use spark-bestfit with Ray
for distributed parallel fitting.

Requirements:
    pip install spark-bestfit[ray]

Usage:
    python basic_fitting.py
"""

import numpy as np
import pandas as pd

from spark_bestfit import DistributionFitter, RayBackend, console_progress


def main():
    # Generate sample data
    np.random.seed(42)
    data = np.random.gamma(shape=2, scale=10, size=5000)

    # Create a pandas DataFrame (RayBackend works with pandas or Ray Datasets)
    df = pd.DataFrame({"value": data})
    print(f"Dataset: {len(df)} samples")
    print(f"  Min: {df['value'].min():.2f}")
    print(f"  Max: {df['value'].max():.2f}")
    print(f"  Mean: {df['value'].mean():.2f}")
    print()

    # Create RayBackend - auto-initializes Ray if not running
    # For local development, this starts a local Ray instance
    backend = RayBackend()
    print(f"Ray backend initialized with {backend.get_parallelism()} CPUs")
    print()

    # Create fitter with the Ray backend
    fitter = DistributionFitter(backend=backend)

    # Fit distributions with progress tracking
    print("Fitting distributions...")
    results = fitter.fit(
        df,
        column="value",
        progress_callback=console_progress,
    )

    # Get top 5 best-fitting distributions
    print("\nTop 5 distributions (by KS statistic):")
    print("-" * 60)
    for i, result in enumerate(results.best(n=5), 1):
        print(f"{i}. {result.distribution:15} KS={result.ks_statistic:.4f}  "
              f"AIC={result.aic:.1f}")

    # Get the best result
    best = results.best(n=1)[0]
    print(f"\nBest fit: {best.distribution}")
    print(f"  Parameters: {best.parameters}")

    # Verify the fit
    if best.distribution == "gamma":
        print("\n  Expected: gamma (data was generated from gamma distribution)")

    # Generate samples
    samples = best.sample(100)
    print(f"\nGenerated 100 samples from {best.distribution}:")
    print(f"  Sample mean: {samples.mean():.2f} (original: {data.mean():.2f})")


if __name__ == "__main__":
    main()
