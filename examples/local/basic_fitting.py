#!/usr/bin/env python3
"""Basic distribution fitting with LocalBackend.

This example demonstrates how to use spark-bestfit without Spark,
using the LocalBackend for thread-based parallel fitting.

Usage:
    python basic_fitting.py
"""

import numpy as np
import pandas as pd

from spark_bestfit import DistributionFitter, LocalBackend, console_progress


def main():
    # Generate sample data (normal distribution with some noise)
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=50, scale=10, size=800),
        np.random.exponential(scale=5, size=200),
    ])

    # Create a pandas DataFrame (LocalBackend works with pandas)
    df = pd.DataFrame({"value": data})
    print(f"Dataset: {len(df)} samples")
    print(f"  Min: {df['value'].min():.2f}")
    print(f"  Max: {df['value'].max():.2f}")
    print(f"  Mean: {df['value'].mean():.2f}")
    print()

    # Create LocalBackend - no Spark required!
    backend = LocalBackend(max_workers=4)

    # Create fitter with the local backend
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
              f"AIC={result.aic:.1f}  SSE={result.sse:.4f}")

    # Get the best result
    best = results.best(n=1)[0]
    print(f"\nBest fit: {best.distribution}")
    print(f"  Parameters: {best.parameters}")

    # Generate samples from the fitted distribution
    samples = best.sample(100)
    print(f"\nGenerated 100 samples from {best.distribution}:")
    print(f"  Sample mean: {samples.mean():.2f}")
    print(f"  Sample std: {samples.std():.2f}")

    # You can also use different metrics for ranking
    print("\nTop 3 by AIC (penalizes complexity):")
    for result in results.best(n=3, metric="aic"):
        print(f"  {result.distribution}: AIC={result.aic:.1f}")


if __name__ == "__main__":
    main()
