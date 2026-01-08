#!/usr/bin/env python3
"""Multi-column fitting with LocalBackend.

This example demonstrates fitting distributions to multiple columns
in a single operation using the LocalBackend.

Usage:
    python multi_column_fitting.py
"""

import numpy as np
import pandas as pd

from spark_bestfit import DistributionFitter, LocalBackend, FitterConfigBuilder


def main():
    # Generate sample data with different distributions per column
    np.random.seed(42)
    df = pd.DataFrame({
        "revenue": np.random.lognormal(mean=10, sigma=1, size=1000),
        "delay_ms": np.random.exponential(scale=50, size=1000),
        "score": np.random.beta(a=2, b=5, size=1000) * 100,
    })

    print("Dataset summary:")
    print(df.describe())
    print()

    # Create backend and fitter
    backend = LocalBackend(max_workers=4)
    fitter = DistributionFitter(backend=backend)

    # Configure fitting with FitterConfigBuilder (v2.2+)
    config = (
        FitterConfigBuilder()
        .with_bins(50)
        .with_sampling(fraction=0.5, seed=42)
        .with_max_distributions(20)  # Limit for faster execution
        .build()
    )

    # Fit all columns in one call
    print("Fitting distributions to all columns...")
    results = fitter.fit(
        df,
        columns=["revenue", "delay_ms", "score"],
        config=config,
    )

    # Results contain fits for all columns
    print(f"\nTotal fit results: {len(results)}")

    # Get best fit for each column
    for col in ["revenue", "delay_ms", "score"]:
        col_results = results.for_column(col)
        best = col_results.best(n=1)[0]
        print(f"\n{col}:")
        print(f"  Best distribution: {best.distribution}")
        print(f"  KS statistic: {best.ks_statistic:.4f}")
        print(f"  Parameters: {best.parameters}")

    # Compare specific distributions across columns
    print("\n" + "=" * 60)
    print("Comparing top 3 fits per column:")
    print("=" * 60)

    for col in ["revenue", "delay_ms", "score"]:
        col_results = results.for_column(col)
        print(f"\n{col}:")
        for i, r in enumerate(col_results.best(n=3), 1):
            print(f"  {i}. {r.distribution:12} KS={r.ks_statistic:.4f}")


if __name__ == "__main__":
    main()
