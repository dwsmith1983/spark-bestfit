#!/usr/bin/env python3
"""Distributed fitting with Ray Datasets.

This example demonstrates using RayBackend with Ray Datasets for
large-scale distributed data processing. Ray Datasets allow you to
work with data larger than memory across a Ray cluster.

Requirements:
    pip install spark-bestfit[ray]

Usage:
    python distributed_fitting.py

For cluster mode:
    ray start --head  # On head node
    python distributed_fitting.py  # Connects to existing cluster
"""

import numpy as np
import pandas as pd
import ray

from spark_bestfit import DistributionFitter, RayBackend, FitterConfigBuilder


def main():
    # Initialize Ray (or connect to existing cluster)
    # Use address="auto" to connect to an existing cluster
    if not ray.is_initialized():
        ray.init()

    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print()

    # Generate a larger dataset
    np.random.seed(42)
    n_samples = 100_000

    # Simulate a mixture: mostly normal, some outliers
    normal_data = np.random.normal(loc=100, scale=15, size=int(n_samples * 0.9))
    outlier_data = np.random.exponential(scale=50, size=int(n_samples * 0.1)) + 150

    data = np.concatenate([normal_data, outlier_data])
    np.random.shuffle(data)

    # Create Ray Dataset for distributed processing
    # In production, you might read from Parquet, CSV, or other sources:
    #   ray.data.read_parquet("s3://bucket/data/")
    df = ray.data.from_pandas(pd.DataFrame({"value": data}))
    print(f"Created Ray Dataset with {df.count()} rows")
    print(f"  Blocks: {df.num_blocks()}")
    print()

    # Create RayBackend
    backend = RayBackend()

    # Configure fitting for large data
    config = (
        FitterConfigBuilder()
        .with_bins(100)  # More bins for larger dataset
        .with_sampling(fraction=0.1, seed=42)  # Sample 10% for fitting
        .with_lazy_metrics(True)  # Defer expensive KS/AD computation
        .build()
    )

    # Create fitter
    fitter = DistributionFitter(backend=backend)

    # Fit distributions
    print("Fitting distributions to Ray Dataset...")
    results = fitter.fit(
        df,
        column="value",
        config=config,
    )

    # Show results (lazy metrics are computed on access)
    print("\nTop 5 distributions:")
    print("-" * 60)
    for i, result in enumerate(results.best(n=5), 1):
        print(f"{i}. {result.distribution:15} AIC={result.aic:.1f}  "
              f"BIC={result.bic:.1f}  SSE={result.sse:.4f}")

    # Get best fit
    best = results.best(n=1, metric="aic")[0]
    print(f"\nBest fit (by AIC): {best.distribution}")
    print(f"  Parameters: {best.parameters}")

    # Access lazy metrics (triggers computation)
    print(f"\nKS statistic: {best.ks_statistic:.4f}")
    print(f"AD statistic: {best.ad_statistic:.4f}")

    # Demonstrate saving/loading results
    print("\nSaving results to JSON...")
    best.save("best_fit.json", format="json")
    print("Saved to best_fit.json")

    # Clean up
    import os
    if os.path.exists("best_fit.json"):
        os.remove("best_fit.json")
        print("Cleaned up temporary file")


if __name__ == "__main__":
    main()
