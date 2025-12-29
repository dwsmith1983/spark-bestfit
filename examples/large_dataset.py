"""Example demonstrating performance with large datasets.

Memory Usage Patterns
=====================

spark-bestfit is designed to minimize memory usage on the driver node:

1. **Histogram Computation** (~1KB per column):
   - Only histogram bin counts are collected to driver
   - For 100 bins: ~800 bytes (100 floats × 8 bytes)
   - Raw data stays distributed in Spark executors

2. **Fitting Sample** (~80KB-800KB):
   - A small sample (default 10,000 rows) is collected for scipy fitting
   - 10,000 rows × 8 bytes = ~80KB
   - This is configurable via FITTING_SAMPLE_SIZE constant

3. **Broadcast Variables** (~1KB):
   - Histogram and sample are broadcast to executors
   - Overhead is minimal as data is already small

4. **Results DataFrame**:
   - ~100 distributions × ~50 bytes each = ~5KB
   - Kept as Spark DataFrame until explicitly collected

Memory Comparison (100M row dataset):
-------------------------------------
- Traditional approach (collect all): ~800MB on driver
- spark-bestfit approach: ~1MB on driver (1000x reduction)

For very large datasets (>100M rows), enable sampling to reduce executor memory:
    fitter.fit(
        df,
        column="value",
        enable_sampling=True,
        sample_threshold=10_000_000,  # Sample when > 10M rows
        max_sample_size=1_000_000,    # Sample down to 1M rows
    )
"""

import time

import numpy as np
from pyspark.sql import SparkSession

from spark_bestfit import DistributionFitter

# Create Spark session with more resources
spark = (
    SparkSession.builder.appName("LargeDatasetFitting")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

print("=" * 80)
print("LARGE DATASET PERFORMANCE DEMO")
print("=" * 80)

# ============================================================================
# Test with varying data sizes
# ============================================================================
data_sizes = [100_000, 1_000_000, 10_000_000]

for size in data_sizes:
    print(f"\n{'=' * 80}")
    print(f"Dataset size: {size:,} rows")
    print("=" * 80)

    # Generate data
    print("Generating data...")
    np.random.seed(42)
    data = np.random.gamma(shape=2.0, scale=2.0, size=size)

    # Create DataFrame
    df = spark.createDataFrame([(float(x),) for x in data], ["value"])
    df = df.cache()  # Cache for consistent timing
    df.count()  # Materialize

    # Fit distributions
    # Note: DistributionFitter uses DEFAULT_EXCLUDED_DISTRIBUTIONS which excludes
    # slow distributions like studentized_range, gausshyper, geninvgauss, etc.
    print(f"\nFitting distributions to {size:,} rows...")
    start_time = time.time()

    fitter = DistributionFitter(spark)
    results = fitter.fit(
        df,
        column="value",
        bins=50,  # Fewer bins for speed
        enable_sampling=True,  # Enable sampling for large datasets
        sample_fraction=None,  # Auto-determine
        max_sample_size=1_000_000,  # Limit to 1M for fitting
        sample_threshold=1_000_000,  # Sample when exceeding 1M rows
    )

    elapsed = time.time() - start_time

    # Get results
    best = results.best(n=1)[0]
    num_fitted = results.count()

    print(f"\n{'Results':^80}")
    print("-" * 80)
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Distributions fitted: {num_fitted}")
    print(f"Rows per second: {size / elapsed:,.0f}")
    print(f"\nBest distribution: {best.distribution}")
    print(f"K-S statistic: {best.ks_statistic:.6f}")
    print(f"p-value: {best.pvalue:.4f}")
    print(f"A-D statistic: {best.ad_statistic:.6f}")
    ad_pval = f"{best.ad_pvalue:.4f}" if best.ad_pvalue else "N/A"
    print(f"A-D p-value: {ad_pval}")
    print(f"Parameters: {[f'{p:.4f}' for p in best.parameters]}")

    # Clean up
    df.unpersist()

# ============================================================================
# Multi-Column Fitting Demo
# ============================================================================
print(f"\n{'=' * 80}")
print("MULTI-COLUMN FITTING DEMO")
print("=" * 80)

print("\nGenerating multi-column dataset...")
np.random.seed(42)
size = 100_000

# Create DataFrame with multiple columns (different distributions)
df_multi = spark.createDataFrame(
    [
        (float(np.random.gamma(2.0, 2.0)), float(np.random.normal(50, 10)), float(np.random.exponential(5)))
        for _ in range(size)
    ],
    ["gamma_col", "normal_col", "exponential_col"],
)
df_multi = df_multi.cache()
df_multi.count()

# Fit all columns in a single call
print(f"\nFitting 3 columns simultaneously ({size:,} rows each)...")
start_time = time.time()

fitter = DistributionFitter(spark)
results = fitter.fit(df_multi, columns=["gamma_col", "normal_col", "exponential_col"])

elapsed = time.time() - start_time

print(f"\n{'Multi-Column Results':^80}")
print("-" * 80)
print(f"Time elapsed: {elapsed:.2f} seconds")
print(f"Columns fitted: {results.column_names}")

# Get best distribution per column
best_per_col = results.best_per_column(n=1)
for col_name, fits in best_per_col.items():
    best = fits[0]
    print(f"\n{col_name}:")
    print(f"  Best distribution: {best.distribution}")
    print(f"  K-S statistic: {best.ks_statistic:.6f}")
    print(f"  p-value: {best.pvalue:.4f}")

# Demonstrate column filtering
print("\nFiltering results for 'normal_col' only:")
normal_results = results.for_column("normal_col")
top_3 = normal_results.best(n=3)
for i, fit in enumerate(top_3, 1):
    print(f"  {i}. {fit.distribution} (KS={fit.ks_statistic:.4f})")

df_multi.unpersist()

print("\n" + "=" * 80)
print("PERFORMANCE DEMO COMPLETED")
print("=" * 80)

spark.stop()
