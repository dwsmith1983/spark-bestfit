"""Basic usage example for spark-bestfit."""

import numpy as np
from pyspark.sql import SparkSession

from spark_bestfit import DistributionFitter

# Create Spark session (user's responsibility)
spark = SparkSession.builder.appName("DistributionFitting").config("spark.sql.shuffle.partitions", "10").getOrCreate()

# Generate sample data from a known distribution (normal distribution)
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100_000)

# Create Spark DataFrame
df = spark.createDataFrame([(float(x),) for x in data], ["value"])

print("=" * 80)
print("BASIC USAGE EXAMPLE")
print("=" * 80)

# ============================================================================
# Example 1: Simple fitting with defaults
# ============================================================================
print("\n1. Simple fitting with defaults")
print("-" * 80)

fitter = DistributionFitter(spark)
results = fitter.fit(df, column="value")

# Get best distribution
best = results.best(n=1)[0]
print(f"\nBest distribution: {best.distribution}")
print(f"SSE: {best.sse:.6f}")
print(f"AIC: {best.aic:.2f}")
print(f"BIC: {best.bic:.2f}")
print(f"Parameters: {[f'{p:.4f}' for p in best.parameters]}")

# ============================================================================
# Example 2: Get top 5 distributions
# ============================================================================
print("\n2. Top 5 distributions by SSE")
print("-" * 80)

top_5 = results.best(n=5, metric="sse")
for i, result in enumerate(top_5, 1):
    print(f"{i}. {result.distribution:20s} SSE={result.sse:.6f}")

# ============================================================================
# Example 3: Custom parameters (non-negative distributions only)
# ============================================================================
print("\n3. Custom parameters (non-negative distributions only)")
print("-" * 80)

# Generate non-negative data (exponential distribution)
data_pos = np.random.exponential(scale=5, size=100_000)
df_pos = spark.createDataFrame([(float(x),) for x in data_pos], ["value"])

fitter_custom = DistributionFitter(spark)
results_custom = fitter_custom.fit(
    df_pos,
    column="value",
    bins=100,  # More bins for better resolution
    support_at_zero=True,  # Only non-negative distributions
    enable_sampling=True,
    sample_fraction=0.3,
)

best_custom = results_custom.best(n=1)[0]
print(f"\nBest non-negative distribution: {best_custom.distribution}")
print(f"SSE: {best_custom.sse:.6f}")

# ============================================================================
# Example 4: Plotting
# ============================================================================
print("\n4. Plotting best fit")
print("-" * 80)

fitter.plot(
    best,
    df,
    "value",
    figsize=(14, 8),
    dpi=100,
    title="Best Fit Distribution (Normal Data)",
    xlabel="Value",
    ylabel="Density",
)

print("Plot displayed!")

# ============================================================================
# Example 5: Using fitted distribution
# ============================================================================
print("\n5. Using fitted distribution")
print("-" * 80)

# Generate samples from fitted distribution
samples = best.sample(size=1000, random_state=42)
print(f"Generated {len(samples)} samples from fitted {best.distribution}")
print(f"Sample mean: {samples.mean():.2f} (original: {data.mean():.2f})")
print(f"Sample std: {samples.std():.2f} (original: {data.std():.2f})")

# Evaluate PDF at specific points
x = np.array([30, 40, 50, 60, 70])
pdf_values = best.pdf(x)
print(f"\nPDF values at {x}:")
for xi, pdf in zip(x, pdf_values):
    print(f"  f({xi}) = {pdf:.6f}")

# ============================================================================
# Example 6: Using active session
# ============================================================================
print("\n6. Using active SparkSession")
print("-" * 80)

# DistributionFitter can use the active session automatically
fitter_active = DistributionFitter()  # No spark parameter needed
print(f"Using session: {fitter_active.spark.sparkContext.appName}")

print("\n" + "=" * 80)
print("Examples completed!")
print("=" * 80)

spark.stop()
