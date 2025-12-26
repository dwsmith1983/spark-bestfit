# Benchmark Results Summary

## Test Environment

| Property | Value |
|----------|-------|
| CPU | Apple M5 |
| Cores | 10 |
| OS | Darwin |
| Python | 3.13.11 |
| Spark | local[*] (single node) |

> **Note:** These benchmarks were run on a local development machine.
> Absolute times will vary based on hardware. The key insight is the
> **scaling characteristics**: sub-linear for data size, O(D) for distribution count.

## Data Size Scaling

| Data Size | Fit Time (mean) | Std Dev |
|-----------|-----------------|---------|
| 25,000 | 9.629s | ±0.028s |
| 100,000 | 12.894s | ±0.029s |
| 500,000 | 12.464s | ±0.030s |
| 1,000,000 | 15.502s | ±0.028s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.378s | ±0.009s |
| 20 | 1.381s | ±0.021s |
| 50 | 1.968s | ±0.044s |
| 100 | 16.214s | ±0.033s |
