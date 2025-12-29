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
| 25,000 | 14.609s | ±0.257s |
| 100,000 | 23.043s | ±2.732s |
| 500,000 | 18.733s | ±0.308s |
| 1,000,000 | 23.110s | ±0.276s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.526s | ±0.018s |
| 20 | 2.059s | ±0.029s |
| 50 | 2.795s | ±0.057s |
| 100 | 24.827s | ±0.358s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 6.481s | ±0.119s |
| 1 Multi-Column Fit | 4.841s | ±0.051s |

**Speedup:** 1.3× faster (25% time saved)
