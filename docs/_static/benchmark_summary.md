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
| 25,000 | 14.696s | ±0.054s |
| 100,000 | 19.638s | ±0.163s |
| 500,000 | 18.816s | ±0.037s |
| 1,000,000 | 23.377s | ±0.072s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.510s | ±0.013s |
| 20 | 2.074s | ±0.029s |
| 50 | 2.815s | ±0.053s |
| 100 | 24.414s | ±0.044s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 6.451s | ±0.042s |
| 1 Multi-Column Fit | 4.867s | ±0.051s |

**Speedup:** 1.3× faster (25% time saved)
