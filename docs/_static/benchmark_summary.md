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
| 25,000 | 14.645s | ±0.071s |
| 100,000 | 19.483s | ±0.081s |
| 500,000 | 18.747s | ±0.029s |
| 1,000,000 | 23.306s | ±0.058s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.512s | ±0.013s |
| 20 | 2.082s | ±0.025s |
| 50 | 2.825s | ±0.060s |
| 100 | 24.403s | ±0.037s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 6.485s | ±0.051s |
| 1 Multi-Column Fit | 4.954s | ±0.048s |

**Speedup:** 1.3× faster (24% time saved)

## Lazy Metrics (v1.5.0+)

Lazy metrics skips KS/AD computation during fitting, computing them on-demand only
for distributions you actually access.

| Workflow | KS/AD Computations |
|----------|-------------------|
| `best(metric="aic")` | **0** (vs 93 eager) |
| `best(metric="ks_statistic")` | **~5** (vs 93 eager) |
| `materialize()` | 93 (same as eager) |

**Key benefit:** Skip 95% of KS/AD computations in typical model selection workflows.
