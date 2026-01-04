# Ray + pandas Benchmark Results Summary

## Test Environment

| Property | Value |
|----------|-------|
| CPU | Apple M5 |
| Cores | 10 |
| OS | Darwin |
| Python | 3.13.11 |
| Backend | Ray + pandas |

> **Note:** These benchmarks were run on a local development machine.
> Absolute times will vary based on hardware. The key insight is the
> **scaling characteristics**: sub-linear for data size, O(D) for distribution count.

## Ray + pandas Data Size Scaling

| Data Size | Fit Time (mean) | Std Dev |
|-----------|-----------------|---------|
| 25,000 | 2.757s | ±0.030s |
| 100,000 | 2.846s | ±0.038s |
| 500,000 | 3.054s | ±0.030s |
| 1,000,000 | 2.704s | ±0.030s |

## Ray + pandas Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.086s | ±0.006s |
| 20 | 0.265s | ±0.013s |
| 50 | 0.576s | ±0.016s |
| 90 | 2.766s | ±0.043s |
| 107 | 2.910s | ±0.030s |

## Ray + pandas Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 0.742s | ±0.018s |
| 1 Multi-Column Fit | 0.746s | ±0.021s |

**Speedup:** 1.0× faster (-1% time saved)

## Lazy Metrics Performance (v1.5.0+)

| Mode | Fit Time (mean) | Std Dev | Speedup |
|------|-----------------|---------|---------|
| Eager (all metrics) | 2.886s | ±0.037s | baseline |
| Lazy (AIC only) | 0.886s | ±0.016s | -69% |
| Lazy (+ KS on-demand) | 0.924s | ±0.015s | -68% |

**AIC-only workflow:** ~69% faster than eager fitting

## New Exclusions (v1.7.0)

Three slow distributions added to `DEFAULT_EXCLUSIONS`:

- `tukeylambda` (~7s) - ill-conditioned optimization
- `nct` (~1.4s) - non-central t distribution
- `dpareto_lognorm` (~0.5s) - double Pareto-lognormal
