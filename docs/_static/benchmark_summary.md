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
| 25,000 | 7.343s | ±0.065s |
| 100,000 | 9.767s | ±0.061s |
| 500,000 | 8.958s | ±0.058s |
| 1,000,000 | 7.603s | ±0.061s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.549s | ±0.011s |
| 20 | 1.065s | ±0.018s |
| 50 | 1.985s | ±0.042s |
| 100 | 8.714s | ±0.044s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 3.506s | ±0.051s |
| 1 Multi-Column Fit | 3.102s | ±0.029s |

**Speedup:** 1.1× faster (12% time saved)

## Lazy Metrics Performance (v1.5.0+)

| Mode | Fit Time (mean) | Std Dev | Speedup |
|------|-----------------|---------|---------|
| Eager (all metrics) | 9.745s | ±0.049s | baseline |
| Lazy (AIC only) | 2.807s | ±0.050s | -71% |
| Lazy (+ KS on-demand) | 2.858s | ±0.041s | -71% |
| Lazy (+ materialize) | 5.435s | ±0.046s | -44% |

**AIC-only workflow:** ~71% faster than eager fitting

## New Exclusions (v1.7.0)

Three slow distributions added to `DEFAULT_EXCLUSIONS`:

- `tukeylambda` (~7s) - ill-conditioned optimization
- `nct` (~1.4s) - non-central t distribution
- `dpareto_lognorm` (~0.5s) - double Pareto-lognormal
