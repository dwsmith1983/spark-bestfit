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
| 25,000 | 7.390s | ±0.070s |
| 100,000 | 9.907s | ±0.049s |
| 500,000 | 9.076s | ±0.053s |
| 1,000,000 | 7.803s | ±0.202s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.563s | ±0.019s |
| 20 | 1.218s | ±0.051s |
| 50 | 2.079s | ±0.066s |
| 90 | 8.874s | ±0.109s |
| 107 | 9.995s | ±0.092s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 3.648s | ±0.045s |
| 1 Multi-Column Fit | 3.079s | ±0.082s |

**Speedup:** 1.2× faster (16% time saved)

## Lazy Metrics Performance (v1.5.0+)

| Mode | Fit Time (mean) | Std Dev | Speedup |
|------|-----------------|---------|---------|
| Eager (all metrics) | 10.417s | ±0.291s | baseline |
| Lazy (AIC only) | 2.833s | ±0.065s | -73% |
| Lazy (+ KS on-demand) | 2.836s | ±0.038s | -73% |
| Lazy (+ materialize) | 5.437s | ±0.054s | -48% |

**AIC-only workflow:** ~73% faster than eager fitting

## New Exclusions (v1.7.0)

Three slow distributions added to `DEFAULT_EXCLUSIONS`:

- `tukeylambda` (~7s) - ill-conditioned optimization
- `nct` (~1.4s) - non-central t distribution
- `dpareto_lognorm` (~0.5s) - double Pareto-lognormal
