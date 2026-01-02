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
| 25,000 | 7.383s | ±0.043s |
| 100,000 | 10.082s | ±0.226s |
| 500,000 | 9.486s | ±0.143s |
| 1,000,000 | 7.750s | ±0.113s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.542s | ±0.015s |
| 20 | 1.084s | ±0.025s |
| 50 | 2.036s | ±0.032s |
| 91 | 8.754s | ±0.049s |
| 107 | 10.297s | ±0.123s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 3.580s | ±0.077s |
| 1 Multi-Column Fit | 3.114s | ±0.086s |

**Speedup:** 1.1× faster (13% time saved)

## Lazy Metrics Performance (v1.5.0+)

| Mode | Fit Time (mean) | Std Dev | Speedup |
|------|-----------------|---------|---------|
| Eager (all metrics) | 9.892s | ±0.048s | baseline |
| Lazy (AIC only) | 2.818s | ±0.038s | -72% |
| Lazy (+ KS on-demand) | 2.886s | ±0.050s | -71% |
| Lazy (+ materialize) | 5.463s | ±0.086s | -45% |

**AIC-only workflow:** ~72% faster than eager fitting

## New Exclusions (v1.7.0)

Three slow distributions added to `DEFAULT_EXCLUSIONS`:

- `tukeylambda` (~7s) - ill-conditioned optimization
- `nct` (~1.4s) - non-central t distribution
- `dpareto_lognorm` (~0.5s) - double Pareto-lognormal
