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
| 25,000 | 4.780s | ±0.047s |
| 100,000 | 6.645s | ±0.140s |
| 500,000 | 5.899s | ±0.055s |
| 1,000,000 | 5.049s | ±0.035s |

## Distribution Count Scaling

| # Distributions | Fit Time (mean) | Std Dev |
|-----------------|-----------------|---------|
| 5 | 0.459s | ±0.012s |
| 20 | 0.869s | ±0.023s |
| 50 | 1.461s | ±0.036s |
| 90 | 5.706s | ±0.045s |
| 107 | 6.567s | ±0.039s |

## Multi-Column Efficiency

| Approach | Fit Time (mean) | Std Dev |
|----------|-----------------|---------|
| 3 Separate Fits | 2.693s | ±0.052s |
| 1 Multi-Column Fit | 2.685s | ±0.044s |

**Note (v2.0.0):** Multi-column fitting no longer provides significant speedup (~0.3%).
The v2.0.0 optimizations reduced per-operation overhead so much that both approaches
now converge to the same performance. Use whichever API is more convenient.

## Lazy Metrics Performance (v1.5.0+)

| Mode | Fit Time (mean) | Std Dev | Speedup |
|------|-----------------|---------|---------|
| Eager (all metrics) | 6.616s | ±0.085s | baseline |
| Lazy (AIC only) | 2.019s | ±0.051s | -69% |
| Lazy (+ KS on-demand) | 2.072s | ±0.045s | -69% |
| Lazy (+ materialize) | 4.705s | ±0.048s | -29% |

**AIC-only workflow:** ~69% faster than eager fitting

## New Exclusions (v1.7.0)

Three slow distributions added to `DEFAULT_EXCLUSIONS`:

- `tukeylambda` (~7s) - ill-conditioned optimization
- `nct` (~1.4s) - non-central t distribution
- `dpareto_lognorm` (~0.5s) - double Pareto-lognormal

## v2.0.0 Performance Improvements

The backend abstraction refactor delivered significant performance gains:

| Version | Test Suite Time | Improvement |
|---------|-----------------|-------------|
| v1.5.0 | ~90 min | baseline |
| v1.7.0 | ~45 min | 50% faster |
| v2.0.0 | ~33 min | 27% faster (63% total) |

### Benchmark Comparison (v1.7.2 → v2.0.0)

| Test | v1.7.2 | v2.0.0 | Change |
|------|--------|--------|--------|
| `test_continuous_fit_10k` | 0.803s | 0.626s | **-22%** |
| `test_discrete_fit_10k` | 1.191s | 0.897s | **-25%** |
| `test_discrete_all_distributions` | 2.830s | 1.887s | **-33%** |
| `test_fit_1m_rows` | 7.803s | 5.049s | **-35%** |
| `test_fit_default_distributions` | 8.874s | 5.706s | **-36%** |
| `test_fit_eager_all_metrics` | 10.417s | 6.616s | **-36%** |
| `test_fit_lazy_aic_only` | 2.833s | 2.019s | **-29%** |
| `test_fit_lazy_materialize` | 5.437s | 4.705s | **-13%** |

**Key optimizations:**

1. **Flattened data_summary schema** - Replaced `MapType` with individual columns
   (`data_min`, `data_max`, etc.) for ~30% faster Arrow serialization

2. **Improved broadcast lifecycle** - Proper cleanup in `finally` blocks reduces
   memory pressure and GC overhead

3. **Backend abstraction** - Cleaner code paths with fewer intermediate objects
