# spark-bestfit

[![CI](https://github.com/dwsmith1983/spark-bestfit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dwsmith1983/spark-bestfit/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spark-bestfit/badge/?version=latest)](https://spark-bestfit.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/spark-bestfit)](https://pypi.org/project/spark-bestfit/)
[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/dwsmith1983/spark-bestfit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Modern distribution fitting library with pluggable backends (Spark, Ray, Local)**

Efficiently fit ~90 scipy.stats distributions to your data using parallel processing. Supports Apache Spark for production clusters, Ray for ML workflows, or local execution for development.

## Why "spark-bestfit"?

The library was originally built for Apache Spark, hence the name. v2.0 added a pluggable backend architecture supporting multiple execution engines:

- **SparkBackend** — Production clusters and large datasets (100M+ rows)
- **RayBackend** — Ray clusters, ML pipelines, and Kubernetes deployments
- **LocalBackend** — Development, testing, and small datasets

**Why keep the name?**
1. **Backward compatibility** — Existing code using `DistributionFitter(spark)` works unchanged
2. **Primary use case** — Spark remains the best choice for very large datasets (100M+ rows)
3. **Package identity** — Renaming would break imports, PyPI links, and documentation

All backends use identical scipy fitting, so **fit quality is identical** regardless of backend choice.

## Features

- **Parallel Processing**: Fits distributions in parallel using Spark, Ray, or local threads
- **~90 Continuous Distributions**: Nearly all scipy.stats continuous distributions
- **16 Discrete Distributions**: Fit count data with Poisson, negative binomial, geometric, and more
- **Multiple Metrics**: K-S statistic, A-D statistic, SSE, AIC, and BIC
- **Confidence Intervals**: Bootstrap confidence intervals for fitted parameters
- **Bounded Fitting**: Fit truncated distributions with natural bounds
- **Multi-Column Fitting**: Fit multiple columns efficiently in a single operation
- **Lazy Metrics**: Skip expensive KS/AD computation; compute on-demand
- **Smart Pre-filtering**: Skip incompatible distributions based on data shape
- **Gaussian Copula**: Correlated multi-column sampling at scale
- **Distributed Sampling**: Generate millions of samples using cluster parallelism
- **Model Serialization**: Save and load fitted distributions to JSON or pickle
- **Visualization**: Built-in plotting for distribution comparison, Q-Q and P-P plots (requires `[plotting]` extra)

## Installation

```bash
pip install spark-bestfit
```

This installs spark-bestfit without PySpark. You provide a compatible Spark environment.

**With PySpark included:**

```bash
pip install spark-bestfit[spark]
```

**With Ray support:**

```bash
pip install spark-bestfit[ray]
```

**With built-in plotting:**

```bash
pip install spark-bestfit[plotting]
```

Combine extras as needed: `pip install spark-bestfit[spark,plotting]`

## Quick Start

```python
from spark_bestfit import DistributionFitter
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=10_000)
df = spark.createDataFrame([(float(x),) for x in data], ["value"])

# Fit distributions
fitter = DistributionFitter(spark)
results = fitter.fit(df, column="value")

# Get best fit
best = results.best(n=1)[0]
print(f"Best: {best.distribution} (KS={best.ks_statistic:.4f})")

# Plot (requires: pip install spark-bestfit[plotting])
fitter.plot(best, df, "value", title="Best Fit Distribution")

# Or use result.pdf(), result.cdf() for DIY plotting with any library
```

## Compatibility Matrix

| Spark Version | Python Versions | NumPy | Pandas | PyArrow |
|---------------|-----------------|-------|--------|---------|
| **3.5.x** | 3.11, 3.12 | 1.24+ (< 2.0) | 1.5+ | 12.0 - 16.x |
| **4.x** | 3.12, 3.13 | 2.0+ | 2.2+ | 17.0+ |

> **Note**: Spark 3.5.x does not support NumPy 2.0.

## Backend Support

| Backend | Use Case | Install |
|---------|----------|---------|
| **SparkBackend** | Production clusters, large datasets | PySpark required (BYO or `[spark]`) |
| **LocalBackend** | Unit testing, development | Included |
| **RayBackend** | Ray clusters, ML workflows | `pip install spark-bestfit[ray]` |

```python
from spark_bestfit import DistributionFitter, SparkBackend, RayBackend, LocalBackend

# SparkBackend (default, backward compatible)
fitter = DistributionFitter(spark)

# Explicit backend
fitter = DistributionFitter(backend=SparkBackend(spark))

# LocalBackend for testing (no Spark required)
fitter = DistributionFitter(backend=LocalBackend())

# RayBackend for Ray clusters
fitter = DistributionFitter(backend=RayBackend())
```

> See [Backend Guide](https://spark-bestfit.readthedocs.io/en/latest/backends.html) for detailed configuration.

## Working with Results

```python
# Get top fits by different metrics
best_ks = results.best(n=1)[0]                    # K-S statistic (default)
best_aic = results.best(n=1, metric="aic")[0]     # AIC
best_ad = results.best(n=1, metric="ad_statistic")[0]  # A-D statistic

# Filter by goodness-of-fit
good_fits = results.filter(ks_threshold=0.05)
significant = results.filter(pvalue_threshold=0.05)

# Use fitted distribution
samples = best.sample(size=10000)
pdf_values = best.pdf(x_array)
percentile_95 = best.ppf(0.95)

# Quality diagnostics
report = results.quality_report()
if report["warnings"]:
    print(f"Warnings: {report['warnings']}")
```

## Bounded Distribution Fitting

Fit distributions with natural constraints (percentages, ages, prices):

```python
# Auto-detect bounds from data min/max
results = fitter.fit(df, column="percentage", bounded=True)

# Explicit bounds
results = fitter.fit(
    df, column="price",
    bounded=True,
    lower_bound=0.0,
    upper_bound=1000.0,
)

# Samples automatically respect bounds
best = results.best(n=1)[0]
samples = best.sample(1000)  # All within [0, 1000]
```

## Discrete Distributions

For count data (integers):

```python
from spark_bestfit import DiscreteDistributionFitter

# Fit discrete distributions
fitter = DiscreteDistributionFitter(spark)
results = fitter.fit(df, column="counts")

# Use AIC for model selection (recommended for discrete)
best = results.best(n=1, metric="aic")[0]
```

## Model Serialization

```python
from spark_bestfit import DistributionFitResult

# Save to JSON (human-readable, recommended)
best.save("model.json")

# Load and use later - no Spark needed for inference!
loaded = DistributionFitResult.load("model.json")
samples = loaded.sample(size=1000)
```

## Gaussian Copula

Generate correlated multi-column samples:

```python
from spark_bestfit import GaussianCopula

# Fit copula from multi-column results
copula = GaussianCopula.fit(results, df)

# Local sampling
samples = copula.sample(n=10_000)

# Distributed sampling (100M+ samples)
samples_df = copula.sample_spark(n=100_000_000)
```

## FitterConfig Builder (v2.2+)

For complex configurations, use the fluent builder pattern:

```python
from spark_bestfit import DistributionFitter, FitterConfigBuilder

# Build a reusable configuration
config = (FitterConfigBuilder()
    .with_bins(100)
    .with_bounds(lower=0, upper=100)
    .with_sampling(fraction=0.1)
    .with_lazy_metrics()
    .with_prefilter()
    .build())

# Use across multiple fits
fitter = DistributionFitter(spark)
for col in ["price", "quantity", "revenue"]:
    results = fitter.fit(df, column=col, config=config)
```

**Why use FitterConfig?**
- **Cleaner code**: No more 15+ parameters in function calls
- **Reusable**: Same config works across multiple fits
- **IDE-friendly**: Better autocomplete and discoverability
- **Immutable**: Frozen dataclass prevents accidental mutation

Individual parameters still work for simple cases (backward compatible).

> See [Configuration Guide](https://spark-bestfit.readthedocs.io/en/latest/features/config.html) for full details.

## Performance Tips

**Lazy metrics** for faster model selection:

```python
# Using FitterConfig (recommended)
config = FitterConfigBuilder().with_lazy_metrics().build()
results = fitter.fit(df, column="value", config=config)

# Or using parameters directly
results = fitter.fit(df, column="value", lazy_metrics=True)

# Fast model selection by AIC
best = results.best(n=1, metric="aic")[0]

# KS computed on-demand only for top candidates
best_ks = results.best(n=1, metric="ks_statistic")[0]
```

**Pre-filtering** for skewed data:

```python
# Skip incompatible distributions (20-50% faster)
config = FitterConfigBuilder().with_prefilter().build()
results = fitter.fit(df, column="value", config=config)
```

> See [Performance & Scaling](https://spark-bestfit.readthedocs.io/en/latest/performance.html) for benchmarks.

## Scope & Limitations

**What it does well:**
- Fit ~90 continuous and 16 discrete scipy.stats distributions in parallel
- Provide robust goodness-of-fit metrics (KS, A-D, AIC, BIC, SSE)
- Generate publication-ready visualizations
- Compute bootstrap confidence intervals

**Known limitations:**
- No real-time/streaming support (batch processing only)
- See [Roadmap](#roadmap) for planned features

## Roadmap

| Version | Focus | Key Features |
|---------|-------|--------------|
| **2.2.0** | API Polish | FitterConfig builder, user-defined distributions |
| **3.0.0** | Advanced | Mixture models, streaming support, right-censored data |

See [GitHub milestones](https://github.com/dwsmith1983/spark-bestfit/milestones) for details.

## Documentation

Full documentation at [spark-bestfit.readthedocs.io](https://spark-bestfit.readthedocs.io/en/latest/):

- [Quickstart Guide](https://spark-bestfit.readthedocs.io/en/latest/quickstart.html)
- [Backend Guide](https://spark-bestfit.readthedocs.io/en/latest/backends.html)
- [Performance & Scaling](https://spark-bestfit.readthedocs.io/en/latest/performance.html)
- [API Reference](https://spark-bestfit.readthedocs.io/en/latest/api.html)
- [Migration Guide](https://spark-bestfit.readthedocs.io/en/latest/migration.html)

## Contributing

Contributions welcome! See [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

1. Fork the repository
2. Create feature branch (`git checkout -b feat/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
