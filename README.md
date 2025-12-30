# spark-bestfit

[![CI](https://github.com/dwsmith1983/spark-bestfit/actions/workflows/ci.yml/badge.svg)](https://github.com/dwsmith1983/spark-bestfit/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spark-bestfit/badge/?version=latest)](https://spark-bestfit.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/spark-bestfit)](https://pypi.org/project/spark-bestfit/)
[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/dwsmith1983/spark-bestfit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Modern Spark distribution fitting library with efficient parallel processing**

Efficiently fit ~100 scipy.stats distributions to your data using Spark's parallel processing with optimized Pandas UDFs and broadcast variables.

## Features

- **Parallel Processing**: Fits distributions in parallel using Spark
- **Multi-Column Fitting**: Fit multiple columns efficiently in a single operation
- **~100 Continuous Distributions**: Access to nearly all scipy.stats continuous distributions
- **16 Discrete Distributions**: Fit count data with Poisson, negative binomial, geometric, and more
- **Histogram-Based Fitting**: Efficient fitting using histogram representation
- **Multiple Metrics**: Compare fits using K-S statistic, A-D statistic, SSE, AIC, and BIC
- **Statistical Validation**: Kolmogorov-Smirnov and Anderson-Darling tests for goodness-of-fit
- **Confidence Intervals**: Bootstrap confidence intervals for fitted parameters
- **Progress Tracking**: Monitor long-running fits with customizable callbacks
- **Distributed Sampling**: Generate millions of samples using Spark's parallelism
- **Gaussian Copula**: Correlated multi-column sampling at scale via Spark ML
- **Fit Quality Warnings**: Automatic warnings for poor fits with detailed diagnostics
- **Model Serialization**: Save and load fitted distributions to JSON or pickle
- **Results API**: Filter, sort, and export results easily
- **Visualization**: Built-in plotting for distribution comparison, Q-Q plots and P-P plots
- **Flexible Configuration**: Customize bins, sampling, and distribution selection

## Scope & Limitations

spark-bestfit is designed for **batch processing** of statistical distribution fitting on Spark DataFrames.

**What it does well:**
- Fit ~100 continuous and 16 discrete scipy.stats distributions in parallel
- Provide robust goodness-of-fit metrics (KS, A-D, AIC, BIC, SSE)
- Generate publication-ready visualizations (histograms, Q-Q plots, P-P plots)
- Compute bootstrap confidence intervals for parameters

**Known limitations:**
- No real-time/streaming support (batch processing only)
- See [Roadmap](#roadmap) for planned features

## Installation

```bash
pip install spark-bestfit
```

This installs spark-bestfit without PySpark. You are responsible for providing a compatible Spark environment (see Compatibility Matrix below).

**With PySpark included** (for users without a managed Spark environment):

```bash
pip install spark-bestfit[spark]
```

## Quick Start

```python
from spark_bestfit import DistributionFitter
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=10_000)

# Create fitter
fitter = DistributionFitter(spark)
df = spark.createDataFrame([(float(x),) for x in data], ["value"])

# Fit distributions
results = fitter.fit(df, column="value")

# Get best fit (by K-S statistic, the default)
best = results.best(n=1)[0]
print(f"Best: {best.distribution} (KS={best.ks_statistic:.4f}, p={best.pvalue:.4f})")

# Plot
fitter.plot(best, df, "value", title="Best Fit Distribution")
```

## Compatibility Matrix

| Spark Version | Python Versions | NumPy | Pandas | PyArrow |
|---------------|-----------------|-------|--------|---------|
| **3.5.x** | 3.11, 3.12 | 1.24+ (< 2.0) | 1.5+ | 12.0 - 16.x |
| **4.x** | 3.12, 3.13 | 2.0+ | 2.2+ | 17.0+ |

> **Note**: Spark 3.5.x does not support NumPy 2.0. If using Spark 3.5 with Python 3.12, ensure `setuptools` is installed (provides `distutils`).

## API Overview

### Fitting Distributions

```python
from spark_bestfit import DistributionFitter

fitter = DistributionFitter(spark, random_seed=123)
results = fitter.fit(
    df,
    column="value",
    bins=100,                    # Number of histogram bins
    support_at_zero=True,        # Only fit non-negative distributions
    enable_sampling=True,        # Enable adaptive sampling
    sample_fraction=0.3,         # Sample 30% of data
    max_distributions=50,        # Limit distributions to fit
)
```

### Multi-Column Fitting

Fit multiple columns efficiently in a single operation:

```python
from spark_bestfit import DistributionFitter

# Create DataFrame with multiple columns
df = spark.createDataFrame([
    (1.0, 10.0, 100.0),
    (2.0, 20.0, 200.0),
    # ...
], ["col_a", "col_b", "col_c"])

fitter = DistributionFitter(spark)

# Fit all columns in one call - shares Spark overhead
results = fitter.fit(df, columns=["col_a", "col_b", "col_c"])

# Get results for a specific column
col_a_results = results.for_column("col_a")
best_a = col_a_results.best(n=1)[0]

# Get best distribution per column
best_per_col = results.best_per_column(n=1)
for col_name, fits in best_per_col.items():
    print(f"{col_name}: {fits[0].distribution} (KS={fits[0].ks_statistic:.4f})")

# List all columns in results
print(results.column_names)  # ['col_a', 'col_b', 'col_c']
```

Multi-column fitting is more efficient than fitting columns separately because it:
- Performs a single `df.count()` call for all columns
- Shares the data sample across all fitting operations
- Minimizes Spark job overhead

> **Benchmark:** Fitting 3 columns together is ~1.3× faster than 3 separate fits (4.8s vs 6.5s).
> See [Performance & Scaling](https://spark-bestfit.readthedocs.io/en/latest/performance.html) for details.

### Working with Results

```python
# Get top 5 distributions (by K-S statistic, the default)
top_5 = results.best(n=5)

# Get best by other metrics
best_sse = results.best(n=1, metric="sse")[0]
best_aic = results.best(n=1, metric="aic")[0]
best_ad = results.best(n=1, metric="ad_statistic")[0]

# Filter by goodness-of-fit
good_fits = results.filter(ks_threshold=0.05)        # K-S statistic < 0.05
significant = results.filter(pvalue_threshold=0.05)  # p-value > 0.05
good_ad = results.filter(ad_threshold=1.0)           # A-D statistic < 1.0

# Convert to pandas for analysis
df_pandas = results.df.toPandas()

# Use fitted distribution
samples = best.sample(size=10000)  # Generate samples
pdf_values = best.pdf(x_array)     # Evaluate PDF
cdf_values = best.cdf(x_array)     # Evaluate CDF

# Access all goodness-of-fit metrics
print(f"K-S: {best.ks_statistic}, p-value: {best.pvalue}")
print(f"A-D: {best.ad_statistic}, A-D p-value: {best.ad_pvalue}")

# Get quality report for fit diagnostics
report = results.quality_report()
if report["warnings"]:
    print(f"Warnings: {report['warnings']}")

# Warn automatically for poor fits
best = results.best(n=1, warn_if_poor=True)[0]
```

> **Note**: Anderson-Darling p-values are only available for 5 distributions (norm, expon,
> logistic, gumbel_r, gumbel_l) where scipy has critical value tables. For other distributions,
> `ad_pvalue` will be `None` but `ad_statistic` is still valid for ranking fits.

### Progress Tracking

Monitor long-running fits with the built-in `console_progress()` utility:

```python
from spark_bestfit.progress import console_progress

results = fitter.fit(df, column="value", progress_callback=console_progress())
print()  # Newline after completion
# Output: Progress: 45/100 tasks (45.0%)
```

For custom callbacks or tqdm integration:

```python
# Custom callback
def on_progress(completed: int, total: int, percent: float) -> None:
    print(f"\rFitting: {completed}/{total} ({percent:.1f}%)", end="", flush=True)

results = fitter.fit(df, column="value", progress_callback=on_progress)

# tqdm integration
from tqdm import tqdm

pbar = None

def tqdm_callback(completed: int, total: int, percent: float) -> None:
    global pbar
    if pbar is None:
        pbar = tqdm(total=total, desc="Fitting")
    pbar.n = completed
    pbar.refresh()

results = fitter.fit(df, column="value", progress_callback=tqdm_callback)
if pbar:
    pbar.close()
```

> **Note**: Progress percentages may fluctuate during fitting as new Spark stages add tasks.
> See [Progress Tracking](https://spark-bestfit.readthedocs.io/en/latest/progress.html) for details.

### Parameter Confidence Intervals

```python
# Compute 95% bootstrap confidence intervals
ci = best.confidence_intervals(df, column="value", alpha=0.05, n_bootstrap=1000, random_seed=42)

# Display with parameter names
print(f"Distribution: {best.distribution}")
for param, (lower, upper) in ci.items():
    print(f"  {param}: [{lower:.4f}, {upper:.4f}]")
```

### Distributed Sampling

Generate large samples using Spark's distributed computing:

```python
# Generate 1 million samples distributed across the cluster
samples_df = best.sample_spark(n=1_000_000, spark=spark)
samples_df.show(5)

# With reproducibility
samples_df = best.sample_spark(n=1_000_000, spark=spark, random_seed=42)

# Control partitioning
samples_df = best.sample_spark(
    n=1_000_000,
    spark=spark,
    num_partitions=16,
    column_name="generated_values"
)
```

> **Tip**: Use `sample_spark()` for very large samples (>10M) to leverage cluster parallelism.
> For smaller samples, `sample(size=N)` returns a local NumPy array and is more efficient.
> See [Distributed Sampling](https://spark-bestfit.readthedocs.io/en/latest/sampling.html) for benchmarks.

### Gaussian Copula

Generate correlated multi-column samples that preserve both marginal distributions and correlation structure:

```python
from spark_bestfit import DistributionFitter, GaussianCopula

# Fit multiple columns
fitter = DistributionFitter(spark)
results = fitter.fit(df, columns=["price", "quantity", "revenue"])

# Fit copula - correlation computed via Spark ML (scales to billions of rows)
copula = GaussianCopula.fit(results, df)

# Local sampling (small scale)
samples = copula.sample(n=10_000)  # Returns Dict[str, np.ndarray]

# Fast uniform sampling (skips marginal transforms, ~24% faster than statsmodels)
uniform_samples = copula.sample(n=10_000_000, return_uniform=True)

# Distributed sampling (large scale) - scales to 100M+ samples
samples_df = copula.sample_spark(n=100_000_000)

# Serialize for later use
copula.save("copula.json")
loaded = GaussianCopula.load("copula.json")
```

**When to use spark-bestfit copula** (vs statsmodels):

| Scenario | statsmodels | spark-bestfit |
|----------|-------------|---------------|
| Data < 10M rows | Faster (use this) | Slower (Spark overhead) |
| Data > 100M rows | Crashes (OOM) | **Works** (distributed) |
| Data already in Spark | Requires `.toPandas()` | **Native** (no conversion) |
| 100M+ samples needed | May OOM | **`sample_spark()`** distributed |

> See [Gaussian Copula](https://spark-bestfit.readthedocs.io/en/latest/copula.html) for details.

### Custom Plotting

```python
fitter.plot(
    best,
    df,
    "value",
    figsize=(16, 10),
    dpi=300,
    histogram_alpha=0.6,
    pdf_linewidth=3,
    title="Distribution Fit",
    xlabel="Value",
    ylabel="Density",
    save_path="output/distribution.png",
)
```

### Q-Q Plots

```python
# Create Q-Q plot for goodness-of-fit assessment
fitter.plot_qq(
    best,
    df,
    "value",
    max_points=1000,           # Sample size for plotting
    title="Q-Q Plot",
    save_path="output/qq_plot.png",
)
```

### P-P Plots

```python
# Create P-P plot for goodness-of-fit assessment
fitter.plot_pp(
    best,
    df,
    "value",
    max_points=1000,           # Sample size for plotting
    title="P-P Plot",
    save_path="output/pp_plot.png",
)
```

### Discrete Distributions

For count data (integers), use `DiscreteDistributionFitter`:

```python
from spark_bestfit import DiscreteDistributionFitter
import numpy as np

# Generate count data
data = np.random.poisson(lam=7, size=10_000)
df = spark.createDataFrame([(int(x),) for x in data], ["counts"])

# Fit discrete distributions
fitter = DiscreteDistributionFitter(spark)
results = fitter.fit(df, column="counts")

# Get best fit - use AIC for model selection (recommended for discrete)
best = results.best(n=1, metric="aic")[0]
print(f"Best: {best.distribution} (AIC={best.aic:.2f})")

# Plot fitted PMF
fitter.plot(best, df, "counts", title="Best Discrete Fit")
```

**Metric Selection for Discrete Distributions:**

| Metric | Use Case |
|--------|----------|
| `aic` | **Recommended** - Proper model selection criterion with complexity penalty |
| `bic` | Similar to AIC but stronger penalty for complex models |
| `ks_statistic` | Valid for ranking fits, but p-values are not reliable for discrete data |
| `ad_statistic` | Valid for ranking fits (not computed for discrete distributions) |
| `sse` | Simple comparison metric |

> **Note**: The K-S and A-D tests assume continuous distributions. For discrete data, the K-S
> statistic can still rank fits, but p-values are conservative and should not be used for
> hypothesis testing. A-D statistics are not computed for discrete distributions.
> Use AIC/BIC for proper model selection.

### Excluding Distributions

```python
from spark_bestfit import DistributionFitter, DEFAULT_EXCLUDED_DISTRIBUTIONS

# View default exclusions
print(DEFAULT_EXCLUDED_DISTRIBUTIONS)

# Include a specific distribution by removing it from exclusions
exclusions = tuple(d for d in DEFAULT_EXCLUDED_DISTRIBUTIONS if d != "wald")
fitter = DistributionFitter(spark, excluded_distributions=exclusions)

# Or exclude nothing (fit all distributions - may be slow)
fitter = DistributionFitter(spark, excluded_distributions=())
```

### Model Serialization

Save fitted distributions to disk and reload them later for inference:

```python
from spark_bestfit import DistributionFitResult

# Save the best fit to JSON (human-readable, recommended)
best.save("model.json")

# Or save to pickle (faster, binary)
best.save("model.pkl", format="pickle")

# Load and use later - no Spark needed for inference!
loaded = DistributionFitResult.load("model.json")
samples = loaded.sample(size=1000)
percentile_95 = loaded.ppf(0.95)
```

> **Tip**: JSON format includes version metadata and is recommended for most use cases.
> See [Serialization](https://spark-bestfit.readthedocs.io/en/latest/serialization.html) for details.

## Roadmap

spark-bestfit enables downstream use cases (simulations, ML, analytics) by providing distribution fitting primitives.

| Version | Focus | Key Features |
|---------|-------|--------------|
| **1.4.0** | Bounded Distributions | Truncated distribution fitting for data with natural bounds |
| **2.0.0** | Custom Distributions | User-defined distribution classes |
| **2.1.0** | Multivariate | Optional multivariate distribution fitting (MVN, MVt) |
| **3.0.0** | Advanced | Mixture models, streaming support |

See the [GitHub milestones](https://github.com/dwsmith1983/spark-bestfit/milestones) for detailed issue tracking.

## Documentation

Full documentation is available at [spark-bestfit.readthedocs.io](https://spark-bestfit.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
