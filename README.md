# spark-bestfit

[![CI](https://github.com/dwsmith1983/spark-bestfit/actions/workflows/ci.yml/badge.svg)](https://github.com/dwsmith1983/spark-bestfit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/spark-bestfit)](https://pypi.org/project/spark-bestfit/)
[![Documentation Status](https://readthedocs.org/projects/spark-bestfit/badge/?version=latest)](https://spark-bestfit.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Modern Spark distribution fitting library with efficient parallel processing**

Efficiently fit ~100 scipy.stats distributions to your data using Spark's parallel processing with optimized Pandas UDFs and broadcast variables.

## Features

- **Parallel Processing**: Fits distributions in parallel using Spark
- **~100 Distributions**: Access to nearly all scipy.stats continuous distributions
- **Histogram-Based Fitting**: Efficient fitting using histogram representation
- **Multiple Metrics**: Compare fits using SSE, AIC, and BIC
- **Results API**: Filter, sort, and export results easily
- **Visualization**: Built-in plotting for distribution comparison
- **Flexible Configuration**: Customize bins, sampling, and distribution selection

## Installation

```bash
pip install spark-bestfit
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

# Get best fit
best = results.best(n=1)[0]
print(f"Best: {best.distribution} with SSE={best.sse:.6f}")

# Plot
fitter.plot(best, df, "value", title="Best Fit Distribution")
```

## Compatibility Matrix

| Spark Version | Python Versions | NumPy | Pandas | PyArrow |
|---------------|-----------------|-------|--------|---------|
| **3.5.x** | 3.11, 3.12 | 1.24+ (< 2.0) | 1.5+ | 12.0 - 16.x |
| **4.0.x** | 3.12, 3.13 | 2.0+ | 2.2+ | 17.0+ |

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

### Working with Results

```python
# Get top 5 distributions by SSE
top_5 = results.best(n=5, metric="sse")

# Get best by AIC
best_aic = results.best(n=1, metric="aic")[0]

# Filter good fits
good_fits = results.filter(sse_threshold=0.01)

# Convert to pandas for analysis
df_pandas = results.to_pandas()

# Use fitted distribution
samples = best.sample(size=10000)  # Generate samples
pdf_values = best.pdf(x_array)     # Evaluate PDF
cdf_values = best.cdf(x_array)     # Evaluate CDF
```

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

## Documentation

Full documentation is available at [spark-bestfit.readthedocs.io](https://spark-bestfit.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
