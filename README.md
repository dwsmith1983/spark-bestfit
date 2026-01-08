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

## Features

- **Parallel Processing**: Spark, Ray, or local thread backends
- **~90 Continuous + 16 Discrete Distributions**
- **Multiple Metrics**: K-S, A-D, SSE, AIC, BIC
- **Bounded Fitting**: Truncated distributions with natural bounds
- **Heavy-Tail Detection**: Warns when data may need special handling
- **Gaussian Copula**: Correlated multi-column sampling
- **Model Serialization**: Save/load to JSON or pickle
- **FitterConfig Builder**: Fluent API for complex configurations

> Full feature list at [spark-bestfit.readthedocs.io](https://spark-bestfit.readthedocs.io/en/latest/)

## Installation

```bash
pip install spark-bestfit              # Core (BYO Spark)
pip install spark-bestfit[spark]       # With PySpark
pip install spark-bestfit[ray]         # With Ray
pip install spark-bestfit[plotting]    # With visualization
```

## Quick Start

```python
from spark_bestfit import DistributionFitter
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
data = np.random.normal(loc=50, scale=10, size=10_000)
df = spark.createDataFrame([(float(x),) for x in data], ["value"])

fitter = DistributionFitter(spark)
results = fitter.fit(df, column="value")

best = results.best(n=1)[0]
print(f"Best: {best.distribution} (KS={best.ks_statistic:.4f})")
```

**Without Spark:**

```python
from spark_bestfit import DistributionFitter, LocalBackend
import pandas as pd

df = pd.DataFrame({"value": np.random.normal(50, 10, 1000)})
fitter = DistributionFitter(backend=LocalBackend())
results = fitter.fit(df, column="value")
```

## Backends

| Backend | Use Case | Install |
|---------|----------|---------|
| **SparkBackend** | Production clusters, 100M+ rows | `[spark]` or BYO |
| **LocalBackend** | Development, testing | Included |
| **RayBackend** | Ray clusters, ML pipelines | `[ray]` |

> See [Backend Guide](https://spark-bestfit.readthedocs.io/en/latest/backends.html) for configuration details.

## Compatibility

| Spark | Python | NumPy |
|-------|--------|-------|
| 3.5.x | 3.11-3.12 | < 2.0 |
| 4.x | 3.12-3.13 | 2.0+ |

## Documentation

Full documentation at **[spark-bestfit.readthedocs.io](https://spark-bestfit.readthedocs.io/en/latest/)**:

- [Quickstart Guide](https://spark-bestfit.readthedocs.io/en/latest/quickstart.html)
- [Backend Guide](https://spark-bestfit.readthedocs.io/en/latest/backends.html)
- [Features](https://spark-bestfit.readthedocs.io/en/latest/index.html#features)
- [Performance & Scaling](https://spark-bestfit.readthedocs.io/en/latest/performance.html)
- [API Reference](https://spark-bestfit.readthedocs.io/en/latest/api.html)

## Contributing

Contributions welcome! See [Contributing Guide](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.
