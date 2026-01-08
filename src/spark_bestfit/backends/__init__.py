"""Execution backend implementations for spark-bestfit.

This package provides different backend implementations for parallel
distribution fitting:

- SparkBackend: Apache Spark using Pandas UDFs (default)
- LocalBackend: Thread-based local execution for testing
- RayBackend: Ray cluster execution (optional, requires ray)

Example:
    >>> from spark_bestfit.backends.spark import SparkBackend
    >>> from spark_bestfit import DistributionFitter
    >>>
    >>> backend = SparkBackend(spark)
    >>> fitter = DistributionFitter(backend=backend)
    >>> results = fitter.fit(df, column='value')

For testing without Spark:
    >>> from spark_bestfit.backends.local import LocalBackend
    >>> backend = LocalBackend()
    >>> fitter = DistributionFitter(backend=backend)

For Ray clusters:
    >>> from spark_bestfit.backends.ray import RayBackend
    >>> backend = RayBackend()  # Auto-initializes Ray
    >>> fitter = DistributionFitter(backend=backend)
"""

from spark_bestfit.backends.factory import BackendFactory
from spark_bestfit.backends.local import LocalBackend

__all__ = ["BackendFactory", "LocalBackend"]

# Conditional Spark import (only if pyspark is installed)
try:
    from spark_bestfit.backends.spark import SparkBackend  # noqa: F401

    __all__.append("SparkBackend")
except ImportError:
    pass  # PySpark not installed, SparkBackend not available

# Conditional Ray import (only if ray is installed)
try:
    from spark_bestfit.backends.ray import RayBackend  # noqa: F401

    __all__.append("RayBackend")
except ImportError:
    pass  # Ray not installed, RayBackend not available
