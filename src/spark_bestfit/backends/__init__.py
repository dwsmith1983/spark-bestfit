"""Execution backend implementations for spark-bestfit.

This package provides different backend implementations for parallel
distribution fitting:

- SparkBackend: Apache Spark using Pandas UDFs (default)
- LocalBackend: Thread-based local execution for testing

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
"""

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.backends.spark import SparkBackend

__all__ = ["SparkBackend", "LocalBackend"]
