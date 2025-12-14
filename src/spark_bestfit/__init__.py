"""spark-bestfit: Modern Spark distribution fitting library.

Efficiently fits ~100 scipy.stats distributions to data using Spark's
parallel processing with optimized Pandas UDFs and broadcast variables.

Example:
    >>> from pyspark.sql import SparkSession
    >>> from spark_bestfit import DistributionFitter
    >>>
    >>> # Create your own SparkSession
    >>> spark = SparkSession.builder.appName("my-app").getOrCreate()
    >>> df = spark.createDataFrame([(float(x),) for x in data], ['value'])
    >>>
    >>> # Fit distributions
    >>> fitter = DistributionFitter(spark)
    >>> results = fitter.fit(df, column='value')
    >>>
    >>> # Get best distribution
    >>> best = results.best(n=1)[0]
    >>> print(f"Best: {best.distribution} with SSE={best.sse:.6f}")
    >>>
    >>> # Plot
    >>> fitter.plot(best, df, 'value', title='Best Fit Distribution')
"""

from spark_bestfit._version import __version__
from spark_bestfit.core import DEFAULT_EXCLUDED_DISTRIBUTIONS, DistributionFitter
from spark_bestfit.distributions import DistributionRegistry
from spark_bestfit.results import DistributionFitResult, FitResults
from spark_bestfit.utils import get_spark_session

__author__ = "Dustin Smith"
__email__ = "dustin.william.smith@gmail.com"

__all__ = [
    # Main class
    "DistributionFitter",
    # Constants
    "DEFAULT_EXCLUDED_DISTRIBUTIONS",
    # Results
    "FitResults",
    "DistributionFitResult",
    # Distribution management
    "DistributionRegistry",
    # Utilities
    "get_spark_session",
    # Version
    "__version__",
]
