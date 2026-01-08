"""spark-bestfit: Modern Spark distribution fitting library.

Efficiently fits ~90 scipy.stats distributions to data using Spark's
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
    >>> # Get best distribution (by K-S statistic, the default)
    >>> best = results.best(n=1)[0]
    >>> print(f"Best: {best.distribution} with KS={best.ks_statistic:.6f}")
    >>>
    >>> # Plot
    >>> fitter.plot(best, df, 'value', title='Best Fit Distribution')
"""

from spark_bestfit._version import __version__
from spark_bestfit.backends.factory import BackendFactory
from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.config import FitterConfig, FitterConfigBuilder

# Conditional Spark import (only if pyspark is installed)
try:
    from spark_bestfit.backends.spark import SparkBackend  # noqa: F401

    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False

# Conditional Ray import (only if ray is installed)
try:
    from spark_bestfit.backends.ray import RayBackend  # noqa: F401

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

from spark_bestfit.copula import GaussianCopula
from spark_bestfit.core import (
    DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS,
    DEFAULT_EXCLUDED_DISTRIBUTIONS,
    DiscreteDistributionFitter,
    DistributionFitter,
    TruncatedFrozenDist,
)
from spark_bestfit.distributions import DiscreteDistributionRegistry, DistributionRegistry
from spark_bestfit.progress import ProgressCallback, ProgressTracker, console_progress
from spark_bestfit.protocols import ExecutionBackend
from spark_bestfit.results import (
    BaseFitResults,
    DistributionFitResult,
    EagerFitResults,
    FitResults,
    FitResultsType,
    LazyFitResults,
    MetricName,
)
from spark_bestfit.serialization import SerializationError
from spark_bestfit.utils import get_spark_session

__author__ = "Dustin Smith"
__email__ = "dustin.william.smith@gmail.com"

__all__ = [
    # Main classes
    "DistributionFitter",
    "DiscreteDistributionFitter",
    "GaussianCopula",
    "TruncatedFrozenDist",
    # Configuration (v2.2)
    "FitterConfig",
    "FitterConfigBuilder",
    # Backends (v2.0)
    "BackendFactory",
    "ExecutionBackend",
    "LocalBackend",
    # SparkBackend and RayBackend added conditionally below
    # Progress tracking
    "ProgressTracker",
    "ProgressCallback",
    "console_progress",
    # Constants
    "DEFAULT_EXCLUDED_DISTRIBUTIONS",
    "DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS",
    # Results
    "FitResults",
    "FitResultsType",
    "BaseFitResults",
    "EagerFitResults",
    "LazyFitResults",
    "DistributionFitResult",
    # Type aliases
    "MetricName",
    # Serialization
    "SerializationError",
    # Distribution management
    "DistributionRegistry",
    "DiscreteDistributionRegistry",
    # Utilities
    "get_spark_session",
    # Version
    "__version__",
]

# Conditionally add SparkBackend to exports if pyspark is installed
if _SPARK_AVAILABLE:
    __all__.append("SparkBackend")

# Conditionally add RayBackend to exports if ray is installed
if _RAY_AVAILABLE:
    __all__.append("RayBackend")
