"""Utility functions for spark-bestfit."""

from typing import Optional

# PySpark is optional - only import if available
try:
    from pyspark.sql import SparkSession

    _PYSPARK_AVAILABLE = True
except ImportError:
    SparkSession = None  # type: ignore[assignment,misc]
    _PYSPARK_AVAILABLE = False


def get_spark_session(spark: Optional["SparkSession"] = None) -> "SparkSession":
    """Get or create a SparkSession.

    If a SparkSession is provided, it is returned as-is.
    If None is provided, attempts to get the active SparkSession.

    Args:
        spark: Optional SparkSession. If None, gets the active session.

    Returns:
        SparkSession instance

    Raises:
        RuntimeError: If no SparkSession is provided and no active session exists

    Example:
        >>> # Use existing session
        >>> spark = SparkSession.builder.appName("my-app").getOrCreate()
        >>> session = get_spark_session(spark)
        >>>
        >>> # Use active session
        >>> session = get_spark_session()  # Gets active session
    """
    if not _PYSPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required but not installed. "
            "Install with: pip install spark-bestfit[spark]\n"
            "Or use a non-Spark backend: LocalBackend() or RayBackend()"
        )

    if spark is not None:
        return spark

    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    raise RuntimeError(
        "No SparkSession provided and no active session found. "
        "Please create a SparkSession first:\n"
        "  spark = SparkSession.builder.appName('my-app').getOrCreate()"
    )
