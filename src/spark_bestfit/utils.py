"""Utility functions for spark-bestfit."""

from typing import Optional

from pyspark.sql import SparkSession


def get_spark_session(spark: Optional[SparkSession] = None) -> SparkSession:
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
