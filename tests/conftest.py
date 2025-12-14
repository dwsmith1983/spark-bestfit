"""Pytest configuration and fixtures for spark-bestfit tests."""

import numpy as np
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for testing.

    Scope: session - reuse across all tests for performance.
    """
    spark = (
        SparkSession.builder.appName("spark-bestfit-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")  # Disable UI for tests
        .getOrCreate()
    )

    yield spark

    spark.stop()


@pytest.fixture
def normal_data():
    """Generate normal distribution data."""
    np.random.seed(42)
    return np.random.normal(loc=50, scale=10, size=10_000)


@pytest.fixture
def exponential_data():
    """Generate exponential distribution data."""
    np.random.seed(42)
    return np.random.exponential(scale=5, size=10_000)


@pytest.fixture
def gamma_data():
    """Generate gamma distribution data."""
    np.random.seed(42)
    return np.random.gamma(shape=2.0, scale=2.0, size=10_000)


@pytest.fixture
def uniform_data():
    """Generate uniform distribution data."""
    np.random.seed(42)
    return np.random.uniform(low=0, high=100, size=10_000)


@pytest.fixture
def small_dataset(spark_session, normal_data):
    """Create small Spark DataFrame (10K rows)."""
    return spark_session.createDataFrame([(float(x),) for x in normal_data], ["value"])


@pytest.fixture
def medium_dataset(spark_session):
    """Create medium Spark DataFrame (100K rows)."""
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=10, size=100_000)
    return spark_session.createDataFrame([(float(x),) for x in data], ["value"])


@pytest.fixture
def positive_dataset(spark_session, exponential_data):
    """Create Spark DataFrame with only positive values."""
    return spark_session.createDataFrame([(float(x),) for x in exponential_data], ["value"])


@pytest.fixture
def constant_dataset(spark_session):
    """Create Spark DataFrame with constant values (edge case)."""
    data = np.full(1000, 42.0)
    return spark_session.createDataFrame([(float(x),) for x in data], ["value"])


@pytest.fixture
def empty_dataset(spark_session):
    """Create empty Spark DataFrame (edge case)."""
    return spark_session.createDataFrame([], ["value DOUBLE"])
