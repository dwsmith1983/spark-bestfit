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


@pytest.fixture
def poisson_data():
    """Generate Poisson distribution data."""
    np.random.seed(42)
    return np.random.poisson(lam=7, size=5_000)


@pytest.fixture
def poisson_dataset(spark_session, poisson_data):
    """Create Spark DataFrame with Poisson count data."""
    return spark_session.createDataFrame([(int(x),) for x in poisson_data], ["counts"])


@pytest.fixture
def nbinom_data():
    """Generate negative binomial distribution data."""
    np.random.seed(42)
    return np.random.negative_binomial(n=5, p=0.4, size=5_000)


@pytest.fixture
def nbinom_dataset(spark_session, nbinom_data):
    """Create Spark DataFrame with negative binomial count data."""
    return spark_session.createDataFrame([(int(x),) for x in nbinom_data], ["counts"])


# Distribution fit result fixtures (shared across test modules)
@pytest.fixture
def normal_result():
    """Create a sample result for normal distribution."""
    from spark_bestfit.results import DistributionFitResult

    return DistributionFitResult(
        distribution="norm",
        parameters=[50.0, 10.0],  # loc, scale
        sse=0.005,
        aic=1500.0,
        bic=1520.0,
    )


@pytest.fixture
def gamma_result():
    """Create a sample result for gamma distribution."""
    from spark_bestfit.results import DistributionFitResult

    return DistributionFitResult(
        distribution="gamma",
        parameters=[2.0, 0.0, 2.0],  # shape, loc, scale
        sse=0.003,
        aic=1400.0,
        bic=1430.0,
    )


@pytest.fixture
def expon_result():
    """Create a sample exponential distribution result."""
    from spark_bestfit.results import DistributionFitResult

    return DistributionFitResult(
        distribution="expon",
        parameters=[0.0, 5.0],  # loc, scale
        sse=0.008,
        aic=1600.0,
        bic=1615.0,
    )


@pytest.fixture
def result_with_ks():
    """Create a result with KS statistic and p-value."""
    from spark_bestfit.results import DistributionFitResult

    return DistributionFitResult(
        distribution="norm",
        parameters=[50.0, 10.0],
        sse=0.005,
        aic=1500.0,
        bic=1520.0,
        ks_statistic=0.015,
        pvalue=0.85,
    )


@pytest.fixture
def sample_histogram():
    """Create sample histogram data."""
    np.random.seed(42)
    data = np.random.normal(50, 10, 10000)
    y_hist, x_edges = np.histogram(data, bins=50, density=True)
    x_hist = (x_edges[:-1] + x_edges[1:]) / 2
    return y_hist, x_hist


@pytest.fixture
def sample_data():
    """Create sample data array for Q-Q plots."""
    np.random.seed(42)
    return np.random.normal(50, 10, 1000)
