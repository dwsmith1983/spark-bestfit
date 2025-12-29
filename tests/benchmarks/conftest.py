"""Pytest configuration and fixtures for performance benchmarks.

These benchmarks are excluded from normal CI runs.
Run with: make benchmark
"""

import numpy as np
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for benchmarks.

    Uses local[*] for maximum parallelism during benchmarks.
    """
    spark = (
        SparkSession.builder.appName("spark-bestfit-benchmarks")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    # Warmup: run a small fit to trigger JVM JIT compilation and Spark initialization
    from spark_bestfit import DistributionFitter

    warmup_data = generate_normal_data(1_000, seed=0)
    warmup_df = spark.createDataFrame([(float(x),) for x in warmup_data], ["value"])
    fitter = DistributionFitter(spark)
    _ = fitter.fit(warmup_df, "value", max_distributions=5, num_partitions=2)
    print("\n[Warmup complete - JVM/Spark initialized]")

    yield spark

    spark.stop()


def generate_normal_data(size: int, seed: int = 42) -> np.ndarray:
    """Generate normal distribution data of specified size."""
    np.random.seed(seed)
    return np.random.normal(loc=50, scale=10, size=size)


def generate_poisson_data(size: int, seed: int = 42) -> np.ndarray:
    """Generate Poisson distribution data of specified size."""
    np.random.seed(seed)
    return np.random.poisson(lam=7, size=size)


@pytest.fixture
def data_10k():
    """Generate 10K rows of normal data."""
    return generate_normal_data(10_000)


@pytest.fixture
def data_25k():
    """Generate 25K rows of normal data."""
    return generate_normal_data(25_000)


@pytest.fixture
def data_100k():
    """Generate 100K rows of normal data."""
    return generate_normal_data(100_000)


@pytest.fixture
def data_500k():
    """Generate 500K rows of normal data."""
    return generate_normal_data(500_000)


@pytest.fixture
def data_1m():
    """Generate 1M rows of normal data."""
    return generate_normal_data(1_000_000)


@pytest.fixture
def df_10k(spark_session, data_10k):
    """Create 10K row DataFrame."""
    return spark_session.createDataFrame([(float(x),) for x in data_10k], ["value"])


@pytest.fixture
def df_25k(spark_session, data_25k):
    """Create 25K row DataFrame."""
    return spark_session.createDataFrame([(float(x),) for x in data_25k], ["value"])


@pytest.fixture
def df_100k(spark_session, data_100k):
    """Create 100K row DataFrame."""
    return spark_session.createDataFrame([(float(x),) for x in data_100k], ["value"])


@pytest.fixture
def df_500k(spark_session, data_500k):
    """Create 500K row DataFrame."""
    return spark_session.createDataFrame([(float(x),) for x in data_500k], ["value"])


@pytest.fixture
def df_1m(spark_session, data_1m):
    """Create 1M row DataFrame."""
    return spark_session.createDataFrame([(float(x),) for x in data_1m], ["value"])


@pytest.fixture
def discrete_data_10k():
    """Generate 10K rows of Poisson data."""
    return generate_poisson_data(10_000)


@pytest.fixture
def discrete_df_10k(spark_session, discrete_data_10k):
    """Create 10K row DataFrame with discrete data."""
    return spark_session.createDataFrame([(int(x),) for x in discrete_data_10k], ["counts"])


# Multi-column fixtures for efficiency benchmarks
def generate_multi_column_data(size: int, seed: int = 42) -> list:
    """Generate data for 3 columns with different distributions."""
    np.random.seed(seed)
    normal = np.random.normal(loc=50, scale=10, size=size)
    exponential = np.random.exponential(scale=5, size=size)
    gamma = np.random.gamma(shape=2.0, scale=2.0, size=size)
    return [(float(n), float(e), float(g)) for n, e, g in zip(normal, exponential, gamma)]


def generate_multi_column_discrete_data(size: int, seed: int = 42) -> list:
    """Generate data for 3 discrete columns with different distributions."""
    np.random.seed(seed)
    poisson = np.random.poisson(lam=7, size=size)
    nbinom = np.random.negative_binomial(n=5, p=0.4, size=size)
    geom = np.random.geometric(p=0.25, size=size)
    return [(int(p), int(n), int(g)) for p, n, g in zip(poisson, nbinom, geom)]


@pytest.fixture
def df_multi_3col_10k(spark_session):
    """Create 10K row DataFrame with 3 continuous columns."""
    data = generate_multi_column_data(10_000)
    return spark_session.createDataFrame(data, ["col_normal", "col_exp", "col_gamma"])


@pytest.fixture
def df_multi_3col_100k(spark_session):
    """Create 100K row DataFrame with 3 continuous columns."""
    data = generate_multi_column_data(100_000)
    return spark_session.createDataFrame(data, ["col_normal", "col_exp", "col_gamma"])


@pytest.fixture
def discrete_df_multi_3col_10k(spark_session):
    """Create 10K row DataFrame with 3 discrete columns."""
    data = generate_multi_column_discrete_data(10_000)
    return spark_session.createDataFrame(data, ["col_poisson", "col_nbinom", "col_geom"])
