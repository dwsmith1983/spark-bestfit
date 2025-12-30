"""Distributed sampling for fitted distributions."""

from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from spark_bestfit.utils import get_spark_session


def create_sample_udf_func(
    distribution: str,
    parameters: List[float],
    samples_per_partition: int,
    random_seed: Optional[int],
):
    """Create a function for generating samples in a Pandas UDF.

    Args:
        distribution: scipy.stats distribution name
        parameters: Distribution parameters
        samples_per_partition: Number of samples each partition should generate
        random_seed: Base random seed (partition id will be added for uniqueness)

    Returns:
        Function suitable for use with mapInPandas
    """

    def generate_samples(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Generate samples for each partition."""
        dist = getattr(st, distribution)

        for pdf in iterator:
            # Get partition id from the dataframe (we pass it as a column)
            if len(pdf) == 0:
                continue

            partition_id = pdf["partition_id"].iloc[0]

            # Create unique seed for this partition
            if random_seed is not None:
                np.random.seed(random_seed + partition_id)

            # Generate samples
            samples = dist.rvs(*parameters, size=samples_per_partition)

            yield pd.DataFrame({"sample": samples})

    return generate_samples


def sample_spark(
    distribution: str,
    parameters: List[float],
    n: int,
    spark: Optional[SparkSession] = None,
    num_partitions: Optional[int] = None,
    random_seed: Optional[int] = None,
    column_name: str = "sample",
) -> DataFrame:
    """Generate distributed samples from a fitted distribution.

    Uses Spark's parallelism to generate samples across the cluster,
    enabling generation of millions of samples efficiently.

    Args:
        distribution: scipy.stats distribution name
        parameters: Distribution parameters (shape, loc, scale)
        n: Total number of samples to generate
        spark: SparkSession. If None, uses the active session.
        num_partitions: Number of partitions to use. Defaults to spark default parallelism.
        random_seed: Random seed for reproducibility. Each partition uses seed + partition_id.
        column_name: Name for the output column (default: "sample")

    Returns:
        Spark DataFrame with single column containing samples

    Example:
        >>> df = sample_spark("norm", [0.0, 1.0], n=1_000_000, spark=spark)
        >>> df.show(5)
        +-------------------+
        |             sample|
        +-------------------+
        | 0.4691122931291924|
        |-0.2828633018445851|
        | 1.0093545783546243|
        +-------------------+
    """
    spark = get_spark_session(spark)
    if num_partitions is None:
        num_partitions = spark.sparkContext.defaultParallelism

    # Calculate samples per partition (distribute evenly)
    base_samples = n // num_partitions
    remainder = n % num_partitions

    # Create a driver dataframe with partition info
    partition_data = []
    for i in range(num_partitions):
        # First 'remainder' partitions get one extra sample
        samples_for_partition = base_samples + (1 if i < remainder else 0)
        if samples_for_partition > 0:
            partition_data.append((i, samples_for_partition))

    # Create DataFrame with partition assignments
    partition_df = spark.createDataFrame(
        partition_data,
        StructType(
            [
                StructField("partition_id", IntegerType(), False),
                StructField("n_samples", IntegerType(), False),
            ]
        ),
    )

    # Repartition to ensure each row goes to its own partition
    partition_df = partition_df.repartition(len(partition_data))

    # Define output schema
    output_schema = StructType([StructField(column_name, DoubleType(), False)])

    # Get distribution object
    dist = getattr(st, distribution)

    def generate_samples_for_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Generate samples for each partition."""
        for pdf in iterator:
            if len(pdf) == 0:
                continue

            for _, row in pdf.iterrows():
                n_samples = int(row["n_samples"])
                partition_id = int(row["partition_id"])

                # Create unique seed for this partition
                if random_seed is not None:
                    rng = np.random.default_rng(random_seed + partition_id)
                    # Use the rng to generate scipy samples
                    samples = dist.rvs(*parameters, size=n_samples, random_state=rng)
                else:
                    samples = dist.rvs(*parameters, size=n_samples)

                yield pd.DataFrame({column_name: samples})

    # Apply the UDF
    result_df = partition_df.mapInPandas(generate_samples_for_partition, schema=output_schema)

    return result_df
