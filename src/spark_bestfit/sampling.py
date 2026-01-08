"""Distributed sampling for fitted distributions.

This module provides functions for generating samples from fitted distributions
using the backend abstraction for distributed or local execution.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import scipy.stats as st

if TYPE_CHECKING:
    from spark_bestfit.protocols import ExecutionBackend


def sample_distributed(
    distribution: str,
    parameters: List[float],
    n: int,
    backend: "ExecutionBackend",
    num_partitions: Optional[int] = None,
    random_seed: Optional[int] = None,
    column_name: str = "sample",
) -> Any:
    """Generate samples from a fitted distribution using backend abstraction.

    Uses the backend's parallelism to generate samples, enabling generation
    of millions of samples efficiently with SparkBackend or local execution
    with LocalBackend.

    Args:
        distribution: scipy.stats distribution name (e.g., "norm", "expon")
        parameters: Distribution parameters (shape, loc, scale)
        n: Total number of samples to generate
        backend: Execution backend (SparkBackend, LocalBackend, etc.)
        num_partitions: Number of partitions to use. Defaults to backend parallelism.
        random_seed: Random seed for reproducibility. Each partition uses seed + partition_id.
        column_name: Name for the output column (default: "sample")

    Returns:
        Backend-specific DataFrame with single column containing samples
        (Spark DataFrame for SparkBackend, pandas DataFrame for LocalBackend)

    Example:
        >>> from spark_bestfit.backends.spark import SparkBackend
        >>> backend = SparkBackend(spark)
        >>> df = sample_distributed("norm", [0.0, 1.0], n=1_000_000, backend=backend)
        >>> df.show(5)
        +-------------------+
        |             sample|
        +-------------------+
        | 0.4691122931291924|
        |-0.2828633018445851|
        | 1.0093545783546243|
        +-------------------+
    """
    # Get distribution object from scipy
    dist = getattr(st, distribution)

    def generate_distribution_samples(
        n_samples: int,
        partition_id: int,
        seed: Optional[int],
    ) -> Dict[str, np.ndarray]:
        """Generate samples from the distribution.

        This function is passed to the backend's generate_samples method.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            samples = dist.rvs(*parameters, size=n_samples, random_state=rng)
        else:
            samples = dist.rvs(*parameters, size=n_samples)

        return {column_name: samples}

    return backend.generate_samples(
        n=n,
        generator_func=generate_distribution_samples,
        column_names=[column_name],
        num_partitions=num_partitions,
        random_seed=random_seed,
    )


def sample_spark(
    distribution: str,
    parameters: List[float],
    n: int,
    spark: Optional[Any] = None,
    num_partitions: Optional[int] = None,
    random_seed: Optional[int] = None,
    column_name: str = "sample",
) -> Any:
    """Generate distributed samples from a fitted distribution using Spark.

    .. deprecated:: 2.0.0
        Will be removed in v3.0.0. Use :func:`sample_distributed` with
        ``SparkBackend`` instead.

    This is a backward-compatible wrapper around sample_distributed().

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
    warnings.warn(
        "sample_spark() is deprecated and will be removed in v3.0.0. "
        "Use sample_distributed(backend=SparkBackend()) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from spark_bestfit.backends.spark import SparkBackend

    backend = SparkBackend(spark)
    return sample_distributed(
        distribution=distribution,
        parameters=parameters,
        n=n,
        backend=backend,
        num_partitions=num_partitions,
        random_seed=random_seed,
        column_name=column_name,
    )
