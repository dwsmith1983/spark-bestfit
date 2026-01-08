Distributed Sampling
====================

After fitting a distribution, you can generate samples using distributed
computing capabilities. This is particularly useful when you need to generate
millions of samples efficiently.

Basic Usage
-----------

Generate distributed samples from a fitted distribution using any backend:

.. code-block:: python

    from spark_bestfit import DistributionFitter
    from spark_bestfit.backends import BackendFactory
    from spark_bestfit.sampling import sample_distributed

    # Fit distribution
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value")
    best = results.best(n=1)[0]

    # Generate 1 million distributed samples
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = sample_distributed(
        distribution=best.distribution,
        parameters=best.parameters,
        n=1_000_000,
        backend=backend,
    )
    samples_df.show(5)

The result is a DataFrame that can be used for further processing:

.. code-block:: text

    +-------------------+
    |             sample|
    +-------------------+
    | 0.4691122931291924|
    |-0.2828633018445851|
    | 1.0093545783546243|
    |  0.582873245234523|
    | -1.23234234234234 |
    +-------------------+

Backend Options
---------------

Use any backend for distributed sampling:

.. code-block:: python

    from spark_bestfit.backends import BackendFactory
    from spark_bestfit.sampling import sample_distributed

    # Spark
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = sample_distributed(best.distribution, best.parameters, n=1_000_000, backend=backend)

    # Ray
    backend = BackendFactory.create("ray")
    samples_df = sample_distributed(best.distribution, best.parameters, n=1_000_000, backend=backend)

    # Local (for testing)
    backend = BackendFactory.create("local", max_workers=4)
    samples_df = sample_distributed(best.distribution, best.parameters, n=1_000_000, backend=backend)

Reproducibility
---------------

Use the ``random_seed`` parameter for reproducible results:

.. code-block:: python

    # Reproducible sampling
    samples1 = sample_distributed(
        best.distribution, best.parameters, n=10000,
        backend=backend, random_seed=42
    )
    samples2 = sample_distributed(
        best.distribution, best.parameters, n=10000,
        backend=backend, random_seed=42
    )
    # samples1 and samples2 will contain the same values

Each partition receives a unique seed derived from the base seed plus the partition ID,
ensuring both reproducibility and statistical independence across partitions.

Partition Control
-----------------

You can control the number of partitions for parallel sampling:

.. code-block:: python

    # Use 16 partitions for sampling
    samples_df = sample_distributed(
        distribution=best.distribution,
        parameters=best.parameters,
        n=1_000_000,
        backend=backend,
        num_partitions=16,
        random_seed=42,
    )

If not specified, the default parallelism for the backend is used.

Custom Column Names
-------------------

Specify a custom column name for the output:

.. code-block:: python

    samples_df = sample_distributed(
        distribution=best.distribution,
        parameters=best.parameters,
        n=10000,
        backend=backend,
        column_name="generated_values"
    )
    # DataFrame has column "generated_values" instead of "sample"

Local vs Distributed Sampling
-----------------------------

spark-bestfit offers two sampling methods:

.. list-table::
   :header-rows: 1

   * - Method
     - Use Case
     - Output
   * - ``sample(size=N)``
     - Small to medium samples (< 10M)
     - NumPy array
   * - ``sample_distributed(n=N, backend=...)``
     - Large samples (> 10M)
     - DataFrame (Spark/pandas)

Performance Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark results on local mode (your mileage may vary on a cluster):

.. list-table::
   :header-rows: 1

   * - N Samples
     - Local (ms)
     - Spark (ms)
     - Winner
   * - 1,000
     - 0.3
     - 336
     - Local
   * - 1,000,000
     - 16
     - 57
     - Local
   * - 10,000,000
     - 149
     - 125
     - **Spark**
   * - 50,000,000
     - 777
     - 481
     - **Spark**

**Key takeaways:**

- **Crossover point**: ~10 million samples in local mode
- **Spark overhead**: ~300ms baseline cost for job setup
- **Cluster advantage**: On a multi-node cluster, the crossover point is lower
  due to true parallelism across workers
- **Memory distribution**: Even when local is faster, distributed sampling distributes
  memory across the cluster, enabling sample sizes that wouldn't fit on a single node

Deprecated: sample_spark()
--------------------------

.. deprecated:: 2.0.0
   The ``sample_spark()`` method is deprecated and will be removed in v3.0.0.
   Use ``sample_distributed()`` with an explicit backend instead.

Migration example:

.. code-block:: python

    # Old (deprecated)
    samples_df = best.sample_spark(n=1_000_000, spark=spark)

    # New (recommended)
    from spark_bestfit.backends import BackendFactory
    from spark_bestfit.sampling import sample_distributed

    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = sample_distributed(
        distribution=best.distribution,
        parameters=best.parameters,
        n=1_000_000,
        backend=backend,
    )

API Reference
-------------

See :func:`spark_bestfit.sampling.sample_distributed` for full API documentation.
