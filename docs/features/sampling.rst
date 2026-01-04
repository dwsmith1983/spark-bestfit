Distributed Sampling
====================

After fitting a distribution, you can generate samples using Spark's distributed
computing capabilities. This is particularly useful when you need to generate
millions of samples efficiently.

Basic Usage
-----------

Generate distributed samples from a fitted distribution:

.. code-block:: python

    from spark_bestfit import DistributionFitter

    # Fit distribution
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, column="value")
    best = results.best(n=1)[0]

    # Generate 1 million distributed samples
    samples_df = best.sample_spark(n=1_000_000, spark=spark)
    samples_df.show(5)

The result is a Spark DataFrame that can be used for further processing:

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

Reproducibility
---------------

Use the ``random_seed`` parameter for reproducible results:

.. code-block:: python

    # Reproducible sampling
    samples1 = best.sample_spark(n=10000, spark=spark, random_seed=42)
    samples2 = best.sample_spark(n=10000, spark=spark, random_seed=42)
    # samples1 and samples2 will contain the same values

Each partition receives a unique seed derived from the base seed plus the partition ID,
ensuring both reproducibility and statistical independence across partitions.

Partition Control
-----------------

You can control the number of partitions for parallel sampling:

.. code-block:: python

    # Use 16 partitions for sampling
    samples_df = best.sample_spark(
        n=1_000_000,
        spark=spark,
        num_partitions=16,
        random_seed=42,
    )

If not specified, the default Spark parallelism is used.

Custom Column Names
-------------------

Specify a custom column name for the output:

.. code-block:: python

    samples_df = best.sample_spark(
        n=10000,
        spark=spark,
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
   * - ``sample_spark(n=N, spark=spark)``
     - Large samples (> 10M)
     - Spark DataFrame

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
- **Memory distribution**: Even when local is faster, ``sample_spark()`` distributes
  memory across the cluster, enabling sample sizes that wouldn't fit on a single node

API Reference
-------------

See :meth:`spark_bestfit.results.DistributionFitResult.sample_spark` for full
API documentation.
