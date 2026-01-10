FAQ & Troubleshooting
=====================

Frequently asked questions and troubleshooting tips for spark-bestfit.

Installation Issues
-------------------

**Q: ModuleNotFoundError: No module named 'pyspark'**

PySpark is an optional dependency. Install it with:

.. code-block:: bash

    pip install spark-bestfit[spark]

Or use the local backend which doesn't require Spark:

.. code-block:: python

    from spark_bestfit.backends import BackendFactory

    backend = BackendFactory.create("local", max_workers=4)

**Q: ImportError with Ray**

Ray is also optional. Install it with:

.. code-block:: bash

    pip install spark-bestfit[ray]

Fitting Issues
--------------

**Q: Why do I get a "heavy-tail characteristics" warning?**

This warning appears when your data has high kurtosis, suggesting heavy-tailed distributions
(like Pareto, Cauchy, or Student's t) may fit better than standard distributions.

Solutions:

1. Use Maximum Spacing Estimation (MSE) for robust fitting:

   .. code-block:: python

       results = fitter.fit(df, column="value", estimation_method="mse")

2. Filter to heavy-tail specific distributions:

   .. code-block:: python

       results = fitter.fit(
           df,
           column="value",
           included_distributions=["pareto", "cauchy", "t", "levy", "burr"]
       )

3. Transform your data (log, sqrt) to reduce tail effects

See :doc:`features/heavy-tail` for detailed guidance.

**Q: My fits have poor p-values (< 0.05)**

Low p-values indicate the data may not follow the fitted distribution well. Consider:

1. **Check data quality**: Remove outliers or invalid values
2. **Try more distributions**: Use ``included_distributions=None`` to test all ~90 distributions
3. **Use bounded fitting**: If your data has natural bounds (e.g., positive values):

   .. code-block:: python

       results = fitter.fit(df, column="value", lower_bound=0)

4. **Check sample size**: Very large samples may reject good fits due to statistical power

**Q: Fitting is slow**

Several strategies to improve performance:

1. **Use prefiltering** to skip unlikely distributions:

   .. code-block:: python

       results = fitter.fit(df, column="value", prefilter=True)

2. **Reduce distribution count**:

   .. code-block:: python

       # Only fit common distributions
       common = ["norm", "gamma", "lognorm", "expon", "weibull_min"]
       results = fitter.fit(df, column="value", included_distributions=common)

3. **Use appropriate backend** for your data size:

   - < 1M rows: Local backend
   - 1M-100M rows: Ray backend
   - > 100M rows: Spark backend

4. **Skip expensive metrics** with lazy evaluation:

   .. code-block:: python

       from spark_bestfit import FitterConfig

       config = FitterConfig().skip_ks_test(True).skip_ad_test(True)
       fitter = DistributionFitter(spark, config=config)

See :doc:`performance` for benchmarks and tuning advice.

Sampling Issues
---------------

**Q: Copula sampling is slow**

The bottleneck is usually the marginal distribution transforms (PPF/inverse CDF).

1. **Use return_uniform=True** if you only need correlation structure:

   .. code-block:: python

       # 20x faster - returns uniform [0,1] samples
       samples = copula.sample(n=1_000_000, return_uniform=True)

2. **Use common distributions** that have fast PPF implementations:
   norm, expon, uniform, lognorm, weibull_min, gamma, beta

3. **Use distributed sampling** for large sample counts:

   .. code-block:: python

       backend = BackendFactory.create("spark", spark_session=spark)
       samples_df = copula.sample_distributed(n=100_000_000, backend=backend)

**Q: Samples don't match my original data distribution**

Verify your fit quality before sampling:

.. code-block:: python

    # Check goodness-of-fit metrics
    best = results.best(n=1)[0]
    print(f"K-S statistic: {best.ks_statistic}")
    print(f"p-value: {best.pvalue}")

    # Visual inspection
    best.diagnostics()  # Shows Q-Q, P-P, histogram, and CDF plots

Memory Issues
-------------

**Q: OutOfMemoryError when fitting large data**

1. **Don't collect to driver** - Use distributed operations:

   .. code-block:: python

       # Good - stays distributed
       results = fitter.fit(df, column="value")

       # Bad - collects all data to driver
       pandas_df = df.toPandas()

2. **Use lazy metrics** to defer computation:

   .. code-block:: python

       config = FitterConfig().skip_ks_test(True).skip_ad_test(True)

3. **Increase Spark driver memory**:

   .. code-block:: python

       spark = SparkSession.builder \
           .config("spark.driver.memory", "8g") \
           .getOrCreate()

**Q: Memory issues with copula sampling**

For very large sample counts, use distributed sampling instead of local:

.. code-block:: python

    # May OOM for n > 10M
    samples = copula.sample(n=100_000_000)

    # Better - distributed across cluster
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = copula.sample_distributed(n=100_000_000, backend=backend)

Serialization Issues
--------------------

**Q: SerializationError when loading a saved model**

Common causes:

1. **Missing required fields** - Ensure JSON has ``distribution`` and ``parameters``
2. **Unknown distribution** - The distribution name must exist in scipy.stats
3. **Version mismatch** - Check ``spark_bestfit_version`` in the JSON file

**Q: Can I load models saved with an older version?**

Yes, spark-bestfit maintains backward compatibility. The ``schema_version`` field tracks
the serialization format. Models saved with v1.x should load in v2.x.

Backend Issues
--------------

**Q: How do I switch backends?**

Use the ``BackendFactory`` for backend-agnostic code:

.. code-block:: python

    from spark_bestfit.backends import BackendFactory

    # Development/testing
    backend = BackendFactory.create("local", max_workers=4)

    # Production with Spark
    backend = BackendFactory.create("spark", spark_session=spark)

    # ML workflows with Ray
    backend = BackendFactory.create("ray")

**Q: Spark jobs hang or fail silently**

1. Check Spark UI (usually http://localhost:4040) for job status
2. Increase executor memory for large data:

   .. code-block:: python

       spark = SparkSession.builder \
           .config("spark.executor.memory", "4g") \
           .getOrCreate()

3. Ensure Spark version compatibility (3.5.x or 4.x)

Getting Help
------------

If your issue isn't covered here:

1. Check the :doc:`api` documentation for method signatures and parameters
2. Review :doc:`migration` for breaking changes between versions
3. Open an issue at https://github.com/dwsmith1983/spark-bestfit/issues
