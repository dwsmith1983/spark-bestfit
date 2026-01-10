FAQ & Troubleshooting
=====================

Frequently asked questions and solutions to common issues.

General Questions
-----------------

Which metric should I use for model selection?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For continuous distributions:**

- **K-S statistic** (default): Best for general-purpose use. Low values indicate
  better fit. Sensitive to both location and shape differences.
- **A-D statistic**: More sensitive to tail behavior. Use when tail accuracy matters
  (e.g., risk analysis, extreme value modeling).
- **AIC/BIC**: Use for model selection when comparing distributions with different
  numbers of parameters. Penalizes model complexity.
- **SSE**: Sum of squared errors between fitted PDF and histogram. Fast but less
  statistically rigorous.

**For discrete distributions:**

Use **AIC** or **BIC**. K-S and A-D tests assume continuous distributions and their
p-values are unreliable for discrete data.

.. code-block:: python

   # Continuous: KS is default
   best = results.best(n=1)[0]

   # Discrete: Use AIC
   best = results.best(n=1, metric="aic")[0]

What's the difference between K-S and A-D statistics?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both are goodness-of-fit tests that measure how well a distribution fits data:

- **K-S (Kolmogorov-Smirnov)**: Measures the maximum vertical distance between the
  empirical and theoretical CDFs. Equally weighted across the entire distribution.

- **A-D (Anderson-Darling)**: Weights the tails more heavily than K-S. Better for
  detecting deviations in the distribution tails.

**Rule of thumb**: Use K-S for general fitting; use A-D when tail behavior matters
(financial risk, extreme events, etc.).

Which backend should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Backend
     - Best For
     - Requirements
   * - **Local**
     - Development, small data (<100K rows)
     - None (default)
   * - **Spark**
     - Production, large data, existing clusters
     - PySpark 3.5+ or 4.x
   * - **Ray**
     - ML pipelines, Ray ecosystem, quick scaling
     - Ray 2.x

See :doc:`backends` for setup details.

Performance Issues
------------------

Fitting is slow - how can I speed it up?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Enable lazy metrics** (biggest impact for model selection):

.. code-block:: python

   results = fitter.fit(df, "value", lazy_metrics=True)
   best = results.best(n=1, metric="aic")[0]  # Fast!

**2. Enable prefiltering** (skip incompatible distributions early):

.. code-block:: python

   results = fitter.fit(df, "value", prefilter=True)

**3. Reduce the number of distributions**:

.. code-block:: python

   results = fitter.fit(df, "value", max_distributions=30)

**4. Use sampling for large datasets**:

.. code-block:: python

   results = fitter.fit(df, "value", enable_sampling=True, sample_fraction=0.1)

**5. Combine optimizations** with FitterConfig:

.. code-block:: python

   from spark_bestfit import FitterConfigBuilder

   config = (FitterConfigBuilder()
       .with_lazy_metrics()
       .with_prefilter()
       .with_max_distributions(30)
       .with_sampling(fraction=0.1)
       .build())

   results = fitter.fit(df, "value", config=config)

I'm getting OutOfMemory errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For driver memory issues:**

1. Use sampling to reduce data collected to driver:

   .. code-block:: python

      results = fitter.fit(df, "value", enable_sampling=True, sample_fraction=0.3)

2. For confidence intervals, limit the sample size:

   .. code-block:: python

      ci = best.confidence_intervals(df, "value", max_rows=5000)

**For executor memory issues:**

1. Increase partitions to reduce partition size:

   .. code-block:: python

      results = fitter.fit(df, "value", num_partitions=200)

2. With Spark, increase executor memory in your cluster config.

Fitting fails with "RuntimeError: DataFrame unpersisted"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This happens when using lazy metrics and accessing KS/AD statistics after the
source DataFrame has been garbage collected or unpersisted.

**Solution**: Materialize before unpersisting:

.. code-block:: python

   results = fitter.fit(df, "value", lazy_metrics=True)

   # Do your fast model selection
   best_aic = results.best(n=1, metric="aic")[0]

   # Materialize BEFORE unpersisting source data
   materialized = results.materialize()

   # Now safe to unpersist
   df.unpersist()

   # Access any metric
   best_ks = materialized.best(n=1, metric="ks_statistic")[0]

Data-Related Issues
-------------------

My data has extreme outliers - what should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1**: Enable heavy-tail detection warnings:

.. code-block:: python

   # spark-bestfit automatically warns if data appears heavy-tailed
   results = fitter.fit(df, "value")
   # Warning: Data may be heavy-tailed. Consider extreme value distributions.

**Option 2**: Use MSE estimation for heavy-tailed data:

.. code-block:: python

   from spark_bestfit import FitterConfigBuilder

   config = FitterConfigBuilder().with_mse_estimation().build()
   results = fitter.fit(df, "value", config=config)

See :doc:`features/heavy-tail` and :doc:`features/mse-estimation` for details.

My data is bounded (e.g., percentages, probabilities)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use bounded distribution fitting to constrain fits to valid ranges:

.. code-block:: python

   # Fit distributions bounded between 0 and 1 (e.g., probabilities)
   results = fitter.fit(df, "probability", lower_bound=0.0, upper_bound=1.0)

   # The fitted distribution will be truncated to [0, 1]
   best = results.best(n=1)[0]
   samples = best.sample(1000)  # All samples in [0, 1]

See :doc:`features/bounded` for details.

My data is non-negative (counts, durations, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``support_at_zero=True`` to only fit distributions defined on [0, infinity):

.. code-block:: python

   # Only fit non-negative distributions
   results = fitter.fit(df, "duration", support_at_zero=True)

No distribution fits well (all have high K-S statistics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can happen when:

1. **Data is multimodal**: scipy distributions are typically unimodal. Consider:
   - Fitting each mode separately
   - Using mixture models (outside spark-bestfit scope)

2. **Data is discrete but you're using continuous fitter**: Use
   ``DiscreteDistributionFitter`` for count data.

3. **Data has a complex shape**: Try:
   - Enabling prefiltering to focus on compatible distributions
   - Checking the histogram to understand the data shape

4. **Data needs transformation**: Consider log-transform for skewed data:

   .. code-block:: python

      import pyspark.sql.functions as F

      df_log = df.withColumn("value_log", F.log("value"))
      results = fitter.fit(df_log, "value_log")

Plotting Issues
---------------

I get "matplotlib not found" error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the plotting extra:

.. code-block:: bash

   pip install spark-bestfit[plotting]

Or manually: ``pip install matplotlib``

Plots look wrong or distorted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Histogram doesn't match data**: Increase the number of bins:

   .. code-block:: python

      results = fitter.fit(df, "value", bins=200)

2. **Q-Q/P-P plots have too few points**: Increase max_points:

   .. code-block:: python

      fitter.plot_qq(best, df, "value", max_points=5000)

Serialization Issues
--------------------

Can I save and load fitted results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, use the serialization module:

.. code-block:: python

   from spark_bestfit.serialization import save_results, load_results

   # Save results
   save_results(results, "fitted_results.pkl")

   # Load results later
   loaded_results = load_results("fitted_results.pkl")
   best = loaded_results.best(n=1)[0]

See :doc:`features/serialization` for details.

Backend-Specific Issues
-----------------------

Spark: "No module named 'pyspark'"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySpark is optional. Install it explicitly:

.. code-block:: bash

   pip install spark-bestfit[spark]

Or install PySpark separately if using a managed cluster.

Ray: Backend not available
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Ray support:

.. code-block:: bash

   pip install spark-bestfit[ray]

Note: Ray requires pyarrow for ``ray.data`` operations.

Getting Help
------------

If you encounter issues not covered here:

1. Check the :doc:`api` documentation
2. Review the :doc:`migration` guide if upgrading versions
3. Report issues at https://github.com/dwsmith1983/spark-bestfit/issues
