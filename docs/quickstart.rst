Quick Start
===========

Requirements
------------

Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Spark Version
     - Python Versions
     - NumPy
     - Pandas
     - PyArrow
   * - **3.5.x**
     - 3.11, 3.12
     - 1.24+ (< 2.0)
     - 1.5+
     - 12.0 - 16.x
   * - **4.x**
     - 3.12, 3.13
     - 2.0+
     - 2.2+
     - 17.0+

.. note::
   Spark 3.5.x does not support NumPy 2.0. If using Spark 3.5 with Python 3.12,
   ensure ``setuptools`` is installed (provides ``distutils``).

Installation
------------

.. code-block:: bash

   pip install spark-bestfit

This installs spark-bestfit without PySpark. You are responsible for providing a compatible
Spark environment (see Compatibility Matrix above).

**With PySpark included** (for users without a managed Spark environment):

.. code-block:: bash

   pip install spark-bestfit[spark]

**With Ray support** (for Ray clusters and ML workflows):

.. code-block:: bash

   pip install spark-bestfit[ray]

See :doc:`/backends` for detailed backend configuration.

Basic Usage
-----------

.. code-block:: python

   from spark_bestfit import DistributionFitter
   import numpy as np
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()

   # Generate sample data
   data = np.random.normal(loc=50, scale=10, size=10_000)

   # Create fitter
   fitter = DistributionFitter(spark)
   df = spark.createDataFrame([(float(x),) for x in data], ["value"])

   # Fit distributions
   results = fitter.fit(df, column="value")

   # Get best fit (by K-S statistic, the default)
   best = results.best(n=1)[0]
   print(f"Best: {best.distribution} (KS={best.ks_statistic:.4f}, p={best.pvalue:.4f})")

   # Plot
   fitter.plot(best, df, "value", title="Best Fit Distribution")

Custom Fitting Parameters
-------------------------

Pass parameters directly to ``fit()`` to customize behavior:

.. code-block:: python

   from spark_bestfit import DistributionFitter

   fitter = DistributionFitter(spark, random_seed=123)
   results = fitter.fit(
       df,
       column="value",
       bins=100,                    # Number of histogram bins
       support_at_zero=True,        # Only fit non-negative distributions
       enable_sampling=True,        # Enable adaptive sampling
       sample_fraction=0.3,         # Sample 30% of data
       max_distributions=50,        # Limit distributions to fit
       num_partitions=16,           # Spark parallelism (None = auto)
       prefilter=True,              # Skip incompatible distributions
       lazy_metrics=True,           # Defer KS/AD computation
   )

See :doc:`/features/prefiltering` and :doc:`/features/lazy-metrics` for performance optimization.

Multi-Column Fitting
--------------------

Fit multiple columns efficiently in a single operation:

.. code-block:: python

   from spark_bestfit import DistributionFitter

   # Create DataFrame with multiple columns
   df = spark.createDataFrame([
       (1.0, 10.0, 100.0),
       (2.0, 20.0, 200.0),
       # ...
   ], ["col_a", "col_b", "col_c"])

   fitter = DistributionFitter(spark)

   # Fit all columns in one call - shares Spark overhead
   results = fitter.fit(df, columns=["col_a", "col_b", "col_c"])

   # Get results for a specific column
   col_a_results = results.for_column("col_a")
   best_a = col_a_results.best(n=1)[0]

   # Get best distribution per column
   best_per_col = results.best_per_column(n=1)
   for col_name, fits in best_per_col.items():
       print(f"{col_name}: {fits[0].distribution} (KS={fits[0].ks_statistic:.4f})")

Working with Results
--------------------

.. code-block:: python

   # Get top 5 distributions (by K-S statistic, the default)
   top_5 = results.best(n=5)

   # Get best by other metrics
   best_sse = results.best(n=1, metric="sse")[0]
   best_aic = results.best(n=1, metric="aic")[0]
   best_ad = results.best(n=1, metric="ad_statistic")[0]

   # Filter by goodness-of-fit
   good_fits = results.filter(ks_threshold=0.05)        # K-S statistic < 0.05
   significant = results.filter(pvalue_threshold=0.05)  # p-value > 0.05
   good_ad = results.filter(ad_threshold=1.0)           # A-D statistic < 1.0

   # Convert to pandas for analysis
   df_pandas = results.df.toPandas()

   # Use fitted distribution
   samples = best.sample(size=10000)  # Generate samples
   pdf_values = best.pdf(x_array)     # Evaluate PDF
   cdf_values = best.cdf(x_array)     # Evaluate CDF

   # Access all goodness-of-fit metrics
   print(f"K-S: {best.ks_statistic}, p-value: {best.pvalue}")
   print(f"A-D: {best.ad_statistic}, A-D p-value: {best.ad_pvalue}")

Parameter Confidence Intervals
------------------------------

Compute bootstrap confidence intervals for fitted distribution parameters:

.. code-block:: python

   # Get the best fit
   best = results.best(n=1)[0]

   # Compute 95% confidence intervals
   ci = best.confidence_intervals(
       df,
       column="value",
       alpha=0.05,              # 95% CI (default)
       n_bootstrap=1000,        # Number of bootstrap samples
       random_seed=42,          # For reproducibility
   )

   # Display results
   print(f"Distribution: {best.distribution}")
   print(f"Parameters: {best.get_param_names()}")
   for param, (lower, upper) in ci.items():
       print(f"  {param}: [{lower:.4f}, {upper:.4f}]")

.. note::
   The ``confidence_intervals()`` method automatically samples large DataFrames (default
   max 10,000 rows) to avoid driver memory issues.

Visualization
-------------

**Distribution plot:**

.. code-block:: python

   fitter.plot(
       best, df, "value",
       figsize=(16, 10),
       title="Distribution Fit",
       save_path="output/distribution.png",
   )

**Q-Q plot** (quantile-quantile):

.. code-block:: python

   fitter.plot_qq(
       best, df, "value",
       max_points=1000,
       title="Q-Q Plot",
       save_path="output/qq_plot.png",
   )

**P-P plot** (probability-probability):

.. code-block:: python

   fitter.plot_pp(
       best, df, "value",
       max_points=1000,
       title="P-P Plot",
       save_path="output/pp_plot.png",
   )

Discrete Distributions
----------------------

For count data (integers), use ``DiscreteDistributionFitter``:

.. code-block:: python

   from spark_bestfit import DiscreteDistributionFitter
   import numpy as np

   # Generate count data
   data = np.random.poisson(lam=7, size=10_000)
   df = spark.createDataFrame([(int(x),) for x in data], ["counts"])

   # Fit discrete distributions
   fitter = DiscreteDistributionFitter(spark)
   results = fitter.fit(df, column="counts")

   # Get best fit - use AIC for model selection (recommended)
   best = results.best(n=1, metric="aic")[0]
   print(f"Best: {best.distribution} (AIC={best.aic:.2f})")

**Metric selection for discrete:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Metric
     - Use Case
   * - ``aic``
     - **Recommended** - Model selection with complexity penalty
   * - ``bic``
     - Stronger penalty for complex models
   * - ``ks_statistic``
     - Valid for ranking, but p-values unreliable for discrete data
   * - ``sse``
     - Simple comparison metric

.. note::
   K-S and A-D tests assume continuous distributions. For discrete data, use AIC/BIC
   for proper model selection.

Excluding Distributions
-----------------------

By default, slow distributions are excluded. To customize:

.. code-block:: python

   from spark_bestfit import DistributionFitter, DEFAULT_EXCLUDED_DISTRIBUTIONS

   # View default exclusions
   print(DEFAULT_EXCLUDED_DISTRIBUTIONS)

   # Include a specific distribution
   exclusions = tuple(d for d in DEFAULT_EXCLUDED_DISTRIBUTIONS if d != "wald")
   fitter = DistributionFitter(spark, excluded_distributions=exclusions)

   # Exclude nothing (fit all - may be slow)
   fitter = DistributionFitter(spark, excluded_distributions=())

Next Steps
----------

- :doc:`/backends` - Backend configuration (Spark, Ray, Local)
- :doc:`/features/bounded` - Bounded distribution fitting
- :doc:`/features/sampling` - Distributed sampling
- :doc:`/features/copula` - Correlated multi-column sampling
- :doc:`/performance` - Performance tuning and benchmarks
