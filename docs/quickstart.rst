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
   Spark 3.5.x does not support NumPy 2.0. If using Spark 3.5 with Python 3.12, ensure ``setuptools`` is installed (provides ``distutils``).

Installation
------------

.. code-block:: bash

   pip install spark-bestfit

This installs spark-bestfit without PySpark. You are responsible for providing a compatible
Spark environment (see Compatibility Matrix above).

**With PySpark included** (for users without a managed Spark environment):

.. code-block:: bash

   pip install spark-bestfit[spark]

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
   )

Parallelism Control
-------------------

The ``num_partitions`` parameter controls how many Spark partitions are used for parallel
distribution fitting. Each partition fits a subset of distributions independently.

**Default behavior (num_partitions=None):**

The library auto-calculates partitions using: ``min(num_distributions, 2 × defaultParallelism)``

- ``num_distributions``: Number of distributions being fitted
- ``defaultParallelism``: Spark's default parallelism (typically equals total executor cores)

**When to override:**

.. code-block:: python

   # Large cluster with many executors - increase parallelism
   results = fitter.fit(df, "value", num_partitions=64)

   # Resource-constrained environment - reduce parallelism
   results = fitter.fit(df, "value", num_partitions=4)

   # Fitting few distributions - let auto-calculate handle it
   results = fitter.fit(df, "value", max_distributions=10)  # num_partitions auto-set

.. note::
   Setting ``num_partitions`` higher than the number of distributions has no benefit,
   as each distribution requires exactly one task.

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

Custom Plotting
---------------

.. code-block:: python

   # Basic plot
   fitter.plot(best, df, "value", title="Distribution Fit")

   # Customized plot
   fitter.plot(
       best,
       df,
       "value",
       figsize=(16, 10),
       dpi=300,
       histogram_alpha=0.6,
       pdf_linewidth=3,
       title="Distribution Fit",
       xlabel="Value",
       ylabel="Density",
       save_path="output/distribution.png",
   )

Q-Q Plots
---------

Q-Q (quantile-quantile) plots provide visual assessment of goodness-of-fit by comparing
sample quantiles against theoretical quantiles. Points close to the diagonal indicate a good fit.

.. code-block:: python

   # Q-Q plot for goodness-of-fit assessment
   fitter.plot_qq(
       best,
       df,
       "value",
       max_points=1000,           # Sample size for plotting
       figsize=(10, 10),
       title="Q-Q Plot",
       save_path="output/qq_plot.png",
   )


P-P Plots
---------

P-P (probability-probability) plots compare the empirical cumulative distribution function (CDF)
of the sample data against the theoretical CDF of the fitted distribution. They are
particularly useful for assessing the fit in the center of the distribution.

.. code-block:: python

   # P-P plot for goodness-of-fit assessment
   fitter.plot_pp(
       best,
       df,
       "value",
       max_points=1000,           # Sample size for plotting
       figsize=(10, 10),
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

   # Plot fitted PMF
   fitter.plot(best, df, "counts", title="Best Discrete Fit")

Metric Selection for Discrete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Metric
     - Use Case
   * - ``aic``
     - **Recommended** - Proper model selection criterion with complexity penalty
   * - ``bic``
     - Similar to AIC but stronger penalty for complex models
   * - ``ks_statistic``
     - Valid for ranking fits, but p-values are not reliable for discrete data
   * - ``ad_statistic``
     - Valid for ranking fits (not computed for discrete distributions)
   * - ``sse``
     - Simple comparison metric

.. note::
   The K-S and A-D tests assume continuous distributions. For discrete data, the K-S
   statistic can still rank fits, but p-values are conservative and should not be used
   for hypothesis testing. A-D statistics are not computed for discrete distributions.
   Use AIC/BIC for proper model selection.

Anderson-Darling Test
---------------------

The Anderson-Darling (A-D) test provides an alternative to the Kolmogorov-Smirnov test with
more weight on the tails of the distribution. Lower A-D statistics indicate better fits.

.. code-block:: python

   # Get best by A-D statistic
   best_ad = results.best(n=1, metric="ad_statistic")[0]
   print(f"Best: {best_ad.distribution} (A-D={best_ad.ad_statistic:.4f})")

   # Filter by A-D threshold
   good_fits = results.filter(ad_threshold=1.0)

.. note::
   A-D p-values are only available for 5 distributions (norm, expon, logistic, gumbel_r,
   gumbel_l) where scipy has critical value tables. For other distributions, ``ad_pvalue``
   will be ``None`` but ``ad_statistic`` is still valid for ranking fits.

Excluding Distributions
-----------------------

By default, slow distributions are excluded. To customize:

.. code-block:: python

   from spark_bestfit import DistributionFitter, DEFAULT_EXCLUDED_DISTRIBUTIONS

   # View default exclusions
   print(DEFAULT_EXCLUDED_DISTRIBUTIONS)

   # Include a specific distribution by removing it from exclusions
   exclusions = tuple(d for d in DEFAULT_EXCLUDED_DISTRIBUTIONS if d != "wald")
   fitter = DistributionFitter(spark, excluded_distributions=exclusions)

   # Or exclude nothing (fit all distributions - may be slow)
   fitter = DistributionFitter(spark, excluded_distributions=())
