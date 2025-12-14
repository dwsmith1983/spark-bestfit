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
   * - **4.0.x**
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

   # Get best fit
   best = results.best(n=1)[0]
   print(f"Best: {best.distribution} with SSE={best.sse:.6f}")

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
   )

Working with Results
--------------------

.. code-block:: python

   # Get top 5 distributions by SSE
   top_5 = results.best(n=5, metric="sse")

   # Get best by AIC
   best_aic = results.best(n=1, metric="aic")[0]

   # Filter good fits
   good_fits = results.filter(sse_threshold=0.01)

   # Convert to pandas for analysis
   df_pandas = results.to_pandas()

   # Use fitted distribution
   samples = best.sample(size=10000)  # Generate samples
   pdf_values = best.pdf(x_array)     # Evaluate PDF
   cdf_values = best.cdf(x_array)     # Evaluate CDF

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
