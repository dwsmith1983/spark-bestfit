Ray Backend
===========

The ``RayBackend`` enables distributed distribution fitting on Ray clusters without
requiring Apache Spark. It's ideal for ML pipelines, Ray-based workflows, and
environments where Spark is unavailable.

Installation
------------

Install with Ray support:

.. code-block:: bash

   pip install spark-bestfit[ray]

Quick Start
-----------

.. code-block:: python

   from spark_bestfit import DistributionFitter, RayBackend
   import pandas as pd
   import numpy as np

   # Auto-initializes Ray if not already running
   backend = RayBackend()
   fitter = DistributionFitter(backend=backend)

   # Works with pandas DataFrames
   df = pd.DataFrame({"value": np.random.normal(50, 10, 10000)})
   results = fitter.fit(df, column="value")

   best = results.best(n=1)[0]
   print(f"Best: {best.distribution} (KS={best.ks_statistic:.4f})")

Ray Dataset Support
-------------------

For large-scale data, use Ray Datasets for distributed operations:

.. code-block:: python

   import ray

   # Create Ray Dataset from pandas
   ds = ray.data.from_pandas(df)

   # Distributed histogram computation and fitting
   results = fitter.fit(ds, column="value")

When using Ray Datasets:

- Histograms are computed distributedly via ``map_batches()``
- Correlations use vectorized sufficient statistics aggregation
- No raw data is collected to the driver node
- Automatically handles partitioning and parallelism

Connecting to Clusters
----------------------

.. code-block:: python

   # Auto-detect existing Ray cluster
   backend = RayBackend(address="auto")

   # Connect to specific Ray cluster
   backend = RayBackend(address="ray://cluster-head:10001")

   # Limit local resources
   backend = RayBackend(num_cpus=8)

Progress Tracking
-----------------

RayBackend supports progress callbacks with per-distribution granularity:

.. code-block:: python

   from spark_bestfit.progress import console_progress

   results = fitter.fit(
       df,
       column="value",
       progress_callback=console_progress()
   )
   # Output: Progress: 45/90 distributions (50.0%)

Unlike SparkBackend (which reports per-task via StatusTracker polling),
RayBackend reports progress after each distribution completes, providing
more accurate real-time feedback.

GaussianCopula Integration
--------------------------

RayBackend works with GaussianCopula for correlated multi-column sampling:

.. code-block:: python

   from spark_bestfit import GaussianCopula

   # Fit multiple columns
   results = fitter.fit(df, columns=["price", "quantity", "revenue"])

   # Fit copula with Ray-computed correlations
   copula = GaussianCopula.fit(results, df)

   # Generate correlated samples
   samples = copula.sample(n=100_000)

Performance Considerations
--------------------------

**When to use RayBackend:**

- Data already in Ray ecosystem (Ray Datasets, Ray Train, etc.)
- ML pipelines using Ray Tune, Ray Serve
- Kubernetes environments with Ray clusters
- When Spark is unavailable or impractical

**When to use SparkBackend:**

- Data already in Spark DataFrames
- Very large datasets (100M+ rows)
- Existing Spark infrastructure

**Performance characteristics:**

- Startup: RayBackend has lower startup overhead than SparkBackend
- Parallelism: Both scale with available CPUs
- Memory: Ray uses plasma object store; Spark uses JVM heap + off-heap
- Fitting: Both use identical scipy fitting algorithms

Example Notebooks
-----------------

See the ``examples/ray/`` directory for complete working examples:

- ``ray_demo.ipynb`` - Basic RayBackend usage and features
- ``usecase_synthetic_data.ipynb`` - Synthetic data generation with RayBackend

API Reference
-------------

.. autoclass:: spark_bestfit.backends.ray.RayBackend
   :members:
   :undoc-members:
   :show-inheritance:
