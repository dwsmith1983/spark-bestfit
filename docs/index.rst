spark-bestfit
==============

Modern distribution fitting library with pluggable backends (Spark, Ray, Local).

Automatically fit ~90 scipy.stats continuous distributions and 16 discrete distributions
to your data using parallel processing. Supports Apache Spark for production clusters,
Ray for ML workflows, or local execution for development.

**Supported Versions:**

- Python 3.11 - 3.13
- Apache Spark 3.5.x and 4.x
- Ray 2.x (optional)
- See :doc:`quickstart` for the full compatibility matrix

Scope & Limitations
-------------------

spark-bestfit is designed for **batch processing** of statistical distribution fitting.

**What it does well:**

- Fit ~90 continuous and 16 discrete scipy.stats distributions in parallel
- Multi-column fitting: fit multiple columns efficiently in a single operation
- Provide robust goodness-of-fit metrics (KS, A-D, AIC, BIC, SSE)
- Generate publication-ready visualizations (histograms, Q-Q plots, P-P plots)
- Compute bootstrap confidence intervals for parameters
- Scale to 100M+ rows with Spark or Ray backends

**Known limitations:**

- No real-time/streaming support (batch processing only)
- User-defined distributions planned for v2.1.0
- Parameters and metrics use 32-bit floats (~7 significant digits) for Spark serialization
  efficiency. Very small values (e.g., p-values < 1e-7) may lose precision.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   backends
   usecases

.. toctree::
   :maxdepth: 2
   :caption: Features

   features/bounded
   features/sampling
   features/serialization
   features/copula
   features/progress
   features/lazy-metrics
   features/prefiltering

.. toctree::
   :maxdepth: 2
   :caption: Reference

   performance
   migration
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
