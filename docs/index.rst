spark-bestfit
==============

Modern Spark distribution fitting library with efficient parallel processing.

Automatically fit ~100 scipy.stats continuous distributions and 16 discrete distributions
to your data using Apache Spark's distributed computing power, with optimized Pandas UDFs.

**Supported Versions:**

- Python 3.11 - 3.13
- Apache Spark 3.5.x and 4.x
- See :doc:`quickstart` for the full compatibility matrix

Scope & Limitations
-------------------

spark-bestfit is designed for **batch processing** of statistical distribution fitting.

**What it does well:**

- Fit ~100 continuous and 16 discrete scipy.stats distributions in parallel
- Multi-column fitting: fit multiple columns efficiently in a single operation
- Provide robust goodness-of-fit metrics (KS, A-D, AIC, BIC, SSE)
- Generate publication-ready visualizations (histograms, Q-Q plots, P-P plots)
- Compute bootstrap confidence intervals for parameters

**Known limitations:**

- No real-time/streaming support (batch processing only)
- Custom distribution support planned for 1.3.0

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   progress
   performance
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
