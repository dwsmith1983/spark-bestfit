API Reference
=============

Core
----

.. automodule:: spark_bestfit.core
   :members:

.. autodata:: spark_bestfit.core.DEFAULT_EXCLUDED_DISTRIBUTIONS

.. autodata:: spark_bestfit.core.DEFAULT_EXCLUDED_DISCRETE_DISTRIBUTIONS

Discrete Fitting
----------------

.. automodule:: spark_bestfit.discrete_fitting
   :members:

Results
-------

.. autoclass:: spark_bestfit.results.DistributionFitResult
   :members: sample, sample_spark, pdf, cdf, ppf
   :show-inheritance:
   :no-index:

.. autoclass:: spark_bestfit.results.BaseFitResults
   :members:
   :show-inheritance:

.. autoclass:: spark_bestfit.results.EagerFitResults
   :members:
   :show-inheritance:

.. autoclass:: spark_bestfit.results.LazyFitResults
   :members:
   :show-inheritance:

.. autofunction:: spark_bestfit.results.create_fit_results

Sampling
--------

.. automodule:: spark_bestfit.sampling
   :members:

Distributions
-------------

.. automodule:: spark_bestfit.distributions
   :members:

Histogram
---------

.. automodule:: spark_bestfit.histogram
   :members:

Plotting
--------

.. automodule:: spark_bestfit.plotting
   :members:

Utilities
---------

.. automodule:: spark_bestfit.utils
   :members:
