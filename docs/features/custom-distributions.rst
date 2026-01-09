Custom Distributions
====================

spark-bestfit supports fitting custom scipy ``rv_continuous`` distributions
alongside the built-in ~90 scipy.stats distributions. This is useful when you
have domain-specific distributions or want to test theoretical models.

Basic Usage
-----------

Register a custom distribution with the fitter, then fit as usual:

.. code-block:: python

   from scipy.stats import rv_continuous
   from spark_bestfit import DistributionFitter, LocalBackend

   # Define a custom distribution
   class PowerDistribution(rv_continuous):
       """Power distribution: f(x) = alpha * x^(alpha-1) for x in [0, 1]"""

       def _pdf(self, x, alpha):
           return alpha * x ** (alpha - 1)

       def _cdf(self, x, alpha):
           return x ** alpha

   # Create fitter and register the custom distribution
   fitter = DistributionFitter(backend=LocalBackend())
   fitter.register_distribution("power", PowerDistribution(a=0, b=1))

   # Fit - your custom distribution is now included
   results = fitter.fit(df, column="value")

   # Check if your distribution won
   best = results.best(n=5)
   for r in best:
       print(f"{r.distribution}: SSE={r.sse:.6f}")

Method Chaining
---------------

Registration methods support chaining for convenience:

.. code-block:: python

   fitter = (
       DistributionFitter(backend=LocalBackend())
       .register_distribution("power", PowerDistribution(a=0, b=1))
       .register_distribution("custom_exp", MyExponential())
   )

Registry API
------------

You can also work with the registry directly:

.. code-block:: python

   from spark_bestfit import DistributionRegistry

   registry = DistributionRegistry()

   # Register
   registry.register_distribution("my_dist", MyDistribution())

   # Check what's registered
   print(registry.get_custom_distributions())  # {'my_dist': <MyDistribution>}
   print(registry.has_custom_distributions())  # True

   # Get distribution object by name (works for scipy and custom)
   dist = registry.get_distribution_object("my_dist")
   dist = registry.get_distribution_object("norm")  # Also works

   # Unregister
   registry.unregister_distribution("my_dist")

Requirements
------------

Custom distributions must:

1. **Subclass** ``scipy.stats.rv_continuous``
2. **Implement** ``_pdf(self, x, *args)`` method
3. **Implement** ``_cdf(self, x, *args)`` method
4. **Be picklable** (for Spark serialization)

The ``fit()`` method is inherited from ``rv_continuous`` and uses MLE by default.

.. code-block:: python

   from scipy.stats import rv_continuous

   class MyDistribution(rv_continuous):
       """Custom distribution with parameter 'alpha'."""

       def _pdf(self, x, alpha):
           # Return probability density at x
           return alpha * np.exp(-alpha * x)

       def _cdf(self, x, alpha):
           # Return cumulative distribution at x
           return 1 - np.exp(-alpha * x)

   # Set support bounds in constructor
   my_dist = MyDistribution(a=0, b=np.inf, name="my_dist")

Example: Beta-Like Distribution
-------------------------------

Here's a more complete example with a custom beta-like distribution:

.. code-block:: python

   import numpy as np
   from scipy.stats import rv_continuous
   from scipy.special import beta as beta_func

   class SimpleBeta(rv_continuous):
       """Simplified beta distribution for demonstration."""

       def _pdf(self, x, a, b):
           return x**(a-1) * (1-x)**(b-1) / beta_func(a, b)

       def _cdf(self, x, a, b):
           from scipy.special import betainc
           return betainc(a, b, x)

   # Create with support [0, 1]
   simple_beta = SimpleBeta(a=0, b=1, name="simple_beta")

   # Register and fit
   fitter = DistributionFitter(backend=LocalBackend())
   fitter.register_distribution("simple_beta", simple_beta)

   results = fitter.fit(df, "value")

Backend Support
---------------

Custom distributions work with all backends:

- **LocalBackend**: Full support
- **SparkBackend**: Full support (distributions are broadcast to executors)
- **RayBackend**: Signature compatible, but custom distributions are not yet
  passed to Ray tasks

.. note::

   For SparkBackend, ensure your custom distribution class is defined in a
   module that's available to all executors (not just in a notebook cell).
   For development, use ``pip install -e .`` to make your package available.

Error Handling
--------------

The registry validates distributions on registration:

.. code-block:: python

   # Name conflicts with scipy.stats
   fitter.register_distribution("norm", my_dist)
   # ValueError: Cannot register 'norm': conflicts with scipy.stats

   # Missing required methods
   fitter.register_distribution("bad", object())
   # TypeError: Distribution must implement ['fit', 'pdf', 'cdf']

   # Duplicate registration
   fitter.register_distribution("my_dist", dist1)
   fitter.register_distribution("my_dist", dist2)
   # ValueError: Distribution 'my_dist' already registered

   # Use overwrite=True to replace
   fitter.register_distribution("my_dist", dist2, overwrite=True)  # OK
