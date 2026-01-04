Bounded Distribution Fitting
============================

spark-bestfit supports fitting distributions with explicit bounds.
This is useful for data that has natural constraints like percentages (0-100),
ages (0+), prices (0+), or any domain-specific limits.

Basic Usage
-----------

Use the ``bounded`` parameter to enable automatic bound detection from your data:

.. code-block:: python

   from spark_bestfit import DistributionFitter

   fitter = DistributionFitter(spark)

   # Auto-detect bounds from data min/max
   results = fitter.fit(df, column="percentage", bounded=True)

   # Get best fit - samples will respect the bounds
   best = results.best(n=1)[0]
   samples = best.sample(1000)  # All samples within [data_min, data_max]

Explicit Bounds
---------------

For precise control, specify bounds explicitly:

.. code-block:: python

   # Both bounds explicit
   results = fitter.fit(
       df,
       column="percentage",
       bounded=True,
       lower_bound=0.0,
       upper_bound=100.0,
   )

   # Only lower bound (e.g., prices must be non-negative)
   results = fitter.fit(
       df,
       column="price",
       bounded=True,
       lower_bound=0.0,
   )

   # Only upper bound
   results = fitter.fit(
       df,
       column="score",
       bounded=True,
       upper_bound=1.0,
   )

.. note::
   When only one bound is specified and ``bounded=True``, the other bound is
   auto-detected from the data. Use ``-inf`` or ``inf`` to explicitly disable
   a bound while keeping the other explicit.

How It Works
------------

Bounded fitting uses a two-step process:

1. **Fit the unbounded distribution**: Standard MLE fitting is performed on the data
   to estimate distribution parameters.

2. **Truncate the distribution**: The fitted distribution is truncated to the specified
   bounds using CDF inversion. This ensures:

   - PDF integrates to 1 over the bounded domain
   - Samples are always within bounds
   - All statistical methods (pdf, cdf, ppf, sample) respect bounds

The truncation uses the formula:

.. code-block:: text

   ppf_truncated(u) = ppf_original(cdf_lb + u * (cdf_ub - cdf_lb))

   where:
     cdf_lb = CDF at lower bound
     cdf_ub = CDF at upper bound
     u ~ Uniform(0, 1)

Working with Bounded Results
----------------------------

The ``DistributionFitResult`` object tracks bounds and applies them automatically:

.. code-block:: python

   best = results.best(n=1)[0]

   # Check bounds
   print(f"Lower bound: {best.lower_bound}")  # e.g., 0.0
   print(f"Upper bound: {best.upper_bound}")  # e.g., 100.0

   # All methods respect bounds automatically
   samples = best.sample(1000)           # Samples within bounds
   pdf_vals = best.pdf(x_values)         # Normalized PDF
   cdf_vals = best.cdf(x_values)         # CDF: 0 below lb, 1 above ub
   quantiles = best.ppf([0.25, 0.5, 0.75])  # Quantiles within bounds

   # Get scipy distribution (already truncated)
   dist = best.get_scipy_dist()
   dist.rvs(size=100)  # Also respects bounds

Serialization
-------------

Bounds are preserved when saving and loading results:

.. code-block:: python

   # Save best result with bounds
   best = results.best(n=1)[0]
   best.save("model.json")

   # Load - bounds are restored
   from spark_bestfit.results import DistributionFitResult
   loaded = DistributionFitResult.load("model.json")
   print(loaded.lower_bound, loaded.upper_bound)  # Bounds preserved

Multi-Column Bounded Fitting
----------------------------

You can specify **different bounds per column** using dictionaries:

.. code-block:: python

   # Different bounds for each column
   results = fitter.fit(
       df,
       columns=["percentage", "price", "age"],
       bounded=True,
       lower_bound={"percentage": 0.0, "price": 0.0, "age": 0.0},
       upper_bound={"percentage": 100.0, "price": 10000.0, "age": 120.0},
   )

   # Each column has its own bounds
   pct_result = results.for_column("percentage").best(n=1)[0]
   print(pct_result.lower_bound, pct_result.upper_bound)  # 0.0, 100.0

   price_result = results.for_column("price").best(n=1)[0]
   print(price_result.lower_bound, price_result.upper_bound)  # 0.0, 10000.0

**Partial dictionaries** are supported - unspecified columns auto-detect from data:

.. code-block:: python

   # Only specify bounds for some columns
   results = fitter.fit(
       df,
       columns=["col_a", "col_b", "col_c"],
       bounded=True,
       lower_bound={"col_a": 0.0},  # Only col_a has explicit lower bound
       upper_bound={"col_b": 100.0},  # Only col_b has explicit upper bound
   )
   # col_c auto-detects both bounds from data

**Scalar bounds** apply to all columns (backward compatible):

.. code-block:: python

   # Same bounds for all columns
   results = fitter.fit(
       df,
       columns=["col_a", "col_b", "col_c"],
       bounded=True,
       lower_bound=0.0,   # Applied to all columns
       upper_bound=1.0,   # Applied to all columns
   )

Use Cases
---------

**Percentages and Proportions (0-100 or 0-1)**

.. code-block:: python

   results = fitter.fit(
       df, column="conversion_rate",
       bounded=True, lower_bound=0.0, upper_bound=1.0
   )

**Non-Negative Values (prices, counts, durations)**

.. code-block:: python

   results = fitter.fit(
       df, column="price",
       bounded=True, lower_bound=0.0
   )

**Age Data**

.. code-block:: python

   results = fitter.fit(
       df, column="age",
       bounded=True, lower_bound=0.0, upper_bound=120.0
   )

**Score Ranges**

.. code-block:: python

   results = fitter.fit(
       df, column="credit_score",
       bounded=True, lower_bound=300.0, upper_bound=850.0
   )

Discrete Distributions
----------------------

Bounded fitting is also supported for discrete distributions:

.. code-block:: python

   from spark_bestfit import DiscreteDistributionFitter

   # Auto-detect bounds
   fitter = DiscreteDistributionFitter(spark)
   results = fitter.fit(df, column="count", bounded=True)

   # Explicit bounds
   results = fitter.fit(
       df,
       column="count",
       bounded=True,
       lower_bound=0,
       upper_bound=100,
   )

   best = results.best(n=1, metric="aic")[0]
   print(best.lower_bound, best.upper_bound)

.. note::
   For discrete distributions, bounds are stored with the fit result but sampling
   uses the underlying scipy distribution. The bounds serve as metadata for the
   valid range of the fitted distribution.

Performance Considerations
--------------------------

Bounded fitting adds minimal overhead:

- Fitting time is unchanged (bounds are applied post-fit)
- Sampling is ~10% slower due to CDF inversion transform
- PDF/CDF/PPF evaluation has negligible overhead

For very large sample generation, the overhead of truncation is small compared
to the random number generation itself.
