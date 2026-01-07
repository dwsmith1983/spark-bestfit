Pre-filtering
=============

spark-bestfit supports **smart pre-filtering** that skips distributions mathematically
incompatible with your data. This eliminates unnecessary fitting attempts based on
data characteristics like skewness and kurtosis.

Why Pre-filter?
---------------

Distribution fitting is expensive. Each scipy ``dist.fit()`` call involves numerical
optimization that takes 50-500ms depending on the distribution. With ~90 distributions (default),
this adds up to significant time - but many distributions have intrinsic shape properties
that make them poor fits for your data.

**Example:** If your data is clearly left-skewed (skewness < -1), distributions like
``expon``, ``gamma``, ``chi2``, ``lognorm`` (which are intrinsically right-skewed)
cannot possibly fit well regardless of how scipy shifts them via ``loc``. Pre-filtering
skips these before the expensive fitting step.

Filtering Layers
----------------

Pre-filtering uses a layered approach based on **intrinsic shape properties**
(not location/scale):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Layer
     - Reliability
     - Description
   * - Skewness sign
     - ~95%
     - Skip positive-skew-only distributions for left-skewed data
   * - Kurtosis
     - ~80%
     - Skip low-kurtosis distributions for heavy-tailed data (aggressive mode)

.. note::
   We do NOT filter by support bounds (``dist.a``/``dist.b``) because scipy's
   ``loc`` parameter can shift any distribution to cover any data range.
   For example, ``expon(loc=-100)`` has support ``[-100, inf)`` and can fit
   negative data. Skewness and kurtosis are intrinsic shape properties that
   cannot be changed by ``loc``/``scale``.

Using Pre-filtering
-------------------

**Using FitterConfig (v2.2+, recommended):**

.. code-block:: python

   from spark_bestfit import DistributionFitter, FitterConfigBuilder

   fitter = DistributionFitter(spark)

   # Safe mode (recommended) - skewness filtering
   config = FitterConfigBuilder().with_prefilter().build()
   results = fitter.fit(df, "value", config=config)

   # Aggressive mode - adds kurtosis filtering
   config = FitterConfigBuilder().with_prefilter(mode="aggressive").build()
   results = fitter.fit(df, "value", config=config)

**Using parameter directly:**

.. code-block:: python

   # Safe mode
   results = fitter.fit(df, "value", prefilter=True)

   # Aggressive mode
   results = fitter.fit(df, "value", prefilter="aggressive")

   # Disabled (default)
   results = fitter.fit(df, "value", prefilter=False)

Performance Impact
------------------

Pre-filtering effectiveness depends on your data's **shape characteristics**:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Data Characteristic
     - Distributions Filtered
     - Example
   * - Symmetric (skew ~ 0)
     - 0%
     - No shape-based filtering applies
   * - Strongly left-skewed (skew < -1)
     - 20-30%
     - Positive-skew-only distributions skipped
   * - Strongly right-skewed (skew > 1)
     - 0%
     - Right-skewed data fits most distributions
   * - Heavy-tailed (aggressive, kurtosis > 10)
     - Additional 5-10%
     - Low-kurtosis distributions like ``uniform`` skipped

**Typical savings:** 20-50% fewer distributions to fit for skewed data, translating to
proportional time savings during the fitting phase.

Fallback Behavior
-----------------

If pre-filtering removes all candidate distributions (which can happen with unusual
data), spark-bestfit automatically falls back to fitting all distributions and logs
a warning:

.. code-block:: text

   WARNING: Pre-filter removed all 90 distributions; falling back to fitting all distributions

This ensures you always get results, even if the pre-filter was too aggressive.

When to Use Pre-filtering
-------------------------

**Use prefilter=True when:**

- Your data is clearly skewed (skewness < -1 or > 1)
- You want faster fitting without sacrificing accuracy
- You're fitting many distributions and want to skip shape-incompatible ones

**Use prefilter="aggressive" when:**

- Your data is heavy-tailed (high kurtosis) and you want to skip light-tailed distributions
- You're comfortable with ~80% reliability on the kurtosis filter
- Speed is more important than fitting every possible distribution

**Use prefilter=False when:**

- Your data is approximately symmetric (skewness ~ 0)
- You want to fit all distributions regardless of theoretical compatibility
- You need complete control over which distributions are attempted

Combining with Lazy Metrics
---------------------------

Pre-filtering and lazy metrics are complementary optimizations:

**Using FitterConfig (v2.2+, recommended):**

.. code-block:: python

   from spark_bestfit import FitterConfigBuilder

   # Maximum performance: fewer distributions + deferred KS/AD
   config = (FitterConfigBuilder()
       .with_prefilter()      # Skip incompatible distributions
       .with_lazy_metrics()   # Defer KS/AD computation
       .build())

   results = fitter.fit(df, "value", config=config)

   # Fast model selection
   best = results.best(n=1, metric="aic")[0]

**Using parameters directly:**

.. code-block:: python

   results = fitter.fit(
       df, "value",
       prefilter=True,
       lazy_metrics=True,
   )

**Combined benefit:** Pre-filtering reduces the number of distributions to fit,
and lazy metrics defers expensive KS/AD computation. Together, they can reduce
total fitting time by 50-80% for typical workflows.
