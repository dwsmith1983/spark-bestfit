FitterConfig Builder
====================

.. versionadded:: 2.2.0

spark-bestfit provides a **fluent builder pattern** for configuring distribution fitting.
The ``FitterConfig`` dataclass and ``FitterConfigBuilder`` offer a cleaner alternative
to passing many parameters to ``fit()``.

Why Use FitterConfig?
---------------------

The ``fit()`` method supports 15+ parameters for continuous distributions:

.. code-block:: python

   # Traditional approach - many parameters
   results = fitter.fit(
       df, column="value",
       bins=100,
       use_rice_rule=False,
       support_at_zero=True,
       max_distributions=50,
       prefilter=True,
       enable_sampling=True,
       sample_fraction=0.1,
       max_sample_size=500_000,
       sample_threshold=5_000_000,
       bounded=True,
       lower_bound=0.0,
       upper_bound=100.0,
       num_partitions=16,
       lazy_metrics=True,
   )

With ``FitterConfigBuilder``, this becomes:

.. code-block:: python

   from spark_bestfit import FitterConfigBuilder

   # Builder pattern - cleaner and reusable
   config = (FitterConfigBuilder()
       .with_bins(100, use_rice_rule=False)
       .with_support_at_zero()
       .with_max_distributions(50)
       .with_prefilter()
       .with_sampling(fraction=0.1, max_size=500_000, threshold=5_000_000)
       .with_bounds(lower=0.0, upper=100.0)
       .with_partitions(16)
       .with_lazy_metrics()
       .build())

   results = fitter.fit(df, column="value", config=config)

**Benefits:**

- **Cleaner code**: Grouped, readable configuration
- **Reusable**: Same config works across multiple fits
- **IDE-friendly**: Better autocomplete and discoverability
- **Immutable**: Frozen dataclass prevents accidental mutation
- **Backward compatible**: Individual parameters still work

Basic Usage
-----------

Create a configuration using the builder:

.. code-block:: python

   from spark_bestfit import DistributionFitter, FitterConfigBuilder, LocalBackend

   # Create a configuration
   config = (FitterConfigBuilder()
       .with_bins(100)
       .with_lazy_metrics()
       .build())

   # Use with fitter
   fitter = DistributionFitter(backend=LocalBackend())
   results = fitter.fit(df, column="value", config=config)

Or create ``FitterConfig`` directly:

.. code-block:: python

   from spark_bestfit import FitterConfig

   config = FitterConfig(
       bins=100,
       lazy_metrics=True,
   )

Builder Methods
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``with_bins(bins, use_rice_rule)``
     - Configure histogram binning (continuous only)
   * - ``with_bounds(lower, upper, auto_detect)``
     - Enable bounded/truncated distribution fitting
   * - ``with_sampling(fraction, max_size, threshold, enabled)``
     - Configure data sampling for large datasets
   * - ``with_lazy_metrics(lazy)``
     - Defer KS/AD computation until accessed
   * - ``with_prefilter(mode)``
     - Pre-filter incompatible distributions
   * - ``with_support_at_zero(enabled)``
     - Only fit non-negative distributions
   * - ``with_max_distributions(n)``
     - Limit number of distributions to fit
   * - ``with_partitions(n)``
     - Set parallel partition count
   * - ``build()``
     - Create immutable ``FitterConfig``

Config Attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Default
     - Description
   * - ``bins``
     - 50
     - Number of histogram bins or tuple of bin edges
   * - ``use_rice_rule``
     - True
     - Auto-determine bin count using Rice rule
   * - ``support_at_zero``
     - False
     - Only fit non-negative distributions
   * - ``max_distributions``
     - None
     - Limit distributions to fit (None = all)
   * - ``prefilter``
     - False
     - Pre-filter incompatible distributions
   * - ``enable_sampling``
     - True
     - Enable sampling for large datasets
   * - ``sample_fraction``
     - None
     - Explicit sample fraction (None = auto)
   * - ``max_sample_size``
     - 1,000,000
     - Max rows when auto-determining sample
   * - ``sample_threshold``
     - 10,000,000
     - Row count above which sampling applies
   * - ``bounded``
     - False
     - Enable truncated distribution fitting
   * - ``lower_bound``
     - None
     - Lower bound (scalar or per-column dict)
   * - ``upper_bound``
     - None
     - Upper bound (scalar or per-column dict)
   * - ``num_partitions``
     - None
     - Parallel partitions (None = auto)
   * - ``lazy_metrics``
     - False
     - Defer KS/AD computation

Reusing Configurations
----------------------

A key benefit of ``FitterConfig`` is reusability across multiple fits:

.. code-block:: python

   from spark_bestfit import DistributionFitter, FitterConfigBuilder

   # Create config once
   config = (FitterConfigBuilder()
       .with_bins(100)
       .with_bounds(lower=0)
       .with_lazy_metrics()
       .build())

   fitter = DistributionFitter(spark)

   # Reuse for multiple columns
   for col in ["price", "quantity", "revenue"]:
       results = fitter.fit(df, column=col, config=config)
       best = results.best(n=1, metric="aic")[0]
       print(f"{col}: {best.distribution}")

   # Reuse for different DataFrames
   for df in [df_train, df_test, df_validation]:
       results = fitter.fit(df, column="value", config=config)

Per-Column Bounds
-----------------

For multi-column fitting with different bounds per column:

.. code-block:: python

   config = (FitterConfigBuilder()
       .with_bounds(
           lower={"price": 0.0, "temperature": -40.0},
           upper={"price": 10000.0, "temperature": 50.0}
       )
       .build())

   results = fitter.fit(df, columns=["price", "temperature"], config=config)

Progress Callback Override
--------------------------

The ``progress_callback`` parameter can be passed directly to ``fit()`` even when
using a config. This allows different callbacks for different fits while reusing
the same config:

.. code-block:: python

   from spark_bestfit import console_progress

   config = (FitterConfigBuilder()
       .with_lazy_metrics()
       .build())

   # Different callback per fit
   results1 = fitter.fit(df, column="col1", config=config, progress_callback=console_progress)
   results2 = fitter.fit(df, column="col2", config=config)  # No callback

Or set the callback on the config itself:

.. code-block:: python

   config = (FitterConfigBuilder()
       .with_lazy_metrics()
       .build()
       .with_progress_callback(console_progress))

Config for Discrete Distributions
---------------------------------

``FitterConfig`` works with both continuous and discrete fitters. Continuous-only
attributes (like ``bins``, ``use_rice_rule``, ``support_at_zero``) are simply
ignored by ``DiscreteDistributionFitter``:

.. code-block:: python

   from spark_bestfit import DiscreteDistributionFitter, FitterConfigBuilder

   # Same config works for both fitters
   config = (FitterConfigBuilder()
       .with_bounds(lower=0, upper=100)
       .with_lazy_metrics()
       .build())

   # Continuous fitter
   continuous_fitter = DistributionFitter(spark)
   continuous_results = continuous_fitter.fit(df, column="value", config=config)

   # Discrete fitter (bins/support_at_zero ignored)
   discrete_fitter = DiscreteDistributionFitter(spark)
   discrete_results = discrete_fitter.fit(df, column="counts", config=config)

Backward Compatibility
----------------------

Individual parameters continue to work as before. When both ``config`` and
individual parameters are provided, **config takes precedence**:

.. code-block:: python

   config = FitterConfigBuilder().with_max_distributions(5).build()

   # Config wins: max_distributions=5 is used, not 10
   results = fitter.fit(
       df, column="value",
       config=config,
       max_distributions=10,  # Ignored when config is provided
   )

Exception: ``progress_callback`` always overrides the config's callback when
passed directly to ``fit()``.

Immutability
------------

``FitterConfig`` is a frozen dataclass. Attempting to modify it raises an error:

.. code-block:: python

   config = FitterConfig(bins=100)
   config.bins = 200  # Raises FrozenInstanceError!

To create a modified config, use ``dataclasses.replace()``:

.. code-block:: python

   from dataclasses import replace

   config = FitterConfig(bins=100, lazy_metrics=True)
   modified = replace(config, bins=200)  # New config with bins=200

Or use the ``with_progress_callback()`` convenience method:

.. code-block:: python

   config = FitterConfig(lazy_metrics=True)
   with_callback = config.with_progress_callback(my_callback)
