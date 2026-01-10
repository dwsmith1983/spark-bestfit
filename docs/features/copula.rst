Gaussian Copula
===============

The ``GaussianCopula`` class enables correlated multi-column sampling at scale.
Unlike standard copula libraries that require loading data into memory, spark-bestfit
computes correlation via Spark ML and generates samples across the cluster.

Why Use a Copula?
-----------------

When you fit distributions to multiple columns independently, the correlation
structure between columns is lost:

.. code-block:: python

    # Independent fitting loses correlation
    results = fitter.fit(df, columns=["price", "quantity", "revenue"])

    # Sampling each column independently - correlation is LOST
    price_samples = results.for_column("price").best(n=1)[0].sample(1000)
    quantity_samples = results.for_column("quantity").best(n=1)[0].sample(1000)
    # These are uncorrelated! Not realistic.

A Gaussian copula preserves both:

- **Marginal distributions**: Each column follows its fitted distribution
- **Correlation structure**: Columns maintain their original relationships

When to Use spark-bestfit Copula
---------------------------------

spark-bestfit is **not faster** than statsmodels for small data. The value is **scale**:

.. list-table::
   :header-rows: 1

   * - Scenario
     - statsmodels
     - spark-bestfit
   * - Data < 10M rows
     - Faster (use this)
     - Slower (Spark overhead)
   * - Data > 100M rows
     - Crashes (OOM)
     - **Works** (distributed)
   * - Data already in Spark
     - Requires ``.toPandas()``
     - **Native** (no conversion)
   * - 100M+ samples needed
     - May OOM
     - **sample_distributed()** scales

Basic Usage
-----------

Fit a copula from multi-column fit results:

.. code-block:: python

    from spark_bestfit import DistributionFitter, GaussianCopula
    from spark_bestfit.backends import BackendFactory

    # Fit multiple columns
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, columns=["price", "quantity", "revenue"])

    # Fit copula - correlation computed via Spark ML (scales to billions)
    copula = GaussianCopula.fit(results, df)

    # Generate correlated samples locally
    samples = copula.sample(n=10000)  # Dict[str, np.ndarray]

    # Or distributed via any backend
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = copula.sample_distributed(n=100_000_000, backend=backend)

The ``df`` parameter is required to compute the correlation matrix. The copula
uses Spearman rank correlation, which is robust to non-linear relationships.

Local vs Distributed Sampling
-----------------------------

.. list-table::
   :header-rows: 1

   * - Method
     - Use Case
     - Output
   * - ``sample(n=N)``
     - Small to medium samples (< 10M)
     - Dict[str, np.ndarray]
   * - ``sample_distributed(n=N, backend=...)``
     - Large samples (> 10M)
     - DataFrame (Spark/pandas)

For small samples, ``sample()`` is faster due to distributed overhead:

.. code-block:: python

    # Local sampling - fast for small n
    samples = copula.sample(n=10000, random_state=42)
    df = pd.DataFrame(samples)

    # Distributed sampling - efficient for large n
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = copula.sample_distributed(n=100_000_000, backend=backend, random_seed=42)

Backend Options
---------------

Use any backend for distributed copula sampling:

.. code-block:: python

    from spark_bestfit.backends import BackendFactory

    # Spark
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = copula.sample_distributed(n=100_000_000, backend=backend)

    # Ray
    backend = BackendFactory.create("ray")
    samples_df = copula.sample_distributed(n=100_000_000, backend=backend)

    # Local (for testing)
    backend = BackendFactory.create("local", max_workers=4)
    samples_df = copula.sample_distributed(n=100_000, backend=backend)

Fast Uniform Sampling
---------------------

Both ``sample()`` and ``sample_distributed()`` support a ``return_uniform=True`` parameter
that skips the marginal distribution transforms, returning uniform [0,1] samples instead.
This matches the behavior of statsmodels and is significantly faster:

.. code-block:: python

    # Fast path - returns uniform samples without marginal transforms
    uniform_samples = copula.sample(n=10_000_000, return_uniform=True)

    # Full transform - slower but returns samples in fitted marginal distributions
    marginal_samples = copula.sample(n=10_000_000)

**When to use ``return_uniform=True``:**

- You only need the correlation structure, not the exact marginal distributions
- You're doing correlation analysis or downstream transforms
- Performance is critical

Performance Benchmarks
----------------------

Sampling performance comparison (3-column copula, local mode):

.. list-table::
   :header-rows: 1

   * - N Samples
     - statsmodels
     - return_uniform
     - with transform
   * - 1,000,000
     - 73 ms
     - **56 ms**
     - 1,547 ms
   * - 10,000,000
     - 725 ms
     - **555 ms**
     - 15,485 ms
   * - 50,000,000
     - 3,706 ms
     - **2,820 ms**
     - 77,820 ms

**Key findings:**

- ``return_uniform=True`` is ~24% faster than statsmodels (same output format)
- Full marginal transforms add ~28x overhead due to scipy's PPF using iterative root-finding
- Use ``return_uniform=True`` when you don't need the exact marginal distributions

Fast PPF Optimization
---------------------

.. versionadded:: 2.7.0

For common distributions, spark-bestfit bypasses scipy's generic PPF machinery (which uses
iterative root-finding) by calling scipy.special functions directly. This optimization is
applied automatically during copula sampling.

**Supported distributions with fast PPF:**

- ``norm`` - Normal/Gaussian
- ``expon`` - Exponential
- ``uniform`` - Uniform
- ``lognorm`` - Log-normal
- ``weibull_min`` - Weibull (minimum)
- ``gamma`` - Gamma
- ``beta`` - Beta

For these distributions, marginal transforms are **~10-20x faster** than the generic scipy path.
Other distributions automatically fall back to scipy.stats.

**Usage:**

No code changes required - the optimization is applied automatically in both ``sample()``
and ``sample_distributed()``:

.. code-block:: python

    # Fast PPF is used automatically for supported distributions
    samples = copula.sample(n=1_000_000)  # Uses fast_ppf for norm, gamma, etc.

**Direct access (advanced):**

If you need to use the fast PPF implementation directly:

.. code-block:: python

    from spark_bestfit.fast_ppf import fast_ppf, has_fast_ppf
    import numpy as np

    # Check if a distribution has fast PPF support
    has_fast_ppf("gamma")  # True
    has_fast_ppf("pareto")  # False

    # Compute PPF directly
    q = np.array([0.1, 0.5, 0.9])
    values = fast_ppf("gamma", (2.0, 0.0, 1.0), q)  # shape=2, loc=0, scale=1

Serialization
-------------

Save and load copulas for later use:

.. code-block:: python

    # Save to JSON (recommended)
    copula.save("copula.json")

    # Or pickle for faster serialization
    copula.save("copula.pkl")

    # Load later
    loaded = GaussianCopula.load("copula.json")
    samples = loaded.sample(n=1000)

The JSON format includes metadata for debugging:

.. code-block:: javascript

    {
      "schema_version": "1.0",
      "spark_bestfit_version": "2.6.0",
      "created_at": "2026-01-04T20:00:00Z",
      "type": "gaussian_copula",
      "column_names": ["price", "quantity", "revenue"],
      "correlation_matrix": [[1.0, 0.8, 0.9], ...],
      "marginals": {
        "price": {"distribution": "gamma", "parameters": [2.0, 0.0, 5.0]},
        ...
      }
    }

How It Works
------------

The Gaussian copula sampling process:

1. **Fit phase**: Compute Spearman correlation matrix via Spark ML (no ``.toPandas()``)
2. **Sample phase**:

   a. Generate multivariate normal samples with the correlation matrix
   b. Transform each normal sample -> uniform via phi (standard normal CDF)
   c. Transform each uniform -> target marginal via PPF (inverse CDF)

This ensures that:

- Each column follows its fitted marginal distribution
- Columns maintain the correlation structure from the original data

Deprecated: sample_spark()
--------------------------

.. deprecated:: 2.0.0
   The ``sample_spark()`` method is deprecated and will be removed in v3.0.0.
   Use ``sample_distributed()`` with an explicit backend instead.

Migration example:

.. code-block:: python

    # Old (deprecated)
    samples_df = copula.sample_spark(n=100_000_000)

    # New (recommended)
    from spark_bestfit.backends import BackendFactory

    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = copula.sample_distributed(n=100_000_000, backend=backend)

API Reference
-------------

See :class:`spark_bestfit.copula.GaussianCopula` for full API documentation.
