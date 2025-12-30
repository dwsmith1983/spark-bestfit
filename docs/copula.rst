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
     - **sample_spark()** distributed

Basic Usage
-----------

Fit a copula from multi-column fit results:

.. code-block:: python

    from spark_bestfit import DistributionFitter, GaussianCopula

    # Fit multiple columns
    fitter = DistributionFitter(spark)
    results = fitter.fit(df, columns=["price", "quantity", "revenue"])

    # Fit copula - correlation computed via Spark ML (scales to billions)
    copula = GaussianCopula.fit(results, df)

    # Generate correlated samples locally
    samples = copula.sample(n=10000)  # Dict[str, np.ndarray]

    # Or distributed via Spark
    samples_df = copula.sample_spark(n=100_000_000)

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
   * - ``sample_spark(n=N)``
     - Large samples (> 10M)
     - Spark DataFrame

For small samples, ``sample()`` is faster due to Spark overhead:

.. code-block:: python

    # Local sampling - fast for small n
    samples = copula.sample(n=10000, random_state=42)
    df = pd.DataFrame(samples)

    # Distributed sampling - efficient for large n
    samples_df = copula.sample_spark(n=100_000_000, random_seed=42)

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

.. code-block:: json

    {
      "schema_version": "1.0",
      "spark_bestfit_version": "1.3.0",
      "created_at": "2025-12-30T20:00:00Z",
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
   b. Transform each normal sample → uniform via Φ (standard normal CDF)
   c. Transform each uniform → target marginal via PPF (inverse CDF)

This ensures that:

- Each column follows its fitted marginal distribution
- Columns maintain the correlation structure from the original data

API Reference
-------------

See :class:`spark_bestfit.copula.GaussianCopula` for full API documentation.
