Multivariate Normal
===================

.. versionadded:: 2.9.0

The ``MultivariateNormalFitter`` class enables direct multivariate normal distribution
fitting as an alternative to copula-based correlation modeling. Use this when the joint
distribution is assumed to be multivariate normal and you need interpretable joint
parameters (mean vector, covariance matrix).

When to Use Multivariate Normal
-------------------------------

This is an **alternative** to copulas, not a replacement:

.. list-table::
   :header-rows: 1

   * - Scenario
     - GaussianCopula
     - MultivariateNormalFitter
   * - Arbitrary marginals
     - **Recommended**
     - Not suitable
   * - Joint MVN assumption
     - Works (via correlations)
     - **Preferred** (direct fit)
   * - Interpretable parameters
     - Marginals + correlation
     - **Mean + covariance**
   * - Statistical testing (MVN)
     - Not applicable
     - **Preferred**
   * - Many columns (20+)
     - **Scales better**
     - May be unstable

Use copulas when you want flexible marginal distributions. Use multivariate normal when
your data is truly jointly normal and you need the mean vector and covariance matrix.

Basic Usage
-----------

Fit a multivariate normal distribution to multi-column data:

.. code-block:: python

    from spark_bestfit import MultivariateNormalFitter, LocalBackend
    import pandas as pd

    # Sample data
    df = pd.DataFrame({
        "x": np.random.normal(10, 2, 1000),
        "y": np.random.normal(20, 3, 1000),
        "z": np.random.normal(30, 4, 1000),
    })

    # Fit multivariate normal
    fitter = MultivariateNormalFitter(backend=LocalBackend())
    result = fitter.fit(df, columns=["x", "y", "z"])

    # Access fitted parameters
    print(result.mean)  # [10.02, 19.97, 30.05] - mean vector
    print(result.cov)   # 3x3 covariance matrix

    # Generate correlated samples
    samples = result.sample(n=10000)

With Spark
----------

The fitter works with Spark DataFrames:

.. code-block:: python

    from spark_bestfit import MultivariateNormalFitter
    from spark_bestfit.backends import BackendFactory

    # Fit from Spark DataFrame
    fitter = MultivariateNormalFitter()  # Auto-detects backend
    result = fitter.fit(spark_df, columns=["price", "quantity", "revenue"])

    # Distributed sampling
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = result.sample_distributed(n=100_000_000, backend=backend)

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

.. code-block:: python

    # Local sampling - fast for small n
    samples = result.sample(n=10000, random_state=42)
    df = pd.DataFrame(samples)

    # Distributed sampling - efficient for large n
    backend = BackendFactory.create("spark", spark_session=spark)
    samples_df = result.sample_distributed(n=100_000_000, backend=backend, random_seed=42)

Result Methods
--------------

The ``MultivariateNormalResult`` provides several useful methods:

**PDF and Log-PDF:**

.. code-block:: python

    # Evaluate probability density
    point = result.mean
    density = result.pdf(point)  # PDF at mean (maximum)

    # Log-PDF (more numerically stable)
    log_density = result.logpdf(point)

    # Batch evaluation
    points = np.array([[10, 20, 30], [11, 21, 31]])
    densities = result.pdf(points)

**Mahalanobis Distance:**

The Mahalanobis distance accounts for the covariance structure and is useful for
outlier detection:

.. code-block:: python

    # Distance from distribution center
    distances = result.mahalanobis(test_data)

    # Outlier detection (~99.7% threshold for MVN)
    outliers = distances > 3.0

**Correlation Matrix:**

.. code-block:: python

    # Get correlation matrix from covariance
    corr = result.correlation_matrix()
    # Diagonal is 1.0, off-diagonal are correlation coefficients

Bias Parameter
--------------

The ``bias`` parameter controls the covariance estimator:

.. code-block:: python

    # Unbiased estimate (default) - ddof=1
    result_unbiased = fitter.fit(df, columns=cols, bias=False)

    # Biased (MLE) estimate - ddof=0
    result_biased = fitter.fit(df, columns=cols, bias=True)

For large samples, the difference is negligible. Use ``bias=True`` for maximum
likelihood estimation consistency.

Numerical Stability
-------------------

The fitter warns if the covariance matrix has a high condition number (> 1e10),
which indicates near-collinear columns:

.. code-block:: python

    # Near-collinear data
    df["z_copy"] = df["z"] + np.random.normal(0, 0.001, len(df))

    # This will emit a warning
    result = fitter.fit(df, columns=["x", "y", "z", "z_copy"])
    # Warning: Covariance matrix has high condition number (1.23e+12)...

Consider removing highly correlated columns if you see this warning.

Serialization
-------------

Save and load results for later use:

.. code-block:: python

    # Save to JSON (recommended)
    result.save("mvn_model.json")

    # Or pickle for faster serialization
    result.save("mvn_model.pkl")

    # Load later
    loaded = MultivariateNormalResult.load("mvn_model.json")
    samples = loaded.sample(n=1000)

The JSON format includes metadata:

.. code-block:: javascript

    {
      "schema_version": "1.0",
      "spark_bestfit_version": "2.9.0",
      "created_at": "2026-01-10T22:00:00Z",
      "type": "multivariate_normal",
      "column_names": ["x", "y", "z"],
      "mean": [10.0, 20.0, 30.0],
      "cov": [[4.0, 2.0, 1.0], [2.0, 9.0, 3.0], [1.0, 3.0, 16.0]],
      "n_samples": 5000
    }

Comparison with Copula
----------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - GaussianCopula
     - MultivariateNormalFitter
   * - Marginal distributions
     - Any (100+ supported)
     - Gaussian only
   * - Correlation type
     - Spearman (rank)
     - Pearson (linear)
   * - Parameters
     - Per-column + correlation
     - Joint mean + covariance
   * - Sampling process
     - Transform through marginals
     - Direct MVN sampling
   * - PDF evaluation
     - Not available
     - Available
   * - Outlier detection
     - Not built-in
     - Mahalanobis distance

API Reference
-------------

See :class:`spark_bestfit.multivariate.MultivariateNormalFitter` and
:class:`spark_bestfit.multivariate.MultivariateNormalResult` for full API documentation.
