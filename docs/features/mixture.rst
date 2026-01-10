Gaussian Mixture Models
=======================

.. versionadded:: 2.9.0

The ``GaussianMixtureFitter`` class fits Gaussian Mixture Models (GMM) to data using
the Expectation-Maximization (EM) algorithm. Use this when your data appears to come
from multiple populations or a single distribution doesn't adequately describe it.

When to Use Mixture Models
--------------------------

Mixture models are appropriate when:

- Data shows multiple modes (peaks in the histogram)
- A single distribution doesn't fit well across all regions
- You want to decompose data into component distributions
- You need soft clustering (probabilistic assignments)

.. list-table::
   :header-rows: 1

   * - Scenario
     - DistributionFitter
     - GaussianMixtureFitter
   * - Unimodal data
     - **Recommended**
     - Overkill
   * - Multi-modal data
     - Poor fit
     - **Recommended**
   * - Clustering
     - Not applicable
     - **Soft clustering**
   * - Model selection
     - 100+ distributions
     - Component count (BIC)

Basic Usage
-----------

Fit a Gaussian mixture model to univariate or multivariate data:

.. code-block:: python

    from spark_bestfit import GaussianMixtureFitter
    import numpy as np

    # Generate bimodal data
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(5, 1, 1000)
    data = np.concatenate([data1, data2])

    # Fit 2-component mixture
    fitter = GaussianMixtureFitter(n_components=2, random_state=42)
    result = fitter.fit(data)

    # Access fitted parameters
    print(result.weights_)      # [0.5, 0.5] - mixing weights
    print(result.means_)        # [[0], [5]] - component means
    print(result.covariances_)  # [[[1]], [[1]]] - component covariances

    # Generate samples from the mixture
    samples = result.sample(n=10000)

Multivariate Data
-----------------

The fitter handles multivariate data seamlessly:

.. code-block:: python

    # 2D bimodal data
    data1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 500)
    data2 = np.random.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 1]], 500)
    data = np.vstack([data1, data2])

    fitter = GaussianMixtureFitter(n_components=2, random_state=42)
    result = fitter.fit(data)

    print(result.means_)        # [[0, 0], [5, 5]]
    print(result.n_features)    # 2

Model Selection with BIC
------------------------

Use the Bayesian Information Criterion (BIC) to select the optimal number of components:

.. code-block:: python

    # Compare models with different n_components
    bics = {}
    for k in range(1, 6):
        fitter = GaussianMixtureFitter(n_components=k, random_state=42)
        result = fitter.fit(data)
        bics[k] = result.bic
        print(f"k={k}: BIC={result.bic:.2f}")

    # Select model with lowest BIC
    best_k = min(bics, key=bics.get)
    print(f"Best: k={best_k}")

BIC penalizes model complexity more than AIC, making it better for model selection.
Both metrics are available:

.. code-block:: python

    print(result.aic)  # Akaike Information Criterion
    print(result.bic)  # Bayesian Information Criterion

Prediction and Soft Clustering
------------------------------

Assign data points to components:

.. code-block:: python

    # Hard assignment (most likely component)
    labels = result.predict(data)
    print(labels)  # [0, 0, 0, ..., 1, 1, 1]

    # Soft assignment (probability for each component)
    probs = result.predict_proba(data)
    print(probs[0])  # [0.99, 0.01] - 99% component 0, 1% component 1

    # Access responsibilities from fitting
    print(result.responsibilities_)  # (n_samples, n_components) array

The ``responsibilities_`` property stores the soft assignments computed during fitting.

Probability Evaluation
----------------------

Evaluate probability density:

.. code-block:: python

    # PDF at specific points
    points = np.array([[0], [2.5], [5]])
    densities = result.pdf(points)

    # Log-PDF (more numerically stable)
    log_densities = result.logpdf(points)

    # Log-likelihood of the fitted model
    print(result.log_likelihood_)

Configuration Options
---------------------

The fitter supports several configuration options:

.. code-block:: python

    fitter = GaussianMixtureFitter(
        n_components=3,       # Number of mixture components
        max_iter=200,         # Maximum EM iterations
        tol=1e-5,             # Convergence tolerance
        n_init=10,            # Number of initializations (best kept)
        init_method="kmeans", # 'kmeans' or 'random'
        random_state=42,      # For reproducibility
        reg_covar=1e-6,       # Regularization for covariance
    )

**Key parameters:**

- ``n_init``: Run EM multiple times with different initializations to avoid local optima.
  Higher values are slower but more likely to find the global optimum.
- ``init_method``: ``"kmeans"`` (default) uses K-means++ for initialization, which is
  generally more robust than ``"random"``.
- ``reg_covar``: Small value added to covariance diagonal for numerical stability.
  Increase if you see singular covariance errors.

Convergence and Diagnostics
---------------------------

Check if the EM algorithm converged:

.. code-block:: python

    result = fitter.fit(data)

    print(result.converged_)  # True if converged before max_iter
    print(result.n_iter_)     # Number of iterations used
    print(result.n_samples_)  # Number of samples used in fitting

If ``converged_`` is ``False``, consider:

1. Increasing ``max_iter``
2. Using more initializations (``n_init``)
3. Reducing ``n_components`` if data doesn't support that many clusters

Serialization
-------------

Save and load fitted models:

.. code-block:: python

    # Save to JSON (recommended)
    result.save("gmm_model.json")

    # Or pickle for faster serialization
    result.save("gmm_model.pkl")

    # Load later
    from spark_bestfit import GaussianMixtureResult
    loaded = GaussianMixtureResult.load("gmm_model.json")

    # Use loaded model
    samples = loaded.sample(n=1000)
    labels = loaded.predict(new_data)

The JSON format includes all fitted parameters and metadata:

.. code-block:: javascript

    {
      "schema_version": "1.1",
      "spark_bestfit_version": "2.9.0",
      "created_at": "2026-01-10T23:00:00Z",
      "type": "gaussian_mixture",
      "n_components": 2,
      "weights": [0.5, 0.5],
      "means": [[0.0], [5.0]],
      "covariances": [[[1.0]], [[1.0]]],
      "converged": true,
      "n_iter": 15,
      "n_samples": 2000,
      "log_likelihood": -3500.5
    }

Numerical Stability
-------------------

The fitter includes regularization for numerical stability:

- ``reg_covar`` is added to covariance diagonals to prevent singularity
- Near-empty components trigger warnings
- Multiple initializations help avoid degenerate solutions

If you encounter issues:

.. code-block:: python

    # Increase regularization
    fitter = GaussianMixtureFitter(
        n_components=2,
        reg_covar=1e-4,  # Increased from default 1e-6
    )

    # Or reduce component count
    fitter = GaussianMixtureFitter(n_components=2)  # Instead of 5

Comparison with sklearn
-----------------------

``GaussianMixtureFitter`` follows a similar API to scikit-learn's ``GaussianMixture``
for familiarity, while following spark_bestfit conventions:

.. list-table::
   :header-rows: 1

   * - Feature
     - sklearn.mixture.GaussianMixture
     - spark_bestfit.GaussianMixtureFitter
   * - Dependencies
     - Requires sklearn
     - No additional deps
   * - Result type
     - Fitted estimator
     - GaussianMixtureResult dataclass
   * - Serialization
     - Joblib/pickle
     - JSON or pickle
   * - Covariance types
     - full/tied/diag/spherical
     - full (others planned)
   * - Integration
     - sklearn ecosystem
     - spark_bestfit ecosystem

API Reference
-------------

See :class:`spark_bestfit.mixture.GaussianMixtureFitter` and
:class:`spark_bestfit.mixture.GaussianMixtureResult` for full API documentation.
