Maximum Spacing Estimation
==========================

spark-bestfit supports **Maximum Spacing Estimation (MSE)** as an alternative
to Maximum Likelihood Estimation (MLE) for parameter fitting. MSE is particularly
robust for heavy-tailed distributions where MLE may fail or produce poor estimates.

What is Maximum Spacing Estimation?
-----------------------------------

MSE estimates distribution parameters by maximizing the **geometric mean of spacings**
between consecutive order statistics of the CDF-transformed data.

For data points x₁ ≤ x₂ ≤ ... ≤ xₙ and CDF F with parameters θ:

1. Transform data: uᵢ = F(xᵢ; θ) where uᵢ ∈ [0,1]
2. Compute spacings: Dᵢ = u₍ᵢ₎ - u₍ᵢ₋₁₎ (with boundary values 0 and 1)
3. Maximize: S(θ) = (1/(n+1)) Σᵢ log(Dᵢ)

**Key advantages over MLE:**

- Always well-defined when the CDF exists (MLE can be unbounded)
- More robust to outliers and extreme values
- Better convergence for heavy-tailed distributions (Pareto, Cauchy, etc.)
- Consistent and asymptotically efficient

When to Use MSE
---------------

.. list-table:: Estimation Method Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Best For
     - Limitations
   * - ``mle``
     - Most distributions, large samples
     - Can fail for heavy tails, unbounded likelihood
   * - ``mse``
     - Heavy-tailed distributions, outliers
     - Slightly slower than MLE
   * - ``auto``
     - Unknown data characteristics
     - Adds detection overhead

**Use MSE when:**

- Fitting heavy-tailed distributions (Pareto, Cauchy, Levy, etc.)
- Data has extreme outliers
- MLE fails to converge or produces unreasonable estimates
- You want more robust parameter estimates

API: estimation_method Parameter
--------------------------------

The ``estimation_method`` parameter accepts three values:

- ``"mle"`` (default): Maximum Likelihood Estimation via ``scipy.stats.fit()``
- ``"mse"``: Maximum Spacing Estimation
- ``"auto"``: Automatically select MSE for heavy-tailed data, MLE otherwise

**Direct parameter usage:**

.. code-block:: python

   from spark_bestfit import DistributionFitter, LocalBackend
   import pandas as pd
   import numpy as np

   # Generate heavy-tailed data
   np.random.seed(42)
   data = np.random.pareto(1.5, 1000) + 1
   df = pd.DataFrame({"value": data})

   fitter = DistributionFitter(backend=LocalBackend())

   # Use MSE for heavy-tailed data
   results = fitter.fit(df, column="value", estimation_method="mse")

   # Auto-detect and select appropriate method
   results = fitter.fit(df, column="value", estimation_method="auto")

**Via FitterConfig:**

.. code-block:: python

   from spark_bestfit import FitterConfigBuilder

   # Build config with MSE
   config = (FitterConfigBuilder()
       .with_estimation_method("mse")
       .with_bins(100)
       .build())

   results = fitter.fit(df, column="value", config=config)

Examples
--------

**Example 1: Fitting Pareto Distribution**

Pareto distributions are notoriously difficult for MLE when the shape parameter
is small. MSE handles this robustly:

.. code-block:: python

   from scipy import stats
   import numpy as np
   import pandas as pd
   from spark_bestfit import DistributionFitter, LocalBackend

   # Generate Pareto data with shape=1.5
   np.random.seed(42)
   data = stats.pareto.rvs(b=1.5, size=1000, random_state=42) + 1
   df = pd.DataFrame({"value": data})

   fitter = DistributionFitter(backend=LocalBackend())

   # MSE provides more stable estimates
   results = fitter.fit(
       df,
       column="value",
       estimation_method="mse",
       max_distributions=10
   )

   best = results.best(n=1)[0]
   print(f"Best fit: {best.distribution}")
   print(f"Parameters: {best.params}")

**Example 2: Auto Mode for Unknown Data**

When you don't know if your data is heavy-tailed, use ``"auto"``:

.. code-block:: python

   # Auto mode detects heavy tails and switches to MSE
   results = fitter.fit(
       df,
       column="value",
       estimation_method="auto"
   )

   # No heavy-tail warning when auto selects MSE

**Example 3: Cauchy Distribution**

Cauchy has undefined mean and variance, making MLE unstable. MSE works well:

.. code-block:: python

   # Generate Cauchy data
   data = stats.cauchy.rvs(loc=5.0, scale=2.0, size=500, random_state=42)
   df = pd.DataFrame({"value": data})

   # MSE gives stable parameter estimates
   results = fitter.fit(
       df,
       column="value",
       estimation_method="mse",
       max_distributions=5
   )

Low-Level API
-------------

For direct access to MSE fitting:

.. code-block:: python

   from spark_bestfit.fitting import fit_mse
   from scipy import stats
   import numpy as np

   # Generate data
   np.random.seed(42)
   data = np.random.normal(10.0, 2.0, 1000)

   # Fit using MSE
   params = fit_mse(stats.norm, data)
   print(f"Parameters: loc={params[0]:.2f}, scale={params[1]:.2f}")

   # With initial parameter guess (for faster convergence)
   params = fit_mse(stats.norm, data, initial_params=(9.0, 1.5))

Integration with Heavy-Tail Detection
-------------------------------------

MSE integrates seamlessly with spark-bestfit's heavy-tail detection:

- When ``estimation_method="auto"``, heavy-tail detection runs automatically
- If heavy tails are detected, MSE is used instead of MLE
- When explicitly using ``estimation_method="mse"``, the heavy-tail warning is suppressed
  (since you're already using the recommended approach)

.. code-block:: python

   import warnings

   # With auto: warning if heavy-tailed but shows we're using MSE
   results = fitter.fit(df, "value", estimation_method="auto")

   # With explicit mse: no warning (you know what you're doing)
   with warnings.catch_warnings(record=True) as w:
       warnings.simplefilter("always")
       results = fitter.fit(df, "value", estimation_method="mse")
       heavy_tail_warnings = [x for x in w if "heavy-tail" in str(x.message)]
       assert len(heavy_tail_warnings) == 0  # No warning

Performance Considerations
--------------------------

MSE is slightly slower than MLE because it requires optimization over the
spacing objective function. Typical overhead:

- **Small datasets (<1000 points)**: ~2x slower than MLE
- **Large datasets (>10000 points)**: ~1.5x slower than MLE

For performance-critical applications with known non-heavy-tailed data,
stick with the default ``estimation_method="mle"``.

References
----------

- Ranneby, B. (1984). "The Maximum Spacing Method. An Estimation Method
  Related to the Maximum Likelihood Method." *Scandinavian Journal of
  Statistics*, 11(2), 93-112.
