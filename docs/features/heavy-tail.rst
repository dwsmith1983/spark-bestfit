Heavy-Tail Detection
====================

spark-bestfit automatically detects **heavy-tailed data characteristics** and
warns you when standard distributions may provide poor fits. This helps identify
data that may need special handling.

What Are Heavy-Tailed Distributions?
------------------------------------

Heavy-tailed distributions have **slower tail decay** than normal or exponential
distributions. They exhibit:

- **High kurtosis**: More extreme values than a normal distribution
- **Extreme outliers**: Maximum values far beyond the 99th percentile
- **Potentially undefined moments**: Some (like Cauchy) have undefined variance

.. list-table:: Common Heavy-Tailed Distributions
   :header-rows: 1
   :widths: 25 30 45

   * - Distribution
     - Tail Behavior
     - Use Case
   * - ``cauchy``
     - Infinite variance
     - Ratios of normals, resonance phenomena
   * - ``pareto``
     - Power-law decay
     - Income distribution, file sizes, network traffic
   * - ``t`` (low df)
     - Heavy for df < 5
     - Financial returns, robust regression
   * - ``levy``
     - Extreme heavy tail
     - Anomalous diffusion
   * - ``burr``
     - Flexible heavy tail
     - Reliability analysis

Automatic Detection
-------------------

When fitting distributions, spark-bestfit checks two indicators:

1. **Excess kurtosis > 6**: Normal distribution has excess kurtosis = 0;
   t-distribution with 5 df has ~6; Cauchy is undefined (very high)

2. **Extreme ratio > 3**: The ratio of max value to 99th percentile

If either indicator triggers, a ``UserWarning`` is emitted:

.. code-block:: python

   from spark_bestfit import DistributionFitter, LocalBackend
   import numpy as np
   import pandas as pd
   import warnings

   # Generate heavy-tailed data
   np.random.seed(42)
   data = np.random.standard_cauchy(1000)
   df = pd.DataFrame({"value": data})

   fitter = DistributionFitter(backend=LocalBackend())

   # Warning is emitted automatically
   with warnings.catch_warnings(record=True) as w:
       warnings.simplefilter("always")
       results = fitter.fit(df, column="value", max_distributions=5)

       if w:
           print(f"Warning: {w[0].message}")
           # UserWarning: Column 'value' exhibits heavy-tail characteristics
           # (high kurtosis (299.7 > 6.0), extreme values (max/p99 = 17.2)).
           # Consider: (1) heavy-tail distributions like pareto, cauchy, t;
           # (2) data transformation (log, sqrt); (3) checking for outliers.

Direct API Usage
----------------

You can also use the detection function directly for diagnostic purposes:

.. code-block:: python

   from spark_bestfit.fitting import detect_heavy_tail, HEAVY_TAIL_DISTRIBUTIONS

   # Detect heavy-tail characteristics
   result = detect_heavy_tail(data)
   print(result)
   # {
   #     'is_heavy_tailed': True,
   #     'kurtosis': 299.7,
   #     'extreme_ratio': 17.2,
   #     'indicators': ['high kurtosis (299.7 > 6.0)', 'extreme values (max/p99 = 17.2)']
   # }

   # Custom threshold
   result = detect_heavy_tail(data, kurtosis_threshold=10.0)

   # List of known heavy-tail distributions
   print(HEAVY_TAIL_DISTRIBUTIONS)
   # frozenset({'cauchy', 'pareto', 't', 'levy', 'burr', 'burr12', 'fisk',
   #            'levy_l', 'levy_stable', 'lomax', 'powerlaw', 'invgauss',
   #            'genhyperbolic', 'johnsonsu'})

Data Statistics
---------------

The fit results now include kurtosis and skewness in the data statistics:

.. code-block:: python

   # After fitting
   best = results.best(n=1)[0]

   # Access via internal DataFrame
   print(results._df[['data_kurtosis', 'data_skewness']].iloc[0])

   # Or compute directly
   from spark_bestfit.fitting import compute_data_stats

   stats = compute_data_stats(data)
   print(f"Kurtosis: {stats['data_kurtosis']:.2f}")
   print(f"Skewness: {stats['data_skewness']:.2f}")

Handling Heavy-Tailed Data
--------------------------

When you see the heavy-tail warning, consider these approaches:

**1. Use Heavy-Tail Distributions**

Limit fitting to heavy-tail distributions:

.. code-block:: python

   from spark_bestfit.fitting import HEAVY_TAIL_DISTRIBUTIONS

   # Only fit heavy-tail distributions
   heavy_tail_list = list(HEAVY_TAIL_DISTRIBUTIONS)
   results = fitter.fit(df, "value", max_distributions=len(heavy_tail_list))

   # Or exclude non-heavy-tail distributions from default set
   fitter = DistributionFitter(
       backend=LocalBackend(),
       excluded_distributions=("norm", "expon", "gamma", "beta")
   )

**2. Transform Data**

Apply transformations to reduce tail heaviness:

.. code-block:: python

   import numpy as np

   # Log transform (for positive data)
   df["log_value"] = np.log(df["value"] + 1)

   # Square root transform
   df["sqrt_value"] = np.sqrt(np.abs(df["value"]))

   # Winsorize (clip extremes)
   lower, upper = np.percentile(df["value"], [1, 99])
   df["winsorized"] = df["value"].clip(lower, upper)

**3. Check for Outliers**

Investigate whether extreme values are errors:

.. code-block:: python

   # Identify extreme values
   threshold = np.percentile(data, 99.9)
   outliers = data[data > threshold]
   print(f"Extreme values: {len(outliers)}")

   # Consider removing if they're data errors
   clean_data = data[data <= threshold]

Suppressing Warnings
--------------------

If you're aware of the heavy-tail nature and want to suppress warnings:

.. code-block:: python

   import warnings

   with warnings.filterwarnings("ignore", message=".*heavy-tail.*"):
       results = fitter.fit(df, column="value")

   # Or globally
   warnings.filterwarnings("ignore", message=".*heavy-tail.*")

When Detection Doesn't Apply
----------------------------

The heavy-tail detection is a heuristic. It may:

- **False positive**: Flag data with a few outliers that isn't truly heavy-tailed
- **False negative**: Miss heavy-tailed data with small samples or clipped values

Use it as a diagnostic aid, not a definitive classification.
