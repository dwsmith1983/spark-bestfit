ADR-0005: Estimation Methods
=============================

:Status: Accepted
:Date: 2026-01-10 (v2.5.0)

Context
-------

Maximum Likelihood Estimation (MLE) via ``scipy.stats.fit()`` is the standard
approach for parameter estimation. However, MLE has known limitations:

1. **Heavy-tailed data**: MLE can fail or produce poor estimates for
   distributions like Pareto, Cauchy, or data with extreme outliers
2. **Unbounded likelihood**: Some distributions have likelihood functions
   that can become unbounded, causing optimizer divergence
3. **Sensitivity to outliers**: A single extreme value can dramatically
   affect MLE estimates

We needed an alternative estimation method for challenging data.

Decision
--------

We added Maximum Spacing Estimation (MSE) as an alternative to MLE,
configurable via ``FitterConfig``::

    config = (FitterConfigBuilder()
        .with_estimation_method("mse")  # or "mle" or "auto"
        .build())

**Methods available:**

1. **MLE** (default): Maximum Likelihood Estimation via ``scipy.stats.fit()``.
   Fast and accurate for most distributions.

2. **MSE**: Maximum Spacing Estimation (Moran-Ranneby). Maximizes the
   geometric mean of spacings between order statistics. More robust for
   heavy-tailed data.

3. **AUTO**: Automatically selects MSE for heavy-tailed data based on:
   - Kurtosis > 10 (leptokurtic)
   - Extreme value analysis (max/min beyond 5 IQR from median)

**Implementation** (``estimation.py``)::

    def estimate_parameters_mse(
        data: np.ndarray,
        distribution: rv_continuous,
        bounds: Tuple[Tuple[float, float], ...],
    ) -> Tuple[float, ...]:
        # 1. Sort data to get order statistics
        # 2. Compute spacings: F(x[i+1]) - F(x[i])
        # 3. Maximize product (or sum of logs) of spacings
        # 4. Use scipy.optimize.minimize with bounds

**Heavy-tail detection** (``estimation.py``)::

    def detect_heavy_tail(data: np.ndarray) -> bool:
        kurtosis = scipy.stats.kurtosis(data)
        if kurtosis > 10:
            return True
        # Also check for extreme outliers via IQR

Consequences
------------

**Positive:**

- Robust fitting for Pareto, Cauchy, and other heavy-tailed distributions
- Automatic detection removes user guesswork
- Graceful degradation: if MSE fails, falls back to MLE
- User can force either method for control

**Negative:**

- MSE is slower than MLE (requires optimization over spacings)
- MSE requires sorted data, adding O(n log n) overhead
- Not all distributions benefit from MSE

**Neutral:**

- ``AUTO`` mode logs which method was selected for transparency
- Heavy-tail warning is emitted when detected (v2.3.0)

References
----------

- `Commit e0fb25b <https://github.com/dwsmith1983/spark-bestfit/commit/e0fb25b>`_: MSE implementation (v2.5.0)
- `PR #98 <https://github.com/dwsmith1983/spark-bestfit/pull/98>`_: Heavy-tail detection (v2.3.0)
- Ranneby, B. (1984). "The Maximum Spacing Method"
- Related: :doc:`ADR-0004-adaptive-sampling` (stratified sampling for tails)
