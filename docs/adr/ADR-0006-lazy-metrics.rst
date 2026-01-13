ADR-0006: Lazy Metrics Pattern
===============================

:Status: Accepted
:Date: 2026-01-01 (v1.5.0)

Context
-------

Distribution fitting computes multiple goodness-of-fit metrics:

- **SSE**: Sum of Squared Errors (always computed, fast)
- **AIC/BIC**: Information criteria (always computed, fast)
- **KS**: Kolmogorov-Smirnov statistic and p-value
- **AD**: Anderson-Darling statistic

KS and AD statistics require evaluating the CDF at every data point, making
them O(n) operations that dominate fitting time for large datasets:

- Fitting 90 distributions with KS/AD: ~15 seconds
- Fitting 90 distributions without KS/AD: ~3 seconds

Many workflows only need AIC/BIC for model selection, making KS/AD computation
wasteful.

Decision
--------

We implemented lazy metric computation via the ``lazy_metrics`` config option::

    config = (FitterConfigBuilder()
        .with_lazy_metrics(True)
        .build())

**Behavior:**

- When ``lazy_metrics=False`` (default): KS/AD computed during parallel fit
- When ``lazy_metrics=True``: KS/AD deferred until accessed

**Implementation** (``results.py``)::

    class LazyFitResult(FitResult):
        def __init__(self, ..., data_sample: np.ndarray):
            self._data_sample = data_sample
            self._ks_statistic: Optional[float] = None
            self._ks_pvalue: Optional[float] = None

        @property
        def ks_statistic(self) -> float:
            if self._ks_statistic is None:
                self._compute_ks()
            return self._ks_statistic

        def _compute_ks(self) -> None:
            # Compute on-demand using stored data sample

**Class hierarchy** (v2.1.0 refinement)::

    FitResult (base)
    ├── EagerFitResult  # KS/AD computed upfront
    └── LazyFitResult   # KS/AD computed on access

**Data lifecycle:**

- Lazy results store a reference to the data sample
- After KS/AD computation, sample can be released via ``release_data()``
- Serialization includes computed values, not raw data

Consequences
------------

**Positive:**

- 5x speedup for AIC/BIC-only workflows
- No API change: properties still accessible, just deferred
- Users only pay for metrics they actually use
- Parallel fitting remains embarrassingly parallel

**Negative:**

- Memory overhead: lazy results hold data sample until metrics accessed
- First access to KS/AD incurs computation delay
- Serialization of lazy results may include uncomputed metrics as None

**Neutral:**

- Default is ``lazy_metrics=False`` for backwards compatibility
- Lazy results explicitly document their deferred behavior

References
----------

- `PR #72 <https://github.com/dwsmith1983/spark-bestfit/pull/72>`_: Lazy metrics (v1.5.0)
- `PR #92 <https://github.com/dwsmith1983/spark-bestfit/pull/92>`_: Eager/lazy class hierarchy (v2.1.0)
- Related: :doc:`ADR-0003-configuration-system` (FitterConfig)
