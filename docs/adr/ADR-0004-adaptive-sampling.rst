ADR-0004: Adaptive Sampling Strategy
=====================================

:Status: Accepted
:Date: 2026-01-11 (v3.0.0)

Context
-------

Distribution fitting on large datasets (100M+ rows) requires sampling to
achieve reasonable performance. The original approach used uniform random
sampling, which works well for symmetric distributions but can miss important
characteristics of skewed data:

1. **Tail underrepresentation**: Heavy-tailed distributions (Pareto, lognormal)
   have rare but important extreme values that uniform sampling may miss
2. **Fitting failures**: Undersampled tails lead to poor parameter estimates,
   especially for shape parameters
3. **One-size-fits-all**: Symmetric data doesn't need stratified sampling's
   overhead

We needed sampling that adapts to data characteristics.

Decision
--------

We implemented adaptive sampling with three modes in ``config.py``::

    class SamplingMode(Enum):
        AUTO = "auto"        # Select based on skewness
        UNIFORM = "uniform"  # Force uniform random sampling
        STRATIFIED = "stratified"  # Force stratified sampling

**Skewness-based selection** (AUTO mode):

- ``|skew| < 0.5`` (mild): Uniform sampling - efficient for symmetric data
- ``0.5 <= |skew| < 2.0`` (moderate): Stratified with 5 percentile bins
- ``|skew| >= 2.0`` (high): Stratified with 10 bins + tail oversampling

**Configuration**::

    config = (FitterConfigBuilder()
        .with_adaptive_sampling(
            enabled=True,
            mode=SamplingMode.AUTO,
            skew_threshold_mild=0.5,
            skew_threshold_high=2.0,
        )
        .build())

**Implementation details:**

1. Skewness is computed on a small preliminary sample (10k rows)
2. Stratified sampling uses percentile-based bins to ensure representation
3. High-skew mode oversamples the 95th+ percentile tail
4. Thresholds are configurable for domain-specific tuning

Consequences
------------

**Positive:**

- Better parameter estimates for skewed distributions
- Automatic detection removes user guesswork
- Configurable thresholds allow domain tuning
- Backwards compatible: ``UNIFORM`` mode preserves old behavior

**Negative:**

- Additional computation for skewness detection
- Stratified sampling is slower than uniform
- Two-stage sampling (detect then sample) adds latency

**Neutral:**

- Default thresholds (0.5, 2.0) based on statistical literature for
  "moderate" and "high" skewness classifications

References
----------

- `PR #160 <https://github.com/dwsmith1983/spark-bestfit/pull/160>`_: Adaptive sampling (v3.0.0)
- `Issue #70 <https://github.com/dwsmith1983/spark-bestfit/issues/70>`_: Feature request
- Related: :doc:`ADR-0005-estimation-methods` (MSE for heavy tails)
