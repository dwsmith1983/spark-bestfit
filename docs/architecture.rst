Architecture
============

This page provides a visual overview of spark-bestfit's architecture,
showing component relationships and data flow during distribution fitting.

Component Architecture
----------------------

spark-bestfit uses a layered architecture with pluggable backends:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         User Application                            │
    │                    (Your Python/Spark/Ray code)                     │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Public API Layer                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │  DistributionFitter    │  DiscreteDistributionFitter                │
    │  GaussianCopula        │  GaussianMixtureFitter                     │
    │  MultivariateNormalFitter                                           │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       Configuration Layer                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  FitterConfig          │  FitterConfigBuilder                       │
    │  DistributionRegistry  │  DiscreteDistributionRegistry              │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   ExecutionBackend Protocol                         │
    │              (Abstract interface - PEP 544 Protocol)                │
    ├─────────────────────────────────────────────────────────────────────┤
    │  SparkBackend          │  RayBackend          │  LocalBackend       │
    │  (Apache Spark)        │  (Ray clusters)      │  (ThreadPool)       │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      Core Fitting Engine                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │  estimation.py         │  fitting.py          │  metrics.py         │
    │  (MLE/MSE estimation)  │  (scipy.stats fit)   │  (KS, AD, AIC, BIC) │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       Results & Output                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  FitResults            │  DistributionFitResult                     │
    │  (Collection)          │  (Individual result with metrics)          │
    └─────────────────────────────────────────────────────────────────────┘

Layer Descriptions
------------------

**Public API Layer**
    User-facing classes for distribution fitting. ``DistributionFitter`` handles
    continuous distributions (~90 scipy.stats distributions), while
    ``DiscreteDistributionFitter`` handles count data (16 discrete distributions).
    Specialized fitters exist for copulas, mixtures, and multivariate normals.

**Configuration Layer**
    Controls fitting behavior. ``FitterConfig`` specifies bins, thresholds,
    excluded distributions, and sampling modes. Distribution registries manage
    the mapping between distribution names and scipy.stats implementations.

**ExecutionBackend Protocol**
    Abstraction layer enabling pluggable backends. Uses Python's structural
    subtyping (PEP 544) - any class implementing the required methods works.
    The ``BackendFactory`` provides convenient creation and auto-detection.

**Core Fitting Engine**
    Pure Python fitting logic that runs on workers. Parameter estimation
    (MLE or MSE), goodness-of-fit metrics (KS, AD, AIC, BIC, SSE), and
    histogram-based fitting algorithms. This code is serialized and sent
    to distributed workers.

**Results & Output**
    Structured results with rich querying. ``FitResults`` holds all fitted
    distributions with methods like ``best(n=5)`` and ``filter()``.
    Individual ``DistributionFitResult`` objects contain parameters,
    metrics, and methods for confidence intervals and sampling.

Data Flow: Distribution Fitting
-------------------------------

The following diagram shows the data flow during a ``fitter.fit()`` call:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │                           INPUT DATA                                 │
    │              (Spark DataFrame / Ray Dataset / pandas)                │
    └──────────────────────────────────────────────────────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
         ┌──────────────────────┐         ┌──────────────────────┐
         │  Distributed Stats   │         │   Distributed        │
         │  (min, max, count)   │         │   Histogram          │
         │       ~O(N)          │         │   Computation        │
         └──────────────────────┘         │       ~O(N)          │
                     │                    └──────────────────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                     DRIVER COLLECTS                                  │
    │              histogram (~8KB) + sample (~80KB)                       │
    │                    Raw data stays distributed                        │
    └──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                   BROADCAST TO WORKERS                               │
    │              histogram + sample cached on each worker                │
    └──────────────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
    │   Worker 1    │         │   Worker 2    │         │   Worker N    │
    │  fit: norm    │         │  fit: gamma   │         │  fit: beta    │
    │  fit: expon   │         │  fit: weibull │         │  fit: lognorm │
    │     ...       │         │     ...       │         │     ...       │
    └───────────────┘         └───────────────┘         └───────────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    COLLECT RESULTS                                   │
    │          ~90 DistributionFitResult objects (~50KB total)             │
    └──────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      FitResults                                      │
    │              .best(n=5)  .filter()  .to_pandas()                     │
    └──────────────────────────────────────────────────────────────────────┘

Key Design Decisions
--------------------

**Histogram-based fitting**
    Rather than sending raw data to workers, spark-bestfit computes a histogram
    once and broadcasts it. This provides sub-linear scaling with data size -
    fitting 1M rows takes nearly the same time as fitting 100K rows.

**Embarrassingly parallel distribution fitting**
    Each distribution is fitted independently, making the workload perfectly
    parallelizable. Distributions are interleaved to spread slow ones
    (like ``burr``, ``t``) across partitions.

**Protocol-based backend abstraction**
    The ``ExecutionBackend`` protocol uses structural subtyping (duck typing),
    so backends don't need to inherit from a base class. This keeps backend
    implementations simple and testable.

**Broadcast variables for efficiency**
    Histogram and sample data are broadcast once to all workers, avoiding
    repeated serialization overhead for each fitting task.

See Also
--------

- :doc:`backends` - Detailed backend comparison and usage
- :doc:`performance` - Scaling characteristics and tuning recommendations
- :doc:`api` - Complete API reference
