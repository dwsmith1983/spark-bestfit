ADR-0001: Multi-Backend Architecture
=====================================

:Status: Accepted
:Date: 2026-01-04 (v2.0.0)

Context
-------

spark-bestfit was originally designed exclusively for Apache Spark, using
Pandas UDFs for parallel distribution fitting. However, this created several
limitations:

1. **Development friction**: Local testing required a full Spark installation
2. **ML workflow gaps**: Ray is increasingly popular for ML pipelines, but
   users had to convert data between formats
3. **Small dataset overhead**: Spark's overhead isn't justified for datasets
   that fit in memory

We needed a way to support multiple execution backends while maintaining a
consistent API and avoiding code duplication.

Decision
--------

We introduced the ``ExecutionBackend`` protocol using Python's structural
subtyping (PEP 544). Any class implementing the required methods is compatible
without explicit inheritance.

**Protocol definition** (``protocols.py``)::

    @runtime_checkable
    class ExecutionBackend(Protocol):
        def broadcast(self, data: Any) -> Any: ...
        def destroy_broadcast(self, handle: Any) -> None: ...
        def parallel_fit(...) -> List[Dict[str, Any]]: ...
        def get_parallelism(self) -> int: ...
        def collect_column(self, df: Any, column: str) -> np.ndarray: ...
        # ... additional methods

**Backend implementations:**

- ``SparkBackend``: Apache Spark via Pandas UDFs (original behavior)
- ``LocalBackend``: ``concurrent.futures.ProcessPoolExecutor`` for development
- ``RayBackend``: Ray distributed computing for ML workflows

**Factory pattern** (``backends/factory.py``)::

    class BackendFactory:
        @classmethod
        def for_dataframe(cls, df: Any) -> ExecutionBackend:
            # Auto-detect: Ray Dataset -> RayBackend
            #              pandas DataFrame -> LocalBackend
            #              else -> SparkBackend

        @classmethod
        def create(cls, backend_type: str, **kwargs) -> ExecutionBackend:
            # Explicit creation by name

**Lazy imports**: Optional dependencies (PySpark, Ray) are imported only when
the corresponding backend is instantiated, allowing installation without all
backends.

Consequences
------------

**Positive:**

- Users can develop locally with ``LocalBackend`` without Spark
- Ray users get native integration without data format conversion
- Consistent API regardless of backend choice
- Duck typing enables future backends without modifying core code
- Optional dependencies reduce installation size

**Negative:**

- Protocol methods must be implemented consistently across backends
- Testing matrix grows with each backend (currently 3x)
- Some backend-specific optimizations may not be portable

**Neutral:**

- Fitters accept an optional ``backend`` parameter; if omitted, auto-detection
  is used based on DataFrame type

References
----------

- `PR #86 <https://github.com/dwsmith1983/spark-bestfit/pull/86>`_: V2.0.0 Multi-backend architecture
- `PR #93 <https://github.com/dwsmith1983/spark-bestfit/pull/93>`_: BackendFactory addition (v2.2.0)
- PEP 544: Structural Subtyping (Protocols)
