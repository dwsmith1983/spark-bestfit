ADR-0003: Configuration System
===============================

:Status: Accepted
:Date: 2026-01-08 (v2.2.0)

Context
-------

The ``fit()`` method originally accepted numerous keyword arguments for
configuration:

.. code-block:: python

    results = fitter.fit(
        df, column="value",
        bins=100,
        bounded=True,
        lower_bound=0,
        enable_sampling=True,
        sample_fraction=0.1,
        lazy_metrics=True,
        # ... many more
    )

This approach had several problems:

1. **API bloat**: New features added more parameters, making the signature
   unwieldy
2. **Discoverability**: Users couldn't easily see all available options
3. **Reusability**: Configurations couldn't be stored, shared, or reused
4. **Type safety**: IDE autocomplete was limited; typos weren't caught

Decision
--------

We introduced a two-part configuration system in ``config.py``:

**FitterConfig** (immutable dataclass)::

    @dataclass(frozen=True)
    class FitterConfig:
        # Histogram
        bins: Union[int, Tuple[float, ...]] = 50
        use_rice_rule: bool = True

        # Distribution selection
        support_at_zero: bool = False
        max_distributions: Optional[int] = None
        prefilter: Union[bool, str] = False

        # Sampling
        enable_sampling: bool = True
        sample_fraction: Optional[float] = None
        max_sample_size: int = 1_000_000
        # ... additional fields

**FitterConfigBuilder** (fluent builder)::

    config = (FitterConfigBuilder()
        .with_bins(100)
        .with_bounds(lower=0, upper=100)
        .with_sampling(fraction=0.1)
        .with_lazy_metrics()
        .build())

    results = fitter.fit(df, column="value", config=config)

**Design principles:**

1. **Immutable**: ``frozen=True`` prevents accidental modification; changes
   create new instances via ``dataclasses.replace()``

2. **Fluent builder**: Method chaining provides discoverable API with IDE
   autocomplete on each ``with_*`` method

3. **Grouped parameters**: Builder methods group related options
   (e.g., ``with_sampling()`` handles fraction, max_size, threshold together)

4. **Backwards compatible**: Individual kwargs still work for simple cases;
   config is optional

5. **Version tagging**: Docstrings note when features were added
   (e.g., "v2.9.0") for migration guidance

Consequences
------------

**Positive:**

- IDE autocomplete shows all options via builder methods
- Configurations can be stored, serialized, and reused across fits
- Type checking catches errors at development time
- Immutability prevents accidental state mutation
- Grouped options reduce cognitive load

**Negative:**

- Two ways to configure (kwargs vs config) may cause confusion initially
- Builder pattern adds some verbosity for simple cases

**Neutral:**

- Default values match historical behavior for backwards compatibility
- Config object is passed through to backends for parallel execution

References
----------

- `PR #93 <https://github.com/dwsmith1983/spark-bestfit/pull/93>`_: FitterConfig builder pattern (v2.2.0)
- Python dataclasses documentation
- Builder pattern (Gang of Four)
