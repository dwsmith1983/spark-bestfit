ADR-0002: Distribution Registry Pattern
========================================

:Status: Accepted
:Date: 2025-12-24 (v0.4.0 discrete), 2026-01-09 (v2.4.0 custom)

Context
-------

spark-bestfit fits data against scipy.stats distributions. However, not all
~100 scipy continuous distributions are practical to fit:

1. **Performance**: Some distributions (``levy_stable``, ``studentized_range``)
   take seconds per fit, making parallel fitting impractical
2. **Numerical stability**: Some distributions (``wald``, ``geninvgauss``) can
   hang or produce invalid results with certain data
3. **Discrete differences**: Discrete distributions lack scipy's ``fit()``
   method and require custom parameter estimation logic
4. **Extensibility**: Users may want to fit custom distributions not in scipy

We needed a centralized way to manage which distributions are available,
excluded, and how they're configured.

Decision
--------

We created two registry classes in ``distributions.py``:

**DistributionRegistry** (continuous distributions)::

    class DistributionRegistry:
        DEFAULT_EXCLUSIONS = {
            "levy_stable",    # Extremely slow
            "studentized_range",  # Very slow
            "geninvgauss",    # Can hang
            # ... 19 total exclusions
        }

        SLOW_DISTRIBUTIONS = {
            "powerlognorm",   # ~160ms
            "t",              # ~144ms
            # ... used for partition weighting
        }

        def get_distributions(
            self,
            support_at_zero: bool = False,
            additional_exclusions: Optional[List[str]] = None,
        ) -> List[str]: ...

        def register_distribution(
            self,
            name: str,
            distribution: rv_continuous,
        ) -> None: ...

**DiscreteDistributionRegistry** (discrete distributions)::

    class DiscreteDistributionRegistry:
        def __init__(self):
            self._param_configs = self._build_param_configs()

        def get_param_config(self, dist_name: str) -> Dict[str, Any]:
            # Returns: initial estimates, bounds, param_names
            # Needed because discrete dists lack fit()

**Key design choices:**

1. **Default exclusions**: Curated list of problematic distributions, not
   a blanket ban. Users can override with ``remove_exclusion()``.

2. **Support filtering**: ``support_at_zero=True`` filters to non-negative
   distributions (where ``dist.a >= 0``), useful for positive-only data.

3. **Slow distribution tracking**: Used for partition weighting to balance
   load across workers (slower dists get fewer per partition).

4. **Custom distribution registration** (v2.4.0): Users can register
   ``rv_continuous`` subclasses with validation of required methods.

Consequences
------------

**Positive:**

- Sensible defaults: Users get fast, stable fitting out of the box
- Flexibility: Power users can include excluded distributions or add custom ones
- Documentation: Exclusion reasons are documented in code comments
- Partition balancing: Slow distributions don't create stragglers

**Negative:**

- Maintenance burden: New scipy versions may add distributions requiring
  evaluation for exclusion
- Custom distributions require scipy rv_continuous interface knowledge

**Neutral:**

- Default exclusions are based on empirical timing measurements (documented
  in code comments with approximate durations)

References
----------

- `PR #22 <https://github.com/dwsmith1983/spark-bestfit/pull/22>`_: Discrete distribution fitting (v0.4.0)
- `PR #76 <https://github.com/dwsmith1983/spark-bestfit/pull/76>`_: Distribution-aware partitioning (v1.7.0)
- `PR #102 <https://github.com/dwsmith1983/spark-bestfit/pull/102>`_: Custom distribution support (v2.4.0)
