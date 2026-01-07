"""Configuration classes for distribution fitting.

This module provides a clean API for configuring distribution fitting
through the FitterConfig dataclass and FitterConfigBuilder fluent builder.

Example:
    >>> config = (FitterConfigBuilder()
    ...     .with_bins(100)
    ...     .with_bounds(lower=0, upper=100)
    ...     .with_sampling(fraction=0.1)
    ...     .build())
    >>> results = fitter.fit(df, column="value", config=config)
"""

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class FitterConfig:
    """Immutable configuration for distribution fitting.

    This dataclass holds all configuration parameters for the fit() method.
    Use FitterConfigBuilder for a fluent API to create configurations.

    Attributes:
        bins: Number of histogram bins or tuple of bin edges (continuous only).
        use_rice_rule: Auto-determine bin count using Rice rule (continuous only).
        support_at_zero: Only fit non-negative distributions (continuous only).
        max_distributions: Limit number of distributions to fit (for testing).
        prefilter: Pre-filter incompatible distributions (continuous only).
        enable_sampling: Enable sampling for large datasets.
        sample_fraction: Fraction to sample (None = auto-determine).
        max_sample_size: Maximum rows when auto-determining sample size.
        sample_threshold: Row count above which sampling is applied.
        bounded: Enable truncated distribution fitting.
        lower_bound: Lower bound for truncated fitting (scalar or per-column dict).
        upper_bound: Upper bound for truncated fitting (scalar or per-column dict).
        num_partitions: Number of parallel partitions (None = auto-determine).
        lazy_metrics: Defer KS/AD computation until accessed.
        progress_callback: Optional callback for progress updates.

    Example:
        >>> config = FitterConfig(bins=100, bounded=True, lower_bound=0)
        >>> results = fitter.fit(df, column="value", config=config)
    """

    # === Histogram (continuous only) ===
    bins: Union[int, Tuple[float, ...]] = 50
    use_rice_rule: bool = True

    # === Distribution Selection ===
    support_at_zero: bool = False
    max_distributions: Optional[int] = None
    prefilter: Union[bool, str] = False

    # === Sampling ===
    enable_sampling: bool = True
    sample_fraction: Optional[float] = None
    max_sample_size: int = 1_000_000
    sample_threshold: int = 10_000_000

    # === Bounds ===
    bounded: bool = False
    lower_bound: Optional[Union[float, Dict[str, float]]] = None
    upper_bound: Optional[Union[float, Dict[str, float]]] = None

    # === Performance ===
    num_partitions: Optional[int] = None
    lazy_metrics: bool = False

    # === Callbacks ===
    progress_callback: Optional[Callable[[int, int, float], None]] = None

    def with_progress_callback(self, callback: Optional[Callable[[int, int, float], None]]) -> "FitterConfig":
        """Return a new config with the specified progress callback.

        Since FitterConfig is frozen, this returns a new instance.

        Args:
            callback: Progress callback function (completed, total, percent).

        Returns:
            New FitterConfig with the callback set.
        """
        return replace(self, progress_callback=callback)


class FitterConfigBuilder:
    """Fluent builder for FitterConfig.

    Provides a chainable API for constructing FitterConfig instances.
    Each method returns self to enable method chaining.

    Example:
        >>> config = (FitterConfigBuilder()
        ...     .with_bins(100)
        ...     .with_bounds(lower=0, upper=100)
        ...     .with_sampling(fraction=0.1)
        ...     .with_lazy_metrics()
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        # Histogram
        self._bins: Union[int, Tuple[float, ...]] = 50
        self._use_rice_rule: bool = True

        # Distribution selection
        self._support_at_zero: bool = False
        self._max_distributions: Optional[int] = None
        self._prefilter: Union[bool, str] = False

        # Sampling
        self._enable_sampling: bool = True
        self._sample_fraction: Optional[float] = None
        self._max_sample_size: int = 1_000_000
        self._sample_threshold: int = 10_000_000

        # Bounds
        self._bounded: bool = False
        self._lower_bound: Optional[Union[float, Dict[str, float]]] = None
        self._upper_bound: Optional[Union[float, Dict[str, float]]] = None

        # Performance
        self._num_partitions: Optional[int] = None
        self._lazy_metrics: bool = False

    def with_bins(
        self,
        bins: Union[int, Tuple[float, ...]] = 50,
        use_rice_rule: bool = True,
    ) -> "FitterConfigBuilder":
        """Configure histogram binning (continuous distributions only).

        Args:
            bins: Number of bins or explicit bin edges.
            use_rice_rule: Auto-determine bins using Rice rule if bins not specified.

        Returns:
            Self for method chaining.
        """
        self._bins = bins
        self._use_rice_rule = use_rice_rule
        return self

    def with_bounds(
        self,
        lower: Optional[Union[float, Dict[str, float]]] = None,
        upper: Optional[Union[float, Dict[str, float]]] = None,
        auto_detect: bool = False,
    ) -> "FitterConfigBuilder":
        """Configure bounded (truncated) distribution fitting.

        Args:
            lower: Lower bound (scalar for all columns, or dict per column).
            upper: Upper bound (scalar for all columns, or dict per column).
            auto_detect: If True, detect bounds from data min/max.

        Returns:
            Self for method chaining.
        """
        self._bounded = auto_detect or (lower is not None or upper is not None)
        self._lower_bound = lower
        self._upper_bound = upper
        return self

    def with_sampling(
        self,
        fraction: Optional[float] = None,
        max_size: int = 1_000_000,
        threshold: int = 10_000_000,
        enabled: bool = True,
    ) -> "FitterConfigBuilder":
        """Configure data sampling for large datasets.

        Args:
            fraction: Explicit sample fraction (None = auto-determine).
            max_size: Maximum sample size when auto-determining.
            threshold: Row count above which sampling is applied.
            enabled: Whether sampling is enabled.

        Returns:
            Self for method chaining.
        """
        self._enable_sampling = enabled
        self._sample_fraction = fraction
        self._max_sample_size = max_size
        self._sample_threshold = threshold
        return self

    def with_lazy_metrics(self, lazy: bool = True) -> "FitterConfigBuilder":
        """Configure lazy metric computation.

        When enabled, expensive KS/AD statistics are computed on-demand
        rather than upfront, improving performance for AIC/BIC workflows.

        Args:
            lazy: Whether to defer KS/AD computation.

        Returns:
            Self for method chaining.
        """
        self._lazy_metrics = lazy
        return self

    def with_prefilter(self, mode: Union[bool, str] = True) -> "FitterConfigBuilder":
        """Configure distribution pre-filtering (continuous only).

        Pre-filters distributions that are incompatible with the data
        based on skewness and kurtosis.

        Args:
            mode: True for default filtering, or string for specific mode.

        Returns:
            Self for method chaining.
        """
        self._prefilter = mode
        return self

    def with_support_at_zero(self, enabled: bool = True) -> "FitterConfigBuilder":
        """Configure non-negative distribution filtering (continuous only).

        When enabled, only fits distributions with support at zero
        (e.g., exponential, gamma, lognormal).

        Args:
            enabled: Whether to filter for non-negative distributions.

        Returns:
            Self for method chaining.
        """
        self._support_at_zero = enabled
        return self

    def with_max_distributions(self, n: Optional[int]) -> "FitterConfigBuilder":
        """Limit the number of distributions to fit.

        Useful for testing or when you only need top candidates.

        Args:
            n: Maximum number of distributions (None = no limit).

        Returns:
            Self for method chaining.
        """
        self._max_distributions = n
        return self

    def with_partitions(self, n: Optional[int]) -> "FitterConfigBuilder":
        """Configure parallel partitions.

        Args:
            n: Number of partitions (None = auto-determine from backend).

        Returns:
            Self for method chaining.
        """
        self._num_partitions = n
        return self

    def build(self) -> FitterConfig:
        """Build the immutable FitterConfig.

        Returns:
            Configured FitterConfig instance.
        """
        return FitterConfig(
            bins=self._bins,
            use_rice_rule=self._use_rice_rule,
            support_at_zero=self._support_at_zero,
            max_distributions=self._max_distributions,
            prefilter=self._prefilter,
            enable_sampling=self._enable_sampling,
            sample_fraction=self._sample_fraction,
            max_sample_size=self._max_sample_size,
            sample_threshold=self._sample_threshold,
            bounded=self._bounded,
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound,
            num_partitions=self._num_partitions,
            lazy_metrics=self._lazy_metrics,
        )
