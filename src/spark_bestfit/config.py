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
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union


class SamplingMode(Enum):
    """Sampling strategy mode for large datasets.

    Attributes:
        AUTO: Automatically select strategy based on data skewness (default).
            Uses uniform for symmetric data, stratified for skewed data.
        UNIFORM: Force uniform random sampling (backwards compatible).
        STRATIFIED: Force stratified sampling across percentile bins.
    """

    AUTO = "auto"
    UNIFORM = "uniform"
    STRATIFIED = "stratified"


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
        adaptive_sampling: Use adaptive sampling based on data skewness (v2.9.0).
        sampling_mode: Sampling strategy (AUTO, UNIFORM, or STRATIFIED).
        skew_threshold_mild: Skewness threshold for mild stratification (default 0.5).
        skew_threshold_high: Skewness threshold for aggressive stratification (default 2.0).
        bounded: Enable truncated distribution fitting.
        lower_bound: Lower bound for truncated fitting (scalar or per-column dict).
        upper_bound: Upper bound for truncated fitting (scalar or per-column dict).
        censoring_column: Column name containing censoring indicator for survival data.
            True/1 = observed event, False/0 = right-censored observation.
            When specified, uses censored MLE for parameter estimation.
        num_partitions: Number of parallel partitions (None = auto-determine).
        lazy_metrics: Defer KS/AD computation until accessed.
        progress_callback: Optional callback for progress updates.
        estimation_method: Parameter estimation method ("mle", "mse", or "auto").

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

    # === Adaptive Sampling (v2.9.0) ===
    adaptive_sampling: bool = True
    sampling_mode: SamplingMode = SamplingMode.AUTO
    skew_threshold_mild: float = 0.5
    skew_threshold_high: float = 2.0

    # === Bounds ===
    bounded: bool = False
    lower_bound: Optional[Union[float, Dict[str, float]]] = None
    upper_bound: Optional[Union[float, Dict[str, float]]] = None

    # === Censored Data (v2.9.0) ===
    censoring_column: Optional[str] = None  # Column with censoring indicator (True=observed, False=censored)

    # === Performance ===
    num_partitions: Optional[int] = None
    lazy_metrics: bool = False

    # === Estimation Method (v2.5.0) ===
    estimation_method: str = "mle"  # "mle", "mse", or "auto"

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

        # Adaptive sampling
        self._adaptive_sampling: bool = True
        self._sampling_mode: SamplingMode = SamplingMode.AUTO
        self._skew_threshold_mild: float = 0.5
        self._skew_threshold_high: float = 2.0

        # Bounds
        self._bounded: bool = False
        self._lower_bound: Optional[Union[float, Dict[str, float]]] = None
        self._upper_bound: Optional[Union[float, Dict[str, float]]] = None

        # Censored data
        self._censoring_column: Optional[str] = None

        # Performance
        self._num_partitions: Optional[int] = None
        self._lazy_metrics: bool = False

        # Estimation method
        self._estimation_method: str = "mle"

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

    def with_adaptive_sampling(
        self,
        enabled: bool = True,
        mode: SamplingMode = SamplingMode.AUTO,
        skew_threshold_mild: float = 0.5,
        skew_threshold_high: float = 2.0,
    ) -> "FitterConfigBuilder":
        """Configure adaptive sampling based on data skewness (v2.9.0).

        When enabled, sampling strategy is selected based on data skewness:
        - |skew| < mild_threshold: Uniform sampling (efficient for symmetric data)
        - mild_threshold <= |skew| < high_threshold: Stratified (5 bins)
        - |skew| >= high_threshold: Stratified (10 bins) with tail oversampling

        Args:
            enabled: Whether adaptive sampling is enabled.
            mode: Sampling strategy (AUTO, UNIFORM, or STRATIFIED).
            skew_threshold_mild: Skewness threshold for mild stratification.
            skew_threshold_high: Skewness threshold for aggressive stratification.

        Returns:
            Self for method chaining.

        Example:
            >>> config = (FitterConfigBuilder()
            ...     .with_adaptive_sampling(enabled=True, mode=SamplingMode.AUTO)
            ...     .build())
        """
        self._adaptive_sampling = enabled
        self._sampling_mode = mode
        self._skew_threshold_mild = skew_threshold_mild
        self._skew_threshold_high = skew_threshold_high
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

    def with_censoring(self, column: str) -> "FitterConfigBuilder":
        """Configure censored data fitting for survival analysis (v2.9.0).

        When fitting censored data, the fitter uses censored MLE which accounts
        for right-censored observations (events that haven't occurred yet).

        Common survival distributions: weibull_min, expon, lognorm, gamma.

        Args:
            column: Name of the column containing the censoring indicator.
                True/1 = observed event, False/0 = right-censored observation.

        Returns:
            Self for method chaining.

        Note:
            KS/AD statistics are skipped for censored fits as they assume
            complete data. Use AIC/BIC for model comparison instead.

        Example:
            >>> config = (FitterConfigBuilder()
            ...     .with_censoring("event_occurred")
            ...     .build())
            >>> results = fitter.fit(df, column="time_to_event", config=config)
        """
        self._censoring_column = column
        return self

    def with_estimation_method(self, method: str = "mle") -> "FitterConfigBuilder":
        """Configure parameter estimation method (v2.5.0).

        Args:
            method: Estimation method to use:
                - "mle": Maximum Likelihood Estimation (default). Uses scipy.stats.fit().
                    Fast and accurate for most distributions.
                - "mse": Maximum Spacing Estimation. More robust for heavy-tailed
                    distributions (Pareto, Cauchy, etc.) where MLE may fail.
                - "auto": Automatically select MSE for heavy-tailed data based on
                    kurtosis and extreme value analysis, MLE otherwise.

        Returns:
            Self for method chaining.

        Example:
            >>> config = (FitterConfigBuilder()
            ...     .with_estimation_method("mse")  # For heavy-tailed data
            ...     .build())
        """
        if method not in ("mle", "mse", "auto"):
            raise ValueError(f"estimation_method must be 'mle', 'mse', or 'auto', got '{method}'")
        self._estimation_method = method
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
            adaptive_sampling=self._adaptive_sampling,
            sampling_mode=self._sampling_mode,
            skew_threshold_mild=self._skew_threshold_mild,
            skew_threshold_high=self._skew_threshold_high,
            bounded=self._bounded,
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound,
            censoring_column=self._censoring_column,
            num_partitions=self._num_partitions,
            lazy_metrics=self._lazy_metrics,
            estimation_method=self._estimation_method,
        )
