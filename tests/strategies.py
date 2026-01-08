"""Hypothesis strategies for spark-bestfit property-based testing.

This module provides reusable hypothesis strategies for generating:
- Valid scipy distribution names
- Valid distribution parameters
- Finite numeric data samples
- Valid metric names
- FitterConfig and FitterConfigBuilder configurations

These strategies enable comprehensive property-based testing of
distribution fitting, serialization, and result handling.
"""

from typing import List, Tuple

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

# =============================================================================
# Distribution Name Strategies
# =============================================================================

# Continuous distributions that are stable for testing
# Excludes distributions that require special parameter constraints or are numerically unstable
STABLE_CONTINUOUS_DISTRIBUTIONS: List[str] = [
    "norm",
    "expon",
    "uniform",
    "gamma",
    "beta",
    "lognorm",
    "weibull_min",
    "pareto",
    "chi2",
    "t",
    "f",
    "logistic",
    "gumbel_r",
    "gumbel_l",
    "laplace",
    "rayleigh",
]

# Discrete distributions that are stable for testing
STABLE_DISCRETE_DISTRIBUTIONS: List[str] = [
    "poisson",
    "binom",
    "nbinom",
    "geom",
    "hypergeom",
    "randint",
]


@st.composite
def scipy_continuous_distribution(draw: st.DrawFn) -> str:
    """Generate a valid scipy continuous distribution name."""
    return draw(st.sampled_from(STABLE_CONTINUOUS_DISTRIBUTIONS))


@st.composite
def scipy_discrete_distribution(draw: st.DrawFn) -> str:
    """Generate a valid scipy discrete distribution name."""
    return draw(st.sampled_from(STABLE_DISCRETE_DISTRIBUTIONS))


# =============================================================================
# Distribution Parameter Strategies
# =============================================================================

# Parameter specifications for each distribution
# Format: (dist_name, param_generators) where param_generators creates valid params
# Note: allow_subnormal=False prevents extreme values like 1e-313 that cause scipy overflow
CONTINUOUS_PARAM_SPECS = {
    "norm": lambda draw: [
        draw(st.floats(min_value=-1000, max_value=1000, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "expon": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "uniform": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "gamma": lambda draw: [
        draw(st.floats(min_value=0.5, max_value=20, allow_subnormal=False)),  # a (shape)
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "beta": lambda draw: [
        draw(st.floats(min_value=0.5, max_value=10, allow_subnormal=False)),  # a
        draw(st.floats(min_value=0.5, max_value=10, allow_subnormal=False)),  # b
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "lognorm": lambda draw: [
        draw(st.floats(min_value=0.1, max_value=3, allow_subnormal=False)),  # s (shape)
        draw(st.floats(min_value=-10, max_value=10, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "weibull_min": lambda draw: [
        draw(st.floats(min_value=0.5, max_value=5, allow_subnormal=False)),  # c (shape)
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "pareto": lambda draw: [
        draw(st.floats(min_value=1.1, max_value=10, allow_subnormal=False)),  # b (shape)
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "chi2": lambda draw: [
        draw(st.floats(min_value=1, max_value=50, allow_subnormal=False)),  # df
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "t": lambda draw: [
        draw(st.floats(min_value=1, max_value=100, allow_subnormal=False)),  # df
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "f": lambda draw: [
        draw(st.floats(min_value=1, max_value=50, allow_subnormal=False)),  # dfn
        draw(st.floats(min_value=1, max_value=50, allow_subnormal=False)),  # dfd
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "logistic": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "gumbel_r": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "gumbel_l": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "laplace": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
    "rayleigh": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),  # loc
        draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),  # scale
    ],
}


@st.composite
def distribution_with_params(draw: st.DrawFn) -> Tuple[str, List[float]]:
    """Generate a distribution name with valid parameters.

    Returns:
        Tuple of (distribution_name, parameters_list)
    """
    dist_name = draw(scipy_continuous_distribution())
    param_generator = CONTINUOUS_PARAM_SPECS.get(dist_name)

    if param_generator:
        params = param_generator(draw)
    else:
        # Default: loc, scale (allow_subnormal=False prevents scipy overflow)
        params = [
            draw(st.floats(min_value=-100, max_value=100, allow_subnormal=False)),
            draw(st.floats(min_value=0.01, max_value=100, allow_subnormal=False)),
        ]

    return (dist_name, params)


# =============================================================================
# Data Sample Strategies
# =============================================================================


@st.composite
def finite_float_array(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 1000,
    min_value: float = -1e6,
    max_value: float = 1e6,
) -> np.ndarray:
    """Generate an array of finite floats (no inf/nan).

    Args:
        draw: Hypothesis draw function
        min_size: Minimum array length
        max_size: Maximum array length
        min_value: Minimum float value
        max_value: Maximum float value

    Returns:
        numpy array of finite floats
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    data = draw(
        st.lists(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(data)


@st.composite
def positive_float_array(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 1000,
    min_value: float = 1e-6,
    max_value: float = 1e6,
) -> np.ndarray:
    """Generate an array of positive finite floats.

    Useful for testing distributions that require positive support.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    data = draw(
        st.lists(
            st.floats(
                min_value=min_value,
                max_value=max_value,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(data)


@st.composite
def integer_count_array(
    draw: st.DrawFn,
    min_size: int = 10,
    max_size: int = 1000,
    min_value: int = 0,
    max_value: int = 100,
) -> np.ndarray:
    """Generate an array of non-negative integers (count data).

    Useful for testing discrete distributions.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    data = draw(
        st.lists(
            st.integers(min_value=min_value, max_value=max_value),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(data)


# =============================================================================
# Metric and Result Strategies
# =============================================================================

# Valid metric names for FitResults.best()
VALID_METRIC_NAMES = ["sse", "aic", "bic", "ks_statistic", "ad_statistic"]


@st.composite
def metric_name(draw: st.DrawFn) -> str:
    """Generate a valid metric name for sorting fit results."""
    return draw(st.sampled_from(VALID_METRIC_NAMES))


@st.composite
def probability(draw: st.DrawFn) -> float:
    """Generate a probability value in (0, 1) exclusive.

    Useful for testing PPF (inverse CDF) which is undefined at 0 and 1.
    Uses 1e-4 to 1-1e-4 to avoid scipy numerical precision issues at extreme tails.
    """
    p = draw(st.floats(min_value=1e-4, max_value=1 - 1e-4))
    assume(0 < p < 1)
    return p


@st.composite
def probabilities(draw: st.DrawFn, min_size: int = 1, max_size: int = 100) -> np.ndarray:
    """Generate an array of probability values in (0, 1).

    Uses 1e-4 to 1-1e-4 to avoid scipy numerical precision issues at extreme tails.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    probs = draw(
        st.lists(
            st.floats(min_value=1e-4, max_value=1 - 1e-4),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(probs)


# =============================================================================
# DistributionFitResult Strategy
# =============================================================================


@st.composite
def distribution_fit_result_data(draw: st.DrawFn) -> dict:
    """Generate valid data for constructing a DistributionFitResult.

    Returns a dict suitable for passing as kwargs to DistributionFitResult().
    """
    dist_name, params = draw(distribution_with_params())

    return {
        "distribution": dist_name,
        "parameters": params,
        "sse": draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        "aic": draw(st.floats(min_value=-1000, max_value=10000, allow_nan=False) | st.none()),
        "bic": draw(st.floats(min_value=-1000, max_value=10000, allow_nan=False) | st.none()),
        "ks_statistic": draw(st.floats(min_value=0, max_value=1, allow_nan=False) | st.none()),
        "pvalue": draw(st.floats(min_value=0, max_value=1, allow_nan=False) | st.none()),
        "ad_statistic": draw(st.floats(min_value=0, max_value=100, allow_nan=False) | st.none()),
        "ad_pvalue": draw(st.floats(min_value=0, max_value=1, allow_nan=False) | st.none()),
        "data_min": draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False) | st.none()),
        "data_max": draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False) | st.none()),
        "data_mean": draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False) | st.none()),
        "data_stddev": draw(st.floats(min_value=0, max_value=1e6, allow_nan=False) | st.none()),
        "data_count": draw(st.floats(min_value=1, max_value=1e9, allow_nan=False) | st.none()),
        "column_name": draw(st.text(min_size=0, max_size=50) | st.none()),
    }


# =============================================================================
# FitterConfig Strategies
# =============================================================================

# Valid prefilter modes
VALID_PREFILTER_MODES = [False, True, "aggressive"]


@st.composite
def fitter_config_data(draw: st.DrawFn) -> dict:
    """Generate valid data for constructing a FitterConfig.

    Returns a dict suitable for passing as kwargs to FitterConfig().
    Excludes progress_callback as it's not serializable/comparable.
    """
    # Generate bounds that make sense together
    lower = draw(st.floats(min_value=-1000, max_value=500, allow_nan=False) | st.none())
    upper = draw(st.floats(min_value=-500, max_value=1000, allow_nan=False) | st.none())

    # Ensure lower <= upper if both are set
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower

    # Bounded should be True if bounds are set
    bounded = draw(st.booleans()) if lower is None and upper is None else True

    return {
        "bins": draw(st.integers(min_value=5, max_value=500) | st.just(50)),
        "use_rice_rule": draw(st.booleans()),
        "support_at_zero": draw(st.booleans()),
        "max_distributions": draw(st.integers(min_value=1, max_value=100) | st.none()),
        "prefilter": draw(st.sampled_from(VALID_PREFILTER_MODES)),
        "enable_sampling": draw(st.booleans()),
        "sample_fraction": draw(
            st.floats(min_value=0.001, max_value=1.0, allow_nan=False) | st.none()
        ),
        "max_sample_size": draw(st.integers(min_value=100, max_value=10_000_000)),
        "sample_threshold": draw(st.integers(min_value=1000, max_value=100_000_000)),
        "bounded": bounded,
        "lower_bound": lower,
        "upper_bound": upper,
        "num_partitions": draw(st.integers(min_value=1, max_value=100) | st.none()),
        "lazy_metrics": draw(st.booleans()),
    }


@st.composite
def fitter_config(draw: st.DrawFn):
    """Generate a valid FitterConfig instance.

    Returns a FitterConfig with random valid parameters.
    """
    from spark_bestfit.config import FitterConfig

    data = draw(fitter_config_data())
    return FitterConfig(**data)


@st.composite
def builder_method_sequence(draw: st.DrawFn) -> List[Tuple[str, dict]]:
    """Generate a sequence of builder method calls.

    Returns a list of (method_name, kwargs) tuples representing
    a valid sequence of FitterConfigBuilder method calls.
    """
    methods = []

    # Randomly include each builder method
    if draw(st.booleans()):
        methods.append(
            (
                "with_bins",
                {
                    "bins": draw(st.integers(min_value=10, max_value=200)),
                    "use_rice_rule": draw(st.booleans()),
                },
            )
        )

    if draw(st.booleans()):
        lower = draw(st.floats(min_value=0, max_value=50, allow_nan=False) | st.none())
        upper = draw(st.floats(min_value=50, max_value=100, allow_nan=False) | st.none())
        methods.append(
            (
                "with_bounds",
                {
                    "lower": lower,
                    "upper": upper,
                    "auto_detect": draw(st.booleans()) if lower is None and upper is None else False,
                },
            )
        )

    if draw(st.booleans()):
        methods.append(
            (
                "with_sampling",
                {
                    "fraction": draw(
                        st.floats(min_value=0.01, max_value=1.0, allow_nan=False) | st.none()
                    ),
                    "max_size": draw(st.integers(min_value=1000, max_value=1_000_000)),
                    "threshold": draw(st.integers(min_value=10000, max_value=10_000_000)),
                    "enabled": draw(st.booleans()),
                },
            )
        )

    if draw(st.booleans()):
        methods.append(("with_lazy_metrics", {"lazy": draw(st.booleans())}))

    if draw(st.booleans()):
        methods.append(
            ("with_prefilter", {"mode": draw(st.sampled_from(VALID_PREFILTER_MODES))})
        )

    if draw(st.booleans()):
        methods.append(("with_support_at_zero", {"enabled": draw(st.booleans())}))

    if draw(st.booleans()):
        methods.append(
            (
                "with_max_distributions",
                {"n": draw(st.integers(min_value=1, max_value=50) | st.none())},
            )
        )

    if draw(st.booleans()):
        methods.append(
            (
                "with_partitions",
                {"n": draw(st.integers(min_value=1, max_value=32) | st.none())},
            )
        )

    return methods
