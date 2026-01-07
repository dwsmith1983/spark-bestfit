"""Hypothesis strategies for spark-bestfit property-based testing.

This module provides reusable hypothesis strategies for generating:
- Valid scipy distribution names
- Valid distribution parameters
- Finite numeric data samples
- Valid metric names

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
CONTINUOUS_PARAM_SPECS = {
    "norm": lambda draw: [
        draw(st.floats(min_value=-1000, max_value=1000)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "expon": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "uniform": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "gamma": lambda draw: [
        draw(st.floats(min_value=0.5, max_value=20)),  # a (shape) - min 0.5 for PPF/CDF numerical stability
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "beta": lambda draw: [
        draw(st.floats(min_value=0.5, max_value=10)),  # a - min 0.5 for PPF/CDF numerical stability
        draw(st.floats(min_value=0.5, max_value=10)),  # b - min 0.5 for PPF/CDF numerical stability
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "lognorm": lambda draw: [
        draw(st.floats(min_value=0.1, max_value=3)),  # s (shape)
        draw(st.floats(min_value=-10, max_value=10)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "weibull_min": lambda draw: [
        draw(st.floats(min_value=0.5, max_value=5)),  # c (shape)
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "pareto": lambda draw: [
        draw(st.floats(min_value=1.1, max_value=10)),  # b (shape)
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "chi2": lambda draw: [
        draw(st.floats(min_value=1, max_value=50)),  # df
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "t": lambda draw: [
        draw(st.floats(min_value=1, max_value=100)),  # df
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "f": lambda draw: [
        draw(st.floats(min_value=1, max_value=50)),  # dfn
        draw(st.floats(min_value=1, max_value=50)),  # dfd
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "logistic": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "gumbel_r": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "gumbel_l": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "laplace": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
    ],
    "rayleigh": lambda draw: [
        draw(st.floats(min_value=-100, max_value=100)),  # loc
        draw(st.floats(min_value=0.01, max_value=100)),  # scale
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
        # Default: loc, scale
        params = [
            draw(st.floats(min_value=-100, max_value=100)),
            draw(st.floats(min_value=0.01, max_value=100)),
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
    """
    p = draw(st.floats(min_value=1e-6, max_value=1 - 1e-6))
    assume(0 < p < 1)
    return p


@st.composite
def probabilities(draw: st.DrawFn, min_size: int = 1, max_size: int = 100) -> np.ndarray:
    """Generate an array of probability values in (0, 1)."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    probs = draw(
        st.lists(
            st.floats(min_value=1e-6, max_value=1 - 1e-6),
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
