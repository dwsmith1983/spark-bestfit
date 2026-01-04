"""Serialization utilities for distribution fit results.

This module provides functionality to save and load fitted distributions
to JSON and pickle formats, enabling model persistence and sharing.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import scipy.stats as st

from spark_bestfit._version import __version__

if TYPE_CHECKING:
    from spark_bestfit.results import DistributionFitResult

# Schema version for forward compatibility
# 1.0: Initial release
# 1.1: Added lower_bound and upper_bound for truncated distributions (v1.4.0)
SCHEMA_VERSION = "1.1"


class SerializationError(Exception):
    """Error during serialization or deserialization of fit results.

    Raised when:
    - File format cannot be determined from extension
    - Required fields are missing in serialized data
    - Distribution name is unknown to scipy.stats
    - JSON parsing fails
    """

    pass


def serialize_to_dict(result: "DistributionFitResult") -> Dict[str, Any]:
    """Convert DistributionFitResult to serializable dictionary with metadata.

    Adds version information and timestamp for forward compatibility tracking.

    Args:
        result: The fit result to serialize

    Returns:
        Dictionary ready for JSON serialization

    Example:
        >>> from spark_bestfit.results import DistributionFitResult
        >>> result = DistributionFitResult(distribution="norm", parameters=[0.0, 1.0], sse=0.01)
        >>> data = serialize_to_dict(result)
        >>> data["distribution"]
        'norm'
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "spark_bestfit_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "distribution": result.distribution,
        "parameters": result.parameters,
        "column_name": result.column_name,
        "metrics": {
            "sse": result.sse,
            "aic": result.aic,
            "bic": result.bic,
            "ks_statistic": result.ks_statistic,
            "pvalue": result.pvalue,
            "ad_statistic": result.ad_statistic,
            "ad_pvalue": result.ad_pvalue,
        },
        # Flat data stats (v2.0: replaced data_summary dict)
        "data_min": result.data_min,
        "data_max": result.data_max,
        "data_mean": result.data_mean,
        "data_stddev": result.data_stddev,
        "data_count": result.data_count,
        # Bounds for truncated distribution fitting (v1.4.0)
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
    }


def deserialize_from_dict(data: Dict[str, Any]) -> "DistributionFitResult":
    """Reconstruct DistributionFitResult from dictionary.

    Validates the data and creates a new result object that can be used
    for sampling, PDF/CDF evaluation, etc.

    Args:
        data: Dictionary from JSON or other source

    Returns:
        Reconstructed DistributionFitResult

    Raises:
        SerializationError: If required fields are missing or distribution is unknown

    Example:
        >>> data = {"distribution": "norm", "parameters": [0.0, 1.0], "metrics": {"sse": 0.01}}
        >>> result = deserialize_from_dict(data)
        >>> result.sample(5)
        array([...])
    """
    from spark_bestfit.results import DistributionFitResult

    validate_schema(data)

    # Extract metrics (supports both nested and flat formats for flexibility)
    metrics = data.get("metrics", {})

    # Handle flat format (legacy or manually created)
    if not metrics:
        metrics = {
            "sse": data.get("sse"),
            "aic": data.get("aic"),
            "bic": data.get("bic"),
            "ks_statistic": data.get("ks_statistic"),
            "pvalue": data.get("pvalue"),
            "ad_statistic": data.get("ad_statistic"),
            "ad_pvalue": data.get("ad_pvalue"),
        }

    return DistributionFitResult(
        distribution=data["distribution"],
        parameters=list(data["parameters"]),
        sse=metrics.get("sse", float("inf")),
        column_name=data.get("column_name"),
        aic=metrics.get("aic"),
        bic=metrics.get("bic"),
        ks_statistic=metrics.get("ks_statistic"),
        pvalue=metrics.get("pvalue"),
        ad_statistic=metrics.get("ad_statistic"),
        ad_pvalue=metrics.get("ad_pvalue"),
        data_min=data.get("data_min"),
        data_max=data.get("data_max"),
        data_mean=data.get("data_mean"),
        data_stddev=data.get("data_stddev"),
        data_count=data.get("data_count"),
        lower_bound=data.get("lower_bound"),
        upper_bound=data.get("upper_bound"),
    )


def validate_schema(data: Dict[str, Any]) -> None:
    """Validate serialized data has required fields and valid distribution.

    Args:
        data: Dictionary to validate

    Raises:
        SerializationError: If validation fails
    """
    # Check required fields
    if "distribution" not in data:
        raise SerializationError("Missing required field: 'distribution'")
    if "parameters" not in data:
        raise SerializationError("Missing required field: 'parameters'")

    # Validate distribution exists in scipy.stats
    dist_name = data["distribution"]
    if not hasattr(st, dist_name):
        raise SerializationError(
            f"Unknown distribution: '{dist_name}'. " "Must be a valid scipy.stats distribution name."
        )

    # Validate parameters is a list
    if not isinstance(data["parameters"], (list, tuple)):
        raise SerializationError("'parameters' must be a list or tuple")


def detect_format(path: Union[str, Path]) -> Literal["json", "pickle"]:
    """Detect serialization format from file extension.

    Args:
        path: File path to check

    Returns:
        Format string: 'json' or 'pickle'

    Raises:
        SerializationError: If extension is not recognized
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return "json"
    elif suffix in (".pkl", ".pickle"):
        return "pickle"
    else:
        raise SerializationError(
            f"Cannot determine format from extension '{suffix}'. "
            "Use .json for JSON format or .pkl/.pickle for pickle format, "
            "or specify format explicitly."
        )


def save_json(data: Dict[str, Any], path: Path, indent: Optional[int] = 2) -> None:
    """Write dictionary to JSON file.

    Args:
        data: Dictionary to serialize
        path: Output file path
        indent: JSON indentation level (None for compact)
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Path) -> Dict[str, Any]:
    """Read dictionary from JSON file.

    Args:
        path: Input file path

    Returns:
        Parsed dictionary

    Raises:
        SerializationError: If JSON parsing fails
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SerializationError(f"Invalid JSON: {e}") from e


def save_pickle(result: "DistributionFitResult", path: Path) -> None:
    """Write result to pickle file.

    Args:
        result: DistributionFitResult to serialize
        path: Output file path
    """
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_pickle(path: Path) -> "DistributionFitResult":
    """Read result from pickle file.

    Warning:
        Only load pickle files from trusted sources.

    Args:
        path: Input file path

    Returns:
        Deserialized DistributionFitResult

    Raises:
        SerializationError: If file is corrupt or cannot be unpickled
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, TypeError, AttributeError) as e:
        raise SerializationError(f"Failed to load pickle file: {e}") from e
