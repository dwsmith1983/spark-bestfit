"""Backend factory for automatic and explicit backend selection.

This module provides a centralized factory for creating execution backends,
eliminating duplicate auto-detection logic across the codebase.

Example:
    >>> from spark_bestfit.backends.factory import BackendFactory
    >>>
    >>> # Auto-detect from DataFrame type
    >>> backend = BackendFactory.for_dataframe(df)
    >>>
    >>> # Explicit creation with options
    >>> backend = BackendFactory.create("local", max_workers=4)
    >>> backend = BackendFactory.create("spark", spark_session=spark)
"""

from typing import Any, Literal

from spark_bestfit.protocols import ExecutionBackend

BackendType = Literal["spark", "local", "ray"]


class BackendFactory:
    """Factory for creating execution backends.

    Provides centralized backend creation with:
    - Auto-detection from DataFrame type
    - Explicit string-based selection
    - Optional dependency handling

    Example:
        >>> # Auto-detect from DataFrame
        >>> backend = BackendFactory.for_dataframe(df)

        >>> # Explicit creation
        >>> backend = BackendFactory.create("local", max_workers=4)

        >>> # Check availability
        >>> if BackendFactory.is_available("spark"):
        ...     backend = BackendFactory.create("spark")
    """

    @classmethod
    def for_dataframe(cls, df: Any) -> ExecutionBackend:
        """Auto-detect and create backend based on DataFrame type.

        Detection order:
        1. Ray Dataset (duck typing: has select_columns and to_pandas)
        2. pandas DataFrame (isinstance check)
        3. Spark DataFrame (default fallback)

        Args:
            df: Input DataFrame (Spark, pandas, or Ray Dataset)

        Returns:
            Appropriate backend instance

        Raises:
            ImportError: If detected backend's dependencies not installed
        """
        import pandas as pd

        # Ray Dataset (duck typing - Ray Dataset has no common base class)
        if hasattr(df, "select_columns") and hasattr(df, "to_pandas"):
            return cls.create("ray")
        # pandas DataFrame
        elif isinstance(df, pd.DataFrame):
            return cls.create("local")
        # Spark DataFrame (default)
        else:
            return cls.create("spark")

    @classmethod
    def create(
        cls,
        backend_type: BackendType,
        **kwargs: Any,
    ) -> ExecutionBackend:
        """Create a specific backend by name.

        Args:
            backend_type: One of "spark", "local", "ray"
            **kwargs: Backend-specific arguments:
                - spark: spark_session (optional SparkSession)
                - local: max_workers (optional int)
                - ray: (no options currently)

        Returns:
            Backend instance

        Raises:
            ValueError: If backend_type is unknown
            ImportError: If required dependencies not installed
        """
        if backend_type == "spark":
            from spark_bestfit.backends.spark import SparkBackend

            return SparkBackend(kwargs.get("spark_session"))
        elif backend_type == "local":
            from spark_bestfit.backends.local import LocalBackend

            return LocalBackend(kwargs.get("max_workers"))
        elif backend_type == "ray":
            from spark_bestfit.backends.ray import RayBackend

            return RayBackend()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}. " f"Valid options: 'spark', 'local', 'ray'")

    @classmethod
    def is_available(cls, backend_type: BackendType) -> bool:
        """Check if a backend's dependencies are installed.

        Args:
            backend_type: Backend to check

        Returns:
            True if dependencies are available
        """
        if backend_type == "local":
            return True  # No external dependencies
        elif backend_type == "spark":
            try:
                import pyspark  # noqa: F401

                return True
            except ImportError:
                return False
        elif backend_type == "ray":
            try:
                import ray  # noqa: F401

                return True
            except ImportError:
                return False
        return False

    @classmethod
    def get_available(cls) -> list[BackendType]:
        """Get list of available backends.

        Returns:
            List of backend types with installed dependencies.
            Always includes "local" as it has no external deps.
        """
        available: list[BackendType] = ["local"]  # Always available
        if cls.is_available("spark"):
            available.append("spark")
        if cls.is_available("ray"):
            available.append("ray")
        return available
