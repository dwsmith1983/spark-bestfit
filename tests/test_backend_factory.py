"""Tests for BackendFactory."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from spark_bestfit.backends.factory import BackendFactory
from spark_bestfit.backends.local import LocalBackend

# PySpark is optional
try:
    from pyspark.sql import SparkSession  # noqa: F401

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

# Ray is optional
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    RAY_AVAILABLE = False


@pytest.fixture(autouse=True)
def shutdown_ray_after_test():
    """Shutdown Ray after each test to avoid interfering with Spark sessions.

    Ray and Spark can interfere with each other's JVM/process management.
    This fixture ensures Ray is shutdown after each test that might initialize it.
    """
    yield
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


class TestBackendFactoryForDataframe:
    """Tests for BackendFactory.for_dataframe() auto-detection."""

    def test_for_dataframe_pandas(self):
        """Pandas DataFrame returns LocalBackend."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        backend = BackendFactory.for_dataframe(df)
        assert isinstance(backend, LocalBackend)

    def test_for_dataframe_spark(self, spark_session):
        """Spark DataFrame returns SparkBackend."""
        from spark_bestfit.backends.spark import SparkBackend

        df = spark_session.createDataFrame([(1,), (2,), (3,)], ["a"])
        backend = BackendFactory.for_dataframe(df)
        assert isinstance(backend, SparkBackend)

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_for_dataframe_ray_duck_typing(self):
        """Object with Ray Dataset attributes returns RayBackend."""
        from spark_bestfit.backends.ray import RayBackend

        # Mock Ray Dataset with duck typing
        mock_ray_ds = MagicMock()
        mock_ray_ds.select_columns = MagicMock()
        mock_ray_ds.to_pandas = MagicMock()

        backend = BackendFactory.for_dataframe(mock_ray_ds)
        assert isinstance(backend, RayBackend)


class TestBackendFactoryCreate:
    """Tests for BackendFactory.create() explicit creation."""

    def test_create_local(self):
        """Create local backend with default settings."""
        backend = BackendFactory.create("local")
        assert isinstance(backend, LocalBackend)
        # Verify backend is functional (has expected method)
        assert hasattr(backend, "parallel_fit")
        # Verify default parallelism is set
        assert backend.get_parallelism() >= 1

    def test_create_local_with_max_workers(self):
        """Create local backend with max_workers option."""
        backend = BackendFactory.create("local", max_workers=2)
        assert isinstance(backend, LocalBackend)
        assert backend.max_workers == 2

    def test_create_spark(self, spark_session):
        """Create spark backend with session."""
        from spark_bestfit.backends.spark import SparkBackend

        backend = BackendFactory.create("spark", spark_session=spark_session)
        assert isinstance(backend, SparkBackend)

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_create_ray(self):
        """Create ray backend."""
        from spark_bestfit.backends.ray import RayBackend

        backend = BackendFactory.create("ray")
        assert isinstance(backend, RayBackend)

    def test_create_invalid_raises(self):
        """Invalid backend type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            BackendFactory.create("invalid")

    def test_create_invalid_error_message(self):
        """Error message includes valid options."""
        with pytest.raises(ValueError, match="'spark', 'local', 'ray'"):
            BackendFactory.create("not_a_backend")


class TestBackendFactoryAvailability:
    """Tests for BackendFactory.is_available() and get_available()."""

    def test_is_available_local(self):
        """Local is always available (no external dependencies)."""
        assert BackendFactory.is_available("local") is True

    def test_is_available_spark_matches_import(self):
        """is_available('spark') matches whether pyspark can be imported."""
        assert BackendFactory.is_available("spark") == PYSPARK_AVAILABLE

    def test_is_available_ray_matches_import(self):
        """is_available('ray') matches whether ray can be imported."""
        assert BackendFactory.is_available("ray") == RAY_AVAILABLE

    def test_get_available_includes_local(self):
        """Local is always in available list."""
        available = BackendFactory.get_available()
        assert "local" in available

    def test_get_available_returns_valid_backend_names(self):
        """get_available returns list of backend names that can be created."""
        available = BackendFactory.get_available()
        assert isinstance(available, list)
        assert all(isinstance(b, str) for b in available)
        # Each available backend should be creatable
        for backend_name in available:
            if backend_name == "local":
                backend = BackendFactory.create(backend_name)
                assert backend is not None

    @pytest.mark.skipif(
        not (PYSPARK_AVAILABLE and RAY_AVAILABLE),
        reason="Requires both PySpark and Ray installed",
    )
    def test_get_available_full_env(self):
        """In full environment with all deps, all backends available."""
        available = BackendFactory.get_available()
        assert "local" in available
        assert "spark" in available
        assert "ray" in available


class TestBackendFactoryImports:
    """Tests for BackendFactory export availability."""

    def test_import_from_backends(self):
        """Can import BackendFactory from backends package."""
        from spark_bestfit.backends import BackendFactory as BF

        assert BF is BackendFactory

    def test_import_from_main_package(self):
        """Can import BackendFactory from main package."""
        from spark_bestfit import BackendFactory as BF

        assert BF is BackendFactory
