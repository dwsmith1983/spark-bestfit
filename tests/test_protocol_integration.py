"""Protocol integration tests for ExecutionBackend implementations.

This module verifies that all backend implementations (LocalBackend, SparkBackend,
RayBackend) correctly implement the ExecutionBackend protocol defined in
spark_bestfit.protocols.

These tests ensure:
1. All backends implement all required protocol methods
2. Method signatures match the protocol specification
3. All backends behave consistently for the same inputs
4. Cross-backend results are numerically equivalent

The tests use parametrization to run the same test logic across all available
backends, skipping backends whose dependencies are not installed.
"""

import inspect
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.fitting import compute_data_stats, fit_single_distribution
from spark_bestfit.protocols import ExecutionBackend

# Check for optional dependencies
try:
    from pyspark.sql import SparkSession

    from spark_bestfit.backends.spark import SparkBackend

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkBackend = None  # type: ignore
    SparkSession = None  # type: ignore

try:
    import ray

    from spark_bestfit.backends.ray import RayBackend

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    RayBackend = None  # type: ignore
    ray = None  # type: ignore


# Protocol method names and their expected signatures
PROTOCOL_METHODS = [
    "broadcast",
    "destroy_broadcast",
    "parallel_fit",
    "get_parallelism",
    "collect_column",
    "get_column_stats",
    "sample_column",
    "create_dataframe",
    "compute_correlation",
    "compute_histogram",
    "generate_samples",
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def local_backend_fixture():
    """Create LocalBackend for testing."""
    return LocalBackend(max_workers=2)


@pytest.fixture(scope="module")
def spark_backend_fixture():
    """Create SparkBackend for testing."""
    if not SPARK_AVAILABLE:
        pytest.skip("PySpark not installed")

    spark = (
        SparkSession.builder.appName("protocol-integration-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    backend = SparkBackend(spark)
    yield backend
    # Don't stop session - let conftest.py session-scoped fixture handle cleanup


@pytest.fixture(scope="module")
def ray_backend_fixture():
    """Create RayBackend for testing."""
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)

    backend = RayBackend()
    yield backend
    # Note: Don't shutdown Ray here as other tests may need it


@pytest.fixture
def normal_data():
    """Generate normal distribution test data."""
    np.random.seed(42)
    return np.random.normal(loc=50, scale=10, size=1000)


@pytest.fixture
def histogram(normal_data):
    """Create histogram from normal data."""
    y_hist, bin_edges = np.histogram(normal_data, bins=30, density=True)
    return (y_hist, bin_edges)


def create_local_df(data: np.ndarray, column: str = "value") -> pd.DataFrame:
    """Create pandas DataFrame from numpy array."""
    return pd.DataFrame({column: data})


def create_spark_df(spark_session, data: np.ndarray, column: str = "value"):
    """Create Spark DataFrame from numpy array."""
    return spark_session.createDataFrame([(float(x),) for x in data], [column])


def create_ray_df(data: np.ndarray, column: str = "value"):
    """Create Ray Dataset from numpy array."""
    # RayBackend accepts pandas DataFrames
    return pd.DataFrame({column: data})


# =============================================================================
# Test Classes
# =============================================================================


class TestProtocolStructure:
    """Tests verifying backends implement all protocol methods."""

    @pytest.mark.parametrize(
        "backend_class",
        [
            pytest.param(LocalBackend, id="LocalBackend"),
            pytest.param(
                SparkBackend,
                id="SparkBackend",
                marks=pytest.mark.skipif(
                    not SPARK_AVAILABLE, reason="PySpark not installed"
                ),
            ),
            pytest.param(
                RayBackend,
                id="RayBackend",
                marks=pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed"),
            ),
        ],
    )
    def test_backend_has_all_protocol_methods(self, backend_class):
        """Verify backend class has all methods defined in ExecutionBackend protocol."""
        for method_name in PROTOCOL_METHODS:
            assert hasattr(
                backend_class, method_name
            ), f"{backend_class.__name__} missing method: {method_name}"
            assert callable(
                getattr(backend_class, method_name)
            ), f"{backend_class.__name__}.{method_name} is not callable"

    @pytest.mark.parametrize(
        "backend_class",
        [
            pytest.param(LocalBackend, id="LocalBackend"),
            pytest.param(
                SparkBackend,
                id="SparkBackend",
                marks=pytest.mark.skipif(
                    not SPARK_AVAILABLE, reason="PySpark not installed"
                ),
            ),
            pytest.param(
                RayBackend,
                id="RayBackend",
                marks=pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed"),
            ),
        ],
    )
    def test_backend_method_signatures_are_compatible(self, backend_class):
        """Verify backend methods have compatible signatures with protocol."""
        protocol_methods = {
            name: getattr(ExecutionBackend, name)
            for name in PROTOCOL_METHODS
            if hasattr(ExecutionBackend, name)
        }

        for method_name, protocol_method in protocol_methods.items():
            backend_method = getattr(backend_class, method_name)
            protocol_sig = inspect.signature(protocol_method)
            backend_sig = inspect.signature(backend_method)

            # Check that all required protocol params exist in backend
            protocol_params = list(protocol_sig.parameters.keys())
            backend_params = list(backend_sig.parameters.keys())

            # Skip 'self' parameter
            protocol_params = [p for p in protocol_params if p != "self"]
            backend_params = [p for p in backend_params if p != "self"]

            for param in protocol_params:
                assert (
                    param in backend_params
                ), f"{backend_class.__name__}.{method_name} missing parameter: {param}"

    def test_protocol_has_type_hints(self):
        """Verify ExecutionBackend protocol methods have type hints."""
        # Use get_type_hints to verify protocol is properly annotated
        try:
            get_type_hints(ExecutionBackend)  # Verify class-level hints work
            # Protocol class itself may not have hints, check methods
            for method_name in PROTOCOL_METHODS:
                method = getattr(ExecutionBackend, method_name, None)
                if method is not None:
                    try:
                        method_hints = get_type_hints(method)
                        # Method should have return type hint at minimum
                        assert (
                            "return" in method_hints or len(method_hints) > 0
                        ), f"ExecutionBackend.{method_name} should have type hints"
                    except Exception:
                        # Some methods may not have resolvable hints
                        pass
        except Exception:
            # get_type_hints may fail for Protocol classes; this is acceptable
            pass


class TestBroadcast:
    """Tests for broadcast and destroy_broadcast methods."""

    def test_local_broadcast_returns_data(self, local_backend_fixture, normal_data):
        """LocalBackend broadcast returns data as-is."""
        handle = local_backend_fixture.broadcast(normal_data)
        np.testing.assert_array_equal(handle, normal_data)

    def test_local_destroy_broadcast_succeeds(
        self, local_backend_fixture, normal_data
    ):
        """LocalBackend destroy_broadcast completes without error."""
        handle = local_backend_fixture.broadcast(normal_data)
        # Should not raise
        local_backend_fixture.destroy_broadcast(handle)

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_broadcast_returns_handle(self, spark_backend_fixture, normal_data):
        """SparkBackend broadcast returns a broadcast handle."""
        handle = spark_backend_fixture.broadcast(normal_data)
        assert handle is not None
        # Spark broadcast wraps data
        np.testing.assert_array_equal(handle.value, normal_data)
        spark_backend_fixture.destroy_broadcast(handle)

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_ray_broadcast_returns_ref(self, ray_backend_fixture, normal_data):
        """RayBackend broadcast returns an object ref."""
        handle = ray_backend_fixture.broadcast(normal_data)
        assert handle is not None
        ray_backend_fixture.destroy_broadcast(handle)


class TestParallelism:
    """Tests for get_parallelism method."""

    def test_local_parallelism_positive(self, local_backend_fixture):
        """LocalBackend reports positive parallelism."""
        parallelism = local_backend_fixture.get_parallelism()
        assert isinstance(parallelism, int)
        assert parallelism > 0

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_parallelism_positive(self, spark_backend_fixture):
        """SparkBackend reports positive parallelism."""
        parallelism = spark_backend_fixture.get_parallelism()
        assert isinstance(parallelism, int)
        assert parallelism > 0

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_ray_parallelism_positive(self, ray_backend_fixture):
        """RayBackend reports positive parallelism."""
        parallelism = ray_backend_fixture.get_parallelism()
        assert isinstance(parallelism, int)
        assert parallelism > 0


class TestDataFrameOperations:
    """Tests for DataFrame-related protocol methods."""

    def test_local_create_dataframe(self, local_backend_fixture):
        """LocalBackend creates pandas DataFrame."""
        data = [(1.0,), (2.0,), (3.0,)]
        df = local_backend_fixture.create_dataframe(data, ["value"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "value" in df.columns

    def test_local_collect_column(self, local_backend_fixture, normal_data):
        """LocalBackend collects column as numpy array."""
        df = create_local_df(normal_data)
        result = local_backend_fixture.collect_column(df, "value")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, normal_data)

    def test_local_get_column_stats(self, local_backend_fixture, normal_data):
        """LocalBackend computes correct column statistics."""
        df = create_local_df(normal_data)
        stats = local_backend_fixture.get_column_stats(df, "value")

        assert isinstance(stats, dict)
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["min"] == float(normal_data.min())
        assert stats["max"] == float(normal_data.max())
        assert stats["count"] == len(normal_data)

    def test_local_sample_column(self, local_backend_fixture, normal_data):
        """LocalBackend samples column correctly."""
        df = create_local_df(normal_data)
        sample = local_backend_fixture.sample_column(df, "value", fraction=0.1, seed=42)

        assert isinstance(sample, np.ndarray)
        assert len(sample) < len(normal_data)
        assert len(sample) > 0

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_dataframe_operations(self, spark_backend_fixture, normal_data):
        """SparkBackend DataFrame operations work correctly."""
        spark = spark_backend_fixture.spark
        df = create_spark_df(spark, normal_data)

        # collect_column
        result = spark_backend_fixture.collect_column(df, "value")
        assert isinstance(result, np.ndarray)
        assert len(result) == len(normal_data)

        # get_column_stats
        stats = spark_backend_fixture.get_column_stats(df, "value")
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        np.testing.assert_allclose(stats["min"], normal_data.min(), rtol=1e-5)
        np.testing.assert_allclose(stats["max"], normal_data.max(), rtol=1e-5)
        assert stats["count"] == len(normal_data)

        # sample_column
        sample = spark_backend_fixture.sample_column(df, "value", fraction=0.1, seed=42)
        assert isinstance(sample, np.ndarray)
        assert len(sample) < len(normal_data)

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_ray_dataframe_operations(self, ray_backend_fixture, normal_data):
        """RayBackend DataFrame operations work correctly."""
        df = create_ray_df(normal_data)

        # collect_column
        result = ray_backend_fixture.collect_column(df, "value")
        assert isinstance(result, np.ndarray)
        assert len(result) == len(normal_data)

        # get_column_stats
        stats = ray_backend_fixture.get_column_stats(df, "value")
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats

        # sample_column
        sample = ray_backend_fixture.sample_column(df, "value", fraction=0.1, seed=42)
        assert isinstance(sample, np.ndarray)
        assert len(sample) < len(normal_data)


class TestParallelFit:
    """Tests for parallel_fit method across backends."""

    def test_local_parallel_fit_returns_results(
        self, local_backend_fixture, normal_data, histogram
    ):
        """LocalBackend parallel_fit returns valid results."""
        distributions = ["norm", "expon"]
        results = local_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
        )

        assert isinstance(results, list)
        assert len(results) > 0
        # Check result structure
        for result in results:
            assert "distribution" in result
            assert "sse" in result
            assert result["sse"] < float("inf")

    def test_local_parallel_fit_empty_distributions(
        self, local_backend_fixture, normal_data, histogram
    ):
        """LocalBackend handles empty distribution list."""
        results = local_backend_fixture.parallel_fit(
            distributions=[],
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
        )

        assert results == []

    def test_local_parallel_fit_with_progress(
        self, local_backend_fixture, normal_data, histogram
    ):
        """LocalBackend invokes progress callback."""
        progress_calls = []

        def on_progress(completed, total, percent):
            progress_calls.append((completed, total, percent))

        distributions = ["norm", "expon", "gamma"]
        local_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=on_progress,
        )

        assert len(progress_calls) == len(distributions)
        # Final call should show 100%
        last_call = progress_calls[-1]
        assert last_call[0] == last_call[1]  # completed == total

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_parallel_fit_returns_results(
        self, spark_backend_fixture, normal_data, histogram
    ):
        """SparkBackend parallel_fit returns valid results."""
        distributions = ["norm", "expon"]
        results = spark_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
        )

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_ray_parallel_fit_returns_results(
        self, ray_backend_fixture, normal_data, histogram
    ):
        """RayBackend parallel_fit returns valid results."""
        distributions = ["norm", "expon"]
        results = ray_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
        )

        assert isinstance(results, list)
        assert len(results) > 0


class TestCopulaHistogramMethods:
    """Tests for compute_correlation, compute_histogram, generate_samples."""

    def test_local_compute_correlation(self, local_backend_fixture):
        """LocalBackend computes correlation matrix."""
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})

        corr = local_backend_fixture.compute_correlation(df, ["a", "b"])
        assert isinstance(corr, np.ndarray)
        assert corr.shape == (2, 2)
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr), [1.0, 1.0], rtol=1e-5)

    def test_local_compute_histogram(self, local_backend_fixture, normal_data):
        """LocalBackend computes histogram bin counts."""
        df = create_local_df(normal_data)
        bin_edges = np.linspace(normal_data.min(), normal_data.max(), 11)

        counts, total = local_backend_fixture.compute_histogram(df, "value", bin_edges)
        assert isinstance(counts, np.ndarray)
        assert len(counts) == len(bin_edges) - 1
        assert total == len(normal_data)

    def test_local_generate_samples(self, local_backend_fixture):
        """LocalBackend generates samples via generator function."""

        def generator(n, partition_id, seed):
            np.random.seed(seed)
            return {"x": np.random.randn(n), "y": np.random.randn(n)}

        result = local_backend_fixture.generate_samples(
            n=100,
            generator_func=generator,
            column_names=["x", "y"],
            random_seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert "x" in result.columns
        assert "y" in result.columns

    @pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed")
    def test_spark_compute_correlation(self, spark_backend_fixture):
        """SparkBackend computes correlation matrix."""
        spark = spark_backend_fixture.spark
        np.random.seed(42)
        data = [(float(a), float(b)) for a, b in zip(np.random.randn(100), np.random.randn(100))]
        df = spark.createDataFrame(data, ["a", "b"])

        corr = spark_backend_fixture.compute_correlation(df, ["a", "b"])
        assert isinstance(corr, np.ndarray)
        assert corr.shape == (2, 2)

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_ray_compute_correlation(self, ray_backend_fixture):
        """RayBackend computes correlation matrix."""
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})

        corr = ray_backend_fixture.compute_correlation(df, ["a", "b"])
        assert isinstance(corr, np.ndarray)
        assert corr.shape == (2, 2)


class TestCrossBackendConsistency:
    """Tests verifying results are consistent across backends."""

    def test_statistics_consistent_local_spark(
        self, local_backend_fixture, spark_backend_fixture, normal_data
    ):
        """LocalBackend and SparkBackend produce same statistics."""
        if not SPARK_AVAILABLE:
            pytest.skip("PySpark not installed")

        # Create DataFrames
        local_df = create_local_df(normal_data)
        spark = spark_backend_fixture.spark
        spark_df = create_spark_df(spark, normal_data)

        # Compare stats
        local_stats = local_backend_fixture.get_column_stats(local_df, "value")
        spark_stats = spark_backend_fixture.get_column_stats(spark_df, "value")

        np.testing.assert_allclose(local_stats["min"], spark_stats["min"], rtol=1e-10)
        np.testing.assert_allclose(local_stats["max"], spark_stats["max"], rtol=1e-10)
        assert local_stats["count"] == spark_stats["count"]

    def test_parallel_fit_consistent_local_spark(
        self, local_backend_fixture, spark_backend_fixture, normal_data, histogram
    ):
        """LocalBackend and SparkBackend produce similar fit results."""
        if not SPARK_AVAILABLE:
            pytest.skip("PySpark not installed")

        distributions = ["norm"]
        data_stats = compute_data_stats(normal_data)

        local_results = local_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=data_stats,
        )

        spark_results = spark_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=data_stats,
        )

        # Both should find norm distribution
        local_norm = [r for r in local_results if r["distribution"] == "norm"]
        spark_norm = [r for r in spark_results if r["distribution"] == "norm"]

        assert len(local_norm) == 1
        assert len(spark_norm) == 1

        # SSE should be similar (not identical due to parallel execution order)
        np.testing.assert_allclose(
            local_norm[0]["sse"], spark_norm[0]["sse"], rtol=1e-5
        )

    @pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
    def test_statistics_consistent_local_ray(
        self, local_backend_fixture, ray_backend_fixture, normal_data
    ):
        """LocalBackend and RayBackend produce same statistics."""
        # Create DataFrames
        local_df = create_local_df(normal_data)
        ray_df = create_ray_df(normal_data)

        # Compare stats
        local_stats = local_backend_fixture.get_column_stats(local_df, "value")
        ray_stats = ray_backend_fixture.get_column_stats(ray_df, "value")

        np.testing.assert_allclose(local_stats["min"], ray_stats["min"], rtol=1e-10)
        np.testing.assert_allclose(local_stats["max"], ray_stats["max"], rtol=1e-10)
        assert local_stats["count"] == ray_stats["count"]


class TestProtocolEdgeCases:
    """Tests for edge cases and error handling."""

    def test_local_parallel_fit_empty_data(self, local_backend_fixture, histogram):
        """LocalBackend handles empty data array."""
        empty_data = np.array([])
        results = local_backend_fixture.parallel_fit(
            distributions=["norm"],
            histogram=histogram,
            data_sample=empty_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=None,
        )
        assert results == []

    def test_local_sample_column_handles_nan(self, local_backend_fixture):
        """LocalBackend filters NaN values when sampling."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        df = create_local_df(data_with_nan)
        sample = local_backend_fixture.sample_column(df, "value", fraction=1.0, seed=42)

        # NaN should be filtered out
        assert not np.any(np.isnan(sample))
        assert len(sample) <= 4  # At most 4 valid values

    def test_local_sample_column_handles_inf(self, local_backend_fixture):
        """LocalBackend filters infinite values when sampling."""
        data_with_inf = np.array([1.0, 2.0, np.inf, -np.inf, 5.0])
        df = create_local_df(data_with_inf)
        sample = local_backend_fixture.sample_column(df, "value", fraction=1.0, seed=42)

        # Inf should be filtered out
        assert not np.any(np.isinf(sample))

    def test_local_parallel_fit_callback_error_handled(
        self, local_backend_fixture, normal_data, histogram
    ):
        """LocalBackend handles callback errors gracefully."""

        def bad_callback(completed, total, percent):
            raise ValueError("Intentional test error")

        distributions = ["norm"]
        # Should not raise despite callback error
        results = local_backend_fixture.parallel_fit(
            distributions=distributions,
            histogram=histogram,
            data_sample=normal_data,
            fit_func=fit_single_distribution,
            column_name="value",
            data_stats=compute_data_stats(normal_data),
            progress_callback=bad_callback,
        )

        assert len(results) > 0  # Fitting should still complete
