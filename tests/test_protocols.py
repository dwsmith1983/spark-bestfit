"""Tests for the ExecutionBackend protocol definition.

This module tests the Protocol-based type system for backend abstraction,
including @runtime_checkable isinstance() checks and conformance verification.
"""

import numpy as np
import pytest

from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.protocols import ExecutionBackend


class TestExecutionBackendProtocol:
    """Tests for ExecutionBackend protocol definition and type checking."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify ExecutionBackend is decorated with @runtime_checkable."""
        # Protocol should be importable and usable with isinstance()
        backend = LocalBackend()
        assert isinstance(backend, ExecutionBackend)

    def test_local_backend_satisfies_protocol(self) -> None:
        """LocalBackend should satisfy ExecutionBackend protocol."""
        backend = LocalBackend(max_workers=2)
        assert isinstance(backend, ExecutionBackend)

    def test_protocol_has_required_methods(self) -> None:
        """Verify protocol defines all required method signatures."""
        # Check that protocol has all expected method attributes
        expected_methods = [
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
        for method_name in expected_methods:
            assert hasattr(ExecutionBackend, method_name), (
                f"Protocol missing method: {method_name}"
            )


class TestNonConformingClasses:
    """Tests that non-conforming classes fail isinstance() checks."""

    def test_empty_class_does_not_satisfy_protocol(self) -> None:
        """Empty class should not satisfy ExecutionBackend protocol."""

        class EmptyBackend:
            pass

        backend = EmptyBackend()
        assert not isinstance(backend, ExecutionBackend)

    def test_partial_implementation_does_not_satisfy_protocol(self) -> None:
        """Class with only some methods should not satisfy protocol."""

        class PartialBackend:
            def broadcast(self, data):
                return data

            def destroy_broadcast(self, handle):
                pass

        backend = PartialBackend()
        assert not isinstance(backend, ExecutionBackend)

    def test_wrong_signature_does_not_satisfy_protocol(self) -> None:
        """Class with wrong method signatures should not satisfy protocol.

        Note: @runtime_checkable only checks method existence, not signatures.
        This test documents current behavior (isinstance returns True if
        all method names exist, regardless of signatures).
        """

        class WrongSignatureBackend:
            def broadcast(self):  # Missing 'data' parameter
                pass

            def destroy_broadcast(self):
                pass

            def parallel_fit(self):
                pass

            def get_parallelism(self):
                pass

            def collect_column(self):
                pass

            def get_column_stats(self):
                pass

            def sample_column(self):
                pass

            def create_dataframe(self):
                pass

            def compute_correlation(self):
                pass

            def compute_histogram(self):
                pass

            def generate_samples(self):
                pass

        backend = WrongSignatureBackend()
        # Note: runtime_checkable only checks attribute existence, not signatures
        assert isinstance(backend, ExecutionBackend)


class TestProtocolDocumentation:
    """Tests for protocol documentation and metadata."""

    def test_protocol_has_docstring(self) -> None:
        """ExecutionBackend should have comprehensive docstring."""
        assert ExecutionBackend.__doc__ is not None
        assert "Protocol" in ExecutionBackend.__doc__
        assert "backend" in ExecutionBackend.__doc__.lower()

    def test_protocol_methods_have_docstrings(self) -> None:
        """All protocol methods should have docstrings."""
        methods_to_check = [
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
        for method_name in methods_to_check:
            method = getattr(ExecutionBackend, method_name)
            assert method.__doc__ is not None, (
                f"Method {method_name} missing docstring"
            )


class TestDuckTypingConformance:
    """Tests for duck typing / structural subtyping with concrete backends."""

    def test_custom_backend_with_all_methods_satisfies_protocol(self) -> None:
        """Custom class with all required methods satisfies protocol."""

        class CustomBackend:
            """Custom backend implementing all required methods."""

            def broadcast(self, data):
                return data

            def destroy_broadcast(self, handle):
                pass

            def parallel_fit(
                self,
                distributions,
                histogram,
                data_sample,
                fit_func,
                column_name,
                data_stats=None,
                num_partitions=None,
                lower_bound=None,
                upper_bound=None,
                lazy_metrics=False,
                is_discrete=False,
                progress_callback=None,
                custom_distributions=None,
                estimation_method="mle",
            ):
                return []

            def get_parallelism(self):
                return 1

            def collect_column(self, df, column):
                return np.array([])

            def get_column_stats(self, df, column):
                return {"min": 0.0, "max": 1.0, "count": 0}

            def sample_column(self, df, column, fraction, seed):
                return np.array([])

            def create_dataframe(self, data, columns):
                return None

            def compute_correlation(self, df, columns, method="spearman"):
                return np.array([[1.0]])

            def compute_histogram(self, df, column, bin_edges):
                return np.array([0]), 0

            def generate_samples(
                self,
                n,
                generator_func,
                column_names,
                num_partitions=None,
                random_seed=None,
            ):
                return None

        backend = CustomBackend()
        assert isinstance(backend, ExecutionBackend)


class TestLocalBackendConformance:
    """Tests verifying LocalBackend satisfies protocol contract."""

    @pytest.fixture
    def backend(self) -> LocalBackend:
        """Create LocalBackend instance for testing."""
        return LocalBackend(max_workers=2)

    def test_broadcast_returns_data(self, backend: LocalBackend) -> None:
        """LocalBackend.broadcast() should return input data unchanged."""
        data = np.array([1, 2, 3])
        result = backend.broadcast(data)
        np.testing.assert_array_equal(result, data)

    def test_destroy_broadcast_is_noop(self, backend: LocalBackend) -> None:
        """LocalBackend.destroy_broadcast() should be a no-op."""
        # Should not raise
        backend.destroy_broadcast(None)
        backend.destroy_broadcast("anything")

    def test_get_parallelism_returns_positive_int(
        self, backend: LocalBackend
    ) -> None:
        """LocalBackend.get_parallelism() should return positive integer."""
        parallelism = backend.get_parallelism()
        assert isinstance(parallelism, int)
        assert parallelism > 0


class TestSparkBackendConformance:
    """Tests verifying SparkBackend satisfies protocol contract."""

    @pytest.fixture
    def spark_backend(self, spark_session):
        """Create SparkBackend instance for testing."""
        from spark_bestfit.backends.spark import SparkBackend

        return SparkBackend(spark_session)

    def test_spark_backend_satisfies_protocol(self, spark_backend) -> None:
        """SparkBackend should satisfy ExecutionBackend protocol."""
        assert isinstance(spark_backend, ExecutionBackend)

    def test_spark_backend_has_all_methods(self, spark_backend) -> None:
        """SparkBackend should have all required protocol methods."""
        expected_methods = [
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
        for method_name in expected_methods:
            assert hasattr(spark_backend, method_name), (
                f"SparkBackend missing method: {method_name}"
            )
            assert callable(getattr(spark_backend, method_name))


class TestRayBackendConformance:
    """Tests verifying RayBackend satisfies protocol contract."""

    @pytest.fixture
    def ray_backend(self):
        """Create RayBackend instance for testing."""
        pytest.importorskip("ray")
        from spark_bestfit.backends.ray import RayBackend

        return RayBackend()

    def test_ray_backend_satisfies_protocol(self, ray_backend) -> None:
        """RayBackend should satisfy ExecutionBackend protocol."""
        assert isinstance(ray_backend, ExecutionBackend)

    def test_ray_backend_has_all_methods(self, ray_backend) -> None:
        """RayBackend should have all required protocol methods."""
        expected_methods = [
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
        for method_name in expected_methods:
            assert hasattr(ray_backend, method_name), (
                f"RayBackend missing method: {method_name}"
            )
            assert callable(getattr(ray_backend, method_name))


class TestProtocolCannotBeInstantiated:
    """Tests verifying Protocol class cannot be directly instantiated."""

    def test_protocol_instantiation_raises_error(self) -> None:
        """Attempting to instantiate ExecutionBackend directly should raise."""
        with pytest.raises(TypeError, match="cannot be instantiated"):
            ExecutionBackend()
