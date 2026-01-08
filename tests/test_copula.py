"""Tests for Gaussian Copula module.

Uses LocalBackend for most tests. Spark-specific sampling tests are in a separate class.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from spark_bestfit import DistributionFitter, GaussianCopula
from spark_bestfit.copula import GaussianCopula
from spark_bestfit.results import DistributionFitResult, FitResults
from spark_bestfit.serialization import SerializationError


@pytest.fixture
def correlated_data():
    """Generate correlated multi-column data as pandas DataFrame."""
    np.random.seed(42)
    n = 5000

    # Generate correlated data using a known correlation structure
    # col_a ~ Normal(50, 10)
    # col_b ~ correlated with col_a, then transformed to Exponential
    # col_c ~ correlated with col_a and col_b

    mean = [0, 0, 0]
    cov = [[1.0, 0.7, 0.5], [0.7, 1.0, 0.3], [0.5, 0.3, 1.0]]

    mvn_samples = np.random.multivariate_normal(mean, cov, size=n)

    # Transform to target distributions
    col_a = 50 + 10 * mvn_samples[:, 0]  # Normal(50, 10)
    col_b = st.expon.ppf(st.norm.cdf(mvn_samples[:, 1]), scale=5)  # Exponential(5)
    col_c = 100 + 20 * mvn_samples[:, 2]  # Normal(100, 20)

    return pd.DataFrame({"col_a": col_a, "col_b": col_b, "col_c": col_c})


@pytest.fixture
def multi_column_results(local_backend, correlated_data):
    """Fit distributions to multi-column data."""
    fitter = DistributionFitter(backend=local_backend, random_seed=42)
    results = fitter.fit(
        correlated_data,
        columns=["col_a", "col_b", "col_c"],
        max_distributions=5,
    )
    return results


@pytest.fixture
def simple_copula(multi_column_results, correlated_data, local_backend):
    """Create a fitted GaussianCopula."""
    return GaussianCopula.fit(multi_column_results, correlated_data, backend=local_backend)


class TestGaussianCopulaFit:
    """Tests for GaussianCopula.fit() method."""

    def test_fit_basic(self, multi_column_results, correlated_data, local_backend):
        """Test basic copula fitting."""
        copula = GaussianCopula.fit(multi_column_results, correlated_data, backend=local_backend)

        assert copula.column_names == ["col_a", "col_b", "col_c"]
        assert len(copula.marginals) == 3
        assert copula.correlation_matrix.shape == (3, 3)

    def test_fit_correlation_matrix_properties(self, simple_copula):
        """Test that correlation matrix has expected properties."""
        corr = simple_copula.correlation_matrix

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr), [1.0, 1.0, 1.0])

        # Should be symmetric
        np.testing.assert_array_almost_equal(corr, corr.T)

        # Off-diagonal should be between -1 and 1
        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)

    def test_fit_preserves_correlation_structure(self, simple_copula):
        """Test that fitted correlation approximately matches original."""
        corr = simple_copula.correlation_matrix

        # Original correlation was approximately:
        # [[1.0, 0.7, 0.5], [0.7, 1.0, 0.3], [0.5, 0.3, 1.0]]
        # Spearman correlation should be close (but not exact due to transformations)

        # Check that correlations are in reasonable range
        assert corr[0, 1] > 0.5  # col_a and col_b should be positively correlated
        assert corr[0, 2] > 0.3  # col_a and col_c should be positively correlated

    def test_fit_marginals_are_distributions(self, simple_copula):
        """Test that marginals are valid distribution fit results."""
        for col, marginal in simple_copula.marginals.items():
            assert isinstance(marginal, DistributionFitResult)
            assert marginal.distribution is not None
            assert len(marginal.parameters) > 0

    def test_fit_specific_columns(self, multi_column_results, correlated_data, local_backend):
        """Test fitting with specific column subset."""
        copula = GaussianCopula.fit(
            multi_column_results, correlated_data, columns=["col_a", "col_b"], backend=local_backend
        )

        assert copula.column_names == ["col_a", "col_b"]
        assert len(copula.marginals) == 2
        assert copula.correlation_matrix.shape == (2, 2)

    def test_fit_requires_minimum_columns(self, multi_column_results, correlated_data, local_backend):
        """Test that fit requires at least 2 columns."""
        with pytest.raises(ValueError, match="at least 2 columns"):
            GaussianCopula.fit(
                multi_column_results, correlated_data, columns=["col_a"], backend=local_backend
            )

    def test_fit_missing_columns_error(self, multi_column_results, correlated_data, local_backend):
        """Test error when columns not in results."""
        with pytest.raises(ValueError, match="not found in results"):
            GaussianCopula.fit(
                multi_column_results, correlated_data, columns=["col_a", "nonexistent"], backend=local_backend
            )


class TestGaussianCopulaSample:
    """Tests for local sampling via sample()."""

    def test_sample_basic(self, simple_copula):
        """Test basic local sampling."""
        samples = simple_copula.sample(n=1000)

        assert isinstance(samples, dict)
        assert set(samples.keys()) == {"col_a", "col_b", "col_c"}
        for col, arr in samples.items():
            assert len(arr) == 1000
            assert np.all(np.isfinite(arr))

    def test_sample_reproducibility(self, simple_copula):
        """Test that sampling is reproducible with seed."""
        samples1 = simple_copula.sample(n=100, random_state=42)
        samples2 = simple_copula.sample(n=100, random_state=42)

        for col in simple_copula.column_names:
            np.testing.assert_array_almost_equal(samples1[col], samples2[col])

    def test_sample_different_seeds_different_results(self, simple_copula):
        """Test that different seeds produce different results."""
        samples1 = simple_copula.sample(n=100, random_state=42)
        samples2 = simple_copula.sample(n=100, random_state=123)

        # At least one column should have different values
        any_different = False
        for col in simple_copula.column_names:
            if not np.allclose(samples1[col], samples2[col]):
                any_different = True
                break
        assert any_different

    def test_sample_preserves_correlation(self, simple_copula):
        """Test that sampled data preserves correlation structure."""
        samples = simple_copula.sample(n=5000, random_state=42)

        # Convert to array for correlation computation
        sample_array = np.column_stack(
            [samples[col] for col in simple_copula.column_names]
        )

        # Compute Spearman correlation of samples
        sampled_corr = np.corrcoef(sample_array.T)

        # Should be close to original correlation matrix
        np.testing.assert_array_almost_equal(
            sampled_corr, simple_copula.correlation_matrix, decimal=1
        )

    def test_sample_return_uniform(self, simple_copula):
        """Test return_uniform=True returns uniform samples."""
        samples = simple_copula.sample(n=10000, random_state=42, return_uniform=True)

        for col in simple_copula.column_names:
            arr = samples[col]
            # Uniform samples should be in [0, 1]
            assert np.all(arr >= 0.0)
            assert np.all(arr <= 1.0)
            # Should be roughly uniformly distributed
            assert 0.4 < np.mean(arr) < 0.6  # Mean should be ~0.5

    def test_sample_return_uniform_preserves_correlation(self, simple_copula):
        """Test that uniform samples preserve correlation structure."""
        samples = simple_copula.sample(n=5000, random_state=42, return_uniform=True)

        sample_array = np.column_stack(
            [samples[col] for col in simple_copula.column_names]
        )

        # Compute Spearman correlation (works for uniform data too)
        from scipy.stats import spearmanr

        sampled_corr = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                sampled_corr[i, j], _ = spearmanr(sample_array[:, i], sample_array[:, j])

        # Should be close to original correlation matrix
        np.testing.assert_array_almost_equal(
            sampled_corr, simple_copula.correlation_matrix, decimal=1
        )

    def test_sample_return_uniform_is_faster(self, simple_copula):
        """Test that return_uniform=True is faster than default."""
        import time

        # Warm up
        simple_copula.sample(n=1000, random_state=42)
        simple_copula.sample(n=1000, random_state=42, return_uniform=True)

        # Time with marginal transform
        start = time.time()
        simple_copula.sample(n=100000, random_state=42)
        time_with_transform = time.time() - start

        # Time without marginal transform
        start = time.time()
        simple_copula.sample(n=100000, random_state=42, return_uniform=True)
        time_uniform = time.time() - start

        # Uniform should be significantly faster
        assert time_uniform < time_with_transform

    def test_sample_marginal_distribution(self, simple_copula):
        """Test that marginals approximately match fitted distributions."""
        samples = simple_copula.sample(n=5000, random_state=42)

        for col, marginal in simple_copula.marginals.items():
            col_samples = samples[col]

            # K-S test against the fitted distribution
            # get_scipy_dist() now returns frozen distribution with parameters applied
            frozen_dist = marginal.get_scipy_dist()
            cdf_func = lambda x, d=frozen_dist: d.cdf(x)  # capture d to avoid late binding
            ks_stat, p_value = st.kstest(col_samples, cdf_func)

            # P-value should be > 0.01 (samples come from the distribution)
            assert p_value > 0.01, f"Column {col}: KS={ks_stat}, p={p_value}"


class TestGaussianCopulaSerialization:
    """Tests for save/load functionality."""

    def test_save_load_json(self, simple_copula):
        """Test JSON round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            simple_copula.save(path)
            loaded = GaussianCopula.load(path)

            assert loaded.column_names == simple_copula.column_names
            np.testing.assert_array_almost_equal(
                loaded.correlation_matrix, simple_copula.correlation_matrix
            )
            for col in simple_copula.column_names:
                assert loaded.marginals[col].distribution == simple_copula.marginals[col].distribution
                assert loaded.marginals[col].parameters == simple_copula.marginals[col].parameters
        finally:
            path.unlink()

    def test_save_load_pickle(self, simple_copula):
        """Test pickle round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            simple_copula.save(path)
            loaded = GaussianCopula.load(path)

            assert loaded.column_names == simple_copula.column_names
            np.testing.assert_array_almost_equal(
                loaded.correlation_matrix, simple_copula.correlation_matrix
            )
        finally:
            path.unlink()

    def test_loaded_copula_can_sample(self, simple_copula):
        """Test that loaded copula can generate samples."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            simple_copula.save(path)
            loaded = GaussianCopula.load(path)

            samples = loaded.sample(n=100, random_state=42)
            assert len(samples["col_a"]) == 100
        finally:
            path.unlink()

    def test_load_invalid_json(self):
        """Test error on invalid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json")
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError, match="Invalid JSON"):
                GaussianCopula.load(path)
        finally:
            path.unlink()

    def test_load_missing_fields(self):
        """Test error on missing required fields."""
        import json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"column_names": ["a", "b"]}, f)  # Missing other fields
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError, match="Missing required field"):
                GaussianCopula.load(path)
        finally:
            path.unlink()

    def test_load_unknown_distribution(self):
        """Test error on unknown distribution."""
        import json

        data = {
            "column_names": ["a", "b"],
            "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
            "marginals": {
                "a": {"distribution": "unknown_dist", "parameters": [1.0]},
                "b": {"distribution": "norm", "parameters": [0.0, 1.0]},
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError, match="Unknown distribution"):
                GaussianCopula.load(path)
        finally:
            path.unlink()


class TestGaussianCopulaEdgeCases:
    """Edge case tests."""

    def test_two_columns_minimum(self, local_backend):
        """Test copula with exactly 2 columns (minimum)."""
        np.random.seed(42)
        n = 1000

        # Generate correlated 2-column data
        cov = [[1.0, 0.8], [0.8, 1.0]]
        mvn = np.random.multivariate_normal([0, 0], cov, size=n)
        col_a = 50 + 10 * mvn[:, 0]
        col_b = 100 + 20 * mvn[:, 1]

        df = pd.DataFrame({"col_a": col_a, "col_b": col_b})

        fitter = DistributionFitter(backend=local_backend, random_seed=42)
        results = fitter.fit(df, columns=["col_a", "col_b"], max_distributions=3)

        copula = GaussianCopula.fit(results, df, backend=local_backend)
        assert len(copula.column_names) == 2

        samples = copula.sample(n=100)
        assert len(samples) == 2

    def test_copula_init_validation(self):
        """Test that __post_init__ validates state."""
        # Too few columns
        with pytest.raises(ValueError, match="at least 2 columns"):
            GaussianCopula(
                column_names=["a"],
                marginals={"a": DistributionFitResult("norm", [0, 1], sse=0.01)},
                correlation_matrix=np.array([[1.0]]),
            )

        # Mismatched columns
        with pytest.raises(ValueError, match="must match marginals keys"):
            GaussianCopula(
                column_names=["a", "b"],
                marginals={"a": DistributionFitResult("norm", [0, 1], sse=0.01)},
                correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            )

        # Wrong correlation matrix shape
        with pytest.raises(ValueError, match="doesn't match"):
            GaussianCopula(
                column_names=["a", "b"],
                marginals={
                    "a": DistributionFitResult("norm", [0, 1], sse=0.01),
                    "b": DistributionFitResult("norm", [0, 1], sse=0.01),
                },
                correlation_matrix=np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]),
            )

    def test_sample_with_zero_samples(self, simple_copula):
        """Test sampling zero samples."""
        samples = simple_copula.sample(n=0)
        for col in simple_copula.column_names:
            assert len(samples[col]) == 0


class TestGaussianCopulaIntegration:
    """Integration tests for end-to-end workflows."""

    def test_fit_sample_verify_workflow(self, local_backend):
        """Test complete workflow: generate data, fit, sample, verify."""
        np.random.seed(42)
        n = 3000

        # Generate data with known correlation
        target_corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        mvn = np.random.multivariate_normal([0, 0], target_corr, size=n)

        # Transform to different distributions
        col_a = 50 + 10 * mvn[:, 0]  # Normal(50, 10)
        col_b = np.exp(mvn[:, 1])  # Log-normal

        df = pd.DataFrame({"col_a": col_a, "col_b": col_b})

        # Fit distributions
        fitter = DistributionFitter(backend=local_backend, random_seed=42)
        results = fitter.fit(df, columns=["col_a", "col_b"], max_distributions=5)

        # Fit copula
        copula = GaussianCopula.fit(results, df, backend=local_backend)

        # Sample
        samples = copula.sample(n=3000, random_state=42)

        # Verify correlation is preserved (within tolerance)
        sample_array = np.column_stack([samples["col_a"], samples["col_b"]])
        sampled_corr = np.corrcoef(sample_array.T)

        # Correlation should be close to original
        assert abs(sampled_corr[0, 1] - copula.correlation_matrix[0, 1]) < 0.1


# ============================================================================
# Spark-specific tests (skip if PySpark not installed)
# ============================================================================

try:
    import pyspark  # noqa: F401

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@pytest.fixture
def spark_correlated_data(spark_session):
    """Generate correlated multi-column data as Spark DataFrame."""
    np.random.seed(42)
    n = 5000

    mean = [0, 0, 0]
    cov = [[1.0, 0.7, 0.5], [0.7, 1.0, 0.3], [0.5, 0.3, 1.0]]

    mvn_samples = np.random.multivariate_normal(mean, cov, size=n)

    col_a = 50 + 10 * mvn_samples[:, 0]
    col_b = st.expon.ppf(st.norm.cdf(mvn_samples[:, 1]), scale=5)
    col_c = 100 + 20 * mvn_samples[:, 2]

    data = [(float(a), float(b), float(c)) for a, b, c in zip(col_a, col_b, col_c)]
    return spark_session.createDataFrame(data, ["col_a", "col_b", "col_c"])


@pytest.fixture
def spark_multi_column_results(spark_session, spark_correlated_data):
    """Fit distributions to multi-column Spark data."""
    fitter = DistributionFitter(spark_session, random_seed=42)
    results = fitter.fit(
        spark_correlated_data,
        columns=["col_a", "col_b", "col_c"],
        max_distributions=5,
    )
    return results


@pytest.fixture
def spark_simple_copula(spark_multi_column_results, spark_correlated_data):
    """Create a fitted GaussianCopula from Spark data."""
    return GaussianCopula.fit(spark_multi_column_results, spark_correlated_data)


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestGaussianCopulaSampleDistributed:
    """Tests for distributed sampling via sample_distributed()."""

    def test_sample_distributed_basic(self, spark_simple_copula, spark_session):
        """Test basic distributed sampling with Spark backend."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = spark_simple_copula.sample_distributed(n=1000, backend=backend)

        assert samples_df.count() == 1000
        assert set(samples_df.columns) == {"col_a", "col_b", "col_c"}

    def test_sample_distributed_reproducibility(self, spark_simple_copula, spark_session):
        """Test distributed sampling reproducibility."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        df1 = spark_simple_copula.sample_distributed(
            n=100, backend=backend, num_partitions=2, random_seed=42
        )
        df2 = spark_simple_copula.sample_distributed(
            n=100, backend=backend, num_partitions=2, random_seed=42
        )

        # Convert to pandas and sort for comparison
        pdf1 = df1.toPandas().sort_values("col_a").reset_index(drop=True)
        pdf2 = df2.toPandas().sort_values("col_a").reset_index(drop=True)

        for col in spark_simple_copula.column_names:
            np.testing.assert_array_almost_equal(pdf1[col].values, pdf2[col].values)

    def test_sample_distributed_different_seeds(self, spark_simple_copula, spark_session):
        """Test that different seeds produce different results."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        df1 = spark_simple_copula.sample_distributed(n=100, backend=backend, random_seed=42)
        df2 = spark_simple_copula.sample_distributed(n=100, backend=backend, random_seed=123)

        pdf1 = df1.toPandas()
        pdf2 = df2.toPandas()

        # Sample values should be different (not identical)
        # Check that at least some values differ
        assert not np.allclose(
            pdf1["col_a"].sort_values().values,
            pdf2["col_a"].sort_values().values,
        )

    def test_sample_distributed_preserves_correlation(self, spark_simple_copula, spark_session):
        """Test that distributed sampling preserves correlation."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = spark_simple_copula.sample_distributed(
            n=5000, backend=backend, random_seed=42
        )
        pdf = samples_df.toPandas()

        # Compute Spearman correlation of samples
        sampled_corr = pdf[spark_simple_copula.column_names].corr(method="spearman").values

        # Should be close to original correlation matrix
        np.testing.assert_array_almost_equal(
            sampled_corr, spark_simple_copula.correlation_matrix, decimal=1
        )

    def test_sample_distributed_custom_partitions(self, spark_simple_copula, spark_session):
        """Test sampling with custom partition count."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = spark_simple_copula.sample_distributed(
            n=1000, backend=backend, num_partitions=4, random_seed=42
        )
        assert samples_df.count() == 1000

    def test_sample_distributed_return_uniform(self, spark_simple_copula, spark_session):
        """Test return_uniform=True in sample_distributed()."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = spark_simple_copula.sample_distributed(
            n=1000, backend=backend, random_seed=42, return_uniform=True
        )

        pdf = samples_df.toPandas()
        for col in spark_simple_copula.column_names:
            arr = pdf[col].values
            # Uniform samples should be in [0, 1]
            assert np.all(arr >= 0.0)
            assert np.all(arr <= 1.0)
            # Should be roughly uniformly distributed
            assert 0.3 < np.mean(arr) < 0.7  # Mean should be ~0.5

    def test_sample_distributed_with_one_sample(self, spark_simple_copula, spark_session):
        """Test distributed sampling with exactly 1 sample."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        df = spark_simple_copula.sample_distributed(n=1, backend=backend, random_seed=42)
        assert df.count() == 1

    def test_save_load_sample_distributed_workflow(self, spark_simple_copula, spark_session):
        """Test serialization then distributed sampling."""
        from spark_bestfit.backends import BackendFactory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            spark_simple_copula.save(path)

            # Load
            loaded = GaussianCopula.load(path)

            # Sample via backend
            backend = BackendFactory.create("spark", spark_session=spark_session)
            samples_df = loaded.sample_distributed(n=1000, backend=backend, random_seed=42)

            assert samples_df.count() == 1000
            assert set(samples_df.columns) == set(spark_simple_copula.column_names)
        finally:
            path.unlink()
