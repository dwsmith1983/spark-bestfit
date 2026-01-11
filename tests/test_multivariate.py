"""Tests for Multivariate Normal distribution fitting module.

Uses LocalBackend for most tests. Spark-specific tests are in a separate class.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spark_bestfit import MultivariateNormalFitter, MultivariateNormalResult
from spark_bestfit.serialization import SerializationError


@pytest.fixture
def mvn_data():
    """Generate multivariate normal data as pandas DataFrame."""
    np.random.seed(42)
    n = 5000

    # Generate data with known mean and covariance
    mean = [10.0, 20.0, 30.0]
    cov = [[4.0, 2.0, 1.0], [2.0, 9.0, 3.0], [1.0, 3.0, 16.0]]

    samples = np.random.multivariate_normal(mean, cov, size=n)

    return pd.DataFrame({"col_a": samples[:, 0], "col_b": samples[:, 1], "col_c": samples[:, 2]})


@pytest.fixture
def simple_mvn_result(local_backend, mvn_data):
    """Create a fitted MultivariateNormalResult."""
    fitter = MultivariateNormalFitter(backend=local_backend)
    return fitter.fit(mvn_data, columns=["col_a", "col_b", "col_c"])


class TestMultivariateNormalFitterFit:
    """Tests for MultivariateNormalFitter.fit() method."""

    def test_fit_basic(self, local_backend, mvn_data):
        """Test basic multivariate normal fitting."""
        fitter = MultivariateNormalFitter(backend=local_backend)
        result = fitter.fit(mvn_data, columns=["col_a", "col_b", "col_c"])

        assert result.column_names == ["col_a", "col_b", "col_c"]
        assert result.mean.shape == (3,)
        assert result.cov.shape == (3, 3)
        assert result.n_samples == 5000

    def test_fit_recovers_parameters(self, local_backend, mvn_data):
        """Test that fit recovers approximately correct parameters."""
        fitter = MultivariateNormalFitter(backend=local_backend)
        result = fitter.fit(mvn_data, columns=["col_a", "col_b", "col_c"])

        # Check mean is close to true mean [10, 20, 30]
        expected_mean = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result.mean, expected_mean, decimal=0)

        # Check covariance diagonal is close to true variances [4, 9, 16]
        expected_var = np.array([4.0, 9.0, 16.0])
        np.testing.assert_array_almost_equal(np.diag(result.cov), expected_var, decimal=0)

    def test_fit_covariance_symmetric(self, simple_mvn_result):
        """Test that covariance matrix is symmetric."""
        cov = simple_mvn_result.cov
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_fit_covariance_positive_definite(self, simple_mvn_result):
        """Test that covariance matrix is positive definite."""
        eigenvalues = np.linalg.eigvalsh(simple_mvn_result.cov)
        assert np.all(eigenvalues > 0)

    def test_fit_bias_parameter(self, local_backend, mvn_data):
        """Test bias parameter affects covariance estimate."""
        fitter = MultivariateNormalFitter(backend=local_backend)

        result_biased = fitter.fit(mvn_data, columns=["col_a", "col_b"], bias=True)
        result_unbiased = fitter.fit(mvn_data, columns=["col_a", "col_b"], bias=False)

        # Biased estimate should be slightly smaller (by factor n/(n-1))
        n = mvn_data.shape[0]
        expected_ratio = n / (n - 1)
        actual_ratio = result_unbiased.cov[0, 0] / result_biased.cov[0, 0]
        np.testing.assert_almost_equal(actual_ratio, expected_ratio, decimal=3)

    def test_fit_requires_minimum_columns(self, local_backend, mvn_data):
        """Test that fit requires at least 2 columns."""
        fitter = MultivariateNormalFitter(backend=local_backend)
        with pytest.raises(ValueError, match="at least 2 columns"):
            fitter.fit(mvn_data, columns=["col_a"])

    def test_fit_missing_columns_error(self, local_backend, mvn_data):
        """Test error when columns not in DataFrame."""
        fitter = MultivariateNormalFitter(backend=local_backend)
        with pytest.raises(ValueError, match="not found"):
            fitter.fit(mvn_data, columns=["col_a", "nonexistent"])

    def test_fit_two_columns(self, local_backend):
        """Test fitting with exactly 2 columns (minimum)."""
        np.random.seed(42)
        n = 1000

        cov = [[4.0, 2.0], [2.0, 9.0]]
        samples = np.random.multivariate_normal([10, 20], cov, size=n)
        df = pd.DataFrame({"x": samples[:, 0], "y": samples[:, 1]})

        fitter = MultivariateNormalFitter(backend=local_backend)
        result = fitter.fit(df, columns=["x", "y"])

        assert len(result.column_names) == 2
        assert result.cov.shape == (2, 2)


class TestMultivariateNormalResultSample:
    """Tests for local sampling via sample()."""

    def test_sample_basic(self, simple_mvn_result):
        """Test basic local sampling."""
        samples = simple_mvn_result.sample(n=1000)

        assert isinstance(samples, dict)
        assert set(samples.keys()) == {"col_a", "col_b", "col_c"}
        for col, arr in samples.items():
            assert len(arr) == 1000
            assert np.all(np.isfinite(arr))

    def test_sample_reproducibility(self, simple_mvn_result):
        """Test that sampling is reproducible with seed."""
        samples1 = simple_mvn_result.sample(n=100, random_state=42)
        samples2 = simple_mvn_result.sample(n=100, random_state=42)

        for col in simple_mvn_result.column_names:
            np.testing.assert_array_almost_equal(samples1[col], samples2[col])

    def test_sample_different_seeds_different_results(self, simple_mvn_result):
        """Test that different seeds produce different results."""
        samples1 = simple_mvn_result.sample(n=100, random_state=42)
        samples2 = simple_mvn_result.sample(n=100, random_state=123)

        # At least one column should have different values
        any_different = False
        for col in simple_mvn_result.column_names:
            if not np.allclose(samples1[col], samples2[col]):
                any_different = True
                break
        assert any_different

    def test_sample_preserves_mean(self, simple_mvn_result):
        """Test that sampled data has approximately correct mean."""
        samples = simple_mvn_result.sample(n=10000, random_state=42)

        sample_array = np.column_stack(
            [samples[col] for col in simple_mvn_result.column_names]
        )
        sampled_mean = np.mean(sample_array, axis=0)

        np.testing.assert_array_almost_equal(
            sampled_mean, simple_mvn_result.mean, decimal=1
        )

    def test_sample_preserves_covariance(self, simple_mvn_result):
        """Test that sampled data has approximately correct covariance."""
        samples = simple_mvn_result.sample(n=10000, random_state=42)

        sample_array = np.column_stack(
            [samples[col] for col in simple_mvn_result.column_names]
        )
        sampled_cov = np.cov(sample_array, rowvar=False)

        np.testing.assert_array_almost_equal(
            sampled_cov, simple_mvn_result.cov, decimal=0
        )

    def test_sample_zero_samples(self, simple_mvn_result):
        """Test sampling zero samples."""
        samples = simple_mvn_result.sample(n=0)
        for col in simple_mvn_result.column_names:
            assert len(samples[col]) == 0


class TestMultivariateNormalResultMethods:
    """Tests for pdf, logpdf, mahalanobis methods."""

    def test_pdf_single_point(self, simple_mvn_result):
        """Test PDF evaluation at single point."""
        point = simple_mvn_result.mean  # At mean, PDF is maximum
        pdf_value = simple_mvn_result.pdf(point)
        assert pdf_value > 0
        assert np.isfinite(pdf_value)

    def test_pdf_batch(self, simple_mvn_result):
        """Test PDF evaluation at multiple points."""
        points = np.array([[10, 20, 30], [11, 21, 31], [9, 19, 29]])
        pdf_values = simple_mvn_result.pdf(points)
        assert len(pdf_values) == 3
        assert np.all(pdf_values > 0)

    def test_logpdf_single_point(self, simple_mvn_result):
        """Test log-PDF evaluation at single point."""
        point = simple_mvn_result.mean
        logpdf_value = simple_mvn_result.logpdf(point)
        assert np.isfinite(logpdf_value)

    def test_logpdf_consistent_with_pdf(self, simple_mvn_result):
        """Test that logpdf is consistent with log(pdf)."""
        points = np.array([[10, 20, 30], [11, 21, 31]])
        pdf_values = simple_mvn_result.pdf(points)
        logpdf_values = simple_mvn_result.logpdf(points)
        np.testing.assert_array_almost_equal(logpdf_values, np.log(pdf_values))

    def test_mahalanobis_at_mean(self, simple_mvn_result):
        """Test Mahalanobis distance at mean is zero."""
        distance = simple_mvn_result.mahalanobis(simple_mvn_result.mean)
        np.testing.assert_almost_equal(distance[0], 0.0, decimal=10)

    def test_mahalanobis_batch(self, simple_mvn_result):
        """Test vectorized Mahalanobis distance (batch input)."""
        points = np.array([
            simple_mvn_result.mean,
            simple_mvn_result.mean + 1.0,
            simple_mvn_result.mean + 2.0,
        ])
        distances = simple_mvn_result.mahalanobis(points)
        assert len(distances) == 3
        assert distances[0] < distances[1] < distances[2]

    def test_mahalanobis_symmetry(self, simple_mvn_result):
        """Test Mahalanobis distance is symmetric around mean."""
        offset = np.array([1.0, 1.0, 1.0])
        point_plus = simple_mvn_result.mean + offset
        point_minus = simple_mvn_result.mean - offset

        dist_plus = simple_mvn_result.mahalanobis(point_plus)
        dist_minus = simple_mvn_result.mahalanobis(point_minus)

        np.testing.assert_almost_equal(dist_plus, dist_minus)

    def test_correlation_matrix(self, simple_mvn_result):
        """Test correlation matrix computation."""
        corr = simple_mvn_result.correlation_matrix()

        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(3))

        # Should be symmetric
        np.testing.assert_array_almost_equal(corr, corr.T)

        # Off-diagonal should be between -1 and 1 (allow small numerical tolerance)
        assert np.all(corr >= -1.0 - 1e-10)
        assert np.all(corr <= 1.0 + 1e-10)


class TestMultivariateNormalResultSerialization:
    """Tests for save/load functionality."""

    def test_save_load_json(self, simple_mvn_result):
        """Test JSON round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            simple_mvn_result.save(path)
            loaded = MultivariateNormalResult.load(path)

            assert loaded.column_names == simple_mvn_result.column_names
            np.testing.assert_array_almost_equal(loaded.mean, simple_mvn_result.mean)
            np.testing.assert_array_almost_equal(loaded.cov, simple_mvn_result.cov)
            assert loaded.n_samples == simple_mvn_result.n_samples
        finally:
            path.unlink()

    def test_save_load_pickle(self, simple_mvn_result):
        """Test pickle round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            simple_mvn_result.save(path)
            loaded = MultivariateNormalResult.load(path)

            assert loaded.column_names == simple_mvn_result.column_names
            np.testing.assert_array_almost_equal(loaded.mean, simple_mvn_result.mean)
            np.testing.assert_array_almost_equal(loaded.cov, simple_mvn_result.cov)
        finally:
            path.unlink()

    def test_loaded_result_can_sample(self, simple_mvn_result):
        """Test that loaded result can generate samples."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            simple_mvn_result.save(path)
            loaded = MultivariateNormalResult.load(path)

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
                MultivariateNormalResult.load(path)
        finally:
            path.unlink()

    def test_load_missing_fields(self):
        """Test error on missing required fields."""
        import json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"column_names": ["a", "b"]}, f)  # Missing mean, cov
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError, match="Missing required field"):
                MultivariateNormalResult.load(path)
        finally:
            path.unlink()


class TestMultivariateNormalEdgeCases:
    """Edge case tests."""

    def test_init_validation_too_few_columns(self):
        """Test that __post_init__ validates minimum columns."""
        with pytest.raises(ValueError, match="at least 2 columns"):
            MultivariateNormalResult(
                column_names=["a"],
                mean=np.array([1.0]),
                cov=np.array([[1.0]]),
            )

    def test_init_validation_mean_shape_mismatch(self):
        """Test that __post_init__ validates mean shape."""
        with pytest.raises(ValueError, match="mean shape"):
            MultivariateNormalResult(
                column_names=["a", "b"],
                mean=np.array([1.0]),  # Should be length 2
                cov=np.array([[1.0, 0.5], [0.5, 1.0]]),
            )

    def test_init_validation_cov_shape_mismatch(self):
        """Test that __post_init__ validates covariance shape."""
        with pytest.raises(ValueError, match="cov shape"):
            MultivariateNormalResult(
                column_names=["a", "b"],
                mean=np.array([1.0, 2.0]),
                cov=np.array([[1.0]]),  # Should be 2x2
            )

    def test_numerical_stability_near_collinear(self, local_backend):
        """Test behavior with near-collinear columns (high condition number)."""
        np.random.seed(42)
        n = 1000

        # Create near-collinear data: col_c = col_a + tiny noise
        # Use extremely small noise to create very high condition number
        col_a = np.random.normal(10, 2, n)
        col_b = np.random.normal(20, 3, n)
        col_c = col_a + np.random.normal(0, 1e-7, n)  # Nearly perfectly correlated

        df = pd.DataFrame({"col_a": col_a, "col_b": col_b, "col_c": col_c})

        fitter = MultivariateNormalFitter(backend=local_backend)

        # Should emit a warning about high condition number
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fitter.fit(df, columns=["col_a", "col_b", "col_c"])

            # Check that warning was issued about condition number
            condition_warnings = [
                warning for warning in w
                if "condition number" in str(warning.message).lower()
            ]
            assert len(condition_warnings) >= 1, f"Expected condition number warning, got: {[str(x.message) for x in w]}"

        # Result should still be valid (regularization applied in __post_init__)
        assert result is not None
        samples = result.sample(n=100, random_state=42)
        assert len(samples["col_a"]) == 100

    def test_high_dimensional(self, local_backend):
        """Test with 15+ columns to verify performance doesn't degrade."""
        np.random.seed(42)
        n_cols = 15
        n_samples = 2000

        # Generate high-dimensional MVN data
        mean = np.arange(n_cols, dtype=float)
        cov = np.eye(n_cols) + 0.1 * np.ones((n_cols, n_cols))  # Positive definite

        samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        columns = [f"col_{i}" for i in range(n_cols)]
        df = pd.DataFrame(samples, columns=columns)

        fitter = MultivariateNormalFitter(backend=local_backend)
        result = fitter.fit(df, columns=columns)

        assert result.n_dimensions == n_cols
        assert result.mean.shape == (n_cols,)
        assert result.cov.shape == (n_cols, n_cols)

        # Sampling should still work
        new_samples = result.sample(n=100, random_state=42)
        assert len(new_samples[columns[0]]) == 100

    def test_n_dimensions_property(self, simple_mvn_result):
        """Test n_dimensions property."""
        assert simple_mvn_result.n_dimensions == 3


class TestMultivariateNormalIntegration:
    """Integration tests for end-to-end workflows."""

    def test_fit_sample_verify_workflow(self, local_backend):
        """Test complete workflow: fit, sample, verify."""
        np.random.seed(42)
        n = 5000

        # Generate data with known parameters
        true_mean = np.array([100.0, 200.0])
        true_cov = np.array([[25.0, 15.0], [15.0, 36.0]])

        samples = np.random.multivariate_normal(true_mean, true_cov, size=n)
        df = pd.DataFrame({"x": samples[:, 0], "y": samples[:, 1]})

        # Fit
        fitter = MultivariateNormalFitter(backend=local_backend)
        result = fitter.fit(df, columns=["x", "y"])

        # Generate new samples
        new_samples = result.sample(n=10000, random_state=42)
        new_array = np.column_stack([new_samples["x"], new_samples["y"]])

        # Verify: new samples should have statistics matching the FITTED model
        # (not necessarily the true parameters, which may differ due to sampling)
        np.testing.assert_array_almost_equal(
            np.mean(new_array, axis=0), result.mean, decimal=0
        )
        # Use relative tolerance for covariance - sampled stats vary
        np.testing.assert_allclose(
            np.cov(new_array, rowvar=False), result.cov, rtol=0.1
        )

    def test_save_load_sample_workflow(self, simple_mvn_result):
        """Test save, load, then sample workflow."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            simple_mvn_result.save(path)

            # Load
            loaded = MultivariateNormalResult.load(path)

            # Sample from loaded result
            samples = loaded.sample(n=1000, random_state=42)

            # Verify samples are valid
            assert len(samples["col_a"]) == 1000
            assert np.all(np.isfinite(samples["col_a"]))
        finally:
            path.unlink()


# ============================================================================
# Spark-specific tests (skip if PySpark not installed)
# ============================================================================

try:
    import pyspark  # noqa: F401

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@pytest.fixture
def spark_mvn_data(spark_session):
    """Generate MVN data as Spark DataFrame."""
    np.random.seed(42)
    n = 5000

    mean = [10.0, 20.0, 30.0]
    cov = [[4.0, 2.0, 1.0], [2.0, 9.0, 3.0], [1.0, 3.0, 16.0]]

    samples = np.random.multivariate_normal(mean, cov, size=n)

    data = [(float(samples[i, 0]), float(samples[i, 1]), float(samples[i, 2])) for i in range(n)]
    return spark_session.createDataFrame(data, ["col_a", "col_b", "col_c"])


@pytest.fixture
def spark_mvn_result(spark_session, spark_mvn_data):
    """Create a fitted MultivariateNormalResult from Spark data."""
    fitter = MultivariateNormalFitter()
    return fitter.fit(spark_mvn_data, columns=["col_a", "col_b", "col_c"])


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not installed")
class TestMultivariateNormalSpark:
    """Spark-specific tests."""

    def test_fit_from_spark_dataframe(self, spark_mvn_result):
        """Test fitting from Spark DataFrame."""
        assert spark_mvn_result.column_names == ["col_a", "col_b", "col_c"]
        assert spark_mvn_result.mean.shape == (3,)
        assert spark_mvn_result.cov.shape == (3, 3)

    def test_sample_distributed(self, spark_mvn_result, spark_session):
        """Test distributed sampling with Spark backend."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = spark_mvn_result.sample_distributed(n=1000, backend=backend)

        assert samples_df.count() == 1000
        assert set(samples_df.columns) == {"col_a", "col_b", "col_c"}

    def test_sample_distributed_reproducibility(self, spark_mvn_result, spark_session):
        """Test distributed sampling reproducibility."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        df1 = spark_mvn_result.sample_distributed(
            n=100, backend=backend, num_partitions=2, random_seed=42
        )
        df2 = spark_mvn_result.sample_distributed(
            n=100, backend=backend, num_partitions=2, random_seed=42
        )

        # Convert to pandas and sort for comparison
        pdf1 = df1.toPandas().sort_values("col_a").reset_index(drop=True)
        pdf2 = df2.toPandas().sort_values("col_a").reset_index(drop=True)

        for col in spark_mvn_result.column_names:
            np.testing.assert_array_almost_equal(pdf1[col].values, pdf2[col].values)

    def test_sample_distributed_preserves_statistics(self, spark_mvn_result, spark_session):
        """Test that distributed sampling preserves mean and covariance."""
        from spark_bestfit.backends import BackendFactory

        backend = BackendFactory.create("spark", spark_session=spark_session)
        samples_df = spark_mvn_result.sample_distributed(
            n=5000, backend=backend, random_seed=42
        )
        pdf = samples_df.toPandas()

        # Check mean
        sampled_mean = pdf[spark_mvn_result.column_names].mean().values
        np.testing.assert_array_almost_equal(
            sampled_mean, spark_mvn_result.mean, decimal=0
        )

        # Check covariance
        sampled_cov = pdf[spark_mvn_result.column_names].cov().values
        np.testing.assert_array_almost_equal(
            sampled_cov, spark_mvn_result.cov, decimal=0
        )
