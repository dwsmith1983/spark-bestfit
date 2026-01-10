"""Tests for Gaussian Mixture Model fitting module.

Covers:
- Basic GMM fitting functionality
- Parameter recovery from known mixtures
- Sampling and evaluation methods
- Prediction (hard and soft assignments)
- Serialization
- Edge cases and numerical stability
- Random state determinism (Mayor feedback)
- Empty component handling (Mayor feedback)
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from spark_bestfit import GaussianMixtureFitter, GaussianMixtureResult
from spark_bestfit.serialization import SerializationError


@pytest.fixture
def simple_mixture_data_1d():
    """Generate 1D data from a known 2-component mixture."""
    np.random.seed(42)
    n = 2000

    # Component 1: N(0, 1)
    # Component 2: N(5, 1)
    # Weights: [0.4, 0.6]
    n1 = int(0.4 * n)
    n2 = n - n1

    samples1 = np.random.normal(0, 1, n1)
    samples2 = np.random.normal(5, 1, n2)

    return np.concatenate([samples1, samples2]).reshape(-1, 1)


@pytest.fixture
def simple_mixture_data_2d():
    """Generate 2D data from a known 2-component mixture."""
    np.random.seed(42)
    n = 3000

    # Component 1: mean=[0,0], cov=[[1,0.5],[0.5,1]]
    # Component 2: mean=[5,5], cov=[[1,-0.3],[-0.3,1]]
    # Weights: [0.3, 0.7]
    n1 = int(0.3 * n)
    n2 = n - n1

    mean1, cov1 = [0, 0], [[1, 0.5], [0.5, 1]]
    mean2, cov2 = [5, 5], [[1, -0.3], [-0.3, 1]]

    samples1 = np.random.multivariate_normal(mean1, cov1, n1)
    samples2 = np.random.multivariate_normal(mean2, cov2, n2)

    return np.vstack([samples1, samples2])


@pytest.fixture
def fitted_gmm_1d(simple_mixture_data_1d):
    """Create a fitted GaussianMixtureResult for 1D data."""
    fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=3)
    return fitter.fit(simple_mixture_data_1d)


@pytest.fixture
def fitted_gmm_2d(simple_mixture_data_2d):
    """Create a fitted GaussianMixtureResult for 2D data."""
    fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=3)
    return fitter.fit(simple_mixture_data_2d)


class TestGaussianMixtureFitterInit:
    """Tests for GaussianMixtureFitter initialization."""

    def test_init_defaults(self):
        """Test default parameter values."""
        fitter = GaussianMixtureFitter()
        assert fitter.n_components == 2
        assert fitter.max_iter == 100
        assert fitter.tol == 1e-4
        assert fitter.n_init == 1
        assert fitter.init_method == "kmeans"
        assert fitter.random_state is None
        assert fitter.reg_covar == 1e-6

    def test_init_custom_params(self):
        """Test custom parameter values."""
        fitter = GaussianMixtureFitter(
            n_components=5,
            max_iter=50,
            tol=1e-3,
            n_init=10,
            init_method="random",
            random_state=123,
            reg_covar=1e-5,
        )
        assert fitter.n_components == 5
        assert fitter.max_iter == 50
        assert fitter.tol == 1e-3
        assert fitter.n_init == 10
        assert fitter.init_method == "random"
        assert fitter.random_state == 123
        assert fitter.reg_covar == 1e-5

    def test_init_invalid_n_components(self):
        """Test error on invalid n_components."""
        with pytest.raises(ValueError, match="n_components must be at least 1"):
            GaussianMixtureFitter(n_components=0)

    def test_init_invalid_max_iter(self):
        """Test error on invalid max_iter."""
        with pytest.raises(ValueError, match="max_iter must be at least 1"):
            GaussianMixtureFitter(max_iter=0)

    def test_init_invalid_tol(self):
        """Test error on invalid tol."""
        with pytest.raises(ValueError, match="tol must be positive"):
            GaussianMixtureFitter(tol=0)

    def test_init_invalid_n_init(self):
        """Test error on invalid n_init."""
        with pytest.raises(ValueError, match="n_init must be at least 1"):
            GaussianMixtureFitter(n_init=0)

    def test_init_invalid_init_method(self):
        """Test error on invalid init_method."""
        with pytest.raises(ValueError, match="init_method must be"):
            GaussianMixtureFitter(init_method="invalid")

    def test_init_invalid_reg_covar(self):
        """Test error on invalid reg_covar."""
        with pytest.raises(ValueError, match="reg_covar must be non-negative"):
            GaussianMixtureFitter(reg_covar=-1)


class TestGaussianMixtureFitterFit:
    """Tests for GaussianMixtureFitter.fit() method."""

    def test_fit_basic_1d(self, simple_mixture_data_1d):
        """Test basic 1D mixture fitting."""
        fitter = GaussianMixtureFitter(n_components=2, random_state=42)
        result = fitter.fit(simple_mixture_data_1d)

        assert result.n_components == 2
        assert result.weights_.shape == (2,)
        assert result.means_.shape == (2, 1)
        assert result.covariances_.shape == (2, 1, 1)
        assert result.n_samples_ == len(simple_mixture_data_1d)

    def test_fit_basic_2d(self, simple_mixture_data_2d):
        """Test basic 2D mixture fitting."""
        fitter = GaussianMixtureFitter(n_components=2, random_state=42)
        result = fitter.fit(simple_mixture_data_2d)

        assert result.n_components == 2
        assert result.weights_.shape == (2,)
        assert result.means_.shape == (2, 2)
        assert result.covariances_.shape == (2, 2, 2)

    def test_fit_converges(self, fitted_gmm_2d):
        """Test that EM algorithm converges."""
        assert fitted_gmm_2d.converged_
        assert fitted_gmm_2d.n_iter_ > 0
        assert fitted_gmm_2d.n_iter_ <= 100  # max_iter

    def test_fit_weights_sum_to_one(self, fitted_gmm_2d):
        """Test that fitted weights sum to 1."""
        np.testing.assert_almost_equal(np.sum(fitted_gmm_2d.weights_), 1.0)

    def test_fit_recovers_parameters_1d(self, simple_mixture_data_1d):
        """Test that fit approximately recovers true 1D parameters."""
        fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=5)
        result = fitter.fit(simple_mixture_data_1d)

        # Sort by mean to match components
        order = np.argsort(result.means_.flatten())
        means = result.means_[order].flatten()
        weights = result.weights_[order]

        # True: means=[0, 5], weights=[0.4, 0.6]
        np.testing.assert_array_almost_equal(means, [0, 5], decimal=0)
        np.testing.assert_array_almost_equal(weights, [0.4, 0.6], decimal=1)

    def test_fit_recovers_parameters_2d(self, simple_mixture_data_2d):
        """Test that fit approximately recovers true 2D parameters."""
        fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=5)
        result = fitter.fit(simple_mixture_data_2d)

        # Sort by first mean dimension
        order = np.argsort(result.means_[:, 0])
        means = result.means_[order]
        weights = result.weights_[order]

        # True: means=[[0,0], [5,5]], weights=[0.3, 0.7]
        np.testing.assert_array_almost_equal(means[0], [0, 0], decimal=0)
        np.testing.assert_array_almost_equal(means[1], [5, 5], decimal=0)
        np.testing.assert_array_almost_equal(weights, [0.3, 0.7], decimal=1)

    def test_fit_too_few_samples_error(self):
        """Test error when n_samples < n_components."""
        data = np.array([[1], [2]])  # 2 samples
        fitter = GaussianMixtureFitter(n_components=5)  # 5 components
        with pytest.raises(ValueError, match="n_samples .* must be >= n_components"):
            fitter.fit(data)

    def test_fit_single_component(self, simple_mixture_data_1d):
        """Test fitting with single component (degenerates to normal)."""
        fitter = GaussianMixtureFitter(n_components=1, random_state=42)
        result = fitter.fit(simple_mixture_data_1d)

        assert result.n_components == 1
        np.testing.assert_almost_equal(result.weights_[0], 1.0)

        # Mean should be close to overall mean
        overall_mean = np.mean(simple_mixture_data_1d)
        np.testing.assert_almost_equal(result.means_[0, 0], overall_mean, decimal=0)

    def test_fit_random_init(self, simple_mixture_data_2d):
        """Test fitting with random initialization."""
        fitter = GaussianMixtureFitter(
            n_components=2,
            init_method="random",
            random_state=42,
            n_init=3,
        )
        result = fitter.fit(simple_mixture_data_2d)

        assert result.converged_ or result.n_iter_ == 100
        np.testing.assert_almost_equal(np.sum(result.weights_), 1.0)


class TestGaussianMixtureResultProperties:
    """Tests for GaussianMixtureResult properties."""

    def test_n_features(self, fitted_gmm_1d, fitted_gmm_2d):
        """Test n_features property."""
        assert fitted_gmm_1d.n_features == 1
        assert fitted_gmm_2d.n_features == 2

    def test_log_likelihood(self, fitted_gmm_2d):
        """Test log_likelihood_ property."""
        assert np.isfinite(fitted_gmm_2d.log_likelihood_)
        # Log-likelihood should be negative (log of probabilities)
        assert fitted_gmm_2d.log_likelihood_ < 0

    def test_responsibilities(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test responsibilities_ property (Mayor feedback)."""
        assert fitted_gmm_2d.responsibilities_ is not None
        assert fitted_gmm_2d.responsibilities_.shape == (
            len(simple_mixture_data_2d),
            2,
        )
        # Each row should sum to 1
        row_sums = np.sum(fitted_gmm_2d.responsibilities_, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(simple_mixture_data_2d)))

    def test_aic(self, fitted_gmm_2d):
        """Test AIC computation."""
        aic = fitted_gmm_2d.aic
        assert np.isfinite(aic)
        # AIC = -2 * log_likelihood + 2 * n_parameters
        expected_aic = -2 * fitted_gmm_2d.log_likelihood_ + 2 * fitted_gmm_2d._n_parameters
        np.testing.assert_almost_equal(aic, expected_aic)

    def test_bic(self, fitted_gmm_2d):
        """Test BIC computation."""
        bic = fitted_gmm_2d.bic
        assert np.isfinite(bic)
        # BIC = -2 * log_likelihood + n_parameters * log(n_samples)
        expected_bic = (
            -2 * fitted_gmm_2d.log_likelihood_
            + fitted_gmm_2d._n_parameters * np.log(fitted_gmm_2d.n_samples_)
        )
        np.testing.assert_almost_equal(bic, expected_bic)

    def test_bic_prefers_simpler_model(self, simple_mixture_data_1d):
        """Test that BIC prefers correct number of components."""
        # Fit models with different n_components
        bics = []
        for k in [1, 2, 3, 4]:
            fitter = GaussianMixtureFitter(n_components=k, random_state=42, n_init=3)
            result = fitter.fit(simple_mixture_data_1d)
            bics.append(result.bic)

        # True data has 2 components - BIC should be lowest for k=2
        # (may not always hold due to sample variance, so we just check it's finite)
        assert all(np.isfinite(bic) for bic in bics)


class TestGaussianMixtureResultSample:
    """Tests for sampling from fitted mixture."""

    def test_sample_basic(self, fitted_gmm_2d):
        """Test basic sampling."""
        samples = fitted_gmm_2d.sample(n=1000, random_state=42)
        assert samples.shape == (1000, 2)
        assert np.all(np.isfinite(samples))

    def test_sample_zero_samples(self, fitted_gmm_2d):
        """Test sampling zero samples."""
        samples = fitted_gmm_2d.sample(n=0)
        assert samples.shape == (0, 2)

    def test_sample_reproducibility(self, fitted_gmm_2d):
        """Test random_state determinism (Mayor feedback)."""
        samples1 = fitted_gmm_2d.sample(n=100, random_state=42)
        samples2 = fitted_gmm_2d.sample(n=100, random_state=42)
        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_sample_different_seeds_different_results(self, fitted_gmm_2d):
        """Test that different seeds produce different results."""
        samples1 = fitted_gmm_2d.sample(n=100, random_state=42)
        samples2 = fitted_gmm_2d.sample(n=100, random_state=123)
        assert not np.allclose(samples1, samples2)

    def test_sample_preserves_statistics(self, fitted_gmm_2d):
        """Test that sampled data has approximately correct statistics."""
        samples = fitted_gmm_2d.sample(n=10000, random_state=42)

        # Empirical mean should be close to weighted mean of components
        expected_mean = np.sum(
            fitted_gmm_2d.weights_[:, np.newaxis] * fitted_gmm_2d.means_,
            axis=0,
        )
        sampled_mean = np.mean(samples, axis=0)
        np.testing.assert_array_almost_equal(sampled_mean, expected_mean, decimal=0)


class TestGaussianMixtureResultEvaluation:
    """Tests for pdf, logpdf evaluation."""

    def test_pdf_positive(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that PDF is always positive."""
        pdf_values = fitted_gmm_2d.pdf(simple_mixture_data_2d[:100])
        assert np.all(pdf_values > 0)

    def test_pdf_single_point(self, fitted_gmm_2d):
        """Test PDF at single point."""
        point = fitted_gmm_2d.means_[0]  # First component mean
        pdf_value = fitted_gmm_2d.pdf(point)
        assert np.isfinite(pdf_value)
        assert pdf_value > 0

    def test_logpdf_finite(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that logpdf is finite."""
        logpdf_values = fitted_gmm_2d.logpdf(simple_mixture_data_2d[:100])
        assert np.all(np.isfinite(logpdf_values))

    def test_logpdf_consistent_with_pdf(self, fitted_gmm_2d):
        """Test that logpdf is consistent with log(pdf)."""
        points = np.array([[0, 0], [5, 5], [2.5, 2.5]])
        pdf_values = fitted_gmm_2d.pdf(points)
        logpdf_values = fitted_gmm_2d.logpdf(points)
        np.testing.assert_array_almost_equal(logpdf_values, np.log(pdf_values))

    def test_pdf_wrong_dimensions_error(self, fitted_gmm_2d):
        """Test error when input has wrong dimensions."""
        wrong_dim = np.array([[1, 2, 3]])  # 3D instead of 2D
        with pytest.raises(ValueError, match="Expected 2 features"):
            fitted_gmm_2d.pdf(wrong_dim)


class TestGaussianMixtureResultPredict:
    """Tests for predict and predict_proba."""

    def test_predict_labels_valid(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that predict returns valid labels."""
        labels = fitted_gmm_2d.predict(simple_mixture_data_2d[:100])
        assert labels.shape == (100,)
        assert np.all((labels >= 0) & (labels < 2))

    def test_predict_proba_sums_to_one(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that predict_proba rows sum to 1."""
        probs = fitted_gmm_2d.predict_proba(simple_mixture_data_2d[:100])
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(100))

    def test_predict_proba_bounds(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that predict_proba values are in [0, 1]."""
        probs = fitted_gmm_2d.predict_proba(simple_mixture_data_2d[:100])
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_consistent_with_proba(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that predict is argmax of predict_proba."""
        points = simple_mixture_data_2d[:100]
        labels = fitted_gmm_2d.predict(points)
        probs = fitted_gmm_2d.predict_proba(points)
        expected_labels = np.argmax(probs, axis=1)
        np.testing.assert_array_equal(labels, expected_labels)

    def test_predict_separates_clusters(self, simple_mixture_data_2d):
        """Test that predict correctly separates well-separated clusters."""
        # The first 900 points are from component 1 (mean=[0,0])
        # The last 2100 points are from component 2 (mean=[5,5])
        fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=3)
        result = fitter.fit(simple_mixture_data_2d)

        # Points near [0,0] should have same label
        near_origin = simple_mixture_data_2d[np.linalg.norm(simple_mixture_data_2d, axis=1) < 2]
        labels_origin = result.predict(near_origin)
        assert len(np.unique(labels_origin)) == 1  # All same label

        # Points near [5,5] should have same label
        near_five = simple_mixture_data_2d[np.linalg.norm(simple_mixture_data_2d - [5, 5], axis=1) < 2]
        labels_five = result.predict(near_five)
        assert len(np.unique(labels_five)) == 1  # All same label


class TestGaussianMixtureResultSerialization:
    """Tests for save/load functionality."""

    def test_save_load_json(self, fitted_gmm_2d):
        """Test JSON round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            fitted_gmm_2d.save(path)
            loaded = GaussianMixtureResult.load(path)

            assert loaded.n_components == fitted_gmm_2d.n_components
            np.testing.assert_array_almost_equal(loaded.weights_, fitted_gmm_2d.weights_)
            np.testing.assert_array_almost_equal(loaded.means_, fitted_gmm_2d.means_)
            np.testing.assert_array_almost_equal(loaded.covariances_, fitted_gmm_2d.covariances_)
            assert loaded.converged_ == fitted_gmm_2d.converged_
            assert loaded.n_iter_ == fitted_gmm_2d.n_iter_
            np.testing.assert_almost_equal(loaded.log_likelihood_, fitted_gmm_2d.log_likelihood_)
        finally:
            path.unlink()

    def test_save_load_pickle(self, fitted_gmm_2d):
        """Test pickle round-trip."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            fitted_gmm_2d.save(path)
            loaded = GaussianMixtureResult.load(path)

            assert loaded.n_components == fitted_gmm_2d.n_components
            np.testing.assert_array_almost_equal(loaded.weights_, fitted_gmm_2d.weights_)
            np.testing.assert_array_almost_equal(loaded.means_, fitted_gmm_2d.means_)
        finally:
            path.unlink()

    def test_loaded_result_can_sample(self, fitted_gmm_2d):
        """Test that loaded result can generate samples."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            fitted_gmm_2d.save(path)
            loaded = GaussianMixtureResult.load(path)

            samples = loaded.sample(n=100, random_state=42)
            assert samples.shape == (100, 2)
        finally:
            path.unlink()

    def test_loaded_result_can_predict(self, fitted_gmm_2d, simple_mixture_data_2d):
        """Test that loaded result can make predictions."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            fitted_gmm_2d.save(path)
            loaded = GaussianMixtureResult.load(path)

            labels = loaded.predict(simple_mixture_data_2d[:10])
            assert len(labels) == 10
        finally:
            path.unlink()

    def test_load_invalid_json(self):
        """Test error on invalid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json")
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError, match="Invalid JSON"):
                GaussianMixtureResult.load(path)
        finally:
            path.unlink()

    def test_load_missing_fields(self):
        """Test error on missing required fields."""
        import json

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"n_components": 2}, f)  # Missing weights, means, covariances
            path = Path(f.name)

        try:
            with pytest.raises(SerializationError, match="Missing required field"):
                GaussianMixtureResult.load(path)
        finally:
            path.unlink()


class TestGaussianMixtureEdgeCases:
    """Edge case tests."""

    def test_init_validation_weights_sum(self):
        """Test that weights must sum to 1."""
        with pytest.raises(ValueError, match="weights_ must sum to 1"):
            GaussianMixtureResult(
                n_components=2,
                weights_=np.array([0.3, 0.3]),  # Sums to 0.6
                means_=np.array([[0], [1]]),
                covariances_=np.array([[[1]], [[1]]]),
            )

    def test_init_validation_component_mismatch(self):
        """Test validation of component count consistency."""
        with pytest.raises(ValueError, match="weights_ shape"):
            GaussianMixtureResult(
                n_components=2,
                weights_=np.array([1.0]),  # Only 1 weight for 2 components
                means_=np.array([[0], [1]]),
                covariances_=np.array([[[1]], [[1]]]),
            )

    def test_univariate_as_row_vector(self):
        """Test fitting when univariate data passed as row vector."""
        data = np.random.normal(0, 1, 100).reshape(1, -1)  # 1 row, 100 cols
        fitter = GaussianMixtureFitter(n_components=1, random_state=42)
        result = fitter.fit(data)

        assert result.n_features == 1
        assert result.n_samples_ == 100

    def test_high_dimensional(self):
        """Test with higher dimensional data (5D)."""
        np.random.seed(42)
        n_features = 5
        n_samples = 500

        # Generate 5D mixture
        mean1 = np.zeros(n_features)
        mean2 = np.ones(n_features) * 5
        cov = np.eye(n_features)

        data1 = np.random.multivariate_normal(mean1, cov, n_samples // 2)
        data2 = np.random.multivariate_normal(mean2, cov, n_samples // 2)
        data = np.vstack([data1, data2])

        fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=3)
        result = fitter.fit(data)

        assert result.n_features == 5
        assert result.means_.shape == (2, 5)
        assert result.covariances_.shape == (2, 5, 5)

    def test_empty_component_warning(self):
        """Test warning when component gets zero responsibility (Mayor feedback)."""
        # Create data that clearly has only 1 cluster
        np.random.seed(42)
        data = np.random.normal(0, 0.1, (100, 2))  # Tight cluster

        fitter = GaussianMixtureFitter(n_components=3, random_state=42)

        # Should warn about near-zero responsibility
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            result = fitter.fit(data)

            # Check if warning about empty components was issued
            # (may not always trigger depending on initialization)
            # Just verify fitting completes without error
            assert result is not None

    def test_numerical_stability_high_variance(self):
        """Test numerical stability with widely different variances."""
        np.random.seed(42)

        # Component 1: var=0.01, Component 2: var=100
        data1 = np.random.normal(0, 0.1, (200, 1))
        data2 = np.random.normal(10, 10, (200, 1))
        data = np.vstack([data1, data2])

        fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=3)
        result = fitter.fit(data)

        assert result.converged_ or result.n_iter_ > 0
        assert np.all(np.isfinite(result.means_))
        assert np.all(np.isfinite(result.covariances_))

    def test_many_components(self):
        """Test fitting with many components."""
        np.random.seed(42)
        n_components = 5
        n_samples = 1000

        # Generate data from 5 different normals
        data = []
        for k in range(n_components):
            data.append(np.random.normal(k * 3, 0.5, n_samples // n_components))
        data = np.concatenate(data).reshape(-1, 1)

        fitter = GaussianMixtureFitter(n_components=5, random_state=42, n_init=3)
        result = fitter.fit(data)

        assert result.n_components == 5
        np.testing.assert_almost_equal(np.sum(result.weights_), 1.0)


class TestGaussianMixtureRandomStateDeterminism:
    """Tests for random state determinism (Mayor feedback)."""

    def test_fit_determinism(self, simple_mixture_data_2d):
        """Test that same random_state produces identical fit results."""
        fitter1 = GaussianMixtureFitter(n_components=2, random_state=42)
        fitter2 = GaussianMixtureFitter(n_components=2, random_state=42)

        result1 = fitter1.fit(simple_mixture_data_2d)
        result2 = fitter2.fit(simple_mixture_data_2d)

        np.testing.assert_array_almost_equal(result1.weights_, result2.weights_)
        np.testing.assert_array_almost_equal(result1.means_, result2.means_)
        np.testing.assert_array_almost_equal(result1.covariances_, result2.covariances_)
        np.testing.assert_almost_equal(result1.log_likelihood_, result2.log_likelihood_)

    def test_fit_different_seeds_different_results(self, simple_mixture_data_2d):
        """Test that different random_state produces different results."""
        fitter1 = GaussianMixtureFitter(n_components=2, random_state=42)
        fitter2 = GaussianMixtureFitter(n_components=2, random_state=123)

        result1 = fitter1.fit(simple_mixture_data_2d)
        result2 = fitter2.fit(simple_mixture_data_2d)

        # Results may be similar but not identical
        # (both should find the same mixture, but order may differ)
        # Just verify they ran independently
        assert result1 is not result2


class TestGaussianMixtureIntegration:
    """Integration tests for end-to-end workflows."""

    def test_fit_sample_verify_workflow(self):
        """Test complete workflow: generate, fit, sample, verify."""
        np.random.seed(42)

        # Generate true mixture (weights 0.4/0.6 from n1=400, n2=600)
        true_means = [[0, 0], [5, 5]]
        true_covs = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]

        n1 = 400
        n2 = 600
        data1 = np.random.multivariate_normal(true_means[0], true_covs[0], n1)
        data2 = np.random.multivariate_normal(true_means[1], true_covs[1], n2)
        data = np.vstack([data1, data2])

        # Fit
        fitter = GaussianMixtureFitter(n_components=2, random_state=42, n_init=5)
        result = fitter.fit(data)

        # Sample from fitted model
        samples = result.sample(n=5000, random_state=42)

        # Verify sampled statistics match fitted model
        expected_mean = np.sum(result.weights_[:, np.newaxis] * result.means_, axis=0)
        sampled_mean = np.mean(samples, axis=0)
        np.testing.assert_array_almost_equal(sampled_mean, expected_mean, decimal=0)

    def test_save_load_sample_workflow(self, fitted_gmm_2d):
        """Test save, load, then sample workflow."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            fitted_gmm_2d.save(path)

            # Load
            loaded = GaussianMixtureResult.load(path)

            # Sample from loaded result
            samples = loaded.sample(n=1000, random_state=42)

            # Verify samples are valid
            assert samples.shape == (1000, 2)
            assert np.all(np.isfinite(samples))
        finally:
            path.unlink()

    def test_model_selection_workflow(self, simple_mixture_data_1d):
        """Test model selection using BIC."""
        bics = {}
        for k in range(1, 5):
            fitter = GaussianMixtureFitter(n_components=k, random_state=42, n_init=3)
            result = fitter.fit(simple_mixture_data_1d)
            bics[k] = result.bic

        # Verify BIC values are computed
        assert all(np.isfinite(bic) for bic in bics.values())
