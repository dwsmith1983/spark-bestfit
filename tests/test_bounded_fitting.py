"""Tests for bounded/truncated distribution fitting (v1.4.0 feature).

These tests verify that:
1. Bounds are correctly auto-detected from data
2. Explicit bounds are used when provided
3. Partial bounds (one explicit, one auto) work
4. Samples respect bounds
5. Bounds survive serialization/deserialization
6. Bounds validation catches errors
7. Metrics are computed on truncated distributions
"""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit import DistributionFitter
from spark_bestfit.results import DistributionFitResult


class TestBoundedFitting:
    """Tests for bounded distribution fitting."""

    def test_bounded_auto_detect(self, spark_session):
        """Bounds are auto-detected from data min/max when bounded=True."""
        # Create data with known bounds
        np.random.seed(42)
        data = np.random.uniform(10, 90, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, column="value", bounded=True, max_distributions=5)

        best = results.best(n=1)[0]

        # Bounds should be auto-detected from data
        assert best.lower_bound is not None
        assert best.upper_bound is not None
        assert best.lower_bound == pytest.approx(data.min(), rel=0.01)
        assert best.upper_bound == pytest.approx(data.max(), rel=0.01)

    def test_bounded_explicit_bounds(self, spark_session):
        """User-specified bounds are used when provided."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=0.0,
            upper_bound=100.0,
            max_distributions=5,
        )

        best = results.best(n=1)[0]

        # Explicit bounds should be used
        assert best.lower_bound == 0.0
        assert best.upper_bound == 100.0

    def test_bounded_partial_bounds_lower_only(self, spark_session):
        """Lower bound explicit, upper bound auto-detected."""
        np.random.seed(42)
        data = np.random.exponential(10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=0.0,  # Explicit
            # upper_bound not specified - auto-detect
            max_distributions=5,
        )

        best = results.best(n=1)[0]

        assert best.lower_bound == 0.0
        assert best.upper_bound is not None
        assert best.upper_bound == pytest.approx(data.max(), rel=0.01)

    def test_bounded_partial_bounds_upper_only(self, spark_session):
        """Upper bound explicit, lower bound auto-detected."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            # lower_bound not specified - auto-detect
            upper_bound=100.0,  # Explicit
            max_distributions=5,
        )

        best = results.best(n=1)[0]

        assert best.lower_bound is not None
        assert best.lower_bound == pytest.approx(data.min(), rel=0.01)
        assert best.upper_bound == 100.0

    def test_bounded_result_sampling(self, spark_session):
        """Samples from bounded result respect bounds."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=20.0,
            upper_bound=80.0,
            max_distributions=5,
        )

        best = results.best(n=1)[0]

        # Sample from the bounded distribution
        samples = best.sample(size=10000, random_state=42)

        # All samples should be within bounds
        assert samples.min() >= 20.0
        assert samples.max() <= 80.0

    def test_bounded_pdf_cdf_ppf(self, spark_session):
        """PDF/CDF/PPF methods work correctly with bounded distributions."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=30.0,
            upper_bound=70.0,
            max_distributions=5,
        )

        best = results.best(n=1)[0]

        # PDF should be zero outside bounds
        x_outside_lower = np.array([20.0, 25.0])
        x_outside_upper = np.array([75.0, 80.0])
        x_inside = np.array([40.0, 50.0, 60.0])

        pdf_lower = best.pdf(x_outside_lower)
        pdf_upper = best.pdf(x_outside_upper)
        pdf_inside = best.pdf(x_inside)

        assert np.allclose(pdf_lower, 0.0)
        assert np.allclose(pdf_upper, 0.0)
        assert all(pdf_inside > 0)

        # CDF should be 0 at lower_bound and 1 at upper_bound
        assert best.cdf(np.array([30.0]))[0] == pytest.approx(0.0, abs=0.01)
        assert best.cdf(np.array([70.0]))[0] == pytest.approx(1.0, abs=0.01)

        # PPF should respect bounds (with small tolerance for floating point)
        quantiles = np.array([0.0, 0.5, 1.0])
        ppf_values = best.ppf(quantiles)
        assert ppf_values[0] >= 30.0 - 1e-6  # Allow tiny floating point error
        assert ppf_values[2] <= 70.0 + 1e-6

    def test_bounds_validation_error(self, spark_session):
        """Error when lower_bound >= upper_bound."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=1000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)

        with pytest.raises(ValueError, match="lower_bound.*must be less than"):
            fitter.fit(
                df,
                column="value",
                bounded=True,
                lower_bound=100.0,
                upper_bound=0.0,  # Invalid: lower >= upper
                max_distributions=5,
            )

        with pytest.raises(ValueError, match="lower_bound.*must be less than"):
            fitter.fit(
                df,
                column="value",
                bounded=True,
                lower_bound=50.0,
                upper_bound=50.0,  # Invalid: lower == upper
                max_distributions=5,
            )

    def test_unbounded_has_no_bounds(self, spark_session):
        """Unbounded fitting has None for bounds."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(df, column="value", max_distributions=5)

        best = results.best(n=1)[0]

        assert best.lower_bound is None
        assert best.upper_bound is None


class TestBoundedSerialization:
    """Tests for bounded distribution serialization."""

    def test_bounded_serialization_json(self, tmp_path):
        """Bounds survive JSON save/load."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            lower_bound=0.0,
            upper_bound=100.0,
        )

        json_path = tmp_path / "bounded_model.json"
        result.save(json_path)

        loaded = DistributionFitResult.load(json_path)

        assert loaded.lower_bound == 0.0
        assert loaded.upper_bound == 100.0
        assert loaded.distribution == "norm"

    def test_bounded_serialization_pickle(self, tmp_path):
        """Bounds survive pickle save/load."""
        result = DistributionFitResult(
            distribution="gamma",
            parameters=[2.0, 0.0, 5.0],
            sse=0.003,
            lower_bound=0.0,
            upper_bound=50.0,
        )

        pkl_path = tmp_path / "bounded_model.pkl"
        result.save(pkl_path)

        loaded = DistributionFitResult.load(pkl_path)

        assert loaded.lower_bound == 0.0
        assert loaded.upper_bound == 50.0
        assert loaded.distribution == "gamma"

    def test_bounded_to_dict(self):
        """to_dict() includes bounds."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            lower_bound=10.0,
            upper_bound=90.0,
        )

        d = result.to_dict()

        assert d["lower_bound"] == 10.0
        assert d["upper_bound"] == 90.0


class TestBoundedGetScipyDist:
    """Tests for get_scipy_dist() with bounded distributions."""

    def test_get_scipy_dist_unbounded(self):
        """Unbounded result returns frozen distribution."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        frozen = result.get_scipy_dist()

        # Should be a frozen distribution with correct parameters
        assert frozen.mean() == pytest.approx(50.0, rel=0.01)
        assert frozen.std() == pytest.approx(10.0, rel=0.01)

    def test_get_scipy_dist_bounded_returns_truncated(self):
        """Bounded result returns truncated distribution."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            lower_bound=30.0,
            upper_bound=70.0,
        )

        frozen = result.get_scipy_dist()

        # Sample should be within bounds
        samples = frozen.rvs(size=1000, random_state=42)
        assert samples.min() >= 30.0
        assert samples.max() <= 70.0

    def test_get_scipy_dist_frozen_false(self):
        """frozen=False returns unfrozen distribution class."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            lower_bound=30.0,
            upper_bound=70.0,
        )

        dist_class = result.get_scipy_dist(frozen=False)

        # Should be the distribution class, not frozen
        assert dist_class is st.norm


class TestBoundedRepr:
    """Tests for __repr__ with bounded distributions."""

    def test_repr_with_bounds(self):
        """__repr__ includes bounds when set."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            lower_bound=0.0,
            upper_bound=100.0,
        )

        repr_str = repr(result)

        assert "lower_bound=0.0000" in repr_str
        assert "upper_bound=100.0000" in repr_str

    def test_repr_without_bounds(self):
        """__repr__ excludes bounds when not set."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        repr_str = repr(result)

        assert "lower_bound" not in repr_str
        assert "upper_bound" not in repr_str


class TestBoundedEdgeCases:
    """Edge case tests for bounded distribution fitting."""

    def test_bounded_false_with_explicit_bounds_ignored(self, spark_session):
        """When bounded=False, explicit bounds should be ignored."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=3000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        # bounded=False but bounds provided - should be ignored
        results = fitter.fit(
            df,
            column="value",
            bounded=False,
            lower_bound=0.0,
            upper_bound=100.0,
            max_distributions=3,
        )

        best = results.best(n=1)[0]

        # Bounds should NOT be set when bounded=False
        assert best.lower_bound is None
        assert best.upper_bound is None

    def test_bounded_one_sided_lower_only(self, spark_session):
        """Test with only lower bound, upper at infinity."""
        np.random.seed(42)
        # Exponential data - naturally >= 0
        data = np.random.exponential(10, size=3000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=0.0,
            # upper_bound auto-detected
            max_distributions=3,
        )

        best = results.best(n=1)[0]

        assert best.lower_bound == 0.0
        assert best.upper_bound is not None

        # Samples should respect lower bound
        samples = best.sample(size=5000, random_state=42)
        assert samples.min() >= 0.0

    def test_bounded_wider_than_data(self, spark_session):
        """Bounds wider than data range should work correctly."""
        np.random.seed(42)
        data = np.random.uniform(40, 60, size=3000)  # Data in [40, 60]
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=0.0,   # Much wider than data
            upper_bound=100.0,
            max_distributions=3,
        )

        best = results.best(n=1)[0]

        assert best.lower_bound == 0.0
        assert best.upper_bound == 100.0

        # Samples can go beyond original data range but within bounds
        samples = best.sample(size=5000, random_state=42)
        assert samples.min() >= 0.0
        assert samples.max() <= 100.0

    def test_bounded_tight_bounds(self, spark_session):
        """Very tight bounds (small range) should work."""
        np.random.seed(42)
        data = np.random.uniform(49.5, 50.5, size=3000)  # Very tight range
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=49.0,
            upper_bound=51.0,
            max_distributions=3,
        )

        best = results.best(n=1)[0]

        # Samples should be within tight bounds
        samples = best.sample(size=5000, random_state=42)
        assert samples.min() >= 49.0
        assert samples.max() <= 51.0


class TestBoundedIntegration:
    """Integration tests for bounded distribution fitting."""

    def test_bounded_result_save_load_preserves_bounds(self, spark_session, tmp_path):
        """Bounded DistributionFitResult survives save/load through full fit flow."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=3000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=20.0,
            upper_bound=80.0,
            max_distributions=3,
        )

        # Get best and save it
        best_original = results.best(n=1)[0]
        json_path = tmp_path / "bounded_result.json"
        best_original.save(json_path)

        # Load and verify
        loaded = DistributionFitResult.load(json_path)

        assert loaded.lower_bound == best_original.lower_bound
        assert loaded.upper_bound == best_original.upper_bound
        assert loaded.lower_bound == 20.0
        assert loaded.upper_bound == 80.0

        # Samples from loaded should respect bounds
        samples = loaded.sample(size=1000, random_state=42)
        assert samples.min() >= 20.0
        assert samples.max() <= 80.0

    def test_bounded_spark_dataframe_roundtrip(self, spark_session):
        """Bounded results in Spark DataFrame preserve bounds through .best()."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=3000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        fitter = DistributionFitter(spark_session)
        results = fitter.fit(
            df,
            column="value",
            bounded=True,
            lower_bound=25.0,
            upper_bound=75.0,
            max_distributions=5,
        )

        # Access results through different methods
        all_results = results.best(n=5)

        # All results should have bounds set
        for result in all_results:
            assert result.lower_bound == 25.0
            assert result.upper_bound == 75.0

            # Sampling should respect bounds
            samples = result.sample(size=500, random_state=42)
            assert samples.min() >= 25.0
            assert samples.max() <= 75.0
