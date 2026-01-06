"""Tests for bounded/truncated distribution fitting (v1.4.0, v1.5.0 features).

These tests verify that:
1. Bounds are correctly auto-detected from data
2. Explicit bounds are used when provided
3. Partial bounds (one explicit, one auto) work
4. Samples respect bounds
5. Bounds survive serialization/deserialization
6. Bounds validation catches errors
7. Metrics are computed on truncated distributions
8. Per-column bounds work for multi-column fitting (v1.5.0)

Uses LocalBackend for most tests. Spark-specific tests are in test_spark_backend.py.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from spark_bestfit import DistributionFitter
from spark_bestfit.results import DistributionFitResult


class TestBoundedFitting:
    """Tests for bounded distribution fitting."""

    def test_bounded_auto_detect(self, local_backend):
        """Bounds are auto-detected from data min/max when bounded=True."""
        # Create data with known bounds
        np.random.seed(42)
        data = np.random.uniform(10, 90, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(df, column="value", bounded=True, max_distributions=5)

        best = results.best(n=1)[0]

        # Bounds should be auto-detected from data
        assert best.lower_bound is not None
        assert best.upper_bound is not None
        assert best.lower_bound == pytest.approx(data.min(), rel=0.01)
        assert best.upper_bound == pytest.approx(data.max(), rel=0.01)

    def test_bounded_explicit_bounds(self, local_backend):
        """User-specified bounds are used when provided."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_partial_bounds_lower_only(self, local_backend):
        """Lower bound explicit, upper bound auto-detected."""
        np.random.seed(42)
        data = np.random.exponential(10, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_partial_bounds_upper_only(self, local_backend):
        """Upper bound explicit, lower bound auto-detected."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_result_sampling(self, local_backend):
        """Samples from bounded result respect bounds."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_pdf_cdf_ppf(self, local_backend):
        """PDF/CDF/PPF methods work correctly with bounded distributions."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounds_validation_error(self, local_backend):
        """Error when lower_bound >= upper_bound."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=1000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

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

    def test_unbounded_has_no_bounds(self, local_backend):
        """Unbounded fitting has None for bounds."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=5000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_false_with_explicit_bounds_ignored(self, local_backend):
        """When bounded=False, explicit bounds should be ignored."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=3000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_one_sided_lower_only(self, local_backend):
        """Test with only lower bound, upper at infinity."""
        np.random.seed(42)
        # Exponential data - naturally >= 0
        data = np.random.exponential(10, size=3000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_wider_than_data(self, local_backend):
        """Bounds wider than data range should work correctly."""
        np.random.seed(42)
        data = np.random.uniform(40, 60, size=3000)  # Data in [40, 60]
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_tight_bounds(self, local_backend):
        """Very tight bounds (small range) should work."""
        np.random.seed(42)
        data = np.random.uniform(49.5, 50.5, size=3000)  # Very tight range
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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


class TestDiscreteBoundedFitting:
    """Tests for discrete bounded distribution fitting."""

    def test_discrete_bounded_auto_detect(self, local_backend):
        """DiscreteDistributionFitter with bounded=True auto-detects bounds."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        data = np.random.poisson(lam=10, size=3000)
        df = pd.DataFrame({"count": data})

        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            column="count",
            bounded=True,
            max_distributions=3,
        )

        best = results.best(n=1, metric="aic")[0]

        # Bounds should be auto-detected
        assert best.lower_bound is not None
        assert best.upper_bound is not None
        assert best.lower_bound == float(data.min())
        assert best.upper_bound == float(data.max())

    def test_discrete_bounded_explicit(self, local_backend):
        """DiscreteDistributionFitter with explicit bounds."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        data = np.random.poisson(lam=10, size=3000)
        df = pd.DataFrame({"count": data})

        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            column="count",
            bounded=True,
            lower_bound=0,
            upper_bound=50,
            max_distributions=3,
        )

        best = results.best(n=1, metric="aic")[0]

        assert best.lower_bound == 0.0
        assert best.upper_bound == 50.0

    def test_discrete_unbounded_has_no_bounds(self, local_backend):
        """DiscreteDistributionFitter without bounded=True has None bounds."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        data = np.random.poisson(lam=10, size=3000)
        df = pd.DataFrame({"count": data})

        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(df, column="count", max_distributions=3)

        best = results.best(n=1, metric="aic")[0]

        assert best.lower_bound is None
        assert best.upper_bound is None

    def test_discrete_bounds_validation_error(self, local_backend):
        """DiscreteDistributionFitter raises error when lower >= upper."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        data = np.random.poisson(lam=10, size=1000)
        df = pd.DataFrame({"count": data})

        fitter = DiscreteDistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="lower_bound.*must be less than"):
            fitter.fit(
                df,
                column="count",
                bounded=True,
                lower_bound=50,
                upper_bound=10,
                max_distributions=3,
            )


class TestBoundedIntegration:
    """Integration tests for bounded distribution fitting."""

    def test_bounded_result_save_load_preserves_bounds(self, local_backend, tmp_path):
        """Bounded DistributionFitResult survives save/load through full fit flow."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=3000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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

    def test_bounded_results_preserve_bounds_through_best(self, local_backend):
        """Bounded results preserve bounds through .best()."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=3000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)
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


class TestPerColumnBounds:
    """Tests for per-column bounds (v1.5.0 feature)."""

    def test_per_column_bounds_dict(self, local_backend):
        """Dict bounds allow different bounds per column."""
        np.random.seed(42)
        # Column 1: uniform in [0, 100]
        # Column 2: uniform in [-50, 50]
        data1 = np.random.uniform(10, 90, size=3000)
        data2 = np.random.uniform(-40, 40, size=3000)
        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["col1", "col2"],
            bounded=True,
            lower_bound={"col1": 0.0, "col2": -50.0},
            upper_bound={"col1": 100.0, "col2": 50.0},
            max_distributions=3,
        )

        # Check col1 bounds
        col1_results = results.for_column("col1").best(n=1)[0]
        assert col1_results.lower_bound == 0.0
        assert col1_results.upper_bound == 100.0

        # Check col2 bounds
        col2_results = results.for_column("col2").best(n=1)[0]
        assert col2_results.lower_bound == -50.0
        assert col2_results.upper_bound == 50.0

    def test_per_column_bounds_auto_detect(self, local_backend):
        """Auto-detect bounds per-column when bounded=True and no explicit bounds."""
        np.random.seed(42)
        data1 = np.random.uniform(10, 90, size=3000)
        data2 = np.random.uniform(-40, 40, size=3000)
        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["col1", "col2"],
            bounded=True,
            max_distributions=3,
        )

        # Each column should have its own auto-detected bounds
        col1_results = results.for_column("col1").best(n=1)[0]
        assert col1_results.lower_bound == pytest.approx(data1.min(), rel=0.01)
        assert col1_results.upper_bound == pytest.approx(data1.max(), rel=0.01)

        col2_results = results.for_column("col2").best(n=1)[0]
        assert col2_results.lower_bound == pytest.approx(data2.min(), rel=0.01)
        assert col2_results.upper_bound == pytest.approx(data2.max(), rel=0.01)

    def test_per_column_bounds_partial_dict(self, local_backend):
        """Partial dict: specify some columns, auto-detect others."""
        np.random.seed(42)
        data1 = np.random.uniform(10, 90, size=3000)
        data2 = np.random.uniform(-40, 40, size=3000)
        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["col1", "col2"],
            bounded=True,
            lower_bound={"col1": 0.0},  # Only col1 lower bound specified
            upper_bound={"col2": 100.0},  # Only col2 upper bound specified
            max_distributions=3,
        )

        # col1: explicit lower, auto-detect upper
        col1_results = results.for_column("col1").best(n=1)[0]
        assert col1_results.lower_bound == 0.0
        assert col1_results.upper_bound == pytest.approx(data1.max(), rel=0.01)

        # col2: auto-detect lower, explicit upper
        col2_results = results.for_column("col2").best(n=1)[0]
        assert col2_results.lower_bound == pytest.approx(data2.min(), rel=0.01)
        assert col2_results.upper_bound == 100.0

    def test_per_column_bounds_scalar_applies_to_all(self, local_backend):
        """Scalar bounds apply to all columns (backward compatible)."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, size=3000)
        data2 = np.random.normal(50, 10, size=3000)
        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["col1", "col2"],
            bounded=True,
            lower_bound=0.0,  # Scalar - applies to all
            upper_bound=100.0,  # Scalar - applies to all
            max_distributions=3,
        )

        # Both columns should have same bounds
        col1_results = results.for_column("col1").best(n=1)[0]
        col2_results = results.for_column("col2").best(n=1)[0]

        assert col1_results.lower_bound == 0.0
        assert col1_results.upper_bound == 100.0
        assert col2_results.lower_bound == 0.0
        assert col2_results.upper_bound == 100.0

    def test_per_column_bounds_unknown_column_error(self, local_backend):
        """Dict with unknown column name raises ValueError."""
        np.random.seed(42)
        data = np.random.normal(50, 10, size=1000)
        df = pd.DataFrame({"value": data})

        fitter = DistributionFitter(backend=local_backend)

        with pytest.raises(ValueError, match="unknown columns"):
            fitter.fit(
                df,
                column="value",
                bounded=True,
                lower_bound={"nonexistent": 0.0},  # Unknown column
                max_distributions=3,
            )

        with pytest.raises(ValueError, match="unknown columns"):
            fitter.fit(
                df,
                column="value",
                bounded=True,
                upper_bound={"typo_column": 100.0},  # Unknown column
                max_distributions=3,
            )

    def test_per_column_bounds_validation_per_column(self, local_backend):
        """lower >= upper error is per-column."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, size=1000)
        data2 = np.random.normal(50, 10, size=1000)
        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)

        # col1 has valid bounds, col2 has invalid bounds
        with pytest.raises(ValueError, match="lower_bound.*must be less than.*col2"):
            fitter.fit(
                df,
                columns=["col1", "col2"],
                bounded=True,
                lower_bound={"col1": 0.0, "col2": 100.0},
                upper_bound={"col1": 100.0, "col2": 50.0},  # col2: lower > upper
                max_distributions=3,
            )

    def test_per_column_bounds_samples_respect_column_bounds(self, local_backend):
        """Samples from each column respect that column's bounds."""
        np.random.seed(42)
        data1 = np.random.normal(50, 10, size=3000)
        data2 = np.random.normal(0, 5, size=3000)
        df = pd.DataFrame({"col1": data1, "col2": data2})

        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["col1", "col2"],
            bounded=True,
            lower_bound={"col1": 20.0, "col2": -15.0},
            upper_bound={"col1": 80.0, "col2": 15.0},
            max_distributions=3,
        )

        # Samples from col1 respect col1's bounds
        col1_best = results.for_column("col1").best(n=1)[0]
        samples1 = col1_best.sample(size=5000, random_state=42)
        assert samples1.min() >= 20.0
        assert samples1.max() <= 80.0

        # Samples from col2 respect col2's bounds
        col2_best = results.for_column("col2").best(n=1)[0]
        samples2 = col2_best.sample(size=5000, random_state=42)
        assert samples2.min() >= -15.0
        assert samples2.max() <= 15.0


class TestDiscretePerColumnBounds:
    """Tests for per-column bounds with DiscreteDistributionFitter (v1.5.0)."""

    def test_discrete_per_column_bounds_dict(self, local_backend):
        """DiscreteDistributionFitter supports per-column bounds dicts."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        data1 = np.random.poisson(lam=10, size=3000)
        data2 = np.random.poisson(lam=20, size=3000)
        df = pd.DataFrame({"count1": data1, "count2": data2})

        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["count1", "count2"],
            bounded=True,
            lower_bound={"count1": 0, "count2": 5},
            upper_bound={"count1": 30, "count2": 50},
            max_distributions=3,
        )

        # Check count1 bounds
        count1_results = results.for_column("count1").best(n=1, metric="aic")[0]
        assert count1_results.lower_bound == 0.0
        assert count1_results.upper_bound == 30.0

        # Check count2 bounds
        count2_results = results.for_column("count2").best(n=1, metric="aic")[0]
        assert count2_results.lower_bound == 5.0
        assert count2_results.upper_bound == 50.0

    def test_discrete_per_column_auto_detect(self, local_backend):
        """DiscreteDistributionFitter auto-detects per-column bounds."""
        from spark_bestfit import DiscreteDistributionFitter

        np.random.seed(42)
        data1 = np.random.poisson(lam=5, size=3000)
        data2 = np.random.poisson(lam=15, size=3000)
        df = pd.DataFrame({"count1": data1, "count2": data2})

        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(
            df,
            columns=["count1", "count2"],
            bounded=True,
            max_distributions=3,
        )

        # Each column has its own auto-detected bounds
        count1_best = results.for_column("count1").best(n=1, metric="aic")[0]
        assert count1_best.lower_bound == float(data1.min())
        assert count1_best.upper_bound == float(data1.max())

        count2_best = results.for_column("count2").best(n=1, metric="aic")[0]
        assert count2_best.lower_bound == float(data2.min())
        assert count2_best.upper_bound == float(data2.max())
