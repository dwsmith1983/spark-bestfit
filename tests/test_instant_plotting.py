"""Tests for instant plotting via cached sample on DistributionFitResult.

These tests cover the fix for DAG recomputation when calling plot(), plot_qq(),
and plot_pp() after fit(). The fix caches the numpy sample during fit() and
threads it through to DistributionFitResult as `cached_sample`.

All tests in this file are written FIRST (RED phase of TDD). They will fail
against the current code because:
  - The field is currently named `sample`, not `cached_sample`
  - The `sample` field shadows the `sample()` method on the dataclass
  - Plot methods check `result.sample` instead of `result.cached_sample`
  - `to_dict()` serializes the entire sample array (should be excluded)
  - Discrete fitter caches float array instead of int array
  - `LazyFitResults.materialize()` drops the samples dict
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend - must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spark_bestfit import DiscreteDistributionFitter, DistributionFitter
from spark_bestfit.backends.local import LocalBackend
from spark_bestfit.collection import EagerFitResults, LazyFitResults, create_fit_results
from spark_bestfit.storage import DistributionFitResult, LazyMetricsContext


# ---------------------------------------------------------------------------
# Bug 1: `sample` field shadows `sample()` method
# ---------------------------------------------------------------------------


class TestCachedSampleFieldVsMethod:
    """Tests that the renamed field `cached_sample` does not shadow `sample()`."""

    def test_cached_sample_field_does_not_shadow_sample_method(self):
        """cached_sample field and sample() method must be distinct attributes.

        FAILS currently because the field is named `sample`, which overwrites
        the `sample()` method defined on DistributionFitResult (slots=True
        dataclass - the field assignment replaces the method in the slot).
        """
        arr = np.array([1.0, 2.0, 3.0])
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            cached_sample=arr,  # new field name
        )

        # Field must hold the array
        assert result.cached_sample is not None
        np.testing.assert_array_equal(result.cached_sample, arr)

        # sample() method must still be callable and return a numpy array
        assert callable(result.sample), "result.sample should be a method, not an array"
        generated = result.sample(size=5)
        assert isinstance(generated, np.ndarray)
        assert len(generated) == 5

    def test_cached_sample_defaults_to_none(self):
        """cached_sample should default to None when not provided."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )
        assert result.cached_sample is None

    def test_sample_method_callable_without_cached_sample(self):
        """sample() method works when cached_sample is not set."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )
        samples = result.sample(size=10)
        assert isinstance(samples, np.ndarray)
        assert len(samples) == 10


# ---------------------------------------------------------------------------
# Bug 1 continued: best() must propagate cached_sample
# ---------------------------------------------------------------------------


class TestFitResultsBestPreservesCachedSample:
    """Tests that best() propagates cached_sample from the fit cache."""

    def test_fit_results_best_preserves_cached_sample(self, pandas_dataset, local_backend):
        """After fit(), best()[0].cached_sample must be a numpy array.

        FAILS currently because the field is named `sample` not `cached_sample`,
        so accessing `.cached_sample` raises AttributeError.
        """
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=3)
        best = results.best(n=1, metric="sse")

        assert len(best) == 1
        result = best[0]

        assert result.cached_sample is not None, "cached_sample must be populated after fit()"
        assert isinstance(result.cached_sample, np.ndarray)
        assert len(result.cached_sample) > 0

    def test_fit_results_best_n_all_have_cached_sample(self, pandas_dataset, local_backend):
        """All results returned by best(n=3) must have cached_sample populated."""
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=5)
        top3 = results.best(n=3, metric="sse")

        for r in top3:
            assert r.cached_sample is not None
            assert isinstance(r.cached_sample, np.ndarray)


# ---------------------------------------------------------------------------
# Bug 1 continued: instant plot methods use cached_sample
# ---------------------------------------------------------------------------


class TestInstantPlotUsesCachedSample:
    """Tests that plot(), plot_qq(), plot_pp() use cached_sample when df is absent."""

    @pytest.fixture
    def result_with_cached_sample(self, normal_data):
        """DistributionFitResult with cached_sample set to normal data."""
        return DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            cached_sample=normal_data,
        )

    def test_instant_plot_uses_cached_sample(self, result_with_cached_sample):
        """plot() without df must use cached_sample and return a figure.

        FAILS currently because plot() checks `result.sample` (the method reference
        after the field shadows it) which is not None but is the numpy array, and the
        logic `if result.sample is not None` is checking the *field* named `sample`.
        After renaming to `cached_sample`, the check must be `result.cached_sample`.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        fig, ax = fitter.plot(result_with_cached_sample)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_instant_plot_qq_uses_cached_sample(self, result_with_cached_sample):
        """plot_qq() without df must use cached_sample and return a figure.

        FAILS currently because plot_qq() checks `result.sample` not `result.cached_sample`.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        fig, ax = fitter.plot_qq(result_with_cached_sample)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_instant_plot_pp_uses_cached_sample(self, result_with_cached_sample):
        """plot_pp() without df must use cached_sample and return a figure.

        FAILS currently because plot_pp() checks `result.sample` not `result.cached_sample`.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        fig, ax = fitter.plot_pp(result_with_cached_sample)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_instant_plot_raises_without_df_or_cached_sample(self):
        """plot() without df AND without cached_sample must raise ValueError.

        This verifies the error path is still intact after the rename.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        result_no_sample = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            # cached_sample intentionally omitted — defaults to None
        )

        with pytest.raises(ValueError, match="(?i)(df|sample|cached_sample)"):
            fitter.plot(result_no_sample)

    def test_instant_plot_qq_raises_without_df_or_cached_sample(self):
        """plot_qq() without df AND without cached_sample must raise ValueError."""
        fitter = DistributionFitter(backend=LocalBackend())
        result_no_sample = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        with pytest.raises(ValueError, match="(?i)(df|sample|cached_sample)"):
            fitter.plot_qq(result_no_sample)

    def test_instant_plot_pp_raises_without_df_or_cached_sample(self):
        """plot_pp() without df AND without cached_sample must raise ValueError."""
        fitter = DistributionFitter(backend=LocalBackend())
        result_no_sample = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        with pytest.raises(ValueError, match="(?i)(df|sample|cached_sample)"):
            fitter.plot_pp(result_no_sample)

    def test_plot_with_df_prefers_df(self, pandas_dataset, normal_data):
        """plot() with both df and cached_sample must complete without error.

        When df is provided, the Spark/pandas path is used for the histogram.
        The test verifies that passing df alongside a cached sample does not break.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name="value",
            cached_sample=normal_data,
        )

        # Providing df should work without error (uses df path for histogram)
        fig, ax = fitter.plot(result, df=pandas_dataset, column="value")
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# Bug 4: to_dict() must NOT serialize cached_sample
# ---------------------------------------------------------------------------


class TestCachedSampleExcludedFromSerialization:
    """Tests that to_dict() does not include the cached_sample array."""

    def test_cached_sample_excluded_from_to_dict(self):
        """to_dict() must not include `cached_sample` key.

        FAILS currently because to_dict() includes `sample` key with a
        10,000-element list, which bloats JSON serialization.
        """
        arr = np.random.normal(50, 10, 500)
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            cached_sample=arr,
        )

        d = result.to_dict()

        assert "cached_sample" not in d, "cached_sample must be excluded from to_dict()"

    def test_to_dict_sample_key_absent_or_none(self):
        """to_dict() must not serialize the sample array under any key.

        After the fix, 'sample' key should either be absent entirely or
        present with value None (not the array). The large array must not
        appear in the serialized form.
        """
        arr = np.random.normal(50, 10, 10_000)
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            cached_sample=arr,
        )

        d = result.to_dict()

        # If a 'sample' key exists it must not hold a large list
        if "sample" in d:
            assert d["sample"] is None or (isinstance(d["sample"], list) and len(d["sample"]) == 0), (
                "to_dict() must not serialize the cached sample array; "
                f"got sample with {len(d['sample'])} elements"
            )

    def test_to_dict_without_cached_sample_is_serializable(self):
        """to_dict() on a result without cached_sample must produce JSON-serializable output."""
        import json

        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            aic=1500.0,
            bic=1520.0,
        )

        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Bug 2: Discrete fitter must cache integer sample
# ---------------------------------------------------------------------------


class TestDiscreteCachedSampleIsInteger:
    """Tests that the discrete fitter caches an integer-typed sample."""

    def test_discrete_cached_sample_is_integer(self, pandas_poisson_dataset, local_backend):
        """cached_sample from DiscreteDistributionFitter must have integer dtype.

        FAILS currently because:
          1. The field is `sample` not `cached_sample` (AttributeError)
          2. Even if renamed, _create_fitting_sample() returns float64 and the
             .astype(int) + create_discrete_sample_data() only run when
             data_sample is None (i.e., only inside _fit_single_column, not the
             cached path).
        """
        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_poisson_dataset, column="counts", max_distributions=3)
        best = results.best(n=1, metric="sse")

        assert len(best) == 1
        result = best[0]

        assert result.cached_sample is not None, "cached_sample must be populated after discrete fit()"
        assert np.issubdtype(result.cached_sample.dtype, np.integer), (
            f"cached_sample for discrete data must be integer dtype, got {result.cached_sample.dtype}"
        )

    def test_discrete_cached_sample_contains_non_negative_integers(self, pandas_poisson_dataset, local_backend):
        """cached_sample for Poisson data must contain non-negative integers."""
        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_poisson_dataset, column="counts", max_distributions=3)
        best = results.best(n=1, metric="sse")

        result = best[0]

        assert result.cached_sample is not None
        assert np.all(result.cached_sample >= 0), "Poisson samples must be non-negative"
        # Values should be finite integers
        assert np.all(np.isfinite(result.cached_sample.astype(float)))


# ---------------------------------------------------------------------------
# Bug 2 continued: discrete instant plot
# ---------------------------------------------------------------------------


class TestDiscreteInstantPlot:
    """Tests for instant plotting with discrete distributions."""

    def test_discrete_instant_plot(self, pandas_poisson_dataset, local_backend):
        """DiscreteDistributionFitter.plot() without df must work after fit().

        FAILS currently because:
          1. result.sample is the array (field shadows method) so the condition
             `if result.sample is not None` passes but checks the wrong attribute.
          2. After rename, plot() must check `result.cached_sample`.
          3. The best result must have `cached_sample` set (not the shadowed `sample`).
        """
        fitter = DiscreteDistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_poisson_dataset, column="counts", max_distributions=3)
        best = results.best(n=1, metric="sse")[0]

        # Verify cached_sample is set (FAILS currently — field is named `sample`)
        assert best.cached_sample is not None, "cached_sample must be set after discrete fit()"

        fig, ax = fitter.plot(best)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_discrete_instant_plot_raises_without_df_or_cached_sample(self):
        """plot() on discrete result without cached_sample must raise ValueError."""
        fitter = DiscreteDistributionFitter(backend=LocalBackend())
        result_no_sample = DistributionFitResult(
            distribution="poisson",
            parameters=[7.0],
            sse=0.002,
            # cached_sample intentionally omitted
        )

        with pytest.raises(ValueError, match="(?i)(df|sample|cached_sample)"):
            fitter.plot(result_no_sample)


# ---------------------------------------------------------------------------
# Bug 3: LazyFitResults.materialize() must preserve samples
# ---------------------------------------------------------------------------


class TestMaterializePreservesSamples:
    """Tests that LazyFitResults.materialize() passes the samples dict through."""

    def test_materialize_preserves_samples_dict(self):
        """materialize() must create EagerFitResults with _samples intact.

        FAILS currently because materialize() calls:
            EagerFitResults(materialized_df)  (or .cache())
        without passing samples=self._samples.
        """
        # Build a minimal pandas-based LazyFitResults with a samples dict
        sample_array = np.random.normal(50, 10, 1000)
        samples = {"value": sample_array}

        results_df = pd.DataFrame(
            [
                {
                    "column_name": "value",
                    "distribution": "norm",
                    "parameters": [50.0, 10.0],
                    "sse": 0.005,
                    "aic": 1500.0,
                    "bic": 1520.0,
                    "ks_statistic": None,
                    "pvalue": None,
                    "ad_statistic": None,
                    "ad_pvalue": None,
                    "data_min": None,
                    "data_max": None,
                    "data_mean": None,
                    "data_stddev": None,
                    "data_count": None,
                    "data_kurtosis": None,
                    "data_skewness": None,
                    "lower_bound": None,
                    "upper_bound": None,
                }
            ]
        )

        # Minimal LazyMetricsContext with a cached sample to avoid source DF access
        context = LazyMetricsContext(
            source_df=pd.DataFrame({"value": sample_array}),
            column="value",
            random_seed=42,
            row_count=len(sample_array),
            lower_bound=None,
            upper_bound=None,
            is_discrete=False,
            cached_sample=sample_array,
        )

        lazy_results = LazyFitResults(
            results_df=results_df,
            lazy_contexts={"value": context},
            samples=samples,
        )

        materialized = lazy_results.materialize()

        # The materialized EagerFitResults must have the samples dict
        assert materialized._samples is not None
        assert "value" in materialized._samples, (
            "materialize() must pass samples=self._samples to EagerFitResults"
        )
        np.testing.assert_array_equal(materialized._samples["value"], sample_array)

    def test_lazy_materialize_best_has_cached_sample(self, pandas_dataset, local_backend):
        """After lazy fit + materialize(), best()[0].cached_sample must be set.

        FAILS currently for two reasons:
          1. materialize() drops the samples dict
          2. The field is named `sample` not `cached_sample`
        """
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", lazy_metrics=True, max_distributions=3)

        assert results.is_lazy, "lazy_metrics=True must return LazyFitResults"

        materialized = results.materialize()

        assert not materialized.is_lazy, "materialize() must return EagerFitResults"

        best = materialized.best(n=1, metric="sse")
        assert len(best) == 1
        result = best[0]

        assert result.cached_sample is not None, (
            "cached_sample must be preserved through lazy fit -> materialize() -> best()"
        )
        assert isinstance(result.cached_sample, np.ndarray)

    def test_lazy_materialize_no_error(self, pandas_dataset, local_backend):
        """materialize() on a lazy fit result must complete without error and preserve samples.

        FAILS currently because materialize() drops the samples dict, so
        _samples is empty on the returned EagerFitResults.
        """
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", lazy_metrics=True, max_distributions=3)

        # Should not raise
        materialized = results.materialize()

        assert isinstance(materialized, EagerFitResults)
        assert materialized.count() > 0

        # samples dict must be propagated (FAILS currently — materialize() drops it)
        assert materialized._samples, "materialize() must preserve the samples dict"


# ---------------------------------------------------------------------------
# End-to-end: fit then instant plot without providing df
# ---------------------------------------------------------------------------


class TestEndToEndInstantPlotting:
    """End-to-end tests that verify the complete instant plotting workflow."""

    def test_fit_then_instant_plot_no_df(self, pandas_dataset, local_backend):
        """Fitting then calling plot() without df must work end-to-end.

        FAILS currently because best()[0].cached_sample raises AttributeError
        (field is named `sample`, not `cached_sample`).
        """
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=3)
        best = results.best(n=1, metric="sse")[0]

        # Verify cached_sample is set (FAILS currently — field is named `sample`)
        assert best.cached_sample is not None, "cached_sample must be set after fit()"

        # Must not raise — uses cached_sample
        fig, ax = fitter.plot(best)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_fit_then_instant_plot_qq_no_df(self, pandas_dataset, local_backend):
        """Fitting then calling plot_qq() without df must work end-to-end.

        FAILS currently because best()[0].cached_sample raises AttributeError
        (field is named `sample`, not `cached_sample`).
        """
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=3)
        best = results.best(n=1, metric="sse")[0]

        # Verify cached_sample is set (FAILS currently — field is named `sample`)
        assert best.cached_sample is not None, "cached_sample must be set after fit()"

        fig, ax = fitter.plot_qq(best)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_fit_then_instant_plot_pp_no_df(self, pandas_dataset, local_backend):
        """Fitting then calling plot_pp() without df must work end-to-end.

        FAILS currently because best()[0].cached_sample raises AttributeError
        (field is named `sample`, not `cached_sample`).
        """
        fitter = DistributionFitter(backend=local_backend)
        results = fitter.fit(pandas_dataset, column="value", max_distributions=3)
        best = results.best(n=1, metric="sse")[0]

        # Verify cached_sample is set (FAILS currently — field is named `sample`)
        assert best.cached_sample is not None, "cached_sample must be set after fit()"

        fig, ax = fitter.plot_pp(best)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
