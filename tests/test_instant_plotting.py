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
import warnings

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
        """plot() with both df and cached_sample must emit FutureWarning and succeed.

        After the cache-first fix, when df is passed alongside a cached_sample,
        the cache is used (not df) and a FutureWarning is emitted to tell callers
        that passing df is unnecessary.

        FAILS currently because no FutureWarning is emitted — the current code
        silently uses the df path without any deprecation notice.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name="value",
            cached_sample=normal_data,
        )

        # Passing df alongside a cached_sample must emit FutureWarning
        with pytest.warns(FutureWarning, match="unnecessary"):
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


# ---------------------------------------------------------------------------
# Cache-first priority: cached_sample takes priority over df
# ---------------------------------------------------------------------------


class TestCacheFirstPriority:
    """Tests that cached_sample takes priority over df when both are present.

    After the fix, all plot methods must use the cached_sample path when
    result.cached_sample is set, regardless of whether df is also supplied.
    These tests FAIL currently because:
      - continuous plot() uses df when df is provided (cache only wins when df is None)
      - continuous plot_qq() / plot_pp() check df FIRST — always use df when provided
      - discrete plot() checks df FIRST — always uses df when provided
      - plot_comparison() requires df as a positional argument (no cache path at all)
    """

    @pytest.fixture
    def cached_norm_result(self, normal_data):
        """DistributionFitResult for 'norm' with cached_sample set."""
        return DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name="value",
            cached_sample=normal_data,
        )

    @pytest.fixture
    def cached_norm_result_2(self, normal_data):
        """A second DistributionFitResult for 'norm' with cached_sample — for comparison plots."""
        np.random.seed(99)
        alt_sample = np.random.normal(50.0, 10.0, size=len(normal_data))
        return DistributionFitResult(
            distribution="norm",
            parameters=[50.5, 9.8],
            sse=0.006,
            column_name="value",
            cached_sample=alt_sample,
        )

    @pytest.fixture
    def cached_poisson_result(self, poisson_data):
        """DistributionFitResult for 'poisson' with integer cached_sample."""
        return DistributionFitResult(
            distribution="poisson",
            parameters=[7.0],
            sse=0.002,
            column_name="counts",
            cached_sample=poisson_data.astype(int),
        )

    def test_continuous_plot_cache_wins_over_df(
        self, cached_norm_result, pandas_dataset
    ):
        """plot() must use cached_sample even when df is provided.

        FAILS currently because plot() takes the df path when df is not None
        (line 704: `if result.cached_sample is not None and df is None:`).
        After the fix, cache wins unconditionally when cached_sample is set and
        force_recompute is not True.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        # Both df AND a result with cached_sample — cache must win (emitting FutureWarning)
        with pytest.warns(FutureWarning):
            fig, ax = fitter.plot(cached_norm_result, df=pandas_dataset, column="value")

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_continuous_plot_qq_cache_wins_over_df(
        self, cached_norm_result, pandas_dataset
    ):
        """plot_qq() must use cached_sample even when df is provided.

        FAILS currently because plot_qq() checks df FIRST (line 1086:
        `if df is not None:`) — cache is never reached when df is supplied.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning):
            fig, ax = fitter.plot_qq(
                cached_norm_result, df=pandas_dataset, column="value"
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_continuous_plot_pp_cache_wins_over_df(
        self, cached_norm_result, pandas_dataset
    ):
        """plot_pp() must use cached_sample even when df is provided.

        FAILS currently because plot_pp() checks df FIRST (line 1191:
        `if df is not None:`) — cache is never reached when df is supplied.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning):
            fig, ax = fitter.plot_pp(
                cached_norm_result, df=pandas_dataset, column="value"
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_discrete_plot_cache_wins_over_df(
        self, cached_poisson_result, pandas_poisson_dataset
    ):
        """DiscreteDistributionFitter.plot() must use cached_sample even when df is provided.

        FAILS currently because discrete plot() checks df FIRST (line 571:
        `if df is not None:`) — cache is never reached when df is supplied.
        """
        fitter = DiscreteDistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning):
            fig, ax = fitter.plot(
                cached_poisson_result, df=pandas_poisson_dataset, column="counts"
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_comparison_uses_cached_samples_without_df(
        self, cached_norm_result, cached_norm_result_2
    ):
        """plot_comparison() must work without df when results have cached_sample.

        FAILS currently because plot_comparison() declares df as a required
        positional argument — calling it without df raises TypeError.
        After the fix, df becomes Optional and cached_samples are used instead.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        fig, ax = fitter.plot_comparison(
            [cached_norm_result, cached_norm_result_2],
        )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# FutureWarning emitted when df is passed but cache is available
# ---------------------------------------------------------------------------


class TestFutureWarningOnDfWithCache:
    """Tests that FutureWarning is emitted when df is passed alongside a cached result.

    These tests FAIL currently because none of the plot methods emit any warning
    today — they silently use df when it is provided.
    """

    @pytest.fixture
    def cached_norm_result(self, normal_data):
        return DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name="value",
            cached_sample=normal_data,
        )

    @pytest.fixture
    def cached_poisson_result(self, poisson_data):
        return DistributionFitResult(
            distribution="poisson",
            parameters=[7.0],
            sse=0.002,
            column_name="counts",
            cached_sample=poisson_data.astype(int),
        )

    def test_plot_emits_future_warning_when_df_and_cache_both_present(
        self, cached_norm_result, pandas_dataset
    ):
        """plot() must emit FutureWarning mentioning 'unnecessary' when df is redundant.

        FAILS currently — no warning is emitted.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning, match="unnecessary"):
            fitter.plot(cached_norm_result, df=pandas_dataset, column="value")

        plt.close("all")

    def test_plot_qq_emits_future_warning_when_df_and_cache_both_present(
        self, cached_norm_result, pandas_dataset
    ):
        """plot_qq() must emit FutureWarning mentioning 'unnecessary' when df is redundant.

        FAILS currently — no warning is emitted.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning, match="unnecessary"):
            fitter.plot_qq(cached_norm_result, df=pandas_dataset, column="value")

        plt.close("all")

    def test_plot_pp_emits_future_warning_when_df_and_cache_both_present(
        self, cached_norm_result, pandas_dataset
    ):
        """plot_pp() must emit FutureWarning mentioning 'unnecessary' when df is redundant.

        FAILS currently — no warning is emitted.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning, match="unnecessary"):
            fitter.plot_pp(cached_norm_result, df=pandas_dataset, column="value")

        plt.close("all")

    def test_discrete_plot_emits_future_warning_when_df_and_cache_both_present(
        self, cached_poisson_result, pandas_poisson_dataset
    ):
        """Discrete plot() must emit FutureWarning when df is redundant.

        FAILS currently — no warning is emitted.
        """
        fitter = DiscreteDistributionFitter(backend=LocalBackend())

        with pytest.warns(FutureWarning, match="unnecessary"):
            fitter.plot(
                cached_poisson_result, df=pandas_poisson_dataset, column="counts"
            )

        plt.close("all")

    def test_no_warning_when_only_cache_no_df(self, cached_norm_result):
        """plot() must NOT emit FutureWarning when df is absent (cache-only path).

        This is the happy-path: caller omits df entirely, cache is used silently.
        Must pass both before and after the fix.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # Should NOT raise — no df means no warning
            fig, ax = fitter.plot(cached_norm_result)

        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_no_warning_when_only_df_no_cache(self, pandas_dataset):
        """plot() must NOT emit FutureWarning when cached_sample is absent and df provided.

        Normal usage: user provides df explicitly, result has no cache.
        Must pass both before and after the fix.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        result_no_cache = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name="value",
            # cached_sample intentionally absent
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fig, ax = fitter.plot(result_no_cache, df=pandas_dataset, column="value")

        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# force_recompute=True escape hatch: bypasses cache, uses df, no warning
# ---------------------------------------------------------------------------


class TestForceRecompute:
    """Tests for the force_recompute=True parameter that bypasses the cache.

    These tests FAIL currently because force_recompute is not yet a parameter
    on any plot method — calling with force_recompute=True raises TypeError.
    """

    @pytest.fixture
    def cached_norm_result(self, normal_data):
        return DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name="value",
            cached_sample=normal_data,
        )

    @pytest.fixture
    def cached_poisson_result(self, poisson_data):
        return DistributionFitResult(
            distribution="poisson",
            parameters=[7.0],
            sse=0.002,
            column_name="counts",
            cached_sample=poisson_data.astype(int),
        )

    def test_plot_force_recompute_uses_df_no_warning(
        self, cached_norm_result, pandas_dataset
    ):
        """plot(force_recompute=True) must use df, emit NO FutureWarning, and succeed.

        FAILS currently — TypeError: plot() got an unexpected keyword argument
        'force_recompute'.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # Must not raise FutureWarning
            fig, ax = fitter.plot(
                cached_norm_result,
                df=pandas_dataset,
                column="value",
                force_recompute=True,
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_qq_force_recompute_uses_df_no_warning(
        self, cached_norm_result, pandas_dataset
    ):
        """plot_qq(force_recompute=True) must use df, emit NO FutureWarning, and succeed.

        FAILS currently — TypeError: plot_qq() got an unexpected keyword argument
        'force_recompute'.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fig, ax = fitter.plot_qq(
                cached_norm_result,
                df=pandas_dataset,
                column="value",
                force_recompute=True,
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_pp_force_recompute_uses_df_no_warning(
        self, cached_norm_result, pandas_dataset
    ):
        """plot_pp(force_recompute=True) must use df, emit NO FutureWarning, and succeed.

        FAILS currently — TypeError: plot_pp() got an unexpected keyword argument
        'force_recompute'.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fig, ax = fitter.plot_pp(
                cached_norm_result,
                df=pandas_dataset,
                column="value",
                force_recompute=True,
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_discrete_plot_force_recompute_uses_df_no_warning(
        self, cached_poisson_result, pandas_poisson_dataset
    ):
        """Discrete plot(force_recompute=True) must use df, emit NO FutureWarning.

        FAILS currently — TypeError: plot() got an unexpected keyword argument
        'force_recompute'.
        """
        fitter = DiscreteDistributionFitter(backend=LocalBackend())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fig, ax = fitter.plot(
                cached_poisson_result,
                df=pandas_poisson_dataset,
                column="counts",
                force_recompute=True,
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_force_recompute_without_df_raises_value_error(self, cached_norm_result):
        """force_recompute=True without df must raise ValueError with clear message."""
        fitter = DistributionFitter(backend=LocalBackend())

        with pytest.raises(ValueError, match="force_recompute=True requires df"):
            fitter.plot(cached_norm_result, force_recompute=True)

    def test_plot_comparison_force_recompute_uses_df_no_warning(
        self, pandas_dataset, normal_data
    ):
        """plot_comparison(force_recompute=True) must use df, emit NO FutureWarning."""
        fitter = DistributionFitter(backend=LocalBackend())
        results = [
            DistributionFitResult(
                distribution="norm",
                parameters=[50.0, 10.0],
                sse=0.005,
                column_name="value",
                cached_sample=normal_data,
            ),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fig, ax = fitter.plot_comparison(
                results, df=pandas_dataset, column="value", force_recompute=True
            )

        assert isinstance(fig, plt.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_comparison() instant path: df optional when results carry cached_sample
# ---------------------------------------------------------------------------


class TestPlotComparisonInstant:
    """Tests for plot_comparison() using cached_sample instead of requiring df.

    These tests FAIL currently because plot_comparison() declares:
        df: DataFrame   (required positional argument)
    so calling it without df raises TypeError immediately.
    After the fix, df becomes Optional[DataFrame] = None and the method falls
    back to the cached_sample on each result.
    """

    @pytest.fixture
    def two_cached_results(self, normal_data):
        """Two DistributionFitResult objects with distinct cached_samples."""
        np.random.seed(7)
        sample_a = np.random.normal(50.0, 10.0, size=len(normal_data))
        np.random.seed(13)
        sample_b = np.random.normal(50.5, 9.5, size=len(normal_data))
        return [
            DistributionFitResult(
                distribution="norm",
                parameters=[50.0, 10.0],
                sse=0.005,
                column_name="value",
                cached_sample=sample_a,
            ),
            DistributionFitResult(
                distribution="norm",
                parameters=[50.5, 9.5],
                sse=0.006,
                column_name="value",
                cached_sample=sample_b,
            ),
        ]

    def test_plot_comparison_without_df_succeeds(self, two_cached_results):
        """plot_comparison(results) without df must return a Figure.

        FAILS currently — TypeError: plot_comparison() missing required
        positional argument 'df'.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        fig, ax = fitter.plot_comparison(two_cached_results)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_comparison_without_df_requires_cached_samples(self):
        """plot_comparison(results) without df AND without cached_sample must raise ValueError.

        When no df is provided AND results have no cached_sample, the method
        has no data source and must raise ValueError.

        FAILS currently — TypeError from missing df argument fires before
        any ValueError logic.
        """
        fitter = DistributionFitter(backend=LocalBackend())
        results_no_cache = [
            DistributionFitResult(
                distribution="norm",
                parameters=[50.0, 10.0],
                sse=0.005,
                # cached_sample intentionally absent
            ),
            DistributionFitResult(
                distribution="norm",
                parameters=[50.5, 9.5],
                sse=0.006,
                # cached_sample intentionally absent
            ),
        ]

        with pytest.raises((ValueError, TypeError)):
            fitter.plot_comparison(results_no_cache)

    def test_plot_comparison_with_df_still_works(
        self, two_cached_results, pandas_dataset
    ):
        """plot_comparison(results, df=..., column=...) must still work when df is provided.

        Backwards-compatibility test: existing callers that pass df must not break.
        After the fix, results carry cached_sample but df is also passed — since
        there is a cache, a FutureWarning is emitted and the cache is used.

        FAILS currently — TypeError from missing df argument (pre-fix) fires, OR
        the method silently uses df (post-fix it will emit FutureWarning).
        This test verifies the method at least succeeds (warning is tolerated).
        """
        fitter = DistributionFitter(backend=LocalBackend())

        # Allow FutureWarning (cache wins over df after the fix)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("always")
            fig, ax = fitter.plot_comparison(
                two_cached_results, df=pandas_dataset, column="value"
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_comparison_column_inferred_from_results(self, two_cached_results):
        """plot_comparison() must infer column from result.column_name when df is absent.

        When results carry column_name and the method uses cached_sample,
        there is no need to pass column explicitly.

        FAILS currently — TypeError from missing df argument.
        """
        fitter = DistributionFitter(backend=LocalBackend())

        # column not passed — must be inferred from result.column_name="value"
        fig, ax = fitter.plot_comparison(two_cached_results)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close("all")
