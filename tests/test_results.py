"""Tests for results module."""

import warnings

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.results import BaseFitResults, DistributionFitResult, FitResults


# Fixtures normal_result and gamma_result are now in conftest.py


class TestDistributionFitResult:
    """Tests for DistributionFitResult dataclass."""

    def test_to_dict(self, normal_result):
        """Test converting result to dictionary."""
        result_dict = normal_result.to_dict()

        assert result_dict["distribution"] == "norm"
        assert result_dict["parameters"] == [50.0, 10.0]  # norm has only loc, scale
        assert result_dict["sse"] == 0.005
        assert result_dict["aic"] == 1500.0
        assert result_dict["bic"] == 1520.0

    def test_get_scipy_dist(self, normal_result):
        """Test getting scipy distribution object."""
        # Default: returns frozen distribution with parameters applied
        frozen = normal_result.get_scipy_dist()
        assert hasattr(frozen, "rvs")
        assert hasattr(frozen, "pdf")

        # With frozen=False: returns unfrozen distribution class
        dist_class = normal_result.get_scipy_dist(frozen=False)
        assert dist_class is st.norm

    def test_sample(self, normal_result):
        """Test generating samples from fitted distribution."""
        samples = normal_result.sample(size=1000, random_state=42)

        assert len(samples) == 1000
        assert isinstance(samples, np.ndarray)

        # Samples should be approximately normal(50, 10)
        assert 45 < samples.mean() < 55
        assert 8 < samples.std() < 12

    def test_sample_reproducible(self, normal_result):
        """Test that sampling is reproducible."""
        samples1 = normal_result.sample(size=1000, random_state=42)
        samples2 = normal_result.sample(size=1000, random_state=42)

        assert np.array_equal(samples1, samples2)

    def test_pdf(self, normal_result):
        """Test evaluating PDF."""
        x = np.array([30, 40, 50, 60, 70])
        pdf_values = normal_result.pdf(x)

        assert len(pdf_values) == len(x)
        assert np.all(pdf_values >= 0)
        assert np.all(np.isfinite(pdf_values))

        # PDF should be highest at mean (50)
        assert pdf_values[2] == np.max(pdf_values)

    def test_cdf(self, normal_result):
        """Test evaluating CDF."""
        x = np.array([30, 40, 50, 60, 70])
        cdf_values = normal_result.cdf(x)

        assert len(cdf_values) == len(x)
        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)

        # CDF should be increasing
        assert np.all(np.diff(cdf_values) >= 0)

        # CDF at mean should be ~0.5
        assert np.isclose(cdf_values[2], 0.5, atol=0.01)

    def test_ppf(self, normal_result):
        """Test evaluating percent point function (inverse CDF)."""
        q = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        ppf_values = normal_result.ppf(q)

        assert len(ppf_values) == len(q)
        assert np.all(np.isfinite(ppf_values))

        # PPF should be increasing
        assert np.all(np.diff(ppf_values) > 0)

        # PPF at 0.5 should be ~mean
        assert np.isclose(ppf_values[2], 50.0, atol=1.0)

    def test_repr(self, normal_result):
        """Test string representation."""
        repr_str = repr(normal_result)

        assert "norm" in repr_str
        assert "0.005" in repr_str  # SSE
        assert "DistributionFitResult" in repr_str

    def test_result_without_aic_bic(self):
        """Test result creation without AIC/BIC."""
        # norm has only 2 params: loc, scale
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005)

        assert result.aic is None
        assert result.bic is None

        # Should still work
        result_dict = result.to_dict()
        assert result_dict["aic"] is None
        assert result_dict["bic"] is None

class TestFitResults:
    """Tests for FitResults class."""

    @pytest.fixture
    def sample_results_df(self, spark_session):
        """Create a sample results DataFrame."""
        data = [
            # distribution, parameters, sse, aic, bic, ks_statistic, pvalue, ad_statistic, ad_pvalue
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.025, 0.90, 0.35, 0.15),
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0, 0.020, 0.95, 0.40, None),
            ("expon", [0.0, 5.0], 0.008, 1600.0, 1615.0, 0.035, 0.75, 0.30, 0.20),
            ("lognorm", [1.0, 0.0, 2.0], 0.010, 1650.0, 1680.0, 0.040, 0.65, 0.60, None),
            ("weibull_min", [1.5, 0.0, 3.0], 0.004, 1450.0, 1480.0, 0.022, 0.92, 0.45, None),
        ]

        return spark_session.createDataFrame(
            data, ["distribution", "parameters", "sse", "aic", "bic", "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue"]
        )

    def test_initialization(self, sample_results_df):
        """Test FitResults initialization."""
        results = FitResults(sample_results_df)

        assert results._df == sample_results_df

    def test_df_to_pandas(self, sample_results_df):
        """Test converting to pandas DataFrame via df property."""
        results = FitResults(sample_results_df)
        df_pandas = results.df.toPandas()

        assert len(df_pandas) == 5
        assert "distribution" in df_pandas.columns
        assert "sse" in df_pandas.columns

    @pytest.mark.parametrize("metric", ["sse", "aic", "bic", "ks_statistic", "ad_statistic"])
    def test_best_by_metric(self, sample_results_df, metric):
        """Test getting best distributions by various metrics."""
        results = FitResults(sample_results_df)
        best = results.best(n=1, metric=metric)

        assert len(best) == 1
        # Different best distribution depending on metric
        if metric == "ad_statistic":
            assert best[0].distribution == "expon"  # expon has lowest A-D (0.30)
        else:
            assert best[0].distribution == "gamma"  # gamma has lowest SSE, AIC, BIC, and ks_statistic

    def test_best_top_n(self, sample_results_df):
        """Test getting top N distributions."""
        results = FitResults(sample_results_df)
        top_3 = results.best(n=3, metric="sse")

        assert len(top_3) == 3

        # Should be sorted by SSE
        assert top_3[0].sse <= top_3[1].sse <= top_3[2].sse

        # Should be: gamma (0.003), weibull_min (0.004), norm (0.005)
        assert top_3[0].distribution == "gamma"
        assert top_3[1].distribution == "weibull_min"
        assert top_3[2].distribution == "norm"

    def test_best_returns_ks_fields(self, sample_results_df):
        """Test that best() returns results with ks_statistic and pvalue fields."""
        results = FitResults(sample_results_df)
        best = results.best(n=1)[0]

        # Should have K-S fields populated
        assert best.ks_statistic is not None
        assert best.pvalue is not None
        assert best.ks_statistic == 0.020  # gamma's ks_statistic
        assert best.pvalue == 0.95  # gamma's pvalue

    def test_best_by_ks_statistic(self, sample_results_df):
        """Test getting best distribution by K-S statistic."""
        results = FitResults(sample_results_df)
        top_3 = results.best(n=3, metric="ks_statistic")

        assert len(top_3) == 3

        # Should be sorted by ks_statistic (ascending)
        assert top_3[0].ks_statistic <= top_3[1].ks_statistic <= top_3[2].ks_statistic

        # gamma (0.020), weibull_min (0.022), norm (0.025)
        assert top_3[0].distribution == "gamma"
        assert top_3[1].distribution == "weibull_min"
        assert top_3[2].distribution == "norm"

    def test_best_by_ad_statistic(self, sample_results_df):
        """Test getting best distribution by A-D statistic."""
        results = FitResults(sample_results_df)
        top_3 = results.best(n=3, metric="ad_statistic")

        assert len(top_3) == 3

        # Should be sorted by ad_statistic (ascending)
        assert top_3[0].ad_statistic <= top_3[1].ad_statistic <= top_3[2].ad_statistic

        # expon (0.30), norm (0.35), gamma (0.40)
        assert top_3[0].distribution == "expon"
        assert top_3[1].distribution == "norm"
        assert top_3[2].distribution == "gamma"

    def test_best_returns_ad_fields(self, sample_results_df):
        """Test that best() returns results with ad_statistic and ad_pvalue fields."""
        results = FitResults(sample_results_df)
        best = results.best(n=1, metric="ad_statistic")[0]

        # Should have A-D fields populated
        assert best.ad_statistic is not None
        assert best.ad_statistic == 0.30  # expon's ad_statistic
        assert best.ad_pvalue == 0.20  # expon's ad_pvalue

    def test_best_returns_none_ad_pvalue_for_unsupported(self, sample_results_df):
        """Test that best() returns None for ad_pvalue when not available."""
        results = FitResults(sample_results_df)
        # Get gamma which doesn't have A-D p-value
        best_gamma = results.best(n=5, metric="sse")
        gamma_result = [r for r in best_gamma if r.distribution == "gamma"][0]

        assert gamma_result.ad_statistic == 0.40
        assert gamma_result.ad_pvalue is None

    def test_best_invalid_metric(self, sample_results_df):
        """Test that invalid metric raises error."""
        results = FitResults(sample_results_df)

        with pytest.raises(ValueError):
            results.best(n=1, metric="invalid_metric")

    @pytest.mark.parametrize("threshold_kwarg,threshold_val,expected_count", [
        ({"sse_threshold": 0.006}, "sse", 3),
        ({"aic_threshold": 1500}, "aic", 2),
        ({"ks_threshold": 0.03}, "ks_statistic", 3),  # gamma, weibull_min, norm
        ({"ad_threshold": 0.40}, "ad_statistic", 2),  # expon (0.30), norm (0.35)
    ])
    def test_filter_by_threshold(self, sample_results_df, threshold_kwarg, threshold_val, expected_count):
        """Test filtering by various threshold types."""
        results = FitResults(sample_results_df)
        filtered = results.filter(**threshold_kwarg)

        assert filtered.count() == expected_count
        df_pandas = filtered.df.toPandas()
        assert all(df_pandas[threshold_val] < list(threshold_kwarg.values())[0])

    def test_filter_by_pvalue_threshold(self, sample_results_df):
        """Test filtering by minimum p-value threshold."""
        results = FitResults(sample_results_df)
        # Filter for p-value > 0.80 (should get gamma, weibull_min, norm)
        filtered = results.filter(pvalue_threshold=0.80)

        assert filtered.count() == 3
        df_pandas = filtered.df.toPandas()
        assert all(df_pandas["pvalue"] > 0.80)

    def test_filter_multiple_criteria(self, sample_results_df):
        """Test filtering by multiple criteria."""
        results = FitResults(sample_results_df)
        filtered = results.filter(sse_threshold=0.010, aic_threshold=1600)

        # Should meet both criteria
        df_pandas = filtered.df.toPandas()
        assert all(df_pandas["sse"] < 0.010)
        assert all(df_pandas["aic"] < 1600)

    def test_summary(self, sample_results_df):
        """Test getting summary statistics."""
        results = FitResults(sample_results_df)
        summary = results.summary()

        assert "min_sse" in summary.columns
        assert "mean_sse" in summary.columns
        assert "max_sse" in summary.columns
        assert "min_ks" in summary.columns
        assert "mean_ks" in summary.columns
        assert "max_ks" in summary.columns
        assert "min_pvalue" in summary.columns
        assert "max_pvalue" in summary.columns
        assert "min_ad" in summary.columns
        assert "mean_ad" in summary.columns
        assert "max_ad" in summary.columns
        assert "total_distributions" in summary.columns

        # Check values
        assert summary["min_sse"].iloc[0] == 0.003
        assert summary["max_sse"].iloc[0] == 0.010
        assert summary["min_ks"].iloc[0] == 0.020  # gamma
        assert summary["max_ks"].iloc[0] == 0.040  # lognorm
        assert summary["min_ad"].iloc[0] == 0.30  # expon
        assert summary["max_ad"].iloc[0] == 0.60  # lognorm
        assert summary["total_distributions"].iloc[0] == 5

    def test_count(self, sample_results_df):
        """Test counting distributions."""
        results = FitResults(sample_results_df)

        assert results.count() == 5
        assert len(results) == 5

    def test_repr(self, sample_results_df):
        """Test string representation."""
        results = FitResults(sample_results_df)
        repr_str = repr(results)

        assert "5 distributions" in repr_str
        assert "gamma" in repr_str  # Best distribution
        assert "0.020" in repr_str  # Best KS statistic

    def test_empty_results(self, spark_session):
        """Test FitResults with empty DataFrame handles all operations gracefully."""
        from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

        schema = StructType(
            [
                StructField("distribution", StringType(), False),
                StructField("parameters", ArrayType(FloatType()), False),
                StructField("sse", FloatType(), False),
                StructField("aic", FloatType(), True),
                StructField("bic", FloatType(), True),
                StructField("ks_statistic", FloatType(), True),
                StructField("pvalue", FloatType(), True),
                StructField("ad_statistic", FloatType(), True),
                StructField("ad_pvalue", FloatType(), True),
            ]
        )
        empty_df = spark_session.createDataFrame([], schema)

        results = FitResults(empty_df)

        # Test count and repr
        assert results.count() == 0
        assert len(results) == 0
        assert "0 distributions" in repr(results)

        # Test best() returns empty list
        assert results.best(n=5) == []

    def test_filter_returns_fitresults_instance(self, sample_results_df):
        """Test that filter returns a FitResults instance."""
        results = FitResults(sample_results_df)
        filtered = results.filter(sse_threshold=0.01)

        # FitResults is now a factory function; use BaseFitResults for isinstance
        assert isinstance(filtered, BaseFitResults)

    def test_filter_no_criteria(self, sample_results_df):
        """Test filter with no criteria returns all results."""
        results = FitResults(sample_results_df)
        filtered = results.filter()

        assert filtered.count() == results.count()

class TestTruncatedFrozenDist:
    """Tests for TruncatedFrozenDist wrapper class."""

    def test_logpdf_within_bounds(self):
        """Test logpdf returns valid values within bounds."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([35, 40, 50, 60, 65])
        logpdf_values = truncated.logpdf(x)

        # All values should be finite (not -inf)
        assert np.all(np.isfinite(logpdf_values))
        # logpdf should be less than 0 for probability density < 1
        assert np.all(logpdf_values < 0)

    def test_logpdf_outside_bounds(self):
        """Test logpdf returns -inf outside bounds."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([20, 25, 75, 80])
        logpdf_values = truncated.logpdf(x)

        # All values outside bounds should be -inf
        assert np.all(logpdf_values == -np.inf)

    def test_logpdf_at_boundaries(self):
        """Test logpdf at exact boundary values."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([30, 70])
        logpdf_values = truncated.logpdf(x)

        # Values at boundaries should be finite
        assert np.all(np.isfinite(logpdf_values))

    def test_logpdf_empty_mask(self):
        """Test logpdf when all values are outside bounds."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=40, ub=60)

        x = np.array([0, 10, 90, 100])
        logpdf_values = truncated.logpdf(x)

        # All should be -inf
        assert np.all(logpdf_values == -np.inf)

    def test_mean_approximation(self):
        """Test mean approximation for truncated distribution."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        mean_val = truncated.mean()

        # Mean of symmetric truncation around the true mean should be close to true mean
        assert np.isfinite(mean_val)
        assert 45 < mean_val < 55

    def test_mean_asymmetric_truncation(self):
        """Test mean shifts with asymmetric truncation."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        # Truncate to keep only upper part
        truncated = TruncatedFrozenDist(frozen, lb=50, ub=80)

        mean_val = truncated.mean()

        # Mean should be above the original mean due to left truncation
        assert mean_val > 50

    def test_std_approximation(self):
        """Test std approximation for truncated distribution."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        std_val = truncated.std()

        # Std of truncated dist should be smaller than original
        assert np.isfinite(std_val)
        assert std_val > 0
        assert std_val < 10  # Original scale is 10

    def test_std_narrow_truncation(self):
        """Test std decreases with narrower truncation."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        wide = TruncatedFrozenDist(frozen, lb=20, ub=80)
        narrow = TruncatedFrozenDist(frozen, lb=40, ub=60)

        # Narrower truncation should have smaller std
        assert narrow.std() < wide.std()

    def test_pdf_consistency_with_logpdf(self):
        """Test that pdf and logpdf are consistent."""
        from spark_bestfit.results import TruncatedFrozenDist

        frozen = st.norm(loc=50, scale=10)
        truncated = TruncatedFrozenDist(frozen, lb=30, ub=70)

        x = np.array([35, 50, 65])
        pdf_values = truncated.pdf(x)
        logpdf_values = truncated.logpdf(x)

        # exp(logpdf) should equal pdf
        np.testing.assert_allclose(np.exp(logpdf_values), pdf_values, rtol=1e-10)


class TestDistributionFitResultEdgeCases:
    """Edge case tests for DistributionFitResult."""

    def test_repr_without_aic_bic(self):
        """Test __repr__ when aic and bic are None."""
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005, aic=None, bic=None)

        # Should not raise when aic/bic are None
        repr_str = repr(result)

        assert "norm" in repr_str
        assert "0.005" in repr_str

    def test_pdf_with_single_value(self):
        """Test PDF evaluation with single value."""
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005)

        pdf_value = result.pdf(np.array([50.0]))

        assert len(pdf_value) == 1
        assert pdf_value[0] > 0

    def test_cdf_bounds(self):
        """Test CDF returns values in [0, 1]."""
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005)

        x = np.linspace(0, 100, 100)
        cdf_values = result.cdf(x)

        assert np.all(cdf_values >= 0)
        assert np.all(cdf_values <= 1)

    def test_ppf_bounds(self):
        """Test PPF with boundary quantiles."""
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005)

        # Test near-boundary quantiles (not exactly 0 or 1)
        ppf_values = result.ppf(np.array([0.001, 0.999]))

        assert np.all(np.isfinite(ppf_values))
        assert ppf_values[0] < ppf_values[1]

    def test_sample_different_sizes(self):
        """Test sampling with different sizes."""
        result = DistributionFitResult(distribution="norm", parameters=[50.0, 10.0], sse=0.005)

        for size in [1, 10, 100, 10000]:
            samples = result.sample(size=size, random_state=42)
            assert len(samples) == size

    def test_to_dict_complete(self):
        """Test to_dict returns all fields."""
        result = DistributionFitResult(
            distribution="gamma",
            parameters=[2.0, 0.0, 5.0],
            sse=0.003,
            aic=1500.0,
            bic=1520.0,
            ks_statistic=0.05,
            pvalue=0.85,
            ad_statistic=0.40,
            ad_pvalue=None,  # gamma doesn't support A-D p-value
        )

        d = result.to_dict()

        assert set(d.keys()) == {
            "column_name", "distribution", "parameters", "sse", "aic", "bic",
            "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue",
            "data_min", "data_max", "data_mean", "data_stddev", "data_count",
            "lower_bound", "upper_bound"  # Added in v1.4.0 for bounded fitting
        }
        assert d["distribution"] == "gamma"
        assert d["parameters"] == [2.0, 0.0, 5.0]
        assert d["sse"] == 0.003
        assert d["aic"] == 1500.0
        assert d["bic"] == 1520.0
        assert d["ks_statistic"] == 0.05
        assert d["pvalue"] == 0.85
        assert d["ad_statistic"] == 0.40
        assert d["ad_pvalue"] is None

    def test_get_scipy_dist_various_distributions(self):
        """Test get_scipy_dist works for various distributions."""
        # Map distribution name to appropriate parameters
        dist_params = {
            "norm": [0.0, 1.0],  # loc, scale
            "expon": [0.0, 1.0],  # loc, scale
            "gamma": [1.0, 0.0, 1.0],  # a, loc, scale
            "beta": [2.0, 2.0, 0.0, 1.0],  # a, b, loc, scale
            "weibull_min": [1.5, 0.0, 1.0],  # c, loc, scale
        }

        for dist_name, params in dist_params.items():
            result = DistributionFitResult(distribution=dist_name, parameters=params, sse=0.01)

            # Default returns frozen distribution
            frozen = result.get_scipy_dist()
            assert hasattr(frozen, "rvs")
            assert hasattr(frozen, "pdf")

            # Check unfrozen distribution class
            dist_class = result.get_scipy_dist(frozen=False)
            assert dist_class.name == dist_name

    def test_get_scipy_dist_invalid_distribution(self):
        """Test get_scipy_dist raises AttributeError for invalid distribution name."""
        result = DistributionFitResult(
            distribution="nonexistent_distribution",
            parameters=[1.0, 0.0, 1.0],
            sse=0.01,
        )

        with pytest.raises(AttributeError):
            result.get_scipy_dist()


class TestGetParamNames:
    """Tests for get_param_names method."""

    def test_normal_distribution_param_names(self, normal_result):
        """Test parameter names for normal distribution."""
        names = normal_result.get_param_names()
        assert names == ["loc", "scale"]

    def test_gamma_distribution_param_names(self, gamma_result):
        """Test parameter names for gamma distribution."""
        names = gamma_result.get_param_names()
        assert names == ["a", "loc", "scale"]

    def test_beta_distribution_param_names(self):
        """Test parameter names for beta distribution."""
        result = DistributionFitResult(
            distribution="beta",
            parameters=[2.0, 5.0, 0.0, 1.0],
            sse=0.01,
        )
        names = result.get_param_names()
        assert names == ["a", "b", "loc", "scale"]

    def test_discrete_distribution_param_names(self):
        """Test parameter names for discrete distribution (poisson)."""
        result = DistributionFitResult(
            distribution="poisson",
            parameters=[5.0],
            sse=0.01,
        )
        names = result.get_param_names()
        assert names == ["mu"]

    def test_discrete_binomial_param_names(self):
        """Test parameter names for binomial distribution."""
        result = DistributionFitResult(
            distribution="binom",
            parameters=[10.0, 0.3],
            sse=0.01,
        )
        names = result.get_param_names()
        assert names == ["n", "p"]


class TestConfidenceIntervals:
    """Tests for confidence_intervals method."""

    def test_continuous_ci_computation(self, spark_session):
        """Test CI computation for continuous distribution."""
        # Create sample normal data
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=1000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        # Create a result for this data
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        # Compute CI
        ci = result.confidence_intervals(
            df, column="value", alpha=0.05, n_bootstrap=100, random_seed=42
        )

        # Should have correct parameter names
        assert "loc" in ci
        assert "scale" in ci

        # CIs should be valid tuples
        for param, (lower, upper) in ci.items():
            assert lower < upper

    def test_discrete_ci_computation(self, spark_session):
        """Test CI computation for discrete distribution."""
        # Create sample Poisson data
        np.random.seed(42)
        data = np.random.poisson(lam=7, size=500)
        df = spark_session.createDataFrame([(int(x),) for x in data], ["count"])

        # Create a result for this data
        result = DistributionFitResult(
            distribution="poisson",
            parameters=[7.0],
            sse=0.005,
        )

        # Compute CI
        ci = result.confidence_intervals(
            df, column="count", alpha=0.05, n_bootstrap=100, random_seed=42
        )

        # Should have correct parameter name
        assert "mu" in ci

        # CI should be valid
        lower, upper = ci["mu"]
        assert lower < upper
        assert lower < 7.0 < upper  # True value should be in CI

    def test_ci_reproducibility(self, spark_session):
        """Test that CI is reproducible with same seed."""
        np.random.seed(42)
        data = np.random.normal(loc=50, scale=10, size=500)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        ci1 = result.confidence_intervals(
            df, column="value", alpha=0.05, n_bootstrap=50, random_seed=123
        )
        ci2 = result.confidence_intervals(
            df, column="value", alpha=0.05, n_bootstrap=50, random_seed=123
        )

        assert ci1["loc"] == ci2["loc"]
        assert ci1["scale"] == ci2["scale"]

    def test_ci_sampling_with_large_data(self, spark_session):
        """Test that CI correctly samples large datasets."""
        np.random.seed(42)
        # Create large dataset (more than max_samples=10000)
        data = np.random.normal(loc=50, scale=10, size=50000)
        df = spark_session.createDataFrame([(float(x),) for x in data], ["value"])

        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
        )

        # Should not error even with large data
        ci = result.confidence_intervals(
            df, column="value", alpha=0.05, n_bootstrap=50,
            max_samples=5000, random_seed=42
        )

        # Should have valid CI
        assert "loc" in ci
        assert "scale" in ci
        for param, (lower, upper) in ci.items():
            assert lower < upper


class TestWarnIfPoor:
    """Tests for warn_if_poor parameter in best() method."""

    @pytest.fixture
    def results_with_poor_fit(self, spark_session):
        """Create results where best fit has poor p-value."""
        data = [
            # distribution, parameters, sse, aic, bic, ks_statistic, pvalue, ad_statistic, ad_pvalue
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.025, 0.02, 0.35, 0.15),  # poor p-value
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0, 0.020, 0.01, 0.40, None),  # worse p-value
        ]
        return spark_session.createDataFrame(
            data, ["distribution", "parameters", "sse", "aic", "bic", "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue"]
        )

    @pytest.fixture
    def results_with_good_fit(self, spark_session):
        """Create results where best fit has good p-value."""
        data = [
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.025, 0.90, 0.35, 0.15),  # good p-value
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0, 0.020, 0.85, 0.40, None),
        ]
        return spark_session.createDataFrame(
            data, ["distribution", "parameters", "sse", "aic", "bic", "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue"]
        )

    def test_warn_if_poor_emits_warning(self, results_with_poor_fit):
        """Test that warning is emitted for poor fit."""
        results = FitResults(results_with_poor_fit)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            best = results.best(n=1, warn_if_poor=True)

            # Filter for UserWarning only (ignore ResourceWarning, etc.)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1
            assert "poor fit" in str(user_warnings[0].message).lower()
            assert "gamma" in str(user_warnings[0].message)  # Best fit is gamma (lowest ks_statistic)
            assert best[0].distribution == "gamma"

    def test_warn_if_poor_no_warning_for_good_fit(self, results_with_good_fit):
        """Test that no warning is emitted for good fit."""
        results = FitResults(results_with_good_fit)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.best(n=1, warn_if_poor=True)

            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_warn_if_poor_disabled_by_default(self, results_with_poor_fit):
        """Test that warning is not emitted when warn_if_poor=False (default)."""
        results = FitResults(results_with_poor_fit)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.best(n=1)  # warn_if_poor=False by default

            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_warn_if_poor_custom_threshold(self, results_with_good_fit):
        """Test custom pvalue_threshold for warning."""
        results = FitResults(results_with_good_fit)

        # With high threshold, even good fits should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.best(n=1, warn_if_poor=True, pvalue_threshold=0.95)

            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 1  # p-value 0.85 < 0.95

    def test_warn_if_poor_with_none_pvalue(self, spark_session):
        """Test that no warning is emitted when pvalue is None."""
        from pyspark.sql.types import ArrayType, DoubleType, FloatType, StringType, StructField, StructType

        schema = StructType([
            StructField("distribution", StringType(), False),
            StructField("parameters", ArrayType(DoubleType()), False),
            StructField("sse", DoubleType(), False),
            StructField("aic", DoubleType(), True),
            StructField("bic", DoubleType(), True),
            StructField("ks_statistic", DoubleType(), True),
            StructField("pvalue", DoubleType(), True),
            StructField("ad_statistic", DoubleType(), True),
            StructField("ad_pvalue", DoubleType(), True),
        ])
        data = [
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.025, None, 0.35, None),
        ]
        df = spark_session.createDataFrame(data, schema)
        results = FitResults(df)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results.best(n=1, warn_if_poor=True)

            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0  # No warning when pvalue is None


class TestQualityReport:
    """Tests for quality_report() method."""

    @pytest.fixture
    def sample_results_df(self, spark_session):
        """Create sample results for quality report testing."""
        data = [
            # distribution, parameters, sse, aic, bic, ks_statistic, pvalue, ad_statistic, ad_pvalue
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.025, 0.90, 0.35, 0.15),
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0, 0.020, 0.95, 0.40, None),
            ("expon", [0.0, 5.0], 0.008, 1600.0, 1615.0, 0.035, 0.75, 0.30, 0.20),
            ("lognorm", [1.0, 0.0, 2.0], 0.010, 1650.0, 1680.0, 0.040, 0.65, 0.60, None),
            ("weibull_min", [1.5, 0.0, 3.0], 0.004, 1450.0, 1480.0, 0.022, 0.92, 0.45, None),
        ]
        return spark_session.createDataFrame(
            data, ["distribution", "parameters", "sse", "aic", "bic", "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue"]
        )

    def test_quality_report_returns_dict(self, sample_results_df):
        """Test that quality_report returns proper dict structure."""
        results = FitResults(sample_results_df)
        report = results.quality_report()

        assert isinstance(report, dict)
        assert "top_fits" in report
        assert "summary" in report
        assert "warnings" in report
        assert "n_acceptable" in report

    def test_quality_report_top_fits(self, sample_results_df):
        """Test that top_fits contains correct results."""
        results = FitResults(sample_results_df)
        report = results.quality_report(n=3)

        assert len(report["top_fits"]) == 3
        assert all(isinstance(r, DistributionFitResult) for r in report["top_fits"])
        # Should be sorted by ks_statistic
        assert report["top_fits"][0].ks_statistic <= report["top_fits"][1].ks_statistic

    def test_quality_report_summary_stats(self, sample_results_df):
        """Test that summary contains expected statistics."""
        results = FitResults(sample_results_df)
        report = results.quality_report()

        summary = report["summary"]
        assert "min_ks" in summary
        assert "max_ks" in summary
        assert "mean_ks" in summary
        assert "min_pvalue" in summary
        assert "max_pvalue" in summary
        assert "total_distributions" in summary

        assert summary["total_distributions"] == 5
        assert summary["min_ks"] == 0.020  # gamma

    def test_quality_report_n_acceptable(self, sample_results_df):
        """Test that n_acceptable counts fits meeting thresholds."""
        results = FitResults(sample_results_df)

        # With default thresholds (pvalue>=0.05, ks<=0.10, ad<=2.0)
        report = results.quality_report()
        # All fits should pass default thresholds
        assert report["n_acceptable"] == 5

    def test_quality_report_strict_thresholds(self, sample_results_df):
        """Test quality_report with strict thresholds."""
        results = FitResults(sample_results_df)

        # With strict thresholds
        report = results.quality_report(
            pvalue_threshold=0.90,
            ks_threshold=0.025,
            ad_threshold=0.40
        )

        # Only gamma (ks=0.020, pvalue=0.95, ad=0.40) and weibull_min (ks=0.022, pvalue=0.92, ad=0.45)
        # might pass, but weibull_min has ad=0.45 > 0.40
        # Actually: gamma ks=0.020 < 0.025, pvalue=0.95 >= 0.90, ad=0.40 <= 0.40 - passes
        #          weibull_min ks=0.022 < 0.025, pvalue=0.92 >= 0.90, ad=0.45 > 0.40 - fails ad
        #          norm ks=0.025 <= 0.025, pvalue=0.90 >= 0.90, ad=0.35 <= 0.40 - passes
        # So 2 should pass
        assert report["n_acceptable"] == 2

    def test_quality_report_warnings_for_poor_fit(self, spark_session):
        """Test that warnings are generated for poor fits."""
        from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

        schema = StructType([
            StructField("distribution", StringType(), False),
            StructField("parameters", ArrayType(DoubleType()), False),
            StructField("sse", DoubleType(), False),
            StructField("aic", DoubleType(), True),
            StructField("bic", DoubleType(), True),
            StructField("ks_statistic", DoubleType(), True),
            StructField("pvalue", DoubleType(), True),
            StructField("ad_statistic", DoubleType(), True),
            StructField("ad_pvalue", DoubleType(), True),
        ])
        data = [
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.15, 0.02, 2.5, None),  # All bad
        ]
        df = spark_session.createDataFrame(data, schema)
        results = FitResults(df)

        report = results.quality_report()

        # Should have warnings for low p-value, high KS, high AD
        assert len(report["warnings"]) >= 1
        warning_text = " ".join(report["warnings"]).lower()
        assert "p-value" in warning_text or "k-s" in warning_text or "a-d" in warning_text

    def test_quality_report_no_acceptable_warning(self, spark_session):
        """Test warning when no distributions meet thresholds."""
        from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

        schema = StructType([
            StructField("distribution", StringType(), False),
            StructField("parameters", ArrayType(DoubleType()), False),
            StructField("sse", DoubleType(), False),
            StructField("aic", DoubleType(), True),
            StructField("bic", DoubleType(), True),
            StructField("ks_statistic", DoubleType(), True),
            StructField("pvalue", DoubleType(), True),
            StructField("ad_statistic", DoubleType(), True),
            StructField("ad_pvalue", DoubleType(), True),
        ])
        data = [
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.15, 0.02, 2.5, None),  # Poor fit
        ]
        df = spark_session.createDataFrame(data, schema)
        results = FitResults(df)

        report = results.quality_report()

        assert report["n_acceptable"] == 0
        assert any("no distributions" in w.lower() for w in report["warnings"])

    def test_quality_report_few_acceptable_warning(self, spark_session):
        """Test warning when only 1-2 distributions meet thresholds."""
        from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

        schema = StructType([
            StructField("distribution", StringType(), False),
            StructField("parameters", ArrayType(DoubleType()), False),
            StructField("sse", DoubleType(), False),
            StructField("aic", DoubleType(), True),
            StructField("bic", DoubleType(), True),
            StructField("ks_statistic", DoubleType(), True),
            StructField("pvalue", DoubleType(), True),
            StructField("ad_statistic", DoubleType(), True),
            StructField("ad_pvalue", DoubleType(), True),
        ])
        data = [
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.05, 0.80, 1.0, None),  # Good
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0, 0.15, 0.02, 2.5, None),  # Poor
            ("expon", [0.0, 5.0], 0.008, 1600.0, 1615.0, 0.15, 0.02, 2.5, None),  # Poor
        ]
        df = spark_session.createDataFrame(data, schema)
        results = FitResults(df)

        report = results.quality_report()

        assert report["n_acceptable"] == 1
        assert any("only 1 distribution" in w.lower() for w in report["warnings"])


class TestFitResultsClassHierarchy:
    """Tests for the FitResults class hierarchy (v2.1.0+)."""

    @pytest.fixture
    def sample_df(self, spark_session):
        """Create sample results DataFrame."""
        data = [
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0, 0.025, 0.90, 0.35, 0.15),
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0, 0.020, 0.95, 0.40, 0.10),
        ]
        return spark_session.createDataFrame(
            data, ["distribution", "parameters", "sse", "aic", "bic", "ks_statistic", "pvalue", "ad_statistic", "ad_pvalue"]
        )

    def test_factory_returns_eager_without_context(self, sample_df):
        """Test that FitResults() returns EagerFitResults without lazy_contexts."""
        from spark_bestfit.results import EagerFitResults

        results = FitResults(sample_df)
        assert isinstance(results, EagerFitResults)
        assert not results.is_lazy

    def test_factory_returns_lazy_with_context(self, sample_df, spark_session):
        """Test that FitResults() returns LazyFitResults with lazy_contexts."""
        from spark_bestfit.results import LazyFitResults, LazyMetricsContext

        # Create a mock lazy context
        context = LazyMetricsContext(
            source_df=sample_df,
            column="test",
            random_seed=42,
            row_count=100,
        )
        results = FitResults(sample_df, lazy_contexts={"test": context})

        assert isinstance(results, LazyFitResults)
        assert results.is_lazy

    def test_eager_is_not_lazy(self, sample_df):
        """Test that EagerFitResults.is_lazy returns False."""
        from spark_bestfit.results import EagerFitResults

        results = EagerFitResults(sample_df)
        assert results.is_lazy is False

    def test_eager_materialize_returns_self(self, sample_df):
        """Test that EagerFitResults.materialize() returns self."""
        from spark_bestfit.results import EagerFitResults

        results = EagerFitResults(sample_df)
        materialized = results.materialize()

        assert materialized is results

    def test_eager_filter_returns_eager(self, sample_df):
        """Test that EagerFitResults.filter() returns EagerFitResults."""
        from spark_bestfit.results import EagerFitResults

        results = EagerFitResults(sample_df)
        filtered = results.filter(sse_threshold=0.01)

        assert isinstance(filtered, EagerFitResults)
        assert not filtered.is_lazy

    def test_both_classes_inherit_base(self, sample_df, spark_session):
        """Test that both classes inherit from BaseFitResults."""
        from spark_bestfit.results import EagerFitResults, LazyFitResults, LazyMetricsContext

        eager = EagerFitResults(sample_df)
        context = LazyMetricsContext(
            source_df=sample_df,
            column="test",
            random_seed=42,
            row_count=100,
        )
        lazy = LazyFitResults(sample_df, lazy_contexts={"test": context})

        assert isinstance(eager, BaseFitResults)
        assert isinstance(lazy, BaseFitResults)

    def test_repr_shows_class_name(self, sample_df):
        """Test that __repr__ shows the correct class name."""
        from spark_bestfit.results import EagerFitResults

        results = EagerFitResults(sample_df)
        repr_str = repr(results)

        assert "EagerFitResults" in repr_str
        assert "2 distributions fitted" in repr_str

    def test_lazy_source_dataframes_property(self, sample_df, spark_session):
        """Test LazyFitResults.source_dataframes property."""
        from spark_bestfit.results import LazyFitResults, LazyMetricsContext

        context = LazyMetricsContext(
            source_df=sample_df,
            column="test",
            random_seed=42,
            row_count=100,
        )
        results = LazyFitResults(sample_df, lazy_contexts={"test": context})

        sources = results.source_dataframes
        assert "test" in sources
        assert sources["test"] is sample_df

    def test_lazy_is_source_available(self, sample_df, spark_session):
        """Test LazyFitResults.is_source_available() method."""
        from spark_bestfit.results import LazyFitResults, LazyMetricsContext

        context = LazyMetricsContext(
            source_df=sample_df,
            column="test",
            random_seed=42,
            row_count=100,
        )
        results = LazyFitResults(sample_df, lazy_contexts={"test": context})

        # Source should be available
        assert results.is_source_available() is True

    def test_type_alias_usage(self, sample_df):
        """Test FitResultsType type alias includes both classes."""
        from spark_bestfit.results import EagerFitResults, FitResultsType, LazyFitResults

        # Just verify the type alias is a Union of both
        import typing
        origin = typing.get_origin(FitResultsType)
        args = typing.get_args(FitResultsType)

        assert origin is typing.Union
        assert EagerFitResults in args
        assert LazyFitResults in args
