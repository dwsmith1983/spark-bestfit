"""Tests for results module."""

import numpy as np
import pytest
import scipy.stats as st

from spark_bestfit.results import DistributionFitResult, FitResults

class TestDistributionFitResult:
    """Tests for DistributionFitResult dataclass."""

    @pytest.fixture
    def normal_result(self):
        """Create a sample result for normal distribution."""
        # norm has NO shape params, only loc and scale
        return DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            aic=1500.0,
            bic=1520.0,
        )

    @pytest.fixture
    def gamma_result(self):
        """Create a sample result for gamma distribution."""
        return DistributionFitResult(
            distribution="gamma",
            parameters=[2.0, 0.0, 2.0],
            sse=0.003,
            aic=1400.0,
            bic=1430.0,
        )

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
        dist = normal_result.get_scipy_dist()

        assert isinstance(dist, st.rv_continuous)
        assert dist.name == "norm"

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
            ("norm", [50.0, 10.0], 0.005, 1500.0, 1520.0),  # loc, scale
            ("gamma", [2.0, 0.0, 2.0], 0.003, 1400.0, 1430.0),  # shape, loc, scale
            ("expon", [0.0, 5.0], 0.008, 1600.0, 1615.0),  # loc, scale
            ("lognorm", [1.0, 0.0, 2.0], 0.010, 1650.0, 1680.0),  # shape, loc, scale
            (
                "weibull_min",
                [1.5, 0.0, 3.0],
                0.004,
                1450.0,
                1480.0,
            ),  # shape, loc, scale
        ]

        return spark_session.createDataFrame(data, ["distribution", "parameters", "sse", "aic", "bic"])

    def test_initialization(self, sample_results_df):
        """Test FitResults initialization."""
        results = FitResults(sample_results_df)

        assert results._df == sample_results_df

    def test_to_pandas(self, sample_results_df):
        """Test converting to pandas DataFrame."""
        results = FitResults(sample_results_df)
        df_pandas = results.to_pandas()

        assert len(df_pandas) == 5
        assert "distribution" in df_pandas.columns
        assert "sse" in df_pandas.columns

    def test_df_property(self, sample_results_df):
        """Test getting Spark DataFrame via df property."""
        results = FitResults(sample_results_df)

        assert results.df == sample_results_df

    @pytest.mark.parametrize("metric", ["sse", "aic", "bic"])
    def test_best_by_metric(self, sample_results_df, metric):
        """Test getting best distributions by various metrics."""
        results = FitResults(sample_results_df)
        best = results.best(n=1, metric=metric)

        assert len(best) == 1
        assert best[0].distribution == "gamma"  # gamma has lowest SSE, AIC, and BIC

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

    def test_best_invalid_metric(self, sample_results_df):
        """Test that invalid metric raises error."""
        results = FitResults(sample_results_df)

        with pytest.raises(ValueError):
            results.best(n=1, metric="invalid_metric")

    @pytest.mark.parametrize("threshold_kwarg,threshold_val,expected_count", [
        ({"sse_threshold": 0.006}, "sse", 3),
        ({"aic_threshold": 1500}, "aic", 2),
    ])
    def test_filter_by_threshold(self, sample_results_df, threshold_kwarg, threshold_val, expected_count):
        """Test filtering by various threshold types."""
        results = FitResults(sample_results_df)
        filtered = results.filter(**threshold_kwarg)

        assert filtered.count() == expected_count
        df_pandas = filtered.to_pandas()
        assert all(df_pandas[threshold_val] < list(threshold_kwarg.values())[0])

    def test_filter_multiple_criteria(self, sample_results_df):
        """Test filtering by multiple criteria."""
        results = FitResults(sample_results_df)
        filtered = results.filter(sse_threshold=0.010, aic_threshold=1600)

        # Should meet both criteria
        df_pandas = filtered.to_pandas()
        assert all(df_pandas["sse"] < 0.010)
        assert all(df_pandas["aic"] < 1600)

    def test_summary(self, sample_results_df):
        """Test getting summary statistics."""
        results = FitResults(sample_results_df)
        summary = results.summary()

        assert "min_sse" in summary.columns
        assert "mean_sse" in summary.columns
        assert "max_sse" in summary.columns
        assert "total_distributions" in summary.columns

        # Check values
        assert summary["min_sse"].iloc[0] == 0.003
        assert summary["max_sse"].iloc[0] == 0.010
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
        assert "0.003" in repr_str  # Best SSE

    def test_empty_results(self, spark_session):
        """Test FitResults with empty DataFrame."""
        from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

        schema = StructType(
            [
                StructField("distribution", StringType(), False),
                StructField("parameters", ArrayType(FloatType()), False),
                StructField("sse", FloatType(), False),
                StructField("aic", FloatType(), True),
                StructField("bic", FloatType(), True),
            ]
        )
        empty_df = spark_session.createDataFrame([], schema)

        results = FitResults(empty_df)

        assert results.count() == 0
        assert "0 distributions" in repr(results)

    def test_best_empty_returns_empty_list(self, spark_session):
        """Test that best() on empty results returns empty list."""
        from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

        schema = StructType(
            [
                StructField("distribution", StringType(), False),
                StructField("parameters", ArrayType(FloatType()), False),
                StructField("sse", FloatType(), False),
                StructField("aic", FloatType(), True),
                StructField("bic", FloatType(), True),
            ]
        )
        empty_df = spark_session.createDataFrame([], schema)

        results = FitResults(empty_df)
        best = results.best(n=5)

        assert best == []

    def test_filter_returns_fitresults_instance(self, sample_results_df):
        """Test that filter returns a FitResults instance."""
        results = FitResults(sample_results_df)
        filtered = results.filter(sse_threshold=0.01)

        assert isinstance(filtered, FitResults)

    def test_filter_no_criteria(self, sample_results_df):
        """Test filter with no criteria returns all results."""
        results = FitResults(sample_results_df)
        filtered = results.filter()

        assert filtered.count() == results.count()

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
        )

        d = result.to_dict()

        assert set(d.keys()) == {"distribution", "parameters", "sse", "aic", "bic"}
        assert d["distribution"] == "gamma"
        assert d["parameters"] == [2.0, 0.0, 5.0]
        assert d["sse"] == 0.003
        assert d["aic"] == 1500.0
        assert d["bic"] == 1520.0

    def test_get_scipy_dist_various_distributions(self):
        """Test get_scipy_dist works for various distributions."""
        distributions = ["norm", "expon", "gamma", "beta", "weibull_min"]

        for dist_name in distributions:
            result = DistributionFitResult(distribution=dist_name, parameters=[1.0, 0.0, 1.0], sse=0.01)

            dist = result.get_scipy_dist()
            assert dist.name == dist_name
