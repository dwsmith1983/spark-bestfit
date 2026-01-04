"""Tests for serialization module."""

import json
import math
import pickle

import numpy as np
import pytest

from spark_bestfit.results import DistributionFitResult
from spark_bestfit.serialization import (
    SCHEMA_VERSION,
    SerializationError,
    deserialize_from_dict,
    detect_format,
    serialize_to_dict,
)


@pytest.fixture
def normal_result():
    """Create a sample result for normal distribution."""
    return DistributionFitResult(
        distribution="norm",
        parameters=[50.0, 10.0],  # loc, scale
        sse=0.005,
        aic=1500.0,
        bic=1520.0,
        ks_statistic=0.025,
        pvalue=0.90,
        ad_statistic=0.35,
        ad_pvalue=0.15,
        data_min=20.5,
        data_max=82.3,
        data_mean=50.1,
        data_stddev=10.2,
        data_count=10000.0,
    )


@pytest.fixture
def gamma_result():
    """Create a sample result for gamma distribution."""
    return DistributionFitResult(
        distribution="gamma",
        parameters=[2.0, 0.0, 5.0],  # a, loc, scale
        sse=0.003,
        column_name="response_time",
        aic=1400.0,
        bic=1430.0,
        ks_statistic=0.020,
        pvalue=0.95,
        ad_statistic=0.40,
        ad_pvalue=None,
        data_min=0.5,
        data_max=45.2,
        data_mean=10.0,
        data_stddev=7.1,
        data_count=5000.0,
    )


class TestJsonRoundTrip:
    """JSON round-trip serialization tests."""

    def test_round_trip_preserves_all_fields(self, normal_result, tmp_path):
        """Test that all fields survive JSON round-trip."""
        path = tmp_path / "model.json"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)

        # Core fields
        assert loaded.distribution == normal_result.distribution
        assert loaded.parameters == normal_result.parameters
        assert loaded.column_name == normal_result.column_name

        # All metrics
        assert loaded.sse == normal_result.sse
        assert loaded.aic == normal_result.aic
        assert loaded.bic == normal_result.bic
        assert loaded.ks_statistic == normal_result.ks_statistic
        assert loaded.pvalue == normal_result.pvalue
        assert loaded.ad_statistic == normal_result.ad_statistic
        assert loaded.ad_pvalue == normal_result.ad_pvalue

        # Data stats
        assert loaded.data_min == normal_result.data_min
        assert loaded.data_max == normal_result.data_max
        assert loaded.data_mean == normal_result.data_mean
        assert loaded.data_stddev == normal_result.data_stddev
        assert loaded.data_count == normal_result.data_count

    def test_json_contains_required_metadata(self, normal_result, tmp_path):
        """Test that JSON output contains version metadata."""
        path = tmp_path / "model.json"
        normal_result.save(path)

        with open(path) as f:
            data = json.load(f)

        assert data["schema_version"] == SCHEMA_VERSION
        assert "spark_bestfit_version" in data
        assert "created_at" in data
        assert "metrics" in data

    @pytest.mark.parametrize(
        "dist_name,params",
        [
            ("norm", [50.0, 10.0]),
            ("gamma", [2.0, 0.0, 5.0]),
            ("beta", [2.0, 5.0, 0.0, 1.0]),
            ("weibull_min", [1.5, 0.0, 3.0]),
            ("expon", [0.0, 5.0]),
            ("poisson", [7.0]),
            ("binom", [10.0, 0.3]),
            ("nbinom", [5.0, 0.4]),
            ("geom", [0.3]),
        ],
    )
    def test_round_trip_various_distributions(self, dist_name, params, tmp_path):
        """Test round-trip for various distribution types."""
        result = DistributionFitResult(
            distribution=dist_name,
            parameters=params,
            sse=0.005,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.distribution == dist_name
        assert loaded.parameters == params

    def test_float_precision_preserved(self, tmp_path):
        """Test that float precision is maintained through round-trip."""
        # Use values that could lose precision
        precise_params = [1.23456789012345, 0.00000001, 999999999.999999]
        result = DistributionFitResult(
            distribution="norm",
            parameters=precise_params,
            sse=1e-15,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        # JSON has ~15-17 digits of precision for floats
        for orig, loaded_val in zip(precise_params, loaded.parameters):
            assert abs(orig - loaded_val) < 1e-10

    def test_path_object_accepted(self, normal_result, tmp_path):
        """Test that Path objects work for save and load."""
        from pathlib import Path

        path = Path(tmp_path) / "model.json"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)
        assert loaded.distribution == normal_result.distribution


class TestPickleRoundTrip:
    """Pickle round-trip serialization tests."""

    def test_pickle_round_trip(self, normal_result, tmp_path):
        """Test pickle format preserves all fields."""
        path = tmp_path / "model.pkl"
        normal_result.save(path, format="pickle")
        loaded = DistributionFitResult.load(path)

        assert loaded.distribution == normal_result.distribution
        assert loaded.parameters == normal_result.parameters
        assert loaded.data_min == normal_result.data_min
        assert loaded.data_count == normal_result.data_count

    def test_pickle_extension_autodetected(self, normal_result, tmp_path):
        """Test .pickle extension is auto-detected."""
        path = tmp_path / "model.pickle"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)
        assert loaded.distribution == normal_result.distribution

    def test_corrupt_pickle_raises_error(self, tmp_path):
        """Test that corrupt pickle file raises SerializationError."""
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a valid pickle file \x80\x04\x95")

        with pytest.raises(SerializationError):
            DistributionFitResult.load(path)


class TestDataStats:
    """Tests for data stats field handling."""

    def test_data_stats_preserved(self, normal_result, tmp_path):
        """Test that data stats survive round-trip."""
        path = tmp_path / "model.json"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.data_count == 10000.0
        assert loaded.data_min == 20.5
        assert loaded.data_max == 82.3
        assert loaded.data_mean == 50.1
        assert loaded.data_stddev == 10.2

    def test_none_data_stats_handled(self, tmp_path):
        """Test that None data stats are preserved."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.01,
            data_min=None,
            data_max=None,
            data_mean=None,
            data_stddev=None,
            data_count=None,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.data_min is None
        assert loaded.data_count is None


class TestNoneAndOptionalFields:
    """Tests for None and optional field handling."""

    def test_all_optional_fields_none(self, tmp_path):
        """Test saving/loading with all optional fields as None."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[50.0, 10.0],
            sse=0.005,
            column_name=None,
            aic=None,
            bic=None,
            ks_statistic=None,
            pvalue=None,
            ad_statistic=None,
            ad_pvalue=None,
            data_min=None,
            data_max=None,
            data_mean=None,
            data_stddev=None,
            data_count=None,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.column_name is None
        assert loaded.aic is None
        assert loaded.bic is None
        assert loaded.ks_statistic is None
        assert loaded.pvalue is None
        assert loaded.ad_statistic is None
        assert loaded.ad_pvalue is None
        assert loaded.data_min is None
        assert loaded.data_count is None

    def test_unicode_column_name(self, tmp_path):
        """Test that unicode column names are preserved."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.01,
            column_name="数据列_données_データ",
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.column_name == "数据列_données_データ"


class TestErrorHandling:
    """Error handling tests."""

    def test_file_not_found(self, tmp_path):
        """Test loading nonexistent file raises FileNotFoundError."""
        path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            DistributionFitResult.load(path)

    def test_invalid_json_raises_error(self, tmp_path):
        """Test malformed JSON raises SerializationError."""
        path = tmp_path / "malformed.json"
        path.write_text("{invalid json content")

        with pytest.raises(SerializationError):
            DistributionFitResult.load(path)

    def test_missing_distribution_field(self, tmp_path):
        """Test missing required field raises SerializationError."""
        path = tmp_path / "invalid.json"
        path.write_text('{"parameters": [1.0, 2.0]}')

        with pytest.raises(SerializationError):
            DistributionFitResult.load(path)

    def test_unknown_distribution_raises_error(self, tmp_path):
        """Test unknown distribution name raises SerializationError."""
        path = tmp_path / "invalid.json"
        data = {
            "schema_version": "1.0",
            "distribution": "nonexistent_distribution_xyz",
            "parameters": [1.0],
            "metrics": {"sse": 0.01},
        }
        path.write_text(json.dumps(data))

        with pytest.raises(SerializationError, match="Unknown distribution"):
            DistributionFitResult.load(path)

    def test_unknown_extension_raises_error(self):
        """Test unknown file extension raises SerializationError."""
        with pytest.raises(SerializationError):
            detect_format("model.txt")

        with pytest.raises(SerializationError):
            detect_format("model.yaml")


class TestFormatDetection:
    """Format detection tests."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("model.json", "json"),
            ("model.JSON", "json"),
            ("model.Json", "json"),
            ("model.pkl", "pickle"),
            ("model.pickle", "pickle"),
            ("model.PKL", "pickle"),
        ],
    )
    def test_format_detection(self, filename, expected):
        """Test format detection from file extension."""
        assert detect_format(filename) == expected

    def test_explicit_format_overrides_extension(self, normal_result, tmp_path):
        """Test that explicit format parameter overrides extension."""
        path = tmp_path / "model.txt"
        normal_result.save(path, format="json")

        with open(path) as f:
            data = json.load(f)
        assert data["distribution"] == "norm"


class TestLoadedFunctionality:
    """Tests that loaded results are fully functional."""

    def test_sample_produces_correct_distribution(self, tmp_path):
        """Test that sample() from loaded model produces statistically correct output."""
        # Use known parameters
        loc, scale = 100.0, 15.0
        result = DistributionFitResult(
            distribution="norm",
            parameters=[loc, scale],
            sse=0.001,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        # Generate many samples and check statistics
        samples = loaded.sample(size=10000, random_state=42)

        # Mean should be close to loc
        assert abs(samples.mean() - loc) < 1.0  # Within 1 unit
        # Std should be close to scale
        assert abs(samples.std() - scale) < 1.0

    def test_pdf_returns_valid_densities(self, normal_result, tmp_path):
        """Test PDF evaluation on loaded result."""
        path = tmp_path / "model.json"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)

        x = np.linspace(20, 80, 100)
        pdf = loaded.pdf(x)

        assert len(pdf) == 100
        assert np.all(pdf >= 0)
        assert np.all(np.isfinite(pdf))
        # PDF should peak near mean (50)
        peak_idx = np.argmax(pdf)
        assert 40 < x[peak_idx] < 60

    def test_cdf_returns_valid_probabilities(self, normal_result, tmp_path):
        """Test CDF evaluation on loaded result."""
        path = tmp_path / "model.json"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)

        x = np.linspace(20, 80, 100)
        cdf = loaded.cdf(x)

        assert len(cdf) == 100
        assert np.all((cdf >= 0) & (cdf <= 1))
        # CDF should be monotonically increasing
        assert np.all(np.diff(cdf) >= 0)
        # CDF at mean should be ~0.5
        cdf_at_mean = loaded.cdf(np.array([50.0]))[0]
        assert 0.4 < cdf_at_mean < 0.6

    def test_ppf_returns_valid_quantiles(self, normal_result, tmp_path):
        """Test PPF (inverse CDF) on loaded result."""
        path = tmp_path / "model.json"
        normal_result.save(path)
        loaded = DistributionFitResult.load(path)

        # Test common quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        ppf_values = [loaded.ppf(q) for q in quantiles]

        # Should be monotonically increasing
        assert all(ppf_values[i] < ppf_values[i + 1] for i in range(len(ppf_values) - 1))
        # Median should be near mean for normal
        assert 45 < ppf_values[2] < 55


class TestSerializationFunctions:
    """Tests for low-level serialization functions."""

    def test_serialize_to_dict_structure(self, normal_result):
        """Test serialize_to_dict produces correct structure."""
        data = serialize_to_dict(normal_result)

        assert data["schema_version"] == SCHEMA_VERSION
        assert "spark_bestfit_version" in data
        assert "created_at" in data
        assert data["distribution"] == "norm"
        assert data["parameters"] == [50.0, 10.0]
        assert "metrics" in data
        assert data["metrics"]["sse"] == 0.005

    def test_deserialize_from_dict_reconstructs(self, normal_result):
        """Test deserialize_from_dict reconstructs correctly."""
        data = serialize_to_dict(normal_result)
        loaded = deserialize_from_dict(data)

        assert loaded.distribution == normal_result.distribution
        assert loaded.parameters == normal_result.parameters
        assert loaded.sse == normal_result.sse

    def test_deserialize_flat_format_backward_compat(self):
        """Test deserialize handles flat format (backward compatibility)."""
        # Older format without nested metrics
        data = {
            "schema_version": "1.0",
            "distribution": "norm",
            "parameters": [0.0, 1.0],
            "sse": 0.01,
            "aic": 100.0,
            "bic": 105.0,
        }
        result = deserialize_from_dict(data)

        assert result.distribution == "norm"
        assert result.sse == 0.01
        assert result.aic == 100.0


class TestCompactJson:
    """Tests for JSON formatting options."""

    def test_compact_json_no_newlines(self, normal_result, tmp_path):
        """Test indent=None produces compact single-line JSON."""
        path = tmp_path / "compact.json"
        normal_result.save(path, indent=None)

        content = path.read_text()
        assert content.count("\n") <= 1

    def test_indented_json_readable(self, normal_result, tmp_path):
        """Test indent=2 produces multi-line readable JSON."""
        path = tmp_path / "readable.json"
        normal_result.save(path, indent=2)

        content = path.read_text()
        assert content.count("\n") > 5


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_parameters_list(self, tmp_path):
        """Test handling of empty parameters list."""
        # Some distributions might have no parameters in edge cases
        result = DistributionFitResult(
            distribution="norm",
            parameters=[],
            sse=0.01,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.parameters == []

    def test_many_parameters(self, tmp_path):
        """Test handling of distributions with many parameters."""
        # Beta has 4 params, but test with more to be safe
        many_params = [float(i) for i in range(10)]
        result = DistributionFitResult(
            distribution="norm",  # Doesn't matter for serialization test
            parameters=many_params,
            sse=0.01,
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.parameters == many_params

    def test_special_float_values_in_metrics(self, tmp_path):
        """Test that special float values are handled appropriately."""
        # NaN and Inf can't be serialized to JSON directly
        # Our serializer should handle this gracefully
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=float("nan"),  # This is a valid fitting failure case
        )
        path = tmp_path / "model.json"

        # Should not raise, but NaN becomes null in JSON
        result.save(path)
        with open(path) as f:
            data = json.load(f)
        # json.load converts NaN to None or raises error depending on implementation
        # Our code should handle this

    def test_column_name_with_spaces(self, tmp_path):
        """Test column names with spaces and special chars."""
        result = DistributionFitResult(
            distribution="norm",
            parameters=[0.0, 1.0],
            sse=0.01,
            column_name="my column with spaces & special!",
        )
        path = tmp_path / "model.json"
        result.save(path)
        loaded = DistributionFitResult.load(path)

        assert loaded.column_name == "my column with spaces & special!"
