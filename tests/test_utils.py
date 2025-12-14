"""Tests for utils module."""

import pytest
from pyspark.sql import SparkSession

from spark_bestfit.utils import get_spark_session


class TestGetSparkSession:
    """Tests for get_spark_session utility function."""

    def test_returns_provided_session(self, spark_session):
        """Test that provided session is returned as-is."""
        result = get_spark_session(spark_session)

        assert result is spark_session

    def test_returns_active_session_when_none_provided(self, spark_session):
        """Test that active session is returned when None is provided."""
        # spark_session fixture ensures an active session exists
        result = get_spark_session(None)

        assert result is not None
        assert isinstance(result, SparkSession)

    def test_same_session_returned_multiple_calls(self, spark_session):
        """Test that same session is returned across multiple calls."""
        result1 = get_spark_session(spark_session)
        result2 = get_spark_session(spark_session)

        assert result1 is result2

    def test_can_use_returned_session(self, spark_session):
        """Test that returned session is functional."""
        session = get_spark_session(spark_session)

        # Should be able to create a DataFrame
        df = session.createDataFrame([(1,), (2,), (3,)], ["value"])

        assert df.count() == 3

    def test_raises_error_when_no_session_available(self):
        """Test that RuntimeError is raised when no session is available."""
        # Stop any existing session
        existing = SparkSession.getActiveSession()
        if existing:
            existing.stop()

        with pytest.raises(RuntimeError, match="No SparkSession provided"):
            get_spark_session(None)

        # Recreate session for other tests (fixture will handle this)
